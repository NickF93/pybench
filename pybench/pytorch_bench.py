from __future__ import annotations

import time
import statistics
import logging
import gc
import os
from contextlib import nullcontext
from logging import StreamHandler, Formatter

import click
from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm
import torch


BASIC_SUITE = ("matmul", "add", "mul", "sum")
EXTENDED_SUITE = ("dot", "transpose", "relu", "rand", "conv2d", "model_forward")
COMPUTE_SUITES = ("basic", "extended", "all", "full")
MEMORY_SUITES = ("memory", "full")
MAX_MEMORY_CHUNK_BYTES = 256 * 1024**2


class ColoredFormatter(Formatter):
    """Logging Formatter to add colors."""
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(message)s" + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + "%(message)s" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "%(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "%(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + "%(message)s" + Style.RESET_ALL,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(level=logging.INFO):
    colorama_init(autoreset=True)
    logger = logging.getLogger("bench")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = StreamHandler()
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
    return logger


def get_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def log_environment_info(logger):
    logger.info(f"PyTorch version: {getattr(torch, '__version__', 'unknown')}")
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        log_cuda_devices(logger)
    if hasattr(torch.backends, "mps"):
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")


def log_cuda_devices(logger):
    device_count = torch.cuda.device_count()
    logger.info(f"CUDA devices: {device_count}")
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        total_memory_mb = props.total_memory / (1024**2)
        logger.info(
            f"CUDA device {idx}: {props.name}, memory={total_memory_mb:.1f} MB, "
            f"multiprocessors={props.multi_processor_count}"
        )


def sync(device: torch.device):
    # Ensure operations are finished on GPU/MPS
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def inference_context():
    inference_mode = getattr(torch, "inference_mode", None)
    if inference_mode is None:
        return nullcontext()
    return inference_mode()


def get_skip_reason(device: torch.device, dt: torch.dtype):
    if dt == torch.float16 and device.type == "cpu":
        return f"Skipping FP16 benchmark on CPU ({device}): unsupported efficiently."
    if dt == torch.float64 and device.type == "mps":
        return f"Skipping FP64 benchmark on MPS ({device}): unsupported."
    return None


def bytes_to_mb(size_bytes):
    return size_bytes / (1024**2)


def bytes_to_gib(size_bytes):
    return size_bytes / (1024**3)


def get_dtype_size(dt):
    if dt == torch.float16:
        return 2
    if dt == torch.float64:
        return 8
    return 4


def benchmark_op(name, fn, device, iterations, logger):
    with inference_context():
        # Warm-up
        for _ in range(5):
            fn()
        sync(device)

        times = []
        for _ in tqdm(range(iterations), desc=f"{name} on {device}", leave=False):
            start = time.perf_counter()
            fn()
            sync(device)
            end = time.perf_counter()
            times.append(end - start)

    total_time = sum(times)
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    logger.info(
        f"[{device}] {name}: total={total_time:.6f}s, avg={avg_time:.6f}s, "
        f"min={min_time:.6f}s, max={max_time:.6f}s, std={std_time:.6f}s"
    )


def build_basic_ops(size, device, dt):
    a = torch.randn((size, size), device=device, dtype=dt)
    b = torch.randn((size, size), device=device, dtype=dt)

    return [
        ("matmul", lambda: a @ b),
        ("add", lambda: a + b),
        ("mul", lambda: a * b),
        ("sum", lambda: a.sum()),
    ]


def build_extended_ops(size, device, dt):
    image_size = min(size, 224)
    batch_size = 8

    a = torch.randn((size, size), device=device, dtype=dt)
    vec = torch.randn((size,), device=device, dtype=dt)
    image_batch = torch.randn(
        (batch_size, 3, image_size, image_size),
        device=device,
        dtype=dt,
    )
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device=device, dtype=dt)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_size * image_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    ).to(device=device, dtype=dt)

    return [
        ("dot", lambda: torch.dot(vec, vec)),
        ("transpose", lambda: a.t()),
        ("relu", lambda: torch.relu(a)),
        ("rand", lambda: torch.rand((size, size), device=device, dtype=dt)),
        ("conv2d", lambda: conv(image_batch)),
        ("model_forward", lambda: model(image_batch)),
    ]


def get_operations(suite, size, device, dt):
    ops = []
    if suite in ("basic", "all", "full"):
        ops.extend(build_basic_ops(size, device, dt))
    if suite in ("extended", "all", "full"):
        ops.extend(build_extended_ops(size, device, dt))
    return ops


def run_operations(ops, device, iterations, logger, skip_failed_ops=False):
    for name, fn in ops:
        try:
            benchmark_op(name, fn, device, iterations, logger)
        except Exception as exc:
            if not skip_failed_ops:
                raise
            logger.warning(f"[{device}] Skipping {name}: {exc}")


def get_cpu_memory_info():
    mem_available = None
    mem_total = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                key, value = line.split(":", 1)
                if key in ("MemAvailable", "MemTotal"):
                    kib = int(value.strip().split()[0])
                    if key == "MemAvailable":
                        mem_available = kib * 1024
                    else:
                        mem_total = kib * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass

    if mem_available is not None:
        return mem_available, mem_total, "/proc/meminfo"

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None, None, "unavailable"

    available = page_size * available_pages
    total = page_size * total_pages
    return available, total, "os.sysconf"


def get_device_memory_info(device):
    if device.type == "cpu":
        return get_cpu_memory_info()
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return free_bytes, total_bytes, "torch.cuda.mem_get_info"
    if device.type == "mps":
        recommended_max_memory = getattr(torch.mps, "recommended_max_memory", None)
        if recommended_max_memory is None:
            return None, None, "torch.mps.recommended_max_memory unavailable"
        total_bytes = recommended_max_memory()
        driver_allocated_memory = getattr(torch.mps, "driver_allocated_memory", None)
        current_allocated_memory = getattr(torch.mps, "current_allocated_memory", None)
        used_bytes = 0
        if driver_allocated_memory is not None:
            used_bytes = driver_allocated_memory()
        elif current_allocated_memory is not None:
            used_bytes = current_allocated_memory()
        return max(total_bytes - used_bytes, 0), total_bytes, "torch.mps"
    return None, None, "unsupported"


def calculate_memory_target_bytes(device, memory_percent, memory_mb):
    available_bytes, total_bytes, source = get_device_memory_info(device)
    if memory_mb is not None:
        return memory_mb * 1024**2, available_bytes, total_bytes, source
    if available_bytes is None:
        raise RuntimeError(f"Could not determine available memory for {device}.")
    return int(available_bytes * (memory_percent / 100.0)), available_bytes, total_bytes, source


def reset_peak_memory_stats(device):
    if device.type != "cuda":
        return
    reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
    if reset_peak is not None:
        reset_peak(device)


def get_memory_stats(device):
    stats = {}
    if device.type == "cuda":
        for name in (
            "memory_allocated",
            "max_memory_allocated",
            "memory_reserved",
            "max_memory_reserved",
        ):
            fn = getattr(torch.cuda, name, None)
            if fn is not None:
                stats[name] = fn(device)
    elif device.type == "mps":
        for name in ("current_allocated_memory", "driver_allocated_memory"):
            fn = getattr(torch.mps, name, None)
            if fn is not None:
                stats[name] = fn()
    return stats


def clear_device_cache(device):
    if device.type == "cuda":
        empty_cache = getattr(torch.cuda, "empty_cache", None)
    elif device.type == "mps":
        empty_cache = getattr(torch.mps, "empty_cache", None)
    else:
        empty_cache = None
    if empty_cache is not None:
        empty_cache()
    gc.collect()


def allocate_memory_chunks(target_bytes, device, dt):
    element_size = get_dtype_size(dt)
    target_elements = max(1, target_bytes // element_size)
    max_chunk_elements = max(1, MAX_MEMORY_CHUNK_BYTES // element_size)
    chunks = []
    allocated_elements = 0

    while allocated_elements < target_elements:
        chunk_elements = min(max_chunk_elements, target_elements - allocated_elements)
        chunks.append(torch.empty((chunk_elements,), device=device, dtype=dt))
        allocated_elements += chunk_elements

    return chunks, allocated_elements * element_size


def touch_memory_chunks(chunks, allocated_bytes, iterations, device):
    times = []
    for i in tqdm(range(iterations), desc=f"memory on {device}", leave=False):
        start = time.perf_counter()
        for chunk in chunks:
            chunk.fill_(float((i % 7) + 1))
            chunk.sum()
        sync(device)
        times.append(time.perf_counter() - start)

    total_time = sum(times)
    touched_bytes = allocated_bytes * 2 * iterations
    bandwidth_gib_s = bytes_to_gib(touched_bytes) / total_time if total_time else 0.0
    return times, bandwidth_gib_s


def release_memory_chunks(chunks, device):
    chunks.clear()
    clear_device_cache(device)


def format_memory_stats(stats):
    return ", ".join(f"{name}={bytes_to_mb(value):.1f} MB" for name, value in stats.items())


def run_memory_stress(device, dt, iterations, memory_percent, memory_mb, logger):
    chunks = []
    try:
        target_bytes, available_bytes, total_bytes, source = calculate_memory_target_bytes(
            device,
            memory_percent,
            memory_mb,
        )
        logger.info(
            f"[{device}] memory target={bytes_to_mb(target_bytes):.1f} MB "
            f"(source={source})"
        )
        if available_bytes is not None:
            total_text = "unknown" if total_bytes is None else f"{bytes_to_mb(total_bytes):.1f} MB"
            logger.info(
                f"[{device}] memory available={bytes_to_mb(available_bytes):.1f} MB, "
                f"total={total_text}"
            )

        reset_peak_memory_stats(device)
        chunks, allocated_bytes = allocate_memory_chunks(target_bytes, device, dt)
        times, bandwidth_gib_s = touch_memory_chunks(
            chunks,
            allocated_bytes,
            iterations,
            device,
        )
        logger.info(
            f"[{device}] memory: allocated={bytes_to_mb(allocated_bytes):.1f} MB, "
            f"avg_touch={statistics.mean(times):.6f}s, bandwidth={bandwidth_gib_s:.3f} GiB/s"
        )
        stats = get_memory_stats(device)
        if stats:
            logger.info(f"[{device}] memory stats: {format_memory_stats(stats)}")
    except (RuntimeError, MemoryError, TypeError) as exc:
        logger.warning(f"[{device}] Memory stress failed: {exc}")
    finally:
        release_memory_chunks(chunks, device)


@click.command()
@click.option(
    "--iterations", default=100, show_default=True,
    type=click.IntRange(min=1),
    help="Number of iterations per operation.",
)
@click.option(
    "--size", default=1024, show_default=True,
    type=click.IntRange(min=1),
    help="Matrix size for square operations.",
)
@click.option(
    "--dtype", default="float", show_default=True,
    type=click.Choice(["float", "double", "half"]),
    help="Data type for tensors.",
)
@click.option(
    "--suite",
    default="basic",
    show_default=True,
    type=click.Choice(["basic", "extended", "all", "memory", "full"]),
    help="Benchmark suite to run.",
)
@click.option(
    "--memory-percent",
    default=70.0,
    show_default=True,
    type=click.FloatRange(min=1.0, max=95.0),
    help="Percent of detected available memory to stress.",
)
@click.option(
    "--memory-mb",
    default=None,
    type=click.IntRange(min=1),
    help="Exact memory stress target in MB; overrides --memory-percent.",
)
@click.option(
    "--seed", default=42, show_default=True,
    help="Random seed.",
)
@click.option(
    "--verbose", is_flag=True,
    help="Enable DEBUG logging.",
)
def main(
    iterations: int,
    size: int,
    dtype: str,
    suite: str,
    memory_percent: float,
    memory_mb: int | None,
    seed: int,
    verbose: bool,
):
    """
    Benchmark common PyTorch operations across available devices.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger(level)

    log_environment_info(logger)

    torch.manual_seed(seed)
    devices = get_devices()
    logger.info(f"Detected devices: {devices}")

    # Map dtype strings to torch dtypes
    dtype_map = {"float": torch.float32, "double": torch.float64, "half": torch.float16}
    dt = dtype_map[dtype]

    for device in devices:
        logger.info(f"\nBenchmarking on device: {device}")

        if suite in COMPUTE_SUITES:
            skip_reason = get_skip_reason(device, dt)
            if skip_reason:
                logger.warning(skip_reason)
            else:
                if suite in ("basic", "all", "full"):
                    run_operations(
                        build_basic_ops(size, device, dt),
                        device,
                        iterations,
                        logger,
                    )
                if suite in ("extended", "all", "full"):
                    try:
                        extended_ops = build_extended_ops(size, device, dt)
                    except Exception as exc:
                        logger.warning(f"[{device}] Skipping extended suite: {exc}")
                    else:
                        run_operations(
                            extended_ops,
                            device,
                            iterations,
                            logger,
                            skip_failed_ops=True,
                        )

        if suite in MEMORY_SUITES:
            run_memory_stress(
                device,
                dt,
                iterations,
                memory_percent,
                memory_mb,
                logger,
            )


if __name__ == "__main__":
    main()
