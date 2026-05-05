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
SUITE_CHOICES = ("basic", "extended", "all", "memory", "full")
DTYPE_CHOICES = ("float", "double", "half")
COMPUTE_SUITES = ("basic", "extended", "all", "full")
MEMORY_SUITES = ("memory", "full")
MAX_MEMORY_CHUNK_BYTES = 256 * 1024**2
REQUIRED_TORCH_ATTRIBUTES = (
    "device",
    "manual_seed",
    "randn",
    "rand",
    "empty",
    "float16",
    "float32",
    "float64",
    "nn",
)


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


def validate_torch_installation():
    missing = [
        f"torch.{name}" for name in REQUIRED_TORCH_ATTRIBUTES if not hasattr(torch, name)
    ]
    if missing:
        raise click.ClickException(
            "PyTorch import is incomplete; activate an environment with a full "
            "PyTorch installation "
            f"(missing required attributes: {', '.join(missing)})."
        )


def is_cuda_available():
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def get_mps_backend():
    backends = getattr(torch, "backends", None)
    return getattr(backends, "mps", None)


def is_mps_available():
    mps_backend = get_mps_backend()
    is_available = getattr(mps_backend, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def get_devices():
    devices = [torch.device("cpu")]
    cuda = getattr(torch, "cuda", None)
    device_count = getattr(cuda, "device_count", None)
    if is_cuda_available() and callable(device_count):
        for i in range(device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    if is_mps_available():
        devices.append(torch.device("mps"))
    return devices


def device_matches_filter(device: torch.device, device_filter: str):
    normalized_filter = device_filter.strip().lower()
    device_type = getattr(device, "type", "").lower()
    device_spec = str(device).lower()
    return normalized_filter == device_type or normalized_filter == device_spec


def filter_devices(devices, device_filter):
    if device_filter is None or not device_filter.strip():
        return list(devices)

    matched_devices = [
        device for device in devices if device_matches_filter(device, device_filter)
    ]
    if not matched_devices:
        available_devices = ", ".join(str(device) for device in devices) or "none"
        raise click.ClickException(
            f"No detected devices match --device {device_filter!r}. "
            f"Available devices: {available_devices}."
        )
    return matched_devices


def log_environment_info(logger):
    logger.info(f"PyTorch version: {getattr(torch, '__version__', 'unknown')}")
    cuda_available = is_cuda_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        log_cuda_devices(logger)
    logger.info(f"MPS available: {is_mps_available()}")


def log_cuda_devices(logger):
    cuda = getattr(torch, "cuda", None)
    device_count = cuda.device_count()
    logger.info(f"CUDA devices: {device_count}")
    for idx in range(device_count):
        props = cuda.get_device_properties(idx)
        total_memory_mb = props.total_memory / (1024**2)
        logger.info(
            f"CUDA device {idx}: {props.name}, memory={total_memory_mb:.1f} MB, "
            f"multiprocessors={props.multi_processor_count}"
        )


def sync(device: torch.device):
    # Ensure operations are finished on GPU/MPS
    if device.type == "cuda":
        cuda = getattr(torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        synchronize = getattr(mps, "synchronize", None)
        if callable(synchronize):
            synchronize()


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


def get_memory_dtype(device: torch.device, dt: torch.dtype, logger):
    if dt == torch.float64 and device.type == "mps":
        logger.warning(
            f"[{device}] Using float32 for memory stress because MPS does not support FP64."
        )
        return torch.float32
    return dt


def bytes_to_mb(size_bytes):
    return size_bytes / (1024**2)


def bytes_to_gib(size_bytes):
    return size_bytes / (1024**3)


def ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def get_dtype_size(dt):
    if dt == torch.float16:
        return 2
    if dt == torch.float64:
        return 8
    return 4


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


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
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return None, None, "torch.cuda unavailable"
        mem_get_info = getattr(cuda, "mem_get_info", None)
        if callable(mem_get_info):
            free_bytes, total_bytes = mem_get_info(device)
            return free_bytes, total_bytes, "torch.cuda.mem_get_info"
        return estimate_cuda_memory_info(device)
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        recommended_max_memory = getattr(mps, "recommended_max_memory", None)
        if not callable(recommended_max_memory):
            return None, None, "torch.mps.recommended_max_memory unavailable"
        total_bytes = recommended_max_memory()
        driver_allocated_memory = getattr(mps, "driver_allocated_memory", None)
        current_allocated_memory = getattr(mps, "current_allocated_memory", None)
        used_bytes = 0
        if callable(driver_allocated_memory):
            used_bytes = driver_allocated_memory()
        elif callable(current_allocated_memory):
            used_bytes = current_allocated_memory()
        return max(total_bytes - used_bytes, 0), total_bytes, "torch.mps"
    return None, None, "unsupported"


def estimate_cuda_memory_info(device):
    cuda = getattr(torch, "cuda", None)
    get_device_properties = getattr(cuda, "get_device_properties", None)
    if not callable(get_device_properties):
        return None, None, "torch.cuda.mem_get_info unavailable"

    props = get_device_properties(device)
    total_bytes = getattr(props, "total_memory", None)
    if total_bytes is None:
        return None, None, "torch.cuda.total_memory unavailable"

    memory_reserved = getattr(cuda, "memory_reserved", None)
    memory_allocated = getattr(cuda, "memory_allocated", None)
    if callable(memory_reserved):
        used_bytes = memory_reserved(device)
    elif callable(memory_allocated):
        used_bytes = memory_allocated(device)
    else:
        used_bytes = 0
    return max(total_bytes - used_bytes, 0), total_bytes, "torch.cuda memory estimate"


def calculate_memory_target_bytes(device, memory_percent, memory_mb):
    available_bytes, total_bytes, source = get_device_memory_info(device)
    if memory_mb is not None:
        return memory_mb * 1024**2, available_bytes, total_bytes, source
    if available_bytes is None:
        raise RuntimeError(f"Could not determine available memory for {device}.")
    if available_bytes <= 0:
        raise RuntimeError(
            f"No available memory detected for {device} (source={source})."
        )
    target_bytes = int(available_bytes * (memory_percent / 100.0))
    if target_bytes <= 0:
        raise RuntimeError(
            f"No available memory target for {device} (source={source})."
        )
    return target_bytes, available_bytes, total_bytes, source


def reset_peak_memory_stats(device):
    if device.type != "cuda":
        return
    cuda = getattr(torch, "cuda", None)
    reset_peak = getattr(cuda, "reset_peak_memory_stats", None)
    if callable(reset_peak):
        reset_peak(device)


def get_memory_stats(device):
    stats = {}
    if device.type == "cuda":
        cuda = getattr(torch, "cuda", None)
        for name in (
            "memory_allocated",
            "max_memory_allocated",
            "memory_reserved",
            "max_memory_reserved",
        ):
            fn = getattr(cuda, name, None)
            if callable(fn):
                stats[name] = fn(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        for name in ("current_allocated_memory", "driver_allocated_memory"):
            fn = getattr(mps, name, None)
            if callable(fn):
                stats[name] = fn()
    return stats


def clear_device_cache(device):
    if device.type == "cuda":
        backend = getattr(torch, "cuda", None)
    elif device.type == "mps":
        backend = getattr(torch, "mps", None)
    else:
        backend = None
    empty_cache = getattr(backend, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    gc.collect()


def allocate_memory_chunks(target_bytes, device, dt):
    if target_bytes <= 0:
        raise RuntimeError("Memory target must be greater than zero.")
    element_size = get_dtype_size(dt)
    target_elements = max(1, ceil_div(target_bytes, element_size))
    max_chunk_elements = max(1, MAX_MEMORY_CHUNK_BYTES // element_size)
    chunks = []
    allocated_elements = 0

    try:
        while allocated_elements < target_elements:
            chunk_elements = min(
                max_chunk_elements,
                target_elements - allocated_elements,
            )
            chunks.append(torch.empty((chunk_elements,), device=device, dtype=dt))
            allocated_elements += chunk_elements
    except Exception:
        chunks.clear()
        clear_device_cache(device)
        raise

    return chunks, sum(tensor_nbytes(chunk) for chunk in chunks)


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
            total_text = (
                "unknown" if total_bytes is None else f"{bytes_to_mb(total_bytes):.1f} MB"
            )
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
    type=click.Choice(DTYPE_CHOICES),
    help="Data type for tensors.",
)
@click.option(
    "--suite",
    default="basic",
    show_default=True,
    type=click.Choice(SUITE_CHOICES),
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
    "--device", default=None, show_default=True,
    help="Device filter to run, for example cpu, cuda, cuda:0, or mps.",
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
    device: str | None,
    verbose: bool,
):
    """
    Benchmark common PyTorch operations across available devices.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger(level)

    validate_torch_installation()
    log_environment_info(logger)

    torch.manual_seed(seed)
    devices = filter_devices(get_devices(), device)
    logger.info(f"Detected devices: {devices}")

    # Map dtype strings to torch dtypes
    dtype_map = {"float": torch.float32, "double": torch.float64, "half": torch.float16}
    dt = dtype_map[dtype]

    for bench_device in devices:
        logger.info(f"\nBenchmarking on device: {bench_device}")

        if suite in COMPUTE_SUITES:
            skip_reason = get_skip_reason(bench_device, dt)
            if skip_reason:
                logger.warning(skip_reason)
            else:
                if suite in ("basic", "all", "full"):
                    run_operations(
                        build_basic_ops(size, bench_device, dt),
                        bench_device,
                        iterations,
                        logger,
                    )
                if suite in ("extended", "all", "full"):
                    try:
                        extended_ops = build_extended_ops(size, bench_device, dt)
                    except Exception as exc:
                        logger.warning(
                            f"[{bench_device}] Skipping extended suite: {exc}"
                        )
                    else:
                        run_operations(
                            extended_ops,
                            bench_device,
                            iterations,
                            logger,
                            skip_failed_ops=True,
                        )

        if suite in MEMORY_SUITES:
            memory_dt = get_memory_dtype(bench_device, dt, logger)
            run_memory_stress(
                bench_device,
                memory_dt,
                iterations,
                memory_percent,
                memory_mb,
                logger,
            )


if __name__ == "__main__":
    main()
