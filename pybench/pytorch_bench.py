from __future__ import annotations

import time
import statistics
import logging
from contextlib import nullcontext
from logging import StreamHandler, Formatter

import click
from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm
import torch


BASIC_SUITE = ("matmul", "add", "mul", "sum")
EXTENDED_SUITE = ("dot", "transpose", "relu", "rand", "conv2d", "model_forward")


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
    if suite in ("basic", "all"):
        ops.extend(build_basic_ops(size, device, dt))
    if suite in ("extended", "all"):
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
    type=click.Choice(["basic", "extended", "all"]),
    help="Benchmark suite to run.",
)
@click.option(
    "--seed", default=42, show_default=True,
    help="Random seed.",
)
@click.option(
    "--verbose", is_flag=True,
    help="Enable DEBUG logging.",
)
def main(iterations: int, size: int, dtype: str, suite: str, seed: int, verbose: bool):
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
        skip_reason = get_skip_reason(device, dt)
        if skip_reason:
            logger.warning(skip_reason)
            continue

        logger.info(f"\nBenchmarking on device: {device}")
        if suite in ("basic", "all"):
            run_operations(
                build_basic_ops(size, device, dt),
                device,
                iterations,
                logger,
            )
        if suite in ("extended", "all"):
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


if __name__ == "__main__":
    main()
