import time
import statistics
import logging
from logging import StreamHandler, Formatter

import click
from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm
import torch


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
    handler = StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger = logging.getLogger("bench")
    logger.setLevel(level)
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


def sync(device: torch.device):
    # Ensure operations are finished on GPU/MPS
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.backends.mps.synchronize()


def benchmark_op(name, fn, device, iterations, logger):
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


@click.command()
@click.option(
    "--iterations", default=100, show_default=True,
    help="Number of iterations per operation.",
)
@click.option(
    "--size", default=1024, show_default=True,
    help="Matrix size for square operations.",
)
@click.option(
    "--dtype", default="float", show_default=True,
    type=click.Choice(["float", "double", "half"]),
    help="Data type for tensors.",
)
@click.option(
    "--seed", default=42, show_default=True,
    help="Random seed.",
)
@click.option(
    "--verbose", is_flag=True,
    help="Enable DEBUG logging.",
)
def main(iterations: int, size: int, dtype: str, seed: int, verbose: bool):
    """
    Benchmark common PyTorch operations across available devices.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger(level)

    # Log environment info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    if hasattr(torch.backends, "mps"):
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")

    torch.manual_seed(seed)
    devices = get_devices()
    logger.info(f"Detected devices: {devices}")

    # Map dtype strings to torch dtypes
    dtype_map = {"float": torch.float32, "double": torch.float64, "half": torch.float16}
    dt = dtype_map[dtype]

    for device in devices:
        # Skip CPU benchmarking for float16 (FP16) for efficiency
        if dt == torch.float16 and device.type == "cpu":
            logger.warning(f"Skipping FP16 benchmark on CPU ({device}): unsupported efficiently.")
            continue

        logger.info(f"\nBenchmarking on device: {device}")

        # Generate fresh tensors on this device
        a = torch.randn((size, size), device=device, dtype=dt)
        b = torch.randn((size, size), device=device, dtype=dt)

        # Define operations
        ops = [
            ("matmul", lambda: a @ b),
            ("add", lambda: a + b),
            ("mul", lambda: a * b),
            ("sum", lambda: a.sum()),
        ]

        for name, fn in ops:
            benchmark_op(name, fn, device, iterations, logger)


if __name__ == "__main__":
    main()
