import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize Colorama\init(autoreset=True)

# Configure root logger with a colored handler
class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, '')
        record.levelname = color + record.levelname + Style.RESET_ALL
        return super().format(record)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(ColorFormatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def list_gpus():
    """Detect and log available CUDA GPUs."""
    count = torch.cuda.device_count()
    if count == 0:
        logger.error("No CUDA-compatible GPUs detected.")
        return []

    gpus = []
    logger.info(f"Detected {count} CUDA device(s)")
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        info = {
            'index': idx,
            'name': props.name,
            'total_memory_MB': props.total_memory / (1024 ** 2),
            'multi_processor_count': props.multi_processor_count
        }
        logger.info(
            f"GPU {idx}: {props.name}, "
            f"Memory: {info['total_memory_MB']:.1f} MB, "
            f"MPs: {info['multi_processor_count']}"
        )
        gpus.append(info)
    return gpus


def benchmark_operations(idx, iterations=10):
    """Run a suite of tensor operations on the GPU and collect timing stats."""
    device = torch.device(f'cuda:{idx}')
    ops = {}

    # Prepare sample tensors/models
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    vec = torch.randn(size, device=device)
    mat = torch.randn(64, 3, 224, 224, device=device)
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*224*224, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(device)

    # Define operations
    operations = [
        ("matmul", lambda: torch.mm(a, b)),
        ("add", lambda: a + b),
        ("mul", lambda: a * b),
        ("dot", lambda: torch.dot(vec, vec)),
        ("sum", lambda: torch.sum(a)),
        ("transpose", lambda: a.t()),
        ("conv2d", lambda: conv(mat)),
        ("activation", lambda: F.relu(a)),
        ("model_forward", lambda: model(mat)),
        ("rand", lambda: torch.rand(1000, 1000, device=device)),
    ]

    # Warm-up all ops
    for name, op in operations:
        for _ in range(2):
            _ = op()
        torch.cuda.synchronize(device)

    # Benchmark each op
    for name, op in operations:
        times = []
        logger.info(Fore.CYAN + f"Benchmarking {name} on GPU {idx}" + Style.RESET_ALL)
        for i in tqdm(range(iterations), desc=f"{name}"):
            start = time.time()
            _ = op()
            torch.cuda.synchronize(device)
            times.append(time.time() - start)
        arr = np.array(times)
        ops[name] = {
            'min': float(arr.min()),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std())
        }
        logger.info(Fore.GREEN + f"{name} stats: {ops[name]}" + Style.RESET_ALL)

    return ops


def main():
    gpus = list_gpus()
    if not gpus:
        return

    overall = {}
    for gpu in gpus:
        idx = gpu['index']
        logger.info(Fore.MAGENTA + f"\nStarting full benchmark on GPU {idx}: {gpu['name']}" + Style.RESET_ALL)
        stats = benchmark_operations(idx, iterations=10000)
        overall[idx] = stats

    # Summary
    logger.info(Fore.MAGENTA + "\n=== Benchmark Summary ===" + Style.RESET_ALL)
    for idx, stats in overall.items():
        logger.info(Fore.MAGENTA + f"GPU {idx}:" + Style.RESET_ALL)
        for name, s in stats.items():
            logger.info(
                f"  {name:15} min: {s['min']:.6f}s, mean: {s['mean']:.6f}s, max: {s['max']:.6f}s"
            )

if __name__ == '__main__':
    main()
