import logging
import time
import numpy as np
import torch
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True)

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


def benchmark_gpu(idx, matrix_size=2048, iterations=10):
    """Run a simple matrix multiplication benchmark on a given GPU."""
    device = torch.device(f'cuda:{idx}')
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    times = []

    # Warm-up
    for _ in range(2):
        _ = torch.mm(a, b)
    torch.cuda.synchronize(device)

    # Benchmark
    for i in tqdm(range(iterations), desc=f"GPU {idx} Benchmark"):
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize(device)
        elapsed = time.time() - start
        times.append(elapsed)
        logger.debug(f"Iter {i}: {elapsed:.6f} s on GPU {idx}")

    times_np = np.array(times)
    stats = {
        'min': float(times_np.min()),
        'max': float(times_np.max()),
        'mean': float(times_np.mean()),
        'median': float(np.median(times_np)),
        'std': float(times_np.std())
    }
    return stats


def main():
    gpus = list_gpus()
    if not gpus:
        return

    all_stats = {}
    for gpu in gpus:
        idx = gpu['index']
        logger.info(Fore.CYAN + f"Starting benchmark on GPU {idx}: {gpu['name']}" + Style.RESET_ALL)
        stats = benchmark_gpu(idx)
        all_stats[idx] = stats
        logger.info(Fore.GREEN + f"Stats for GPU {idx}: {stats}" + Style.RESET_ALL)

    # Summary
    logger.info(Fore.MAGENTA + "\nBenchmark Summary:" + Style.RESET_ALL)
    for idx, stats in all_stats.items():
        logger.info(
            f"GPU {idx} - min: {stats['min']:.6f}s, max: {stats['max']:.6f}s, "
            f"mean: {stats['mean']:.6f}s, median: {stats['median']:.6f}s, std: {stats['std']:.6f}s"
        )

if __name__ == '__main__':
    main()