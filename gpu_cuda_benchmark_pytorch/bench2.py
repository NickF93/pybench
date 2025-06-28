import logging
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

# Logger configuration: log only messages with WARNING level or higher.
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def benchmark(device: str, iterations: int = 1000, matrix_size: int = 2048) -> float:
    """
    Run a heavy matrix multiplication benchmark on a specified GPU device.

    The benchmark includes:
      - A warm-up phase with 10 multiplications to stabilize performance.
      - A measured phase with a specified number of iterations using CUDA events.
    
    Args:
        device (str): The CUDA device on which to run the benchmark (e.g., "cuda:0").
        iterations (int): Number of matrix multiplications to perform.
        matrix_size (int): Size of the square matrices to be multiplied.
        
    Returns:
        float: The elapsed time in milliseconds for the benchmark.
    """
    # Set the current CUDA device
    torch.cuda.set_device(device)

    # Create two large random matrices on the specified device
    x: Tensor = torch.randn(matrix_size, matrix_size, device=device)
    y: Tensor = torch.randn(matrix_size, matrix_size, device=device)

    # Warm-up: run a few multiplications to "warm-up" the GPU
    for _ in range(10):
        _ = torch.mm(x, y)
    torch.cuda.synchronize()

    # Initialize CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record the start time
    start_event.record()
    # Use tqdm to display a progress bar during the iterations
    for _ in tqdm(range(iterations), desc=f"Benchmarking on {device}", leave=True):
        _ = torch.mm(x, y)
    # Record the end time
    end_event.record()

    # Synchronize to ensure all operations are complete
    torch.cuda.synchronize()

    # Calculate the elapsed time in milliseconds
    elapsed_time: float = start_event.elapsed_time(end_event)
    return elapsed_time


def main() -> None:
    """
    Main function to run the heavy benchmark on two NVIDIA GPUs (cuda:0 and cuda:1).

    If the system does not have at least two GPUs, a warning is logged.
    """
    num_devices: int = torch.cuda.device_count()
    if num_devices < 2:
        logger.warning(
            f"System does not have at least 2 NVIDIA GPUs. Found {num_devices} GPU(s)."
        )
        return

    # Increase the workload: using a larger matrix and more iterations
    iterations = 10000
    matrix_size = 2048

    # Benchmark on the first GPU (cuda:0)
    device0: str = "cuda:0"
    print(f"Benchmarking on {device0} ({torch.cuda.get_device_name(0)})")
    time0: float = benchmark(device0, iterations=iterations, matrix_size=matrix_size)
    print(f"Elapsed time on {device0}: {time0:.2f} ms\n")

    # Benchmark on the second GPU (cuda:1)
    device1: str = "cuda:1"
    print(f"Benchmarking on {device1} ({torch.cuda.get_device_name(1)})")
    time1: float = benchmark(device1, iterations=iterations, matrix_size=matrix_size)
    print(f"Elapsed time on {device1}: {time1:.2f} ms")


if __name__ == "__main__":
    main()
