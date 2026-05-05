# pybench

Small PyTorch benchmark for comparing common tensor operations across available
CPU, CUDA, and MPS devices.

## Usage

Run the default basic suite:

```bash
python pybench/pytorch_bench.py
```

Run the extended operation suite:

```bash
python pybench/pytorch_bench.py --suite extended
```

Run every available operation with a custom matrix size and iteration count:

```bash
python pybench/pytorch_bench.py --suite all --size 2048 --iterations 50
```

Stress RAM/VRAM only, using 25% of detected available memory:

```bash
python pybench/pytorch_bench.py --suite memory --memory-percent 25 --iterations 3
```

Run compute and memory stress together, with an exact memory target per device:

```bash
python pybench/pytorch_bench.py --suite full --memory-mb 4096 --iterations 5
```

Run all compute and memory stress only on CPU:

```bash
python pybench/pytorch_bench.py --suite full --device cpu --memory-mb 4096
```

Useful options:

- `--suite basic|extended|all|memory|full`: choose the benchmark suite.
- `--iterations N`: number of timed iterations per operation.
- `--size N`: square matrix/vector size for tensor operations.
- `--dtype float|double|half`: tensor dtype where supported by the device.
- `--device cpu|cuda|cuda:N|mps`: restrict every selected suite to matching devices.
- `--memory-percent N`: percent of detected available RAM/VRAM to stress.
- `--memory-mb N`: exact memory stress target in MB per device.

Memory stress intentionally applies RAM/VRAM pressure. It runs only for the
`memory` and `full` suites, and devices are stressed sequentially.
