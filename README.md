# pybench

Small PyTorch stress and benchmark tool for comparing common tensor operations
across available CPU, CUDA, and MPS devices.

## Usage

Run the default stress suite:

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

Run the fixed pybench-v1 benchmark profile and print a final score:

```bash
python pybench/pytorch_bench.py --mode benchmark --device cpu
```

Include the memory bandwidth subtests in the benchmark score:

```bash
python pybench/pytorch_bench.py --mode benchmark --benchmark-memory --memory-mb 512
```

Useful options:

- `--mode stress|benchmark`: run the configurable stress tests or the fixed
  scoring benchmark profile.
- `--suite basic|extended|all|memory|full`: choose the benchmark suite.
- `--iterations N`: number of timed iterations per operation.
- `--size N`: square matrix/vector size for tensor operations.
- `--dtype float|double|half`: tensor dtype where supported by the device.
- `--device cpu|cuda|cuda:N|mps`: restrict every selected suite to matching devices.
- `--memory-percent N`: percent of detected available RAM/VRAM to stress.
- `--memory-mb N`: exact memory stress target in MB per device.
- `--benchmark-memory`: include fixed memory bandwidth subtests in benchmark mode.
- `--benchmark-min-time N`: minimum timing duration per benchmark subtest.

Stress mode intentionally applies RAM/VRAM pressure when `--suite memory` or
`--suite full` is selected, and devices are stressed sequentially.

Benchmark mode always runs the fixed compute profile. By default its final score
is compute-only. With `--benchmark-memory`, pybench reports compute, memory, and
final scores where the final score is the geometric mean of compute and memory.
Scores are internal pybench-v1 scores, not Geekbench, SPEC, or MLPerf scores.
