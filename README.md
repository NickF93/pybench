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

Useful options:

- `--suite basic|extended|all`: choose the benchmark suite.
- `--iterations N`: number of timed iterations per operation.
- `--size N`: square matrix/vector size for tensor operations.
- `--dtype float|double|half`: tensor dtype where supported by the device.
