import importlib
import logging
import sys
import types
import unittest
from unittest import mock

from click.testing import CliRunner


class FakeTensor:
    def __init__(self, name="tensor", elements=1, element_size=4):
        self.name = name
        self.elements = elements
        self.element_size_value = element_size
        self.fill_values = []

    def __matmul__(self, other):
        return FakeTensor("matmul")

    def __add__(self, other):
        return FakeTensor("add")

    def __mul__(self, other):
        return FakeTensor("mul")

    def sum(self):
        return FakeTensor("sum")

    def t(self):
        return FakeTensor("transpose")

    def fill_(self, value):
        self.fill_values.append(value)
        return self

    def numel(self):
        return self.elements

    def element_size(self):
        return self.element_size_value


class FakeDevice:
    def __init__(self, spec):
        self.spec = spec
        self.type = spec.split(":", 1)[0]

    def __repr__(self):
        return self.spec

    def __str__(self):
        return self.spec


class FakeCuda:
    def __init__(self, available=False, count=0):
        self.available = available
        self.count = count
        self.synchronized = []
        self.mem_info = (6 * 1024**3, 8 * 1024**3)
        self.empty_cache_calls = 0
        self.reset_peak_calls = []

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def synchronize(self, device):
        self.synchronized.append(device)

    def get_device_properties(self, idx):
        return types.SimpleNamespace(
            name=f"Fake GPU {idx}",
            total_memory=8 * 1024**3,
            multi_processor_count=80,
        )

    def mem_get_info(self, device):
        return self.mem_info

    def empty_cache(self):
        self.empty_cache_calls += 1

    def reset_peak_memory_stats(self, device):
        self.reset_peak_calls.append(device)

    def memory_allocated(self, device):
        return 1024**2

    def max_memory_allocated(self, device):
        return 2 * 1024**2

    def memory_reserved(self, device):
        return 3 * 1024**2

    def max_memory_reserved(self, device):
        return 4 * 1024**2


class FakeMps:
    def __init__(self):
        self.synchronize_calls = 0
        self.recommended = 10 * 1024**3
        self.driver_allocated = 2 * 1024**3
        self.current_allocated = 1024**3
        self.empty_cache_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1

    def recommended_max_memory(self):
        return self.recommended

    def driver_allocated_memory(self):
        return self.driver_allocated

    def current_allocated_memory(self):
        return self.current_allocated

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeMpsBackend:
    def __init__(self, available=False):
        self.available = available

    def is_available(self):
        return self.available


class FakeModule:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs
        return self

    def __call__(self, *args, **kwargs):
        return FakeTensor("module")


def make_fake_torch():
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "fake"
    fake_torch.float16 = object()
    fake_torch.float32 = object()
    fake_torch.float64 = object()
    fake_torch.device = FakeDevice
    fake_torch.cuda = FakeCuda()
    fake_torch.mps = FakeMps()
    fake_torch.backends = types.SimpleNamespace(mps=FakeMpsBackend())
    fake_torch.manual_seed = mock.Mock()
    fake_torch.created_tensors = []
    fake_torch.randn = mock.Mock(side_effect=lambda *args, **kwargs: FakeTensor())
    fake_torch.rand = mock.Mock(side_effect=lambda *args, **kwargs: FakeTensor("rand"))
    fake_torch.dot = mock.Mock(return_value=FakeTensor("dot"))
    fake_torch.relu = mock.Mock(return_value=FakeTensor("relu"))

    def empty(shape, device=None, dtype=None):
        elements = shape[0] if isinstance(shape, tuple) else shape
        if dtype == fake_torch.float16:
            element_size = 2
        elif dtype == fake_torch.float64:
            element_size = 8
        else:
            element_size = 4
        tensor = FakeTensor("empty", elements=elements, element_size=element_size)
        fake_torch.created_tensors.append(tensor)
        return tensor

    fake_torch.empty = mock.Mock(side_effect=empty)
    fake_torch.nn = types.SimpleNamespace(
        Conv2d=FakeModule,
        Flatten=FakeModule,
        Linear=FakeModule,
        ReLU=FakeModule,
        Sequential=FakeModule,
    )
    return fake_torch


def load_bench_module(fake_torch=None):
    sys.modules.pop("pybench.pytorch_bench", None)
    fake_torch = fake_torch or make_fake_torch()
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        module = importlib.import_module("pybench.pytorch_bench")
    return module, fake_torch


class PyTorchBenchTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("pybench.pytorch_bench", None)

    def test_sync_uses_torch_mps_synchronize(self):
        module, fake_torch = load_bench_module()

        module.sync(fake_torch.device("mps"))

        self.assertEqual(fake_torch.mps.synchronize_calls, 1)

    def test_sync_passes_cuda_device_to_cuda_synchronize(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cuda:0")

        module.sync(device)

        self.assertEqual(fake_torch.cuda.synchronized, [device])

    def test_cli_rejects_non_positive_iterations_and_size(self):
        module, _ = load_bench_module()
        runner = CliRunner()

        iterations_result = runner.invoke(module.main, ["--iterations", "0"])
        size_result = runner.invoke(module.main, ["--size", "0"])

        self.assertNotEqual(iterations_result.exit_code, 0)
        self.assertIn("Invalid value for '--iterations'", iterations_result.output)
        self.assertNotEqual(size_result.exit_code, 0)
        self.assertIn("Invalid value for '--size'", size_result.output)

    def test_cli_accepts_supported_suites_and_rejects_invalid_suite(self):
        module, _ = load_bench_module()
        runner = CliRunner()

        with mock.patch.object(module, "run_memory_stress"):
            for suite in ("basic", "extended", "all", "memory", "full"):
                with self.subTest(suite=suite):
                    result = runner.invoke(
                        module.main,
                        ["--suite", suite, "--iterations", "1", "--size", "2"],
                    )
                    self.assertEqual(result.exit_code, 0, result.output)

        invalid_result = runner.invoke(module.main, ["--suite", "legacy"])

        self.assertNotEqual(invalid_result.exit_code, 0)
        self.assertIn("Invalid value for '--suite'", invalid_result.output)

    def test_cli_validates_memory_controls(self):
        module, _ = load_bench_module()
        runner = CliRunner()

        low_percent = runner.invoke(module.main, ["--memory-percent", "0"])
        high_percent = runner.invoke(module.main, ["--memory-percent", "96"])
        bad_mb = runner.invoke(module.main, ["--memory-mb", "0"])

        self.assertNotEqual(low_percent.exit_code, 0)
        self.assertIn("Invalid value for '--memory-percent'", low_percent.output)
        self.assertNotEqual(high_percent.exit_code, 0)
        self.assertIn("Invalid value for '--memory-percent'", high_percent.output)
        self.assertNotEqual(bad_mb.exit_code, 0)
        self.assertIn("Invalid value for '--memory-mb'", bad_mb.output)

    def test_setup_logger_is_idempotent(self):
        module, _ = load_bench_module()
        logger = logging.getLogger("bench")
        original_handlers = list(logger.handlers)
        original_level = logger.level
        original_propagate = logger.propagate
        logger.handlers.clear()

        try:
            module.setup_logger(logging.INFO)
            module.setup_logger(logging.DEBUG)

            self.assertEqual(len(logger.handlers), 1)
            self.assertEqual(logger.level, logging.DEBUG)
            self.assertFalse(logger.propagate)
        finally:
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
            for handler in original_handlers:
                logger.addHandler(handler)
            logger.setLevel(original_level)
            logger.propagate = original_propagate

    def test_get_operations_selects_expected_suite_names(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cpu")

        basic = module.get_operations("basic", 2, device, fake_torch.float32)
        extended = module.get_operations("extended", 2, device, fake_torch.float32)
        all_ops = module.get_operations("all", 2, device, fake_torch.float32)
        full_ops = module.get_operations("full", 2, device, fake_torch.float32)
        memory_ops = module.get_operations("memory", 2, device, fake_torch.float32)

        self.assertEqual([name for name, _ in basic], list(module.BASIC_SUITE))
        self.assertEqual([name for name, _ in extended], list(module.EXTENDED_SUITE))
        self.assertEqual(
            [name for name, _ in all_ops],
            list(module.BASIC_SUITE + module.EXTENDED_SUITE),
        )
        self.assertEqual(
            [name for name, _ in full_ops],
            list(module.BASIC_SUITE + module.EXTENDED_SUITE),
        )
        self.assertEqual(memory_ops, [])

    def test_log_environment_info_includes_cuda_properties(self):
        fake_torch = make_fake_torch()
        fake_torch.cuda = FakeCuda(available=True, count=1)
        module, _ = load_bench_module(fake_torch)
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())

        module.log_environment_info(logger)

        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertIn("CUDA devices: 1", messages)
        self.assertIn(
            "CUDA device 0: Fake GPU 0, memory=8192.0 MB, multiprocessors=80",
            messages,
        )

    def test_run_operations_skips_failed_ops_and_continues(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        ops = [("bad", mock.Mock()), ("good", mock.Mock())]

        with mock.patch.object(
            module,
            "benchmark_op",
            side_effect=[RuntimeError("unsupported op"), None],
        ) as benchmark_op:
            module.run_operations(
                ops,
                fake_torch.device("cuda:0"),
                iterations=1,
                logger=logger,
                skip_failed_ops=True,
            )

        self.assertEqual(benchmark_op.call_count, 2)
        logger.warning.assert_called_once_with(
            "[cuda:0] Skipping bad: unsupported op"
        )

    def test_device_memory_info_covers_cpu_cuda_and_mps(self):
        module, fake_torch = load_bench_module()

        with mock.patch.object(
            module,
            "get_cpu_memory_info",
            return_value=(1000, 2000, "test-cpu"),
        ):
            self.assertEqual(
                module.get_device_memory_info(fake_torch.device("cpu")),
                (1000, 2000, "test-cpu"),
            )

        fake_torch.cuda.mem_info = (3000, 4000)
        self.assertEqual(
            module.get_device_memory_info(fake_torch.device("cuda:0")),
            (3000, 4000, "torch.cuda.mem_get_info"),
        )

        fake_torch.mps.recommended = 5000
        fake_torch.mps.driver_allocated = 1250
        self.assertEqual(
            module.get_device_memory_info(fake_torch.device("mps")),
            (3750, 5000, "torch.mps"),
        )

    def test_calculate_memory_target_uses_percent_and_mb_override(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cpu")

        with mock.patch.object(
            module,
            "get_device_memory_info",
            return_value=(1000, 2000, "test"),
        ):
            target, available, total, source = module.calculate_memory_target_bytes(
                device,
                memory_percent=70.0,
                memory_mb=None,
            )
            overridden, _, _, _ = module.calculate_memory_target_bytes(
                device,
                memory_percent=70.0,
                memory_mb=10,
            )

        self.assertEqual((target, available, total, source), (700, 1000, 2000, "test"))
        self.assertEqual(overridden, 10 * 1024**2)

    def test_run_memory_stress_allocates_touches_logs_and_releases(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")

        with (
            mock.patch.object(
                module,
                "calculate_memory_target_bytes",
                return_value=(32, 64, 128, "test"),
            ),
            mock.patch.object(module, "MAX_MEMORY_CHUNK_BYTES", 16),
            mock.patch.object(module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
        ):
            module.run_memory_stress(
                device,
                fake_torch.float32,
                iterations=2,
                memory_percent=70.0,
                memory_mb=None,
                logger=logger,
            )

        self.assertEqual(fake_torch.empty.call_count, 2)
        self.assertTrue(all(tensor.fill_values for tensor in fake_torch.created_tensors))
        self.assertFalse(logger.warning.called)
        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertTrue(any("memory: allocated=0.0 MB" in message for message in messages))

    def test_run_memory_stress_warns_and_cleans_up_on_failure(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cuda:0")

        with (
            mock.patch.object(
                module,
                "calculate_memory_target_bytes",
                return_value=(32, 64, 128, "test"),
            ),
            mock.patch.object(
                module,
                "allocate_memory_chunks",
                side_effect=RuntimeError("out of memory"),
            ),
        ):
            module.run_memory_stress(
                device,
                fake_torch.float32,
                iterations=1,
                memory_percent=70.0,
                memory_mb=None,
                logger=logger,
            )

        logger.warning.assert_called_once_with(
            "[cuda:0] Memory stress failed: out of memory"
        )
        self.assertEqual(fake_torch.cuda.empty_cache_calls, 1)

    def test_skip_reason_covers_cpu_float16(self):
        module, fake_torch = load_bench_module()

        reason = module.get_skip_reason(
            fake_torch.device("cpu"),
            fake_torch.float16,
        )

        self.assertEqual(
            reason,
            "Skipping FP16 benchmark on CPU (cpu): unsupported efficiently.",
        )

    def test_main_skips_mps_float64_with_warning(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        devices = [fake_torch.device("cpu"), fake_torch.device("mps")]

        with (
            mock.patch.object(module, "setup_logger", return_value=logger),
            mock.patch.object(module, "get_devices", return_value=devices),
            mock.patch.object(module, "benchmark_op") as benchmark_op,
        ):
            module.main.callback(
                iterations=1,
                size=2,
                dtype="double",
                suite="basic",
                memory_percent=70.0,
                memory_mb=None,
                seed=123,
                verbose=False,
            )

        logger.warning.assert_called_once_with(
            "Skipping FP64 benchmark on MPS (mps): unsupported."
        )
        self.assertEqual(benchmark_op.call_count, 4)


if __name__ == "__main__":
    unittest.main()
