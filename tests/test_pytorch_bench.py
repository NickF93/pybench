import importlib
import logging
import sys
import types
import unittest
from unittest import mock

from click.testing import CliRunner


class FakeTensor:
    def __init__(self, name="tensor"):
        self.name = name

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


class FakeMps:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


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
    fake_torch.randn = mock.Mock(side_effect=lambda *args, **kwargs: FakeTensor())
    fake_torch.rand = mock.Mock(side_effect=lambda *args, **kwargs: FakeTensor("rand"))
    fake_torch.dot = mock.Mock(return_value=FakeTensor("dot"))
    fake_torch.relu = mock.Mock(return_value=FakeTensor("relu"))
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

        for suite in ("basic", "extended", "all"):
            with self.subTest(suite=suite):
                result = runner.invoke(
                    module.main,
                    ["--suite", suite, "--iterations", "1", "--size", "2"],
                )
                self.assertEqual(result.exit_code, 0, result.output)

        invalid_result = runner.invoke(module.main, ["--suite", "legacy"])

        self.assertNotEqual(invalid_result.exit_code, 0)
        self.assertIn("Invalid value for '--suite'", invalid_result.output)

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

        self.assertEqual([name for name, _ in basic], list(module.BASIC_SUITE))
        self.assertEqual([name for name, _ in extended], list(module.EXTENDED_SUITE))
        self.assertEqual(
            [name for name, _ in all_ops],
            list(module.BASIC_SUITE + module.EXTENDED_SUITE),
        )

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
                seed=123,
                verbose=False,
            )

        logger.warning.assert_called_once_with(
            "Skipping FP64 benchmark on MPS (mps): unsupported."
        )
        self.assertEqual(benchmark_op.call_count, 4)


if __name__ == "__main__":
    unittest.main()
