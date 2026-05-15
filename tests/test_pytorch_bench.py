import importlib
import json
import logging
import sys
import types
import unittest
from unittest import mock

from click.testing import CliRunner


class FakeTensor:
    def __init__(self, name="tensor", elements=1, element_size=4, scalar=None):
        self.name = name
        self.elements = elements
        self.element_size_value = element_size
        self.fill_values = []
        self.scalar = scalar
        self.sum_calls = 0
        self.add_values = []

    def __matmul__(self, other):
        return FakeTensor("matmul")

    def __add__(self, other):
        return FakeTensor("add")

    def __mul__(self, other):
        return FakeTensor("mul")

    def sum(self, *args, **kwargs):
        self.sum_calls += 1
        if self.fill_values:
            return FakeTensor("sum", scalar=self.elements * self.fill_values[-1])
        return FakeTensor("sum", scalar=self.scalar)

    def t(self):
        return FakeTensor("transpose")

    def fill_(self, value):
        self.fill_values.append(value)
        return self

    def add_(self, value):
        self.add_values.append(value)
        return self

    def numel(self):
        return self.elements

    def element_size(self):
        return self.element_size_value

    def item(self):
        return 1.0 if self.scalar is None else self.scalar

    def all(self):
        return self


class FakeFinite:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


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
            pci_bus_id=None,
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


class FakeNvml(types.ModuleType):
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    nvmlClocksEventReasonGpuIdle = 0x1
    nvmlClocksEventReasonSwPowerCap = 0x4
    nvmlClocksEventReasonHwThermalSlowdown = 0x40
    NVML_CLOCK_THROTTLE_REASON_GPU_IDLE = 0x1
    NVML_CLOCK_THROTTLE_REASON_SW_POWER_CAP = 0x4
    NVML_CLOCK_THROTTLE_REASON_HW_THERMAL_SLOWDOWN = 0x40

    def __init__(self, temperature=70, throttle_bits=0):
        super().__init__("pynvml")
        self.temperature = temperature
        self.throttle_bits = throttle_bits
        self.shutdown_calls = 0
        self.pci_bus_ids = []
        self.event_reason_calls = 0
        self.throttle_reason_calls = 0

    def nvmlInit(self):
        return None

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle-{index}"

    def nvmlDeviceGetHandleByPciBusId(self, pci_bus_id):
        self.pci_bus_ids.append(pci_bus_id)
        return f"pci-handle-{pci_bus_id}"

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return self.temperature

    def nvmlDeviceGetPowerUsage(self, handle):
        return 125000

    def nvmlDeviceGetMemoryInfo(self, handle):
        return types.SimpleNamespace(used=2 * 1024**3, total=8 * 1024**3)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return types.SimpleNamespace(gpu=88, memory=42)

    def nvmlDeviceGetClockInfo(self, handle, clock_type):
        return 2100 if clock_type == self.NVML_CLOCK_SM else 9000

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, handle):
        self.throttle_reason_calls += 1
        return self.throttle_bits

    def nvmlDeviceGetCurrentClocksEventReasons(self, handle):
        self.event_reason_calls += 1
        return self.throttle_bits


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
    fake_torch.isfinite = mock.Mock(return_value=FakeFinite(True))

    def empty(shape, device=None, dtype=None):
        if shape == ():
            elements = 1
        elif isinstance(shape, tuple):
            elements = shape[0]
        else:
            elements = shape
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


def make_incomplete_torch():
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "incomplete"
    return fake_torch


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

    def test_cli_reports_incomplete_torch_installation(self):
        module, _ = load_bench_module(make_incomplete_torch())
        runner = CliRunner()

        result = runner.invoke(module.main, ["--iterations", "1", "--size", "1"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("PyTorch import is incomplete", result.output)
        self.assertIn("missing required attributes", result.output)

    def test_backend_detection_handles_missing_optional_namespaces(self):
        fake_torch = make_fake_torch()
        del fake_torch.cuda
        del fake_torch.backends
        del fake_torch.mps
        module, _ = load_bench_module(fake_torch)
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())

        module.log_environment_info(logger)
        devices = module.get_devices()

        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertIn("CUDA available: False", messages)
        self.assertEqual([str(device) for device in devices], ["cpu"])

    def test_filter_devices_matches_device_type_and_exact_device_spec(self):
        module, fake_torch = load_bench_module()
        devices = [
            fake_torch.device("cpu"),
            fake_torch.device("cuda:0"),
            fake_torch.device("cuda:1"),
            fake_torch.device("mps"),
        ]

        all_devices = module.filter_devices(devices, None)
        cuda_devices = module.filter_devices(devices, "cuda")
        exact_cuda_device = module.filter_devices(devices, "cuda:1")
        mps_devices = module.filter_devices(devices, "MPS")

        self.assertEqual(
            [str(device) for device in all_devices],
            ["cpu", "cuda:0", "cuda:1", "mps"],
        )
        self.assertEqual(
            [str(device) for device in cuda_devices],
            ["cuda:0", "cuda:1"],
        )
        self.assertEqual([str(device) for device in exact_cuda_device], ["cuda:1"])
        self.assertEqual([str(device) for device in mps_devices], ["mps"])

    def test_cli_rejects_unmatched_device_filter(self):
        module, _ = load_bench_module()
        runner = CliRunner()

        result = runner.invoke(
            module.main,
            ["--device", "cuda", "--iterations", "1", "--size", "2"],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No detected devices match --device 'cuda'", result.output)

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

    def test_cli_suite_choices_match_suite_constant(self):
        module, _ = load_bench_module()
        suite_option = next(
            param for param in module.main.params if "--suite" in param.opts
        )

        self.assertEqual(suite_option.type.choices, module.SUITE_CHOICES)

    def test_cli_mode_choices_match_mode_constant(self):
        module, _ = load_bench_module()
        mode_option = next(
            param for param in module.main.params if "--mode" in param.opts
        )

        self.assertEqual(mode_option.type.choices, module.MODE_CHOICES)

    def test_cli_correctness_choices_match_constant(self):
        module, _ = load_bench_module()
        correctness_option = next(
            param for param in module.main.params if "--correctness" in param.opts
        )

        self.assertEqual(correctness_option.type.choices, module.CORRECTNESS_CHOICES)

    def test_gpu_health_preset_sets_defaults_and_preserves_overrides(self):
        fake_torch = make_fake_torch()
        fake_torch.cuda = FakeCuda(available=True, count=1)
        module, _ = load_bench_module(fake_torch)
        runner = CliRunner()

        with mock.patch.object(module, "run_stress_for_device") as run_stress:
            result = runner.invoke(
                module.main,
                [
                    "--preset",
                    "gpu-health",
                    "--duration",
                    "3",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        args = run_stress.call_args.args
        self.assertEqual(str(args[0]), "cuda:0")
        self.assertEqual(args[1], "full")
        self.assertEqual(args[5], 80.0)
        self.assertIsNone(args[6])
        self.assertEqual(args[7], 3.0)
        self.assertEqual(args[8], "sampled")
        self.assertTrue(args[12].enabled)

        run_stress.reset_mock()
        with mock.patch.object(module, "run_stress_for_device", run_stress):
            override_result = runner.invoke(
                module.main,
                [
                    "--preset",
                    "gpu-health",
                    "--duration",
                    "3",
                    "--memory-mb",
                    "64",
                ],
            )

        self.assertEqual(override_result.exit_code, 0, override_result.output)
        override_args = run_stress.call_args.args
        self.assertEqual(override_args[5], 70.0)
        self.assertEqual(override_args[6], 64)

    def test_duration_is_forwarded_to_stress_runner(self):
        module, _ = load_bench_module()
        runner = CliRunner()

        with mock.patch.object(module, "run_stress_for_device") as run_stress:
            result = runner.invoke(
                module.main,
                ["--duration", "2", "--iterations", "99", "--size", "1"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        args = run_stress.call_args.args
        self.assertEqual(args[4], 99)
        self.assertEqual(args[7], 2.0)

    def test_progress_options_are_forwarded_to_stress_runner(self):
        module, _ = load_bench_module()
        runner = CliRunner()

        with mock.patch.object(module, "run_stress_for_device") as run_stress:
            result = runner.invoke(
                module.main,
                [
                    "--duration",
                    "2",
                    "--size",
                    "1",
                    "--no-progress",
                    "--progress-interval",
                    "7",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse(run_stress.call_args.kwargs["progress"])
        self.assertEqual(run_stress.call_args.kwargs["progress_interval"], 7.0)

    def test_duration_is_rejected_in_benchmark_mode(self):
        module, _ = load_bench_module()
        runner = CliRunner()

        result = runner.invoke(
            module.main,
            ["--mode", "benchmark", "--duration", "2"],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--duration is only supported in stress mode", result.output)

    def test_json_report_writes_health_shape(self):
        module, _ = load_bench_module()
        runner = CliRunner()

        with runner.isolated_filesystem():
            with mock.patch.object(module, "run_stress_for_device"):
                result = runner.invoke(
                    module.main,
                    ["--json-report", "report.json", "--iterations", "1", "--size", "1"],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            with open("report.json", encoding="utf-8") as report_file:
                report = json.load(report_file)

        self.assertEqual(report["schema_version"], module.JSON_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["status"], module.HEALTH_PASS)
        self.assertEqual(report["config"]["mode"], "stress")
        self.assertEqual(report["devices"][0]["name"], "cpu")

    def test_telemetry_records_nvml_temperature_failure(self):
        fake_torch = make_fake_torch()
        fake_torch.cuda = FakeCuda(available=True, count=1)
        module, fake_torch = load_bench_module(fake_torch)
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        health = module.HealthReport(max_temp_c=90.0)
        fake_nvml = FakeNvml(
            temperature=95,
            throttle_bits=FakeNvml.NVML_CLOCK_THROTTLE_REASON_HW_THERMAL_SLOWDOWN,
        )

        with mock.patch.dict(sys.modules, {"pynvml": fake_nvml}):
            monitor = module.TelemetryMonitor(
                True,
                [fake_torch.device("cuda:0")],
                1.0,
                logger,
                health,
            )
            monitor.sample(fake_torch.device("cuda:0"), force=True)
            monitor.close()

        self.assertEqual(health.status, module.HEALTH_FAIL)
        self.assertEqual(fake_nvml.shutdown_calls, 1)
        messages = [issue.message for issue in health.failures]
        self.assertIn("temperature exceeded limit 90.0C", messages)
        self.assertTrue(any("hard throttle" in message for message in messages))

    def test_nvml_throttle_decode_uses_documented_fallback_bits(self):
        module, _ = load_bench_module()

        reasons = module.decode_nvml_throttle_reasons(
            types.SimpleNamespace(),
            0x1 | 0x4 | 0x200,
        )

        self.assertEqual(reasons, ("gpu_idle", "sw_power_cap", "unknown_0x200"))

    def test_nvml_backend_falls_back_when_event_reason_api_is_absent(self):
        class OldNvml(FakeNvml):
            nvmlDeviceGetCurrentClocksEventReasons = None

        fake_torch = make_fake_torch()
        fake_torch.cuda = FakeCuda(available=True, count=1)
        module, fake_torch = load_bench_module(fake_torch)
        fake_nvml = OldNvml(
            throttle_bits=FakeNvml.NVML_CLOCK_THROTTLE_REASON_SW_POWER_CAP,
        )
        backend = module.NvmlTelemetryBackend(fake_nvml)

        sample = backend.sample(fake_torch.device("cuda:0"))

        self.assertEqual(sample["throttle_reasons"], ("sw_power_cap",))
        self.assertEqual(fake_nvml.throttle_reason_calls, 1)

    def test_nvml_backend_falls_back_when_event_reason_api_fails(self):
        class EventFailingNvml(FakeNvml):
            def nvmlDeviceGetCurrentClocksEventReasons(self, handle):
                self.event_reason_calls += 1
                raise RuntimeError("event reasons unavailable")

        fake_torch = make_fake_torch()
        fake_torch.cuda = FakeCuda(available=True, count=1)
        module, fake_torch = load_bench_module(fake_torch)
        fake_nvml = EventFailingNvml(
            throttle_bits=FakeNvml.NVML_CLOCK_THROTTLE_REASON_SW_POWER_CAP,
        )
        backend = module.NvmlTelemetryBackend(fake_nvml)

        sample = backend.sample(fake_torch.device("cuda:0"))

        self.assertEqual(sample["throttle_reasons"], ("sw_power_cap",))
        self.assertEqual(fake_nvml.event_reason_calls, 1)
        self.assertEqual(fake_nvml.throttle_reason_calls, 1)

    def test_sw_power_cap_telemetry_is_recorded_without_health_warning(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cuda:0")
        health = module.HealthReport(max_temp_c=90.0)

        health.add_telemetry_sample(
            module.TelemetrySample(
                timestamp="now",
                elapsed_s=1.0,
                device=str(device),
                source="nvml",
                throttle_reasons=("gpu_idle", "sw_power_cap"),
            )
        )

        self.assertEqual(health.status, module.HEALTH_PASS)
        self.assertEqual(
            health.telemetry_summary(device)["throttle_reasons"],
            ["gpu_idle", "sw_power_cap"],
        )

    def test_correctness_failure_exits_nonzero(self):
        fake_torch = make_fake_torch()
        fake_torch.isfinite = mock.Mock(return_value=FakeFinite(False))
        module, _ = load_bench_module(fake_torch)
        runner = CliRunner()

        result = runner.invoke(
            module.main,
            ["--correctness", "smoke", "--iterations", "1", "--size", "1"],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Health check failed", result.output)

    def test_memory_smoke_correctness_validates_first_touch_once(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cpu")
        chunk = FakeTensor(elements=4)

        with (
            mock.patch.object(module, "sync"),
            mock.patch.object(
                module,
                "validate_memory_sum",
                wraps=module.validate_memory_sum,
            ) as validate_memory_sum,
        ):
            module.touch_memory_chunks_once(
                [chunk],
                device,
                fake_torch.float32,
                iteration_index=0,
                correctness="smoke",
            )

        self.assertEqual(chunk.sum_calls, 1)
        validate_memory_sum.assert_called_once()
        self.assertIn("observed_sum", validate_memory_sum.call_args.kwargs)

    def test_memory_touch_without_validation_avoids_reduction(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cpu")
        chunk = FakeTensor(elements=4)

        module.touch_memory_chunks_once(
            [chunk],
            device,
            fake_torch.float32,
            iteration_index=1,
            correctness="sampled",
            correctness_interval=10,
        )

        self.assertEqual(chunk.sum_calls, 0)
        self.assertEqual(chunk.add_values, [0])

    def test_memory_correctness_policy_covers_smoke_sampled_and_strict(self):
        module, _ = load_bench_module()

        self.assertTrue(module.should_validate_memory_iteration("smoke", 0, 10))
        self.assertFalse(module.should_validate_memory_iteration("smoke", 1, 10))
        self.assertTrue(module.should_validate_memory_iteration("sampled", 0, 10))
        self.assertTrue(module.should_validate_memory_iteration("sampled", 10, 10))
        self.assertFalse(module.should_validate_memory_iteration("sampled", 11, 10))
        self.assertTrue(module.should_validate_memory_iteration("strict", 99, 10))

    def test_duration_skips_failed_operation_after_first_failure(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        health = module.HealthReport(max_temp_c=90.0)
        bad_op = mock.Mock(side_effect=RuntimeError("boom"))
        good_op = mock.Mock(return_value=FakeTensor())

        with (
            mock.patch.object(module, "warm_up_operations", return_value=set()),
            mock.patch.object(module, "sync"),
        ):
            module.run_operations_for_duration(
                [("bad", bad_op), ("good", good_op)],
                device,
                module.time.perf_counter() + 0.003,
                logger,
                health=health,
            )

        self.assertEqual(bad_op.call_count, 1)
        self.assertGreater(good_op.call_count, 0)
        self.assertEqual(health.status, module.HEALTH_FAIL)
        self.assertTrue(
            any("operation bad failed: boom" == issue.message for issue in health.failures)
        )

    def test_duration_skips_operation_that_fails_during_warmup(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        health = module.HealthReport(max_temp_c=90.0)
        bad_op = mock.Mock(side_effect=RuntimeError("warm boom"))
        good_op = mock.Mock(return_value=FakeTensor())

        with mock.patch.object(module, "sync"):
            module.run_operations_for_duration(
                [("bad", bad_op), ("good", good_op)],
                device,
                module.time.perf_counter() + 0.003,
                logger,
                health=health,
            )

        self.assertEqual(bad_op.call_count, 1)
        self.assertGreater(good_op.call_count, 5)
        self.assertTrue(
            any(
                "operation bad failed during warmup: warm boom" == issue.message
                for issue in health.failures
            )
        )

    def test_duration_memory_failure_marks_context_without_success_summary(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        health = module.HealthReport(max_temp_c=90.0)
        memory_context = module.MemoryStressContext(
            device=device,
            chunks=[FakeTensor()],
            allocated_bytes=4,
            available_bytes=8,
            total_bytes=16,
            source="test",
        )
        memory_times = []

        with (
            mock.patch.object(module, "warm_up_operations", return_value=set()),
            mock.patch.object(module, "sync"),
            mock.patch.object(
                module,
                "touch_memory_chunks_once",
                side_effect=RuntimeError("bad memory"),
            ) as touch_memory,
        ):
            module.run_operations_for_duration(
                [("good", lambda: FakeTensor())],
                device,
                module.time.perf_counter() + 0.003,
                logger,
                health=health,
                memory_context=memory_context,
                memory_dt=fake_torch.float32,
                memory_times=memory_times,
            )

        self.assertTrue(memory_context.failed)
        self.assertEqual(len(memory_times), 0)
        self.assertEqual(touch_memory.call_count, 1)
        summaries = health.memory_summaries[str(device)]
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0].failed)
        self.assertEqual(summaries[0].error, "bad memory")

    def test_duration_full_builds_compute_before_memory_allocation(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        health = module.HealthReport(max_temp_c=90.0)
        order = []

        def build_basic(*args, **kwargs):
            order.append("basic")
            return [("basic", lambda: FakeTensor())]

        def build_extended(*args, **kwargs):
            order.append("extended")
            return []

        def prepare_memory(*args, **kwargs):
            order.append("memory")
            return module.MemoryStressContext(
                device=device,
                chunks=[FakeTensor()],
                allocated_bytes=4,
                available_bytes=8,
                total_bytes=16,
                source="test",
            )

        with (
            mock.patch.object(module, "build_basic_ops", side_effect=build_basic),
            mock.patch.object(module, "build_extended_ops", side_effect=build_extended),
            mock.patch.object(module, "prepare_memory_stress", side_effect=prepare_memory),
            mock.patch.object(module, "run_operations_for_duration"),
            mock.patch.object(
                module,
                "run_memory_context_until_deadline",
                return_value=[0.1],
            ),
            mock.patch.object(module, "release_memory_chunks"),
        ):
            module.run_stress_for_device(
                device,
                suite="full",
                size=1,
                dt=fake_torch.float32,
                iterations=1,
                memory_percent=70.0,
                memory_mb=None,
                duration=1.0,
                correctness="off",
                correctness_interval=10,
                logger=logger,
                health=health,
                telemetry_monitor=None,
            )

        self.assertEqual(order, ["basic", "extended", "memory"])

    def test_stress_cleanup_failure_warns_without_raising(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        health = module.HealthReport(max_temp_c=90.0)
        memory_context = module.MemoryStressContext(
            device=device,
            chunks=[FakeTensor()],
            allocated_bytes=4,
            available_bytes=8,
            total_bytes=16,
            source="test",
        )

        with (
            mock.patch.object(module, "build_basic_ops", return_value=[]),
            mock.patch.object(module, "prepare_memory_stress", return_value=memory_context),
            mock.patch.object(
                module,
                "run_memory_context_until_deadline",
                return_value=[0.1],
            ),
            mock.patch.object(
                module,
                "release_memory_chunks",
                side_effect=RuntimeError("cleanup timeout"),
            ),
        ):
            module.run_stress_for_device(
                device,
                suite="full",
                size=1,
                dt=fake_torch.float32,
                iterations=1,
                memory_percent=70.0,
                memory_mb=None,
                duration=1.0,
                correctness="off",
                correctness_interval=10,
                logger=logger,
                health=health,
                telemetry_monitor=None,
            )

        self.assertEqual(health.status, module.HEALTH_WARN)
        self.assertTrue(
            any("memory cleanup failed: cleanup timeout" == issue.message for issue in health.warnings)
        )

    def test_telemetry_torch_memory_failures_warn_without_aborting_sample(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        health = module.HealthReport(max_temp_c=90.0)
        monitor = module.TelemetryMonitor(True, [device], 1.0, logger, health)

        with (
            mock.patch.object(
                module,
                "get_memory_stats",
                side_effect=RuntimeError("stats unavailable"),
            ),
            mock.patch.object(
                module,
                "get_device_memory_info",
                side_effect=RuntimeError("info unavailable"),
            ),
        ):
            monitor.sample(device, force=True)

        self.assertEqual(health.telemetry_summary(device)["sample_count"], 1)
        warning_messages = [issue.message for issue in health.warnings]
        self.assertIn("PyTorch memory stats unavailable: stats unavailable", warning_messages)
        self.assertIn("PyTorch memory info unavailable: info unavailable", warning_messages)

    def test_nvml_backend_prefers_pci_bus_id_handle_lookup(self):
        fake_torch = make_fake_torch()
        fake_torch.cuda = FakeCuda(available=True, count=1)
        module, fake_torch = load_bench_module(fake_torch)
        fake_torch.cuda.get_device_properties = mock.Mock(
            return_value=types.SimpleNamespace(pci_bus_id="0000:01:00.0")
        )
        fake_nvml = FakeNvml()
        backend = module.NvmlTelemetryBackend(fake_nvml)

        handle = backend.get_handle(fake_torch.device("cuda:0"))

        self.assertEqual(handle, "pci-handle-0000:01:00.0")
        self.assertEqual(fake_nvml.pci_bus_ids, ["0000:01:00.0"])

    def test_telemetry_summary_includes_utilization_memory_and_clocks(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cuda:0")
        health = module.HealthReport(max_temp_c=90.0)

        health.add_telemetry_sample(
            module.TelemetrySample(
                timestamp="now",
                elapsed_s=1.0,
                device=str(device),
                source="nvml",
                memory_used_mb=2048.0,
                utilization_gpu_percent=88.0,
                utilization_memory_percent=42.0,
                sm_clock_mhz=2100.0,
                memory_clock_mhz=9000.0,
            )
        )

        summary = health.telemetry_summary(device)

        self.assertEqual(summary["max_memory_used_mb"], 2048.0)
        self.assertEqual(summary["max_gpu_utilization_percent"], 88.0)
        self.assertEqual(summary["max_memory_utilization_percent"], 42.0)
        self.assertEqual(summary["max_sm_clock_mhz"], 2100.0)
        self.assertEqual(summary["max_memory_clock_mhz"], 9000.0)

    def test_cli_memory_failure_exits_nonzero_and_writes_json_report(self):
        module, _ = load_bench_module()
        runner = CliRunner()
        failed_result = module.MemoryStressResult(
            device="cpu",
            iterations=0,
            allocated_bytes=0,
            total_time=0.0,
            avg_touch_time=0.0,
            bandwidth_gib_s=0.0,
            stats={},
            failed=True,
            error="bad memory",
        )

        with runner.isolated_filesystem():
            with mock.patch.object(module, "run_memory_stress", return_value=failed_result):
                result = runner.invoke(
                    module.main,
                    [
                        "--suite",
                        "memory",
                        "--iterations",
                        "1",
                        "--json-report",
                        "report.json",
                    ],
                )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Health check failed", result.output)
            with open("report.json", encoding="utf-8") as report_file:
                report = json.load(report_file)

        self.assertEqual(report["status"], module.HEALTH_FAIL)
        self.assertEqual(report["issues"][0]["message"], "memory stress failed: bad memory")

    def test_geometric_mean_and_benchmark_summary_scores(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cpu")
        compute_results = [
            module.BenchmarkResult(
                name="compute-a",
                category="compute",
                device=device,
                median_time=1.0,
                iqr_time=0.0,
                throughput=1.0,
                throughput_unit="ops/s",
                score=100.0,
            ),
            module.BenchmarkResult(
                name="compute-b",
                category="compute",
                device=device,
                median_time=1.0,
                iqr_time=0.0,
                throughput=1.0,
                throughput_unit="ops/s",
                score=400.0,
            ),
        ]
        memory_results = [
            module.BenchmarkResult(
                name="memory-a",
                category="memory",
                device=device,
                median_time=1.0,
                iqr_time=0.0,
                throughput=1.0,
                throughput_unit="GiB/s",
                score=900.0,
            )
        ]

        summary = module.summarize_benchmark_scores(compute_results, memory_results)

        self.assertAlmostEqual(module.geometric_mean([100.0, 400.0]), 200.0)
        self.assertAlmostEqual(summary.compute_score, 200.0)
        self.assertAlmostEqual(summary.memory_score, 900.0)
        self.assertAlmostEqual(summary.final_score, (200.0 * 900.0) ** 0.5)

    def test_run_benchmark_tests_scores_throughput_and_skips_failures(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        tests = [
            module.BenchmarkTest(
                name="bad",
                category="compute",
                fn=lambda: None,
                work_units=1.0,
                throughput_unit="ops/s",
                baseline=1.0,
            ),
            module.BenchmarkTest(
                name="good",
                category="compute",
                fn=lambda: None,
                work_units=4.0,
                throughput_unit="ops/s",
                baseline=2.0,
            ),
        ]

        with mock.patch.object(
            module,
            "measure_benchmark_test",
            side_effect=[RuntimeError("unsupported op"), (0.5, 0.1)],
        ):
            results = module.run_benchmark_tests(
                tests,
                device,
                min_run_time=0.25,
                logger=logger,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "good")
        self.assertEqual(results[0].throughput, 8.0)
        self.assertEqual(results[0].score, 4000.0)
        logger.warning.assert_called_once_with(
            "[cpu] Skipping benchmark bad: unsupported op"
        )

    def test_run_benchmark_tests_logs_progress_around_measurement(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        test = module.BenchmarkTest(
            name="good",
            category="compute",
            fn=lambda: None,
            work_units=4.0,
            throughput_unit="ops/s",
            baseline=2.0,
        )

        with mock.patch.object(
            module,
            "measure_benchmark_test",
            return_value=(0.5, 0.1),
        ) as measure:
            results = module.run_benchmark_tests(
                [test],
                device,
                min_run_time=0.25,
                logger=logger,
                progress=True,
            )

        self.assertEqual(len(results), 1)
        measure.assert_called_once_with(test, device, 0.25)
        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertIn("[cpu] benchmark good: started", messages)
        self.assertIn("[cpu] benchmark good: finished", messages)

    def test_run_benchmark_mode_runs_compute_by_default_without_memory(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        compute_test = module.BenchmarkTest(
            name="compute",
            category="compute",
            fn=lambda: None,
            work_units=1.0,
            throughput_unit="ops/s",
            baseline=1.0,
        )
        compute_result = module.BenchmarkResult(
            name="compute",
            category="compute",
            device=device,
            median_time=1.0,
            iqr_time=0.0,
            throughput=1.0,
            throughput_unit="ops/s",
            score=1000.0,
        )

        with (
            mock.patch.object(
                module,
                "build_benchmark_compute_tests",
                return_value=[compute_test],
            ) as build_compute,
            mock.patch.object(
                module,
                "build_benchmark_memory_tests",
            ) as build_memory,
            mock.patch.object(
                module,
                "run_benchmark_tests",
                return_value=[compute_result],
            ) as run_tests,
        ):
            summaries = module.run_benchmark_mode(
                [device],
                fake_torch.float32,
                benchmark_memory=False,
                memory_mb=None,
                benchmark_min_time=0.25,
                logger=logger,
            )

        build_compute.assert_called_once_with(device, fake_torch.float32)
        build_memory.assert_not_called()
        run_tests.assert_called_once_with(
            [compute_test],
            device,
            0.25,
            logger,
            progress=True,
        )
        self.assertAlmostEqual(summaries[device].final_score, 1000.0)

    def test_run_benchmark_mode_includes_memory_when_enabled_and_releases_chunks(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")
        chunk = FakeTensor("chunk")
        compute_test = module.BenchmarkTest(
            name="compute",
            category="compute",
            fn=lambda: None,
            work_units=1.0,
            throughput_unit="ops/s",
            baseline=1.0,
        )
        memory_test = module.BenchmarkTest(
            name="memory",
            category="memory",
            fn=lambda: None,
            work_units=1.0,
            throughput_unit="GiB/s",
            baseline=1.0,
        )
        compute_result = module.BenchmarkResult(
            name="compute",
            category="compute",
            device=device,
            median_time=1.0,
            iqr_time=0.0,
            throughput=1.0,
            throughput_unit="ops/s",
            score=1000.0,
        )
        memory_result = module.BenchmarkResult(
            name="memory",
            category="memory",
            device=device,
            median_time=1.0,
            iqr_time=0.0,
            throughput=1.0,
            throughput_unit="GiB/s",
            score=250.0,
        )

        with (
            mock.patch.object(
                module,
                "build_benchmark_compute_tests",
                return_value=[compute_test],
            ),
            mock.patch.object(
                module,
                "build_benchmark_memory_tests",
                return_value=([memory_test], [chunk]),
            ) as build_memory,
            mock.patch.object(
                module,
                "run_benchmark_tests",
                side_effect=[[compute_result], [memory_result]],
            ) as run_tests,
            mock.patch.object(module, "release_memory_chunks") as release_chunks,
        ):
            summaries = module.run_benchmark_mode(
                [device],
                fake_torch.float32,
                benchmark_memory=True,
                memory_mb=128,
                benchmark_min_time=0.25,
                logger=logger,
            )

        build_memory.assert_called_once_with(device, fake_torch.float32, 128, logger)
        self.assertEqual(run_tests.call_count, 2)
        release_chunks.assert_called_once_with([chunk], device)
        self.assertAlmostEqual(summaries[device].final_score, (1000.0 * 250.0) ** 0.5)

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

    def test_cuda_memory_info_estimates_when_mem_get_info_is_unavailable(self):
        fake_torch = make_fake_torch()
        fake_torch.cuda.mem_get_info = None
        module, _ = load_bench_module(fake_torch)

        available, total, source = module.get_device_memory_info(
            fake_torch.device("cuda:0")
        )

        self.assertEqual(total, 8 * 1024**3)
        self.assertEqual(available, total - 3 * 1024**2)
        self.assertEqual(source, "torch.cuda memory estimate")

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

    def test_calculate_memory_target_rejects_zero_detected_target(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cpu")

        with mock.patch.object(
            module,
            "get_device_memory_info",
            return_value=(0, 2000, "test"),
        ):
            with self.assertRaisesRegex(RuntimeError, "No available memory"):
                module.calculate_memory_target_bytes(
                    device,
                    memory_percent=70.0,
                    memory_mb=None,
                )

    def test_allocate_memory_chunks_ceilings_to_target_and_reports_actual_bytes(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cpu")

        with mock.patch.object(module, "MAX_MEMORY_CHUNK_BYTES", 6):
            chunks, allocated_bytes = module.allocate_memory_chunks(
                target_bytes=5,
                device=device,
                dt=fake_torch.float32,
            )

        self.assertEqual(len(chunks), 2)
        self.assertGreaterEqual(allocated_bytes, 5)
        self.assertEqual(
            allocated_bytes,
            sum(tensor.numel() * tensor.element_size() for tensor in chunks),
        )

    def test_allocate_memory_chunks_cleans_partial_allocations_on_failure(self):
        module, fake_torch = load_bench_module()
        device = fake_torch.device("cuda:0")
        successful_chunk = FakeTensor("partial", elements=1, element_size=4)
        fake_torch.empty.side_effect = [successful_chunk, RuntimeError("out of memory")]

        with (
            mock.patch.object(module, "MAX_MEMORY_CHUNK_BYTES", 4),
            mock.patch.object(module, "clear_device_cache") as clear_device_cache,
            self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            module.allocate_memory_chunks(
                target_bytes=8,
                device=device,
                dt=fake_torch.float32,
            )

        clear_device_cache.assert_called_once_with(device)

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

    def test_no_progress_disables_tqdm_for_fixed_iteration_stress(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("cpu")

        with (
            mock.patch.object(module, "sync"),
            mock.patch.object(
                module,
                "tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ) as tqdm_mock,
        ):
            module.benchmark_op(
                "op",
                lambda: FakeTensor(),
                device,
                iterations=1,
                logger=logger,
                progress=False,
            )

        self.assertTrue(tqdm_mock.call_args.kwargs["disable"])

    def test_progress_reporter_logs_time_only_heartbeat(self):
        module, _ = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        clock_values = iter([0.0, 30.0, 61.0, 90.0])
        reporter = module.ProgressReporter(
            logger,
            enabled=True,
            interval=60.0,
            clock=lambda: next(clock_values),
        )

        reporter.start("[cpu] duration stress", total_seconds=120.0)
        reporter.maybe_log()
        reporter.maybe_log()
        reporter.finish()

        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertIn("[cpu] duration stress: started, expected=2m 0s", messages)
        self.assertIn(
            "[cpu] duration stress: elapsed=1m 1s, remaining=59s, progress=51%",
            messages,
        )
        self.assertIn("[cpu] duration stress: finished in 1m 30s", messages)

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
                device=None,
                mode="stress",
                benchmark_memory=False,
                benchmark_min_time=0.5,
                verbose=False,
            )

        logger.warning.assert_called_once_with(
            "Skipping FP64 benchmark on MPS (mps): unsupported."
        )
        self.assertEqual(benchmark_op.call_count, 4)

    def test_main_device_filter_applies_to_compute_and_memory_suites(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        cpu_device = fake_torch.device("cpu")
        cuda_device = fake_torch.device("cuda:0")
        basic_op = ("basic", lambda: None)
        extended_op = ("extended", lambda: None)

        with (
            mock.patch.object(module, "setup_logger", return_value=logger),
            mock.patch.object(
                module,
                "get_devices",
                return_value=[cpu_device, cuda_device],
            ),
            mock.patch.object(
                module,
                "build_basic_ops",
                return_value=[basic_op],
            ) as build_basic_ops,
            mock.patch.object(
                module,
                "build_extended_ops",
                return_value=[extended_op],
            ) as build_extended_ops,
            mock.patch.object(module, "benchmark_op") as benchmark_op,
            mock.patch.object(module, "run_memory_stress") as run_memory_stress,
        ):
            module.main.callback(
                iterations=1,
                size=2,
                dtype="float",
                suite="full",
                memory_percent=70.0,
                memory_mb=None,
                seed=123,
                device="cpu",
                mode="stress",
                benchmark_memory=False,
                benchmark_min_time=0.5,
                verbose=False,
            )

        build_basic_ops.assert_called_once_with(2, cpu_device, fake_torch.float32)
        build_extended_ops.assert_called_once_with(2, cpu_device, fake_torch.float32)
        self.assertEqual(
            [call.args[2] for call in benchmark_op.call_args_list],
            [cpu_device, cpu_device],
        )
        run_memory_stress.assert_called_once_with(
            cpu_device,
            fake_torch.float32,
            1,
            70.0,
            None,
            logger,
            progress=True,
        )

    def test_main_uses_float32_memory_stress_dtype_for_mps_float64(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        device = fake_torch.device("mps")

        with (
            mock.patch.object(module, "setup_logger", return_value=logger),
            mock.patch.object(module, "get_devices", return_value=[device]),
            mock.patch.object(module, "run_memory_stress") as run_memory_stress,
        ):
            module.main.callback(
                iterations=1,
                size=2,
                dtype="double",
                suite="memory",
                memory_percent=70.0,
                memory_mb=None,
                seed=123,
                device=None,
                mode="stress",
                benchmark_memory=False,
                benchmark_min_time=0.5,
                verbose=False,
            )

        logger.warning.assert_called_once_with(
            "[mps] Using float32 for memory stress because MPS does not support FP64."
        )
        run_memory_stress.assert_called_once_with(
            device,
            fake_torch.float32,
            1,
            70.0,
            None,
            logger,
            progress=True,
        )

    def test_main_benchmark_mode_uses_device_filter(self):
        module, fake_torch = load_bench_module()
        logger = types.SimpleNamespace(info=mock.Mock(), warning=mock.Mock())
        cpu_device = fake_torch.device("cpu")
        cuda_device = fake_torch.device("cuda:0")

        with (
            mock.patch.object(module, "setup_logger", return_value=logger),
            mock.patch.object(
                module,
                "get_devices",
                return_value=[cpu_device, cuda_device],
            ),
            mock.patch.object(module, "run_benchmark_mode") as run_benchmark_mode,
        ):
            module.main.callback(
                iterations=1,
                size=2,
                dtype="float",
                suite="full",
                memory_percent=70.0,
                memory_mb=None,
                seed=123,
                device="cuda:0",
                mode="benchmark",
                benchmark_memory=True,
                benchmark_min_time=0.25,
                verbose=False,
            )

        run_benchmark_mode.assert_called_once_with(
            [cuda_device],
            fake_torch.float32,
            True,
            None,
            0.25,
            logger,
            progress=True,
        )


if __name__ == "__main__":
    unittest.main()
