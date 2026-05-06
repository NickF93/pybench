from __future__ import annotations

import importlib
import json
import time
import statistics
import logging
import gc
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import StreamHandler, Formatter
from typing import Any, Callable

import click
from click.core import ParameterSource
from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm
import torch


BASIC_SUITE = ("matmul", "add", "mul", "sum")
EXTENDED_SUITE = ("dot", "transpose", "relu", "rand", "conv2d", "model_forward")
MODE_CHOICES = ("stress", "benchmark")
SUITE_CHOICES = ("basic", "extended", "all", "memory", "full")
DTYPE_CHOICES = ("float", "double", "half")
PRESET_CHOICES = ("gpu-health",)
CORRECTNESS_CHOICES = ("off", "smoke", "sampled", "strict")
COMPUTE_SUITES = ("basic", "extended", "all", "full")
MEMORY_SUITES = ("memory", "full")
MAX_MEMORY_CHUNK_BYTES = 256 * 1024**2
BENCHMARK_VERSION = "pybench-v1"
BENCHMARK_DEFAULT_MIN_TIME = 0.5
BENCHMARK_WARMUP_RUNS = 3
BENCHMARK_MEMORY_DEFAULT_MB = 256
BENCHMARK_MEMORY_AVAILABLE_FRACTION = 0.10
BENCHMARK_BASELINES = {
    "matmul_1024": 150.0,
    "elementwise_add_16m": 25.0,
    "reduction_sum_16m": 20.0,
    "conv2d_128": 750.0,
    "memory_fill": 20.0,
    "memory_read": 20.0,
}
DEFAULT_TELEMETRY_INTERVAL = 5.0
DEFAULT_PROGRESS_INTERVAL = 60.0
DEFAULT_MAX_TEMP_C = 90.0
DEFAULT_CORRECTNESS_INTERVAL = 10
GPU_HEALTH_DEFAULT_DURATION = 600.0
GPU_HEALTH_DEFAULT_MEMORY_PERCENT = 80.0
JSON_REPORT_SCHEMA_VERSION = 1
MAX_TELEMETRY_SAMPLES_PER_DEVICE = 2000
HEALTH_PASS = "PASS"
HEALTH_WARN = "WARN"
HEALTH_FAIL = "FAIL"
HARD_THROTTLE_REASONS = frozenset(
    {
        "hw_slowdown",
        "hw_thermal_slowdown",
        "hw_power_brake_slowdown",
        "sw_thermal_slowdown",
    }
)
IGNORED_THROTTLE_REASONS = frozenset({"gpu_idle", "sw_power_cap"})
NVML_THROTTLE_REASON_MAP = (
    (
        "gpu_idle",
        0x0000000000000001,
        (
            "nvmlClocksEventReasonGpuIdle",
            "nvmlClocksThrottleReasonGpuIdle",
            "NVML_CLOCKS_THROTTLE_REASON_GPU_IDLE",
            "NVML_CLOCK_THROTTLE_REASON_GPU_IDLE",
        ),
    ),
    (
        "applications_clocks_setting",
        0x0000000000000002,
        (
            "nvmlClocksEventReasonApplicationsClocksSetting",
            "nvmlClocksThrottleReasonApplicationsClocksSetting",
            "NVML_CLOCKS_THROTTLE_REASON_APPLICATIONS_CLOCKS_SETTING",
            "NVML_CLOCK_THROTTLE_REASON_APPLICATIONS_CLOCKS_SETTING",
        ),
    ),
    (
        "sw_power_cap",
        0x0000000000000004,
        (
            "nvmlClocksEventReasonSwPowerCap",
            "nvmlClocksThrottleReasonSwPowerCap",
            "NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP",
            "NVML_CLOCK_THROTTLE_REASON_SW_POWER_CAP",
        ),
    ),
    (
        "hw_slowdown",
        0x0000000000000008,
        (
            "nvmlClocksEventReasonHwSlowdown",
            "nvmlClocksThrottleReasonHwSlowdown",
            "NVML_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN",
            "NVML_CLOCK_THROTTLE_REASON_HW_SLOWDOWN",
        ),
    ),
    (
        "sync_boost",
        0x0000000000000010,
        (
            "nvmlClocksEventReasonSyncBoost",
            "nvmlClocksThrottleReasonSyncBoost",
            "NVML_CLOCKS_THROTTLE_REASON_SYNC_BOOST",
            "NVML_CLOCK_THROTTLE_REASON_SYNC_BOOST",
        ),
    ),
    (
        "sw_thermal_slowdown",
        0x0000000000000020,
        (
            "nvmlClocksEventReasonSwThermalSlowdown",
            "nvmlClocksThrottleReasonSwThermalSlowdown",
            "NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL_SLOWDOWN",
            "NVML_CLOCK_THROTTLE_REASON_SW_THERMAL_SLOWDOWN",
        ),
    ),
    (
        "hw_thermal_slowdown",
        0x0000000000000040,
        (
            "nvmlClocksEventReasonHwThermalSlowdown",
            "nvmlClocksThrottleReasonHwThermalSlowdown",
            "NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL_SLOWDOWN",
            "NVML_CLOCK_THROTTLE_REASON_HW_THERMAL_SLOWDOWN",
        ),
    ),
    (
        "hw_power_brake_slowdown",
        0x0000000000000080,
        (
            "nvmlClocksEventReasonHwPowerBrakeSlowdown",
            "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
            "NVML_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN",
            "NVML_CLOCK_THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN",
        ),
    ),
    (
        "display_clock_setting",
        0x0000000000000100,
        (
            "nvmlClocksEventReasonDisplayClockSetting",
            "nvmlClocksThrottleReasonDisplayClockSetting",
            "NVML_CLOCKS_THROTTLE_REASON_DISPLAY_CLOCK_SETTING",
            "NVML_CLOCK_THROTTLE_REASON_DISPLAY_CLOCK_SETTING",
        ),
    ),
)
REQUIRED_TORCH_ATTRIBUTES = (
    "device",
    "manual_seed",
    "randn",
    "rand",
    "empty",
    "float16",
    "float32",
    "float64",
    "nn",
)


@dataclass(frozen=True)
class BenchmarkTest:
    name: str
    category: str
    fn: Callable[[], object]
    work_units: float
    throughput_unit: str
    baseline: float


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    category: str
    device: torch.device
    median_time: float
    iqr_time: float
    throughput: float
    throughput_unit: str
    score: float


@dataclass(frozen=True)
class BenchmarkSummary:
    compute_score: float | None
    memory_score: float | None
    final_score: float | None


@dataclass(frozen=True)
class OperationSummary:
    name: str
    device: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float


@dataclass(frozen=True)
class MemoryStressResult:
    device: str
    iterations: int
    allocated_bytes: int
    total_time: float
    avg_touch_time: float
    bandwidth_gib_s: float
    stats: dict[str, int]
    failed: bool = False
    error: str | None = None


@dataclass
class MemoryStressContext:
    device: torch.device
    chunks: list
    allocated_bytes: int
    available_bytes: int | None
    total_bytes: int | None
    source: str
    failed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class HealthIssue:
    severity: str
    device: str | None
    message: str


@dataclass(frozen=True)
class TelemetrySample:
    timestamp: str
    elapsed_s: float
    device: str
    source: str
    temperature_c: float | None = None
    power_w: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    utilization_gpu_percent: float | None = None
    utilization_memory_percent: float | None = None
    sm_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None
    throttle_reasons: tuple[str, ...] = ()
    torch_memory_mb: dict[str, float] | None = None
    memory_available_mb: float | None = None
    memory_info_source: str | None = None


class HealthReport:
    def __init__(self, max_temp_c: float):
        self.max_temp_c = max_temp_c
        self.issues: list[HealthIssue] = []
        self.operation_summaries: dict[str, list[OperationSummary]] = {}
        self.memory_summaries: dict[str, list[MemoryStressResult]] = {}
        self.benchmark_summaries: dict[str, BenchmarkSummary] = {}
        self.telemetry_samples: dict[str, list[TelemetrySample]] = {}
        self.telemetry_dropped_samples: dict[str, int] = {}
        self._issue_keys: set[tuple[str, str | None, str]] = set()

    def warn(self, device: torch.device | str | None, message: str):
        self._add_issue("warning", device, message)

    def fail(self, device: torch.device | str | None, message: str):
        self._add_issue("failure", device, message)

    def _add_issue(self, severity: str, device: torch.device | str | None, message: str):
        device_name = None if device is None else str(device)
        key = (severity, device_name, message)
        if key in self._issue_keys:
            return
        self._issue_keys.add(key)
        self.issues.append(HealthIssue(severity, device_name, message))

    def add_operation_summary(self, summary: OperationSummary):
        self.operation_summaries.setdefault(summary.device, []).append(summary)

    def add_memory_summary(self, summary: MemoryStressResult):
        self.memory_summaries.setdefault(summary.device, []).append(summary)
        if summary.failed is True:
            self.fail(summary.device, f"memory stress failed: {summary.error}")

    def add_benchmark_summary(self, device: torch.device, summary: BenchmarkSummary):
        self.benchmark_summaries[str(device)] = summary
        if summary.final_score is None:
            self.warn(device, "benchmark produced no final score")

    def add_telemetry_sample(self, sample: TelemetrySample):
        samples = self.telemetry_samples.setdefault(sample.device, [])
        if len(samples) < MAX_TELEMETRY_SAMPLES_PER_DEVICE:
            samples.append(sample)
        else:
            self.telemetry_dropped_samples[sample.device] = (
                self.telemetry_dropped_samples.get(sample.device, 0) + 1
            )

        if sample.temperature_c is not None and sample.temperature_c >= self.max_temp_c:
            self.fail(
                sample.device,
                f"temperature exceeded limit {self.max_temp_c:.1f}C",
            )

        hard_reasons = [
            reason
            for reason in sample.throttle_reasons
            if reason in HARD_THROTTLE_REASONS
        ]
        if hard_reasons:
            self.fail(
                sample.device,
                f"NVML hard throttle detected: {', '.join(sorted(hard_reasons))}",
            )
            return

        soft_reasons = [
            reason
            for reason in sample.throttle_reasons
            if reason not in IGNORED_THROTTLE_REASONS
        ]
        if soft_reasons:
            self.warn(
                sample.device,
                f"NVML throttle detected: {', '.join(sorted(soft_reasons))}",
            )

    @property
    def failures(self):
        return [issue for issue in self.issues if issue.severity == "failure"]

    @property
    def warnings(self):
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def status(self):
        if self.failures:
            return HEALTH_FAIL
        if self.warnings:
            return HEALTH_WARN
        return HEALTH_PASS

    def device_status(self, device: torch.device | str):
        device_name = str(device)
        failures = [
            issue
            for issue in self.failures
            if issue.device in (None, device_name)
        ]
        if failures:
            return HEALTH_FAIL
        warnings = [
            issue
            for issue in self.warnings
            if issue.device in (None, device_name)
        ]
        if warnings:
            return HEALTH_WARN
        return HEALTH_PASS

    def telemetry_summary(self, device: torch.device | str):
        device_name = str(device)
        samples = self.telemetry_samples.get(device_name, [])
        if not samples:
            return {
                "sample_count": 0,
                "dropped_samples": self.telemetry_dropped_samples.get(device_name, 0),
            }

        def max_present(attr):
            values = [
                getattr(sample, attr)
                for sample in samples
                if getattr(sample, attr) is not None
            ]
            return max(values) if values else None

        throttle_reasons = sorted(
            {
                reason
                for sample in samples
                for reason in sample.throttle_reasons
            }
        )
        return {
            "sample_count": len(samples),
            "dropped_samples": self.telemetry_dropped_samples.get(device_name, 0),
            "max_temperature_c": max_present("temperature_c"),
            "max_power_w": max_present("power_w"),
            "max_memory_used_mb": max_present("memory_used_mb"),
            "max_gpu_utilization_percent": max_present(
                "utilization_gpu_percent"
            ),
            "max_memory_utilization_percent": max_present(
                "utilization_memory_percent"
            ),
            "max_sm_clock_mhz": max_present("sm_clock_mhz"),
            "max_memory_clock_mhz": max_present("memory_clock_mhz"),
            "throttle_reasons": throttle_reasons,
        }


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
    logger = logging.getLogger("bench")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = StreamHandler()
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
    return logger


def validate_torch_installation():
    missing = [
        f"torch.{name}" for name in REQUIRED_TORCH_ATTRIBUTES if not hasattr(torch, name)
    ]
    if missing:
        raise click.ClickException(
            "PyTorch import is incomplete; activate an environment with a full "
            "PyTorch installation "
            f"(missing required attributes: {', '.join(missing)})."
        )


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def option_source_is_default(name: str):
    context = click.get_current_context(silent=True)
    if context is None:
        return True
    get_parameter_source = getattr(context, "get_parameter_source", None)
    if not callable(get_parameter_source):
        return True
    return get_parameter_source(name) == ParameterSource.DEFAULT


def apply_preset_defaults(
    preset,
    mode,
    suite,
    device,
    memory_percent,
    memory_mb,
    duration,
    telemetry,
    correctness,
):
    if preset is None:
        return (
            mode,
            suite,
            device,
            memory_percent,
            duration,
            telemetry,
            correctness,
        )
    if preset != "gpu-health":
        raise click.ClickException(f"Unsupported preset: {preset}")

    if option_source_is_default("mode"):
        mode = "stress"
    if option_source_is_default("suite"):
        suite = "full"
    if option_source_is_default("device"):
        device = "cuda"
    if (
        option_source_is_default("memory_percent")
        and option_source_is_default("memory_mb")
        and memory_mb is None
    ):
        memory_percent = GPU_HEALTH_DEFAULT_MEMORY_PERCENT
    if option_source_is_default("duration"):
        duration = GPU_HEALTH_DEFAULT_DURATION
    if option_source_is_default("telemetry"):
        telemetry = True
    if option_source_is_default("correctness"):
        correctness = "sampled"

    return mode, suite, device, memory_percent, duration, telemetry, correctness


def safe_logger_error(logger, message):
    error = getattr(logger, "error", None)
    if callable(error):
        error(message)
    else:
        logger.warning(message)


def is_cuda_available():
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def get_mps_backend():
    backends = getattr(torch, "backends", None)
    return getattr(backends, "mps", None)


def is_mps_available():
    mps_backend = get_mps_backend()
    is_available = getattr(mps_backend, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def get_devices():
    devices = [torch.device("cpu")]
    cuda = getattr(torch, "cuda", None)
    device_count = getattr(cuda, "device_count", None)
    if is_cuda_available() and callable(device_count):
        for i in range(device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    if is_mps_available():
        devices.append(torch.device("mps"))
    return devices


def device_matches_filter(device: torch.device, device_filter: str):
    normalized_filter = device_filter.strip().lower()
    device_type = getattr(device, "type", "").lower()
    device_spec = str(device).lower()
    return normalized_filter == device_type or normalized_filter == device_spec


def filter_devices(devices, device_filter):
    if device_filter is None or not device_filter.strip():
        return list(devices)

    matched_devices = [
        device for device in devices if device_matches_filter(device, device_filter)
    ]
    if not matched_devices:
        available_devices = ", ".join(str(device) for device in devices) or "none"
        raise click.ClickException(
            f"No detected devices match --device {device_filter!r}. "
            f"Available devices: {available_devices}."
        )
    return matched_devices


def log_environment_info(logger):
    logger.info(f"PyTorch version: {getattr(torch, '__version__', 'unknown')}")
    cuda_available = is_cuda_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        log_cuda_devices(logger)
    logger.info(f"MPS available: {is_mps_available()}")


def log_cuda_devices(logger):
    cuda = getattr(torch, "cuda", None)
    device_count = cuda.device_count()
    logger.info(f"CUDA devices: {device_count}")
    for idx in range(device_count):
        props = cuda.get_device_properties(idx)
        total_memory_mb = props.total_memory / (1024**2)
        logger.info(
            f"CUDA device {idx}: {props.name}, memory={total_memory_mb:.1f} MB, "
            f"multiprocessors={props.multi_processor_count}"
        )


def sync(device: torch.device):
    # Ensure operations are finished on GPU/MPS
    if device.type == "cuda":
        cuda = getattr(torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        synchronize = getattr(mps, "synchronize", None)
        if callable(synchronize):
            synchronize()


def inference_context():
    inference_mode = getattr(torch, "inference_mode", None)
    if inference_mode is None:
        return nullcontext()
    return inference_mode()


def get_skip_reason(device: torch.device, dt: torch.dtype):
    if dt == torch.float16 and device.type == "cpu":
        return f"Skipping FP16 benchmark on CPU ({device}): unsupported efficiently."
    if dt == torch.float64 and device.type == "mps":
        return f"Skipping FP64 benchmark on MPS ({device}): unsupported."
    return None


def get_memory_dtype(device: torch.device, dt: torch.dtype, logger):
    if dt == torch.float64 and device.type == "mps":
        logger.warning(
            f"[{device}] Using float32 for memory stress because MPS does not support FP64."
        )
        return torch.float32
    return dt


def bytes_to_mb(size_bytes):
    return size_bytes / (1024**2)


def bytes_to_gib(size_bytes):
    return size_bytes / (1024**3)


def ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def get_dtype_size(dt):
    if dt == torch.float16:
        return 2
    if dt == torch.float64:
        return 8
    return 4


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def get_torch_benchmark_timer():
    try:
        benchmark_module = importlib.import_module("torch.utils.benchmark")
    except ImportError:
        benchmark_module = getattr(getattr(torch, "utils", None), "benchmark", None)
    return getattr(benchmark_module, "Timer", None)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def calculate_iqr(values):
    if len(values) < 4:
        return 0.0
    return percentile(values, 0.75) - percentile(values, 0.25)


def geometric_mean(values):
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return None
    return math.exp(
        sum(math.log(value) for value in positive_values) / len(positive_values)
    )


def format_optional_score(score):
    if score is None:
        return "n/a"
    return f"{score:.1f}"


def iter_tensor_results(value):
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_tensor_results(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensor_results(item)
        return
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        yield value


def tensor_is_finite(tensor):
    isfinite = getattr(torch, "isfinite", None)
    if not callable(isfinite):
        return True
    try:
        finite = isfinite(tensor)
    except (RuntimeError, TypeError):
        return True
    all_fn = getattr(finite, "all", None)
    if callable(all_fn):
        finite = all_fn()
    item = getattr(finite, "item", None)
    if callable(item):
        return bool(item())
    return bool(finite)


def validate_result_is_finite(name, result, device):
    checked = False
    for tensor in iter_tensor_results(result):
        checked = True
        if not tensor_is_finite(tensor):
            raise RuntimeError(f"{name} produced NaN or Inf on {device}")
    return checked


def should_validate_iteration(correctness, iteration_index, correctness_interval):
    if correctness == "strict":
        return True
    if correctness == "sampled":
        return iteration_index % correctness_interval == 0
    return False


def should_validate_warmup(correctness, warmup_index):
    if correctness == "strict":
        return True
    return correctness in ("smoke", "sampled") and warmup_index == 0


def should_validate_memory_iteration(correctness, iteration_index, correctness_interval):
    return should_validate_iteration(
        correctness,
        iteration_index,
        correctness_interval,
    ) or should_validate_warmup(correctness, iteration_index)


def scalar_to_float(value):
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (RuntimeError, TypeError, ValueError):
            return None
    try:
        return float(value)
    except (RuntimeError, TypeError, ValueError):
        return None


def tensor_sum_for_validation(tensor, dt):
    sum_fn = getattr(tensor, "sum", None)
    if not callable(sum_fn):
        return None
    try:
        if dt == torch.float16:
            return sum_fn(dtype=torch.float32)
        return sum_fn()
    except TypeError:
        return sum_fn()


def validate_memory_sum(name, tensor, fill_value, dt, device, observed_sum=None):
    value = (
        observed_sum
        if observed_sum is not None
        else tensor_sum_for_validation(tensor, dt)
    )
    actual = scalar_to_float(value)
    if actual is None:
        return
    expected = tensor.numel() * fill_value
    tolerance = max(abs(expected) * 1e-3, 1e-2)
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"{name} memory validation failed on {device}: "
            f"expected {expected:.6g}, got {actual:.6g}"
        )


def measure_benchmark_test(benchmark_test, device, min_run_time):
    timer_class = get_torch_benchmark_timer()
    if callable(timer_class):
        with inference_context():
            sync(device)

            def timed_fn():
                result = benchmark_test.fn()
                sync(device)
                return result

            timer = timer_class(
                stmt="timed_fn()",
                globals={"timed_fn": timed_fn},
            )
            measurement = timer.blocked_autorange(min_run_time=min_run_time)
            sync(device)
        return float(measurement.median), float(getattr(measurement, "iqr", 0.0))

    return measure_benchmark_test_fallback(benchmark_test, device, min_run_time)


def measure_benchmark_test_fallback(benchmark_test, device, min_run_time):
    times = []
    with inference_context():
        for _ in range(BENCHMARK_WARMUP_RUNS):
            benchmark_test.fn()
            sync(device)

        started_at = time.perf_counter()
        while len(times) < 3 or time.perf_counter() - started_at < min_run_time:
            iteration_started_at = time.perf_counter()
            benchmark_test.fn()
            sync(device)
            times.append(time.perf_counter() - iteration_started_at)

    return statistics.median(times), calculate_iqr(times)


def make_benchmark_result(benchmark_test, device, median_time, iqr_time):
    throughput = benchmark_test.work_units / median_time if median_time > 0 else 0.0
    score = (
        1000.0 * throughput / benchmark_test.baseline
        if benchmark_test.baseline > 0
        else 0.0
    )
    return BenchmarkResult(
        name=benchmark_test.name,
        category=benchmark_test.category,
        device=device,
        median_time=median_time,
        iqr_time=iqr_time,
        throughput=throughput,
        throughput_unit=benchmark_test.throughput_unit,
        score=score,
    )


def log_benchmark_result(logger, result):
    logger.info(
        f"[{result.device}] benchmark {result.name}: "
        f"median={result.median_time:.6f}s, iqr={result.iqr_time:.6f}s, "
        f"throughput={result.throughput:.3f} {result.throughput_unit}, "
        f"score={result.score:.1f}"
    )


def run_benchmark_tests(
    benchmark_tests,
    device,
    min_run_time,
    logger,
    progress=True,
):
    results = []
    for benchmark_test in benchmark_tests:
        try:
            if progress:
                logger.info(f"[{device}] benchmark {benchmark_test.name}: started")
            median_time, iqr_time = measure_benchmark_test(
                benchmark_test,
                device,
                min_run_time,
            )
            result = make_benchmark_result(
                benchmark_test,
                device,
                median_time,
                iqr_time,
            )
        except Exception as exc:
            logger.warning(f"[{device}] Skipping benchmark {benchmark_test.name}: {exc}")
            continue

        log_benchmark_result(logger, result)
        if progress:
            logger.info(f"[{device}] benchmark {benchmark_test.name}: finished")
        results.append(result)
    return results


def summarize_benchmark_scores(compute_results, memory_results):
    compute_score = geometric_mean(result.score for result in compute_results)
    memory_score = geometric_mean(result.score for result in memory_results)

    final_inputs = []
    if compute_score is not None:
        final_inputs.append(compute_score)
    if memory_score is not None:
        final_inputs.append(memory_score)
    final_score = geometric_mean(final_inputs)

    return BenchmarkSummary(
        compute_score=compute_score,
        memory_score=memory_score,
        final_score=final_score,
    )


def log_benchmark_summary(logger, device, summary):
    logger.info(
        f"[{device}] {BENCHMARK_VERSION} score: "
        f"compute={format_optional_score(summary.compute_score)}, "
        f"memory={format_optional_score(summary.memory_score)}, "
        f"final={format_optional_score(summary.final_score)}"
    )


def build_benchmark_compute_tests(device, dt):
    matmul_size = 1024
    vector_elements = 4096 * 4096
    conv_batch_size = 8
    conv_image_size = 128
    dtype_size = get_dtype_size(dt)

    matmul_a = torch.randn((matmul_size, matmul_size), device=device, dtype=dt)
    matmul_b = torch.randn((matmul_size, matmul_size), device=device, dtype=dt)
    add_a = torch.randn((vector_elements,), device=device, dtype=dt)
    add_b = torch.randn((vector_elements,), device=device, dtype=dt)
    reduction_input = torch.randn((vector_elements,), device=device, dtype=dt)
    image_batch = torch.randn(
        (conv_batch_size, 3, conv_image_size, conv_image_size),
        device=device,
        dtype=dt,
    )
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(
        device=device,
        dtype=dt,
    )

    matmul_gflop = (2 * matmul_size**3) / 1e9
    elementwise_gib = bytes_to_gib(3 * vector_elements * dtype_size)
    reduction_gib = bytes_to_gib(vector_elements * dtype_size)

    return [
        BenchmarkTest(
            name="matmul_1024",
            category="compute",
            fn=lambda: matmul_a @ matmul_b,
            work_units=matmul_gflop,
            throughput_unit="GFLOP/s",
            baseline=BENCHMARK_BASELINES["matmul_1024"],
        ),
        BenchmarkTest(
            name="elementwise_add_16m",
            category="compute",
            fn=lambda: add_a + add_b,
            work_units=elementwise_gib,
            throughput_unit="GiB/s",
            baseline=BENCHMARK_BASELINES["elementwise_add_16m"],
        ),
        BenchmarkTest(
            name="reduction_sum_16m",
            category="compute",
            fn=lambda: reduction_input.sum(),
            work_units=reduction_gib,
            throughput_unit="GiB/s",
            baseline=BENCHMARK_BASELINES["reduction_sum_16m"],
        ),
        BenchmarkTest(
            name="conv2d_128",
            category="compute",
            fn=lambda: conv(image_batch),
            work_units=float(conv_batch_size),
            throughput_unit="images/s",
            baseline=BENCHMARK_BASELINES["conv2d_128"],
        ),
    ]


def calculate_benchmark_memory_target_bytes(device, memory_mb):
    available_bytes, total_bytes, source = get_device_memory_info(device)
    if memory_mb is not None:
        return memory_mb * 1024**2, available_bytes, total_bytes, source

    default_bytes = BENCHMARK_MEMORY_DEFAULT_MB * 1024**2
    if available_bytes is None or available_bytes <= 0:
        return default_bytes, available_bytes, total_bytes, source

    target_bytes = min(
        default_bytes,
        int(available_bytes * BENCHMARK_MEMORY_AVAILABLE_FRACTION),
    )
    return max(target_bytes, 1), available_bytes, total_bytes, source


def build_benchmark_memory_tests(device, dt, memory_mb, logger):
    target_bytes, available_bytes, total_bytes, source = (
        calculate_benchmark_memory_target_bytes(device, memory_mb)
    )
    logger.info(
        f"[{device}] benchmark memory target={bytes_to_mb(target_bytes):.1f} MB "
        f"(source={source})"
    )
    if available_bytes is not None:
        total_text = "unknown" if total_bytes is None else f"{bytes_to_mb(total_bytes):.1f} MB"
        logger.info(
            f"[{device}] benchmark memory available="
            f"{bytes_to_mb(available_bytes):.1f} MB, total={total_text}"
        )

    chunks, allocated_bytes = allocate_memory_chunks(target_bytes, device, dt)
    work_units = bytes_to_gib(allocated_bytes)

    def fill_chunks():
        for chunk in chunks:
            chunk.fill_(1.0)

    def read_chunks():
        total = None
        for chunk in chunks:
            value = chunk.sum()
            total = value if total is None else total + value
        return total

    return [
        BenchmarkTest(
            name="memory_fill",
            category="memory",
            fn=fill_chunks,
            work_units=work_units,
            throughput_unit="GiB/s",
            baseline=BENCHMARK_BASELINES["memory_fill"],
        ),
        BenchmarkTest(
            name="memory_read",
            category="memory",
            fn=read_chunks,
            work_units=work_units,
            throughput_unit="GiB/s",
            baseline=BENCHMARK_BASELINES["memory_read"],
        ),
    ], chunks


def run_benchmark_mode(
    devices,
    dt,
    benchmark_memory,
    memory_mb,
    benchmark_min_time,
    logger,
    progress=True,
):
    summaries = {}
    for device in devices:
        logger.info(f"\nBenchmarking score on device: {device} ({BENCHMARK_VERSION})")
        skip_reason = get_skip_reason(device, dt)
        if skip_reason:
            logger.warning(skip_reason)
            continue

        try:
            compute_tests = build_benchmark_compute_tests(device, dt)
            compute_results = run_benchmark_tests(
                compute_tests,
                device,
                benchmark_min_time,
                logger,
                progress=progress,
            )
        except Exception as exc:
            logger.warning(f"[{device}] Benchmark compute failed: {exc}")
            compute_results = []

        memory_results = []
        memory_chunks = []
        if benchmark_memory:
            try:
                memory_dt = get_memory_dtype(device, dt, logger)
                memory_tests, memory_chunks = build_benchmark_memory_tests(
                    device,
                    memory_dt,
                    memory_mb,
                    logger,
                )
                memory_results = run_benchmark_tests(
                    memory_tests,
                    device,
                    benchmark_min_time,
                    logger,
                    progress=progress,
                )
            except (RuntimeError, MemoryError, TypeError) as exc:
                logger.warning(f"[{device}] Skipping benchmark memory: {exc}")
            finally:
                if memory_chunks:
                    release_memory_chunks(memory_chunks, device)

        summary = summarize_benchmark_scores(compute_results, memory_results)
        summaries[device] = summary
        log_benchmark_summary(logger, device, summary)

    return summaries


def make_operation_summary(name, device, times):
    total_time = sum(times)
    avg_time = statistics.mean(times) if times else 0.0
    min_time = min(times) if times else 0.0
    max_time = max(times) if times else 0.0
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return OperationSummary(
        name=name,
        device=str(device),
        iterations=len(times),
        total_time=total_time,
        avg_time=avg_time,
        min_time=min_time,
        max_time=max_time,
        std_time=std_time,
    )


def log_operation_summary(logger, summary):
    logger.info(
        f"[{summary.device}] {summary.name}: total={summary.total_time:.6f}s, "
        f"avg={summary.avg_time:.6f}s, min={summary.min_time:.6f}s, "
        f"max={summary.max_time:.6f}s, std={summary.std_time:.6f}s"
    )


def format_duration(seconds):
    if 0 < seconds < 1:
        return "<1s"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class ProgressReporter:
    def __init__(
        self,
        logger,
        enabled=True,
        interval=DEFAULT_PROGRESS_INTERVAL,
        clock=time.perf_counter,
    ):
        self.logger = logger
        self.enabled = enabled
        self.interval = max(float(interval), 1.0)
        self.clock = clock
        self.stage_name = None
        self.total_seconds = None
        self.started_at = None
        self.next_log_at = None

    def start(self, stage_name, total_seconds=None):
        self.stage_name = stage_name
        self.total_seconds = total_seconds
        self.started_at = self.clock()
        self.next_log_at = self.started_at + self.interval
        if not self.enabled:
            return
        if total_seconds is None:
            self.logger.info(f"{stage_name}: started")
        else:
            self.logger.info(
                f"{stage_name}: started, expected={format_duration(total_seconds)}"
            )

    def maybe_log(self):
        if not self.enabled or self.started_at is None:
            return
        now = self.clock()
        if now < self.next_log_at:
            return
        while self.next_log_at <= now:
            self.next_log_at += self.interval
        elapsed = now - self.started_at
        if self.total_seconds is None:
            self.logger.info(
                f"{self.stage_name}: elapsed={format_duration(elapsed)}"
            )
            return
        remaining = max(self.total_seconds - elapsed, 0.0)
        percent = min(100.0, (elapsed / self.total_seconds) * 100.0)
        self.logger.info(
            f"{self.stage_name}: elapsed={format_duration(elapsed)}, "
            f"remaining={format_duration(remaining)}, progress={percent:.0f}%"
        )

    def finish(self):
        if not self.enabled or self.started_at is None:
            return
        elapsed = self.clock() - self.started_at
        self.logger.info(
            f"{self.stage_name}: finished in {format_duration(elapsed)}"
        )


def benchmark_op(
    name,
    fn,
    device,
    iterations,
    logger,
    correctness="off",
    correctness_interval=DEFAULT_CORRECTNESS_INTERVAL,
    telemetry_monitor=None,
    progress=True,
):
    with inference_context():
        for warmup_index in range(5):
            result = fn()
            if should_validate_warmup(correctness, warmup_index):
                validate_result_is_finite(name, result, device)
        sync(device)

        times = []
        for iteration_index in tqdm(
            range(iterations),
            desc=f"{name} on {device}",
            leave=False,
            disable=not progress,
        ):
            start = time.perf_counter()
            result = fn()
            if should_validate_iteration(
                correctness,
                iteration_index,
                correctness_interval,
            ):
                validate_result_is_finite(name, result, device)
            sync(device)
            end = time.perf_counter()
            times.append(end - start)
            if telemetry_monitor is not None:
                telemetry_monitor.sample(device)

    summary = make_operation_summary(name, device, times)
    log_operation_summary(logger, summary)
    return summary


def benchmark_op_once(
    name,
    fn,
    device,
    iteration_index,
    correctness,
    correctness_interval,
):
    start = time.perf_counter()
    result = fn()
    if should_validate_iteration(correctness, iteration_index, correctness_interval):
        validate_result_is_finite(name, result, device)
    sync(device)
    return time.perf_counter() - start


def build_basic_ops(size, device, dt):
    a = torch.randn((size, size), device=device, dtype=dt)
    b = torch.randn((size, size), device=device, dtype=dt)

    return [
        ("matmul", lambda: a @ b),
        ("add", lambda: a + b),
        ("mul", lambda: a * b),
        ("sum", lambda: a.sum()),
    ]


def build_extended_ops(size, device, dt):
    image_size = min(size, 224)
    batch_size = 8

    a = torch.randn((size, size), device=device, dtype=dt)
    vec = torch.randn((size,), device=device, dtype=dt)
    image_batch = torch.randn(
        (batch_size, 3, image_size, image_size),
        device=device,
        dtype=dt,
    )
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device=device, dtype=dt)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_size * image_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    ).to(device=device, dtype=dt)

    return [
        ("dot", lambda: torch.dot(vec, vec)),
        ("transpose", lambda: a.t()),
        ("relu", lambda: torch.relu(a)),
        ("rand", lambda: torch.rand((size, size), device=device, dtype=dt)),
        ("conv2d", lambda: conv(image_batch)),
        ("model_forward", lambda: model(image_batch)),
    ]


def get_operations(suite, size, device, dt):
    ops = []
    if suite in ("basic", "all", "full"):
        ops.extend(build_basic_ops(size, device, dt))
    if suite in ("extended", "all", "full"):
        ops.extend(build_extended_ops(size, device, dt))
    return ops


def run_operations(
    ops,
    device,
    iterations,
    logger,
    skip_failed_ops=False,
    correctness="off",
    correctness_interval=DEFAULT_CORRECTNESS_INTERVAL,
    telemetry_monitor=None,
    health=None,
    progress=True,
):
    summaries = []
    for name, fn in ops:
        try:
            summary = benchmark_op(
                name,
                fn,
                device,
                iterations,
                logger,
                correctness=correctness,
                correctness_interval=correctness_interval,
                telemetry_monitor=telemetry_monitor,
                progress=progress,
            )
        except Exception as exc:
            if health is not None:
                health.fail(device, f"operation {name} failed: {exc}")
            if not skip_failed_ops:
                raise
            logger.warning(f"[{device}] Skipping {name}: {exc}")
            continue
        summaries.append(summary)
        if health is not None:
            health.add_operation_summary(summary)
    return summaries


def warm_up_operations(
    ops,
    device,
    correctness,
    health=None,
    logger=None,
    skip_failed_ops=False,
):
    failed_ops = set()
    with inference_context():
        for name, fn in ops:
            try:
                for warmup_index in range(5):
                    result = fn()
                    if should_validate_warmup(correctness, warmup_index):
                        validate_result_is_finite(name, result, device)
            except Exception as exc:
                if health is not None:
                    health.fail(device, f"operation {name} failed during warmup: {exc}")
                if logger is not None:
                    logger.warning(f"[{device}] Skipping {name}: {exc}")
                if not skip_failed_ops:
                    raise
                failed_ops.add(name)
        sync(device)
    return failed_ops


def run_operations_for_duration(
    ops,
    device,
    deadline,
    logger,
    correctness="off",
    correctness_interval=DEFAULT_CORRECTNESS_INTERVAL,
    telemetry_monitor=None,
    health=None,
    memory_context=None,
    memory_dt=None,
    memory_times=None,
    duration=None,
    progress=True,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
):
    if not ops:
        return []

    times_by_name = {name: [] for name, _ in ops}
    iteration_by_name = {name: 0 for name, _ in ops}
    failed_ops = set(
        warm_up_operations(
            ops,
            device,
            correctness,
            health=health,
            logger=logger,
            skip_failed_ops=True,
        ) or ()
    )

    progress_reporter = ProgressReporter(
        logger,
        enabled=progress,
        interval=progress_interval,
    )
    progress_reporter.start(f"[{device}] duration stress", duration)

    with inference_context():
        while time.perf_counter() < deadline:
            cycle_completed = False
            for name, fn in ops:
                if time.perf_counter() >= deadline:
                    break
                if name in failed_ops:
                    continue
                try:
                    elapsed = benchmark_op_once(
                        name,
                        fn,
                        device,
                        iteration_by_name[name],
                        correctness,
                        correctness_interval,
                    )
                except Exception as exc:
                    failed_ops.add(name)
                    if health is not None:
                        health.fail(device, f"operation {name} failed: {exc}")
                    logger.warning(f"[{device}] Skipping {name}: {exc}")
                    continue
                times_by_name[name].append(elapsed)
                iteration_by_name[name] += 1
                cycle_completed = True
                if telemetry_monitor is not None:
                    telemetry_monitor.sample(device)
                progress_reporter.maybe_log()
            if (
                memory_context is not None
                and memory_times is not None
                and not memory_context.failed
                and cycle_completed
            ):
                try:
                    memory_iteration = len(memory_times)
                    memory_times.append(
                        touch_memory_chunks_once(
                            memory_context.chunks,
                            memory_context.device,
                            memory_dt,
                            memory_iteration,
                            correctness,
                            correctness_interval,
                        )
                    )
                except (RuntimeError, MemoryError, TypeError) as exc:
                    record_memory_context_failure(
                        memory_context,
                        memory_times,
                        exc,
                        logger,
                        health,
                    )
                else:
                    if telemetry_monitor is not None:
                        telemetry_monitor.sample(device)
                    progress_reporter.maybe_log()
            if failed_ops and len(failed_ops) == len(ops):
                break
    progress_reporter.finish()

    summaries = []
    for name, _ in ops:
        summary = make_operation_summary(name, device, times_by_name[name])
        log_operation_summary(logger, summary)
        summaries.append(summary)
        if health is not None:
            health.add_operation_summary(summary)
    return summaries


def get_cpu_memory_info():
    mem_available = None
    mem_total = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                key, value = line.split(":", 1)
                if key in ("MemAvailable", "MemTotal"):
                    kib = int(value.strip().split()[0])
                    if key == "MemAvailable":
                        mem_available = kib * 1024
                    else:
                        mem_total = kib * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass

    if mem_available is not None:
        return mem_available, mem_total, "/proc/meminfo"

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None, None, "unavailable"

    available = page_size * available_pages
    total = page_size * total_pages
    return available, total, "os.sysconf"


def get_device_memory_info(device):
    if device.type == "cpu":
        return get_cpu_memory_info()
    if device.type == "cuda":
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return None, None, "torch.cuda unavailable"
        mem_get_info = getattr(cuda, "mem_get_info", None)
        if callable(mem_get_info):
            free_bytes, total_bytes = mem_get_info(device)
            return free_bytes, total_bytes, "torch.cuda.mem_get_info"
        return estimate_cuda_memory_info(device)
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        recommended_max_memory = getattr(mps, "recommended_max_memory", None)
        if not callable(recommended_max_memory):
            return None, None, "torch.mps.recommended_max_memory unavailable"
        total_bytes = recommended_max_memory()
        driver_allocated_memory = getattr(mps, "driver_allocated_memory", None)
        current_allocated_memory = getattr(mps, "current_allocated_memory", None)
        used_bytes = 0
        if callable(driver_allocated_memory):
            used_bytes = driver_allocated_memory()
        elif callable(current_allocated_memory):
            used_bytes = current_allocated_memory()
        return max(total_bytes - used_bytes, 0), total_bytes, "torch.mps"
    return None, None, "unsupported"


def estimate_cuda_memory_info(device):
    cuda = getattr(torch, "cuda", None)
    get_device_properties = getattr(cuda, "get_device_properties", None)
    if not callable(get_device_properties):
        return None, None, "torch.cuda.mem_get_info unavailable"

    props = get_device_properties(device)
    total_bytes = getattr(props, "total_memory", None)
    if total_bytes is None:
        return None, None, "torch.cuda.total_memory unavailable"

    memory_reserved = getattr(cuda, "memory_reserved", None)
    memory_allocated = getattr(cuda, "memory_allocated", None)
    if callable(memory_reserved):
        used_bytes = memory_reserved(device)
    elif callable(memory_allocated):
        used_bytes = memory_allocated(device)
    else:
        used_bytes = 0
    return max(total_bytes - used_bytes, 0), total_bytes, "torch.cuda memory estimate"


def calculate_memory_target_bytes(device, memory_percent, memory_mb):
    available_bytes, total_bytes, source = get_device_memory_info(device)
    if memory_mb is not None:
        return memory_mb * 1024**2, available_bytes, total_bytes, source
    if available_bytes is None:
        raise RuntimeError(f"Could not determine available memory for {device}.")
    if available_bytes <= 0:
        raise RuntimeError(
            f"No available memory detected for {device} (source={source})."
        )
    target_bytes = int(available_bytes * (memory_percent / 100.0))
    if target_bytes <= 0:
        raise RuntimeError(
            f"No available memory target for {device} (source={source})."
        )
    return target_bytes, available_bytes, total_bytes, source


def reset_peak_memory_stats(device):
    if device.type != "cuda":
        return
    cuda = getattr(torch, "cuda", None)
    reset_peak = getattr(cuda, "reset_peak_memory_stats", None)
    if callable(reset_peak):
        reset_peak(device)


def get_memory_stats(device):
    stats = {}
    if device.type == "cuda":
        cuda = getattr(torch, "cuda", None)
        for name in (
            "memory_allocated",
            "max_memory_allocated",
            "memory_reserved",
            "max_memory_reserved",
        ):
            fn = getattr(cuda, name, None)
            if callable(fn):
                stats[name] = fn(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        for name in ("current_allocated_memory", "driver_allocated_memory"):
            fn = getattr(mps, name, None)
            if callable(fn):
                stats[name] = fn()
    return stats


def safe_get_memory_stats(device, health=None):
    try:
        return get_memory_stats(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        if health is not None:
            health.warn(device, f"PyTorch memory stats unavailable: {exc}")
        return {}


def safe_get_device_memory_info(device, health=None):
    try:
        return get_device_memory_info(device)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        if health is not None:
            health.warn(device, f"PyTorch memory info unavailable: {exc}")
        return None, None, "unavailable"


def format_torch_memory_mb(stats):
    return {name: bytes_to_mb(value) for name, value in stats.items()}


def get_cuda_device_index(device):
    spec = str(device)
    if ":" not in spec:
        current_device = getattr(getattr(torch, "cuda", None), "current_device", None)
        return int(current_device()) if callable(current_device) else 0
    return int(spec.split(":", 1)[1])


def get_cuda_pci_bus_id(device):
    cuda = getattr(torch, "cuda", None)
    get_device_properties = getattr(cuda, "get_device_properties", None)
    if not callable(get_device_properties):
        return None
    try:
        props = get_device_properties(device)
    except (RuntimeError, TypeError, ValueError):
        return None
    pci_bus_id = getattr(props, "pci_bus_id", None)
    if not pci_bus_id:
        return None
    return str(pci_bus_id)


def decode_nvml_throttle_reasons(nvml, reason_bits):
    if not reason_bits:
        return ()
    names = []
    known_bits = 0
    for name, fallback_value, attrs in NVML_THROTTLE_REASON_MAP:
        values = {fallback_value}
        values.update(
            value
            for attr in attrs
            if isinstance((value := getattr(nvml, attr, 0)), int) and value
        )
        for value in values:
            known_bits |= value
            if reason_bits & value:
                names.append(name)
                break
    unknown_bits = reason_bits & ~known_bits
    if unknown_bits:
        names.append(f"unknown_0x{unknown_bits:x}")
    return tuple(names)


class NvmlTelemetryBackend:
    def __init__(self, nvml):
        self.nvml = nvml
        self.handles: dict[str, Any] = {}

    @classmethod
    def create(cls, logger, health):
        try:
            nvml = importlib.import_module("pynvml")
        except ImportError:
            health.warn(None, "NVML telemetry unavailable: install nvidia-ml-py")
            return None
        try:
            nvml.nvmlInit()
        except Exception as exc:
            health.warn(None, f"NVML telemetry unavailable: {exc}")
            return None
        logger.info("NVML telemetry enabled")
        return cls(nvml)

    def close(self):
        shutdown = getattr(self.nvml, "nvmlShutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

    def get_handle(self, device):
        device_key = str(device)
        if device_key in self.handles:
            return self.handles[device_key]

        pci_bus_id = get_cuda_pci_bus_id(device)
        get_handle_by_pci = getattr(self.nvml, "nvmlDeviceGetHandleByPciBusId", None)
        if pci_bus_id and callable(get_handle_by_pci):
            try:
                self.handles[device_key] = get_handle_by_pci(pci_bus_id)
                return self.handles[device_key]
            except TypeError:
                self.handles[device_key] = get_handle_by_pci(pci_bus_id.encode("ascii"))
                return self.handles[device_key]
            except Exception:
                pass

        index = get_cuda_device_index(device)
        self.handles[device_key] = self.nvml.nvmlDeviceGetHandleByIndex(index)
        return self.handles[device_key]

    def sample(self, device):
        handle = self.get_handle(device)
        nvml = self.nvml

        def call(name, *args):
            fn = getattr(nvml, name, None)
            if not callable(fn):
                return None
            return fn(handle, *args)

        def call_first_supported(*names):
            last_exc = None
            for name in names:
                fn = getattr(nvml, name, None)
                if not callable(fn):
                    continue
                try:
                    value = fn(handle)
                except Exception as exc:
                    last_exc = exc
                    continue
                if value is not None:
                    return value
            if last_exc is not None:
                raise last_exc
            return None

        temperature_const = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
        sm_clock_const = getattr(nvml, "NVML_CLOCK_SM", 0)
        memory_clock_const = getattr(nvml, "NVML_CLOCK_MEM", 0)

        temperature_c = call("nvmlDeviceGetTemperature", temperature_const)
        power_mw = call("nvmlDeviceGetPowerUsage")
        memory_info = call("nvmlDeviceGetMemoryInfo")
        utilization = call("nvmlDeviceGetUtilizationRates")
        sm_clock_mhz = call("nvmlDeviceGetClockInfo", sm_clock_const)
        memory_clock_mhz = call("nvmlDeviceGetClockInfo", memory_clock_const)
        throttle_bits = call_first_supported(
            "nvmlDeviceGetCurrentClocksEventReasons",
            "nvmlDeviceGetCurrentClocksThrottleReasons",
        )

        return {
            "temperature_c": float(temperature_c)
            if temperature_c is not None
            else None,
            "power_w": float(power_mw) / 1000.0 if power_mw is not None else None,
            "memory_used_mb": bytes_to_mb(memory_info.used)
            if memory_info is not None and hasattr(memory_info, "used")
            else None,
            "memory_total_mb": bytes_to_mb(memory_info.total)
            if memory_info is not None and hasattr(memory_info, "total")
            else None,
            "utilization_gpu_percent": float(utilization.gpu)
            if utilization is not None and hasattr(utilization, "gpu")
            else None,
            "utilization_memory_percent": float(utilization.memory)
            if utilization is not None and hasattr(utilization, "memory")
            else None,
            "sm_clock_mhz": float(sm_clock_mhz)
            if sm_clock_mhz is not None
            else None,
            "memory_clock_mhz": float(memory_clock_mhz)
            if memory_clock_mhz is not None
            else None,
            "throttle_reasons": decode_nvml_throttle_reasons(nvml, throttle_bits),
        }


class TelemetryMonitor:
    def __init__(
        self,
        enabled,
        devices,
        interval_s,
        logger,
        health,
    ):
        self.enabled = enabled
        self.devices = list(devices)
        self.interval_s = interval_s
        self.logger = logger
        self.health = health
        self.started_at = time.perf_counter()
        self.last_sample_at: dict[str, float] = {}
        self.nvml = None
        if enabled and any(device.type == "cuda" for device in self.devices):
            self.nvml = NvmlTelemetryBackend.create(logger, health)

    def close(self):
        if self.nvml is not None:
            self.nvml.close()

    def sample_all(self, force=False):
        for device in self.devices:
            self.sample(device, force=force)

    def sample(self, device, force=False):
        if not self.enabled:
            return
        now = time.perf_counter()
        device_name = str(device)
        last_sample_at = self.last_sample_at.get(device_name)
        if (
            not force
            and last_sample_at is not None
            and now - last_sample_at < self.interval_s
        ):
            return
        self.last_sample_at[device_name] = now
        self._sample_device(device, now)

    def _sample_device(self, device, now):
        torch_stats = safe_get_memory_stats(device, self.health)
        available_bytes, total_bytes, source = safe_get_device_memory_info(
            device,
            self.health,
        )
        sample_data = {
            "timestamp": utc_now_iso(),
            "elapsed_s": now - self.started_at,
            "device": str(device),
            "source": "torch",
            "torch_memory_mb": format_torch_memory_mb(torch_stats),
            "memory_available_mb": bytes_to_mb(available_bytes)
            if available_bytes is not None
            else None,
            "memory_total_mb": bytes_to_mb(total_bytes)
            if total_bytes is not None
            else None,
            "memory_info_source": source,
        }
        if device.type == "cuda" and self.nvml is not None:
            try:
                sample_data.update(self.nvml.sample(device))
                sample_data["source"] = "nvml"
            except Exception as exc:
                self.health.warn(device, f"NVML sample failed: {exc}")
        self.health.add_telemetry_sample(TelemetrySample(**sample_data))


def clear_device_cache(device):
    if device.type == "cuda":
        backend = getattr(torch, "cuda", None)
    elif device.type == "mps":
        backend = getattr(torch, "mps", None)
    else:
        backend = None
    empty_cache = getattr(backend, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    gc.collect()


def allocate_memory_chunks(target_bytes, device, dt):
    if target_bytes <= 0:
        raise RuntimeError("Memory target must be greater than zero.")
    element_size = get_dtype_size(dt)
    target_elements = max(1, ceil_div(target_bytes, element_size))
    max_chunk_elements = max(1, MAX_MEMORY_CHUNK_BYTES // element_size)
    chunks = []
    allocated_elements = 0

    try:
        while allocated_elements < target_elements:
            chunk_elements = min(
                max_chunk_elements,
                target_elements - allocated_elements,
            )
            chunks.append(torch.empty((chunk_elements,), device=device, dtype=dt))
            allocated_elements += chunk_elements
    except Exception:
        chunks.clear()
        clear_device_cache(device)
        raise

    return chunks, sum(tensor_nbytes(chunk) for chunk in chunks)


def touch_memory_chunks_once(
    chunks,
    device,
    dt,
    iteration_index,
    correctness="off",
    correctness_interval=DEFAULT_CORRECTNESS_INTERVAL,
):
    fill_value = float((iteration_index % 7) + 1)
    start = time.perf_counter()
    should_validate = should_validate_memory_iteration(
        correctness,
        iteration_index,
        correctness_interval,
    )
    for chunk in chunks:
        chunk.fill_(fill_value)
        if should_validate:
            observed_sum = tensor_sum_for_validation(chunk, dt)
            validate_memory_sum(
                "memory",
                chunk,
                fill_value,
                dt,
                device,
                observed_sum=observed_sum,
            )
        else:
            chunk.sum()
    sync(device)
    return time.perf_counter() - start


def touch_memory_chunks(
    chunks,
    allocated_bytes,
    iterations,
    device,
    dt,
    correctness="off",
    correctness_interval=DEFAULT_CORRECTNESS_INTERVAL,
    telemetry_monitor=None,
    progress=True,
):
    times = []
    for i in tqdm(
        range(iterations),
        desc=f"memory on {device}",
        leave=False,
        disable=not progress,
    ):
        times.append(
            touch_memory_chunks_once(
                chunks,
                device,
                dt,
                i,
                correctness,
                correctness_interval,
            )
        )
        if telemetry_monitor is not None:
            telemetry_monitor.sample(device)

    total_time = sum(times)
    touched_bytes = allocated_bytes * 2 * iterations
    bandwidth_gib_s = bytes_to_gib(touched_bytes) / total_time if total_time else 0.0
    return times, bandwidth_gib_s


def release_memory_chunks(chunks, device):
    chunks.clear()
    clear_device_cache(device)


def format_memory_stats(stats):
    return ", ".join(f"{name}={bytes_to_mb(value):.1f} MB" for name, value in stats.items())


def prepare_memory_stress(device, dt, memory_percent, memory_mb, logger):
    target_bytes, available_bytes, total_bytes, source = calculate_memory_target_bytes(
        device,
        memory_percent,
        memory_mb,
    )
    logger.info(
        f"[{device}] memory target={bytes_to_mb(target_bytes):.1f} MB "
        f"(source={source})"
    )
    if available_bytes is not None:
        total_text = "unknown" if total_bytes is None else f"{bytes_to_mb(total_bytes):.1f} MB"
        logger.info(
            f"[{device}] memory available={bytes_to_mb(available_bytes):.1f} MB, "
            f"total={total_text}"
        )

    reset_peak_memory_stats(device)
    chunks, allocated_bytes = allocate_memory_chunks(target_bytes, device, dt)
    return MemoryStressContext(
        device=device,
        chunks=chunks,
        allocated_bytes=allocated_bytes,
        available_bytes=available_bytes,
        total_bytes=total_bytes,
        source=source,
    )


def make_memory_stress_result(device, allocated_bytes, times, stats, failed=False, error=None):
    total_time = sum(times)
    touched_bytes = allocated_bytes * 2 * len(times)
    bandwidth_gib_s = bytes_to_gib(touched_bytes) / total_time if total_time else 0.0
    return MemoryStressResult(
        device=str(device),
        iterations=len(times),
        allocated_bytes=allocated_bytes,
        total_time=total_time,
        avg_touch_time=statistics.mean(times) if times else 0.0,
        bandwidth_gib_s=bandwidth_gib_s,
        stats=stats,
        failed=failed,
        error=error,
    )


def record_memory_context_failure(context, times, error, logger, health=None):
    context.failed = True
    context.error = str(error)
    result = make_memory_stress_result(
        context.device,
        context.allocated_bytes,
        times,
        safe_get_memory_stats(context.device, health),
        failed=True,
        error=context.error,
    )
    log_memory_stress_result(logger, result)
    if health is not None:
        health.add_memory_summary(result)
    return result


def log_memory_stress_result(logger, result):
    if result.failed:
        logger.warning(f"[{result.device}] Memory stress failed: {result.error}")
        return
    logger.info(
        f"[{result.device}] memory: allocated={bytes_to_mb(result.allocated_bytes):.1f} MB, "
        f"avg_touch={result.avg_touch_time:.6f}s, bandwidth={result.bandwidth_gib_s:.3f} GiB/s"
    )
    if result.stats:
        logger.info(f"[{result.device}] memory stats: {format_memory_stats(result.stats)}")


def run_memory_stress(
    device,
    dt,
    iterations,
    memory_percent,
    memory_mb,
    logger,
    correctness="off",
    correctness_interval=DEFAULT_CORRECTNESS_INTERVAL,
    telemetry_monitor=None,
    progress=True,
):
    context = None
    try:
        context = prepare_memory_stress(device, dt, memory_percent, memory_mb, logger)
        times, _ = touch_memory_chunks(
            context.chunks,
            context.allocated_bytes,
            iterations,
            device,
            dt,
            correctness=correctness,
            correctness_interval=correctness_interval,
            telemetry_monitor=telemetry_monitor,
            progress=progress,
        )
        result = make_memory_stress_result(
            device,
            context.allocated_bytes,
            times,
            safe_get_memory_stats(device),
        )
        log_memory_stress_result(logger, result)
        return result
    except (RuntimeError, MemoryError, TypeError) as exc:
        allocated_bytes = context.allocated_bytes if context is not None else 0
        result = make_memory_stress_result(
            device,
            allocated_bytes,
            [],
            safe_get_memory_stats(device),
            failed=True,
            error=str(exc),
        )
        log_memory_stress_result(logger, result)
        if context is None:
            clear_device_cache(device)
        return result
    finally:
        if context is not None:
            release_memory_chunks(context.chunks, device)


def run_memory_context_until_deadline(
    context,
    deadline,
    dt,
    correctness,
    correctness_interval,
    telemetry_monitor=None,
    logger=None,
    duration=None,
    progress=True,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
):
    times = []
    progress_reporter = None
    if logger is not None:
        progress_reporter = ProgressReporter(
            logger,
            enabled=progress,
            interval=progress_interval,
        )
        progress_reporter.start(f"[{context.device}] memory duration stress", duration)
    while time.perf_counter() < deadline:
        times.append(
            touch_memory_chunks_once(
                context.chunks,
                context.device,
                dt,
                len(times),
                correctness,
                correctness_interval,
            )
        )
        if telemetry_monitor is not None:
            telemetry_monitor.sample(context.device)
        if progress_reporter is not None:
            progress_reporter.maybe_log()
    if progress_reporter is not None:
        progress_reporter.finish()
    return times


def build_environment_report():
    report = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": is_cuda_available(),
        "mps_available": is_mps_available(),
    }
    if is_cuda_available():
        cuda_devices = []
        cuda = getattr(torch, "cuda", None)
        for idx in range(cuda.device_count()):
            props = cuda.get_device_properties(idx)
            cuda_devices.append(
                {
                    "index": idx,
                    "name": getattr(props, "name", "unknown"),
                    "total_memory_mb": bytes_to_mb(getattr(props, "total_memory", 0)),
                    "multiprocessors": getattr(
                        props,
                        "multi_processor_count",
                        None,
                    ),
                }
            )
        report["cuda_devices"] = cuda_devices
    return report


def operation_summary_to_dict(summary):
    return {
        "name": summary.name,
        "device": summary.device,
        "iterations": summary.iterations,
        "total_time_s": summary.total_time,
        "avg_time_s": summary.avg_time,
        "min_time_s": summary.min_time,
        "max_time_s": summary.max_time,
        "std_time_s": summary.std_time,
    }


def memory_summary_to_dict(summary):
    return {
        "device": summary.device,
        "iterations": summary.iterations,
        "allocated_mb": bytes_to_mb(summary.allocated_bytes),
        "total_time_s": summary.total_time,
        "avg_touch_time_s": summary.avg_touch_time,
        "bandwidth_gib_s": summary.bandwidth_gib_s,
        "stats_mb": format_torch_memory_mb(summary.stats),
        "failed": summary.failed,
        "error": summary.error,
    }


def benchmark_summary_to_dict(summary):
    return {
        "compute_score": summary.compute_score,
        "memory_score": summary.memory_score,
        "final_score": summary.final_score,
    }


def telemetry_sample_to_dict(sample):
    return {
        "timestamp": sample.timestamp,
        "elapsed_s": sample.elapsed_s,
        "device": sample.device,
        "source": sample.source,
        "temperature_c": sample.temperature_c,
        "power_w": sample.power_w,
        "memory_used_mb": sample.memory_used_mb,
        "memory_total_mb": sample.memory_total_mb,
        "utilization_gpu_percent": sample.utilization_gpu_percent,
        "utilization_memory_percent": sample.utilization_memory_percent,
        "sm_clock_mhz": sample.sm_clock_mhz,
        "memory_clock_mhz": sample.memory_clock_mhz,
        "throttle_reasons": list(sample.throttle_reasons),
        "torch_memory_mb": sample.torch_memory_mb or {},
        "memory_available_mb": sample.memory_available_mb,
        "memory_info_source": sample.memory_info_source,
    }


def build_json_report(
    health,
    config,
    devices,
    environment,
    started_at,
    finished_at,
):
    device_names = [str(device) for device in devices]
    return {
        "schema_version": JSON_REPORT_SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": health.status,
        "config": config,
        "environment": environment,
        "devices": [
            {"name": device, "status": health.device_status(device)}
            for device in device_names
        ],
        "issues": [
            {
                "severity": issue.severity,
                "device": issue.device,
                "message": issue.message,
            }
            for issue in health.issues
        ],
        "benchmark_summaries": {
            device: benchmark_summary_to_dict(summary)
            for device, summary in health.benchmark_summaries.items()
        },
        "operation_summaries": {
            device: [
                operation_summary_to_dict(summary)
                for summary in summaries
            ]
            for device, summaries in health.operation_summaries.items()
        },
        "memory_summaries": {
            device: [
                memory_summary_to_dict(summary)
                for summary in summaries
            ]
            for device, summaries in health.memory_summaries.items()
        },
        "telemetry": {
            device: {
                "summary": health.telemetry_summary(device),
                "samples": [
                    telemetry_sample_to_dict(sample)
                    for sample in health.telemetry_samples.get(device, [])
                ],
            }
            for device in device_names
        },
    }


def write_json_report(path, report, logger):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
        report_file.write("\n")
    logger.info(f"Wrote JSON report: {path}")


def log_health_summary(logger, health, devices):
    logger.info(f"Health summary: {health.status}")
    for device in devices:
        logger.info(f"[{device}] health={health.device_status(device)}")
        telemetry_summary = health.telemetry_summary(device)
        if telemetry_summary["sample_count"]:
            logger.info(
                f"[{device}] telemetry: samples={telemetry_summary['sample_count']}, "
                f"max_temp={telemetry_summary['max_temperature_c']}, "
                f"max_power={telemetry_summary['max_power_w']}, "
                f"max_memory={telemetry_summary['max_memory_used_mb']}, "
                f"max_gpu_util={telemetry_summary['max_gpu_utilization_percent']}, "
                f"max_mem_util={telemetry_summary['max_memory_utilization_percent']}, "
                f"max_sm_clock={telemetry_summary['max_sm_clock_mhz']}, "
                f"max_mem_clock={telemetry_summary['max_memory_clock_mhz']}, "
                f"throttle_reasons={telemetry_summary['throttle_reasons']}"
            )
    for issue in health.warnings:
        prefix = "" if issue.device is None else f"[{issue.device}] "
        logger.warning(f"{prefix}WARN: {issue.message}")
    for issue in health.failures:
        prefix = "" if issue.device is None else f"[{issue.device}] "
        safe_logger_error(logger, f"{prefix}FAIL: {issue.message}")


def finish_run(
    health,
    config,
    devices,
    environment,
    started_at,
    json_report,
    logger,
):
    finished_at = utc_now_iso()
    log_health_summary(logger, health, devices)
    if json_report:
        report = build_json_report(
            health,
            config,
            devices,
            environment,
            started_at,
            finished_at,
        )
        write_json_report(json_report, report, logger)
    if health.failures:
        raise click.ClickException("Health check failed.")


def build_compute_operations_for_suite(suite, size, device, dt, logger, health=None):
    ops = []
    if suite in ("basic", "all", "full"):
        ops.extend(build_basic_ops(size, device, dt))
    if suite in ("extended", "all", "full"):
        try:
            ops.extend(build_extended_ops(size, device, dt))
        except Exception as exc:
            if health is not None:
                health.fail(device, f"extended suite setup failed: {exc}")
            logger.warning(f"[{device}] Skipping extended suite: {exc}")
    return ops


def run_stress_for_device(
    device,
    suite,
    size,
    dt,
    iterations,
    memory_percent,
    memory_mb,
    duration,
    correctness,
    correctness_interval,
    logger,
    health,
    telemetry_monitor,
    progress=True,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
):
    logger.info(f"\nBenchmarking on device: {device}")
    deadline = time.perf_counter() + duration if duration is not None else None
    memory_context = None
    memory_times = []
    compute_enabled = suite in COMPUTE_SUITES
    memory_enabled = suite in MEMORY_SUITES
    memory_dt = get_memory_dtype(device, dt, logger) if memory_enabled else None
    ops = []

    try:
        if compute_enabled:
            skip_reason = get_skip_reason(device, dt)
            if skip_reason:
                logger.warning(skip_reason)
            else:
                try:
                    ops = build_compute_operations_for_suite(
                        suite,
                        size,
                        device,
                        dt,
                        logger,
                        health=health,
                    )
                except Exception as exc:
                    health.fail(device, f"compute setup failed: {exc}")
                    logger.warning(f"[{device}] Compute setup failed: {exc}")

        if deadline is not None and memory_enabled:
            try:
                memory_context = prepare_memory_stress(
                    device,
                    memory_dt,
                    memory_percent,
                    memory_mb,
                    logger,
                )
            except (RuntimeError, MemoryError, TypeError) as exc:
                result = make_memory_stress_result(
                    device,
                    0,
                    [],
                    safe_get_memory_stats(device, health),
                    failed=True,
                    error=str(exc),
                )
                log_memory_stress_result(logger, result)
                health.add_memory_summary(result)

        if compute_enabled and ops:
            if deadline is None:
                try:
                    run_operations(
                        ops,
                        device,
                        iterations,
                        logger,
                        skip_failed_ops=suite in ("extended", "all", "full"),
                        correctness=correctness,
                        correctness_interval=correctness_interval,
                        telemetry_monitor=telemetry_monitor,
                        health=health,
                        progress=progress,
                    )
                except Exception as exc:
                    health.fail(device, f"compute stress failed: {exc}")
                    logger.warning(f"[{device}] Compute stress failed: {exc}")
            else:
                run_operations_for_duration(
                    ops,
                    device,
                    deadline,
                    logger,
                    correctness=correctness,
                    correctness_interval=correctness_interval,
                    telemetry_monitor=telemetry_monitor,
                    health=health,
                    memory_context=memory_context,
                    memory_dt=memory_dt,
                    memory_times=memory_times,
                    duration=duration,
                    progress=progress,
                    progress_interval=progress_interval,
                )

        if memory_enabled:
            if deadline is None:
                memory_kwargs = {}
                if correctness != "off":
                    memory_kwargs["correctness"] = correctness
                    memory_kwargs["correctness_interval"] = correctness_interval
                if telemetry_monitor is not None and telemetry_monitor.enabled:
                    memory_kwargs["telemetry_monitor"] = telemetry_monitor
                memory_kwargs["progress"] = progress
                result = run_memory_stress(
                    device,
                    memory_dt,
                    iterations,
                    memory_percent,
                    memory_mb,
                    logger,
                    **memory_kwargs,
                )
                if isinstance(result, MemoryStressResult):
                    health.add_memory_summary(result)
            elif memory_context is not None:
                if not memory_context.failed and (not compute_enabled or not memory_times):
                    try:
                        memory_times.extend(
                            run_memory_context_until_deadline(
                                memory_context,
                                deadline,
                                memory_dt,
                                correctness,
                                correctness_interval,
                                telemetry_monitor=telemetry_monitor,
                                logger=logger,
                                duration=duration,
                                progress=progress,
                                progress_interval=progress_interval,
                            )
                        )
                    except (RuntimeError, MemoryError, TypeError) as exc:
                        record_memory_context_failure(
                            memory_context,
                            memory_times,
                            exc,
                            logger,
                            health,
                        )
                if not memory_context.failed:
                    result = make_memory_stress_result(
                        device,
                        memory_context.allocated_bytes,
                        memory_times,
                        safe_get_memory_stats(device, health),
                    )
                    log_memory_stress_result(logger, result)
                    health.add_memory_summary(result)
    except (RuntimeError, MemoryError, TypeError) as exc:
        health.fail(device, f"stress failed: {exc}")
        logger.warning(f"[{device}] Stress failed: {exc}")
    finally:
        if memory_context is not None:
            release_memory_chunks(memory_context.chunks, device)


@click.command()
@click.option(
    "--iterations", default=100, show_default=True,
    type=click.IntRange(min=1),
    help="Number of iterations per operation.",
)
@click.option(
    "--size", default=1024, show_default=True,
    type=click.IntRange(min=1),
    help="Matrix size for square operations.",
)
@click.option(
    "--dtype", default="float", show_default=True,
    type=click.Choice(DTYPE_CHOICES),
    help="Data type for tensors.",
)
@click.option(
    "--mode",
    default="stress",
    show_default=True,
    type=click.Choice(MODE_CHOICES),
    help="Run stress tests or the fixed-score benchmark profile.",
)
@click.option(
    "--suite",
    default="basic",
    show_default=True,
    type=click.Choice(SUITE_CHOICES),
    help="Benchmark suite to run.",
)
@click.option(
    "--memory-percent",
    default=70.0,
    show_default=True,
    type=click.FloatRange(min=1.0, max=95.0),
    help="Percent of detected available memory to stress.",
)
@click.option(
    "--memory-mb",
    default=None,
    type=click.IntRange(min=1),
    help="Exact memory stress target in MB; overrides --memory-percent.",
)
@click.option(
    "--seed", default=42, show_default=True,
    help="Random seed.",
)
@click.option(
    "--device", default=None, show_default=True,
    help="Device filter to run, for example cpu, cuda, cuda:0, or mps.",
)
@click.option(
    "--benchmark-memory",
    is_flag=True,
    help="Include the fixed memory bandwidth tests in benchmark mode.",
)
@click.option(
    "--benchmark-min-time",
    default=BENCHMARK_DEFAULT_MIN_TIME,
    show_default=True,
    type=click.FloatRange(min=0.001),
    help="Minimum timing duration per benchmark subtest.",
)
@click.option(
    "--preset",
    default=None,
    type=click.Choice(PRESET_CHOICES),
    help="Apply a preset profile, for example gpu-health.",
)
@click.option(
    "--telemetry/--no-telemetry",
    default=False,
    show_default=True,
    help="Collect PyTorch and NVML telemetry where supported.",
)
@click.option(
    "--telemetry-interval",
    default=DEFAULT_TELEMETRY_INTERVAL,
    show_default=True,
    type=click.FloatRange(min=0.1),
    help="Seconds between telemetry samples.",
)
@click.option(
    "--max-temp-c",
    default=DEFAULT_MAX_TEMP_C,
    show_default=True,
    type=click.FloatRange(min=1.0),
    help="Fail health checks at or above this GPU temperature.",
)
@click.option(
    "--correctness",
    default="off",
    show_default=True,
    type=click.Choice(CORRECTNESS_CHOICES),
    help="Correctness checking depth for stress operations.",
)
@click.option(
    "--correctness-interval",
    default=DEFAULT_CORRECTNESS_INTERVAL,
    show_default=True,
    type=click.IntRange(min=1),
    help="Operation interval for sampled correctness checks.",
)
@click.option(
    "--duration",
    default=None,
    type=click.FloatRange(min=0.001),
    help="Run stress mode for this many seconds per device; overrides --iterations.",
)
@click.option(
    "--json-report",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Write a machine-readable JSON report.",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show low-overhead wall-clock progress logs.",
)
@click.option(
    "--progress-interval",
    default=DEFAULT_PROGRESS_INTERVAL,
    show_default=True,
    type=click.FloatRange(min=1.0),
    help="Seconds between progress heartbeat logs.",
)
@click.option(
    "--verbose", is_flag=True,
    help="Enable DEBUG logging.",
)
def main(
    iterations: int,
    size: int,
    dtype: str,
    mode: str,
    suite: str,
    memory_percent: float,
    memory_mb: int | None,
    seed: int,
    device: str | None,
    benchmark_memory: bool,
    benchmark_min_time: float,
    verbose: bool,
    preset: str | None = None,
    telemetry: bool = False,
    telemetry_interval: float = DEFAULT_TELEMETRY_INTERVAL,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    correctness: str = "off",
    correctness_interval: int = DEFAULT_CORRECTNESS_INTERVAL,
    duration: float | None = None,
    json_report: str | None = None,
    progress: bool = True,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
):
    """
    Benchmark common PyTorch operations across available devices.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger(level)

    validate_torch_installation()

    mode, suite, device, memory_percent, duration, telemetry, correctness = (
        apply_preset_defaults(
            preset,
            mode,
            suite,
            device,
            memory_percent,
            memory_mb,
            duration,
            telemetry,
            correctness,
        )
    )
    if mode == "benchmark" and duration is not None:
        raise click.ClickException("--duration is only supported in stress mode.")

    log_environment_info(logger)
    environment = build_environment_report()

    torch.manual_seed(seed)
    devices = filter_devices(get_devices(), device)
    logger.info(f"Detected devices: {devices}")
    started_at = utc_now_iso()
    health = HealthReport(max_temp_c=max_temp_c)
    telemetry_monitor = TelemetryMonitor(
        telemetry,
        devices,
        telemetry_interval,
        logger,
        health,
    )
    config = {
        "preset": preset,
        "mode": mode,
        "suite": suite,
        "iterations": iterations,
        "duration_s": duration,
        "size": size,
        "dtype": dtype,
        "memory_percent": memory_percent,
        "memory_mb": memory_mb,
        "seed": seed,
        "device": device,
        "benchmark_memory": benchmark_memory,
        "benchmark_min_time": benchmark_min_time,
        "telemetry": telemetry,
        "telemetry_interval_s": telemetry_interval,
        "max_temp_c": max_temp_c,
        "correctness": correctness,
        "correctness_interval": correctness_interval,
        "progress": progress,
        "progress_interval_s": progress_interval,
    }

    # Map dtype strings to torch dtypes
    dtype_map = {"float": torch.float32, "double": torch.float64, "half": torch.float16}
    dt = dtype_map[dtype]

    try:
        telemetry_monitor.sample_all(force=True)
        if mode == "benchmark":
            summaries = run_benchmark_mode(
                devices,
                dt,
                benchmark_memory,
                memory_mb,
                benchmark_min_time,
                logger,
                progress=progress,
            )
            for bench_device, summary in summaries.items():
                health.add_benchmark_summary(bench_device, summary)
        else:
            for bench_device in devices:
                run_stress_for_device(
                    bench_device,
                    suite,
                    size,
                    dt,
                    iterations,
                    memory_percent,
                    memory_mb,
                    duration,
                    correctness,
                    correctness_interval,
                    logger,
                    health,
                    telemetry_monitor,
                    progress=progress,
                    progress_interval=progress_interval,
                )
        telemetry_monitor.sample_all(force=True)
    finally:
        telemetry_monitor.close()

    finish_run(
        health,
        config,
        devices,
        environment,
        started_at,
        json_report,
        logger,
    )


if __name__ == "__main__":
    main()
