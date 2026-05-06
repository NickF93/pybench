#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBENCH="${SCRIPT_DIR}/pybench/pytorch_bench.py"

DEVICE=""
SMOKE_DURATION="300"
SOAK_DURATION="1800"
MEMORY_PERCENT="80"
MAX_TEMP_C="90"
TELEMETRY_INTERVAL="5"
PROGRESS_INTERVAL="60"
SIZE="2048"
DTYPE="float"
OUT_DIR="${SCRIPT_DIR}/reports/gpu-validation"
PYTHON_BIN="${PYTHON:-python3}"
SKIP_BENCHMARK=0
ALLOW_WARN=0
DRY_RUN=0
PROGRESS=1

usage() {
    cat <<'USAGE'
Usage: ./validate_gpu.sh --device cuda:N [options]

Validate a CUDA GPU with pybench health stress, VRAM pressure, telemetry,
correctness checks, and an optional benchmark baseline.

Required:
  --device DEVICE              CUDA device filter to validate, for example cuda:0.

Options:
  --smoke-duration SECONDS     First strict validation stage duration. Default: 300.
  --soak-duration SECONDS      Longer sampled validation stage duration. Default: 1800.
  --memory-percent PERCENT     Percent of available VRAM to stress. Default: 80.
  --max-temp-c C               Fail at or above this GPU temperature. Default: 90.
  --telemetry-interval SECONDS Seconds between telemetry samples. Default: 5.
  --progress-interval SECONDS  Seconds between progress logs. Default: 60.
  --size N                     Matrix size for stress operations. Default: 2048.
  --dtype TYPE                 float, double, or half. Default: float.
  --out-dir PATH               Base directory for timestamped artifacts.
  --python PATH                Python executable to run pybench. Default: ${PYTHON:-python3}.
  --skip-benchmark             Skip the final benchmark baseline.
  --allow-warn                 Treat WARN health summaries as successful.
  --progress                   Show low-overhead progress logs. Default.
  --no-progress                Disable wrapper and pybench progress logs.
  --dry-run                    Print commands without running them.
  -h, --help                   Show this help.
USAGE
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        fail "${option} requires a value"
    fi
}

while (($#)); do
    case "$1" in
        --device)
            require_value "$1" "${2:-}"
            DEVICE="$2"
            shift 2
            ;;
        --smoke-duration)
            require_value "$1" "${2:-}"
            SMOKE_DURATION="$2"
            shift 2
            ;;
        --soak-duration)
            require_value "$1" "${2:-}"
            SOAK_DURATION="$2"
            shift 2
            ;;
        --memory-percent)
            require_value "$1" "${2:-}"
            MEMORY_PERCENT="$2"
            shift 2
            ;;
        --max-temp-c)
            require_value "$1" "${2:-}"
            MAX_TEMP_C="$2"
            shift 2
            ;;
        --telemetry-interval)
            require_value "$1" "${2:-}"
            TELEMETRY_INTERVAL="$2"
            shift 2
            ;;
        --progress-interval)
            require_value "$1" "${2:-}"
            PROGRESS_INTERVAL="$2"
            shift 2
            ;;
        --size)
            require_value "$1" "${2:-}"
            SIZE="$2"
            shift 2
            ;;
        --dtype)
            require_value "$1" "${2:-}"
            DTYPE="$2"
            shift 2
            ;;
        --out-dir)
            require_value "$1" "${2:-}"
            OUT_DIR="$2"
            shift 2
            ;;
        --python)
            require_value "$1" "${2:-}"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=1
            shift
            ;;
        --allow-warn)
            ALLOW_WARN=1
            shift
            ;;
        --progress)
            PROGRESS=1
            shift
            ;;
        --no-progress)
            PROGRESS=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

[[ -n "$DEVICE" ]] || fail "--device is required; use cuda or cuda:N"
if [[ ! "$DEVICE" =~ ^cuda(:[0-9]+)?$ ]]; then
    fail "--device must be cuda or cuda:N for GPU validation"
fi

case "$DTYPE" in
    float|double|half) ;;
    *) fail "--dtype must be float, double, or half" ;;
esac
for numeric_option in SMOKE_DURATION SOAK_DURATION PROGRESS_INTERVAL; do
    numeric_value="${!numeric_option}"
    if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
        case "$numeric_option" in
            SMOKE_DURATION) option_name="--smoke-duration" ;;
            SOAK_DURATION) option_name="--soak-duration" ;;
            *) option_name="--progress-interval" ;;
        esac
        fail "${option_name} must be a positive whole number of seconds"
    fi
done

print_command() {
    local arg
    printf '  '
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
    printf '\n'
}

format_seconds() {
    local seconds="${1%.*}"
    local hours minutes
    if [[ -z "$seconds" ]]; then
        seconds=0
    fi
    hours=$((seconds / 3600))
    minutes=$(((seconds % 3600) / 60))
    seconds=$((seconds % 60))
    if ((hours > 0)); then
        printf '%dh %dm %ds' "$hours" "$minutes" "$seconds"
    elif ((minutes > 0)); then
        printf '%dm %ds' "$minutes" "$seconds"
    else
        printf '%ds' "$seconds"
    fi
}

print_plan() {
    local total_seconds=$((SMOKE_DURATION + SOAK_DURATION))
    echo "Plan:"
    echo "  1/3 smoke validation     $(format_seconds "$SMOKE_DURATION")"
    echo "  2/3 soak validation      $(format_seconds "$SOAK_DURATION")"
    if ((SKIP_BENCHMARK)); then
        echo "  3/3 benchmark baseline   skipped"
    else
        echo "  3/3 benchmark baseline   ETA unavailable"
    fi
    echo "  estimated timed stress   $(format_seconds "$total_seconds")"
}

read_report_status() {
    "$PYTHON_BIN" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as report_file:
    report = json.load(report_file)
status = report.get("status")
if not isinstance(status, str) or not status:
    raise SystemExit("missing JSON report status")
print(status)
' "$1"
}

capture_nvidia_smi() {
    local name="$1"
    local path="${RUN_DIR}/nvidia-smi-${name}.txt"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi >"$path" 2>&1 || true
    fi
}

progress_loop() {
    local stage_name="$1"
    local stage_duration="$2"
    local total_offset="$3"
    local total_duration="$4"
    local log_file="$5"
    local started_at elapsed remaining stage_percent total_percent message sleep_pid

    trap 'kill "$sleep_pid" >/dev/null 2>&1 || true; exit 0' TERM INT
    started_at="$(date +%s)"
    while true; do
        sleep "$PROGRESS_INTERVAL" &
        sleep_pid=$!
        wait "$sleep_pid" || exit 0
        elapsed=$(($(date +%s) - started_at))
        remaining=$((stage_duration - elapsed))
        if ((remaining < 0)); then
            remaining=0
        fi
        if ((stage_duration > 0)); then
            stage_percent=$((elapsed * 100 / stage_duration))
        else
            stage_percent=100
        fi
        if ((stage_percent > 100)); then
            stage_percent=100
        fi
        if ((total_duration > 0)); then
            total_percent=$(((total_offset + elapsed) * 100 / total_duration))
        else
            total_percent=100
        fi
        if ((total_percent > 100)); then
            total_percent=100
        fi
        message="[${stage_name}] elapsed=$(format_seconds "$elapsed") remaining=$(format_seconds "$remaining") stage=${stage_percent}% total~=${total_percent}%"
        echo "$message" | tee -a "$log_file"
    done
}

run_stage() {
    local stage_name="$1"
    local stage_duration="$2"
    local total_offset="$3"
    local log_file="$4"
    local json_file="$5"
    shift 5
    local progress_pid=""

    echo
    echo "Running ${stage_name}..."
    print_command "$@"

    if ((DRY_RUN)); then
        return 0
    fi

    if ((PROGRESS && stage_duration > 0)); then
        progress_loop "$stage_name" "$stage_duration" "$total_offset" "$TOTAL_TIMED_DURATION" "$log_file" &
        progress_pid=$!
    elif ((PROGRESS)); then
        echo "[${stage_name}] running; ETA unavailable" | tee -a "$log_file"
    fi

    "$@" 2>&1 | tee -a "$log_file"
    local command_status=${PIPESTATUS[0]}
    if [[ -n "$progress_pid" ]]; then
        kill "$progress_pid" >/dev/null 2>&1 || true
        wait "$progress_pid" >/dev/null 2>&1 || true
    fi
    if ((command_status != 0)); then
        echo "${stage_name} failed: command exited with ${command_status}" >&2
        return "$command_status"
    fi

    local report_status
    if ! report_status="$(read_report_status "$json_file")"; then
        echo "${stage_name} failed: missing or invalid JSON report: ${json_file}" >&2
        return 1
    fi

    case "$report_status" in
        PASS)
            echo "${stage_name} status: PASS"
            ;;
        WARN)
            if ((ALLOW_WARN)); then
                echo "${stage_name} status: WARN (allowed)"
            else
                echo "${stage_name} failed: status WARN" >&2
                return 1
            fi
            ;;
        FAIL)
            echo "${stage_name} failed: status FAIL" >&2
            return 1
            ;;
        *)
            echo "${stage_name} failed: unknown status ${report_status}" >&2
            return 1
            ;;
    esac
}

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEVICE_LABEL="${DEVICE//:/_}"
RUN_DIR="${OUT_DIR}/${TIMESTAMP}_${DEVICE_LABEL}"
TOTAL_TIMED_DURATION=$((SMOKE_DURATION + SOAK_DURATION))

SMOKE_LOG="${RUN_DIR}/smoke.log"
SMOKE_JSON="${RUN_DIR}/smoke.json"
SOAK_LOG="${RUN_DIR}/soak.log"
SOAK_JSON="${RUN_DIR}/soak.json"
BENCHMARK_LOG="${RUN_DIR}/benchmark.log"
BENCHMARK_JSON="${RUN_DIR}/benchmark.json"

echo "GPU validation target: ${DEVICE}"
echo "Artifacts: ${RUN_DIR}"
print_plan

if ((DRY_RUN)); then
    echo "Dry run: commands will not be executed."
else
    mkdir -p "$RUN_DIR"
    capture_nvidia_smi "before"
fi

overall_status=0

SMOKE_CMD=(
    "$PYTHON_BIN" "$PYBENCH"
    --preset gpu-health
    --device "$DEVICE"
    --duration "$SMOKE_DURATION"
    --memory-percent "$MEMORY_PERCENT"
    --telemetry
    --telemetry-interval "$TELEMETRY_INTERVAL"
    --progress-interval "$PROGRESS_INTERVAL"
    --max-temp-c "$MAX_TEMP_C"
    --correctness strict
    --size "$SIZE"
    --dtype "$DTYPE"
    --json-report "$SMOKE_JSON"
)
if ((PROGRESS)); then
    SMOKE_CMD+=(--progress)
else
    SMOKE_CMD+=(--no-progress)
fi

run_stage "smoke validation" "$SMOKE_DURATION" 0 "$SMOKE_LOG" "$SMOKE_JSON" "${SMOKE_CMD[@]}"
stage_status=$?
if ((stage_status != 0)); then
    overall_status=$stage_status
fi

if ((overall_status == 0)); then
    SOAK_CMD=(
        "$PYTHON_BIN" "$PYBENCH"
        --preset gpu-health
        --device "$DEVICE"
        --duration "$SOAK_DURATION"
        --memory-percent "$MEMORY_PERCENT"
        --telemetry
        --telemetry-interval "$TELEMETRY_INTERVAL"
        --progress-interval "$PROGRESS_INTERVAL"
        --max-temp-c "$MAX_TEMP_C"
        --correctness sampled
        --size "$SIZE"
        --dtype "$DTYPE"
        --json-report "$SOAK_JSON"
    )
    if ((PROGRESS)); then
        SOAK_CMD+=(--progress)
    else
        SOAK_CMD+=(--no-progress)
    fi

    run_stage "soak validation" "$SOAK_DURATION" "$SMOKE_DURATION" "$SOAK_LOG" "$SOAK_JSON" "${SOAK_CMD[@]}"
    stage_status=$?
    if ((stage_status != 0)); then
        overall_status=$stage_status
    fi
fi

if ((overall_status == 0 && SKIP_BENCHMARK == 0)); then
    BENCHMARK_CMD=(
        "$PYTHON_BIN" "$PYBENCH"
        --mode benchmark
        --device "$DEVICE"
        --benchmark-memory
        --telemetry
        --telemetry-interval "$TELEMETRY_INTERVAL"
        --progress-interval "$PROGRESS_INTERVAL"
        --max-temp-c "$MAX_TEMP_C"
        --dtype "$DTYPE"
        --json-report "$BENCHMARK_JSON"
    )
    if ((PROGRESS)); then
        BENCHMARK_CMD+=(--progress)
    else
        BENCHMARK_CMD+=(--no-progress)
    fi

    run_stage "benchmark baseline" 0 "$TOTAL_TIMED_DURATION" "$BENCHMARK_LOG" "$BENCHMARK_JSON" "${BENCHMARK_CMD[@]}"
    stage_status=$?
    if ((stage_status != 0)); then
        overall_status=$stage_status
    fi
elif ((overall_status == 0)); then
    echo
    echo "Skipping benchmark baseline."
fi

if ((DRY_RUN == 0)); then
    capture_nvidia_smi "after"
fi

echo
if ((overall_status == 0)); then
    echo "GPU validation completed successfully."
    echo "Artifacts: ${RUN_DIR}"
else
    echo "GPU validation failed."
    echo "Artifacts: ${RUN_DIR}"
fi

exit "$overall_status"
