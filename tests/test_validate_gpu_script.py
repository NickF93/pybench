import json
import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "validate_gpu.sh"


class ValidateGpuScriptTests(unittest.TestCase):
    def make_fake_python(self, temp_dir, status="PASS", exit_code=0):
        fake_python = Path(temp_dir) / "fake-python"
        fake_log = Path(temp_dir) / "fake-python-args.jsonl"
        fake_python.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import sys

                with open(os.environ["FAKE_PYTHON_LOG"], "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(sys.argv[1:]) + "\\n")

                if len(sys.argv) > 1 and sys.argv[1] == "-c":
                    with open(sys.argv[-1], encoding="utf-8") as report_file:
                        report = json.load(report_file)
                    print(report.get("status", ""))
                    raise SystemExit(0)

                args = sys.argv[1:]
                if "--json-report" in args:
                    report_path = args[args.index("--json-report") + 1]
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w", encoding="utf-8") as report_file:
                        json.dump({"status": os.environ.get("FAKE_PYBENCH_STATUS", "PASS")}, report_file)
                        report_file.write("\\n")
                print("fake pybench")
                raise SystemExit(int(os.environ.get("FAKE_PYBENCH_EXIT", "0")))
                """
            ),
            encoding="utf-8",
        )
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

        env = os.environ.copy()
        env["FAKE_PYTHON_LOG"] = str(fake_log)
        env["FAKE_PYBENCH_STATUS"] = status
        env["FAKE_PYBENCH_EXIT"] = str(exit_code)
        return fake_python, fake_log, env

    def run_script(self, args, env=None):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def read_pybench_invocations(self, fake_log):
        calls = [
            json.loads(line)
            for line in Path(fake_log).read_text(encoding="utf-8").splitlines()
        ]
        return [call for call in calls if call and call[0].endswith("pybench/pytorch_bench.py")]

    def test_help_exits_successfully(self):
        result = self.run_script(["--help"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)

    def test_missing_device_fails(self):
        result = self.run_script([])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--device is required", result.stderr)

    def test_non_cuda_device_fails(self):
        result = self.run_script(["--device", "cpu"])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--device must be cuda or cuda:N", result.stderr)

    def test_dry_run_prints_all_default_stage_commands(self):
        result = self.run_script(["--device", "cuda:0", "--dry-run"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("--duration 300", result.stdout)
        self.assertIn("--duration 1800", result.stdout)
        self.assertIn("--correctness strict", result.stdout)
        self.assertIn("--correctness sampled", result.stdout)
        self.assertIn("--mode benchmark", result.stdout)
        self.assertIn("--benchmark-memory", result.stdout)

    def test_default_run_invokes_smoke_soak_and_benchmark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_python, fake_log, env = self.make_fake_python(temp_dir)
            result = self.run_script(
                [
                    "--device",
                    "cuda:0",
                    "--python",
                    str(fake_python),
                    "--out-dir",
                    str(Path(temp_dir) / "reports"),
                ],
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = self.read_pybench_invocations(fake_log)
            self.assertEqual(len(calls), 3)
            self.assertIn("--preset", calls[0])
            self.assertEqual(calls[0][calls[0].index("--correctness") + 1], "strict")
            self.assertEqual(calls[0][calls[0].index("--duration") + 1], "300")
            self.assertEqual(calls[1][calls[1].index("--correctness") + 1], "sampled")
            self.assertEqual(calls[1][calls[1].index("--duration") + 1], "1800")
            self.assertIn("--mode", calls[2])
            self.assertIn("--benchmark-memory", calls[2])

    def test_warn_status_fails_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_python, fake_log, env = self.make_fake_python(temp_dir, status="WARN")
            result = self.run_script(
                [
                    "--device",
                    "cuda:0",
                    "--python",
                    str(fake_python),
                    "--out-dir",
                    str(Path(temp_dir) / "reports"),
                ],
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("status WARN", result.stderr)
            self.assertEqual(len(self.read_pybench_invocations(fake_log)), 1)

    def test_nonzero_pybench_exit_stops_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_python, fake_log, env = self.make_fake_python(temp_dir, exit_code=7)
            result = self.run_script(
                [
                    "--device",
                    "cuda:0",
                    "--python",
                    str(fake_python),
                    "--out-dir",
                    str(Path(temp_dir) / "reports"),
                ],
                env=env,
            )

            self.assertEqual(result.returncode, 7)
            self.assertIn("command exited with 7", result.stderr)
            self.assertEqual(len(self.read_pybench_invocations(fake_log)), 1)

    def test_allow_warn_permits_warn_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_python, fake_log, env = self.make_fake_python(temp_dir, status="WARN")
            result = self.run_script(
                [
                    "--device",
                    "cuda:0",
                    "--python",
                    str(fake_python),
                    "--out-dir",
                    str(Path(temp_dir) / "reports"),
                    "--allow-warn",
                ],
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(self.read_pybench_invocations(fake_log)), 3)

    def test_skip_benchmark_omits_benchmark_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_python, fake_log, env = self.make_fake_python(temp_dir)
            result = self.run_script(
                [
                    "--device",
                    "cuda:0",
                    "--python",
                    str(fake_python),
                    "--out-dir",
                    str(Path(temp_dir) / "reports"),
                    "--skip-benchmark",
                ],
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = self.read_pybench_invocations(fake_log)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("--mode", calls[0])
            self.assertNotIn("--mode", calls[1])


if __name__ == "__main__":
    unittest.main()
