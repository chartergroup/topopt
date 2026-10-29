"""`topopt gpu-check` (CONTRACTS.md sections 8 and 20, L1b).

The child processes are faked (a runner that records the commands and returns the output of a real run), and the
OpenCL devices are faked, so no driver is loaded and no GPU is needed. The real children run by hand on a machine
with a device.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from helpers import fake_opencl, make_device
from topopt import cli, gpu_check

REPO = Path(__file__).resolve().parent.parent

PARITY_M4 = """device: 0: Apple M4 (GPU, platform Apple, OpenCL 1.2, 64-bit float no); case auto = C
free dofs: 7260, elements: 2200

| reference | gpu-mg case | max rel diff of u | max rel diff of compliance | iterations | gpu-mg solve s | reference s |
|---|---|---|---|---|---|---|
| superlu | C | 2.90e-11 | 2.71e-11 | 13 inner in 3 refinement steps | 1.11 | 0.35 |
| cholesky | C | 2.90e-11 | 2.71e-11 | 13 inner in 3 refinement steps | 1.11 | 0.04 |

reference for the exit code: cholesky. Note: the gpu-mg time includes the first kernel build on devices that compile at the first launch; it is not a speed measurement.
PASS
"""
TESTS_M4 = "..........\n98 passed, 10 skipped, 3 warnings in 27.10s\n"
TESTS_BAD = ("..F\n=================================== FAILURES ===================================\n"
             "=========================== short test summary info ============================\n"
             "FAILED tests/test_gpu_mg.py::test_a[C] - AssertionError\nFAILED tests/test_gpu_mg.py::test_b[C] - x\n"
             "2 failed, 96 passed, 10 skipped in 30.00s\n")
BENCH_MD = "# gpu-mg benchmark\n\n- machine: test\n\n| voxel mm | solver | status |\n|---|---|---|\n| 1.2 | default | ok |\n"


def dev_dict(index=0, name="Apple M4", kind="GPU", fp64=False, platform_name="Apple", duplicate_of=None, vendor=""):
    d = make_device(index, name, kind, fp64, platform_name, vendor).facts()
    d["duplicate_of"] = duplicate_of
    return d


def machine(devices=None, pyopencl="2026.1.4", problem=None, auto=0, recommendations=None, system="darwin"):
    """The `environment_data` dict, as far as gpu-check reads it."""
    devices = devices if devices is not None else [dev_dict()]
    return {"topopt_version": "0.1.0", "python": {"version": "3.12.15"}, "platform": {"name": "macos arm64"},
            "ram": {"total_bytes": 16 * 2**30}, "cores": {"usable": 8},
            "checkout": {"code_folder": "/code", "git": True, "kind": "git checkout", "head": "abc1234",
                         "head_date": "2026-10-21T00:00:00+00:00", "branch": "main", "dirty": False, "changed_files": 0,
                         "untracked_files": 0, "default_branch_ref": "origin/main", "ahead": 0, "behind": 0,
                         "last_fetch_utc": "2026-10-21T00:00:00Z", "last_fetch_age_s": 60.0},
            "opencl": {"pyopencl": pyopencl, "problem": problem, "install_command": "uv pip install --python /py -e '/code[gpu]'",
                       "devices": devices, "auto_device": auto if devices else None},
            "recommendations": recommendations or []}


@pytest.fixture
def root(tmp_path):
    """A folder that looks like a checkout: the two tools and the four test files."""
    r = tmp_path / "checkout"
    for rel in (*gpu_check.TOOL_FILES, *gpu_check.TEST_FILES):
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text("# fake\n")
    return r


def bench_report_with_dofs():
    """A report of `tools/bench_gpu_mg.py` as the bench step reads it for the calibration: the M4 device and two gpu-mg sizes with 2 load cases."""
    def run(h, free_dofs, solve_s, iteration_s):
        its = [100, 110, 100, 90]                                         # PCG iterations of the optimizer iterations
        return {"h": h, "solver": "gpu-mg", "status": "ok", "free_dofs": free_dofs, "n_load_cases": 2, "median_iter_s": iteration_s,
                "gpu_mg": {"case": "C", "median_s": {"solve_s": solve_s},
                           "per_iteration": [{"solve_s": solve_s * k / 100, "device_iterations": k} for k in its]}}
    return {"opencl": {"device": {"platform": "Apple", "name": "Apple M4", "vendor": "Apple", "kind": "GPU", "fp64": False}},
            "runs": [run(1.2, 100_000, 0.30, 0.40), run(1.0, 300_000, 0.90, 1.20)]}


class Runner:
    """Records the commands and answers like the real children. `results` maps parity, tests and bench to a dict."""

    def __init__(self, parity=(0, PARITY_M4), tests=(0, TESTS_M4), bench=(0, "report: x\n"), calibratable=False):
        self.results = {"parity": parity, "tests": tests, "bench": bench}
        self.calibratable = calibratable        # True: the bench report has the device, the dofs and the PCG iterations of a real one
        self.calls = []

    def __call__(self, cmd, env=None, cwd=None, timeout=None):
        key = "parity" if cmd[1].endswith("gpu_mg_parity.py") else "bench" if cmd[1].endswith("bench_gpu_mg.py") else "tests"
        self.calls.append({"step": key, "cmd": list(cmd), "env": env, "cwd": cwd, "timeout": timeout})
        rc, text = self.results[key]
        if key == "bench" and rc == 0:
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            runs = [{"h": h, "solver": s, "status": "ok", "median_iter_s": t}
                    for h, s, t in ((1.2, "default", 0.765), (1.2, "gpu-mg", 0.287), (1.0, "default", 2.235), (1.0, "gpu-mg", 0.439))]
            report = {"runs": runs}
            if self.calibratable:
                report = bench_report_with_dofs()
            (out / "bench_gpu_mg.json").write_text(json.dumps(report))
            (out / "bench_gpu_mg.md").write_text(BENCH_MD)
        return {"returncode": rc, "output": text, "seconds": 1.5, "timed_out": rc is None}

    def steps(self):
        return [c["step"] for c in self.calls]


def go(root, tmp_path, data=None, runner=None, **kw):
    said = []
    runner = runner or Runner()
    kw.setdefault("have_module", lambda name: True)
    code = gpu_check.run(out=tmp_path / "out", root=root, runner=runner, data=data or machine(), say=said.append, **kw)
    report = json.loads((tmp_path / "out" / "gpu-check.json").read_text())
    return code, report, runner, said


def states(report):
    return {s["id"]: s["state"] for s in report["steps"]}


# --- the four steps ---------------------------------------------------------------------------------------------------


def test_all_steps_pass_and_the_report_is_written_and_printed(root, tmp_path):
    code, report, runner, said = go(root, tmp_path)
    assert code == 0 and report["result"] == "PASS" and report["exit_code"] == 0
    assert states(report) == {"facts": "passed", "parity": "passed", "tests": "passed", "bench": "passed"}
    assert runner.steps() == ["parity", "tests", "bench"]
    parity, tests, bench = runner.calls
    assert parity["cmd"][1:] == [str(root / "tools" / "gpu_mg_parity.py"), "--device", "0"] and parity["cwd"] == str(root)
    assert tests["cmd"][1:4] == ["-m", "pytest", "-q"] and list(gpu_check.TEST_FILES) == tests["cmd"][-4:]
    assert bench["cmd"][2:] == ["--out", str(tmp_path / "out" / "bench"), "--sizes", "1.2,1.0", "--iterations", "20", "--device", "0"]
    md = (tmp_path / "out" / "gpu-check.md").read_text()
    assert said[-1] == md                                                      # the markdown is printed at the end
    assert "**Result: PASS**" in md and "# topopt gpu-check report" in md
    assert "98 passed, 10 skipped" in md and "2.67x" in md and "5.09x" in md
    assert "2.90e-11" in md and "limit 1e-6" in md
    assert (tmp_path / "out" / "parity.log").read_text() == PARITY_M4 and (tmp_path / "out" / "tests.log").is_file()
    assert "experimental" not in md and "The numbers are those of this machine." in md and "Apple M4" in md and "origin/main: 0 commit(s) ahead, 0 behind" in md


def test_only_the_test_step_gets_the_real_gpu_variable(root, tmp_path, monkeypatch):
    monkeypatch.delenv("TOPOPT_TEST_REAL_GPU", raising=False)
    code, report, runner, _ = go(root, tmp_path)
    parity, tests, bench = runner.calls
    assert tests["env"]["TOPOPT_TEST_REAL_GPU"] == "1"
    assert "TOPOPT_TEST_REAL_GPU" not in parity["env"] and "TOPOPT_TEST_REAL_GPU" not in bench["env"]
    assert os.environ.get("TOPOPT_TEST_REAL_GPU") is None            # the parent environment is not changed


def test_skip_options_skip_the_step_and_still_pass(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, skip_bench=True)
    assert code == 0 and states(report)["bench"] == "skipped" and runner.steps() == ["parity", "tests"]
    assert report["result"] == "PASS (skipped: bench)"
    code, report, runner, _ = go(root, tmp_path, skip_tests=True, skip_bench=True)
    assert code == 0 and runner.steps() == ["parity"] and report["result"] == "PASS (skipped: tests, bench)"


def test_a_failed_parity_stops_the_flow_and_says_what_to_run(root, tmp_path):
    runner = Runner(parity=(1, "device: x\nFAIL: a relative difference is above 1e-06\n"))
    code, report, runner, _ = go(root, tmp_path, runner=runner)
    assert code == 1 and runner.steps() == ["parity"]
    assert states(report) == {"facts": "passed", "parity": "failed", "tests": "not run", "bench": "not run"}
    assert report["result"].startswith("FAIL at step parity: FAIL: a relative difference is above")
    assert report["next"] == [f"{sys.executable} {root / 'tools' / 'gpu_mg_parity.py'} --device 0"]
    assert "step parity failed" in report["steps"][2]["summary"]
    assert "## Run next" in (tmp_path / "out" / "gpu-check.md").read_text()


def test_failed_tests_stop_the_bench_and_name_the_failures(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, runner=Runner(tests=(1, TESTS_BAD)))
    assert code == 1 and runner.steps() == ["parity", "tests"] and states(report)["bench"] == "not run"
    tests = report["steps"][2]
    assert tests["state"] == "failed" and tests["summary"] == "2 failed, 96 passed, 10 skipped in 30.00s"
    assert "FAILED tests/test_gpu_mg.py::test_a[C]" in tests["detail"]
    assert "-x -q -rf" in report["next"][0] and report["next"][0].startswith(f"cd {root} && TOPOPT_TEST_REAL_GPU=1 ")
    assert tests["command"].startswith(f"cd {root} && TOPOPT_TEST_REAL_GPU=1 ")      # the test paths are relative (review n4)


def test_a_failed_bench_fails_the_check(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, runner=Runner(bench=(1, "h1.2_gpu-mg: error\n")))
    assert code == 1 and states(report)["bench"] == "failed" and report["result"].startswith("FAIL at step bench")
    assert "bench_gpu_mg.py" in report["next"][0]


def test_the_table_of_an_earlier_run_is_not_the_result_of_a_failed_bench(root, tmp_path):
    """Review m1: the same report folder, the second bench fails before it writes files."""
    code, report, _runner, _ = go(root, tmp_path)
    assert code == 0 and "2.67x" in (tmp_path / "out" / "gpu-check.md").read_text()
    code, report, _runner, _ = go(root, tmp_path, runner=Runner(bench=(1, "clBuildProgram failed\n")))
    bench = report["steps"][-1]
    assert code == 1 and bench["state"] == "failed" and bench["summary"] == "clBuildProgram failed"
    assert "2.67x" not in (tmp_path / "out" / "gpu-check.md").read_text() and "2.67x" not in json.dumps(report)


def test_ctrl_c_stops_the_step_and_still_writes_the_report(root, tmp_path):
    """Review m5: the report says `stopped by the user`; the steps that did not run are `not run`."""
    base = Runner()

    def runner(cmd, **kw):
        if cmd[1:3] == ["-m", "pytest"]:
            raise KeyboardInterrupt
        return base(cmd, **kw)
    said = []
    code = gpu_check.run(out=tmp_path / "out", root=root, runner=runner, data=machine(), say=said.append,
                         have_module=lambda n: True)
    report = json.loads((tmp_path / "out" / "gpu-check.json").read_text())
    assert code == 1 and states(report) == {"facts": "passed", "parity": "passed", "tests": "failed", "bench": "not run"}
    assert report["result"] == "FAIL at step tests: stopped by the user (Ctrl-C)"
    assert any("stopped by the user" in t for t in said) and (tmp_path / "out" / "gpu-check.md").is_file()
    assert said[-1] == (tmp_path / "out" / "gpu-check.md").read_text()
    assert report["next"] == [f"{sys.executable} -m topopt gpu-check --out {tmp_path / 'out'}"]


def test_ctrl_c_in_the_facts_step_leaves_four_steps(root, tmp_path, monkeypatch):
    from topopt import environment

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(environment, "environment_data", interrupted)
    code = gpu_check.run(out=tmp_path / "out", root=root, runner=Runner(), data=None, say=lambda t="": None)
    report = json.loads((tmp_path / "out" / "gpu-check.json").read_text())
    assert code == 1 and states(report) == {"facts": "failed", "parity": "not run", "tests": "not run", "bench": "not run"}


def test_a_step_that_hits_the_time_limit_says_so(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, runner=Runner(parity=(None, "partial output")), step_timeout_s=5)
    assert code == 1 and "stopped by the step time limit" in report["steps"][1]["summary"] and runner.calls[0]["timeout"] == 5


def test_pytest_missing_is_not_run_with_the_fix_and_the_bench_still_runs(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, have_module=lambda name: name != "pytest")
    assert code == 1 and runner.steps() == ["parity", "bench"]
    assert states(report)["tests"] == "not run" and states(report)["bench"] == "passed"
    assert report["result"].startswith("NOT RUN: tests: pytest is not installed")
    assert report["next"] == [f"uv pip install --python {sys.executable} -e '{root}[dev]'"]


def test_missing_test_files_are_not_run(root, tmp_path):
    (root / gpu_check.TEST_FILES[0]).unlink()
    code, report, runner, _ = go(root, tmp_path)
    assert states(report)["tests"] == "not run" and code == 1 and "test files are not in" in report["steps"][2]["summary"]


# --- what blocks the facts step ------------------------------------------------------------------------------------------


def test_no_pyopencl_gives_the_install_command_of_the_opencl_facts_on_every_platform(root, tmp_path):
    """The offer `install_gpu_extra` is gone (P9-4d); the `opencl` facts hold the one install command."""
    data = machine(devices=[], pyopencl=None, problem="missing: No module named 'pyopencl'")
    code, report, runner, _ = go(root, tmp_path, data=data)
    assert code == 1 and runner.calls == []
    assert states(report) == {"facts": "not run", "parity": "not run", "tests": "not run", "bench": "not run"}
    assert report["next"][0] == data["opencl"]["install_command"] == "uv pip install --python /py -e '/code[gpu]'"
    assert report["next"][1] == f"{sys.executable} -m topopt gpu-check --out {tmp_path / 'out'}"
    assert "pyopencl is not usable: missing" in report["steps"][0]["summary"]


def test_pyopencl_that_loads_with_no_device(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, data=machine(devices=[], problem="no OpenCL device"))
    assert code == 1 and runner.calls == [] and "shows no device" in report["steps"][0]["summary"]
    assert "The other solvers work without one" in report["steps"][0]["detail"]


def cpu_machine():
    return machine(devices=[dev_dict(0, "pocl cpu", "CPU", True, "Portable Computing Language"),
                            dev_dict(1, "pocl cpu", "CPU", True, "Portable Computing Language", duplicate_of=0)], auto=0)


def test_only_a_cpu_device_is_the_no_gpu_path_and_device_names_the_smoke_test(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, data=cpu_machine())
    assert code == 1 and runner.calls == []
    assert "no OpenCL device that is not a CPU is visible" in report["steps"][0]["summary"]
    assert "1: pocl cpu (CPU, the same device as 0)" in report["steps"][0]["summary"]
    assert report["next"] == [f"{sys.executable} -m topopt gpu-check --out {tmp_path / 'out'} --device 0"]
    # with --device the same machine runs the steps (a smoke test on a CPU device)
    code, report, runner, _ = go(root, tmp_path, data=cpu_machine(), device="0", skip_bench=True)
    assert code == 0 and runner.steps() == ["parity", "tests"] and report["device_option"] == "0"
    assert "(named with --device 0)" in (tmp_path / "out" / "gpu-check.md").read_text()
    assert runner.calls[0]["cmd"][-2:] == ["--device", "0"]


def test_the_second_listing_of_a_cpu_device_counts_as_the_same_device_for_the_tests(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, data=cpu_machine(), device="1", skip_bench=True)
    assert code == 0 and states(report)["tests"] == "passed" and runner.calls[0]["cmd"][-1] == "1"


def test_a_device_that_is_not_the_auto_device_skips_the_tests(root, tmp_path):
    data = machine(devices=[dev_dict(0, "Real GPU", "GPU"), dev_dict(1, "A CPU", "CPU", True, "Other")], auto=0)
    code, report, runner, _ = go(root, tmp_path, data=data, device="CPU")
    assert code == 0 and runner.steps() == ["parity", "bench"]
    assert states(report)["tests"] == "skipped" and "take the device that auto takes" in report["steps"][2]["summary"]
    assert runner.calls[0]["cmd"][-1] == "1" and runner.calls[1]["cmd"][-1] == "1"


def test_a_device_that_fits_nothing_is_a_blocker_with_the_device_list(root, tmp_path):
    code, report, runner, _ = go(root, tmp_path, data=machine(), device="nope")
    assert code == 1 and runner.calls == [] and "'nope'" in report["steps"][0]["summary"] and "Apple M4" in report["steps"][0]["summary"]


def test_a_wheel_install_has_no_tools_and_gets_the_clone_command(tmp_path):
    wheel = tmp_path / "site-packages"
    (wheel / "topopt").mkdir(parents=True)
    code, report, runner, _ = go(wheel, tmp_path)
    assert code == 1 and runner.calls == []
    assert "gpu-check needs a checkout" in report["steps"][0]["summary"]
    assert report["next"][0] == "git clone https://github.com/chartergroup/topopt.git"
    assert report["next"][1].startswith("cd topopt && "), "the clone folder is named after the repo"
    assert any("[dev,mcp,gpu]" in c for c in report["next"]) and report["next"][-1].endswith("topopt gpu-check")


def test_the_commands_to_run_again_keep_the_options_of_the_user(root, tmp_path):
    """Review n4: the re-run command had dropped `--out`, `--skip-*`, the time limit and `--device`."""
    data = machine(devices=[], pyopencl=None, problem="missing: No module named 'pyopencl'")
    code, report, _runner, _ = go(root, tmp_path, data=data, device="Apple", skip_bench=True, step_timeout_s=30)
    again = report["next"][-1]
    assert f"--out {tmp_path / 'out'}" in again and "--skip-bench" in again and "--step-timeout-s 30" in again
    assert again.endswith("--device Apple") and "--skip-tests" not in again
    code, report, _runner, _ = go(root, tmp_path, data=cpu_machine(), skip_tests=True)
    assert report["next"] == [f"{sys.executable} -m topopt gpu-check --out {tmp_path / 'out'} --skip-tests --device 0"]


def test_a_wheel_install_with_a_cpu_device_gets_no_device_advice_that_fails_again(tmp_path):
    """Review n3: `--device 0` cannot help while tools/ is missing."""
    wheel = tmp_path / "site-packages"
    (wheel / "topopt").mkdir(parents=True)
    code, report, runner, _ = go(wheel, tmp_path, data=cpu_machine())
    assert code == 1 and report["next"][0] == "git clone https://github.com/chartergroup/topopt.git"
    assert not any("--device" in c for c in report["next"])


def test_the_case_that_auto_gives_comes_from_the_one_function_of_the_solver(root, tmp_path, monkeypatch):
    from topopt.gpu_mg import device as gpu_device
    code, report, _runner, _ = go(root, tmp_path)
    assert "the case that auto gives: C" in report["steps"][0]["detail"]               # Apple M4: no 64-bit float
    monkeypatch.setattr(gpu_device, "choose_case", lambda dev, requested="auto": "Z")
    code, report, _runner, _ = go(root, tmp_path)
    assert "the case that auto gives: Z" in report["steps"][0]["detail"]


def test_a_cpu_device_says_before_the_first_child_that_the_times_are_not_gpu_times(root, tmp_path):
    events, base = [], Runner()

    def runner(cmd, **kw):
        events.append(("child", cmd[1]))
        return base(cmd, **kw)
    gpu_check.run(out=tmp_path / "out", root=root, runner=runner, data=cpu_machine(), device="0", skip_bench=True,
                  say=lambda text="": events.append(("say", text)), have_module=lambda n: True)
    note = next(i for i, e in enumerate(events) if e[0] == "say" and "not GPU times" in e[1])
    assert note < next(i for i, e in enumerate(events) if e[0] == "child")


def test_an_error_in_the_facts_is_a_not_run_step(root, tmp_path, monkeypatch):
    from topopt import environment

    def broken(*a, **k):
        raise RuntimeError("scipy is broken")
    monkeypatch.setattr(environment, "environment_data", broken)
    said = []
    code = gpu_check.run(out=tmp_path / "out", root=root, runner=Runner(), data=None, say=said.append)
    report = json.loads((tmp_path / "out" / "gpu-check.json").read_text())
    assert code == 1 and "RuntimeError: scipy is broken" in report["steps"][0]["summary"]
    assert report["next"][0].endswith("-m topopt doctor") and states(report)["parity"] == "not run"


def test_no_command_in_the_report_has_a_trailing_comment(root, tmp_path):
    """zsh does not read `#` in an interactive shell: a pasted `cmd   # note` would pass the note as arguments."""
    for kw in ({"data": cpu_machine()}, {"data": machine(devices=[], pyopencl=None, problem="missing: x")},
               {"runner": Runner(parity=(1, "FAIL"))}, {"runner": Runner(tests=(1, TESTS_BAD))}, {"runner": Runner(bench=(1, "x"))}):
        _code, report, _runner, _ = go(root, tmp_path, **kw)
        assert report["next"] and not any(" #" in c for c in report["next"])


# --- real facts, real child processes ---------------------------------------------------------------------------------------


def test_the_facts_come_from_the_environment_data_of_this_machine(root, tmp_path, monkeypatch):
    """No fake dict: `environment_data()` runs, with a faked OpenCL state (no driver is loaded)."""
    fake_opencl(monkeypatch, "devices", [make_device(0, "Test GPU", "GPU", platform_name="Test")])
    said = []
    code = gpu_check.run(out=tmp_path / "out", root=root, runner=Runner(), say=said.append, have_module=lambda n: True)
    report = json.loads((tmp_path / "out" / "gpu-check.json").read_text())
    assert code == 0 and report["device"]["name"] == "Test GPU" and report["checkout"]["code_folder"] == str(REPO)


def test_the_command_without_pyopencl_prints_the_report_and_exits_1(tmp_path, monkeypatch, capsys):
    """The no-pyopencl path through the command line. The check is the real `environment_data()` with pyopencl missing."""
    fake_opencl(monkeypatch, "missing")
    code = cli.main(["gpu-check", "--out", str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert code == 1 and "NOT RUN: facts: pyopencl is not usable" in out and "# topopt gpu-check report" in out
    assert (tmp_path / "r" / "gpu-check.md").is_file() and (tmp_path / "r" / "gpu-check.json").is_file()
    assert "[gpu]" in json.loads((tmp_path / "r" / "gpu-check.json").read_text())["next"][0].replace("'", "")


def test_the_command_with_a_cpu_device_only_is_the_no_gpu_path(tmp_path, monkeypatch, capsys):
    fake_opencl(monkeypatch, "devices", [make_device(0, "pocl", "CPU", True, "Portable Computing Language")])
    assert cli.main(["gpu-check", "--out", str(tmp_path / "r")]) == 1
    assert "no OpenCL device that is not a CPU is visible" in capsys.readouterr().out


def test_run_child_reports_exit_codes_timeouts_and_missing_programs():
    ok = gpu_check.run_child([sys.executable, "-c", "print('hi')"])
    assert ok["returncode"] == 0 and ok["output"].strip() == "hi" and ok["timed_out"] is False
    bad = gpu_check.run_child([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"])
    assert bad["returncode"] == 3 and "out" in bad["output"] and "--- stderr ---" in bad["output"] and "err" in bad["output"]
    slow = gpu_check.run_child([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert slow["returncode"] is None and slow["timed_out"] is True and "--step-timeout-s" in slow["output"]
    gone = gpu_check.run_child(["/no/such/program"])
    assert gone["returncode"] is None and "FileNotFoundError" in gone["output"]


def test_run_child_does_not_fail_on_output_that_is_not_utf8():
    """Review n7: a driver or a compiler can print a byte that is not UTF-8."""
    res = gpu_check.run_child([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'ab\\xff\\xfecd\\n')"])
    assert res["returncode"] == 0 and res["output"].startswith("ab") and res["output"].strip().endswith("cd")


@pytest.mark.skipif(sys.platform == "win32", reason="process groups")
def test_a_time_limit_stops_the_child_and_what_the_child_started(tmp_path):
    """Review m5: `bench_gpu_mg.py` starts `topopt run`; the orphan must not go on after the time limit."""
    mark = tmp_path / "grandchild-lived"
    grandchild = f"import time, pathlib; time.sleep(2.5); pathlib.Path({str(mark)!r}).write_text('alive')"
    child = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
    res = gpu_check.run_child([sys.executable, "-c", child], timeout=1)
    assert res["timed_out"] is True
    time.sleep(3.5)                                  # the grandchild would have written its file by now
    assert not mark.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="process groups")
def test_ctrl_c_stops_the_child_and_what_the_child_started(tmp_path, monkeypatch):
    mark = tmp_path / "grandchild-lived"
    grandchild = f"import time, pathlib; time.sleep(2.5); pathlib.Path({str(mark)!r}).write_text('alive')"
    child = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
    real = gpu_check.subprocess.Popen.communicate
    calls = []

    def interrupt_once(self, *a, **k):
        if not calls:
            calls.append(1)
            time.sleep(0.5)                          # the child has started its own child by now
            raise KeyboardInterrupt
        return real(self, *a, **k)
    monkeypatch.setattr(gpu_check.subprocess.Popen, "communicate", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        gpu_check.run_child([sys.executable, "-c", child])
    time.sleep(3.5)
    assert not mark.exists()


# --- the readers of the child output ------------------------------------------------------------------------------------------


def test_parity_reader():
    info = gpu_check.parse_parity(PARITY_M4)
    assert info["verdict"] == "PASS" and info["rows"] == 2
    assert info["max_du"] == pytest.approx(2.90e-11) and info["max_dc"] == pytest.approx(2.71e-11)
    assert gpu_check.parse_parity("error: no device\n") == {"verdict": None, "max_du": None, "max_dc": None, "rows": 0}


def test_pytest_reader():
    assert gpu_check.parse_pytest(TESTS_M4)["summary"] == "98 passed, 10 skipped, 3 warnings in 27.10s"
    bad = gpu_check.parse_pytest(TESTS_BAD)
    assert bad["summary"] == "2 failed, 96 passed, 10 skipped in 30.00s" and len(bad["failed"]) == 2
    assert gpu_check.parse_pytest("collected 0 items\n\nno tests ran in 0.01s\n")["summary"] == "no tests ran in 0.01s"


def test_bench_reader(tmp_path):
    assert gpu_check.bench_facts(tmp_path) == {"summary": None, "markdown": None}
    (tmp_path / "bench_gpu_mg.json").write_text(json.dumps({"runs": [
        {"h": 1.0, "solver": "default", "status": "ok", "median_iter_s": 2.235},
        {"h": 1.0, "solver": "gpu-mg", "status": "ok", "median_iter_s": 0.439},
        {"h": 1.2, "solver": "gpu-mg", "status": "error"}]}))
    (tmp_path / "bench_gpu_mg.md").write_text(BENCH_MD)
    info = gpu_check.bench_facts(tmp_path)
    assert info["summary"] == "1.2 mm gpu-mg error; 1.0 mm 0.439 s per iteration, 5.09x the default solver (2.235 s)"
    assert info["markdown"].startswith("- machine: test") and "# gpu-mg benchmark" not in info["markdown"]


# --- the calibration of the time model (P2a) -----------------------------------------------------------------------------


def test_the_bench_step_saves_the_constants_of_the_device_for_check(root, tmp_path, state_folder):
    from topopt.gpu_mg import check as gpu_time
    code, report, runner, said = go(root, tmp_path, runner=Runner(calibratable=True))
    assert code == 0 and states(report)["bench"] == "passed"
    file = state_folder / "gpu-constants.json"
    assert report["calibration_file"] == str(file) and file.is_file()
    rec = report["calibration"]
    assert rec["device"]["name"] == "Apple M4" and rec["case"] == "C" and rec["columns_per_launch"] == 2
    # c: the median of solve_s / (iterations * dofs) is 0.3 / (100 * 100,000) = 30 ns for both sizes; host: (0.40 - 0.30) / 100,000 = 1 us
    assert rec["c_s_per_dof_iteration"] == pytest.approx(30e-9) and rec["host_s_per_dof"] == pytest.approx(1.0e-6)
    bench = next(s for s in report["steps"] if s["id"] == "bench")
    assert "calibration for `topopt check`: c = 30.0 ns" in bench["detail"] and f"Saved in {file}" in bench["detail"]
    assert any("Saved in" in line for line in said) and f"Saved in {file}" in (tmp_path / "out" / "gpu-check.md").read_text()
    est = gpu_time.estimate_seconds_gpu_mg(200_000, 2, 50, device=machine()["opencl"]["devices"][0], case="C")
    assert est.basis == "calibration" and est.constants.c2_s == pytest.approx(30e-9)         # `topopt check` reads the file


def test_a_bench_report_that_cannot_calibrate_leaves_no_file_and_says_why(root, tmp_path, state_folder):
    code, report, _, _ = go(root, tmp_path)                    # the fake bench report has no device and no load-case count
    assert code == 0 and report["calibration"] is None and report["calibration_file"] is None
    bench = next(s for s in report["steps"] if s["id"] == "bench")
    assert "no calibration for `topopt check`: the bench report names no OpenCL device" in bench["detail"]
    assert not state_folder.exists()


def test_a_state_folder_that_cannot_be_written_is_not_an_error(root, tmp_path, monkeypatch):
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setenv("TOPOPT_STATE_DIR", str(blocker / "state"))
    code, report, _, _ = go(root, tmp_path, runner=Runner(calibratable=True))
    assert code == 0 and report["result"] == "PASS" and report["calibration_file"] is None and report["calibration"]
    bench = next(s for s in report["steps"] if s["id"] == "bench")
    assert "Not saved" in bench["detail"] and "so `check` does not use it" in bench["detail"]


def test_a_run_that_skips_the_bench_does_not_touch_the_calibration(root, tmp_path, state_folder):
    code, report, runner, _ = go(root, tmp_path, runner=Runner(calibratable=True), skip_bench=True)
    assert code == 0 and "bench" not in runner.steps() and report["calibration"] is None and not state_folder.exists()
