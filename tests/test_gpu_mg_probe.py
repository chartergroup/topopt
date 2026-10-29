"""Tests of the child process of the OpenCL probe of `settings.solver: auto` (#353, CONTRACTS.md section 20).

No OpenCL and no pyopencl: the child command is replaced by a small script, and the cache tests use fake devices in this process.
A test that calls the product in a thread joins it with a time limit, so a fault cannot hang the suite. The test with the real child is
in `test_gpu_mg_probe_pocl.py`.
"""
import json
import os
import sys
import subprocess
import threading
import time

import pytest

from helpers import in_process_probe, make_device
from test_optimize import make_case
from topopt import cli, solver_choice
from topopt.gpu_mg import device, probe

REAL_PYOPENCL_INSTALLED = probe.pyopencl_installed      # the fixture below replaces it; N3 tests the real one

LIMIT = 1.0         # the monkeypatched time limit of the child, seconds
MARGIN = 8.0        # what the call may take above the limit: the start of the Python of the script and the kill


@pytest.fixture(autouse=True)
def own_answers(monkeypatch, tmp_path):
    """The answers of the probe are new for each test, `auto` looks at the child (the session sets `TOPOPT_AUTO_GPU=0`), the limit is small."""
    monkeypatch.setattr(probe, "_ANSWERS", {})
    monkeypatch.delenv("TOPOPT_AUTO_GPU", raising=False)
    monkeypatch.setattr(probe, "PROBE_TIMEOUT_S", LIMIT)
    monkeypatch.setattr(probe, "run_probe", probe.run_child)
    monkeypatch.setattr(probe, "pyopencl_installed", lambda: True)      # the fake child scripts also run on a machine without pyopencl


def script(monkeypatch, tmp_path, body: str):
    """Make the child command a Python script with `body`. The script gets the file `pid.txt` of the test folder as argument 1."""
    path = tmp_path / "p353_child.py"
    path.write_text(body)
    monkeypatch.setattr(probe, "child_command", lambda d, c, pick: [sys.executable, "-P", str(path), str(tmp_path / "pid.txt")])
    return tmp_path / "pid.txt"


HANG = "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(300)\n"


def in_thread(fn, *args, wait=LIMIT + MARGIN + 10):
    """Call `fn` in a thread and return `(result, seconds)`. A call that does not return fails the test."""
    box = {}

    def target():
        t0 = time.monotonic()
        try:
            box["result"] = fn(*args)
        except BaseException as e:      # shown by the assert below
            box["error"] = e
        box["seconds"] = time.monotonic() - t0

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(wait)
    assert not t.is_alive(), "the call did not return"
    assert "error" not in box, box["error"]
    return box["result"], box["seconds"]


def gone(pid_file, wait=15.0) -> bool:
    """True when the child of the hang script has ended and has been reaped (no process with its pid)."""
    deadline = time.monotonic() + wait
    pid = int(pid_file.read_text())
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_a_child_that_does_not_return_gives_the_cpu_rule_with_the_limit_in_the_reason(monkeypatch, tmp_path):
    pid_file = script(monkeypatch, tmp_path, HANG)
    choice, seconds = in_thread(solver_choice.run_solver, "auto")
    assert choice.solver == choice.cpu and not choice.auto_gpu and choice.fallback is None and choice.device is None
    assert "did not finish in 1 s" in choice.reason and "child process was stopped" in choice.reason
    assert "settings.solver" in choice.reason and "TOPOPT_AUTO_GPU=0" in choice.reason
    assert LIMIT <= seconds < LIMIT + MARGIN
    assert gone(pid_file), "the child process of the probe is still there"


def test_the_caller_does_not_wait_for_the_child_to_end(monkeypatch, tmp_path):
    """A child in a driver call can be in a wait that a kill does not end at once: the reaper thread waits, not the caller."""
    waited = []
    real_wait = probe.subprocess.Popen.wait
    monkeypatch.setattr(probe.subprocess.Popen, "wait", lambda self, *a, **k: (time.sleep(3), waited.append(1), real_wait(self, *a, **k))[2])
    script(monkeypatch, tmp_path, HANG)
    _answer, seconds = in_thread(probe.probe, "auto", "auto", True)
    assert LIMIT <= seconds < LIMIT + 2.5, seconds       # the wait of 3 s is in the thread of the reaper
    deadline = time.monotonic() + 10
    while not waited and time.monotonic() < deadline:
        time.sleep(0.1)
    assert waited


def test_check_data_returns_when_the_child_does_not_return(monkeypatch, tmp_path):
    pid_file = script(monkeypatch, tmp_path, HANG)
    data, seconds = in_thread(cli.check_data, str(make_case(tmp_path, h=2.0, iterations=3)), "auto")
    assert data["solver"]["choice"]["reason"].count("did not finish in 1 s") == 1 and data["solver"]["resolved"] == data["solver"]["cpu_solver"]
    assert "did not finish in 1 s" in data["opencl"]["problem"] and data["opencl"]["devices"] == []
    assert seconds < LIMIT + MARGIN + 6      # one child for the pick and the facts: the limit is paid once
    assert gone(pid_file)


def test_the_opencl_facts_after_a_timeout_say_the_device_list_did_not_finish(monkeypatch, tmp_path):
    pid_file = script(monkeypatch, tmp_path, HANG)
    f, seconds = in_thread(probe.facts, "auto", "auto")
    assert "did not finish in 1 s" in f["problem"] and f["devices"] == [] and f["chosen_device"] is None and f["install_command"]
    assert seconds < LIMIT + MARGIN
    assert gone(pid_file)


def test_environment_returns_when_the_child_does_not_return(monkeypatch, tmp_path):
    from topopt.environment import environment_data
    pid_file = script(monkeypatch, tmp_path, HANG)
    data, seconds = in_thread(environment_data, None, wait=2 * (LIMIT + MARGIN) + 30)
    assert "did not finish in 1 s" in data["opencl"]["problem"]
    assert data["gpu_solver"]["used"] is False and "did not finish in 1 s" in data["gpu_solver"]["why"]
    assert "gave no list" in data["gpu_solver"]["to_use_the_gpu"] and "TOPOPT_AUTO_GPU=0" in data["gpu_solver"]["to_use_the_gpu"]
    assert gone(pid_file)


def test_the_mcp_tools_check_problem_and_environment_return_when_the_child_does_not_return(monkeypatch, tmp_path):
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    pid_file = script(monkeypatch, tmp_path, HANG)
    yaml_file = make_case(tmp_path, h=2.0, iterations=3)
    r, _s = in_thread(server.check_problem, yaml_file.name)
    assert r["ok"] is True and "did not finish in 1 s" in r["solver"]["choice"]["reason"]
    e, _s = in_thread(server.environment, wait=2 * (LIMIT + MARGIN) + 30)
    assert "did not finish in 1 s" in e["opencl"]["problem"]
    assert gone(pid_file)


def test_the_disabled_variable_waits_for_no_child(monkeypatch, tmp_path):
    """`TOPOPT_AUTO_GPU=0` is the way out with no wait: the pick makes no child."""
    monkeypatch.setenv("TOPOPT_AUTO_GPU", "0")
    monkeypatch.setattr(probe, "run_probe", lambda *a: pytest.fail("a child was started"))
    choice = solver_choice.run_solver("auto")
    assert choice.solver == choice.cpu and "TOPOPT_AUTO_GPU=0" in choice.reason


ANSWER = {"facts": {"pyopencl": "1", "problem": None, "install_command": "x", "platforms": [], "devices": [], "auto_device": None,
                    "auto_case": None, "chosen_device": None}, "device": None, "case": None, "reason": "no OpenCL device on this machine",
          "self_test": None, "timed_out": False}


def test_driver_text_on_stdout_does_not_break_the_answer(monkeypatch, tmp_path):
    body = (f"import sys, json\nsys.stdout.write('driver says hello\\n' + 'a part of a line')\n"
            f"sys.stdout.write('\\n{probe.PREFIX}{{broken\\n')\n"      # an earlier line with the prefix that is not JSON
            f"sys.stdout.write('\\n{probe.PREFIX}' + json.dumps({ANSWER!r}) + '\\n')\n"
            f"sys.stdout.write('driver says goodbye\\n')\n")
    script(monkeypatch, tmp_path, body)
    a = probe.probe("auto", "auto", True)
    assert a["reason"] == "no OpenCL device on this machine" and a["timed_out"] is False and a["facts"]["pyopencl"] == "1"


def test_a_child_that_ends_with_an_error_gives_the_cpu_rule_with_the_reason(monkeypatch, tmp_path):
    script(monkeypatch, tmp_path, "import sys\nsys.stderr.write('Segmentation fault in libOpenCL\\n')\nsys.exit(3)\n")
    choice = solver_choice.run_solver("auto")
    assert choice.solver == choice.cpu and not choice.auto_gpu
    assert "exit code 3" in choice.reason and "no answer" in choice.reason and "Segmentation fault in libOpenCL" in choice.reason
    f = probe.facts("auto", "auto")
    assert "exit code 3" in f["problem"] and f["devices"] == []


def test_a_child_that_prints_no_answer_gives_the_cpu_rule_with_the_reason(monkeypatch, tmp_path):
    script(monkeypatch, tmp_path, "print('only noise')\n")
    choice = solver_choice.run_solver("auto")
    assert choice.solver == choice.cpu and "exit code 0" in choice.reason and "no answer" in choice.reason


def test_a_child_that_cannot_start_gives_the_cpu_rule_with_the_reason(monkeypatch):
    monkeypatch.setattr(probe, "child_command", lambda d, c, pick: ["/nonexistent/python", "-P"])
    choice = solver_choice.run_solver("auto")
    assert choice.solver == choice.cpu and "did not start" in choice.reason


def test_the_answer_of_the_child_is_kept_for_the_process(monkeypatch, tmp_path):
    counter = tmp_path / "count.txt"
    body = (f"import json\nopen({str(counter)!r}, 'a').write('x')\nprint('{probe.PREFIX}' + json.dumps({ANSWER!r}))\n")
    script(monkeypatch, tmp_path, body)
    solver_choice.run_solver("auto")
    probe.facts("auto", "auto")           # an answer with a pick also serves the facts
    solver_choice.run_solver("auto")
    assert counter.read_text() == "x"
    device.forget_devices()
    solver_choice.run_solver("auto")
    assert counter.read_text() == "xx"


def test_an_explicit_gpu_mg_makes_no_child(monkeypatch, tmp_path):
    """The user asked for the GPU: the facts of `check` come from this process, not from a child."""
    monkeypatch.setattr(probe, "run_probe", lambda *a: pytest.fail("a child was started"))
    monkeypatch.setattr(device, "facts", lambda d="auto": {"pyopencl": "x", "devices": [], "chosen_device": None})

    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "pyopencl" else real(name, *a))

    class S:
        solver, gpu_device, gpu_case = "gpu-mg", "auto", "auto"
    assert cli.opencl_facts(S())["pyopencl"] == "x"


# --- the cache of a passed self-test ------------------------------------------------------------------------------------------------------


PASSED = {"passed": True, "rel_diff": 3.0e-11, "seconds": 9.8, "free_dofs": 3888, "tol": 1e-8, "max_diff": 1e-6, "device": "Test GPU", "case": "C",
          "error": None}


@pytest.fixture
def runs(monkeypatch):
    """`selftest.run` is faked and counted; the cache file of the test is in its own state folder (tests/conftest.py)."""
    from topopt.gpu_mg import selftest
    calls = []

    def fake(dev, case, tol, max_diff):
        calls.append((dev.name, case))
        return dict(box["result"])
    box = {"result": PASSED}
    monkeypatch.setattr(selftest, "run", fake)
    device.forget_devices()
    yield calls, box
    device.forget_devices()


def gpu(driver="1.0", name="Test GPU"):
    return make_device(0, name, "GPU", platform_name="Test Platform", driver=driver)


def fresh_process():
    """What a new process knows: nothing but the cache file."""
    device.forget_devices()


def test_a_pass_is_written_and_a_new_process_runs_no_self_test(runs):
    calls, _box = runs
    first = device.self_test(gpu(), "C")
    assert first["passed"] and first["cached"] is False and first["tested_at"] and first["cache_note"] is None
    fresh_process()
    second = device.self_test(gpu(), "C")
    assert len(calls) == 1 and second["cached"] is True and second["passed"] and second["tested_at"] == first["tested_at"]
    assert second["seconds"] == 9.8 and second["rel_diff"] == 3.0e-11
    data = json.loads(device.selftest_path().read_text())
    assert data["schema"] == 1 and len(data["passes"]) == 1
    key = data["passes"][0]["key"]
    assert key["driver"] == "1.0" and key["case"] == "C" and key["tol"] == 1e-8 and key["max_diff"] == 1e-6 and key["topopt"] and key["platform"] == "Test Platform"
    assert device.selftest_path().parent.name == os.path.basename(os.environ["TOPOPT_STATE_DIR"])


def test_the_line_says_cached_result_of_the_utc_time_in_place_of_the_seconds(runs):
    device.self_test(gpu(), "C")
    fresh_process()
    t = device.self_test(gpu(), "C")
    line = solver_choice.SolverChoice("auto", "gpu-mg", "amg", fallback="amg", device=gpu().facts(), case="C", self_test=t).line()
    assert f"(cached result of {t['tested_at']})" not in line      # no brackets: the text is a clause
    assert f"cached result of {t['tested_at']}" in line and "9.8 s" not in line and "self-test passed" in line
    fresh = solver_choice.SolverChoice("auto", "gpu-mg", "amg", fallback="amg", device=gpu().facts(), case="C", self_test=PASSED).line()
    assert "9.8 s" in fresh and "cached" not in fresh


@pytest.mark.parametrize("change", ["driver", "case", "version", "name", "tol", "topopt", "code"])
def test_another_key_field_tests_again(runs, monkeypatch, change):
    calls, _box = runs
    device.self_test(gpu(), "C")
    fresh_process()
    dev, case = gpu(), "C"
    if change == "driver":
        dev = gpu(driver="2.0")
    elif change == "case":
        case = "B"
    elif change == "version":
        dev = gpu(); dev.version = "OpenCL 3.0"
    elif change == "name":
        dev = gpu(name="Other GPU")
    elif change == "tol":
        monkeypatch.setattr(device, "SELFTEST_MAX_DIFF", 1e-7)
    elif change == "code":
        monkeypatch.setattr(device, "code_hash", lambda: "another" * 8)
    else:
        import topopt
        monkeypatch.setattr(topopt, "__version__", "99.0")
    r = device.self_test(dev, case)
    assert len(calls) == 2 and r["cached"] is False


def test_a_failure_and_an_error_are_not_written(runs):
    calls, box = runs
    box["result"] = {**PASSED, "passed": False, "rel_diff": 4e-4}
    assert device.self_test(gpu(), "C")["passed"] is False
    box["result"] = {**PASSED, "passed": False, "rel_diff": None, "error": "Error: kernel build failed"}
    fresh_process()
    assert device.self_test(gpu(), "C")["error"]
    assert not device.selftest_path().exists()
    box["result"] = PASSED
    fresh_process()
    device.self_test(gpu(), "C")
    assert len(calls) == 3 and device.selftest_path().exists()


def test_a_timeout_is_not_written_as_a_pass(monkeypatch, tmp_path):
    script(monkeypatch, tmp_path, HANG)
    in_thread(solver_choice.run_solver, "auto")
    assert not device.selftest_path().exists()


@pytest.mark.parametrize("text", ["not json {", json.dumps({"schema": 9, "passes": []}), json.dumps({"schema": 1, "passes": 5}), "[]", ""])
def test_a_bad_file_gives_a_self_test_and_a_fact(runs, text):
    calls, _box = runs
    path = device.selftest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    r = device.self_test(gpu(), "C")
    assert len(calls) == 1 and r["passed"] and (text == "" or "not a cache file of this version" in r["cache_note"])
    # the pass replaced the bad file
    fresh_process()
    assert device.self_test(gpu(), "C")["cached"] is True and len(calls) == 1


def test_a_bad_file_shows_the_fact_in_the_line(runs):
    path = device.selftest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json {")
    t = device.self_test(gpu(), "C")
    # the pass rewrites the file, so the note of the bad file is the note of the result
    assert t["cache_note"] is not None and "not a cache file" in t["cache_note"]
    line = solver_choice.SolverChoice("auto", "gpu-mg", "amg", fallback="amg", device=gpu().facts(), case="C", self_test=t).line()
    assert "the self-test cache was not used: " in line


def test_a_file_that_cannot_be_written_gives_a_fact_not_an_error(runs, monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    monkeypatch.setenv("TOPOPT_STATE_DIR", str(blocker / "state"))
    t = device.self_test(gpu(), "C")
    assert t["passed"] and t["cached"] is False and "cannot be written" in t["cache_note"]


def test_a_file_that_cannot_be_read_gives_a_fact_not_an_error(runs, monkeypatch):
    path = device.selftest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()            # a folder where the file should be: reading it is an OSError
    t = device.self_test(gpu(), "C")
    assert t["passed"] and "cannot be read" in t["cache_note"]


def test_the_driver_is_a_fact_of_the_device(monkeypatch):
    class D:
        name, type, double_fp_config, driver_version = "x", 4, 1, "24.1.0"
    assert device.DeviceInfo(0, "p", "x", "GPU", True, driver="24.1.0").facts()["driver"] == "24.1.0"


def test_run_solver_uses_the_cache_through_the_in_process_probe(runs, monkeypatch):
    """Through the seam: the fake world of the test, the real `self_test` with a fake `selftest.run`."""
    calls, _box = runs
    in_process_probe(monkeypatch)
    monkeypatch.setattr(device, "pyopencl_problem", lambda: None)
    monkeypatch.setattr(device, "enumerate_devices", lambda: [gpu()])
    monkeypatch.setattr(solver_choice, "pypardiso_available", lambda: False)
    c1 = solver_choice.run_solver("auto")
    assert c1.solver == "gpu-mg" and c1.self_test["cached"] is False
    fresh_process()
    in_process_probe(monkeypatch)
    c2 = solver_choice.run_solver("auto")
    assert c2.solver == "gpu-mg" and c2.self_test["cached"] is True and len(calls) == 1


def test_the_code_hash_follows_the_bytes_of_the_py_files_of_the_package(tmp_path):
    (tmp_path / "b.py").write_text("x = 1\n")
    (tmp_path / "a.py").write_text("y = 2\n")
    (tmp_path / "notes.txt").write_text("not code")
    first = device._code_hash.__wrapped__(str(tmp_path))
    assert first == device._code_hash.__wrapped__(str(tmp_path))
    (tmp_path / "notes.txt").write_text("other")
    assert device._code_hash.__wrapped__(str(tmp_path)) == first          # only *.py counts
    (tmp_path / "a.py").write_text("y = 3\n")
    assert device._code_hash.__wrapped__(str(tmp_path)) != first
    assert len(device.code_hash()) == 64 and device.selftest_key(gpu(), "C")["code"] == device.code_hash()


def test_with_the_variable_at_0_the_facts_start_no_child_and_say_not_asked(monkeypatch):
    monkeypatch.setenv("TOPOPT_AUTO_GPU", "0")
    monkeypatch.setattr(probe, "run_probe", lambda *a: pytest.fail("a child was started"))
    f = probe.facts("auto", "auto")
    assert f["problem"] == "not asked: TOPOPT_AUTO_GPU=0 is set in the environment" and f["devices"] == [] and f["chosen_device"] is None
    assert f["pyopencl"] == probe.pyopencl_version()
    from topopt.environment import environment_data
    data = environment_data()
    assert data["opencl"]["problem"].startswith("not asked") and data["gpu_solver"]["to_use_the_gpu"] is None


def test_with_the_variable_at_0_environment_does_not_say_that_an_explicit_gpu_mg_cannot_run(monkeypatch):
    """The variable keeps the device list unasked. It does not change an explicit `settings.solver: gpu-mg`, so the entry must not say "cannot run"."""
    monkeypatch.setenv("TOPOPT_AUTO_GPU", "0")
    monkeypatch.setattr(probe, "run_probe", lambda *a: pytest.fail("a child was started"))
    from topopt.environment import environment_data
    entry = environment_data()["solver_setting"]["gpu-mg"]
    assert entry["resolved"] is None and "cannot run" not in entry["error"]
    assert "not checked" in entry["error"] and "TOPOPT_AUTO_GPU=0" in entry["error"] and "settings.solver: gpu-mg" in entry["error"]


def test_the_child_environment_has_the_folder_of_the_topopt_package_first(monkeypatch):
    import topopt
    from pathlib import Path
    folder = str(Path(topopt.__file__).resolve().parent.parent)
    monkeypatch.setenv("PYTHONPATH", "/old/one" + os.pathsep + "/old/two")
    assert probe.child_environment()["PYTHONPATH"] == folder + os.pathsep + "/old/one" + os.pathsep + "/old/two"
    monkeypatch.delenv("PYTHONPATH")
    assert probe.child_environment()["PYTHONPATH"] == folder
    seen = {}
    real = probe.subprocess.Popen

    def spy(cmd, **kw):
        seen["env"] = kw.get("env")
        return real([sys.executable, "-P", "-c", "print('x')"], **{k: v for k, v in kw.items() if k != "env"})
    monkeypatch.setattr(probe.subprocess, "Popen", spy)
    probe.run_child("auto", "auto", True)
    assert seen["env"]["PYTHONPATH"].split(os.pathsep)[0] == folder


# --- fix round 2 ------------------------------------------------------------------------------------------------------------------------


FULL = {"facts": {"pyopencl": "1", "problem": None, "install_command": "x", "platforms": ["P"], "devices": [{"index": 0}], "auto_device": 0,
                  "auto_case": "B", "chosen_device": 0},
        "device": {"name": "G", "platform": "P", "kind": "GPU"}, "case": "B", "reason": None,
        "self_test": {"passed": True, "rel_diff": 1e-11, "seconds": 1.0, "tol": 1e-8, "max_diff": 1e-6, "error": None}, "timed_out": False}
FULL_LINE = probe.PREFIX + json.dumps(FULL)


def test_a_complete_answer_is_used_when_the_child_dies_by_a_signal_and_the_exit_code_is_a_fact(monkeypatch, tmp_path):
    script(monkeypatch, tmp_path, f"import os, signal\nprint({FULL_LINE!r}, flush=True)\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    a = probe.probe("auto", "auto", True)
    assert a["device"]["name"] == "G" and a["self_test"]["passed"] is True and a["child_exit_code"] != 0 and a["timed_out"] is False


def test_the_exit_code_of_a_child_that_answered_is_in_the_facts_that_check_and_environment_show(monkeypatch, tmp_path):
    script(monkeypatch, tmp_path, f"import sys\nprint({FULL_LINE!r}, flush=True)\nsys.exit(3)\n")
    assert probe.facts("auto", "auto")["child_exit_code"] == 3
    from topopt.environment import environment_data
    assert environment_data()["opencl"]["child_exit_code"] == 3


def test_the_last_line_that_parses_is_the_answer_whatever_comes_after_it(monkeypatch, tmp_path):
    script(monkeypatch, tmp_path, f"print({FULL_LINE!r})\nprint('TOPOPT-PROBE {{oops')\n")
    a = probe.probe("auto", "auto", True)
    assert a["device"]["name"] == "G" and "child_exit_code" not in a and "child_exit_code" not in a["facts"]


def test_no_line_that_parses_gives_a_reason_with_the_exit_code(monkeypatch, tmp_path):
    script(monkeypatch, tmp_path, "print('TOPOPT-PROBE {oops')\nprint('TOPOPT-PROBE [1]')\nimport sys\nsys.exit(4)\n")
    choice = solver_choice.run_solver("auto")
    assert "exit code 4" in choice.reason and "no answer that can be read" in choice.reason and choice.solver == choice.cpu


def test_without_pyopencl_no_child_starts(monkeypatch):
    monkeypatch.setattr(probe, "pyopencl_installed", lambda: False)
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *a, **k: pytest.fail("a child was started"))
    choice = solver_choice.run_solver("auto")
    assert choice.solver == choice.cpu and choice.reason.startswith("pyopencl is not usable here (missing: ")
    assert probe.facts("auto", "auto")["problem"].startswith("missing: ") and probe.facts("auto", "auto")["pyopencl"] is None


def test_the_timeout_texts_say_that_a_new_process_asks_again():
    problem, reason = probe.timeout_texts()
    assert "a new process asks again" in problem and "MCP server asks again after a restart" in problem
    assert "A new process tests again" in reason and "MCP server tests again after a restart" in reason


def test_the_child_command_has_p_right_after_the_interpreter():
    cmd = probe.child_command("auto", "auto", True)
    assert cmd[0] == sys.executable and cmd[1] == "-P" and cmd[2:4] == ["-m", probe.MODULE]


def test_the_topopt_folder_is_first_on_the_child_path_unless_it_is_a_site_packages_folder(monkeypatch):
    import topopt
    from pathlib import Path
    folder = Path(topopt.__file__).resolve().parent.parent
    monkeypatch.setenv("PYTHONPATH", "/old")
    monkeypatch.setattr(probe, "_site_folders", lambda: set())
    assert probe.child_environment()["PYTHONPATH"] == str(folder) + os.pathsep + "/old"
    monkeypatch.setattr(probe, "_site_folders", lambda: {folder})
    assert probe.child_environment()["PYTHONPATH"] == "/old"
    monkeypatch.delenv("PYTHONPATH")
    assert "PYTHONPATH" not in probe.child_environment()


def test_check_with_solver_auto_over_a_gpu_mg_yaml_uses_the_child(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "pyopencl" else real(name, *a))
    monkeypatch.setattr(probe, "facts", lambda d="auto", c="auto": {"from": "child"})
    monkeypatch.setattr(device, "facts", lambda d="auto": {"from": "process"})

    class S:
        solver, gpu_device, gpu_case = "gpu-mg", "auto", "auto"
    assert cli.opencl_facts(S()) == {"from": "process"}
    assert cli.opencl_facts(S(), "auto") == {"from": "child"}
    assert cli.opencl_facts(S(), "gpu-mg") == {"from": "process"}


@pytest.mark.parametrize("variable", [None, "0"])
def test_environment_asks_no_driver_in_the_process_when_the_driver_hangs(tmp_path, variable):
    """A pyopencl whose platform list never returns (found on the path of the process and of the child): `environment_data` returns."""
    fake = tmp_path / "fakecl" / "pyopencl"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("import time\nVERSION_TEXT = 'fake'\nclass Error(Exception):\n    pass\ndef get_platforms():\n    time.sleep(1000)\n")
    code = ("import json, sys, time\n"
            "from topopt.gpu_mg import probe\n"
            "probe.PROBE_TIMEOUT_S = 2\n"
            "from topopt.environment import environment_data\n"
            "t0 = time.monotonic()\n"
            "d = environment_data()\n"
            "print('RESULT ' + json.dumps({'seconds': time.monotonic() - t0, 'in_parent': 'pyopencl' in sys.modules, 'problem': d['opencl']['problem'],\n"
            "                              'gpu_mg': d['solver_setting']['gpu-mg']}))\n")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(probe.__file__))))
    env = {k: v for k, v in os.environ.items() if k != "TOPOPT_AUTO_GPU"}
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path / "fakecl"), repo])
    env["TOPOPT_STATE_DIR"] = str(tmp_path / "state")
    if variable is not None:
        env["TOPOPT_AUTO_GPU"] = variable
    r = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(next(line for line in r.stdout.splitlines() if line.startswith("RESULT "))[len("RESULT "):])
    assert out["in_parent"] is False
    if variable is None:
        assert "did not finish in 2 s" in out["problem"] and out["seconds"] < 30 and out["gpu_mg"]["resolved"] is None
    else:
        assert out["problem"] == "not asked: TOPOPT_AUTO_GPU=0 is set in the environment" and out["seconds"] < 20
        assert "TOPOPT_AUTO_GPU=0" in out["gpu_mg"]["error"]


def test_the_real_pyopencl_check_finds_the_package_without_importing_it(monkeypatch):
    """N3: the other tests replace `pyopencl_installed`; this one runs the real function, with pyopencl found and with it not found."""
    import importlib.util
    before = "pyopencl" in sys.modules
    assert REAL_PYOPENCL_INSTALLED() == (importlib.util.find_spec("pyopencl") is not None)
    assert ("pyopencl" in sys.modules) == before
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "pyopencl" else real(name, *a, **k))
    assert REAL_PYOPENCL_INSTALLED() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object() if name == "pyopencl" else real(name, *a, **k))
    assert REAL_PYOPENCL_INSTALLED() is True


def test_the_real_site_folders_hold_the_site_packages_folder_of_this_python():
    """N3: the other tests replace `_site_folders`; this one runs the real function."""
    import site
    from pathlib import Path
    folders = probe._site_folders()
    assert folders and all(isinstance(f, Path) and f == f.resolve() for f in folders)
    assert Path(site.getsitepackages()[0]).resolve() in folders
