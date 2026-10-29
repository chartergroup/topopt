"""Tests for an x86_64 Python on a Mac (an Intel Mac, or Rosetta on Apple silicon), V5.

The platform, the machine type and the Rosetta check (`solver_choice._sysctl_proc_translated`) are
faked, so every test runs on every platform. No test here needs a Mac, Accelerate or a real pypardiso.
The fixes under test: the kind of Python (`mac_python_kind`), the hint, the user error for an explicit
`cholesky`, the facts of `environment`, and that macOS arm64 and Linux x86_64 do not change.
"""
import json
import platform
import re
import sys
import types

import pytest
import yaml

from helpers import fake_shim
from test_optimize import make_case
from topopt import cli, solver_choice
from topopt.cli import main
from topopt.config import UserError
from topopt.environment import environment_data
from topopt.optimize import make_solver


@pytest.fixture
def fake_mac(monkeypatch):
    """`fake_mac("x86_64", "1")` fakes a Mac: the machine type and the text of `sysctl.proc_translated`.

    The second value is "1" (Rosetta), "0" or "" (the key does not exist: an Intel Mac), or None
    (the command did not run). pypardiso is not installed. The return value counts the calls.
    """
    def set_mac(machine, translated=None, plat="darwin"):
        calls = []

        def probe():
            calls.append(1)
            return translated
        monkeypatch.setattr(sys, "platform", plat)
        monkeypatch.setattr(platform, "machine", lambda: machine)
        monkeypatch.setattr(solver_choice, "_sysctl_proc_translated", probe)

        def no_pypardiso():
            raise ModuleNotFoundError("No module named 'pypardiso'", name="pypardiso")
        monkeypatch.setattr(solver_choice, "import_pypardiso", no_pypardiso)
        return calls
    return set_mac


# --- the kind of Python ---------------------------------------------------------------------------


@pytest.mark.parametrize("plat, machine, translated, kind", [
    ("darwin", "arm64", "0", "arm64"),
    ("darwin", "x86_64", "1", "rosetta"),       # Apple silicon, translated
    ("darwin", "x86_64", "", "intel"),          # macOS before 11 has no key: the command fails
    ("darwin", "x86_64", "0", "intel"),         # an Intel Mac with macOS 11 or newer: not translated
    ("darwin", "x86_64", None, "unknown"),      # the command did not run
    ("linux", "x86_64", "1", None),
    ("linux", "aarch64", None, None),
])
def test_mac_python_kind(fake_mac, plat, machine, translated, kind):
    fake_mac(machine, translated, plat)
    assert solver_choice.mac_python_kind() == kind
    assert solver_choice.is_x86_mac() is (kind in ("rosetta", "intel", "unknown"))


def test_the_rosetta_check_runs_only_for_an_x86_python_on_a_mac(fake_mac):
    for plat, machine in (("darwin", "arm64"), ("linux", "x86_64")):
        calls = fake_mac(machine, "1", plat)
        solver_choice.mac_python_kind()
        solver_choice.platform_hint()
        assert calls == []                              # no command on macOS arm64 or on Linux
    calls = fake_mac("x86_64", "1")
    solver_choice.mac_python_kind()
    assert calls == [1]


@pytest.mark.parametrize("stdout, code, expect", [("1\n", 0, "1"), ("0\n", 0, "0"), ("", 1, "")])
def test_sysctl_probe_reads_the_command(monkeypatch, stdout, code, expect):
    """The real function, with a fake `subprocess.run`: the command is read-only and has a time limit."""
    seen = []

    def run(cmd, **kw):
        seen.append((cmd, kw))
        return types.SimpleNamespace(stdout=stdout, returncode=code)
    monkeypatch.setattr(solver_choice.subprocess, "run", run)
    assert solver_choice._sysctl_proc_translated() == expect
    assert seen[0][0] == ["/usr/sbin/sysctl", "-n", "sysctl.proc_translated"] and seen[0][1]["timeout"] > 0


def test_sysctl_probe_uses_the_full_path_first_then_the_bare_name(monkeypatch):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd[0])
        if cmd[0] == "/usr/sbin/sysctl":
            raise FileNotFoundError(cmd[0])
        return types.SimpleNamespace(stdout="1\n", returncode=0)
    monkeypatch.setattr(solver_choice.subprocess, "run", run)
    assert solver_choice._sysctl_proc_translated() == "1" and seen == ["/usr/sbin/sysctl", "sysctl"]


def test_sysctl_probe_without_the_command_is_none(monkeypatch):
    def run(cmd, **kw):
        raise FileNotFoundError("sysctl")
    monkeypatch.setattr(solver_choice.subprocess, "run", run)
    assert solver_choice._sysctl_proc_translated() is None


def test_the_rosetta_check_runs_once_for_each_process(monkeypatch):
    """One `environment` call asked for it 7 times before: it is cached now (the function is in the cache key)."""
    runs = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(solver_choice, "_sysctl_proc_translated", lambda: runs.append(1) or "1")
    assert [solver_choice.mac_python_kind() for _ in range(5)] == ["rosetta"] * 5
    solver_choice.platform_hint()
    solver_choice.is_x86_mac()
    assert runs == [1]


# --- the hint ---------------------------------------------------------------------------------------


def test_hint_for_rosetta_has_the_commands_and_the_best_effort_facts(fake_mac):
    fake_mac("x86_64", "1")
    hint = solver_choice.platform_hint()
    assert "Rosetta" in hint and "arm64 Python is the fast path" in hint and "Open using Rosetta" in hint
    assert 'file "$(command -v uv)"' in hint and f"rm -rf {sys.prefix}" in hint.replace("'", "")
    assert "solver auto gives amg" in hint and "not calibrated" in hint and "best effort" in hint
    assert "older than the main lock" in hint and "Linux only" in hint
    assert "\n" not in hint


def test_hint_for_an_intel_mac_does_not_send_the_user_to_an_arm64_python(fake_mac):
    fake_mac("x86_64", "")
    hint = solver_choice.platform_hint()
    assert "Intel Mac" in hint and "arm64" not in hint.replace("Apple silicon", "")
    assert "solver auto gives amg" in hint and "not calibrated" in hint and "not on an Intel Mac" in hint
    assert "best effort" in hint and "older than the main lock" in hint and "rm -rf" not in hint


def test_hint_when_the_check_did_not_run_covers_both(fake_mac):
    fake_mac("x86_64", None)
    hint = solver_choice.platform_hint()
    assert "did not run" in hint and "arm64 Python" in hint and "Intel Mac has none" in hint and "best effort" in hint


def test_no_hint_on_macos_arm64_or_linux(fake_mac):
    for plat, machine in (("darwin", "arm64"), ("linux", "x86_64"), ("linux", "aarch64")):
        fake_mac(machine, "1", plat)
        assert solver_choice.platform_hint() is None


# --- the solver rule --------------------------------------------------------------------------------


@pytest.mark.parametrize("translated", ["1", "", None])
def test_auto_on_an_x86_mac_is_amg_and_never_imports_pypardiso(fake_mac, monkeypatch, translated):
    fake_mac("x86_64", translated)
    imported = []
    monkeypatch.setattr(solver_choice, "import_pypardiso", lambda: imported.append(1))
    assert solver_choice.resolve_solver("auto") == "amg"
    assert solver_choice.resolve_solver("amg") == "amg"
    assert imported == []


@pytest.mark.parametrize("translated", ["1", "", None])
def test_cholesky_on_an_x86_mac_is_a_user_error_with_the_options(fake_mac, translated):
    fake_mac("x86_64", translated)
    with pytest.raises(UserError) as e:
        solver_choice.resolve_solver("cholesky")
    msg = str(e.value)
    assert "Apple silicon only" in msg and "'amg'" in msg and "'auto'" in msg and "\n" not in msg
    assert "install" not in msg and "build" not in msg      # no offer to build a shim that cannot count
    assert ("arm64 Python" in msg) is (translated == "1")   # only Rosetta has a native Python to switch to


def test_pardiso_on_an_x86_mac_does_not_offer_cholesky(fake_mac):
    fake_mac("x86_64", "")
    with pytest.raises(UserError) as e:
        solver_choice.resolve_solver("pardiso")
    msg = str(e.value)
    assert "Linux x86_64 only" in msg and "'amg'" in msg and "cholesky" not in msg


def test_macos_arm64_and_linux_do_not_change(fake_mac, monkeypatch, tmp_path):
    """The platform choice for macOS arm64 and Linux x86_64 is the one of main. No Rosetta check runs."""
    # macOS arm64 with a shim that loads, and without
    calls = fake_mac("arm64", "1")
    fake_shim(monkeypatch, tmp_path, "ok")
    assert [solver_choice.resolve_solver(s) for s in ("auto", "cholesky", "amg")] == ["cholesky", "cholesky", "amg"]
    with pytest.raises(UserError, match="Linux x86_64"):
        solver_choice.resolve_solver("pardiso")
    fake_shim(monkeypatch, tmp_path, "missing")
    assert solver_choice.resolve_solver("auto") == "amg" and solver_choice.resolve_solver("cholesky") == "cholesky"
    assert "build.sh" in solver_choice.warning_line("auto", "amg")
    # Linux x86_64 without and with pypardiso
    fake_mac("x86_64", "1", "linux")
    assert solver_choice.resolve_solver("auto") == "amg"
    with pytest.raises(UserError, match="Apple Accelerate"):
        solver_choice.resolve_solver("cholesky")
    fake_mac("aarch64", None, "linux")
    with pytest.raises(UserError) as e:
        solver_choice.resolve_solver("pardiso")
    assert "'cholesky' (macOS)" in str(e.value)       # the text of main for pardiso on a platform that is not an x86 Mac
    fake_mac("x86_64", "1", "linux")
    monkeypatch.setattr(solver_choice, "import_pypardiso", lambda: types.ModuleType("pypardiso"))
    assert solver_choice.resolve_solver("auto") == "pardiso"
    assert calls == []


def test_make_solver_stops_with_a_user_error_not_a_runtime_error(fake_mac):
    """`refine` and the optimizer call `make_solver`. They never reach the missing shim file."""
    import numpy as np
    fake_mac("x86_64", "")
    with pytest.raises(UserError, match="Apple silicon only"):
        make_solver("cholesky", np.arange(3), (1, 1, 1), 1.0)


def test_backend_note_says_why_cholesky_is_not_available(fake_mac, monkeypatch, tmp_path):
    fake_shim(monkeypatch, tmp_path, "ok")       # a shim that would load: it does not count here
    fake_mac("x86_64", "1")
    note = cli.backend_availability()["cholesky"]
    assert note["available"] is False and "Rosetta" in note["note"] and "Apple silicon only" in note["note"]
    fake_mac("x86_64", "")
    assert "Intel Mac" in cli.backend_availability()["cholesky"]["note"]
    assert cli.backend_availability()["pardiso"]["available"] is False


# --- what the user sees: check, run, environment --------------------------------------------------


def test_check_shows_the_hint_and_the_amg_estimate(tmp_path, capsys, fake_mac):
    fake_mac("x86_64", "")
    assert main(["check", str(make_case(tmp_path, h=2.0))]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^solver:\s+amg", out, re.M)
    assert re.search(r"^hint: Python runs as x86_64 on an Intel Mac.*not calibrated", out, re.M)
    assert re.search(r"^time estimate:", out, re.M)


def test_check_json_has_the_hint_and_no_error(tmp_path, capsys, fake_mac):
    fake_mac("x86_64", "1")
    assert main(["check", "--json", str(make_case(tmp_path, h=2.0))]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["solver"]["resolved"] == "amg" and doc["solver"]["error"] is None
    assert "Rosetta" in doc["hint"] and doc["backends"]["cholesky"]["available"] is False
    assert doc["time_estimate"]["total_s"] > 0


def test_run_on_an_x86_mac_logs_the_hint_and_finishes(tmp_path, capsys, fake_mac):
    fake_mac("x86_64", "1")
    p = make_case(tmp_path, h=2.0, iterations=2)
    assert main(["run", str(p), "--out", str(tmp_path / "out"), "--quiet"]) == 0
    status = json.loads((tmp_path / "out" / "status.json").read_text())
    assert status["state"] == "done"
    log = (tmp_path / "out" / "run.log").read_text()
    assert "warning: Python runs as x86_64 on this Mac. Apple silicon runs this Python under Rosetta" in log
    assert "not calibrated" in log and "warning: solver auto" not in log   # no shim line: this is not Apple silicon native


def test_cholesky_setting_on_an_x86_mac_ends_with_exit_1_and_no_internal_error(tmp_path, capsys, fake_mac):
    fake_mac("x86_64", "")
    p = make_case(tmp_path, h=2.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "cholesky"
    p.write_text(yaml.safe_dump(d))
    assert main(["run", str(p), "--out", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Apple silicon only" in err and "amg" in err and "internal error" not in err
    status = json.loads((tmp_path / "out" / "status.json").read_text())
    assert status["state"] == "error" and "internal error" not in status["message"]
    assert main(["check", "--json", str(p)]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["solver"]["resolved"] is None and "Apple silicon only" in doc["solver"]["error"]
    assert doc["time_estimate"]["total_s"] > 0       # the estimate stays useful


def test_environment_facts_for_each_kind_of_python(fake_mac):
    expect = {("darwin", "x86_64", "1"): "rosetta", ("darwin", "x86_64", ""): "intel",
              ("darwin", "x86_64", None): "unknown", ("darwin", "arm64", "0"): "arm64",
              ("linux", "x86_64", "0"): None}
    for (plat, machine, translated), kind in expect.items():
        fake_mac(machine, translated, plat)
        d = environment_data()
        assert d["platform"]["mac_python"] == kind
        assert (d["platform"]["hint"] is None) is (kind in (None, "arm64"))
        assert d["platform"]["apple_silicon"] is (kind == "arm64")
        json.dumps(d["platform"])
    fake_mac("x86_64", "1")
    d = environment_data()
    assert d["auto_solver"] == "amg" and d["solver_setting"]["cholesky"]["resolved"] is None
    assert "Rosetta" in d["solver_setting"]["cholesky"]["error"]
    assert d["recommendations"] == [] or all(r["action"] == "install_pyamg" for r in d["recommendations"])
