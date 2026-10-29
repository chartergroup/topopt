"""Tests for tools/platform_probe.py: it reports the platform and the solver, and it writes nothing."""
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from topopt import solver_choice

TOOL = Path(__file__).resolve().parent.parent / "tools" / "platform_probe.py"
spec = importlib.util.spec_from_file_location("platform_probe", TOOL)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_probe_runs_in_a_new_process_prints_json_and_writes_no_file(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(TOOL), "--json", "--no-import"], capture_output=True, text=True,
                          cwd=tmp_path, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    d = json.loads(done.stdout)
    assert {"machine", "python", "packages", "solver"} <= set(d)
    assert d["machine"]["machine"] == platform.machine() and d["python"]["version"] == platform.python_version()
    assert {r["package"] for r in d["packages"]} >= {"numpy", "numba", "llvmlite", "pypardiso", "gmsh"}
    assert d["solver"]["available"] and d["solver"]["settings"]["amg"]["resolved"] == "amg"
    assert list(tmp_path.iterdir()) == []              # no file in the working folder


def test_import_test_reports_ok_failed_and_a_crash(tmp_path):
    assert probe.try_import("json", str(tmp_path))["status"] == "ok"
    bad = probe.try_import("no_such_module_v5", str(tmp_path))
    assert bad["status"] == "failed" and "No module named" in bad["detail"]


@pytest.mark.parametrize("translated, hw_arm64, kind, expect", [
    ("1", "1", "rosetta", "under Rosetta"),
    (None, None, "intel", "Intel Mac"),        # macOS before 11: the key does not exist
    ("0", None, "intel", "Intel Mac"),         # an Intel Mac with macOS 11 or newer: the key is 0
])
def test_probe_names_rosetta_and_an_intel_mac(monkeypatch, translated, hw_arm64, kind, expect):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.5", ("", "", ""), "x86_64"))
    monkeypatch.setattr(probe, "sysctl", lambda key: {"sysctl.proc_translated": translated,
                                                      "hw.optional.arm64": hw_arm64}.get(key))
    monkeypatch.setattr(solver_choice, "_sysctl_proc_translated", lambda: translated or "")
    d = probe.collect(do_import=False)
    assert expect in d["machine"]["python_kind"]
    assert d["solver"]["mac_python_kind"] == kind and d["solver"]["settings"]["auto"]["resolved"] == "amg"
    assert d["solver"]["settings"]["cholesky"]["error"]
    text = probe.render(d)
    assert "topopt solver rule:" in text and "Send this text back" in text


def test_probe_on_native_apple_silicon(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(probe, "sysctl", lambda key: "0" if key == "sysctl.proc_translated" else None)
    assert probe.machine_facts()["python_kind"] == "native arm64 Python"
