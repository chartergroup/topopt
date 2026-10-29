"""Tests for the core-file limits of the gmsh child (topopt/tet/worker.py, CONTRACTS.md section 14).

No test makes a real core file. The crash test checks the limit and the dumpable flag in the child first
and raises SIGSEGV only when both are in place. The platform logic runs with fakes, so each platform
branch runs on every host. None of these tests needs gmsh.
"""
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from topopt.tet import MeshingCrashed, mesher, worker

ROOT = Path(__file__).resolve().parent.parent
LINUX = sys.platform.startswith("linux")


class FakeResource:
    """The part of the `resource` module that the worker uses."""
    RLIMIT_CORE = 4

    def __init__(self, soft=1000, hard=2000, fail=False):
        self.limits, self.fail, self.set_calls = (soft, hard), fail, []

    def getrlimit(self, which):
        assert which == self.RLIMIT_CORE
        return self.limits

    def setrlimit(self, which, limits):
        if self.fail:
            raise ValueError("fake: not allowed")
        self.set_calls.append((which, limits))
        self.limits = limits


class FakeLibc:
    """The part of libc that the worker uses. `prctl` gets ctypes values; the fake records their numbers."""

    def __init__(self, result=0):
        self.result, self.calls = result, []

    def prctl(self, *args):
        self.calls.append(tuple(a.value for a in args))
        return self.result


def child_command(*tail):
    """The command of the parent (`mesher._worker_command`) with `tail` instead of the two file paths."""
    return mesher._worker_command("in", "out")[:-2] + list(tail)


def child_env(**extra):
    """The environment that the parent gives to the child (`mesher.mesh_surface_isolated`)."""
    env = dict(os.environ)
    env.pop(worker.KEEP_CORE_ENV, None)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(extra)
    return env


def run_child_report(**extra_env):
    done = subprocess.run(child_command("--limits"), capture_output=True, text=True, env=child_env(**extra_env))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


# --- the platform logic, with fakes --------------------------------------------------------------

def test_linux_sets_the_soft_limit_and_clears_dumpable():
    res, libc = FakeResource(soft=1000, hard=2000), FakeLibc()
    status = worker.limit_core_dumps(environ={}, platform="linux", resource_module=res, libc=libc)
    assert res.set_calls == [(4, (0, 2000))]  # soft 0, hard limit unchanged
    assert libc.calls == [(worker.PR_SET_DUMPABLE, 0, 0, 0, 0)]
    assert status["rlimit_core"].startswith("set") and status["dumpable"].startswith("set")


def test_macos_sets_the_soft_limit_only():
    res, libc = FakeResource(), FakeLibc()
    status = worker.limit_core_dumps(environ={}, platform="darwin", resource_module=res, libc=libc)
    assert res.set_calls == [(4, (0, 2000))] and libc.calls == []
    assert status["rlimit_core"].startswith("set") and status["dumpable"].startswith("skipped")


def test_a_failed_call_does_not_stop_the_other_step_and_does_not_raise():
    res, libc = FakeResource(fail=True), FakeLibc(result=-1)
    status = worker.limit_core_dumps(environ={}, platform="linux", resource_module=res, libc=libc)
    assert status["rlimit_core"].startswith("failed") and status["dumpable"].startswith("failed")
    assert libc.calls  # the limit failed, and prctl was still tried
    # The same when the module or libc cannot load at all.
    broken = object()  # no getrlimit, no prctl
    status = worker.limit_core_dumps(environ={}, platform="linux", resource_module=broken, libc=broken)
    assert status["rlimit_core"].startswith("failed") and status["dumpable"].startswith("failed")


@pytest.mark.parametrize("value, keeps", [("1", True), ("yes", True), ("0", False), ("", False)])
def test_the_override_variable(value, keeps):
    res, libc = FakeResource(), FakeLibc()
    status = worker.limit_core_dumps(environ={worker.KEEP_CORE_ENV: value}, platform="linux",
                                     resource_module=res, libc=libc)
    assert (res.set_calls == [] and libc.calls == []) is keeps
    assert status["rlimit_core"].startswith("skipped") is keeps


# --- a child that the parent starts ---------------------------------------------------------------

def test_the_child_has_core_limit_0_and_the_parent_keeps_its_limits():
    before = (resource.getrlimit(resource.RLIMIT_CORE), worker.core_facts()["dumpable"])
    report = run_child_report()
    assert report["core_soft"] == 0
    assert report["core_hard"] == before[0][1]  # the hard limit is the one of the parent
    assert report["steps"]["rlimit_core"].startswith("set")
    if LINUX:
        assert report["steps"]["dumpable"].startswith("set"), report["steps"]
        assert report["dumpable"] == 0
    else:
        assert report["steps"]["dumpable"].startswith("skipped")
    assert (resource.getrlimit(resource.RLIMIT_CORE), worker.core_facts()["dumpable"]) == before


def test_the_override_variable_keeps_the_limits_of_the_parent():
    parent = worker.core_facts()
    report = run_child_report(**{worker.KEEP_CORE_ENV: "1"})
    assert report["core_soft"] == parent["core_soft"] and report["dumpable"] == parent["dumpable"]
    assert report["steps"]["rlimit_core"].startswith("skipped")


# The child of the crash test: the real `main` (it sets the limit and the flag), then a `run` that
# raises SIGSEGV where gmsh would crash. It crashes only if the limit and the flag are in place. If a
# step failed, it exits with 97 and no core file can come from it.
CRASH_CHILD = """
import os, resource, signal, sys
from topopt.tet import worker

def crash_if_no_core_is_possible(in_path, out_path):
    soft = resource.getrlimit(resource.RLIMIT_CORE)[0]
    if soft != 0 or (sys.platform.startswith("linux") and worker.core_facts()["dumpable"] != 0):
        os._exit(97)
    os.kill(os.getpid(), signal.SIGSEGV)

worker.run = crash_if_no_core_is_possible
worker.main(sys.argv[1:])
"""


def test_a_crash_of_the_child_leaves_no_core_file_and_the_parent_reports_the_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the child inherits the working directory
    monkeypatch.delenv(worker.KEEP_CORE_ENV, raising=False)
    monkeypatch.setattr(mesher, "_worker_command",
                        lambda in_path, out_path: [sys.executable, "-c", CRASH_CHILD, in_path, out_path])
    work = tmp_path / "work"
    work.mkdir()
    before = (resource.getrlimit(resource.RLIMIT_CORE), worker.core_facts()["dumpable"])
    V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2]])
    with pytest.raises(MeshingCrashed) as caught:
        mesher.mesh_surface_isolated(V, F, 1.0, workdir=str(work))
    crash = caught.value.crash
    assert crash["returncode"] != 97, "the child had no core limit or dumpable flag, so it did not crash"
    assert crash["signal"] == "SIGSEGV" and crash["returncode"] == -11  # the existing crash result
    assert not (work / "child_out.npz").exists()  # no output file
    assert sorted(p.name for p in tmp_path.rglob("core*")) == []
    assert (resource.getrlimit(resource.RLIMIT_CORE), worker.core_facts()["dumpable"]) == before
