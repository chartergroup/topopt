"""Shared markers for the tests. Import them with `from helpers import ...`."""

import functools
import types
from pathlib import Path

import pytest

from topopt import solver_choice

# Tests that call the Accelerate Cholesky solver run on macOS with Apple silicon only (an x86_64 Python
# on a Mac, Rosetta or an Intel Mac, has no shim: `resolve_solver("cholesky")` is a user error there).
needs_accelerate = pytest.mark.skipif(
    not solver_choice.is_macos_arm64(),
    reason=f"needs Apple Accelerate (solver 'cholesky', Apple silicon only); this platform is "
           f"{solver_choice.platform_name()}")

# Tests that call PARDISO run on Linux x86_64 with the pardiso extra installed.
# pypardiso is only imported on Linux x86_64 (never on macOS).
_HAS_PARDISO = solver_choice.is_linux_x86_64() and solver_choice.pypardiso_available()
needs_pardiso = pytest.mark.skipif(
    not _HAS_PARDISO,
    reason="needs pypardiso (Linux x86_64 only: install the 'pardiso' extra)")


# The size cap of a role playbook, in characters (topopt_mcp/roles/*.md and the agent copies in agents/*.md). The tests of the role text
# (test_team_guide.py) and of the plugin agents (test_plugin_manifest.py) both use it. Role playbooks stay short because the agent reads
# them in full; 8500 leaves room for the terrace facts of #452 in the finishing role.
ROLE_TEXT_MAX_CHARS = 8500


def recorded_hook_run() -> Path:
    """The folder of the recorded hook run (an older build), `recordings/hook_*`. A clone without a recording gets a path that does not exist, so
    `(recorded_hook_run() / "result" / "final.stl").is_file()` is False and the tests that need the recording skip."""
    recordings = Path(__file__).resolve().parent.parent / "recordings"
    runs = sorted(p for p in recordings.glob("hook_*") if p.is_dir())
    return runs[0] if runs else recordings / "no-recorded-run"


def on_the_platform_of_the_measurement() -> bool:
    """True on Linux x86_64, where the routes of the tet surface step (manifold3d 3.5.4: the boolean seam, the weld, the fold repair) were measured.
    The routes depend on the floating-point order of the boolean, so another platform can end a case another way."""
    import platform
    return platform.system() == "Linux" and platform.machine() == "x86_64"


# Tests of the buckling row (A6b) need a direct factor of K: Apple Cholesky on macOS arm64, PARDISO on Linux x86_64.
_HAS_ACCELERATE = solver_choice.is_macos_arm64() and solver_choice.accelerate_problem() is None
needs_direct_solver = pytest.mark.skipif(
    not (_HAS_ACCELERATE or _HAS_PARDISO),
    reason="needs a direct solver (Apple Cholesky on Apple silicon, or PARDISO on Linux x86_64 with the 'pardiso' extra)")


@functools.lru_cache(maxsize=1)
def tet_probe():
    """The probe of the tet extra, run once per session in a child Python (this process never loads libgmsh for it)."""
    from topopt.tet.gmsh_probe import probe_subprocess
    return probe_subprocess()


@functools.lru_cache(maxsize=1)
def ocp_problem():
    """None when OpenCASCADE (OCP, the extra `step`) imports in this process, else a one-line reason (the import can fail on a missing system library)."""
    try:
        import OCP  # noqa: F401
    except ImportError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


# Tests that write a STEP file run when OCP imports. A test of the missing-extra message does not use this marker: it runs a child Python
# in which OCP cannot be imported.
needs_ocp = pytest.mark.skipif(ocp_problem() is not None,
                               reason=f"needs OpenCASCADE (the 'step' extra: uv pip install -e \".[step]\"): {ocp_problem()}")


# Tests that call gmsh run when the probe says "ok" (the 'tet' extra and its system libraries).
# The marker uses the fixture `require_gmsh` (tests/conftest.py), which runs the probe in a child
# Python, once per session, and only when a test with the marker runs.
needs_gmsh = pytest.mark.usefixtures("require_gmsh")


_REAL_LOAD = solver_choice._load_library


def fake_shim(monkeypatch, tmp_path, state="ok"):
    """Fake the Accelerate shim of a (faked) Mac. Return the path of the library.

    "ok": the file is there and it loads. "missing": no file. "broken": a file that does not load (the
    real loader fails on it). "no_symbol": it loads, but one function of the shim is not in it.
    The folder name has a space, so a test also checks the quoting of the build command.
    """
    path = tmp_path / "native dir" / "libaccel_chol.dylib"
    monkeypatch.setattr(solver_choice, "accelerate_library_path", lambda: path)
    monkeypatch.setattr(solver_choice, "_load_library", _REAL_LOAD)
    if state == "missing":
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a library")
    if state != "broken":
        names = solver_choice.ACCEL_SYMBOLS if state == "ok" else solver_choice.ACCEL_SYMBOLS[:-1]
        monkeypatch.setattr(solver_choice, "_load_library",
                            lambda p: types.SimpleNamespace(**{n: None for n in names}))
    return path


def _imports(name: str) -> bool:
    """True if the module imports. An `OSError` or any other error counts as missing (see `needs_gmsh`)."""
    import importlib
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# Tests of `topopt.verify` need scikit-fem (the `verify` extra). cholespy is one of the factors.
needs_skfem = pytest.mark.skipif(not _imports("skfem"), reason="needs scikit-fem (the 'verify' extra)")
needs_cholespy = pytest.mark.skipif(not _imports("cholespy"), reason="needs cholespy (the 'verify' extra)")


SELF_TEST_PASSED = {"passed": True, "rel_diff": 3.0e-11, "seconds": 1.5, "free_dofs": 3888, "tol": 1e-8, "max_diff": 1e-6,
                    "device": "Test GPU", "case": "C", "error": None}
SELF_TEST_FAILED = {**SELF_TEST_PASSED, "passed": False, "rel_diff": 4.2e-4}


def in_process_probe(monkeypatch):
    """Make the OpenCL probe of `auto` and of the OpenCL facts run in this process (`gpu_mg.probe.run_probe`), so that a test can fake the devices.

    The product starts a child process with a time limit; a fake in this process would not reach it. The answers of the process start empty and go
    back to their old value when the test ends.
    """
    from topopt.gpu_mg import probe
    monkeypatch.setattr(probe, "run_probe", probe.probe_here)
    monkeypatch.setattr(probe, "_ANSWERS", {})


def fake_opencl(monkeypatch, state="no_device", devices=None, self_test=SELF_TEST_FAILED):
    """Fake the OpenCL state of the machine. No driver is loaded, so it is safe on a host with a GPU.

    `state`: "no_device" (pyopencl loads, no device), "devices" (pyopencl loads, `devices` is the list of
    `DeviceInfo`), "missing" (pyopencl is not installed) or "load_failed" (it is installed and does not load).
    `self_test` is the result of the self-test of `auto` on every fake device. The default is a failed test, so the
    CPU rule stays the pick of `auto`; no kernel runs. The test session sets `TOPOPT_AUTO_GPU=0` (tests/conftest.py); this function deletes it,
    so the fake devices are the ones that `auto` looks at.
    """
    from topopt.gpu_mg import device
    in_process_probe(monkeypatch)
    monkeypatch.delenv("TOPOPT_AUTO_GPU", raising=False)
    monkeypatch.setattr(device, "self_test", lambda dev, case: dict(self_test))
    problem = {"no_device": None, "devices": None, "missing": ("missing", "No module named 'pyopencl'"),
               "load_failed": ("load_failed", "ImportError: libOpenCL.so.1: cannot open shared object file")}[state]
    monkeypatch.setattr(device, "pyopencl_problem", lambda: problem)
    monkeypatch.setattr(device, "enumerate_devices", lambda: list(devices or []))


def make_device(index=0, name="Test GPU", kind="GPU", fp64=False, platform_name="Test Platform", vendor="", **kw):
    """A `DeviceInfo` for the tests of the OpenCL facts."""
    from topopt.gpu_mg.device import DeviceInfo
    return DeviceInfo(index=index, platform=platform_name, name=name, kind=kind, fp64=fp64, vendor=vendor,
                      version="OpenCL 1.2", max_work_group_size=256, max_mem_alloc=2**30, global_mem=2**32,
                      local_mem=32768, constant_mem=65536, compute_units=8, **kw)


_FACTOR_NOTES = {
    "cholesky": "Apple silicon only (this platform is fake)",
    "pardiso": "pypardiso is not installed. Install it with: uv pip install -e '.[pardiso]'",
    "cholespy": "cholespy is not installed (No module named 'cholespy'). The 'verify' extra holds it; install the extra with: uv pip install --python PY -e '.[verify]'",
    "superlu": "scipy SuperLU does not load (ImportError: fake). Check the install of scipy",
}


def fake_factor_availability(*working):
    """The result of `factor_availability()` for a fake host where only the factors in `working` can run."""
    return {k: {"available": k in working, "note": f"{k} works (fake)" if k in working else v}
            for k, v in _FACTOR_NOTES.items()}


def switch_off_deflation(monkeypatch):
    """P2d: make every design of every `GpuSolver` run without the deflation of the rigid modes (`GpuMG.use_deflation(False)`: the launches and the PCG of
    before P2d). The test then measures the engine of before."""
    from topopt.gpu_mg.model import GpuSolver

    def off(self, T):
        self.engine.use_deflation(False)
        return None
    monkeypatch.setattr(GpuSolver, "_set_deflation", off)
