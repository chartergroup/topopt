"""Shared test setup: no test opens a real browser."""

import glob
import importlib.util
import os
import sys
import webbrowser

import pytest

# The tests of the solver gpu-mg run on a CPU OpenCL device (pocl). A real GPU is used only
# when this variable is "1" (a run on a Mac). Without it, a visible GPU has this effect:
#   Linux: the session stops (exit code 3).
#   macOS: the tests with the marker `opencl_context` are skipped with one reason, and the rest of pytest runs.
REAL_GPU_VARIABLE = "TOPOPT_TEST_REAL_GPU"
AUTO_GPU_VARIABLE = "TOPOPT_AUTO_GPU"

from helpers import tet_probe as _tet_probe  # noqa: E402,F401  (one cached probe per session; also used by tests)


@pytest.fixture(autouse=True)
def opened_urls(monkeypatch):
    """Replace `webbrowser.open` for every test. The fixture value is the list of opened URLs.

    `topopt run --dashboard` opens the dashboard and, after the run, the refine
    page. Tests read this list to check the URLs.

    A test that starts `python -m topopt run --dashboard` as a child process does not get this patch. `BROWSER=true`
    makes `webbrowser` in the child run the command `true`, so no real browser tab opens.
    """
    urls: list[str] = []

    def fake_open(url, *args, **kwargs):
        urls.append(url)
        return True

    monkeypatch.setattr(webbrowser, "open", fake_open)
    monkeypatch.setenv("BROWSER", "true")
    return urls


@pytest.fixture(autouse=True)
def state_folder(tmp_path_factory, monkeypatch, request):
    """Point `TOPOPT_STATE_DIR` (the per-user folder of topopt: the calibration file of `topopt gpu-check`) at a folder of this test.

    No test reads or writes `~/.topopt`: a result of `check`, `derived_numbers` or `build_spec` with gpu-mg must not depend on a calibration
    that the user made, and a bad file there must not break the tests (P2a review m2). The registry of topopt servers (`servers/` in this folder,
    CONTRACTS.md section 43) is in this folder too: no test reads or writes the registry folder of the user, also not a server that a test starts
    over stdio. The folder is not made until a test writes in it; a test that needs another folder sets the variable itself. The value is the path.
    """
    import hashlib
    folder = tmp_path_factory.getbasetemp() / "state" / hashlib.md5(request.node.nodeid.encode()).hexdigest()[:12]
    monkeypatch.setenv("TOPOPT_STATE_DIR", str(folder))
    # The registry reads `XDG_STATE_HOME` too (Linux, CONTRACTS.md section 43.1). A runner that sets it must not give a test a way to the real
    # folder of the user, for example a test that removes `TOPOPT_STATE_DIR`. A test of the XDG rule sets the variable itself.
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    if importlib.util.find_spec("mcp") is not None:
        # A server that a test starts over stdio (`StdioServerParameters` without `env`) gets the small default environment of the MCP SDK, which
        # does not hold TOPOPT_STATE_DIR. Such a server would write its entry in the registry folder of the user (CONTRACTS.md section 43, P9-14).
        from mcp.client import stdio
        default_environment = stdio.get_default_environment
        monkeypatch.setattr(stdio, "get_default_environment", lambda: {**default_environment(), "TOPOPT_STATE_DIR": str(folder)})
    return folder


@pytest.fixture
def require_gmsh():
    """Skip the test if the probe of the tet extra is not "ok". The skip reason holds the whole probe text.

    Use it with the marker `needs_gmsh` of `helpers.py`. Do NOT use `pytest.importorskip("gmsh")`:
    it errors with `OSError` when a system library is missing.
    """
    probe = _tet_probe()
    if not probe.ok:
        pytest.skip(f"needs gmsh (the 'tet' extra): {probe.state}: {probe.detail}. {probe.fix}")


@pytest.fixture
def sdf_cut_only(monkeypatch):
    """The `sdf` mesh step without the limit of the field to the envelope (#319), so the cuts take what is past the envelope.

    For the tests of the cut to the envelope, of the cut at the shared planes and of the calibration rounds that pay for a cut: their cases
    have material past the envelope only without the limit. The post step and `make_mesh_sdf` then use `SdfParams` with
    `envelope_limit=False` as the default.
    """
    import dataclasses

    from topopt import smooth_sdf

    @dataclasses.dataclass(frozen=True)
    class CutOnly(smooth_sdf.SdfParams):
        envelope_limit: bool = False

    monkeypatch.setattr(smooth_sdf, "SdfParams", CutOnly)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a test that takes 5 s or more (a test with the mark in its file, or a node id in tests/tiers/slow.txt); the fast tier of the CI is -m 'not slow'")
    config.addinivalue_line("markers", "equivalence: a test in the selection of acceptance level 1: it compares the demo with a record of the old pipeline "
                                       "(fixtures made from its source lines) or checks the CI machinery of the golden file (tests/test_ci_scripts.py); "
                                       "pytest -m equivalence runs them")
    config.addinivalue_line("markers", "platform_sensitive: a test in a file of tests/tiers/platform_files.txt (it can behave differently on macOS and Linux); "
                                       "the fast tier of the macOS leg is -m 'platform_sensitive and not slow'")
    config.addinivalue_line("markers", "opencl_context: the test makes an OpenCL context (skipped on a Mac unless "
                                       f"{REAL_GPU_VARIABLE}=1; see tests/conftest.py)")


def _icd_names(environ=None, platform=None):
    """Linux only: the driver names in every place where an ICD loader looks. Nothing is loaded.

    Always read: the folder `OCL_ICD_VENDORS` (else `/etc/OpenCL/vendors`) and the folder `.libs` of
    `PYOPENCL_HOME` (else of the pyopencl package). The loader inside the pyopencl wheel reads both and
    ignores `OCL_ICD_FILENAMES`, so a value of that variable does not make a folder safe. The names in
    `OCL_ICD_FILENAMES` are added for a system loader that reads it.
    """
    environ = os.environ if environ is None else environ
    if not (platform or sys.platform).startswith("linux"):
        return []
    names = [n for n in environ.get("OCL_ICD_FILENAMES", "").split(":") if n]
    folders = [environ.get("OCL_ICD_VENDORS") or "/etc/OpenCL/vendors"]
    home = environ.get("PYOPENCL_HOME")
    if not home:
        spec = importlib.util.find_spec("pyopencl")      # finds the folder, does not import the package
        home = list(spec.submodule_search_locations)[0] if spec and spec.submodule_search_locations else None
    if home:
        folders.append(os.path.join(home, ".libs"))
    for folder in folders:
        for f in sorted(glob.glob(os.path.join(folder, "*.icd"))):
            try:
                names.append(open(f).read().strip())
            except OSError:
                names.append(f)
    return names


def _icd_problem(environ=None, platform=None):
    """Linux only: why the OpenCL ICD list holds a driver that is not pocl, or None.

    It reads the ICD files and loads nothing. So a GPU driver is not loaded by the guard itself.
    """
    other = sorted({n for n in _icd_names(environ, platform) if "pocl" not in n.lower()})
    return f"the OpenCL ICD list has a driver that is not pocl: {', '.join(other)}" if other else None


def _opencl_problem():
    """Why a test run could make a context on a device that is not a CPU, or None. Loads the drivers (no context)."""
    problem = _icd_problem()
    if problem is None:
        try:
            import pyopencl as cl
            devices = [d for p in cl.get_platforms() for d in p.get_devices()]
        except Exception:               # a stand-in or a broken install, or no platform: the tests skip
            return None
        bad = [d.name for d in devices if not d.type & cl.device_type.CPU]
        if bad:
            problem = f"a non-CPU OpenCL device is visible: {', '.join(bad)}"
    return problem


def pytest_sessionstart(session):
    """A visible non-CPU OpenCL device, without the opt-in variable: stop the session (Linux) or skip the
    tests that make a context (macOS).

    Before that, the session keeps `settings.solver: auto` on the CPU rule (`TOPOPT_AUTO_GPU=0`), so a machine with an OpenCL GPU runs the `auto`
    tests as a machine without one. The tests that check the pick of `auto` fake their devices and delete the variable. A value that the user set
    stays, and so does the session of `TOPOPT_TEST_REAL_GPU=1`.
    """
    if os.environ.get(REAL_GPU_VARIABLE) != "1":
        os.environ.setdefault(AUTO_GPU_VARIABLE, "0")
    session.config._opencl_skip = None
    if os.environ.get(REAL_GPU_VARIABLE) == "1" or importlib.util.find_spec("pyopencl") is None:
        return
    problem = _opencl_problem()
    if not problem:
        return
    if sys.platform == "darwin":
        session.config._opencl_skip = (f"{problem}; the tests that make an OpenCL context use a real GPU only "
                                       f"when {REAL_GPU_VARIABLE}=1 (then they run on the GPU, case C)")
        return
    advice = (f"Use the pocl-only environment (OCL_ICD_VENDORS = the folder pyopencl/.libs of the venv, "
              f"OCL_ICD_FILENAMES unset, POCL_DEVICES=pthread), or set {REAL_GPU_VARIABLE}=1 to allow a real GPU.")
    pytest.exit(f"{problem}. {advice}", returncode=3)


def _tier_list(name):
    """The lines of tests/tiers/<name> that are not empty and do not start with #. Made by scripts/ci/make_tiers.py."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiers", name)
    try:
        with open(path, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    except FileNotFoundError:
        return set()


def pytest_collection_modifyitems(config, items):
    slow, platform_files = _tier_list("slow.txt"), _tier_list("platform_files.txt")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for item in items:
        # The lists use `tests/test_x.py::name`. item.nodeid is relative to the pytest rootdir, which is not always the repo
        # (an ini file above the repo moves it), so build the id from the file path.
        path = getattr(item, "path", None)      # a test of this file calls the hook with plain objects that have no path
        if path is None:
            continue
        file = os.path.relpath(str(path), repo).replace(os.sep, "/")
        if f"{file}::{item.nodeid.split('::', 1)[-1]}" in slow:
            item.add_marker(pytest.mark.slow)
        if file in platform_files:
            item.add_marker(pytest.mark.platform_sensitive)
    reason = getattr(config, "_opencl_skip", None)
    if reason:
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if item.get_closest_marker("opencl_context"):
                item.add_marker(skip)
