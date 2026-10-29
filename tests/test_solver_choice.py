"""Tests for the platform rule of the solver (topopt/solver_choice.py).

The platform and the pypardiso import are faked, so every rule runs on every
platform. No test here needs Accelerate or a real pypardiso.
"""
import json
import os
import platform
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import yaml

from helpers import fake_shim, needs_pardiso
from test_fe import _beam
from test_optimize import make_case
from topopt import cli, solver_choice
from topopt import solver as solver_module
from topopt.cli import estimate_seconds, main
from topopt.config import UserError, load_config
from topopt.optimize import make_solver
from topopt.solver import LinearSolver

REPO = Path(__file__).resolve().parent.parent


class Import:
    """Stand-in for `solver_choice.import_pypardiso`. It counts the calls."""

    def __init__(self, module):
        self.module, self.calls = module, 0

    def __call__(self):
        self.calls += 1
        if isinstance(self.module, Exception):   # installed, but it does not load
            raise self.module
        if self.module is None:
            raise ModuleNotFoundError("No module named 'pypardiso'", name="pypardiso")
        return self.module


@pytest.fixture
def fake_platform(monkeypatch):
    """`fake_platform("linux", "x86_64", pypardiso=None)` fakes the platform and the import."""
    def set_platform(plat, machine, pypardiso=None):
        monkeypatch.setattr(sys, "platform", plat)
        monkeypatch.setattr(platform, "machine", lambda: machine)
        imp = Import(pypardiso)
        monkeypatch.setattr(solver_choice, "import_pypardiso", imp)
        return imp
    return set_platform


def beam_dofs():
    edof, idx, free, node, ke = _beam(6, 3, 3)
    return free, (6, 3, 3)


# --- the rule for "auto" ---------------------------------------------------------


def test_macos_arm64_auto_is_cholesky_and_never_imports_pypardiso(monkeypatch, tmp_path, fake_platform):
    imp = fake_platform("darwin", "arm64", pypardiso=types.ModuleType("pypardiso"))
    fake_shim(monkeypatch, tmp_path, "ok")
    tried = []

    class Finder:   # any import of pypardiso, by any route, ends up here
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "pypardiso":
                tried.append(name)
            return None

    monkeypatch.delitem(sys.modules, "pypardiso", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])
    assert solver_choice.resolve_solver("auto") == "cholesky"
    assert solver_choice.resolve_solver("cholesky") == "cholesky"
    free, shape = beam_dofs()
    solver, name = make_solver("auto", free, shape, 1.0)
    assert name == "cholesky" and solver.requested == "cholesky" and solver.nullspace is None
    assert solver_choice.recommendation("auto", name) is None
    assert solver_choice.warning_line("auto", name) is None
    assert imp.calls == 0 and tried == []


def test_linux_without_pypardiso_auto_is_amg_with_nullspace(fake_platform):
    imp = fake_platform("linux", "x86_64", pypardiso=None)
    assert solver_choice.resolve_solver("auto") == "amg"
    free, shape = beam_dofs()
    solver, name = make_solver("auto", free, shape, 1.0)
    assert name == "amg" and solver.requested == "amg"
    assert solver.nullspace is not None and solver.nullspace.shape == (free.size, 6)
    assert imp.calls >= 1   # it asked for pypardiso, and did not get it


def test_linux_with_pypardiso_auto_is_pardiso(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=types.ModuleType("pypardiso"))
    assert solver_choice.resolve_solver("auto") == "pardiso"
    free, shape = beam_dofs()
    solver, name = make_solver("auto", free, shape, 1.0)
    assert name == "pardiso" and solver.requested == "pardiso" and solver.nullspace is None
    assert solver_choice.recommendation("auto", name) is None


@pytest.mark.parametrize("plat, machine", [("linux", "aarch64"), ("darwin", "x86_64"), ("win32", "AMD64")])
def test_other_platforms_auto_is_amg_without_an_offer(fake_platform, plat, machine):
    imp = fake_platform(plat, machine, pypardiso=types.ModuleType("pypardiso"))
    assert solver_choice.resolve_solver("auto") == "amg"
    assert solver_choice.recommendation("auto", "amg") is None   # the extra cannot install here
    assert imp.calls == 0


def test_amg_setting_is_amg_everywhere(fake_platform):
    for plat, machine in (("darwin", "arm64"), ("linux", "x86_64")):
        fake_platform(plat, machine, pypardiso=None)
        assert solver_choice.resolve_solver("amg") == "amg"
        assert solver_choice.recommendation("amg", "amg") is None   # the user chose amg


# --- a solver that cannot run: one clear user error ---------------------------------


def test_cholesky_on_linux_is_a_user_error_with_the_options(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=None)
    with pytest.raises(UserError) as e:
        solver_choice.resolve_solver("cholesky")
    msg = str(e.value)
    assert "linux x86_64" in msg and "pardiso" in msg and "amg" in msg and "\n" not in msg


def test_cholesky_in_the_linear_solver_is_a_user_error_too(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=None)
    with pytest.raises(UserError, match="Accelerate"):
        LinearSolver("cholesky").factor(sp.csc_matrix(np.eye(3) * 2.0))


def test_pardiso_setting_without_the_package_names_the_install_and_the_options(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=None)
    with pytest.raises(UserError) as e:
        solver_choice.resolve_solver("pardiso")
    assert "uv pip install" in str(e.value) and "amg" in str(e.value) and "Intel" in str(e.value)


def test_pardiso_setting_on_macos_is_a_user_error(fake_platform):
    imp = fake_platform("darwin", "arm64", pypardiso=types.ModuleType("pypardiso"))
    with pytest.raises(UserError, match="Linux x86_64"):
        solver_choice.resolve_solver("pardiso")
    assert imp.calls == 0


@pytest.mark.parametrize("cmd", ["run", "check"])
def test_cholesky_setting_on_linux_ends_with_exit_1(tmp_path, capsys, fake_platform, cmd):
    fake_platform("linux", "x86_64", pypardiso=None)
    p = make_case(tmp_path, h=2.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "cholesky"
    p.write_text(yaml.safe_dump(d))
    args = ["run", str(p), "--out", str(tmp_path / "r"), "--quiet"] if cmd == "run" else ["check", str(p)]
    assert main(args) == 1   # a user error, not "internal error" (2)
    err = capsys.readouterr().err
    assert err.startswith("error:") and "linux x86_64" in err and "pardiso" in err and "amg" in err
    assert "Traceback" not in err and "internal error" not in err
    if cmd == "run":
        status = json.loads((tmp_path / "r" / "status.json").read_text())
        assert status["state"] == "error" and not (tmp_path / "r" / "run.json").exists()


def test_yaml_accepts_pardiso(tmp_path):
    from topopt.config import load_config
    p = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "pardiso"
    p.write_text(yaml.safe_dump(d))
    assert load_config(p).settings.solver == "pardiso"
    d["settings"]["solver"] = "mkl"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="auto, cholesky, pardiso, amg or gpu-mg"):
        load_config(p)


# --- check: the solver and the offer ------------------------------------------------


def test_check_on_linux_without_pypardiso_makes_the_offer(tmp_path, capsys, fake_platform):
    fake_platform("linux", "x86_64", pypardiso=None)
    p = make_case(tmp_path, h=2.0)
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert "time estimate" in out and "free dofs" in out            # the old lines stay
    assert re.search(r"^solver:\s+amg", out, re.M)
    line = [l for l in out.splitlines() if l.startswith("recommendation: ")][0]
    assert not line.startswith("recommendation: {") and '"action"' not in line         # Q1-13: a text line, not a raw dict
    assert "Command (install_pypardiso, extra 'pardiso'): uv pip install" in line and "[pardiso]" in line
    assert "times faster for this problem" in line and "Simplified Software License" in line
    assert main(["check", "--json", str(p)]) == 0                                       # the JSON keeps the keys
    rec = json.loads(capsys.readouterr().out)["recommendation"]
    assert rec["action"] == "install_pypardiso" and rec["extra"] == "pardiso"
    assert "uv pip install" in rec["install_command"] and "[pardiso]" in rec["install_command"]
    assert str(solver_choice.HOOK_AMG_S) in rec["reason"] and str(solver_choice.HOOK_PARDISO_S) in rec["reason"]
    note = rec["license_note"]
    for name in ("mkl", "tbb", "tcmlib", "Simplified Software License", "intel-openmp",
                 "intel-cmplr-lib-ur", "EULA", "umf", "Apache-2.0", "456 MB", "not pinned"):
        assert name in note, name
    # reason, speedup_factor and estimate_s agree
    est = rec["estimate_s"]
    assert rec["speedup_factor"] == pytest.approx(est["without_pardiso"] / est["with_pardiso"], abs=0.06)
    assert f"{est['without_pardiso']} s with amg" in rec["reason"]
    assert f"{est['with_pardiso']} s with pardiso" in rec["reason"]
    # the estimate scales only the time per iteration: the ratio stays under the factor
    assert 1 <= rec["speedup_factor"] < solver_choice.AMG_TIME_FACTOR


def test_check_json(tmp_path, capsys, fake_platform):
    fake_platform("linux", "x86_64", pypardiso=None)
    assert main(["check", "--json", str(make_case(tmp_path, h=2.0))]) == 0
    doc = json.loads(capsys.readouterr().out)
    choice = doc["solver"].pop("choice")            # GPU first: what the run uses first and the facts of `auto`
    assert doc["solver"] == {"setting": "auto", "resolved": "amg", "cpu_solver": "amg", "platform": "linux x86_64", "error": None}
    assert choice["solver"] == "amg" and choice["cpu_solver"] == "amg" and choice["fallback"] is None and choice["device"] is None
    assert doc["time_estimate"]["solver"] == "amg"
    assert doc["recommendation"]["extra"] == "pardiso"
    for key in ("name", "grid_shape", "voxel_size_mm", "active_voxels", "free_dofs", "iterations", "warnings"):
        assert key in doc


def test_check_with_pardiso_has_no_offer(tmp_path, capsys, fake_platform):
    fake_platform("linux", "x86_64", pypardiso=types.ModuleType("pypardiso"))
    assert main(["check", str(make_case(tmp_path, h=2.0))]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^solver:\s+pardiso", out, re.M) and "recommendation" not in out
    assert main(["check", "--json", str(make_case(tmp_path, h=2.0))]) == 0
    assert json.loads(capsys.readouterr().out)["recommendation"] is None


def test_check_on_macos_with_the_shim_reports_cholesky(tmp_path, capsys, monkeypatch, fake_platform):
    imp = fake_platform("darwin", "arm64", pypardiso=None)
    fake_shim(monkeypatch, tmp_path, "ok")
    assert main(["check", str(make_case(tmp_path, h=2.0))]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^solver:\s+cholesky", out, re.M) and "recommendation" not in out
    assert "solver auto -> amg" not in out
    assert imp.calls == 0


# --- macOS without a working shim: auto runs amg and says how to build it -------------------------


def test_accelerate_problem_present_missing_and_not_loading(monkeypatch, tmp_path):
    fake_shim(monkeypatch, tmp_path, "ok")
    assert solver_choice.accelerate_problem() is None
    fake_shim(monkeypatch, tmp_path / "a", "missing")
    kind, text = solver_choice.accelerate_problem()
    assert kind == "missing" and "not found" in text
    fake_shim(monkeypatch, tmp_path / "b", "broken")          # a file that the real loader rejects
    kind, text = solver_choice.accelerate_problem()
    assert kind == "load_failed" and "OSError" in text
    fake_shim(monkeypatch, tmp_path / "c", "no_symbol")       # it loads, but a function is missing
    kind, text = solver_choice.accelerate_problem()
    assert kind == "load_failed" and "AttributeError" in text


@pytest.mark.parametrize("state, words", [("ok", None), ("missing", "is not built"),
                                          ("broken", "built but it does not load")])
def test_macos_arm64_auto_follows_the_shim(monkeypatch, tmp_path, fake_platform, state, words):
    fake_platform("darwin", "arm64", pypardiso=None)
    path = fake_shim(monkeypatch, tmp_path, state)
    resolved = solver_choice.resolve_solver("auto")
    assert resolved == ("cholesky" if state == "ok" else "amg")
    assert solver_choice.resolve_solver("cholesky") == "cholesky"   # an explicit name stays what it says
    assert solver_choice.resolve_solver("amg") == "amg"
    line = solver_choice.warning_line("auto", resolved)
    if state == "ok":
        assert line is None
        return
    assert line.startswith("solver auto -> amg") and words in line
    assert f"{solver_choice.AMG_TIME_FACTOR:g}x slower per iteration" in line and "an estimate" in line
    assert f"sh '{path.parent}/build.sh'" in line and "Xcode" in line      # the build command, quoted
    assert solver_choice.warning_line("amg", "amg") is None
    assert solver_choice.warning_line("cholesky", "cholesky") is None
    # The model that runs gets the nullspace, like on Linux.
    free, shape = beam_dofs()
    solver, name = make_solver("auto", free, shape, 1.0)
    assert name == "amg" and solver.nullspace is not None


def test_macos_without_shim_check_warns_and_estimates_amg(tmp_path, capsys, monkeypatch, fake_platform):
    fake_platform("darwin", "arm64", pypardiso=None)
    fake_shim(monkeypatch, tmp_path, "missing")
    assert main(["check", str(make_case(tmp_path, h=2.0))]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^solver:\s+amg", out, re.M) and "warning: solver auto -> amg" in out
    assert main(["check", "--json", str(make_case(tmp_path, h=2.0))]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["solver"]["resolved"] == "amg" and doc["solver"]["error"] is None
    assert any(w.startswith("solver auto -> amg") for w in doc["warnings"])
    assert doc["backends"]["cholesky"]["available"] is False and "build.sh" in doc["backends"]["cholesky"]["note"]
    assert doc["time_estimate"]["total_s"] == doc["time_estimate"]["amg_total_s"]   # the estimate is for amg


def test_run_on_macos_without_shim_warns_once_and_records_amg(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(solver_choice, "is_macos_arm64", lambda: True)
    monkeypatch.setattr(solver_choice, "is_linux_x86_64", lambda: False)
    fake_shim(monkeypatch, tmp_path, "missing")
    out = run_case(tmp_path, capsys)
    log = (out / "run.log").read_text()
    warns = [l for l in log.splitlines() if l.startswith("warning: solver auto -> amg")]
    assert len(warns) == 1 and "the Accelerate shim is not built" in warns[0] and "build.sh" in warns[0]
    assert json.loads((out / "run.json").read_text())["solver"] == "amg"
    assert json.loads((out / "result" / "metrics.json").read_text())["solver"] == "amg"
    assert "solver amg" in log


def test_run_with_cholesky_on_macos_without_shim_still_stops(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(solver_choice, "is_macos_arm64", lambda: True)
    monkeypatch.setattr(sys, "platform", "darwin")
    fake_shim(monkeypatch, tmp_path, "missing")
    p = make_case(tmp_path, h=2.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "cholesky"
    p.write_text(yaml.safe_dump(d))
    assert main(["run", str(p), "--out", str(tmp_path / "r")]) == 1
    err = capsys.readouterr().err
    assert "cholesky" in err and "build.sh" in err and "settings.solver: amg" in err


def test_estimate_factors():
    chol = estimate_seconds(109_548, 80)
    assert estimate_seconds(109_548, 80, "cholesky") == chol
    assert estimate_seconds(109_548, 80, "pardiso") == chol
    per_it, total = estimate_seconds(109_548, 80, "amg")
    f = solver_choice.AMG_TIME_FACTOR
    # only the time per iteration is scaled: the setup and post-process time stay
    fixed = cli.SETUP_S + cli.POST_S
    assert per_it == pytest.approx(chol[0] * f)
    assert total == pytest.approx(fixed + (chol[1] - fixed) * f)
    assert total < chol[1] * f
    # few iterations: the fixed part dominates, and the estimate stays near it
    assert estimate_seconds(109_548, 3, "amg")[1] < 2 * estimate_seconds(109_548, 3, "cholesky")[1]


# --- run: the solver is recorded, one warning line ----------------------------------------


def run_case(tmp_path, capsys, iterations=3):
    p = make_case(tmp_path, h=2.0, iterations=iterations)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    capsys.readouterr()
    return out


def test_run_with_auto_amg_warns_once_and_records_amg(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(solver_choice, "is_macos_arm64", lambda: False)
    monkeypatch.setattr(solver_choice, "is_linux_x86_64", lambda: True)
    monkeypatch.setattr(solver_choice, "import_pypardiso", Import(None))
    out = run_case(tmp_path, capsys)
    log = (out / "run.log").read_text()
    warns = [l for l in log.splitlines() if "pypardiso" in l]
    assert len(warns) == 1 and warns[0].startswith("warning: solver auto -> amg")
    assert "depends on the problem" in warns[0] and "uv pip install" in warns[0] and "Intel" in warns[0]
    assert json.loads((out / "run.json").read_text())["solver"] == "amg"
    assert json.loads((out / "result" / "metrics.json").read_text())["solver"] == "amg"
    assert (out / "result" / "refine.html").exists()          # refine used the same solver
    assert "solver amg" in log


@needs_pardiso
def test_run_with_auto_pardiso_has_no_warning(tmp_path, capsys):
    out = run_case(tmp_path, capsys)
    assert "warning" not in (out / "run.log").read_text()
    assert json.loads((out / "run.json").read_text())["solver"] == "pardiso"
    assert json.loads((out / "result" / "metrics.json").read_text())["solver"] == "pardiso"
    assert (out / "result" / "refine.html").exists()


# --- pypardiso is imported in one place only -------------------------------------------------


def test_pypardiso_has_one_import_site():
    hits = []
    for path in (REPO / "topopt").rglob("*.py"):
        for no, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"^\s*(import|from)\s+pypardiso|import_module\(.pypardiso", line):
                hits.append(f"{path.relative_to(REPO)}:{no}")
    assert len(hits) == 1 and hits[0].startswith("topopt/solver_choice.py:"), hits


def test_main_dependencies_have_no_mkl():
    text = (REPO / "pyproject.toml").read_text()
    main_deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "pardiso" not in main_deps.lower() and "mkl" not in main_deps.lower()
    assert re.search(r'^pardiso = \["pypardiso==[\d.]+; platform_system == \'Linux\' and '
                     r'platform_machine == \'x86_64\'"\]', text, re.M)


# --- the thread cap ----------------------------------------------------------------------------


THREAD_VARS = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")


def run_py(code, env_extra=None, drop=THREAD_VARS):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()[-1]


# Run `python -m topopt --help` (the real entry module, through runpy), then print what main() did.
CAP_CODE = ("import os, runpy, sys\n"
            "sys.argv = ['topopt', '--help']\n"
            "try:\n"
            "    runpy.run_module('topopt', run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
            "print(os.environ.get('OPENBLAS_NUM_THREADS'), 'numpy' in sys.modules)")


def test_cli_import_does_not_load_numpy():
    # The cap must run before numpy loads. So topopt.cli must not import numpy at the top.
    assert run_py("import sys, topopt.cli; print('numpy' in sys.modules)") == "False"


@pytest.mark.skipif(sys.platform != "linux", reason="the thread cap is for Linux only")
def test_python_dash_m_topopt_caps_the_threads_before_numpy_loads():
    expected = str(max(1, min(4, len(os.sched_getaffinity(0)))))
    assert run_py(CAP_CODE) == f"{expected} False"                       # numpy is not loaded yet
    assert run_py(CAP_CODE, {"OPENBLAS_NUM_THREADS": "7"}) == "7 False"  # the user's value stays


@pytest.mark.skipif(sys.platform != "linux", reason="the thread cap is for Linux only")
@pytest.mark.parametrize("var", ["GOTO_NUM_THREADS", "OMP_NUM_THREADS"])
def test_other_thread_variables_of_the_user_also_stop_the_cap(var):
    assert run_py(CAP_CODE, {var: "1"}) == "None False"


def test_thread_cap_uses_the_usable_cpus(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "numpy")   # the cap only reads sys.modules; the import cache stays
    for v in THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    cli.cap_blas_threads()
    assert os.environ.pop("OPENBLAS_NUM_THREADS") == "2"


def test_thread_cap_does_nothing_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    cli.cap_blas_threads()
    assert "OPENBLAS_NUM_THREADS" not in os.environ


# --- pypardiso installed, but MKL does not load ----------------------------------------------------

LOAD_ERROR = OSError("libmkl_rt.so.2: cannot open shared object file")


def test_pypardiso_problem_tells_missing_from_load_failed(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=None)
    assert solver_choice.pypardiso_problem()[0] == "missing"
    fake_platform("linux", "x86_64", pypardiso=LOAD_ERROR)
    kind, text = solver_choice.pypardiso_problem()
    assert kind == "load_failed" and "libmkl_rt" in text
    fake_platform("linux", "x86_64", pypardiso=ModuleNotFoundError("No module named 'mkl'", name="mkl"))
    assert solver_choice.pypardiso_problem()[0] == "load_failed"   # a module under pypardiso is missing
    fake_platform("linux", "x86_64", pypardiso=types.ModuleType("pypardiso"))
    assert solver_choice.pypardiso_problem() is None


def test_load_failure_gives_another_action_and_names_the_cause(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=LOAD_ERROR)
    assert solver_choice.resolve_solver("auto") == "amg"
    rec = solver_choice.recommendation("auto", "amg")
    assert rec["action"] == "fix_pypardiso_load" and "libmkl_rt" in rec["error"]
    assert "PYPARDISO_MKL_RT" in rec["reason"] and "not installed" not in rec["reason"]
    line = solver_choice.warning_line("auto", "amg")
    assert "libmkl_rt" in line and "installed but it does not load" in line
    with pytest.raises(UserError) as e:
        solver_choice.resolve_solver("pardiso")
    assert "libmkl_rt" in str(e.value) and "installed but it does not load" in str(e.value)
    assert "is not installed" not in str(e.value)
    # the missing package keeps the install action
    fake_platform("linux", "x86_64", pypardiso=None)
    assert solver_choice.recommendation("auto", "amg")["action"] == "install_pypardiso"
    with pytest.raises(UserError, match="not installed"):
        solver_choice.resolve_solver("pardiso")


# --- check shows the facts before it reports a solver that cannot run -------------------------------


def test_check_with_cholesky_on_linux_prints_the_facts_then_the_error(tmp_path, capsys, fake_platform):
    fake_platform("linux", "x86_64", pypardiso=None)
    p = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "cholesky"
    p.write_text(yaml.safe_dump(d))
    assert main(["check", str(p)]) == 1
    cap = capsys.readouterr()
    for key in ("grid shape:", "free dofs:", "time estimate:"):
        assert key in cap.out
    assert re.search(r"^solver:\s+none", cap.out, re.M)
    assert cap.err.startswith("error:") and "pardiso" in cap.err and "amg" in cap.err
    assert main(["check", "--json", str(p)]) == 1
    cap = capsys.readouterr()
    doc = json.loads(cap.out)
    assert doc["solver"]["resolved"] is None and "Accelerate" in doc["solver"]["error"]
    assert doc["free_dofs"] > 0 and doc["time_estimate"]["total_s"] > 0 and "error:" in cap.err


# --- x86_64 Python on a Mac ---------------------------------------------------------------------


def test_x86_64_python_on_a_mac_gets_a_hint(tmp_path, capsys, fake_platform, monkeypatch):
    fake_platform("darwin", "x86_64", pypardiso=None)
    monkeypatch.setattr(solver_choice, "_sysctl_proc_translated", lambda: "1")   # Rosetta (tests/test_intel_mac.py has the other kinds)
    assert "arm64 Python" in solver_choice.platform_hint()
    assert main(["check", str(make_case(tmp_path, h=2.0))]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^solver:\s+amg", out, re.M) and "hint: Python runs as x86_64" in out
    fake_platform("darwin", "arm64", pypardiso=None)
    assert solver_choice.platform_hint() is None
    fake_platform("linux", "x86_64", pypardiso=None)
    assert solver_choice.platform_hint() is None


# --- PARDISO errors: no stale factor ---------------------------------------------------------------


class FakePardiso:
    """Stand-in for pypardiso.PyPardisoSolver. `fail` makes the next factor call raise."""
    fail = None

    def __init__(self, mtype=2):
        self.phases, self.frees, self.phase = [], 0, None

    def set_phase(self, phase):
        self.phase = phase
        self.phases.append(phase)

    def free_memory(self, everything=False):
        self.frees += 1

    def _call_pardiso(self, A, b):
        if self.phase in (12, 22) and FakePardiso.fail:
            raise RuntimeError(FakePardiso.fail)
        return np.array(b, dtype=float)


@pytest.fixture
def fake_pardiso_solver(fake_platform, monkeypatch):
    module = types.ModuleType("pypardiso")
    module.PyPardisoSolver = FakePardiso
    imp = fake_platform("linux", "x86_64", pypardiso=module)
    monkeypatch.setattr(solver_module, "import_pypardiso", imp)   # solver.py imported the name
    FakePardiso.fail = None
    yield LinearSolver("pardiso")
    FakePardiso.fail = None


def test_failed_pardiso_factor_leaves_no_stale_factor(fake_pardiso_solver):
    ls = fake_pardiso_solver
    edof, idx, free, node, ke = _beam(6, 3, 3)
    from topopt.fe import Assembler
    asm = Assembler(edof, ke, free)
    K = asm.K(np.ones(edof.shape[0]))
    ls.factor(K)
    assert ls.solve(np.ones(K.shape[0])).shape == (K.shape[0],)
    ls.factor(asm.K(np.ones(edof.shape[0]) * 2))
    assert 22 in ls._ps.phases   # the second factor is numeric-only
    FakePardiso.fail = "The Pardiso solver failed with error code -4"
    frees = ls._ps.frees
    with pytest.raises(RuntimeError, match="not positive definite"):
        ls.factor(asm.K(np.ones(edof.shape[0]) * 3))
    assert ls._ps.frees > frees and ls._pd_A is None and not ls._pd_ready
    with pytest.raises(RuntimeError, match="no factor"):    # not a solve with the old factor
        ls.solve(np.ones(K.shape[0]))
    # the solver recovers: the next factor starts with a new analysis (phase 12)
    FakePardiso.fail = None
    ls._ps.phases.clear()
    ls.factor(K)
    assert ls._ps.phases[0] == 12 and ls.solve(np.ones(K.shape[0])).shape == (K.shape[0],)


def test_pardiso_solve_after_free_is_a_clear_error(fake_pardiso_solver):
    ls = fake_pardiso_solver
    edof, idx, free, node, ke = _beam(6, 3, 3)
    from topopt.fe import Assembler
    ls.factor(Assembler(edof, ke, free).K(np.ones(edof.shape[0])))
    ls.free()
    with pytest.raises(RuntimeError, match="no factor"):
        ls.solve(np.ones(ls.n))


def test_pardiso_memory_error_code_is_translated():
    from topopt.solver import _pardiso_reason
    assert "not enough memory" in _pardiso_reason(RuntimeError("failed with error code -2. See docs"))
    assert _pardiso_reason(RuntimeError("other")) == "other"


def test_check_with_pardiso_setting_and_a_load_failure_names_the_cause(tmp_path, capsys, fake_platform):
    fake_platform("linux", "x86_64", pypardiso=LOAD_ERROR)
    p = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "pardiso"
    p.write_text(yaml.safe_dump(d))
    assert main(["check", str(p)]) == 1
    cap = capsys.readouterr()
    assert "free dofs:" in cap.out and "libmkl_rt" in cap.out and "libmkl_rt" in cap.err
    assert "cannot run on" not in cap.out and "cholesky" not in cap.out
    assert "cholesky" not in cap.err


def test_warning_line_does_not_promise_one_factor(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=None)
    line = solver_choice.warning_line("auto", "amg")
    assert "depends on the problem" in line and "4.1x" not in line


# --- GPU first: `run_solver` (the solver that a run uses first; CONTRACTS.md sections 4 and 20) -----------------------------------------


def _gpu(index=0, name="Test GPU", kind="GPU", fp64=False, **kw):
    from topopt.gpu_mg.device import DeviceInfo
    facts = dict(max_work_group_size=256, max_mem_alloc=2**30, global_mem=2**32, local_mem=32768, constant_mem=65536,
                 compute_units=8)
    return DeviceInfo(index=index, platform="Test Platform", name=name, kind=kind, fp64=fp64, version="OpenCL 1.2",
                      **{**facts, **kw})


PASSED = {"passed": True, "rel_diff": 3.0e-11, "seconds": 1.5, "free_dofs": 3888, "tol": 1e-8, "max_diff": 1e-6,
          "device": "Test GPU", "case": "C", "error": None}


@pytest.fixture
def gpu_world(monkeypatch):
    """`gpu_world([devices], self_test=dict)` fakes pyopencl, the device list and the self-test. The platform rule gives amg.

    The test session sets `TOPOPT_AUTO_GPU=0` (tests/conftest.py); the fixture deletes it, so the fake devices are the ones that `auto` looks at."""
    from topopt.gpu_mg import device
    from helpers import in_process_probe
    in_process_probe(monkeypatch)       # the product asks a child process; the fakes below act in this process
    monkeypatch.delenv("TOPOPT_AUTO_GPU", raising=False)
    calls = {"self_test": 0}

    def set_world(devices, self_test=PASSED):
        from topopt.gpu_mg import probe
        probe.forget()      # a test can set a world twice
        def fake_self_test(dev, case):
            calls["self_test"] += 1
            return dict(self_test)
        monkeypatch.setattr(device, "pyopencl_problem", lambda: None)
        monkeypatch.setattr(device, "enumerate_devices", lambda: list(devices))
        monkeypatch.setattr(device, "self_test", fake_self_test)
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(solver_choice, "pypardiso_available", lambda: False)
        return calls
    return set_world


def test_auto_picks_gpu_mg_on_a_gpu_that_passes_the_self_test(gpu_world):
    gpu_world([_gpu()])
    c = solver_choice.run_solver("auto")
    assert c.solver == "gpu-mg" and c.auto_gpu and c.fallback == "amg" and c.cpu == "amg" and c.reason is None
    assert c.device["name"] == "Test GPU" and c.case == "C" and c.self_test["passed"] is True
    line = c.line()
    assert line.startswith("solver: auto -> gpu-mg on Test GPU (GPU, platform Test Platform, case C)")
    assert "self-test passed" in line and "3.0e-11" in line and "go on with amg" not in line and "goes on with amg" in line
    assert c.as_dict()["fallback"] == "amg" and c.as_dict()["line"] == line
    assert solver_choice.resolve_solver("auto") == "amg"          # the matrix rule keeps its meaning


def test_auto_takes_the_cpu_rule_when_only_a_cpu_device_exists(gpu_world):
    calls = gpu_world([_gpu(0, "Pocl CPU", "CPU", True)])
    c = solver_choice.run_solver("auto")
    assert c.solver == "amg" and not c.auto_gpu and c.fallback is None and c.device is None
    assert "Pocl CPU is a CPU device, not a GPU" in c.reason and c.line().startswith("solver: auto -> amg (")
    assert calls["self_test"] == 0           # no self-test for a device that does not count


def test_an_accelerator_does_not_count_as_a_gpu(gpu_world):
    gpu_world([_gpu(0, "Some Accelerator", "ACCELERATOR")])
    c = solver_choice.run_solver("auto")
    assert c.solver == "amg" and "ACCELERATOR device, not a GPU" in c.reason


def test_auto_takes_the_gpu_before_a_cpu_device(gpu_world):
    gpu_world([_gpu(0, "Pocl CPU", "CPU", True), _gpu(1, "Real GPU")])
    assert solver_choice.run_solver("auto").device["name"] == "Real GPU"


def test_the_device_of_the_yaml_decides(gpu_world):
    gpu_world([_gpu(0, "Real GPU"), _gpu(1, "Pocl CPU", "CPU", True)])
    cfg = types.SimpleNamespace(buckling=types.SimpleNamespace(min_factor=None),
                                settings=types.SimpleNamespace(gpu_device="pocl", gpu_case="auto"))
    c = solver_choice.run_solver("auto", cfg)
    assert c.solver == "amg" and "Pocl CPU is a CPU device" in c.reason
    cfg.settings.gpu_device = "nothing like it"
    c = solver_choice.run_solver("auto", cfg)       # a device name that fits none is a fact, not a stop
    assert c.solver == "amg" and "nothing like it" in c.reason


def test_pyopencl_missing_gives_the_cpu_rule_with_the_reason(gpu_world, monkeypatch):
    from topopt.gpu_mg import device
    gpu_world([_gpu()])
    monkeypatch.setattr(device, "pyopencl_problem", lambda: ("missing", "No module named 'pyopencl'"))
    c = solver_choice.run_solver("auto")
    assert c.solver == "amg" and not c.auto_gpu
    assert "pyopencl is not usable here (missing: No module named 'pyopencl')" in c.reason


def test_no_opencl_device_gives_the_cpu_rule(gpu_world):
    gpu_world([])
    c = solver_choice.run_solver("auto")
    assert c.solver == "amg" and "no OpenCL device" in c.reason


def test_a_driver_fault_does_not_stop_auto(gpu_world, monkeypatch):
    from topopt.gpu_mg import device
    gpu_world([_gpu()])

    def boom():
        raise RuntimeError("driver fault")
    monkeypatch.setattr(device, "enumerate_devices", boom)
    c = solver_choice.run_solver("auto")
    assert c.solver == "amg" and "RuntimeError: driver fault" in c.reason


def test_a_gpu_without_a_feature_or_with_a_case_it_cannot_run_gives_the_cpu_rule(gpu_world):
    gpu_world([_gpu(local_mem=1024)])
    c = solver_choice.run_solver("auto")
    assert c.solver == "amg" and "cannot run gpu-mg" in c.reason and "local memory is 1024 bytes" in c.reason
    gpu_world([_gpu(fp64=False)])
    cfg = types.SimpleNamespace(buckling=types.SimpleNamespace(min_factor=None),
                                settings=types.SimpleNamespace(gpu_device="auto", gpu_case="B"))
    c = solver_choice.run_solver("auto", cfg)
    assert c.solver == "amg" and "no 64-bit float" in c.reason


def test_a_failed_self_test_gives_the_cpu_rule_with_the_numbers(gpu_world):
    gpu_world([_gpu()], self_test={**PASSED, "passed": False, "rel_diff": 4.2e-4})
    c = solver_choice.run_solver("auto")
    assert c.solver == "amg" and not c.auto_gpu and c.device is None
    assert "self-test of gpu-mg on Test GPU" in c.reason and "did not pass" in c.reason
    assert "4.20e-04" in c.reason and "above the limit 1e-06" in c.reason
    gpu_world([_gpu()], self_test={**PASSED, "passed": False, "rel_diff": None, "error": "Error: kernel build failed"})
    assert "error Error: kernel build failed" in solver_choice.run_solver("auto").reason


def test_the_buckling_row_keeps_auto_on_the_cpu_rule(gpu_world):
    calls = gpu_world([_gpu()])
    cfg = types.SimpleNamespace(buckling=types.SimpleNamespace(min_factor=2.0),
                                settings=types.SimpleNamespace(gpu_device="auto", gpu_case="auto"))
    c = solver_choice.run_solver("auto", cfg)
    assert c.solver == "amg" and "buckling row (buckling.min_factor) needs a direct factor" in c.reason
    assert calls["self_test"] == 0


def test_explicit_values_are_unchanged(gpu_world):
    gpu_world([_gpu()])
    for setting in ("amg", "gpu-mg"):
        c = solver_choice.run_solver(setting)
        assert c.solver == setting and c.fallback is None and not c.auto_gpu and c.line() is None
    with pytest.raises(UserError, match="solver 'cholesky' needs Apple Accelerate"):
        solver_choice.run_solver("cholesky")
    with pytest.raises(UserError, match="must be one of"):
        solver_choice.run_solver("nope")


def test_auto_gpu_0_keeps_auto_on_the_cpu_rule_and_names_the_variable(gpu_world, monkeypatch):
    """`TOPOPT_AUTO_GPU=0` is the first check of `auto`: pyopencl is not asked, no device is listed, no self-test runs."""
    from topopt.gpu_mg import device
    calls = gpu_world([_gpu()])
    assert solver_choice.run_solver("auto").solver == "gpu-mg"        # the same world, without the variable
    calls["self_test"] = 0

    def must_not_run(*a, **k):
        raise AssertionError("the variable is the first check: this must not run")
    monkeypatch.setattr(device, "pyopencl_problem", must_not_run)
    monkeypatch.setattr(device, "enumerate_devices", must_not_run)
    monkeypatch.setenv("TOPOPT_AUTO_GPU", "0")
    assert device.auto_gpu("auto", "auto") == (None, "TOPOPT_AUTO_GPU=0 is set in the environment")
    c = solver_choice.run_solver("auto")
    assert c.solver == "amg" and c.cpu == "amg" and not c.auto_gpu and c.fallback is None and c.device is None and c.self_test is None
    assert c.reason == "TOPOPT_AUTO_GPU=0 is set in the environment"
    assert c.line() == "solver: auto -> amg (TOPOPT_AUTO_GPU=0 is set in the environment)"
    assert c.as_dict()["cpu_solver"] == "amg" and c.as_dict()["line"] == c.line()
    assert calls["self_test"] == 0
    assert solver_choice.resolve_solver("auto") == "amg"


@pytest.mark.parametrize("value", ["1", "", "false", "00", " 0"])
def test_any_other_value_of_auto_gpu_gives_the_normal_rule(gpu_world, monkeypatch, value):
    gpu_world([_gpu()])
    monkeypatch.setenv("TOPOPT_AUTO_GPU", value)
    c = solver_choice.run_solver("auto")
    assert c.solver == "gpu-mg" and c.auto_gpu and c.reason is None


def test_auto_gpu_0_does_not_change_an_explicit_value(gpu_world, monkeypatch):
    gpu_world([_gpu()])
    monkeypatch.setenv("TOPOPT_AUTO_GPU", "0")
    c = solver_choice.run_solver("gpu-mg")
    assert c.solver == "gpu-mg" and c.fallback is None and not c.auto_gpu and c.line() is None
    assert solver_choice.run_solver("amg").solver == "amg"


def test_auto_gpu_0_reaches_a_subprocess():
    """A child process has the same environment, so its `auto` is on the CPU rule too (the check is before pyopencl loads)."""
    code = ("from topopt.gpu_mg import device; print(device.auto_gpu('auto', 'auto'))")
    env = {**os.environ, "TOPOPT_AUTO_GPU": "0"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "(None, 'TOPOPT_AUTO_GPU=0 is set in the environment')"


def test_explicit_gpu_mg_without_a_device_keeps_its_error(gpu_world):
    gpu_world([])
    with pytest.raises(UserError, match="no OpenCL device"):
        solver_choice.run_solver("gpu-mg")


def test_the_self_test_runs_once_for_each_device_and_case(monkeypatch):
    """`device.self_test` caches the result for the process; `forget_devices` drops it."""
    from topopt.gpu_mg import device, selftest
    calls = []
    monkeypatch.setattr(selftest, "run", lambda dev, case, tol, max_diff: calls.append((dev.name, case, tol, max_diff)) or dict(PASSED))
    device.forget_devices()
    try:
        a, b = _gpu(0, "A"), _gpu(1, "B")
        for _ in range(3):
            device.self_test(a, "C")
        device.self_test(a, "B")
        device.self_test(b, "C")
        assert calls == [("A", "C", 1e-8, 1e-6), ("A", "B", 1e-8, 1e-6), ("B", "C", 1e-8, 1e-6)]
        device.forget_devices()
        device.self_test(a, "C")
        assert len(calls) == 3         # the passes are on the drive: the next process (here, after the forget) does not test again
        device.selftest_path().unlink()
        device.forget_devices()
        device.self_test(a, "C")
        assert len(calls) == 4
    finally:
        device.forget_devices()


def test_the_self_test_numbers_are_module_constants():
    from topopt.gpu_mg import device
    assert device.SELFTEST_TOL == 1e-8 and device.SELFTEST_MAX_DIFF == 1e-6


def test_check_data_names_the_gpu_choice(gpu_world, tmp_path):
    calls = gpu_world([_gpu()])
    d = cli.check_data(make_case(tmp_path, h=2.0, iterations=3))
    assert d["solver"]["resolved"] == "gpu-mg" and d["solver"]["choice"]["fallback"] == "amg"
    assert d["solver"]["cpu_solver"] == "amg"          # the CPU rule is its own field: CI compares it, and the pick of auto is information
    assert d["solver"]["choice"]["line"].startswith("solver: auto -> gpu-mg on Test GPU")
    assert d["solver"]["choice"]["device"]["name"] == "Test GPU" and d["time_estimate"]["solver"] == "gpu-mg"
    assert calls["self_test"] == 1


def test_check_prints_the_choice_and_the_reason(gpu_world, tmp_path, capsys):
    yaml_path = make_case(tmp_path, h=2.0, iterations=3)
    gpu_world([_gpu()])
    assert main(["check", str(yaml_path)]) == 0
    out = capsys.readouterr().out
    assert "solver:         gpu-mg  (settings.solver auto" in out and "solver choice:  auto -> gpu-mg on Test GPU" in out
    gpu_world([_gpu(0, "Pocl CPU", "CPU", True)])
    assert main(["check", str(yaml_path)]) == 0
    out = capsys.readouterr().out
    assert "solver:         amg" in out and "solver choice:  auto -> amg (the OpenCL device Pocl CPU is a CPU device, not a GPU" in out


def test_check_run_inputs_returns_the_choice(gpu_world, tmp_path):
    gpu_world([_gpu()])
    problem, choice = cli.check_run_inputs(load_config(make_case(tmp_path, h=2.0, iterations=3)))
    assert choice.solver == "gpu-mg" and choice.fallback == "amg" and problem is not None



def test_the_forced_failure_example_loads_and_asks_for_numbers_that_no_device_reaches():
    """examples/hook/hook_gpu_fallback.yaml (P9-4d): the config accepts the gpu numbers; a 64-bit residual cannot go below about 1e-16 and the backward error of
    a solve at the rounding floor is about 4e-17, so a true relative residual of 1e-30 and a backward error of 1e-30 are out of reach in case B and in case C.
    The rule is on (a backward error above 0): with 0 the residual of the PCG recurrence decides in case B, and that residual can go below the true one."""
    cfg = load_config(Path(__file__).resolve().parent.parent / "examples" / "hook" / "hook_gpu_fallback.yaml")
    s = cfg.settings
    assert s.solver == "auto" and s.gpu_case == "auto" and s.gpu_tol == 1e-30 and s.gpu_backward_error == 1e-30
    assert 0 < s.gpu_tol < 1e-17 and 0 < s.gpu_backward_error < 1e-17 and s.iterations <= 20      # below any FP64 floor; a short run
