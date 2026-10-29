"""The solver gpu-mg in robust mode (G1e; CONTRACTS.md sections 20 and 22b).

`robust.ROBUST_SOLVERS` lists the solvers that have a parity test. This is the test of gpu-mg: a copy of
`tests/test_robust.py::test_every_solver_gives_the_robust_compliance_of_a_scipy_solve` that runs on an OpenCL device
(pocl on the CPU on this project's Linux host, see `tests/test_gpu_mg.py`) and uses the tolerances of `tests/test_gpu_mg.py`.
The reference is `scipy.sparse.linalg.spsolve` of the same eroded system.
"""
import json

import numpy as np
import pytest
import yaml

pytest.importorskip("pyopencl")
pytestmark = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1

from test_gpu_mg import CASES, DirectModel, GRAD_RTOL, RUN_RTOL, SMALL, SOLVE_RTOL, case_yaml, dev, need_case  # noqa: E402,F401
from test_robust import reference_by_spsolve, settings_of  # noqa: E402
from topopt import robust  # noqa: E402
from topopt.cli import check_data, main  # noqa: E402
from topopt.config import load_config  # noqa: E402
from topopt.gpu_mg.model import GpuModel  # noqa: E402
from topopt.problem import build_problem  # noqa: E402
from topopt.solver_choice import resolve_solver  # noqa: E402


def robust_model(tmp_path, case):
    """The fixture of `test_robust.py` (1,080 free dofs) with `solver: gpu-mg`, and the reference model: the same problem with an
    assembled matrix. Both models filter and project the same way, so the eroded system is the same."""
    prob = build_problem(load_config(settings_of(tmp_path, solver="gpu-mg", gpu_case=case, robust="eroded", eta_eroded=0.55)))
    m = GpuModel(prob, None, SMALL)
    assert m.solver_name == "gpu-mg" and m.solver.case == case and m.robust is not None
    return m, DirectModel(prob)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("beta", [1.0, 8.0])
def test_gpu_mg_gives_the_robust_compliance_of_a_scipy_solve(tmp_path, dev, case, beta):
    """The compliance and the slope of gpu-mg in robust mode equal those of a `spsolve` of the same eroded system.

    Measured on pocl (1,080 free dofs, eta_e 0.55, this design, beta 1 and 8): case C differs from the spsolve value by 3.1e-11
    and 2.9e-10 in the compliance and by 1.7e-10 and 6.3e-10 in the slope (relative to its largest entry); case B by 5e-14 and
    8e-14, and by 8.9e-11 and 9.4e-11. The bounds are the ones of the gpu-mg tests (`SOLVE_RTOL`, `GRAD_RTOL`: 1e-8).
    """
    need_case(case, dev)
    m, ref = robust_model(tmp_path, case)
    x = np.random.default_rng(4).uniform(0.3, 0.7, m.n_design)
    c_ref, dc_ref = reference_by_spsolve(ref, x, beta)
    c, dc, v, _ = m.evaluate(x, beta)
    assert abs(c - c_ref) / c_ref < SOLVE_RTOL
    assert np.abs(dc - dc_ref).max() < GRAD_RTOL * np.abs(dc_ref).max()
    m.solver.free()


def test_the_eroded_density_reaches_the_device(tmp_path, dev):
    """The parity check has power: the robust value of gpu-mg is far from its value for the nominal density."""
    m, ref = robust_model(tmp_path, "C")
    x = np.random.default_rng(4).uniform(0.3, 0.7, m.n_design)
    c, *_ = m.evaluate(x, 8.0)
    rho_tilde, rho = m.density(x, 8.0)
    c_nominal = float(m.weights @ m.compliance(m.solve(m.stiffness(rho)), rho)[0])
    assert abs(c_nominal - c) / c > 1e-3
    m.solver.free()


@pytest.mark.parametrize("mode", ["eroded", "off"])
def test_a_solve_that_does_not_converge_says_when_it_solved_the_eroded_design(tmp_path, dev, mode):
    """Review n7. The stop of G1d is a fact with options. In robust mode it also says which design was solved."""
    from topopt.config import UserError
    from topopt.gpu_mg.model import GpuParams
    prob = build_problem(load_config(settings_of(tmp_path, solver="gpu-mg", robust=mode)))
    m = GpuModel(prob, None, GpuParams(min_free=300, max_iter_first=1, max_iter=1, max_refine=1))
    with pytest.raises(UserError, match=r"did not reach the relative residual.*Options.*'amg'") as e:
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    assert ("robust mode: the solved design is the eroded design" in str(e.value)) is (mode == "eroded")
    m.solver.free()


def test_gpu_mg_has_a_parity_test_so_robust_mode_gives_no_solver_fact(tmp_path, dev):
    assert "gpu-mg" in robust.ROBUST_SOLVERS and robust.solver_fact("gpu-mg") is None
    d = check_data(settings_of(tmp_path, solver="gpu-mg", robust="eroded"))
    assert d["solver"]["resolved"] == "gpu-mg"
    assert not [w for w in d["warnings"] if "parity test" in w]
    assert d["derived"]["robust"]["solver"] == "gpu-mg" and d["derived"]["robust"]["solver_tested"] is True


def test_a_robust_run_with_gpu_mg_equals_the_run_of_the_default_solver(tmp_path, dev):
    """A whole run (3 load cases, beta 1 to 8, the stress step of the refine page): the eroded history, and the nominal and the
    eroded compliance of `robust.json` (the extra solve of `robust.finish_run`) equal those of the default solver."""
    p = case_yaml(tmp_path, solver="auto", h=2.0, iterations=12, three_cases=True)
    d = yaml.safe_load(p.read_text())
    d["settings"].update({"robust": "eroded", "eta_eroded": 0.55, "beta_schedule": [[0, 1], [3, 2], [6, 4], [9, 8]]})
    p.write_text(yaml.safe_dump(d))
    assert main(["run", str(p), "--out", str(tmp_path / "r_ref"), "--quiet", "--solver", "auto"]) == 0
    assert main(["run", str(p), "--out", str(tmp_path / "r_gpu"), "--quiet", "--solver", "gpu-mg"]) == 0
    hist = {k: np.array([json.loads(line)["compliance"] for line in (tmp_path / k / "history.jsonl").read_text().splitlines()])
            for k in ("r_ref", "r_gpu")}
    summary = {k: json.loads((tmp_path / k / "robust.json").read_text()) for k in ("r_ref", "r_gpu")}
    direct = resolve_solver("auto") in ("cholesky", "pardiso")      # AMG is an iteration: a looser limit
    tol = RUN_RTOL if direct else 1e-3
    assert len(hist["r_gpu"]) == 13
    assert np.max(np.abs(hist["r_gpu"] - hist["r_ref"]) / hist["r_ref"]) < tol
    for key in ("compliance_eroded", "compliance_nominal"):
        assert abs(summary["r_gpu"][key] - summary["r_ref"][key]) / summary["r_ref"][key] < tol
    log = (tmp_path / "r_gpu" / "run.log").read_text()
    assert "no parity test" not in log and "robust: eta_n" in log and "refine failed" not in log
    run_json = json.loads((tmp_path / "r_gpu" / "run.json").read_text())
    assert run_json["solver"] == "gpu-mg" and run_json["derived"]["robust"]["solver_tested"] is True
    timing = [json.loads(line) for line in (tmp_path / "r_gpu" / "gpu_mg.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in timing] == ["main"] * 13 + ["trial"]   # the last line: the solve of the nominal design at the end


# --- a resume that runs no pass (G1e review m2) -------------------------------------------------------------------------


def straight_tuples(folder):
    return [(r["line"], r["kind"], r["iteration"], r["solve"])
            for r in (json.loads(line) for line in (folder / "gpu_mg.jsonl").read_text().splitlines())]


@pytest.mark.parametrize("how", ["finished", "killed_after_the_last_checkpoint"])
def test_a_resume_that_runs_no_pass_gives_the_timing_file_of_a_run_that_was_not_stopped(tmp_path, dev, how):
    """The checkpoint of the last pass is `complete`: the resume runs no pass, only the nominal solve of `robust.finish_run` and the
    post step. The solve gets the iteration of the last pass (before the fix: 0), and the old line of that solve goes (before the
    fix: a finished run got a second one)."""
    from test_checkpoint import finish, killed, log, resume
    p = case_yaml(tmp_path, case="C", h=2.0, iterations=6)
    d = yaml.safe_load(p.read_text())
    d["settings"].update({"robust": "eroded", "eta_eroded": 0.55, "beta_schedule": [[0, 1], [2, 2], [4, 4], [6, 8]]})
    p.write_text(yaml.safe_dump(d))
    straight, out = tmp_path / "straight", tmp_path / "r"
    finish(p, straight)
    expected = [(i, "main", i, i) for i in range(7)] + [(7, "trial", 6, 7)]        # 7 passes, then the nominal solve
    assert straight_tuples(straight) == expected
    if how == "finished":
        finish(p, out)
    else:
        killed(p, out, "after_checkpoint:6")
        assert straight_tuples(out) == expected[:7]                                  # the kill came before the nominal solve
    resume(p, out)
    assert "No pass is left to run" in log(out)
    assert straight_tuples(out) == expected
    assert json.loads((out / "status.json").read_text())["state"] == "done"
    if how == "finished":
        assert "drops 1 line of the solve at the end of the run" in log(out)
        resume(p, out)                                                               # a second resume: the same file
        assert straight_tuples(out) == expected
