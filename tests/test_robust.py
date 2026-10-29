"""Tests for A3b (CONTRACTS.md section 22b): the robust erosion form, low-cost.

Volume, snapshots and final mesh use the nominal design (eta_n 0.5). The stiffness and the compliance
use the eroded design (eta_e). One solve for each pass, and one more at the end.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_solid_filter import CAP, sub, write_case
from topopt import robust
from topopt.cli import backend_availability, check_data, estimate_seconds, main
from topopt.config import UserError, load_config
from topopt.filter import heaviside_deriv, heaviside_project
from topopt.formulation import DesignPoint, Formulation, WRT, volume_constraint, weighted_compliance
from topopt.mma import Mma
from topopt.optimize import Model, beta_at, formulation_from_config, minimize, run
from topopt.post import case_stiffness
from topopt.problem import build_problem
from topopt.snapshot import RunWriter

SCHEDULE = [[0, 1], [30, 2], [50, 4], [65, 8]]   # the default beta schedule: the highest beta is 8
TOOLS = Path(__file__).resolve().parents[1] / "tools"


def settings_of(tmp_path, **settings):
    settings = {"filter_radius": 3.0, "beta_schedule": SCHEDULE, **settings}
    return write_case(tmp_path, solids=[CAP], **settings)


def make_model(tmp_path, **settings):
    return Model(build_problem(load_config(settings_of(tmp_path, **settings))))


def tight(m):
    """AMG-CG is not exact at its default tolerance: tighten it for a check of values."""
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13
    return m


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the keys ----------------------------------------------------------------


def test_default_is_off_and_the_model_has_no_robust_object(tmp_path):
    s = load_config(settings_of(tmp_path)).settings
    assert (s.robust, s.eta_eroded) == ("off", None)
    m = make_model(sub(tmp_path, "m"))
    assert m.robust is None
    rho_tilde, rho = m.density(np.full(m.n_design, 0.4), 2.0)
    assert m.stiffness_density(rho_tilde, rho, 2.0) is rho     # the same array: the default path is unchanged
    m.solver.free()


@pytest.mark.parametrize("value, mode", [("eroded", "eroded"), (True, "eroded"), (False, "off"), ("off", "off")])
def test_robust_key_values(tmp_path, value, mode):
    assert load_config(settings_of(tmp_path, robust=value)).settings.robust == mode


@pytest.mark.parametrize("bad", ["worst", "dilated", 1, 0.5, None])
def test_robust_key_rejects_other_values(tmp_path, bad):
    with pytest.raises(UserError, match="settings.robust' must be off, eroded or wall"):
        load_config(settings_of(tmp_path, robust=bad))


@pytest.mark.parametrize("bad", [0.5, 0.3, 1.0, 1.2, "high", True, float("nan")])
def test_eta_eroded_must_be_between_the_nominal_threshold_and_one(tmp_path, bad):
    with pytest.raises(UserError, match="eta_eroded"):
        load_config(settings_of(tmp_path, robust="eroded", eta_eroded=bad))


def test_eta_eroded_is_read_and_the_object_knows_its_source(tmp_path):
    p = settings_of(tmp_path, robust="eroded", eta_eroded=0.62)
    rb = robust.from_settings(load_config(p).settings)
    assert (rb.eta_n, rb.eta_e, rb.source) == (0.5, 0.62, "settings")
    rb2 = robust.from_settings(load_config(settings_of(sub(tmp_path, "d"), robust="eroded")).settings)
    assert (rb2.eta_e, rb2.source) == (robust.ETA_ERODED_DEFAULT, "default")
    assert robust.from_settings(load_config(settings_of(sub(tmp_path, "o"))).settings) is None


# --- the two designs ---------------------------------------------------------


@pytest.mark.parametrize("eta", [0.55, 0.6])
def test_eroded_design_is_below_the_nominal_one_and_volume_is_nominal(tmp_path, eta):
    m = tight(make_model(tmp_path, robust="eroded", eta_eroded=eta))
    d = tight(make_model(sub(tmp_path, "d")))
    rng = np.random.default_rng(3)
    x = rng.uniform(0.2, 0.8, m.n_design)
    for beta in (1.0, 4.0, 8.0):
        rho_tilde, rho = m.density(x, beta)
        rho_e = m.stiffness_density(rho_tilde, rho, beta)
        assert (rho_e <= rho + 1e-15).all() and rho_e.mean() < rho.mean()
        assert np.allclose(rho_e, heaviside_project(rho_tilde, beta, eta), rtol=0, atol=0)
        c_e, _dc, v, _dv = m.evaluate(x, beta)
        c_n, _, v_n, _ = d.evaluate(x, beta)
        assert v == v_n == float(rho.mean())            # the volume is the nominal one in both modes
        assert c_e > c_n                                # less material: the eroded compliance is higher
    m.solver.free()
    d.solver.free()


@pytest.mark.equivalence
def test_the_pipeline_projection_formula():
    """The projection of the pipeline (`pp_core_final.densities`) is the same function at eta_n and at eta_e."""
    rt = np.linspace(0, 1, 11)
    for eta in (0.5, 0.55, 0.6):
        a, b = np.tanh(6.0 * eta), np.tanh(6.0 * (1 - eta))
        assert np.array_equal(heaviside_project(rt, 6.0, eta), (a + np.tanh(6.0 * (rt - eta))) / (a + b))
    assert robust.projected(0.55, 6.0, 0.55) == pytest.approx(0.5016, abs=5e-5)    # KB: 0.5016 at the threshold
    assert robust.projected(0.5, 6.0, 0.5) == pytest.approx(0.5, abs=1e-15)
    assert robust.projected(0.3, 3.0, 0.6) == pytest.approx(float(heaviside_project(np.array([0.3]), 3.0, 0.6)[0]), abs=1e-15)


# --- the gradient (central finite differences) -------------------------------

# Measured (this file, Linux, PARDISO and AMG at rtol 1e-13; 12 voxels, 4 seeds): the best step of
# 1e-3, 1e-4 and 1e-5 gives a relative error of 2.4e-8 to 1.2e-7 for the compliance and 3e-10 to 5e-8
# for the volume, at beta 1 and 8 and eta_e 0.55 and 0.6. The noise of the solves at a fixed step
# is much larger: 1e-5 at the step 1e-6 and 5e-4 at the step 1e-6 for eta_e 0.6 at beta 16. A wrong slope
# (the nominal one for the compliance) gives 1e-2 and above (`test_fd_check_sees_a_wrong_slope`).
FD_LIMIT = 1e-6


def fd_errors(m, x, beta, sample):
    """Best-step relative errors (compliance, volume) of a central difference for the voxels in `sample`."""
    _c, dc, _v, dv = m.evaluate(x, beta)
    best = np.full((2, sample.size), np.inf)
    for eps in (1e-3, 1e-4, 1e-5):
        for k, e in enumerate(sample):
            xp, xm = x.copy(), x.copy()
            xp[e] += eps
            xm[e] -= eps
            cp, _, vp, _ = m.evaluate(xp, beta)
            cm, _, vm, _ = m.evaluate(xm, beta)
            err = [abs((cp - cm) / (2 * eps) - dc[e]) / abs(dc[e]), abs((vp - vm) / (2 * eps) - dv[e]) / abs(dv[e])]
            best[:, k] = np.minimum(best[:, k], err)
    return best


@pytest.mark.parametrize("eta", [0.55, 0.6])
def test_gradients_match_finite_differences_at_beta_1_and_the_highest_beta(tmp_path, eta):
    """FD check of the whole chain in robust mode: filter, both projections, SIMP, solve. Solid voxels are there."""
    m = tight(make_model(tmp_path, robust="eroded", eta_eroded=eta))
    top = max(b for _, b in load_config(settings_of(sub(tmp_path, "c"))).settings.beta_schedule)
    assert top == 8.0
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.7, m.n_design)
    sample = rng.choice(m.n_design, size=12, replace=False)
    for beta in (1.0, top):
        best = fd_errors(m, x, beta, sample)
        assert best.max() < FD_LIMIT, f"eta {eta} beta {beta}: best-step error {best.max():.2e}"
    m.solver.free()


def test_gradient_with_the_ignore_rule_of_the_filter_at_solid_voxels(tmp_path):
    """Robust mode with `filter_solid_neighbours: ignore` (A3a): the rows of F sum to 1, b is 0, the chain is as before."""
    m = tight(make_model(tmp_path, robust="eroded", eta_eroded=0.55, filter_solid_neighbours="ignore"))
    assert m.problem.config.settings.filter_solid_neighbours == "ignore"
    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 0.7, m.n_design)
    sample = rng.choice(m.n_design, size=8, replace=False)
    for beta in (1.0, 8.0):
        best = fd_errors(m, x, beta, sample)
        assert best.max() < FD_LIMIT, f"ignore rule, beta {beta}: best-step error {best.max():.2e}"
    m.solver.free()


def test_fd_check_sees_a_wrong_slope(tmp_path, monkeypatch):
    """The check has power: the nominal slope for the compliance gives an error far above the limit."""
    m = tight(make_model(tmp_path, robust="eroded", eta_eroded=0.6))
    monkeypatch.setattr(robust.Robust, "eroded_slope",
                        lambda self, rho_tilde, beta: heaviside_deriv(rho_tilde, beta, 0.5))
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.7, m.n_design)
    best = fd_errors(m, x, 4.0, rng.choice(m.n_design, size=6, replace=False))
    assert best[0].max() > 1e-2
    m.solver.free()


def test_formulation_gradients_use_the_eroded_slope_for_compliance_and_the_nominal_for_volume(tmp_path):
    """The path of the optimizer (`Formulation`, `DesignPoint`, `gradient_wrt_x`) gives the same slopes as `evaluate`."""
    m = tight(make_model(tmp_path, robust="eroded", eta_eroded=0.55))
    form = formulation_from_config(m.problem.config, m.n_design)
    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 0.7, m.n_design)
    beta = 4.0
    rho_tilde, rho = m.density(x, beta)
    U = m.solve(m.stiffness(m.stiffness_density(rho_tilde, rho, beta)))
    point = m.design_point(x, beta, rho_tilde, rho, U)
    assert point.rho is rho and point.rho_stiff is not rho
    state = Mma(m.n_design, form.n_rows, 0.2, xmin=form.xmin).state
    ev = form.evaluate(point, state)
    c, dc, v, dv = m.evaluate(x, beta)
    ref = state.f0_ref
    assert point.compliance == pytest.approx(c, rel=1e-12)
    assert np.allclose(ev.df0 * ref, dc, rtol=1e-10, atol=0)
    assert np.allclose(ev.dg[0] * m.problem.config.settings.volume_fraction, dv, rtol=1e-10, atol=0)
    assert ev.constraints[0]["value"] == pytest.approx(v, rel=1e-12)
    m.solver.free()


def test_the_default_model_chains_rho_stiff_like_rho(tmp_path):
    """Without robust mode a response with wrt 'rho_stiff' gets the same slope as one with wrt 'rho' (bit-equal)."""
    assert "rho_stiff" in WRT
    m = tight(make_model(tmp_path))
    rng = np.random.default_rng(5)
    x = rng.uniform(0.3, 0.7, m.n_design)
    rho_tilde, _ = m.density(x, 4.0)
    g = rng.normal(size=m.n_design)
    a = m.gradient_wrt_x([g, g], ["rho", "rho_stiff"], rho_tilde, 4.0)
    assert np.array_equal(a[0], a[1])
    m.solver.free()


# --- the solvers --------------------------------------------------------------

# The solvers of the allow-list that this platform has, found at collection time. A solver that the platform
# lacks is not a skipped test: the test is not made. The reference is `scipy.sparse.linalg.spsolve` on every
# platform, so no solver is compared with itself.
# gpu-mg is in the list but is not a backend of `backend_availability`: its parity test is tests/test_gpu_mg_robust.py (G1e),
# which needs an OpenCL device.
_BACKENDS = backend_availability()
RUNNABLE_SOLVERS = [name for name in robust.ROBUST_SOLVERS if name in _BACKENDS and _BACKENDS[name]["available"]]

# Measured (Linux, PARDISO and AMG-CG at its run rtol 1e-5, 1,080 free dofs, beta 1, 4 and 8, eta_e 0.55): the
# compliance differs from the spsolve value by 2e-15 to 1.3e-13 (PARDISO) and 2e-13 to 8e-13 (AMG). The largest
# slope entry differs, relative to the largest slope, by 1e-13 to 4e-13 (PARDISO) and 6e-8 to 2.1e-7 (AMG). On
# the 2.5 mm hook the AMG nominal and eroded compliance differ by 8.7e-9 and 1.3e-8 (review of A3b). A bound is
# about ten times the larger measured value.
PARITY_BOUND = {"direct": (1e-10, 1e-10), "amg": (1e-7, 1e-5)}   # (compliance, slope relative to the largest)


def reference_by_spsolve(m, x, beta):
    """Eroded compliance and its slope with respect to x from a platform-free solve of the same matrix."""
    from scipy.sparse.linalg import spsolve
    rho_tilde, rho = m.density(x, beta)
    rho_e = m.stiffness_density(rho_tilde, rho, beta)
    U = spsolve(m.stiffness(rho_e).tocsc(), m.f_red)
    c_cases, dc_drho = m.compliance(U, rho_e)
    return float(m.weights @ c_cases), m.chain_rule(dc_drho, rho_tilde, beta)[0]


def test_the_platform_has_a_solver_to_test():
    assert "amg" in RUNNABLE_SOLVERS and set(RUNNABLE_SOLVERS) <= set(robust.ROBUST_SOLVERS)


@pytest.mark.parametrize("solver", RUNNABLE_SOLVERS)
@pytest.mark.parametrize("beta", [1.0, 8.0])
def test_every_solver_gives_the_robust_compliance_of_a_scipy_solve(tmp_path, solver, beta):
    """Each solver that the platform has runs robust mode: the compliance and the slope equal those of a
    `scipy.sparse.linalg.spsolve` of the same eroded system. AMG runs at its run tolerance (no `tight`)."""
    m = make_model(tmp_path, solver=solver, robust="eroded", eta_eroded=0.55)
    assert m.solver_name == solver
    x = np.random.default_rng(4).uniform(0.3, 0.7, m.n_design)
    c_ref, dc_ref = reference_by_spsolve(m, x, beta)
    c, dc, v, _ = m.evaluate(x, beta)
    c_bound, dc_bound = PARITY_BOUND["amg" if solver == "amg" else "direct"]
    assert abs(c - c_ref) / c_ref < c_bound
    assert np.abs(dc - dc_ref).max() / np.abs(dc_ref).max() < dc_bound
    m.solver.free()


def test_the_reference_sees_a_wrong_eroded_density(tmp_path):
    """The parity check has power: the nominal density in the reference gives a compliance far from the robust value."""
    m = make_model(tmp_path, solver="amg", robust="eroded", eta_eroded=0.55)
    x = np.random.default_rng(4).uniform(0.3, 0.7, m.n_design)
    c_ref, _ = reference_by_spsolve(m, x, 8.0)
    d = make_model(sub(tmp_path, "d"), solver="amg")
    c_nominal, *_ = d.evaluate(x, 8.0)
    assert abs(c_nominal - c_ref) / c_ref > 1e-3
    m.solver.free()
    d.solver.free()


def test_a_solver_without_a_parity_test_is_a_warning_fact_and_the_run_goes_on(tmp_path, monkeypatch):
    text = robust.solver_fact("future-solver")
    assert text and "future-solver" in text and "Options:" in text and "cholesky, pardiso, amg, gpu-mg" in text
    assert "no parity test" in text and "works by construction" in text and "topopt verify" in text
    assert robust.solver_fact("amg") is None and robust.solver_fact(None) is None
    assert robust.solver_fact("gpu-mg") is None                    # G1e: tests/test_gpu_mg_robust.py is its parity test
    from topopt.solver_choice import resolve_solver
    runs = resolve_solver("auto")                                  # the solver that really runs here
    monkeypatch.setattr(robust, "ROBUST_SOLVERS", tuple(s for s in robust.ROBUST_SOLVERS if s != runs))
    m = Model(build_problem(load_config(settings_of(tmp_path, robust="eroded"))))   # no refusal
    assert m.robust is not None
    m.solver.free()
    p = settings_of(sub(tmp_path, "r"), robust="eroded", iterations=2)
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0      # the run goes on
    log = (out / "run.log").read_text()
    assert f"warning: settings.robust is eroded, and the solver '{runs}' has no parity test for it" in log
    run_json = json.loads((out / "run.json").read_text())
    assert run_json["derived"]["robust"]["solver_tested"] is False
    assert (out / "robust.json").is_file()


def test_check_prints_the_warning_for_a_solver_without_a_parity_test(tmp_path, monkeypatch, capsys):
    from topopt.solver_choice import resolve_solver
    runs = resolve_solver("auto")
    monkeypatch.setattr(robust, "ROBUST_SOLVERS", tuple(s for s in robust.ROBUST_SOLVERS if s != runs))
    p = settings_of(tmp_path, robust="eroded")
    d = check_data(p)
    fact = [w for w in d["warnings"] if "has no parity test for it" in w]
    assert len(fact) == 1 and d["derived"]["robust"]["facts"][0] == fact[0]
    assert d["derived"]["robust"]["solver"] == runs and d["derived"]["robust"]["solver_tested"] is False
    assert main(["check", str(p)]) == 0                           # a warning, not an error
    out = capsys.readouterr().out
    assert [ln for ln in out.splitlines() if ln.startswith("warning: ") and f"the solver '{runs}' has no parity test" in ln]
    assert out.count("has no parity test for it") == 1            # printed once


def test_a_solver_with_a_parity_test_has_no_solver_fact(tmp_path):
    d = check_data(settings_of(tmp_path, robust="eroded"))
    assert not [w for w in d["warnings"] if "parity test" in w]
    assert d["derived"]["robust"]["solver_tested"] is True


# --- facts with options -------------------------------------------------------


def test_member_below_two_voxels_is_a_fact_with_options_and_the_run_goes_on(tmp_path):
    p = settings_of(tmp_path, robust="eroded", min_member_mm=3.0)      # 1.5 voxels at 2 mm
    d = check_data(p)                                                  # no exception
    fact = [w for w in d["warnings"] if w.startswith("robust mode with min_member_mm 3 mm")]
    assert len(fact) == 1 and "Options:" in fact[0] and "voxel_size 1 mm or less" in fact[0]
    assert fact[0] in d["derived"]["robust"]["facts"]
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0   # not a crash, not a refusal
    assert "warning: robust mode with min_member_mm 3 mm" in (out / "run.log").read_text()
    ok = check_data(settings_of(sub(tmp_path, "ok"), robust="eroded", min_member_mm=5.0))   # 2.5 voxels
    assert not [w for w in ok["warnings"] if w.startswith("robust mode with")]


def test_a_small_filter_radius_is_a_fact(tmp_path):
    """Robust mode with the default radius (1.5 voxels): the wall limit is below one voxel."""
    p = write_case(tmp_path, solids=[CAP], robust="eroded", beta_schedule=SCHEDULE)
    d = check_data(p)
    fact = [w for w in d["warnings"] if w.startswith("the eroded threshold 0.55 keeps a wall of")]
    assert len(fact) == 1 and "Options: set settings.min_member_mm" in fact[0]


def test_a_low_last_beta_is_a_fact(tmp_path):
    p = settings_of(tmp_path, robust="eroded", beta_schedule=[[0, 1], [10, 2]])
    rep = check_data(p)["derived"]["robust"]
    fact = [w for w in rep["facts"] if w.startswith("the highest beta of settings.beta_schedule is 2")]
    assert len(fact) == 1 and "Options:" in fact[0]
    assert rep["eroded_at_nominal_threshold"] > robust.EROSION_VISIBLE
    ok = check_data(settings_of(sub(tmp_path, "ok"), robust="eroded"))["derived"]["robust"]
    assert not [w for w in ok["facts"] if w.startswith("the highest beta")] and ok["eroded_at_nominal_threshold"] < 0.4


def rule_case(tmp_path, **settings):
    """A case without `filter_radius`: the radius comes from `min_member_mm` (the A3a rule) or the default."""
    return write_case(tmp_path, solids=[CAP], beta_schedule=SCHEDULE, **settings)


def test_the_radius_rule_is_a_fact_in_robust_mode(tmp_path):
    """`min_member_mm` with robust mode: the A3a radius rule is for eta 0.5; the fact gives the wall limit and the option."""
    p = rule_case(tmp_path, robust="eroded", min_member_mm=5.0)
    rep = check_data(p)["derived"]["robust"]
    fact = [w for w in rep["facts"] if w.startswith("the radius rule of min_member_mm")]
    assert len(fact) == 1
    # R = 1.877 * 5 = 9.385 mm; the wall limit of eta_e 0.55 is 0.5948 R = 5.58 mm; the option is 5 / 0.5948 = 8.41 mm
    assert "filter_radius = 1.877 x min_member_mm" in fact[0] and "threshold 0.5." in fact[0]
    assert "walls of about 5.58 mm or more" in fact[0] and "request 5 mm" in fact[0]
    assert "settings.filter_radius to 1.681 x min_member_mm = 8.41 mm" in fact[0] and "Option:" in fact[0]
    assert rep["radius_for_request_mm"] == pytest.approx(5.0 / robust.plate_limit_per_radius(0.55), abs=1e-3)
    assert fact[0] in check_data(p)["warnings"]
    # no fact: robust off, an explicit radius, or no request
    off = check_data(rule_case(sub(tmp_path, "o"), min_member_mm=5.0))
    assert not [w for w in off["warnings"] if w.startswith("the radius rule")]
    explicit = check_data(rule_case(sub(tmp_path, "e"), robust="eroded", min_member_mm=5.0, filter_radius=8.0))
    assert not [w for w in explicit["warnings"] if w.startswith("the radius rule")]
    none = check_data(rule_case(sub(tmp_path, "n"), robust="eroded"))
    assert not [w for w in none["warnings"] if w.startswith("the radius rule")]
    # the option follows eta_e: 0.6 gives 5 / 0.66 mm
    r6 = check_data(rule_case(sub(tmp_path, "s"), robust="eroded", eta_eroded=0.6, min_member_mm=5.0))["derived"]["robust"]
    assert r6["radius_for_request_mm"] == pytest.approx(5.0 / robust.plate_limit_per_radius(0.6), abs=1e-3)


def test_eta_without_robust_is_a_fact(tmp_path):
    d = check_data(settings_of(tmp_path, eta_eroded=0.6))
    assert [w for w in d["derived"]["robust"]["facts"] if "is set, and settings.robust is off" in w]
    assert d["derived"]["robust"]["on"] is False


def test_objective_and_rows_of_the_base_run_in_robust_mode(tmp_path):
    """Every objective and constraint that the YAML can make runs: worst case, and a compliance limit per case."""
    p = settings_of(tmp_path, robust="eroded", case_combine="max", iterations=3)
    d = yaml.safe_load(p.read_text())
    d["load_cases"][0]["compliance_limit"] = 1.0
    p.write_text(yaml.safe_dump(d))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run_json = json.loads((out / "run.json").read_text())
    assert run_json["objective"] == "worst_case_compliance"
    assert [c["name"] for c in run_json["constraints"]] == ["volume", "compliance:down"]
    assert json.loads((out / "result" / "metrics.json").read_text())["robust"]["on"] is True


def test_report_numbers_agree_with_the_continuum_values():
    assert robust.plate_limit_per_radius(0.5) == pytest.approx(0.5328, abs=5e-5)    # the rule of members.py
    assert robust.plate_limit_per_radius(0.55) * 3.75 == pytest.approx(2.23, abs=0.005)   # pipeline README: 2.23 mm at R 3.75
    assert robust.plate_limit_per_radius(0.6) > robust.plate_limit_per_radius(0.55) > robust.plate_limit_per_radius(0.5)
    tool = load_tool("calibrate_eta_voxel")
    assert tool.rod_limit_per_radius(0.5) * 3.75 == pytest.approx(3.51, abs=0.01)     # pipeline README: 3.51 mm
    assert tool.rod_limit_per_radius(0.55) * 3.75 == pytest.approx(3.744, abs=0.01)
    assert tool.flat_face_depth_per_radius(0.55) * 3.75 == pytest.approx(0.187, abs=0.002)   # pipeline README: 0.187 mm


# --- the voxel fixture (mesh-free) -------------------------------------------


def test_voxel_fixture_follows_the_continuum_for_a_wide_radius():
    tool = load_tool("calibrate_eta_voxel")
    R = 6.0
    for n in (4, 8):                       # a wall of n voxels: the voxel value is near the continuum value
        u = n / (2 * R)
        assert tool.wall_centre(n, R) == pytest.approx(2 * u - 2 * u ** 3 + u ** 4, abs=0.03)
    assert tool.rod_centre(8, R) == pytest.approx(tool.rod_value(4.0 / R), abs=0.05)
    n_survive, value = tool.smallest(tool.wall_centre, 3.75, 0.55)
    assert n_survive == 3 and value >= 0.55
    committed = json.loads((Path(__file__).resolve().parents[1] / "docs" / "a3b-eta-voxel-fixture.json").read_text())
    row = next(r for r in committed["rows"] if r["radius_voxels"] == 3.75 and r["eta"] == 0.55)
    assert (row["wall_survive_voxels"], row["rod_survive_voxels"]) == (n_survive, tool.smallest(tool.rod_centre, 3.75, 0.55)[0])


# --- a robust run, end to end -------------------------------------------------


def run_case(tmp_path, name="r", **settings):
    p = settings_of(sub(tmp_path, name + "_in"), **settings)
    out = tmp_path / name
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    return p, out


def test_robust_run_writes_the_facts_to_the_log_robust_json_and_metrics(tmp_path):
    p, out = run_case(tmp_path, robust="eroded", eta_eroded=0.6, iterations=6)
    log = (out / "run.log").read_text()
    assert "robust: eroded form, eta_n 0.5, eta_e 0.6 (settings.eta_eroded)" in log
    assert "compliance nominal" in log and "in the robust part" in log
    summary = [ln for ln in log.splitlines() if ln.startswith("robust: summary:")]
    assert len(summary) == 1 and log.index("done in") < log.index(summary[0])
    rj = json.loads((out / "robust.json").read_text())
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    r = metrics["robust"]
    assert (r["on"], r["mode"], r["eta_n"], r["eta_e"], r["eta_e_source"]) == (True, "eroded", 0.5, 0.6, "settings")
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    # M1: the final values are the nominal design (the part that you print); the eroded ones are in the block
    assert metrics["compliance_final"] == rj["compliance_nominal"] == r["compliance_nominal"]
    assert r["compliance_eroded"] == pytest.approx(hist[-1]["compliance"], rel=1e-12)   # the optimizer's compliance
    assert r["compliance_eroded"] == rj["compliance_eroded"] > metrics["compliance_final"]
    assert metrics["compliance_initial"] == hist[0]["compliance"]          # the eroded design at pass 0 (the note says so)
    assert "compliance_final and stiffness_cases are the nominal design" in r["note"]
    assert r["compliance_cases_nominal"] == rj["compliance_cases_nominal"]
    assert r["compliance_nominal"] < r["compliance_eroded"]
    assert r["robust_s"] > 0 and r["robust_parts_s"]["nominal_solve"] > 0 and r["run_time_s"] == metrics["run_time_s"]
    assert r["robust_s"] == rj["robust_s"] and r["complete"] is True
    assert metrics["run_time_s"] >= rj["run_s"] >= rj["robust_s"]     # the total includes the nominal solve and the post-process
    assert [c["name"] for c in r["stiffness_cases_eroded"]] == ["down", "side"] == [c["name"] for c in metrics["stiffness_cases"]]
    assert "stiffness_cases_nominal" not in r
    assert all(n["N_per_mm"] > e["N_per_mm"] for n, e in zip(metrics["stiffness_cases"], r["stiffness_cases_eroded"]))
    run_cases = json.loads((out / "run.json").read_text())["load_cases"]
    assert metrics["stiffness_cases"] == case_stiffness(run_cases, rj["compliance_cases_nominal"])     # one formula
    assert r["stiffness_cases_eroded"] == case_stiffness(run_cases, hist[-1]["compliance_cases"])
    run_json = json.loads((out / "run.json").read_text())
    assert run_json["derived"]["robust"]["eta_e"] == 0.6 and run_json["derived"]["robust"]["on"] is True
    keys = list(metrics)
    assert keys[-1] == "verdict" and keys.index("robust") > keys.index("triangles")   # robust (A3b) is added after the old keys; the verdict (D1b) is the last key
    assert f"eta_n 0.5, eta_e 0.6; compliance nominal {rj['compliance_nominal']:.6g}, eroded {rj['compliance_eroded']:.6g}" in summary[0]
    assert f"total run time {metrics['run_time_s']:g} s" in summary[0] and f"{rj['robust_s']:.2f} s in the robust part" in summary[0]
    assert "history curve the eroded design" in summary[0] and "compliance_final is the nominal design" in summary[0]
    assert "The curve above (history) is the eroded design" in log and "are the nominal design" in log


def test_robust_run_snapshots_and_volume_are_the_nominal_design(tmp_path):
    p, out = run_case(tmp_path, robust="eroded", iterations=6)
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    run_json = json.loads((out / "run.json").read_text())
    labels = np.frombuffer((out / "labels.bin").read_bytes(), np.uint8)
    snap = np.frombuffer((out / "snapshots" / "it_0006.bin").read_bytes(), np.uint8)
    design = labels == 1
    assert (snap[design] / 255.0).mean() == pytest.approx(hist[-1]["volume_fraction"], abs=1 / 255)
    assert hist[-1]["constraints"][0]["value"] == hist[-1]["volume_fraction"]     # the volume row is the nominal volume


def test_the_nominal_compliance_is_the_compliance_of_the_final_design_in_a_default_model(tmp_path):
    """`compliance_nominal` (one solve after the last pass) equals `evaluate(x, beta_last)` of a default model."""
    p = settings_of(tmp_path, robust="eroded", iterations=6)
    out = RunWriter(tmp_path / "r", time.time(), quiet=True)
    out.clean()
    x = run(build_problem(load_config(p)), out)
    rj = json.loads((tmp_path / "r" / "robust.json").read_text())
    d = make_model(sub(tmp_path, "d"))                       # the same case, robust off
    beta_last = beta_at(d.problem.config.settings.beta_schedule, 6)
    c_nominal, _dc, _v, _dv = d.evaluate(x, beta_last)
    bound = 5e-8 if d.solver_name == "amg" else 1e-10        # AMG at its run tolerance: 8.7e-9 measured in the review
    assert rj["compliance_nominal"] == pytest.approx(c_nominal, rel=bound, abs=0)
    assert rj["compliance_eroded"] > rj["compliance_nominal"] * (1 + 1e-4)
    m = make_model(sub(tmp_path, "m"), robust="eroded", iterations=6)
    assert m.evaluate(x, beta_last)[0] == pytest.approx(rj["compliance_eroded"], rel=max(bound, 1e-9), abs=0)
    d.solver.free()
    m.solver.free()


def test_the_eroded_projection_runs_once_for_each_pass(tmp_path, monkeypatch):
    """One eroded projection and one eroded slope for each pass (the point gets the array from the loop)."""
    calls = {"eroded": 0, "slope": 0}
    eroded, slope = robust.Robust.eroded, robust.Robust.eroded_slope

    def count_eroded(self, rho_tilde, beta):
        calls["eroded"] += 1
        return eroded(self, rho_tilde, beta)

    def count_slope(self, rho_tilde, beta):
        calls["slope"] += 1
        return slope(self, rho_tilde, beta)

    monkeypatch.setattr(robust.Robust, "eroded", count_eroded)
    monkeypatch.setattr(robust.Robust, "eroded_slope", count_slope)
    m = make_model(tmp_path, robust="eroded")
    minimize(m, formulation_from_config(m.problem.config, m.n_design), iterations=3)
    assert calls == {"eroded": 4, "slope": 4}                      # 4 passes
    m.solver.free()


def test_a_run_that_ends_before_the_nominal_solve_keeps_the_eroded_final_values(tmp_path):
    """No `robust.json`: `post.finish` has only the eroded values, and the block says so."""
    from topopt.post import finish
    p, out = run_case(tmp_path, robust="eroded", iterations=3)
    (out / "robust.json").unlink()
    metrics = finish(out)
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert metrics["compliance_final"] == hist[-1]["compliance"]
    assert metrics["robust"]["complete"] is False and "eroded values of the last history line" in metrics["robust"]["note"]


def test_minimize_has_no_nominal_compliance_in_robust_mode(tmp_path):
    """`minimize` has no `finish_run`: a script gets the eroded history and no `robust.json` (CONTRACTS 22b)."""
    m = make_model(tmp_path, robust="eroded")
    res = minimize(m, formulation_from_config(m.problem.config, m.n_design), iterations=2)
    assert not any("nominal" in k for rec in res.history for k in rec)
    m.solver.free()


def test_default_run_has_no_robust_json_and_a_robust_run_does_not_leak_into_it(tmp_path):
    out = tmp_path / "r"
    p_on = settings_of(sub(tmp_path, "on"), robust="eroded", iterations=2)
    assert main(["run", str(p_on), "--out", str(out), "--quiet"]) == 0
    assert (out / "robust.json").is_file()
    p_off = settings_of(sub(tmp_path, "off"), iterations=2)
    assert main(["run", str(p_off), "--out", str(out), "--quiet"]) == 0   # same folder: the old file must go
    assert not (out / "robust.json").exists()
    assert json.loads((out / "result" / "metrics.json").read_text())["robust"] == {"on": False}
    assert "robust:" not in (out / "run.log").read_text()


def test_minimize_in_robust_mode_reports_the_eroded_compliance_and_the_nominal_density(tmp_path):
    m = make_model(tmp_path, robust="eroded")
    form = Formulation(weighted_compliance(), [volume_constraint(m.n_design, 0.3)])
    seen = []
    res = minimize(m, form, iterations=3, on_iteration=lambda rec, p: seen.append((rec, p)))
    rec, p = seen[-1]
    assert isinstance(p, DesignPoint) and p.rho_stiff is not p.rho
    assert rec["volume_fraction"] == pytest.approx(float(res.rho.mean()))          # the nominal volume
    assert rec["compliance"] == pytest.approx(p.compliance)
    m.solver.free()


# --- the estimate and the printout --------------------------------------------


def test_estimate_includes_the_nominal_solve_at_the_end():
    per0, t0 = estimate_seconds(100_000, 80, "pardiso")
    per1, t1 = estimate_seconds(100_000, 80, "pardiso", robust_mode=True)
    assert per0 == per1
    assert t1 - t0 == pytest.approx(per0 * (81 * robust.time_factor(80) - 81))
    assert robust.time_factor(80) == pytest.approx((81 * robust.PASS_TIME_FACTOR + robust.EXTRA_SOLVES) / 81)


def test_check_prints_the_robust_lines_and_the_estimate_has_the_factor(tmp_path, capsys):
    p = settings_of(tmp_path, robust="eroded", min_member_mm=5.0)
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("robust:"))
    assert "eroded form, eta_n 0.5, eta_e 0.55 (default)" in line and "one solve for each pass" in line
    d = check_data(p)
    assert d["derived"]["robust"]["time_factor"] == round(robust.time_factor(d["iterations"]), 4)
    base = check_data(settings_of(sub(tmp_path, "b"), min_member_mm=5.0))
    assert d["time_estimate"]["total_s"] >= base["time_estimate"]["total_s"]
    assert base["derived"]["robust"] == {"mode": "off", "on": False, "facts": []}


def test_old_run_without_robust_gives_the_off_block():
    assert robust.metrics_block({"derived": {}}, None, 12.0) == {"on": False}
    assert robust.metrics_block({}, None, 12.0) == {"on": False}
    run = {"derived": {"robust": {"on": True, "mode": "eroded", "eta_n": 0.5, "eta_e": 0.55, "eta_e_source": "default"}}}
    assert robust.metrics_block(run, None, 12.0)["complete"] is False       # the run ended before the summary


# --- the standalone check script ----------------------------------------------


def test_robust_check_script_prints_the_volumes_the_compliance_and_the_fd_check(tmp_path, capsys):
    script = load_tool("robust_check")
    p = settings_of(tmp_path, robust="eroded", eta_eroded=0.6)
    assert script.main([str(p), "--beta", "8", "--entry", "5"]) == 0
    out = capsys.readouterr().out
    assert "settings.robust is eroded: eta_n 0.5, eta_e 0.6 (settings)" in out
    nominal, eroded = [float(v) for v in out.split("mean density nominal ")[1].split(" (")[0].replace("eroded", "").split(",")]
    assert eroded < nominal
    assert "compliance of the nominal design" in out and "best-step relative error" in out and out.rstrip().endswith("ok")
    off = settings_of(sub(tmp_path, "o"))
    assert script.main([str(off)]) == 0
    assert "settings.robust is off" in capsys.readouterr().out
    assert script.main([str(tmp_path / "missing.yaml")]) == 1
