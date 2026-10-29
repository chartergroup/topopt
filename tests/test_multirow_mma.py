"""Tests for the multi-row MMA API (A1): rows, gradients, equivalence, limits, YAML keys.

Tolerances (each one is stated at the test that uses it):
- FD gradients: central difference, relative to the largest gradient of the row.
- Equivalence of m=1 and m=2 runs: see EQUIV_RTOL.
- Feasibility: see FEAS_TOL.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_optimize import make_case
from topopt.config import UserError, load_config
from topopt.formulation import (Constraint, Formulation, Objective, Response, compliance_constraint,
                                volume_constraint, weighted_compliance, worst_case_compliance)
from topopt.mma import Mma, MmaState
from topopt.optimize import Model, formulation_from_config, minimize, run
from topopt.problem import build_problem
from topopt.snapshot import RunWriter

# m=1 against m=2 on a volume-only problem, 40 iterations (measured: 7e-12 / 8e-9 for a linear
# inactive row, 9e-7 / 3.8e-3 for the log row, 1.7e-7 / 3.8e-3 for a copy of the volume row):
EQUIV_INACTIVE = dict(c_rtol=1e-9, rho_max=1e-6)   # second row inactive and linear: compliance rel., density max abs.
EQUIV_INACTIVE_LOG = dict(c_rtol=1e-5, rho_max=1e-2, rho_mean=1e-3)   # inactive log row (gradient ~1/c: larger)
EQUIV_DUPLICATE = dict(c_rtol=1e-6, rho_max=1e-2, rho_mean=1e-3)   # exact copy of the volume row
FEAS_TOL = 5e-3       # g = c/limit - 1 at the end must be at or below this (0.5% of the limit)
FD_BETAS = (1.0, 2.0, 8.0)


def case_yaml(tmp_path, iterations=30, limits=None, case_combine=None, h=1.0):
    """The cantilever of test_optimize with optional compliance limits and objective."""
    p = make_case(tmp_path, h=h, iterations=iterations)
    d = yaml.safe_load(p.read_text())
    for name, lim in (limits or {}).items():
        next(c for c in d["load_cases"] if c["name"] == name)["compliance_limit"] = lim
    if case_combine:
        d["settings"]["case_combine"] = case_combine
    p.write_text(yaml.safe_dump(d))
    return p


def make_model(tmp_path, **kw):
    m = Model(build_problem(load_config(case_yaml(tmp_path, **kw))))
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13      # AMG is not exact at the run tolerance: tighten it for FD tests
    return m


def make_model_in(folder, **kw):
    folder.mkdir(exist_ok=True)
    return make_model(folder, **kw)


def fd_settings(m):
    """Step and tolerance of the FD tests. A direct solver is exact; AMG-CG is not."""
    return (1e-4, 1e-4) if m.solver_name == "amg" else (1e-6, 1e-5)


def evaluate(m, form, x, beta, ref=1.0):
    m.solver._x_prev = None             # AMG: cold start (KB pitfall amg-warm-start-stale-solution)
    rho_tilde, rho = m.density(x, beta)
    point = m.design_point(x, beta, rho_tilde, rho, m.solve(m.stiffness(rho)))
    state = MmaState(m.n_design)
    state.f0_ref = ref                  # a fixed scale: FD needs the same function at each point
    return form.evaluate(point, state)


def check_fd(m, form, beta, rows_names, samples=10, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 0.7, m.n_design)
    eps, tol = fd_settings(m)
    ev = evaluate(m, form, x, beta)
    assert ev.g.shape[0] == len(rows_names)
    worst = 0.0
    for e in rng.choice(m.n_design, size=min(samples, m.n_design), replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        evp, evm = evaluate(m, form, xp, beta), evaluate(m, form, xm, beta)
        fd_g = (evp.g - evm.g) / (2 * eps)
        fd_f0 = (evp.f0 - evm.f0) / (2 * eps)
        for r, name in enumerate(rows_names):
            err = abs(fd_g[r] - ev.dg[r, e]) / max(abs(ev.dg[r, e]), 1e-3 * np.abs(ev.dg[r]).max())
            worst = max(worst, err)
            assert err < tol, f"row {name} voxel {e} beta {beta}: fd {fd_g[r]:.6g} vs {ev.dg[r, e]:.6g}"
        err = abs(fd_f0 - ev.df0[e]) / max(abs(ev.df0[e]), 1e-3 * np.abs(ev.df0).max(), 1e-300)
        worst = max(worst, err)
        assert err < tol, f"objective voxel {e} beta {beta}: fd {fd_f0:.6g} vs {ev.df0[e]:.6g}"
    return worst


# --- FD tests of the built-in rows ------------------------------------------


@pytest.mark.parametrize("beta", FD_BETAS)
def test_fd_weighted_objective_volume_and_case_limits(tmp_path, beta):
    """Rows volume and compliance:<case> and the weighted-sum objective. Limit: see fd_settings."""
    m = make_model(tmp_path, h=2.0)
    form = Formulation(weighted_compliance(), [
        volume_constraint(m.n_design, 0.3),
        compliance_constraint("down", 0.5, "compliance:down"),
        compliance_constraint(1, 0.7, "compliance:side")])
    check_fd(m, form, beta, ["volume", "compliance:down", "compliance:side"])
    m.solver.free()


@pytest.mark.parametrize("beta", FD_BETAS)
def test_fd_worst_case_objective_rows(tmp_path, beta):
    """The bound form: rows w_i c_i / ref for each case, then the volume row."""
    m = make_model(tmp_path, h=2.0)
    form = Formulation(worst_case_compliance(2), [volume_constraint(m.n_design, 0.3)])
    assert form.n_rows == 3
    check_fd(m, form, beta, ["bound:down", "bound:side", "volume"])
    m.solver.free()


def test_fd_gradient_wrt_x_and_rho_tilde(tmp_path):
    """The same function given with wrt='rho', 'rho_tilde' and 'x' gives the same row."""
    m = make_model(tmp_path, h=2.0)
    beta = 2.0
    rng = np.random.default_rng(5)
    x = rng.uniform(0.3, 0.7, m.n_design)
    w = rng.uniform(0.5, 1.5, m.n_design)
    rho_tilde, rho = m.density(x, beta)
    # a function of rho: sum w * rho. Its slopes with respect to rho_tilde and x follow by chain rule.
    from topopt.filter import heaviside_deriv
    from topopt.optimize import ETA
    dproj = heaviside_deriv(rho_tilde, beta, ETA)
    got = m.gradient_wrt_x([w, w * dproj, m.FT @ (w * dproj)], ["rho", "rho_tilde", "x"], rho_tilde, beta)
    ref = m.FT @ (w * dproj)
    for row in got:
        np.testing.assert_allclose(row, ref, rtol=1e-12)
    m.solver.free()


def test_extra_adjoint_load_point_displacement(tmp_path):
    """A response that is not a compliance (a displacement at one dof) works through the API.

    Adjoint: K lam = e_p, d u_p/d rho = -lam^T dK/drho u. This is what A5 needs.
    """
    m = make_model(tmp_path, h=2.0)
    dof = int(m.n_free - 2)    # some free dof
    rhs = np.zeros(m.n_free)
    rhs[dof] = 1.0

    def u_p(point):
        lam = point.solve_adjoint(rhs)
        return Response(float(point.U[dof, 0]), -point.stiffness_derivative(lam, point.U[:, 0]))

    form = Formulation(weighted_compliance(), [Constraint("u_point", u_p, limit=-1e-3, scale=1e-3)])
    check_fd(m, form, 2.0, ["u_point"])
    m.solver.free()


def test_adjoint_solve_keeps_amg_warm_start(tmp_path):
    """`warm=False` does not change the warm start of the main solve."""
    p = case_yaml(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "amg"
    p.write_text(yaml.safe_dump(d))
    m = Model(build_problem(load_config(p)))
    rho = np.full(m.n_design, 0.5)
    U = m.solve(m.stiffness(rho))
    prev = m.solver._x_prev.copy()
    m.solver.solve(np.ones(m.n_free), warm=False)
    np.testing.assert_array_equal(m.solver._x_prev, prev)
    np.testing.assert_allclose(m.solver.solve(m.f_red), U, rtol=1e-4, atol=1e-12)
    m.solver.free()


# --- Mma object --------------------------------------------------------------


def test_mma_state_is_one_object_and_row_check():
    mma = Mma(10, 3, 0.2)
    # "rows" (the state of the stage-aware rows) and "move" (the move limit of the step acceptance) are the A2 additions:
    # a resume from the MmaState needs them (KB pitfalls/resume-differs-from-straight-run). Both are empty/None by default.
    assert set(vars(mma.state)) == {"n", "it", "x", "xold1", "xold2", "low", "upp", "f0_ref", "rows", "move"}
    assert mma.state.rows == {} and mma.state.move is None
    with pytest.raises(ValueError, match="3 rows"):
        mma.step(np.full(10, 0.5), 1.0, np.zeros(10), np.zeros(2), np.zeros((2, 10)))
    with pytest.raises(ValueError, match="not finite"):
        mma.step(np.full(10, 0.5), 1.0, np.zeros(10), np.array([np.nan, 0, 0]), np.zeros((3, 10)))


def test_formulation_errors_are_clear():
    with pytest.raises(ValueError, match="one constraint row or more"):
        Formulation(weighted_compliance(), [])
    c = volume_constraint(5, 0.3)
    with pytest.raises(ValueError, match="unique"):
        Formulation(weighted_compliance(), [c, c])
    with pytest.raises(ValueError, match="scale"):
        Constraint("x", lambda p: (0.0, np.zeros(1)), limit=1.0, scale=-1.0)


def test_bad_gradient_size_names_the_constraint(tmp_path):
    m = make_model(tmp_path, h=2.0)
    bad = Constraint("my_limit", lambda p: Response(1.0, np.zeros(3)))
    with pytest.raises(ValueError, match="my_limit.*needs %d values" % m.n_design):
        evaluate(m, Formulation(weighted_compliance(), [bad]), np.full(m.n_design, 0.5), 1.0)
    m.solver.free()


# --- equivalence and feasibility ---------------------------------------------


def final(result):
    return result.history[-1]["compliance"], result.rho


def test_equivalence_second_row_does_not_change_a_volume_only_run(tmp_path):
    """m=1 against m=2 on a volume-only problem.

    (a) The second row is a compliance limit far above the result (inactive): the run is the
    same to EQUIV_INACTIVE (linear row) or EQUIV_INACTIVE_LOG (the built-in log row; its gradient is
    about 1/c, so the interior-point residual is larger). (b) The second row is a copy of the volume row. The dual system is
    nearly singular then, and the step differs by about 1e-9 in the first iteration;
    this grows over 40 iterations in a few voxels. Limits: EQUIV_DUPLICATE.
    """
    m = make_model(tmp_path, iterations=40)
    n, vf = m.n_design, 0.3
    one = minimize(m, Formulation(weighted_compliance(), [volume_constraint(n, vf)]))
    c1, rho1 = final(one)
    dup = Constraint("volume_copy", volume_constraint(n, vf).evaluate, limit=vf)
    big = 100 * one.history[0]["compliance_cases"][0]
    loose_log = compliance_constraint("down", big)
    loose_lin = Constraint("loose_linear", loose_log.evaluate, limit=big)    # linear, price 1000
    for second, tol in ((loose_lin, EQUIV_INACTIVE), (loose_log, EQUIV_INACTIVE_LOG),
                        (dup, EQUIV_DUPLICATE)):
        two = minimize(m, Formulation(weighted_compliance(), [volume_constraint(n, vf), second]))
        c2, rho2 = final(two)
        d = np.abs(rho2 - rho1)
        assert abs(c2 - c1) / c1 < tol["c_rtol"], (second.name, c1, c2)
        assert d.max() < tol["rho_max"], second.name
        assert d.mean() < tol.get("rho_mean", 1e-6), second.name
    m.solver.free()


def test_mma_state_alone_resumes_the_run():
    """A copy of `mma.state` in a fresh Mma gives the same next step (full-state checkpoint)."""
    import copy
    n = 30
    rng = np.random.default_rng(2)
    a = Mma(n, 2, 0.2)
    x = np.full(n, 0.4)
    args = [(float(rng.uniform(0.5, 1.5)), rng.normal(size=n), rng.normal(size=2),
             rng.normal(size=(2, n))) for _ in range(6)]
    for f0, df0, g, dg in args[:4]:
        x = a.step(x, f0, df0, g, dg)
    b = Mma(n, 2, 0.2)
    b.state = copy.deepcopy(a.state)
    np.testing.assert_array_equal(b.state.x, x)
    for f0, df0, g, dg in args[4:]:
        xa, xb = a.step(x, f0, df0, g, dg), b.step(x, f0, df0, g, dg)
        np.testing.assert_array_equal(xa, xb)
        x = xa


def test_default_run_matches_the_value_before_a1(tmp_path):
    """The 30-iteration cantilever of test_optimize gave these values at 41a2f39 (before A1).

    At that commit the scale of the objective was the one of pass 0 for the whole run: `objective_scale="first"` since P3a (the default is `stage`,
    the next test). The numbers are the same to 1e-14, so the pin still shows that the volume-only formulation is the code before A1.
    """
    m = make_model(tmp_path, iterations=30)
    if m.solver_name == "amg":
        pytest.skip("the values come from a direct solver; AMG-CG differs at its tolerance")
    res = minimize(m, formulation_from_config(m.problem.config, m.n_design), objective_scale="first")
    assert res.history[-1]["compliance"] == pytest.approx(0.06852387239868557, rel=1e-9)
    assert res.history[-1]["volume_fraction"] == pytest.approx(0.2999785521411673, rel=1e-9)
    assert res.history[10]["compliance"] == pytest.approx(0.10358490350897777, rel=1e-9)
    m.solver.free()


def test_default_run_with_the_objective_scale_of_each_stage(tmp_path):
    """The same cantilever with the default `objective_scale: stage` (P3a): the scale is taken again at the passes 10 and 20 (the schedule of `make_case`).

    The passes up to the first stage start are those of `first` (pass 10 has the same design: its step is the first one that uses the new scale), and the
    later passes differ. Values measured on Linux with PARDISO (1 or 2 threads); a Cholesky run on macOS must agree to the same 1e-9.
    """
    m = make_model(tmp_path, iterations=30)
    if m.solver_name == "amg":
        pytest.skip("the values come from a direct solver; AMG-CG differs at its tolerance")
    res = minimize(m, formulation_from_config(m.problem.config, m.n_design))
    h = res.history
    assert h[10]["compliance"] == pytest.approx(0.10358490350897777, rel=1e-9)        # the design of pass 10: the step of pass 9 had the old scale
    assert h[11]["compliance"] == pytest.approx(0.09573614269330322, rel=1e-9)        # `first`: 0.09587166855971771
    assert h[21]["compliance"] == pytest.approx(0.06977050064693344, rel=1e-9)        # `first`: 0.06992241298995622
    assert h[-1]["compliance"] == pytest.approx(0.0684471495204924, rel=1e-9)         # `first`: 0.06852387239868557
    assert h[-1]["volume_fraction"] == pytest.approx(0.2999765707650664, rel=1e-9)    # `first`: 0.2999785521411673
    m.solver.free()


def test_default_formulation_is_the_volume_only_problem(tmp_path):
    m = make_model(tmp_path)
    form = formulation_from_config(m.problem.config, m.n_design)
    assert form.n_rows == 1 and form.objective.form == "sum"
    assert [c.name for c in form.constraints] == ["volume"]
    m.solver.free()


def test_feasible_case_limit_is_met(tmp_path):
    """A limit 15% below the unconstrained compliance of a case is met: g <= FEAS_TOL."""
    base = make_model_in(tmp_path / "a", iterations=40)
    free = minimize(base, formulation_from_config(base.problem.config, base.n_design))
    c_side = free.history[-1]["compliance_cases"][1]
    base.solver.free()
    limit = 0.85 * c_side
    m = make_model_in(tmp_path / "b", iterations=40, limits={"side": limit})
    res = minimize(m, formulation_from_config(m.problem.config, m.n_design))
    by_name = {c["name"]: c for c in res.constraints}
    assert by_name["compliance:side"]["limit"] == pytest.approx(limit)
    assert by_name["compliance:side"]["g"] <= FEAS_TOL
    assert by_name["volume"]["g"] <= FEAS_TOL
    assert res.history[-1]["compliance_cases"][1] <= limit * (1 + FEAS_TOL)
    assert not res.unmet
    m.solver.free()


@pytest.mark.parametrize("fraction", [0.01, 0.3, 0.6])
def test_impossible_limit_is_a_fact_and_the_volume_stays(tmp_path, fraction):
    """A limit that cannot be met (fractions of the free compliance of the case): no exception.

    The volume is the budget: its g stays at or below FEAS_TOL and only the limit row is
    reported unmet, with the cause and the options in the run log.
    """
    m = make_model_in(tmp_path / "a", iterations=40)
    c_side = minimize(m, formulation_from_config(m.problem.config, m.n_design)
                      ).history[-1]["compliance_cases"][1]
    m.solver.free()
    (tmp_path / "b").mkdir()
    prob = build_problem(load_config(case_yaml(tmp_path / "b", iterations=40,
                                               limits={"side": fraction * c_side})))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    run(prob, out)
    hist = [json.loads(l) for l in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    rows = {c["name"]: c for c in hist[-1]["constraints"]}
    assert rows["volume"]["g"] <= FEAS_TOL
    assert rows["compliance:side"]["violation"] > FEAS_TOL
    assert rows["compliance:side"]["g"] == pytest.approx(rows["compliance:side"]["violation"])
    log = (tmp_path / "run" / "run.log").read_text()
    assert "constraint not met: compliance:side" in log
    assert "constraint not met: volume" not in log
    assert "raise settings.volume_fraction, relax the limit" in log
    assert json.loads((tmp_path / "run" / "status.json").read_text())["step"] == "snapshot"


def test_worst_case_objective_lowers_the_largest_weighted_case(tmp_path):
    """minimize max_i w_i c_i gives a smaller max than minimize sum w_i c_i, and z bounds the terms."""
    m = make_model(tmp_path, iterations=40)
    n = m.n_design
    s = minimize(m, Formulation(weighted_compliance(), [volume_constraint(n, 0.3)]))
    w = m.weights
    max_sum = float(np.max(w * s.history[-1]["compliance_cases"]))
    t = minimize(m, Formulation(worst_case_compliance(2), [volume_constraint(n, 0.3)]))
    max_worst = float(np.max(w * t.history[-1]["compliance_cases"]))
    assert max_worst < max_sum
    assert t.history[-1]["objective"]["form"] == "max"
    assert t.history[-1]["objective"]["value"] == pytest.approx(max_worst)
    ref = t.mma.state.f0_ref
    assert t.mma.z == pytest.approx(max_worst / ref, rel=1e-2)    # z is the bound of the terms
    m.solver.free()


# --- YAML keys and the run folder ---------------------------------------------


def test_yaml_keys_default_and_values(tmp_path):
    cfg = load_config(case_yaml(tmp_path))
    assert cfg.settings.case_combine == "sum"
    assert all(c.compliance_limit is None for c in cfg.load_cases)
    cfg = load_config(case_yaml(tmp_path, limits={"side": 4.5}, case_combine="max"))
    assert cfg.settings.case_combine == "max"
    assert [c.compliance_limit for c in cfg.load_cases] == [None, 4.5]


@pytest.mark.parametrize("limit, message", [
    (0, "load_cases\\[1\\].compliance_limit' must be > 0"),
    (-2.0, "must be > 0"),
    ("big", "must be a number"),
    (None, "must be a number"),
])
def test_yaml_compliance_limit_errors(tmp_path, limit, message):
    p = case_yaml(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["load_cases"][1]["compliance_limit"] = limit
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match=message):
        load_config(p)


def test_yaml_objective_error(tmp_path):
    p = case_yaml(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["settings"]["case_combine"] = "minmax"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="settings.case_combine' must be sum or max"):
        load_config(p)


def test_run_writes_constraint_fields_and_keeps_old_ones(tmp_path):
    prob = build_problem(load_config(case_yaml(tmp_path, iterations=10, limits={"down": 50.0},
                                               case_combine="max")))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    run(prob, out)
    hist = [json.loads(l) for l in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    old = {"iteration", "elapsed_s", "compliance", "compliance_cases", "volume_fraction", "beta",
           "change", "grayness", "t_iter_s"}
    assert old <= set(hist[0]) and set(hist[0]) - old == {"objective", "constraints"}
    assert [c["name"] for c in hist[0]["constraints"]] == ["volume", "compliance:down"]
    assert set(hist[0]["constraints"][0]) == {"name", "value", "limit", "g", "violation"}
    assert hist[0]["objective"]["form"] == "max"
    run_json = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run_json["objective"] == "worst_case_compliance" and run_json["objective_form"] == "max"
    assert run_json["constraints"] == [{"name": "volume", "limit": 0.3},
                                       {"name": "compliance:down", "limit": 50.0}]
    # "compliance" stays the weighted sum, also with the worst-case objective
    w = [3.0, 1.0]
    assert hist[0]["compliance"] == pytest.approx(sum(a * b for a, b in zip(w, hist[0]["compliance_cases"])))


# --- the example script ---------------------------------------------------------


def test_example_custom_constraint_runs_end_to_end():
    path = Path(__file__).resolve().parent.parent / "examples" / "custom_constraint.py"
    spec = importlib.util.spec_from_file_location("custom_constraint_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    last = mod.main(iterations=30)
    by_name = {c["name"]: c for c in last["constraints"]}
    assert set(by_name) == {"volume", "tip_zone_density"}
    assert by_name["tip_zone_density"]["value"] <= -0.5 + 5e-3     # mean density of the zone >= 0.5


# --- review fixes (prices, log form, one-case max, xmin, imports, YAML) ----------------


def test_compliance_rows_are_log_with_price_100_and_report_linear_g(tmp_path):
    m = make_model(tmp_path, h=2.0)
    form = Formulation(weighted_compliance(), [volume_constraint(m.n_design, 0.3),
                                               compliance_constraint("side", 0.05)])
    ev = evaluate(m, form, np.full(m.n_design, 0.5), 2.0)
    c = ev.constraints[1]
    assert list(ev.c) == [1000.0, 100.0]
    assert ev.g[1] == pytest.approx(np.log(c["value"] / 0.05))     # what MMA gets
    assert c["g"] == pytest.approx(c["value"] / 0.05 - 1.0)        # what the report keeps
    assert c["limit"] == 0.05
    m.solver.free()


def test_constraint_form_and_price_checks():
    f = lambda p: (1.0, np.zeros(2))   # noqa: E731
    with pytest.raises(ValueError, match="log form needs a limit above 0"):
        Constraint("x", f, limit=0.0, form="log")
    with pytest.raises(ValueError, match="form must be linear or log"):
        Constraint("x", f, limit=1.0, form="sqrt")
    with pytest.raises(ValueError, match="price must be > 0"):
        Constraint("x", f, limit=1.0, price=0.0)
    assert Constraint("x", f, limit=1.0).price == 1000.0    # custom rows: linear, price 1000


def test_mma_price_vector_lets_the_cheap_row_give_way():
    """Two rows that cannot both be met: the row with the lower price is violated more."""
    n = 20
    x = np.full(n, 0.5)
    df0 = np.zeros(n)
    dg = np.vstack([np.ones(n), -np.ones(n)]) / n    # g1 = mean - a, g2 = b - mean: contradict
    g = np.array([0.5 - 0.3, 0.7 - 0.5])
    out = {}
    for name, c in (("cheap2", [1000.0, 100.0]), ("cheap1", [100.0, 1000.0])):
        mma = Mma(n, 2, 0.2)
        out[name] = mma.step(x, 1.0, df0, g, dg, c=c).mean()
    assert out["cheap2"] < 0.5 < out["cheap1"]    # each side follows the dearer row
    with pytest.raises(ValueError, match="price c"):
        Mma(n, 2, 0.2).step(x, 1.0, df0, g, dg, c=[1000.0, 0.0])


def test_one_case_max_form_is_the_sum_form(tmp_path):
    p = case_yaml(tmp_path, iterations=20, case_combine="max")
    d = yaml.safe_load(p.read_text())
    d["load_cases"] = d["load_cases"][:1]
    p.write_text(yaml.safe_dump(d))
    m = Model(build_problem(load_config(p)))
    form = formulation_from_config(m.problem.config, m.n_design)
    assert form.objective.form == "sum" and form.n_rows == 1
    assert worst_case_compliance(1).form == "sum"
    m.solver.free()


def test_max_form_needs_terms_above_zero(tmp_path):
    m = make_model(tmp_path, h=2.0)
    bad = Objective("neg", lambda p: [Response(-1.0, np.zeros(p.n_design)),
                                      Response(2.0, np.zeros(p.n_design))], "max", 2)
    with pytest.raises(ValueError, match="'max' form needs terms above 0"):
        evaluate(m, Formulation(bad, [volume_constraint(m.n_design, 0.3)]),
                 np.full(m.n_design, 0.5), 1.0)
    m.solver.free()


def test_xmin_goes_to_mma(tmp_path):
    m = make_model(tmp_path, iterations=12)
    form = formulation_from_config(m.problem.config, m.n_design)
    assert form.xmin == 0.0
    form.xmin = 0.02
    res = minimize(m, form)
    assert res.mma.xmin == 0.02 and res.x.min() >= 0.02 - 1e-15
    m.solver.free()


def test_adjoint_solve_keeps_last_iters(tmp_path):
    p = case_yaml(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "amg"
    p.write_text(yaml.safe_dump(d))
    m = Model(build_problem(load_config(p)))
    m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    before = list(m.solver.last_iters)
    m.solver.solve(np.ones(m.n_free), warm=False)
    assert m.solver.last_iters == before
    m.solver.free()


@pytest.mark.parametrize("text, value", [("5e-3", 0.005), ("2E2", 200.0), (" 0.25 ", 0.25)])
def test_yaml_compliance_limit_accepts_numeric_text(tmp_path, text, value):
    """YAML 1.1 loads 5e-3 as text. For compliance_limit a string that float() reads is that number."""
    p = case_yaml(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["load_cases"][1]["compliance_limit"] = text
    p.write_text(yaml.safe_dump(d))
    assert load_config(p).load_cases[1].compliance_limit == value
    raw = p.read_text().replace("compliance_limit: '" + text + "'", "compliance_limit: " + text.strip())
    p.write_text(raw)       # the same file written by hand: a bare 5e-3
    assert load_config(p).load_cases[1].compliance_limit == value


def test_yaml_numeric_text_is_only_for_compliance_limit(tmp_path):
    p = case_yaml(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["load_cases"][1]["weight"] = "1e0"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="weight' must be a number"):
        load_config(p)


def test_check_data_shows_limit_and_case_combine(tmp_path):
    from topopt.cli import derived_numbers
    dv = derived_numbers(load_config(case_yaml(tmp_path, limits={"side": 4.5}, case_combine="max")))
    assert [c["compliance_limit"] for c in dv["load_cases"]] == [None, 4.5]
    assert dv["case_combine"] == "max"


def test_part_builder_passes_case_combine_and_beta_schedule(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(ws))
    spec = {"name": "w", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [40, 20, 20]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 20]}}]},
        "load": {"tip": [{"box": {"min": [36, 8, 8], "max": [40, 12, 12]}}]}},
        "load_cases": [{"name": "a", "forces": [{"region": "tip", "force": [0, 0, -10.0]}],
                        "compliance_limit": 5.0},
                       {"name": "b", "forces": [{"region": "tip", "force": [0, 5.0, 0]}]}],
        "settings": {"voxel_size": 2.0, "case_combine": "max", "beta_schedule": [[0, 1], [5, 2]]}}
    r = server.build_regions(spec)
    assert r["ok"], r
    doc = yaml.safe_load((ws / r["yaml_rel"]).read_text())
    assert doc["settings"]["case_combine"] == "max"
    assert doc["settings"]["beta_schedule"] == [[0, 1], [5, 2]]
    cfg = load_config(ws / r["yaml_rel"])
    assert cfg.settings.case_combine == "max" and cfg.load_cases[0].compliance_limit == 5.0
