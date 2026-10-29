"""A5 rows in robust mode (A3b, CONTRACTS.md sections 22b and 31): the gradient and the end values.

Merge-gate test: it needs the code of A3b (`settings.robust`, `rho_stiff`) and of A5 (the rows).
"""

import json

import numpy as np
import pytest
import yaml

import test_motion_rows as base_tests
from test_motion_rows import ALL_ROWS, FD_DIRECT, MmaState, fd_error, limits_case, sub
from topopt import rows_motion as rm
from topopt.cli import main
from topopt.config import load_config
from topopt.formulation import Formulation, weighted_compliance
from topopt.optimize import Model, formulation_from_config, run
from topopt.problem import build_problem
from topopt.snapshot import RunWriter


def robust_case(tmp_path, eta=0.6, **kw):
    p = limits_case(tmp_path, ALL_ROWS, **kw)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(robust="eroded", eta_eroded=eta)
    p.write_text(yaml.safe_dump(d))
    return p


def evaluate_robust(m, form, x, beta=2.0):
    """Like `test_motion_rows.evaluate`, but solved with the density of the stiffness (the eroded one)."""
    m.solver._x_prev = None
    rho_tilde, rho = m.density(x, beta)
    rho_stiff = m.stiffness_density(rho_tilde, rho, beta)
    point = m.design_point(x, beta, rho_tilde, rho, m.solve(m.stiffness(rho_stiff)), rho_stiff)
    state = MmaState(m.n_design)
    state.f0_ref = 1.0
    return form.evaluate(point, state)


@pytest.fixture(autouse=True)
def robust_evaluate(monkeypatch):
    """`fd_error` of the base tests then differences the eroded design."""
    monkeypatch.setattr(base_tests, "evaluate", evaluate_robust)


def tight_model(p):
    m = Model(build_problem(load_config(p)))
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13
    assert m.robust is not None
    return m


@pytest.mark.parametrize("eta", [0.55, 0.6])
def test_row_gradients_in_robust_mode_match_finite_differences(tmp_path, eta):
    """The rows read U of the eroded design: wrt 'rho_stiff' makes the chain rule use the eroded slope."""
    m = tight_model(robust_case(tmp_path, eta))
    form = formulation_from_config(m.problem.config, m.n_design, m)
    form = Formulation(weighted_compliance(), form.constraints)
    err = {n: e for n, e in fd_error(m, form, steps=(1e-3, 1e-4)).items() if n != "volume"}
    assert len(err) == 10 and max(err.values()) < FD_DIRECT, err
    m.solver.free()


def test_the_default_wrt_would_be_wrong_in_robust_mode(tmp_path, monkeypatch):
    """The check sees the error: with wrt 'rho' the gradient is that of the nominal projection."""
    monkeypatch.setattr(rm, "GRAD_WRT", "rho")
    m = tight_model(robust_case(tmp_path, 0.6))
    form = formulation_from_config(m.problem.config, m.n_design, m)
    form = Formulation(weighted_compliance(), form.constraints)
    err = {n: e for n, e in fd_error(m, form, steps=(1e-3, 1e-4)).items() if n != "volume"}
    assert max(err.values()) > 100 * FD_DIRECT, err
    m.solver.free()


def test_in_a_default_run_rho_stiff_is_rho(tmp_path):
    """Without robust mode the rows give the same gradient with 'rho_stiff' and with 'rho' (bit for bit)."""
    def grads(wrt):
        old = rm.GRAD_WRT
        rm.GRAD_WRT = wrt
        try:
            m = Model(build_problem(load_config(limits_case(sub(tmp_path, wrt), ALL_ROWS))))
            form = formulation_from_config(m.problem.config, m.n_design, m)
            x = np.random.default_rng(1).uniform(0.3, 0.7, m.n_design)
            rho_tilde, rho = m.density(x, 2.0)
            point = m.design_point(x, 2.0, rho_tilde, rho, m.solve(m.stiffness(rho)))
            return form.evaluate(point, __import__("topopt.mma", fromlist=["MmaState"]).MmaState(m.n_design)).dg
        finally:
            rm.GRAD_WRT = old
    a, b = grads("rho_stiff"), grads("rho")
    from topopt.solver_choice import resolve_solver
    if resolve_solver("auto") != "amg":
        np.testing.assert_array_equal(a, b)
    else:                                       # AMG: the adjoint solves differ in the last digits
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12 * np.abs(b).max())


def test_robust_run_reports_the_end_values_of_the_nominal_design(tmp_path):
    """`value_end` of metrics.json is the nominal design (as compliance_final); the eroded one is `value_end_eroded`."""
    p = robust_case(tmp_path, 0.6, h=2.0, iterations=6)
    out_dir = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out_dir), "--quiet"]) == 0
    metrics = json.loads((out_dir / "result" / "metrics.json").read_text())
    rj = json.loads((out_dir / "robust.json").read_text())
    hist = [json.loads(line) for line in (out_dir / "history.jsonl").read_text().splitlines()]
    assert set(rj["rows_nominal"]) == {r["name"] for r in metrics["rows"]} and len(metrics["rows"]) == 10
    for r in metrics["rows"]:
        last = next(c for c in hist[-1]["constraints"] if c["name"] == r["name"])
        assert r["value_end"] == rj["rows_nominal"][r["name"]]
        assert r["value_end_eroded"] == pytest.approx(rm.physical_value(r["name"], last["value"], last["limit"]))
        assert r["value_end"] != r["value_end_eroded"]          # the two designs differ
        assert r["g_end"] == pytest.approx(rm.guarded_value(r["name"], r["value_end"], r["limit"]) / r["limit"] - 1)
        assert r["met"] == (r["g_end"] <= 5e-3)
        first = next(c for c in hist[0]["constraints"] if c["name"] == r["name"])
        assert r["value_start"] == pytest.approx(rm.physical_value(r["name"], first["value"], r["limit"]))   # iteration 0: eroded


def test_robust_nominal_values_equal_a_solve_of_the_final_nominal_design(tmp_path):
    p = robust_case(tmp_path, 0.6, h=2.0, iterations=5)
    prob = build_problem(load_config(p))
    out = RunWriter(tmp_path / "r", quiet=True)
    out.clean()
    x = run(prob, out)
    from topopt.solver_choice import resolve_solver
    run_solver = resolve_solver("auto")
    rj = json.loads((tmp_path / "r" / "robust.json").read_text())
    m = tight_model(p)
    from topopt.optimize import beta_at
    beta = beta_at(m.problem.config.settings.beta_schedule, m.problem.config.settings.iterations)
    _rt, rho_n = m.density(x, beta)
    U = m.solve(m.stiffness(rho_n))
    ms = rm.MotionSet(m.problem.config)
    ms.bind_model(m)
    ref = ms.physical_values(U)
    for name, v in rj["rows_nominal"].items():
        assert v == pytest.approx(ref[name], rel=1e-6 if run_solver != "amg" else 1e-4, abs=1e-12)
    m.solver.free()


def test_a_default_run_has_no_nominal_block(tmp_path):
    p = limits_case(tmp_path, ALL_ROWS, h=2.0, iterations=3)
    out_dir = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out_dir), "--quiet"]) == 0
    assert not (out_dir / "robust.json").exists()
    rows = json.loads((out_dir / "result" / "metrics.json").read_text())["rows"]
    assert len(rows) == 10 and all("value_end_eroded" not in r for r in rows)
