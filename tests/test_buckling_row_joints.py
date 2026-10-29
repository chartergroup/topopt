"""The buckling row and the joints of A7b (CONTRACTS.md sections 38 and 39): the row runs with the condensed stiffness S.

A run with joints holds the factor of the bordered matrix `[[K + Kb, C], [C^T, D]]`. Its grid block inverse is `S^-1` with
`S = K + Kb - C D^-1 C^T` (not `K`). The row uses `S` for the eigen solve, the K-normalisation, the adjoint and both residuals.
`dS/drho = dK/drho` (`Kb`, `C` and `D` do not depend on the density), so the three gradient parts are the same. The pad case of
`test_joints.py` is the model here. The row was refused for joints before this change (the review measured `factor_mismatch`).
"""
import json

import numpy as np
import pytest
import scipy.linalg as sla

from helpers import needs_direct_solver
from test_buckling_row import STEPS, ModesSpy, point_at, sum_of_k_normalised
from test_joints import case as joint_case
from test_multirow_mma import evaluate
from topopt import buckling as bk
from topopt import buckling_row as br
from topopt import buckling_voxel as bv
from topopt.cli import check_data, main
from topopt.config import load_config
from topopt.formulation import Formulation, volume_constraint, weighted_compliance
from topopt.joints import support_matrix
from topopt.optimize import Model, formulation_from_config
from topopt.problem import build_problem


def joint_row_yaml(tmp_path, solver="auto", joint=None, **buckling):
    def add(d):
        d["buckling"] = {"min_factor": 100.0, **buckling}
    return joint_case(tmp_path, h=2.0, mutate=add, solver=solver, joint=joint)


def joint_model(tmp_path, **kw):
    m = Model(build_problem(load_config(joint_row_yaml(tmp_path, **kw))))
    assert m.problem.joints is not None
    return m


def test_the_config_and_the_formulation_do_not_refuse_the_row_with_joints(tmp_path):
    cfg = load_config(joint_row_yaml(tmp_path))
    assert br.config_refusal(cfg) is None or "no such factor" in br.config_refusal(cfg)    # only a solver without a direct factor refuses
    assert not hasattr(br, "joints_refusal")


@needs_direct_solver
def test_the_formulation_and_the_model_accept_a_joint_model(tmp_path):
    m = joint_model(tmp_path)
    assert br.model_refusal(m) is None
    assert br.config_refusal(m.problem.config) is None
    form = formulation_from_config(m.problem.config, m.n_design, m)
    assert [c.name for c in form.constraints if c.name.startswith("buckling:")] == ["buckling:down"]
    m.solver.free()


def test_a_joint_run_with_a_solver_that_has_no_direct_factor_is_still_refused(tmp_path):
    pytest.importorskip("pyamg")
    m = joint_model(tmp_path, solver="amg")
    assert m.solver_name == "amg"
    assert "no such factor" in br.model_refusal(m) and "options" in br.model_refusal(m)
    assert "no such factor" in br.config_refusal(m.problem.config)


def test_preflight_and_check_give_the_beam_fact_and_do_not_refuse(tmp_path, capsys):
    p = joint_row_yaml(tmp_path)
    cfg = load_config(p)
    info = br.preflight(cfg, "pardiso", 10_000, 5_000, ram_bytes=64 * 10 ** 9)
    assert info["will_run"] and info["reason"] is None and not info["warnings"]
    assert any("no geometric stiffness" in f and "not for buckling of the beams" in f for f in info["facts"])
    d = check_data(p)
    assert d["buckling_row"]["reason"] != "joints"
    assert not [w for w in d["warnings"] if "buckling row:" in w and "Schur" in w]
    assert main(["check", str(p)]) == 0
    assert "Schur complement, not of K" not in capsys.readouterr().out


@needs_direct_solver
def test_a_normal_pass_with_a_joint_has_a_small_adjoint_residual_and_the_dense_value(tmp_path):
    """The row value with a joint against a dense `scipy.linalg.eigh` of the whole bordered matrix (padded `Kg`), independent of
    ARPACK, of the factor and of the Schur complement. The pass is a result (status ok, no failure)."""
    m = joint_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 100.0, 40.0, 6, max_seconds=None, skip=False)
    resp = row(point_at(m, rho))
    info = resp.info
    assert info["status"] == bk.OK and info["evaluated"] and row.get_state()["n_fail"] == 0
    assert info["residual_adjoint"] <= 1e-11 and info["residual_u"] <= 1e-9 and info["residual_max"] <= 1e-8
    geo = br.Geometry(m)
    K = m.stiffness(rho).toarray()
    C, D = m.problem.joints.border(m.free_dofs)
    n, nb = C.shape
    B = np.block([[K, C.toarray()], [C.toarray().T, D.toarray()]])
    u = np.linalg.solve(B, np.vstack([m.f_red[:, :1], np.zeros((nb, 1))]))[:n, 0]
    u_all = np.zeros(m.n_dof_all)
    u_all[m.free_dofs] = u
    rho_all = np.ones(geo.n_el)
    rho_all[m.design_pos] = rho
    Kg = bv.hex8_kg(m.assembler, geo.edof, u_all, geo.h, geo.E, geo.nu, rho_all ** m.penal).toarray()
    Kgp = np.zeros((n + nb, n + nb))
    Kgp[:n, :n] = Kg
    mu = sla.eigh(-Kgp, B, eigvals_only=True)[::-1][:6]
    assert resp.value == pytest.approx(br.ks_modes(mu, 100.0, 40.0)[0], rel=1e-8, abs=1e-12)
    assert info["load_factors"][0] == pytest.approx(1.0 / mu[0], rel=1e-8)
    m.solver.free()


@needs_direct_solver
def test_a_factor_of_another_density_is_still_caught_with_a_joint(tmp_path):
    """The factor in `model.solver` is the bordered factor of another density: the adjoint residual (with S) sees it."""
    m = joint_model(tmp_path)
    rng = np.random.default_rng(5)
    rho1, rho2 = rng.uniform(0.3, 0.7, m.n_design), rng.uniform(0.3, 0.7, m.n_design)
    pt = point_at(m, rho1)
    m.solver.factor(m.stiffness(rho2))
    row = br.BucklingRow(0, 100.0, 40.0, 6, max_seconds=None, skip=False)
    info = row(m.design_point(rho1, 1.0, rho1, rho1, pt.U, rho1)).info
    assert info["status"] == br.FACTOR_MISMATCH and info["residual_adjoint"] > bk.RESIDUAL_FACT
    m.solver.free()


@needs_direct_solver
@pytest.mark.parametrize("wrong", [True, False])
def test_a_failed_eigen_solve_with_a_joint_checks_the_bordered_factor(tmp_path, monkeypatch, wrong):
    """With a joint, K is the condensed S: the check after a failed eigen solve uses S and the bordered factor, like the adjoint."""
    ModesSpy(monkeypatch, fail={0})
    m = joint_model(tmp_path)
    rng = np.random.default_rng(5)
    rho1, rho2 = rng.uniform(0.3, 0.7, m.n_design), rng.uniform(0.3, 0.7, m.n_design)
    pt = point_at(m, rho1)
    if wrong:
        m.solver.factor(m.stiffness(rho2))
    row = br.BucklingRow(0, 100.0, 40.0, 6, max_seconds=None, skip=False)
    info = row(m.design_point(rho1, 1.0, rho1, rho1, pt.U, rho1)).info
    assert info["status"] == (br.FACTOR_MISMATCH if wrong else bk.NOT_CONVERGED)
    assert (info["residual_adjoint"] > bk.RESIDUAL_FACT) == wrong
    m.solver.free()


@needs_direct_solver
def test_fd_through_the_filter_and_the_projection_with_a_joint(tmp_path):
    """The row `buckling:down` of the pad case with a joint (E 210000) through `Formulation.evaluate` (filter, Heaviside, volume row,
    compliance), against central differences of the same function. KS q 40 over 6 modes, F = the lowest factor at the FD design
    (z is about 1), 10 voxels, best of the steps 1e-3, 3e-4, 1e-4, the error relative to the voxel gradient with the floor 1e-3 of the
    largest gradient of the row (the rule of `test_buckling_row.py`). Bar 1e-6 (T1_BAR there). Measured worst error 5.4e-9 (Apple Cholesky, one machine).
    A gradient that used K where the solver holds S would be wrong at the first digit."""
    m = joint_model(tmp_path, joint={"E": 210000.0})

    def form(F):
        return Formulation(weighted_compliance(), [volume_constraint(m.n_design, 0.3), br.buckling_ks_constraint("down", F)])

    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 0.7, m.n_design)
    beta = 1.0
    F = evaluate(m, form(1.0), x, beta).constraints[1]["lowest_factor"]
    f = form(F)
    ev = evaluate(m, f, x, beta)
    entry = ev.constraints[1]
    assert entry["status"] == bk.OK and entry["evaluated"] and entry["residual_adjoint"] <= bk.RESIDUAL_FACT
    assert abs(entry["g"]) < 0.05
    floor = 1e-3 * np.abs(ev.dg[1]).max()
    worst = 0.0
    for e in rng.choice(m.n_design, size=10, replace=False):
        best = np.inf
        for eps in STEPS:
            xp, xm = x.copy(), x.copy()
            xp[e] += eps
            xm[e] -= eps
            fd = (evaluate(m, f, xp, beta).g[1] - evaluate(m, f, xm, beta).g[1]) / (2 * eps)
            best = min(best, abs(fd - ev.dg[1, e]) / max(abs(ev.dg[1, e]), floor))
        worst = max(worst, best)
    print("FD worst error with a joint", worst)
    assert worst < 1e-6, worst
    m.solver.free()


@needs_direct_solver
def test_run_with_a_joint_and_the_row_evaluates_and_writes_no_refusal(tmp_path):
    p = joint_row_yaml(tmp_path, min_factor=100.0)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "done"
    last = [json.loads(ln) for ln in (out / "history.jsonl").read_text().splitlines()][-1]
    row = [c for c in last["constraints"] if c["name"] == "buckling:down"][0]
    assert row["n_eval"] >= 1 and row["n_fail"] == 0 and row["lowest_factor"] is not None


@needs_direct_solver
def test_the_start_vector_with_a_joint_has_the_size_of_the_condensed_matrix_and_is_given_to_the_next_solve(tmp_path, monkeypatch):
    """#155: with joints `K` of the eigen solve is the condensed `S` (a `LinearOperator`), so the vector is built with `S` and has the size of `S`,
    which is the number of grid dofs of the free set. The pad model is small (the dense path of `buckling_modes` runs and ignores `v0`): this test is
    about what the row keeps and passes, not about the solves."""
    spy = ModesSpy(monkeypatch)
    m = joint_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 100.0, 40.0, 6, max_seconds=None, skip=False)
    row(point_at(m, rho, 1.0, 0))
    row(point_at(m, rho, 1.0, 1))
    n = spy.calls[0]["K"].shape[0]
    assert n == m.n_free and spy.v0s[0] is None
    assert spy.v0s[1] is not None and spy.v0s[1].shape == (n,)
    np.testing.assert_allclose(spy.v0s[1], sum_of_k_normalised(spy.calls[0]["res"], spy.calls[0]["K"]), rtol=1e-12, atol=0)
    assert row.get_state()["v0"].shape == (n,)
    m.solver.free()
