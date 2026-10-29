"""Tests of the buckling row (A6b, CONTRACTS.md section 39): `topopt.buckling_row`.

Tolerances (each one is stated at the test that uses it). The rule for a bar of an FD test: 30 times the worst measured error of
that case, rounded up to a power of ten; and it stays at least 1e3 below the error without the stress part of the gradient.

Noise floor of an FD check of a buckling factor (Linux, PARDISO, ARPACK at the A6a tolerance 1e-8): the eigen solve has about 1e-8
relative noise in `mu`, so the central difference has a floor of about 1e-8 / step. The A2 steps (1e-4, 3e-5, 1e-5) are too small.
The tests use the best of the steps 1e-3, 3e-4 and 1e-4 for each voxel. Measured on the A2 cantilever (h = 2 mm, 249 design voxels,
1080 dofs, case `down`, KS q 40 over 6 modes, 10 voxels, worst voxel; the numbers of this branch):
- beta 1: best of three steps 7.7e-9; by step 1e-3: 1.1e-7, 3e-4: 9.8e-9, 1e-4: 2.1e-7, 3e-5: 8.7e-7.
- beta 4: best of three steps 1.7e-8; by step 1e-3: 6.6e-7, 3e-4: 6.1e-8, 1e-4: 8.4e-8, 3e-5: 6.1e-7.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla
import trimesh
import yaml

from helpers import needs_direct_solver
from test_multirow_mma import evaluate, make_model
from topopt import buckling as bk
from topopt import buckling_row as br
from topopt import buckling_voxel as bv
from topopt.config import load_config
from topopt.formulation import DesignPoint, Formulation, volume_constraint, weighted_compliance
from topopt.optimize import Model
from topopt.problem import build_problem

STEPS = (1e-3, 3e-4, 1e-4)
FD_BETAS = (1.0, 4.0)
T1_BAR = 1e-6          # 30 x 1.74e-8 = 5.2e-7, rounded up to a power of ten (the error without the stress part is 30 to 92)


# --- models ------------------------------------------------------------------


def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def column_yaml(tmp_path, a=2, b=3, length=30, load=1000.0, lateral=0.0, iterations=30, volume_fraction=0.5, name="column",
                extra_settings=None, spring=None, mirror=None):
    """A fixed-free column `a x b` mm across and `length` mm long (voxel 1 mm), a plate below, a cap on top with the axial load
    `load` N (and `lateral` N along x). `spring` (a stiffness): the base is a spring pad on all axes and no clamp (A7a).
    `mirror` (a list of planes) goes to the YAML key (A4a)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    box(tmp_path / "design.stl", (0, 0, 0), (a, b, length))
    box(tmp_path / "plate.stl", (0, 0, -2), (a, b, 0))
    box(tmp_path / "cap.stl", (0, 0, length), (a, b, length + 2))
    d = {"name": name, "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"cap": "cap.stl"}},
         "load_cases": [{"name": "axial", "weight": 1.0, "forces": [{"region": "cap", "force": [lateral, 0.0, -load]}]}],
         "material": {"E": 110000.0, "nu": 0.34},
         "settings": {"voxel_size": 1.0, "volume_fraction": volume_fraction, "iterations": iterations,
                      "beta_schedule": [[0, 1], [int(iterations * 0.3), 2], [int(iterations * 0.5), 4], [int(iterations * 0.7), 8]],
                      **(extra_settings or {})}}
    if spring is not None:
        d["regions"]["fixed"] = []
        d["regions"]["load"]["plate"] = "plate.stl"
        d["springs"] = [{"region": "plate", "stiffness": [spring, spring, spring]}]
    if mirror is not None:
        d["mirror"] = mirror
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def column_model(tmp_path, **kw):
    m = Model(build_problem(load_config(column_yaml(tmp_path, **kw))))
    return m


def point_at(m, rho, beta=1.0, iteration=None, last=False):
    """A `DesignPoint` at the stiffness density `rho` (no filter): the stiffness is factored, `U` solved."""
    U = m.solve(m.stiffness(rho))
    return m.design_point(rho, beta, rho, rho, U, rho, iteration=iteration, last=last)


# --- the dense reference ------------------------------------------------------


def dense_state(m, geo, rho, ci):
    """Dense K, the displacement of the case, and Kg, for the density `rho` (no ARPACK, no sparse factor)."""
    K = m.stiffness(rho).toarray()
    u = np.linalg.solve(K, m.f_red[:, ci])
    u_all = np.zeros(m.n_dof_all)
    u_all[m.free_dofs] = u
    rho_all = np.ones(geo.n_el)
    rho_all[m.design_pos] = rho
    Kg = bv.hex8_kg(m.assembler, geo.edof, u_all, geo.h, geo.E, geo.nu, rho_all ** m.penal).toarray()
    return K, u, u_all, rho_all, Kg


def dense_ks(m, geo, rho, ci, F, q, modes):
    """The KS row value S from `scipy.linalg.eigh`: an eigen solver that is independent of ARPACK and of the factor."""
    K, _u, _ua, _ra, Kg = dense_state(m, geo, rho, ci)
    mu = sla.eigh(-Kg, K, eigvals_only=True)[::-1][:modes]
    return br.ks_modes(mu, F, q)[0]


def dense_parts(m, geo, rho, ci, F, q, modes):
    """`(explicit + K part, stress part)` of the gradient of S, from the dense modes (module text of `buckling_row`)."""
    K, u, u_all, rho_all, Kg = dense_state(m, geo, rho, ci)
    mu, phi = sla.eigh(-Kg, K)                       # K-orthonormal columns, mu ascending
    mu, phi = mu[::-1][:modes], phi[:, ::-1][:, :modes]
    _S, _p, w = br.ks_modes(mu, F, q)
    explicit, kpart, s, _e = br.gradient_parts(m, geo, rho_all, u_all, mu, phi, w)
    lam = np.linalg.solve(K, s)
    point = DesignPoint(m, rho, 1.0, rho, rho, np.linalg.solve(K, m.f_red), rho)
    return explicit + kpart, point.stiffness_derivative(lam, u)


def fd_best(fun, rho, e, steps=STEPS):
    """Central differences of `fun(rho)` at voxel `e` for each step."""
    out = []
    for eps in steps:
        rp, rm = rho.copy(), rho.copy()
        rp[e] += eps
        rm[e] -= eps
        out.append((fun(rp) - fun(rm)) / (2 * eps))
    return out


def raw_errors(m, rho, ci, F, q, modes, voxels, grad_fn):
    """Worst over the voxels of the best-of-steps error of the gradient `grad_fn`, relative to the voxel gradient (floor 1e-3 of the largest)."""
    geo = br.Geometry(m)
    grad = grad_fn
    floor = 1e-3 * np.abs(grad).max()
    worst = 0.0
    for e in voxels:
        fds = fd_best(lambda r: dense_ks(m, geo, r, ci, F, q, modes), rho, e)
        err = min(abs(fd - grad[e]) for fd in fds) / max(abs(grad[e]), floor)
        worst = max(worst, err)
    return worst


# --- T3: the aggregate on plain arrays ---------------------------------------


def test_ks_modes_is_an_upper_bound_with_a_known_offset():
    rng = np.random.default_rng(0)
    for k, F, q in ((6, 150.0, 40.0), (3, 0.01, 10.0), (1, 7.0, 40.0), (12, 1.0, 80.0)):
        mu = rng.uniform(-0.5, 1.0, k) / F
        S, p, w = br.ks_modes(mu, F, q)
        z = F * mu
        assert z.max() <= S <= z.max() + math.log(k) / q + 1e-15
        assert abs(p.sum() - 1.0) < 1e-14 and np.all(p >= 0)
        assert np.allclose(w, F * p, rtol=0, atol=1e-15)
        # the value equals a direct log-sum-exp (no shift), for a z that does not overflow
        assert abs(S - math.log(np.exp(q * z).sum()) / q) < 1e-13


def test_ks_modes_does_not_overflow_and_a_negative_mu_has_no_weight():
    S, p, w = br.ks_modes(np.array([1e6, 5e5, -3e5, 0.0]), 1.0, 40.0)
    assert np.isfinite(S) and abs(S - 1e6) < 1e-6 and abs(p[0] - 1.0) < 1e-15
    S, p, _w = br.ks_modes(np.array([0.0, -1.0]), 2.0, 40.0)    # no positive mu (a tension case): S is at most ln(k)/q
    assert -2.0 / 1.0 <= S <= math.log(2) / 40.0 + 1e-15


def test_k_normalise_gives_unit_k_norm():
    import scipy.sparse as sp
    rng = np.random.default_rng(1)
    A = rng.standard_normal((30, 30))
    K = sp.csc_matrix(A @ A.T + 30 * np.eye(30))
    phi = rng.standard_normal((30, 4)) * np.array([3.0, 0.2, 1.0, 50.0])
    out = br.k_normalise(phi, K)
    assert np.abs(np.diag(out.T @ (K @ out)) - 1.0).max() < 1e-13          # 1e-13: a few roundings of a 30 x 30 product
    assert not np.shares_memory(out, phi)                                  # a copy


# --- T2: the formula with an independent eigen solver -------------------------


@needs_direct_solver
def test_raw_gradient_against_a_dense_oracle_and_the_stress_part_is_needed(tmp_path):
    """The gradient of S with respect to `rho_stiff` (no filter) against central differences of a dense `scipy.linalg.eigh`.

    A gray column 2 x 3 x 30 voxels, an axial load and a lateral one, `rho` random in 0.3 to 0.7, KS q 40 over 6 modes,
    F = the lowest factor (z is about 1), 10 voxels, best of the steps 1e-3, 3e-4, 1e-4.
    Negative control: the gradient without the stress part (explicit + K part) is wrong by a factor of about 1 or more.

    Measured (this branch, Linux, PARDISO), the committed sample (seed 5): full gradient 1.6e-5 at 4 threads and 9.4e-6 at 1 thread, without the stress part 3.4.
    Six other samples (seeds 0 to 4): full 2.3e-6 to 6.6e-5, without the stress part 1.4 to 36. Bar 1e-3 for the full gradient:
    30 x 1.6e-5 = 4.9e-4, rounded up to a power of ten, and 3.4e3 below the error without the stress part.
    The error is the truncation of the central difference (it falls with the step squared), not the eigen solve.
    """
    m = column_model(tmp_path, lateral=50.0)
    ci, q, modes = 0, 40.0, 6
    rng = np.random.default_rng(5)
    rho = rng.uniform(0.3, 0.7, m.n_design)
    geo = br.Geometry(m)
    K, _u, _ua, _ra, Kg = dense_state(m, geo, rho, ci)
    F = 1.0 / sla.eigh(-Kg, K, eigvals_only=True)[-1]
    rest, stress = dense_parts(m, geo, rho, ci, F, q, modes)
    voxels = rng.choice(m.n_design, size=10, replace=False)
    full_err = raw_errors(m, rho, ci, F, q, modes, voxels, rest + stress)
    control_err = raw_errors(m, rho, ci, F, q, modes, voxels, rest)
    print("T2 errors", full_err, control_err)
    assert full_err < 1e-3, full_err
    assert control_err > 1.0, control_err
    assert np.linalg.norm(stress) > 0.5 * np.linalg.norm(rest + stress)     # the stress part is not a small correction
    m.solver.free()


# --- T1: FD through the real chain --------------------------------------------


@needs_direct_solver
@pytest.mark.parametrize("beta", FD_BETAS)
def test_fd_through_the_filter_and_the_projection(tmp_path, beta):
    """The row `buckling:down` on the A2 cantilever through `Formulation.evaluate` (filter, Heaviside, volume row and compliance).

    KS q 40 over 6 modes, F = the lowest factor at the FD design (z is about 1), 10 voxels, best of the steps 1e-3, 3e-4, 1e-4, the
    error relative to the voxel gradient with the A2 floor of 1e-3 of the largest gradient of the row. Bar `T1_BAR` (module text).
    """
    m = make_model(tmp_path, h=2.0)
    assert m.solver_name in ("pardiso", "cholesky")

    def form(F):
        return Formulation(weighted_compliance(), [volume_constraint(m.n_design, 0.3), br.buckling_ks_constraint("down", F)])

    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 0.7, m.n_design)
    F = evaluate(m, form(1.0), x, beta).constraints[1]["lowest_factor"]
    f = form(F)
    ev = evaluate(m, f, x, beta)
    entry = ev.constraints[1]
    assert entry["status"] == bk.OK and entry["evaluated"] and entry["residual_adjoint"] <= bk.RESIDUAL_FACT
    assert abs(entry["g"]) < 0.05                                           # z is about 1, so the row is near its limit
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
    assert worst < T1_BAR, worst
    m.solver.free()


# --- T4: the factor check is the adjoint residual -----------------------------


@needs_direct_solver
def test_a_normal_pass_has_a_small_adjoint_residual_and_unit_k_norm(tmp_path):
    """A normal evaluated pass. `residual_adjoint` is at most 1e-11: 30 x the worst measured 2.5e-13 (hook, passes 0 and 80; this column 1.9e-15), rounded up."""
    m = column_model(tmp_path, lateral=50.0)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    info = row(point_at(m, rho)).info
    assert info["status"] == bk.OK and info["evaluated"] and info["reason"] == "no_skip"
    assert info["residual_adjoint"] <= 1e-11 and info["residual_u"] <= 1e-9 and info["residual_max"] <= 1e-8
    m.solver.free()


@needs_direct_solver
def test_a_factor_of_another_density_is_caught_by_the_adjoint_residual_and_not_by_the_u_residual(tmp_path):
    """K and U of the point are right, the factor in `model.solver` is the factor of another density (review M1).

    `residual_u` stays small (1.8e-11 here: the U of the pass is right), so the residual of U cannot see the fault.
    `residual_adjoint` reads 1.04 (measured; hook: 1.5e4), above `RESIDUAL_FACT`. The result is "no usable result":
    status `factor_mismatch`, the last value (0 at the first pass), a zero gradient, `n_fail` 1, the row unknown again,
    one fact line in the log, and no exception.
    """
    m = column_model(tmp_path, lateral=50.0)
    rng = np.random.default_rng(5)
    rho1, rho2 = rng.uniform(0.3, 0.7, m.n_design), rng.uniform(0.3, 0.7, m.n_design)
    pt = point_at(m, rho1)
    m.solver.factor(m.stiffness(rho2))
    lines = []
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False, log=lines.append)
    resp = row(m.design_point(rho1, 1.0, rho1, rho1, pt.U, rho1))
    info = resp.info
    assert info["status"] == br.FACTOR_MISMATCH and not info["evaluated"]
    assert info["residual_u"] < 1e-9 and info["residual_adjoint"] > bk.RESIDUAL_FACT
    assert resp.value == 0.0 and not resp.grad.any()
    st = row.get_state()
    assert st["n_fail"] == 1 and st["n_known"] == 0 and st["last_value"] is None
    assert len(lines) == 1 and "factor_mismatch" in lines[0] and "options" in lines[0]
    row(m.design_point(rho1, 1.0, rho1, rho1, pt.U, rho1))
    assert len(lines) == 1                              # the first failure only
    m.solver.free()


@needs_direct_solver
def test_a_near_singular_design_is_a_valid_result(tmp_path):
    """Void layers held by E_min (a column with a cut): K is near singular, `residual_u` reads 4.7e-8 (one layer) and 1.2e-7 (two),
    and the eigen residual 3.7e-5 and 1.5e-5. The adjoint residual stays at 1.7e-15 and the pass is a result (status ok, no failure)."""
    m = column_model(tmp_path, lateral=50.0)
    ctr = m.design_centers()
    for layers in ([(14, 15)], [(14, 15), (7, 8)]):
        rho = np.full(m.n_design, 0.5)
        for lo, hi in layers:
            rho[(ctr[:, 2] > lo) & (ctr[:, 2] < hi)] = 0.0
        row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
        info = row(point_at(m, rho)).info
        assert info["status"] == bk.OK and info["evaluated"] and row.get_state()["n_fail"] == 0
        assert info["residual_adjoint"] <= 1e-11
    m.solver.free()


# --- T5: repeated modes --------------------------------------------------------


@needs_direct_solver
def test_exact_double_mode_the_ks_sum_over_a_complete_cluster_is_right_and_one_mode_is_not(tmp_path):
    """A square column 2 x 2 x 40 voxels, uniform gray density 0.5, axial load: the modes come in exact pairs (lambda 0.0310 twice,
    0.2757 twice, ...; the relative gap of the first pair is 4e-10).

    (a) `modes` 6 holds complete pairs: the FD error of the KS gradient against the dense oracle is at most 3.9e-5 (4 threads) or 3.3e-5 (1 thread) for the
    committed 10 voxels (5 other samples of 10 voxels: 5.7e-6 to 7.9e-5, the truncation of the step). `near_repeated` is true, `cluster_cut` false.
    Bar 1e-2: 30 x 3.9e-5 = 1.2e-3, rounded up to a power of ten.
    (b) `modes` 1 cuts the pair: the error is 0.23 (4 threads, all 5 samples) or 0.93 (1 thread): it depends on the basis that the eigen solver picks
    inside the pair, because the single-mode gradient of a repeated factor has no meaning. The row says so: `cluster_cut` true with `ks_share_last` 1. The test asserts that the error is above 1e-2 and `cluster_cut`.
    """
    m = column_model(tmp_path, a=2, b=2, length=40, lateral=0.0)
    geo = br.Geometry(m)
    rho = np.full(m.n_design, 0.5)
    K, _u, _ua, _ra, Kg = dense_state(m, geo, rho, 0)
    F = 1.0 / sla.eigh(-Kg, K, eigvals_only=True)[-1]
    voxels = np.random.default_rng(2).choice(m.n_design, size=10, replace=False)
    pt = point_at(m, rho)
    out = {}
    for modes in (6, 1):
        row = br.BucklingRow(0, F, 40.0, modes, max_seconds=None, skip=False)
        resp = row(pt)
        out[modes] = (resp.info, raw_errors(m, rho, 0, F, 40.0, modes, voxels, resp.grad))
    info6, err6 = out[6]
    info1, err1 = out[1]
    print("T5 errors", err6, err1)
    assert info6["near_repeated"] and not info6["cluster_cut"] and info6["status"] == bk.OK
    assert err6 < 1e-2, err6
    assert info1["cluster_cut"] and abs(info1["ks_share_last"] - 1.0) < 1e-12
    assert err1 > 1e-2, err1
    m.solver.free()


# --- the loop on a small column (T6, T7, T14) ----------------------------------

LOOP = dict(a=3, b=3, length=30, load=1500.0, lateral=75.0, iterations=60)   # 3 x 3 x 30 voxels, 270 design voxels
NOT_MET = 5e-3                                                                # `formulation.NOT_MET`


def loop_model(tmp_path, **kw):
    return column_model(tmp_path, **{**LOOP, **kw})


def volume_form(m, *rows, objective=None):
    return Formulation(objective or weighted_compliance(), [volume_constraint(m.n_design, 0.5), *rows])


def final_density(m, res):
    beta = m.problem.config.settings.beta_schedule[-1][1]
    return m.density(res.x, beta)[1]


def dense_lowest(m, rho, ci=0):
    """The lowest buckling factor of the density `rho` from the dense eigen solver (independent of the row)."""
    geo = br.Geometry(m)
    K, _u, _ua, _ra, Kg = dense_state(m, geo, rho, ci)
    return 1.0 / sla.eigh(-Kg, K, eigvals_only=True)[-1]


def no_row_trajectory(m):
    """Run without a row; a probe row evaluates the lowest factor at every pass (no effect on MMA). Return the list of factors."""
    from topopt.optimize import minimize
    probe = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    lam = []
    minimize(m, volume_form(m), on_iteration=lambda rec, point: lam.append(probe(point).info["lowest_factor"]))
    return lam


class FakePoint:
    """The three attributes that `evaluation_reason` reads."""

    def __init__(self, iteration, beta=1.0, last=False):
        self.iteration, self.beta, self.last = iteration, beta, last


def state(**kw):
    st = {"n_known": 2, "last_lowest": 10.0, "last_eval": 3, "beta": 1.0}
    st.update(kw)
    return st


def test_evaluation_reason_table():
    F = 4.0                               # SKIP_FACTOR x F = 6
    cases = [
        (state(), FakePoint(None), "no_pass_number"),                              # a point built by hand
        (state(n_known=0), FakePoint(4), "unknown"),                               # the first pass
        (state(n_known=1), FakePoint(4), "unknown"),                               # known only after two evaluations
        (state(beta=None), FakePoint(4), "stage_start"),
        (state(), FakePoint(4, beta=2.0), "stage_start"),                          # the beta changed
        (state(), FakePoint(4, last=True), "last_pass"),
        (state(last_lowest=6.0), FakePoint(4), "near"),                            # the lowest factor is at SKIP_FACTOR x F
        (state(last_lowest=6.01), FakePoint(4), None),                             # just above: skip
        (state(last_eval=3), FakePoint(3 + br.EVERY), "every"),
        (state(last_eval=3), FakePoint(3 + br.EVERY - 1), None),
        (state(last_eval=None), FakePoint(4), "every"),
        (state(last_lowest=None), FakePoint(4), None),                             # no positive factor at the last evaluation: skip mode
        (state(n_known=0), FakePoint(None), "no_pass_number"),                     # the order: no pass number first
        (state(n_known=0, beta=None), FakePoint(4, beta=2.0, last=True), "unknown"),   # unknown before stage_start and last_pass
        (state(beta=None), FakePoint(4, last=True), "stage_start"),                # stage_start before last_pass
        (state(last_lowest=6.0), FakePoint(4, last=True), "last_pass"),            # last_pass before near
    ]
    for st, pt, want in cases:
        assert br.evaluation_reason(st, pt, F) == want, (st, vars(pt), want)
    assert (br.SKIP_FACTOR, br.EVERY, br.KNOWN_EVALUATIONS, br.KS_Q_DEFAULT) == (1.5, 5, 2, 40.0)


@needs_direct_solver
def test_skipped_passes_return_the_last_value_and_a_zero_gradient(tmp_path):
    """Hand-made points with pass numbers. F is far below the lowest factor, so the row skips: passes 0 and 1 evaluate (known after two
    evaluations), 2 to 5 skip, 6 evaluates (`every`: 5 passes since pass 1), a new beta evaluates (`stage_start`), the last pass evaluates."""
    m = loop_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    F = 1e-3 * dense_lowest(m, rho)
    row = br.BucklingRow(0, F, 40.0, 6, max_seconds=None)
    seen = []
    last_value = None
    for it, beta, last in [(0, 1.0, False), (1, 1.0, False), (2, 1.0, False), (3, 1.0, False), (5, 1.0, False), (6, 1.0, False),
                           (7, 2.0, False), (8, 2.0, False), (9, 2.0, True)]:
        pt = point_at(m, rho, beta, it, last)
        resp = row(pt)
        seen.append((it, resp.info["evaluated"], resp.info["reason"]))
        if resp.info["evaluated"]:
            last_value = resp.value
            assert resp.grad.any()
        else:
            assert resp.value == last_value and not resp.grad.any()
            assert resp.info["status"] == br.SKIPPED and resp.info["solves"] == 0 and resp.info["residual_adjoint"] is None
            assert resp.value - 1.0 <= 1.0 / br.SKIP_FACTOR - 1.0 + math.log(6) / 40.0      # the bound of the plan (R2)
    assert seen == [(0, True, "unknown"), (1, True, "unknown"), (2, False, "skipped"), (3, False, "skipped"), (5, False, "skipped"),
                    (6, True, "every"), (7, True, "stage_start"), (8, False, "skipped"), (9, True, "last_pass")], seen
    st = row.get_state()
    assert st["n_pass"] == 9 and st["n_eval"] == 5 and st["n_fail"] == 0
    m.solver.free()


@needs_direct_solver
def test_skip_rule_in_the_loop_on_the_column(tmp_path):
    """The rule in `minimize` on the 3 x 3 x 30 column (60 passes, beta stages at 18, 30, 42; 1 thread: the run repeats).

    F comes from a probe run without the row: 1.5 F is the middle of the start factor and the lowest factor of the trajectory (start 0.1735,
    lowest 0.0739 at pass 10, so F = 0.0825). The row is inactive at first, then near the limit when the column is thin.
    Asserted: passes 0 and 1 evaluate; some early passes skip; every stage start evaluates; the last pass evaluates; every pass after an
    evaluation with a lowest factor of at most 1.5 F evaluates; a skipped entry has the value and the factors of the last evaluation and
    `g <= 1/SKIP_FACTOR - 1 + ln(modes)/ks_q` (-0.2885, the R2 bound); no failure; the entries have exactly `INFO_KEYS` and are JSON without nan.
    """
    from topopt.optimize import minimize
    m = loop_model(tmp_path)
    lam = no_row_trajectory(m)
    F = (lam[0] + min(lam)) / 3.0
    assert lam[0] > 1.5 * F > min(lam)
    row = br.buckling_ks_constraint(0, F)
    res = minimize(m, volume_form(m, row))
    entries = [h["constraints"][1] for h in res.history]
    betas = [h["beta"] for h in res.history]
    bound = 1.0 / br.SKIP_FACTOR - 1.0 + math.log(6) / 40.0
    prev, near_passes, skipped = None, 0, 0
    for i, e in enumerate(entries):
        assert all(k in e for k in br.INFO_KEYS)
        json.dumps(e, allow_nan=False)
        assert e["status"] in (bk.OK, br.SKIPPED) and e["n_fail"] == 0
        if i in (0, 1):
            assert e["evaluated"]
        if i == len(entries) - 1:
            assert e["evaluated"] and e["reason"] == "last_pass"
        if i and betas[i] != betas[i - 1]:
            assert e["evaluated"], f"stage start at pass {i} was skipped"
        if prev is not None and prev["lowest_factor"] is not None and prev["lowest_factor"] <= 1.5 * F and i > 1:
            assert e["evaluated"], f"pass {i} follows a lowest factor of {prev['lowest_factor']:.4g} (1.5 F = {1.5 * F:.4g}) and was skipped"
            near_passes += 1
        if not e["evaluated"]:
            skipped += 1
            assert e["g"] <= bound + 1e-12 and e["value"] == prev["value"] and e["lowest_factor"] == prev["lowest_factor"]
            assert e["load_factors"] == prev["load_factors"] and e["last_eval"] == prev["last_eval"]
        prev = e
    assert skipped >= 3 and near_passes >= 3                                 # both regimes happened
    assert entries[-1]["n_eval"] == sum(1 for e in entries if e["evaluated"]) and entries[-1]["n_pass"] == len(entries)
    m.solver.free()


@needs_direct_solver
def test_the_active_row_holds_the_factor(tmp_path):
    """The row is active: run A has no row, run B has `min_factor` = 3 x the lowest factor of the final design of A (dense probe).

    The 3 x 3 x 30 column, 60 passes, 1 thread. Measured (this branch): A ends at 0.12148; B (F = 0.36445) ends at 0.36435 (the dense probe),
    ratio 0.9997, row g +2.8e-4 (the end of an active run moves by about 1e-3 from run to run and with the thread count: +-2.5e-4 and a ratio
    of 1.0003 in other runs), all 61 passes evaluated (the row is near its limit all the time). The plan said 1.5 x; the
    compliance-optimal design of A is a poor column (the density of A is far from B: max |dx| 1.0), and 1.5 x is not active
    at the end (the lowest factor of B ends at 1.6 F, g -0.37). With 2 x the end is 1.007 F, with 4 x 1.007 F, with 3 x 1.0003 F.
    Asserted: the last row g is at most `NOT_MET`; the dense lowest factor of the final density is at least F / (1 + NOT_MET) and at most 1.05 F
    (the row is active, not slack); the row's own `lowest_factor` at the last pass agrees with the dense value to 1e-6 (ARPACK tol 1e-8);
    the density of B differs from A.
    """
    from topopt.optimize import minimize
    m = loop_model(tmp_path)
    resA = minimize(m, volume_form(m))
    lamA = dense_lowest(m, final_density(m, resA))
    F = 3.0 * lamA
    resB = minimize(m, volume_form(m, br.buckling_ks_constraint(0, F)))
    e = resB.history[-1]["constraints"][1]
    lamB = dense_lowest(m, final_density(m, resB))
    print("T7", lamA, F, lamB, lamB / F, e["g"])
    assert e["g"] <= NOT_MET and e["evaluated"]
    assert F / (1.0 + NOT_MET) <= lamB <= 1.05 * F
    assert abs(e["lowest_factor"] - lamB) / lamB < 1e-6
    assert float(np.abs(resB.x - resA.x).max()) > 0.1
    assert not [line for line in resB.unmet if line.startswith("constraint not met")]       # the facts of the row are lines of `unmet` too
    assert any(line.startswith("fact: buckling row buckling:0") for line in resB.unmet)
    m.solver.free()


# --- T8: no usable result -------------------------------------------------------


def fake_modes(status, n_modes=0, mu=None):
    """A stand-in for `buckling_modes`: a result of the shape of the real one, with `status` and optional `mu`."""
    def fake(Kg, K, fac, k=6, **kw):
        n = K.shape[0]
        m = 0 if mu is None else len(mu)
        arr = np.zeros(0) if mu is None else np.asarray(mu, float)
        lam = np.full(m, np.inf)
        phi = np.random.default_rng(0).standard_normal((n, m))
        return {"status": status, "mu": arr, "lam": lam, "phi": phi, "residual": np.zeros(m), "solves": 7, "seconds": 0.01,
                "converged": m, "k": k}
    return fake


@needs_direct_solver
@pytest.mark.parametrize("status", (bk.NOT_CONVERGED, bk.TIME_BUDGET))
def test_a_pass_with_no_eigen_result_keeps_the_last_value_and_asks_again(tmp_path, monkeypatch, status):
    """`not_converged` and `time_budget` (a faked `buckling_modes`): the first pass gives value 0 (g -1) and a zero gradient;
    the row is unknown again, so the next two passes evaluate (a real result), `n_fail` counts it, and the log gets one line."""
    m = loop_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    lines = []
    row = br.BucklingRow(0, 1e-3, 40.0, 6, max_seconds=None, log=lines.append)
    with monkeypatch.context() as mp:
        mp.setattr(bk, "buckling_modes", fake_modes(status))
        r0 = row(point_at(m, rho, 1.0, 0))
    assert r0.value == 0.0 and not r0.grad.any() and r0.info["status"] == status and not r0.info["evaluated"]
    assert row.get_state()["n_fail"] == 1 and row.get_state()["n_known"] == 0
    assert len(lines) == 1 and status in lines[0]
    if status == bk.TIME_BUDGET:                                           # the options go with the status (A6b fix round 1, m1)
        assert "options: raise buckling.max_seconds" in lines[0] and "may have no compression" not in lines[0]
    else:
        assert "The case may have no compression" in lines[0] and "options: raise buckling.max_seconds" not in lines[0]
    r1, r2, r3 = (row(point_at(m, rho, 1.0, it)) for it in (1, 2, 3))
    assert [r.info["evaluated"] for r in (r1, r2, r3)] == [True, True, False]
    assert [r.info["reason"] for r in (r1, r2, r3)] == ["unknown", "unknown", "skipped"]
    assert r1.info["n_fail"] == 1 and r3.value == r2.value and not r3.grad.any()
    json.dumps(r0.info, allow_nan=False)
    assert tuple(r0.info) == br.INFO_KEYS == tuple(r1.info)
    assert len(lines) == 1
    m.solver.free()


@needs_direct_solver
def test_no_positive_factor_is_a_result(tmp_path, monkeypatch):
    """All `mu` at or below 0 (faked: a case with no compression): status `no_positive_factor`, the KS value is at most ln(k)/q (g about -0.95
    for F = 1... here F is tiny, so the value is near 0), the gradient is finite, the row counts it as known and goes to skip mode,
    and the entry is JSON (no inf: `lowest_factor` is null)."""
    m = loop_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None)
    with monkeypatch.context() as mp:
        mp.setattr(bk, "buckling_modes", fake_modes(bk.OK, mu=[-1e-3 * (j + 1) for j in range(7)]))
        r0 = row(point_at(m, rho, 1.0, 0))
        r1 = row(point_at(m, rho, 1.0, 1))
    for r in (r0, r1):
        assert r.info["status"] == br.bk.NO_POSITIVE and r.info["evaluated"] and r.info["lowest_factor"] is None
        assert r.info["load_factors"] == [] and np.all(np.isfinite(r.grad))
        assert -1e-3 <= r.value <= math.log(6) / 40.0 + 1e-15
        json.dumps(r.info, allow_nan=False)
    assert row.get_state()["n_known"] == 2
    r2 = row(point_at(m, rho, 1.0, 2))
    assert not r2.info["evaluated"] and r2.info["lowest_factor"] is None            # no positive factor: skip mode (the cheap one)
    m.solver.free()


@needs_direct_solver
def test_a_tension_load_is_not_an_error(tmp_path):
    """A column in tension (a real ARPACK run, 2 x 3 x 30 voxels). Compression needs 25 solves; tension has only tiny positive `mu` (the
    lateral contraction, 3.8e-4 and below) and ARPACK needs many more: 666 solves for density 1 (status ok, lowest factor 2.7e3), and for
    a random gray density it does not converge in 100 restarts (1388 solves, `not_converged`). Both are results of the row, not errors:
    ok with a finite gradient, or no usable result with a zero gradient and `n_fail` 1."""
    m = column_model(tmp_path / "t", load=-1000.0, lateral=0.0)
    ones = np.full(m.n_design, 1.0)
    resp = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)(point_at(m, ones))
    assert resp.info["status"] == bk.OK and resp.info["evaluated"] and np.all(np.isfinite(resp.grad)) and resp.info["lowest_factor"] > 1e3
    gray = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    resp = row(point_at(m, gray))
    assert resp.info["status"] in (bk.OK, bk.NO_POSITIVE, bk.NOT_CONVERGED) and np.all(np.isfinite(resp.grad))
    assert resp.info["evaluated"] == (resp.info["status"] != bk.NOT_CONVERGED) and (resp.info["evaluated"] or not resp.grad.any())
    m.solver.free()


# --- T14: a stress row and the buckling row together -----------------------------


@needs_direct_solver
def test_a_stress_row_and_the_buckling_row_together(tmp_path):
    """Step acceptance on (a stress row), compliance + volume + stress `stress:axial` (700 MPa) + `buckling:0`, 60 passes, 1 thread.

    F = 1.3 x the lowest factor of the final design of a run with the stress row only (0.0994, dense probe), so F = 0.1292.
    Measured: the run finishes; the buckling row is evaluated at every pass (61 of 61), `n_fail` 0, the adjoint residual at most 2.4e-15;
    step acceptance rejected tries (20 fact lines). End g of the rows: volume +0.0108, stress row +0.0013, buckling row -1.5e-5 (the stress-only
    run ends at -0.18). The combination needs more passes than each row alone: the plan review measured the stress row at +0.26 (60 passes)
    and +0.0128 (120 passes) on a 6 x 6 x 60 column (CONTRACTS section 39, limits). The test asserts the rows are finite, the buckling row
    is met at the end, and the factor check holds at every evaluated pass; it does not assert that the stress row is met.
    """
    from topopt.ks_stress import stress_ks_constraint
    from topopt.optimize import minimize
    m = loop_model(tmp_path)
    stress = lambda: stress_ks_constraint(0, 700.0, name="stress:axial")
    resS = minimize(m, volume_form(m, stress()))
    F = 1.3 * dense_lowest(m, final_density(m, resS))
    res = minimize(m, volume_form(m, stress(), br.buckling_ks_constraint(0, F)))
    entries = [h["constraints"][2] for h in res.history]
    assert all(e["name"] == "buckling:0" for e in entries) and len(entries) == 61
    assert entries[-1]["n_fail"] == 0 and entries[-1]["n_eval"] == 61
    assert all(e["residual_adjoint"] <= bk.RESIDUAL_FACT for e in entries if e["evaluated"])
    ends = {c["name"]: c["g"] for c in res.constraints}
    print("T14 end g", ends)
    assert all(np.isfinite(v) for v in ends.values()) and ends["buckling:0"] <= NOT_MET
    assert res.step_facts                                       # step acceptance worked on the stress row
    assert F / (1.0 + NOT_MET) <= dense_lowest(m, final_density(m, res))
    m.solver.free()


# --- T16: springs and a mirror plane -------------------------------------------------


def chain_fd_worst(m, beta, case=0):
    """The T1 check (through `Formulation.evaluate`, best of three steps, 10 voxels) on any model. Returns the worst error and the entry."""
    def form(F):
        return Formulation(weighted_compliance(), [volume_constraint(m.n_design, 0.5), br.buckling_ks_constraint(case, F)])
    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 0.7, m.n_design)
    F = evaluate(m, form(1.0), x, beta).constraints[1]["lowest_factor"]
    f = form(F)
    ev = evaluate(m, f, x, beta)
    floor = 1e-3 * np.abs(ev.dg[1]).max()
    worst = 0.0
    for e in rng.choice(m.n_design, size=10, replace=False):
        best = min(abs((evaluate(m, f, np.where(np.arange(m.n_design) == e, x + eps, x), beta).g[1]
                        - evaluate(m, f, np.where(np.arange(m.n_design) == e, x - eps, x), beta).g[1]) / (2 * eps) - ev.dg[1, e])
                   / max(abs(ev.dg[1, e]), floor) for eps in STEPS)
        worst = max(worst, best)
    return worst, ev.constraints[1]


T16_BAR = 1e-3      # 30 x the worst measured 1.9e-5 = 5.7e-4, rounded up (the column has a larger step truncation than the cantilever)


@needs_direct_solver
@pytest.mark.parametrize("beta", FD_BETAS)
def test_fd_with_a_spring_support(tmp_path, beta):
    """A7a: the column on a spring pad (stiffness 1e5 on all axes, no clamp), lateral 50 N. K has Kb, Kg has none.

    Measured worst best-of-three: 7.2e-6 (beta 1) and 1.2e-5 (beta 4) with 1e5; 1.9e-5 and 5.8e-6 with 1e6. Bar `T16_BAR`.
    """
    m = Model(build_problem(load_config(column_yaml(tmp_path, lateral=50.0, spring=1e5))))
    assert m.kb is not None
    worst, entry = chain_fd_worst(m, beta)
    assert entry["status"] == bk.OK and entry["residual_adjoint"] <= bk.RESIDUAL_FACT and worst < T16_BAR, worst
    m.solver.free()


@needs_direct_solver
@pytest.mark.parametrize("beta", FD_BETAS)
def test_fd_with_a_mirror_plane(tmp_path, beta):
    """A4a: the 2 x 3 x 30 column with the mirror plane x = 1 mm (the design variables are averaged), axial load. Measured worst
    best-of-three: 7.6e-6 (beta 1) and 1.1e-5 (beta 4). Bar `T16_BAR`."""
    m = Model(build_problem(load_config(column_yaml(tmp_path, lateral=0.0, mirror=[{"normal": "x", "at_mm": 1.0}]))))
    worst, entry = chain_fd_worst(m, beta)
    assert entry["status"] == bk.OK and entry["residual_adjoint"] <= bk.RESIDUAL_FACT and worst < T16_BAR, worst
    m.solver.free()


# --- W2: the warm start of the eigen solve, and the row tolerance (#155) --------------------------------------------------

WARM_BAR = 1e-8     # 30 x the worst measured difference to a cold solve, rounded up to a power of ten (5.1e-11 in the factors, 2.1e-10 in the gradient)
WARM_STEP = 0.002   # the second design of a warm test: the first design plus this times a normal draw (25 solves warm against 36 cold, up to 0.005)


class ModesSpy:
    """A stand-in for `bk.buckling_modes` that records each call (start vector, tolerance, matrix, result).

    The call numbers in `fail` give a faked failure (`status`, no result); the others are real."""

    def __init__(self, monkeypatch, fail=(), status=bk.NOT_CONVERGED):
        self.real, self.fail, self.status, self.calls = bk.buckling_modes, set(fail), status, []
        monkeypatch.setattr(bk, "buckling_modes", self)

    def __call__(self, Kg, K, fac, k=6, **kw):
        n = len(self.calls)
        res = fake_modes(self.status)(Kg, K, fac, k=k, **kw) if n in self.fail else self.real(Kg, K, fac, k=k, **kw)
        v0 = kw.get("v0")
        self.calls.append(dict(v0=None if v0 is None else np.array(v0), tol=kw.get("tol"), K=K, res=res))
        return res

    @property
    def v0s(self):
        return [c["v0"] for c in self.calls]


def row_at(m, row, rho, iteration):
    return row(point_at(m, rho, 1.0, iteration))


def sum_of_k_normalised(res, K):
    """The start vector of the next solve, from the module text: the sum over all returned columns of `c / sqrt(c^T K c)`."""
    out = np.zeros(K.shape[0])
    for j in range(res["phi"].shape[1]):
        c = res["phi"][:, j]
        out += c / math.sqrt(float(c @ (K @ c)))
    return out


@needs_direct_solver
def test_the_warm_start_gives_the_same_factors_the_value_and_the_gradient_with_fewer_solves(tmp_path, monkeypatch):
    """Evaluation 1 of a row starts cold; evaluation 2 gets the sum of the K-normalised returned columns of evaluation 1 (all `modes + 1`, computed here
    with an own sum). A cold row at the same second design is the reference. Measured on the 3 x 3 x 30 column (1536 free dofs, so ARPACK runs: `ncv`
    24 < n; the dense path ignores `v0`): 25 solves warm against 36 cold; factors, value and gradient agree to `WARM_BAR`."""
    spy = ModesSpy(monkeypatch)
    m = loop_model(tmp_path)
    assert max(2 * 7 + 1, bk.NCV_MIN) < m.n_free
    rng = np.random.default_rng(5)
    rho0 = rng.uniform(0.3, 0.7, m.n_design)
    rho1 = np.clip(rho0 + WARM_STEP * rng.standard_normal(m.n_design), 0.05, 1.0)
    warm = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    first, second = row_at(m, warm, rho0, 0), row_at(m, warm, rho1, 1)
    cold = row_at(m, br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False), rho1, 1)
    assert spy.v0s[0] is None and spy.v0s[2] is None                                 # the first evaluation of a row is cold
    assert spy.calls[0]["res"]["phi"].shape[1] == 7                                  # modes + 1 columns
    np.testing.assert_allclose(spy.v0s[1], sum_of_k_normalised(spy.calls[0]["res"], spy.calls[0]["K"]), rtol=1e-12, atol=0)
    assert second.info["solves"] < cold.info["solves"] == first.info["solves"]
    for r in (first, second, cold):
        assert r.info["status"] == bk.OK and r.info["residual_adjoint"] <= bk.RESIDUAL_FACT
    got, want = np.array(second.info["load_factors"]), np.array(cold.info["load_factors"])
    assert got.size == want.size == 6 and np.max(np.abs(got - want) / want) < WARM_BAR
    assert abs(second.value - cold.value) < WARM_BAR * abs(cold.value)
    assert np.abs(second.grad - cold.grad).max() < WARM_BAR * np.abs(cold.grad).max()
    m.solver.free()


@needs_direct_solver
def test_a_skipped_pass_keeps_the_start_vector_and_the_next_evaluation_uses_it(tmp_path, monkeypatch):
    spy = ModesSpy(monkeypatch)
    m = loop_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1e-3, 40.0, 6, max_seconds=None)                         # far from the limit: passes 2 to 4 are skipped
    infos = [row_at(m, row, rho, it).info for it in range(7)]
    assert [i["evaluated"] for i in infos] == [True, True, False, False, False, False, True]
    assert [v is None for v in spy.v0s] == [True, False, False]
    kept = row.get_state()["v0"]
    np.testing.assert_array_equal(spy.v0s[2], sum_of_k_normalised(spy.calls[1]["res"], spy.calls[1]["K"]))
    assert kept.shape == (m.n_free,) and kept.dtype == np.float64
    m.solver.free()


@needs_direct_solver
def test_the_start_is_cold_after_a_failed_first_solve_and_the_last_good_vector_stays_after_a_later_failure(tmp_path, monkeypatch):
    """Calls 0 and 3 fail. Call 1 is cold (nothing was kept after the failure of call 0); call 2 gets the vector of call 1; call 3 gets the vector
    of call 2 and fails; call 4 gets the vector of call 2 again (the same bytes: a failure keeps the last good vector)."""
    spy = ModesSpy(monkeypatch, fail={0, 3})
    m = loop_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    infos = [row_at(m, row, rho, it).info for it in range(5)]
    assert [i["evaluated"] for i in infos] == [False, True, True, False, True]
    assert spy.v0s[0] is None and spy.v0s[1] is None
    v1 = sum_of_k_normalised(spy.calls[1]["res"], spy.calls[1]["K"])
    v2 = sum_of_k_normalised(spy.calls[2]["res"], spy.calls[2]["K"])
    np.testing.assert_allclose(spy.v0s[2], v1, rtol=1e-12, atol=0)
    np.testing.assert_allclose(spy.v0s[3], v2, rtol=1e-12, atol=0)                   # call 2 had a result: its vector replaced the old one
    np.testing.assert_array_equal(spy.v0s[4], spy.v0s[3])                            # call 3 failed: the same vector
    m.solver.free()


@needs_direct_solver
def test_a_factor_mismatch_does_not_replace_the_start_vector(tmp_path, monkeypatch):
    """A pass whose factor is the factor of another density has `factor_mismatch`: the eigenvectors came from a wrong inverse, so they are not kept."""
    spy = ModesSpy(monkeypatch)
    m = loop_model(tmp_path, lateral=50.0)
    rng = np.random.default_rng(5)
    rho1, rho2 = rng.uniform(0.3, 0.7, m.n_design), rng.uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    assert row_at(m, row, rho1, 0).info["status"] == bk.OK
    good = row.get_state()["v0"].copy()
    pt = point_at(m, rho1, 1.0, 1)
    m.solver.factor(m.stiffness(rho2))
    assert row(pt).info["status"] == br.FACTOR_MISMATCH
    np.testing.assert_array_equal(row.get_state()["v0"], good)
    assert spy.v0s[1] is not None                                                  # the mismatch pass itself was warm: it had the vector of pass 0
    m.solver.free()


@needs_direct_solver
@pytest.mark.parametrize("status", [bk.NOT_CONVERGED, bk.TIME_BUDGET])
@pytest.mark.parametrize("wrong", [True, False])
def test_a_failed_eigen_solve_with_a_wrong_factor_is_a_factor_mismatch(tmp_path, monkeypatch, status, wrong):
    """A wrong factor can stop the eigen solve before the adjoint residual shows it (the 3.13 CI run of 2026-10-26 gave `not_converged`).
    One solve with the factor then tells: a wrong factor is `factor_mismatch`; with the right factor the status of the eigen solve stays."""
    ModesSpy(monkeypatch, fail={1}, status=status)
    m = loop_model(tmp_path, lateral=50.0)
    rng = np.random.default_rng(5)
    rho1, rho2 = rng.uniform(0.3, 0.7, m.n_design), rng.uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    assert row_at(m, row, rho1, 0).info["status"] == bk.OK
    good = row.get_state()["v0"].copy()
    pt = point_at(m, rho1, 1.0, 1)
    if wrong:
        m.solver.factor(m.stiffness(rho2))
    info = row(pt).info
    assert info["status"] == (br.FACTOR_MISMATCH if wrong else status) and not info["evaluated"]
    assert (info["residual_adjoint"] > bk.RESIDUAL_FACT) == wrong and info["adjoint_seconds"] > 0
    assert row.get_state()["fail_status"] == info["status"]
    np.testing.assert_array_equal(row.get_state()["v0"], good)
    m.solver.free()


@needs_direct_solver
def test_the_check_after_a_failed_eigen_solve_passes_a_near_singular_design_with_the_right_factor(tmp_path, monkeypatch):
    """A column with void layers is valid but near-singular: the residual of the load (`residual_u`) is above RESIDUAL_FACT there. The check
    solves K x = K v (a fixed random v), so the right factor stays `not_converged`; a check with the load would say `factor_mismatch`."""
    ModesSpy(monkeypatch, fail={0})
    m = column_model(tmp_path, lateral=50.0, length=60)
    ctr = m.design_centers()
    rho = np.full(m.n_design, 0.5)
    for z in range(2, 58, 2):
        rho[(ctr[:, 2] > z) & (ctr[:, 2] < z + 1)] = 0.0
    lines = []
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False, log=lines.append)
    info = row(point_at(m, rho, 1.0, 0)).info
    assert info["residual_u"] > bk.RESIDUAL_FACT                                   # the load would not do as the right-hand side
    assert info["status"] == bk.NOT_CONVERGED and info["residual_adjoint"] <= bk.RESIDUAL_FACT
    assert len(lines) == 1 and br.FAULT not in lines[0] and "no compression" in lines[0]
    m.solver.free()


@needs_direct_solver
def test_the_fact_of_a_mismatch_found_by_the_check_names_the_check_solve(tmp_path, monkeypatch):
    ModesSpy(monkeypatch, fail={0})
    m = loop_model(tmp_path, lateral=50.0)
    rng = np.random.default_rng(5)
    rho1, rho2 = rng.uniform(0.3, 0.7, m.n_design), rng.uniform(0.3, 0.7, m.n_design)
    pt = point_at(m, rho1, 1.0, 0)
    m.solver.factor(m.stiffness(rho2))
    lines = []
    info = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False, log=lines.append)(pt).info
    assert info["status"] == br.FACTOR_MISMATCH and len(lines) == 1
    assert "The residual of a check solve with the factor after the failed eigen solve is " in lines[0] and br.FAULT in lines[0]
    m.solver.free()


@needs_direct_solver
@pytest.mark.parametrize("bad", ["short", "long", "nan", "inf", "zero", "matrix"])
def test_a_kept_vector_that_does_not_fit_is_dropped_and_the_solve_starts_cold(tmp_path, monkeypatch, bad):
    spy = ModesSpy(monkeypatch)
    m = loop_model(tmp_path)
    n = m.n_free                                                                    # also the size of K (with joints: of the condensed S)
    vectors = {"short": np.ones(n - 1), "long": np.ones(n + 1), "nan": np.where(np.arange(n) == 3, np.nan, 1.0),
               "inf": np.where(np.arange(n) == 3, np.inf, 1.0), "zero": np.zeros(n), "matrix": np.ones((n, 2))}
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    row.set_state({**row.get_state(), "v0": vectors[bad]})
    resp = row_at(m, row, rho, 0)
    assert spy.v0s == [None] and resp.info["status"] == bk.OK
    kept = row.get_state()["v0"]
    assert kept.shape == (n,) and np.all(np.isfinite(kept)) and kept.any()          # replaced by the vector of this result
    m.solver.free()


def test_a_state_of_other_settings_drops_the_start_vector_and_a_state_of_the_same_settings_keeps_it(monkeypatch):
    def row(F=1.0, q=40.0, modes=6):
        return br.BucklingRow(0, F, q, modes, max_seconds=None)

    vector = np.arange(1.0, 6.0)
    saved = {**row().get_state(), "v0": vector, "n_known": 2, "last_lowest": 3.0}
    same = row()
    same.set_state(saved)
    assert same.get_state()["v0"] is vector and same.get_state()["n_known"] == 2
    for other in (row(F=1.5), row(q=20.0), row(modes=5)):                            # min_factor, ks_q, modes
        other.set_state(saved)
        st = other.get_state()
        assert st["v0"] is None and st["n_known"] == 0
        assert (st["min_factor"], st["ks_q"], st["modes"]) == (other.min_factor, other.q, other.modes)
    monkeypatch.setattr(br, "ROW_TOL", 1e-6)                                         # a later change of the tolerance (N1) is one line
    looser = row()
    looser.set_state(saved)
    st = looser.get_state()
    assert st["v0"] is None and st["n_known"] == 0 and st["tol"] == 1e-6
    again = row()
    again.set_state(st)                                                              # the same tolerance now: nothing to drop
    assert again.get_state()["n_known"] == 0


def test_a_state_without_a_vector_or_a_tolerance_loads_as_it_was_solved_at_the_old_tolerance(monkeypatch):
    """A state that topopt wrote before #155 has neither key. It loads; its solves ran at `bk.TOL`, so it is the same setting while `ROW_TOL` is `bk.TOL`
    and another one after a change of `ROW_TOL`."""
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None)
    old = {k: v for k, v in {**row.get_state(), "n_known": 2, "last_lowest": 3.0}.items() if k not in ("v0", "tol")}
    row.set_state(old)
    st = row.get_state()
    assert st["v0"] is None and st["tol"] == bk.TOL and st["n_known"] == 2
    monkeypatch.setattr(br, "ROW_TOL", 1e-6)
    later = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None)
    later.set_state(old)
    assert later.get_state()["n_known"] == 0 and later.get_state()["tol"] == 1e-6


@needs_direct_solver
def test_the_row_tolerance_comes_from_one_constant(tmp_path, monkeypatch):
    """`ROW_TOL` is `bk.TOL` today (no result changes). The call of the eigen solve and the state read it; changing the constant is the only change that N1 needs."""
    assert br.ROW_TOL == bk.TOL == 1e-8 and (bk.MAXITER, bk.NCV_MIN) == (100, 24)
    spy = ModesSpy(monkeypatch)
    m = loop_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    row = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    row_at(m, row, rho, 0)
    assert spy.calls[0]["tol"] == bk.TOL and row.get_state()["tol"] == bk.TOL
    monkeypatch.setattr(br, "ROW_TOL", 1e-6)
    loose = br.BucklingRow(0, 1.0, 40.0, 6, max_seconds=None, skip=False)
    row_at(m, loose, rho, 0)
    assert spy.calls[1]["tol"] == 1e-6 and loose.get_state()["tol"] == 1e-6 and spy.calls[0]["tol"] == bk.TOL
    m.solver.free()
