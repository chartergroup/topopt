"""P2b: springs (item 2) and joints (item 3) on gpu-mg. `gpu_mg/border.py` puts `Kb` of the springs and the bonds, and the rank-6m term
`- C D^-1 C^T` of the joints, into every multigrid level.

The spring cases are the pad case of `tests/test_springs.py` (a spring pad at the free end of a beam, 8,121 free dofs at voxel size 1). The
reference is the direct solve (`scipy.sparse.linalg.spsolve`) of the reduced `K + Kb` that the model of the other solvers builds. The joint
cases are the cube of `tools/joint_check.py` (a stiff cube held only by a beam to a fixed point: no fixed region) and the pad case of
`tests/test_joints.py`; the reference is the bordered system `[[K + Kb, C], [C^T, D]]` solved by `splu`. The tests of the host hierarchy
need no OpenCL; the other tests run the real engine on the OpenCL device (pocl on the host of the tests).
"""
import json
import math
import warnings
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, splu

from test_gpu_mg import need_case
from test_gpu_mg_x_merge import fake_cpu_device  # noqa: F401  (a fixture)
from test_joints import jc
from test_joints import case as joint_case
from test_joints import prob_of as joint_prob
from test_springs import box, case, prob_of
from topopt import cli
from topopt import springs as sp
from topopt.cli import main
from topopt.config import load_config
from topopt.fe import edof_table
from topopt.gpu_mg import kernels
from topopt.gpu_mg.border import grid_border
from topopt.gpu_mg.hierarchy import HostMG
from topopt.optimize import Model, reduced_system
from topopt.problem import build_problem
from topopt.voxelize import LOAD

opencl = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1

VARIANTS = {"clamp": dict(clamp=True, k=(100.0, 100.0, 100.0)),
            "floating": dict(clamp=False, k=(100.0, 100.0, 100.0)),       # the springs are the only support
            "stiff": dict(clamp=True, k=(1e5, 1e5, 1e5))}


def without_springs(d):
    d.pop("springs")


def fresh(tmp_path, name):
    """A new empty folder `name` under `tmp_path` (a case writes its STL files there)."""
    d = tmp_path / name
    d.mkdir()
    return d


def reference(tmp_path, variant, h=1.0, seed=3):
    """`(model, rho, K, U)`: the amg model of the case (it holds `K + Kb`), a random design, the reduced matrix and its direct solve."""
    m = Model(prob_of(fresh(tmp_path, "ref"), h=h, solver="amg", **VARIANTS[variant]))
    rho = 0.1 + 0.8 * np.random.default_rng(seed).uniform(size=m.n_design)
    K = m.stiffness(rho).tocsc()
    U = spsolve(K, m.f_red)
    m.solver.free()
    return m, rho, K, U.reshape(m.n_free, -1)


def host_of(prob, border=True):
    edof, aidx, free = reduced_system(prob)
    border = grid_border(prob) if border else None
    return HostMG(prob.shape, edof, aidx, prob.fixed_dofs, prob.h, prob.config.material.nu, min_free=300, border=border), free


# --- part 1: the host hierarchy, no OpenCL --------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_host_operator_is_K_plus_Kb_and_the_diagonal_blocks_and_coarse_inverse_hold_the_springs(tmp_path, variant):
    m, rho, K, _ = reference(tmp_path, variant)
    prob = m.problem
    host, free = host_of(prob)
    host.setup_host(m.element_E(rho))
    L0 = host.levels[0]
    fmap = L0.lookup[free // 3].astype(np.int64) * 3 + free % 3
    rng = np.random.default_rng(1)
    u = rng.standard_normal(free.size)
    x = np.zeros(3 * L0.nn)
    x[fmap] = u
    y = np.zeros((L0.nn, 3))
    host.apply_fine(x.reshape(-1, 3), y)
    ref = K @ u
    assert np.linalg.norm(y.reshape(-1)[fmap] - ref) <= 1e-13 * np.linalg.norm(ref)          # the FP64 residual of the solve is K + Kb
    host0, _ = host_of(prob, border=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)         # the floating case: without the springs the coarsest matrix is singular
        host0.setup_host(m.element_E(rho))
    y0 = np.zeros((L0.nn, 3))
    host0.apply_fine(x.reshape(-1, 3), y0)
    assert np.linalg.norm(y0.reshape(-1)[fmap] - ref) > 1e-6 * np.linalg.norm(ref)         # without the border it is not (the test sees the springs)
    # |K| |x| + |Kb| |x|: the absolute operator of the backward-error rule
    ya, ya0 = np.zeros((L0.nn, 3)), np.zeros((L0.nn, 3))
    host.apply_fine_abs(x.reshape(-1, 3), ya)
    host0.apply_fine_abs(x.reshape(-1, 3), ya0)
    rows, cols, vals = prob.springs.triplets()
    kb = sps.csr_matrix((vals, (rows, cols)), shape=(3 * np.prod(np.array(prob.shape) + 1),) * 2)[free][:, free]
    assert np.allclose(ya.reshape(-1)[fmap], ya0.reshape(-1)[fmap] + abs(kb) @ np.abs(u), rtol=1e-13, atol=0)
    # the diagonal 3 x 3 blocks of the smoother of level 0: the inverse of the diagonal blocks of K + Kb
    place = -np.ones(3 * L0.nn, np.int64)
    place[fmap] = np.arange(free.size)
    checked = 0
    for n in host.border.levels[0].nodes:
        p = place[3 * n:3 * n + 3]
        if (p < 0).any():
            continue
        blk = K[p][:, p].toarray()
        d = L0.dinv[n]
        inv = np.array([[d[0], d[1], d[2]], [d[1], d[3], d[4]], [d[2], d[4], d[5]]])
        assert np.abs(inv @ blk - np.eye(3)).max() < 1e-10
        checked += 1
    assert checked > 20
    # the dense inverse of the coarsest level is the inverse of K_c + Kb_c
    C = host.levels[-1]
    free_c = np.flatnonzero(C.fix.reshape(-1) == 0)
    index = -np.ones(3 * C.nn, np.int64)
    index[free_c] = np.arange(free_c.size)
    Mc = host.cK.toarray()
    host.border.add_dense(host.n_levels - 1, Mc, index)
    assert np.abs(host.Ainv_full[np.ix_(free_c, free_c)] @ Mc - np.eye(free_c.size)).max() < 1e-9
    assert not np.allclose(Mc, host.cK.toarray())                                              # and the springs are in it


def test_the_galerkin_levels_keep_the_total_spring_stiffness_when_nothing_is_fixed(tmp_path):
    """`Kb_l = P^T Kb_{l-1} P` and the trilinear transfer keeps constants: the entries of `Kb_l` sum to the stiffness `k` of the spring on each
    axis, on every level, and a constant field of one axis has no coupling to the other axes."""
    prob = prob_of(tmp_path, solver="amg", **VARIANTS["floating"])
    host, _ = host_of(prob)
    assert host.n_levels >= 3 and prob.fixed_dofs.size == 0
    for l, lb in enumerate(host.border.levels):
        ones = np.ones(lb.Kb.shape[0])
        for d in range(3):
            v = np.zeros_like(ones)
            v[d::3] = 1.0
            assert v @ (lb.Kb @ v) == pytest.approx(100.0, rel=1e-12), (l, d)
            assert abs(v @ (lb.Kb @ np.roll(v, 1))) < 1e-9
        assert lb.n_nodes == len(lb.nodes) == len(np.unique(lb.nodes)) and lb.Kb.shape == (3 * lb.n_nodes,) * 2
        assert (lb.Kb != lb.Kb.T).nnz == 0 or abs(lb.Kb - lb.Kb.T).max() < 1e-12              # symmetric
    assert host.border.summary()["nodes"] == [lb.n_nodes for lb in host.border.levels]


def test_the_coarse_matrices_of_the_springs_are_zero_on_the_dirichlet_dofs_of_their_level(tmp_path):
    prob = prob_of(tmp_path, solver="amg", **VARIANTS["clamp"])
    host, _ = host_of(prob)
    for l, lb in enumerate(host.border.levels):
        fixed = host.levels[l].fix.reshape(-1)[lb.dofs].astype(bool)
        Kb = lb.Kb.tocsr()
        assert abs(Kb[fixed]).sum() == 0 and abs(Kb[:, fixed]).sum() == 0


def test_a_problem_without_springs_has_no_border_and_the_old_program(tmp_path):
    prob = prob_of(tmp_path, solver="amg", **VARIANTS["clamp"], mutate=without_springs)
    assert grid_border(prob) is None and host_of(prob)[0].border is None
    for c in ("B", "C"):
        assert "#define BORDER" not in kernels.source(c, 2) and "#define BORDER" in kernels.source(c, 2, border=True)
        assert "bord_w_v" in kernels.source(c, 2, border=True) and "bord_w_v" not in kernels.source(c, 2)
    assert "bord_w_o" in kernels.source("B", 2, border=True) and "bord_w_o" not in kernels.source("C", 2, border=True)


def test_gpu_mg_takes_springs_in_check_and_the_old_refusal_text_is_gone(tmp_path, fake_cpu_device):
    d = cli.check_data(case(tmp_path, solver="gpu-mg"))
    assert d["solver"]["resolved"] == "gpu-mg"
    assert not [w for w in d["warnings"] if "cannot start here" in w]
    assert not hasattr(sp, "GPU_MG_REFUSAL") and not hasattr(sp, "GPU_MG_JOINT_REFUSAL") and not hasattr(sp, "gpu_mg_problem")


# --- part 2: the real engine on the OpenCL device ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dev():
    """The device that `auto` takes. The test is skipped when there is none, or without pyopencl."""
    pytest.importorskip("pyopencl")
    from topopt.config import UserError
    from topopt.gpu_mg import device
    try:
        return device.select("auto", "auto")[0]
    except UserError as e:
        pytest.skip(f"no usable OpenCL device: {e}")


def gpu_model(tmp_path, case_name, variant, tol=1e-9, h=1.0):
    from topopt.gpu_mg.model import GpuModel, GpuParams
    prob = prob_of(fresh(tmp_path, f"gpu{case_name}"), h=h, solver="gpu-mg", mutate=lambda d: d["settings"].update(gpu_case=case_name),
                   **VARIANTS[variant])
    return GpuModel(prob, None, GpuParams(tol=tol, min_free=300, backward_error=0.0))


@opencl
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_spring_case_on_gpu_mg_equals_the_direct_solve(tmp_path, dev, case_name, variant):
    """The displacement and the compliance of the spring case on the device equal the direct solve of `K + Kb` within the solve tolerance
    (relative residual 1e-9). `floating`: the springs are the only support. `stiff`: 1e5 N/mm on each axis."""
    need_case(case_name, dev)
    m_ref, rho, K, Uref = reference(tmp_path, variant)
    m = gpu_model(tmp_path, case_name, variant)
    try:
        assert m.solver.host.n_levels >= 3 and m.kb is not None and m.solver.host.border.k == 0
        U = m.solve(m.stiffness(rho))
        its = m.solver.records[-1]["iterations"][0]
    finally:
        m.solver.free()
    assert U.shape == Uref.shape
    assert np.linalg.norm(U - Uref) <= 1e-7 * np.linalg.norm(Uref)
    c, cref = float(m.f_red[:, 0] @ U[:, 0]), float(m_ref.f_red[:, 0] @ Uref[:, 0])
    assert c == pytest.approx(cref, rel=1e-8)
    assert its["accepted_by"] == "tol" and (its["pcg"][0] < 60 if case_name == "B" else max(its["inner"]) < 40)      # the multigrid works: tens of iterations


@opencl
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_the_reactions_of_a_gpu_mg_solve_match_the_direct_solver_and_balance(tmp_path, dev, case_name):
    need_case(case_name, dev)
    m_ref, rho, K, Uref = reference(tmp_path, "clamp")
    m = gpu_model(tmp_path, case_name, "clamp", tol=1e-10)
    try:
        U = m.solve(m.stiffness(rho))
        got = m.reaction_facts(U, rho)
    finally:
        m.solver.free()
    want = m_ref.reaction_facts(Uref, rho)
    g, w = got["cases"][0], want["cases"][0]
    for key in ("applied", "clamp"):
        assert np.allclose(g[key]["force_n"], w[key]["force_n"], rtol=1e-6, atol=1e-9)
        assert np.allclose(g[key]["moment_nmm"], w[key]["moment_nmm"], rtol=1e-6, atol=1e-8)
    assert np.allclose(g["springs"]["pad"]["force_n"], w["springs"]["pad"]["force_n"], rtol=1e-6, atol=1e-9)
    assert abs(np.array(g["springs"]["pad"]["force_n"])).max() > 1e-3                 # the spring carries a force: the test is not empty
    assert g["force_error"] < 1e-8 and g["moment_error"] < 1e-8 and g["residual_rel"] < 1e-9


@opencl
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_run_with_springs_finishes_on_gpu_mg_and_writes_the_reactions(tmp_path, dev, case_name):
    need_case(case_name, dev)
    p = case(tmp_path, solver="gpu-mg", mutate=lambda d: d["settings"].update(gpu_case=case_name, iterations=3))
    out = tmp_path / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert len(hist) >= 3 and hist[-1]["iteration"] == len(hist) - 1 and all(np.isfinite(r["compliance"]) for r in hist)
    r = hist[-1]["reactions"]["cases"][0]                                           # the last pass writes the reactions (section 35)
    assert abs(np.array(r["springs"]["pad"]["force_n"])).max() > 1e-3 and r["force_error"] < 1e-5 and r["residual_rel"] < 1e-5
    log = (out / "run.log").read_text()
    assert "reactions:" in log and "gpu-mg: device" in log and "spring term" not in log
    run = json.loads((out / "run.json").read_text())
    assert run["springs"]["regions"][0]["region"] == "pad"
    recs = [json.loads(line) for line in (out / "gpu_mg.jsonl").read_text().splitlines()]
    assert recs and all(rec["case"] == case_name for rec in recs)


@opencl
def test_the_engine_has_the_support_launches_only_with_a_border(tmp_path, dev):
    from topopt.gpu_mg.model import GpuModel, GpuParams
    need_case("C", dev)
    with_springs = gpu_model(tmp_path, "C", "clamp")
    def plain_c(d):
        without_springs(d)
        d["settings"]["gpu_case"] = "C"

    plain_prob = build_problem(load_config(case(fresh(tmp_path, "plain"), solver="gpu-mg", mutate=plain_c)))
    plain = GpuModel(plain_prob, None, GpuParams(min_free=300))
    try:
        e1, e0 = with_springs.solver.engine, plain.solver.engine
        names1 = [o.name for o in e1.iter_ops]
        names0 = [o.name for o in e0.iter_ops]
        assert "bord_w" not in names0 and e0.border is None and "border" not in e0.nbytes_parts
        # every application of the operator has a launch before it: 4 on each level but the coarsest, one for the outer product
        n_apply = 4 * (e1.nlev - 1) + 1
        assert names1.count("bord_w") == n_apply
        assert [n for n in names1 if n != "bord_w"] == names0                           # the other launches are the same, in the same order
        assert e1.nbytes_parts["border"] > 0 and with_springs.solver.facts()["border"]["springs"] == 1
        assert plain.solver.facts()["border"] is None
        assert with_springs.kb is not None and plain.kb is None
    finally:
        with_springs.solver.free()
        plain.solver.free()


# =================================================== item 3: joints ==============================================================


def cube_host(n=8, ratio=1.0, min_free=300, axis=(1, 0, 0), fixed=None):
    """`(cube, joint set, host hierarchy)` of the cube of `tools/joint_check.py`. `ratio` = E_block / E_beam (1e9: a very soft beam)."""
    c = jc.cube(n)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    jt = jc.make_joint(c, a * 20.0, ("round", 2.0), E_beam=c.E / ratio, k_a=100.0 * c.E / c.h)
    edof, active = edof_table(c.shape, c.labels == LOAD)
    fixed = np.zeros(0, np.int64) if fixed is None else fixed
    host = HostMG(c.shape, edof, active, fixed, c.h, c.nu, min_free=min_free, border=grid_border(SimpleNamespace(springs=None, joints=jt)))
    host.setup_host(np.full(edof.shape[0], c.E))
    return c, jt, host


def schur(c, jt):
    """The condensed operator of the cube on its free dofs: `K + Kb - C D^-1 C^T` (sparse K + Kb, dense low-rank term)."""
    K = jc.kb_added(c, jt)
    C, D = jt.border(c.free)
    return K, C, D


def test_the_host_operator_with_a_joint_is_the_condensed_bordered_matrix_and_the_absolute_operator_bounds_it():
    c, jt, host = cube_host()
    K, C, D = schur(c, jt)
    L0 = host.levels[0]
    fmap = L0.lookup[c.free // 3].astype(np.int64) * 3 + c.free % 3
    assert host.border.k == 6 and host.border.summary()["joints"] == 1 and host.n_levels >= 3 and L0.fix.sum() == 0
    u = np.random.default_rng(2).standard_normal(c.free.size)
    x = np.zeros(3 * L0.nn)
    x[fmap] = u
    y = np.zeros((L0.nn, 3))
    host.apply_fine(x.reshape(-1, 3), y)
    ref = K @ u - C @ np.linalg.solve(D.toarray(), C.T @ u)
    assert np.linalg.norm(y.reshape(-1)[fmap] - ref) <= 1e-12 * np.linalg.norm(ref)
    ya = np.zeros((L0.nn, 3))
    host.apply_fine_abs(x.reshape(-1, 3), ya)
    assert (ya.reshape(-1)[fmap] >= np.abs(ref) * (1 - 1e-12)).all()       # |K| |x| + |Kb| |x| + |C| |D^-1| |C^T| |x| bounds |A x| entry by entry


def test_the_diagonal_blocks_and_the_coarse_inverse_hold_the_joint_term():
    c, jt, host = cube_host(n=4, min_free=30)
    K, C, D = schur(c, jt)
    A = K.toarray() - C @ np.linalg.solve(D.toarray(), C.T.toarray())
    L0 = host.levels[0]
    fmap = L0.lookup[c.free // 3].astype(np.int64) * 3 + c.free % 3
    assert host.n_levels >= 2
    worst = 0.0
    for n in range(L0.nn):                    # every node: the bonded nodes carry the joint term, the others the element blocks
        p = np.flatnonzero(np.isin(fmap, [3 * n, 3 * n + 1, 3 * n + 2]))
        d = L0.dinv[n]
        inv = np.array([[d[0], d[1], d[2]], [d[1], d[3], d[4]], [d[2], d[4], d[5]]])
        worst = max(worst, np.abs(inv @ A[np.ix_(p, p)] - np.eye(3)).max())
    assert worst < 1e-9
    blk = host.border.diag_terms(0)[1]
    assert np.abs(blk).max() > 0 and np.abs(blk - blk.transpose(0, 2, 1)).max() < 1e-9 * np.abs(blk).max()     # a symmetric term on the bonded nodes
    Cs = host.levels[-1]
    free_c = np.flatnonzero(Cs.fix.reshape(-1) == 0)
    index = -np.ones(3 * Cs.nn, np.int64)
    index[free_c] = np.arange(free_c.size)
    Mc = host.cK.toarray()
    host.border.add_dense(host.n_levels - 1, Mc, index)
    assert np.abs(host.Ainv_full[np.ix_(free_c, free_c)] @ Mc - np.eye(free_c.size)).max() < 1e-6        # a floating cube: the matrix is stiff and the beam soft


def test_the_galerkin_levels_keep_the_rigid_motions_of_the_body_in_C_and_Kb():
    """`C_l = P^T C_{l-1}` and `Kb_l = P^T Kb_{l-1} P` with the trilinear transfer: a rigid motion is a linear field, so `C_l^T v_l = C_0^T v_0` and
    `v_l^T Kb_l v_l = v_0^T Kb_0 v_0` on every level (no Dirichlet dof in the cube). The coarse levels have 3 + 6 independent such motions."""
    c, jt, host = cube_host(min_free=100)
    assert host.n_levels >= 3
    rng = np.random.default_rng(5)
    t, w, x0 = rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(3)
    ct, kt = [], []
    for l, (lb, Lv) in enumerate(zip(host.border.levels, host.levels)):
        nzc1, nyc1 = Lv.nodedims[2], Lv.nodedims[1]
        g = Lv.nidx[lb.nodes].astype(np.int64)
        ijk = np.stack([g // (nzc1 * nyc1), (g // nzc1) % nyc1, g % nzc1], axis=1) * (c.h * 2 ** l) + c.origin
        v = (t[None, :] + np.cross(w[None, :], ijk - x0[None, :])).reshape(-1)                 # the rigid motion at the bonded nodes
        ct.append(lb.C.T @ v)
        kt.append(v @ (lb.Kb @ v))
    for l in range(1, len(ct)):
        assert np.allclose(ct[l], ct[0], rtol=1e-11, atol=1e-11 * np.abs(ct[0]).max()), l
        assert kt[l] == pytest.approx(kt[0], rel=1e-11), l


# --- F1-7: the noise of the FP64 host check on a part that only joints hold ------------------------------------------------
#
# In a rigid motion of such a part `Kb x` and `C D^-1 C^T x` are each far larger than their difference, so the rounding of the sum
# `C^T x` (a sum over all bonded dofs) shows in the residual of the check of gpu-mg. The reference of these tests is exact: `fractions`
# on the stored float64 numbers (so it is the same on macOS and Linux; no long double).

UNIT_ROUNDOFF = 2.0 ** -53


def rigid_motion(host, c, t, w):
    """`(x, xs)`: a rigid motion `t + w x r` of level 0 at its bonded nodes, `x` (nn, 3) and `xs` the values at the bonded dofs."""
    L0, lb = host.levels[0], host.border.levels[0]
    g = L0.nidx[lb.nodes].astype(np.int64)
    nz1, ny1 = L0.nodedims[2], L0.nodedims[1]
    r = np.stack([g // (nz1 * ny1), (g // nz1) % ny1, g % nz1], axis=1) * c.h + c.origin
    x = np.zeros((L0.nn, 3))
    x.reshape(-1)[lb.dofs] = (t[None, :] + np.cross(w[None, :], r)).reshape(-1)
    return x, x.reshape(-1)[lb.dofs]


def exact_terms(lb, Dinv, xs):
    """`(Kb xs, C D^-1 C^T xs)` as lists of `Fraction`, from the stored float64 numbers of the level border `lb` (no rounding at all)."""
    xf = [Fraction(v) for v in xs]
    Kb, k = lb.Kb, lb.C.shape[1]
    kx = [sum((Fraction(v) * xf[j] for v, j in zip(Kb.data[a:b], Kb.indices[a:b])), Fraction(0)) for a, b in zip(Kb.indptr[:-1], Kb.indptr[1:])]
    ct = [sum((Fraction(lb.C[i, j]) * xf[i] for i in range(xs.size)), Fraction(0)) for j in range(k)]
    s = [sum((Fraction(Dinv[j, m]) * ct[m] for m in range(k)), Fraction(0)) for j in range(k)]
    cs = [sum((Fraction(lb.C[i, j]) * s[j] for j in range(k)), Fraction(0)) for i in range(xs.size)]
    return kx, cs


def norm_of(values):
    """The 2-norm of a list of `Fraction` or float numbers (the squares are summed with `math.fsum`)."""
    return math.sqrt(math.fsum(float(v) ** 2 for v in values))


def test_the_joint_term_of_the_host_operator_in_a_large_rigid_motion_is_right_to_the_roundings_of_its_terms():
    """A cube that only a joint holds, the beam 1e9 times softer than the block, moves as a rigid body (translation about 1e3, rotation about 1).
    `Border.add_to` gives `Kb x - C D^-1 C^T x`; the two terms are each about 5e6 and their exact difference is about 1e-9, so the sum
    `C^T x` must be accurate for the difference to mean anything. The exact difference comes from `fractions`.

    The bound is 4 unit roundoffs of `|Kb x|`. Measured on this cube (OpenBLAS of the numpy wheel, one host), in units of `|Kb x|` times 2^-53:
    the rounding of `Kb x` alone is 1.1 and the pairwise sum of `C^T x` gives 1.6 to 1.7 for 1,158 to 18,438 bonded dofs (this test: 10,374);
    the BLAS product of d43f580 gave 5.6, 20, 31 and 48. The floor is about 1.7: the result of `Kb x` and the last product are rounded once.
    The factor 4 is that floor with a margin of 2.3. A BLAS that sums in a better order than OpenBLAS passes the old code too: this test
    is for the sum that numpy defines, not for every BLAS."""
    c, jt, host = cube_host(n=24, ratio=1e9)
    bd, lb = host.border, host.border.levels[0]
    rng = np.random.default_rng(1)
    x, xs = rigid_motion(host, c, 1e3 * rng.standard_normal(3), rng.standard_normal(3))
    y = np.zeros_like(x)
    bd.add_to(0, x, y)
    kx, cs = exact_terms(lb, bd.Dinv, xs)
    size = norm_of(kx)                                               # |Kb x|, about 5.6e6
    exact = [a - b for a, b in zip(kx, cs)]
    assert lb.dofs.size > 10000 and norm_of(exact) < 1e-6 * size       # the terms cancel: the premise of the test
    noise = norm_of([Fraction(v) - e for v, e in zip(y.reshape(-1)[lb.dofs], exact)])
    assert noise <= 4 * UNIT_ROUNDOFF * size, f"noise {noise / size / UNIT_ROUNDOFF:.1f} unit roundoffs of |Kb x|"


@pytest.mark.parametrize("n_tiles, extra", [(0, 1), (0, 100), (1, -1), (1, 0), (1, 1), (3, 17)])
def test_ct_dot_sums_each_row_to_the_rounding_of_a_pairwise_sum_for_any_length_and_tile(n_tiles, extra):
    """`ct_dot(CT, xs)` against `math.fsum` of the same products, for a length below, at and above the tile of 4096 columns. The bound is 16 unit
    roundoffs of the sum of the absolute terms: the worst pattern that was measured (one large term followed by many terms below half of its
    last bit) gives exactly 8.0 for `ct_dot` (n 100 to 12,305), so 16 is that with a margin of 2; a sum that adds the terms one after the other
    gives 50 to 6,152 on the same pattern and fails. Random terms give 0.01 to 0.1."""
    from topopt.gpu_mg.border import CT_TILE, ct_dot
    n = n_tiles * CT_TILE + extra
    rng = np.random.default_rng(n)
    CT = rng.standard_normal((6, n))
    CT[:3] += 0.5                                                 # three rows with a common sign, three with cancellation
    xs = 1e3 + rng.standard_normal(n)
    got = ct_dot(CT, xs)
    assert got.shape == (6,)
    for i in range(6):
        terms = CT[i] * xs
        assert abs(got[i] - math.fsum(terms)) <= 16 * UNIT_ROUNDOFF * math.fsum(np.abs(terms)), i
    big = np.full((2, n), 2.0 ** -54)
    big[:, 0] = 1.0
    got = ct_dot(big, np.ones(n))
    assert np.abs(got - math.fsum(big[0])).max() <= 16 * UNIT_ROUNDOFF * math.fsum(big[0])    # the worst pattern


@opencl
def test_a_joints_only_cube_with_a_soft_axial_beam_is_accepted_by_tol_and_the_check_is_not_noise(dev):
    """Case B on pocl. A cube of 40 voxels (206,763 free dofs) that only a joint holds, an axial load, a beam whose axial stiffness is 1.5e-9 of the
    bond (section props: A 0.03 mm2, bending and torsion stiff, so the first pass converges). The host check (`gpu_tol` 1e-6, the backward-error
    rule on) measured the true residual 3.36e-6 to 3.43e-6 on d43f580, which is the noise of the BLAS sum `C^T x` (the answer was accepted by the
    backward error and not by the target), and 3.51e-7 to 3.63e-7 with the pairwise sum (accepted by `tol`). The numbers move a little with the
    BLAS thread count; each is about 3 times from the target. One host, pocl; the compliance is checked against the closed form of the beam, so a wrong answer would show."""
    need_case("B", dev)
    c = jc.cube(40)
    jt = jc.make_joint(c, np.array([20.0, 0.0, 0.0]), ("props", 0.03, 100.0, 100.0, 100.0), k_a=100.0 * c.E / c.h)
    f = jc.nodal_load(c, np.array([5.0, 0.0, 0.0]), np.zeros(3))
    info = {}
    u = jc.solve_gpu(c, jt, f, rtol=1e-6, gpu_case="B", info=info, backward_error=1e-15)
    rec = info["record"]
    assert rec["true_rel"] < 1e-6 and rec["accepted_by"] == "tol", rec
    assert float(f @ u) == pytest.approx(5.0 ** 2 * 20.0 / (c.E * 0.03), rel=1e-3)           # F^2 L / (E A) of the beam (the bond adds 1e-9)


def test_the_floor_note_of_the_run_log_says_that_a_part_with_joints_has_a_noise_of_its_own():
    """The run.log line of the backward-error rule (`GpuSolver._floor_note`): the floor 4e-17 is for parts without joints (F1-7)."""
    from topopt.gpu_mg.model import GpuSolver
    note = GpuSolver._floor_note
    plain = note(SimpleNamespace(host=SimpleNamespace(border=None)))
    springs = note(SimpleNamespace(host=SimpleNamespace(border=SimpleNamespace(k=0))))
    joints = note(SimpleNamespace(host=SimpleNamespace(border=SimpleNamespace(k=6))))
    assert plain == springs == "On the designs measured so far a solve at the FP64 floor has a backward error of about 4e-17."
    assert "this run has joints" in joints and "noise of its own" in joints and "CONTRACTS.md section 20" in joints
    assert "about 4e-17; this run" in joints                     # the number stays, with its condition: designs without joints


@opencl
@pytest.mark.parametrize("with_joint", [True, False], ids=["joint", "no joint"])
def test_the_failure_text_gives_the_floor_of_parts_without_joints_and_for_a_joint_run_the_noise_of_the_check(dev, with_joint):
    """A solve that gets 1 iteration in the first pass and 1 in each of 3 restart steps fails. The text has the floor sentence: without a joint
    'a larger value means that the iteration did not converge', with a joint (F1-7) the measured noise of the check (in the relative residual, not
    in the backward error) and the measured backward error of a direct solver, and no claim either way."""
    from topopt.gpu_mg.model import JOINT_FLOOR_FACT, SolveFailure
    need_case("B", dev)
    c = jc.cube(8)
    jt = jc.make_joint(c, np.array([20.0, 0.0, 0.0]), ("round", 2.0), k_a=100.0 * c.E / c.h) if with_joint else None
    f = jc.nodal_load(c, np.array([5.0, 0.0, 0.0]), np.zeros(3))
    fixed = None
    if not with_joint:                                                       # the cube needs a support: one face is clamped
        ids = np.unique(edof_table(c.shape, c.labels == LOAD)[0] // 3)
        ny1, nz1 = c.shape[1] + 1, c.shape[2] + 1
        fixed = (3 * ids[ids // (ny1 * nz1) == 1][:, None] + np.arange(3)).ravel()
    with pytest.raises(SolveFailure) as e:
        jc.solve_gpu(c, jt, f, rtol=1e-9, gpu_case="B", backward_error=1e-15, fixed=fixed, params=dict(max_iter_first=1, max_iter=1, max_refine=3))
    text = str(e.value)
    assert "did not reach the relative residual" in text and "backward error" in text
    if with_joint:
        assert JOINT_FLOOR_FACT in text and "relative residual: about 1.6 unit roundoffs of |Kb u|" in text
        assert "a larger value means that the iteration did not converge" not in text and "does not by itself mean" not in text
    else:
        assert "On the designs without joints measured so far a solve at the FP64 rounding floor has a backward error of about 4e-17 to 2e-16; a larger value means that the iteration did not converge." in text
        assert JOINT_FLOOR_FACT not in text


# (case, beam diameter, beam length). Case B (32-bit V-cycle, 64-bit outer operator) takes the standard thin beam of `tools/joint_check.py`
# (beam over bond stiffness 7e-9 on its softest mode). Case C takes a short thick beam (1e-5), the beam that it solved before the deflation of the rigid
# modes (P2d) too. The thin beam in case C is in `tests/test_gpu_mg_deflation.py`; `test_case_c_without_the_deflation_...` below keeps the stop of the
# engine of before.
CUBE_BEAMS = [("B", 2.0, 20.0), ("C", 8.0, 10.0)]


@opencl
@pytest.mark.parametrize("case_name, d, L", CUBE_BEAMS)
def test_the_cube_with_a_joint_on_gpu_mg_matches_the_bordered_splu_and_for_a_slender_beam_the_closed_forms(dev, case_name, d, L):
    """The loads of `tools/joint_check.py` on a cube that only the beam holds (no fixed region): axial, bending, torsion, and the cross terms. The
    displacement and the compliance of gpu-mg equal those of the bordered `splu` within the solve tolerance times the condition of the cube. For the
    slender beam (L/d 10) the compliance is also near the closed form."""
    need_case(case_name, dev)
    for axis in ((1, 0, 0), (1, 2, 2)):
        xp, p, q = jc.perpendicular_pair(axis)
        loads = {"axial": (5 * xp, np.zeros(3)), "bending": (5 * p, np.zeros(3)), "torsion": (np.zeros(3), 5 * xp),
                 "force p + moment q": (3 * p, 7 * q), "force q + moment p": (3 * q, 7 * p)}
        for name, load in loads.items():
            c = jc.cube(8)
            a = np.asarray(axis, float) / np.linalg.norm(axis)
            jt = jc.make_joint(c, a * L, ("round", d), k_a=100.0 * c.E / c.h)
            f = jc.nodal_load(c, *load)
            info = {}
            u = jc.solve_gpu(c, jt, f, rtol=1e-11 if case_name == "B" else 1e-9, gpu_case=case_name, info=info)      # case C: its 32-bit inner solves stall near 1e-10
            u_ref, _ = jc.solve(c, jt, f, "splu")
            assert info["case"] == case_name and info["levels"] >= 3 and info["record"]["accepted_by"] == "tol" and info["border"]["border_dofs"] == 6
            # the direct solve has an error of its own: the condition of a stiff block on a soft beam is 1e8 to 1e10, times 1e-16
            assert np.linalg.norm(u - u_ref) <= 1e-5 * np.linalg.norm(u_ref), (axis, name)
            assert float(f @ u) == pytest.approx(float(f @ u_ref), rel=1e-6)
            if L / d >= 10:
                _got, ref = jc.case(8, axis, load, L=L, d=d)
                assert abs(float(f @ u) / ref - 1) < 2e-3                                   # the closed form (axial +2.8e-4: block and bond in series)


def soft_cube(c, d, L, ratio=1.0):
    """`(joint set, f, beam over bond stiffness)` of the cube with a round beam of diameter `d` and length `L` along x, and an axial load."""
    import scipy.linalg as sla
    jt = jc.make_joint(c, np.array([L, 0, 0.0]), ("round", d), E_beam=c.E / ratio, k_a=100.0 * c.E / c.h)
    j = jt.joints[0]
    return jt, jc.nodal_load(c, np.array([5.0, 0, 0]), np.zeros(3)), float(sla.eigh(j.B, j.D_bond, eigvals_only=True).min())


@opencl
def test_a_very_soft_beam_converges_on_case_b_and_the_iterations_are_a_fact(dev, capsys):
    """The research ratios: the bond over the beam 2e9 (a round beam d 1 mm over 20 mm) and 2e10 (d 1 over 40, softer than the research cube). The
    multigrid with the border on every level still converges; the iteration counts against the same cube with no joint and a clamp on one face are
    printed (`pytest -s`) and bound loosely. Case B only: the 64-bit outer operator keeps the answer right."""
    need_case("B", dev)
    c = jc.cube(8)
    f = jc.nodal_load(c, np.array([5.0, 0, 0]), np.zeros(3))
    ids = np.unique(edof_table(c.shape, c.labels == LOAD)[0] // 3)
    ny1, nz1 = c.shape[1] + 1, c.shape[2] + 1
    face = ids[ids // (ny1 * nz1) == 1]                                                    # the nodes on the face x = 1 (the cube starts at index 1)
    info = {}
    jc.solve_gpu(c, None, f, rtol=1e-9, gpu_case="B", info=info, fixed=(3 * face[:, None] + np.arange(3)).ravel())     # no joint, clamped
    free_its = info["record"]["executed"]
    rows = {"joint-free clamp": (None, free_its)}
    for label, d, L in (("beam d1/L20", 1.0, 20.0), ("beam d1/L40", 1.0, 40.0), ("beam d2/L20", 2.0, 20.0)):
        jt, f_axial, lam = soft_cube(c, d, L)
        info = {}
        u = jc.solve_gpu(c, jt, f_axial, rtol=1e-9, gpu_case="B", info=info)
        u_ref, _ = jc.solve(c, jt, f_axial, "splu")
        assert info["record"]["accepted_by"] == "tol" and np.linalg.norm(u - u_ref) <= 1e-5 * np.linalg.norm(u_ref), label
        rows[label] = (lam, info["record"]["executed"])
        assert rows[label][1] <= 5 * free_its + 30, (label, rows[label])
    with capsys.disabled():
        print("\nP2b soft beam, case B, PCG iterations executed (beam over bond stiffness on the softest mode): "
              + ", ".join(f"{k} {v[1]}" + ("" if v[0] is None else f" ({v[0]:.1e})") for k, v in rows.items()))


@opencl
@pytest.mark.parametrize("ratio", [1.0, 0.0638], ids=["7e-9", "1.1e-7"])
def test_case_c_without_the_deflation_stops_with_the_facts_and_not_a_wrong_answer_or_a_crash(dev, ratio, monkeypatch):
    """The engine of before P2d (the deflation of the rigid modes switched off, `helpers.switch_off_deflation`). Case C computes the operator in 32-bit float.
    For a beam at 7e-9 and at 1.1e-7 of the bond stiffness on a cube that only a joint holds, the loop does
    not reach the target (measured on pocl; at 7e-9 and on the Arc the inner PCG breaks down to NaN, at 1.1e-7 the loop stagnates or the residual grows: the
    cause depends on the ratio and on the device, review R1, F6). The result is a
    `SolveFailure` with the facts (the ratio, the 32-bit float, the cause of the stop) and the options including a stiffer beam, not an `IndexError` of
    the refinement loop (before P2b a non-finite residual ended there) and not a wrong answer: an accepted result is checked in 64-bit.
    The text states no rule: it does not say that a beam softer than some ratio is not resolved.

    With the deflation case C solves both ratios (`tests/test_gpu_mg_deflation.py`), and the stop with the facts is tested there at a softer beam.

    1.1e-7 is near the edge: in one full suite on pocl (6 workers, high load) the loop reached the target at this ratio, and on pocl alone a
    beam at 7e-8 solves while 3.7e-8 and 1.1e-7 stop (S-fl; the cause of the run-to-run change is not proven). So at 1.1e-7 the test takes
    either outcome: the stop with the facts, or an accepted answer that equals the 64-bit direct solve."""
    from helpers import switch_off_deflation
    switch_off_deflation(monkeypatch)
    from topopt.gpu_mg.model import SolveFailure
    c = jc.cube(8)
    jt, f, lam = soft_cube(c, 2.0, 20.0, ratio)
    info = {}
    try:
        u = jc.solve_gpu(c, jt, f, rtol=1e-9, gpu_case="C", info=info)
    except SolveFailure as err:
        failure = err
    else:
        assert ratio != 1.0, "case C reached the target at 7e-9: the 32-bit inner PCG did not break down"
        u_ref, _ = jc.solve(c, jt, f, "splu")
        assert info["record"]["accepted_by"] == "tol" and np.linalg.norm(u - u_ref) <= 1e-5 * np.linalg.norm(u_ref)
        return
    text = str(failure)
    assert failure.its["failed"] and "did not reach the relative residual" in text and "case C" in text and "32-bit float" in text
    assert f"the beam is {lam:.1e} of the bond stiffness on its softest mode" in text
    assert "not finite" in text or "stopped early" in text or "did not reach" in text           # the cause depends on the device and on the ratio
    assert "Options:" in text and "a stiffer beam (a larger section, a shorter beam)" in text
    assert "the deflation of the rigid modes was not used" in text
    assert "softer than about" not in text and "is not resolved" not in text


def bordered_reference(m, rho):
    """`(U, K)` of a model (amg, it holds `K + Kb`): the direct solve of the bordered system with a Jacobi-scaled `splu`."""
    K = m.stiffness(rho).tocsc()
    C, D = m.problem.joints.border(m.free_dofs)
    A = sps.bmat([[K, C], [C.T, D]], format="csc")
    sc = 1.0 / np.sqrt(A.diagonal())
    rhs = np.vstack([m.f_red, np.zeros((C.shape[1], m.f_red.shape[1]))])
    x = sc[:, None] * splu((sps.diags(sc) @ A @ sps.diags(sc)).tocsc()).solve(sc[:, None] * rhs)
    return x[:K.shape[0]]


def two_joints(tmp_path):
    def mutate(d):
        box(tmp_path / "pad2.stl", (8, -3, 0), (12, 0, 10))
        d["regions"]["load"]["pad2"] = "pad2.stl"
        d["joints"].append({"region": "pad2", "to": [10, -20, 5], "section": {"kind": "round", "d": 4.0}})
    return mutate


JOINT_VARIANTS = {"one joint": dict(), "no fixed region": dict(clamp=False), "two joints": dict(two=True)}


def joint_pair(tmp_path, variant, case_name=None, tol=1e-9):
    """`(reference model, rho, Uref, gpu model)` of a joint case: one joint on a clamped part, the joint as the only support, or two joints."""
    from topopt.gpu_mg.model import GpuModel, GpuParams
    kw = dict(JOINT_VARIANTS[variant])
    two = kw.pop("two", False)
    d_ref, d_gpu = fresh(tmp_path, "ref"), fresh(tmp_path, "gpu")
    ref = Model(joint_prob(d_ref, solver="amg", mutate=two_joints(d_ref) if two else None, **kw))
    rho = 0.1 + 0.8 * np.random.default_rng(3).uniform(size=ref.n_design)
    U = bordered_reference(ref, rho)
    ref.solver.free()

    def gpu_mutate(d):
        if two:
            two_joints(d_gpu)(d)
        d["settings"]["gpu_case"] = case_name
    m = GpuModel(joint_prob(d_gpu, solver="gpu-mg", mutate=gpu_mutate, **kw), None, GpuParams(tol=tol, min_free=300, backward_error=0.0))
    return ref, rho, U, m


def close_tree(a, b, rtol=1e-6, atol=1e-9, path=""):
    """Compare two nested dicts and lists of numbers."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            close_tree(a[k], b[k], rtol, atol, f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close_tree(x, y, rtol, atol, f"{path}/{i}")
    elif isinstance(a, (int, float)):
        assert a == pytest.approx(b, rel=rtol, abs=atol), path


@opencl
@pytest.mark.parametrize("variant", sorted(JOINT_VARIANTS))
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_joint_case_on_gpu_mg_equals_the_bordered_direct_solve_and_the_reactions_match(tmp_path, dev, case_name, variant):
    """Displacement, compliance, the body motion `q`, `joint_wrenches` and the reactions of a joint case (the same design) on the device equal those of
    the bordered `splu` solve within the solve tolerance. The joint is the only support in `no fixed region`; `two joints`: 12 border dofs."""
    from topopt.optimize import reduced_system
    from topopt.springs import reaction_facts
    need_case(case_name, dev)
    ref, rho, Uref, m = joint_pair(tmp_path, variant, case_name)
    try:
        assert m.kb is not None and m.solver.host.border.k == 6 * (2 if variant == "two joints" else 1) and m.solver.host.n_levels >= 3
        U = m.solve(m.stiffness(rho))
        got = m.reaction_facts(U, rho)
        its = m.solver.records[-1]["iterations"][0]
    finally:
        m.solver.free()
    assert np.linalg.norm(U - Uref) <= 1e-6 * np.linalg.norm(Uref)
    assert float(m.f_red[:, 0] @ U[:, 0]) == pytest.approx(float(ref.f_red[:, 0] @ Uref[:, 0]), rel=1e-7)
    assert its["accepted_by"] == "tol"
    edof, _aidx, free = reduced_system(ref.problem)
    want = reaction_facts(ref.problem, edof, ref.element_E(rho), ref.ke, free, Uref)
    g, w = got["cases"][0], want["cases"][0]
    assert set(g["joints"]) == set(w["joints"]) and len(g["joints"]) == (2 if variant == "two joints" else 1)
    close_tree(g["joints"], w["joints"], rtol=1e-5, atol=1e-8)                     # part and ground wrenches, q, bond energy share, nodal force
    for key in ("applied", "sum_force_n", "sum_moment_nmm"):
        close_tree(g[key], w[key], rtol=1e-5, atol=1e-7)
    assert abs(np.array(next(iter(g["joints"].values()))["part"]["force_n"])).max() > 1e-3          # the joint carries a force: the test is not empty
    assert g["force_error"] < 1e-6 and g["moment_error"] < 1e-6


@opencl
@pytest.mark.parametrize("variant", ["one joint", "no fixed region"])
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_run_with_a_joint_finishes_on_gpu_mg_and_writes_the_wrenches(tmp_path, dev, case_name, variant):
    from topopt.cli import main
    need_case(case_name, dev)
    kw = {} if variant == "one joint" else dict(clamp=False)
    p = joint_case(tmp_path, solver="gpu-mg", mutate=lambda d: d["settings"].update(gpu_case=case_name, iterations=3), **kw)
    out = tmp_path / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert len(hist) >= 3 and all(np.isfinite(r["compliance"]) for r in hist) and "amg_iterations" not in hist[-1]
    r = hist[-1]["reactions"]["cases"][0]
    assert set(r["joints"]) == {"pad"} and abs(np.array(r["joints"]["pad"]["part"]["force_n"])).max() > 1e-3
    assert r["force_error"] < 1e-5 and r["residual_rel"] < 1e-5
    log = (out / "run.log").read_text()
    assert "reactions:" in log and "gpu-mg: device" in log and "no stiffness matrix to border" not in log
    run = json.loads((out / "run.json").read_text())
    assert run["joints"] is not None
    recs = [json.loads(line) for line in (out / "gpu_mg.jsonl").read_text().splitlines()]
    assert recs and all(rec["case"] == case_name for rec in recs)


# --- part 4: the texts of `check` (item 4) ----------------------------------------------------------------------------------


def roller_mutate(d):
    """A roller (axis y only) on the y = 10 face of the design: the supports are per-axis."""
    d["regions"]["fixed"] = list(d["regions"]["fixed"]) + [{"file": "roller.stl", "axes": ["y"]}]


def test_check_gives_the_beam_over_bond_fact_for_a_joint_run_on_gpu_mg_only(tmp_path, fake_cpu_device):
    """P2b: a joint run on gpu-mg has one `check` warning: the softest beam over the bond stiffness, the 32-bit float of case C, the measured cases
    with their source, and the options. It states no rule and predicts no stop. Another solver has no such line, and a spring run has none."""
    from topopt.gpu_mg.border import beam_over_bond
    prob = joint_prob(fresh(tmp_path, "p"), solver="gpu-mg")
    region, lam = beam_over_bond(prob.joints)
    d = cli.check_data(joint_case(fresh(tmp_path, "a"), solver="gpu-mg"))
    lines = [w for w in d["warnings"] if w.startswith("joints on gpu-mg: ")]
    assert len(lines) == 1 and not [w for w in d["warnings"] if "cannot start here" in w]
    text = lines[0]
    assert f"'{region}', {lam:.1e} of the bond stiffness" in text and "case C" in text and "32-bit float" in text
    assert "3.3e-7" in text and "1.1e-7" in text and "9.8e-9" in text and "P2b review" in text           # the measured cases and their source
    assert "rigid modes of each body" in text and "9.9e-12" in text and "9.9e-14" in text and "P2d" in text   # P2d: what the deflation changed
    assert "6 harmonic columns for each bonded patch" in text and "P2d-D3h" in text                          # P2d-D3h: the columns of the bonded patches
    assert "4.9e-10" in text and "5.3e-6 against 1e-6" in text and "600,771 free dofs" in text and "187 s" in text    # the measured two-joint part, with the stop of the rigid modes alone
    assert "8 K bytes for each free dof on the device" in text and "24 K + 16 H bytes on the host" in text           # the memory of the columns (formulas of `Deflation.host_bytes`)
    assert "2.5 GB at 2.2 mm" in text and "3.5 GB at 1.0 mm" in text and "counts them in the loop phase of case C" in text      # the measured peak of the whole run, and the columns in the estimate (F4-36)
    assert "Before F4-36 the estimate was 0.69 GB at 2.2 mm and 1.21 GB at 1.0 mm" in text and "2.6 GB and 3.7 GB in case B" in text      # review m5: the old estimate against the measured peaks
    assert "cut at 1,500 s" in text and "peak up to the cut" in text and "whole" not in text
    assert "rest of the gap is not the columns and not the joints" in text and "power 2/3" in text and "cause of the rest of the gap is not known" not in text   # F4-36: what the hook showed
    from topopt.gpu_mg.check import estimate_memory
    lug10, lug22 = estimate_memory(600_771, 6_450_000, "B", 2), estimate_memory(75_489, 0, "B", 2)
    assert f"at least {lug22['bytes'] / 1e9:.1f} GB at 2.2 mm and {lug10['bytes'] / 1e9:.1f} GB at 1.0 mm" in text and f"({lug10['typical_bytes'] / 1e9:.1f} GB for a layout like the hook example)" in text
    assert "0.06 GB and 0.46 GB of the case C peak" in text
    assert "One run on pocl (4 threads)" in text and "One run on an Intel Arc iGPU" in text and "not measured on a real part" in text   # review m6, m8: the runs, the host, what is not measured
    assert "6 to 7 steps and about 1,800 inner iterations" in text and "81 to 83 PCG iterations in case B" in text                    # review m8: the adjoint solves at 1.0 mm
    assert "preconditioner is the 32-bit V-cycle of the device" in text and "Apple" not in text                                      # review n2
    assert "settings.gpu_case B" in text and "another solver" in text and "a stiffer beam" in text
    assert "softer than about" not in text and "is not resolved" not in text and "stops with this fact" not in text
    other = cli.check_data(joint_case(fresh(tmp_path, "b"), solver="amg"))
    assert not [w for w in other["warnings"] if w.startswith("joints on gpu-mg")]
    springs = cli.check_data(case(fresh(tmp_path, "c"), solver="gpu-mg"))
    assert not [w for w in springs["warnings"] if w.startswith("joints on gpu-mg")]


def test_the_text_output_of_check_prints_the_gpu_mg_joint_line(tmp_path, fake_cpu_device, capsys):
    """`cmd_check` prints only the warnings that start with an entry of a fixed tuple: the line of `check_data` must be in it (review R1, F2)."""
    assert main(["check", str(joint_case(tmp_path, solver="gpu-mg")), "--solver", "gpu-mg"]) == 0
    out = capsys.readouterr().out
    assert "joints on gpu-mg: the softest beam is that of 'pad'" in out and "a stiffer beam" in out


@pytest.mark.parametrize("which", ["joints", "springs"])
def test_a_per_axis_support_with_joints_and_with_springs_is_not_refused_on_gpu_mg(tmp_path, fake_cpu_device, capsys, which):
    """P2c lifted the last refusal of P2b: `check` has no warning for a per-axis support next to springs or a joint, and `run` goes past the start
    (the fake device stops it later, at the device). The solves are in `tests/test_gpu_mg_axes.py`."""
    box(tmp_path / "roller.stl", (0, 10, 0), (20, 12, 10))
    p = (joint_case if which == "joints" else case)(tmp_path, solver="gpu-mg", mutate=roller_mutate)
    d = cli.check_data(p)
    assert d["extended"]["supports"]["axes"]["roller.stl"] == "y"
    assert not [w for w in d["warnings"] if "cannot start here" in w or "clamps all 3 dofs" in w] and not hasattr(cli, "GPU_MG_AXES")
    out = tmp_path / "out"
    main(["run", str(p), "--out", str(out), "--quiet"])
    assert "clamps all 3 dofs" not in capsys.readouterr().err
    assert "memory estimate: about" in (out / "run.log").read_text()


# --- part 5: the memory line of a gpu-mg joint run (review R1, F5) ------------------------------------------------------------


def test_the_run_memory_line_of_a_gpu_mg_joint_run_counts_no_bordered_copy_of_K(tmp_path):
    """gpu-mg is matrix-free: it does not assemble `K + Kb` and makes no bordered copy. The other solvers do (section 38)."""
    from topopt.cli import estimate_memory_bytes, run_memory_lines
    from topopt.members import filter_size
    from topopt.optimize import reduced_system
    prob = joint_prob(fresh(tmp_path, "p"), h=2.0, solver="gpu-mg")
    edof, _a, free = reduced_system(prob)
    fsz = filter_size(float(prob.config.settings.filter_radius), float(prob.h), int((prob.labels == 1).sum()), prob.shape)
    lines = run_memory_lines(prob, {"size": fsz}, solver="gpu-mg", case="B")
    assert "bordered copy" not in lines[0]
    from topopt.gpu_mg.check import problem_memory
    gpu = problem_memory(int(free.size), int(np.prod(prob.shape)), "B", 1, fsz)      # the gpu-mg model, as `topopt check` gives it (no Cholesky number)
    assert f"memory estimate: about {cli.gib(gpu['bytes'])}" in lines[0] and f"the gpu-mg model, set by its {gpu['phase']} phase" in lines[0]
    assert "the bordered copy of K for the joints" in run_memory_lines(prob, {"size": fsz}, solver="amg")[0]
    assert "the bordered copy of K for the joints" in run_memory_lines(prob, {"size": fsz})[0]       # no solver given: the old line


def test_the_run_log_of_a_gpu_mg_joint_run_has_no_bordered_copy_line(tmp_path, fake_cpu_device, capsys):
    """The call in `cmd_run` passes the resolved solver. The fake device stops the run later, at the device, after the memory line."""
    out = tmp_path / "out"
    main(["run", str(joint_case(tmp_path, h=2.0, solver="gpu-mg")), "--out", str(out), "--quiet"])
    log = (out / "run.log").read_text()
    assert "memory estimate: about" in log and "the bordered copy of K" not in log


def test_check_data_of_a_gpu_mg_joint_run_has_the_gpu_mg_memory_model_and_no_bordered_copy(tmp_path, fake_cpu_device):
    """`check_data` already gave the gpu-mg model for `bytes` (`gpu_mg.check.apply`); the bordered copy is only in `cholesky_bytes`, the number of the
    other solver for comparison, which has it. A test pins both so that the line cannot return."""
    from topopt.gpu_mg.check import problem_memory
    d = cli.check_data(joint_case(tmp_path, h=2.0, solver="gpu-mg"))
    me = d["memory_estimate"]
    case = d["time_estimate"]["gpu_mg"]["case"]
    want = problem_memory(d["free_dofs"], d["grid"]["voxels"], case, 1, d["derived"]["filter"]["size"])      # with the filter numbers of the check
    assert me["model"] == "gpu-mg" and me["bytes"] == want["bytes"] and "bordered_copy_bytes" not in me
    amg = cli.check_data(joint_case(fresh(tmp_path, "a"), h=2.0, solver="amg"))["memory_estimate"]
    assert me["cholesky_bytes"] == amg["bytes"] and amg["bordered_copy_bytes"] > 0


def test_the_build_check_of_a_gpu_mg_joint_part_does_not_count_a_bordered_copy(tmp_path, monkeypatch, fake_cpu_device):
    """`regions_builder.check_yaml` compares the memory estimate with the RAM. With gpu-mg and a joint the estimate has no bordered copy, so a RAM
    between the two numbers is no error for gpu-mg and is one for amg."""
    from topopt import regions_builder
    from topopt.cli import bordered_bytes, estimate_memory_bytes
    from topopt.members import filter_size
    from topopt.optimize import reduced_system
    prob = joint_prob(fresh(tmp_path, "p"), h=2.0, solver="amg")
    edof, _a, free = reduced_system(prob)
    fsz = filter_size(float(prob.config.settings.filter_radius), float(prob.h), int((prob.labels == 1).sum()), prob.shape)
    g, f, e = int(np.prod(prob.shape)), int(free.size), fsz["held_bytes_estimate"]
    bord = bordered_bytes(prob, edof, free)
    without, with_b = estimate_memory_bytes(g, f, e), estimate_memory_bytes(g, f, e, bord)
    assert with_b > without + 100_000
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: (without + with_b) // 2)
    for solver, ok in (("gpu-mg", True), ("amg", False)):
        lines = []
        done, kind = regions_builder.check_yaml(joint_case(fresh(tmp_path, solver), h=2.0, solver=solver),
                                              say=lambda *a, **k: lines.append(" ".join(map(str, a))))
        text = "\n".join(lines)
        assert ("above the RAM" in text) is (not ok), (solver, text)


# --- review R1, N4: the local memory of the joint kernel ----------------------------------------------------------------------------------


def test_the_local_memory_of_the_joint_kernel_is_compared_with_the_device_size_and_a_stop_gives_the_facts(tmp_path):
    """`bord_w` keeps `nr * nk` numbers in local memory (`nk` = 6 for each joint; 8 bytes in case B, 4 bytes in case C). When that is more than
    `CL_DEVICE_LOCAL_MEM_SIZE`, the text gives the needed bytes, the device bytes, the joints count and the options. It adds no number of its own."""
    import dataclasses
    from topopt.gpu_mg.model import border_local_memory_problem
    from helpers import make_device
    border = grid_border(joint_prob(fresh(tmp_path, "p"), solver="gpu-mg"))
    assert border.k == 6
    dev = make_device()
    assert border_local_memory_problem(border, 2, "B", dev) is None and border_local_memory_problem(border, 2, "C", dev) is None
    small = dataclasses.replace(dev, local_mem=90)                       # case B needs 2 x 6 x 8 = 96 bytes, case C needs 48
    text = border_local_memory_problem(border, 2, "B", small)
    assert "1 joints have 6 border dofs" in text and "2 x 6 x 8 bytes = 96 bytes" in text and "has 90 bytes of local memory" in text
    assert "case B" in text and dev.name in text and "Options: use fewer joints, or set settings.solver" in text
    assert border_local_memory_problem(border, 2, "C", small) is None                        # 48 bytes fit
    text = border_local_memory_problem(border, 2, "C", dataclasses.replace(dev, local_mem=40))
    assert "2 x 6 x 4 bytes = 48 bytes" in text and "has 40 bytes" in text
    assert border_local_memory_problem(border, 2, "B", dataclasses.replace(dev, local_mem=0)) is None          # the device gave no size
    assert border_local_memory_problem(None, 2, "B", small) is None
    springs = grid_border(prob_of(fresh(tmp_path, "s"), solver="gpu-mg"))
    assert springs.k == 0 and border_local_memory_problem(springs, 2, "B", dataclasses.replace(dev, local_mem=1)) is None      # springs only: no border dofs


@opencl
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_joint_run_on_a_device_with_too_little_local_memory_stops_before_the_engine_with_the_facts(tmp_path, dev, monkeypatch, case_name):
    """The stop is a `UserError` made in `GpuSolver.__init__`, before the program is built: the engine is not made."""
    import dataclasses
    from topopt.config import UserError
    from topopt.gpu_mg import device as devmod, engine
    small = dataclasses.replace(dev, local_mem=20)           # one load case column: 6 x 8 = 48 bytes in case B, 6 x 4 = 24 in case C
    monkeypatch.setattr(devmod, "select", lambda d="auto", c="auto": (small, case_name))
    monkeypatch.setattr(engine.GpuMG, "__init__", lambda self, *a, **k: pytest.fail("the engine was made"))
    with pytest.raises(UserError) as e:
        joint_pair(tmp_path, "one joint", case_name)
    text = str(e.value)
    assert "has 20 bytes of local memory" in text and f"case {case_name}" in text and "1 joints have 6 border dofs" in text
    assert "Options: use fewer joints" in text
