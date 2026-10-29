"""P2d-D3h: the harmonic columns of the bonded patches in the deflation of the gpu-mg case C with joints (`Deflation.extend`, `GpuMG.precondition`,
`GpuMG.upload_deflation`, `GpuSolver.set_design`).

The cause (measured in the research of P2d-2): the 32-bit error of the joint term is mostly a net force and moment on each bonded patch, and a part
answers it in the soft motions where one sleeve moves against the other. The six rigid modes of D3 do not cover those motions. For each patch the harmonic
columns are its rigid motions extended elastically into the part (the minimum of the element energy, the other patches and the Dirichlet dofs at zero, no
joint term); the extension is solved for each design with an FP64 PCG on the host and the V-cycle of the device as preconditioner.

The test part is a "dumbbell" (`dumbbell`): a thin bar with a wide plate at each end, a joint on each plate (two patches, one body, no fixed region). Two
joints on a thin bar are the smallest part found on which D3 alone stops in case C on pocl and the harmonic columns solve (the two-joint part of the research has the
same shape: two big sleeves on a slender part). The host-only tests use a smaller dumbbell and a block-Jacobi preconditioner (`jacobi`); the device tests run on the
OpenCL device of the other gpu-mg tests (pocl on the host of the tests). The reference is the Jacobi-scaled bordered `splu` of the same design.
"""
import inspect
import json
import re

import numpy as np
import pytest
import scipy.linalg as sla
import yaml

from test_gpu_mg import need_case
from test_gpu_mg_border import bordered_reference, host_of
from test_gpu_mg_deflation import jacobi, rel_err
from test_springs import box
from topopt.config import UserError, load_config
from topopt.optimize import Model
from topopt.problem import build_problem

opencl = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1

FALL = 0.10                  # engine.STAGNATION_FALL (a test of the engine pins the value)
BIG = dict(bar=(48, 4, 4), W=16, T=2)        # D3 alone stops on this part at a beam over bond stiffness of 1e-9 (measured 12 steps, residual 9.5e-4)
SMALL = dict(bar=(16, 4, 4), W=8, T=2)       # the host-only tests and the device tests that need only a part with two bonded patches
STAG = dict(bar=(12, 4, 4), W=6, T=2)        # the part where a chunk of 10 iterations is too short for the extension


@pytest.fixture(scope="module")
def dev():
    """The device that `auto` takes. The test is skipped when there is none, or without pyopencl."""
    pytest.importorskip("pyopencl")
    from topopt.gpu_mg import device
    try:
        return device.select("auto", "auto")[0]
    except UserError as e:
        pytest.skip(f"no usable OpenCL device: {e}")


def dumbbell(tmp_path, name, solver="gpu-mg", case_name="C", energies=None, bar=(48, 4, 4), W=16, T=2, clamp=False, mutate=None):
    """The problem of a bar `bar` (voxels of 1 mm) with a square plate of side `W` and thickness `T` at each end, a joint on each plate (a round beam of 4 mm to a
    far point on the axis) and one load on the bar. `energies`: the E of the two joints (None: the default). `clamp`: a fixed block at the middle of the bar.
    `mutate(cfg, d)`: a change of the config `cfg` (a dict) before it is written; `d` is the folder of the STL files (a change can write a file too)."""
    d = tmp_path / name
    d.mkdir()
    bx, by, bz = bar
    y0, z0 = (by - W) / 2, (bz - W) / 2
    box(d / "design.stl", (0, 0, 0), bar)
    box(d / "pa.stl", (-T, y0, z0), (0, y0 + W, z0 + W))
    box(d / "pb.stl", (bx, y0, z0), (bx + T, y0 + W, z0 + W))
    box(d / "tip.stl", (bx / 2 - 2, by / 2 - 1, bz), (bx / 2 + 2, by / 2 + 1, bz + 2))
    box(d / "clamp.stl", (bx / 2 - 6, 0, 0), (bx / 2 - 4, by, bz))
    joint = lambda reg, to, e: {"region": reg, "to": to, "section": {"kind": "round", "d": 4.0}, **({} if e is None else {"E": float(e)})}
    e = energies or (None, None)
    cfg = {"name": "dumbbell",
           "regions": {"design": "design.stl", "fixed": ["clamp.stl"] if clamp else [], "load": {"pa": "pa.stl", "pb": "pb.stl", "tip": "tip.stl"}},
           "joints": [joint("pa", [-20, by / 2, bz / 2], e[0]), joint("pb", [bx + 20, by / 2, bz / 2], e[1])],
           "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
           "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": 3, "solver": solver, "gpu_case": case_name}}
    if mutate:
        mutate(cfg, d)
    (d / "c.yaml").write_text(yaml.safe_dump(cfg))
    return build_problem(load_config(d / "c.yaml"))


def joint_voxel_between_fixed_voxels(cfg, d):
    """`mutate` of `dumbbell` (the SMALL part): the region of the first joint is ONE voxel, x from -2 to -1, with a fixed voxel on each side along x (x from -3 to
    -2, and from -1 to 0, which touches the bar). All 8 nodes of the joint voxel belong to a fixed voxel, so every bonded dof of that joint is Dirichlet and its
    patch has no free node. The second joint (the plate at the other end) is not changed."""
    box(d / "pa.stl", (-2, 1, 1), (-1, 2, 2))
    box(d / "fa.stl", (-3, 1, 1), (-2, 2, 2))
    box(d / "fb.stl", (-1, 1, 1), (0, 2, 2))
    cfg["regions"]["fixed"] = ["fa.stl", "fb.stl"]


def third_joint_on_a_region_that_touches_the_first(cfg, d):
    """`mutate` of `dumbbell` (the SMALL part): a third plate (x from -4 to -2) that touches the plate of the first joint (x from -2 to 0) on one face, with a joint
    on it. The bonded patches of the two joints share nodes on that face."""
    box(d / "pc.stl", (-4, 0, 0), (-2, 4, 4))
    cfg["regions"]["load"]["pc"] = "pc.stl"
    cfg["joints"].append({"region": "pc", "to": [-24, 2, 2], "section": {"kind": "round", "d": 4.0}})


def with_a_stress_limit(cfg, d):
    """`mutate` of `dumbbell`: a stress limit on the load case, so that a run has a stress row and its post-process makes the stress report."""
    cfg["load_cases"][0]["stress_limit"] = 500.0


def energies_for(tmp_path, lam, **kw):
    """The joint E of the two joints that give the beam over bond stiffness `lam` on the softest mode of each."""
    probe = dumbbell(tmp_path, "probe", **kw)
    l1 = [float(sla.eigh(j.B, j.D_bond, eigvals_only=True).min()) for j in probe.joints.joints]
    e1 = [(j.E if j.E is not None else probe.config.material.E) for j in probe.config.joints]
    return [e * lam / l for e, l in zip(e1, l1)]


def soft_dumbbell(tmp_path, lam, rhos=(1.0,), params=None, case_name="C", out=None, **kw):
    """`(GpuModel, [rho], [reference U])` of the dumbbell with the beams scaled to the beam over bond stiffness `lam` on their softest mode. The first design
    is `0.1 + 0.8 u` for a random `u` (seed 3); the design number i is that design times `rhos[i]` (clipped to 0.02 .. 1), so that `rhos=(1.0, 1.25)` is two
    designs with other densities. The reference is the Jacobi-scaled bordered `splu` of the same design."""
    from topopt.gpu_mg.model import GpuModel, GpuParams
    en = energies_for(tmp_path, lam, **kw)
    ref = Model(dumbbell(tmp_path, "ref", solver="amg", energies=en, **kw))
    base = 0.1 + 0.8 * np.random.default_rng(3).uniform(size=ref.n_design)
    rho_list = [np.clip(base * f, 0.02, 1.0) for f in rhos]
    u_ref = [bordered_reference(ref, r) for r in rho_list]
    ref.solver.free()
    m = GpuModel(dumbbell(tmp_path, "gpu", energies=en, case_name=case_name, **kw), out, params or GpuParams(tol=1e-6, min_free=300, backward_error=0.0))
    return m, rho_list, u_ref


def small_host(tmp_path, clamp=False, rho=None, mutate=None):
    """The host hierarchy of the small dumbbell after `setup_host` of the design `rho` (default: a random design of E between 0.3 and 1 times the E of the
    material), with the border of its joints (`mutate`: see `dumbbell`)."""
    prob = dumbbell(tmp_path, "h", **SMALL, clamp=clamp, mutate=mutate)
    host, _ = host_of(prob)
    if rho is None:
        rho = 0.3 + 0.7 * np.random.default_rng(1).uniform(size=host.levels[0].nel)
    host.setup_host(prob.config.material.E * rho)
    host.E0 = prob.config.material.E
    return host


def make(host, tol=1e-8, maxit=300, block=2, **kw):
    """`(Deflation, facts)` after an extension with the Jacobi preconditioner."""
    from topopt.gpu_mg.deflation import Deflation
    d = Deflation(host)
    return d, d.extend(host, jacobi(host), block, maxit, FALL, tol=tol, **kw)


def residuals(host, d):
    """The relative residual of each harmonic column of `d`, made again from the columns and the operator: `|K_ff u + K_fs u_s| / |K_ff u_s' + ...|`, that is the
    norm over the free dofs of `K u` against the norm over the free dofs of `K u_s` (`u_s` = the prescribed motion `d.U`, the right-hand side of the extension)."""
    L = host.levels[0]
    y = np.zeros((L.nn, 3))
    out = np.empty(d.n_harm)
    for j in range(d.n_harm):
        host.apply_elements(d.H[j].reshape(-1, 3), y)
        num = np.linalg.norm(y.reshape(-1)[d.free])
        host.apply_elements(d.U[j].reshape(-1, 3), y)
        out[j] = num / np.linalg.norm(y.reshape(-1)[d.free])
    return out


# --- the columns (host only) -------------------------------------------------------------------------------------------------------------------


def rigid_basis(L, nodes):
    """(3 n, 6) the three translations and the three rotations (about the centre) of the nodes `nodes` of level `L`, one column each."""
    from topopt.gpu_mg.deflation import node_positions
    r = node_positions(L)[nodes]
    r = r - r.mean(axis=0)
    rigid = np.zeros((len(nodes), 3, 6))
    rigid[:, :, :3] = np.eye(3)
    for a in range(3):
        rigid[:, :, 3 + a] = np.cross(np.eye(3)[a][None, :], r)
    return rigid.reshape(-1, 6)


def test_a_harmonic_column_is_the_patch_motion_on_its_patch_zero_on_the_other_patch_and_at_fixed_dofs_and_harmonic_inside(tmp_path):
    """The two patches have 6 columns each. A column of patch A equals a rigid motion of A on A, is zero on B and on every Dirichlet dof (here a clamp block in
    the bar and a roller on one more node), and `K_ff u + K_fs u_s` is small against its tolerance: it is the elastic extension and not a guess."""
    host = small_host(tmp_path, clamp=True)
    L = host.levels[0]
    from topopt.gpu_mg.deflation import patch_nodes
    candidates = np.setdiff1d(np.flatnonzero(~L.fix.astype(bool).any(axis=1)), np.concatenate(patch_nodes(host)))      # free nodes that are on no patch
    L.fix[candidates[len(candidates) // 2]] = [1, 0, 0]                                 # a roller on one of them
    tol = 1e-8
    d, ext = make(host, tol=tol)
    assert d.per_patch == [6, 6] and d.n_harm == 12 and d.H.shape == (12, d.n3) and ext["converged"] and ext["columns"] == 12
    fixed = L.fix.reshape(-1).astype(bool)
    assert fixed.sum() > 3 and np.abs(d.H[:, fixed]).max() == 0                      # zero at the Dirichlet dofs (the clamp and the roller)
    dofs = [(3 * nodes[:, None] + np.arange(3)).reshape(-1) for nodes in d.patches]
    assert not np.intersect1d(dofs[0], dofs[1]).size
    for m in range(2):
        rows = slice(6 * m, 6 * m + 6)
        assert np.abs(d.H[rows][:, dofs[1 - m]]).max() == 0                           # the other patch is held at zero
        on = d.H[rows][:, dofs[m]]
        assert np.abs(on - d.U[rows][:, dofs[m]]).max() == 0 and np.abs(on).max() > 0        # the patch keeps the motion that was prescribed
        rigid = rigid_basis(L, d.patches[m])
        coef = np.linalg.lstsq(rigid, on.T, rcond=None)[0]
        assert np.abs(rigid @ coef - on.T).max() < 1e-12                              # and that motion is a rigid motion of the patch (cut at the Dirichlet dofs)
    res = residuals(host, d)
    assert res.max() <= tol * 1.0001 and ext["rel_max"] <= tol * 1.0001 and abs(res.max() - ext["rel_max"]) <= 0.2 * ext["rel_max"]


def test_the_extensions_of_one_rigid_motion_of_all_patches_add_up_to_that_motion_of_the_whole_body(tmp_path):
    """No Dirichlet dof: a translation of both patches is a motion of no energy, so its extension is the translation of the whole bar, and (the extension is
    linear) the extensions of the translation of each patch alone, with the other patch at zero, add up to it. This is why the raw harmonic columns are nearly
    dependent on the rigid columns of the body (see the orthonormal basis in `set_design`). A check of the extension, apart from the product code path."""
    host = small_host(tmp_path)
    d, _ = make(host, tol=1e-10)
    L = host.levels[0]
    tx = []
    for m in range(2):
        nodes = d.patches[m]
        dofs = (3 * nodes[:, None] + np.arange(3)).reshape(-1)
        B = d.U[6 * m:6 * m + 6][:, dofs].T                                           # the six prescribed motions of this patch, one column each
        t = np.zeros(3 * len(nodes))
        t[0::3] = 1.0                                                                 # a unit translation along x
        c = np.linalg.lstsq(B, t, rcond=None)[0]
        assert np.abs(B @ c - t).max() < 1e-12                                        # it is in the span of the six patch columns
        tx.append(c @ d.H[6 * m:6 * m + 6])
    both = tx[0] + tx[1]
    assert np.abs(both[0::3] - 1.0).max() < 1e-4 and np.abs(both[1::3]).max() < 1e-4 and np.abs(both[2::3]).max() < 1e-4
    y = np.zeros((L.nn, 3))
    host.apply_elements(both.reshape(-1, 3), y)
    assert np.abs(y).max() < 1e-4 * np.abs(L.kel).max()


def test_a_joint_whose_bonded_dofs_are_all_fixed_has_an_empty_patch_and_no_column_and_the_build_does_not_fail(tmp_path):
    """Review m1. The first joint is on ONE voxel with a fixed voxel on each side (`joint_voxel_between_fixed_voxels`): the Dirichlet entries are cut from its
    `C` block, so `patch_nodes` gives an empty patch for it. The build skipped nothing before and `body_modes` raised `IndexError` on the empty patch. Now the
    patch has 0 columns (`per_patch`), the other patch has its 6, every column is zero at the fixed dofs, and the extension and `set_design` run."""
    from topopt.gpu_mg.deflation import Deflation, patch_nodes
    host = small_host(tmp_path, mutate=joint_voxel_between_fixed_voxels)
    L = host.levels[0]
    assert host.border.k == 12 and L.fix.all(axis=1).sum() == 16                       # two joints; the 8 nodes of each of the 2 fixed voxels
    nodes = patch_nodes(host)
    assert [n.size for n in nodes][0] == 0 and nodes[1].size > 0
    d = Deflation(host)                                                              # IndexError before
    assert len(d.patches) == 2 and d.per_patch == [0, 6] and d.n_harm == 6 and d.U.shape == (6, d.n3) and d.dk == d.n_rigid + 6
    ext = d.extend(host, jacobi(host), 2, 300, FALL, tol=1e-8)
    assert ext["columns"] == 6 and ext["converged"] and d.H.shape == (6, d.n3)
    d.set_design(host)
    fixed = L.fix.reshape(-1).astype(bool)
    assert d.ready and d.k == d.dk and np.abs(d.Zt[:d.n_rigid][:, fixed]).max() == 0 and np.abs(d.AZt[:, fixed]).max() == 0
    assert np.abs(d.Zt[:, fixed]).max() <= 1e-12            # the QR of the harmonic block leaves rounding (measured 2.5e-14) at the fixed dofs
    Z = d.Zt[:d.k]
    assert np.abs(Z @ Z.T - np.eye(d.k)).max() < 1e-10 and np.linalg.eigvalsh(d.E).min() > 0
    dofs = (3 * nodes[1][:, None] + np.arange(3)).reshape(-1)                         # the six columns are the extensions of the motions of the second patch
    assert np.abs(d.H[:, dofs] - d.U[:, dofs]).max() == 0 and np.abs(d.H[:, dofs]).max() > 0


def test_two_patches_that_share_a_node_each_keep_their_own_motion_there_and_the_node_is_no_unknown(tmp_path):
    """Review n1. A third joint is on a plate that touches the plate of the first joint, so the two patches share nodes (`third_joint_on_a_region_that_touches_the_first`).
    `Deflation` writes the motion of each patch on all its nodes, so at a shared node the columns of BOTH patches are not zero: each holds the motion of its own
    patch, and the two differ (the centres of the patches differ). A column is zero at the nodes of the other patches that the patch does not share. The shared
    nodes are not unknowns of the extension, and the columns are harmonic elsewhere."""
    host = small_host(tmp_path, mutate=third_joint_on_a_region_that_touches_the_first)
    d, ext = make(host, tol=1e-8)
    assert d.per_patch == [6, 6, 6] and ext["converged"]
    a, c = d.patches[0], d.patches[2]
    shared = np.intersect1d(a, c)
    assert shared.size > 0 and not np.intersect1d(a, d.patches[1]).size and not np.intersect1d(c, d.patches[1]).size
    dof = lambda nodes: (3 * nodes[:, None] + np.arange(3)).reshape(-1)
    sd, only_a, only_c, other = dof(shared), dof(np.setdiff1d(a, c)), dof(np.setdiff1d(c, a)), dof(d.patches[1])
    assert not d.free[sd].any() and d.free[dof(np.setdiff1d(np.arange(host.levels[0].nn), np.concatenate(d.patches)))].all()
    ha, hc = d.H[0:6], d.H[12:18]
    assert np.abs(ha[:, sd] - d.U[0:6][:, sd]).max() == 0 and np.abs(ha[:, sd]).max() > 0         # patch a: its own motion on the shared nodes
    assert np.abs(hc[:, sd] - d.U[12:18][:, sd]).max() == 0 and np.abs(hc[:, sd]).max() > 0       # patch c: its own motion on the same nodes
    assert np.abs(ha[:, sd] - hc[:, sd]).max() > 1e-3                                              # and the two motions differ there
    assert np.abs(ha[:, only_c]).max() == 0 and np.abs(hc[:, only_a]).max() == 0                  # zero on the nodes of the other patch that are not shared
    assert np.abs(d.H[0:6][:, other]).max() == 0 and np.abs(d.H[12:18][:, other]).max() == 0
    assert residuals(host, d).max() <= 1e-8 * 1.0001


def test_the_warm_start_gives_the_same_columns_within_the_tolerance_with_fewer_iterations(tmp_path):
    """The second design (every density changed by up to 5 percent) is solved cold and warm from the extension of the first. Both reach the tolerance (checked
    again from the columns), the columns are the same within a few hundred times the tolerance (the residual over the smallest eigenvalue of the extension), and
    the warm start needs fewer iterations (pocl host, Jacobi: 438 against 871 at 1e-6; 40 against 513 at 1e-4)."""
    from topopt.gpu_mg.deflation import Deflation
    rng = np.random.default_rng(1)
    host = small_host(tmp_path)
    ne, e0 = host.levels[0].nel, host.E0
    rho1 = 0.3 + 0.7 * rng.uniform(size=ne)
    rho2 = np.clip(rho1 * (1 + 0.05 * rng.uniform(-1, 1, size=ne)), 0.05, 1.0)
    tol = 1e-6
    host.setup_host(e0 * rho1)
    warm = Deflation(host)
    first = warm.extend(host, jacobi(host), 2, 300, FALL, tol=tol)
    host.setup_host(e0 * rho2)
    w = warm.extend(host, jacobi(host), 2, 300, FALL, tol=tol)
    cold = Deflation(host)
    c = cold.extend(host, jacobi(host), 2, 300, FALL, tol=tol)
    assert not first["warm"] and not c["warm"] and w["warm"]
    assert w["converged"] and c["converged"] and w["rel_max"] <= tol and c["rel_max"] <= tol
    assert w["iterations"] < 0.6 * c["iterations"], (w["iterations"], c["iterations"])
    assert residuals(host, warm).max() <= tol * 1.0001 and residuals(host, cold).max() <= tol * 1.0001
    diff = np.abs(warm.H - cold.H).max() / np.abs(cold.H).max()
    assert diff <= 1e-3, diff                              # measured 1.3e-4
    assert cold.ext_total["designs"] == 1 and warm.ext_total["designs"] == 2 and warm.ext_total["warm"] == 1
    assert warm.ext_total["iterations"] == first["iterations"] + w["iterations"]


@pytest.mark.parametrize("block", [1, 2, 5, 12])
def test_the_block_size_changes_no_column_a_short_last_block_repeats_its_last_column(tmp_path, block):
    host = small_host(tmp_path)
    ref, _ = make(host, tol=1e-9, block=2)
    d, ext = make(host, tol=1e-9, block=block)
    assert ext["converged"] and len(ext["iterations_per_block"]) == -(-12 // block)
    assert np.abs(d.H - ref.H).max() / np.abs(ref.H).max() <= 1e-5


def test_the_extension_runs_in_chunks_with_a_restart_between_them_and_reaches_the_tolerance(tmp_path):
    """A chunk is `maxit` iterations. With a short chunk a column needs several chunks; the true residual is made again between them (a restart), and the extension
    still reaches the tolerance. The one-chunk run has one chunk for each block."""
    host = small_host(tmp_path)
    one, e1 = make(host, tol=1e-8, maxit=300)
    short, e2 = make(host, tol=1e-8, maxit=15)
    assert e1["chunks"] == len(e1["iterations_per_block"]) and e1["converged"]
    assert e2["chunks"] > e1["chunks"] and e2["converged"] and e2["stopped"] == ["tol"] * len(e2["stopped"]) and e2["rel_max"] <= 1e-8
    assert max(e2["iterations_per_block"]) > 15                                 # more than one chunk of 15 in some block
    assert np.abs(short.H - one.H).max() / np.abs(one.H).max() <= 1e-4


def test_an_extension_that_cannot_reach_its_tolerance_stops_on_stagnation_and_the_columns_stay_valid(tmp_path):
    """A tolerance below the FP64 floor: the residual stops falling, and the extension stops after two chunks that fell by less than `STAGNATION_FALL` (10
    percent) and not by a cap. It is a fact (`converged` False, `stopped`, `rel_max`), and the columns are still columns of `Z`: `E` is positive definite and
    the coarse correction leaves a load A-orthogonal to `Z`."""
    host = small_host(tmp_path)
    d, ext = make(host, tol=1e-40, maxit=40)
    assert not ext["converged"] and ext["stopped"] == ["stagnation"] * len(ext["stopped"]) and ext["tol"] == 1e-40
    assert 1e-40 < ext["rel_max"] < 1e-9 and np.isfinite(d.H).all()
    assert min(ext["iterations_per_block"]) >= 40                                  # at least one chunk
    d.set_design(host)
    assert d.ready and np.linalg.eigvalsh(d.E).min() > 0
    b = np.random.default_rng(0).standard_normal((2, d.n3)) * (~host.levels[0].fix.reshape(-1).astype(bool))
    _, b2 = d.coarse(b)
    assert np.abs(d.Zt[:d.k] @ b2.T).max() <= 1e-6 * np.abs(d.Zt[:d.k] @ b.T).max()
    assert d.ext_total["above_tol"] == 1


def test_a_preconditioner_that_is_not_positive_stops_the_extension_with_the_fact_and_one_that_is_not_finite_is_an_error(tmp_path):
    from topopt.gpu_mg.deflation import Deflation, DeflationError
    host = small_host(tmp_path)
    good = jacobi(host)
    d = Deflation(host)
    ext = d.extend(host, lambda R: -good(R), 2, 300, FALL, tol=1e-8)
    assert not ext["converged"] and ext["stopped"] == ["breakdown"] * len(ext["stopped"]) and np.isfinite(d.H).all()
    d = Deflation(host)
    with pytest.raises(DeflationError, match="not finite"):
        d.extend(host, lambda R: np.full_like(R, np.nan), 2, 300, FALL, tol=1e-8)
    assert d.H is None and not d._fresh                                                # nothing was stored


def test_z_is_the_rigid_columns_then_an_orthonormal_basis_of_the_harmonic_columns_and_E_is_well_conditioned(tmp_path):
    """The raw harmonic columns of a body that no Dirichlet dof holds add up to its rigid modes, so `Z^T A Z` of the raw columns is nearly singular. `set_design`
    keeps the rigid columns, orthogonalizes the harmonic ones against them, and takes an orthonormal basis of what is left. The span is the span of all columns."""
    host = small_host(tmp_path)
    d, _ = make(host, tol=1e-6)
    raw = d.H.copy()
    rigid = d.Zt[:d.n_rigid].copy()
    d.set_design(host)
    k = d.k
    assert k == d.dk == 18 and np.array_equal(d.Zt[:d.n_rigid], rigid)               # the rigid columns are not changed
    Z = d.Zt[:k]
    assert np.abs(Z @ Z.T - np.eye(k)).max() < 1e-12                                  # orthonormal, the harmonic ones orthogonal to the rigid ones
    span = np.vstack([rigid, raw])
    assert np.abs(span - (span @ Z.T) @ Z).max() <= 1e-9 * np.abs(span).max()      # every raw column is in the span of Z
    L = host.levels[0]
    y = np.zeros((L.nn, 3))
    # E of the orthonormal columns is a compression of A: its condition number is far below the one of the raw columns
    cols = np.vstack([rigid, raw])
    AR = np.empty_like(cols)
    for j in range(cols.shape[0]):
        host.apply_fine(cols[j].reshape(-1, 3), y)
        AR[j] = y.reshape(-1)
    Er = cols @ AR.T
    Er = 0.5 * (Er + Er.T)
    assert np.linalg.cond(d.E) < np.linalg.cond(Er) / 10, (np.linalg.cond(d.E), np.linalg.cond(Er))


def test_a_dependent_harmonic_column_is_dropped_by_the_rank_and_the_columns_of_the_program_are_padded_with_zeros(tmp_path):
    """The program of the device has `dk` columns; a design with fewer independent columns pads `Z`, `AZ` and `E^-1` with zeros, so the projection of the kernels
    is that of the independent columns only. Here the harmonic columns of the first patch are copies of rigid columns."""
    host = small_host(tmp_path)
    d, _ = make(host, tol=1e-8)
    d.H[:6] = d.Zt[:6]
    d._fresh = True
    d.set_design(host)
    assert d.k == 12 and d.dk == 18 and d.ready
    assert np.abs(d.Zt[12:]).max() == 0 and np.abs(d.AZt[12:]).max() == 0 and np.abs(d.Zt32[12:]).max() == 0 and np.abs(d.AZt32[12:]).max() == 0
    assert np.abs(d.Einv32[12:]).max() == 0 and np.abs(d.Einv32[:, 12:]).max() == 0 and d.Einv32.shape == (18, 18) and d.Einv.shape == (12, 12)
    z = np.random.default_rng(0).standard_normal((1, d.n3))
    c = d.AZt32.astype(np.float64) @ z.T
    out = z - ((d.Einv32.astype(np.float64) @ c).T @ d.Zt32.astype(np.float64))
    want = z - (d.Einv @ (d.AZt[:12] @ z.T)).T @ d.Zt[:12]
    assert np.abs(out - want).max() <= 1e-4 * np.abs(want).max()
    a, _ = d.coarse(z)
    assert a.shape == (1, 12)


def test_the_tolerance_of_the_extension_is_one_named_constant_used_in_one_place():
    """Decision of P2d-D3h: the one new number is `deflation.EXTENSION_TOL`; the model passes none, and the chunk and the stagnation rule are the existing numbers.
    The scan covers the whole `topopt` package and `topopt_mcp` (the syntax tree: a comment or a docstring does not count): the name occurs in `deflation.py`
    twice, the definition and the one use in `Deflation.extend`, and nowhere else (review m9: the model read it a second time for the `run.log` line)."""
    import ast
    import pathlib

    import topopt
    from topopt.gpu_mg import deflation, model
    root = pathlib.Path(topopt.__file__).resolve().parents[1]
    hits = []
    for pkg in ("topopt", "topopt_mcp"):
        for f in sorted((root / pkg).rglob("*.py")):
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Name) and node.id == "EXTENSION_TOL") or (isinstance(node, ast.Attribute) and node.attr == "EXTENSION_TOL"):
                    hits.append(f"{pkg}/{f.relative_to(root / pkg)}")
    assert hits == ["topopt/gpu_mg/deflation.py"] * 2, hits                  # the definition and the one use
    assert "EXTENSION_TOL" in inspect.getsource(deflation.Deflation.extend) and deflation.EXTENSION_TOL == 1e-4
    assert "tol" not in inspect.signature(model.GpuSolver._set_deflation).parameters
    call = re.search(r"d\.extend\((.*?)\)\n", inspect.getsource(model.GpuSolver._set_deflation)).group(1)
    assert call == "self.host, g.precondition, g.nr, self.params.max_iter, engine.STAGNATION_FALL"


# --- the engine ----------------------------------------------------------------------------------------------------------------------------------


@opencl
def test_precondition_is_the_vcycle_of_the_device_and_leaves_the_engine_ready_for_a_solve(tmp_path, dev):
    need_case("C", dev)
    m, rhos, u_refs = soft_dumbbell(tmp_path, 1e-9, clamp=True, **SMALL)
    try:
        s, g = m.solver, m.solver.engine
        s.set_design(m.stiffness(rhos[0]))
        rng = np.random.default_rng(0)
        free = ~s.host.levels[0].fix.reshape(-1).astype(bool)
        assert (~free).any()                                                       # the clamp
        r = rng.standard_normal((g.nr, g.n3)) * free
        z = g.precondition(r)
        assert z.shape == r.shape and z.dtype == np.float64 and np.isfinite(z).all() and np.abs(z[:, ~free]).max() == 0
        assert np.array_equal(z, g.precondition(r))                              # no state: the same answer twice
        y = np.zeros((s.host.levels[0].nn, 3))
        for c in range(g.nr):
            s.host.apply_fine(z[c].reshape(-1, 3), y)
            assert np.linalg.norm(r[c] - y.reshape(-1)) < np.linalg.norm(r[c])      # one V-cycle reduces the residual of a random vector
            assert float(r[c] @ z[c]) > 0
        with pytest.raises(ValueError, match="precondition: the residuals have shape"):
            g.precondition(r[:, :-3])
        u = s.solve(m.f_red)                                                       # the solve after the calls is right
        assert rel_err(u, u_refs[0]) <= 1e-5
    finally:
        m.solver.free()


@opencl
def test_z_follows_the_design_the_device_gets_the_new_Z_AZ_and_Einv_and_the_second_solve_uses_them(tmp_path, dev):
    """Two designs with other densities: the harmonic columns, `AZ` and `E` change; the device buffers hold the new numbers (read back); both solves are right
    against the bordered `splu` of their own design."""
    import pyopencl as cl
    need_case("C", dev)
    m, rhos, u_refs = soft_dumbbell(tmp_path, 1e-9, rhos=(1.0, 1.25), **SMALL)
    try:
        s, g, d = m.solver, m.solver.engine, m.solver.defl
        got = []
        for rho, u_ref in zip(rhos, u_refs):
            s.set_design(m.stiffness(rho))
            Zd = np.empty((d.dk, g.n3), np.float32)
            AZd = np.empty_like(Zd)
            Ed = np.empty((d.dk, d.dk), np.float32)
            cl.enqueue_copy(g.q, Zd, g.Zd)
            cl.enqueue_copy(g.q, AZd, g.AZd)
            cl.enqueue_copy(g.q, Ed, g.Einv_d)
            g.q.finish()
            assert np.array_equal(Zd, d.Zt32) and np.array_equal(AZd, d.AZt32) and np.array_equal(Ed, d.Einv32)
            got.append((d.Zt.copy(), d.AZt.copy(), d.E.copy()))
            u = s.solve(m.f_red)
            assert rel_err(u, u_ref) <= 1e-5
            s.end_design()
        (Z1, A1, E1), (Z2, A2, E2) = got
        assert Z1.shape[0] == Z2.shape[0] == 18                                    # 6 rigid + 12 harmonic columns
        assert np.array_equal(Z1[:6], Z2[:6])                                      # the rigid columns do not depend on the design
        assert np.abs(Z1[6:] - Z2[6:]).max() > 1e-3                                  # the harmonic columns follow the density
        assert np.abs(A1 - A2).max() > 1e-3 * np.abs(A1).max() and np.abs(E1 - E2).max() > 1e-3 * np.abs(E1).max()
    finally:
        m.solver.free()


@opencl
def test_the_records_and_the_facts_say_cold_then_warm_and_the_extension_gets_cheaper(tmp_path, dev):
    """The record of each design has `deflation.extension` (iterations, chunks, seconds, warm or cold, the largest residual, why it stopped) and the times
    `defl_ext` and `defl_upload`; `GpuSolver.facts()` has the columns, the last extension, the totals and the memory of `Z` and `AZ` on the device and on the host."""
    from topopt.gpu_mg import deflation
    from topopt.snapshot import RunWriter
    need_case("C", dev)
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    m, rhos, _ = soft_dumbbell(tmp_path, 1e-9, rhos=(1.0, 1.1, 1.1), out=out, **SMALL)
    try:
        s = m.solver
        for rho in rhos:
            m.solve(m.stiffness(rho))
        recs = s.records
        exts = [r["deflation"]["extension"] for r in recs]
        assert [e["warm"] for e in exts] == [False, True, True] and all(e["converged"] and e["tol"] == deflation.EXTENSION_TOL for e in exts)
        assert exts[1]["iterations"] < exts[0]["iterations"] / 2 and exts[2]["iterations"] <= exts[1]["iterations"]       # the same design again: nothing to do
        assert exts[2]["iterations"] == 0 and exts[2]["rel_max"] <= exts[2]["tol"]
        for r, e in zip(recs, exts):
            assert r["deflation"]["on"] and r["deflation"]["columns"] == 18 and r["deflation"]["harmonic_columns"] == 12 and r["deflation"]["rigid_columns"] == 6
            assert e["columns"] == 12 and len(e["iterations_per_block"]) == -(-12 // s.engine.nr) and sum(e["iterations_per_block"]) == e["iterations"] and (e["chunks"] == 0) == (e["iterations"] == 0)
            assert set(e["stopped"]) == {"tol"} and e["seconds"] >= 0
            assert r["setup_parts_s"]["defl_ext"] >= 0 and r["setup_parts_s"]["defl_upload"] >= 0
            assert r["setup_s"] >= r["setup_parts_s"]["defl_ext"] + r["setup_parts_s"]["defl"] and r["upload_s"] >= r["setup_parts_s"]["defl_upload"]
        json.dumps(recs)                                                           # the record is JSON
        f = s.facts()["deflation"]
        assert f["on"] and f["columns"] == 18 and f["rigid_columns"] == 6 and f["harmonic_columns"] == 12 and f["patches"] == 2 and f["columns_per_patch"] == [6, 6]
        assert f["extension"] == s.defl.ext and f["extension_total"]["designs"] == 3 and f["extension_total"]["warm"] == 2
        assert f["extension_total"]["iterations"] == sum(e["iterations"] for e in exts) and f["extension_tol"] == deflation.EXTENSION_TOL
        g = s.engine
        assert f["device_bytes"] == g.nbytes_parts["defl"] == (2 * g.dk * g.n3 + g.dk * g.dk + g.nr * g.dk * g.G + g.nr * g.dk) * 4
        assert f["host_bytes"] == s.defl.dk * s.defl.n3 * 24 + 2 * 12 * s.defl.n3 * 8 + s.defl.dk ** 2 * 4
    finally:
        m.solver.free()
    log = (tmp_path / "run" / "run.log").read_text()
    assert "12 harmonic column(s) [6, 6] of 2 bonded patch(es)" in log and "18 columns in all" in log
    mb = lambda n: f"{n / 2**20:.1f}"
    assert (f"the device holds {mb(f['device_bytes'])} MB for the deflation (Z, AZ, E^-1 and the work buffers of its three kernels), the host {mb(f['host_bytes'])} MB "
            "(Z and AZ in 64-bit and 32-bit float, the prescribed values, the harmonic columns)") in log                  # review n3: what the byte counts hold
    first = [x for x in log.splitlines() if "deflation of the first design: extension of 12 harmonic column(s) of 2 bonded patch(es), cold start" in x]
    assert len(first) == 1 and "PCG iterations in" in first[0] and "largest relative residual" in first[0] and "against 0.0001" in first[0]
    end = [x for x in log.splitlines() if "extension of the harmonic columns over the run: 3 design(s), 2 warm" in x]
    assert len(end) == 1 and "0 above the tolerance" in end[0]
    assert "stopped above its tolerance" not in log


@opencl
def test_the_run_log_gives_the_tolerance_that_the_extension_used_and_not_the_constant(tmp_path, monkeypatch, dev):
    """Review m9. The extension runs with another tolerance than `EXTENSION_TOL` (1e-3: the caller passes it). Every tolerance in `run.log`, the record and the facts
    is the one that the extension used. On 0843b8b the setup line of the run gave the constant (0.0001) that the extension did not use."""
    from topopt.gpu_mg import deflation
    from topopt.snapshot import RunWriter
    need_case("C", dev)
    real = deflation.Deflation.extend
    monkeypatch.setattr(deflation.Deflation, "extend", lambda self, host, precond, block, maxit, fall, tol=None: real(self, host, precond, block, maxit, fall, tol=1e-3))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    m, rhos, _ = soft_dumbbell(tmp_path, 1e-9, rhos=(1.0, 1.1), out=out, **SMALL)
    try:
        for rho in rhos:
            m.solve(m.stiffness(rho))
        recs, f = m.solver.records, m.solver.facts()["deflation"]
    finally:
        m.solver.free()
    assert [r["deflation"]["extension"]["tol"] for r in recs] == [1e-3, 1e-3] and f["extension_tol"] == 1e-3
    log = (tmp_path / "run" / "run.log").read_text()
    assert "0.0001" not in log and "1e-04" not in log                              # the constant is not in the log
    first = [x for x in log.splitlines() if "deflation of the first design: extension of 12 harmonic column(s)" in x]
    assert len(first) == 1 and "against 0.001" in first[0]
    assert "0 above the tolerance 0.001, largest final relative residual" in log     # the end-of-run line names the tolerance of the extension
    setup = [x for x in log.splitlines() if "deflation of the rigid modes in the inner PCG is on" in x]
    assert len(setup) == 1 and "the line of the first design gives the tolerance that the extension used" in setup[0]      # before any extension there is none to name


@opencl
def test_the_stress_report_writes_one_line_to_run_log_for_each_cold_extension_and_no_other_output(tmp_path, dev):
    """Review m4. The stress report solves two designs (grey and thresholded); each call of `solve_design` builds a new solver and makes a cold extension. Each
    extension is one line of `run.log` with its iterations and seconds, and the line comes before the solve (the thresholded design of this tiny part does not
    solve: the cost is paid and the line is there). No new key in `metrics.json`. On 0843b8b the run.log had no line for them."""
    from topopt.cli import main
    need_case("C", dev)
    dumbbell(tmp_path, "s", **SMALL, mutate=with_a_stress_limit)
    out = tmp_path / "out"
    assert main(["run", str(tmp_path / "s" / "c.yaml"), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text().splitlines()
    lines = [x for x in log if x.startswith("stress report, ")]
    assert [x.split(": ")[0] for x in lines] == ["stress report, grey design", "stress report, thresholded design"], lines
    for x in lines:
        assert re.search(r"extension of the 12 harmonic column\(s\) of the deflation for this one solve, cold start: \d+ PCG iterations in \d+ chunk\(s\), "
                         r"\d+\.\d s, largest relative residual \d\.\de-\d+ against 0\.0001$", x), x
    assert len([x for x in log if "extension of the harmonic columns over the run:" in x]) == 1          # the line of the optimizer run is the same as before
    stress = json.loads((out / "result" / "metrics.json").read_text())["stress"]
    assert not [k for k in stress if "extension" in k or "deflation" in k or "harmonic" in k]            # no key for it in the JSON output
    assert not [k for k in stress["cases"]["down"] if "extension" in k or "deflation" in k or "harmonic" in k]


@opencl
def test_compute_stress_gives_no_line_in_case_b_and_none_without_a_writer(tmp_path, dev):
    """No joint term in the extension of case B: the stress solve of case B has no extension, so no line. Without a writer (`log` None) nothing is written."""
    from topopt.refine.stress import compute_stress
    need_case("B", dev)
    prob = dumbbell(tmp_path, "b", case_name="B", **SMALL)
    snap = np.full(int(np.prod(prob.shape)), 255, dtype=np.uint8)
    lines = []
    got = compute_stress(prob, snap, log=lines.append)
    assert lines == [] and got["vm_cases"].shape[0] == 1
    C = dumbbell(tmp_path, "c", case_name="C", **SMALL)
    compute_stress(C, snap)                                                        # a writer is not needed
    compute_stress(C, snap, log=lines.append)
    assert len(lines) == 1 and "cold start" in lines[0] and "12 harmonic column(s)" in lines[0]


@opencl
def test_free_releases_the_host_arrays_of_the_deflation_too(tmp_path, dev):
    """Review m7. `GpuSolver.free` released the device buffers but kept the host arrays of the deflation (Z and AZ in two types, the prescribed values, the
    harmonic columns, E): 357.5 MiB at 1.0 mm on the two-joint part, until the solver was collected. The engine holds the `Deflation` too, so the arrays are
    dropped in the object. No array object lives on after `free` (weak references), the facts stay, and a second call does nothing."""
    import gc
    import weakref
    need_case("C", dev)
    m, rhos, _ = soft_dumbbell(tmp_path, 1e-9, **SMALL)
    s = m.solver
    m.solve(m.stiffness(rhos[0]))
    d = s.defl
    names = ["Zt", "Zt32", "AZt", "AZt32", "U", "H", "E", "Einv", "Einv32"]
    refs = {n: weakref.ref(getattr(d, n)) for n in names}
    nbytes, summary = d.host_bytes(), d.summary()
    assert nbytes > 0 and all(r() is not None for r in refs.values()) and d.ready
    s.free()
    gc.collect()
    assert [n for n, r in refs.items() if r() is not None] == []                    # no array object lives on
    assert all(getattr(d, n) is None for n in names) and not d.ready
    assert d.host_bytes() == nbytes and d.summary() == summary and d.ext["converged"] and d.ext_total["designs"] == 1     # the facts stay
    s.free()                                                                       # a second call does nothing


@opencl
def test_an_extension_that_stops_above_its_tolerance_is_a_fact_in_the_log_and_the_record_and_the_solve_is_still_right(tmp_path, dev):
    """A chunk of 10 iterations (`GpuParams.max_iter`) is too short for the extension of this part: the residual falls by less than `STAGNATION_FALL` over two
    chunks before the tolerance, in both designs (measured on pocl: the largest residual 1.3e-2 and 3.6e-3 against 1e-4). The log has one line for the run and the
    count at the end, the record has `converged` False, and the solve is accepted (any `Z` is right; the refinement needs a few more steps)."""
    from topopt.gpu_mg import deflation
    from topopt.gpu_mg.model import GpuParams
    from topopt.snapshot import RunWriter
    need_case("C", dev)
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    m, rhos, u_refs = soft_dumbbell(tmp_path, 1e-7, rhos=(1.0, 1.05), params=GpuParams(tol=1e-6, max_iter=10, min_free=300, backward_error=0.0), out=out, **STAG)
    try:
        us = [m.solve(m.stiffness(r)) for r in rhos]
        recs = m.solver.records
        for r in recs:
            e = r["deflation"]["extension"]
            assert not e["converged"] and "stagnation" in e["stopped"] and e["rel_max"] > deflation.EXTENSION_TOL and e["tol"] == deflation.EXTENSION_TOL
            assert r["iterations"][0]["accepted_by"] == "tol"
        assert all(rel_err(u, ref) <= 1e-5 for u, ref in zip(us, u_refs))
        assert m.solver.facts()["deflation"]["extension_total"]["above_tol"] == 2
    finally:
        m.solver.free()
    log = (tmp_path / "run" / "run.log").read_text()
    assert log.count("the extension of the harmonic columns stopped above its tolerance") == 1 and "stagnation" in log
    assert "still valid for the deflation" in log and "2 above the tolerance 0.0001" in log


@opencl
def test_d3_alone_stops_on_the_dumbbell_and_the_harmonic_columns_solve_it(tmp_path, monkeypatch, dev):
    """Case C, beam over bond stiffness 1e-9, a slender bar with a wide bonded plate at each end (8,817 free dofs). With the harmonic columns the solve is
    accepted by the tolerance in at most 5 refinement steps. The rigid columns of D3 alone (`patch_nodes` gives no patch) end in the usual `SolveFailure`
    whose text names the rigid modes and not the harmonic columns. All asserts hold on every device.

    The inner iteration count of a step is NOT asserted. The inner relative residual of the first step has a floor near the inner tolerance 1e-2, and a
    little change of the FP64 extension (the host thread count) moves the floor to one side of 1e-2 or the other. So the first step can stop at the cap of a
    step (300 iterations) on any device. Measured with the test part, one run for each line (review of f919f04; the Arc: a run of d577246):

        device             inner iterations of the 4 steps   total   refine_steps   true_rel
        pocl, 4 threads    17, 16, 22, 13                      68    4              2.5e-7 to 3.2e-7
        pocl, 2 threads    300, 12, 24, 17                    353    4              same range
        pocl, 1 thread     300, 15, 300, 24                   639    4              same range
        Intel Arc          300, 13, 19, 15                    347    4              4.5e-7

    The error against the bordered `splu` was 4.9e-8 to 7.1e-8 on pocl. In all these runs the extension converged and was cold. D3 alone (the second half)
    ends with 12 steps of 300 iterations and `true_rel` 9.5e-4 (pocl, 4 threads), 1.14e-3 (2 threads) and 6.3e-4 (1 thread): at least 6 times the bound
    1e-4 of the test. The residual of D3 alone on the Arc was not recorded (the half passed there with the asserts below)."""
    from topopt.gpu_mg import deflation
    from topopt.gpu_mg.model import SolveFailure
    need_case("C", dev)
    m, rhos, u_refs = soft_dumbbell(tmp_path, 1e-9, **BIG)
    try:
        u = m.solve(m.stiffness(rhos[0]))
        rec = m.solver.records[-1]["iterations"][0]
        ext = m.solver.records[-1]["deflation"]["extension"]
    finally:
        m.solver.free()
    assert rec["accepted_by"] == "tol" and rec["true_rel"] <= 1e-6 and rec["refine_steps"] <= 5, rec          # accepted, within the target, in few steps
    assert rel_err(u, u_refs[0]) <= 1e-5 and ext["converged"] and not ext["warm"]
    monkeypatch.setattr(deflation, "patch_nodes", lambda host: [])
    sub = tmp_path / "d3"
    sub.mkdir()
    m, rhos, u_refs = soft_dumbbell(sub, 1e-9, **BIG)
    try:
        assert m.solver.defl.n_harm == 0 and m.solver.defl.dk == 6
        with pytest.raises(SolveFailure) as e:
            m.solve(m.stiffness(rhos[0]))
    finally:
        m.solver.free()
    its = e.value.its
    assert its["failed"], its                                                                               # D3 alone does not solve it
    assert its["refine_steps"] == 12 and not its["true_rel"] <= 1e-4, its                                   # 12 steps; the residual is not within 1e-4 (a nan is not either)
    assert "the inner PCG deflated the rigid modes (1 body, 6 columns) and the host solved them in FP64" in str(e.value)
    assert "harmonic" not in str(e.value)


@opencl
def test_a_part_with_no_patch_column_has_the_deflation_of_d3_with_z_copied_once(tmp_path, monkeypatch, dev):
    """`patch_nodes` gives no patch: the harmonic columns are absent (D3 as it was), `Z` is copied to the device once, the record has no extension."""
    import pyopencl as cl
    from topopt.gpu_mg import deflation, engine
    need_case("C", dev)
    monkeypatch.setattr(deflation, "patch_nodes", lambda host: [])
    copies = []
    real = engine.cl.enqueue_copy

    def spy(q, dst, src, *a, **k):
        copies.append(dst)
        return real(q, dst, src, *a, **k)
    m, rhos, _ = soft_dumbbell(tmp_path, 1e-5, rhos=(1.0, 1.1), **SMALL)
    monkeypatch.setattr(engine.cl, "enqueue_copy", spy)
    try:
        s, g, d = m.solver, m.solver.engine, m.solver.defl
        assert d.n_harm == 0 and d.dk == d.n_rigid == 6 and not d.z_changes and g.dk == 6
        for rho in rhos:
            s.set_design(m.stiffness(rho))
            s.end_design()
        assert sum(1 for x in copies if x is g.Zd) == 1 and sum(1 for x in copies if x is g.AZd) == 2        # Z once, AZ for each design
        assert all("extension" not in r["deflation"] and r["deflation"]["harmonic_columns"] == 0 for r in s.records)
        assert all("defl_ext" not in r["setup_parts_s"] for r in s.records)
        assert s.facts()["deflation"]["extension"] is None
    finally:
        monkeypatch.undo()
        m.solver.free()


@opencl
def test_a_clamp_cuts_the_harmonic_columns_at_the_fixed_dofs_and_the_part_is_solved(tmp_path, dev):
    need_case("C", dev)
    m, rhos, u_refs = soft_dumbbell(tmp_path, 1e-9, clamp=True, **SMALL)
    try:
        s, d = m.solver, m.solver.defl
        u = m.solve(m.stiffness(rhos[0]))
        fixed = s.host.levels[0].fix.reshape(-1).astype(bool)
        assert fixed.any() and d.n_rigid == 6 and d.n_harm == 12 and d.dk == 18                      # the rigid columns are cut at the clamp (D3); the patches still move
        assert np.abs(d.H[:, fixed]).max() == 0 and np.abs(d.Zt[:, fixed]).max() == 0 and np.abs(d.AZt[:, fixed]).max() == 0
        assert rel_err(u, u_refs[0]) <= 1e-5 and s.records[-1]["iterations"][0]["accepted_by"] == "tol"
    finally:
        m.solver.free()


@opencl
def test_case_c_solves_a_part_with_a_joint_whose_bonded_dofs_are_all_fixed(tmp_path, dev):
    """Review m1 on the device: the first joint has an empty patch (`joint_voxel_between_fixed_voxels`). The solver builds (it raised `IndexError` before), the
    deflation has the 6 rigid columns and the 6 harmonic columns of the second patch only, and the solve is accepted by the tolerance and right."""
    need_case("C", dev)
    m, rhos, u_refs = soft_dumbbell(tmp_path, 1e-9, mutate=joint_voxel_between_fixed_voxels, **SMALL)
    try:
        u = m.solve(m.stiffness(rhos[0]))
        d, rec = m.solver.defl, m.solver.records[-1]
        assert d.per_patch == [0, 6] and d.n_harm == 6 and d.dk == 12 and rec["deflation"]["on"] and rec["deflation"]["columns"] == 12
        assert rec["iterations"][0]["accepted_by"] == "tol" and rel_err(u, u_refs[0]) <= 1e-5
        assert m.solver.facts()["deflation"]["columns_per_patch"] == [0, 6]
    finally:
        m.solver.free()


@opencl
def test_a_device_error_in_the_vcycle_of_the_extension_is_a_user_error_with_the_facts(tmp_path, monkeypatch, dev):
    import pyopencl as cl
    from topopt.gpu_mg import engine
    need_case("C", dev)
    m, rhos, _ = soft_dumbbell(tmp_path, 1e-9, **SMALL)

    def broken(self, R):
        raise cl.RuntimeError("clEnqueueNDRangeKernel failed: OUT_OF_RESOURCES")
    monkeypatch.setattr(engine.GpuMG, "precondition", broken)
    with pytest.raises(UserError) as e:
        m.solve(m.stiffness(rhos[0]))
    text = str(e.value)
    assert dev.name in text and "case C" in text and "ran the V-cycle for the harmonic columns of the deflation" in text and "OUT_OF_RESOURCES" in text
    assert "buffers were released" in text and m.solver._freed


@opencl
def test_a_design_whose_extension_is_not_finite_runs_without_the_deflation_and_the_next_design_is_deflated(tmp_path, monkeypatch, dev):
    """The V-cycle gives a value that is not finite in the first design: `DeflationError`, that design runs as before P2d (a pad that case C solves without the
    deflation too), the record and the log say why, and the next design makes its extension again from nothing."""
    from topopt.gpu_mg import engine
    from topopt.snapshot import RunWriter
    need_case("C", dev)
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    m, rhos, u_refs = soft_dumbbell(tmp_path, 1e-3, rhos=(1.0, 1.0), out=out, **SMALL)
    g = m.solver.engine
    real = engine.GpuMG.precondition
    calls = []

    def flaky(self, R):
        calls.append(1)
        return np.full(R.shape, np.nan) if len(calls) == 1 else real(self, R)
    monkeypatch.setattr(engine.GpuMG, "precondition", flaky)
    try:
        u1 = m.solve(m.stiffness(rhos[0]))
        assert not g.defl_active and g.iter_ops == g._ops_plain
        u2 = m.solve(m.stiffness(rhos[1]))
        assert g.defl_active
        recs = m.solver.records
    finally:
        m.solver.free()
    assert recs[0]["deflation"]["on"] is False and "not finite" in recs[0]["deflation"]["why"]
    assert recs[1]["deflation"]["on"] is True and recs[1]["deflation"]["extension"]["warm"] is False        # nothing was stored by the failed extension
    assert rel_err(u1, u_refs[0]) <= 1e-5 and rel_err(u2, u_refs[1]) <= 1e-5
    assert "deflation of the rigid modes and patches is off for this design" in (tmp_path / "run" / "run.log").read_text()
