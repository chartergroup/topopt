"""P2c: per-axis supports (`{file, axes}`, CONTRACTS.md section 26.2) on gpu-mg.

The multigrid keeps one Dirichlet flag for each DOF on every level (`Level.fix`, shape `(nn, 3)`), and the kernels test bit 0, 1 and 2 of the
fix mask one by one. Until P2c the fine level set all 3 flags of a node that had one fixed dof, so `build_fine` was the only place that clamped
a whole node. The tests of part 1 need no OpenCL: they check the tables of the host hierarchy, level by level. The tests of part 2 run the real
engine on the OpenCL device (pocl on the host of the tests).

The cases (cantilever block x 0..20, y 0..10, z 0..10 mm, voxel size 1, 8,000 to 10,000 free dofs; `kind`):

* `roller`: three single-axis supports and nothing else: the wall x -2..0 fixes x, the floor y -2..0 fixes y, the floor z -2..0 fixes z. A node
  where two of them meet gets two fixed dofs, a corner node gets three. The three together block the six rigid-body modes.
* `slider`: the wall x -2..0 fixes y and z (the part slides along x), and a stop x 20..22 fixes x.

`full_clamps` (a `mutate` function) turns every support of a case into a full clamp: the same geometry, the case that a roller or slider case is
compared with.
"""
import json

import numpy as np
import pytest
import scipy.sparse as sps
import yaml
from scipy.sparse.linalg import spsolve

from test_springs import box
from topopt.cli import main
from topopt.config import load_config
from topopt.fe import hex8_ke
from topopt.gpu_mg import hierarchy
from topopt.gpu_mg.hierarchy import HostMG
from topopt.optimize import Model, reduced_system
from topopt.problem import build_problem

KINDS = ("roller", "slider")
FORCE = {"roller": [-0.6, -0.4, -2.0], "slider": [0.6, 0.4, -2.0]}


def fresh(tmp_path, name):
    """A new empty folder `name` under `tmp_path` (a case writes its STL files there)."""
    d = tmp_path / name
    d.mkdir()
    return d


def axes_case(tmp_path, kind, h=1.0, solver="amg", mutate=None, gpu_case=None):
    """The YAML of the case `kind` (see the module text) in the folder `tmp_path`."""
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", (8, 4, 10), (12, 6, 13))
    regions = {"design": "design.stl", "load": {"tip": "tip.stl"}}
    if kind == "roller":
        box(tmp_path / "floor_y.stl", (0, -2, 0), (20, 0, 10))
        box(tmp_path / "floor_z.stl", (0, 0, -2), (20, 10, 0))
        regions["fixed"] = [{"file": "plate.stl", "axes": ["x"]}, {"file": "floor_y.stl", "axes": ["y"]},
                            {"file": "floor_z.stl", "axes": ["z"]}]
    else:
        box(tmp_path / "stop.stl", (20, 0, 0), (22, 10, 10))
        regions["fixed"] = [{"file": "plate.stl", "axes": ["y", "z"]}, {"file": "stop.stl", "axes": ["x"]}]
    d = {"name": kind, "regions": regions,
         "load_cases": [{"name": "a", "forces": [{"region": "tip", "force": FORCE[kind]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 3, "solver": solver}}
    if gpu_case:
        d["settings"]["gpu_case"] = gpu_case
    if mutate:
        mutate(d)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def full_clamps(d):
    """`mutate`: every `fixed` entry becomes a path, a full clamp of all 3 axes."""
    d["regions"]["fixed"] = [e["file"] if isinstance(e, dict) else e for e in d["regions"]["fixed"]]


def prob_of(tmp_path, kind, **kw):
    return build_problem(load_config(axes_case(tmp_path, kind, **kw)))


def host_of(prob, min_free=300):
    edof, aidx, free = reduced_system(prob)
    return HostMG(prob.shape, edof, aidx, prob.fixed_dofs, prob.h, prob.config.material.nu, min_free=min_free), free


def reference(tmp_path, kind, h=1.0, seed=3, mutate=None):
    """`(model, rho, K, U)`: the amg model of the case (its assembled reduced matrix), a random design, and the direct solve."""
    m = Model(prob_of(fresh(tmp_path, "ref"), kind, h=h, mutate=mutate))
    rho = 0.1 + 0.8 * np.random.default_rng(seed).uniform(size=m.n_design)
    K = m.stiffness(rho).tocsc()
    U = spsolve(K, m.f_red).reshape(m.n_free, -1)
    m.solver.free()
    return m, rho, K, U


# --- part 1: the host hierarchy, no OpenCL --------------------------------------------------------------------------------


def grid_dofs_of(L, mask):
    """The grid dof numbers of the flags set in `mask` (nn, 3) of the level `L`."""
    n, a = np.nonzero(mask)
    return np.sort(3 * L.nidx[n].astype(np.int64) + a)


@pytest.mark.parametrize("kind", KINDS)
def test_level_0_fixes_exactly_the_fixed_dofs_of_the_problem_and_a_node_may_be_partly_fixed(tmp_path, kind):
    prob = prob_of(tmp_path, kind)
    host, free = host_of(prob)
    L0 = host.levels[0]
    assert prob.config.has_partial_supports() and L0.fix.shape == (L0.nn, 3)
    assert np.array_equal(grid_dofs_of(L0, L0.fix), prob.fixed_dofs)                # the dofs of the problem, not the nodes
    per_node = L0.fix.sum(axis=1)
    assert ((per_node > 0) & (per_node < 3)).any()                                   # a node with some dofs fixed and some free
    assert (per_node == 0).any() and L0.fix.sum() == prob.fixed_dofs.size
    assert 3 * L0.nn - L0.fix.sum() == free.size                                     # the free dofs of the model are the free dofs of level 0
    fmap = L0.lookup[free // 3].astype(np.int64) * 3 + free % 3
    assert (L0.fix.reshape(-1)[fmap] == 0).all()


def test_a_fixed_dof_on_a_node_without_an_active_element_is_skipped_as_before(tmp_path):
    """A fixed dof whose node belongs to no active element has no place in the level: the old node-level rule skipped it, and so does the per-dof rule."""
    prob = prob_of(tmp_path, "roller")
    edof, aidx, _free = reduced_system(prob)
    host, _ = host_of(prob)
    L0 = host.levels[0]
    outside = int(np.flatnonzero(L0.lookup < 0)[0])                  # a node of the box that no active element uses
    extra = np.sort(np.append(prob.fixed_dofs, [3 * outside, 3 * outside + 2]))
    more = HostMG(prob.shape, edof, aidx, extra, prob.h, prob.config.material.nu, min_free=300)
    assert np.array_equal(more.levels[0].fix, L0.fix)


def node_xyz(L, origin, ids=None):
    """The coordinates (mm) of the nodes `ids` (default: all) of the level `L`."""
    nzc1, nyc1 = L.nodedims[2], L.nodedims[1]
    g = L.nidx.astype(np.int64) if ids is None else L.nidx[ids].astype(np.int64)
    return np.asarray(origin) + np.stack([g // (nzc1 * nyc1), (g // nzc1) % nyc1, g % nzc1], axis=1) * L.h


def test_the_axes_of_the_roller_case_are_the_axes_of_its_supports(tmp_path):
    """The wall fixes x, floor y fixes y, floor z fixes z. Count the fixed dofs on each axis, and the nodes where supports meet: 40 nodes carry
    two fixed dofs (the three edges of the corner, 10 + 10 + 20 nodes), and the corner node (0, 0, 0) carries three."""
    prob = prob_of(tmp_path, "roller")
    host, _ = host_of(prob)
    L0 = host.levels[0]
    assert L0.fix.sum(axis=0).tolist() == [3 * 11 * 11, 21 * 3 * 11, 21 * 11 * 3]                  # the nodes of each support block
    assert np.bincount(L0.fix.sum(axis=1), minlength=4).tolist()[1:] == [1749 - 2 * 40 - 3, 40, 1]
    corner = np.flatnonzero(L0.fix.sum(axis=1) == 3)
    assert np.allclose(node_xyz(L0, prob.origin, corner), [[0.0, 0.0, 0.0]])
    on_wall = L0.fix[:, 0].astype(bool)                                                            # x is fixed exactly where x <= 0 (the wall block)
    xyz = node_xyz(L0, prob.origin)
    assert np.array_equal(on_wall, (xyz[:, 0] <= 1e-9) & (xyz[:, 1] >= -1e-9) & (xyz[:, 2] >= -1e-9))


@pytest.mark.parametrize("kind", KINDS)
def test_a_full_clamp_gives_the_node_level_flags_as_before(tmp_path, kind):
    """No change for the default path: the fixed nodes of a full clamp have all 3 flags, on every level."""
    prob = prob_of(tmp_path, kind, mutate=full_clamps)
    assert not prob.config.has_partial_supports()
    host, _ = host_of(prob)
    for L in host.levels:
        assert set(np.unique(L.fix.sum(axis=1))) <= {0, 3}
    L0 = host.levels[0]
    fixed_nodes = np.unique(prob.fixed_dofs // 3)
    assert np.array_equal(np.sort(L0.nidx[L0.fix.sum(axis=1) == 3].astype(np.int64)), fixed_nodes)


def neighbour_or(F, C, shape):
    """The "any" rule for each dof, by loops over the grid coordinates (no vectors): the flag of a coarse node is set when a fine node within one fine step of it
    (the 27 positions around twice its coordinates) has the flag set."""
    nx, ny, nz = shape
    pos = {int(g): n for n, g in enumerate(F.nidx)}
    nzc1, nyc1 = C.nodedims[2], C.nodedims[1]
    out = np.zeros((C.nn, 3), np.uint8)
    for n, g in enumerate(C.nidx.astype(np.int64)):
        k, j, i = g % nzc1, (g // nzc1) % nyc1, g // (nzc1 * nyc1)
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    fi, fj, fk = 2 * i + da, 2 * j + db, 2 * k + dc
                    if 0 <= fi <= nx and 0 <= fj <= ny and 0 <= fk <= nz:
                        f = pos.get((fi * (ny + 1) + fj) * (nz + 1) + fk)
                        if f is not None:
                            out[n] |= F.fix[f]
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_the_coarse_levels_fix_a_dof_when_a_fine_node_next_to_it_fixes_that_dof(tmp_path, kind):
    """The rule of the coarsening is the "any" rule for each dof, not for each node: a coarse node next to a roller on x does not get y and z."""
    host, _ = host_of(prob_of(tmp_path, kind))
    assert host.n_levels >= 3
    for l in range(1, host.n_levels):
        F, C = host.levels[l - 1], host.levels[l]
        assert np.array_equal(C.fix, neighbour_or(F, C, F.shape)), l
        per_node = C.fix.sum(axis=1)
        assert ((per_node > 0) & (per_node < 3)).any(), l                      # a partly fixed node exists on every level


def rigid_modes(L, origin_mm=(0.0, 0.0, 0.0)):
    """The six rigid-body modes (3 nn, 6) of a level, from the node coordinates (mm, in units of the voxel size of the level)."""
    nzc1, nyc1 = L.nodedims[2], L.nodedims[1]
    g = L.nidx.astype(np.int64)
    xyz = np.stack([g // (nzc1 * nyc1), (g // nzc1) % nyc1, g % nzc1], axis=1) * L.h + np.asarray(origin_mm)
    B = np.zeros((L.nn, 3, 6))
    for a in range(3):
        B[:, a, a] = 1.0
    x, y, z = xyz.T
    B[:, 1, 3], B[:, 2, 3] = -z, y            # about x
    B[:, 2, 4], B[:, 0, 4] = -x, z            # about y
    B[:, 0, 5], B[:, 1, 5] = -y, x            # about z
    return B.reshape(3 * L.nn, 6)


@pytest.mark.parametrize("kind", KINDS)
def test_the_fixed_dofs_of_every_level_block_the_six_rigid_body_modes(tmp_path, kind):
    """The coarse matrix has to stay non-singular. The rule is per dof because the transfer acts on each dof alone: a fine dof is zero when the same dof of every parent is zero."""
    host, _ = host_of(prob_of(tmp_path, kind))
    scale = None
    for l, L in enumerate(host.levels):
        B = rigid_modes(L)
        if scale is None:
            scale = 0.5 * float(np.linalg.norm(np.array(L.shape) * L.h))
        B[:, 3:] /= scale
        rows = np.flatnonzero(L.fix.reshape(-1))
        sv = np.linalg.svd(B[rows], compute_uv=False)
        assert sv.size == 6 and sv[-1] > 1e-3 * sv[0], (l, sv)                    # the free modes of a level would show here as a zero singular value


def level_matrix(L, ke_unit, l):
    """The unreduced matrix (3 nn, 3 nn) of level `l` with the element stiffness `L.Ee0` (the host tables, assembled with scipy)."""
    ke = ke_unit * 2.0 ** l
    dofs = (3 * L.enodes[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24).astype(np.int64)
    rows = np.repeat(dofs, 24, axis=1).reshape(-1)
    cols = np.tile(dofs, (1, 24)).reshape(-1)
    vals = (L.Ee0[:, None] * ke.reshape(1, -1)).reshape(-1)
    return sps.coo_matrix((vals, (rows, cols)), shape=(3 * L.nn, 3 * L.nn)).tocsr()


def diagonal_blocks(A, nn):
    """The (nn, 3, 3) diagonal blocks of the sparse matrix `A`."""
    bsr = A.tobsr(blocksize=(3, 3))
    bsr.sum_duplicates()
    rows = np.repeat(np.arange(nn), np.diff(bsr.indptr))
    on = bsr.indices == rows
    out = np.zeros((nn, 3, 3))
    out[rows[on]] = bsr.data[on]
    return out


def test_every_level_applies_the_operator_with_the_fixed_dofs_cut_one_dof_at_a_time(tmp_path):
    """The host operator `matvec`, the diagonal 3 x 3 blocks of the smoother and the dense inverse of the coarsest level, on every level of the roller case,
    equal the matrices that scipy assembles with the fixed rows and columns of each dof cut (a partly fixed node has a 1 x 1 or 2 x 2 free block)."""
    prob = prob_of(tmp_path, "roller")
    host, _free = host_of(prob)
    m_ref, rho, _K, _U = reference(fresh(tmp_path, "r"), "roller")
    host.setup_host(m_ref.element_E(rho))
    ke_unit = hex8_ke(1.0, prob.config.material.nu, prob.h)
    rng = np.random.default_rng(4)
    for l, L in enumerate(host.levels):
        A = level_matrix(L, ke_unit, l)
        fixed = L.fix.reshape(-1).astype(bool)
        keep = np.flatnonzero(~fixed)
        x = rng.standard_normal(3 * L.nn)
        x[fixed] = 0.0
        y = np.zeros((L.nn, 3))
        hierarchy.matvec(x.reshape(-1, 3), y, L.nelem, L.enodes, L.Ee0, L.kel, L.fix)
        ref = A @ x
        assert np.abs(y.reshape(-1)[fixed]).max() == 0.0                                     # zero on the fixed dofs of every axis
        assert np.linalg.norm(y.reshape(-1)[keep] - ref[keep]) <= 1e-12 * np.linalg.norm(ref[keep]), l
        # the smoother: the inverse of the free part of the diagonal block, and 1 on a fixed dof
        blk = diagonal_blocks(A, L.nn)
        fx = L.fix.astype(bool)
        blk[fx[:, :, None] | fx[:, None, :]] = 0.0
        blk[fx[:, 0], 0, 0] = blk[fx[:, 1], 1, 1] = blk[fx[:, 2], 2, 2] = 1.0
        d = L.dinv
        inv = np.stack([np.stack([d[:, 0], d[:, 1], d[:, 2]], 1), np.stack([d[:, 1], d[:, 3], d[:, 4]], 1), np.stack([d[:, 2], d[:, 4], d[:, 5]], 1)], 1)
        partly = int(((L.fix.sum(axis=1) > 0) & (L.fix.sum(axis=1) < 3)).sum())
        assert np.abs(inv @ blk - np.eye(3)).max() < 1e-9 and partly > 0, (l, partly)
    C = host.levels[-1]
    keep = np.flatnonzero(C.fix.reshape(-1) == 0)
    Kc = level_matrix(C, ke_unit, host.n_levels - 1).toarray()[np.ix_(keep, keep)]
    assert np.abs(host.Ainv_full[np.ix_(keep, keep)] @ Kc - np.eye(keep.size)).max() < 1e-8
    fixed = C.fix.reshape(-1).astype(bool)
    assert not host.Ainv_full[fixed].any() and not host.Ainv_full[:, fixed].any()           # zero rows and columns on the fixed dofs


@pytest.mark.parametrize("kind", KINDS)
def test_the_fine_operator_and_the_free_dof_map_of_the_hierarchy_equal_the_model(tmp_path, kind):
    """`apply_fine` on the free dofs of the model is `K u` of the assembled reduced matrix (the FP64 residual of the solve), and it is zero on the fixed dofs."""
    m_ref, rho, K, _ = reference(tmp_path, kind)
    host, free = host_of(m_ref.problem)
    host.setup_host(m_ref.element_E(rho))
    L0 = host.levels[0]
    fmap = L0.lookup[free // 3].astype(np.int64) * 3 + free % 3
    u = np.random.default_rng(1).standard_normal(free.size)
    x = np.zeros(3 * L0.nn)
    x[fmap] = u
    y = np.zeros((L0.nn, 3))
    host.apply_fine(x.reshape(-1, 3), y)
    ref = K @ u
    assert np.linalg.norm(y.reshape(-1)[fmap] - ref) <= 1e-13 * np.linalg.norm(ref)
    mask = np.ones(3 * L0.nn, bool)
    mask[fmap] = False
    assert mask.sum() == m_ref.problem.fixed_dofs.size and np.abs(y.reshape(-1)[mask]).max() == 0.0


def pad_edge_roller(tmp_path, case_name=None):
    """`mutate` for the pad case of `test_springs`: the clamp plate stays, and a wall x 22..24 that fixes only x touches the end of the spring pad
    (x 20..22). The pad nodes of the plane x = 22 (the edges of the pad) are then partly fixed nodes that carry a spring."""
    box(tmp_path / "wall2.stl", (22, 0, 0), (24, 10, 10))

    def mutate(d):
        d["regions"]["fixed"] = ["plate.stl", {"file": "wall2.stl", "axes": ["x"]}]
        if case_name:
            d["settings"]["gpu_case"] = case_name
    return mutate


def test_springs_on_the_nodes_of_a_roller_are_cut_one_dof_at_a_time_on_every_level(tmp_path):
    """The terms of P2b (`Kb`, in `border.py`) are cut on the fixed dofs of each level, one dof at a time: on a node with a spring and a fixed x dof, the
    x row and column of `Kb` are zero and the y and z entries stay. The host operator equals `K + Kb` of the assembled model."""
    from test_springs import prob_of as spring_prob
    from topopt.gpu_mg.border import grid_border
    ref = Model(spring_prob(fresh(tmp_path, "ref"), solver="amg", mutate=pad_edge_roller(tmp_path / "ref")))
    rho = 0.1 + 0.8 * np.random.default_rng(5).uniform(size=ref.n_design)
    K = ref.stiffness(rho).tocsc()
    prob = ref.problem
    edof, aidx, free = reduced_system(prob)
    host = HostMG(prob.shape, edof, aidx, prob.fixed_dofs, prob.h, prob.config.material.nu, min_free=300, border=grid_border(prob))
    host.setup_host(ref.element_E(rho))
    assert host.n_levels >= 3 and prob.config.has_partial_supports()
    for l, lb in enumerate(host.border.levels):
        fixed = host.levels[l].fix[lb.nodes].astype(bool)                                # (nb, 3) the Dirichlet flags of the bonded nodes
        flat = fixed.reshape(-1)
        Kb = lb.Kb.tocsr()
        assert abs(Kb[flat]).sum() == 0 and abs(Kb[:, flat]).sum() == 0, l
        partly = fixed.any(axis=1) & ~fixed.all(axis=1)
        assert partly.any(), l                                                          # a bonded node with some dofs fixed exists on every level
        rows = np.asarray(abs(Kb).sum(axis=1)).reshape(-1, 3)
        if l == 0:
            assert (rows[partly][~fixed[partly]] > 0).all()                             # the free dofs of those nodes keep their spring stiffness
    L0 = host.levels[0]
    fmap = L0.lookup[free // 3].astype(np.int64) * 3 + free % 3
    u = np.random.default_rng(1).standard_normal(free.size)
    x = np.zeros(3 * L0.nn)
    x[fmap] = u
    y = np.zeros((L0.nn, 3))
    host.apply_fine(x.reshape(-1, 3), y)
    ref_y = K @ u
    assert np.linalg.norm(y.reshape(-1)[fmap] - ref_y) <= 1e-13 * np.linalg.norm(ref_y)


# --- part 2: the real engine on the OpenCL device ---------------------------------------------------------------------------

opencl = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1
SOLVE_TOL = 1e-9                          # relative residual of the solves below (the bars are made for it)


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


def need_case(case_name, dev):
    """Skip a test of case B on a device without 64-bit floats."""
    if case_name == "B" and not dev.fp64:
        pytest.skip(f"case B needs 64-bit floats; {dev.name} has none")


def gpu_model(prob, tol=SOLVE_TOL):
    from topopt.gpu_mg.model import GpuModel, GpuParams
    return GpuModel(prob, None, GpuParams(tol=tol, min_free=300, backward_error=0.0))


def gpu_axes_model(tmp_path, kind, case_name, tol=SOLVE_TOL, h=1.0, name="gpu", mutate=None):
    return gpu_model(prob_of(fresh(tmp_path, name), kind, h=h, solver="gpu-mg", gpu_case=case_name, mutate=mutate), tol)


def its_of(m):
    """The record of the last solve: the entry of `iterations` for its first block of columns."""
    return m.solver.records[-1]["iterations"][0]


def count(its):
    """The PCG iterations of a solve record: case B `executed`; case C the largest inner count of a refinement step (and the steps)."""
    return (its["executed"], 0) if "executed" in its else (max(its["inner"]), its["refine_steps"])


def clamp_balance(rf):
    """The sum of the applied load and the force on the fixed dofs of a case (zero for a balanced solve), and the sizes."""
    c = rf["cases"][0]
    return np.array(c["applied"]["force_n"]) + np.array(c["clamp"]["force_n"]), np.abs(c["applied"]["force_n"]).max()


@opencl
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_roller_and_a_slider_case_on_gpu_mg_equal_the_direct_solve_and_the_reactions_match(tmp_path, dev, case_name, kind):
    """The displacement and the compliance on the device equal the direct solve of the reduced matrix within the solve tolerance (relative residual 1e-9).
    The reactions (`reaction_facts`) match the direct solver, and a support counts only the force on its fixed dofs (CONTRACTS.md section 26.2): the
    force on the dofs of one axis balances the load along that axis, and the force on a free dof is zero. Case C is the 32-bit device path."""
    need_case(case_name, dev)
    m_ref, rho, K, Uref = reference(tmp_path, kind)
    m = gpu_axes_model(tmp_path, kind, case_name)
    try:
        host = m.solver.host
        assert host.n_levels >= 3 and m.problem.config.has_partial_supports()
        assert ((host.levels[0].fix.sum(axis=1) > 0) & (host.levels[0].fix.sum(axis=1) < 3)).any()           # the device has partly fixed nodes
        U = m.solve(m.stiffness(rho))
        its = its_of(m)
        got = m.reaction_facts(U, rho)
    finally:
        m.solver.free()
    want = m_ref.reaction_facts(Uref, rho)
    assert U.shape == Uref.shape
    assert np.linalg.norm(U - Uref) <= 1e-7 * np.linalg.norm(Uref)
    assert float(m.f_red[:, 0] @ U[:, 0]) == pytest.approx(float(m_ref.f_red[:, 0] @ Uref[:, 0]), rel=1e-8)
    assert its["accepted_by"] == "tol"
    g, w = got["cases"][0], want["cases"][0]
    for key in ("applied", "clamp"):
        assert np.allclose(g[key]["force_n"], w[key]["force_n"], rtol=1e-6, atol=1e-9)
        assert np.allclose(g[key]["moment_nmm"], w[key]["moment_nmm"], rtol=1e-6, atol=1e-8)
    gap, size = clamp_balance(got)
    assert np.abs(gap).max() <= 1e-7 * size and size > 0.5                          # the load is carried by the fixed dofs, axis by axis
    assert g["force_error"] < 1e-7 and g["moment_error"] < 1e-7 and g["residual_rel"] < 1e-8


@opencl
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_the_device_solution_is_zero_on_the_fixed_dofs_of_every_axis(tmp_path, dev, case_name, kind):
    """The kernels cut the fixed dofs with bit 0, 1 and 2 of the fix mask: the vector that the engine returns is exactly zero on each fixed dof of a partly fixed node."""
    need_case(case_name, dev)
    m_ref, rho, _K, Uref = reference(tmp_path, kind)
    m = gpu_axes_model(tmp_path, kind, case_name)
    try:
        s = m.solver
        s.set_design(m.stiffness(rho))
        Bn = np.zeros((s.nr, s.nn0 * 3))
        for i in range(s.nr):
            Bn[i][s.fmap] = m.f_red[:, 0]
        Xn, _its = s._solve_block(Bn.reshape(s.nr, s.nn0, 3))
        s.end_design()
        fixed = s.host.levels[0].fix.reshape(-1).astype(bool)
        X = np.asarray(Xn).reshape(s.nr, -1)
        assert fixed.sum() == m.problem.fixed_dofs.size and np.abs(X[:, fixed]).max() == 0.0
        assert np.linalg.norm(X[0][s.fmap] - Uref[:, 0]) <= 1e-7 * np.linalg.norm(Uref[:, 0])
        partly = np.flatnonzero((s.host.levels[0].fix.sum(axis=1) > 0) & (s.host.levels[0].fix.sum(axis=1) < 3))
        free_dof_of_partly = [3 * n + a for n in partly for a in range(3) if not s.host.levels[0].fix[n, a]]
        assert np.abs(X[0][free_dof_of_partly]).max() > 0                                # the free dofs of a partly fixed node move
    finally:
        m.solver.free()


@opencl
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_the_pcg_iterations_of_a_roller_case_against_the_same_case_with_full_clamps(tmp_path, dev, capsys, case_name, kind):
    """The iterations are a fact (`pytest -s` prints them). The roller keeps more dofs free than the clamp, and the bar is loose: a rise of the count above
    a few times the clamp count is a finding for the report, and this test then fails so that the number is looked at."""
    need_case(case_name, dev)
    rows = {}
    for label, mutate in (("full clamps", full_clamps), (kind, None)):
        tag = label.replace(" ", "_")
        m_ref, rho, K, Uref = reference(fresh(tmp_path, f"r_{tag}"), kind, mutate=mutate)
        m = gpu_axes_model(fresh(tmp_path, f"g_{tag}"), kind, case_name, mutate=mutate)
        try:
            U = m.solve(m.stiffness(rho))
            assert np.linalg.norm(U - Uref) <= 1e-7 * np.linalg.norm(Uref)
            rows[label] = (count(its_of(m)), m.solver.host.n_levels, m.n_free)
        finally:
            m.solver.free()
    (its_clamp, steps_c), (its_axes, steps_a) = rows["full clamps"][0], rows[kind][0]
    assert its_axes <= 2 * its_clamp + 5 and steps_a <= steps_c + 4, rows     # a rule for each node gives 2.8-4.6x (review m1)
    with capsys.disabled():
        what = "PCG iterations executed" if case_name == "B" else "largest inner PCG count, refinement steps"
        print(f"\nP2c case {case_name}, {what}: full clamps {rows['full clamps'][0]} ({rows['full clamps'][2]} free dofs), "
              f"{kind} {rows[kind][0]} ({rows[kind][2]} free dofs), levels {rows[kind][1]}")


def plate_axes(*axes, case_name=None):
    """`mutate` for the pad cases of `test_springs` and `test_joints`: the wall `plate.stl` fixes only `axes`; `case_name` sets `settings.gpu_case`."""
    def mutate(d):
        d["regions"]["fixed"] = [{"file": "plate.stl", "axes": list(axes)}]
        if case_name:
            d["settings"]["gpu_case"] = case_name
    return mutate


@opencl
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_roller_next_to_springs_on_gpu_mg_equals_the_direct_solve_and_the_reactions_match(tmp_path, dev, case_name):
    """The pad case of `test_springs.py` (a spring pad at the free end, 100 N/mm on each axis) with a wall that fixes x and y only: z is held by the
    springs alone. The border of P2b (`Kb` on every level) and the per-dof flags work together. The direct solve is the reduced `K + Kb`."""
    from test_springs import prob_of as spring_prob
    need_case(case_name, dev)
    ref = Model(spring_prob(fresh(tmp_path, "ref"), solver="amg", mutate=plate_axes("x", "y")))
    rho = 0.1 + 0.8 * np.random.default_rng(3).uniform(size=ref.n_design)
    Uref = spsolve(ref.stiffness(rho).tocsc(), ref.f_red).reshape(ref.n_free, -1)
    ref.solver.free()
    m = gpu_model(spring_prob(fresh(tmp_path, "gpu"), solver="gpu-mg", mutate=plate_axes("x", "y", case_name=case_name)))
    try:
        assert m.problem.config.has_partial_supports() and m.kb is not None and m.solver.host.border.k == 0 and m.solver.host.n_levels >= 3
        U = m.solve(m.stiffness(rho))
        its = its_of(m)
        got = m.reaction_facts(U, rho)
    finally:
        m.solver.free()
    want = ref.reaction_facts(Uref, rho)
    assert np.linalg.norm(U - Uref) <= 1e-7 * np.linalg.norm(Uref)
    assert float(m.f_red[:, 0] @ U[:, 0]) == pytest.approx(float(ref.f_red[:, 0] @ Uref[:, 0]), rel=1e-8)
    assert its["accepted_by"] == "tol"
    g, w = got["cases"][0], want["cases"][0]
    for key in ("applied", "clamp"):
        assert np.allclose(g[key]["force_n"], w[key]["force_n"], rtol=1e-6, atol=1e-9)
        assert np.allclose(g[key]["moment_nmm"], w[key]["moment_nmm"], rtol=1e-6, atol=1e-8)
    assert np.allclose(g["springs"]["pad"]["force_n"], w["springs"]["pad"]["force_n"], rtol=1e-6, atol=1e-9)
    assert g["clamp"]["force_n"][2] == 0.0                                           # the wall has no fixed z dof: it carries no z force
    assert g["springs"]["pad"]["force_n"][2] == pytest.approx(2.0, rel=1e-6)          # the springs carry the whole 2 N along z
    assert g["force_error"] < 1e-7 and g["moment_error"] < 1e-7 and g["residual_rel"] < 1e-8


@opencl
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_slider_next_to_a_joint_on_gpu_mg_equals_the_bordered_direct_solve_and_the_reactions_match(tmp_path, dev, case_name):
    """The pad case of `test_joints.py` (a beam of d 4 mm from the pad to a fixed point) with a wall that fixes y and z only (it slides along x). The
    border with 6 dofs per joint and the per-dof flags work together. The direct solve is the bordered system by `splu`."""
    from test_gpu_mg_border import bordered_reference, close_tree
    from test_joints import prob_of as joint_prob
    from topopt.springs import reaction_facts
    need_case(case_name, dev)
    ref = Model(joint_prob(fresh(tmp_path, "ref"), solver="amg", mutate=plate_axes("y", "z")))
    rho = 0.1 + 0.8 * np.random.default_rng(3).uniform(size=ref.n_design)
    Uref = bordered_reference(ref, rho)
    ref.solver.free()
    m = gpu_model(joint_prob(fresh(tmp_path, "gpu"), solver="gpu-mg", mutate=plate_axes("y", "z", case_name=case_name)))
    try:
        assert m.problem.config.has_partial_supports() and m.kb is not None and m.solver.host.border.k == 6 and m.solver.host.n_levels >= 3
        U = m.solve(m.stiffness(rho))
        its = its_of(m)
        got = m.reaction_facts(U, rho)
    finally:
        m.solver.free()
    edof, _aidx, free = reduced_system(ref.problem)
    want = reaction_facts(ref.problem, edof, ref.element_E(rho), ref.ke, free, Uref)
    assert np.linalg.norm(U - Uref) <= 1e-6 * np.linalg.norm(Uref)
    assert float(m.f_red[:, 0] @ U[:, 0]) == pytest.approx(float(ref.f_red[:, 0] @ Uref[:, 0]), rel=1e-7)
    assert its["accepted_by"] == "tol"
    g, w = got["cases"][0], want["cases"][0]
    close_tree(g["joints"], w["joints"], rtol=1e-5, atol=1e-8)                      # wrenches, body motion, bond energy share, nodal force
    for key in ("applied", "clamp", "sum_force_n", "sum_moment_nmm"):
        close_tree(g[key], w[key], rtol=1e-5, atol=1e-7)
    assert abs(np.array(g["joints"]["pad"]["part"]["force_n"])).max() > 1e-3          # the joint carries a force: the test is not empty
    assert g["clamp"]["force_n"][1] != 0.0 and g["force_error"] < 1e-6 and g["moment_error"] < 1e-6


@opencl
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_a_short_run_with_a_per_axis_support_finishes_on_gpu_mg_and_check_has_no_refusal(tmp_path, dev, case_name, kind):
    """`topopt check` has no `cannot start here` warning, and `topopt run` solves 4 passes on the device. The compliance of every pass equals the run of
    the default solver when that one is direct (else a looser bar: AMG is an iteration)."""
    from topopt import cli
    from topopt.solver_choice import resolve_solver
    need_case(case_name, dev)

    def settings(d):
        d["settings"].update(iterations=4, gpu_tol=1e-8)

    p = axes_case(fresh(tmp_path, "gpu"), kind, solver="gpu-mg", gpu_case=case_name, mutate=settings)
    d = cli.check_data(p)
    assert d["solver"]["resolved"] == "gpu-mg" and not [w for w in d["warnings"] if "cannot start here" in w or "clamps all 3 dofs" in w]
    out = tmp_path / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert len(hist) >= 4 and all(np.isfinite(r["compliance"]) for r in hist)
    assert json.loads((out / "run.json").read_text())["solver"] == "gpu-mg"
    log = (out / "run.log").read_text()
    assert "gpu-mg: device" in log and f"case {case_name}" in log
    recs = [json.loads(line) for line in (out / "gpu_mg.jsonl").read_text().splitlines()]
    assert recs and all(rec["case"] == case_name for rec in recs)
    ref_out = tmp_path / "ref_out"
    q = axes_case(fresh(tmp_path, "auto"), kind, solver="auto", mutate=settings)
    assert main(["run", str(q), "--out", str(ref_out), "--quiet"]) == 0
    ref = [json.loads(line) for line in (ref_out / "history.jsonl").read_text().splitlines()]
    c_gpu = np.array([r["compliance"] for r in hist])
    c_ref = np.array([r["compliance"] for r in ref])
    direct = resolve_solver("auto") in ("cholesky", "pardiso")
    assert c_gpu.shape == c_ref.shape and np.max(np.abs(c_gpu - c_ref) / c_ref) < (1e-7 if direct else 1e-3)


@opencl
@pytest.mark.parametrize("case_name", ["B", "C"])
def test_springs_on_the_nodes_of_a_roller_on_gpu_mg_equal_the_direct_solve(tmp_path, dev, case_name):
    """A spring pad that a roller wall touches (see `pad_edge_roller`): the nodes of the pad edge carry a spring and a fixed x dof. The device solve of
    `K + Kb` equals the direct solve, and the reactions match."""
    from test_springs import prob_of as spring_prob
    need_case(case_name, dev)
    ref = Model(spring_prob(fresh(tmp_path, "ref"), solver="amg", mutate=pad_edge_roller(tmp_path / "ref")))
    rho = 0.1 + 0.8 * np.random.default_rng(3).uniform(size=ref.n_design)
    Uref = spsolve(ref.stiffness(rho).tocsc(), ref.f_red).reshape(ref.n_free, -1)
    ref.solver.free()
    gdir = fresh(tmp_path, "gpu")
    m = gpu_model(spring_prob(gdir, solver="gpu-mg", mutate=pad_edge_roller(gdir, case_name)))
    try:
        U = m.solve(m.stiffness(rho))
        its = its_of(m)
        got = m.reaction_facts(U, rho)
    finally:
        m.solver.free()
    want = ref.reaction_facts(Uref, rho)
    assert np.linalg.norm(U - Uref) <= 1e-7 * np.linalg.norm(Uref) and its["accepted_by"] == "tol"
    assert float(m.f_red[:, 0] @ U[:, 0]) == pytest.approx(float(ref.f_red[:, 0] @ Uref[:, 0]), rel=1e-8)
    g, w = got["cases"][0], want["cases"][0]
    for key in ("clamp",):
        assert np.allclose(g[key]["force_n"], w[key]["force_n"], rtol=1e-6, atol=1e-9)
    assert np.allclose(g["springs"]["pad"]["force_n"], w["springs"]["pad"]["force_n"], rtol=1e-6, atol=1e-9)
    assert g["force_error"] < 1e-7 and g["moment_error"] < 1e-7
