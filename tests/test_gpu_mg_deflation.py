"""P2d: deflation of the rigid modes in the inner PCG of gpu-mg case C with joints (`gpu_mg/deflation.py`, the kernels `defl_*`, `GpuMG.pcg`).

The cause (measured in the research of P2d): in a rigid motion of a part that joints hold, `Kb x` and `C D^-1 C^T x` cancel, and 32-bit float
loses the difference when the beam is soft against the bond. Case C stopped on such a part. With the deflation, the rigid modes of each body are solved in
FP64 on the host and removed from the 32-bit inner PCG.

The test problems are the ones of the research: the cube of `tools/joint_check.py` (a round beam d 2 x L 20 along x, `soft_cube`; its beam over bond
stiffness is 7.0e-9 with E_beam = E_block and scales with E_beam) with four loads, and the pad case of `tests/test_joints.py` (the joint is the only support;
a clamp plate; two joints on one body; a separate block with its own joint = two bodies). The reference is the 64-bit `splu` of the bordered system. The
device tests run on the OpenCL device of the other gpu-mg tests (pocl on the host of the tests).

A test that compares counts "with" and "without" the deflation runs the same engine with `use_deflation(False)` (the launches and the PCG of before P2d).
The tests that need the deflation do not import it at the top of the module, so that the file loads on a tree without it.

P2d-D3h: `Z` has two kinds of column. The rigid columns (D3) are made once; the harmonic columns of the bonded patches depend on the design and are made in
`set_design` after an extension (`Deflation.extend`). The tests of the harmonic columns are in `tests/test_gpu_mg_harmonic.py`; the tests here check that the
cases of D3 still solve with them, and the host numbers of the rigid columns. A host-only test makes the extension with a block-Jacobi preconditioner (`jacobi`).
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as sla

from helpers import switch_off_deflation
from test_gpu_mg import need_case
from test_gpu_mg_border import bordered_reference, cube_host, fresh, host_of, joint_pair, soft_cube
from test_joints import case as joint_case
from test_joints import jc
from test_joints import prob_of as joint_prob
from test_springs import box
from topopt.cli import main
from topopt.config import Material
from topopt.fe import edof_table
from topopt.gpu_mg import kernels
from topopt.optimize import Model
from topopt.voxelize import LOAD

opencl = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1

CUBE_LOADS = {"axial": (np.array([5.0, 0, 0]), np.zeros(3)), "bend_y": (np.array([0, 5.0, 0]), np.zeros(3)),
              "torsion": (np.zeros(3), np.array([5.0, 0, 0])), "force+moment": (np.array([0, 3.0, 0]), np.array([0, 0, 7.0]))}

# `soft_cube(c, 2.0, 20.0, ratio)`: ratio = E_block / E_beam. The beam over bond stiffness on the softest mode is 7.0e-9 for the ratio 1 and scales with 1 / ratio.
RATIO_OF = {"1e-9": 7.0, "7e-9": 1.0, "3.7e-8": 0.19}
LAM_OF = {"1e-9": 1e-9, "7e-9": 7e-9, "3.7e-8": 3.7e-8}

DEFL_NAMES = ("defl_dot", "defl_coef", "defl_apply")          # literal: the module must load on a tree without the deflation


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


def jacobi(host):
    """A block-Jacobi preconditioner for `Deflation.extend` on the host: `dinv` of level 0 (the 3 x 3 blocks of the design, with the joint terms)."""
    from topopt.gpu_mg.hierarchy import apply_dinv
    L = host.levels[0]

    def precond(R):
        out, d = np.empty_like(R), np.zeros((L.nn, 3))
        for c in range(R.shape[0]):
            apply_dinv(np.ascontiguousarray(R[c]).reshape(-1, 3), L.dinv, d)
            out[c] = d.reshape(-1)
        return out
    return precond


def make_solver(c, jt, case_name="C", rtol=1e-6, fixed=None, out=None):
    """`(GpuSolver, edof, free)` of the cube `c` with the joint set `jt` (the construction of `joint_check.solve_gpu`, with the solver kept)."""
    from topopt.gpu_mg import device
    from topopt.gpu_mg.model import GpuParams, GpuSolver
    dev, case = device.select("auto", case_name)
    edof, active = edof_table(c.shape, c.labels == LOAD)
    free = c.free if fixed is None else np.setdiff1d(c.free, fixed)
    cfg = SimpleNamespace(has_partial_supports=lambda: False, settings=SimpleNamespace(robust="off"), material=Material(E=c.E, nu=c.nu))
    problem = SimpleNamespace(config=cfg, shape=c.shape, fixed_dofs=np.zeros(0, np.int64) if fixed is None else np.asarray(fixed),
                              h=c.h, springs=None, joints=jt)
    solver = GpuSolver(free, edof, active, problem, dev, case, 1, GpuParams(tol=rtol, min_free=300, backward_error=0.0), out)
    return solver, edof, free


def solve_one(solver, edof, c, f, fixed_free=None):
    """The displacement of the free dofs and the record of the solve."""
    solver.set_design(np.full(edof.shape[0], c.E))
    u = solver.solve(f if fixed_free is None else f[fixed_free])
    solver.end_design()
    return u, solver.records[-1]["iterations"][0]


def total(rec):
    """The inner PCG iterations of a case C record, summed over the refinement steps."""
    return int(sum(rec["inner"]))


def rel_err(u, ref):
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def pinned(c, kind):
    """The grid dofs of a pin on the cube: one corner node ("corner") or the nodes of one edge ("edge")."""
    ids = np.unique(edof_table(c.shape, c.labels == LOAD)[0] // 3)
    ny1, nz1 = c.shape[1] + 1, c.shape[2] + 1
    i, j, k = ids // (ny1 * nz1), (ids // nz1) % ny1, ids % nz1
    sel = {"corner": (i == 1) & (j == 1) & (k == 1), "edge": (i == 1) & (j == 1)}[kind]
    return (3 * ids[sel][:, None] + np.arange(3)).ravel()


def two_bodies(tmp_path):
    """`mutate` for the pad case: a separate 4 mm block with its own joint (a second body: it touches no other region)."""
    def mutate(d):
        box(tmp_path / "pad2.stl", (24, 0, 0), (28, 4, 4))
        d["regions"]["load"]["pad2"] = "pad2.stl"
        d["joints"].append({"region": "pad2", "to": [26, -20, 2], "section": {"kind": "round", "d": 4.0}})
    return mutate


def one_body_two_joints(tmp_path):
    """`mutate` for the pad case: a second joint on a pad of the same part."""
    def mutate(d):
        box(tmp_path / "pad2.stl", (8, -3, 0), (12, 0, 10))
        d["regions"]["load"]["pad2"] = "pad2.stl"
        d["joints"].append({"region": "pad2", "to": [10, -20, 5], "section": {"kind": "round", "d": 4.0}})
    return mutate


def soft_pad(tmp_path, lam, case_name="C", clamp=False, two=None, tol=1e-6, out=None):
    """`(GpuModel, rho, reference U)` of the pad case of `tests/test_joints.py` with every beam scaled to the beam over bond stiffness `lam` on its softest
    mode. `clamp`: with the clamp plate (False: the joint is the only support). `two`: None, "one_body" (a second joint on the same part) or "two_bodies"
    (a separate block with its own joint). The reference is the Jacobi-scaled bordered `splu` of the same design."""
    from topopt.gpu_mg.model import GpuModel, GpuParams
    mutate_of = {None: None, "one_body": one_body_two_joints, "two_bodies": two_bodies}[two]

    def build(sub, solver, energies):
        d = fresh(tmp_path, sub)

        def mutate(y):
            y["settings"]["gpu_case"] = case_name
            if mutate_of is not None:
                mutate_of(d)(y)
                if energies is not None:
                    y["joints"][1]["E"] = float(energies[1])
        return joint_prob(d, clamp=clamp, solver=solver, mutate=mutate, joint={} if energies is None else {"E": float(energies[0])})

    probe = build("probe", "gpu-mg", None)
    lam1 = [float(sla.eigh(j.B, j.D_bond, eigvals_only=True).min()) for j in probe.joints.joints]
    e1 = [(j.E if j.E is not None else probe.config.material.E) for j in probe.config.joints]
    energies = [e * lam / l1 for e, l1 in zip(e1, lam1)]
    ref = Model(build("ref", "amg", energies))
    rho = 0.1 + 0.8 * np.random.default_rng(3).uniform(size=ref.n_design)
    u_ref = bordered_reference(ref, rho)
    ref.solver.free()
    m = GpuModel(build("gpu", "gpu-mg", energies), out, GpuParams(tol=tol, min_free=300, backward_error=0.0))
    return m, rho, u_ref


# --- the host part (no OpenCL) -----------------------------------------------------------------------------------------------


def test_the_modes_are_six_for_each_body_orthonormal_zero_on_the_other_bodies_and_rigid(tmp_path):
    """`Z` is built for each body, not once for the part (6 global modes stop a part of two bodies: REPORT.md E). The pad and a separate block that
    touches nothing are two connected components of the active elements (an element joins its 8 nodes)."""
    from topopt.gpu_mg.deflation import Deflation, body_labels, node_positions
    d0 = fresh(tmp_path, "p")
    host, _ = host_of(joint_prob(d0, clamp=False, solver="gpu-mg", mutate=two_bodies(d0)))
    L = host.levels[0]
    nb, lab = body_labels(L)
    d = Deflation(host)
    assert nb == d.nbodies == 2 and d.n_rigid == 12 and d.per_body == [6, 6] and d.Zt.shape == (d.dk, 3 * L.nn) and d.dk == 12 + d.n_harm
    assert d.n_harm == 12 and d.per_patch == [6, 6]                                         # D3h: one patch for each joint, 6 harmonic columns for each
    assert min(d.nodes_per_body) < 300 < max(d.nodes_per_body)                               # the block is small, the pad case is not
    assert np.abs(d.Zt[:12] @ d.Zt[:12].T - np.eye(12)).max() < 1e-12
    pos = node_positions(L)
    for b in range(2):
        nodes = np.flatnonzero(lab == b)
        dofs = (3 * nodes[:, None] + np.arange(3)).reshape(-1)
        other = np.setdiff1d(np.arange(3 * L.nn), dofs)
        rows = slice(6 * b, 6 * b + 6)
        assert np.abs(d.Zt[rows][:, other]).max() == 0 and np.abs(d.Zt[rows][:, dofs]).max() > 0     # a mode of one body is zero on the other body
        r = pos[nodes] - pos[nodes].mean(axis=0)
        rigid = np.zeros((len(nodes), 3, 6))                                                         # the 3 translations and the 3 rotations of the body
        rigid[:, :, :3] = np.eye(3)
        for a in range(3):
            rigid[:, :, 3 + a] = np.cross(np.eye(3)[a][None, :], r)
        rigid = rigid.reshape(-1, 6)
        coef = np.linalg.lstsq(rigid, d.Zt[rows][:, dofs].T, rcond=None)[0]
        assert np.abs(rigid @ coef - d.Zt[rows][:, dofs].T).max() < 1e-12                           # every mode is a rigid field of its body (no fixed dof here)


def test_a_mode_is_zero_on_a_fixed_dof_and_not_on_the_other_axes_of_the_node():
    """The modes are cut per dof: the `fix` flags of level 0 are (nn, 3) and a node may have one fixed axis (PR #87 makes the flags per dof)."""
    from topopt.gpu_mg.deflation import Deflation
    c, jt, host = cube_host(n=4, min_free=30)
    L = host.levels[0]
    k = L.nn // 2
    L.fix[k] = [1, 0, 0]
    d = Deflation(host)
    Z, U = d.Zt[:d.n_rigid], d.U                                                              # the rigid modes, and the motions that the patches prescribe
    assert d.n_rigid == 6 and Z[:, 3 * k].max() == 0 == Z[:, 3 * k].min()                  # x of the node: zero in all modes
    assert np.abs(Z[:, 3 * k + 1]).max() > 0 and np.abs(Z[:, 3 * k + 2]).max() > 0
    assert U[:, 3 * k].max() == 0 == U[:, 3 * k].min() and not d.free[3 * k]
    L.fix[k] = [1, 1, 1]
    d = Deflation(host)
    assert d.Zt[:d.n_rigid, 3 * k:3 * k + 3].max() == 0 == d.Zt[:d.n_rigid, 3 * k:3 * k + 3].min()
    assert d.U[:, 3 * k:3 * k + 3].max() == 0 == d.U[:, 3 * k:3 * k + 3].min() and not d.free[3 * k:3 * k + 3].any()


def test_a_body_with_every_dof_fixed_has_no_mode_and_the_other_body_keeps_its_six(tmp_path):
    """The rank step: the cut modes of a body with no free dof are zero columns, and a zero column has no place in `Z` (a QR would give it an arbitrary direction,
    maybe on a Dirichlet dof)."""
    from topopt.gpu_mg.deflation import Deflation, body_labels
    d0 = fresh(tmp_path, "p")
    host, _ = host_of(joint_prob(d0, clamp=False, solver="gpu-mg", mutate=two_bodies(d0)))
    L = host.levels[0]
    _nb, lab = body_labels(L)
    small = int(np.argmin(np.bincount(lab)))
    L.fix[lab == small] = 1
    d = Deflation(host)
    assert d.per_body[small] == 0 and d.per_body[1 - small] == 6 and d.n_rigid == 6
    assert sorted(d.per_patch) == [0, 6] and d.dk == 12 and d.Zt.shape[0] == 12                # the patch of the fixed body has no motion either (the same rank rule)
    assert np.abs(d.Zt[:, np.flatnonzero(np.repeat(lab == small, 3))]).max() == 0
    assert np.abs(d.U[:, np.flatnonzero(np.repeat(lab == small, 3))]).max() == 0


def test_the_numbers_of_a_design_are_AZ_E_its_inverse_and_the_coarse_correction_leaves_a_load_that_is_A_orthogonal_to_Z():
    from topopt.gpu_mg.deflation import Deflation
    c, jt, host = cube_host(n=8, ratio=1.0)              # a joint-only cube, beam over bond 7e-9
    d = Deflation(host)
    assert not d.ready
    with pytest.raises(RuntimeError, match="call extend after setup_host"):                  # D3h: the harmonic columns are made for each design, before the numbers
        d.set_design(host)
    d.extend(host, jacobi(host), 2, 300, 0.10)
    d.set_design(host)
    L = host.levels[0]
    k = d.k
    assert k == d.dk == 12 and d.ready
    y = np.zeros((L.nn, 3))
    for j in range(k):
        host.apply_fine(d.Zt[j].reshape(-1, 3), y)
        assert np.array_equal(d.AZt[j], y.reshape(-1))                              # one apply_fine for each column
    M = d.Zt[:k] @ d.AZt[:k].T
    assert np.array_equal(d.E, 0.5 * (M + M.T)) and np.linalg.eigvalsh(d.E).min() > 0
    assert np.abs(d.Einv @ d.E - np.eye(k)).max() < 1e-6          # E holds the rigid columns (energy of the size of the beam) and the harmonic ones (the size of the bond)
    assert d.Zt32.dtype == d.AZt32.dtype == d.Einv32.dtype == np.float32 and d.Einv32.shape == (d.dk, d.dk)
    b = np.random.default_rng(0).standard_normal((2, 3 * L.nn))
    a, b2 = d.coarse(b)
    # Z^T (b - A Z a) = 0 to the noise of the host operator (E is the symmetric part of Z^T (AZ), and the host product has rounding noise in a rigid motion: F1-7)
    assert a.shape == (2, k) and np.abs(d.Zt[:k] @ b2.T).max() <= 1e-6 * np.abs(d.Zt[:k] @ b.T).max()
    host.apply_fine(d.correction(a)[0].reshape(-1, 3), y)
    assert np.abs(y.reshape(-1) - a[0] @ d.AZt[:k]).max() <= 1e-6 * np.abs(y).max()       # A (Z a) is (AZ) a, to the noise of the host operator in a rigid motion (F1-7)


def test_a_matrix_E_that_is_not_positive_definite_is_an_error_with_the_fact():
    from topopt.gpu_mg.deflation import Deflation, DeflationError
    c, jt, host = cube_host(n=4, min_free=30)
    d = Deflation(host)
    d.extend(host, jacobi(host), 2, 300, 0.10)

    def negative(x, y):
        y[:] = -x
    host.apply_fine = negative
    with pytest.raises(DeflationError, match="not positive definite") as e:
        d.set_design(host)
    assert str(d.dk) in str(e.value) and not d.ready


# --- the program text ----------------------------------------------------------------------------------------------------------


def test_the_program_text_is_the_text_of_before_unless_case_c_with_joints_asks_for_the_deflation():
    """Case B, case C without a border and case C with springs only never get the deflation kernels; with joints, `defl_dk` appends them after all other
    kernels (the text before them is the text without `defl_dk`). The added text has no 64-bit type and no pragma (case C needs none)."""
    for case in ("B", "C"):
        for nr in (1, 2):
            for border, nk in ((False, 0), (True, 0), (True, 6), (True, 12)):
                base = kernels.source(case, nr, border=border, nk=nk)
                assert kernels.source(case, nr, border=border, nk=nk, defl_dk=0) == base
                assert "defl_" not in base and "DK" not in base
                if not (case == "C" and border and nk):
                    with pytest.raises(ValueError, match="deflation kernels are for case C with joints"):
                        kernels.source(case, nr, border=border, nk=nk, defl_dk=6)
    base = kernels.source("C", 2, border=True, nk=6)
    for dk in (6, 12, 24):
        s = kernels.source("C", 2, border=True, nk=6, defl_dk=dk)
        added = s[len(base):]
        assert s.startswith(base) and f"#define DK {dk}\n" in added and all(f"__kernel void {n}(" in added for n in DEFL_NAMES)
        assert "double" not in added and "fp64" not in added.lower() and "pragma OPENCL EXTENSION" not in added


@opencl
@pytest.mark.parametrize("dk", [6, 12, 24])
def test_the_deflation_kernels_build_with_opencl_1_2_for_any_number_of_columns(dev, dk):
    import pyopencl as cl
    ctx = cl.Context([dev.obj])
    options = kernels.build_options("C")
    prg = cl.Program(ctx, kernels.source("C", 2, border=True, nk=6, defl_dk=dk)).build(options=options)
    for n in DEFL_NAMES:
        assert cl.Kernel(prg, n).get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev.obj) >= 1


# --- the engine of before when there is nothing to deflate -----------------------------------------------------------------------------------


@opencl
def test_case_b_and_case_c_without_joints_have_the_launches_the_program_and_the_buffers_of_before(tmp_path, dev):
    """Case B with joints, case C with springs only (a border with no border dofs): no deflation object, no `defl` buffer, no deflation launch, no
    deflation kernel in the program, no `deflation` key in the facts."""
    from test_gpu_mg_border import gpu_model
    need_case("B", dev)
    _ref, _rho, _U, mb = joint_pair(tmp_path, "one joint", "B")
    ms = gpu_model(tmp_path, "C", "clamp")
    try:
        for m in (mb, ms):
            s = m.solver
            g = s.engine
            assert s.defl is None and s.defl_off is None and g.defl is None and g.defl_off is None and s.facts()["deflation"] is None
            assert "defl" not in g.nbytes_parts and g.dk == 0 and "defl_" not in g.source
            assert not [o.name for o in g.iter_ops if o.name.startswith("defl_")] and g.iter_ops == g._ops_plain
            m.solve(m.stiffness(np.full(m.n_design, 0.6)))
            assert "deflation" not in s.records[-1] and "defl" not in s.records[-1]["setup_parts_s"]
        assert mb.solver.case == "B" and ms.solver.case == "C"
    finally:
        mb.solver.free()
        ms.solver.free()


@opencl
def test_case_c_with_joints_adds_three_launches_before_p_update_and_two_vectors_of_the_modes(dev):
    """The order of the launches of the plan: V-cycle, defl_dot, defl_coef, defl_apply, p_update, operator, upd_xr. The buffers: Z and AZ (dk vectors of the
    dofs of level 0 in 32 bit: 2 * dk vectors, 24 for the cube with its 6 rigid modes and the 6 harmonic columns of its one patch), E^-1, the partial sums and
    the coefficients."""
    need_case("C", dev)
    c = jc.cube(8)
    jt, f, lam = soft_cube(c, 2.0, 20.0, 1.0)
    s, edof, free = make_solver(c, jt)
    try:
        g = s.engine
        names = [o.name for o in g.iter_ops]
        plain = [o.name for o in g._ops_plain]
        assert g.defl is s.defl and g.defl is not None and g.dk == 12 and not g.defl_off and "DK" in g.source
        i = names.index("defl_dot")
        assert names[i:i + 3] == list(DEFL_NAMES) and names[i - 1] == "cheb_post1_dot" and names[i + 3] == "p_update"
        assert names[:i] + names[i + 3:] == plain                          # the other launches are the same, in the same order
        iso = 4
        want = (2 * g.dk * g.n3 + g.dk * g.dk + g.nr * g.dk * g.G + g.nr * g.dk) * iso
        assert g.nbytes_parts["defl"] == want
        d = s.facts()["deflation"]
        assert d["on"] and d["bodies"] == 1 and d["columns"] == 12 and d["rigid_columns"] == 6 and d["harmonic_columns"] == 6 and d["patches"] == 1
        assert d["device_bytes"] == want and d["work_group"] == g.wd >= 1 and d["host_bytes"] > 0
    finally:
        s.free()


@opencl
@pytest.mark.parametrize("which", ["cube", "two bodies"])
def test_the_three_kernels_project_z_to_be_A_orthogonal_to_the_modes_and_make_the_two_partial_dots_again(tmp_path, dev, which):
    """Run `defl_dot`, `defl_coef`, `defl_apply` on a random z (and random r and z_old) and compare with the float64 formula z - Z E^-1 (AZ)^T z, with
    (AZ)^T z' near zero, and with the partial dots r.z' and r.z_old that `cheb_dot` made from the unprojected z. `two bodies`: 12 columns."""
    import pyopencl as cl
    need_case("C", dev)
    if which == "cube":
        c = jc.cube(8)
        jt, f, lam = soft_cube(c, 2.0, 20.0, 1.0)
        s, edof, free = make_solver(c, jt)
        s.set_design(np.full(edof.shape[0], c.E))
    else:
        m, rho, _u = soft_pad(tmp_path, 1e-9, clamp=False, two="two_bodies")
        s = m.solver
        s.set_design(m.stiffness(rho))
    try:
        g, d = s.engine, s.defl
        assert g.dk == d.dk == (12 if which == "cube" else 24) and d.k == d.dk
        nr, n3 = g.nr, g.n3
        rng = np.random.default_rng(4)
        z, ro, zo = (rng.standard_normal((nr, n3)).astype(np.float32) for _ in range(3))
        free = (s.host.levels[0].fix.reshape(-1) == 0)
        z[:, ~free] = ro[:, ~free] = zo[:, ~free] = 0.0
        for buf, v in ((g.vout, z), (g.r_o, ro), (g.zo, zo)):
            cl.enqueue_copy(g.q, buf, v)
        g.run([o for o in g.iter_ops if o.name in DEFL_NAMES])
        g.q.finish()
        got = np.empty_like(z)
        cl.enqueue_copy(g.q, got, g.vout)
        pa = np.empty(2 * nr * g.NG1, np.float32)
        cl.enqueue_copy(g.q, pa, g.P1)
        g.q.finish()
        z64 = z.astype(np.float64)
        k = d.k
        want = z64 - (d.Einv @ (d.AZt[:k] @ z64.T)).T @ d.Zt[:k]
        assert np.abs(got - want).max() <= 2e-4 * np.abs(want).max()                                  # 32-bit arithmetic of the kernels
        assert np.abs(d.AZt[:k] @ got.T.astype(np.float64)).max() <= 1e-3 * np.abs(d.AZt[:k] @ z64.T).max()      # A-orthogonal to Z (before: O(1))
        pa = pa.astype(np.float64).reshape(2, nr, g.NG1).sum(axis=2)
        assert np.allclose(pa[0], (ro.astype(np.float64) * got).sum(axis=1), rtol=2e-4) and np.allclose(pa[1], (ro.astype(np.float64) * zo).sum(axis=1), rtol=2e-4)
    finally:
        s.free()


# --- the solves ------------------------------------------------------------------------------------------------------------------


@opencl
@pytest.mark.parametrize("name", ["1e-9", "7e-9", "3.7e-8"])
def test_case_c_solves_the_soft_cube_for_the_four_loads_and_gives_the_answer_of_the_bordered_splu(dev, name):
    """Cube n 8 held by a joint only, case C, `gpu_tol` 1e-6, beam over bond stiffness 1e-9, 7e-9, 3.7e-8: axial, bending, torsion, force + moment. Before
    P2d (3a8e8a2) case C solved none of the four loads at these ratios. Measured with the deflation (pocl): 2 to 3 refinement steps and 8 to 10 inner
    iterations; case B needs 9 to 12. The bounds are 4 steps and 12 iterations."""
    need_case("C", dev)
    c = jc.cube(8)
    jt, _f, lam = soft_cube(c, 2.0, 20.0, RATIO_OF[name])
    assert lam == pytest.approx(LAM_OF[name], rel=0.1)
    s, edof, free = make_solver(c, jt)
    try:
        for load, (force, moment) in CUBE_LOADS.items():
            f = jc.nodal_load(c, force, moment)
            u, rec = solve_one(s, edof, c, f)
            u_ref, _ = jc.solve(c, jt, f, "splu")
            assert rec["accepted_by"] == "tol" and rec["refine_steps"] <= 4 and total(rec) <= 12, (load, rec)
            assert rel_err(u, u_ref) <= 1e-5, load
    finally:
        s.free()


@opencl
def test_case_c_solves_the_soft_cube_of_16_voxels_at_7e_9(dev):
    """n 16 (14,739 dofs) with the same E ratio as the cube of 8 (the bond term grows with the cube, so the beam over bond stiffness is 1.3e-9). This size
    needs the pairwise sum of F1-7 in the host check. Measured: 3 steps, 12 to 13 iterations."""
    need_case("C", dev)
    c = jc.cube(16)
    jt, f, lam = soft_cube(c, 2.0, 20.0, 1.0)
    assert lam == pytest.approx(1.3e-9, rel=0.1)
    s, edof, free = make_solver(c, jt)
    try:
        u, rec = solve_one(s, edof, c, f)
        u_ref, _ = jc.solve(c, jt, f, "splu")
        assert rec["accepted_by"] == "tol" and rec["refine_steps"] <= 4 and total(rec) <= 16, rec
        assert rel_err(u, u_ref) <= 1e-5
    finally:
        s.free()


@opencl
@pytest.mark.parametrize("lam", [1e-9, 1.1e-7])
def test_case_c_solves_the_pad_that_only_a_joint_holds(tmp_path, dev, lam):
    """The pad of `tests/test_joints.py` with no fixed region (8,979 dofs). Before P2d case C stopped at 1e-9 to 1.1e-7. Measured: 3 steps and 32 iterations."""
    need_case("C", dev)
    m, rho, u_ref = soft_pad(tmp_path, lam, clamp=False)
    try:
        u = m.solve(m.stiffness(rho))
        rec = m.solver.records[-1]["iterations"][0]
    finally:
        m.solver.free()
    assert rec["accepted_by"] == "tol" and rec["refine_steps"] <= 4 and total(rec) <= 50, rec
    assert rel_err(u, u_ref) <= 1e-5


@opencl
def test_case_c_solves_a_pad_with_two_joints_on_one_body(tmp_path, dev):
    need_case("C", dev)
    m, rho, u_ref = soft_pad(tmp_path, 1e-9, clamp=False, two="one_body")
    try:
        assert m.solver.defl.nbodies == 1 and m.solver.defl.n_rigid == 6 and m.solver.defl.dk == 18      # 6 rigid + 6 harmonic for each of the 2 patches
        u = m.solve(m.stiffness(rho))
        rec = m.solver.records[-1]["iterations"][0]
    finally:
        m.solver.free()
    assert rec["accepted_by"] == "tol" and rec["refine_steps"] <= 5 and total(rec) <= 120, rec           # measured 3 to 4 steps, 79 to 81 iterations
    assert rel_err(u, u_ref) <= 1e-5


@opencl
def test_case_c_solves_a_part_of_two_bodies_with_the_modes_of_each_body(tmp_path, dev):
    """A pad and a separate block with its own joint: `Z` has 6 modes for each body (12 rigid columns) and 6 harmonic columns for each patch. Six global modes
    stop this part (REPORT.md E)."""
    need_case("C", dev)
    m, rho, u_ref = soft_pad(tmp_path, 1e-9, clamp=False, two="two_bodies")
    try:
        d = m.solver.defl
        assert d.nbodies == 2 and d.n_rigid == 12 and d.dk == 24 and m.solver.engine.dk == 24
        u = m.solve(m.stiffness(rho))
        rec = m.solver.records[-1]["iterations"][0]
    finally:
        m.solver.free()
    assert rec["accepted_by"] == "tol" and rec["refine_steps"] <= 5 and total(rec) <= 60, rec           # measured 3 steps, 32 to 33 iterations
    assert rel_err(u, u_ref) <= 1e-5


@opencl
def test_the_deflation_does_not_cost_iterations_on_a_pad_with_a_clamp_plate(tmp_path, monkeypatch, dev):
    """A clamped body is well posed: the rigid modes are cut by the Dirichlet dofs and the baseline solves it (measured 3 steps, 25 to 31 iterations at 1e-9
    and 3.3e-7). The deflated count is not above the count of the same engine without the deflation."""
    need_case("C", dev)
    counts = {}
    for on in (True, False):
        if not on:
            switch_off_deflation(monkeypatch)
        m, rho, u_ref = soft_pad(fresh(tmp_path, str(on)), 1e-9, clamp=True)
        try:
            u = m.solve(m.stiffness(rho))
            rec = m.solver.records[-1]["iterations"][0]
            assert (m.solver.defl is not None) and rel_err(u, u_ref) <= 1e-5 and rec["accepted_by"] == "tol"
        finally:
            m.solver.free()
        counts[on] = total(rec)
    assert counts[True] <= counts[False], counts


@opencl
@pytest.mark.parametrize("kind", ["corner", "edge"])
def test_case_c_solves_a_cube_that_is_only_pinned_at_a_node_or_along_an_edge(dev, kind):
    """A pinned part has soft rotations too. Before P2d: a pin at one corner node needed 2100 to 3600 inner iterations or stopped (this test fails on 3a8e8a2
    with a SolveFailure); a pin along an edge needed 30 to 577 iterations in 3 to 12 steps. With the deflation: 2 to 3 steps and 8 to 12 iterations."""
    need_case("C", dev)
    c = jc.cube(8)
    jt, _f, lam = soft_cube(c, 2.0, 20.0, 1.0)
    fixed = pinned(c, kind)
    s, edof, free = make_solver(c, jt, fixed=fixed)
    ix = np.searchsorted(c.free, free)
    try:
        for load, (force, moment) in CUBE_LOADS.items():
            f = jc.nodal_load(c, force, moment)
            u, rec = solve_one(s, edof, c, f, fixed_free=ix)
            K = jc.kb_added(c, jt).tocsr()[ix][:, ix].tocsc()
            C, D = jt.border(free)
            ref = bordered_reference_cut(K, C, D, f[ix])
            assert rec["accepted_by"] == "tol" and rec["refine_steps"] <= 4 and total(rec) <= 16, (kind, load, rec)
            assert rel_err(u, ref) <= 1e-5, (kind, load)
    finally:
        s.free()


def bordered_reference_cut(K, C, D, f):
    """The bordered `splu` solve of `[[K, C], [C^T, D]] [u, q] = [f, 0]` (Jacobi scaled) for a reduced `K`."""
    import scipy.sparse as sps
    from scipy.sparse.linalg import splu
    A = sps.bmat([[K, C], [C.T, D]], format="csc")
    sc = 1.0 / np.sqrt(A.diagonal())
    rhs = np.concatenate([f, np.zeros(C.shape[1])])
    x = sc * splu((sps.diags(sc) @ A @ sps.diags(sc)).tocsc()).solve(sc * rhs)
    return x[:len(f)]


@opencl
@pytest.mark.parametrize("ratio, solves", [(7e2, True), (7e4, False)], ids=["9.9e-12", "9.9e-14"])
def test_the_two_cases_of_the_check_line_for_the_deflated_cube_hold_at_gpu_tol_1e_6(dev, ratio, solves):
    """`border.check_line` says: the cube (n 8, one load) that only a joint holds solved at 9.9e-12 with the rigid modes removed (2 to 3 refinement steps)
    and stopped at 9.9e-14 and below (pocl). If this test changes, the numbers of that text change with it. The stop is a fact of 32-bit float and of the
    FP64 host check, not a limit of the solver: the second case may start to solve some day, and then the text and this test are to be made again."""
    from topopt.gpu_mg.model import SolveFailure
    need_case("C", dev)
    c = jc.cube(8)
    jt, f, lam = soft_cube(c, 2.0, 20.0, ratio)
    assert lam == pytest.approx(9.9e-12 if solves else 9.9e-14, rel=0.05)
    s, edof, free = make_solver(c, jt, rtol=1e-6)
    try:
        if solves:
            u, rec = solve_one(s, edof, c, f)
            u_ref, _ = jc.solve(c, jt, f, "splu")
            assert rec["accepted_by"] == "tol" and rec["refine_steps"] <= 3, rec
            assert rel_err(u, u_ref) <= 1e-5
        else:
            with pytest.raises(SolveFailure):
                solve_one(s, edof, c, f)
    finally:
        s.free()


@opencl
def test_a_part_that_the_deflation_cannot_resolve_still_stops_with_the_facts(dev):
    """The beam over bond stiffness 1e-11 on the cube (a ratio of 7e2): the inner PCG stagnates at the noise of the host check (measured 3.9e-8 on pocl), and
    at 1e-15 and below the device gives values that are not finite. The solve stops with a `SolveFailure`; the text says the deflation was used."""
    from topopt.gpu_mg.model import SolveFailure
    need_case("C", dev)
    c = jc.cube(8)
    jt, f, lam = soft_cube(c, 2.0, 20.0, 7e2)
    s, edof, free = make_solver(c, jt, rtol=1e-9)
    try:
        with pytest.raises(SolveFailure) as e:
            solve_one(s, edof, c, f)
    finally:
        s.free()
    text = str(e.value)
    assert e.value.its["failed"] and "did not reach the relative residual" in text and "case C" in text and "32-bit float" in text
    assert f"the beam is {lam:.1e} of the bond stiffness on its softest mode" in text
    assert "the inner PCG deflated the rigid modes (1 body, 6 columns) and 6 harmonic columns of 1 bonded patch(es)" in text
    assert "(the patch motions extended into the part) and the host solved them in FP64" in text
    assert "a stiffer beam (a larger section, a shorter beam)" in text and "Options:" in text


# --- when the deflation cannot run ----------------------------------------------------------------------------------------------


def benign_cube():
    """A cube with a stiff short beam (d 8, L 10): case C solves it without the deflation too. Return `(c, jt, f)`."""
    c = jc.cube(8)
    jt = jc.make_joint(c, np.array([10.0, 0, 0]), ("round", 8.0), k_a=100.0 * c.E / c.h)
    return c, jt, jc.nodal_load(c, np.array([5.0, 0, 0]), np.zeros(3))


def assert_runs_as_before(s, edof, c, jt, f, why):
    """The solver has no deflation, says why, has the launches of before, and solves the benign cube to the answer of the bordered `splu`."""
    g = s.engine
    assert s.defl is None and g.defl is None and why in s.defl_off and why in g.defl_off
    assert not [o.name for o in g.iter_ops if o.name.startswith("defl_")] and g.iter_ops == g._ops_plain
    fact = s.facts()["deflation"]
    assert fact["on"] is False and why in fact["why"]
    u, rec = solve_one(s, edof, c, f)
    u_ref, _ = jc.solve(c, jt, f, "splu")
    assert rec["accepted_by"] == "tol" and rel_err(u, u_ref) <= 1e-5
    assert s.records[-1]["deflation"]["on"] is False and why in s.records[-1]["deflation"]["why"]


@opencl
def test_a_program_that_does_not_build_with_the_deflation_kernels_runs_without_them_and_says_why(monkeypatch, dev):
    need_case("C", dev)
    monkeypatch.setattr(kernels, "DEFL_KERNEL", "this is not OpenCL C\n")
    c, jt, f = benign_cube()
    s, edof, free = make_solver(c, jt)
    try:
        assert "defl_" not in s.engine.source                                                 # the program was built again without the kernels
        assert_runs_as_before(s, edof, c, jt, f, "the OpenCL compiler could not build the deflation kernels for 12 columns")
    finally:
        s.free()


@opencl
def test_a_device_that_allows_a_smaller_work_group_for_defl_apply_than_the_node_kernels_use_runs_without_the_deflation(monkeypatch, dev):
    from topopt.gpu_mg import engine
    need_case("C", dev)
    real = engine.GpuMG.kernel_limit
    monkeypatch.setattr(engine.GpuMG, "kernel_limit", lambda self, name: 16 if name == "defl_apply" else real(self, name))
    c, jt, f = benign_cube()
    s, edof, free = make_solver(c, jt)
    try:
        assert s.engine.wg > 16 and (s.engine.wg, s.engine.wgv) == (128, 256)               # the other sizes are the sizes of before
        assert_runs_as_before(s, edof, c, jt, f, "the device allows work-group size 16 for the deflation kernel defl_apply")
    finally:
        s.free()


@opencl
@pytest.mark.parametrize("how", ["a buffer larger than the device allows", "the device cannot allocate"])
def test_a_device_that_cannot_hold_the_deflation_buffers_runs_without_the_deflation_and_says_why(monkeypatch, dev, how):
    import pyopencl as cl
    from topopt.gpu_mg import engine
    need_case("C", dev)
    if how.startswith("a buffer"):
        monkeypatch.setattr(engine.GpuMG, "max_alloc", lambda self: 1000)
        why = "is larger than the largest buffer of the device"
    else:
        real = engine.GpuMG.buf

        def buf(self, host=None, nbytes=None, flags=None, part="other"):
            if part == "defl" and nbytes is not None and nbytes > 100:
                raise cl.MemoryError("clCreateBuffer failed: MEM_OBJECT_ALLOCATION_FAILURE")
            return real(self, host, nbytes, flags, part)
        monkeypatch.setattr(engine.GpuMG, "buf", buf)
        why = "the device could not make the deflation buffers"
    c, jt, f = benign_cube()
    s, edof, free = make_solver(c, jt)
    try:
        assert "defl" not in s.engine.nbytes_parts or s.engine.nbytes_parts["defl"] == 0           # nothing of it is counted or kept
        assert_runs_as_before(s, edof, c, jt, f, why)
    finally:
        s.free()


@opencl
def test_a_part_with_no_free_rigid_mode_has_nothing_to_deflate(dev):
    """Every body has all its dofs fixed: `Z` has no column, the program has no deflation kernel, and `defl_off` says it."""
    from topopt.gpu_mg import engine
    need_case("C", dev)
    c, jt, host = cube_host(n=8, min_free=300)
    g = engine.GpuMG(host, dev.obj, case="C", nr=2, deflation=SimpleNamespace(dk=0))
    try:
        assert g.defl is None and "no rigid mode is free" in g.defl_off and "defl_" not in g.source and g.iter_ops == g._ops_plain
    finally:
        g.release()


@opencl
def test_a_design_whose_matrix_E_fails_runs_without_the_deflation_and_the_next_design_tries_again(tmp_path, monkeypatch, dev):
    """`DeflationError` in the host part of one design: that design runs as before (and `run.log` and the record say why); the next design is deflated."""
    from topopt.gpu_mg.deflation import Deflation, DeflationError
    from topopt.snapshot import RunWriter
    need_case("C", dev)
    calls = []
    real = Deflation.set_design

    def flaky(self, host):
        calls.append(1)
        if len(calls) == 1:
            raise DeflationError("the matrix E = Z^T A Z of the 6 rigid modes is not positive definite in FP64 (a test)")
        return real(self, host)
    monkeypatch.setattr(Deflation, "set_design", flaky)
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    m, rho, u_ref = soft_pad(tmp_path, 1e-5, clamp=False, out=out)         # a pad that case C solves without the deflation too
    try:
        g = m.solver.engine
        u1 = m.solve(m.stiffness(rho))
        assert g.defl is not None and not g.defl_active and g.iter_ops == g._ops_plain
        u2 = m.solve(m.stiffness(rho))
        assert g.defl_active and any(o.name == "defl_dot" for o in g.iter_ops)
        recs = m.solver.records
    finally:
        m.solver.free()
    assert recs[0]["deflation"]["on"] is False and "not positive definite" in recs[0]["deflation"]["why"] and "defl" not in recs[0]["setup_parts_s"]
    d1 = recs[1]["deflation"]
    assert d1["on"] is True and d1["bodies"] == 1 and d1["columns"] == 12 and d1["rigid_columns"] == 6 and d1["harmonic_columns"] == 6
    assert recs[1]["setup_parts_s"]["defl"] > 0 and recs[1]["setup_parts_s"]["defl_ext"] > 0
    assert rel_err(u1, u_ref) <= 1e-5 and rel_err(u2, u_ref) <= 1e-5
    log = (tmp_path / "run" / "run.log").read_text()
    assert "deflation of the rigid modes and patches is off for this design" in log and "the next design tries the deflation again" in log


# --- run.log, the record and the run -------------------------------------------------------------------------------------------------------------


@opencl
@pytest.mark.parametrize("case_name", ["C", "B"])
def test_run_log_and_record_say_that_the_deflation_is_on_for_case_c_with_joints_and_say_nothing_for_case_b(tmp_path, dev, case_name):
    need_case(case_name, dev)
    p = joint_case(tmp_path, solver="gpu-mg", clamp=False, mutate=lambda d: d["settings"].update(gpu_case=case_name, iterations=2))
    out = tmp_path / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text()
    recs = [json.loads(line) for line in (out / "gpu_mg.jsonl").read_text().splitlines()]
    assert recs and all(r["case"] == case_name for r in recs)
    if case_name == "B":
        assert "rigid modes" not in log and all("deflation" not in r and "defl" not in r["setup_parts_s"] for r in recs)      # not the word alone: a folder name has it
        return
    line = [x for x in log.splitlines() if "deflation of the rigid modes in the inner PCG is on" in x]
    assert len(line) == 1 and "1 body, 6 rigid mode column(s) [6] and 6 harmonic column(s) [6] of 1 bonded patch(es)" in line[0]
    assert "a method setting" in line[0] and "s once (bodies, modes, QR)" in line[0] and "12 columns in all" in line[0]
    host = [x for x in log.splitlines() if "deflation of the first design" in x]
    assert len(host) == 1 and "extension of 6 harmonic column(s) of 1 bonded patch(es), cold start" in host[0] and "12 products with the FP64 operator" in host[0]
    assert all(r["deflation"]["on"] and r["deflation"]["columns"] == 12 and r["setup_parts_s"]["defl"] > 0 for r in recs)
    assert [r["deflation"]["extension"]["warm"] for r in recs] == [False] + [True] * (len(recs) - 1)     # a design starts from the extension of the one before
    assert all(r["setup_s"] >= r["setup_parts_s"]["defl"] + r["setup_parts_s"]["defl_ext"] for r in recs)      # the host time is in the setup time of the record
