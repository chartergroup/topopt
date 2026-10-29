"""Joints against closed forms: a stiff cube with a beam to a fixed point (CONTRACTS.md section 38).

Run: python tools/joint_check.py [--solver splu|pardiso|amg|gpu-mg] [--n N] [--gpu-case auto|B|C]
The cube is one load region (n x n x n voxels of size 1 mm). All its exposed faces are bonded to a rigid body, and a
round beam goes from the centre of the cube to a fixed point. The loads are nodal: a force, a moment (the distributing
coupling of `loads.moment_spread`), or both. The engine value is the compliance `f . u` of the bordered system. The
closed form is the cantilever with the clamp at the far point and the free end at the joint: axial `F^2 L / (E A)`,
torsion `M^2 L / (G J)`, bending `F^2 L^3 / (3 E I)`, and the cross term `+F M L^2 / (E I)` of a force with a moment.
The block and the bond add a small flexibility in series (axial: +2.8e-4 relative for E_block = E_beam).

`--solver gpu-mg` (P2b) solves the same cube with the matrix-free multigrid on the OpenCL device (`settings.gpu_device` rules: the device that
`auto` takes). The cube has no fixed region: the joint alone holds it, which is the floating case of the multigrid. On the host of the tests
run it with the pocl ICD only.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from topopt.config import JointSpec, LoadRegion, Material, SectionSpec
from topopt.fe import Assembler, edof_table, hex8_ke, rigid_body_modes
from topopt.joints import BorderedSolver, build_joints, section_properties
from topopt.loads import moment_spread, node_ids, node_positions
from topopt.springs import SpringMatrix
from topopt.voxelize import EMPTY, LOAD


def cube(n, h=1.0, E=1.0, nu=0.3, margin=1):
    """A cube of n voxels on a side, one load region (id 0) in an empty grid. Return the pieces of a problem."""
    shape = (n + 2 * margin,) * 3
    labels = np.full(shape, EMPTY, dtype=np.uint8)
    ids = np.full(shape, -1, dtype=np.int16)
    sl = (slice(margin, margin + n),) * 3
    labels[sl], ids[sl] = LOAD, 0
    origin = -h * np.array(shape) / 2.0                      # the cube is centred at the origin
    edof, _ = edof_table(shape, labels == LOAD)
    free = np.unique(edof)
    asm = Assembler(edof, hex8_ke(1.0, nu, h), free)
    K = asm.K(np.full(edof.shape[0], E))
    return SimpleNamespace(labels=labels, ids=ids, shape=shape, origin=origin, h=h, free=free, asm=asm, K=K, E=E, nu=nu)


def make_joint(c, to, section=("round", 2.0), E_beam=1.0, nu_beam=0.3, k_a=None, orient=None, origin=None):
    """`JointSet` of the cube. `to` is the far point (mm); `k_a` (N/mm3) is the bond stiffness per area (None: the default)."""
    spec = JointSpec("block", tuple(map(float, to)), SectionSpec(section[0], tuple(section[1:])),
                     None if origin is None else tuple(origin), None if orient is None else tuple(orient), k_a, E_beam, nu_beam)
    cfg = SimpleNamespace(regions=SimpleNamespace(load=(LoadRegion("block", Path("block.stl")),)), joints=(spec,),
                          material=Material(E=c.E, nu=c.nu))
    return build_joints(cfg, c.labels, c.ids, c.origin, c.h)


def nodal_load(c, force, moment):
    """Load vector on the free dofs: the force shared equally by the nodes, and the moment by `moment_spread`."""
    ids = node_ids(c.ids == 0)
    pos = node_positions(ids, c.shape, c.origin, c.h)
    f = np.tile(np.asarray(force, float) / len(ids), (len(ids), 1))
    if np.linalg.norm(moment) > 0:
        f = f + moment_spread(pos - pos.mean(axis=0), np.asarray(moment, float))
    full = np.zeros(3 * int(np.prod(np.array(c.shape) + 1)))
    full[(3 * ids[:, None] + np.arange(3)).ravel()] = f.ravel()
    return full[c.free]


def kb_added(c, jt):
    """The stiffness matrix `K + Kb` of the cube with the bond terms added."""
    K = c.K.copy()
    SpringMatrix(*jt.triplets(), c.asm._indptr, c.asm._indices, c.free).add_to(K)
    return K


def solve_gpu(c, jt, f, rtol=1e-10, gpu_case="auto", min_free=300, info=None, fixed=None, backward_error=0.0, params=None):
    """Displacement of the free dofs with the solver gpu-mg (P2b). `fixed`: grid dofs of a clamp (none: the joint holds the cube).

    `info` (a dict) gets the device case, the number of levels and the record of the solve (`pcg`: the iterations of the first pass,
    or `inner` for case C) and `accepted_by`. `backward_error` is the limit of the backward-error rule (0: off, so case B takes the recurrence
    residual of its first pass and makes no check of the true residual; F1-7). `params`: more fields of `GpuParams` (a dict).
    """
    from topopt.gpu_mg import device
    from topopt.gpu_mg.model import GpuParams, GpuSolver
    dev, case_name = device.select("auto", gpu_case)
    edof, active = edof_table(c.shape, c.labels == LOAD)
    free = c.free if fixed is None else np.setdiff1d(c.free, fixed)
    cfg = SimpleNamespace(settings=SimpleNamespace(robust="off"), material=Material(E=c.E, nu=c.nu))
    problem = SimpleNamespace(config=cfg, shape=c.shape, fixed_dofs=np.zeros(0, np.int64) if fixed is None else np.asarray(fixed),
                              h=c.h, springs=None, joints=jt)
    solver = GpuSolver(free, edof, active, problem, dev, case_name, 1, GpuParams(tol=rtol, min_free=min_free, backward_error=backward_error, **(params or {})))
    try:
        solver.set_design(np.full(edof.shape[0], c.E))
        u = solver.solve(f if fixed is None else f[np.searchsorted(c.free, free)])
        solver.end_design()
        if info is not None:
            info.update(case=case_name, levels=solver.host.n_levels, record=solver.records[-1]["iterations"][0],
                        border=None if solver.host.border is None else solver.host.border.summary())
    finally:
        solver.free()
    return u


def solve(c, jt, f, solver="splu", rtol=1e-12, info=None, gpu_case="auto"):
    """Displacement of the free dofs and the body motion. `solver`: splu (scipy), pardiso or amg (`BorderedSolver`), or gpu-mg (P2b)."""
    if solver == "gpu-mg":
        u = solve_gpu(c, jt, f, rtol=max(rtol, 1e-10), gpu_case=gpu_case, info=info)
        j = jt.joints[0]
        return u, -np.linalg.solve(j.D, j.C_free(c.free).T @ u)
    K = kb_added(c, jt)
    if solver == "splu":
        C, D = jt.border(c.free)
        A = sps.bmat([[K, C], [C.T, D]], format="csc")
        s = 1.0 / np.sqrt(A.diagonal())                       # Jacobi scaling, as `LinearSolver` does
        x = s * spla.splu((sps.diags(s) @ A @ sps.diags(s)).tocsc()).solve(s * np.concatenate([f, np.zeros(C.shape[1])]))
        return x[:len(f)], x[len(f):]
    ns = rigid_body_modes(c.free, c.shape, c.h) if solver == "amg" else None
    bs = jt.make_solver(solver, c.free, c.shape, c.h, ns, True)
    bs.rtol = rtol
    bs.factor(K)
    u = bs.solve(f)
    q = -np.linalg.solve(jt.joints[0].D, jt.joints[0].C_free(c.free).T @ u)
    bs.free()
    return u, q


def closed_form(a, F, M, E, G, A, I, J, L):
    """Compliance of a cantilever with the free end at the joint and the clamp at the far point.

    `a` is the unit vector from the joint to the far point; `x' = -a` points from the clamp to the free end.
    Frame (x', p, q) is right-handed; the sum does not depend on the choice of p for a round section.
    """
    xp = -np.asarray(a, float)
    p = np.cross(xp, np.eye(3)[np.argmin(np.abs(xp))])
    p /= np.linalg.norm(p)
    q = np.cross(xp, p)
    F, M = np.asarray(F, float), np.asarray(M, float)
    Fx, Fp, Fq, Mx, Mp, Mq = F @ xp, F @ p, F @ q, M @ xp, M @ p, M @ q
    c = Fx ** 2 * L / (E * A) + Mx ** 2 * L / (G * J)
    c += Fp ** 2 * L ** 3 / (3 * E * I) + Fp * Mq * L ** 2 / (E * I) + Mq ** 2 * L / (E * I)     # plane (x', p), moment about q
    c += Fq ** 2 * L ** 3 / (3 * E * I) - Fq * Mp * L ** 2 / (E * I) + Mp ** 2 * L / (E * I)     # plane (x', q), moment about p
    return c


def perpendicular_pair(a):
    """The unit vectors p and q that `closed_form` uses for the axis `a` (so that a test can load along them)."""
    xp = -np.asarray(a, float) / np.linalg.norm(a)
    p = np.cross(xp, np.eye(3)[np.argmin(np.abs(xp))])
    p /= np.linalg.norm(p)
    return xp, p, np.cross(xp, p)


def case(n, axis, load, L=20.0, d=2.0, ratio=1.0, c_bond=100.0, solver="splu", info=None, gpu_case="auto"):
    """`(engine compliance, closed form)` for a cube of `n` voxels; `load = (force, moment)`; `ratio = E_block / E_beam`."""
    c = cube(n)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    jt = make_joint(c, a * L, ("round", d), E_beam=c.E / ratio, k_a=c_bond * c.E / c.h)       # the area centroid is the origin
    j = jt.joints[0]
    f = nodal_load(c, *load)
    u, _ = solve(c, jt, f, solver, info=info, gpu_case=gpu_case)
    props = section_properties(SectionSpec("round", (d,)))
    E_beam = c.E / ratio
    ref = closed_form(a, load[0], load[1], E_beam, E_beam / (2 * (1 + 0.3)), props["A"], props["Iy"], props["J"],
                      float(np.linalg.norm(j.far - j.ref)))
    return float(f @ u), ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--solver", default="splu", choices=["splu", "pardiso", "amg", "gpu-mg"])
    ap.add_argument("--n", type=int, default=8, help="voxels on a side of the cube")
    ap.add_argument("--gpu-case", default="auto", choices=["auto", "B", "C"], help="with --solver gpu-mg: the precision case (B needs 64-bit float on the device)")
    a = ap.parse_args()
    for name, axis in (("x", (1, 0, 0)), ("oblique (1,2,2)", (1, 2, 2))):
        xp, p, q = perpendicular_pair(axis)
        loads = {"axial": (5 * xp, np.zeros(3)), "bending p": (5 * p, np.zeros(3)), "bending q": (5 * q, np.zeros(3)),
                 "torsion": (np.zeros(3), 5 * xp), "force p + moment q": (3 * p, 7 * q), "force q + moment p": (3 * q, 7 * p)}
        for lname, load in loads.items():
            info = {}
            got, ref = case(a.n, axis, load, solver=a.solver, info=info, gpu_case=a.gpu_case)
            extra = "" if not info else f"  (gpu-mg case {info['case']}, {info['levels']} levels, {info['record']})"
            print(f"{name:16s} {lname:20s} engine {got:.8e}  closed form {ref:.8e}  error {got / ref - 1:+.2e}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
