"""Joints: a beam from a load region to a fixed far point, with a bordered solve (CONTRACTS.md section 38).

A joint entry (`Config.joints`) names a load region. The exposed faces of the region (`loads.exposed_faces`, the
rule of the springs) are bonded to one rigid body with 6 dofs `q = (t, w)` about a reference point, and an
Euler-Bernoulli beam joins the body to a fixed far point. With the far end fixed, the beam adds a 6 x 6 matrix `B` on `q`.

The bond energy is `1/2 k_a int |u - R q|^2 dA`, `R = [I, -skew(x - ref)]`. With the consistent Q1 face matrix it gives
a term `Kb` on the grid dofs (added to `K` in place, as a spring), a coupling `C` (grid dofs x 6) and a block `D_bond`
(6 x 6). The bordered matrix is `[[K + Kb, C], [C^T, D_bond + B]]`. The supports do not depend on the density, so the
sensitivities do not change, and the force vector has zeros on the border.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from .config import JOINT_AXIS_OPTIONS, SectionSpec, UserError, joint_axis_problem
from .loads import exposed_faces, node_positions
from .solver import LinearSolver
from .springs import FACE_UNIT, NEAR_SINGULAR, SpringMatrix, face_nodes, face_triplets, sum_duplicates

BOND_C = 30.0     # the bond stiffness per area is `k_a = BOND_C E / h` (E of the run material, h the voxel size)
BOND_C_SOURCE = (
    "module constant c = 30 in k_a = c E / h (E: the run material, h: the voxel size). From "
    "measurements on the hook grid: the lowest AMG iteration count of four series, and a bond share of the compliance "
    "of 2e-5 (round d 2 mm beam of the run material) and 1.1e-3 (round d 6 mm beam, E 210000). "
    "'bond_stiffness_per_area' in the joint entry replaces it")
_SERIES = np.arange(1, 200, 2, dtype=np.float64)       # odd terms of the torsion series of a rectangle


# --- sections and the beam -----------------------------------------------------------------------


def rect_torsion_constant(b: float, h: float) -> float:
    """Torsion constant J of a solid rectangle b x h (mm4), by the usual series.

    With `a` the longer side and `t` the shorter one: `J = a t^3 (1/3 - 64/pi^5 (t/a) sum tanh(n pi a / 2t) / n^5)`,
    n = 1, 3, 5, ... (99 terms; the rest of the sum is below 1e-10). A square gives 0.140577 a^4.
    """
    a, t = max(b, h), min(b, h)
    s = float(np.sum(np.tanh(_SERIES * np.pi * a / (2.0 * t)) / _SERIES ** 5))
    return a * t ** 3 * (1.0 / 3.0 - 64.0 / np.pi ** 5 * (t / a) * s)


def section_properties(sec: SectionSpec) -> dict:
    """`A` (mm2), `Iy`, `Iz`, `J` (mm4) and `depth` (mm, the size that `L/d` uses) of a beam section.

    `Iy` resists bending in the plane (axis, z) and `Iz` in the plane (axis, y). `rect`: `b` runs along the section
    y axis and `h` along the section z axis. `depth`: `d`, `od`, the longer side of a rectangle, or the diameter of
    the circle of equal area (`props`).
    """
    v = sec.values
    if sec.kind in ("round", "tube"):
        od, idia = (v[0], 0.0) if sec.kind == "round" else v
        i = np.pi * (od ** 4 - idia ** 4) / 64.0
        return {"A": np.pi * (od ** 2 - idia ** 2) / 4.0, "Iy": i, "Iz": i, "J": 2.0 * i, "depth": od}
    if sec.kind == "rect":
        b, h = v
        return {"A": b * h, "Iy": b * h ** 3 / 12.0, "Iz": h * b ** 3 / 12.0, "J": rect_torsion_constant(b, h),
                "depth": max(b, h)}
    return {"A": v[0], "Iy": v[1], "Iz": v[2], "J": v[3], "depth": float(np.sqrt(4.0 * v[0] / np.pi))}


def beam_frame(axis, orient=None) -> np.ndarray:
    """Orthonormal `Q` (3, 3): its columns are the beam axis and the section y and z axes, in global coordinates.

    The section y axis is `orient` with its part along the beam removed. Without `orient` it comes from the global
    axis least parallel to the beam.
    """
    ex = np.asarray(axis, dtype=np.float64)
    ex = ex / np.linalg.norm(ex)
    helper = np.eye(3)[np.argmin(np.abs(ex))] if orient is None else np.asarray(orient, dtype=np.float64)
    ez = np.cross(ex, helper)
    ez /= np.linalg.norm(ez)
    return np.column_stack([ex, np.cross(ez, ex), ez])


def beam_stiffness(E, nu, props, length, Q) -> np.ndarray:
    """6 x 6 stiffness (global axes, dofs `t, w`) of the beam at the joint end when the far end is fixed.

    Euler-Bernoulli: `EA/L`, `GJ/L`, `12EI/L^3`, `6EI/L^2`, `4EI/L`. The beam goes from the joint to the far point
    along `Q[:, 0]`; the local y plane has `Iz`, the local z plane has `Iy` (the coupling sign flips).
    """
    G, L = E / (2.0 * (1.0 + nu)), length
    k = np.zeros((6, 6))
    k[0, 0], k[3, 3] = E * props["A"] / L, G * props["J"] / L
    for i, j, I, s in ((1, 5, props["Iz"], 1.0), (2, 4, props["Iy"], -1.0)):
        k[i, i], k[j, j] = 12.0 * E * I / L ** 3, 4.0 * E * I / L
        k[i, j] = k[j, i] = s * 6.0 * E * I / L ** 2
    T = np.zeros((6, 6))
    T[:3, :3] = T[3:, 3:] = Q
    return T @ k @ T.T


# --- the bond -------------------------------------------------------------------------------------


def _skew(r):
    """(..., 3, 3) matrices S with `S @ v = r x v`."""
    z = np.zeros(r.shape[:-1])
    return np.stack([np.stack([z, -r[..., 2], r[..., 1]], -1),
                     np.stack([r[..., 2], z, -r[..., 0]], -1),
                     np.stack([-r[..., 1], r[..., 0], z], -1)], -2)


def bond_blocks(nodes, shape, origin, h, ref, k_a) -> tuple:
    """`(Kb, C, crow, D)` of the bond with `k_a` (N/mm3) on the faces `nodes` (F, 4), rigid body about `ref`.

    `Kb` is `(rows, cols, vals)` on the grid dofs, sorted, duplicates added. `C` is a CSR matrix with the rows of the
    bonded dofs only, `(len(crow), 6)`; `crow` holds their grid dof numbers (sorted). The rows on fixed dofs are cut by
    the caller. `D` is `(6, 6)`: a 6 x 6 matrix of each face,
    then a pairwise sum over the faces (`np.sum` on a contiguous axis). One `einsum` over faces and nodes has a
    rounding error of 8e-14 of `max |D|`, and the pairwise sum cuts it by 30 to 800 times.
    """
    n_dof = 3 * int(np.prod(np.array(shape) + 1))
    kf = k_a * h * h
    pos = node_positions(nodes.ravel(), shape, origin, h).reshape(nodes.shape + (3,)) - np.asarray(ref, dtype=np.float64)
    R = np.zeros(nodes.shape + (3, 6))                   # u_point = R q, q = (t, w)
    R[..., :3] = np.eye(3)
    R[..., 3:] = -_skew(pos)
    Kb = sum_duplicates(*face_triplets(nodes, np.full(3, kf)), n_dof)
    M = np.einsum("ij,fjab->fiab", FACE_UNIT, R)         # (F, 4, 3, 6)
    rows = np.broadcast_to((3 * nodes[:, :, None] + np.arange(3))[..., None], M.shape)
    crow, inv = np.unique(rows.ravel(), return_inverse=True)     # the bonded dofs: C has no rows for the other grid dofs
    C = sps.csr_matrix((-kf * M.ravel(), (inv, np.broadcast_to(np.arange(6), M.shape).ravel())), shape=(crow.size, 6))
    Df = np.einsum("fiad,fiae->fde", R, M).reshape(len(R), 36)
    D = kf * np.ascontiguousarray(Df.T).sum(axis=1).reshape(6, 6)
    return Kb, C, crow, D


@dataclass
class Joint:
    """One joint: the bond on the grid, the beam, and the 6 x 6 blocks of the border."""

    name: str                       # the load region
    region_id: int                  # index in `regions.load`
    ref: np.ndarray                 # (3,) mm, reference point of the rigid body
    far: np.ndarray                 # (3,) mm, the fixed end of the beam
    B: np.ndarray                   # (6, 6) beam at the joint end
    D_bond: np.ndarray              # (6, 6)
    C: sps.csr_matrix               # (len(crow), 6): the rows of the bonded dofs
    crow: np.ndarray                # grid dof numbers of the rows of C, sorted
    rows: np.ndarray                # Kb of the bond on the grid dofs (all rows, also fixed ones)
    cols: np.ndarray
    vals: np.ndarray
    facts: dict = field(default_factory=dict)

    @property
    def D(self) -> np.ndarray:
        return self.D_bond + self.B

    def C_free(self, free_dofs) -> sps.csr_matrix:
        """`C` on the free dofs: `(n_free, 6)`. The rows of the bonded dofs that are fixed are cut."""
        fd = np.asarray(free_dofs, dtype=np.int64)
        if fd.size == 0:
            return sps.csr_matrix((0, 6))
        sorter = np.argsort(fd, kind="stable")
        at = sorter[np.minimum(np.searchsorted(fd, self.crow, sorter=sorter), fd.size - 1)]
        hit = np.flatnonzero(fd[at] == self.crow)
        coo = self.C[hit].tocoo()
        return sps.csr_matrix((coo.data, (at[hit][coo.row], coo.col)), shape=(fd.size, 6))

    def C_dot(self, q, n_dof) -> np.ndarray:
        """`C q` on all `n_dof` grid dofs (zero on the dofs that are not bonded)."""
        out = np.zeros(n_dof)
        out[self.crow] = self.C @ q
        return out


@dataclass
class JointSet:
    """The joints of a problem, the dofs of the grid, and the JSON `facts`."""

    joints: list
    n_dof: int
    region_ids: list
    origin: np.ndarray              # (3,) mm, the corner of the grid
    facts: dict = field(default_factory=dict)

    def triplets(self) -> tuple:
        """Kb of all bonds together: `(rows, cols, vals)`, sorted, duplicates added."""
        return sum_duplicates(np.concatenate([j.rows for j in self.joints]),
                              np.concatenate([j.cols for j in self.joints]),
                              np.concatenate([j.vals for j in self.joints]), self.n_dof)

    def border(self, free_dofs) -> tuple:
        """`(C, D)` of the bordered matrix: `C` is `(n_free, 6 m)` CSC, `D` is `(6 m, 6 m)` CSR (block diagonal)."""
        fd = np.asarray(free_dofs, dtype=np.int64)
        C = sps.hstack([j.C_free(fd) for j in self.joints], format="csc")
        return C, sps.block_diag([sps.csr_matrix(j.D) for j in self.joints], format="csr")

    def augmented_modes(self, Bg, free_dofs, shape, h) -> np.ndarray:
        """The rigid-body modes `Bg` (n_free, 6) of the grid with `6 m` rows for the bodies (near-null space for AMG).

        Translation: the identity on the 3 translation rows of each body. Rotation `n`: `e_n x (ref - x_mean)` on the
        translation rows (`x_mean` is the centre that `fe.rigid_body_modes` uses) and 1 on rotation row `n`. So a body
        that the part carries along does not turn against it.
        """
        fd = np.asarray(free_dofs, dtype=np.int64)
        node = fd // 3
        ny1, nz1 = shape[1] + 1, shape[2] + 1
        x = np.stack([node // (ny1 * nz1), (node // nz1) % ny1, node % nz1], axis=1).astype(np.float64) * h
        xbar = x.mean(axis=0)
        B = np.zeros((Bg.shape[0] + 6 * len(self.joints), 6))
        B[:Bg.shape[0]] = Bg
        for k, j in enumerate(self.joints):
            r0 = Bg.shape[0] + 6 * k
            c = (j.ref - self.origin) - xbar
            B[r0:r0 + 3, :3] = np.eye(3)
            for n in range(3):
                B[r0:r0 + 3, 3 + n] = np.cross(np.eye(3)[n], c)
                B[r0 + 3 + n, 3 + n] = 1.0
        return B

    def make_solver(self, name, free_dofs, shape, h, nullspace, block3) -> "BorderedSolver":
        """The solver of a run with joints. `nullspace` is the grid's `rigid_body_modes` for AMG, else None."""
        ns = None if nullspace is None else self.augmented_modes(nullspace, free_dofs, shape, h)
        return BorderedSolver(name, self.border(free_dofs), nullspace=ns, block3=block3)

    def border_terms(self, u) -> list:
        """For each joint `(q, g)`: the body motion `q = -D^-1 C^T u` and the nodal force `g = Kb u + C q` of the bond.

        `u` is the displacement of all grid dofs (zero on fixed dofs). The force of the bond on the part is `-g`.
        The sum of the wrench of `g` about `ref` is `B q`.
        """
        out = []
        for j in self.joints:
            q = -np.linalg.solve(j.D, j.C.T @ u[j.crow])
            g = np.bincount(j.rows, weights=j.vals * u[j.cols], minlength=self.n_dof) + j.C_dot(q, self.n_dof)
            out.append((q, g))
        return out


def support_matrix(problem, assembler, free_dofs):
    """`Kb` of the springs and of the bonds of the joints on the pattern of `assembler`'s matrix, or None without both."""
    sp, jt = problem.springs, problem.joints
    if jt is None:
        return None if sp is None else sp.matrix(assembler, free_dofs)
    parts = ([sp.triplets()] if sp is not None else []) + [jt.triplets()]
    rows, cols, vals = sum_duplicates(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                                      np.concatenate([p[2] for p in parts]), jt.n_dof)
    return SpringMatrix(rows, cols, vals, assembler._indptr, assembler._indices, free_dofs)


def _material_from(spec) -> str:
    """Where the beam `E` and `nu` come from: the run material, the joint entry, or one of each."""
    if spec.E is None and spec.nu is None:
        return "the run material"
    if spec.E is not None and spec.nu is not None:
        return "the joint entry"
    return "the joint entry (E) and the run material (nu)" if spec.E is not None else "the joint entry (nu) and the run material (E)"


def beam_facts(props, length, Q, B, D_bond, E_ref: float, h: float, orient=None) -> dict:
    """The facts of a joint that show a bond that is too soft or a beam that is nearly free: `depth_mm`, `length_over_depth`, `frame`, `rho`,
    `beam_min_translation_over_Eh` and `beam_min_rotation_over_Eh3`.

    `E_ref` is the modulus of the run material and `h` the voxel size of the run (the unit of the two stiffness ratios); `orient` is the
    `orient` of the entry (None: the default section axis). One function for
    `build_joints` (the voxel model) and `topopt verify` (`verify/supports_p2.py`), so the two records hold the same numbers.
    """
    ratio = np.diag(D_bond) / np.diag(B)
    return {"depth_mm": props["depth"], "length_over_depth": length / props["depth"],
            "frame": {"x": [float(v) for v in Q[:, 0]], "y": [float(v) for v in Q[:, 1]], "z": [float(v) for v in Q[:, 2]],
                      "y_from": "orient" if orient is not None else "default: the global axis least parallel to the beam"},
            "rho": float(ratio.min()),
            "beam_min_translation_over_Eh": float(np.linalg.eigvalsh(B[:3, :3])[0]) / (E_ref * h),
            "beam_min_rotation_over_Eh3": float(np.linalg.eigvalsh(B[3:, 3:])[0]) / (E_ref * h ** 3)}


def build_joints(config, labels, load_ids, origin, h: float) -> JointSet:
    """The bond blocks and the beam of every `config.joints` entry. Raise `UserError` for a joint that cannot work."""
    index = {lr.name: i for i, lr in enumerate(config.regions.load)}
    shape = labels.shape
    n_dof = 3 * int(np.prod(np.array(shape) + 1))
    mat = config.material
    joints, facts = [], []
    for spec in config.joints:
        where = f"joint on load region '{spec.region}'"
        faces = exposed_faces(labels, load_ids == index[spec.region])
        nodes = face_nodes(faces, shape)
        if len(nodes) == 0:
            raise UserError(
                f"{where}: no face of the region looks toward empty space or a keep-out region (faces toward design, "
                "fixed or other load voxels are not bonded). Options: shape the region so that the bonded face is free "
                "(a sleeve or a pad has its free face away from the design), or use another region")
        pos = node_positions(nodes.ravel(), shape, origin, h).reshape(nodes.shape + (3,))
        ref = np.array(spec.origin) if spec.origin is not None else pos.mean(axis=1).mean(axis=0)   # all faces have one area
        far = np.array(spec.to)
        bad = joint_axis_problem(ref, far, spec.orient)
        if bad:
            raise UserError(f"{where}: {bad}. {JOINT_AXIS_OPTIONS}")
        axis = far - ref
        length = float(np.linalg.norm(axis))
        Q = beam_frame(axis, spec.orient)
        E = mat.E if spec.E is None else spec.E
        nu = mat.nu if spec.nu is None else spec.nu
        props = section_properties(spec.section)
        k_a = spec.bond_stiffness_per_area if spec.bond_stiffness_per_area is not None else BOND_C * mat.E / h
        B = beam_stiffness(E, nu, props, length, Q)
        Kb, C, crow, D_bond = bond_blocks(nodes, shape, origin, h, ref, k_a)
        bf = beam_facts(props, length, Q, B, D_bond, mat.E, h, spec.orient)       # rho, the smallest beam stiffness over E h and over E h^3, L/d, the frame
        share = float(sla.eigh(B, B + D_bond, eigvals_only=True).max())
        area = len(nodes) * h * h
        by_dir = {("-" if sg < 0 else "+") + "xyz"[ax]: int(m.sum()) for ax, sg, m in faces if m.any()}
        f = {"region": spec.region, "faces": len(nodes), "faces_by_direction": by_dir, "area_mm2": area,
             "nodes": int(np.unique(nodes).size),
             "reference_mm": [float(v) for v in ref],
             "reference_from": "the joint entry (origin)" if spec.origin is not None else "the area centroid of the bonded faces",
             "to_mm": [float(v) for v in far], "length_mm": length, "axis": [float(v) for v in Q[:, 0]],
             "section": spec.section.as_dict(), "section_area_mm2": props["A"], "iy_mm4": props["Iy"],
             "iz_mm4": props["Iz"], "j_mm4": props["J"], "depth_mm": bf["depth_mm"],
             "length_over_depth": bf["length_over_depth"],
             "beam_E_mpa": E, "beam_nu": nu, "beam_G_mpa": E / (2.0 * (1.0 + nu)),
             "beam_material_from": _material_from(spec),
             "frame": bf["frame"],
             "bond_stiffness_per_area_n_per_mm3": k_a,
             "bond_stiffness_from": ("the joint entry (bond_stiffness_per_area)" if spec.bond_stiffness_per_area is not None
                                     else "c E / h with c = %g" % BOND_C),
             "c_equivalent": k_a * h / mat.E,
             "bond_diagonal": [float(v) for v in np.diag(D_bond)], "beam_diagonal": [float(v) for v in np.diag(B)],
             "rho": bf["rho"], "bond_share_max": share,
             "beam_min_translation_over_Eh": bf["beam_min_translation_over_Eh"],
             "beam_min_rotation_over_Eh3": bf["beam_min_rotation_over_Eh3"]}
        joints.append(Joint(spec.region, index[spec.region], ref, far, B, D_bond, C, crow, *Kb, facts=f))
        facts.append(f)
    return JointSet(joints, n_dof, [j.region_id for j in joints], np.asarray(origin, dtype=np.float64),
                    {"joints": facts, "bordered_dofs": 6 * len(joints),
                     "bond": {"c": BOND_C, "source": BOND_C_SOURCE}})


# --- memory ---------------------------------------------------------------------------------------


def stiffness_nnz(edof, free_dofs) -> int:
    """Stored entries of the stiffness matrix `K` on the free dofs, from the dof table (exact for `Assembler`'s pattern).

    Two nodes are coupled when they share an element, and each coupling gives a 3 x 3 block: the entries are the sum
    over the coupled nodes `a, b` of `(free dofs of a) * (free dofs of b)`.
    """
    nodes, inv = np.unique(np.asarray(edof)[:, 0::3] // 3, return_inverse=True)
    inv = inv.reshape(-1, 8)
    n_el = inv.shape[0]
    inc = sps.csr_matrix((np.ones(inv.size), (np.repeat(np.arange(n_el), 8), inv.ravel())), shape=(n_el, nodes.size))
    pattern = (inc.T @ inc).tocsr()
    pattern.data[:] = 1.0
    f = np.bincount(np.searchsorted(nodes, np.asarray(free_dofs) // 3), minlength=nodes.size).astype(np.float64)
    return int(round(float(f @ (pattern @ f))))


def bordered_copy_bytes(nnz: int) -> int:
    """Bytes of the bordered copy of `K` that `BorderedSolver.factor` builds in each pass: the CSC matrix (value and row index of
    each entry) and the transient COO list of `sps.bmat` (value, row and column of each entry). The sizes are those of the dtypes."""
    f8, i4 = np.dtype(np.float64).itemsize, np.dtype(np.int32).itemsize
    return int(nnz) * ((f8 + i4) + (f8 + 2 * i4))


# --- the solver -----------------------------------------------------------------------------------


class BorderedSolver(LinearSolver):
    """`LinearSolver` for the bordered matrix `[[K, C], [C^T, D]]` of a run with joints.

    The caller gives and gets vectors of the grid dofs (`n_free` rows): `factor(K)` borders the matrix, `solve(B)`
    pads `B` with `6 m` zero rows and cuts the result to `n_free` rows. The warm start of AMG and the checkpoint
    state keep the bordered size, and the same class restores them.
    """

    def __init__(self, method, border, nullspace=None, **kw):
        super().__init__(method, nullspace=nullspace, **kw)
        self.C, self.D = border
        self.n_free, self.n_border = self.C.shape

    def factor(self, K, nullspace=None):
        super().factor(sps.bmat([[K, self.C], [self.C.T, self.D]], format="csc"), nullspace)

    def solve(self, B, warm: bool = True):
        B = np.asarray(B, dtype=np.float64)
        one = B.ndim == 1
        B = B.reshape(self.n_free, -1)
        padded = np.zeros((self.n_free + self.n_border, B.shape[1]))
        padded[:self.n_free] = B
        X = super().solve(padded, warm)[:self.n_free]
        return X[:, 0] if one else X


# --- reactions ------------------------------------------------------------------------------------


def joint_wrenches(j: Joint, q) -> dict:
    """The wrench on the part and the ground reaction of joint `j` with the body motion `q`.

    On the part: `-B q`, the force and the moment about the reference point. The ground reaction is the wrench that
    the fixed end gives the beam: the force `-(B q)[:3]` moved to the far point (moment about the far point).
    """
    f = j.B @ q
    part = -f
    ground_moment = -f[3:] - np.cross(j.ref - j.far, f[:3])
    return {"part": {"force_n": [float(v) for v in part[:3]], "moment_nmm": [float(v) for v in part[3:]],
                     "about_mm": [float(v) for v in j.ref]},
            "ground": {"force_n": [float(v) for v in -f[:3]], "moment_nmm": [float(v) for v in ground_moment],
                       "about_mm": [float(v) for v in j.far]},
            "translation_mm": [float(v) for v in q[:3]], "rotation_rad": [float(v) for v in q[3:]]}


# --- texts ----------------------------------------------------------------------------------------


def joint_lines(facts: dict | None) -> list:
    """Text lines for `topopt check` and the run log from the `joints` facts (an empty list for None)."""
    out = []
    for f in (facts or {}).get("joints", []):
        v = lambda x: "[" + ", ".join(f"{a:.4g}" for a in x) + "]"
        by_dir = ", ".join(f"{k} {n}" for k, n in f["faces_by_direction"].items())
        sec = ", ".join(f"{k} {a:g}" for k, a in f["section"].items() if k != "kind")
        out.append(
            f"joint:          '{f['region']}': {f['faces']} faces ({by_dir}), {f['area_mm2']:.4g} mm2, bonded to a rigid body at "
            f"{v(f['reference_mm'])} mm ({f['reference_from']}); beam to {v(f['to_mm'])} mm, length {f['length_mm']:.4g} mm "
            f"(L/d {f['length_over_depth']:.3g}, Euler-Bernoulli), section {f['section']['kind']} {sec}, "
            f"E {f['beam_E_mpa']:g} MPa, nu {f['beam_nu']:g} ({f['beam_material_from']})")
        y = f["frame"]
        out.append(
            f"joint frame:    '{f['region']}': beam axis {v(f['axis'])}, section y axis {v(y['y'])} ({y['y_from']}), "
            f"A {f['section_area_mm2']:.4g} mm2, Iy {f['iy_mm4']:.4g}, Iz {f['iz_mm4']:.4g}, J {f['j_mm4']:.4g} mm4")
        out.append(
            f"joint beam:     '{f['region']}': smallest beam stiffness {f['beam_min_translation_over_Eh']:.3g} of E h (translation) and "
            f"{f['beam_min_rotation_over_Eh3']:.3g} of E h^3 (rotation); E h of the run material, h the voxel size")
        out.append(
            f"joint bond:     '{f['region']}': {f['bond_stiffness_per_area_n_per_mm3']:.4g} N/mm3 = c {f['c_equivalent']:.4g} E/h "
            f"({f['bond_stiffness_from']}); rho (smallest bond/beam stiffness ratio of the six dofs) {f['rho']:.3g}; "
            f"the bond holds at most {100 * f['bond_share_max']:.3g} percent of the joint flexibility")
    if facts and facts.get("joints"):
        out.append(f"joints:         {facts['bordered_dofs']} border dofs in the stiffness matrix; bond c: {facts['bond']['source']}")
    return out


def near_singular_lines(facts: dict | None, E: float, h: float) -> list:
    """One line for each joint whose smallest beam stiffness is below `springs.NEAR_SINGULAR` of `E h` (translation) or `E h^3`
    (rotation): the same pattern and the same constant as the spring lines (`check` and `run`)."""
    out = []
    for f in (facts or {}).get("joints", []):
        low = [f"{f[key]:.1e} of {unit} ({axis})" for key, unit, axis in
               (("beam_min_translation_over_Eh", "E h", "translation"), ("beam_min_rotation_over_Eh3", "E h^3", "rotation"))
               if f[key] < NEAR_SINGULAR]
        if low:
            out.append(f"joint on '{f['region']}': its smallest beam stiffness is " + " and ".join(low)
                       + f" (E {E:g} MPa x voxel {h:g} mm), below {NEAR_SINGULAR:g}: the part is nearly free on that axis, so the "
                       "stiffness matrix is nearly singular and the displacement can be large or wrong. "
                       "Options: raise the section or E, shorten the beam, or add a fixed region")
    return out


# --- buckling (CONTRACTS.md sections 33, 38, 39) ----------------------------------------------------


class CondensedStiffness(spla.LinearOperator):
    """The stiffness `S = K + Kb - C D^-1 C^T` of the grid dofs of a run with joints, as an operator.

    `K` is the sparse grid matrix with `Kb` (springs and bonds) added, as `Model.stiffness` gives it. `C` and `D` are the border
    of the bordered matrix; they do not depend on the density. The grid block of the inverse of the bordered matrix is `S^-1`, so
    `S` is the matrix that the bordered solve inverts: the static solution of a load case is `u = S^-1 f`. The buckling eigen
    problem uses `S` where a run without joints uses `K`. `S` is symmetric. `toarray()` gives it dense (for small problems).
    """

    def __init__(self, K, border: "Border"):
        super().__init__(dtype=np.float64, shape=K.shape)
        self.K, self.border = K, border

    def _matvec(self, v):
        v = np.asarray(v, dtype=np.float64).reshape(self.shape[0], -1)
        out = self.K @ v - self.border.C @ self.border.solve(self.border.CT @ v)
        return out.reshape(-1) if out.shape[1] == 1 else out

    def _matmat(self, X):
        return self._matvec(X)

    def _adjoint(self):
        return self

    def toarray(self) -> np.ndarray:
        C = self.border.C.toarray()
        return self.K.toarray() - C @ self.border.solve(C.T)


class Border:
    """The border `(C, D)` of the free dofs, with the factor of `D` (6 m x 6 m, block diagonal; computed once)."""

    def __init__(self, C, D):
        self.C = sps.csc_matrix(C)
        self.CT = self.C.T.tocsr()
        self.lu = sla.lu_factor(D.toarray())

    def solve(self, X):
        """`D^-1 X` for a vector or a matrix with 6 m rows."""
        return sla.lu_solve(self.lu, X)

    def condense(self, K) -> CondensedStiffness:
        return CondensedStiffness(K, self)


def condensed_border(joint_set: JointSet, free_dofs) -> Border:
    """The `Border` of the joints of a problem on the free dofs."""
    return Border(*joint_set.border(free_dofs))


BUCKLING_BEAMS_FACT = (
    "This run has joints (section 38). The eigen problem uses the stiffness S = K + Kb - C D^-1 C^T of the grid dofs, which is the "
    "matrix that the bordered solve inverts (the bond and the beam are inside S). The beams have no geometric stiffness in this "
    "model, so the load factors are for buckling of the part and not for buckling of the beams (the tubes or the rods that the joints stand for).")
BUCKLING_BEAMS_OPTION = ("To check a beam, use its own load of buckling (for example the Euler load of its section and length, "
                         "with the axial force that the run reports for the joint). This check does not do it.")


def buckling_note(n_joints: int) -> dict:
    """The part of the `buckling` record (metrics.json) that a run with joints adds: what the load factors are for."""
    return {"joints": n_joints, "stiffness": "S = K + Kb - C D^-1 C^T (the Schur complement of the bordered matrix)",
            "beams_have_geometric_stiffness": False, "meaning": "the load factors are for buckling of the part, not of the beams"}
