"""Joints and springs on the P2 tet model of `topopt verify`. CONTRACTS.md sections 16, 35 and 38.

The voxel optimizer holds a part with springs (`topopt/springs.py`) and joints (`topopt/joints.py`). This module gives the P2 model the same
physics, so `topopt verify` can check a part that has `fixed: []` and joints.

- Spring: `k_d / A` is the stiffness per area on axis `d`. The matrix `k N^T N` is integrated over the boundary facets of the region and added
  to `K`. With `stiffness` (N/mm, the total of the region) `A` is the area of the tet facets. With `stiffness_per_area` the value is used as it is.
- Joint: the bond energy is `1/2 k_a int |u - R q|^2 dA` on the boundary facets of the joint region. `q = (t, w)` are the 6 dofs of a rigid body,
  `R = [I, -skew(x - ref)]`. It gives `Kb` (added to `K`), the coupling `C` and the block `D_bond`. The beam matrix `B` comes from
  `joints.beam_stiffness` (the same function as the voxel model). The bordered matrix is `[[K + Kb, C], [C^T, D_bond + B]]`. It is one CSR
  matrix with one factor and no Schur step: `K` alone is singular when the YAML has `fixed: []`.
- The penalty is `k_a = c E / h` with `c = 30` (`joints.BOND_C`) and `h` the voxel size of the run, or `bond_stiffness_per_area` of the entry.
  The reference point is `origin` of the entry, or the area centroid of the facets of the region (the voxel model uses the centroid of its faces).
- Regions: the facets of a support region are the boundary facets of the tet mesh with 3 vertices in the band of the region STL (the rule of a
  load region), without the facets that look toward the part. This is the rule of the voxel model (`exposed_faces`: a face toward design,
  fixed or other load voxels is not bonded, a face toward empty space or a keep-out region is): a facet is dropped when the point `step`
  along its outward normal lies in the design, a fixed or a load region, and not in a keep-out region (`bonded_facets`). The two sets of
  faces are still not the same sets (the tet surface is a marching-cubes surface); the record gives both areas and their ratio.
- The beam has no geometric stiffness. The buckling check of the tet model uses `S = K + Kb - C D^-1 C^T` and `buckling_note` says so.

Provenance: the bond system, the bordered matrix and the CSR build are the ideas of `pipeline/lib/lug_physics.py` (`build_bond_system`,
`assemble_bordered_system`) and `pipeline/lib/p2_core.py` (`bordered_csr`) of an earlier pipeline of the same author (no licence header). The code is
written again for the YAML of this repo: the facet integral uses the P2 shape functions of `P2Model` and a quadrature rule of scikit-fem, the
beam is `topopt.joints.beam_stiffness`, and the border is `topopt.joints.Border` (not copied).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps

from ..config import UserError, joint_axis_problem
from ..joints import (BOND_C, BOND_C_SOURCE, BUCKLING_BEAMS_OPTION, Border, beam_facts, beam_frame, beam_stiffness,
                      joint_wrenches, near_singular_lines, section_properties)
from .p2 import require_skfem

KIND = ("joints (bond to a rigid body and an Euler-Bernoulli beam) and springs on the P2 tet model, with the physics of the voxel model "
        "(CONTRACTS.md sections 35 and 38)")
FACETS_NOTE = ("The facets of a region are the boundary facets of the tet mesh with 3 vertices in the band of the region STL, without the "
               "facets that look toward the part: a facet is dropped when the point neighbour_step_mm along its outward normal lies in the "
               "design, a fixed or a load region and not in a keep-out region. This is the rule of the voxel model, which bonds a face "
               "toward empty space or a keep-out region and no face toward design, fixed or load voxels. The rule reads the regions of the "
               "YAML, not the density of the run. The two sets of faces are still not the same: the tet surface is a marching-cubes "
               "surface. area_mm2 is the bonded area of the tet mesh and area_over_voxel_area is that area over the bonded area of the "
               "voxel run (voxel_area_mm2). band_facets and band_area_mm2 are the facets before the rule; facets_toward_design are the "
               "facets that it dropped.")
TRI_ORDER = 4              # quadrature order on a facet: exact for N^T N (degree 4), N R (degree 3) and R^T R (degree 2)
BUCKLING_LINE = ("With joints the geometric stiffness Kg is that of the tets only. The beam has no Kg, so the load factors are for "
                 "buckling of the part and not of the beams.")


@dataclass
class Bonded:
    """The facets of one support region that carry the bond or the spring, and what the rule of the voxel model dropped (`bonded_facets`)."""
    facets: np.ndarray             # (k, 3) vertex indices of the bonded facets
    facet_area: np.ndarray         # (k,) mm2
    band_facets: int               # the facets of the region rule (band), before the voxel rule
    band_area_mm2: float
    toward_design: int             # the facets that look toward the part: not bonded
    toward_design_area_mm2: float


def bonded_facets(model, sel, neighbours: list, keep_out: list, step: float) -> Bonded:
    """Apply the voxel rule to the facets `sel.facets` of a support region (`regions.Selection` with `elems`).

    `neighbours` are the `trimesh` meshes of the design, the fixed regions and the load regions (the voxel labels DESIGN, FIXED and LOAD),
    `keep_out` the meshes of the keep-out regions (the label KEEP_OUT wins over the others, as in `voxelize`). A facet is dropped when the
    point `step` (mm) along its outward normal lies inside one of `neighbours` and in none of `keep_out`. `step` is the distance of the
    centre of the neighbour voxel from a voxel face.
    """
    from .regions import outward_normals, signed_distance
    facets, area = sel.facets, sel.facet_area
    toward = np.zeros(len(facets), bool)
    if len(facets):
        verts = model.p.T
        pts = verts[facets].mean(axis=1) + step * outward_normals(verts, model.t.T, facets, sel.elems)
        for mesh in neighbours:
            todo = np.flatnonzero(~toward)
            if len(todo):
                toward[todo[signed_distance(mesh, pts[todo], step)[0] > 0.0]] = True
        for mesh in keep_out:
            cand = np.flatnonzero(toward)
            if len(cand):
                toward[cand[signed_distance(mesh, pts[cand], step)[0] > 0.0]] = False
    keep = ~toward
    return Bonded(facets[keep], area[keep], int(len(facets)), float(area.sum()), int(toward.sum()), float(area[toward].sum()))


def bordered_copy_bytes(nnz: int) -> int:
    """Bytes of the bordered copy of `Kff` (`bordered_csr`) that a run with joints keeps next to `Kff`: a value (float64) and a column index
    (int32) for each stored entry of `Kff`. The `6 m` border rows and columns are small and are not counted."""
    return int(nnz) * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)


def voxel_areas(run: dict) -> dict:
    """`{region: bonded area of the voxel run, mm2}` from the `run.json` dict (`joints.joints[]` and `springs.regions[]`)."""
    out = {}
    for rec in ((run.get("joints") or {}).get("joints") or []) + ((run.get("springs") or {}).get("regions") or []):
        if isinstance(rec, dict) and rec.get("area_mm2") is not None:
            out[rec["region"]] = float(rec["area_mm2"])
    return out


def voxel_frames(run: dict) -> dict:
    """`{region: section y axis}` of the joints of the voxel run, from `joints.joints[].frame.y` of the `run.json` dict."""
    return {rec["region"]: list(rec["frame"]["y"]) for rec in ((run.get("joints") or {}).get("joints") or [])
            if isinstance(rec, dict) and isinstance(rec.get("frame"), dict) and rec["frame"].get("y") is not None}


def _area_facts(s, voxel_area) -> dict:
    """The facts of the bonded area of a region: the voxel area, the ratio, and (for a `Bonded`) what the voxel rule dropped."""
    area = float(s.facet_area.sum())
    out = dict(voxel_area_mm2=voxel_area, area_over_voxel_area=None if not voxel_area else area / voxel_area)
    if isinstance(s, Bonded):
        out.update(band_facets=s.band_facets, band_area_mm2=s.band_area_mm2, facets_toward_design=s.toward_design,
                   area_toward_design_mm2=s.toward_design_area_mm2)
    return out


def check_bonded(cfg, bonded: dict) -> None:
    """Raise `UserError` for a joint or spring region that has no facet to bond (the voxel model stops in the same case)."""
    for kind, specs in (("joint", cfg.joints), ("spring", cfg.springs)):
        for spec in specs:
            if not len(bonded[spec.region].facets):
                raise _no_facet(kind, spec.region, bonded[spec.region])


def _no_facet(kind: str, region: str, s) -> UserError:
    return UserError(
        f"{kind} on load region '{region}': no facet of the region looks away from the part"
        + (f" ({s.band_facets} facets in the band, all look toward the design, a fixed or a load region)" if isinstance(s, Bonded) else "")
        + ". The voxel model bonds a face toward empty space or a keep-out region and no face toward the part. "
        "Options: shape the region so that the bonded face is free, or use another region")


def _shape_data(model, facets):
    """The P2 facet data: node ids `(k, 6)`, shape values `N (nq, 6)`, weights `(k, nq)` (area times rule weight), points `(k, nq, 3)`.

    The 6 nodes of a facet are its 3 vertices, then the edges (0,1), (1,2), (0,2): the order of `bc.traction_load`.
    """
    require_skfem()
    from skfem.quadrature import get_quadrature_tri
    f = np.asarray(facets, np.int64)
    ids = np.concatenate([f, np.stack([model.edge_node_ids(f[:, 0], f[:, 1]), model.edge_node_ids(f[:, 1], f[:, 2]),
                                       model.edge_node_ids(f[:, 0], f[:, 2])], axis=1)], axis=1)
    X, W = get_quadrature_tri(TRI_ORDER)
    lam = np.column_stack([1.0 - X.sum(axis=0), X[0], X[1]])                    # (nq, 3)
    N = np.column_stack([lam[:, 0] * (2 * lam[:, 0] - 1), lam[:, 1] * (2 * lam[:, 1] - 1), lam[:, 2] * (2 * lam[:, 2] - 1),
                         4 * lam[:, 0] * lam[:, 1], 4 * lam[:, 1] * lam[:, 2], 4 * lam[:, 0] * lam[:, 2]])
    P = model.p.T[f]                                                            # (k, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    w = area[:, None] * (2.0 * np.asarray(W))[None, :]
    pts = np.einsum("qb,kbd->kqd", lam, P)
    return ids, N, w, pts, area


def _face_matrix(model, ids, N, w, axes_k) -> sps.csr_matrix:
    """`sum_d k_d int N_a N_b dA` on dof `3 a + d`, for the facets with the node ids `ids` and the stiffness per axis `axes_k` (3,)."""
    M = np.einsum("kq,qa,qb->kab", w, N, N)                                     # (k, 6, 6)
    rows, cols, vals = [], [], []
    for d in range(3):
        if axes_k[d] > 0.0:
            rows.append(np.broadcast_to(3 * ids[:, :, None] + d, M.shape).ravel())
            cols.append(np.broadcast_to(3 * ids[:, None, :] + d, M.shape).ravel())
            vals.append((axes_k[d] * M).ravel())
    n = model.N
    return sps.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()


def _skew(r):
    z = np.zeros(r.shape[:-1])
    return np.stack([np.stack([z, -r[..., 2], r[..., 1]], -1), np.stack([r[..., 2], z, -r[..., 0]], -1),
                     np.stack([-r[..., 1], r[..., 0], z], -1)], -2)


def bond_blocks_p2(model, facets, ref, k_a) -> tuple:
    """`(Kb, C, D)` of the bond of the facets to a rigid body about `ref`, with the penalty `k_a` (N/mm3).

    `Kb` is `(N, N)` CSR, `C` is `(N, 6)` CSR and `D` is `(6, 6)`. The energy is `1/2 k_a int |u - R q|^2 dA`.
    """
    ids, N, w, pts, _area = _shape_data(model, facets)
    Kb = _face_matrix(model, ids, N, w, np.full(3, k_a))
    R = np.zeros(pts.shape[:2] + (3, 6))                                        # u at the point of the body = R q
    R[..., :3] = np.eye(3)
    R[..., 3:] = -_skew(pts - np.asarray(ref, dtype=np.float64))
    CN = -k_a * np.einsum("kq,qa,kqcj->kacj", w, N, R)                          # (k, 6, 3, 6): the coupling of node a, component c
    rows = (3 * ids[:, :, None] + np.arange(3)[None, None, :])[..., None] * np.ones(6, dtype=np.int64)
    cols = np.broadcast_to(np.arange(6), CN.shape)
    C = sps.coo_matrix((CN.ravel(), (rows.ravel(), cols.ravel())), shape=(model.N, 6)).tocsr()
    D = k_a * np.einsum("kq,kqci,kqcj->ij", w, R, R)
    return Kb, C, 0.5 * (D + D.T)


def spring_matrix_p2(model, facets, k_per_area) -> sps.csr_matrix:
    """`Kb` of a spring with the stiffness per area `k_per_area` (3,) N/mm3 on the boundary facets `facets`: `k int N^T N dA`."""
    ids, N, w, _pts, _area = _shape_data(model, facets)
    return _face_matrix(model, ids, N, w, np.asarray(k_per_area, dtype=np.float64))


def bordered_csr(K: sps.csr_matrix, C: sps.spmatrix, D: sps.spmatrix, chunk_rows: int = 400_000) -> sps.csr_matrix:
    """`[[K, C], [C^T, D]]` as one CSR matrix, built without a COO round trip.

    `K` is `(n, n)`, `C` is `(n, nb)` and `D` is `(nb, nb)`. The result equals `scipy.sparse.bmat([[K, C], [C.T, D]])`.
    """
    n = K.shape[0]
    K = sps.csr_matrix(K)
    K.sort_indices()
    Cc = sps.csr_matrix(C)
    Cc.sort_indices()
    nb = Cc.shape[1]
    bottom = sps.hstack([Cc.T.tocsr(), sps.csr_matrix(D)], format="csr")
    bottom.sort_indices()
    len_k, len_c = np.diff(K.indptr), np.diff(Cc.indptr)
    new_ptr = np.concatenate([[0], np.cumsum(len_k + len_c)]).astype(np.int64)
    top = int(new_ptr[-1])
    indices = np.empty(top + bottom.nnz, np.int32)
    data = np.empty(top + bottom.nnz, np.float64)
    for r0 in range(0, n, chunk_rows):
        r1 = min(r0 + chunk_rows, n)
        k0, k1 = K.indptr[r0], K.indptr[r1]
        rows = np.repeat(np.arange(r0, r1), len_k[r0:r1])
        pos = np.arange(k0, k1, dtype=np.int64) + Cc.indptr[rows]
        indices[pos] = K.indices[k0:k1]
        data[pos] = K.data[k0:k1]
        c0, c1 = Cc.indptr[r0], Cc.indptr[r1]
        rows_c = np.repeat(np.arange(r0, r1), len_c[r0:r1])
        pos_c = new_ptr[rows_c] + len_k[rows_c] + (np.arange(c0, c1, dtype=np.int64) - Cc.indptr[rows_c])
        indices[pos_c] = Cc.indices[c0:c1] + n
        data[pos_c] = Cc.data[c0:c1]
    indices[top:] = bottom.indices
    data[top:] = bottom.data
    indptr = np.concatenate([new_ptr, top + bottom.indptr[1:]]).astype(np.int64)
    return sps.csr_matrix((data, indices, indptr), shape=(n + nb, n + nb))


# --------------------------------------------------------------------------- #
# The support set
# --------------------------------------------------------------------------- #

@dataclass
class PJoint:
    """One joint on the tet model: the bond blocks on all dofs, the beam, and the facts."""
    name: str
    ref: np.ndarray
    far: np.ndarray
    B: np.ndarray                  # (6, 6) beam at the joint end
    D_bond: np.ndarray             # (6, 6)
    Kb: sps.csr_matrix             # (N, N)
    C: sps.csr_matrix              # (N, 6)
    facts: dict = field(default_factory=dict)

    @property
    def D(self) -> np.ndarray:
        return self.D_bond + self.B


@dataclass
class PSpring:
    name: str
    Kb: sps.csr_matrix
    facts: dict = field(default_factory=dict)


class SupportSystem:
    """The bordered matrix of one level and the solves on it. Built by `SupportSet.system(Kff, free)`.

    The grid vectors (`n_free` rows) go in and come out of `RefinedSystem`; the `6 m` border rows are inside. A set with springs only
    has no border: the matrix is `K + Kb` and the padding is empty.
    """

    def __init__(self, Kff, C, D, free):
        self.Kff, self.C, self.D, self.free = Kff, C, D, free
        self.n_free, self.n_border = Kff.shape[0], C.shape[1]
        self.A = Kff if self.n_border == 0 else bordered_csr(Kff, C, D)

    def padded(self, rf) -> "RefinedSystem":
        return RefinedSystem(rf, self)

    def grid_solver(self, fac):
        """The object for the buckling check: `solve(v)` gives the grid block of the bordered solve, which is `S^-1 v`."""
        return fac if self.n_border == 0 else _GridBlock(fac, self.n_free, self.n_border)

    def stiffness(self):
        """The matrix of the buckling eigen problem: `K + Kb` without joints, else `S = K + Kb - C D^-1 C^T` as an operator."""
        return self.Kff if self.n_border == 0 else Border(self.C, self.D).condense(self.Kff)


class _GridBlock:
    def __init__(self, fac, n_free, n_border):
        self.fac, self.n, self.nb = fac, n_free, n_border

    def solve(self, v):
        v = np.asarray(v, dtype=np.float64)
        pad = np.zeros((self.n + self.nb,) + v.shape[1:])
        pad[:self.n] = v
        return np.asarray(self.fac.solve(pad))[:self.n]


class RefinedSystem:
    """`RefinedFactor` of the bordered matrix, with the interface of the plain one: grid vectors in and out.

    The right side is padded with zeros. The border rows of a solution are `q = -D^-1 C^T u`, so the residual of the border row is
    zero by construction and the measured residual is the one of the grid rows.
    """

    def __init__(self, rf, system: SupportSystem):
        self.rf, self.s = rf, system
        self._lu = sla.lu_factor(system.D.toarray()) if system.n_border else None

    def _pad(self, b):
        b = np.asarray(b, dtype=np.float64)
        out = np.zeros((self.s.n_free + self.s.n_border,) + b.shape[1:])
        out[:self.s.n_free] = b
        return out

    def _full(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.s.n_border:
            return x
        x2 = x.reshape(self.s.n_free, -1)
        q = -sla.lu_solve(self._lu, self.s.C.T @ x2)
        return np.vstack([x2, q]).reshape((self.s.n_free + self.s.n_border,) + x.shape[1:])

    def solve(self, b):
        return self.rf.solve(self._pad(b))[:self.s.n_free]

    def residual_norms(self, b, x):
        return self.rf.residual_norms(self._pad(b), self._full(x))

    def backward_error(self, b, x):
        return self.rf.backward_error(self._pad(b), self._full(x))

    def free(self):
        self.rf.free()


class SupportSet:
    """The joints and springs of a level. `Ks` is the sum of the spring matrices and the bond matrices `Kb`, added to `K`."""

    def __init__(self, joints: list, springs: list, n_dof: int, facts: dict):
        self.joints, self.springs, self.n_dof, self.facts = joints, springs, n_dof, facts
        mats = [j.Kb for j in joints] + [s.Kb for s in springs]
        self.Ks = sum(mats[1:], mats[0]).tocsr()

    @property
    def n_border(self) -> int:
        return 6 * len(self.joints)

    def system(self, Kff, free) -> SupportSystem:
        """The bordered system on the free dofs. `Kff` is `(K + Ks)[free][:, free]`."""
        free = np.asarray(free, dtype=np.int64)
        if not self.joints:
            return SupportSystem(Kff, sps.csc_matrix((len(free), 0)), sps.csr_matrix((0, 0)), free)
        C = sps.hstack([j.C[free] for j in self.joints], format="csc")
        D = sps.block_diag([sps.csr_matrix(j.D) for j in self.joints], format="csr")
        return SupportSystem(Kff, C, D, free)

    # ---- after the solve ----
    def terms(self, u) -> dict:
        """The bodies and the nodal forces for the displacement `u` (all dofs, zero on the fixed ones).

        `q` of each joint is `-D^-1 C^T u`. The nodal force of a bond is `g = Kb u + C q` (the bond acts on the part with `-g`); the nodal
        force of a spring is `Kb u`. `Cq` is the sum of `C q` and `force` the force on the part from all supports `-(Ks u + C q)`.
        """
        out = dict(joints={}, springs={}, Cq=np.zeros(self.n_dof))
        for j in self.joints:
            q = -np.linalg.solve(j.D, j.C.T @ u)
            cq = j.C @ q
            out["joints"][j.name] = (q, j.Kb @ u + cq)
            out["Cq"] += cq
        for s in self.springs:
            out["springs"][s.name] = s.Kb @ u
        out["force"] = -(self.Ks @ u + out["Cq"])
        return out

    def case_record(self, model, u, f, terms: dict | None = None) -> dict:
        """The record of one load case with the load vector `f`: the force and the moment of each bond and of each spring on the part,
        the body motion, the beam, and the share of the strain energy that the bond holds."""
        terms = self.terms(u) if terms is None else terms
        half_c = 0.5 * float(f @ u)
        X = model.node_xyz.T
        cjs, cs = {}, {}
        for j in self.joints:
            q, g = terms["joints"][j.name]
            node = np.flatnonzero(g.reshape(-1, 3).any(axis=1))
            g3 = g.reshape(-1, 3)[node]
            force = -g3.sum(axis=0)                                            # on the part
            moment = -np.cross(X[node] - j.ref, g3).sum(axis=0)
            beam = joint_wrenches(SimpleNamespace(B=j.B, ref=j.ref, far=j.far), q)
            want = np.concatenate([beam["part"]["force_n"], beam["part"]["moment_nmm"]])
            got = np.concatenate([force, moment])
            e_bond = 0.5 * float(u @ g - q @ j.B @ q)
            cjs[j.name] = dict(
                bond_force_N=[float(v) for v in force], bond_moment_Nmm=[float(v) for v in moment],
                about_mm=[float(v) for v in j.ref],
                bond_wrench_vs_beam_relative=float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300)),
                beam=beam, translation_mm=beam["translation_mm"], rotation_rad=beam["rotation_rad"],
                bond_energy_share=float(e_bond / half_c) if half_c else None)
        for s in self.springs:
            g3 = terms["springs"][s.name].reshape(-1, 3)
            node = np.flatnonzero(g3.any(axis=1))
            cs[s.name] = dict(force_N=[float(v) for v in -g3[node].sum(axis=0)],
                              moment_Nmm=[float(v) for v in -np.cross(X[node], g3[node]).sum(axis=0)], about_mm=[0.0, 0.0, 0.0])
        return dict(joints=cjs, springs=cs)

    def buckling_note(self) -> dict | None:
        """What the load factors of the tet buckling check are for, when the set has joints (None without joints)."""
        if not self.joints:
            return None
        return dict(joints=len(self.joints), stiffness="S = K + Kb - C D^-1 C^T (the Schur complement of the bordered matrix)",
                    beams_have_geometric_stiffness=False, fact=BUCKLING_LINE, option=BUCKLING_BEAMS_OPTION,
                    meaning="the load factors are for buckling of the part, not of the beams")


def build_supports(model, cfg, sel: dict, h: float, vox_area: dict | None = None, step: float | None = None,
                   vox_y: dict | None = None) -> SupportSet:
    """The joints and springs of the YAML on the P2 model. `sel` is `{region name: Bonded or Selection}` (the facets of each region).

    Raise `UserError` for a joint with a degenerate beam axis, or for a region with no facet to bond. `h` is the voxel size of the run
    (the penalty is `c E / h`). `vox_area` is `voxel_areas(run.json)`; `step` is the distance of the voxel rule (`bonded_facets`).
    `vox_y` is `voxel_frames(run.json)`: a joint with no `orient` takes the section y axis of the voxel run, because the default rule ("the
    global axis least parallel to the beam") ties when the beam lies along an axis, and the reference point of the tet facets, which is not the
    reference point of the voxel faces, then decides which axis it is. The section is not round in general, so the axis decides the stiffness.
    """
    mat = cfg.material
    vox_area = vox_area or {}
    vox_y = vox_y or {}
    n_dof = model.N
    joints, springs, jfacts, sfacts = [], [], [], []
    for spec in cfg.joints:
        s = sel[spec.region]
        where = f"joint on load region '{spec.region}'"
        facets = s.facets
        if not len(facets):
            raise _no_facet("joint", spec.region, s)
        P = model.p.T[facets].mean(axis=1)
        ref = np.array(spec.origin, dtype=np.float64) if spec.origin is not None \
            else (P * s.facet_area[:, None]).sum(axis=0) / s.facet_area.sum()
        far = np.array(spec.to, dtype=np.float64)
        bad = joint_axis_problem(ref, far, spec.orient)
        if bad:
            raise UserError(f"{where}: {bad}. Options: move 'to' or 'origin', or give another 'orient'")
        axis = far - ref
        length = float(np.linalg.norm(axis))
        orient, y_from = spec.orient, None
        if orient is None and spec.region in vox_y and joint_axis_problem(ref, far, vox_y[spec.region]) is None:
            orient, y_from = vox_y[spec.region], "the section y axis of the voxel run (run.json): the default rule ties when the beam lies along an axis"
        Q = beam_frame(axis, orient)
        E = mat.E if spec.E is None else spec.E
        nu = mat.nu if spec.nu is None else spec.nu
        props = section_properties(spec.section)
        k_a = spec.bond_stiffness_per_area if spec.bond_stiffness_per_area is not None else BOND_C * mat.E / h
        B = beam_stiffness(E, nu, props, length, Q)
        Kb, C, D_bond = bond_blocks_p2(model, facets, ref, k_a)
        share = float(sla.eigh(B, B + D_bond, eigvals_only=True).max())
        bf = beam_facts(props, length, Q, B, D_bond, mat.E, h, spec.orient)      # the function of the voxel model: rho, L/d, the frame, the near-free ratios
        f = dict(region=spec.region, facets=int(len(facets)), area_mm2=float(s.facet_area.sum()), nodes=int(np.unique(
            np.concatenate([model.edge_node_ids(facets[:, 0], facets[:, 1]), model.edge_node_ids(facets[:, 1], facets[:, 2]),
                            model.edge_node_ids(facets[:, 0], facets[:, 2]), facets.ravel()])).size),
                 reference_mm=[float(v) for v in ref],
                 reference_from="the joint entry (origin)" if spec.origin is not None else "the area centroid of the bonded facets",
                 to_mm=[float(v) for v in far], length_mm=length, axis=[float(v) for v in Q[:, 0]],
                 section=spec.section.as_dict(), section_area_mm2=props["A"], iy_mm4=props["Iy"], iz_mm4=props["Iz"], j_mm4=props["J"],
                 depth_mm=bf["depth_mm"], length_over_depth=bf["length_over_depth"],
                 frame=dict(bf["frame"], y_from=y_from or bf["frame"]["y_from"]),
                 beam_E_mpa=float(E), beam_nu=float(nu),
                 bond_stiffness_per_area_n_per_mm3=float(k_a),
                 bond_stiffness_from=("the joint entry (bond_stiffness_per_area)" if spec.bond_stiffness_per_area is not None
                                      else "c E / h with c = %g, h the voxel size of the run" % BOND_C),
                 c_equivalent=float(k_a * h / mat.E),
                 bond_diagonal=[float(v) for v in np.diag(D_bond)], beam_diagonal=[float(v) for v in np.diag(B)],
                 rho=bf["rho"], bond_share_max=share, beam_min_translation_over_Eh=bf["beam_min_translation_over_Eh"],
                 beam_min_rotation_over_Eh3=bf["beam_min_rotation_over_Eh3"], **_area_facts(s, vox_area.get(spec.region)))
        joints.append(PJoint(spec.region, ref, far, B, D_bond, Kb, C, f))
        jfacts.append(f)
    for spec in cfg.springs:
        s = sel[spec.region]
        if not len(s.facets):
            raise _no_facet("spring", spec.region, s)
        area = float(s.facet_area.sum())
        k_pa = np.array(spec.per_area, dtype=np.float64) if spec.per_area is not None else np.array(spec.stiffness, dtype=np.float64) / area
        Kb = spring_matrix_p2(model, s.facets, k_pa)
        f = dict(region=spec.region, facets=int(len(s.facets)), area_mm2=area,
                 stiffness_n_per_mm=[float(v) for v in k_pa * area], stiffness_per_area_n_per_mm3=[float(v) for v in k_pa],
                 stiffness_from="the spring entry (stiffness: the total is kept, the area is the one of the tet facets)"
                 if spec.stiffness is not None else "the spring entry (stiffness_per_area: used as it is)",
                 **_area_facts(s, vox_area.get(spec.region)))
        springs.append(PSpring(spec.region, Kb, f))
        sfacts.append(f)
    facts = dict(kind=KIND, joints=jfacts, springs=sfacts, bordered_dofs=6 * len(joints), facets_note=FACETS_NOTE,
                 near_singular_lines=near_singular_lines({"joints": jfacts}, mat.E, h),      # the lines of `topopt check` for a nearly free beam
                 bond=dict(c=BOND_C, source=BOND_C_SOURCE) if joints else None,
                 voxel_size_mm=float(h), neighbour_step_mm=None if step is None else float(step))
    return SupportSet(joints, springs, n_dof, facts)
