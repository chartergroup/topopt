"""Elastic supports: springs on the exposed faces of load regions (CONTRACTS.md section 35).

A spring entry (`Config.springs`) names a load region. The faces are the pressure faces of
`loads.exposed_faces`: a face of a voxel of the region whose neighbour is EMPTY or KEEP_OUT, or lies outside the grid.
Each face gets the consistent Q1 face matrix `k_face * FACE_UNIT` on each axis, a sparse term `Kb` that is added to
`K`. A spring adds no dof and does not depend on the density, so the sensitivities do not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import UserError
from .loads import exposed_faces, node_positions

# Q1 face matrix for a face of unit area: nodes in the order 00, 01, 10, 11 over the two axes of the face. The
# entries sum to 1; each row sums to 1/4 (the lumped rule). A lumped rule over-stiffens the rotation of a support
# by 1 + 2/n^2 (n voxels across), so the matrix is consistent.
_M1 = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
FACE_UNIT = np.kron(_M1, _M1)
NEAR_SINGULAR = 1e-6     # a spring stiffness below this part of `E h` is a fact in `check` (a nearly free body)
AUDIT = 1e-12            # relative size of the audit: the sum of the face matrices against the stiffness


def face_nodes(faces, shape) -> np.ndarray:
    """Node numbers, array (F, 4), of the faces `[(axis, sign, bool voxel mask)]` (order of `FACE_UNIT`)."""
    ny1, nz1 = shape[1] + 1, shape[2] + 1
    out = []
    for axis, sign, m in faces:
        ijk = np.argwhere(m)
        if not len(ijk):
            continue
        a, b = [d for d in range(3) if d != axis]
        cols = []
        for da in (0, 1):
            for db in (0, 1):
                c = ijk.copy()
                c[:, axis] += 1 if sign > 0 else 0
                c[:, a] += da
                c[:, b] += db
                cols.append((c[:, 0] * ny1 + c[:, 1]) * nz1 + c[:, 2])
        out.append(np.stack(cols, axis=1))
    return np.concatenate(out) if out else np.zeros((0, 4), dtype=np.int64)


def face_triplets(nodes: np.ndarray, k_face, unit=FACE_UNIT) -> tuple:
    """`(rows, cols, vals)` of the face matrices on grid dofs. `k_face` (3,) is the stiffness of one face, N/mm.

    `unit` is the matrix for a face of unit area (a test passes `np.eye(4) / 4` for the lumped rule).
    """
    rows, cols, vals = [], [], []
    shape = (len(nodes), 4, 4)
    for d in range(3):
        if k_face[d] > 0.0:
            rows.append(np.broadcast_to(3 * nodes[:, :, None] + d, shape).ravel())
            cols.append(np.broadcast_to(3 * nodes[:, None, :] + d, shape).ravel())
            vals.append(np.broadcast_to(k_face[d] * unit, shape).ravel())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def sum_duplicates(rows, cols, vals, n: int) -> tuple:
    """Add the entries that share a (row, col) place. The result is sorted by row, then col."""
    u, inv = np.unique(rows * n + cols, return_inverse=True)
    return u // n, u % n, np.bincount(inv, weights=vals, minlength=u.size)


@dataclass
class SpringRegion:
    name: str
    faces: int
    area: float                 # mm2
    k_total: np.ndarray         # (3,) N/mm
    rows: np.ndarray            # Kb of this region on grid dofs (all rows, also fixed ones)
    cols: np.ndarray
    vals: np.ndarray
    nodes: int


@dataclass
class SpringSet:
    """The springs of a problem: `regions`, the dofs that carry a spring, and the JSON `facts`."""

    regions: list
    n_dof: int                  # dofs of the whole grid
    dofs: np.ndarray            # sorted unique grid dofs that have a spring (they count as supports in the rigid-mode test)
    region_ids: list            # index of each spring region in `regions.load`
    facts: dict = field(default_factory=dict)

    def triplets(self) -> tuple:
        """All regions together: `(rows, cols, vals)`, sorted, duplicates added."""
        return sum_duplicates(np.concatenate([r.rows for r in self.regions]),
                              np.concatenate([r.cols for r in self.regions]),
                              np.concatenate([r.vals for r in self.regions]), self.n_dof)

    def matrix(self, assembler, free_dofs) -> "SpringMatrix":
        """`Kb` on the pattern of the reduced stiffness matrix of `assembler` (its index arrays; no assembly)."""
        return SpringMatrix(*self.triplets(), assembler._indptr, assembler._indices, free_dofs)


def support_mask(problem) -> np.ndarray:
    """`(nx, ny, nz)` bool: the voxels that hold the part. The FIXED voxels, and the voxels of the spring and joint regions.

    One rule for the readers that ask "is this piece held?" (the interface check, the load path of the stress report,
    the loose pieces of the buckling check, `topopt check`). A spring region is LOAD, so `labels == FIXED` alone misses it.
    """
    from .voxelize import FIXED, LOAD
    mask = problem.labels == FIXED
    ids = ([] if problem.springs is None else problem.springs.region_ids) \
        + ([] if problem.joints is None else problem.joints.region_ids)       # a joint region holds the part too (section 38)
    if ids:
        mask = mask | (np.isin(problem.load_region_ids, ids) & (problem.labels == LOAD))
    return mask


def near_singular_lines(facts: dict | None, E: float, h: float) -> list:
    """One line for each spring region whose smallest stiffness is below `NEAR_SINGULAR` of `E h` (`check` and `run`)."""
    return [f"spring on '{r['region']}': its smallest stiffness is {r['relative_to_Eh']:.1e} of E h "
            f"(E {E:g} MPa x voxel {h:g} mm), below {NEAR_SINGULAR:g}: the part is nearly free on that axis, so the "
            "stiffness matrix is nearly singular and the displacement is large. "
            "Options: raise the stiffness, or add a fixed region"
            for r in (facts or {}).get("regions", []) if r["relative_to_Eh"] < NEAR_SINGULAR]


def build_springs(config, labels, load_ids, h: float) -> SpringSet:
    """The face matrices of every `config.springs` entry. Raise `UserError` for a region with no exposed face."""
    index = {lr.name: i for i, lr in enumerate(config.regions.load)}
    n_dof = 3 * int(np.prod(np.array(labels.shape) + 1))
    regions, ids, facts = [], [], []
    for s in config.springs:
        faces = exposed_faces(labels, load_ids == index[s.region])
        nodes = face_nodes(faces, labels.shape)
        if len(nodes) == 0:
            raise UserError(
                f"spring on load region '{s.region}': no face of the region looks toward empty space or a keep-out "
                "region (faces toward design, fixed or other load voxels get no spring). Options: shape the region "
                "so that the spring face is free (a pad has its free face away from the design), or use another region")
        area = len(nodes) * h * h
        k_total = np.array(s.stiffness) if s.stiffness is not None else np.array(s.per_area) * area
        r, c, v = sum_duplicates(*face_triplets(nodes, k_total / len(nodes)), n_dof)
        got = np.array([v[r % 3 == d].sum() for d in range(3)])
        audit = float(np.abs(got - k_total).max() / k_total.max())
        if audit > AUDIT:
            raise UserError(f"spring on load region '{s.region}': the face matrices sum to {got}, not {k_total}")
        regions.append(SpringRegion(s.region, len(nodes), area, k_total, r, c, v, int(np.unique(nodes).size)))
        ids.append(index[s.region])
        kmin = float(k_total[k_total > 0].min())
        by_dir = {("-" if sg < 0 else "+") + "xyz"[ax]: int(m.sum()) for ax, sg, m in faces if m.any()}
        facts.append({"region": s.region, "faces": len(nodes), "faces_by_direction": by_dir,
                      "area_mm2": area, "nodes": regions[-1].nodes,
                      "stiffness_n_per_mm": [float(x) for x in k_total],
                      "stiffness_per_area_n_per_mm3": [float(x) for x in k_total / area],
                      "relative_to_Eh": kmin / (config.material.E * h), "audit_rel": audit})
    dofs = np.unique(np.concatenate([r.rows for r in regions]))
    return SpringSet(regions, n_dof, dofs, ids, {"regions": facts, "spring_dofs": int(dofs.size)})


def _reduced(free_dofs, dofs):
    """Place of each grid dof in the reduced system, -1 for a fixed dof."""
    i = np.minimum(np.searchsorted(free_dofs, dofs), free_dofs.size - 1)
    return np.where(free_dofs[i] == dofs, i, -1)


class SpringMatrix:
    """`Kb` as positions in `K.data` of the reduced stiffness matrix. `add_to(K)` adds it in place.

    The pattern of `K` is the same for every call of the assembler (`indptr`, `indices` of the assembler), so the
    positions are found once and the solvers keep their caches (the index arrays are not touched). A face matrix entry
    on a fixed dof is dropped.
    """

    def __init__(self, rows, cols, vals, indptr, indices, free_dofs):
        fd = np.asarray(free_dofs, dtype=np.int64)
        ri, ci = _reduced(fd, rows), _reduced(fd, cols)
        keep = (ri >= 0) & (ci >= 0)
        n = len(indptr) - 1
        have = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr)) * n + indices          # column-major keys of K
        order = None if np.all(have[1:] > have[:-1]) else np.argsort(have)
        want = ci[keep] * n + ri[keep]
        at = np.minimum(np.searchsorted(have if order is None else have[order], want), have.size - 1)
        pos = at if order is None else order[at]
        if np.any(have[pos] != want):
            raise ValueError("a spring entry lies outside the pattern of the stiffness matrix")
        self.pos, self.vals, self.nnz = pos, vals[keep], len(indices)

    def add_to(self, K) -> None:
        if K.data.size != self.nnz:
            raise ValueError("the stiffness matrix has another pattern than the spring matrix")
        K.data[self.pos] += self.vals


# --- reactions -------------------------------------------------------------------------------


def _wrench(vec, shape, origin, h) -> tuple:
    """Force, moment about `origin` and the sum of the sizes of the nodal moments of a dof vector."""
    f3 = vec.reshape(-1, 3)
    nz = np.flatnonzero(f3.any(axis=1))
    if not nz.size:
        return np.zeros(3), np.zeros(3), 0.0
    m = np.cross(node_positions(nz, shape, origin, h), f3[nz])
    return f3[nz].sum(axis=0), m.sum(axis=0), float(np.linalg.norm(m, axis=1).sum())


def reaction_facts(problem, edof, Ee, ke, free_dofs, U) -> dict:
    """Reactions of the last pass, for each load case (section 35).

    `Ee` is the element stiffness of the solve that gave `U` (`Model.element_E` of its density). Clamp:
    `(K u)_i + (Kb u)_i - f_i` on the fixed dofs `i` (element forces at the fixed dofs, less the load that sits on
    them). Spring region: `-Kb_r u`, the force of the spring on the part. Joint (section 38): the force of the bond on
    the part, `-(Kb u + C q)` at the nodes, with `q = -D^-1 C^T u`; `joints` also gives the wrench on the part
    (`-B q` about the reference point), the ground reaction at the far point, the body motion and the share of the strain
    energy that the bond holds. With the applied load `f` they sum to the
    net force and moment of the residual of the solve on the free dofs. The error is the largest component of that
    sum over the sum of the sizes of the terms (force), and the same for the moment about the grid origin. For a direct
    solver it is round-off. For AMG it can be far below the residual of the solve, so it does not bound the error of
    the displacement; `residual_rel` (the norm of the residual over the norm of the load on the free dofs) does.
    """
    sp, jt, fixed, shape, h, origin = problem.springs, problem.joints, problem.fixed_dofs, problem.shape, problem.h, problem.origin
    regions = [] if sp is None else sp.regions
    n_dof = 3 * int(np.prod(np.array(shape) + 1))
    u = np.zeros(n_dof)
    cases = []
    for c, name in enumerate(problem.case_names):
        u[free_dofs] = U[:, c]
        terms = {"applied": problem.forces[c].copy()}
        kbu = np.zeros(n_dof)
        for r in regions:
            part = np.bincount(r.rows, weights=r.vals * u[r.cols], minlength=n_dof)
            kbu += part
            terms[f"spring:{r.name}"] = -part
        jfacts = {}
        if jt is not None:       # joints (section 38): the body motion q from the grid solution, the nodal force g of the bond
            from .joints import joint_wrenches
            for j, (q, g) in zip(jt.joints, jt.border_terms(u)):
                kbu += g
                terms[f"joint:{j.name}"] = -g
                jfacts[j.name] = (joint_wrenches(j, q), 0.5 * float(u @ g - q @ j.B @ q))      # wrenches, energy of the bond
        ku = np.bincount(edof.ravel(), weights=(Ee[:, None] * (u[edof] @ ke)).ravel(), minlength=n_dof)
        resid = (ku + kbu - problem.forces[c])[free_dofs]
        if fixed.size:
            clamp = np.zeros(n_dof)
            clamp[fixed] = ku[fixed] + kbu[fixed] - problem.forces[c][fixed]
            terms["clamp"] = clamp
        w = {k: _wrench(v, shape, origin, h) for k, v in terms.items()}
        res_f = sum(x[0] for x in w.values())
        res_m = sum(x[1] for x in w.values())
        f_scale = sum(float(np.abs(x[0]).max()) for x in w.values())
        m_scale = sum(x[2] for x in w.values())
        show = lambda x: {"force_n": [float(a) for a in x[0]], "moment_nmm": [float(a) for a in x[1]]}
        case = {"name": name, "applied": show(w["applied"])}
        if "clamp" in w:
            case["clamp"] = show(w["clamp"])
        case["springs"] = {r.name: show(w[f"spring:{r.name}"]) for r in regions}
        f_free = float(np.linalg.norm(problem.forces[c][free_dofs]))
        if jt is not None:
            half_c = 0.5 * float(problem.forces[c][free_dofs] @ U[:, c])      # half the compliance: the strain energy
            case["joints"] = {k: {**wr, "bond_energy_share": e_bond / half_c if half_c else float("nan"),
                                  "nodal": show(w[f"joint:{k}"])} for k, (wr, e_bond) in jfacts.items()}
        case.update(sum_force_n=[float(a) for a in res_f], sum_moment_nmm=[float(a) for a in res_m],
                    residual_rel=float(np.linalg.norm(resid) / max(f_free, 1e-300)),
                    force_error=float(np.abs(res_f).max() / max(f_scale, 1e-300)),
                    moment_error=float(np.abs(res_m).max() / max(m_scale, 1e-300)))
        cases.append(case)
    return {"moment_about_mm": [float(v) for v in origin], "cases": cases}


REACTION_WARN = 1e-6      # a balance error above this gets the prefix `warning:` in the run log


def reaction_lines(rf: dict) -> list:
    """Text lines for the run log from `reaction_facts`. A balance error above `REACTION_WARN` gets the prefix `warning:`."""
    fmt = lambda x: "[" + ", ".join(f"{a:.4g}" for a in x["force_n"]) + "] N"
    vec = lambda x: "[" + ", ".join(f"{a:.4g}" for a in x) + "]"
    out = []
    for c in rf["cases"]:
        parts = [f"load {fmt(c['applied'])}"] + ([f"clamp {fmt(c['clamp'])}"] if "clamp" in c else []) \
            + [f"spring '{k}' {fmt(v)}" for k, v in c["springs"].items()] \
            + [f"joint '{k}' {fmt(v['nodal'])}" for k, v in c.get("joints", {}).items()]
        bad = max(c["force_error"], c["moment_error"]) > REACTION_WARN
        out.append(("warning: " if bad else "") + f"reactions:      {c['name']}: " + ", ".join(parts)
                   + f"; the sum is zero to {c['force_error']:.1e} (force) and {c['moment_error']:.1e} (moment), relative"
                   + f"; residual of the solve {c['residual_rel']:.1e} of the load"
                   + ("; the solve is not accurate, or the reactions use another stiffness than the solve" if bad else ""))
        for k, v in c.get("joints", {}).items():
            p, g = v["part"], v["ground"]
            out.append(f"reactions:      {c['name']}: joint '{k}': on the part {vec(p['force_n'])} N and {vec(p['moment_nmm'])} N mm "
                       f"about {vec(p['about_mm'])} mm; ground reaction {vec(g['force_n'])} N and {vec(g['moment_nmm'])} N mm "
                       f"about the far point {vec(g['about_mm'])} mm; the bond holds {100 * v['bond_energy_share']:.3g} percent "
                       "of the strain energy")
    return out
