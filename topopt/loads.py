"""Nodal loads and fixed dofs (CONTRACTS.md section 3).

Node `(i, j, k)` has the flat index `(i*(ny+1) + j)*(nz+1) + k`.
Its dofs are `3n`, `3n+1`, `3n+2`.

Load and support types (CONTRACTS.md section 26.2): a moment, a remote force (a distributing coupling,
RBE3-like), a pressure on the exposed faces of a load region, per-axis supports, a check that the
supports block the six rigid-body modes, and the lost force and moment on fixed dofs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ALL_AXES, Config, UserError

AUDIT = 1e-9           # relative size of the force and moment audits
MOMENT_RCOND = 1e-6    # axes of the node cloud with an eigenvalue ratio below this do not carry a moment
RIGID_RCOND = 1e-9     # singular value ratio of the support test below which a rigid-body mode is free
_MODE_NAMES = ("translation along x", "translation along y", "translation along z",
               "rotation about x", "rotation about y", "rotation about z")


def node_mask(voxel_mask: np.ndarray) -> np.ndarray:
    """Bool array (nx+1, ny+1, nz+1): True for each node of a True voxel."""
    nx, ny, nz = voxel_mask.shape
    nm = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                nm[di:di + nx, dj:dj + ny, dk:dk + nz] |= voxel_mask
    return nm


def node_ids(voxel_mask: np.ndarray) -> np.ndarray:
    """Sorted unique flat node numbers of the nodes of the True voxels."""
    return np.flatnonzero(node_mask(voxel_mask).ravel()).astype(np.int64)


def fixed_dofs_from(fixed_voxels: np.ndarray) -> np.ndarray:
    """Sorted unique dofs (x, y, z) of all nodes of the fixed voxels."""
    n = node_ids(fixed_voxels)
    return (3 * n[:, None] + np.arange(3, dtype=np.int64)).ravel()


def build_forces(config: Config, load_region_ids: np.ndarray, labels: np.ndarray | None = None,
                 origin=None, h: float | None = None, facts: dict | None = None) -> np.ndarray:
    """Force array `(n_cases, 3*n_nodes)`. Equal share on each node of a region.

    A plain force gets the equal share. A moment or a remote force (`at`) gets the distributing
    coupling, and a pressure gets `p h^2 / 4` on each node of each exposed face (section 26.2). These
    need `labels`, `origin` and `h`. With a dict `facts`, `facts["entries"]` gets one list of `LoadEntry`
    for each case (only when the config has such a load).
    """
    nx, ny, nz = load_region_ids.shape
    n_dof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    region_index = {lr.name: i for i, lr in enumerate(config.regions.load)}
    nodes = {}
    forces = np.zeros((len(config.load_cases), n_dof), dtype=np.float64)
    extended = any(f.moment is not None or f.at is not None for c in config.load_cases for f in c.forces) \
        or any(c.pressures for c in config.load_cases)
    if extended and (labels is None or origin is None or h is None):
        raise ValueError("a moment, a remote force or a pressure needs labels, origin and h")
    all_entries = []
    for c, case in enumerate(config.load_cases):
        expected = np.zeros(3)
        entries = []
        for f in case.forces:
            idx = region_index[f.region]
            if idx not in nodes:
                nodes[idx] = node_ids(load_region_ids == idx)
            n = nodes[idx]
            if n.size == 0:
                raise UserError(f"load region '{f.region}' has no node")
            if f.moment is None and f.at is None:
                share = np.asarray(f.force, dtype=np.float64) / n.size
                for d in range(3):
                    forces[c, 3 * n + d] += share[d]
                if extended:
                    entries.append(_plain_entry(f, n, load_region_ids.shape, origin, h))
            else:
                e = _wrench_entry(f, n, load_region_ids.shape, origin, h)
                forces[c, (3 * e.nodes[:, None] + np.arange(3)).ravel()] += e.f.ravel()
                entries.append(e)
            expected += np.asarray(f.force, dtype=np.float64)
        for p in case.pressures:
            e = _pressure_entry(p, load_region_ids == region_index[p.region], labels, origin, h)
            forces[c, (3 * e.nodes[:, None] + np.arange(3)).ravel()] += e.f.ravel()
            entries.append(e)
            expected += e.force_n
        got = forces[c].reshape(-1, 3).sum(axis=0)
        scale = max(np.abs(expected).max(), 1e-300)
        # The sum can be 0 (a moment, a closed surface, two equal and opposite forces on two regions: a squeeze): the size of
        # the nodal forces is the scale. Without it the audit compared a rounding of 1e-15 with 1e-300 and refused a real load.
        scale = max(scale, float(np.abs(forces[c]).sum()))
        if np.abs(got - expected).max() > AUDIT * scale:
            raise UserError(f"load case '{case.name}': force sum {got} differs from {expected}")
        all_entries.append(entries)
    if facts is not None and extended:
        facts["entries"] = all_entries
    return forces


# --- load and support types (CONTRACTS.md section 26.2) -------------------------------------


@dataclass
class LoadEntry:
    """The nodal forces of one force entry or one pressure of a load case, with its facts."""

    kind: str                  # force, remote_force, moment, force_moment, remote_force_moment, pressure
    region: str
    nodes: np.ndarray          # node numbers
    pos: np.ndarray            # (m, 3) node positions, mm
    f: np.ndarray              # (m, 3) nodal forces, N
    about: np.ndarray          # mm: node centroid of the region. The moment is about this point
    force_n: np.ndarray        # (3,) resultant force, N
    moment_nmm: np.ndarray     # (3,) moment about `about`, N mm (the applied one: `moment` + (at - about) x force)
    info: dict = field(default_factory=dict)   # JSON facts: at_mm, pressure_mpa, faces, area_mm2, audit


def node_positions(ids: np.ndarray, shape, origin, h: float) -> np.ndarray:
    """Positions (mm), array (len(ids), 3), of the nodes with the flat numbers `ids`."""
    nx, ny, nz = (int(s) for s in shape)
    i = ids // ((ny + 1) * (nz + 1))
    j = (ids // (nz + 1)) % (ny + 1)
    k = ids % (nz + 1)
    return np.asarray(origin, dtype=np.float64) + h * np.stack([i, j, k], axis=1).astype(np.float64)


def entry_kind(f) -> str:
    """The `kind` of a force entry (CONTRACTS.md section 26.2)."""
    if f.moment is None:
        return "remote_force" if f.at is not None else "force"
    if not any(v != 0.0 for v in f.force):
        return "moment"
    return "remote_force_moment" if f.at is not None else "force_moment"


def moment_spread(r: np.ndarray, moment: np.ndarray, where: str = "the region") -> np.ndarray:
    """Nodal forces `w x r_n` with zero net force and the moment `moment` about the centroid.

    `r` holds the node positions from the centroid, array (n, 3). `w = I^-1 M` with
    `I = sum(|r|^2 E - r r^T)` (the same matrix as the inertia tensor of unit point masses).
    This is the least-squares spread of the moment over the nodes (the distributing, RBE3-like coupling).
    Axes where the nodes are too close to a line (an eigenvalue of `I` below `MOMENT_RCOND` of the largest)
    carry no moment. A moment with a part about such an axis raises a `UserError`.
    """
    inertia = float((r * r).sum()) * np.eye(3) - r.T @ r
    lam, vec = np.linalg.eigh(inertia)
    top = float(lam[-1])
    carried = lam > MOMENT_RCOND * top if top > 0.0 else np.zeros(3, dtype=bool)
    m_in = vec.T @ moment
    m_size = float(np.linalg.norm(moment))
    bad = float(np.linalg.norm(m_in[~carried]))
    if bad > AUDIT * m_size:
        axis = vec[:, ~carried][:, int(np.argmax(np.abs(m_in[~carried])))]
        ratio = float(lam[0] / top) if top > 0.0 else 0.0
        raise UserError(
            f"{where}: its {r.shape[0]} nodes cannot carry the moment about the axis "
            f"[{axis[0]:.3g}, {axis[1]:.3g}, {axis[2]:.3g}]. The moment matrix of the nodes has a smallest-to-largest "
            f"eigenvalue ratio of {ratio:.1e} (the limit is {MOMENT_RCOND:g}): the nodes lie on a line along this axis. "
            "Options: make the region wider across that axis (one voxel is two nodes across), "
            "apply the moment as a force pair on two load regions, or use a smaller voxel_size")
    omega = vec[:, carried] @ (m_in[carried] / lam[carried])
    return np.cross(omega, r)


def _rel(err, ref, floor) -> float:
    """Largest component of `err` over the larger of the largest component of `ref` and `floor`."""
    return float(np.abs(err).max() / max(float(np.abs(ref).max()), floor, 1e-300))


def _plain_entry(f, n, shape, origin, h) -> LoadEntry:
    """A force entry with no moment and no `at`: the equal share (its facts only, the forces are filled above)."""
    pos = node_positions(n, shape, origin, h)
    force = np.asarray(f.force, dtype=np.float64)
    return LoadEntry("force", f.region, n, pos, np.tile(force / n.size, (n.size, 1)), pos.mean(axis=0),
                     force, np.zeros(3), {"nodes": int(n.size)})


def _wrench_entry(f, n, shape, origin, h) -> LoadEntry:
    """Nodal forces of a force with a moment and/or an `at` point: the distributing coupling."""
    pos = node_positions(n, shape, origin, h)
    about = pos.mean(axis=0)
    r = pos - about
    force = np.asarray(f.force, dtype=np.float64)
    moment = np.zeros(3) if f.moment is None else np.asarray(f.moment, dtype=np.float64)
    if f.at is not None:
        moment = moment + np.cross(np.asarray(f.at, dtype=np.float64) - about, force)
    nodal = np.tile(force / n.size, (n.size, 1))
    if np.any(moment):
        nodal = nodal + moment_spread(r, moment, f"load region '{f.region}'")
    # Audit: the nodal forces give back the force and the moment (the floor is the size of the terms: round-off).
    err_f = _rel(nodal.sum(axis=0) - force, force, float(np.abs(nodal).sum()))
    err_m = _rel(np.cross(r, nodal).sum(axis=0) - moment, moment, float(np.abs(np.cross(r, nodal)).sum()))
    if max(err_f, err_m) > AUDIT:
        raise UserError(f"load region '{f.region}': the nodal forces differ from the force and moment "
                        f"(force {err_f:.2g}, moment {err_m:.2g} relative)")
    info = {"nodes": int(n.size), "at_mm": None if f.at is None else [float(v) for v in f.at],
            "audit_force_rel": err_f, "audit_moment_rel": err_m}
    return LoadEntry(entry_kind(f), f.region, n, pos, nodal, about, force, moment, info)


def exposed_faces(labels: np.ndarray, mask: np.ndarray):
    """The pressure faces of the voxels in `mask`: `[(axis, sign, bool mask)]`, six items.

    A voxel face is exposed when its neighbour is EMPTY or KEEP_OUT, or lies outside the grid. A face
    toward a design, fixed or other load voxel is not exposed: it gets no pressure.
    """
    from .voxelize import EMPTY, KEEP_OUT
    shape = labels.shape
    pad = np.pad(labels, 1, constant_values=EMPTY)
    out = []
    for axis in range(3):
        for sign in (-1, 1):
            sl = [slice(1, 1 + s) for s in shape]
            sl[axis] = slice(1 + sign, 1 + sign + shape[axis])
            nb = pad[tuple(sl)]
            out.append((axis, sign, mask & ((nb == EMPTY) | (nb == KEEP_OUT))))
    return out


def _pressure_entry(p, mask, labels, origin, h) -> LoadEntry:
    """Nodal forces of a pressure: `p h^2 / 4` on each of the 4 nodes of each exposed face, along -normal."""
    shape = labels.shape
    ny1, nz1 = shape[1] + 1, shape[2] + 1
    faces = exposed_faces(labels, mask)
    n_faces = sum(int(m.sum()) for _, _, m in faces)
    if n_faces == 0:
        raise UserError(
            f"pressure on load region '{p.region}': no face of the region looks toward empty space or a "
            "keep-out region (faces toward design, fixed or other load voxels get no pressure). "
            "Options: shape the region so that the loaded face is free, or use a force")
    ids, w = [], []
    vec_area = np.zeros(3)
    for axis, sign, m in faces:
        ijk = np.argwhere(m)
        if ijk.size == 0:
            continue
        vec_area[axis] += sign * h * h * len(ijk)
        a, b = [d for d in range(3) if d != axis]
        for da in (0, 1):
            for db in (0, 1):
                c = ijk.copy()
                c[:, axis] += 1 if sign > 0 else 0
                c[:, a] += da
                c[:, b] += db
                ids.append((c[:, 0] * ny1 + c[:, 1]) * nz1 + c[:, 2])
                wi = np.zeros((len(ijk), 3))
                wi[:, axis] = -p.value_mpa * h * h / 4.0 * sign     # force on the node: -p A n_out / 4
                w.append(wi)
    ids, w = np.concatenate(ids), np.concatenate(w)
    uniq, inv = np.unique(ids, return_inverse=True)
    f = np.zeros((uniq.size, 3))
    np.add.at(f, inv, w)
    pos = node_positions(uniq, shape, origin, h)
    about = pos.mean(axis=0)
    force = -p.value_mpa * vec_area + 0.0       # (no -0.0) independent of the nodal sum: the vector area of the faces
    audit = _rel(f.sum(axis=0) - force, force, float(np.abs(f).sum()))
    if audit > AUDIT:
        raise UserError(f"pressure on load region '{p.region}': nodal sum differs from p x area ({audit:.2g} relative)")
    info = {"nodes": int(uniq.size), "pressure_mpa": p.value_mpa, "faces": n_faces,
            "area_mm2": n_faces * h * h, "vector_area_mm2": [float(v) for v in vec_area], "audit_force_rel": audit}
    return LoadEntry("pressure", p.region, uniq, pos, f, about, force,
                     np.cross(pos - about, f).sum(axis=0), info)


def load_cases_json(problem, centers: dict) -> list:
    """The `load_cases` list of `run.json` (CONTRACTS.md section 5): `force` is the resultant, `kind` says what it is.

    A force entry gets `kind` and, when the YAML has them, `moment` (N mm, about `moment_about_mm`: the node
    centroid of the region, with the term `(at - centroid) x force` of a remote force added) and `at` (mm).
    A pressure is one more entry of the case: `force` is its resultant, with `pressure_mpa` and `area_mm2`.
    """
    cfg = problem.config
    ents = (problem.extended or {}).get("entries")
    out = []
    for ci, c in enumerate(cfg.load_cases):
        forces = []
        for fi, f in enumerate(c.forces):
            d = {"region": f.region, "force": list(f.force), "center": centers[f.region], "kind": entry_kind(f)}
            if f.moment is not None or f.at is not None:
                e = ents[ci][fi]
                d["moment"] = [float(v) for v in e.moment_nmm]
                d["moment_about_mm"] = [float(v) for v in e.about]
                if f.at is not None:
                    d["at"] = list(f.at)
            forces.append(d)
        for pi, p in enumerate(c.pressures):
            e = ents[ci][len(c.forces) + pi]
            forces.append({"region": p.region, "force": [float(v) for v in e.force_n],
                           "center": centers[p.region], "kind": "pressure",
                           "pressure_mpa": p.value_mpa, "area_mm2": e.info["area_mm2"]})
        out.append({"name": c.name, "weight": c.weight, "forces": forces})
    return out


def derived_extended(case, fs: list, entries) -> tuple[list, list]:
    """Facts of one load case for `derived_numbers`: add `kind`, `moment_nmm` and `at_mm` to the force dicts `fs`.

    Return `(pressures, resultant)`: one dict for each pressure of the case (`region`, `value_mpa`, and `faces`,
    `area_mm2`, `force_n` when the nodal data `entries` of the case is given) and the sum of their resultants (N).
    """
    for f, d in zip(case.forces, fs):
        d["kind"] = entry_kind(f)
        if f.moment is not None:
            d["moment_nmm"] = [float(v) for v in f.moment]
        if f.at is not None:
            d["at_mm"] = [float(v) for v in f.at]
    pressures, resultant = [], np.zeros(3)
    for j, p in enumerate(case.pressures):
        d = {"region": p.region, "value_mpa": p.value_mpa}
        if entries is not None:
            e = entries[len(case.forces) + j]
            d.update(faces=e.info["faces"], area_mm2=e.info["area_mm2"], force_n=[float(v) for v in e.force_n])
            resultant += e.force_n
        pressures.append(d)
    return pressures, [float(v) for v in resultant]


def extended_facts(problem) -> dict:
    """JSON facts of the moment, remote force, pressure and per-axis support loads of a `Problem` (section 26.2).

    Keys: `loads` (for each case its `entries`: `region`, `kind`, `force_n`, `moment_nmm`, `moment_about_mm`, and the
    facts of the entry: `nodes`, `at_mm`, `pressure_mpa`, `faces`, `area_mm2`, `audit_force_rel`, `audit_moment_rel`)
    and `supports` (`axes` for each fixed file, `fixed_dofs`, `free_modes`, `singular_value_ratio`) and, with
    springs, `springs` (for each region its faces, `faces_by_direction`, area, stiffness and `relative_to_Eh`; section 35) and,
    with joints, `joints` (for each joint its faces, beam, section, frame and bond facts; section 38).
    """
    cfg, ext = problem.config, problem.extended or {}
    out = {}
    if "entries" in ext:
        out["loads"] = [{"case": c.name, "entries": [
            {"region": e.region, "kind": e.kind, "force_n": [float(v) for v in e.force_n],
             "moment_nmm": [float(v) for v in e.moment_nmm], "moment_about_mm": [float(v) for v in e.about],
             **e.info} for e in es]} for c, es in zip(cfg.load_cases, ext["entries"])]
    if cfg.regions.fixed_axes or "springs" in ext or "joints" in ext:
        out["supports"] = {"axes": {p.name: "".join("xyz"[a] for a in ax)
                                    for p, ax in zip(cfg.regions.fixed, cfg.regions.fixed_axes)},
                           **ext.get("supports", {})}
    if "springs" in ext:        # section 35
        out["springs"] = ext["springs"]
    if "joints" in ext:         # section 38
        out["joints"] = ext["joints"]
    return out


def extended_lines(facts: dict | None) -> list[str]:
    """Text lines for `topopt check` and the run log from `extended_facts` (an empty list for None)."""
    out = []
    for c in (facts or {}).get("loads", []):
        for e in c["entries"]:
            if e["kind"] == "force":
                continue
            if e["kind"] == "pressure":
                out.append(f"pressure:       {c['case']} on '{e['region']}': {e['pressure_mpa']:g} MPa on {e['faces']} faces "
                           f"({e['area_mm2']:.4g} mm2), resultant {_vec(e['force_n'])} N; the nodal sum matches "
                           f"p x area to {e['audit_force_rel']:.1e} (relative)")
            else:
                out.append(f"load:           {c['case']} on '{e['region']}': {e['kind']}, resultant force {_vec(e['force_n'])} N"
                           + (f" at {_vec(e['at_mm'])} mm" if e.get("at_mm") else "")
                           + f", moment {_vec(e['moment_nmm'])} N mm about the node centroid {_vec(e['moment_about_mm'])} mm "
                           f"(distributing coupling, no added stiffness); audit: force {e['audit_force_rel']:.1e}, "
                           f"moment {e['audit_moment_rel']:.1e} (relative)")
    s = (facts or {}).get("supports")
    if s:
        modes = (f"rigid-body modes: {6 - s['free_modes']} of 6 blocked (singular value ratio "
                 f"{s['singular_value_ratio']:.2g})" if "free_modes" in s else s.get("rigid_body_test", ""))
        out.append("supports:       " + "; ".join(t for t in (
            ", ".join(f"{k} fixes {v}" for k, v in s["axes"].items()), modes) if t))
    for r in ((facts or {}).get("springs") or {}).get("regions", []):       # section 35
        by_dir = ", ".join(f"{k} {n}" for k, n in (r.get("faces_by_direction") or {}).items())
        out.append(f"spring:         '{r['region']}': {r['faces']} faces" + (f" ({by_dir})" if by_dir else "")
                   + f", {r['area_mm2']:.4g} mm2, "
                   f"{_vec(r['stiffness_n_per_mm'])} N/mm in total = {_vec(r['stiffness_per_area_n_per_mm3'])} N/mm3 "
                   f"(consistent face matrices; the region is solid and carries load); smallest stiffness "
                   f"{r['relative_to_Eh']:.2g} of E h")
    if (facts or {}).get("joints"):         # section 38
        from .joints import joint_lines
        out += joint_lines(facts["joints"])
    return out


def _vec(v) -> str:
    return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"


def fixed_dofs_axes(config: Config, members) -> np.ndarray:
    """Sorted unique fixed dofs when some `fixed` entry names axes: each entry fixes only its axes.

    `members[i]` is the bool voxel mask of entry `i` after the label priority (from `voxelize`).
    A node that two entries share gets the axes of both.
    """
    parts = []
    for mask, axes in zip(members, config.regions.fixed_axes):
        n = node_ids(mask)
        parts += [3 * n + a for a in axes]
    return np.unique(np.concatenate(parts))


def check_rigid_modes(fixed_dofs: np.ndarray, shape, h: float, spring_dofs=None) -> dict:
    """Test that the fixed dofs block the six rigid-body modes. Raise a `UserError` if one stays free.

    The test needs the fixed dofs only: a rigid motion of the part is blocked when no combination of the six
    modes (three translations, three rotations) has zero displacement on all fixed dofs. A free mode makes
    the stiffness matrix singular, so every solver would stop with its own error. Return the facts.
    `spring_dofs` (section 35) are the dofs that carry a spring with k > 0: they count as supports, because a mode
    with zero displacement there costs no spring energy. The test is geometric: it does not weigh the stiffness.
    """
    from .fe import rigid_body_modes
    nx, ny, nz = (int(s) for s in shape)
    scale = 0.5 * h * float(np.sqrt(nx * nx + ny * ny + nz * nz))     # half the grid diagonal, mm
    dofs = np.asarray(fixed_dofs, dtype=np.int64)
    if spring_dofs is not None:
        dofs = np.union1d(dofs, spring_dofs)
    if dofs.size:
        B = rigid_body_modes(dofs, shape, h)
        B[:, 3:] /= max(scale, 1e-300)        # rotation modes in units of the part size: like the translations
        R = np.linalg.qr(B, mode="r")          # (min(n, 6), 6): the singular values of B, without the long side
        _, sv, vt = np.linalg.svd(R)
    else:                                      # no support at all: the six modes are free
        sv, vt = np.zeros(0), np.eye(6)
    top = max(float(sv[0]), 1e-300) if sv.size else 1e-300
    rank = int((sv > RIGID_RCOND * top).sum())
    facts = {"fixed_dofs": int(len(fixed_dofs)), "free_modes": 6 - rank,
             "singular_value_ratio": float(sv[-1]) / top if sv.size == 6 else 0.0}
    if spring_dofs is not None:
        facts["spring_dofs"] = int(len(spring_dofs))
    if rank < 6:
        weight = (vt[rank:] ** 2).sum(axis=0)        # how much of each of the six modes is free
        names = [_MODE_NAMES[i] for i in range(6) if weight[i] > 0.05]
        raise UserError(
            f"the supports{' and springs' if spring_dofs is not None else ''} leave {6 - rank} of the 6 rigid-body "
            f"modes free ({', '.join(names)}). "
            "The part could move or turn without any support force, so the stiffness matrix is singular "
            "and no solver can solve it. Options: fix more axes of a support (axes: [x, y, z] is a full clamp), "
            "add a support region that blocks the free motion, "
            "give a spring a stiffness on the free axes (a spring on one axis blocks that translation only), "
            "add a joint on a load region (a beam to a fixed point blocks all six modes), "
            "or check that the support regions touch the part")
    return facts


def lost_force(config: Config, load_region_ids: np.ndarray, fixed_dofs: np.ndarray,
               forces: np.ndarray, entries=None) -> dict:
    """Force that the solver drops because its dof is fixed (a fact, no check).

    `forces` is the array of `build_forces` (before the drop). The solver keeps only the free
    dofs, so the force on the fixed dofs does no work. The numbers use the same array as the solver.
    A per-axis support drops only the force along its fixed axes. Returns `{"cases": [...], "max_lost_percent": float}`.
    Each case has `name`, `total_n`,
    `total_magnitude_n`, `lost_n`, `lost_magnitude_n`, `lost_percent` (lost magnitude over total
    magnitude; None when the total is about 0 but some force is lost), `lost_percent_of_sum`
    (lost magnitude over the sum of the region force magnitudes; it cannot cancel) and `regions` (for each load region with a loss: `region`, `nodes`,
    `fixed_nodes`, `lost_n`, `lost_magnitude_n`).

    With `entries` (`facts["entries"]` of `build_forces`) a case that has a moment, a remote force or a pressure is
    measured on its nodal forces: the sum of the sizes of the nodal forces replaces the sum of the region force
    sizes (for a plain force the two are equal). A lost force or moment at or below `AUDIT` of that sum is round-off
    and counts as 0. A case with no resultant force (a moment) has `lost_percent` None, so the text says that.
    Such a case also has `lost_moment_nmm`, `lost_moment_magnitude_nmm`, `lost_moment_percent` (of the applied
    moment; None when it is about 0 and some is lost) and `lost_moment_about_mm`: the sum over its entries about
    one point, the origin of the part coordinates. Each region has its own lost moment about its node centroid
    (`moment_about_mm`). A case of plain forces only has the keys and numbers of a YAML without the new types.
    `max_lost_percent` also counts the moment percent.
    """
    fd = np.asarray(fixed_dofs, dtype=np.int64)
    fixed_nodes = np.unique(fd // 3)
    fmask = None            # (n_nodes, 3) bool: the fixed dofs. Needed when a node has only some dofs fixed
    if fd.size != 3 * fixed_nodes.size or entries is not None:
        fmask = np.zeros(forces.shape[1], dtype=bool)
        fmask[fd] = True
        fmask = fmask.reshape(-1, 3)
    partial = fmask is not None and fd.size != 3 * fixed_nodes.size
    region_index = {lr.name: i for i, lr in enumerate(config.regions.load)}
    shared = {}
    cases = []
    for c, case in enumerate(config.load_cases):
        f3 = forces[c].reshape(-1, 3)
        total = f3.sum(axis=0)
        rows = f3[fixed_nodes]
        lost = (rows * fmask[fixed_nodes] if partial else rows).sum(axis=0)
        t_mag, l_mag = float(np.linalg.norm(total)), float(np.linalg.norm(lost))
        basis = float(sum(np.linalg.norm(f.force) for f in case.forces))
        regions = []
        ents = None if entries is None else entries[c]
        if ents is not None and all(e.kind == "force" for e in ents):
            ents = None      # plain forces only: the numbers and keys of a YAML without the new types
        if ents is None:
            for f in case.forces:
                idx = region_index[f.region]
                if idx not in shared:
                    n = node_ids(load_region_ids == idx)
                    shared[idx] = (n.size, int(np.isin(n, fixed_nodes).sum()),
                                   fmask[n].sum(axis=0) if partial else None)
                n_all, n_fix, n_fix_axis = shared[idx]
                if n_fix:
                    share = np.asarray(f.force, dtype=np.float64) * ((n_fix_axis if partial else n_fix) / n_all)
                    regions.append({"region": f.region, "nodes": int(n_all), "fixed_nodes": n_fix,
                                    "lost_n": [float(v) for v in share],
                                    "lost_magnitude_n": float(np.linalg.norm(share))})
        else:   # moment, remote force or pressure in the case: exact numbers from the nodal forces of each entry
            # The basis is the sum of the sizes of the nodal forces (for a plain force this is |F|). It does not
            # vanish for a pure moment, whose resultant is 0.
            basis = sum(float(np.linalg.norm(e.f, axis=1).sum()) for e in ents)
            m_lost, m_app, m_basis = np.zeros(3), np.zeros(3), 0.0
            for e in ents:
                keep = fmask[e.nodes]
                about0 = np.cross(e.pos, e.f)                     # moment of each nodal force about the origin
                m_app += about0.sum(axis=0)
                m_basis += float(np.linalg.norm(about0, axis=1).sum())
                if not keep.any():
                    continue
                lost_e = (e.f * keep).sum(axis=0)
                rel = np.cross(e.pos - e.about, e.f)              # about the region centroid
                lost_m = np.cross(e.pos - e.about, e.f * keep).sum(axis=0)
                if np.linalg.norm(lost_e) <= AUDIT * basis:        # round-off, not a loss
                    lost_e = np.zeros(3)
                if np.linalg.norm(lost_m) <= AUDIT * max(float(np.linalg.norm(rel, axis=1).sum()), 1e-300):
                    lost_m = np.zeros(3)
                m_lost += np.cross(e.pos, e.f * keep).sum(axis=0)
                regions.append({"region": e.region, "kind": e.kind, "nodes": int(e.nodes.size),
                                "fixed_nodes": int(keep.any(axis=1).sum()),
                                "lost_n": [float(v) for v in lost_e],
                                "lost_magnitude_n": float(np.linalg.norm(lost_e)),
                                "lost_moment_nmm": [float(v) for v in lost_m],
                                "lost_moment_magnitude_nmm": float(np.linalg.norm(lost_m)),
                                "moment_about_mm": [float(v) for v in e.about]})
            if l_mag <= AUDIT * basis:
                lost, l_mag = np.zeros(3), 0.0
            if float(np.linalg.norm(m_lost)) <= AUDIT * max(m_basis, 1e-300):
                m_lost = np.zeros(3)
            m_mag = float(np.linalg.norm(m_lost))
            regions = [r for r in regions if r["lost_magnitude_n"] > 0 or r["lost_moment_magnitude_nmm"] > 0]
        case_out = {"name": case.name,
                    "total_n": [float(v) for v in total], "total_magnitude_n": t_mag,
                    "lost_n": [float(v) for v in lost], "lost_magnitude_n": l_mag,
                    "lost_percent": (100.0 * l_mag / t_mag if t_mag > 1e-9 * max(basis, 1e-300)
                                     else (0.0 if l_mag == 0 else None)),
                    "lost_percent_of_sum": 100.0 * l_mag / basis if basis > 0 else 0.0,
                    "regions": regions}
        if ents is not None:
            app = float(np.linalg.norm(m_app))
            case_out["lost_moment_nmm"] = [float(v) for v in m_lost]
            case_out["lost_moment_magnitude_nmm"] = m_mag
            case_out["lost_moment_about_mm"] = [0.0, 0.0, 0.0]
            case_out["lost_moment_percent"] = (100.0 * m_mag / app if app > AUDIT * max(m_basis, 1e-300)
                                               else (0.0 if m_mag == 0 else None))
        cases.append(case_out)
    pct = [k["lost_percent"] if k["lost_percent"] is not None else k["lost_percent_of_sum"]
           for k in cases]
    pct += [k["lost_moment_percent"] for k in cases if k.get("lost_moment_percent") is not None]
    return {"cases": cases, "max_lost_percent": max(pct, default=0.0)}


def lost_force_warning(lf: dict) -> str | None:
    """One warning text for `check_data` when some force or moment is lost, else None."""
    bad = [k for k in lf["cases"] if k["lost_magnitude_n"] > 0 or k.get("lost_moment_magnitude_nmm", 0) > 0]
    if not bad:
        return None
    parts = []
    for k in bad:
        txt = ""
        if k["lost_magnitude_n"] > 0:
            p = k["lost_percent"]
            if p is None and "lost_moment_nmm" in k:     # a case with a moment and no resultant force
                txt = (f"'{k['name']}': the case has no resultant force (a moment), and {k['lost_magnitude_n']:.4g} N of "
                       f"nodal force is lost ({k['lost_percent_of_sum']:.1f} percent of the sum of the nodal force sizes)")
            elif p is None or p > 100:
                txt = (f"'{k['name']}': {k['lost_magnitude_n']:.4g} N of {k['total_magnitude_n']:.4g} N (forces of several "
                       f"regions cancel; {k['lost_percent_of_sum']:.1f} percent of the sum of the region force sizes)")
            else:
                txt = (f"'{k['name']}': {k['lost_magnitude_n']:.4g} N of {k['total_magnitude_n']:.4g} N "
                       f"({p:.1f} percent; {k['lost_percent_of_sum']:.1f} percent of the sum of the region force sizes)")
        if k.get("lost_moment_magnitude_nmm", 0) > 0:
            pm = k["lost_moment_percent"]
            mtxt = (f"lost moment {k['lost_moment_magnitude_nmm']:.4g} N mm about the origin"
                    + ("" if pm is None else f" ({pm:.1f} percent of the applied moment about the origin)"))
            txt = f"{txt}; {mtxt}" if txt else f"'{k['name']}': {mtxt}"
        parts.append(txt)
    return (f"force is lost: {'; '.join(parts)}. Reason: load nodes touch fixed nodes; their force is "
            "dropped by the solver (the part sees less force, and the stiffness figure is too high). "
            "Options: move the load region off the fixed region, shrink the overlap, or accept it.")


# --- load cases that repeat another case (F4-5, CONTRACTS.md section 13) -----------------------------------------------
#
# The model is linear. A case whose loads are the exact opposite of another case gives the displacements u -> -u, and a case
# with the same loads gives the same u. The compliance f.u, the relaxed von Mises stress, the springs and joints (they are
# part of K), and the displacement and alignment rows with the sense abs are even in u: they get the same value and the same
# gradient from both cases. Rows that follow the sign of u do not: the buckling row (the geometric stiffness Kg is linear
# in the stress, so its sign flips) and the displacement rows with the sense max or min. The functions below only state
# this as a fact with options. No tolerance: two loads are the same when every number is equal as a float.
#
# A group is the first case and every later case that repeats it or is its exact opposite. Two cases of a group have the same
# load or the exact opposite, so a group has at most two kinds of case: the kind of the first case (itself and the repeats) and
# the opposite kind. Two cases of one kind are copies in every row. A row can tell the two kinds apart only if it follows the sign.


def _force_order(t):
    """Sort key of a force entry `(region, force, moment, at)`: it never compares a vector with None."""
    return (t[0], t[1], t[2] is not None, t[2] or (), t[3] is not None, t[3] or ())


def _case_loads(case, sign: float) -> tuple:
    """The loads of a case as a tuple that does not depend on the order of the entries, every force, moment and pressure
    times `sign` (1.0 or -1.0). A remote force keeps its point `at`: the opposite load acts at the same point.
    Multiplying by -1.0 is exact in floating point, so there is no rounding in the comparison."""
    neg = lambda v: None if v is None else tuple(sign * x for x in v)
    forces = sorted(((f.region, neg(f.force), neg(f.moment), f.at) for f in case.forces), key=_force_order)
    pressures = sorted((p.region, sign * p.value_mpa) for p in case.pressures)
    return tuple(forces), tuple(pressures)


def _sign_rows(config, same: list, other: list) -> list[str]:
    """The rows of the YAML that follow the sign of the load: they give another value for a case of the opposite kind.

    `same`: the names of the cases that have the load of the first case; `other`: the names of the opposite cases. The buckling row
    has one row for each case. A displacement limit with the sense max or min tells the kinds apart only if it lists a case of each
    kind (`cases` lists every case if the YAML has no key); a limit that lists cases of one kind only reads the same value
    from every case it lists, and the other kind gives it nothing.
    """
    out = []
    if not other:
        return out
    if config.buckling.min_factor is not None:
        out.append("the buckling row (buckling.min_factor), because compression in one case is tension in the other, and a "
                   "case with no compression has no buckling load")
    if config.buckling.check:
        out.append("the buckling check after the run (buckling.check), for the same reason")
    for L in config.limits.displacement:
        if L.sense != "abs" and set(L.cases) & set(same) and set(L.cases) & set(other):
            out.append(f"the displacement limit '{L.name}' (sense {L.sense}), because it reads the sign of the motion")
    return out


def _listed_rows(config, names: list) -> list[dict]:
    """The `limits` rows that list some of the cases `names` in a `cases` key. The loader stops on a `cases` list that names a
    case which is not in the YAML, so a user who removes a case must change these rows. The loader fills a missing key with
    every case, so a row whose list is every case of the YAML is not found (it cannot be told from a row with no key)."""
    out = []
    for kind, rows in (("displacement", config.limits.displacement), ("alignment", config.limits.alignment)):
        for L in rows:
            named = [n for n in names if n in L.cases]
            if named and len(L.cases) < len(config.load_cases):
                out.append({"name": L.name, "kind": kind, "sense": getattr(L, "sense", None), "cases": named})
    return out


def repeated_load_cases(config) -> list[dict]:
    """The groups of load cases that have one load: one dict for each group of at least two cases (JSON-safe).

    Two cases match when they have the same entries of `forces` and `pressures`, in any order (the same load region, force,
    moment and `at`, and pressure for each entry, as a multiset): equal (`relations` "equal") or equal with the other sign
    (`relations` "opposite"). A remote force has the same point `at` in both. Two entries are the same when every number is equal
    as a float: no tolerance. A split entry (two entries on one region against one summed entry) or an extra zero entry gives no
    match. A group is the first case and every later case that matches it, so three cases with one load give one group of three.
    A load case with an all-zero load matches itself with both signs: it is "equal". Proportional cases with another factor
    are not matched.

    The dict has `cases` (the names; the first is the first case of the group), `relations` (one for each later case: its
    relation to the first), `weights`, `compliance_limits` and `stress_limits_mpa` (one value for each case, null if none),
    `objective` ("sum", "max" or "mass": how the weights count), `objective_weight` (the weight of the group as one case: the sum of
    the weights for "sum", the largest for "max", null for "mass"), `remote_force` (a case has a force with a point `at`),
    `buckling_row` (the YAML has the buckling row), `sign_rows` (the rows that follow the sign, as texts; only if a case is an
    opposite) and `limit_rows` (the `limits` rows that list a case of the group, see `_listed_rows`).
    """
    cases = list(config.load_cases)
    plus = [_case_loads(c, 1.0) for c in cases]
    minus = [_case_loads(c, -1.0) for c in cases]
    st = config.settings
    form = "mass" if st.objective == "mass" else st.case_combine
    found: dict[int, list] = {}
    for j in range(1, len(cases)):
        for i in range(j):
            if plus[i] == plus[j]:
                relation = "equal"
            elif minus[i] == plus[j]:
                relation = "opposite"
            else:
                continue
            found.setdefault(i, []).append((j, relation))
            break          # the first earlier case that matches is the first case of its group
    out = []
    for first in sorted(found):
        group = [cases[first]] + [cases[j] for j, _ in found[first]]
        relations = [r for _, r in found[first]]
        names, weights = [c.name for c in group], [c.weight for c in group]
        same = [names[0]] + [n for n, r in zip(names[1:], relations) if r == "equal"]
        other = [n for n, r in zip(names[1:], relations) if r == "opposite"]
        out.append({
            "cases": names, "relations": relations, "weights": weights,
            "compliance_limits": [c.compliance_limit for c in group],
            "stress_limits_mpa": [c.stress_limit_mpa for c in group],
            "objective": form,
            "objective_weight": None if form == "mass" else max(weights) if form == "max" else sum(weights),
            "remote_force": any(f.at is not None for c in group for f in c.forces),
            "buckling_row": config.buckling.min_factor is not None,
            "sign_rows": _sign_rows(config, same, other),
            "limit_rows": _listed_rows(config, names),
        })
    return out


def _g(x) -> str:
    """A number as its exact shortest text (`repr`), without the `.0` of a whole number, so a copied value equals the YAML value."""
    t = repr(float(x))
    return t[:-2] if t.endswith(".0") else t


def _names(names: list) -> str:
    """`'a'`, `'a' and 'b'`, `'a', 'b' and 'c'`."""
    q = [f"'{n}'" for n in names]
    return q[0] if len(q) == 1 else ", ".join(q[:-1]) + " and " + q[-1]


def _limit_text(group: dict, key: str, what: str, unit: str):
    """The sentence about one kind of limit of a group, and the limit that counts (the lowest). `("", None)` without a limit."""
    names = group["cases"]
    have = [(n, v) for n, v in zip(names, group[key]) if v is not None]
    if not have:
        return "", None
    pair = len(names) == 2
    low = min(v for _, v in have)
    if len(have) == 1:
        who, value = have[0]
        return (f"Only '{who}' has a {what} ({_g(value)}{unit}). It holds for {'both cases' if pair else 'all the cases'}, "
                "because they have the same compliance and the same stress."), value
    if all(v == low for _, v in have):
        lead = ("Both cases have" if pair else "All the cases have") if len(have) == len(names) else f"{_names([n for n, _ in have])} have"
        return f"{lead} the {what} {_g(low)}{unit}, so the {'two ' if len(have) == 2 else ''}rows are copies of each other.", low
    shown = [_g(v) + unit + ("" if len(have) == len(names) else f" ('{n}')") for n, v in have]
    return (f"The {what}s are {', '.join(shown[:-1])} and {shown[-1]}. "
            f"The {'lower' if len(have) == 2 else 'lowest'} one, {_g(low)}{unit}, is the one that counts."), low


def _rows_text(rows: list) -> str:
    """The sentence about the `limits` rows that list a case which the user may remove."""
    items = [f"the {r['kind']} limit '{r['name']}'" + (f" (sense {r['sense']})" if r["sense"] in ("max", "min") else "")
             + f" lists {_names(r['cases'])}" for r in rows]
    text = ("The `limits` rows that list cases: " + "; ".join(items) + ". A `cases` list must not name a case that you remove, or "
            "the loader stops: name the kept case in it.")
    if any(r["sense"] in ("max", "min") for r in rows):
        text += " If the kept case has the opposite load of the case that a max or min row lists, swap max and min."
    return text


def repeated_load_lines(groups: list | None) -> list[str]:
    """One `fact:` line for each group of `repeated_load_cases`: why the cases repeat, what differs, the cost and the options.

    The same text goes to `topopt check`, to the run log and to the warnings of `check_data` (so to the MCP tool `check_problem`).
    """
    lines = []
    for g in groups or []:
        names, rels = g["cases"], g["relations"]
        first, later = names[0], names[1:]
        pair = len(names) == 2
        kind, both = ("pair", "both cases") if pair else ("group", "all the cases")
        ops = [n for n, r in zip(later, rels) if r == "opposite"]
        eqs = [n for n, r in zip(later, rels) if r == "equal"]
        at = " (a remote force keeps its point `at`)" if g["remote_force"] else ""
        if pair and ops:
            parts = [f"fact: load case '{later[0]}' is the exact opposite of '{first}': the same load regions, and each force, "
                     f"moment and pressure has the other sign{at}."]
        elif pair:
            parts = [f"fact: load case '{later[0]}' repeats '{first}': the same load regions, and the same forces, moments and "
                     "pressures."]
        else:
            said = ([f"{_names(ops)} {'is' if len(ops) == 1 else 'are'} the exact opposite of '{first}'"] if ops else []) + \
                   ([f"{_names(eqs)} {'repeats' if len(eqs) == 1 else 'repeat'} '{first}'"] if eqs else [])
            parts = [f"fact: load cases {_names(names)} have one load: " + ", and ".join(said) + ". The load regions are the same, "
                     f"and each force, moment and pressure is equal{' or has the other sign' + at if ops else ''}."]
        if ops:
            parts.append("The model is linear, so the opposite load gives the same displacements with the other sign. The "
                         "compliance, the von Mises stress, and the displacement and alignment rows with the sense abs are then "
                         f"the same for {both}. The springs and joints are part of K, so they do not change this. The reactions "
                         "of the springs, the joints and the clamps have the other sign; the compliance and the stress do not.")
            if g["sign_rows"]:
                parts.append(f"These rows follow the sign, so they give other values for {_names(ops)}: "
                             + "; ".join(g["sign_rows"]) + ".")
            else:
                parts.append("No row of this YAML follows the sign. A buckling row or check would, and so would a displacement "
                             "limit with the sense max or min that lists cases of both signs. "
                             f"So {_names(later)} give{'s' if len(later) == 1 else ''} the optimizer no load that '{first}' "
                             "does not give.")
        else:
            parts.append(f"The model is linear, so {both} give the same displacements, and every row has the same value in "
                         f"{'both' if pair else 'all of them'}. Only the settings of the cases (the weight and the limits) can differ.")
        give = []
        if g["objective"] == "sum":
            parts.append(f"The weights add in the sum objective, so the {kind} acts like one case with the weight "
                         + " + ".join(_g(w) for w in g["weights"]) + f" = {_g(g['objective_weight'])}.")
            give.append(f"the weight {_g(g['objective_weight'])}")
        elif g["objective"] == "max":
            parts.append(f"The max objective takes the {'larger' if pair else 'largest'} weight, so the {kind} acts like one case "
                         f"with the weight {_g(g['objective_weight'])}.")
            give.append(f"the weight {_g(g['objective_weight'])}")
        for key, what, unit, name in (("compliance_limits", "compliance limit", " N*mm", "compliance_limit"),
                                      ("stress_limits_mpa", "stress limit", " MPa", "stress limit")):
            text, low = _limit_text(g, key, what, unit)
            if text:
                parts.append(text)
                give.append(f"the {name} {_g(low)}{unit}")
        who = _names(later) if pair else "each of " + _names(later)
        cost = (f"Cost: {who} adds one load column to the solve of every pass, and one adjoint solve for each stress, "
                "displacement, alignment or buckling row that it has.")
        if g["buckling_row"]:
            cost += f" The buckling row also does one eigen solve for each case in every pass, so {who} adds that cost too."
        parts.append(cost)
        quoted = ", ".join(f"'{n}'" for n in names)
        keep = f"keep one of the cases {quoted} and remove the other{'' if pair else 's'}" + \
               (f"; give the kept case {' and '.join(give)}" if give else "") + "."
        if g["sign_rows"]:
            opt = f"options: keep {both}, because the rows above follow the sign. Or, if you do not need these rows, {keep}"
        else:
            opt = f"options: {keep}"
        if g["limit_rows"]:
            opt += " " + _rows_text(g["limit_rows"])
        parts.append(opt + f" If the {kind} is a mistake, change one of the loads.")
        lines.append(" ".join(parts))
    return lines


# --- load cases that apply no load (U21, CONTRACTS.md section 13) ------------------------------------------------------
#
# A case applies no load when every component of every `force` and `moment` is exactly 0.0 and no pressure is other than 0.0
# (`LoadCase.applies_load`). No tolerance, and no new number. A case with a force sum of 0 is a load if its parts are not 0 (a
# squeeze, a moment), so the test reads the YAML numbers and not the nodal sum. The weight of a case is its share in the objective
# and is never a load (the engine has no body force).
#
# Measured on the hook, 4 passes: the compliance of such a case is 0 for every design, and its objective scale is 1
# (`Formulation.respond`: `abs(value) or 1.0`), so a run with no load at all minimizes a constant. A case with no load next to a case
# with a load adds nothing. Four rows cannot work with it and stopped the run with an internal error (exit 2) or gave no result:
# a `compliance_limit` on the case (the log form needs a value above 0), `case_combine: max` (the max form needs terms above 0), and the
# buckling row and check (the eigen solve has no start vector). `no_load_refusal` turns these into the loader error with options;
# `no_load_lines` is the fact for the rest.
#
# F4-37: the part also feels no load for other reasons, and they get the same answer (the same rows refuse, the same fact for
# the rest). No tolerance and no new number again; each test is an exact 0.0, apart from the one use of `np.finfo(float).tiny`.
#   "nodal_zero"  The load vector of the case is 0.0 at every node (`not np.any(f)`): opposite forces on the same nodes cancel, or a
#                 force is so small that its nodal share is 0.0 in double precision. The words do not say "fixed".
#   "fixed"       The load vector is not 0.0, and it is 0.0 on every free dof (`reaches_free_dofs`): every force lands on fixed dofs.
#                 The solver keeps the free dofs only, so it sees no load. The `lost_force` fact of the case says 100 percent.
#   "fixed_net"   The same, but the `lost_force` fact does not say 100 percent: the forces cancel in the net sum that the fact measures
#                 (a squeeze on fixed nodes). The words say so.
#                 Known where the grid and the fixed dofs are: `problem.build_problem` (`reach_kinds`). A case that loses a part of its
#                 load keeps the lost-force fact and gets no refusal: the rule is "no free dof gets any load", never a share.
#   "zero"        The load reaches a free dof, but the compliance of the case is 0.0 or subnormal at run time, that is below
#                 `np.finfo(float).tiny` (a force of 1e-300 N: c = f.u underflows). A subnormal value has lost its precision: the log
#                 form divides by it (`formulation.py`, `d / r.value`) and the eigen solve of the buckling row (ARPACK -9) breaks on
#                 it. This is the IEEE 754 definition of a subnormal value, no tuned threshold. Known only when the pass has run:
#                 `optimize.run` asks `zero_compliance_refusal` for the cases that the tests above call a load.
# The load vector on the free dofs is the load vector with the fixed dofs left out. The free dofs of the solver are the dofs of the
# active voxels that are not fixed, and the force sits on the nodes of LOAD voxels, which are active, so the two sets agree on every
# dof that has a force.

NO_LOAD = "every force and moment is exactly 0 and no pressure is other than 0"
NODAL_ZERO = ("the nodal forces of {the} add up to 0.0 at every node (opposite forces on the same nodes cancel, or a force is so small that "
              "its nodal share is 0.0 in double precision)")
FIXED = ("every force of {the} lands on fixed dofs, so the solver drops all of it and no free dof gets a load; "
         "the `lost_force` fact says 100 percent")
FIXED_NET = ("every nodal force of {the} lands on fixed dofs, so the solver drops all of it and no free dof gets a load; the forces cancel in the "
             "net sum that the `lost_force` fact measures, so that fact does not say 100 percent")
ZERO = "the compliance of {the} is 0.0 or subnormal in double precision, so its load does no measurable work on the part"
KINDS = ("yaml", "nodal_zero", "fixed", "fixed_net", "zero")


def reaches_free_dofs(forces: np.ndarray, fixed_dofs: np.ndarray) -> list[bool]:
    """For each case: True if its load vector is not exactly 0.0 on some free dof (a dof that is not fixed).

    `forces` is the array of `build_forces` `(n_cases, n_dof)`. Exact: `not np.any(f[free])`, with no tolerance. A case that applies no
    load (every number of the YAML is 0.0) is False too, so the entry is False for every case with no load, whatever the reason.
    """
    fixed = np.asarray(fixed_dofs, dtype=np.int64)
    return [bool(not np.isin(np.flatnonzero(f), fixed).all()) for f in forces]


def reach_kinds(forces: np.ndarray, fixed_dofs: np.ndarray, lost_of=None) -> list[str]:
    """For each case: why its load reaches a free dof or not. `free` (it does), `nodal_zero` (the whole load vector is 0.0), `fixed` (some nodal
    force is not 0.0, all of them are on fixed dofs, and the `lost_force` fact of the case says 100 percent) or `fixed_net` (the same, but the
    fact does not say 100 percent: the forces cancel in its net sum). A case with no load in the YAML is `nodal_zero` too.

    The tests are exact (`np.any`). `lost_of` is a function with no argument that gives the `lost_force` dict; it is called once, and only
    where a case has a force on fixed dofs and none on a free dof. The fact says 100 percent where it prints `100.0 percent` for the case
    (the lost force or, for a case with a moment, the lost moment).
    """
    out, lost = [], None
    for i, (f, reaches) in enumerate(zip(forces, reaches_free_dofs(forces, fixed_dofs))):
        if reaches:
            out.append("free")
        elif not np.any(f):
            out.append("nodal_zero")
        else:
            lost = lost_of() if lost is None else lost
            rec = lost["cases"][i]    # the force or the moment: a moment alone has no resultant force, its fact is `lost_moment_percent`
            says = [rec["lost_percent"], rec.get("lost_moment_percent")]
            out.append("fixed" if any(v is not None and f"{v:.1f}" == "100.0" for v in says) else "fixed_net")
    return out


def no_load_kinds(config, reach=None, runtime=()) -> dict:
    """`{name: kind}` of the load cases that do no work on the part, in the order of the YAML. The kind says why: `yaml` (U21: every
    number of the case is 0.0), `nodal_zero`, `fixed` or `fixed_net` (`reach[i]` of `reach_kinds`: the load vector on the free dofs is 0.0)
    or `zero` (the name is in `runtime`: the compliance of the case is 0.0 or subnormal). `reach` is None where there is no grid: only
    `yaml` cases are found then."""
    out = {}
    for i, c in enumerate(config.load_cases):
        if not c.applies_load:
            out[c.name] = "yaml"
        elif reach is not None and reach[i] != "free":
            out[c.name] = reach[i]
        elif c.name in runtime:
            out[c.name] = "zero"
    return out


def _plural(names: list, one: str, many: str) -> str:
    return one if len(names) == 1 else many


def _reason(kinds: dict) -> str:
    """The words in the brackets of the refusal and the fact: why each case does no work. One text if every case has the same reason,
    else one clause for each reason, with the names."""
    groups = {k: [n for n, kk in kinds.items() if kk == k] for k in KINDS}
    texts = {"yaml": NO_LOAD, "nodal_zero": NODAL_ZERO, "fixed": FIXED, "fixed_net": FIXED_NET, "zero": ZERO}
    texts = {k: t if k == "yaml" else t.format(the=_plural(groups[k], "the case", "each case")) for k, t in texts.items()}
    used = [k for k in KINDS if groups[k]]
    if len(used) == 1:
        return texts[used[0]]
    return "; ".join(f"{_names(groups[k])}: {texts[k]}" for k in used)


def _give(kinds: dict) -> str:
    """The first option of the refusal and the fact: what changes the load. One text for each way to change it (a `fixed` and a `fixed_net`
    case have the same one), joined with a semicolon. With more than one text each names its cases, so that "the case" is never unclear."""
    ways = {"yaml": ("yaml",), "nodal_zero": ("nodal_zero",), "fixed": ("fixed", "fixed_net"), "zero": ("zero",)}
    groups = {w: [n for n, kk in kinds.items() if kk in ks] for w, ks in ways.items()}
    several = sum(1 for g in groups.values() if g) > 1
    the = lambda g: _names(g) if several else _plural(g, "the case", "each case")   # noqa: E731
    texts = {"yaml": lambda g: f"give {the(g)} a force, a moment or a pressure",
             "nodal_zero": lambda g: f"put opposite forces of {the(g)} on different load regions, or give {the(g)} a larger force",
             "fixed": lambda g: f"move the load region of {_names(g)} off the fixed regions, so that its force reaches a free dof",
             "zero": lambda g: f"give {the(g)} a larger force, moment or pressure"}
    return "; ".join(texts[w](groups[w]) for w in ways if groups[w])


def no_load_refusal(config, reach=None, runtime=()) -> str | None:
    """The text of the loader error for a case that does no work next to a row that cannot work with it, else None.

    The reasons (each was measured, see above): no case applies a load and the objective is compliance (it is 0 for every design, so
    nothing in the YAML shapes the part); a `compliance_limit` on a case with no load; `case_combine: max` with the compliance objective;
    the buckling row or check. Every other case with no load gives only the fact of `no_load_lines`. A case with no load is one of the
    five kinds of `no_load_kinds`: `reach` (from `reach_kinds`) and `runtime` (the names of the cases with a compliance that is 0.0 or
    subnormal) add the kinds `nodal_zero`, `fixed`, `fixed_net` and `zero` to the `yaml` kind of U21.

    The options: change the load of the case (give it a load, put opposite forces on different regions, move it off the fixed regions,
    make it larger, each for its own kind); remove the case (not when
    no case does work: `load_cases` would be empty); and the option of each row (not under the compliance objective when no case does
    work: only a load helps then). The sentence about the `limits` rows follows the option "remove the case" and only a YAML with `limits`
    gets it.
    """
    kinds = no_load_kinds(config, reach, runtime)
    if not kinds:
        return None
    zero = [c for c in config.load_cases if c.name in kinds]
    st, bk = config.settings, config.buckling
    names = [c.name for c in zero]
    who = _names(names)
    every = len(zero) == len(config.load_cases)
    why, others = [], []
    if every and st.objective == "compliance":
        why.append("no case applies a load, and the objective is compliance: it is 0 for every design, so nothing in the YAML shapes the part")
    limited = [c for c in zero if c.compliance_limit is not None]
    if limited:
        late = any(kinds[c.name] == "zero" for c in limited)    # a run-time kind: the run stops at the first pass with that compliance, not always pass 0
        stop = (f"the run stops at the first pass where {_plural(limited, 'its', 'their')} compliance is 0.0 or subnormal" if late
                else "the run would stop at pass 0")
        why.append(f"{_names([c.name for c in limited])} {_plural(limited, 'has', 'have')} a compliance_limit, and the row needs a compliance "
                   f"above 0 ({stop})")
        others.append(f"remove the compliance_limit of {_plural(limited, 'the case', 'those cases')}")
    if st.objective == "compliance" and st.case_combine == "max":
        why.append("settings.case_combine is max, and the max objective needs a compliance above 0 for every case")
        others.append("use settings.case_combine: sum")
    keys = [k for k, on in (("buckling.min_factor", bk.min_factor is not None), ("buckling.check", bool(bk.check))) if on]
    if keys:
        why.append(f"{' and '.join(keys)} {_plural(keys, 'needs', 'need')} a load in every case (the eigen solve of a case with no load has "
                   "no start vector)")
        others.append("remove " + " and ".join(keys))
    if not why:
        return None
    options = [_give(kinds)]
    if not every:                       # with no case left, `load_cases` would be empty and the loader would stop on that
        options.append(f"remove {_plural(names, 'the case', 'the cases')}")
    if not (every and st.objective == "compliance"):   # with no load at all under the compliance objective, only a load helps
        options += others
    text = (f"load case{_plural(names, '', 's')} {who} {_plural(names, 'applies', 'apply')} no load ({_reason(kinds)}), and the run cannot work: "
            + "; ".join(why) + ". Options: " + "; ".join(options))
    if not every and (config.limits.displacement or config.limits.alignment):
        text += ". A `limits` row that lists a case you remove must name another case in `cases`"
    return text


def zero_compliance_refusal(config, compliance_cases, reach) -> str | None:
    """The user error for a run in which a case that has a load has a compliance that is 0.0 or subnormal in double precision (below
    `np.finfo(float).tiny`, the smallest normal double) next to a row that cannot work with it (the log form of a `compliance_limit`, the max
    form of `case_combine: max`, the buckling row), else None. `optimize.run` asks at every pass, before the rows are evaluated. A case with
    no load in the YAML or on the free dofs is not asked: the loader and `build_problem` have answered for it. `compliance_cases` has one
    number for each case.

    A subnormal value has lost its precision: the log form divides the slope by it (`d / r.value`) and the eigen solve of the buckling row
    (ARPACK -9) breaks on it. A normal value, however small, works.
    """
    runtime = [c.name for i, c in enumerate(config.load_cases)
               if c.applies_load and reach[i] == "free" and abs(compliance_cases[i]) < np.finfo(float).tiny]
    return no_load_refusal(config, reach, runtime) if runtime else None


def no_load_lines(config, reach=None) -> list[str]:
    """One `fact:` line for the load cases that apply no load: what follows and the options. [] if every case applies a load.

    The same text goes to `topopt check` and `--json`, to the first lines of `run.log`, to the warnings of `check_data` (so to the MCP tool
    `check_problem`) and to the part builder. A case that cannot run with a row (`no_load_refusal`) never reaches a run. `reach` (from
    `reach_kinds`, None without a grid) adds the cases whose load vector is 0.0 on the free dofs.
    """
    kinds = no_load_kinds(config, reach)
    if not kinds:
        return []
    zero = [c for c in config.load_cases if c.name in kinds]
    st = config.settings
    names = [c.name for c in zero]
    every = len(zero) == len(config.load_cases)
    parts = [f"fact: load case{_plural(names, '', 's')} {_names(names)} {_plural(names, 'applies', 'apply')} no load: {_reason(kinds)}. "
             f"A case with no load has compliance 0 and stress 0 for every design, so {_plural(names, 'it adds', 'they add')} nothing to "
             "the objective."]
    if st.objective == "compliance" and st.case_combine == "sum":
        weights = ", ".join(_g(c.weight) for c in zero)
        parts.append(f"The weight of the case ({weights}) counts for nothing." if len(names) == 1
                     else f"The weights of {_names(names)} ({weights}) count for nothing.")
    limited = [c for c in zero if c.stress_limit_mpa is not None]
    if limited:
        said = ", ".join(f"'{c.name}' {_g(c.stress_limit_mpa)} MPa" for c in limited)
        parts.append(f"The stress limit ({said}) is met by any design.")
    ids = {c.name for c in zero}
    rows = [(kind, L) for kind, group in (("displacement", config.limits.displacement), ("alignment", config.limits.alignment))
            for L in group if L.cases and set(L.cases) <= ids]
    if rows:
        parts.append("The " + "; the ".join(f"{kind} limit '{L.name}' lists {_names(list(L.cases))} only" for kind, L in rows)
                     + f", so any design meets {_plural(rows, 'it', 'them')}.")
    parts.append(f"Cost: {_plural(names, 'the case still adds', 'each case still adds')} one load column to the solve of every pass, and one "
                 "adjoint solve for each stress, displacement or alignment row that it has.")
    if every and st.objective == "mass":
        parts.append("No case applies a load, so the part has nothing to carry: the mass objective gives the lightest design that the rows "
                     "allow.")
    listed = _listed_rows(config, names)
    opt = f"options: {_give(kinds)}; or remove {_plural(names, 'the case', 'the cases')}."
    if listed:
        opt += (" The `limits` rows that list " + _plural(names, "it", "them") + ": "
                + "; ".join(f"the {r['kind']} limit '{r['name']}' lists {_names(r['cases'])}" for r in listed)
                + ". A `cases` list must not name a case that you remove, or the loader stops: name another case in it.")
    parts.append(opt)
    return [" ".join(parts)]
