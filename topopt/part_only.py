"""The part without the bodies that are not printed (`regions.not_printed`, CONTRACTS.md section 9.1).

A problem can hold bodies that are not part of the print: the legs and the axle that a bracket sits on, a plate that the part is
bolted to. They are fixed, load or solid regions (or keep-out regions) because the optimizer needs them, and `result/final.stl`
holds them. With `regions.not_printed` the post step also writes `result/part_only.stl`: the mesh of `final.stl` with the listed
bodies cut out by an exact boolean (manifold3d). `final.stl` does not change.

The cut uses the primitives of the exact interface step (`topopt.interfaces`, section 23): the STL of a removed body is loaded as a
manifold, subtracted from the mesh, and the result gets the same float32 clean-up as `final.stl`. The contact face of the part with a
removed body is the surface of that body (the STL), not a voxel step. The rule of the interface step that drops a body which no frozen
shape touches is NOT used: a piece that only a removed body held is part of the print.

Loose pieces. The volume step of the sdf method (CONTRACTS.md section 19, step 7) shifts the surface by one level, so `final.stl`
holds a thin skin around a frozen body. The cut leaves the skin of a removed body as thin loose pieces, and a loose piece that meets
the part along an edge makes the STL file non-manifold. A piece of the cut that holds no material of the density field is therefore
dropped, and the facts say how many and how much (`leftovers_dropped`). "Holds material" means: it contains the centre of a voxel with
the label design, fixed or load and a density above 0.5, the same threshold as the sharp field of the mesh step. A thin skin that is
joined to the part by a face stays; it is a fact of the mesh, not of this step.

The thin film at the contact faces (P4b-6). A skin that is joined to the part by a face stays as a thin wall behind the contact face of a
removed body. Such a wall carries load as a membrane in a stress check of the file (`topopt verify`), and it makes slivers for a mesher.
The facts always say how thin it is: for each removed body, the contact area, the thinnest wall at its contact faces (the lower end of the
film, not a value to use) and the contact area with a wall below the voxel size. With
`settings.part_only_min_film_mm` (a wall thickness in mm; default off) the cut also removes the film: the material of the part at a contact
triangle whose wall is thinner than that value. The rule is exact, with no distance test:
- A contact triangle is a triangle of the result that comes from the mesh of a removed body. Each input of the boolean carries its own
  original id (`Manifold.reserve_ids`, `Mesh.run_original_id`), and manifold3d keeps the id of the source on each output triangle.
- The wall of a contact triangle is the first hit of a ray from its centre into the part (against its outward normal) with a triangle
  other than the start triangle (excluded by its index). The second hit, if there is one, ends the air gap behind the wall.
- A thin triangle (wall below the value) gives a prism: the triangle and its copy moved along the inward normal by the value, but not past
  the middle of the air gap behind the wall on the centre ray of the triangle. Off the centre ray the prism cuts any material of the part
  under the triangle, up to its depth (a rib behind the film, the edge of a film). So `film_removed_mm3` is all the material that the cut
  removes. The prisms are joined and subtracted from the part. The rule for loose pieces then runs once more on the result. A contact
  triangle with a thicker wall keeps the exact face.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np

from .post import PART_ONLY_FILE
from .resource_errors import reraise_if_resource

log = logging.getLogger("topopt.part_only")

FILE = PART_ONLY_FILE
THRESHOLD = 0.5   # density above this is material: the cut of the sharp field in the mesh step (CONTRACTS.md section 19, step 1)


def bodies_of_run(run: dict) -> list[tuple[str, Path]]:
    """`(label, path)` of each removed body of a run, from the block `part_only` of `run.json` (written at the start of the run).

    `run` is the dict that `runinfo.resolved_run` gives, so that a relative `yaml_path` starts at the run folder. The path is the stored
    region file (relative to the YAML folder, or absolute), else the YAML folder plus the file name, as `post.region_paths` finds it.
    Return `[]` for a run without the block.
    """
    from .post import _resolve_region

    base = Path(run.get("yaml_path", "")).parent
    out = []
    for e in (run.get("part_only") or {}).get("entries") or []:
        stored = e.get("file")
        out.append((f"{e['role']}:{e['name']}", _resolve_region(base, stored, Path(stored or e["name"]).name)))
    return out


class FieldMaterial:
    """The voxel centres of the density field that hold material: label design, fixed or load, density above `THRESHOLD`."""

    def __init__(self, rho, labels, origin, h):
        from .post import DESIGN, FIXED, LOAD

        self.mask = ((labels == DESIGN) | (labels == FIXED) | (labels == LOAD)) & (np.asarray(rho) > THRESHOLD)
        self.origin = np.asarray(origin, float)
        self.h = float(h)

    def points_in(self, lo, hi) -> np.ndarray:
        """The centres of the material voxels inside the box `lo` to `hi` (mm), as an (n, 3) array."""
        shape = np.asarray(self.mask.shape)
        a = np.ceil((np.asarray(lo, float) - self.origin) / self.h - 0.5).astype(int)
        b = np.floor((np.asarray(hi, float) - self.origin) / self.h - 0.5).astype(int)
        a, b = np.maximum(a, 0), np.minimum(b, shape - 1)
        if (b < a).any():
            return np.zeros((0, 3))
        sub = self.mask[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]
        idx = np.argwhere(sub) + a
        return self.origin + self.h * (idx + 0.5)

    def held_by(self, verts, faces) -> bool:
        """True if the closed mesh contains the centre of at least one material voxel."""
        import trimesh

        from .post import contains_chunks

        v = np.asarray(verts, float)
        pts = self.points_in(v.min(axis=0), v.max(axis=0))
        if len(pts) == 0:
            return False
        pts = pts[np.argsort(np.linalg.norm(pts - v.mean(axis=0), axis=1))]   # the centre of a body is the likely place to be inside
        mesh = trimesh.Trimesh(v, np.asarray(faces, np.int64), process=False)
        return any(bool(c.any()) for c in contains_chunks(mesh, pts))


def _tagged(verts, faces):
    """`(manifold, id)` of a closed mesh. Every triangle of the manifold carries the new original `id`, and manifold3d keeps it on the
    output triangles of a boolean (`Mesh.run_original_id`). The geometry and the triangle order are those of `smooth_sdf._manifold_of`."""
    import manifold3d as m3d
    import trimesh

    def build(v, f):
        ident = m3d.Manifold.reserve_ids(1)
        mesh = m3d.Mesh(vert_properties=np.ascontiguousarray(v, np.float32), tri_verts=np.ascontiguousarray(f, np.uint32),
                        run_index=np.array([0, 3 * len(f)], np.uint32), run_original_id=np.array([ident], np.uint32))
        man = m3d.Manifold(mesh)
        if man.status() != m3d.Error.NoError:
            raise ValueError(f"not a manifold: {man.status()}")
        return man, int(ident)

    try:
        return build(verts, faces)
    except ValueError:   # the fallback of `smooth_sdf._manifold_of`
        tm = trimesh.Trimesh(verts, faces, process=True)
        tm.merge_vertices()
        return build(tm.vertices, tm.faces)


def _tagged_file(path):
    """`(manifold, id)` of an STL file: the load of `smooth_sdf._load_manifold`, with an id."""
    import trimesh

    t = trimesh.load(path, force="mesh")
    t.merge_vertices()
    return _tagged(t.vertices, t.faces)


def _pieces(man, material: FieldMaterial):
    """Split a manifold into connected pieces. Return `(kept, dropped)`: the pieces that hold material of the field (all, if there is one piece)."""
    from . import smooth_sdf

    parts = man.decompose() if not man.is_empty() else []
    keep, dropped = [], []
    for p in parts:
        v, f = smooth_sdf._from_manifold(p)
        (keep if len(parts) == 1 or material.held_by(v, f) else dropped).append(p)
    return keep, dropped


class Walls:
    """The contact triangles of a manifold and the wall behind each one.

    A contact triangle is an output triangle that comes from the mesh of a removed body (its original id). `owner` is the index in
    `labels`. The wall is measured along the inward normal from the centre of the triangle: `wall` is the distance to the first hit with
    a triangle other than the start triangle, `gap` the distance from that hit to the next one (the air behind the wall). Both are NaN
    when the ray has no such hit (or the triangle has no area, so no normal). Rows are the contact triangles only.
    """

    def __init__(self, man, ids: dict[str, int]):
        import trimesh

        self.labels = list(ids)
        mg = man.to_mesh64()
        self.verts = np.asarray(mg.vert_properties, np.float64)[:, :3]
        self.tris = np.asarray(mg.tri_verts, np.int64)
        run_ids, starts = np.asarray(mg.run_original_id), np.asarray(mg.run_index)
        per_tri = np.repeat(run_ids, np.diff(starts) // 3)
        owner = np.full(len(self.tris), -1, np.int64)
        for k, label in enumerate(self.labels):
            owner[per_tri == ids[label]] = k
        self.row_tri = np.flatnonzero(owner >= 0)
        self.owner = owner[self.row_tri]
        a, b, c = (self.verts[self.tris[self.row_tri, i]] for i in range(3))
        cross = np.cross(b - a, c - a)
        norm = np.linalg.norm(cross, axis=1)
        self.corners = np.stack([a, b, c], axis=1)
        self.area = norm / 2.0
        self.inward = np.zeros_like(cross)
        np.divide(-cross, norm[:, None], out=self.inward, where=norm[:, None] > 0)
        self.wall = np.full(len(self.row_tri), np.nan)
        self.gap = np.full(len(self.row_tri), np.nan)
        ray_rows = np.flatnonzero(norm > 0)
        if len(ray_rows) == 0:
            return
        origin = self.corners[ray_rows].mean(axis=1)
        direction = self.inward[ray_rows]
        mesh = trimesh.Trimesh(self.verts, self.tris, process=False)
        hit_tri, hit_ray, where = mesh.ray.intersects_id(origin, direction, multiple_hits=True, return_locations=True)
        keep = hit_tri != self.row_tri[ray_rows][hit_ray]          # the start triangle is excluded by its index
        hit_ray, where = hit_ray[keep], where[keep]
        if len(hit_ray) == 0:
            return
        dist = np.einsum("ij,ij->i", where - origin[hit_ray], direction[hit_ray])
        order = np.lexsort((dist, hit_ray))
        hit_ray, dist = hit_ray[order], dist[order]
        first = np.flatnonzero(np.r_[True, hit_ray[1:] != hit_ray[:-1]])
        self.wall[ray_rows[hit_ray[first]]] = dist[first]
        has_second = (first + 1 < len(hit_ray)) & (np.r_[hit_ray[1:], -1][first] == hit_ray[first])
        self.gap[ray_rows[hit_ray[first][has_second]]] = dist[first[has_second] + 1] - dist[first[has_second]]

    def facts(self, voxel_mm: float | None = None) -> dict:
        """`{label: {area_mm2, triangles, thinnest_wall_mm}}` (and `unmeasured_triangles` when a ray had no hit).

        With `voxel_mm` each record also has `area_below_voxel_mm2`: the contact area with a wall below the voxel size of the run.
        """
        out = {}
        for k, label in enumerate(self.labels):
            rows = self.owner == k
            wall = self.wall[rows]
            rec = {"area_mm2": round(float(self.area[rows].sum()), 4), "triangles": int(rows.sum()),
                   "thinnest_wall_mm": round(float(np.nanmin(wall)), 6) if np.isfinite(wall).any() else None}
            if voxel_mm is not None:
                rec["area_below_voxel_mm2"] = round(float(self.area[rows][np.nan_to_num(wall, nan=np.inf) < voxel_mm].sum()), 4)
            if np.isnan(wall).any():
                rec["unmeasured_triangles"] = int(np.isnan(wall).sum())
            out[label] = rec
        return out


def _film_prisms(walls: Walls, thin: np.ndarray, depth_mm: float):
    """One prism (a manifold) for each thin contact triangle: the triangle and its copy moved along the inward normal.

    The depth is `depth_mm` (the setting), but not past the middle of the air gap behind the wall on the centre ray of the triangle. Off
    that ray the prism cuts any material under the triangle, up to its depth. Return the list of manifolds and the owner of each.
    """
    import manifold3d as m3d

    rows = np.flatnonzero(thin)
    depth = np.where(np.isfinite(walls.gap[rows]), np.minimum(depth_mm, walls.wall[rows] + walls.gap[rows] / 2.0), depth_mm)
    prisms = []
    for r, d in zip(rows, depth):
        tri = walls.corners[r]
        prisms.append(m3d.Manifold.hull_points(np.vstack([tri, tri + d * walls.inward[r]])))
    return prisms, walls.owner[rows]


def _pinches(man) -> np.ndarray:
    """The positions where the surface of `man` touches itself: two vertices with the same float32 position that no edge joins.

    The file stores float32, so this is what a reader of the file sees as a vertex with two fans of faces (a non-manifold vertex). The
    test is exact (equal float32 coordinates), with no distance. Return an (n, 3) float32 array, one row for each position.
    """
    mg = man.to_mesh64()
    pos = np.asarray(mg.vert_properties, np.float64)[:, :3].astype(np.float32)
    tris = np.asarray(mg.tri_verts, np.int64)
    _, inverse, counts = np.unique(pos, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    crowded = counts[inverse] > 1
    if not crowded.any():
        return np.zeros((0, 3), np.float32)
    edge = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    joined = np.sort(edge[crowded[edge[:, 0]] & crowded[edge[:, 1]] & (inverse[edge[:, 0]] == inverse[edge[:, 1]])], axis=1)
    joined_set = {(int(x), int(y)) for x, y in joined}
    at: dict[int, list[int]] = {}
    for v in np.flatnonzero(crowded):
        at.setdefault(int(inverse[v]), []).append(int(v))
    bad = []
    for pos_id, vs in at.items():
        group = {v: v for v in vs}

        def find(x):
            while group[x] != x:
                group[x] = group[group[x]]
                x = group[x]
            return x

        for x in vs:                              # vertices that an edge joins are one vertex
            for y in vs:
                if x < y and (x, y) in joined_set:
                    group[find(x)] = find(y)
        if len({find(v) for v in vs}) > 1:
            bad.append(np.asarray(pos[vs[0]]))
    return np.array(bad, np.float32).reshape(-1, 3)


def _new_pinches(man, had: set) -> np.ndarray:
    """The pinch positions of `man` that are not in `had` (the positions that the part had before the cut): an (n, 3) float32 array."""
    return np.array([row for row in _pinches(man).tolist() if tuple(row) not in had], np.float32).reshape(-1, 3)


def _drop_film(part, walls: Walls, min_film_mm: float):
    """Subtract the prisms of the thin contact triangles from `part`. Return `(manifold, thin, cut, removed, had)`.

    `thin`: bool array over the rows of `walls` (wall below the value). `cut`: the rows that got a prism. A cut of the thin rows alone can
    leave the surface touching itself at a vertex (`_pinches`): a thin triangle between two thicker ones leaves two columns that meet along
    one line. Then the contact triangles that have a corner at that vertex get a prism too (each one at most to the depth of the value),
    and the cut runs again. The set only grows, so the loop ends. Pinches that the part already had before the cut are not touched.
    `removed`: label index -> volume of the part inside the prisms of that body (mm3), and the key `None` for the volume that the join
    of all prisms removed. `had`: the pinch positions of the part before the cut (the set that `_new_pinches` compares with).
    The loop can end with a pinch left, when no contact triangle with a wall has a corner there: `cut` counts what is left (`pinches_left`).
    """
    import manifold3d as m3d

    thin = np.isfinite(walls.wall) & (walls.wall < min_film_mm)
    cut, removed = thin.copy(), {}
    if not cut.any():
        return part, thin, cut, removed, set()
    corners32 = walls.corners.astype(np.float32)
    had = {tuple(row) for row in _pinches(part).tolist()}
    while True:
        prisms, owner = _film_prisms(walls, cut, min_film_mm)
        out = part - m3d.Manifold.batch_boolean(prisms, m3d.OpType.Add)
        bad = _new_pinches(out, had)
        at = (corners32[:, :, None, :] == bad[None, None, :, :]).all(axis=3).any(axis=(1, 2))   # contact triangles with a corner at a pinch
        more = at & ~cut & np.isfinite(walls.wall)
        if len(bad) == 0 or not more.any():
            break                              # done, or no triangle can change: `cut` counts the pinches that are left (`pinches_left`)
        cut = cut | more
    for k in sorted(set(owner.tolist())):
        mine = [p for p, o in zip(prisms, owner) if o == k]
        removed[k] = float((part ^ (m3d.Manifold.batch_boolean(mine, m3d.OpType.Add) if len(mine) > 1 else mine[0])).volume())
    removed[None] = float(part.volume() - out.volume())
    return out, thin, cut, removed, had


def _film_fault(exc: Exception) -> RuntimeError:
    """The error of a fault in the measure or in the rule with the setting on: what failed, and the option that avoids it."""
    return RuntimeError(f"the film rule failed ({type(exc).__name__}: {exc}). "
                        "Option: remove settings.part_only_min_film_mm to get the exact cut")


def cut(verts, faces, bodies: list[tuple[str, Path]], material: FieldMaterial, min_film_mm: float | None = None):
    """Cut the STL bodies out of the mesh. Return `(verts, faces, facts)`; the mesh is float32-exact, oriented and cleaned.

    `facts`: `overlap_mm3` (label -> volume of the mesh inside that body), `leftovers_dropped` (`bodies`, `volume_mm3`: the pieces
    of the cut that hold no material of the field), `contact` (label -> `area_mm2`, `triangles`, `thinnest_wall_mm`,
    `area_below_voxel_mm2` of the contact faces of that body; with `min_film_mm` also the film facts, see `_film_facts`, and
    `pinches_left`: the number of places where the cut surface touches itself at a vertex, 0 when the repair finished). `min_film_mm`
    is `settings.part_only_min_film_mm`:
    None is off (the mesh is the exact cut). With it off, a fault in the measure of the contact faces gives `contact` `{}` and the key
    `contact_error`; the mesh is the same. Raise `ValueError` if nothing is left. With `min_film_mm` a fault in the measure or in the rule is a
    `RuntimeError` that names the fault and the option (`_film_fault`).
    """
    import manifold3d as m3d

    from . import smooth_sdf
    from .interfaces import CLEAN_TOL_MM

    body, _body_id = _tagged(np.asarray(verts, np.float64), np.asarray(faces, np.int64))
    overlap, cutter, ids = {}, None, {}
    for label, path in bodies:
        man, ids[label] = _tagged_file(path)
        overlap[label] = round(float((body ^ man).volume()), 4)
        cutter = man if cutter is None else cutter + man
    keep, dropped = _pieces(body - cutter, material)
    if not keep:
        raise ValueError("the cut leaves no material of the density field: the removed bodies cover the part")
    part = m3d.Manifold.compose(keep) if len(keep) > 1 else keep[0]
    walls, contact, contact_error = None, {}, None
    try:
        walls = Walls(part, ids)
        contact = walls.facts(material.h)
    except Exception as exc:   # the facts are extra with the setting off: the file does not depend on them
        if min_film_mm is not None:
            raise _film_fault(exc) from exc
        contact_error = f"{type(exc).__name__}: {exc}"
        log.warning("part_only: the contact faces were not measured: %s", contact_error)
    film, pinches_left, empty = None, 0, False
    if min_film_mm is not None:
        try:
            part, thin, cut_rows, removed, had = _drop_film(part, walls, min_film_mm)
            if cut_rows.any():
                keep, more = _pieces(part, material)
                empty = not keep
                if not empty:
                    part = m3d.Manifold.compose(keep) if len(keep) > 1 else keep[0]
                    dropped += more
                    pinches_left = len(_new_pinches(part, had))      # measured on the part that is written, after the loose-piece rule
            if not empty:
                film = _film_facts(walls, thin, cut_rows, removed, Walls(part, ids) if cut_rows.any() else walls, contact, min_film_mm)
        except Exception as exc:
            raise _film_fault(exc) from exc
        if empty:
            raise ValueError(f"settings.part_only_min_film_mm of {min_film_mm:g} mm removes every piece of the part (every wall at a contact "
                             "face is thinner). Option: a smaller value, or remove the key")
    v, f = smooth_sdf._from_manifold(part)
    v, f = smooth_sdf._orient(v, f)
    v = v.astype(np.float32).astype(np.float64)
    v, f, clean = smooth_sdf._cleanup_float32(v, f, CLEAN_TOL_MM)
    facts = {"overlap_mm3": overlap,
             "leftovers_dropped": {"bodies": len(dropped), "volume_mm3": round(float(sum(p.volume() for p in dropped)), 4)},
             "contact": film["contact"] if film else contact,
             "clean": clean}
    if film:
        facts["film_removed_mm3"] = film["removed_mm3"]
        facts["pinches_left"] = pinches_left
    if contact_error:
        facts["contact_error"] = contact_error
    return v, f, facts


def _film_facts(walls: Walls, thin: np.ndarray, cut_rows: np.ndarray, removed: dict, after: Walls, contact: dict, min_film_mm: float) -> dict:
    """The film facts of one cut: the `contact` record of each body with the film keys, and the volume that the prisms removed in all.

    Per body: `thin_area_mm2` and `thin_triangles` (the contact triangles with a wall below the value), `cut_area_mm2` and `cut_triangles`
    (the triangles that got a prism: the thin ones, and the ones next to a vertex where the thin cut alone would leave the surface touching
    itself), `film_removed_mm3` (the volume of the part inside the prisms of that body: the film and any other material that a prism cuts), `area_after_mm2`, `thinnest_wall_after_mm` and
    `thin_area_after_mm2` (the contact faces that stay, measured again on the result: the cut makes new triangles, and a wedge of film can be
    thinner at the centre of a new triangle than at the centre of the old one).
    """
    out, now = {}, after.facts()
    for k, label in enumerate(walls.labels):
        mine, cut = thin & (walls.owner == k), cut_rows & (walls.owner == k)
        out[label] = {**contact[label],
                      "thin_area_mm2": round(float(walls.area[mine].sum()), 4), "thin_triangles": int(mine.sum()),
                      "cut_area_mm2": round(float(walls.area[cut].sum()), 4), "cut_triangles": int(cut.sum()),
                      "film_removed_mm3": round(removed.get(k, 0.0), 4),
                      "area_after_mm2": now[label]["area_mm2"], "thinnest_wall_after_mm": now[label]["thinnest_wall_mm"],
                      "thin_area_after_mm2": round(float(after.area[(after.owner == k) & (after.wall < min_film_mm)].sum()), 4)}
        if "unmeasured_triangles" in now[label]:
            out[label]["unmeasured_triangles_after"] = now[label]["unmeasured_triangles"]
    return {"contact": out, "removed_mm3": round(removed.get(None, 0.0), 4)}


def shortest_edge_mm(data: bytes) -> float | None:
    """The shortest edge of an STL file as written (mm), or None for an empty mesh. A fact about slivers, not a limit."""
    import trimesh

    back = trimesh.load(io.BytesIO(data), file_type="stl")
    return round(float(back.edges_unique_length.min()), 6) if len(back.faces) else None


def make(run: dict, verts, faces, rho, labels, origin, h, dens: float, final_facts: dict, method: str, result_dir) -> dict:
    """Make `result/part_only.stl` and return the block `part_only` of `metrics.json` (CONTRACTS.md section 9.1).

    `verts`, `faces` are the mesh of `final.stl`; `final_facts` is its reload record (`mesh.stl_roundtrip`); `dens` is the density
    in g/cm3; `method` is the mesh method that made the mesh. A fault gives `{"applied": false, "error": ...}` and no file: the
    run is finished, and `final.stl` is valid. A resource error (a memory error, a failed thread start) is not a fault of the cut: it raises
    `PostResourceError` and the post step stops (CONTRACTS.md section 19, #447). An older `part_only.stl` is gone in both cases.
    """
    import trimesh

    from . import interfaces
    from .post import _write_atomic

    t0 = time.monotonic()
    plan = run.get("part_only") or {}
    out_path = Path(result_dir) / FILE
    block: dict = {"applied": False, "file": None, "removed": list(plan.get("removed") or [])}
    min_film = plan.get("min_film_mm")   # settings.part_only_min_film_mm, written by the start of the run; absent when the setting is off
    if min_film is not None:
        block["min_film_mm"] = float(min_film)
    out_path.unlink(missing_ok=True)   # a file of an older post-process of this folder belongs to an older final.stl: gone before the cut
    try:
        bodies = bodies_of_run(run)
        missing = [str(q) for _l, q in bodies if not Path(q).is_file()]
        if missing:
            raise FileNotFoundError("region file not found: " + ", ".join(missing))
        v, f, facts = cut(verts, faces, bodies, FieldMaterial(rho, labels, origin, h), None if min_film is None else float(min_film))
        data = trimesh.Trimesh(v, f, process=False).export(file_type="stl")
        got = interfaces.mesh_facts(data, dens)
        before = float(final_facts["volume_mm3"])
        after = float(got["volume_mm3"])
        block.update({
            "applied": True, "file": FILE,
            "volume_before_mm3": round(before, 3), "volume_after_mm3": round(after, 3), "removed_mm3": round(before - after, 3),
            "overlap_mm3": facts["overlap_mm3"], "leftovers_dropped": facts["leftovers_dropped"], "contact": facts["contact"],
            "bodies": got["stl_roundtrip"]["bodies"], "mass_g": got["mass_g"], "triangles": got["triangles"],
            "watertight": bool(got["stl_roundtrip"]["ok"]), "shortest_edge_mm": shortest_edge_mm(data),
            "stl_roundtrip": got["stl_roundtrip"], "mesh_method": method,
        })
        if min_film is not None:
            block["film_removed_mm3"] = facts["film_removed_mm3"]
            block["pinches_left"] = facts["pinches_left"]
        notes = []
        if facts.get("contact_error"):
            notes.append(f"the contact faces were not measured ({facts['contact_error']}); the file is the exact cut")
        if not any(facts["overlap_mm3"].values()):
            notes.append("no removed body shares volume with final.stl: part_only.stl has the shape of final.stl")
        if method != "sdf":
            notes.append(f"final.stl came from the {method} method: a removed body is a voxel shape in it, so the cut leaves "
                         "voxel-step pieces next to the body (see leftovers_dropped and bodies). Option: settings.mesh_method: sdf")
        if facts.get("pinches_left"):
            notes.append(f"the cut leaves {facts['pinches_left']} place(s) where the surface of part_only.stl touches itself at a vertex (a "
                         "non-manifold vertex): topopt.tet refuses a surface like this. Options: another value of "
                         "settings.part_only_min_film_mm, or remove the key to get the exact cut")
        if min_film is None:
            thin = {label: (c["thinnest_wall_mm"], c.get("area_below_voxel_mm2", 0.0)) for label, c in facts["contact"].items()
                    if c["thinnest_wall_mm"] is not None and c["thinnest_wall_mm"] < float(h)}
            if thin:
                notes.append(film_note(thin, float(h)))
        if notes:
            block["notes"] = notes
        if not block["watertight"]:
            log.warning("part_only.stl is not one closed manifold after the float32 reload: %d bodies, %d open edges, "
                        "%d non-manifold edges, %d zero-area faces", got["stl_roundtrip"]["bodies"], got["stl_roundtrip"]["open_edges"],
                        got["stl_roundtrip"]["nonmanifold_edges"], got["stl_roundtrip"]["zero_area_faces"])
        _write_atomic(out_path, data)
    except Exception as exc:   # the run keeps final.stl; the record says what happened
        reraise_if_resource(exc, "the part-only cut", missing="result/part_only.stl")   # a resource error stops the post step (#447)
        block.update({"applied": False, "file": None, "error": f"{type(exc).__name__}: {exc}"})
        log.warning("part_only.stl not made: %s", block["error"])
    block["t_s"] = round(time.monotonic() - t0, 3)
    return block


def film_note(thin: dict, h: float) -> str:
    """The note for a cut with the film setting off. `thin`: label -> `(thinnest wall, contact area with a wall below the voxel size)` for the
    bodies whose thinnest wall is below the voxel size `h`. The note gives these facts, says that the thinnest wall is not a value to use,
    and gives the option with the way to find a value. It recommends no number."""
    bodies = "; ".join(f"{label}: thinnest wall {wall:g} mm, {area:g} mm2 of contact face with a wall below the voxel size"
                       for label, (wall, area) in thin.items())
    return (f"the contact faces of some removed bodies have a wall thinner than the voxel size ({h:g} mm): {bodies}. "
            "A thin film like this stays after the exact cut; it carries load as a membrane in a stress check of part_only.stl. "
            "The thinnest wall is the lower end of the film, not a value to use: a value just above it removes almost nothing. "
            "Option: write settings.part_only_min_film_mm: <the wall thickness that you want to cut off, in mm> in a new YAML file in the folder "
            "of the old one, and resume the finished run with it (the post step only; CONTRACTS.md section 9.1). Then read thin_area_mm2, "
            "cut_area_mm2 and thin_area_after_mm2 in metrics.part_only.contact, and change the value if the film that you want gone is still there")


def log_lines(block: dict | None) -> list[str]:
    """The lines for `run.log` and the screen: what was cut, how much volume, how many bodies. Empty for a run without the block."""
    if not block:
        return []
    if not block.get("applied"):
        return [f"part_only: not made ({block.get('error', 'no reason recorded')})"]
    line = (f"part_only: {len(block['removed'])} body(ies) cut out ({', '.join(block['removed'])}); "
            f"volume {block['volume_before_mm3']} -> {block['volume_after_mm3']} mm3, {block['bodies']} body(ies) in "
            f"result/{block['file']}" + ("" if block["watertight"] else ", NOT one closed manifold"))
    out = [line]
    lost = block["leftovers_dropped"]
    if lost["bodies"]:
        out.append(f"part_only: dropped {lost['bodies']} loose piece(s) of {lost['volume_mm3']} mm3 that hold no material of the field")
    if block.get("min_film_mm") is not None:
        rec = list((block.get("contact") or {}).values())
        left = [c["thinnest_wall_after_mm"] for c in rec if c.get("thinnest_wall_after_mm") is not None]
        out.append(f"part_only: film rule (settings.part_only_min_film_mm {block['min_film_mm']:g} mm): "
                   f"thin {sum(c.get('thin_area_mm2', 0.0) for c in rec):.3f} mm2 ({sum(c.get('thin_triangles', 0) for c in rec)} triangle(s)); "
                   f"cut {sum(c.get('cut_area_mm2', 0.0) for c in rec):.3f} mm2 ({sum(c.get('cut_triangles', 0) for c in rec)} triangle(s), "
                   f"with the repair of pinch vertices); {block.get('film_removed_mm3', 0.0):.4f} mm3 of material removed; the thinnest wall that stays is "
                   + (f"{min(left):g} mm" if left else "not measured (no contact face is left)"))
    out += ["part_only: " + n for n in block.get("notes", [])]
    return out
