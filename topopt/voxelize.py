"""Voxel grid and region labels from STL files (CONTRACTS.md section 3).

Method: for each region, cast one ray along +z for each (i, j) column. The
ray hits the surface at some z values. Sorted hits come in pairs (enter,
leave). The voxel centres between a pair are inside. This is fast because
the number of rays is nx*ny, not nx*ny*nz.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import trimesh

from . import symmetry
from .config import Config, UserError
from .unit_fact import largest_size, remember_size, unit_hint

log = logging.getLogger("topopt.voxelize")

EMPTY, DESIGN, FIXED, LOAD, KEEP_OUT = 0, 1, 2, 3, 4

# Small shifts of the ray start, as a fraction of h. They move the ray off
# edges and vertices of the mesh, so a ray does not hit two triangles at once.
_JITTER = (0.000731, 0.000419)


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Read one STL. Stop if it is empty or not watertight."""
    try:
        mesh = trimesh.load(path, file_type="stl", force="mesh")
    except Exception as e:  # noqa: BLE001 - show any read error as a user error
        raise UserError(f"cannot read STL {path}: {e}")
    if len(mesh.faces) == 0:
        raise UserError(f"STL has no triangles: {path}")
    if not mesh.is_watertight:
        raise UserError(f"STL is not watertight (not closed): {path}. {_open_edges_text(mesh)} "
                        "To close it: export the part again from the CAD tool as a closed solid, or repair the file with a mesh tool, "
                        "for example the filter Close Holes of MeshLab or the 3D-Print Toolbox of Blender; in Python, "
                        "trimesh.repair.fill_holes(mesh) closes a hole of 3 or 4 edges, and "
                        "trimesh.repair.fill_holes(mesh, use_fan=True) a larger convex hole (trimesh is installed with topopt)")
    return mesh


def _open_edges_text(mesh: trimesh.Trimesh) -> str:
    """What is open: the edges that one face uses (a hole) and the edges that more than two faces use (not a manifold)."""
    e = mesh.edges_sorted.astype(np.int64)
    _, counts = np.unique(e[:, 0] * max(len(mesh.vertices), 1) + e[:, 1], return_counts=True)
    return (f"It has {int((counts == 1).sum()):,} open edge(s) (an edge that only one face uses) and "
            f"{int((counts > 2).sum()):,} non-manifold edge(s) (an edge that more than two faces use).")


def suggest_voxel_size(h: float, count: float, limit: float) -> float:
    """A voxel size that makes `count` voxels (at size h) fit under `limit`."""
    s = h * (count / limit) ** (1.0 / 3.0) * 1.03
    return math.ceil(s * 100.0) / 100.0


def grid_from_bounds(lo: np.ndarray, hi: np.ndarray, h: float, max_voxels: float | None = None):
    """Origin and shape: the box lo..hi snapped outward to whole voxels.

    With `max_voxels`, the voxel count is computed in floating point first. A count above the
    limit raises the grid-guard `UserError` before the integer cast (a tiny `h` overflows it).
    """
    nf = np.maximum(np.ceil((hi - lo) / h - 1e-6), 1.0)
    if max_voxels is not None:
        count = float(np.prod(nf))
        if count > max_voxels:
            try:
                advice = f"Try voxel_size {suggest_voxel_size(h, count, max_voxels)} or more, or raise"
            except (OverflowError, ValueError):  # count is infinite
                advice = "Use a larger voxel_size, or raise"
            count_txt = f"{int(count)}" if count < 1e15 else f"about {count:.3g}"
            raise UserError(
                f"grid would have {count_txt} voxels at voxel_size {h}, the limit is "
                f"{max_voxels} (max_grid_voxels). {advice} "
                f"settings.max_grid_voxels in the YAML if this machine has the memory")
    n = nf.astype(np.int64)
    n = np.maximum(n, 1)
    return lo.astype(np.float64), tuple(int(x) for x in n)


def inside_mask(mesh: trimesh.Trimesh, origin: np.ndarray, h: float,
                shape: tuple[int, int, int]) -> np.ndarray:
    """Bool array (nx, ny, nz): True where the voxel centre is inside the mesh."""
    nx, ny, nz = shape
    mask = np.zeros(shape, dtype=bool)
    lo, hi = mesh.bounds
    # Only the columns under the bounding box of this mesh.
    i0 = max(int(math.floor((lo[0] - origin[0]) / h)), 0)
    i1 = min(int(math.ceil((hi[0] - origin[0]) / h)), nx)
    j0 = max(int(math.floor((lo[1] - origin[1]) / h)), 0)
    j1 = min(int(math.ceil((hi[1] - origin[1]) / h)), ny)
    if i1 <= i0 or j1 <= j0:
        return mask
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()
    z_start = lo[2] - h
    org = np.empty((ii.size, 3))
    org[:, 0] = origin[0] + h * (ii + 0.5 + _JITTER[0])
    org[:, 1] = origin[1] + h * (jj + 0.5 + _JITTER[1])
    org[:, 2] = z_start
    dirs = np.tile([0.0, 0.0, 1.0], (ii.size, 1))
    locs, ray_ids, _ = mesh.ray.intersects_location(org, dirs, multiple_hits=True)
    if len(ray_ids) == 0:
        return mask
    order = np.lexsort((locs[:, 2], ray_ids))
    ray_ids = ray_ids[order]
    z = locs[order, 2]
    # Position of each hit inside its ray (0, 1, 2, ...).
    counts = np.bincount(ray_ids, minlength=ii.size)
    starts = np.cumsum(counts) - counts
    rank = np.arange(len(ray_ids)) - starts[ray_ids]
    # A ray with an odd number of hits is broken. Drop it.
    good = (counts % 2 == 0)[ray_ids]
    ray_ids, z, rank = ray_ids[good], z[good], rank[good]
    # Voxel index of the first centre at or above z.
    kz = np.ceil((z - origin[2]) / h - 0.5).astype(np.int64)
    kz = np.clip(kz, 0, nz)
    # Difference array along z: +1 at enter, -1 at leave.
    diff = np.zeros((ii.size, nz + 1), dtype=np.int8)
    sign = np.where(rank % 2 == 0, 1, -1).astype(np.int8)
    np.add.at(diff, (ray_ids, kz), sign)
    col = np.cumsum(diff[:, :nz], axis=1) > 0
    mask[ii, jj, :] = col
    return mask


def interface_add_meshes(config: Config) -> dict:
    """Name -> trimesh of each union interface (empty without interfaces). The shapes are in `topopt/interfaces.py`."""
    if not config.interfaces:
        return {}
    from . import interfaces

    return interfaces.add_meshes(config)


def interface_masks(config: Config, adds: dict, origin: np.ndarray, h: float, shape, warn: bool = True):
    """Voxel masks of the interface shapes (CONTRACTS.md section 23): `(union masks, cut masks)`, two dicts name -> bool array.

    The labels use the existing rule of this module (a voxel is inside when its centre is inside). `adds` is the dict of
    `interfaces.add_meshes(config)`. The hole of a tube is a cut shape named `<name> (hole)`. A shape that holds no voxel
    gets a log warning (`warn`): the labels ignore it, and the exact step in the post-process still applies it.
    Without interfaces both dicts are empty.
    """
    if not config.interfaces:
        return {}, {}
    from . import interfaces

    cuts = interfaces.cut_meshes(config, (origin, origin + h * np.asarray(shape, float)))
    out = []
    for group in (adds, cuts):
        masks = {}
        for name, m in group.items():
            mk = inside_mask(m, origin, h, shape)
            if warn and not mk.any():
                log.warning("interface '%s' has no voxel at voxel_size %s: the labels ignore it; the exact step still applies it", name, h)
            masks[name] = mk
        out.append(masks)
    return out[0], out[1]


def compose_labels(reg, masks: dict, iface_union: dict, iface_cut: dict):
    """Labels and load ids from the voxel masks of the regions, by priority: keep_out > fixed > load > solid > design.

    `masks` has one boolean array for each region (the key is the label that `voxelize` gives it); all arrays have one shape,
    the whole grid or a part of it (the plane facts use it for one layer). `iface_union` and `iface_cut` are the masks of the
    interface shapes (`interface_masks`). Return `(labels, load_ids)`; `load_ids` is -1 where the label is not LOAD or the voxel
    is a solid or union voxel (LOAD with no load case).
    """
    shape = masks["design"].shape
    labels = np.zeros(shape, dtype=np.uint8)
    load_ids = np.full(shape, -1, dtype=np.int16)
    labels[masks["design"]] = DESIGN
    for mk in iface_union.values():   # a union interface is LOAD without a load case (load_ids stays -1); a load region wins on overlap
        labels[mk] = LOAD
    # Solid: frozen at density 1 like a load region with no force (label LOAD, no load id).
    # It is set before the loads, so a load region keeps its voxels (priority: load > solid).
    for i, p in enumerate(reg.solid):
        labels[masks[f"solid[{i}] ({p.name})"]] = LOAD
    # Load: low priority first, so the first region in the list wins on overlap.
    for idx in range(len(reg.load) - 1, -1, -1):
        mk = masks[f"load '{reg.load[idx].name}'"]
        labels[mk] = LOAD
        load_ids[mk] = idx
    for i, p in enumerate(reg.fixed):
        labels[masks[f"fixed[{i}] ({p.name})"]] = FIXED
    for i, p in enumerate(reg.keep_out):
        labels[masks[f"keep_out[{i}] ({p.name})"]] = KEEP_OUT
    for mk in iface_cut.values():   # a cut interface is KEEP_OUT, like a keep-out region
        labels[mk] = KEEP_OUT
    load_ids[labels != LOAD] = -1
    return labels, load_ids


def voxelize(config: Config, facts: dict | None = None):
    """Return `(origin, h, shape, labels, load_region_ids)` for a config.

    With `facts` (a dict), the function adds `facts["solid"]` when the config has `regions.solid`:
    the counts of solid voxels and of the overlaps with other regions (CONTRACTS.md section 22a).
    The counts are facts. None of them stops the build.
    With a dict `facts`, the function adds `facts["fixed_members"]` when a `fixed` entry names axes
    (`regions.fixed_axes`): for each entry, the voxels that it keeps after the label priority.
    """
    reg = config.regions
    st = config.settings
    h = st.voxel_size

    names = [("design", reg.design, ("design", "design"))]     # (label, file, (role, name) as the plane facts name the region)
    names += [(f"fixed[{i}] ({p.name})", p, ("fixed", p.stem)) for i, p in enumerate(reg.fixed)]
    names += [(f"keep_out[{i}] ({p.name})", p, ("keep_out", p.stem)) for i, p in enumerate(reg.keep_out)]
    names += [(f"load '{lr.name}'", lr.path, ("load", lr.name)) for lr in reg.load]
    names += [(f"solid[{i}] ({p.name})", p, ("solid", p.stem)) for i, p in enumerate(reg.solid)]
    meshes = {}
    for label, p, _role in names:
        meshes[label] = load_mesh(p)
        if label == "design":
            remember_size(p, meshes[label])   # TC-3: the unit fact takes the size from this mesh (no second read of the file)

    def with_unit_hint(text: str) -> str:   # TC-3: a part in inches or metres loads too small; the stop texts below get the sentence of the fact (the same rule)
        hint = unit_hint(config, largest_size(meshes["design"]))
        return f"{text}. {hint}" if hint else text

    iface_adds = interface_add_meshes(config)   # {} without interfaces; a union shape widens the grid like a region
    lo = np.min([m.bounds[0] for m in (*meshes.values(), *iface_adds.values())], axis=0)
    hi = np.max([m.bounds[1] for m in (*meshes.values(), *iface_adds.values())], axis=0)
    lo = symmetry.shift_lo(config.mirror, lo, h)   # mirror key: each plane becomes a grid plane (section 26.1)
    try:
        origin, shape = grid_from_bounds(lo, hi, h, st.max_grid_voxels)
    except UserError as e:     # the grid guard is in grid_from_bounds (before the integer cast)
        raise UserError(with_unit_hint(str(e))) from e

    masks = {}
    for label, m in meshes.items():
        mk = inside_mask(m, origin, h, shape)
        if not mk.any():
            raise UserError(with_unit_hint(
                f"region {label} has no voxel at voxel_size {h}. "
                f"Use a smaller voxel_size or make the region larger"))
        masks[label] = mk

    iface_union, iface_cut = interface_masks(config, iface_adds, origin, h, shape)
    labels, load_ids = compose_labels(reg, masks, iface_union, iface_cut)
    for name, mk in iface_union.items():   # a keep-out region wins over a union interface: say how much of it that takes
        for i, p in enumerate(reg.keep_out):
            n = int((mk & masks[f"keep_out[{i}] ({p.name})"]).sum())
            if n:
                log.warning("union interface '%s' overlaps keep-out region %s in %d voxels: the keep-out region wins "
                            "(KEEP_OUT label; the exact step cuts the region out of the shape)", name, p.name, n)

    n_design = int((labels == DESIGN).sum())
    if n_design == 0:
        msg = "region design has no free voxel left after fixed, load and keep_out"
        taken = [f"{name} ({'union' if name in iface_union else 'cut'}) {int((mk & masks['design']).sum())}"
                 for name, mk in {**iface_union, **iface_cut}.items() if (mk & masks["design"]).any()]
        if taken:   # name the interfaces that took the voxels (a face with the normal on the part side takes them all)
            msg = ("region design has no free voxel left after fixed, load, keep_out and interfaces. Design voxels that each "
                   f"interface takes: {', '.join(taken)}. Check the `normal` of a face, the `axis` of a bore, and the sizes")
        raise UserError(msg)
    for idx, lr in enumerate(reg.load):
        if not (load_ids == idx).any():
            raise UserError(f"load region '{lr.name}' has no voxel left (fixed or keep_out covers it)")
    for i, p in enumerate(reg.fixed):
        if not (masks[f"fixed[{i}] ({p.name})"] & (labels == FIXED)).any():
            raise UserError(f"fixed region {p.name} has no voxel left (keep_out covers it)")

    if facts is not None:   # flat faces that are not on a voxel face: the voxelized region is larger or smaller than the STL
        def changed(label, axis, side, effect):     # how many voxels of the part the plane moves (see `plane_change`)
            return plane_change(reg, masks, iface_union, iface_cut, labels, load_ids, label, axis, side, effect)
        facts["planes"] = plane_facts([(*role, meshes[label], label) for label, _p, role in names], origin, h, changed)
    if facts is not None and reg.solid:
        facts["solid"] = solid_facts(config, masks, labels, load_ids)
    if facts is not None:   # what the cuts take from the frozen regions (CONTRACTS.md section 23); the key is absent when no cut takes a voxel
        taken = taken_facts(config, meshes, masks, iface_adds, iface_union, iface_cut, origin, h)
        if taken:
            facts["taken"] = taken
    if facts is not None and reg.fixed_axes:   # per-axis supports (CONTRACTS.md section 26.2)
        facts["fixed_members"] = [masks[f"fixed[{i}] ({p.name})"] & (labels == FIXED)
                                  for i, p in enumerate(reg.fixed)]

    active = int(((labels >= DESIGN) & (labels <= LOAD)).sum())
    if active > st.max_active_voxels:
        raise UserError(with_unit_hint(
            f"model has {active} active voxels, the limit is {st.max_active_voxels} "
            f"(max_active_voxels). Try voxel_size "
            f"{suggest_voxel_size(h, active, st.max_active_voxels)} or more, or raise "
            f"settings.max_active_voxels in the YAML if this machine has the memory"))
    return origin, h, shape, labels, load_ids


def solid_facts(config: Config, masks: dict, labels: np.ndarray, load_ids: np.ndarray) -> dict:
    """Counts for `regions.solid`: what stays solid, and what the other regions take or add.

    Priority: keep_out > fixed > load > solid > design. A voxel of a solid STL that lies in a
    region of higher priority takes that label. The counts are facts, not errors. They are read from the final
    `labels`, so every cut or frozen shape that the labels hold is counted, also one that has no region file.
    """
    reg = config.regions
    solid_masks = [masks[f"solid[{i}] ({p.name})"] for i, p in enumerate(reg.solid)]
    union = np.logical_or.reduce(solid_masks)
    design = masks["design"]
    # What wins at a voxel is read from the final labels, not from the region masks: so a cut that the labels hold but
    # the masks do not (an interface cutter of CONTRACTS.md section 23) is counted too.
    kept = union & (labels == LOAD) & (load_ids < 0)   # the voxels that are solid and nothing else
    count = lambda m: int(m.sum())  # noqa: E731
    return {
        "stl_voxels": count(union),                     # voxels of all solid STLs together
        "voxels": count(kept),                          # solid voxels that stay solid (label LOAD, no load id)
        "inside_design_stl": count(kept & design),
        "outside_design_stl": count(kept & ~design),    # they add material outside the design STL
        # The three counts do not overlap: each voxel is counted for the region that wins.
        "in_keep_out": count(union & (labels == KEEP_OUT)),   # are removed (keep_out wins)
        "in_fixed": count(union & (labels == FIXED)),         # take the label FIXED
        "in_load": count(union & (labels == LOAD) & (load_ids >= 0)),   # take the label LOAD with a load id
        "in_two_solid_stls": int((np.sum(solid_masks, axis=0, dtype=np.int16) > 1).sum()),
        "regions": [{"name": p.stem, "stl_voxels": count(m), "voxels": count(m & kept),
                     "thickness_voxels": _thickness_voxels(m & kept)}
                    for p, m in zip(reg.solid, solid_masks)],
    }


TRIM_MIN_MM = 2.0   # a cut may take this much of a frozen region and still count as a machining trim


def _bbox(mask: np.ndarray):
    """Slices of the bounding box of `mask`, or None for an empty mask."""
    if not mask.any():
        return None
    return tuple(slice(int(a.min()), int(a.max()) + 1) for a in np.nonzero(mask))


def _depth_into_cut(cut: np.ndarray, taken: np.ndarray, h: float) -> float:
    """How far (mm) the deepest voxel of `taken` lies inside the cut shape `cut`: the distance from its centre to the nearest voxel outside the shape, less half a voxel.

    A voxel next to the outside is half a voxel deep. The shape ends at its own voxels (a padding of one empty voxel), so this is for a cut with a closed shape (an STL or a keep-out region).
    """
    from scipy import ndimage

    box = _bbox(cut)
    edt = ndimage.distance_transform_edt(np.pad(cut[box], 1))
    return max(0.0, float(edt[np.pad(taken[box], 1)].max()) - 0.5) * h


def taken_facts(config: Config, meshes: dict, masks: dict, iface_adds: dict, iface_union: dict, iface_cut: dict,
                origin: np.ndarray, h: float) -> list[dict]:
    """What each cut takes from each frozen region, before the run (CONTRACTS.md section 23, "What a cut takes").

    A cut is an interface cut step (a bore, the hole of a tube, a face, an STL with `op: cut`) or a keep-out region. A frozen
    region is a fixed, load or solid region or a union interface. The label KEEP_OUT wins over all of them, so the voxels that
    are in both are taken. One dict for each pair that has at least one such voxel: `cut`, `cut_kind` (bore, tube hole, face,
    stl, keep-out region), `region`, `region_role` (fixed, load, solid, interface), `voxels` (taken), `region_voxels` (the
    voxels of the region STL or of the union shape), `share`, `depth_mm`, `trim_mm`, `warn`. The depth is how far the region
    goes past the cut surface. A bore or a tube hole has no depth (`None`): it exists to remove material. For a face
    without `radius_mm` the depth is the largest signed distance of the STL vertices of the region past the plane through `point`
    (the stock is part of the allowed trim, so it is not taken off);
    for a bounded face it is that of the taken voxel centres; for an STL cut or a keep-out region it is the largest depth of the taken
    voxels inside the cut shape. `trim_mm` is `max(stock_mm + voxel_size, TRIM_MIN_MM)`; `warn` is `depth_mm > trim_mm`.
    A union shape is not a cut: the pair of a keep-out region and a union interface has the log warning of `voxelize` already, and is here too.
    """
    reg = config.regions
    frozen = []   # (role, name, mask, mesh)
    for i, p in enumerate(reg.fixed):
        frozen.append(("fixed", p.stem, masks[f"fixed[{i}] ({p.name})"], meshes[f"fixed[{i}] ({p.name})"]))
    for lr in reg.load:
        frozen.append(("load", lr.name, masks[f"load '{lr.name}'"], meshes[f"load '{lr.name}'"]))
    for i, p in enumerate(reg.solid):
        frozen.append(("solid", p.stem, masks[f"solid[{i}] ({p.name})"], meshes[f"solid[{i}] ({p.name})"]))
    for name, mk in iface_union.items():
        frozen.append(("interface", name, mk, iface_adds.get(name)))
    cuts = []   # (name, kind, spec, mask, own interface name)
    if config.interfaces:   # a run without interfaces never imports `topopt.interfaces`
        from . import interfaces
        for spec in config.interfaces:
            for op, part in interfaces.pieces(spec):
                key = interfaces.piece_key(spec, part)
                if op == "cut" and key in iface_cut:
                    cuts.append((key, "tube hole" if part == "hole" else spec.kind, spec, iface_cut[key], spec.name))
    for i, p in enumerate(reg.keep_out):
        cuts.append((p.stem, "keep-out region", None, masks[f"keep_out[{i}] ({p.name})"], None))
    boxes = [_bbox(mk) for _r, _n, mk, _m in frozen]
    out = []
    for cname, ckind, spec, cmask, own in cuts:
        for (role, rname, rmask, rmesh), box in zip(frozen, boxes):
            if box is None or rname == own and role == "interface":
                continue
            taken = cmask[box] & rmask[box]
            n = int(taken.sum())
            if n == 0:
                continue
            stock = 0.0 if spec is None else spec.stock_mm
            trim = max(stock + h, TRIM_MIN_MM)
            depth = None
            if ckind == "face":
                nrm = np.asarray(spec.axis, float)
                if spec.radius_mm is None and rmesh is not None:   # the STL vertices: the depth does not depend on the voxel size
                    pts = np.asarray(rmesh.vertices, float)
                else:
                    full = np.zeros_like(rmask)
                    full[box] = taken
                    pts = origin + h * (np.argwhere(full) + 0.5)
                depth = max(0.0, float(((pts - np.asarray(spec.point, float)) @ nrm).max()))   # from the plane through `point`: the stock is part of the allowed trim
            elif ckind in ("stl", "keep-out region"):
                full = np.zeros_like(rmask)
                full[box] = taken
                depth = _depth_into_cut(cmask, full, h)
            rv = int(rmask.sum())
            out.append({"cut": cname, "cut_kind": ckind, "region": rname, "region_role": role, "voxels": n,
                        "region_voxels": rv, "share": round(n / rv, 4),
                        "depth_mm": None if depth is None else round(depth, 2),
                        "trim_mm": None if depth is None else round(trim, 2),
                        "bounded": bool(ckind == "face" and spec.radius_mm is not None),
                        "warn": bool(depth is not None and depth > trim)})
    return out


def _region_text(f: dict) -> str:
    return f"{'interface shape' if f['region_role'] == 'interface' else f['region_role'] + ' region'} '{f['region']}'"


def taken_lines(taken: list | None) -> list[str]:
    """One fact line for each pair of `taken_facts`, for `topopt check` and the run log: [] when no cut takes a voxel."""
    out = []
    for f in taken or []:
        what = f"{f['voxels']} of {f['region_voxels']} voxels ({100 * f['share']:.1f} %) of {_region_text(f)}"
        if f["depth_mm"] is not None:
            what += f", up to {f['depth_mm']:g} mm {'past the plane' if f['cut_kind'] == 'face' else 'into the cut shape'}"
        else:
            what += " (a hole removes material)"
        who = f"keep-out region '{f['cut']}'" if f["cut_kind"] == "keep-out region" else f"{f['cut']} ({f['cut_kind']})"
        out.append(f"interface cut:  {who} takes {what}")
    return out


def taken_warnings(taken: list | None) -> list[str]:
    """One warning for each pair of `taken_facts` with `warn`: a cut that takes more of a frozen region than a machining trim.

    A warning starts with `interface '<name>' (` or `keep-out region '<name>' takes`. It gives the facts and the options.
    """
    out = []
    for f in taken or []:
        if not f["warn"]:
            continue
        share = f"{f['voxels']} of {f['region_voxels']} voxels ({100 * f['share']:.1f} %) of {_region_text(f)}"
        if f["cut_kind"] == "face":
            bounded = f["bounded"]
            where = (f"up to {f['depth_mm']:g} mm past the plane (the trim limit is {f['trim_mm']:g} mm). "
                     + ("A face with 'radius_mm' cuts the disc of that radius on the side of its normal. "
                        "Options: make 'radius_mm' smaller, move the plane, or accept the cut"
                        if bounded else
                        "A face cuts everything on the side of its normal. "
                        "Options: give the face a 'radius_mm' (a bounded cut), move the plane, or accept the cut"))
            out.append(f"interface '{f['cut']}' (face, cut) takes {share}, {where}")
        elif f["cut_kind"] == "stl":
            out.append(f"interface '{f['cut']}' (stl, cut) takes {share}, up to {f['depth_mm']:g} mm into the cut shape "
                       f"(the trim limit is {f['trim_mm']:g} mm). Options: move the cut shape or the region, or accept the cut")
        else:
            out.append(f"keep-out region '{f['cut']}' takes {share}, up to {f['depth_mm']:g} mm into the keep-out region "
                       f"(the trim limit is {f['trim_mm']:g} mm; a keep-out region wins over a fixed, load or solid region). "
                       "Options: move the keep-out region or the region, or accept the cut")
    return out


def _thickness_voxels(mask: np.ndarray) -> float:
    """About how many voxels across the thickest part of `mask` is (same rule as `topopt check`)."""
    if not mask.any():
        return 0.0
    from scipy import ndimage
    box = tuple(slice(int(ax.min()), int(ax.max()) + 1) for ax in np.nonzero(mask))
    return round(2.0 * float(ndimage.distance_transform_edt(np.pad(mask[box], 1)).max()) - 1.0, 1)


def solid_lines(s: dict | None) -> list[str]:
    """Text lines for `topopt check` and the run log from `facts["solid"]` (None: no solid regions)."""
    if not s:
        return []

    def vox(n, many, one):   # "1 voxel stays ..." and "2 voxels stay ..."
        return f"{n} voxel {one}" if n == 1 else f"{n} voxels {many}"

    parts = [vox(s["voxels"], "stay solid at density 1", "stays solid at density 1")
             + f" ({s['stl_voxels']} {'voxel' if s['stl_voxels'] == 1 else 'voxels'} in the solid STLs)"]
    for key, many, one in (("in_fixed", "lie in a fixed region (they are FIXED)", "lies in a fixed region (it is FIXED)"),
                           ("in_load", "lie in a load region (they keep the load)", "lies in a load region (it keeps the load)"),
                           ("in_keep_out", "lie in a keep-out region (they are removed)", "lies in a keep-out region (it is removed)"),
                           ("in_two_solid_stls", "lie in two solid STLs", "lies in two solid STLs")):
        if s[key]:
            parts.append(vox(s[key], many, one))
    if s["outside_design_stl"]:
        parts.append(vox(s["outside_design_stl"], "lie outside the design STL (they add material to the part)",
                         "lies outside the design STL (it adds material to the part)"))
    return ["solid regions: " + "; ".join(parts)]


# ---- region planes that are not on the voxel grid (CONTRACTS.md section 3, "Region planes off the voxel grid") ----
_AXES = "xyz"


def _plane_tol(position: float, h: float) -> float:
    """How far a plane may be from a voxel face (mm) and still count as on it.

    `symmetry.GRID_TOL` of a voxel, or 4 float32 steps at this coordinate, whichever is larger: an STL stores float32, so
    a plane that was on the grid in the CAD tool can be off by the float32 rounding. This is a floating-point tolerance,
    not a setting.
    """
    return max(symmetry.GRID_TOL * h, 4.0 * float(np.spacing(np.float32(abs(position)))))


def plane_facts(regions: list, origin: np.ndarray, h: float, changed=None) -> list[dict]:
    """The flat faces of each region on its bounding box whose plane is not on a voxel face of the grid.

    `regions` is a list of `(role, name, mesh, key)` (`key` is the label that `voxelize` gives the region; it is only passed to
    `changed`). A voxel belongs to a region when its centre is inside the STL
    (`inside_mask`), so the voxelized region ends at the voxel face that is nearest to the plane: the region is larger
    or smaller than the STL by the distance to that face (at most half a voxel). The ray of a column in x and y is
    shifted by `_JITTER` of a voxel; the rule uses the same shift, so it also gives the right side for a plane that is
    within that shift of a half-voxel tie. Only a flat face counts (a triangle with
    all three vertices on the bounding-box plane): a sphere or the side of a cylinder has no plane to shift.

    Returns one dict `{"role", "name", "planes": [...]}` for each region that has such a plane. A plane is
    `{"axis", "side" ("min" or "max"), "at_mm", "from_origin_voxels", "offset_mm", "offset_voxels", "effect" ("larger" or
    "smaller")}`. The offsets are positive numbers; `effect` says what they do to the voxelized region at that plane.
    With `changed(key, axis_index, side, effect)` (it returns a voxel count: `plane_change`) a plane also has `changed_voxels`
    and `changes_part` (`changed_voxels > 0`).
    """
    out = []
    for role, name, mesh, key in regions:
        faces = np.asarray(mesh.faces)
        planes = []
        for a, ax in enumerate(_AXES):
            coord = np.asarray(mesh.vertices, dtype=np.float64)[:, a][faces]      # (n, 3): this coordinate of the 3 vertices of each triangle
            for side, p in (("min", float(mesh.bounds[0][a])), ("max", float(mesh.bounds[1][a]))):
                tol = _plane_tol(p, h)
                if not np.any(np.all(np.abs(coord - p) <= tol, axis=1)):
                    continue                                    # no flat face on this plane
                u = (p - float(origin[a])) / h                  # voxels from the grid origin
                jit = _JITTER[a] if a < 2 else 0.0              # the ray of a column is shifted by this part of a voxel (x, y)
                d = math.ceil(u - 0.5 - jit) - u                # to the voxel face that the voxel centres decide on
                if abs(d) * h <= tol:
                    continue                                    # on a voxel face
                larger = d < 0 if side == "min" else d > 0
                plane = {"axis": ax, "side": side, "at_mm": round(p, 4), "from_origin_voxels": round(u, 4),
                         "offset_mm": round(abs(d) * h, 4), "offset_voxels": round(abs(d), 4),
                         "effect": "larger" if larger else "smaller"}
                if changed is not None:
                    n = int(changed(key, a, side, plane["effect"]))
                    plane["changed_voxels"], plane["changes_part"] = n, n > 0
                planes.append(plane)
        if planes:
            out.append({"role": role, "name": name, "planes": planes})
    return out


def plane_change(reg, masks: dict, iface_union: dict, iface_cut: dict, labels: np.ndarray, load_ids: np.ndarray,
                 key: str, axis: int, side: str, effect: str) -> int:
    """How many voxels of the part change if the plane of the region `key` were on the voxel face: the voxel layer that the offset adds or removes.

    The region ends at its extreme voxel layer on `axis` (the first one for a min plane, the last one for a max plane).
    A plane that makes the region "larger" has added that layer; one that makes it "smaller" has removed the next layer
    (the voxels with the same footprint, beyond it). The function composes the labels of that one layer twice with
    `compose_labels`: as they are, and without the added layer or with the removed one. It counts the voxels where the two
    differ in what the part is: DESIGN, FIXED, LOAD (and the load region of a LOAD voxel); KEEP_OUT and empty are the same
    (neither is part of the active voxels). A layer under a region of higher priority, or outside the part, gives 0: the plane
    does not change the part. This adds no number: it counts voxels.
    """
    mask = masks[key]
    others = tuple(i for i in range(3) if i != axis)
    used = np.nonzero(mask.any(axis=others))[0]
    edge = int(used.min() if side == "min" else used.max())                 # the extreme layer of the region
    foot = np.take(mask, edge, axis=axis)                                   # its footprint (2D)
    step = -1 if side == "min" else 1
    layer = edge if effect == "larger" else edge + step                     # the layer that the offset adds (larger) or removes (smaller)
    if not 0 <= layer < mask.shape[axis]:
        return int(foot.sum())                                              # beyond the grid: the part would reach there
    sl = tuple(slice(layer, layer + 1) if i == axis else slice(None) for i in range(3))
    sub = {k: m[sl] for k, m in masks.items()}
    sub[key] = (np.zeros_like(sub[key]) if effect == "larger" else np.expand_dims(foot, axis))
    lab2, ids2 = compose_labels(reg, sub, {k: m[sl] for k, m in iface_union.items()}, {k: m[sl] for k, m in iface_cut.items()})

    def part(lab, ids):      # what the part is in each voxel: the active labels and the load region; KEEP_OUT is empty
        active = (lab >= DESIGN) & (lab <= LOAD)
        return np.where(active, lab.astype(np.int32) * 1000 + ids, 0)
    return int((part(labels[sl], load_ids[sl]) != part(lab2, ids2)).sum())


_ROLE_WORDS = {"design": "design", "fixed": "fixed", "keep_out": "keep-out", "load": "load", "solid": "solid"}


def _mm(x: float) -> str:
    return f"{x:.2f}" if x >= 0.005 else f"{x:.2g}"


def plane_lines(planes: list | None, h: float, origin) -> list[str]:
    """The one text line for the planes of `plane_facts` that change the part: [] when none does.

    A plane with `changes_part` False (no voxel of the part is in the layer that the offset adds or removes) is left out of the
    line; it stays in the facts. A fact without `changes_part` counts as one that changes the part. The line names the voxel size
    and the grid origin once, gives one short item for each plane, the options once, and where the rest of the facts are.
    """
    items, total = [], 0
    for r in planes or []:
        for q in r["planes"]:
            total += 1
            if not q.get("changes_part", True):
                continue
            who = _ROLE_WORDS[r["role"]] + ("" if r["role"] == "design" else f" '{r['name']}'")
            items.append(f"{who} {q['axis']} {q['side']} {q['at_mm']:g} mm: {_mm(q['offset_mm'])} mm {q['effect']}")
    if not items:
        return []
    o = ", ".join(f"{float(origin[i]):g}" for i in range(3))
    return [f"region planes off the voxel grid: {len(items)} of {total} change the part (voxel {h:g} mm; grid origin ({o}) mm, "
            "the min corner of all regions, so it moves when such a region moves). Each item is the region, the plane, and how "
            "much larger or smaller than the STL the region is there: " + "; ".join(items) + ". A voxel is in a region when its "
            "centre is in the STL, so each plane moves to the nearest voxel face. Options: move or resize the region so that "
            "the plane is on a voxel face. Or choose a voxel size that divides the distance from the grid origin. Or extend a "
            "keep-out region past the part. All planes, with the offsets in voxels: `check.region_planes`"]
