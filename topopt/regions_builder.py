"""Build region STLs and a topopt problem YAML from a short spec (a dict).

Used by the MCP tool `build_regions` and by the skill script
`.claude/skills/part-to-topopt/build_regions.py` (same CLI):
    python build_regions.py SPEC.yaml --out DIR [--check] [--voxel H]

Units: mm, N, MPa. See the part-to-topopt SKILL.md for the spec format. The kind `stl` takes the user's own STL as a solid, with holes plugged on request
(CONTRACTS.md section 45.1) and a frozen ring round each screw hole (section 45.2; the kind `screw_ring` is that ring as a fixed or load region).
Every `settings` key of the spec that the run YAML takes goes to the YAML as given, except the keys that the builder writes itself
(`BUILDER_OWNED_SETTINGS`); the same holds for the top-level keys (`BUILDER_OWNED_TOP_LEVEL`, `limits`, `buckling`, ... are copied)
and the `material` keys. A key that the YAML does not take stops the build (`check_spec_keys`).
Exit code of the CLI: 0 = ok, 1 = spec or check error.

Only `build_in_workspace` (the MCP tool) reads the preferences file of the work folder: it fills a `material` (this includes
`material.filament`), `print` (this includes `print.printer_model`) or `settings.voxel_size` that the spec does not have at all, and says so
(`topopt.preferences.fill_spec`). The CLI does not read it.
"""
from __future__ import annotations

import argparse
import math
import re
import reprlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import yaml
from manifold3d import CrossSection, Manifold, Mesh, OpType

from .features import MM_DIGITS, UNIT_DIGITS

from .region_rules import MIN_REGION_THICK, MIN_REGION_VOXELS  # noqa: F401  (one place for the two numbers, with cli.py)

SEG = 40                    # circle segments
ROLES = ("design", "fixed", "load", "solid", "keep_out")
# Time model (CONTRACTS 11): the same numbers as topopt.cli.estimate_seconds.
REF_DOFS, REF_S, DOF_POWER, SETUP_S, POST_S = 109_548, 1.30, 2.4, 10.0, 15.0
GOAL_S = 120.0              # the aim of the auto voxel picker: the estimate scaled by the fill (`time_factor`) is about this; not a limit
CAP_ACTIVE = 41000          # default aim of the auto voxel picker; settings.target_active or a voxel overrides it
# Measured 2026-10-11: compact hook (fill 0.36) ran 1.0 x the estimate. Slender ladle
# (fill 0.06 to 0.09) ran 0.27 x the estimate. "fill" = active voxels / grid voxels.
FILL_SLENDER, FILL_COMPACT, F_SLENDER = 0.085, 0.30, 0.27
MAX_FROZEN = 0.60           # forced-solid voxels / active voxels: a warning level


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")   # a folder or file stem: no separators, not '.' or '..'


class SpecError(Exception):
    pass


# ---- primitives -------------------------------------------------------------
def _vec(v, n=3, what="value"):
    try:
        a = [float(x) for x in v]
    except Exception:
        raise SpecError(f"{what} must be a list of numbers, got {v!r}")
    if len(a) != n:
        raise SpecError(f"{what} needs {n} numbers, got {v!r}")
    return a


def _rot_z_to(d):
    """3x3 rotation matrix that turns +z into the direction d."""
    d = np.asarray(d, float)
    n = np.linalg.norm(d)
    if n == 0:
        raise SpecError("direction has zero length")
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _place(m: Manifold, R, t) -> Manifold:
    mat = np.hstack([R, np.asarray(t, float).reshape(3, 1)])
    return m.transform(mat)


def _cyl(p0, p1, r0, r1=None) -> Manifold:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L <= 0 or r0 <= 0:
        raise SpecError("cylinder needs p0 != p1 and radius > 0")
    m = Manifold.cylinder(L, r0, r0 if r1 is None else r1, SEG)
    return _place(m, _rot_z_to(p1 - p0), p0)


def _sphere(c, r) -> Manifold:
    if r <= 0:
        raise SpecError("radius must be > 0")
    return Manifold.sphere(r, SEG).translate(c)


def make_primitive(kind: str, p: dict, ctx: "BuildContext | None" = None, at: str | None = None) -> Manifold:
    """The solid of one spec item. `ctx` (the base folder of a relative STL path, the report lines) and `at` (the key path of the item) are for the `stl` kind."""
    if kind == "stl":
        return make_stl(p, ctx or BuildContext(), f"{at or 'regions'}.stl")
    if kind == "screw_ring":
        return make_ring_item(p, ctx or BuildContext(), f"{at or 'regions'}.screw_ring")
    if not isinstance(p, dict):
        raise SpecError(f"{kind}: parameters must be a mapping")
    if kind == "box":
        if "min" in p and "max" in p:
            lo, hi = _vec(p["min"], what="box.min"), _vec(p["max"], what="box.max")
        elif "center" in p and "size" in p:
            c, s = _vec(p["center"], what="box.center"), _vec(p["size"], what="box.size")
            lo = [c[i] - s[i] / 2 for i in range(3)]
            hi = [c[i] + s[i] / 2 for i in range(3)]
        else:
            raise SpecError("box needs min+max or center+size")
        size = [hi[i] - lo[i] for i in range(3)]
        if min(size) <= 0:
            raise SpecError(f"box has zero or negative size: {size}")
        return Manifold.cube(size).translate(lo)
    if kind == "cylinder":
        return _cyl(_vec(p["p0"], what="cylinder.p0"), _vec(p["p1"], what="cylinder.p1"),
                    float(p["radius"]))
    if kind == "sphere":
        return _sphere(_vec(p["center"], what="sphere.center"), float(p["radius"]))
    if kind == "capsule":
        p0, p1, r = _vec(p["p0"]), _vec(p["p1"]), float(p["radius"])
        return _cyl(p0, p1, r) + _sphere(p0, r) + _sphere(p1, r)
    if kind == "bowl":
        # Hollow hemisphere. "open" is the direction the bowl opens to. Default +z.
        c, r, w = _vec(p["center"], what="bowl.center"), float(p["radius"]), float(p["wall"])
        if w <= 0 or w >= r:
            raise SpecError("bowl.wall must be > 0 and < radius")
        shell = Manifold.sphere(r, SEG * 2) - Manifold.sphere(r - w, SEG * 2)
        half = Manifold.cube([4 * r, 4 * r, 2 * r]).translate([-2 * r, -2 * r, -2 * r])
        shell = shell ^ half                       # keep the -z half: the cup opens toward +z
        return _place(shell, _rot_z_to(_vec(p.get("open", [0, 0, 1]), what="bowl.open")), c)
    if kind == "prism":
        # Polygon [[a,b],...] in a plane, extruded along the third axis.
        plane = p.get("plane", "xz")
        ext = _vec(p["extent"], 2, "prism.extent")      # [min, max] along the normal axis
        pts = [_vec(q, 2, "prism.points item") for q in p["points"]]
        if len(pts) < 3 or ext[1] <= ext[0]:
            raise SpecError("prism needs 3+ points and extent [min, max] with max > min")
        sol = CrossSection([pts]).extrude(ext[1] - ext[0]).translate([0, 0, ext[0]])
        # Local (u, v, w) -> world axes
        perm = {"xy": [0, 1, 2], "xz": [0, 2, 1], "yz": [1, 2, 0]}.get(plane) if isinstance(plane, str) else None
        if perm is None:
            raise SpecError("prism.plane must be xy, xz or yz")
        R = np.zeros((3, 3))
        for local, world in enumerate(perm):
            R[world, local] = 1.0
        if np.linalg.det(R) < 0:                         # keep the winding outward
            R[:, 1] *= -1
            sol = CrossSection([[(a, -b) for a, b in pts][::-1]]).extrude(
                ext[1] - ext[0]).translate([0, 0, ext[0]])
        return _place(sol, R, [0, 0, 0])
    raise SpecError(f"unknown primitive '{kind}'. Use box, cylinder, sphere, capsule, bowl, prism, stl, screw_ring")


def build_shape(items, where: str, ctx: "BuildContext | None" = None, at: str | None = None) -> Manifold:
    """items: list of {kind: params} (add) or {subtract: {kind: params}}.

    `ctx` and `at` are for the `stl` kind: `at` is the key path of `items` in the spec (`regions.design`); an error of an `stl` item names it.
    """
    single = isinstance(items, dict)
    if single:
        items = [items]
    if not isinstance(items, list) or not items:
        raise SpecError(f"{where}: needs a list of primitives")
    ctx = ctx or BuildContext()
    result = None
    for i, it in enumerate(items):
        if not isinstance(it, dict) or len(it) != 1:
            raise SpecError(f"{where}[{i}]: each item must have exactly one key, got {it!r}")
        (key, val), = it.items()
        here = (at or where) if single else f"{at or where}[{i}]"
        try:
            if key == "subtract":
                if not isinstance(val, dict) or len(val) != 1:
                    raise SpecError("subtract needs one primitive")
                (k2, v2), = val.items()
                if result is None:
                    raise SpecError("subtract cannot be the first item")
                ctx.in_subtract = True      # an `stl` item with screw holes, and a ring item, are not for a cutter
                try:
                    result = result - make_primitive(k2, v2, ctx, f"{here}.subtract")
                finally:
                    ctx.in_subtract = False
            else:
                m = make_primitive(key, val, ctx, here)
                result = m if result is None else result + m
        except KeyError as e:
            raise SpecError(f"{where}[{i}] ({key}): missing key {e}")
        except _KeyedError:
            raise                                   # the text names the key path of the item already
        except SpecError as e:
            raise SpecError(f"{where}[{i}] ({key}): {e}")
    if result.is_empty() or result.volume() <= 0:
        raise SpecError(f"{where}: the shape is empty")
    return result


def to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=True)


# ---- the `stl` kind: a region from the user's own STL (CONTRACTS.md section 45.1) -----------------------------------------
# This is NOT the `stl` kind of the YAML block `interfaces` (section 23): that one unites or cuts a finished solid after the run. Here the file is
# a solid in the region spec, and `build_regions` writes its own region STL from it.
STL_KEYS = ("path", "plug_holes", "screw_holes", "ring_mm")
STL_EXAMPLE = "{stl: {path: part.stl, plug_holes: [H1], screw_holes: [H2]}}"
RING_KEYS = ("path", "hole", "ring_mm")
RING_EXAMPLE = "{screw_ring: {path: part.stl, hole: H2}}"
RING_ROLES = ("fixed", "load", "solid")       # a frozen shape: the ring item is not a design space and not a cutter
# The frozen ring round a screw hole (CONTRACTS.md section 45.2). The thickness t of the ring is a
# method setting with a default rule, and the spec key `ring_mm` sets it for one hole or for all holes of an item. The report gives t for each hole.
# Default rule: t = RING_PER_DIAMETER * d, with d the diameter of the hole (`find_features`), and at least RING_MIN_VOXELS voxel sizes of the written
# YAML, so that the optimizer sees the ring as frozen voxels. The two numbers are a proposal that was measured on a plate, not a physical limit.
RING_PER_DIAMETER = 0.5
RING_MIN_VOXELS = 2
# A wall vertex of a hole lies on the cylinder of the `find_features` reply up to the rounding of that reply: the `point` is rounded to MM_DIGITS (a
# shift of at most 0.87e-4 mm across the axis), the `radius_mm` too (0.5e-4 mm), and a component of the unit `axis` to UNIT_DIGITS (an angle error
# of at most 0.87e-6 rad, times the distance along the axis). The tolerance of the wall test is the fit residual of the hole plus the two bounds
# below, so it has no number of its own. Measured effect: all wall vertices are found from this tolerance up to 500 times more;
# at a quarter of it a rotated part at 300 mm loses a vertex, at a tenth 30 percent of them.
WALL_ROUND_MM = 2 * 10 ** -MM_DIGITS         # the first two bounds added and rounded up (1.4e-4 mm to 2e-4 mm)
WALL_AXIS_PER_MM = 10 ** -UNIT_DIGITS        # the third bound, for each mm along the axis


class _KeyedError(SpecError):
    """A `SpecError` whose text starts with the key path of the item already: `build_shape` adds no prefix."""


class BuildContext:
    """What a build needs besides the spec: the folder where a relative STL path starts, the lines for the report, and the screw holes.

    `voxel` is the voxel size of the YAML that will be written (None while it is not known): a ring is at least `RING_MIN_VOXELS` voxel sizes thick.
    `cache` holds the `find_features` reply of each file, so that one build reads a file once. `role` and `region` say where the item is.
    """

    def __init__(self, base=None, notes: list | None = None, base_words: str | None = None, voxel: float | None = None, cache: dict | None = None):
        self.base = None if base is None else Path(base)
        self.notes = [] if notes is None else notes
        self.base_words = base_words or ("the folder of the spec file" if base is not None else "the current folder")
        self.voxel = voxel
        self.cache: dict = {} if cache is None else cache
        self.role: str | None = None
        self.region: str | None = None
        self.in_subtract = False
        self.screws: list[dict] = []        # one record for each screw hole of an `stl` item (the tube interface of the YAML)
        self.rings: list[dict] = []         # one record for each `screw_ring` item (a region item)

    def again(self, voxel: float) -> "BuildContext":
        """A new context for a second build at the voxel size `voxel`: same folder and cache, empty lines and records."""
        return BuildContext(self.base, None, self.base_words, voxel, self.cache)


def _ids_ok(v) -> bool:
    return v == "all" or (isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v))


def _ring_number_ok(v) -> bool:
    return _number_like(v) and float(v) > 0


RING_MM_EXPECT = "a number in mm above 0 (all screw holes of the item), or a mapping hole id -> number, for example {H1: 2.5}"


def _check_stl_item(p, kp: str) -> None:
    """The shape of an `stl` item (the keys and the types). `kp` is the key path of the item, ending in `.stl`."""
    if not isinstance(p, dict):
        raise _shape(kp, f"a mapping with a path, for example {STL_EXAMPLE}", p)
    for k in p:
        if k not in STL_KEYS:
            raise _KeyedError(f"{kp}.{k} is not a key of an stl item. Keys: {', '.join(STL_KEYS)}. Example: {STL_EXAMPLE}")
    if "path" not in p:
        raise _KeyedError(f"{kp}.path is missing. Example: {STL_EXAMPLE}")
    if not isinstance(p["path"], str) or not p["path"].strip():
        raise _shape(f"{kp}.path", "a file name (text), for example part.stl", p["path"])
    for key in ("plug_holes", "screw_holes"):
        if p.get(key) is not None and not _ids_ok(p[key]):
            raise _shape(f"{kp}.{key}", "a list of hole ids from find_features, for example [H1, H2], or all", p[key])
    check_ring_mm(p, kp)


def check_ring_mm(p: dict, kp: str) -> None:
    """The shape of `ring_mm` of an `stl` item: a number above 0, or a mapping hole id -> number that names holes of `screw_holes`."""
    rm = p.get("ring_mm")
    if rm is None:
        return
    shown = p.get("screw_holes")
    if shown is None or shown == []:
        raise _KeyedError(f"{kp}.ring_mm is set, but the item has no screw_holes. ring_mm is the thickness of the ring round a screw hole")
    if isinstance(rm, dict):
        if not rm:
            raise _shape(f"{kp}.ring_mm", RING_MM_EXPECT, rm)
        for k, val in rm.items():
            if not isinstance(k, str) or not k.strip():
                raise _shape(f"{kp}.ring_mm", RING_MM_EXPECT, rm)
            if not _ring_number_ok(val):
                raise _shape(f"{kp}.ring_mm.{k}", "a number in mm above 0, for example 2.5", val)
            if isinstance(shown, list) and k not in shown:
                raise _KeyedError(f"{kp}.ring_mm.{k}: {k} is not in screw_holes ({', '.join(shown)})")
    elif not _ring_number_ok(rm):
        raise _shape(f"{kp}.ring_mm", RING_MM_EXPECT, rm)


def _check_ring_item(p, kp: str) -> None:
    """The shape of a `screw_ring` item. `kp` is the key path of the item, ending in `.screw_ring`."""
    if not isinstance(p, dict):
        raise _shape(kp, f"a mapping with a path and a hole, for example {RING_EXAMPLE}", p)
    for k in p:
        if k not in RING_KEYS:
            raise _KeyedError(f"{kp}.{k} is not a key of a screw_ring item. Keys: {', '.join(RING_KEYS)}. Example: {RING_EXAMPLE}")
    for need in ("path", "hole"):
        if need not in p:
            raise _KeyedError(f"{kp}.{need} is missing. Example: {RING_EXAMPLE}")
    if not isinstance(p["path"], str) or not p["path"].strip():
        raise _shape(f"{kp}.path", "a file name (text), for example part.stl", p["path"])
    if not isinstance(p["hole"], str) or not p["hole"].strip():
        raise _shape(f"{kp}.hole", "one hole id from find_features, for example H2", p["hole"])
    if p.get("ring_mm") is not None and not _ring_number_ok(p["ring_mm"]):
        raise _shape(f"{kp}.ring_mm", "a number in mm above 0, for example 2.5", p["ring_mm"])


def _line(text: str) -> str:
    return " ".join(str(text).split())


def _wall_vertices(V: np.ndarray, tree, h: dict, tol: float | None = None):
    """`(index, tolerance)`: the vertices of the mesh on the wall of the hole `h` of `find_features`, between its two ends.

    The default tolerance is the one of the comment at `WALL_ROUND_MM`. The check after the plug gives a looser one.
    """
    axis, p0 = np.asarray(h["axis"], float), np.asarray(h["point"], float)
    length, radius = float(h["length_mm"]), float(h["radius_mm"])
    if tol is None:
        tol = float(h["fit"]["max_mm"]) + WALL_ROUND_MM + WALL_AXIS_PER_MM * length
    near = np.asarray(tree.query_ball_point(p0 + axis * (length / 2.0), math.hypot(radius + tol, length / 2.0 + tol)), dtype=np.int64)
    rel = V[near] - p0
    s = rel @ axis
    d = np.linalg.norm(rel - s[:, None] * axis, axis=1)
    return near[(np.abs(d - radius) <= tol) & (s >= -tol) & (s <= length + tol)], tol


def _holes_words(holes: list[dict]) -> str:
    """The holes in words, one group for each size: `H1 (d 4 mm, 4 mm long, through)`, or `H3, H4, H5 (d 5 mm, 4 mm long, through)` for equal holes."""
    groups: dict[str, list[str]] = {}
    for h in holes:
        groups.setdefault(f"d {h['diameter_mm']:g} mm, {h['length_mm']:g} mm long, {h['depth']}", []).append(h["id"])
    return ", ".join(f"{', '.join(ids)} ({size})" for size, ids in groups.items())


def _found(ctx: BuildContext, file: Path, where: str) -> dict:
    """The `find_features` reply of `file`, read once for one build. `where` is the key path for an error."""
    from . import features
    from .config import UserError
    if file not in ctx.cache:
        try:
            ctx.cache[file] = features.find_features(file)
        except UserError as e:
            raise _KeyedError(f"{where}: {file.name}: find_features failed: {_line(e)}")
    return ctx.cache[file]


def _plug_holes(m: Manifold, V: np.ndarray, F: np.ndarray, health: dict, found: dict, file: Path, wanted, kp: str, say: list[str]):
    """Unite `m` with the plug of each wanted hole. Returns `(manifold, plugged holes, open holes)`.

    The plug of a hole is the convex hull of its wall vertices: the vertices of the mesh on the cylinder of the hole, between its two ends. A wall
    that is a regular polygon of n sides gives the prism of that polygon, which is the void of the hole to the last point. The plug has the same
    vertices as the rim of the hole, so its ends lie in the faces of the part (no material outside the part) and its sides lie on the wall (no skin
    of void at the facets). The cone of a drill point and a chamfer at the mouth are not part of the wall: they stay.
    """
    from scipy.spatial import cKDTree

    holes = {h["id"]: h for h in found["holes"]}
    known = _holes_words(found["holes"]) or "none (find_features lists no hole in this mesh)"
    ids = list(holes) if wanted == "all" else list(dict.fromkeys(wanted))
    for i in ids:
        if i not in holes:
            raise _KeyedError(f"{kp}.plug_holes: {i!r} is not a hole of {file.name}. Known ids: {known}")
    tree = cKDTree(V)
    plugs = []
    for i in ids:
        h = holes[i]
        pick, _tol = _wall_vertices(V, tree, h)
        if len(pick) < 4:
            raise _KeyedError(f"{kp}.plug_holes: {i}: only {len(pick)} wall vertices are in the mesh at the place of the hole, and a plug needs 4")
        plug = Manifold.hull_points(V[pick])
        if plug.is_empty() or plug.volume() <= 0:
            raise _KeyedError(f"{kp}.plug_holes: {i}: the wall vertices of the hole do not make a solid")
        plugs.append(plug)
    if plugs:
        m = Manifold.batch_boolean([m] + plugs, OpType.Add)
        if m.status().name != "NoError" or m.is_empty():
            raise _KeyedError(f"{kp}.plug_holes: manifold3d could not unite the plugs with the part ({m.status().name})")
        left = _check_plugs(m, health, [holes[i] for i in ids], say, kp)
        if not to_trimesh(m).is_watertight:     # the test of `write_outputs`, here with the key path: a rough rim leaves float32 slivers that pinch the surface
            stay = [i for i, n in zip(ids, left) if n]
            raise _KeyedError(f"{kp}.plug_holes: the plugged mesh is not a closed solid" + (f" ({', '.join(stay)}: wall faces stay in the part)" if stay else "")
                              + ". The wall or the rim of a hole is not an exact cylinder with flat ends in this mesh (noise or a curved rim, as in an optimizer result), "
                              "so the plug cannot meet it. Plug the other holes only, or use a mesh with exact holes")
    for i in ids:
        h = holes[i]
        if not h["closed_ring"]:
            say.append(f"note: {kp}.plug_holes: {i} is an arc of {h['arc_deg']:g} deg, not a closed ring (a notch). The plug fills the circle up to the chord of the arc.")
        if h["depth"] == "closed":
            say.append(f"note: {kp}.plug_holes: {i} is a cavity inside the part (both ends closed). The plug fills it.")
    return m, [holes[i] for i in ids], [h for k, h in holes.items() if k not in ids]


def _check_plugs(m: Manifold, health: dict, holes: list[dict], say: list[str], kp: str) -> np.ndarray:
    """Facts about the part after the plugs: a `WARNING:` line when a wall face stays in the part, or an inner void appears (for example the cone of a drill point).

    Returns the number of wall faces that stay, for each hole."""
    from scipy.spatial import cKDTree

    from . import features
    from .printcheck import mesh_health
    mesh = m.to_mesh()
    Vr, Fr = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), np.asarray(mesh.tri_verts).astype(np.int64)
    tree_r = cKDTree(Vr)
    tag = np.full(len(Vr), -1, dtype=np.int64)
    for k, h in enumerate(holes):
        # The tolerance of the cylinder test of `find_features` (`FIT_RESIDUAL_FRAC` of the radius), looser than the tolerance of the plug: a wall vertex
        # that the plug missed is found here, and the wall face that it holds stays in the part.
        pick, _ = _wall_vertices(Vr, tree_r, h, features.FIT_RESIDUAL_FRAC * float(h["radius_mm"]))
        tag[pick] = k
    t0, t1, t2 = tag[Fr[:, 0]], tag[Fr[:, 1]], tag[Fr[:, 2]]
    on_wall = np.flatnonzero((t0 >= 0) & (t0 == t1) & (t1 == t2))
    sin_lim = math.sin(math.radians(features.AXIS_NORMAL_DEG))
    left = np.zeros(len(holes), dtype=np.int64)
    if len(on_wall):
        P = Vr[Fr[on_wall]]
        nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
        axes = np.asarray([h["axis"] for h in holes], float)[t0[on_wall]]
        side = np.abs(np.einsum("ij,ij->i", nrm, axes)) <= sin_lim          # a wall face is across the axis; the end caps are not
        left = np.bincount(t0[on_wall][side], minlength=len(holes))
    for k, h in enumerate(holes):
        if left[k]:
            say.append(f"WARNING: {kp}.plug_holes: {h['id']}: {int(left[k])} faces of its wall stay in the part, so the hole is not filled to the wall. Check the mesh near the hole.")
    after = mesh_health(Vr, Fr)
    if len(after["voids"]) > len(health["voids"]):
        new = sorted(after["voids"], key=lambda v: -v["volume_mm3"])[:3]
        say.append(f"WARNING: {kp}.plug_holes: after the plugs the part holds {len(after['voids'])} inner void(s) (before: {len(health['voids'])}), the largest "
                   + ", ".join(f"{v['volume_mm3']:g} mm3" for v in new)
                   + ". The plug fills the cylinder wall only: the cone of a drill point or a space behind a closed end stays.")
    return left


def make_stl(p: dict, ctx: BuildContext, kp: str) -> Manifold:
    """The solid of an `stl` item: the closed mesh of the file, with the wanted holes plugged. `kp` is the key path of the item (`regions.design[0].stl`)."""
    from .config import UserError
    from .printcheck import load_stl_float32, mesh_health
    _check_stl_item(p, kp)
    screws = p.get("screw_holes")
    if screws not in (None, []) and (ctx.role == "keep_out" or ctx.in_subtract):
        raise _KeyedError(f"{kp}.screw_holes: a screw ring is a frozen shape of the part. This item is {'a cutter inside subtract' if ctx.in_subtract else 'in keep_out'}, "
                          "where the part is removed. Put the part in design, fixed, load or solid")
    name = p["path"]
    file = _stl_file(name, ctx, f"{kp}.path")

    def fail(cause) -> _KeyedError:
        return _KeyedError(f"{kp}.path: {name}: {_line(cause)}")

    try:
        V, F = load_stl_float32(file)
        health = mesh_health(V, F)
    except UserError as e:
        raise fail(e)
    if not health["watertight"]:
        raise fail(f"the mesh is not a closed solid ({health['open_edges']} open edges, {health['nonmanifold_edges']} edges with more than 2 faces). "
                   "The builder does not repair a mesh")
    if health["winding_conflicts"]:
        raise fail(f"the face winding is not consistent ({health['winding_conflicts']} edges where neighbour faces run the same way round). "
                   "The builder does not repair a mesh")
    if health["inside_out"] or health["inverted_shells"]:
        raise fail("the faces point inward (the volume is negative). The builder does not repair a mesh")
    m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))
    if m.status().name != "NoError":
        raise fail(f"manifold3d refuses the mesh ({m.status().name}): it is not a valid solid")
    if m.is_empty() or m.volume() <= 0:
        raise fail("the mesh has no volume")
    size = V.max(axis=0) - V.min(axis=0)
    text = (f"stl {kp[:-4]}: {name}, {len(F)} faces, closed solid, size {size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g} mm, "
            f"volume {m.volume():.1f} mm3")
    if len(health["bodies"]) > 1:
        text += f", {len(health['bodies'])} bodies"
    extra: list[str] = []
    wanted = p.get("plug_holes")
    plugged: list[dict] = []
    if wanted is not None and wanted != []:
        m, plugged, kept = _plug_holes(m, V, F, health, _found(ctx, file, f"{kp}.plug_holes"), file, wanted, kp, extra)
        text += "; plugged " + (_holes_words(plugged) or "none")
        text += "; open " + (", ".join(h["id"] for h in kept) or "none")
    if screws not in (None, []):
        _add_screws(ctx, kp, name, file, _found(ctx, file, f"{kp}.screw_holes"), m, screws, p.get("ring_mm"), {h["id"] for h in plugged}, extra)
    ctx.notes.append(text)
    ctx.notes.extend(extra)
    return m


def _stl_file(name: str, ctx: BuildContext, kp: str) -> Path:
    """The file of a `path` key (`kp` is the key path of the `path`): absolute, or relative to the folder of the build. A folder or a missing file is a `SpecError`."""
    file = Path(name).expanduser()
    if not file.is_absolute():
        file = (ctx.base or Path.cwd()) / file
    if file.is_dir():
        raise _KeyedError(f"{kp}: {name}: this is a folder. Give an STL file")
    if not file.is_file():
        raise _KeyedError(f"{kp}: {name}: file not found (looked for {file}; a relative path starts at {ctx.base_words})")
    return file


# ---- screw rings (CONTRACTS.md section 45.2) ---------------------------------------------------------------------------------
def ring_thickness(h: dict, ring_mm, voxel) -> tuple[float, str]:
    """`(t, why)`: the thickness of the ring round the hole `h` of `find_features`, and the reason for the value.

    `ring_mm` (a number, or None) is the value of the spec: it is used as it is. Else t = `RING_PER_DIAMETER` * d, raised to `RING_MIN_VOXELS` voxel
    sizes when the voxel size of the written YAML is known (`voxel`, None while it is not).
    """
    if ring_mm is not None:
        return float(ring_mm), "ring_mm from the spec"
    t = RING_PER_DIAMETER * float(h["diameter_mm"])
    if voxel is not None and t < RING_MIN_VOXELS * voxel:
        return RING_MIN_VOXELS * voxel, f"raised to {RING_MIN_VOXELS} voxels"
    return t, f"default {RING_PER_DIAMETER:g} x d"


def _ring_radii(h: dict, t: float) -> tuple[float, float]:
    """`(inner, outer)` radius of the ring: the radius of the hole, and that plus t (to 1e-6 mm, so that the YAML has no float noise)."""
    r = float(h["radius_mm"])
    return r, round(r + t, 6)


def _ring_refusal(h: dict) -> str | None:
    """Why the hole `h` of `find_features` cannot take a ring (None if it can)."""
    if not h["closed_ring"]:
        return f"{h['id']} is an arc of {h['arc_deg']:g} deg, not a closed ring (a notch): a ring needs the whole circle"
    if h["depth"] == "closed":
        return f"{h['id']} is a cavity inside the part (both ends closed): no screw goes in"
    if h["depth"] == "unknown":
        return f"{h['id']} has an end that find_features could not judge (the mesh is open there), so the ring has no known direction"
    return None


def _ring_shape(h: dict, t: float, name: str = "ring") -> Manifold:
    """The ring of the hole `h` with thickness t: the shape of the `tube` interface of section 23 (the same polygons), with no stock."""
    from . import interfaces
    from .config import Interface
    r_in, r_out = _ring_radii(h, t)
    spec = Interface(name, "tube", "union", 0.0, tuple(h["point"]), tuple(h["axis"]), r_out, r_in, float(h["length_mm"]))
    return interfaces.build_shape(spec, 0.0, None, part="ring")


def _add_screws(ctx: BuildContext, kp: str, name: str, file: Path, found: dict, item: Manifold, wanted, ring_mm, plugged: set, say: list[str]) -> None:
    """Record the screw holes of an `stl` item. The tube interfaces and the report lines are made when the voxel size is known (`screw_rings`)."""
    holes = {h["id"]: h for h in found["holes"]}
    known = _holes_words(found["holes"]) or "none (find_features lists no hole in this mesh)"
    ids = list(holes) if wanted == "all" else list(dict.fromkeys(wanted))
    for i in ids:
        if i not in holes:
            raise _KeyedError(f"{kp}.screw_holes: {i!r} is not a hole of {file.name}. Known ids: {known}")
    both = [i for i in ids if i in plugged]
    if both:
        raise _KeyedError(f"{kp}.screw_holes: {', '.join(both)} is also in plug_holes. A hole is filled or it takes a screw. Name the screw holes in a list")
    for i in ids:
        why = _ring_refusal(holes[i])
        if why:
            raise _KeyedError(f"{kp}.screw_holes: {why}. Known ids: {known}")
    for i, h in holes.items():
        if i in ids:
            if isinstance(ring_mm, dict):
                value = ring_mm.get(i)
            else:
                value = ring_mm
            ctx.screws.append({"kp": kp, "region": ctx.region or "part", "file": file, "name": name, "hole": h,
                               "ring_mm": None if value is None else float(value), "item": item})
    if isinstance(ring_mm, dict):
        for i in ring_mm:
            if i not in ids:
                raise _KeyedError(f"{kp}.ring_mm.{i}: {i} is not a screw hole of this item (screw_holes: {', '.join(ids)})")


def make_ring_item(p: dict, ctx: BuildContext, kp: str) -> Manifold:
    """The solid of a `screw_ring` item: the ring round one hole of an STL, for `fixed`, `load` or `solid`. `kp` is the key path (`regions.fixed.wall[0].screw_ring`).

    It is the ring of the `tube` interface of the same hole: the same polygons and the same thickness rule (`ring_thickness`).
    """
    _check_ring_item(p, kp)
    if ctx.in_subtract or (ctx.role is not None and ctx.role not in RING_ROLES):
        where = "inside subtract" if ctx.in_subtract else f"in {ctx.role}"
        raise _KeyedError(f"{kp}: a screw ring is a frozen shape, and this item is {where}. Use it in {', '.join(RING_ROLES)}")
    name = p["path"]
    file = _stl_file(name, ctx, f"{kp}.path")
    found = _found(ctx, file, f"{kp}.path")
    holes = {h["id"]: h for h in found["holes"]}
    i = p["hole"]
    if i not in holes:
        raise _KeyedError(f"{kp}.hole: {i!r} is not a hole of {file.name}. Known ids: {_holes_words(found['holes']) or 'none (find_features lists no hole in this mesh)'}")
    why = _ring_refusal(holes[i])
    if why:
        raise _KeyedError(f"{kp}.hole: {why}")
    h = holes[i]
    value = None if p.get("ring_mm") is None else float(p["ring_mm"])
    t, reason = ring_thickness(h, value, ctx.voxel)
    ctx.rings.append({"kp": kp, "file": file, "name": name, "hole": h, "ring_mm": value, "t": t, "why": reason})
    return _ring_shape(h, t)


def _ring_noise_mm3(h: dict, t: float) -> float:
    """The volume below which the overlap of a ring with another shape is noise, not material.

    Two things move the ring by a hair, and both have a number already: the ring ends lie in the faces of the part up to the rounding of the
    `point` of `find_features` (`WALL_ROUND_MM` at each end, over the cross-section of the ring), and the polygon of the ring wall lies up to
    the chord sag of section 23 (`interfaces.SAG_MM`) off the circle of the hole, along the whole circumference.
    """
    from . import interfaces
    r_in, r_out = _ring_radii(h, t)
    end_area = math.pi * (r_out ** 2 - r_in ** 2)
    return 2.0 * end_area * WALL_ROUND_MM + 2.0 * math.pi * r_in * float(h["length_mm"]) * interfaces.SAG_MM


def _unique(name: str, used: set) -> str:
    """`name` made of safe characters, and not in `used` (a number is added when the name is taken)."""
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    out, n = base, 1
    while out in used:
        n += 1
        out = f"{base}_{n}"
    used.add(out)
    return out


def screw_rings(ctx: BuildContext, shapes: dict, voxel: float) -> tuple[list[dict], list[str]]:
    """`(interfaces, lines)` for the screw holes of the `stl` items of `ctx`, at the voxel size `voxel` of the YAML that is written.

    `interfaces` are the `tube` entries for the `interfaces` block (section 23), `lines` are the report lines (one for each screw hole and each
    `screw_ring` item, then a `note:` for a ring that leaves the part and a `WARNING:` for a ring in a keep-out region or thinner than the floor).
    """
    ifaces: list[dict] = []
    lines: list[str] = []
    used: set = set()
    keep_out = [(stem, m) for stem, (role, m) in shapes.items() if role == "keep_out"]
    thickness: dict = {}      # (file, hole id) -> [(key path, t)]: the interface and the item of one hole should agree
    for r in ctx.screws:
        h = r["hole"]
        t, why = ring_thickness(h, r["ring_mm"], voxel)
        r_in, r_out = _ring_radii(h, t)
        name = _unique(f"{r['region']}_{Path(r['name']).stem}_{h['id']}", used)
        ifaces.append({"name": name, "kind": "tube", "point": list(h["point"]), "axis": list(h["axis"]), "radius_mm": r_out,
                       "inner_radius_mm": r_in, "length_mm": float(h["length_mm"])})
        thickness.setdefault((r["file"], h["id"]), []).append((r["kp"], t))
        floor = f"; 0.5 x d is {RING_PER_DIAMETER * h['diameter_mm']:g} mm and the voxel is {voxel:g} mm" if why.startswith("raised") else ""
        lines.append(f"screw ring {r['kp']}: {h['id']} (d {h['diameter_mm']:g} mm), t {t:g} mm ({why}{floor}); tube interface '{name}'")
        if r["ring_mm"] is not None and t < RING_MIN_VOXELS * voxel:
            lines.append(f"WARNING: {r['kp']}.ring_mm: {h['id']}: t {t:g} mm is under {RING_MIN_VOXELS} voxels (the voxel is {voxel:g} mm), so the ring is "
                         f"less than {RING_MIN_VOXELS} voxels thick in the optimizer. A frozen ring that thin may hold few voxels")
        ring = _ring_shape(h, t)
        noise = _ring_noise_mm3(h, t)
        outside = (ring - r["item"]).volume()
        if outside > noise:
            lines.append(f"note: {r['kp']}: the ring of {h['id']} (t {t:g} mm) has {outside:.3g} mm3 outside the part ({r['name']}): the hole is near an edge or a face. "
                         "The exact union adds this material outside your part")
        for stem, ko in keep_out:
            overlap = (ring ^ ko).volume()
            if overlap > noise:
                lines.append(f"WARNING: {r['kp']}: the ring of {h['id']} (t {t:g} mm) overlaps the keep_out region '{stem}' by {overlap:.3g} mm3. "
                             "The keep-out region wins (CONTRACTS.md section 23): the ring is cut there")
    for r in ctx.rings:
        h = r["hole"]
        thickness.setdefault((r["file"], h["id"]), []).append((r["kp"], r["t"]))
        floor = f"; 0.5 x d is {RING_PER_DIAMETER * h['diameter_mm']:g} mm and the voxel is {voxel:g} mm" if r["why"].startswith("raised") else ""
        lines.append(f"screw ring {r['kp']}: {h['id']} of {r['name']} (d {h['diameter_mm']:g} mm), t {r['t']:g} mm ({r['why']}{floor})")
        if r["ring_mm"] is not None and r["t"] < RING_MIN_VOXELS * voxel:
            lines.append(f"WARNING: {r['kp']}.ring_mm: t {r['t']:g} mm is under {RING_MIN_VOXELS} voxels (the voxel is {voxel:g} mm)")
    for (file, hid), rows in thickness.items():
        if len({round(t, 9) for _, t in rows}) > 1:
            lines.append(f"WARNING: the ring of {hid} of {file.name} has two thicknesses: " + ", ".join(f"{t:g} mm at {k}" for k, t in rows)
                         + ". The ring item and the tube interface of one hole should have the same ring_mm")
    return ifaces, lines


# ---- spec -> shapes ---------------------------------------------------------
def named_items(role: str, val) -> dict[str, list]:
    """Return {name: items}. design = one shape. fixed/keep_out = list or mapping."""
    if role == "design":
        return {"design": val}
    if isinstance(val, dict) and not _looks_like_item(val):
        return dict(val)
    if isinstance(val, list) and val and all(isinstance(v, dict) and len(v) == 1
                                             and next(iter(v)) in _KINDS | {"subtract"}
                                             for v in val):
        return {role if role == "load" else f"{role}_1": val}     # one shape, many items
    raise SpecError(f"regions.{role}: use a mapping name -> list of primitives")


_KINDS = {"box", "cylinder", "sphere", "capsule", "bowl", "prism", "stl", "screw_ring"}


def _looks_like_item(d: dict) -> bool:
    (k, v), = d.items() if len(d) == 1 else ((None, None),)
    return k in _KINDS | {"subtract"} and isinstance(v, dict)   # a name maps to a LIST


def _spec_path(role: str, val, name: str) -> str:
    """The key path in the spec of the items of one region (`regions.design`, `regions.fixed.wall`, `regions.load` for the list form)."""
    if role == "design":
        return "regions.design"
    if isinstance(val, dict) and not _looks_like_item(val):
        return f"regions.{role}.{name}"
    return f"regions.{role}"


def build_all(spec: dict, base=None, notes: list | None = None, base_words: str | None = None, ctx: "BuildContext | None" = None):
    """The shape of each region: `{file stem: (role, Manifold)}`.

    `base` is the folder where a relative `path` of an `stl` item starts (None: the current folder; `base_words` names it in an error). `notes`, if given, receives the report lines of the
    `stl` items (one line each, and a `WARNING:` line for a fact the user has to see). `ctx` (a `BuildContext`) replaces the three: the caller keeps the screw
    holes that the items record in it (`screw_rings`).
    """
    reg = spec.get("regions")
    if not isinstance(reg, dict):
        raise SpecError("spec needs a 'regions' mapping")
    for k in reg:
        if k not in ROLES:
            raise SpecError(f"unknown region role '{k}'. Use {', '.join(ROLES)}")
    for need in ("design", "fixed", "load"):
        if need not in reg and not (need == "fixed" and (spec.get("springs") or spec.get("joints"))):     # a spring and a joint are supports
            raise SpecError(f"spec needs regions.{need}")
    ctx = ctx or BuildContext(base, notes, base_words)
    shapes: dict[str, tuple[str, Manifold]] = {}        # file stem -> (role, shape)
    for role in ROLES:
        if role not in reg:
            continue
        for name, items in named_items(role, reg[role]).items():
            stem = "design" if role == "design" else str(name)
            if not SAFE_NAME.fullmatch(stem):
                raise SpecError(f"regions.{role}: region name {stem!r} must be letters, digits, "
                                "'_', '-' or '.' (it becomes a file name)")
            if stem in shapes:
                raise SpecError(f"region name '{stem}' is used twice")
            ctx.role, ctx.region = role, stem
            shapes[stem] = (role, build_shape(items, f"regions.{role}.{name}", ctx, _spec_path(role, reg[role], str(name))))
    return shapes


def _region_ref(entry: dict):
    """The `region` of a force, pressure, spring or joint. A list or mapping cannot go in a set: its text is a name that no region has."""
    ref = entry.get("region")
    return repr(ref) if isinstance(ref, (list, dict)) else ref


def check_forces(spec: dict, shapes) -> None:
    """Every load region needs a force, a moment, a pressure or a spring. A forced-solid body without one is a `solid`.

    Also checks the spec key `fixed_axes` (a mapping fixed name -> list of axes) and the pressure targets.
    """
    cases = spec.get("load_cases") or []
    used = {_region_ref(f) for c in cases for f in c.get("forces") or []}
    pressed = {_region_ref(p) for c in cases for p in c.get("pressures") or []}
    sprung = {_region_ref(e) for e in spec.get("springs") or [] if isinstance(e, dict)}     # a spring uses its region
    jointed = {_region_ref(e) for e in spec.get("joints") or [] if isinstance(e, dict)}     # so does a joint (section 38)
    loads = {s for s, (r, _) in shapes.items() if r == "load"}
    for name in sorted(sprung - loads):
        raise SpecError(f"springs use region '{name}', but regions.load has no such name")
    for name in sorted(jointed - loads):
        raise SpecError(f"joints use region '{name}', but regions.load has no such name")
    for name in sorted((used | pressed) - loads):
        what = ""
        if name in pressed and shapes.get(name, ("",))[0] == "fixed":
            what = " (a fixed region is clamped, so a pressure on it would be lost)"
        elif name in pressed and shapes.get(name, ("",))[0] == "design":
            what = " (a pressure on the design surface changes with the design)"
        raise SpecError(f"load_cases use region '{name}', but regions.load has no such name{what}")
    for name in sorted(loads - used - pressed - sprung - jointed):
        raise SpecError(f"load region '{name}' has no force or pressure in load_cases, and no spring or joint. "
                        "If it only needs to stay solid, move it to regions.solid")
    fixed = {s for s, (r, _) in shapes.items() if r == "fixed"}
    axes = spec.get("fixed_axes") or {}
    if not isinstance(axes, dict):
        raise SpecError("fixed_axes must be a mapping: fixed region name -> list of axes (x, y, z)")
    for name, ax in axes.items():
        if name not in fixed:
            raise SpecError(f"fixed_axes names '{name}', but regions.fixed has no such name "
                            f"(known: {', '.join(sorted(fixed))})")
        if not (isinstance(ax, list) and ax and all(isinstance(a, str) and a.lower() in "xyz" and len(a) == 1
                                                    for a in ax) and len({a.lower() for a in ax}) == len(ax)):
            raise SpecError(f"fixed_axes.{name} must be a list of the axes x, y, z (distinct), "
                            f"for example [y] for a roller; got {ax!r}")


def spec_assumptions(spec: dict) -> list[str]:
    """The optional `assumptions` of a spec: a list of non-empty strings."""
    asm = spec.get("assumptions")
    if asm is None:
        return []
    if not isinstance(asm, list) or not all(isinstance(x, str) and x.strip() for x in asm):
        raise SpecError("assumptions must be a list of non-empty strings")
    return list(asm)


# ---- shape check ------------------------------------------------------------------------------------------------------------
# What `make_primitive` reads with `float()` (a number), by kind. The vectors (`_vec`) and `prism.plane` have their own check at build time.
_PRIMITIVE_NUMBERS = {"cylinder": ("radius",), "sphere": ("radius",), "capsule": ("radius",), "bowl": ("radius", "wall")}
_FORCE_EXAMPLE = "{region: tip, force: [0, 0, -2]}"
_CASE_EXAMPLE = "{name: hang, forces: [" + _FORCE_EXAMPLE + "]}"
# key -> (the shape in words, with an example) of the settings that the builder reads with `float()` or `int()`
_SETTING_NUMBERS = {"volume_fraction": "a number, for example 0.25", "penal": "a number, for example 3.0"}
_SETTING_INTEGERS = {"iterations": "a whole number, for example 80", "target_active": "a whole number of voxels, for example 40000"}
_MATERIAL_NUMBERS = {"E": "a number in MPa, for example 3500", "nu": "a number, for example 0.35",
                     "density": "a number in g/cm3, for example 1.24"}


def _got(v) -> str:
    """The type and a short repr of a spec value, for an error text. `reprlib` cuts a long value, so there is no length of our own."""
    return "null" if v is None else f"{type(v).__name__} {reprlib.repr(v)}"


def _shape(path: str, expect: str, got) -> SpecError:
    return SpecError(f"{path} must be {expect}, got {_got(got)}")


def _number_like(v) -> bool:
    """A number that `float()` reads: a finite int or float, or a text such as "5e-1" (YAML 1.1 loads 5e-1 as text). A bool is not a number."""
    if isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError, OverflowError):
        return False


def _integer_like(v) -> bool:
    """A value that `int()` reads, as the builder does for `iterations` and `target_active`. A bool is not a number."""
    if isinstance(v, bool):
        return False
    try:
        int(v)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _list_of_mappings(v, path: str, example: str) -> None:
    """`v` is absent (None) or a list of mappings, as `check_forces` reads `forces`, `pressures`, `springs` and `joints`."""
    if v is None:
        return
    if not isinstance(v, list):
        raise _shape(path, f"a list of mappings, for example [{example}]", v)
    for j, e in enumerate(v):
        if not isinstance(e, dict):
            raise _shape(f"{path}[{j}]", f"a mapping, for example {example}", e)


def _walk_primitives(node, path: str, above: tuple = ()) -> None:
    """Find each primitive under `regions` (a one-key mapping with a kind as key) and check the numbers and the point list that it reads.

    `above` holds the ids of the containers on the way down: a YAML anchor can make a list hold itself, and the walk must not run forever.
    """
    if isinstance(node, (dict, list)):
        if id(node) in above:
            raise SpecError(f"{path} holds itself (a YAML anchor that points back). Write the shape out, with no anchor that contains itself")
        above += (id(node),)
    if isinstance(node, dict):
        if len(node) == 1:
            (kind, p), = node.items()
            if kind in ("stl", "screw_ring") and not isinstance(p, list):      # a list is a region that is named `stl`
                (_check_stl_item if kind == "stl" else _check_ring_item)(p, f"{path}.{kind}")
                return
            if kind in _KINDS and isinstance(p, dict):
                for key in _PRIMITIVE_NUMBERS.get(kind, ()):
                    if key in p and not _number_like(p[key]):
                        raise _shape(f"{path}.{kind}.{key}", "a number in mm, for example 3.0", p[key])
                if kind == "prism" and "points" in p and not isinstance(p["points"], list):
                    raise _shape(f"{path}.{kind}.points", "a list of [a, b] pairs in mm, for example [[0, 0], [10, 0], [0, 10]]", p["points"])
                return
        for k, v in node.items():
            _walk_primitives(v, f"{path}.{k}", above)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_primitives(v, f"{path}[{i}]", above)


def check_spec_shape(spec: dict) -> None:
    """One walk over the spec before anything is built: each value that the builder reads has the type and shape it expects.

    A value of the wrong type is a `SpecError` that names the key path (`load_cases[0].forces[1]`), what the spec has (the type and a short
    repr) and what is expected (with an example). Without this walk, `float([1])` or `None.get(...)` gave `SpecError: TypeError: ...` (or a
    traceback) with no key. Only the shape is checked here. A number out of its range, an unknown key and a missing key stay with the code that
    knows the range: `make_primitive` (the vectors and `prism.plane`), `check_forces` (the region names and `fixed_axes`), `check_spec_keys`
    (an unknown top-level, `settings` or `material` key) and the YAML
    loader (`check_yaml` runs `load_config` on the written YAML; it checks the blocks that the builder copies as they are: the content of the
    load cases, `springs`, `joints`, `mirror` and `print`). The text `assumptions` has its own check (`spec_assumptions`), run here too.

    A number may be a text that `float()` reads, as before; a bool is not a number. A `null` for an optional key is the same as no key.
    """
    _walk_primitives(spec.get("regions"), "regions")
    cases = spec.get("load_cases")
    if not isinstance(cases, list):
        raise _shape("load_cases", f"a list of load cases, for example [{_CASE_EXAMPLE}]", cases)
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            raise _shape(f"load_cases[{i}]", f"a mapping, for example {_CASE_EXAMPLE}", c)
        _list_of_mappings(c.get("forces"), f"load_cases[{i}].forces", _FORCE_EXAMPLE)
        _list_of_mappings(c.get("pressures"), f"load_cases[{i}].pressures", "{region: pad, value_mpa: 0.1}")
    _list_of_mappings(spec.get("springs"), "springs", "{region: pad, stiffness: [100, 100, 100]}")
    _list_of_mappings(spec.get("joints"), "joints", "{region: pad, to: [30, 10, 60], section: {kind: round, d: 2}}")
    _list_of_mappings(spec.get("interfaces"), "interfaces", "{name: pin_bore, kind: bore, point: [0, 0, 0], axis: [0, 0, 1], radius_mm: 3, length_mm: 10}")
    st = spec.get("settings")
    if st is not None and not isinstance(st, dict):
        raise _shape("settings", "a mapping, for example {volume_fraction: 0.25, voxel_size: auto, iterations: 80}", st)
    st = st or {}
    for key, what in _SETTING_NUMBERS.items():
        if key in st and not _number_like(st[key]):
            raise _shape(f"settings.{key}", what, st[key])
    if "voxel_size" in st and st["voxel_size"] != "auto" and not _number_like(st["voxel_size"]):
        raise _shape("settings.voxel_size", "a number in mm, for example 2.0, or auto", st["voxel_size"])
    if "iterations" in st and not _integer_like(st["iterations"]):
        raise _shape("settings.iterations", _SETTING_INTEGERS["iterations"], st["iterations"])
    if st.get("target_active") and not _integer_like(st["target_active"]):     # a null or 0 `target_active` is no aim: the builder skips it
        raise _shape("settings.target_active", _SETTING_INTEGERS["target_active"], st["target_active"])
    if not isinstance(st.get("allow_large_grid", False), bool):
        raise SpecError("settings.allow_large_grid must be true or false")
    mat = spec.get("material")
    if mat is not None and not isinstance(mat, dict):
        raise _shape("material", "a mapping, for example {E: 3500, nu: 0.35, density: 1.24}", mat)
    for key, what in _MATERIAL_NUMBERS.items():
        if key in (mat or {}) and not _number_like(mat[key]):
            raise _shape(f"material.{key}", what, mat[key])
    fil = (mat or {}).get("filament")
    if fil is not None and (not isinstance(fil, str) or not fil.strip()):
        raise _shape("material.filament", "the name or the id of a filament as text, for example polymaker-polylite-pla", fil)
    spec_assumptions(spec)


def spec_mass_key(st: dict) -> bool:
    """True when the spec asks for a mass in grams (`settings.target_mass_g`, I7), False when it has `volume_fraction` or neither key.

    Both keys is a `SpecError` with the text of the YAML loader: the builder does not choose which one the user meant.
    """
    from .config import BOTH_VOLUME_KEYS_TEXT
    if "target_mass_g" in st and "volume_fraction" in st:
        raise SpecError(BOTH_VOLUME_KEYS_TEXT)
    return "target_mass_g" in st


# ---- the `settings` keys of a spec (F4-46) ----------------------------------------------------------------------------------
# One rule: the builder copies every `settings` key of the spec that the run YAML takes (`config.SETTINGS_REQUIRED` and
# `config.SETTINGS_OPTIONAL`), as given; the YAML loader checks the value. Two named sets are not copied. A key in neither set and not
# in the config stops the build (`check_settings_keys`). A new key of the config is copied with no change here.
#
# Run keys that the builder writes by its own rule. A spec value is not ignored: each rule starts from it.
BUILDER_OWNED_SETTINGS = {
    "voxel_size": "the builder writes the size of the grid it built: the number of the spec, `auto` (the voxel picker) or `--voxel`",
    "iterations": "written as int(), default 80; the same number sizes the voxel aim",
    "volume_fraction": "written as float(), default 0.25; the key is absent when the spec has `target_mass_g`",
    "target_mass_g": "written as given in place of `volume_fraction` (I7: one of the two); both keys is an error (`spec_mass_key`)",
}
# Keys that only the builder reads. The run YAML does not take them, so they are never written.
BUILDER_ONLY_SETTINGS = {
    "target_active": "the aim of the voxel picker, in active voxels (`auto_voxel`)",
    "allow_large_grid": "lets the build go on when the memory estimate of the grid is above the RAM (`build_spec`)",
}
# The order of the copied keys in the YAML: the keys that the builder copied before F4-46 keep the order they had (a built YAML stays
# byte-equal); every other key follows in the order of `config.SETTINGS_OPTIONAL`. A key that is not listed is copied all the same.
_LEGACY_COPY_ORDER = ("filter_radius", "penal", "move_limit", "solver", "max_grid_voxels", "max_active_voxels", "beta_schedule",
                      "case_combine", "mesh_method", "objective_scale", "min_member_mm", "filter_solid_neighbours", "stop_rules",
                      "perimeter_max_mm2")


def check_settings_keys(st: dict) -> None:
    """A `settings` key that is neither a key of the run YAML nor a builder-only key is a `SpecError` with the text of the YAML loader.

    The text names the key and the keys that the spec takes, so the build never drops a key without a word.
    """
    from .config import SETTINGS_OPTIONAL, SETTINGS_REQUIRED, unknown_key_text
    known = [*SETTINGS_REQUIRED, *SETTINGS_OPTIONAL]
    for k in st:
        if k not in known and k not in BUILDER_ONLY_SETTINGS:
            raise SpecError(unknown_key_text(k, "settings", [*known, *BUILDER_ONLY_SETTINGS]))


def copied_settings(st: dict) -> dict:
    """The `settings` keys of the spec that go to the YAML as given: every key of the run YAML except the builder-owned ones.

    The value is not changed or checked here: `load_config` checks it (`check_yaml`).
    """
    from .config import SETTINGS_OPTIONAL, SETTINGS_REQUIRED
    check_settings_keys(st)
    rank = {k: i for i, k in enumerate(dict.fromkeys((*_LEGACY_COPY_ORDER, *SETTINGS_OPTIONAL, *SETTINGS_REQUIRED)))}
    return {k: st[k] for k in sorted((k for k in st if k not in BUILDER_OWNED_SETTINGS and k not in BUILDER_ONLY_SETTINGS), key=rank.__getitem__)}


# ---- the top-level keys and the `material` keys of a spec (F4-46, round 2) -----------------------------------------------------
# The same rule as for `settings`: the builder copies every top-level key of the spec that the run YAML takes (`config.TOP_REQUIRED` and
# `config.TOP_OPTIONAL`), as given, and the loader checks it (`limits`, `buckling`, `interfaces`, `acceptance`, `bore_checks`, ...). Two named
# sets are not copied. A key in neither set and not in the config stops the build (`check_spec_keys`). A new top-level key of the config is
# copied with no change here.
#
# Top-level keys that the builder writes itself, in the skeleton of the YAML (`write_outputs`). A spec value is not ignored.
BUILDER_OWNED_TOP_LEVEL = {
    "name": "the YAML name and the file stem; `build_in_workspace` sets it from its `name` argument; default `part`",
    "regions": "the primitives of the spec become STL files; the YAML names the files, by role",
    "load_cases": "written as given (the loader checks them); the builder also reads the forces to check the load regions",
    "settings": "a block with its own rules for each key: see BUILDER_OWNED_SETTINGS and `copied_settings`",
    "material": "E, nu and density are written as float() with the defaults of the YAML, or a filament with only the numbers that the spec gives; see BUILDER_OWNED_MATERIAL and `material_block`",
}
# Keys that only the spec has. The run YAML does not take them, so they are never written.
SPEC_ONLY_TOP_LEVEL = {
    "fixed_axes": "a fixed region name -> axes; the YAML gets it as the `{file, axes}` entry of that region in `regions.fixed`",
}
# The keys of `material` that the YAML takes (`config.MATERIAL_OPTIONAL`). The builder writes all of them, so a spec value is used.
BUILDER_OWNED_MATERIAL = {
    "E": "written as float(), default 3500 MPa",
    "nu": "written as float(), default 0.35",
    "density": "written as float(), default 1.24 g/cm3",
    "filament": "written stripped; with a filament, E, nu and density are written only when the spec gives them (`material_block`)",
}
# Before F4-46 an empty value of these keys asked for nothing and was not written; the YAML stays byte-equal for such a spec. A null is
# never written (it is the same as no key). Every other value is written as given.
_OMITTED_WHEN_EMPTY = ("springs", "joints", "assumptions", "mirror")
# The order of the copied keys in the YAML: the keys that the builder copied before F4-46 keep their order; the other keys follow in the
# order of `config.TOP_OPTIONAL`. A key that is not listed is copied all the same.
_LEGACY_TOP_ORDER = ("springs", "joints", "assumptions", "mirror", "print")


def check_spec_keys(spec: dict) -> None:
    """A top-level key, a `settings` key or a `material` key that the spec route does not know is a `SpecError` with the text of the YAML loader.

    The text names the key and the keys that the spec takes (the loader's keys and the spec-only ones), so the build never drops a key
    without a word. The top level says `top level`, as the loader does.
    """
    from .config import MATERIAL_OPTIONAL, TOP_OPTIONAL, TOP_REQUIRED, unknown_key_text
    known = [*TOP_REQUIRED, *TOP_OPTIONAL]
    for k in spec:
        if k not in known and k not in SPEC_ONLY_TOP_LEVEL:
            raise SpecError(unknown_key_text(k, "top level", [*known, *SPEC_ONLY_TOP_LEVEL]))
    if isinstance(spec.get("settings"), dict):      # a block of the wrong type is a shape error (`check_spec_shape`)
        check_settings_keys(spec["settings"])
    if isinstance(spec.get("material"), dict):
        for k in spec["material"]:
            if k not in MATERIAL_OPTIONAL:
                raise SpecError(unknown_key_text(k, "material", MATERIAL_OPTIONAL))


def copied_material(mat: dict) -> dict:
    """The `material` keys of the spec that go to the YAML as given: every key of the run YAML except the builder-owned ones (`BUILDER_OWNED_MATERIAL`)."""
    return {k: v for k, v in mat.items() if k not in BUILDER_OWNED_MATERIAL}


def copied_top_level(spec: dict) -> dict:
    """The top-level keys of the spec that go to the YAML as given: every key of the run YAML except the builder-owned ones.

    The value is not changed or checked here: `load_config` checks it (`check_yaml`).
    """
    from .config import TOP_OPTIONAL, TOP_REQUIRED
    check_spec_keys(spec)
    rank = {k: i for i, k in enumerate(dict.fromkeys((*_LEGACY_TOP_ORDER, *TOP_OPTIONAL, *TOP_REQUIRED)))}
    keys = [k for k, v in spec.items() if k not in BUILDER_OWNED_TOP_LEVEL and k not in SPEC_ONLY_TOP_LEVEL and v is not None
            and not (k in _OMITTED_WHEN_EMPTY and not v)]
    return {k: spec[k] for k in sorted(keys, key=rank.__getitem__)}


def material_block(mat: dict) -> dict:
    """The `material` block of the written YAML. With a `filament`, the block holds the filament and only the numbers that the spec gives:
    the loader takes the others from the filament profile (a default written here would win over the profile). The other keys of the
    run YAML follow as given (`copied_material`)."""
    if mat.get("filament") is not None:
        return {"filament": mat["filament"].strip(), **{k: float(mat[k]) for k in ("E", "nu", "density") if mat.get(k) is not None},
                **copied_material(mat)}
    return {"E": float(mat.get("E", 3500.0)), "nu": float(mat.get("nu", 0.35)), "density": float(mat.get("density", 1.24)),
            **copied_material(mat)}


def write_outputs(spec: dict, shapes, out: Path, voxel_size: float, rings: list | None = None) -> Path:
    """Write the region STLs and the problem YAML. `rings` are the `tube` entries of the screw holes (`screw_rings`); they follow the `interfaces` of the spec."""
    check_spec_keys(spec)     # F4-46: no file is written for a spec with a key that the YAML does not take
    mass_key = spec_mass_key(spec.get("settings") or {})
    out.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for stem, (role, m) in shapes.items():
        tm = to_trimesh(m)
        if not tm.is_watertight or tm.volume <= 0:
            raise SpecError(f"region '{stem}' is not a closed solid")
        tm.export(out / f"{stem}.stl")
        meshes[stem] = tm
    by_role = {r: [s for s, (rr, _) in shapes.items() if rr == r] for r in ROLES}
    mat = spec.get("material") or {}
    st = spec.get("settings") or {}
    doc = {
        "name": str(spec.get("name", "part")),
        "regions": {
            "design": "design.stl",
            # A `fixed_axes` name fixes only those axes (a roller); the other names are full clamps.
            "fixed": [{"file": f"{s}.stl", "axes": list((spec.get("fixed_axes") or {})[s])}
                      if s in (spec.get("fixed_axes") or {}) else f"{s}.stl" for s in by_role["fixed"]],
            "keep_out": [f"{s}.stl" for s in by_role["keep_out"]],
            "load": {s: f"{s}.stl" for s in by_role["load"]},
        },
        "load_cases": spec.get("load_cases") or [],
        "material": material_block(mat),
        "settings": {"voxel_size": float(voxel_size),
                     # The YAML has one of the two keys (I7), as given. A mass in grams stays a mass: `build_problem` derives the fraction,
                     # and the builder does not write a fraction in its place (that would loosen what the user asked for).
                     **({"target_mass_g": st["target_mass_g"]} if mass_key else
                        {"volume_fraction": float(st.get("volume_fraction", 0.25))}),   # as given: the mean density of the FREE design voxels
                     "iterations": int(st.get("iterations", 80))},
    }
    # F4-46: every other key of the run YAML goes through as given, the loader checks it. The builder-owned keys are written above.
    doc["settings"].update(copied_settings(st))
    # A `solid` body is frozen at density 1, with no clamp and no force (CONTRACTS.md section 22a).
    if by_role["solid"]:
        doc["regions"]["solid"] = [f"{s}.stl" for s in by_role["solid"]]
    # F4-46: every other top-level key of the run YAML goes through as given (`springs`, `joints`, `assumptions`, `mirror`, `print`,
    # `limits`, `buckling`, `interfaces`, `acceptance`, `bore_checks`); the loader checks it. Section 45.2: the `tube` entries of the screw rings
    # follow the spec's own `interfaces` entries (the order of the list is the order of the boolean steps).
    top = {**spec, "interfaces": [*(spec.get("interfaces") or []), *rings]} if rings else spec
    doc.update(copied_top_level(top))
    path = out / f"{doc['name']}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    return path


def time_factor(fill: float) -> float:
    """Scale of the model estimate by the fill of the part: 1 for compact parts, 0.27 for slender parts (two runs on Apple silicon: the hook and the ladle)."""
    t = (fill - FILL_SLENDER) / (FILL_COMPACT - FILL_SLENDER)
    return F_SLENDER + (1.0 - F_SLENDER) * min(1.0, max(0.0, t))


def target_active(fill: float, iterations: int, solver: str = "cholesky", gpu=None) -> int:
    """Default aim: active voxels for an estimated run time of GOAL_S (the model scaled by the fill), at most CAP_ACTIVE.

    For `solver` "gpu-mg" the free dofs come from the gpu-mg model (`gpu_mg.check.free_dofs_for_seconds`): it is linear in the
    free dofs and has no fill term (the time does not depend on how compact the part is), so the time factor is 1. `gpu` is a
    `gpu_mg.check.TimeInputs` (the load cases and the contrast of the spec; without it, one load case and no contrast).
    """
    f = time_factor(fill)
    if solver == "gpu-mg":
        from .gpu_mg.check import free_dofs_for_seconds
        free = free_dofs_for_seconds(GOAL_S, iterations, gpu)
        if free is None:     # a CPU OpenCL device: the model has no seconds, so no time goal: the cap of the aim decides
            return CAP_ACTIVE
    else:
        per_it = (GOAL_S - SETUP_S - POST_S) / (iterations + 1) / f      # model s/it (before the fill factor)
        free = REF_DOFS * (per_it / REF_S) ** (1 / DOF_POWER)
    ratio = 3.5 - 0.6 * (f - F_SLENDER) / (1 - F_SLENDER)             # free dofs / active voxel (the shape of the part)
    return int(min(CAP_ACTIVE, free / ratio))


def spec_time_model(spec: dict, user_vf: float):
    """`(solver, TimeInputs or None)` for the voxel aim: ("gpu-mg", inputs) when the spec names gpu-mg and it can run here, else ("cholesky", None).

    The aim does not read the `gpu_*` settings of the spec (the builder copies them to the YAML since F4-46; `check_yaml` reads them there),
    so the device is the one that `auto` takes and the tolerance is the default; the contrast comes from `volume_fraction` and `penal`
    (3 when the spec has none, as the loader).
    """
    st = spec.get("settings") or {}
    if st.get("solver") != "gpu-mg":
        return "cholesky", None
    from .config import UserError
    from .solver_choice import resolve_solver
    try:
        resolve_solver("gpu-mg")
    except UserError:       # `check_yaml` says it and uses the default model; so does the aim
        return "cholesky", None
    from .gpu_mg.check import TimeInputs, chosen_device, device_case
    from .gpu_mg.device import facts
    dev = chosen_device(facts("auto"))
    return "gpu-mg", TimeInputs(n_cases=max(1, len(spec.get("load_cases") or [])), volume_fraction=user_vf,
                                penal=float(st.get("penal", 3.0)), device=dev, case=device_case(dev, "auto"))


def auto_voxel(shapes, iterations: int = 80, target: int | None = None, solver: str = "cholesky", gpu=None) -> float:
    """voxel = cbrt(active volume / target), rounded up to 0.05 mm.

    `target` is the number of active voxels. Without it, the default aim `target_active(fill)` is used (`solver` and `gpu` as there).
    """
    act = None
    for role in ("design", "fixed", "load", "solid"):
        for r, m in shapes.values():
            if r == role:
                act = m if act is None else act + m
    for r, m in shapes.values():
        if r == "keep_out":
            act = act - m
    vol = act.volume()
    lo = np.min([m.bounding_box()[:3] for _, m in shapes.values()], axis=0)
    hi = np.max([m.bounding_box()[3:] for _, m in shapes.values()], axis=0)
    fill = vol / float(np.prod(hi - lo))
    h = (vol / (target if target else target_active(fill, iterations, solver, gpu))) ** (1 / 3)
    return math.ceil(h * 20) / 20


# ---- check ------------------------------------------------------------------
def check_yaml(path: Path, say=print, allow_large_grid: bool = False):
    """Run the topopt input layer. Send each report line to `say` (default: print).

    Return `(ok, kind)`. `kind` is "time" when only the voxel count or time failed
    (a bigger voxel helps), else "other" (a bigger voxel does not help); "memory" when the
    memory estimate is above the RAM (see `allow_large_grid`).
    Soft levels (small region, forced-solid share, a time estimate above the budget of the user) give `WARNING:` lines, not errors.
    The only time budget is the one of the user, `settings.stop_rules.time_budget_s`; without it the time estimate is a fact and there is no budget warning.
    The function does not change the YAML. `settings.volume_fraction` is the mean density of the FREE design
    voxels (CONTRACTS.md section 2); the report states this basis once and gives the target volume and mass of
    the run (`topopt.cli.volume_target`, the numbers of `topopt check`).
    """
    from scipy import ndimage
    from .cli import (bordered_bytes, budget_effect, budget_words, estimate_memory_bytes, estimate_seconds, fit_voxel_size, gib,
                      physical_ram_bytes, time_budget_s, volume_target)
    from .config import UserError, load_config
    from .connectivity import (RUN_STOPS, held_pieces_message, message as pieces_message, support_touch, touch_message,
                               unsupported_pieces)
    from .loads import no_load_lines, repeated_load_cases, repeated_load_lines
    from .optimize import reduced_system
    from .problem import build_problem, with_volume
    from .solver_choice import resolve_solver
    from .symmetry import summary_lines
    from .voxelize import DESIGN, FIXED, KEEP_OUT, LOAD, plane_lines

    try:
        cfg = load_config(path)
        prob = build_problem(cfg)
    except UserError as e:
        say(f"ERROR: {e}")
        msg = str(e)
        return False, ("time" if "limit is" in msg and "Try voxel_size" in msg else "other")
    cfg = with_volume(cfg, prob)   # I7: a YAML with settings.target_mass_g has the volume fraction that the problem derived
    lab = prob.labels
    ok, kind = True, "other"
    say(f"grid:    {prob.shape[0]} x {prob.shape[1]} x {prob.shape[2]} voxels, "
          f"voxel {prob.h} mm, origin {np.round(prob.origin, 1).tolist()}")
    names = {DESIGN: "design", FIXED: "fixed", LOAD: "load+solid", KEEP_OUT: "keep_out"}
    say("voxels:  " + ", ".join(f"{n}={int((lab == c).sum())}" for c, n in names.items()))
    for i, lr in enumerate(cfg.regions.load):
        m = prob.load_region_ids == i
        n = int(m.sum())
        thick = float(ndimage.distance_transform_edt(np.pad(m, 1)).max()) if n else 0.0
        say(f"         '{lr.name}': {n} voxels, thickness {2 * thick - 1:.1f} voxels")
        if n < MIN_REGION_VOXELS or thick < MIN_REGION_THICK:
            say(f"WARNING: region '{lr.name}' is small or thin ({n} voxels, about "
                  f"{2 * thick - 1:.1f} voxels across; {MIN_REGION_VOXELS} voxels and 3 across is the "
                  "usual level). The optimizer can cut it off and remove it. "
                  "Make it the whole solid feature it acts on, or use a smaller voxel_size.")
    for r in (prob.region_facts.get("solid") or {}).get("regions", []):
        say(f"         solid '{r['name']}': {r['voxels']} voxels, thickness {r['thickness_voxels']:.1f} voxels")
        if r["voxels"] < MIN_REGION_VOXELS or (r["thickness_voxels"] + 1) / 2 < MIN_REGION_THICK:
            say(f"WARNING: region '{r['name']}' is small or thin ({r['voxels']} voxels, about "
                  f"{r['thickness_voxels']:.1f} voxels across; {MIN_REGION_VOXELS} voxels and 3 across is the "
                  "usual level). The optimizer can cut it off and remove it. "
                  "Make it the whole solid feature it acts on, or use a smaller voxel_size.")
    active = (lab == DESIGN) | (lab == FIXED) | (lab == LOAD)
    n_act = int(active.sum())
    n_free_vox = int((lab == DESIGN).sum())
    n_frozen = n_act - n_free_vox
    say(f"active:  {n_act} voxels ({n_frozen} forced solid, {n_free_vox} free design)")
    for line in summary_lines(prob.mirror):   # the `mirror` key: parity, orphans, load asymmetry
        say("mirror:  " + line)
    if n_act and n_frozen > MAX_FROZEN * n_act:
        say(f"WARNING: forced-solid voxels are {100 * n_frozen / n_act:.0f} percent of the active "
              f"voxels ({100 * MAX_FROZEN:.0f} percent is the usual level). Little is left to optimize. "
              "Make the fixed, load and solid regions smaller, or the design space bigger.")
    vf = cfg.settings.volume_fraction
    vol = volume_target(vf, float(prob.h), cfg.material.density, n_act, n_free_vox)
    if cfg.settings.target_mass_g is not None:
        say(f"volume_fraction: {vf:.6g} is derived from settings.target_mass_g {cfg.settings.target_mass_g:g} g; it is the mean density of the "
            f"{n_free_vox} free design voxels, and the {n_frozen} forced-solid voxels are solid on top")
    else:
        say(f"volume_fraction: {vf:g} is the mean density of the {n_free_vox} free design voxels; the {n_frozen} "
            f"forced-solid voxels are solid on top (the YAML keeps {vf:g})")
    say(f"target:  ({n_frozen} + {vf:g} * {n_free_vox}) voxels * {float(prob.h):g}^3 mm3 = "
        f"{vol['target_volume_mm3']:.0f} mm3, {vol['target_mass_g']:.1f} g at {cfg.material.density:g} g/cm3 "
        "(the numbers of `topopt check`)")
    say(f"share:   the target is {(n_frozen + vf * n_free_vox) / n_act:.4f} of the {n_act} active voxels "
        "(for information; written nowhere)")
    _e, _a, free = reduced_system(prob)
    try:   # the estimate depends on the solver that really runs on this machine
        solver = resolve_solver(cfg.settings.solver)
    except UserError as e:
        say(f"note: {e}. The estimate uses the default model.")
        solver = "cholesky"
    fill = n_act / float(np.prod(prob.shape))
    budget = time_budget_s(cfg.settings)     # the budget of the user, or None: then there is no budget warning
    say(f"free dofs: {free.size}")
    if solver == "gpu-mg":
        # Its own model (P2a): linear in the free dofs, with a band, and no fill term: a slender part is not faster.
        from .cli import gpu_mg_estimate, opencl_facts
        from .gpu_mg.check import TimeInputs, basis_text
        est = gpu_mg_estimate(free.size, cfg.settings.iterations, TimeInputs.from_config(cfg, opencl_facts(cfg.settings)))
        per_it, total = est.s_per_iteration, est.total_s
        if total is None:     # a CPU OpenCL device has no constants: no seconds
            say("time estimate: none for gpu-mg on this device (the model is linear in the free dofs and has no constants for a CPU device)")
        else:
            say(f"time estimate: gpu-mg model about {total:.0f} s, band {est.low_s:.0f} to {est.high_s:.0f} s "
                f"(linear in the free dofs, so the fill {fill:.2f} does not change it)")
        say("time basis: " + basis_text(est.as_dict()) + ". It assumes: " + "; ".join(est.assumptions))
        if budget is not None and total is not None and total > budget:
            fit = fit_voxel_size(float(prob.h), total, budget, solver)
            say(f"WARNING: the gpu-mg estimate is above {budget_words(budget)}; {budget_effect(cfg.settings.stop_rules)}. "
                + (f"A voxel_size of about {fit} mm would fit." if fit else ""))
    else:
        per_it, total = estimate_seconds(free.size, cfg.settings.iterations, solver)
        f = time_factor(fill)
        scaled = SETUP_S + POST_S + f * (total - SETUP_S - POST_S)
        say(f"time estimate: about {total:.0f} s (the model); scaled by the fill {fill:.2f} (factor {f:.2f}): "
            f"about {scaled:.0f} s")
        if budget is not None and total > budget:
            fit = fit_voxel_size(float(prob.h), total, budget, solver)
            say(f"WARNING: the estimate is above {budget_words(budget)}; {budget_effect(cfg.settings.stop_rules)}. "
                + (f"A voxel_size of about {fit} mm would fit." if fit else ""))
    if solver == "amg":
        # With amg the run takes longer than the model of the fast solvers. This is a fact, not a stop.
        pd_total = estimate_seconds(free.size, cfg.settings.iterations, "pardiso")[1]
        say(f"note: solver amg. The estimate is {total:.0f} s with amg and about {pd_total:.0f} s "
            "with pardiso (rough model). "
            "To use pardiso: see the recommendation of `topopt check`.")
    ram = physical_ram_bytes()
    from .members import filter_size
    fbytes = filter_size(float(cfg.settings.filter_radius), float(prob.h), n_free_vox, prob.shape)["held_bytes_estimate"]
    bord = 0 if solver == "gpu-mg" else bordered_bytes(prob, _e, free)     # joints (section 38): the bordered copy of K; 0 without joints and with gpu-mg (matrix-free, P2b)
    mem = estimate_memory_bytes(int(np.prod(prob.shape)), int(free.size), fbytes, bord)   # the filter too
    if ram:
        say(f"memory estimate: about {gib(mem)} of {gib(ram)} RAM (rough model, with the filter)")
        if mem > ram and not allow_large_grid:
            say(f"ERROR: the memory estimate ({gib(mem)}) is above the RAM of this machine "
                  f"({gib(ram)}). The run can crash the machine (a 0.5 mm whole-domain grid "
                  "did on a 16 GB Mac). Use a larger voxel_size, or set settings.allow_large_grid: "
                  "true in the spec to build anyway.")
            ok, kind = False, "memory"
        elif mem > ram:
            say(f"WARNING: the memory estimate is above the RAM; allow_large_grid is set, so the "
                  "build goes on.")
    else:
        say("memory estimate: not available (the RAM size is not known here)")
    # One connected body, with every load region joined to a fixed region.
    comp, nc = ndimage.label(active)
    touch = support_touch(prob)
    loose = pieces_message(unsupported_pieces(prob))     # the same rule and text as `topopt check`
    if loose:     # a piece with no support: `topopt run` stops, so this is an error (a bigger voxel does not help)
        say("ERROR: " + loose + RUN_STOPS)
        ok, kind = False, "other"
    elif nc > 1:  # every piece holds a support: the run works, so this is a fact
        sizes = sorted(np.bincount(comp.ravel())[1:].tolist(), reverse=True)
        say("WARNING: " + held_pieces_message(sizes, touch["held_by"]))
    if not (lab == DESIGN).any():
        say("ERROR: no design voxel is left.")
        ok, kind = False, "other"
    # A support that shares no face with the design space is a fact, not an error: it can hold the part through load or
    # solid voxels, and then the run works. The pieces check above is the error where the run cannot work.
    touch_text = touch_message(touch)
    if touch_text:
        say("WARNING: " + touch_text)
    for line in plane_lines(prob.region_facts.get("planes"), float(prob.h), prob.origin):   # one line, the same text as `topopt check`
        say("note: " + line)      # a fact that is true for most parts, so it is a note and not a WARNING
    for line in repeated_load_lines(repeated_load_cases(cfg)):   # F4-5: a load case that repeats another or is its opposite
        say(line)             # the same text as `topopt check` and the run log; it starts with "fact:", so it is no WARNING
    for line in no_load_lines(cfg, prob.load_reach or None):   # U21 and F4-37: a load case that applies no load (or none to a free dof)
        say(line)             # the same text as `topopt check` and the run log
    say("RESULT:", "OK" if ok else "FAILED")
    return ok, (kind if not ok else "ok")


# ---- build ------------------------------------------------------------------
def _rings_at_voxel(spec: dict, ctx: BuildContext, shapes: dict, h: float):
    """`(ctx, shapes)` for the voxel size `h`. A `screw_ring` item with no `ring_mm` is at least `RING_MIN_VOXELS` voxel sizes thick, so the
    shapes are built again when that floor gives another thickness than the one they were built with (the `find_features` reply is in the cache, so the holes are not searched again; the mesh is read again)."""
    stale = any(r["ring_mm"] is None and ring_thickness(r["hole"], None, h)[0] != r["t"] for r in ctx.rings)
    if not stale:
        return ctx, shapes
    ctx = ctx.again(h)
    return ctx, build_all(spec, ctx=ctx)


def build_spec(spec: dict, out: Path, voxel: float | None = None, check: bool = False,
               say=print, base=None, base_words: str | None = None) -> dict:
    """Build the shapes, write STLs + YAML into `out`, run the check. Raise SpecError on a bad spec.

    With `check`, build into a temp folder and leave `out` alone. Report lines go to `say`.
    `base` is the folder where a relative `path` of an `stl` item starts (None: the current folder); `base_words` names it in an error.
    Return {ok, kind, voxel_size, voxel_auto, retries, yaml (Path), out (Path)}.
    """
    if not isinstance(spec, dict):
        raise SpecError("spec must be a mapping")
    if not spec.get("load_cases"):
        raise SpecError("spec needs load_cases")
    check_spec_shape(spec)     # the one shape check: a value of the wrong type is a SpecError that names the key
    check_spec_keys(spec)     # F4-46: a top-level, `settings` or `material` key that the YAML does not take stops the build here, before the geometry
    try:
        notes: list[str] = []
        ctx = BuildContext(base, notes, base_words)
        shapes = build_all(spec, ctx=ctx)
        for line in notes:     # one line for each `stl` item, then the facts about it
            say(line)
        check_forces(spec, shapes)
        st = spec.get("settings") or {}
        mass_key = spec_mass_key(st)   # both keys: a SpecError
        # A bad number is a spec error here; the YAML keeps the value as given. A spec with `target_mass_g` has no fraction before the grid
        # exists, so the time model of the voxel aim takes the default 0.25 (it matters for gpu-mg only; the report says so below).
        user_vf = 0.25 if mass_key else float(st.get("volume_fraction", 0.25))
        v = voxel if voxel else st.get("voxel_size", "auto")
        target = int(st["target_active"]) if st.get("target_active") else None
        t_solver, t_gpu = spec_time_model(spec, user_vf)
        h = auto_voxel(shapes, int(st.get("iterations", 80)), target, t_solver, t_gpu) if v == "auto" else float(v)
        allow_large = st.get("allow_large_grid", False)     # a bool: `check_spec_shape` checked it
        ctx, shapes = _rings_at_voxel(spec, ctx, shapes, h)          # a screw_ring item is at least 2 voxels thick: it needs the voxel size
        rings, ring_lines = screw_rings(ctx, shapes, h)
    except SpecError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        raise SpecError(f"{type(e).__name__}: {e}")
    if check:
        out = Path(tempfile.mkdtemp(prefix="part_check_"))
    yml = write_outputs(spec, shapes, Path(out), h, rings)
    for line in ring_lines:     # one line for each screw hole: the thickness of its ring and why
        say(line)
    say(f"wrote {yml} (voxel_size {h} mm{', auto' if v == 'auto' else ''})")
    if mass_key and v == "auto" and t_gpu is not None:
        say(f"note: the voxel aim took volume_fraction {user_vf:g} for the gpu-mg time model, because the spec has settings.target_mass_g and "
            "the fraction is derived from the grid; the check below reports the derived fraction")
    from .cli import GRID_BYTES_PER_VOXEL, gib, physical_ram_bytes
    from .voxelize import grid_from_bounds
    lo = np.min([m.bounding_box()[:3] for _, m in shapes.values()], axis=0)
    hi = np.max([m.bounding_box()[3:] for _, m in shapes.values()], axis=0)
    grid_bytes = GRID_BYTES_PER_VOXEL * int(np.prod(grid_from_bounds(lo, hi, h)[1]))
    ram = physical_ram_bytes()
    if ram and grid_bytes > ram and not allow_large:
        say(f"ERROR: the grid alone needs about {gib(grid_bytes)} (rough model) and this machine "
            f"has {gib(ram)} RAM. Building it can crash the machine (a 0.5 mm whole-domain "
            "grid did on a 16 GB Mac). Use a larger voxel_size, or set settings.allow_large_grid: "
            "true in the spec to build anyway (settings.max_grid_voxels may need a raise too).")
        return {"ok": False, "kind": "memory", "voxel_size": h, "voxel_auto": v == "auto",
                "retries": 0, "yaml": yml, "out": Path(out)}
    ok, kind = check_yaml(yml, say, allow_large)
    tries = 0
    # Auto size: grow 8 percent and retry, only for time or voxel-count failures.
    # Other failures (a piece with no support, small region) do not get better with a bigger voxel.
    while not ok and kind == "time" and v == "auto" and tries < 3:
        tries += 1
        h = round(h * 1.08, 2)
        ctx, shapes = _rings_at_voxel(spec, ctx, shapes, h)
        rings, new_lines = screw_rings(ctx, shapes, h)
        yml = write_outputs(spec, shapes, Path(out), h, rings)
        say(f"retry with voxel_size {h} mm")
        if new_lines != ring_lines:     # the floor of 2 voxels moved the ring
            ring_lines = new_lines
            for line in ring_lines:
                say(line)
        ok, kind = check_yaml(yml, say, allow_large)
    return {"ok": ok, "kind": kind, "voxel_size": h, "voxel_auto": v == "auto",
            "retries": tries, "yaml": yml, "out": Path(out)}


def build_in_workspace(spec, work: Path, name: str | None = None) -> dict:
    """Data entry point (MCP tool). `spec` is a dict or a YAML string.

    Write the STLs and the problem YAML to <work>/runs/<name>/ and return facts:
    the builder report, `check` (topopt.cli.check_data of the written YAML) and relative paths.
    A bad spec gives {"ok": False, "error": one line}. Builder problems (a region that is too
    thin, a piece with no support, a memory estimate above the RAM) are facts in `problems`; the files stay,
    and `ok` is False with `builder_ok` False (a build that did not build is not a success).
    """
    if isinstance(spec, str):
        try:
            spec = yaml.safe_load(spec)
        except yaml.YAMLError as e:
            return {"ok": False, "error": "spec is not valid YAML: " + " ".join(str(e).split())}
    if not isinstance(spec, dict):
        return {"ok": False, "error": "spec must be a mapping (dict or YAML text)"}
    spec = dict(spec)
    nm = str(name if name is not None else spec.get("name", "part"))
    if not SAFE_NAME.fullmatch(nm):
        return {"ok": False, "error": f"name {nm!r} must be letters, digits, '_', '-' or '.' "
                                      "(it is a folder and file name)"}
    spec["name"] = nm
    from .preferences import fill_spec      # a block that the spec does not have at all comes from the preferences file, with a fact
    try:
        preference_facts = fill_spec(spec, work)
    except Exception as e:      # the preferences are an extra: a failure here is a fact, and the spec builds as it is
        preference_facts = [f"The preferences file was not used: internal error: {type(e).__name__}: {e}"]
    out = Path(work) / "runs" / nm
    lines: list[str] = []
    try:
        existed = out.exists() and any(out.iterdir())
        r = build_spec(spec, out, say=lambda *a: lines.append(" ".join(map(str, a))), base=Path(work), base_words="the work folder")
    except SpecError as e:
        return {"ok": False, "error": " ".join(str(e).split())}
    except Exception as e:
        return {"ok": False, "error": f"internal error: {type(e).__name__}: {e}"}

    def rel(p: Path) -> str:
        return Path(p).resolve().relative_to(Path(work).resolve()).as_posix()

    doc = yaml.safe_load(Path(r["yaml"]).read_text())
    rg = doc["regions"]
    fixed_files = [x["file"] if isinstance(x, dict) else x for x in rg["fixed"]]   # a per-axis entry is a mapping
    used = {Path(r["yaml"]).name, rg["design"], *fixed_files, *rg["keep_out"], *rg["load"].values(),
            *rg.get("solid", [])}
    on_disk = {f.name for f in out.iterdir() if f.is_file()}
    res = {"ok": bool(r["ok"]), "name": nm, "run_dir_rel": rel(out), "yaml_rel": rel(r["yaml"]),
           "files_rel": sorted(rel(out / f) for f in used),
           "other_files_rel": sorted(rel(out / f) for f in on_disk - used),
           "overwrote_existing_folder": existed,
           "voxel_size": r["voxel_size"], "voxel_auto": r["voxel_auto"], "retries": r["retries"],
           "builder_ok": r["ok"],
           "problems": [ln[len("ERROR: "):] for ln in lines if ln.startswith("ERROR: ")],
           "builder_warnings": [ln[len("WARNING: "):] for ln in lines if ln.startswith("WARNING: ")],
           "report": lines}
    from .cli import check_data
    from .config import UserError
    try:
        res["check"] = check_data(r["yaml"])
        res["warnings"] = res["check"]["warnings"]
        res["derived"] = res["check"]["derived"]
        res["assumptions"] = res["check"]["assumptions"]
    except UserError as e:
        res["check"] = None
        res["warnings"] = []
        res["check_error"] = str(e)
    except Exception as e:
        res["check"] = None
        res["warnings"] = []
        res["check_error"] = f"internal error: {type(e).__name__}: {e}"
    res.setdefault("derived", None)
    res.setdefault("assumptions", spec_assumptions(spec))
    if preference_facts:
        res["preferences_facts"] = preference_facts
    res["warnings"] = res["warnings"] + [w for w in res["builder_warnings"] if w not in res["warnings"]]
    if nm == "live":
        res["warnings"] = res["warnings"] + [
            "runs/live is the default run folder of run_topopt and the dashboard: a run "
            "started without out_dir writes its snapshots there"]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--out", required=True, help="output folder (STLs + YAML)")
    ap.add_argument("--check", action="store_true",
                    help="build into a temp folder and only run the check (write nothing to --out)")
    ap.add_argument("--voxel", type=float, help="override settings.voxel_size")
    a = ap.parse_args(argv)
    try:
        spec = yaml.safe_load(Path(a.spec).read_text())
        if not isinstance(spec, dict):
            raise SpecError("spec must be a mapping")
        r = build_spec(spec, Path(a.out), a.voxel, a.check, base=Path(a.spec).resolve().parent, base_words="the folder of the spec file")
        return 0 if r["ok"] else 1
    except SpecError as e:
        print(f"SPEC ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
