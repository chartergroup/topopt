"""Minimal print report (CONTRACTS.md section 18).

The report reads the final mesh and gives plain numbers for printing:

- overhang area, plate contact area and a support-volume estimate, for the configured build
  direction and for each of the six axis directions as options;
- bed fit, with the part extents in the build frame;
- a thinnest-member estimate (inward chords along the surface normal).

It works on the triangle mesh, not on voxels. It needs numpy and trimesh (the ray queries use the
`rtree` that the project pins). It does not change the optimizer or the mesh. All numbers are
facts or estimates: nothing here stops a run.

The section has `partial` and `coverage` (TC-2, CONTRACTS.md section 44): which of its five sub-checks (`bed_fit`, `overhang`,
`plate_contact`, `support_volume`, `thinnest_member`) ran, failed or were not run. A failure of the thinnest-member estimate does not take
the other numbers away; a failure of the rest gives `{"ok": false, "error": ...}` as before.

Method (ported from the earlier pipeline of the same author, generalized):

- Overhang rule and areas: `Study.metrics` (`pipeline/s9_print/orientation_study.py:64-90`).
- Support volume: the "gap" definition of `printability.py:46-61`: the distance from each overhang
  face down to the plate or to the part, times the projected face area.
- Thickness: `ray_thickness` (`pipeline/s9_print/thick.py`), with trimesh instead of VTK.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import check_coverage
from .config import PrintSettings

# The six axis directions: label and unit vector (up, away from the plate).
AXIS_OPTIONS = (
    ("+x", (1.0, 0.0, 0.0)), ("-x", (-1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)), ("-y", (0.0, -1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)), ("-z", (0.0, 0.0, -1.0)),
)

# The settings that are "assumed" when the YAML does not give them. The angle follows the printer type.
ASSUMABLE_KEYS = ("build_direction", "bed_mm", "printer_type")

PLATE_NORMAL = -0.99      # a plate-contact face has n.b below this (about 8 degrees from flat)
PLATE_HEIGHT_MM = 0.3     # and its centre is less than this above the lowest point
RAY_OFFSET_MM = 0.05      # a support ray starts this far below the face centre
MAX_SUPPORT_RAYS = 30000  # at most this many overhang faces get a ray (random sample, fixed seed)
SUPPORT_SEED = 0

THICKNESS_SAMPLES = 60000   # the value that printability.py gives to ray_thickness (its own default is 30000)
THICKNESS_SEED = 1
THICKNESS_PERCENTILE = 1.0
THICKNESS_MAX_MM = 12.0   # a chord that finds no wall within this length reads this value
_CHORD_START_MM = 0.02    # the chord starts this far inside the surface
_CHORD_IGNORE_MM = 0.03   # a hit closer than this to the surface point is the surface itself
_CHORD_STEP_MM = 4.0      # a ray is cast in pieces of this length (a short piece has a small box)
_EDGE_TOL = 1e-9
RAY_CHUNK = 2000          # at most this many rays in one pass
PAIR_BUDGET = 2_000_000   # and at most this many (ray, triangle) pairs in one pass (a lone ray can have more)
AREA_THRESHOLDS_MM = (0.8, 1.2, 2.0)   # `area_below_mm2`: surface area whose chord is thinner than each value
TURN_STEP_DEG = 1         # bed fit: the turn about the build axis is searched from 0 to 89 degrees in these steps
MARGIN_TIE_MM = 1e-6      # bed fit: margins that differ less than this are equal (the smaller turn wins)
BED_TURN_MIN_GAIN_MM = 10.0   # bed fit: if turn 0 fits, a turn is reported only if it gains this much margin over turn 0

# Added to `definitions` only when the run has a printer profile (a run without one keeps its output byte for byte).
HEIGHT_FIT_DEFINITION = ("the part height along the build direction (the third extent in the build frame; a turn about the build axis "
                         "does not change it) against the build height of the printer profile (CONTRACTS.md section 18.7). height_margin_mm "
                         "is build height minus part height, and height_fit is true if it is not negative. bed_fit is false if the part "
                         "does not fit the bed or is too tall. With no profile, or a profile that does not state the build height, "
                         "the height is not limited and these two keys are absent")

DEFINITIONS = {
    "overhang": "faces with n.b < -cos(angle) (strict), n the outward face normal and b the unit "
                "build direction (up); the angle is measured from the plate; plate-contact faces "
                "are not counted; a face within about 1e-15 of the limit can fall on either side "
                "(floating point)",
    "plate_contact": f"faces with n.b < {PLATE_NORMAL} whose centre is less than {PLATE_HEIGHT_MM} mm "
                     "above the lowest point along b",
    "support_volume": "estimate of the space under the overhang faces: sum over the faces of "
                      "(free gap down to the plate or to the part) x (projected face area), in mm3; "
                      "a solid-column upper bound, real supports use less material; for up to "
                      f"{MAX_SUPPORT_RAYS} faces (random sample, fixed seed) scaled to the full overhang area",
    "bed_fit": "extents_mm are the part extents in the build frame (two in-plane extents and the "
               "height). The footprint is the part projected on the plate. For a turn about the build "
               "axis of 0 to 89 degrees in 1-degree steps, the margin is the smaller of the two spare "
               "lengths: bed side minus footprint side, the sides sorted (so a turn of 90 degrees is the "
               "same as 0). For a part in the middle of the bed, the gap to each edge is half of the spare "
               "length. bed_turn_deg is the turn with the largest margin. If the part fits as it is, a "
               f"turn is reported only if it gains at least {BED_TURN_MIN_GAIN_MM:g} mm of margin; "
               "otherwise bed_turn_deg is 0. bed_footprint_mm is the footprint at that turn and "
               "bed_margin_mm that margin. bed_fit is true if the margin is not negative. If no turn "
               "fits, bed_margin_mm is negative: the footprint is that many mm longer than the bed at "
               "the best turn. The height is not limited",
    "thinnest_member": f"estimate: percentile {THICKNESS_PERCENTILE:g} of the inward chord lengths "
                       "(surface point along the negative face normal to the first wall), in mm. "
                       f"It means: {THICKNESS_PERCENTILE:g} percent of the sampled surface area has a "
                       "thinner chord. A thin feature on less than about "
                       f"{THICKNESS_PERCENTILE:g} percent of the area does not move it: see "
                       "area_below_mm2. min_mm is the shortest chord of the sample; it includes noise "
                       "from sharp edges. A value equal to max_chord_mm means that many mm or more",
    "coverage": check_coverage.DEFINITION,
    "area_below_mm2": "surface area (mm2) whose chord is thinner than each threshold in mm: the "
                      "share of the sampled chords below the threshold times the total area; "
                      "area_below_pct is the same share in percent. The sampling error of a share f "
                      "is about sqrt(f (1 - f) / samples)",
}


# ---------------------------------------------------------------------------
# directions and frames
# ---------------------------------------------------------------------------


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def axis_of(b, tol: float = 1e-9):
    """Return (axis index, sign) if `b` is along a world axis, else None."""
    b = np.asarray(b, dtype=float)
    k = int(np.argmax(np.abs(b)))
    if abs(abs(b[k]) - 1.0) <= tol and np.abs(np.delete(b, k)).max() <= tol:
        return k, (1.0 if b[k] > 0 else -1.0)
    return None


def direction_label(b) -> str:
    """'+z' for an axis direction, else the vector as text."""
    ax = axis_of(b)
    if ax is not None:
        return ("+" if ax[1] > 0 else "-") + "xyz"[ax[0]]
    return "[" + ", ".join(f"{float(v):.4g}" for v in b) + "]"


def build_frame(b) -> np.ndarray:
    """Rows u, v, b of an orthonormal build frame.

    For an axis direction, u and v are the other two world axes in the order x, y, z; v changes its
    sign where needed so that u x v = b (a right-handed frame; the extents do not change). For
    another direction, u is perpendicular to b and to the world axis that is least aligned with b.
    """
    b = _unit(b)
    ax = axis_of(b)
    if ax is not None:
        k = ax[0]
        others = [i for i in range(3) if i != k]
        u, v = np.eye(3)[others[0]], np.eye(3)[others[1]]
        if np.dot(np.cross(u, v), b) < 0:
            v = -v
        return np.stack([u, v, b])
    ref = np.eye(3)[int(np.argmin(np.abs(b)))]
    u = _unit(np.cross(b, ref))
    v = np.cross(b, u)
    return np.stack([u, v, b])


def extents_in_frame(points, b) -> np.ndarray:
    """The extents (in-plane u, in-plane v, height along b) of a point set in the build frame."""
    q = np.asarray(points, dtype=float) @ build_frame(b).T
    return q.max(axis=0) - q.min(axis=0)


def fits_bed(extents_uv, bed_mm) -> bool:
    """True if the two in-plane extents fit the bed (sorted against sorted, so a 90 degree turn is allowed)."""
    e = sorted(float(x) for x in extents_uv[:2])
    d = sorted(float(x) for x in bed_mm)
    return bool(e[0] <= d[0] + 1e-9 and e[1] <= d[1] + 1e-9)


def _footprint_hull(points, b) -> np.ndarray:
    """The convex hull (n, 2) of the part projected on the plate, in the build frame."""
    xy = (np.asarray(points, dtype=float) @ build_frame(b).T)[:, :2]
    if len(xy) <= 4:
        return xy
    try:
        from scipy.spatial import ConvexHull
        return xy[ConvexHull(xy).vertices]
    except Exception:  # noqa: BLE001 - a flat or one-line footprint has no hull: its extreme points do
        keep = {int(i) for i in (xy[:, 0].argmin(), xy[:, 0].argmax(), xy[:, 1].argmin(), xy[:, 1].argmax())}
        return xy[sorted(keep)]


def bed_fit_turn(points, b, bed_mm, build_height_mm=None) -> dict:
    """Bed fit of a point set for the build direction `b`, with a turn about the build axis.

    The turn is searched from 0 to 89 degrees in `TURN_STEP_DEG` steps on the convex hull of the
    footprint (as `footprint()` of the pipeline, `orientation_study.py:26-40`). At each turn the
    margin is the smaller of the two spare lengths: bed side minus footprint side, with the two
    sides sorted against the two bed sides (so a turn of 90 degrees is the same as 0). The best turn
    has the largest margin; of the turns within `MARGIN_TIE_MM` of that margin, the smallest turn wins.
    If turn 0 fits, the best turn is reported only if its margin is at least `BED_TURN_MIN_GAIN_MM`
    larger than the margin at turn 0; otherwise the report keeps turn 0. If turn 0 does not fit, the
    best turn is reported.
    Return the keys of a report entry: `bed_fit` (the margin at the reported turn is not negative),
    `bed_turn_deg`, `bed_footprint_mm` and `bed_margin_mm` (at that turn; negative when no turn fits:
    the footprint is that many mm longer than the bed).
    With `build_height_mm` (the printer profile states it), the part height along `b` is also compared: the keys `height_fit` and
    `height_margin_mm` (build height minus part height) are added, and `bed_fit` is false if the part is too tall. Without it the
    height is not limited and the keys are the ones above.
    """
    hull = _footprint_hull(points, b)
    ang = np.radians(np.arange(0, 90, TURN_STEP_DEG, dtype=float))
    c, sn = np.cos(ang), np.sin(ang)
    x, y = hull[:, 0:1], hull[:, 1:2]
    w = np.ptp(x * c + y * sn, axis=0)          # footprint sides at each turn (the same turn as the pipeline)
    h = np.ptp(-x * sn + y * c, axis=0)
    d = sorted(float(v) for v in bed_mm)
    margin = np.minimum(d[0] - np.minimum(w, h), d[1] - np.maximum(w, h))
    i = int(np.argmax(margin >= margin.max() - MARGIN_TIE_MM))      # the first turn within the tie
    if margin[0] >= -1e-9 and margin[i] - margin[0] < BED_TURN_MIN_GAIN_MM - 1e-9:
        i = 0                                   # turn 0 fits and a turn gains too little to be worth a report
    out = {"bed_fit": bool(margin[i] >= -1e-9),
           "bed_turn_deg": int(round(np.degrees(ang[i]))),
           "bed_footprint_mm": [round(float(w[i]), 3), round(float(h[i]), 3)],
           "bed_margin_mm": round(float(margin[i]), 3) + 0.0}
    if build_height_mm is not None:
        spare = float(build_height_mm) - float(extents_in_frame(points, b)[2])
        out["height_fit"] = bool(spare >= -1e-9)
        out["height_margin_mm"] = round(spare, 3) + 0.0
        out["bed_fit"] = bool(out["bed_fit"] and out["height_fit"])
    return out


# ---------------------------------------------------------------------------
# mesh arrays
# ---------------------------------------------------------------------------


class _Geometry:
    """Arrays of a triangle mesh. A mesh with negative volume is treated as inside out."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.V = np.asarray(mesh.vertices, dtype=float)
        self.F = np.asarray(mesh.faces)
        self.T = self.V[self.F]                      # (n, 3, 3)
        cross = np.cross(self.T[:, 1] - self.T[:, 0], self.T[:, 2] - self.T[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        self.A = 0.5 * area2
        self.N = np.zeros_like(cross)
        ok = area2 > 0
        self.N[ok] = cross[ok] / area2[ok, None]
        volume = float(np.einsum("ij,ij->i", self.T[:, 0], np.cross(self.T[:, 1], self.T[:, 2])).sum() / 6.0)
        self.inside_out = volume < 0
        if self.inside_out:
            self.N = -self.N
        self.C = self.T.mean(axis=1)
        self.total_area = float(self.A.sum())


# ---------------------------------------------------------------------------
# ray passes with a bound on the (ray, triangle) pairs
# ---------------------------------------------------------------------------


def _box_query(tree, lo, hi):
    """`(ids, counts)`: the triangles whose boxes meet each box (`lo`, `hi` are (n, 3)); `ids` are grouped by box."""
    if hasattr(tree, "intersection_v"):
        ids, counts = tree.intersection_v(lo, hi)
        return np.asarray(ids, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    lists = [list(tree.intersection(np.r_[lo[i], hi[i]])) for i in range(len(lo))]
    counts = np.array([len(c) for c in lists], dtype=np.int64)
    ids = np.fromiter((t for c in lists for t in c), dtype=np.int64, count=int(counts.sum()))
    return ids, counts


def _pair_slices(tree, lo, hi, state, stats=None):
    """Yield `(r0, r1, ids, counts)`: slices of the rays with their candidate triangles.

    A slice has at most `RAY_CHUNK` rays and at most `PAIR_BUDGET` (ray, triangle) pairs; a lone ray
    that has more pairs runs alone. The memory of a pass grows with the pairs, so a mesh with long
    sliver triangles (a fan cap, with big boxes) gets short slices and a normal mesh gets long ones.
    The box query itself runs on groups of rays; `state["q"]` is the group size. It starts small and
    follows the pairs per ray that the last group had, so no query returns much more than the budget.
    `stats`, if given, collects `passes` and the largest slice: `max_pairs`, `max_rays`.
    """
    n = len(lo)
    s = 0
    while s < n:
        k = int(max(1, min(state["q"], n - s, RAY_CHUNK)))
        ids, counts = _box_query(tree, lo[s:s + k], hi[s:s + k])
        cum = np.cumsum(counts)
        r0 = 0
        while r0 < k:
            base = int(cum[r0 - 1]) if r0 else 0
            r1 = int(np.searchsorted(cum, base + PAIR_BUDGET, side="right"))
            r1 = min(max(r1, r0 + 1), k, r0 + RAY_CHUNK)
            pairs = int(cum[r1 - 1]) - base
            if stats is not None:
                stats["passes"] = stats.get("passes", 0) + 1
                stats["max_pairs"] = max(stats.get("max_pairs", 0), pairs)
                stats["max_rays"] = max(stats.get("max_rays", 0), r1 - r0)
            yield s + r0, s + r1, ids[base:base + pairs], counts[r0:r1]
            r0 = r1
        mean = max(1.0, float(counts.mean()))
        state["q"] = int(max(1, min(PAIR_BUDGET / mean, RAY_CHUNK)))
        s += k


# ---------------------------------------------------------------------------
# overhang, plate contact, support volume
# ---------------------------------------------------------------------------


def _support_volume(g: _Geometry, b, idx, zc, stats=None):
    """Gap definition (printability.py:46-61). Return (volume mm3, number of rays).

    `idx` are the overhang faces, `zc` the height of each face centre above the lowest point.
    Each face casts one ray down (along -b) from 0.05 mm below its centre. The gap is the distance
    from the centre to the first hit on the part, or the height above the plate if nothing is hit.
    The rays go to the trimesh engine in slices of at most `RAY_CHUNK` rays and about `PAIR_BUDGET`
    candidate pairs (the pairs are counted with the box query of the rtree).
    """
    if len(idx) == 0:
        return 0.0, 0
    if len(idx) > MAX_SUPPORT_RAYS:
        pick = np.random.default_rng(SUPPORT_SEED).choice(idx, MAX_SUPPORT_RAYS, replace=False)
    else:
        pick = idx
    if axis_of(b) is not None:
        mesh, bl, centres = g.mesh, b, g.C[pick]        # the rays follow a world axis: a thin box per ray
    else:
        import trimesh
        frame = build_frame(b)                          # turn the part so that the rays follow z
        mesh = trimesh.Trimesh(g.V @ frame.T, g.F, process=False)
        bl, centres = np.array([0.0, 0.0, 1.0]), g.C[pick] @ frame.T
    origins = centres - RAY_OFFSET_MM * bl
    directions = np.tile(-bl, (len(pick), 1))
    gap = zc[pick].copy()                               # nothing hit: the part rests on the plate
    # the box of each ray: from its origin down to the lowest point of the part (slightly padded)
    low = float((np.asarray(mesh.vertices, dtype=float) @ bl).min())
    end = origins - bl * ((origins @ bl - low) + 1e-3)[:, None]
    lo = np.minimum(origins, end) - 1e-5
    hi = np.maximum(origins, end) + 1e-5
    for r0, r1, _ids, _counts in _pair_slices(mesh.triangles_tree, lo, hi, {"q": 64}, stats):
        loc, ray, _tri = mesh.ray.intersects_location(origins[r0:r1], directions[r0:r1], multiple_hits=False)
        if len(ray):
            hit = ray + r0
            gap[hit] = (centres[hit] - loc) @ bl
    weight = np.maximum(-(g.N[pick] @ b), 0.0) * g.A[pick]
    scale = g.A[idx].sum() / g.A[pick].sum()
    return float((gap * weight).sum() * scale), int(len(pick))


def measure_direction(g: _Geometry, b, angle_deg: float, bed_mm, support: bool = True, stats=None, build_height_mm=None) -> dict:
    """The overhang, contact, support and bed-fit numbers for one build direction `b`. `build_height_mm`: see `bed_fit_turn`."""
    b = _unit(b)
    z = g.V @ b
    zmin = float(z.min())
    nb = g.N @ b
    zc = g.C @ b - zmin
    over = nb < -math.cos(math.radians(angle_deg))
    plate = (nb < PLATE_NORMAL) & (zc < PLATE_HEIGHT_MM)
    ov = over & ~plate
    overhang_area = float(g.A[ov].sum())
    ext = extents_in_frame(g.V, b)
    out = {
        "label": direction_label(b),
        "direction": [float(v) + 0.0 for v in b],
        "extents_mm": [round(float(v), 3) for v in ext],
        "height_mm": round(float(ext[2]), 3),
        **bed_fit_turn(g.V, b, bed_mm, build_height_mm),
        "overhang_area_mm2": round(overhang_area, 3),
        "overhang_fraction": round(overhang_area / g.total_area, 5) if g.total_area > 0 else 0.0,
        "plate_contact_area_mm2": round(float(g.A[plate].sum()), 3),
    }
    if support:
        vol, rays = _support_volume(g, b, np.flatnonzero(ov), zc, stats)
        out["support_volume_mm3"] = round(vol, 1)
        out["support_rays"] = rays
    else:
        out["support_volume_mm3"] = None
        out["support_rays"] = 0
    return out


# ---------------------------------------------------------------------------
# thickness: inward chords
# ---------------------------------------------------------------------------


def _first_hits(g: _Geometry, origin, direction, d_max: float, step: float = _CHORD_STEP_MM,
                chunk: int | None = None, stats=None):
    """Distance along each ray to the first wall, by short pieces, in batches of `chunk` rays.

    The batches bound the memory: a batch has at most `chunk` rays, and a pass of a batch has at
    most `PAIR_BUDGET` (ray, triangle) pairs (see `_pair_slices`). The result does not depend on
    the batch size or the budget. `chunk` defaults to `RAY_CHUNK`. `stats` collects the passes.
    """
    chunk = chunk or RAY_CHUNK
    n = len(origin)
    state = {"q": 64}
    parts = [_first_hits_batch(g, origin[s:s + chunk], direction[s:s + chunk], d_max, step, state, stats)
             for s in range(0, max(n, 1), chunk)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _first_hits_batch(g: _Geometry, origin, direction, d_max: float, step: float, state, stats=None):
    """Distance along each ray of one batch to the first wall, by short pieces.

    The ray starts at `origin` and goes along the unit `direction`. A hit closer than 0.03 mm is the
    surface itself and does not count. Return (distance, near): `distance` is `inf` where nothing is
    hit within `d_max`; `near` marks a ray that has a hit between 0.02 and 0.03 mm and no other hit.

    trimesh casts a ray with the box of the whole ray, which is big for a ray that is not along an
    axis. Here each piece of `step` mm has its own small box. The box query is the one of the
    mesh's rtree (`mesh.triangles_tree`), the hit test is the Moeller-Trumbore test.
    """
    tree = g.mesh.triangles_tree
    n = len(origin)
    dist = np.full(n, np.inf)
    near = np.zeros(n, dtype=bool)
    active = np.arange(n)
    a = _CHORD_START_MM
    while a < d_max and len(active):
        e = min(a + step, d_max)
        o, d = origin[active], direction[active]
        p0, p1 = o + d * a, o + d * e
        lo = np.minimum(p0, p1) - 1e-6
        hi = np.maximum(p0, p1) + 1e-6
        t = np.full(len(active), np.inf)
        nr = np.zeros(len(active), dtype=bool)
        for r0, r1, ids, counts in _pair_slices(tree, lo, hi, state, stats):
            if len(ids) == 0:
                continue
            ray = np.repeat(np.arange(r1 - r0), counts)
            dv = d[r0:r1][ray]
            v0 = g.T[ids, 0]
            e1 = g.T[ids, 1] - v0
            e2 = g.T[ids, 2] - v0
            pvec = np.cross(dv, e2)
            det = np.einsum("ij,ij->i", e1, pvec)
            good = np.abs(det) > 1e-14
            inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
            tvec = o[r0:r1][ray] - v0
            u = np.einsum("ij,ij->i", tvec, pvec) * inv
            qvec = np.cross(tvec, e1)
            v = np.einsum("ij,ij->i", dv, qvec) * inv
            s = np.einsum("ij,ij->i", e2, qvec) * inv
            hit = good & (u >= -_EDGE_TOL) & (v >= -_EDGE_TOL) & (u + v <= 1.0 + _EDGE_TOL) \
                & (s >= a - 1e-9) & (s <= e + 1e-9)
            far = hit & (s > _CHORD_IGNORE_MM)
            np.minimum.at(t[r0:r1], ray[far], s[far])
            nr[r0:r1][ray[hit & ~far]] = True
        found = np.isfinite(t)
        dist[active[found]] = t[found]
        near[active] |= nr
        active = active[~found]
        a = e
    near &= ~np.isfinite(dist)
    return dist, near


def thickness_estimate(g: _Geometry, samples: int = THICKNESS_SAMPLES, seed: int = THICKNESS_SEED,
                       percentile: float = THICKNESS_PERCENTILE, max_mm: float = THICKNESS_MAX_MM,
                       return_chords: bool = False, stats=None):
    """The thinnest-member estimate: a low percentile of inward chord lengths on the surface.

    Surface points are sampled by area (fixed seed). At each point a chord goes along the negative
    face normal to the first wall (at most `max_mm`; no wall gives `max_mm`). A point whose chord
    only finds the surface itself is dropped (the pipeline does the same).

    The percentile alone does not see a thin feature on a small part of the surface, so the result
    also has `area_below_mm2` and `area_below_pct`: for each threshold of `AREA_THRESHOLDS_MM`, the
    area whose chord is thinner (each sample stands for the same part of the total area).
    """
    import trimesh

    pts, fi = trimesh.sample.sample_surface(g.mesh, samples, seed=seed)
    normals = g.N[fi]
    keep = np.linalg.norm(normals, axis=1) > 0.5          # a degenerate face has no normal
    pts, normals = pts[keep], normals[keep]
    dist, near = _first_hits(g, pts, -normals, max_mm, stats=stats)
    chords = np.where(np.isfinite(dist), dist, max_mm)
    chords = chords[~near]
    if len(chords) == 0:
        out = {"thinnest_member_mm": None, "min_mm": None, "samples": 0, "percentiles_mm": None,
               "area_below_mm2": None, "area_below_pct": None}
    else:
        p = {f"p{q:g}": round(float(np.percentile(chords, q)), 3) for q in (0.1, 1.0, 5.0, 50.0)}
        share = {f"{thr:.1f}": float((chords < thr).mean()) for thr in AREA_THRESHOLDS_MM}
        out = {"thinnest_member_mm": round(float(np.percentile(chords, percentile)), 3),
               "min_mm": round(float(chords.min()), 3), "samples": int(len(chords)),
               "percentiles_mm": p,
               "area_below_mm2": {k: round(v * g.total_area, 2) for k, v in share.items()},
               "area_below_pct": {k: round(100.0 * v, 3) for k, v in share.items()}}
    out.update({
        "percentile": percentile, "seed": seed, "max_chord_mm": max_mm,
        "method": "inward chord along the surface normal on the final mesh",
        "estimate": True,
    })
    return (out, chords) if return_chords else out


def thickness_failed(exc: Exception) -> dict:
    """The `thinnest_member` block of an estimate that stopped with an exception (TC-2): the shape of "no chord measured", plus `error`.

    The estimate is one sub-check of the report. A failure of it must not take the overhang, plate-contact, support and bed-fit numbers away
    (they are not computed from it), so `build_report` keeps them and names this one in `coverage.failed`.
    """
    return {"thinnest_member_mm": None, "min_mm": None, "samples": 0, "percentiles_mm": None,
            "area_below_mm2": None, "area_below_pct": None,
            "percentile": THICKNESS_PERCENTILE, "seed": THICKNESS_SEED, "max_chord_mm": THICKNESS_MAX_MM,
            "method": "inward chord along the surface normal on the final mesh", "estimate": True,
            "error": " ".join(f"{type(exc).__name__}: {exc}".split())}


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def build_report(mesh, settings: PrintSettings | None = None, *, thickness_samples: int = THICKNESS_SAMPLES) -> dict:
    """The `print` section of metrics.json for a trimesh mesh (CONTRACTS.md section 18).

    `settings` are the print settings; None gives the defaults (all assumed). The mesh is in mm.
    """
    t0 = time.perf_counter()
    settings = settings or PrintSettings()
    g = _Geometry(mesh)
    angle = settings.angle_deg
    support = settings.support_needed
    b0 = _unit(settings.build_direction)

    cache: dict[str, dict] = {}

    def entry(b):
        key = direction_label(b)
        if axis_of(b) is None or key not in cache:
            cache[key] = measure_direction(g, b, angle, settings.bed_mm, support, build_height_mm=settings.build_height_mm)
        return cache[key]

    options = [entry(np.array(v)) for _label, v in AXIS_OPTIONS]
    configured = entry(b0)
    try:
        thinnest = thickness_estimate(g, thickness_samples)
    except Exception as exc:  # noqa: BLE001 - one sub-check; the others stand, and `coverage` names this one (TC-2)
        import logging
        logging.getLogger("topopt.print_report").warning("thinnest-member estimate failed: %s", exc)
        thinnest = thickness_failed(exc)
    assumed = [k for k in ASSUMABLE_KEYS if k not in settings.given and k not in settings.derived]
    angle_source = "yaml" if settings.self_supporting_angle_deg is not None \
        else f"default of printer_type {settings.printer_type}"
    section = {
        "ok": True,
        "settings": {
            "build_direction": configured["direction"],
            "build_direction_label": configured["label"],
            "bed_mm": list(settings.bed_mm),
            "printer_type": settings.printer_type,
            "self_supporting_angle_deg": angle,
            "angle_source": angle_source,
            "assumed": assumed,
        },
        "support_needed": support,
        "total_area_mm2": round(g.total_area, 3),
        "configured": configured,
        "options": options,
        "thinnest_member": thinnest,
        "definitions": DEFINITIONS,
    }
    if assumed:
        section["settings"]["assumed_note"] = ("these keys are not in the YAML print block: the report "
                                              "uses the defaults of CONTRACTS.md section 18")
    if settings.profile is not None:   # a run without a printer_model keeps the section as it was
        section["settings"].update(profile_settings(settings))
        section["definitions"] = {**DEFINITIONS, "height_fit": HEIGHT_FIT_DEFINITION}
    if not support:
        section["support_note"] = ("no support needed: the powder bed supports the part "
                                   f"(printer_type {settings.printer_type}); the overhang area is still given")
    if g.inside_out:
        section["warning"] = "the mesh has negative volume: the normals were flipped for this report"
    section["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return check_coverage.attach(section, check_coverage.from_print_section(section), after="ok")


def settings_from_run(run: dict) -> PrintSettings:
    """The print settings of a run (key `print` of run.json). A run without it uses the defaults, all assumed."""
    return PrintSettings.from_dict(run.get("print"))


def section_for_run(run: dict, mesh) -> dict:
    """The `print` section for `post.finish`. A failure gives `{"ok": false, "error": ...}`, not an exception.

    The report is an add-on to the run: a fault in it leaves the mesh and the other metrics in place.
    The error text is a fact in metrics.json. The section has `partial` and `coverage` (TC-2): a section that could not be made is
    `partial: true` with the one failed check `print_report`.
    """
    try:
        return build_report(mesh, settings_from_run(run))
    except Exception as exc:  # noqa: BLE001 - recorded as a fact, see the docstring
        import logging
        logging.getLogger("topopt.post").warning("print report failed: %s", exc)
        error = f"{type(exc).__name__}: {exc}"
        return check_coverage.attach({"ok": False, "error": error}, check_coverage.from_failed_print_section(error), after="ok")


def profile_settings(settings: PrintSettings) -> dict:
    """The keys that name the printer profile of a run, for the report `settings` and for `topopt check` (CONTRACTS.md section 18.7)."""
    p = settings.profile
    return {
        "printer_model": p.id, "printer_name": p.name, "catalog_version": p.catalog_version or None,
        "printer_origin": p.origin, "profile_from": p.origin_text,
        "build_height_mm": p.build_height_mm,
        "derived": {k: settings.key_source(k) for k in settings.derived},
    }


def check_facts(points, settings: PrintSettings | None = None) -> dict:
    """Bed-fit facts for a point set (the 8 corners of the design-space box), before a run.

    Used by `topopt check`. The points bound the part, so a fit is certain and a misfit may not
    remain after the optimizer removes material.
    """
    settings = settings or PrintSettings()
    pts = np.asarray(points, dtype=float)
    b0 = _unit(settings.build_direction)

    def one(b):
        ext = extents_in_frame(pts, b)
        return {"label": direction_label(b), "direction": [float(v) + 0.0 for v in _unit(b)],
                "extents_mm": [round(float(v), 3) for v in ext], **bed_fit_turn(pts, b, settings.bed_mm, settings.build_height_mm)}

    out = {
        "bed_mm": list(settings.bed_mm),
        "assumed": [k for k in ASSUMABLE_KEYS if k not in settings.given and k not in settings.derived],
        "printer_type": settings.printer_type,
        "box_extents_mm": [round(float(v), 3) for v in pts.max(axis=0) - pts.min(axis=0)],
        "configured": one(b0),
        "options": [one(np.array(v)) for _label, v in AXIS_OPTIONS],
        "note": "the box is the design space (design, fixed and load voxels); the part lies inside it",
    }
    if settings.profile is not None:   # a run without a printer_model keeps these facts as they were
        out.update(profile_settings(settings))
    return out


def mask_box_corners(mask, origin, h) -> np.ndarray:
    """The 8 corners (mm) of the box around the True voxels of a (nx, ny, nz) mask."""
    lo, hi = [], []
    for k in range(3):
        other = tuple(j for j in range(3) if j != k)
        where = np.flatnonzero(mask.any(axis=other))
        lo.append(float(origin[k]) + h * where[0])
        hi.append(float(origin[k]) + h * (where[-1] + 1))
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _mm(values) -> str:
    """One format for lengths in the text lines: 4 significant digits, joined with ' x '."""
    return " x ".join(f"{float(v):.4g}" for v in values)


def fit_phrase(entry: dict) -> str:
    """'fits', 'fits turned N degrees (margin M mm)' or 'does not fit at any turn (best: N degrees, X mm over)'.

    An entry with a build height (`height_fit`) can also be too tall: 'fits the bed but is X mm too tall for the build height',
    or the first phrase and ', and is X mm too tall for the build height'.
    """
    turn, margin = entry.get("bed_turn_deg") or 0, entry.get("bed_margin_mm")
    tall = f"is {-float(entry['height_margin_mm']):.4g} mm too tall for the build height" if entry.get("height_fit") is False else ""
    footprint_fits = entry["bed_fit"] or (bool(tall) and margin is not None and float(margin) >= -1e-9)
    if not footprint_fits:
        text = f"does not fit at any turn (best: {turn} degrees, {-float(margin):.4g} mm over)"
        return f"{text}, and {tall}" if tall else text
    if tall:
        return f"fits the bed but {tall}"
    return f"fits turned {turn} degrees (margin {float(margin):.4g} mm)" if turn > 0 else "fits"


def check_lines(facts: dict) -> list[str]:
    """Text lines for `topopt check`: the bed-fit facts of `check_facts`."""
    bed = _mm(facts["bed_mm"])
    cfg = facts["configured"]
    if facts.get("printer_model"):    # a printer profile: name it, and say which keys it filled (CONTRACTS.md section 18.7)
        derived = list(facts.get("derived") or {})
        parts = ([f"from the profile: {', '.join(derived)}"] if derived else []) \
            + ([f"assumed: {', '.join(facts['assumed'])}"] if facts["assumed"] else [])
        word = "; ".join(parts) or "all from the YAML print block"
        lines = [f"print settings: bed {bed} mm, build direction {cfg['label']}, printer: {facts['printer_name']} (profile "
                 f"{facts['printer_model']}, {facts['profile_from']}), type {facts['printer_type']} ({word})"]
    else:
        word = "assumed: " + ", ".join(facts["assumed"]) if facts["assumed"] else "all from the YAML print block"
        lines = [f"print settings: bed {bed} mm, build direction {cfg['label']}, printer {facts['printer_type']} ({word})"]
    fit = [o["label"] + (f" (turned {o['bed_turn_deg']} degrees)" if (o.get("bed_turn_deg") or 0) > 0 else "")
           for o in facts["options"] if o["bed_fit"]]
    nofit = [o["label"] for o in facts["options"] if not o["bed_fit"]]
    foot = cfg["bed_footprint_mm"]
    head = (f"design-space box {_mm(facts['box_extents_mm'])} mm; with {cfg['label']} up it "
            f"{fit_phrase(cfg)}; footprint {_mm(foot)} mm, height {float(cfg['extents_mm'][2]):.4g} mm")
    if cfg.get("height_fit") is not None:
        head += f" (build height {float(facts['build_height_mm']):.4g} mm, margin {float(cfg['height_margin_mm']):.4g} mm)"
    lines.append("print bed fit:  " + head
                 + ("; fits with " + " ".join(fit) + " up" if fit else "")
                 + ("; does not fit with " + " ".join(nofit) + " up" if nofit else ""))
    return lines
