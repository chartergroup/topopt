"""Apply the marks of the refine page to the finished part (CONTRACTS.md section 29).

This module edits the finished part (`result/final.stl`) at the marks of the Refine page. It gives facts and options.
The agent decides. It writes into a NEW folder: the edited STL (`edited.stl`) and `apply_marks.json`. The run
folder, `final.stl` and every input file stay as they are.

The method is the one of the pipeline stage `s7_cleanup` (`apply_marks3.py`, `lib2.py`), on a signed distance field
around each mark, with the values of the page:

- `remove`: a cut that protects the parent (medial balls), then the parent check.
- `add`: a closing with a ball. If the closing adds little, a neck offset.
- `smooth`: an isotropic fairing for a point, a web fairing for a stroke (`pick_smooth_mode`; both use `radius_mm` and
  `strength`). `smooth_mode: tube` (asked for; auto does not pick it) rebuilds a whole strut as a faired tube; it does not use
  `radius_mm`, and the report gives its planned volume and its reach.

Each mark is one step. A step takes the current mesh, builds a signed distance field in a crop box around the mark,
edits the field, and puts the new surface back inside a zone around the changed nodes. The old surface and the new
surface are two triangulations of one surface, so the seam is not left to a boolean: the two open pieces are cut apart and
their boundary loops are sewn (edge to edge by projection where the loops are close, a strip and a weld elsewhere). The
plain boolean is the fallback, and the step facts (`seam`) say which route ran. Outside the zone the triangles stay as they
are. The report gives the surface facts of `topopt verify` (fold edges, shortest edge, closed manifold) for the part before
and after (`totals.surface_quality`). A step that gives a mesh that is not closed, or that splits the part into bodies that
are not slivers, is rolled back, and the report says why. The fixed, load and keep-out regions are frozen: an edit near them is clipped
(the weight of the edit falls to 0), the exact regions of the YAML are checked in each step (a change inside them is
undone, and the step is rolled back when that does not give a valid part), and the report gives the clip as a fact.

Python: numpy, scipy, scikit-image, trimesh and manifold3d. Run it as a script:
`python -m topopt.apply_marks RUN_DIR MARKS.json [--out DIR]` (JSON on stdout).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import feedback as fb
from .config import UserError

APPLY_FORMAT = 1
KIND_ORDER = ("remove", "smooth", "add")     # apply_marks3.py:115 (REMOVE, then SMOOTH, then ADD)
EMPTY, DESIGN, FIXED, LOAD, KEEP_OUT = 0, 1, 2, 3, 4
BAND_MM = 8.0                                # the field is clipped to +-8 mm (apply_marks3.py: np.clip(T, -8, 8))
CHANGE_TOL_MM = 1e-4                         # a node counts as changed when the field moves by more than this
EDIT_LEVEL = 1e-5                            # marching cubes level (mm); a node exactly at 0 gives degenerate faces
PAGE_RADIUS_DEFAULT = 4.0                    # the page default; feedback.py uses it when radius_mm is missing

# Values that the page does not record. Each one is the value of the pipeline (stage s7_cleanup), named as a default.
# The anchor is the line of the pipeline source (snapshot d43bcf3). `value` is used when the call gives none.
DEFAULTS: dict[str, tuple[float, str]] = {
    "close_mm": (3.0, "apply_marks3.py:89 do_add(r=3.0): the ball of the closing"),
    "neck_offset_mm": (0.5, "apply_marks3.py:91 off_mm = PRM(mid, 'offset', 0.5): used only when the mark has no amount_mm"),
    "neck_min_mm3": (51.8, "apply_marks3.py:105 the neck offset when the closing adds less than 0.4 g (line 104); "
                           "lib2.py:10 RHO_G = 0.0078 * 0.99 g/mm3, so 0.4 g = 51.8 mm3"),
    "parent_tol_mm": (0.2, "lib2.py:148 parent_check(tol=0.2)"),
    "frozen_keep_mm": (3.0, "lib2.py:175 fz_w: the weight is 0 within 3 mm of a frozen part; apply_marks3.py:38 frozen_keep=3.0"),
    "frozen_full_mm": (6.0, "lib2.py:175 fz_w: (distance - 3) / 3, so the weight is 1 at 6 mm and beyond"),
    "remove_close_mm": (2.0, "apply_marks3.py:38 close_r=2.0: closing of the parent envelope"),
    "remove_margin_mm": (0.5, "apply_marks3.py:38 margin=0.5: growth of the parent envelope"),
    "remove_excl_mm": (2.0, "apply_marks3.py:38 excl_pad=2.0: medial balls centred this close to the feature are not parent"),
    "remove_dom_mm": (2.0, "apply_marks3.py:38 dom_pad=2.0: the removal domain reaches this far from the feature"),
    "foot_rho_mm": (0.9, "lib2.py:91 build_region2(foot_rho=0.9): a thin leftover next to the cut is a foot (lines 110-117)"),
    "foot_max_mm3": (32.4, "lib2.py:91 foot_max_g=0.25 g, line 117; lib2.py:10 RHO_G: 0.25 g = 32.4 mm3"),
    "cut_smooth_mm": (0.35, "lib2.py:126 cutter(sm=0.35): smoothing of the cut field"),
    "polish_mm": (0.6, "lib2.py:139 polish_sigma=0.6: scar polish, never inside the parent envelope"),
    "polish_zone_mm": (2.5, "lib2.py:138 zone: 2.5 mm around the cut"),
    "iso_sigma_mm": (1.0, "apply_marks3.py:46 op_iso(sg=1.0) and :83 PRM(mid, 'isigma', 1.0); the mark strength multiplies it"),
    "fair_sigma_mm": (4.0, "apply_marks3.py:84 PRM(mid, 'sigma', 4.0); lib2.py:239 op_fair2(sa=4.0); the mark strength multiplies it"),
    "tube_window_mm": (10.0, "lib2.py:181 op_smooth2(WIN=10.0) and apply_marks3.py:82 PRM(mid, 'win', 10.0); the mark strength multiplies it"),
    "tube_ext_max_mm": (12.0, "lib2.py:181 op_smooth2(ext_max=24.0), loop at line 204; half of it here, to keep the crop box small"),
    "tube_blend_end_mm": (2.0, "lib2.py:229 smooth minimum k = 2.0 where the distance to an end of the tube is under 4 mm, else 0.05"),
    "web_flatness": (1.75, "s5_candidates.py:17 and :42: FAIR when the flatness of the section is 1.75 or more (SMOOTH below)"),
    "path_step_mm": (1.0, "lib2.py:176 resample(path, step=1.0)"),
    "strength_default": (1.0, "apply_marks3.py:46 and :83 isigma 1.0: a mark with no strength gets the pipeline sigma"),
}
# Values that the code derives. They are not pipeline defaults, and the report names them.
TUBE_MIN_ARC_MM = 6.0       # a stroke shorter than this is not a strut: the isotropic fairing is used
PITCH_DIVISOR = 4.0         # default crop pitch = voxel size of the run / 4, kept between 0.2 and 0.5 mm
PITCH_DEFAULT_RANGE = (0.2, 0.5)    # 0.5 mm is the grid of the pipeline (H); finer than 0.2 mm costs time, not accuracy
FROZEN_TOL_MM3 = 0.01      # the exact frozen check calls a volume below this noise (float32 STL, field error)
CROP_MARGIN_MM = 3.0        # a crop box reaches at most this far outside the bounds of the part
BYTES_PER_NODE = 200        # peak memory of one crop box, estimate; measured on the hook: 200 B per node (623 MB at 2.2 M nodes, 1060 MB at 4.4 M)
FIXED_MB = 300              # the process needs about this much more (Python, the mesh, the booleans): measured 150 to 350 MB
NEAR_ZERO_NODES = 2.0       # a node can move the surface when the field is within this many nodes of the zero level
SEAM_MATCH_NODES = 3.0      # the old part and the new surface are one seam when each boundary loop stays within this many nodes
#                             of its partner. The zone ends 2.5 nodes beyond the changed nodes; there the two surfaces differ by
#                             the error of the field: 0.065 mm at most on the hook and the ladle (0.13 node at 0.5 mm). Where the
#                             wall of the zone meets a flat face at a small angle, the loops move by that error divided by the
#                             tangent of the angle: 0.7 mm (2 nodes) on a bump next to a flat face (tests/test_apply_marks_seam.py)
SEAM_REACH_NODES = 1.0      # the inner vertices of the new surface within this many nodes of the seam follow the move of the seam
#                             (a share that falls to 0 at the reach), so a thin triangle next to the seam does not turn over
SEAM_WELD_PASSES = 20       # the weld repeats until no edge collapses; a collapse removes a vertex, so it ends; this only guards a loop
#                             (the seam facts say `weld_stopped_by_guard` when a pass still collapsed at the end)
SEAM_NEAREST_VERTICES = 3   # a point is tested against the edges at this many nearest vertices of a loop: on a sharp turn or a long
#                             edge the nearest vertex is not the end of the nearest edge. A wrong edge is caught by the order test
#                             of the projection (every step along the loop must advance), then the loop pair goes to the strip
SEAM_FOLLOW_SOFTENING = 0.05  # the inner vertices follow the seam with weights 1 / (d + 0.05 x reach): 5 % of the reach keeps a vertex
#                             that lies on a vertex of the loop from taking a weight of infinity


def _area_floor(V: np.ndarray, length: float) -> float:
    """The area (mm2) below which a triangle has no normal that float32 STL coordinates define: a vertex is known to the float32
    spacing of the largest coordinate (7.6e-6 mm at 80 mm, 3.1e-5 mm at 300 mm), a triangle of edge `length` (the grid pitch)
    then has an area known to length x spacing. A formula of the data, not a setting."""
    return float(length * np.spacing(np.float32(max(float(np.abs(V).max()), 1.0))))

OPTION_DEFAULTS = {
    "radius_scale": 1.0, "pitch_mm": None, "max_crop_voxels": 8_000_000, "max_total_voxels": 60_000_000,
    "frozen_keep_mm": None, "frozen_full_mm": None, "close_mm": None, "neck_min_mm3": None, "parent_tol_mm": None,
    "parent_check": "report", "smooth_mode": "auto", "remove_mode": "cut", "mirror_plane": None,
    "skip": [], "per_mark": {}, "check_frozen_exact": True, "loose_body_mm3": None, "max_odd_rays": 0,
}
OPTION_TEXT = {
    "radius_scale": "multiplies radius_mm of every mark (default 1.0)",
    "pitch_mm": "node spacing of the crop box in mm (default: voxel size of the run / 4)",
    "max_crop_voxels": "the voxel-count guard of one crop box (default 8,000,000)",
    "max_total_voxels": "the voxel-count guard of all crop boxes of one call (default 60,000,000). It limits the time of a call more than the memory: each step frees its arrays",
    "frozen_keep_mm": "no edit within this distance of a frozen region (default 3.0)",
    "frozen_full_mm": "the edit has full weight beyond this distance (default 6.0)",
    "close_mm": "ball radius of the closing of an add mark (default 3.0)",
    "neck_min_mm3": "an add mark gets the neck offset when the closing adds less than this volume (default 51.8)",
    "parent_tol_mm": "tolerance of the parent check of a remove mark (default 0.2)",
    "parent_check": "report (default: apply, give the check as a fact) or enforce (roll back a remove mark that fails it)",
    "smooth_mode": "auto (default: iso for a point, fair for a stroke), tube (rebuilds the whole member; does not use "
                   "radius_mm), iso or fair",
    "remove_mode": "cut (default: parent-protected cut) or shave (move the surface in by amount_mm)",
    "mirror_plane": "{axis: x|y|z, at_mm: number}: the plane of a mirrored mark (default: the plane of the mark or the "
                    "run; else y = 0, the plane of the page)",
    "skip": "list of mark ids that are not applied",
    "per_mark": "{id: {radius_mm, amount_mm, strength, mirror, smooth_mode, remove_mode, skip}}: values for one mark",
    "check_frozen_exact": "also check the frozen regions with their STL files, when the YAML is on disk (default true)",
    "loose_body_mm3": "a new body below this volume is a sliver of the boolean and is dropped; a larger new body rolls the "
                      "step back (default: max(10 x pitch^3, 0.001))",
    "max_odd_rays": "columns of the inside test that may stay odd (default 0): a ray that crosses the surface an odd "
                    "number of times grazes an edge (it is cast again, moved by a fraction of a node) or the mesh is not a "
                    "clean solid there. Above the limit the step is rolled back",
}
SMOOTH_MODES = ("auto", "tube", "iso", "fair")
REMOVE_MODES = ("cut", "shave")
PER_MARK_KEYS = ("radius_mm", "amount_mm", "strength", "mirror", "smooth_mode", "remove_mode", "skip")

VALUES_USED = {
    "remove": ["points", "radius_mm", "mirror", "amount_mm (remove_mode shave only)"],
    "add": ["points", "radius_mm", "amount_mm (the neck offset)", "mirror"],
    "smooth": ["points", "radius_mm (iso and fair; smooth_mode tube does not use it)", "strength", "mirror"],
}
VALUES_NOT_USED = ["normals", "note", "cleanup"]


# ---- small helpers ----------------------------------------------------------------------------------
def _f(x, nd=3):
    """A plain float, rounded (JSON)."""
    return None if x is None else round(float(x), nd)


def _cells(shape) -> int:
    return int(np.prod([int(s) for s in shape], dtype=np.int64))


def _peak_rss_mb() -> float | None:
    """The peak resident memory of this process in MB (Linux: KiB, macOS: bytes), or None."""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return None
    return round(v / (1024 * 1024) if sys.platform == "darwin" else v / 1024, 1)


def _load_avg() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def _cosw(d: np.ndarray, radius: float) -> np.ndarray:
    """The cosine weight: 1 at d = 0, 0 at d >= radius (apply_marks3.py: cosw)."""
    return np.where(d < radius, 0.5 * (1.0 + np.cos(np.pi * np.minimum(d, radius) / radius)), 0.0).astype(np.float32)


def _densify(path: np.ndarray, step: float) -> np.ndarray:
    """Points along a polyline, at most `step` apart; the vertices are kept."""
    path = np.asarray(path, float)
    if len(path) == 1:
        return path.copy()
    out = [path[0:1]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


def _resample(path: np.ndarray, step: float = 1.0) -> np.ndarray:
    """A path with points `step` apart, lightly smoothed inside (lib2.py: resample)."""
    p = np.asarray(path, float)
    if len(p) < 2:
        return p
    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if u[-1] < 1e-9:
        return p[:1]
    uu = np.arange(0.0, u[-1] + 1e-6, step)
    if uu[-1] < u[-1] - 1e-6:
        uu = np.r_[uu, u[-1]]
    q = np.stack([np.interp(uu, u, p[:, k]) for k in range(3)], 1)
    for _ in range(2):
        if len(q) > 2:
            q[1:-1] = 0.25 * q[:-2] + 0.5 * q[1:-1] + 0.25 * q[2:]
    return q


def _dedupe(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > tol:
            keep.append(i)
    return points[keep]


# ---- the labels of the run: frozen regions and design domain ----------------------------------------------
class Labels:
    """The voxel classes of the run (labels.bin). They give the frozen regions and the design domain.

    Frozen: fixed, load and keep-out voxels. Domain: the voxels that can hold material (design, fixed, load).
    The distance fields are made once on the voxel grid and read at any point (linear interpolation).
    """

    def __init__(self, labels: np.ndarray, origin, h: float):
        self.labels = labels
        self.origin = np.asarray(origin, float)
        self.h = float(h)
        self.shape = labels.shape
        F = (labels == FIXED) | (labels == LOAD) | (labels == KEEP_OUT)
        self.frozen = F
        D = (labels >= DESIGN) & (labels <= LOAD)
        big = 1e3
        # distance to the nearest frozen voxel BOUNDARY (the voxel centre distance minus half a voxel), in mm
        self.fd_vox = (np.maximum(ndi.distance_transform_edt(~F) * h - 0.5 * h, 0.0).astype(np.float32)
                       if F.any() else np.full(labels.shape, big, np.float32))
        # signed distance of the domain (negative inside)
        if D.all():
            self.sd_domain = np.full(labels.shape, -big, np.float32)
        elif not D.any():
            self.sd_domain = np.full(labels.shape, big, np.float32)
        else:
            self.sd_domain = (np.where(D, -(ndi.distance_transform_edt(D) - 0.5),
                                       ndi.distance_transform_edt(~D) - 0.5) * h).astype(np.float32)
        FL = (labels == FIXED) | (labels == LOAD)
        self.fixed_load_near = ndi.binary_dilation(FL, iterations=1) if FL.any() else FL

    def _index(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, float) - self.origin) / self.h - 0.5          # voxel-centre index coordinates

    def label_at(self, pts: np.ndarray) -> np.ndarray:
        """The class of the voxel that holds each point (EMPTY outside the grid)."""
        i = np.floor((np.asarray(pts, float) - self.origin) / self.h).astype(np.int64)
        ok = np.all((i >= 0) & (i < np.array(self.shape)), axis=1)
        out = np.zeros(len(i), np.uint8)
        ii = i[ok]
        out[ok] = self.labels[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def frozen_distance(self, pts: np.ndarray) -> np.ndarray:
        """Distance in mm from each point to the nearest frozen voxel (0 inside one)."""
        idx = self._index(pts).T
        fd = ndi.map_coordinates(self.fd_vox, idx, order=1, mode="nearest").astype(np.float32)
        lab = self.label_at(pts)
        fd[(lab == FIXED) | (lab == LOAD) | (lab == KEEP_OUT)] = 0.0
        return fd

    def domain_sd(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance in mm to the design domain (negative inside; large outside the grid)."""
        idx = self._index(pts).T
        return ndi.map_coordinates(self.sd_domain, idx, order=1, mode="constant", cval=1e3).astype(np.float32)

    def holds_fixed_or_load(self, mesh, faces: np.ndarray) -> bool:
        """Does a body (the triangles `faces` of `mesh`) hold a fixed or load voxel? By volume: a voxel centre of the
        run grid, or one next to it, lies inside the body (ray test of the voxelizer). A body inside a thick block
        counts although no vertex of it is near the region."""
        import trimesh
        from .voxelize import inside_mask
        body = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)[faces], process=False)
        if not len(body.faces):
            return False
        inside = inside_mask(body, self.origin, self.h, self.shape)
        return bool((inside & self.fixed_load_near).any())


# ---- the crop box: a grid of nodes around a mark -----------------------------------------------------------
class Crop:
    """A grid of nodes at `lo + pitch * (i, j, k)`. All fields of one step live on it."""

    def __init__(self, lo, shape, pitch: float):
        self.lo = np.asarray(lo, float)
        self.shape = tuple(int(s) for s in shape)
        self.p = float(pitch)

    @property
    def hi(self) -> np.ndarray:
        return self.lo + self.p * (np.array(self.shape) - 1)

    def axes(self):
        return [self.lo[a] + self.p * np.arange(self.shape[a]) for a in range(3)]

    def points(self, sl=None) -> np.ndarray:
        """The nodes (n, 3) of a sub-box (a tuple of slices), in C order, float32 offsets from `lo`."""
        sl = sl or (slice(None),) * 3
        ax = [a[s] for a, s in zip(self.axes(), sl)]
        g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        return g.astype(np.float64)

    def sub_slice(self, lo_mm, hi_mm):
        """The slices of the nodes inside the box lo_mm..hi_mm, and the origin of the sub-box in nodes."""
        a = np.maximum(np.floor((np.asarray(lo_mm) - self.lo) / self.p).astype(int), 0)
        b = np.minimum(np.ceil((np.asarray(hi_mm) - self.lo) / self.p).astype(int) + 1, self.shape)
        return tuple(slice(int(x), int(max(y, x))) for x, y in zip(a, b))

    def node_of(self, pts) -> np.ndarray:
        """The nearest node index of each point (clipped to the box)."""
        ii = np.round((np.asarray(pts, float) - self.lo) / self.p).astype(int)
        return np.clip(ii, 0, np.array(self.shape) - 1)


def _dist_to_samples(crop: Crop, samples: np.ndarray, rmax: float, workers: int = 1):
    """Distance in mm from each node to the nearest sample point. Nodes farther than `rmax` from all samples get 1e3.

    The tree is queried inside the box of the samples +- rmax only. Returns (distance array, slices of that box).
    """
    from scipy.spatial import cKDTree
    out = np.full(crop.shape, 1e3, np.float32)
    sl = crop.sub_slice(samples.min(0) - rmax, samples.max(0) + rmax)
    if any(s.stop <= s.start for s in sl):
        return out, sl
    pts = crop.points(sl)
    d, _ = cKDTree(samples).query(pts, workers=workers)
    sub = d.reshape([s.stop - s.start for s in sl]).astype(np.float32)
    sub[sub > rmax] = 1e3
    out[sl] = sub
    return out, sl


# ---- the signed distance field of the current mesh ---------------------------------------------------------
class _CountingRay:
    """The ray engine of a mesh, with a count of the rays that have an odd number of hits (voxelize.inside_mask drops
    such a ray; pitfalls/stl-overlapping-bodies-xor) and the x, y of those rays."""

    def __init__(self, ray):
        self.ray, self.odd, self.rays, self.odd_xy = ray, 0, 0, np.zeros((0, 2))

    def intersects_location(self, origins, directions, multiple_hits=True):
        locs, ids, tris = self.ray.intersects_location(origins, directions, multiple_hits=multiple_hits)
        self.rays = int(len(origins))
        odd = (np.bincount(ids, minlength=len(origins)) % 2 == 1) if len(ids) else np.zeros(len(origins), bool)
        self.odd = int(odd.sum())
        self.odd_xy = np.asarray(origins)[odd][:, :2]
        return locs, ids, tris


class _MeshView:
    """What inside_mask reads of a mesh: the bounds and the ray engine."""

    def __init__(self, mesh):
        self.bounds = mesh.bounds
        self.ray = _CountingRay(mesh.ray)


RECAST_SHIFTS = ((0.21, 0.13), (-0.17, 0.29), (0.33, -0.23), (-0.29, -0.31))     # in nodes, inside one column
RECAST_MAX_COLUMNS = 200


def _recast_odd_columns(sub, S: np.ndarray, odd_xy: np.ndarray, org: np.ndarray, p: float) -> tuple[int, int]:
    """Repair the columns of the inside mask whose ray hit the surface an odd number of times (a ray that grazes an edge
    or a vertex of the mesh): the column is cast again with the ray moved by a fraction of a node. Changes S in place.
    Returns (repaired, left). A column that stays odd is `outside` in S (as inside_mask leaves it)."""
    from .voxelize import inside_mask
    ii = np.floor((odd_xy[:, 0] - org[0]) / p).astype(int)
    jj = np.floor((odd_xy[:, 1] - org[1]) / p).astype(int)
    cols = sorted(set(zip(ii.tolist(), jj.tolist())))
    if len(cols) > RECAST_MAX_COLUMNS:
        return 0, len(cols)
    repaired = 0
    for i, j in cols:
        for dx, dy in RECAST_SHIFTS:
            view = _MeshView(sub)
            col = inside_mask(view, np.array([org[0] + (i + dx) * p, org[1] + (j + dy) * p, org[2]]), p, (1, 1, S.shape[2]))
            if view.ray.odd == 0:
                S[i, j, :] = col[0, 0, :]
                repaired += 1
                break
    return repaired, len(cols) - repaired


def signed_distance(mesh, crop: Crop, cap: float = BAND_MM, stats: dict | None = None) -> np.ndarray:
    """Signed distance (mm, negative inside) of a closed mesh at the nodes of the crop, clipped to +-cap.

    Method: the inside mask at the nodes (ray parity along z, as the voxelizer), the distance to the mask boundary
    from two Euclidean distance transforms, and the exact distance to the mesh on the one layer of nodes at each side of
    the surface. The surface is within a few thousandths of a mm of the zero level on a smooth part. The mask is
    cut at the box without a wall: a solid that reaches the box face goes on outside it. A ray with an odd number of
    hits (it grazes an edge) is cast again with a small shift; `stats` (a dict) gets `odd_rays` (the columns that stay
    odd, which the inside test drops), `odd_rays_repaired` and `rays`.
    """
    import trimesh
    from .voxelize import inside_mask

    lo, p, shape = crop.lo, crop.p, crop.shape
    hi = crop.hi
    tri = mesh.triangles
    tmin, tmax = tri.min(axis=1), tri.max(axis=1)
    prism = (tmax[:, 0] >= lo[0] - p) & (tmin[:, 0] <= hi[0] + p) & (tmax[:, 1] >= lo[1] - p) & (tmin[:, 1] <= hi[1] + p)
    near = (np.all(tmax >= lo - 3 * p, axis=1) & np.all(tmin <= hi + 3 * p, axis=1))
    if not prism.any():
        return np.full(shape, cap, np.float32)
    sub = trimesh.Trimesh(mesh.vertices, mesh.faces[prism], process=False)
    view = _MeshView(sub)
    org = lo - p / 2.0
    S = inside_mask(view, org, p, shape)
    odd, rays, repaired = view.ray.odd, view.ray.rays, 0
    if odd:
        repaired, odd = _recast_odd_columns(sub, S, view.ray.odd_xy, org, p)
    if stats is not None:
        stats["odd_rays"], stats["odd_rays_repaired"], stats["rays"] = odd, repaired, rays
    if not S.any():
        return np.full(shape, cap, np.float32)
    if S.all():
        return np.full(shape, -cap, np.float32)
    sd = np.where(S, -(ndi.distance_transform_edt(S) - 0.5), ndi.distance_transform_edt(~S) - 0.5) * p
    shell = ndi.binary_dilation(S) & ~ndi.binary_erosion(S, border_value=1)
    idx = np.argwhere(shell)
    if len(idx) and near.any():
        sub_near = trimesh.Trimesh(mesh.vertices, mesh.faces[near], process=False)
        _, dist, _ = trimesh.proximity.ProximityQuery(sub_near).on_surface(lo + idx * p)
        sd[tuple(idx.T)] = np.where(S[tuple(idx.T)], -dist, dist)
    return np.clip(sd, -cap, cap).astype(np.float32)


def _ball_closing(S: np.ndarray, r: float, p: float) -> np.ndarray:
    """The closing of a node mask with a ball of radius r mm: erode(dilate(S)). The box face has no wall."""
    if r <= 0 or not S.any():
        return S.copy()
    D = ndi.distance_transform_edt(~S) * p <= r
    if D.all():
        return D
    return (ndi.distance_transform_edt(D) * p > r) | S


def _signed_from_mask(M: np.ndarray, p: float, cap: float = 6.0) -> np.ndarray:
    """Signed distance of a node mask (negative inside), half a node corrected, clipped to +-cap."""
    if M.all():
        return np.full(M.shape, -cap, np.float32)
    if not M.any():
        return np.full(M.shape, cap, np.float32)
    sd = np.where(M, -(ndi.distance_transform_edt(M) - 0.5), ndi.distance_transform_edt(~M) - 0.5) * p
    return np.clip(sd, -cap, cap).astype(np.float32)


# ---- medial axis, parent envelope, thickness ---------------------------------------------------------------
_NEIGHBOURS = [np.array(o) - 1 for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]


def _medial_points(S: np.ndarray, tol: float = 0.3):
    """Discrete medial axis (lib2.py medial_points). Returns (mask, r): r is the distance to the nearest empty node in
    node units. A solid node is medial when its ball is not inside the ball of a neighbour."""
    r = ndi.distance_transform_edt(S).astype(np.float32)
    contained = np.zeros(S.shape, bool)
    rp = np.pad(r, 1)
    nx, ny, nz = S.shape
    for d in _NEIGHBOURS:
        dd = float(np.linalg.norm(d))
        q = rp[1 + d[0]:1 + d[0] + nx, 1 + d[1]:1 + d[1] + ny, 1 + d[2]:1 + d[2] + nz]
        contained |= q >= r + dd - tol
    return S & ~contained & (r >= 1.0), r


def _balls(shape, centres: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """The union of balls (centres in node indices, radii in node units) as a mask (lib2.py balls_mask)."""
    m = np.zeros(shape, bool)
    shp = np.array(shape)
    for c, r in zip(centres, radii):
        r0 = int(math.ceil(r))
        lo = np.maximum(c - r0, 0)
        hi = np.minimum(c + r0 + 1, shp)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        gi = np.ogrid[sl]
        d2 = (gi[0] - c[0]) ** 2 + (gi[1] - c[1]) ** 2 + (gi[2] - c[2]) ** 2
        m[sl] |= d2 <= r * r
    return m


def member_thickness(S: np.ndarray, zone: np.ndarray, p: float, reach_mm: float = 8.0):
    """The thickness of the member at the zone: the inscribed diameters (mm) of the medial-axis balls that reach the
    zone. Returns {"min", "median", "n"} or None.

    A medial ball is the largest ball in the solid at its centre, so the diameter is the thickness of the member
    there. A convex edge or corner has no ball of its own and takes the balls of the body under it, so an edge is not
    read as a thin member (it was in the first version). Only the window around the zone (padded by reach_mm) is
    computed, and a solid that reaches the window face goes on outside it. This is a C3a-only method (it is not the
    ball opening of tools/check_part.py nor the thickness of the pipeline)."""
    if not zone.any():
        return None
    ii = np.argwhere(zone)
    pad = int(math.ceil(reach_mm / p))
    a = np.maximum(ii.min(0) - pad, 0)
    b = np.minimum(ii.max(0) + pad + 1, S.shape)
    win = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
    Sw, Zw = S[win], zone[win] & S[win]
    if not Zw.any():
        return None
    ma, r = _medial_points(Sw)
    idx = np.argwhere(ma)
    if not len(idx):
        return None
    rad = r[tuple(idx.T)] - 0.5
    dz = ndi.distance_transform_edt(~Zw)
    use = dz[tuple(idx.T)] <= rad + 1.0
    if not use.any():
        return None
    dia = 2.0 * rad[use] * p
    return {"min": float(dia.min()), "median": float(np.median(dia)), "n": int(use.sum())}


def changed_nodes(Tc: np.ndarray, Tn: np.ndarray, p: float) -> np.ndarray:
    """The nodes where an edit can move the surface: the field changes by more than the tolerance, and it is within two
    nodes of the zero level before or after, or it changes sign. A change far from the surface (the smoothing of the
    distance far outside or inside) does not move the surface and does not count."""
    moved = np.abs(Tn - Tc) > CHANGE_TOL_MM
    near = np.minimum(np.abs(Tc), np.abs(Tn)) < NEAR_ZERO_NODES * p
    return moved & (near | ((Tn < 0) != (Tc < 0)))


def _grow(mask: np.ndarray, r: float, p: float) -> np.ndarray:
    return ndi.distance_transform_edt(~mask) * p <= r if mask.any() else mask.copy()


# ---- the operators --------------------------------------------------------------------------------------
class NoEffect(Exception):
    """The operator found nothing to change. `reason` is a fact for the report."""

    def __init__(self, reason: str, facts: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.facts = facts or {}


class OpFail(Exception):
    """The operator cannot work on this input (for example a tube on a stroke that is not on a strut)."""


class Ctx:
    """What an operator needs: the crop, the field of the current mesh, the paths of the mark, and the values."""

    def __init__(self, crop: Crop, Tc: np.ndarray, paths: list, radius: float, amount, strength, prm: dict,
                 labels: Labels | None):
        self.crop, self.Tc, self.p = crop, Tc, crop.p
        self.S = Tc < 0
        self.paths = [np.asarray(q, float) for q in paths]
        self.radius, self.amount, self.strength = float(radius), amount, strength
        self.prm = prm
        self.labels = labels
        self.notes: list[str] = []
        self.neck_scale = 1.0            # share of the neck threshold for this site (see _prepare)
        self.samples = np.vstack([_densify(q, max(crop.p * 0.5, 0.05)) for q in self.paths])
        self._dpath = None
        self._fd = None
        self._dom = None
        self._fz = {}
        self.support_slices = None

    def dpath(self, rmax: float) -> np.ndarray:
        """Distance to the paths (1e3 beyond rmax), cached for the largest rmax asked."""
        if self._dpath is None or self._dpath[0] < rmax:
            d, sl = _dist_to_samples(self.crop, self.samples, rmax)
            self._dpath = (rmax, d)
            self.support_slices = sl
            self._fd = self._dom = None
            self._fz = {}
        return self._dpath[1]

    def _fill(self, fn, base):
        out = np.full(self.crop.shape, base, np.float32)
        sl = self.support_slices
        if self.labels is not None and sl is not None and all(s.stop > s.start for s in sl):
            out[sl] = fn(self.crop.points(sl)).reshape(out[sl].shape)
        return out

    def frozen_distance(self) -> np.ndarray:
        """Distance to the nearest frozen voxel at the nodes of the support box (1e3 elsewhere)."""
        if self._fd is None:
            self._fd = self._fill(self.labels.frozen_distance, 1e3) if self.labels is not None \
                else np.full(self.crop.shape, 1e3, np.float32)
        return self._fd

    def domain_sd(self) -> np.ndarray:
        """Signed distance to the design domain at the nodes of the support box (-1e3 elsewhere: no clip there)."""
        if self._dom is None:
            self._dom = self._fill(self.labels.domain_sd, -1e3) if self.labels is not None \
                else np.full(self.crop.shape, -1e3, np.float32)
        return self._dom

    def frozen_weight(self, enabled: bool = True) -> np.ndarray:
        """The weight of an edit near frozen regions: 0 within frozen_keep, 1 beyond frozen_full (lib2.py fz_w)."""
        if not enabled or self.labels is None:
            return np.ones(self.crop.shape, np.float32)
        if enabled not in self._fz:
            d0, d1 = self.prm["frozen_keep_mm"], self.prm["frozen_full_mm"]
            fd = self.frozen_distance()
            self._fz[enabled] = (fd > d0).astype(np.float32) if d1 <= d0 else \
                np.clip((fd - d0) / (d1 - d0), 0.0, 1.0).astype(np.float32)
        return self._fz[enabled]

    def clip_possible(self) -> bool:
        """Can the frozen weight change an edit (a frozen voxel is within frozen_full of the support)?"""
        return bool((self.frozen_weight(True) < 1.0).any())

    def clip_domain(self, Tn: np.ndarray) -> np.ndarray:
        """No new material outside the design domain (a part that already sticks out of it stays as it is)."""
        if self.labels is None:
            return Tn.astype(np.float32)
        return np.maximum(Tn, np.minimum(self.domain_sd(), self.Tc)).astype(np.float32)


class OpResult:
    """The result of an operator: the new field (frozen weight on, and off for the clip fact), the mode, the facts."""

    def __init__(self, build, mode: str, info: dict, planned_mm3: float | None = None):
        self._build = build
        self.mode = mode
        self.info = info
        self.planned_mm3 = planned_mm3     # edit volume without the frozen weight, when the operator knows it
        self._cache: dict = {}

    def field(self, frozen: bool = True) -> np.ndarray:
        if frozen not in self._cache:
            self._cache[frozen] = self._build(frozen)
        return self._cache[frozen]


def _volume_of(T: np.ndarray, p: float) -> np.ndarray:
    """The inside fraction of each node cell, from the field (linear in the distance, 0 to 1)."""
    return np.clip(0.5 - T / p, 0.0, 1.0)


def _edit_volume(Tc: np.ndarray, Tn: np.ndarray, p: float) -> tuple[float, float]:
    """(added, removed) volume in mm3 of an edit of the field. The cell fraction is linear in the field, so a
    surface that moves by less than a node still counts."""
    dv = _volume_of(Tn, p) - _volume_of(Tc, p)
    return float(dv[dv > 0].sum()) * p ** 3, float(-dv[dv < 0].sum()) * p ** 3


def _strength(cx: Ctx) -> tuple[float, bool]:
    ok = isinstance(cx.strength, (int, float)) and not isinstance(cx.strength, bool) and cx.strength > 0
    return (float(cx.strength) if ok else 1.0), not ok


def op_add(cx: Ctx) -> OpResult:
    """A closing with a ball (apply_marks3.py do_add), and a neck offset when the closing adds little.

    The closing fills notches narrower than 2 x close_mm. The weight is the cosine of the distance to the mark
    (radius = radius_mm of the page), times the frozen weight. The neck offset is amount_mm of the page. Material is
    only added. The neck threshold (0.4 g in the pipeline) counts both mirror sides there; here a site gets the
    share of the images that it holds (`neck_scale`, set in `_prepare`).
    """
    p, Tc, S, prm = cx.p, cx.Tc, cx.S, cx.prm
    R = cx.radius
    W0 = _cosw(cx.dpath(R), R)
    if not (W0 > 0).any():
        raise NoEffect("no node of the crop box lies within radius_mm of the mark")
    Sc = _ball_closing(S, prm["close_mm"], p) | S
    sd = ndi.gaussian_filter(_signed_from_mask(Sc, p), 0.3 / p, mode="nearest")            # apply_marks3.py:101: 0.6 nodes of H = 0.5 mm, i.e. 0.3 mm
    has_amount = isinstance(cx.amount, (int, float)) and not isinstance(cx.amount, bool) and cx.amount > 0
    offset = float(cx.amount) if has_amount else prm["neck_offset_mm"]
    info = {"closing_mm": prm["close_mm"], "zone_radius_mm": R, "neck_offset_mm": offset,
            "neck_offset_is_default": not has_amount}
    state: dict = {}

    def build(frozen: bool):
        W = W0 * cx.frozen_weight(frozen)
        Tn = np.minimum(Tc, Tc + W * (sd - Tc))
        g_close = float(((Tn < 0) & (Tc >= 0)).sum()) * p ** 3
        if "neck" not in state:                      # the decision is made with the frozen weight on, and kept
            limit = prm["neck_min_mm3"] * cx.neck_scale
            state["neck"] = g_close < limit
            info["closing_added_mm3"] = _f(g_close, 2)
            info["neck_min_mm3_used"] = _f(limit, 2)
            info["neck_offset_used"] = bool(state["neck"])
        if state["neck"]:
            off = offset * W0 * cx.frozen_weight(frozen)
            Tn2 = np.where((Tc > -3.0) & (Tc < 3.0), Tn - off, Tn)                           # apply_marks3.py:106
            Tn = np.minimum(Tn, Tn2)
        return cx.clip_domain(Tn)

    res = OpResult(build, "closing", info)
    res.field(True)
    res.mode = "closing + neck offset" if state["neck"] else "closing"
    return res


def op_iso(cx: Ctx) -> OpResult:
    """A mild isotropic fairing of the field (apply_marks3.py op_iso): the Gaussian-smoothed field near the surface,
    sigma = iso_sigma_mm x strength, in a cosine zone of radius radius_mm around the mark."""
    p, Tc, prm = cx.p, cx.Tc, cx.prm
    strength, default = _strength(cx)
    sigma = prm["iso_sigma_mm"] * strength
    W0 = _cosw(cx.dpath(cx.radius), cx.radius)
    if not (W0 > 0).any():
        raise NoEffect("no node of the crop box lies within radius_mm of the mark")
    Ts = ndi.gaussian_filter(np.clip(Tc, -BAND_MM, BAND_MM), sigma / p, mode="nearest")

    def build(frozen: bool):
        W = W0 * cx.frozen_weight(frozen)
        return cx.clip_domain(np.where(np.abs(Tc) < 6.0, Tc + W * (Ts - Tc), Tc))

    info = {"sigma_mm": _f(sigma), "zone_radius_mm": cx.radius, "strength_used": strength,
            "strength_is_default": default}
    return OpResult(build, "iso", info)


def _path_tangents(path: np.ndarray, sigma_pts: float = 2.0) -> np.ndarray:
    tg = np.gradient(path, axis=0)
    tg /= np.maximum(np.linalg.norm(tg, axis=1), 1e-12)[:, None]
    if len(tg) > 2:
        tg = ndi.gaussian_filter1d(tg, sigma_pts, axis=0, mode="nearest")
        tg /= np.maximum(np.linalg.norm(tg, axis=1), 1e-12)[:, None]
    return tg


def op_fair(cx: Ctx) -> OpResult:
    """Fairing along a flat web (apply_marks3.py op_fair2): a Gaussian along the web direction, sigma = fair_sigma_mm x
    strength, weighted by the cosine of the distance to the path (radius_mm). The direction at a node is the tangent
    of its nearest path point. Only the nodes of the zone are computed."""
    from scipy.spatial import cKDTree
    p, Tc, prm = cx.p, cx.Tc, cx.prm
    strength, default = _strength(cx)
    sa = prm["fair_sigma_mm"] * strength
    zone = cx.radius
    W0 = _cosw(cx.dpath(zone), zone)
    if not (W0 > 0).any():
        raise NoEffect("no node of the crop box lies within radius_mm of the mark")
    paths = [_resample(q, prm["path_step_mm"]) for q in cx.paths]
    path = np.vstack(paths)
    tg = np.vstack([_path_tangents(q) if len(q) > 1 else np.array([[1.0, 0.0, 0.0]]) for q in paths])
    iso = ndi.gaussian_filter(np.clip(Tc, -BAND_MM, BAND_MM), 0.5 / p, mode="nearest")
    zone_idx = np.argwhere((W0 > 0) & (np.abs(Tc) < 6.0))
    Ts = Tc.copy()
    if len(zone_idx):
        _, near = cKDTree(path).query(cx.crop.lo + zone_idx * p, workers=1)
        t = tg[near]
        n = int(math.ceil(2.5 * sa / p))
        ks = np.arange(-n, n + 1) * p
        wk = np.exp(-0.5 * (ks / sa) ** 2)
        wk /= wk.sum()
        acc = np.zeros(len(zone_idx), np.float32)
        fidx = zone_idx.astype(np.float64)
        for k, wv in zip(ks, wk):
            acc += wv * ndi.map_coordinates(iso, (fidx + (k / p) * t).T, order=1, mode="nearest")
        Ts[tuple(zone_idx.T)] = acc

    def build(frozen: bool):
        W = W0 * cx.frozen_weight(frozen)
        return cx.clip_domain(np.where(np.abs(Tc) < 6.0, Tc + W * (Ts - Tc), Tc))

    info = {"sigma_mm": _f(sa), "zone_radius_mm": zone, "strength_used": strength, "strength_is_default": default,
            "zone_nodes": int(len(zone_idx))}
    return OpResult(build, "fair", info)


def op_shave(cx: Ctx) -> OpResult:
    """The README meaning of a remove mark: the surface moves in by amount_mm at the mark, with a cosine falloff
    (radius_mm) in the tangent plane. It thins the part and does not protect the parent. Used with remove_mode = shave.

    The weight comes from the distance in the tangent plane, sqrt(d^2 - T^2) (d: the distance of the node to the mark,
    T: the field at the node), so the weight is 1 below the mark at any depth; the band is amount + 2 nodes. The facts
    give the depth that was planned at the mark (the frozen weight can lower it) and, after the splice, the depth that
    was reached."""
    Tc, p = cx.Tc, cx.p
    has_amount = isinstance(cx.amount, (int, float)) and not isinstance(cx.amount, bool) and cx.amount > 0
    depth = float(cx.amount) if has_amount else 1.0
    band = depth + 2.0 * p
    d = cx.dpath(math.hypot(cx.radius, band))
    dt = np.where(d < 1e3, np.sqrt(np.maximum(d.astype(np.float64) ** 2 - Tc.astype(np.float64) ** 2, 0.0)), 1e3)
    W0 = _cosw(dt, cx.radius)
    if not (W0 > 0).any():
        raise NoEffect("no node of the crop box lies within radius_mm of the mark")

    def build(frozen: bool):
        W = W0 * cx.frozen_weight(frozen)
        return np.where(np.abs(Tc) < band, Tc + depth * W, Tc).astype(np.float32)

    info = {"depth_mm": depth, "depth_is_default": not has_amount, "zone_radius_mm": cx.radius}
    res = OpResult(build, "shave", info)
    pts = np.vstack(cx.paths)
    at = ndi.map_coordinates(res.field(True), ((pts - cx.crop.lo) / p).T, order=1, mode="nearest")
    info["depth_at_mark_mm"] = _f(float(np.median(np.maximum(at, 0.0))), 2)
    return res


def _skeleton_nodes(Sb: np.ndarray, region: np.ndarray, pad: int = 4) -> np.ndarray:
    """The nodes of the 3-D skeleton of the solid (scikit-image `skeletonize`) inside `region`. Only a window around the
    region is thinned; the nodes next to the window face are left out, so the cut face makes no false branch."""
    from skimage.morphology import skeletonize
    out = np.zeros(Sb.shape, bool)
    if not region.any():
        return out
    ii = np.argwhere(region)
    a = np.maximum(ii.min(0) - pad, 0)
    b = np.minimum(ii.max(0) + pad + 1, Sb.shape)
    win = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
    sk = np.asarray(skeletonize(Sb[win])).astype(bool)
    inner = np.zeros(sk.shape, bool)
    inner[2:-2, 2:-2, 2:-2] = True
    out[win] = sk & inner
    return out & region


def _parent_check(cx: Ctx, Tn: np.ndarray, Rx: np.ndarray, Pc: np.ndarray, ma: np.ndarray, r_before: np.ndarray,
                  tol: float) -> dict:
    """The parent check (lib2.py parent_check): the cut leaves the parent as it was, to a tolerance. Before and after,
    the inscribed diameter at (a) the skeleton nodes and (b) the medial-axis nodes outside the feature zone and within
    10 mm of it (min, mean and the largest local drop), and (c) the depth of the cut inside the closed parent envelope.
    `within_tolerance` is true when no drop and no depth is over the tolerance. It is a fact about the cut, not a
    verdict on the part: the check does not see a pit that the cut digs into a flat face."""
    p = cx.p
    Sb, Sa = cx.S, Tn < 0
    r_after = ndi.distance_transform_edt(Sa).astype(np.float32)
    near = ndi.distance_transform_edt(~Rx) * p <= 10.0
    inner = np.zeros(Sb.shape, bool)
    inner[2:-2, 2:-2, 2:-2] = True
    sel = ma & ~Rx & near & inner
    out = {"tolerance_mm": tol, "n_points": int(sel.sum())}
    if sel.any():
        rb, ra = (r_before[sel] - 0.5) * p, (r_after[sel] - 0.5) * p
        drop = rb - ra
        out.update(max_drop_mm=_f(drop.max()), mean_drop_mm=_f(drop.mean()), n_over_tolerance=int((drop > tol).sum()))
    else:
        out.update(max_drop_mm=0.0, mean_drop_mm=0.0, n_over_tolerance=0)
    skn = _skeleton_nodes(Sb, ~Rx & near & inner)
    skel = {"n_points": int(skn.sum())}
    if skn.any():
        rb, ra = (r_before[skn] - 0.5) * p, (r_after[skn] - 0.5) * p
        drop = rb - ra
        skel.update(min_drop_mm=_f(rb.min() - ra.min()), mean_drop_mm=_f(rb.mean() - ra.mean()),
                    max_local_drop_mm=_f(drop.max()), n_over_tolerance=int((drop > tol).sum()))
    else:
        skel.update(min_drop_mm=0.0, mean_drop_mm=0.0, max_local_drop_mm=0.0, n_over_tolerance=0)
    out["skeleton"] = skel
    removed = Sb & ~Sa
    Pb = Pc & Sb
    inP = removed & Pb
    depth = ndi.distance_transform_edt(Pb) * p if Pb.any() else np.zeros(Sb.shape)
    out["removed_in_parent_mm3"] = _f(inP.sum() * p ** 3, 2)
    out["max_depth_into_parent_mm"] = _f(depth[inP].max() if inP.any() else 0.0)
    out["within_tolerance"] = bool(out["max_drop_mm"] <= tol and out["max_depth_into_parent_mm"] <= tol
                                   and skel["min_drop_mm"] <= tol and skel["mean_drop_mm"] <= tol
                                   and skel["max_local_drop_mm"] <= tol)
    return out


def op_remove(cx: Ctx) -> OpResult:
    """A cut that protects the parent (apply_marks3.py do_remove, lib2.py build_region2 and op_remove2).

    The feature is the solid within radius_mm of the mark. The parent is the union of the medial balls whose centre
    is not close to the feature, closed with a ball (remove_close_mm) and grown by remove_margin_mm. The cut is the
    removal domain minus the parent, beyond frozen_keep_mm from the frozen regions, plus a thin foot. The cut field is
    smoothed, then the scar is polished where it does not enter the parent.
    """
    p, Tc, S, prm = cx.p, cx.Tc, cx.S, cx.prm
    R_mm = cx.radius
    dom, excl = prm["remove_dom_mm"], prm["remove_excl_mm"]
    d = cx.dpath(R_mm + dom + 1.0)
    R = S & (d <= R_mm)
    if not R.any():
        raise NoEffect("no solid within radius_mm of the mark: the mark is not on the part")
    # the parts of the feature that touch the mark (apply_marks3.py:33)
    lab, n = ndi.label(R, structure=np.ones((3, 3, 3)))
    mk = np.zeros(S.shape, bool)
    pts = np.vstack(cx.paths)
    mk[tuple(cx.crop.node_of(pts).T)] = True
    near = ndi.distance_transform_edt(~mk) * p < max(2.5, 2 * p)
    keep = np.unique(lab[near & R])
    R = np.isin(lab, keep[keep > 0])
    if not R.any():
        raise NoEffect("no solid of the feature lies within 2.5 mm of a point of the mark")
    dR = ndi.distance_transform_edt(~R) * p
    Rd = (dR <= dom) & S
    Rx = (dR <= excl) & S
    ma, r = _medial_points(S)
    idx = np.argwhere(ma)
    rad = r[tuple(idx.T)] - 0.01
    outside = ~Rx[tuple(idx.T)]
    reach = rad * p + 2.0 * prm["remove_close_mm"] + prm["remove_margin_mm"] + dom + 2 * p      # a closing joins balls up to 2 x close apart
    sel = outside & (dR[tuple(idx.T)] <= reach)                  # only the balls that can touch the removal domain
    P0 = _balls(S.shape, idx[sel], rad[sel])
    Pc = _ball_closing(P0, prm["remove_close_mm"], p) | P0
    Pm = Pc if prm["remove_margin_mm"] <= 0 else _grow(Pc, prm["remove_margin_mm"], p)
    fd = cx.frozen_distance()
    allow = fd > prm["frozen_keep_mm"]
    rem_all = Rd & ~Pm
    rem = rem_all & allow
    foot = np.zeros(S.shape, bool)
    if rem.any() and prm["foot_rho_mm"] > 0:                       # lib2.py:105 leftover feet
        rho = prm["foot_rho_mm"]
        pad = int(math.ceil(rho / p)) + 2
        S1 = S & ~rem
        Sp = np.pad(S1, pad, mode="edge")
        e = ndi.distance_transform_edt(Sp) * p >= rho
        opened = (ndi.distance_transform_edt(~e) * p <= rho)[(slice(pad, -pad),) * 3] if e.any() else np.zeros(S.shape, bool)
        resid = S1 & ~opened & ~(Pc & S) & allow
        df = ndi.distance_transform_edt(~rem) * p
        lab2, _ = ndi.label(resid & (df < 3.0), structure=np.ones((3, 3, 3)))
        k2 = np.unique(lab2[(df < 1.2) & (lab2 > 0)])
        foot = np.isin(lab2, k2[k2 > 0])
        if foot.sum() * p ** 3 > prm["foot_max_mm3"]:
            foot[:] = False
    C = rem | foot
    clipped_mm3 = float((rem_all & ~allow).sum()) * p ** 3
    planned = float((rem_all | foot).sum()) * p ** 3
    if not C.any():
        facts = {"feature_region_mm3": _f(R.sum() * p ** 3, 2), "parent_envelope_covers_region": not rem_all.any(),
                 "clipped_by_frozen_mm3": _f(clipped_mm3, 2)}
        why = ("the parent envelope covers all solid of the region: it is not a feature that sticks out"
               if not rem_all.any() else
               f"all of the cut lies within frozen_keep_mm ({prm['frozen_keep_mm']:g} mm) of a frozen region. Options: "
               "frozen_keep_mm smaller (the exact check of the frozen regions still protects them), a mark farther from "
               "the region, or the rerun_with_hints route")
        raise NoEffect(why, facts)
    c = (ndi.distance_transform_edt(~C) - ndi.distance_transform_edt(C)) * p
    c = ndi.gaussian_filter(np.clip(c, -4, 4), prm["cut_smooth_mm"] / p, mode="nearest")                       # lib2.py:126 cutter
    Tn = np.where(np.abs(c) < 3.0, np.maximum(Tc, -c), Tc)
    zone = (ndi.distance_transform_edt(~C) * p <= prm["polish_zone_mm"]) & ~Pc & allow          # lib2.py:137 polish
    Ts = ndi.gaussian_filter(np.clip(Tn, -4, 4), prm["polish_mm"] / p, mode="nearest")
    Tn = np.where(zone, np.maximum(Tn, Ts), Tn).astype(np.float32)
    Tn = np.maximum(Tn, Tc)                                          # a remove mark never adds material
    info = {"feature_region_mm3": _f(R.sum() * p ** 3, 2), "removal_domain_mm3": _f(Rd.sum() * p ** 3, 2),
            "cut_region_mm3": _f(rem.sum() * p ** 3, 2), "foot_mm3": _f(foot.sum() * p ** 3, 2),
            "parent_balls": int(sel.sum()), "clipped_by_frozen_mm3": _f(clipped_mm3, 2),
            "radius_mm": R_mm, "removal_reach_mm": dom}
    info["parent_check"] = _parent_check(cx, Tn, Rx, Pc, ma, r, prm["parent_tol_mm"])
    res = OpResult(lambda frozen: Tn, "cut", info, planned_mm3=planned)
    return res


# ---- smooth: a faired tube, and the choice of the mode ---------------------------------------------------
def _section(cx: Ctx, S: np.ndarray, q: np.ndarray, t: np.ndarray, srad: float, thick: float):
    """The cross-section of the solid at q (normal t): (centroid, area in mm2, flatness, bounded) or None.

    The section is the connected solid with |axial offset| < thick and distance to the axis < srad, the part that
    holds the node nearest to q. Flatness is sqrt(l1 / l2) of the covariance of the section (1 for a round strut).
    The last item says whether the section is bounded: it does not reach the wall of the cylinder of radius srad, so
    the solid at q is not wider than a strut. It looks at a small window only (lib2.py op_smooth2: section)."""
    crop, p = cx.crop, cx.p
    r = srad + 2 * p
    sl = crop.sub_slice(q - r, q + r)
    if any(s.stop <= s.start for s in sl):
        return None
    Sw = S[sl]
    if not Sw.any():
        return None
    ax = crop.axes()
    xs, ys, zs = ax[0][sl[0]], ax[1][sl[1]], ax[2][sl[2]]
    rx, ry, rz = (xs - q[0])[:, None, None], (ys - q[1])[None, :, None], (zs - q[2])[None, None, :]
    al = rx * t[0] + ry * t[1] + rz * t[2]
    d2 = rx * rx + ry * ry + rz * rz
    mm = Sw & (np.abs(al) < thick) & (d2 - al * al < srad * srad)
    if not mm.any():
        return None
    lab, _ = ndi.label(mm, structure=np.ones((3, 3, 3)))
    k = np.unravel_index(np.argmin(np.where(mm, d2, np.inf)), mm.shape)
    ii = np.argwhere(lab == lab[k])
    pts = np.stack([xs[ii[:, 0]], ys[ii[:, 1]], zs[ii[:, 2]]], 1)
    c = pts.mean(0)
    V = pts - c
    V = V - (V @ t)[:, None] * t
    ev = np.linalg.eigvalsh(V.T @ V / len(V))
    q_rel = pts - q
    radial = np.sqrt(np.maximum((q_rel ** 2).sum(1) - (q_rel @ t) ** 2, 0.0))
    bounded = bool(radial.max() < srad - 1.5 * p)
    return c, len(pts) * p ** 3 / (2.0 * thick), math.sqrt(max(ev[2], 0.0) / max(ev[1], 1e-12)), bounded


def _snap_axis(cx: Ctx, edt: np.ndarray, A: np.ndarray, steps: int = 14, step: float = 0.4, maxmove: float = 5.0):
    """Move the stroke points to the ridge of the inscribed radius (the strut axis) by gradient ascent
    (apply_marks3.py snap_axis)."""
    p, lo = cx.p, cx.crop.lo
    g = [ndi.gaussian_filter(x, 0.5, mode="nearest") for x in np.gradient(edt, p)]
    out = []
    for p0 in A:
        q = p0.copy()
        for _ in range(steps):
            ii = ((q - lo) / p)[:, None]
            gv = np.array([ndi.map_coordinates(c, ii, order=1, mode="nearest")[0] for c in g])
            nrm = float(np.linalg.norm(gv))
            if nrm < 0.05:
                break
            q2 = q + step * gv / nrm
            if np.linalg.norm(q2 - p0) > maxmove:
                break
            q = q2
        out.append(q)
    return np.array(out)


def analyse_strut(cx: Ctx) -> dict:
    """The strut under the stroke: the snapped axis, the sections along it, the flatness. Raises OpFail when the
    stroke is not on a strut."""
    prm, p, S = cx.prm, cx.p, cx.S
    path = _dedupe(cx.paths[0])
    arc = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
    if arc < TUBE_MIN_ARC_MM:
        raise OpFail(f"the stroke is {arc:.1f} mm long; a strut needs at least {TUBE_MIN_ARC_MM:g} mm")
    A = _resample(path, prm["path_step_mm"])
    raw = ndi.distance_transform_edt(S) * p
    edt = ndi.gaussian_filter(raw, 0.6 / p, mode="nearest").astype(np.float32)      # smooth: the ridge for the snap
    A = _snap_axis(cx, edt, A)
    # the inscribed radius from the raw distance (the smoothed ridge is lower than the strut radius by the smoothing)
    rin = ndi.map_coordinates(raw, ((A - cx.crop.lo) / p).T, order=1, mode="nearest") - 0.5 * p
    r_med = float(np.median(rin))
    if r_med < 0.35:
        raise OpFail(f"the stroke is not on a strut: the inscribed radius along it is {r_med:.2f} mm")
    srad, thick = max(2.0 * r_med, 1.5), max(0.6, 1.2 * p)
    tg = np.gradient(A, axis=0)
    tg /= np.maximum(np.linalg.norm(tg, axis=1), 1e-12)[:, None]
    ctr, area, flat = [], [], []
    for q, t in zip(A, tg):
        s = None
        for _ in range(4):
            s = _section(cx, S, q, t, srad, thick)
            if s is None:
                raise OpFail("the solid has no section at a point of the stroke")
            q = s[0]
        s = _section(cx, S, q, t, srad, thick)
        if s is None:
            raise OpFail("the solid has no section at a point of the stroke")
        if not s[3]:
            raise OpFail(f"the section at a point of the stroke is not bounded: the solid there reaches the wall of the "
                         f"cylinder of radius {srad:.1f} mm (twice the inscribed radius). It is a block or a wide face, "
                         "not a strut")
        ctr.append(q)
        area.append(s[1])
        flat.append(s[2])
    return {"A": A, "ctr": np.array(ctr), "area": np.array(area), "tg": tg, "srad": srad, "thick": thick,
            "r_med_mm": r_med, "flatness": float(np.median(flat)), "arc_mm": arc}


def op_tube(cx: Ctx, an: dict) -> OpResult:
    """Rebuild the strut under the stroke as one faired tube (apply_marks3.py op_smooth2, lib2.py op_smooth2).

    The axis is the stroke, moved to the strut axis and extended along the strut while the section stays alike. The
    radius profile is the area-equivalent radius smoothed over a window. The old strut is removed, and the tube is
    joined with a smooth minimum (k = tube_blend_end_mm in the 4 mm at each end). The weight is 1 within 6 mm of the
    axis, 0 beyond 9 mm, times the frozen weight."""
    from scipy.interpolate import UnivariateSpline
    from scipy.spatial import cKDTree
    prm, p, S, Tc, crop = cx.prm, cx.p, cx.S, cx.Tc, cx.crop
    strength, default = _strength(cx)
    srad, thick = an["srad"], an["thick"]
    ctr, med = an["ctr"], float(np.median(an["area"]))

    def sect(q, t):
        s = _section(cx, S, q, t, srad, thick)
        return (q, 0.0) if s is None else (s[0], s[1])

    def extend(q, t):
        pts, hist = [], [med] * 5
        t = t / np.linalg.norm(t)
        for _ in range(int(prm["tube_ext_max_mm"])):
            q2, a2 = q + t, 0.0
            for _ in range(3):
                q2, a2 = sect(q2, t)
            ref = float(np.median(hist[-6:]))
            if np.linalg.norm(q2 - q - t) > 1.6 or a2 > 1.4 * ref or a2 < 0.6 * ref:
                break
            tn = (q2 - q) / np.linalg.norm(q2 - q)
            if tn @ t < 0.92:
                break
            t = 0.6 * t + 0.4 * tn
            t /= np.linalg.norm(t)
            q = q2
            pts.append(q)
            hist.append(a2)
        return pts[:-1] if len(pts) > 1 else []

    e0 = extend(ctr[0], ctr[0] - ctr[1])
    e1 = extend(ctr[-1], ctr[-1] - ctr[-2])
    full = np.vstack([np.array(e0)[::-1] if e0 else np.zeros((0, 3)), ctr, np.array(e1) if e1 else np.zeros((0, 3))])
    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))]
    if len(u) < 5:
        raise OpFail("the axis of the strut has fewer than 5 points")
    sp = [UnivariateSpline(u, full[:, k], k=3, s=len(u) * 0.35 ** 2) for k in range(3)]
    uu = np.arange(-3.0, u[-1] + 3.0 + 1e-6, 0.25)
    Q = np.stack([s_(uu) for s_ in sp], 1)
    Tq = np.gradient(Q, axis=0)
    Tq /= np.maximum(np.linalg.norm(Tq, axis=1), 1e-12)[:, None]
    Ar = np.maximum(np.array([sect(q, t)[1] for q, t in zip(Q, Tq)]), 1.0)
    win = min(prm["tube_window_mm"] * strength, max(4.0, u[-1] / 3.0))
    Ar_s = ndi.uniform_filter1d(Ar, max(int(round(win / 0.25)), 1), mode="nearest")
    r_s, r_raw = np.sqrt(Ar_s / np.pi), np.sqrt(Ar / np.pi)
    inside = (uu >= 0) & (uu <= u[-1])
    if not np.isfinite(r_s).all() or r_s[inside].min() < 0.3:
        raise OpFail("the radius profile of the strut is too small to rebuild")
    sup = crop.sub_slice(Q.min(0) - 9.5, Q.max(0) + 9.5)
    cx.support_slices = sup
    cx._fd = cx._dom = None
    cx._fz = {}
    pts = crop.points(sup)
    dax, j = cKDTree(Q).query(pts, workers=1)
    shp = Tc[sup].shape
    dax, uidx = dax.reshape(shp).astype(np.float32), uu[j].reshape(shp)
    Ttube = (dax - r_s[j].reshape(shp)).astype(np.float32)
    S_s, Tc_s = S[sup], Tc[sup]
    cand = S_s & (dax < 1.3 * r_raw[j].reshape(shp) + 0.3)       # the old strut: solid within 1.3 r (lib2.py:181 rf=1.3, :225)
    lab, _ = ndi.label(cand, structure=np.ones((3, 3, 3)))
    cl = np.unique(lab[(dax < 1.0) & cand])
    rem0 = np.isin(lab, cl[cl > 0])
    endd = np.minimum(uidx, u[-1] - uidx)
    k_end = np.where(endd < 4.0, prm["tube_blend_end_mm"], 0.05).astype(np.float32)
    W0 = np.clip((9.0 - dax) / 3.0, 0.0, 1.0).astype(np.float32)   # 1 within 6 mm of the axis, 0 beyond 9 mm (lib2.py:231)

    def build(frozen: bool):
        fz = cx.frozen_weight(frozen)[sup]
        rem = rem0 & (fz >= 1.0)                         # the old strut stays where the frozen weight clips the edit
        To = Tc_s.copy()
        To[rem] = np.maximum(To[rem], 1.0)
        if rem.any():
            near = ndi.distance_transform_edt(~rem) * p < 2.5
            sm = ndi.gaussian_filter(np.clip(To, -8, 8), 0.5 / p, mode="nearest")
            To = np.where(near & (np.abs(To) < 4.0), sm, To)
        h = np.clip(0.5 + 0.5 * (Ttube - To) / k_end, 0.0, 1.0)
        Ts = Ttube * (1 - h) + To * h - k_end * h * (1 - h)
        Tn = Tc.copy()
        Tn[sup] = To + W0 * fz * (Ts - To)
        return cx.clip_domain(Tn)

    info = {"arc_mm": _f(u[-1]), "extension_start_mm": len(e0), "extension_end_mm": len(e1),
            "section_radius_mm": _f(srad), "window_mm": _f(win), "strength_used": strength,
            "strength_is_default": default, "radius_raw_min_mm": _f(r_raw[inside].min()),
            "radius_raw_max_mm": _f(r_raw[inside].max()), "radius_fair_min_mm": _f(r_s[inside].min()),
            "radius_fair_max_mm": _f(r_s[inside].max()), "old_strut_mm3": _f(rem0.sum() * p ** 3, 2),
            "flatness": _f(an["flatness"], 2)}
    return OpResult(build, "tube", info)


def pick_smooth_mode(paths: list, requested: str = "auto"):
    """The mode of a smooth mark without a tube: ("iso" | "fair", reason).

    auto: a click (one point) is `iso`, a mild fairing around the point. A stroke is `fair`, a fairing along the
    stroke. Both use radius_mm and strength, so the edit stays in the zone that the page shows. `tube` is not
    chosen by auto: it rebuilds the whole member and does not use radius_mm (a requested tube is made in
    `_build_edit`, and falls back to iso with a reason when the stroke is not on a strut)."""
    if requested == "iso":
        return "iso", "asked: smooth_mode iso"
    if requested == "fair":
        return "fair", "asked: smooth_mode fair"
    stroke = any(len(_dedupe(np.asarray(q, float))) > 1 for q in paths)
    if stroke:
        return "fair", "auto: a stroke is a fairing along the stroke (fair); smooth_mode tube rebuilds the whole member"
    return "iso", "auto: one point is a fairing around the point (iso)"


# ---- the mesh: checks, the splice ------------------------------------------------------------------------
def mesh_facts(mesh) -> dict:
    """Closed, winding, volume, bodies, open and non-manifold edges, zero-area faces of a trimesh."""
    import trimesh
    f = np.asarray(mesh.faces)
    nv = len(mesh.vertices)
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1).astype(np.int64)
    _, cnt = np.unique(e[:, 0] * nv + e[:, 1], return_counts=True)
    bodies = len(trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(f))))
    vol = float(mesh.volume)
    return {"triangles": int(len(f)), "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent), "volume_mm3": _f(vol, 3), "bodies": int(bodies),
            "open_edges": int((cnt == 1).sum()), "non_manifold_edges": int((cnt > 2).sum()),
            "zero_area_faces": int((mesh.area_faces <= 0).sum()),
            "closed": bool(mesh.is_watertight and mesh.is_winding_consistent and vol > 0)}


def _mesh_of(man):
    """A trimesh (float32 vertices, as the STL holds them) from a manifold3d solid."""
    import trimesh
    o = man.to_mesh()
    v = np.asarray(o.vert_properties, np.float32)[:, :3].astype(np.float64)
    return trimesh.Trimesh(v, np.asarray(o.tri_verts, np.int64), process=False)


def _manifold(mesh):
    """A manifold3d solid from a trimesh, or None. It repairs one case: duplicate vertices (a merge, then a retry)."""
    import trimesh
    from .post import _to_manifold
    faces = np.asarray(mesh.faces)
    if mesh.volume < 0:
        faces = faces[:, ::-1]
    try:
        return _to_manifold(mesh.vertices, faces)
    except ValueError:
        pass
    fixed = trimesh.Trimesh(mesh.vertices, faces, process=True)
    fixed.merge_vertices()
    try:
        return _to_manifold(fixed.vertices, fixed.faces)
    except ValueError:
        return None


def _surface_mesh(field: np.ndarray, crop: Crop, start, level: float = EDIT_LEVEL) -> "object":
    """The zero level of a block of the field, as a closed mesh (the block is closed with a wall of outside nodes).
    `start` is the index of the first node of the block. Outward normals."""
    import trimesh
    from skimage.measure import marching_cubes
    f = np.pad(field, 1, constant_values=BAND_MM)
    verts, faces, _n, _v = marching_cubes(f, level=level, gradient_direction="descent", allow_degenerate=False)
    verts = crop.lo + crop.p * (np.asarray(start) + verts - 1.0)
    m = trimesh.Trimesh(verts, faces.copy(), process=False)
    if m.volume < 0:
        m.invert()
    return m


# ---- the seam: the new surface sewn to the old one ---------------------------------------------------------
def _tet_surface():
    """The surface module of `topopt verify` (numpy, scipy and numba; it does not import gmsh). The seam and the report
    use its definitions of a fold edge, a pinch vertex and the shortest edge, so they say what verify will see."""
    from .tet import surface
    return surface


SURFACE_QUALITY_NOTE = (
    "Facts about the STL as `topopt verify` reads it (equal vertices merged on a 1e-6 mm grid). closed_manifold is false when "
    "an edge has not two faces, the orientation is not consistent, or a vertex pinches the surface (non_manifold_vertices): "
    "verify refuses such a surface. fold_edges counts the edges whose two faces point back on each other (a fin of no "
    "thickness). verify also refuses a prepared surface that has an edge shorter than verify_edge_limit_mm; "
    "edges_under_verify_limit counts the edges of this file under that value. The remesh of verify can remove short edges, so "
    "that count is a hint, not a verdict. These are facts, not a refusal: the file is written in each case.")


def _surface_quality(path) -> dict:
    """Facts about the surface of an STL file, with the definitions of `topopt verify` (`topopt.tet.surface`): closed
    manifold (pinch vertices included), open and non-manifold edges, fold edges, the shortest edge. {"error": text} when the
    file cannot be read as a surface."""
    try:
        ts = _tet_surface()
        V, F = ts.load_stl_surface(str(path))
        r = ts.surface_report(V, F)
        eu, ev, _cnt = ts.edge_table(F, len(V))
        L = np.linalg.norm(V[eu] - V[ev], axis=1)
        return {"triangles": int(r["faces"]), "closed_manifold": bool(r["closed_manifold"]), "open_edges": int(r["open_edges"]),
                "non_manifold_edges": int(r["nonmanifold_edges"]),
                "non_manifold_vertices": int(r["nonmanifold_vertices"]) if r["nonmanifold_vertices"] >= 0 else None,
                "bodies": int(r["bodies"]), "fold_edges": int(r["fold_edges"]), "edge_min_mm": _f(r["edge_min_mm"], 5),
                "edge_p1_mm": _f(r["edge_p1_mm"], 4), "edges_under_verify_limit": int((L < ts.MIN_EDGE_REJECT_MM).sum())}
    except Exception as exc:                                    # SurfaceRejected, ImportError, a broken file: a fact, not a crash
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tet_surface_limit():
    """The shortest edge that `topopt verify` accepts in a prepared surface (mm), or None when its module cannot be imported."""
    try:
        return _tet_surface().MIN_EDGE_REJECT_MM
    except ImportError:
        return None


def _pieces(man):
    """(vertices, triangles, source id of each triangle) of a manifold3d solid. The source ids are those that `as_original`
    gave to the operands of the boolean."""
    m = man.to_mesh()
    ri, ro = np.asarray(m.run_index, np.int64), np.asarray(m.run_original_id, np.int64)
    tri = np.asarray(m.tri_verts, np.int64)
    src = np.empty(len(tri), np.int64)
    for k in range(len(ro)):
        src[ri[k] // 3: ri[k + 1] // 3] = ro[k]
    return np.asarray(m.vert_properties, np.float32)[:, :3].astype(np.float64), tri, src


def _boundary_loops(F: np.ndarray, nv: int):
    """The open boundary of a set of triangles as closed chains of vertex ids, in the direction of the faces (an edge is on
    the boundary when its twin, the same two vertices the other way, is missing). A list of chains, or None when two
    boundary edges leave one vertex (the boundary touches itself)."""
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
    b = d[~np.isin(d[:, 1] * nv + d[:, 0], d[:, 0] * nv + d[:, 1])]
    if not len(b):
        return []
    if len(np.unique(b[:, 0])) != len(b) or len(np.unique(b[:, 1])) != len(b):
        return None
    nxt = dict(zip(b[:, 0].tolist(), b[:, 1].tolist()))
    loops, seen = [], set()
    for s in sorted(nxt):
        if s in seen:
            continue
        chain, v = [], s
        while v in nxt and v not in seen:
            seen.add(v)
            chain.append(v)
            v = nxt[v]
        if v != s:
            return None
        loops.append(chain)
    return loops


def _clip_ears(F: np.ndarray, nv: int):
    """Drop the triangles that have two boundary edges (an ear: its three vertices are on the boundary). Returns (faces, count).

    A cut of the new surface by the wall can leave such a triangle, with one very short edge. Moved onto the old boundary,
    its three vertices lie on one polyline and it turns over. It has no area worth keeping (the facts give the count)."""
    count = 0
    while True:
        d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
        edge_face = np.tile(np.arange(len(F)), 3)
        boundary = ~np.isin(d[:, 1] * nv + d[:, 0], d[:, 0] * nv + d[:, 1])
        ears = np.flatnonzero(np.bincount(edge_face[boundary], minlength=len(F)) >= 2)
        if not len(ears):
            return F, count
        taken, drop = set(), []
        for f in ears.tolist():
            vs = set(F[f].tolist())
            if not (vs & taken):
                taken |= vs
                drop.append(f)
        F = np.delete(F, drop, axis=0)
        count += len(drop)


def _project_on_loop(V: np.ndarray, loop: list, pts: np.ndarray):
    """For each point, (edge index i, parameter t in 0..1, distance in mm) of the nearest point on the closed polyline
    `loop`; edge i runs from loop[i] to loop[i + 1]. The edges next to the SEAM_NEAREST_VERTICES nearest vertices are tried."""
    from scipy.spatial import cKDTree
    B, n = V[loop], len(loop)
    near = cKDTree(B).query(pts, k=min(SEAM_NEAREST_VERTICES, n))[1].reshape(len(pts), -1)
    best, edge, par = np.full(len(pts), np.inf), np.zeros(len(pts), np.int64), np.zeros(len(pts))
    for c in range(near.shape[1]):
        for e in ((near[:, c] - 1) % n, near[:, c]):
            a, ab = B[e], B[(e + 1) % n] - B[e]
            t = np.clip(np.einsum("ij,ij->i", pts - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0.0, 1.0)
            d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
            better = d < best
            best[better], edge[better], par[better] = d[better], e[better], t[better]
    return edge, par, best


def _loop_gap(V: np.ndarray, la: list, lb: list) -> float:
    """The largest distance (mm) from a vertex of the loop `la` to the closed polyline `lb`."""
    return float(_project_on_loop(V, lb, V[la])[2].max())


class _SeamFail(Exception):
    """A loop pair cannot be sewn in this way. The message is the reason."""


def _clip_polygon(V: np.ndarray, ring: list, normal: np.ndarray):
    """Triangulate a simple polygon by ear clipping. `ring` holds vertex ids, counter-clockwise around `normal`. Returns a
    list of (a, b, c) with the same orientation, or None when no ear is found (a polygon that is not simple)."""
    nz = normal / np.linalg.norm(normal)
    e1 = np.cross(nz, [1.0, 0.0, 0.0] if abs(nz[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nz, e1)
    xy = {v: (float(V[v] @ e1), float(V[v] @ e2)) for v in ring}

    def cross(a, b, c):
        return (xy[b][0] - xy[a][0]) * (xy[c][1] - xy[a][1]) - (xy[b][1] - xy[a][1]) * (xy[c][0] - xy[a][0])

    ring, tris = list(ring), []
    while len(ring) > 3:
        for i in range(len(ring)):
            a, b, c = ring[i - 1], ring[i], ring[(i + 1) % len(ring)]
            if cross(a, b, c) <= 0.0:
                continue
            if any(v not in (a, b, c) and cross(a, b, v) >= 0.0 and cross(b, c, v) >= 0.0 and cross(c, a, v) >= 0.0 for v in ring):
                continue
            tris.append((a, b, c))
            del ring[i]
            break
        else:
            return None
    return tris + [tuple(ring)] if cross(*ring) > 0.0 else None


def _followers(FA: np.ndarray, FB: np.ndarray, Q: list) -> np.ndarray:
    """The vertices of the new surface that follow the move of the loop Q: all of FB except Q itself and every vertex that the old
    part FA uses. After a first loop pair is sewn, FB holds vertices of the old part (the P vertices that Q vertices became, the
    chain vertices); a second loop within the reach must not move them, because the old part keeps its positions in tier 1."""
    return np.setdiff1d(np.unique(FB), np.union1d(np.asarray(Q, np.int64), np.unique(FA)))


def _sew_by_projection(V: np.ndarray, FA: np.ndarray, FB: np.ndarray, P: list, Q: list, limit: float, reach: float,
                       floor: float):
    """Sew the loop P of the old part to the loop Q of the new surface without a strip: every vertex of Q goes onto the
    polyline of P (its nearest point), and each side gets the vertices of the other as vertices of its boundary edges. Both
    boundaries are then the same chain of vertices, and the faces join edge to edge. Returns (FA, FB); `V` is changed in place
    (the vertices of Q move by their distance to P, 0.065 mm at most on the hook and the ladle).

    P is the chain of the old part in the direction of its faces, Q the chain of the new surface in the same direction in
    space (so the edges of the new surface run q_j+1 -> q_j). A vertex of Q within `limit` of a vertex of P becomes that vertex
    (the closest vertex of Q claims it first; an unclaimed Q vertex next to a claimed P vertex, and the short Q-Q edges of marching
    cubes, stay, so edges under `limit` can remain: the report counts `edges_under_verify_limit`). In this tier the old part keeps
    all its positions; only its boundary triangles are split (the vertices of FA do not follow a later loop, `_followers`). The other vertices of the new surface within `reach` of the loop move
    with it, by a share that falls to 0 at `reach`, so a thin triangle next to the loop does not turn over. Raises _SeamFail
    when the projection is not one turn in order, or a triangle of the new surface would still turn over. V is then as it was."""
    n, m = len(P), len(Q)
    edge, par, _dist = _project_on_loop(V, P, V[Q])
    plen = np.linalg.norm(V[[P[(i + 1) % n] for i in range(n)]] - V[P], axis=1)
    d_start, d_end = par * plen[edge], (1.0 - par) * plen[edge]
    to_end = d_end < d_start
    near = np.where(to_end, d_end, d_start)
    target = np.where(to_end, (edge + 1) % n, edge)
    snapped, claimed = {}, {}
    for j in np.argsort(near, kind="stable"):
        if near[j] < limit and int(target[j]) not in claimed:
            claimed[int(target[j])] = int(j)
            snapped[int(j)] = int(target[j])
    sigma = np.array([float(snapped[j]) if j in snapped else float(edge[j] + par[j]) for j in range(m)])
    step = np.diff(np.r_[sigma, sigma[0]]) % n
    if m > 1 and (np.any(step <= 0.0) or abs(float(step.sum()) - n) > 1e-6):
        raise _SeamFail("the vertices of the new boundary do not follow the old boundary in order (a loop that doubles back)")
    old = V.copy()
    try:
        img = {Q[j]: P[snapped[j]] for j in snapped}
        # where the vertices of Q go: onto a vertex of P, or onto the edge of P at the foot of the perpendicular
        goal = np.empty((m, 3))
        for j in range(m):
            if j in snapped:
                goal[j] = V[P[snapped[j]]]
            else:
                a, b = V[P[int(edge[j])]], V[P[(int(edge[j]) + 1) % n]]
                goal[j] = (a + par[j] * (b - a)).astype(np.float32).astype(np.float64)
        move = goal - V[Q]
        rest = _followers(FA, FB, Q)
        if len(rest):
            from scipy.spatial import cKDTree
            dd, ii = cKDTree(V[Q]).query(V[rest], k=min(SEAM_NEAREST_VERTICES, m), distance_upper_bound=reach)
            dd, ii = dd.reshape(len(rest), -1), ii.reshape(len(rest), -1)
            ok = np.isfinite(dd)
            w = np.where(ok, 1.0 / (np.where(ok, dd, 0.0) + SEAM_FOLLOW_SOFTENING * reach), 0.0)
            share = np.clip(1.0 - np.where(ok.any(axis=1), np.where(ok, dd, np.inf).min(axis=1), reach) / reach, 0.0, 1.0)
            d = (w[:, :, None] * move[np.where(ok, ii, 0)]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-300)[:, None]
            V[rest] = (V[rest] + share[:, None] * d).astype(np.float32).astype(np.float64)
        V[Q] = goal
        return _sew_split(V, old, FA, FB, P, Q, sigma, img, n, m, floor)
    except _SeamFail:
        V[:] = old
        raise


def _sew_split(V, old, FA, FB, P, Q, sigma, img, n, m, floor):
    """The rest of `_sew_by_projection`: the vertices have moved; split the boundary triangles of both sides along the
    common chain of vertices and check that no triangle of the new surface turned over. Returns (FA, FB)."""
    # the chain of vertices along the seam, in the direction of P
    ring = sorted([(float(k), P[k]) for k in range(n)] + [(float(sigma[j]), Q[j]) for j in range(m) if Q[j] not in img])
    R = [v for _s, v in ring]
    pos = {v: i for i, v in enumerate(R)}
    K = len(R)

    def split(faces, emap, origin, u, v, chain, clip):
        """Replace the face on the edge u -> v by triangles on the chain u ... v. On the old part the chain is straight
        (a fan from the third vertex); on the new surface it bends by the gap, so the polygon is clipped by ears."""
        fi = emap.get((u, v))
        if fi is None:
            raise _SeamFail("a boundary edge has no face")
        f = faces[fi]
        k = [t for t in range(3) if f[t] == u and f[(t + 1) % 3] == v]
        if not k:
            raise _SeamFail("a boundary edge is not in its face")
        c = f[(k[0] + 2) % 3]
        for a, b in zip(f, f[1:] + f[:1]):
            emap.pop((a, b), None)
        if clip:
            o = origin[fi]
            nrm = np.cross(old[FB[o, 1]] - old[FB[o, 0]], old[FB[o, 2]] - old[FB[o, 0]])
            new = _clip_polygon(V, chain + [c], nrm) if np.linalg.norm(nrm) > 0.0 else None
            if new is None:
                raise _SeamFail("the polygon of a boundary triangle of the new surface and the old boundary is not simple")
        else:
            new = [(chain[t], chain[t + 1], c) for t in range(len(chain) - 1)]
        faces[fi] = new[0]
        for t in new[1:]:
            faces.append(t)
            origin.append(origin[fi])
        for idx in [fi] + list(range(len(faces) - len(new) + 1, len(faces))):
            a3 = faces[idx]
            for a, b in zip(a3, a3[1:] + a3[:1]):
                emap[(a, b)] = idx

    # the old part: split the boundary triangle at each edge of P by the vertices of Q on it
    fa = [tuple(int(x) for x in f) for f in FA.tolist()]
    near_a = np.flatnonzero(np.isin(FA, P).any(axis=1)).tolist()
    emap_a = {(a, b): i for i in near_a for a, b in zip(fa[i], fa[i][1:] + fa[i][:1])}
    origin_a = list(range(len(fa)))
    for i in range(n):
        u, v = P[i], P[(i + 1) % n]
        k0, k1 = pos[u], pos[v]
        chain = [R[(k0 + t) % K] for t in range(((k1 - k0) % K) + 1)]
        if len(chain) > 2:
            split(fa, emap_a, origin_a, u, v, chain, False)
    # the new surface: the vertices of Q that became vertices of P, then split at each edge of Q by the vertices of P on it
    fb = [tuple(img.get(int(x), int(x)) for x in f) for f in FB.tolist()]
    emap_b = {(a, b): i for i, f in enumerate(fb) for a, b in zip(f, f[1:] + f[:1])}
    origin_b = list(range(len(fb)))
    for j in range(m):
        qa, qb = Q[j], Q[(j + 1) % m]
        u, v = img.get(qb, qb), img.get(qa, qa)
        k0, k1 = pos[u], pos[v]
        chain = [R[(k0 - t) % K] for t in range(((k0 - k1) % K) + 1)]
        if len(chain) > 2:
            split(fb, emap_b, origin_b, u, v, chain, True)
    FB2 = np.array(fb, np.int64).reshape(-1, 3)
    nrm_old = np.cross(old[FB[:, 1]] - old[FB[:, 0]], old[FB[:, 2]] - old[FB[:, 0]])[origin_b]
    nrm_new = np.cross(V[FB2[:, 1]] - V[FB2[:, 0]], V[FB2[:, 2]] - V[FB2[:, 0]])
    turned = (np.linalg.norm(nrm_old, axis=1) > 2 * floor) & (np.einsum("ij,ij->i", nrm_old, nrm_new) <= 0.0)
    if turned.any():
        k = int(np.flatnonzero(turned)[0])
        raise _SeamFail(f"{int(turned.sum())} triangles of the new surface would turn over (the first: area "
                        f"{0.5 * np.linalg.norm(nrm_old[k]):.2g} mm2 before, {0.5 * np.linalg.norm(nrm_new[k]):.2g} after)")
    return np.array(fa, np.int64).reshape(-1, 3), FB2


def _zip_loops(V: np.ndarray, P: list, Q: list) -> list:
    """The triangles between two close loops, as a strip. The edges p_i -> p_i+1 of P lack their twin in the old part, and the
    edges q_j+1 -> q_j of Q lack theirs in the new surface (Q is the chain of the new surface, reversed). Each step takes the
    triangle with the shorter new cross edge; the strip has len(P) + len(Q) triangles and closes."""
    n, m = len(P), len(Q)
    j0 = int(np.argmin(np.linalg.norm(V[Q] - V[P[0]], axis=1)))
    Q = Q[j0:] + Q[:j0]
    tris, i, j = [], 0, 0
    while i < n or j < m:
        pi, qj = P[i % n], Q[j % m]
        if i == n:
            adv_p = False
        elif j == m:
            adv_p = True
        else:
            adv_p = np.linalg.norm(V[P[(i + 1) % n]] - V[qj]) <= np.linalg.norm(V[pi] - V[Q[(j + 1) % m]])
        if adv_p:
            tris.append((P[(i + 1) % n], pi, qj))
            i += 1
        else:
            tris.append((qj, Q[(j + 1) % m], pi))
            j += 1
    return tris


def _weld_seam(V: np.ndarray, F: np.ndarray, rank: np.ndarray, limit: float, floor: float):
    """Collapse the edges shorter than `limit` that touch the seam (a vertex with rank 0 or 1), then the edges that two fold faces
    share. Returns (faces, collapses, moves, guard): `moves` lists the distance (mm) that each removed vertex of the old loop
    (rank 0) went, `guard` is true when SEAM_WELD_PASSES ended a weld that still collapsed edges.

    A collapse moves the vertex of the larger rank onto the other (rank 0: a vertex of the old loop, 1: of the new surface, 2: any
    other). The vertices of the new surface move first, but two vertices of the old loop that are closer than `limit` become one,
    and so does the pair of a fold (any length): that is the one way in which tier 2 changes the positions of the old part (see
    `old_vertices_removed` and `old_removed_move_max_mm` in the seam facts). A collapse is made only when the link condition holds
    (the surface stays a closed manifold) and when no triangle that has a normal turns over. Where the two loops of the seam
    cross, the strip between them has a twist (a fold); the cross edges there are the shortest, so the weld removes the twist."""
    nv = len(V)
    F = F.copy()
    alive = np.ones(len(F), bool)
    flat = F.ravel()
    face_of = np.argsort(flat, kind="stable") // 3
    ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=nv))])
    inc: dict = {}

    def faces_of(v):
        s = inc.get(v)
        if s is None:
            s = inc[v] = {f for f in face_of[ptr[v]:ptr[v + 1]].tolist() if alive[f]}
        return s

    def normal(t):
        return np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]])

    gone = np.zeros(nv, bool)
    moves: list = []

    def collapse(keep, rem) -> bool:
        fk, fr = faces_of(keep), faces_of(rem)
        shared = fk & fr
        if len(shared) != 2:
            return False
        apex = {int(x) for f in shared for x in F[f]} - {keep, rem}
        around_k = {int(x) for f in fk for x in F[f]} - {keep, rem}
        around_r = {int(x) for f in fr for x in F[f]} - {keep, rem}
        if len(apex) != 2 or (around_k & around_r) != apex:
            return False
        moved = fr - shared
        for f in moved:
            t = F[f].copy()
            old = normal(t)
            t[t == rem] = keep
            new = normal(t)
            if np.linalg.norm(old) > 2 * floor and np.dot(old, new) <= 0.0:
                return False
        for f in moved:
            F[f][F[f] == rem] = keep
            fk.add(f)
        for f in shared:
            alive[f] = False
            for x in F[f]:
                faces_of(int(x)).discard(f)
        fr.clear()
        gone[rem] = True
        if rank[rem] == 0:
            moves.append(float(np.linalg.norm(V[rem] - V[keep])))
        return True

    def seam_edges():
        edges = set()
        for v in np.flatnonzero((rank < 2) & ~gone).tolist():
            for f in faces_of(v):
                a, b, c = (int(x) for x in F[f])
                edges.update((min(x, y), max(x, y)) for x, y in ((a, b), (b, c), (c, a)) if v in (x, y))
        return edges

    total, guard = 0, False
    for _ in range(SEAM_WELD_PASSES):
        short = sorted((float(np.linalg.norm(V[x] - V[y])), x, y) for x, y in seam_edges())
        n = 0
        for length, x, y in short:
            if length >= limit:
                break
            if gone[x] or gone[y]:
                continue
            keep, rem = (x, y) if (rank[x], x) <= (rank[y], y) else (y, x)
            n += collapse(keep, rem)
        total += n
        if n == 0:
            break
    else:
        guard = True
    # the folds that stay are twists of the strip where the two loops cross: collapse the edge that two fold faces share
    ts = _tet_surface()
    for _ in range(SEAM_WELD_PASSES):
        live = np.flatnonzero(alive)
        f1, f2 = ts.fold_face_pairs(V, F[live])
        n = 0
        for a, b in zip(live[f1].tolist(), live[f2].tolist()):
            if not (alive[a] and alive[b]):
                continue
            x, y = sorted(set(F[a].tolist()) & set(F[b].tolist()))[:2]
            if gone[x] or gone[y] or (rank[x] >= 2 and rank[y] >= 2):
                continue
            keep, rem = (x, y) if (rank[x], x) <= (rank[y], y) else (y, x)
            n += collapse(keep, rem)
        total += n
        if n == 0:
            break
    else:
        guard = True
    return F[alive], total, moves, guard


def _graft(cur_man, new_man, zone_man, pitch: float | None):
    """The part `cur_man` with the zone `zone_man` replaced by `new_man`, and the seam sewn. Returns (mesh, facts, why).

    (cur - zone) + (new ^ zone) leaves a seam that is a mess. The old surface and the new surface are two triangulations of
    one surface: along the wall of the zone they differ by 0 to 65 um (float32 rounding included), and they cross each other
    many times. The boolean cannot tell which one is higher, so a ribbon of sliver triangles of the wall stays open, with
    folds (zero-thickness fins) and pinch vertices; `topopt verify` cannot mesh that.

    Here the two pieces are made apart (A = cur - zone, B = new ^ zone), the triangles of the wall (the faces that come from
    the zone) are dropped, and the open boundary loops are sewn. Each loop of A is matched to one loop of B (the nearest, by
    the Hausdorff gap; the gap must be under SEAM_MATCH_NODES nodes). Tier 1, `_sew_by_projection`: the vertices of the loop
    of B move onto the loop of A (to a vertex of A when it is within the shortest edge that verify accepts), the boundary
    triangles of A are split by the new vertices, the boundary triangles of B are re-cut, and the inner vertices of B near
    the loop move a decaying share. The two loops then are one edge chain: no strip, no fold. A flip check against the
    normals of the pieces refuses a move that turns a triangle over. Tier 2 (a loop pair does not project: a loop that doubles
    back, a polygon that is not simple, or a triangle that would turn over): `_zip_loops` makes a strip between the loops, and `_weld_seam`
    collapses the cross edges that are shorter than the shortest edge verify accepts (link condition, flip check) and the
    fold pairs of the strip. A strip between two loops that cross has a twist at each crossing, so a few fold edges can stay
    there (the facts count them). Tier 3 is the caller's: `why` names the reason that the seam was not sewn and the caller
    uses the plain boolean.
    Returns (mesh, facts, why); `why` is None when the seam is sewn and the sewn mesh is a closed manifold."""
    import manifold3d as m3d
    from scipy.optimize import linear_sum_assignment
    try:
        ts = _tet_surface()
    except ImportError as exc:
        return None, None, f"topopt.tet.surface cannot be imported ({exc})"
    cm, nm, zm = cur_man.as_original(), new_man.as_original(), zone_man.as_original()
    A, B = cm - zm, nm ^ zm
    for name, s in (("part minus zone", A), ("new surface in zone", B)):
        if s.status() != m3d.Error.NoError or s.is_empty():
            return None, None, f"the {name} is empty or not a valid solid ({s.status()})"
    zid = zm.original_id()
    boolean_volume = float(A.volume() + B.volume())          # the two pieces do not overlap: this is the volume of the boolean
    VA, FA, sA = _pieces(A)
    VB, FB, sB = _pieces(B)
    wall = int((sA == zid).sum() + (sB == zid).sum())
    FA, FB = FA[sA != zid], FB[sB != zid]
    FB, ears = _clip_ears(FB, len(VB))
    LA, LB = _boundary_loops(FA, len(VA)), _boundary_loops(FB, len(VB))
    if LA is None or LB is None:
        return None, None, "a boundary loop of the cut touches itself (a pinch)"
    if not LA or len(LA) != len(LB):
        return None, None, f"the cut of the old part has {len(LA)} boundary loops and the cut of the new surface {len(LB)}"
    off = len(VA)
    V = np.vstack([VA, VB])
    FB = FB + off
    LB = [[v + off for v in reversed(l)] for l in LB]
    cost = np.array([[max(_loop_gap(V, a, b), _loop_gap(V, b, a)) for b in LB] for a in LA])
    ia, ib = linear_sum_assignment(cost)
    gap = float(cost[ia, ib].max())
    if pitch is not None and gap > SEAM_MATCH_NODES * pitch:
        return None, None, (f"the loops of the two cuts are {gap:.3f} mm apart, more than {SEAM_MATCH_NODES:g} nodes "
                            f"({SEAM_MATCH_NODES * pitch:.3f} mm): they are not one seam")
    limit = ts.MIN_EDGE_REJECT_MM
    qlen = np.concatenate([np.linalg.norm(V[l] - V[np.roll(l, -1)], axis=1) for l in LB])
    reach = SEAM_REACH_NODES * pitch if pitch is not None else float(np.median(qlen))
    floor = _area_floor(V, pitch if pitch is not None else float(np.median(qlen)))
    projected, strip, strip_loops, why_strip = 0, [], [], []
    for a, b in zip(ia, ib):
        try:
            FA, FB = _sew_by_projection(V, FA, FB, LA[a], LB[b], limit, reach, floor)
            projected += 1
        except _SeamFail as exc:
            why_strip.append(str(exc))
            strip += _zip_loops(V, LA[a], LB[b])
            strip_loops.append((LA[a], LB[b]))
    welded, old_moves, guard = 0, [], False
    if strip:
        rank = np.full(len(V), 2, np.int64)
        for la, lb in strip_loops:
            rank[lb] = 1
            rank[la] = 0
        F, welded, old_moves, guard = _weld_seam(V, np.vstack([FA, FB, np.asarray(strip, np.int64)]), rank, limit, floor)
    else:
        F = np.vstack([FA, FB])
    import trimesh
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    new_id = np.cumsum(used) - 1
    mesh = trimesh.Trimesh(V[used], new_id[F], process=False)
    if not ts.surface_is_closed_manifold(mesh.vertices, mesh.faces) or mesh.volume <= 0:
        return None, None, "the sewn part is not a closed manifold"
    facts = {"method": "sewn", "loops": len(LA), "loops_joined_edge_to_edge": projected, "loops_joined_by_strip": len(strip_loops),
             "loop_gap_max_mm": _f(gap, 4), "wall_triangles_dropped": wall, "ear_triangles_dropped": ears,
             "strip_triangles": len(strip), "strip_edges_welded": int(welded), "short_edge_mm": limit,
             "volume_vs_boolean_mm3": _f(float(mesh.volume) - boolean_volume, 4)}
    if why_strip:
        facts["why_strip"] = why_strip
    if strip:
        facts["old_vertices_removed"] = len(old_moves)
        facts["old_removed_move_max_mm"] = _f(max(old_moves), 4) if old_moves else 0.0
        if guard:
            facts["weld_stopped_by_guard"] = True
    return mesh, facts, None


def _region_residual(man, keep, region) -> float:
    """The volume (mm3) in which the solid `man` differs from the solid `keep` inside `region` (added plus removed)."""
    return float(((man ^ region) - keep).volume() + ((keep ^ region) - man).volume())


def _swap_inside(base, patch, region, pitch: float | None = None, exact_inside: bool = False):
    """(base - region) + (patch ^ region) with a sewn seam. Returns (mesh, solid, seam facts, why). When the seam cannot be sewn
    (`why` names the reason), the plain boolean is used and the facts say so. mesh and solid are None when the boolean fails.

    exact_inside: the patch must stay exact inside `region` (the undo of a change in a frozen region). The sewing moves the vertices
    of the patch side near the cut (the cut of `base` is the edit, not a second triangulation of the same surface, so the loops can
    be 0.2 to 0.9 mm apart). The result is measured: `region_residual_mm3` in the facts. When the patch differs in the region by
    FROZEN_TOL_MM3 or more, the plain boolean is used (exact by construction) and the facts say so."""
    import manifold3d as m3d
    mesh, facts, why = _graft(base, patch, region, pitch)
    if mesh is not None:
        man = _manifold(mesh)
        if man is None:
            why = "the sewn part is not a valid solid for the boolean (manifold3d refused it)"
        elif not exact_inside:
            return mesh, man, facts, None
        else:
            residual = _region_residual(man, patch, region)
            facts["region_residual_mm3"] = _f(residual, 6)
            if residual < FROZEN_TOL_MM3:
                return mesh, man, facts, None
            why = (f"the sewn undo changes the region by {residual:.4f} mm3 (the tolerance is {FROZEN_TOL_MM3} mm3): the sewing "
                   "moved the frozen side")
    out = (base - region) + (patch ^ region)
    if out.status() != m3d.Error.NoError or out.is_empty():
        return None, None, {"method": "boolean", "why_not_sewn": why}, f"the boolean failed: {out.status()}"
    return _mesh_of(out), out, {"method": "boolean", "why_not_sewn": why}, None


ZONE_NODES = 2.5     # the splice zone reaches this many nodes beyond the changed nodes
EDGE_NODES = 3       # a changed node this close to the edge of the crop box stops the step


def _splice(cur_man, Tn: np.ndarray, Tc: np.ndarray, crop: Crop, chg: np.ndarray, simplify: bool = False):
    """Put the new surface into the part inside a zone around the changed nodes.
    Returns (mesh, solid, zone bounding box (lo, hi) in mm, seam facts, why). `solid` is the manifold3d solid of the mesh;
    `why` is None or the reason.

    The zone is the set of nodes within 2.5 nodes of a changed node (a smooth blob, not a box: a long diagonal strut
    does not pull its whole bounding box into the splice). The part is cut open by the zone, and the surface of the
    new field in the zone is put in: part = (part - zone) + (new solid ^ zone). Where the field did not change, the new
    surface is the old one to within the error of the field (a few thousandths of a mm), but it is another triangulation
    of it: the seam is sewn along the two cuts (`_graft`), not left to the boolean. Outside the zone the triangles of
    the part stay as they are."""
    import manifold3d as m3d
    idx = np.argwhere(chg)
    a, b = idx.min(0), idx.max(0)
    shp = np.array(crop.shape)
    if np.any(a < EDGE_NODES) or np.any(b > shp - 1 - EDGE_NODES):
        faces = [f"{'xyz'[i]}-" for i in range(3) if a[i] < EDGE_NODES] + \
            [f"{'xyz'[i]}+" for i in range(3) if b[i] > shp[i] - 1 - EDGE_NODES]
        near_mm = min([float(a[i]) * crop.p for i in range(3)] + [float(shp[i] - 1 - b[i]) * crop.p for i in range(3)])
        return None, None, (None, None), None, (
            f"the edit reaches the face {', '.join(faces)} of the crop box (the nearest changed node is {near_mm:.2f} mm "
            f"from it; the limit is {EDGE_NODES} nodes = {EDGE_NODES * crop.p:.2f} mm), so the box is too small for this "
            "edit. Options: a smaller radius_mm (per_mark or radius_scale), or the rerun_with_hints route")
    pad = int(math.ceil(ZONE_NODES)) + 3
    lo_i, hi_i = np.maximum(a - pad, 0), np.minimum(b + pad + 1, shp)
    blk = tuple(slice(int(x), int(y)) for x, y in zip(lo_i, hi_i))
    zone = ndi.distance_transform_edt(~chg[blk]) <= ZONE_NODES
    zi = np.argwhere(zone)
    z_lo = crop.lo + crop.p * (lo_i + zi.min(0))
    z_hi = crop.lo + crop.p * (lo_i + zi.max(0))
    z_mesh = _surface_mesh(_signed_from_mask(zone, crop.p, cap=BAND_MM), crop, lo_i)
    z_man = _manifold(z_mesh)
    new_man = _manifold(_surface_mesh(Tn[blk], crop, lo_i))
    if z_man is None or new_man is None:
        return None, None, (z_lo, z_hi), None, ("the new surface or the zone is not a valid solid for the boolean "
                                                "(manifold3d refused it)")
    mesh, out, seam, why = _swap_inside(cur_man, new_man, z_man, crop.p)
    if why:
        return None, None, (z_lo, z_hi), seam, why
    if simplify:
        mesh, out, why = _simplified(mesh)
        if why:
            return None, None, (z_lo, z_hi), seam, why
    return mesh, out, (z_lo, z_hi), seam, None


def _simplified(mesh):
    """(mesh, solid, why) of the float32 mesh after `Manifold.simplify(1e-5)`: it merges the vertices that float32 puts together, on
    the float32 mesh, and casts again (pitfalls/float32-stl-reload-pinch)."""
    import manifold3d as m3d
    man = _manifold(mesh)
    if man is None:
        return None, None, "the float32 mesh of the boolean is not a valid solid (a pinch)"
    out = man.simplify(1e-5)
    if out.status() != m3d.Error.NoError or out.is_empty():
        return None, None, f"the simplify step failed: {out.status()}"
    return _mesh_of(out), out, None


# ---- options, planes, marks -----------------------------------------------------------------------------
AXES = {"x": 0, "y": 1, "z": 2}


def _axis_of(axis):
    """The axis index of x|y|z (any case), of 0|1|2, or of a 3-vector with one non-zero component; else None."""
    if isinstance(axis, str) and axis.strip().lower() in AXES:
        return AXES[axis.strip().lower()]
    if isinstance(axis, int) and not isinstance(axis, bool) and 0 <= axis <= 2:
        return axis
    if isinstance(axis, (list, tuple)) and len(axis) == 3:
        v = [fb._num(x) for x in axis]
        nz = [i for i, x in enumerate(v) if x is not None and x != 0]
        if None not in v and len(nz) == 1:
            return nz[0]
    return None


_AT_KEYS = ("at_mm", "at", "coordinate_mm", "coordinate", "position_mm", "position", "offset_mm", "offset")


def _plane_from(spec, source: str):
    """A mirror plane from {axis|normal: x|y|z, at_mm: number} (a few spellings are read), or None."""
    if not isinstance(spec, dict):
        return None
    ai = _axis_of(spec.get("axis", spec.get("normal")))
    if ai is None:
        return None
    at = None
    for key in _AT_KEYS:
        at = fb._num(spec.get(key))
        if at is not None:
            break
    if at is None:
        at = 0.0 if not any(k in spec for k in _AT_KEYS) else None
    if at is None:
        return None
    return {"axis": "xyz"[ai], "axis_index": ai, "at_mm": float(at), "source": source}


def _planes_of_run(run: dict) -> tuple[list[dict], list[str]]:
    """The mirror planes that run.json records, and facts about the planes that are not used.

    Layout of the engine item A4a (CONTRACTS 26.1): `run["mirror"]` is a dict {"planes": [{axis, at_mm, align, parity,
    k, plane_index, ...}], design_voxels, load_asymmetry, end}; the YAML key is a list of up to 3 planes with distinct
    axes (`normal`, `at_mm`). A list of planes, or one plane, in `run["mirror"]` is read too. The planes of
    different axes commute, so the mirror images are the group of the reflections; of two planes on one axis, the
    first is used (the group would not end) and the fact says so."""
    node = (run or {}).get("mirror")
    if isinstance(node, dict) and isinstance(node.get("planes"), list):
        items = node["planes"]
    elif isinstance(node, list):
        items = node
    elif isinstance(node, dict):
        items = [node]
    else:
        items = []
    planes, notes = [], []
    for i, x in enumerate(items):
        pl = _plane_from(x, f"run.json (mirror, plane {i + 1})" if len(items) > 1 else "run.json (mirror)")
        if pl is None:
            notes.append(f"plane {i + 1} of run.json mirror is not read ({x!r}: it has no axis and at_mm)")
        elif any(q["axis"] == pl["axis"] for q in planes):
            notes.append(f"plane {i + 1} of run.json mirror (axis {pl['axis']}, at {pl['at_mm']:g} mm) has the axis of an "
                         "earlier plane: it is not used (the images would not end). Option: mirror_plane for one plane")
        else:
            planes.append(pl)
    return planes[:3], notes


def _plane_of_run(run: dict):
    """The first mirror plane of run.json, or None (see `_planes_of_run`)."""
    planes, _ = _planes_of_run(run)
    return planes[0] if planes else None


def _norm_options(options) -> tuple[dict, list[str]]:
    """The effective options of `apply_marks`, and facts about the options that were not understood."""
    opt = json.loads(json.dumps(OPTION_DEFAULTS))
    problems: list[str] = []
    if options is not None and not isinstance(options, dict):
        return opt, [f"options must be an object; {type(options).__name__} was given and is ignored"]
    for k, v in (options or {}).items():
        if k not in opt:
            problems.append(f"option {k!r} is not known (known: {', '.join(opt)}); ignored")
        elif k in ("radius_scale", "pitch_mm", "max_crop_voxels", "max_total_voxels", "frozen_keep_mm", "frozen_full_mm",
                   "close_mm", "neck_min_mm3", "parent_tol_mm", "loose_body_mm3", "max_odd_rays"):
            x = fb._num(v)
            lo_ok = 0 if k in ("frozen_keep_mm", "frozen_full_mm", "neck_min_mm3", "parent_tol_mm", "loose_body_mm3",
                               "max_odd_rays") else 1e-9
            if x is None or x < lo_ok or (k in ("max_crop_voxels", "max_total_voxels") and x < 1000):
                problems.append(f"option {k}={v!r} is not a valid number; the default is used")
            else:
                opt[k] = x
        elif k == "parent_check":
            if v in ("report", "enforce"):
                opt[k] = v
            else:
                problems.append(f"option parent_check={v!r} is not report or enforce; report is used")
        elif k == "smooth_mode":
            if v in SMOOTH_MODES:
                opt[k] = v
            else:
                problems.append(f"option smooth_mode={v!r} is not one of {', '.join(SMOOTH_MODES)}; auto is used")
        elif k == "remove_mode":
            if v in REMOVE_MODES:
                opt[k] = v
            else:
                problems.append(f"option remove_mode={v!r} is not cut or shave; cut is used")
        elif k == "mirror_plane":
            pl = _plane_from(v, "call option mirror_plane")
            if pl is None:
                problems.append(f"option mirror_plane={v!r} is not {{axis: x|y|z, at_mm: number}}; ignored")
            else:
                opt[k] = pl
        elif k == "skip":
            ids = v if isinstance(v, list) else [v]
            opt[k] = [i for i in ids if isinstance(i, int) and not isinstance(i, bool)]
            if len(opt[k]) != len(ids):
                problems.append("option skip holds a value that is not a mark id; it is ignored")
        elif k == "per_mark":
            if not isinstance(v, dict):
                problems.append("option per_mark must be {mark id: {values}}; ignored")
                continue
            per = {}
            for mid, vals in v.items():
                try:
                    key = int(mid)
                except (TypeError, ValueError):
                    problems.append(f"per_mark key {mid!r} is not a mark id; ignored")
                    continue
                if not isinstance(vals, dict):
                    problems.append(f"per_mark[{mid}] must be an object; ignored")
                    continue
                bad = [x for x in vals if x not in PER_MARK_KEYS]
                if bad:
                    problems.append(f"per_mark[{mid}] has unknown keys {bad} (known: {', '.join(PER_MARK_KEYS)}); ignored")
                per[key] = {x: y for x, y in vals.items() if x in PER_MARK_KEYS}
            opt[k] = per
        elif k == "check_frozen_exact":
            opt[k] = bool(v)
    d0 = opt["frozen_keep_mm"] if opt["frozen_keep_mm"] is not None else DEFAULTS["frozen_keep_mm"][0]
    d1 = opt["frozen_full_mm"] if opt["frozen_full_mm"] is not None else DEFAULTS["frozen_full_mm"][0]
    if opt["frozen_keep_mm"] is not None and opt["frozen_full_mm"] is None and d0 > d1:
        opt["frozen_full_mm"] = d0 + (DEFAULTS["frozen_full_mm"][0] - DEFAULTS["frozen_keep_mm"][0])
        problems.append(f"frozen_keep_mm {d0} is above the default frozen_full_mm; frozen_full_mm is {opt['frozen_full_mm']}")
    return opt, problems


def _parameters(opt: dict) -> tuple[dict, list[str]]:
    """The numbers of the operators: the pipeline default, or the value that the call gave. And the names that are defaults."""
    prm = {k: v[0] for k, v in DEFAULTS.items()}
    overridden = []
    for k in ("frozen_keep_mm", "frozen_full_mm", "close_mm", "neck_min_mm3", "parent_tol_mm"):
        if opt.get(k) is not None:
            prm[k] = float(opt[k])
            overridden.append(k)
    return prm, overridden


KIND_PARAMS = {
    "remove": ("frozen_keep_mm", "remove_close_mm", "remove_margin_mm", "remove_excl_mm", "remove_dom_mm", "foot_rho_mm",
               "foot_max_mm3", "cut_smooth_mm", "polish_mm", "polish_zone_mm", "parent_tol_mm"),
    "shave": ("frozen_keep_mm", "frozen_full_mm"),
    "add": ("close_mm", "neck_offset_mm", "neck_min_mm3", "frozen_keep_mm", "frozen_full_mm"),
    "iso": ("iso_sigma_mm", "frozen_keep_mm", "frozen_full_mm"),
    "fair": ("fair_sigma_mm", "path_step_mm", "frozen_keep_mm", "frozen_full_mm"),
    "tube": ("tube_window_mm", "tube_ext_max_mm", "tube_blend_end_mm", "path_step_mm", "web_flatness",
             "frozen_keep_mm", "frozen_full_mm"),
}


def _mirror_points(pts: np.ndarray, plane: dict) -> np.ndarray:
    out = np.array(pts, float, copy=True)
    out[:, plane["axis_index"]] = 2.0 * plane["at_mm"] - out[:, plane["axis_index"]]
    return out


def _path_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The smallest distance between two polylines (from points along them, 0.25 mm apart)."""
    from scipy.spatial import cKDTree
    pa, pb = _densify(a, 0.25), _densify(b, 0.25)
    return float(cKDTree(pb).query(pa, workers=1)[0].min())


# ---- the state of a call, the guards, the facts of one step ------------------------------------------------
class _Guard(Exception):
    """A crop box that is above a voxel-count guard. The facts and the options go to the report."""

    def __init__(self, reason: str, facts: dict, options: list):
        super().__init__(reason)
        self.reason, self.facts, self.options = reason, facts, options


class _Rollback(Exception):
    """The edit made a part that fails a check. The step is undone, and `reason` is the fact."""

    def __init__(self, reason: str, code: str, facts: dict | None = None):
        super().__init__(reason)
        self.reason, self.code, self.facts = reason, code, facts or {}


class _State:
    """What the steps of one call share: the current part, the labels, the options and the counters."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.cells_used = 0
        self.boxes: list = []

    def mass_g(self, vol_mm3: float | None):
        return None if (vol_mm3 is None or not self.density) else vol_mm3 / 1000.0 * self.density


def _plan_crop(st: _State, pts: np.ndarray, pad: float, margin: float = CROP_MARGIN_MM):
    """The crop box around the points: pad mm each side, cut to the bounds of the part plus `margin` mm (at least 3 mm,
    and at least the reach of the operator, so a filter reads real field and not the box face). None when empty."""
    margin = max(float(margin), CROP_MARGIN_MM)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    lo = np.maximum(lo, st.bounds[0] - margin)
    hi = np.minimum(hi, st.bounds[1] + margin)
    if np.any(hi <= lo):
        return None
    return Crop(lo, np.ceil((hi - lo) / st.pitch).astype(int) + 1, st.pitch)


def _guard_crop(st: _State, crop: Crop, what: str) -> None:
    """Count the nodes of a crop box. Above a guard: a fact with options, not a crash."""
    cells = _cells(crop.shape)
    one, total = st.opt["max_crop_voxels"], st.opt["max_total_voxels"]
    facts = {"crop_nodes": cells, "crop_shape": list(crop.shape), "pitch_mm": st.pitch, "what": what,
             "max_crop_voxels": int(one), "max_total_voxels": int(total), "nodes_used_by_earlier_steps": st.cells_used,
             "memory_estimate_mb": round(cells * BYTES_PER_NODE / 1e6) + FIXED_MB,
             "memory_estimate_note": f"{BYTES_PER_NODE} bytes per node, plus about {FIXED_MB} MB for the process"}
    if cells > one:
        ext, fit = crop.hi - crop.lo, st.pitch * (cells / one) ** (1.0 / 3.0)
        for _ in range(200):                              # the node count is ceil(extent / pitch) + 1 on each axis
            if _cells(np.ceil(ext / fit).astype(int) + 1) <= one:
                break
            fit *= 1.03
        fit = math.ceil(fit * 100) / 100
        facts["pitch_that_fits_mm"] = fit
        raise _Guard(f"the crop box for {what} has {cells:,} nodes, above the guard of {int(one):,}", facts,
                     [f"options: pitch_mm {fit} (a coarser grid)", "fewer or shorter marks in this call",
                      f"max_crop_voxels above {cells:,} (about {facts['memory_estimate_mb']:,} MB at this size, "
                      f"{BYTES_PER_NODE} bytes per node plus about {FIXED_MB} MB for the process)"])
    if st.cells_used + cells > total:
        raise _Guard(f"this step needs {cells:,} nodes; the steps of this call used {st.cells_used:,} of the guard "
                     f"{int(total):,}", facts,
                     ["fewer marks in this call (call again for the rest, with skip)", f"max_total_voxels above "
                      f"{st.cells_used + cells:,}"])
    st.cells_used += cells


def _reload_facts(mesh):
    """The facts of the mesh after the trip through STL bytes (float32, vertices merged), and that mesh."""
    import trimesh
    data = mesh.export(file_type="stl")
    back = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh")
    return mesh_facts(back), back


def _components(mesh):
    import trimesh
    return trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))


def _faces_signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """The signed volume of the solid that a set of faces closes (signed tetrahedra): positive when the faces look
    outward (a body), negative when they look inward (a cavity inside the part)."""
    t = np.asarray(vertices, float)[faces]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6.0


def _faces_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """The absolute volume of the solid (or the cavity) that a set of faces closes."""
    return abs(_faces_signed_volume(vertices, faces))


def _loose_limit(st: _State) -> float:
    """The volume below which a new body is a sliver of the boolean (mm3): the option loose_body_mm3, else the larger
    of 10 voxels of the crop grid and 0.001 mm3."""
    v = st.opt.get("loose_body_mm3")
    return float(v) if v is not None else max(10.0 * st.pitch ** 3, 1e-3)


def _body_facts(st: _State, mesh):
    """The bodies of the mesh, the largest first: faces, volume (mm3), whether the faces close a cavity inside the part
    (they look inward), and whether the body holds a fixed or load voxel."""
    out = []
    for c in _components(mesh):
        sv = _faces_signed_volume(mesh.vertices, mesh.faces[c])
        out.append({"faces": c, "volume_mm3": abs(sv), "cavity": sv < 0,
                    "holds_fixed_or_load": bool(st.labels is not None and st.labels.holds_fixed_or_load(mesh, c))})
    out.sort(key=lambda i: -i["volume_mm3"])
    return out


def _body_list(bodies: list, keep: int = 6) -> list:
    """The bodies as a fact: volume and whether they hold a fixed or load region (the largest `keep`)."""
    return [{"volume_mm3": _f(b["volume_mm3"], 4), "triangles": int(len(b["faces"])), "cavity": b["cavity"],
             "holds_fixed_or_load": b["holds_fixed_or_load"]} for b in bodies[:keep]]


def _drop_bodies(st: _State, mesh):
    """Drop the slivers: bodies below the loose limit that hold no fixed or load voxel, the smallest first, until the
    count is not above the count before. Returns (mesh, summary of the dropped bodies or None, all bodies).

    A body above the limit stays, and the step is rolled back by the caller: a part that falls in two is not an edit
    that the tool makes silently. The boolean can leave bodies of no volume (four triangles) where the new surface
    meets the old one; these are the slivers."""
    import trimesh
    bodies = _body_facts(st, mesh)
    limit = _loose_limit(st)
    n = len(bodies)
    dropped = []
    for it in sorted([b for b in bodies if b["volume_mm3"] < limit and not b["holds_fixed_or_load"]],
                     key=lambda b: b["volume_mm3"]):
        if n <= st.bodies:
            break
        dropped.append(it)
        n -= 1
    if not dropped:
        return mesh, None, bodies
    gone = np.concatenate([it["faces"] for it in dropped])
    keep = np.ones(len(mesh.faces), bool)
    keep[gone] = False
    out = trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)
    out.remove_unreferenced_vertices()
    return out, {"count": len(dropped), "triangles": int(len(gone)),
                 "total_volume_mm3": _f(sum(it["volume_mm3"] for it in dropped), 6),
                 "largest_volume_mm3": _f(max(it["volume_mm3"] for it in dropped), 6),
                 "limit_mm3": _f(limit, 6)}, bodies


PINCH_SHIFT_MM = 2e-4      # pinched vertices move this far apart, into their own solid (pipeline: build_c3.py f32_fix)
PINCH_PASSES = 6           # as f32_fix


def _separate_pinched(mesh):
    """A copy of the mesh in which vertices with the same float32 position are moved apart, or None when there is none.
    Returns (mesh or None, passes, pairs left).

    Two vertices of the boolean that float32 puts at one point pinch the surface when the STL is read again
    (pitfalls/float32-stl-reload-pinch). As in the pipeline (build_c3.py f32_fix): in a group of vertices with one
    float32 position the first stays and the j-th other moves 2e-4 x j mm against its vertex normal, into its own
    solid, and the mesh goes to float32 again; up to 6 passes. Only the vertices of the groups move."""
    import trimesh
    V = np.asarray(mesh.vertices, np.float32).astype(np.float64)
    F = np.asarray(mesh.faces)
    moved, passes, left = False, 0, 0
    for _ in range(PINCH_PASSES):
        _, inv, cnt = np.unique(V, axis=0, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).ravel()
        dup = cnt[inv] > 1
        left = int(dup.sum())
        if not dup.any():
            break
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        vn = np.zeros_like(V)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]
        order = np.argsort(inv, kind="stable")
        first = np.r_[True, inv[order][1:] != inv[order][:-1]]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(len(V)), 0))
        rank = np.empty(len(V), np.int64)
        rank[order] = np.arange(len(V)) - grp_start
        mv = dup & (rank > 0)
        V[mv] -= (rank[mv, None] * PINCH_SHIFT_MM) * vn[mv]
        V = V.astype(np.float32).astype(np.float64)
        moved, passes = True, passes + 1
    if not moved:
        return None, 0, 0
    _, inv, cnt = np.unique(V, axis=0, return_inverse=True, return_counts=True)
    left = int((cnt[np.asarray(inv).ravel()] > 1).sum())
    return trimesh.Trimesh(V, F, process=False), passes, left


def _reload_with_pinch_repair(mesh, man, facts):
    """(mesh, solid, reload facts, passes) of a part that is not closed after the float32 reload: the pinched vertices are moved
    apart (`_separate_pinched`) and the part is checked again. The input comes back with passes 0 when the repair does not help."""
    mesh2, passes, _left = _separate_pinched(mesh)
    man2 = None if mesh2 is None else _manifold(mesh2)
    if man2 is not None:
        facts2, _ = _reload_facts(mesh2)
        if facts2["closed"]:
            return mesh2, man2, facts2, passes
    return mesh, man, facts, 0


def _frozen_change(st: _State, man_old, man_new, lo, hi):
    """(added, removed) volume in mm3 inside the frozen regions between two solids that differ in the box lo..hi, and the
    part of the regions in the box (a solid or None). (None, None, None) when the regions are not known."""
    import manifold3d as m3d
    F = getattr(st, "frozen_solid", None)
    if F is None:
        return None, None, None
    lo, hi = np.asarray(lo, float) - 1.0, np.asarray(hi, float) + 1.0
    box = m3d.Manifold.cube(tuple(float(x) for x in (hi - lo)), False).translate(tuple(float(x) for x in lo))
    Fz = F ^ box
    if Fz.is_empty():
        return 0.0, 0.0, Fz
    add = float(((Fz ^ man_new) - man_old).volume())
    rem = float(((Fz ^ man_old) - man_new).volume())
    return add, rem, Fz


def _splice_checked(st: _State, Tn, Tc, crop, chg, step: dict):
    """The splice, the frozen restore, the reload check, and the bodies rule. Returns (mesh, solid, reload facts).

    A part that fails the reload check is repaired in two ways, the local one first: pinched vertices move apart
    (only those vertices change); then a simplify of the whole float32 mesh (the triangles outside the box can change).
    A change inside a frozen region (the regions of the YAML, exact) is undone in the step; when the undo does not
    give a valid closed part, the step is rolled back."""
    last = "the splice did not run"
    base = None
    frozen_failed = None
    for how in ("plain", "separate", "simplify"):
        passes = 0
        if how == "simplify" and base is not None:
            # the plain splice (and its sewing) is not made again: the simplify takes the plain mesh
            blo, bhi, seam = base[1], base[2], base[3]
            mesh, man, why = _simplified(base[0])
        elif how == "plain" or how == "simplify":
            mesh, man, (blo, bhi), seam, why = _splice(st.man, Tn, Tc, crop, chg, simplify=(how == "simplify"))
            if why and "edge of the crop box" in why:
                raise _Rollback(why, "crop_edge")
            if how == "plain" and mesh is not None:
                base = (mesh, blo, bhi, seam)
        else:
            if base is None:
                continue
            mesh, passes, left = _separate_pinched(base[0])
            blo, bhi, seam = base[1], base[2], base[3]
            man = None if mesh is None else _manifold(mesh)
            why = "no vertices share a float32 position" if mesh is None else (
                None if man is not None else "the part with separated vertices is not a valid solid")
        if mesh is None or man is None:
            last = why or last
            continue
        restored = None
        add, rem, Fz = _frozen_change(st, st.man, man, blo, bhi)
        if add is not None and add + rem >= FROZEN_TOL_MM3:
            fmesh, fixed, fseam, _why = _swap_inside(man, st.man, Fz, st.pitch, exact_inside=True)
            if fmesh is None:
                frozen_failed = (f"the edit changes the frozen regions by {add + rem:.4f} mm3 and the undo is not a valid "
                                 "solid")
                last = frozen_failed
                continue
            mesh, man, restored = fmesh, fixed, _f(add + rem, 4)
            seam = {**(seam or {}), "frozen_undo": fseam}
        facts, _back = _reload_facts(mesh)
        undo_pinch = None
        if not facts["closed"] and restored is not None:
            # the boolean of the undo can put two vertices at one float32 position too: move those apart (the same local repair)
            mesh, man, facts, undo_pinch = _reload_with_pinch_repair(mesh, man, facts)
        if not facts["closed"]:
            last = ("the part is not closed after the float32 reload from STL bytes (open edges "
                    f"{facts['open_edges']}, non-manifold edges {facts['non_manifold_edges']})")
            if restored is not None:
                frozen_failed = (f"the edit changes the frozen regions by {restored} mm3, and the part with the change "
                                 "undone is not closed after the float32 reload")
            continue
        step["splice_zone_bbox_mm"] = {"lo": [_f(v) for v in blo], "hi": [_f(v) for v in bhi]}
        step["seam"] = seam
        if undo_pinch:
            how, passes = "separate", undo_pinch
        step["float32_repair"] = {"plain": None, "separate": f"pinched vertices moved apart by 2e-4 mm into their own "
                                                              f"solid ({passes} pass{'es' if passes != 1 else ''})",
                                  "simplify": "simplify(1e-5) of the whole float32 mesh"}[how]
        if restored is not None:
            step["frozen_restored_mm3"] = restored
        if facts["bodies"] > st.bodies:
            mesh2, dropped, bodies = _drop_bodies(st, mesh)
            limit = _loose_limit(st)
            after = facts["bodies"]
            if dropped:
                man2 = _manifold(mesh2)
                if man2 is None:
                    last = "the part without the loose body is not a valid solid"
                    continue
                facts2, _ = _reload_facts(mesh2)
                step["dropped_bodies"] = dropped
                if facts2["closed"] and facts2["bodies"] <= st.bodies:
                    return mesh2, man2, facts2
                after = facts2["bodies"]
            listing = ", ".join(f"{b['volume_mm3']:.4g}" + (" (a cavity inside the part)" if b["cavity"] else "")
                                + (" (holds fixed or load)" if b["holds_fixed_or_load"] else "") for b in bodies[:6])
            raise _Rollback(
                f"the edit splits the part: {st.bodies} {'body' if st.bodies == 1 else 'bodies'} before, {after} after. "
                f"Bodies of the new surface (mm3): {listing}. "
                + (f"{dropped['count']} {'sliver was' if dropped['count'] == 1 else 'slivers were'} dropped and the part is still in "
                   f"{after} bodies. " if dropped else "")
                + f"A body is dropped only when it holds no fixed or load region and is below {limit:.4g} mm3. Options: a "
                "smaller radius_mm, another mark position, or loose_body_mm3 above the volume of a body that may go",
                "splits_part", {"bodies_before": st.bodies, "bodies_after": after, "bodies": _body_list(bodies),
                                "loose_body_limit_mm3": _f(limit, 6), **({"dropped_bodies": dropped} if dropped else {})})
        return mesh, man, facts
    if frozen_failed:
        raise _Rollback(frozen_failed + ": the step is undone. Options: frozen_keep_mm larger (a wider clip), a smaller "
                        "radius_mm, or the rerun_with_hints route", "frozen_restore_failed",
                        {"frozen_keep_mm": st.prm["frozen_keep_mm"]})
    raise _Rollback(last, "splice_failed")


def _depth_along_normal(old_mesh, new_mesh, pts: np.ndarray):
    """The median depth (mm) of the new surface below the points, along the normal of the old surface at each point
    (a ray into the part). None when no ray hits the new surface."""
    import trimesh
    pts = np.asarray(pts, float)
    _c, _d, fid = trimesh.proximity.closest_point(old_mesh, pts)
    n = np.asarray(old_mesh.face_normals)[fid]
    org = pts + 0.05 * n
    locs, ray_ids, _t = new_mesh.ray.intersects_location(org, -n, multiple_hits=False)
    if not len(ray_ids):
        return None
    d = np.linalg.norm(locs - org[ray_ids], axis=1) - 0.05
    return float(np.median(d))


def _surface_distance(mesh, pts: np.ndarray):
    return fb._surface_distance(mesh, pts)


def _stress_facts(d: Path, paths: list, radius: float, h: float) -> dict | None:
    """The stress of the part before the edit, near a mark (B-094: read access to result/stress.npz). None when the
    file is missing."""
    f = d / "result" / "stress.npz"
    if not f.is_file():
        return None
    try:
        from scipy.spatial import cKDTree
        z = np.load(f)
        centers, vm = np.asarray(z["centers"], float), np.asarray(z["vm_max"], float)
        tree = cKDTree(np.vstack([_densify(q, max(h * 0.5, 0.25)) for q in paths]))
        dist, _ = tree.query(centers, workers=1)
        sel = dist <= radius + h
        out = {"file": "result/stress.npz", "unit": "MPa (von Mises, the part before the edit; a voxel estimate of the "
                                                    "run grid, not a stress of the mesh: it can be off near a sharp edge)",
               "global_max_mpa": _f(vm.max(), 2), "global_p99_mpa": _f(np.percentile(vm, 99), 2),
               "voxels_near": int(sel.sum())}
        if sel.any():
            out["near_max_mpa"] = _f(vm[sel].max(), 2)
            out["near_mean_mpa"] = _f(vm[sel].mean(), 2)
            out["near_max_share_of_p99_pct"] = _f(100.0 * vm[sel].max() / max(np.percentile(vm, 99), 1e-12), 1)
            names = [str(x) for x in z["case_names"]] if "case_names" in z.files else []
            cases = {}
            for i, nm in enumerate(names):
                key = f"vm_case_{i}"
                if key in z.files:
                    cases[nm] = _f(np.asarray(z[key], float)[sel].max(), 2)
            if cases:
                out["near_max_by_case_mpa"] = cases
        return out
    except (OSError, KeyError, ValueError) as exc:
        return {"file": "result/stress.npz", "error": f"cannot be read: {exc}"}


# ---- one step: the edit of one site ------------------------------------------------------------------------
def _strength_factor(strength) -> float:
    ok = isinstance(strength, (int, float)) and not isinstance(strength, bool) and strength > 0
    return float(strength) if ok else 1.0


def _build_edit(st: _State, site: dict, step: dict):
    """The crop box, the field and the operator of a site. Returns (Ctx, OpResult). Raises NoEffect, OpFail, _Guard,
    _Rollback (odd rays)."""
    kind, R, prm = site["kind"], site["radius"], st.prm
    paths = site["paths"]
    pts = np.vstack(paths)

    def make(pad: float, margin: float, what: str) -> Ctx:
        crop = _plan_crop(st, pts, pad, margin)
        if crop is None:
            raise NoEffect(f"the mark lies outside the part: it is more than {max(margin, CROP_MARGIN_MM):g} mm from the "
                           "bounds of final.stl")
        _guard_crop(st, crop, what)
        stats: dict = {}
        Tc = signed_distance(st.mesh, crop, stats=stats)
        step["odd_rays"] = stats.get("odd_rays", 0)
        if stats.get("odd_rays_repaired"):
            step["odd_rays_repaired"] = stats["odd_rays_repaired"]
        if stats.get("odd_rays", 0) > st.opt["max_odd_rays"]:
            raise _Rollback(f"{stats['odd_rays']} rays of the inside test hit the surface an odd number of times, also "
                            f"after a second cast with the ray moved (of {stats.get('rays')} rays; the limit is "
                            f"max_odd_rays {st.opt['max_odd_rays']:g}): the mesh is not a clean solid at this mark, so "
                            "the field would have holes. Options: max_odd_rays higher, or the rerun_with_hints route",
                            "odd_rays", {"odd_rays": stats["odd_rays"], "odd_rays_repaired": stats.get("odd_rays_repaired", 0)})
        cx = Ctx(crop, Tc, paths, R, site["amount"], site["strength"], prm, st.labels)
        cx.neck_scale = site.get("neck_scale", 1.0)
        return cx

    if kind == "remove":
        if site["remove_mode"] == "shave":
            cx = make(R + 6.0 + site["amount_value"], 3.0 + site["amount_value"], "a shave")
            return cx, op_shave(cx)
        cx = make(R + prm["remove_dom_mm"] + BAND_MM, prm["remove_dom_mm"] + 2.0 * prm["remove_close_mm"]
                  + prm["remove_margin_mm"] + 1.0, "a cut")
        return cx, op_remove(cx)
    if kind == "add":
        cx = make(R + 2.0 * prm["close_mm"] + 4.0, 2.0 * prm["close_mm"] + 1.0, "a closing")
        return cx, op_add(cx)
    s = _strength_factor(site["strength"])
    sigma_iso, sigma_fair = prm["iso_sigma_mm"] * s, prm["fair_sigma_mm"] * s
    if site["smooth_mode"] == "tube":
        pad1 = max(14.0, R + max(6.0, 3.0 * sigma_iso) + 1.0, R + 2.5 * sigma_fair + 2.0)
        cx1 = make(pad1, prm["tube_ext_max_mm"] + 10.5, "the analysis of a tube")
        try:
            an = analyse_strut(cx1)
            step["mode_reason"] = "asked: smooth_mode tube"
            cx2 = make(prm["tube_ext_max_mm"] + 3.0 + 9.5 + 1.0, 10.5, "a tube")
            res = op_tube(cx2, an)
            return cx2, res
        except OpFail as exc:
            step["mode_reason"] = f"iso: smooth_mode tube was asked, and the tube could not be made ({exc})"
            mode = "iso"
        except _Guard as g:
            g.options.append("smooth_mode iso or fair: these need a smaller box")
            raise
        return cx1, op_iso(cx1)
    mode, why = pick_smooth_mode(paths, site["smooth_mode"])
    step["mode_reason"] = why
    if mode == "fair":
        cx = make(R + 2.5 * sigma_fair + 2.0, 2.5 * sigma_fair + 1.0, "a web fairing")
        return cx, op_fair(cx)
    cx = make(R + max(6.0, 3.0 * sigma_iso) + 1.0, 3.0 * sigma_iso + 1.0, "an isotropic fairing")
    return cx, op_iso(cx)


def _no_effect(cx: Ctx, res: OpResult) -> tuple[str, dict]:
    """The fact when an edit moves no node that can move the surface: the cause (a frozen region clips it, or the surface
    is already as the operator makes it) and the options."""
    facts = dict(res.info)
    nearest = None
    if cx.labels is not None:
        nearest = _f(float(cx.labels.frozen_distance(cx.samples).min()), 2)
        facts["nearest_frozen_to_mark_mm"] = nearest
    if cx.labels is not None and cx.clip_possible() and changed_nodes(cx.Tc, res.field(False), cx.p).any():
        d0, d1 = cx.prm["frozen_keep_mm"], cx.prm["frozen_full_mm"]
        facts["cause"] = "frozen_clip"
        return (f"a frozen region clips the edit to nothing: the nearest frozen region is {nearest:.2f} mm from the mark, "
                f"and the weight of an edit is 0 within frozen_keep_mm ({d0:g} mm) and 1 beyond frozen_full_mm ({d1:g} mm). "
                "Options: frozen_keep_mm smaller (the exact check of the frozen regions still protects them), a mark "
                "farther from the region, or the rerun_with_hints route"), facts
    facts["cause"] = "surface_unchanged"
    return (f"the edit ({res.mode}) does not move the surface: at this mark it is already as the operator makes it "
            "(a flat face or a smooth member stays). Options: a larger radius_mm or strength, another smooth_mode, "
            "remove_mode shave, or the rerun_with_hints route"), facts


def _edit_step(st: _State, site: dict) -> dict:
    """One site: build the edit, check it, put it in the part. The part changes only when every check passes."""
    t0 = time.monotonic()
    step: dict = {"label": site["label"], "kind": site["kind"], "status": "applied",
                  "paths_mm": [[[_f(v) for v in q] for q in path] for path in site["paths"]]}
    cx = res = Tc = Tn = None
    try:
        cx, res = _build_edit(st, site, step)
        Tc, p, crop = cx.Tc, cx.p, cx.crop
        step["mode"] = res.mode
        step["crop"] = {"lo_mm": [_f(v) for v in crop.lo], "hi_mm": [_f(v) for v in crop.hi],
                        "shape": list(crop.shape), "nodes": _cells(crop.shape), "pitch_mm": p}
        Tn = res.field(True)
        chg = changed_nodes(Tc, Tn, p)
        if not chg.any():
            raise NoEffect(*_no_effect(cx, res))
        added, removed = _edit_volume(Tc, Tn, p)
        if site["kind"] == "remove" and res.mode == "cut":
            clipped = float(res.info.get("clipped_by_frozen_mm3") or 0.0)
        elif cx.labels is not None and cx.clip_possible():
            a0, r0 = _edit_volume(Tc, res.field(False), p)
            clipped = max(a0 + r0 - added - removed, 0.0)
        else:
            clipped = 0.0
        planned = added + removed + clipped
        from scipy.spatial import cKDTree
        idx = np.argwhere(chg)
        sub = idx[:: max(len(idx) // 200_000, 1)]
        reach = float(cKDTree(cx.samples).query(crop.lo + sub * p, workers=1)[0].max())
        step["edit"] = {"added_mm3": _f(added, 2), "removed_mm3": _f(removed, 2),
                        "farthest_changed_node_from_mark_mm": _f(reach, 2),
                        "planned_mm3": _f(planned, 2), "clipped_by_frozen_mm3": _f(clipped, 2),
                        "clipped_share_pct": _f(100.0 * clipped / planned, 1) if planned > 0 else 0.0,
                        "note": "from the field (cell fractions); the part volume below is exact"}
        if cx.labels is not None:
            fd = cx.labels.frozen_distance(crop.lo + idx * p)
            step["frozen"] = {"nearest_frozen_to_edit_mm": _f(fd.min(), 2),
                              "nearest_frozen_to_mark_mm": _f(cx.labels.frozen_distance(cx.samples).min(), 2),
                              "frozen_keep_mm": cx.prm["frozen_keep_mm"], "frozen_full_mm": cx.prm["frozen_full_mm"]}
        vol0 = float(st.mesh.volume)
        mesh, man, facts = _splice_checked(st, Tn, Tc, crop, chg, step)
        zone = _grow(chg, 1.0, p)
        tb, ta = member_thickness(Tc < 0, zone, p), member_thickness(Tn < 0, zone, p)
        step["thickness_near_mm"] = {
            "before_min": _f(tb["min"], 2) if tb else None, "before_median": _f(tb["median"], 2) if tb else None,
            "after_min": _f(ta["min"], 2) if ta else None, "after_median": _f(ta["median"], 2) if ta else None,
            "medial_nodes_before": tb["n"] if tb else 0, "medial_nodes_after": ta["n"] if ta else 0,
            "note": "the diameter of the medial-axis balls (the largest ball in the solid at its centre) that reach "
                    "the changed nodes: the thickness of the member there. The median is the member; the minimum can "
                    "be set by the tip of a wedge at a convex edge (a ball of no size). A C3a method, not the "
                    "thickness of the pipeline"}
        info = dict(res.info)
        pc = info.pop("parent_check", None)
        if pc is not None:
            step["parent_check"] = pc
            if st.opt["parent_check"] == "enforce" and not pc.get("within_tolerance"):
                raise _Rollback("the parent check failed and parent_check is enforce: the cut goes into the load path "
                                f"(max drop {pc.get('max_drop_mm')} mm, depth {pc.get('max_depth_into_parent_mm')} mm, "
                                f"tolerance {pc.get('tolerance_mm')} mm)", "parent_check", {"parent_check": pc})
        step["operator"] = info
        if res.mode == "shave":
            dd = _depth_along_normal(st.mesh, mesh, np.vstack(site["paths"]))
            if dd is not None:
                info["depth_reached_mm"] = _f(dd, 2)
                info["depth_note"] = ("depth_reached_mm is the depth of the new surface below the mark, along the normal "
                                      "of the surface before. The field is exact only on the first layer of nodes at "
                                      f"the surface; deeper it is within half a node ({_f(0.5 * p, 2)} mm) of the "
                                      "distance, so the depth can differ from amount_mm by about that much")
        if res.mode == "tube":
            step["edit"]["note"] = (f"a tube rebuilds the member: about {_f(planned, 0):g} mm3 planned, reach "
                                    f"{_f(reach, 1):g} mm from the stroke (radius_mm is not used by a tube); the part volume "
                                    "below is exact")
        vol1 = float(mesh.volume)
        step["volume"] = {"before_mm3": _f(vol0, 2), "after_mm3": _f(vol1, 2), "change_mm3": _f(vol1 - vol0, 2),
                          "change_g": _f(st.mass_g(vol1 - vol0), 3)}
        step["part"] = {k: facts[k] for k in ("triangles", "closed", "watertight", "bodies", "open_edges",
                                              "non_manifold_edges", "zero_area_faces")}
        if "frozen_restored_mm3" in step:
            step.setdefault("frozen", {})["restored_mm3"] = step["frozen_restored_mm3"]
        st.mesh, st.man, st.bodies = mesh, man, facts["bodies"]
        st.boxes.append((np.array(step["splice_zone_bbox_mm"]["lo"]), np.array(step["splice_zone_bbox_mm"]["hi"])))
    except NoEffect as exc:
        step.update(status="no_effect", reason=exc.reason, facts=exc.facts)
    except OpFail as exc:
        step.update(status="failed", reason=str(exc))
    except _Guard as exc:
        step.update(status="guard", reason=exc.reason, facts=exc.facts, options=exc.options)
    except _Rollback as exc:
        step.update(status="rolled_back", reason=exc.reason, rollback=exc.code, facts=exc.facts)
    step["seconds"] = round(time.monotonic() - t0, 2)
    return step


# ---- marks to sites ----------------------------------------------------------------------------------------
def _planes_for(raw, doc: dict, run: dict, opt: dict) -> tuple[list[dict], list[str]]:
    """The mirror planes of a mark: the call option, the mark, the marks file (one plane each), then the planes of
    run.json (up to 3), then the page default (y = 0). Returns (planes, facts)."""
    if opt["mirror_plane"]:
        return [opt["mirror_plane"]], []
    for spec, src in ((raw.get("mirror_plane") if isinstance(raw, dict) else None, "the mark (mirror_plane)"),
                      (doc.get("mirror_plane"), "the marks file (mirror_plane)")):
        pl = _plane_from(spec, src)
        if pl:
            return [pl], []
    planes, facts = _planes_of_run(run)
    if planes:
        return planes, facts
    return [{"axis": "y", "axis_index": 1, "at_mm": 0.0,
             "source": "default: y = 0, the plane of the page (its label is 'mirror y=0')"}], facts


def _mirror_images(pts: np.ndarray, planes: list[dict]) -> list[np.ndarray]:
    """The path and its images in the group of the reflections about the planes (2 ** len(planes) paths, the path
    first). Images that are the same path (a mark on a plane) are given once."""
    out = [np.asarray(pts, float)]
    for pl in planes:
        out += [_mirror_points(q, pl) for q in out]
    keep: list[np.ndarray] = []
    for q in out:
        if not any(q.shape == k.shape and float(np.abs(q - k).max()) < 1e-3 for k in keep):
            keep.append(q)
    return keep


def _image_groups(images: list[np.ndarray], reach: float) -> tuple[list[list[int]], dict]:
    """The images that go into one site: two images whose paths are less than twice the reach apart are one site (one
    edit, so an overlap is not edited twice); the closure over all pairs (up to 8 images). Returns (groups, gaps)."""
    n = len(images)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            gaps[(i, j)] = _path_distance(images[i], images[j])
            if gaps[(i, j)] < 2.0 * reach:
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), gaps


def _valid_pos(x):
    v = fb._num(x)
    return v if v is not None and v > 0 else None


def _mode_of(mid, kind: str, key: str, over: dict, mk: dict, modes: tuple, opt: dict, problems: list):
    """The smooth or remove mode of one mark. The order: the call option `per_mark[id]`, then the key of the mark (the file),
    then the call option `smooth_mode` / `remove_mode`, then the default. `OPTION_DEFAULTS` always holds a value, so the call
    option cannot be told from the default, and a mark that states its mode keeps it. A value that is not in `modes`, or a
    key of the other type, is a fact in `problems` and is not used."""
    if over.get(key) in modes:
        return over[key]
    own = mk.get(key)
    if own is None:
        return opt[key]
    for_kind = "smooth" if key == "smooth_mode" else "remove"
    if kind != for_kind:
        problems.append(f"mark {mid}: {key} {own!r} is for a {for_kind} mark, and this mark is {kind}; the key is not used")
        return opt[key]
    if own in modes:
        return own
    problems.append(f"mark {mid}: {key} {own!r} is not one of {', '.join(modes)}; the call option {opt[key]!r} is used")
    return opt[key]


def _prepare(doc: dict, opt: dict, run: dict, st: _State) -> tuple[list[dict], list[dict]]:
    """The entries of the report (one per mark) and the sites (one step each), in the order of the pipeline."""
    entries: list[dict] = []
    sites: list[dict] = []
    seen: set = set()
    prm = st.prm
    for index, raw in enumerate(doc["marks"]):
        mk, errs = fb._norm_mark(raw, index)
        if mk is None:
            entries.append({"id": raw.get("id") if isinstance(raw, dict) else None, "index": index, "type": None,
                            "status": "unreadable", "reason": "; ".join(errs), "steps": []})
            continue
        mid = mk["id"]
        over = opt["per_mark"].get(mid, {})
        if mid in seen:
            st.problems.append(f"mark id {mid} appears more than once; per_mark and skip act on all of them")
        seen.add(mid)
        kind = mk["type"]
        r0 = _valid_pos(over.get("radius_mm")) or mk["radius_mm"]
        R = r0 * opt["radius_scale"]
        amount = over["amount_mm"] if "amount_mm" in over else mk["amount_mm"]
        strength = over["strength"] if "strength" in over else mk["strength"]
        mirror = bool(over["mirror"]) if "mirror" in over else bool(mk["mirror"])
        smode = _mode_of(mid, kind, "smooth_mode", over, mk, SMOOTH_MODES, opt, st.problems)
        rmode = _mode_of(mid, kind, "remove_mode", over, mk, REMOVE_MODES, opt, st.problems)
        defaults = []
        if mk["radius_default_used"] and "radius_mm" not in over:
            defaults.append({"name": "radius_mm", "value": PAGE_RADIUS_DEFAULT,
                             "anchor": "the page default; radius_mm is missing in the file"})
        has_amount = _valid_pos(amount) is not None
        if kind == "add" and not has_amount:
            defaults.append({"name": "neck_offset_mm", "value": prm["neck_offset_mm"],
                             "anchor": DEFAULTS["neck_offset_mm"][1]})
        if kind == "remove" and rmode == "shave" and not has_amount:
            defaults.append({"name": "shave_depth_mm", "value": 1.0, "anchor": "no pipeline value: the page default amount"})
        if kind == "smooth" and _valid_pos(strength) is None:
            defaults.append({"name": "strength", "value": 1.0, "anchor": DEFAULTS["strength_default"][1]})
        entry = {"id": mid, "index": index, "type": kind, "status": "pending", "reason": None,
                 "values": {"points": len(mk["points"]), "radius_mm": mk["radius_mm"], "radius_used_mm": _f(R),
                            "amount_mm": amount, "strength": strength, "mirror": mirror,
                            "smooth_mode": smode if kind == "smooth" else None,
                            "remove_mode": rmode if kind == "remove" else None},
                 "defaults_used": defaults, "steps": [], "mirror": None,
                 "values_not_used": VALUES_NOT_USED + (["amount_mm (the cut mode takes no depth)"]
                                                        if kind == "remove" and rmode == "cut" else [])}
        pts = mk["points"]
        d_surf = _surface_distance(st.mesh, pts)
        entry["surface_distance_mm"] = d_surf
        if d_surf and max(d_surf) > 2 * st.h:
            entry["surface_note"] = (f"a point is {max(d_surf):.2f} mm from the surface of final.stl, more than two voxels: "
                                     "the mark may belong to another mesh")
        entries.append(entry)
        if mid in opt["skip"] or over.get("skip"):
            entry.update(status="skipped_by_option", reason="the options say skip: this mark is not applied")
            continue
        base = {"kind": kind, "mark_id": mid, "radius": R, "amount": amount, "strength": strength,
                "amount_value": _valid_pos(amount) or 1.0,
                "smooth_mode": smode, "remove_mode": rmode, "order": (KIND_ORDER.index(kind), index), "entry": entry}
        if not mirror:
            sites.append({**base, "paths": [pts], "label": "mark"})
            continue
        planes, plane_facts = _planes_for(raw, doc, run, opt)
        images = _mirror_images(pts, planes)
        info = {"plane": {k: planes[0][k] for k in ("axis", "at_mm", "source")},
                "planes": [{k: pl[k] for k in ("axis", "at_mm", "source")} for pl in planes], "images": len(images)}
        if plane_facts:
            info["plane_facts"] = plane_facts
        dm = [_surface_distance(st.mesh, q) for q in images[1:]]
        far = [max(x) for x in dm if x]
        if far:
            info["mirror_surface_distance_mm"] = dm[0] if len(dm) == 1 else dm
        if far and max(far) > 2 * st.h:
            info["surface_note"] = (f"a mirrored point is {max(far):.2f} mm from the surface of final.stl, more than two "
                                    "voxels: the part may not be symmetric about the plane" + ("s" if len(planes) > 1 else ""))
        reach = R + (prm["remove_dom_mm"] if kind == "remove" else 0.0)
        if len(images) == 1:
            info.update(combined=True, sites=1, note="the mark lies on the mirror plane" + ("s" if len(planes) > 1 else "")
                        + ": its mirror image is the mark itself")
            sites.append({**base, "paths": [pts], "label": "mark (on the mirror plane)"})
        else:
            groups, gaps = _image_groups(images, reach)
            info.update(gap_mm=_f(min(gaps.values())), sites=len(groups), images_per_site=[len(g) for g in groups],
                        combined=len(groups) == 1)
            if any(len(g) > 1 for g in groups):
                info["note"] = (f"images less than twice the reach ({2 * reach:.2f} mm) apart are one site, with one edit "
                                "(the weight is the larger of the two), so an overlap is not edited twice")
            for g in groups:
                one = len(g) == len(images)
                label = ("mark + mirror image (one site)" if one and len(g) == 2 else
                         f"mark + {len(g) - 1} mirror images (one site)" if one else
                         "mark" if g == [0] else "mirror image" if len(g) == 1 else
                         f"{len(g)} images (one site)")
                sites.append({**base, "paths": [images[i] for i in g], "label": label,
                              "neck_scale": len(g) / len(images)})
        entry["mirror"] = info
    sites.sort(key=lambda s: s["order"])
    return entries, sites


def _aggregate(entry: dict) -> None:
    """The status and the facts of a mark from its steps."""
    steps = entry["steps"]
    if not steps:
        return
    ok = [s for s in steps if s["status"] == "applied"]
    if len(ok) == len(steps):
        entry["status"] = "applied"
    elif ok:
        entry["status"] = "partly_applied"
    else:
        codes = {"no_effect": "no_effect", "guard": "guard", "rolled_back": "rolled_back", "failed": "no_effect"}
        entry["status"] = codes.get(steps[0]["status"], steps[0]["status"])
    bad = [s for s in steps if s["status"] != "applied"]
    if bad:
        entry["reason"] = "; ".join(f"{s['label']}: {s.get('reason')}" for s in bad)
    if not ok:
        return
    ed = {k: _f(sum(s["edit"][k] for s in ok), 2) for k in ("added_mm3", "removed_mm3", "planned_mm3", "clipped_by_frozen_mm3")}
    vol = sum(s["volume"]["change_mm3"] for s in ok)
    ed["part_volume_change_mm3"] = _f(vol, 2)
    ed["farthest_changed_node_from_mark_mm"] = max(s["edit"]["farthest_changed_node_from_mark_mm"] for s in ok)
    ed["edit_reach_mm"] = ed["farthest_changed_node_from_mark_mm"]
    rest = sum(s.get("frozen_restored_mm3") or 0.0 for s in ok)
    if rest:
        ed["restored_in_frozen_mm3"] = _f(rest, 4)
        entry["frozen_note"] = (f"the edit zone met a frozen region: {_f(rest, 4)} mm3 of the edit inside it was undone "
                                "(the exact check of the regions of the YAML)")
    notes = [s["edit"]["note"] for s in ok if s["edit"].get("note", "").startswith("a tube")]
    if notes:
        entry["edit_note"] = notes[0]
    gs = [s["volume"]["change_g"] for s in ok if s["volume"]["change_g"] is not None]
    ed["part_mass_change_g"] = _f(sum(gs), 3) if gs else None
    planned = sum(s["edit"]["planned_mm3"] for s in ok)
    ed["clipped_share_pct"] = _f(100.0 * ed["clipped_by_frozen_mm3"] / planned, 1) if planned > 0 else 0.0
    entry["edit"] = ed
    tb = [s["thickness_near_mm"] for s in ok]

    def _agg(key, fn):
        v = [t[key] for t in tb if t.get(key) is not None]
        return _f(fn(v), 2) if v else None
    entry["thickness_near_mm"] = {"before_min": _agg("before_min", min), "before_median": _agg("before_median", np.median),
                                  "after_min": _agg("after_min", min), "after_median": _agg("after_median", np.median)}
    entry["modes"] = sorted({s["mode"] for s in ok})
    pcs = [s["parent_check"] for s in ok if "parent_check" in s]
    if pcs:
        entry["parent_check"] = {"within_tolerance": all(pc["within_tolerance"] for pc in pcs),
                                 "max_drop_mm": max(pc["max_drop_mm"] for pc in pcs),
                                 "max_depth_into_parent_mm": max(pc["max_depth_into_parent_mm"] for pc in pcs),
                                 "skeleton_max_local_drop_mm": max(pc["skeleton"]["max_local_drop_mm"] for pc in pcs),
                                 "tolerance_mm": pcs[0]["tolerance_mm"]}
    fz = [s["frozen"]["nearest_frozen_to_edit_mm"] for s in ok if "frozen" in s]
    if fz:
        entry["nearest_frozen_to_edit_mm"] = min(fz)
    rec = sum(s.get("odd_rays_repaired", 0) for s in ok)
    if rec:
        entry["odd_rays_repaired"] = rec
    entry["bodies_after"] = ok[-1]["part"]["bodies"]
    entry["closed_after"] = all(s["part"]["closed"] for s in ok)
    dropped = [s["dropped_bodies"] for s in ok if s.get("dropped_bodies")]
    if dropped:
        entry["dropped_bodies"] = {"count": sum(x["count"] for x in dropped),
                                   "total_volume_mm3": _f(sum(x["total_volume_mm3"] for x in dropped), 4),
                                   "largest_volume_mm3": max(x["largest_volume_mm3"] for x in dropped),
                                   "limit_mm3": dropped[0]["limit_mm3"]}


# ---- checks of the whole result ----------------------------------------------------------------------------
def _triangle_keys(mesh) -> np.ndarray:
    """One 64-bit key per triangle, from the float32 bits of its three vertices (the order of the vertices does not
    matter). It tells which triangles of the part are the same after the edit."""
    v = np.ascontiguousarray(np.asarray(mesh.vertices, np.float32))
    u = v.view(np.uint32).astype(np.uint64)
    mult = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9], np.uint64)
    with np.errstate(over="ignore"):
        hv = (u * mult).sum(axis=1)
        hf = hv[np.asarray(mesh.faces)]
        return (hf.sum(axis=1) * np.uint64(0xFF51AFD7ED558CCD)) ^ ((hf * hf).sum(axis=1) * np.uint64(0xC4CEB9FE1A85EC53)) \
            ^ (hf.prod(axis=1) * np.uint64(0x2545F4914F6CDD1D))


def _replaced_triangles(mesh0, mesh1) -> dict:
    """How many triangles of the part before are not in the part after, bit for bit."""
    k0, k1 = _triangle_keys(mesh0), _triangle_keys(mesh1)
    gone = ~np.isin(k0, k1)
    return {"triangles_before": int(len(k0)), "triangles_kept_bit_equal": int((~gone).sum()),
            "triangles_replaced": int(gone.sum()), "triangles_after": int(len(k1)),
            "replaced_share_pct": _f(100.0 * gone.sum() / max(len(k0), 1), 2),
            "note": "a triangle that the zone cuts counts as replaced; a large flat triangle is split into many small ones"}


def _load_frozen(d: Path, run: dict, yaml_arg):
    """The frozen regions of the YAML of the run (fixed, load, keep-out, and the forced-solid regions when the engine has
    them) as one solid, and the facts. Returns (solid or None, facts). The YAML must be on disk."""
    from .config import load_config
    from .voxelize import load_mesh
    ypath = Path(str(yaml_arg)).expanduser().resolve() if yaml_arg else fb._run_yaml_path(run, d)
    if ypath is None or not ypath.is_file():
        return None, {"done": False, "reason": f"the YAML of the run is not on disk ({ypath}); the check by labels.bin is "
                      "done (nearest_frozen_to_edit_mm of each step). Option: yaml_path"}
    try:
        cfg = load_config(ypath)
        regs = [("fixed", p) for p in cfg.regions.fixed] + [("keep_out", p) for p in cfg.regions.keep_out] \
            + [("load", lr.path) for lr in cfg.regions.load]
        solid = getattr(cfg.regions, "solid", None)           # regions.solid (#30), when the engine has it
        if solid:
            regs += [("solid", getattr(s, "path", s)) for s in (solid if isinstance(solid, (list, tuple)) else [solid])]
        mans = []
        for role, path in regs:
            m = _manifold(load_mesh(Path(path)))
            if m is not None:
                mans.append(m)
    except UserError as exc:
        return None, {"done": False, "reason": f"the YAML of the run does not load: {exc}"}
    if not mans:
        return None, {"done": False, "reason": "no frozen region is a valid solid for a boolean"}
    F = mans[0]
    for m in mans[1:]:
        F = F + m
    facts = {"done": True, "regions": len(mans), "yaml": str(ypath)}
    ys = fb._yaml_state(run, fb._epoch(run.get("started_at")), d)
    if ys.get("newer_than_run") or ys.get("newer_region_files"):
        facts["yaml_note"] = ("the YAML or a region file is newer than the run: the regions of this check may differ from "
                              "the regions that the run used")
    return F, facts


def _frozen_exact(st: _State, man0, man1):
    """The volume of the edit inside the frozen regions (exact, on the whole part). Returns (facts, restored, failed):
    `restored` is (mesh, solid) when a change was undone; `failed` is true when the change cannot be undone.

    The steps undo a change in a frozen region themselves (`_splice_checked`), so this is the check of the result as a
    whole; the undo here is the fallback."""
    F = st.frozen_solid
    if F is None:
        return dict(st.frozen_facts), None, False
    add = float(((man1 - man0) ^ F).volume())
    rem = float(((man0 - man1) ^ F).volume())
    out = {**st.frozen_facts, "added_in_frozen_mm3": _f(add, 4), "removed_from_frozen_mm3": _f(rem, 4),
           "unchanged": bool(add + rem < FROZEN_TOL_MM3), "tolerance_mm3": FROZEN_TOL_MM3,
           "restored_by_steps_mm3": _f(sum(s.get("frozen_restored_mm3") or 0.0 for e in st.entries for s in e["steps"]), 4)}
    if add + rem < FROZEN_TOL_MM3:
        return out, None, False
    # getattr: the test stub of this check has no grid; the pitch only scales the loop match and the reach of the seam
    mesh, fixed_man, _seam, _why = _swap_inside(man1, man0, F, getattr(st, "pitch", None), exact_inside=True)
    if mesh is not None:
        facts, _ = _reload_facts(mesh)
        if not facts["closed"]:
            mesh, fixed_man, facts, _passes = _reload_with_pinch_repair(mesh, fixed_man, facts)
        # measure again: the note says "undone" only when the regions of the part that we return are exact
        add2 = float(((fixed_man - man0) ^ F).volume())
        rem2 = float(((man0 - fixed_man) ^ F).volume())
        out["added_after_undo_mm3"], out["removed_after_undo_mm3"] = _f(add2, 4), _f(rem2, 4)
        if facts["closed"] and facts["bodies"] <= max(st.bodies, 1) and add2 + rem2 < FROZEN_TOL_MM3:
            out["restored"] = True
            out["note"] = "the part of the edit inside the frozen regions was undone"
            return out, (mesh, fixed_man), False
        if facts["closed"] and facts["bodies"] <= max(st.bodies, 1):
            out["restored"] = False
            out["note"] = (f"the undo of the edit in the frozen regions left a change of {add2 + rem2:.4f} mm3 "
                           f"({add2:.4f} added, {rem2:.4f} removed; the tolerance is {FROZEN_TOL_MM3} mm3): the steps are rolled "
                           "back (nothing is applied). Options: frozen_keep_mm larger, smaller radius_mm, or the rerun_with_hints "
                           "route")
            return out, None, True
    out["restored"] = False
    out["note"] = ("the edit changes a frozen region and the undo does not give a valid closed part: the steps are rolled "
                   "back (nothing is applied). Options: frozen_keep_mm larger, smaller radius_mm, or the rerun_with_hints "
                   "route")
    return out, None, True


# ---- the output folder -------------------------------------------------------------------------------------
def _earlier_apply_folder(out: Path) -> bool:
    try:
        data = json.loads((out / "apply_marks.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("format") == APPLY_FORMAT


def _plan_out(out_dir, d: Path, input_dirs: set, overwrite: bool) -> Path:
    """The folder to write, or a refusal. Nothing is written here. Every refusal names the fact and the options."""
    new_name = "Name a new folder in out_dir, or leave out_dir out: the default is a new folder beside the run folder."
    if out_dir is None:
        return fb._unique_dir(d.parent / f"{d.name}-marks")
    given = Path(os.path.abspath(Path(out_dir).expanduser()))
    out = given.resolve()
    if os.path.islink(given):
        raise fb._Refused(f"out_dir {given} is a symbolic link. The tool does not write through a symbolic link.",
                          "symbolic_link", given, [new_name])
    if out.exists() and not out.is_dir():
        raise fb._Refused(f"out_dir {out} is a file, not a folder.", "not_a_folder", out, [new_name])
    for p in (out, *out.parents):
        if (p / "run.json").is_file() or (p / "status.json").is_file():
            where = "is a run folder" if p == out else f"is inside the run folder {p}"
            raise fb._Refused(f"out_dir {out} {where} (it has run.json or status.json). The files of a run are not "
                              "written. " + new_name, "run_folder", p, [new_name])
    if out in input_dirs:
        raise fb._Refused(f"out_dir {out} holds an input file of the run (the YAML or a region STL). The tool does not "
                          "write into the folder of an input. " + new_name, "input_folder", out, [new_name])
    holds = out.is_dir() and any(out.iterdir())
    if holds and not overwrite:
        raise fb._Refused(f"out_dir {out} holds files. Options: name a new folder; or set overwrite=true if it is an "
                          "earlier apply_marks folder (it has an apply_marks.json).", "not_empty", out,
                          [new_name, "overwrite=true, only for an earlier apply_marks folder"])
    if holds and not _earlier_apply_folder(out):
        raise fb._Refused(f"out_dir {out} holds files and is not an earlier apply_marks folder (no apply_marks.json of "
                          f"format {APPLY_FORMAT}). overwrite=true is accepted for an earlier apply_marks folder only. "
                          + new_name, "not_an_apply_folder", out, [new_name])
    return out


_ABS_PATH = None


def _inside(path, root) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def _scrub_absolute(text: str, out: Path, root: Path) -> str:
    """Replace every absolute path inside a string: a path relative to the folder of apply_marks.json when the path is
    inside `root` (the common root of the run folder and the out folder), else the file name."""
    global _ABS_PATH
    import re
    if _ABS_PATH is None:
        _ABS_PATH = re.compile(r"(?<![\w.:/])(?:/[^\s\"'`,;:()\[\]{}<>|*?]+)+|(?<![\w])[A-Za-z]:\\[^\s\"'`,;()\[\]{}<>|*?]+")

    def one(m):
        raw, trail = m.group(0), ""
        while raw and raw[-1] == ".":
            raw, trail = raw[:-1], "." + trail
        if raw.count("/") < 2 and "\\" not in raw:
            return m.group(0)                      # one segment (/x) is not a path of the host
        if _inside(raw, root):
            return fb._relpath(raw, out) + trail
        return Path(raw.replace("\\", "/")).name + trail
    return _ABS_PATH.sub(one, text)


def _walk_strings(x, fn):
    if isinstance(x, str):
        return fn(x)
    if isinstance(x, list):
        return [_walk_strings(v, fn) for v in x]
    if isinstance(x, dict):
        return {k: _walk_strings(v, fn) for k, v in x.items()}
    return x


def _portable(report: dict, out: Path, d: Path, marks_file) -> dict:
    """The report as apply_marks.json holds it: no absolute path. A path is relative to that folder, or a file name.

    The run folder and the out folder are replaced first. Then every absolute path that is left in any string (inside a
    sentence too) is replaced: relative when it is inside the common root of the run folder and the out folder, else the
    file name, so a path never climbs to the root of the host (pitfalls/absolute-paths-in-artifacts)."""
    rep = json.loads(json.dumps(report))
    try:
        root = Path(os.path.commonpath([str(out), str(d)]))
    except ValueError:
        root = out
    reps = [(str(out), "."), (str(d), fb._relpath(d, out))]
    rep = fb._scrub_strings(rep, sorted(reps, key=lambda x: -len(x[0])))
    rep = _walk_strings(rep, lambda t: _scrub_absolute(t, out, root))
    rep["run_dir"] = fb._relpath(d, out)
    rep["out_dir"] = "."
    rep["marks_file"] = Path(marks_file).name if marks_file else None
    rep["paths_are_relative_to"] = "the folder of this file (apply_marks.json)"
    return rep


# ---- the call ----------------------------------------------------------------------------------------------
def apply_marks(run_dir, marks, *, out_dir=None, options=None, overwrite: bool = False, allow_stale: bool = False,
                yaml_path=None) -> dict:
    """Apply the marks of the Refine page to the finished part, in a NEW folder. Facts and options; no verdict.

    run_dir: a finished run (needs run.json, labels.bin and result/final.stl; result/stress.npz adds stress facts).
    The arguments after `marks` are keyword-only.
    marks: the path of a marks JSON (version 1, units mm, the frame of final.stl), its JSON text, or the dict.
    out_dir: the new folder. Default: `<run folder>-marks` beside the run folder (a number is added when it exists).
    Refused (the report says why and what to do): a run folder or a folder inside one, a folder that holds an input
    file of the run, a symbolic link, and a folder that holds files and is not an earlier apply_marks folder.
    options: see OPTION_TEXT (radius_scale, pitch_mm, max_crop_voxels, max_total_voxels, frozen_keep_mm,
    frozen_full_mm, close_mm, neck_min_mm3, parent_tol_mm, parent_check, smooth_mode, remove_mode, mirror_plane,
    skip, per_mark, check_frozen_exact, loose_body_mm3, max_odd_rays).
    allow_stale: apply the marks when `source_sha256` of the marks file is not the hash of final.stl.
    yaml_path: the YAML of the run, when run.json names a file that is not there (used for the exact frozen check).
    Returns the full report (apply_marks.json holds it), or `{"ok": false, "error": ...}` when it cannot start.
    Writes `edited.stl` and `apply_marks.json` into the new folder. The run folder and every input stay as they are.
    """
    try:
        return _apply(Path(run_dir).expanduser().resolve(), marks, out_dir, options, overwrite, allow_stale, yaml_path)
    except fb._Refused as exc:
        return {"ok": False, "error": str(exc),
                "refused": {"reason": exc.reason, "path": exc.path, "options": exc.options}}
    except UserError as exc:
        return {"ok": False, "error": str(exc)}
    except MemoryError:
        return {"ok": False, "error": "out of memory. Options: pitch_mm larger (a coarser grid), fewer marks in this call, "
                                      "or a smaller max_crop_voxels."}
    except Exception as exc:  # a fact about this tool, not a crash of the caller
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


def _pipeline_defaults(entries: list, prm: dict, overridden: list) -> dict:
    """The values of the operators that were used, each with its pipeline anchor, for the report."""
    op_of = {"cut": "remove", "shave": "shave", "closing": "add", "closing + neck offset": "add", "iso": "iso",
             "fair": "fair", "tube": "tube"}
    used = set()
    for e in entries:
        for stp in e.get("steps", []):
            if stp.get("mode") in op_of:
                used.add(op_of[stp["mode"]])
    return {k: {n: {"value": prm[n], "anchor": DEFAULTS[n][1], "overridden_by_option": n in overridden}
                for n in KIND_PARAMS[k] if n in DEFAULTS} for k in sorted(used)}


def _density_of(run: dict, metrics: dict | None):
    v = fb._num(((run or {}).get("derived") or {}).get("density_g_cm3"))
    if v:
        return v
    m = metrics or {}
    if fb._num(m.get("mass_final_g")) and fb._num(m.get("volume_final_mm3")):
        return m["mass_final_g"] / m["volume_final_mm3"] * 1000.0
    return None


def _apply(d: Path, marks, out_dir, options, overwrite: bool, allow_stale: bool, yaml_arg) -> dict:
    t0 = time.monotonic()
    import trimesh

    if not d.is_dir():
        raise UserError(f"run folder not found: {d}")
    problems: list[str] = []
    notes: list[str] = []
    run = fb._load_json(d / "run.json", problems, "run.json")
    if run is None:
        raise UserError(f"{d / 'run.json'} is missing or unreadable: the grid of the run is not known")
    grid = fb._grid_of(run)
    if grid is None:
        raise UserError("run.json has no valid shape, h and origin")
    shape, h, origin = grid
    final = d / "result" / "final.stl"
    if not final.is_file():
        from .runjob import stopped_hint
        raise UserError(f"{final} is missing: there is no finished part to edit. Run `topopt run` to the end first "
                        "(the post-process writes final.stl)." + stopped_hint(d))
    opt, opt_problems = _norm_options(options)
    problems += opt_problems
    doc = fb.read_marks(marks)
    marks_file = None if isinstance(marks, dict) or str(marks).lstrip().startswith("{") \
        else str(Path(marks).expanduser().resolve())
    marks_sha = fb._sha256(Path(marks_file)) if marks_file else hashlib.sha256(
        json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()
    final_sha = fb._sha256(final)
    sha_marks = doc.get("source_sha256")
    marks_match = None if not sha_marks else (sha_marks == final_sha)
    if marks_match is False and not allow_stale:
        return {"ok": False, "marks_match_final_stl": False,
                "error": "source_sha256 of the marks differs from the hash of final.stl of this run: the marks were made "
                         "for another mesh, so their positions may not fit this part. Nothing was changed.",
                "refused": {"reason": "stale_marks", "path": None,
                            "options": ["allow_stale=true: apply the marks anyway (check the surface distance of each mark "
                                        "in the report)", "export the marks again from result/refine.html of this run"]},
                "facts": {"marks_sha256_source": sha_marks, "final_stl_sha256": final_sha, "marks_count": len(doc["marks"]),
                          "source_stl_of_marks": doc.get("source_stl")}}
    if marks_match is False:
        problems.append("source_sha256 of the marks differs from final.stl of this run (allow_stale is set): the marks were "
                        "made for another mesh, so the positions may not match this part. The report gives the distance of "
                        "each mark to the surface.")
    if doc.get("units", "mm") != "mm":
        problems.append(f"units is {doc.get('units')!r}, not mm: the numbers are used as mm")

    # -- the new folder: checked before the work
    ypath = Path(str(yaml_arg)).expanduser().resolve() if yaml_arg else fb._run_yaml_path(run, d)
    input_dirs = set()
    if ypath is not None and ypath.is_file():
        yd, _ = fb._yaml_doc(ypath)
        input_dirs = {ypath.parent.resolve()} | {p.parent for p in fb._region_paths((yd or {}).get("regions"), ypath.parent)}
    out = _plan_out(out_dir, d, input_dirs, overwrite)
    fb._check_targets(out, {"edited.stl", "apply_marks.json"})

    # -- the part, the labels, the state
    labels, lab_problem = fb._read_labels(d, run)
    if labels is None:
        raise UserError(f"the frozen regions are not known: {lab_problem}. labels.bin comes with the run (`topopt run` "
                        "writes it at the start). Option: restore it from a copy of the run folder.")
    mesh0 = trimesh.load(str(final), file_type="stl", force="mesh")
    man0 = _manifold(mesh0)
    if man0 is None:
        raise UserError("result/final.stl is not a valid closed solid for a boolean: the edit cannot start. "
                        "`topopt check` and post-process tell more.")
    metrics = fb._load_json(d / "result" / "metrics.json", problems, "result/metrics.json", expected=False)
    pitch_default = opt["pitch_mm"] is None
    pitch = float(np.clip(h / PITCH_DIVISOR, *PITCH_DEFAULT_RANGE)) if pitch_default else float(opt["pitch_mm"])
    prm, overridden = _parameters(opt)
    facts0 = mesh_facts(mesh0)
    st = _State(d=d, run=run, h=h, pitch=pitch, labels=Labels(labels, origin, h), mesh=mesh0, mesh0=mesh0, man=man0,
                man0=man0, bounds=np.asarray(mesh0.bounds), density=_density_of(run, metrics), opt=opt, prm=prm,
                problems=problems, bodies=facts0["bodies"])
    if not facts0["closed"]:
        problems.append("final.stl is not closed (watertight, one winding, positive volume) before the edit: "
                        f"open edges {facts0['open_edges']}, non-manifold edges {facts0['non_manifold_edges']}")
    st.frozen_solid, st.frozen_facts = _load_frozen(d, run, yaml_arg) if opt["check_frozen_exact"] else (None, {})
    entries, sites = _prepare(doc, opt, run, st)
    st.entries = entries
    for site in sites:
        site["entry"]["steps"].append(_edit_step(st, site))
    for e in entries:
        if e.get("status") == "pending" and not e["steps"]:
            e.update(status="no_effect", reason="no step was made for this mark")

    # -- the result as a whole: the exact check of the frozen regions
    changed = bool(st.boxes)
    frozen_exact = None
    if changed and opt["check_frozen_exact"]:
        frozen_exact, restored, failed = _frozen_exact(st, man0, st.man)
        if failed:
            for e in entries:
                for stp in e["steps"]:
                    if stp["status"] == "applied":
                        stp.update(status="rolled_back", rollback="frozen_restore_failed",
                                   reason="the edit of the whole call changes a frozen region and the undo is not a valid "
                                          "closed part, so every step of the call is rolled back", facts={})
            st.mesh, st.man, st.bodies, st.boxes = mesh0, man0, facts0["bodies"], []
            changed = False
            problems.append("every step is rolled back: " + frozen_exact["note"])
        elif restored is not None:
            st.mesh, st.man = restored
            st.bodies = mesh_facts(st.mesh)["bodies"]
    for e in entries:
        _aggregate(e)
    # stress near each mark (B-094), from the part before the edit
    no_stress = False
    for e in entries:
        if e.get("steps") and e["type"]:
            paths = [np.asarray(q, float) for s in e["steps"] for q in s["paths_mm"]]
            e["stress_near_mark"] = _stress_facts(d, paths, e["values"]["radius_used_mm"], h)
            no_stress = no_stress or e["stress_near_mark"] is None
    if no_stress:
        notes.append("result/stress.npz is missing: the stress facts near the marks are not given (`topopt refine` writes it)")
    applied = [e for e in entries if e["status"] in ("applied", "partly_applied")]

    vol0, vol1 = float(mesh0.volume), float(st.mesh.volume)
    facts1 = mesh_facts(st.mesh) if changed else facts0
    totals: dict = {
        "volume_before_mm3": _f(vol0, 2), "volume_after_mm3": _f(vol1, 2), "change_mm3": _f(vol1 - vol0, 2),
        "change_pct": _f(100.0 * (vol1 - vol0) / vol0, 3) if vol0 else None,
        "mass_before_g": _f(st.mass_g(vol0), 3), "mass_after_g": _f(st.mass_g(vol1), 3),
        "change_g": _f(st.mass_g(vol1 - vol0), 3), "density_g_cm3": _f(st.density, 4),
        "triangles_before": facts0["triangles"], "triangles_after": facts1["triangles"],
        "bodies_before": facts0["bodies"], "bodies_after": facts1["bodies"], "closed_before": facts0["closed"],
        "closed_after": facts1["closed"],
        "marks": {k: sum(1 for e in entries if e["status"] == k) for k in
                  sorted({e["status"] for e in entries})},
        "steps": sum(len(e["steps"]) for e in entries), "steps_applied": len(st.boxes),
        "crop_nodes_total": st.cells_used, "frozen_exact": frozen_exact,
        "frozen_by_labels": {"nearest_frozen_to_any_edit_mm": min(
            [e["nearest_frozen_to_edit_mm"] for e in entries if "nearest_frozen_to_edit_mm" in e], default=None),
            "clipped_by_frozen_mm3": _f(sum(e["edit"]["clipped_by_frozen_mm3"] for e in applied), 2) if applied else 0.0}}
    restored_mm3 = sum(stp.get("frozen_restored_mm3") or 0.0 for e in entries for stp in e["steps"]
                       if stp["status"] == "applied")
    if frozen_exact and frozen_exact.get("restored"):
        restored_mm3 += (frozen_exact.get("added_in_frozen_mm3") or 0.0) + (frozen_exact.get("removed_from_frozen_mm3") or 0.0)
    totals["restored_mm3"] = _f(restored_mm3, 4)
    if restored_mm3 > 0:
        notes.append(f"{_f(restored_mm3, 4)} mm3 of the edit lay inside the frozen regions and was undone (totals.restored_mm3; "
                     "the mark lines that met a region say it).")
    if changed:
        totals["triangles_replaced"] = _replaced_triangles(mesh0, st.mesh)
        totals["edit_reach_mm"] = max(stp["edit"]["farthest_changed_node_from_mark_mm"] for e in entries
                                      for stp in e["steps"] if stp["status"] == "applied")

    # -- write
    files: dict = {"edited_stl": None, "report": "apply_marks.json"}
    reload_facts = None
    try:
        out.mkdir(parents=True, exist_ok=True)
        if changed:
            (out / "edited.stl").write_bytes(st.mesh.export(file_type="stl"))
            files["edited_stl"] = "edited.stl"
            back = trimesh.load(str(out / "edited.stl"), file_type="stl", force="mesh")
            reload_facts = mesh_facts(back)
            totals["reload_from_disk"] = reload_facts
            totals["edited_stl_sha256"] = fb._sha256(out / "edited.stl")
            sq = {"before": _surface_quality(d / "result" / "final.stl"), "after": _surface_quality(out / "edited.stl"),
                  "verify_edge_limit_mm": _tet_surface_limit(), "note": SURFACE_QUALITY_NOTE}
            totals["surface_quality"] = sq
            if sq["after"].get("closed_manifold") is False:
                problems.append("edited.stl is not a closed manifold in the sense of topopt verify (see "
                                "totals.surface_quality.after): the file is written, and verify would refuse it")
            plain = [stp["seam"] for e in entries for stp in e["steps"]
                     if stp["status"] == "applied" and (stp.get("seam") or {}).get("method") == "boolean"]
            fold0, fold1 = sq["before"].get("fold_edges"), sq["after"].get("fold_edges")
            if plain and fold0 is not None and fold1 is not None and fold1 > fold0:
                why = sorted({x.get("why_not_sewn") or "no reason given" for x in plain})
                problems.append(f"{len(plain)} applied step{'s' if len(plain) != 1 else ''} joined the new surface to the part with the "
                                f"plain boolean ({'; '.join(why[:2])}), and edited.stl has {fold1} fold edges (final.stl: {fold0}): "
                                "the surface check of topopt verify may refuse it (see totals.surface_quality and steps[].seam)")
            if not reload_facts["closed"] or reload_facts["bodies"] > facts0["bodies"]:
                problems.append("edited.stl does not reload as one closed part (see totals.reload_from_disk): the file is "
                                "written, and the facts are in the report")
        else:
            notes.append("No mark changed the part: edited.stl is not written (it would be a copy of final.stl).")
    except OSError as exc:
        raise UserError(f"cannot write in {out}: {type(exc).__name__}: {exc}. Name another out_dir.")
    stale = sorted(p.name for p in out.iterdir() if p.is_file() and p.name not in {"edited.stl", "apply_marks.json"})
    if stale:
        notes.append("Files from an earlier call stay in the folder: " + ", ".join(stale[:10]))
    if not changed and (out / "edited.stl").exists():
        notes.append("edited.stl in this folder is from an earlier call, not from this one.")

    # -- notes, routes, options
    notes += [
        "Each mark is one step: a crop box around the mark, the signed distance field of the part in it, the edit, and "
        "a splice of the new surface into the part, in a zone around the changed nodes. Triangles outside the zones are "
        "the triangles of final.stl, bit for bit (totals.triangles_replaced counts the others).",
        "Order: remove, then smooth, then add (the order of the pipeline stage s7_cleanup); marks of one kind in file "
        "order. A step that fails a check is rolled back, and the report says why.",
        "No FE run was made on the edited part: its stiffness and stress are not known. Stress facts near a mark are "
        "from the part before the edit.",
        *(["The parent check compares the medial axis and the skeleton before and after a cut. It does not see a pit that "
           "a cut digs into a flat face: read volume_change, thickness_near_mm and the stress near the mark with it."]
          if any(e.get("parent_check") for e in entries) else []),
        "Used from a mark: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in VALUES_USED.items())
        + ". Not used: " + ", ".join(VALUES_NOT_USED) + "."]
    if any(e.get("mirror") for e in entries):
        srcs = sorted({e["mirror"]["plane"]["source"] for e in entries if e.get("mirror")})
        notes.append("mirror: the plane comes from " + "; ".join(srcs) + ". run.json mirror (engine item A4a) is read as "
                     "{planes: [{axis, at_mm}]}: with more than one plane the mark has its images in the group of the "
                     "reflections (up to 8 paths), and images less than twice the reach apart are one site.")
    routes = [
        {"id": "use_edited_stl", "text": "edited.stl is the finished part with the marks that were applied. The run folder "
                                          "and final.stl are not changed. Read the facts of each mark first."},
        {"id": "rerun_with_hints", "text": "marks_to_hints (the same marks) makes a new YAML: remove becomes a keep-out "
                                            "region and add a forced-solid region. A new run of the optimizer then finds "
                                            "the shape, load paths included. It takes a full run. It fits when a mark was "
                                            "rolled back, clipped by a frozen region, or failed the parent check."}]
    try:
        import importlib.util
        if importlib.util.find_spec("topopt.verify") is not None:
            routes.append({"id": "verify", "text": "topopt.verify is in this install: it can check the edited part with an "
                                                   "independent FE run."})
    except (ImportError, ValueError):
        pass
    for e in entries:
        pc = e.get("parent_check") or {}
        if e["status"] in ("no_effect", "rolled_back", "guard", "partly_applied") or (pc and not pc.get("within_tolerance")) \
                or (e.get("edit") or {}).get("clipped_share_pct", 0) >= 50:
            e["route_note"] = "the rerun_with_hints route may fit this mark: " + (
                "the edit did not run" if e["status"] in ("no_effect", "rolled_back", "guard") else
                "the cut goes into the parent" if pc and not pc.get("within_tolerance") else "a frozen region clips the edit")
    summary = {f"{e['type'] or 'mark'}:{e['status']}": 0 for e in entries}
    for e in entries:
        summary[f"{e['type'] or 'mark'}:{e['status']}"] += 1
    report = {
        "ok": True, "format": APPLY_FORMAT, "run_dir": str(d), "run_name": run.get("name"), "marks_file": marks_file,
        "marks_sha256": marks_sha, "marks_count": len(doc["marks"]), "marks_match_final_stl": marks_match,
        "allow_stale": bool(allow_stale), "source": {"final_stl_sha256": final_sha},
        "out_dir": str(out), "grid": {"h": h, "shape": shape, "origin": origin, "pitch_mm": pitch,
                                      "pitch_is_default": pitch_default,
                                      "pitch_rule": f"voxel size of the run / {PITCH_DIVISOR:g}, kept between "
                                                    f"{PITCH_DEFAULT_RANGE[0]} and {PITCH_DEFAULT_RANGE[1]} mm"},
        "options": opt, "option_catalog": OPTION_TEXT, "marks": entries, "summary": summary, "totals": totals,
        "files": files, "routes": routes, "pipeline_defaults": _pipeline_defaults(entries, prm, overridden),
        "values_used": VALUES_USED, "values_not_used": VALUES_NOT_USED, "problems": problems, "notes": notes,
        "elapsed_s": round(time.monotonic() - t0, 2), "peak_rss_mb": _peak_rss_mb(), "load_avg": _load_avg(),
    }
    report = fb._jsonable(report)
    try:
        (out / "apply_marks.json").write_text(json.dumps(_portable(report, out, d, marks_file), indent=1))
    except OSError as exc:
        raise UserError(f"cannot write apply_marks.json in {out}: {exc}")
    return report


# ---- the short report and the script -------------------------------------------------------------------------
def _seam_methods(entry: dict):
    """How many applied steps of a mark joined the new surface to the part by each method ({"sewn": 2, "boolean": 1}), or None."""
    counts: dict = {}
    for stp in entry.get("steps", []):
        m = (stp.get("seam") or {}).get("method") if stp.get("status") == "applied" else None
        if m:
            counts[m] = counts.get(m, 0) + 1
    return counts or None


def short_report(report: dict) -> dict:
    """The short reply of the MCP tool `apply_marks`: what the agent needs to decide. The full report is in
    apply_marks.json (and in the return value of `apply_marks`)."""
    if not report.get("marks") and not report.get("out_dir"):
        return report                               # an error, or a refusal, with no report
    marks = []
    for e in report.get("marks", []):
        ed = e.get("edit") or {}
        mir = e.get("mirror") or {}
        pc = e.get("parent_check") or {}
        sn = e.get("stress_near_mark") or {}
        line = {"id": e.get("id"), "type": e.get("type"), "status": e.get("status"), "modes": e.get("modes"),
                "radius_used_mm": (e.get("values") or {}).get("radius_used_mm"),
                "mirror": (e.get("values") or {}).get("mirror"), "mirror_one_site": mir.get("combined"),
                "volume_change_mm3": ed.get("part_volume_change_mm3"), "mass_change_g": ed.get("part_mass_change_g"),
                "edit_reach_mm": ed.get("edit_reach_mm"), "edit_note": e.get("edit_note"),
                "clipped_by_frozen_mm3": ed.get("clipped_by_frozen_mm3"),
                "restored_in_frozen_mm3": ed.get("restored_in_frozen_mm3"), "frozen_note": e.get("frozen_note"),
                "thickness_near_mm": e.get("thickness_near_mm"),
                "parent_check_within_tolerance": pc.get("within_tolerance"), "parent_max_drop_mm": pc.get("max_drop_mm"),
                "bodies_after": e.get("bodies_after"), "closed_after": e.get("closed_after"),
                "seam_methods": _seam_methods(e), "dropped_bodies": e.get("dropped_bodies"),
                "nearest_frozen_to_edit_mm": e.get("nearest_frozen_to_edit_mm"),
                "stress_near_max_mpa": sn.get("near_max_mpa"), "stress_global_p99_mpa": sn.get("global_p99_mpa"),
                "defaults_used": [x["name"] for x in e.get("defaults_used", [])] or None,
                "surface_note": e.get("surface_note"), "reason": e.get("reason"), "route_note": e.get("route_note")}
        marks.append({k: v for k, v in line.items() if v is not None})
    t = report.get("totals") or {}
    keep = ("volume_before_mm3", "volume_after_mm3", "change_mm3", "change_pct", "mass_before_g", "mass_after_g", "change_g",
            "bodies_before", "bodies_after", "closed_after", "steps", "steps_applied", "marks", "restored_mm3")
    totals = {k: t.get(k) for k in keep}
    fe = t.get("frozen_exact") or {}
    totals["frozen_exact"] = {k: fe.get(k) for k in ("done", "unchanged", "added_in_frozen_mm3", "removed_from_frozen_mm3",
                                                    "reason", "restored") if k in fe}
    rl = t.get("reload_from_disk")
    if rl:
        totals["reload_from_disk"] = {k: rl.get(k) for k in ("closed", "bodies", "triangles")}
    sq = t.get("surface_quality")
    if sq:
        small = ("closed_manifold", "fold_edges", "non_manifold_vertices", "edge_min_mm", "edges_under_verify_limit", "error")
        totals["surface_quality"] = {w: {k: sq[w][k] for k in small if k in sq[w]} for w in ("before", "after")}
        totals["surface_quality"]["verify_edge_limit_mm"] = sq.get("verify_edge_limit_mm")
    for k in ("triangles_replaced", "edit_reach_mm"):
        if t.get(k) is not None:
            totals[k] = t[k]
    g = report.get("grid") or {}
    out = {"ok": report.get("ok"), "run_dir": report.get("run_dir"), "out_dir": report.get("out_dir"),
           "files": report.get("files"), "summary": report.get("summary"), "marks": marks, "totals": totals,
           "grid": {k: g.get(k) for k in ("h", "pitch_mm", "pitch_is_default")},
           "routes": report.get("routes"), "problems": report.get("problems"),
           "detail": False, "report_file": "apply_marks.json",
           "more": "apply_marks.json holds the full report: each step with its crop box, the edit volumes, the operator "
                   "numbers, the pipeline defaults with their anchors, the options with their text. Call again with "
                   "detail=true to get it in the reply."}
    if report.get("marks_match_final_stl") is False:
        out["marks_match_final_stl"] = False
    if report.get("error"):
        out["error"] = report["error"]
    return out


def _plane_arg(text: str) -> dict:
    """`y=0.5` or `x` (at 0) to a plane."""
    axis, _, at = text.partition("=")
    pl = _plane_from({"axis": axis.strip(), "at_mm": float(at) if at.strip() else 0.0}, "call option mirror_plane")
    if pl is None:
        raise UserError(f"--mirror-plane {text!r} is not x, y or z, with an optional =number")
    return pl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m topopt.apply_marks", description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("marks")
    ap.add_argument("--out", default=None, help="the new folder (default: <run folder>-marks)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-stale", action="store_true", help="apply marks that were made for another final.stl")
    ap.add_argument("--yaml", default=None, help="the YAML of the run, when run.json names a file that is not there")
    ap.add_argument("--detail", action="store_true", help="print the full report (apply_marks.json always holds it)")
    ap.add_argument("--pitch", type=float, default=None, help="node spacing of the crop box in mm")
    ap.add_argument("--radius-scale", type=float, default=None)
    ap.add_argument("--parent-check", choices=["report", "enforce"], default=None)
    ap.add_argument("--smooth-mode", choices=list(SMOOTH_MODES), default=None)
    ap.add_argument("--remove-mode", choices=list(REMOVE_MODES), default=None)
    ap.add_argument("--mirror-plane", default=None, help="for example y=0 or x=12.5")
    ap.add_argument("--skip", type=int, nargs="*", default=None, help="ids of marks that are not applied")
    ap.add_argument("--max-crop-voxels", type=float, default=None)
    a = ap.parse_args(argv)
    opts = {k: v for k, v in (("pitch_mm", a.pitch), ("radius_scale", a.radius_scale), ("parent_check", a.parent_check),
                              ("smooth_mode", a.smooth_mode), ("remove_mode", a.remove_mode), ("skip", a.skip),
                              ("max_crop_voxels", a.max_crop_voxels)) if v is not None}
    try:
        if a.mirror_plane:
            opts["mirror_plane"] = _plane_arg(a.mirror_plane)
        res = apply_marks(a.run_dir, a.marks, out_dir=a.out, options=opts, overwrite=a.overwrite,
                          allow_stale=a.allow_stale, yaml_path=a.yaml)
    except (UserError, ValueError) as exc:
        res = {"ok": False, "error": str(exc)}
    if not a.detail:
        res = short_report(res)
    print(json.dumps(fb._jsonable(res), indent=1))
    if not res.get("ok"):
        print(f"error: {res.get('error')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
