"""SDF mesh step: volume-preserving smoothing of the final density (CONTRACTS.md section 19).

The method is ported from the s9 print pass of the pipeline (`smooth_vp`, `close_kd`,
`mesh_field`, Taubin with pinned vertices), written for any voxel size:

1. The density is made finer (`up` times) and cut at 0.5. This is the sharp field.
2. Fixed and load regions (frozen) are added with their exact STL shape, if the STL exists.
   Keep-out regions are removed. Without STL files the voxel labels are used.
3. The sharp field becomes a signed distance field (SDF, negative inside).
4. A low-pass `2G - G^2` smooths the SDF. The second-order shrink of a plain Gaussian cancels.
   A local iso-level shift `c(x)` then keeps the solid volume in a moving Gaussian window
   equal to the volume of the sharp field. Where the envelope is known, the window counts only the solid inside it (#319).
5. The frozen solid is united with the smooth field, the keep-out solid is cut out. Where the envelope is known, the field is
   limited to it (the maximum with its SDF, `envelope_sdf`; #319): most of the material past a face of the design space came from
   the voxel layer of the density field at the face, and the cut alone paid it back as a thicker part everywhere.
6. Marching cubes, Taubin with pinned vertices (frozen and cut surfaces and, with the limit, the envelope stay),
   then the exact boolean operations with the STL files (manifold3d). With the limit, a vertex that lies inside a shared plane (below)
   by less than the band moves onto the plane first (`flush_to_planes`). Just before the booleans, the vertices that lie
   outside a region and within the band `pin_tol` x fine voxel of its exact surface are moved one band deep into the
   region (`snap_into_regions`): the smooth surface does not lie on the exact faces (micrometres to hundredths of a millimetre off),
   and the union or cut of two almost coplanar surfaces leaves a strip of micro-steps and slivers. A point-in-solid test of the
   whole region decides "outside" (not the normal of one face). The volume calibration sees the move.
   Limits: a vertex whose target one band deep is not inside the region (a region thinner than one band) stays; where a region is
   thinner than two bands, the vertices from its two sides can cross each other inside it before the union.
   The mesh is then cut at a plane that the design space shares with a fixed, load or solid region (`shared_planes`, `clip_to_planes`) when a part
   of it stands past the plane over a face of that region (or within the band of its edge): the snap takes only what is within the band of a face,
   and the surface can go farther past it (the staircase of the frozen voxels, the low-pass, the level shift). A plane that the mesh goes past
   only away from the region faces is not cut; the facts list it (`left`).
   Then the mesh is cut to the envelope (`make_envelope`, `cut_to_envelope`): the design space united with the frozen regions and the union pieces
   of the interfaces at their as-printed size. An exact intersection, so that no part of the result lies outside the room that the user gave, also on
   a slanted or curved face of the design space. The volume calibration builds the mesh with the cut, so the mass target holds. With the
   limit of step 5, only a few vertices are left to cut (marching cubes cuts a concave edge of the envelope with a chamfer).
   Before the smoothing, where the envelope is known, the sharp field grows to the surface of the design space (`grow_to_surface`; `settings.envelope_fill`,
   CONTRACTS.md section 19): the staircase of a slanted face is filled, and the cut to the envelope then makes the face flat. The volume target is the volume before the growth.
   After the booleans, the closed voids are filled (`_drop_voids`, before `_orient`): the snap can close the narrow rim of a thin slit
   between a region face and the design, and the union then seals the rest of it as a void, which `_orient` would turn into a thin second body.
   A void that encloses a closed cavity of the sharp field stays: the design has the cavity, and the volume target does not hold it,
   so the mesh keeps it as a cavity, and `_orient` keeps its faces turned inward.

All lengths in `SdfParams` are in multiples of the voxel size `h` unless the name ends in `_mm`.
The module needs only numpy, scipy, scikit-image, trimesh and manifold3d.
"""
from __future__ import annotations

import gc
import io
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import ENVELOPE_FILL_DEFAULT
from .resource_errors import reraise_if_resource

log = logging.getLogger("topopt.smooth_sdf")

# The cost of the envelope fill, in `mesh.envelope_fill.note` and the log when the fill ran (CONTRACTS.md section 19)
FILL_COST_NOTE = ("the fill moves material from the members to the skin of the design space: at the same mass the stiffness can drop, "
                  "more on a coarse grid (CONTRACTS.md section 19). Option: settings.envelope_fill: off")

# The estimated peak memory of the step (`grid_fact`; the guard counts only the voxels): BYTES_PER_VOXEL for each fine voxel plus
# BYTES_PER_TRIANGLE for each estimated triangle of the mesh (`surface_triangles`, #429). Measured as the rise of the step above its start:
# the grid stage takes 64 to 72 bytes per fine voxel (a generated lattice of 14 million fine voxels, the hook at 22 million). The mesh stage
# takes the rest, which grows with the mesh: on the lattice at 6.2 and 14 million fine voxels (2.0 and 5.0 million triangles), the peak above
# 70 bytes per voxel is 1,410 and 1,392 bytes per estimated triangle (1,302 and 1,273 per triangle of the mesh). The constants are the upper
# end. The hook at 9.6 and 22 million voxels takes 106 and 95 bytes per voxel (the higher of two runs), and the estimate gives 133 and 119
# (its estimated triangle count is about twice the real one).
BYTES_PER_VOXEL = 70
BYTES_PER_TRIANGLE = 1450
# Default limit of the fine grid. Override: env TOPOPT_SDF_MAX_VOXELS.
MAX_SDF_VOXELS = 30_000_000
# The budget of a distance query: `_TriangleDistance` holds about PAIRS point-piece pairs at a time and asks NEAREST pieces for each point
# first; the trimesh queries of this step (`_query`) hold about PAIRS point-face pairs at a time and count them PAIRS // NEAREST points at a
# time (#429).
PAIRS, NEAREST = 262144, 32


@dataclass(frozen=True)
class SdfParams:
    sigma: float = 1.0          # low-pass sigma, in h (the pipeline uses 0.9 mm on a 0.5 mm voxel)
    window: float = 1.67        # sigma of the volume window, in sigma units
    level_range: float = 0.67   # level shifts from -r to +r, in sigma units
    n_levels: int = 13
    clip: float = 2.8           # clip of the SDF before smoothing, in sigma units
    near: float = 1.33          # c(x) is used where |F| < near, in sigma units
    taubin_iterations: int = 10
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53
    pin_tol: float = 0.35       # vertices closer than this to a frozen or cut surface stay in Taubin; the band of the snap; in fine voxels
    calibrate_steps: int = 3    # global level steps so that the mesh volume meets the sharp volume (the plain update, see `calibrate`)
    calibrate_steps_fill: int = 8   # the round limit when the envelope fill ran, and when a run still misses after `calibrate_steps` rounds (#348; the loop ends when the error is small, so a part that needs 3 rounds takes 3)
    extra_rounds_from: float = 1e-3  # a run without the fill takes the extra rounds of #348 only when its error after `calibrate_steps` rounds is at least this (the bar of #348: 0.1 %)
    envelope_limit: bool = True     # #319: the field is limited to the envelope before marching cubes, and the local volume window counts only the solid inside it
    surface_depth: float = 1.0      # envelope surface: a voxel up to this deep (in h) under the design-space surface can be grown
    surface_overshoot: float = 1.0  # envelope surface: and up to this far (in fine voxels) past it; the cut to the envelope then makes the face
    fill_max_share: float = 0.05    # envelope fill: skip it when the volume that it would add inside the design space is above this share of the volume target
    fill_slope_min: float = 0.2     # envelope fill: the target of a calibration round moves by the miss divided by the measured slope of the mesh volume to the target; the slope is kept in this range
    fill_slope_max: float = 1.5
    envelope_fill: str = ENVELOPE_FILL_DEFAULT  # envelope fill: off (the default), auto (skip it above fill_max_share) or on (run it above the cap too)
    band: float = 3.0           # width of the exact SDF near an STL surface, in fine voxels
    simplify_tol: float = 1e-5  # manifold3d simplify tolerance (mm) before the STL is written; 0 = off
    deep_depth: float = 1.5     # a sharp-field voxel this far (fine voxels) outside the mesh counts as lost


class SdfGridTooLarge(Exception):
    """The fine grid is over the guard. `.fact` holds the numbers and the options."""

    def __init__(self, fact: dict):
        super().__init__(fact["message"])
        self.fact = fact


# -- guard -------------------------------------------------------------------
def physical_ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def grid_fact(fine_shape, h: float, limit: int | None = None, triangles: int = 0) -> dict:
    """Facts about the SDF grid: voxel count, estimated peak memory, the limit and the options. `triangles`: the estimated triangle
    count of the mesh (`surface_triangles`)."""
    n = int(np.prod([int(s) for s in fine_shape], dtype=np.float64))
    note = None
    if limit is None:
        env = os.environ.get("TOPOPT_SDF_MAX_VOXELS")
        limit = MAX_SDF_VOXELS
        if env:
            try:
                limit = int(float(env))
            except ValueError:
                note = (f"TOPOPT_SDF_MAX_VOXELS={env!r} is not a number, so the default limit {MAX_SDF_VOXELS} is used. "
                        "Option: set a number (for example 60000000) or unset the variable.")
                log.warning(note)
    est = n * BYTES_PER_VOXEL + int(triangles) * BYTES_PER_TRIANGLE
    ram = physical_ram_bytes()
    ok = n <= limit
    fact = {"voxels": n, "limit": int(limit), "est_triangles": int(triangles), "est_peak_gib": round(est / 2 ** 30, 2),
            "ram_gib": round(ram / 2 ** 30, 1) if ram else None, "ok": ok}
    if note:
        fact["env_note"] = note
    if not ok:   # the options are only given when the step cannot run
        options = [f"voxel_size {round(h * (n / limit) ** (1 / 3) * 1.03 + 0.005, 2)} or more",
                   "settings.mesh_method: blur (the old Gaussian path, no SDF grid)",
                   "env TOPOPT_SDF_MAX_VOXELS=<number> to raise the limit"]
        fact["message"] = (f"the SDF grid has {n} voxels (about {est / 2 ** 30:.1f} GiB peak"
                           + (f", this machine has {ram / 2 ** 30:.0f} GiB" if ram else "")
                           + f"), the limit is {limit}. Options: " + "; ".join(options) + ".")
        fact["options"] = options
    else:
        fact["message"] = f"the SDF grid has {n} voxels (about {est / 2 ** 30:.1f} GiB peak), the limit is {limit}."
    return fact


def surface_triangles(rho, up: int) -> int:
    """The estimated triangle count of the mesh, before the fine grid is made (#429): 2 x up^2 triangles for each face between a solid
    (rho > 0.5) and a void coarse voxel, the outside counted as void. Measured: the mesh has 1.08 to 1.09 times this count on a generated
    lattice (thin straight members), 0.44 to 0.47 times on the hook and 0.44 to 0.90 times on 28 finished runs; a part with large flat
    faces has fewer (the simplify step merges them)."""
    s = np.pad(np.asarray(rho) > 0.5, 1)
    return 2 * up * up * int(sum(np.count_nonzero(np.diff(s, axis=a)) for a in range(3)))


# -- signed distance fields ----------------------------------------------------
def binary_sdf(mask: np.ndarray, pitch: float) -> np.ndarray:
    """SDF (negative inside, float32, in mm) of a voxel mask. The surface is half a voxel from the centres."""
    from scipy import ndimage as ndi

    out = np.empty(mask.shape, np.float32)
    d = ndi.distance_transform_edt(~mask)
    out[...] = d - 0.5
    del d
    d = ndi.distance_transform_edt(mask)
    np.copyto(out, (0.5 - d).astype(np.float32), where=mask)
    del d
    out *= np.float32(pitch)
    return out


def _sample_surface(mesh, spacing: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tri = mesh.triangles
    area = mesh.area_faces
    n = np.maximum(np.ceil(area / spacing ** 2).astype(np.int64), 1)
    idx = np.repeat(np.arange(len(tri)), n)
    r1 = rng.random(len(idx))
    r2 = rng.random(len(idx))
    flip = r1 + r2 > 1
    r1[flip] = 1 - r1[flip]
    r2[flip] = 1 - r2[flip]
    pts = tri[idx, 0] + (tri[idx, 1] - tri[idx, 0]) * r1[:, None] + (tri[idx, 2] - tri[idx, 0]) * r2[:, None]
    return np.vstack([pts, mesh.vertices])


def stl_sdf(mesh, origin_f: np.ndarray, pitch: float, shape, band: float) -> np.ndarray:
    """SDF of a closed mesh on the grid: within `band` voxels of the surface the distance to the nearest surface sample, else +-clip.

    The sign comes from a ray-cast mask, the distance from a KD-tree on dense surface samples (spacing pitch / 2.5). It is not the
    distance to the triangles: it is never smaller (measured at c080f62 on one host, Linux x86_64: on the hook, the ladle and the bracket
    it is larger in 99 % of the voxels of the band, by 3 % of a fine voxel on average and by up to 0.7 of a fine voxel in the worst
    voxel). So the zero level of the field does not lie on the exact face, and `snap_into_regions` takes the exact surface of the
    region, not this field.
    """
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    from .voxelize import inside_mask

    mask = inside_mask(mesh, origin_f, pitch, shape)
    clip = np.float32((band + 1.0) * pitch)
    out = np.where(mask, -clip, clip).astype(np.float32)
    if not mask.any():
        return out
    # Work on the crop around the mask only.
    idx = np.argwhere(mask)
    pad = int(np.ceil(band)) + 2
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + 1 + pad, np.asarray(shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    m = mask[sl]
    shell = m ^ ndi.binary_erosion(m, border_value=0)
    near = ndi.distance_transform_edt(~shell) <= band + 0.5
    ii = np.argwhere(near)
    pts = origin_f + pitch * (ii + lo + 0.5)
    kd = cKDTree(_sample_surface(mesh, pitch / 2.5))
    d, _ = kd.query(pts, workers=-1)
    d = np.minimum(d, clip).astype(np.float32)
    sub = out[sl]
    sign = np.where(m[near], -1.0, 1.0).astype(np.float32)
    sub[near] = sign * d
    return out


def _point_segment_distance(p: np.ndarray, s0: np.ndarray, s1: np.ndarray):
    """Distance of each point of `p` to the segment `(s0, s1)` of the same row, and the closest point. A segment of no length is a point."""
    sv = s1 - s0
    ll = (sv * sv).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(np.where(ll > 0, ((p - s0) * sv).sum(1) / ll, 0.0), 0.0, 1.0)
    q = s0 + sv * t[:, None]
    return np.sqrt(((p - q) ** 2).sum(1)), q


def _point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, point: bool = False):
    """Distance of each point of `p` to the triangle `(a, b, c)` of the same row (arrays of shape (n, 3)); with `point`, `(distance,
    closest point)`.

    The closest point is found by the region of the point: a vertex, an edge or the inside of the triangle (the tests with the dot products
    of the edges). A triangle of no area, or one so thin that these tests lose their digits (the square of the sine of its angle at `a`
    under 1e-8), gives the distance to the nearest of its 3 edges: the error is at most its width.
    """
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        q = a + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]                                   # the inside
        e = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[:, None], b + (c - b) * e[:, None], q)   # edge bc
        w = d2 / (d2 - d6)
        q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], a + ac * w[:, None], q)                 # edge ac
        q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, q)                                               # vertex c
        v = d1 / (d1 - d3)
        q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], a + ab * v[:, None], q)                 # edge ab
        q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, q)                                               # vertex b
        q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, q)                                                # vertex a
        d = np.sqrt(((p - q) ** 2).sum(1))
        flat = np.flatnonzero(~(den > 1e-8 * (ab * ab).sum(1) * (ac * ac).sum(1)))
    if len(flat):
        pf, best_d, best_q = p[flat], None, None
        for s0, s1 in ((a, b), (b, c), (c, a)):
            ds, qs = _point_segment_distance(pf, s0[flat], s1[flat])
            if best_d is None:
                best_d, best_q = ds, qs
            else:
                nearer = ds < best_d
                best_d, best_q = np.where(nearer, ds, best_d), np.where(nearer[:, None], qs, best_q)
        d[flat], q[flat] = best_d, best_q
    corners = np.sqrt(np.minimum(np.minimum((ap ** 2).sum(1), (bp ** 2).sum(1)), (cp ** 2).sum(1)))
    out = np.fmin(d, corners)
    if not point:
        return out
    corner = np.stack([a, b, c], 1)[np.arange(len(p)), np.argmin(np.stack([(ap ** 2).sum(1), (bp ** 2).sum(1), (cp ** 2).sum(1)], 1), 1)]
    return out, np.where((~(d <= corners))[:, None], corner, q)


def _split_to_size(tri: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Split each triangle (array (n, 3, 3)) at the middle of its longest edge until no edge is longer than `size`. Return the pieces and
    the index of the triangle of each piece. A long thin triangle gives about (length / size) pieces."""
    done, done_src, src = [], [], np.arange(len(tri))
    for _ in range(64):
        edge = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2)       # edge i goes from vertex i to vertex i + 1
        big = edge.max(1) > size
        done.append(tri[~big])
        done_src.append(src[~big])
        tri, src, i = tri[big], src[big], edge[big].argmax(1)
        if not len(tri):
            break
        r = np.arange(len(tri))
        v0, v1, v2 = tri[r, i], tri[r, (i + 1) % 3], tri[r, (i + 2) % 3]
        mid = 0.5 * (v0 + v1)
        tri, src = np.concatenate([np.stack([v0, mid, v2], 1), np.stack([mid, v1, v2], 1)]), np.concatenate([src, src])
    return np.concatenate(done + [tri]), np.concatenate(done_src + [src])


class _TriangleDistance:
    """Exact distance of points to the nearest of the triangles `tri` (array (n, 3, 3)), with little memory. Make it once, call it per chunk.

    The triangles are split into pieces no longer than `size`, and a KD-tree on the centres of the pieces gives each point its `k` nearest
    pieces. The point takes the least distance to the triangles of these pieces (each triangle once). That is the exact distance when the
    `k`-th centre is at least that distance plus the largest centre-to-corner distance of a piece away: then every other piece, and so
    every other triangle, is farther. The points without that proof go round again with twice as many pieces, up to all of them. The work
    is in chunks of about `pairs` point-piece pairs (a few tens of MB).
    """

    def __init__(self, tri: np.ndarray, size: float):
        from scipy.spatial import cKDTree

        self.tri = np.asarray(tri, np.float64)
        self.pieces, self.src = _split_to_size(self.tri, size)
        centre = self.pieces.mean(1)
        self.reach = float(np.sqrt(((self.pieces - centre[:, None]) ** 2).sum(2)).max())
        self.kd = cKDTree(centre)

    def __call__(self, pts: np.ndarray, k: int = NEAREST, pairs: int = PAIRS, point: bool = False):
        """The distance of each point; with `point`, `(distance, closest point)`."""
        tri, pieces, src, reach, kd = self.tri, self.pieces, self.src, self.reach, self.kd
        pts = np.asarray(pts, np.float64)
        out = np.empty(len(pts))
        near = np.empty((len(pts), 3)) if point else None
        todo = np.arange(len(pts))
        while len(todo):
            kk = min(k, len(pieces))
            chunk, left = max(1, pairs // kk), []
            for s in range(0, len(todo), chunk):
                idx = todo[s:s + chunk]
                p = pts[idx]
                dc, j = kd.query(p, k=kk, workers=-1)
                dc, j = dc.reshape(len(p), kk), j.reshape(len(p), kk)
                own = np.sort(src[j], axis=1)
                first = np.ones(own.shape, bool)
                first[:, 1:] = own[:, 1:] != own[:, :-1]
                row, col = np.nonzero(first)                 # each triangle of a point once; the rows come in order
                t = tri[own[row, col]]
                start = np.searchsorted(row, np.arange(len(p)))
                if point:
                    dist, q = _point_triangle_distance(p[row], t[:, 0], t[:, 1], t[:, 2], point=True)
                    best = np.lexsort((dist, row))[start]    # the nearest triangle of each point (rows in order, the least distance first)
                    d = dist[best]
                    near[idx] = q[best]
                else:
                    dist = _point_triangle_distance(p[row], t[:, 0], t[:, 1], t[:, 2])
                    d = np.minimum.reduceat(dist, start)
                out[idx] = d
                if kk < len(pieces):
                    left.append(idx[dc[:, -1] < d + reach])
            todo = np.concatenate(left) if left else todo[:0]
            k *= 2
        return (out, near) if point else out


def _triangle_distance(tri: np.ndarray, pts: np.ndarray, size: float, k: int = NEAREST, pairs: int = PAIRS, point: bool = False):
    """`_TriangleDistance(tri, size)(pts, ...)`: the exact distance of each point to the nearest triangle; with `point`, `(distance,
    closest point)`."""
    return _TriangleDistance(tri, size)(pts, k=k, pairs=pairs, point=point)


def _query_pairs(mesh, points) -> np.ndarray:
    """For each point, the faces of `mesh` that a `trimesh.proximity` query looks at: the faces whose box meets the box of the sphere
    through the nearest vertex (as `trimesh.proximity.nearby_faces`). Counted PAIRS // NEAREST points at a time; the ids are not kept."""
    from scipy.spatial import cKDTree
    import trimesh

    kd = cKDTree(mesh.vertices[mesh.referenced_vertices])
    tree = mesh.triangles_tree
    count = np.empty(len(points), np.int64)
    step = PAIRS // NEAREST
    for s in range(0, len(points), step):
        p = points[s:s + step]
        r = kd.query(p)[0][:, None] + trimesh.tol.merge
        count[s:s + step] = tree.intersection_v(p - r, p + r)[1]
    return count


def _query(fn, mesh, points):
    """`fn(mesh, points)`, a `trimesh.proximity` query (`closest_point`, `signed_distance`), with about PAIRS point-face pairs at a time (#429).

    trimesh makes its arrays for all the pairs of a point and a nearby face of one call at once: on a mesh of a million vertices (6.2
    million fine voxels) one query of the snap took 3.5 GB more, and one of the cut to the envelope 2.9 GB; a snap of 37,000 points to a
    region of 2,000 faces took 3.0 GB (hundreds of faces for each point). So the pairs of each point are counted first (`_query_pairs`),
    and the calls take runs of points with up to PAIRS pairs (a point with more takes a call of its own). The answer for a point does not
    depend on the other points of the call, so the chunks give the same arrays. A call that fits in one chunk is the call itself.
    """
    points = np.asarray(points, np.float64)
    if len(points) < 2 or len(mesh.faces) == 0:
        return fn(mesh, points)
    cum = np.concatenate([[0], np.cumsum(_query_pairs(mesh, points))])
    if cum[-1] <= PAIRS:
        return fn(mesh, points)
    bounds = [0]
    while bounds[-1] < len(points):
        b = bounds[-1]
        bounds.append(max(b + 1, int(np.searchsorted(cum, cum[b] + PAIRS, side="right")) - 1))
    parts = [fn(mesh, points[b:e]) for b, e in zip(bounds[:-1], bounds[1:])]
    if isinstance(parts[0], tuple):
        return tuple(np.concatenate(col) for col in zip(*parts))
    return np.concatenate(parts)


def envelope_sdf(mesh, origin_f: np.ndarray, pitch: float, shape, band: float) -> np.ndarray:
    """SDF of the envelope for the limit of the field (#319): the exact distance to the triangles near the surface, +-clip elsewhere.

    The sign comes from the ray-cast mask (`inside_mask`), as in `stl_sdf`, and the clip is the one of `stl_sdf`, (band + 1) fine voxels.
    The distance of `stl_sdf` goes to random samples of the surface. Next to the surface it can be larger than the true distance by a large
    part of a fine voxel, and the zero level of a limit made with it then lies up to about 0.4 of a fine voxel past the face. So every voxel
    within 2 voxels (in each axis) of a voxel at the surface (a voxel with a neighbour on the other side of the mask) gets the distance to
    the triangles (`_triangle_distance`). That holds every voxel closer than 2 voxels to a face that the mask resolves (voxel centres on
    both sides), more than the limit reads: its zero level, the pin band of Taubin (0.35 voxels) and the test of a point on the envelope.
    (On the envelopes of the series, 4 layers gave the same values within 2 voxels and took twice the time.) The envelope fills most of the grid,
    so the work goes slab by slab along the first axis, with no distance transform of the whole grid: the memory beyond the mask and the
    result is a few bytes per voxel (the distance transform and the voxel list of `stl_sdf` took about 75 bytes per voxel at the peak).
    """
    from scipy import ndimage as ndi

    from .voxelize import inside_mask

    mask = inside_mask(mesh, origin_f, pitch, shape)
    clip = np.float32((band + 1.0) * pitch)
    out = np.where(mask, -clip, clip).astype(np.float32)
    if mask.any():
        cube = np.ones((3, 3, 3), bool)
        surface = mask & ~ndi.binary_erosion(mask, cube, border_value=0)
        surface |= ndi.binary_dilation(mask, cube) & ~mask
        near = ndi.binary_dilation(surface, cube, iterations=2)
        del surface
        dist = _TriangleDistance(np.asarray(mesh.triangles, np.float64), 2.0 * pitch)
        for s0 in range(0, shape[0], 16):
            idx = np.nonzero(near[s0:s0 + 16])
            if not len(idx[0]):
                continue
            pts = origin_f + pitch * (np.column_stack(idx) + (s0, 0, 0) + 0.5)
            d = np.minimum(dist(pts), clip)
            out[s0:s0 + 16][idx] = (np.where(mask[s0:s0 + 16][idx], -1.0, 1.0) * d).astype(np.float32)
    # A grid point on the face (a face on a plane of voxel centres) has the distance 0, and marching cubes on a value at the level makes
    # triangles of no area and a flat body of no volume that no boolean takes. Such a point counts as just outside.
    tiny = np.float32(1e-3 * pitch)
    out[np.abs(out) < tiny] = tiny
    return out


# -- low-pass with local volume correction ---------------------------------------------
def smooth_vp(F: np.ndarray, ref_solid: np.ndarray, sigma: float, pitch: float, p: SdfParams,
              lock_in: np.ndarray | None = None, lock_out: np.ndarray | None = None, room: np.ndarray | None = None):
    """Low-pass `2G - G^2` of the SDF `F`, then a local level shift `c(x)`.

    `c` is chosen so that the solid volume in a Gaussian window equals the volume of `ref_solid`.
    The solid of a level `l` is `((F3 < l) | lock_in) & ~lock_out`, the same rule as the final
    composition, so frozen and cut voxels do not disturb the shift.
    Return `(F3 - c, c)`; all arrays are float32.
    """
    from scipy import ndimage as ndi

    f32 = np.float32
    s = sigma / pitch
    sw = p.window * s

    def gf(a, sg):
        return ndi.gaussian_filter(a, sg, mode="nearest")

    clip = p.clip * sigma
    G1 = gf(np.clip(F, -clip, clip).astype(f32), s)
    G2 = gf(G1, s)
    G1 *= 2
    G1 -= G2
    del G2
    F3 = G1
    A0 = gf(ref_solid.astype(f32), sw)
    near = (np.abs(F3) < p.near * sigma) | (np.abs(F) < p.near * sigma)
    levels = np.linspace(-p.level_range * sigma, p.level_range * sigma, p.n_levels)
    c = np.full(F.shape, np.nan, f32)
    prev = None
    for k, lv in enumerate(levels):
        solid = F3 < lv
        if lock_in is not None:
            solid |= lock_in
        if lock_out is not None:
            solid &= ~lock_out
        if room is not None:   # #319: the solid past the envelope is cut away, so it does not count; c(x) gives its volume back nearby
            solid &= room
        A = gf(solid.astype(f32), sw)
        del solid
        if prev is None:
            c[A >= A0] = lv
        else:
            Ap, lp = prev
            hit = (Ap < A0) & (A >= A0) & np.isnan(c)
            tt = np.where(A > Ap, (A0 - Ap) / np.maximum(A - Ap, 1e-9), 0.5).astype(f32)
            c[hit] = (lp + tt * (lv - lp))[hit]
            del tt, hit
        prev = (A, lv)
    last = prev[0]
    nanm = np.isnan(c)
    c[nanm] = np.where(last < A0, levels[-1], levels[0]).astype(f32)[nanm]
    del nanm, last, prev, A, A0
    c *= near
    num = gf(c, sw)
    den = gf(near.astype(f32), sw)
    del near
    cs = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0).astype(f32)
    del num, den, c
    F3 -= cs
    gc.collect()
    return F3, cs


# -- mesh -------------------------------------------------------------------------------
def taubin_pinned(V: np.ndarray, F: np.ndarray, iters: int, lam: float, mu: float,
                  pin: np.ndarray | None) -> np.ndarray:
    """Taubin smoothing with a uniform Laplacian. Pinned vertices stay."""
    import scipy.sparse as sp

    n = len(V)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    L = sp.diags(1.0 / np.maximum(deg, 1)) @ A
    V = V.copy()
    free = np.ones(n, bool) if pin is None else ~pin
    for _ in range(iters):
        for f in (lam, mu):
            d = L @ V - V
            V[free] += f * d[free]
    return V


def _signed_volume(verts, faces) -> float:
    v = np.asarray(verts, np.float64)
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def _march(F: np.ndarray):
    from skimage.measure import marching_cubes

    v, f, _n, _val = marching_cubes(F, level=0.0, allow_degenerate=False)
    return v, f   # normals point to higher values (outside of an SDF)


def stl_roundtrip(verts, faces) -> dict:
    """Write the mesh as binary STL in memory and report the facts of the loaded file (`stl_facts`)."""
    import trimesh

    return stl_facts(trimesh.Trimesh(verts, faces, process=False).export(file_type="stl"))


# -- exact boolean operations ---------------------------------------------------------------
def _load_manifold(path):
    """Manifold of an STL file. A mesh in memory (an interface shape, CONTRACTS.md section 23) is taken as it is."""
    import trimesh

    from .post import _to_manifold

    if hasattr(path, "vertices"):
        return _to_manifold(path.vertices, path.faces)
    t = trimesh.load(path, force="mesh")
    t.merge_vertices()
    return _to_manifold(t.vertices, t.faces)


def _manifold_of(verts, faces):
    import trimesh

    from .post import _to_manifold

    try:
        return _to_manifold(verts, faces)
    except ValueError:
        tm = trimesh.Trimesh(verts, faces, process=True)
        tm.merge_vertices()
        return _to_manifold(tm.vertices, tm.faces)


def _from_manifold(man):
    out = man.to_mesh()
    return (np.asarray(out.vert_properties, np.float64)[:, :3], np.asarray(out.tri_verts, np.int64))


def _split_faces(faces, n_verts):
    """Component label of each face (faces that share a vertex are one component)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]]])
    g = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_verts, n_verts))
    _n, lab = connected_components(g, directed=False)
    return lab[faces[:, 0]]


def _keep_faces(verts, faces, keep_mask):
    f = faces[keep_mask]
    used = np.unique(f)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[f]


def _keep_touching(verts, faces, touches):
    """Keep the bodies for which `touches(body_vertex_ids)` is true. If none, keep the largest.

    A void (a closed shell that turns inward: a negative signed volume) is not a body. It stays with the body around it, and
    `_drop_voids` decides on it later; a void whose innermost body around it is removed goes with that body, and its volume is not part of
    the removed volume. The faces must turn outward from the material (the marching cubes and the manifold3d output do): a mesh that is
    inside out has only voids, and nothing is removed.
    Return (verts, faces, n_removed, removed_volume): the count of the removed bodies and the volume of their material.
    """
    lab = _split_faces(faces, len(verts))
    n = int(lab.max()) + 1
    if n <= 1:
        return verts, faces, 0, 0.0
    vols = np.zeros(n)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    np.add.at(vols, lab, np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0)
    body = vols >= 0.0
    keep = ~body
    for k in np.flatnonzero(body):
        keep[k] = bool(touches(np.unique(faces[lab == k])))
    if not keep[body].any():
        keep[int(np.argmax(np.where(body, vols, -np.inf)))] = True
    gone = body & ~keep
    removed = float(vols[gone].sum())
    if gone.any():   # a void goes with the innermost body around it: removed with a removed body
        for k in np.flatnonzero(~body):
            point = verts[faces[lab == k][0, 0]]
            around = [j for j in np.flatnonzero(body) if _inside_shell(point, verts, faces[lab == j])]
            if around and gone[min(around, key=lambda j: vols[j])]:
                keep[k] = False
                removed += float(vols[k])
    vv, ff = _keep_faces(verts, faces, keep[lab])
    return vv, ff, int(gone.sum()), removed


def _drop_voids(verts, faces, keep=None):
    """Fill the closed voids of a mesh: drop each closed shell that holds empty space inside the material.

    The faces must turn outward from the material (the marching cubes and the manifold3d output do): a solid body then has a
    positive signed volume and a void (a shell with empty space inside it) a negative one. `_orient` makes the faces of each shell turn
    outward, so it would turn a void inside out, and the file would then hold a thin second body inside the material. A void is made,
    for example, when the snap closes the narrow rim of a thin slit between the face of a fixed region and the design, and the exact
    union with the region seals the rest. The volume of the void is material after the fill, so the volume calibration counts it as material.

    `keep(shell_verts, shell_faces)`: True for a void that stays (a closed cavity of the sharp field, so the void is part of the
    design and its volume is not material in the target either). Such a void is not filled, and `_orient` keeps its faces turned inward.

    Return `(verts, faces, facts)`; `facts` has `count` and `volume_mm3` (the volume of the filled voids, positive), and `kept`
    (`count`, `volume_mm3`) when a void stays. With nothing to fill, the arrays come back as they are. A mesh with no solid shell (all
    shells turn inward) is not changed either: its faces are not oriented.
    """
    none = {"count": 0, "volume_mm3": 0.0}
    if len(faces) == 0:
        return verts, faces, none
    lab = _split_faces(faces, len(verts))
    if len(np.unique(lab)) <= 1:
        return verts, faces, none
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vols = np.zeros(int(lab.max()) + 1)
    np.add.at(vols, lab, np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0)
    void = vols < 0.0
    if not void.any() or not (vols > 0.0).any():
        return verts, faces, none
    stays = np.zeros_like(void)
    if keep is not None:
        for k in np.flatnonzero(void):
            stays[k] = bool(keep(verts, faces[lab == k]))
    fill = void & ~stays
    facts = {"count": int(fill.sum()), "volume_mm3": float(-vols[fill].sum()) if fill.any() else 0.0}
    if stays.any():
        facts["kept"] = {"count": int(stays.sum()), "volume_mm3": float(-vols[stays].sum())}
    if not fill.any():
        return verts, faces, facts
    vv, ff = _keep_faces(verts, faces, ~fill[lab])
    return vv, ff, facts


def _inside_shell(point, verts, shell_faces) -> bool:
    """True if the point lies inside the closed shell, whatever the orientation of its faces (the winding number of the print check)."""
    from .printcheck import _winding_number

    return bool(abs(_winding_number(np.asarray(point, np.float64), np.asarray(verts, np.float64)[shell_faces])) > 0.5)


def closed_cavities(S, origin_f, pitch: float) -> dict:
    """The closed cavities of the sharp field `S` (bool, fine grid): the components of its empty voxels that do not reach the border.

    Return `{"count", "volume_mm3", "points"}`: `points` has one world point for each cavity, the middle of its voxels that lie farthest
    from the solid (or one of them, if the middle is not in the cavity), so that the wall of the cavity in the mesh encloses it. On a
    flat cavity two fine layers tie, and a voxel centre lies a quarter of a coarse voxel off the mid-plane: the smoothed wall can miss it.
    A keep-out region that the solid encloses is a cavity too.
    """
    from scipy import ndimage as ndi

    lab, n = ndi.label(~S)
    if n <= 1:
        return {"count": 0, "volume_mm3": 0.0, "points": np.zeros((0, 3))}
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    points, voxels = [], 0
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or k in border:
            continue
        sub = lab[sl] == k
        voxels += int(sub.sum())
        deep = ndi.distance_transform_edt(np.pad(sub, 1))[1:-1, 1:-1, 1:-1]
        top = np.argwhere(deep == deep.max())
        at = top.mean(axis=0)   # a flat cavity of an even count of fine layers ties two of them: the point is the mid-plane between them
        if not sub[tuple(np.clip(np.floor(at + 0.5).astype(int), 0, np.array(sub.shape) - 1))]:
            at = top[0].astype(np.float64)   # the middle of the deepest voxels is outside a cavity that bends: take one of them
        points.append(np.asarray(origin_f, np.float64) + pitch * (at + [q.start for q in sl] + 0.5))
    del lab
    return {"count": len(points), "volume_mm3": voxels * pitch ** 3, "points": np.array(points).reshape(-1, 3)}


def _orient(verts, faces):
    """Turn the faces outward from the material, with one winding. A cavity keeps its faces turned inward: a closed shell
    with a negative volume inside a shell with a positive volume is the wall of a void, and `fix_normals` alone turns it outward,
    a second body inside the material. The signs are read with the mesh turned right side out, as the print check reads them. An input
    whose outer shell is inside out and whose cavity turns inward is ambiguous and comes out as two bodies; the callers pass the output of
    manifold3d, which has no such shells."""
    import trimesh

    verts, faces = np.asarray(verts, np.float64), np.asarray(faces, np.int64)
    cavity = None
    lab = _split_faces(faces, len(verts)) if len(faces) else np.zeros(0, np.int64)
    if len(faces) and lab.max() > 0:
        vols = np.zeros(int(lab.max()) + 1)
        np.add.at(vols, lab, np.einsum("ij,ij->i", verts[faces[:, 0]], np.cross(verts[faces[:, 1]], verts[faces[:, 2]])) / 6.0)
        vols = vols if vols.sum() >= 0.0 else -vols   # the signs of the mesh turned right side out, as the print check reads them
        bodies = np.flatnonzero(vols > 0.0)
        cav = [k for k in np.flatnonzero(vols < 0.0)
               if any(_inside_shell(verts[faces[lab == k][0, 0]], verts, faces[lab == j]) for j in bodies)]
        cavity = np.isin(lab, cav) if cav else None
    tm = trimesh.Trimesh(verts, faces, process=False)
    tm.fix_normals()
    if tm.volume < 0:
        tm.invert()
    v, f = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64)
    if cavity is not None:   # the faces keep their order; each cavity shell turns inward again
        lab2 = _split_faces(f, len(v))
        for k in np.unique(lab2[cavity]):
            ids = lab2 == k
            sv = np.einsum("ij,ij->i", v[f[ids, 0]], np.cross(v[f[ids, 1]], v[f[ids, 2]])).sum()
            if sv > 0.0:
                f[ids] = f[ids][:, ::-1]
    return v, f


def _face_normals(verts, faces):
    """The normals of the faces, not scaled to length 1 (the cross products)."""
    return np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])


def _plane_tol(*meshes) -> float:
    """How far two planes of float32 STL files can differ and still be one plane (mm). It follows from the files, it is not a setting.

    Each coordinate is rounded to float32, that is half a step of `np.spacing` at the largest coordinate. The offset of a plane adds at
    most 3 such errors for one unit normal (sqrt(3) of them), in each of the two files: 2 x sqrt(3) steps. A box with round numbers gives 0.
    """
    top = max((float(np.abs(m.vertices).max()) for m in meshes if len(m.vertices)), default=0.0)
    return 2.0 * np.sqrt(3.0) * float(np.spacing(np.float32(top)))


def _plane_frame(n):
    """Two unit vectors `(u, w)` in the plane with the unit normal `n`; `u x w = n`."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def shared_planes(design, regions):
    """The planes that a face of the design space shares with a face of a frozen region, and where the region faces lie in each.

    `design`: the closed trimesh mesh of the design space. `regions`: the closed trimesh meshes of the fixed, load and solid regions.
    A plane `(n, d)` (unit outward normal `n`, the points with `n . x = d`) is shared when
    - the design space lies on its inner side (`n . x <= d` for every vertex of the design space: the plane bounds the design space,
      also a non-convex one), and a face of the design space lies in it, and
    - a face of a region lies in it too, with the same outward normal (all 3 vertices of the triangle are on the plane).
    "On the plane" is `_plane_tol`, the rounding of the float32 STL files, so a pair of faces that differ more than that is not found
    (a tilted pair from two CAD exports can be missed; the clip then does nothing, as before).
    Return a list of `(n, d, boxes)`: `n` a float64 array of 3, `boxes` a float64 array `(m, 4)` with one row `(u_min, u_max, w_min, w_max)`
    for each connected piece of the region faces in that plane, in the coordinates `x . u`, `x . w` of `_plane_frame(n)`.
    A mesh with a negative volume is read with its faces turned over.
    """
    import trimesh

    tol = _plane_tol(design, *regions)

    def outward(m):
        sgn = -1.0 if m.volume < 0 else 1.0
        return sgn * np.asarray(m.face_normals, np.float64), m.area_faces > 0

    nrm, ok = outward(design)
    tri = design.triangles[ok]
    n_all, d_all = nrm[ok], np.einsum("ij,ij->i", nrm[ok], tri[:, 0, :])
    # candidate planes of the design space: one per distinct (normal, offset) of its faces; the key has the resolution of the files
    key = np.column_stack([n_all.astype(np.float32), np.round(d_all / (tol if tol > 0 else 1.0))])
    _u, first = np.unique(key, axis=0, return_index=True)
    n_c, d_c = n_all[first], d_all[first]
    dv = np.asarray(design.vertices, np.float64)
    keep = np.zeros(len(n_c), bool)
    for lo in range(0, len(n_c), 256):                 # the design space must lie on the inner side of the plane
        sl = slice(lo, lo + 256)
        keep[sl] = (dv @ n_c[sl].T).max(0) <= d_c[sl] + tol
    n_c, d_c = n_c[keep], d_c[keep]
    pieces = [[] for _ in range(len(n_c))]             # for each plane: the (vertices, faces) of the region faces that lie in it
    for r in regions:
        if not len(n_c):
            break
        rn, rok = outward(r)
        rv = np.asarray(r.vertices, np.float64)
        rf = np.asarray(r.faces)[rok]
        rn = rn[rok]
        for lo in range(0, len(n_c), 64):
            sl = slice(lo, lo + 64)
            on = np.abs(rv @ n_c[sl].T - d_c[sl]) <= tol                    # (vertices, planes)
            tri_on = on[rf[:, 0]] & on[rf[:, 1]] & on[rf[:, 2]] & ((rn @ n_c[sl].T) > 0)
            for j in np.flatnonzero(tri_on.any(0)):
                pieces[lo + j].append((rv, rf[tri_on[:, j]]))
    out = []
    for k, parts in enumerate(pieces):
        if not parts:
            continue
        u, w = _plane_frame(n_c[k])
        boxes = []
        for rv, rf in parts:
            for piece in trimesh.Trimesh(rv, rf, process=False).split(only_watertight=False):   # the faces that join are one piece
                pu, pw = piece.vertices @ u, piece.vertices @ w
                boxes.append((pu.min(), pu.max(), pw.min(), pw.max()))
        out.append((n_c[k], float(d_c[k]), np.array(boxes, np.float64)))
    return out


def clip_to_planes(verts, faces, planes, tol: float, margin: float = 0.0, inside=None, dry: bool = False):
    """Cut the mesh at a shared plane that a part of it goes past over the face of a region, so that no part lies past that plane (manifold3d).

    `planes` as `shared_planes` returns them. A vertex is a reason to cut a plane when it lies past the plane by more than `tol`, over a face
    of a region (its place in the plane is in a box of the plane, grown by `margin`), and not in a region. `inside(points)` (optional) tells
    which points lie in a frozen region (inside it, or on its surface): a region may go past a plane in another part of its shape, and the
    exact union puts it back. The cut takes the whole mesh back to the plane (`trim_by_plane`), not only the part over the face, so the
    side face of the part stays flat. A plane that the mesh goes past only away from every region face is not cut and is in `left`.
    A mesh that has no reason to cut comes back as the same arrays. `dry=True` finds the facts and does not cut (the arrays come back,
    `volume_removed_mm3` is 0).
    Return `(verts, faces, facts)`. The facts have `planes` (the number of shared planes), `margin_mm`, `vertices_beyond` (the vertices on
    the outer side of the planes that were cut, by more than `tol` and not in a region), `largest_excess_mm`, `volume_removed_mm3`,
    `clipped` (one entry for each plane that was cut: `normal`, `offset_mm`, `vertices`, `over_face` the vertices over a region face,
    `largest_excess_mm`) and `left` (one entry for each plane that was not cut and that the mesh goes past: `normal`, `offset_mm`,
    `vertices`, `largest_excess_mm`).
    """
    facts = {"planes": len(planes), "margin_mm": float(margin), "vertices_beyond": 0, "largest_excess_mm": 0.0, "volume_removed_mm3": 0.0,
             "clipped": [], "left": []}
    if not planes or len(faces) == 0:
        return verts, faces, facts
    pts = np.asarray(verts, np.float64)
    excess = [pts @ n - d for n, d, _boxes in planes]
    past = [np.flatnonzero(e > tol) for e in excess]                           # beyond the plane by more than the rounding
    every = np.unique(np.concatenate(past)) if any(len(q) for q in past) else np.zeros(0, np.int64)
    in_region = np.zeros(len(pts), bool)
    if inside is not None and len(every):
        in_region[every] = np.asarray(inside(pts[every]), bool)                # material of a frozen region: the union keeps it
    hit = []
    for k, (n, d, boxes) in enumerate(planes):
        ids = past[k][~in_region[past[k]]]
        if not len(ids):
            continue
        u, w = _plane_frame(n)
        pu, pw = pts[ids] @ u, pts[ids] @ w
        over = np.zeros(len(ids), bool)
        for b in boxes:
            over |= (pu >= b[0] - margin) & (pu <= b[1] + margin) & (pw >= b[2] - margin) & (pw <= b[3] + margin)
        entry = {"normal": [round(float(x), 9) for x in n], "offset_mm": float(d), "vertices": int(len(ids)),
                 "largest_excess_mm": float(excess[k][ids].max())}
        if over.any():
            facts["clipped"].append({**entry, "over_face": int(over.sum())})
            hit.append(k)
        else:
            facts["left"].append(entry)
    if not hit:
        return verts, faces, facts
    facts["vertices_beyond"] = int(sum(c["vertices"] for c in facts["clipped"]))
    facts["largest_excess_mm"] = float(max(c["largest_excess_mm"] for c in facts["clipped"]))
    if dry:
        return verts, faces, facts
    v0 = abs(_signed_volume(verts, faces))
    man = _manifold_of(verts, faces)
    for k in hit:
        n, d, _boxes = planes[k]
        man = man.trim_by_plane(tuple(float(x) for x in -n), float(-d))        # keeps n . x <= d
    v, f = _from_manifold(man)
    if len(f) == 0:
        raise ValueError("the cut at the shared planes left no faces")
    facts["volume_removed_mm3"] = v0 - abs(_signed_volume(v, f))
    return v, f, facts


def make_envelope(design, regions, pieces=()):
    """The envelope of the part: the design space united with the frozen regions and the union pieces of the interfaces.

    `design`: the closed trimesh mesh of the design space. `regions`: the closed trimesh meshes of the fixed, load and solid regions.
    `pieces`: closed meshes of the union pieces of the interfaces (a boss, the wall of a tube, an STL union) at their as-printed size.
    Keep-out regions and the cut pieces of the interfaces (bores, holes) are not in the envelope: they take material away from the part,
    they do not widen the room that it may take.
    Return a dict: `man` (the manifold), `tm` (its closed trimesh mesh, for the distance), `design` (the design mesh), `face_of_tri`
    (for each triangle of the design: the index of its entry in `faces`; the last entry is the curved part), `faces` (the entries of the
    faces of the design space: `kind` `plane` with `normal` and `offset_mm`, and at the end one `curved` entry for the triangles that are
    in no plane facet), `tol` (`_plane_tol` of all the files) and the counts `regions` and `pieces`.
    A plane facet is a connected group of two or more coplanar triangles of the design mesh (`trimesh` `facets`).
    """
    import trimesh

    meshes = [design, *regions, *pieces]
    man = None
    for m in meshes:
        one = _load_manifold(m)
        man = one if man is None else man + one
    tm = trimesh.Trimesh(*_from_manifold(man), process=False)
    sgn = -1.0 if design.volume < 0 else 1.0
    nrm = sgn * np.asarray(design.face_normals, np.float64)
    tri = np.asarray(design.triangles, np.float64)
    face_of_tri = np.full(len(design.faces), -1, np.int64)
    entries = []
    for fac in design.facets:
        t0 = fac[int(np.argmax(design.area_faces[fac]))]          # the largest triangle of the facet gives the plane
        n = nrm[t0]
        face_of_tri[fac] = len(entries)
        entries.append({"kind": "plane", "normal": [round(float(x), 9) + 0.0 for x in n], "offset_mm": float(n @ tri[t0, 0])})
    face_of_tri[face_of_tri < 0] = len(entries)
    entries.append({"kind": "curved"})
    return {"man": man, "tm": tm, "design": design, "face_of_tri": face_of_tri, "faces": entries, "tol": _plane_tol(*meshes),
            "regions": len(regions), "pieces": len(pieces)}


def cut_to_envelope(verts, faces, env, dry: bool = False):
    """Cut the mesh to the envelope (an exact intersection, manifold3d), so that no part of it lies outside the room that the user gave.

    `env` as `make_envelope` returns it. A vertex is past the envelope when its distance outside the envelope is more than `env["tol"]`,
    the rounding of the float32 STL files (`_plane_tol`). A mesh with no vertex past is not touched (the same arrays). Otherwise the mesh
    is intersected with the envelope, the whole mesh and not only the vertices that are past: the cut face then lies on the faces of the
    envelope. `dry=True` counts the vertices that are past and does not cut.
    The facts come for each face of the design space: a vertex goes to the face with the closest point to it, and the volume of each
    piece that the cut takes away is split between the faces by the vertices of the piece (a piece at an edge counts for both faces; the
    parts add up to the volume that was removed).
    Return `(verts, faces, facts)`. The facts have `cut` (true if the mesh was cut), `vertices_beyond`, `largest_excess_mm` (before the
    cut), `volume_removed_mm3`, `tol_mm`, `faces` (one entry for each face of the design space with a vertex past: `kind` `plane` with
    `normal` and `offset_mm`, or `curved`; `vertices`, `largest_excess_mm`, `volume_removed_mm3`) and, when no cut was made, `reason`.
    `volume_removed_pct` is `volume_removed_mm3` in percent of the volume of the mesh before the cut.
    """
    import trimesh

    facts = {"cut": False, "vertices_beyond": 0, "largest_excess_mm": 0.0, "volume_removed_mm3": 0.0, "volume_removed_pct": 0.0,
             "tol_mm": float(env["tol"]), "faces": []}
    if len(faces) == 0:
        facts["reason"] = "the mesh has no faces"
        return verts, faces, facts
    pts = np.asarray(verts, np.float64)
    # The envelope holds the design space, so a vertex is past it by no more than it is outside the design space: the distance to the (small)
    # design mesh finds the vertices to look at, and only they get the distance to the envelope.
    near = np.flatnonzero(_query(trimesh.proximity.signed_distance, env["design"], pts) < -env["tol"])
    excess = np.zeros(len(pts))
    if len(near):
        excess[near] = np.maximum(-_query(trimesh.proximity.signed_distance, env["tm"], pts[near]), 0.0)
    past = np.flatnonzero(excess > env["tol"])
    if not len(past):
        facts["reason"] = "no vertex of the mesh lies past the envelope by more than the rounding of the STL files"
        return verts, faces, facts
    facts["vertices_beyond"] = int(len(past))
    facts["largest_excess_mm"] = float(excess[past].max())
    if dry:
        facts["reason"] = "the first pass of the volume calibration only looks"
        return verts, faces, facts
    n_face = len(env["faces"])

    def face_ids(points):
        _p, _d, tri = _query(trimesh.proximity.closest_point, env["design"], points)
        return env["face_of_tri"][tri]

    fid = face_ids(pts[past])
    man = _manifold_of(verts, faces)
    cut = man ^ env["man"]
    v, f = _from_manifold(cut)
    if len(f) == 0:
        raise ValueError("the cut to the envelope left no faces")
    vol = np.zeros(n_face)
    pieces = (man - env["man"]).decompose()               # the pieces that the cut takes away; each goes to the faces by its vertices
    if pieces:
        pv = [np.asarray(pc.to_mesh().vert_properties, np.float64)[:, :3] for pc in pieces]
        votes = np.zeros((len(pieces), n_face))
        np.add.at(votes, (np.repeat(np.arange(len(pieces)), [len(x) for x in pv]), face_ids(np.vstack(pv))), 1.0)
        vol = (np.array([float(pc.volume()) for pc in pieces])[:, None] * votes / votes.sum(axis=1, keepdims=True)).sum(axis=0)
    for k in np.unique(fid):
        sel = past[fid == k]
        facts["faces"].append({**env["faces"][k], "vertices": int(len(sel)), "largest_excess_mm": float(excess[sel].max()),
                               "volume_removed_mm3": float(vol[k])})
    facts["faces"].sort(key=lambda e: (-e["largest_excess_mm"], e["kind"]))
    facts["cut"] = True
    v_before = float(man.volume())
    facts["volume_removed_mm3"] = float(v_before - cut.volume())
    facts["volume_removed_pct"] = 100.0 * facts["volume_removed_mm3"] / v_before if v_before > 0 else 0.0
    return v, f, facts


def snap_into_regions(verts, faces, surfaces, band_mm: float):
    """Move the vertices that lie just outside a region into the region, so that the exact boolean removes them.

    `surfaces`: closed trimesh meshes, the exact surface of the union of the `fixed` and `load` STL files, and of the union of the
    `keep_out` STL files. The smooth surface lies micrometres to hundredths of a millimetre off these faces (the distance field `U` is
    not exact, see `stl_sdf`). The union or the cut of two surfaces that are almost coplanar then leaves a strip of micro-steps and slivers.

    A vertex is moved when its distance to the nearest surface is at most `band_mm` and it lies OUTSIDE the solid of that surface. The
    side comes from `trimesh.proximity.signed_distance`: the normal of the closest face where the closest point lies in that face (the
    vector to the vertex is then parallel to the normal), and a ray test against the whole solid where it lies on an edge or at a corner
    (a reflex edge, for example the tip of a notch narrower than 90 degrees, has no side by the normal of one face). A vertex within
    1e-8 mm (`trimesh.tol.merge`) of the surface counts as on it and stays; so does a vertex inside a region and a vertex farther than
    the band. A moved vertex goes to the point one band deep inside: the closest point plus `band_mm` along the direction from the vertex
    through that point. When that point is not inside the solid (a region thinner than one band) the vertex stays. A move that would turn
    a face over (the dot product of its normal before and after is not above 0) is halved, for the vertices of the faces that turn, until
    no face turns over; a move of 0 turns no face. A closest point that is not finite gives no move.
    The limit: where a region is thinner than two bands, the vertices from its two sides can cross each other inside it before the union.

    Return `(new float64 vertices, facts)`; the facts are the counts and the largest move, in mm.
    """
    import trimesh

    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    n = len(verts)
    best_d = np.full(n, np.inf)
    best_p = verts.copy()
    best_s = np.full(n, -1)
    for k, tm in enumerate(surfaces):
        lo, hi = tm.bounds[0] - band_mm, tm.bounds[1] + band_mm
        cand = np.flatnonzero(((verts >= lo) & (verts <= hi)).all(1))
        if not len(cand):
            continue
        p, d, _tri = _query(trimesh.proximity.closest_point, tm, verts[cand])
        better = d < best_d[cand]
        idx = cand[better]
        best_d[idx], best_p[idx], best_s[idx] = d[better], p[better], k
    in_band = best_d <= band_mm
    to_closest = best_p - verts                                    # from the vertex to its closest point
    length = np.linalg.norm(to_closest, axis=1)
    cand = in_band & np.isfinite(best_p).all(1) & (length > 0)
    outside = np.zeros(n, bool)
    for k, tm in enumerate(surfaces):
        ids = np.flatnonzero(cand & (best_s == k))
        if len(ids):
            outside[ids] = _query(trimesh.proximity.signed_distance, tm, verts[ids]) < 0
    ids = np.flatnonzero(outside)
    target = best_p[ids] + band_mm * (to_closest[ids] / length[ids, None])
    deep = np.zeros(len(ids), bool)
    for k, tm in enumerate(surfaces):
        sel = best_s[ids] == k
        if sel.any():
            deep[sel] = _query(trimesh.proximity.signed_distance, tm, target[sel]) >= 0
    target_outside = int((~deep).sum())
    ids, target = ids[deep], target[deep]
    new, turned, rounds = _move_keeping_faces(verts, faces, ids, target)
    moved = np.linalg.norm(new - verts, axis=1)
    facts = {"moved": int((moved > 0).sum()), "in_band": int(in_band.sum()), "of": int(n), "target_outside": target_outside,
             "largest_move_mm": float(moved.max()) if n else 0.0, "turned_faces": turned,
             "halving_rounds": rounds, "band_mm": float(band_mm), "depth_mm": float(band_mm)}
    return new, facts


def _move_keeping_faces(verts, faces, ids, target):
    """Move the vertices `ids` to `target`. A move that would turn a face over (the dot product of its normal before and after is not
    above 0) is halved, for the vertices of the faces that turn, until no face turns over; a move of 0 turns no face.
    Return `(new vertices, the faces that turned in some round, the halving rounds)`."""
    n = len(verts)
    near = np.zeros(n, bool)
    near[ids] = True
    new = verts.copy()
    new[ids] = target
    turned_ever = np.zeros(len(faces), bool)
    rounds = 0
    if near.any():
        n0 = _face_normals(verts, faces)
        real = np.einsum("ij,ij->i", n0, n0) > 0                   # a face of no area cannot turn
        while True:
            turned = real & (np.einsum("ij,ij->i", n0, _face_normals(new, faces)) <= 0)
            mv = np.zeros(n, bool)
            mv[faces[turned].ravel()] = True
            mv &= near
            if not mv.any():
                break
            turned_ever |= turned
            rounds += 1
            half = verts[mv] + 0.5 * (new[mv] - verts[mv])
            # a halving that changes nothing (the move is at the last digit) gives the vertex back: the loop always ends
            new[mv] = np.where((half == new[mv]).all(1)[:, None], verts[mv], half)
    return new, int(turned_ever.sum()), int(rounds)


def flush_to_planes(verts, faces, planes, band_mm: float, on_envelope, inside=None):
    """Move each vertex that lies inside a shared plane by less than `band_mm` onto the plane (#319, with the limit of the field).

    `planes` as `shared_planes` returns them: a face of the design space and a face of a region lie in each. With the limit of the field
    no part of the smooth surface stands past such a plane, so the cut at the planes (`clip_to_planes`), which made the side of the part
    flush with the plane before, does nothing. A surface that ends a hair inside the plane then meets the region face in that plane with a
    strip narrower than a tenth of a millimetre after the exact union, and the surface preparation of `topopt verify` folds such a strip.
    A vertex moves to its foot on the nearest such plane (the deepest that is within the band) only where `on_envelope(foot)` is true
    (the foot lies on the envelope, not past it); a move that would turn a face over is halved (`_move_keeping_faces`). `inside(points)`
    (optional) tells which points lie in a frozen region (inside it, or on its surface): such a vertex stays where it is. The exact union
    keeps the region there anyway, and a member that goes into a region through such a plane must not be pulled back onto the plane: a
    loose body then touched the region face only in that plane and stayed in the file as a second body.
    Return `(new float64 vertices, facts)`: `vertices` (moved), `largest_move_mm`, `turned_faces`, `halving_rounds`, `band_mm`.
    """
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    depth = np.full(len(verts), np.inf)
    target = verts.copy()
    for nrm, d, _boxes in planes:
        e = verts @ nrm - d                                        # below 0: inside the plane
        sel = (e < 0) & (-e < band_mm) & (-e < depth)
        target[sel] = verts[sel] - e[sel, None] * nrm
        depth[sel] = -e[sel]
    ids = np.flatnonzero(np.isfinite(depth))
    if len(ids):
        ids = ids[np.asarray(on_envelope(target[ids]), bool)]
    if len(ids) and inside is not None:
        ids = ids[~np.asarray(inside(verts[ids]), bool)]
    new, turned, rounds = _move_keeping_faces(verts, faces, ids, target[ids])
    moved = np.linalg.norm(new - verts, axis=1)
    return new, {"vertices": int((moved > 0).sum()), "largest_move_mm": float(moved.max()) if len(moved) else 0.0,
                 "turned_faces": turned, "halving_rounds": rounds, "band_mm": float(band_mm)}


# -- main entry -------------------------------------------------------------------------------
def _cleanup_float32(verts, faces, tol: float, log_skip: bool = True):
    """Merge near-coincident float32 vertices (manifold3d simplify) so the STL reload does not pinch the surface.

    Input and output are float32-exact. Return (verts, faces, facts). On any problem, the mesh is returned unchanged, and the facts have
    `error` and `error_type`. The volume change is in the facts. See pitfall `float32-stl-reload-pinch`.
    `log_skip=False`: the caller writes the warning line itself (the mesh step, with its round: #445).
    """
    try:
        man = _manifold_of(verts, faces)
        v0 = abs(float(man.volume()))
        v, f = _from_manifold(man.simplify(tol))
        v, f = _orient(v, f)
        v = v.astype(np.float32).astype(np.float64)
        return v, f, {"tol_mm": tol, "triangles_before": int(len(faces)), "triangles_after": int(len(f)),
                      "volume_change_mm3": abs(_signed_volume(v, f)) - v0}
    except Exception as exc:
        reraise_if_resource(exc, "the float32 clean-up (simplify)")
        if log_skip:
            log.warning("float32 clean-up (simplify) skipped: %s", exc)
        return verts, faces, {"tol_mm": tol, "error": str(exc), "error_type": type(exc).__name__}


def deep_loss(S, verts, faces, origin_f, pitch: float, depth: float = 1.5) -> dict:
    """Local loss against the sharp field `S` (bool, fine grid): sharp voxels more than `depth` fine voxels outside the mesh.

    The volume error of the step is a calibration residual and can hide a lost member. This measure cannot be paid
    for by a shift elsewhere. The mesh is sampled at one generic offset in the voxel; the depth margin absorbs the noise.
    """
    import trimesh
    from scipy import ndimage as ndi

    from .voxelize import inside_mask

    mesh = trimesh.Trimesh(verts, faces, process=False)
    M = inside_mask(mesh, origin_f + pitch * np.array([0.23, 0.29, 0.31]), pitch, S.shape)
    M |= inside_mask(mesh, origin_f + pitch * np.array([-0.19, -0.27, -0.13]), pitch, S.shape)
    d = ndi.distance_transform_edt(~M)   # in fine voxels
    deep = S & ~M & (d > depth)
    lab, n = ndi.label(deep)
    sizes = np.bincount(lab.ravel())[1:] * pitch ** 3 if n else np.array([0.0])
    return {"deep_lost_mm3": round(float(deep.sum()) * pitch ** 3, 3), "parts": int(n),
            "largest_mm3": round(float(sizes.max()), 3), "depth_fine_voxels": depth,
            "sharp_mm3": round(float(S.sum()) * pitch ** 3, 1)}


def thresholded_volume(rho, labels, up: int, pad: int) -> float:
    """Volume of the sharp field (density cut at 0.5, islands removed) in coarse voxel units. Multiply by h^3 for mm^3."""
    from . import post

    fine = post._upsample(rho, up)
    fixed_fine = post._upsample(((labels == post.FIXED) | (labels == post.LOAD)).astype(np.float32), up) > 0.5
    fine = post._keep_fixed_components(fine, fixed_fine, 0.5)
    return float((fine > 0.5).sum()) / up ** 3


def grow_to_surface(S, lock_out, design, origin_f, pitch: float, h: float, p: SdfParams):
    """Grow the sharp field `S` to the surface of the design space (the "envelope surface" step). Return (new S, facts).

    `d` is the signed distance of the closed design mesh (negative inside). A candidate is a fine voxel with `-surface_depth * h < d <
    surface_overshoot * pitch` that is not in `S` and not in a keep-out region. It is grown when `S` holds the nearest voxel of `v - k * n`
    for some k = 1 .. K (`n` is the outward unit gradient of `d`, K = floor((depth * h + overshoot * pitch) / pitch)). A candidate with no gradient
    is not grown. The facts: `grown_inside_mm3` (the grown voxels with d <= 0), `grown_overshoot_mm3` (the grown voxels past the surface).
    """
    depth, over = p.surface_depth * h, p.surface_overshoot * pitch
    band = int(np.ceil(depth / pitch)) + int(np.ceil(p.surface_overshoot)) + 2
    d = stl_sdf(design, origin_f, pitch, S.shape, band)
    cand = (d > -depth) & (d < over) & ~lock_out & ~S
    ci = np.argwhere(cand)
    del cand
    sh = np.array(S.shape)

    def at(off):
        j = np.clip(ci + off, 0, sh - 1)
        return d[j[:, 0], j[:, 1], j[:, 2]]

    g = np.stack([at(e) - at(-e) for e in np.eye(3, dtype=int)], axis=1).astype(np.float64)
    gn = np.linalg.norm(g, axis=1)
    has_n = gn > 1e-9
    g[has_n] /= gn[has_n, None]
    K = int(np.floor((depth + over) / pitch + 1e-9))
    ok = np.zeros(len(ci), bool)
    for k in range(1, K + 1):
        j = np.rint(ci - k * g).astype(np.int64)
        inb = ((j >= 0) & (j < sh)).all(axis=1)
        jj = np.clip(j, 0, sh - 1)
        ok |= inb & S[jj[:, 0], jj[:, 1], jj[:, 2]]
    ok &= has_n
    gi = ci[ok]
    out = S.copy()
    out[tuple(gi.T)] = True
    # A grown voxel that does not join the solid (a group in a valley, with a gap to the solid) would become a loose body: drop it.
    from scipy import ndimage as ndi
    comp, _n = ndi.label(out)
    keep = np.unique(comp[S])
    loose = ~np.isin(comp[tuple(gi.T)], keep[keep > 0])
    out[tuple(gi[loose].T)] = False
    gi = gi[~loose]
    inside = d[gi[:, 0], gi[:, 1], gi[:, 2]] <= 0
    vox = pitch ** 3
    return out, {"ran": True, "candidates": int(len(ci)), "grown": int(len(gi)), "loose_dropped": int(loose.sum()),
                 "grown_inside_mm3": float(inside.sum()) * vox, "grown_overshoot_mm3": float((~inside).sum()) * vox}


def make_mesh_sdf(rho, labels, origin, h, *, up: int, pad: int, fixed_stls=(), load_stls=(),
                  keep_out_stls=(), params: SdfParams | None = None, max_voxels: int | None = None,
                  extra_frozen=(), extra_cut=(), design_stl=None, envelope_extra=None):
    """SDF-smoothed watertight mesh. Return (verts float64, faces int64, info).

    `fixed_stls`, `load_stls`, `keep_out_stls`: lists of paths (or None for a missing file).
    A group uses its STL files only if all of them exist; else it uses the voxel labels.
    `design_stl`: the path of the design space STL. With it (and the STL files of the frozen regions), the smooth mesh is cut at each
    plane that the design space shares with a frozen region and that a part of the mesh stands past over a face of that region
    (`shared_planes`, `clip_to_planes`), so that no part of it goes past the plane. `info["plane_clip"]` has the facts.
    The mesh is then cut to the envelope (`make_envelope`, `cut_to_envelope`): the design space, the STL files of the frozen regions and
    `envelope_extra` (closed meshes of the union pieces of the interfaces at their as-printed size; the default is `extra_frozen`, the same
    pieces at the size that the field has). `info["envelope_cut"]` has the facts.
    Where the envelope is known, the field is first limited to it (`SdfParams.envelope_limit`, on by default; #319): the local volume
    window counts only the solid inside it, the blend takes the maximum with `envelope_sdf`, Taubin pins the vertices on it, a vertex
    inside a shared plane by less than the band moves onto the plane (`flush_to_planes`), and the calibration runs once with the cuts.
    Without the design space file, or without STL files for the frozen regions, the envelope is not known, and the field is not limited
    and the mesh is not cut.
    `extra_frozen`, `extra_cut`: closed meshes in memory (interface shapes, CONTRACTS.md section 23). They add exact
    frozen and cut shapes to the SDFs, so the smooth surface joins them. The exact booleans for them are not made here:
    the interface step of the post-process does them after the smoothing.
    `info["snap"]` has the facts of the move of the vertices to the region faces (`snap_into_regions`), or is None when no region
    uses an STL file (no exact boolean, so no seam). Without the move, `{"error": ...}` says why the mesh was not moved. When the fact
    `volume_change_mm3` is missing, `volume_change_error` says why.
    Raises `SdfGridTooLarge` when the fine grid is over the guard.
    """
    import trimesh
    from scipy import ndimage as ndi
    from scipy.ndimage import map_coordinates

    from . import post

    p = params or SdfParams()
    t0 = time.monotonic()
    tm_ = {}
    pitch = h / up
    shape_f = tuple((int(n) + 2 * pad) * up for n in np.shape(rho))   # the shape of the padded fine grid
    fact = grid_fact(shape_f, h, max_voxels, triangles=surface_triangles(rho, up))
    if not fact["ok"]:
        raise SdfGridTooLarge(fact)
    fine = post._upsample(rho, up)
    origin = np.asarray(origin, np.float64)
    origin_f = origin - pad * h   # world position of the corner of fine voxel (0, 0, 0)
    sigma = p.sigma * h

    fixed_lab = post._upsample(((labels == post.FIXED) | (labels == post.LOAD)).astype(np.float32), up) > 0.5
    hole_lab = post._upsample((labels == post.KEEP_OUT).astype(np.float32), up) > 0.5
    S = fine > 0.5
    del fine

    # --- frozen (fixed + load) and cut (keep-out) solids ---
    def usable(paths):
        paths = [Path(q) if q is not None else None for q in paths]
        return bool(paths) and all(q is not None and q.is_file() for q in paths), paths

    stl_f, paths_f = usable(list(fixed_stls) + list(load_stls))
    if stl_f and not load_stls and (labels == post.LOAD).any():
        stl_f = False   # load voxels exist but no load STL: the labels give the frozen shape (they cannot be split)
    stl_c, paths_c = usable(list(keep_out_stls))
    for what, ok, ps in (("fixed/load", stl_f, paths_f), ("keep-out", stl_c, paths_c)):
        miss = [str(q) for q in ps if q is None or not Path(q).is_file()]
        if miss:
            log.warning("%s STL missing, the voxel labels give the shape of these regions: %s", what, ", ".join(miss))
    extra_f, extra_c = list(extra_frozen), list(extra_cut)
    shapes_c = bool(extra_c) and not keep_out_stls   # all keep-out voxels come from interface cuts: no label staircase
    src = {"frozen": "stl" if stl_f else "labels",
           "keep_out": "stl" if stl_c else ("shapes" if shapes_c else ("labels" if hole_lab.any() else "none"))}
    if extra_f or extra_c:
        src["interfaces"] = {"frozen": len(extra_f), "keep_out": len(extra_c)}
    U = None
    if stl_f:
        for q in paths_f:
            m = trimesh.load(q, force="mesh")
            d = stl_sdf(m, origin_f, pitch, shape_f, p.band)
            U = d if U is None else np.minimum(U, d)
            del d
        S &= ~fixed_lab
    else:
        U = binary_sdf(fixed_lab, pitch)
    for m in extra_f:
        d = stl_sdf(m, origin_f, pitch, shape_f, p.band)
        U = np.minimum(U, d)
        del d
    Cs = None
    if stl_c:
        for q in paths_c:
            m = trimesh.load(q, force="mesh")
            d = stl_sdf(m, origin_f, pitch, shape_f, p.band)
            Cs = d if Cs is None else np.minimum(Cs, d)
            del d
    elif hole_lab.any() and not shapes_c:
        Cs = binary_sdf(hole_lab, pitch)
    for m in extra_c:
        d = stl_sdf(m, origin_f, pitch, shape_f, p.band)
        Cs = d if Cs is None else np.minimum(Cs, d)
        del d
    lock_in = U < 0
    lock_out = (Cs < 0) if Cs is not None else np.zeros(shape_f, bool)
    lock_in &= ~lock_out
    S = (S | lock_in) & ~lock_out
    del fixed_lab, hole_lab
    # islands that do not touch the frozen solid are removed
    comp, n = ndi.label(S)
    removed_islands = 0
    if n > 1:
        touch = np.unique(comp[lock_in & S])
        touch = touch[touch > 0]
        if len(touch) == 0:
            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            touch = np.array([int(np.argmax(sizes))])
        drop = S & ~np.isin(comp, touch)
        removed_islands = n - len(touch)
        S &= ~drop
    del comp
    if not S.any():
        raise ValueError("the final field has no solid voxels")
    # The envelope: the design space, the frozen regions and the union pieces of the interfaces (as printed). The sharp field grows to its surface
    # (below), and the smooth mesh is cut to it in `build`
    # (an exact intersection), after the cut at the shared planes. It needs the STL of the design space and the STL files of the frozen regions.
    envelope, env_note, env_warn, design_e = None, None, False, None
    if not stl_f:
        env_note = "the fixed, load and solid regions have no STL file (the voxel labels give their shape), so the envelope is not known and the mesh is not cut to it"
    elif design_stl is None:
        env_note = "the run names no design space STL, so the envelope is not known and the mesh is not cut to it"
    else:
        try:
            if not Path(design_stl).is_file():
                env_note, env_warn = f"the design space STL file is not there, so the mesh is not cut to the envelope: {design_stl}", True
            else:
                design_e = trimesh.load(design_stl, force="mesh")
                if not design_e.is_watertight:   # the inside and the outside are known only for a closed mesh
                    env_note, env_warn = f"the design space STL is not a closed mesh, so the mesh is not cut to the envelope: {design_stl}", True
                else:
                    envelope = make_envelope(design_e, [trimesh.load(q, force="mesh") for q in paths_f],
                                             list(extra_f if envelope_extra is None else envelope_extra))
        except Exception as exc:   # the mesh is then not cut, as before
            reraise_if_resource(exc, "the envelope of the design space")
            env_note, env_warn = f"the envelope was not made: {type(exc).__name__}: {exc}", True
            envelope = None
    if env_warn:
        log.warning("%s", env_note)


    # --- the envelope surface: fill the staircase of the design-space surface ---
    # A face of the design space that is slanted to the grid axes gives a staircase in the density field. The low-pass keeps the steps, and the cut to
    # the envelope makes only the top of each step flat. The sharp field grows into the valleys: a fine voxel near the surface is added when the sharp
    # field holds a voxel on the inward normal below it (so nothing grows sideways into a window). The volume target stays the volume before
    # the growth, so the mass does not change.
    surf = {"ran": False}
    v_target = float(S.sum()) * pitch ** 3
    if p.envelope_fill == "off":
        surf["reason"] = "settings.envelope_fill is off"
    elif envelope is None:
        surf["reason"] = env_note
    else:
        try:
            S_grown, surf = grow_to_surface(S, lock_out, design_e, origin_f, pitch, h, p)
            share = surf["grown_inside_mm3"] / v_target if v_target > 0 else 0.0
            if share > p.fill_max_share and p.envelope_fill != "on":   # the payment of the growth by the mass-neutral calibration would thin the members
                surf = {"ran": False, "grown_inside_mm3": surf["grown_inside_mm3"],
                        "reason": f"the growth would add {100 * share:.1f} % of the volume target inside the design space, above the limit "
                                  f"{100 * p.fill_max_share:.1f} % (SdfParams.fill_max_share), so the field is not grown. Option: settings.envelope_fill: on"}
                log.warning("envelope fill skipped: %s", surf["reason"])
            else:
                S = S_grown
                surf["note"] = FILL_COST_NOTE
                if share <= p.fill_max_share:   # one warning line when the fill ran, with the cost
                    log.warning("envelope fill ran: it adds %.1f %% of the volume target inside the design space. %s", 100 * share, FILL_COST_NOTE)
                else:   # `on` runs the fill above the cap; the fact says so, and the one warning line gives the limit and the cost
                    surf["cap_overridden"] = True
                    log.warning("envelope fill runs above the limit: it adds %.1f %% of the volume target inside the design space, the limit is %.1f %% "
                                "(settings.envelope_fill: on). %s", 100 * share, 100 * p.fill_max_share, FILL_COST_NOTE)
        except Exception as exc:   # the field stays as it is
            reraise_if_resource(exc, "the growth to the design space surface")
            surf = {"ran": False, "reason": f"the growth to the design space surface failed: {type(exc).__name__}: {exc}"}
            log.warning("%s", surf["reason"])
    surf["target_mm3"] = v_target
    if "grown_inside_mm3" in surf:
        surf["grown_share_pct"] = 100.0 * surf["grown_inside_mm3"] / v_target if v_target > 0 else 0.0
    v_thr = v_target
    # The closed cavities of the sharp field: their walls stay in the mesh as voids (`_drop_voids` with `keep`), so a void of the design
    # is not filled, and the calibration does not thin the part to pay for it (the target, the sharp volume, does not hold the void either).
    cavities = closed_cavities(S, origin_f, pitch)
    tm_["fields"] = time.monotonic() - t0

    # --- the limit of the field to the envelope (#319) ---
    # The voxel layer at a face of the design space reaches up to half a voxel past it (a design voxel counts when its centre is inside),
    # and the low-pass and the level shift add to that. The field is limited to the envelope before marching cubes, so the surface does not
    # pass it, and the local volume window of `smooth_vp` counts only the solid inside it: the level near the face rises and the member that
    # lost its outer layer gets the volume back there, not as a thickening of the whole part.
    Env, limit_note, t_l = None, None, time.monotonic()
    if envelope is None:
        limit_note = "the envelope is not known (see `reason`), so the field is not limited to it"
    elif not p.envelope_limit:
        limit_note = "SdfParams.envelope_limit is off: the cuts take what is past the envelope"
    else:
        try:
            Env = envelope_sdf(envelope["tm"], origin_f, pitch, shape_f, p.band)
        except Exception as exc:   # the mesh is then cut to the envelope only, as before
            reraise_if_resource(exc, "the limit of the field to the envelope")
            limit_note = f"the field was not limited to the envelope, the cuts take what is past it: {type(exc).__name__}: {exc}"
            log.warning("%s", limit_note)
    t_l = time.monotonic() - t_l

    # --- smoothing ---
    t1 = time.monotonic()
    phi0 = binary_sdf(S, pitch)
    F3, cs = smooth_vp(phi0, S, sigma, pitch, p, lock_in, lock_out, room=(Env < 0) if Env is not None else None)
    del phi0, cs
    cutf = -Cs if Cs is not None else None
    del Cs, lock_in, lock_out
    gc.collect()

    def frozen_near(verts_world):
        """True where a vertex is on or inside a frozen region (U < pin_tol), by trilinear sampling of U."""
        idx = (np.asarray(verts_world) - origin_f) / pitch - 0.5
        return map_coordinates(U, idx.T, order=1, mode="nearest") < p.pin_tol * pitch

    def mc_at(delta):
        F = np.minimum(F3 - np.float32(delta), U)
        if Env is not None:
            F = np.maximum(F, Env)
        if cutf is not None:
            F = np.maximum(F, cutf)
        v, f = _march(F)
        return v, f, abs(_signed_volume(v * pitch, f))

    def secant(delta, vol, target, hist, steps):
        """Move the global level shift so that the marching-cubes volume meets `target`.

        Return (delta, vol, stop). `stop` is None, or a text with the fact that ended the steps before the volume met
        the target: the target is not above 0 (the relative error needs it), or the slope is 0 or not finite (the
        volume does not follow the shift, for example when the frozen and cut shapes hold the whole surface).
        """
        for _ in range(steps):
            if target <= 0.0:
                return delta, vol, f"the marching-cubes volume target of this round is {target:.3f} mm3, so there is no relative error to meet"
            if abs(vol - target) / target < 2e-4:
                break
            if len(hist) < 2:
                slope = -float(trimesh.Trimesh(v_last[0] * pitch, v_last[1], process=False).area)
            else:
                (d0, v0), (d1, v1) = hist[-2], hist[-1]
                slope = (v1 - v0) / (d1 - d0) if d1 != d0 else -1.0
            step = (target - vol) / slope if slope != 0.0 else float("nan")
            if not np.isfinite(step):
                return delta, vol, (f"the mesh volume stays {vol:.3f} mm3 when the level shift moves (slope {slope + 0.0:g} mm2), "
                                    f"so the shift {delta:.4f} mm cannot reach the marching-cubes volume target {target:.3f} mm3 of this round")
            delta = delta + step
            v, f, vol = mc_at(delta)
            v_last[:] = [v, f]
            hist.append((delta, vol))
        return delta, vol, None

    def union_of(meshes):
        out = None
        for m in meshes:
            man = _load_manifold(m)
            out = man if out is None else out + man
        return out

    # The interface shapes (section 23) are united and cut after this step. The calibration measures the volume that the
    # mesh will have with them (the label shapes), else a cut that lies over a united STL region (the exact union puts the
    # region back) would count as extra material, and the shift would thin all members to pay for it.
    F_man, C_man = (union_of(extra_f), union_of(extra_c)) if (extra_f or extra_c) else (None, None)

    # The exact surfaces of the STL regions for the snap (`snap_into_regions`): the union of the fixed and load files, and the union of the
    # keep-out files. They are made once, because the calibration builds the mesh several times. If they cannot be made, the mesh is not moved.
    snap_surfaces, snap_error = [], None
    if stl_f or stl_c:
        try:
            snap_surfaces = [trimesh.Trimesh(*_from_manifold(union_of(ps)), process=False)
                             for ps, on in ((paths_f, stl_f), (paths_c, stl_c)) if on]
        except Exception as exc:
            reraise_if_resource(exc, "the exact surfaces of the regions")
            snap_error = f"the exact surfaces of the regions were not made: {type(exc).__name__}: {exc}"
            log.warning("%s; the smooth mesh is not moved to the region faces", snap_error)

    # The planes that the design space shares with a frozen region. The smooth surface can go past such a plane (the staircase of the
    # frozen voxels, the low-pass and the level shift together reach more than the band of the snap), and the exact union with the region
    # does not take that material away. The smooth mesh is cut at these planes in `build`. Only an STL region has an exact plane.
    clip_planes, clip_tol, clip_note = [], 0.0, None
    if stl_f and design_stl is not None:
        try:
            if not Path(design_stl).is_file():
                clip_note = f"the design space STL file is not there, so no plane of the design space was looked for: {design_stl}"
                log.warning(clip_note)
            else:
                design_m = trimesh.load(design_stl, force="mesh")
                if not design_m.is_watertight:   # the side of a face (outward or inward) is known only for a closed mesh
                    clip_note = f"the design space STL is not a closed mesh, so no plane of it was looked for: {design_stl}"
                    log.warning(clip_note)
                else:
                    region_m = [trimesh.load(q, force="mesh") for q in paths_f]
                    clip_planes = shared_planes(design_m, region_m)
                    clip_tol = _plane_tol(design_m, *region_m)
        except Exception as exc:   # the mesh is then not cut, as before
            reraise_if_resource(exc, "the planes shared with the regions")
            clip_note = f"the planes that the design space shares with the regions were not made: {type(exc).__name__}: {exc}"
            log.warning("%s; the smooth mesh is not cut at them", clip_note)
            clip_planes = []

    # An interface shape that is united (boss, tube, stl; section 23) can go past such a plane: it is part of the field, the smooth surface joins it,
    # and the interface step unites its exact shape after this one. The plane is not cut then (the cut would take away the join).
    clip_found, clip_skipped = len(clip_planes), 0
    if clip_planes and extra_f:
        ev = np.vstack([np.asarray(m.vertices, np.float64) for m in extra_f])
        free = [q for q in clip_planes if float((ev @ q[0] - q[1]).max()) <= clip_tol]
        clip_skipped, clip_planes = len(clip_planes) - len(free), free

    def in_frozen(points):
        """True where a point lies in a fixed, load or solid region or on its surface (the exact surface of the union of the STL files)."""
        surf = snap_surfaces[0] if snap_surfaces else trimesh.Trimesh(*_from_manifold(union_of(paths_f)), process=False)
        return _query(trimesh.proximity.signed_distance, surf, points) >= -clip_tol

    def on_envelope(points):
        """True where a point lies on the envelope or inside it (the SDF of the limit at the point, within the nudge of `envelope_sdf`)."""
        return map_coordinates(Env, ((points - origin_f) / pitch - 0.5).T, order=1, mode="nearest") <= 2e-3 * pitch

    # The fallbacks of `build`: each one is in this list (round, step, error type, error text) and in the log, so that a fallback in a round
    # that is not kept leaves a fact too (#445). `where` says which build runs: the pass, the round, and the build without the move.
    fallbacks: list = []
    where = {"pass": 0, "round": None, "build": "calibration"}

    def note_fallback(step, exc, outcome, error_type=None):
        """Record the fallback of a step of `build`: one entry in `fallbacks` and one warning line. `outcome` says what the mesh keeps.
        `exc` is the exception, or the text of an error that a helper caught (then `error_type` names its type)."""
        entry = {"pass": where["pass"], "round": where["round"], "build": where["build"], "step": step,
                 "error_type": error_type or type(exc).__name__, "error": str(exc)}
        fallbacks.append(entry)
        place = "the build without the move" if where["build"] == "without_move" else f"round {where['round']} of pass {where['pass']}"
        log.warning("mesh fallback in %s, step %s: %s: %s; %s", place, step, entry["error_type"], entry["error"], outcome)

    def design_void(verts_, shell_faces):
        """True if the void shell encloses a closed cavity of the sharp field: a void of the design, not one that the snap and the union made."""
        pts = cavities["points"]
        box = verts_[np.unique(shell_faces)]
        sel = pts[np.all((pts >= box.min(0)) & (pts <= box.max(0)), axis=1)]
        return any(_inside_shell(q, verts_, shell_faces) for q in sel)

    def build(v, f, snap=True, cut=True):
        """Taubin with pinned vertices, then the snap, the cut at the shared planes, the cut to the envelope and the exact boolean steps.
        Return (verts, faces, facts).

        `snap=False` builds the same mesh without the move of the vertices to the region faces (for the fact `volume_change_mm3`).
        `cut=False` looks for the planes to cut and for the vertices past the envelope and does not cut (the facts say what the cuts would take).
        """
        steps = {}
        snapf = None
        pin = None
        if p.taubin_iterations > 0:
            pts = v.T
            pin = np.abs(map_coordinates(U, pts, order=1, mode="nearest")) < p.pin_tol * pitch
            if cutf is not None:
                pin |= np.abs(map_coordinates(cutf, pts, order=1, mode="nearest")) < p.pin_tol * pitch
            if Env is not None:   # a vertex on the envelope stays there (#319)
                pin |= np.abs(map_coordinates(Env, pts, order=1, mode="nearest")) < p.pin_tol * pitch
        V = (taubin_pinned(v, f, p.taubin_iterations, p.taubin_lambda, p.taubin_mu, pin)
             if p.taubin_iterations > 0 else v)
        verts = origin_f + pitch * (V + 0.5)
        faces = f.astype(np.int64)
        # A body that does not touch a frozen region is an island of the smoothing: remove it.
        fz_at = frozen_near(verts)
        verts, faces, n_extra, extra_vol = _keep_touching(verts, faces, lambda ids: fz_at[ids].any())
        steps["smoothed"] = abs(_signed_volume(verts, faces))
        clipf = {"planes": clip_found, "margin_mm": p.pin_tol * pitch, "vertices_beyond": 0, "largest_excess_mm": 0.0,
                 "volume_removed_mm3": 0.0, "clipped": [], "left": [], "skipped_for_interfaces": clip_skipped}
        if clip_note:
            clipf["note"] = clip_note
        envf = {"cut": False, "vertices_beyond": 0, "largest_excess_mm": 0.0, "volume_removed_mm3": 0.0, "volume_removed_pct": 0.0, "faces": []}
        if envelope is None:
            envf["reason"] = env_note
        else:
            envf.update(tol_mm=float(envelope["tol"]), regions=envelope["regions"], interface_pieces=envelope["pieces"])
        exact = {"union": False, "cut": False}
        if stl_f or stl_c:
            moved, mfaces = verts, faces   # the mesh as the exact step takes it; `verts` stays the smooth mesh, which a failed step keeps
            if Env is not None and clip_planes:   # #319: a surface that ends within the band inside a shared plane ends on it
                try:
                    moved, envf["flush"] = flush_to_planes(verts, faces, clip_planes, p.pin_tol * pitch, on_envelope, inside=in_frozen)
                except Exception as exc:   # the mesh is then not moved; the facts say so
                    reraise_if_resource(exc, "the move of the mesh onto the shared planes")
                    note_fallback("flush", exc, "the mesh is not moved onto the shared planes")
                    envf["flush"] = {"error": f"{type(exc).__name__}: {exc}"}
            if snap and snap_error:
                snapf = {"error": snap_error}
            elif snap:
                try:
                    moved, snapf = snap_into_regions(moved, faces, snap_surfaces, p.pin_tol * pitch)
                except Exception as exc:   # the exact step then runs on the mesh as it is
                    reraise_if_resource(exc, "the snap to the region faces")
                    note_fallback("snap", exc, "the mesh is not moved to the region faces")
                    snapf = {"error": f"{type(exc).__name__}: {exc}"}
            if clip_planes:   # after the snap: what the snap took into a region is not past the plane any more
                try:
                    moved, mfaces, found = clip_to_planes(moved, mfaces, clip_planes, clip_tol, margin=p.pin_tol * pitch,
                                                          inside=in_frozen, dry=not cut)
                    clipf = {**found, "planes": clip_found, "skipped_for_interfaces": clip_skipped, **({"note": clip_note} if clip_note else {})}
                except Exception as exc:   # the mesh is then not cut; the facts say so
                    reraise_if_resource(exc, "the cut at the shared planes")
                    note_fallback("plane_clip", exc, "the mesh is not cut at the shared planes")
                    clipf["error"] = f"{type(exc).__name__}: {exc}"
            if envelope is not None:   # after the cut at the planes: what is still past the envelope (curved and slanted faces, the other faces)
                try:
                    moved, mfaces, found = cut_to_envelope(moved, mfaces, envelope, dry=not cut)
                    envf.update(found)
                except Exception as exc:   # the mesh is then not cut; the facts say so
                    reraise_if_resource(exc, "the cut to the envelope")
                    note_fallback("envelope_cut", exc, "the mesh is not cut to the envelope")
                    envf["error"] = f"{type(exc).__name__}: {exc}"
            try:
                body = _manifold_of(moved, mfaces)
                if stl_f:
                    for q in paths_f:
                        body = body + _load_manifold(q)
                    exact["union"] = True
                    steps["after_union"] = body.volume()
                if stl_c:
                    cutter = None
                    for q in paths_c:
                        man = _load_manifold(q)
                        cutter = man if cutter is None else cutter + man
                    body = body - cutter
                    exact["cut"] = True
                    steps["after_cut"] = body.volume()
                verts, faces = _from_manifold(body)
                # after the cut, drop the pieces that no longer touch a frozen region (same rule in every source case)
                fz_at = frozen_near(verts)
                verts, faces, n2, ev2 = _keep_touching(verts, faces, lambda ids: fz_at[ids].any())
                n_extra += n2
                extra_vol += ev2
            except Exception as exc:   # keep the smooth mesh; the facts say what happened
                reraise_if_resource(exc, "the exact boolean step")
                note_fallback("exact", exc, "the smooth mesh is kept")
                exact["error"] = str(exc)
                if snapf is not None and "error" not in snapf:
                    snapf = {"error": f"the exact boolean step failed, so the mesh was not moved: {type(exc).__name__}: {exc}"}
        verts, faces, voids = _drop_voids(verts, faces, keep=design_void if cavities["count"] else None)   # before `_orient` (it keeps a cavity)
        steps["voids_kept"] = voids.pop("kept", {"count": 0, "volume_mm3": 0.0})
        steps["voids_filled"] = voids
        verts, faces = _orient(verts, faces)
        verts = verts.astype(np.float32).astype(np.float64)   # the STL holds float32
        if p.simplify_tol > 0:
            verts, faces, clean = _cleanup_float32(verts, faces, p.simplify_tol, log_skip=False)
            steps["cleanup"] = clean
            if "error" in clean:   # a geometry error kept the mesh as it was: the fact is in the record of the mesh step (`exact`)
                exact["cleanup_error"] = clean["error"]
                note_fallback("cleanup", clean["error"], "the mesh is not cleaned up", error_type=clean.get("error_type"))
        steps["final"] = abs(_signed_volume(verts, faces))
        if F_man is not None or C_man is not None:
            try:
                b = _manifold_of(verts, faces)
                b = b + F_man if F_man is not None else b
                b = b - C_man if C_man is not None else b
                steps["with_interfaces"] = float(b.volume())
            except Exception as exc:   # the calibration then uses the volume of the mesh as it is
                reraise_if_resource(exc, "the volume with the interface shapes")
                note_fallback("interface_volume", exc, "the calibration uses the mesh volume")
                exact["interface_volume_error"] = f"{type(exc).__name__}: {exc}"
        return verts, faces, {"steps": steps, "exact": exact, "extra": n_extra, "extra_vol": extra_vol, "snap": snapf, "plane_clip": clipf,
                                  "envelope_cut": envf}

    # The loop ends when the error is below 5e-4. A run without the fill that still misses by `extra_rounds_from` (0.1 %) or more after
    # `calibrate_steps` rounds then gets more rounds, up to `calibrate_steps_fill`, with the measured slope (#348); any other run keeps its bytes.
    rounds_max = max(p.calibrate_steps, p.calibrate_steps_fill)
    v_last = [None, None]   # the marching-cubes mesh of the last round (`secant` and the build without the move read it)

    def wants_cut(bf):
        """True when the build has a part past a plane over a region face or past the envelope: the pass with the cuts must run."""
        return bool((clip_planes and bf["plane_clip"].get("clipped")) or bf["envelope_cut"].get("vertices_beyond"))

    def calibrate(cut):
        """The volume calibration: the secant on the level shift, a mesh built in each round. `cut` is for `build`.

        Return `(verts, faces, facts of the kept build, rounds, marching-cubes volume at shift 0, volume of the kept round)`. The kept round is
        the last one, except when an extra round of #348 missed more than the best round before it (see below).
        """
        where["pass"] += 1
        delta = 0.0
        v, f, vol = mc_at(delta)
        v_last[:] = [v, f]
        vol_raw = vol
        hist = [(delta, vol)]
        target = v_thr
        rounds = []
        seen = []   # (target, final volume) of each round, for the slope of the fill runs
        best = None   # the extra rounds of #348: (|error|, index in `rounds`, verts, faces, facts, volume, marching-cubes mesh) of the best round from `calibrate_steps` on
        for k in range(max(rounds_max, 0) + 1):
            delta, vol, stop = secant(delta, vol, target, hist, 3) if k else (delta, vol, None)
            where["round"] = k
            verts, faces, bf = build(v_last[0], v_last[1], cut=cut) if k else build(v, f, cut=cut)
            # The sharp volume of a removed body is not part of the target: the shift must not pay for it with other members.
            v_goal = v_thr - bf["extra_vol"]
            if v_goal <= 0.0:
                raise ValueError(f"the volume target is not above 0: the sharp volume is {v_thr:.3f} mm3 and the bodies "
                                 f"that do not touch a frozen region take {bf['extra_vol']:.3f} mm3 of it")
            v_final = bf["steps"].get("with_interfaces", bf["steps"]["final"])   # with interface shapes: the volume they give
            err = (v_final - v_goal) / v_goal
            cut_mm3 = bf["plane_clip"].get("volume_removed_mm3", 0.0)   # the cut of this round: the shift of the next round pays it
            env_mm3 = bf["envelope_cut"].get("volume_removed_mm3", 0.0)   # the same for the cut to the envelope
            rounds.append({"delta_mm": float(delta), "mc_mm3": float(vol), "final_mm3": v_final,
                           "target_mm3": float(v_goal), "error": err, **({"plane_clip_removed_mm3": float(cut_mm3)} if cut_mm3 else {}),
                           **({"envelope_cut_removed_mm3": float(env_mm3)} if env_mm3 else {}),
                           **({"stopped": stop} if stop else {})})
            if stop:   # more rounds would build the same mesh again
                log.warning("SDF volume calibration stopped in round %d, the mesh is kept as it is: %s", k, stop)
                break
            if k >= p.calibrate_steps and not surf.get("ran"):
                if best is not None and abs(err) >= best[0]:
                    # An extra round that misses more ends the loop, and the best round is kept: the volume can jump with the level (a slit
                    # closes and its void is filled), and then the rounds circle round the target. So the result of a pass is never worse
                    # than its round of the plain limit, and a run whose first extra round misses more keeps the bytes it had before #348.
                    rounds[best[1]]["next_round_error"] = err
                    del rounds[best[1] + 1:]
                    _e, _i, verts, faces, bf, v_final, v_last[:] = best   # `v_last`: the build without the move (snap fact) is at the kept level
                    break
                best = (abs(err), len(rounds) - 1, verts, faces, bf, v_final, list(v_last))
                if k == p.calibrate_steps and abs(err) < p.extra_rounds_from:
                    break   # the run meets the bar of #348 with the plain rounds: it keeps the bytes that it had before
            if abs(err) < 5e-4 or k >= rounds_max:
                break
            if not cut and k >= p.calibrate_steps and not surf.get("ran") and wants_cut(bf):
                break   # the pass with the cuts follows and starts again, so more rounds of this pass would be lost time
            # The linear update assumes that the cuts take a fixed volume. When the mesh lies past the envelope (or on it, with the fill), about half
            # of each level change is cut away, so the miss halves in each round. A fill run divides the miss by the measured slope of the final volume
            # to the target from its second round on. Any other run does so in its extra rounds (#348: the bar of the tests missed by 1.15 % after
            # 3 rounds), so a run that ends within the plain rounds keeps its bytes.
            seen.append((target, v_final))
            slope = 1.0
            if (surf.get("ran") or k >= p.calibrate_steps) and len(seen) >= 2 and seen[-1][0] != seen[-2][0]:
                slope = min(max((seen[-1][1] - seen[-2][1]) / (seen[-1][0] - seen[-2][0]), p.fill_slope_min), p.fill_slope_max)
            target = target - (v_final - v_goal) / slope
        return verts, faces, bf, rounds, vol_raw, v_final

    t2 = time.monotonic()
    # First without the cut at the shared planes: a run whose result has no part past a plane over a region face then has the same bytes as
    # before (the cut of an early round would move the level shift of the later ones). When the last round has such a part, the calibration
    # runs again from its start with the cut, so that the level shift pays for what the cut takes.
    # With the field limited to the envelope (#319), the bytes of a run change anyway, and the cuts take only what marching cubes leaves past a
    # concave edge of the envelope: one pass with the cuts from the start.
    t_cut = 0.0
    if Env is not None:
        cut_pass, passes = True, 1
        verts, faces, bf, rounds, vol_raw, v_final = calibrate(cut=True)
    else:
        cut_pass, passes = False, 1
        verts, faces, bf, rounds, vol_raw, v_final = calibrate(cut=False)
        if wants_cut(bf):
            cut_pass, passes = True, 2
            t_c = time.monotonic()
            verts, faces, bf, rounds, vol_raw, v_final = calibrate(cut=True)
            t_cut = time.monotonic() - t_c
    bf["plane_clip"]["passes"] = passes
    bf["envelope_cut"]["passes"] = passes
    bf["envelope_cut"]["field_limit"] = Env is not None
    if limit_note:
        bf["envelope_cut"]["field_limit_note"] = limit_note
    snap = bf["snap"]
    if snap and snap.get("moved"):
        # What the move does to the volume at this level shift: the volume of the last build minus the volume of the same marching-cubes
        # mesh built without the move. The calibration has already paid it back with the global shift (see `calibration`).
        where.update(build="without_move", round=None)
        try:
            _v0, _f0, b0 = build(v_last[0], v_last[1], snap=False, cut=cut_pass)
            if b0["exact"].get("error"):   # the volume of a mesh without the booleans is not the volume to compare with
                snap["volume_change_error"] = f"the exact boolean step of the build without the move failed: {b0['exact']['error']}"
            else:
                snap["volume_change_mm3"] = float(v_final - b0["steps"].get("with_interfaces", b0["steps"]["final"]))
        except Exception as exc:
            reraise_if_resource(exc, "the volume change of the snap")
            snap["volume_change_error"] = f"{type(exc).__name__}: {exc}"
            note_fallback("snap_volume_change", exc, "the fact volume_change_mm3 is missing")
    for e in fallbacks:   # the kept round is the last round of the last pass (the rounds list ends with it)
        e["kept"] = bool(e["build"] == "calibration" and e["pass"] == passes and e["round"] == len(rounds) - 1)
    tm_["smooth"] = time.monotonic() - t1
    tm_["mesh"] = time.monotonic() - t2
    # The time cost of the cuts: the whole mesh step, and the second calibration pass that only a cut (planes or envelope) starts.
    bf["envelope_cut"]["t_s"] = {"mesh_step": round(tm_["mesh"], 3), "cut_pass": round(t_cut, 3), "field_limit": round(t_l, 3)}
    steps, exact, n_extra, extra_vol = bf["steps"], bf["exact"], bf["extra"], bf["extra_vol"]
    vol_smooth = steps["smoothed"]
    if n_extra:
        log.warning("SDF mesh step removed %d body(ies) of %.1f mm3 that do not touch a frozen region; "
                    "the volume target leaves them out", n_extra, extra_vol)
    voids = steps.get("voids_filled") or {"count": 0, "volume_mm3": 0.0}
    if voids["count"]:
        log.warning("SDF mesh step filled %d closed void(s) of %.2f mm3 (a thin slit between a fixed or load region and the design "
                    "that the exact union sealed); the volume calibration counts them as material", voids["count"], voids["volume_mm3"])
    kept = steps.get("voids_kept") or {"count": 0, "volume_mm3": 0.0}
    if kept["count"]:
        log.warning("SDF mesh step kept %d closed void(s) of %.2f mm3 of the design: the design has a cavity with no opening there, so the mesh "
                    "has one too; its volume is not material, in the mesh and in the volume target (print_check closed_voids reports it)",
                    kept["count"], kept["volume_mm3"])
    pc = bf["plane_clip"]
    if pc.get("vertices_beyond"):
        log.warning("SDF mesh step cut the smooth mesh at %d plane(s) that the design space shares with a fixed, load or solid region: "
                    "%d vertices lay past a plane by up to %.3f mm and %.2f mm3 were removed; the volume calibration counts the cut",
                    len(pc["clipped"]), pc["vertices_beyond"], pc["largest_excess_mm"], pc["volume_removed_mm3"])
    ec = bf["envelope_cut"]
    if ec.get("cut"):
        log.warning("SDF mesh step cut the smooth mesh to the envelope (design space, frozen regions, union interface shapes): %d vertices lay "
                    "past it by up to %.3f mm on %d face(s) of the design space and %.2f mm3 (%.2f %%) were removed; the volume calibration counts the cut",
                    ec["vertices_beyond"], ec["largest_excess_mm"], len(ec["faces"]), ec["volume_removed_mm3"], ec["volume_removed_pct"])
    t3 = time.monotonic()
    loss = deep_loss(S, verts, faces, origin_f, pitch, p.deep_depth)
    tm_["local_loss"] = time.monotonic() - t3
    del S

    mesh = trimesh.Trimesh(verts, faces, process=False)
    info = {
        "method": "sdf",
        "params": {**asdict(p), "up": up, "sigma_mm": round(sigma, 4), "pitch_mm": round(pitch, 4)},
        "sources": src,
        "exact": exact,
        "snap": snap if (stl_f or stl_c) else None,
        "volume_steps_mm3": steps,
        "volume_thresholded_mm3": v_thr,
        "volume_smoothed_mm3": vol_smooth,
        "volume_before_calibration_mm3": vol_raw,
        "calibration": rounds,
        "fallbacks": fallbacks,
        "islands_removed": int(removed_islands),
        "extra_bodies_removed": int(n_extra),
        "extra_bodies_volume_mm3": float(extra_vol),
        "voids_filled": voids,
        "voids_kept": kept,
        "plane_clip": bf["plane_clip"],
        "envelope_cut": bf["envelope_cut"],
        "envelope_fill": surf,
        "local_loss": loss,
        "watertight": bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0),
        "grid": fact,
        "t_s": {k: round(v_, 3) for k, v_ in tm_.items()},
    }
    return verts, faces, info


# -- checks of the final mesh --------------------------------------------------------------
def stl_facts(data: bytes) -> dict:
    """Load STL bytes (the file as written) and report the facts of the loaded mesh.

    `ok` is true when the file is watertight, has a consistent winding and a positive volume.
    `open_edges` have 1 face, `nonmanifold_edges` have more than 2.
    A file with more than one body also has `body_volumes_mm3`: the signed volume of each body, the largest first (a body that turns
    inward, for example a void that was turned over, has a negative volume).
    """
    import trimesh

    back = trimesh.load(io.BytesIO(data), file_type="stl")
    nf = len(back.faces)
    shells = back.split(only_watertight=False) if nf else []
    parts = len(shells)
    open_e = nonman = 0
    if nf:
        _e, cnt = np.unique(back.edges_sorted, axis=0, return_counts=True)
        open_e, nonman = int((cnt == 1).sum()), int((cnt > 2).sum())
    facts = {"watertight": bool(back.is_watertight), "winding_consistent": bool(back.is_winding_consistent),
             "bodies": int(parts), "open_edges": open_e, "nonmanifold_edges": nonman,
             "zero_area_faces": int((back.area_faces < 1e-12).sum()) if nf else 0,
             "volume_mm3": round(float(back.volume), 3) if nf else 0.0, "triangles": int(nf)}
    facts["ok"] = bool(facts["watertight"] and facts["winding_consistent"] and facts["volume_mm3"] > 0)
    if parts > 1:
        facts["body_volumes_mm3"] = sorted((round(float(q.volume), 3) for q in shells), key=abs, reverse=True)
    return facts


def label_checks(verts, faces, labels, origin, h: float, up: int, pad: int) -> dict:
    """Numbers against the label voxel boxes (for a run without region STL files).

    `fixed_missing_mm3`: volume of the fixed and load label voxels that the mesh does not hold.
    `keep_out_inside_mm3`: volume of the mesh inside the keep-out label voxels.
    The mesh is sampled at 8 points in each fine voxel, so each number has a resolution of about one fine voxel.
    """
    import itertools

    import trimesh

    from . import post
    from .voxelize import inside_mask

    pitch = h / up
    fixed_lab = post._upsample(((labels == post.FIXED) | (labels == post.LOAD)).astype(np.float32), up) > 0.5
    hole_lab = post._upsample((labels == post.KEEP_OUT).astype(np.float32), up) > 0.5
    origin_f = np.asarray(origin, np.float64) - pad * h
    mesh = trimesh.Trimesh(np.asarray(verts, np.float64), np.asarray(faces), process=False)
    occ = np.zeros(fixed_lab.shape, np.float32)
    for o in itertools.product((-0.25, 0.25), repeat=3):
        occ += inside_mask(mesh, origin_f + pitch * (np.array(o) + 0.013), pitch, fixed_lab.shape)
    occ /= 8.0
    return {"fixed_missing_mm3": round(float((1.0 - occ[fixed_lab]).sum()) * pitch ** 3, 3),
            "fixed_label_mm3": round(float(fixed_lab.sum()) * pitch ** 3, 1),
            "keep_out_inside_mm3": round(float(occ[hole_lab].sum()) * pitch ** 3, 3),
            "keep_out_label_mm3": round(float(hole_lab.sum()) * pitch ** 3, 1),
            "method": "8 sub-samples per fine voxel against the label voxel boxes"}


def region_checks(verts, faces, fixed: dict, keep_out: dict) -> dict:
    """Measure the final mesh against the region STL files (exact booleans, manifold3d).

    `fixed` and `keep_out` map a name to a path. For each fixed (and load) region:
    `uncovered_mm3` is the volume of the region (minus keep-out) that the mesh does not hold,
    and `max_outside_mm` is how far that missing part reaches from the mesh surface.
    For each keep-out region: `intersect_mm3` is the volume of mesh inside the region.
    """
    import trimesh

    body = _manifold_of(np.asarray(verts, np.float64), np.asarray(faces, np.int64))
    cutter = None
    out_ko = {}
    for name, q in keep_out.items():
        man = _load_manifold(q)
        out_ko[name] = {"intersect_mm3": float((body ^ man).volume())}
        cutter = man if cutter is None else cutter + man
    bmesh = None
    out_fx = {}
    for name, q in fixed.items():
        man = _load_manifold(q)
        if cutter is not None:
            man = man - cutter
        rest = man - body
        vol = float(rest.volume())
        reach = 0.0
        if vol > 1e-9:
            rv, _rf = _from_manifold(rest)
            if bmesh is None:
                bmesh = trimesh.Trimesh(np.asarray(verts, np.float64), np.asarray(faces), process=False)
            reach = float(_query(trimesh.proximity.closest_point, bmesh, rv)[1].max())
        out_fx[name] = {"uncovered_mm3": vol, "max_outside_mm": reach, "region_mm3": float(man.volume())}
    return {"fixed": out_fx, "keep_out": out_ko}
