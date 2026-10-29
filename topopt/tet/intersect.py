"""Places where a triangle surface crosses itself (P4, F2-6). It needs numpy and numba only. It does not import gmsh.

gmsh stops on a surface in which an edge goes through a face ("PLC Error: A segment and a facet intersect"). It names one place.
The surface preparation of `topopt.tet.surface` can make such a crossing from a surface that has none.
A collapse moves a vertex by up to the shape error bound and checks the triangles around it only.
`find_crossings` finds all the crossings at once. It needs a fraction of a second for 30,000 faces and about 10 s for 600,000.
So the mesher can refine around every place in one retry, instead of one place for each retry.
It can also tell an STL that crosses itself (no retry helps) from a prepared surface that does.

What it finds: a pair of faces that share no vertex, where an edge of one face goes through the other face.
What it does not find:
- Faces that share a vertex or an edge and overlap. That is a fold, and `topopt.tet.surface.repair_folds` handles it.
- Faces that lie in one plane and overlap. The test of an edge against a face is not defined for an edge in the plane of the face.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

# Tolerance of the edge-against-face test, in the units of the barycentric coordinates and of the edge parameter (no unit):
# an edge that ends on a face (or runs along its rim) is a touch, not a crossing.
TOUCH_TOL = 1e-9
# The cells of the grid are this many times the median edge length of the surface, or bigger (see `grid_cell`).
CELL_EDGES = 3.0
# The first size of the list that the kernel fills; the kernel counts every pair, so a bigger list is made when it is needed.
FIRST_CAPACITY = 4096


@dataclass
class Crossings:
    """All crossings of a surface: `total` pairs of faces.

    `pairs` (total, 2) has the face indices. `points` (total, 3) has a point for each pair (mm, the frame of the surface) where an
    edge of one face goes through the other.
    """
    total: int
    pairs: np.ndarray
    points: np.ndarray

    def __bool__(self) -> bool:
        return self.total > 0


@nb.njit(cache=True)
def _segment_through_triangle(p0, p1, a, b, c, tol, hit):
    """1 if the segment p0-p1 goes through the triangle a-b-c (a touch does not count); the point is put in `hit`."""
    e1 = b - a
    e2 = c - a
    d = p1 - p0
    h = np.cross(d, e2)
    det = np.dot(e1, h)
    scale = np.sqrt(np.dot(e1, e1) * np.dot(e2, e2) * np.dot(d, d))
    if abs(det) <= 1e-12 * scale:                   # the segment lies in the plane of the face (or is parallel to it)
        return 0
    f = 1.0 / det
    s = p0 - a
    u = f * np.dot(s, h)
    if u < -tol or u > 1.0 + tol:
        return 0
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < -tol or u + v > 1.0 + tol:
        return 0
    t = f * np.dot(e2, q)
    if t < tol or t > 1.0 - tol:
        return 0
    for k in range(3):
        hit[k] = p0[k] + t * d[k]
    return 1


@nb.njit(cache=True)
def _faces_cross(V, F, i, j, tol, hit):
    A = V[F[i]]
    B = V[F[j]]
    for k in range(3):
        if _segment_through_triangle(A[k], A[(k + 1) % 3], B[0], B[1], B[2], tol, hit):
            return 1
        if _segment_through_triangle(B[k], B[(k + 1) % 3], A[0], A[1], A[2], tol, hit):
            return 1
    return 0


@nb.njit(cache=True)
def _cell_range(lo_x, hi_x, origin, cell):
    return int((lo_x - origin) / cell), int((hi_x - origin) / cell)


@nb.njit(cache=True)
def _crossings(V, F, cell, origin, nx, ny, nz, tol, capacity):
    nf = F.shape[0]
    bmin = np.empty((nf, 3))
    bmax = np.empty((nf, 3))
    for f in range(nf):
        for a in range(3):
            lo_, hi_ = 1e300, -1e300
            for k in range(3):
                x = V[F[f, k], a]
                lo_ = min(lo_, x)
                hi_ = max(hi_, x)
            bmin[f, a] = lo_
            bmax[f, a] = hi_
    ncell = nx * ny * nz
    start = np.zeros(ncell + 1, np.int64)
    for f in range(nf):
        i0, i1 = _cell_range(bmin[f, 0], bmax[f, 0], origin[0], cell)
        j0, j1 = _cell_range(bmin[f, 1], bmax[f, 1], origin[1], cell)
        k0, k1 = _cell_range(bmin[f, 2], bmax[f, 2], origin[2], cell)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    start[(i * ny + j) * nz + k + 1] += 1
    for c in range(ncell):
        start[c + 1] += start[c]
    items = np.empty(start[ncell], np.int64)
    fill = start[:-1].copy()
    for f in range(nf):
        i0, i1 = _cell_range(bmin[f, 0], bmax[f, 0], origin[0], cell)
        j0, j1 = _cell_range(bmin[f, 1], bmax[f, 1], origin[1], cell)
        k0, k1 = _cell_range(bmin[f, 2], bmax[f, 2], origin[2], cell)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    c = (i * ny + j) * nz + k
                    items[fill[c]] = f
                    fill[c] += 1
    pairs = np.empty((capacity, 2), np.int64)
    points = np.empty((capacity, 3))
    hit = np.empty(3)
    total = 0
    for c in range(ncell):
        s, e = start[c], start[c + 1]
        for x in range(s, e):
            fi = items[x]
            for y in range(x + 1, e):
                fj = items[y]
                shared = False
                for a in range(3):
                    for b in range(3):
                        if F[fi, a] == F[fj, b]:
                            shared = True
                if shared:
                    continue
                apart = False
                for a in range(3):
                    if bmax[fi, a] < bmin[fj, a] or bmax[fj, a] < bmin[fi, a]:
                        apart = True
                if apart:
                    continue
                # a pair that shares several cells is tested in one cell only: the one that holds the lowest corner of the
                # common box
                ci = int((max(bmin[fi, 0], bmin[fj, 0]) - origin[0]) / cell)
                cj = int((max(bmin[fi, 1], bmin[fj, 1]) - origin[1]) / cell)
                ck = int((max(bmin[fi, 2], bmin[fj, 2]) - origin[2]) / cell)
                if (ci * ny + cj) * nz + ck != c:
                    continue
                if _faces_cross(V, F, fi, fj, tol, hit):
                    if total < capacity:
                        pairs[total, 0] = fi
                        pairs[total, 1] = fj
                        for a in range(3):
                            points[total, a] = hit[a]
                    total += 1
    return pairs, points, total


def grid_cell(V: np.ndarray, F: np.ndarray) -> float:
    """The edge of a cell of the grid (mm): `max(3 x median edge, (volume of the bounding box / number of faces) ** (1/3))`.

    The first term gives a few faces in each cell of a surface.
    The second term keeps the grid from having more cells than the surface has faces.
    A large box around small faces needs it.
    A 600 mm box and a median edge of 0.3 mm would need 2.4 GB for the cells with the first term alone (P4 review, L3). The list of crossings does not depend on the cell size.
    """
    edge = np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))
    box = np.prod(V.max(axis=0) - V.min(axis=0))
    return max(CELL_EDGES * float(edge), float(box / len(F)) ** (1.0 / 3.0), 1e-9)


def find_crossings(V: np.ndarray, F: np.ndarray) -> Crossings:
    """All pairs of faces of the surface (V, F) where an edge of one face goes through the other.

    Faces that share a vertex are not tested. `V` (n, 3) float, `F` (m, 3) int.
    The result is deterministic: the pairs come in the order of the cells of a grid.
    """
    V = np.ascontiguousarray(V, np.float64)
    F = np.ascontiguousarray(F, np.int64)
    if len(F) < 2:
        return Crossings(0, np.empty((0, 2), np.int64), np.empty((0, 3)))
    cell = grid_cell(V, F)
    origin = V.min(axis=0) - cell
    n = np.ceil((V.max(axis=0) + cell - origin) / cell).astype(np.int64) + 1
    capacity = FIRST_CAPACITY
    while True:
        pairs, points, total = _crossings(V, F, cell, origin, int(n[0]), int(n[1]), int(n[2]), TOUCH_TOL, capacity)
        if total <= capacity:
            return Crossings(int(total), pairs[:total].copy(), points[:total].copy())
        capacity = int(total)                       # the count is exact: one more pass fills the whole list


def describe_places(points: np.ndarray, limit: int = 10) -> str:
    """The first `limit` places as text, `(x, y, z)` in mm, and the number that are left out. `limit` is the length of the list in
    a message only; the counts in the message are exact."""
    pts = np.asarray(points, float).reshape(-1, 3)
    shown = ", ".join("(" + ", ".join(f"{x:.2f}" for x in p) + ")" for p in pts[:limit])
    return shown + (f" and {len(pts) - limit} more" if len(pts) > limit else "")
