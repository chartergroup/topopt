"""Surface preparation for the tet mesher: STL -> clean closed manifold surface.

Ported from the file `pipeline/lib/p2_surface.py` of an earlier pipeline of the same author (commit d43bcf3),
without the vtk method and without text that is specific to one part. It needs numba, numpy, scipy and
trimesh, which are core dependencies. It does not import gmsh.

Why this module exists: gmsh meshes a discrete (STL) surface as a fixed boundary, so the
volume mesh is as fine as the STL triangles. A marching-cubes STL can have very short
edges and thin triangles. The surface must be coarsened first.

The decimator keeps the surface a closed 2-manifold:

* Edge collapses are half-edge collapses: one end vertex is moved onto the other,
  so a collapse adds no new position. Edge splits add vertices on edges, and the
  smoothing moves vertices along the surface. The shape change is bounded by the
  error limit eps below and reported as `deviation_from_input_mm`.
* An edge collapses only if the link condition holds (the two end vertices
  share exactly two neighbours).
* An edge collapses only if no triangle flips, no triangle turns more than
  about 53 degrees, no triangle becomes worse than a floor quality, and no
  edge grows beyond the limit.
* The mean squared distance (quadric error, area weighted) from the new
  vertex to all original planes that the two end vertices represent must be
  below eps^2. This bounds the shape change.
* Many collapses are done in one round. The collapses of one round have
  disjoint vertex neighbourhoods, so the checks of one collapse stay valid
  when the others are applied.
* Edge flips (Delaunay criterion, nearly coplanar faces only) then improve
  the triangle shapes.

Everything is deterministic: no random numbers, a fixed processing order.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numba as nb
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..config import UserError

# Absolute edge length in mm. Edges shorter than this are collapsed first, and a prepared surface with
# a shorter edge is rejected. It is NOT scaled with the target size: for a part that is smaller than
# about 5 mm (or for an STL in other units) pass a smaller `min_edge_mm` to `prepare_surface`.
MIN_EDGE_REJECT_MM = 0.05


# --------------------------------------------------------------------------- #
# Surface bookkeeping
# --------------------------------------------------------------------------- #

def edge_table(F: np.ndarray, nv: int):
    """Unique undirected edges (eu < ev) with the number of faces on each."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    key = e[:, 0].astype(np.int64) * nv + e[:, 1]
    uk, cnt = np.unique(key, return_counts=True)
    return (uk // nv).astype(np.int64), (uk % nv).astype(np.int64), cnt


def tri_quality(V: np.ndarray, F: np.ndarray):
    """Triangle quality 4*sqrt(3)*A/sum(l^2): 1 = equilateral, 0 = sliver."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    l0 = np.linalg.norm(b - c, axis=1)
    l1 = np.linalg.norm(c - a, axis=1)
    l2 = np.linalg.norm(a - b, axis=1)
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return 4.0 * np.sqrt(3.0) * A / np.maximum(l0 ** 2 + l1 ** 2 + l2 ** 2, 1e-300)


def _csr_incidence(F: np.ndarray, nv: int):
    """Vertex -> incident faces, CSR (ptr, idx)."""
    flat = F.ravel()
    order = np.argsort(flat, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=nv))]).astype(np.int64)
    return ptr, (order // 3).astype(np.int64)


def _csr_adjacency(eu: np.ndarray, ev: np.ndarray, nv: int):
    """Vertex -> sorted neighbour vertices, CSR (ptr, idx)."""
    a = np.concatenate([eu, ev])
    b = np.concatenate([ev, eu])
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=nv))]).astype(np.int64)
    return ptr, b.astype(np.int64)


@nb.njit(cache=True)
def _is_bow_tie(F, vf_ptr, vf_idx, v):
    """True when the faces around the vertex v (that has at least one face) are not one single cycle."""
    k = vf_ptr[v + 1] - vf_ptr[v]
    # link edges (a,b): oriented other two vertices of each face around v
    la = np.empty(k, np.int64)
    lb = np.empty(k, np.int64)
    for i in range(k):
        f = vf_idx[vf_ptr[v] + i]
        if F[f, 0] == v:
            la[i] = F[f, 1]; lb[i] = F[f, 2]
        elif F[f, 1] == v:
            la[i] = F[f, 2]; lb[i] = F[f, 0]
        else:
            la[i] = F[f, 0]; lb[i] = F[f, 1]
    # walk the oriented cycle starting at link edge 0
    used = np.zeros(k, np.uint8)
    cur = 0
    used[0] = 1
    steps = 1
    while True:
        nxt = -1
        for j in range(k):
            if used[j] == 0 and la[j] == lb[cur]:
                nxt = j
                break
        if nxt < 0:
            break
        used[nxt] = 1
        cur = nxt
        steps += 1
    return steps != k or lb[cur] != la[0]


@nb.njit(cache=True)
def _nonmanifold_vertices(F, vf_ptr, vf_idx, nv):
    """Count vertices whose face fan is not one single cycle (bow-tie vertices)."""
    bad = 0
    for v in range(nv):
        if vf_ptr[v + 1] - vf_ptr[v] == 0:
            continue
        if _is_bow_tie(F, vf_ptr, vf_idx, v):
            bad += 1
    return bad


@nb.njit(cache=True)
def _bow_ties_among(F, vf_ptr, vf_idx, verts):
    """Count the vertices of `verts` (each has at least one face) whose face fan is not one single cycle."""
    bad = 0
    for i in range(len(verts)):
        if _is_bow_tie(F, vf_ptr, vf_idx, verts[i]):
            bad += 1
    return bad


def fold_face_pairs(V: np.ndarray, F: np.ndarray, cos_limit: float = -0.999):
    """Face pairs that share an edge and lie almost exactly on top of each other (normals antiparallel).

    This is a fold or a zero-thickness fin. gmsh reports it as "self-intersecting facets"."""
    nv, nf = len(V), len(F)
    he_a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    he_b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    he_f = np.concatenate([np.arange(nf)] * 3)
    key = np.minimum(he_a, he_b).astype(np.int64) * nv + np.maximum(he_a, he_b)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    cnt = np.diff(np.r_[start, len(ks)])
    start = start[cnt == 2]
    f1, f2 = he_f[order[start]], he_f[order[start + 1]]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nn = np.linalg.norm(n, axis=1)
    c = np.einsum("ij,ij->i", n[f1], n[f2]) / np.maximum(nn[f1] * nn[f2], 1e-300)
    sel = c < cos_limit
    return f1[sel], f2[sel]


def fold_edge_count(V: np.ndarray, F: np.ndarray, cos_limit: float = -0.999) -> int:
    return int(len(fold_face_pairs(V, F, cos_limit)[0]))


def _collapse_faces(F: np.ndarray, keep: int, rem: int):
    """`(F2, kept)` after the collapse of the vertex `rem` onto `keep`: the faces with two equal vertices go, `kept` is the boolean
    mask of the faces of `F` that stay. `(None, None)` when the collapse does not remove exactly two faces (the edge is not a
    manifold edge)."""
    G = F.copy()
    G[G == rem] = keep
    kept = (G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])
    return (G[kept], kept) if int(kept.sum()) == len(F) - 2 else (None, None)


def _collapse_is_good_global(V: np.ndarray, F2: np.ndarray, base: int) -> bool:
    """The judge of `repair_folds` for one collapse, on the whole surface: the result is a closed oriented manifold and has fewer fold
    edges than the `base` of the surface before. About 60 ms for 28,000 faces. `_collapse_is_good_local` gives the same answer."""
    return surface_is_closed_manifold(V, F2) and fold_edge_count(V, F2) < base


def _collapse_is_good_local(V: np.ndarray, F: np.ndarray, vf_ptr: np.ndarray, vf_idx: np.ndarray, keep: int, rem: int) -> bool:
    """The answer of `_collapse_is_good_global` for the collapse of `rem` onto `keep`, from the faces around the two vertices only.

    `F` must be a closed oriented manifold (the caller checks it once), `vf_ptr` and `vf_idx` are `_csr_incidence(F, len(V))`. A
    collapse changes only the faces that touch `rem` (they get `keep`; the two that hold both ends go). So the result is a closed
    oriented manifold when, in the patch of all faces that touch `keep`, `rem` or a neighbour of them, no directed edge occurs twice, the
    reverse of every directed edge at a vertex of the star is there, and no vertex of the star has a fan that is not one cycle. The fold
    edges can change only at the edges of the faces that changed, and each such edge has both its faces in the patch; the edges of
    the patch that did not change have the same state before and after. So the fold edges go down in the surface when they go down
    in the patch. No number is used."""
    fr = vf_idx[vf_ptr[rem]:vf_ptr[rem + 1]]
    fk = vf_idx[vf_ptr[keep]:vf_ptr[keep + 1]]
    if np.count_nonzero(np.isin(fr, fk)) != 2:                         # the faces that hold both ends: a manifold edge has two
        return False
    star = np.unique(np.concatenate([F[fr].ravel(), F[fk].ravel()]))   # keep, rem and the neighbours of both
    patch = np.unique(np.concatenate([vf_idx[vf_ptr[w]:vf_ptr[w + 1]] for w in star]))
    ids, inv = np.unique(F[patch], return_inverse=True)
    n = len(ids)
    P = inv.reshape(-1, 3).astype(np.int64)                            # the patch, with vertex numbers 0 .. n-1
    lk, lr = int(np.searchsorted(ids, keep)), int(np.searchsorted(ids, rem))
    Q = np.where(P == lr, lk, P)
    Q = Q[(Q[:, 0] != Q[:, 1]) & (Q[:, 1] != Q[:, 2]) & (Q[:, 0] != Q[:, 2])]
    d = np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 0]]])
    dkey = d[:, 0] * n + d[:, 1]
    ud = np.unique(dkey)
    if len(ud) != len(dkey):                                           # a directed edge twice: not oriented
        return False
    star_now = np.searchsorted(ids, star[star != rem])                 # the vertices whose fan the collapse changes
    at_star = np.isin(d, star_now).any(axis=1)
    if not np.isin(d[at_star, 1] * n + d[at_star, 0], ud).all():       # a directed edge with no reverse: an open or non-manifold edge
        return False
    qptr, qidx = _csr_incidence(Q, n)
    with_faces = star_now[np.diff(qptr)[star_now] > 0]
    if _bow_ties_among(Q, qptr, qidx, with_faces):
        return False
    Vl = V[ids]
    return fold_edge_count(Vl, Q) < fold_edge_count(Vl, P)


def repair_folds(V: np.ndarray, F: np.ndarray, max_fixes: int = 400, verbose: bool = False):
    """Remove fold edges by edge collapses of the fold faces' own edges.

    Each collapse is kept only if the surface stays a closed oriented manifold and the number of
    fold edges goes down. Returns (F, folds_before, folds_after).

    The collapses are judged on the faces around the two vertices (`_collapse_is_good_local`), not on the whole surface: the answer is
    the same, and the cost of one judgement does not grow with the surface. Measured (F4-27, revision of this change, one Linux x86_64
    host): the first try of `prepare_surface` for the f6a phone stand at 0.973 mm (315 fold edges, 143 fixes, 3 fold edges left) took
    959 s with the whole-surface judge and 96 s with this one, and the repaired surface was the same (equal `F`). On surfaces of f7c,
    f0_root, f6a and a sphere the two judges gave the same answer for each of 10,238 candidates (3,313 accepted). The cost of the
    whole-surface judge is about 60 ms for each candidate at 28,000 faces, and the fold pairs that cannot be fixed are tried again
    before every new fix. A surface that is not a closed oriented manifold at the start is judged on the
    whole surface, as before."""
    F2, before, after, _rows = _repair_folds(V, F, max_fixes, verbose)
    return F2, before, after


def _repair_folds(V: np.ndarray, F: np.ndarray, max_fixes: int = 400, verbose: bool = False, rows: np.ndarray | None = None):
    """`repair_folds` that also carries `rows` (one label for each face of `F`, default the row numbers) through the repair.
    Returns (F, folds_before, folds_after, rows): `rows[i]` is the label of the face `F[i]` of the result."""
    F = np.asarray(F, np.int64).copy()
    rows = np.arange(len(F)) if rows is None else np.asarray(rows)
    f1, f2 = fold_face_pairs(V, F)
    before = len(f1)
    local = bool(before) and surface_is_closed_manifold(V, F)
    fixes = 0
    while len(f1) and fixes < max_fixes:
        base = len(f1)
        if local:
            vf_ptr, vf_idx = _csr_incidence(F, len(V))
        done = False
        for i in range(len(f1)):
            # candidate edges: those of the two fold faces first, then every edge of the faces around them
            core = np.unique(np.concatenate([F[f1[i]], F[f2[i]]]))
            if local:
                ring = np.unique(np.concatenate([vf_idx[vf_ptr[c]:vf_ptr[c + 1]] for c in core]))
            else:
                ring = np.flatnonzero(np.isin(F, core).any(axis=1))
            def edges_of(faces):
                return {tuple(sorted((int(a), int(b)))) for fa in faces
                        for a, b in ((fa[0], fa[1]), (fa[1], fa[2]), (fa[2], fa[0]))}
            first = edges_of((F[f1[i]], F[f2[i]]))
            rest = edges_of(F[ring]) - first
            key = lambda e: float(np.linalg.norm(V[e[0]] - V[e[1]]))
            pairs = sorted(first, key=key) + sorted(rest, key=key)
            for u, v in pairs:
                for keep, rem in ((u, v), (v, u)):
                    if local:
                        F2, kept = (_collapse_faces(F, keep, rem) if _collapse_is_good_local(V, F, vf_ptr, vf_idx, keep, rem)
                                    else (None, None))
                    else:
                        F2, kept = _collapse_faces(F, keep, rem)
                        if F2 is not None and not _collapse_is_good_global(V, F2, base):
                            F2 = None
                    if F2 is not None:
                        F, rows = F2, rows[kept]
                        done = True
                        break
                if done:
                    break
            if done:
                break
        if not done:
            break
        fixes += 1
        f1, f2 = fold_face_pairs(V, F)
        if verbose:
            print(f"    fold repair {fixes}: {len(f1)} folds left", flush=True)
    return F, before, len(f1), rows


def surface_is_closed_manifold(V: np.ndarray, F: np.ndarray) -> bool:
    """Fast closed oriented 2-manifold test (every edge in two faces, one per direction; no bow-tie vertex)."""
    nv = len(V)
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    dkey = d[:, 0].astype(np.int64) * nv + d[:, 1]
    ud = np.unique(dkey)
    if len(ud) != len(dkey):
        return False
    rkey = d[:, 1].astype(np.int64) * nv + d[:, 0]
    if not np.isin(rkey, ud).all():
        return False
    ptr, idx = _csr_incidence(F, nv)
    return int(_nonmanifold_vertices(F, ptr, idx, nv)) == 0


def surface_report(V: np.ndarray, F: np.ndarray) -> dict:
    """Closed-manifold audit and size statistics of a triangle surface."""
    nv, nf = len(V), len(F)
    eu, ev, cnt = edge_table(F, nv)
    L = np.linalg.norm(V[eu] - V[ev], axis=1)
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    dkey = d[:, 0].astype(np.int64) * nv + d[:, 1]
    ud = np.unique(dkey)
    rkey = d[:, 1].astype(np.int64) * nv + d[:, 0]
    oriented = bool(len(ud) == len(dkey) and np.isin(rkey, ud).all())
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = float(0.5 * np.linalg.norm(n, axis=1).sum())
    vol = float(np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) / 6.0)
    G = sp.coo_matrix((np.ones(len(eu)), (eu, ev)), shape=(nv, nv))
    ncomp = int(connected_components(G, directed=False)[0]) - int((np.bincount(F.ravel(), minlength=nv) == 0).sum())
    ptr, idx = _csr_incidence(F, nv)
    bow = int(_nonmanifold_vertices(F, ptr, idx, nv)) if oriented else -1
    q = tri_quality(V, F)
    return dict(
        vertices=nv, faces=nf, volume_mm3=vol, area_mm2=area,
        edges=int(len(eu)), open_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
        consistently_oriented=oriented, nonmanifold_vertices=bow, bodies=ncomp,
        closed_manifold=bool((cnt == 2).all() and oriented and bow == 0),
        fold_edges=fold_edge_count(V, F),
        edge_min_mm=float(L.min()), edge_p1_mm=float(np.percentile(L, 1)), edge_median_mm=float(np.median(L)),
        edge_mean_mm=float(L.mean()), edge_max_mm=float(L.max()),
        tri_quality_min=float(q.min()), tri_quality_p1=float(np.percentile(q, 1)),
        tri_quality_median=float(np.median(q)),
    )


# --------------------------------------------------------------------------- #
# Edge-collapse decimation (half-edge collapse, link condition, quadric bound)
# --------------------------------------------------------------------------- #

@nb.njit(cache=True)
def _common_neighbours(adj_ptr, adj_idx, u, v):
    i, iend = adj_ptr[u], adj_ptr[u + 1]
    j, jend = adj_ptr[v], adj_ptr[v + 1]
    c = 0
    while i < iend and j < jend:
        a, b = adj_idx[i], adj_idx[j]
        if a == b:
            c += 1; i += 1; j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return c


@nb.njit(cache=True)
def _quad_cost(QA, Qb, Qc, W, u, v, p):
    """Mean squared distance of point p to the planes represented by u and v."""
    ax = QA[u, 0] + QA[v, 0]; axy = QA[u, 1] + QA[v, 1]; axz = QA[u, 2] + QA[v, 2]
    ay = QA[u, 3] + QA[v, 3]; ayz = QA[u, 4] + QA[v, 4]; az = QA[u, 5] + QA[v, 5]
    bx = Qb[u, 0] + Qb[v, 0]; by = Qb[u, 1] + Qb[v, 1]; bz = Qb[u, 2] + Qb[v, 2]
    c = Qc[u] + Qc[v]
    w = W[u] + W[v]
    x, y, z = p[0], p[1], p[2]
    e = (ax * x * x + ay * y * y + az * z * z + 2.0 * (axy * x * y + axz * x * z + ayz * y * z)
         + 2.0 * (bx * x + by * y + bz * z) + c)
    return max(e, 0.0) / max(w, 1e-300)


@nb.njit(cache=True)
def _neighbour_reference(V, F, vf_ptr, vf_idx, f, tiny):
    """Sum of the unit normals of the reliable (non-sliver) faces across the three edges of face f."""
    rx = 0.0; ry = 0.0; rz = 0.0; cnt = 0
    for e in range(3):
        p = F[f, e]
        q = F[f, (e + 1) % 3]
        for ii in range(vf_ptr[p], vf_ptr[p + 1]):
            g = vf_idx[ii]
            if g == f:
                continue
            if F[g, 0] == q or F[g, 1] == q or F[g, 2] == q:
                a, b, c = F[g, 0], F[g, 1], F[g, 2]
                ex = (V[b, 1] - V[a, 1]) * (V[c, 2] - V[a, 2]) - (V[b, 2] - V[a, 2]) * (V[c, 1] - V[a, 1])
                ey = (V[b, 2] - V[a, 2]) * (V[c, 0] - V[a, 0]) - (V[b, 0] - V[a, 0]) * (V[c, 2] - V[a, 2])
                ez = (V[b, 0] - V[a, 0]) * (V[c, 1] - V[a, 1]) - (V[b, 1] - V[a, 1]) * (V[c, 0] - V[a, 0])
                en = np.sqrt(ex * ex + ey * ey + ez * ez)
                l2 = max((V[b, 0] - V[a, 0]) ** 2 + (V[b, 1] - V[a, 1]) ** 2 + (V[b, 2] - V[a, 2]) ** 2,
                         (V[c, 0] - V[b, 0]) ** 2 + (V[c, 1] - V[b, 1]) ** 2 + (V[c, 2] - V[b, 2]) ** 2,
                         (V[a, 0] - V[c, 0]) ** 2 + (V[a, 1] - V[c, 1]) ** 2 + (V[a, 2] - V[c, 2]) ** 2)
                if en > 1e-14 and en >= tiny * np.sqrt(l2):
                    rx += ex / en; ry += ey / en; rz += ez / en
                    cnt += 1
                break
    return rx, ry, rz, cnt


@nb.njit(cache=True)
def _collapse_ok(V, F, vf_ptr, vf_idx, keep, rem, Lmax, cos_min, q_floor, tiny):
    """Can vertex `rem` be moved onto vertex `keep` without a bad triangle?

    `tiny` is an altitude limit (mm). A triangle whose altitude over its longest edge (2*area/longest
    edge) is below `tiny` is a sliver: its own normal is noise, so the new normal is compared with the
    reliable neighbours across its edges. Every other triangle must keep its orientation (no fold-over)."""
    pk = V[keep]
    for ii in range(vf_ptr[rem], vf_ptr[rem + 1]):
        f = vf_idx[ii]
        a, b, c = F[f, 0], F[f, 1], F[f, 2]
        if a == keep or b == keep or c == keep:
            continue                                   # this face disappears
        # old and new vertex positions of the face
        if a == rem:
            o0 = V[a]; o1 = V[b]; o2 = V[c]; n0 = pk; n1 = V[b]; n2 = V[c]
            x = b; y = c
        elif b == rem:
            o0 = V[a]; o1 = V[b]; o2 = V[c]; n0 = V[a]; n1 = pk; n2 = V[c]
            x = c; y = a
        else:
            o0 = V[a]; o1 = V[b]; o2 = V[c]; n0 = V[a]; n1 = V[b]; n2 = pk
            x = a; y = b
        # area vectors
        ox = (o1[1] - o0[1]) * (o2[2] - o0[2]) - (o1[2] - o0[2]) * (o2[1] - o0[1])
        oy = (o1[2] - o0[2]) * (o2[0] - o0[0]) - (o1[0] - o0[0]) * (o2[2] - o0[2])
        oz = (o1[0] - o0[0]) * (o2[1] - o0[1]) - (o1[1] - o0[1]) * (o2[0] - o0[0])
        mx = (n1[1] - n0[1]) * (n2[2] - n0[2]) - (n1[2] - n0[2]) * (n2[1] - n0[1])
        my = (n1[2] - n0[2]) * (n2[0] - n0[0]) - (n1[0] - n0[0]) * (n2[2] - n0[2])
        mz = (n1[0] - n0[0]) * (n2[1] - n0[1]) - (n1[1] - n0[1]) * (n2[0] - n0[0])
        no = np.sqrt(ox * ox + oy * oy + oz * oz)
        nn = np.sqrt(mx * mx + my * my + mz * mz)
        if nn < 1e-14:
            return False
        lo2 = max((o1[0] - o0[0]) ** 2 + (o1[1] - o0[1]) ** 2 + (o1[2] - o0[2]) ** 2,
                  (o2[0] - o1[0]) ** 2 + (o2[1] - o1[1]) ** 2 + (o2[2] - o1[2]) ** 2,
                  (o0[0] - o2[0]) ** 2 + (o0[1] - o2[1]) ** 2 + (o0[2] - o2[2]) ** 2)
        if no < tiny * np.sqrt(lo2):                       # sliver: |n| = 2 A, altitude = 2 A / longest edge
            # its own normal is noise: compare the new normal with the reliable neighbours instead
            rx, ry, rz, cnt = _neighbour_reference(V, F, vf_ptr, vf_idx, f, tiny)
            rn = np.sqrt(rx * rx + ry * ry + rz * rz)
            if cnt > 0 and rn > 1e-12 and (mx * rx + my * ry + mz * rz) < cos_min * nn * rn:
                return False
        elif (ox * mx + oy * my + oz * mz) < cos_min * no * nn:
            return False                               # flipped or turned too far
        # edge lengths of the two edges that change: (keep,x) and (keep,y)
        lkx = np.sqrt((pk[0] - V[x][0]) ** 2 + (pk[1] - V[x][1]) ** 2 + (pk[2] - V[x][2]) ** 2)
        lky = np.sqrt((pk[0] - V[y][0]) ** 2 + (pk[1] - V[y][1]) ** 2 + (pk[2] - V[y][2]) ** 2)
        lrx = np.sqrt((V[rem][0] - V[x][0]) ** 2 + (V[rem][1] - V[x][1]) ** 2 + (V[rem][2] - V[x][2]) ** 2)
        lry = np.sqrt((V[rem][0] - V[y][0]) ** 2 + (V[rem][1] - V[y][1]) ** 2 + (V[rem][2] - V[y][2]) ** 2)
        if lkx > max(Lmax, lrx) or lky > max(Lmax, lry):
            return False
        # triangle quality: must stay above the floor or not get worse
        l2 = (n1[0] - n0[0]) ** 2 + (n1[1] - n0[1]) ** 2 + (n1[2] - n0[2]) ** 2 \
            + (n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2 + (n2[2] - n1[2]) ** 2 \
            + (n0[0] - n2[0]) ** 2 + (n0[1] - n2[1]) ** 2 + (n0[2] - n2[2]) ** 2
        l2o = (o1[0] - o0[0]) ** 2 + (o1[1] - o0[1]) ** 2 + (o1[2] - o0[2]) ** 2 \
            + (o2[0] - o1[0]) ** 2 + (o2[1] - o1[1]) ** 2 + (o2[2] - o1[2]) ** 2 \
            + (o0[0] - o2[0]) ** 2 + (o0[1] - o2[1]) ** 2 + (o0[2] - o2[2]) ** 2
        qn = 2.0 * np.sqrt(3.0) * nn / max(l2, 1e-300)          # |n| = 2 A
        qo = 2.0 * np.sqrt(3.0) * no / max(l2o, 1e-300)
        if qn < q_floor and qn < qo:
            return False
    return True


@nb.njit(cache=True)
def _eval_edges(V, F, QA, Qb, Qc, W, eu, ev, cand, adj_ptr, adj_idx, vf_ptr, vf_idx,
                Lmax_c, eps2_c, sliver_len, cos_min, q_floor, out_keep, out_rem, out_ok):
    tiny = sliver_len
    for ci in range(cand.shape[0]):
        k = cand[ci]
        Lmax = Lmax_c[ci]
        eps2 = eps2_c[ci]
        u, v = eu[k], ev[k]
        out_ok[ci] = 0
        if _common_neighbours(adj_ptr, adj_idx, u, v) != 2:
            continue
        L = np.sqrt((V[u, 0] - V[v, 0]) ** 2 + (V[u, 1] - V[v, 1]) ** 2 + (V[u, 2] - V[v, 2]) ** 2)
        cu = _quad_cost(QA, Qb, Qc, W, u, v, V[u])      # keep u, remove v
        cv = _quad_cost(QA, Qb, Qc, W, u, v, V[v])      # keep v, remove u
        allow_u = (cu <= eps2) or (L <= sliver_len)
        allow_v = (cv <= eps2) or (L <= sliver_len)
        first_u = cu <= cv
        for trial in range(2):
            use_u = first_u if trial == 0 else (not first_u)
            if use_u:
                if allow_u and _collapse_ok(V, F, vf_ptr, vf_idx, u, v, Lmax, cos_min, q_floor, tiny):
                    out_keep[ci] = u; out_rem[ci] = v; out_ok[ci] = 1
                    break
            else:
                if allow_v and _collapse_ok(V, F, vf_ptr, vf_idx, v, u, Lmax, cos_min, q_floor, tiny):
                    out_keep[ci] = v; out_rem[ci] = u; out_ok[ci] = 1
                    break


@nb.njit(cache=True)
def _select_independent(order, eu, ev, adj_ptr, adj_idx, nv):
    """Greedy: pick edges in `order` whose closed vertex neighbourhoods are disjoint."""
    blocked = np.zeros(nv, np.uint8)
    sel = np.empty(order.shape[0], np.int64)
    n = 0
    for oi in range(order.shape[0]):
        ci = order[oi]
        u, v = eu[ci], ev[ci]
        ok = blocked[u] == 0 and blocked[v] == 0
        if ok:
            for ii in range(adj_ptr[u], adj_ptr[u + 1]):
                if blocked[adj_idx[ii]] != 0:
                    ok = False
                    break
        if ok:
            for ii in range(adj_ptr[v], adj_ptr[v + 1]):
                if blocked[adj_idx[ii]] != 0:
                    ok = False
                    break
        if ok:
            blocked[u] = 1; blocked[v] = 1
            for ii in range(adj_ptr[u], adj_ptr[u + 1]):
                blocked[adj_idx[ii]] = 1
            for ii in range(adj_ptr[v], adj_ptr[v + 1]):
                blocked[adj_idx[ii]] = 1
            sel[n] = ci
            n += 1
    return sel[:n]


def _initial_quadrics(V: np.ndarray, F: np.ndarray):
    nv = len(V)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = 0.5 * np.linalg.norm(n, axis=1)
    nu = n / np.maximum(2.0 * area, 1e-300)[:, None]
    d = -np.einsum("ij,ij->i", nu, V[F[:, 0]])
    QA = np.zeros((nv, 6)); Qb = np.zeros((nv, 3)); Qc = np.zeros(nv); W = np.zeros(nv)
    comp = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for k in range(3):
        idx = F[:, k]
        for j, (a, b) in enumerate(comp):
            QA[:, j] += np.bincount(idx, weights=area * nu[:, a] * nu[:, b], minlength=nv)
        for a in range(3):
            Qb[:, a] += np.bincount(idx, weights=area * d * nu[:, a], minlength=nv)
        Qc += np.bincount(idx, weights=area * d * d, minlength=nv)
        W += np.bincount(idx, weights=area, minlength=nv)
    return QA, Qb, Qc, W


def _compact(V, F, *arrays):
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    newid = np.cumsum(used) - 1
    return (V[used], newid[F].astype(np.int64)) + tuple(a[used] for a in arrays)


def collapse_decimate(V: np.ndarray, F: np.ndarray, h: float, eps: float | None = None,
                      lmax_factor: float = 1.6, lcol_factor: float = 0.9,
                      cos_min: float = 0.6, q_floor: float = 0.05, p0_cos: float = 0.3,
                      sliver_len: float = MIN_EDGE_REJECT_MM, max_rounds: int = 400,
                      verbose: bool = True, size_field=None):
    """Coarsen a closed manifold surface to an edge length of about h.

    h            target edge length (mm). Edges shorter than lcol_factor*h are collapsed
                 where the shape error allows.
    eps          shape error bound (mm, rms distance to the original planes); default 0.06*h.
    size_field   optional function P (n,3) -> local target edge length (n,), for local refinement.
                 The edge length limits and the error bound scale with the local size.
    Returns (V, F, info).
    """
    t0 = time.time()
    V = np.ascontiguousarray(V, float)
    F = np.ascontiguousarray(F, np.int64)
    eps = 0.06 * h if eps is None else float(eps)
    eps_rel = eps / h
    hfun = size_field if size_field is not None else (lambda P: np.full(len(P), h))
    QA, Qb, Qc, W = _initial_quadrics(V, F)
    nf0 = len(F)
    rounds = 0
    phase = 0                       # phase 0: remove edges shorter than sliver_len (link condition only)
    while rounds < max_rounds:
        rounds += 1
        nv = len(V)
        eu, ev, cnt = edge_table(F, nv)
        L = np.linalg.norm(V[eu] - V[ev], axis=1)
        if phase == 0:
            cand = np.flatnonzero(L < sliver_len).astype(np.int64)
            if cand.size == 0:
                phase = 1
                continue
            p_Lmax = np.full(cand.size, 1e30)
            p_eps2 = np.full(cand.size, 1e30)
            p_cos, p_q = p0_cos, 0.0
        else:
            hv = hfun(V)
            hmin = np.minimum(hv[eu], hv[ev])
            cand = np.flatnonzero(L < lcol_factor * hmin).astype(np.int64)
            if cand.size == 0:
                break
            p_Lmax = lmax_factor * hmin[cand]
            p_eps2 = (eps_rel * hmin[cand]) ** 2
            p_cos, p_q = cos_min, q_floor
        adj_ptr, adj_idx = _csr_adjacency(eu, ev, nv)
        vf_ptr, vf_idx = _csr_incidence(F, nv)
        ok = np.zeros(cand.size, np.uint8)
        keep = np.zeros(cand.size, np.int64)
        rem = np.zeros(cand.size, np.int64)
        _eval_edges(V, F, QA, Qb, Qc, W, eu, ev, cand, adj_ptr, adj_idx, vf_ptr, vf_idx,
                    p_Lmax, p_eps2, sliver_len, p_cos, p_q, keep, rem, ok)
        good = np.flatnonzero(ok)
        if good.size == 0:
            if phase == 0:
                phase = 1
                continue
            break
        # shortest edges first; ties by edge id (deterministic)
        order_local = good[np.lexsort((cand[good], L[cand[good]]))]
        # selection works on positions in `cand`, so map eu/ev through cand
        sel = _select_independent(order_local, eu[cand], ev[cand], adj_ptr, adj_idx, nv)
        if sel.size == 0:
            if phase == 0:
                phase = 1
                continue
            break
        k_keep, k_rem = keep[sel], rem[sel]
        remap = np.arange(nv, dtype=np.int64)
        remap[k_rem] = k_keep
        QA[k_keep] += QA[k_rem]; Qb[k_keep] += Qb[k_rem]; Qc[k_keep] += Qc[k_rem]; W[k_keep] += W[k_rem]
        F = remap[F]
        good_f = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[good_f]
        V, F, QA, Qb, Qc, W = _compact(V, F, QA, Qb, Qc, W)
        if verbose and (rounds % 5 == 0 or rounds < 3 or sel.size < 5):
            print(f"    collapse round {rounds:3d} (phase {phase}): {cand.size:6d} candidates, {good.size:6d} valid, "
                  f"{sel.size:6d} collapsed, {len(F):7d} faces", flush=True)
    info = dict(h_mm=h, eps_mm=eps, rounds=rounds, faces_in=nf0, faces_out=len(F),
                seconds=time.time() - t0)
    return V, F, info


# --------------------------------------------------------------------------- #
# Edge flips (Delaunay criterion on nearly coplanar face pairs)
# --------------------------------------------------------------------------- #

@nb.njit(cache=True)
def _select_flips(cand, f1, f2, nf):
    used = np.zeros(nf, np.uint8)
    sel = np.empty(cand.shape[0], np.int64)
    n = 0
    for i in range(cand.shape[0]):
        k = cand[i]
        if used[f1[k]] == 0 and used[f2[k]] == 0:
            used[f1[k]] = 1; used[f2[k]] = 1
            sel[n] = k; n += 1
    return sel[:n]


def flip_edges(V: np.ndarray, F: np.ndarray, max_dihedral_deg: float = 12.0, passes: int = 8):
    """Flip interior edges where that improves the worst triangle quality.

    A flip is made only if the two faces are nearly coplanar (feature edges are
    never touched), the new edge does not exist yet, no triangle flips and the
    smaller of the two triangle qualities increases.
    """
    nv = len(V)
    cos_d = np.cos(np.radians(max_dihedral_deg))
    total = 0
    for _ in range(passes):
        nf = len(F)
        # half edges: face f, local edge i opposite vertex F[f,(i+2)%3]
        he_a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
        he_b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
        he_c = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])        # opposite vertex
        he_f = np.concatenate([np.arange(nf)] * 3)
        lo = np.minimum(he_a, he_b); hi = np.maximum(he_a, he_b)
        key = lo.astype(np.int64) * nv + hi
        order = np.argsort(key, kind="stable")
        ks = key[order]
        pair_start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        pair_cnt = np.diff(np.r_[pair_start, len(ks)])
        pair_start = pair_start[pair_cnt == 2]
        h1 = order[pair_start]; h2 = order[pair_start + 1]
        a = he_a[h1]; b = he_b[h1]; c = he_c[h1]; d = he_c[h2]    # edge (a,b), opposite c and d
        f1 = he_f[h1]; f2 = he_f[h2]
        exist_keys = np.unique(key)
        new_lo = np.minimum(c, d).astype(np.int64); new_hi = np.maximum(c, d)
        new_key = new_lo * nv + new_hi
        pos = np.searchsorted(exist_keys, new_key)
        pos[pos >= len(exist_keys)] = len(exist_keys) - 1
        exists = exist_keys[pos] == new_key
        Pa, Pb, Pc, Pd = V[a], V[b], V[c], V[d]
        n1 = np.cross(Pb - Pa, Pc - Pa)                 # face (a,b,c)
        n2 = np.cross(Pa - Pb, Pd - Pb)                 # face (b,a,d)
        nn1 = np.linalg.norm(n1, axis=1); nn2 = np.linalg.norm(n2, axis=1)
        # reference normal: the normal of the larger old face (slivers have noisy normals)
        nref = np.where((nn1 >= nn2)[:, None], n1 / np.maximum(nn1, 1e-300)[:, None],
                        n2 / np.maximum(nn2, 1e-300)[:, None])
        coplanar = (np.einsum("ij,ij->i", n1, n2) > cos_d * nn1 * nn2) | \
                   (np.minimum(nn1, nn2) < 1e-3 * np.maximum(nn1, nn2))
        # new faces after the flip: (a,d,c) and (b,c,d)
        m1 = np.cross(Pd - Pa, Pc - Pa)
        m2 = np.cross(Pc - Pb, Pd - Pb)
        mm1 = np.linalg.norm(m1, axis=1); mm2 = np.linalg.norm(m2, axis=1)
        no_flip = (mm1 > 1e-12) & (mm2 > 1e-12) \
            & (np.einsum("ij,ij->i", m1, nref) > 0.9 * mm1) \
            & (np.einsum("ij,ij->i", m2, nref) > 0.9 * mm2) \
            & (np.einsum("ij,ij->i", m1, m2) > 0.8 * mm1 * mm2)   # the two new faces must not fold on each other

        def q(P0, P1, P2, nrm):
            l2 = ((P1 - P0) ** 2).sum(1) + ((P2 - P1) ** 2).sum(1) + ((P0 - P2) ** 2).sum(1)
            return 2.0 * np.sqrt(3.0) * nrm / np.maximum(l2, 1e-300)

        q_old = np.minimum(q(Pa, Pb, Pc, nn1), q(Pb, Pa, Pd, nn2))
        q_new = np.minimum(q(Pa, Pd, Pc, mm1), q(Pb, Pc, Pd, mm2))
        good = coplanar & ~exists & no_flip & (q_new > q_old * 1.05 + 1e-6) & (c != d)
        cand = np.flatnonzero(good)
        if cand.size == 0:
            break
        cand = cand[np.argsort(-(q_new[cand] - q_old[cand]), kind="stable")]
        sel = _select_flips(cand, f1, f2, nf)
        if sel.size == 0:
            break
        F = F.copy()
        F[f1[sel]] = np.column_stack([a[sel], d[sel], c[sel]])
        F[f2[sel]] = np.column_stack([b[sel], c[sel], d[sel]])
        total += sel.size
    return F, total


# --------------------------------------------------------------------------- #
# Edge splitting and tangential smoothing (isotropic clean-up)
# --------------------------------------------------------------------------- #

def split_long_edges(V: np.ndarray, F: np.ndarray, Lsplit, max_passes: int = 80):
    """Bisect edges longer than Lsplit at their midpoints (independent faces per pass).
    Lsplit is a number or a function of the edge midpoints (n,3) -> limits (n,).

    Long edges of the CAD STL lie on planar faces (end faces, flats), so the midpoint is on
    the surface. Long thin triangles there would give sliver tets."""
    V = np.asarray(V, float).copy()
    F = np.asarray(F, np.int64).copy()
    total = 0
    for _ in range(max_passes):
        nv, nf = len(V), len(F)
        he_a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
        he_b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
        he_c = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
        he_f = np.concatenate([np.arange(nf)] * 3)
        key = np.minimum(he_a, he_b).astype(np.int64) * nv + np.maximum(he_a, he_b)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        cnt = np.diff(np.r_[start, len(ks)])
        start = start[cnt == 2]
        h1, h2 = order[start], order[start + 1]
        a, b, c, d = he_a[h1], he_b[h1], he_c[h1], he_c[h2]
        f1, f2 = he_f[h1], he_f[h2]
        Lh = np.linalg.norm(V[a] - V[b], axis=1)
        lim = Lsplit(0.5 * (V[a] + V[b])) if callable(Lsplit) else Lsplit
        cand = np.flatnonzero(Lh > lim)
        if cand.size == 0:
            break
        cand = cand[np.argsort(-Lh[cand], kind="stable")]
        sel = _select_flips(cand, f1, f2, nf)
        if sel.size == 0:
            break
        m_idx = nv + np.arange(sel.size)
        V = np.vstack([V, 0.5 * (V[a[sel]] + V[b[sel]])])
        newF = np.vstack([np.column_stack([m_idx, b[sel], c[sel]]),
                          np.column_stack([m_idx, a[sel], d[sel]])])
        F[f1[sel]] = np.column_stack([a[sel], m_idx, c[sel]])
        F[f2[sel]] = np.column_stack([b[sel], m_idx, d[sel]])
        F = np.vstack([F, newF])
        total += sel.size
    return V, F, total


@nb.njit(cache=True)
def _smooth_pass(V, F, vf_ptr, vf_idx, adj_ptr, adj_idx, cos_feat, omega, cos_keep):
    """One Gauss-Seidel sweep of tangential Laplacian smoothing on smooth vertices."""
    nv = V.shape[0]
    moved = 0
    for v in range(nv):
        k = vf_ptr[v + 1] - vf_ptr[v]
        if k < 3:
            continue
        # area-weighted vertex normal
        nx = 0.0; ny = 0.0; nz = 0.0
        for ii in range(vf_ptr[v], vf_ptr[v + 1]):
            f = vf_idx[ii]
            a, b, c = F[f, 0], F[f, 1], F[f, 2]
            ex = (V[b, 1] - V[a, 1]) * (V[c, 2] - V[a, 2]) - (V[b, 2] - V[a, 2]) * (V[c, 1] - V[a, 1])
            ey = (V[b, 2] - V[a, 2]) * (V[c, 0] - V[a, 0]) - (V[b, 0] - V[a, 0]) * (V[c, 2] - V[a, 2])
            ez = (V[b, 0] - V[a, 0]) * (V[c, 1] - V[a, 1]) - (V[b, 1] - V[a, 1]) * (V[c, 0] - V[a, 0])
            nx += ex; ny += ey; nz += ez
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        if nn < 1e-14:
            continue
        nx /= nn; ny /= nn; nz /= nn
        # feature test: every incident face must be close to the vertex normal
        smooth = True
        for ii in range(vf_ptr[v], vf_ptr[v + 1]):
            f = vf_idx[ii]
            a, b, c = F[f, 0], F[f, 1], F[f, 2]
            ex = (V[b, 1] - V[a, 1]) * (V[c, 2] - V[a, 2]) - (V[b, 2] - V[a, 2]) * (V[c, 1] - V[a, 1])
            ey = (V[b, 2] - V[a, 2]) * (V[c, 0] - V[a, 0]) - (V[b, 0] - V[a, 0]) * (V[c, 2] - V[a, 2])
            ez = (V[b, 0] - V[a, 0]) * (V[c, 1] - V[a, 1]) - (V[b, 1] - V[a, 1]) * (V[c, 0] - V[a, 0])
            en = np.sqrt(ex * ex + ey * ey + ez * ez)
            if en < 1e-14 or (ex * nx + ey * ny + ez * nz) < cos_feat * en:
                smooth = False
                break
        if not smooth:
            continue
        cx = 0.0; cy = 0.0; cz = 0.0
        m = adj_ptr[v + 1] - adj_ptr[v]
        for ii in range(adj_ptr[v], adj_ptr[v + 1]):
            w = adj_idx[ii]
            cx += V[w, 0]; cy += V[w, 1]; cz += V[w, 2]
        cx = cx / m - V[v, 0]; cy = cy / m - V[v, 1]; cz = cz / m - V[v, 2]
        dn = cx * nx + cy * ny + cz * nz
        cx -= dn * nx; cy -= dn * ny; cz -= dn * nz
        px = V[v, 0] + omega * cx; py = V[v, 1] + omega * cy; pz = V[v, 2] + omega * cz
        # accept only if no incident triangle turns or collapses
        ok = True
        for ii in range(vf_ptr[v], vf_ptr[v + 1]):
            f = vf_idx[ii]
            a, b, c = F[f, 0], F[f, 1], F[f, 2]
            oa = V[a]; ob = V[b]; oc = V[c]
            if a == v:
                na0 = px; na1 = py; na2 = pz; nb0 = ob[0]; nb1 = ob[1]; nb2 = ob[2]; nc0 = oc[0]; nc1 = oc[1]; nc2 = oc[2]
            elif b == v:
                na0 = oa[0]; na1 = oa[1]; na2 = oa[2]; nb0 = px; nb1 = py; nb2 = pz; nc0 = oc[0]; nc1 = oc[1]; nc2 = oc[2]
            else:
                na0 = oa[0]; na1 = oa[1]; na2 = oa[2]; nb0 = ob[0]; nb1 = ob[1]; nb2 = ob[2]; nc0 = px; nc1 = py; nc2 = pz
            ox = (ob[1] - oa[1]) * (oc[2] - oa[2]) - (ob[2] - oa[2]) * (oc[1] - oa[1])
            oy = (ob[2] - oa[2]) * (oc[0] - oa[0]) - (ob[0] - oa[0]) * (oc[2] - oa[2])
            oz = (ob[0] - oa[0]) * (oc[1] - oa[1]) - (ob[1] - oa[1]) * (oc[0] - oa[0])
            mx = (nb1 - na1) * (nc2 - na2) - (nb2 - na2) * (nc1 - na1)
            my = (nb2 - na2) * (nc0 - na0) - (nb0 - na0) * (nc2 - na2)
            mz = (nb0 - na0) * (nc1 - na1) - (nb1 - na1) * (nc0 - na0)
            no = np.sqrt(ox * ox + oy * oy + oz * oz)
            nn2 = np.sqrt(mx * mx + my * my + mz * mz)
            if nn2 < 1e-14 or (ox * mx + oy * my + oz * mz) < cos_keep * no * nn2:
                ok = False
                break
        if ok:
            V[v, 0] = px; V[v, 1] = py; V[v, 2] = pz
            moved += 1
    return moved


def smooth_surface(V: np.ndarray, F: np.ndarray, sweeps: int = 6, cos_feat: float = 0.94,
                   omega: float = 0.5, cos_keep: float = 0.85) -> np.ndarray:
    """Tangential smoothing of smooth (non-feature) vertices. Feature vertices do not move."""
    V = np.ascontiguousarray(V, float).copy()
    nv = len(V)
    eu, ev, _ = edge_table(F, nv)
    adj_ptr, adj_idx = _csr_adjacency(eu, ev, nv)
    vf_ptr, vf_idx = _csr_incidence(F, nv)
    for _ in range(sweeps):
        _smooth_pass(V, F, vf_ptr, vf_idx, adj_ptr, adj_idx, cos_feat, omega, cos_keep)
    return V


# --------------------------------------------------------------------------- #
# Driver

class SurfaceRejected(UserError):
    """The STL is not a closed oriented manifold, or the prepared surface broke a rule."""


class PreparedSurfaceRejected(SurfaceRejected):
    """The surface that was PREPARED for the mesh size broke a rule (not the input STL). The prepared surface depends on the
    mesh size, so another size can give another result. A caller can name that as a next step."""


class _ShortEdgeRejected(PreparedSurfaceRejected):
    """The prepared surface has an edge under the limit: the one refusal that the retries answer. `original` is the
    report of the surface that was prepared, `surface` is the refused prepared surface `(V, F, steps, report)`."""


class _WeldFailed(Exception):
    """The weld retry cannot give a surface to prepare. The message is the reason."""


class _CleanFailed(Exception):
    """The clean retry (the last retry) cannot give a prepared surface. The message is the reason."""


def surface_deviation(V0: np.ndarray, F0: np.ndarray, V1: np.ndarray, F1: np.ndarray, n: int = 8000) -> dict:
    """Distance (mm) from a deterministic sample of the original vertices to the prepared surface."""
    import trimesh
    m = trimesh.Trimesh(V1, F1, process=False)
    idx = np.random.RandomState(0).choice(len(V0), min(n, len(V0)), replace=False)
    _, d, _ = trimesh.proximity.closest_point(m, V0[idx])
    return dict(samples=int(len(idx)), mean_mm=float(d.mean()), p99_mm=float(np.percentile(d, 99)), max_mm=float(d.max()))


@dataclass
class PreparedSurface:
    V: np.ndarray
    F: np.ndarray
    original: dict
    prepared: dict
    steps: list = field(default_factory=list)


def load_stl_surface(path: str):
    """Read an STL (ascii or binary) and merge duplicate vertices. Returns V (n,3), F (m,3)."""
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.faces, np.int64)
    if len(F) == 0:
        raise SurfaceRejected(f"{path} has no triangles: it is not an STL surface")
    # merge coincident vertices (STL repeats every vertex per face); merge on a 1e-6 mm grid
    key = np.round(V / 1e-6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    V = V[first]
    F = inv[F]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[ok]
    V, F = _compact(V, F)
    return V, F


def _body_euler(F: np.ndarray, nv: int):
    """(body of each vertex, Euler characteristic of each body) of a surface. A vertex that no face uses is in no body (-1).
    The bodies are numbered in the order of their lowest vertex. For a closed oriented body the genus is (2 - chi) / 2."""
    eu, ev, _cnt = edge_table(F, nv)
    n, lab = connected_components(sp.coo_matrix((np.ones(len(eu)), (eu, ev)), shape=(nv, nv)), directed=False)
    used = np.zeros(nv, bool)
    used[F.ravel()] = True
    nv_b = np.bincount(lab[used], minlength=n)
    chi = (nv_b - np.bincount(lab[eu], minlength=n) + np.bincount(lab[F[:, 0]], minlength=n))[nv_b > 0]
    body = np.cumsum(nv_b > 0) - 1
    return np.where(used, body[lab], -1), chi.astype(np.int64)


def _pinch_vertices(F: np.ndarray, nv: int) -> np.ndarray:
    """The vertices of a closed oriented surface whose faces do not make one fan (a vertex where two parts of the surface meet).
    Two corners of a vertex are joined when their faces share an edge at that vertex; a pinch vertex has more than one group."""
    a = F.ravel()                                        # corner c = 3 f + k is the vertex F[f, k]
    b = F[:, [1, 2, 0]].ravel()                          # the far end of the half-edge that starts at corner c
    key = a * nv + b
    order = np.argsort(key)
    opp = order[np.searchsorted(key[order], b * nv + a)]  # the half-edge (b, a) of the other face
    nxt = 3 * (opp // 3) + (opp % 3 + 1) % 3             # its corner at a
    n, lab = connected_components(sp.coo_matrix((np.ones(len(a)), (np.arange(len(a)), nxt)), shape=(len(a), len(a))), directed=False)
    pairs = np.unique(a * n + lab)
    return np.flatnonzero(np.bincount(pairs // n, minlength=nv) > 1)


def _where_not_manifold(V: np.ndarray, F: np.ndarray, rep: dict) -> str:
    """Where a surface is not a closed manifold: the first place of each kind, as text for a refusal (empty when none is found)."""
    eu, ev, cnt = edge_table(F, len(V))
    out = []
    for text, mask in (("open edge", cnt == 1), ("non-manifold edge", cnt > 2)):
        k = np.flatnonzero(mask)
        if len(k):
            out.append(f"first {text} at ({', '.join(f'{x:.4f}' for x in 0.5 * (V[eu[k[0]]] + V[ev[k[0]]]))})")
    if rep["consistently_oriented"] and not len(out):
        pinch = _pinch_vertices(F, len(V))
        if len(pinch):
            out.append(f"first pinch vertex at ({', '.join(f'{x:.4f}' for x in V[pinch[0]])})")
    return f"; {'; '.join(out)}" if out else ""


def weld_short_edges(V: np.ndarray, F: np.ndarray, limit: float, report: dict | None = None, kept_faces: list | None = None,
                     check_folds: bool = True):
    """The surface with every group of vertices that edges under `limit` join welded to one vertex. Returns (V, F, facts).

    An edge collapse cannot remove every short edge. A hair-thin tube (three vertices around a handle of 1e-5 mm, as marching
    cubes makes on a flat face that lies on the node lattice) has no collapse that keeps the link condition, and a flat sliver
    can fail the normal check; the prepared surface then keeps an edge under the limit and `prepare_surface` refuses it. Here
    the vertices that a short edge joins (a connected group) become one vertex at the mean of the group. The faces that get two
    equal vertices go, and so do the faces that occur twice on the same three vertices (a fin of no thickness: what is left of a
    tube). The weld repeats while an edge is under the limit; each pass removes a vertex, so it ends. No other number is used.

    The weld has the radius `limit`: no vertex that is welded may be moved farther than `limit` from where it was, in any pass (a
    vertex whose faces all go counts at the place where its group was welded). A chain of short edges, as the ring of a fine hole of
    1 mm, moves its vertices by more, and the weld is refused: it would fill the hole. The radius is not a bound on the change of the
    surface: a vertex that is on no short edge is a group of one and does not move, also when all its faces go. The weld then removes
    it, as the apex of a spike whose base edges are under `limit`, and the surface changes there by the height of the spike. The facts
    give the volume change and the vertices dropped. The weld does not refuse a change of topology: a hair
    handle that goes is a change of the genus, and it is the purpose of the weld. The facts give the Euler characteristic and the
    genus of each body before and after.

    Raises `_WeldFailed` (the reason in the message) when the surface has no edge under the limit, when the weld leaves no
    triangle, when a vertex moves more than `limit`, when the welded surface is not a closed manifold, when the number of bodies
    changes, when it has more fold edges than the surface had, or when its volume is not positive. `report` is
    `surface_report(V, F)` when the caller has it. `facts` is the step of `PreparedSurface.steps` (`step` is `weld_short_edges`).
    `V` and `F` are as `load_stl_surface` gives them: a face uses every vertex.

    `kept_faces` (optional): a list; if the weld is not refused, the weld appends to it the rows of `F` that the faces of the result
    come from (`Fw[i]` is the face `F[kept_faces[0][i]]` with its vertices welded). The clean retry uses them for its test of the
    face normals.

    `check_folds` (default True): with False the weld does not refuse a result that has more fold edges than the surface had, and the
    caller decides (the clean retry repairs them with `repair_folds`, and refuses if it cannot). Every other rule stays."""
    nv0 = len(V)
    eu, ev, _cnt = edge_table(F, nv0)
    short = np.linalg.norm(V[eu] - V[ev], axis=1) < limit
    n_in = int(short.sum())
    if not n_in:
        raise _WeldFailed(f"the input surface has no edge under {limit:g} mm: there is nothing to weld")
    r0 = report if report is not None else surface_report(V, F)
    body0, chi0 = _body_euler(F, nv0)
    at = np.arange(nv0)           # the vertex of the current surface that each input vertex is in (-1: no face is left on it)
    move = 0.0                    # the largest distance from an input vertex to where its group was welded
    Vw, Fw, passes, welded = V, F, 0, 0
    kept = np.arange(len(F))      # the row of F that each face of Fw comes from
    while short.any():
        passes += 1
        welded += int(short.sum())
        n, label = connected_components(
            sp.coo_matrix((np.ones(int(short.sum())), (eu[short], ev[short])), shape=(len(Vw), len(Vw))), directed=False)
        size = np.bincount(label, minlength=n)
        Vw = np.stack([np.bincount(label, weights=Vw[:, d], minlength=n) / size for d in range(3)], axis=1)
        live = at >= 0
        at = np.where(live, label[at], -1)
        move = max(move, float(np.linalg.norm(Vw[at[live]] - V[live], axis=1).max()))
        Fw = label[Fw]
        proper = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
        Fw, kept = Fw[proper], kept[proper]
        _u, inv, count = np.unique(np.sort(Fw, axis=1), axis=0, return_inverse=True, return_counts=True)
        once = count[inv.reshape(-1)] == 1
        Fw, kept = Fw[once], kept[once]
        if not len(Fw):
            raise _WeldFailed("the weld leaves no triangle")
        used = np.zeros(len(Vw), bool)
        used[Fw.ravel()] = True
        at = np.where((at >= 0) & used[at], (np.cumsum(used) - 1)[at], -1)
        Vw, Fw = _compact(Vw, Fw)
        eu, ev, _cnt = edge_table(Fw, len(Vw))
        short = np.linalg.norm(Vw[eu] - Vw[ev], axis=1) < limit
    if move > limit:
        raise _WeldFailed(f"the weld moves a vertex by {move:.6g} mm, more than the weld radius {limit:g} mm")
    r1 = surface_report(Vw, Fw)
    if not r1["closed_manifold"]:
        raise _WeldFailed(f"the welded surface is not a closed manifold (open edges {r1['open_edges']}, non-manifold edges "
                          f"{r1['nonmanifold_edges']}, oriented {r1['consistently_oriented']}, "
                          f"non-manifold vertices {r1['nonmanifold_vertices']}){_where_not_manifold(Vw, Fw, r1)}")
    if r1["bodies"] != r0["bodies"]:
        raise _WeldFailed(f"the weld changes the number of bodies ({r0['bodies']} to {r1['bodies']})")
    if check_folds and r1["fold_edges"] > r0["fold_edges"]:
        raise _WeldFailed(f"the welded surface has more fold edges ({r1['fold_edges']}) than the input ({r0['fold_edges']})")
    if r1["volume_mm3"] <= 0:
        raise _WeldFailed("the volume of the welded surface is not positive")
    # the bodies one by one: the body of the welded surface that holds the first vertex that is left of each input body
    body1, chi1 = _body_euler(Fw, len(Vw))
    live = np.flatnonzero(at >= 0)
    first = np.full(len(chi0), -1, np.int64)
    first[body0[live][::-1]] = live[::-1]
    chi_after = [None if v < 0 else int(chi1[body1[at[v]]]) for v in first]
    dropped = int((at < 0).sum())
    facts = dict(step="weld_short_edges", limit_mm=float(limit), passes=passes, short_edges_in_input=n_in, edges_welded=welded,
                 vertices_before=nv0, vertices_after=int(len(Vw)), vertices_merged=int(nv0 - dropped - len(Vw)),
                 vertices_dropped=dropped, faces_before=int(len(F)), faces_after=int(len(Fw)),
                 bodies_before=int(r0["bodies"]), bodies_after=int(r1["bodies"]), max_vertex_move_mm=float(move),
                 euler_before=[int(c) for c in chi0], euler_after=chi_after,
                 genus_before=[int((2 - c) // 2) for c in chi0],
                 genus_after=[None if c is None else int((2 - c) // 2) for c in chi_after],
                 volume_before_mm3=float(r0["volume_mm3"]), volume_after_mm3=float(r1["volume_mm3"]),
                 volume_change_mm3=float(r1["volume_mm3"] - r0["volume_mm3"]))
    if kept_faces is not None:
        kept_faces.append(kept)
    return Vw, Fw, facts


def _face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """The normals of the faces, not scaled (their length is twice the area)."""
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


def _turned_faces(V0: np.ndarray, F0: np.ndarray, V1: np.ndarray, F1: np.ndarray) -> int:
    """How many faces have turned over: the normal of face i in (V1, F1) has a negative dot product with the normal of face i in
    (V0, F0). A sign test: it uses no limit. A face of no area (a normal of zero) has the product 0 and does not count."""
    return int((np.einsum("ij,ij->i", _face_normals(V0, F0), _face_normals(V1, F1)) < 0).sum())


def _turned_faces_by_stage(V0: np.ndarray, F0: np.ndarray, V1: np.ndarray, F1: np.ndarray, V2: np.ndarray, F2: np.ndarray) -> tuple[int, int]:
    """`_turned_faces(V0, F0, V2, F2)` with the source of each turn, for a surface that went in two stages, (V0, F0) to (V1, F1) to
    (V2, F2), with the same face i in the three. Returns (turned by the first stage, turned by the second stage): a face that is turned
    in the end counts for the first stage when its normal in (V1, F1) already has a negative dot product with its normal in (V0, F0),
    else for the second. A face that the first stage turned and the second turned back does not count (it is not turned in the end)."""
    n0, n1, n2 = _face_normals(V0, F0), _face_normals(V1, F1), _face_normals(V2, F2)
    final = np.einsum("ij,ij->i", n0, n2) < 0
    first = np.einsum("ij,ij->i", n0, n1) < 0
    return int((final & first).sum()), int((final & ~first).sum())


def _turn_reason(counts: list) -> str:
    """The text of the refusal for turned faces. `counts` is a list of (count, source) with a source such as "the weld of them"; a
    source with a count of 0 is left out."""
    said = " and ".join(f"{src} turns {n} {'face' if n == 1 else 'faces'} over" for n, src in counts if n)
    return f"{said} (the normal of a face that stays has a negative dot product with its normal before the weld)"


def _removed_vertices(V: np.ndarray, F: np.ndarray, Vc: np.ndarray, Fc: np.ndarray) -> tuple[int, float]:
    """(count, largest distance in mm to the surface (Vc, Fc)) of the vertices of (V, F) that have no vertex of (Vc, Fc) at the same
    place: the vertices that were collapsed, welded or dropped. It uses all of them, no sample (`surface_deviation` uses a sample)."""
    import trimesh

    def rows(A):
        return np.ascontiguousarray(A, dtype=np.float64).view(np.dtype((np.void, 24))).ravel()
    gone = np.flatnonzero(~np.isin(rows(V), rows(Vc)))
    if not len(gone):
        return 0, 0.0
    _p, d, _t = trimesh.proximity.closest_point(trimesh.Trimesh(Vc, Fc, process=False), V[gone])
    return int(len(gone)), float(d.max())


def _repair_folds_step(V: np.ndarray, F: np.ndarray, stage: str, repairs: list):
    """`repair_folds` on a surface of the clean retry. Returns (V, F, report, stays) of the repaired surface: `stays[i]` is the row of
    the face of the input that is the face i of the result. Appends its fact to `repairs`: the `stage` (`collapse` or `weld`), the
    fold edges of the surface before the repair (`fold_edges_made`: the ones that the collapse or the weld made and the ones that the
    surface had) and after it (`fold_edges_left`), the vertices that the repair removed, and the faces that stay and that the repair
    turned over (`faces_turned`: a collapse of the repair moves a vertex, and it can turn a face; a fact, not a rule)."""
    F2, before, after, stays = _repair_folds(V, F)
    V2, F2 = _compact(V, F2)
    repairs.append(dict(stage=stage, fold_edges_made=int(before), fold_edges_left=int(after), vertices_removed=int(len(V) - len(V2)),
                        faces_turned=_turned_faces(V, F[stays], V2, F2)))
    return V2, F2, surface_report(V2, F2), stays


def clean_short_edges(V: np.ndarray, F: np.ndarray, limit: float, max_folds: int):
    """The refused prepared surface with its edges under `limit` removed: collapse first, then weld what stays.
    Returns (V, F, facts, report). `facts` is the step of `PreparedSurface.steps` (`step` is `clean_short_edges`), `report` is
    `surface_report` of the result.

    The last retry of `prepare_surface`. The surface preparation collapses the edges under `limit` first, but the steps after the
    collapse (the flips, the tangential smoothing, the fold repair) move vertices and can leave an edge under `limit`. This can occur
    where two tessellations of one wall meet (a bore against a keep-out region that was drawn with another number of segments), or
    where the mesh meets a region face with a step of a few hundredths of a millimetre: the boolean makes hundreds of edges from
    1e-5 to 5e-2 mm, the collapse removes them, and the smoothing pulls two vertices together again. Here the surface that was
    refused is cleaned in two steps, with the same limit and no other number:

    1. The first phase of `collapse_decimate` runs again (the link condition and the normal check; the second phase is off with
       `lcol_factor=0`). It moves a vertex onto its neighbour, so it adds no new position. The sliver phase can turn a face over, and
       that makes fold edges (two faces on top of each other with opposite normals): the fold repair of the first try (`repair_folds`)
       runs on the collapsed surface when it has more fold edges than `max_folds` (F4-27).
    2. The edges that stay (a collapse that the normal check stops, for example at a step in the surface) are welded with the weld of
       `weld_short_edges` and its rules: the radius `limit` for the vertices that are welded, a closed manifold, the same number of
       bodies, a positive volume. The weld is called with `check_folds=False`: the fold edges are judged here, after the fold repair.
       The weld of the refused surface has few edges to join, so it does not make the chains that stop the weld of the input. The weld
       can make fold edges; `repair_folds` runs on the welded surface when it has more fold edges than the surface before the weld
       (F4-27). One more rule: no face that stays may turn over (its normal has a negative dot product with its normal before the
       weld). The test runs on the result, after the fold repair. The reason names the source: the weld, or the fold repair after
       the weld (review MJ1: for the f7c part at 0.977 mm, attempt 2, the fold repair turned the face, with a vertex that moved
       0.19 mm and a cosine of -0.91 between the normals; the weld moves a vertex by 0.05 mm at most). A face that the repair removes
       is not in the result and is not tested.

    The fold repair is the step of the first try, with no new number. If it cannot bring the fold edges down to the allowed number, or
    if it leaves an edge under `limit`, the retry is refused and the reason names the fold edges that the repair left. The steps are in
    the fact `fold_repairs` (the stage, the fold edges before the repair (`fold_edges_made`), the fold edges left, the vertices
    removed, and `faces_turned`: the faces that stay and that the repair turned over). `faces_turned` after the collapse is a fact
    and not a rule: the repair of the first try does not test it either, and a test would refuse f6a (attempt 1: 3 faces turned, the
    cosines -0.17, -0.15 and -1.00; the part meshes at attempt 3).

    Measured (F4-27, revision of this change, one Linux x86_64 host), on the `final.stl` of the f6a phone stand at 0.973 mm: 25 edges
    under the limit, 19 collapsed and 6 welded (largest move 0.0233 mm); the collapse made 6 fold edges and the weld made 3, and the
    fold repair brought both to 0 (removing 2 and 1 vertices); 60 faces removed, volume change -0.0023 mm3 of 8508.9, largest distance
    of a removed vertex to the cleaned surface 0.0049 mm. Before the change this retry was refused for those fold edges.

    The radius bounds the move of the vertices that are welded. A vertex whose faces all go (the apex of a spike whose base edges are
    under `limit`) is removed, and its distance to the cleaned surface is not bounded by `limit`. The facts give the change:
    `removed_vertices` and `removed_vertex_distance_max_mm` are exact (all the vertices of the refused surface that have no vertex
    of the cleaned surface at the same place), and `deviation_from_refused_mm` is a sample.

    Raises `_CleanFailed` (the reason in the message) when the surface has no edge under `limit`, when the weld of the edges that stay
    is refused, when the weld or the fold repair after it turns a face over (the reason names which), when the result is not a closed
    manifold, when it has more fold edges than `max_folds` after the
    fold repair, when it has an edge under `limit`, or when its volume is not positive. `max_folds` is the number of fold edges that
    the refused surface and the input have, whichever is lower."""
    nv0 = len(V)
    eu, ev, _cnt = edge_table(F, nv0)
    n0 = int((np.linalg.norm(V[eu] - V[ev], axis=1) < limit).sum())
    if not n0:
        raise _CleanFailed(f"the prepared surface has no edge under {limit:g} mm: there is nothing to clean")
    Vc, Fc, info = collapse_decimate(V, F, limit, eps=limit, lcol_factor=0.0, sliver_len=limit, verbose=False)
    rep = surface_report(Vc, Fc)
    if not rep["closed_manifold"]:
        raise _CleanFailed(f"the collapsed surface is not a closed manifold (open edges {rep['open_edges']}, non-manifold edges "
                           f"{rep['nonmanifold_edges']}, oriented {rep['consistently_oriented']}, "
                           f"non-manifold vertices {rep['nonmanifold_vertices']}){_where_not_manifold(Vc, Fc, rep)}")
    repairs: list = []
    if rep["fold_edges"] > max_folds:
        made = rep["fold_edges"]
        Vc, Fc, rep, _stays = _repair_folds_step(Vc, Fc, "collapse", repairs)
        if rep["fold_edges"] > max_folds or not rep["closed_manifold"]:
            raise _CleanFailed(f"the collapsed surface has more fold edges ({made}) than the refused surface has ({max_folds}), "
                               f"and the fold repair leaves {rep['fold_edges']}")
    eu, ev, _cnt = edge_table(Fc, len(Vc))
    n1 = int((np.linalg.norm(Vc[eu] - Vc[ev], axis=1) < limit).sum())
    weld = dict(edges_welded=0, vertices_merged=0, vertices_dropped=0, max_vertex_move_mm=0.0, passes=0)
    if n1:
        Vk, Fk, rows, allowed = Vc, Fc, [], rep["fold_edges"]
        stay = f"{n1} of {n0} edges under {limit:g} mm stay after the collapse (shortest {rep['edge_min_mm']:.4f} mm), "
        try:
            Vc, Fc, w = weld_short_edges(Vk, Fk, limit, report=rep, kept_faces=rows, check_folds=False)
        except _WeldFailed as exc:
            raise _CleanFailed(f"{stay}and the weld of them is refused: {exc}") from None
        kept = np.asarray(rows[0])     # the row of Fk for each face of the welded surface
        weld = {k: w[k] for k in weld}
        rep = surface_report(Vc, Fc)   # the weld checked it: closed, the same bodies. The fold edges are checked here.
        Vw, Fw, stays = Vc, Fc, np.arange(len(Fc))
        if rep["fold_edges"] > allowed:
            made = rep["fold_edges"]
            Vc, Fc, rep, stays = _repair_folds_step(Vw, Fw, "weld", repairs)
            if rep["fold_edges"] > allowed or not rep["closed_manifold"]:
                raise _CleanFailed(f"{stay}and the weld of them is refused: the welded surface has more fold edges ({made}) than the "
                                   f"surface before the weld ({allowed}), and the fold repair leaves {rep['fold_edges']}")
        # The faces that stay (the repair can remove a face, and then the face is not in the result). Both the weld and the fold repair
        # move vertices; the reason says which of the two turns a face (F4-27, review MJ1: in f7c the repair did, not the weld).
        by_weld, by_repair = _turned_faces_by_stage(Vk, Fk[kept[stays]], Vw, Fw[stays], Vc, Fc)
        if by_weld or by_repair:
            raise _CleanFailed(f"{stay}and " + _turn_reason([(by_weld, "the weld of them"), (by_repair, "the fold repair after the weld")]))
    if repairs and rep["edge_min_mm"] < limit:
        raise _CleanFailed(f"the fold repair leaves an edge of {rep['edge_min_mm']:.4f} mm, under {limit:g} mm")
    if rep["volume_mm3"] <= 0:
        raise _CleanFailed("the volume of the cleaned surface is not positive")
    v0 = float(np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) / 6.0)
    removed, removed_max = _removed_vertices(V, F, Vc, Fc)
    facts = dict(step="clean_short_edges", limit_mm=float(limit), short_edges_before=n0, short_edges_after_collapse=n1,
                 collapse_rounds=int(info["rounds"]), **weld, vertices_before=nv0, vertices_after=int(len(Vc)),
                 faces_before=int(len(F)), faces_after=int(len(Fc)), volume_before_mm3=v0, volume_after_mm3=float(rep["volume_mm3"]),
                 volume_change_mm3=float(rep["volume_mm3"] - v0), removed_vertices=removed,
                 removed_vertex_distance_max_mm=removed_max, deviation_from_refused_mm=surface_deviation(V, F, Vc, Fc),
                 fold_repairs=repairs)
    return Vc, Fc, facts, rep


def prepare_surface(path: str, h: float, method: str = "collapse", verbose: bool = True,
                    eps_factor: float = 0.1, size_field=None,
                    min_edge_mm: float = MIN_EDGE_REJECT_MM) -> PreparedSurface:
    """STL -> closed manifold surface with edge length about h and no edge shorter than `min_edge_mm`.

    `size_field(P)` (optional) gives a local target edge length for the points P (n, 3).

    method "collapse" (default): split edges longer than 1.6 h, flip, collapse short edges
    under the shape-error bound eps = eps_factor*h, flip, smooth, flip.
    Other values of `method` raise ValueError. (The pipeline had a second method "vtk". It is not
    ported: it needs vtk, and it broke the closed manifold on the real STL of the pipeline.)

    Weld retry (only after one refusal): when the prepared surface still has an edge under `min_edge_mm`, the input surface is
    welded with that same limit (`weld_short_edges`) and prepared again. An input that is prepared at the first try is not
    touched. The step `weld_short_edges` is the first of `PreparedSurface.steps` (the edges welded, the vertices merged, the volume
    change, the bodies before and after).

    Clean retry (F1-5, only after the weld retry did not help): the refused prepared surface of the first try is cleaned with
    `clean_short_edges` (the collapse of the edges under `min_edge_mm` again, then the weld of the edges that stay; no face may turn
    over). The step `clean_short_edges` is the last of `PreparedSurface.steps`; it gives the vertices that the cleaning removed and
    the largest distance of one of them to the cleaned surface, which the weld radius does not bound. No second preparation runs.
    A surface that is prepared at the first try, or after the weld of the input, never reaches this retry.

    When no retry gives a surface that is prepared, the refusal is the first one, and its text says that each retry was tried and
    why it failed."""
    t0 = time.time()
    V, F = load_stl_surface(path)
    kw = dict(method=method, verbose=verbose, eps_factor=eps_factor, size_field=size_field, min_edge_mm=min_edge_mm)
    try:
        return _prepare_arrays(V, F, h, t0, **kw)
    except _ShortEdgeRejected as refused:
        first, original, kept = str(refused), refused.original, refused.surface
    try:
        Vw, Fw, weld = weld_short_edges(V, F, min_edge_mm, report=original)
        weld["refused_first"] = first
        if verbose:
            print(f"  the prepared surface was refused ({first}); retrying with the short edges of the input welded: "
                  f"{weld['vertices_merged']:,} vertices merged, {weld['vertices_dropped']:,} dropped, largest move "
                  f"{weld['max_vertex_move_mm']:.4g} mm, volume change {weld['volume_change_mm3']:+.4g} mm^3", flush=True)
        prep = _prepare_arrays(Vw, Fw, h, t0, **kw)
    except (_WeldFailed, SurfaceRejected) as exc:
        weld_failed = str(exc)
    else:
        prep.original = original
        prep.prepared["deviation_from_input_mm"] = surface_deviation(V, F, prep.V, prep.F)
        prep.steps.insert(0, weld)
        return prep
    # The last retry: the refused surface of the first try, with its edges under the limit cleaned (no full second preparation).
    try:
        Vc, Fc, clean, rep1 = clean_short_edges(kept[0], kept[1], min_edge_mm, min(original["fold_edges"], kept[3]["fold_edges"]))
    except _CleanFailed as exc:
        raise PreparedSurfaceRejected(f"{first}; the weld retry was tried and did not help: {weld_failed}; the clean retry of the "
                                      f"prepared surface did not help: {exc}") from None
    clean["refused_first"] = first
    clean["weld_refused"] = weld_failed
    if verbose:
        print(f"  the prepared surface was refused ({first}) and the weld did not help ({weld_failed}); cleaned its "
              f"{clean['short_edges_before']:,} edges under {min_edge_mm:g} mm ({clean['short_edges_before'] - clean['short_edges_after_collapse']:,} "
              f"collapsed, {clean['vertices_merged']:,} vertices welded): volume change {clean['volume_change_mm3']:+.4g} mm^3", flush=True)
    rep1["deviation_from_input_mm"] = surface_deviation(V, F, Vc, Fc)
    rep1["seconds"] = time.time() - t0
    return PreparedSurface(Vc, Fc, original, rep1, [*kept[2], clean])


def _prepare_arrays(V: np.ndarray, F: np.ndarray, h: float, t0: float, method: str, verbose: bool, eps_factor: float,
                    size_field, min_edge_mm: float) -> PreparedSurface:
    """`prepare_surface` for a loaded surface. Raises `_ShortEdgeRejected` for an edge under `min_edge_mm`."""
    rep0 = surface_report(V, F)
    if verbose:
        print(f"  input surface: {rep0['faces']:,} faces, edge min {rep0['edge_min_mm']:.2e} mm, "
              f"closed manifold {rep0['closed_manifold']}, volume {rep0['volume_mm3']:.1f} mm^3", flush=True)
    if not rep0["closed_manifold"]:
        raise SurfaceRejected(
            f"input surface is not a closed oriented manifold: open edges {rep0['open_edges']}, "
            f"non-manifold edges {rep0['nonmanifold_edges']}, oriented {rep0['consistently_oriented']}, "
            f"non-manifold vertices {rep0['nonmanifold_vertices']}")
    if rep0["volume_mm3"] <= 0:
        raise SurfaceRejected("surface volume is not positive (inverted normals?)")
    steps = []
    if method == "collapse":
        hfun = size_field if size_field is not None else (lambda P: np.full(len(P), h))
        V1, F1, ns = split_long_edges(V, F, lambda P: 1.6 * hfun(P))
        steps.append(dict(step="split_long_edges", limit_mm=1.6 * h, splits=int(ns), faces=len(F1),
                          local_refinement=size_field is not None))
        F1, nfl = flip_edges(V1, F1)
        steps.append(dict(step="flip_edges", flips=int(nfl)))
        if verbose:
            print(f"  split {ns:,} long edges, {nfl:,} flips; collapsing from {len(F1):,} faces", flush=True)
        V2, F2, info = collapse_decimate(V1, F1, h, eps=eps_factor * h, verbose=verbose, size_field=size_field,
                                         sliver_len=min_edge_mm)
        steps.append(dict(step="collapse_decimate", **info))
        F2, nfl = flip_edges(V2, F2)
        V2 = smooth_surface(V2, F2)
        F2, nfl2 = flip_edges(V2, F2)
        steps.append(dict(step="flip_smooth_flip", flips=int(nfl + nfl2)))
        F2, f_before, f_after = repair_folds(V2, F2, verbose=verbose)
        V2, F2 = _compact(V2, F2)
        steps.append(dict(step="repair_folds", folds_before=int(f_before), folds_after=int(f_after)))
        if f_before and verbose:
            print(f"  fold repair: {f_before} -> {f_after} fold edges", flush=True)
    else:
        raise ValueError(method)
    rep1 = surface_report(V2, F2)
    rep1["deviation_from_input_mm"] = surface_deviation(V, F, V2, F2)
    rep1["seconds"] = time.time() - t0
    if verbose:
        print(f"  prepared surface: {rep1['faces']:,} faces, edge min/p1/median/mean/max {rep1['edge_min_mm']:.3f}/"
              f"{rep1['edge_p1_mm']:.3f}/{rep1['edge_median_mm']:.2f}/{rep1['edge_mean_mm']:.2f}/{rep1['edge_max_mm']:.2f} mm, "
              f"tri quality min/p1/median {rep1['tri_quality_min']:.3f}/{rep1['tri_quality_p1']:.3f}/"
              f"{rep1['tri_quality_median']:.3f}, closed manifold {rep1['closed_manifold']}, fold edges {rep1['fold_edges']}, "
              f"volume change {rep1['volume_mm3'] / rep0['volume_mm3'] - 1:+.3%}, distance of input vertices to the "
              f"prepared surface mean/p99/max {rep1['deviation_from_input_mm']['mean_mm']:.3f}/"
              f"{rep1['deviation_from_input_mm']['p99_mm']:.3f}/{rep1['deviation_from_input_mm']['max_mm']:.3f} mm, "
              f"{rep1['seconds']:.0f} s", flush=True)
    if not rep1["closed_manifold"]:
        raise PreparedSurfaceRejected(f"prepared surface is not a closed manifold: open edges {rep1['open_edges']}, "
                                      f"non-manifold edges {rep1['nonmanifold_edges']}, "
                                      f"non-manifold vertices {rep1['nonmanifold_vertices']}")
    if rep1["fold_edges"] > rep0["fold_edges"]:
        raise PreparedSurfaceRejected(f"prepared surface has {rep1['fold_edges']} fold edges (input: {rep0['fold_edges']})")
    if rep1["edge_min_mm"] < min_edge_mm:
        refused = _ShortEdgeRejected(f"prepared surface has an edge of {rep1['edge_min_mm']:.4f} mm "
                                     f"(limit {min_edge_mm} mm)")
        refused.original = rep0
        refused.surface = (V2, F2, steps, rep1)
        raise refused
    return PreparedSurface(V2, F2, rep0, rep1, steps)
