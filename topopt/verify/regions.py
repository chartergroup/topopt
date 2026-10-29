"""Which mesh nodes and facets belong to a region STL. CONTRACTS.md section 16, "Region rule".

A point belongs to a region when it is inside the region STL, or within a band of its surface.
The signed distance is positive inside. So the rule is: signed distance >= -band.

Why a band: the final surface of a run is a marching-cubes surface. It lies up to half a voxel
from the wall of a region, and it can lie 1e-5 mm outside it on a plane. Measured on the hook
(`final.stl`, voxel 1.2 mm): strictly inside catches 1,615 of the plate vertices, and a band of
0.5 voxel catches 14,114 (a band of 1 voxel: 14,887). The default band is 0.5 times the voxel size
of the run. The report holds the counts for 0, 0.5 and 1 times the band, so you see the effect.

- A fixed region: every P2 node (vertex or edge midpoint) that belongs to the region.
- A load region: every boundary facet of the tet mesh whose 3 vertices all belong to the region.
- A measure region (role `measure`, the A5 rows of `regions.measure`, B3): every P2 node that belongs to the region, as for a
  fixed region, and no facet is needed. It can lie inside the part.
- A keep-out region is not used here: it keeps material out, and the final part has none there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import EmptyRegion
from .names import opt

# How many mesh points to test exactly when a region is empty (to report the nearest distance).
NEAREST_SAMPLE = 2000

# The roles that need at least one node. A `solid` or `keep_out` region is read only for the facts of a removed body (`topopt verify --part-only`,
# CONTRACTS.md section 16): it gives no boundary condition, so its caller always passes `allow_empty`.
NODE_ROLES = ("fixed", "measure", "solid", "keep_out")


def boundary_facets(tets: np.ndarray, return_elems: bool = False):
    """The facets of a tet mesh that belong to one tet only: (k, 3) vertex indices, each row sorted.

    `tets` is (m, 4). A facet shared by two tets is inside the part.
    With `return_elems`, also return the index of the tet of each facet, (k,).
    """
    tets = np.asarray(tets, np.int64)
    f = tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
    owner = np.repeat(np.arange(len(tets)), 4)
    f = np.sort(f, axis=1)
    order = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
    fs = f[order]
    same = np.all(fs[1:] == fs[:-1], axis=1)
    shared = np.zeros(len(fs), bool)
    shared[1:] |= same
    shared[:-1] |= same
    if return_elems:
        return fs[~shared], owner[order][~shared]
    return fs[~shared]


def outward_normals(nodes: np.ndarray, tets: np.ndarray, facets: np.ndarray, elems: np.ndarray) -> np.ndarray:
    """Unit normals (k, 3) of boundary facets, pointing out of the part.

    `nodes` (n, 3) are the vertices, `tets` (m, 4) the tets, `facets` (k, 3) the boundary facets and `elems` (k,) the
    tet of each facet (`boundary_facets(..., return_elems=True)`). The side is the one away from the 4th vertex of the tet.
    A facet with no area gets a zero vector.
    """
    P = nodes[facets]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln > 0, ln, 1.0)[:, None]
    tv = np.asarray(tets, np.int64)[elems]
    opposite = np.array([[v for v in row if v not in set(f)][0] for row, f in zip(tv.tolist(), facets.tolist())]) \
        if len(facets) else np.zeros(0, np.int64)
    inward = nodes[opposite] - P[:, 0] if len(facets) else np.zeros((0, 3))
    sign = np.where(np.einsum("kd,kd->k", n, inward) > 0, -1.0, 1.0)
    return n * sign[:, None]


def normal_coherence(normals: np.ndarray, areas: np.ndarray) -> dict:
    """How much the facets of a load region face one way: `|sum(a n)| / sum(a)` (1 for a flat patch, near 0 for opposite faces).

    Return `coherence` and the area-weighted `mean_normal` (a unit vector, or zeros if the facets cancel).
    A low value means that the region took facets of different sides (for example both faces of a thin wall).
    """
    total = float(np.sum(areas))
    if total <= 0 or not len(areas):
        return dict(coherence=None, mean_normal=[0.0, 0.0, 0.0])
    v = (normals * areas[:, None]).sum(axis=0)
    c = float(np.linalg.norm(v) / total)
    mean = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.zeros(3)
    return dict(coherence=c, mean_normal=[float(x) for x in mean])


def facet_areas(nodes: np.ndarray, facets: np.ndarray) -> np.ndarray:
    """Area of each triangle (k,) for the vertex coordinates `nodes` (n, 3)."""
    P = nodes[facets]
    return 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)


def load_region_mesh(path):
    """Read one region STL as a merged, closed `trimesh.Trimesh` (the loader of the voxelizer)."""
    from ..voxelize import load_mesh
    return load_mesh(Path(path))


def signed_distance(region_mesh, points: np.ndarray, reach: float):
    """Signed distance of points (n, 3) to the region surface, positive inside.

    Points farther than `reach` from the bounding box of the region get `-inf` (they are not tested
    exactly: they are outside the band). Return `(sd, tested)` where `tested` is the boolean mask of the tested points.
    """
    import trimesh
    lo, hi = region_mesh.bounds[0] - reach, region_mesh.bounds[1] + reach
    tested = np.all((points >= lo) & (points <= hi), axis=1)
    sd = np.full(len(points), -np.inf)
    idx = np.flatnonzero(tested)
    if len(idx):
        sd[idx] = trimesh.proximity.signed_distance(region_mesh, points[idx])
    return sd, tested


def _nearest_distance(region_mesh, points: np.ndarray) -> float:
    """The smallest distance (mm) from `points` to the region (0 or less if a point is inside)."""
    lo, hi = region_mesh.bounds
    gap = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    lb = np.linalg.norm(gap, axis=1)
    pick = np.argsort(lb)[:NEAREST_SAMPLE]
    import trimesh
    sd = trimesh.proximity.signed_distance(region_mesh, points[pick])
    return float(-sd.max())


@dataclass
class Selection:
    """The nodes and facets of one region. `facts` is the plain dict for `verify.json`.

    `sd` is the signed distance of every P2 node to the region surface (positive inside, `-inf` far away) and `mesh`
    is the region surface (`trimesh.Trimesh`). The later steps use them (distance to a region), they are not written.
    """
    name: str
    role: str                      # "fixed", "load" or "measure" ("solid" and "keep_out" for the facts of a removed body)
    band_mm: float
    node_mask: np.ndarray          # (nn,) bool, over the P2 nodes (vertices, then edge midpoints)
    facets: np.ndarray             # (k, 3) selected boundary facets (vertex indices)
    facet_area: np.ndarray         # (k,) mm^2
    facts: dict = field(default_factory=dict)
    sd: np.ndarray | None = None
    mesh: object | None = None
    empty: bool = False
    elems: np.ndarray | None = None  # (k,) the tet of each selected facet, when `select_region` got `belems` (K1-A: the outward normals of the bonded facets)


def _centroid(model, sel_nodes, facets, areas):
    if len(facets):
        P = model.p.T[facets].mean(axis=1)
        return (P * areas[:, None]).sum(axis=0) / areas.sum()
    return model.node_xyz.T[sel_nodes].mean(axis=0)


def select_region(model, bfacets: np.ndarray, region_path, name: str, role: str, band_mm: float, *,
                  belems: np.ndarray | None = None, region_mesh=None, sd_cache: tuple | None = None,
                  allow_empty: bool = False) -> Selection:
    """Select the nodes and boundary facets of the region STL `region_path`.

    `model` is the `P2Model`, `bfacets` the boundary facets of its mesh (`boundary_facets`), `role` is
    "fixed" or "measure" (each needs at least one node) or "load" (needs at least one facet). The roles "solid" and "keep_out"
    are for the facts of a removed body (`part_only`): they count nodes like "fixed", and give no boundary condition.
    Raise `EmptyRegion` if the region matches nothing. The text has the region, the band and the nearest distance.
    With `allow_empty`, return a `Selection` with `empty` True instead (for a sensitivity study at another band).
    `belems` (the tet of each boundary facet) adds the normal check (`normal_coherence`) to the facts of a load region.
    `region_mesh` and `sd_cache` (`(sd, tested)` computed for a band at least as large) skip the work that repeats
    when the same region is selected again at another band.
    """
    mesh = region_mesh if region_mesh is not None else load_region_mesh(region_path)
    X = model.node_xyz.T
    sd, tested = sd_cache if sd_cache is not None else signed_distance(mesh, X, band_mm)
    nv = model.nv
    verts = model.p.T
    bands = {"0": 0.0, "0.5x": 0.5 * band_mm, "1x": band_mm}

    def facet_mask(b):
        return np.all(sd[:nv][bfacets] >= -b, axis=1) if len(bfacets) else np.zeros(0, bool)

    node_mask = sd >= -band_mm
    fmask = facet_mask(band_mm)
    facets = bfacets[fmask]
    areas = facet_areas(verts, facets) if len(facets) else np.zeros(0)
    counts = {k: {"nodes": int((sd >= -b).sum()), "facets": int(facet_mask(b).sum())} for k, b in bands.items()}
    ids = np.flatnonzero(node_mask)
    empty = (len(ids) == 0) if role in NODE_ROLES else (len(facets) == 0)
    if empty and allow_empty:
        facts = dict(region=name, role=role, file=str(region_path), band_mm=float(band_mm), empty=True,
                     nodes=int(len(ids)), facets=0, area_mm2=0.0, counts_by_band=counts)
        return Selection(name, role, float(band_mm), node_mask, facets, areas, facts, sd, mesh, True)
    if empty:
        if role in NODE_ROLES:
            nearest = _nearest_distance(mesh, X)
            what = "no mesh node"
        else:
            Vd = np.where(np.isfinite(sd[:nv]), sd[:nv], -1e9)
            if len(bfacets) and np.any(tested[:nv]):
                nearest = float(-Vd[bfacets].min(axis=1).max())    # the best facet: its worst vertex
                if not np.isfinite(nearest) or nearest > 1e8:
                    nearest = _nearest_distance(mesh, verts)
            else:
                nearest = _nearest_distance(mesh, verts)
            what = "no boundary facet with all 3 vertices in the region"
        facts = dict(region=name, role=role, file=str(region_path), band_mm=float(band_mm),
                     nearest_distance_mm=float(nearest), counts=counts)
        raise EmptyRegion(
            f"{role} region '{name}' ({Path(region_path).name}) matches {what}. The nearest mesh point is "
            f"{nearest:.3g} mm from the region (band {band_mm:g} mm). Check that the STL files and the run are in "
            f"the same frame (mm). You can also set {opt('region_band')}.", facts)
    facts = dict(
        region=name, role=role, file=str(region_path), band_mm=float(band_mm),
        nodes=int(len(ids)), vertices=int(node_mask[:nv].sum()), facets=int(len(facets)),
        area_mm2=float(areas.sum()),
        centroid_mm=[float(x) for x in _centroid(model, ids, facets, areas)],
        counts_by_band=counts,
    )
    if role == "load" and belems is not None and len(facets):
        nrm = outward_normals(verts, model.t.T, facets, belems[fmask])
        facts["normal_check"] = dict(
            **normal_coherence(nrm, areas),
            note="|sum(a n)| / sum(a) over the selected facets, with outward normals: 1 is one flat side, "
                 "a low value means facets of different sides (for example both faces of a thin wall)")
    return Selection(name, role, float(band_mm), node_mask, facets, areas, facts, sd, mesh,
                     elems=None if belems is None else belems[fmask])


def surface_samples(region_mesh, spacing: float, max_points: int = 400_000) -> np.ndarray:
    """Points on the surface of a region: its vertices and a random sample (density about 2 per spacing^2)."""
    import trimesh
    n = int(min(max_points, max(1000, 2.0 * float(region_mesh.area) / spacing ** 2)))
    try:
        pts, _ = trimesh.sample.sample_surface(region_mesh, n, seed=0)
    except TypeError:                                         # an older trimesh has no seed
        pts, _ = trimesh.sample.sample_surface(region_mesh, n)
    return np.vstack([np.asarray(region_mesh.vertices, float), np.asarray(pts, float)])


def near_region_mask(model, sel: Selection, points: np.ndarray, dist_mm: float) -> np.ndarray:
    """Which points are within `dist_mm` of a region, or inside it: a boolean array like `points[..., 0]`.

    `points` is (nel, nq, 3) (quadrature points, `P2Model.qp_points`). The distance is to a sample of the region
    surface (spacing at most 0.5 mm and a quarter of `dist_mm`), so it is exact to about that spacing. A tet with
    a node inside the region (signed distance >= 0) counts as inside, for all its points.
    """
    from scipy.spatial import cKDTree
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    lo, hi = sel.mesh.bounds[0] - dist_mm, sel.mesh.bounds[1] + dist_mm
    cand = np.flatnonzero(np.all((flat >= lo) & (flat <= hi), axis=1))
    near = np.zeros(len(flat), bool)
    if len(cand):
        tree = cKDTree(surface_samples(sel.mesh, min(0.5, dist_mm / 4.0)))
        d, _ = tree.query(flat[cand])
        near[cand] = d <= dist_mm
    near = near.reshape(shape)
    inside = (sel.sd[model.enodes] >= 0).any(axis=1)
    return near | inside[:, None]
