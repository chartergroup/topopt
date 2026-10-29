"""Von Mises stress and the volume-weighted statistics of a P2 solution. CONTRACTS.md section 16.

Ported from `pipeline/lib/p2_core.py` (`von_mises`, `weighted_percentile`) and `stress_report` in
`pipeline/s5_verify/p2_verify.py` (revision d43bcf3), without the code of that project. `tet_quality` is the
function of the tet mesher (`topopt.tet.mesher.tet_quality`): one copy only.

Numbers of the report (all in MPa when E is in MPa):

- `peak_qp_MPa`: the largest von Mises value at the 14 quadrature points of any tet.
- `peak_vertex_MPa`: the largest value at the 4 vertices of any tet. The stress of a P2 tet is linear,
  so this is the largest value in the tet. There is no averaging over the tets of a node.
- `p50`, `p95`, `p99`, `p99_9` ("by volume"): percentiles of the von Mises values at the quadrature
  points, weighted by the volume that each point stands for. The percentile is a step function: it
  is the first sample at or above the weight. A small region of high stress does not move it.
- The same peak and percentile without the tets of quality below 0.05 (`SLIVER_QUALITY`).
- `peak_qp_over_p99_9`: the peak over p99.9. A value far above 1 means a small region of high stress.
- `near_regions` (with the region masks): the p99.9 and the peak without the points within a distance of a fixed or a
  load region, and the share of the top 0.1 % volume near each region. On the hook, the load patch carries most of it.
"""
from __future__ import annotations

import numpy as np

from ..tet.mesher import SLIVER_QUALITY, tet_quality
from .p2 import P2Model

__all__ = ["SLIVER_QUALITY", "near_region_stats", "stress_report", "tet_quality", "von_mises", "weighted_percentile"]


def von_mises(sig: np.ndarray) -> np.ndarray:
    """Von Mises stress of Voigt stress (..., 6) in the order xx, yy, zz, xy, yz, xz."""
    sx, sy, sz, txy, tyz, tzx = (sig[..., k] for k in range(6))
    return np.sqrt(np.maximum(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                              + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2), 0.0))


def weighted_percentile(values: np.ndarray, weights: np.ndarray, pct) -> np.ndarray:
    """Percentiles of `values` weighted by `weights` (volume), step-wise (no interpolation).

    `pct` is a number or a list in percent (0 to 100). Return an array.
    """
    v = np.asarray(values).ravel()
    w = np.asarray(weights, float).ravel()
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    pct = np.atleast_1d(np.asarray(pct, float)) / 100.0
    idx = np.minimum(np.searchsorted(cw, pct, side="left"), len(v) - 1)
    return v[idx]


def near_region_stats(vm: np.ndarray, wv: np.ndarray, p999: float, masks: dict, dist_mm: float,
                      percentile: float = 99.9) -> dict:
    """The stress numbers without the points near the regions, and where the top 0.1 % of the volume is.

    `vm` (nel, nq) is the von Mises stress at the quadrature points, `wv` the volume of each point, `p999` the
    p99.9 of all points and `masks` a dict `region name -> (nel, nq) bool` (near that region, see
    `regions.near_region_mask`). The top volume is the points at or above `p999`: `top_volume_share_near` gives, for
    each region, the share of that volume (by volume) that is near it. `any` is near at least one region and `none`
    is the rest. A point can be near two regions, so the shares need not add up to 1.
    """
    any_near = np.zeros(vm.shape, bool)
    for m in masks.values():
        any_near |= m
    top = vm >= p999
    top_w = float(wv[top].sum())
    share = {name: (float(wv[top & m].sum() / top_w) if top_w > 0 else None) for name, m in masks.items()}
    share["any"] = float(wv[top & any_near].sum() / top_w) if top_w > 0 else None
    share["none"] = None if share["any"] is None else 1.0 - share["any"]
    far = ~any_near
    rec = dict(distance_mm=float(dist_mm), regions=list(masks),
               volume_fraction_excluded=float(wv[any_near].sum() / wv.sum()),
               top_volume_share_near=share,
               top_volume_definition="quadrature points with a von Mises value at or above the p99.9 of all points, weighted by volume")
    if far.any():
        pct = weighted_percentile(vm[far], wv[far], [99.9, percentile])
        rec.update(p99_9_by_volume_MPa_excluding_near_regions=float(pct[0]),
                   percentile_by_volume_MPa_excluding_near_regions=float(pct[1]),
                   peak_qp_MPa_excluding_near_regions=float(vm[far].max()))
    else:
        rec.update(p99_9_by_volume_MPa_excluding_near_regions=None,
                   percentile_by_volume_MPa_excluding_near_regions=None, peak_qp_MPa_excluding_near_regions=None)
    return rec


def stress_report(model: P2Model, u: np.ndarray, E: float, nu: float, quality: np.ndarray | None = None,
                  percentile: float = 99.9, top_n: int = 10, near_masks: dict | None = None,
                  near_mm: float = 0.0):
    """Stress numbers of one load case. Return `(report, sig, elem_max)`.

    `report` is a dict of plain numbers (see the module text). `sig` is the stress at the quadrature
    points (nel, nq, 6). `elem_max` (nel,) float32 is the largest von Mises value at the quadrature
    points of each tet. `quality` (nel,) is the tet quality; with it, the report has the top
    elements and the numbers without the tets of quality below `SLIVER_QUALITY`.
    `percentile` is the percentile `p_user` that the report adds to p50, p95, p99 and p99.9.
    `near_masks` (dict `region name -> (nel, nq) bool`) adds the block `near_regions` (see `near_region_stats`);
    `near_mm` is the distance that the masks stand for.
    """
    sig, lam, w = model.stress_qp(u, E, nu)
    vm = von_mises(sig)                                          # (nel,nq)
    wv = w[None, :] * model.vol[:, None]
    e, q = np.unravel_index(int(np.argmax(vm)), vm.shape)
    xyz = model.qp_points(lam[q:q + 1], np.array([e]))[0, 0]
    pct = weighted_percentile(vm, wv, [50.0, 95.0, 99.0, 99.9, percentile])
    vmv = von_mises(model.stress_at_vertices(u, E, nu))
    ev, qv = np.unravel_index(int(np.argmax(vmv)), vmv.shape)
    xyzv = model.p[:, model.t[qv, ev]]
    rec = dict(
        peak_qp_MPa=float(vm[e, q]), peak_qp_xyz_mm=[float(x) for x in xyz], peak_qp_element=int(e),
        p50_by_volume_MPa=float(pct[0]), p95_by_volume_MPa=float(pct[1]),
        p99_by_volume_MPa=float(pct[2]), p99_9_by_volume_MPa=float(pct[3]),
        percentile=float(percentile), percentile_by_volume_MPa=float(pct[4]),
        peak_vertex_MPa=float(vmv[ev, qv]), peak_vertex_xyz_mm=[float(x) for x in xyzv],
        quadrature_points_per_element=int(vm.shape[1]),
    )
    rec["peak_qp_over_p99_9"] = float(vm[e, q] / pct[3]) if pct[3] > 0 else None
    if near_masks:
        rec["near_regions"] = near_region_stats(vm, wv, float(pct[3]), near_masks, near_mm, percentile)
    elem_max = vm.max(axis=1).astype(np.float32)
    if quality is not None:
        cent = model.centroids()
        top = []
        for e_ in np.argsort(-elem_max)[:top_n]:
            top.append(dict(element=int(e_), max_vm_qp_MPa=float(elem_max[e_]),
                            centroid_mm=[float(x) for x in cent[e_]],
                            tet_quality=float(quality[e_]), tet_volume_mm3=float(model.vol[e_])))
        rec["top_elements"] = top
        good = quality >= SLIVER_QUALITY
        rec["tets_below_quality_cut"] = int((~good).sum())
        if good.any():
            vmv_good = vmv[good]
            rec["peak_qp_MPa_excluding_bad_tets"] = float(elem_max[good].max())
            rec["peak_vertex_MPa_excluding_bad_tets"] = float(vmv_good.max())
            ex = weighted_percentile(vm[good], wv[good], [99.9, percentile])
            rec["p99_9_by_volume_MPa_excluding_bad_tets"] = float(ex[0])
            rec["percentile_by_volume_MPa_excluding_bad_tets"] = float(ex[1])
    return rec, sig, elem_max
