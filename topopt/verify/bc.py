"""Loads, supports and the equilibrium check of a P2 solution. CONTRACTS.md section 16.

Traction rule: a force `F` (N) on a load region is a traction `F / A` (N/mm^2, uniform in space) on the
boundary facets of the region, where `A` is their total area. The nodal forces are the consistent
forces of a straight-sided P2 facet: a facet of area `a` gives `F a / (3 A)` to each of its 3 edge
midpoints and nothing to its vertices (the integral of a P2 vertex function over a triangle is 0, and
of an edge function it is a/3). The resultant is `F`; the relative error is reported.
Supports: every P2 node of a fixed region has all 3 dofs fixed.

The equilibrium check (`equilibrium`) tests the stiffness matrix and the solve, not the load. The reactions are
computed from the same load vector, so a load that is twice too large gives the same small residual. The
check on the load is `resultant_error` (the resultant against the YAML force) and `load_centre` (where it acts).
"""
from __future__ import annotations

import numpy as np

from .regions import facet_areas


def traction_load(model, facets: np.ndarray, force, area: float | None = None):
    """The nodal force vector (N,) of the uniform traction that sums to `force` over `facets`.

    `facets` (k, 3) are vertex indices. Return `(f, info)`; `info` has `area_mm2`, `traction_MPa`
    (3 values, N/mm^2 = MPa) and `resultant_N` (the sum of `f`, 3 values).
    """
    F = np.asarray(force, float)
    a = facet_areas(model.p.T, facets)
    total = float(a.sum()) if area is None else float(area)
    if not total > 0.0:
        raise ValueError("the load region has no area")
    w = a / (3.0 * total)                                     # share of F for each edge node of the facet
    ids = np.stack([model.edge_node_ids(facets[:, 0], facets[:, 1]),
                    model.edge_node_ids(facets[:, 1], facets[:, 2]),
                    model.edge_node_ids(facets[:, 0], facets[:, 2])], axis=1)      # (k, 3)
    f = np.zeros(model.N)
    for c in range(3):
        np.add.at(f, 3 * ids.ravel() + c, np.repeat(w * F[c], 3))
    res = f.reshape(-1, 3).sum(axis=0)
    return f, dict(area_mm2=total, traction_MPa=[float(x) for x in F / total], resultant_N=[float(x) for x in res])


def fixed_dofs(node_ids: np.ndarray) -> np.ndarray:
    """All 3 dofs of each node in `node_ids`, sorted and unique."""
    ids = np.unique(np.asarray(node_ids, np.int64))
    return (3 * ids[:, None] + np.arange(3)[None, :]).ravel()


def resultant_error(resultant, force) -> float:
    """Relative difference of the applied resultant and the wanted force: |R - F| / |F| (0 if F is 0 and R is 0)."""
    F = np.asarray(force, float)
    d = float(np.linalg.norm(np.asarray(resultant, float) - F))
    n = float(np.linalg.norm(F))
    return d / n if n > 0 else d


def equilibrium(model, K, u: np.ndarray, f: np.ndarray, fixed: np.ndarray, support: dict | None = None) -> dict:
    """Force and moment balance of the solution: the applied loads against the support reactions.

    The reaction at a fixed dof is `R = (K u - f)` there (the node equilibrium `K u = f + R`).
    Moments are about the centroid of the mesh nodes. `force_relative_residual` is
    |applied + reactions| / |applied|, and `moment_relative_residual` is the same for the moments.
    A large value means the stiffness matrix or the solve is wrong. It cannot find a wrong load: the reaction
    comes from the same load vector, so a load of twice the size gives the same residual (see the module text).

    With joints and springs (`supports_p2`, K1-A) `K` is `K + Kb` (springs and bonds). `support` is `{"Cq": C q, "force": -(Kb u + C q)}`
    (all dofs): the bond couples to the body, so the reaction at a fixed dof is `(K + Kb) u + C q - f`, and the force of the supports on
    the part is added to the balance. The result then has the keys `support_force_N` and `support_moment_Nmm`.
    """
    Ku = K @ u
    if support is not None:
        Ku = Ku + support["Cq"]
    R = np.zeros(model.N)
    R[fixed] = Ku[fixed] - f[fixed]
    X = model.node_xyz.T
    o = model.p.mean(axis=1)
    Fv, Rv = f.reshape(-1, 3), R.reshape(-1, 3)
    F_app, F_r = Fv.sum(axis=0), Rv.sum(axis=0)
    M_app = np.cross(X - o, Fv).sum(axis=0)
    M_r = np.cross(X - o, Rv).sum(axis=0)
    rf, rm = F_app + F_r, M_app + M_r
    extra = {}
    if support is not None:
        Sv = support["force"].reshape(-1, 3)
        F_s, M_s = Sv.sum(axis=0), np.cross(X - o, Sv).sum(axis=0)
        rf, rm = rf + F_s, rm + M_s
        extra = dict(support_force_N=[float(x) for x in F_s], support_moment_Nmm=[float(x) for x in M_s])
    return dict(
        applied_force_N=[float(x) for x in F_app], reaction_force_N=[float(x) for x in F_r],
        applied_moment_Nmm=[float(x) for x in M_app], reaction_moment_Nmm=[float(x) for x in M_r],
        residual_force_N=[float(x) for x in rf], residual_moment_Nmm=[float(x) for x in rm],
        force_relative_residual=float(np.linalg.norm(rf) / max(np.linalg.norm(F_app), 1e-300)),
        moment_relative_residual=float(np.linalg.norm(rm) / max(np.linalg.norm(M_app), 1e-300)),
        origin_mm=[float(x) for x in o], **extra)


def load_centre(p2_centroid, optimizer_center, force, reference) -> dict:
    """Where a force acts in the P2 model and in the optimizer, and the moment about a reference point.

    `p2_centroid` is the area-weighted centroid of the load facets (the point of a uniform traction), `optimizer_center`
    the `center` of the force in `run.json` (the centre of the load voxels), `force` the force vector (N) and
    `reference` the point for the moment (the area-weighted centroid of the fixed regions). Return a dict of facts:
    the two centres, their distance, the moments `r x F` and `moment_ratio = |M_p2| / |M_optimizer|`
    (None if the optimizer centre or its moment is not known).
    """
    c = np.asarray(p2_centroid, float)
    ref = np.asarray(reference, float)
    F = np.asarray(force, float)
    out = dict(p2_traction_centroid_mm=[float(x) for x in c], reference_mm=[float(x) for x in ref],
               moment_p2_Nmm=[float(x) for x in np.cross(c - ref, F)])
    if optimizer_center is None:
        return dict(out, optimizer_center_mm=None, distance_mm=None, moment_ratio=None)
    o = np.asarray(optimizer_center, float)
    m_p2, m_opt = np.cross(c - ref, F), np.cross(o - ref, F)
    n_opt = float(np.linalg.norm(m_opt))
    return dict(out, optimizer_center_mm=[float(x) for x in o], distance_mm=float(np.linalg.norm(c - o)),
                moment_optimizer_Nmm=[float(x) for x in m_opt],
                moment_ratio=float(np.linalg.norm(m_p2) / n_opt) if n_opt > 0 else None)
