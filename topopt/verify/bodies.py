"""The bodies of the tet mesh that `topopt verify` reads. CONTRACTS.md section 16, "Bodies and a singular matrix".

A body is a set of tets that are linked through shared vertices. Two bodies share no node, so the stiffness matrix
has no entry that links them. A body with no fixed node can move as a rigid piece: the matrix is singular and the
direct solvers stop on a zero pivot. This module counts the bodies before the solve and says which one has no support.

- `find_bodies(model)`: the bodies of a `P2Model`, largest volume first.
- `add_regions(...)`: the fixed nodes and the load facets of each body. With `held` (the regions of a joint or a spring) it also counts
  the facets that a support holds.
- `check_supported(...)`: raise `UnsolvableMesh` (exit code 1) for a body with no fixed node and no joint or spring facet.
- `empty_load_error(...)`: the same error for a load region that matches no facet of the mesh.

A mesh of several bodies is not an error when every body has a fixed node: the matrix is not singular.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .errors import UnsolvableMesh
from .names import opt


@dataclass
class Bodies:
    """The bodies of one mesh. `facts` is one dict for each body (the plain data for `verify.json`)."""
    count: int
    vertex_body: np.ndarray            # (nv,) body of each vertex
    node_body: np.ndarray              # (nn,) body of each P2 node (vertex or edge midpoint)
    facts: list = field(default_factory=list)


def find_bodies(model) -> Bodies:
    """The connected bodies of the mesh of `model`. Body 0 has the largest volume."""
    from scipy import sparse as sp
    from scipy.sparse.csgraph import connected_components

    nv = model.nv
    e = model.edges
    graph = sp.coo_matrix((np.ones(e.shape[1], np.int8), (e[0], e[1])), shape=(nv, nv))
    count, lab = connected_components(graph, directed=False)
    elem = lab[model.t[0]]
    volume = np.bincount(elem, weights=model.vol, minlength=count)
    order = np.argsort(-volume, kind="stable")
    rank = np.empty(count, np.int64)
    rank[order] = np.arange(count)
    lab = rank[lab]
    elem = lab[model.t[0]]
    tets = np.bincount(elem, minlength=count)
    vol = np.bincount(elem, weights=model.vol, minlength=count)
    node_body = np.concatenate([lab, lab[e[0]]])
    xyz = model.node_xyz
    facts = []
    for k in range(count):
        pts = xyz[:, node_body == k]
        facts.append(dict(body=int(k + 1), volume_mm3=float(vol[k]), tets=int(tets[k]),
                          bbox_min_mm=[float(x) for x in pts.min(axis=1)], bbox_max_mm=[float(x) for x in pts.max(axis=1)],
                          fixed_nodes=0, load_facets={}))
    return Bodies(int(count), lab, node_body, facts)


def add_regions(bodies: Bodies, fixed_sel, load_sel: dict, held=()) -> None:
    """Write the fixed nodes and the load facets of each body into `bodies.facts`.

    `fixed_sel` is a list of `Selection`, `load_sel` is `{name: Selection}`. `held` is a list of `Selection` of the regions of a joint
    or a spring (K1-A): a body that has a facet of one of them is held by that support. Each body then has the key `support_facets`.
    The key is not there when `held` is empty, so the facts of a plain run do not change.
    """
    n = bodies.count
    union = np.any([s.node_mask for s in fixed_sel], axis=0) if fixed_sel else np.zeros(len(bodies.node_body), bool)
    fixed = np.bincount(bodies.node_body[union], minlength=n)
    for k, f in enumerate(bodies.facts):
        f["fixed_nodes"] = int(fixed[k])
        f["load_facets"] = {}
    for name, s in load_sel.items():
        per = np.bincount(bodies.vertex_body[s.facets[:, 0]], minlength=n) if len(s.facets) else np.zeros(n, int)
        for k, f in enumerate(bodies.facts):
            if per[k]:
                f["load_facets"][name] = int(per[k])
    if held:
        support = sum(np.bincount(bodies.vertex_body[s.facets[:, 0]], minlength=n) if len(s.facets) else np.zeros(n, int) for s in held)
        for k, f in enumerate(bodies.facts):
            f["support_facets"] = int(support[k])


def report(bodies: Bodies) -> dict:
    """The block of `verify.json` (`mesh.bodies`)."""
    return dict(count=bodies.count, bodies=bodies.facts,
                note="bodies are sets of tets that share vertices. A body with no fixed node cannot be solved: "
                     "the check stops before the solve with exit code 1")


def one_line(bodies: Bodies) -> str:
    """For the log: the count and the volume of each body."""
    if bodies.count == 1:
        return f"bodies: 1 ({bodies.facts[0]['volume_mm3']:.1f} mm3)"
    shown = ", ".join(f"{f['volume_mm3']:.1f}" for f in bodies.facts[:6]) + (", ..." if bodies.count > 6 else "")
    return f"bodies: {bodies.count} (volumes {shown} mm3)"


def _body_text(f: dict) -> str:
    lo, hi = f["bbox_min_mm"], f["bbox_max_mm"]
    box = ", ".join(f"{a} {lo[i]:.4g} to {hi[i]:.4g}" for i, a in enumerate("xyz"))
    loads = ", ".join(f"{n} {c} facets" for n, c in f["load_facets"].items()) or "no load facet"
    held = f", {f['support_facets']:,} joint or spring facets" if "support_facets" in f else ""
    return (f"body {f['body']}: {f['volume_mm3']:.4g} mm3, {f['tets']:,} tets, {box} mm, "
            f"{f['fixed_nodes']:,} fixed nodes{held}, {loads}")


def floating(bodies: Bodies) -> list[int]:
    """The 0-based indices of the bodies that have no fixed node and no facet of a joint or a spring."""
    return [k for k, f in enumerate(bodies.facts) if f["fixed_nodes"] == 0 and f.get("support_facets", 0) == 0]


def _closest_approach(bodies: Bodies, k: int, fixed_sel, reach: float | None, what: str = "a fixed region") -> str:
    """How close body `k` comes to a fixed region (or, with joints and springs, to a fixed, joint or spring region: `what`), from the signed
    distances (-inf: farther than the reach)."""
    best, name = -np.inf, None
    for s in fixed_sel:
        if s.sd is None:
            continue
        m = s.sd[bodies.node_body == k]
        if len(m) and m.max() > best:
            best, name = float(m.max()), s.name
    if name is not None and np.isfinite(best):
        return f"closest approach to {what}: {-best:.3g} mm from '{name}'"
    return "no node within " + (f"{reach:.3g} mm of" if reach else "the reach of") + f" {what}"


def check_supported(bodies: Bodies, fixed_sel, band_mm: float, reach: float | None = None, extra: str = "",
                    facts: dict | None = None, support_sel=()) -> None:
    """Raise `UnsolvableMesh` if a body has no fixed node and no joint or spring facet. The text holds the facts, the likely causes and the options.

    `extra` is a text that is added at the end, and `facts` are keys that are added to the facts (`topopt verify --part-only` says which file
    it meshed and what the removed bodies give). `support_sel` are the `Selection` of the joint and spring regions (K1-A): the text then
    speaks of "a fixed, joint or spring region", and the closest approach is measured to all of them.
    """
    bad = floating(bodies)
    if not bad:
        return
    held = any("support_facets" in f for f in bodies.facts)         # the YAML has a joint or a spring (`add_regions(held=...)`)
    what = "a fixed, joint or spring region" if held else "a fixed region"
    parts = "; ".join(_body_text(bodies.facts[k]) for k in bad[:3]) + ("; ..." if len(bad) > 3 else "")
    near = "; ".join(f"body {k + 1}: {_closest_approach(bodies, k, [*fixed_sel, *support_sel], reach, what)}" for k in bad[:3])
    biggest = bodies.facts[0]
    raise UnsolvableMesh(
        f"the final mesh has {bodies.count} bodies and {len(bad)} of them has no fixed node"
        + (" and no facet of a joint or spring" if held else "") + f", so it can move as a rigid piece "
        f"and the stiffness matrix is singular ({parts}). The largest body is {biggest['volume_mm3']:.4g} mm3 with "
        f"{biggest['fixed_nodes']:,} fixed nodes. Fixed region band {band_mm:g} mm; {near}. "
        f"This is not a part that carries the load to the supports. Likely causes: a piece of the final part is cut off "
        f"(a thin member, a floating tip), or the {'fixed, joint or spring' if held else 'fixed'} region does not reach this body. "
        f"Options: look at the result (topopt render <run>); {opt('region_band')} larger if the body should touch {what}; "
        f"or run the optimizer again with a setting that keeps the load path (for example a compliance objective)." + (f" {extra.strip()}" if extra else ""),
        dict(reason="body_without_support", bodies=bodies.count, floating_bodies=[k + 1 for k in bad],
             band_mm=float(band_mm), body_facts=bodies.facts, **(facts or {})))


def empty_load_error(exc, bodies: Bodies, extra: str = "", extra_facts: dict | None = None) -> UnsolvableMesh:
    """The error for a load region that matches no boundary facet (`EmptyRegion` of `select_region`).

    The load region touches no body of the mesh: the final part has no surface at the load. Exit code 1.
    `extra` and `extra_facts` are added to the text and the facts, as for `check_supported`.
    """
    f = dict(exc.facts)
    name = f.get("region", "?")
    near = f.get("nearest_distance_mm")
    biggest = bodies.facts[0]
    return UnsolvableMesh(
        f"load region '{name}' touches no body of the final mesh: no boundary facet has its 3 vertices in the region "
        f"(band {f.get('band_mm', 0):g} mm; the nearest mesh point is {near:.3g} mm away). The mesh has {bodies.count} "
        f"{'body' if bodies.count == 1 else 'bodies'} (the largest: {biggest['volume_mm3']:.4g} mm3, "
        f"{biggest['fixed_nodes']:,} fixed nodes). This is not a part that carries the load: there is no material at the load. "
        f"Likely causes: the optimizer removed the material at the load region, or the STL files and the run are in "
        f"different frames (mm). Options: look at the result (topopt render <run>); {opt('region_band')} larger; "
        f"or run the optimizer again with a setting that keeps the load path (for example a compliance objective)." + (f" {extra.strip()}" if extra else ""),
        dict(reason="load_region_touches_no_body", bodies=bodies.count, body_facts=bodies.facts, **f, **(extra_facts or {})))
