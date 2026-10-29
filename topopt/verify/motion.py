"""The A5 rows (displacement and alignment limits) on the P2 tet model of `topopt verify` (B3, CONTRACTS.md sections 16 and 31).

The YAML of the run can have `limits.displacement` and `limits.alignment` rows. The optimizer meets them on the voxel model.
Verify computes the same rows from the P2 displacement of the printed STL, with the same code as the optimizer
(`topopt.rows_motion.MotionSet.bind_nodes`): the same rigid fits, rank checks, signs and units. The record is the key `motion`
of `verify.json`. A row has the tet value, the voxel value of the run (`value_end` of `result/metrics.json`, key `rows`), their
difference, the limit and `tet_over_limit`. These are facts. There is no `met` flag and no exit code for a row.

Node rule. A region is the set of P2 nodes (vertices and edge midpoints) whose signed distance to the region STL is at least
`-band` (the region rule of `topopt.verify.regions`). The displacement of a region is the equal-weight mean over these nodes. The
voxel model uses the equal-weight mean over the grid nodes of the voxels of the region. The two sets are not the same sets
of points, and the midpoints are most of the P2 nodes, so the two means can differ also for the same displacement field.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .. import rows_motion as rm
from ..config import UserError
from . import regions as vregions
from .errors import EmptyRegion

KIND = "displacement and alignment rows (A5) on the P2 tet model of the printed STL"
RULE = ("Displacement of a region: the equal-weight mean over the P2 nodes (vertices and edge midpoints) of the region, whose "
        "signed distance to the region STL is at least -band. Alignment: the least-squares rigid fit over the same nodes about their "
        "mean, the shift at the midpoint of the two centres and the tilt from the rotation difference (topopt.rows_motion). "
        "Signs: displacement_max is y, displacement_min is -y, the other rows are |y|.")
VOXEL_SOURCE = ("value_end of the key rows of result/metrics.json: the value of the optimizer density of the last pass "
                "(gray), or of the nominal design in a robust run. It is not the value of the printed STL.")


def voxel_rows(info) -> dict:
    """`{row name: row}` from the key `rows` of `result/metrics.json` of the run, or {}."""
    try:
        rows = json.loads((info.run_dir / "result" / "metrics.json").read_text()).get("rows") or []
        return {r["name"]: r for r in rows if isinstance(r, dict) and "name" in r}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def wanted(cfg) -> bool:
    """Does the YAML have A5 rows?"""
    return bool(rm.row_specs(cfg))


FIXED_FACT_OPTIMIZER = "so the value is always 0 and the row cannot steer the design."
FIXED_FACT_TETS = "so the value is always 0 on the tet model."


def _tet_fact(text: str) -> str:
    """The fact of `rows_motion.all_fixed_fact` ends with a sentence about the optimizer; verify steers nothing."""
    return text.replace(FIXED_FACT_OPTIMIZER, FIXED_FACT_TETS)


def _skipped(reason: str, message: str, options: list, **extra) -> dict:
    return dict(kind=KIND, status="skipped", reason=reason, rule=RULE, rows=[], facts=[message], options=options, **extra)


def error_record(message: str) -> dict:
    """The record of rows that stopped on an unexpected error: a fact with its options. `topopt verify` goes on."""
    return dict(kind=KIND, status="error", reason="exception", message=message, rule=RULE, rows=[],
                facts=[f"The rows stopped: {message}"],
                options=["Look at the region files and at the rows of the YAML (`topopt check` checks them on the voxel model), then run "
                         "verify again. Verify goes on without the rows."])


def motion_record(cfg, info, model, free, load_sel, bfacets, band: float, U, selections: dict | None = None) -> dict | None:
    """The record `motion` for the displacements `U` (n_free, n_cases) of the cases, or None if the YAML has no row.

    `load_sel` is `{name: Selection}` of the load regions that the cases use, `bfacets` the boundary facets of the mesh,
    `band` the band of the region rule (mm). With `selections` (a dict), each region that the rows select here (a measure region, or a load
    region that has no force in a case) is also stored in it by name: `topopt verify` names the regions of the mode facts with them. A row that cannot work on the tets (a region without a node, a rigid fit of
    rank below 6) is a skipped record with the fact and the options; `topopt verify` goes on.
    """
    ms = rm.MotionSet(cfg)
    if not ms.specs:
        return None
    t0 = time.perf_counter()
    pos = np.full(model.N, rm.FIXED_DOF, dtype=np.int64)
    pos[free] = np.arange(len(free))
    X = model.node_xyz.T
    load_paths = {lr.name: lr.path for lr in cfg.regions.load}
    measure_paths = {mr.name: mr.path for mr in cfg.regions.measure}
    seen: dict = {}

    def lookup(name):
        if name in load_sel:
            sel, role = load_sel[name], "load region"
        else:
            path = load_paths.get(name) or measure_paths.get(name)
            if path is None:
                raise UserError(f"region '{name}' is not a measure or a load region")
            mesh = vregions.load_region_mesh(path)
            sel = vregions.select_region(model, bfacets, path, name, "measure", band, region_mesh=mesh,
                                         sd_cache=vregions.signed_distance(mesh, X, band))
            role = "load region (no force in a case)" if name in load_paths else "measure region"
            if selections is not None:
                selections[name] = sel
        nodes = np.flatnonzero(sel.node_mask)
        n_fixed = int(np.all(pos[3 * nodes[:, None] + np.arange(3)] == rm.FIXED_DOF, axis=1).sum())
        seen[name] = dict(role=role, nodes=int(nodes.size), vertices=int(np.sum(nodes < model.nv)), fixed_nodes=n_fixed,
                          centre_mm=[float(x) for x in X[nodes].mean(axis=0)] if nodes.size else None)
        return rm.RegionNodes(name, nodes, X[nodes], int(nodes.size), n_fixed)

    try:
        facts = ms.bind_nodes(pos, len(free), lookup)
    except (UserError, EmptyRegion) as exc:
        return _skipped("rows_cannot_work", str(exc),
                        ["Fix the region or the row as the fact says (the same rows are checked by `topopt check` on the voxel model), "
                         "then run verify again. Verify goes on without the rows."], regions=seen)
    facts = [_tet_fact(t) for t in facts]
    voxel = voxel_rows(info)
    names = [c.name for c in cfg.load_cases]
    rows = []
    for r in ms.rows:
        kind = r.name.split(":", 1)[0]
        tet = r.physical(U)
        v = voxel.get(r.name)
        vv = float(v["value_end"]) if v and v.get("value_end") is not None else None
        rows.append(dict(name=r.name, kind=kind, unit=rm.ROW_UNITS[kind], case=names[r.case], limit=float(r.limit),
                         tet_value=float(tet), voxel_value=vv,
                         difference=float(tet - vv) if vv is not None else None,
                         tet_over_voxel=float(tet / vv) if vv else None,
                         tet_over_limit=float(tet / r.limit)))
    options = []
    if any(row["voxel_value"] is None for row in rows):
        facts = list(facts) + ["Some rows have no voxel value: the key rows of result/metrics.json of the run does not name them "
                               "(the run was made without these limits, or the YAML changed after the run)."]
        options.append("Run the optimizer again with the limits in the YAML to get the voxel values.")
    return dict(kind=KIND, status="done", rule=RULE, voxel_value_source=VOXEL_SOURCE, band_mm=float(band), regions=seen,
                rows=rows, facts=facts, options=options, seconds=round(time.perf_counter() - t0, 3))
