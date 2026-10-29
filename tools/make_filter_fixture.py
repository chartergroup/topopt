#!/usr/bin/env python3
"""Write tests/data/filter_pipeline_fixture.json: the pipeline filter on small uniform voxel grids.

The fixture is the reference for `tests/test_solid_filter.py::test_filter_equals_pipeline_fixture`.
The demo filter (`topopt.filter.build_filter`) must give the same matrix when no solid voxel is
a neighbour (CONTRACTS.md section 22a).

The matrix is built with the real `build_filter` of the pipeline (`pipeline/lib/pp_core_final.py`,
volume-weighted linear hat filter, row-normalised, float64). The script reads that one function
from the source file and runs it. It does not import the module, because the module imports the
whole pipeline (skfem and more). The pipeline is a separate project: give its path.

    python tools/make_filter_fixture.py --pipeline /path/to/pipeline-checkout

Reference: earlier pipeline, revision d43bcf3, pipeline/lib/pp_core_final.py lines 108-123.
Run it with the main venv of this repo (numpy and scipy only).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

SOURCE = "pipeline/lib/pp_core_final.py"
REVISION = "earlier pipeline d43bcf3"
# (shape, voxel size in mm, filter radius in mm). The radius in voxels is never the square root of
# a whole number, so no neighbour sits exactly on the radius (the pipeline keeps d <= R, the demo d < R).
CASES = [
    {"shape": [5, 4, 4], "h": 1.0, "radius": 2.3},
    {"shape": [4, 4, 3], "h": 1.2, "radius": 1.8},
    {"shape": [6, 5, 4], "h": 0.5, "radius": 1.3},
]


def pipeline_build_filter(root: Path):
    """The function `build_filter` of the pipeline file, compiled from its source."""
    path = root / SOURCE
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_filter")
    lines = path.read_text().splitlines()[node.lineno - 1:node.end_lineno]
    print(f"pipeline build_filter: {path} lines {node.lineno}-{node.end_lineno}")
    ns = {"np": np, "sp": sp, "cKDTree": cKDTree}
    exec(compile("\n".join(lines), str(path), "exec"), ns)   # noqa: S102 - the source of the pipeline file
    return ns["build_filter"]


def case_matrix(build_filter, shape, h, radius):
    """Pipeline matrix of a uniform grid, rows in the C order of the voxels (as the demo filter)."""
    ijk = np.indices(shape).reshape(3, -1).T
    centroids = h * (ijk + 0.5)
    vol = np.full(len(ijk), h ** 3)
    return build_filter(centroids, vol, radius).tocoo()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pipeline", required=True, type=Path, help="root of the pipeline checkout")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tests" / "data"
                    / "filter_pipeline_fixture.json")
    a = ap.parse_args()
    bf = pipeline_build_filter(a.pipeline)
    cases = []
    for c in CASES:
        m = case_matrix(bf, c["shape"], c["h"], c["radius"])
        order = np.lexsort((m.col, m.row))
        cases.append({**c, "n": int(np.prod(c["shape"])), "rows": m.row[order].tolist(),
                      "cols": m.col[order].tolist(), "values": [float(v) for v in m.data[order]]})
        print(f"case {c}: {len(order)} entries")
    doc = {"source": f"{REVISION} {SOURCE} build_filter (lines 108-123), centroids h*(ijk+0.5), vol h^3",
           "note": "COO triplets of the pipeline filter H on a uniform voxel grid, sorted by (row, col). "
                   "Values are float64 and round-trip exactly.",
           "cases": cases}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {a.out} ({a.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
