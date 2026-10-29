#!/usr/bin/env python3
"""Make the reference of the default filter: sha256 of `F.data`, `F.indices`, `F.indptr` and `b`.

    PYTHONPATH=<a tree of origin/main 7555b19> python tools/make_filter_reference.py --out tests/data/filter_default_reference.json

The reference in `tests/data/filter_default_reference.json` was made with the filter of origin/main
7555b19 (before A3a). `tests/test_solid_filter.py` builds the same grids with the current code and
compares the hashes: the default rule (`density_one`) must give the same bytes as before A3a.
Each case has FIXED, LOAD, KEEP_OUT and EMPTY voxels near DESIGN voxels, at a radius of 1.5 voxels
(the default) and of 3.75 voxels. The grid is built here (no random numbers), so the hashes do not
depend on the test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

H = 2.0                          # mm: the radius in mm is R_VOXELS * H, an exact binary fraction
R_VOXELS = (1.5, 3.75)
SHAPE = (14, 10, 9)


def case_labels() -> np.ndarray:
    """One grid with every label near design voxels (codes of `topopt.voxelize`)."""
    EMPTY, DESIGN, FIXED, LOAD, KEEP_OUT = 0, 1, 2, 3, 4
    lab = np.full(SHAPE, DESIGN, dtype=np.uint8)
    lab[:2] = FIXED                      # a plate on one side
    lab[11:, 3:7, 2:6] = LOAD            # a load block at the other end
    lab[5:8, 4:6, 0:3] = FIXED           # a fixed stub inside the design space
    lab[6:9, 6:9, 6:9] = KEEP_OUT        # a keep-out box
    lab[3:5, 0:2, 5:9] = EMPTY           # empty voxels (outside the part) at a corner
    lab[9, 2, 3] = LOAD                  # a single frozen voxel
    return lab


def digest(build_filter) -> list[dict]:
    """The hashes of `build_filter(labels, h, radius)` (the default rule) for each radius."""
    lab = case_labels()
    rows = []
    for rv in R_VOXELS:
        F, FT, b = build_filter(lab, H, rv * H)
        sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()  # noqa: E731
        rows.append({"radius_voxels": rv, "nnz": int(F.nnz), "n_design": int(F.shape[0]),
                     "dtypes": [str(F.data.dtype), str(F.indices.dtype), str(F.indptr.dtype), str(b.dtype)],
                     "sha256": {"F.data": sha(F.data), "F.indices": sha(F.indices), "F.indptr": sha(F.indptr),
                                "b": sha(b), "FT.data": sha(FT.data), "FT.indices": sha(FT.indices),
                                "FT.indptr": sha(FT.indptr)}})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="write the JSON here (else print it)")
    a = ap.parse_args()
    from topopt import filter as topopt_filter
    doc = {"made_with": "topopt.filter.build_filter of origin/main 7555b19 (before A3a), git archive, Linux x86_64, numpy "
                        + np.__version__,
           "h_mm": H, "shape": list(SHAPE),
           "note": "The default rule (density_one) must give the same bytes. See tools/make_filter_reference.py.",
           "cases": digest(topopt_filter.build_filter)}
    text = json.dumps(doc, indent=1) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
