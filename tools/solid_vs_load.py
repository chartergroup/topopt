#!/usr/bin/env python3
"""Show what `regions.solid` does, and compare it with the old form (a load region with no force).

    python tools/solid_vs_load.py problem.yaml

The script loads the YAML with the normal config path, voxelizes it, and prints the label counts and the
solid facts. Then it builds the old form of the same problem (each solid STL becomes a load region
that no load case uses, listed after the real load regions) and prints whether the labels, the forces,
the fixed dofs and the filter of both forms are identical. It stops with exit code 1 when they differ.
It runs no optimizer.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import numpy as np

from topopt.config import LoadRegion, UserError, load_config
from topopt.filter import build_filter
from topopt.problem import build_problem
from topopt.voxelize import DESIGN, EMPTY, FIXED, KEEP_OUT, LOAD, solid_lines

NAMES = {EMPTY: "empty", DESIGN: "design", FIXED: "fixed", LOAD: "load (load + solid)", KEEP_OUT: "keep_out"}


def counts(labels: np.ndarray) -> dict:
    return {name: int((labels == code).sum()) for code, name in NAMES.items()}


def old_form(cfg):
    """The same config with each solid STL as a load region without force (after the real ones)."""
    reg = cfg.regions
    extra = tuple(LoadRegion(f"solid_{i}_{p.stem}", p) for i, p in enumerate(reg.solid))
    return dataclasses.replace(cfg, regions=dataclasses.replace(reg, load=reg.load + extra, solid=()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("yaml")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.yaml)
        new = build_problem(cfg)
        old = build_problem(old_form(cfg))
    except UserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    st = cfg.settings
    print(f"grid {new.shape}, voxel {new.h:g} mm, {len(cfg.regions.solid)} solid STL(s)")
    print("label counts, regions.solid form:   ", counts(new.labels))
    print("label counts, old form (no force):  ", counts(old.labels))
    for line in solid_lines(new.region_facts.get("solid")) or ["solid regions: none in this YAML"]:
        print(line)
    for r in new.region_facts.get("solid", {}).get("regions", []):
        print(f"  solid '{r['name']}': {r['stl_voxels']} voxels in the STL, {r['voxels']} stay solid")
    same = {
        "labels": np.array_equal(new.labels, old.labels),
        "forces": np.array_equal(new.forces, old.forces),
        "fixed_dofs": np.array_equal(new.fixed_dofs, old.fixed_dofs),
    }
    for rule in ("density_one", "ignore"):
        Fn, _, bn = build_filter(new.labels, new.h, st.filter_radius, rule)
        Fo, _, bo = build_filter(old.labels, old.h, st.filter_radius, rule)
        same[f"filter {rule}"] = (Fn != Fo).nnz == 0 and np.array_equal(bn, bo)
    # The load ids differ only on solid voxels: -1 here, the index of the extra load region in the old form.
    solid_only = (new.labels == LOAD) & (new.load_region_ids < 0)
    n_real = len(cfg.regions.load)
    same["load ids (real load regions)"] = bool(np.array_equal(new.load_region_ids[~solid_only],
                                                                old.load_region_ids[~solid_only]))
    same["load ids (solid voxels: -1 vs extra region)"] = bool(
        (old.load_region_ids[solid_only] >= n_real).all()) if solid_only.any() else True
    for k, v in same.items():
        print(f"identical, {k}: {v}")
    return 0 if all(same.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
