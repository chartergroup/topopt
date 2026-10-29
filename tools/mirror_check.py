#!/usr/bin/env python3
"""Check the mirror map of a problem YAML (CONTRACTS.md section 26.1). It does not solve.

Usage: python tools/mirror_check.py PROBLEM.yaml [--seed N]

The script loads the YAML through the normal path (`load_config`), builds the problem (with the parity
choice), builds the index map of each plane, and prints facts:

- the parity that is used, and the label mismatch counts of both parities;
- the orphans: design voxels with no design partner (count, share, and the cause of the image);
- the involution check `perm[perm] == arange`;
- `|F P x - P F x|` for a random `x`: the density filter commutes with the map when the label field is
  symmetric, so the value is 0 up to rounding. Near an orphan it is not 0: that is a fact about the part;
- the largest difference of the filter constant `b` between the two partners;
- the asymmetry `|f - R f| / |f|` of each load case about each plane.

Exit code: 0 when every map is an involution, 1 when one is not, 2 when the YAML has no `mirror` key or is wrong.
"""
import argparse
import inspect
import sys

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the mirror map of a problem YAML.")
    ap.add_argument("yaml", help="problem YAML with a mirror key")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random x (default 0)")
    args = ap.parse_args(argv)

    from topopt import symmetry
    from topopt.config import UserError, load_config
    from topopt.filter import build_filter
    from topopt.problem import build_problem

    try:
        cfg = load_config(args.yaml)
        problem = build_problem(cfg)
    except UserError as e:
        print(f"error: {e}")
        return 2
    if problem.mirror is None:
        print("the YAML has no mirror key: nothing to check")
        return 2
    st = cfg.settings
    kw = {}
    if "solid_neighbours" in inspect.signature(build_filter).parameters:   # the filter rule exists with A3a
        kw = {"solid_neighbours": st.filter_solid_neighbours}
    F, _FT, b = build_filter(problem.labels, problem.h, st.filter_radius, **kw)
    x = np.random.default_rng(args.seed).uniform(0.0, 1.0, F.shape[0])
    print(f"grid {problem.shape[0]} x {problem.shape[1]} x {problem.shape[2]}, voxel {problem.h:g} mm, "
          f"origin {[round(float(v), 4) for v in problem.origin]}, {F.shape[0]} design voxels")
    ok = True
    for line in symmetry.summary_lines(problem.mirror):
        print(line)
    for pl in problem.mirror["planes"]:
        axis = symmetry.AXES.index(pl["axis"])
        perm, causes = symmetry.design_map(problem.labels, axis, pl["k"])
        involution = bool(np.array_equal(perm[perm], np.arange(perm.size)))
        ok = ok and involution
        name = f"plane {pl['axis']} = {pl['at_mm']:g} mm"
        print(f"{name}: involution: {'ok' if involution else 'FAILED'}; parity {pl['parity']}, k {pl['k']}; "
              f"orphans {sum(causes.values())} {causes}")
        diff = np.abs(F @ x[perm] - (F @ x)[perm])
        print(f"{name}: |F P x - P F x| max {diff.max():.3g}, mean {diff.mean():.3g} (random x, seed {args.seed}); "
              f"|b - b[perm]| max {np.abs(b - b[perm]).max():.3g}; voxels with a difference above 1e-12: "
              f"{int((diff > 1e-12).sum())}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
