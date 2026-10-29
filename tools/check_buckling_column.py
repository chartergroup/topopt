#!/usr/bin/env python
"""Run the buckling check on a column through the CLI and print the factors against Euler.

    python tools/check_buckling_column.py [--across 4 6 8] [--aspect 1.5] [--solver auto] [--keep DIR]

For each number of voxels across, the script writes four STL files and a YAML file of a fixed-free column
(a plate below it, a plate on top that carries the load), runs `topopt run --buckling` with one iteration, and
reads `result/metrics.json`. The column is 5 mm on the weak side and 20 times as long (100 mm), E 110000 MPa,
nu 0.34 (the column of the pipeline Euler test). The load is `Pcr` of the Euler formula, so an exact model
gives the load factor 1.

The factor is above 1 because full integration hex8 voxels are too stiff in bending. It falls to 1 with the
mesh size. The measured values on this geometry are +4.1 percent at 4 voxels across,
+2.0 at 6 and +1.2 at 8 (weak side 1, strong side 1.5). The script prints the error and does not judge it.

The check needs a direct solver (macOS: Apple Cholesky, Linux x86_64: the `pardiso` extra). With AMG the run goes
on and the script prints the fact that the check gives.

The load plate spreads the force over two layers of nodes, half on the lower one. So the load acts at the mean
height `L + h/2` above the clamp, and the script uses this length in the Euler formula. With it the script gives
the same factors as the direct model of tests/test_buckling.py (1.0411 at 4 across, 1.0197 at 6).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

E, NU, WEAK_MM, LENGTH_RATIO = 110000.0, 0.34, 5.0, 20


def box(path: Path, lo, hi) -> None:
    import trimesh
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def write_case(folder: Path, across: int, aspect: float, solver: str):
    """The YAML of one column. Return `(yaml_path, euler_force_N, voxels, voxel_mm)`."""
    import yaml
    h = WEAK_MM / across
    w, d = WEAK_MM, round(WEAK_MM * aspect / h) * h           # the strong side is a whole number of voxels
    length = LENGTH_RATIO * across * h
    box(folder / "plate.stl", (0, 0, -h), (w, d, 0))
    box(folder / "column.stl", (0, 0, 0), (w, d, length))
    box(folder / "top.stl", (0, 0, length), (w, d, length + h))
    inertia = d * w ** 3 / 12.0                              # the weak axis is x
    p_cr = np.pi ** 2 * E * inertia / (2.0 * (length + h / 2.0)) ** 2
    spec = {"name": f"column_{across}",
            "regions": {"design": "column.stl", "fixed": ["plate.stl"], "load": {"top": "top.stl"}},
            "load_cases": [{"name": "compress", "forces": [{"region": "top", "force": [0.0, 0.0, -float(p_cr)]}]}],
            "material": {"E": E, "nu": NU},
            "settings": {"voxel_size": h, "volume_fraction": 0.95, "iterations": 1, "solver": solver},
            "buckling": {"check": True, "modes": 4}}
    path = folder / "column.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path, p_cr, across * round(WEAK_MM * aspect / h) * (LENGTH_RATIO * across + 2), h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--across", type=int, nargs="+", default=[4, 6, 8], help="voxels across the weak side")
    ap.add_argument("--aspect", type=float, default=1.5, help="strong side over weak side (1.0: a square column "
                    "with a double mode)")
    ap.add_argument("--solver", default="auto", help="settings.solver of the run (default auto)")
    ap.add_argument("--keep", metavar="DIR", help="keep the files in DIR (default: a temporary folder)")
    args = ap.parse_args(argv)
    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="buckling_column_"))
    print(f"files in {root}")
    print(f"{'across':>6} {'voxels':>7} {'lambda_1':>9} {'error %':>8} {'lambda_2':>9} {'near-repeated':>13} "
          f"{'solver':>8} {'check s':>8}")
    code = 0
    for n in args.across:
        folder = root / f"n{n}"
        folder.mkdir(parents=True, exist_ok=True)
        yaml_path, _p_cr, voxels, _h = write_case(folder, n, args.aspect, args.solver)
        run = subprocess.run([sys.executable, "-m", "topopt", "run", str(yaml_path), "--out", str(folder / "run"),
                              "--quiet", "--buckling"], capture_output=True, text=True)
        if run.returncode != 0:
            print(f"{n:>6}  topopt run failed (exit {run.returncode}): {run.stderr.strip().splitlines()[-1:]}")
            code = 1
            continue
        b = json.loads((folder / "run" / "result" / "metrics.json").read_text())["buckling"]
        if b["status"] != "done":
            print(f"{n:>6}  check {b['status']}: {' '.join(b['facts'])}")
            for opt in b["options"]:
                print(f"        option: {opt}")
            code = 2
            continue
        case = b["cases"][0]
        lam = case["load_factors"]
        if not lam:
            print(f"{n:>6}  no load factor: {case['status']}: {' '.join(case['facts'])}")
            code = 2
            continue
        print(f"{n:>6} {voxels:>7} {lam[0]:>9.5f} {100 * (lam[0] - 1):>+8.2f} "
              f"{lam[1] if len(lam) > 1 else float('nan'):>9.4f} {str(case['near_repeated']):>13} "
              f"{b['solver']:>8} {b['seconds']:>8.1f}")
    print("Euler: lambda = 1 for the load Pcr. The voxel model reads above 1 (stiff in bending); "
          "the error falls as the number of voxels across grows.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
