"""Displacement and alignment limits in the YAML (A5, CONTRACTS.md section 31), end to end.

The part is a small cantilever. Three probe regions (`regions.measure`) sit on it: `mid` (mid-span),
and `a` and `b` (two cross-sections). The YAML has both kinds of limit:

  limits.displacement   the mean displacement of `mid` along -z must stay below a limit (sense: max)
  limits.alignment      the shift and the tilt of `a` against `b` must stay below limits

Step 1 runs with large limits, so no row is active, and prints the rows block of `metrics.json`
(value at iteration 0, value at the end, limit, met). Step 2 sets each limit to 70 % of the end value
of step 1 and runs again: the rows are active, the run pays with compliance, and the block shows it.

Run:  python examples/motion_rows.py [iterations]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import yaml

from topopt.cli import main as topopt


def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def write_case(folder: Path, iterations: int, limits: dict) -> Path:
    box(folder / "design.stl", (0, 0, 0), (20, 10, 10))
    box(folder / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(folder / "tip.stl", (18, 4, 0), (20, 6, 3))
    box(folder / "mid.stl", (8, 0, 0), (12, 10, 10))
    box(folder / "a.stl", (6, 0, 0), (8, 10, 10))
    box(folder / "b.stl", (14, 0, 0), (16, 10, 10))
    doc = {
        "name": "cantilever_limits",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"},
                    "measure": {"mid": "mid.stl", "a": "a.stl", "b": "b.stl"}},
        "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
        "limits": {
            "displacement": [{"name": "mid_down", "region": "mid", "direction": [0, 0, -1], "sense": "max",
                              "limit_mm": limits["mid"]}],
            "alignment": [{"region_a": "a", "region_b": "b", "axis": [1, 0, 0],
                           "shift_limit_mm": limits["shift"], "tilt_limit_rad": limits["tilt"]}]},
        "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": iterations,
                     "beta_schedule": [[0, 1], [10, 2], [20, 4]]},
    }
    path = folder / "limits.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def run(folder: Path, iterations: int, limits: dict) -> dict:
    yaml_path = write_case(folder, iterations, limits)
    out = folder / "run"
    if topopt(["run", str(yaml_path), "--out", str(out), "--quiet"]) != 0:
        raise SystemExit("the run failed")
    return json.loads((out / "result" / "metrics.json").read_text())


def show(title: str, metrics: dict) -> None:
    print(title)
    for r in metrics["rows"]:
        print(f"  {r['name']:26s} limit {r['limit']:.4g} {r['unit']:3s}  start {r['value_start']:.4g}  "
              f"end {r['value_end']:.4g}  met {r['met']}")
    print(f"  compliance_final {metrics['compliance_final']:.5g}")


def main(iterations: int = 40) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "1").mkdir()
        (Path(tmp) / "2").mkdir()
        first = run(Path(tmp) / "1", iterations, {"mid": 100.0, "shift": 100.0, "tilt": 100.0})
        show("step 1: large limits (no row is active)", first)
        end = {r["name"].split(":")[0]: r["value_end"] for r in first["rows"]}
        second = run(Path(tmp) / "2", iterations,
                     {"mid": 0.7 * end["displacement_max"], "shift": 0.7 * end["shift"],
                      "tilt": 0.7 * end["tilt"]})
        show("step 2: limits at 70 % of the end values of step 1", second)
    return second


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
