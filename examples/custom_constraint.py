"""Run MMA with a custom constraint, using only the Python API.

The problem is a small cantilever beam. The objective is the weighted compliance.
There are two constraints: the volume (built in) and a custom one. The custom one
says: the mean density of the tip zone (x >= 14 mm) must be 0.5 or more. It shows the
three steps to add a constraint:

  1. write a function `DesignPoint -> Response(value, grad)`. The gradient is with
     respect to the physical density `point.rho`. The library does the chain rule
     through the projection and the filter. (A row that uses the displacements `U`
     returns `wrt="rho_stiff"` and reads `point.rho_stiff`: `stiffness_derivative` is the
     slope to `rho_stiff`, which equals `rho` in a default run.)
  2. wrap it in `Constraint(name, function, limit=...)`. A row is met when value <= limit.
  3. give the list of constraints to `Formulation` and call `minimize`.

Run:  python examples/custom_constraint.py [iterations]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import yaml

from topopt.config import load_config
from topopt.formulation import Constraint, Formulation, Response, volume_constraint, weighted_compliance
from topopt.optimize import Model, minimize
from topopt.problem import build_problem


def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def make_yaml(folder: Path, iterations: int) -> Path:
    """A cantilever: design block 20 x 10 x 10 mm, a fixed plate on the left, a load at the tip."""
    box(folder / "design.stl", (0, 0, 0), (20, 10, 10))
    box(folder / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(folder / "tip.stl", (18, 4, 0), (20, 6, 3))
    doc = {
        "name": "cantilever",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
        "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
        "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": iterations,
                     "beta_schedule": [[0, 1], [10, 2], [20, 4]]},
    }
    path = folder / "cantilever.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def main(iterations: int = 30) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        model = Model(build_problem(load_config(make_yaml(Path(tmp), iterations))))
        zone = model.design_centers()[:, 0] >= 14.0          # design voxels of the tip zone

        def tip_zone_density(point):
            """Value: minus the mean density of the zone. Minus, because a row is value <= limit."""
            grad = np.zeros(point.n_design)                  # d value / d rho, one entry per voxel
            grad[zone] = -1.0 / zone.sum()
            return Response(-float(point.rho[zone].mean()), grad)

        tip_zone = Constraint("tip_zone_density", tip_zone_density, limit=-0.5)   # mean >= 0.5
        form = Formulation(weighted_compliance(),
                           [volume_constraint(model.n_design, 0.3), tip_zone])
        result = minimize(model, form)
        model.solver.free()
    last = result.history[-1]
    for c in last["constraints"]:
        print(f"{c['name']:18s} value {c['value']:8.4f}  limit {c['limit']:8.4f}  "
              f"violation {c['violation']:.4f}")
    print(f"compliance {last['compliance']:.5g}")
    return last


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
