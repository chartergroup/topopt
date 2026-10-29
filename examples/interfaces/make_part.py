"""Write the region STLs of the interfaces example (a bracket with a bore, a boss and a machined face).

Run:  .venv/bin/python examples/interfaces/make_part.py

Units: mm. The files are ASCII STL with 17 significant digits, so a text diff or a pull request can carry them
and the reader gets the same float64 numbers back. The script can run many times. It overwrites the files.
The bore, the boss and the face are not STL files: they are the `interfaces` list of bracket.yaml (CONTRACTS.md section 23).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from manifold3d import Manifold

OUT = Path(__file__).resolve().parent

# name -> (x range, y range, z range)
BOXES = {
    "design": ((0.0, 40.0), (0.0, 20.0), (0.0, 16.0)),   # the space the part can fill
    "plate": ((0.0, 4.0), (0.0, 20.0), (0.0, 16.0)),     # fixed: the mounting plate (its outer 2 mm are machined away)
    "tip": ((37.0, 40.0), (8.0, 12.0), (0.0, 4.0)),      # load: the free end
}


def write_ascii_stl(path: Path, name: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """ASCII STL with 17 significant digits for every number."""
    f = lambda v: format(float(v), ".17g")   # noqa: E731
    lines = [f"solid {name}"]
    for tri in faces:
        a, b, c = (np.asarray(verts[i], float) for i in tri)
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        lines.append(f"  facet normal {f(n[0])} {f(n[1])} {f(n[2])}")
        lines.append("    outer loop")
        lines += [f"      vertex {f(p[0])} {f(p[1])} {f(p[2])}" for p in (a, b, c)]
        lines += ["    endloop", "  endfacet"]
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    for name, (x, y, z) in BOXES.items():
        m = Manifold.cube((x[1] - x[0], y[1] - y[0], z[1] - z[0])).translate((x[0], y[0], z[0])).to_mesh64()
        write_ascii_stl(OUT / f"{name}.stl", name, np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))
        print("wrote", OUT / f"{name}.stl")


if __name__ == "__main__":
    main()
