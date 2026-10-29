"""Build the region STLs for the footbridge example.

Run:  .venv/bin/python examples/footbridge/make_footbridge.py [--out DIR]   (default: the folder of this script)

Our own footbridge: every value below is a named round number. A thin solid skin on top carries the walking surface, two feet at the
ends stand on the ground, and a half-ellipse under the span is kept free. The optimizer builds everything between the skin and the
feet.

Units: mm. x runs along the span (0..280), y across it (the centre line is y = 0), z is up (the ground is z = 0).
The script can run many times. It overwrites the files in its folder (or in DIR) and prints each region.
Only this folder has bridge-specific code. The engine stays general.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

OUT = Path(__file__).resolve().parent

# ---- parameters (mm) -------------------------------------------------------
SPAN = 280.0               # length of the bridge (x)
WIDTH = 36.0               # width of the deck (y)
DECK_TOP = 88.0            # top of the solid skin
SKIN = 2.5                 # the solid skin under the walking surface (z 85.5..88)
WALK = 3.0                 # the walking surface on top of the skin (z 88..91): the load, in two halves at mid-span
FOOT_X, FOOT_Z = 24.0, 30.0   # each foot: 24 mm of the span at each end, 30 mm high from the ground, the full width (fixed)
CLEAR_HALF = 100.0         # the clearance under the span: a half ellipse, 100 mm half width at the ground ...
CLEAR_HEIGHT = 56.0        # ... and 56 mm high at mid-span (keep out)
CLEAR_PAST = (4.0, 5.0)    # the clearance reaches 4 mm past each side (y) and 5 mm below the ground (z)
INSET = 2.5                # the design space ends 2.5 mm inside the two side faces of the skin
OVERLAP = 1.0              # the design space reaches 1 mm into the skin (the solid skin wins there)
SEG = 96                   # segments of the half ellipse

Y0, Y1 = -WIDTH / 2, WIDTH / 2
SKIN_Z0 = DECK_TOP - SKIN


def box(x, y, z) -> Manifold:
    """A box from (min, max) pairs."""
    return Manifold.cube([x[1] - x[0], y[1] - y[0], z[1] - z[0]]).translate([x[0], y[0], z[0]])


def clearance() -> Manifold:
    """The half ellipse under the span (seen from the side, in x-z), closed below the ground, as a prism along y."""
    t = np.linspace(0.0, np.pi, SEG + 1)
    below = -CLEAR_PAST[1]
    pts = [(SPAN / 2 + CLEAR_HALF, below)] + [(SPAN / 2 + CLEAR_HALF * np.cos(a), CLEAR_HEIGHT * np.sin(a)) for a in t] \
        + [(SPAN / 2 - CLEAR_HALF, below)]
    y0, y1 = Y0 - CLEAR_PAST[0], Y1 + CLEAR_PAST[0]
    # the outline is extruded along +z; the turn of 90 deg about x makes the outline (x, z) and the extrusion -y
    return CrossSection([pts]).extrude(y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])


def to_trimesh(man: Manifold) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3], faces=np.asarray(mesh.tri_verts), process=True)


def build() -> dict[str, trimesh.Trimesh]:
    """The region meshes, by file name without `.stl`."""
    regions = {
        "design": box((0, SPAN), (Y0 + INSET, Y1 - INSET), (0, SKIN_Z0 + OVERLAP)),
        "feet_a": box((0, FOOT_X), (Y0, Y1), (0, FOOT_Z)),
        "feet_b": box((SPAN - FOOT_X, SPAN), (Y0, Y1), (0, FOOT_Z)),
        "walk_left": box((0, SPAN / 2), (Y0, Y1), (DECK_TOP, DECK_TOP + WALK)),
        "walk_right": box((SPAN / 2, SPAN), (Y0, Y1), (DECK_TOP, DECK_TOP + WALK)),
        "clearance": clearance(),
        "deck_skin": box((0, SPAN), (Y0, Y1), (SKIN_Z0, DECK_TOP)),
    }
    return {name: to_trimesh(man) for name, man in regions.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the region STLs of the footbridge example.")
    ap.add_argument("--out", type=Path, default=OUT, help="the folder for the STL files (default: the folder of this script)")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'region':12s} {'watertight':10s} {'volume mm3':>11s}  bounds min / max")
    for name, m in build().items():
        assert m.is_watertight, name
        m.export(out / f"{name}.stl")
        b = m.bounds
        print(f"{name:12s} {str(m.is_watertight):10s} {m.volume:11.1f}  {np.round(b[0], 1)} / {np.round(b[1], 1)}")


if __name__ == "__main__":
    main()
