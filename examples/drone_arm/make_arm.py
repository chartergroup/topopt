"""Build the region STLs for the 5-inch drone-arm example.

Run:  .venv/bin/python examples/drone_arm/make_arm.py [--out DIR]   (default: the folder of this script)

Our own parametric arm: every value below is a named round number that is typical for a 5-inch freestyle frame (a 226 mm
motor-to-motor diagonal, a 6 mm plate arm, a 40 mm square body stack, M3 motor slots for the 16 x 16 and 19 x 19 mm patterns).
No outline, curve or point comes from the drawing of another frame.

Units: mm. The frame centre is the origin, x goes out along the arm to the motor, z is up (thrust); the plate is z = 0..6.
The script can run many times. It overwrites the files in its folder (or in DIR). It prints the plain plate arm, whose mass is
`target_mass_g` of drone_arm.yaml. Only this folder has arm-specific code. The engine stays general.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, JoinType

OUT = Path(__file__).resolve().parent

# ---- parameters (mm) -------------------------------------------------------
WHEELBASE = 226.0          # motor-to-motor diagonal of the frame
MOTOR_X = WHEELBASE / 2    # frame centre to motor centre, on the arm axis
STACK_HALF = 20.0          # the body plates are a 40 x 40 mm square; the arm leaves through a corner (true X),
                           # so the square is turned 45 deg here
PLATE_Z = (0.0, 6.0)       # the arm plate
ENVELOPE_Z = (-6.0, 12.0)  # design space: 6 mm more below and above the plate
ROOT_X = 6.0               # the round root end of the arm starts this far from the frame centre, inside the stack
ROOT_W = 10.0              # body width at the root; the root end is a half circle of this width
NECK_W = 5.0               # body width at the motor centre (the straight taper ends there; about 6 mm where the body meets the motor end)
END_R = 16.5               # round motor end of the arm: about 1.5 mm of wall past the outer end of a motor slot
FILLET_R = 6.0             # the inner corners where the body meets the motor end are rounded with this radius
ROOT_HOLES_X = (14.0, 24.0)  # two clamp bolts on the arm axis, under the stack
M3_R = 1.6                 # M3 clearance
MOTOR_PATTERN = 16.0       # motor bolt slots: from the 16 x 16 mm M3 pattern of a 22xx motor ...
MOTOR_PATTERN_MAX = 19.0   # ... out to the 19 x 19 mm pattern, on the diagonals, so that both motor sizes fit
SHAFT_R = 4.0              # centre hole: room for the shaft end and its clip
PAD_R, PAD_Z0 = 14.0, 2.0  # motor pad: a solid disc r 14 under the motor base, z 2..6
BELL_R = 15.0              # the turning motor bell + 1 mm, kept free above the pad
HEAD_EXTRA = 1.5           # screw head and driver room below the plate, added around each bolt slot
EPS = 0.5                  # keep-outs reach past the envelope
SEG = 96                   # segments of a circle
DENSITY = 1.24             # g/cm3, the default printed polymer of the tool


def circle(c, r):
    return CrossSection.circle(r, SEG).translate((float(c[0]), float(c[1])))


def prism(sec, z0, z1):
    return sec.extrude(z1 - z0).translate((0.0, 0.0, z0))


def to_trimesh(man) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    # process=False: the manifold mesh is clean, and a merge of near vertices (the rounded offsets have some) would open it
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3], faces=np.asarray(mesh.tri_verts), process=False)


def outline():
    """The arm seen from above: a half-circle root end, a straight taper and a round motor end, with rounded inner corners."""
    root_c = ROOT_X + ROOT_W / 2
    body = CrossSection([[(root_c, -ROOT_W / 2), (MOTOR_X, -NECK_W / 2), (MOTOR_X, NECK_W / 2), (root_c, ROOT_W / 2)]])
    shape = body + circle((root_c, 0.0), ROOT_W / 2) + circle((MOTOR_X, 0.0), END_R)
    return shape.offset(FILLET_R, JoinType.Round, circular_segments=SEG).offset(-FILLET_R, JoinType.Round, circular_segments=SEG)


def build() -> tuple[dict[str, trimesh.Trimesh], float]:
    """The region meshes, and the area of the plain plate arm (the outline minus the holes)."""
    motor = (MOTOR_X, 0.0)
    arm = outline()
    d = STACK_HALF * math.sqrt(2.0)          # corner of the turned stack square on the arm axis
    stack = CrossSection([[(d, 0.0), (0.0, d), (-d, 0.0), (0.0, -d)]])
    root_holes = [circle((x, 0.0), M3_R) for x in ROOT_HOLES_X]
    r0, r1 = MOTOR_PATTERN / math.sqrt(2.0), MOTOR_PATTERN_MAX / math.sqrt(2.0)   # the slot ends on each diagonal
    slots = []
    for a in (45.0, 135.0, 225.0, 315.0):
        u = (math.cos(math.radians(a)), math.sin(math.radians(a)))
        slots.append(CrossSection.batch_hull([circle((MOTOR_X + r * u[0], r * u[1]), M3_R) for r in (r0, r1)]))
    shaft = circle(motor, SHAFT_R)
    plate = arm - shaft
    for c in root_holes + slots:
        plate = plate - c
    zb, zt = PLATE_Z
    z0, z1 = ENVELOPE_Z
    regions = {
        "design": prism(arm, z0, z1),
        "frame_clamp": prism(arm ^ stack, zb, zt),
        "plate_stack": prism(stack, z0 - EPS, zb) + prism(stack, zt, z1 + EPS),
        "root_bolt_hole_a": prism(root_holes[0], z0 - EPS, z1 + EPS),
        "root_bolt_hole_b": prism(root_holes[1], z0 - EPS, z1 + EPS),
        "motor_shaft": prism(shaft, z0 - EPS, z1 + EPS),
        "motor_bell": prism(circle(motor, BELL_R), zt, z1 + EPS),
        "motor_pad": prism(circle(motor, PAD_R), PAD_Z0, zt),
        "arm_original": prism(plate, zb, zt),
    }
    # each slot goes through; the head room is only under the plate (a counterbore would cut the rim of the motor end)
    for i, s in enumerate(slots):
        regions[f"motor_bolt_hole_{i + 1}"] = prism(s, z0 - EPS, zt + EPS) + prism(s.offset(HEAD_EXTRA, JoinType.Round), z0 - EPS, zb)
    return {name: to_trimesh(man) for name, man in regions.items()}, plate.area()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the region STLs of the drone-arm example.")
    ap.add_argument("--out", type=Path, default=OUT, help="the folder for the STL files (default: the folder of this script)")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    meshes, area = build()
    print(f"{'region':20s} {'watertight':10s} {'volume mm3':>11s}  bounds min / max")
    for name, m in meshes.items():
        assert m.is_watertight, name
        m.export(out / f"{name}.stl")
        b = m.bounds
        print(f"{name:20s} {str(m.is_watertight):10s} {m.volume:11.1f}  {np.round(b[0], 1)} / {np.round(b[1], 1)}")
    vol = float(meshes["arm_original"].volume)
    print(f"\nplain plate arm: {area:.0f} mm2 (the outline minus the holes) x {PLATE_Z[1] - PLATE_Z[0]:g} mm = {vol:.0f} mm3, "
          f"{vol * DENSITY / 1000.0:.3f} g at {DENSITY} g/cm3 (target_mass_g of drone_arm.yaml)")


if __name__ == "__main__":
    main()
