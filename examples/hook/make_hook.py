"""Build the region STLs for the T-shirt wall hook demo.

Run:  .venv/bin/python examples/hook/make_hook.py [--no-table] [--preview]

Units: mm. The wall is the plane x = 0. The hook goes out in +x.
z is up. y is sideways. The script can run many times. It overwrites files.
Only this folder has hook-specific code. The engine stays general.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

OUT = Path(__file__).resolve().parent

# ---- parameters (mm) -------------------------------------------------------
PLATE_T = 3.5            # plate thickness (x)
PLATE_W = 35.0           # plate width (y)
PLATE_Z = (-30.0, 36.0)  # plate extent (z). Taller at the top for the top screw.
HOLE_Z_TOP = 30.0        # top screw (y = 0). Above the lip, so a screwdriver can reach it.
HOLE_Z_BOTTOM = -20.0    # bottom screw (y = 0)
HOLE_D = 4.5             # screw shaft diameter
HEAD_D = 9.0             # clearance for the screw head and the screwdriver
HEAD_X1 = 80.0           # screwdriver channel: from the plate face through the full part depth
DESIGN_W = 28.0          # design space width (y)
DESIGN_X0 = 2.0          # design space starts inside the plate
# Design space outline in the x-z plane (x, z), counter-clockwise.
DESIGN_POLY = [(DESIGN_X0, -30.0), (72.0, 8.0), (72.0, 11.0),
               (56.0, 15.0), (DESIGN_X0, 36.0)]
SADDLE_X = (56.0, 76.0)  # saddle extent in x
SADDLE_Z = (11.0, 15.0)  # saddle extent in z
SADDLE_W = 18.0          # saddle width (y)
LIP_X = (73.0, 76.0)     # lip at the tip
LIP_Z = (11.0, 24.0)     # lip top is 9 mm above the saddle top
SEG = 48                 # segments of cylinders


def box(x, y, z):
    """Box from (min, max) pairs."""
    size = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
    return Manifold.cube(size).translate([x[0], y[0], z[0]])


def xcyl(x0, x1, y, z, d):
    """Cylinder along x, from x0 to x1, axis at (y, z)."""
    c = Manifold.cylinder(x1 - x0, d / 2, d / 2, SEG)
    return c.rotate([0, 90, 0]).translate([x0, y, z])


def to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=True)


def build() -> dict[str, trimesh.Trimesh]:
    plate = box((0, PLATE_T), (-PLATE_W / 2, PLATE_W / 2), PLATE_Z)
    # Screw holes are keep-out. Each one has a shaft and a screwdriver channel
    # through the full part depth, so the screw can be driven from the front.
    holes = {}
    for name, z in (("screw_hole_top", HOLE_Z_TOP), ("screw_hole_bottom", HOLE_Z_BOTTOM)):
        holes[name] = (xcyl(-1.0, PLATE_T + 0.5, 0.0, z, HOLE_D)
                       + xcyl(PLATE_T, HEAD_X1, 0.0, z, HEAD_D))
    # Design prism: polygon in x-z, extruded along y.
    prism = CrossSection([DESIGN_POLY]).extrude(DESIGN_W)
    design = prism.translate([0, 0, -DESIGN_W / 2]).rotate([90, 0, 0])
    # Tip: saddle plus lip, one closed body.
    saddle = box(SADDLE_X, (-SADDLE_W / 2, SADDLE_W / 2), SADDLE_Z)
    lip = box(LIP_X, (-SADDLE_W / 2, SADDLE_W / 2), LIP_Z)
    tip = saddle + lip
    parts = {"design": design, "wall_plate": plate, "hook_tip": tip, **holes}
    return {k: to_trimesh(v) for k, v in parts.items()}


def report(meshes: dict[str, trimesh.Trimesh]) -> None:
    print(f"{'region':20s} {'watertight':10s} {'volume mm3':>11s}  bounds min / max")
    for name, m in meshes.items():
        b = m.bounds
        print(f"{name:20s} {str(m.is_watertight):10s} {m.volume:11.1f}  "
              f"{np.round(b[0], 1)} / {np.round(b[1], 1)}")
        if not m.is_watertight or m.volume <= 0:
            raise SystemExit(f"error: {name} is not a closed solid")
    allb = np.vstack([m.bounds for m in meshes.values()])
    print("union bounds:", np.round(allb.min(0), 1), np.round(allb.max(0), 1))


def voxel_counts(meshes: dict[str, trimesh.Trimesh], h: float) -> dict[str, int]:
    """Count voxels by label. Same rules as CONTRACTS.md section 3."""
    allb = np.vstack([m.bounds for m in meshes.values()])
    lo = np.floor(allb.min(0) / h) * h
    hi = np.ceil(allb.max(0) / h) * h
    shape = np.round((hi - lo) / h).astype(int)
    axes = [lo[a] + h * (np.arange(shape[a]) + 0.5) for a in range(3)]
    inside = {}
    for name, m in meshes.items():
        b = m.bounds
        sl = [np.where((axes[a] >= b[0][a]) & (axes[a] <= b[1][a]))[0] for a in range(3)]
        mask = np.zeros(shape, bool)
        if all(len(s) for s in sl):
            g = np.stack(np.meshgrid(*[axes[a][sl[a]] for a in range(3)],
                                     indexing="ij"), -1).reshape(-1, 3)
            mask[np.ix_(*sl)] = m.contains(g).reshape([len(s) for s in sl])
        inside[name] = mask
    keep = inside["screw_hole_top"] | inside["screw_hole_bottom"]
    fixed = inside["wall_plate"] & ~keep
    load = inside["hook_tip"] & ~keep & ~fixed
    design = inside["design"] & ~keep & ~fixed & ~load
    return {"grid": int(np.prod(shape)), "design": int(design.sum()),
            "fixed": int(fixed.sum()), "load": int(load.sum()),
            "active": int(design.sum() + fixed.sum() + load.sum())}


def main() -> None:
    meshes = build()
    for name, m in meshes.items():
        m.export(OUT / f"{name}.stl")
    report(meshes)
    if "--no-table" not in sys.argv:
        print("\nvoxel size | grid voxels | design | fixed | load | active")
        for h in (0.8, 1.0, 1.25, 1.5):
            c = voxel_counts(meshes, h)
            print(f"{h:10.2f} | {c['grid']:11d} | {c['design']:6d} | "
                  f"{c['fixed']:5d} | {c['load']:4d} | {c['active']:6d}")
    if "--preview" in sys.argv:
        preview(meshes)


def preview(meshes) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("matplotlib is not installed. No preview.")
        return
    colors = {"design": "#9ecae1", "wall_plate": "#d62728", "hook_tip": "#2ca02c",
              "screw_hole_top": "#ff7f0e", "screw_hole_bottom": "#ff7f0e"}
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for name, m in meshes.items():
        alpha = 0.25 if name == "design" else 0.9
        ax.add_collection3d(Poly3DCollection(m.vertices[m.faces], alpha=alpha,
                                             facecolor=colors[name], linewidths=0.1))
    ax.set_xlim(0, 80); ax.set_ylim(-40, 40); ax.set_zlim(-40, 40)
    ax.set_xlabel("x (out)"); ax.set_ylabel("y"); ax.set_zlabel("z (up)")
    ax.view_init(25, -60)
    fig.savefig(OUT / "preview.png", dpi=110)
    print("wrote preview.png")


if __name__ == "__main__":
    main()
