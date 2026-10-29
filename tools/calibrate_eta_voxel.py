#!/usr/bin/env python3
"""Mesh-free calibration of the eroded threshold on the voxel grid (CONTRACTS.md section 22b).

A thin wall or a round rod is a field of 1 (inside) and 0 (outside) on a voxel grid. The script filters
it with the real `build_filter` of the engine and reads the filtered value at the centre of the member.
A member survives the eroded design when this value reaches the threshold eta_e (the eroded density is
then at least 0.5 at beta -> infinity). The script gives, for each filter radius R (in voxels) and each
eta, the thinnest wall and the thinnest rod (in voxels) that survive, next to the continuum values
(an infinite hat filter, closed form for the wall and the flat face, numeric for the rod).

    python tools/calibrate_eta_voxel.py --out docs/a3b-eta-voxel-fixture.json

The pipeline made the same test on its tet mesh (`calib_h16`, rods and plates of 3 to 10 mm, R from 3 to
4.5 mm, eta from 0.55 to 0.7). This is the test for the voxel grid of the demo. It needs no solve and no
mesh, so it runs in seconds. The grid has whole voxels: a wall of n voxels has a centre value that steps
with n. The column "survive" is the smallest whole number that reaches eta.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from topopt.filter import build_filter, heaviside_project
from topopt.robust import plate_limit_per_radius
from topopt.voxelize import DESIGN

RADII = (2.0, 3.0, 3.75, 4.69, 5.63, 7.5)      # filter radius in voxels (the A3a rule gives 3.75 to 5.63 for 2 to 3 mm at 1 mm)
ETAS = (0.5, 0.55, 0.6, 0.65, 0.7)
MAX_MEMBER = 24                                  # voxels: the largest member that the search tries


# --- continuum values (infinite hat filter w = R - d) -------------------------------------------


def flat_face_depth_per_radius(eta: float) -> float:
    """Depth (in radii) inside a flat face where the filtered value is `eta`: v - v^3 + v^4/2 = eta - 0.5."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid - mid ** 3 + mid ** 4 / 2 < eta - 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rod_value(a_per_radius: float, n: int = 20001) -> float:
    """Filtered value on the axis of a rod of radius a (in radii): (6 / R^4) * int_0^a W2(r) r dr.

    W2(r) = R sqrt(R^2 - r^2) - r^2 asinh(sqrt(R^2 - r^2) / r) is the kernel integrated along the axis.
    """
    r = np.linspace(0.0, a_per_radius, n)
    z = np.sqrt(np.maximum(1.0 - r * r, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        w2 = z - np.where(r > 0, r * r * np.arcsinh(z / np.where(r > 0, r, 1.0)), 0.0)
    return float(6.0 * np.trapezoid(w2 * r, r))


def rod_limit_per_radius(eta: float) -> float:
    """Diameter (in radii) of the thinnest rod whose axis reaches `eta`."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if rod_value(mid) < eta:
            lo = mid
        else:
            hi = mid
    return 2.0 * 0.5 * (lo + hi)


# --- the voxel grid -----------------------------------------------------------------------------


def _filtered(field: np.ndarray, radius: float) -> np.ndarray:
    """rho_tilde of a 0/1 field on a grid where every voxel is a design voxel (no solid, no void)."""
    labels = np.full(field.shape, DESIGN, dtype=np.uint8)
    F, _FT, b = build_filter(labels, 1.0, radius)
    return (F @ field.reshape(-1).astype(float) + b).reshape(field.shape)


def wall_centre(n: int, radius: float) -> float:
    """Largest filtered value in a wall of `n` voxels (a plate in x, all of y and z), grid margin of one radius."""
    m = int(math.ceil(radius)) + 1
    field = np.zeros((n + 2 * m, 2 * m + 1, 2 * m + 1))
    field[m:m + n] = 1.0
    return float(_filtered(field, radius)[m:m + n, m, m].max())


def rod_centre(d: int, radius: float) -> float:
    """Filtered value on the axis of a round rod of `d` voxels across (along x), grid margin of one radius."""
    m = int(math.ceil(radius)) + 1
    ny = d + 2 * m
    j = np.arange(ny) - (ny - 1) / 2.0
    disc = (j[:, None] ** 2 + j[None, :] ** 2) <= (d / 2.0) ** 2
    field = np.zeros((2 * m + 1, ny, ny))
    field[:, disc] = 1.0
    f = _filtered(field, radius)
    axis = np.argwhere(disc & (np.hypot(j[:, None], j[None, :]) <= 0.71))   # the voxel(s) at the axis
    return float(max(f[m, a, b] for a, b in axis))


def smallest(centre, radius: float, eta: float):
    """Smallest whole member size (voxels) whose centre value reaches `eta`, and that value. None if none up to MAX_MEMBER."""
    for n in range(1, MAX_MEMBER + 1):
        v = centre(n, radius)
        if v >= eta:
            return n, v
    return None, None


def calibrate(radii=RADII, etas=ETAS) -> dict:
    rows = []
    for R in radii:
        for eta in etas:
            n_wall, v_wall = smallest(wall_centre, R, eta)
            n_rod, v_rod = smallest(rod_centre, R, eta)
            rows.append({
                "radius_voxels": R, "eta": eta,
                "wall_survive_voxels": n_wall, "wall_centre_value": None if v_wall is None else round(v_wall, 4),
                "wall_continuum_voxels": round(plate_limit_per_radius(eta) * R, 3),
                "rod_survive_voxels": n_rod, "rod_centre_value": None if v_rod is None else round(v_rod, 4),
                "rod_continuum_voxels": round(rod_limit_per_radius(eta) * R, 3),
                "flat_face_erosion_voxels": round(flat_face_depth_per_radius(eta) * R, 3),
            })
            print(f"R {R:g} eta {eta:g}: wall {n_wall} (continuum {rows[-1]['wall_continuum_voxels']}), "
                  f"rod {n_rod} (continuum {rows[-1]['rod_continuum_voxels']}), "
                  f"face erosion {rows[-1]['flat_face_erosion_voxels']} voxels", flush=True)
    # The projected value at the threshold is not 0.5 for eta above 0.5 (a small effect that falls with beta).
    at_threshold = {f"eta_{e:g}": {f"beta_{b:g}": round(float(heaviside_project(np.array([e]), b, e)[0]), 5)
                                   for b in (1, 2, 4, 8, 16)} for e in etas}
    at_half = {f"eta_{e:g}": {f"beta_{b:g}": round(float(heaviside_project(np.array([0.5]), b, e)[0]), 4)
                              for b in (1, 2, 4, 8, 16)} for e in etas}
    return {"what": "thinnest wall and rod (whole voxels) whose filtered centre value reaches eta, real build_filter, "
                    "mesh-free (no solve)",
            "continuum": "infinite hat filter w = R - d: wall 2u - 2u^3 + u^4 = eta with u = t / 2R; flat face "
                         "v - v^3 + v^4/2 = eta - 0.5; rod by numeric integration",
            "rows": rows,
            "projected_at_threshold": at_threshold,
            "eroded_density_at_rho_tilde_0.5": at_half,
            "machine": {"platform": platform.platform(), "python": sys.version.split()[0]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="write the JSON data file here")
    a = ap.parse_args()
    data = calibrate()
    if a.out:
        a.out.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
