#!/usr/bin/env python3
"""Calibrate the rule of `settings.min_member_mm` on one small case (CONTRACTS.md sections 22a and 22b).

For each requested minimum member size the script runs the normal path (`topopt run`: YAML,
`load_config`, `build_problem`, the optimizer, the post-process) and measures the thinnest member
of the result. It writes the numbers to a JSON file, so the plan document can state the measured
error of the rule.

    python tools/calibrate_min_member.py --out docs/a3a-min-member-calibration.json

With `--robust-eta 0.55 0.6` each request is also run in robust mode (`settings.robust: eroded`, CONTRACTS.md
section 22b) at each threshold eta_e. The thickness is then measured on the nominal design (the snapshot), and the
row has the nominal and the eroded compliance. The A3b calibration is `docs/a3b-eta-calibration.json`.

The voxel-count guards of the YAML are in force (the case sets none above the defaults). The case is
a small cantilever bracket on a 1 mm grid, with two load cases and a long beta schedule (a sharper design than the
default; it is not crisp: the result has a grayness, `4 rho (1 - rho)` averaged over the design voxels, of about 0.1).

Measure (voxel level, no mesh): the material is the density 0.5 level of the last snapshot. The local
thickness of a material voxel is the diameter of the largest ball in the material that holds it (a
distance rule on a grid refined by 2). The script reports the 1st and 5th percentile and the median of
the design material in three sets: all of it ("all"), the part that is at least MARGIN voxels away
from the free faces of the design space ("interior"), and the rest, within MARGIN voxels of a free face
("face_layer"). The density filter has one side only at a free
face, so a skin there can be about half as thin as an interior member. The measure has an error of
about a quarter of a voxel. It is not the thickness of the final mesh: measure that with the thickness
report of the print check.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy import ndimage

from topopt.post import _last_snapshot

# The case: mm. The grid is 48 x 24 x 16 voxels at H, plus the plate.
H = 1.0
DESIGN = ((0, 0, 0), (48, 24, 16))
PLATE = ((-2, 0, 0), (0, 24, 16))
TIP = ((46, 10, 12), (48, 14, 16))
CASES = [("down", 1.0, [0, 0, -1.0]), ("side", 1.0, [0, 1.0, -0.5])]
VOLUME_FRACTION = 0.3
ITERATIONS = 150
BETA_SCHEDULE = [[0, 1], [40, 2], [70, 4], [100, 8], [130, 16]]   # a longer run than the default: a sharper design (still gray, see `grayness`)
VALUES_MM = (2.0, 2.5, 3.0)     # requested minimum member sizes (2 to 3 voxels)
MARGIN = 2                      # voxels: the "interior" is this far from the free faces of the design space


def box_stl(path: Path, lo, hi) -> None:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def write_case(folder: Path, settings: dict) -> Path:
    """Write the region STLs and the YAML of the case. `settings` are added to `settings`."""
    box_stl(folder / "design.stl", *DESIGN)
    box_stl(folder / "plate.stl", *PLATE)
    box_stl(folder / "tip.stl", *TIP)
    doc = {
        "name": "calibration_bracket",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
        "load_cases": [{"name": n, "weight": w, "forces": [{"region": "tip", "force": f}]}
                       for n, w, f in CASES],
        "settings": {"voxel_size": H, "volume_fraction": VOLUME_FRACTION, "iterations": ITERATIONS,
                     "beta_schedule": BETA_SCHEDULE, **settings},
    }
    path = folder / "case.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def local_thickness(material: np.ndarray, h: float, step: float = 0.25) -> np.ndarray:
    """Local thickness (mm) of each material voxel: the diameter of the largest ball that holds it.

    The grid is refined by 2 (each voxel gives 2 x 2 x 2 cells), so an even thickness has a centre
    cell. The distance is to the nearest empty cell centre, and the edge of the grid counts as empty.
    A member of m cells across gives m - 0.5 (m even) or m + 0.5 (m odd) cells: the bias is
    a quarter of a voxel at most.
    """
    fine = np.repeat(np.repeat(np.repeat(material, 2, 0), 2, 1), 2, 2)
    edt = ndimage.distance_transform_edt(np.pad(fine, 1))[1:-1, 1:-1, 1:-1]
    thick = np.zeros(fine.shape)
    for r in np.arange(1.0, edt.max() + step, step):
        core = edt >= r
        if not core.any():
            break
        opened = ndimage.distance_transform_edt(~core) <= r
        thick[opened & fine] = 2.0 * r - 0.5
    nx, ny, nz = material.shape
    return thick.reshape(nx, 2, ny, 2, nz, 2).mean(axis=(1, 3, 5)) * (h / 2.0)   # mean of the 8 cells


def interior_mask(shape, margin: int) -> np.ndarray:
    """Voxels at least `margin` voxels from the free faces of the design space.

    The grid starts at the plate (x low). The face next to the plate is not free (solid neighbours),
    so only the x high face, and both y and z faces, count.
    """
    nx, ny, nz = shape
    i, j, k = np.indices(shape)
    return (i < nx - margin) & (j >= margin) & (j < ny - margin) & (k >= margin) & (k < nz - margin)


def summarize(t: np.ndarray, request_mm: float) -> dict:
    """Percentiles of the local thickness values `t` (mm) and the share below the request."""
    if t.size == 0:
        return {"voxels": 0}
    return {"voxels": int(t.size),
            "min_mm": round(float(t.min()), 3),
            "p1_mm": round(float(np.percentile(t, 1)), 3),
            "p5_mm": round(float(np.percentile(t, 5)), 3),
            "p10_mm": round(float(np.percentile(t, 10)), 3),
            "median_mm": round(float(np.median(t)), 3),
            "share_below_request": round(float((t < request_mm).mean()), 4)}


def measure(density: np.ndarray, labels: np.ndarray, h: float, request_mm: float, margin: int = MARGIN) -> dict:
    """The thickness numbers of a result. `density` is the full grid (0..1), `labels` the voxel labels."""
    from topopt.voxelize import DESIGN as D
    material = density > 0.5
    t = local_thickness(material, h)
    design = material & (labels == D)
    inner = interior_mask(material.shape, margin)
    return {"all": summarize(t[design], request_mm),
            "interior": summarize(t[design & inner], request_mm),
            "face_layer": summarize(t[design & ~inner], request_mm),
            "face_layer_share_of_material": round(float((design & ~inner).sum() / max(int(design.sum()), 1)), 3)}


def run_value(request_mm: float, explicit_factor: float | None, volume_fraction: float = VOLUME_FRACTION,
              robust_eta: float | None = None) -> dict:
    """One run of `topopt run` for one request. Return the facts of the run and the thickness numbers.

    `robust_eta`: run in robust mode (settings.robust eroded) with this eta_e."""
    from topopt.cli import main as topopt_main
    from topopt.config import load_config

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        settings = ({"min_member_mm": request_mm} if explicit_factor is None
                    else {"filter_radius": explicit_factor * request_mm})
        settings = {**settings, "volume_fraction": volume_fraction}
        if robust_eta is not None:
            settings.update({"robust": "eroded", "eta_eroded": robust_eta})
        yaml_path = write_case(folder, settings)
        st = load_config(yaml_path).settings
        out = folder / "run"
        t0 = time.time()
        if topopt_main(["run", str(yaml_path), "--out", str(out), "--quiet"]) != 0:
            raise SystemExit(f"topopt run failed for {request_mm} mm")
        seconds = round(time.time() - t0, 1)
        metrics = json.loads((out / "result" / "metrics.json").read_text())
        run_json = json.loads((out / "run.json").read_text())
        shape = tuple(run_json["shape"])
        labels = np.frombuffer((out / "labels.bin").read_bytes(), np.uint8).reshape(shape)
        last = _last_snapshot(out)   # the highest pass number (a text sort puts it_10000.bin before it_9999.bin)
        density = np.frombuffer(last.read_bytes(), np.uint8).reshape(shape) / 255.0
    m = measure(density, labels, H, request_mm)
    rb = metrics.get("robust") or {"on": False}
    return {"requested_mm": request_mm, "robust_eta": robust_eta, "filter_radius_mm": st.filter_radius,
            "radius_source": st.filter_radius_source, "radius_voxels": round(st.filter_radius / H, 3),
            "thickness": m,
            "error_p5_interior": round(m["interior"]["p5_mm"] / request_mm - 1.0, 3),
            "error_median_interior": round(m["interior"]["median_mm"] / request_mm - 1.0, 3),
            "error_p5_all": round(m["all"]["p5_mm"] / request_mm - 1.0, 3),
            "compliance_final": metrics["compliance_final"], "mass_final_g": metrics["mass_final_g"],
            "compliance_nominal": rb.get("compliance_nominal"), "compliance_eroded": rb.get("compliance_eroded"),
            "robust_s": rb.get("robust_s"),
            "grayness": round(float(np.mean(4 * density[labels == 1] * (1 - density[labels == 1]))), 4),
            "metrics_filter": metrics["filter"], "solver": run_json["solver"],
            "n_free_dofs": run_json["n_free_dofs"], "seconds": seconds,
            "peak_rss_mib_so_far": peak_rss_mib()}


def peak_rss_mib(maxrss: int | None = None) -> float:
    """Peak resident memory of this process in MiB. `ru_maxrss` is in KiB on Linux and in bytes on macOS."""
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if maxrss is None else maxrss
    return round(raw / (2 ** 20 if sys.platform == "darwin" else 1024.0), 1)


def calibrate(values, explicit_factor=None, volume_fraction=VOLUME_FRACTION, robust_etas=(None,)) -> dict:
    from topopt import members
    rows = []
    for eta in robust_etas:
        for v in values:
            row = run_value(v, explicit_factor, volume_fraction, eta)
            rows.append(row)
            a, b = row["thickness"]["all"], row["thickness"]["interior"]
            tag = "" if eta is None else f", robust eta_e {eta:g}"
            print(f"requested {v:g} mm{tag}: R {row['filter_radius_mm']:g} mm; thickness all p1 {a['p1_mm']:.2f} "
                  f"p5 {a['p5_mm']:.2f}; interior p1 {b['p1_mm']:.2f} p5 {b['p5_mm']:.2f} median {b['median_mm']:.2f} "
                  f"({100 * row['error_p5_interior']:+.0f} % at p5); compliance {row['compliance_final']:.4g}, "
                  f"{row['seconds']} s", flush=True)
    return {"case": {"design_mm": DESIGN, "plate_mm": PLATE, "tip_mm": TIP, "voxel_mm": H,
                     "volume_fraction": volume_fraction, "iterations": ITERATIONS,
                     "beta_schedule": BETA_SCHEDULE, "load_cases": [{"name": n, "weight": w, "force_n": f} for n, w, f in CASES]},
            "rule": (members.RULE_TEXT if explicit_factor is None else
                     f"filter_radius = {explicit_factor:g} * request, set in the YAML (exploration, not the engine rule)"),
            "radius_per_member": members.RADIUS_PER_MEMBER,   # the constant of the engine rule
            "explicit_factor": explicit_factor,
            "robust_etas": [e for e in robust_etas if e is not None],
            "measure": {"kind": "local thickness of the design material (density > 0.5), voxel level, grid refined by 2",
                        "interior_margin_voxels": MARGIN, "bias_mm": 0.25 * H},
            "machine": {"platform": platform.platform(), "python": sys.version.split()[0]},
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="write the JSON data file here")
    ap.add_argument("--values", type=float, nargs="+", default=list(VALUES_MM),
                    help="requested minimum member sizes in mm")
    ap.add_argument("--volume-fraction", type=float, default=VOLUME_FRACTION,
                    help="volume fraction of the case (default %(default)s): a sensitivity check")
    ap.add_argument("--explicit-factor", type=float,
                    help="exploration: set filter_radius = factor * value in the YAML instead of min_member_mm")
    ap.add_argument("--robust-eta", type=float, nargs="+",
                    help="also run each request in robust mode at these thresholds eta_e (section 22b)")
    a = ap.parse_args()
    data = calibrate(a.values, a.explicit_factor, a.volume_fraction, tuple(a.robust_eta) if a.robust_eta else (None,))
    if a.out:
        a.out.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
