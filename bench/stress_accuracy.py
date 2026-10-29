"""How accurate is the hex8 voxel stress? Two cases, several voxel sizes, theory as the reference.

The stress of the optimizer report (`topopt.stress_report`) is `D0 * strain` with the full modulus at the
centre of each hex8 voxel (`topopt.refine.stress.compute_stress`). This script measures its error:

(a) beam: an aligned cantilever (36 x 6 x 6 mm) with a tip load. Reference: beam theory My/I plus the
    parabolic shear. Window: x from 12 to 24 mm (at least two beam heights from the clamp, 12 mm from the
    tip). Voxel sizes: 1, 2, 4, 8 and 16 voxels through the thickness.
(b) hole: a plate (30 x 10 x 0.5 mm) with a central hole of 2 mm diameter (d/W = 0.2) under tension. The hole
    is a staircase of voxels. Reference for the peak: Peterson's finite-width factor, Kt(net) = 3.00 - 3.13
    (d/W) + 3.66 (d/W)^2 - 1.53 (d/W)^3 = 2.508 times the net section stress (chart value, about 1 %).
    Reference for p99.9: the Kirsch field of the infinite plate, scaled to the Peterson peak; the band
    from the two scalings (the gross stress, and the Peterson peak) is the reference uncertainty.
    Voxel sizes: hole diameter of 4, 8 and 16 voxels.

Statistics: the peak and the p99.9 by volume of the von Mises stress over the voxels of the window (a) or over
all design voxels (b). Error = voxel / reference - 1. The voxel statistics are for the design voxels, like the report.
Every run stays under the default voxel-count guard (`settings.max_active_voxels` 100000).

(c) optional cross-check: a finished hook run with `verify/verify.json` of B2 (`--cross-check RUN_DIR`): the voxel
    p99.9 and peak against the P2 values of the same part. This is another model, not theory.

Verdict: no statistic has a verdict for now. The data also show that many "p99.9" rows have
fewer than 5000 voxels: the top 0.1 % of the volume is then fewer than 5 voxels and p99.9 is a peak (`p99_9_equals_peak` in each
row).

The band (F1-8c) is the union of two sets of data: the rows of this bench (a beam, a plate
with a hole and one real part, the hook against P2) and the 51 cells of 5 public parts (`bench/stress_public_parts.json`, made by
`bench/stress_public_parts.py`: the voxel p99.9 and the voxel peak against the P2 values of `topopt verify`). `derive_rule` takes the
largest error on each side over both sets, with one rounding rule. The entry keeps the two sets apart (`sources`), so that a text can
name the set and the row or cell that sets each end of the band. The band is the largest error measured, not a bound for other parts.

Run:  python bench/stress_accuracy.py [--out bench/stress_accuracy.json] [--cross-check RUN_DIR]
      python bench/stress_accuracy.py --rederive-rule [--out bench/stress_accuracy.json]
Time: about 1 minute with a direct solver. `--rederive-rule` runs no bench: it reads the rows in the file and the cells in
`bench/stress_public_parts.json`, and writes `verdict_rule` again. After it, run `python bench/stress_public_parts.py --resummarize`
(the summary there counts against the bench band alone). The data files are committed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml

from topopt.config import load_config
from topopt.problem import build_problem
from topopt.refine.stress import compute_stress
from topopt.stress_report import (BENCH_FILE, MIN_MEMBER_VOXELS_EST, PART_STRESS_CHECK, PUBLIC_PARTS_FILE, band_range_text,
                                  band_sources_text, bench_band_text, p99_9_equals_peak, percentile_by_volume, range_text,
                                  top_tail_voxels)

HERE = Path(__file__).resolve().parent
GUARD = 100000          # the default `settings.max_active_voxels`
BEAM = dict(L=36.0, b=6.0, t=6.0, P=10.0, x0=12.0, x1=24.0)
HOLE = dict(L=30.0, W=10.0, t=0.5, d=2.0, sigma_gross=10.0)
BEAM_N = (1, 2, 4, 8, 16)        # voxels through the thickness
HOLE_N = (4, 8, 16)              # voxels across the hole diameter


def box(path, lo, hi):
    m = trimesh.creation.box(extents=np.asarray(hi, float) - np.asarray(lo, float))
    m.apply_translation((np.asarray(lo, float) + np.asarray(hi, float)) / 2)
    m.export(path)


def solve(folder: Path, doc: dict):
    """Voxelize, solve with all design voxels at rho = 1, return (problem, stress dict, seconds)."""
    p = folder / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    cfg = load_config(p)
    problem = build_problem(cfg)
    n_active = int(((problem.labels >= 1) & (problem.labels <= 3)).sum())
    if n_active > GUARD:
        raise RuntimeError(f"{n_active} active voxels: above the guard {GUARD}")
    t0 = time.time()
    stress = compute_stress(problem, np.full(int(np.prod(problem.shape)), 255, np.uint8))
    return problem, stress, n_active, time.time() - t0


def force_centroid(problem, case=0):
    """Force-weighted centre of the nodal forces of a load case (mm)."""
    f = problem.forces[case]
    nodes = np.flatnonzero(np.abs(f.reshape(-1, 3)).sum(axis=1))
    nx, ny, nz = problem.shape
    ijk = np.stack(np.unravel_index(nodes, (nx + 1, ny + 1, nz + 1)), axis=1)
    w = np.abs(f.reshape(-1, 3)[nodes]).sum(axis=1)
    return np.asarray(problem.origin) + problem.h * (ijk * w[:, None]).sum(axis=0) / w.sum()


# --- (a) beam -----------------------------------------------------------------


def beam_theory(x, zp, xc, P=BEAM["P"], b=BEAM["b"], t=BEAM["t"]):
    """Von Mises of beam theory: bending stress My/I and parabolic shear. `zp` from the mid height."""
    I = b * t ** 3 / 12.0
    sig = P * (xc - x) * zp / I
    tau = 1.5 * P / (b * t) * (1.0 - (2.0 * zp / t) ** 2)
    return np.sqrt(sig ** 2 + 3.0 * tau ** 2)


def run_beam(n):
    g = BEAM
    h = g["t"] / n
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        box(folder / "design.stl", (0, 0, 0), (g["L"], g["b"], g["t"]))
        box(folder / "wall.stl", (-6, 0, 0), (0, g["b"], g["t"]))
        box(folder / "tip.stl", (g["L"], 0, 0), (g["L"] + 6, g["b"], g["t"]))
        doc = {"name": "beam", "regions": {"design": "design.stl", "fixed": ["wall.stl"], "load": {"tip": "tip.stl"}},
               "load_cases": [{"name": "tip", "forces": [{"region": "tip", "force": [0, 0, -g["P"]]}]}],
               "settings": {"voxel_size": h, "volume_fraction": 0.5, "iterations": 1}}
        problem, stress, n_active, secs = solve(folder, doc)
    c = stress["centers"]
    vm = stress["vm_cases"][0]
    design = stress["design"]
    win = design & (c[:, 0] >= g["x0"]) & (c[:, 0] <= g["x1"])
    xc = force_centroid(problem)[0]
    zp = c[:, 2] - g["t"] / 2.0
    th_centre = beam_theory(c[:, 0], zp, xc)           # theory at the voxel centres
    # continuum reference over the window: fine grid of (x, z'), uniform in y
    xs = np.linspace(g["x0"], g["x1"], 1201)
    zs = np.linspace(-g["t"] / 2, g["t"] / 2, 241)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    th_field = beam_theory(X, Z, xc)
    ref_peak = float(th_field.max())
    ref_p999 = percentile_by_volume(th_field)
    v = vm[win]
    peak, p999 = float(v.max()), percentile_by_volume(v)
    # the statistics of the report: all design voxels, the clamp end and the load end included. Reference:
    # the same beam theory over the whole design length (it has no clamp or tip effect).
    xs_all = np.linspace(0.0, g["L"], 2401)
    Xa, Za = np.meshgrid(xs_all, zs, indexing="ij")
    th_all = beam_theory(Xa, Za, xc)
    va = vm[design]
    peak_all, p999_all = float(va.max()), percentile_by_volume(va)
    n_win = int(win.sum())
    ref_peak_all, ref_p999_all = float(th_all.max()), percentile_by_volume(th_all)
    x_peak = float(c[design][int(np.argmax(va)), 0])
    return {
        "whole_design": {
            "peak_MPa": peak_all, "peak_reference_MPa": ref_peak_all,
            "peak_error_pct": 100 * (peak_all / ref_peak_all - 1), "peak_at_x_mm": x_peak,
            "p99_9_MPa": p999_all, "p99_9_reference_MPa": ref_p999_all,
            "p99_9_error_pct": 100 * (p999_all / ref_p999_all - 1),
            "voxels": int(va.size), "p99_9_equals_peak": p99_9_equals_peak(int(va.size))},
        "voxels_through_thickness": n, "voxel_size_mm": h, "active_voxels": n_active, "window_voxels": int(win.sum()),
        "solve_s": round(secs, 2), "load_centroid_x_mm": float(xc),
        "peak_MPa": peak, "peak_reference_MPa": ref_peak, "peak_error_pct": 100 * (peak / ref_peak - 1),
        "peak_error_at_voxel_centres_pct": 100 * (peak / float(th_centre[win].max()) - 1),
        "p99_9_MPa": p999, "p99_9_reference_MPa": ref_p999, "p99_9_error_pct": 100 * (p999 / ref_p999 - 1),
        "p99_9_error_at_voxel_centres_pct": 100 * (p999 / percentile_by_volume(th_centre[win]) - 1),
        "window_p99_9_equals_peak": p99_9_equals_peak(n_win)}


# --- (b) plate with a hole ---------------------------------------------------------


def peterson_kt_net(r):
    return 3.00 - 3.13 * r + 3.66 * r ** 2 - 1.53 * r ** 3


def kirsch_vm(x, y, a, s):
    """Von Mises (plane stress) of the infinite plate with a hole of radius `a`, tension `s` along x."""
    r2 = np.maximum(x ** 2 + y ** 2, a * a)
    th = np.arctan2(y, x)
    q, q2 = (a * a) / r2, (a * a / r2) ** 2
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    sr = 0.5 * s * (1 - q) + 0.5 * s * (1 - 4 * q + 3 * q2) * c2
    st = 0.5 * s * (1 + q) - 0.5 * s * (1 + 3 * q2) * c2
    tr = -0.5 * s * (1 + 2 * q - 3 * q2) * s2
    return np.sqrt(sr ** 2 - sr * st + st ** 2 + 3 * tr ** 2)


def hole_reference():
    g = HOLE
    a, ratio = g["d"] / 2, g["d"] / g["W"]
    sg = g["sigma_gross"]
    s_net = sg * g["W"] / (g["W"] - g["d"])
    kt = peterson_kt_net(ratio)
    peak = kt * s_net
    # field on a fine grid over the plate (x, y from the hole centre), points inside the hole are dropped
    xs = np.arange(-g["L"] / 2 + 0.005, g["L"] / 2, 0.01)
    ys = np.arange(-g["W"] / 2 + 0.005, g["W"] / 2, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    keep = X ** 2 + Y ** 2 > a * a
    out = {}
    for name, s_inf in (("matched", peak / 3.0), ("gross", sg)):
        out[name] = percentile_by_volume(kirsch_vm(X, Y, a, s_inf)[keep])
    return {"kt_net": kt, "sigma_net_MPa": s_net, "peak_MPa": peak,
            "p99_9_matched_MPa": out["matched"], "p99_9_gross_MPa": out["gross"]}


def run_hole(n):
    g = HOLE
    h = g["d"] / n
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        box(folder / "design.stl", (0, 0, 0), (g["L"], g["W"], g["t"]))
        box(folder / "wall.stl", (-1, 0, 0), (0, g["W"], g["t"]))
        box(folder / "pull.stl", (g["L"], 0, 0), (g["L"] + 1, g["W"], g["t"]))
        hole = trimesh.creation.cylinder(radius=g["d"] / 2, height=3 * g["t"], sections=720)
        hole.apply_translation((g["L"] / 2, g["W"] / 2, g["t"] / 2))
        hole.export(folder / "hole.stl")
        P = g["sigma_gross"] * g["W"] * g["t"]
        doc = {"name": "hole", "regions": {"design": "design.stl", "fixed": ["wall.stl"], "keep_out": ["hole.stl"],
                                           "load": {"pull": "pull.stl"}},
               "load_cases": [{"name": "pull", "forces": [{"region": "pull", "force": [P, 0, 0]}]}],
               "settings": {"voxel_size": h, "volume_fraction": 0.5, "iterations": 1}}
        problem, stress, n_active, secs = solve(folder, doc)
    ref = hole_reference()
    v = stress["vm_cases"][0][stress["design"]]
    peak, p999 = float(v.max()), percentile_by_volume(v)
    return {
        "voxels_across_hole": n, "voxel_size_mm": h, "active_voxels": n_active, "design_voxels": int(v.size),
        "solve_s": round(secs, 2), "peak_MPa": peak, "peak_reference_MPa": ref["peak_MPa"],
        "peak_error_pct": 100 * (peak / ref["peak_MPa"] - 1),
        "p99_9_MPa": p999, "p99_9_reference_MPa": ref["p99_9_matched_MPa"],
        "p99_9_error_pct": 100 * (p999 / ref["p99_9_matched_MPa"] - 1),
        "p99_9_error_vs_gross_scaling_pct": 100 * (p999 / ref["p99_9_gross_MPa"] - 1),
        "p99_9_equals_peak": p99_9_equals_peak(int(v.size))}, ref


MIN_MEMBER_VOXELS = 4      # voxels through the thinnest loaded member
MIN_HOLE_VOXELS = 4        # voxels across the diameter of a hole or fillet


def cross_check(run_dir) -> dict:
    """A real shape: the voxel statistics of a finished run against the P2 check of B2 (`verify/verify.json`).

    The reference is the quadratic-tet solution of B2 on `final.stl` (p99.9 by volume, and the peak at the
    quadrature points). It is another model of the same part, not theory. It is read-only data: this
    function writes nothing. The member thickness is the estimate of `topopt.stress_report.member_thickness`.
    """
    import hashlib

    from topopt.post import _last_snapshot, build_field
    from topopt.refine.build import problem_from_run
    from topopt.stress_report import member_check, physical_statistics

    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    problem = problem_from_run(run_dir, run)
    snap = np.frombuffer(_last_snapshot(run_dir).read_bytes(), np.uint8)
    stress = compute_stress(problem, snap)
    solid = build_field(snap.astype(np.float32).reshape(problem.shape) / 255.0, problem.labels) >= 0.5
    vpath = run_dir / "verify" / "verify.json"
    ver = json.loads(vpath.read_text())
    out = {"run": run.get("name"), "voxel_size_mm": float(problem.h),
           "verify_json_sha256": hashlib.sha256(vpath.read_bytes()).hexdigest(),
           "reference": "B2 P2 check (quadratic tets on final.stl): p99.9 by volume, peak at the quadrature points",
           "cases": {}}
    for i, p in enumerate(physical_statistics(stress)):
        pv = ver["cases"][p["name"]]["von_mises"]
        m = member_check(stress, solid, problem.origin, problem.h, i)
        out["cases"][p["name"]] = {
            "voxel_p99_9_MPa": p["p99_9_by_volume_MPa"], "p2_p99_9_MPa": pv["p99_9_by_volume_MPa"],
            "p99_9_error_pct": 100 * (p["p99_9_by_volume_MPa"] / pv["p99_9_by_volume_MPa"] - 1),
            "voxel_peak_MPa": p["peak_MPa"], "p2_peak_qp_MPa": pv["peak_qp_MPa"],
            "peak_error_pct": 100 * (p["peak_MPa"] / pv["peak_qp_MPa"] - 1),
            "design_voxels": p["voxels"], "p99_9_equals_peak": p["p99_9_equals_peak"],
            "member_voxels_p10": m["p10_voxels"], "member_voxels_median": m["median_voxels"]}
    out["split"] = hook_split(run_dir, problem, snap, ver)
    return out


NEAR_MM = 2.0   # voxels and P2 points within this distance of a load or fixed region count as "near"
# P2 values of the B2 fix round that this script cannot recompute (they need a P2 run with other loads).
STORED_P2_PULL_OFF = {"source": "stored values of an earlier P2 run (fix round of the load patch)",
                  "p2_p99_9_near_region_left_out_MPa": 1.95,
                  "p2_p99_9_nodal_spread_like_the_voxel_model_MPa": 1.63,
                  "p2_stiffness_nodal_spread_N_per_mm": 1091}


def hook_split(run_dir, problem, snap, ver) -> dict:
    """Split the hook comparison into effects (REVIEW M3 of A2, `runs/b2conf.json` of the review).

    1. voxel p99.9 as the report makes it (grey voxels, SIMP stiffness) against the P2 value of `verify.json`;
    2. grey voxels out: the thresholded design (rho >= 0.5 solid, the rest void), the geometry of `final.stl`;
    3. load patch out: voxels within `NEAR_MM` of a load or fixed region left out. P2 side: the stored value of the B2
       fix round, and a proxy from `verify/fields.npz` (element maximum with tet volumes, the ratio far / all);
    4. load model like for like: the P2 value with the nodal spread of the voxel model.
    Stiffness is given for each model: it shows how much of the gap is the load model.
    """
    from scipy.ndimage import distance_transform_edt

    from topopt.voxelize import FIXED, LOAD

    h, lab = problem.h, problem.labels
    d_near = np.minimum(distance_transform_edt(lab != LOAD), distance_transform_edt(lab != FIXED)) * h

    def voxel_numbers(density_u8):
        st = compute_stress(problem, density_u8)
        ijk = np.floor((st["centers"] - np.asarray(problem.origin)) / h).astype(int)
        near = d_near[ijk[:, 0], ijk[:, 1], ijk[:, 2]] <= NEAR_MM
        res = {}
        for i, name in enumerate(st["case_names"]):
            vm, des = st["vm_cases"][i], st["design"]
            f = problem.forces[i].reshape(-1, 3).sum(axis=0)
            res[name] = {"p99_9_MPa": percentile_by_volume(vm[des]), "peak_MPa": float(vm[des].max()),
                         "p99_9_far_from_regions_MPa": percentile_by_volume(vm[des & ~near]),
                         "stiffness_N_per_mm": float(f @ f / st["compliance_cases"][i])}
        return res

    grey = voxel_numbers(snap)
    thresholded = voxel_numbers(np.where(snap >= 128, 255, 0).astype(np.uint8))
    proxy = {}
    fpath = Path(run_dir) / "verify" / "fields.npz"
    if fpath.is_file():
        f = np.load(fpath)
        pts, tets = f["p"].T, f["t"].T
        a, b, c, d = (pts[tets[:, k]] for k in range(4))
        vol = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
        ijk = np.clip(np.floor(((a + b + c + d) / 4.0 - np.asarray(problem.origin)) / h).astype(int), 0,
                      np.array(problem.shape) - 1)
        near = d_near[ijk[:, 0], ijk[:, 1], ijk[:, 2]] <= NEAR_MM

        def wperc(v, w, q=99.9):
            o = np.argsort(v)
            cw = np.cumsum(w[o]) / w.sum()
            return float(v[o][min(int(np.searchsorted(cw, q / 100.0)), v.size - 1)])

        for name in grey:
            v = f[f"vm_elem_max_{name}"].astype(float)
            allp, farp = wperc(v, vol), wperc(v[~near], vol[~near])
            top = v >= allp
            proxy[name] = {"element_max_p99_9_all_MPa": allp, "element_max_p99_9_far_MPa": farp,
                           "far_over_all": farp / allp,
                           "share_of_top_0_1_percent_volume_near_regions": float(vol[top & near].sum() / vol[top].sum())}
    cases = {}
    for name in grey:
        p2 = ver["cases"][name]["von_mises"]["p99_9_by_volume_MPa"]
        cases[name] = {
            "voxel": {"grey_simp_as_reported": grey[name], "thresholded": thresholded[name]},
            "p2_p99_9_surface_traction_MPa": p2,
            "p2_stiffness_surface_traction_N_per_mm": ver["cases"][name]["stiffness_N_per_mm"],
            "p2_element_max_proxy": proxy.get(name),
            "error_pct": {
                "grey_vs_p2": 100 * (grey[name]["p99_9_MPa"] / p2 - 1),
                "thresholded_vs_p2": 100 * (thresholded[name]["p99_9_MPa"] / p2 - 1)}}
    pull = cases.get("pull_off")
    if pull:
        g = grey["pull_off"]
        pull["load_patch_left_out"] = {
            "voxel_grey_far_MPa": g["p99_9_far_from_regions_MPa"],
            "p2_far_MPa": STORED_P2_PULL_OFF["p2_p99_9_near_region_left_out_MPa"],
            "error_pct": 100 * (g["p99_9_far_from_regions_MPa"] / STORED_P2_PULL_OFF["p2_p99_9_near_region_left_out_MPa"] - 1)}
        pull["load_model_like_for_like"] = {
            "voxel_grey_MPa": g["p99_9_MPa"],
            "p2_nodal_spread_MPa": STORED_P2_PULL_OFF["p2_p99_9_nodal_spread_like_the_voxel_model_MPa"],
            "error_pct": 100 * (g["p99_9_MPa"] / STORED_P2_PULL_OFF["p2_p99_9_nodal_spread_like_the_voxel_model_MPa"] - 1),
            "stiffness_ratio_voxel_over_p2_traction": g["stiffness_N_per_mm"] / pull["p2_stiffness_surface_traction_N_per_mm"],
            "stiffness_ratio_voxel_over_p2_nodal": g["stiffness_N_per_mm"] / STORED_P2_PULL_OFF["p2_stiffness_nodal_spread_N_per_mm"]}
        pull["stored_values"] = STORED_P2_PULL_OFF
    return {
        "what": "the hook against P2 (B2) split into effects: the grey voxels, the load patch, the load model",
        "near_mm": NEAR_MM, "cases": cases,
        "reading": "About 12 points of the pull_off error of -53 % come from the load model (surface traction in P2, nodal spread "
                   "in the voxel model); about -41 % is a voxel error of this part (hex8 at about 4 voxels through the member, "
                   "stair steps). The grey voxels and the load patch do not explain it."}


def public_parts():
    """The sibling script `bench/stress_public_parts.py` (the folder `bench` is not a package)."""
    spec = importlib.util.spec_from_file_location("stress_public_parts", HERE / "stress_public_parts.py")
    assert spec is not None and spec.loader is not None, "bench/stress_public_parts.py is missing"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _edges(rows) -> dict:
    """The band of a list of rows `(error in percent, name of the row)`: the largest error on each side, rounded up to whole
    percent (`under_pct`, `over_pct`), and the measured ends with the name of the row that sets each one."""
    import math
    lo, hi = min(rows, key=lambda r: r[0]), max(rows, key=lambda r: r[0])
    return {"under_pct": int(math.ceil(max(0.0, -lo[0]))), "over_pct": int(math.ceil(max(0.0, hi[0]))),
            "measured_min_pct": round(lo[0], 2), "measured_min_from": lo[1],
            "measured_max_pct": round(hi[0], 2), "measured_max_from": hi[1]}


def derive_rule(beam, hole, cross=None, public=None) -> dict:
    """The rule that the data support.

    The band of a statistic is the largest error on each side over the rows of the bench and (with `public`) the cells of the 5
    public parts, rounded up to whole percent. For the rows of the bench: the sizes that meet the two minimum voxel counts. The
    errors of the beam are those of the whole design (the statistic of the report), the window values give no larger error. For
    p99.9 of the hole, both references (the Peterson scaling and the gross scaling) count. A cross-check against B2 (a real shape,
    if given) counts for both statistics: it is the only bench data on an organic design.

    `public` is the list of the cells of `bench/stress_public_parts.json` (F1-8, Refs #129; F1-8c). Their error is the voxel
    value against the P2 value of `topopt verify`: for p99.9 `voxel_error_pct`, for the peak `100 * (1 / p2_peak_over_voxel_peak
    - 1)`. The band is then the union of the two sets. The entry keeps them apart in `sources` (`bench`, `public`, and which of
    the two sets each end of the band comes from) with the row or cell that sets each end, so that a text can name the source of
    each end. `measured_min_pct` and `measured_max_pct` are the union. Without `public` the entry is the band of the bench alone.

    No statistic has a verdict ("no verdict without data"):

    - Each end of a band is set by one row or cell, with no margin.
    - The member gate does not predict the error: the hook passes the gate (p10 3.84 voxels) and its error is -53 %, while the
      beam at 4 voxels has -28 %.
    - Rows with fewer than 5000 voxels have a "p99.9" that is the peak (`p99_9_equals_peak`).
    - The peak is below the reference in every measured case, the sign for other shapes is not known, and the peak of a corner or
      a load point depends on the voxel size.

    The band is the largest error measured, and not a bound for other parts: the public cells widened the band of the bench (the
    lower edge of p99.9 from -54 % to -59 %, the upper edge from +13 % to +57 %), and the member gate did not keep the error
    inside the bench band.
    """
    use_b = [r for r in beam if r["voxels_through_thickness"] >= MIN_MEMBER_VOXELS]
    use_h = [r for r in hole if r["voxels_across_hole"] >= MIN_HOLE_VOXELS]
    beam_name = lambda r: f"beam, {r['voxels_through_thickness']} voxels through the thickness"
    hole_name = lambda r, note="": f"plate with a hole, {r['voxels_across_hole']} voxels across the hole{note}"
    # the rows of the bench: (error in percent, name of the row)
    peak = ([(r["whole_design"]["peak_error_pct"], beam_name(r)) for r in use_b]
            + [(r["peak_error_pct"], hole_name(r)) for r in use_h])
    p999 = ([(r["whole_design"]["p99_9_error_pct"], beam_name(r)) for r in use_b]
            + [(r["p99_9_error_pct"], hole_name(r, ", Peterson scaling of the reference")) for r in use_h]
            + [(r["p99_9_error_vs_gross_scaling_pct"], hole_name(r, ", gross scaling of the reference")) for r in use_h])
    # rows of the band with a true percentile (fewer voxels than 5000 means the "p99.9" is the peak)
    true_rows = ([r["whole_design"]["p99_9_error_pct"] for r in use_b if not r["whole_design"]["p99_9_equals_peak"]]
                 + [r[k] for r in use_h if not r["p99_9_equals_peak"]
                    for k in ("p99_9_error_pct", "p99_9_error_vs_gross_scaling_pct")])
    n_rows = len(use_b) + 2 * len(use_h)
    if cross:
        hook_name = lambda k: f"the hook, {k}, voxel {cross['voxel_size_mm']:g} mm, against P2"
        peak += [(c["peak_error_pct"], hook_name(k)) for k, c in cross["cases"].items()]
        p999 += [(c["p99_9_error_pct"], hook_name(k)) for k, c in cross["cases"].items()]
        true_rows += [c["p99_9_error_pct"] for c in cross["cases"].values() if not c["p99_9_equals_peak"]]
        n_rows += len(cross["cases"])
    bench_peak, bench_p999 = _edges(peak), _edges(p999)
    pp: Any = public_parts() if public else None
    summary: dict = {}
    pub_p999: list = []
    pub_peak: list = []
    if public:
        # the summary counts against the BENCH band alone: the union contains every cell by construction
        summary = pp.summarize(public, bench_p999["under_pct"], bench_p999["over_pct"], MIN_MEMBER_VOXELS_EST)
        pub_p999 = [(c["voxel_error_pct"], pp.cell_label(c)) for c in public]
        pub_peak = [(round(pp.error_pct(c["p2_peak_over_voxel_peak"]), 2), pp.cell_label(c)) for c in public]
    peak, p999 = peak + pub_peak, p999 + pub_p999
    parts_n = len({c["part"] for c in public}) if public else 0

    def sources(bench_edges, pub_rows):
        out: dict[str, Any] = {"bench": {"data": BENCH_FILE, **bench_edges}}
        if public:
            out["public"] = {"data": PUBLIC_PARTS_FILE, "parts": parts_n, "cells": len(public), **_edges(pub_rows)}
            out["lower_end"] = "bench" if bench_edges["measured_min_pct"] <= out["public"]["measured_min_pct"] else "public"
            out["upper_end"] = "bench" if bench_edges["measured_max_pct"] >= out["public"]["measured_max_pct"] else "public"
        else:
            out["lower_end"] = out["upper_end"] = "bench"
        return out

    common = {"min_voxels": MIN_MEMBER_VOXELS, "min_voxels_hole_diameter": MIN_HOLE_VOXELS,
              "data": BENCH_FILE + (f" and {PUBLIC_PARTS_FILE}" if public else ""),
              "cases": ["beam", "hole"] + (["hook_vs_B2"] if cross else []) + (["public_parts"] if public else [])}
    valid_when = ("the members at the high-stress voxels have an estimated 3.5 voxels or more through them "
                  "(the report checks this with `member_thickness`), and a hole or fillet has at least 4 voxels "
                  "across its diameter (the report does not check this)")
    pub_true = sum(1 for c in public if not c["p99_9_equals_peak"]) if public else 0
    bench_db = {"shapes": 2 + (1 if cross else 0), "real_parts": 1 if cross else 0, "rows": n_rows,
                "rows_where_p99_9_is_a_true_percentile": len(true_rows),
                "rows_where_p99_9_equals_the_peak": n_rows - len(true_rows)}
    note = "bench shapes: beam, plate with a hole, and the hook against B2 (the only real part of the bench)"
    data_base = {"rows": n_rows, "rows_where_p99_9_is_a_true_percentile": len(true_rows),
                 "rows_where_p99_9_equals_the_peak": n_rows - len(true_rows), "bench": bench_db}
    if public:
        hook_pub = sorted({c["voxel_mm"] for c in public if c["part"] == "hook"})
        data_base["rows"] += len(public)
        data_base["rows_where_p99_9_is_a_true_percentile"] += pub_true
        data_base["rows_where_p99_9_equals_the_peak"] += len(public) - pub_true
        data_base["public"] = {"parts": parts_n, "cells": len(public), "cells_where_p99_9_is_a_true_percentile": pub_true,
                               "cells_where_p99_9_equals_the_peak": len(public) - pub_true}
        note += (f"; public parts: {', '.join(sorted({c['part'] for c in public}))} (a cell is a part, a load case and a run type)"
                 + (f"; the hook is in both sets, with other runs (bench {cross['voxel_size_mm']:g} mm, public "
                    f"{', '.join(f'{v:g}' for v in hook_pub)} mm)" if cross and hook_pub else ""))
    data_base["note"] = note

    # --- p99.9
    e999 = {"verdict": False, **_union(p999), **common, "sources": sources(bench_p999, pub_p999),
            "valid_when": valid_when, "data_base": data_base}
    edge = min(cross["cases"].items(), key=lambda kv: kv[1]["p99_9_error_pct"]) if cross else None
    b4 = next((r for r in beam if r["voxels_through_thickness"] == MIN_MEMBER_VOXELS), None)
    gate = ""
    if edge and b4 and cross:
        gate = (f"The member gate does not predict the error: the hook at {cross['voxel_size_mm']:g} mm passes it "
                f"(p10 {edge[1]['member_voxels_p10']:.2f} voxels) and has {edge[1]['p99_9_error_pct']:+.0f} %, while the beam at "
                f"{MIN_MEMBER_VOXELS} voxels has {b4['p99_9_error_pct']:+.0f} %. ")
    db = data_base
    base_text = (f"The band of the error of the voxel p99.9 (voxel value / reference - 1) is {band_range_text(e999)} of the "
                 f"reference" + (", the union of the bench and the public parts" if public else "")
                 + f": {band_sources_text(e999)}. Each end is set by one {'row or cell' if public else 'row'}, with no margin. ")
    if public:
        base_text += f"The bench alone gave {bench_band_text(e999)}. "
    data_text = (f"The data base is {db['bench']['rows']} rows of the bench ({db['bench']['shapes']} shapes, "
                 f"{db['bench']['real_parts']} real part)"
                 + (f" and {db['public']['cells']} cells of {db['public']['parts']} public parts" if public else "")
                 + f"; {db['bench']['rows_where_p99_9_equals_the_peak']} of the {db['bench']['rows']} rows"
                 + (f" and {db['public']['cells_where_p99_9_equals_the_peak']} of the {db['public']['cells']} cells"
                    if public else "")
                 + " have fewer than 5000 voxels, so their p99.9 is the peak. ")
    reason999 = ("no verdict for now (no verdict without data). " + base_text + gate + data_text
                 + ("The band is the largest error measured, not a bound for other parts: "
                    f"{pp.fact_text(summary)}. {PART_STRESS_CHECK}" if public else ""))
    if public:
        gm = summary["member_gate"]
        e999["valid_when"] = (valid_when + ". It is the condition of the bench and not a guarantee: on public parts "
                              f"{gm['cells_outside_bench_band']} of the {gm['cells_meeting']} cells that met the member gate had an "
                              f"error outside the bench band ({range_text(bench_p999['under_pct'], bench_p999['over_pct'])}, "
                              f"{BENCH_FILE}; cells in {PUBLIC_PARTS_FILE})")
    e999["reason"] = reason999.rstrip()

    # --- peak
    epk = {"verdict": False, **_union(peak), **common, "sources": sources(bench_peak, pub_peak),
           "valid_when": valid_when, "data_base": data_base}
    if public:
        gate_cells = [c for c in public if c["member_p10_voxels"] >= MIN_MEMBER_VOXELS_EST]
        beyond = sum(1 for c in gate_cells if pp.error_pct(c["p2_peak_over_voxel_peak"]) < -bench_peak["under_pct"])
        epk["valid_when"] = (valid_when + ". It is the condition of the bench and not a guarantee: on public parts "
                             f"{beyond} of the {len(gate_cells)} cells that met the member gate had a voxel peak more than "
                             f"{bench_peak['under_pct']} % below the P2 peak ({bench_peak['under_pct']} % is the lower end of the bench "
                             f"band, {BENCH_FILE}; cells in {PUBLIC_PARTS_FILE})")
    epk["reason"] = (
        "the voxel peak is below the reference in every measured case (the centre of a voxel lies half a voxel below the "
        "surface), the sign is not known for other shapes, and the peak of a corner or load point depends on the voxel size. "
        f"The band of the error (voxel value / reference - 1) is {band_range_text(epk)} of the reference: {band_sources_text(epk)}."
        + (f" The reference of the public cells is the P2 peak at the quadrature points; the P2 peak depends on the mesh of "
           f"`topopt verify` (`limits` in {PUBLIC_PARTS_FILE}). {PART_STRESS_CHECK}" if public else ""))
    return {"p99_9": e999, "peak": epk,
            "error_definition": "error = voxel value / reference - 1. The reference lies between value / (1 + over) "
                                "and value / (1 - under)"}


def _union(rows) -> dict:
    """`under_pct`, `over_pct` and the measured ends of the union of the bench and the public cells (`rows` holds both sets)."""
    e = _edges(rows)
    return {k: e[k] for k in ("under_pct", "over_pct", "measured_min_pct", "measured_max_pct")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "stress_accuracy.json"))
    ap.add_argument("--cross-check", metavar="RUN_DIR",
                    help="a finished run folder that has verify/verify.json (B2): adds the voxel against P2 numbers")
    ap.add_argument("--rederive-rule", action="store_true",
                    help="no bench run: read --out, make `verdict_rule` again from the sizes and the cross-check in it "
                         "(a new text of the `reason` needs no new measurement), and write it back")
    a = ap.parse_args()
    public = public_parts().load()["cells"]
    if a.rederive_rule:
        doc = json.loads(Path(a.out).read_text())
        doc["verdict_rule"] = derive_rule(doc["beam"]["sizes"], doc["hole"]["sizes"], doc["cross_check"], public)
        Path(a.out).write_text(json.dumps(doc, indent=1))
        print("wrote the rule again into", a.out)
        return
    beam = [run_beam(n) for n in BEAM_N]
    for r in beam:
        print("beam", r["voxels_through_thickness"], f"peak {r['peak_error_pct']:+.1f} %  p99.9 {r['p99_9_error_pct']:+.1f} %", flush=True)
    hole, ref = [], None
    for n in HOLE_N:
        r, ref = run_hole(n)
        hole.append(r)
        print("hole", n, f"peak {r['peak_error_pct']:+.1f} %  p99.9 {r['p99_9_error_pct']:+.1f} %", flush=True)
    cross = cross_check(a.cross_check) if a.cross_check else None
    if cross:
        for k, c in cross["cases"].items():
            print("cross-check", k, f"p99.9 {c['p99_9_error_pct']:+.1f} %  peak {c['peak_error_pct']:+.1f} %", flush=True)
    doc = {
        "format": 1, "script": "bench/stress_accuracy.py",
        "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
        "measure": "von Mises of D0 * strain, full E, hex8 centre strain, design voxels, rho = 1 (compute_stress)",
        "guard_active_voxels": GUARD,
        "beam": {"geometry_mm": BEAM, "reference": "Euler-Bernoulli My/I and parabolic shear, load at the force centroid",
                 "window": "x from 12 to 24 mm, design voxels", "sizes": beam},
        "hole": {"geometry_mm": HOLE, "reference": "Peterson Kt(net) for the peak; Kirsch field scaled to the peak for p99.9",
                 "reference_values": ref, "sizes": hole},
        "cross_check": cross,
        "verdict_rule": derive_rule(beam, hole, cross, public),
    }
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print("rule:", json.dumps({k: v for k, v in doc["verdict_rule"].items() if k != "error_definition"}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
