#!/usr/bin/env python3
"""Check the hints folder that `marks_to_hints` (topopt.feedback) wrote, from the files on disk.

Usage: python tools/check_hints.py RUN_DIR MARKS.json HINTS_DIR [--make [--radius-scale F]]

With --make, the script first runs `python -m topopt.feedback hints RUN_DIR MARKS.json --out HINTS_DIR`,
and hashes the run folder, the input YAML and its region STL files before and after.
Checks (each prints PASS or FAIL; exit code 0 if all pass):
  1. hints.json and the proposed YAML are in the folder, and hints.json has no absolute path.
  2. The proposed YAML loads through the normal config path (topopt.config.load_config).
  3. Every region STL that the YAML names is in the folder, and is a closed solid.
  4. Centres and radii: for each applied mark, every vertex of its region STL is within one voxel of the radius
     from the mark path (the points of the mark, and the points with y -> -y when `mirror` is set);
     the region reaches the radius around each path point; with `mirror`, both copies are there.
  5. The voxel labels of the proposed YAML differ from the labels of the input YAML only within one voxel of a
     region (the radius around a mark path), and the grid (origin, shape, voxel size) is the same.
  6. The input YAML, its region STL files and every file of the run folder have the same hash after the tool.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

results: list[bool] = []


def rec(name: str, ok: bool, info: str = "") -> None:
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")


def tree_hash(paths) -> dict:
    out = {}
    for r in paths:
        r = Path(r)
        files = [r] if r.is_file() else sorted(p for p in r.rglob("*") if p.is_file())
        for p in files:
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _absolute_paths(obj, where: str = "") -> list[str]:
    """The places in a JSON object where a text starts with / or a drive letter (an absolute path)."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _absolute_paths(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += _absolute_paths(v, f"{where}[{i}]")
    elif isinstance(obj, str) and (obj.startswith("/") or (len(obj) > 2 and obj[1:3] in (":\\", ":/"))):
        found.append(where)
    return found


def input_files(yaml_path: Path) -> list[Path]:
    """The YAML and the region files that it names (resolved relative to the YAML folder)."""
    doc = yaml.safe_load(yaml_path.read_text())
    reg = doc["regions"]
    names = [reg["design"]] + list(reg.get("fixed") or []) + list(reg.get("keep_out") or []) \
        + list((reg.get("load") or {}).values())
    return [yaml_path] + [(yaml_path.parent / n).resolve() for n in names]


def path_distance(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Distance from each point to a polyline (or to a single point)."""
    if len(path) == 1:
        return np.linalg.norm(points - path[0], axis=1)
    best = None
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(points - (a + t[:, None] * ab), axis=1)
        best = d if best is None else np.minimum(best, d)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("marks")
    ap.add_argument("hints_dir")
    ap.add_argument("--make", action="store_true", help="run the tool first (and hash before and after)")
    ap.add_argument("--radius-scale", type=float, default=None, help="passed to the tool with --make")
    a = ap.parse_args(argv)
    run_dir, marks_path, out = Path(a.run_dir).resolve(), Path(a.marks).resolve(), Path(a.hints_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    from topopt.runinfo import run_yaml_path
    in_yaml = run_yaml_path(run, run_dir)     # a relative yaml_path starts at the run folder
    watched = [run_dir, *input_files(in_yaml)]
    before = tree_hash(watched)
    if a.make:
        cmd = [sys.executable, "-m", "topopt.feedback", "hints", str(run_dir), str(marks_path), "--out", str(out)]
        if a.radius_scale:
            cmd += ["--radius-scale", str(a.radius_scale)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        rec("the tool ran", r.returncode == 0, (r.stderr.strip().splitlines() or ["ok"])[-1])
        if r.returncode != 0:
            return 1
    # 1. files
    report_path = out / "hints.json"
    rec("hints.json is in the folder", report_path.is_file(), str(report_path))
    if not report_path.is_file():
        return 1
    report = json.loads(report_path.read_text())
    new_yaml = out / Path(report["proposed_yaml"]["path"]).name     # hints.json holds paths relative to its folder
    rec("hints.json has no absolute path", not _absolute_paths(report), ", ".join(_absolute_paths(report)[:3]) or "none")
    rec("the proposed YAML is in the folder", new_yaml.parent == out and new_yaml.is_file(), new_yaml.name)
    # 2. the normal config path
    from topopt.config import UserError, load_config
    from topopt.voxelize import voxelize
    try:
        cfg = load_config(new_yaml)
        rec("the proposed YAML loads (load_config)", True,
            f"{len(cfg.regions.keep_out)} keep-out, {len(cfg.regions.load)} load regions, voxel {cfg.settings.voxel_size} mm")
    except UserError as exc:
        rec("the proposed YAML loads (load_config)", False, str(exc))
        return 1
    h = cfg.settings.voxel_size
    # 3. region STLs
    stls = [cfg.regions.design, *cfg.regions.fixed, *cfg.regions.keep_out, *[lr.path for lr in cfg.regions.load]]
    bad = []
    for p in stls:
        m = trimesh.load(str(p), file_type="stl", force="mesh")
        if p.parent != out or not (m.is_watertight and m.volume > 0):
            bad.append(p.name)
    rec("region STLs are in the folder and closed solids", not bad, f"{len(stls)} files" + (f", bad: {bad}" if bad else ""))
    # 4. centres and radii
    marks = {m["id"]: m for m in json.loads(marks_path.read_text())["marks"]}
    scale = float(report["options"]["radius_scale"])
    paths_of: dict[int, list[np.ndarray]] = {}
    n_checked = 0
    for entry in report["marks"]:
        if entry["status"] != "applied":
            continue
        mk = marks[entry["id"]]
        pts = np.asarray(mk["points"], float)
        paths = [pts] + ([pts * np.array([1.0, -1.0, 1.0])] if mk.get("mirror") else [])
        paths_of[entry["id"]] = paths
        r = float(mk["radius_mm"]) * scale
        mesh = trimesh.load(str(out / entry["region_file"]), file_type="stl", force="mesh")
        v = np.asarray(mesh.vertices)
        d = np.min([path_distance(v, p) for p in paths], axis=0)
        facts = entry["facts"]
        cut = bool(facts["applied_region"]["cut_to"]) or facts["outside_grid_box_pct"] > 0
        inside = float(d.max()) <= r + h
        tight = cut or float(np.abs(d - r).max()) <= h
        reach = True
        for p in paths:
            for q in p:
                dq = np.linalg.norm(v - q, axis=1)
                reach &= bool(cut or (dq.min() <= r + h and dq.max() >= r - h))
        both = (not mk.get("mirror")) or all(
            bool((path_distance(v, p) <= r + h).any()) for p in paths)
        ok = inside and tight and reach and both
        n_checked += 1
        rec(f"mark {entry['id']} ({mk['type']}): centre and radius within one voxel", ok,
            f"{entry['region_file']}: max |distance - radius| = {float(np.abs(d - r).max()):.3f} mm, "
            f"radius {r:.3f} mm, voxel {h} mm" + (", cut by a region or the box (looser check)" if cut else "")
            + (", both copies present" if mk.get("mirror") and both else ""))
    rec("at least one applied mark was checked", n_checked > 0, f"{n_checked} marks")
    # 5. labels
    try:
        o0, h0, s0, lab0, _ = voxelize(load_config(in_yaml))
        o1, h1, s1, lab1, _ = voxelize(cfg)
    except UserError as exc:
        rec("the input YAML and the proposed YAML voxelize", False, str(exc))
        return 1
    same_grid = (h0 == h1 and tuple(s0) == tuple(s1) and bool(np.allclose(o0, o1)))
    rec("the grid is the same as the input grid", same_grid, f"shape {list(s1)}, origin {np.round(o1, 3).tolist()}")
    if same_grid:
        idx = np.argwhere(lab0 != lab1)
        if len(idx) == 0:
            rec("labels differ only near the marks", True, "no voxel differs")
        else:
            centres = o0 + h0 * (idx + 0.5)
            best = np.full(len(idx), 1e9)
            for mid, paths in paths_of.items():
                r = float(marks[mid]["radius_mm"]) * scale
                for p in paths:
                    best = np.minimum(best, path_distance(centres, p) - r)
            worst = float(best.max())
            rec("labels differ only near the marks", worst <= h0,
                f"{len(idx)} voxels differ; the farthest is {worst:.3f} mm outside a mark radius (limit {h0} mm)")
    # 6. unchanged inputs
    after = tree_hash(watched)
    changed = [p for p in set(before) | set(after) if before.get(p) != after.get(p)]
    rec("input YAML, its STL files and the run folder are unchanged", not changed,
        f"{len(before)} files hashed before and after" + (f"; changed: {changed[:3]}" if changed else ""))
    print("RESULT:", "OK" if all(results) else "FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
