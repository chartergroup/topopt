#!/usr/bin/env python3
"""Apply the marks of a marks.json to a run, one mark kind at a time, and check each output from the files on disk.

Usage: python tools/check_apply.py RUN_DIR MARKS.json [--out DIR] [--pitch MM] [--allow-stale] [--all-at-once]

For each kind that the marks file holds (remove, smooth, add) the script calls `topopt.apply_marks.apply_marks` with
the marks of that kind only, into DIR/<kind> (default DIR: `<run folder>-marks-check`). It prints the facts of each
mark (status, mode, volume and mass change, clip by the frozen regions, thickness near the mark, parent check, bodies)
and the output paths. With --all-at-once it also applies every mark in one call, into DIR/all.
Checks (each prints PASS or FAIL; exit code 0 if all pass), from the files that the call wrote:
  1. The run folder, the YAML and its region STL files have the same hash after the calls.
  2. edited.stl loads again from disk as one closed solid (watertight, one winding, one body), and its volume is the
     volume that the report gives.
  3. apply_marks.json is there and has no absolute path.
  4. The volume goes in the direction of the kind: down for remove, up for add (smooth: either way).
A kind whose marks all have no effect (or are rolled back) writes no edited.stl: the script prints the reason and does
not count it as a failure.
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import apply_marks as am  # noqa: E402
from topopt import feedback  # noqa: E402

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


ABS_IN_TEXT = re.compile(r"(?<![\w.:/])/[^\s\"'`,;()\[\]]+/[^\s\"'`,;()\[\]]+|(?<![\w])[A-Za-z]:[\\\\/][^\s\"'`,;()]+"
                         r"|(?:\.\./){3,}")


def absolute_paths(obj, where: str = "") -> list[str]:
    """An absolute path, or a relative one that climbs 3 folders or more, anywhere in the text of the report (also in the
    middle of a sentence)."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += absolute_paths(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += absolute_paths(v, f"{where}[{i}]")
    elif isinstance(obj, str):
        for m in ABS_IN_TEXT.finditer(obj):
            found.append(f"{where}={m.group(0)[:60]}")
    return found


def show_marks(rep: dict) -> None:
    for e in rep["marks"]:
        ed = e.get("edit") or {}
        pc = e.get("parent_check") or {}
        th = e.get("thickness_near_mm") or {}
        print(f"  mark {e['id']} {e['type']}: {e['status']}"
              + (f" ({'; '.join(e.get('modes') or [])})" if e.get("modes") else "")
              + (f" - {e['reason']}" if e.get("reason") else ""))
        if ed:
            print(f"    volume {ed.get('part_volume_change_mm3')} mm3 ({ed.get('part_mass_change_g')} g); added "
                  f"{ed.get('added_mm3')}, removed {ed.get('removed_mm3')}; clipped by frozen regions "
                  f"{ed.get('clipped_by_frozen_mm3')} mm3 ({ed.get('clipped_share_pct')} %); reach "
                  f"{ed.get('farthest_changed_node_from_mark_mm')} mm")
            print(f"    thickness near the mark (min, median) {th.get('before_min')}, {th.get('before_median')} -> "
                  f"{th.get('after_min')}, {th.get('after_median')} mm; bodies {e.get('bodies_after')}; "
                  f"closed {e.get('closed_after')}; nearest frozen {e.get('nearest_frozen_to_edit_mm')} mm"
                  + (f"; parent check within_tolerance={pc.get('within_tolerance')} (drop {pc.get('max_drop_mm')} mm)" if pc else ""))
        if e.get("defaults_used"):
            print("    pipeline defaults used: " + ", ".join(f"{d['name']}={d['value']}" for d in e["defaults_used"]))
        sn = e.get("stress_near_mark") or {}
        if sn.get("near_max_mpa") is not None:
            print(f"    stress near the mark (before the edit): max {sn['near_max_mpa']} MPa, p99 of the part {sn['global_p99_mpa']} MPa")


def check_output(label: str, kind: str | None, rep: dict) -> None:
    out = Path(rep["out_dir"])
    stl = out / "edited.stl"
    t = rep["totals"]
    if rep["files"]["edited_stl"] is None:
        print(f"  no edited.stl for {label}: no mark changed the part (see the reasons above)")
    else:
        m = trimesh.load(str(stl), file_type="stl", force="mesh")
        bodies = len(trimesh.graph.connected_components(m.face_adjacency))
        rec(f"{label}: edited.stl reloads as one closed solid", m.is_watertight and m.is_winding_consistent and bodies == 1
            and abs(m.volume - t["volume_after_mm3"]) < 0.02,
            f"watertight {m.is_watertight}, bodies {bodies}, volume {m.volume:.2f} (report {t['volume_after_mm3']}), {len(m.faces)} triangles")
        d = t["change_mm3"]
        if kind == "remove":
            rec(f"{label}: volume goes down", d < 0, f"{d} mm3")
        elif kind == "add":
            rec(f"{label}: volume goes up", d > 0, f"{d} mm3")
    rj = out / "apply_marks.json"
    ok = rj.is_file()
    ab = absolute_paths(json.loads(rj.read_text())) if ok else ["missing"]
    rec(f"{label}: apply_marks.json is there and has no absolute path", ok and not ab, "; ".join(ab[:3]))
    print(f"  output: {out}  ({', '.join(sorted(p.name for p in out.iterdir()))})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("marks")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pitch", type=float, default=None, help="node spacing of the crop box in mm (default: voxel size / 4)")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--all-at-once", action="store_true")
    a = ap.parse_args()
    run = Path(a.run_dir).expanduser().resolve()
    doc = feedback.read_marks(a.marks)
    run_json = json.loads((run / "run.json").read_text())
    ypath = feedback._run_yaml_path(run_json, run)
    inputs = [run] + ([ypath.parent] if ypath is not None and ypath.is_file() else [])
    before = tree_hash(inputs)
    out_root = Path(a.out).expanduser().resolve() if a.out else Path(tempfile.mkdtemp(prefix=f"{run.name}-marks-check-"))
    opts = {"pitch_mm": a.pitch} if a.pitch else None
    kinds = [k for k in am.KIND_ORDER if any(isinstance(m, dict) and m.get("type") == k for m in doc["marks"])]
    print(f"run {run}\nmarks {a.marks}: {len(doc['marks'])} marks; kinds: {', '.join(kinds) or 'none'}\nout {out_root}")
    jobs = [(k, {**doc, "marks": [m for m in doc["marks"] if isinstance(m, dict) and m.get("type") == k]}) for k in kinds]
    if a.all_at_once:
        jobs.append((None, doc))
    for kind, sub in jobs:
        label = kind or "all"
        print(f"\n== {label}")
        rep = am.apply_marks(run, sub, out_dir=out_root / label, options=opts, allow_stale=a.allow_stale)
        if not rep.get("ok"):
            rec(f"{label}: the call ran", False, str(rep.get("error")))
            continue
        show_marks(rep)
        t = rep["totals"]
        print(f"  totals: volume {t['volume_before_mm3']} -> {t['volume_after_mm3']} mm3 ({t['change_mm3']}), mass "
              f"{t['mass_before_g']} -> {t['mass_after_g']} g, bodies {t['bodies_before']} -> {t['bodies_after']}, closed "
              f"{t['closed_after']}; frozen exact check {(t.get('frozen_exact') or {}).get('unchanged')}; "
              f"{rep['elapsed_s']} s, peak {rep['peak_rss_mb']} MB, load {rep['load_avg']}")
        check_output(label, kind, rep)
    rec("the run folder and the input folder have the same hashes after the calls", tree_hash(inputs) == before,
        f"{len(before)} files")
    print(f"\n{sum(results)} of {len(results)} checks pass")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
