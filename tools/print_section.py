#!/usr/bin/env python3
"""Print the `print` section of the print report for a finished STL (CONTRACTS.md section 18).

Usage: python tools/print_section.py FINAL.STL [--yaml PROBLEM.yaml] [--build-direction +z]
           [--bed 220] [--printer fdm] [--angle 45] [--json]

The settings come from the `print` block of the YAML, then from the options, then from the defaults.
The text shows the values for the configured build direction and for each of the six axis directions.
`--json` prints the whole section as JSON. The tool reads the STL only: it does not change any file.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the code folder: tools/ is next to topopt/

import trimesh
import yaml

from topopt.config import UserError, parse_print_block
from topopt.check_coverage import line_of
from topopt.print_report import build_report, fit_phrase


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stl")
    ap.add_argument("--yaml", help="problem YAML with an optional print block")
    ap.add_argument("--build-direction", help="axis name (+x -x +y -y +z -z) or 'x,y,z'")
    ap.add_argument("--bed", type=float, help="bed side in mm (square)")
    ap.add_argument("--printer", help="fdm, sla, lpbf or sls")
    ap.add_argument("--angle", type=float, help="self-supporting angle from the plate, in degrees")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    block = {}
    if a.yaml:
        block = (yaml.safe_load(Path(a.yaml).read_text()) or {}).get("print") or {}
    if a.build_direction:
        s = a.build_direction
        block["build_direction"] = [float(v) for v in s.split(",")] if "," in s else s
    if a.bed is not None:
        block["bed_mm"] = a.bed
    if a.printer:
        block["printer_type"] = a.printer
    if a.angle is not None:
        block["self_supporting_angle_deg"] = a.angle
    try:
        settings = parse_print_block(block, a.yaml or "<options>")
    except UserError as e:
        sys.exit(f"error: {e}")

    mesh = trimesh.load(a.stl, force="mesh")
    t0 = time.perf_counter()
    r = build_report(mesh, settings)
    dt = time.perf_counter() - t0
    if a.json:
        print(json.dumps(r, indent=1))
        return
    s = r["settings"]
    print(f"{a.stl}: {len(mesh.faces)} triangles, area {r['total_area_mm2']:.1f} mm2, report {dt:.2f} s")
    print(line_of(r))        # TC-2: which sub-checks ran, failed or were not run (CONTRACTS.md section 44)
    print(f"settings: build {s['build_direction_label']}, bed {s['bed_mm'][0]:g} x {s['bed_mm'][1]:g} mm, "
          f"printer {s['printer_type']}, angle {s['self_supporting_angle_deg']:g} deg ({s['angle_source']}); "
          f"assumed: {', '.join(s['assumed']) or 'none'}")
    if "support_note" in r:
        print(r["support_note"])
    head = f"{'build':>10} {'overhang mm2':>13} {'contact mm2':>12} {'support mm3':>12} {'rays':>6}  {'extents (u, v, height) mm':<28} bed"
    print(head)
    rows = [("configured", r["configured"])] + [(o["label"], o) for o in r["options"]]
    for name, o in rows:
        vol = "-" if o["support_volume_mm3"] is None else f"{o['support_volume_mm3']:.0f}"
        ext = " x ".join(f"{v:.1f}" for v in o["extents_mm"])
        label = f"{name} {o['label']}" if name == "configured" else name
        print(f"{label:>10} {o['overhang_area_mm2']:13.1f} {o['plate_contact_area_mm2']:12.1f} {vol:>12} "
              f"{o['support_rays']:6d}  {ext:<28} {fit_phrase(o).replace(' degrees', ' deg')}")
    t = r["thinnest_member"]
    if t.get("error"):
        print(f"thinnest member (estimate): not measured, the estimate stopped: {t['error']}")
        return
    print(f"thinnest member (estimate): {t['thinnest_member_mm']} mm = p{t['percentile']:g} of {t['samples']} chords; "
          f"min {t['min_mm']} mm; percentiles {t['percentiles_mm']}; seed {t['seed']}")
    if t.get("area_below_mm2"):
        print("area with a chord thinner than (mm): " + ", ".join(
            f"{k} mm: {t['area_below_mm2'][k]:g} mm2 ({t['area_below_pct'][k]:g} %)" for k in t["area_below_mm2"]))


if __name__ == "__main__":
    main()
