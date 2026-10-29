"""Check result/final.stl of a run folder after it is loaded again from disk (CONTRACTS.md section 19).

    .venv/bin/python tools/check_final_stl.py <run_dir> [--json]

The mesh is read from the file as float32, as a slicer reads it. The script reports:
- watertight, consistent winding, number of bodies, zero-area faces;
- for each keep-out STL: the volume of the mesh inside it (mm^3);
- for each fixed and load STL: the volume it has that the mesh does not hold (mm^3)
  and how far that missing part reaches from the mesh surface (mm). The two numbers are reported
  separately. A region fails if either one is over its tolerance;
- without region STL files: the numbers against the label voxel boxes (`labels.bin`): fixed volume
  that the mesh does not hold and mesh volume inside the keep-out labels (resolution: one fine voxel).

The exit code is 0 if the mesh is one watertight body and the keep-out and region numbers
are at or under the tolerances (`--tol-mm3`, `--tol-mm`), else 1. With labels only, the volumes
count against `--tol-label-mm3` (default 5 fine voxels of volume at the voxel size).

A run with `interfaces` (CONTRACTS.md section 23) is checked more. The tool reads `result/final.stl` (the as-printed
mesh) and, with stock, `result/nominal.stl` (the finished mesh) from disk. Each file gets the reload check. For each
built-in interface the tool measures the bore, boss or tube radius, or the face position, in each file, and reports the
deviation from the YAML value and the stock that the two files show. The interface cutters count as keep-out shapes. The
size tolerance is the chord sag of the facets plus 1e-4 mm (`--tol-interface-mm` replaces it). Each surface also gets a coverage: the area found divided by the area the mesh should show (the shape's wall where there
is material behind it). A surface under 50 percent coverage is not within tolerance. A surface that is not in the mesh is
reported as not found; it fails if metrics.json found it at post time, or if the mesh has material behind where it should be
(a filled hole). A bore in empty space has no expected area and does not fail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _within(feat: dict, tol: float | None) -> bool:
    """A found feature is within the tolerance (`tol`, else its own tolerance) and covers enough of its expected area.

    A feature that is not found is not within. A feature without an expected area (no body, or a surface in empty space)
    has no coverage test.
    """
    from topopt import interfaces

    if not feat.get("found"):
        return False
    cov = feat.get("coverage")
    if cov is not None and cov < interfaces.COVERAGE_MIN:
        return False
    return abs(feat["deviation_mm"]) <= (feat["tolerance_mm"] if tol is None else tol)


def _recorded_found(run_dir: Path) -> dict:
    """(interface name, variant, feature) -> found, from metrics.json at post time. Empty if there is no record."""
    try:
        rec = json.loads((run_dir / "result" / "metrics.json").read_text()).get("interfaces") or {}
    except (OSError, ValueError):
        return {}
    out = {}
    for it in rec.get("items", []):
        for var in ("as_printed", "nominal"):
            for feat, m in ((it.get(var) or {}).get("measured") or {}).items():
                out[(it["name"], var, feat)] = bool(m.get("found"))
    return out


def _interface_checks(run_dir: Path, run: dict, ctx, mesh, tol: float | None, tol_mm3: float, files_ok: bool, keep: dict) -> dict:
    """Section 23 checks of final.stl and nominal.stl: reload check, keep-out, measured size and stock per interface."""
    import trimesh

    from topopt import interfaces, smooth_sdf

    problems: list[str] = []
    nom_path = run_dir / "result" / "nominal.stl"
    nominal = None
    if nom_path.is_file():
        nominal = trimesh.load(nom_path, file_type="stl")
    if ctx.has_stock and nominal is None:
        problems.append("the run has stock, but result/nominal.stl is not there")
    if nominal is not None and not ctx.has_stock:
        problems.append("result/nominal.stl is there, but no interface has stock (an old file)")
    out = {"nominal_file": str(nom_path) if nominal is not None else None, "nominal_mesh": None, "nominal_keep_out": None,
           "items": [], "problems": problems}
    if nominal is not None:
        nf = smooth_sdf.stl_facts(nominal.export(file_type="stl"))
        out["nominal_mesh"] = nf
        if not (nf["ok"] and nf["bodies"] == 1):
            problems.append("nominal.stl is not one closed manifold after the reload")
        if files_ok:
            cut = ctx.cutters(interfaces.NOMINAL)
            rc = smooth_sdf.region_checks(nominal.vertices, nominal.faces, {}, {**keep, **cut}) if (keep or cut) else {"keep_out": {}}
            out["nominal_keep_out"] = rc["keep_out"]
            bad = [k for k, v in rc["keep_out"].items() if v["intersect_mm3"] > tol_mm3]
            if bad:
                problems.append("nominal.stl holds material inside: " + ", ".join(bad))
    recorded = _recorded_found(run_dir)
    meshes = {"as_printed": mesh, "nominal": nominal}
    bodies = {var: (interfaces.mesh_body(m.vertices, m.faces) if m is not None else None) for var, m in meshes.items()}
    for spec in ctx.specs:
        it = {"name": spec.name, "kind": spec.kind, "op": spec.op, "stock_mm": spec.stock_mm, "as_printed": None, "nominal": None,
              "stock_measured_mm": None}
        for var, m in meshes.items():
            if m is None or spec.kind == "stl":
                continue
            feats = interfaces.measure(spec, var, m.vertices, m.faces, body=bodies[var], box=ctx.box, removed=ctx.removed(var))
            for fk, f in feats.items():
                f["within_tolerance"] = _within(f, tol)
                area = f.get("expected_area_mm2") or 0.0
                if not f["found"] and recorded.get((spec.name, var, fk)):
                    problems.append(f"{spec.name} ({var}) {fk}: found at post time, not found in the file")
                elif not f["found"] and area >= interfaces.AREA_MIN_MM2:
                    problems.append(f"{spec.name} ({var}) {fk}: no surface found, the mesh has material behind {area:.1f} mm2 of it "
                                    "(the shape is filled or missing)")
                elif f["found"] and not f["within_tolerance"]:
                    cov = f.get("coverage")
                    if cov is not None and cov < interfaces.COVERAGE_MIN:
                        problems.append(f"{spec.name} ({var}) {fk}: the surface covers {100 * cov:.0f} percent of its expected area "
                                        f"({f['area_mm2']} of {area:.1f} mm2), under {100 * interfaces.COVERAGE_MIN:.0f} percent")
                    else:
                        problems.append(f"{spec.name} ({var}) {fk}: deviation {f['deviation_mm']:.6f} mm is over the tolerance")
            it[var] = feats
        if it["as_printed"] and it["nominal"]:
            st = interfaces._stock_between(spec, it["as_printed"], it["nominal"])
            it["stock_measured_mm"] = st
            for fk, v in st.items():
                if v is None:
                    continue
                ta, tn = (tol, tol) if tol is not None else (it["as_printed"][fk]["tolerance_mm"], it["nominal"][fk]["tolerance_mm"])
                if abs(v - spec.stock_mm) > ta + tn:
                    problems.append(f"{spec.name} {fk}: the files show stock {v:.6f} mm, the YAML says {spec.stock_mm} mm")
        out["items"].append(it)
    out["ok"] = not problems
    return out


def check_run(run_dir, tol_mm3: float = 1e-3, tol_mm: float = 1e-3, tol_label_mm3: float | None = None,
              tol_interface_mm: float | None = None) -> dict:
    import numpy as np
    import trimesh

    from topopt import interfaces, post, smooth_sdf
    from topopt.runinfo import resolved_run

    run_dir = Path(run_dir)
    run = resolved_run(json.loads((run_dir / "run.json").read_text()), run_dir)   # a relative yaml_path starts at the run folder
    mesh = trimesh.load(run_dir / "result" / "final.stl", file_type="stl")
    facts = smooth_sdf.stl_facts(mesh.export(file_type="stl"))
    fixed, load, keep = post.region_paths(run)
    regions = post.check_regions(fixed, load)
    out = {"file": str(run_dir / "result" / "final.stl"), "mesh": facts, "regions": None, "keep_out": None, "labels": None,
           "problems": []}
    why = out["problems"]   # one reason line for each failed check
    if not (facts["ok"] and facts["bodies"] == 1):
        why.append(f"final.stl is not one closed manifold after the reload (bodies {facts['bodies']}, open edges {facts['open_edges']}, "
                   f"non-manifold edges {facts['nonmanifold_edges']}, zero-area faces {facts['zero_area_faces']})")
    files_ok = bool(regions) and all(Path(q).is_file() for q in list(regions.values()) + list(keep.values()))
    ctx = None
    if run.get("interfaces"):
        ctx = interfaces.from_run(run, run["origin"], float(run["h"]), tuple(run["shape"]))
    ok = facts["ok"] and facts["bodies"] == 1
    if files_ok:
        keep_all = {**keep, **ctx.cutters(interfaces.AS_PRINTED)} if ctx is not None else keep
        rc = smooth_sdf.region_checks(mesh.vertices, mesh.faces, regions, keep_all)
        out["regions"], out["keep_out"] = rc["fixed"], rc["keep_out"]
        ok = ok and all(v["intersect_mm3"] <= tol_mm3 for v in rc["keep_out"].values())
        ok = ok and all(v["uncovered_mm3"] <= tol_mm3 and v["max_outside_mm"] <= tol_mm for v in rc["fixed"].values())
        method = run.get("mesh_method") or "sdf"
        for k, v in rc["keep_out"].items():
            if v["intersect_mm3"] > tol_mm3:
                why.append(f"keep-out {k}: {v['intersect_mm3']:.3g} mm3 of the mesh is inside it (over {tol_mm3:g} mm3)")
        for k, v in rc["fixed"].items():
            if v["uncovered_mm3"] > tol_mm3 or v["max_outside_mm"] > tol_mm:
                why.append(f"region {k}: {v['uncovered_mm3']:.3g} mm3 of its STL is not in the mesh, reaching {v['max_outside_mm']:.3g} mm "
                           f"from the surface" + (" (mesh method blur: this region comes from the voxel labels, which hold a staircase "
                                                  "of up to half a voxel at each face; the sdf method holds the STL)" if method == "blur" else ""))
    elif (run_dir / "labels.bin").is_file():
        h, up = float(run["h"]), post.UPSAMPLE
        labels = np.frombuffer((run_dir / "labels.bin").read_bytes(), np.uint8).reshape(tuple(run["shape"]))
        lc = smooth_sdf.label_checks(mesh.vertices, mesh.faces, labels, run["origin"], h, up, post.PAD)
        out["labels"] = lc
        tol = tol_label_mm3 if tol_label_mm3 is not None else 5.0 * (h / up) ** 3
        ok = ok and lc["fixed_missing_mm3"] <= tol and lc["keep_out_inside_mm3"] <= tol
    if ctx is not None:
        out["interfaces"] = _interface_checks(run_dir, run, ctx, mesh, tol_interface_mm, tol_mm3, files_ok, keep)
        ok = ok and out["interfaces"]["ok"]
    out["ok"] = bool(ok)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tol-mm3", type=float, default=1e-3, help="volume tolerance, mm^3")
    ap.add_argument("--tol-mm", type=float, default=1e-3, help="reach tolerance, mm")
    ap.add_argument("--tol-label-mm3", type=float, default=None, help="label-case volume tolerance, mm^3")
    ap.add_argument("--tol-interface-mm", type=float, default=None,
                    help="size tolerance of each built-in interface, mm (default: chord sag of the facets + 1e-4)")
    a = ap.parse_args(argv)
    res = check_run(a.run_dir, a.tol_mm3, a.tol_mm, a.tol_label_mm3, a.tol_interface_mm)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        m = res["mesh"]
        print(f"{res['file']}: watertight={m['watertight']} winding={m['winding_consistent']} "
              f"bodies={m['bodies']} open_edges={m['open_edges']} nonmanifold_edges={m['nonmanifold_edges']} "
              f"zero_area_faces={m['zero_area_faces']} volume={m['volume_mm3']} mm3")
        for k, v in (res["keep_out"] or {}).items():
            print(f"keep-out {k}: mesh inside = {v['intersect_mm3']:.3g} mm3")
        for k, v in (res["regions"] or {}).items():
            print(f"region {k}: missing = {v['uncovered_mm3']:.3g} mm3, reach = {v['max_outside_mm']:.3g} mm")
        if res["labels"]:
            lb = res["labels"]
            print(f"labels: fixed volume not held = {lb['fixed_missing_mm3']:.3g} mm3 (of {lb['fixed_label_mm3']}), "
                  f"mesh inside keep-out = {lb['keep_out_inside_mm3']:.3g} mm3 (of {lb['keep_out_label_mm3']})")
        ic = res.get("interfaces")
        if ic:
            nm = ic["nominal_mesh"]
            if nm:
                print(f"{ic['nominal_file']}: watertight={nm['watertight']} winding={nm['winding_consistent']} "
                      f"bodies={nm['bodies']} open_edges={nm['open_edges']} nonmanifold_edges={nm['nonmanifold_edges']} "
                      f"zero_area_faces={nm['zero_area_faces']} volume={nm['volume_mm3']} mm3")
            for it in ic["items"]:
                for var in ("as_printed", "nominal"):
                    for fk, f in (it[var] or {}).items():
                        if f["found"]:
                            print(f"interface {it['name']} ({it['kind']}, {var}) {fk}: measured {f['measured_mm']:.6f} mm, "
                                  f"expected {f['expected_mm']:.6f} mm, deviation {f['deviation_mm']:+.6f} mm "
                                  f"({'within' if f['within_tolerance'] else 'OVER'} {f['tolerance_mm']:.6f} mm)"
                                  + (f", covers {100 * f['coverage']:.0f} percent of {f['expected_area_mm2']} mm2"
                                     if f.get("coverage") is not None else ""))
                        else:
                            print(f"interface {it['name']} ({it['kind']}, {var}) {fk}: no surface found near {f['expected_mm']:.6f} mm")
                if it["stock_measured_mm"]:
                    print(f"interface {it['name']}: stock in the files = {it['stock_measured_mm']} mm (YAML {it['stock_mm']} mm)")
            for pr in ic["problems"]:
                print(f"problem: {pr}")
        for pr in res.get("problems", []):
            print(f"problem: {pr}")
        print("OK" if res["ok"] else "FAIL")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
