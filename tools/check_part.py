#!/usr/bin/env python3
"""Check a finished topopt run as a printable part.

Usage: python tools/check_part.py RUN_DIR YAML [--ref REF_METRICS.json] [--bed 220] [--pitch 0.25] [--min-thick 1.2]

The bed size and the build direction come from the `print` block of the YAML when it has them
(CONTRACTS.md section 18). `--bed` replaces the bed size. Without any `print` key the bed is 220 mm
and the check takes the two smallest extents (flat side down, strict "<": the line is as before).
With `print` keys the rule is the one of the print report: "<=", and with a build direction the
extents in the build frame and a turn about the build axis. A wrong `print` block gives one `error:` line.

Checks: watertight, one body, volume, bed fit, screwdriver path for every
hole-type keep_out region (file name contains "hole", axis along x), thin members, load tip material, mass check
(metrics.json vs STL volume x density), comparison with a reference run.
Exit code 0 if all checks pass, 1 if one fails.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, trimesh, yaml
from scipy import ndimage

res = []

def voxelize(mesh, p):
    """Fill by ray scan along x: one ray per (y,z) column. Needs a closed mesh."""
    lo, hi = mesh.bounds
    ys = np.arange(lo[1]+p/2, hi[1], p); zs = np.arange(lo[2]+p/2, hi[2], p)
    nx = int(np.ceil((hi[0]-lo[0])/p)) + 1
    m = np.zeros((nx, len(ys), len(zs)), bool)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    org = np.c_[np.full(Y.size, lo[0]-1.0), Y.ravel(), Z.ravel()]
    loc, ray, _ = mesh.ray.intersects_location(org, np.tile([1.0, 0, 0], (len(org), 1)), multiple_hits=True)
    order = np.lexsort((loc[:, 0], ray)); loc, ray = loc[order], ray[order]
    for r in np.unique(ray):
        xs = loc[ray == r, 0]
        if len(xs) % 2: continue
        iy, iz = divmod(r, len(zs))
        for a, b in zip(xs[0::2], xs[1::2]):
            m[int((a-lo[0])/p):int((b-lo[0])/p)+1, iy, iz] = True
    return np.pad(m, 2), lo - 2*p
def topopt_config():
    """The module `topopt.config` (the code folder is next to tools/)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import topopt.config
    return topopt.config

def print_settings(cfg, ypath):
    """The `print` block of the YAML as `PrintSettings`, or None if the YAML has none."""
    if "print" not in cfg:
        return None
    return topopt_config().parse_print_block(cfg["print"], ypath)

def bed_check(mesh, ext, ps, bed_option):
    """Check 1b, the bed fit. Without `print` keys in the YAML the line is as before this item.

    With `print` keys the rule is the one of the print report (`fits_bed`, at most; with a build
    direction also a turn about the build axis, in 1-degree steps: the turn with the largest margin, if the part does not fit
    without it or if the turn adds at least `BED_TURN_MIN_GAIN_MM` of margin).
    `--bed` replaces the bed size. A `print.printer_model` that states a build height adds the height test to the build-direction rule (without
    a build direction the height is not known, so it is not tested here; `topopt printcheck` and `topopt check` test it).
    """
    bed = (bed_option, bed_option) if bed_option is not None \
        else (ps.bed_mm if ps and ("bed_mm" in ps.given or "bed_mm" in ps.derived) else (220.0, 220.0))   # derived: from a printer_model
    name = f"fits {bed[0]:g}x{bed[1]:g} mm bed"
    if ps is None or not ps.given:
        se, sb = sorted(ext), sorted(bed)       # the old rule: strict, the two smallest extents
        rec(name, se[0] < sb[0] and se[1] < sb[1],
            f"extents={np.round(ext,1).tolist()} (any 2 axes < bed; flat side down)")
    elif "build_direction" in ps.given:
        from topopt.print_report import bed_fit_turn, direction_label, extents_in_frame, fit_phrase
        fe = extents_in_frame(mesh.vertices, ps.build_direction)
        fit = bed_fit_turn(mesh.vertices, ps.build_direction, bed, build_height_mm=ps.build_height_mm)   # a printer_model with a build height: the part must be low enough too
        turn = ""
        if fit["bed_turn_deg"] or not fit["bed_fit"]:       # a turn above 0, or no fit: say it, with the footprint
            turn = f", {fit_phrase(fit)}; footprint {fit['bed_footprint_mm'][0]:g} x {fit['bed_footprint_mm'][1]:g} mm"
        rec(name, fit["bed_fit"],
            f"extents in the build frame (build direction {direction_label(ps.build_direction)}) = "
            f"{np.round(fe,1).tolist()} (two in-plane extents, then height){turn}")
    else:
        from topopt.print_report import fits_bed
        two = sorted(ext)[:2]                   # no build direction given: the two smallest extents
        rec(name, fits_bed(two, bed),
            f"extents={np.round(ext,1).tolist()} (any 2 axes <= bed; flat side down)")


def rec(name, ok, info=""):
    res.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("yaml")
    ap.add_argument("--ref"); ap.add_argument("--bed", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=0.25)
    ap.add_argument("--min-thick", type=float, default=1.2)
    ap.add_argument("--face-x", type=float, default=5.0, help="ray start in front of plate")
    ap.add_argument("--back-x", type=float, default=-5.0, help="ray start behind wall")
    ap.add_argument("--disc-r", type=float, default=4.3)
    ap.add_argument("--shaft-r", type=float, default=2.0)
    a = ap.parse_args()
    run = Path(a.run_dir); ypath = Path(a.yaml); base = ypath.parent
    cfg = yaml.safe_load(ypath.read_text())
    user_error = topopt_config().UserError if "print" in cfg else ()   # no `print` block: nothing to catch
    try:
        ps = print_settings(cfg, ypath)
    except user_error as e:        # a wrong block: one `error:` line and exit 1 (as tools/print_section.py), before any check
        sys.exit(f"error: {e}")
    mesh = trimesh.load(run / "result" / "final.stl", force="mesh")
    metrics = json.loads((run / "result" / "metrics.json").read_text())
    dens = cfg.get("material", {}).get("density", 1.24)

    # 1. basic geometry
    parts = mesh.split(only_watertight=False)
    rec("watertight", mesh.is_watertight, f"winding_consistent={mesh.is_winding_consistent}")
    rec("one body", len(parts) == 1, f"{len(parts)} bodies; sizes(tri)={sorted(len(p.faces) for p in parts)[-5:]}")
    vol = abs(mesh.volume)
    rec("volume > 0", vol > 0, f"{vol:.1f} mm3")
    ext = mesh.extents
    # V2: the bed size and the build direction of the YAML `print` block (when it has them).
    bed_check(mesh, ext, ps, a.bed)

    # 2. screwdriver path
    rng = np.random.default_rng(0)
    def disc(r, x, n=1500):
        rr = r*np.sqrt(rng.random(n)); th = 2*np.pi*rng.random(n)
        return np.c_[np.full(n, x), rr*np.cos(th), rr*np.sin(th)]
    # Only hole-type keep-outs (file name contains "hole") get the path checks.
    # Other keep-outs (a cavity, a hand space) are not screw holes. See CONTRACTS.md section 2.
    for kp in cfg["regions"].get("keep_out", []):
        if "hole" not in Path(kp).stem.lower():
            print(f"[SKIP] path checks {kp}: keep-out is not a hole (name has no 'hole')")
            continue
        k = trimesh.load(base / kp, force="mesh"); c = k.bounds.mean(0)
        r = np.hypot(k.vertices[:,1]-c[1], k.vertices[:,2]-c[2])
        wide = r.max()
        o = disc(a.disc_r, a.face_x); o[:,1] += c[1]; o[:,2] += c[2]
        inside = mesh.contains(o[:1] * 0 + [a.face_x, c[1], c[2]])[0]
        d = np.tile([1.0, 0, 0], (len(o), 1))
        hit = mesh.ray.intersects_any(o, d)
        rec(f"screwdriver path {kp}", (not inside) and not hit.any(),
            f"axis y={c[1]:.2f} z={c[2]:.2f}, wide r={wide:.2f}, disc r={a.disc_r}, start inside part={bool(inside)}, rays blocked={int(hit.sum())}/{len(hit)}")
        # radial scan of the printed hole: smallest radius over x inside the plate
        angs = np.linspace(0, 2*np.pi, 72, endpoint=False); minr = []
        for x in np.arange(0.25, k.bounds[1][0] if False else 3.5, 0.5):
            org = np.tile([x, c[1], c[2]], (len(angs), 1))
            dirs = np.c_[np.zeros(len(angs)), np.cos(angs), np.sin(angs)]
            loc, ray, _ = mesh.ray.intersects_location(org, dirs, multiple_hits=False)
            rad = np.full(len(angs), np.inf)
            if len(ray): rad[ray] = np.linalg.norm(loc - org[ray], axis=1)
            minr.append(rad.min())
        nom = float(r[np.isclose(k.vertices[:, 0], k.bounds[0][0], atol=0.01)].max())
        rec(f"printed hole radius {kp}", min(minr) >= nom - 0.3,
            f"nominal shaft r={nom:.2f}, smallest printed r={min(minr):.2f} mm (min over x 0.25..3.25 and 72 angles)")
        o = disc(a.shaft_r, a.back_x); o[:,1] += c[1]; o[:,2] += c[2]
        hit = mesh.ray.intersects_any(o, d)
        frac = 1 - hit.mean()
        rec(f"shaft path {kp}", frac > 0.9, f"shaft r={a.shaft_r}, rays free={frac*100:.1f}%")

    # 3. thin members: opening with a ball of diameter min_thick
    p = a.pitch
    m, org0 = voxelize(mesh, p)
    edt = ndimage.distance_transform_edt(m) * p
    def removed(d):
        core = edt >= d/2
        if not core.any(): return 1.0
        back = ndimage.distance_transform_edt(~core) * p <= d/2
        return float((m & ~back).sum()) / m.sum()
    sweep = {d: removed(d) for d in (0.6, 0.8, 1.0, a.min_thick, 1.6, 2.0)}
    # thinnest wall: smallest d where opening removes > 0.05% volume
    thin = next((d for d in sorted(sweep) if sweep[d] > 5e-4), None)
    maxr = edt.max()*2
    rec(f"thin members < {a.min_thick} mm", sweep[a.min_thick] < 5e-3,
        f"volume lost by opening: " + ", ".join(f"{d:g}mm={v*100:.3f}%" for d, v in sweep.items()) +
        f"; thickest={maxr:.1f} mm; first diameter with loss>0.05% (noise level): {thin}")

    # 4. load tip material
    for name, tp in cfg["regions"].get("load", {}).items():
        t = trimesh.load(base / tp, force="mesh"); lo, hi = t.bounds
        i0 = np.maximum(((lo-org0)/p).astype(int), 0); i1 = np.maximum(((hi-org0)/p).astype(int), 0)
        sub = m[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        frac = sub.mean() if sub.size else 0.0
        rec(f"material in load box '{name}'", frac > 0.05, f"{frac*100:.1f}% of box volume is solid; box={lo.tolist()}..{hi.tolist()}")

    # 5. mass
    mass = vol/1000*dens
    mm = metrics["mass_final_g"]
    rec("mass vs STL volume", abs(mm-mass)/mass < 0.01,
        f"metrics {mm:.2f} g, STL {mass:.2f} g (density {dens}), diff {100*(mm-mass)/mass:+.2f}%")

    # 6. reference
    if a.ref:
        ref = json.loads(Path(a.ref).read_text()); bad = []
        for key in ("mass_final_g", "volume_final_mm3", "compliance_final", "triangles"):
            if key in ref and key in metrics:
                dv = (metrics[key]-ref[key])/ref[key]*100
                print(f"   {key}: run {metrics[key]:.4g} ref {ref[key]:.4g} ({dv:+.2f}%)")
                if abs(dv) > 1 and key != "triangles": bad.append(key)
        for c1, c2 in zip(metrics.get("stiffness_cases", []), ref.get("stiffness_cases", [])):
            if c1["N_per_mm"] is None or c2["N_per_mm"] is None:   # null: a moment, a remote force or a pressure
                print(f"   stiffness {c1['name']}: run {c1['N_per_mm']} ref {c2['N_per_mm']} (not compared: null)")
                continue
            dv = (c1["N_per_mm"]-c2["N_per_mm"])/c2["N_per_mm"]*100
            print(f"   stiffness {c1['name']}: run {c1['N_per_mm']} ref {c2['N_per_mm']} ({dv:+.2f}%)")
            if abs(dv) > 1: bad.append(c1["name"])
        rec("matches reference within 1%", not bad, f"outside: {bad}")
    print("run_time_s:", metrics.get("run_time_s"))
    sys.exit(0 if all(res) else 1)

if __name__ == "__main__":
    main()
