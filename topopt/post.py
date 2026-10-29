"""Post-process of a finished run (CONTRACTS.md section 9).

`finish(run_dir)` reads the last snapshot, makes a smooth watertight mesh,
and writes result/final.stl, result/preview.bin and result/metrics.json.
With `regions.not_printed` in the YAML it also writes result/part_only.stl (`topopt.part_only`, CONTRACTS.md section 9.1).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from topopt import robust
from topopt.config import ENVELOPE_FILL_DEFAULT, ENVELOPE_FILL_VALUES
from topopt.dashboard_server import pack_mesh
from topopt.resource_errors import ResourceStopError, reraise_if_resource
from topopt.rows_motion import rows_block as motion_rows_block

log = logging.getLogger("topopt.post")

EMPTY, DESIGN, FIXED, LOAD, KEEP_OUT = 0, 1, 2, 3, 4
UPSAMPLE = 2  # Fine grid factor for a smooth surface.
PAD = 2  # Empty voxels around the field (coarse units).
TAUBIN_ITERATIONS = 20
BLUR_SIGMA = 1.6  # Gaussian blur before marching cubes, in coarse voxels.
MAX_SHRINK = 0.03  # The smooth step must not remove more volume than this.
PART_ONLY_FILE = "part_only.stl"   # result/part_only.stl, only for a run with regions.not_printed (CONTRACTS.md section 9.1)
_SNAP = re.compile(r"^it_(\d{4,})\.bin$")


# -- small file helpers --
def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> None:
    _write_atomic(path, json.dumps(obj, indent=1).encode())


def _read_history(run_dir: Path) -> list[dict]:
    """The rows of history.jsonl. A partial last line is skipped and stated in run.log (a fact).

    A line that does not parse and is not the last line is an error (`HistoryError`, a ValueError).
    """
    from topopt.runinfo import append_log_line, read_history
    path = run_dir / "history.jsonl"
    if not path.exists():
        return []
    rows, facts = read_history(path)
    for fact in facts:
        log.warning(fact)
        # The snapshot of the last iteration is written before its history row, so the mesh can be one step ahead.
        note = ""
        if rows:
            note = (f" The metrics are from iteration {rows[-1].get('iteration')}; "
                    "the mesh is from the last snapshot on disk.")
        append_log_line(run_dir, "warning: " + fact + "." + note)
    return rows


def _last_snapshot(run_dir: Path) -> Path:
    best = None
    for p in (run_dir / "snapshots").iterdir():
        m = _SNAP.match(p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    if best is None:
        raise FileNotFoundError(f"no snapshot in {run_dir / 'snapshots'}")
    return best[1]


def case_stiffness(cases, compliances, warn: bool = True, reaches=None) -> list:
    """N/mm of each load case: `|F|^2 / c`. `cases` is `run.json` `load_cases`, `compliances` the compliance of each case.
    `reaches` is `load_reaches_free_dofs` of each case (`run.json` `derived.load_cases`, F4-37), or None for an old run.json.

    One formula for `stiffness_cases` and for `robust.stiffness_cases_eroded` (section 22b).
    `N_per_mm` is null for a case without forces (an old run.json, one warning) or without a compliance.
    It is defined for plain forces only: a case with a moment, a remote force or a pressure gets null and a `reason`
    (CONTRACTS.md section 26.2). A case whose forces are all 0 (it applies no load, U21) also gets null and a `reason`; so does a case
    whose load reaches no free dof (`reaches` is false, F4-37: its nodal forces cancel, or all of them sit on fixed dofs).
    """
    out, warned = [], False
    for i, case in enumerate(cases):
        val, reason = None, None
        forces = case.get("forces")
        if forces is None:
            if warn and not warned:
                log.warning("run.json has no load case forces: N_per_mm is null")
                warned = True
        elif i < len(compliances):
            kinds = sorted({f.get("kind", "force") for f in forces} - {"force"})
            if compliances[i] > 0:
                if kinds:
                    reason = (f"the case has loads of kind {', '.join(kinds)}: |F|^2 / c holds for plain forces "
                              "(c is the work of all loads, F is only their resultant)")
                else:
                    total = np.sum([f["force"] for f in forces], axis=0)
                    val = round(float(total @ total) / float(compliances[i]), 3)
            elif not kinds and not any(v != 0.0 for f in forces for v in f["force"]):
                # U21: every force of the case is 0, so its compliance is 0 (`LoadCase.applies_load` is false for it)
                reason = "the case applies no load: every force is 0, so its compliance is 0 and |F|^2 / c is not defined"
            elif reaches and i < len(reaches) and reaches[i] is False:
                reason = ("the load of the case reaches no free dof (load_reaches_free_dofs is false), so its compliance is 0 and "
                          "|F|^2 / c is not defined")
        out.append({"name": case.get("name", f"case{i}"), "N_per_mm": val, **({"reason": reason} if reason else {})})
    return out


# -- field and mesh --
def build_field(density: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the float density (0..1) with the fixed rules applied.

    Fixed and load voxels are 1. Empty and keep-out voxels are 0.
    """
    rho = density.astype(np.float32)
    rho[(labels == FIXED) | (labels == LOAD)] = 1.0
    rho[(labels == EMPTY) | (labels == KEEP_OUT)] = 0.0
    return rho


def _upsample(rho: np.ndarray, up: int, order: int = 1) -> np.ndarray:
    """Pad the field, then make the grid `up` times finer."""
    from scipy.ndimage import zoom

    padded = np.pad(rho, PAD, mode="constant")
    if up == 1:
        return padded
    out = zoom(padded, up, order=order, grid_mode=True, mode="grid-constant")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _keep_fixed_components(fine: np.ndarray, fixed_fine: np.ndarray, level: float = 0.5) -> np.ndarray:
    """Set to 0 every solid island that does not touch the fixed region.

    If no solid touches the fixed region, keep the largest island.
    """
    from scipy.ndimage import label

    solid = fine > level
    comp, n = label(solid)
    if n <= 1:
        return fine
    touch = np.unique(comp[fixed_fine & solid])
    touch = touch[touch > 0]
    if len(touch) == 0:
        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        touch = np.array([int(np.argmax(sizes))])
    drop = solid & ~np.isin(comp, touch)
    out = fine.copy()
    out[drop] = 0.0
    return out


def _signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    v = verts.astype(np.float64)
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def make_field(rho, labels, up: int = UPSAMPLE, blur: float = BLUR_SIGMA):
    """Return (fine field, fixed mask, target volume) for marching cubes.

    The blur removes the voxel steps. `find_level` then keeps the solid
    volume equal to the target (the volume of the sharp field).
    """
    from scipy.ndimage import gaussian_filter

    fine = _upsample(rho, up)
    fixed_fine = _upsample(((labels == FIXED) | (labels == LOAD)).astype(np.float32), up) > 0.5
    hole_fine = _upsample(((labels == KEEP_OUT)).astype(np.float32), up) > 0.5
    target = float((fine > 0.5).sum()) * (1.0 / up) ** 3  # in coarse voxel units
    if blur > 0:
        fine = gaussian_filter(fine, blur * up, mode="constant")
        fine[fixed_fine] = 1.0
        fine[hole_fine] = 0.0
    return fine, fixed_fine, target


def _mc_volume(fine: np.ndarray, level: float, up: int) -> float:
    """Volume of the marching-cubes surface, in coarse voxel units."""
    from skimage.measure import marching_cubes

    v, f, _n, _vals = marching_cubes(fine, level=level)
    return abs(_signed_volume(v, f)) / up ** 3


def find_level(fine: np.ndarray, target: float, up: int, steps: int = 6) -> float:
    """Find the iso level that gives the target volume (bisection)."""
    lo, hi = 0.25, 0.6  # volume falls when the level rises
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if _mc_volume(fine, mid, up) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_mesh(rho, labels, origin, h, up: int = UPSAMPLE, passes: int = TAUBIN_ITERATIONS,
              blur: float = BLUR_SIGMA):
    """Make a smooth watertight mesh. Return (verts float64, faces int64, info)."""
    import trimesh
    from skimage.measure import marching_cubes

    fine, fixed_fine, target = make_field(rho, labels, up, blur)
    level = find_level(fine, target, up) if blur > 0 else 0.5
    fine = _keep_fixed_components(fine, fixed_fine, level)
    if fine.max() <= level:
        raise ValueError("the final field has no solid voxels")

    verts, faces, _n, _v = marching_cubes(fine, level=level)
    faces = faces[:, ::-1]  # skimage winds inward here. Flip for outward normals.
    # Fine index f has its centre at world origin + (h/up) * (f - PAD*up + 0.5).
    step = h / up
    verts = np.asarray(origin, dtype=np.float64) + step * (verts - PAD * up + 0.5)

    mesh = trimesh.Trimesh(verts, faces.copy(), process=False)
    vol0 = abs(mesh.volume)
    if passes > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=passes)
    vol1 = abs(mesh.volume)
    shrink = (vol0 - vol1) / vol0 if vol0 > 0 else 0.0
    if shrink > MAX_SHRINK:
        log.warning("smoothing removed %.1f%% volume, more than %.0f%%", 100 * shrink, 100 * MAX_SHRINK)
    # Make sure the winding is consistent and the volume is positive.
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    info = {"shrink_pct": 100.0 * shrink, "level": level,
            "watertight": bool(mesh.is_watertight and mesh.is_winding_consistent)}
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64), info


def _to_manifold(verts, faces):
    import manifold3d as m3d

    mesh = m3d.Mesh(vert_properties=np.ascontiguousarray(verts, np.float32),
                    tri_verts=np.ascontiguousarray(faces, np.uint32))
    man = m3d.Manifold(mesh)
    if man.status() != m3d.Error.NoError:
        raise ValueError(f"not a manifold: {man.status()}")
    return man


def keep_out_paths(run: dict) -> list[Path]:
    """Return the keep-out STL paths of a run (next to the YAML file). `run` as `runinfo.resolved_run` gives it."""
    names = run.get("regions", {}).get("keep_out", []) or []
    base = Path(run.get("yaml_path", "")).parent
    return [base / n for n in names]


def subtract_keep_out(verts, faces, stl_paths):
    """Cut the exact keep-out STL shapes out of the mesh (manifold3d).

    Return (verts, faces, info). On any problem, log a warning and return
    the mesh unchanged. Only the largest body stays.
    """
    import trimesh

    info = {"keep_out_cut": False, "extra_bodies": 0}
    if not stl_paths:
        return verts, faces, info
    missing = [os.path.normpath(p) for p in stl_paths if not Path(p).is_file()]   # the shown text has no `..`
    if missing:
        log.warning("keep-out STL missing, no exact cut: %s", ", ".join(missing))
        return verts, faces, info
    try:
        body = None
        try:
            body = _to_manifold(verts, faces)
        except ValueError:
            tm = trimesh.Trimesh(verts, faces, process=True)
            tm.merge_vertices()
            body = _to_manifold(tm.vertices, tm.faces)
        cutter = None
        for p in stl_paths:
            t = trimesh.load(p, force="mesh")
            t.merge_vertices()
            man = _to_manifold(t.vertices, t.faces)
            cutter = man if cutter is None else cutter + man
        out = (body - cutter).to_mesh()
        v = np.asarray(out.vert_properties, np.float64)[:, :3]
        f = np.asarray(out.tri_verts, np.int64)
        tm = trimesh.Trimesh(v, f, process=False)
        parts = tm.split(only_watertight=False)
        if len(parts) > 1:
            parts = sorted(parts, key=lambda q: abs(q.volume), reverse=True)
            info["extra_bodies"] = len(parts) - 1
            log.warning("keep-out cut made %d extra bodies: kept the largest", len(parts) - 1)
            tm = parts[0]
        tm.fix_normals()
        if tm.volume < 0:
            tm.invert()
        info["keep_out_cut"] = True
        return np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64), info
    except Exception as exc:  # keep the unsubtracted mesh
        reraise_if_resource(exc, "the cut of the keep-out regions")
        log.warning("keep-out cut failed, mesh not cut: %s", exc)
        info["keep_out_error"] = f"{type(exc).__name__}: {exc}"
        return verts, faces, info


# Points for each trimesh `contains` call. Measured with trimesh `ray_triangle` (no embree), 1 thread, on a real run
# (2.5 mm voxels, 10 bodies, the largest of 206,550 faces, 0.69 GiB in use before the check). Peak = VmHWM during
# the check. Rise = the peak minus the memory in use at the start. Medians of 3 runs (the worst case: 4 runs).
#   chunk (points)                                  300     64     32     16      8      4      2      1
#   load path check of the run, peak GiB           6.77   2.35   1.45   1.06   0.90   0.80   0.76   0.76
#   load path check of the run, seconds            16.2    4.1    2.3    1.5    1.3    1.6    1.0    1.0
#   worst case, 300 points on the big body,
#     none inside (no early stop), rise GiB        4.56   1.14   0.60   0.35   0.22   0.14   0.09   0.06
#     none inside, seconds                          8.1    7.8    7.6    7.7    7.5    7.7    8.1    6.7
#   worst case, 200 bodies of 80 faces,
#     300 points each, none inside, seconds         1.8    2.0      -    3.1    3.5    4.7    7.1   10.7
# The memory is about 15 MB for each point of a chunk. The time on a big body does not depend on the chunk.
# Each call costs about 0.15 ms, which shows on many small bodies. At 8 the rise is about 0.2 GiB.
CONTAINS_CHUNK = 8


def contains_chunks(mesh, points, chunk: int | None = None):
    """Yield `mesh.contains(points[i:i + chunk])` as a bool array for each chunk, in order.

    Without embree, trimesh casts the rays with `ray_triangle`, and the transient memory grows with the number of
    points (about 15 MB for each point on a body of 206,550 faces). A chunk (default `CONTAINS_CHUNK`) bounds it.
    A caller that needs only `.any()` can stop at the first chunk with a point inside. A shape that is not (n, 3)
    goes to `mesh.contains` as it is, so trimesh gives its own error.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        yield np.asarray(mesh.contains(pts), bool)
        return
    step = int(chunk or CONTAINS_CHUNK)
    for i in range(0, len(pts), step):
        yield np.asarray(mesh.contains(pts[i:i + step]), bool)


def points_inside(verts, faces, points) -> np.ndarray:
    """Return a bool array: which points are inside the closed mesh. The points are tested in chunks."""
    import trimesh

    mesh = trimesh.Trimesh(verts, faces, process=False)
    parts = list(contains_chunks(mesh, points))
    return np.concatenate(parts) if parts else np.zeros(0, bool)


def _verdict(run_dir: Path, hist: list[dict], run: dict) -> dict:
    """The convergence verdict for `metrics.json`: the one the run wrote (`verdict.json`), else one from the history.

    It never stops the post step: a failure gives `{"state": "unknown", "text": <reason>}`.
    """
    try:
        from topopt.verdict import verdict_for_post
        return verdict_for_post(run_dir, hist, int(run.get("iterations", max(len(hist) - 1, 0))))
    except Exception as exc:
        log.warning("verdict not computed: %s", exc)
        return {"format": 1, "state": "unknown", "text": f"the verdict could not be computed: {exc}"}


# -- mesh step: method choice (CONTRACTS.md section 19) --
MESH_METHODS = ("sdf", "blur")
DEFAULT_MESH_METHOD = "sdf"


def _resolve_region(base: Path, stored, name: str) -> Path:
    """The STL path of a region: the stored path (relative to the YAML folder, or absolute), then the YAML folder plus the name."""
    cands = []
    if stored:
        q = Path(stored)
        cands.append(q if q.is_absolute() else base / q)
    cands.append(base / name)
    for c in cands:
        if c.is_file():
            return c
    return cands[0]


def region_paths(run: dict) -> tuple[dict, dict, dict]:
    """Return the STL paths of a run as three dicts: fixed (by file name), load (by region name), keep-out (by file name).
    The solid regions (`solid:<file name>`) are in the second dict, after the load regions: they are load regions that no load case uses.

    `run.json` has `region_files` (paths relative to the YAML folder, or absolute) since section 19. An older run folder, or a
    moved one, falls back to the YAML folder plus the file name. A file that is not there stays in the dict
    (the mesh step then uses the voxel labels and says so). The YAML folder comes from `run["yaml_path"]` as it is: give the
    dict that `runinfo.resolved_run(run, run_dir)` returns, so a relative `yaml_path` starts at the run folder.
    """
    reg = run.get("regions", {}) or {}
    rf = run.get("region_files", {}) or {}
    base = Path(run.get("yaml_path", "")).parent
    fx_s, ko_s, ld_s = rf.get("fixed") or [], rf.get("keep_out") or [], rf.get("load") or {}
    fixed = {n: _resolve_region(base, fx_s[i] if i < len(fx_s) else None, n) for i, n in enumerate(reg.get("fixed") or [])}
    load = {k: _resolve_region(base, ld_s.get(k), n) for k, n in (reg.get("load") or {}).items()}
    keep = {n: _resolve_region(base, ko_s[i] if i < len(ko_s) else None, n) for i, n in enumerate(reg.get("keep_out") or [])}
    # A solid region (`regions.solid`, A3a section 22a) is a frozen body with no clamp and no load: a load region that no load
    # case uses. It goes after the load regions, in the order of the old form, so the mesh step gets the same files in the same
    # order: its exact STL shape is united with the field, and no body that touches it is dropped.
    sd_s = rf.get("solid") or []
    load.update({f"solid:{n}": _resolve_region(base, sd_s[i] if i < len(sd_s) else None, n)
                 for i, n in enumerate(reg.get("solid") or [])})
    return fixed, load, keep


def design_path(run: dict) -> Path | None:
    """The STL path of the design space of a run (as `region_paths` finds a region file), or None when the run does not name one."""
    name = (run.get("regions", {}) or {}).get("design")
    stored = (run.get("region_files", {}) or {}).get("design")
    if not name and not stored:
        return None
    return _resolve_region(Path(run.get("yaml_path", "")).parent, stored, name or Path(stored).name)


def check_regions(fixed: dict, load: dict) -> dict:
    """The fixed and load regions of one dict. Keys are `fixed:<file>` and `load:<name>`, so no name can hide another."""
    return {**{f"fixed:{k}": v for k, v in fixed.items()}, **{f"load:{k}": v for k, v in load.items()}}


def mesh_step(run: dict, rho, labels, origin, h):
    """Make the final mesh with the method of the run, then the exact interfaces. Return (verts, faces, info, record).

    A run with `interfaces` in run.json (CONTRACTS.md section 23) gets the exact union and cut of the interface shapes
    after the smoothing, in both methods. The mesh is then the as-printed mesh, and `info["interfaces"]` holds the
    result of that step (`interfaces.Result`; `finish` takes it out). A run without interfaces is not touched.
    """
    ctx, phase = None, "the set-up of the interface shapes"
    try:
        if run.get("interfaces"):
            from . import interfaces

            ctx = interfaces.from_run(run, origin, h, np.shape(labels))
        phase = "the blur mesh method"   # the SDF path names its own steps: an error that comes here from `_mesh_method` is of the blur path
        verts, faces, info, rec = _mesh_method(run, rho, labels, origin, h, ctx)
        phase = "the exact interface step"
        if ctx is not None:
            verts, faces, info["interfaces"] = interfaces.apply_step(ctx, verts, faces)
    except Exception as exc:   # the blur path has no fallback: a resource error there stops the step with a message that names the cause
        reraise_if_resource(exc, phase)
        raise
    return verts, faces, info, rec


def _write_interfaces(res, result: Path, stl_data: bytes, dens: float, mesh_block=None):
    """`nominal.stl` and the `interfaces` record of metrics.json (section 23). None for a run without interfaces."""
    if res is None:
        (result / "nominal.stl").unlink(missing_ok=True)   # a file of an older post-process of this folder would be wrong
        return None
    from . import interfaces

    return interfaces.write_outputs(res, result, stl_data, dens, (mesh_block or {}).get("calibration"))


def fill_mode(value) -> str:
    """The value of `settings.envelope_fill` in run.json as `auto`, `on` or `off`; a missing or unknown value gives ENVELOPE_FILL_DEFAULT (a boolean of an older record: true is `on`, false is `off`)."""
    if value is True or value is False:
        return "on" if value else "off"
    if value in ENVELOPE_FILL_VALUES:
        return value
    if value is not None:   # a missing key is a run from before F4-55; any other value is a hand edit
        log.warning("run.json has envelope_fill %r, which is not auto, on or off: the post step uses %s", value, ENVELOPE_FILL_DEFAULT)
    return ENVELOPE_FILL_DEFAULT


def _mesh_method(run: dict, rho, labels, origin, h, ctx=None):
    """The method of the run (`sdf` or `blur`) without the interface step. Return (verts, faces, info, record).

    `record` is the `mesh` key of metrics.json. A run folder from before section 19 has no
    `mesh_method` in run.json and gets the default method (the same for `envelope_fill`: the default is ENVELOPE_FILL_DEFAULT). If the SDF path cannot run (the grid is over the guard,
    or any error), the step logs a warning with the fact, uses `blur`, and records `fallback`.
    With `ctx` (interfaces), the `sdf` method joins the exact interface shapes in its SDFs (the booleans come later).
    """
    from . import smooth_sdf

    method = run.get("mesh_method") or DEFAULT_MESH_METHOD
    if method not in MESH_METHODS:
        raise ValueError(f"mesh_method must be one of {', '.join(MESH_METHODS)}, got {method!r}")
    fixed, load, keep = region_paths(run)
    fallback = None
    if method == "sdf":
        try:
            pins = {} if ctx is None else dict(zip(("extra_frozen", "extra_cut"), ctx.pin_meshes()))
            if ctx is not None:   # the union pieces at the as-printed size widen the envelope that the result is cut to (section 19)
                pins["envelope_extra"] = ctx.union_meshes()
            verts, faces, info = smooth_sdf.make_mesh_sdf(
                rho, labels, origin, h, up=UPSAMPLE, pad=PAD, fixed_stls=list(fixed.values()),
                load_stls=list(load.values()), keep_out_stls=list(keep.values()), design_stl=design_path(run),
                params=smooth_sdf.SdfParams(envelope_fill=fill_mode(run.get("envelope_fill"))), **pins)
            rec = {k: info[k] for k in ("method", "params", "sources", "exact", "snap", "volume_thresholded_mm3",
                                        "volume_smoothed_mm3", "calibration", "fallbacks", "islands_removed",
                                        "extra_bodies_removed", "extra_bodies_volume_mm3", "voids_filled", "voids_kept", "plane_clip", "envelope_cut", "envelope_fill", "local_loss", "grid", "t_s")}
            rec["requested"] = "sdf"
            rec["fallback"] = None
            return verts, faces, info, rec
        except smooth_sdf.SdfGridTooLarge as exc:
            fallback = exc.fact
            log.warning("SDF mesh step skipped: %s The old Gaussian path is used for this run.", fallback["message"])
        except Exception as exc:   # any other error: the run keeps its mesh
            reraise_if_resource(exc, "the SDF mesh step")   # a resource error stops the step: the blur path after it crashed the process (section 19)
            fallback = {"error": f"{type(exc).__name__}: {exc}"}
            log.warning("SDF mesh step failed (%s). The old Gaussian path is used for this run. "
                        "Option: set mesh_method: blur to skip the SDF step.", fallback["error"])
    t0 = time.monotonic()
    verts, faces, info = make_mesh(rho, labels, origin, h)
    verts, faces, cut = subtract_keep_out(verts, faces, list(keep.values()))
    rec = {"method": "blur", "requested": method,
           "params": {"upsample": UPSAMPLE, "blur_sigma": BLUR_SIGMA, "taubin_iterations": TAUBIN_ITERATIONS},
           "volume_thresholded_mm3": smooth_sdf.thresholded_volume(rho, labels, UPSAMPLE, PAD) * h ** 3,
           "keep_out_cut": bool(cut.get("keep_out_cut")), **({"keep_out_error": cut["keep_out_error"]} if "keep_out_error" in cut else {}),
           "fallback": fallback,
           "t_s": {"mesh": round(time.monotonic() - t0, 3)}}
    return verts, faces, info, rec


def _mesh_record(rec: dict, run: dict, verts, faces, volume: float, stl_data: bytes, labels, origin, h, ctx=None) -> dict:
    """Finish the `mesh` record of metrics.json: final volume, the STL as loaded again, region checks.

    `rec["watertight"]` is true when the STL file, loaded again as float32, is watertight with a consistent winding.
    """
    from . import smooth_sdf

    t0 = time.monotonic()
    thr = rec.get("volume_thresholded_mm3")
    rec["volume_final_mm3"] = round(volume, 1)
    rec["volume_error_pct"] = round(100.0 * (volume - thr) / thr, 3) if thr else None
    facts = smooth_sdf.stl_facts(stl_data)
    rec["stl_roundtrip"] = facts
    rec["watertight"] = bool(facts["ok"])
    if rec.get("method") == "sdf" and not facts["ok"]:
        log.warning("the STL file is not a closed manifold after the float32 reload: %d bodies, %d open edges, "
                    "%d non-manifold edges, %d zero-area faces; watertight is reported as false",
                    facts["bodies"], facts["open_edges"], facts["nonmanifold_edges"], facts["zero_area_faces"])
    if rec.get("method") == "sdf":
        fixed, load, keep = region_paths(run)
        regions = check_regions(fixed, load)
        files = list(regions.values()) + list(keep.values())
        if ctx is not None:   # section 23: the interface cutters count as keep-out shapes (the as-printed ones)
            keep = {**keep, **ctx.cutters()}
        if regions and all(Path(q).is_file() for q in files):
            try:
                rec["checks"] = smooth_sdf.region_checks(verts, faces, regions, keep)
            except Exception as exc:   # a check must not stop the post step
                log.warning("region checks failed: %s", exc)
                rec["checks"] = {"error": str(exc)}
        else:   # no STL files: the labels gave the shapes, so the numbers are against the label voxel boxes
            try:
                rec["checks"] = {"labels": smooth_sdf.label_checks(verts, faces, labels, origin, h, UPSAMPLE, PAD)}
            except Exception as exc:
                log.warning("label checks failed: %s", exc)
                rec["checks"] = {"error": str(exc)}
    rec.setdefault("t_s", {})["checks"] = round(time.monotonic() - t0, 3)
    return rec


def _load_paths(run_dir, run, problem, snap, final_stl) -> dict:
    """The load path facts of the run for `metrics.load_paths` (I6): for each load region, does the thresholded design and does `final.stl` join it to a support?

    The result of `stress_report.load_path_facts`, made for every run (before I6 only a run with a stress row had it). `problem` is the `Problem` of the run;
    without it `stress_report.run_load_path_facts` rebuilds the problem from the YAML of the run. A fact only: it changes no result, no exit code and
    no verdict. Never raises: an error becomes `{"error": text}`.
    """
    try:
        from topopt.stress_report import run_load_path_facts
        return run_load_path_facts(run_dir, run, problem, snap, final_stl)
    except Exception as exc:
        log.warning("load path facts failed: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


# -- the route that repeats only the post step (CONTRACTS.md section 19, "Resource errors") --
RESUME_OPTIONS = (("mesh_method", "--mesh-method"), ("envelope_fill", "--envelope-fill"), ("solver", "--solver"))   # the run options that the resume compares


def resume_route(run_dir, run: dict) -> str:
    """The text for a `ResourceStopError`: the command that repeats only the post step on `run_dir`.

    The resume refuses a run whose settings differ from the checkpoint. A run option (`--mesh-method`, `--envelope-fill`, `--solver`) is in those
    settings, so the command gets each one that the checkpoint holds and the YAML does not. The MCP tool `run_topopt` has no argument for them:
    the MCP route is named only when the command needs none. If the checkpoint or the YAML cannot be read, the text asks for the options of the first run.
    """
    yaml = run.get("yaml_path")
    flags, unknown = [], False
    try:
        from topopt.checkpoint import read_checkpoint
        from topopt.config import load_config

        recorded = read_checkpoint(run_dir).meta["inputs"]["effective"]["settings"]
        wanted = load_config(yaml).settings
        for key, flag in RESUME_OPTIONS:
            have = recorded.get(key, ENVELOPE_FILL_DEFAULT if key == "envelope_fill" else None)   # the default is left out of the recorded settings
            if have is not None and have != getattr(wanted, key):
                flags.append(f"{flag} {have}")
    except Exception:   # a route is a help: any problem reading the folder or the YAML only changes the text
        unknown = True
    command = f"topopt run {yaml or '<yaml>'} --out {run_dir} --resume" + "".join(f" {f}" for f in flags)
    text = f"To repeat only the post step on this run: `{command}`."
    if unknown:
        text += " Give the options of the first run again (--mesh-method, --envelope-fill, --solver), if it had them."
    elif flags:
        text += " The options are in the command because the resume compares them; run_topopt has no argument for them."
    else:
        text += " (MCP: run_topopt with resume=true.)"
    return text + " It removes result/ first."


# -- main entry --
def finish(run_dir, problem=None, cache=None, write_done: bool = True) -> dict:
    """Post-process a finished run: the mesh, `result/final.stl`, `result/metrics.json`, `preview.bin`.

    `problem` (optional) is the `Problem` of the run: the stress section of `metrics.json` (only for a run with a
    stress row) uses it instead of a rebuild. `cache` (optional dict) gets the voxel stress of the last design under
    "stress" when the section made it, so that the caller can pass it on and the last design is not solved a second time.
    `write_done=False` leaves the status at `postprocess`: the caller has a step to run before the run is
    done (the buckling check, CONTRACTS.md section 33) and writes `done` itself.
    """
    run_dir = Path(run_dir)
    t0 = time.monotonic()
    result = run_dir / "result"
    result.mkdir(exist_ok=True)

    from topopt.runinfo import resolved_run
    run = resolved_run(json.loads((run_dir / "run.json").read_text()), run_dir)   # a relative yaml_path starts at the run folder
    shape = tuple(run["shape"])
    h = float(run["h"])
    origin = np.asarray(run["origin"], np.float64)
    labels = np.frombuffer((run_dir / "labels.bin").read_bytes(), np.uint8).reshape(shape)
    snap = np.frombuffer(_last_snapshot(run_dir).read_bytes(), np.uint8).reshape(shape)
    hist = _read_history(run_dir)
    elapsed0 = float(hist[-1]["elapsed_s"]) if hist else 0.0
    robust_json = run_dir / "robust.json"   # robust mode (section 22b)
    robust_summary = json.loads(robust_json.read_text()) if robust_json.is_file() else None
    if robust_summary:   # the solve of the nominal design comes after the last history line
        elapsed0 = max(elapsed0, float(robust_summary["run_s"]))

    # Status: start of post-process.
    def status(state: str, step: str, message: str = "") -> None:
        iteration = int(hist[-1]["iteration"]) if hist else 0
        _write_json(run_dir / "status.json", {
            "state": state, "iteration": iteration, "step": step,
            "elapsed_s": round(elapsed0 + (time.monotonic() - t0), 2), "message": message,
        })

    status("postprocess", "postprocess")
    try:
        rho = build_field(snap.astype(np.float32) / 255.0, labels)
        verts, faces, info, mesh_rec = mesh_step(run, rho, labels, origin, h)
        iface = info.pop("interfaces", None)   # the result of the exact interface step (section 23), None without interfaces
        import trimesh as _tm
        _m = _tm.Trimesh(verts, faces, process=False)
        info["watertight"] = bool(_m.is_watertight and _m.is_winding_consistent and _m.volume > 0)
        volume = _signed_volume(verts, faces)

        import trimesh
        mesh = trimesh.Trimesh(verts, faces, process=False)
        stl_data = mesh.export(file_type="stl")
        _write_atomic(result / "final.stl", stl_data)
        _write_atomic(result / "preview.bin", pack_mesh(verts, faces))
        mesh_block = _mesh_record(mesh_rec, run, verts, faces, volume, stl_data, labels, origin, h,
                                  None if iface is None else iface.context)
        if mesh_block["method"] == "sdf":   # the file as a slicer reads it decides (float32 reload)
            info["watertight"] = mesh_block["watertight"]

        # V2 print report (CONTRACTS.md section 18). It reads the mesh in memory (the one that was just written to
        # final.stl, before the float32 rounding of the file) and only adds the `print` key. A fault in it gives
        # {"ok": false, "error": ...}, not an exception.
        from topopt.print_report import section_for_run
        print_section = section_for_run(run, mesh)

        # Mass: the density comes from run.json (design-space mass and voxel count).
        n_active = float(run.get("n_active", 0))
        mass_ds = float(run.get("mass_design_space_g", 0.0))
        dens = mass_ds * 1000.0 / (n_active * h ** 3) if n_active > 0 and mass_ds > 0 else 1.24
        mass_final = volume / 1000.0 * dens
        asked = (run.get("derived") or {}).get("target_mass_asked_g")   # I7: set only by a run whose YAML has settings.target_mass_g
        iface_rec = _write_interfaces(iface, result, stl_data, dens, mesh_block)   # nominal.stl and the record; None without interfaces
        part_only_rec = None   # regions.not_printed (CONTRACTS.md section 9.1): result/part_only.stl and its facts; None without the key
        if run.get("part_only"):
            from . import part_only
            part_only_rec = part_only.make(run, verts, faces, rho, labels, origin, h, dens, mesh_block["stl_roundtrip"],
                                           mesh_block["method"], result)
        else:
            (result / PART_ONLY_FILE).unlink(missing_ok=True)   # a file of an older post-process of this folder would be wrong

        # Stiffness for each case: |F|^2 / c (plain forces only, `case_stiffness`). In robust mode the final values are the nominal design (the part
        # that the user prints): the solve after the last pass gives them (robust.json, section 22b).
        cases = run.get("load_cases", [])
        last_cases = hist[-1].get("compliance_cases", []) if hist else []
        compliance_final = float(hist[-1]["compliance"]) if hist else None
        final_cases = last_cases
        if robust_summary:
            compliance_final = float(robust_summary["compliance_nominal"])
            final_cases = robust_summary["compliance_cases_nominal"]
        reaches = [c.get("load_reaches_free_dofs") for c in (run.get("derived") or {}).get("load_cases", [])]
        stiff = case_stiffness(cases, final_cases, reaches=reaches)
        stiff_eroded = case_stiffness(cases, last_cases, warn=False, reaches=reaches) if robust_summary else None

        metrics = {
            "mass_design_space_g": round(mass_ds, 2),
            "mass_final_g": round(mass_final, 2),
            # I7: the mass that the user asked for (`settings.target_mass_g`), only in a run that has it. A fact next to `mass_final_g`: the engine
            # sets no test on it (CONTRACTS.md section 3). The text of the log line is `volume_basis.mass_asked_line`.
            **({"mass_asked_g": float(asked)} if asked is not None else {}),
            "mass_saved_pct": round(100.0 * (1.0 - mass_final / mass_ds), 1) if mass_ds > 0 else 0.0,
            "volume_final_mm3": round(volume, 1),
            "compliance_initial": float(hist[0]["compliance"]) if hist else None,
            "compliance_final": compliance_final,
            "stiffness_cases": stiff,
            "run_time_s": 0.0,
            "iterations": int(run.get("iterations", len(hist) - 1)),
            "solver": run.get("solver"),   # the solver that really ran (from run.json)
            # filter radius, solid rule, min_member_mm request and derived radius (section 22a). null for an old run.
            "filter": (run.get("derived") or {}).get("filter"),
            # constraint values of the last pass (A1). null for an old run. A violation is a fact.
            "constraints": hist[-1].get("constraints") if hist else None,
            "rows": motion_rows_block(hist, (robust_summary or {}).get("rows_nominal")),   # A5: displacement and alignment rows, start and end values
            "watertight": info["watertight"],
            "triangles": int(len(faces)),
            "mesh": mesh_block,
            "robust": None,   # robust mode facts (section 22b), set below
        }
        # I6: two facts in every run, made from the last snapshot. `load_paths`: does `final.stl` (and the thresholded design) join each load region to a
        # support. `threshold_stability`: the volume and the body count of the thresholded design at the levels 0.4, 0.5 and 0.6. Facts only (CONTRACTS.md section 5).
        from topopt import stage_facts
        metrics["load_paths"] = load_paths = _load_paths(run_dir, run, problem, snap, result / "final.stl")
        metrics["threshold_stability"] = stage_facts.threshold_stability(labels, snap, h)
        if run.get("stress"):   # a run with a stress row (A2): relaxed and physical stress numbers
            from topopt.stress_report import stress_section
            metrics["stress"] = stress_section(run_dir, run, hist, problem, cache=cache, load_paths=load_paths)
        metrics["print"] = print_section
        if iface_rec is not None:
            metrics["interfaces"] = iface_rec
        if part_only_rec is not None:
            metrics["part_only"] = part_only_rec
        # springs (CONTRACTS.md section 35): the reactions of the last pass; robust mode: those of the nominal design (robust.json)
        react = (hist[-1].get("reactions") if hist else None) or (robust_summary or {}).get("reactions")
        if react:
            metrics["reactions"] = react
        if run.get("joints"):    # joints (section 38): the beam, section and bond facts of the run, and the AMG iteration counts
            metrics["joints"] = dict(run["joints"])
            its = [r["amg_iterations"] for r in hist if r.get("amg_iterations")]
            if its:
                metrics["joints"]["amg_iterations"] = {"passes": len(its), "last": its[-1], "max": max(max(i) for i in its)}
        metrics["run_time_s"] = round(elapsed0 + (time.monotonic() - t0), 1)
        metrics["robust"] = robust.metrics_block(run, robust_summary, metrics["run_time_s"], stiff_eroded)
        verdict = _verdict(run_dir, hist, run)
        # F1-3a: a mass run whose last pass is grey (the verdict facts have `grey`) gets the body count of `final.stl` here. `verdict.json` is written before
        # this step and stays as it is, so the number cannot be in the verdict text. The key exists only in such a run.
        grey_design = stage_facts.grey_design_record(verdict.get("facts"), metrics.get("threshold_stability"), mesh_block)
        if grey_design is not None:
            metrics["grey_design"] = grey_design
        metrics["verdict"] = verdict   # last key: the convergence verdict (section 24.2)
        if iface_rec is not None:   # tests of `full_surface` and frozen regions that keep too little (section 23): keys only when there is a fact
            from . import interfaces
            extra, sentence = interfaces.verdict_extra(iface_rec)
            if extra:
                metrics["verdict"] = {**metrics["verdict"], **extra, "text": metrics["verdict"]["text"] + sentence}
        _write_json(result / "metrics.json", metrics)
        if part_only_rec is not None:   # the same facts go into run.json, next to the entries that the run wrote at its start
            raw = json.loads((run_dir / "run.json").read_text())
            raw["part_only"] = {**part_only_rec, "entries": (raw.get("part_only") or {}).get("entries", [])}
            _write_json(run_dir / "run.json", raw)
    except Exception as exc:
        if isinstance(exc, ResourceStopError):   # section 19: name the route that repeats only the post step on this run folder
            exc.route = resume_route(run_dir, run)
        status("error", "postprocess", f"post-process failed: {exc}")
        raise
    from topopt.printcheck import write_for_run
    write_for_run(run_dir)   # result/print_check.json (CONTRACTS.md section 32): reports only, never raises, and `done` stays whatever it finds
    if write_done:
        status("done", "done")
    return metrics
