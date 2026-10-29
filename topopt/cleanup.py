"""Automatic geometric cleanup: a LIST of candidates as a marks file (CONTRACTS.md section 29.3).

This module finds spurs, thin bumps, uneven struts, flat webs and loose bodies on the finished part (`result/final.stl`) and writes
them as a marks file (`marks.json`, format of section 29) with the facts of each candidate. It edits nothing and judges nothing.
The `marks:` source of the ratchet (section 36.7) applies each mark with `apply_marks` and judges it with the rules of the YAML.
The agent reads the facts first and decides what to spend judge runs on.

The method is the one of the pipeline stage `s7_cleanup` (`s1h_half.py` to `s6_final.py`), with the differences that section 29.3
names: the solid comes from `topopt.voxelize.inside_mask` on the whole part (no VTK, no half mesh), the frozen distance comes from
the voxel labels of the run, the lattice is anchored to a multiple of the pitch (and to a mirror plane), and the stress comes from
`verify/fields.npz`. Every threshold and constant of the method is in one of two tables, each with its pipeline source line:
`THRESHOLDS` (limits that the call can set) and `METHOD` (constants of the method; not settable). A few inline literals are
guards, rounding and fill values only (a division guard, a digit count, the pad of the distance transform); none is a threshold.

Python: numpy, scipy, scikit-image, trimesh (all core). Run it as a script:
`python -m topopt.cleanup list RUN_DIR [--out DIR] [--set NAME=VALUE ...]` (JSON on stdout).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import apply_marks as am
from . import feedback as fb
from .config import UserError

CLEANUP_FORMAT = 1
PIPELINE_DENSITY_G_MM3 = 0.0078        # s5_candidates.py:11 and s2h_analyse.py:11 (s1h_half.py:13 and lib2.py:10 use 0.0078 x 0.99)
RATCHET_FILES = ("ratchet.jsonl", "ratchet.json")    # topopt.ratchet LOG_FILE and STATE_FILE (a test compares them)
DETECTORS = ("body", "spur", "bump", "member")
ORDERS = ("pipeline", "stress")


def _vol(grams: float) -> float:
    """A pipeline mass threshold (g) as a volume (mm3), at the pipeline density. The thresholds are geometric sizes."""
    return grams / PIPELINE_DENSITY_G_MM3


S1H, S2H, S3, S4, S5, S6 = ("s1h_half.py", "s2h_analyse.py", "s3_sections.py", "s4_bumps.py", "s5_candidates.py", "s6_final.py")

# The limits of the user: the call can set each one. name: (value, unit, pipeline source). A mass threshold of the pipeline is
# written as `grams / 0.0078`, so the pipeline number stays visible.
THRESHOLDS: dict[str, tuple[float, str, str]] = {
    "pitch_mm": (0.5, "mm", f"{S1H}:13, {S2H}:11, {S3}:9, {S4}:7, {S5}:11: the voxel size H of the pipeline"),
    "opening_radius_mm": (1.5, "mm", f"{S5}:15 and :45: the ball of the opening residue ({S4}:10 has 1.0 as its own default)"),
    "spur_min_length_mm": (5.0, "mm", f"{S5}:38: remove_skeleton min_len_mm"),
    "spur_min_volume_mm3": (_vol(0.4), "mm3", f"{S5}:38: min_mass_g 0.4 at 0.0078 g/mm3"),
    "bump_min_volume_mm3": (_vol(0.06), "mm3", f"{S5}:50: a residue below 0.06 g is not a bump"),
    "bump_min_final_volume_mm3": (_vol(0.2), "mm3", f"{S6}:18: a bump below 0.2 g is dropped (the old-mark exemption is not ported)"),
    "bump_extent_min_mm": (5.0, "mm", f"{S5}:53: the longest side of the box of a bump, at least"),
    "bump_extent_max_mm": (30.0, "mm", f"{S5}:53: the longest side of the box of a bump, at most"),
    "member_min_length_mm": (9.0, "mm", f"{S5}:41: smooth min_len_mm"),
    "member_min_volume_mm3": (_vol(0.8), "mm3", f"{S5}:41: min_mass_g 0.8 at 0.0078 g/mm3"),
    "member_min_radius_mm": (1.2, "mm", f"{S5}:41: min_rmean_mm"),
    "member_min_cv": (0.15, "ratio", f"{S5}:41: min_cv"),
    "member_min_ratio": (1.7, "ratio", f"{S5}:41: min_ratio (largest over smallest section radius)"),
    "mild_min_length_mm": (20.0, "mm", f"{S5}:41: the mild tier, length"),
    "mild_min_volume_mm3": (_vol(3.0), "mm3", f"{S5}:41: the mild tier, 3 g at 0.0078 g/mm3"),
    "mild_min_cv": (0.10, "ratio", f"{S5}:41: the mild tier, CV"),
    "mild_min_ratio": (1.4, "ratio", f"{S5}:41: the mild tier, max/min"),
    "web_flatness": (am.DEFAULTS["web_flatness"][0], "ratio", f"{S5}:17 and :42 (read from apply_marks.DEFAULTS): FAIR at this flatness or more"),
    "frozen_near_mm": (3.0, "mm", f"{S5}:30 (fzinfo) and :36: a voxel closer than this to a frozen region counts as near"),
    "frozen_max_share": (0.05, "share", f"{S5}:36: at most this share of the voxels of a candidate is near a frozen region"),
    "near_frozen_mm": (6.0, "mm", f"{S5}:40 and {S6}:27: a candidate closer than this to a frozen region has confidence low"),
    "max_candidates": (10, "count", "none in the pipeline; a choice of this tool: about 5 to 6 minutes of judge time on the hook"),
}
# Constants of the method. Each has its pipeline source line. They are not settable; CONTRACTS 29.3 lists them.
METHOD: dict[str, tuple[float, str, str]] = {
    "grid_pad_mm": (4.0, "mm", f"{S1H}:15: the grid reaches 4 mm beyond the box of the part"),
    "pair_radius_pitches": (1.8, "x pitch", f"{S2H}:27 and {S3}:17: skeleton points closer than this are neighbours"),
    "pair_tolerance_mm": (1e-6, "mm", f"{S2H}:27 and {S3}:17"),
    "junction_trim_min_mm": (1.5, "mm", f"{S2H}:23: the junction zone is max(radius, this) + 0.5 mm"),
    "junction_trim_add_mm": (0.5, "mm", f"{S2H}:23"),
    "member_min_skeleton_points": (2, "count", f"{S2H}:40"),
    "member_min_voxels": (4, "count", f"{S2H}:41-42"),
    "path_min_points": (4, "count", f"{S3}:16 and :23"),
    "station_window_mm": (3.0, "mm", f"{S3}:31: the section area is smoothed over this length"),
    "station_floor_mm": (0.3, "mm", f"{S3}:29 and :31"),
    "end_zone_min_mm": (1.5, "mm", f"{S3}:36: the end zones of a member are left out of the section statistics"),
    "end_zone_radius_factor": (1.5, "x radius", f"{S3}:36"),
    "inner_stations_min": (4, "count", f"{S3}:37"),
    "flatness_stations": (12, "count", f"{S3}:42: about this many stations carry a flatness value"),
    "flatness_min_voxels": (12, "count", f"{S3}:44: a station needs this many voxels"),
    "section_ratio_floor_mm": (0.05, "mm", f"{S3}:47: the smallest radius in max/min"),
    "bump_spur_overlap_mm": (1.0, "mm", f"{S5}:57: a bump voxel this close to a spur candidate overlaps it"),
    "bump_spur_overlap_share": (0.3, "share", f"{S5}:57: a bump with more than this share overlapping a spur is dropped"),
    "stress_radius_mm": (1.0, "mm", f"{S5}:25-26: the elements within this distance of the candidate voxels"),
    "stress_percentile": (99.9, "percent", f"{S5}:23-28: the stress fact is this percentile of the element stress (the keys p999_* keep the pipeline name)"),
    "fair_zone_mm": (5.0, "mm", "it_geom_fair.py:4: Rinf, the radius of the fairing zone of a FAIR mark"),
}
PATH_STEP_MM = am.DEFAULTS["path_step_mm"][0]           # lib2.py:176, read from apply_marks
REMOVAL_REACH_MM = am.DEFAULTS["remove_dom_mm"][0]      # the removal domain of the cut operator
MAX_VOXELS_DEFAULT = am.OPTION_DEFAULTS["max_total_voxels"]
_OFFSETS = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]    # the 26 neighbours, a fixed order

OUTSIDE_GRID_OPTIONS = [
    "Option: an STL of the part without this body (another tool), judged with the `stl:` source of the ratchet (CONTRACTS 36.7); "
    "it is an exact removal that needs no mark.",
    "Option: run the optimizer again with a design box that holds the body, if the body is part of the design.",
]
NOTE_STRUTS = ("The spur, SMOOTH and FAIR detectors assume a lattice of struts. On a shell or plate part (a bowl) the skeleton gives "
               "sheet fragments as members, and the candidates may not be struts. The facts in this block show how much of the part "
               "the members cover; the reference values of two demo parts are in CONTRACTS 29.3.")


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _rl(a, nd=4):
    return [round(float(v), nd) for v in a]


# ---- options -----------------------------------------------------------------------------------------------------
def _norm_options(options) -> tuple[dict, set, list[str]]:
    """The effective options, the names that the call set, and facts about what was not understood."""
    opt = {k: v[0] for k, v in THRESHOLDS.items()}
    opt.update(order="pipeline", kinds=None, max_voxels=MAX_VOXELS_DEFAULT, max_gb=None)
    given: set = set()
    problems: list[str] = []
    if options is not None and not isinstance(options, dict):
        return opt, given, [f"options must be an object; {type(options).__name__} was given and is ignored"]
    known = ", ".join(sorted(list(THRESHOLDS) + ["order", "kinds", "max_voxels", "max_gb"]))
    for k, v in (options or {}).items():
        if k == "max_candidates":
            if v is None:
                opt[k], given = None, given | {k}
            elif fb._num(v) is not None and fb._num(v) >= 1 and float(v) == int(v):
                opt[k], given = int(v), given | {k}
            else:
                problems.append(f"option max_candidates={v!r} is not a whole number from 1 or null (all); {opt[k]} is used")
        elif k in THRESHOLDS:
            x = fb._num(v)
            lo_ok = 0.0 if k == "frozen_max_share" else 1e-9
            if x is None or x < lo_ok:
                problems.append(f"option {k}={v!r} is not a valid number; the default {THRESHOLDS[k][0]:g} is used")
            else:
                opt[k], given = x, given | {k}
        elif k == "order":
            if v in ORDERS:
                opt[k], given = v, given | {k}
            else:
                problems.append(f"option order={v!r} is not one of {', '.join(ORDERS)}; pipeline is used")
        elif k == "kinds":
            items = [s.strip() for s in v.split(",")] if isinstance(v, str) else list(v) if isinstance(v, (list, tuple)) else None
            if not items or any(x not in DETECTORS for x in items):
                problems.append(f"option kinds={v!r} is not a list of {', '.join(DETECTORS)}; the default is used")
            else:
                opt[k], given = [x for x in DETECTORS if x in items], given | {k}
        elif k in ("max_voxels", "max_gb"):
            x = fb._num(v)
            if x is None or x < (1 if k == "max_voxels" else 1e-9):
                problems.append(f"option {k}={v!r} is not a valid number; the default is used")
            else:
                opt[k], given = x, given | {k}
        elif k in METHOD:
            problems.append(f"option {k!r} is a constant of the method and is not settable (CONTRACTS 29.3 lists it); ignored")
        else:
            problems.append(f"option {k!r} is not known (known: {known}); ignored")
    return opt, given, problems


def _threshold_table(opt: dict, given: set) -> dict:
    return {k: {"value": _r(opt[k], 6) if isinstance(opt[k], float) else opt[k], "unit": u, "anchor": a, "overridden": k in given}
            for k, (_v, u, a) in THRESHOLDS.items()}


# ---- the lattice and the guard -----------------------------------------------------------------------------------
def _lattice(bounds, pitch: float, planes: list[dict]):
    """(origin, shape) of the detection lattice. The origin is a multiple of the pitch on each axis; with a mirror plane at
    `at_mm` on axis k the origin on k is `at_mm - m x pitch`, so the plane is a lattice plane (a voxel face)."""
    from .voxelize import grid_from_bounds
    pad = METHOD["grid_pad_mm"][0]
    lo_pad = np.asarray(bounds[0], float) - pad
    lo = np.floor(lo_pad / pitch) * pitch
    for pl in planes:
        k = pl["axis_index"]
        m = math.ceil((pl["at_mm"] - lo_pad[k]) / pitch - 1e-9)
        lo[k] = pl["at_mm"] - m * pitch
    return grid_from_bounds(lo, np.asarray(bounds[1], float) + pad, pitch)


def _detector_estimate(shape, pitch: float, opt: dict) -> dict:
    """Voxels and memory of the detector, against the guards. A model (BYTES_PER_NODE and FIXED_MB of apply_marks) with the
    measured figures next to it."""
    from .verify import memory as vmem
    from .voxelize import suggest_voxel_size
    voxels = int(np.prod([int(s) for s in shape], dtype=np.int64))
    need = voxels * am.BYTES_PER_NODE + am.FIXED_MB * 1e6
    lim = vmem.limits(opt["max_gb"])
    est = {"kind": "model", "voxels": voxels, "shape": [int(s) for s in shape], "pitch_mm": pitch,
           "memory_estimate_mb": round(need / 1e6),
           "memory_model": f"{am.BYTES_PER_NODE} bytes per voxel plus {am.FIXED_MB} MB for the process (the model of apply_marks)",
           "measured": "peak memory and time of this detector: hook 2.24 M voxels (0.5 mm): 0.35 GB and 6 s; ladle 9.29 M voxels (0.5 mm): "
                       "0.84 GB and 9 s; hook 10.19 M voxels (0.3 mm): 0.85 GB and 12 s (Linux x86_64, other jobs running). "
                       "The model is 2 to 3 times above these figures.",
           "max_voxels": int(opt["max_voxels"]), "available_ram_mb": None if lim["available_bytes"] is None else round(lim["available_bytes"] / 1e6),
           "limit_mb": None if lim["limit_bytes"] is None else round(lim["limit_bytes"] / 1e6), "limit_source": lim["limit_source"]}
    if voxels > opt["max_voxels"]:
        est["refused"] = "voxel_guard"
        est["pitch_that_fits_mm"] = suggest_voxel_size(pitch, voxels, opt["max_voxels"])
    elif lim["limit_bytes"] is not None and need > lim["limit_bytes"]:
        est["refused"] = "memory_guard"
        allowed = max((lim["limit_bytes"] - am.FIXED_MB * 1e6) / am.BYTES_PER_NODE, 1.0)
        est["pitch_that_fits_mm"] = suggest_voxel_size(pitch, voxels, allowed)
    return est


def _guard_refusal(est: dict) -> dict:
    if est["refused"] == "voxel_guard":
        msg = f"the detection grid has {est['voxels']:,} voxels at pitch {est['pitch_mm']:g} mm, above the guard of {est['max_voxels']:,}"
        opts = [f"pitch_mm {est['pitch_that_fits_mm']} or more (a coarser grid)", f"max_voxels above {est['voxels']:,}"]
    else:
        msg = (f"the memory estimate {est['memory_estimate_mb']:,} MB for {est['voxels']:,} voxels is above the limit "
               f"{est['limit_mb']:,} MB ({est['limit_source']})")
        opts = [f"pitch_mm {est['pitch_that_fits_mm']} or more (a coarser grid)", "max_gb higher, if the machine has the memory"]
    return {"ok": False, "error": msg + ". Nothing was computed.",
            "refused": {"reason": est["refused"], "path": None, "options": opts}, "facts": {"detector_estimate": est}}


# ---- the output folder -------------------------------------------------------------------------------------------
def _ratchet_root_of(path: Path):
    """The ratchet root that holds `path` (the folder itself or a parent), or None."""
    for p in (path, *path.parents):
        if any((p / n).is_file() for n in RATCHET_FILES):
            return p
    return None


def _earlier_cleanup_folder(out: Path) -> bool:
    try:
        doc = json.loads((out / "cleanup.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and doc.get("format") == CLEANUP_FORMAT


def _plan_out(out_dir, d: Path, input_dirs: set, overwrite: bool) -> Path:
    """The folder to write, or a refusal. Nothing is written here. Every refusal names the fact and the options."""
    in_root = any((d.parent / n).is_file() for n in RATCHET_FILES)
    default = (d.parent.parent / f"{d.parent.name}-cleanup-{d.name}") if in_root else d.parent / f"{d.name}-cleanup"
    new_name = f"Name a new folder in out_dir, or leave out_dir out: the default is {fb._unique_dir(default)}."
    if out_dir is None:
        return fb._unique_dir(default)
    given = Path(os.path.abspath(Path(out_dir).expanduser()))
    out = given.resolve()
    if os.path.islink(given):
        raise fb._Refused(f"out_dir {given} is a symbolic link. The tool does not write through a symbolic link.",
                          "symbolic_link", given, [new_name])
    if out.exists() and not out.is_dir():
        raise fb._Refused(f"out_dir {out} is a file, not a folder.", "not_a_folder", out, [new_name])
    for p in (out, *out.parents):
        if (p / "run.json").is_file() or (p / "status.json").is_file():
            where = "is a run folder" if p == out else f"is inside the run folder {p}"
            raise fb._Refused(f"out_dir {out} {where} (it has run.json or status.json). The files of a run are not "
                              "written. " + new_name, "run_folder", p, [new_name])
    rr = _ratchet_root_of(out)
    if rr is not None:
        where = "is a ratchet root" if rr == out else f"is inside the ratchet root {rr}"
        raise fb._Refused(f"out_dir {out} {where} (it has ratchet.jsonl or ratchet.json). The next ratchet run on that root would "
                          "stop, because the root would hold a file that is not its own. " + new_name, "ratchet_root", rr, [new_name])
    if out in input_dirs:
        raise fb._Refused(f"out_dir {out} holds an input file of the run (the YAML or a region STL). The tool does not "
                          "write into the folder of an input. " + new_name, "input_folder", out, [new_name])
    holds = out.is_dir() and any(out.iterdir())
    if holds and not overwrite:
        raise fb._Refused(f"out_dir {out} holds files. Options: name a new folder; or set overwrite=true if it is an "
                          "earlier cleanup folder (it has a cleanup.json).", "not_empty", out,
                          [new_name, "overwrite=true, only for an earlier cleanup folder"])
    if holds and not _earlier_cleanup_folder(out):
        raise fb._Refused(f"out_dir {out} holds files and is not an earlier cleanup folder (no cleanup.json of format "
                          f"{CLEANUP_FORMAT}). overwrite=true is accepted for an earlier cleanup folder only. " + new_name,
                          "not_a_cleanup_folder", out, [new_name])
    return out


# ---- the walk over a voxel set -----------------------------------------------------------------------------------
def _dfs_walk(nbrs: list, start: int, visited: np.ndarray, walk: list) -> None:
    """Depth first, with the way back written out: the first visit of a node and each return to it are in `walk`, so two
    consecutive entries are neighbours in the graph. Iterative (a set can have a long chain)."""
    visited[start] = True
    walk.append(start)
    stack = [[start, 0]]
    while stack:
        top = stack[-1]
        v, o = top
        lst = nbrs[v]
        while o < len(lst) and visited[lst[o]]:
            o += 1
        if o < len(lst):
            top[1] = o + 1
            c = lst[o]
            visited[c] = True
            walk.append(c)
            stack.append([c, 0])
        else:
            stack.pop()
            if stack:
                walk.append(stack[-1][0])


def _walk_voxels(idx: np.ndarray) -> list[np.ndarray]:
    """The 26-neighbour walks over a set of voxel indices (n, 3): one walk for each 26-connected component. Returns a list of
    arrays of rows of `idx`, each in walk order, so two consecutive rows of one walk are 26-neighbours. The components come in the
    order of their first voxel in index order. No segment joins two components: `apply_marks` cuts along every segment between two
    points, and a segment between two components can cross material (CONTRACTS 29.3 part 5)."""
    idx = np.asarray(idx, np.int64)
    n = len(idx)
    order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))
    idx = idx[order]
    lo = idx.min(0) - 1
    sh = idx.max(0) - lo + 2
    lin = np.ravel_multi_index((idx - lo).T, tuple(int(s) for s in sh))
    srt = np.argsort(lin, kind="stable")
    lin_s = lin[srt]
    nb = np.full((n, 26), -1, np.int64)
    for o, off in enumerate(_OFFSETS):
        t = lin + (off[0] * int(sh[1]) * int(sh[2]) + off[1] * int(sh[2]) + off[2])
        pos = np.minimum(np.searchsorted(lin_s, t), n - 1)
        hit = lin_s[pos] == t
        nb[hit, o] = srt[pos[hit]]
    nbrs = [[c for c in row if c >= 0] for row in nb.tolist()]
    visited = np.zeros(n, bool)
    walks: list[np.ndarray] = []
    while not visited.all():
        walk: list[int] = []
        _dfs_walk(nbrs, int(np.flatnonzero(~visited)[0]), visited, walk)
        walks.append(order[np.asarray(walk, np.int64)])
    return walks


def _walk_vertices(mesh, faces: np.ndarray) -> list[np.ndarray]:
    """The walks over the vertices of a body, along its mesh edges (for a body that holds no voxel centre): one walk for each
    group of vertices that the edges connect. Returns a list of arrays of points."""
    sub = mesh.faces[faces]
    verts = np.unique(sub)
    e = np.vstack([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    loc = {int(v): i for i, v in enumerate(verts)}
    nbrs: list[list[int]] = [[] for _ in verts]
    for a, b in e.tolist():
        nbrs[loc[a]].append(loc[b])
        nbrs[loc[b]].append(loc[a])
    nbrs = [sorted(x) for x in nbrs]
    visited = np.zeros(len(verts), bool)
    walks: list[np.ndarray] = []
    while not visited.all():
        walk: list[int] = []
        _dfs_walk(nbrs, int(np.flatnonzero(~visited)[0]), visited, walk)
        walks.append(mesh.vertices[verts[np.asarray(walk, np.int64)]])
    return walks


def _voxel_centres(idx: np.ndarray, origin, pitch: float) -> np.ndarray:
    return np.asarray(origin, float) + (np.asarray(idx, float) + 0.5) * pitch


def _mark_radius(pitch: float) -> float:
    """Half the voxel diagonal, rounded up to 1e-4: a ball around each voxel centre covers its voxel."""
    return math.ceil(pitch * math.sqrt(3.0) / 2.0 * 1e4 - 1e-9) / 1e4


# ---- the detectors -----------------------------------------------------------------------------------------------
def _stress_source(d: Path, final_sha: str, problems: list, notes: list):
    """The per-element stress of `verify/` for THIS final.stl: (centres, {case: values}, kd-tree) or None, with a note."""
    vdir = d / "verify"
    vj = fb._load_json(vdir / "verify.json", [], "verify.json", expected=False) if (vdir / "verify.json").is_file() else None
    name = ((vj or {}).get("files") or {}).get("fields") or "fields.npz"
    f = vdir / name
    if not f.is_file():
        notes.append("verify/fields.npz is missing, so the candidates have no stress facts. Option: run `topopt verify` on the run "
                     "folder first, or list on acc_00 of a ratchet root (a judged folder has it).")
        return None
    if vj is not None:
        ref = ((vj.get("run") or {}).get("final_stl_sha256"))
        if ref and ref != final_sha:
            notes.append("verify/fields.npz was made for another final.stl (the hash in verify.json differs), so the stress facts are "
                         "not given. Option: run `topopt verify` again.")
            return None
    try:
        from scipy.spatial import cKDTree
        z = np.load(f)
        cases = [k[len("vm_elem_max_"):] for k in z.files if k.startswith("vm_elem_max_")]
        if not cases:
            raise ValueError("no vm_elem_max_<case> array")
        vm = {c: np.asarray(z["vm_elem_max_" + c], float) for c in cases}
        cent = np.asarray(z["p"], float)[:, np.asarray(z["t"])].mean(axis=1).T
    except Exception as exc:
        problems.append(f"verify/{name} cannot be read as a stress field ({type(exc).__name__}: {exc}); the stress facts are not given")
        return None
    return {"file": f"verify/{name}", "cases": cases, "vm": vm, "tree": cKDTree(cent),
            "part_p999": {c: float(np.percentile(v, METHOD["stress_percentile"][0])) for c, v in vm.items()}}


def _stress_facts(src, pts: np.ndarray, radius: float):
    if src is None or len(pts) == 0:
        return None
    el = set(src["tree"].query(pts)[1].tolist())
    for x in src["tree"].query_ball_point(pts, radius):
        el.update(x)
    el = np.array(sorted(el))
    p999 = {c: float(np.percentile(v[el], METHOD["stress_percentile"][0])) for c, v in src["vm"].items()}
    mx = {c: float(v[el].max()) for c, v in src["vm"].items()}
    cp = max(p999, key=lambda c: (p999[c], c))
    cm = max(mx, key=lambda c: (mx[c], c))
    part = src["part_p999"][cp]
    return {"p999_mpa": _r(p999[cp], 3), "case": cp, "max_mpa": _r(mx[cm], 3), "max_case": cm, "elements": int(len(el)),
            "part_p999_mpa": _r(part, 3), "ratio": _r(p999[cp] / part, 3) if part > 0 else None}


def _detect(mesh, origin, shape, pitch: float, labels, opt: dict, stress_src, T: dict) -> dict:
    """The pipeline steps s1h to s6 on one lattice. Returns the candidates (without marks), the facts of the members and the grid."""
    import trimesh
    from scipy import ndimage as ndi
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cg
    from skimage.morphology import skeletonize
    from .voxelize import inside_mask

    M = {k: v[0] for k, v in METHOD.items()}
    dV = pitch ** 3
    t0 = time.perf_counter()
    S = inside_mask(mesh, origin, pitch, shape)
    T["voxelize_s"] = round(time.perf_counter() - t0, 3)
    solid_voxels = int(S.sum())
    out: dict = {"solid_voxels": solid_voxels, "candidates": [], "members_table": 0, "members_ok": 0, "branches": 0,
                 "skeleton_points": 0, "dead_end_members": 0, "frozen_excluded": 0}
    if solid_voxels == 0:
        return out
    pos = lambda ix: np.asarray(origin, float) + (np.asarray(ix, float) + 0.5) * pitch      # noqa: E731

    # s1h: skeleton, junctions, branches
    t0 = time.perf_counter()
    sk = skeletonize(S) > 0
    idx = np.argwhere(sk)
    out["skeleton_points"] = int(len(idx))
    if len(idx):
        K = np.ones((3, 3, 3), np.uint8)
        K[1, 1, 1] = 0
        nbc = ndi.convolve(sk.astype(np.uint8), K, mode="constant")[sk]
        junc = nbc > 2
        nj = sk.copy()
        nj[tuple(idx[junc].T)] = False
        lab, nl = ndi.label(nj, structure=np.ones((3, 3, 3)))
        blab = lab[tuple(idx.T)]
    else:
        nl, junc, nbc, blab = 0, np.zeros(0, bool), np.zeros(0, int), np.zeros(0, int)
    del sk
    out["branches"] = int(nl)
    T["skeleton_s"] = round(time.perf_counter() - t0, 3)

    # the frozen distance at the solid voxels (index lattice)
    t0 = time.perf_counter()
    vox = np.argwhere(S)
    vp = pos(vox)
    FD = np.full(shape, 99.0, np.float32)
    FD[tuple(vox.T)] = labels.frozen_distance(vp)
    T["frozen_distance_s"] = round(time.perf_counter() - t0, 3)

    # s2h: the member table
    t0 = time.perf_counter()
    mem = []
    if len(idx):
        idx_f = idx.astype(np.float64)
        skp = pos(idx)
        _c, rad, _t = trimesh.proximity.closest_point(mesh, skp)                      # s2h:19
        tree = cKDTree(idx_f)
        _dd, asg = tree.query(vox.astype(np.float64))                                  # s2h:21 (index units)
        if junc.any():
            jt = cKDTree(idx_f[junc])
            rj = rad[junc]
            djv, jv = jt.query(vox.astype(np.float64))
            trim = djv * pitch > (np.maximum(rj[jv], M["junction_trim_min_mm"]) + M["junction_trim_add_mm"])      # s2h:23
        else:
            trim = np.ones(len(vox), bool)
        order = np.argsort(blab, kind="stable")
        cuts = np.searchsorted(blab[order], np.arange(1, nl + 2))
        vb_all = blab[asg]
        vo = np.argsort(vb_all, kind="stable")
        vcuts = np.searchsorted(vb_all[vo], np.arange(1, nl + 2))
        for b in range(1, nl + 1):
            ids = order[cuts[b - 1]:cuts[b]]
            if len(ids) < M["member_min_skeleton_points"]:
                continue
            vi = vo[vcuts[b - 1]:vcuts[b]]
            vi = vi[trim[vi]]
            if len(vi) < M["member_min_voxels"]:
                continue
            mem.append({"id": b, "ends": int((nbc[ids] == 1).sum()), "ids": ids, "vi": vi, "vol": len(vi) * dV,
                        "ok": False})
    out["members_table"] = len(mem)
    T["members_s2h_s"] = round(time.perf_counter() - t0, 3)

    # s3: the section profile of each member
    t0 = time.perf_counter()
    for q in mem:
        ids, vi = q["ids"], q["vi"]
        n = len(ids)
        if n < M["path_min_points"]:
            continue
        Pi = idx[ids].astype(np.float64)
        pr = cKDTree(Pi).query_pairs(M["pair_radius_pitches"] + M["pair_tolerance_mm"] / pitch, output_type="ndarray")
        if len(pr) == 0:
            continue
        pr = pr[np.lexsort((pr[:, 1], pr[:, 0]))]
        w = np.linalg.norm(Pi[pr[:, 0]] - Pi[pr[:, 1]], axis=1) * pitch
        G = sp.coo_matrix((w, (pr[:, 0], pr[:, 1])), shape=(n, n)).tocsr()
        d0 = cg.dijkstra(G, indices=0, directed=False)
        d0[np.isinf(d0)] = -1
        a = int(np.argmax(d0))
        ds = cg.dijkstra(G, indices=a, directed=False)
        fin = ~np.isinf(ds)
        if fin.sum() < M["path_min_points"]:
            continue
        ii = np.flatnonzero(fin)
        o = ii[np.argsort(ds[ii], kind="stable")]
        s = ds[o]
        Po = pos(idx[ids][o])
        vv = vp[vi]
        _d, asg2 = cKDTree(idx[ids][o].astype(np.float64)).query(vox[vi].astype(np.float64))
        nst = len(o)
        cnt = np.bincount(asg2, minlength=nst).astype(float)
        dsp = np.gradient(s)
        dsp[dsp < M["station_floor_mm"]] = M["station_floor_mm"]
        wn = max(3, int(round(M["station_window_mm"] / max(np.median(dsp), M["station_floor_mm"]))))
        A = uniform_filter1d(cnt * dV, wn, mode="nearest") / uniform_filter1d(dsp, wn, mode="nearest")
        r = np.sqrt(np.maximum(A, 1e-6) / np.pi)
        L = float(s[-1] - s[0])
        ez = max(M["end_zone_min_mm"], M["end_zone_radius_factor"] * np.median(r))
        inner = (s > s[0] + ez) & (s < s[-1] - ez)
        if inner.sum() < M["inner_stations_min"]:
            inner = np.ones(nst, bool)
        ri = r[inner]
        tang = np.gradient(Po, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        fl = []
        for k in range(0, nst, max(1, nst // int(M["flatness_stations"]))):
            sel = (asg2 >= max(0, k - wn)) & (asg2 <= k + wn)
            if sel.sum() < M["flatness_min_voxels"]:
                continue
            X = vv[sel] - vv[sel].mean(0)
            t = tang[k]
            X = X - np.outer(X @ t, t)
            e = np.linalg.eigvalsh(np.cov(X.T))
            fl.append(np.sqrt(max(e[2], 1e-9) / max(e[1], 1e-9)))
        q.update(ok=True, L=L, rmean=float(ri.mean()), cv=float(ri.std() / ri.mean()),
                 ratio=float(ri.max() / max(ri.min(), M["section_ratio_floor_mm"])),
                 flat=float(np.median(fl)) if fl else 1.0, path=Po)
    ok_members = [q for q in mem if q["ok"]]
    out["members_ok"] = len(ok_members)
    out["dead_end_members"] = sum(1 for q in ok_members if q["ends"] >= 1)
    T["members_s3_s"] = round(time.perf_counter() - t0, 3)
    out["_members"] = ok_members

    # s4: the opening residue
    t0 = time.perf_counter()
    rho = opt["opening_radius_mm"]
    edt = ndi.distance_transform_edt(np.pad(S, 2, mode="edge"))[2:-2, 2:-2, 2:-2] * pitch
    core = edt >= rho
    del edt
    opened = ndi.distance_transform_edt(~core) * pitch <= rho
    del core
    res = S & ~opened
    del opened
    rl, rn = ndi.label(res, structure=np.ones((3, 3, 3)))
    del res
    rsz = np.bincount(rl.ravel())[1:] if rn else np.zeros(0, int)
    objs = ndi.find_objects(rl)
    T["residue_s"] = round(time.perf_counter() - t0, 3)

    # s5: the rules
    t0 = time.perf_counter()
    cands: list[dict] = []

    def frozen(ix):
        f = FD[tuple(np.asarray(ix).T)]
        return float((f < opt["frozen_near_mm"]).mean()), float(f.min())

    for q in ok_members:
        vix = vox[q["vi"]]
        share, fdm = frozen(vix)
        if share > opt["frozen_max_share"] or fdm <= 0:                                 # s5:36 (fd_min <= 0: inside a frozen voxel)
            out["frozen_excluded"] += 1
            continue
        common = {"idx": vix, "fd_min": fdm, "frozen_share": share, "member": q["id"], "dead_ends": q["ends"],
                  "length": q["L"], "rmean": q["rmean"], "cv": q["cv"], "ratio": q["ratio"], "flat": q["flat"]}
        if q["ends"] >= 1 and q["L"] >= opt["spur_min_length_mm"] and q["vol"] >= opt["spur_min_volume_mm3"]:      # s5:38
            cands.append({**common, "detector": "spur", "operator": "remove", "volume": q["vol"],
                          "source": f"{S5}:38", "path": None,
                          "rules": {"spur_min_length_mm": (opt["spur_min_length_mm"], q["L"]),
                                    "spur_min_volume_mm3": (opt["spur_min_volume_mm3"], q["vol"])}})
        elif q["ends"] == 0 and q["L"] >= opt["member_min_length_mm"] and q["vol"] >= opt["member_min_volume_mm3"] \
                and q["rmean"] >= opt["member_min_radius_mm"]:                                                          # s5:41
            normal = q["cv"] >= opt["member_min_cv"] and q["ratio"] >= opt["member_min_ratio"]
            mild = q["L"] >= opt["mild_min_length_mm"] and q["vol"] >= opt["mild_min_volume_mm3"] \
                and q["cv"] >= opt["mild_min_cv"] and q["ratio"] >= opt["mild_min_ratio"]
            if normal or mild:
                fair = q["flat"] >= opt["web_flatness"]
                rules = {"member_min_length_mm": (opt["member_min_length_mm"], q["L"]),
                         "member_min_volume_mm3": (opt["member_min_volume_mm3"], q["vol"]),
                         "member_min_radius_mm": (opt["member_min_radius_mm"], q["rmean"])}
                if normal:
                    rules.update(member_min_cv=(opt["member_min_cv"], q["cv"]), member_min_ratio=(opt["member_min_ratio"], q["ratio"]))
                else:
                    rules.update(mild_min_length_mm=(opt["mild_min_length_mm"], q["L"]),
                                 mild_min_volume_mm3=(opt["mild_min_volume_mm3"], q["vol"]),
                                 mild_min_cv=(opt["mild_min_cv"], q["cv"]), mild_min_ratio=(opt["mild_min_ratio"], q["ratio"]))
                rules["web_flatness"] = (opt["web_flatness"], q["flat"])
                cands.append({**common, "detector": "member", "operator": "fair" if fair else "smooth", "volume": q["vol"],
                              "source": f"{S5}:41-43", "path": q["path"], "tier": "normal" if normal else "mild", "rules": rules})
    spur_idx = [c["idx"] for c in cands if c["detector"] == "spur"]
    spur_tree = cKDTree(np.vstack(spur_idx).astype(np.float64)) if spur_idx else None
    near_idx = M["bump_spur_overlap_mm"] / pitch
    for i in range(rn):
        v = float(rsz[i]) * dV
        sl = objs[i]
        if v < opt["bump_min_volume_mm3"] or sl is None:                                                                # s5:50
            continue
        ext = np.array([x.stop - x.start for x in sl]) * pitch
        if ext.max() < opt["bump_extent_min_mm"] or ext.max() > opt["bump_extent_max_mm"]:                              # s5:53
            continue
        cc = np.argwhere(rl[sl] == i + 1) + np.array([x.start for x in sl])
        share, fdm = frozen(cc)
        if share > opt["frozen_max_share"] or fdm <= 0:
            out["frozen_excluded"] += 1
            continue
        if spur_tree is not None:                                                                                       # s5:57
            dq = spur_tree.query(cc.astype(np.float64))[0]
            if dq.min() < near_idx and (dq < near_idx).mean() > M["bump_spur_overlap_share"]:
                continue
        if v < opt["bump_min_final_volume_mm3"]:                                                                        # s6:18
            continue
        cands.append({"detector": "bump", "operator": "remove", "idx": cc, "fd_min": fdm, "frozen_share": share,
                      "volume": v, "length": float(ext.max()), "rmean": float(np.cbrt(3.0 * v / (4.0 * np.pi))),
                      "cv": None, "ratio": None, "flat": None, "dead_ends": None, "member": None, "path": None,
                      "source": f"{S4}:11-15, {S5}:48-59", "residue_component": i + 1,
                      "rules": {"bump_min_volume_mm3": (opt["bump_min_volume_mm3"], v),
                                "bump_extent_min_mm": (opt["bump_extent_min_mm"], float(ext.max())),
                                "bump_extent_max_mm": (opt["bump_extent_max_mm"], float(ext.max())),
                                "bump_min_final_volume_mm3": (opt["bump_min_final_volume_mm3"], v)}})
    del rl
    T["rules_s"] = round(time.perf_counter() - t0, 3)

    # the facts of every candidate: location, stress
    t0 = time.perf_counter()
    for c in cands:
        P = pos(c["idx"])
        c["centre"] = P.mean(0)
        c["bbox"] = np.array([P.min(0) - 0.5 * pitch, P.max(0) + 0.5 * pitch])
        c["stress"] = _stress_facts(stress_src, P, M["stress_radius_mm"])
    T["facts_s"] = round(time.perf_counter() - t0, 3)
    out["candidates"] = cands
    out["_solid_volume"] = solid_voxels * dV
    return out


# ---- bodies ------------------------------------------------------------------------------------------------------
def _outside_grid(bbox: np.ndarray, labels) -> bool:
    """True when the box of a body has no common point with the box of the run grid (`labels`). Such a mark cannot be applied:
    `apply_marks` reads the frozen distance outside the grid at the nearest grid voxel (a wrong reason), or fails in the marching
    cubes (CONTRACTS 29.3 limit 7). A body that crosses the grid edge is inside by this test: the marks of such a body work."""
    g_lo = np.asarray(labels.origin, float)
    g_hi = g_lo + np.asarray(labels.shape, float) * labels.h
    return bool((bbox[0] > g_hi).any() or (bbox[1] < g_lo).any())


def _bodies(mesh, labels, final: Path, origin, shape, pitch: float):
    """The bodies of final.stl and the removal candidates among them (section 9). Returns (facts, candidates, outside): `outside`
    are the removable bodies that lie outside the run grid (no mark is made for them)."""
    import trimesh
    from .voxelize import inside_mask
    comps = am._components(mesh)
    rows = []
    for c in comps:
        sv = am._faces_signed_volume(mesh.vertices, mesh.faces[c])
        holds = bool(labels.holds_fixed_or_load(mesh, c))
        V = mesh.vertices[np.unique(mesh.faces[c])]
        rows.append({"faces": c, "volume_mm3": abs(sv), "cavity": sv < 0, "holds_fixed_or_load": holds,
                     "bbox": np.array([V.min(0), V.max(0)]), "triangles": int(len(c))})
    rows.sort(key=lambda b: (-b["volume_mm3"], tuple(b["bbox"][0])))
    any_holds = any(b["holds_fixed_or_load"] and not b["cavity"] for b in rows)
    facts = {"count": len(rows), "cavities": sum(1 for b in rows if b["cavity"]),
             "holding_fixed_or_load": sum(1 for b in rows if b["holds_fixed_or_load"] and not b["cavity"]),
             "non_manifold_vertices": None, "list": [], "note": None}
    q = am._surface_quality(final)
    facts["non_manifold_vertices"] = q.get("non_manifold_vertices")
    cands = []
    outside = []
    for b in rows:
        removable = (not b["cavity"]) and (not b["holds_fixed_or_load"]) and any_holds
        out_grid = _outside_grid(b["bbox"], labels)
        facts["list"].append({"volume_mm3": _r(b["volume_mm3"], 3), "triangles": b["triangles"], "cavity": b["cavity"],
                              "holds_fixed_or_load": b["holds_fixed_or_load"], "removable": removable, "outside_run_grid": out_grid,
                              "bbox_mm": [_rl(b["bbox"][0], 3), _rl(b["bbox"][1], 3)],
                              "centre_mm": _rl(b["bbox"].mean(0), 3)})
        if not removable:
            continue
        common = {"detector": "body", "operator": "remove", "volume": b["volume_mm3"], "bbox": b["bbox"], "centre": b["bbox"].mean(0),
                  "length": float(np.linalg.norm(b["bbox"][1] - b["bbox"][0]))}
        if out_grid:
            outside.append(common)
            continue
        body = trimesh.Trimesh(mesh.vertices, mesh.faces[b["faces"]], process=False)
        mask = inside_mask(body, origin, pitch, shape)
        idx = np.argwhere(mask)
        if len(idx):
            walks = [(_voxel_centres(idx[w], origin, pitch), int(len(np.unique(w)))) for w in _walk_voxels(idx)]
            vox_count = int(len(idx))
        else:
            walks, vox_count = [(w, 0) for w in _walk_vertices(mesh, b["faces"])], 0
        cands.append({"detector": "body", "operator": "remove", "idx": idx, "walks": walks, "volume": b["volume_mm3"],
                      "length": float(np.linalg.norm(b["bbox"][1] - b["bbox"][0])), "rmean": float(np.cbrt(3.0 * b["volume_mm3"] / (4.0 * np.pi))),
                      "cv": None, "ratio": None, "flat": None, "dead_ends": None, "member": None, "path": None,
                      "fd_min": None, "frozen_share": None, "centre": b["bbox"].mean(0), "bbox": b["bbox"], "stress": None,
                      "source": "apply_marks._body_facts (no fixed and no load voxel)", "voxels_in_body": vox_count,
                      "rules": {}})
    if not any_holds:
        facts["note"] = ("no body holds a fixed or load voxel (the labels may not match final.stl), so no body is a removal "
                         "candidate: a mark for every body would remove the whole part")
    return facts, cands, outside


# ---- mirror ------------------------------------------------------------------------------------------------------
def _reflect(idx: np.ndarray, axis: int, plane_index: int) -> np.ndarray:
    out = idx.copy()
    out[:, axis] = 2 * plane_index - idx[:, axis] - 1
    return out


def _group_elements(axes: list[int]):
    """The reflections of the group as lists of axis indices, without the identity."""
    out = []
    for r in range(1, len(axes) + 1):
        out += [list(c) for c in itertools.combinations(axes, r)]
    return out


def _merge_twins(cands: list[dict], planes: list[dict], origin, pitch: float):
    """Merge mirror twins into one candidate (section 10). B is a twin of A (same detector and operator) when the mirror image of A's
    voxel set (the indices reflected about the plane, on the lattice where the plane is a voxel face) shares a voxel with B's set. For
    2 or 3 planes the images are the elements of the group of the reflections. A set of twins is one candidate: the larger one (ties
    by the position), with the facts of the others in `twins`. Returns (candidates, number of twins merged)."""
    if not planes:
        return cands, 0
    pm = {pl["axis_index"]: int(round((pl["at_mm"] - origin[pl["axis_index"]]) / pitch)) for pl in planes}
    elems = _group_elements(sorted(pm))
    feat = [i for i, c in enumerate(cands) if c["detector"] != "body" and len(c["idx"])]
    if len(feat) < 1:
        return cands, 0
    imgs: dict = {}
    for i in feat:
        for e, el in enumerate(elems):
            a = cands[i]["idx"]
            for k in el:
                a = _reflect(a, k, pm[k])
            imgs[(i, e)] = a
    allv = [cands[i]["idx"] for i in feat] + list(imgs.values())
    base = np.min([a.min(0) for a in allv], axis=0) - 1
    span = np.max([a.max(0) for a in allv], axis=0) - base + 2

    def keys(a):
        return np.unique(np.ravel_multi_index((a - base).T, tuple(int(x) for x in span)))

    own = {i: keys(cands[i]["idx"]) for i in feat}
    img = {k: keys(v) for k, v in imgs.items()}
    parent = {i: i for i in feat}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pair: dict = {}
    on_plane: dict = {}
    for i in feat:
        for e in range(len(elems)):
            k_own = int(np.intersect1d(img[(i, e)], own[i], assume_unique=True).size)
            if k_own and len(elems[e]) == 1:
                on_plane[i] = max(on_plane.get(i, 0), k_own)
        for j in feat:
            if j <= i or cands[j]["detector"] != cands[i]["detector"] or cands[j]["operator"] != cands[i]["operator"]:
                continue
            best = (0, 0.0)
            for e in range(len(elems)):
                ov = int(np.intersect1d(img[(i, e)], own[j], assume_unique=True).size)
                if ov > best[0]:
                    best = (ov, ov / int(np.union1d(img[(i, e)], own[j]).size))
            if best[0]:
                pair[(i, j)] = best
                parent[find(j)] = find(i)
    groups: dict = {}
    for i in feat:
        groups.setdefault(find(i), []).append(i)
    rep_of: dict = {}
    for g in groups.values():
        rep = min(g, key=lambda i: (-cands[i]["volume"], tuple(round(float(x), 3) for x in cands[i]["centre"])))
        rep_of.update({i: rep for i in g})
        c = cands[rep]
        if rep in on_plane:
            c["on_mirror_plane_voxels"] = on_plane.get(rep, 0)
        if len(g) > 1:
            twins = []
            for o in sorted(set(g) - {rep}):
                pr = pair.get((min(rep, o), max(rep, o))) or max([v for (a, b), v in pair.items() if o in (a, b)], key=lambda v: v[0])
                co = cands[o]
                twins.append({"detector": co["detector"], "centre_mm": _rl(co["centre"], 3), "detected_volume_mm3": _r(co["volume"], 3),
                              "twin_overlap_voxels": pr[0], "twin_iou": _r(pr[1], 3)})
            c["twins"] = twins
    keep = [c for i, c in enumerate(cands) if i not in rep_of or rep_of[i] == i]
    return keep, len(feat) - len(groups)


# ---- applicability -----------------------------------------------------------------------------------------------
def _applicability(det: dict, opt: dict) -> dict:
    mem = det.get("_members", [])
    vol = sum(q["vol"] for q in mem)
    dead = [q for q in mem if q["ends"] >= 1]
    flat = [q for q in mem if q["flat"] >= opt["web_flatness"]]
    solid = det.get("_solid_volume") or 0.0
    f = lambda a, b: _r(a / b, 3) if b else None            # noqa: E731
    return {"members_in_table": det["members_table"], "members_with_profile": len(mem), "branches": det["branches"],
            "skeleton_points": det["skeleton_points"], "dead_end_member_share": f(len(dead), len(mem)),
            "dead_end_member_volume_share": f(sum(q["vol"] for q in dead), vol), "member_volume_share": f(vol, solid),
            "flat_member_volume_share": f(sum(q["vol"] for q in flat), vol), "note": None}


# ---- the marks ---------------------------------------------------------------------------------------------------
def _sort_key(c: dict, order: str):
    group = 0 if c["operator"] == "remove" else (1 if c["operator"] == "smooth" else 2)
    primary = -c["volume"] if group == 0 else -(c["cv"] or 0.0)
    st = (c.get("stress") or {}).get("ratio")
    cen = tuple(round(float(x), 3) for x in c["centre"])
    if order == "stress":
        return (group, st is None, st if st is not None else 0.0, primary, cen)
    return (group, primary, cen)


def _reason(c: dict, opt: dict) -> str:
    v = f"{c['volume']:.0f} mm3"
    if c["detector"] == "spur":
        return f"Dead-end spur: a skeleton branch with a free end, {c['length']:.0f} mm long, {v}."
    if c["detector"] == "bump":
        return f"Thin bump: solid thinner than {2 * opt['opening_radius_mm']:g} mm that sticks out, {c['length']:.0f} mm long, {v}."
    if c["detector"] == "body":
        return f"Loose body: it holds no fixed and no load voxel, {v}."
    if c["operator"] == "fair":
        return (f"Flat web (flatness {c['flat']:.2f}), section radius CV {c['cv']:.2f}, max/min {c['ratio']:.1f}, {c['length']:.0f} mm "
                "long; the FAIR operator fairs it in place.")
    return (f"Uneven strut: section radius CV {c['cv']:.2f}, max/min {c['ratio']:.1f}, {c['length']:.0f} mm long; "
            "the SMOOTH operator rebuilds it as one faired tube.")


def _make_marks(c: dict, first_id: int, origin, pitch: float, opt: dict, density, mirror: bool) -> list[dict]:
    """The marks of one candidate: one mark for each 26-connected component of its voxel set (a spur, a bump or a body has one in
    nearly every case). A path candidate (SMOOTH, FAIR) is one mark. The facts of the candidate are in each mark; `component`
    says which part of the detected set the mark holds."""
    if c["detector"] == "body":
        parts = c["walks"]
    elif c["operator"] == "remove":
        parts = [(_voxel_centres(c["idx"][w], origin, pitch), int(len(np.unique(w)))) for w in _walk_voxels(c["idx"])]
    else:
        parts = [(am._resample(c["path"], PATH_STEP_MM), int(len(c["idx"])))]
    if c["operator"] == "remove":
        radius, extra = _mark_radius(pitch), {"remove_mode": "cut"}
    elif c["operator"] == "fair":
        radius, extra = METHOD["fair_zone_mm"][0], {"smooth_mode": "fair"}
    else:
        radius, extra = max(round(c["rmean"], 4), 1e-4), {"smooth_mode": "tube"}
    n = len(parts)
    vol = c["volume"]
    conf = "low" if (c["fd_min"] is not None and c["fd_min"] < opt["near_frozen_mm"]) else "high"
    marks = []
    for k, (pts, nvox) in enumerate(parts, 1):
        note = _reason(c, opt)
        if n > 1:
            note += f" Part {k} of {n}: the detected set has {n} separate parts, and each part is one mark."
        mark = {"id": first_id + k - 1, "type": "remove" if c["operator"] == "remove" else "smooth",
                "points": np.round(pts, 4).tolist(), "radius_mm": radius, "mirror": bool(mirror), **extra, "note": note}
        measured = {"detected_volume_mm3": _r(vol, 3), "removal_reach_mm": REMOVAL_REACH_MM if c["operator"] == "remove" else None,
                    "mass_g_at_run_density": _r(vol * density * 1e-3, 4) if density else None,
                    "volume_basis": f"voxels x pitch^3; the pipeline mass limits are volumes at {PIPELINE_DENSITY_G_MM3} g/mm3 (geometric sizes)",
                    "length_mm": _r(c["length"], 3), "mean_radius_mm": _r(c["rmean"], 3),
                    "radius_basis": "sphere of equal volume" if c["detector"] in ("bump", "body") else "mean section radius",
                    "length_in_voxels": _r(c["length"] / pitch, 2), "mean_radius_in_voxels": _r(c["rmean"] / pitch, 2),
                    "voxels": int(len(c["idx"])), "points": int(len(pts)), "cv": _r(c["cv"], 3), "max_over_min": _r(c["ratio"], 3),
                    "flatness": _r(c["flat"], 3), "dead_ends": c["dead_ends"], "frozen_share": _r(c["frozen_share"], 3),
                    "components": n}
        facts = {"detector": c["detector"], "operator": {"remove": "remove cut", "smooth": "smooth tube", "fair": "smooth fair"}[c["operator"]],
                 "source": c["source"], "confidence": conf, "reason": note,
                 "location": {"centre_mm": _rl(c["centre"], 3), "bbox_mm": [_rl(c["bbox"][0], 3), _rl(c["bbox"][1], 3)],
                              "nearest_frozen_mm": _r(c["fd_min"], 3), "near_frozen": conf == "low"},
                 "thresholds_used": {kk: {"limit": _r(a, 6), "value": _r(b, 4)} for kk, (a, b) in c["rules"].items()},
                 "measured": measured, "stress": c.get("stress"),
                 "mirror": ({"twins": c.get("twins", []), "on_mirror_plane_voxels": c.get("on_mirror_plane_voxels", 0)}
                            if c.get("twins") or c.get("on_mirror_plane_voxels") else None)}
        if n > 1:
            facts["component"] = {"index": k, "of": n, "voxels": nvox,
                                  "detected_volume_mm3": _r(nvox * pitch ** 3, 3) if c["operator"] == "remove" else None}
        if c.get("tier"):
            facts["tier"] = c["tier"]
        mark["cleanup"] = facts
        marks.append(mark)
    return marks


def _marks_text(doc: dict) -> str:
    """The marks file as text: the header with indent, each point on its own line."""
    marks = doc["marks"]
    head = json.dumps({**doc, "marks": "@@MARKS@@"}, indent=1)
    parts = []
    for m in marks:
        pts = m["points"]
        body = json.dumps({**m, "points": "@@POINTS@@"}, indent=1)
        txt = "[\n" + ",\n".join(json.dumps(p) for p in pts) + "\n]"
        parts.append(body.replace('"@@POINTS@@"', txt))
    return head.replace('"@@MARKS@@"', "[\n" + ",\n".join(parts) + "\n]" if parts else "[]") + "\n"


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---- the judge estimate ------------------------------------------------------------------------------------------
def _judge_estimate(d: Path, yaml_path, marks_file, count: int, root, opt: dict) -> dict:
    """The cost of judging the list: a part that `ratchet.check_report` MEASURES (printcheck on the start part) and parts that are
    models (the verify time; the apply time is not in it). With `root`, the mean time of the candidates in its log is measured."""
    try:
        from . import ratchet
        rep = ratchet.check_report(d, yaml_path, source=f"marks:{marks_file}" if marks_file else None, root=root)
    except UserError as exc:
        return {"available": False, "reason": str(exc)}
    except Exception as exc:                     # a fact about the estimate, not a crash of the list
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    tools = rep.get("tools") or {}
    pc, vf = tools.get("printcheck") or {}, tools.get("verify")
    out: dict = {"available": True, "yaml": Path(rep["yaml"]).name, "rules_ready": bool(rep.get("ready")),
                 "rules_not_ready": [{"rule": r["rule"], "missing": r["missing"]} for r in rep.get("rules", []) if not r["ready"]]}
    pcs = pc.get("seconds") if pc.get("ready") else None
    vs = (vf or {}).get("estimated_seconds") if vf is not None else 0.0
    est = rep.get("estimate") or {}
    # The count is the count of THIS list. With `root`, check_report subtracts the candidates that the log of the root already holds
    # (of any source), so only its baseline state is taken from it: 0 calls when the root has its baseline, else 1 (CONTRACTS 29.3 part 6).
    n = count
    base = (est["judge_calls"] - est["candidates"]) if ("judge_calls" in est and "candidates" in est) else 1
    calls = None if n is None else n + base
    out.update(candidates=n, judge_calls=calls, baseline_included=None if n is None else bool(calls > n),
               printcheck_seconds={"value": pcs, "kind": "measured: printcheck on the start part (the list call includes this check)"},
               verify_seconds_each={"value": vs, "kind": "model: the verify time estimate at the mesh size of the rules"},
               apply_seconds_each={"value": None, "kind": "not included: 3 to 13 s for each mark on the hook (measured with apply_marks)"},
               verify_memory_gb=(vf or {}).get("estimate_gb"), verify_ready=None if vf is None else bool(vf.get("ready")),
               verify_problems=(vf or {}).get("reasons") or [])
    if pcs is not None and vs is not None and calls is not None:
        out["seconds_total_model"] = round(calls * (pcs + vs), 1)
        out["seconds_total_note"] = ("calls x (printcheck measured + verify model). The model of the verify time was 2.1 times below "
                                     "the real judge time on the hook, because apply_marks and the remesh are not in it.")
    if root is not None and (Path(root) / "ratchet.jsonl").is_file():
        from .ratchet import read_log
        rows, _ = read_log(root)
        tot = [r["judge"]["timing_s"]["total"] for r in rows if r.get("type") == "candidate"
               and ((r.get("judge") or {}).get("timing_s") or {}).get("total") is not None]
        src = [r["judge"]["timing_s"]["source"] for r in rows if r.get("type") == "candidate"
               and ((r.get("judge") or {}).get("timing_s") or {}).get("source") is not None]
        if tot:
            out["measured_from_root"] = {"candidates_in_log": len(tot), "seconds_each_mean": round(sum(tot) / len(tot), 1),
                                         "source_seconds_each_mean": round(sum(src) / len(src), 1) if src else None,
                                         "seconds_total": None if n is None else round(n * sum(tot) / len(tot), 1), "kind": "measured: judge.timing_s.total of the log of that root"}
    if vf is not None and not vf.get("ready"):
        out["options"] = ["a coarser mesh_size_mm in the acceptance rules (the memory and time of verify fall)",
                          "fewer candidates only shorten the time: the judge calls run one after the other"]
    return out


# ---- the list ----------------------------------------------------------------------------------------------------
def cleanup_list(run_dir, *, out_dir=None, options=None, overwrite: bool = False, estimate_only: bool = False,
                 yaml_path=None, root=None) -> dict:
    """List the cleanup candidates of a finished part as a marks file, in a NEW folder. Facts and options; nothing is edited.

    run_dir: a finished run, or an `acc_NN` folder of a ratchet root (needs run.json, labels.bin and result/final.stl;
        verify/fields.npz adds stress facts).
    out_dir: the new folder. Default: `<run folder>-cleanup` beside the run folder (a number is added when it exists); for an
        `acc_NN` folder of a ratchet root `<root>-cleanup-acc_NN` beside the root. Refused (the reply says why and what to do): a
        run folder or a folder inside one, a ratchet root or a folder inside one, a folder with an input file, a symbolic link, and
        a folder that holds files and is not an earlier cleanup folder.
    options: the names of THRESHOLDS (pitch_mm, opening_radius_mm, spur_*, bump_*, member_*, mild_*, web_flatness, frozen_*,
        near_frozen_mm, max_candidates), `order` (pipeline or stress), `kinds` (body, spur, bump, member), `max_voxels`, `max_gb`.
    estimate_only: the detector estimate and the judge estimate; no detection, no file.
    yaml_path: the YAML of the run, when run.json names a file that is not there.
    root: a ratchet root whose log gives the measured judge time.
    Returns the full report (cleanup.json holds it), or `{"ok": false, "error": ...}` when it cannot start.
    Writes `marks.json` and `cleanup.json` into the new folder. The run folder and every input stay as they are.
    """
    try:
        return _list(Path(run_dir).expanduser().resolve(), out_dir, options, overwrite, estimate_only, yaml_path, root)
    except fb._Refused as exc:
        return {"ok": False, "error": str(exc), "refused": {"reason": exc.reason, "path": exc.path, "options": exc.options}}
    except UserError as exc:
        return {"ok": False, "error": str(exc)}
    except MemoryError:
        return {"ok": False, "error": "out of memory. Options: pitch_mm larger (a coarser grid), or max_gb lower to refuse earlier."}
    except Exception as exc:  # a fact about this tool, not a crash of the caller
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


def _list(d: Path, out_dir, options, overwrite: bool, estimate_only: bool, yaml_arg, root) -> dict:
    t_start = time.monotonic()
    import trimesh
    if not d.is_dir():
        raise UserError(f"run folder not found: {d}")
    problems: list[str] = []
    notes: list[str] = []
    run = fb._load_json(d / "run.json", problems, "run.json")
    if run is None:
        raise UserError(f"{d / 'run.json'} is missing or unreadable: the grid of the run is not known")
    grid = fb._grid_of(run)
    if grid is None:
        raise UserError("run.json has no valid shape, h and origin")
    _shape_run, h_run, origin_run = grid
    final = d / "result" / "final.stl"
    if not final.is_file():
        from .runjob import stopped_hint
        raise UserError(f"{final} is missing: there is no finished part to scan. Run `topopt run` to the end first."
                        + stopped_hint(d))
    opt, given, oprob = _norm_options(options)
    problems += oprob
    pitch = float(opt["pitch_mm"])
    planes, plane_facts = am._planes_of_run(run)
    notes += plane_facts
    ypath = Path(str(yaml_arg)).expanduser().resolve() if yaml_arg else fb._run_yaml_path(run, d)
    input_dirs = set()
    if ypath is not None and ypath.is_file():
        yd, _ = fb._yaml_doc(ypath)
        input_dirs = {ypath.parent.resolve()} | {p.parent for p in fb._region_paths((yd or {}).get("regions"), ypath.parent)}
    root_path = Path(root).expanduser().resolve() if root else None
    out = None if estimate_only else _plan_out(out_dir, d, input_dirs, overwrite)
    if out is not None:
        fb._check_targets(out, {"marks.json", "cleanup.json"})
    mesh = trimesh.load(str(final), file_type="stl", force="mesh")
    origin, shape = _lattice(mesh.bounds, pitch, planes)
    est = _detector_estimate(shape, pitch, opt)
    if est.get("refused"):
        return _guard_refusal(est)
    if estimate_only:
        je = _judge_estimate(d, ypath, None, opt["max_candidates"], root_path, opt)
        return {"ok": True, "estimate_only": True, "run_dir": str(d), "out_dir": None, "files": {"marks": None, "report": None},
                "estimate": {"detector": est, "judge": je, "judge_note": "up to max_candidates candidates; the count is known after the search"},
                "options": _threshold_table(opt, given), "problems": problems, "notes": notes,
                "elapsed_s": round(time.monotonic() - t_start, 2), "peak_rss_mb": am._peak_rss_mb(), "load_avg": am._load_avg()}
    labels_arr, lab_problem = fb._read_labels(d, run)
    if labels_arr is None:
        raise UserError(f"the frozen regions are not known: {lab_problem}. labels.bin comes with the run (`topopt run` writes it at the start).")
    labels = am.Labels(labels_arr, origin_run, h_run)
    final_sha = fb._sha256(final)
    metrics = fb._load_json(d / "result" / "metrics.json", problems, "result/metrics.json", expected=False)
    density = am._density_of(run, metrics)
    stress_src = _stress_source(d, final_sha, problems, notes)
    T: dict = {}

    det = _detect(mesh, origin, shape, pitch, labels, opt, stress_src, T)
    t0 = time.perf_counter()
    body_facts, body_cands, body_outside = _bodies(mesh, labels, final, origin, shape, pitch)
    T["bodies_s"] = round(time.perf_counter() - t0, 3)
    feature_cands, merged = _merge_twins(det["candidates"], planes, origin, pitch)
    appl = _applicability(det, opt)

    # which kinds go into the file (section 5.4 and 9)
    auto = opt["kinds"] is None
    kinds = ("body",) if (auto and body_cands) else (tuple(k for k in ("spur", "bump", "member")) if auto else tuple(opt["kinds"]))
    everything = body_cands + feature_cands
    chosen = [c for c in everything if c["detector"] in kinds]
    skipped_kind = [c for c in everything if c["detector"] not in kinds]
    chosen.sort(key=lambda c: (0 if c["detector"] == "body" else 1,) + _sort_key(c, opt["order"]))
    cap = opt["max_candidates"]
    listed = chosen if cap is None else chosen[:cap]
    over_budget = [] if cap is None else chosen[cap:]
    why_kind = ("a removable body exists, so the file holds the body marks only: judge them first with the printcheck rules, then list "
                "the acc_01 of that root for the features (a printcheck-only root would judge a feature cut with no stress rule)"
                if auto and body_cands else f"kinds does not hold {{detector}}")
    mix = (not auto) and "body" in kinds and any(k != "body" for k in kinds) and bool(body_cands)

    marks = []
    for c in listed:
        marks += _make_marks(c, len(marks) + 1, origin, pitch, opt, density, bool(c.get("twins")))
    if any(c["detector"] in ("spur", "member") for c in listed):
        appl["note"] = NOTE_STRUTS

    def line(c, why):
        return {"detector": c["detector"], "operator": c["operator"], "centre_mm": _rl(c["centre"], 3),
                "detected_volume_mm3": _r(c["volume"], 3), "length_mm": _r(c["length"], 3), "why_not_listed": why,
                "stress_ratio": (c.get("stress") or {}).get("ratio")}
    not_listed = [line(c, f"budget: max_candidates {cap}") for c in over_budget] + \
                 [line(c, why_kind.replace("{detector}", c["detector"])) for c in skipped_kind]
    for c in body_outside:                       # a removable body outside the run grid: a fact, no mark (limit 7 of 29.3)
        not_listed.append(dict(line(c, "outside_run_grid: the box of the body has no common point with the grid of the run, so "
                                       "apply_marks cannot cut it"), outside_run_grid=True, options=OUTSIDE_GRID_OPTIONS))
    if body_outside:
        notes.append(f"{len(body_outside)} loose body(ies) lie outside the grid of the run and have no mark (not_listed, outside_run_grid). "
                     + OUTSIDE_GRID_OPTIONS[0])
    stress_head = None if stress_src is None else {"source": stress_src["file"], "cases": stress_src["cases"],
                                                   "part_p999_mpa": {c: _r(v, 3) for c, v in stress_src["part_p999"].items()}}
    by_type: dict = {}
    for m in marks:
        by_type[m["cleanup"]["operator"]] = by_type.get(m["cleanup"]["operator"], 0) + 1
    counts = {"found": len(everything) + len(body_outside) + merged, "after_twin_merge": len(everything) + len(body_outside),
              "listed": len(listed), "marks": len(marks), "not_listed": len(not_listed), "by_operator": by_type, "twins_merged": merged,
              "frozen_excluded": det["frozen_excluded"], "outside_run_grid": len(body_outside)}
    mirror_head = None if not planes else {"planes": [{k: pl[k] for k in ("axis", "at_mm", "source")} for pl in planes],
                                           "pairs_merged": merged}
    header = {"format": CLEANUP_FORMAT, "tool": "topopt.cleanup list",
              "run": {"name": run.get("name"), "final_stl_sha256": final_sha, "h_mm": h_run, "density_g_cm3": _r(density, 4)},
              "pitch_mm": pitch, "pitch_is_default": "pitch_mm" not in given,
              "grid": {"shape": [int(s) for s in shape], "voxels": est["voxels"], "origin_mm": _rl(origin, 6)},
              "thresholds": _threshold_table(opt, given), "kinds": list(kinds), "mirror": mirror_head,
              "applicability": appl,
              "bodies": {k: body_facts[k] for k in ("count", "cavities", "holding_fixed_or_load", "non_manifold_vertices", "note")}
                        | {"removable": [b for b in body_facts["list"] if b["removable"]]},
              "stress": stress_head,
              "real_change": "the real change of each mark is in ratchet.jsonl (source.facts.change_mm3 under source.ref = marks.json#<id>) "
                             "and in cand_NNN/apply_marks.json (marks[0].edit.part_volume_change_mm3 and modes); detected_volume_mm3 "
                             "is the detected set, and the cut reaches removal_reach_mm around the points",
              "counts": counts, "order": "bodies; remove by volume; smooth by CV; fair by CV" if opt["order"] == "pipeline"
              else "bodies; remove, smooth, fair, each by the lowest stress ratio first (unknown last)"}
    if mix:
        header["mixed_kinds_note"] = ("this file mixes bodies and features: a root with printcheck-only rules would judge the feature cuts "
                                      "with no stress and no stiffness rule")
    doc = {"version": 1, "units": "mm", "source_sha256": final_sha, "cleanup": header, "marks": marks}
    try:
        out.mkdir(parents=True, exist_ok=True)
        marks_file = out / "marks.json"
        _write_text(marks_file, _marks_text(fb._jsonable(doc)))
    except OSError as exc:
        raise UserError(f"cannot write in {out}: {type(exc).__name__}: {exc}. Name another out_dir.")
    marks_sha = fb._sha256(marks_file)
    if not marks:
        notes.append("No candidate was found at these limits: marks.json has an empty list. Options: other limits (--set NAME=VALUE), "
                     "another pitch_mm, or no cleanup (the part has nothing that the detectors list).")
    je = _judge_estimate(d, ypath, marks_file, len(marks), root_path, opt) if marks else {"available": False, "reason": "no candidate"}
    T["total_s"] = round(time.monotonic() - t_start, 2)
    stale = sorted(p.name for p in out.iterdir() if p.is_file() and p.name not in {"marks.json", "cleanup.json"})
    if stale:
        notes.append("Files from an earlier call stay in the folder: " + ", ".join(stale[:10]))
    routes = [{"id": "ratchet", "text": f"python -m topopt.ratchet run {d} --source marks:{marks_file} (a new --root): the marks: source "
                                         "applies each mark with apply_marks and judges it with the rules of the YAML."},
              {"id": "refine", "text": "marks.json is a marks file of format 1: apply_marks and the Refine page read it; the keys cleanup, "
                                        "smooth_mode and remove_mode are extra keys."}]
    report = {"ok": True, "format": CLEANUP_FORMAT, "run_dir": str(d), "run_name": run.get("name"), "out_dir": str(out),
              "files": {"marks": "marks.json", "report": "cleanup.json"}, "marks_sha256": marks_sha,
              "source": {"final_stl_sha256": final_sha}, "header": header, "options": header["thresholds"], "method": {
                  k: {"value": v[0], "unit": v[1], "anchor": v[2]} for k, v in METHOD.items()},
              "marks": [{"id": m["id"], "type": m["type"], "mirror": m["mirror"], "points": len(m["points"]), "radius_mm": m["radius_mm"],
                         "smooth_mode": m.get("smooth_mode"), "remove_mode": m.get("remove_mode"), "note": m["note"],
                         "cleanup": m["cleanup"]} for m in marks],
              "bodies": body_facts, "not_listed": not_listed, "counts": counts,
              "estimate": {"detector": est, "judge": je}, "routes": routes, "problems": problems, "notes": notes,
              "timings_s": T, "elapsed_s": T["total_s"], "peak_rss_mb": am._peak_rss_mb(), "load_avg": am._load_avg()}
    report = fb._jsonable(report)
    try:
        portable = fb._scrub_strings(report, sorted([(str(out), "."), (str(d), fb._relpath(d, out))], key=lambda x: -len(x[0])))
        portable["run_dir"] = fb._relpath(d, out)
        portable["out_dir"] = "."
        portable["paths_are_relative_to"] = "the folder of this file (cleanup.json)"
        _write_text(out / "cleanup.json", json.dumps(portable, indent=1) + "\n")
    except OSError as exc:
        raise UserError(f"cannot write cleanup.json in {out}: {exc}")
    return report


# ---- the short report and the script -----------------------------------------------------------------------------
def short_report(report: dict) -> dict:
    """The short reply: what the agent needs to decide. The full report is in cleanup.json (and in the return value of
    `cleanup_list`)."""
    if report.get("estimate_only"):
        return report
    if not report.get("marks") and not report.get("out_dir") and not report.get("header"):
        return report                               # an error, or a refusal, with no report
    lines = []
    for m in report.get("marks", []):
        c = m["cleanup"]
        me, loc = c["measured"], c["location"]
        s = c.get("stress") or {}
        line = {"id": m["id"], "type": m["type"], "operator": c["operator"], "detector": c["detector"], "confidence": c["confidence"],
                "centre_mm": loc["centre_mm"], "detected_volume_mm3": me["detected_volume_mm3"], "length_mm": me["length_mm"],
                "points": me["points"], "component": c.get("component"), "nearest_frozen_mm": loc["nearest_frozen_mm"], "stress_ratio": s.get("ratio"),
                "stress_p999_mpa": s.get("p999_mpa"), "mirror": m["mirror"], "twins": len((c.get("mirror") or {}).get("twins") or []) or None,
                "cv": me.get("cv"), "flatness": me.get("flatness")}
        lines.append({k: v for k, v in line.items() if v is not None})
    h = report.get("header") or {}
    est = report.get("estimate") or {}
    je = est.get("judge") or {}
    out = {"ok": report.get("ok"), "run_dir": report.get("run_dir"), "out_dir": report.get("out_dir"), "files": report.get("files"),
           "counts": report.get("counts"), "kinds": h.get("kinds"), "candidates": lines, "not_listed": len(report.get("not_listed") or []),
           "pitch_mm": h.get("pitch_mm"), "applicability": h.get("applicability"), "bodies": h.get("bodies"),
           "stress": h.get("stress"), "mirror": h.get("mirror"),
           "estimate": {"detector": {k: (est.get("detector") or {}).get(k) for k in ("kind", "voxels", "memory_estimate_mb", "limit_mb")},
                        "judge": {k: je.get(k) for k in ("available", "reason", "candidates", "judge_calls", "printcheck_seconds",
                                                           "verify_seconds_each", "apply_seconds_each", "seconds_total_model",
                                                           "measured_from_root", "verify_memory_gb", "rules_ready", "rules_not_ready",
                                                           "options") if k in je}},
           "routes": report.get("routes"), "problems": report.get("problems"), "notes": report.get("notes"),
           "elapsed_s": report.get("elapsed_s"), "detail": False, "report_file": "cleanup.json",
           "more": "cleanup.json holds the full report: the facts of every candidate with the thresholds used, the list of the bodies, "
                   "the candidates that were not listed, the constants of the method, the timings. Call again with detail=true to get it "
                   "in the reply."}
    if report.get("error"):
        out["error"] = report["error"]
    return out


def _parse_set(items) -> dict:
    out = {}
    for s in items or []:
        k, sep, v = s.partition("=")
        if not sep:
            raise UserError(f"--set {s!r} is not NAME=VALUE")
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m topopt.cleanup", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="list the cleanup candidates of a finished part as a marks file (nothing is edited)")
    p.add_argument("run_dir")
    p.add_argument("--out", default=None, help="the new folder (default: <run folder>-cleanup)")
    p.add_argument("--overwrite", action="store_true", help="write into an earlier cleanup folder")
    p.add_argument("--yaml", default=None, help="the YAML of the run, when run.json names a file that is not there")
    p.add_argument("--pitch", type=float, default=None, help="the detection pitch in mm (default 0.5)")
    p.add_argument("--max-candidates", type=int, default=None, help="the number of marks written (default 10)")
    p.add_argument("--all", action="store_true", help="write every candidate (no budget)")
    p.add_argument("--order", choices=list(ORDERS), default=None)
    p.add_argument("--kinds", default=None, help="comma list of body,spur,bump,member (default: bodies alone when a removable body exists)")
    p.add_argument("--max-voxels", type=float, default=None)
    p.add_argument("--max-gb", type=float, default=None)
    p.add_argument("--root", default=None, help="a ratchet root whose log gives the measured judge time")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a limit of THRESHOLDS, for example spur_min_length_mm=4")
    p.add_argument("--estimate", action="store_true", help="the estimate only: no search, no file")
    p.add_argument("--detail", action="store_true", help="print the full report (cleanup.json always holds it)")
    a = ap.parse_args(argv)
    try:
        opts = _parse_set(a.set)
        for k, v in (("pitch_mm", a.pitch), ("order", a.order), ("kinds", a.kinds), ("max_voxels", a.max_voxels), ("max_gb", a.max_gb)):
            if v is not None:
                opts[k] = v
        if a.all:
            opts["max_candidates"] = None
        elif a.max_candidates is not None:
            opts["max_candidates"] = a.max_candidates
        res = cleanup_list(a.run_dir, out_dir=a.out, options=opts, overwrite=a.overwrite, estimate_only=a.estimate,
                           yaml_path=a.yaml, root=a.root)
    except (UserError, ValueError) as exc:
        res = {"ok": False, "error": str(exc)}
    if not a.detail:
        res = short_report(res)
    print(json.dumps(fb._jsonable(res), indent=1))
    if not res.get("ok"):
        print(f"error: {res.get('error')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
