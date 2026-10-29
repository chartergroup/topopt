"""Refine by feedback: run history, compare two runs, marks as hints (CONTRACTS.md section 21).

This module is advisory. It gives facts, estimates and options. The agent decides.

- `run_history(root)`: facts of the run folders under a folder. It writes nothing.
- `compare_runs(run_a, run_b)`: the setup diff and the metrics diff of two runs. It writes nothing.
- `make_hints(run_dir, marks)`: turns the marks of the refine page into facts and options. It writes one
  NEW folder: a proposed YAML (a copy of the YAML of the run, with the changes), region STL files and
  `hints.json`.

None of them changes a mesh, a run folder, an input YAML or a region STL of the run. A mark is a point
of emphasis, not an absolute: the tool says which option the proposed YAML uses, and the options say
what else is possible.

Run it as a script: `python -m topopt.feedback history|compare|hints ...` (JSON on stdout).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from . import earlier_run
from .config import UserError, load_config
from .runinfo import run_yaml_path, sha256_file

HINTS_FORMAT = 1
LABEL_NAMES = {0: "empty", 1: "design", 2: "fixed", 3: "load", 4: "keep_out"}
RUN_FILES = ("run.json", "labels.bin", "status.json", "history.jsonl", "run.log")
RESULT_FILES = ("final.stl", "metrics.json", "preview.bin", "refine.html", "stress.npz")
NEWER_TOLERANCE_S = 2.0        # started_at has a resolution of 1 s: a file this much newer is "newer"
LABELS_CHECK_AUTO_MAX = 10_000_000   # voxels: above this, compare asks for check_labels=true
METRIC_KEYS = ("compliance_initial", "compliance_final", "mass_design_space_g", "mass_final_g",
               "mass_saved_pct", "volume_final_mm3", "run_time_s", "iterations", "triangles")
# run.json keys that the setup diff does not list: the date, the YAML path (shown on its own), and the numbers
# that follow from other keys (the `derived` block holds forces, volumes and estimates; `problem_key` is a hash of other keys and of the
# region files). A few `derived` keys stay.
RUN_JSON_SKIP = ("format", "started_at", "yaml_path", "derived", "lost_force", "problem_key")
RUN_JSON_DERIVED_KEPT = ("filter_radius_mm", "min_feature_mm", "case_combine", "density_g_cm3",
                         "volume.target_volume_mm3", "volume.target_mass_g")
# Filter facts are copied generically, so a new filter key (for example the solid rule and min_member_mm of A3a) shows in
# compare without a code change: every `derived` key of run.json whose name starts with FILTER_PREFIX, and every
# top-level key of metrics.json with that start. Inside such a block, FILTER_SKIP names prose and estimates that follow
# from other keys (a list of sentences, size estimates, a note text): they are not copied.
FILTER_PREFIX = "filter"
FILTER_SKIP = ("facts", "size", "min_member_note")
# What run.json does not record: a change of these in the YAML cannot be seen by comparing the YAML with run.json.
NOT_IN_RUN_JSON = ("material.E", "material.nu", "settings.penal", "settings.beta_schedule", "settings.move_limit")
HISTORY_DEFAULT_LIMIT = 20
HISTORY_MAX_DEPTH = 5          # levels of folders that run_history walks
HISTORY_MAX_FOLDERS = 5000     # folders that run_history looks at before it stops (and says so)

MARK_TYPES = ("add", "remove", "smooth")
# What a mark type can become, as options. `implemented` says that `options` can choose it here.
MARK_OPTIONS = {
    "remove": [
        {"id": "keep_out", "implemented": True,
         "text": "A keep-out STL at the mark: a sphere, or a chain of capsules along a stroke, of radius "
                 "radius_mm x radius_scale (also at y -> -y with mirror). No material goes into it. It is a "
                 "hard void, and post-process cuts the final mesh with the exact shape. With "
                 "protect_regions, the fixed and load regions are cut out of it. A region of an earlier mark "
                 "is cut out of it too: the first mark wins an overlap."},
        {"id": "skip", "implemented": True,
         "text": "No YAML change for this mark. The facts stay in the report."},
        {"id": "smaller_keep_out", "implemented": True,
         "text": "Use options.radius_scale below 1 (for example 0.5): a smaller void, as a point of "
                 "emphasis and not as a rule."},
        {"id": "volume_fraction", "implemented": False,
         "text": "Lower settings.volume_fraction. This removes material everywhere, not at the mark."},
    ],
    "add": [
        {"id": "solid", "implemented": True,
         "text": "A forced-solid region at the mark: a sphere, or a chain of capsules along a stroke, of radius "
                 "radius_mm x radius_scale (also at y -> -y with mirror). The YAML key is a load region that no "
                 "load case uses. It stays solid and carries no force, as the `solid` role of build_regions. "
                 "With clip_to_design, it is cut to the design space. Post-process removes every solid island "
                 "that does not touch the fixed region, so a region that touches no material of the run can "
                 "be missing in the final mesh (the fact `final_material_in_region_mm3` says if it touches)."},
        {"id": "skip", "implemented": True,
         "text": "No YAML change for this mark. The facts stay in the report."},
        {"id": "smaller_solid", "implemented": True,
         "text": "Use options.radius_scale below 1: a smaller solid region."},
        {"id": "volume_fraction", "implemented": False,
         "text": "Raise settings.volume_fraction. This adds material everywhere, not at the mark."},
        {"id": "design_space", "implemented": False,
         "text": "If the mark lies outside the design space, no voxel can hold material there. Change the "
                 "design STL (a bigger design space) and run again."},
    ],
    "smooth": [
        {"id": "none", "implemented": True,
         "text": "The YAML format of this version has no key for a smoothing at one place. The mark stays a fact."},
        {"id": "filter_radius", "implemented": False,
         "text": "settings.filter_radius is global: a larger value gives thicker, smoother members "
                 "everywhere. It is not applied by this tool."},
        {"id": "mesh_edit", "implemented": False,
         "text": "A local smoothing of final.stl is a mesh edit. This tool edits no mesh."},
    ],
}


def mark_options(run: dict | None) -> dict:
    """The option catalog for one run: `MARK_OPTIONS`, and for a run whose YAML has `settings.target_mass_g` (I7: `derived.target_mass_asked_g` in run.json)
    the option `volume_fraction` names that key. Such a YAML has no `settings.volume_fraction`, and an edit that adds it is refused (both keys).
    The option id stays; only the text changes."""
    derived = run.get("derived") if isinstance(run, dict) else None
    if not isinstance(derived, dict) or derived.get("target_mass_asked_g") is None:
        return MARK_OPTIONS
    named = "settings.target_mass_g (the asked mass; the YAML of this run has no settings.volume_fraction)"
    return {mtype: [dict(o, text=o["text"].replace("settings.volume_fraction", named)) if o["id"] == "volume_fraction" else o for o in items]
            for mtype, items in MARK_OPTIONS.items()}


VALUES_USED = {"remove": ["points", "radius_mm", "mirror"], "add": ["points", "radius_mm", "mirror"],
               "smooth": []}
VALUES_NOT_USED = ["normals", "amount_mm", "strength", "note", "smooth_mode", "remove_mode", "cleanup"]


# ---- small helpers ----------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")


def _epoch(text) -> float | None:
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _num(x):
    """A finite float, or None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def _jsonable(obj):
    """Plain Python data (numpy numbers and arrays become numbers and lists)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _load_json(path: Path, problems: list, label: str, *, expected: bool = True):
    """The JSON object in a file, or None. A missing or bad file is a fact in `problems`."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        if expected:
            problems.append(f"{label} is missing")
        return None
    except OSError as exc:
        problems.append(f"{label} cannot be read: {exc}")
        return None
    try:
        data = json.loads(text)
    except ValueError as exc:
        problems.append(f"{label} is not valid JSON ({exc}); a file that is still being written gives this")
        return None
    if not isinstance(data, dict):
        problems.append(f"{label} is not a JSON object")
        return None
    return data


def _history_rows(path: Path, problems: list) -> list[dict]:
    """The rows of history.jsonl. A line that cannot be read is skipped and named as a fact."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        problems.append("history.jsonl is missing")
        return []
    except OSError as exc:
        problems.append(f"history.jsonl cannot be read: {exc}")
        return []
    rows, bad = [], []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad.append(n)
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            bad.append(n)
    if bad:
        where = "the last line" if bad == [len(lines)] else f"lines {bad[:5]}"
        problems.append(f"history.jsonl: {where} cannot be read (skipped; a partial write gives this)")
    return rows


def _is_run_dir(d: Path) -> bool:
    return (d / "run.json").is_file() or (d / "status.json").is_file()


def _scan_runs(root: Path, depth: int, max_folders: int) -> tuple[list[Path], bool]:
    """(run folders, stopped). `stopped` is true when the scan looked at `max_folders` folders and stopped."""
    if _is_run_dir(root):
        return [root], False
    found: list[Path] = []
    looked = 0
    stopped = False

    def walk(d: Path, level: int) -> None:
        nonlocal looked, stopped
        if level > depth or stopped:
            return
        try:
            children = sorted(c for c in d.iterdir()
                              if c.is_dir() and not c.is_symlink() and not c.name.startswith("."))
        except OSError:
            return
        for c in children:
            looked += 1
            if looked > max_folders:
                stopped = True
                return
            try:
                if _is_run_dir(c):
                    found.append(c)
                else:
                    walk(c, level + 1)
            except OSError:
                continue
            if stopped:
                return

    walk(root, 1)
    return found, stopped


def find_runs(root: Path, depth: int = 2, max_folders: int = HISTORY_MAX_FOLDERS) -> list[Path]:
    """Run folders at `root` or under it, up to `depth` levels. A run folder holds run.json or status.json.

    Symbolic links and folders that start with a dot are not followed. A run folder is not searched inside.
    The scan stops after `max_folders` folders.
    """
    return _scan_runs(root, depth, max_folders)[0]


# ---- the YAML of a run, on disk ---------------------------------------------------------------------
def _yaml_doc(path: Path):
    """(doc, problem) of a YAML file. `doc` is a dict or None."""
    try:
        doc = yaml.safe_load(path.read_text())
    except OSError as exc:
        return None, f"the YAML cannot be read: {exc}"
    except yaml.YAMLError as exc:
        return None, f"the YAML has a syntax error: {str(exc).splitlines()[0] if str(exc) else exc}"
    if not isinstance(doc, dict):
        return None, "the YAML is not a mapping"
    return doc, None


def _region_files(yaml_path: Path, doc: dict | None, *, with_hash: bool) -> list[dict]:
    """The region files that a YAML names, resolved as the loader does (relative to the YAML folder).

    Each item: key (role and name), role, file (as written), path, exists, size, mtime, sha256.
    The key is `design`, `fixed:<file name>`, `keep_out:<file name>` or `load:<region name>`; a list under
    another key gives `<key>:<file name>` and a mapping `<key>:<region name>`.
    """
    regions = (doc or {}).get("regions")
    if not isinstance(regions, dict):
        return []
    base = yaml_path.parent
    items: list[tuple[str, str, object]] = []
    for role, val in regions.items():       # every key under `regions`, so a new key is not missed
        if isinstance(val, str):
            items.append((str(role), str(role), val))
        elif isinstance(val, list):
            files = [v["file"] if isinstance(v, dict) and isinstance(v.get("file"), str) else v for v in val]   # {file, axes}
            items += [(str(role), f"{role}:{Path(str(v)).name}", v) for v in files if isinstance(v, str)]
        elif isinstance(val, dict):
            items += [(str(role), f"{role}:{k}", v) for k, v in val.items() if isinstance(v, str)]
    out, seen = [], {}
    for role, key, file in items:
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key = f"{key}#{seen[key]}"
        p = Path(file)
        p = (p if p.is_absolute() else base / p).resolve()
        item = {"key": key, "role": role, "file": file, "path": str(p), "exists": p.is_file(),
                "size": None, "mtime": None, "sha256": None}
        if item["exists"]:
            try:
                item["size"] = p.stat().st_size
                item["mtime"] = p.stat().st_mtime
                if with_hash:
                    item["sha256"] = _sha256(p)
            except OSError:
                pass
        out.append(item)
    return out


def _flatten(obj, prefix: str = "") -> dict:
    """Dotted keys of a nested dict. A list of plain values is one value; a list of dicts gets indices."""
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


_MISSING = object()


def _setup_view(run: dict) -> dict:
    """The flat setup values of a run.json (see RUN_JSON_SKIP)."""
    flat = _flatten({k: v for k, v in run.items() if k not in RUN_JSON_SKIP})
    der = run.get("derived") if isinstance(run.get("derived"), dict) else {}
    for k in RUN_JSON_DERIVED_KEPT:
        node = der
        for part in k.split("."):
            node = node.get(part, _MISSING) if isinstance(node, dict) else _MISSING
        if node is not _MISSING:
            flat[f"derived.{k}"] = node
    for k, v in der.items():            # every filter fact that run.json holds (see FILTER_PREFIX)
        if str(k).startswith(FILTER_PREFIX):
            flat.update(_flatten(_drop_keys(v, FILTER_SKIP), f"derived.{k}"))
    return flat


def _drop_keys(obj, skip):
    """A copy of nested dicts without the keys in `skip` (at any depth)."""
    if isinstance(obj, dict):
        return {k: _drop_keys(v, skip) for k, v in obj.items() if k not in skip}
    return obj


def _filter_view(metrics: dict | None) -> dict:
    """The flat filter facts of a metrics dict: every top-level key that starts with FILTER_PREFIX."""
    flat: dict = {}
    for k, v in (metrics or {}).items():
        if str(k).startswith(FILTER_PREFIX):
            flat.update(_flatten(_drop_keys(v, FILTER_SKIP), str(k)))
    return flat


def _diff_flat(fa: dict, fb: dict, skip=()) -> list[dict]:
    """The keys with another value in a and b. A key that exists on one side only has `missing_in`."""
    out = []
    keys = list(fa) + [k for k in fb if k not in fa]
    for k in keys:
        if any(k == s or k.startswith(s + ".") for s in skip):
            continue
        va, vb = fa.get(k, _MISSING), fb.get(k, _MISSING)
        if va == vb:
            continue
        item = {"key": k, "a": None if va is _MISSING else va, "b": None if vb is _MISSING else vb}
        if va is _MISSING:
            item["missing_in"] = "a"
        elif vb is _MISSING:
            item["missing_in"] = "b"
        out.append(item)
    return out


# ---- facts of one run -------------------------------------------------------------------------------
# Keys that a force entry of run.json can have besides `region` and `force` (CONTRACTS.md section 26.2): a moment, a remote
# force and a pressure. A run without them gives the same dict as before.
EXTENDED_FORCE_KEYS = ("moment", "at", "pressure_mpa")


def _force_view(f: dict) -> dict:
    """A force entry of run.json for the comparison: `region`, `force`, and `kind` and the keys above when run.json has them."""
    out = {"region": f.get("region"), "force": f.get("force")}
    if f.get("kind", "force") != "force":
        out["kind"] = f["kind"]
    out.update({k: f[k] for k in EXTENDED_FORCE_KEYS if k in f})
    return out


def _settings(run: dict | None) -> dict | None:
    if not run:
        return None
    der = run.get("derived") if isinstance(run.get("derived"), dict) else {}
    cases = []
    for c in run.get("load_cases") or []:
        if isinstance(c, dict):
            cases.append({"name": c.get("name"), "weight": c.get("weight"),
                          "forces": [_force_view(f) for f in c.get("forces") or [] if isinstance(f, dict)]})
    out = {"voxel_size_mm": run.get("h"), "volume_fraction": run.get("volume_fraction"),
           "iterations": run.get("iterations"), "solver": run.get("solver"),
           "filter_radius_mm": der.get("filter_radius_mm"), "case_combine": der.get("case_combine"),
           "density_g_cm3": der.get("density_g_cm3"), "grid_shape": run.get("shape"),
           "n_active": run.get("n_active"), "n_free_dofs": run.get("n_free_dofs"),
           "objective": run.get("objective"), "load_cases": cases, "regions": run.get("regions")}
    sup = run.get("supports")
    if isinstance(sup, dict) and sup.get("axes"):     # a per-axis support (section 26.2): the axes of each fixed region
        out["supports_axes"] = sup["axes"]
    sp = run.get("springs")
    if isinstance(sp, dict) and sp.get("regions"):    # springs (section 35): the stiffness of each spring region
        out["springs"] = [{"region": r.get("region"), "stiffness_n_per_mm": r.get("stiffness_n_per_mm")}
                          for r in sp["regions"] if isinstance(r, dict)]
    jt = run.get("joints")
    if isinstance(jt, dict) and jt.get("joints"):     # joints (section 38): the far point and the section of each joint
        out["joints"] = [{"region": r.get("region"), "to_mm": r.get("to_mm"), "section": r.get("section")}
                         for r in jt["joints"] if isinstance(r, dict)]
    return out


# The one rule for the `yaml_path` of run.json lives in `runinfo.run_yaml_path`. This name stays for `apply_marks` and `tools/check_apply.py`.
_run_yaml_path = run_yaml_path


def _shown(path) -> str | None:
    """A path for a message: without `..` (text only; the file is read through the path that the resolver gave)."""
    return None if path is None else os.path.normpath(path)


def _yaml_state(run: dict | None, started: float | None, run_dir: Path | None = None) -> dict:
    """Facts about the YAML that run.json names: is it there, and is it newer than the run?"""
    p = _run_yaml_path(run, run_dir)
    path = os.path.normpath(p) if p is not None else None   # the shown text has no `..`; the file is read through `p`
    out = {"path": path, "exists": False, "mtime": None, "newer_than_run": None, "newer_region_files": []}
    if p is None:
        return out
    out["exists"] = p.is_file()
    mt = _mtime(p)
    if mt is not None:
        out["mtime"] = _iso(mt)
        if started is not None:
            out["newer_than_run"] = mt > started + NEWER_TOLERANCE_S
    if out["exists"] and started is not None:
        doc, _ = _yaml_doc(p)
        out["newer_region_files"] = [
            it["file"] for it in _region_files(p, doc, with_hash=False)
            if it["mtime"] is not None and it["mtime"] > started + NEWER_TOLERANCE_S]
    return out


def _run_mtime(d: Path) -> float:
    """The latest change of the files that tell the state of a run (one stat each)."""
    mtimes = [m for m in (_mtime(d / n) for n in ("status.json", "run.json", "history.jsonl",
                                                  "result/metrics.json")) if m is not None]
    return max(mtimes) if mtimes else (_mtime(d) or 0.0)


def run_facts(run_dir) -> dict:
    """Facts of one run folder, read from the files on disk. A missing or partial file is a fact in `problems`."""
    d = Path(run_dir)
    problems: list[str] = []
    run = _load_json(d / "run.json", problems, "run.json")
    status_exists = (d / "status.json").exists()
    status = _load_json(d / "status.json", problems, "status.json")
    rows = _history_rows(d / "history.jsonl", problems)
    state = (status or {}).get("state") or ("unreadable" if status_exists else "unknown")
    done_or_unknown = state in ("done", "unknown", "unreadable")
    metrics_raw = _load_json(d / "result" / "metrics.json", problems, "result/metrics.json",
                             expected=done_or_unknown)
    stl = d / "result" / "final.stl"
    if done_or_unknown and not stl.is_file():
        problems.append("result/final.stl is missing")
    started = _epoch((run or {}).get("started_at")) or _mtime(d / "run.json")
    mt = _run_mtime(d)
    snap_dir = d / "snapshots"
    try:
        n_snap = sum(1 for _ in snap_dir.glob("it_*.bin"))
    except OSError:
        n_snap = 0
    files = {n: (d / n).is_file() for n in RUN_FILES}
    files["snapshots"] = n_snap
    files.update({f"result/{n}": (d / "result" / n).is_file() for n in RESULT_FILES})
    last = rows[-1] if rows else None
    metrics, source = None, None
    if metrics_raw is not None:
        metrics = {k: metrics_raw.get(k) for k in METRIC_KEYS + ("stiffness_cases", "solver", "watertight")
                   if k in metrics_raw}
        metrics.update({k: v for k, v in metrics_raw.items() if str(k).startswith(FILTER_PREFIX)})
        source = "result/metrics.json"
    elif last is not None:
        metrics = {"compliance_last": last.get("compliance")}
        source = "history.jsonl (last row; result/metrics.json is missing)"
    if metrics is not None and last is not None and _num(last.get("volume_fraction")) is not None:
        metrics["volume_fraction_final"] = last["volume_fraction"]
    return {
        "name": (run or {}).get("name"), "dir_name": d.name, "run_dir": str(d), "state": state,
        "iteration": (status or {}).get("iteration"), "iterations": (run or {}).get("iterations"),
        "step": (status or {}).get("step"), "message": (status or {}).get("message") or None,
        "started_at": (run or {}).get("started_at"), "modified": _iso(mt), "mtime": mt,
        "yaml": _yaml_state(run, started, d), "settings": _settings(run),
        "metrics": metrics, "metrics_source": source,
        "last_history": None if last is None else {
            k: last.get(k) for k in ("iteration", "elapsed_s", "compliance", "volume_fraction", "change")},
        "files": files, "problems": problems,
    }


COMPACT_METRICS = ("compliance_initial", "compliance_final", "mass_design_space_g", "mass_final_g", "mass_saved_pct",
                   "volume_final_mm3", "volume_fraction_final", "run_time_s", "compliance_last")


def _compact(f: dict) -> dict:
    """The short view of `run_facts` that `run_history` gives by default."""
    st = f.get("settings") or {}
    keep = ("voxel_size_mm", "volume_fraction", "iterations", "solver", "filter_radius_mm", "n_active")
    metrics = f.get("metrics") or {}
    return {
        "name": f.get("name"), "dir_name": f.get("dir_name"), "run_dir": f.get("run_dir"), "state": f.get("state"),
        "iteration": f.get("iteration"), "iterations": f.get("iterations"), "message": f.get("message"),
        "started_at": f.get("started_at"), "modified": f.get("modified"), "mtime": f.get("mtime"),
        "yaml_path": (f.get("yaml") or {}).get("path"), "yaml_exists": (f.get("yaml") or {}).get("exists"),
        "yaml_newer_than_run": (f.get("yaml") or {}).get("newer_than_run"),
        "settings": None if not st else dict({k: st.get(k) for k in keep},
                                             load_cases=[c.get("name") for c in st.get("load_cases") or []]),
        "metrics": None if f.get("metrics") is None else {k: metrics[k] for k in COMPACT_METRICS if k in metrics},
        "metrics_source": f.get("metrics_source"), "problems": f.get("problems", []),
    }


# The keys of a row that come from run.json, `result/` and `history.jsonl`: after a refused job over an earlier run they are of the earlier run
# (P9-19, `earlier_run.mark_row`). The other keys come from `status.json`, the job files and the folder.
HISTORY_EARLIER_KEYS = ("name", "iterations", "started_at", "yaml_path", "yaml_exists", "yaml_newer_than_run", "settings", "metrics",
                        "metrics_source")
HISTORY_EARLIER_KEYS_DETAIL = ("name", "iterations", "started_at", "yaml", "settings", "metrics", "metrics_source", "last_history", "files")
COMPARE_EARLIER_KEYS = ("name", "iterations", "started_at", "yaml_path")


def _refused(f: dict, d: Path) -> bool:
    """True if the folder `d`, with the facts `f` of `run_facts`, holds a refused job over an earlier run (`earlier_run.refused_or_false`, P9-19)."""
    return earlier_run.refused_or_false(d, {k: f.get(k) for k in ("state", "iteration", "step", "message")})


def run_history(root, depth: int = 2, limit: int | None = HISTORY_DEFAULT_LIMIT, detail: bool = False) -> dict:
    """Facts of the runs at `root` or under it (see `find_runs`), newest first. It writes nothing.

    Each run: state, iteration, the YAML of the run (is it there, is it newer than the run), the key settings,
    the main metrics (compliance, volume, mass, run time) and `problems` (a missing or partial file as a fact).
    `detail` gives the full facts of each run (all settings, all metrics, the files that exist).
    `limit`: the newest runs only (default 20). `count` is the number of runs found; a note says when runs are
    left out. `limit` 0 or None gives all runs. `depth` is at most 5; the scan stops after 5000 folders (a note).
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {"ok": False, "error": f"folder not found: {root}"}
    notes: list[str] = []
    try:
        want = int(depth)
    except (TypeError, ValueError):
        want = 2
    use_depth = min(max(1, want), HISTORY_MAX_DEPTH)
    if want != use_depth:
        notes.append(f"depth {depth} is not used: depth is between 1 and {HISTORY_MAX_DEPTH}; {use_depth} was used.")
    found, stopped = _scan_runs(root, use_depth, HISTORY_MAX_FOLDERS)
    if stopped:
        notes.append(f"The scan stopped after {HISTORY_MAX_FOLDERS} folders: runs below them are not listed. "
                     "Give a root that is closer to the runs.")
    total = len(found)
    found.sort(key=_run_mtime, reverse=True)          # one stat each; the facts are read for the newest only
    shown = found[:int(limit)] if limit is not None and int(limit) > 0 else found
    if len(shown) < total:
        notes.append(f"{total - len(shown)} older runs are left out (limit {int(limit)}; count is the number found). "
                     "limit 0 lists all runs.")
    runs, refused = [], set()
    for d in shown:
        try:
            runs.append(run_facts(d))
            if _refused(runs[-1], d):
                refused.add(runs[-1]["run_dir"])
        except Exception as exc:  # one bad folder must not hide the others
            runs.append({"name": None, "dir_name": d.name, "run_dir": str(d), "state": "unreadable",
                         "mtime": 0.0, "modified": None, "yaml": {}, "settings": None, "metrics": None,
                         "problems": [f"cannot read the run folder: {type(exc).__name__}: {exc}"]})
    runs.sort(key=lambda r: r.get("mtime") or 0.0, reverse=True)
    if not detail:
        runs = [_compact(r) for r in runs]
    keys = HISTORY_EARLIER_KEYS_DETAIL if detail else HISTORY_EARLIER_KEYS
    runs = [earlier_run.mark_row(r, keys) if r.get("run_dir") in refused else r for r in runs]   # P9-19
    return {"ok": True, "root": str(root), "depth": use_depth, "count": total, "listed": len(runs),
            "detail": bool(detail), "runs": runs,
            "notes": notes + ["The numbers come from the files on disk (run.json, status.json, history.jsonl, "
                              "result/metrics.json). A run that is still going has no metrics.json.",
                              "yaml_newer_than_run: the YAML file has a later modification time than started_at "
                              f"(more than {NEWER_TOLERANCE_S:g} s). Then the YAML on disk may not be the one that ran."]}


# ---- compare ----------------------------------------------------------------------------------------
def _read_labels(d: Path, run: dict | None):
    """(labels array, problem). labels.bin holds the voxel classes of the run (uint8, shape of run.json)."""
    if not run or not run.get("shape"):
        return None, "run.json has no grid shape"
    try:
        shape = tuple(int(s) for s in run["shape"])
        raw = (d / "labels.bin").read_bytes()
    except FileNotFoundError:
        return None, "labels.bin is missing"
    except (OSError, ValueError, TypeError) as exc:
        return None, f"labels.bin cannot be read: {exc}"
    if len(raw) != int(np.prod(shape)):
        return None, f"labels.bin has {len(raw)} bytes, the grid {list(shape)} needs {int(np.prod(shape))}"
    return np.frombuffer(raw, np.uint8).reshape(shape), None


def _grid_of(run: dict | None):
    """(shape, h, origin) of a run.json, or None when a key is missing or wrong."""
    try:
        return ([int(v) for v in run["shape"]], float(run["h"]), [float(v) for v in run["origin"]])
    except (KeyError, TypeError, ValueError):
        return None


def _same_grid(run_a: dict, run_b: dict) -> bool:
    try:
        return (list(run_a["shape"]) == list(run_b["shape"])
                and abs(float(run_a["h"]) - float(run_b["h"])) < 1e-9
                and np.allclose(run_a["origin"], run_b["origin"], atol=1e-6))
    except (KeyError, TypeError, ValueError):
        return False


def _label_changes(la: np.ndarray, lb: np.ndarray, origin, h: float) -> dict:
    diff = la != lb
    n = int(diff.sum())
    out = {"voxels": n, "transitions": {}, "bbox_mm": None}
    if n:
        pair = la[diff].astype(np.int64) * 8 + lb[diff].astype(np.int64)
        codes, counts = np.unique(pair, return_counts=True)
        for c, k in zip(codes.tolist(), counts.tolist()):
            out["transitions"][f"{LABEL_NAMES.get(c // 8, c // 8)}->{LABEL_NAMES.get(c % 8, c % 8)}"] = int(k)
        idx = np.argwhere(diff)
        o = np.asarray(origin, float)
        out["bbox_mm"] = {"min": [round(float(v), 3) for v in o + h * idx.min(axis=0)],
                          "max": [round(float(v), 3) for v in o + h * (idx.max(axis=0) + 1)]}
    return out


def _floats(v):
    """A list of floats, or the value itself if it is not a list of numbers (so that two such values can still be compared)."""
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return v


def _yaml_axes(fixed):
    """The axes that the `regions.fixed` of a YAML fix, as run.json `supports.axes` has them ({file name: "yz"}).

    None when no entry is a `{file, axes}` mapping (every entry is a full clamp, and run.json has no `supports`).
    """
    if not isinstance(fixed, list) or not any(isinstance(x, dict) for x in fixed):
        return None
    out = {}
    for x in fixed:
        name = x.get("file") if isinstance(x, dict) else x
        axes = x.get("axes") if isinstance(x, dict) else list("xyz")
        if isinstance(name, str) and isinstance(axes, list):
            out[Path(name).name] = "".join(a for a in "xyz" if a in {str(v).lower() for v in axes})
    return out


def _moment_diffs(i: int, yfs: list, rfs: list, add) -> None:
    """The `at` and the moment of each force entry: the YAML against run.json (CONTRACTS.md section 26.2).

    run.json holds the moment about the node centroid of the region (`moment_about_mm`), with the term of a remote
    force added: moment = M + (at - centroid) x F. The YAML value is brought to that form before the comparison.
    """
    def same(a, b):
        return len(a) == len(b) and all(abs(x - y) <= 1e-9 * max(1.0, abs(y)) for x, y in zip(a, b))

    for j, (y, r) in enumerate(zip(yfs, rfs)):
        at_y, at_r = _floats(y["at"]) if "at" in y else None, _floats(r["at"]) if "at" in r else None
        if at_y != at_r:
            add(f"load_cases[{i}].forces[{j}].at", at_y, at_r)
        has_y, has_r = ("moment" in y or "at" in y), "moment" in r
        if has_y != has_r:
            add(f"load_cases[{i}].forces[{j}].moment", y.get("moment"), r.get("moment"))
        elif has_y:
            m = _floats(y["moment"]) if "moment" in y else [0.0, 0.0, 0.0]
            f = _floats(y["force"]) if "force" in y else [0.0, 0.0, 0.0]
            c = _floats(r.get("moment_about_mm"))
            try:
                if at_y is not None:     # M + (at - c) x F
                    d = [a - b for a, b in zip(at_y, c)]
                    m = [m[0] + d[1] * f[2] - d[2] * f[1], m[1] + d[2] * f[0] - d[0] * f[2],
                         m[2] + d[0] * f[1] - d[1] * f[0]]
                if not same(m, _floats(r["moment"])):
                    add(f"load_cases[{i}].forces[{j}].moment", y.get("moment"), r["moment"])
            except (TypeError, ValueError, IndexError):     # a value that is not a vector: say that the YAML differs
                add(f"load_cases[{i}].forces[{j}].moment", y.get("moment"), r.get("moment"))


def _yaml_vs_run(run: dict, doc: dict) -> tuple[list[dict], list[str], list[dict]]:
    """(differences, compared, not_compared). The values in the YAML on disk that differ from what run.json recorded
    at the start, the names of the values that were compared (so the report can say what it did not compare), and
    the values that the engine derives (not compared: the YAML does not hold them)."""
    diffs: list[dict] = []
    compared: list[str] = []
    not_compared: list[dict] = []

    def add(key, yaml_val, run_val):
        diffs.append({"key": key, "yaml": yaml_val, "run_json": run_val})

    def close(a, b):
        fa, fb = _num(a), _num(b)
        return fa is not None and fb is not None and abs(fa - fb) <= 1e-9 * max(1.0, abs(fb))

    st = doc.get("settings") if isinstance(doc.get("settings"), dict) else {}
    mat = doc.get("material") if isinstance(doc.get("material"), dict) else {}
    for ykey, rkey in (("voxel_size", "h"), ("volume_fraction", "volume_fraction"), ("iterations", "iterations")):
        if rkey in run:
            if ykey == "volume_fraction" and ykey not in st and "target_mass_g" in st:   # I7: the engine derives the fraction from the mass and the voxel labels
                not_compared.append({"key": "settings.volume_fraction", "run_json_key": "volume_fraction", "run_json": run[rkey],
                                     "reason": "derived by the engine from settings.target_mass_g and the voxel labels; not compared"})
                continue
            compared.append(f"settings.{ykey}")
            if not close(st.get(ykey), run[rkey]):
                add(f"settings.{ykey}", st.get(ykey), run[rkey])
    asked = (run.get("derived") or {}).get("target_mass_asked_g") if isinstance(run.get("derived"), dict) else None
    if "target_mass_g" in st or asked is not None:   # I7: the mass that the user asked for (run.json holds it in derived)
        compared.append("settings.target_mass_g")
        if not close(st.get("target_mass_g"), asked):
            add("settings.target_mass_g", st.get("target_mass_g"), asked)
    if "solver" in run and st.get("solver", "auto") != "auto":
        compared.append("settings.solver")
        if st.get("solver") != run["solver"]:
            add("settings.solver", st.get("solver"), run["solver"])
    if "name" in run:
        compared.append("name")
        if doc.get("name") != run["name"]:
            add("name", doc.get("name"), run["name"])
    # run.json keeps these in its `derived` block: the YAML value, or the default of the loader when the key is missing
    der = run.get("derived") if isinstance(run.get("derived"), dict) else {}
    # The filter radius is compared only when the YAML sets it. Without it the engine derives the radius (1.5 x
    # voxel_size, or from settings.min_member_mm): the rule is not repeated here, so a change of the rule is no
    # false difference. The filter facts of the two runs are compared by `_setup_view` (derived.filter*).
    if "filter_radius_mm" in der and "filter_radius" not in st:
        not_compared.append({"key": "settings.filter_radius", "run_json_key": "derived.filter_radius_mm",
                             "run_json": der["filter_radius_mm"], "reason": "derived by the engine; not compared"})
    for ykey, dkey, default in (("settings.filter_radius", "filter_radius_mm", None),
                                ("settings.case_combine", "case_combine", "sum"),
                                ("material.density", "density_g_cm3", 1.24)):
        if dkey not in der:
            continue
        if ykey == "settings.filter_radius" and "filter_radius" not in st:
            continue
        part, name = ykey.split(".")
        yval = (st if part == "settings" else mat).get(name, default)
        compared.append(ykey)
        same = close(yval, der[dkey]) if isinstance(der[dkey], (int, float)) else yval == der[dkey]
        if not same:
            add(ykey, yval, der[dkey])
    lcs = [c for c in doc.get("load_cases") or [] if isinstance(c, dict)] if isinstance(doc.get("load_cases"), list) else []
    rcs = [c for c in run.get("load_cases") or [] if isinstance(c, dict)] if isinstance(run.get("load_cases"), list) else []
    compared += ["load_cases (names, weights, forces, moments, pressures)"]
    if [c.get("name") for c in lcs] != [c.get("name") for c in rcs]:
        add("load_cases.names", [c.get("name") for c in lcs], [c.get("name") for c in rcs])
    else:
        for i, (yc, rc) in enumerate(zip(lcs, rcs)):
            if not close(yc.get("weight", 1.0), rc.get("weight")):
                add(f"load_cases[{i}].weight", yc.get("weight", 1.0), rc.get("weight"))
            yfs = [f for f in yc.get("forces") or [] if isinstance(f, dict)]
            rfs = [f for f in rc.get("forces") or [] if isinstance(f, dict) and f.get("kind") != "pressure"]
            # `force` is optional in the YAML when `moment` is given: run.json holds the resultant, 0 0 0
            yf = [(f.get("region"), _floats(f["force"]) if "force" in f else [0.0, 0.0, 0.0]) for f in yfs]
            rf = [(f.get("region"), _floats(f.get("force"))) for f in rfs]
            if yf != rf:
                add(f"load_cases[{i}].forces", yf, rf)
            else:
                _moment_diffs(i, yfs, rfs, add)
            yp = [(p.get("region"), _floats([p.get("value_mpa")])) for p in yc.get("pressures") or [] if isinstance(p, dict)]
            rp = [(f.get("region"), _floats([f.get("pressure_mpa")])) for f in rc.get("forces") or []
                  if isinstance(f, dict) and f.get("kind") == "pressure"]
            if yp != rp:
                add(f"load_cases[{i}].pressures", yp, rp)
    rg = doc.get("regions") if isinstance(doc.get("regions"), dict) else {}
    rr = run.get("regions") if isinstance(run.get("regions"), dict) else {}

    def file_names(v):
        if isinstance(v, str):
            return Path(v).name
        if isinstance(v, list):    # an entry can be {file, axes} (section 26.2)
            return [Path(str(x["file"] if isinstance(x, dict) and "file" in x else x)).name for x in v]
        if isinstance(v, dict):
            return {k: Path(str(x)).name for k, x in v.items()}
        return v
    for k in rr:                          # every region key that run.json recorded
        compared.append(f"regions.{k} (file names)")
        empty = [] if isinstance(rr[k], list) else ({} if isinstance(rr[k], dict) else None)
        names = file_names(rg.get(k)) if k in rg else empty
        if names != rr[k]:
            add(f"regions.{k}", names, rr[k])
    ya = _yaml_axes(rg.get("fixed"))      # the axes of the supports: run.json has them in `supports.axes` (section 26.2)
    ra = (run.get("supports") or {}).get("axes") if isinstance(run.get("supports"), dict) else None
    if ya is not None or ra is not None:
        compared.append("regions.fixed (axes)")
        if ya != ra:
            add("regions.fixed (axes)", ya, ra)
    ys = [e for e in doc["springs"] if isinstance(e, dict)] if isinstance(doc.get("springs"), list) else []
    rs = [r for r in (run.get("springs") or {}).get("regions", []) if isinstance(r, dict)] if isinstance(run.get("springs"), dict) else []
    if ys or rs:                          # springs (section 35): the region and the stiffness, in the form that the YAML uses
        compared.append("springs (region, stiffness)")
        rmap = {r.get("region"): r for r in rs}
        for e in ys:
            key = "stiffness_per_area" if "stiffness_per_area" in e else "stiffness"
            rkey = "stiffness_per_area_n_per_mm3" if key == "stiffness_per_area" else "stiffness_n_per_mm"
            v = e.get(key)
            v = [v, v, v] if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            r = rmap.get(e.get("region"))
            rv = None if r is None else r.get(rkey)
            if not (isinstance(v, list) and isinstance(rv, list) and len(v) == len(rv) and all(close(a, b) for a, b in zip(v, rv))):
                add(f"springs[{e.get('region')}].{key}", v, rv)
        for name in rmap:
            if name not in {e.get("region") for e in ys}:
                add(f"springs[{name}]", None, rmap[name].get("stiffness_n_per_mm"))
    yj = [e for e in doc["joints"] if isinstance(e, dict)] if isinstance(doc.get("joints"), list) else []
    rj = [r for r in (run.get("joints") or {}).get("joints", []) if isinstance(r, dict)] if isinstance(run.get("joints"), dict) else []
    if yj or rj:                          # joints (section 38): the region, the far point and the section
        compared.append("joints (region, to, section)")
        jmap = {r.get("region"): r for r in rj}
        sec = lambda d: {k: (float(v) if _num(v) is not None else v) for k, v in d.items()} if isinstance(d, dict) else d
        for e in yj:
            r = jmap.get(e.get("region"))
            if r is None or _floats(e.get("to")) != _floats(r.get("to_mm")):
                add(f"joints[{e.get('region')}].to", e.get("to"), None if r is None else r.get("to_mm"))
            if r is not None and sec(e.get("section")) != sec(r.get("section")):
                add(f"joints[{e.get('region')}].section", e.get("section"), r.get("section"))
        for name in jmap:
            if name not in {e.get("region") for e in yj}:
                add(f"joints[{name}]", None, jmap[name].get("to_mm"))
    return diffs, compared, not_compared


def _labels_vs_disk(d: Path, run: dict | None, yaml_path: Path | None, mode) -> dict:
    """Do the YAML and STL files on disk voxelize to the labels that the run recorded (labels.bin)?"""
    if mode is False:
        return {"checked": False, "reason": "not requested (check_labels=false)"}
    grid = _grid_of(run)
    if grid is None:
        return {"checked": False, "reason": "run.json has no grid (shape, h, origin)"}
    n = int(np.prod(grid[0]))
    if mode is None and n > LABELS_CHECK_AUTO_MAX:
        return {"checked": False, "reason": f"the grid has {n} voxels; set check_labels=true to voxelize it again "
                                            "(a rough cost: about the setup time of the run)"}
    if yaml_path is None or not yaml_path.is_file():
        return {"checked": False, "reason": "the YAML of the run is not on disk"}
    run_labels, problem = _read_labels(d, run)
    if run_labels is None:
        return {"checked": False, "reason": problem}
    from .voxelize import voxelize
    try:
        origin, h, shape, labels, _ids = voxelize(load_config(yaml_path))
    except UserError as exc:
        return {"checked": False, "reason": f"the YAML on disk does not load: {exc}"}
    grid_same = (list(shape) == grid[0] and abs(h - grid[1]) < 1e-9 and bool(np.allclose(origin, grid[2], atol=1e-6)))
    if not grid_same:
        return {"checked": True, "grid_same": False, "labels_same": False,
                "disk_grid": {"shape": list(shape), "h": h, "origin": [float(v) for v in origin]},
                "run_grid": {"shape": grid[0], "h": grid[1], "origin": grid[2]}}
    nd = int((labels != run_labels).sum())
    return {"checked": True, "grid_same": True, "labels_same": nd == 0, "voxels_differ": nd}


def _inputs_check(d: Path, run: dict | None, check_labels) -> dict:
    """Facts about the YAML and STL files on disk: can they be the files that ran? (run.json has no hash.)

    `facts` has every sentence. `warnings` has the ones that say the files on disk may not be the ones that ran.
    """
    facts: list[str] = []
    warnings: list[str] = []
    out: dict = {"yaml": None, "yaml_vs_run_json": None, "labels_vs_disk": None, "facts": facts,
                 "warnings": warnings, "line": None}

    def warn(text: str) -> None:
        facts.append(text)
        warnings.append(text)

    if run is None:
        warn("run.json is missing or unreadable: the YAML of the run is not known.")
        out["line"] = "run.json is missing or unreadable: the YAML of the run is not known"
        return out
    started = _epoch(run.get("started_at")) or _mtime(d / "run.json")
    out["yaml"] = ys = _yaml_state(run, started, d)
    ypath = _run_yaml_path(run, d)
    if not ys["exists"]:
        warn(f"The YAML file of the run is not on disk: {ys['path']}.")
    else:
        if ys["newer_than_run"]:
            warn(f"The YAML file is newer than the run (modified {ys['mtime']}, run started "
                 f"{run.get('started_at')}): the YAML on disk may not be the one that ran.")
        if ys["newer_region_files"]:
            nf = ys["newer_region_files"]
            warn(f"{len(nf)} region file{'s' if len(nf) != 1 else ''} newer than the run: {', '.join(nf)}. "
                 "The STL on disk may not be the one that ran.")
        doc, problem = _yaml_doc(ypath)
        if problem:
            warn(f"{problem}.")
        else:
            try:
                diffs, compared, not_compared = _yaml_vs_run(run, doc)
            except Exception as exc:  # a YAML with odd values: a fact, and the labels check still tells more
                diffs, compared, not_compared = [], [], []
                facts.append(f"The values of the YAML on disk could not be compared with run.json ({exc}).")
            out["yaml_vs_run_json"] = {"differs": diffs, "compared": compared, "not_compared": not_compared,
                                       "not_recorded": list(NOT_IN_RUN_JSON)}
            for x in diffs:
                warn(f"The YAML on disk has {x['key']} = {x['yaml']!r}; run.json recorded {x['run_json']!r}: "
                     "the YAML was changed after the run, or another YAML ran.")
            if compared and not diffs:
                facts.append("Compared with run.json, equal: " + ", ".join(compared) + ".")
            for x in not_compared:
                facts.append(f"{x['key']} is not in the YAML: {x['reason']}. run.json recorded "
                             f"{x['run_json_key']} = {x['run_json']!r}.")
            facts.append("run.json does not record " + ", ".join(NOT_IN_RUN_JSON)
                         + ", so a change of them in the YAML cannot be seen here.")
    out["labels_vs_disk"] = lv = _labels_vs_disk(d, run, ypath, check_labels)
    if lv.get("checked"):
        if lv.get("labels_same"):
            facts.append("The YAML and STL files on disk voxelize to the same labels as labels.bin of the run.")
        elif lv.get("grid_same") is False:
            warn("The YAML and STL files on disk make another grid than the run recorded: "
                 "they are not the files that ran.")
        else:
            warn(f"The YAML and STL files on disk make {lv['voxels_differ']} voxel labels that differ "
                 "from labels.bin of the run: they are not the files that ran.")
    else:
        facts.append(f"The labels of the files on disk were not compared with labels.bin: {lv['reason']}.")
    facts.append("run.json stores no hash of the YAML or the STL files, so these checks are evidence, not proof.")
    out["line"] = _run_line(out)
    return out


def _run_line(chk: dict) -> str | None:
    """One sentence for a run: what says that the files on disk may not be the ones that ran, and what still matches."""
    if not chk.get("warnings"):
        return None
    ys = chk.get("yaml") or {}
    vr = chk.get("yaml_vs_run_json") or {}
    lv = chk.get("labels_vs_disk") or {}
    said, matches = [], []
    if not ys.get("exists"):
        said.append("the YAML is not on disk")
    else:
        if ys.get("newer_than_run"):
            said.append("the YAML is newer than the run")
        if ys.get("newer_region_files"):
            said.append(f"{len(ys['newer_region_files'])} region file(s) are newer than the run")
    if vr.get("differs"):
        said.append("YAML values differ from run.json: " + ", ".join(x["key"] for x in vr["differs"]))
    elif vr.get("compared"):
        matches.append("the YAML values equal run.json")
    if lv.get("checked"):
        if lv.get("labels_same"):
            matches.append("the voxel labels equal labels.bin")
        elif lv.get("grid_same") is False:
            said.append("the files on disk make another grid than the run")
        else:
            said.append(f"{lv.get('voxels_differ')} voxel labels differ from labels.bin")
    if not said:
        said.append("run.json is not complete enough to compare")
    text = "; ".join(said)
    if matches:
        text += ". Still true: " + " and ".join(matches) + " (evidence, not proof: run.json has no hash)"
    return text


def _metric_rows(ma: dict | None, mb: dict | None) -> tuple[list[dict], list[dict]]:
    """Numeric metric rows (a, b, delta, delta_pct) and the non-numeric differences."""
    rows, other = [], []
    ma, mb = ma or {}, mb or {}
    keys = [k for k in METRIC_KEYS + ("volume_fraction_final", "compliance_last") if k in ma or k in mb]
    for k in keys:
        a, b = _num(ma.get(k)), _num(mb.get(k))
        row = {"key": k, "a": ma.get(k), "b": mb.get(k), "delta": None, "delta_pct": None}
        if a is not None and b is not None:
            row["delta"] = b - a
            row["delta_pct"] = round(100.0 * (b - a) / abs(a), 3) if a != 0 else None
        rows.append(row)
    sa = {c.get("name"): c.get("N_per_mm") for c in ma.get("stiffness_cases") or [] if isinstance(c, dict)}
    sb = {c.get("name"): c.get("N_per_mm") for c in mb.get("stiffness_cases") or [] if isinstance(c, dict)}
    for name in list(sa) + [n for n in sb if n not in sa]:
        a, b = _num(sa.get(name)), _num(sb.get(name))
        row = {"key": f"stiffness[{name}] N/mm", "a": sa.get(name), "b": sb.get(name), "delta": None,
               "delta_pct": None}
        if a is not None and b is not None:
            row["delta"] = b - a
            row["delta_pct"] = round(100.0 * (b - a) / abs(a), 3) if a != 0 else None
        rows.append(row)
    for k in ("solver", "watertight"):
        if k in ma or k in mb:
            if ma.get(k) != mb.get(k):
                other.append({"key": k, "a": ma.get(k), "b": mb.get(k)})
    return rows, other


def compare_runs(run_a, run_b=None, check_labels=None, variants=None) -> dict:
    """Runs side by side. It writes nothing.

    Two-run form (`run_a`, `run_b`): the setup diff and the metrics diff. See `_compare_runs`.
    Table form (`run_a`, `variants`): one row per run and the setup diff of each variant against the base `run_a`. See `_variants_table`.
    """
    try:
        if run_b is None and variants is None:
            return {"ok": False, "error": _FORMS_ERROR % "compare_runs got neither run_b nor variants"}
        if run_b is not None and variants is not None:
            return {"ok": False, "error": _FORMS_ERROR % "compare_runs got run_b and variants together"}
        if not _named(run_a) or (run_b is not None and not _named(run_b)):
            return {"ok": False, "error": _FORMS_ERROR % "run_a and run_b must be run folders (a non-empty text)"}
        if variants is not None:
            return _variants_table(run_a, variants, check_labels)
        return _compare_runs(run_a, run_b, check_labels)
    except UserError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # a fact about this tool, not a crash of the caller
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


_FORMS_ERROR = ("%s. compare_runs did nothing. It has two forms: (run_a, run_b) puts two runs side by side; "
                "(run_a, variants=[run folders]) gives a table of the base run_a and the variants. Give one of them.")


def _compare_runs(run_a, run_b, check_labels=None) -> dict:
    """Two runs side by side: the setup diff and the metrics diff. It writes nothing.

    Run A is the base, run B the change; `delta` is B - A. The source of each value is named:
    `run.json` (recorded when the run started), `labels.bin` (the voxel classes of the run), the YAML and STL
    files on disk (read now), `result/metrics.json`. run.json holds the YAML path only, so the YAML
    and STL files on disk can be newer than the run or can differ from it: `inputs` says so as facts.
    check_labels: None = voxelize the YAML on disk and compare with labels.bin when the grid has at most
    10 million voxels; True = always; False = never.
    """
    a, b = Path(run_a).expanduser().resolve(), Path(run_b).expanduser().resolve()
    for d in (a, b):
        if not d.is_dir():
            return {"ok": False, "error": f"run folder not found: {d}"}
    fa, fb = run_facts(a), run_facts(b)
    refused = {"A": _refused(fa, a), "B": _refused(fb, b)}   # P9-19: the folder of a refused job over an earlier run
    pa, pb = [], []
    ra = _load_json(a / "run.json", pa, "run.json")
    rb = _load_json(b / "run.json", pb, "run.json")
    notes: list[str] = []
    # -- setup from run.json
    run_json_diff = None
    if ra is not None and rb is not None:
        run_json_diff = [dict(x, source="run.json of each run (recorded at the start)")
                         for x in _diff_flat(_setup_view(ra), _setup_view(rb))]
    else:
        notes.append("run.json is missing or unreadable in " + " and ".join(
            n for n, r in (("A", ra), ("B", rb)) if r is None) + ": the setup diff from run.json is not available.")
    # -- YAML and STL files on disk
    ya, yb = _run_yaml_path(ra, a), _run_yaml_path(rb, b)
    doc_a, doc_b = (_yaml_doc(y)[0] if y and y.is_file() else None for y in (ya, yb))
    yaml_diff = {"same_file": bool(ya and yb and ya.resolve() == yb.resolve()), "diff": None,
                 "source": "the YAML files on disk, read now (not proven to be the files that ran: see inputs)"}
    if doc_a is not None and doc_b is not None and not yaml_diff["same_file"]:
        yaml_diff["diff"] = _diff_flat(_flatten(doc_a), _flatten(doc_b))
    elif yaml_diff["same_file"]:
        notes.append("Both runs name the same YAML file. Its content on disk is one state, so no YAML diff is "
                     "made; the run.json diff shows what each run recorded.")
    missing_yaml = [n for n, y, doc in (("A", ya, doc_a), ("B", yb, doc_b)) if y is None or doc is None]
    files_a = _region_files(ya, doc_a, with_hash=True) if ya and doc_a is not None else []
    files_b = _region_files(yb, doc_b, with_hash=True) if yb and doc_b is not None else []
    ia, ib = {f["key"]: f for f in files_a}, {f["key"]: f for f in files_b}
    region_diff, same = [], 0
    if missing_yaml:   # no list of region files for one side: a diff would list every file of the other side
        notes.append("The region files are not compared: the YAML of run " + " and ".join(missing_yaml)
                     + " is missing or unreadable.")
        ia = ib = {}
    for key in list(ia) + [k for k in ib if k not in ia]:
        x, y = ia.get(key), ib.get(key)

        def brief(f):
            return None if f is None else {"file": f["file"], "sha256": f["sha256"], "size": f["size"]}
        if x is None:
            status = "only_in_b"
        elif y is None:
            status = "only_in_a"
        elif x["sha256"] is None or y["sha256"] is None:
            status = "unknown"
        elif x["sha256"] == y["sha256"]:
            status = "same"
        else:
            status = "changed"
        if status == "same":
            same += 1
        else:
            region_diff.append({"key": key, "status": status, "a": brief(x), "b": brief(y),
                                "source": "region files on disk, hashed now"})
    # -- labels of the two runs
    la, pla = _read_labels(a, ra)
    lb, plb = _read_labels(b, rb)
    if la is not None and lb is not None and ra and rb and _same_grid(ra, rb):
        labels_diff = dict(_label_changes(la, lb, ra["origin"], float(ra["h"])), comparable=True,
                           source="labels.bin of each run (recorded at the start)")
    elif la is not None and lb is not None:
        labels_diff = {"comparable": False, "reason": "the grids differ (shape, voxel size or origin)",
                       "a": {"shape": ra.get("shape"), "h": ra.get("h"), "origin": ra.get("origin")},
                       "b": {"shape": rb.get("shape"), "h": rb.get("h"), "origin": rb.get("origin")}}
    else:
        labels_diff = {"comparable": False, "reason": "; ".join(p for p in (pla and f"A: {pla}", plb and f"B: {plb}") if p)}
    # -- metrics
    rows, other = _metric_rows(fa["metrics"], fb["metrics"])
    metrics_diff = {"rows": rows, "other": other,
                    "source_a": fa["metrics_source"], "source_b": fb["metrics_source"]}
    filt_a, filt_b = _filter_view(fa["metrics"]), _filter_view(fb["metrics"])
    if filt_a or filt_b:       # the filter block of result/metrics.json, side by side (A3a: radius, solid rule, ...)
        metrics_diff["filter"] = {"a": filt_a or None, "b": filt_b or None, "differs": _diff_flat(filt_a, filt_b),
                                  "source": "result/metrics.json"}
    # -- comparability facts
    sa, sb = fa["settings"] or {}, fb["settings"] or {}
    if sa and sb:
        if sa.get("load_cases") != sb.get("load_cases"):
            notes.append("The load cases differ between the runs: compliance and stiffness belong to different loads.")
        if sa.get("supports_axes") != sb.get("supports_axes"):
            notes.append("The axes of the supports differ between the runs: a roller and a full clamp give different stiffness.")
        if sa.get("voxel_size_mm") != sb.get("voxel_size_mm"):
            notes.append(f"The voxel sizes differ ({sa.get('voxel_size_mm')} and {sb.get('voxel_size_mm')} mm): "
                         "compliance has a part that comes from the grid.")
        if sa.get("volume_fraction") != sb.get("volume_fraction"):
            notes.append(f"The volume fractions differ ({sa.get('volume_fraction')} and {sb.get('volume_fraction')}): "
                         "less material gives a higher compliance.")
        if sa.get("iterations") != sb.get("iterations"):
            notes.append(f"The iteration counts differ ({sa.get('iterations')} and {sb.get('iterations')}): "
                         "a run with fewer iterations may not have converged.")
    ta = _target_volume(ra)
    tb = _target_volume(rb)
    if ta is not None and tb is not None and abs(ta - tb) > 1e-6 * max(1.0, abs(ta)):
        notes.append(f"The target volumes differ ({ta:g} and {tb:g} mm3). The volume fraction applies to the free "
                     "design voxels, so a keep-out or solid region changes the target volume with it.")
    inputs = {"a": _inputs_check(a, ra, check_labels), "b": _inputs_check(b, rb, check_labels),
              "note": "run.json holds the YAML path and file names, no hash. The files on disk are read now."}
    # -- summary lines
    summary = []
    notes.append("run.json does not record " + ", ".join(NOT_IN_RUN_JSON) + ". A change of them between the runs is "
                 "not seen here, except where the two YAML files on disk differ (setup.yaml_diff).")
    for r in rows:
        if r["delta"] is not None and r["delta"] != 0 and (r["delta_pct"] is None or abs(r["delta_pct"]) >= 0.005):
            pct = f" ({r['delta_pct']:+.1f} %)" if r["delta_pct"] is not None else ""
            summary.append(f"{r['key']}: {r['a']} -> {r['b']}{pct}")
    if run_json_diff:
        summary.append(f"run.json differs in {len(run_json_diff)} keys: "
                       + ", ".join(x["key"] for x in run_json_diff[:6]) + (" ..." if len(run_json_diff) > 6 else ""))
    fchg = [x for x in (run_json_diff or []) if x["key"].startswith("derived." + FILTER_PREFIX)] or \
           (metrics_diff.get("filter") or {}).get("differs", [])
    if fchg:                    # the filter decides the member size, so it has its own line
        summary.append("filter differs: " + ", ".join(f"{x['key']} {x['a']!r} -> {x['b']!r}" for x in fchg[:4])
                       + (" ..." if len(fchg) > 4 else ""))
    for r in region_diff:
        summary.append(f"region file {r['key']}: {r['status']}")
    if labels_diff.get("comparable") and labels_diff["voxels"]:
        summary.append(f"{labels_diff['voxels']} voxel labels differ: "
                       + ", ".join(f"{k} {v}" for k, v in labels_diff["transitions"].items()))
    elif labels_diff.get("comparable"):
        summary.append("the voxel labels of the two runs are equal")
    for side, chk in (("A", inputs["a"]), ("B", inputs["b"])):
        if chk.get("line"):
            summary.append(f"run {side}: {chk['line']}")
    problems = [f"A: {p}" for p in fa["problems"] + pa] + [f"B: {p}" for p in fb["problems"] + pb]
    seen_p: set = set()
    problems = [p for p in problems if not (p in seen_p or seen_p.add(p))]
    for side in ("B", "A"):        # P9-19: these first, A before B
        if refused[side]:
            notes.insert(0, f"Run {side} is a folder where an input check refused a job and kept an earlier run (see {side.lower()}.earlier_run). "
                            f"The setup, the metrics and the inputs of run {side} in this reply are of that earlier run.")
            summary.insert(0, f"run {side}: the folder holds an earlier run that a refused job did not change; the setup, metrics and inputs "
                              f"of run {side} are of the earlier run")
    return {"ok": True, "a": _brief_run(fa, refused["A"]), "b": _brief_run(fb, refused["B"]),
            "setup": {"run_json_diff": run_json_diff, "yaml_diff": yaml_diff,
                      "region_files": {"diff": None if missing_yaml else region_diff,
                                       "same_count": None if missing_yaml else same},
                      "labels_diff": labels_diff},
            "metrics": metrics_diff, "inputs": inputs, "notes": notes, "summary": summary,
            "problems": problems}


# ---- the table of a base and its variants ---------------------------------------------------------------
TABLE_EARLIER_KEYS = ("name", "mass_final_g", "mass_saved_pct", "stiffness", "stiffness_source", "verdict", "verify", "print", "render")


def _named(x) -> bool:
    """True for a run folder written as a text or a path that is not blank. A blank one would read the current folder."""
    return isinstance(x, (str, os.PathLike)) and bool(os.fspath(x).strip())


def _variant_folders(run_a, variants) -> list[Path]:
    """The folders of the base and of the variants, in call order. A bad argument is a `UserError` that names both forms."""
    if not isinstance(variants, (list, tuple)):
        raise UserError(_FORMS_ERROR % f"variants must be a list of run folders, not {type(variants).__name__}")
    if not variants:
        raise UserError(_FORMS_ERROR % "variants is an empty list")
    if not all(_named(x) for x in [run_a, *variants]):
        raise UserError(_FORMS_ERROR % "run_a and every entry of variants must be a non-empty string (a run folder)")
    dirs = [Path(x).expanduser().resolve() for x in [run_a, *variants]]
    for d in dirs:
        if not d.is_dir():
            raise UserError(f"run folder not found: {d}. compare_runs did nothing.")
    for i, d in enumerate(dirs):
        if d in dirs[:i]:
            raise UserError(f"{d} is named twice (the base and the variants must be different folders). compare_runs did nothing.")
    return dirs


def _table_row(d: Path, f: dict, refused: bool) -> dict:
    """One row of the table: the facts of one run folder, read from its files. A fact that no file holds is null and one entry in `problems`."""
    problems: list[str] = list(f["problems"])
    raw = _load_json(d / "result" / "metrics.json", [], "result/metrics.json", expected=False) or {}
    m = f.get("metrics") or {}
    row: dict = {"name": f.get("name"), "dir_name": f.get("dir_name"), "run_dir": f.get("run_dir"), "state": f.get("state"),
                 "mass_final_g": _num(m.get("mass_final_g")), "mass_saved_pct": _num(m.get("mass_saved_pct"))}
    if row["mass_final_g"] is None:
        problems.append("mass_final_g is not in result/metrics.json")
    cases = m.get("stiffness_cases")
    if isinstance(cases, list) and cases:
        row["stiffness"] = {str(c.get("name")): _num(c.get("N_per_mm")) for c in cases if isinstance(c, dict)}
        for c in cases:
            if isinstance(c, dict) and _num(c.get("N_per_mm")) is None:
                problems.append(f"stiffness {c.get('name')}: no value ({c.get('reason') or 'result/metrics.json gives none'})")
    else:
        row["stiffness"] = None
        problems.append("stiffness_cases is not in result/metrics.json")
    row["stiffness_source"] = "result/metrics.json stiffness_cases (the optimizer, voxel model)" if row["stiffness"] is not None else None
    verdict = raw.get("verdict")
    if not isinstance(verdict, dict):
        verdict = _load_json(d / "verdict.json", [], "verdict.json", expected=False)
    row["verdict"] = verdict.get("state") if isinstance(verdict, dict) and isinstance(verdict.get("state"), str) else None
    if row["verdict"] is None:
        problems.append("verdict is not in result/metrics.json or verdict.json")
    row["verify"] = _table_verify(d, problems)
    pr = raw.get("print")
    if isinstance(pr, dict) and pr:
        row["print"] = ({"ok": False, "error": pr.get("error")} if pr.get("ok") is False else
                        {"ok": pr.get("ok"), "support_needed": pr.get("support_needed"),
                         "bed_fit": (pr.get("configured") or {}).get("bed_fit"),
                         "overhang_area_mm2": (pr.get("configured") or {}).get("overhang_area_mm2"),
                         "support_volume_mm3": (pr.get("configured") or {}).get("support_volume_mm3"),
                         "thinnest_member_mm": (pr.get("thinnest_member") or {}).get("thinnest_member_mm")})
    else:
        row["print"] = None
        problems.append("print is not in result/metrics.json (the print check did not run, or the run is from an older build)")
    row["render"] = _table_render(d, problems)
    row["problems"] = problems
    return earlier_run.mark_row(row, TABLE_EARLIER_KEYS, "row") if refused else row


def _table_verify(d: Path, problems: list) -> dict:
    """`verify/verify.json` as three words (the rule of `part_card._verify`) and, when there is a result, the numbers per load case."""
    path = d / "verify" / "verify.json"
    if not path.is_file():
        return {"status": "not run"}
    v = _load_json(path, problems, "verify/verify.json")
    if v is None:
        return {"status": "no result"}
    if v.get("ok") is not True or v.get("estimate_only") is True:
        return {"status": "no result"}
    out: dict = {"status": "done", "cases": {}}
    for name, c in (v.get("cases") or {}).items():
        c = c if isinstance(c, dict) else {}
        out["cases"][str(name)] = {"stiffness_N_per_mm": _num(c.get("stiffness_N_per_mm")),
                                   "p99_9_MPa": _num((c.get("von_mises") or {}).get("p99_9_by_volume_MPa"))}
    limit = v.get("limit")
    if isinstance(limit, dict) and limit:
        out["limit_exceeded"] = limit.get("exceeded") if isinstance(limit.get("exceeded"), bool) else None
    out["source"] = "verify/verify.json (the tet model)"
    run = v.get("run") if isinstance(v.get("run"), dict) else {}
    name = run.get("verified_file") if isinstance(run.get("verified_file"), str) else "final.stl"
    recorded = run.get("final_stl_sha256")
    out["same_mesh"] = None
    if isinstance(recorded, str):          # the rule of `part_card._same_mesh`: the SHA-256 of the file now in result/
        try:
            out["same_mesh"] = sha256_file(d / "result" / Path(name).name) == recorded
        except OSError:
            out["same_mesh"] = None
        if out["same_mesh"] is False:
            problems.append(f"verify/verify.json is NOT of the mesh now in result/{Path(name).name} (another SHA-256): run verify again")
    return out


def _table_render(d: Path, problems: list) -> dict | None:
    """The newest `renders/*_montage.png` (the name rule of `render.py`), by path. Nothing is made here."""
    try:
        found = [p for p in (d / "renders").glob("*_montage.png") if p.is_file()]
        found.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    except OSError:
        found = []
    whole = [p for p in found if "_section-" not in p.name]    # a cut view is `<stem>_section-x10_montage.png` (render.py)
    if not found:
        problems.append(f"render: no *_montage.png in renders/; render_part with run_dir={d} writes the images")
        return None
    newest = (whole or found)[-1]
    try:
        newer = newest.stat().st_mtime_ns >= (d / "result" / "final.stl").stat().st_mtime_ns
    except OSError:
        newer = None
    return {"montage_rel": f"renders/{newest.name}", "newer_than_final": newer, "section": newest not in whole}


def _pct(a, b) -> str:
    return f" ({(b - a) / abs(a) * 100:+.1f} %)" if a not in (0, 0.0) else ""


def _variants_table(run_a, variants, check_labels=None) -> dict:
    """A table of the base `run_a` and the `variants`: one row per run, and the setup diff of each variant against the base. It writes nothing.

    Each `vs_base` entry is built by `_compare_runs(base, variant)`, the function of the two-run form, and keeps the setup diff, the
    label check, the summary lines and the notes. A user who wants the whole pair calls the two-run form.
    """
    dirs = _variant_folders(run_a, variants)
    facts = [run_facts(d) for d in dirs]
    refused = [_refused(f, d) for f, d in zip(facts, dirs)]
    rows = [_table_row(d, f, r) for d, f, r in zip(dirs, facts, refused)]
    vs_base, notes, summary, problems = [], [], [], []
    for d, row, r in zip(dirs[1:], rows[1:], refused[1:]):
        try:
            pair = _compare_runs(dirs[0], d, check_labels)
        except Exception as exc:  # one pair that fails does not take the table down
            vs_base.append({"ok": False, "run_dir": str(d), "error": f"internal error: {type(exc).__name__}: {exc}"})
            problems.append(f"{d.name}: the compare with the base failed: {type(exc).__name__}: {exc}")
            continue
        vs_base.append({"ok": True, "run_dir": str(d), "setup": pair["setup"],
                        "labels_vs_disk": {"base": pair["inputs"]["a"].get("labels_vs_disk"),
                                           "variant": pair["inputs"]["b"].get("labels_vs_disk")},
                        "summary": pair["summary"], "notes": pair["notes"]})
    base = rows[0]
    for row in rows[1:]:
        line = [f"{row['dir_name']}: {row['state']}"]
        if base["mass_final_g"] is not None and row["mass_final_g"] is not None:
            line.append(f"mass_final_g {base['mass_final_g']:g} -> {row['mass_final_g']:g}{_pct(base['mass_final_g'], row['mass_final_g'])}")
        for case, v in (row["stiffness"] or {}).items():
            b = (base["stiffness"] or {}).get(case)
            if b is not None and v is not None:
                line.append(f"stiffness {case} {b:g} -> {v:g}{_pct(b, v)}")
        summary.append("; ".join(line))
    notes.append("Rows are in call order, the base first. The stiffness is the value of the optimizer (voxel model). The stiffness in "
                 "`verify` is of the tet model: the two differ.")
    sets = {tuple(sorted(r["stiffness"])) for r in rows if r["stiffness"]}
    if len(sets) > 1:
        notes.append("The runs do not all have the same load cases in their stiffness: compare a column only where every run has it.")
    for r, row, d in zip(refused, rows, dirs):
        if r:
            notes.append(f"{row['dir_name']} is a folder where an input check refused a job and kept an earlier run (see its earlier_run): "
                         "its facts in this table are of the earlier run.")
    return {"ok": True, "base": _brief_run(facts[0], refused[0]), "runs": rows, "vs_base": vs_base, "notes": notes,
            "summary": summary, "problems": problems}


def _target_volume(run: dict | None):
    """derived.volume.target_volume_mm3 of a run.json, or None."""
    der = (run or {}).get("derived")
    vol = der.get("volume") if isinstance(der, dict) else None
    return _num(vol.get("target_volume_mm3")) if isinstance(vol, dict) else None


def _brief_run(f: dict, refused: bool = False) -> dict:
    brief = {k: f.get(k) for k in ("name", "dir_name", "run_dir", "state", "iteration", "iterations",
                                   "started_at", "modified")} | {"yaml_path": f["yaml"]["path"]}
    return earlier_run.mark_row(brief, COMPARE_EARLIER_KEYS, "entry") if refused else brief   # P9-19


# ---- marks as hints ---------------------------------------------------------------------------------
def read_marks(marks) -> dict:
    """The marks document: a path to marks.json (the export of the refine page), its JSON text, or the dict.

    Raises UserError when the file is missing, is not JSON, or is not a version 1 marks file.
    """
    if isinstance(marks, str) and marks.lstrip().startswith("{"):   # the JSON text ("Copy JSON" of the page)
        try:
            marks = json.loads(marks)
        except ValueError as exc:
            raise UserError(f"the marks text is not valid JSON: {exc}")
    if isinstance(marks, dict):
        doc = marks
    else:
        p = Path(marks).expanduser()
        try:
            doc = json.loads(p.read_text())
        except OSError as exc:
            raise UserError(f"cannot read the marks file {p}: {exc}")
        except ValueError as exc:
            raise UserError(f"the marks file {p} is not valid JSON: {exc}")
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("marks"), list):
        raise UserError("this is not a marks file: it needs \"version\": 1 and a \"marks\" list "
                        "(the Refine page exports it; see topopt/refine/README.md)")
    return doc


def _norm_mark(m, index: int) -> tuple[dict | None, list[str]]:
    """One mark, checked. Returns (mark, problems). A mark with problems has mark = None."""
    errs: list[str] = []
    if not isinstance(m, dict):
        return None, ["the mark is not an object"]
    mid = m.get("id", index + 1)
    if isinstance(mid, bool) or not isinstance(mid, int):
        errs.append(f"id {mid!r} is not an integer")
    mtype = m.get("type")
    if mtype not in MARK_TYPES:
        errs.append(f"type {mtype!r} is not add, remove or smooth")
    pts = m.get("points")
    arr = None
    if not isinstance(pts, list) or not pts:
        errs.append("points is empty or not a list")
    else:
        try:
            arr = np.asarray(pts, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
                raise ValueError
        except (ValueError, TypeError):
            arr = None
            errs.append("points must be a list of [x, y, z] numbers")
    radius = _num(m.get("radius_mm", 4.0))
    if radius is None or radius <= 0:
        errs.append(f"radius_mm {m.get('radius_mm')!r} is not a number above 0")
    if errs:
        return None, errs
    return {"id": mid, "type": mtype, "points": arr, "radius_mm": radius,
            "amount_mm": m.get("amount_mm"), "strength": m.get("strength"),
            "mirror": bool(m.get("mirror", False)), "note": str(m.get("note") or ""),
            # optional keys of the mark (raw: `apply_marks` checks them against its mode lists)
            "smooth_mode": m.get("smooth_mode"), "remove_mode": m.get("remove_mode"),
            "radius_default_used": "radius_mm" not in m}, []


def _norm_options(options) -> tuple[dict, list[str]]:
    """The effective options of `make_hints`, and facts about the options that were not understood."""
    opt = {"remove": "keep_out", "add": "solid", "radius_scale": 1.0, "protect_regions": True,
           "clip_to_design": True, "per_mark": {}}
    problems: list[str] = []
    for k, v in (options or {}).items():
        if k not in opt:
            problems.append(f"option {k!r} is not known (known: {', '.join(opt)}); ignored")
        elif k in ("remove", "add"):
            ok = {"remove": ("keep_out", "skip"), "add": ("solid", "skip")}[k]
            if v in ok:
                opt[k] = v
            else:
                problems.append(f"option {k}={v!r} is not one of {list(ok)}; {opt[k]!r} is used")
        elif k == "radius_scale":
            if _num(v) is not None and v > 0:
                opt[k] = float(v)
            else:
                problems.append(f"option radius_scale={v!r} is not a number above 0; 1.0 is used")
        elif k in ("protect_regions", "clip_to_design"):
            opt[k] = bool(v)
        elif k == "per_mark":
            if isinstance(v, dict):
                opt[k] = {str(i): c for i, c in v.items()}
            else:
                problems.append("option per_mark must map a mark id to an option; ignored")
    return opt, problems


def _manifold_of(mesh):
    """A manifold3d solid from a trimesh, or None when the mesh is not a valid closed solid.

    It calls `post._to_manifold` and does the retry of `post.subtract_keep_out` (merge the vertices, then try again).
    """
    import trimesh
    from .post import _to_manifold

    faces = np.asarray(mesh.faces)
    if mesh.volume < 0:                 # an inside-out mesh: turn the faces
        faces = faces[:, ::-1]
    try:
        return _to_manifold(mesh.vertices, faces)
    except ValueError:
        pass
    fixed = trimesh.Trimesh(mesh.vertices, faces, process=True)
    fixed.merge_vertices()
    try:
        return _to_manifold(fixed.vertices, fixed.faces)
    except ValueError:
        return None


def _dedupe(points: np.ndarray) -> np.ndarray:
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > 1e-6:
            keep.append(i)
    return points[keep]


def _path_manifold(points: np.ndarray, r: float):
    """A sphere for one point, a chain of capsules for a stroke: every point within r of the path."""
    from .regions_builder import make_primitive
    pts = _dedupe(points)
    if len(pts) == 1:
        return make_primitive("sphere", {"center": pts[0].tolist(), "radius": r})
    out = None
    for p0, p1 in zip(pts[:-1], pts[1:]):
        c = make_primitive("capsule", {"p0": p0.tolist(), "p1": p1.tolist(), "radius": r})
        out = c if out is None else out + c
    return out


def _stl_bytes(man) -> bytes:
    from .regions_builder import to_trimesh
    return to_trimesh(man).export(file_type="stl")


def _mesh_from_bytes(data: bytes):
    import trimesh
    return trimesh.load(io.BytesIO(data), file_type="stl", force="mesh")


def _unique_name(name: str, taken: set) -> str:
    """`name`, or `name` with a number when `taken` has it. Names that differ only in case count as equal
    (a file system with no case (macOS default) would write one file over the other)."""
    low = {t.lower() for t in taken}
    if name.lower() not in low:
        return name
    stem, suffix = os.path.splitext(name)
    n = 2
    while f"{stem}_{n}{suffix}".lower() in low:
        n += 1
    return f"{stem}_{n}{suffix}"


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    n = 2
    while (base.parent / f"{base.name}-{n}").exists():
        n += 1
    return base.parent / f"{base.name}-{n}"


def _voxel_facts(mask: np.ndarray, labels: np.ndarray, named: dict) -> dict:
    """Voxels of a hint region: by class (as the engine labels them) and per original region."""
    n = int(mask.sum())
    by_label = {name: int((labels[mask] == code).sum()) for code, name in LABEL_NAMES.items()}
    overlap: dict = {"design": 0, "fixed": {}, "load": {}, "keep_out": {}}
    for (role, name), m in named.items():
        k = int((mask & m).sum())
        if role == "design":
            overlap["design"] = k
        elif k:
            overlap[role][name] = k
    return {"voxels_touched": n, "voxels_by_label": by_label, "overlap_regions": overlap}


class _Refused(UserError):
    """A step that is stopped because it could lose data. `reason`, `path` and `options` are facts for the caller."""

    def __init__(self, message: str, reason: str, path=None, options=()):
        super().__init__(message)
        self.reason = reason
        self.path = None if path is None else str(path)
        self.options = list(options)


def make_hints(run_dir, marks, out_dir=None, options=None, overwrite: bool = False, check: bool = True,
               yaml_path=None) -> dict:
    """Marks of the refine page as facts and options; a proposed YAML and region STLs in a NEW folder.

    run_dir: a finished run (needs run.json and the YAML it names; result/final.stl for the surface facts).
    marks: the path of a marks JSON (version 1, units mm, the frame of final.stl), its JSON text, or the dict.
    out_dir: the new folder. Default: `<run folder>-hints` beside the run folder (a number is added when it
    exists). Refused (the report says why and what to do): a run folder or a folder inside one, a folder that
    holds an input file of the run (the folder of the YAML or of a region STL), a symbolic link, a folder that
    holds files and is not an earlier hints folder, and any folder that holds files when `overwrite` is false.
    options: `remove` ("keep_out" or "skip"), `add` ("solid" or "skip"), `radius_scale` (1.0), `protect_regions`
    (true), `clip_to_design` (true), `per_mark` ({mark id: option}).
    check: also run the check of the proposed YAML (the data of `topopt check`) and put it in the report.
    yaml_path: the YAML of the run, when run.json names a file that is not there (a moved run). The labels
    check in the report says if it fits the run.
    Returns the full report (hints.json holds it with paths relative to the new folder), or
    `{"ok": false, "error": ...}` when it cannot start. `ok` is false with the report when the proposed YAML does
    not load. The input YAML, the run folder, every region STL of the run and every mesh stay as they are.
    """
    try:
        return _make_hints(Path(run_dir).expanduser().resolve(), marks, out_dir, options, overwrite, check, yaml_path)
    except _Refused as exc:
        return {"ok": False, "error": str(exc),
                "refused": {"reason": exc.reason, "path": exc.path, "options": exc.options}}
    except UserError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # a fact about this tool, not a crash of the caller
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


class _Shared:
    """What the marks share: the grid, the labels, the region masks, the cut tools and the files to write."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self._protect = self._design = self._final = None
        self._have: set = set()
        self.labels_now = self.labels.copy()                  # the labels with the hints of the earlier marks
        self.hint_cells = np.zeros(self.labels.shape, bool)   # the voxels that earlier hints hold
        self.hint_union = None                                # the earlier hint regions as one solid
        self.hint_regions: list = []                          # (mark id, solid) of each earlier hint

    def protect(self):
        """The fixed and load regions as one solid (a keep-out hint is cut by it), or None."""
        if "protect" not in self._have:
            self._have.add("protect")
            parts = []
            for (role, name), mesh in self.meshes.items():
                if role in ("fixed", "load"):
                    man = _manifold_of(mesh)
                    if man is None:
                        self.problems.append(f"region {name} is not a valid solid for a boolean: "
                                             "it is not cut out of the hints")
                    else:
                        parts.append(man)
            self._protect = _union(parts) if parts else None
        return self._protect

    def design(self):
        if "design" not in self._have:
            self._have.add("design")
            self._design = _manifold_of(self.meshes[("design", "design")])
            if self._design is None:
                self.problems.append("the design STL is not a valid solid for a boolean: "
                                     "a solid hint is not cut to the design space")
        return self._design

    def final_solid(self):
        """final.stl as a solid, or None (no file, or not a valid closed solid)."""
        if "final" not in self._have:
            self._have.add("final")
            if self.final_mesh is not None:
                self._final = _manifold_of(self.final_mesh)
                if self._final is None:
                    self.problems.append("result/final.stl is not a valid solid for a boolean: "
                                         "the fact final_material_in_region_mm3 is not given")
        return self._final

    def add_hint(self, mid, region, drawn, rmask, kind: str) -> None:
        """Record an applied hint: its solid, its voxels, and the labels that it changes.

        `drawn` is the region before the cut by the earlier hints. The union of the drawn regions is the same set as
        the union of the cut regions, and it has no faces that two regions share (shared faces make an STL that
        does not close after the float32 round trip).
        """
        self.hint_union = drawn if self.hint_union is None else self.hint_union + drawn
        self.hint_regions.append((mid, region))
        self.hint_cells |= rmask
        if kind == "keep_out":
            self.labels_now[rmask] = self.keep_out_code
        else:
            free = rmask & ((self.labels_now == self.design_code) | (self.labels_now == 0))
            self.labels_now[free] = self.load_code


def _region_paths(regions, base: Path) -> list[Path]:
    """The files named under `regions` (every key), resolved against the YAML folder, in order, no repeats."""
    out: list[Path] = []

    def walk(node) -> None:
        if isinstance(node, str):
            p = Path(node).expanduser()
            p = (p if p.is_absolute() else base / p).resolve()
            if p.is_file() and p not in out:
                out.append(p)
        elif isinstance(node, list):
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            for x in node.values():
                walk(x)
    walk(regions)
    return out


def _rewrite_regions(node, base: Path, copies: dict):
    """The `regions` of the YAML with each file name replaced by the name of its copy in the new folder."""
    if isinstance(node, str):
        p = Path(node).expanduser()
        p = (p if p.is_absolute() else base / p).resolve()
        return copies.get(str(p), node)
    if isinstance(node, list):
        return [_rewrite_regions(x, base, copies) for x in node]
    if isinstance(node, dict):
        return {k: _rewrite_regions(x, base, copies) for k, x in node.items()}
    return node


def _earlier_hints_folder(out: Path) -> bool:
    """Does the folder hold a hints.json of the format that this module writes?"""
    try:
        data = json.loads((out / "hints.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("format") == HINTS_FORMAT


def _plan_out_dir(out_dir, d: Path, input_dirs: set, overwrite: bool) -> Path:
    """The folder to write, or a refusal. Nothing is written here. Every refusal names the fact and the options."""
    new_name = "Name a new folder in out_dir, or leave out_dir out: the default is a new folder beside the run folder."
    if out_dir is None:
        return _unique_dir(d.parent / f"{d.name}-hints")
    given = Path(os.path.abspath(Path(out_dir).expanduser()))
    out = given.resolve()
    if os.path.islink(given):
        raise _Refused(f"out_dir {given} is a symbolic link. The tool does not write through a symbolic link.",
                       "symbolic_link", given, [new_name])
    if out.exists() and not out.is_dir():
        raise _Refused(f"out_dir {out} is a file, not a folder.", "not_a_folder", out, [new_name])
    for p in (out, *out.parents):
        if (p / "run.json").is_file() or (p / "status.json").is_file():
            where = "is a run folder" if p == out else f"is inside the run folder {p}"
            raise _Refused(f"out_dir {out} {where} (it has run.json or status.json). The files of a run are not "
                           "written. " + new_name, "run_folder", p, [new_name])
    if out in input_dirs:
        raise _Refused(f"out_dir {out} holds an input file of the run (the YAML or a region STL). The tool does not "
                       "write into the folder of an input. " + new_name, "input_folder", out, [new_name])
    holds_files = out.is_dir() and any(out.iterdir())
    if holds_files and not overwrite:
        raise _Refused(f"out_dir {out} holds files. Options: name a new folder; or set overwrite=true if it is an "
                       "earlier hints folder (it has a hints.json).", "not_empty", out,
                       [new_name, "overwrite=true, only for an earlier hints folder"])
    if holds_files and not _earlier_hints_folder(out):
        raise _Refused(f"out_dir {out} holds files and is not an earlier hints folder (no hints.json of format "
                       f"{HINTS_FORMAT}). overwrite=true is accepted for an earlier hints folder only. " + new_name,
                       "not_a_hints_folder", out, [new_name])
    return out


def _check_targets(out: Path, names) -> None:
    """Refuse when a file that will be written is a symbolic link or not a plain file. Nothing is written."""
    for name in sorted(set(names)):
        t = out / name
        if os.path.islink(t):
            raise _Refused(f"{t} is a symbolic link. The tool does not write through a symbolic link, and it does not "
                           "remove one. Name a new folder in out_dir.", "symbolic_link", t,
                           ["Name a new folder in out_dir."])
        if t.exists() and not t.is_file():
            raise _Refused(f"{t} is not a plain file. Name a new folder in out_dir.", "not_a_file", t,
                           ["Name a new folder in out_dir."])


def _relpath(path, start) -> str:
    try:
        return os.path.relpath(str(path), str(start))
    except ValueError:    # another drive
        return Path(path).name


def _scrub_strings(obj, replacements: list):
    """The same data with each absolute folder in a text replaced (longest first)."""
    if isinstance(obj, dict):
        return {k: _scrub_strings(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_strings(v, replacements) for v in obj]
    if isinstance(obj, str):
        for old, new in replacements:
            obj = obj.replace(old, new)
    return obj


def _portable(report: dict, out: Path, d: Path, yaml_path: Path, marks_file) -> dict:
    """The report as hints.json holds it: no absolute path. A path is relative to the folder of hints.json, or a name."""
    rep = json.loads(json.dumps(report))
    reps = sorted([(str(out), "."), (str(d), _relpath(d, out)), (str(yaml_path.parent), _relpath(yaml_path.parent, out))],
                  key=lambda x: -len(x[0]))
    rep = _scrub_strings(rep, reps)
    rep["run_dir"] = _relpath(d, out)
    rep["out_dir"] = "."
    rep["marks_file"] = Path(marks_file).name if marks_file else None
    rep["source"]["yaml_path"] = yaml_path.name
    rep["proposed_yaml"]["path"] = Path(rep["proposed_yaml"]["path"]).name
    if isinstance(rep.get("check"), dict) and rep["check"].get("yaml"):
        rep["check"]["yaml"] = Path(rep["check"]["yaml"]).name
    rep["paths_are_relative_to"] = "the folder of this file (hints.json)"
    return rep


def _make_hints(d: Path, marks, out_dir, options, overwrite: bool, check: bool, yaml_arg=None) -> dict:
    t0 = time.monotonic()
    import manifold3d as m3d
    import trimesh
    from .voxelize import DESIGN, FIXED, KEEP_OUT, LOAD, inside_mask, load_mesh, voxelize

    if not d.is_dir():
        raise UserError(f"run folder not found: {d}")
    problems: list[str] = []
    notes: list[str] = []
    run = _load_json(d / "run.json", problems, "run.json")
    if run is None:
        raise UserError(f"{d / 'run.json'} is missing or unreadable: the YAML of the run is not known")
    recorded = _run_yaml_path(run, d)
    if yaml_arg:
        yaml_path = Path(str(yaml_arg)).expanduser().resolve()
        if not yaml_path.is_file():
            raise UserError(f"yaml_path {yaml_path} is not a file")
        if recorded is None or recorded.resolve() != yaml_path:
            notes.append(f"yaml_path was given: {yaml_path}. run.json names {_shown(recorded)}. The labels check in "
                         "grid.labels_match_run says if this YAML and its STL files give the labels of the run.")
    else:
        if recorded is None or not recorded.is_file():
            raise UserError(f"the YAML of the run is not on disk: {_shown(recorded)}. The hint copies it and needs it. "
                            "Option: give yaml_path (the YAML of the run, at its new place).")
        yaml_path = recorded.resolve()
    yaml_text = yaml_path.read_text()
    yaml_sha = _sha256(yaml_path)
    cfg = load_config(yaml_path)
    raw_doc = yaml.safe_load(yaml_text)
    opts, opt_problems = _norm_options(options)
    problems += opt_problems
    doc = read_marks(marks)
    marks_file = None if isinstance(marks, dict) or str(marks).lstrip().startswith("{") \
        else str(Path(marks).expanduser().resolve())
    marks_sha = _sha256(Path(marks_file)) if marks_file else hashlib.sha256(
        json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()

    # -- the new folder: every check is done before the work (voxelization, booleans)
    src_files = _region_paths(raw_doc.get("regions"), yaml_path.parent)
    input_dirs = {yaml_path.parent.resolve()} | {p.parent for p in src_files}
    out = _plan_out_dir(out_dir, d, input_dirs, overwrite)
    copies: dict[str, str] = {}      # resolved source path -> file name in the new folder
    taken = {yaml_path.name, "hints.json"}
    for p in src_files:
        copies[str(p)] = _unique_name(p.name, taken | set(copies.values()))
    names_taken = set(copies.values()) | taken
    planned: list[str] = []          # a superset of the hint files that can be written
    for index, raw in enumerate(doc["marks"]):
        if isinstance(raw, dict) and raw.get("type") in ("add", "remove") and isinstance(raw.get("id", index + 1), int):
            n = _unique_name(f"hint_{raw['type']}_{raw.get('id', index + 1)}.stl", names_taken | set(planned))
            planned.append(n)
    _check_targets(out, set(copies.values()) | {yaml_path.name, "hints.json"} | set(planned))

    # -- the grid and the labels of the YAML on disk
    origin, h, shape, labels, _ids = voxelize(cfg)
    run_labels, rl_problem = _read_labels(d, run)
    grid_same = (list(shape) == list(run.get("shape") or []) and abs(h - float(run.get("h", -1))) < 1e-9
                 and bool(np.allclose(origin, run.get("origin", [0, 0, 0]), atol=1e-6)))
    labels_match = bool(grid_same and run_labels is not None and np.array_equal(labels, run_labels))
    if not labels_match:
        problems.append("The YAML and STL files on disk do not give the labels that the run recorded"
                        + (f" ({rl_problem})" if rl_problem else "")
                        + ". The marks refer to final.stl of the run; the proposed YAML is built from the YAML on disk.")
    # -- region meshes and masks (the same calls as the voxelizer)
    items = [("design", "design", cfg.regions.design)]
    items += [("fixed", p.name, p) for p in cfg.regions.fixed]
    items += [("keep_out", p.name, p) for p in cfg.regions.keep_out]
    items += [("load", lr.name, lr.path) for lr in cfg.regions.load]
    loaded = [load_mesh(path) for _role, _name, path in items]
    lo = np.min([m.bounds[0] for m in loaded], axis=0)
    hi = np.max([m.bounds[1] for m in loaded], axis=0)
    named = {(role, name): inside_mask(m, origin, h, shape) for (role, name, _p), m in zip(items, loaded)}
    meshes = {(role, name): m for (role, name, _p), m in zip(items, loaded)}
    design_bounds = loaded[0].bounds
    symmetric = bool(abs(design_bounds[0][1] + design_bounds[1][1]) <= h)
    # -- final.stl
    final = d / "result" / "final.stl"
    final_mesh = final_sha = None
    if final.is_file():
        try:
            final_sha = _sha256(final)
            final_mesh = trimesh.load(str(final), file_type="stl", force="mesh")
        except Exception as exc:  # a gap in the facts
            problems.append(f"result/final.stl cannot be read: {exc}")
    else:
        problems.append("result/final.stl is missing: the surface distance of the marks is not given")
    sha_marks = doc.get("source_sha256")
    marks_match = None if not (sha_marks and final_sha) else (sha_marks == final_sha)
    if marks_match is False:
        problems.append("source_sha256 of the marks differs from final.stl of this run: the marks were made for "
                        "another mesh, so the positions may not match this run")
    if doc.get("units", "mm") != "mm":
        problems.append(f"units is {doc.get('units')!r}, not mm: the numbers are used as mm")

    sh = _Shared(origin=origin, h=h, shape=shape, labels=labels, named=named, meshes=meshes, problems=problems,
                 box=m3d.Manifold.cube((hi - lo).tolist()).translate(lo.tolist()), final_mesh=final_mesh,
                 inside_mask=inside_mask, design_code=DESIGN, keep_out_code=KEEP_OUT, load_code=LOAD,
                 stl_files={}, taken=set(names_taken), new_keep_out=[], new_solid={},
                 load_names={lr.name for lr in cfg.regions.load}, n_design=int((labels == DESIGN).sum()))
    entries: list[dict] = []
    seen_ids: set = set()
    for index, raw in enumerate(doc["marks"]):
        mk, errs = _norm_mark(raw, index)
        if mk is None:
            entries.append({"id": raw.get("id") if isinstance(raw, dict) else None, "index": index, "type": None,
                            "status": "unreadable", "option_used": None, "reason": "; ".join(errs),
                            "facts": {}, "options": []})
            continue
        if mk["id"] in seen_ids:
            problems.append(f"mark id {mk['id']} appears more than once; the region file names get a number")
        seen_ids.add(mk["id"])
        entries.append(_one_mark(mk, index, opts, sh))

    # -- the files to write: check them again (the names of the hints are known now), then write
    _check_targets(out, set(copies.values()) | {yaml_path.name, "hints.json"} | set(sh.stl_files))
    new_doc = yaml.safe_load(yaml_text)
    rg = _rewrite_regions(new_doc.get("regions") or {}, yaml_path.parent, copies)
    kept = list(rg.get("keep_out") or [])
    rg["keep_out"] = kept + sh.new_keep_out
    if not rg["keep_out"]:
        rg.pop("keep_out", None)
    rg.setdefault("load", {})
    rg["load"].update(sh.new_solid)
    new_doc["regions"] = rg
    n_k = sum(1 for e in entries if e["status"] == "applied" and e["option_used"] == "keep_out")
    n_s = sum(1 for e in entries if e["status"] == "applied" and e["option_used"] == "solid")
    new_doc["assumptions"] = list(new_doc.get("assumptions") or []) + [
        f"Proposed from run {d.name} and the marks file {Path(marks_file).name if marks_file else '(inline)'}: "
        f"{n_k} keep-out and {n_s} solid hint regions. The marks are points of emphasis, not absolutes."]
    new_yaml = out / yaml_path.name
    try:
        out.mkdir(parents=True, exist_ok=True)
        # The region STLs of the run are copied flat next to the new YAML: post-process looks for keep-out files there.
        for src, name in copies.items():
            shutil.copyfile(src, out / name)
        for fname, data in sh.stl_files.items():
            (out / fname).write_bytes(data)
        new_yaml.write_text(yaml.safe_dump(new_doc, sort_keys=False, default_flow_style=None))
    except OSError as exc:      # includes shutil.SameFileError
        raise UserError(f"cannot write in {out}: {type(exc).__name__}: {exc}. Name another out_dir.")
    stale = sorted(p.name for p in out.iterdir()
                   if p.is_file() and p.name not in set(copies.values()) | set(sh.stl_files) | {yaml_path.name, "hints.json"})
    notes += [
        "Comments of the input YAML are not copied: the proposed YAML is written with PyYAML.",
        "The region STL files of the run are copied next to the proposed YAML. The YAML names them by file name "
        "only, and post-process looks for the keep-out files next to the YAML.",
        "Used from a mark: points, radius_mm, mirror. Not used: " + ", ".join(VALUES_NOT_USED)
        + ". A keep-out or solid region has no depth and no strength.",
        "mirror: the plane is y = 0 of the frame of final.stl, which is the frame of the region STL files. "
        + ("The design space is symmetric about y = 0 within one voxel." if symmetric else
           "The design space is NOT symmetric about y = 0: a mirrored copy can fall outside it."),
        "The first mark wins an overlap: a hint region is cut by the hint regions of the marks before it in the file."]
    if stale:
        notes.append("Files from an earlier call stay in the folder (they are not in the proposed YAML): "
                     + ", ".join(stale[:10]))

    # -- the proposed YAML through the normal path
    proposed: dict = {"path": str(new_yaml), "loads": False, "error": None}
    labels_changed = None
    try:
        o2, h2, s2, lab2, _ = voxelize(load_config(new_yaml))
        proposed["loads"] = True
        same_grid = list(s2) == list(shape) and abs(h2 - h) < 1e-9 and bool(np.allclose(o2, origin, atol=1e-9))
        labels_changed = {"grid_unchanged": same_grid}
        if same_grid:
            labels_changed.update(_label_changes(labels, lab2, origin, h))
        else:
            labels_changed["note"] = ("the proposed YAML makes another grid: its regions reach outside the bounds "
                                      "of the input regions")
    except UserError as exc:
        proposed["error"] = str(exc)
        problems.append(f"the proposed YAML does not load: {exc}")
        _map_loader_error(str(exc), entries)
    check_out = check_error = None
    if check and proposed["loads"]:
        try:
            from .cli import check_data
            check_out = check_data(new_yaml)
        except UserError as exc:
            check_error = str(exc)
        except Exception as exc:
            check_error = f"internal error: {type(exc).__name__}: {exc}"
    summary: dict = {}
    for e in entries:
        k = f"{e['type'] or 'mark'}:{e['status']}"
        summary[k] = summary.get(k, 0) + 1
    summary["proposed_yaml"] = "loads" if proposed["loads"] else f"does not load: {proposed['error']}"
    report = {
        "ok": bool(proposed["loads"]), "format": HINTS_FORMAT, "run_dir": str(d), "run_name": run.get("name"),
        "marks_file": marks_file, "marks_sha256": marks_sha, "marks_count": len(doc["marks"]),
        "marks_match_final_stl": marks_match,
        "source": {"yaml_path": str(yaml_path), "yaml_sha256": yaml_sha, "final_stl_sha256": final_sha},
        "out_dir": str(out), "proposed_yaml": proposed,
        "grid": {"origin": origin, "h": h, "shape": shape, "same_as_run": grid_same, "labels_match_run": labels_match,
                 "design_symmetric_about_y0": symmetric, "bounds_mm": {"min": lo, "max": hi}},
        "options": opts, "option_catalog": mark_options(run), "marks": entries, "summary": summary,
        "files": {"yaml": new_yaml.name, "region_files": sorted(sh.stl_files),
                  "copied_region_files": sorted(copies.values()), "report": "hints.json"},
        "labels_changed": labels_changed, "volume_target": _volume_target(run, check_out),
        "check": check_out, "check_error": check_error,
        "values_used": VALUES_USED, "values_not_used": VALUES_NOT_USED, "problems": problems, "notes": notes,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    if not proposed["loads"]:
        report["error"] = f"the proposed YAML does not load: {proposed['error']}"
    report = _jsonable(report)
    try:
        (out / "hints.json").write_text(json.dumps(_portable(report, out, d, yaml_path, marks_file), indent=1))
    except OSError as exc:
        raise UserError(f"cannot write hints.json in {out}: {exc}")
    return report


def _volume_target(run: dict, check_out) -> dict:
    """The target volume before (run.json) and after (the check of the proposed YAML). A hint changes it: the
    volume fraction applies to the free design voxels, and solid voxels come on top."""
    before = None
    der = run.get("derived") if isinstance(run.get("derived"), dict) else {}
    vol = der.get("volume") if isinstance(der.get("volume"), dict) else {}
    if "target_volume_mm3" in vol:
        before = {"target_volume_mm3": vol.get("target_volume_mm3"), "target_mass_g": vol.get("target_mass_g")}
    after = None
    if isinstance(check_out, dict):
        v = ((check_out.get("derived") or {}).get("volume") or {})
        if "target_volume_mm3" in v:
            after = {"target_volume_mm3": v.get("target_volume_mm3"), "target_mass_g": v.get("target_mass_g")}
    note = ("before: derived.volume of run.json. after: the check of the proposed YAML. The volume fraction "
            "applies to the free design voxels; forced-solid voxels come on top, keep-out voxels are not free.")
    if der.get("target_mass_asked_g") is not None:   # I7: the YAML has settings.target_mass_g; the fraction follows the voxels, the mass stays
        note += (" The YAML has settings.target_mass_g: the engine derives the volume fraction from the voxels of each YAML, so the "
                 "target mass stays the one asked and the fraction changes with the free and forced-solid voxels.")
    return {"before": before, "after": after, "note": note}


def _map_loader_error(text: str, entries: list) -> None:
    """Put a loader error on the mark entry that it names (a hint region), when it names one."""
    import re
    hit = []
    m = re.search(r"load region '([^']+)' has no voxel", text) or re.search(r"region load '([^']+)' has no voxel", text)
    if m:
        hit = [e for e in entries if e.get("facts", {}).get("load_region_name") == m.group(1)]
        if not hit:      # a region of the problem that a keep-out hint covers
            hit = [e for e in entries if e.get("status") == "applied" and e.get("option_used") == "keep_out"
                   and m.group(1) in (e.get("facts", {}).get("overlap_regions", {}).get("load") or {})]
    m = re.search(r"region keep_out\[\d+\] \(([^)]+)\) has no voxel", text)
    if m:
        hit = [e for e in entries if e.get("region_file") == m.group(1)]
    m = re.search(r"fixed region (\S+) has no voxel left", text)
    if m:
        hit = [e for e in entries if e.get("status") == "applied" and e.get("option_used") == "keep_out"
               and m.group(1) in (e.get("facts", {}).get("overlap_regions", {}).get("fixed") or {})]
    for e in hit:
        e["status"] = "invalid"
        e["reason"] = f"the loader stops on this region: {text}"
        e["loader_error"] = text


def _union(parts):
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def _surface_distance(mesh, pts: np.ndarray):
    """Distance in mm from each point to the surface of a mesh, or None if it cannot be computed."""
    try:
        import trimesh
        _closest, dist, _tri = trimesh.proximity.closest_point(mesh, pts)
        return [round(float(x), 3) for x in dist]
    except Exception:
        return None


def _entry(mk: dict, index: int, facts: dict, status: str, used: str | None, options: list, reason: str | None) -> dict:
    """One mark in the report. `option_used` is the option that the proposed YAML uses for it."""
    for o in options:
        o["used"] = o["id"] == used
    return {"id": mk["id"], "index": index, "type": mk["type"], "status": status, "option_used": used,
            "reason": reason, "facts": facts, "options": options}


def _one_mark(mk: dict, index: int, opts: dict, sh: _Shared) -> dict:
    """The entry of one mark: the facts, the options, and the option that the proposed YAML uses.

    Status: applied (a region file goes into the YAML), skipped_by_option, no_effect (the region holds no free
    design voxel, or is empty, or lies inside the region of an earlier mark), facts_only (a type with no YAML
    key in this version), invalid (the loader stops on the region; the proposed YAML does not load).
    The marks are processed in file order. The first mark wins an overlap: a hint region is cut by the hint
    regions of the earlier marks, and its effect is counted on the labels that already hold them.
    """
    h, shape, labels, origin = sh.h, sh.shape, sh.labels, sh.origin
    mtype, mid = mk["type"], mk["id"]
    r = mk["radius_mm"] * opts["radius_scale"]
    pts = mk["points"]
    flip = np.array([1.0, -1.0, 1.0])
    options = [{"id": o["id"], "implemented": o["implemented"]} for o in MARK_OPTIONS[mtype]]
    facts: dict = {"position_mm": [round(float(v), 3) for v in pts[0]], "points": len(pts),
                   "path_mm": [[round(float(v), 3) for v in p] for p in pts],
                   "path_length_mm": round(float(np.linalg.norm(np.diff(_dedupe(pts), axis=0), axis=1).sum()), 3),
                   "radius_mm": mk["radius_mm"], "radius_used_mm": r, "mirror": mk["mirror"],
                   "amount_mm": mk["amount_mm"], "strength": mk["strength"], "note": mk["note"]}
    if mk["radius_default_used"]:
        facts["radius_note"] = "radius_mm is missing in the file: 4 mm (the page default) is used"
    if sh.final_mesh is not None:
        dist = _surface_distance(sh.final_mesh, pts)
        facts["surface_distance_mm"] = dist
        if dist and max(dist) > 2 * h:
            facts["surface_note"] = (f"a point is {max(dist):.2f} mm from the surface of final.stl, more than two "
                                     "voxels: the mark may belong to another mesh")
    # the mark as drawn, cut to the box of the input regions so the grid does not move
    paths = [pts] + ([pts * flip] if mk["mirror"] else [])
    full = _union([_path_manifold(p, r) for p in paths])
    vol_full = float(full.volume())
    base = full ^ sh.box
    vol_box = 0.0 if base.is_empty() else float(base.volume())
    facts["outside_grid_box_pct"] = round(100.0 * (1.0 - vol_box / vol_full), 2) if vol_full > 0 else 0.0
    if mk["mirror"]:
        mir = _path_manifold(paths[1], r) ^ sh.box
        if mir.is_empty():
            facts["mirror_copy"] = {"outside_grid_box": True, "voxels_touched": 0, "design_voxels": 0,
                                    "note": "the mirrored copy (y -> -y) lies outside the bounds of the region STL "
                                            "files: it is not in the YAML"}
        else:
            mm = sh.inside_mask(_mesh_from_bytes(_stl_bytes(mir)), origin, h, shape)
            facts["mirror_copy"] = {"outside_grid_box": False, "voxels_touched": int(mm.sum()),
                                    "design_voxels": int((labels[mm] == sh.design_code).sum())}
    if base.is_empty():
        return _entry(mk, index, facts, "no_effect", None, options,
                      "the mark lies outside the bounds of the region STL files")
    raw_mask = sh.inside_mask(_mesh_from_bytes(_stl_bytes(base)), origin, h, shape)
    facts.update(_voxel_facts(raw_mask, labels, sh.named))
    facts["design_voxels_share_pct"] = round(100.0 * facts["voxels_by_label"]["design"] / max(sh.n_design, 1), 3)
    if mtype == "smooth":
        return _entry(mk, index, facts, "facts_only", "none", options,
                      "the YAML format has no key that smooths one place; the facts and options are in the report")
    default = "keep_out" if mtype == "remove" else "solid"
    choice = opts["per_mark"].get(str(mid), opts[mtype])
    if choice not in (default, "skip"):
        facts["option_note"] = f"per_mark option {choice!r} does not fit a {mtype} mark: {default} is used"
        choice = default
    if choice == "skip":
        return _entry(mk, index, facts, "skipped_by_option", "skip", options,
                      "the options say skip: no YAML change for this mark")
    region, cut = base, []
    vol_start = vol_box

    def cut_by(what: str, tool) -> None:
        """Cut the region with a solid; name the tool in `cut` only when it removed volume."""
        nonlocal region
        before = 0.0 if region.is_empty() else float(region.volume())
        region = tool(region)
        after = 0.0 if region.is_empty() else float(region.volume())
        if after < before * (1.0 - 1e-6):
            cut.append(what)

    if mtype == "remove" and opts["protect_regions"] and sh.protect() is not None:
        cut_by("the fixed and load regions", lambda m: m - sh.protect())
    if mtype == "add" and opts["clip_to_design"] and sh.design() is not None:
        cut_by("the design space", lambda m: m ^ sh.design())
    # the first mark wins an overlap: cut the regions of the earlier hints out of this one
    drawn = region
    earlier = []
    if sh.hint_union is not None and not region.is_empty():
        earlier = [eid for eid, eman in sh.hint_regions if not (region ^ eman).is_empty()]
        if earlier:
            pre_mask = sh.inside_mask(_mesh_from_bytes(_stl_bytes(region)), origin, h, shape)
            vol_pre = float(region.volume())
            cut_by("the regions of the earlier marks " + ", ".join(str(i) for i in earlier), lambda m: m - sh.hint_union)
            vol_post = 0.0 if region.is_empty() else float(region.volume())
            facts["overlap_earlier_hints"] = {
                "marks": earlier, "voxels": int((pre_mask & sh.hint_cells).sum()),
                "volume_mm3": round(vol_pre - vol_post, 2),
                "text": "the first mark wins an overlap: this region is cut by the regions of the marks before it"}
    if region.is_empty():
        if earlier:
            why = ("the region lies inside the regions of the earlier marks " + ", ".join(str(i) for i in earlier)
                   + ": the first mark wins an overlap, so nothing is left of this one")
        else:
            why = "nothing is left of the region after it is cut to " + " and ".join(cut or ["the grid box"])
        return _entry(mk, index, facts, "no_effect", "none", options, why)
    data = _stl_bytes(region)
    rmesh = _mesh_from_bytes(data)
    rmask = sh.inside_mask(rmesh, origin, h, shape)
    eff = int((rmask & (sh.labels_now == sh.design_code)).sum())     # on the labels that hold the earlier hints
    facts["applied_region"] = {"cut_to": cut, "voxels": int(rmask.sum()), "design_voxels": eff,
                               "volume_mm3": round(float(region.volume()), 2),
                               "cut_volume_pct": round(100.0 * (1.0 - float(region.volume()) / vol_start), 2)
                               if vol_start > 0 else 0.0}
    fin = sh.final_solid()
    if fin is not None:
        inter = region ^ fin
        vol_in = 0.0 if inter.is_empty() else float(inter.volume())
        facts["final_material_in_region_mm3"] = round(vol_in, 2)
        facts["touches_final_material"] = vol_in > 0
    else:
        facts["touches_final_material"] = None
    if eff == 0:
        why = ("no free design voxel lies in the region: it changes no label, and an STL with no voxel stops the loader"
               if mtype == "remove" else
               "no free design voxel lies in the region: it is already solid, or outside the design space")
        if earlier:
            why += " (after the regions of the earlier marks are cut out)"
        return _entry(mk, index, facts, "no_effect", "none", options, why)
    if not rmesh.is_watertight:
        return _entry(mk, index, facts, "no_effect", "none", options,
                      "the region STL is not closed, so the loader would stop on it")
    fname = _unique_name(f"hint_{mtype}_{mid}.stl", sh.taken)
    sh.taken.add(fname)
    sh.stl_files[fname] = data
    if mtype == "remove":
        sh.new_keep_out.append(fname)
    else:
        rname = _unique_name(f"hint_add_{mid}", sh.load_names | set(sh.new_solid))
        sh.new_solid[rname] = fname
        facts["load_region_name"] = rname
    sh.add_hint(mid, region, drawn, rmask, "keep_out" if mtype == "remove" else "solid")
    entry = _entry(mk, index, facts, "applied", choice, options, None)
    entry["region_file"] = fname
    entry["region"] = {"kind": "sphere" if len(_dedupe(pts)) == 1 else "capsule chain",
                       "centres_mm": [[round(float(v), 4) for v in p] for p in _dedupe(pts)],
                       "radius_mm": r, "mirror": mk["mirror"],
                       "bbox_mm": {"min": [round(float(v), 3) for v in rmesh.bounds[0]],
                                   "max": [round(float(v), 3) for v in rmesh.bounds[1]]}}
    return entry



def short_report(report: dict) -> dict:
    """The short reply of `marks_to_hints`: what the agent needs to decide. The full report is in hints.json
    (and in the return value of `make_hints`)."""
    if not report.get("marks") and not report.get("out_dir"):
        return report                               # an error with no report
    marks = []
    for e in report.get("marks", []):
        f = e.get("facts") or {}
        ar = f.get("applied_region") or {}
        line = {"id": e.get("id"), "type": e.get("type"), "status": e.get("status"), "option_used": e.get("option_used"),
                "radius_used_mm": f.get("radius_used_mm"), "mirror": f.get("mirror"),
                "voxels_touched": f.get("voxels_touched"),
                "design_voxels": ar.get("design_voxels", (f.get("voxels_by_label") or {}).get("design")),
                "region_file": e.get("region_file"), "reason": e.get("reason")}
        if "touches_final_material" in f:
            line["touches_final_material"] = f["touches_final_material"]
        if f.get("overlap_earlier_hints"):
            line["overlaps_earlier_marks"] = f["overlap_earlier_hints"]["marks"]
        if (f.get("mirror_copy") or {}).get("outside_grid_box"):
            line["mirror_copy_outside_grid_box"] = True
        marks.append({k: v for k, v in line.items() if v is not None})
    chk = report.get("check")
    check = None
    if isinstance(chk, dict):
        te = chk.get("time_estimate") or {}
        check = {"active_voxels": chk.get("active_voxels"), "free_design_voxels": chk.get("free_design_voxels"),
                 "forced_solid_share": chk.get("forced_solid_share"),
                 "components": (chk.get("connectivity") or {}).get("components"),
                 "time_estimate": {k: te.get(k) for k in ("solver", "total_s", "limit_s")},
                 "max_lost_percent": (chk.get("lost_force") or {}).get("max_lost_percent"),
                 "warnings": chk.get("warnings")}
    g = report.get("grid") or {}
    out = {"ok": report.get("ok"), "run_dir": report.get("run_dir"), "out_dir": report.get("out_dir"),
           "proposed_yaml": report.get("proposed_yaml"), "summary": report.get("summary"), "marks": marks,
           "labels_changed": {k: v for k, v in (report.get("labels_changed") or {}).items() if k != "bbox_mm"} or None,
           "volume_target": (report.get("volume_target") or {}).copy(), "check": check,
           "check_error": report.get("check_error"),
           "grid": {k: g.get(k) for k in ("h", "shape", "labels_match_run", "design_symmetric_about_y0")},
           "files": report.get("files"), "problems": report.get("problems"),
           "detail": False, "report_file": "hints.json",
           "more": "hints.json holds the full report: facts for each mark, the options with their text, the notes. "
                   "Call again with detail=true to get it in the reply."}
    out["volume_target"].pop("note", None)
    if report.get("error"):
        out["error"] = report["error"]
    if report.get("marks_match_final_stl") is False:
        out["marks_match_final_stl"] = False
    return out


# ---- script -----------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m topopt.feedback", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history", help="facts of the runs under a folder (writes nothing)")
    h.add_argument("root")
    h.add_argument("--depth", type=int, default=2)
    h.add_argument("--limit", type=int, default=HISTORY_DEFAULT_LIMIT, help="the newest runs only; 0 = all")
    h.add_argument("--detail", action="store_true", help="the full facts of each run")
    c = sub.add_parser("compare", help="two runs side by side (writes nothing)")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--check-labels", choices=["auto", "yes", "no"], default="auto")
    m = sub.add_parser("hints", help="marks.json to a proposed YAML, region STLs and hints.json in a new folder")
    m.add_argument("run_dir")
    m.add_argument("marks")
    m.add_argument("--out", default=None)
    m.add_argument("--overwrite", action="store_true")
    m.add_argument("--no-check", action="store_true")
    m.add_argument("--yaml", default=None, help="the YAML of the run, when run.json names a file that is not there")
    m.add_argument("--detail", action="store_true", help="print the full report (hints.json always holds it)")
    m.add_argument("--remove", choices=["keep_out", "skip"], default=None)
    m.add_argument("--add", choices=["solid", "skip"], default=None)
    m.add_argument("--radius-scale", type=float, default=None)
    m.add_argument("--no-protect", action="store_true", help="do not cut the fixed and load regions out of a keep-out hint")
    m.add_argument("--no-clip", action="store_true", help="do not cut a solid hint to the design space")
    a = ap.parse_args(argv)
    if a.cmd == "history":
        res = run_history(a.root, a.depth, a.limit, a.detail)
    elif a.cmd == "compare":
        res = compare_runs(a.run_a, a.run_b, {"auto": None, "yes": True, "no": False}[a.check_labels])
    else:
        opts = {k: v for k, v in (("remove", a.remove), ("add", a.add), ("radius_scale", a.radius_scale)) if v is not None}
        if a.no_protect:
            opts["protect_regions"] = False
        if a.no_clip:
            opts["clip_to_design"] = False
        res = make_hints(a.run_dir, a.marks, a.out, opts, a.overwrite, not a.no_check, a.yaml)
        if not a.detail:
            res = short_report(res)
    print(json.dumps(_jsonable(res), indent=1))
    if not res.get("ok"):
        print(f"error: {res.get('error')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
