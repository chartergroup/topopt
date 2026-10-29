"""Checkpoint and resume of `topopt run` (CONTRACTS.md section 24.2).

After every pass, the run writes `checkpoint.npz` in the run folder. It holds the whole state that the next pass
reads, so a run that is killed can continue and give the same result as a run that was not killed.

File: one `.npz` (no compression). The arrays have names (`x`, `mma_xold1`, ...). The array `meta_json` holds a JSON
text with the format version, the iteration, the scalars, the last history record and the input hashes. One file means one
atomic write: the arrays and the text are never from two different passes.

Write: `<name>.tmp`, flush to the drive (`fcntl.F_FULLFSYNC` where it exists, else `fsync`), then `os.replace`, then a flush of
the folder. The file of the pass before moves to `checkpoint.prev.npz` first. A kill at any point leaves at least one valid file.
`read_checkpoint` takes `checkpoint.npz` if it is valid, else `checkpoint.prev.npz`. A file that lacks a core field (or has an
array of the wrong size) is not valid: the reader goes on to the older file, or refuses with the reason, before the folder changes.

The state of the rows (`MmaState.rows`, JSON data) can hold arrays: the buckling row keeps its start vector `v0` (#155). `pack_rows` puts each
array in the `.npz` under a name `rowvec_<n>` (exact float64) and leaves `{"npz_array": name}` in the JSON text; `unpack_rows` reverses it.
`mma.rows` and `rows_before` share an array that is equal in both. A checkpoint without such arrays (an older topopt) is read as it was.

Other code can add state. An object (the `Model`, the solver) with a method `checkpoint_state() -> {name: array}` and a method
`restore_checkpoint_state({name: array}) -> [facts]` gets its arrays saved under `<owner>__<name>`. A checkpoint that
lacks a field gives a fact in the run log, and the run goes on.

Input hashes are in the checkpoint only (not in `run.json`). A resume compares them and says what changed.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .config import ENVELOPE_FILL_DEFAULT, GPU_BACKWARD_ERROR_DEFAULT, GPU_MAX_ITER_DEFAULT, GPU_TOL_DEFAULT, OBJECTIVE_SCALE_DEFAULT, UserError
from .mma_sub import ASYINIT
from .runinfo import key_root, region_files, sha256_file, sha256_handle, write_atomic   # the one file-hash and region-file helpers (CONTRACTS.md section 24.1)

CHECKPOINT_FILE = "checkpoint.npz"
PRE_P2B_FIRST_PASS_CAP = 300     # the first-pass cap of gpu-mg before settings.gpu_max_iter existed (a checkpoint of G2 has no record of it)
PREV_FILE = "checkpoint.prev.npz"
TMP_FILE = CHECKPOINT_FILE + ".tmp"
FORMAT = 1
KIND = "topopt-checkpoint"
META = "meta_json"
FSYNC = True    # flush the file to disk before the rename.
LONG = 160      # a value in a "changed" fact is cut after this many characters
ROW_ARRAY = "rowvec_"      # the arrays of the row states in the .npz: "rowvec_0", "rowvec_1", ... (no "__": that is the form of the model and solver arrays)
ARRAY_REF = "npz_array"    # the key of a reference to such an array in the JSON text of a row state


class ResumeRefused(UserError):
    """A resume cannot go on, or the inputs changed and there is no override. `facts` and `options` are lists of text."""

    def __init__(self, message: str, facts=(), options=()):
        self.reason, self.facts, self.options = message, list(facts), list(options)
        super().__init__(message + (" Options: " + "; ".join(self.options) + "." if self.options else ""))


def _iso(epoch: float | None = None) -> str:
    when = datetime.now() if epoch is None else datetime.fromtimestamp(epoch)
    return when.astimezone().isoformat(timespec="seconds")


# --- input hashes --------------------------------------------------------------------------

def _jsonable(obj):
    """A copy of `obj` that went through JSON (tuples become lists), so that a saved value equals a new one."""
    return json.loads(json.dumps(obj))


def parsed_yaml(path) -> dict | None:
    """The YAML file as parsed data (what `yaml.safe_load` gives, through JSON), or None if it cannot be read as a mapping.

    The comparison of two runs uses this and not the bytes: a comment or the order of the keys is not a change of the problem, and a key that the
    loader does not turn into a setting (a later item adds such keys) still is.
    """
    try:
        data = yaml.safe_load(Path(path).read_text())
        return json.loads(json.dumps(data, default=str)) if isinstance(data, dict) else None
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return None


def effective_settings(config) -> dict:
    """What decides the numbers of a run, after the defaults: settings, material and load cases.

    Not in it: the two voxel guards, the stop rules and `fine_pass` (they do not change the numbers of a pass), `perimeter_max_mm2` when it
    is unset (so a checkpoint written before A4c still resumes), and `gpu_tol`, `gpu_backward_error` and `gpu_max_iter` when each
    has its default (G2, P2b: so a checkpoint written before them still resumes, with any solver; a value that is not the default
    is a change), and `objective_scale` at its default `stage` (P3a: the same rule, so a checkpoint written before the key still
    resumes; the resume says in a fact that the run goes on with `stage`, and `first` is a change: `_objective_scale_fact`),
    and `part_only_min_film_mm` when it is unset (P4b-6: the same rule as `perimeter_max_mm2`; a value is a change, like `mesh_method`, and so is `envelope_fill` when it is not ENVELOPE_FILL_DEFAULT:
    the key acts in the post step only, and the resume of a finished run repeats that step).
    The buckling row (`buckling.min_factor`, `ks_q`, `modes`) is in it only when the row is on.
    `settings.target_mass_g` (I7) is in it only when it is set (a run without it has the effective settings and the hash it had before the key).
    Then `volume_fraction` is not in it: the engine derives it from the target, the voxel labels and the density, and the plans compare those
    (`target_mass_g` here, the region files and the labels in `input_hashes`, `material.density` below). So a plan that has the config of
    `load_config` (the fraction is None there) and a run that has the config of `build_problem` (the derived fraction) give the same dict.
    """
    st = dataclasses.asdict(config.settings)
    if st.get("target_mass_g") is None:
        st.pop("target_mass_g", None)
    else:
        st.pop("volume_fraction", None)
    for key in ("max_active_voxels", "max_grid_voxels", "stop_rules", "fine_pass"):   # fine_pass (#318): it acts on a later fine pass only
        st.pop(key, None)
    st.pop("wall_mm", None)   # #328: a record of the wall rule; the radius and eta_e that it gave are in the dict, and so is robust: wall
    if st.get("perimeter_max_mm2") is None:   # A4c: a run without the key has the effective settings (and the hash) it had before the key existed
        st.pop("perimeter_max_mm2", None)
    for key, default in (("gpu_tol", GPU_TOL_DEFAULT), ("gpu_backward_error", GPU_BACKWARD_ERROR_DEFAULT),
                         ("gpu_max_iter", GPU_MAX_ITER_DEFAULT)):
        if st.get(key) == default:   # G2, P2b: the same rule as for the key above: a run at the default has the effective settings it had before
            st.pop(key, None)
    if st.get("envelope_fill") == ENVELOPE_FILL_DEFAULT:   # F4-55: a value equal to the default; a run at the default has the effective settings (and the hash) it had before the key
        st.pop("envelope_fill", None)
    if st.get("part_only_min_film_mm") is None:   # P4b-6: a run without the key has the effective settings (and the hash) it had before the key existed
        st.pop("part_only_min_film_mm", None)
    if st.get("objective_scale") == OBJECTIVE_SCALE_DEFAULT:   # P3a: the same rule again. `first` stays in: it is the old behavior, and a change from it is a change
        st.pop("objective_scale", None)
    cases = {c.name: dataclasses.asdict(c) for c in config.load_cases}   # every field of a case, also those that a later item adds
    out = {"settings": st, "material": dataclasses.asdict(config.material), "load_cases": cases}
    b = config.buckling
    if b.min_factor is not None:   # the buckling row (section 39) changes the numbers of a pass. Without it the dict is as it was
        from .config import BUCKLING_KS_Q
        out["buckling_row"] = {"min_factor": b.min_factor, "ks_q": BUCKLING_KS_Q if b.ks_q is None else b.ks_q, "modes": b.modes}
    return _jsonable(out)


def region_hashes(config) -> dict:
    """SHA-256 of each region file, as `{"<role>:<path relative to the YAML folder>": hash}`.

    The keys come from `runinfo.region_files` (every field of `config.regions`, so a later kind of region file is covered). The case
    manifest uses the same function, so a checkpoint and a case record give the same key for the same file: the path as the YAML
    names it (a link keeps its own name), and `<role>:<outside>/<12 hex>/<file name>` for a file outside the work folder (`key_root`),
    so the file holds no folder name of the outside place.
    """
    files = region_files(config.regions, Path(config.yaml_path).parent, root=key_root(config.yaml_path), yaml_path=config.yaml_path)
    return {key: sha256_file(path) for key, path in files.items()}


def input_hashes(config, problem=None, solver: str | None = None) -> dict:
    """The inputs of a run as hashes and numbers. `problem` adds the grid and the hash of the voxel labels."""
    eff = effective_settings(config)
    out = {
        "yaml_sha256": sha256_file(config.yaml_path),
        "yaml_parsed": parsed_yaml(config.yaml_path),
        "regions": region_hashes(config),
        "effective": eff,
        "effective_sha256": hashlib.sha256(json.dumps(eff, sort_keys=True).encode()).hexdigest(),
    }
    if solver is not None:
        out["solver"] = solver
    if problem is not None:
        from .voxelize import DESIGN
        out["grid"] = {"shape": [int(s) for s in problem.shape], "n_design": int((problem.labels == DESIGN).sum()),
                       "origin": [float(v) for v in problem.origin], "h": float(problem.h)}
        out["labels_sha256"] = hashlib.sha256(np.ascontiguousarray(problem.labels, dtype=np.uint8).tobytes()).hexdigest()
    return out


# --- the design voxels ---------------------------------------------------------------------

MMA_ASYINIT = ASYINIT   # the first asymptotes are x -/+ ASYINIT * (xmax - xmin) (Svanberg 2007, eq. 3.11)


def design_mask_bits(labels) -> np.ndarray:
    """The design voxels of a grid as packed bits (1 bit for each grid voxel, C order). A checkpoint saves it next to `x`."""
    from .voxelize import DESIGN
    return np.packbits(np.asarray(labels).reshape(-1) == DESIGN)


def design_ids(bits, n_voxels: int) -> np.ndarray:
    """The flat grid ids of the design voxels, in the order of the design variables, from packed bits (`design_mask_bits`)."""
    return np.flatnonzero(np.unpackbits(np.asarray(bits, dtype=np.uint8), count=int(n_voxels)))


@dataclasses.dataclass
class DesignMap:
    """Map the design voxels of a checkpoint onto the design voxels of this run, by position (flat grid id).

    A voxel in both keeps its saved value. A voxel only in this run (`new`) gets the start value. A voxel only in the
    checkpoint (`dropped`) is left out. The grid is the same (shape, voxel size, origin): only the regions that fill it changed.
    """
    n_old: int
    n_new: int
    src: np.ndarray    # index in the saved arrays, of each kept voxel
    dst: np.ndarray    # index in the arrays of this run, of each kept voxel

    @property
    def kept(self) -> int:
        return int(self.src.size)

    @property
    def new(self) -> int:
        return self.n_new - self.kept

    @property
    def dropped(self) -> int:
        return self.n_old - self.kept

    def apply(self, saved, start):
        """`saved` (first axis: the saved design voxels) on the design voxels of this run. `start` fills the new voxels.

        `start` is a number, or an array with the shape of the result.
        """
        saved = np.asarray(saved)
        out = np.array(np.broadcast_to(np.asarray(start, dtype=saved.dtype), (self.n_new,) + saved.shape[1:]))
        out[self.dst] = saved[self.src]
        return out

    def text(self) -> str:
        return (f"{self.kept} of the {self.n_old} saved design voxels are design voxels in this run too (they keep their saved values by "
                f"position); {self.new} design voxels are new (they start at the start value, as in a new run); "
                f"{self.dropped} saved design voxels are no longer design voxels (dropped)")


def map_design(old_ids, new_ids) -> DesignMap:
    """The `DesignMap` from the flat grid ids of the saved design voxels to those of this run (both sorted)."""
    old_ids, new_ids = np.asarray(old_ids), np.asarray(new_ids)
    common = np.intersect1d(old_ids, new_ids, assume_unique=True)
    return DesignMap(int(old_ids.size), int(new_ids.size), np.searchsorted(old_ids, common), np.searchsorted(new_ids, common))


ORIGIN_SHIFT_MAX = 0.5   # voxels. A grid whose origin moved by less than this is the same grid: a saved cell goes onto the nearest cell


def origin_shift(old: dict, new: dict) -> float | None:
    """The largest move of the grid origin along one axis, in voxels, or None if a checkpoint has no origin."""
    o0, o1, h = old.get("origin"), new.get("origin"), new.get("h")
    if o0 is None or o1 is None or not h:
        return None
    return max(abs(float(a) - float(b)) for a, b in zip(o0, o1)) / float(h)


def grid_mismatch(old: dict, new: dict) -> str | None:
    """Why the saved design does not fit the grid of this run, or None. Shape, voxel size and origin (up to `ORIGIN_SHIFT_MAX`)."""
    if [int(v) for v in old.get("shape") or []] != [int(v) for v in new["shape"]]:
        return (f"the checkpoint has {old.get('n_design')} design voxels on the grid {old.get('shape')}, "
                f"and this problem has {new.get('n_design')} on {new['shape']}")
    h0, h1 = old.get("h"), new.get("h")
    if h0 is not None and h1 is not None and abs(float(h0) - float(h1)) > 1e-9 * max(abs(float(h0)), abs(float(h1))):
        return f"the voxel size is {float(h0):g} mm in the checkpoint and {float(h1):g} mm now"
    shift = origin_shift(old, new)
    if shift is not None and shift >= ORIGIN_SHIFT_MAX:
        return (f"the grid origin moved by {shift:.2f} voxels ({old.get('origin')} -> {new.get('origin')} mm): a saved cell would "
                "lie on another place of the part")
    return None


def _short(v) -> str:
    text = json.dumps(v, sort_keys=True) if not isinstance(v, str) else v
    return text if len(text) <= LONG else text[:LONG] + "..."


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _has_mapping(v) -> bool:
    return isinstance(v, dict) or (isinstance(v, list) and any(_has_mapping(i) for i in v))


def _yaml_paths(v, prefix: str = "") -> dict:
    """The leaves of parsed YAML as `{key path: value}`: `settings.volume_fraction`, `load_cases[0].forces[0].force`.

    A list that holds no mapping is one leaf (`beta_schedule`, a force vector). A list of mappings goes by index.
    """
    if isinstance(v, dict) and v:
        out = {}
        for k, x in v.items():
            out.update(_yaml_paths(x, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(v, list) and v and _has_mapping(v):
        out = {}
        for i, x in enumerate(v):
            out.update(_yaml_paths(x, f"{prefix}[{i}]"))
        return out
    return {prefix: v}


# Keys of the YAML that do not change the numbers of a pass: a change is a note, not a change of the problem. They are the same keys that
# `effective_settings` leaves out.
SOFT_KEYS = ("settings.stop_rules", "settings.max_active_voxels", "settings.max_grid_voxels", "settings.fine_pass")
NOT_SET = "<not set>"


def _setting_solver(inputs: dict) -> str | None:
    """`settings.solver` of an `input_hashes` dict (the setting of the YAML: "auto" or a solver name), or None when the dict has no effective settings."""
    return ((inputs.get("effective") or {}).get("settings") or {}).get("solver")


def diff_inputs(old: dict, new: dict, soft: dict | None = None) -> tuple[list[str], list[str]]:
    """Compare two `input_hashes` dicts. Return `(changes, notes)`.

    `changes`: the inputs that decide the numbers and differ. A resume needs the override for these:
    - the parsed YAML, key path by key path (`settings.volume_fraction`, `name`, `mirror`, `load_cases[0].forces[0].force`). All keys count,
      also a key that the loader does not turn into a setting. Not the stop rules and the two voxel guards: they are notes.
    - the effective settings after the defaults, if the parsed YAML is the same (a default of topopt changed), or if one side has no parsed YAML;
    - each region file (`runinfo.region_files` keys), the voxel labels, the solver that runs (not when `settings.solver` is `auto` on both sides:
      `auto` can pick another solver at each start, and during a run when gpu-mg fails, so another pick is a note).
    `notes`: a difference that does not decide the numbers (stop rules and guards; the YAML text with the same parsed content; another pick of `auto`).
    `soft` (P4b-5): `{key path: reason}` of more keys that are notes in this call, in the YAML and in the effective settings. A continuation
    passes `settings.iterations`: the pass count of the new run comes from the parent checkpoint and `--more-iterations`, not from the YAML.
    The YAML line is a note with the reason. The same key of the effective settings is left out (the YAML line says it).
    """
    soft = soft or {}
    changes, notes = [], []
    pa, pb = old.get("yaml_parsed"), new.get("yaml_parsed")
    yaml_changes = 0
    if pa is not None and pb is not None:
        a, b = _yaml_paths(pa), _yaml_paths(pb)
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key, NOT_SET), b.get(key, NOT_SET)
            if va == vb:
                continue
            line = f"{key}: {_short(va)} -> {_short(vb)}"
            if any(key == k or key.startswith(k + ".") for k in SOFT_KEYS):
                notes.append(line + " (this does not change the numbers of a pass)")
            elif any(key == k or key.startswith(k + ".") for k in soft):
                reason = next(r for k, r in soft.items() if key == k or key.startswith(k + "."))
                notes.append(line + f" ({reason})")
            else:
                changes.append(line)
                yaml_changes += 1
    if yaml_changes == 0:   # the same parsed YAML (or none to compare): the effective settings can still differ (another default)
        a, b = _flatten(old.get("effective", {})), _flatten(new.get("effective", {}))
        for key in sorted(set(a) | set(b)):
            if any(key == k or key.startswith(k + ".") for k in soft):
                continue
            if a.get(key, NOT_SET) != b.get(key, NOT_SET):
                changes.append(f"{key}: {_short(a.get(key, NOT_SET))} -> {_short(b.get(key, NOT_SET))}"
                               + (" (the YAML is the same: a default of topopt changed)" if pa is not None and pb is not None else ""))
    ra, rb = old.get("regions", {}), new.get("regions", {})
    gone, fresh = [k for k in sorted(ra) if k not in rb], [k for k in sorted(rb) if k not in ra]
    for key in list(gone):   # the same content under another relative path of the same role: one fact, not two
        twin = next((k for k in fresh if k.split(":", 1)[0] == key.split(":", 1)[0] and rb[k] == ra[key]), None)
        if twin is not None:
            changes.append(f"region file {key} is now {twin}: the content is the same (sha256 {ra[key][:12]}), only the path in the YAML changed")
            gone.remove(key)
            fresh.remove(twin)
    changes += [f"region file {key} is no longer in the YAML" for key in gone]
    changes += [f"region file {key} is new in the YAML" for key in fresh]
    for key in sorted(set(ra) & set(rb)):
        if ra[key] != rb[key]:
            changes.append(f"region file {key} has other content (sha256 {ra[key][:12]} -> {rb[key][:12]})")
    if "solver" in old and "solver" in new and old["solver"] != new["solver"]:
        line = f"the solver that runs: {old['solver']} -> {new['solver']}"
        if _setting_solver(old) == "auto" and _setting_solver(new) == "auto":   # P9-4d: `auto` can pick another solver (a fallback, a GPU that is gone)
            notes.append(line + " (settings.solver is auto in both runs, so this is another pick of auto and not a change of the inputs. "
                                "The numbers of the passes can differ in the last digits)")
        else:
            changes.append(line)
    if "labels_sha256" in old and "labels_sha256" in new and old["labels_sha256"] != new["labels_sha256"]:
        changes.append("the voxel labels are not the same (the regions fill other voxels)")
    if old.get("yaml_sha256") != new.get("yaml_sha256") and not changes and not notes:
        notes.append("the YAML text changed (comments or format only): the parsed content and the effective settings are the same")
    return changes, notes


# --- the file --------------------------------------------------------------------------------

def _flush(fd: int) -> None:
    """Flush a file to the drive. `fcntl.F_FULLFSYNC` where it exists (macOS: `fsync` does not flush the drive cache), else `os.fsync`."""
    try:
        import fcntl
        full = getattr(fcntl, "F_FULLFSYNC", None)
    except ImportError:   # no fcntl on this system
        full = None
    if full is not None:
        try:
            fcntl.fcntl(fd, full)
            return
        except OSError:
            pass   # this file system does not do it (for example a network drive): `fsync` is the next best
    os.fsync(fd)


def _flush_folder(path: Path) -> None:
    """After the renames: flush the folder, so that the new names are on the drive too. A system that cannot do it is skipped."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        _flush(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def is_array_ref(value) -> bool:
    return isinstance(value, dict) and set(value) == {ARRAY_REF}


def pack_rows(rows: dict | None, arrays: dict) -> dict | None:
    """`rows` (`MmaState.rows`: `{row name: {key: value}}`) as JSON data. A numpy array that is the value of a key of a row state is put in `arrays`
    and replaced by `{"npz_array": name}` (`ROW_ARRAY` + a number). An equal array that is in `arrays` already keeps its name, so `mma.rows` and
    `rows_before` write one array for a row that did not change. Only the values of the first level are looked at. `rows` is not changed.
    """
    if rows is None:
        return None
    out = {}
    for name, state in rows.items():
        if not isinstance(state, dict):
            out[name] = state
            continue
        out[name] = {key: {ARRAY_REF: _row_array_name(value, arrays)} if isinstance(value, np.ndarray) else value
                     for key, value in state.items()}
    return out


def _row_array_name(value, arrays: dict) -> str:
    a = np.ascontiguousarray(value)
    for name, b in arrays.items():
        if name.startswith(ROW_ARRAY) and b.dtype == a.dtype and np.array_equal(a, b):
            return name
    name = f"{ROW_ARRAY}{sum(n.startswith(ROW_ARRAY) for n in arrays)}"
    arrays[name] = a
    return name


def row_array_refs(rows) -> list:
    """The names of the arrays that `rows` (the JSON form that `pack_rows` makes) refers to."""
    if not isinstance(rows, dict):
        return []
    return [v[ARRAY_REF] for state in rows.values() if isinstance(state, dict) for v in state.values() if is_array_ref(v)]


def unpack_rows(rows: dict, arrays: dict) -> dict:
    """A copy of `rows` (the JSON form) with each reference replaced by a copy of its array from `arrays` (the other values are copied too)."""
    return {name: {key: np.array(arrays[v[ARRAY_REF]]) if is_array_ref(v) else copy.deepcopy(v) for key, v in state.items()}
            if isinstance(state, dict) else copy.deepcopy(state) for name, state in rows.items()}


def write_checkpoint(run_dir, arrays: dict, meta: dict) -> int:
    """Write the checkpoint atomically. The old file becomes `checkpoint.prev.npz`. Return the size in bytes.

    With `FSYNC`: the file is flushed to the drive before the renames, and the folder after them.
    """
    d = Path(run_dir)
    tmp, cur = d / TMP_FILE, d / CHECKPOINT_FILE
    text = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    with open(tmp, "wb") as f:
        np.savez(f, **{META: text}, **arrays)
        f.flush()
        if FSYNC:
            _flush(f.fileno())
    size = tmp.stat().st_size
    if cur.exists():
        os.replace(cur, d / PREV_FILE)
    os.replace(tmp, cur)
    if FSYNC:
        _flush_folder(d)
    return size


@dataclasses.dataclass
class Checkpoint:
    """A checkpoint that was read and checked. `file` is the file name, `bad` the files that were not valid (name: reason).

    `sha256` is the hash of the bytes that were read (the same open file), not of the file name read again (P4b-5, fix round 1).
    """
    meta: dict
    arrays: dict
    file: str
    facts: list
    bad: dict
    sha256: str | None = None

    @property
    def iteration(self) -> int:
        return int(self.meta["iteration"])


def _load_file(path: Path) -> tuple[dict, dict, str]:
    """Read one checkpoint file fully (this checks the CRC of each array): `(meta, arrays, sha256)`.

    The hash is of the same open file that was read, so it names the bytes of these arrays, also when a running parent replaces the file
    by name after the read. Raise ValueError with a reason if the file is not valid.
    """
    try:
        with open(path, "rb") as f:
            with np.load(f, allow_pickle=False) as z:
                names = list(z.files)
                meta = json.loads(bytes(z[META]).decode()) if META in names else None
                arrays = {n: z[n] for n in names if n != META}
            f.seek(0)
            digest = sha256_handle(f)
    except Exception as exc:   # any failure to read means: this file cannot be used (a cut file, a bad CRC, not a zip)
        raise ValueError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("kind") != KIND:
        raise ValueError(f"the file is not a topopt checkpoint (no '{META}' array with kind '{KIND}')")
    if meta.get("format") != FORMAT:
        raise ValueError(f"unknown checkpoint version {meta.get('format')!r} (this topopt reads version {FORMAT})")
    for key in ("iteration", "n_design", "mma", "inputs"):
        if key not in meta:
            raise ValueError(f"the checkpoint has no '{key}'")
    _check_core(meta, arrays)
    return meta, arrays, digest


def _check_core(meta: dict, arrays: dict) -> None:
    """Raise ValueError with the reason if a field that a resume reads is missing or has the wrong form.

    The reader calls it for each file, so a file that fails goes to the older file, or to a refusal with the reason,
    before anything in the run folder changes. The fields of other owners (`model__...`, `solver__...`) are not core: a
    missing one gives a fact at the restore.
    """
    def need(ok, why):
        if not ok:
            raise ValueError(why)

    def array(name, shape):
        a = arrays.get(name)
        need(a is not None, f"the array '{name}' is missing")
        need(a.shape == shape, f"the array '{name}' has the shape {a.shape}, expected {shape}")
        return a

    try:
        n = int(meta["n_design"])
        need(isinstance(meta["iteration"], int) and meta["iteration"] >= 0, "'iteration' is not a whole number")
    except (TypeError, ValueError):
        raise ValueError("'n_design' or 'iteration' is not a number") from None
    m = meta["mma"]
    need(isinstance(m, dict), "'mma' is not a mapping")
    for key in ("it", "f0_ref", "rows", "move", "x"):
        need(key in m, f"the checkpoint has no 'mma.{key}'")
    need(isinstance(m["rows"], dict), "'mma.rows' is not a mapping")
    for where, rows in (("mma.rows", m["rows"]), ("rows_before", meta.get("rows_before"))):     # the arrays of the row states (#155)
        for ref in row_array_refs(rows):
            need(ref in arrays, f"the array '{ref}' of '{where}' is missing")
    need(m["x"] in ("none", "same", "saved"), f"'mma.x' is {m['x']!r}, expected none, same or saved")
    array("x", (n,))
    array("mma_low", (n, 1))
    array("mma_upp", (n, 1))
    if int(m["it"]) >= 1:   # after the first step the MMA state has the two designs before
        array("mma_xold1", (n,))
        array("mma_xold2", (n,))
    if m["x"] == "saved":
        array("mma_x", (n,))
    sa = meta.get("step_acceptance")
    need(sa is None or (isinstance(sa.get("counters"), list) and len(sa["counters"]) == 4), "'step_acceptance.counters' is not a list of 4")
    need(meta.get("prev") is None or isinstance(meta["prev"], dict), "'prev' is not a history record")
    need(isinstance(meta["inputs"], dict), "'inputs' is not a mapping")
    grid = meta["inputs"].get("grid")
    need(isinstance(grid, dict) and isinstance(grid.get("shape"), list) and len(grid["shape"]) == 3,
         "the checkpoint has no grid shape ('inputs.grid')")
    mask = arrays.get("design_mask")
    need(mask is not None and mask.dtype == np.uint8 and mask.ndim == 1, "the array 'design_mask' is missing or not packed bits")
    n_vox = int(np.prod([int(v) for v in grid["shape"]]))
    need(mask.size * 8 >= n_vox and int(np.unpackbits(mask, count=n_vox).sum()) == n,
         f"the design voxels of 'design_mask' do not match 'n_design' ({n}) on the grid {grid['shape']}")


RESTART_OPTION = "run without --resume: that starts a new run and deletes the old run files"
CONTINUE_RESTART_OPTION = ("run the YAML as a new run in a new folder, with the full settings.iterations (this folder has no checkpoint "
                           "that a continuation can use)")


def read_checkpoint(run_dir, restart: str = RESTART_OPTION) -> Checkpoint:
    """The last valid checkpoint of a run folder. `checkpoint.npz` first, then `checkpoint.prev.npz`.

    Raise `ResumeRefused` if there is none. The text gives the reason for each file and the options. `restart` is the last option
    of the refusal (a continuation names its own: the parent folder is not deleted).
    """
    d = Path(run_dir)
    problems, found = {}, None
    for name in (CHECKPOINT_FILE, PREV_FILE):
        path = d / name
        if not path.exists():
            continue
        try:
            found = (name,) + _load_file(path)
            break
        except ValueError as exc:
            problems[name] = str(exc)
    if found is None:
        if not problems:
            raise ResumeRefused(
                f"There is no checkpoint in {d} (no {CHECKPOINT_FILE}, no {PREV_FILE}). A run writes one after each pass, so "
                "this run did not finish its first pass, or the folder is not a run folder.",
                options=[restart])
        unknown = [n for n, r in problems.items() if r.startswith("unknown checkpoint version")]
        why = "; ".join(f"{n}: {r}" for n, r in problems.items())
        raise ResumeRefused(
            f"The checkpoint of {d} cannot be used: {why}.", facts=[f"{n}: {r}" for n, r in problems.items()],
            options=(["use the topopt version that wrote the checkpoint (see its `topopt_version`), or "] if unknown else [])
            + [restart])
    name, meta, arrays, digest = found
    facts = [f"resume: {n} cannot be used ({r})" for n, r in problems.items()]
    if problems:
        facts.append(f"resume: used {name} (iteration {meta['iteration']})")
    return Checkpoint(meta, arrays, name, facts, problems, digest)


def describe(run_dir) -> dict | None:
    """Short facts about the checkpoint of a folder for `run_status`, or None if the folder has none.

    It reads the JSON text only, so it is cheap. It does not check the arrays (`read_checkpoint` does).
    """
    d = Path(run_dir)
    for name in (CHECKPOINT_FILE, PREV_FILE):
        path = d / name
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(bytes(z[META]).decode())
            return {"file": name, "iteration": meta["iteration"], "iterations": meta.get("n_iterations"),
                    "complete": bool(meta.get("complete")), "written_at": meta.get("written_at"),
                    "elapsed_s": meta.get("elapsed_s"), "topopt_version": meta.get("topopt_version"),
                    "format": meta.get("format"), "stop_rule": (meta.get("stop") or {}).get("rule"),
                    "resume_points": len(meta.get("resume_points") or []),
                    **({"continued_from": meta["continued_from"]} if meta.get("continued_from") else {}),
                    **({"fine_from": meta["fine_from"]} if meta.get("fine_from") else {})}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            continue
    return None


# --- the resume plan -------------------------------------------------------------------------

@dataclasses.dataclass
class Continuation:
    """P4b-5: what a continuation (`topopt run --continue-from PARENT --more-iterations N`) takes from the parent run folder.

    `k` is the pass of the parent checkpoint. The new run has `iterations` = `k` + `more` passes, as a run of that length has.
    `replay`: the checkpoint is a complete one (the parent ended at pass `k`, which only measured). The new run runs pass `k` again,
    with an update step (`last=False`). Else the parent did not finish: the new run is a fork that starts at pass `k` + 1, and no pass is run again.
    `rows_before`: the checkpoint holds the rows before pass `k` (a replay needs them). `record` is the `continued_from` value of run.json.
    """
    parent: Path
    k: int
    more: int
    iterations: int
    replay: bool
    rows_before: bool
    record: dict


@dataclasses.dataclass
class ResumePlan:
    """What a resume found and will do. `apply` changes the run folder."""
    ck: Checkpoint
    facts: list                  # lines for run.log, each starts with `resume:` (`continue:` for a continuation)
    changes: list                # inputs that changed (empty if none). Non-empty only with the override.
    resume_points: list = dataclasses.field(default_factory=list)   # earlier points and this one, after `apply`
    design_map: DesignMap | None = None   # the regions fill other design voxels: map the saved arrays by position (None: the same voxels)
    continued_from: dict | None = None    # P4b-5: the `continued_from` record of a continuation, and of the resume of one (run.json and each checkpoint)
    cont: Continuation | None = None      # P4b-5: set by `plan_continue` only
    fine_from: dict | None = None         # #318: the `fine_from` record of a fine pass, and of the resume of one (run.json and each checkpoint)

    @property
    def replay(self) -> bool:
        """True for a continuation of a complete checkpoint: pass `ck.iteration` runs again."""
        return self.cont is not None and self.cont.replay

    @property
    def kept_iteration(self) -> int:
        """The last pass whose history line, snapshot and timing line the run folder keeps. A replay runs the checkpoint pass again."""
        return self.ck.iteration - 1 if self.replay else self.ck.iteration

    @property
    def kept_complete(self) -> bool:
        """True if the kept files end with the complete checkpoint of a finished run (a replay keeps no pass of it)."""
        return bool(self.ck.meta.get("complete")) and not self.replay


def plan_resume(run_dir, config, problem=None, solver: str | None = None, allow_changed: bool = False) -> ResumePlan:
    """Read the checkpoint and compare it with this run. Write nothing. Raise `ResumeRefused` if the run cannot go on.

    `problem` and `solver` (the solver that runs) are optional: without them the grid, the design voxels and the voxel labels
    are not compared. A change of an input that decides the numbers is refused unless `allow_changed` (the override).
    Some cases cannot work with any override, and the text gives the reason and the options:
    a checkpoint that cannot be read or has an unknown version, another grid (shape, voxel size, or an origin that moved by half a
    voxel or more), a checkpoint of a finished run with another `settings.iterations`, a checkpoint at or past the last pass
    of an unfinished run. On the same grid, regions that fill other voxels are a change: with the override the saved arrays go onto
    the new design voxels by position (`ResumePlan.design_map`).
    """
    ck = read_checkpoint(run_dir)
    meta = ck.meta
    it, n_new = ck.iteration, int(config.settings.iterations)
    new_inputs = input_hashes(config, problem, solver)
    old_inputs = meta["inputs"]
    old_grid, new_grid = old_inputs["grid"], new_inputs.get("grid")
    if new_grid is not None:
        why = grid_mismatch(old_grid, new_grid)
        if why:
            raise ResumeRefused(
                f"Resume cannot work: {why}. The saved design does not fit the new grid.",
                [], ["restore the YAML and the STL files of the run (`yaml_path` in run.json)",
                     "run without --resume: that starts a new run and deletes the old run files"])
    if meta.get("complete") and n_new != int(meta.get("n_iterations", n_new)):
        raise ResumeRefused(
            f"A resume of a finished run repeats its post step only. The checkpoint is from a run that finished at pass {it} "
            f"(settings.iterations was {meta.get('n_iterations')}, now {n_new}). This version does not make a finished run longer.",
            [], ["use settings.iterations equal to the old value (the run then repeats its post step only)",
                 "topopt run <yaml> --out <new folder> --continue-from <this folder> --more-iterations N (MCP: run_topopt with continue_from and "
                 "more_iterations): a new run folder that adds N passes to this finished run. This folder does not change",
                 "run without --resume in a new folder with the larger settings.iterations"])
    if not meta.get("complete") and it >= n_new:
        raise ResumeRefused(
            f"Resume cannot work: the checkpoint is at pass {it}, and settings.iterations is {n_new}. No pass is left, and pass {it} "
            "was not the last pass of the run that wrote it.",
            [], [f"use settings.iterations above {it}", "run without --resume: that starts a new run and deletes the old run files"])
    cont_from = meta.get("continued_from")   # P4b-5: the checkpoint is of a continuation; its pass count is the one of the continuation
    fine_from = meta.get("fine_from")        # #318: the checkpoint is of a fine pass; `config_of_fine` gave this config its voxel, maximum and schedule
    changes, notes = diff_inputs(old_inputs, new_inputs, soft=CONTINUED_SOFT if cont_from else None)
    design_map = None
    if problem is not None:   # the design voxels of the checkpoint and of this run, by position on the same grid
        from .voxelize import DESIGN
        old_ids = design_ids(ck.arrays["design_mask"], int(np.prod(old_grid["shape"])))
        new_ids = np.flatnonzero(np.asarray(problem.labels).reshape(-1) == DESIGN)
        if not np.array_equal(old_ids, new_ids):
            design_map = map_design(old_ids, new_ids)
            changes.append("the design voxels are not the same: " + design_map.text())
    facts = [f"resume: continue from {ck.file}, iteration {it} of {meta.get('n_iterations')} (written {meta.get('written_at')}, "
             f"topopt {meta.get('topopt_version')})." + ("" if meta.get("complete") else f" The first pass to run is {it + 1}.")]
    facts += list(ck.facts)
    if cont_from:
        facts.append(_continued_resume_fact(cont_from, new_inputs, config))
    if fine_from:
        state, recorded = bool(config.settings.stop_rules.enabled), fine_early_stop(fine_from, cont_from)[0]
        facts.append(_fine_resume_fact(fine_from, new_inputs, int(config.settings.iterations), state, recorded))
        if state != recorded:   # the flag of this resume changed the state: the records of the resumed run, and of any later resume or continuation, carry it
            from .fine_pass import SETTLED_RULE
            fine_from = dict(fine_from, stop_rule=SETTLED_RULE if state else None)
            if cont_from:
                ss = dict(cont_from.get("step_settings") or {})
                ss["stop_rules"] = dict(ss.get("stop_rules") or {}, enabled=state)
                cont_from = dict(cont_from, step_settings=ss)
    if meta.get("complete"):
        facts.append(f"resume: the checkpoint is from the last pass of a finished run"
                     + (f" (stop rule '{meta['stop']['rule']}')" if meta.get("stop") else "")
                     + ". No pass is left to run. The post step runs again.")
    stopped = _stopped_fact(run_dir, ck)
    if stopped:
        facts.append(stopped)
    if meta.get("topopt_version") != __version__:
        facts.append(f"resume: the checkpoint was written by topopt {meta.get('topopt_version')}; this is topopt {__version__}")
    if meta.get("platform") and meta["platform"] != _platform():
        facts.append(f"resume: the checkpoint was written on {meta['platform']}; this is {_platform()}. "
                     "The numbers can differ in the last digits.")
    for gpu_fact in (_gpu_mg_fact(meta, config, solver, changes), _gpu_max_iter_fact(meta, config, solver, changes)):
        if gpu_fact:
            facts.append(gpu_fact)
    scale_fact = _objective_scale_fact(meta, config)
    if scale_fact:
        facts.append(scale_fact)
    shift = None if new_grid is None else origin_shift(old_grid, new_grid)
    if shift:
        facts.append(f"resume: the grid origin moved by {shift:.3f} voxels (less than {ORIGIN_SHIFT_MAX:g}). The saved design goes onto "
                     "the cells with the same index.")
    facts += [f"resume: {n}" for n in notes]
    if changes:
        listing = "; ".join(changes)
        if not allow_changed:
            raise ResumeRefused(
                f"Resume stopped: the inputs changed since the checkpoint (iteration {it}): {listing}. "
                "The resumed run would not be the same run.",
                changes, ["give --allow-changed-inputs (MCP: allow_changed_inputs=true) to continue with the new inputs; "
                          "the run log and the verdict then name the changes",
                          "restore the old inputs and resume",
                          *_objective_scale_options(changes, meta),
                          "run without --resume: that starts a new run and deletes the old run files"])
        facts.append(f"resume: the inputs changed since the checkpoint: {listing}. The override is given, so the run goes on "
                     "with the new inputs.")
        from .resume_fact import pass_key_note   # P7: a finished run runs no pass, so a changed key of the passes changes no design
        note = pass_key_note(changes, bool(meta.get("complete")), labels_compared=problem is not None and "labels_sha256" in old_inputs)
        if note:
            facts.append("resume: " + note["fact"])
    return ResumePlan(ck, facts, changes, list(meta.get("resume_points") or []), design_map, continued_from=cont_from, fine_from=fine_from)


def _stopped_fact(run_dir, ck: Checkpoint, continuation: bool = False) -> str | None:
    """P4b-4: one `resume:` fact when a request to stop ended the run of this folder at this checkpoint, else None.

    `continuation` (P4b-5): the fact is for a plan of `--continue-from`. A request after the last pass is then no "post step runs now": the
    new run runs that pass again, and it reads the request files of its own folder only.

    The sign is `status.json` with the state `stopped` at the pass of the checkpoint (a resume has not yet written it again: it comes
    after the plan). A stop in the optimizer loop also has a verdict `stopped_by_request`, which names who asked and when.
    """
    try:
        status = json.loads((Path(run_dir) / "status.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(status, dict) or status.get("state") != "stopped" or status.get("iteration") != ck.iteration:
        return None
    from .verdict import read_verdict
    verdict = read_verdict(run_dir) or {}
    req = verdict.get("stop_request") if verdict.get("state") == "stopped_by_request" else None
    by = f" (requested by {req.get('by')} at {req.get('requested_at')})" if isinstance(req, dict) else ""
    if ck.meta.get("complete"):
        if continuation:
            return (f"continue: the parent was stopped by a request after its last pass{by}. This run runs that pass again and goes on. "
                    "The request does not stop this run.")
        return f"resume: the run was stopped by a request after its last pass{by}. The optimizer is done; the post step runs now."
    return f"resume: the run was stopped by a request at pass {ck.iteration}{by}."


def _gpu_mg_fact(meta: dict, config, solver: str | None, changes: list[str]) -> str | None:
    """G2: one `resume:` fact when the stop numbers of gpu-mg are not those of the part that wrote the checkpoint.

    `effective_settings` leaves `gpu_tol` and `gpu_backward_error` out at their defaults, so the input comparison cannot show that a
    default changed, or that the checkpoint is from before the two settings existed (then the first part ran at the fixed target 1e-8
    and without the backward-error rule). A gpu-mg run records both numbers in `meta["gpu_mg"]` (`GpuModel.checkpoint_meta`).
    A change that the YAML shows is in `changes` already: a checkpoint with the record gets no second fact for it. A checkpoint from
    before G2 (no record) gets its fact next to the change line: that line says `<not set>`, and only the fact says how the first part ran.
    """
    if solver != "gpu-mg":
        return None
    st = config.settings
    now = f"settings.gpu_tol {st.gpu_tol:g} and settings.gpu_backward_error {st.gpu_backward_error:g}"
    old = meta.get("gpu_mg")
    if old is None:
        if meta.get("inputs", {}).get("solver") != "gpu-mg":     # a checkpoint of another solver: the change of solver is a change already
            return None
        return ("resume: the checkpoint was written before settings.gpu_tol and settings.gpu_backward_error existed (G2). Its first part ran "
                f"at the fixed target 1e-8 and without the backward-error rule; this part runs at {now}. Option: gpu_tol: 1e-8 and "
                "gpu_backward_error: 0 in the YAML give the old behavior (the resume then needs --allow-changed-inputs)")
    if any(c.startswith(("settings.gpu_tol", "settings.gpu_backward_error")) for c in changes):
        return None
    was = (old.get("tol"), old.get("backward_error"))
    if was == (st.gpu_tol, st.gpu_backward_error):
        return None
    return (f"resume: the first part ran at settings.gpu_tol {was[0]:g} and settings.gpu_backward_error {was[1]:g}; this part runs at {now} "
            "(the YAML is the same: a default of topopt changed)")


def _objective_scale_fact(meta: dict, config, pass_left: bool = False) -> str | None:
    """P3a: one `resume:` fact when the checkpoint is from before `settings.objective_scale` (the build that wrote it has no such setting).

    `effective_settings` leaves the key out at its default `stage`, so the input comparison cannot show that the checkpoint is from before
    the key. A checkpoint of this build records the value in `meta["objective_scale"]`. The first part of an older run kept the scale of pass 0
    for the whole run, which is `first`. A checkpoint of a finished run has no pass left, so it gets no fact. A continuation (`pass_left`) runs
    passes after the checkpoint pass of a finished run, so it gets the fact too.
    """
    if "objective_scale" in meta or (meta.get("complete") and not pass_left):
        return None
    if config.settings.objective_scale == "first":
        return ("resume: the checkpoint was written before settings.objective_scale existed (P3a). Its first part kept the objective scale of pass 0, "
                "and settings.objective_scale: first does the same, so the run goes on as the old build did")
    return ("resume: the checkpoint was written before settings.objective_scale existed (P3a). Its first part kept the objective scale of pass 0; "
            "this part takes the scale again at the first pass of each later beta stage (settings.objective_scale: stage, the default). "
            "Option: settings.objective_scale: first in the YAML keeps the old behavior (the resume then needs --allow-changed-inputs)")


def _objective_scale_options(changes: list[str], meta: dict) -> list[str]:
    """P3a: the option that the refusal names when the key `settings.objective_scale` changed (an empty list otherwise)."""
    if not any(c.startswith("settings.objective_scale:") for c in changes):
        return []
    if "objective_scale" not in meta:     # a checkpoint of a build before P3a: its run kept the scale of pass 0, which is `first`
        return ["the checkpoint is from before settings.objective_scale: take the key out of the YAML to resume with stage (a resume: fact says what changes), "
                "or give --allow-changed-inputs to resume with first, which is what the old build did"]
    return ["settings.objective_scale: first in the YAML keeps the scale rule of the checkpoint (a key that is not set means stage); "
            "with --allow-changed-inputs and stage, the scale of the checkpoint stays until the next beta stage start, and then each stage takes its own"]


def _gpu_max_iter_fact(meta: dict, config, solver: str | None, changes: list[str]) -> str | None:
    """P2b: one `resume:` fact when the first-pass cap of gpu-mg is not the one of the part that wrote the checkpoint.

    The cap (`settings.gpu_max_iter`) does not change the target of a solve, only the path to it, so it is a fact and not a refusal.
    `effective_settings` leaves it out at its default, so only the record `meta["gpu_mg"]["max_iter"]` shows that a default changed. A
    checkpoint of G2 has the record without this key: its first part ran with the cap that was fixed then (`PRE_P2B_FIRST_PASS_CAP`).
    A checkpoint from before G2 (no record) is the case of `_gpu_mg_fact`. A change that the YAML shows is in `changes` already.
    """
    old = meta.get("gpu_mg")
    if solver != "gpu-mg" or old is None or any(c.startswith("settings.gpu_max_iter") for c in changes):
        return None
    now = config.settings.gpu_max_iter
    if "max_iter" not in old:
        was, why = PRE_P2B_FIRST_PASS_CAP, " (the checkpoint was written before settings.gpu_max_iter existed)"
    else:
        was, why = old["max_iter"], " (the YAML is the same: a default of topopt changed)"
    if was == now:
        return None
    return (f"resume: the first part of the run capped the first PCG pass of a gpu-mg solve at {was} iterations; this part runs at "
            f"settings.gpu_max_iter {now}{why}. The cap changes the path of a solve, not its target (case B only)")


def _platform() -> str:
    import platform
    return f"{sys.platform} {platform.machine()}"


def started_at(run_dir) -> str | None:
    """`started_at` of the `run.json` in a run folder (the start of the first run), or None."""
    try:
        value = json.loads((Path(run_dir) / "run.json").read_text()).get("started_at")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _select_lines(raw: bytes, last_iteration: int) -> dict:
    """The lines of a JSON lines file (bytes) with `iteration <= last_iteration`, byte for byte. Facts as numbers.

    Keys: `keep` (the lines, each ends with a newline), `last` (the iteration of the last kept line or None), `dropped` (the iteration of each
    line that is after `last_iteration`), `bad_lines` (lines that do not parse), `newline_added` (the last kept line had no newline).
    """
    keep, last, dropped, bad, added = [], None, [], 0, False
    for line in raw.splitlines(keepends=True):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            n = int(row["iteration"])
        except (ValueError, KeyError, TypeError):
            bad += 1
            continue
        if n <= last_iteration:
            keep.append(line if line.endswith(b"\n") else line + b"\n")
            last = n
            added = not line.endswith(b"\n")   # of the last line that is kept
        else:
            dropped.append(n)
    return {"keep": keep, "last": last, "dropped": dropped, "bad_lines": bad, "newline_added": added}


def cut_history(run_dir, last_iteration: int, name: str = "history.jsonl") -> dict:
    """Keep the history lines up to `last_iteration`, byte for byte. Return facts as numbers (see `ResumePlan.apply`).

    A line that does not parse (a partial last line, or a damaged line) is dropped. A missing file is a fact too.
    The file is replaced atomically, and only when a line is dropped or the last kept line has no newline (`newline_added`).
    `name` is another JSON lines file of the run folder with an `iteration` in each line (the timing file `gpu_mg.jsonl`
    of the solver gpu-mg is cut the same way).
    """
    path = Path(run_dir) / name
    out = {"present": path.exists(), "kept": 0, "last_kept": None, "dropped": [], "bad_lines": 0, "newline_added": False}
    if not out["present"]:
        return out
    sel = _select_lines(path.read_bytes(), last_iteration)
    out.update(kept=len(sel["keep"]), last_kept=sel["last"], dropped=sel["dropped"], bad_lines=sel["bad_lines"],
               newline_added=sel["newline_added"])
    # A last kept line that parses but has no newline (the kill came between the text and the newline) gets its newline: the next
    # line that a resumed run appends must start on a new line (D1a follow-up: cut the history back to the last newline).
    # A partial line after the last kept line is dropped, and then there is nothing to add.
    if out["dropped"] or out["bad_lines"] or out["newline_added"]:
        write_atomic(path, b"".join(sel["keep"]))
    return out


def copy_lines(src, dst, last_iteration: int) -> dict:
    """Write the lines of the JSON lines file `src` with `iteration <= last_iteration` to `dst`, byte for byte (P4b-5). `src` is only read.

    Return the same numbers as `cut_history` (`present`, `kept`, `last_kept`, `dropped`, `bad_lines`, `newline_added`) and `bytes` written.
    `dst` is written atomically, also when no line is kept (an empty file).
    """
    src = Path(src)
    out = {"present": src.exists(), "kept": 0, "last_kept": None, "dropped": [], "bad_lines": 0, "newline_added": False, "bytes": 0}
    if not out["present"]:
        return out
    sel = _select_lines(src.read_bytes(), last_iteration)
    data = b"".join(sel["keep"])
    write_atomic(Path(dst), data)
    out.update(kept=len(sel["keep"]), last_kept=sel["last"], dropped=sel["dropped"], bad_lines=sel["bad_lines"],
               newline_added=sel["newline_added"], bytes=len(data))
    return out


def _snapshot_number(p: Path):
    stem = p.name
    if stem.startswith("it_") and stem.endswith(".bin") and stem[3:-4].isdigit():
        return int(stem[3:-4])
    return None


def apply_plan(plan: ResumePlan, out, reason: str | None = None) -> None:
    """Change the run folder for a resume and write the facts to `run.log`. `reason` (optional) goes into the resume point as `reason`
    (`solver_fallback` for the restart of a run after gpu-mg failed: the resume point then says that no user resumed the run).

    1. Remove the files that cannot be used: a part file of a checkpoint write and a checkpoint that failed the check.
    2. Cut `history.jsonl` to the checkpoint iteration. Remove the snapshots after it.
    3. Remove `result/`, `verdict.json` and the other result folders of the run (`snapshot.RUN_DIRS`: `verify/` after #21) of the earlier run.
       The post step makes them again.
    4. Set the run time of `out` to the saved run time (`elapsed_s` goes on, the time that the run was not running does not count).
    """
    d, ck = Path(out.dir), plan.ck
    it = ck.iteration
    for line in plan.facts:   # the plan first: a resume that stops inside this function still leaves its facts in run.log
        out.log(line)
    out.log(f"resume: the run folder changes now. history.jsonl and the snapshots are cut to iteration {it}; result/ and verdict.json "
            "are removed (the post step makes them again). The facts of the cut follow when it is done.")
    notes = []
    unusable = [n for n, r in ck.bad.items() if not r.startswith("unknown checkpoint version")]   # a newer file stays
    for name in [TMP_FILE] + unusable:
        if (d / name).exists():
            (d / name).unlink()
            notes.append(f"removed {name} (a part file or a file that cannot be used)")
    cut = cut_history(d, it)
    entry = {"iteration": it, "checkpoint": ck.file, "resumed_at": _iso(), "dropped_history_iterations": cut["dropped"],
             "dropped_snapshots": [], "changes": list(plan.changes)}
    if reason:
        entry["reason"] = reason
    if cut["dropped"] or cut["bad_lines"]:
        what = []
        if cut["dropped"]:
            what.append(f"history lines for iterations {min(cut['dropped'])} to {max(cut['dropped'])}")
        if cut["bad_lines"]:
            what.append(f"{cut['bad_lines']} line(s) of history.jsonl that do not parse (a partial last line, or a damaged line)")
        notes.append("dropped " + " and ".join(what))
    if cut["newline_added"]:
        notes.append("the last line of history.jsonl had no newline (the kill came before it): the newline is added")
    if not cut["present"]:
        notes.append("history.jsonl is missing")
    elif cut["last_kept"] is None or cut["last_kept"] < it:
        notes.append(f"history.jsonl ends at iteration {cut['last_kept']}, before the checkpoint iteration {it}: "
                     "the rows in between are not in the file")
    snaps = d / "snapshots"
    snaps.mkdir(exist_ok=True)
    gone = []
    for p in sorted(snaps.iterdir()):
        n = _snapshot_number(p)
        if p.name.endswith(".tmp") or (n is not None and n > it):
            if n is not None:
                gone.append(n)
            p.unlink()
    entry["dropped_snapshots"] = gone
    if gone:
        notes.append(f"dropped snapshots {min(gone)} to {max(gone)}")
    missing = [n for n in range(it + 1) if not (snaps / f"it_{n:04d}.bin").exists()]
    if missing:
        notes.append(f"snapshots missing in the folder: {missing[:5]}" + (" ..." if len(missing) > 5 else ""))
    from . import snapshot
    # What the post step and later tools made of the earlier final.stl: `result/`, and any other folder of the run (`verify/`). `snapshots/` is cut above.
    for p in [d / name for name in snapshot.RUN_DIRS if name != "snapshots"] + [d / "verdict.json"]:
        if p.is_dir():
            shutil.rmtree(p)
            notes.append(f"removed {p.name}/ of the earlier run")
        elif p.exists():
            p.unlink()
    out.t_start = time.time() - float(ck.meta.get("elapsed_s") or 0.0)
    plan.resume_points = plan.resume_points + [entry]
    if notes:
        out.log("resume: " + ". ".join(n[0].upper() + n[1:] for n in notes) + ".")


# --- continue a finished run (P4b-5, CONTRACTS.md section 24.3) -------------------------------

CONTINUED_SOFT = {"settings.iterations": "the pass count of a continuation comes from the parent checkpoint and --more-iterations, "
                                         "not from the YAML"}
TIMING_FILE = "gpu_mg.jsonl"   # the timing file of the solver gpu-mg: a literal, as in `snapshot.RUN_FILES` (this module must not import `topopt.gpu_mg`)


def with_iterations(config, iterations: int):
    """The config with `settings.iterations` replaced. A continuation runs `K + N` passes, whatever the YAML says."""
    return dataclasses.replace(config, settings=dataclasses.replace(config.settings, iterations=int(iterations)))


def check_more_iterations(value) -> int:
    """`--more-iterations` and `run_topopt(more_iterations=...)`: a whole number of 1 or more, with no upper limit. Raise `UserError`."""
    n = None
    if isinstance(value, int) and not isinstance(value, bool):
        n = value
    elif isinstance(value, str):
        try:
            n = int(value.strip()) if value.strip().lstrip("+-").isdecimal() else None   # int() also fails for "--5"
        except ValueError:
            n = None
    if n is None or n < 1:
        raise UserError(f"--more-iterations must be a whole number of 1 or more (the number of passes to add), got {value!r}. "
                        "Options: give a number such as 20 to add 20 passes; there is no upper limit.")
    return n


def _same_place(a, b) -> bool:
    """True if two paths that exist are one file system object (`os.path.samefile`: same device and inode). False if one is gone."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def check_new_folder(parent, new) -> None:
    """Raise `UserError` with options if the new run folder would change the parent run folder (CONTRACTS.md section 24.3).

    This is the one hard block of a continuation: a run in the parent folder would delete or change the parent. The cases are:
    the new folder is the parent, the new folder is inside the parent, or the parent is inside a folder that a new run deletes
    (`snapshot.RUN_DIRS` of the new folder). The folders are compared by identity (`os.path.samefile`) and not by the text of the path:
    a path in another letter case on macOS, a bind mount and a link name the same folder. For a new folder that does not exist yet, the
    deepest folder of its path that exists and each folder above it are compared with the parent. The parent is only read.
    """
    from .snapshot import RUN_DIRS
    p, n = Path(parent).resolve(), Path(new).resolve()
    for i, above in enumerate([n, *n.parents]):   # the new folder (i is 0), then the folders above it: the first ones may not exist yet
        if not _same_place(above, p):
            continue
        if i == 0:
            raise UserError(
                f"--out is the parent run folder {p}. A continuation writes a new run, and the parent must not change. Options: give "
                f"another --out, for example {p.parent / (p.name + '-more')}; a run that did not finish can also go on in its own folder "
                "with --resume.")
        raise UserError(
            f"--out {n} is inside the parent run folder {p}. A new file there changes the parent. "
            "Options: give an --out outside the parent folder.")
    holders = [p, *p.parents]   # the parent, and each folder above it
    for name in RUN_DIRS:
        deleted = n / name
        if any(_same_place(held, deleted) for held in holders):   # False if `deleted` does not exist
            raise UserError(
                f"The parent run folder {p} is inside {deleted}, and a new run in {n} deletes that folder. "
                "Options: give another --out, or move the parent folder.")


def config_of_continuation(run_dir, config):
    """The config for a `--resume`: with the pass count of the checkpoint if the run of the folder is a continuation, else `config`.

    A continuation writes `continued_from` into each checkpoint (`iterations` is its pass count). A resume of a killed continuation takes the
    count from there and not from the YAML (`plan_resume` says so in a fact). The checkpoint is read for its text only.
    """
    cont = (describe(run_dir) or {}).get("continued_from")
    if not cont or int(cont.get("iterations", config.settings.iterations)) == int(config.settings.iterations):
        return config
    return with_iterations(config, int(cont["iterations"]))


def _continued_resume_fact(cont: dict, new_inputs: dict, config) -> str:
    """The fact of `plan_resume` for a checkpoint of a continuation: where the pass count comes from."""
    yaml_it = ((new_inputs.get("yaml_parsed") or {}).get("settings") or {}).get("iterations")
    text = (f"resume: this run is a continuation of {cont.get('run_dir')} (its pass {cont.get('iteration')} plus {cont.get('more_iterations')} "
            f"more passes). Its pass count is {cont.get('iterations')}, from the checkpoint.")
    if yaml_it is not None and yaml_it != cont.get("iterations"):
        text += f" The settings.iterations of the YAML ({yaml_it}) is not used."
    return text


# --- the fine pass (#318, CONTRACTS.md section 24.6) -----------------------------------------------

FINE_SOFT = {"settings.voxel_size": "a fine pass has its own voxel: the voxel that the part needs, or --fine-voxel",
             "settings.iterations": "a fine pass has a maximum of passes: the pass count of the parent run, or --fine-passes",
             "settings.beta_schedule": "the beta schedule of a fine pass goes on from the passes of the parent run"}
FINE_CONTINUED_SOFT = {"settings.voxel_size": "a continuation of a fine pass keeps the voxel of the fine pass",
                       "settings.beta_schedule": "a continuation of a fine pass keeps the beta schedule of the fine pass, seen from the passes of its parent"}
FINE_RADIUS_SOFT = {"settings.filter_radius": "the filter radius is 1.5 voxels (the default) and follows the voxel of the fine pass"}
FINE_WALL_RADIUS_SOFT = {"settings.filter_radius": "the wall rule gives the filter radius max(w / L(eta_e), 1.5 voxels) for the voxel of the run, so it follows "
                                                   "the voxel of the fine pass (members.radius_for_grid)"}
LABELS_CHANGED = "the voxel labels are not the same"   # the start of that text of `diff_inputs`: on another voxel the labels are never the same


def config_of_fine_record(rec: dict, config, early_stop: bool | None = None):
    """The config of a fine pass from its `fine_from` record: the voxel, the maximum pass count, the beta schedule, the tolerance and the window.

    `early_stop` None takes the stop rule from the record (`stop_rule` is `settled`, or null for a run with `--no-stop-rules`); False turns every
    stop rule off (a new `--no-stop-rules`). The YAML gives the rest. The caller sets `settings.iterations` after this call when the run has
    another pass count than the maximum (a continuation of a fine pass).
    """
    from .fine_pass import SETTLED_RULE, with_fine_settings
    on = (rec.get("stop_rule") == SETTLED_RULE) if early_stop is None else bool(early_stop)
    fine = with_fine_settings(config, rec["voxel_mm"], rec["max_passes"], rec["schedule_offset"], rec["tolerance"], on)
    rules = dataclasses.replace(fine.settings.stop_rules, window=int(rec.get("window", fine.settings.stop_rules.window)))
    return dataclasses.replace(fine, settings=dataclasses.replace(fine.settings, stop_rules=rules))


def fine_early_stop(fine_rec: dict, cont_rec: dict | None = None, explicit: bool | None = None) -> tuple[bool, str]:
    """`(state, source)`: are the stop rules on in a run that is a fine pass or a continuation of one, and where does that come from.

    The state of a run comes from its own record, in this order: an explicit `--stop-rules` / `--no-stop-rules` of this command (`explicit` True or
    False; source `flag`), else the step settings of the continuation (`cont_rec["step_settings"]["stop_rules"]["enabled"]`; source `continuation`),
    else the `stop_rule` of the `fine_from` record (`settled`, or null for a fine pass with `--no-stop-rules`; source `fine_from`). The record of the
    parent of a continuation is not used for a run that has its own: a continuation with `--no-stop-rules` stays without stop rules when it is resumed.
    """
    from .fine_pass import SETTLED_RULE
    if explicit is not None:
        return bool(explicit), "flag"
    enabled = (((cont_rec or {}).get("step_settings") or {}).get("stop_rules") or {}).get("enabled")
    if enabled is not None:
        return bool(enabled), "continuation"
    return fine_rec.get("stop_rule") == SETTLED_RULE, "fine_from"


def config_of_fine(run_dir, config, early_stop: bool | None = None):
    """The config for a `--resume` of a fine pass: with the voxel, the maximum pass count, the schedule and the tolerance of the checkpoint.

    Each checkpoint of a fine pass has the `fine_from` record. A resume takes these from it and does not read the parent run (it can be gone).
    A run folder that is not a fine pass gives `config` as it is. The checkpoint is read for its text only. Call it BEFORE `config_of_continuation`:
    a continuation of a fine pass has its own pass count in `continued_from`. The stop rules come from the record of the run itself (`fine_early_stop`:
    the step settings of the continuation, else `fine_from`), and an explicit `early_stop` (`--stop-rules` True, `--no-stop-rules` False) wins.
    """
    meta = describe(run_dir) or {}
    rec = meta.get("fine_from")
    if not rec:
        return config
    return config_of_fine_record(rec, config, fine_early_stop(rec, meta.get("continued_from"), early_stop)[0])


def config_for_continuation(ck: "Checkpoint", config, early_stop: bool | None = None):
    """The config for `--continue-from` a parent whose checkpoint `ck` is a fine pass: the voxel, the shifted schedule and the stop rule of the fine pass.

    A continuation of a fine pass goes on with the same problem, on the grid of the fine pass (the YAML is the same file). Else `config` as it is.
    The stop rules are those of the parent (`fine_early_stop`: its step settings when it is a continuation, else its `fine_from`), and an explicit
    `early_stop` (`--stop-rules` True, `--no-stop-rules` False) of the new run wins. The caller sets the pass count (`with_iterations`) after this call.
    """
    rec = ck.meta.get("fine_from")
    return config if not rec else config_of_fine_record(rec, config, fine_early_stop(rec, ck.meta.get("continued_from"), early_stop)[0])


def _fine_stop_text(fine_rec: dict, cont_rec: dict | None, state: bool) -> str:
    """The sentence of the fact of `plan_continue` about the stop rules of a fine parent: the state this run uses, and where it comes from."""
    parent_state = fine_early_stop(fine_rec, cont_rec)[0]
    if state != parent_state:
        return (f"The stop rules are {'on' if state else 'off'} in this run: the parent has them {'on' if parent_state else 'off'}, and "
                f"{'--stop-rules' if state else '--no-stop-rules'} of this run sets them.")
    if state:
        return f"It keeps the stop rule 'settled' (objective change at or below {fine_rec.get('tolerance'):g})."
    return "It keeps the setting that all stop rules are off."


def _fine_resume_fact(rec: dict, new_inputs: dict, n_passes: int | None = None, early_stop: bool | None = None, recorded: bool | None = None) -> str:
    """The fact of `plan_resume` for a checkpoint of a fine pass: where the voxel, the maximum and the tolerance come from.

    `early_stop` is the state of the stop rules that this resume uses (`config.settings.stop_rules.enabled`) and `recorded` the state in the
    checkpoint (`fine_early_stop`): the fact says the state that is used, and when the two differ, that the flag of this resume set it.
    """
    n = rec.get("max_passes") if n_passes is None else n_passes
    on = bool(rec.get("stop_rule")) if early_stop is None else bool(early_stop)
    ends = f"it ends when the loss has settled (objective change at or below {rec.get('tolerance'):g})" if on else "all stop rules are off"
    if recorded is not None and bool(recorded) != on:
        ends += (f" (the checkpoint has the stop rules {'on' if recorded else 'off'}; {'--stop-rules' if on else '--no-stop-rules'} of this resume "
                 "sets them " + ("on" if on else "off") + ")")
    elif early_stop is not None:
        ends += " (as the checkpoint has it)"
    text = (f"resume: this run is a fine pass from {rec.get('run_dir')} (its pass {rec.get('iteration')}, voxel {rec.get('parent_voxel_mm'):g} mm). "
            f"Its voxel is {rec.get('voxel_mm'):g} mm, its maximum is {n} passes and {ends}, all from the checkpoint. The parent is not read.")
    st = (new_inputs.get("yaml_parsed") or {}).get("settings") or {}
    used = [k for k in ("voxel_size", "iterations") if k in st]
    if used:
        text += " settings." + " and settings.".join(used) + " of the YAML " + ("is" if len(used) == 1 else "are") + " not used."
    return text


def plan_fine(ck: Checkpoint, config, problem=None, solver: str | None = None, allow_changed: bool = False, parent_dir=None) -> list[str]:
    """Compare the inputs of a fine pass with the checkpoint `ck` of the parent run. Write nothing. Return the facts (lines for `run.log`).

    `config` is the config of the fine run (`fine_pass.with_fine_settings`). The voxel, the pass count, the beta schedule and, when it
    follows the voxel (the default 1.5 voxels), the filter radius are allowed to differ (`FINE_SOFT`). Any other change of an input that
    decides the numbers is refused unless `allow_changed`. The labels differ on another voxel, so that text is left out. A change of the
    solver that runs is a fact: a finer grid can need another solver. Raise `ResumeRefused` with options.
    """
    meta = ck.meta
    new_inputs = input_hashes(config, problem, solver)
    soft = dict(FINE_SOFT)
    if config.settings.filter_radius_source == "default":
        soft.update(FINE_RADIUS_SOFT)
    elif config.settings.filter_radius_source == "wall":
        soft.update(FINE_WALL_RADIUS_SOFT)
    changes, notes = diff_inputs(meta["inputs"], new_inputs, soft=soft)
    facts = []
    changes = [c for c in changes if not c.startswith(LABELS_CHANGED)]
    for c in [c for c in changes if c.startswith("the solver that runs")]:
        changes.remove(c)
        facts.append(f"fine pass: {c} (the solver is chosen for the grid of this run)")
    facts += [f"fine pass: {n}" for n in notes]
    if config.settings.filter_radius_source == "wall":   # the wall rule follows the grid: say the numbers (the input diff skips a soft effective key)
        old_r = ((meta["inputs"].get("effective") or {}).get("settings") or {}).get("filter_radius")
        new_r = float(config.settings.filter_radius)
        if old_r is not None and abs(float(old_r) - new_r) > 1e-9 * max(new_r, 1.0):
            facts.append(f"fine pass: the wall rule gives the filter radius {new_r:.4g} mm on the fine grid (voxel {config.settings.voxel_size:g} mm; "
                         f"max(w / L(eta_e), 1.5 voxels), w {config.settings.wall_mm:g} mm); the parent run used {float(old_r):.4g} mm. "
                         f"The records of the fine run state the radius that it uses.")
    if changes:
        listing = "; ".join(changes)
        where = f" {parent_dir}" if parent_dir is not None else ""
        if not allow_changed:
            raise ResumeRefused(
                f"Fine pass stopped: the inputs changed since the parent run{where} (pass {ck.iteration}): {listing}. "
                "The fine pass would not be the same problem.",
                changes, ["give --allow-changed-inputs (MCP: allow_changed_inputs=true) to run with the new inputs; "
                          "the run log and run.json then name the changes",
                          "restore the old inputs and run the fine pass again",
                          "run the YAML as a new run in a new folder: that starts at pass 0"])
        facts.append(f"fine pass: the inputs changed since the parent run: {listing}. The override is given, so the run goes on with the new inputs.")
    return facts


def beta_stretches(schedule, first: int, last: int) -> list[dict]:
    """The stretches of one beta in the passes `first` to `last`: `[{from_pass, to_pass, beta}]`. No loop over the passes."""
    from .optimize import beta_at
    starts = [first] + sorted({int(a) for a, _b in schedule if first < int(a) <= last})
    out = []
    for i, a in enumerate(starts):
        b = (starts[i + 1] - 1) if i + 1 < len(starts) else last
        beta = beta_at(schedule, a)
        if out and out[-1]["beta"] == beta:
            out[-1]["to_pass"] = b
        else:
            out.append({"from_pass": a, "to_pass": b, "beta": beta})
    return out


def _rules_text(r: dict) -> str:
    """The stop rules (`dataclasses.asdict(StopRules)`) as words."""
    if not r["enabled"]:
        return "off"
    text = (f"on: the objective changes by {r['objective_change']:g} or less over {r['window']} passes and the grayness is "
            f"{r['grayness_max']:g} or less")
    return text + (f", or the run time reaches {r['time_budget_s']:g} s" if r["time_budget_s"] is not None else "")


def step_settings(config, meta: dict, first: int, last: int, solver: str | None, rows_state: str) -> dict:
    """The settings of the passes of a continuation as data (`continued_from.step_settings` in run.json) (CONTRACTS.md section 24.3).

    `config` is the config of the new run (`settings.iterations` is `last`). `rows_state` says whether `rows_before` is used.
    """
    from .optimize import objective_scale_resets
    st = config.settings
    stretches = beta_stretches(st.beta_schedule, first, last)
    resets = [x["from_pass"] for x in stretches if objective_scale_resets(st.objective_scale, st.beta_schedule, x["from_pass"])]
    return {"first_pass": first, "last_pass": last, "beta": stretches,
            "move_limit": {"setting": st.move_limit, "checkpoint": (meta.get("mma") or {}).get("move")},
            "objective_scale": {"mode": st.objective_scale, "set_again_at": resets},
            "stop_rules": dataclasses.asdict(st.stop_rules), "solver": solver or st.solver, "rows_before": rows_state}


def step_lines(ss: dict) -> list[str]:
    """The `continue:` lines of the step settings (`step_settings`): one fact in each line."""
    lines = [f"continue: step settings: this run runs pass {ss['first_pass']} to pass {ss['last_pass']}."]
    lines.append("continue: beta by stretch (settings.beta_schedule, counted from pass 0): " + "; ".join(
        f"{x['beta']:g} for pass {x['from_pass']}" + ("" if x["to_pass"] == x["from_pass"] else f" to {x['to_pass']}") for x in ss["beta"]) + ".")
    mv = ss["move_limit"]
    lines.append(f"continue: move limit {mv['setting']:g} (settings.move_limit)"
                 + ("" if mv["checkpoint"] is None else f"; the checkpoint holds {mv['checkpoint']:g}, the limit that step acceptance reached")
                 + ".")
    sc = ss["objective_scale"]
    lines.append(f"continue: objective scale '{sc['mode']}' (settings.objective_scale)"
                 + (": the scale is set again at pass " + ", ".join(str(n) for n in sc["set_again_at"]) if sc["set_again_at"] else
                    ": no pass sets the scale again") + ".")
    lines.append(f"continue: stop rules {_rules_text(ss['stop_rules'])}. The run time counts from the start of the parent.")
    lines.append(f"continue: solver {ss['solver']}"   # the MCP plan does not resolve `auto`: the run process picks the solver (it can be another Python)
                 + (" (the run process picks the solver; run.json of the run names it)." if ss["solver"] == "auto" else "."))
    lines.append(f"continue: rows before the pass: {ss['rows_before']}.")
    return lines


def _missing_rows_before_fact(meta: dict, k: int, n_new: int, names: list) -> str:
    """The fact for a replay when the checkpoint has no `rows_before`: say what that means for the rows that the checkpoint holds.

    A stress row (`stress:<case>`, the KS scale) gives the same numbers: the scale comes from the design and the beta of the stage,
    and pass `k` does not change it after the stage start. When pass `k` is a stage start, the label `c_changed` of the row
    (and a stage-start note of a small `c`) can differ. The buckling row
    (`buckling:<case>`) can differ, because its evaluation at the last pass changed its state. Only a buckling row gets the option.
    """
    stress = [n for n in names if n.startswith("stress:")]
    buckling = [n for n in names if n.startswith("buckling:")]
    other = [n for n in names if n not in stress and n not in buckling]
    text = (f"continue: the checkpoint has no rows_before (it was written by topopt {meta.get('topopt_version')}, before this key). "
            f"Pass {k} starts from the rows ({', '.join(names)}) as they were after pass {k}.")
    if stress:
        text += (" A stress row (KS) gives the same numbers. When this pass is a stage start, the c_changed label of the row"
                 " (and a stage-start note of a small c) can differ.")
    if buckling:
        text += (" The buckling row can differ, because its evaluation at the last pass changed its state."
                 f" Option: run the YAML again from pass 0 in a new folder with settings.iterations {n_new}.")
    if other:
        text += f" For the other rows ({', '.join(other)}) the result can differ."
    return text


def plan_continue(parent_dir, config, problem=None, solver: str | None = None, more=None, allow_changed: bool = False,
                  new_dir=None, ck: Checkpoint | None = None) -> ResumePlan:
    """Read the checkpoint of the parent run folder and plan a continuation with `more` passes. Write nothing; the parent is only read.

    `ck` is a checkpoint that the caller read with `read_checkpoint(parent, CONTINUE_RESTART_OPTION)`. A parent that runs writes a new
    checkpoint after each pass, so a process reads the checkpoint once and gives that read to every step: the pass count, the plan,
    `run.json`, the log, the hash and the copy then all name the same pass K. Without `ck` the function reads the checkpoint itself.

    The new run has `K + more` passes, where K is the pass of the checkpoint, and `config.settings.iterations` is not used (a fact says so).
    A complete checkpoint (the parent ended at pass K, which only measured) is a replay: pass K runs again with its update step, and the new
    run equals a run of `K + more` passes. A checkpoint that is not complete (the parent was stopped, killed or ended with an error) is a fork:
    the new run starts at pass K + 1 and runs no pass again. `new_dir` (the new run folder) is checked with `check_new_folder`.
    Raise `ResumeRefused` if the continuation cannot go on: no usable checkpoint, another grid, or changed inputs without `allow_changed`.
    The facts are lines for `run.log` that start with `continue:`. `ResumePlan.cont` is set; `ResumePlan.continued_from` is the run.json record.
    """
    n_more = check_more_iterations(more)
    parent = Path(parent_dir)
    if not parent.is_dir():
        raise ResumeRefused(f"Continue cannot work: {parent} is not a folder.", [],
                            ["give the run folder that you want to continue (the --out of the earlier `topopt run`)"])
    if new_dir is not None:
        check_new_folder(parent, new_dir)
    if ck is None:
        ck = read_checkpoint(parent, CONTINUE_RESTART_OPTION)
    meta = ck.meta
    k = ck.iteration
    n_new = k + n_more
    yaml_iterations = int(config.settings.iterations)
    config = with_iterations(config, n_new)
    replay = bool(meta.get("complete"))
    first = k if replay else k + 1
    new_inputs = input_hashes(config, problem, solver)
    shown = ((new_inputs.get("yaml_parsed") or {}).get("settings") or {}).get("iterations")   # what the YAML file says (`config` already has K + N)
    yaml_iterations = int(shown) if shown is not None else yaml_iterations
    old_inputs = meta["inputs"]
    old_grid, new_grid = old_inputs["grid"], new_inputs.get("grid")
    if new_grid is not None:
        why = grid_mismatch(old_grid, new_grid)
        if why:
            raise ResumeRefused(
                f"Continue cannot work: {why}. The saved design does not fit the new grid.", [],
                ["restore the YAML and the STL files of the parent (`yaml_path` in the run.json of the parent)",
                 "run the YAML as a new run in a new folder: that starts at pass 0"])
    fine_rec = meta.get("fine_from")   # #318: the parent is a fine pass (or a continuation of one): `config` has its voxel and schedule (`config_for_continuation`)
    soft = dict(CONTINUED_SOFT)
    if fine_rec:
        soft.update(FINE_CONTINUED_SOFT)
        if config.settings.filter_radius_source == "default":
            soft.update(FINE_RADIUS_SOFT)
    changes, notes = diff_inputs(old_inputs, new_inputs, soft=soft)
    design_map = None
    if problem is not None:   # the design voxels of the checkpoint and of this run, by position on the same grid
        from .voxelize import DESIGN
        old_ids = design_ids(ck.arrays["design_mask"], int(np.prod(old_grid["shape"])))
        new_ids = np.flatnonzero(np.asarray(problem.labels).reshape(-1) == DESIGN)
        if not np.array_equal(old_ids, new_ids):
            design_map = map_design(old_ids, new_ids)
            changes.append("the design voxels are not the same: " + design_map.text())
    rows = (meta.get("mma") or {}).get("rows") or {}
    has_before = "rows_before" in meta
    if not replay:
        rows_state = "not used: no pass runs again (a fork)"
    elif has_before:
        rows_state = "used (the rows at the start of pass %d, saved in the checkpoint)" % k
    elif rows:
        rows_state = "missing in the checkpoint; the rows after pass %d are used" % k
    else:
        rows_state = "not needed: no row keeps a value between passes"
    ss = step_settings(config, meta, first, n_new, solver, rows_state)
    parent_n = meta.get("n_iterations")
    facts = [f"continue: a new run folder from {parent}, checkpoint {ck.file}, pass {k} of {parent_n} (written {meta.get('written_at')}, "
             f"topopt {meta.get('topopt_version')}). The parent is only read. It does not change."]
    facts += [f.replace("resume:", "continue:", 1) for f in ck.facts]
    if replay:
        facts.append(f"continue: pass {k} was the last pass of the parent. It measured the design and did not change it. This run runs pass {k} "
                     f"again with its update step, then goes on to pass {n_new}. The result is that of a run of {n_new} passes.")
    else:
        facts.append(f"continue: the parent did not finish. Its checkpoint is at pass {k}, and the run was to end at pass {parent_n}. It was stopped, "
                     f"killed or ended with an error. This run is a fork: it starts at pass {k + 1} from that checkpoint, runs no pass again, "
                     f"and goes on to pass {n_new}.")
    if fine_rec:
        facts.append(f"continue: the parent is a fine pass from {fine_rec.get('run_dir')} (voxel {fine_rec.get('voxel_mm'):g} mm). This run keeps its voxel "
                     f"and its beta schedule (seen from pass {fine_rec.get('schedule_offset')} of the earlier runs). "
                     + _fine_stop_text(fine_rec, meta.get("continued_from"), bool(config.settings.stop_rules.enabled))
                     + " The voxel and the beta schedule of the YAML are not used.")
    facts.append(f"continue: the pass count of this run is {n_new} ({k} + {n_more} more passes)."
                 + (f" settings.iterations of the YAML ({yaml_iterations}) is not used." if yaml_iterations != n_new else ""))
    stop = meta.get("stop")
    if stop:
        rule = (stop or {}).get("rule")
        if config.settings.stop_rules.enabled:
            facts.append(f"continue: the parent ended at pass {k} by the stop rule '{rule}'. The stop rules are on in this run, so the rule is "
                         f"judged again at pass {k} with the same numbers, and it can end this run at pass {k} again, with no new pass. "
                         "Options: --no-stop-rules (MCP: stop_rules=false), or other thresholds in settings.stop_rules.")
        else:
            facts.append(f"continue: the parent ended at pass {k} by the stop rule '{rule}'. The stop rules are off in this run, "
                         f"so it goes on to pass {n_new}.")
    stopped = _stopped_fact(parent, ck, continuation=True)
    if stopped:
        facts.append(stopped.replace("resume:", "continue:", 1))
    if replay and not has_before and rows:
        facts.append(_missing_rows_before_fact(meta, k, n_new, sorted(rows)))
    from .runjob import job_facts
    job = job_facts(parent)
    if job is not None and job["state"] == "running":
        facts.append(f"continue: the parent is running now (pid {job.get('pid')}). This run starts from the checkpoint that it read, at pass {k}, "
                     f"and does not read the checkpoint of the parent again. The parent goes on and does not change. "
                     f"Its passes after pass {k} are not in this run.")
    if meta.get("topopt_version") != __version__:
        facts.append(f"continue: the checkpoint was written by topopt {meta.get('topopt_version')}; this is topopt {__version__}")
    if meta.get("platform") and meta["platform"] != _platform():
        facts.append(f"continue: the checkpoint was written on {meta['platform']}; this is {_platform()}. The numbers can differ in the last digits.")
    for gpu_fact in (_gpu_mg_fact(meta, config, solver, changes), _gpu_max_iter_fact(meta, config, solver, changes)):
        if gpu_fact:
            facts.append(gpu_fact.replace("resume:", "continue:", 1))
    scale_fact = _objective_scale_fact(meta, config, pass_left=True)
    if scale_fact:
        facts.append(scale_fact.replace("resume:", "continue:", 1))
    shift = None if new_grid is None else origin_shift(old_grid, new_grid)
    if shift:
        facts.append(f"continue: the grid origin moved by {shift:.3f} voxels (less than {ORIGIN_SHIFT_MAX:g}). The saved design goes onto "
                     "the cells with the same index.")
    facts += [f"continue: {n}" for n in notes]
    facts += step_lines(ss)
    if changes:
        listing = "; ".join(changes)
        if not allow_changed:
            raise ResumeRefused(
                f"Continue stopped: the inputs changed since the checkpoint of the parent (pass {k}): {listing}. "
                "The new run would not continue the same run.",
                changes, ["give --allow-changed-inputs (MCP: allow_changed_inputs=true) to continue with the new inputs; "
                          "the run log and run.json then name the changes",
                          "restore the old inputs and continue",
                          *_objective_scale_options(changes, meta),
                          "run the YAML as a new run in a new folder: that starts at pass 0"])
        facts.append(f"continue: the inputs changed since the checkpoint: {listing}. The override is given, so the run goes on with the new inputs.")
    record = _jsonable({
        "run_dir": str(parent.resolve()), "iteration": k, "parent_iterations": parent_n, "iterations": n_new, "more_iterations": n_more,
        "replay": replay, "checkpoint": ck.file, "checkpoint_sha256": ck.sha256,
        "topopt_version": meta.get("topopt_version"), "step_settings": ss})
    cont = Continuation(parent, k, n_more, n_new, replay, has_before, record)
    return ResumePlan(ck, facts, changes, [], design_map, continued_from=record, cont=cont, fine_from=fine_rec)


def apply_continue(plan: ResumePlan, out) -> None:
    """Fill the new run folder from the parent for a continuation, and write the facts to `run.log`. Call it after `RunWriter.clean()`.

    1. Copy the history lines up to `plan.kept_iteration` (byte for byte), the snapshots up to that pass, and the lines of the timing file
       `gpu_mg.jsonl` up to that pass. The parent is only read: no lock, no write, no new file in it.
    2. Set the run time of `out` to the saved run time of the parent (`elapsed_s` goes on).
    `resume_points` of the plan stays empty: the new run has no resume yet.
    """
    d, ck, cont = Path(out.dir), plan.ck, plan.cont
    parent, last = cont.parent, plan.kept_iteration
    snaps = sorted(((n, p) for p in (parent / "snapshots").iterdir() if (n := _snapshot_number(p)) is not None and n <= last)
                   if (parent / "snapshots").is_dir() else [])
    size = {"history": 0, "timing": 0}
    for name, key in (("history.jsonl", "history"), (TIMING_FILE, "timing")):
        if (parent / name).is_file():
            size[key] = (parent / name).stat().st_size
    snap_bytes = sum(p.stat().st_size for _n, p in snaps)
    for line in plan.facts:   # the plan first: a continuation that stops inside this function still leaves its facts in run.log
        out.log(line)
    out.log(f"continue: the new run folder gets a copy of the history up to pass {last}, {len(snaps)} snapshots ({snap_bytes} bytes) and the "
            f"timing file of the solver gpu-mg if the parent has one. The copy uses at most {snap_bytes + size['history'] + size['timing']} bytes "
            "of disk space (the snapshots, history.jsonl and gpu_mg.jsonl of the parent). The parent does not change.")
    notes = []
    hist = copy_lines(parent / "history.jsonl", d / "history.jsonl", last)
    if not hist["present"]:
        notes.append("the parent has no history.jsonl")
    elif hist["last_kept"] is None or hist["last_kept"] < last:
        notes.append(f"history.jsonl of the parent ends at pass {hist['last_kept']}, before pass {last}: the lines in between are not in the file")
    if hist["bad_lines"]:
        notes.append(f"{hist['bad_lines']} line(s) of the history of the parent do not parse and are not copied")
    (d / "snapshots").mkdir(exist_ok=True)
    for n, p in snaps:
        tmp = d / "snapshots" / (p.name + ".tmp")
        shutil.copyfile(p, tmp)
        os.replace(tmp, d / "snapshots" / p.name)
    have = {n for n, _p in snaps}
    missing = [n for n in range(last + 1) if n not in have]
    if missing:
        notes.append(f"snapshots missing in the parent: {missing[:5]}" + (" ..." if len(missing) > 5 else ""))
    timing = copy_lines(parent / TIMING_FILE, d / TIMING_FILE, last)
    if timing["present"]:
        notes.append(f"{TIMING_FILE}: {timing['kept']} lines up to pass {last} are copied")
    out.t_start = time.time() - float(ck.meta.get("elapsed_s") or 0.0)
    plan.resume_points = []
    out.log(f"continue: copied {hist['kept']} history lines ({hist['bytes']} bytes) and {len(snaps)} snapshots ({snap_bytes} bytes)"
            + (". " + ". ".join(n[0].upper() + n[1:] for n in notes) if notes else "") + ".")


# --- the state of a run ----------------------------------------------------------------------

def _hooks(model):
    """The objects that can save state: `(name, object)` for the model and the solver, if they have the methods."""
    return [(n, o) for n, o in (("model", model), ("solver", getattr(model, "solver", None)))
            if o is not None and hasattr(o, "checkpoint_state")]


class Checkpointer:
    """Saves the state of `optimize.run` after each pass, and puts a loaded state back.

    The state is in: `x`, the `MmaState` (`mma.state`), the step acceptance rule (`rule`: the move limit is in the MMA state;
    the counters and the solve of the accepted trial design are here), the last history record, the run time, and the state that
    the model and the solver offer through `checkpoint_state()`. `mma.state` is read at each save, not kept: step acceptance
    replaces the state object.
    """

    def __init__(self, out, model, mma, rule, config, inputs: dict, plan: ResumePlan | None = None, fine_from: dict | None = None):
        self.out, self.model, self.mma, self.rule, self.config = out, model, mma, rule, config
        self.inputs, self.plan = inputs, plan
        self.resume_points = list(plan.resume_points) if plan else []
        self.continued_from = plan.continued_from if plan else None   # P4b-5: each checkpoint of a continuation carries the record
        self.fine_from = fine_from or (plan.fine_from if plan else None)   # #318: the same for a fine pass (a new run gives the record, a resume has it in the plan)
        self.design_mask = design_mask_bits(model.problem.labels)   # the design voxels of this run, saved with every checkpoint
        self.n_written = self.bytes = 0
        self.seconds = 0.0

    # -- save --
    def save(self, it: int, x, prev: dict | None, complete: bool = False, stop: dict | None = None,
             rows_before: dict | None = None) -> None:
        """Write the checkpoint of pass `it`. `rows_before` (P4b-5) is the `MmaState.rows` at the start of the pass, as JSON data.

        It is saved in a complete checkpoint only (a continuation runs that pass again, and the rows that keep a value between passes must start
        as they did). An optional key: a checkpoint without it is read with a fact.
        """
        t0 = time.perf_counter()
        st, rule = self.mma.state, self.rule
        arrays = {"x": np.ascontiguousarray(x, dtype=np.float64), "mma_low": st.low, "mma_upp": st.upp,
                  "design_mask": self.design_mask}
        for name in ("xold1", "xold2"):
            if getattr(st, name) is not None:
                arrays["mma_" + name] = getattr(st, name)
        if st.x is None:
            mma_x = "none"
        elif np.array_equal(st.x, arrays["x"]):
            mma_x = "same"
        else:
            mma_x, arrays["mma_x"] = "saved", st.x
        meta = {
            "kind": KIND, "format": FORMAT, "topopt_version": __version__, "written_at": _iso(), "platform": _platform(),
            "iteration": int(it), "n_iterations": int(self.config.settings.iterations), "complete": bool(complete),
            "stop": stop, "n_design": int(arrays["x"].size), "elapsed_s": self.out.elapsed(),
            "mma": {"it": int(st.it), "f0_ref": st.f0_ref, "rows": pack_rows(st.rows, arrays), "move": st.move, "x": mma_x},
            "step_acceptance": {"counters": [rule.n_passes, rule.n_rejected, rule.n_rejected_passes, rule.n_forced]},
            "prev": prev, "inputs": self.inputs, "resume_points": self.resume_points, "extras": [],
            "objective_scale": self.config.settings.objective_scale,   # P3a: the effective settings leave `stage` out; a checkpoint without this key is from before P3a
        }
        if self.continued_from:
            meta["continued_from"] = self.continued_from
        parts = getattr(self.out, "solver_parts", None)
        if parts:       # P9-4d: which solver ran which passes (run.json `solver_parts`); the pass `it` is the last pass of the part that runs
            parts[-1]["last_pass"] = int(it)
            meta["solver_parts"] = parts
        if self.fine_from:
            meta["fine_from"] = self.fine_from
        if complete and rows_before is not None:
            meta["rows_before"] = pack_rows(rows_before, arrays)
        carry = rule.carry
        if carry is not None and carry[0] is x:
            arrays["carry_U"] = carry[2]
            meta["step_acceptance"]["carry_beta"] = carry[1]
        extra = getattr(self.model, "checkpoint_meta", None)
        if extra is not None:     # G2: gpu-mg records its two stop numbers (the effective settings leave them out at their defaults)
            meta.update(extra())
        for owner, obj in _hooks(self.model):
            for name, value in obj.checkpoint_state().items():
                arrays[f"{owner}__{name}"] = np.asarray(value)
                meta["extras"].append(f"{owner}__{name}")
        self.bytes += write_checkpoint(self.out.dir, arrays, meta)
        self.out.first_pass_saved()   # F4-47: the first saved pass ends the window in which `RunWriter.restore_earlier` can give the earlier run back
        self.n_written += 1
        self.seconds += time.perf_counter() - t0

    def summary_line(self) -> str | None:
        if not self.n_written:
            return None
        return (f"fact: {self.n_written} checkpoints written to {CHECKPOINT_FILE}, {self.bytes / self.n_written / 1e6:.2f} MB and "
                f"{1000 * self.seconds / self.n_written:.1f} ms each, {self.seconds:.2f} s in total "
                f"({100 * self.seconds / max(self.out.elapsed(), 1e-9):.2f} % of the run time)")

    # -- restore --
    def restore(self):
        """Put the saved state in place. Return `(x, prev, first, records)`.

        `x` and `prev` (the last history record) are from the checkpoint. `first` is the number of the first pass to run
        (after the last pass if the checkpoint is from a finished run). `records` are the history records up to the
        checkpoint (the stop rule and the verdict read them). A field that the checkpoint lacks gives a fact in the run log.
        """
        from .optimize import beta_at
        from .runinfo import read_history
        ck, replay = self.plan.ck, self.plan.replay
        meta, arrays, facts = ck.meta, ck.arrays, []
        dm, n = self.plan.design_map, self.model.n_design
        start = np.full(n, self.config.settings.volume_fraction)   # the start design of a new run: it fills new design voxels
        half = MMA_ASYINIT * (self.mma.xmax - self.mma.xmin)
        starts = {"x": start, "low": (start - half).reshape(-1, 1), "upp": (start + half).reshape(-1, 1)}

        def on_design(saved, key):   # a saved array on the design voxels of this run (by position if the regions changed)
            return saved if dm is None else dm.apply(saved, starts[key])

        x = np.ascontiguousarray(on_design(arrays["x"], "x"), dtype=np.float64)
        st, m = self.mma.state, meta["mma"]
        # A replay (P4b-5) runs pass `ck.iteration` again with its update step: the rows start as they did at that pass. A checkpoint without
        # `rows_before` has only the rows after the pass (the plan says so in a fact).
        rows = meta["rows_before"] if replay and "rows_before" in meta else m["rows"]
        st.it, st.f0_ref, st.rows, st.move = int(m["it"]), m["f0_ref"], unpack_rows(dict(rows), arrays), m["move"]
        st.low, st.upp = on_design(arrays["mma_low"], "low"), on_design(arrays["mma_upp"], "upp")
        st.xold1, st.xold2 = (on_design(arrays[k], "x") if k in arrays else None for k in ("mma_xold1", "mma_xold2"))
        st.x = None if m["x"] == "none" else x.copy() if m["x"] == "same" else on_design(arrays["mma_x"], "x")
        sa = meta.get("step_acceptance")
        if sa is None:
            facts.append("the checkpoint has no step acceptance counters: the counts in the end summary restart at 0")
        else:
            self.rule.n_passes, self.rule.n_rejected, self.rule.n_rejected_passes, self.rule.n_forced = sa["counters"]
        first = ck.iteration + 1
        if replay:
            first = ck.iteration      # the pass of the checkpoint runs again, with an update step
        elif meta.get("complete"):
            first = max(first, int(self.config.settings.iterations) + 1)
        if "carry_U" in arrays and sa and sa.get("carry_beta") == beta_at(self.config.settings.beta_schedule, first):
            if self.plan.changes:
                # The saved solve is the displacement of the old inputs (forces, material, penalty, filter, regions). The next pass
                # would read it as the solve of this design. It is not used: the next pass solves the design again.
                facts.append("the step acceptance solve of the saved design is not used: the inputs changed, so that solve belongs to "
                             "the old inputs. The next pass solves the design again")
            elif meta.get("inputs", {}).get("solver") not in (None, self.inputs.get("solver")):
                # Another solver wrote the checkpoint (a fallback of `auto`, P9-4d): its solve is not the solve of this solver.
                facts.append(f"the step acceptance solve of the saved design is not used: solver {meta['inputs']['solver']} made it and this part runs "
                             f"solver {self.inputs.get('solver')}. The next pass solves the design again")
            elif not hasattr(self.model.solver, "factor"):
                # The solver of gpu-mg (G1d) is matrix-free: it has no factor to make again (D1b plan, G1d note).
                facts.append("the step acceptance solve of the saved design is not used: the solver has no factor to make again "
                             "(solver gpu-mg). The next pass solves the design again")
            else:
                # The straight run solved this design in the step acceptance. The adjoint solves of the next pass use that
                # factor, so factor again here (no solve): the next pass then reads the same U as the straight run.
                rho_tilde, rho = self.model.density(x, sa["carry_beta"])
                self.model.solver.factor(self.model.stiffness(rho))
                self.rule.carry = (x, sa["carry_beta"], arrays["carry_U"])
        known = []
        for owner, obj in _hooks(self.model):
            known.append(owner + "__")
            mine = {n[len(owner) + 2:]: v for n, v in arrays.items() if n.startswith(owner + "__")}
            facts += list(obj.restore_checkpoint_state(mine) or [])
        stray = [n for n in arrays if "__" in n and not n.startswith(tuple(known))]
        if stray:
            facts.append(f"the checkpoint has fields that this topopt does not use: {', '.join(stray)}")
        prev = meta.get("prev")
        try:
            records = read_history(Path(self.out.dir) / "history.jsonl", strict=False).rows
        except OSError:
            records = []
        if replay:
            # The checkpoint holds the record of pass K, which only measured. The pass before it is the record that pass K reads (stage note).
            prev = records[-1] if records else None
        elif not records and prev is not None:
            records = [prev]
        if prev is None and ck.iteration > 0:
            facts.append("the checkpoint has no history record: the beta stage note at the next pass can differ")
        for line in facts:
            self.out.log(("continue: " if self.plan.cont is not None else "resume: ") + line)
        return x, prev, first, records
