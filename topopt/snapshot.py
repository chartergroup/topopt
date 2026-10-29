"""Run directory writer (CONTRACTS.md section 5).

Whole files (`status.json`, `labels.bin`, `run.json`, the snapshots) are atomic. Each one is
written to `<name>.tmp` in the same folder first. Then `os.replace` moves it to the final name.
A reader never sees a part file.

`history.jsonl` and `run.log` only grow. They are not atomic as files. A history record goes to
disk as one whole line, in one write, followed by a flush. A `run.log` line is one write, and the
file is closed after each line. A reader can still see a partial last line (a read in the middle of a
write, or a killed process). Read `history.jsonl` with `topopt.runinfo.read_history`: it skips a partial
last line and gives that as a fact.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .runinfo import peak_rss_info, sha256_file

# The names a new run deletes in an old run folder (and their `.tmp` files). No other file is touched.
# "gpu_mg.jsonl" is the timing file of the solver gpu-mg (a literal: this module must not import `topopt.gpu_mg`, pyopencl is optional).
RUN_FILES = ("run.json", "labels.bin", "status.json", "history.jsonl", "run.log",
             "checkpoint.npz", "checkpoint.prev.npz", "verdict.json", "robust.json", "gpu_mg.jsonl")
RUN_DIRS = ("snapshots", "result", "verify")
# A run that stopped in its setup leaves these two names (and an empty `snapshots/`). They hold no result, so they do not make an old run.
SETUP_LEAVES = ("status.json", "run.log")


def old_run_files(run_dir) -> list[str]:
    """The names in `run_dir` that hold the data of an earlier run, in the order of `RUN_FILES` and `RUN_DIRS`. An empty list: no earlier run.

    A name of `RUN_FILES` counts if the file is there, except `SETUP_LEAVES`. A name of `RUN_DIRS` counts if the folder holds an entry.
    A folder that holds only what a run leaves when it stops in its setup (after a refused run, for example) has no earlier run: a new run
    loses nothing there. `topopt run` uses this list to decide when its input checks must come before `RunWriter.clean()`.
    """
    d = Path(run_dir)
    found = [name for name in RUN_FILES if name not in SETUP_LEAVES and (d / name).is_file()]
    for name in RUN_DIRS:
        try:
            if (d / name).is_dir() and next(iter((d / name).iterdir()), None) is not None:
                found.append(name)
        except OSError:
            continue
    return found


# F4-47: `RunWriter.clean()` does not delete an earlier run. It moves the names of `RUN_FILES` and `RUN_DIRS` into this hidden folder of the run
# folder, and the run deletes the folder when it has saved its first pass (`RunWriter.first_pass_saved`). A refusal or an error before that
# moves the files back (`RunWriter.restore_earlier`). `ASIDE_MOVING` is the same folder while its files move in or out. Each change of state is one
# `rename` of the folder, so a kill never leaves a state that the next run cannot read (`recover_earlier_run`). CONTRACTS.md section 24.1.
ASIDE_DIR = ".earlier-run"
ASIDE_MOVING = ".earlier-run.moving"
CHECKPOINT_NAMES = ("checkpoint.npz", "checkpoint.prev.npz")   # the first saved pass writes the first one (`checkpoint.write_checkpoint`)


def _wipe_run(d: Path) -> None:
    """Delete the names that a run writes in `d`: the files of `RUN_FILES` with their `.tmp` files, and the folders of `RUN_DIRS`."""
    for name in RUN_FILES:
        for f in (d / name, d / (name + ".tmp")):
            if f.exists():
                f.unlink()
    for name in RUN_DIRS:
        if (d / name).exists():
            shutil.rmtree(d / name)


def _move_back(d: Path) -> None:
    """Move every entry of `ASIDE_MOVING` into `d` and delete that folder. Each entry is one `os.replace`: a kill stops between two of them."""
    moving = d / ASIDE_MOVING
    for name in sorted(os.listdir(moving)):
        os.replace(moving / name, d / name)
    moving.rmdir()


def _restore(d: Path) -> None:
    """Delete what a run wrote in `d` after `clean()` moved the earlier run aside, and move the earlier run back. `ASIDE_DIR` must be complete."""
    _wipe_run(d)
    os.rename(d / ASIDE_DIR, d / ASIDE_MOVING)   # from here a kill leaves a folder whose entries all go back, and nothing is deleted again
    _move_back(d)


def recover_earlier_run(run_dir) -> str | None:
    """Make the run folder right after a run that a kill stopped before it saved its first pass. Return a sentence for the log, or None.

    `topopt run` calls this first, under the job lock. Three states can be there, and none of them is in a folder that a run ended on its own:
    - `ASIDE_MOVING`: a kill stopped a move of the earlier run (in or out). No file of a new run is in the folder, and every entry goes back.
    - `ASIDE_DIR` and no checkpoint in the folder: the run that moved the earlier run aside did not save a pass. The files that it wrote are
      deleted and the earlier run is moved back.
    - `ASIDE_DIR` and a checkpoint in the folder: that run saved its first pass and was killed before it deleted the folder. The earlier run
      is given up already, so the folder is deleted (nothing to say).
    """
    d = Path(run_dir)
    if (d / ASIDE_MOVING).is_dir():
        _move_back(d)
        return (f"a run stopped while it moved the earlier run of this folder aside ({ASIDE_MOVING}/). The earlier run is back in place.")
    if not (d / ASIDE_DIR).is_dir():
        return None
    if any((d / name).exists() for name in CHECKPOINT_NAMES):
        shutil.rmtree(d / ASIDE_DIR, ignore_errors=True)
        return None
    _restore(d)
    return (f"a run stopped before it saved its first pass (the process was killed, or the host went down, or the files could not move back) and "
            f"left the earlier run of this folder in {ASIDE_DIR}/. The earlier run is back in place, and the files of the stopped run are deleted.")


def has_aside(run_dir) -> bool:
    """True if the folder holds a state of `recover_earlier_run`: a run that never saved a pass left an earlier run aside, or stopped while it moved it."""
    d = Path(run_dir)
    return (d / ASIDE_DIR).is_dir() or (d / ASIDE_MOVING).is_dir()


PROBLEM_KEY_FORMAT = 1


def _canonical(obj):
    """Plain JSON data for the problem key: a number is a float (3 and 3.0 are one value), a tuple is a list, a dataclass is its fields."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return float(obj)
    return str(obj)


def problem_key(cfg, shape, h) -> str:
    """The key of the problem of a run, `run.json` `problem_key` (CONTRACTS.md section 5): the SHA-256 of a canonical JSON.

    It hashes the inputs of the optimization model as the solver gets them (`problem.config`, with the grid `shape` and `h`): the geometry by
    content, the grid, the loads, the material and the rows. It does not hash the loop and the solver settings, the post step or the checks.
    So runs of the same model group together whatever the method settings, the folder, the name of the YAML or the names of the region files are.
    It holds: the grid (`shape`) and the voxel size `h`; `volume_fraction`, `objective`, `case_combine`; the material; every load case (its name,
    weight, forces with moment and remote point, pressures, the compliance and stress limits; not the free-text source of an allowable); the symmetry
    planes, the springs, the joints and the limits rows; the buckling row limit and the perimeter bound; the interfaces (an STL interface by the
    SHA-256 of its file); and the SHA-256 of every region STL by role (design; fixed with its axes, keep_out and solid, each list sorted; load and
    measure by region name). Names that the key holds on purpose: the names of the load cases and of the load and measure regions
    (the rows refer to them, so a rename gives a new key). `volume_fraction` is also in the key under `objective: mass`, where it is the start
    density. It does not hold: `iterations`, the solver and its tolerances, the filter radius and `min_member_mm`, `penal`, the beta schedule, the
    move limit, the stop rules, the robust mode, the mesh method, the stress-row method keys, the guards, the print block, the checks
    (`acceptance`, `bore_checks`), `regions.not_printed` and `settings.part_only_min_film_mm` (they shape `result/part_only.stl` in the post
    step only: the solver gets the same model, `tests/test_part_only.py`), the name, the assumptions and every path. So a copy of the problem in another folder has the same key, a changed
    force has another, and the same problem with more passes has the same.
    """
    st, rg = cfg.settings, cfg.regions
    axes = rg.fixed_axes or tuple((0, 1, 2) for _ in rg.fixed)
    files = {
        "design": sha256_file(rg.design),
        "fixed": sorted([sha256_file(p), list(a)] for p, a in zip(rg.fixed, axes)),
        "keep_out": sorted(sha256_file(p) for p in rg.keep_out),
        "solid": sorted(sha256_file(p) for p in rg.solid),
        "load": {lr.name: sha256_file(lr.path) for lr in rg.load},
        "measure": {m.name: sha256_file(m.path) for m in rg.measure},
    }
    cases = []
    for c in cfg.load_cases:
        d = dataclasses.asdict(c)
        d.pop("allowable_source", None)    # free text about where the allowable comes from
        cases.append(d)
    interfaces = []
    for i in cfg.interfaces:
        d = dataclasses.asdict(i)
        d["path"] = None if i.path is None else sha256_file(i.path)
        interfaces.append(d)
    body = {
        "format": PROBLEM_KEY_FORMAT,
        "shape": [int(v) for v in shape], "h": float(h),
        "volume_fraction": st.volume_fraction, "objective": st.objective, "case_combine": st.case_combine,
        "perimeter_max_mm2": getattr(st, "perimeter_max_mm2", None),
        "buckling_min_factor": cfg.buckling.min_factor,
        "material": cfg.material, "load_cases": cases,
        "mirror": cfg.mirror, "springs": cfg.springs, "joints": cfg.joints, "limits": cfg.limits, "interfaces": interfaces,
        "files": files,
    }
    text = json.dumps(_canonical(body), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _atomic_write(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class RunWriter:
    """Writes run.json, labels.bin, status.json, history.jsonl, snapshots, run.log (and robust.json in robust mode)."""

    def __init__(self, run_dir, t_start=None, quiet=False):
        self.dir = Path(run_dir)
        self.t_start = time.time() if t_start is None else t_start
        self._phase_t0 = self.t_start   # the start of this command. A resume and a continuation move `t_start` back (the clock of the parent run); the phases do not follow it
        self.quiet = quiet
        self.state = "starting"
        self.cur_iteration, self.cur_step = 0, "setup"   # last step written, for the error status
        self.first_pass = 0   # the first pass of THIS process (E1-P2): 0 for a new run. A resume or a continuation: None, then `optimize.run` sets it
        self._marks: dict = {}          # state -> the time at which `status` set it first (`phase_times`)
        self._loop_peak: int | None = None   # the peak memory when the loop ended (`phase_times`)
        self.earlier_aside = False   # `clean` moved an earlier run into `ASIDE_DIR`, and no pass is saved yet (F4-47)
        self.solver_parts = None     # P9-4d: which solver ran which passes (a list of dicts, `solver_fallback.part_start`); None until `optimize.run` starts a part

    def elapsed(self) -> float:
        return time.time() - self.t_start

    def clean(self):
        """Make the folder ready for a new run: no file of an old run in the way, and the new folders.

        If the folder holds an earlier run (`old_run_files`), the run is not deleted at once (F4-47): its names of `RUN_FILES` and `RUN_DIRS` move
        into `ASIDE_DIR`. `first_pass_saved` deletes that folder, and `restore_earlier` moves the files back. Else the leftovers are deleted.
        A folder of an older state (`ASIDE_DIR`, `ASIDE_MOVING`) is deleted first: `recover_earlier_run` has run before, so it holds nothing to keep.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in (ASIDE_DIR, ASIDE_MOVING):
            shutil.rmtree(self.dir / old, ignore_errors=True)
        for name in RUN_FILES:   # a half-written file of a killed run is not a part of an earlier run
            if (self.dir / (name + ".tmp")).exists():
                (self.dir / (name + ".tmp")).unlink()
        if old_run_files(self.dir):
            moving = self.dir / ASIDE_MOVING
            moving.mkdir()
            try:
                for name in (*RUN_FILES, *RUN_DIRS):
                    if os.path.lexists(self.dir / name):
                        os.replace(self.dir / name, moving / name)
            except BaseException:   # a move that fails (a folder on another file system, a missing right) or Ctrl+C: the folder is as it was
                _move_back(self.dir)
                raise
            os.rename(moving, self.dir / ASIDE_DIR)
            self.earlier_aside = True
        else:
            _wipe_run(self.dir)
        (self.dir / "snapshots").mkdir()

    def first_pass_saved(self):
        """The first pass is saved (`Checkpointer.save`): delete the earlier run that `clean` moved aside. Do nothing without one. A failure to delete is
        not an error: the folder is garbage, and `recover_earlier_run` deletes it, because a checkpoint of this run is in the folder."""
        if self.earlier_aside:
            self.earlier_aside = False
            shutil.rmtree(self.dir / ASIDE_DIR, ignore_errors=True)

    def restore_earlier(self) -> bool:
        """Before the first pass is saved: delete the files that this run wrote and move the earlier run back (`topopt run` then writes `status.json`).

        Return False if `clean` moved no earlier run aside. An `OSError` of the move goes on: the folder then holds the files that did not move,
        and the next `topopt run` in the folder moves the rest (`recover_earlier_run`).
        """
        if not self.earlier_aside:
            return False
        self.earlier_aside = False
        _restore(self.dir)
        return True

    # --- log and status ----------------------------------------------------

    def log(self, line: str):
        with open(self.dir / "run.log", "a") as f:
            f.write(line + "\n")
        if not self.quiet:
            print(line, flush=True)

    def status(self, state=None, iteration=0, step="setup", message=""):
        if state is not None:
            self.state = state
            self._mark(state)
        if state != "error":
            self.cur_iteration, self.cur_step = int(iteration), step
        s = {"state": self.state, "iteration": int(iteration), "step": step,
             "elapsed_s": round(self.elapsed(), 2), "message": message}
        if self.first_pass != 0:   # a resume or a continuation (E1-P2): the live time estimate counts its passes from here. null: not read yet
            s["first_pass"] = self.first_pass
        _atomic_write(self.dir / "status.json", json.dumps(s).encode())

    def _mark(self, state: str):
        """Keep the first time of each state, and the peak memory at the end of the loop, for `phase_times`. Never raises."""
        self._marks.setdefault(state, time.time())
        if state in ("postprocess", "stopped") and self._loop_peak is None:
            try:
                self._loop_peak = int(peak_rss_info()[0])
            except Exception:   # a fact for the job record must not stop the run
                pass

    def phase_times(self) -> dict:
        """The phases of the run as the job record (`job.json`) states them, from the first time of each state in `status`:

        `setup_s` from the start of the command to the first pass (`running`); `loop_s` from there to `postprocess` (or `stopped`: a request
        ended the loop); `post_s` from `postprocess` to `done` (the check of buckling, if asked, is in it; the refine step is not).
        The start of the command is the time that the writer got when it was made, also for a resume and a continuation: `t_start` moves back
        by the elapsed time of the parent run there (so that `status.json` `elapsed_s` goes on from it), and the phases of this job do not.
        `peak_rss_loop_bytes` is the peak memory at the end of the loop, before the post step. A phase that did not end (a run that failed
        in it) is None, as is the peak of a loop that did not end.
        """
        m = self._marks
        start, running, post, done = self._phase_t0, m.get("running"), m.get("postprocess"), m.get("done")
        loop_end = post if post is not None else m.get("stopped")

        def span(a, b):
            return None if a is None or b is None else round(max(0.0, b - a), 2)
        return {"setup_s": span(start, running), "loop_s": span(running, loop_end), "post_s": span(post, done),
                "peak_rss_loop_bytes": self._loop_peak}

    def step(self, iteration, step):
        """Mark a new step of the loop (state running)."""
        self.status("running", iteration, step)

    # --- files of the run --------------------------------------------------

    def write_labels(self, labels: np.ndarray):
        _atomic_write(self.dir / "labels.bin", np.ascontiguousarray(labels, dtype=np.uint8).tobytes())

    @staticmethod
    def _rel(cfg, q) -> str:
        """A region file as `run.json` stores it: relative to the YAML folder, or the path as it is if it lies outside."""
        q = Path(q)
        try:
            return q.resolve().relative_to(Path(cfg.yaml_path).resolve().parent).as_posix()
        except ValueError:
            return str(q)

    @staticmethod
    def _part_only_plan(cfg) -> dict:
        """The `part_only` block at the start of the run (CONTRACTS.md section 9.1): which bodies the post step leaves out.

        `entries` name each body by role, name and region file (the `region_files` form), so that the post step finds the
        file again. The post step adds the volumes, the body count and the other facts to this block. `min_film_mm` is the value of
        `settings.part_only_min_film_mm`; the key is absent when the setting is off (P4b-6), so a run without it has the block it had before.
        """
        plan = {"file": "part_only.stl", "removed": [n.label for n in cfg.not_printed],
                "entries": [{"role": n.role, "name": n.name, "file": RunWriter._rel(cfg, n.path)} for n in cfg.not_printed]}
        if cfg.settings.part_only_min_film_mm is not None:
            plan["min_film_mm"] = float(cfg.settings.part_only_min_film_mm)
        return plan

    @staticmethod
    def _region_files(cfg) -> dict:
        def rel(q):
            return RunWriter._rel(cfg, q)
        return {"design": rel(cfg.regions.design),
                "fixed": [rel(q) for q in cfg.regions.fixed],
                "keep_out": [rel(q) for q in cfg.regions.keep_out],
                "solid": [rel(q) for q in cfg.regions.solid],   # A3a (section 22a): frozen, with an exact shape
                "load": {lr.name: rel(lr.path) for lr in cfg.regions.load}}

    def write_run_json(self, problem, n_active, n_free_dofs, solver, started_at=None, continued_from=None, fine_from=None):
        from .cli import derived_numbers
        from .loads import load_cases_json, lost_force
        from .optimize import formulation_from_config
        from .voxelize import DESIGN
        cfg = problem.config
        n_design = int((problem.labels == DESIGN).sum())
        form = formulation_from_config(cfg, n_design)
        st = cfg.settings
        origin = np.asarray(problem.origin, dtype=float)
        centers = {}
        for idx, lr in enumerate(cfg.regions.load):
            ijk = np.argwhere(problem.load_region_ids == idx)
            centers[lr.name] = [round(float(v), 4) for v in origin + problem.h * (ijk.mean(axis=0) + 0.5)]
        run = {
            "format": 1,
            "name": cfg.name,
            "yaml_path": str(cfg.yaml_path),
            "problem_key": self._problem_key(cfg, problem.shape, problem.h),   # DL1a: the same problem has the same key (see `problem_key`)
            "started_at": started_at or datetime.now().astimezone().isoformat(timespec="seconds"),
            "shape": [int(s) for s in problem.shape],
            "origin": [float(v) for v in origin],
            "h": float(problem.h),
            "n_active": int(n_active),
            "n_free_dofs": int(n_free_dofs),
            "iterations": int(st.iterations),
            "volume_fraction": float(st.volume_fraction),
            "load_cases": load_cases_json(problem, centers),
            "regions": {
                "design": cfg.regions.design.name,
                "fixed": [p.name for p in cfg.regions.fixed],
                "keep_out": [p.name for p in cfg.regions.keep_out],
                "solid": [p.name for p in cfg.regions.solid],
                "load": {lr.name: lr.path.name for lr in cfg.regions.load}},
            "region_files": self._region_files(cfg),   # relative to the YAML folder, or absolute
            "solver": solver,
            "solver_setting": st.solver,    # P9-4d: the value of settings.solver (`solver` is the solver of the last part that ran)
            "solver_parts": copy.deepcopy(self.solver_parts) if self.solver_parts else [],
            "mesh_method": st.mesh_method,
            "envelope_fill": st.envelope_fill,
            "objective": form.objective.name,
            "objective_form": form.objective.form,
            "constraints": [{"name": c.name, "limit": c.limit} for c in form.constraints],
            **self._stress_block(cfg),
            "mass_design_space_g": n_active * problem.h ** 3 / 1000.0 * cfg.material.density,
            "assumptions": list(cfg.assumptions),
            "lost_force": getattr(self, "lost_force", None) or lost_force(
                cfg, problem.load_region_ids, problem.fixed_dofs, problem.forces,
                (problem.extended or {}).get("entries")),
            "derived": derived_numbers(
                cfg, problem.shape, float(problem.h), n_active,
                n_design, n_free_dofs, solver, problem.extended, problem.load_reach or None),
        }
        if getattr(problem, "mirror", None):   # the `mirror` key of the YAML: planes, orphans, load asymmetry (section 26.1)
            run["mirror"] = problem.mirror
        if cfg.regions.fixed_axes:   # per-axis supports (CONTRACTS.md section 26.2)
            run["supports"] = {"axes": {p.name: "".join("xyz"[a] for a in ax)
                                        for p, ax in zip(cfg.regions.fixed, cfg.regions.fixed_axes)},
                               **(problem.extended or {}).get("supports", {})}
        run["print"] = cfg.print_settings.to_dict()   # V2: the print settings of this run (CONTRACTS.md section 18)
        if cfg.filament is not None:   # CONTRACTS.md section 18.8: the filament, its facts and the source of each material number; absent without `material.filament`
            run["filament"] = cfg.filament.to_dict()
        if continued_from:   # P4b-5 (CONTRACTS.md section 24.3): the run is a continuation of another run folder. The key is absent in any other run
            run["continued_from"] = continued_from
        if fine_from:   # #318 (CONTRACTS.md section 24.6): the run is a fine pass from another run folder. The key is absent in any other run
            run["fine_from"] = fine_from
        if cfg.interfaces:   # CONTRACTS.md section 23: the exact shapes for the post step; the key is absent without them
            run["interfaces"] = [i.to_dict(Path(cfg.yaml_path).resolve().parent) for i in cfg.interfaces]
        if problem.region_facts.get("taken"):   # CONTRACTS.md section 23: what the cuts take from the frozen regions; absent when no cut takes a voxel
            run["interface_taken"] = problem.region_facts["taken"]
        if cfg.not_printed:   # CONTRACTS.md section 9.1: the bodies that result/part_only.stl leaves out; the key is absent without them
            run["part_only"] = self._part_only_plan(cfg)
        if problem.springs is not None or problem.joints is not None:   # springs (CONTRACTS.md section 35), joints (section 38)
            from .loads import extended_facts
            ex = extended_facts(problem)
            run["supports"] = ex["supports"]
            for key in ("springs", "joints"):
                if key in ex:
                    run[key] = ex[key]
        _atomic_write(self.dir / "run.json", json.dumps(run, indent=1).encode())

    def _problem_key(self, cfg, shape, h) -> str | None:
        """`problem_key`, or None if it cannot be made: a fact for the data loop must not stop a run. One line in run.log says why."""
        try:
            return problem_key(cfg, shape, h)
        except Exception as exc:
            try:
                self.log(f"warning: problem_key is null in run.json: {type(exc).__name__}: {exc}")
            except OSError:
                pass   # the log is not writable: the run goes on without the line
            return None

    @staticmethod
    def _stress_block(cfg) -> dict:
        """`run.json` key "stress" (CONTRACTS.md section 17): the limits as the user gave them.

        Only for a run with a stress row. The post-process and `topopt.stress_report` read it.
        """
        cases = [c for c in cfg.load_cases if c.stress_limit_mpa is not None]
        if not cases:
            return {}
        st = cfg.settings
        return {"stress": {
            "q": st.stress_q, "power": st.stress_power, "scale": st.stress_scale,
            "step_acceptance": st.stress_step_acceptance,
            "material": {"E": cfg.material.E, "nu": cfg.material.nu, "density": cfg.material.density},
            "cases": [{"name": c.name, "stress_limit_MPa": c.stress_limit_mpa,
                       "limit_from": c.stress_limit_from, "allowable_mpa": c.allowable_mpa,
                       "allowable_source": c.allowable_source, "safety_factor": c.safety_factor,
                       "explicit_stress_limit_MPa": c.stress_limit}
                      for c in cases]}}

    def write_solver_record(self):
        """Put the solver record into run.json: `solver_parts` (a list; `solver_fallback` describes the entries) and `solver` (the solver of the last part).

        Called when a part of the run starts, ends or switches (`optimize.run`, `solver_fallback.run_with_fallback`). Does nothing when run.json is not there.
        """
        path = self.dir / "run.json"
        try:
            run = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        run["solver_parts"] = copy.deepcopy(self.solver_parts) if self.solver_parts else []
        if self.solver_parts and self.solver_parts[-1].get("first_pass") is not None:
            run["solver"] = self.solver_parts[-1]["solver"]
        _atomic_write(path, json.dumps(run, indent=1).encode())

    def write_mirror_end(self, mirror: dict):
        """Put the mirror facts with the end values of the deviation back into the `mirror` key of run.json."""
        path = self.dir / "run.json"
        run = json.loads(path.read_text())
        run["mirror"] = mirror
        _atomic_write(path, json.dumps(run, indent=1).encode())

    def write_part_voxel(self, block: dict):
        """Put the `part_voxel` block (`fine_pass.part_voxel_block`, CONTRACTS.md section 24.6) into run.json: written once, after the last pass.
        A fact for the user that never stops the run: with no run.json (a run body that wrote none) there is nothing to update."""
        path = self.dir / "run.json"
        if not path.is_file():
            return
        run = json.loads(path.read_text())
        run["part_voxel"] = block
        _atomic_write(path, json.dumps(run, indent=1).encode())

    def write_perimeter(self, block: dict):
        """Put the `perimeter` block (CONTRACTS.md section 26.3) into run.json: at the start of the run, and again at the end with `p_final_mm2`."""
        path = self.dir / "run.json"
        run = json.loads(path.read_text())
        run["perimeter"] = block
        _atomic_write(path, json.dumps(run, indent=1).encode())

    def write_snapshot(self, iteration: int, density_u8: np.ndarray):
        _atomic_write(self.dir / "snapshots" / f"it_{iteration:04d}.bin", density_u8.tobytes())

    def write_robust(self, summary: dict):
        """The summary of the robust part, written once at the end of the loop (CONTRACTS.md section 22b)."""
        _atomic_write(self.dir / "robust.json", json.dumps(summary, indent=1).encode())

    def add_history(self, rec: dict):
        """Add one record as one whole line: one write, then a flush (not atomic as a file; see the docstring)."""
        data = (json.dumps(rec) + "\n").encode()
        with open(self.dir / "history.jsonl", "ab") as f:
            f.write(data)
            f.flush()
