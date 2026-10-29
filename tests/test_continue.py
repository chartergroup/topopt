"""Tests of `topopt run --continue-from PARENT --more-iterations N` (P4b-5, CONTRACTS.md section 24.3, finding F4-14).

A continuation is a new run folder. It starts from the last checkpoint of a run folder (the parent) and has K + N passes in all, where K is the
pass of that checkpoint. The parent is only read. A parent that finished (its checkpoint is complete) runs pass K again, now with its update
step (a replay). A parent that did not finish (stopped, killed) gives a fork: the new run starts at pass K + 1 and runs no pass again.

The runs are child processes with one thread (like tests/test_checkpoint.py), so a direct solver gives the same bytes in each run: a
continuation then equals the run of K + N passes byte for byte. Where it cannot (AMG, the buckling row from a checkpoint with no `rows_before`),
the tests hold the measured differences as constants. The MCP part is in tests/test_continue_mcp.py, gpu-mg on pocl in
tests/test_continue_gpu_mg.py.
"""

import functools
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import test_buckling_row_resume as br
import test_checkpoint as tc
from test_checkpoint import (ENV1, arrays_of, child, clean, edit, finish, history, killed, log, meta_of, needs_direct, resume, run_args,
                             same_run, tree_state)
from test_buckling_row import column_yaml
from test_buckling_row_wiring import with_buckling
from test_optimize import make_case
from test_stress_report import stress_yaml
from topopt import checkpoint as C
from topopt.cli import main
from topopt.runjob import STOPPED_EXIT
from topopt.snapshot import RunWriter

N_IT = 24                       # the passes of the full run; the beta stages start at 10 and 20 (stress case: 10 and 18)
STOP = Path(__file__).with_name("stop_at.py")
DROP = ("continued_from",)      # the key of run.json that only a continuation has

# The measured differences of a continuation that cannot be exact (CONTRACTS.md section 24.3, "Measured").
AMG_RTOL = 1e-6                 # AMG: two straight runs differ by 1.3e-8 (compliance) on this case, so does a continuation
OLD_ROW_EXACT_RTOL = 1e-9       # a buckling row, a checkpoint with no rows_before, K = 12: the numbers are equal
OLD_ROW_MOVED_RTOL = 1e-5       # the same at K = 4: the evaluation schedule moves. Measured: compliance 6.7e-7, volume 8.0e-7 (K = 3: 3.0e-6, 1.8e-6)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_state(d):
    """Every file and folder of `d` (the folder itself too): the kind, the modification time in ns and the SHA-256 of a file. It includes `job.*`."""
    d = Path(d)
    state = {}
    for p in [d] + sorted(d.rglob("*")):
        st = p.lstat()
        state[str(p.relative_to(d))] = (stat.S_ISDIR(st.st_mode), st.st_mtime_ns, None if stat.S_ISDIR(st.st_mode) else sha256(p))
    return state


def cont(yaml_path, out, parent, n, *extra, point="never:-1"):
    """`topopt run YAML --out OUT --continue-from PARENT --more-iterations N` in a child process (1 thread). Exit code 0 is asserted."""
    return finish(yaml_path, out, "--continue-from", str(parent), "--more-iterations", str(n), *extra, point=point)


def refused(capsys, yaml_path, out, *flags):
    """The CLI in this process with `flags`: the exit code is 1 and the output is one error. Return the text of stderr."""
    assert main(run_args(yaml_path, out, *flags)) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and len(err.strip().splitlines()) == 1, err
    return err


class Family:
    """A case of `total` passes: the run of all of them (`base`) and the finished runs of k passes (`parent(k)`), each made once.

    The YAML must be alone in its folder (with its STL files): a parent gets a copy of the folder with `settings.iterations: k`.
    """

    def __init__(self, root, yaml_path, total=N_IT):
        self.root, self.yaml, self.total = Path(root), Path(yaml_path), total
        self._parents = {}

    @functools.cached_property
    def base(self):
        out = self.root / "base"
        finish(self.yaml, out)
        return out

    def parent_yaml(self, k):
        self.parent(k)
        return self.root / f"case{k}" / self.yaml.name

    def parent(self, k):
        if k not in self._parents:
            import shutil
            dst = self.root / f"case{k}"
            shutil.copytree(self.yaml.parent, dst)
            edit(dst / self.yaml.name, lambda d: d["settings"].update(iterations=k))
            out = self.root / f"parent{k}"
            finish(dst / self.yaml.name, out)
            self._parents[k] = out
        return self._parents[k]


@pytest.fixture(scope="module")
def cantilever(tmp_path_factory):
    root = tmp_path_factory.mktemp("cont_cantilever")
    (root / "case").mkdir()
    return Family(root, make_case(root / "case", h=1.0, iterations=N_IT))


@pytest.fixture(scope="module")
def stress(tmp_path_factory):
    root = tmp_path_factory.mktemp("cont_stress")
    (root / "case").mkdir()
    return Family(root, stress_yaml(root / "case", iterations=N_IT))


@pytest.fixture(scope="module")
def column(tmp_path_factory):
    root = tmp_path_factory.mktemp("cont_column")
    return Family(root, with_buckling(column_yaml(root / "case", **br.COLUMN), min_factor=br.F))


@pytest.fixture
def row_timing(monkeypatch):
    monkeypatch.setattr(tc, "TIMING", tc.TIMING | br.ROW_TIMING)


@pytest.fixture
def work(cantilever, tmp_path):
    """A copy of the case in a new folder (the YAML and the STL files): a test can edit it."""
    import shutil
    dst = tmp_path / "case"
    shutil.copytree(cantilever.yaml.parent, dst)
    return dst / cantilever.yaml.name


def old_checkpoint(parent, tmp_path):
    """A copy of the parent folder whose checkpoint has no `rows_before` (a checkpoint of topopt before P4b-5)."""
    import shutil
    old = tmp_path / "parent-old"
    shutil.copytree(parent, old)
    tc.rewrite_checkpoint(old, meta_drop=("rows_before",))
    return old


# --- equal to the run of K + N passes ---------------------------------------------------------

@needs_direct
@pytest.mark.parametrize("k", [12, 19, 20])
def test_a_continuation_equals_the_run_of_the_full_length_byte_for_byte(cantilever, tmp_path, k):
    """K 12: inside a beta stage. K 19: the pass before a stage start. K 20: the checkpoint pass is a stage start (the replay sets the scale again)."""
    parent = cantilever.parent(k)
    before = parent_state(parent)
    out = tmp_path / "new"
    cont(cantilever.yaml, out, parent, N_IT - k)
    same_run(cantilever.base, out, run_json_drop=DROP)
    rec = read_json(out / "run.json")["continued_from"]
    assert Path(rec["run_dir"]) == parent.resolve()
    assert (rec["iteration"], rec["parent_iterations"], rec["iterations"], rec["more_iterations"], rec["replay"]) == (k, k, N_IT, N_IT - k, True)
    assert rec["checkpoint"] == "checkpoint.npz" and rec["checkpoint_sha256"] == sha256(parent / "checkpoint.npz")
    ss = rec["step_settings"]
    assert (ss["first_pass"], ss["last_pass"]) == (k, N_IT) and ss["objective_scale"]["set_again_at"] == [20]
    assert ss["rows_before"].startswith("used") and ss["stop_rules"]["enabled"] is False
    assert C.read_checkpoint(out).meta["continued_from"] == rec           # every checkpoint of a continuation holds the record
    assert C.describe(out)["continued_from"] == rec
    text = log(out)
    assert f"This run runs pass {k} again with its update step, then goes on to pass {N_IT}." in text
    assert f"the pass count of this run is {N_IT} ({k} + {N_IT - k} more passes)." in text
    assert "continue: step settings: this run runs pass %d to pass %d." % (k, N_IT) in text
    assert parent_state(parent) == before                                  # the parent is only read: no byte, no mtime, no new file
    lines = (out / "history.jsonl").read_bytes().splitlines(keepends=True)
    assert lines[:k] == (parent / "history.jsonl").read_bytes().splitlines(keepends=True)[:k]        # the lines before pass K are copied, byte for byte
    for n in range(k):
        assert (out / "snapshots" / f"it_{n:04d}.bin").read_bytes() == (parent / "snapshots" / f"it_{n:04d}.bin").read_bytes()
    assert read_json(out / "verdict.json")["resume_points"] == [] and (out / "job.DONE").exists()
    elapsed = [r["elapsed_s"] for r in history(out)]
    assert elapsed == sorted(elapsed) and elapsed[k] > elapsed[k - 1]       # the run time goes on from the parent (the clock is set back by its elapsed_s)


@needs_direct
def test_the_disk_fact_gives_the_bytes_that_the_copy_uses(cantilever, tmp_path):
    """No cap on the copy, but a fact with the bytes."""
    k, parent = 12, cantilever.parent(12)
    out = tmp_path / "new"
    cont(cantilever.yaml, out, parent, 3)
    text = log(out)
    snaps = sum((parent / "snapshots" / f"it_{n:04d}.bin").stat().st_size for n in range(k))        # the passes 0 to K - 1: pass K runs again
    hist = (parent / "history.jsonl").stat().st_size
    assert f"a copy of the history up to pass {k - 1}, {k} snapshots ({snaps} bytes)" in text
    assert f"The copy uses at most {snaps + hist} bytes of disk space" in text
    first = len(b"".join((parent / "history.jsonl").read_bytes().splitlines(keepends=True)[:k]))
    assert f"continue: copied {k} history lines ({first} bytes) and {k} snapshots ({snaps} bytes)." in text


@needs_direct
def test_the_pass_count_comes_from_the_flag_and_not_from_the_yaml(cantilever, work, tmp_path):
    """The YAML says 24 passes, the parent has 12 and N is 6: the new run has 18 passes and equals the run of 18 passes."""
    edit(work, lambda d: d["settings"].update(iterations=18))
    base18 = tmp_path / "base18"
    finish(work, base18)
    edit(work, lambda d: d["settings"].update(iterations=N_IT))
    out = tmp_path / "new"
    cont(work, out, cantilever.parent(12), 6)
    same_run(base18, out, run_json_drop=DROP)
    assert len(history(out)) == 19 and read_json(out / "run.json")["iterations"] == 18
    assert "settings.iterations of the YAML (24) is not used." in log(out)


def history_diffs(a, b):
    """The `(pass, path, value in a, value in b)` of every place where the history records of two runs differ. The timing fields are left out."""
    found = []

    def walk(x, y, path, it):
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(x.get(key, "<missing>"), y.get(key, "<missing>"), f"{path}.{key}" if path else key, it)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]", it)
        elif x != y:
            found.append((it, path, x, y))
    for ra, rb in zip(history(a), history(b)):
        walk(clean(ra), clean(rb), "", ra["iteration"])
    return found


STAGE_STARTS = (10, 18)   # the beta stages of the stress case start at these passes


@needs_direct
@pytest.mark.parametrize("k, rows", [(k, rows) for k in (9, 10, 17, 18) for rows in ("with", "without")])
def test_stress_rows_with_step_acceptance_continue_to_the_same_bytes_with_and_without_rows_before(stress, tmp_path, k, rows):
    """K 9 and 17: the pass before a stage start. K 10 and 18: a stage start (the stages start at 10 and 18), where the rows before and after pass K
    differ, so a replay that takes the wrong rows or the wrong `prev` shows (the stage note of the pass is missing). With `rows_before` all four are
    byte-equal. Without it the numbers are equal, and at a stage start the label `c_changed` of each stress row at pass K can differ."""
    parent = stress.parent(k)
    if rows == "without":
        parent = old_checkpoint(parent, tmp_path)
    out = tmp_path / "new"
    cont(stress.yaml, out, parent, N_IT - k)
    if rows == "without" and k in STAGE_STARTS:
        assert history_diffs(stress.base, out) == [(k, "constraints[0].c_changed", True, False), (k, "constraints[1].c_changed", True, False)]
        for f in [f"snapshots/{p.name}" for p in (stress.base / "snapshots").iterdir()] + ["result/final.stl", "result/preview.bin", "labels.bin"]:
            assert (stress.base / f).read_bytes() == (out / f).read_bytes(), f
        same_run(stress.base, out, exact=False, rtol=0.0, run_json_drop=DROP)
    else:
        same_run(stress.base, out, run_json_drop=DROP)
    assert any("step_acceptance" in r for r in history(out))
    text = log(out)
    if rows == "without":
        assert "the checkpoint has no rows_before" in text and "A stress row (KS) gives the same numbers." in text
        assert "When this pass is a stage start, the c_changed label of the row" in text
        assert "buckling row can differ" not in text and "run the YAML again from pass 0" not in text
        assert read_json(out / "run.json")["continued_from"]["step_settings"]["rows_before"].startswith("missing")
    else:
        assert "the checkpoint has no rows_before" not in text
    if k in STAGE_STARTS:
        assert any(r.get("stage", {}).get("iteration") == k for r in history(out) if r["iteration"] == k), "the stage note of pass K is in the history"


@needs_direct
@pytest.mark.parametrize("k", [4, 12])
def test_the_buckling_row_continues_to_the_same_bytes(column, row_timing, tmp_path, k):
    """The state of the row (what it evaluated, its last value) is in `rows_before`: the replay starts from the state before pass K."""
    out = tmp_path / "new"
    cont(column.yaml, out, column.parent(k), N_IT - k)
    same_run(column.base, out, run_json_drop=DROP)
    flags = [e["evaluated"] for e in br.entries(out)]
    assert flags == [e["evaluated"] for e in br.entries(column.base)] and any(flags[2:]) and not all(flags)
    assert [br.stateful(e) for e in br.entries(out)] == [br.stateful(e) for e in br.entries(column.base)]


@needs_direct
@pytest.mark.parametrize("edit", ["stripped", "main"])
def test_the_buckling_row_continues_from_a_checkpoint_with_no_start_vector_with_a_cold_first_solve(column, row_timing, tmp_path, edit):
    """#155: the start vector of the warm start is in `rows_before` (the first test above shows that the continuation needs it for the same solves).
    A parent without it (`stripped`: `v0` None; `main`: a checkpoint of topopt before the vector, with neither `v0` nor `tol`) still continues: the
    first evaluation after the checkpoint starts cold (more solves), the flags and the factors stay the same, and the facts do not speak about it."""
    import shutil
    k = 12
    parent = tmp_path / "parent"
    shutil.copytree(column.parent(k), parent)
    {"stripped": br.strip_vectors, "main": br.main_style}[edit](parent)
    out = tmp_path / "new"
    cont(column.yaml, out, parent, N_IT - k)
    same_run(column.base, out, exact=False, rtol=OLD_ROW_EXACT_RTOL, run_json_drop=DROP)
    got, want = [br.stateful(e) for e in br.entries(out)], [br.stateful(e) for e in br.entries(column.base)]
    flags = [e["evaluated"] for e in want]
    first = next(i for i in range(k, len(flags)) if flags[i])
    assert [e["evaluated"] for e in got] == flags and all(e["n_fail"] == 0 for e in got)
    assert got[:first] == want[:first] and got[first]["solves"] > want[first]["solves"]
    text = log(out)
    assert "the checkpoint has no rows_before" not in text and "buckling row can differ" not in text and "start vector" not in text


@needs_direct
@pytest.mark.parametrize("k, rtol", [(12, OLD_ROW_EXACT_RTOL), (4, OLD_ROW_MOVED_RTOL)])
def test_the_buckling_row_from_a_checkpoint_with_no_rows_before_is_close_and_says_so(column, row_timing, tmp_path, k, rtol):
    """A checkpoint of an older topopt has only the rows after pass K. The row changed its state at that pass, so the replay can differ:
    at K 12 the numbers are equal, at K 4 the evaluation schedule moves (the base skipped the passes 2 to 5, the continuation evaluates from 4)."""
    out = tmp_path / "new"
    cont(column.yaml, out, old_checkpoint(column.parent(k), tmp_path), N_IT - k)
    same_run(column.base, out, exact=False, rtol=rtol, run_json_drop=DROP)
    text = log(out)
    assert "the checkpoint has no rows_before" in text and "The buckling row can differ" in text
    assert f"Option: run the YAML again from pass 0 in a new folder with settings.iterations {N_IT}." in text
    assert "A stress row (KS)" not in text
    assert all(e["n_fail"] == 0 for e in br.entries(out))
    if k == 12:
        assert (out / "snapshots" / f"it_{N_IT:04d}.bin").read_bytes() == (column.base / "snapshots" / f"it_{N_IT:04d}.bin").read_bytes()


def test_an_amg_continuation_is_as_close_as_two_straight_runs(tmp_path):
    """AMG is not reproducible bit for bit (the start of the spectral radius estimate is random): the numbers are close, not equal."""
    (tmp_path / "c").mkdir()
    p = make_case(tmp_path / "c", h=1.0, iterations=N_IT)
    edit(p, lambda d: d["settings"].update(solver="amg"))
    base = tmp_path / "base"
    finish(p, base)
    edit(p, lambda d: d["settings"].update(iterations=12))
    parent = tmp_path / "parent"
    finish(p, parent)
    edit(p, lambda d: d["settings"].update(iterations=N_IT))
    out = tmp_path / "new"
    cont(p, out, parent, 12)
    assert "solver amg" in log(out)
    same_run(base, out, exact=False, rtol=AMG_RTOL, run_json_drop=DROP)


# --- a parent that did not finish: a fork ---------------------------------------------------------

def stopped_at(yaml_path, out, k):
    """A run that asks itself to stop after the checkpoint of pass k (tests/stop_at.py). The exit code is 5."""
    done = subprocess.run([sys.executable, str(STOP), f"pass:{k}", *run_args(yaml_path, out)], env=ENV1, capture_output=True, text=True, timeout=600)
    assert done.returncode == STOPPED_EXIT, f"exit {done.returncode}\n" + done.stdout[-1500:] + done.stderr[-2500:]


@needs_direct
def test_a_fork_from_a_stopped_run_equals_the_full_run(cantilever, tmp_path):
    """PR-4: a run that a request stopped at pass 9. No pass runs again: the new run starts at pass 10 and has 9 + 15 passes."""
    parent = tmp_path / "stopped"
    stopped_at(cantilever.yaml, parent, 9)
    before = parent_state(parent)
    out = tmp_path / "new"
    cont(cantilever.yaml, out, parent, 15)
    same_run(cantilever.base, out, run_json_drop=DROP)
    rec = read_json(out / "run.json")["continued_from"]
    assert (rec["iteration"], rec["parent_iterations"], rec["iterations"], rec["replay"]) == (9, N_IT, N_IT, False)
    assert rec["step_settings"]["first_pass"] == 10 and rec["step_settings"]["rows_before"].startswith("not used")
    text = log(out)
    assert "the parent did not finish. Its checkpoint is at pass 9, and the run was to end at pass 24." in text
    assert "This run is a fork: it starts at pass 10 from that checkpoint, runs no pass again" in text
    assert "stopped by a request at pass 9" in text
    assert parent_state(parent) == before and (parent / "job.STOPPED").exists()


@needs_direct
def test_a_parent_that_a_request_stopped_after_its_last_pass_is_a_replay_and_equals_the_longer_run(cantilever, tmp_path):
    """PR-4: a request that comes after the last pass ends the run before the post step. The checkpoint is complete (`RunStopped.complete`),
    so the continuation runs pass 24 again with its update step, as for a finished run. The request in the parent is not read by the new run."""
    parent = tmp_path / "stopped"
    stopped_at(cantilever.yaml, parent, N_IT)
    assert C.read_checkpoint(parent).meta["complete"] is True and read_json(parent / "status.json")["state"] == "stopped"
    assert (parent / "job.STOPPED").exists() and not (parent / "result").exists()
    before = parent_state(parent)
    longer_case = tmp_path / "longer_case"
    shutil.copytree(cantilever.yaml.parent, longer_case)
    edit(longer_case / cantilever.yaml.name, lambda d: d["settings"].update(iterations=N_IT + 6))
    longer = tmp_path / "longer"
    finish(longer_case / cantilever.yaml.name, longer)
    out = tmp_path / "new"
    cont(cantilever.yaml, out, parent, 6)
    same_run(longer, out, run_json_drop=DROP + ("yaml_path",))                # the longer run has its own YAML file
    rec = read_json(out / "run.json")["continued_from"]
    assert (rec["iteration"], rec["parent_iterations"], rec["iterations"], rec["replay"]) == (N_IT, N_IT, N_IT + 6, True)
    text = log(out)
    assert "continue: the parent was stopped by a request after its last pass." in text
    assert "This run runs that pass again and goes on. The request does not stop this run." in text
    assert "the post step runs now" not in text                              # that is the text of a resume
    assert read_json(out / "verdict.json")["state"] != "stopped_by_request" and (out / "job.DONE").exists() and not (out / "job.stop").exists()
    assert parent_state(parent) == before                                    # the parent is only read


@needs_direct
def test_a_fork_from_a_killed_run_equals_the_full_run(cantilever, tmp_path):
    parent = tmp_path / "killed"
    killed(cantilever.yaml, parent, "after_checkpoint:12")
    before = parent_state(parent)
    out = tmp_path / "new"
    cont(cantilever.yaml, out, parent, 12)
    same_run(cantilever.base, out, run_json_drop=DROP)
    assert read_json(out / "run.json")["continued_from"]["replay"] is False and "This run is a fork" in log(out)
    assert parent_state(parent) == before                                   # the kill left a stale job.lock; the continuation did not touch it
    assert (parent / "job.lock").exists() and not (out / "job.lock").exists()


@needs_direct
def test_a_fork_with_fewer_passes_than_the_parent_had_planned_is_a_run_of_the_new_length(cantilever, tmp_path):
    """A killed parent was to end at pass 24. The new run has 12 + 5 = 17 passes: the count of the parent plan does not matter."""
    parent = tmp_path / "killed"
    killed(cantilever.yaml, parent, "after_checkpoint:12")
    out = tmp_path / "new"
    cont(cantilever.yaml, out, parent, 5)
    assert [r["iteration"] for r in history(out)] == list(range(18)) and read_json(out / "run.json")["iterations"] == 17
    assert len(list((out / "snapshots").iterdir())) == 18 and (out / "result" / "final.stl").is_file()


# --- a parent that runs: its checkpoint is read once (fix round 1, finding 1) ------------------------

def test_a_parent_that_goes_on_while_the_run_starts_is_read_once_and_the_run_has_n_new_passes(cantilever, tmp_path, monkeypatch):
    """The parent writes the passes 7 to 12 after the CLI read its checkpoint (pass 6). The pass count (6 + 3), the plan, run.json, the log,
    `checkpoint_sha256` and the copy come from that one read: 3 new passes run, and nothing of the passes 7 to 12 of the parent is copied."""
    from topopt import cli
    early, later, parent = tmp_path / "at6", tmp_path / "at12", tmp_path / "parent"
    killed(cantilever.yaml, early, "after_checkpoint:6")
    killed(cantilever.yaml, later, "after_checkpoint:12")
    shutil.copytree(early, parent)
    read = sha256(parent / "checkpoint.npz")
    assert read != sha256(later / "checkpoint.npz")
    plan, seen = cli._plan_continue, []

    def parent_goes_on(*args, **kwargs):
        if not seen:   # the first plan is the one of `cmd_run`: the parent writes the files of the passes 7 to 12 before it
            for name in ("checkpoint.npz", "checkpoint.prev.npz", "history.jsonl", "status.json"):
                shutil.copy2(later / name, parent / name)
            for s in (later / "snapshots").iterdir():
                shutil.copy2(s, parent / "snapshots" / s.name)
            (parent / "job.lock").write_text(json.dumps({"pid": os.getpid()}))    # a live holder: the parent runs
        seen.append(args[1].settings.iterations)
        return plan(*args, **kwargs)

    monkeypatch.setattr(cli, "_plan_continue", parent_goes_on)
    out = tmp_path / "new"
    assert main(run_args(cantilever.yaml, out, *flags_for(parent, 3))) == 0
    assert (parent / "history.jsonl").read_bytes() == (later / "history.jsonl").read_bytes() and len(seen) >= 2    # the parent did move
    assert set(seen) == {9}                                                         # both plans (before the folder, under the lock) have 6 + 3 passes
    run = read_json(out / "run.json")
    rec = run["continued_from"]
    assert (run["iterations"], rec["iteration"], rec["iterations"], rec["more_iterations"], rec["replay"]) == (9, 6, 9, 3, False)
    assert rec["step_settings"]["first_pass"] == 7 and rec["step_settings"]["last_pass"] == 9
    assert rec["checkpoint_sha256"] == read                                         # the hash of the bytes that were read, not of the file now
    assert [r["iteration"] for r in history(out)] == list(range(10))                # the lines of the passes 0 to 6 are copied; the passes 7 to 9 run
    assert sorted(p.name for p in (out / "snapshots").iterdir()) == [f"it_{n:04d}.bin" for n in range(10)]
    for n in range(7):                                                              # the copy is of the passes up to K only
        assert (out / "snapshots" / f"it_{n:04d}.bin").read_bytes() == (early / "snapshots" / f"it_{n:04d}.bin").read_bytes()
    text = log(out)
    assert "the pass count of this run is 9 (6 + 3 more passes)." in text and "pass count of this run is 15" not in text
    assert "the parent is running now" in text and "starts from the checkpoint that it read, at pass 6" in text
    assert (out / "job.DONE").exists() and (out / "result" / "final.stl").is_file()


# --- kill and resume, a chain -----------------------------------------------------------------------

@needs_direct
@pytest.mark.parametrize("point", ["after_checkpoint:12", "after_checkpoint:16", "before_checkpoint:17"])
def test_a_killed_continuation_resumes_with_a_plain_resume_and_the_pass_count_of_the_checkpoint(cantilever, tmp_path, point):
    """The YAML of the parent says 12 passes. The continuation has 24. A plain `--resume` with that YAML takes the count from the checkpoint."""
    parent_yaml, parent = cantilever.parent_yaml(12), cantilever.parent(12)
    out = tmp_path / "new"
    killed(parent_yaml, out, point, "--continue-from", str(parent), "--more-iterations", "12")
    resume(parent_yaml, out)
    same_run(cantilever.base, out, run_json_drop=DROP + ("yaml_path",))
    text = log(out)
    assert f"resume: this run is a continuation of {parent.resolve()} (its pass 12 plus 12 more passes). Its pass count is 24, from the checkpoint." in text
    assert "The settings.iterations of the YAML (12) is not used." in text
    run_json = read_json(out / "run.json")
    assert run_json["iterations"] == 24 and run_json["continued_from"] == C.read_checkpoint(out).meta["continued_from"]
    assert read_json(out / "verdict.json")["resume_points"][-1]["iteration"] == int(point.split(":")[1]) - (1 if point.startswith("before") else 0)


@needs_direct
def test_a_resume_of_a_killed_continuation_with_a_changed_input_is_refused_and_the_override_goes_on(cantilever, tmp_path, capsys):
    """`--resume` of a continuation uses `plan_resume`: a changed input is a refusal that writes nothing, and `--allow-changed-inputs` goes on.
    The pass count still comes from the checkpoint, whatever the YAML says."""
    work = tmp_path / "case"
    shutil.copytree(cantilever.parent_yaml(12).parent, work)
    yaml_path, parent, out = work / cantilever.yaml.name, cantilever.parent(12), tmp_path / "new"
    killed(yaml_path, out, "after_checkpoint:16", "--continue-from", str(parent), "--more-iterations", "12")
    before = tree_state(out)
    edit(yaml_path, lambda d: d["settings"].update(volume_fraction=0.35))
    err = refused(capsys, yaml_path, out, "--resume")
    assert "Resume stopped: the inputs changed since the checkpoint (iteration 16)" in err and "settings.volume_fraction: 0.3 -> 0.35" in err
    assert "--allow-changed-inputs" in err and "settings.iterations" not in err
    assert tree_state(out) == before                                           # the refusal wrote nothing in the run folder
    resume(yaml_path, out, "--allow-changed-inputs")
    text = log(out)
    assert "resume: the inputs changed since the checkpoint: settings.volume_fraction: 0.3 -> 0.35. The override is given" in text
    assert "Its pass count is 24, from the checkpoint." in text
    assert read_json(out / "run.json")["iterations"] == 24 and [r["iteration"] for r in history(out)] == list(range(25))
    assert (out / "job.DONE").exists() and read_json(out / "run.json")["continued_from"]["iteration"] == 12


@needs_direct
def test_a_continuation_killed_before_its_first_checkpoint_is_started_again_with_the_same_command(cantilever, tmp_path, capsys):
    parent_yaml, parent = cantilever.parent_yaml(12), cantilever.parent(12)
    out = tmp_path / "new"
    killed(parent_yaml, out, "before_checkpoint:12", "--continue-from", str(parent), "--more-iterations", "12")
    assert not (out / "checkpoint.npz").exists() and (out / "history.jsonl").exists()
    err = refused(capsys, parent_yaml, out, "--resume")
    assert "no checkpoint" in err
    cont(parent_yaml, out, parent, 12)                                       # the new run cleans the folder of the killed one
    same_run(cantilever.base, out, run_json_drop=DROP + ("yaml_path",))


@needs_direct
def test_a_chain_of_two_continuations_equals_the_full_run(cantilever, tmp_path):
    """9 + 6 = 15 passes, then 15 + 9 = 24. The second parent is a continuation, and the first one is never changed."""
    first, second = tmp_path / "first", tmp_path / "second"
    cont(cantilever.yaml, first, cantilever.parent(9), 6)
    assert len(history(first)) == 16
    before = parent_state(first)
    cont(cantilever.yaml, second, first, 9)
    same_run(cantilever.base, second, run_json_drop=DROP)
    rec = read_json(second / "run.json")["continued_from"]
    assert Path(rec["run_dir"]) == first.resolve() and (rec["iteration"], rec["parent_iterations"], rec["iterations"]) == (15, 15, 24)
    assert read_json(first / "run.json")["continued_from"]["run_dir"] == str(cantilever.parent(9).resolve())      # the first one keeps its own record
    assert parent_state(first) == before


# --- the new folder, the flags, the inputs ---------------------------------------------------------------

def flags_for(parent, n=5):
    return ["--continue-from", str(parent), "--more-iterations", str(n)]


def test_a_new_folder_equal_to_or_inside_the_parent_is_refused_and_nothing_is_made(cantilever, tmp_path, capsys):
    parent = cantilever.parent(12)
    before = parent_state(parent)
    err = refused(capsys, cantilever.yaml, parent, *flags_for(parent))
    assert "--out is the parent run folder" in err and "another --out" in err and "--resume" in err
    err = refused(capsys, cantilever.yaml, str(parent) + "/", *flags_for(parent))                     # the same folder, written another way
    assert "--out is the parent run folder" in err
    link = tmp_path / "link"
    link.symlink_to(parent, target_is_directory=True)
    err = refused(capsys, cantilever.yaml, link, *flags_for(parent))                                 # the same folder by a link
    assert "--out is the parent run folder" in err
    err = refused(capsys, cantilever.yaml, parent / "more", *flags_for(parent))
    assert "is inside the parent run folder" in err and not (parent / "more").exists()
    err = refused(capsys, cantilever.yaml, parent / "snapshots" / "x", *flags_for(parent))
    assert "is inside the parent run folder" in err and not (parent / "snapshots" / "x").exists()
    assert parent_state(parent) == before


def test_a_parent_inside_the_folders_that_the_new_run_deletes_is_refused(cantilever, tmp_path, capsys):
    """`topopt run` deletes `snapshots/`, `result/` and `verify/` of its folder. A parent there would be deleted."""
    import shutil
    out = tmp_path / "new"
    for name in ("snapshots", "result", "verify"):
        inner = out / name / "p"
        shutil.copytree(cantilever.parent(12), inner)
        before = parent_state(inner)
        err = refused(capsys, cantilever.yaml, out, *flags_for(inner))
        assert f"is inside {out / name}" in err and "deletes that folder" in err
        assert parent_state(inner) == before
        shutil.rmtree(out / name)
    assert list(out.iterdir()) == []


@pytest.fixture
def case_insensitive(monkeypatch):
    """A file system that folds the letter case (macOS APFS) on a Linux disk: `os.stat` finds a path that exists in another case.
    The real names do not change, and `os.lstat` (used by `resolve()`) is not changed: `resolve()` does not fold the case on a Mac either."""
    real = os.stat

    def folded(path, *args, **kwargs):
        try:
            return real(path, *args, **kwargs)
        except FileNotFoundError:
            at = Path(os.fspath(path))
            if not at.is_absolute():
                raise
            here = Path(at.anchor)
            for part in at.parts[1:]:
                names = {n.lower(): n for n in os.listdir(here)}
                if part.lower() not in names:
                    raise
                here = here / names[part.lower()]
            return real(here, *args, **kwargs)

    monkeypatch.setattr(os, "stat", folded)


def test_a_new_folder_is_compared_with_the_parent_by_identity_and_not_by_the_text_of_the_path(cantilever, tmp_path, capsys, case_insensitive):
    """Fix round 1, finding 3: on a Mac `~/Runs/Hook/more` and `~/runs/hook` are one folder. The check takes the deepest folder of the new path that
    exists, and compares it with the parent by `samefile` (the same for a bind mount on Linux). The new folder must not be made in the parent."""
    parent = tmp_path / "runs" / "hook"
    shutil.copytree(cantilever.parent(12), parent)
    before = parent_state(parent)
    err = refused(capsys, cantilever.yaml, tmp_path / "RUNS" / "HOOK", *flags_for(parent))                       # the parent in another case
    assert "--out is the parent run folder" in err
    err = refused(capsys, cantilever.yaml, tmp_path / "RUNS" / "HOOK" / "more", *flags_for(parent))             # a new folder in it: it does not exist yet
    assert "is inside the parent run folder" in err
    err = refused(capsys, cantilever.yaml, tmp_path / "Runs" / "Hook" / "a" / "b", *flags_for(parent))          # no part of the new path below the parent exists
    assert "is inside the parent run folder" in err
    assert parent_state(parent) == before and sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["hook"]


def test_a_parent_in_a_folder_that_the_new_run_deletes_is_found_by_identity(cantilever, tmp_path, capsys, case_insensitive):
    """The new folder `NEW` is written in another case than the folder that holds the parent: `NEW/snapshots` is still the folder with the parent in it."""
    inner = tmp_path / "new" / "snapshots" / "p"
    shutil.copytree(cantilever.parent(12), inner)
    before = parent_state(inner)
    err = refused(capsys, cantilever.yaml, tmp_path / "NEW", *flags_for(inner))
    assert "deletes that folder" in err and "Options: give another --out, or move the parent folder." in err
    assert parent_state(inner) == before


def test_a_new_folder_below_a_link_to_the_parent_is_refused_and_nothing_is_made(cantilever, tmp_path, capsys):
    """A link into the parent, with the new folder below it (not the link itself): the real place is in the parent."""
    parent = tmp_path / "parent"
    shutil.copytree(cantilever.parent(12), parent)
    link = tmp_path / "link"
    link.symlink_to(parent, target_is_directory=True)
    before = parent_state(parent)
    err = refused(capsys, cantilever.yaml, link / "a" / "b", *flags_for(parent))
    assert "is inside the parent run folder" in err
    assert parent_state(parent) == before and not (parent / "a").exists()


@pytest.mark.parametrize("value", ["0", "-3", "abc", "2.5", "1e2", " ", "\u00b2", "--5", "+-3", "1_0", "\u00bd"])   # "\u00b2": isdigit() is true, int() fails
def test_a_wrong_number_of_passes_is_refused_before_anything_is_made(cantilever, tmp_path, capsys, value):
    out = tmp_path / "new"
    err = refused(capsys, cantilever.yaml, out, "--continue-from", str(cantilever.parent(12)), f"--more-iterations={value}")
    assert "must be a whole number of 1 or more" in err and "there is no upper limit" in err and not out.exists()


def test_the_flags_go_together_and_not_with_resume(cantilever, tmp_path, capsys):
    out = tmp_path / "new"
    parent = cantilever.parent(12)
    assert "--continue-from and --more-iterations go together" in refused(capsys, cantilever.yaml, out, "--continue-from", str(parent))
    assert "--continue-from and --more-iterations go together" in refused(capsys, cantilever.yaml, out, "--more-iterations", "5")
    assert "--resume goes on in the folder of --out" in refused(capsys, cantilever.yaml, out, "--resume", *flags_for(parent))
    assert "--allow-changed-inputs needs --resume, --continue-from or --fine-from" in refused(capsys, cantilever.yaml, out, "--allow-changed-inputs")
    assert "is not a folder" in refused(capsys, cantilever.yaml, out, *flags_for(tmp_path / "nowhere"))
    assert not out.exists()


def test_a_parent_with_no_checkpoint_is_refused_with_the_reason_and_nothing_is_made(cantilever, tmp_path, capsys):
    empty, out = tmp_path / "empty", tmp_path / "new"
    empty.mkdir()
    err = refused(capsys, cantilever.yaml, out, *flags_for(empty))
    assert "no checkpoint" in err and "without --resume" not in err and not out.exists()
    assert tree_state(empty) == {}


def test_changed_inputs_are_refused_with_facts_and_options_and_nothing_is_made(cantilever, work, tmp_path, capsys):
    parent = cantilever.parent(12)
    before, out = parent_state(parent), tmp_path / "new"
    edit(work, lambda d: d["settings"].update(volume_fraction=0.35))
    err = refused(capsys, work, out, *flags_for(parent))
    assert "settings.volume_fraction: 0.3 -> 0.35" in err and "--allow-changed-inputs" in err
    assert "restore the old inputs" in err and "run the YAML as a new run in a new folder" in err
    assert "settings.iterations" not in err                                  # the pass count is a soft key of a continuation
    assert not out.exists() and parent_state(parent) == before


@needs_direct
def test_the_override_continues_with_the_new_inputs_and_says_so(cantilever, work, tmp_path):
    parent = cantilever.parent(12)
    out = tmp_path / "new"
    edit(work, lambda d: d["settings"].update(volume_fraction=0.35))
    cont(work, out, parent, 12, "--allow-changed-inputs")
    assert "continue: the inputs changed since the checkpoint: settings.volume_fraction: 0.3 -> 0.35. The override is given" in log(out)
    rows, old = history(out), history(parent)
    assert [clean(r) for r in rows[:12]] == [clean(r) for r in old[:12]] and rows[-1]["volume_fraction"] > 0.32
    assert len(rows) == N_IT + 1


@needs_direct
def test_a_change_of_the_yaml_text_only_is_a_note_and_not_a_refusal(cantilever, tmp_path):
    """The YAML of the parent (12 passes) with a comment added: the parsed content is the same, so it is a note."""
    import shutil
    parent = cantilever.parent(12)
    shutil.copytree(cantilever.parent_yaml(12).parent, tmp_path / "case")
    yaml_path = tmp_path / "case" / cantilever.yaml.name
    yaml_path.write_text("# a comment\n" + yaml_path.read_text())
    out = tmp_path / "new"
    cont(yaml_path, out, parent, 12)
    same_run(cantilever.base, out, run_json_drop=DROP + ("yaml_path",))
    text = log(out)
    assert "continue: the YAML text changed (comments or format only)" in text and "the inputs changed since the checkpoint" not in text
    assert "settings.iterations of the YAML (12) is not used." in text


# --- a parent that a stop rule ended -----------------------------------------------------------------

@pytest.fixture
def stopped_by_rule(tmp_path):
    """The cantilever with stop rules that fire before pass 24: the run (`parent`) and the run of 24 passes with the rules off (`base`)."""
    (tmp_path / "c").mkdir()
    p = make_case(tmp_path / "c", h=1.0, iterations=N_IT)
    edit(p, lambda d: d["settings"].update(stop_rules={"enabled": True, "objective_change": 1.0, "window": 2, "grayness_max": 1.0}))
    parent, base = tmp_path / "parent", tmp_path / "base"
    finish(p, parent)
    finish(p, base, "--no-stop-rules")
    k = C.describe(parent)["iteration"]
    assert k < N_IT and read_json(parent / "verdict.json")["state"] == "stopped_by_rule"
    return p, parent, base, k


@needs_direct
def test_a_parent_that_a_stop_rule_ended_is_judged_again_at_its_last_pass_with_the_rules_on(stopped_by_rule, tmp_path):
    """With the rules on, the rule is judged again at pass K with the same numbers. The fact gives the options."""
    p, parent, _base, k = stopped_by_rule
    rule = C.read_checkpoint(parent).meta["stop"]["rule"]
    before = parent_state(parent)
    out = tmp_path / "new"
    cont(p, out, parent, 5)
    assert read_json(out / "verdict.json")["state"] == "stopped_by_rule"
    assert [clean(r) for r in history(out)] == [clean(r) for r in history(parent)] and len(history(out)) == k + 1       # ends at pass K again
    text = log(out)
    assert f"the parent ended at pass {k} by the stop rule '{rule}'. The stop rules are on in this run, so the rule is judged again at pass {k}" in text
    assert "Options: --no-stop-rules (MCP: stop_rules=false), or other thresholds in settings.stop_rules." in text
    assert parent_state(parent) == before


@needs_direct
def test_a_parent_that_a_stop_rule_ended_goes_on_with_the_rules_off_and_equals_the_full_run(stopped_by_rule, tmp_path):
    p, parent, base, k = stopped_by_rule
    rule = C.read_checkpoint(parent).meta["stop"]["rule"]
    out = tmp_path / "new"
    cont(p, out, parent, N_IT - k, "--no-stop-rules")
    same_run(base, out, run_json_drop=DROP)
    assert read_json(out / "verdict.json")["state"] == read_json(base / "verdict.json")["state"] != "stopped_by_rule"
    assert f"the parent ended at pass {k} by the stop rule '{rule}'. The stop rules are off in this run, so it goes on to pass {N_IT}." in log(out)
    assert read_json(out / "run.json")["continued_from"]["step_settings"]["stop_rules"]["enabled"] is False


# --- pass numbers above 9999  -------------------------------------------------------
# The name of a snapshot stays `it_{:04d}.bin`: a number of five digits makes a longer name, and a text sort of the names is wrong
# (`it_10000.bin` < `it_9999.bin`). Every reader takes the number from the name. These tests use a made-up run folder, because no test can run 10,000 passes.

BIG = 10003


def big_run(folder, last=BIG, k=None, complete=True):
    """A run folder with the passes 0 to `last`: a snapshot of one byte and a history line for each, and the checkpoint of pass `k` (made up)."""
    folder = Path(folder)
    (folder / "snapshots").mkdir(parents=True)
    for n in range(last + 1):
        (folder / "snapshots" / f"it_{n:04d}.bin").write_bytes(bytes([n % 251]))
    with open(folder / "history.jsonl", "w") as f:
        for n in range(last + 1):
            f.write(json.dumps({"iteration": n, "elapsed_s": n * 0.001, "compliance": 1.0 + n}) + "\n")
    if k is not None:
        C.write_checkpoint(folder, arrays_of(1), meta_of(k, complete=complete))
    return folder


def test_the_text_order_of_the_names_is_wrong_above_9999(tmp_path):
    """The reason for the tests below: a reader that sorts the names as text takes pass 9999 as the last one."""
    d = big_run(tmp_path / "run", last=10001)
    assert sorted(p.name for p in (d / "snapshots").iterdir())[-1] == "it_9999.bin"
    assert max(C._snapshot_number(p) for p in (d / "snapshots").iterdir()) == 10001


def test_the_readers_of_a_run_folder_order_the_snapshots_by_pass_number(tmp_path):
    from topopt import feedback, post
    from topopt.dashboard_server import RunSource
    from topopt.replay import Replay
    d = big_run(tmp_path / "run", last=BIG)
    its = RunSource(d).snapshot_iterations()                                    # the dashboard
    assert its == list(range(BIG + 1)) and its[-1] == BIG
    assert RunSource(d).snapshot_bytes(BIG) == bytes([BIG % 251])
    assert post._last_snapshot(d).name == f"it_{BIG}.bin"                       # the post step takes the last snapshot for the mesh
    clock = [0.0]
    rep = Replay(d, now=lambda: clock[0])                                       # the replay of a recorded run
    rep.start()
    assert rep.read(f"snapshots/it_{BIG}.bin") is None                          # the clock is at 0 s: only pass 0 is shown
    clock[0] = 1e6
    assert rep.read(f"snapshots/it_{BIG}.bin") == bytes([BIG % 251]) and rep.snapshot_iterations(list(range(BIG + 1)))[-1] == BIG
    assert feedback.run_facts(d)["files"]["snapshots"] == BIG + 1               # the facts of a run count the files
    tool = importlib.util.spec_from_file_location("compare_runs", Path(__file__).resolve().parent.parent / "tools" / "compare_runs.py")
    compare_runs = importlib.util.module_from_spec(tool)
    tool.loader.exec_module(compare_runs)                                       # tools/compare_runs.py orders the files for its report
    names = [p.name for p in compare_runs.snapshot_files(d)]
    assert names[:2] == ["it_0000.bin", "it_0001.bin"] and names[-2:] == [f"it_{BIG - 1}.bin", f"it_{BIG}.bin"]


def test_the_run_writer_names_a_snapshot_of_five_digits_and_the_readers_find_it(tmp_path):
    from topopt.dashboard_server import RunSource
    d = big_run(tmp_path / "run", last=9999)
    out = RunWriter(d, quiet=True)
    out.write_snapshot(10000, np.full(3, 7, np.uint8))
    assert (d / "snapshots" / "it_10000.bin").read_bytes() == bytes([7, 7, 7])
    assert RunSource(d).snapshot_iterations()[-2:] == [9999, 10000]


def test_a_continuation_copies_the_snapshots_by_number_above_9999(tmp_path):
    """A replay (the checkpoint is complete at pass 10001) keeps the passes 0 to 10000; a fork keeps 0 to 10001."""
    for complete, kept in ((True, 10000), (False, 10001)):
        parent = big_run(tmp_path / f"parent{int(complete)}", last=10001, k=10001, complete=complete)
        ck = C.read_checkpoint(parent)
        plan = C.ResumePlan(ck, ["continue: made up"], [], continued_from={}, cont=C.Continuation(parent, 10001, 5, 10006, complete, True, {}))
        assert plan.kept_iteration == kept
        before = parent_state(parent)
        out = RunWriter(tmp_path / f"new{int(complete)}", quiet=True)
        out.clean()
        C.apply_continue(plan, out)
        names = sorted(p.name for p in (out.dir / "snapshots").iterdir())
        assert len(names) == kept + 1 and f"it_{kept}.bin" in names and f"it_{kept + 1}.bin" not in names
        rows = history(out.dir)
        assert [r["iteration"] for r in rows] == list(range(kept + 1))
        assert (out.dir / "history.jsonl").read_bytes() == b"".join((parent / "history.jsonl").read_bytes().splitlines(keepends=True)[:kept + 1])
        text = log(out.dir)
        assert "missing" not in text and f"history up to pass {kept}, {kept + 1} snapshots" in text
        assert parent_state(parent) == before


def test_a_resume_cuts_the_snapshots_by_number_above_9999(tmp_path):
    d = big_run(tmp_path / "run", last=BIG, k=10001, complete=False)
    plan = C.ResumePlan(C.read_checkpoint(d), ["resume: made up"], [])
    C.apply_plan(plan, RunWriter(d, quiet=True))
    names = sorted(p.name for p in (d / "snapshots").iterdir())
    assert len(names) == 10002 and "it_10001.bin" in names and "it_10002.bin" not in names
    assert history(d)[-1]["iteration"] == 10001
    text = log(d)
    assert "Dropped snapshots 10002 to 10003" in text and "missing" not in text
