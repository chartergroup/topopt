"""The buckling row and a resume (A6b, CONTRACTS.md section 39, test T13).

The state of the row (`n_known`, `last_value`, `last_eval`, the counts) is part of `MmaState.rows`, so a run that was killed and resumed
must do the same passes as the run that was not stopped: the same `evaluated` flags, the same state, the same bytes at 1 thread.
The times of the row (`*_seconds`) differ from run to run and are not compared (`tools/compare_runs.py` strips only the top-level
`elapsed_s` and `t_iter_s`; `test_checkpoint.same_run` strips the keys of its `TIMING`, and this file adds the three keys of the row).

The warm start (#155) puts the start vector `v0` of the eigen solve in the state. It is an array in the `.npz` of the checkpoint, also for
`rows_before`. A resume that did not restore it would start the next solve cold: the entry keeps `solves`, which T13 compares, so it fails
(`test_a_resume_without_the_start_vector_...` shows that with a checkpoint whose vector was taken out).
"""
import json
import shutil

import numpy as np
import pytest

import test_checkpoint as tc
from helpers import needs_direct_solver
from test_buckling_row import column_yaml
from test_buckling_row_wiring import with_buckling
from topopt import checkpoint as C
from topopt.cli import main

ROW_TIMING = {"eigen_seconds", "kg_seconds", "adjoint_seconds"}
F = 0.08            # between the start and the lowest factor of this column (T6 of `test_buckling_row.py`): both regimes appear
COLUMN = dict(a=3, b=3, length=30, load=1500.0, lateral=75.0, iterations=24)


def entries(folder):
    return [next(c for c in r["constraints"] if c["name"] == "buckling:axial") for r in tc.history(folder)]


def stateful(e):
    """The part of a row entry that a resume must reproduce."""
    return {k: v for k, v in e.items() if k not in ROW_TIMING}


NAME = "buckling:axial"


@pytest.fixture
def row_timing(monkeypatch):
    monkeypatch.setattr(tc, "TIMING", tc.TIMING | ROW_TIMING)


def rewrite_row_states(out, edit_state):
    """Write `checkpoint.npz` of a folder again with `edit_state(state)` applied to each row state of `mma.rows` and `rows_before`, and without
    the arrays of the row states. `edit_state` must take the references out of the states. No `.prev` file stays."""
    ck = C.read_checkpoint(out)
    for rows in (ck.meta["mma"]["rows"], ck.meta.get("rows_before") or {}):
        for state in rows.values():
            edit_state(state)
    assert not any(C.row_array_refs(r) for r in (ck.meta["mma"]["rows"], ck.meta.get("rows_before")))
    arrays = {k: v for k, v in ck.arrays.items() if not k.startswith(C.ROW_ARRAY)}
    (out / C.PREV_FILE).unlink(missing_ok=True)
    (out / C.CHECKPOINT_FILE).unlink()
    C.write_checkpoint(out, arrays, ck.meta)


def strip_vectors(out):
    """The checkpoint as if the restore had lost the start vectors: `v0` is None in every row state."""
    rewrite_row_states(out, lambda state: state.update(v0=None))


def main_style(out):
    """The checkpoint that topopt wrote before #155: the row states have neither `v0` nor `tol`, and the file has no array of a row."""
    rewrite_row_states(out, lambda state: (state.pop("v0", None), state.pop("tol", None)))


@needs_direct_solver
def test_a_killed_run_with_the_row_resumes_to_the_same_passes_and_bytes(tmp_path, row_timing):
    p = with_buckling(column_yaml(tmp_path / "c", **COLUMN), min_factor=F)
    base = tmp_path / "base"
    tc.finish(p, base)
    flags = [e["evaluated"] for e in entries(base)]
    assert flags[:2] == [True, True] and any(flags[2:]) and not all(flags)            # both regimes in the run
    assert all(e["n_fail"] == 0 for e in entries(base))
    assert all(e["residual_adjoint"] <= 1e-6 for e in entries(base) if e["evaluated"])
    ck = C.read_checkpoint(base)                                                       # the finished run: a complete checkpoint with rows_before
    assert ck.meta["complete"] and C.row_array_refs(ck.meta["rows_before"]) and C.row_array_refs(ck.meta["mma"]["rows"])
    for n, point in enumerate(("after_checkpoint:9", "before_checkpoint:14", "after_checkpoint:3")):
        out = tmp_path / f"r{n}"
        tc.killed(p, out, point)
        killed_ck = C.read_checkpoint(out)                                             # the start vector is in the file, as an exact float64 array
        ref = killed_ck.meta["mma"]["rows"][NAME]["v0"]
        assert ref == {C.ARRAY_REF: "rowvec_0"}
        vec = killed_ck.arrays["rowvec_0"]
        assert vec.dtype == np.float64 and vec.ndim == 1 and np.all(np.isfinite(vec)) and vec.any()
        assert killed_ck.meta["mma"]["rows"][NAME]["tol"] == 1e-8
        tc.resume(p, out)
        tc.same_run(base, out)
        assert [stateful(e) for e in entries(out)] == [stateful(e) for e in entries(base)]
        assert [e["evaluated"] for e in entries(out)] == flags


@needs_direct_solver
def test_a_resume_with_a_changed_min_factor_asks_for_the_override_and_evaluates_at_once(tmp_path, capsys):
    p = with_buckling(column_yaml(tmp_path / "c", **COLUMN), min_factor=F)
    base = tmp_path / "base"
    tc.finish(p, base)
    out = tmp_path / "r"
    tc.killed(p, out, "after_checkpoint:9")
    with_buckling(p, min_factor=F * 1.1)
    assert main(tc.run_args(p, out, "--resume")) == 1                                   # a changed input is refused
    assert "buckling.min_factor" in capsys.readouterr().err
    tc.resume(p, out, "--allow-changed-inputs")
    got, old = entries(out), entries(base)
    assert [stateful(e) for e in got[:10]] == [stateful(e) for e in old[:10]]            # the passes before the resume are the old ones
    assert got[10]["evaluated"] and got[10]["reason"] == "unknown"                       # the saved state is of another setting
    assert got[10]["min_factor"] == pytest.approx(F * 1.1)
    assert all(e["n_fail"] == 0 for e in got)
    assert json.loads((out / "verdict.json").read_text())["resume_points"][-1]["iteration"] == 9


@needs_direct_solver
def test_a_resume_without_the_start_vector_in_the_checkpoint_does_a_different_solve_and_one_from_main_starts_cold(tmp_path, row_timing):
    """The control of T13. The same killed run is resumed from three files: the checkpoint as it is (T13: all the same), one whose start vector was taken
    out (`v0` None, the arrays gone: what a restore that forgot the vector would give), and one of main (neither `v0` nor `tol`). The last two start the
    first evaluation after the kill cold, so its `solves` differs from the run that was not stopped; they agree with each other, the flags and the factors
    stay the same, and the next evaluations are warm again (the vector is kept again after the first solve)."""
    p = with_buckling(column_yaml(tmp_path / "c", **COLUMN), min_factor=F)
    base = tmp_path / "base"
    tc.finish(p, base)
    flags = [e["evaluated"] for e in entries(base)]
    kill = 9
    first = next(i for i in range(kill + 1, len(flags)) if flags[i])                   # the first evaluation after the resume
    runs = {}
    dead = tmp_path / "dead"
    tc.killed(p, dead, f"after_checkpoint:{kill}")                                      # one killed run; each resume gets its own copy
    for name, edit in (("same", None), ("stripped", strip_vectors), ("main", main_style)):
        out = tmp_path / name
        shutil.copytree(dead, out)
        if edit is not None:
            edit(out)
        tc.resume(p, out)
        runs[name] = [stateful(e) for e in entries(out)]
    want = [stateful(e) for e in entries(base)]
    assert runs["same"] == want
    for name in ("stripped", "main"):
        got = runs[name]
        assert [e["evaluated"] for e in got] == flags and all(e["n_fail"] == 0 for e in got)
        assert got[:first] == want[:first]                                              # the passes before the first evaluation are the same
        assert got[first]["solves"] > want[first]["solves"]                              # cold against warm
        for g, w in zip(got, want):
            if g["evaluated"]:
                assert g["lowest_factor"] == pytest.approx(w["lowest_factor"], rel=1e-8)
    assert runs["stripped"] == runs["main"]
