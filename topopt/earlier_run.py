"""A run folder that holds a refused job and an earlier run: the rule that finds it, and the mark for a row (CONTRACTS.md section 24.1).

A fresh `topopt run` that an input check refuses keeps the earlier run in its folder (F4-43). Only `status.json` and the job files show the
refused job. A reader that gives facts of a folder must not give the facts of the earlier run as the facts of the refused job.
- `refused_over_earlier_run(d, status, job)` is the one rule (P9-18). `run_status` and `run_result` read it through
  `server._refused_over_earlier_run`. `list_runs`, `run_history` and `compare_runs` use it too (P9-19).
- `mark_row(row, keys)` is the one mark for a row (P9-19): the keys of the earlier run move into the key `earlier_run`, with a note, and they are
  `null` at the top of the row. The shape is the shape of the key `earlier_run` of `run_status`.

This module needs no other topopt module at import time. The rule imports `cli` (for the sentence), `snapshot` and `runjob` when it needs them,
and only for a status that has passed the cheap tests, so a reader that lists many folders loads them at most once.
"""
from __future__ import annotations

import datetime
from pathlib import Path

UNREAD = object()   # the argument `job` when the caller has not read the job record: the rule reads it, and only if it needs it


def _epoch(text) -> float | None:
    """Epoch seconds of an ISO time text (`started_at` of job.json), or None."""
    try:
        return datetime.datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def refused_over_earlier_run(d, status: dict | None, job: dict | None | object = UNREAD) -> bool:
    """True if `status` is the status of a job that an input check refused in a folder that holds an earlier run (F4-43, P9-18).

    Such a job writes `status.json` (state `error`, pass 0, step `setup`) and the job files, and nothing else
    (CONTRACTS.md section 24.1). The error text ends with the one sentence `cli.EARLIER_RUN_KEPT`, which no other job writes. So the
    sentence is the record of that case. These tests keep the mark off when the files are not the case any more:
    - the job record is still `running`, or it began after `status.json` was written (a resume that failed before it wrote a status,
      a job that has just started): the status is the status of an older job;
    - the folder holds no earlier run now (`snapshot.old_run_files` is empty).
    `job` is the job record (`runjob.job_facts`), or None if the folder has none. Left out, the rule reads it.
    """
    d = Path(d)
    if not isinstance(status, dict) or (status.get("state"), status.get("iteration"), status.get("step")) != ("error", 0, "setup"):
        return False
    message = status.get("message")
    if not isinstance(message, str):
        return False
    from .cli import EARLIER_RUN_KEPT
    if not message.endswith(EARLIER_RUN_KEPT):
        return False
    if job is UNREAD:
        from .runjob import job_facts
        job = job_facts(d)
    if job is not None:
        began = _epoch(job.get("started_at"))
        try:
            written = (d / "status.json").stat().st_mtime
        except OSError:
            return False
        # `began - 1`: started_at has whole seconds (ISO text). The second is its resolution, not a tolerance to tune.
        if job.get("state") == "running" or (began is not None and written < began - 1):
            return False
    from .snapshot import old_run_files
    return bool(old_run_files(d))


def refused_or_false(d, status: dict | None, job: dict | None | object = UNREAD) -> bool:
    """`refused_over_earlier_run` for a reader that lists many folders: a fault in the rule gives False (the row of before), never a lost row."""
    try:
        return refused_over_earlier_run(d, status, job)
    except Exception:
        return False


# The keys of a row that tell the refused job: the pass of its status, its state and the text of its error. They stay at the top of the row.
JOB_KEYS = ("state", "iteration", "message")


def _names(keys) -> str:
    keys = [f"`{k}`" for k in keys]
    return keys[0] if len(keys) == 1 else ", ".join(keys[:-1]) + " and " + keys[-1]


def mark_row(row: dict, keys, unit: str = "row") -> dict:
    """A copy of `row` that says which keys are of the earlier run in the folder (P9-19). Use it for a row of a refused job only.

    The keys of `keys` that the row has move into `earlier_run` (with the key `note`), and they are `null` where they were: the refused job
    has no value for them, as a job that wrote no data has none. `earlier_run` stands after `state`, as in the reply of `run_status`. Every
    other key and the order of the keys stay. Every fact stays in the row: the earlier run is still in the folder.
    `unit` names the thing that `row` is in the note ("row", or "entry" for a side of `compare_runs`).
    """
    moved = [k for k in keys if k in row]
    of_job = [k for k in JOB_KEYS if k in row]
    note = ("The input check refused this job before it wrote a run file. The folder still holds an earlier run, and this job did not "
            "change it. ")
    if moved:
        note += f"The facts in this block are of that earlier run: {_names(moved)}. At the top of this {unit} they are null for the refused job. "
    if of_job:
        note += f"{_names(of_job)} {'is' if len(of_job) == 1 else 'are'} of the refused job. "
    note += "run_status gives the facts of both."
    block = {"note": note, **{k: row[k] for k in moved}}
    out: dict = {}
    for key, value in row.items():
        out[key] = None if key in moved else value
        if key == "state":
            out["earlier_run"] = block
    out.setdefault("earlier_run", block)
    return out
