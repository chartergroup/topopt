"""A live time estimate for a run that has made two passes (E1-P2, CONTRACTS.md section 24.5).

`topopt check` gives a start estimate of the run time. It stays as it is. Once a run has made two passes, the run itself
is a better predictor (experiment E1, a run-time study, 2026-10-23: 586 runs, 111 problems, one shared Linux host;
runs made by the code revisions before main bb6e193): the time of the passes 1 and 2 gives the time of the later passes. This module turns that into "about N s left (up to M s)". The same
function feeds `run.log` (`optimize.run`), `run_status` (`topopt_mcp/server.py`) and the bar of the dashboard
(`dashboard_server.py` adds `time_left` to `/api/status`).

The formula, after the pass k of `history.jsonl` (pass 0 is the first pass)::

    left = mean(t_iter_s of the passes S+1 and S+2) * (N - k) + POST
    up_to = left * 1.33

- S is the first pass of THIS process: 0 for a new run. A resume or a continuation starts later, and `status.json` says where
  (`first_pass`). Pass S is not in the mean: E1 measured the window of the passes 0 and 1, and it was slower than the later
  passes (abs log of the total 0.211 and bias -0.19, against 0.136 and 0.002 for the passes 1 and 2).
- N is the last pass number, `run.json` `iterations`. `optimize.run` runs the passes `first` to N, so N + 1 passes in a new
  run, and after the pass k the passes k + 1 to N are left: N - k passes. E1 model C counts the same: at the end of pass 2 it
  has N - 2 passes left.
- POST is the median time of the post step by size band of free dofs. It is a constant in this file. The medians are of all work
  after the loop in the E1 rows, and they are pooled over the same rows as the error numbers below. A run with an opt-in post
  check, for example buckling, has a longer post step.
- The mean is taken over the passes S+1 and S+2 also at the later passes: this item keeps the rule that E1 measured. A running mean
  over the passes S+1 to k measured better at the later passes (abs log of the remaining time after pass 10: 0.113 against 0.156;
  after pass 20: 0.095 against 0.136; CONTRACTS.md section 24.5). It is not in this file.
- The estimate is for the planned passes. A stop rule or a request to stop can end the run earlier.

No function here reads a file, except `read_rows`, which reads `history.jsonl`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# --- constants: the only new numbers of E1-P2 ----------------------------------------
# Source: the E1 run-time study (fitted on measured runs); the column `post_s` of its data set, bands of `models.ModelC.band`.
# The rows are the 586 runs with 10,000 free dofs or more that have t_iter_s for the passes 1 and 2 (111 problems, one shared Linux host;
# experiment E1, run-time study, 2026-10-23; the runs were made by the code revisions before main bb6e193). `post_s` is `metrics.json` `run_time_s`
# minus the last `elapsed_s` of `history.jsonl`. The POST medians are pooled over the same rows as the error numbers (no hold-out).
# A run with an opt-in post check, for example buckling, has a longer post step than these medians.
BOUND_FACTOR = 1.33                 # E1 section 3.3: the q0.9 multiplier of model C. On the remaining time after pass 2, 88 % of 585 runs were inside it.
POST_S_BELOW_50K = 4.43             # median post step, below 50,000 free dofs: n = 435 runs, 74 problems
POST_S_50K_TO_200K = 7.51           # 50,000 to 199,999 free dofs: n = 145 runs, 32 problems
POST_S_200K_UP = 21.67              # 200,000 free dofs and more: n = 6 runs, 5 problems (a thin band: treat it as a rough number)
BAND_EDGES = (50_000, 200_000)      # free dofs; the bands of `models.ModelC.band`
MEASURED_FROM_DOFS = 10_000         # E1 measured runs of this size and more; below it the same constants are used and nothing was measured
PROBE_OFFSETS = (1, 2)              # the passes after the first pass of the process whose t_iter_s is averaged
FIRST_PASS_WITH_ESTIMATE = 2        # the estimate exists once the pass S+2 is done
MEASURED = ("experiment E1 (run-time study): after pass 2 the remaining time was within a factor 1.17 (median) and 1.64 (9 of 10 runs) "
            "of this value; the bound covered 88 % of 585 runs; 10,000 free dofs and more, one shared Linux host")


def post_band(n_free_dofs) -> tuple[str, float]:
    """(the name of the size band, the median post step in seconds) for a number of free dofs."""
    n = float(n_free_dofs)
    if n < BAND_EDGES[0]:
        return "below 50,000 free dofs", POST_S_BELOW_50K
    if n < BAND_EDGES[1]:
        return "50,000 to 199,999 free dofs", POST_S_50K_TO_200K
    return "200,000 free dofs and more", POST_S_200K_UP


def _number(value):
    """A finite number above 0 (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def _whole(value):
    """A whole number of 0 or more (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if math.isfinite(value) and value >= 0 and value == int(value) else None


def time_left(records, planned, n_free_dofs, start=0):
    """The estimate after the last pass in `records`: `(estimate, None)`, or `(None, reason)`. No file is read.

    `records` are the history records (dicts with `iteration` and `t_iter_s`) of the run. `planned` is the last pass number
    (`run.json` `iterations`). `n_free_dofs` sets the band of the post step. `start` is the first pass of this process.
    The estimate is a dict::

        {"left_s": ..., "up_to_s": ..., "after_pass": k, "basis": {...}}

    `basis` holds the numbers that give `left_s`: `pass_s` (the mean of the two probe passes), `probe_passes`, `passes_left`,
    `post_s`, `post_band`, `bound_factor`, `assumes` and `measured`. A run of less than 10,000 free dofs also has `not_measured`.
    The reason is a short sentence. A run that has not made the pass `start` + 2 has no estimate.
    """
    start = _whole(start)
    if start is None:
        return None, "the first pass of this process is not known"
    done: dict[int, dict] = {}
    for rec in records or ():
        it = _whole(rec.get("iteration")) if isinstance(rec, dict) else None
        if it is not None and it >= start:
            done[it] = rec                     # the last record of a pass wins
    last = max(done) if done else None
    need = start + FIRST_PASS_WITH_ESTIMATE
    n_last = _whole(planned)
    if n_last is not None and start > n_last:
        return None, f"no pass is left to run (this process starts at pass {start}, after the last pass {n_last})"
    if last is None or last < need:
        got = "no pass is done" if last is None else f"the last pass done is {last}"
        return None, f"no estimate before pass {need} is done ({got}" + (f"; this process started at pass {start}" if start else "") + ")"
    if n_last is None:
        return None, "run.json has no number of passes (iterations)"
    dofs = _number(n_free_dofs)
    if dofs is None:
        return None, "run.json has no number of free dofs (n_free_dofs)"
    times = []
    for off in PROBE_OFFSETS:
        t = _number(done.get(start + off, {}).get("t_iter_s"))
        if t is None:
            return None, f"the history has no usable t_iter_s for pass {start + off}"
        times.append(t)
    pass_s = sum(times) / len(times)
    passes_left = max(n_last - last, 0)
    band, post_s = post_band(dofs)
    left = pass_s * passes_left + post_s
    basis = {"pass_s": round(pass_s, 3), "probe_passes": [start + o for o in PROBE_OFFSETS], "passes_left": passes_left,
             "post_s": post_s, "post_band": band, "bound_factor": BOUND_FACTOR,
             "assumes": "all planned passes run; a stop rule or a request to stop can end the run earlier",
             "measured": MEASURED}
    if dofs < MEASURED_FROM_DOFS:
        basis["not_measured"] = (f"experiment E1 (run-time study) measured runs of {MEASURED_FROM_DOFS:,} free dofs and more; "
                                 f"this run has {int(dofs):,}")
    return {"left_s": round(left, 1), "up_to_s": round(left * BOUND_FACTOR, 1), "after_pass": last, "basis": basis}, None


def left_text(est: dict) -> str:
    """`about 93 s left (up to 124 s)`: the words of the summary of `run_status`."""
    return f"about {round(est['left_s'])} s left (up to {round(est['up_to_s'])} s)"


def log_line(est: dict) -> str:
    """The line of `run.log` for an estimate (one line, once for each process)."""
    b = est["basis"]
    a, c = b["probe_passes"]
    more = f". Experiment E1 (run-time study) did not measure runs of less than {MEASURED_FROM_DOFS:,} free dofs" if "not_measured" in b else ""
    return (f"fact: time: about {round(est['left_s'])} s left after pass {est['after_pass']} (up to {round(est['up_to_s'])} s; "
            f"the passes {a} and {c} took {b['pass_s']:.2f} s on average, passes left: {b['passes_left']}, "
            f"the post step takes about {b['post_s']:g} s; this holds if no stop rule or request to stop ends the run earlier{more})")


# --- readers: the files of a run folder ------------------------------------------------------------------------------------------------

def rows_of_bytes(raw) -> list[dict]:
    """The records of a `history.jsonl` text that parse (a cut last line and a bad line are skipped). `raw` is bytes or None."""
    rows = []
    for line in (raw or b"").decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def read_rows(run_dir) -> list[dict]:
    """The records of `history.jsonl` in a run folder. An unreadable or missing file gives an empty list."""
    try:
        return rows_of_bytes((Path(run_dir) / "history.jsonl").read_bytes())
    except OSError:
        return []


# `runjob.job_facts` states that say the run process has ended while `status.json` may still say `running`: a process that was
# killed leaves no marker. A reader that finds one of them gives no estimate. `no_record` is the state of `run_status` for a process
# of the MCP server that ended and wrote no job record.
ENDED_JOB_STATES = ("failed", "no_record", "stale_lock", "unfinished")


def job_ended(job) -> bool:
    """True if the job facts (`runjob.job_facts`, or the `job` of `run_status`) say that the run process has ended."""
    return isinstance(job, dict) and job.get("state") in ENDED_JOB_STATES


def run_time_left(get_rows, run, status):
    """The estimate of a run from its files: `(estimate, None)` or `(None, reason)`.

    `get_rows` is a function with no argument that gives the history records (`read_rows(run_dir)`): a run that has no
    estimate reads nothing. `run` is `run.json` and `status` is `status.json` (dicts, or None). A run that does not loop now
    (state other than `running`) has none. `status.json` key `first_pass` is the first pass of this process; a missing key means
    0, and null means that a resume has not read its checkpoint yet.
    """
    status = status if isinstance(status, dict) else {}
    state = status.get("state")
    if state != "running":
        return None, f"the run is not in its passes (state: {state or 'unknown'})"
    first = status["first_pass"] if "first_pass" in status else 0
    if first is None:
        return None, "this process has not read its checkpoint yet"
    run = run if isinstance(run, dict) else {}
    return time_left(get_rows(), run.get("iterations"), run.get("n_free_dofs"), first)
