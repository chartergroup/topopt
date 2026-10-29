"""A measured "up to" value next to the start estimates of `topopt check` (CONTRACTS.md section 5.1). Facts only.

Experiment E1 (a run-time study) compared the start estimates `derived.time_estimate_s` and `derived.memory_estimate_gib` with
the finished runs of one shared Linux host. For each solver this module holds the multiplier that covers 90 percent of those runs:
the 0.9 quantile of `actual / estimate` over the E1 runs of the solver. `bound_facts` returns, for one problem, the estimate, the
"up to" value (the estimate times the multiplier) or `None` with a note that says why, and the number of runs behind it.
The multipliers are from the E1 runs with 10,000 free dofs and more; a smaller problem gets no "up to" value, and its note gives
what E1 measured below 10,000 free dofs.

Nothing uses these values to warn or to refuse: every comparison with a budget or with the RAM stays on the point estimate.
"""
from __future__ import annotations

from typing import NamedTuple

SOURCE = ("experiment E1 (run-time study, 2026-10-23); runs made by the code revisions before main bb6e193; "
          "one shared Linux host; 10,000 free dofs and more")
RULE = ("up_to is the estimate times the multiplier. The multiplier is the 0.9 quantile of actual / estimate over the finished "
        "E1 runs with the same solver and 10,000 free dofs and more. Facts only: no warning, refusal or budget uses it")
LINUX_X86_64 = "linux x86_64"      # `solver_choice.platform_name()` on the platform of the E1 runs
E1_MIN_FREE_DOFS = 10_000          # the smallest problem of the E1 rows behind the multipliers (the measured range, not a setting)
RANGE_NOTE = f"The multiplier was measured at {E1_MIN_FREE_DOFS:,} free dofs and more."
GPU_MG_TIME_NOTE = "E1 ran gpu-mg before the deflation of case C (rigid modes and bonded patches) was added to gpu-mg."


class Row(NamedTuple):
    """The E1 runs of one solver: the multiplier (0.9 quantile of actual / estimate), the runs, their problems, and the share of those
    runs that is at or below the estimate times the multiplier."""
    multiplier: float
    n_runs: int
    n_problems: int
    covered_share: float


# Time: actual `metrics.json` `run_time_s` over `derived.time_estimate_s`. E1 runs with 10,000 free dofs and more: 596 runs, 114 problems.
TIME_ROWS = {
    "pardiso": Row(3.12, 542, 105, 0.90),
    "amg": Row(1.60, 22, 12, 0.86),
    "gpu-mg": Row(2.74, 32, 12, 0.91),
}
# Memory: the valid `job.json` `peak_rss_bytes` (the whole process) over `derived.memory_estimate_gib`. 363 runs, 88 problems.
# For gpu-mg `derived.memory_estimate_gib` is the Cholesky model value, not the gpu-mg memory model of `topopt check`.
MEMORY_ROWS = {
    "pardiso": Row(1.54, 332, 81, 0.90),
    "amg": Row(1.03, 8, 7, 0.88),
    "gpu-mg": Row(0.73, 23, 8, 0.78),
}
GPU_MG_MEMORY_NOTE = ("For gpu-mg the estimate of `derived.memory_estimate_gib` is the Cholesky model value, "
                      "not the gpu-mg memory model of the `memory estimate` line of `topopt check`.")


class Small(NamedTuple):
    """The E1 runs of one solver below 10,000 free dofs: the runs, their problems, and the median and the 0.9 quantile of
    actual / estimate. Facts for the note only: no "up to" value comes from them."""
    n_runs: int
    n_problems: int
    median: float
    q90: float


# The same quantities and E1 rows as above, below 10,000 free dofs (memory: the runs with a valid peak).
SMALL_TIME_ROWS = {
    "pardiso": Small(717, 16, 0.32, 0.71),
    "amg": Small(190, 9, 0.17, 0.74),
    "gpu-mg": Small(105, 5, 0.24, 0.45),
}
SMALL_MEMORY_ROWS = {
    "pardiso": Small(136, 11, 1.02, 1.34),
    "amg": Small(32, 3, 0.96, 1.04),
    "gpu-mg": Small(27, 5, 1.51, 1.86),
}


class StressRuns(NamedTuple):
    """The E1 runs that have stress rows (`n_stress_rows` above 0), for one solver. `median` and `share_above_1_6` are of
    actual / model total of the time estimate; the two multipliers are the 0.9 quantiles of actual / estimate."""
    n_runs: int
    n_problems: int
    median: float
    share_above_1_6: float
    time_multiplier: float
    memory_n_runs: int
    memory_multiplier: float


STRESS_RUNS = {
    "pardiso": StressRuns(53, 6, 2.23, 0.70, 4.01, 51, 2.83),
    "gpu-mg": StressRuns(1, 1, 8.24, 1.0, 8.24, 1, 0.26),
}


def stress_runs_text(sentence: bool = False) -> str:
    """The E1 fact on the runs with stress rows that `topopt check` gives next to the time range (a run with stress rows).

    It is about the time range of `cli.STRESS_RUN_TIME_FACTOR` (1.6): how the finished PARDISO runs with stress rows compare with the
    model total. `sentence` starts it with a capital letter.
    """
    s = STRESS_RUNS["pardiso"]
    text = (f"in experiment E1, {s.n_runs} finished PARDISO runs with stress rows in {s.n_problems} problems on one shared Linux host "
            f"had a median of {s.median:g} times the model total, and {round(100 * s.share_above_1_6)} percent took more than 1.6 times it")
    return text[0].upper() + text[1:] if sentence else text


def stress_sentence(solver: str, quantity: str) -> str:
    """What the E1 runs with stress rows and this solver showed, for the notes of a problem with stress rows."""
    s = STRESS_RUNS.get(solver)
    if s is None:
        return f"Experiment E1 has no run with stress rows and solver {solver}."
    if quantity == "time":
        return (f"This problem has stress rows. The {s.n_runs} E1 runs with stress rows and this solver ({s.n_problems} problems) took "
                f"a median of {s.median:g} times the estimate; {round(100 * s.share_above_1_6)} percent took more than 1.6 times it; "
                f"the 0.9 quantile is {s.time_multiplier:g} times.")
    return (f"This problem has stress rows. The {s.memory_n_runs} E1 runs with stress rows and this solver had a peak of at most "
            f"{s.memory_multiplier:g} times the estimate (0.9 quantile).")


def _fact(quantity: str, estimate, rows: dict, solver, platform: str, stress_rows: int, free_dofs: int) -> dict:
    row = rows.get(solver)
    what = {"time": "run time", "memory": "peak memory"}[quantity]
    took = {"time": "took", "memory": "had a peak of"}[quantity]
    digits, shown = {"time": (1, "{:.0f} s"), "memory": (2, "{:.1f} GiB")}[quantity]
    fact = {"estimate": estimate, "up_to": None, "multiplier": None, "n_runs": 0, "n_problems": 0, "covered_share": None,
            "basis": SOURCE, "note": "", "line": ""}
    if solver is None:
        fact["note"] = "No solver can start for this problem here, so no run takes this estimate. There is no up-to value."
        fact["line"] = "no up-to value: no solver can start here"
        return fact
    if platform != LINUX_X86_64:
        fact["note"] = (f"Experiment E1 has no finished run on this platform ({platform}): all its runs are on Linux x86_64. "
                        "There is no up-to value.")
        fact["line"] = f"no up-to value: experiment E1 has no run on this platform ({platform})"
        return fact
    if row is None:
        fact["note"] = f"Experiment E1 has no finished run with solver {solver}. There is no up-to value."
        fact["line"] = f"no up-to value: experiment E1 has no run with solver {solver}"
        return fact
    extra = (" " + (GPU_MG_MEMORY_NOTE if quantity == "memory" else GPU_MG_TIME_NOTE)) if solver == "gpu-mg" else ""
    if free_dofs < E1_MIN_FREE_DOFS:     # outside the measured range: no multiplier applies (review of E1-P3, m1)
        small = (SMALL_TIME_ROWS if quantity == "time" else SMALL_MEMORY_ROWS).get(solver)
        below = (f" Below {E1_MIN_FREE_DOFS:,} free dofs, E1 has {small.n_runs} runs with solver {solver} ({small.n_problems} problems): "
                 f"the median {what} was {small.median:g} times the estimate, and 90 percent {took} at most {small.q90:g} times it."
                 if small else f" E1 has no run with solver {solver} below {E1_MIN_FREE_DOFS:,} free dofs.")
        fact["note"] = f"This problem has {free_dofs:,} free dofs. {RANGE_NOTE} There is no up-to value.{below}{extra}"
        fact["line"] = f"no up-to value: experiment E1 measured problems with {E1_MIN_FREE_DOFS:,} free dofs and more"
        return fact
    fact.update(multiplier=row.multiplier, n_runs=row.n_runs, n_problems=row.n_problems, covered_share=row.covered_share)
    tail = " " + stress_sentence(solver, quantity) if stress_rows > 0 else ""
    if estimate is None:
        fact["note"] = "There is no point estimate here, so there is no up-to value." + extra
        fact["line"] = "no up-to value: there is no point estimate"
    elif row.multiplier < 1:
        fact["note"] = (f"The {what} of {row.n_runs} E1 runs with solver {solver} ({row.n_problems} problems) was at most "
                        f"{row.multiplier:g} times the estimate (0.9 quantile). That is below 1, so there is no up-to value above "
                        f"the estimate. {RANGE_NOTE}{extra}{tail}")
        of = "the Cholesky model value (`derived.memory_estimate_gib`)" if quantity == "memory" and solver == "gpu-mg" else "the estimate"
        fact["line"] = (f"no up-to value: the {what} of {row.n_runs} runs of experiment E1 was at most {row.multiplier:g} times "
                        f"{of} (0.9 quantile), below 1")
    else:
        fact["up_to"] = round(estimate * row.multiplier, digits)
        share = f"{round(100 * row.covered_share)} percent of {row.n_runs} E1 runs with solver {solver}"
        fact["note"] = f"{share} ({row.n_problems} problems) {took} at most {row.multiplier:g} times the estimate. {RANGE_NOTE}{extra}{tail}"
        fact["line"] = (f"up to about {shown.format(fact['up_to'])}: {round(100 * row.covered_share)} percent of {row.n_runs} runs of "
                        f"experiment E1 {took} at most {row.multiplier:g} times the estimate")
    return fact


def bound_facts(solver, platform: str, time_estimate_s, memory_estimate_gib, stress_rows: int = 0, *, free_dofs: int) -> dict:
    """The facts of the E1 "up to" values for one problem. Pure: the same input gives the same dict.

    `solver` is the solver that runs (`cholesky`, `pardiso`, `amg`, `gpu-mg`), or `None` when `topopt check` finds that no solver can
    start (the estimates then use the model of cholesky, and no run takes them). `platform` is `solver_choice.platform_name()`.
    The estimates are `derived.time_estimate_s` and `derived.memory_estimate_gib` (`None` when there is none). `stress_rows` is the
    number of load cases with a stress limit; it only adds the E1 facts of the runs with stress rows to the notes. `free_dofs` is the
    number of free dofs of the problem: below 10,000 (`E1_MIN_FREE_DOFS`, the smallest E1 row behind the multipliers) there is no
    `up_to`, and the note gives what E1 measured there.

    Returns `{"solver", "platform", "rule", "time", "memory"}`. `time` and `memory` are dicts with `estimate`, `up_to` (`None` when
    there is no value), `multiplier`, `n_runs`, `n_problems`, `covered_share` (the share of those E1 runs at or below the estimate
    times the multiplier), `basis` (the source text), `note` (the whole fact) and `line` (the short text that `topopt check` prints
    after the estimate). There is no `up_to` on a platform or for a solver that E1 did not run, none below 10,000 free dofs, and none
    when the measured multiplier is below 1; `n_runs` is 0 when no multiplier applies.
    """
    return {"solver": solver, "platform": platform, "rule": RULE,
            "time": _fact("time", time_estimate_s, TIME_ROWS, solver, platform, stress_rows, free_dofs),
            "memory": _fact("memory", memory_estimate_gib, MEMORY_ROWS, solver, platform, stress_rows, free_dofs)}


def derived_keys(solver, platform: str, time_estimate_s, memory_estimate_gib, stress_rows: int = 0, *, free_dofs: int) -> dict:
    """The keys that `cli.derived_numbers` adds to `derived`: `time_estimate_up_to_s`, `memory_estimate_up_to_gib`, `estimate_bounds`."""
    facts = bound_facts(solver, platform, time_estimate_s, memory_estimate_gib, stress_rows, free_dofs=free_dofs)
    return {"time_estimate_up_to_s": facts["time"]["up_to"], "memory_estimate_up_to_gib": facts["memory"]["up_to"],
            "estimate_bounds": facts}
