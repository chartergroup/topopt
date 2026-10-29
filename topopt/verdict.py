"""Stop rules and the convergence verdict (CONTRACTS.md section 24.2).

The functions work on history records (the dicts of `history.jsonl`) and on `topopt.config.StopRules`.
They need no numpy. Only `write_verdict` and `read_verdict` touch a file.

The verdict keeps two parts apart: `feasible` (every constraint row is met) and `converged` (the objective
has settled: the objective change, the grayness and the last beta stage). A run can be feasible and not
converged, and it can be converged and not feasible. Neither part is called "infeasible" or "not converged"
because of the other (KB pitfall `verdict-mislabels-feasible-unconverged`). `state: converged` and the stop
rule `converged` need both: converged and feasible.

The rule that ends a run and the tests that the verdict reports are the same function (`measures`).
So a run that a rule stopped has a verdict with the same numbers. The text gives each test with its value,
its threshold and "met" or "not met". It does not judge the run: the default thresholds are not tuned for a part.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from .formulation import NOT_MET, no_result, no_result_text
from . import stage_facts
from .runinfo import write_atomic

VERDICT_FILE = "verdict.json"
FORMAT = 1
STATES = ("converged", "iteration_limit", "stopped_by_rule", "stopped_by_request", "unfinished")
FEASIBLE_TOL = NOT_MET   # the same limit as the "constraint not met" lines of the run log


def _objective(row: dict) -> float:
    """The objective value of a history record (the weighted compliance if the record has no objective)."""
    obj = row.get("objective")
    if isinstance(obj, dict) and obj.get("value") is not None:
        return float(obj["value"])
    return float(row["compliance"])


def _rel_change(new: float, old: float) -> float:
    scale = abs(new)
    if scale == 0.0:
        return 0.0 if old == 0.0 else 1e300   # a finite number: the verdict is JSON
    return abs(new - old) / scale


def _fmt(v) -> str:
    return "none" if v is None else f"{v:.3g}"


def reaches_last_beta_stage(last_pass: int, schedule) -> bool:
    """The test `last_beta_stage`: the last pass is at or after the start of the last stage of the beta schedule.

    The verdict (`measures`) and the fact before the run (`beta_stage_fact`) use this one comparison.
    """
    return int(last_pass) >= int(schedule[-1][0])


def measures(rows: list[dict], rules, schedule) -> dict:
    """The convergence measures at the last row of `rows`, and the thresholds. `rows` are history records in pass order.

    Three measures decide `converged_met` (the objective has settled at this pass), and with the fourth, `all_met`:
      objective_change  the largest relative change of the objective between two passes, over the last `window`
                        passes. It needs `window + 1` passes in a row at the same beta (a beta step moves the
                        objective, so it does not count). None while there are not enough.
      grayness          `grayness` of the last pass, at most `grayness_max`.
      last_beta_stage   the last pass is at or after the start of the last stage of the beta schedule. A design
                        that has not seen the last beta is not a result.
      max_g             (feasibility, not convergence) the largest linear g of the constraint rows, at most
                        `FEASIBLE_TOL`. `row` names it. A row with no result at the last pass (`no_result`: the buckling
                        row after a failed evaluation) is not judged: `max_g` names it, and has the extra keys `judged`
                        (false), `status`, `held_from_pass` and `text`; `met` is false and stays a bool. A judged `max_g`
                        has no `judged` key (the verdict of a run without such a row keeps its bytes).
    `last_step_change` (the largest change of one design variable in the last step) is information: no threshold.
    """
    last = rows[-1]
    it, beta = int(last["iteration"]), float(last["beta"])
    need = rules.window + 1
    at_beta, nxt = 0, None
    for r in reversed(rows):
        if float(r["beta"]) != beta or (nxt is not None and int(r["iteration"]) != nxt - 1):
            break
        at_beta, nxt = at_beta + 1, int(r["iteration"])
    change = None
    if at_beta >= need:
        values = [_objective(r) for r in rows[-need:]]
        change = max(_rel_change(b, a) for a, b in zip(values, values[1:]))
        if not math.isfinite(change):   # a value that is nan: not judged
            change = None
    start, last_beta = schedule[-1]
    reached = reaches_last_beta_stage(it, schedule)
    gray = None if last.get("grayness") is None else float(last["grayness"])
    cons = last.get("constraints") or []
    unjudged = next((c for c in cons if no_result(c)), None)
    worst = unjudged or (max(cons, key=lambda c: c["g"]) if cons else None)
    out = {
        "objective_change": {"value": change, "threshold": rules.objective_change, "window": rules.window,
                             "met": change is not None and change <= rules.objective_change,
                             "passes_at_beta": at_beta, "passes_needed": need, "beta": beta},
        "grayness": {"value": gray, "threshold": rules.grayness_max,
                     "met": gray is not None and gray <= rules.grayness_max},
        "max_g": {"value": None if worst is None else float(worst["g"]), "threshold": FEASIBLE_TOL,
                  "met": worst is None or (unjudged is None and float(worst["g"]) <= FEASIBLE_TOL),
                  "row": None if worst is None else worst["name"]},
        "last_beta_stage": {"value": reached, "met": reached, "beta": beta, "last_beta": float(last_beta), "from_iteration": int(start)},
    }
    if unjudged is not None:
        out["max_g"].update(judged=False, status=unjudged.get("status"), held_from_pass=unjudged.get("last_eval"),
                            text="not judged: " + no_result_text(unjudged))
    if len(rows) > 1 and rows[-2].get("change") is not None:
        out["last_step_change"] = {"value": float(rows[-2]["change"]), "threshold": None, "met": None}
    return out


def converged_met(m: dict) -> bool:
    """The objective has settled: the objective change, the grayness and the last beta stage are met. Feasibility is not in it."""
    return all(m[k]["met"] for k in ("objective_change", "grayness", "last_beta_stage"))


def all_met(m: dict) -> bool:
    """Converged and feasible: the stop rule `converged` fires, and `state: converged` is set, at this pass."""
    return converged_met(m) and m["max_g"]["met"]


SETTLED = "settled"   # the stop rule of a fine pass (CONTRACTS.md section 24.6)


def settled_met(m: dict) -> bool:
    """The loss has settled and the constraints are met: the objective change test and the largest g test. Not the grayness or the last beta stage."""
    return m["objective_change"]["met"] and m["max_g"]["met"]


# What a run that ends before the last beta stage cost on the hook example (origin/main 4f6d39d, CPU, one run each, one part). Runs IT20 and BSc are
# of the P7 series E1, run IT20o is of P7-4: the schedule that the fact offers for this case.
# Stiffness: the optimizer value of `hang` / `pull_off` against the base run (40 passes, own schedule, grayness 0.025).
HOOK_PRICE = ("Measured on examples/hook/hook_short.yaml (2 mm voxels, one part, one run each; stiffness hang / pull_off against the 40-pass run "
              "with its own schedule, grayness 0.025): 20 passes, schedule as it was (ends at beta 2): x0.791 / x0.834, grayness 0.142 (IT20); "
              "20 passes, schedule scaled: x0.989 / x0.970, grayness 0.026 (IT20o); 40 passes, default schedule (ends at beta 2): "
              "x0.801 / x0.869, grayness 0.138 (BSc)")


def scaled_beta_schedule(iterations: int, schedule) -> tuple[int, list[list]]:
    """The schedule scaled to `iterations` passes: `(span, [[start, beta], ...])`. It has more than one stage.

    Each start pass is multiplied by `iterations / span` and rounded half up to a pass. The span is the run length that the schedule fits: the last
    start plus the length of the stage before it (the last stage as long as that stage). The default schedule gives 65 + 15 = 80.
    The last start is rounded down: `last * iterations / span` is below `iterations`, so the last start is at most `iterations - 1`, and the design
    changes at least once at the last beta (the last pass only measures). An earlier start is not above the last one. Stages that fall on one pass
    become one stage: the later one stays.
    """
    starts = [int(s) for s, _ in schedule]
    span = 2 * starts[-1] - starts[-2]
    last = starts[-1] * iterations // span
    scaled = {min((2 * s * iterations + span) // (2 * span), last): b for s, (_, b) in zip(starts[:-1], schedule[:-1])}   # half up, in whole numbers
    scaled[last] = schedule[-1][1]
    return span, [[s, b] for s, b in sorted(scaled.items())]


def _beta_text(b) -> str:
    """A beta as YAML reads it back to the same number: the short form (`8`, `2.5`) when it gives that number, else `repr`.
    PyYAML reads `1e-05` as a string, so a mantissa gets a `.` (`1.0e-05`)."""
    b = float(b)
    text = f"{b:g}"
    if "e" not in text and float(text) == b:
        return text
    text = repr(b)
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = (mantissa if "." in mantissa else mantissa + ".0") + "e" + exponent
    return text


def _schedule_text(pairs) -> str:
    return "[" + ", ".join(f"[{int(s)}, {_beta_text(b)}]" for s, b in pairs) + "]"


def beta_stage_fact(iterations: int, schedule, continued_from: dict | None = None) -> list[str]:
    """A fact before the run when the planned passes end before the last beta stage (F4-18). `[]` when the run reaches it.

    The run makes the passes 0 to `iterations`, and a stop rule can only end it earlier. So the last pass is `iterations` at most, and the
    test is the one of the verdict (`reaches_last_beta_stage`). The fact gives the options first and then the price that was measured on the
    hook. Nothing blocks the run.

    `continued_from` is the run.json record of a continuation (`--continue-from`), and of the resume of one (CONTRACTS.md section 24.3). The
    pass count of that run is `iteration` + `more_iterations` and not `settings.iterations`, so the fact names the count and the flag, and the
    second option is a new run (a changed `beta_schedule` is a changed input of a continuation). The numbers and the price are the same.
    """
    if reaches_last_beta_stage(iterations, schedule):
        return []
    last, last_beta = int(schedule[-1][0]), _beta_text(schedule[-1][1])
    start, beta = [(int(s), b) for s, b in schedule if int(s) <= iterations][-1]
    if continued_from:
        k, n = int(continued_from["iteration"]), int(continued_from["more_iterations"])
        return [
            f"fact: the pass count of this run is {iterations} ({k} + {n} more_iterations); settings.iterations of the YAML is not used. "
            f"The last stage of settings.beta_schedule {_schedule_text(schedule)} (beta {last_beta}) starts at pass {last}. "
            f"The run ends at pass {iterations} at beta {_beta_text(beta)} (the stage from pass {start}) and does not reach "
            f"beta {last_beta}: the verdict will say that the beta schedule has not reached its last stage, and a design that has not seen the "
            f"last beta can stay grey. "
            f"options: more_iterations {last - k} or more (the flag --more-iterations; pass {last} only measures; each pass above it is one "
            f"change of the design at beta {last_beta}); or a new run, not a continuation, with the schedule scaled to its passes "
            f"(`topopt check` of the YAML with that settings.iterations gives the beta_schedule). {HOOK_PRICE}."]
    span, scaled = scaled_beta_schedule(iterations, schedule)
    stage_before = last - int(schedule[-2][0])
    same_pass = ""
    if len(scaled) < len(schedule):
        first = "" if float(scaled[0][1]) == float(schedule[0][1]) else f" (the run then starts at beta {_beta_text(scaled[0][1])})"
        same_pass = f" Stages that fall on one pass are one stage: the later one stays{first}."
    return [
        f"fact: settings.iterations is {iterations}; the last stage of settings.beta_schedule {_schedule_text(schedule)} (beta {last_beta}) "
        f"starts at pass {last}. The run ends at pass {iterations} at beta {_beta_text(beta)} (the stage from pass {start}) and does not reach "
        f"beta {last_beta}: the verdict will say that the beta schedule has not reached its last stage, and a design that has not seen the "
        f"last beta can stay grey. "
        f"options: settings.iterations {last} or more (pass {last} only measures; each pass above it is one change of the design at beta "
        f"{last_beta}); or beta_schedule: {_schedule_text(scaled)} (each start x {iterations}/{span}, rounded to a pass, the last one down; "
        f"{span} = the last start {last} + the {stage_before} passes of the stage before it).{same_pass} {HOOK_PRICE}."]


def _verdict_word(met) -> str:
    return "met" if met else "not met"


def _tests_sentence(m: dict) -> str:
    """Each test with its value, its threshold and met or not met. The text of the verdict and of a stop rule."""
    oc, g, mg, ls = m["objective_change"], m["grayness"], m["max_g"], m["last_beta_stage"]
    if oc["value"] is None:
        obj = (f"objective change: not judged, it needs {oc['passes_needed']} passes in a row at one beta "
               f"(the run has {oc['passes_at_beta']} at beta {oc['beta']:g}) (threshold {_fmt(oc['threshold'])}: not met)")
    else:
        obj = (f"objective change {_fmt(oc['value'])} (relative) in the last {oc['window']} passes "
               f"(threshold {_fmt(oc['threshold'])}: {_verdict_word(oc['met'])})")
    gray = ("grayness: not in the history (not met)" if g["value"] is None else
            f"grayness {_fmt(g['value'])} (threshold {_fmt(g['threshold'])}: {_verdict_word(g['met'])})")
    row = "" if mg["row"] is None else f"row {mg['row']}, "
    gmax = ("largest g: no constraint row (met)" if mg["value"] is None else
            f"largest g: row {mg['row']} {mg['text']} (threshold {_fmt(mg['threshold'])}: not met)" if not mg.get("judged", True) else
            f"largest g {mg['value']:+.3g} ({row}threshold {_fmt(mg['threshold'])}: {_verdict_word(mg['met'])})")
    stage = (f"the beta schedule has reached its last stage (beta {ls['last_beta']:g} from pass {ls['from_iteration']})" if ls["met"] else
             f"the beta schedule has not reached its last stage (beta {ls['last_beta']:g} starts at pass {ls['from_iteration']}, "
             f"the run ends at beta {ls['beta']:g})")
    return "; ".join((obj, gray, gmax, stage))


def _settled_not_feasible(m: dict) -> str:
    mg = m["max_g"]
    if not mg.get("judged", True):
        return f"The objective has settled; row {mg['row']} is not judged ({mg['text'].removeprefix('not judged: ')})."
    return f"The objective has settled; row {mg['row']} is not met (g {mg['value']:+.3g}, threshold {_fmt(mg['threshold'])})."


# --- the stop rules ------------------------------------------------------------------------

def stop_decision(rows: list[dict], rules, schedule, elapsed_s: float, settled: bool = False) -> dict | None:
    """The stop rule that fires at the last row of `rows`, as a dict, or None. None too if the rules are off.

    `rows` are the history records so far, with a record for the pass that is being judged as the last row.
    The rules, in this order:
      settled       (only with `settled`, the fine pass of CONTRACTS.md section 24.6) the objective change test and the largest g test of
                    `measures` are met. The grayness test and the last beta stage test are not used: the rule is for a run that starts
                    from a design that has passed the beta schedule already. The tolerance is `rules.objective_change`, the window `rules.window`.
      converged     all four measures of `measures` are met
      time_budget   `elapsed_s` is at or above `rules.time_budget_s`
    The dict has `rule`, `iteration`, `text` and the numbers of the rule (`measures`, or `elapsed_s` and `time_budget_s`).
    """
    if not rules.enabled or not rows:
        return None
    it = int(rows[-1]["iteration"])
    m = measures(rows, rules, schedule)
    if settled and settled_met(m):
        oc, mg = m["objective_change"], m["max_g"]
        g = "no constraint row" if mg["value"] is None else f"largest g {mg['value']:+.3g}, row {mg['row']}"
        return {"rule": SETTLED, "iteration": it, "measures": m,
                "text": f"stop rule '{SETTLED}' at pass {it}: the loss has settled. The tests that decide: objective change {_fmt(oc['value'])} (relative) "
                        f"in the last {oc['window']} passes at beta {oc['beta']:g}, at or below the tolerance {_fmt(oc['threshold'])}; the constraints "
                        f"are met ({g}, threshold {_fmt(mg['threshold'])}). The grayness test and the last beta stage test are not used by this rule. "
                        "Tests at this pass: " + _tests_sentence(m) + "."}
    if all_met(m):
        return {"rule": "converged", "iteration": it, "measures": m,
                "text": f"stop rule 'converged' at pass {it}. Tests: " + _tests_sentence(m) + "."}
    if rules.time_budget_s is not None and elapsed_s >= rules.time_budget_s:
        return {"rule": "time_budget", "iteration": it, "elapsed_s": round(float(elapsed_s), 2),
                "time_budget_s": rules.time_budget_s, "measures": m,
                "text": f"stop rule 'time_budget' at pass {it}: the run time {elapsed_s:.1f} s is at or above the budget "
                        f"of {rules.time_budget_s:g} s. Convergence tests at this pass: " + _tests_sentence(m) + "."}
    return None


# --- the verdict ---------------------------------------------------------------------------

def schedule_from_rows(rows: list[dict]):
    """The beta schedule that the rows show: `[(first pass of each stretch at one beta, beta), ...]`.

    For a folder that has no `verdict.json`. The real schedule is in the YAML; the rows cannot show a stage that the run never reached.
    """
    out = []
    for r in rows:
        if not out or out[-1][1] != float(r["beta"]):
            out.append((int(r["iteration"]), float(r["beta"])))
    return out or [(0, 1.0)]


def build_verdict(rows: list[dict], rules, schedule, n_iter: int, stop: dict | None = None,
                  resume_points=(), notes=(), request: dict | None = None) -> dict:
    """The verdict of a run (CONTRACTS.md section 24.2). `rows` are all history records of the run, in pass order.

    state  stopped_by_rule (`stop` is set) | stopped_by_request (`request` is set) |
           converged (the last pass is `n_iter`, converged and feasible) |
           iteration_limit (the last pass is `n_iter` and `converged` and `feasible` are not both true) |
           unfinished (the history ends before `n_iter` and no rule stopped the run)
    `request` (P4b-4) is the request to stop that ended the run at the last row: `requested_at` and `by` (of `runjob.read_stop`). The verdict
    then has the key `stop_request` (`requested_at`, `by`, `iteration`). Only this state has the key: every other verdict keeps its bytes.
    The tests in the text are the tests of the last pass, with the same numbers as a stop rule would use; they do not judge the run.
    `converged` is the objective test (`converged_met`) and `feasible` is the constraint test. They are two facts.
    `facts` is the `stage_end` of the last pass (`stage_facts`: the volume used, the load path; a mass run also the material and, when its last pass is grey,
    the grey design with the cuts 0.4, 0.5 and 0.6) with its `iteration`, or None (a run that is not finished, or a history with no `stage_end`). The text ends
    with a " Facts: ..." sentence for it. No test uses these facts. The body count of `final.stl` is not in it: the post step makes it, after this file is written.
    """
    if not rows:
        return {"format": FORMAT, "state": "unfinished", "text": "the history has no pass", "final_iteration": None,
                "iterations": n_iter, "converged": False, "feasible": None, "measures": None,
                "stop_rules": dataclasses.asdict(rules), "stop": stop, "resume_points": list(resume_points),
                "notes": list(notes), "facts": None}
    final = int(rows[-1]["iteration"])
    m = measures(rows, rules, schedule)
    converged, feasible = converged_met(m), m["max_g"]["met"]
    if stop is not None:
        state, text = "stopped_by_rule", stop["text"] + f" (the iteration limit is {n_iter})"
    elif request is not None:
        state, text = "stopped_by_request", f"stopped by a request at pass {final} of {n_iter}. Tests: " + _tests_sentence(m) + "."
    elif final >= n_iter:
        state = "converged" if converged and feasible else "iteration_limit"
        head = (f"converged at the last pass ({final})." if state == "converged" else
                f"iteration limit reached at pass {final}." + (" " + _settled_not_feasible(m) if converged else ""))
        text = head + " Tests: " + _tests_sentence(m) + "."
    else:
        state = "unfinished"
        text = f"the history ends at pass {final} of {n_iter} and no stop rule fired"
    facts = stage_facts.verdict_facts(rows[-1]) if state != "unfinished" else None   # the facts of the last pass (P3a): not a test
    out = {"format": FORMAT, "state": state, "text": text + stage_facts.verdict_sentence(facts), "final_iteration": final,
           "iterations": n_iter, "converged": converged, "feasible": feasible, "measures": m,
           "stop_rules": dataclasses.asdict(rules), "stop": stop, "resume_points": list(resume_points),
           "notes": list(notes), "facts": facts}
    if state == "stopped_by_request":
        out["stop_request"] = {"requested_at": request.get("requested_at"), "by": request.get("by"), "iteration": final}
    return out


def write_verdict(run_dir, verdict: dict) -> None:
    write_atomic(Path(run_dir) / VERDICT_FILE, json.dumps(verdict, indent=1).encode())


def read_verdict(run_dir) -> dict | None:
    """The `verdict.json` of a run folder, or None if it is missing or does not parse."""
    try:
        data = json.loads((Path(run_dir) / VERDICT_FILE).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("format") == FORMAT else None


def verdict_for_post(run_dir, rows: list[dict], n_iter: int) -> dict:
    """The verdict for `metrics.json`. The one that the run wrote, if it matches the history. Else one from the history.

    The second case is a folder with no `verdict.json` (an old run, a fake run, a changed history). The default thresholds
    and the beta schedule of the rows are used, and `source` says so.
    """
    from .config import StopRules
    v = read_verdict(run_dir)
    if v is not None and rows and v.get("final_iteration") == int(rows[-1]["iteration"]):
        return v
    out = build_verdict(rows, StopRules(), schedule_from_rows(rows), n_iter,
                        notes=["verdict computed from the history with the default thresholds: "
                               + ("verdict.json does not match the history" if v is not None else "the run folder has no verdict.json")])
    out["source"] = "history"
    return out
