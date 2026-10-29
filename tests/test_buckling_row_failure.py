"""The buckling row with no result at the end of a run (A6b fix round 1, M1) and a failure that repeats (m1).

Short real runs of the 2 x 3 x 30 voxel column (`column_yaml`) with a faked `buckling_modes`: it gives a real result or a
failure, by the number of the call. `min_factor` is 1e-3, so the row is far from its limit and the skip rule is the only rule
that decides (passes 0 and 1 evaluate, a later pass evaluates only at a stage start, at the last pass, or every 5 passes).
Nothing here loads OpenCL.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import needs_direct_solver
import test_checkpoint as tc
from test_buckling_row import FakePoint, column_yaml, fake_modes, loop_model, point_at, state
from test_buckling_row_wiring import ONE_BETA, with_buckling, with_settings
from topopt import buckling as bk
from topopt import buckling_row as br
from topopt import cli
from topopt.cli import main
from topopt.config import StopRules
from topopt.formulation import Formulation, no_result, no_result_text
from topopt import verdict as V

NAME = "buckling:axial"


def flaky(monkeypatch, fails, status=bk.NOT_CONVERGED):
    """`buckling_modes` fails (status `status`, no result) when `fails(n)` is true for the call number n (from 0), else it is real.
    Returns the list of the call numbers."""
    real, calls = bk.buckling_modes, []

    def fake(Kg, K, fac, k=6, **kw):
        n = len(calls)
        calls.append(n)
        return fake_modes(status)(Kg, K, fac, k=k, **kw) if fails(n) else real(Kg, K, fac, k=k, **kw)

    monkeypatch.setattr(bk, "buckling_modes", fake)
    return calls


def run(tmp_path, iterations=4, stop_rules=None):
    p = with_buckling(column_yaml(tmp_path, iterations=iterations, extra_settings=ONE_BETA), min_factor=1e-3)
    if stop_rules is not None:
        with_settings(p, stop_rules=stop_rules)
    out = tmp_path / "r"
    code = main(["run", str(p), "--out", str(out), "--quiet"])
    rows = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines()]
    entries = [next(c for c in r["constraints"] if c["name"] == NAME) for r in rows]
    return code, entries, json.loads((out / "verdict.json").read_text()), (out / "run.log").read_text()


@needs_direct_solver
def test_no_result_at_the_last_pass_is_not_judged_and_not_feasible(tmp_path, monkeypatch):
    """Passes 0 and 1 give a result, passes 2 and 3 are skipped, the last pass (4) fails: the row has no result there. The verdict does not judge
    the row (`feasible` false, a bool; `max_g` names the row, `judged` and `met` false, the status and the pass of the value held),
    and the log has the `unmet` line and the fact."""
    calls = flaky(monkeypatch, lambda n: n == 2)
    code, entries, verdict, log = run(tmp_path)
    assert code == 0 and len(calls) == 3
    assert [e["evaluated"] for e in entries] == [True, True, False, False, False]
    assert [e["status"] for e in entries] == ["ok", "ok", "skipped", "skipped", bk.NOT_CONVERGED]
    assert verdict["feasible"] is False and verdict["state"] == "iteration_limit"
    mg = verdict["measures"]["max_g"]
    assert mg["row"] == NAME and mg["judged"] is False and mg["met"] is False and mg["status"] == bk.NOT_CONVERGED
    assert mg["held_from_pass"] == 1 and mg["value"] == pytest.approx(entries[-1]["g"])
    assert mg["text"] == "not judged: no result at the last pass (status not_converged; the value held is from pass 1)"
    assert f"row {NAME} {mg['text']} (threshold 0.005: not met)" in verdict["text"]
    assert f"constraint not judged: {NAME}  no result at the last pass (status not_converged; the value held is from pass 1)" in log
    assert f"fact: {NAME}: no result at the last pass (status not_converged; the value held is from pass 1)" in log
    assert "`feasible` is false" in log
    metrics = json.loads((tmp_path / "r" / "result" / "metrics.json").read_text())      # n2: the list `constraints` carries the keys of the row
    assert next(c for c in metrics["constraints"] if c["name"] == NAME)["status"] == bk.NOT_CONVERGED
    assert set(br.INFO_KEYS) <= set(next(c for c in metrics["constraints"] if c["name"] == NAME))


@needs_direct_solver
def test_no_result_at_every_pass_names_the_start_value(tmp_path, monkeypatch):
    """The eigen solve fails at every evaluation: no pass gave a result, the row holds its start value (g -1) and is not judged."""
    flaky(monkeypatch, lambda n: True)
    code, entries, verdict, log = run(tmp_path)
    assert code == 0 and not any(e["evaluated"] for e in entries) and all(e["last_eval"] is None for e in entries)
    assert all(e["g"] == pytest.approx(-1.0) for e in entries)
    mg = verdict["measures"]["max_g"]
    assert verdict["feasible"] is False and mg["row"] == NAME and mg["judged"] is False and mg["met"] is False
    assert mg["held_from_pass"] is None
    assert mg["text"] == "not judged: no result at the last pass (status not_converged; no pass gave a result, the value held is the start value)"
    assert f"fact: {NAME}: no result at the last pass" in log and f"constraint not judged: {NAME}" in log
    assert not [x for x in log.splitlines() if x.startswith("constraint not met")]     # g is -1: the row is not "met", it is not judged


@needs_direct_solver
def test_a_skipped_last_pass_keeps_its_value_and_feasible_stays_true(tmp_path, monkeypatch):
    """A stop rule ends the run at pass 2, where the skip rule holds the value of pass 1. The held value is a result: the row is judged,
    `feasible` is true, the rule `converged` fires, and the fact about the skipped pass stays."""
    flaky(monkeypatch, lambda n: False)
    rules = {"objective_change": 1e9, "window": 2, "grayness_max": 1.0}
    code, entries, verdict, log = run(tmp_path, iterations=8, stop_rules=rules)
    assert [e["evaluated"] for e in entries] == [True, True, False]
    assert entries[-1]["status"] == "skipped" and entries[-1]["last_eval"] == 1
    assert verdict["state"] == "stopped_by_rule" and verdict["stop"]["rule"] == "converged"
    mg = verdict["measures"]["max_g"]
    assert verdict["feasible"] is True and "judged" not in mg and mg["met"] is True and "text" not in mg
    assert f"fact: {NAME}: the last pass was not evaluated (a stop rule ended the run" in log
    assert "constraint not judged" not in log


# --- the rule on plain dicts (no run) -----------------------------------------


def entry(evaluated, status, last_eval, g=-0.5):
    return {"name": NAME, "g": g, "evaluated": evaluated, "status": status, "last_eval": last_eval}


def test_no_result_table():
    assert no_result(entry(False, bk.NOT_CONVERGED, 4)) and no_result(entry(False, bk.TIME_BUDGET, None))
    assert no_result(entry(False, "factor_mismatch", 2)) and no_result(entry(False, "skipped", None))   # nothing is held
    assert not no_result(entry(True, "ok", 5)) and not no_result(entry(True, bk.NO_POSITIVE, None))
    assert not no_result(entry(False, "skipped", 3))                          # the value is held: it is a result
    assert not no_result({"name": "volume", "g": 0.1})                        # a row that has no `evaluated` key
    assert no_result_text(entry(False, bk.TIME_BUDGET, 3)) == \
        "no result at the last pass (status time_budget; the value held is from pass 3)"


def history(last):
    """Passes 0 to 25 at the beta schedule of the verdict tests, with the volume row and the buckling row (`last` at the last pass)."""
    schedule = [(0, 1.0)]
    rows = [{"iteration": i, "beta": 1.0, "grayness": 0.01, "objective": {"value": 1.0}, "change": 0.0,
             "constraints": [{"name": "volume", "g": -0.1}, entry(True, "ok", i)]} for i in range(25)]
    rows.append({"iteration": 25, "beta": 1.0, "grayness": 0.01, "objective": {"value": 1.0}, "change": 0.0,
                 "constraints": [{"name": "volume", "g": -0.1}, last]})
    return rows, schedule


def test_the_verdict_measures_do_not_judge_a_row_with_no_result():
    rules = StopRules(enabled=True)
    rows, schedule = history(entry(False, bk.NOT_CONVERGED, 24, g=-0.9))
    m = V.measures(rows, rules, schedule)
    assert V.converged_met(m) and not V.all_met(m)                           # the objective settled; the row is not judged
    assert m["max_g"]["judged"] is False and m["max_g"]["met"] is False and m["max_g"]["row"] == NAME
    assert V.stop_decision(rows, rules, schedule, 0.0) is None               # the rule `converged` needs a judged row
    v = V.build_verdict(rows, rules, schedule, 25)
    assert v["feasible"] is False and v["converged"] is True and v["state"] == "iteration_limit"
    assert v["feasible"] is False and "The objective has settled; row buckling:axial is not judged (no result at the last pass" in v["text"]
    held = V.measures(history(entry(False, "skipped", 24, g=-0.9))[0], rules, schedule)
    assert "judged" not in held["max_g"] and held["max_g"]["met"] is True and V.all_met(held)
    plain = V.measures(history(entry(True, "ok", 25, g=0.1))[0], rules, schedule)       # judged, and not met by its g
    assert "judged" not in plain["max_g"] and plain["max_g"]["met"] is False and plain["max_g"]["row"] == NAME


def test_a_stage_that_ends_with_no_result_is_not_called_feasible():
    form = SimpleNamespace(stage_facts=True, constraints=[])
    prev = {"beta": 1.0, "constraints": [{"name": "volume", "g": -0.1}, entry(False, bk.NOT_CONVERGED, 3, g=-0.9)]}
    note = Formulation.stage_note(form, prev, 2.0, 10)
    assert note["feasible"] is False and note["row"] == NAME and "status not_converged" in note["no_result"]
    assert "NOT feasible: buckling:axial not judged, no result at the last pass" in Formulation.stage_lines(note)[0]
    ok = {"beta": 1.0, "constraints": [{"name": "volume", "g": -0.1}, entry(False, "skipped", 3, g=-0.9)]}
    note = Formulation.stage_note(form, ok, 2.0, 10)
    assert note["feasible"] is True and "no_result" not in note


# --- m1: a failure that repeats -----------------------------------------------


def attempts(entries):
    """The passes where the eigen solve ran (the fake gives `solves` 7 for a failure)."""
    return [i for i, e in enumerate(entries) if e["solves"]]


@needs_direct_solver
def test_a_failure_that_repeats_backs_off_to_the_cadence(tmp_path, monkeypatch):
    """Every eigen solve fails (13 passes, one beta). Passes 0 and 1 evaluate (unknown); after 2 failures in a row the row evaluates at
    `every` 5 passes since the last try (6, 11) and at the last pass (12): 5 evaluations, not 13. The skipped passes carry the status
    of the failure (no value is known), the log has one first-failure line with the status text, and the run is not judged."""
    calls = flaky(monkeypatch, lambda n: True)
    code, entries, verdict, log = run(tmp_path, iterations=12)
    assert code == 0 and len(entries) == 13 and len(calls) == 5
    assert attempts(entries) == [0, 1, 6, 11, 12]
    reasons = [e["reason"] for e in entries]
    assert reasons[:2] == ["unknown", "unknown"] and reasons[6] == reasons[11] == "every" and reasons[12] == "last_pass"
    assert {r for i, r in enumerate(reasons) if i not in (0, 1, 6, 11, 12)} == {"skipped"}
    assert {e["status"] for e in entries} == {bk.NOT_CONVERGED}                  # the skipped passes too: no value is known
    assert [e["n_fail"] for e in entries] == [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5]
    assert verdict["feasible"] is False and verdict["measures"]["max_g"]["judged"] is False
    first = [x for x in log.splitlines() if "no usable buckling result" in x]
    assert len(first) == 1 and "At pass 0" not in first[0]
    assert "After 2 failures in a row it evaluates only at a stage start, at the last pass and every 5 passes" in first[0]
    assert "The case may have no compression" in first[0] and "raise buckling.max_seconds" not in first[0]


@needs_direct_solver
def test_a_result_ends_the_back_off(tmp_path, monkeypatch):
    """The first two solves fail, the third (pass 6, `every`) gives a result: the row is unknown with one result, so pass 7 evaluates
    too, and then the skip rule is the one of a known row (the next evaluation is the last pass, 12)."""
    calls = flaky(monkeypatch, lambda n: n < 2)
    code, entries, verdict, log = run(tmp_path, iterations=12)
    assert code == 0 and len(calls) == 5
    assert [i for i, e in enumerate(entries) if e["evaluated"]] == [6, 7, 12] and attempts(entries) == [0, 1, 6, 7, 12]
    assert [e["reason"] for e in entries if e["reason"] != "skipped"] == ["unknown", "unknown", "every", "unknown", "last_pass"]
    assert [e["evaluated"] for e in entries][6:8] == [True, True] and entries[12]["evaluated"]
    assert entries[3]["status"] == bk.NOT_CONVERGED and entries[8]["status"] == "skipped"       # a value is known again from pass 6
    assert verdict["feasible"] is True and "judged" not in verdict["measures"]["max_g"]


def test_the_back_off_in_the_reason_table():
    """`evaluation_reason` after `KNOWN_EVALUATIONS` failures in a row: no `unknown` and no `near`; the cadence counts from the last try."""
    F = 100.0
    back = dict(n_known=0, streak=2, last_try=1, last_lowest=10.0, last_eval=None)
    table = [
        (state(n_known=0, streak=1), FakePoint(2), "unknown"),                        # one failure: ask again at once
        (state(**back), FakePoint(2), None),                                          # 1 pass since the try: skip, `near` does not apply
        (state(**back), FakePoint(5), None),
        (state(**back), FakePoint(6), "every"),                                       # EVERY passes since the last try
        (state(**dict(back, streak=7)), FakePoint(3, beta=2.0), "stage_start"),
        (state(**back), FakePoint(3, last=True), "last_pass"),
        (state(**back), FakePoint(None), "no_pass_number"),
        (state(**dict(back, last_try=4)), FakePoint(8), None),
        (state(**dict(back, last_try=4)), FakePoint(9), "every"),
        (state(n_known=0), FakePoint(2), "unknown"),                                  # a state with no `streak` key (older state)
    ]
    for st, pt, want in table:
        assert br.evaluation_reason(st, pt, F) == want, (st, vars(pt), want)


@needs_direct_solver
def test_the_state_of_a_back_off_is_set_again_and_a_changed_setting_ends_it(tmp_path):
    row = br.BucklingRow(0, 150.0, 40.0, 6, max_seconds=None)
    st = row.get_state()
    assert (st["streak"], st["last_try"], st["fail_status"]) == (0, None, None)
    old = {k: v for k, v in st.items() if k not in ("streak", "last_try", "fail_status")}
    row.set_state(old)                                                              # a state of an earlier run has no back-off keys
    assert row.get_state()["streak"] == 0
    row.set_state(dict(st, n_known=2, streak=3, last_try=7, fail_status=bk.TIME_BUDGET))
    assert row.get_state()["streak"] == 3 and row.get_state()["fail_status"] == bk.TIME_BUDGET
    row2 = br.BucklingRow(0, 160.0, 40.0, 6, max_seconds=None)
    row2.set_state(dict(st, min_factor=150.0, n_known=2, streak=3, last_try=7, fail_status=bk.TIME_BUDGET))
    assert row2.get_state()["streak"] == 0 and row2.get_state()["fail_status"] is None and row2.get_state()["n_known"] == 0


@needs_direct_solver
def test_a_tension_column_with_a_real_failure_resumes_to_the_same_passes(tmp_path, monkeypatch):
    """A real run with a case that fails at every evaluation (a 3 x 3 x 30 column in tension: `not_converged`), killed in the back-off and
    resumed: the same passes and the same bytes as the run that was not stopped (the streak and the last try are in the saved state)."""
    monkeypatch.setattr(tc, "TIMING", tc.TIMING | {"eigen_seconds", "kg_seconds", "adjoint_seconds"})
    p = with_buckling(column_yaml(tmp_path / "c", a=3, b=3, length=30, load=-1500.0, iterations=12, extra_settings=ONE_BETA), min_factor=1.0)
    base = tmp_path / "base"
    tc.finish(p, base)
    got = [next(c for c in r["constraints"] if c["name"] == NAME) for r in tc.history(base)]
    assert attempts(got) == [0, 1, 6, 11, 12] and {e["status"] for e in got} == {bk.NOT_CONVERGED}
    for n, point in enumerate(("after_checkpoint:3", "before_checkpoint:8")):
        out = tmp_path / f"r{n}"
        tc.killed(p, out, point)
        tc.resume(p, out)
        tc.same_run(base, out)
        assert attempts([next(c for c in r["constraints"] if c["name"] == NAME) for r in tc.history(out)]) == [0, 1, 6, 11, 12]
        assert tc.log(out).count("no usable buckling result") == tc.log(base).count("no usable buckling result") == 1   # n3


# --- m2: the text of a cut cluster ---------------------------------------------


def facts_entry(**kw):
    e = {"name": NAME, "value": -0.5, "g": -0.5, "min_factor": 150.0, "lowest_factor": 300.0, "last_eval": 4, "n_pass": 5, "n_eval": 5,
         "n_fail": 0, "evaluated": True, "reason": "last_pass", "status": "ok", "modes": 6, "cluster_cut": False,
         "ks_share_last": 0.01, "low_density_energy_share": 0.0}
    e.update(kw)
    return e


def test_the_cut_cluster_fact_gives_the_measured_band_and_the_limit_of_the_flag():
    assert not [x for x in br._facts(facts_entry()) if "near-repeated level" in x]
    text = " ".join(x for x in br._facts(facts_entry(cluster_cut=True, ks_share_last=1.0)) if "near-repeated level" in x)
    assert "0.23 to 1.8 times this share on a square column" in text and "depends on the basis that ARPACK picks" in text
    assert "The flag sees only the modes that ARPACK returned" in text and "raise buckling.modes" in text
    assert "0.3 to 0.6" not in text


# --- m3: the time facts of `topopt check` -------------------------------------


def test_evaluation_range_is_the_skip_rule_and_every_pass():
    """The hook: 80 iterations, beta stages from 0, 30, 50, 65, two load cases. The low count is the passes that always run (0, 1, 30,
    50, 65, 80) and one evaluation for each 5 passes between them: 18 for each case (the E3 run evaluated `hang` 18 times). The high count is 81 for each case."""
    schedule = ((0, 1.0), (30, 2.0), (50, 4.0), (65, 8.0))
    assert br.evaluation_range(80, schedule, 2) == (36, 162)
    assert br.evaluation_range(80, schedule, 1) == (18, 81)
    assert br.evaluation_range(12, ((0, 1.0),), 1) == (5, 13)             # passes 0, 1, 6, 11 and 12: the run of the back-off test
    assert br.evaluation_range(1, ((0, 1.0), (30, 2.0)), 1) == (2, 2)     # a stage beyond the last pass does not count
    assert br.evaluation_range(0, ((0, 1.0),), 1) == (1, 1)


def test_the_check_gives_the_count_the_measured_cost_and_the_worst_case(tmp_path):
    from topopt.cli import check_data
    on = with_buckling(column_yaml(tmp_path, iterations=30), min_factor=150)
    d = check_data(on)["buckling_row"]
    assert d["evaluations_low"] == br.evaluation_range(30, ((0, 1), (9, 2), (15, 4), (21, 8)), 1)[0] and d["evaluations_high"] == 31
    text = " ".join(d["facts"])
    assert f"The row evaluates {d['evaluations_low']} to 31 times in this run" in text
    assert "9.3 s for `hang` and 12.2 s for `pull_off` on the hook at 1.2 mm" in text and "one machine, 4 threads, E3" in text
    assert "5.5 to 7.2 times the median pass without the row" in text and "The time of another model is not estimated" in text
    assert "costs up to buckling.max_seconds (120 s) for each load case" in text
    assert "time estimate of the run does not include the row" in text
    assert d["warnings"] == []                                              # facts: nothing warns on them


# --- m4: the role files list the optional rows --------------------------------


def test_the_fea_role_lists_the_optional_rows_and_gives_no_count():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for path in (root / "topopt_mcp" / "roles" / "fea.md", root / "agents" / "fea-engineer.md"):
        text = path.read_text(encoding="utf-8")
        line = next(x for x in text.splitlines() if x.startswith("- Optional YAML rows"))
        for key in ("`compliance_limit`", "`stress_limit`", "`limits.displacement`", "`limits.alignment`", "`buckling.min_factor`"):
            assert key in line, (path.name, key)
        assert "two optional rows" not in text and "perimeter" not in line


# --- the nits ---------------------------------------------------------------------


@needs_direct_solver
def test_n3_the_first_failure_line_is_said_once_for_a_state_that_is_set_again(tmp_path, monkeypatch):
    m = loop_model(tmp_path)
    rho = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    lines = []
    row = br.BucklingRow(0, 1e-3, 40.0, 6, max_seconds=None, log=lines.append)
    monkeypatch.setattr(bk, "buckling_modes", fake_modes(bk.NOT_CONVERGED))
    row(point_at(m, rho, 1.0, 0))
    assert len(lines) == 1 and row.get_state()["logged"] is True
    again = br.BucklingRow(0, 1e-3, 40.0, 6, max_seconds=None, log=lines.append)         # a resume: a new object with the saved state
    again.set_state(row.get_state())
    again(point_at(m, rho, 1.0, 1))
    assert len(lines) == 1 and again.get_state()["n_fail"] == 2
    older = {k: v for k, v in row.get_state().items() if k != "logged"}                  # a state with no such key
    fresh = br.BucklingRow(0, 1e-3, 40.0, 6, max_seconds=None, log=lines.append)
    fresh.set_state(older)
    assert fresh.get_state()["logged"] is False
    m.solver.free()


@needs_direct_solver
def test_n5_the_row_does_not_build_the_table_of_active_voxels_a_second_time(tmp_path, monkeypatch):
    """`reduced_system` builds the edof table of all active voxels. The start of a run builds it once for the memory lines, with or without the row."""
    from topopt import optimize
    counts = []
    real = optimize.reduced_system

    def counting(problem):
        counts.append(1)
        return real(problem)

    monkeypatch.setattr(optimize, "reduced_system", counting)
    seen = {}
    for name, on in (("plain", False), ("row", True)):
        counts.clear()
        p = column_yaml(tmp_path / name, iterations=2, extra_settings=ONE_BETA)
        if on:
            with_buckling(p, min_factor=1e-3)
        assert main(["run", str(p), "--out", str(tmp_path / f"{name}_out"), "--quiet"]) == 0
        seen[name] = len(counts)
    assert seen["row"] == seen["plain"]


def test_n7_the_check_says_the_refusal_text_for_gpu_mg_when_opencl_is_missing(tmp_path, monkeypatch):
    from test_buckling_row_wiring import avail
    from topopt.config import load_config
    p = with_buckling(with_settings(column_yaml(tmp_path), solver="gpu-mg"), min_factor=150)
    cfg = load_config(p)
    info = br.preflight(cfg, None, 120_000, 115_836, ram_bytes=64 * 10 ** 9, avail=avail(pardiso=True))   # the solver cannot start here
    assert info["will_run"] is False and info["reason"] == "solver_not_direct"
    assert info["warnings"] == ["buckling row: " + br.config_refusal(cfg, avail(pardiso=True))]
    assert "The solver of this run is gpu-mg" in info["warnings"][0]
