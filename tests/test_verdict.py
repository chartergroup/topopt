"""Tests for the stop rules and the convergence verdict (topopt/verdict.py, settings.stop_rules). No run is started here."""

import json

import pytest
import yaml

from topopt.config import StopRules, UserError, load_config
from topopt import verdict as V

from test_optimize import make_case

SCHEDULE = [(0, 1.0), (10, 2.0), (20, 4.0)]


def rows(n, *, beta=None, obj=lambda i: 1.0, gray=0.05, g=-0.01, first=0, name="volume"):
    """History records for the passes `first` to `n - 1`. `beta` is a function of the pass number (default: SCHEDULE)."""
    def beta_of(i):
        return max(b for s, b in SCHEDULE if s <= i) if beta is None else beta(i)
    return [{"iteration": i, "beta": beta_of(i), "compliance": 9.0, "objective": {"value": obj(i)}, "grayness": gray,
             "constraints": [{"name": name, "g": g}], "change": 0.01} for i in range(first, n)]


RULES = StopRules(enabled=True)


def test_measures_all_met():
    m = V.measures(rows(26), RULES, SCHEDULE)
    assert V.all_met(m)
    assert m["objective_change"]["value"] == 0.0 and m["objective_change"]["passes_at_beta"] == 6
    assert m["last_beta_stage"]["from_iteration"] == 20 and m["max_g"]["row"] == "volume"


def test_objective_change_needs_window_plus_one_passes_at_one_beta():
    # 5 passes at beta 4 (20 to 24) are not enough for window 5: the change is not judged. 6 passes are.
    m = V.measures(rows(25), RULES, SCHEDULE)["objective_change"]
    assert (m["value"], m["met"], m["passes_at_beta"], m["passes_needed"]) == (None, False, 5, 6)
    assert V.measures(rows(26), RULES, SCHEDULE)["objective_change"]["met"]
    # the same with a step of the objective at the beta step: it is not inside the window
    m = V.measures(rows(26, obj=lambda i: 1.0 if i >= 20 else 3.0), RULES, SCHEDULE)["objective_change"]
    assert m["met"] and m["value"] == 0.0


def test_objective_change_over_the_threshold():
    m = V.measures(rows(30, obj=lambda i: 1.0 + 0.01 * i), RULES, SCHEDULE)["objective_change"]
    assert not m["met"] and 0.0 < m["value"] < 0.01 and m["value"] > RULES.objective_change


def test_a_gap_in_the_passes_ends_the_stretch():
    rs = rows(30)
    del rs[24]    # the history lacks pass 24: the stretch at beta 4 is 25 to 29 (5 passes)
    assert V.measures(rs, RULES, SCHEDULE)["objective_change"]["passes_at_beta"] == 5


def test_grayness_max_g_and_last_stage_are_separate_measures():
    assert not V.measures(rows(26, gray=0.2), RULES, SCHEDULE)["grayness"]["met"]
    assert not V.measures(rows(26, gray=None), RULES, SCHEDULE)["grayness"]["met"]
    mg = V.measures(rows(26, g=0.02), RULES, SCHEDULE)["max_g"]
    assert not mg["met"] and mg["row"] == "volume" and mg["value"] == 0.02
    assert V.measures(rows(26, g=0.004), RULES, SCHEDULE)["max_g"]["met"]       # inside the tolerance (5e-3)
    assert V.measures(rows(26, g=0.006), RULES, SCHEDULE)["max_g"]["met"] is False
    assert not V.measures(rows(15), RULES, SCHEDULE)["last_beta_stage"]["met"]  # the beta of the last stage is not reached


def test_max_g_names_the_worst_row():
    rs = rows(26)
    rs[-1]["constraints"] = [{"name": "volume", "g": -0.1}, {"name": "stress:down", "g": 0.3}, {"name": "x", "g": 0.1}]
    mg = V.measures(rs, RULES, SCHEDULE)["max_g"]
    assert (mg["row"], mg["value"]) == ("stress:down", 0.3)


def test_the_objective_falls_back_to_compliance():
    rs = rows(26)
    for r in rs:
        del r["objective"]
    assert V.all_met(V.measures(rs, RULES, SCHEDULE))


def test_stop_rules_are_off_by_default():
    assert StopRules().enabled is False
    assert V.stop_decision(rows(26), StopRules(), SCHEDULE, 1e9) is None
    assert V.stop_decision([], RULES, SCHEDULE, 0.0) is None


def test_stop_rule_converged():
    s = V.stop_decision(rows(26), RULES, SCHEDULE, 5.0)
    assert s["rule"] == "converged" and s["iteration"] == 25 and "stop rule 'converged' at pass 25" in s["text"]
    assert V.stop_decision(rows(25), RULES, SCHEDULE, 5.0) is None


def test_stop_rule_time_budget_names_what_is_not_met():
    rules = StopRules(enabled=True, time_budget_s=10.0)
    assert V.stop_decision(rows(12), rules, SCHEDULE, 9.9) is None
    s = V.stop_decision(rows(12), rules, SCHEDULE, 10.0)
    assert s["rule"] == "time_budget" and s["time_budget_s"] == 10.0 and s["elapsed_s"] == 10.0
    assert "Not converged" not in s["text"] and "the beta schedule has not reached its last stage" in s["text"]
    assert "Convergence tests at this pass" in s["text"]
    # a run that has converged is reported as converged, not as out of time
    assert V.stop_decision(rows(26), rules, SCHEDULE, 99.0)["rule"] == "converged"


def test_verdict_states():
    n = 30
    v = V.build_verdict(rows(n), RULES, SCHEDULE, n_iter=n)                 # the last pass is 29: not the limit pass
    assert v["state"] == "unfinished" and v["final_iteration"] == 29 and v["feasible"] is True
    v = V.build_verdict(rows(n + 1), RULES, SCHEDULE, n_iter=n)            # pass 30 is the limit pass, and it converged
    assert v["state"] == "converged" and v["converged"] and v["feasible"] and "converged at the last pass (30)" in v["text"]
    v = V.build_verdict(rows(n + 1, gray=0.3), RULES, SCHEDULE, n_iter=n)
    assert v["state"] == "iteration_limit" and not v["converged"] and v["feasible"]
    assert not v["text"].startswith("Not converged") and "Not converged" not in v["text"]
    assert "grayness 0.3 (threshold 0.1: not met)" in v["text"]
    stop = V.stop_decision(rows(26), RULES, SCHEDULE, 1.0)
    v = V.build_verdict(rows(26), RULES, SCHEDULE, n_iter=n, stop=stop)
    assert v["state"] == "stopped_by_rule" and v["converged"] and v["stop"]["rule"] == "converged"
    assert "the iteration limit is 30" in v["text"]


def test_the_verdict_keeps_feasible_and_converged_apart():
    """KB pitfall verdict-mislabels-feasible-unconverged: `converged` and `feasible` are two facts, each from its own tests."""
    v = V.build_verdict(rows(31, obj=lambda i: 1.0 + 0.02 * i), RULES, SCHEDULE, n_iter=30)
    assert v["feasible"] is True and v["converged"] is False and v["state"] == "iteration_limit"
    assert "infeasible" not in v["text"]
    # a settled objective with a row that is not met: converged, not feasible. The state is not `converged` (it needs both)
    v = V.build_verdict(rows(31, g=0.5), RULES, SCHEDULE, n_iter=30)
    assert v["converged"] is True and v["feasible"] is False and v["state"] == "iteration_limit"
    assert "The objective has settled; row volume is not met" in v["text"]
    assert "largest g +0.5 (row volume, threshold 0.005: not met)" in v["text"]
    # a run that is neither: both facts false, and the text names both tests
    v = V.build_verdict(rows(31, g=0.5, obj=lambda i: 1.0 + 0.02 * i), RULES, SCHEDULE, n_iter=30)
    assert v["converged"] is False and v["feasible"] is False and "The objective has settled" not in v["text"]


def test_the_converged_flag_does_not_read_the_constraint_rows():
    m = V.measures(rows(31, g=0.5), RULES, SCHEDULE)
    assert V.converged_met(m) and not V.all_met(m)
    assert V.converged_met(V.measures(rows(31), RULES, SCHEDULE)) and V.all_met(V.measures(rows(31), RULES, SCHEDULE))


def test_the_stop_rule_converged_needs_converged_and_feasible():
    assert V.stop_decision(rows(26, g=0.5), RULES, SCHEDULE, 1.0) is None     # settled, but a row is not met: no stop


def test_the_text_gives_each_test_with_value_threshold_and_met_or_not_met():
    """The default hook case: a change of 1.4e-4 in 5 passes (limit 1e-4). The text has no leading label that judges the run."""
    v = V.build_verdict(rows(31, obj=lambda i: 1.0 + 0.00015 * i), RULES, SCHEDULE, n_iter=30)
    assert v["state"] == "iteration_limit" and v["converged"] is False and v["feasible"] is True
    text = v["text"]
    assert text.startswith("iteration limit reached at pass 30. Tests: ")
    assert "objective change 0.000149 (relative) in the last 5 passes (threshold 0.0001: not met)" in text
    assert "grayness 0.05 (threshold 0.1: met)" in text
    assert "largest g -0.01 (row volume, threshold 0.005: met)" in text
    assert "the beta schedule has reached its last stage (beta 4 from pass 20)" in text
    v = V.build_verdict(rows(31, obj=lambda i: 1.0 + 0.00003 * i), RULES, SCHEDULE, n_iter=30)
    assert v["state"] == "converged" and v["text"].startswith("converged at the last pass (30). Tests: ")
    assert "(threshold 0.0001: met)" in v["text"]
    v = V.build_verdict(rows(25), RULES, SCHEDULE, n_iter=24)               # 5 passes at beta 4: the change is not judged
    assert "objective change: not judged, it needs 6 passes in a row at one beta (the run has 5 at beta 4)" in v["text"]


def test_the_rule_and_the_verdict_use_the_same_measures():
    stop = V.stop_decision(rows(26), RULES, SCHEDULE, 1.0)
    v = V.build_verdict(rows(26), RULES, SCHEDULE, n_iter=30, stop=stop)
    assert v["measures"] == stop["measures"]


def test_verdict_without_rows_and_json_safety():
    v = V.build_verdict([], RULES, SCHEDULE, n_iter=30)
    assert v["state"] == "unfinished" and v["converged"] is False and v["feasible"] is None
    # an objective of 0 and a change from 0 are finite JSON numbers
    rs = rows(30, obj=lambda i: 0.0 if i < 28 else 1.0)
    text = json.dumps(V.build_verdict(rs, RULES, SCHEDULE, n_iter=30), allow_nan=False)
    assert json.loads(text)["state"] in V.STATES


def test_write_read_and_post_fallback(tmp_path):
    rs = rows(31)
    assert V.read_verdict(tmp_path) is None
    v = V.build_verdict(rs, RULES, SCHEDULE, n_iter=30)
    V.write_verdict(tmp_path, v)
    assert V.read_verdict(tmp_path) == v
    assert V.verdict_for_post(tmp_path, rs, 30) == v                       # the file matches the history: used
    other = V.verdict_for_post(tmp_path, rs[:20], 30)                       # the file does not match: computed again
    assert other["source"] == "history" and any("does not match" in n for n in other["notes"])
    (tmp_path / V.VERDICT_FILE).write_text("{ not json")
    assert V.read_verdict(tmp_path) is None
    (tmp_path / V.VERDICT_FILE).write_text(json.dumps({"format": 99}))
    assert V.read_verdict(tmp_path) is None
    assert V.verdict_for_post(tmp_path / "none", rs, 30)["notes"][0].endswith("the run folder has no verdict.json")


def test_schedule_from_rows():
    assert V.schedule_from_rows(rows(25)) == [(0, 1.0), (10, 2.0), (20, 4.0)]
    assert V.schedule_from_rows([]) == [(0, 1.0)]


# --- settings.stop_rules in the YAML -------------------------------------------------------

def _cfg(tmp_path, value="absent"):
    p = make_case(tmp_path)
    if value != "absent":
        d = yaml.safe_load(p.read_text())
        d["settings"]["stop_rules"] = value
        p.write_text(yaml.safe_dump(d))
    return load_config(p)


def test_stop_rules_config(tmp_path):
    assert _cfg(tmp_path).settings.stop_rules == StopRules(enabled=False)
    assert _cfg(tmp_path, False).settings.stop_rules.enabled is False
    assert _cfg(tmp_path, True).settings.stop_rules == StopRules(enabled=True)
    r = _cfg(tmp_path, {"objective_change": 1e-3, "window": 8, "grayness_max": 0.2, "time_budget_s": 90}).settings.stop_rules
    assert r == StopRules(enabled=True, objective_change=1e-3, window=8, grayness_max=0.2, time_budget_s=90)
    assert _cfg(tmp_path, {"enabled": False, "window": 3}).settings.stop_rules.enabled is False
    assert _cfg(tmp_path, {"objective_change": "1e-4"}).settings.stop_rules.objective_change == 1e-4   # YAML 1.1 text


@pytest.mark.parametrize("value, text", [
    ("yes please", "settings.stop_rules"), ({"enabled": "no"}, "enabled"), ({"windo": 5}, "windo"),
    ({"window": 0}, "window"), ({"window": 2.5}, "window"), ({"objective_change": 0}, "objective_change"),
    ({"grayness_max": 2}, "grayness_max"), ({"time_budget_s": -1}, "time_budget_s")])
def test_stop_rules_config_errors(tmp_path, value, text):
    with pytest.raises(UserError, match=text):
        _cfg(tmp_path, value)


# --- the stop rule 'settled' of a fine pass (#318, CONTRACTS.md section 24.6) ------------------------------------------------------

FINE_SCHEDULE = [(0, 8.0)]
FINE_RULES = StopRules(enabled=True, objective_change=1e-3)


def test_settled_fires_without_the_grayness_and_the_last_beta_stage_tests():
    rs = rows(8, beta=lambda i: 8.0, gray=0.22)       # six passes at one beta, flat objective, grey design (the arm has 0.22 in every run)
    m = V.measures(rs, FINE_RULES, [(0, 8.0), (30, 16.0)])
    assert not m["grayness"]["met"] and not m["last_beta_stage"]["met"] and V.settled_met(m) and not V.all_met(m)
    stop = V.stop_decision(rs, FINE_RULES, [(0, 8.0), (30, 16.0)], 1.0, settled=True)
    assert stop["rule"] == "settled" and stop["iteration"] == 7
    assert "stop rule 'settled' at pass 7" in stop["text"] and "objective change 0" in stop["text"] and "tolerance 0.001" in stop["text"]
    assert "grayness test and the last beta stage test are not used" in stop["text"]
    # the plain rule 'converged' does not fire on the same rows, and without `settled=True` nothing does
    assert V.stop_decision(rs, FINE_RULES, [(0, 8.0), (30, 16.0)], 1.0) is None


def test_settled_needs_the_window_the_tolerance_and_feasible_rows():
    sched = FINE_SCHEDULE
    assert V.stop_decision(rows(5, beta=lambda i: 8.0), FINE_RULES, sched, 0.0, settled=True) is None          # 5 passes: window + 1 are needed
    assert V.stop_decision(rows(6, beta=lambda i: 8.0), FINE_RULES, sched, 0.0, settled=True)["iteration"] == 5
    drift = rows(12, beta=lambda i: 8.0, obj=lambda i: 1.0 + 0.01 * i)                                         # about 1 % a pass, above 1e-3
    assert V.stop_decision(drift, FINE_RULES, sched, 0.0, settled=True) is None
    assert V.stop_decision(drift, StopRules(enabled=True, objective_change=0.05), sched, 0.0, settled=True)["rule"] == "settled"
    assert V.stop_decision(rows(8, beta=lambda i: 8.0, g=0.02), FINE_RULES, sched, 0.0, settled=True) is None  # a row that is not met
    assert V.stop_decision(rows(8, beta=lambda i: 8.0), StopRules(enabled=False), sched, 0.0, settled=True) is None


def test_a_beta_step_restarts_the_window_of_settled():
    beta = lambda i: 4.0 if i < 4 else 8.0                                                                      # noqa: E731
    sched = [(0, 4.0), (4, 8.0)]
    assert V.stop_decision(rows(9, beta=beta), FINE_RULES, sched, 0.0, settled=True) is None                    # 5 passes at beta 8
    assert V.stop_decision(rows(10, beta=beta), FINE_RULES, sched, 0.0, settled=True)["iteration"] == 9


def test_the_verdict_of_a_run_that_settled_is_stopped_by_rule_and_names_the_rule():
    rs = rows(8, beta=lambda i: 8.0, gray=0.22)
    stop = V.stop_decision(rs, FINE_RULES, FINE_SCHEDULE, 0.0, settled=True)
    v = V.build_verdict(rs, FINE_RULES, FINE_SCHEDULE, 20, stop)
    assert v["state"] == "stopped_by_rule" and v["stop"]["rule"] == "settled" and v["final_iteration"] == 7
    assert v["text"].startswith("stop rule 'settled' at pass 7") and "(the iteration limit is 20)" in v["text"]
    assert v["stop_rules"]["objective_change"] == 1e-3


def test_the_verdict_of_a_run_at_its_maximum_says_the_change_is_above_the_tolerance():
    rs = rows(8, beta=lambda i: 8.0, obj=lambda i: 1.0 + 0.01 * i)
    v = V.build_verdict(rs, FINE_RULES, FINE_SCHEDULE, 7)
    assert v["state"] == "iteration_limit" and not v["measures"]["objective_change"]["met"]
    assert "iteration limit reached at pass 7" in v["text"] and "threshold 0.001: not met" in v["text"]
