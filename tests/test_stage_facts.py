"""Tests for the volume-used fact at the end of a beta stage and of the run (P3a, topopt/stage_facts.py, CONTRACTS.md section 13).

`run()` keeps `stage_end` in the history record of the last pass of each beta stage and of the run: the volume used (only with a volume row). The run log,
`verdict.json` (`facts` and the end of the text), `metrics.json` and the MCP `run_status` carry it. No test and no limit uses it: it is a fact.
The load path fact in the same record has its tests in test_load_path_fact.py.
"""

import json

import numpy as np
import pytest
import yaml

from topopt import stage_facts
from topopt.cli import main
from topopt.config import load_config
from topopt.problem import build_problem
from topopt.config import StopRules
from topopt.verdict import build_verdict
from topopt.voxelize import DESIGN

from test_checkpoint import child, clean, finish, history, killed, needs_direct, resume, run_args
from test_objective_scale import case_yaml, sub

STAGE_ENDS = [4, 9, 14]      # the last pass of each stage of the schedule [[0, 1], [5, 2], [10, 4]] with 14 iterations


def run_case(tmp_path, variant="sum", name="r", edit=None, **kw):
    p = case_yaml(sub(tmp_path, name + "_in"), variant=variant, **kw)
    if edit is not None:
        d = yaml.safe_load(p.read_text())
        edit(d)
        p.write_text(yaml.safe_dump(d))
    out = tmp_path / name
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    return p, out


def read_history(out):
    return [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def sum_run(tmp_path_factory):
    """One run of the 2 mm cantilever: the compliance objective with a volume row, schedule [[0, 1], [5, 2], [10, 4]], 14 iterations."""
    return run_case(tmp_path_factory.mktemp("sf"))


# --- the volume fact -------------------------------------------------------------------------------------------

def test_the_thresholded_share_uses_the_print_threshold():
    rho = np.array([0.0, 0.4999, 0.5, 0.5001, 1.0])
    assert stage_facts.thresholded_fraction(rho) == pytest.approx(3 / 5)
    assert stage_facts.THRESHOLD == 0.5
    # the snapshot stores round(255 rho) and the load path rule tests the byte for 128: the same cut as rho >= 0.5
    for r in (0.0, 0.25, 0.4999, 0.4981, 0.5, 0.5001, 0.5019, 0.75, 1.0):
        assert (np.rint(np.clip(r, 0, 1) * 255) >= 128) == (r >= 0.5)


def test_the_volume_fact_divides_by_the_limit_of_the_volume_row():
    rows = [{"name": "compliance:down", "limit": 3.0, "value": 2.0, "g": -0.3},
            {"name": "volume", "limit": 0.3, "value": 0.15, "g": -0.5}]
    rho = np.array([0.9] * 3 + [0.1] * 7)
    v = stage_facts.volume_fact(rows, rho)
    assert v == {"limit": 0.3, "projected": 0.15, "thresholded": 0.3, "used_projected": 0.5, "used_thresholded": pytest.approx(1.0)}
    text = stage_facts.volume_text(v)
    assert text == "volume: 0.50 of the allowed volume used (projected design), 1.00 (thresholded part at 0.5)"


def test_no_volume_row_gives_no_volume_fact():
    assert stage_facts.volume_fact([{"name": "compliance:down", "limit": 3.0, "value": 2.0, "g": -0.3}], np.ones(4)) is None
    assert stage_facts.volume_fact([], np.ones(4)) is None


# --- the run ---------------------------------------------------------------------------------------------------

def test_stage_end_is_in_the_record_of_the_last_pass_of_each_stage_only(sum_run):
    _, out = sum_run
    rows = read_history(out)
    assert [r["iteration"] for r in rows if "stage_end" in r] == STAGE_ENDS
    ends = {r["iteration"]: r["stage_end"] for r in rows if "stage_end" in r}
    assert [(e["beta"], e["next_beta"]) for e in ends.values()] == [(1.0, 2.0), (2.0, 4.0), (4.0, None)]
    assert all({"beta", "next_beta", "volume"} <= set(e) for e in ends.values())


def test_the_volume_fact_is_the_row_and_the_thresholded_part_of_that_pass(sum_run):
    p, out = sum_run
    problem = build_problem(load_config(p))
    design = (problem.labels == DESIGN).reshape(-1)
    for r in read_history(out):
        if "stage_end" not in r:
            continue
        v = r["stage_end"]["volume"]
        row = next(c for c in r["constraints"] if c["name"] == "volume")
        assert v["limit"] == row["limit"] == 0.3 and v["projected"] == row["value"] == pytest.approx(r["volume_fraction"])
        assert v["used_projected"] == pytest.approx(row["value"] / row["limit"])
        snap = np.frombuffer((out / "snapshots" / f"it_{r['iteration']:04d}.bin").read_bytes(), np.uint8)
        assert v["thresholded"] == pytest.approx(float((snap[design] >= 128).mean()))
        assert v["used_thresholded"] == pytest.approx(v["thresholded"] / row["limit"])


def test_the_run_log_has_each_fact_after_the_line_of_its_pass(sum_run):
    _, out = sum_run
    lines = (out / "run.log").read_text().splitlines()
    facts = [i for i, ln in enumerate(lines) if "of the allowed volume used" in ln and ln.startswith("fact: ")]
    assert len(facts) == 3                                                    # a line for each of the passes 4, 9 and 14
    for i in facts:
        assert "(projected design)" in lines[i] and "(thresholded part at 0.5)" in lines[i]
    for it, tag, i in zip(STAGE_ENDS, ("end of the beta 1 stage, pass 4", "end of the beta 2 stage, pass 9", "last pass 14"), facts):
        assert lines[i].startswith(f"fact: {tag}: volume: ")
        at = next(k for k, ln in enumerate(lines) if ln.startswith(f"it {it:3d}  beta="))
        assert at < i
        assert not any(ln.startswith("it ") for ln in lines[at + 1:i])         # the fact follows the line of its own pass


def test_the_verdict_has_the_facts_of_the_last_pass_and_its_text_ends_with_them(sum_run):
    _, out = sum_run
    rows = read_history(out)
    v = json.loads((out / "verdict.json").read_text())
    assert v["facts"] == {"iteration": 14, **rows[-1]["stage_end"]}
    assert v["text"].endswith(" Facts: " + stage_facts.volume_text(rows[-1]["stage_end"]["volume"]) + ".")
    assert "Tests: " in v["text"] and v["text"].index("Tests: ") < v["text"].index(" Facts: ")
    assert v["feasible"] is True and v["converged"] is False         # the facts change no test
    assert json.loads((out / "result" / "metrics.json").read_text())["verdict"]["facts"] == v["facts"]
    assert "verdict: " + v["text"] in (out / "run.log").read_text()


def test_a_volume_fact_far_under_the_limit_does_not_change_the_tests():
    row = {"iteration": 0, "beta": 1.0, "compliance": 1.0, "grayness": 0.5, "objective": {"value": 1.0}, "constraints": [
        {"name": "volume", "limit": 0.3, "value": 0.1, "g": -0.667}]}
    plain = build_verdict([row], StopRules(), [(0, 1.0)], 0)
    fact = {"beta": 1.0, "next_beta": None, "volume": {"limit": 0.3, "projected": 0.1, "thresholded": 0.05, "used_projected": 1 / 3, "used_thresholded": 1 / 6}}
    with_facts = build_verdict([dict(row, stage_end=fact)], StopRules(), [(0, 1.0)], 0)
    assert plain["facts"] is None and "Facts" not in plain["text"]
    assert with_facts["measures"] == plain["measures"] and with_facts["feasible"] is plain["feasible"] is True
    assert with_facts["text"] == plain["text"] + " Facts: volume: 0.33 of the allowed volume used (projected design), 0.17 (thresholded part at 0.5)."
    assert with_facts["facts"] == {"iteration": 0, **fact}


def test_a_finished_run_without_stage_end_in_its_history_has_no_facts():
    row = {"iteration": 0, "beta": 1.0, "compliance": 1.0, "grayness": 0.5, "objective": {"value": 1.0}, "constraints": []}
    v = build_verdict([row], StopRules(), [(0, 1.0)], 0)
    assert v["facts"] is None and "Facts" not in v["text"]
    assert build_verdict([], StopRules(), [(0, 1.0)], 5)["facts"] is None


def test_an_unfinished_history_has_no_facts():
    rows = [{"iteration": i, "beta": 1.0, "compliance": 1.0, "grayness": 0.5, "objective": {"value": 1.0}, "constraints": []} for i in range(3)]
    rows[-1]["stage_end"] = {"beta": 1.0, "next_beta": None}
    v = build_verdict(rows, StopRules(), [(0, 1.0)], 10)
    assert v["state"] == "unfinished" and v["facts"] is None


# --- other runs ------------------------------------------------------------------------------------------------

def test_the_mass_objective_has_no_volume_fact(tmp_path):
    _, out = run_case(tmp_path, "mass")
    ends = [r["stage_end"] for r in read_history(out) if "stage_end" in r]
    assert len(ends) == 3 and all("volume" not in e for e in ends)
    v = json.loads((out / "verdict.json").read_text())
    assert "volume:" not in v["text"] and "volume" not in v["facts"]
    assert "of the allowed volume used" not in (out / "run.log").read_text()


def test_a_stop_rule_ends_the_run_with_the_facts_of_its_pass(tmp_path):
    _, out = run_case(tmp_path, edit=lambda d: d["settings"].update(stop_rules={"time_budget_s": 1e-6}))
    rows = read_history(out)
    v = json.loads((out / "verdict.json").read_text())
    assert v["state"] == "stopped_by_rule" and len(rows) < 15
    assert "stage_end" in rows[-1] and rows[-1]["stage_end"]["next_beta"] is None
    assert v["facts"] == {"iteration": rows[-1]["iteration"], **rows[-1]["stage_end"]}
    assert " Facts: " + stage_facts.volume_text(rows[-1]["stage_end"]["volume"]) in v["text"]       # the design of pass 0 is grey: its thresholded part is void


def test_a_schedule_that_repeats_a_beta_has_no_stage_end_between(tmp_path):
    _, out = run_case(tmp_path, edit=lambda d: d["settings"].update(beta_schedule=[[0, 1], [5, 1], [10, 2]]))
    ends = {r["iteration"]: r["stage_end"] for r in read_history(out) if "stage_end" in r}
    assert list(ends) == [9, 14] and (ends[9]["beta"], ends[9]["next_beta"]) == (1.0, 2.0)


def test_a_single_beta_stage_has_the_facts_of_the_last_pass_only(tmp_path):
    _, out = run_case(tmp_path, edit=lambda d: d["settings"].update(beta_schedule=[[0, 2]]))
    assert [r["iteration"] for r in read_history(out) if "stage_end" in r] == [14]


@pytest.mark.parametrize("variant", ["max", "robust", "stress_compliance"])
def test_the_other_forms_have_the_volume_fact(tmp_path, variant):
    _, out = run_case(tmp_path, variant)
    rows = read_history(out)
    assert [r["iteration"] for r in rows if "stage_end" in r] == STAGE_ENDS
    assert all(r["stage_end"]["volume"]["limit"] == 0.3 for r in rows if "stage_end" in r)
    assert json.loads((out / "verdict.json").read_text())["facts"]["iteration"] == 14


def test_run_and_minimize_follow_the_same_path_and_only_run_makes_the_facts(sum_run):
    """The facts only read the design: the run has the numbers of `minimize` (which has no `stage_end`)."""
    p, out = sum_run
    from topopt.optimize import Model, formulation_from_config, minimize
    model = Model(build_problem(load_config(p)))
    form = formulation_from_config(model.problem.config, model.n_design, model)
    try:
        res = minimize(model, form)
    finally:
        model.solver.free()
    rows = read_history(out)
    assert all("stage_end" not in r for r in res.history)             # `minimize` makes no facts: only `run()` does
    for a, b in zip(rows, res.history):
        assert a["compliance"] == pytest.approx(b["compliance"], rel=1e-8) and a["volume_fraction"] == pytest.approx(b["volume_fraction"], rel=1e-8)


# --- resume ----------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def whole_run(tmp_path_factory):
    """The run in a child process with one thread (`ENV1`), as the resume tests of test_checkpoint.py make it."""
    base = tmp_path_factory.mktemp("whole")
    p = case_yaml(sub(base, "in"), variant="sum")
    finish(p, base / "out")
    return p, base / "out"


@needs_direct
@pytest.mark.parametrize("point", ["after_checkpoint:4", "after_checkpoint:11"])
def test_a_resumed_run_has_the_same_facts(tmp_path, whole_run, point):
    """The kill is at the pass of a stage end (4: the run goes on at the first pass of the next stage) and inside a stage (11).
    The records of the resumed run are those of the whole run."""
    p, whole = whole_run
    cut = tmp_path / "cut"
    killed(p, cut, point)
    resume(p, cut)
    a, b = history(whole), history(cut)
    assert [r["iteration"] for r in b if "stage_end" in r] == STAGE_ENDS
    assert clean([r.get("stage_end") for r in a]) == clean([r.get("stage_end") for r in b])
    va, vb = json.loads((whole / "verdict.json").read_text()), json.loads((cut / "verdict.json").read_text())
    assert va["facts"] == vb["facts"] and va["text"].split(" Facts: ")[1] == vb["text"].split(" Facts: ")[1]
    fact_lines = lambda d: [ln for ln in (d / "run.log").read_text().splitlines() if ln.startswith("fact: end of the beta") or ln.startswith("fact: last pass")]
    assert fact_lines(whole)[-2:] == fact_lines(cut)[-2:] and len(fact_lines(cut)) >= 2


# --- MCP -------------------------------------------------------------------------------------------------------

def test_run_status_and_run_result_carry_the_facts(sum_run):
    pytest.importorskip("mcp")
    from topopt_mcp import replies, server
    _, out = sum_run
    st = server.run_status(str(out))
    assert st["verdict"]["facts"]["iteration"] == 14 and "Facts: volume: " in st["summary"]
    assert "of the allowed volume used (projected design)" in st["summary"]
    res = server.run_result(str(out), preview=False)
    assert res["metrics"]["verdict"]["facts"] == st["verdict"]["facts"]
    assert "verdict.facts" in server.run_status.__doc__ and "metrics.verdict" in replies.guide("run_result")
