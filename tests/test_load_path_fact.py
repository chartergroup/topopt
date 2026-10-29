"""Tests for the load path fact at the end of a beta stage and of the run (P3a, topopt/stage_facts.py, CONTRACTS.md section 13).

`run()` keeps the load path of the thresholded design in `stage_end.load_path` (the record of the last pass of each beta stage and of the run): which load regions
are not joined to a fixed, spring or joint region. It is the rule of `stress_report.load_path_facts` (6-connected labeling of the design voxels with density 0.5 or
more and all fixed and load voxels), made at each stage end, from the snapshot bytes of that pass: no solve. The run log, `verdict.json` and the MCP tools carry it.
No test and no limit uses it: it is a fact. The volume fact in the same record has its tests in test_stage_facts.py.
"""

import json

import numpy as np
import pytest

from topopt import stage_facts
from topopt.config import StopRules, load_config
from topopt.problem import build_problem
from topopt.stress_report import load_path_facts
from topopt.verdict import build_verdict
from topopt.voxelize import DESIGN, FIXED, LOAD

from test_objective_scale import case_yaml
from test_stage_facts import STAGE_ENDS, read_history, run_case, sum_run  # noqa: F401  (sum_run is a fixture)


# --- the load path fact ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stand(tmp_path_factory):
    return build_problem(load_config(case_yaml(tmp_path_factory.mktemp("lp"), variant="sum")))


def snapshot_of(problem, design_value):
    snap = np.zeros(problem.labels.size, np.uint8)
    snap[((problem.labels == FIXED) | (problem.labels == LOAD)).reshape(-1)] = 255      # fixed and load voxels are solid
    snap[(problem.labels == DESIGN).reshape(-1)] = design_value
    return snap


def test_a_void_design_does_not_join_the_load_region(stand):
    fact = stage_facts.load_path_fact(stand, snapshot_of(stand, 0))
    assert fact["regions"] == {"tip": False} and fact["not_joined"] == ["tip"]
    assert fact["bodies"] == 2 and fact["bodies_with_a_support"] == 1       # the fixed plate, and the load box on its own
    assert "error" not in fact


def test_a_solid_design_joins_the_load_region(stand):
    fact = stage_facts.load_path_fact(stand, snapshot_of(stand, 255))
    assert fact["regions"] == {"tip": True} and fact["not_joined"] == [] and fact["bodies"] == 1


def test_the_cut_is_a_byte_of_128(stand):
    assert stage_facts.load_path_fact(stand, snapshot_of(stand, 127))["not_joined"] == ["tip"]
    assert stage_facts.load_path_fact(stand, snapshot_of(stand, 128))["not_joined"] == []


def test_the_fact_is_the_stress_report_rule(stand):
    snap = snapshot_of(stand, 0)
    snap[np.flatnonzero((stand.labels == DESIGN).reshape(-1))[::2]] = 200       # a half-filled design
    facts = load_path_facts(stand, snap)
    mine = stage_facts.load_path_fact(stand, snap)
    assert mine["regions"] == {n: r["thresholded_design"]["connected_to_fixed"] for n, r in facts["regions"].items()}
    assert mine["bodies"] == facts["thresholded_design_bodies"]


def test_a_wrong_snapshot_gives_an_error_and_no_exception(stand):
    fact = stage_facts.load_path_fact(stand, np.zeros(3, np.uint8))
    assert "error" in fact and "regions" not in fact
    assert stage_facts.load_path_text(fact).startswith("load path: the check was not made (")


def test_the_load_path_sentence_names_the_regions_and_the_beta():
    ok = {"regions": {"tip": True}, "not_joined": [], "bodies": 1, "bodies_with_a_support": 1}
    bad = {"regions": {"a": False, "b": True, "c": False}, "not_joined": ["a", "c"], "bodies": 4, "bodies_with_a_support": 1}
    assert stage_facts.load_path_text(ok, 2.0) == ("load path: every load region is joined to a fixed, spring or joint region "
                                                   "in the thresholded design (density 0.5) at beta 2")
    text = stage_facts.load_path_text(bad, 1.0)
    assert text.startswith("load path: load regions 'a', 'c' are not joined to a fixed, spring or joint region") and "at beta 1." in text
    assert "A low beta gives a grey design" in text
    assert stage_facts.load_path_text({"regions": {}, "not_joined": [], "bodies": 1, "bodies_with_a_support": 1}) == \
        "load path: the problem has no load region"


# --- the run ---

def test_the_load_path_fact_is_the_rule_on_the_snapshot_of_that_pass(sum_run):
    p, out = sum_run
    problem = build_problem(load_config(p))
    for r in read_history(out):
        if "stage_end" not in r:
            continue
        snap = np.frombuffer((out / "snapshots" / f"it_{r['iteration']:04d}.bin").read_bytes(), np.uint8)
        assert r["stage_end"]["load_path"] == stage_facts.load_path_fact(problem, snap)
        facts = load_path_facts(problem, snap)
        assert r["stage_end"]["load_path"]["regions"] == {n: x["thresholded_design"]["connected_to_fixed"] for n, x in facts["regions"].items()}


def test_the_verdict_text_names_a_load_region_that_is_not_joined():
    row = {"iteration": 3, "beta": 2.0, "compliance": 1.0, "grayness": 0.5, "objective": {"value": 1.0}, "constraints": [],
           "stage_end": {"beta": 2.0, "next_beta": None, "load_path": {"regions": {"tip": False}, "not_joined": ["tip"], "bodies": 2,
                                                                       "bodies_with_a_support": 1}}}
    v = build_verdict([row], StopRules(), [(0, 2.0)], 3)
    assert " Facts: load path: load region 'tip' is not joined to a fixed, spring or joint region" in v["text"] and "volume:" not in v["text"]
    assert v["facts"]["load_path"]["not_joined"] == ["tip"]


def test_the_mass_objective_has_the_load_path_fact(tmp_path):
    _, out = run_case(tmp_path, "mass")
    ends = [r["stage_end"] for r in read_history(out) if "stage_end" in r]
    assert len(ends) == 3 and all("volume" not in e and "regions" in e["load_path"] for e in ends)
    v = json.loads((out / "verdict.json").read_text())
    assert "volume:" not in v["text"] and v["facts"]["load_path"] == ends[-1]["load_path"]
    log = (out / "run.log").read_text()
    assert "of the allowed volume used" not in log
    assert len([ln for ln in log.splitlines() if ln.startswith("fact: ") and "load path:" in ln]) == 3      # and the verdict line has it once more
    assert v["text"].count("load path:") == 1




def test_the_record_has_the_load_path_at_the_stage_ends_only(sum_run):
    _, out = sum_run
    rows = read_history(out)
    assert [r["iteration"] for r in rows if "load_path" in r.get("stage_end", {})] == STAGE_ENDS
    for r in rows:
        if "stage_end" in r:
            lp = r["stage_end"]["load_path"]
            assert set(lp) == {"regions", "not_joined", "bodies", "bodies_with_a_support"} and list(lp["regions"]) == ["tip"]
            assert lp["not_joined"] == [n for n, joined in lp["regions"].items() if not joined]


def test_the_run_log_has_the_load_path_line_with_the_beta_of_each_stage(sum_run):
    _, out = sum_run
    lines = (out / "run.log").read_text().splitlines()
    facts = [(i, ln) for i, ln in enumerate(lines) if ln.startswith("fact: ") and "load path:" in ln]
    assert len(facts) == 3
    for (i, ln), it, tag, beta in zip(facts, STAGE_ENDS, ("end of the beta 1 stage, pass 4", "end of the beta 2 stage, pass 9", "last pass 14"), (1, 2, 4)):
        assert ln.startswith(f"fact: {tag}: load path: ") and f"(density 0.5) at beta {beta}" in ln
        at = next(k for k, x in enumerate(lines) if x.startswith(f"it {it:3d}  beta="))
        assert at < i and not any(x.startswith("it ") for x in lines[at + 1:i])      # after the line of its own pass, before the next pass
    rows = read_history(out)
    for (i, ln), r in zip(facts, [r for r in rows if "stage_end" in r]):
        assert ("not joined" in ln) == bool(r["stage_end"]["load_path"]["not_joined"])


def test_the_verdict_has_the_load_path_of_the_last_pass_and_names_it_only_when_a_region_is_not_joined(sum_run):
    _, out = sum_run
    rows = read_history(out)
    v = json.loads((out / "verdict.json").read_text())
    assert v["facts"]["load_path"] == rows[-1]["stage_end"]["load_path"] and v["facts"]["load_path"]["not_joined"] == []
    assert "load path" not in v["text"]                  # every region is joined at the last pass: the text has the volume sentence only


@pytest.mark.parametrize("variant", ["max", "robust", "stress"])
def test_the_other_forms_have_the_load_path_fact(tmp_path, variant):
    _, out = run_case(tmp_path, variant)
    rows = read_history(out)
    assert [r["iteration"] for r in rows if "stage_end" in r] == STAGE_ENDS
    assert all(list(r["stage_end"]["load_path"]["regions"]) == ["tip"] for r in rows if "stage_end" in r)
    assert json.loads((out / "verdict.json").read_text())["facts"]["load_path"] == rows[-1]["stage_end"]["load_path"]


def test_a_check_that_fails_is_a_fact_in_the_log_and_the_verdict_and_does_not_stop_the_run(tmp_path, monkeypatch):
    import topopt.stress_report as sr
    monkeypatch.setattr(sr, "_load_path_facts", lambda *a: (_ for _ in ()).throw(RuntimeError("no labeling today")))
    _, out = run_case(tmp_path)
    rows = read_history(out)
    assert all(r["stage_end"]["load_path"] == {"error": "RuntimeError: no labeling today"} for r in rows if "stage_end" in r)
    assert "load path: the check was not made (RuntimeError: no labeling today)" in (out / "run.log").read_text()
    v = json.loads((out / "verdict.json").read_text())
    assert v["state"] == "iteration_limit" and "Facts: volume: " in v["text"] and "load path: the check was not made" in v["text"]


def test_the_facts_sentence_is_in_the_verdict_text_when_run_status_says_so():
    """P3a review, F6: the docstring of `run_status` says when the text ends with "Facts: ...", and the verdict does exactly that."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    doc = " ".join(server.run_status.__doc__.split())
    assert ('the verdict text ends with "Facts: ..." for each fact that the run has: the volume used (the projected design, and the thresholded part at density 0.5; '
            'only with a volume row); the load path (which load regions the thresholded part does not join to a fixed, spring or joint region; in the text only when '
            'a region is not joined)') in doc
    assert "for a run with objective mass, the `material`" in doc and "the `grey` design" in doc          # F1-3a: a mass run has a Facts sentence of its own

    def verdict_of(end):
        row = {"iteration": 3, "beta": 2.0, "compliance": 1.0, "grayness": 0.5, "objective": {"value": 1.0}, "constraints": [], "stage_end": end}
        return build_verdict([row], StopRules(), [(0, 2.0)], 3)["text"]

    joined = {"regions": {"tip": True}, "not_joined": [], "bodies": 1, "bodies_with_a_support": 1}
    apart = {"regions": {"tip": False}, "not_joined": ["tip"], "bodies": 2, "bodies_with_a_support": 1}
    volume = {"limit": 0.3, "projected": 0.3, "thresholded": 0.2, "used_projected": 1.0, "used_thresholded": 2 / 3}
    base = {"beta": 2.0, "next_beta": None}
    assert " Facts: volume: 1.00 of the allowed volume used" in verdict_of({**base, "volume": volume, "load_path": joined})
    assert "load path" not in verdict_of({**base, "volume": volume, "load_path": joined})                  # all joined: the volume sentence only
    assert " Facts: load path: load region 'tip' is not joined" in verdict_of({**base, "load_path": apart})  # no volume row, a region apart
    assert "Facts" not in verdict_of({**base, "load_path": joined})            # a history from before F1-3a (no volume row, no material fact), all joined: no sentence
