"""Tests for the levers of a run (CONTRACTS.md section 41.10, item P7 PR-3).

`topopt/levers_run.py` reads a run folder and lists the facts that the lever cards name (`Symptom.facts`) and that the ENGINE flags in that run,
each with its cards, cheapest layer first. The block `levers` of the MCP tool `run_result` and the command `topopt levers --run <folder>` show it.
The tests use one real short run and copies of its folder with edited files (a flag present, a flag absent).
"""
import ast
import asyncio
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from topopt import levers, levers_run as lr, verdict  # noqa: E402
from topopt import levers_cards as lc  # noqa: E402
from topopt.cli import build_parser, main  # noqa: E402
from topopt.formulation import NOT_MET  # noqa: E402
from topopt_mcp import server  # noqa: E402

from test_optimize import make_case  # noqa: E402

HABIT = re.compile(r"\b(must|never|always)\b", re.I)
CARD_KEYS = {(c.group, c.key): c for c in lc.CARDS}


# --- the folders ------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """One real short run (cantilever, voxel 2 mm, 12 passes): the folder as the engine wrote it."""
    tmp = tmp_path_factory.mktemp("levers_run_real")
    path = make_case(tmp, h=2.0, iterations=12)
    out = tmp / "run"
    assert main(["run", str(path), "--out", str(out), "--quiet"]) == 0
    return out


def read(folder, name):
    return json.loads((Path(folder) / name).read_text())


def write(folder, name, data):
    (Path(folder) / name).write_text(json.dumps(data, indent=1))


def edit(folder, name, fn):
    data = read(folder, name)
    fn(data)
    write(folder, name, data)


@pytest.fixture(scope="module")
def clean(real, tmp_path_factory):
    """A copy of the real run in which no flag holds: every test of the verdict met, a good state, no warning. A case turns on ONE flag."""
    d = tmp_path_factory.mktemp("levers_run_clean") / "run"
    shutil.copytree(real, d)

    def good(m):
        v = m["verdict"]
        v.update(state="converged", converged=True, feasible=True)
        for k in ("objective_change", "grayness", "max_g", "last_beta_stage"):
            v["measures"][k]["met"] = True
        v["facts"]["load_path"]["not_joined"] = []
        m["constraints"] = [dict(c, g=0.0) for c in m["constraints"]]
        m["rows"] = []
        m["mesh"]["stl_roundtrip"]["watertight"] = True
        m["print"]["configured"]["bed_fit"] = True

    edit(d, "result/metrics.json", good)
    edit(d, "status.json", lambda s: s.update(state="done"))
    edit(d, "run.json", lambda r: r.update(lost_force={"cases": [], "max_lost_percent": 0.0}))
    return d


@pytest.fixture
def folder(clean, tmp_path):
    d = tmp_path / "run"
    shutil.copytree(clean, d)
    return d


def facts_of(result):
    return [e["fact"] for e in result["entries"]]


def entry(result, fact):
    found = [e for e in result["entries"] if e["fact"] == fact]
    assert len(found) == 1, (fact, facts_of(result))
    return found[0]


def card_keys(e):
    return [c["key"] for c in e["cards"]]


# --- the real run ----------------------------------------------------------------------------------------------------------------------------------

def test_a_real_short_run_lists_the_flags_of_its_verdict(real):
    """The run ends at its pass limit: the engine says so in the verdict, and the entry has the value from the file and the cards from the table."""
    metrics = read(real, "result/metrics.json")
    assert metrics["verdict"]["state"] == "iteration_limit" and metrics["verdict"]["converged"] is False
    res = lr.run_levers(real)
    assert res["notes"] == [] and res["read"] == ["metrics", "run", "history", "status"]
    state = entry(res, "metrics.verdict.state")
    assert state["hits"] == [{"at": "metrics.verdict.state", "value": "iteration_limit"}]
    # the pass limit: go on from the run (layer 2, two cards of the tool run_topopt), then the whole run again with a new limit (layer 1)
    assert card_keys(state) == ["run_topopt:continue_from", "run_topopt:more_iterations", "settings.iterations"]
    c = next(c for c in state["cards"] if c["key"] == "settings.iterations")
    card = CARD_KEYS[("yaml", "settings.iterations")]
    assert (c["kind"], c["layer"], c["group"]) == (card.kind, card.layer, "yaml") and c["words"] == card.symptoms[0].words
    objective = entry(res, "metrics.verdict.measures.objective_change.met")
    assert objective["hits"][0]["value"] is False
    measure = metrics["verdict"]["measures"]["objective_change"]
    assert objective["hits"][0]["with"] == {"value": measure["value"], "threshold": measure["threshold"]}
    assert {"settings.stop_rules.objective_change", "settings.move_limit"} <= set(card_keys(objective))
    # a fact that is only a plain number is never listed, however many cards name it
    assert "metrics.stiffness_cases[].N_per_mm" not in facts_of(res) and "metrics.mass_final_g" not in facts_of(res)
    assert json.loads(json.dumps(res)) == res                # JSON-safe


def test_a_run_with_no_flag_lists_nothing(clean):
    res = lr.run_levers(clean)
    assert res["entries"] == [] and res["notes"] == []
    assert "No fact that the cards name is flagged" in lr.format_text(res)


def test_the_final_mesh_flag_is_not_listed_for_a_whole_part_and_the_threshold_stability_is_never_listed(real):
    """I6: every run has `metrics.load_paths` and `metrics.threshold_stability`. A part that the mesh joins to its support lists nothing of them; the stability is a fact only."""
    res = lr.run_levers(real)
    assert "metrics.load_paths.regions.<region>.final_stl.connected_to_fixed" not in facts_of(res)
    assert not [f for f in facts_of(res) if f.startswith("metrics.threshold_stability")]
    why = lr.decisions()
    assert "fact only" in why["metrics.threshold_stability.levels[].volume_mm3"] and why["metrics.threshold_stability.levels[].bodies"] == why["metrics.threshold_stability.levels[].volume_mm3"]
    assert why["metrics.load_paths.regions.<region>.final_stl.connected_to_fixed"] == "flagged"
    assert why["metrics.load_paths.regions.<region>.final_stl.touches_a_body"] == "joined"


def test_a_cut_part_lists_the_final_mesh_flag_with_its_load_region_and_the_two_cards(real, tmp_path):
    """The post step run again on a part that is cut in two: `final.stl` does not join the load region, the engine says so, and the entry names the region and the cards."""
    from test_load_paths_stability import cut_copy
    from topopt import post
    out = cut_copy(real, tmp_path)
    post.finish(out)
    e = entry(lr.run_levers(out), "metrics.load_paths.regions.<region>.final_stl.connected_to_fixed")
    assert e["hits"][0]["value"] is False and e["hits"][0]["of"] == "tip" and e["hits"][0]["with"]["touches_a_body"] is True
    assert {"regions.fixed", "settings.mesh_method"} <= set(card_keys(e))
    assert all(c["layer"] in (1, 3) for c in e["cards"]) and json.loads(json.dumps(e)) == e


# --- one case for each flag: present when the flag holds, absent when it does not -----------------------------------------------------------------------

def set_(path, value):
    """An edit of metrics.json: set the value at a dotted path (a list index is `0`)."""
    def fn(m):
        node = m
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        node[keys[-1]] = value
    return fn


def metrics_edit(*fns):
    return lambda d: [edit(d, "result/metrics.json", fn) for fn in fns]


def with_history(fn):
    def apply(d):
        rows = [json.loads(line) for line in (d / "history.jsonl").read_text().splitlines() if line.strip()]
        fn(rows)
        (d / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return apply


def ratchet_log(records):
    def apply(d):
        root = d.parent / (d.name + "-ratchet")
        root.mkdir()
        (root / "ratchet.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return apply


def verify_json(data):
    """Write verify/verify.json in the run folder (the result of `verify_run`; `bore_facts` is a top-level key of it)."""
    def apply(d):
        (d / "verify").mkdir()
        (d / "verify" / "verify.json").write_text(json.dumps(data, indent=1))
    return apply


M = "metrics.verdict.measures"
GATE = {"gate": {"requested": True, "passed": False, "failed": ["stress:hang"], "not_gated": [],
                 "checks": {"stress": {"hang": {"value": 2.0, "limit": 1.0, "met": False, "margin": 0.5}}}}}
BORES = {"bore_facts": [{"name": "pin", "status": "empty", "funnel": {"facets": 0}, "facts": ["no wall facet within the band"]},
                        {"name": "hub", "status": "ok", "funnel": {"facets": 40}, "facts": []}]}
# fact path -> (an edit that turns the flag on, the value that the hit shows, a key that must be among the cards, the `with` of the hit or None)
CASES = {
    "metrics.verdict.state": (metrics_edit(set_("verdict.state", "unfinished")), "unfinished", "settings.iterations", None),
    "metrics.verdict.feasible": (metrics_edit(set_("verdict.feasible", False)), False, "load_cases[].compliance_limit", None),
    "metrics.verdict.notes": (metrics_edit(set_("verdict.notes", ["verdict computed from the history with the default thresholds: the run folder has no verdict.json"])),
                              ["verdict computed from the history with the default thresholds: the run folder has no verdict.json"], "settings.stop_rules", None),
    f"{M}.grayness.met": (metrics_edit(set_("verdict.measures.grayness.met", False), set_("verdict.measures.grayness.value", 0.2)), False,
                          "settings.beta_schedule", {"value": 0.2}),
    f"{M}.objective_change.met": (metrics_edit(set_("verdict.measures.objective_change.met", False), set_("verdict.measures.objective_change.value", 0.03),
                                               set_("verdict.measures.objective_change.threshold", 0.002)),
                                  False, "settings.stop_rules.objective_change", {"value": 0.03, "threshold": 0.002}),
    f"{M}.last_beta_stage.met": (metrics_edit(set_("verdict.measures.last_beta_stage.met", False)), False, "settings.beta_schedule", None),
    "metrics.verdict.facts.load_path.not_joined": (metrics_edit(set_("verdict.facts.load_path.not_joined", ["tip"])), ["tip"], "regions.fixed", None),
    "metrics.stress.cases.<case>.relaxed.met": (metrics_edit(set_("stress", {"cases": {"hang": {"summary": "the optimizer row is NOT met", "stress_limit_MPa": 0.5,
                                                                                               "relaxed": {"met": False, "peak_MPa": 0.52}}}})),
                                                False, "load_cases[].stress_limit", {"peak_MPa": 0.52, "stress_limit_MPa": 0.5}),
    "metrics.stress.cases.<case>.relaxed.peak_above_limit": (
        metrics_edit(set_("stress", {"cases": {"pull_off": {"stress_limit_MPa": 0.5, "relaxed": {"met": True, "peak_MPa": 0.957, "peak_above_limit": True}}}})), True,
        "settings.stress_scale", {"peak_MPa": 0.957, "stress_limit_MPa": 0.5, "met": True}),
    "metrics.load_paths.regions.<region>.final_stl.connected_to_fixed": (
        metrics_edit(set_("load_paths.regions.tip.final_stl.connected_to_fixed", False), set_("load_paths.regions.tip.final_stl.touches_a_body", False),
                     set_("load_paths.regions.tip.final_stl.bodies_with_the_region", 0)),
        False, "regions.fixed", {"touches_a_body": False, "bodies_with_the_region": 0}),
    "metrics.constraints[].g": (metrics_edit(set_("constraints.0.g", 0.5)), 0.5, "load_cases", None),
    "metrics.rows[].met": (metrics_edit(set_("rows", [{"name": "displacement:tip", "met": False, "value_end": 1.2, "limit": 1.0, "g_end": 0.2}])), False,
                           "limits", {"value_end": 1.2, "limit": 1.0, "g_end": 0.2}),
    "metrics.mesh.stl_roundtrip.watertight": (metrics_edit(set_("mesh.stl_roundtrip.watertight", False)), False, "settings.mesh_method", None),
    "metrics.mesh.fallback": (metrics_edit(set_("mesh.fallback", {"error": "RuntimeError: boom"}), set_("mesh.requested", "sdf")), {"error": "RuntimeError: boom"},
                              "settings.mesh_method", {"requested": "sdf"}),
    "metrics.mesh.extra_bodies_removed": (metrics_edit(set_("mesh.extra_bodies_removed", 2), set_("mesh.extra_bodies_volume_mm3", 14.5)), 2,
                                          "settings.mesh_method", {"extra_bodies_volume_mm3": 14.5}),
    "metrics.print.configured.bed_fit": (metrics_edit(set_("print.configured.bed_fit", False)), False, "print.build_direction", None),
    "metrics.buckling.status": (metrics_edit(set_("buckling", {"status": "skipped", "facts": ["too much memory"], "options": ["set max_gb"]})), "skipped",
                                "buckling.max_gb", {"facts": ["too much memory"], "options": ["set max_gb"]}),
    "metrics.buckling.facts": (metrics_edit(set_("buckling", {"status": "done", "facts": ["1 of 2 pairs has mu <= 0"], "options": []})), ["1 of 2 pairs has mu <= 0"],
                               "buckling.max_gb", {"options": []}),
    "metrics.interfaces.error": (metrics_edit(set_("interfaces", {"applied": False, "error": "no cut"})), "no cut", "interfaces", {"applied": False}),
    "metrics.interfaces.applied": (metrics_edit(set_("interfaces", {"applied": False, "error": None})), False, "interfaces", None),
    "metrics.interfaces.items[].as_printed.measured.<measure>.within_tolerance": (
        metrics_edit(set_("interfaces", {"applied": True, "items": [{"name": "pin", "as_printed": {"measured": {
            "radius": {"within_tolerance": False, "deviation_mm": -0.2, "coverage": 0.9, "measured_mm": 1.3, "expected_mm": 1.5}}}}]})), False,
        "interfaces[].radius_mm", {"deviation_mm": -0.2, "coverage": 0.9, "measured_mm": 1.3, "expected_mm": 1.5}),
    "metrics.verdict.interface_tests[].met": (
        metrics_edit(set_("verdict.interface_tests", [{"interface": "socket", "variant": "nominal", "full_coverage": 0.75, "threshold": 0.98, "met": False,
                                                      "surfaces": []}])), False, "interfaces[].full_surface", {"full_coverage": 0.75}),
    "metrics.interfaces.warnings": (
        metrics_edit(set_("interfaces", {"applied": True, "warnings": ["interface 'socket' has full_surface, and the nominal mesh shows 0.75 of its full area"]})),
        ["interface 'socket' has full_surface, and the nominal mesh shows 0.75 of its full area"], "interfaces[].full_surface", None),
    "metrics.filter.facts": (metrics_edit(set_("filter.facts", ["min_member_mm 1 mm is 0.5 voxels at voxel_size 2 mm. A member needs 2 voxels or more to carry load."])),
                             ["min_member_mm 1 mm is 0.5 voxels at voxel_size 2 mm. A member needs 2 voxels or more to carry load."], "settings.min_member_mm", None),
    "metrics.part_only.pinches_left": (metrics_edit(set_("part_only", {"applied": True, "file": "part_only.stl", "min_film_mm": 0.5, "pinches_left": 2, "watertight": True})),
                                       2, "settings.part_only_min_film_mm", None),
    "metrics.part_only.watertight": (metrics_edit(set_("part_only", {"applied": True, "file": "part_only.stl", "watertight": False, "shortest_edge_mm": 0.01})),
                                     False, "settings.part_only_min_film_mm", None),
    "verify.bore_facts[].status": (verify_json(BORES), "empty", "bore_checks.bores[].point",
                                   {"funnel": {"facets": 0}, "facts": ["no wall facet within the band"]}),
    "verify.gate.passed": (verify_json(GATE), False, "verify_run:gate", {"failed": ["stress:hang"]}),
    "run.lost_force.max_lost_percent": (lambda d: edit(d, "run.json", lambda r: r.update(lost_force={"cases": [], "max_lost_percent": 12.5})), 12.5,
                                        "regions.fixed", None),
    "run.derived.load_cases[].applies_load": (lambda d: edit(d, "run.json", lambda r: r["derived"]["load_cases"][0].update(applies_load=False)), False,
                                              "load_cases[].forces[].force", None),
    "run.derived.load_cases[].load_reaches_free_dofs": (lambda d: edit(d, "run.json", lambda r: r["derived"]["load_cases"][0].update(load_reaches_free_dofs=False)),
                                                        False, "load_cases[].forces[].force", None),      # F4-37
    "history.step_acceptance.accepted": (with_history(lambda rows: [r.update(step_acceptance={"tries": 4, "accepted": False}) for r in rows[3:6]]), False,
                                         "settings.stress_step_acceptance", {"tries": 4}),
    "run_status": (lambda d: edit(d, "status.json", lambda s: s.update(state="stopped")), "stopped", "run_topopt:resume", None),
    "ratchet.decision": (ratchet_log([{"id": "c001", "decision": "REJECT", "failed": ["watertight"], "checks": []}]), "REJECT", "ratchet_run", None),
    "ratchet.checks[].state": (ratchet_log([{"id": "c001", "decision": "ACCEPT", "failed": [], "checks": [
        {"rule": "max_drop_pct", "state": "pass"}, {"rule": "watertight", "state": "fail"}]}]), "fail", "acceptance", None),
}


def test_the_cases_cover_every_flag():
    assert set(CASES) == {f.path for f in lr.FLAGS}


@pytest.mark.parametrize("path", sorted(CASES))
def test_a_flag_gives_its_entry_when_it_holds_and_not_when_it_does_not(path, folder):
    turn_on, value, card, joined = CASES[path]
    assert path not in facts_of(lr.run_levers(folder))             # absent in the clean folder
    turn_on(folder)
    res = lr.run_levers(folder)
    e = entry(res, path)
    hits = e["hits"]
    assert hits[0]["value"] == value, hits
    if joined is not None:
        assert hits[0]["with"] == joined, hits
    assert card in card_keys(e), card_keys(e)
    assert e["flag"] and e["cards"]
    paths = {path, *next(f for f in lr.FLAGS if f.path == path).joined}
    for c in e["cards"]:                                           # kind, layer and words are the ones of the table
        t = CARD_KEYS[(c["group"], c["key"])]
        assert (c["kind"], c["layer"]) == (t.kind, t.layer) and set(c["words"].split("; ")) <= {s.words for s in t.symptoms}
        whens = {s.when for s in t.symptoms if paths & set(s.facts) and s.when}       # the `when` text of the symptoms of this entry, where there is one
        assert set(c.get("when", "").split(" | ")) - {""} == whens, (c["key"], c.get("when"), whens)


def test_the_buckling_entry_is_listed_once(folder):
    """A skipped check has a status, facts and options: they are one entry (the joined paths are not listed again)."""
    edit(folder, "result/metrics.json", set_("buckling", {"status": "skipped", "facts": ["f"], "options": ["o"]}))
    res = lr.run_levers(folder)
    assert [f for f in facts_of(res) if f.startswith("metrics.buckling")] == ["metrics.buckling.status"]


def test_a_load_case_that_feels_no_load_is_one_entry_for_each_cause_and_for_an_old_run_json(folder):
    """Review n3: a YAML-zero case has `applies_load` false and `load_reaches_free_dofs` false, and both name the same card: one entry (the second path joins it).
    A case that only loses its force has `load_reaches_free_dofs` false alone. An older run.json has no `load_reaches_free_dofs`: `applies_load` is the entry then."""
    reach, applies = "run.derived.load_cases[].load_reaches_free_dofs", "run.derived.load_cases[].applies_load"
    cases = lambda r: r["derived"]["load_cases"]        # noqa: E731
    edit(folder, "run.json", lambda r: cases(r)[0].update(applies_load=False, load_reaches_free_dofs=False))
    res = lr.run_levers(folder)
    assert [f for f in facts_of(res) if "load_cases" in f] == [reach]
    assert entry(res, reach)["hits"][0]["with"] == {"applies_load": False}
    edit(folder, "run.json", lambda r: cases(r)[0].update(applies_load=True, load_reaches_free_dofs=False))
    res = lr.run_levers(folder)
    assert [f for f in facts_of(res) if "load_cases" in f] == [reach] and entry(res, reach)["hits"][0]["with"] == {"applies_load": True}
    edit(folder, "run.json", lambda r: (cases(r)[0].update(applies_load=False), cases(r)[0].pop("load_reaches_free_dofs")))
    assert [f for f in facts_of(lr.run_levers(folder)) if "load_cases" in f] == [applies]


def test_the_error_and_the_applied_flag_of_the_interfaces_are_one_entry(folder):
    """The step that did not apply has an error text: `applied` joins that entry. Without an error text (null), `applied` is the entry."""
    edit(folder, "result/metrics.json", set_("interfaces", {"applied": False, "error": "no cut"}))
    assert [f for f in facts_of(lr.run_levers(folder)) if f.startswith("metrics.interfaces")] == ["metrics.interfaces.error"]
    edit(folder, "result/metrics.json", set_("interfaces", {"applied": False, "error": None}))
    assert [f for f in facts_of(lr.run_levers(folder)) if f.startswith("metrics.interfaces")] == ["metrics.interfaces.applied"]


def test_the_stress_row_that_is_not_met_names_the_case_and_the_cards_of_the_limit(folder):
    """`relaxed.met` is the flag of the engine (g at or below NOT_MET at the last pass): the entry has the name of the case and the card of the stress limit."""
    edit(folder, "result/metrics.json", set_("stress", {"cases": {"hang": {"relaxed": {"met": True}}, "pull": {"relaxed": {"met": False}}, "old": {"relaxed": None}}}))
    hits = entry(lr.run_levers(folder), "metrics.stress.cases.<case>.relaxed.met")["hits"]
    assert hits == [{"at": "metrics.stress.cases.pull.relaxed.met", "value": False, "of": "pull"}]


def test_a_stress_row_that_is_not_met_shows_the_peak_and_the_limit_and_lists_the_stress_scale_card_too(folder):
    """E2/SS2 and E2/SI110: the entry joins `relaxed.peak_MPa` and `stress_limit_MPa` to the hit, and the card of `settings.stress_scale` names the same fact."""
    CASES["metrics.stress.cases.<case>.relaxed.met"][0](folder)
    e = entry(lr.run_levers(folder), "metrics.stress.cases.<case>.relaxed.met")
    assert {c["key"] for c in e["cards"]} == {"load_cases[].stress_limit", "settings.stress_scale"}
    assert e["hits"][0]["with"] == {"peak_MPa": 0.52, "stress_limit_MPa": 0.5}


def test_a_displacement_row_that_is_not_met_lists_the_limit_card_that_holds_the_e2_rows(folder):
    """E2/LDc: a row of kind displacement with met false has `limits.displacement[].limit_mm` in its entry (the card with the rows LDa, LDb, LDc and the price note), with its `when` text."""
    row = {"name": "displacement:tip_pull:pull_off", "kind": "displacement", "unit": "mm", "limit": 0.019, "value_start": 1.0136, "value_end": 0.0207015, "g_end": 0.0896, "met": False}
    edit(folder, "result/metrics.json", set_("rows", [row]))
    e = entry(lr.run_levers(folder), "metrics.rows[].met")
    assert e["hits"][0]["of"] == "displacement:tip_pull:pull_off" and e["hits"][0]["with"] == {"value_end": 0.0207015, "limit": 0.019, "g_end": 0.0896}
    card = next(c for c in e["cards"] if c["key"] == "limits.displacement[].limit_mm")
    assert card["kind"] == "requirement" and card["layer"] == 1 and "E2/LDc" in card["when"]
    assert {"limits.displacement[].limit_mm", "limits", "limits.displacement[].price"} <= {c["key"] for c in e["cards"]}
    assert "limits.displacement" not in {c["key"] for c in e["cards"]}


def stress_case(met, peak, above, limit=0.5):
    return {"stress_limit_MPa": limit, "relaxed": {"met": met, "peak_MPa": peak, "peak_above_limit": above}}


def test_a_stress_row_that_is_met_with_the_relaxed_peak_above_the_limit_is_listed_by_the_flag_of_the_engine(folder):
    """F4-34, E2/SS05 (`stress_scale` 0.5): met was true with the relaxed peak at 0.957 MPa for a limit of 0.5 MPa. The engine writes `relaxed.peak_above_limit`
    (the test of the run log line), so the block lists the case with the peak and the limit and adds no threshold of its own. `relaxed.met` stays quiet."""
    edit(folder, "result/metrics.json", set_("stress", {"cases": {"pull_off": stress_case(True, 0.957, True), "hang": stress_case(True, 0.04, False, 1.5)}}))
    res = lr.run_levers(folder)
    assert "metrics.stress.cases.<case>.relaxed.met" not in facts_of(res)
    e = entry(res, "metrics.stress.cases.<case>.relaxed.peak_above_limit")
    assert e["hits"] == [{"at": "metrics.stress.cases.pull_off.relaxed.peak_above_limit", "value": True, "of": "pull_off",
                          "with": {"peak_MPa": 0.957, "stress_limit_MPa": 0.5, "met": True}}]
    assert {c["key"] for c in e["cards"]} == {"load_cases[].stress_limit", "settings.stress_scale"}
    assert "peak_above_limit" in next(c for c in e["cards"] if c["key"] == "settings.stress_scale")["when"]


def test_a_row_that_is_not_met_with_the_peak_above_the_limit_has_the_two_entries_and_a_row_with_the_peak_at_the_limit_none(folder):
    edit(folder, "result/metrics.json", set_("stress", {"cases": {"a": stress_case(False, 0.9, True), "b": stress_case(False, 0.5, False), "c": stress_case(True, 0.5, False)}}))
    res = lr.run_levers(folder)
    assert [h["of"] for h in entry(res, "metrics.stress.cases.<case>.relaxed.met")["hits"]] == ["a", "b"]
    assert [h["of"] for h in entry(res, "metrics.stress.cases.<case>.relaxed.peak_above_limit")["hits"]] == ["a"]
    edit(folder, "result/metrics.json", set_("stress", {"cases": {"a": stress_case(True, 0.5, False), "old": {"relaxed": {"met": True, "peak_MPa": 0.5, "peak_above_limit": None}}}}))
    assert not [f for f in facts_of(lr.run_levers(folder)) if f.startswith("metrics.stress")]        # null (an old history) and false are no fact


def test_the_rule_true_flags_the_boolean_true_only():
    assert lr.flagged("true", None, True) is True
    assert [lr.flagged("true", None, v) for v in (False, None, 1, "true", [True])] == [False] * 5
    assert [lr.flagged("false", None, v) for v in (False, None, 0)] == [True, False, False]       # the rule false is as before


@pytest.mark.parametrize("facts, expected", [([], False), (["a warning"], True), (None, False)])
def test_a_filter_warning_is_flagged_when_the_list_has_a_line(folder, facts, expected):
    """`min_feature_note` is written in every run and is no flag; `facts` is a list that is empty unless the engine warns."""
    edit(folder, "result/metrics.json", set_("filter.facts", facts))
    assert ("metrics.filter.facts" in facts_of(lr.run_levers(folder))) is expected


@pytest.mark.parametrize("fallback, extra, expected", [(None, 0, []), (None, 1, ["metrics.mesh.extra_bodies_removed"]),
                                                       ({"message": "grid"}, 0, ["metrics.mesh.fallback"])])
def test_the_mesh_flags_are_the_fallback_and_the_removed_bodies(folder, fallback, extra, expected):
    edit(folder, "result/metrics.json", lambda m: m["mesh"].update(fallback=fallback, extra_bodies_removed=extra))
    assert [f for f in facts_of(lr.run_levers(folder)) if f.startswith("metrics.mesh.") and "watertight" not in f] == expected


@pytest.mark.parametrize("statuses, expected", [(["ok", "ok"], []), (["ok", "empty"], ["hub"]), (["error", "ok"], ["pin"]), (["empty", "error"], ["pin", "hub"])])
def test_a_bore_that_is_not_checked_is_listed_by_its_name(folder, statuses, expected):
    bores = [dict(b, status=st) for b, st in zip(BORES["bore_facts"], statuses)]
    verify_json({"bore_facts": bores})(folder)
    res = lr.run_levers(folder)
    assert "verify" in res["read"]
    got = [h["of"] for e in res["entries"] if e["fact"] == "verify.bore_facts[].status" for h in e["hits"]]
    assert got == expected


def test_the_verify_result_of_an_older_mesh_gives_a_note_and_a_damaged_one_gives_a_note(folder):
    import hashlib
    sha = hashlib.sha256((folder / "result" / "final.stl").read_bytes()).hexdigest()
    verify_json({"run": {"verified_file": "final.stl", "final_stl_sha256": sha}, **BORES})(folder)
    assert lr.run_levers(folder)["notes"] == []                                  # the result is for the mesh that the folder has
    edit(folder, "verify/verify.json", lambda v: v["run"].update(final_stl_sha256="0" * 64))
    res = lr.run_levers(folder)
    assert any("older result/final.stl" in n and "verify_run" in n for n in res["notes"]), res["notes"]
    assert "verify.bore_facts[].status" in facts_of(res)                          # still listed: the note says what it describes
    (folder / "verify" / "verify.json").write_text("{")
    res = lr.run_levers(folder)
    assert any("verify/verify.json does not parse" in n for n in res["notes"]) and "verify" not in res["read"]


# --- fix round 1: a file that is not there is not a file that does not parse (F7), and a verify.json with odd types is no crash (F1) ------------------------------

def shows(d):
    """The three forms of a folder, as `run_result` and the command use them; none raises, and the compact form is JSON."""
    res = lr.run_levers(d)
    json.dumps(lr.compact(res))
    lr.format_text(res)
    return res


@pytest.mark.parametrize("verified_file", [5, 5.5, True, ["final.stl"], {"a": 1}, None, "", "no such file.stl", "a\x00b"])
def test_a_verify_json_with_a_file_name_that_is_not_text_is_no_crash(folder, verified_file):
    """F1: `run.verified_file` is used only when it is text; anything else is the default of verify, `final.stl`."""
    verify_json({"run": {"verified_file": verified_file, "final_stl_sha256": "abc"}, **BORES})(folder)
    res = shows(folder)
    assert "verify.bore_facts[].status" in facts_of(res)
    if verified_file in (5, 5.5, True, None, "") or isinstance(verified_file, (list, dict)):          # the default file is final.stl: its sha256 is not "abc"
        assert any("older result/final.stl" in n for n in res["notes"]), res["notes"]
    else:                                                                                          # a name with no file in result/: no note (nothing to compare)
        assert not any("older result" in n for n in res["notes"]), res["notes"]
    out = server.run_result(str(folder), preview=False)
    assert "error" not in out["levers"] and out["levers"]["entries"]                               # the agent keeps the levers of the run


def test_a_verify_json_with_odd_types_gives_facts(folder):
    for data in ({"run": [1, 2], **BORES}, {"cases": {}}, {"bore_facts": {"a": 1}}, {"bore_facts": ["x", 3, None, {"status": 7}, {"status": "error"}]}):
        (folder / "verify").mkdir(exist_ok=True)
        (folder / "verify" / "verify.json").write_text(json.dumps(data))
        shows(folder)
    (folder / "verify" / "verify.json").write_text("[1, 2, 3]")
    assert any("verify/verify.json does not parse" in n for n in shows(folder)["notes"])


def put(d, name, text):
    path = Path(d) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode())


def drop(d, name):
    (Path(d) / name).unlink()


def only_note(res, part):
    found = [n for n in res["notes"] if part in n]
    assert len(found) == 1, (part, res["notes"])
    return found[0]


# a file that is there and does not parse: a note that says "does not parse" (and never "is not there")
BROKEN_FILES = {
    "metrics_binary": (lambda d: put(d, "result/metrics.json", b"\x89PNG\r\n\x1a\n\x00\xff"), "result/metrics.json does not parse"),
    "metrics_list": (lambda d: put(d, "result/metrics.json", "[1, 2]"), "result/metrics.json does not parse"),
    "metrics_cut": (lambda d: put(d, "result/metrics.json", '{"verdict": '), "result/metrics.json does not parse"),
    "metrics_too_deep": (lambda d: put(d, "result/metrics.json", "[" * 100000), "result/metrics.json does not parse"),
    "run_json_list": (lambda d: put(d, "run.json", "[1]"), "run.json does not parse"),
    "run_json_garbage": (lambda d: put(d, "run.json", "{{{"), "run.json does not parse"),
    "status_garbage": (lambda d: put(d, "status.json", "{{{"), "status.json does not parse"),
    "status_list": (lambda d: put(d, "status.json", "[1]"), "status.json does not parse"),
    "verify_garbage": (lambda d: put(d, "verify/verify.json", "{oops"), "verify/verify.json does not parse"),
    "verify_list": (lambda d: put(d, "verify/verify.json", "[1, 2, 3]"), "verify/verify.json does not parse"),
    "history_unreadable": (lambda d: (drop(d, "history.jsonl"), (d / "history.jsonl").mkdir()), "history.jsonl does not parse"),
    "ratchet_unreadable": (lambda d: (d.parent / (d.name + "-ratchet") / "ratchet.jsonl").mkdir(parents=True), "ratchet.jsonl does not parse"),
}


@pytest.mark.parametrize("name", sorted(BROKEN_FILES))
def test_a_file_that_does_not_parse_gives_a_note_that_says_so(folder, name):
    damage, part = BROKEN_FILES[name]
    damage(folder)
    res = shows(folder)
    only_note(res, part)
    assert not any("is not there" in n for n in res["notes"]), res["notes"]
    if name.startswith("metrics"):
        assert "metrics.verdict.state" in facts_of(res)                       # verdict.json still gives the verdict
    if name.startswith("status"):
        assert "run_status" not in facts_of(res)


# a file that is not there: a note for the three documents of every run, no note for the optional ones
MISSING_FILES = {
    "metrics": (lambda d: drop(d, "result/metrics.json"), "result/metrics.json is not there"),
    "run_json": (lambda d: drop(d, "run.json"), "run.json is not there"),
    "history": (lambda d: drop(d, "history.jsonl"), "history.jsonl is not there"),
}


@pytest.mark.parametrize("name", sorted(MISSING_FILES))
def test_a_file_that_is_not_there_gives_a_note_for_the_documents_of_every_run(folder, name):
    damage, part = MISSING_FILES[name]
    damage(folder)
    res = shows(folder)
    only_note(res, part)
    assert not any("does not parse" in n for n in res["notes"]), res["notes"]


def test_the_optional_files_give_no_note_when_they_are_not_there(folder):
    """status.json, verify/verify.json and the ratchet log: most runs have no verify result and no ratchet; an old run has no status.json."""
    drop(folder, "status.json")
    res = shows(folder)
    assert res["notes"] == [] and "status" not in res["read"] and "verify" not in res["read"] and "ratchet" not in res["read"]


HISTORY_DAMAGE = {
    "garbage_line_in_the_middle": (lambda t: t.splitlines()[:5] + ["{not json"] + t.splitlines()[5:], 1),
    "partial_last_line": (lambda t: [*t.splitlines()[:-1], t.splitlines()[-1][:-40]], 1),
    "lines_that_are_not_objects": (lambda t: t.splitlines()[:3] + ["[1, 2]", '"x"', "5", "null"] + t.splitlines()[3:], 4),
    "two_bad_lines": (lambda t: ["{bad"] + t.splitlines() + ["{bad"], 2),
}


@pytest.mark.parametrize("name", sorted(HISTORY_DAMAGE))
def test_the_skipped_lines_of_the_history_give_one_note_with_their_count(folder, name):
    fn, count = HISTORY_DAMAGE[name]
    rows = len((folder / "history.jsonl").read_text().splitlines())
    put(folder, "history.jsonl", "\n".join(fn((folder / "history.jsonl").read_text())) + "\n")
    res = shows(folder)
    assert only_note(res, "history.jsonl:") == f"history.jsonl: {count} line(s) do not parse and were skipped; the facts of the other lines are read."
    assert "history" in res["read"] and rows > count


def test_a_history_with_no_bad_line_gives_no_note_and_an_empty_one_gives_none_either(folder):
    assert lr.run_levers(folder)["notes"] == []
    put(folder, "history.jsonl", "")
    res = shows(folder)
    assert res["notes"] == [] and "history" in res["read"]


def test_a_ratchet_log_with_bad_lines_gives_one_note_with_the_count_and_its_good_record_is_read(folder):
    root = folder.parent / (folder.name + "-ratchet")
    root.mkdir()
    good = json.dumps({"id": "c001", "decision": "REJECT", "failed": ["x"], "checks": []})
    (root / "ratchet.jsonl").write_text("{bad\n" + good + "\n[1]\n")
    res = shows(folder)
    assert only_note(res, "ratchet.jsonl") == f"{root.name}/ratchet.jsonl: 2 line(s) do not parse and were skipped; the facts of the other lines are read."
    assert "ratchet.decision" in facts_of(res)
    (root / "ratchet.jsonl").write_text("{bad\n[1]\n")
    res = shows(folder)
    assert only_note(res, "ratchet.jsonl").startswith(f"{root.name}/ratchet.jsonl: 2 line(s) do not parse") and "ratchet.decision" not in facts_of(res)
    (root / "ratchet.jsonl").write_text(good + "\n")
    assert shows(folder)["notes"] == []


@pytest.mark.parametrize("fact, key", [("metrics.filter.facts", "settings.min_member_mm"), ("metrics.mesh.fallback", "settings.mesh_method"),
                                       ("metrics.verdict.notes", "settings.stop_rules"), ("metrics.stress.cases.<case>.relaxed.met", "load_cases[].stress_limit")])
def test_the_when_text_of_a_symptom_comes_with_its_card(folder, fact, key):
    """Where a symptom says which value of the fact it means (`Symptom.when`), the card in the entry carries that text: in the full form, the compact form and the command text."""
    CASES[fact][0](folder)
    res = lr.run_levers(folder)
    card = next(c for c in entry(res, fact)["cards"] if c["key"] == key)
    expected = [s.when for s in CARD_KEYS[("yaml", key)].symptoms if fact in s.facts]
    assert card["when"] == expected[0] and expected[0], card
    compact_card = next(c for c in next(e for e in lr.compact(res)["entries"] if e["fact"] == fact)["cards"] if c["key"] == key)
    assert compact_card["when"] == card["when"]
    assert f"Present when: {card['when']}" in lr.format_text(res)


def test_a_card_whose_symptoms_have_no_when_has_no_when_key(real):
    """The `when` of a card in an entry is the text of its symptoms that name a path of the entry; where they have none, the key is not there."""
    entries = lr.run_levers(real)["entries"]
    assert any(c.get("when") for e in entries for c in e["cards"]), "no card of the real run has a when text: the test would prove nothing"
    for e in entries:
        flag = next(f for f in lr.FLAGS if f.path == e["fact"])
        paths = {flag.path, *flag.joined}
        for c in e["cards"]:
            card = CARD_KEYS[(c["group"], c["key"])]
            whens = {s.when for s in card.symptoms if paths & set(s.facts) and s.when}
            assert ("when" in c) is bool(whens), (e["fact"], c["key"])


@pytest.mark.parametrize("g, expected", [(NOT_MET, False), (NOT_MET * 2, True), (-1.0, False)])
def test_a_constraint_row_is_flagged_above_the_not_met_value_of_the_engine(folder, g, expected):
    edit(folder, "result/metrics.json", set_("constraints.0.g", g))
    assert ("metrics.constraints[].g" in facts_of(lr.run_levers(folder))) is expected


def test_a_null_value_is_no_fact(folder):
    """A key that is written as null says nothing: no rule flags it (the rule `not` would, if None were a value)."""
    edit(folder, "result/metrics.json", lambda m: (m["verdict"].update(state=None, feasible=None), m.update(buckling={"status": None, "facts": None})))
    edit(folder, "run.json", lambda r: r.update(lost_force={"max_lost_percent": None}))
    assert lr.run_levers(folder)["entries"] == []


@pytest.mark.parametrize("state, expected", [("converged", False), ("stopped_by_rule", False), ("iteration_limit", True), ("unfinished", True),
                                              ("stopped_by_request", True)])
def test_the_verdict_state_is_flagged_for_an_end_that_is_not_a_convergence(folder, state, expected):
    """A stop rule that ended the run by its own tests is not a fault: only the pass limit, a request and an unfinished history are listed."""
    edit(folder, "result/metrics.json", set_("verdict.state", state))
    assert ("metrics.verdict.state" in facts_of(lr.run_levers(folder))) is expected
    assert state in verdict.STATES


def test_a_row_that_is_not_met_and_the_verdict_that_says_so_are_one_entry(folder):
    """The row (with its name) and `verdict.feasible` name the same cards: one entry. The verdict alone is listed when no row is above the value (a row with no result)."""
    edit(folder, "result/metrics.json", lambda m: (m["constraints"][0].update(g=0.5), m["verdict"].update(feasible=False)))
    res = lr.run_levers(folder)
    assert "metrics.verdict.feasible" not in facts_of(res)
    hit = entry(res, "metrics.constraints[].g")["hits"][0]
    assert hit["value"] == 0.5 and hit["with"] == {"verdict.feasible": False} and hit["of"] == read(folder, "result/metrics.json")["constraints"][0]["name"]
    edit(folder, "result/metrics.json", set_("constraints.0.g", 0.0))
    assert facts_of(lr.run_levers(folder)) == ["metrics.verdict.feasible"]


def test_a_grayness_below_the_verdict_test_with_the_last_beta_stage_not_met_lists_the_stage_and_the_text_says_that_the_grayness_does_not_show_it(folder):
    """E3/LAD-IT40 (40 passes, the default schedule ends at beta 2): grayness 0.085 is under the verdict test of 0.1, so `grayness.met` holds, and `last_beta_stage` is not met.
    The engine flags the stage and not the grayness; the beta schedule card is in the entry, and its grayness symptom gives the fact in its text (`topopt levers --run`)."""
    metrics_edit(set_("verdict.measures.grayness.value", 0.085), set_("verdict.measures.grayness.met", True),
                 set_("verdict.measures.last_beta_stage.met", False))(folder)
    res = lr.run_levers(folder)
    assert f"{M}.grayness.met" not in facts_of(res)
    e = entry(res, f"{M}.last_beta_stage.met")
    assert e["hits"][0]["value"] is False and "settings.beta_schedule" in card_keys(e)
    text = lr.format_text(res)
    assert lc.GRAY_NOT_ENOUGH in text and "E3/LAD-IT40" in text
    assert lc.GRAY_NOT_ENOUGH not in json.dumps(lr.compact(res))       # the short form for `run_result` does not carry the detect text: its size does not change


def test_the_not_met_value_is_the_one_of_the_engine():
    assert NOT_MET == verdict.FEASIBLE_TOL and lr.FLAGS[[f.path for f in lr.FLAGS].index("metrics.constraints[].g")].arg == NOT_MET


@pytest.mark.parametrize("value, expected", [(0.0, False), (0.001, True)])
def test_a_lost_force_is_flagged_above_zero(folder, value, expected):
    edit(folder, "run.json", lambda r: r.update(lost_force={"cases": [], "max_lost_percent": value}))
    assert ("run.lost_force.max_lost_percent" in facts_of(lr.run_levers(folder))) is expected


@pytest.mark.parametrize("state, expected", [("done", False), ("running", False), ("stopped", True), ("error", True)])
def test_the_state_of_the_run_is_flagged_for_a_stop_and_an_error(folder, state, expected):
    edit(folder, "status.json", lambda s: s.update(state=state))
    assert ("run_status" in facts_of(lr.run_levers(folder))) is expected


def test_the_states_that_the_flags_name_are_states_of_the_engine():
    assert "converged" in verdict.STATES and set(verdict.STATES) >= {"iteration_limit", "stopped_by_rule", "stopped_by_request", "unfinished"}
    where = {"runjob.py": '"stopped"', "post.py": 'status("error"', "buckling_voxel.py": '"status": "done"', "ratchet.py": '"REJECT"'}
    for name, word in where.items():
        assert word in (ROOT / "topopt" / name).read_text(encoding="utf-8"), (name, word)
    assert '"fail"' in (ROOT / "topopt" / "ratchet.py").read_text(encoding="utf-8")


def test_a_row_of_a_list_gives_its_own_values_and_its_name(folder):
    """Two rows, one met and one not: one hit, with the name and the numbers of the row that is not met."""
    rows = [{"name": "displacement:a", "met": True, "value_end": 0.5, "limit": 1.0, "g_end": -0.5},
            {"name": "alignment:b", "met": False, "value_end": 2.0, "limit": 1.0, "g_end": 1.0}]
    edit(folder, "result/metrics.json", set_("rows", rows))
    hits = entry(lr.run_levers(folder), "metrics.rows[].met")["hits"]
    assert hits == [{"at": "metrics.rows[1].met", "value": False, "of": "alignment:b", "with": {"value_end": 2.0, "limit": 1.0, "g_end": 1.0}}]


def test_a_history_fact_is_summarised_with_the_passes(folder):
    with_history(lambda rows: [r.update(step_acceptance={"tries": n, "accepted": False}) for n, r in zip((2, 5, 3), rows[2:5])])(folder)
    rows = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines() if line.strip()]
    hits = entry(lr.run_levers(folder), "history.step_acceptance.accepted")["hits"]
    assert hits == [{"at": "history.step_acceptance.accepted", "value": False, "passes": 3, "first_pass": rows[2]["iteration"],
                     "last_pass": rows[4]["iteration"], "with": {"tries": 3}}]


def test_the_ratchet_log_beside_the_run_is_read_by_its_last_record(folder):
    records = [{"id": "c001", "decision": "REJECT", "failed": ["x"], "checks": []}, {"id": "c002", "decision": "ACCEPT", "failed": [], "checks": []}]
    ratchet_log(records)(folder)
    assert "ratchet.decision" not in facts_of(lr.run_levers(folder))             # the last record is an accept
    shutil.rmtree(folder.parent / (folder.name + "-ratchet"))
    assert "ratchet.decision" not in facts_of(lr.run_levers(folder))


# --- the order: the cheapest layer first --------------------------------------------------------------------------------------------------------------

def test_the_order_of_the_layers_follows_their_cost_texts():
    assert sorted(lr.COST_ORDER) == list(lc.LAYERS) and lr.COST_ORDER[0] == 3
    assert "no new pass" in levers.LAYER_TEXT[3] and "seconds" in levers.LAYER_TEXT[3] and "new run" in levers.LAYER_TEXT[1]
    assert lr.LAYERS_LINE.startswith("cheapest layer first: 3 (post step only), 4")


def test_entries_and_cards_are_ordered_by_the_cheapest_layer_first(folder):
    """Layers 3, 4, 2 and 1 all occur: a grayness test (cards of layers 3 and 1), the ratchet (4), a stopped run (2), the state (2, then the whole run of layer 1)."""
    def worse(m):
        m["verdict"]["state"] = "iteration_limit"
        m["verdict"]["measures"]["grayness"].update(met=False, value=0.3)

    edit(folder, "result/metrics.json", worse)
    edit(folder, "status.json", lambda s: s.update(state="stopped"))
    ratchet_log([{"id": "c001", "decision": "REJECT", "failed": ["x"], "checks": []}])(folder)
    res = lr.run_levers(folder)
    rank = lambda layer: lr.COST_ORDER.index(layer)       # noqa: E731
    firsts = [rank(e["cards"][0]["layer"]) for e in res["entries"]]
    assert firsts == sorted(firsts) and {e["cards"][0]["layer"] for e in res["entries"]} == {3, 4, 2}, [(e["fact"], e["cards"][0]["layer"]) for e in res["entries"]]
    assert {c["layer"] for e in res["entries"] for c in e["cards"]} == {3, 4, 2, 1}      # the cards of layer 1 come after the cheaper ones of their entry
    state = entry(res, "metrics.verdict.state")
    assert card_keys(state)[0] == "run_topopt:continue_from" and card_keys(state)[-1] == "settings.iterations"
    for e in res["entries"]:
        layers = [rank(c["layer"]) for c in e["cards"]]
        assert layers == sorted(layers), (e["fact"], [(c["key"], c["layer"]) for c in e["cards"]])
    first = res["entries"][0]
    assert first["fact"] == f"{M}.grayness.met" and card_keys(first)[0] == "settings.stop_rules.grayness_max" and first["cards"][0]["layer"] == 3


# --- a run that stopped, and a folder that is not complete ---------------------------------------------------------------------------------------------

def test_a_run_without_metrics_gives_a_fact_and_the_verdict_not_a_crash(real, tmp_path):
    """A request stopped the run: no result/. The note says what is missing; the verdict.json and the status are read."""
    d = tmp_path / "run"
    shutil.copytree(real, d)
    shutil.rmtree(d / "result")
    edit(d, "status.json", lambda s: s.update(state="stopped"))
    edit(d, "verdict.json", lambda v: v.update(state="stopped_by_request"))
    res = lr.run_levers(d)
    assert any("result/metrics.json is not there" in n and "state: stopped" in n for n in res["notes"]), res["notes"]
    assert "metrics" in res["read"] and "run_status" in facts_of(res)
    assert entry(res, "metrics.verdict.state")["hits"][0]["value"] == "stopped_by_request"
    assert "run_topopt:resume" in card_keys(entry(res, "run_status")) and "stop_run" in card_keys(entry(res, "run_status"))
    assert "metrics.print.configured.bed_fit" not in facts_of(res)
    text = lr.format_text(res)
    assert "Note: result/metrics.json is not there" in text and "run_topopt:resume" in text


def test_a_folder_with_no_file_and_a_damaged_file_give_notes(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    res = lr.run_levers(empty)
    assert res["entries"] == [] and res["read"] == [] and len(res["notes"]) == 3
    (empty / "result").mkdir()
    (empty / "result" / "metrics.json").write_text('{"verdict": ')
    (empty / "run.json").write_text("[1, 2]")
    (empty / "history.jsonl").write_text('{"iteration": 0}\n{"iteration"')
    res = lr.run_levers(empty)
    assert res["entries"] == [] and any("result/metrics.json does not parse" in n for n in res["notes"]) and any("run.json does not parse" in n for n in res["notes"])
    assert any("history.jsonl: 1 line(s) do not parse" in n for n in res["notes"]) and not any("is not there" in n for n in res["notes"])
    assert lr.run_levers(tmp_path / "nothing here")["entries"] == []


# --- the facts that the function reads are facts of the cards (no second list) ---------------------------------------------------------------------------------

def all_fact_paths():
    return {f for c in lc.CARDS for s in c.symptoms for f in s.facts}


def test_every_path_that_the_function_reads_is_a_fact_path_of_a_card():
    facts = all_fact_paths()
    read = {f.path for f in lr.FLAGS} | {j for f in lr.FLAGS for j in f.joined}
    assert read <= facts, sorted(read - facts)
    for f in lr.FLAGS:
        assert f.rule in lr.RULES and f.says and f.anchor, f.path
        assert not HABIT.search(f.says), f.says


def test_every_fact_path_of_the_cards_has_a_decision():
    """A new fact path in a card fails here until `FLAGS` flags it or `NOT_FLAGGED` says why not (the same pattern as 'a new key gets a card')."""
    decided = lr.decisions()
    assert set(decided) == all_fact_paths(), sorted(all_fact_paths() - set(decided))
    stale = [p for paths in lr.NOT_FLAGGED.values() for p in paths if p not in all_fact_paths()]
    assert not stale, f"NOT_FLAGGED names paths that no card has: {stale}"
    listed = [p for paths in lr.NOT_FLAGGED.values() for p in paths]
    assert len(listed) == len(set(listed)), "a path is in NOT_FLAGGED twice"
    assert not set(listed) & ({f.path for f in lr.FLAGS} | {j for f in lr.FLAGS for j in f.joined}), "a flagged or joined path is also in NOT_FLAGGED"
    flagged = {f.path for f in lr.FLAGS}
    assert {p for p, why in decided.items() if why == "flagged"} == flagged
    assert {p for p, why in decided.items() if why == "joined"} == {j for f in lr.FLAGS for j in f.joined} - flagged


def test_every_path_that_is_read_is_in_a_document_of_the_run_folder():
    """A path starts with a source of the cards that the run folder holds (`metrics`, `run`, `history`, `verify`, `ratchet`), or it is a tool fact that status.json holds."""
    for f in lr.FLAGS:
        source, _ = lr._split(f.path)
        assert source in lr.DOCUMENTS, f.path
        assert f.path in lr.TOOL_FACTS or source in lc.FACT_SOURCES, f.path


def test_the_module_has_no_number_of_its_own():
    """The rule: no new threshold, tolerance or limit. No float is typed in the module; the integers are the layers and 0 and 1."""
    tree = ast.parse((ROOT / "topopt" / "levers_run.py").read_text(encoding="utf-8"))
    numbers = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    assert not [v for v in numbers if isinstance(v, float)], numbers
    assert set(numbers) <= {0, 1, 2, 3, 4}, sorted(set(numbers))
    imported = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "formulation"]
    assert [a.name for n in imported for a in n.names] == ["NOT_MET"]


def test_the_texts_of_the_module_have_no_habitual_caps():
    text = lr.format_text({"run_dir": "x", "read": [], "notes": [], "order": lr.LAYERS_LINE, "entries": []}) + lr.POINTER + lr.LAYERS_LINE
    assert not HABIT.search(text) and not HABIT.search(" ".join(why for why in lr.NOT_FLAGGED))


# --- the forms: compact, text ------------------------------------------------------------------------------------------------------------------------

def test_the_compact_form_names_the_cards_by_key_and_gives_no_card_text(real):
    full = lr.run_levers(real)
    block = lr.compact(full)
    assert [e["fact"] for e in block["entries"]] == facts_of(full)
    assert "topopt://levers" in block["cards"] and "topopt://levers/{query}" in block["cards"] and "cheapest layer first" in block["order"]
    for e in block["entries"]:
        for c in e["cards"]:
            assert set(c) <= {"key", "tool", "kind", "layer", "words", "when"} and c["kind"] in lc.KINDS
        for h in e["hits"]:
            assert h.get("at") != e["fact"]                         # the path is not repeated when it is the fact
    text = json.dumps(block)
    for c in lc.CARDS:
        assert c.effect not in text and (not c.call or c.call not in text)


def test_the_compact_form_marks_a_tool_card(folder):
    ratchet_log([{"id": "c001", "decision": "REJECT", "failed": ["x"], "checks": []}])(folder)
    block = lr.compact(lr.run_levers(folder))
    cards = [c for e in block["entries"] for c in e["cards"]]
    assert {"key": "ratchet_run", "tool": True, "kind": "method", "layer": 4, "words": "is the edit allowed, keep or reject"} in cards


def test_the_text_has_each_entry_and_the_full_text_of_each_card_once(real):
    res = lr.run_levers(real)
    text = lr.format_text(res)
    positions = [text.index(f"{n}. {e['fact']}") for n, e in enumerate(res["entries"], 1)]
    assert positions == sorted(positions) and text.startswith("# Levers for the facts of ")
    keys = {(c["group"], c["key"]) for e in res["entries"] for c in e["cards"]}
    for group, key in keys:
        assert len(re.findall(rf"^#### {re.escape(key)}(?:  \(also:.*)?$", text, re.M)) == 1       # the card text of PR-1 (`render_card`), once
        assert levers.render_card(CARD_KEYS[(group, key)]) in text


# --- the MCP tool and the command -----------------------------------------------------------------------------------------------------------------------

def test_run_result_has_the_levers_block(real):
    res = server.run_result(str(real), preview=False)
    assert res["done"] is True and res["metrics"]["verdict"]["state"] == "iteration_limit"
    assert res["levers"] == lr.compact(lr.run_levers(real))
    assert res["levers"]["entries"] and res["levers"]["entries"][0]["cards"][0]["layer"] == 3          # the cheapest layer first
    assert "topopt://levers" in res["levers"]["cards"]
    out = asyncio.run(asyncio.wait_for(server.mcp.call_tool("run_result", {"run_dir": str(real), "preview": False}), 60))
    structured = getattr(out, "structured_content", None) or getattr(out, "structuredContent", None) or json.loads(out.content[0].text)
    assert structured["levers"]["entries"] == res["levers"]["entries"]


def test_run_result_of_a_clean_run_has_an_empty_block(clean):
    res = server.run_result(str(clean), preview=False)
    assert res["levers"]["entries"] == [] and "topopt://levers" in res["levers"]["cards"] and "notes" not in res["levers"]


def test_run_result_of_a_stopped_run_has_the_block(real, tmp_path):
    d = tmp_path / "run"
    shutil.copytree(real, d)
    shutil.rmtree(d / "result")
    edit(d, "status.json", lambda s: s.update(state="stopped"))
    res = server.run_result(str(d))
    assert res["done"] is False and res["stopped"] is True
    assert "run_topopt:resume" in [c["key"] for e in res["levers"]["entries"] for c in e["cards"]] and res["levers"]["notes"]


def test_a_fault_in_the_block_does_not_break_run_result(real, monkeypatch):
    def broken(run_dir):
        raise RuntimeError("boom")
    monkeypatch.setattr(lr, "run_levers", broken)
    res = server.run_result(str(real), preview=False)
    assert res["done"] is True and "mass_final_g" in res["metrics"]
    assert res["levers"]["entries"] == [] and "boom" in res["levers"]["error"]


def test_the_description_and_the_guide_of_run_result_name_the_block_and_the_resource():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    from topopt_mcp import replies
    text = tools["run_result"].description + "\n" + replies.guide("run_result")   # the lines of the keys moved to the guide (#174)
    assert "`levers`" in text and "topopt://levers" in text and "topopt://levers/{query}" in text and "cheapest layer first" in text and len(tools["run_result"].description) <= 4096
    assert not HABIT.search(text.split("`levers`")[1])


def cli(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_the_command_prints_the_entries_of_a_run(capsys, real):
    code, out, err = cli(capsys, "levers", "--run", str(real))
    assert code == 0 and err == ""
    res = lr.run_levers(real)
    assert out == lr.format_text(res)
    for e in res["entries"]:
        assert e["fact"] in out and all(f"- {c['key']} [" in out for c in e["cards"])
    assert "settings.iterations" in out and "#### settings.iterations" in out


def test_the_command_exit_codes(capsys, real, tmp_path):
    code, out, err = cli(capsys, "levers", "--run", str(tmp_path / "none"))
    assert code == 1 and "is not a run folder" in err and out == ""
    code, out, err = cli(capsys, "levers", "--run", str(real), "settings.penal")
    assert code == 1 and "takes no key or word" in err
    code, out, _ = cli(capsys, "levers", "settings.penal")                        # the old form is unchanged
    assert code == 0 and "settings.penal" in out and "Levers for the facts" not in out


def test_the_parser_has_the_option():
    ns = build_parser().parse_args(["levers", "--run", "some/run"])
    assert ns.run == "some/run" and ns.query == []
    assert build_parser().parse_args(["levers", "film"]).run is None
