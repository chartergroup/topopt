"""Tests for fix round 1 of the ratchet (D3 review): a missing, partial, stale or unknown fact is never a pass.

Each test names the finding of the review (B1 to B3, M1, M2, m1 to m9, n1 to n7; `verify/d3-review/REVIEW.md`). The keep-rule
tests use plain dicts. The tests with the real `printcheck` need no gmsh and no solve.
"""
import copy
import json
import math
import re
import shutil
import sys
from pathlib import Path

import pytest
import trimesh
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from ratchet_helpers import box_stl  # noqa: E402
from test_feedback import fake_run, make_case  # noqa: E402
from test_ratchet import F, R, case_yaml, states, with_case  # noqa: E402

from topopt import ratchet as rt  # noqa: E402
from topopt.config import UserError, load_config  # noqa: E402
from topopt.ratchet_sources import StlFolderSource  # noqa: E402


def quiet(*a, **k):
    return None


# --- m1: an empty rule set -------------------------------------------------------------------------------------------

def test_m1_an_empty_rule_set_is_refused_by_keep_and_by_make_rules():
    with pytest.raises(UserError, match="no rules"):
        rt.make_rules([])
    empty = rt.Rules((), ("a",), {}, None)
    with pytest.raises(rt.RatchetError, match="no rules"):
        rt.keep(F(), F(), empty)
    with pytest.raises(rt.RatchetError, match="every candidate"):
        rt.keep({"errors": {"judge": {"exit_code": 2, "message": "boom"}}}, F(), empty)     # the reviewer's probe


# --- n1, n2: a value that is not a finite number is not a pass --------------------------------------------------------

NAN, INF = float("nan"), float("inf")
STRESS, NOT_WORSE, CAP = {"rule": "stress"}, {"rule": "stress", "against_parent": "not_worse"}, \
    {"rule": "stress", "against_parent": "cap_or_parent"}
STIFF = {"rule": "stiffness", "max_drop_pct": 5, "source": "test"}
FALL = {"rule": "mass", "must_fall": True}
RISE = {"rule": "mass", "max_rise_pct": 2, "source": "test"}
LOOSE = {"rule": "loose_island", "max_mm3": 10, "source": "test"}


@pytest.mark.parametrize("name,rule,cand,parent,rule_name", [
    ("stress nan", STRESS, with_case(F(), "a", p_by_volume_MPa=NAN), F(), "stress"),
    ("stress -inf", STRESS, with_case(F(), "a", p_by_volume_MPa=-INF), F(), "stress"),
    ("stress text", STRESS, with_case(F(), "a", p_by_volume_MPa="7"), F(), "stress"),
    ("stress true", STRESS, with_case(F(), "a", p_by_volume_MPa=True), F(), "stress"),
    ("not_worse parent nan", NOT_WORSE, F(), with_case(F(), "a", p_by_volume_MPa=NAN), "stress"),
    ("cap_or_parent parent nan", CAP, F(), with_case(F(), "a", p_by_volume_MPa=NAN), "stress"),
    ("stiffness +inf", STIFF, with_case(F(), "a", stiffness_N_per_mm=INF), F(), "stiffness"),
    ("stiffness nan", STIFF, with_case(F(), "a", stiffness_N_per_mm=NAN), F(), "stiffness"),
    ("stiffness zero", STIFF, with_case(F(), "a", stiffness_N_per_mm=0.0), F(), "stiffness"),
    ("stiffness parent nan", STIFF, F(), with_case(F(), "a", stiffness_N_per_mm=NAN), "stiffness"),
    ("mass nan", FALL, F(mass_g=NAN), F(), "mass"),
    ("mass negative (inside-out STL)", FALL, F(mass_g=-1.0), F(), "mass"),      # n2
    ("mass zero", RISE, F(mass_g=0.0), F(), "mass"),
    ("mass parent nan", FALL, F(), F(mass_g=NAN), "mass"),
    ("watertight text false", {"rule": "watertight"}, F(watertight="false"), F(), "watertight"),
    ("watertight number", {"rule": "watertight"}, F(watertight=1), F(), "watertight"),
    ("watertight nan", {"rule": "watertight"}, F(watertight=NAN), F(), "watertight"),
    ("bodies nan", {"rule": "bodies"}, F(bodies=NAN), F(), "bodies"),
    ("loose nan", LOOSE, F(loose_mm3=NAN), F(), "loose_island"),
    ("loose relative, parent nan", dict(LOOSE, relative_to_parent=True), F(), F(loose_mm3=NAN), "loose_island"),
    ("symmetry nan", {"rule": "symmetry", "max_mm": 0.2, "source": "test"}, F(mirror_error_mm=NAN), F(), "symmetry"),
    ("symmetry negative", {"rule": "symmetry", "max_mm": 0.2, "source": "test"}, F(mirror_error_mm=-1.0), F(), "symmetry"),
])
def test_n1_a_value_that_is_not_a_finite_number_is_no_input_and_a_reject(name, rule, cand, parent, rule_name):
    out = rt.keep(cand, parent, R(rule, has_mirror=True))
    assert out.decision == "REJECT" and out.failed == (rule_name,), name
    assert {c.state for c in out.checks if c.rule == rule_name and c.case in (None, "a")} == {"no_input"}, name
    assert all(c.reason for c in out.checks)


# --- B3: a print check that has a limit and is unknown is not a pass ----------------------------------------------------

UNKNOWN_WALL = {"id": "wall_thickness", "limit": 15.0, "limit_key": "min_wall_mm",
                "reason": "the limit 15 mm is above the chord cap of 12 mm"}


def test_b3_print_gate_with_an_unknown_check_that_has_a_limit_is_no_input_with_the_reason():
    out = rt.keep(F(print_unknown=[UNKNOWN_WALL]), F(), R({"rule": "print_gate"}))
    check = out.checks[0]
    assert out.decision == "REJECT" and out.failed == ("print_gate",) and check.state == "no_input"
    assert "wall_thickness" in check.reason and "15" in check.reason and "min_wall_mm" in check.reason
    assert "not a pass" in check.reason and "Options" in check.reason


def test_b3_unknown_without_a_limit_is_a_fact_and_the_gate_passes():
    out = rt.keep(F(print_unknown=[]), F(), R({"rule": "print_gate"}))
    assert out.decision == "ACCEPT" and out.checks[0].state == "pass"


def test_b3_the_gate_state_must_be_pass_and_the_list_of_unknown_checks_must_be_there():
    rules = R({"rule": "print_gate"})
    for gate in ("off", "unknown", 3):
        out = rt.keep(F(print_gate=gate), F(), rules)
        assert out.checks[0].state == "no_input" and "not 'pass'" in out.checks[0].reason, gate
    facts = F()
    del facts["print_unknown"]
    out = rt.keep(facts, F(), rules)
    assert out.checks[0].state == "no_input" and "print_unknown" in out.checks[0].reason
    assert rt.keep(F(print_gate="fail", print_failed=["wall_thickness"]), F(), rules).checks[0].state == "fail"


def test_b3_printcheck_facts_keep_the_unknown_checks_that_have_a_limit(tmp_path):
    from topopt import printcheck
    y = case_yaml(tmp_path, None, print_block={"min_wall_mm": 15})
    stl = box_stl(tmp_path / "box.stl", size=(24, 12, 12))
    facts = rt.printcheck_facts(printcheck.check(stl, yaml_path=y, strict=True), 1.24)
    assert facts["print_gate"] == "pass"                                         # the strict gate does not fail on `unknown`
    assert [u["id"] for u in facts["print_unknown"]] == ["wall_thickness"]
    assert facts["print_unknown"][0]["limit"] == 15.0 and facts["print_unknown"][0]["limit_key"] == "min_wall_mm"
    plain = rt.printcheck_facts(printcheck.check(stl, yaml_path=case_yaml(tmp_path / "p", None), strict=True), 1.24)
    assert plain["print_unknown"] == []                                          # unknown with no limit stays a fact


def real_case(tmp_path, rules, *, print_block=None, cands=None):
    """A fake finished run (a 24 x 12 x 12 box), a YAML with the rules, and a folder of candidate STL files."""
    ypath = make_case(tmp_path / "in")
    doc = yaml.safe_load(ypath.read_text())
    doc["acceptance"] = {"rules": rules}
    if print_block:
        doc["print"] = print_block
    ypath.write_text(yaml.safe_dump(doc, sort_keys=False))
    run = fake_run(tmp_path / "runs" / "a", ypath)
    folder = tmp_path / "cands"
    folder.mkdir()
    if cands is None:
        same = trimesh.creation.box(extents=(24, 12, 12))
        same.apply_translation((12, 0, 6))
        cands = {"same.stl": same}
    for name, mesh in cands.items():
        mesh.export(str(folder / name))
    return run, ypath, folder


def run_real(tmp_path, run, ypath, folder, **kw):
    info, rules = rt.build_rules_for_run(run, ypath)
    res = rt.run_ratchet(run, StlFolderSource(folder), rules, root=tmp_path / "root", yaml_path=info.yaml_path, say=quiet, **kw)
    rows, _ = rt.read_log(tmp_path / "root")
    return res, rows


def test_b3_the_min_wall_15_case_of_the_review_is_a_reject_with_no_input(tmp_path):
    run, ypath, folder = real_case(tmp_path, [{"rule": "print_gate"}], print_block={"min_wall_mm": 15})
    res, rows = run_real(tmp_path, run, ypath, folder)
    base, cand = rows
    assert base["decision"] == "BASELINE" and base["failed"] == ["print_gate"]        # a fact; the run goes on
    assert cand["decision"] == "REJECT" and cand["failed"] == ["print_gate"] and res["accepted"] == []
    check = cand["checks"][0]
    assert check["state"] == "no_input" and "wall_thickness" in check["reason"] and "chord cap" in check["reason"]
    assert cand["facts"]["print_gate"] == "pass" and cand["facts"]["print_unknown"][0]["id"] == "wall_thickness"


def test_b3_the_same_part_without_a_print_limit_is_accepted(tmp_path):
    run, ypath, folder = real_case(tmp_path, [{"rule": "print_gate"}])
    res, rows = run_real(tmp_path, run, ypath, folder)
    assert rows[1]["decision"] == "ACCEPT" and rows[1]["checks"][0]["state"] == "pass" and res["accepted"] == ["c001"]


# --- B2: the loose volume is the volume of all the bodies, or it is no_input -----------------------------------------------

def with_debris(count, size=1.0):
    mesh = trimesh.creation.box(extents=(24, 12, 12))
    mesh.apply_translation((12, 0, 6))
    cubes = []
    for i in range(count):
        cube = trimesh.creation.box(extents=(size, size, size))
        cube.apply_translation((40 + 3 * i, 0, 6))
        cubes.append(cube)
    return trimesh.util.concatenate([mesh, *cubes])


def test_b2_the_loose_volume_counts_every_body_not_the_ten_that_the_report_lists(tmp_path):
    run, ypath, folder = real_case(tmp_path, [{"rule": "loose_island", "max_mm3": 10, "source": "test"}],
                                   cands={"debris12.stl": with_debris(12)})
    res, rows = run_real(tmp_path, run, ypath, folder)
    cand = rows[1]
    assert cand["facts"]["bodies"] == 13 and cand["facts"]["loose_mm3"] == pytest.approx(12.0, abs=1e-3)
    assert cand["facts"]["loose_lower_bound"] is False and "all bodies" in cand["facts"]["sources"]["loose_mm3"]
    assert cand["decision"] == "REJECT" and cand["failed"] == ["loose_island"] and res["accepted"] == []
    assert cand["checks"][0]["state"] == "fail" and "12" in cand["checks"][0]["reason"]


def test_b2_the_exact_sum_lets_a_candidate_under_the_cap_pass(tmp_path):
    run, ypath, folder = real_case(tmp_path, [{"rule": "loose_island", "max_mm3": 13, "source": "test"}],
                                   cands={"debris12.stl": with_debris(12)})
    res, rows = run_real(tmp_path, run, ypath, folder)
    assert rows[1]["decision"] == "ACCEPT" and rows[1]["facts"]["loose_mm3"] == pytest.approx(12.0, abs=1e-3)


def test_b2_when_the_sum_of_all_bodies_cannot_be_made_the_bound_is_no_input_and_never_a_pass(tmp_path, monkeypatch):
    from topopt import printcheck
    real, calls = printcheck.mesh_health, []

    def flaky(V, F):                     # call 1: printcheck of the baseline; 2: printcheck of the candidate; 3: the exact sum
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("no health")
        return real(V, F)
    monkeypatch.setattr(printcheck, "mesh_health", flaky)
    run, ypath, folder = real_case(tmp_path, [{"rule": "loose_island", "max_mm3": 13, "source": "test"}],
                                   cands={"debris12.stl": with_debris(12)})
    res, rows = run_real(tmp_path, run, ypath, folder)
    cand = rows[1]
    assert cand["facts"]["loose_lower_bound"] is True and cand["facts"]["loose_mm3"] == pytest.approx(9.0, abs=1e-3)
    assert cand["decision"] == "REJECT" and cand["checks"][0]["state"] == "no_input"
    assert "lower bound" in cand["checks"][0]["reason"] and "no health" in " ".join(cand["facts"]["notes"])


def test_b2_keep_treats_a_lower_bound_as_no_input_unless_it_is_already_above_the_limit():
    spec = {"rule": "loose_island", "max_mm3": 10, "source": "test"}
    under = rt.keep(F(loose_mm3=9.0, loose_lower_bound=True), F(), R(spec))
    assert under.decision == "REJECT" and under.checks[0].state == "no_input"
    over = rt.keep(F(loose_mm3=11.0, loose_lower_bound=True), F(), R(spec))
    assert over.checks[0].state == "fail" and "lower bound" in over.checks[0].reason
    rel = dict(spec, relative_to_parent=True)
    assert rt.keep(F(loose_mm3=105.0, loose_lower_bound=True), F(loose_mm3=100.0), R(rel)).checks[0].state == "no_input"
    assert rt.keep(F(loose_mm3=115.0, loose_lower_bound=True), F(loose_mm3=100.0), R(rel)).checks[0].state == "fail"
    parent_bound = rt.keep(F(loose_mm3=100.0), F(loose_mm3=100.0, loose_lower_bound=True), R(rel))
    assert parent_bound.checks[0].state == "no_input" and "parent" in parent_bound.checks[0].reason


# --- m2: the mirror error with no plane ----------------------------------------------------------------------------------

def test_m2_no_mirror_plane_is_an_error_and_the_symmetry_rule_is_no_input(tmp_path):
    stl = box_stl(tmp_path / "b.stl", size=(4, 4, 4))
    with pytest.raises(UserError, match="no mirror plane"):
        rt.mirror_error_mm(stl, [])
    shadow = rt.make_run_folder(tmp_path / "shadow", fake_run(tmp_path / "r", make_case(tmp_path / "in")), stl, None,
                                info={"id": "x"})
    ctx = rt.JudgeContext(yaml_path=tmp_path / "in" / "bracket.yaml", needs=frozenset({"mirror"}), mirror_planes=())
    facts = rt.judge_facts(shadow, None, ctx)
    assert "mirror" in facts["errors"] and "no mirror plane" in facts["errors"]["mirror"]["message"]
    out = rt.keep(facts, F(), R({"rule": "symmetry", "max_mm": 0.2, "source": "test"}, has_mirror=True))
    assert out.checks[0].state == "no_input" and out.decision == "REJECT"


# --- M2: delta_pct in every check, a tolerance for not_worse -----------------------------------------------------------------

def stress_a(v):
    return with_case(F(), "a", p_by_volume_MPa=v)


def nw(**kw):
    return R(dict({"rule": "stress", "against_parent": "not_worse", "cases": ("a",)}, **kw))


def test_m2_not_worse_with_no_tolerance_is_strict_and_the_check_says_how_far_the_value_moved():
    out = rt.keep(stress_a(10.02), stress_a(10.0), nw())
    check = out.checks[0]
    assert out.decision == "REJECT" and check.state == "fail"
    assert check.delta_pct == pytest.approx(0.2) and check.delta == pytest.approx(0.02)
    assert "moved +0.2 %" in check.reason and "no tolerance_pct is set" in check.reason
    assert check.to_dict()["delta_pct"] == pytest.approx(0.2)
    better = rt.keep(stress_a(9.9), stress_a(10.0), nw()).checks[0]
    assert better.state == "pass" and better.delta_pct == pytest.approx(-1.0) and "moved -1 %" in better.reason


def test_m2_a_tolerance_with_a_source_lets_a_small_rise_pass_and_a_larger_one_fail():
    rules = nw(tolerance_pct=0.5, tolerance_pct_source="owner: remesh noise 0.3 %")
    small = rt.keep(stress_a(10.02), stress_a(10.0), rules)
    assert small.decision == "ACCEPT" and small.checks[0].delta_pct == pytest.approx(0.2)
    assert "within the tolerance" in small.checks[0].reason and "owner: remesh noise 0.3 %" in small.checks[0].reason
    assert small.checks[0].limit == "<= parent + 0.5 %" and small.checks[0].limit_value == pytest.approx(10.05)
    big = rt.keep(stress_a(10.08), stress_a(10.0), rules)
    assert big.decision == "REJECT" and "the tolerance is 0.5 % (source: owner: remesh noise 0.3 %)" in big.checks[0].reason
    unsourced = rt.keep(stress_a(10.02), stress_a(10.0), nw(tolerance_pct=0.5))
    assert "source not recorded" in unsourced.checks[0].reason


def test_m2_cap_or_parent_uses_the_tolerance_on_the_parent_branch_only():
    over = with_case(with_case(F(), "a", p_by_volume_MPa=60.0), "b", p_by_volume_MPa=5.0)       # the limit of a is 50
    parent = stress_a(70.0)
    cap = {"rule": "stress", "against_parent": "cap_or_parent", "cases": ("a",), "tolerance_pct": 10, "tolerance_pct_source": "t"}
    assert rt.keep(stress_a(76.0), parent, R(cap)).decision == "ACCEPT"         # 70 + 10 % = 77 is the cap
    assert rt.keep(stress_a(78.0), parent, R(cap)).decision == "REJECT"
    assert rt.keep(over, parent, R(cap)).checks[0].delta_pct == pytest.approx(100 * (60 - 70) / 70)
    strict = dict(cap)
    del strict["tolerance_pct"], strict["tolerance_pct_source"]
    assert rt.keep(stress_a(76.0), parent, R(strict)).decision == "REJECT"
    assert rt.keep(stress_a(60.0), None, R(cap)).checks[0].limit == "<= 50 MPa"  # the baseline reads the limit only


def test_m2_every_check_with_a_parent_value_records_delta_and_delta_pct():
    rules = R(STRESS, STIFF, RISE, dict(LOOSE, relative_to_parent=True), NOT_WORSE)
    out = rt.keep(with_case(F(mass_g=1.25, loose_mm3=4.0), "a", stiffness_N_per_mm=99.0), F(mass_g=1.24, loose_mm3=1.0), rules)
    by = {(c.rule, c.case, c.form): c for c in out.checks}
    assert by[("stiffness", "a", "delta")].delta_pct == pytest.approx(-1.0)
    assert by[("mass", None, "delta")].delta == pytest.approx(0.01)
    assert by[("loose_island", None, "delta")].delta == pytest.approx(3.0) and by[("loose_island", None, "delta")].delta_pct == pytest.approx(300.0)
    stress = [c for c in out.checks if c.rule == "stress"]
    assert len(stress) == 4 and all(c.delta_pct == 0.0 for c in stress)
    zero = rt.keep(F(loose_mm3=3.0), F(loose_mm3=0.0), R(dict(LOOSE, relative_to_parent=True))).checks[0]
    assert zero.delta == 3.0 and zero.delta_pct is None and "delta_pct" not in zero.to_dict()


@pytest.mark.parametrize("rule,message", [
    ({"rule": "stress", "tolerance_pct": 0.5}, "against_parent is 'absolute'"),
    ({"rule": "stress", "against_parent": "not_worse", "tolerance_pct": -1}, "must be >= 0.0"),
    ({"rule": "stress", "against_parent": "not_worse", "margin": 0.9}, "reads no limit"),         # n3
    ({"rule": "stress", "against_parent": "not_worse", "tolerance_pct_source": "x"}, "has no 'tolerance_pct'"),
])
def test_m2_n3_the_loader_refuses_a_tolerance_or_a_margin_that_the_form_does_not_read(tmp_path, rule, message):
    with pytest.raises(UserError) as err:
        load_config(case_yaml(tmp_path, {"rules": [rule]}))
    assert message in str(err.value)


def test_m2_the_loader_reads_the_tolerance_with_its_source(tmp_path):
    cfg = load_config(case_yaml(tmp_path, {"rules": [
        {"rule": "stress", "against_parent": "not_worse", "tolerance_pct": 0.5, "tolerance_pct_source": "reviewer noise facts"}]}))
    rule = cfg.acceptance.rules[0]
    assert rule.get("tolerance_pct") == 0.5 and rule.source_of("tolerance_pct") == "reviewer noise facts"
    rule_wide = load_config(case_yaml(tmp_path / "b", {"rules": [
        {"rule": "stress", "against_parent": "cap_or_parent", "tolerance_pct": 1, "margin": 0.9, "source": "lead"}]})).acceptance.rules[0]
    assert rule_wide.source_of("tolerance_pct") == "lead" and rule_wide.source_of("margin") == "lead"


def test_m2_the_tolerance_is_part_of_the_rules_hash(tmp_path):
    a = rt.rules_from_config(load_config(case_yaml(tmp_path, {"rules": [{"rule": "stress", "against_parent": "not_worse"}]})))
    b = rt.rules_from_config(load_config(case_yaml(tmp_path / "b", {"rules": [
        {"rule": "stress", "against_parent": "not_worse", "tolerance_pct": 0.5}]})))
    assert a.digest() != b.digest()


def test_the_check_dict_of_a_record_keeps_delta_pct(tmp_path):
    out = rt.keep(stress_a(10.02), stress_a(10.0), nw())
    row = json.loads(json.dumps(out.to_dict()))
    assert row["checks"][0]["delta_pct"] == pytest.approx(0.2) and not math.isnan(row["checks"][0]["delta"])
    assert copy.deepcopy(row)["checks"][0]["form"] == "not_worse"


# --- check: the facts before a start (M2 note, n6, n7, B3 on the start part) -------------------------------------------------------

def test_n7_check_runs_printcheck_on_the_start_part_and_says_what_the_print_gate_means_for_it(tmp_path):
    run, ypath, folder = real_case(tmp_path, [{"rule": "print_gate"}, {"rule": "watertight"}], print_block={"min_wall_mm": 15})
    rep = rt.check_report(run, ypath, f"stl:{folder}")
    pc = rep["tools"]["printcheck"]
    assert pc["ready"] is True and pc["seconds"] >= 0 and pc["print_gate"] == "pass"
    assert [u["id"] for u in pc["print_unknown"]] == ["wall_thickness"]
    gate = next(r for r in rep["rules"] if r["rule"] == "print_gate")
    assert gate["ready"] is False and "wall_thickness" in gate["missing"][0] and "Every candidate fails" in gate["missing"][0]
    assert next(r for r in rep["rules"] if r["rule"] == "watertight")["ready"] is True and rep["ready"] is False
    assert "printcheck: ready" in rt.format_check(rep)


def test_n7_check_names_a_printcheck_that_cannot_run(tmp_path, monkeypatch):
    run, ypath, folder = real_case(tmp_path, [{"rule": "watertight"}])
    from topopt import printcheck
    monkeypatch.setattr(printcheck, "check", lambda *a, **k: (_ for _ in ()).throw(UserError("cannot read the STL")))
    rep = rt.check_report(run, ypath, f"stl:{folder}")
    assert rep["tools"]["printcheck"]["ready"] is False and rep["ready"] is False
    assert "cannot read the STL" in rep["rules"][0]["missing"][0] and "NOT READY" in rt.format_check(rep)


def test_m2_check_says_when_a_parent_rule_has_no_tolerance_and_the_note_has_no_number(tmp_path):
    run, ypath, folder = real_case(tmp_path, [{"rule": "stress", "against_parent": "not_worse"}])
    stress = next(r for r in rt.check_report(run, ypath, None)["rules"] if r["rule"] == "stress")
    assert len(stress["notes"]) == 1 and "no tolerance_pct" in stress["notes"][0] and "delta_pct" in stress["notes"][0]
    assert re.search(r"\d\s*(%|mm)", stress["notes"][0]) is None                  # the note gives no measured number
    run2, ypath2, _ = real_case(tmp_path / "b", [{"rule": "stress", "against_parent": "not_worse", "tolerance_pct": 0.5}])
    assert next(r for r in rt.check_report(run2, ypath2, None)["rules"] if r["rule"] == "stress")["notes"] == []


def test_n6_the_estimate_for_a_root_that_goes_on_counts_only_what_is_left(tmp_path):
    run, ypath, folder = real_case(tmp_path, [{"rule": "watertight"}])
    for i in range(2, 4):
        box_stl(folder / f"p{i}.stl", size=(24, 12, 12))
    fresh = rt.check_report(run, ypath, f"stl:{folder}", root=tmp_path / "root")
    assert fresh["estimate"]["candidates"] == 3 and fresh["estimate"]["judge_calls"] == 4
    info, rules = rt.build_rules_for_run(run, ypath)
    rt.run_ratchet(run, StlFolderSource(folder), rules, root=tmp_path / "root", yaml_path=info.yaml_path, max_candidates=1, say=quiet)
    again = rt.check_report(run, ypath, f"stl:{folder}", root=tmp_path / "root")["estimate"]
    assert again["candidates"] == 2 and again["already_judged_in_root"] == 1 and again["judge_calls"] == 2


# --- m7: the README example works on the hook ------------------------------------------------------------------------------

def _readme_block():
    text = (REPO / "docs" / "reference" / "verify.md").read_text()
    section = text[text.index("## Ratchet (keep a candidate edit"):]
    return yaml.safe_load(re.search(r"```yaml\n(acceptance:.*?)```", section, re.S).group(1))


def test_m7_the_readme_example_loads_on_the_hook_and_decides_without_a_stress_limit(tmp_path):
    hook = tmp_path / "hook"
    shutil.copytree(REPO / "examples" / "hook", hook)
    doc = yaml.safe_load((hook / "hook.yaml").read_text())
    assert not any("stress_limit" in c or "allowable_mpa" in c for c in doc["load_cases"])        # the hook has no stress limit
    doc.update(_readme_block())
    (hook / "case.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg = load_config(hook / "case.yaml")
    rules = rt.rules_from_config(cfg)
    assert [r.rule for r in rules.rules] == ["watertight", "bodies", "stress", "stiffness", "mass"]
    parent = F(mass_g=1.3, cases={"hang": {"stiffness_N_per_mm": 100.0, "p_by_volume_MPa": 1.0},
                                  "pull_off": {"stiffness_N_per_mm": 100.0, "p_by_volume_MPa": 1.0}})
    cand = F(mass_g=1.2, cases={"hang": {"stiffness_N_per_mm": 99.0, "p_by_volume_MPa": 1.015},
                                "pull_off": {"stiffness_N_per_mm": 99.0, "p_by_volume_MPa": 0.9}})
    out = rt.keep(cand, parent, rules)
    assert out.decision == "ACCEPT", [(c.rule, c.case, c.state, c.reason) for c in out.checks if c.state != "pass"]
    stress = [c for c in out.checks if c.rule == "stress"]
    assert {c.case for c in stress} == {"hang", "pull_off"} and all(c.state == "pass" for c in stress)
    assert "within the tolerance" in stress[0].reason or "moved" in stress[0].reason


# --- the mesh size is a fact too (found by the probe e1 in the re-run) -----------------------------------------------------------

@pytest.mark.parametrize("who", ["candidate", "parent"])
@pytest.mark.parametrize("rule", [{"rule": "stiffness", "max_drop_pct": 5, "source": "t"},
                                  {"rule": "stress", "against_parent": "not_worse"}])
def test_a_solved_comparison_with_an_unknown_mesh_size_is_no_input(rule, who):
    rules = R(rule)
    cand, parent = F(), F()
    (cand if who == "candidate" else parent)["mesh_size_mm"] = None
    out = rt.keep(cand, parent, rules)
    assert out.decision == "REJECT" and set(states(out).values()) == {"no_input"}
    assert "mesh size" in out.checks[0].reason and who in out.checks[0].reason


# --- Integration of D3 (#54) and R2: a relative yaml_path of the start run ---------------------------------------------------

def test_a_run_folder_made_from_a_start_run_with_a_relative_yaml_path_names_the_same_yaml(tmp_path):
    """R2 stores `yaml_path` relative to the run folder (the recording). A ratchet folder is at another depth, so the stored
    text must be resolved at the start run, as `make_run_folder` does for a given YAML."""
    (tmp_path / "cases").mkdir()
    yaml_file = tmp_path / "cases" / "c.yaml"
    yaml_file.write_text("name: c\n")
    start = tmp_path / "recordings" / "r1"
    (start / "result").mkdir(parents=True)
    (start / "run.json").write_text(json.dumps({"yaml_path": "../../cases/c.yaml"}))
    stl = box_stl(tmp_path / "b.stl", size=(4, 4, 4))
    dest = rt.make_run_folder(tmp_path / "deep" / "er" / "shadow", start, stl, None, info={"id": "x"})
    stored = json.loads((dest / "run.json").read_text())["yaml_path"]
    assert Path(stored).is_absolute() and Path(stored) == yaml_file.resolve()
    absolute = rt.make_run_folder(tmp_path / "abs", dest, stl, None, info={"id": "y"})   # an absolute path stays the same
    assert json.loads((absolute / "run.json").read_text())["yaml_path"] == stored
