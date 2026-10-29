"""The rules `buckling`, `alignment` and `displacement` of the ratchet (K1-B, CONTRACTS.md section 36). They read the keys `buckling` and
`motion` of verify.json. The tests use plain fact dicts and a fake verify. They need no gmsh and no solve."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_ratchet import F, R, states  # noqa: E402

from topopt import ratchet as rt  # noqa: E402
from topopt.config import UserError  # noqa: E402

BK = {"rule": "buckling", "min_factor": 5.0, "source": "test"}
DISP = {"rule": "displacement"}
ALIGN = {"rule": "alignment"}


def facts(factor_a=8.0, factor_b=9.0, shift=0.05, tilt=0.002, disp=0.3):
    f = F()
    f["cases"]["a"]["buckling_lowest_factor"] = factor_a
    f["cases"]["b"]["buckling_lowest_factor"] = factor_b
    f["motion_rows"] = {
        "displacement:tip:a": {"kind": "displacement", "case": "a", "limit": 0.5, "value": disp},
        "shift:seats:a": {"kind": "shift", "case": "a", "limit": 0.1, "value": shift},
        "tilt:seats:a": {"kind": "tilt", "case": "a", "limit": 0.005, "value": tilt}}
    return f


def verify_report(factor=8.0, bk_status="done"):
    """A verify.json dict with the keys that `verify_facts` reads."""
    return {"size_mm": 1.5, "cases": {"a": {"stiffness_N_per_mm": 1.0, "compliance_Nmm": 1.0, "max_displacement_mm": 0.1,
                                            "von_mises": {"percentile": 99.9, "percentile_by_volume_MPa": 3.0, "peak_qp_MPa": 4.0}}},
            "buckling": {"status": bk_status, "reason": "memory" if bk_status != "done" else None,
                         "cases": [{"name": "a", "load_factors": [factor, 12.0]}] if bk_status == "done" else []},
            "motion": {"status": "done", "rows": [{"name": "shift:seats:a", "kind": "shift", "case": "a", "limit": 0.1, "tet_value": 0.04,
                                                   "tet_over_limit": 0.4}]}}


# --- the facts of verify ----------------------------------------------------------------------------------------------

def test_verify_facts_read_the_buckling_factor_and_the_motion_rows():
    f = rt.verify_facts(verify_report())
    assert f["cases"]["a"]["buckling_lowest_factor"] == 8.0
    assert f["motion_rows"]["shift:seats:a"] == {"kind": "shift", "case": "a", "limit": 0.1, "value": 0.04}
    assert "buckling_problem" not in f and "motion_problem" not in f
    assert f["sources"]["cases.a.buckling_lowest_factor"].startswith("verify.json:buckling")


def test_verify_facts_say_why_the_buckling_check_has_no_factor():
    f = rt.verify_facts(verify_report(bk_status="skipped"))
    assert "buckling_lowest_factor" not in f["cases"]["a"] and "memory" in f["buckling_problem"]
    plain = verify_report()
    del plain["buckling"], plain["motion"]
    f = rt.verify_facts(plain)
    assert "buckling_problem" not in f and "motion_rows" not in f      # a verify without the check has no key


# --- buckling ---------------------------------------------------------------------------------------------------------

def test_buckling_keeps_a_factor_at_or_above_the_minimum_and_rejects_one_below():
    rules = R(BK)
    assert rt.keep(facts(), facts(), rules).decision == "ACCEPT"
    assert rt.keep(facts(factor_a=5.0), facts(), rules).decision == "ACCEPT"        # on the limit passes
    bad = rt.keep(facts(factor_b=4.9), facts(), rules)
    assert bad.decision == "REJECT" and bad.failed == ("buckling",)
    assert states(bad) == {("buckling", "a"): "pass", ("buckling", "b"): "fail"}
    assert "4.9" in bad.checks[1].reason and bad.checks[1].limit_value == 5.0


def test_buckling_not_worse_compares_with_the_parent_and_takes_a_tolerance():
    rules = R({"rule": "buckling", "against_parent": "not_worse"})
    assert rt.keep(facts(factor_a=8.0), facts(), rules).decision == "ACCEPT"
    assert rt.keep(facts(factor_a=9.0), facts(), rules).decision == "ACCEPT"        # a rise is no harm
    assert rt.keep(facts(factor_a=7.9), facts(), rules).decision == "REJECT"        # no tolerance: any fall fails
    tol = R({"rule": "buckling", "against_parent": "not_worse", "tolerance_pct": 5, "source": "test"})
    assert rt.keep(facts(factor_a=7.7), facts(), tol).decision == "ACCEPT"          # 8 - 5 % = 7.6
    assert rt.keep(facts(factor_a=7.5), facts(), tol).decision == "REJECT"
    assert states(rt.keep(facts(), None, rules))[("buckling", "a")] == "off"        # the baseline has no parent


def test_buckling_cap_or_parent_takes_the_looser_floor():
    rules = R({"rule": "buckling", "min_factor": 10.0, "against_parent": "cap_or_parent", "source": "test"})
    assert rt.keep(facts(factor_a=8.0), facts(factor_a=8.0), rules).checks[0].state == "pass"     # the parent is below 10: the parent counts
    assert rt.keep(facts(factor_a=7.0), facts(factor_a=8.0), rules).checks[0].state == "fail"
    assert rt.keep(facts(factor_a=8.0), None, rules).checks[0].state == "fail"                    # baseline: the floor 10 alone


def test_buckling_without_a_factor_is_no_input_with_the_reason():
    f = facts()
    del f["cases"]["a"]["buckling_lowest_factor"]
    out = rt.keep(f, facts(), R(BK))
    assert out.decision == "REJECT" and states(out)[("buckling", "a")] == "no_input"
    assert "no buckling load factor" in out.checks[0].reason
    f["buckling_problem"] = "the buckling check did not run in verify (status 'skipped', memory)"
    assert "memory" in rt.keep(f, facts(), R(BK)).checks[0].reason
    f = facts()
    f["cases"]["a"]["buckling_lowest_factor"] = None
    assert "found no load factor" in rt.keep(f, facts(), R(BK)).checks[0].reason
    f = facts()
    f["errors"] = {"verify": {"exit_code": 2, "message": "boom"}}
    del f["cases"]["a"]["buckling_lowest_factor"]
    assert "boom" in rt.keep(f, facts(), R(BK)).checks[0].reason


# --- alignment and displacement ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("rule,bad", [(ALIGN, dict(shift=0.2)), (ALIGN, dict(tilt=0.006)), (DISP, dict(disp=0.6))])
def test_motion_rules_keep_a_row_inside_its_limit_and_reject_one_outside(rule, bad):
    rules = R(rule)
    assert rt.keep(facts(), facts(), rules).decision == "ACCEPT"
    out = rt.keep(facts(**bad), facts(), rules)
    assert out.decision == "REJECT" and out.failed == (rule["rule"],)
    assert any(c.state == "fail" and "limits." in (c.limit_source or "") for c in out.checks)


def test_alignment_reads_the_shift_row_and_the_tilt_row_and_displacement_reads_neither():
    out = rt.keep(facts(), facts(), R(ALIGN))
    assert len(out.checks) == 2 and {c.reason.split()[0] for c in out.checks} == {"alignment"}
    assert [c.case for c in out.checks] == ["a", "a"]
    assert len(rt.keep(facts(), facts(), R(DISP)).checks) == 1


def test_motion_rule_not_worse_with_a_tolerance():
    rules = R({"rule": "alignment", "against_parent": "not_worse", "tolerance_pct": 10, "source": "test"})
    assert rt.keep(facts(shift=0.054), facts(shift=0.05), rules).decision == "ACCEPT"      # 0.05 + 10 % = 0.055
    assert rt.keep(facts(shift=0.056), facts(shift=0.05), rules).decision == "REJECT"
    strict = R({"rule": "displacement", "against_parent": "not_worse"})
    assert rt.keep(facts(disp=0.31), facts(disp=0.3), strict).decision == "REJECT"
    assert rt.keep(facts(disp=0.2), facts(disp=0.3), strict).decision == "ACCEPT"


def test_a_motion_rule_with_no_rows_is_no_input_and_says_why():
    f = facts()
    del f["motion_rows"]
    out = rt.keep(f, facts(), R(ALIGN))
    assert out.decision == "REJECT" and out.failed == ("alignment",) and states(out) == {("alignment", None): "no_input"}
    assert "limits.alignment" in out.checks[0].reason
    f["motion_problem"] = "the motion rows did not run in verify (status 'skipped', rows_cannot_work)"
    assert "rows_cannot_work" in rt.keep(f, facts(), R(DISP)).checks[0].reason
    g = facts()
    del g["motion_rows"]
    g["errors"] = {"verify": {"exit_code": 2, "message": "boom"}}
    assert "boom" in rt.keep(g, facts(), R(ALIGN)).checks[0].reason
    assert rt.keep(f, None, R(ALIGN)).failed == ("alignment",)                             # a fact at the baseline


def test_a_non_finite_motion_value_is_no_input():
    f = facts()
    f["motion_rows"]["shift:seats:a"]["value"] = float("nan")
    out = rt.keep(f, facts(), R(ALIGN))
    assert out.decision == "REJECT" and sorted(c.state for c in out.checks) == ["no_input", "pass"]


# --- the YAML ---------------------------------------------------------------------------------------------------------

def test_the_loader_takes_the_new_forms_and_refuses_the_wrong_ones(tmp_path):
    from test_ratchet import case_yaml
    from topopt.config import load_config

    def make_acceptance_yaml(sub, rules):
        return load_config(case_yaml(tmp_path / sub, {"rules": rules}, second_case=False, stress=False))
    ok = make_acceptance_yaml("ok", [{"rule": "buckling", "against_parent": "not_worse", "tolerance_pct": 2, "tolerance_pct_source": "t"},
                                         {"rule": "alignment", "against_parent": "cap_or_parent", "tolerance_pct": 1, "source": "t"},
                                         {"rule": "displacement"}])
    assert [r.rule for r in ok.acceptance.rules] == ["buckling", "alignment", "displacement"]
    for i, (rule, text) in enumerate([({"rule": "buckling"}, "needs min_factor"),
                       ({"rule": "buckling", "min_factor": 3, "against_parent": "not_worse"}, "reads no limit"),
                       ({"rule": "alignment", "tolerance_pct": 1}, "absolute")]):
        with pytest.raises(UserError, match=text):
            make_acceptance_yaml(f"bad{i}", [rule])


# --- the judge --------------------------------------------------------------------------------------------------------

def test_the_judge_turns_the_buckling_check_on_only_for_a_buckling_rule(tmp_path, monkeypatch):
    import topopt.verify.run as vr
    from test_ratchet import quiet, start_run
    from ratchet_helpers import stl_folder
    from topopt.config import load_config
    from topopt.ratchet_sources import StlFolderSource
    calls = []

    def fake_verify(shadow, **kw):
        calls.append(kw)
        rep = verify_report()
        rep["cases"]["down"] = rep["cases"].pop("a")
        rep["buckling"]["cases"][0]["name"] = "down"
        return rep
    monkeypatch.setattr(vr, "verify_run", fake_verify)
    r, ypath = start_run(tmp_path)
    stl_folder(tmp_path / "cands", 1)
    run_json = json.loads((r / "run.json").read_text())
    for name, spec, want in (("with", {"rule": "buckling", "min_factor": 5, "source": "t"}, True),
                             ("without", {"rule": "stiffness", "max_drop_pct": 5, "source": "t"}, False)):
        rules = rt.make_rules([spec], case_names=("down",))
        ctx = rt.JudgeContext.from_config(load_config(ypath), rules, run_json)
        calls.clear()
        res = rt.run_ratchet(r, StlFolderSource(tmp_path / "cands"), rules, root=tmp_path / name, ctx=ctx, yaml_path=ypath, say=quiet)
        assert res["exit_code"] == 0 and res["accepted"] == ["c001"]
        assert all(c["buckling"] is want for c in calls), name
    assert "buckling" in rt.make_rules([BK]).needs() and "buckling" not in rt.make_rules([ALIGN]).needs()


def test_the_ratchet_reads_the_lowest_factor_also_when_the_rigid_body_rule_ran():
    """Decision 1 of the K1 stack: the projected factor is a fact of the gate entry; the ratchet and the gate read the measured value."""
    rep = verify_report(factor=0.13)
    rep["buckling"]["cases"][0]["rigid_body"] = {"applied": True, "status": "ok", "load_factors": [3.9]}
    f = rt.verify_facts(rep)
    assert f["cases"]["a"]["buckling_lowest_factor"] == 0.13
    assert f["sources"]["cases.a.buckling_lowest_factor"] == "verify.json:buckling.cases[a].load_factors[0]"
    plain = rt.verify_facts(verify_report(factor=8.0))
    assert plain["cases"]["a"]["buckling_lowest_factor"] == 8.0
    assert plain["sources"]["cases.a.buckling_lowest_factor"] == "verify.json:buckling.cases[a].load_factors[0]"
