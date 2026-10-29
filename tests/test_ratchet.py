"""Tests for the ratchet: the keep rule on plain dicts, the `acceptance` YAML block, the log and the runner with a fake judge
(CONTRACTS.md section 36). They need no gmsh and no solve.
"""
import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from ratchet_helpers import box_stl, fake_judge, stl_folder  # noqa: E402
from test_feedback import fake_run, make_case, tree_hash  # noqa: E402

from topopt import ratchet as rt  # noqa: E402
from topopt.config import UserError, load_config  # noqa: E402
from topopt.ratchet_sources import StlFolderSource, natural_key, parse_source  # noqa: E402
from topopt.runinfo import sha256_file  # noqa: E402
from topopt.runjob import JobHeld, run_job  # noqa: E402

CASES = ("a", "b")
LIMITS = {"a": (50.0, "load_cases[a].stress_limit (source: test)"), "b": (20.0, "load_cases[b].stress_limit (source: test)")}


def R(*rules, **kw):
    kw.setdefault("case_names", CASES)
    kw.setdefault("stress_limits", LIMITS)
    kw.setdefault("max_bodies", 1)
    kw.setdefault("max_bodies_source", "print.max_bodies (the YAML)")
    return rt.make_rules(list(rules), **kw)


def F(**kw):
    """The facts of a part that passes every rule; a test changes what it needs."""
    f = {"watertight": True, "open_edges": 0, "nonmanifold_edges": 0, "volume_mm3": 1000.0, "mass_g": 1.24, "bodies": 1,
         "loose_mm3": 0.0, "loose_lower_bound": False, "closed_voids": 0, "print_gate": "pass", "print_failed": [], "print_unknown": [],
         "mesh_size_mm": 1.5, "percentile": 99.9, "mirror_error_mm": 0.01,
         "cases": {"a": {"stiffness_N_per_mm": 100.0, "compliance_Nmm": 1.0, "max_displacement_mm": 0.1,
                         "p_by_volume_MPa": 10.0, "peak_qp_MPa": 12.0},
                   "b": {"stiffness_N_per_mm": 200.0, "compliance_Nmm": 0.5, "max_displacement_mm": 0.05,
                         "p_by_volume_MPa": 5.0, "peak_qp_MPa": 6.0}},
         "errors": {}, "sources": {}}
    f.update(kw)
    return f


def with_case(f, case, **kw):
    f = copy.deepcopy(f)
    f["cases"][case].update(kw)
    return f


def states(out):
    return {(c.rule, c.case): c.state for c in out.checks}


# --- T1: the keep rule on plain dicts --------------------------------------------------------------------------------

def test_a_parent_of_none_is_a_baseline_and_the_rules_against_a_parent_are_off():
    rules = R({"rule": "watertight"}, {"rule": "stiffness", "max_drop_pct": 5, "source": "test"}, {"rule": "mass", "must_fall": True},
              {"rule": "stress", "against_parent": "not_worse"})
    out = rt.keep(F(), None, rules)
    assert out.decision == "BASELINE" and out.failed == ()
    s = states(out)
    assert s[("watertight", None)] == "pass"
    assert s[("stiffness", "a")] == s[("mass", None)] == s[("stress", "a")] == "off"
    bad = rt.keep(F(watertight=False, open_edges=3), None, rules)
    assert bad.decision == "BASELINE" and bad.failed == ("watertight",)       # a fact; nothing stops


@pytest.mark.parametrize("facts,state", [(F(), "pass"), (F(watertight=False, open_edges=2), "fail")])
def test_watertight_pass_and_fail(facts, state):
    out = rt.keep(facts, F(), R({"rule": "watertight"}))
    assert states(out)[("watertight", None)] == state
    assert out.decision == ("ACCEPT" if state == "pass" else "REJECT")
    if state == "fail":
        assert "2 open edge" in out.checks[0].reason and out.failed == ("watertight",)


def test_bodies_limit_comes_from_the_print_limit_and_a_missing_limit_is_no_input():
    assert rt.keep(F(bodies=1), F(), R({"rule": "bodies"})).decision == "ACCEPT"
    out = rt.keep(F(bodies=2), F(), R({"rule": "bodies"}))
    assert out.decision == "REJECT" and out.checks[0].limit_source == "print.max_bodies (the YAML)"
    out = rt.keep(F(), F(), R({"rule": "bodies"}, max_bodies=None))
    assert states(out)[("bodies", None)] == "no_input" and out.decision == "REJECT"


def test_loose_island_absolute_and_relative_to_the_parent():
    spec = {"rule": "loose_island", "max_mm3": 10, "source": "test"}
    assert rt.keep(F(loose_mm3=9.0), F(loose_mm3=100.0), R(spec)).decision == "ACCEPT"
    assert rt.keep(F(loose_mm3=11.0), F(loose_mm3=0.0), R(spec)).decision == "REJECT"
    rel = dict(spec, relative_to_parent=True)
    assert rt.keep(F(loose_mm3=105.0), F(loose_mm3=100.0), R(rel)).decision == "ACCEPT"           # +5 against a cap of 10
    out = rt.keep(F(loose_mm3=115.0), F(loose_mm3=100.0), R(rel))
    assert out.decision == "REJECT" and "increase 15" in out.checks[0].reason
    assert states(rt.keep(F(loose_mm3=115.0), None, R(rel)))[("loose_island", None)] == "off"
    lower = rt.keep(F(loose_mm3=20.0, loose_lower_bound=True), F(), R(spec))
    assert "lower bound" in lower.checks[0].reason


def test_stress_absolute_with_a_margin_and_the_cases_filter():
    rule = {"rule": "stress"}
    assert rt.keep(F(), F(), R(rule)).decision == "ACCEPT"
    out = rt.keep(with_case(F(), "b", p_by_volume_MPa=21.0), F(), R(rule))
    assert out.decision == "REJECT" and states(out) == {("stress", "a"): "pass", ("stress", "b"): "fail"}
    assert out.checks[1].limit_source.startswith("load_cases[b].stress_limit")
    assert rt.keep(with_case(F(), "b", p_by_volume_MPa=21.0), F(), R(dict(rule, cases=("a",)))).decision == "ACCEPT"
    m = dict(rule, margin=0.5, margin_source="owner")                    # the effective limit of a is 25 MPa
    assert rt.keep(with_case(F(), "a", p_by_volume_MPa=24.0), F(), R(m)).decision == "ACCEPT"
    out = rt.keep(with_case(F(), "a", p_by_volume_MPa=26.0), F(), R(m))
    assert out.decision == "REJECT" and "margin 0.5 (source: owner)" in out.checks[0].limit_source


def test_stress_parent_forms_not_worse_and_cap_or_parent():
    over = with_case(with_case(F(), "a", p_by_volume_MPa=60.0), "b", p_by_volume_MPa=5.0)       # a is over its limit of 50
    parent = with_case(F(), "a", p_by_volume_MPa=70.0)
    # absolute: 60 > 50 fails, although it is better than the parent
    assert rt.keep(over, parent, R({"rule": "stress"})).decision == "REJECT"
    # cap_or_parent: 60 <= max(50, 70) passes
    out = rt.keep(over, parent, R({"rule": "stress", "against_parent": "cap_or_parent", "cases": ("a",)}))
    assert out.decision == "ACCEPT" and out.checks[0].form == "cap_or_parent"
    worse = with_case(F(), "a", p_by_volume_MPa=75.0)
    assert rt.keep(worse, parent, R({"rule": "stress", "against_parent": "cap_or_parent", "cases": ("a",)})).decision == "REJECT"
    # not_worse: no limit needed, the value must not rise
    nolimit = {"a": (None, "no stress limit in the YAML"), "b": LIMITS["b"]}
    nw = R({"rule": "stress", "against_parent": "not_worse", "cases": ("a",)}, stress_limits=nolimit)
    assert rt.keep(with_case(F(), "a", p_by_volume_MPa=70.0), parent, nw).decision == "ACCEPT"       # equal
    assert rt.keep(worse, parent, nw).decision == "REJECT"
    # a form that needs a limit and the case has none: no_input with the options
    out = rt.keep(F(), F(), R({"rule": "stress", "cases": ("a",)}, stress_limits=nolimit))
    assert out.checks[0].state == "no_input" and "against_parent: not_worse" in out.checks[0].reason
    # at the baseline: cap_or_parent is read against the limit alone, not_worse is off
    assert rt.keep(over, None, R({"rule": "stress", "against_parent": "cap_or_parent", "cases": ("a",)})).failed == ("stress",)


def test_stiffness_drop_in_percent_against_the_parent_at_the_same_mesh_size():
    rule = {"rule": "stiffness", "max_drop_pct": 5, "source": "owner brief"}
    parent = F()
    assert rt.keep(with_case(F(), "a", stiffness_N_per_mm=95.0), parent, R(rule)).decision == "ACCEPT"       # exactly 5 %
    out = rt.keep(with_case(F(), "a", stiffness_N_per_mm=94.0), parent, R(rule))
    assert out.decision == "REJECT" and "drop 6" in out.checks[0].reason and "owner brief" in out.checks[0].limit_source
    assert rt.keep(with_case(F(), "a", stiffness_N_per_mm=150.0), parent, R(rule)).decision == "ACCEPT"      # stiffer
    only_b = R(dict(rule, cases=("b",)))
    assert rt.keep(with_case(F(), "a", stiffness_N_per_mm=1.0), parent, only_b).decision == "ACCEPT"
    gone = with_case(F(), "a", stiffness_N_per_mm=None)
    assert rt.keep(gone, parent, R(rule)).checks[0].state == "no_input"
    assert rt.keep(F(), with_case(F(), "a", stiffness_N_per_mm=None), R(rule)).checks[0].state == "no_input"


def test_mass_must_fall_or_may_rise_by_a_percent():
    fall = R({"rule": "mass", "must_fall": True})
    assert rt.keep(F(mass_g=1.2), F(), fall).decision == "ACCEPT"
    assert rt.keep(F(mass_g=1.24), F(), fall).decision == "REJECT"                    # equal is not a fall
    rise = R({"rule": "mass", "max_rise_pct": 2, "source": "test"})
    assert rt.keep(F(mass_g=1.24 * 1.02), F(), rise).decision == "ACCEPT"
    out = rt.keep(F(mass_g=1.24 * 1.03), F(), rise)
    assert out.decision == "REJECT" and out.checks[0].form == "delta"
    assert rt.keep(F(mass_g=None), F(), fall).checks[0].state == "no_input"


def test_print_gate_rule_reads_the_strict_gate():
    assert rt.keep(F(), F(), R({"rule": "print_gate"})).decision == "ACCEPT"
    out = rt.keep(F(print_gate="fail", print_failed=["wall_thickness"]), F(), R({"rule": "print_gate"}))
    assert out.decision == "REJECT" and "wall_thickness" in out.checks[0].reason


def test_symmetry_rule_reads_the_mirror_error():
    rule = {"rule": "symmetry", "max_mm": 0.2, "source": "test"}
    assert rt.keep(F(mirror_error_mm=0.1), F(), R(rule, has_mirror=True)).decision == "ACCEPT"
    assert rt.keep(F(mirror_error_mm=0.3), F(), R(rule, has_mirror=True)).decision == "REJECT"
    f = F()
    del f["mirror_error_mm"]
    f["errors"] = {"mirror": {"exit_code": 2, "message": "boom"}}
    out = rt.keep(f, F(), R(rule, has_mirror=True))
    assert out.checks[0].state == "no_input" and "boom" in out.checks[0].reason


def test_a_rule_that_is_not_listed_makes_no_check_and_a_listed_rule_with_a_missing_fact_fails_with_the_reason():
    out = rt.keep(F(watertight=False, bodies=5, mass_g=1.25), F(), R({"rule": "mass", "max_rise_pct": 1000, "source": "t"}))
    assert [c.rule for c in out.checks] == ["mass"] and out.decision == "ACCEPT"       # watertight and bodies are off
    f = F()
    del f["bodies"]
    f["errors"] = {"printcheck": {"exit_code": 1, "message": "the STL has no faces"}}
    out = rt.keep(f, F(), R({"rule": "bodies"}))
    assert out.decision == "REJECT" and "printcheck failed (the STL has no faces)" in out.checks[0].reason


def test_every_check_that_ran_names_its_limit_source_and_its_input_source():
    rules = R({"rule": "watertight"}, {"rule": "bodies"}, {"rule": "stress"}, {"rule": "stiffness", "max_drop_pct": 5},
              {"rule": "mass", "must_fall": True}, {"rule": "loose_island", "max_mm3": 1}, {"rule": "print_gate"})
    out = rt.keep(F(mass_g=1.0), F(), rules)
    for c in out.checks:
        assert c.limit_source and c.input_source, c
    stiff = next(c for c in out.checks if c.rule == "stiffness")
    assert "source not recorded" in stiff.limit_source                                  # a number with no source text
    assert json.loads(json.dumps(out.to_dict()))["decision"] == "ACCEPT"


def test_two_mesh_sizes_in_one_comparison_are_a_code_error():
    parent = F(mesh_size_mm=1.5)
    for rule in ({"rule": "stiffness", "max_drop_pct": 5}, {"rule": "stress", "against_parent": "not_worse"}):
        with pytest.raises(rt.RatchetError, match="one mesh size"):
            rt.keep(F(mesh_size_mm=1.6), parent, R(rule))
    assert rt.keep(F(mesh_size_mm=1.6), parent, R({"rule": "mass", "must_fall": True}, )).decision == "REJECT"   # not a solved fact


def test_the_ratchet_modules_have_no_limit_and_no_part_name():
    """The limits come from the case YAML. A literal limit or a part name in the code would be a hidden default."""
    for path in sorted((REPO / "topopt").glob("ratchet*.py")):
        text = path.read_text()
        for number in (r"235", r"0\.98", r"400", r"0\.99", r"800"):
            assert not re.search(rf"(?<![\w.]){number}(?![\w.])", text), f"{path.name} has the number {number}"
        for word in ("lug", "hook", "ladle", "bracket", "T-shirt", "gravy"):
            assert not re.search(rf"\b{word}\b", text, re.I), f"{path.name} names a part: {word}"


# --- the acceptance block of the YAML ---------------------------------------------------------------------------------

def case_yaml(tmp_path, acceptance=None, *, second_case=True, mirror=None, stress=True, print_block=None):
    ypath = make_case(tmp_path / "in")
    doc = yaml.safe_load(ypath.read_text())
    if second_case:
        doc["load_cases"].append({"name": "up", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, 5.0]}]})
    if stress:
        doc["load_cases"][0].update(allowable_mpa=100.0, safety_factor=2.0, allowable_source="supplier sheet")
        if second_case:
            doc["load_cases"][1].update(stress_limit=30.0)
    if mirror:
        doc["mirror"] = mirror
    if print_block:
        doc["print"] = print_block
    if acceptance is not None:
        doc["acceptance"] = acceptance
    out = tmp_path / "in" / "bracket_acc.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


def test_the_yaml_block_is_read_strictly_and_a_yaml_without_it_loads_as_before(tmp_path):
    assert load_config(make_case(tmp_path / "plain")).acceptance is None
    acc = {"mesh_size_mm": 1.5, "percentile": 99, "owner_gate": True,
           "rules": [{"rule": "watertight"}, {"rule": "stiffness", "max_drop_pct": 5, "source": "owner brief", "cases": ["down"]},
                     {"rule": "mass", "must_fall": True}, {"rule": "stress", "margin": 0.9, "margin_source": "lead",
                                                           "against_parent": "cap_or_parent"}]}
    cfg = load_config(case_yaml(tmp_path, acc))
    a = cfg.acceptance
    assert (a.mesh_size_mm, a.percentile, a.owner_gate) == (1.5, 99.0, True)
    assert [r.rule for r in a.rules] == ["watertight", "stiffness", "mass", "stress"]
    stiff, stress = a.rules[1], a.rules[3]
    assert stiff.get("max_drop_pct") == 5.0 and stiff.get("cases") == ("down",) and stiff.source_of("max_drop_pct") == "owner brief"
    assert stress.source_of("margin") == "lead" and stress.get("against_parent") == "cap_or_parent"
    assert a.rules[2].source_of("max_rise_pct") == "source not recorded"


@pytest.mark.parametrize("block,message", [
    ({"rules": []}, "must be a list with one rule or more"),
    ({"rules": [{"rule": "nope"}]}, "not a rule (known rules: watertight"),
    ({"rules": [{"rule": "watertight", "limit": 3}]}, "unknown key 'limit'"),
    ({"rules": [{"rule": "stiffness"}]}, "missing key 'max_drop_pct'"),
    ({"rules": [{"rule": "stiffness", "max_drop_pct": -1}]}, "must be >= 0.0"),
    ({"rules": [{"rule": "stiffness", "max_drop_pct": 5, "cases": ["zzz"]}]}, "not a load case"),
    ({"rules": [{"rule": "watertight"}, {"rule": "watertight"}]}, "lists the rule 'watertight' again"),
    ({"rules": [{"rule": "mass"}]}, "exactly one of must_fall"),
    ({"rules": [{"rule": "mass", "must_fall": True, "max_rise_pct": 1}]}, "exactly one of must_fall"),
    ({"rules": [{"rule": "stress", "against_parent": "better"}]}, "must be absolute, not_worse, cap_or_parent"),
    ({"rules": [{"rule": "stress", "margin": 1.5}]}, "must be <= 1.0"),
    ({"rules": [{"rule": "stress", "margin_source": "x"}]}, "has no 'margin'"),
    ({"rules": [{"rule": "symmetry", "max_mm": 1}]}, "no top-level 'mirror'"),
    ({"rules": [{"rule": "watertight"}], "owner_gate": "yes"}, "must be true or false"),
    ({"rules": [{"rule": "watertight"}], "percentile": 0}, "must be > 0.0"),
    ({"rules": [{"rule": "watertight"}], "extra": 1}, "unknown key 'extra' in 'acceptance'"),
    ({}, "missing key 'rules'"),
])
def test_the_yaml_block_errors_name_the_key_and_the_file(tmp_path, block, message):
    with pytest.raises(UserError) as err:
        load_config(case_yaml(tmp_path, block))
    assert message in str(err.value) and "bracket_acc.yaml" in str(err.value)


def test_the_symmetry_rule_is_allowed_with_a_mirror_block(tmp_path):
    cfg = load_config(case_yaml(tmp_path, {"rules": [{"rule": "symmetry", "max_mm": 0.2, "source": "test"}]},
                                mirror=[{"normal": "y", "at_mm": 0.0}]))
    assert cfg.acceptance.rules[0].get("max_mm") == 0.2
    assert rt.rules_from_config(cfg).has_mirror


def test_rules_read_the_limits_of_the_yaml_with_their_sources(tmp_path):
    cfg = load_config(case_yaml(tmp_path, {"rules": [{"rule": "stress"}, {"rule": "bodies"}]},
                                print_block={"max_bodies": 2}))
    rules = rt.rules_from_config(cfg)
    lim_down, src_down = rules.stress_limits["down"]
    assert lim_down == 50.0 and "allowable_mpa / safety_factor" in src_down and "supplier sheet" in src_down
    assert rules.stress_limits["up"][0] == 30.0 and "source not recorded" in rules.stress_limits["up"][1]
    assert (rules.max_bodies, rules.max_bodies_source) == (2, "print.max_bodies (the YAML)")
    assumed = rt.rules_from_config(load_config(case_yaml(tmp_path / "b", {"rules": [{"rule": "bodies"}]})))
    assert assumed.max_bodies == 1 and "assumed" in assumed.max_bodies_source
    no_limit = rt.rules_from_config(load_config(case_yaml(tmp_path / "c", {"rules": [{"rule": "stress"}]}, stress=False)))
    assert no_limit.stress_limits["down"][0] is None
    with pytest.raises(UserError, match="no rules"):
        rt.rules_from_config(load_config(make_case(tmp_path / "d")))


def test_the_rules_hash_changes_when_a_limit_changes(tmp_path):
    a = rt.rules_from_config(load_config(case_yaml(tmp_path, {"rules": [{"rule": "stress"}]})))
    b = rt.rules_from_config(load_config(case_yaml(tmp_path / "x", {"rules": [{"rule": "stress"}]})))
    c = rt.rules_from_config(load_config(case_yaml(tmp_path / "y", {"rules": [{"rule": "stress", "margin": 0.9}]})))
    assert a.digest() == b.digest() != c.digest()


# --- the symmetry helper ----------------------------------------------------------------------------------------------

def test_mirror_error_is_small_for_a_symmetric_part_and_the_size_of_a_bump_for_an_asymmetric_one(tmp_path):
    base = trimesh.creation.box(extents=(20, 10, 6))
    sym = tmp_path / "sym.stl"
    base.export(str(sym))
    bump = trimesh.creation.box(extents=(4, 3, 3))
    bump.apply_translation((0, 6.5, 0))                       # y from 5 to 8: sticks out 3 mm at +y; the part is not symmetric
    both = trimesh.boolean.union([base, bump], engine="manifold")
    asym = tmp_path / "asym.stl"
    both.export(str(asym))
    plane = {"axis": "y", "at_mm": 0.0}
    assert rt.mirror_error_mm(sym, [plane]) < 1e-6
    err = rt.mirror_error_mm(asym, [plane])
    assert 2.9 < err < 3.1                                    # the bump reaches 3 mm beyond the face y = 5 on one side only
    assert rt.mirror_error_mm(asym, [{"axis": "x", "at_mm": 0.0}]) < 1e-6                     # symmetric about x
    assert rt.mirror_error_mm(asym, [plane, {"axis": 0, "at_mm": 0.0}]) == pytest.approx(err)  # the largest over the planes
    shifted = rt.mirror_error_mm(sym, [{"axis": "y", "at_mm": 1.0}])                          # a plane off the centre
    assert shifted == pytest.approx(2.0, abs=1e-6)


# --- the sources ------------------------------------------------------------------------------------------------------

def test_the_stl_folder_source_sorts_by_number_and_skips_what_the_log_holds(tmp_path):
    folder = stl_folder(tmp_path / "c", 11)
    src = StlFolderSource(folder)
    assert [p.name for p in src.files][:3] == ["part_1.stl", "part_2.stl", "part_3.stl"] and src.files[-1].name == "part_11.stl"
    assert natural_key("part_2.stl") < natural_key("part_10.stl")
    first = src.propose(tmp_path, tmp_path / "out", [])
    assert first.ref == "part_1.stl" and first.stl == folder / "part_1.stl"
    def row(p):
        return {"type": "candidate", "source": {"ref": p.name}, "inputs": {"candidate_sha256": sha256_file(p)}}
    hist = [row(folder / "part_1.stl"), row(folder / "part_2.stl")]
    assert src.propose(tmp_path, tmp_path / "out", hist).ref == "part_3.stl"
    everything = [row(p) for p in src.files]
    assert src.propose(tmp_path, tmp_path / "out", everything) is None


def test_parse_source_refuses_an_unknown_form(tmp_path):
    for bad in ("marks", "nope:x", "stl:", ""):
        with pytest.raises(UserError):
            parse_source(bad)
    with pytest.raises(UserError, match="does not exist"):
        parse_source(f"stl:{tmp_path / 'missing'}")
    (tmp_path / "empty").mkdir()
    with pytest.raises(UserError, match="no .stl file"):
        parse_source(f"stl:{tmp_path / 'empty'}")


# --- T2: the log and the runner with a fake judge --------------------------------------------------------------------

class Kill(BaseException):
    """A simulated kill of the process: it is not an `Exception`, so no `except Exception` takes it."""


def start_run(tmp_path):
    ypath = make_case(tmp_path / "in")
    return fake_run(tmp_path / "runs" / "a", ypath), ypath


def quiet(*a, **k):
    pass


def ctx_for(ypath=None, needs=("printcheck", "verify")):
    return rt.JudgeContext(yaml_path=ypath, needs=frozenset(needs))


def passing(mass, stiff=100.0, **kw):
    """Facts of a candidate that is closed, one body, lighter by `mass`, with this stiffness for the case `down`."""
    return F(mass_g=mass, cases={"down": {"stiffness_N_per_mm": stiff, "p_by_volume_MPa": 1.0}}, **kw)


RULES_MASS_STIFF = dict(case_names=("down",), stress_limits={"down": (50.0, "test")})


def mass_stiff_rules(**kw):
    return rt.make_rules([{"rule": "watertight"}, {"rule": "mass", "must_fall": True},
                          {"rule": "stiffness", "max_drop_pct": 5, "source": "test"}], **RULES_MASS_STIFF, **kw)


def run(tmp_path, table, n=3, *, root=None, rules=None, **kw):
    r, ypath = start_run(tmp_path) if not (tmp_path / "runs" / "a").exists() else (tmp_path / "runs" / "a", tmp_path / "in" / "bracket.yaml")
    folder = tmp_path / "cands"
    if not folder.exists():
        stl_folder(folder, n)
    kw.setdefault("judge", fake_judge(table))
    kw.setdefault("ctx", ctx_for(ypath))
    return rt.run_ratchet(r, StlFolderSource(folder), rules or mass_stiff_rules(), root=root or tmp_path / "root",
                          yaml_path=ypath, say=quiet, **kw)


def rows_of(root):
    return rt.read_log(root)[0]


def test_accepts_chain_through_acc_folders_and_rejects_leave_the_parent_and_the_start_run_unchanged(tmp_path):
    start, _ = start_run(tmp_path)
    before = tree_hash(start)
    table = {"base": passing(1.24), "c001": passing(1.20), "c002": passing(1.10), "c003": passing(1.30), "c004": passing(1.00, stiff=90.0)}
    res = run(tmp_path, table, n=4)
    assert res["exit_code"] == 0 and res["accepted"] == ["c001", "c002"] and res["rejected"] == ["c003", "c004"]
    assert res["parent"] == "acc_02"
    root = tmp_path / "root"
    rows = rows_of(root)
    assert [(r["type"], r["id"], r["decision"]) for r in rows] == [
        ("baseline", "c000", "BASELINE"), ("candidate", "c001", "ACCEPT"), ("candidate", "c002", "ACCEPT"),
        ("candidate", "c003", "REJECT"), ("candidate", "c004", "REJECT")]
    assert [r["parent"] for r in rows] == [None, "acc_00", "acc_01", "acc_02", "acc_02"]       # each against the current parent
    assert rows[3]["failed"] == ["mass"] and rows[4]["failed"] == ["stiffness"]
    assert [r["promoted"] for r in rows] == ["acc_00", "acc_01", "acc_02", None, None]
    assert tree_hash(start) == before                                                          # the start run is never written
    for name in ("acc_00", "acc_01", "acc_02"):                                                 # run-shaped, no metrics.json
        assert (root / name / "run.json").is_file() and (root / name / "result" / "final.stl").is_file()
        assert not (root / name / "result" / "metrics.json").exists()
        assert (root / name / "labels.bin").exists() == (name == "acc_00")                      # a copy of the start run; an stl source needs no labels
    assert (root / "acc_02" / "result" / "final.stl").read_bytes() == (tmp_path / "cands" / "part_2.stl").read_bytes()
    assert (root / "shadow_003").is_dir() and (root / "shadow_004").is_dir() and not (root / "shadow_001").exists()
    state = json.loads((root / "ratchet.json").read_text())
    assert state["parent"] == "acc_02" and state["state"] == "finished" and state["next_acc"] == 3
    shadow_run = json.loads((root / "acc_01" / "run.json").read_text())
    assert shadow_run["ratchet"]["id"] == "c001" and Path(shadow_run["yaml_path"]).is_absolute()
    assert (root / "job.DONE").is_file()


def test_the_record_holds_the_checks_the_inputs_and_the_judge_block(tmp_path):
    table = {"base": passing(1.24), "c001": dict(passing(1.20), tools={"verify": "exit 0"}, timing_s={"verify": 3.5},
                                                  peak_rss_bytes=123)}
    run(tmp_path, table, n=1)
    rows = rows_of(tmp_path / "root")
    r = rows[1]
    assert r["format"] == 1 and r["seq"] == 1 and r["at"].endswith("Z") and r["owner_approval"] == "not_required"
    assert r["source"]["kind"] == "stl" and r["source"]["ref"] == "part_1.stl"
    assert set(r["inputs"]) >= {"candidate_sha256", "parent_sha256", "yaml_sha256", "rules_sha256"}
    assert r["inputs"]["candidate_sha256"] != r["inputs"]["parent_sha256"] and r["errors"] is None
    assert {c["rule"] for c in r["checks"]} == {"watertight", "mass", "stiffness"}
    assert r["judge"]["tools"] == {"verify": "exit 0"} and r["judge"]["timing_s"]["verify"] == 3.5
    assert "total" in r["judge"]["timing_s"] and r["judge"]["peak_rss_bytes"] == 123
    assert "timing_s" not in r["facts"] and r["facts"]["mass_g"] == 1.20


def test_a_parent_number_above_99_is_chosen_by_integer_not_by_a_sort_of_names(tmp_path):
    rules = rt.make_rules([{"rule": "watertight"}])
    res = run(tmp_path, {"base": F(), **{f"c{i:03d}": F() for i in range(1, 102)}}, n=101, rules=rules)
    assert res["parent"] == "acc_101" and res["accepted"][-1] == "c101"
    rows = rows_of(tmp_path / "root")
    by_id = {r["id"]: r for r in rows}
    assert by_id["c100"]["parent"] == "acc_99" and by_id["c100"]["promoted"] == "acc_100"
    assert by_id["c101"]["parent"] == "acc_100"                       # a sort of the names would give acc_99
    assert sorted(["acc_99", "acc_100"])[-1] == "acc_99"              # the trap
    assert (tmp_path / "root" / "acc_101" / "run.json").is_file()


def test_a_kill_between_the_log_line_and_the_move_gives_one_promotion_after_the_restart(tmp_path, monkeypatch):
    table = {"base": passing(1.24), "c001": passing(1.20), "c002": passing(1.10)}
    real_move = rt._move_dir
    calls = []

    def dying_move(src, dst):
        calls.append(dst.name)
        raise Kill()
    monkeypatch.setattr(rt, "_move_dir", dying_move)
    with pytest.raises(Kill):
        run(tmp_path, table, n=2)
    root = tmp_path / "root"
    rows = rows_of(root)
    assert [r["id"] for r in rows] == ["c000", "c001"] and rows[1]["promoted"] == "acc_01"     # the log line is there
    assert not (root / "acc_01").exists() and (root / "shadow_001").is_dir()                  # the move is not
    assert (root / "job.FAILED").is_file() and not (root / "job.DONE").exists()               # a kill is no DONE
    monkeypatch.setattr(rt, "_move_dir", real_move)
    res = run(tmp_path, table, n=2)
    assert res["exit_code"] == 0 and res["parent"] == "acc_02"
    rows = rows_of(root)
    assert [r["id"] for r in rows] == ["c000", "c001", "c002"]                                 # no second line for c001
    assert (root / "acc_01" / "result" / "final.stl").is_file() and not (root / "shadow_001").exists()
    assert rows[2]["parent"] == "acc_01" and rows[2]["decision"] == "ACCEPT"                   # c002 was judged against acc_01
    assert calls == ["acc_01"]


def test_a_kill_during_the_judge_leaves_no_line_and_the_candidate_is_judged_again(tmp_path):
    table = {"base": passing(1.24), "c001": Kill()}
    with pytest.raises(Kill):
        run(tmp_path, table, n=2)
    root = tmp_path / "root"
    assert [r["id"] for r in rows_of(root)] == ["c000"] and (root / "shadow_001").is_dir()
    table["c001"] = passing(1.20)
    table["c002"] = passing(1.10)
    res = run(tmp_path, table, n=2)
    assert res["accepted"] == ["c001", "c002"] and [r["id"] for r in rows_of(root)] == ["c000", "c001", "c002"]


def test_a_partial_last_line_is_skipped_and_the_next_record_starts_on_its_own_line(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    rt.append_row(root, {"id": "c000", "type": "baseline", "seq": 0})
    rt.append_row(root, {"id": "c001", "type": "candidate", "seq": 1})
    path = root / "ratchet.jsonl"
    path.write_bytes(path.read_bytes() + b'{"id": "c002", "type": "cand')                      # a kill in the middle of a write
    rows, facts = rt.read_log(root)
    assert [r["id"] for r in rows] == ["c000", "c001"] and "partial last line" in facts[0] and "ratchet.jsonl" in facts[0]
    rt.append_row(root, {"id": "c003", "type": "candidate", "seq": 3})
    rows, facts = rt.read_log(root)
    assert [r["id"] for r in rows] == ["c000", "c001", "c003"] and "do not parse" in facts[0]
    assert all(line.endswith("}") for line in path.read_text().splitlines()[:2])
    assert rt.derive(rows).next_seq == 4


def test_a_second_runner_on_the_same_root_gets_job_held(tmp_path):
    table = {"base": passing(1.24), "c001": passing(1.20)}
    root = tmp_path / "root"

    def body(job):
        with pytest.raises(JobHeld):
            run(tmp_path, table, n=1, root=root)
        return 0
    run_job(root, body)
    assert not (root / LOG if False else root / "ratchet.jsonl").exists()                      # the second runner wrote nothing


def test_the_owner_gate_stops_before_the_promotion_and_approve_promotes(tmp_path):
    table = {"base": passing(1.24), "c001": passing(1.20), "c003": passing(1.10)}       # the id is c + the number of the record
    res = run(tmp_path, table, n=2, owner_gate=True)
    root = tmp_path / "root"
    assert res["exit_code"] == 3 and res["pending"] == ["c001"] and res["stopped"] == "owner_gate"
    rows = rows_of(root)
    assert rows[-1]["id"] == "c001" and rows[-1]["owner_approval"] == "pending" and rows[-1]["promoted"] is None
    assert not (root / "acc_01").exists() and (root / "shadow_001").is_dir() and (root / "job.DONE").is_file()   # a pause is no failure
    assert rt.status(root)["pending"] == ["c001"] and rt.status(root)["state"] == "waiting_for_owner"
    again = run(tmp_path, table, n=2, owner_gate=True)                                         # nothing is judged while one waits
    assert again["exit_code"] == 3 and len(rows_of(root)) == 2
    out = rt.approve(root, say=quiet)
    assert out == {"id": "c001", "owner_approval": "given", "promoted": "acc_01", "parent": "acc_01"}
    assert (root / "acc_01" / "result" / "final.stl").is_file() and not (root / "shadow_001").exists()
    assert not (root / "acc_01" / "labels.bin").exists()                  # as without the gate: an stl source needs no labels
    assert rows_of(root)[-1]["type"] == "approval" and rt.status(root)["pending"] == []
    with pytest.raises(UserError, match="no candidate waits"):
        rt.approve(root, say=quiet)
    res = run(tmp_path, table, n=2, owner_gate=True)                       # the approval used the number 2: goes on with c003
    rows = rows_of(root)
    assert rows[-1]["id"] == "c003" and rows[-1]["parent"] == "acc_01" and res["pending"] == ["c003"]


def test_approve_can_decline_and_it_checks_the_id(tmp_path):
    table = {"base": passing(1.24), "c001": passing(1.20), "c003": passing(1.10)}
    run(tmp_path, table, n=2, owner_gate=True)
    root = tmp_path / "root"
    with pytest.raises(UserError, match="does not wait"):
        rt.approve(root, "c009", say=quiet)
    out = rt.approve(root, decline=True, say=quiet)
    assert out["owner_approval"] == "declined" and out["promoted"] is None and not (root / "acc_01").exists()
    assert (root / "shadow_001").is_dir() and rt.status(root)["pending"] == []
    res = run(tmp_path, table, n=2, owner_gate=True)                                           # c003 is judged against acc_00
    assert rows_of(root)[-1]["parent"] == "acc_00" and res["pending"] == ["c003"]


def test_a_failing_judge_or_a_failed_tool_is_a_reject_record_and_not_a_crash(tmp_path):
    table = {"base": passing(1.24), "c001": RuntimeError("the judge broke"),
             "c002": dict(passing(1.10), errors={"verify": {"exit_code": 2, "message": "gmsh crashed"}})}
    del table["c002"]["cases"]
    res = run(tmp_path, table, n=2)
    assert res["exit_code"] == 0 and res["rejected"] == ["c001", "c002"]
    rows = rows_of(tmp_path / "root")
    assert rows[1]["errors"]["judge"]["message"].startswith("internal error: RuntimeError: the judge broke")
    assert rows[2]["errors"] == {"verify": {"exit_code": 2, "message": "gmsh crashed"}}
    stiff = next(c for c in rows[2]["checks"] if c["rule"] == "stiffness")
    assert stiff["state"] == "no_input" and "verify failed (gmsh crashed)" in stiff["reason"]
    assert (tmp_path / "root" / "job.DONE").is_file()


def test_a_source_with_no_change_is_a_reject_record_with_the_reason(tmp_path):
    class NoEffect:
        kind, needs_labels = "noop", False

        def __init__(self):
            self.i = 0

        def describe(self):
            return {"kind": "noop", "count": 2}

        def propose(self, parent_dir, out_dir, history):
            refs = {(r.get("source") or {}).get("ref") for r in history}
            for ref in ("m1", "m2"):
                if ref not in refs:
                    err = {"exit_code": 1, "message": "refused"} if ref == "m2" else None
                    return rt.Candidate("noop", ref, None, None, note=f"{ref} changed nothing", error=err)
            return None
    r, ypath = start_run(tmp_path)
    res = rt.run_ratchet(r, NoEffect(), mass_stiff_rules(), root=tmp_path / "root", judge=fake_judge({"base": passing(1.24)}),
                         ctx=ctx_for(ypath), yaml_path=ypath, say=quiet)
    rows = rows_of(tmp_path / "root")
    assert res["rejected"] == ["c001", "c002"] and rows[1]["failed"] == ["no_candidate"] and rows[1]["decision"] == "REJECT"
    assert rows[1]["source"]["note"] == "m1 changed nothing" and rows[1]["judge"] is None and rows[1]["checks"] == []
    assert rows[2]["errors"] == {"source": {"exit_code": 1, "message": "refused"}}
    assert res["parent"] == "acc_00"


def test_a_baseline_that_cannot_be_judged_is_exit_2_with_a_failed_marker_and_no_acc_folder(tmp_path):
    bad = {"base": {"errors": {"verify": {"exit_code": 2, "message": "gmsh is missing"}}}}
    res = run(tmp_path, bad, n=1)
    root = tmp_path / "root"
    assert res["exit_code"] == 2 and res["stopped"] == "baseline_failed" and res["accepted"] == []
    rows = rows_of(root)
    assert len(rows) == 1 and rows[0]["type"] == "baseline" and rows[0]["promoted"] is None and rows[0]["errors"]
    assert (root / "job.FAILED").is_file() and not (root / "job.DONE").exists()
    ok = {"base": passing(1.24), "c001": passing(1.20)}
    res = run(tmp_path, ok, n=1)                                                                # the baseline is judged again
    assert res["exit_code"] == 0 and [r["type"] for r in rows_of(root)] == ["baseline", "baseline", "candidate"]
    assert rows_of(root)[2]["parent"] == "acc_00"


def test_max_candidates_stops_and_the_next_call_goes_on_with_the_next_ones(tmp_path):
    table = {"base": passing(1.24), "c001": passing(1.20), "c002": passing(1.10), "c003": passing(1.00)}
    res = run(tmp_path, table, n=3, max_candidates=1)
    assert res["stopped"] == "max_candidates" and res["accepted"] == ["c001"]
    res = run(tmp_path, table, n=3, max_candidates=5)
    assert res["accepted"] == ["c002", "c003"] and res["stopped"] == "no more candidates"
    assert [r["source"]["ref"] for r in rows_of(tmp_path / "root")[1:]] == ["part_1.stl", "part_2.stl", "part_3.stl"]


def test_the_baseline_mesh_size_is_the_size_of_every_candidate(tmp_path):
    calls = []
    table = {"base": passing(1.24, mesh_size_mm=1.7547), "c001": passing(1.20, mesh_size_mm=1.7547),
             "c002": passing(1.10, mesh_size_mm=1.7547)}
    run(tmp_path, table, n=2, judge=fake_judge(table, calls=calls), ctx=rt.JudgeContext(needs=frozenset({"printcheck", "verify"})))
    assert calls == [("base", None, None), ("c001", "acc_00", 1.7547), ("c002", "acc_01", 1.7547)]
    rows = rows_of(tmp_path / "root")
    assert rows[0]["facts"]["mesh_size_mm"] == 1.7547 and json.loads((tmp_path / "root" / "ratchet.json").read_text())["mesh_size_mm"] == 1.7547


def test_a_root_of_another_ratchet_is_refused_unless_overwrite_moves_it_aside(tmp_path):
    table = {"base": passing(1.24), "c001": passing(1.20)}
    run(tmp_path, table, n=1)
    root = tmp_path / "root"
    other = rt.make_rules([{"rule": "watertight"}], **RULES_MASS_STIFF)
    with pytest.raises(UserError, match="rules .* differ"):
        run(tmp_path, table, n=1, rules=other)
    res = run(tmp_path, table, n=1, rules=other, overwrite=True)
    assert res["exit_code"] == 0 and (root / "earlier_01" / "ratchet.jsonl").is_file() and (root / "earlier_01" / "acc_01").is_dir()
    assert [r["type"] for r in rows_of(root)] == ["baseline", "candidate"]
    (tmp_path / "foreign").mkdir()
    (tmp_path / "foreign" / "notes.txt").write_text("x")
    with pytest.raises(UserError, match="other files"):
        run(tmp_path, table, n=1, root=tmp_path / "foreign")


def test_the_start_run_must_be_a_run_without_stock_and_the_root_must_be_outside_a_run_folder(tmp_path):
    start, ypath = start_run(tmp_path)
    (start / "result" / "nominal.stl").write_bytes((start / "result" / "final.stl").read_bytes())
    stl_folder(tmp_path / "cands", 1)
    kw = dict(judge=fake_judge({"base": passing(1.24)}), ctx=ctx_for(ypath), yaml_path=ypath, say=quiet)
    with pytest.raises(UserError, match="machining stock"):
        rt.run_ratchet(start, StlFolderSource(tmp_path / "cands"), mass_stiff_rules(), root=tmp_path / "root", **kw)
    (start / "result" / "nominal.stl").unlink()
    with pytest.raises(UserError, match="inside the run folder"):
        rt.run_ratchet(start, StlFolderSource(tmp_path / "cands"), mass_stiff_rules(), root=start / "ratchet", **kw)
    marks = tmp_path / "marks.json"
    marks.write_text(json.dumps({"version": 1, "units": "mm", "marks": []}))
    (start / "labels.bin").unlink()
    with pytest.raises(UserError, match="no labels.bin"):
        rt.run_ratchet(start, parse_source(f"marks:{marks}"), mass_stiff_rules(), root=tmp_path / "root", **kw)
    assert rt.run_ratchet(start, StlFolderSource(tmp_path / "cands"), mass_stiff_rules(), root=tmp_path / "root", **kw)["exit_code"] == 0


def test_status_reports_counts_the_last_records_and_the_reasons(tmp_path):
    table = {"base": passing(1.24), "c001": passing(1.20), "c002": passing(1.30)}
    run(tmp_path, table, n=2)
    st = rt.status(tmp_path / "root")
    assert st["counts"] == {"candidates": 2, "ACCEPT": 1, "REJECT": 1} and st["parent"] == "acc_01" and st["state"] == "finished"
    assert st["last"][-1]["failed"] == ["mass"] and "mass" in st["last"][-1]["why"][0]
    assert st["job"]["state"] == "done" and "1 accepted, 1 rejected" in st["summary"]
    with pytest.raises(UserError, match="does not exist"):
        rt.status(tmp_path / "nowhere")


# --- the weld retry of the surface preparation is a note of the row (C3c) ---------------------------------------------

def test_verify_facts_carry_a_note_only_when_the_input_was_welded():
    """A candidate that verify judged after it welded the short edges of the input has the same line as the verify text, in
    `notes`. It is a note, not a rule: no other fact changes, and a normal run has no `notes` key."""
    from topopt.verify.report import weld_line
    case = {"stiffness_N_per_mm": 10.0, "compliance_Nmm": 1.0, "max_displacement_mm": 0.1,
            "von_mises": {"percentile": 99.9, "percentile_by_volume_MPa": 5.0, "peak_qp_MPa": 8.0}}
    report = {"size_mm": 1.0, "cases": {"a": case}, "mesh": {"tets": 10}}
    plain = rt.verify_facts(report)
    assert "notes" not in plain and "notes" not in plain["sources"]
    weld = {"limit_mm": 0.05, "edges_welded": 890, "vertices_merged": 673, "vertices_dropped": 13, "max_vertex_move_mm": 0.0335,
            "volume_change_mm3": -0.0032, "genus_before": [2], "genus_after": [0], "euler_before": [-2], "euler_after": [2]}
    got = rt.verify_facts(dict(report, mesh={"tets": 10, "surface_weld_retry": weld}))
    assert got["notes"] == [weld_line(weld)] and "the input was welded" in got["notes"][0]
    assert got["sources"]["notes"] == "verify.json:mesh.surface_weld_retry"
    assert {k: v for k, v in got.items() if k not in ("notes", "sources")} == {k: v for k, v in plain.items() if k != "sources"}


def test_judge_facts_keeps_the_notes_of_printcheck_and_adds_the_weld_note(tmp_path, monkeypatch):
    """`notes` of printcheck (for example the loose volume that could not be added up) stay when verify adds its note."""
    from topopt import printcheck
    from topopt.verify import run as verify_run_module
    from topopt.verify.report import weld_line
    weld = {"limit_mm": 0.05, "edges_welded": 5, "vertices_merged": 4, "vertices_dropped": 0, "max_vertex_move_mm": 0.00001,
            "volume_change_mm3": -0.1, "genus_before": [1], "genus_after": [0], "euler_before": [0], "euler_after": [2]}
    case = {"stiffness_N_per_mm": 10.0, "compliance_Nmm": 1.0, "max_displacement_mm": 0.1,
            "von_mises": {"percentile": 99.9, "percentile_by_volume_MPa": 5.0, "peak_qp_MPa": 8.0}}
    report = {"size_mm": 1.0, "cases": {"a": case}, "mesh": {"surface_weld_retry": weld}, "exit_code": 0}
    monkeypatch.setattr(printcheck, "check", lambda *a, **k: {})
    monkeypatch.setattr(rt, "printcheck_facts", lambda rep, density: {"watertight": True, "loose_lower_bound": True})
    monkeypatch.setattr(rt, "_exact_loose", lambda stl, facts: facts.setdefault("notes", []).append("a note of printcheck"))
    monkeypatch.setattr(verify_run_module, "verify_run", lambda *a, **k: report)
    ctx = rt.JudgeContext(yaml_path=tmp_path / "x.yaml", needs=frozenset({"printcheck", "verify"}))
    facts = rt.judge_facts(tmp_path / "shadow", None, ctx)
    assert facts["notes"] == ["a note of printcheck", weld_line(weld)] and facts["errors"] == {}


# --- T3: the marks source and the real printcheck, with no solve -------------------------------------------------------

def marks_file(path, *marks, sha=None):
    from test_feedback import marks_doc
    Path(path).write_text(json.dumps(marks_doc(*marks, sha=sha)))
    return Path(path)


def test_marks_source_with_printcheck_rules_accepts_a_cut_and_rejects_a_mark_off_the_part(tmp_path):
    from test_apply_marks import FAST, make_run, sha, spur_part
    from test_feedback import mark
    from topopt import apply_marks as am
    run, ypath = make_run(tmp_path, spur_part())
    cfg = load_config(ypath)
    rules = rt.make_rules([{"rule": "watertight"}, {"rule": "bodies"}, {"rule": "mass", "must_fall": True}, {"rule": "print_gate"}],
                          case_names=("down",), max_bodies=1, max_bodies_source="print.max_bodies (the YAML)")
    mf = marks_file(tmp_path / "marks.json",
                    mark(1, "remove", [[12, 2.5, 10]], radius=2.5),               # the spur at +y
                    mark(2, "add", [[200, 200, 200]], radius=2.0),                # off the part: no effect
                    mark(3, "remove", [[12, -2.5, 10]], radius=2.5),              # the spur at -y, after the first promotion
                    sha=sha(run / "result" / "final.stl"))
    before = tree_hash(tmp_path / "in", run)
    src = parse_source(f"marks:{mf}", yaml_path=ypath)
    src.options = dict(FAST)
    root = tmp_path / "root"
    res = rt.run_ratchet(run, src, rules, root=root, yaml_path=ypath, say=quiet,
                         ctx=rt.JudgeContext.from_config(cfg, rules, json.loads((run / "run.json").read_text())))
    assert res["exit_code"] == 0 and res["accepted"] == ["c001", "c003"] and res["rejected"] == ["c002"], res
    assert tree_hash(tmp_path / "in", run) == before                              # the start run and the inputs: not written
    rows = rows_of(root)
    assert [r["decision"] for r in rows] == ["BASELINE", "ACCEPT", "REJECT", "ACCEPT"]
    assert rows[2]["failed"] == ["no_candidate"] and "outside the part" in rows[2]["source"]["note"]
    assert rows[1]["source"]["ref"] == "marks.json#1" and rows[1]["source"]["facts"]["marks_status"] == "applied"
    assert rows[1]["facts"]["mass_g"] < rows[0]["facts"]["mass_g"] and rows[1]["facts"]["watertight"] is True
    assert rows[3]["parent"] == "acc_01" and rows[3]["source"]["facts"]["source_sha256_stale"] is True    # the hash is of the start part
    assert rows[1]["source"]["facts"]["source_sha256_stale"] is False
    assert rows[3]["facts"]["mass_g"] < rows[1]["facts"]["mass_g"]
    assert rows[1]["judge"]["tools"] == {"printcheck": "exit 0"} and "verify" not in rows[1]["judge"]["timing_s"]
    for name in ("acc_00", "acc_01", "acc_02"):                                   # a marks source copies the labels: apply_marks reads them
        assert (root / name / "labels.bin").is_file() and not (root / name / "result" / "metrics.json").exists()
    # T3: an acc_NN folder is accepted by apply_marks as the parent run, and apply_marks does not write into it
    acc = root / "acc_02"
    acc_before = tree_hash(acc)
    rep = am.apply_marks(acc, marks_file(tmp_path / "m2.json", mark(1, "remove", [[12, 2.5, 10]], radius=2.5)),
                         out_dir=tmp_path / "again", options=FAST, yaml_path=ypath)
    assert rep["ok"] is True, rep
    assert tree_hash(acc) == acc_before


def test_run_history_lists_the_acc_folders_as_runs_with_a_missing_metrics_file(tmp_path):
    """The `acc_NN` folders look like finished runs without `result/metrics.json`: `run_history` lists them with that problem."""
    from topopt.feedback import run_history
    table = {"base": passing(1.24), "c001": passing(1.20)}
    run(tmp_path, table, n=1)
    got = run_history(tmp_path / "root")
    names = {r["dir_name"]: r for r in got["runs"]}
    assert {"acc_00", "acc_01"} <= set(names)
    assert any("metrics.json" in p for p in names["acc_01"]["problems"])


def test_the_ratchet_log_and_state_are_not_taken_for_runs_and_the_shadow_has_no_metrics(tmp_path):
    table = {"base": passing(1.24), "c001": passing(1.30)}
    run(tmp_path, table, n=1)
    root = tmp_path / "root"
    assert (root / "shadow_001" / "run.json").is_file() and not (root / "shadow_001" / "result" / "metrics.json").exists()
    info = json.loads((root / "shadow_001" / "run.json").read_text())["ratchet"]
    assert info["id"] == "c001" and info["parent"] == "acc_00"


# --- T7: the mesh size of the baseline reaches every verify -----------------------------------------------------------

def fake_verify_report(size, stiffness=100.0):
    return {"exit_code": 0, "size_mm": size,
            "cases": {"down": {"stiffness_N_per_mm": stiffness, "compliance_Nmm": 1.0, "max_displacement_mm": 0.1,
                               "von_mises": {"percentile": 99.9, "percentile_by_volume_MPa": 3.0, "peak_qp_MPa": 4.0}}}}


def test_the_real_judge_gives_the_baseline_mesh_size_to_every_verify_and_turns_the_load_sensitivity_off(tmp_path, monkeypatch):
    import topopt.verify.run as vr
    calls = []

    def fake_verify(shadow, **kw):
        calls.append((Path(shadow).name, kw))
        assert (Path(shadow) / "run.json").is_file() and (Path(shadow) / "result" / "final.stl").is_file()
        return fake_verify_report(kw.get("mesh_size") or 1.7547)
    monkeypatch.setattr(vr, "verify_run", fake_verify)
    r, ypath = start_run(tmp_path)
    stl_folder(tmp_path / "cands", 2)
    cfg = load_config(ypath)
    rules = rt.make_rules([{"rule": "watertight"}, {"rule": "stiffness", "max_drop_pct": 5, "source": "test"}],
                          case_names=("down",))
    ctx = rt.JudgeContext.from_config(cfg, rules, json.loads((r / "run.json").read_text()))
    res = rt.run_ratchet(r, StlFolderSource(tmp_path / "cands"), rules, root=tmp_path / "root", ctx=ctx, yaml_path=ypath, say=quiet)
    assert res["exit_code"] == 0 and res["accepted"] == ["c001", "c002"]
    assert len(calls) == 3                                                   # the baseline and two candidates
    assert "mesh_size" not in calls[0][1]                                    # the baseline: verify picks the size
    assert [c[1]["mesh_size"] for c in calls[1:]] == [1.7547, 1.7547]        # every candidate: the size of the baseline
    assert all(c[1]["load_sensitivity"] is False for c in calls)
    assert all(c[1]["buckling"] is False for c in calls)                    # not `buckling.check` of the YAML (B3)
    assert all(c[1]["bore_checks"] is False for c in calls)                 # not the `bore_checks` block of the YAML (B4)
    assert rows_of(tmp_path / "root")[1]["facts"]["mesh_size_mm"] == 1.7547


def test_the_percentile_and_the_mesh_size_of_the_yaml_go_to_every_verify(tmp_path, monkeypatch):
    import topopt.verify.run as vr
    calls = []
    monkeypatch.setattr(vr, "verify_run", lambda shadow, **kw: calls.append(kw) or fake_verify_report(kw["mesh_size"]))
    r, ypath = start_run(tmp_path)
    stl_folder(tmp_path / "cands", 1)
    rules = rt.make_rules([{"rule": "stiffness", "max_drop_pct": 5, "source": "test"}], case_names=("down",),
                          mesh_size_mm=2.0, percentile=99.0)
    ctx = rt.JudgeContext.from_config(load_config(ypath), rules, json.loads((r / "run.json").read_text()))
    rt.run_ratchet(r, StlFolderSource(tmp_path / "cands"), rules, root=tmp_path / "root", ctx=ctx, yaml_path=ypath, say=quiet)
    assert [(c["mesh_size"], c["percentile"]) for c in calls] == [(2.0, 99.0), (2.0, 99.0)]


def test_verify_is_not_called_when_no_rule_needs_it(tmp_path, monkeypatch):
    import topopt.verify.run as vr
    monkeypatch.setattr(vr, "verify_run", lambda *a, **k: pytest.fail("verify must not run"))
    r, ypath = start_run(tmp_path)
    stl_folder(tmp_path / "cands", 2)
    rules = rt.make_rules([{"rule": "watertight"}, {"rule": "mass", "must_fall": True}, {"rule": "bodies"}], case_names=("down",),
                          max_bodies=1, max_bodies_source="print.max_bodies (the YAML)")
    assert "verify" not in rules.needs()
    ctx = rt.JudgeContext.from_config(load_config(ypath), rules, json.loads((r / "run.json").read_text()))
    res = rt.run_ratchet(r, StlFolderSource(tmp_path / "cands"), rules, root=tmp_path / "root", ctx=ctx, yaml_path=ypath, say=quiet)
    assert res["exit_code"] == 0 and len(rows_of(tmp_path / "root")) == 3


def test_a_judge_that_solves_at_another_mesh_size_is_a_code_error_that_stops_the_run(tmp_path):
    table = {"base": passing(1.24, mesh_size_mm=1.5), "c001": passing(1.20, mesh_size_mm=1.9)}
    with pytest.raises(rt.RatchetError, match="mesh size"):
        run(tmp_path, table, n=1)
    root = tmp_path / "root"
    assert (root / "job.FAILED").is_file() and [r["id"] for r in rows_of(root)] == ["c000"]       # no record for a code error


def test_a_failing_verify_is_a_no_input_reject_with_the_message_of_the_error(tmp_path, monkeypatch):
    import topopt.verify.run as vr
    from topopt.verify.errors import VerifyError
    n = []

    def fake_verify(shadow, **kw):
        n.append(1)
        if len(n) > 1:
            raise VerifyError("gmsh could not mesh the part")
        return fake_verify_report(1.7)
    monkeypatch.setattr(vr, "verify_run", fake_verify)
    r, ypath = start_run(tmp_path)
    stl_folder(tmp_path / "cands", 1)
    rules = rt.make_rules([{"rule": "stiffness", "max_drop_pct": 5, "source": "test"}], case_names=("down",))
    ctx = rt.JudgeContext.from_config(load_config(ypath), rules, json.loads((r / "run.json").read_text()))
    res = rt.run_ratchet(r, StlFolderSource(tmp_path / "cands"), rules, root=tmp_path / "root", ctx=ctx, yaml_path=ypath, say=quiet)
    row = rows_of(tmp_path / "root")[1]
    assert res["rejected"] == ["c001"] and row["errors"]["verify"] == {"exit_code": 2, "message": "gmsh could not mesh the part"}
    assert row["checks"][0]["state"] == "no_input" and "gmsh could not mesh the part" in row["checks"][0]["reason"]


# --- the script commands ----------------------------------------------------------------------------------------------

PRINT_RULES = {"rules": [{"rule": "watertight"}, {"rule": "bodies"}, {"rule": "mass", "must_fall": True}]}


def cli_case(tmp_path, acceptance=None):
    ypath = case_yaml(tmp_path, PRINT_RULES if acceptance is None else acceptance, second_case=False, stress=False)
    return fake_run(tmp_path / "runs" / "a", ypath), ypath


def cli(capsys, *argv):
    code = rt.main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_check_run_status_approve_and_the_exit_codes(tmp_path, capsys):
    start, ypath = cli_case(tmp_path)
    stl_folder(tmp_path / "cands", 3)
    root = tmp_path / "root"
    code, out, err = cli(capsys, "check", start, "--source", f"stl:{tmp_path / 'cands'}", "--json")
    rep = json.loads(out)
    assert code == 0 and [r["rule"] for r in rep["rules"]] == ["watertight", "bodies", "mass"], out
    assert rep["estimate"]["candidates"] == 3 and rep["estimate"]["verify_calls"] == 0 and rep["ready"] is True
    assert not any(r["missing"] for r in rep["rules"]) and rep["tools"]["verify"] is None
    code, out, err = cli(capsys, "check", start)
    assert code == 0 and "watertight" in out and "mass" in out

    code, out, err = cli(capsys, "run", start, "--source", f"stl:{tmp_path / 'cands'}", "--root", root, "--stop-before-promote")
    assert code == 3 and "c001 ACCEPT" in out and "-> " not in out.split("c001 ACCEPT")[1].splitlines()[0]
    code, out, err = cli(capsys, "status", root)
    assert code == 0 and "waiting_for_owner" in out and "c001" in out
    code, out, err = cli(capsys, "status", root, "--json")
    assert json.loads(out)["pending"] == ["c001"]
    code, out, err = cli(capsys, "approve", root)
    assert code == 0 and "acc_01" in out and (root / "acc_01" / "result" / "final.stl").is_file()
    code, out, err = cli(capsys, "approve", root)
    assert code == 1 and "no candidate waits" in err
    code, out, err = cli(capsys, "run", start, "--source", f"stl:{tmp_path / 'cands'}", "--root", root, "--max-candidates", 5)
    assert code == 0 and rows_of(root)[-1]["id"] in ("c003", "c004")         # goes on; the later boxes are larger than acc_01: REJECT
    assert [r["decision"] for r in rows_of(root) if r["type"] == "candidate"] == ["ACCEPT", "REJECT", "REJECT"]
    code, out, err = cli(capsys, "status", root)
    assert code == 0 and "finished" in out and "1 accepted, 2 rejected" in out


def test_cli_exit_code_1_for_bad_input_and_2_for_a_baseline_that_cannot_be_judged(tmp_path, capsys):
    start, ypath = cli_case(tmp_path)
    stl_folder(tmp_path / "cands", 1)
    code, out, err = cli(capsys, "run", start, "--source", "marks", "--root", tmp_path / "r1")
    assert code == 1 and "not understood" in err
    code, out, err = cli(capsys, "run", tmp_path / "nowhere", "--source", f"stl:{tmp_path / 'cands'}", "--root", tmp_path / "r2")
    assert code == 1 and err.startswith("error:")
    plain, _ = start_run(tmp_path / "plain")                                  # a YAML with no acceptance block
    code, out, err = cli(capsys, "run", plain, "--source", f"stl:{tmp_path / 'cands'}", "--root", tmp_path / "r3")
    assert code == 1 and "no 'acceptance' block" in err and not (tmp_path / "r3").exists()
    code, out, err = cli(capsys, "status", tmp_path / "nowhere")
    assert code == 1 and "does not exist" in err
    (start / "result" / "final.stl").write_text("this is not an STL file")       # printcheck cannot read the start part
    code, out, err = cli(capsys, "run", start, "--source", f"stl:{tmp_path / 'cands'}", "--root", tmp_path / "r4")
    assert code == 2 and (tmp_path / "r4" / "job.FAILED").is_file() and rows_of(tmp_path / "r4")[0]["errors"]["printcheck"]


def test_cli_a_second_runner_gets_exit_code_1(tmp_path, capsys):
    start, ypath = cli_case(tmp_path)
    stl_folder(tmp_path / "cands", 1)
    root = tmp_path / "root"

    def body(job):
        code, out, err = cli(capsys, "run", start, "--source", f"stl:{tmp_path / 'cands'}", "--root", root)
        assert code == 1 and "Another run holds the job lock" in err and "--override-lock" in err, err
        return 0
    run_job(root, body)
    assert not (root / "ratchet.jsonl").exists()


def test_cli_judge_prints_the_baseline_and_the_candidate_outcome_and_writes_only_to_out(tmp_path, capsys):
    start, ypath = cli_case(tmp_path)
    stl_folder(tmp_path / "cands", 1)
    before = tree_hash(start)
    out_dir = tmp_path / "judged"
    code, out, err = cli(capsys, "judge", tmp_path / "cands" / "part_1.stl", "--parent", start, "--out", out_dir)
    assert code == 0 and "BASELINE" in out and "ACCEPT" in out and str(out_dir) in out
    assert tree_hash(start) == before
    assert (out_dir / "candidate" / "result" / "final.stl").is_file()


def test_the_owner_gate_with_a_marks_source_promotes_with_the_labels(tmp_path):
    from test_apply_marks import FAST, make_run, spur_part
    from test_feedback import mark
    run, ypath = make_run(tmp_path, spur_part())
    cfg = load_config(ypath)
    rules = rt.make_rules([{"rule": "watertight"}, {"rule": "mass", "must_fall": True}], case_names=("down",))
    mf = marks_file(tmp_path / "marks.json", mark(1, "remove", [[12, 2.5, 10]], radius=2.5))
    src = parse_source(f"marks:{mf}", yaml_path=ypath)
    src.options = dict(FAST)
    ctx = rt.JudgeContext.from_config(cfg, rules, json.loads((run / "run.json").read_text()))
    root = tmp_path / "root"
    res = rt.run_ratchet(run, src, rules, root=root, ctx=ctx, yaml_path=ypath, owner_gate=True, say=quiet)
    assert res["exit_code"] == 3 and res["pending"] == ["c001"] and not (root / "acc_01").exists()
    assert rt.approve(root, say=quiet)["promoted"] == "acc_01"
    assert (root / "acc_01" / "labels.bin").is_file()                     # apply_marks can take acc_01 as a parent


def test_a_caller_without_a_context_gets_the_density_and_the_planes_from_the_yaml_of_the_run(tmp_path):
    ypath = case_yaml(tmp_path, {"rules": [{"rule": "mass", "must_fall": True}]}, second_case=False, stress=False,
                      mirror=[{"normal": "y", "at_mm": 0.0}])
    start = fake_run(tmp_path / "runs" / "a", ypath)
    rules = rt.make_rules([{"rule": "mass", "must_fall": True}, {"rule": "symmetry", "max_mm": 1.0, "source": "test"}],
                          case_names=("down",), has_mirror=True)
    ctx = rt._default_context(start, rules, ypath, 1.7)
    assert ctx.density_g_cm3 == load_config(ypath).material.density and ctx.mesh_size_mm == 1.7
    assert ctx.mirror_planes and ctx.needs == frozenset({"printcheck", "mirror"})
    broken = rt._default_context(tmp_path / "nowhere", rules, None, None)               # no run.json: a small context, no crash
    assert broken.density_g_cm3 is None and broken.needs == frozenset({"printcheck", "mirror"})
