"""A load case that repeats another case or is its exact opposite (F4-5; `loads.repeated_load_cases`, CONTRACTS.md section 13).

The model is linear. The opposite load gives the same displacements with the other sign, so the compliance, the relaxed von Mises
stress and the displacement and alignment rows with the sense abs get the same value from both cases. The buckling row and the
displacement rows with the sense max or min follow the sign. The tools state this as a fact with options, one fact for each group
of cases that have one load: `topopt check` (text and `--json`), the first lines of `run.log`, the warnings and the field
`repeated_load_cases` of `check_data` (so the MCP tool `check_problem`), and the part builder. No tolerance: two loads are the same
when every number is equal as a float.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla
import yaml

from test_buckling_row import column_yaml, dense_state
from test_multirow_mma import evaluate
from test_optimize import box
from topopt import buckling_row as br
from topopt import loads
from topopt.cli import check_data, main
from topopt.config import UserError, load_config
from topopt.optimize import Model, formulation_from_config, problem_facts
from topopt.problem import build_problem
from topopt.regions_builder import build_spec, check_yaml

REPO = Path(__file__).resolve().parent.parent

DOWN = {"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}
UP = {"name": "up", "forces": [{"region": "tip", "force": [0, 0, 2.0]}]}


def write_case(tmp_path, cases, settings=None, **top):
    """The cantilever of test_optimize (design x 0..20, plate x -2..0, load box `tip` at the free end, load slab `side`
    on top) with the given load cases. `settings` and the keys of `top` (`limits`, `buckling`, ...) go into the YAML."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", (18, 4, 0), (20, 6, 3))
    box(tmp_path / "side.stl", (6, 0, 8), (12, 10, 10))
    d = {"name": "pair", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl", "side": "side.stl"}},
         "load_cases": cases,
         "settings": {"voxel_size": 2.0, "volume_fraction": 0.3, "iterations": 2, "beta_schedule": [[0, 1]], **(settings or {})},
         **top}
    p = tmp_path / "pair.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def pairs(tmp_path, cases, settings=None, **top):
    return loads.repeated_load_cases(load_config(write_case(tmp_path, cases, settings, **top)))


def text_of(tmp_path, cases, settings=None, **top):
    [line] = loads.repeated_load_lines(pairs(tmp_path, cases, settings, **top))
    return line


def with_(case, **keys):
    return dict(case, **keys)


# --- what is a repeat --------------------------------------------------------------------------------------------------


def test_an_exact_opposite_is_found_and_the_pair_is_json_safe(tmp_path):
    [p] = pairs(tmp_path, [with_(DOWN, weight=1.0), with_(UP, weight=2.0)])
    assert (p["cases"], p["relations"], p["weights"]) == (["down", "up"], ["opposite"], [1.0, 2.0])
    assert (p["objective"], p["objective_weight"], p["sign_rows"], p["limit_rows"]) == ("sum", 3.0, [], [])
    assert (p["remote_force"], p["buckling_row"]) == (False, False)
    assert json.loads(json.dumps(p)) == p                                           # plain numbers, strings, lists and None


def test_an_equal_repeat_is_found_and_has_no_row_that_follows_the_sign(tmp_path):
    [p] = pairs(tmp_path, [DOWN, with_(DOWN, name="again")], buckling={"min_factor": 10.0})
    assert (p["cases"], p["relations"], p["sign_rows"]) == (["down", "again"], ["equal"], [])    # a repeat has the sign of the first case
    assert "follow the sign" not in text_of(tmp_path / "t", [DOWN, with_(DOWN, name="again")], buckling={"min_factor": 10.0})


@pytest.mark.parametrize("other", [
    {"region": "tip", "force": [0, 0, 2.0000000000000004]},      # the last bit of one component: no tolerance
    {"region": "tip", "force": [1e-12, 0, 2.0]},                  # another component
    {"region": "tip", "force": [0, 0, 4.0]},                      # proportional with another factor: out of scope
    {"region": "tip", "force": [0, 0, 1.0]},
    {"region": "side", "force": [0, 0, 2.0]},                     # another load region
], ids=["last-bit", "component", "factor-2", "half", "region"])
def test_a_pair_that_differs_is_no_fact(tmp_path, other):
    cases = [DOWN, {"name": "up", "forces": [other]}]
    p = write_case(tmp_path, cases)
    cfg = load_config(p)
    assert loads.repeated_load_cases(cfg) == []
    assert not [f for f in problem_facts(cfg) if "load case" in f]
    d = check_data(p)
    assert d["repeated_load_cases"] == [] and not [w for w in d["warnings"] if "load case" in w]


def test_a_case_with_one_more_entry_is_no_repeat(tmp_path):
    more = {"name": "up", "forces": UP["forces"] + [{"region": "side", "force": [0, 0, 1.0]}]}
    assert pairs(tmp_path, [DOWN, more]) == []


def test_the_order_of_the_entries_does_not_matter_and_every_load_type_changes_sign(tmp_path):
    a = {"name": "a", "forces": [{"region": "tip", "force": [1.0, 2.0, -3.0]},
                                 {"region": "side", "force": [0.5, 0.0, 0.0], "at": [9.0, 5.0, 12.0]},
                                 {"region": "side", "force": [0.0, 0.0, 0.0], "moment": [10.0, 0.0, -4.0]}],
         "pressures": [{"region": "side", "value_mpa": 0.25}, {"region": "tip", "value_mpa": -0.5}]}
    b = {"name": "b", "forces": [{"region": "side", "force": [0.0, 0.0, 0.0], "moment": [-10.0, -0.0, 4.0]},
                                 {"region": "tip", "force": [-1.0, -2.0, 3.0]},
                                 {"region": "side", "force": [-0.5, -0.0, -0.0], "at": [9.0, 5.0, 12.0]}],
         "pressures": [{"region": "tip", "value_mpa": 0.5}, {"region": "side", "value_mpa": -0.25}]}
    [p] = pairs(tmp_path, [a, b])
    assert (p["cases"], p["relations"]) == (["a", "b"], ["opposite"])
    # each of these changes breaks the match
    for name, change in {
        "the point at moved": lambda c: c["forces"][2].update(at=[9.0, 5.0, 12.5]),
        "the moment kept its sign": lambda c: c["forces"][0].update(moment=[10.0, 0.0, -4.0]),
        "the pressure kept its sign": lambda c: c["pressures"][0].update(value_mpa=-0.5),
        "a remote force became a plain force": lambda c: c["forces"][2].pop("at"),
        "a pressure is missing": lambda c: c["pressures"].pop(),
    }.items():
        bad = json.loads(json.dumps(b))
        change(bad)
        assert pairs(tmp_path / re.sub(r"\W", "_", name), [a, bad]) == [], name


def test_three_cases_with_one_load_give_one_group_with_the_relation_of_each_case_to_the_first(tmp_path):
    cases = [DOWN, with_(UP, name="up"), with_(DOWN, name="down2"), {"name": "side", "forces": [{"region": "side", "force": [1.0, 0, 0]}]}]
    [g] = pairs(tmp_path, cases)
    assert (g["cases"], g["relations"]) == (["down", "up", "down2"], ["opposite", "equal"])
    assert len(g["weights"]) == len(g["compliance_limits"]) == len(g["stress_limits_mpa"]) == 3


def test_two_loads_give_two_groups_in_the_order_of_their_first_cases(tmp_path):
    side = {"name": "side", "forces": [{"region": "side", "force": [1.0, 0, 0]}]}
    cases = [DOWN, side, with_(side, name="side_back", forces=[{"region": "side", "force": [-1.0, 0, 0]}]), with_(UP, name="up")]
    found = pairs(tmp_path, cases)
    assert [(g["cases"], g["relations"]) for g in found] == [(["down", "up"], ["opposite"]), (["side", "side_back"], ["opposite"])]


def test_a_zero_load_matches_with_both_signs_and_is_called_equal(tmp_path):
    zero = {"region": "tip", "force": [0, 0, 0]}
    # the mass objective: under the compliance objective the loader refuses a YAML in which no case applies a load (U21)
    [p] = pairs(tmp_path, [{"name": "z1", "forces": [zero]}, {"name": "z2", "forces": [zero]}], settings={"objective": "mass"})
    assert p["relations"] == ["equal"]


def test_the_shipped_examples_have_no_repeated_case():
    yamls = sorted((REPO / "examples").glob("*/*.yaml"))
    assert yamls
    for y in yamls:
        if "spec" in y.name:
            continue                                    # a part spec for build_regions, not a problem YAML
        assert loads.repeated_load_cases(load_config(y)) == [], y.name


# --- the text -----------------------------------------------------------------------------------------------------------


def sentences(text):
    return [x for x in re.split(r"(?<=[.;:]) ", text) if x]


def test_an_opposite_pair_says_why_what_differs_the_cost_and_the_options(tmp_path):
    cases = [with_(DOWN, weight=1.0, compliance_limit=5.0), with_(UP, weight=2.0, compliance_limit=4.0)]
    t = text_of(tmp_path, cases)
    assert t.startswith("fact: load case 'up' is the exact opposite of 'down': the same load regions, and each force, moment and "
                        "pressure has the other sign.")                                      # no remote force: no clause about `at`
    assert "The model is linear, so the opposite load gives the same displacements with the other sign." in t        # why
    assert "the von Mises stress" in t and "springs and joints" in t and "sense abs" in t                               # the same rows
    assert "No row of this YAML follows the sign." in t                                                               # what differs: none here
    assert "The weights add in the sum objective, so the pair acts like one case with the weight 1 + 2 = 3." in t       # settings
    assert "The compliance limits are 5 N*mm and 4 N*mm. The lower one, 4 N*mm, is the one that counts." in t
    assert "Cost: 'up' adds one load column to the solve of every pass, and one adjoint solve for each stress, displacement, " \
           "alignment or buckling row that it has." in t
    assert t.endswith("options: keep one of the cases 'down', 'up' and remove the other; give the kept case the weight 3 and the "
                      "compliance_limit 4 N*mm. If the pair is a mistake, change one of the loads.")
    assert max(len(x.split()) for x in sentences(t)) <= 30                                                              # short sentences


def test_an_equal_repeat_says_that_only_the_settings_can_differ(tmp_path):
    t = text_of(tmp_path, [with_(DOWN, weight=3.0), with_(DOWN, name="again", weight=1.0)])
    assert t.startswith("fact: load case 'again' repeats 'down': the same load regions, and the same forces, moments and pressures.")
    assert "every row has the same value in both. Only the settings of the cases (the weight and the limits) can differ." in t
    assert "the weight 3 + 1 = 4." in t
    assert "options: keep one of the cases 'down', 'again' and remove the other; give the kept case the weight 4." in t
    assert "reactions" not in t                                                       # the reactions of a repeat have the same sign


def test_the_options_do_not_choose_the_case_by_its_order(tmp_path):
    """F11: the later case can be the real load. The text names no case to remove and none to keep."""
    t = text_of(tmp_path, [with_(DOWN, weight=5.0), with_(UP, weight=234.0)])
    assert "options: keep one of the cases 'down', 'up' and remove the other; give the kept case the weight 239." in t
    assert "remove 'up'" not in t and "remove 'down'" not in t and "give 'down'" not in t


def test_the_reactions_have_the_other_sign_but_the_compliance_and_the_stress_do_not(tmp_path):
    t = text_of(tmp_path, [DOWN, UP])
    assert "The springs and joints are part of K, so they do not change this." in t
    assert "The reactions of the springs, the joints and the clamps have the other sign; the compliance and the stress do not." in t


def test_the_remote_force_clause_shows_only_if_a_force_has_a_point(tmp_path):
    """F5: the clause about `at` is for a pair with a remote force."""
    remote = {"region": "side", "force": [0.5, 0.0, 0.0], "at": [9.0, 5.0, 12.0]}
    flip = dict(remote, force=[-0.5, -0.0, -0.0])
    [g] = pairs(tmp_path / "a", [{"name": "a", "forces": [remote]}, {"name": "b", "forces": [flip]}])
    assert g["remote_force"] is True
    t = text_of(tmp_path / "b", [{"name": "a", "forces": [remote]}, {"name": "b", "forces": [flip]}])
    assert "has the other sign (a remote force keeps its point `at`)." in t
    plain = text_of(tmp_path / "c", [DOWN, UP])
    assert "remote force" not in plain and "`at`" not in plain


def test_the_rows_that_follow_the_sign_are_named_from_the_yaml(tmp_path):
    third = {"name": "third", "forces": [{"region": "side", "force": [1.0, 0, 0]}]}
    both = [DOWN, UP]
    lim = {"displacement": [{"name": "zmax", "region": "tip", "limit_mm": 0.05, "direction": [0, 0, 1], "sense": "max"}]}
    t = text_of(tmp_path / "a", both, limits=lim)
    assert "the displacement limit 'zmax' (sense max), because it reads the sign of the motion" in t and "No row of this" not in t
    assert "options: keep both cases, because the rows above follow the sign. Or, if you do not need these rows, keep one of " \
           "the cases 'down', 'up' and remove the other; give the kept case the weight 2." in t
    t = text_of(tmp_path / "b", both, buckling={"min_factor": 10.0})
    assert "the buckling row (buckling.min_factor), because compression in one case is tension in the other" in t
    assert "a case with no compression has no buckling load" in t
    t = text_of(tmp_path / "c", both, buckling={"check": True})
    assert "the buckling check after the run (buckling.check)" in t
    # the sense abs (the default) and a limit on other cases do not follow the sign
    abs_lim = {"displacement": [{"name": "norm", "region": "tip", "limit_mm": 0.05}]}
    assert "No row of this YAML follows the sign" in text_of(tmp_path / "d", both, limits=abs_lim)
    other = {"displacement": [dict(lim["displacement"][0], cases=["third"])]}
    assert "No row of this YAML follows the sign" in text_of(tmp_path / "e", both + [third], limits=other)
    both_cases = {"displacement": [dict(lim["displacement"][0], sense="min", cases=["up", "down"])]}
    assert "the displacement limit 'zmax' (sense min), because it reads the sign of the motion" in text_of(tmp_path / "f", both + [third], limits=both_cases)


@pytest.mark.parametrize("named", [["down"], ["up"]], ids=["first-case", "second-case"])
@pytest.mark.parametrize("sense", ["max", "min"])
def test_a_signed_limit_that_lists_only_one_case_of_the_pair_is_no_sign_row(tmp_path, named, sense):
    """F1: the pair gives such a row nothing new. The case that the row lists is the one to keep, and the text says so in the
    rows it names (the options do not choose by order, so a row on the second case is the same as a row on the first)."""
    third = {"name": "third", "forces": [{"region": "side", "force": [1.0, 0, 0]}]}
    lim = {"displacement": [{"name": "zmax", "region": "tip", "limit_mm": 0.05, "direction": [0, 0, 1], "sense": sense, "cases": named}]}
    [g] = pairs(tmp_path, [DOWN, UP, third], limits=lim)
    assert g["sign_rows"] == []
    assert g["limit_rows"] == [{"name": "zmax", "kind": "displacement", "sense": sense, "cases": named}]
    t = loads.repeated_load_lines([g])[0]
    assert "No row of this YAML follows the sign." in t and "keep both" not in t and "because the rows above follow the sign" not in t
    assert f"the displacement limit 'zmax' (sense {sense}) lists '{named[0]}'" in t


def test_the_effect_of_the_weights_follows_the_objective(tmp_path):
    cases = [with_(DOWN, weight=3.0), with_(UP, weight=5.0)]
    [mx] = pairs(tmp_path / "m", cases, {"case_combine": "max"})
    assert (mx["objective"], mx["objective_weight"]) == ("max", 5.0)
    t = loads.repeated_load_lines([mx])[0]
    assert "The max objective takes the larger weight, so the pair acts like one case with the weight 5." in t
    assert "options: keep one of the cases 'down', 'up' and remove the other; give the kept case the weight 5." in t
    [ms] = pairs(tmp_path / "s", cases, {"objective": "mass"})
    assert (ms["objective"], ms["objective_weight"]) == ("mass", None)
    t = loads.repeated_load_lines([ms])[0]
    assert "weight" not in t                                                                         # the mass objective has no weights
    assert t.endswith("options: keep one of the cases 'down', 'up' and remove the other. If the pair is a mistake, change one of the loads.")


def test_the_limits_in_the_text(tmp_path):
    base = dict(settings={"objective": "mass"})
    same = text_of(tmp_path / "a", [with_(DOWN, compliance_limit=5.0), with_(UP, compliance_limit=5.0)], **base)
    assert "Both cases have the compliance limit 5 N*mm, so the two rows are copies of each other." in same
    one = text_of(tmp_path / "b", [DOWN, with_(UP, stress_limit=7.5)], **base)
    assert "Only 'up' has a stress limit (7.5 MPa). It holds for both cases, because they have the same compliance and the same stress." in one
    assert one.endswith("options: keep one of the cases 'down', 'up' and remove the other; give the kept case the stress limit 7.5 MPa. "
                        "If the pair is a mistake, change one of the loads.")
    two = text_of(tmp_path / "c", [with_(DOWN, allowable_mpa=40.0, safety_factor=8.0), with_(UP, stress_limit=3.0)], **base)
    assert "The stress limits are 5 MPa and 3 MPa. The lower one, 3 MPa, is the one that counts." in two


def test_the_numbers_in_the_text_are_the_exact_numbers_of_the_yaml(tmp_path):
    """F8: a value that the user copies from the text equals the YAML value (no rounding to 6 digits)."""
    cases = [with_(DOWN, weight=0.1234567, compliance_limit=0.1234567891), with_(UP, weight=0.2, compliance_limit=7.0)]
    [g] = pairs(tmp_path, cases)
    t = loads.repeated_load_lines([g])[0]
    assert f"with the weight 0.1234567 + 0.2 = {g['objective_weight']!r}." in t
    assert "0.1234567891 N*mm and 7 N*mm" in t and "The lower one, 0.1234567891 N*mm" in t
    weight, limit = re.search(r"give the kept case the weight (\S+) and the compliance_limit (\S+) N\*mm\.", t).groups()
    assert (float(weight), float(limit)) == (0.1234567 + 0.2, 0.1234567891)


def test_a_buckling_row_adds_its_eigen_solve_to_the_cost(tmp_path):
    """F3: the buckling row does one eigen solve for each case and pass, not only an adjoint solve."""
    without = text_of(tmp_path / "a", [DOWN, UP])
    assert "eigen" not in without
    with_row = text_of(tmp_path / "b", [DOWN, UP], buckling={"min_factor": 10.0})
    assert "Cost: 'up' adds one load column to the solve of every pass, and one adjoint solve for each stress, displacement, " \
           "alignment or buckling row that it has. The buckling row also does one eigen solve for each case in every pass, " \
           "so 'up' adds that cost too." in with_row
    check_only = text_of(tmp_path / "c", [DOWN, UP], buckling={"check": True})
    assert "eigen" not in check_only                                                     # the check runs once after the run, not in a pass
    equal = text_of(tmp_path / "d", [DOWN, with_(DOWN, name="again")], buckling={"min_factor": 10.0})
    assert "so 'again' adds that cost too." in equal


def test_the_limits_rows_that_list_a_case_are_named_in_the_options(tmp_path):
    """F4: the loader stops on a `cases` list that names a removed case, so the options name the rows to change."""
    third = {"name": "third", "forces": [{"region": "side", "force": [1.0, 0, 0]}]}
    lim = {"displacement": [{"name": "norm", "region": "tip", "limit_mm": 0.05, "cases": ["up"]},
                            {"name": "other", "region": "tip", "limit_mm": 0.05, "cases": ["third"]},
                            {"name": "whole", "region": "tip", "limit_mm": 0.05}],
           "alignment": [{"name": "al", "region_a": "tip", "region_b": "side", "axis": [0, 0, 1], "shift_limit_mm": 0.05, "cases": ["down", "up"]}]}
    [g] = pairs(tmp_path, [DOWN, UP, third], limits=lim)
    assert [(r["name"], r["kind"], r["cases"]) for r in g["limit_rows"]] == [("norm", "displacement", ["up"]), ("al", "alignment", ["down", "up"])]
    t = loads.repeated_load_lines([g])[0]
    assert "The `limits` rows that list cases: the displacement limit 'norm' lists 'up'; the alignment limit 'al' lists 'down' and 'up'." in t
    assert "A `cases` list must not name a case that you remove, or the loader stops: name the kept case in it." in t
    assert "swap max and min" not in t                                                  # no row with a sign
    assert "'other'" not in t and "'whole'" not in t                                    # a row on other cases, and a row with no list
    # the loader stops on the removed case: this is why the text names the rows
    d = yaml.safe_load(write_case(tmp_path / "x", [DOWN, UP, third], limits=lim).read_text())
    d["load_cases"] = [c for c in d["load_cases"] if c["name"] != "up"]
    q = tmp_path / "x" / "removed.yaml"
    q.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="not a load case"):
        load_config(q)
    # a row with a sign: the text also says to swap max and min
    signed = {"displacement": [{"name": "zmax", "region": "tip", "limit_mm": 0.05, "direction": [0, 0, 1], "sense": "max", "cases": ["up"]}]}
    t = text_of(tmp_path / "s", [DOWN, UP, third], limits=signed)
    assert "the displacement limit 'zmax' (sense max) lists 'up'" in t
    assert "If the kept case has the opposite load of the case that a max or min row lists, swap max and min." in t


# --- a group of three or more cases ---------------------------------------------------------------------------------


def test_a_group_of_three_gives_one_fact_with_the_summed_weight(tmp_path):
    """F2: A, B = -A and C = -A with the weights 1, 2 and 4 act like one case with the weight 7."""
    cases = [with_(DOWN, weight=1.0), with_(UP, weight=2.0), with_(UP, name="up2", weight=4.0)]
    [g] = pairs(tmp_path, cases)
    assert (g["cases"], g["relations"], g["weights"], g["objective_weight"]) == (["down", "up", "up2"], ["opposite", "opposite"], [1.0, 2.0, 4.0], 7.0)
    t = loads.repeated_load_lines([g])[0]
    assert t.startswith("fact: load cases 'down', 'up' and 'up2' have one load: 'up' and 'up2' are the exact opposite of 'down'.")
    assert "the group acts like one case with the weight 1 + 2 + 4 = 7." in t
    assert "options: keep one of the cases 'down', 'up', 'up2' and remove the others; give the kept case the weight 7." in t
    assert "weight 3" not in t and "weight 5" not in t and t.count("options:") == 1
    assert "Cost: each of 'up' and 'up2' adds one load column" in t
    assert "So 'up' and 'up2' give the optimizer no load that 'down' does not give." in t
    assert "If the group is a mistake, change one of the loads." in t


def test_a_group_with_an_equal_and_an_opposite_case_says_each_relation(tmp_path):
    cases = [with_(DOWN, weight=1.0), with_(UP, weight=2.0), with_(DOWN, name="again", weight=4.0)]
    [g] = pairs(tmp_path, cases)
    assert (g["cases"], g["relations"]) == (["down", "up", "again"], ["opposite", "equal"])
    t = loads.repeated_load_lines([g])[0]
    assert "have one load: 'up' is the exact opposite of 'down', and 'again' repeats 'down'." in t
    assert "The load regions are the same, and each force, moment and pressure is equal or has the other sign." in t
    assert "So 'up' and 'again' give the optimizer no load that 'down' does not give." in t


def test_a_group_of_equal_cases_has_no_sign_rows_even_with_a_buckling_row(tmp_path):
    [g] = pairs(tmp_path, [DOWN, with_(DOWN, name="b"), with_(DOWN, name="c")], buckling={"min_factor": 10.0})
    assert g["relations"] == ["equal", "equal"] and g["sign_rows"] == []
    t = loads.repeated_load_lines([g])[0]
    assert "load cases 'down', 'b' and 'c' have one load: 'b' and 'c' repeat 'down'." in t and "follow the sign" not in t
    assert "The load regions are the same, and each force, moment and pressure is equal." in t


def test_a_group_takes_the_largest_weight_and_the_lowest_limit(tmp_path):
    cases = [with_(DOWN, weight=1.0, compliance_limit=6.0), with_(UP, weight=9.0, compliance_limit=4.0),
             with_(UP, name="up2", weight=3.0)]
    [mx] = pairs(tmp_path / "m", cases, {"case_combine": "max"})
    assert (mx["objective"], mx["objective_weight"]) == ("max", 9.0)
    t = loads.repeated_load_lines([mx])[0]
    assert "The max objective takes the largest weight, so the group acts like one case with the weight 9." in t
    assert "The compliance limits are 6 N*mm ('down') and 4 N*mm ('up'). The lower one, 4 N*mm, is the one that counts." in t
    [sm] = pairs(tmp_path / "s", [with_(c, compliance_limit=lim) for c, lim in zip(cases, (6.0, 4.0, 5.0))])
    ts = loads.repeated_load_lines([sm])[0]
    assert "The compliance limits are 6 N*mm, 4 N*mm and 5 N*mm. The lowest one, 4 N*mm, is the one that counts." in ts
    assert "give the kept case the weight 13 and the compliance_limit 4 N*mm." in ts
    equal = text_of(tmp_path / "e", [with_(c, compliance_limit=5.0) for c in (DOWN, UP, with_(UP, name="up2"))])
    assert "All the cases have the compliance limit 5 N*mm, so the rows are copies of each other." in equal


# --- one text in every tool --------------------------------------------------------------------------------------------


def test_problem_facts_check_data_and_check_json_give_one_text(tmp_path, capsys):
    p = write_case(tmp_path, [with_(DOWN, weight=1.0), with_(UP, weight=2.0)])
    [fact] = [f for f in problem_facts(load_config(p)) if "load case" in f]
    d = check_data(p)
    assert [w for w in d["warnings"] if "load case" in w] == [fact]                    # the warnings (so the MCP tool) have the same text
    assert d["repeated_load_cases"] == loads.repeated_load_cases(load_config(p))
    json.dumps(d["repeated_load_cases"])
    assert main(["check", str(p), "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert [f for f in j["facts"] if "load case" in f] == [fact] and j["repeated_load_cases"] == d["repeated_load_cases"]


def test_topopt_check_prints_the_fact_once(tmp_path, capsys):
    p = write_case(tmp_path, [DOWN, UP])
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count("fact: load case 'up' is the exact opposite of 'down'") == 1
    assert out.count("options: keep one of the cases 'down', 'up'") == 1
    assert main(["check", str(write_case(tmp_path / "n", [DOWN, with_(UP, name="side_push", forces=[{"region": "tip", "force": [1, 0, 0]}])]))]) == 0
    assert "load case" not in capsys.readouterr().out


def test_a_group_of_three_gives_one_line_in_every_tool(tmp_path, capsys):
    cases = [with_(DOWN, weight=1.0), with_(UP, weight=2.0), with_(UP, name="up2", weight=4.0)]
    p = write_case(tmp_path, cases)
    [fact] = [f for f in problem_facts(load_config(p)) if "load case" in f]
    d = check_data(p)
    assert [w for w in d["warnings"] if "load case" in w] == [fact] and len(d["repeated_load_cases"]) == 1
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count("fact: load cases 'down', 'up' and 'up2'") == 1 and out.count("options:") == 1


def test_the_run_log_has_the_fact_before_the_first_pass(tmp_path):
    p = write_case(tmp_path, [DOWN, UP])
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text()
    [fact] = [f for f in problem_facts(load_config(p)) if "load case" in f]
    assert log.count(fact) == 1
    assert log.index(fact) < log.index("iter")                                         # the facts come before the first pass


def test_the_mcp_tool_check_problem_has_the_field_and_the_warning(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    res = server.check_problem(str(write_case(tmp_path, [DOWN, UP])))
    assert res["ok"] and [(r["cases"], r["relations"]) for r in res["repeated_load_cases"]] == [(["down", "up"], ["opposite"])]
    assert len([w for w in res["warnings"] if w.startswith("fact: load case 'up' is the exact opposite of 'down'")]) == 1
    from topopt_mcp import replies      # P9-11: the line that explains the keys of the reply moved to the guide topopt://replies/check_problem
    assert "topopt://replies/check_problem" in server.check_problem.__doc__ and "Key repeated_load_cases:" in replies.guide("check_problem")


SPEC = {
    "name": "pair_part",
    "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}]},
        "load": {"tip": [{"box": {"min": [18, 4, 0], "max": [20, 6, 4]}}]},
    },
    "load_cases": [{"name": "down", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -2.0]}]},
                   {"name": "up", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, 2.0]}]}],
    "settings": {"volume_fraction": 0.3, "voxel_size": 2.0, "iterations": 2},
}


def test_the_part_builder_prints_the_same_text_as_check(tmp_path):
    lines = []
    r = build_spec(SPEC, tmp_path / "o", say=lambda *a: lines.append(" ".join(map(str, a))))
    assert r["ok"]
    mine = [ln for ln in lines if "load case" in ln]
    assert len(mine) == 1 and mine[0].startswith("fact: load case 'up' is the exact opposite of 'down'")      # a fact: no WARNING prefix
    assert mine[0] in check_data(r["yaml"])["warnings"]                                                      # one text in both tools
    lines2 = []
    check_yaml(r["yaml"], say=lambda *a: lines2.append(" ".join(map(str, a))))
    assert mine[0] in lines2


# --- the claims of the text are true -----------------------------------------------------------------------------------


def test_the_rows_with_an_even_value_are_the_same_in_both_cases_and_the_signed_rows_flip(tmp_path):
    """Compliance, relaxed von Mises stress and the abs displacement row give the same value and gradient for the pair. The
    displacement row with the sense max reads the sign: its value and gradient change sign. Equal limits, so the rows compare."""
    cases = [with_(DOWN, compliance_limit=1.0, stress_limit=5.0), with_(UP, compliance_limit=1.0, stress_limit=5.0)]
    lim = {"displacement": [{"name": "norm", "region": "tip", "limit_mm": 0.05},
                            {"name": "zmax", "region": "tip", "limit_mm": 0.05, "direction": [0, 0, 1], "sense": "max"}]}
    p = write_case(tmp_path, cases, limits=lim)
    cfg = load_config(p)
    assert [q["relations"] for q in loads.repeated_load_cases(cfg)] == [["opposite"]]
    m = Model(build_problem(cfg))
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13
    form = formulation_from_config(cfg, m.n_design)
    x = np.random.default_rng(5).uniform(0.3, 0.7, m.n_design)
    ev = evaluate(m, form, x, 2.0)
    index = {c.name: form.n_bound + i for i, c in enumerate(form.constraints)}
    value = {e["name"]: e["value"] for e in ev.constraints}
    scale = lambda a: float(np.abs(a).max())
    for kind in ("compliance", "stress", "displacement:norm"):
        a, b = f"{kind}:down", f"{kind}:up"
        assert value[b] == pytest.approx(value[a], rel=1e-9), kind
        assert scale(ev.dg[index[a]] - ev.dg[index[b]]) <= 1e-8 * scale(ev.dg[index[a]]), kind
    a, b = "displacement_max:zmax:down", "displacement_max:zmax:up"
    assert value[a] < 0 < value[b] and value[b] == pytest.approx(-value[a], rel=1e-9)             # the sign flips
    assert scale(ev.dg[index[a]] + ev.dg[index[b]]) <= 1e-8 * scale(ev.dg[index[a]])               # and so does the gradient
    assert scale(ev.dg[index[a]] - ev.dg[index[b]]) > 1e-3 * scale(ev.dg[index[a]])


def test_the_buckling_spectrum_flips_with_the_sign_of_the_load(tmp_path):
    """Kg is linear in the stress, so the opposite load has the spectrum of -Kg reversed. A column in compression has a lowest
    buckling factor of about 2 (mu = 1 / factor, 0.53); the same column in tension keeps only a small positive mu (0.0004: a
    local compression at the clamp, from the restrained Poisson contraction), so its factor is more than 1000 times higher. The
    buckling row of the opposite case is another row: the text of the fact says so."""
    p = column_yaml(tmp_path, a=2, b=3, length=12)
    d = yaml.safe_load(p.read_text())
    axial = d["load_cases"][0]
    d["load_cases"].append({"name": "pull", "weight": 1.0, "forces": [{"region": "cap", "force": [0.0, 0.0, 1000.0]}]})
    d["buckling"] = {"min_factor": 150}
    p.write_text(yaml.safe_dump(d))
    cfg = load_config(p)
    [pair] = loads.repeated_load_cases(cfg)
    assert (pair["cases"], pair["relations"]) == (["axial", "pull"], ["opposite"]) and "buckling row" in pair["sign_rows"][0]
    assert axial["forces"][0]["force"][2] == -1000.0
    m = Model(build_problem(cfg))
    geo = br.Geometry(m)
    rho = np.full(m.n_design, 0.8)
    mu = []
    for ci in (0, 1):
        K, _u, _ua, _ra, Kg = dense_state(m, geo, rho, ci)
        mu.append(np.sort(sla.eigh(-Kg, K, eigvals_only=True)))
    assert mu[0][-1] > 0 and mu[1][-1] < 1e-2 * mu[0][-1]                  # the lowest factor 1 / mu_max differs by orders
    assert np.allclose(mu[1], -mu[0][::-1], rtol=1e-6, atol=1e-9 * mu[0][-1])      # the whole spectrum is reversed
