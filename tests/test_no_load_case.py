"""A load case that applies no load (U21; `LoadCase.applies_load`, `loads.no_load_lines`, `loads.no_load_refusal`, CONTRACTS.md section 13).

A case applies no load when every component of every `force` and `moment` is exactly 0.0 and no pressure is other than 0.0. No
tolerance. The case adds nothing to the objective, and its weight and its stress limit do nothing. Where such a case sits next to a
row that cannot work with it (a `compliance_limit`, `case_combine: max`, the buckling row and check; each stopped a run with an
internal error when it was measured), or no case applies a load under the compliance objective (the objective is 0 for every
design), the loader refuses the YAML with options. Every other YAML gets one `fact:` line in `topopt check`, `check_problem`
(`warnings`), `run.log` and the part builder, and `derived.load_cases[].applies_load`.
"""
import json
import os
import re
import sys
import time

import pytest

from test_repeated_load_cases import SPEC, write_case
from topopt import loads
from topopt.cli import check_data, main
from topopt.config import UserError, load_config
from topopt.optimize import problem_facts
from topopt.regions_builder import build_spec, check_yaml

ZERO = {"name": "zero", "forces": [{"region": "tip", "force": [0, 0, 0]}]}
DOWN = {"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}
FACT = "fact: load case 'zero' applies no load"
MASS = {"objective": "mass"}


def with_(case, **keys):
    return dict(case, **keys)


def fact_lines(lines):
    return [w for w in lines if "applies no load" in w or "apply no load" in w]


# --- the rule: what is a load ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("entries, pressures, applies", [
    ([{"region": "tip", "force": [0, 0, 0]}], [], False),
    ([{"region": "tip", "force": [-0.0, 0.0, -0.0]}], [], False),
    ([{"region": "tip", "force": [0, 0, 0]}, {"region": "side", "force": [0, 0, 0]}], [], False),
    ([{"region": "tip", "force": [0, 0, 0], "moment": [0, 0, 0]}], [], False),
    ([{"region": "tip", "force": [0, 0, 0], "at": [1.0, 2.0, 3.0]}], [], False),     # a remote force of 0 N
    ([{"region": "tip", "force": [0, 0, 0]}], [{"region": "side", "value_mpa": 0.0}], False),
    ([{"region": "tip", "force": [0, 0, 1e-300]}], [], True),                         # no tolerance: any other number is a load
    ([{"region": "tip", "force": [0, 0, 0], "moment": [0, 20.0, 0]}], [], True),      # a moment alone is a load (the nodal sum is 0)
    ([{"region": "tip", "force": [0, 0, 0]}, {"region": "side", "force": [0, 0, -1.0]}], [], True),
    ([], [{"region": "tip", "value_mpa": 0.5}], True),
    ([{"region": "tip", "force": [0, 0, 0]}], [{"region": "tip", "value_mpa": -0.25}], True),
], ids=["zero", "negative-zero", "two-zero-entries", "zero-moment", "zero-remote", "zero-pressure", "tiny", "moment",
        "zero-and-force", "pressure", "zero-force-and-pressure"])
def test_a_case_applies_a_load_when_any_force_moment_or_pressure_is_not_exactly_zero(tmp_path, entries, pressures, applies):
    case = {"name": "zero", "forces": entries, "pressures": pressures}
    p = write_case(tmp_path, [case, DOWN])                       # a normal case next to it: no refusal under the compliance objective
    cfg = load_config(p)
    d = check_data(p)
    # The warning counts come first: the code before U21 has no `applies_load`, so a test that read it first never reached the counts.
    assert len(fact_lines(d["warnings"])) == (0 if applies else 1)
    assert len(fact_lines(problem_facts(cfg))) == (0 if applies else 1)
    assert [c.applies_load for c in cfg.load_cases] == [applies, True]
    assert [c["applies_load"] for c in d["derived"]["load_cases"]] == [applies, True]


def test_the_weight_is_no_load(tmp_path):
    """`weight` is the share of the case in the objective; the engine has no body force."""
    cfg = load_config(write_case(tmp_path, [with_(ZERO, weight=234.0), DOWN]))
    assert [c.applies_load for c in cfg.load_cases] == [False, True]


def test_the_sum_of_the_nodal_forces_is_no_test_for_a_pure_moment(tmp_path):
    """A moment gives `total_magnitude_n` 0.0, as a case with no load does: only `applies_load` tells them apart."""
    moment = {"name": "twist", "forces": [{"region": "tip", "force": [0, 0, 0], "moment": [0, 20.0, 0]}]}
    d = check_data(write_case(tmp_path, [moment, ZERO], settings=MASS))
    twist, zero = d["derived"]["load_cases"]
    assert twist["total_magnitude_n"] == zero["total_magnitude_n"] == 0.0
    assert (twist["applies_load"], zero["applies_load"]) == (True, False)
    assert [w for w in fact_lines(d["warnings"])] and "'twist'" not in fact_lines(d["warnings"])[0]


def test_a_squeeze_is_a_load_and_gets_no_word(tmp_path):
    """Two equal and opposite forces on two regions: the sum is 0, the load is real (case 3 of the brief)."""
    squeeze = {"name": "squeeze", "forces": [{"region": "tip", "force": [0, 0, -2.0]}, {"region": "side", "force": [0, 0, 2.0]}]}
    p = write_case(tmp_path, [squeeze])
    d = check_data(p)                                           # the force audit passes (it refused a squeeze on 4bb1e2e)
    assert [c["applies_load"] for c in d["derived"]["load_cases"]] == [True]
    assert not fact_lines(d["warnings"]) and not fact_lines(problem_facts(load_config(p)))


def test_a_squeeze_of_other_numbers_passes_the_force_audit(tmp_path):
    """The audit compared a rounding of the nodal sum (1e-15) with the scale 1e-300 when the sum of the forces was 0."""
    for value in (0.1, 3.3, 7.0, 1e-3):
        squeeze = {"name": "squeeze", "forces": [{"region": "tip", "force": [0, value, 0]}, {"region": "side", "force": [0, -value, 0]}]}
        d = check_data(write_case(tmp_path / str(value), [squeeze]))
        assert d["derived"]["load_cases"][0]["applies_load"] is True


# --- a case with no load next to a normal case: one fact in every tool ---------------------------------------------------


def test_a_case_with_no_load_next_to_a_load_gives_one_fact_and_ok_stays_true(tmp_path, capsys, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    p = write_case(tmp_path, [ZERO, DOWN])
    cfg = load_config(p)
    [fact] = fact_lines(problem_facts(cfg))
    assert fact.startswith(FACT)
    d = check_data(p)
    assert fact_lines(d["warnings"]) == [fact]                 # the warnings (so the MCP tool) have the same text
    assert [(c["name"], c["applies_load"], c["total_magnitude_n"]) for c in d["derived"]["load_cases"]] == [("zero", False, 0.0), ("down", True, 2.0)]
    assert main(["check", str(p), "--json"]) == 0
    assert fact_lines(json.loads(capsys.readouterr().out)["facts"]) == [fact]
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count(FACT) == 1 and out.count("options: give the case a force, a moment or a pressure; or remove the case.") == 1
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    res = server.check_problem(str(p))
    assert res["ok"] is True and fact_lines(res["warnings"]) == [fact]


def test_the_keys_of_derived_are_kept(tmp_path):
    d = check_data(write_case(tmp_path, [ZERO, DOWN]))
    old = {"name", "weight", "forces", "total_magnitude_n", "compliance_limit", "stress_limit_mpa", "stress_limit_from", "allowable_mpa",
           "safety_factor", "allowable_source"}
    assert old <= set(d["derived"]["load_cases"][0]) and set(d["derived"]["load_cases"][0]) - old == {"applies_load", "load_reaches_free_dofs", "load_reach"}   # F4-37 added the second key and the kind (review m1)


def test_the_fact_says_what_follows_the_cost_and_the_options(tmp_path):
    [t] = fact_lines(problem_facts(load_config(write_case(tmp_path, [with_(ZERO, weight=234.0), DOWN]))))
    assert "every force and moment is exactly 0 and no pressure is other than 0" in t
    assert "compliance 0 and stress 0 for every design, so it adds nothing to the objective" in t
    assert "The weight of the case (234) counts for nothing." in t
    assert "Cost: the case still adds one load column to the solve of every pass" in t
    assert t.endswith("options: give the case a force, a moment or a pressure; or remove the case.")
    assert "\n" not in t and "stress limit" not in t.lower()


def test_the_weight_sentence_follows_the_objective(tmp_path):
    [t] = fact_lines(problem_facts(load_config(write_case(tmp_path, [ZERO, DOWN], settings=MASS))))
    assert "weight" not in t                                    # the mass objective reads no weight


def test_a_stress_limit_on_the_case_is_met_by_any_design(tmp_path):
    cases = [with_(ZERO, stress_limit=5.0, weight=3.0), DOWN]
    [t] = fact_lines(problem_facts(load_config(write_case(tmp_path, cases))))
    assert "The stress limit ('zero' 5 MPa) is met by any design." in t and "The weight of the case (3) counts for nothing." in t


def test_a_displacement_limit_on_the_case_only_is_named(tmp_path):
    lim = {"displacement": [{"name": "d0", "region": "tip", "limit_mm": 0.05, "cases": ["zero"]},
                            {"name": "d1", "region": "tip", "limit_mm": 0.05, "cases": ["down"]}]}
    [t] = fact_lines(problem_facts(load_config(write_case(tmp_path, [ZERO, DOWN], limits=lim))))
    assert "The displacement limit 'd0' lists 'zero' only, so any design meets it." in t
    assert "'d1'" not in t.split("options:")[0]
    assert "The `limits` rows that list it: the displacement limit 'd0' lists 'zero'." in t
    assert "A `cases` list must not name a case that you remove, or the loader stops: name another case in it." in t


def test_two_cases_with_no_load_give_one_line_with_both_names(tmp_path):
    two = with_(ZERO, name="zero2", weight=2.5)
    lines = fact_lines(problem_facts(load_config(write_case(tmp_path, [ZERO, two, DOWN]))))
    assert len(lines) == 1
    t = lines[0]
    assert t.startswith("fact: load cases 'zero' and 'zero2' apply no load") and "so they add nothing to the objective" in t
    assert "The weights of 'zero' and 'zero2' (1, 2.5) count for nothing." in t and "give each case a force" in t


def test_a_case_with_a_stress_limit_next_to_a_load_is_not_refused_under_the_mass_objective(tmp_path):
    cases = [with_(ZERO, stress_limit=5.0), with_(DOWN, stress_limit=5.0)]
    for combine in ("sum", "max"):                                # the mass objective has no use for case_combine, so max is no crash
        cfg = load_config(write_case(tmp_path / combine, cases, settings={**MASS, "case_combine": combine}))
        assert len(fact_lines(problem_facts(cfg))) == 1


# --- no case applies a load --------------------------------------------------------------------------------------------------


def test_the_mass_objective_with_no_load_at_all_gets_the_fact_and_is_not_refused(tmp_path):
    p = write_case(tmp_path, [with_(ZERO, stress_limit=5.0)], settings=MASS)
    d = check_data(p)
    [t] = fact_lines(d["warnings"])
    assert "No case applies a load, so the part has nothing to carry: the mass objective gives the lightest design that the rows allow." in t
    assert "The stress limit ('zero' 5 MPa) is met by any design." in t
    assert d["derived"]["load_cases"][0]["applies_load"] is False


def run_error(path, capsys, argv):
    assert main([*argv, str(path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1      # one line, no traceback
    return err


REFUSED = [
    # (id, cases, settings, top, text of the reason)
    ("all-zero-compliance", [ZERO], {}, {},
     "no case applies a load, and the objective is compliance: it is 0 for every design, so nothing in the YAML shapes the part"),
    ("all-zero-stress-limit", [with_(ZERO, stress_limit=5.0)], {}, {}, "no case applies a load, and the objective is compliance"),
    ("two-zero-cases", [ZERO, with_(ZERO, name="zero2")], {}, {}, "no case applies a load, and the objective is compliance"),
    ("zero-pressure", [{"name": "zero", "forces": [], "pressures": [{"region": "tip", "value_mpa": 0.0}]}], {}, {},
     "no case applies a load, and the objective is compliance"),
    ("compliance-limit", [with_(ZERO, compliance_limit=50.0), DOWN], {}, {},
     "'zero' has a compliance_limit, and the row needs a compliance above 0 (the run would stop at pass 0)"),
    ("compliance-limit-mass", [with_(ZERO, compliance_limit=50.0)], MASS, {},
     "'zero' has a compliance_limit, and the row needs a compliance above 0"),
    ("max-combine", [ZERO, DOWN], {"case_combine": "max"}, {},
     "settings.case_combine is max, and the max objective needs a compliance above 0 for every case"),
    ("buckling-row", [ZERO, DOWN], {}, {"buckling": {"min_factor": 5.0}},
     "buckling.min_factor needs a load in every case (the eigen solve of a case with no load has no start vector)"),
    ("buckling-check", [ZERO, DOWN], {}, {"buckling": {"check": True}}, "buckling.check needs a load in every case"),
    ("buckling-both", [ZERO, DOWN], {}, {"buckling": {"min_factor": 5.0, "check": True}},
     "buckling.min_factor and buckling.check need a load in every case"),
]


@pytest.mark.parametrize("cases, settings, top, reason", [r[1:] for r in REFUSED], ids=[r[0] for r in REFUSED])
def test_a_case_with_no_load_next_to_a_row_that_cannot_work_is_refused_with_options(tmp_path, capsys, cases, settings, top, reason):
    p = write_case(tmp_path, cases, settings, **top)
    with pytest.raises(UserError) as e:
        load_config(p)
    msg = str(e.value)
    assert re.match(r"load cases? '[^']+'.* (?:applies|apply) no load \(every force and moment is exactly 0 and no pressure is other than 0\), "
                    r"and the run cannot work: ", msg)
    assert reason in msg and "\n" not in msg and msg.endswith(f"(file: {p.resolve()})")
    assert re.search(r"Options: give (the case|each case) a force, a moment or a pressure", msg)
    assert "`limits`" not in msg.split(" (file: ")[0]            # these YAMLs have no `limits`, so the sentence about the rows is not there
    with pytest.raises(UserError, match="(applies|apply) no load"):
        check_data(p)
    assert reason in run_error(p, capsys, ["check"])


@pytest.mark.parametrize("cases, settings, top, reason", [r[1:] for r in REFUSED[:1] + REFUSED[4:5] + REFUSED[7:8]], ids=["all-zero", "compliance-limit", "buckling-row"])
def test_run_and_check_problem_stop_with_the_same_error_and_no_run_folder(tmp_path, capsys, monkeypatch, cases, settings, top, reason):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    p = write_case(tmp_path, cases, settings, **top)
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and reason in err and err.count("\n") == 1 and not out.exists()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    res = server.check_problem(str(p))
    assert res["ok"] is False and reason in res["error"]


def test_run_topopt_gives_the_text_in_derived_error_and_the_process_stops_with_exit_1(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    from pathlib import Path
    repo = str(Path(__file__).resolve().parent.parent)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -m topopt run {{yaml}} --out {{out}} --quiet")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([repo] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    p = write_case(tmp_path, [ZERO])
    started = server.run_topopt(p.name, out_dir="runs/z")
    assert "applies no load" in started["derived_error"] and "the objective is compliance" in started["derived_error"]
    log = Path(started["log_file"])
    for _ in range(100):                                                # the process ends at the loader, before any mesh
        if log.exists() and "error: load case 'zero' applies no load" in log.read_text():
            break
        time.sleep(0.1)
    assert "error: load case 'zero' applies no load" in log.read_text()
    assert not (tmp_path / "runs" / "z" / "history.jsonl").exists()


def test_each_option_of_a_refusal_works(tmp_path):
    """The YAML loads when the case gets a load, when the case is removed, and when the row is removed or changed."""
    limited = with_(ZERO, compliance_limit=50.0)
    load_config(write_case(tmp_path / "a", [with_(limited, forces=[{"region": "tip", "force": [0, 0, 1.0]}]), DOWN]))   # give the case a force
    load_config(write_case(tmp_path / "b", [DOWN]))                                                                    # remove the case
    load_config(write_case(tmp_path / "c", [ZERO, DOWN]))                                                              # remove the compliance_limit
    load_config(write_case(tmp_path / "d", [ZERO, DOWN], {"case_combine": "sum"}))                                     # case_combine: sum
    load_config(write_case(tmp_path / "e", [DOWN], buckling={"min_factor": 5.0}))                                      # the case removed, the row stays
    load_config(write_case(tmp_path / "f", [with_(ZERO, forces=[{"region": "tip", "force": [0, 0, 1.0]}])]))           # no case applies a load: the one option
    with pytest.raises(UserError, match="'load_cases' must be a list with one case or more"):                           # ... "remove the case" is not one
        load_config(write_case(tmp_path / "g", []))


def options_of(msg):
    """The options of a refusal: the text after `Options: `, without the sentence about the `limits` rows and the `(file: ...)`."""
    return msg.split("Options: ")[1].split(" (file: ")[0].split(". A `limits` row")[0]


def refusal(path, cases, settings=None, **top):
    with pytest.raises(UserError) as e:
        load_config(write_case(path, cases, settings, **top))
    return str(e.value)


def test_the_options_of_a_refusal_fit_the_yaml(tmp_path):
    give, climit = "give the case a force, a moment or a pressure", with_(ZERO, compliance_limit=50.0)
    # No case applies a load: "remove the case" would leave no case, and the loader stops on that.
    assert options_of(refusal(tmp_path / "a", [ZERO])) == give
    assert options_of(refusal(tmp_path / "b", [ZERO, with_(ZERO, name="zero2")])) == "give each case a force, a moment or a pressure"
    # Under the compliance objective, with no load at all, only a load helps: the option of a row does not.
    assert options_of(refusal(tmp_path / "c", [climit])) == give
    # Under the mass objective the option of the row works.
    assert options_of(refusal(tmp_path / "d", [climit], MASS)) == f"{give}; remove the compliance_limit of the case"
    # Some case applies a load: the options stay, with the plural form for several cases.
    assert options_of(refusal(tmp_path / "e", [climit, DOWN])) == f"{give}; remove the case; remove the compliance_limit of the case"
    two = with_(ZERO, name="zero2", compliance_limit=40.0)
    assert options_of(refusal(tmp_path / "f", [climit, two, DOWN])) == (
        "give each case a force, a moment or a pressure; remove the cases; remove the compliance_limit of those cases")


def test_the_sentence_about_the_limits_rows_is_there_only_with_limits_and_a_case_to_remove(tmp_path):
    sentence = "A `limits` row that lists a case you remove must name another case in `cases`"
    limits = {"displacement": [{"name": "d0", "region": "tip", "limit_mm": 0.05, "cases": ["zero"]}]}
    climit = with_(ZERO, compliance_limit=50.0)
    assert sentence not in refusal(tmp_path / "a", [climit, DOWN])                                   # the YAML has no `limits`
    text = refusal(tmp_path / "b", [climit, DOWN], limits=limits)
    assert text.count(sentence) == 1 and text.index("remove the case") < text.index(sentence)       # it follows the option "remove the case"
    assert sentence not in refusal(tmp_path / "c", [climit], MASS, limits=limits)                    # no case is left to remove: no such option


def test_the_part_builder_says_the_fact_or_the_error(tmp_path):
    spec = json.loads(json.dumps(SPEC))
    spec["load_cases"] = [{"name": "zero", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, 0]}]}] + spec["load_cases"][:1]
    lines = []
    r = build_spec(spec, tmp_path / "o", say=lambda *a: lines.append(" ".join(map(str, a))))
    assert r["ok"]
    [mine] = [ln for ln in lines if "applies no load" in ln]
    assert mine.startswith(FACT) and mine in check_data(r["yaml"])["warnings"]
    lines2 = []
    check_yaml(r["yaml"], say=lambda *a: lines2.append(" ".join(map(str, a))))
    assert mine in lines2
    spec["load_cases"] = spec["load_cases"][:1]
    lines3 = []
    r = build_spec(spec, tmp_path / "p", say=lambda *a: lines3.append(" ".join(map(str, a))))
    assert not r["ok"] and any(ln.startswith("ERROR: load case 'zero' applies no load") for ln in lines3)


def test_the_shipped_examples_apply_a_load():
    from pathlib import Path
    yamls = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*/*.yaml"))
    assert yamls
    for y in yamls:
        if "spec" in y.name:
            continue                                    # a part spec for build_regions, not a problem YAML
        assert all(c.applies_load for c in load_config(y).load_cases), y.name


# --- the claims of the text are true -------------------------------------------------------------------------------------------


def read_pass0(out):
    first = json.loads((out / "history.jsonl").read_text().splitlines()[0])
    return first["compliance"], first["compliance_cases"]


def test_the_run_log_has_the_fact_once_before_the_first_pass_and_the_case_adds_nothing(tmp_path):
    with_zero = tmp_path / "z"
    alone = tmp_path / "a"
    assert main(["run", str(write_case(tmp_path / "pz", [with_(ZERO, weight=7.0), DOWN])), "--out", str(with_zero), "--quiet"]) == 0
    assert main(["run", str(write_case(tmp_path / "pa", [DOWN])), "--out", str(alone), "--quiet"]) == 0
    log = (with_zero / "run.log").read_text()
    assert log.count(FACT) == 1
    assert log.index(FACT) < re.search(r"^it +0 ", log, re.M).start()                 # the facts come before the first pass
    assert FACT not in (alone / "run.log").read_text()
    c_zero, cases = read_pass0(with_zero)
    c_alone, _ = read_pass0(alone)
    assert cases[0] == 0.0 and c_zero == pytest.approx(c_alone, rel=1e-9)               # its compliance is 0 and its weight (7) counts for nothing
    run = json.loads((with_zero / "run.json").read_text())
    assert [c["applies_load"] for c in run["derived"]["load_cases"]] == [False, True]
    zero_k, down_k = json.loads((with_zero / "result" / "metrics.json").read_text())["stiffness_cases"]
    assert zero_k["N_per_mm"] is None and "the case applies no load" in zero_k["reason"]            # null stays null, and it says why
    assert down_k["N_per_mm"] > 0 and "reason" not in down_k


def test_the_null_stiffness_of_other_cases_has_the_reason_it_had():
    from topopt.post import case_stiffness
    force = lambda v, kind="force": {"region": "tip", "force": v, "kind": kind}                      # noqa: E731
    out = case_stiffness([{"name": "zero", "forces": [force([0.0, 0.0, 0.0])]},
                          {"name": "down", "forces": [force([0.0, 0.0, -2.0])]},
                          {"name": "twist", "forces": [force([0.0, 0.0, 0.0], "moment")]},
                          {"name": "lost", "forces": [force([0.0, 0.0, -2.0])]}], [0.0, 0.5, 0.25, 0.0])
    assert [(o["name"], o["N_per_mm"]) for o in out] == [("zero", None), ("down", 8.0), ("twist", None), ("lost", None)]
    assert "the case applies no load: every force is 0" in out[0]["reason"] and "|F|^2 / c is not defined" in out[0]["reason"]
    assert "reason" not in out[1]
    assert "loads of kind moment" in out[2]["reason"]                                                # the old reason of a moment is kept
    assert "reason" not in out[3]                                       # a compliance of 0 from a force that is not 0 has no such reason


def test_a_mass_run_with_no_load_goes_on_and_says_so_in_the_log(tmp_path):
    out = tmp_path / "m"
    p = write_case(tmp_path / "p", [with_(ZERO, stress_limit=5.0)], settings=MASS)
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text()
    assert log.count(FACT) == 1 and "No case applies a load, so the part has nothing to carry" in log
    assert "error" not in json.loads((out / "result" / "metrics.json").read_text())["stress"]       # the stress report was made (it divided by the peak 0.0)


def test_a_yaml_that_gets_only_the_fact_has_no_refusal_text(tmp_path):
    assert loads.no_load_refusal(load_config(write_case(tmp_path, [ZERO, DOWN], settings=MASS))) is None
    assert loads.no_load_refusal(load_config(write_case(tmp_path / "n", [DOWN]))) is None


def test_a_stress_limit_on_a_case_with_no_load_keeps_the_stress_report_of_the_run(tmp_path):
    """The stress report divided the limit by the grey peak of the case, which is 0.0 without a load: the report of every case was lost."""
    out = tmp_path / "s"
    p = write_case(tmp_path / "p", [with_(ZERO, stress_limit=5.0), with_(DOWN, stress_limit=5.0)])
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    stress = json.loads((out / "result" / "metrics.json").read_text())["stress"]
    assert "error" not in stress and set(stress["cases"]) == {"zero", "down"}
    zero = stress["cases"]["zero"]
    assert zero["physical"]["peak_MPa"] == 0.0 and "Both voxel estimates are at or below the limit." in zero["summary"]
    assert "times the grey peak" not in zero["summary"]
