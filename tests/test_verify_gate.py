"""The opt-in hard gate of `topopt verify --gate` (K1-B, CONTRACTS.md sections 16 ("The gate"), 31, 33 and 36).

The first part tests the gate logic on plain dicts (no gmsh, no solve). The second part runs `topopt verify --gate` on the recorded hook run at the
mesh size 3 mm (like `test_verify_buckling_run.py`). The last test is `slow`: a small column with `buckling.min_factor` above and below its Euler factor.
"""
import contextlib
import copy
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_gmsh, needs_skfem  # noqa: E402
from test_verify_run import HOOK_YAML, RECORDING, copy_run, needs_recording  # noqa: E402

from topopt.config import UserError  # noqa: E402
from topopt.verify import gate  # noqa: E402
from test_verify_buckling import column  # noqa: E402,F401  (a fixture: the small tet column and the real buckling record)

SIZE = 3.0


# --------------------------------------------------------------------------- #
# The logic, on plain dicts
# --------------------------------------------------------------------------- #

def limits(**kw) -> dict:
    out = dict(stress={"a": dict(limit=50.0, source="load_cases[a].stress_limit"), "b": dict(limit=20.0, source="load_cases[b].stress_limit")},
               buckling=dict(limit=5.0, source="buckling.min_factor"), applies_load={"a": True, "b": True},
               rows=[dict(name="displacement:tip:a", check="displacement", case="a", limit=0.5, source="limits.displacement[tip].limit_mm"),
                     dict(name="shift:seats:b", check="alignment", case="b", limit=0.1, source="limits.alignment[seats].shift_limit_mm")])
    out.update(kw)
    return out


def level(size=1.5, stress=(10.0, 5.0), factors=(8.0, 9.0), disp=0.3, shift=0.05, bk_status="done", motion_status="done") -> dict:
    return dict(
        size_mm=size,
        cases={n: {"von_mises": {"percentile_by_volume_MPa": v, "peak_qp_MPa": 2.5 * v}} for n, v in zip("ab", stress)},
        buckling={"status": bk_status, "reason": "memory", "cases": [{"name": n, "load_factors": [f, 2 * f]} for n, f in zip("ab", factors)]
                  if bk_status == "done" else []},
        motion={"status": motion_status, "reason": "rows_cannot_work", "rows": [
            {"name": "displacement:tip:a", "case": "a", "unit": "mm", "tet_value": disp, "limit": 0.5},
            {"name": "shift:seats:b", "case": "b", "unit": "mm", "tet_value": shift, "limit": 0.1}] if motion_status == "done" else []})


def met(g, kind, name):
    return g["checks"][kind][name]["met"]


def test_a_part_inside_every_limit_passes_and_every_check_has_value_limit_source_and_met():
    g = gate.evaluate(level(), limits(), percentile=99.9)
    assert g["passed"] is True and g["failed"] == [] and g["not_gated"] == [] and g["requested"] is True
    assert set(g["checks"]) == {"stress", "buckling", "displacement", "alignment"}
    rec = g["checks"]["stress"]["a"]
    assert rec["value"] == 10.0 and rec["limit"] == 50.0 and rec["source"] == "load_cases[a].stress_limit" and rec["met"] is True
    assert rec["percentile"] == 99.9 and g["mesh_size_mm"] == 1.5
    assert g["checks"]["buckling"]["b"]["value"] == 9.0 and g["checks"]["buckling"]["b"]["limit"] == 5.0
    assert g["checks"]["displacement"]["displacement:tip:a"]["case"] == "a"
    assert "change_from_previous" not in rec and "previous_level_mm" not in g          # one level: no change
    json.dumps(g)                                                                   # JSON-safe


@pytest.mark.parametrize("kw, kind, name", [
    (dict(stress=(50.1, 5.0)), "stress", "a"), (dict(stress=(10.0, 20.5)), "stress", "b"),
    (dict(factors=(4.9, 9.0)), "buckling", "a"), (dict(disp=0.6), "displacement", "displacement:tip:a"),
    (dict(shift=0.2), "alignment", "shift:seats:b")])
def test_each_check_fails_the_gate_on_its_own(kw, kind, name):
    g = gate.evaluate(level(**kw), limits(), percentile=99.9)
    assert g["passed"] is False and g["failed"] == [f"{kind}:{name}"]
    assert met(g, kind, name) is False
    assert sum(r["met"] is False for recs in g["checks"].values() for r in recs.values()) == 1


def test_a_value_on_the_limit_passes_and_there_is_no_tolerance():
    g = gate.evaluate(level(stress=(50.0, 20.0), factors=(5.0, 5.0), disp=0.5, shift=0.1), limits(), percentile=99.9)
    assert g["passed"] is True
    assert gate.evaluate(level(stress=(50.0000001, 5.0)), limits(), percentile=99.9)["passed"] is False
    assert gate.evaluate(level(factors=(4.9999999, 9.0)), limits(), percentile=99.9)["passed"] is False


@pytest.mark.parametrize("kw, kind, name, text", [
    (dict(bk_status="skipped"), "buckling", "a", "skipped"),
    (dict(motion_status="skipped"), "displacement", "displacement:tip:a", "rows_cannot_work"),
    (dict(motion_status="error"), "alignment", "shift:seats:b", "error")])
def test_a_check_that_could_not_run_has_met_null_and_fails_the_gate(kw, kind, name, text):
    g = gate.evaluate(level(**kw), limits(), percentile=99.9)
    rec = g["checks"][kind][name]
    assert rec["met"] is None and rec["value"] is None and text in rec["reason"]
    assert g["passed"] is False and f"{kind}:{name}" in g["failed"]


def test_a_missing_or_non_finite_value_has_met_null():
    lv = level()
    lv["cases"]["a"]["von_mises"]["percentile_by_volume_MPa"] = float("nan")
    g = gate.evaluate(lv, limits(), percentile=99.9)
    assert met(g, "stress", "a") is None and not g["passed"]
    lv = level()
    del lv["cases"]["b"]
    assert met(gate.evaluate(lv, limits(), percentile=99.9), "stress", "b") is None
    lv = level()
    lv["buckling"]["cases"] = lv["buckling"]["cases"][:1]
    g = gate.evaluate(lv, limits(), percentile=99.9)
    assert met(g, "buckling", "b") is None and "no entry" in g["checks"]["buckling"]["b"]["reason"]
    lv = level()
    lv["buckling"]["cases"][0]["load_factors"] = []
    assert met(gate.evaluate(lv, limits(), percentile=99.9), "buckling", "a") is None
    lv = level()
    del lv["buckling"]                                                                # the check was not on
    g = gate.evaluate(lv, limits(), percentile=99.9)
    assert met(g, "buckling", "a") is None and "no buckling record" in g["checks"]["buckling"]["a"]["reason"]
    lv = level()
    lv["motion"]["rows"] = lv["motion"]["rows"][:1]
    g = gate.evaluate(lv, limits(), percentile=99.9)
    assert met(g, "alignment", "shift:seats:b") is None and "no such row" in g["checks"]["alignment"]["shift:seats:b"]["reason"]
    lv = level()
    del lv["motion"]
    assert met(gate.evaluate(lv, limits(), percentile=99.9), "displacement", "displacement:tip:a") is None


def test_a_check_with_no_limit_is_listed_as_not_gated_and_does_not_fail():
    lim = limits(buckling=None, rows=[], stress={"a": dict(limit=50.0, source="s"), "b": dict(limit=None, source=None)})
    g = gate.evaluate(level(), lim, percentile=99.9)
    assert g["passed"] is True and set(g["checks"]) == {"stress"} and set(g["checks"]["stress"]) == {"a"}
    listed = {(n["check"], n.get("case")) for n in g["not_gated"]}
    assert listed == {("stress", "b"), ("buckling", None), ("displacement", None), ("alignment", None)}
    assert all(n["reason"] for n in g["not_gated"])
    # a case that applies no load has no buckling check
    g = gate.evaluate(level(), limits(applies_load={"a": True, "b": False}), percentile=99.9)
    assert set(g["checks"]["buckling"]) == {"a"} and ("buckling", "b") in {(n["check"], n.get("case")) for n in g["not_gated"]}
    assert g["passed"] is True
    # nothing gated at all is no pass
    none = limits(buckling=None, rows=[], stress={"a": dict(limit=None, source=None), "b": dict(limit=None, source=None)})
    g = gate.evaluate(level(), none, percentile=99.9)
    assert g["passed"] is False and g["failed"] == [] and "no check was gated" in g["reason"]


def _tension(*names, status="no_positive_factor"):
    return [{"name": n, "status": status, "load_factors": []} for n in names]


def test_a_load_case_with_no_compression_is_not_gated_and_is_neither_a_failure_nor_a_pass():
    """D-1: the buckling check ran and found nothing that can buckle (status `no_positive_factor`). That is a result."""
    lv = level()
    lv["buckling"]["cases"][0] = _tension("a")[0]
    g = gate.evaluate(lv, limits(), percentile=99.9)
    assert set(g["checks"]["buckling"]) == {"b"} and g["passed"] is True and g["failed"] == []
    (entry,) = [n for n in g["not_gated"] if n["check"] == "buckling"]
    assert entry["case"] == "a" and entry["reason"] == "no compression in this load case, nothing can buckle"
    assert "gate: buckling a: not gated. no compression in this load case, nothing can buckle" in gate.gate_lines(g)
    # the other buckling statuses have no factor and stay `met: null`: they fail the gate
    for status in ("not_converged", "time_budget", "error"):
        lv = level()
        lv["buckling"]["cases"][0] = _tension("a", status=status)[0]
        g = gate.evaluate(lv, limits(), percentile=99.9)
        assert met(g, "buckling", "a") is None and "buckling:a" in g["failed"] and g["passed"] is False
    # a case with no compression does not hide a check that fails
    lv = level(stress=(60.0, 5.0))
    lv["buckling"]["cases"][0] = _tension("a")[0]
    g = gate.evaluate(lv, limits(), percentile=99.9)
    assert g["failed"] == ["stress:a"] and g["passed"] is False


@needs_skfem
def test_the_gate_reads_the_record_that_the_buckling_check_writes_for_a_case_without_compression(column, tmp_path):
    """D-1 on the record of `buckling_p2.run_check` (the real writer, a small tet column), not on a typed dict."""
    rec, _, _ = column.check(tmp_path, ("tension", "push"), sigs={"tension": column.tension, "push": column.push})
    assert rec["cases"][0]["status"] == "no_positive_factor" and rec["cases"][1]["status"] == "ok"
    lim = limits(stress={"tension": dict(limit=None, source=None), "push": dict(limit=None, source=None)}, rows=[],
                 buckling=dict(limit=5.0, source="buckling.min_factor"), applies_load={"tension": True, "push": True})
    g = gate.evaluate(dict(size_mm=2.0, cases={}, buckling=rec), lim, percentile=99.9)
    assert set(g["checks"]["buckling"]) == {"push"} and g["checks"]["buckling"]["push"]["met"] is True and g["passed"] is True
    (entry,) = [n for n in g["not_gated"] if n["check"] == "buckling"]
    assert entry["case"] == "tension" and entry["reason"] == gate.NO_COMPRESSION
    only_tension = dict(lim, applies_load={"tension": True, "push": False})
    alone = gate.evaluate(dict(size_mm=2.0, cases={}, buckling=dict(rec, cases=rec["cases"][:1])), only_tension, percentile=99.9)
    assert alone["passed"] is False and alone["failed"] == [] and "no check was gated" in alone["reason"]


def test_a_gate_whose_checks_are_all_not_applicable_after_the_solve_is_no_pass_and_says_why():
    """D-1 and D-5: `passed` is false with an empty `failed` list only when no check was gated, and then the gate has a `reason`."""
    only_bk = limits(stress={"a": dict(limit=None, source=None), "b": dict(limit=None, source=None)}, rows=[])
    lv = level()
    lv["buckling"]["cases"] = _tension("a", "b")
    g = gate.evaluate(lv, only_bk, percentile=99.9)
    assert g["checks"] == {} and g["passed"] is False and g["failed"] == []
    assert "no check was gated" in g["reason"] and {n["case"] for n in g["not_gated"] if n["check"] == "buckling"} == {"a", "b"}
    assert gate.gate_lines(g)[-1].startswith("gate: FAILED (no check was gated")
    ok = gate.evaluate(level(), limits(), percentile=99.9)
    assert "reason" not in ok                                                            # a gate that gated a check has no reason


def test_every_record_has_a_margin_that_is_1_or_more_when_met_and_a_stress_record_has_the_peak():
    """D-4: limit / value for a stress and a row, value / limit for a buckling factor; the peak at one point is for information."""
    g = gate.evaluate(level(stress=(10.0, 40.0), factors=(8.0, 2.5), disp=0.25, shift=0.2), limits(), percentile=99.9)
    assert g["checks"]["stress"]["a"]["margin"] == pytest.approx(5.0) and g["checks"]["stress"]["b"]["margin"] == pytest.approx(0.5)
    assert g["checks"]["buckling"]["a"]["margin"] == pytest.approx(1.6) and g["checks"]["buckling"]["b"]["margin"] == pytest.approx(0.5)
    assert g["checks"]["displacement"]["displacement:tip:a"]["margin"] == pytest.approx(2.0)
    assert g["checks"]["alignment"]["shift:seats:b"]["margin"] == pytest.approx(0.5)
    for recs in g["checks"].values():
        for rec in recs.values():
            assert (rec["margin"] >= 1) == (rec["met"] is True)                      # the margin and `met` say the same
    assert g["checks"]["stress"]["a"]["peak_qp_MPa"] == 25.0 and g["checks"]["stress"]["b"]["peak_qp_MPa"] == 100.0
    zero = gate.evaluate(level(stress=(0.0, 5.0)), limits(), percentile=99.9)
    assert zero["checks"]["stress"]["a"]["margin"] is None and zero["checks"]["stress"]["a"]["met"] is True      # nothing to divide
    none = gate.evaluate(level(bk_status="skipped"), limits(), percentile=99.9)
    assert none["checks"]["buckling"]["a"]["margin"] is None
    json.dumps(g)
    assert "met, margin 5" in "\n".join(gate.gate_lines(g)) and "NOT MET, margin 0.5" in "\n".join(gate.gate_lines(g))


def test_with_levels_the_gate_gives_the_change_from_the_level_before():
    g = gate.evaluate(level(size=1.5, stress=(10.0, 5.0), factors=(8.0, 9.0)), limits(), percentile=99.9,
                      previous=level(size=2.0, stress=(8.0, 5.0), factors=(10.0, 9.0), disp=0.0))
    assert g["previous_level_mm"] == 2.0
    a = g["checks"]["stress"]["a"]
    assert a["previous_value"] == 8.0 and a["change_from_previous"] == pytest.approx(0.25)
    assert g["checks"]["stress"]["b"]["change_from_previous"] == 0.0
    assert g["checks"]["buckling"]["a"]["change_from_previous"] == pytest.approx(-0.2)
    assert g["checks"]["displacement"]["displacement:tip:a"]["change_from_previous"] is None      # the coarser value is 0
    assert g["passed"] is True                                                                      # the change is a fact: it does not decide


def test_the_lines_say_each_check_the_not_gated_ones_and_the_verdict():
    g = gate.evaluate(level(factors=(4.0, 9.0)), limits(rows=[]), percentile=99.9, previous=level(size=2.0))
    text = "\n".join(gate.gate_lines(g))
    assert "gate: buckling a: 4 (needs >= 5 load factor; buckling.min_factor): NOT MET" in text
    assert "gate: stress a: 10 (needs <= 50 MPa; load_cases[a].stress_limit): met" in text and "Change from the level 2 mm" in text
    assert "gate: displacement: not gated" in text and text.endswith("gate: FAILED (buckling:a)")
    assert gate.gate_lines(gate.evaluate(level(), limits(), percentile=99.9))[-1] == "gate: PASSED"
    null = gate.evaluate(level(bk_status="skipped"), limits(), percentile=99.9)
    assert "NOT MET (could not run)" in "\n".join(gate.gate_lines(null))


def test_the_plain_lines_use_the_gate_limit_when_there_is_no_allowable():
    from topopt.verify.report import plain_lines
    res = dict(cases={n: {"von_mises": {"percentile": 99.9, "percentile_by_volume_MPa": v, "p99_9_by_volume_MPa": v, "peak_qp_MPa": 2.5 * v}}
                     for n, v in zip("ab", (10.0, 25.0))})
    res["gate"] = gate.evaluate(level(stress=(10.0, 25.0)), limits(), percentile=99.9)
    lines = plain_lines(res)
    assert "The gate limit is 50 MPa (load_cases[a].stress_limit): the value is at or below the limit" in lines[0]
    assert "above the limit" in lines[1]
    assert not any("No allowable stress was given" in x for x in lines)


def test_the_plain_lines_of_the_gate_path_have_the_margin_the_peak_note_and_the_verdict():
    """D-4: `verify_status.plain` is what an MCP agent reads. The gate path says what the allowable path says, and the verdict."""
    from topopt.verify.report import plain_lines
    lim = limits(stress={"a": dict(limit=20.0, source="load_cases[a].stress_limit"), "b": dict(limit=20.0, source="load_cases[b].stress_limit")},
                 buckling=None, rows=[])
    lv = level(stress=(10.0, 30.0))
    res = dict(cases=lv["cases"], gate=gate.evaluate(lv, lim, percentile=99.9))
    for n in "ab":
        res["cases"][n]["von_mises"].update(percentile=99.9, p99_9_by_volume_MPa=res["cases"][n]["von_mises"]["percentile_by_volume_MPa"])
    lines = plain_lines(res)
    assert "margin 2.00 (limit divided by value)" in lines[0]
    assert "The highest value at one point, 25 MPa, is above the limit; the check compares the p99.9 value only." in lines[0]
    assert "above the limit, margin 0.67" in lines[1] and "highest value at one point" not in lines[1]       # a failed case needs no peak note
    assert "gate: FAILED (stress:b)" in lines
    res["gate"] = gate.evaluate(level(stress=(10.0, 12.0)), lim, percentile=99.9)
    assert "gate: PASSED" in plain_lines(res)
    del res["gate"]
    assert not any(x.startswith("gate:") for x in plain_lines(res))                                              # no gate: as before


# --------------------------------------------------------------------------- #
# The limits from the config
# --------------------------------------------------------------------------- #

def hook_config(tmp_path, **kw):
    from topopt.config import load_config
    return load_config(hook_yaml(tmp_path, **kw))


def hook_yaml(tmp_path, cases=None, measure=None, **blocks) -> Path:
    """The hook YAML with absolute region paths, new top-level blocks, `cases`: {case name: {key: value}} merged into the load cases,
    and `measure`: {region name: STL file name in examples/hook} as measure regions."""
    d = yaml.safe_load(HOOK_YAML.read_text())
    base = HOOK_YAML.parent
    r = d["regions"]
    r["design"] = str(base / r["design"])
    r["fixed"] = [str(base / x) for x in r["fixed"]]
    r["keep_out"] = [str(base / x) for x in r["keep_out"]]
    r["load"] = {k: str(base / v) for k, v in r["load"].items()}
    if measure:
        r["measure"] = {k: str(base / v) for k, v in measure.items()}
    for c in d["load_cases"]:
        c.update((cases or {}).get(c["name"], {}))
    d.update(blocks)
    p = tmp_path / "hook_gate.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


ROWS = {"displacement": [{"name": "tip_pull", "region": "tip", "limit_mm": 0.05, "cases": ["pull_off"]}]}


def test_the_limits_come_from_keys_that_exist(tmp_path):
    cfg = hook_config(tmp_path, cases={"hang": {"stress_limit": 1.5}, "pull_off": {"allowable_mpa": 40.0, "safety_factor": 2.0}},
                      buckling={"min_factor": 3.0}, limits=ROWS)
    lim = gate.gate_limits(cfg)
    assert lim["stress"]["hang"] == {"limit": 1.5, "source": "load_cases[hang].stress_limit"}
    assert lim["stress"]["pull_off"] == {"limit": 20.0, "source": "load_cases[pull_off].allowable_mpa / safety_factor"}
    assert lim["buckling"] == {"limit": 3.0, "source": "buckling.min_factor"}
    assert lim["rows"] == [dict(name="displacement:tip_pull:pull_off", check="displacement", case="pull_off", limit=0.05,
                                source="limits.displacement[tip_pull].limit_mm")]
    over = gate.gate_limits(cfg, allowable_mpa=7.0)                                # --allowable-mpa replaces the limit of every case
    assert {c: s["limit"] for c, s in over["stress"].items()} == {"hang": 7.0, "pull_off": 7.0}
    assert "--allowable-mpa" in over["stress"]["hang"]["source"]
    plain = gate.gate_limits(hook_config(tmp_path))
    assert plain["buckling"] is None and plain["rows"] == [] and all(s["limit"] is None for s in plain["stress"].values())


def test_an_override_that_replaces_a_limit_of_the_yaml_keeps_the_replaced_limit_and_says_if_it_is_looser(tmp_path):
    """D-2: `--allowable-mpa` can loosen a requirement of the YAML. The record shows the replaced limit; it is a fact, not a refusal."""
    cfg = hook_config(tmp_path, cases={"hang": {"stress_limit": 20.0}, "pull_off": {"allowable_mpa": 40.0, "safety_factor": 2.0}})
    over = gate.gate_limits(cfg, allowable_mpa=30.0)
    assert over["stress"]["hang"]["replaced_limit"] == 20.0 and over["stress"]["hang"]["loosened"] is True          # 30 is above 20
    assert over["stress"]["hang"]["replaced_source"] == "load_cases[hang].stress_limit"
    assert over["stress"]["pull_off"]["replaced_limit"] == 20.0 and over["stress"]["pull_off"]["loosened"] is True
    assert over["stress"]["pull_off"]["replaced_source"] == "load_cases[pull_off].allowable_mpa / safety_factor"
    tight = gate.gate_limits(cfg, allowable_mpa=5.0)
    assert tight["stress"]["hang"]["loosened"] is False and tight["stress"]["hang"]["replaced_limit"] == 20.0         # tighter: a fact too
    same = gate.gate_limits(cfg, allowable_mpa=20.0)
    assert same["stress"]["hang"]["loosened"] is False
    plain = gate.gate_limits(cfg)
    assert not any(k in plain["stress"]["hang"] for k in ("replaced_limit", "replaced_source", "loosened"))           # no override: no keys
    bare = gate.gate_limits(hook_config(tmp_path), allowable_mpa=30.0)                                              # nothing in the YAML to replace
    assert all("replaced_limit" not in s and "loosened" not in s for s in bare["stress"].values())
    # the record and the line
    lv = level(stress=(25.0, 5.0))
    lim = dict(stress={"a": dict(limit=30.0, source="--allowable-mpa (all cases)", replaced_limit=20.0,
                                 replaced_source="load_cases[a].stress_limit", loosened=True),
                       "b": dict(limit=3.0, source="--allowable-mpa (all cases)", replaced_limit=20.0,
                                 replaced_source="load_cases[b].stress_limit", loosened=False)},
               buckling=None, applies_load={"a": True, "b": True}, rows=[])
    g = gate.evaluate(lv, lim, percentile=99.9)
    a, b = g["checks"]["stress"]["a"], g["checks"]["stress"]["b"]
    assert (a["replaced_limit"], a["replaced_source"], a["loosened"], a["met"]) == (20.0, "load_cases[a].stress_limit", True, True)
    assert (b["replaced_limit"], b["loosened"], b["met"]) == (20.0, False, False)
    text = "\n".join(gate.gate_lines(g))
    assert "The limit replaces 20 MPa of load_cases[a].stress_limit; the new limit is looser" in text
    assert "The limit replaces 20 MPa of load_cases[b].stress_limit; the new limit is tighter" in text
    json.dumps(g)


def test_a_percentile_that_is_not_the_default_is_a_fact_of_the_stress_record():
    default = gate.evaluate(level(), limits(), percentile=99.9)["checks"]["stress"]["a"]
    assert "replaced_percentile" not in default and "percentile_loosened" not in default
    low = gate.evaluate(level(), limits(), percentile=95.0)["checks"]["stress"]["a"]
    assert low["percentile"] == 95.0 and low["replaced_percentile"] == 99.9 and low["percentile_loosened"] is True
    high = gate.evaluate(level(), limits(), percentile=100.0)["checks"]["stress"]["a"]
    assert high["replaced_percentile"] == 99.9 and high["percentile_loosened"] is False
    text = "\n".join(gate.gate_lines(gate.evaluate(level(), limits(), percentile=95.0)))
    assert "The percentile 95 replaces the default 99.9; the test is looser (a lower percentile gives a lower value)" in text


def test_a_check_that_cannot_run_is_not_counted_by_the_plan():
    """D-5: only a runnable check counts, so `passed: false` never comes with an empty `failed` list and no `reason`."""
    no_load = dict(stress={"a": dict(limit=None, source=None)}, buckling=dict(limit=3.0, source="buckling.min_factor"),
                   applies_load={"a": False}, rows=[])
    with pytest.raises(UserError, match="no limit to check") as e:
        gate.gate_plan(no_load, buckling_flag=None)
    assert "no load case applies a load" in str(e.value)
    with pytest.raises(UserError, match="no limit to check"):
        gate.gate_plan(no_load, buckling_flag=False)                       # a check that cannot run is not asked for, so no `--no-buckling` clash
    with_stress = dict(no_load, stress={"a": dict(limit=2.0, source="s")})
    assert gate.gate_plan(with_stress, buckling_flag=None) is False        # the buckling check is not needed: no case applies a load
    assert gate.gate_plan(with_stress, buckling_flag=False) is False
    two = dict(no_load, stress={"a": dict(limit=None, source=None), "b": dict(limit=None, source=None)}, applies_load={"a": False, "b": True})
    assert gate.gate_plan(two, buckling_flag=None) is True                 # one case applies a load: the check runs
    with pytest.raises(UserError, match="--no-buckling"):
        gate.gate_plan(two, buckling_flag=False)


def test_alignment_rows_name_the_shift_and_the_tilt_key(tmp_path):
    cfg = hook_config(tmp_path, measure={"plate": "wall_plate.stl"},
                      limits={"alignment": [{"name": "seats", "region_a": "tip", "region_b": "plate", "axis": [1, 0, 0],
                                             "shift_limit_mm": 0.1, "tilt_limit_rad": 0.002}]})
    rows = gate.gate_limits(cfg)["rows"]
    assert [(r["name"], r["check"], r["limit"], r["source"]) for r in rows] == [
        ("shift:seats:hang", "alignment", 0.1, "limits.alignment[seats].shift_limit_mm"),
        ("shift:seats:pull_off", "alignment", 0.1, "limits.alignment[seats].shift_limit_mm"),
        ("tilt:seats:hang", "alignment", 0.002, "limits.alignment[seats].tilt_limit_rad"),
        ("tilt:seats:pull_off", "alignment", 0.002, "limits.alignment[seats].tilt_limit_rad")]


def test_a_gate_with_no_limit_is_refused_before_the_run(tmp_path):
    none = gate.gate_limits(hook_config(tmp_path))
    with pytest.raises(UserError, match="no limit to check") as e:
        gate.gate_plan(none, buckling_flag=None)
    assert "--allowable-mpa" in str(e.value)
    with_bk = copy.deepcopy(none)
    with_bk["buckling"] = dict(limit=3.0, source="buckling.min_factor")
    assert gate.gate_plan(with_bk, buckling_flag=None) is True and gate.gate_plan(with_bk, buckling_flag=True) is True
    with pytest.raises(UserError, match="--no-buckling"):
        gate.gate_plan(with_bk, buckling_flag=False)
    assert gate.gate_plan(dict(none, stress={"hang": dict(limit=1.0, source="s")}), buckling_flag=False) is False


@needs_recording
def test_the_preflight_refuses_a_gate_with_no_limit_and_turns_the_buckling_check_on_for_min_factor(tmp_path):
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    with pytest.raises(UserError, match="no limit to check"):
        verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, estimate_only=True, gate=True)
    res = verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, estimate_only=True, gate=True, allowable_mpa=5.0)
    assert "buckling_check" not in res and "no_allowable" not in res                 # the gate has its own limits
    y = hook_yaml(tmp_path, buckling={"min_factor": 2.0})
    res = verify_run(run, yaml_path=y, mesh_size=SIZE, estimate_only=True, gate=True)
    assert res["buckling_check"]["on"] is True and "gate" in res["buckling_check"]["source"]
    assert "buckling_bytes" in res["estimates_before_meshing"][0]                    # the estimate carries the check
    with pytest.raises(UserError, match="--no-buckling"):
        verify_run(run, yaml_path=y, mesh_size=SIZE, estimate_only=True, gate=True, buckling=False)
    off = verify_run(run, yaml_path=y, mesh_size=SIZE, estimate_only=True)
    assert "buckling_check" not in off and "no_allowable" in off                     # no gate: as before


def test_the_cli_and_the_mcp_tool_have_the_switch():
    from topopt.cli import build_parser
    p = build_parser()
    assert p.parse_args(["verify", "x", "--gate"]).gate is True and p.parse_args(["verify", "x"]).gate is False
    pytest.importorskip("mcp")
    from topopt_mcp import server
    args = (None, None, None, None, 99.9, None, None, False, None, None, True)
    assert server._verify_args(*args) == [] and server._verify_args(*args, gate=True) == ["--gate"]
    import inspect
    assert inspect.signature(server.verify_run).parameters["gate"].default is False


def test_the_verify_run_description_has_a_short_gate_sentence_the_replace_fact_and_the_phrases_that_k1b_had_cut():
    """D-6 and D-2: the gate sentence is short and points to the card; the key names and explanations of `base` (6084bd8) are back; the length stays under 4096.

    K1-FU-C: the list of result keys moved from the description to the card `verify_run`, and the description points to the card.
    The key names and explanations of D-2 are read in the card; the description keeps `exit_code` with its values and the pointer.
    """
    pytest.importorskip("mcp")
    import asyncio
    from topopt import levers
    from topopt_mcp import server
    tool = next(t for t in asyncio.run(server.mcp.list_tools()) if t.name == "verify_run")
    text = tool.description
    assert len(text) <= 4096
    (gate_line,) = [x.strip() for x in text.splitlines() if x.strip().startswith("gate:")]
    assert len(gate_line) <= 140 and "topopt://levers/verify_run:gate" in gate_line and "exit_code 3 if it fails" in gate_line
    assert "With gate it replaces the YAML stress limits (a larger value loosens them: the user's consent)" in text
    assert "read near_regions and load_band_sensitivity" in text
    card = levers.reference("verify_run")
    for phrase in ("peak_qp_over_p99_9", "counts_by_band", "near_regions (p99.9 without the points near the regions; the share of the top 0.1 % volume near each)",
                   "load_centre (optimizer centre, P2 traction centroid, distance, moment ratio)", "yaml_vs_run (YAML against run.json)", "limit, gate, files"):
        assert phrase in card, phrase
    assert "(0 done, 3 limit failed)" in text
    assert re.search(r"topopt://levers/verify_run(?![:\w])", text), "the pointer to the card that holds the result keys"


# --------------------------------------------------------------------------- #
# The command, on the recorded hook run
# --------------------------------------------------------------------------- #

def cli(run, *argv, yaml_path=HOOK_YAML):
    from topopt.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        sizes = [] if "--levels" in argv else ["--mesh-size", str(SIZE)]
        code = main(["verify", str(run), "--yaml", str(yaml_path), *sizes, "--no-sensitivity", *map(str, argv)])
    return code, out.getvalue(), err.getvalue()


def result(run) -> dict:
    return json.loads((Path(run) / "verify" / "verify.json").read_text())


@needs_gmsh
@needs_skfem
@needs_recording
def test_the_exit_code_is_3_when_the_gate_fails_and_0_when_it_passes(tmp_path):
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--gate", "--allowable-mpa", 1000)
    assert code == 0, err
    res = result(run)
    assert res["exit_code"] == 0 and res["gate"]["passed"] is True and set(res["gate"]["checks"]) == {"stress"}
    assert set(res["gate"]["checks"]["stress"]) == {"hang", "pull_off"}
    assert res["limit"]["allowable_MPa"] == 1000.0 and not res["limit"]["exceeded"]           # the old block stays
    assert "no_allowable" not in res and "gate: PASSED" in out and "gate: stress hang" in out
    rec = res["gate"]["checks"]["stress"]["pull_off"]
    assert rec["met"] is True and rec["limit"] == 1000.0 and "--allowable-mpa" in rec["source"] and rec["value"] > 0
    assert rec["value"] == res["cases"]["pull_off"]["von_mises"]["percentile_by_volume_MPa"]
    assert rec["margin"] == pytest.approx(1000.0 / rec["value"]) and rec["peak_qp_MPa"] == res["cases"]["pull_off"]["von_mises"]["peak_qp_MPa"]
    from topopt.verify.report import plain_lines
    plain = plain_lines(res)                                                         # what `verify_status.plain` gives an MCP agent
    assert "gate: PASSED" in plain
    code, out, err = cli(run, "--gate", "--allowable-mpa", 0.001)
    assert code == 3, err
    res = result(run)
    assert res["exit_code"] == 3 and res["gate"]["passed"] is False and res["gate"]["failed"] == ["stress:hang", "stress:pull_off"]
    assert "gate: FAILED" in out


@needs_gmsh
@needs_skfem
@needs_recording
def test_the_per_case_limits_of_the_yaml_gate_each_case_and_the_other_checks_join_when_the_yaml_has_them(tmp_path):
    run = copy_run(tmp_path / "run")
    y = hook_yaml(tmp_path, cases={"hang": {"stress_limit": 1000.0}, "pull_off": {"stress_limit": 0.001}})
    code, out, err = cli(run, "--gate", yaml_path=y)
    assert code == 3, err
    g = result(run)["gate"]
    assert g["failed"] == ["stress:pull_off"] and met(g, "stress", "hang") is True and met(g, "stress", "pull_off") is False
    from topopt.verify.report import plain_lines
    plain = plain_lines(result(run))                                                  # the gate path (no --allowable-mpa): margin, verdict
    assert any(x.startswith("case hang") and "The gate limit is 1000 MPa" in x and "margin " in x for x in plain)
    assert "gate: FAILED (stress:pull_off)" in plain
    assert g["checks"]["stress"]["pull_off"]["source"] == "load_cases[pull_off].stress_limit"
    # buckling.min_factor turns the check on; a limits row joins; all met
    ok = hook_yaml(tmp_path, cases={"hang": {"stress_limit": 1000.0}, "pull_off": {"stress_limit": 1000.0}},
                   buckling={"min_factor": 0.5}, limits=ROWS)
    run2 = copy_run(tmp_path / "run2")
    code, out, err = cli(run2, "--gate", yaml_path=ok)
    assert code == 0, err
    res = result(run2)
    assert res["buckling_check"]["on"] is True and "gate" in res["buckling_check"]["source"] and res["buckling"]["status"] == "done"
    g = res["gate"]
    assert g["passed"] is True and set(g["checks"]) == {"stress", "buckling", "displacement"}
    assert g["checks"]["buckling"]["hang"]["limit"] == 0.5 and g["checks"]["buckling"]["hang"]["value"] > 100          # a very stiff hook
    row = g["checks"]["displacement"]["displacement:tip_pull:pull_off"]
    assert row["met"] is True and row["limit"] == 0.05 and row["value"] == res["motion"]["rows"][0]["tet_value"]
    assert {n["check"] for n in g["not_gated"]} == {"alignment"}
    # a factor below the minimum, and a row above its limit, each fail the gate on their own
    bad = hook_yaml(tmp_path, cases={"hang": {"stress_limit": 1000.0}, "pull_off": {"stress_limit": 1000.0}},
                    buckling={"min_factor": 1e9}, limits={"displacement": [dict(ROWS["displacement"][0], limit_mm=0.021)]})
    run3 = copy_run(tmp_path / "run3")
    code, out, err = cli(run3, "--gate", yaml_path=bad)
    assert code == 3, err
    assert result(run3)["gate"]["failed"] == ["buckling:hang", "buckling:pull_off", "displacement:displacement:tip_pull:pull_off"]


@needs_gmsh
@needs_skfem
@needs_recording
def test_allowable_mpa_over_a_limit_of_the_yaml_keeps_the_replaced_limit_in_the_result_the_line_and_the_plain_text(tmp_path):
    """D-2 on a real run: one tighter and one looser override of the same YAML limits."""
    from topopt.verify.report import plain_lines
    y = hook_yaml(tmp_path, cases={"hang": {"stress_limit": 1000.0}, "pull_off": {"stress_limit": 0.001}})
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--gate", "--allowable-mpa", 5000, yaml_path=y)         # above both YAML limits: a pass that comes from a looser number
    assert code == 0, err
    res = result(run)
    st = res["gate"]["checks"]["stress"]
    assert st["pull_off"]["replaced_limit"] == 0.001 and st["pull_off"]["loosened"] is True and st["pull_off"]["met"] is True
    assert st["pull_off"]["replaced_source"] == "load_cases[pull_off].stress_limit" and "--allowable-mpa" in st["pull_off"]["source"]
    assert st["hang"]["replaced_limit"] == 1000.0 and st["hang"]["loosened"] is True
    assert "gate: stress pull_off" in out and "The limit replaces 0.001 MPa of load_cases[pull_off].stress_limit; the new limit is looser" in out
    assert "the new limit is looser" in "\n".join(plain_lines(res))                   # the plain text has the fact too
    run2 = copy_run(tmp_path / "run2")
    code, out, err = cli(run2, "--gate", "--allowable-mpa", 0.0005, yaml_path=y)      # below both: tighter, and a fail
    assert code == 3, err
    st = result(run2)["gate"]["checks"]["stress"]
    assert st["pull_off"]["loosened"] is False and st["hang"]["loosened"] is False and "the new limit is tighter" in out


@needs_gmsh
@needs_skfem
@needs_recording
def test_a_check_that_the_gate_asked_for_and_that_could_not_run_fails_the_gate(tmp_path, monkeypatch):
    from topopt.verify import buckling_p2
    run = copy_run(tmp_path / "run")
    y = hook_yaml(tmp_path, cases={"hang": {"stress_limit": 1000.0}, "pull_off": {"stress_limit": 1000.0}}, buckling={"min_factor": 0.5})

    def boom(*a, **k):
        raise RuntimeError("no eigen solver")
    monkeypatch.setattr(buckling_p2, "run_check", boom)
    code, out, err = cli(run, "--gate", yaml_path=y)
    assert code == 3
    res = result(run)
    assert res["buckling"]["status"] == "error"                                       # verify went on, as before
    g = res["gate"]
    assert g["passed"] is False and met(g, "buckling", "hang") is None and "no eigen solver" in g["checks"]["buckling"]["hang"]["reason"]
    assert met(g, "stress", "hang") is True and "NOT MET (could not run)" in out


@needs_gmsh
@needs_skfem
@needs_recording
def test_the_gate_reads_the_finest_level_and_gives_the_change_from_the_level_before(tmp_path):
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--gate", "--allowable-mpa", 1000, "--levels", "3.0,2.5")
    assert code == 0, err
    res = result(run)
    g = res["gate"]
    assert g["mesh_size_mm"] == 2.5 and g["previous_level_mm"] == 3.0 and res["reference_level_mm"] == 2.5
    rec = g["checks"]["stress"]["pull_off"]
    assert rec["value"] == res["cases"]["pull_off"]["von_mises"]["percentile_by_volume_MPa"]
    assert rec["change_from_previous"] == pytest.approx(res["levels"]["relative_change_first_to_last"]["cases"]["pull_off"]["p99_9_by_volume_MPa"])


@needs_skfem                  # the command meets the missing extra first (exit code 2), not the gate
@needs_recording
def test_a_gate_with_no_limit_and_a_gate_against_no_buckling_are_exit_code_1(tmp_path):
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--gate")
    assert code == 1 and "no limit to check" in err
    y = hook_yaml(tmp_path, buckling={"min_factor": 2.0})
    code, out, err = cli(run, "--gate", "--no-buckling", yaml_path=y)
    assert code == 1 and "--no-buckling" in err


@needs_gmsh
@needs_skfem
@needs_recording
def test_without_the_switch_the_result_has_no_gate_key_and_the_old_exit_code(tmp_path):
    """The key sets of a plain run are pinned in test_verify_buckling_run.py (the B3 test); here: no `gate`, and a limits row is a fact."""
    run = copy_run(tmp_path / "run")
    y = hook_yaml(tmp_path, cases={"pull_off": {"stress_limit": 0.001}}, buckling={"min_factor": 1e9, "check": True},
                  limits={"displacement": [dict(ROWS["displacement"][0], limit_mm=0.021)]})
    code, out, err = cli(run, yaml_path=y)
    res = result(run)
    assert code == 0 and "gate" not in res and res["exit_code"] == 0 and "no_allowable" in res
    assert "gate:" not in out and res["motion"]["status"] == "done" and "met" not in res["motion"]["rows"][0]
    code, out, err = cli(run, "--allowable-mpa", 0.001, yaml_path=y)
    assert code == 3 and "gate" not in result(run)                                      # the old allowable still decides


# --------------------------------------------------------------------------- #
# slow: a column, with buckling.min_factor above and below its Euler factor
# --------------------------------------------------------------------------- #

EULER_E, EULER_NU, COLUMN_L, SECTION = 110000.0, 0.34, 100.0, 5.0


def column_case(root: Path, min_factor: float) -> tuple[Path, Path]:
    """A run folder of a column (5 x 5 x 100 mm, fixed foot, load P_cr on the top face) and its YAML. The final.stl is the column."""
    import trimesh
    from test_feedback import box, fake_run
    d = root / "in"
    d.mkdir(parents=True)
    box(d / "design.stl", (0, 0, 0), (SECTION, SECTION, COLUMN_L))
    box(d / "foot.stl", (0, 0, -3), (SECTION, SECTION, 0))
    box(d / "top.stl", (0, 0, COLUMN_L), (SECTION, SECTION, COLUMN_L + 3))
    pcr = np.pi ** 2 * EULER_E * (SECTION ** 4 / 12.0) / (2.0 * COLUMN_L) ** 2
    doc = {"name": "column", "regions": {"design": "design.stl", "fixed": ["foot.stl"], "load": {"top": "top.stl"}},
           "load_cases": [{"name": "axial", "weight": 1.0, "forces": [{"region": "top", "force": [0.0, 0.0, -float(pcr)]}]}],
           "material": {"E": EULER_E, "nu": EULER_NU, "density": 1.24},
           "buckling": {"min_factor": min_factor, "modes": 3},
           "settings": {"voxel_size": 1.0, "volume_fraction": 0.5, "iterations": 5}}
    ypath = d / "column.yaml"
    ypath.write_text(yaml.safe_dump(doc, sort_keys=False))
    run = fake_run(root / "run", ypath)
    col = trimesh.creation.box(extents=(SECTION, SECTION, COLUMN_L))
    col.apply_translation(-col.bounds[0])
    col.export(run / "result" / "final.stl")
    return run, ypath


@needs_gmsh
@needs_skfem
@pytest.mark.slow
def test_a_column_passes_a_buckling_minimum_below_its_euler_factor_and_fails_one_above(tmp_path):
    """The load is the Euler load of the fixed-free column, so the lowest tet factor is about 1 (within 1.5 %, test_verify_buckling.py)."""
    from topopt.cli import main
    seen = {}
    for name, min_factor, want_code, want_met in (("below", 0.9, 0, True), ("above", 1.1, 3, False)):
        run, ypath = column_case(tmp_path / name, min_factor)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["verify", str(run), "--yaml", str(ypath), "--mesh-size", "2.0", "--gate", "--no-sensitivity"])
        assert code == want_code, err.getvalue()
        g = result(run)["gate"]
        rec = g["checks"]["buckling"]["axial"]
        assert rec["met"] is want_met and rec["limit"] == min_factor and rec["source"] == "buckling.min_factor"
        assert g["passed"] is want_met
        assert abs(rec["value"] - 1.0) < 1.5e-2
        assert {n["check"] for n in g["not_gated"]} >= {"stress", "displacement", "alignment"}        # the YAML has no other limit
        seen[name] = rec["value"]
    assert seen["below"] == pytest.approx(seen["above"])                                              # the same part, the same factor


# --------------------------------------------------------------------------- #
# The gate with joints (K1-A): a part with `fixed: []` that two joints hold
# --------------------------------------------------------------------------- #

@needs_gmsh
@needs_skfem
def test_the_gate_reads_stress_and_buckling_of_a_part_that_joints_hold(tmp_path):
    from test_verify_supports import _bar_run
    from topopt.verify import verify_run
    run = _bar_run(tmp_path)
    p = Path(json.loads((run / "run.json").read_text())["yaml_path"])
    d = yaml.safe_load(p.read_text())
    d["load_cases"][0]["stress_limit"] = 1000.0
    d["buckling"] = {"min_factor": 0.5, "modes": 3}
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    res = verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False, gate=True)
    assert res["regions"]["fixed"] == [] and res["supports"]["joints"] and res["exit_code"] == 0, res.get("gate")
    g = res["gate"]
    assert g["passed"] is True and set(g["checks"]) == {"stress", "buckling"}
    st, bk = g["checks"]["stress"]["down"], g["checks"]["buckling"]["down"]
    assert st["met"] is True and 0 < st["value"] < 1000.0 and st["source"] == "load_cases[down].stress_limit"
    case = res["buckling"]["cases"][0]
    assert "rigid_body" in case          # K1-C: two joints and no fixed node, the lowest mode is a swing on the supports, so the rule ran
    assert bk["value"] == case["load_factors"][0] == res["buckling"]["lowest_load_factor"]["value"]       # the gate reads the measured value
    assert bk["projected_value"] == case["rigid_body"]["load_factors"][0] > bk["value"] and bk["value_source"] == "buckling.cases[down].load_factors[0]"
    assert bk["run_lowest_load_factor"] == res["buckling"]["lowest_load_factor"]
    assert bk["met"] is True and bk["value"] > 0.5
    d["load_cases"][0]["stress_limit"] = 1e-6
    d["buckling"]["min_factor"] = 1e12
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    res = verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False, gate=True)
    assert res["exit_code"] == 3 and res["gate"]["failed"] == ["stress:down", "buckling:down"]


# --------------------------------------------------------------------------- #
# K1-C: the rigid-body rule and the value that the gate reads
# --------------------------------------------------------------------------- #

def soft_joint_level(rule="ok"):
    """A part on soft joints: the lowest factor 0.13 is a motion of the supports. Without the rigid-body motions it is 3.9.

    `rule` is `"ok"` (the projected solve gave factors), `"failed"` (it gave none) or `"off"` (the rule did not run: the case has no key
    `rigid_body`, as `buckling_p2` writes it).
    """
    lv = level(factors=(8.0, 9.0))
    case = {"name": "a", "load_factors": [0.13, 0.14]}
    if rule == "ok":
        case["rigid_body"] = {"applied": True, "status": "ok", "load_factors": [3.9, 12.5]}
    elif rule == "failed":
        case["rigid_body"] = {"applied": True, "status": "time_budget", "load_factors": []}
    lv["buckling"]["cases"][0] = case
    lv["buckling"]["lowest_load_factor"] = {"case": "a", "value": 0.13}
    return lv


MIN2 = dict(buckling=dict(limit=2.0, source="buckling.min_factor"))


def test_when_the_rigid_body_rule_ran_the_gate_reads_the_lowest_factor_and_the_projected_one_is_a_fact():
    """Decision 1 of the K1 stack: a higher number in place of the measured one would loosen buckling.min_factor, a requirement."""
    g = gate.evaluate(soft_joint_level(), limits(**MIN2), percentile=99.9)
    rec = g["checks"]["buckling"]["a"]
    assert rec["value"] == 0.13 and rec["met"] is False and g["passed"] is False and g["failed"] == ["buckling:a"]
    assert rec["value_source"] == "buckling.cases[a].load_factors[0]" and "read" not in rec
    assert rec["run_lowest_load_factor"] == {"case": "a", "value": 0.13}
    assert rec["projected_value"] == 3.9 and rec["projected_source"] == "buckling.cases[a].rigid_body.load_factors[0]"
    assert "the part itself buckles at 3.9" in rec["projected_reason"] and "0.13" in rec["projected_reason"] and "loosen" in rec["projected_reason"]
    other = g["checks"]["buckling"]["b"]
    assert other["value"] == 9.0 and not any(k.startswith("projected") for k in other)
    json.dumps(g)


def test_when_the_projected_solve_gave_no_factor_the_gate_entry_says_why():
    """D6 of the review of #137: the reason was only in the facts of the case, and the test was an identity test on a string."""
    g = gate.evaluate(soft_joint_level("failed"), limits(**MIN2), percentile=99.9)
    rec = g["checks"]["buckling"]["a"]
    assert rec["value"] == 0.13 and rec["projected_value"] is None
    assert "time_budget" in rec["projected_reason"] and "no projected value" in rec["projected_reason"] and "0.13" in rec["projected_reason"]


def test_a_part_on_which_the_rule_did_not_run_has_no_projected_keys():
    rec = gate.evaluate(soft_joint_level("off"), limits(**MIN2), percentile=99.9)["checks"]["buckling"]["a"]
    assert rec["value"] == 0.13 and not any(k.startswith("projected") for k in rec)
    rec = gate.evaluate(level(), limits(), percentile=99.9)["checks"]["buckling"]["a"]
    assert rec["value"] == 8.0 and rec["value_source"] == "buckling.cases[a].load_factors[0]" and not any(k.startswith("projected") for k in rec)


def test_one_gate_line_holds_the_gate_value_the_value_of_the_run_and_the_projected_fact():
    """D2 of the review of #137: three numbers for one case, each with its key, on one line."""
    g = gate.evaluate(soft_joint_level(), limits(**MIN2), percentile=99.9)
    assert g["checks"]["buckling"]["a"]["run_lowest_load_factor"] == {"case": "a", "value": 0.13}
    assert g["checks"]["buckling"]["b"]["run_lowest_load_factor"] == {"case": "a", "value": 0.13}      # the headline is the lowest over all cases
    (line,) = [t for t in gate.gate_lines(g) if t.startswith("gate: buckling a:")]
    assert "0.13 (needs >= 2" in line and "NOT MET" in line
    assert ("Value from buckling.cases[a].load_factors[0]; the lowest of all cases is 0.13 (case a, buckling.lowest_load_factor)" in line)
    assert ("Without the rigid-body motions of the part the value is 3.9 (buckling.cases[a].rigid_body.load_factors[0]); "
            "a fact, the gate does not read it") in line
    (other,) = [t for t in gate.gate_lines(g) if t.startswith("gate: buckling b:")]
    assert "the lowest of all cases is 0.13 (case a" in other and "Without the rigid-body motions" not in other
    (failed,) = [t for t in gate.gate_lines(gate.evaluate(soft_joint_level("failed"), limits(**MIN2), percentile=99.9))
                 if t.startswith("gate: buckling a:")]
    assert "The solve without the rigid-body motions of the part gave no value (buckling.cases[a].rigid_body.load_factors[0])" in failed


def test_the_report_line_of_a_case_names_the_projected_value_as_a_fact():
    from topopt.verify import report
    lv = soft_joint_level()
    bk = dict(lv["buckling"], solver="superlu", modes=2, seconds=1.0, facts=[], options=[])
    bk["cases"] = [dict(c, solves=3, seconds=0.5) for c in bk["cases"]]
    lines = report.buckling_lines(bk)
    assert lines[1].startswith("  case a: lowest load factor 0.13 (without the rigid-body motions of the part: 3.9, a fact);")
    assert "without the rigid-body" not in lines[2]


@needs_skfem
def test_the_gate_value_does_not_rise_when_the_support_gets_softer():
    """The review of #137 (D1): the floating column of `test_verify_buckling_modes.py` on a beam from stiff to soft.

    The factor without the rigid-body motions is 3.86 for every beam (it does not see the support), so a gate that read it passed the soft
    beams and failed the stiff ones. The gate value is the lowest load factor: it falls with the beam, and a limit that a stiff beam fails is
    failed by every softer beam.
    """
    from test_verify_buckling_modes import Floating
    beams = [(40.0, 10.0, 110000.0), (10.0, 20.0, 110000.0), (10.0, 30.0, 40000.0), (10.0, 30.0, 20000.0), (10.0, 30.0, 8000.0), (10.0, 30.0, 2100.0)]
    lowest, records = [], []
    for d, length, e_beam in beams:
        col = Floating(d, length, e_beam, cells=(1, 1, 8))
        try:
            rec, _ = col.run(modes=4)
        finally:
            col.close()
        lowest.append(rec["load_factors"][0])
        records.append(rec)

    def gated(rec, limit):
        lv = dict(size_mm=1.0, cases={}, buckling=dict(status="done", cases=[dict(rec, name="a")]))
        lim = limits(stress={"a": dict(limit=None, source=None)}, rows=[], buckling=dict(limit=limit, source="buckling.min_factor"),
                     applies_load={"a": True})
        return gate.evaluate(lv, lim, percentile=99.9)["checks"]["buckling"]["a"]

    values = [gated(rec, 0.5)["value"] for rec in records]
    assert values == lowest and all(a > b for a, b in zip(values, values[1:])), values
    for limit in (0.2, 0.5, 0.7, 0.95):
        met = [gated(rec, limit)["met"] for rec in records]
        assert met == sorted(met, reverse=True), (limit, met)          # once a beam fails, every softer beam fails
    soft = [r for r in records if "rigid_body" in r]
    assert len(soft) >= 2                                              # the two softest beams ran the rule
    projected = [gated(r, 0.5)["projected_value"] for r in soft]
    assert projected[0] == pytest.approx(projected[1], rel=1e-3) and min(projected) > 3.0 * max(values)       # a fact, and not the gate value
