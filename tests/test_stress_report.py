"""Tests for the stress section of metrics.json (A2) and the B-051 accuracy data.

The accuracy data `bench/stress_accuracy.json` is checked three ways: the rule in `ACCURACY` equals the rule in the
file, the rule follows from the sizes in the file, and two rows (beam 4 voxels, hole 4 voxels) are made again.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_optimize import make_case
from topopt import stress_report as sr
from topopt.cli import main
from topopt.config import load_config
from topopt.problem import build_problem

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "stress_accuracy.json"


def load_bench():
    spec = importlib.util.spec_from_file_location("stress_accuracy_bench", ROOT / "bench" / "stress_accuracy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- statistics ----------------------------------------------------------------


def test_percentile_by_volume_is_the_stepwise_rule():
    v = np.arange(1, 1001, dtype=float)
    np.random.default_rng(0).shuffle(v)
    assert sr.percentile_by_volume(v, 50.0) == 500.0       # the first sample whose cumulative weight reaches p
    assert sr.percentile_by_volume(v, 99.0) == 990.0
    assert sr.percentile_by_volume(v, 99.9) in (999.0, 1000.0)   # float rounding of 0.999: one step
    assert sr.percentile_by_volume(v, 100.0) == 1000.0
    assert sr.percentile_by_volume([3.0], 99.9) == 3.0


def test_member_thickness_of_flat_members():
    """A flat member of n voxels gives n (even n) or n + 1 (odd n), up to a saturation near 6 or 7."""
    for n, expect in ((1, 2.0), (2, 2.0), (3, 4.0), (4, 4.0), (5, 6.0), (6, 6.0)):
        solid = np.zeros((12, 12, n + 6), bool)
        solid[:, :, 3:3 + n] = True
        ijk = np.array([[6, 6, 3 + n // 2], [6, 6, 3 + (n - 1) // 2]])
        assert sr.member_thickness(solid, ijk).max() == expect, n


def test_the_committed_accuracy_gives_no_verdict():
    """Decided: no verdict without data. The bands stay as measured ranges (REVIEW M3)."""
    for stat in ("p99_9", "peak"):
        acc = sr.ACCURACY[stat]
        assert acc["verdict"] is False and acc["reason"]
        v = sr.verdict(stat, 1.0, 10.0, {"met": True, "p10_voxels": 4.0}, 20000)
        assert v["pass"] is None and v["status"].startswith("no verdict for this statistic")
    band = sr.ACCURACY["p99_9"]
    assert (band["under_pct"], band["over_pct"]) == (59, 57)          # F1-8c: the union of the bench (54, 13) and the public cells
    assert (band["sources"]["bench"]["under_pct"], band["sources"]["bench"]["over_pct"]) == (54, 13)
    db = band["data_base"]
    assert (db["bench"]["shapes"], db["bench"]["real_parts"], db["bench"]["rows"]) == (3, 1, 11)
    assert (db["public"]["parts"], db["public"]["cells"], db["public"]["cells_where_p99_9_equals_the_peak"]) == (5, 51, 31)
    assert "lower end" in band["reason"] and "upper end" in band["reason"] and sr.PUBLIC_PARTS_FILE in band["reason"]
    v = sr.verdict("p99_9", 1.0, 10.0, {"met": True, "p10_voxels": 4.0}, 300)
    assert v["status"].startswith("p99.9 equals the peak at this voxel count (300 voxels")


def test_verdict_rule_for_an_entry_with_a_verdict(monkeypatch):
    """The pass and fail logic stays for the day a verdict is enabled (the band is the union of the bench and the public cells)."""
    acc = dict(sr.ACCURACY["p99_9"], verdict=True)
    monkeypatch.setattr(sr, "ACCURACY", {"p99_9": acc, "peak": sr.ACCURACY["peak"]})
    ok = {"met": True, "p10_voxels": 4.0}
    lim = 10.0
    passed = sr.verdict("p99_9", 4.0, lim, ok, 20000)        # 4 / (1 - 0.59) = 9.76 <= 10
    assert passed["pass"] is True and passed["error_band_pct"]["reference_high_MPa"] == pytest.approx(4.0 / 0.41)
    assert passed["error_band_pct"]["sources"] == sr.ACCURACY["p99_9"]["sources"]       # F1-8c: the band names the source of each end
    inside = sr.verdict("p99_9", 5.0, lim, ok, 20000)        # 5 / 0.41 = 12.2 > 10 and 5 / 1.57 = 3.2 < 10
    assert inside["pass"] is None and "inside the error band" in inside["status"]
    failed = sr.verdict("p99_9", 20.0, lim, ok, 20000)       # 20 / 1.57 = 12.7 > 10
    assert failed["pass"] is False
    assert sr.verdict("p99_9", 12.0, lim, {"met": False, "p10_voxels": 2.0}, 20000)["pass"] is None
    assert "thinner than the band was measured for" in sr.verdict(
        "p99_9", 1.0, lim, {"met": False, "p10_voxels": 2.0}, 20000)["status"]
    # a p99.9 that equals the peak has no verdict, whatever the band says
    eq = sr.verdict("p99_9", 1.0, lim, ok, 300)
    assert eq["pass"] is None and eq["status"].startswith("no verdict: p99.9 equals the peak")


def test_p99_9_equals_the_peak_below_5000_voxels():
    """Step-wise p99.9 of n equal voxels rests on the top 0.1 % of the volume: fewer than 5 voxels below n = 5000."""
    assert sr.top_tail_voxels(5000) == pytest.approx(5.0) and sr.top_tail_voxels(300) == pytest.approx(0.3)
    assert sr.p99_9_equals_peak(300) and sr.p99_9_equals_peak(4999) and not sr.p99_9_equals_peak(5000)
    assert not sr.p99_9_equals_peak(0)
    v = np.random.default_rng(1).uniform(1, 2, 999)
    assert sr.percentile_by_volume(v) == v.max()                  # for n < 1000 the value is the largest
    assert "300 voxels" in sr.equals_peak_text(300) and "fewer than 5" in sr.equals_peak_text(300)


def test_equals_the_peak_text_gives_the_two_values_where_they_differ():
    """R1-N1: from 1000 voxels p99.9 is the voxel at rank 2 or more from the top, so it is not the peak itself."""
    assert [sr.p99_9_rank_from_top(n) for n in (1, 300, 999)] == [1, 1, 1]                 # below 1000 voxels: the peak
    assert sr.p99_9_rank_from_top(1375) == 2 and sr.p99_9_rank_from_top(4999) >= 2
    rng = np.random.default_rng(2)
    for n in (300, 1000, 1375, 2000, 4999, 5000):                                           # the rank follows the function
        v = rng.uniform(1, 2, n)
        assert sr.percentile_by_volume(v) == np.sort(v)[-sr.p99_9_rank_from_top(n)]
    assert "rank" not in sr.equals_peak_text(300, 1.5, 1.5)                      # below 1000 voxels the two are equal
    t = sr.equals_peak_text(1375, 5.99, 6.18)
    assert "1375 voxels" in t and "p99.9 is the voxel at rank 2 from the top: 5.99 MPa; peak 6.18 MPa" in t
    assert "rank" not in sr.equals_peak_text(1375)                               # no values, no rank text
    phys = {"voxels": 1375, "p99_9_equals_peak": True, "p99_9_by_volume_MPa": 5.99, "peak_MPa": 6.18,
            "allowable_mpa": None}
    e = sr.safety_entry("p99_9", 5.0, 5.99, phys, {"met": False, "p10_voxels": 2.0})
    assert "rank 2 from the top" in e["status"] and "peak 6.18 MPa" in e["status"]
    text = sr.safety_text({"stress_limit_MPa": 5.0}, phys)
    assert "the peak is 6.18 MPa (physical, full E, 1375 design voxels with rho >= 0.5). p99.9 equals the peak" in text
    assert "(p99.9 is the voxel at rank 2 from the top: 5.99 MPa; peak 6.18 MPa)" in text
    differ = sr.safety_text({"stress_limit_MPa": 5.0}, {**phys, "p99_9_equals_peak": False})
    assert "the peak is 6.18 MPa and p99.9 by volume is 5.99 MPa (physical, full E, 1375 design voxels with rho >= 0.5)." in differ
    assert "(physical, full E, 300 design voxels with rho >= 0.5)." in sr.safety_text(
        {"stress_limit_MPa": 5.0}, {"voxels": 300, "p99_9_equals_peak": True, "p99_9_by_volume_MPa": 6.0, "peak_MPa": 6.0})


def fake_case(met, grey, thr, lim=5.0, p10=2.0, above=77, voxels=171):
    """A case entry of the stress section with the keys that `case_summary` reads."""
    entry = {"stress_limit_MPa": lim,
             "relaxed": {"met": met, "peak_MPa": 4.57, "row_g": 0.0005 if met else 0.2},
             "physical": {"voxels": voxels, "peak_MPa": grey, "voxels_above_limit": above},
             "member_check": {"p10_voxels": p10}}
    if thr is not None:
        entry["physical_thresholded"] = {"peak_MPa": thr}
    return entry


def test_the_summary_puts_the_relaxed_result_and_the_voxel_estimates_together():
    """R1-1 and R1-3: "met" is about the relaxed measure. The text gives the range of the two voxel estimates."""
    t = sr.case_summary("pull_off", fake_case(True, 6.19, 8.01))                  # the committed example, 2.4 mm
    assert t.startswith("the optimizer row is met for the relaxed stress (relaxed peak 4.57 MPa, limit 5 MPa, g +0.0005).")
    assert "peak: 6.19 to 8.01 MPa (thresholded design 8.01, grey design 6.19); limit 5 MPa" in t and t.count("Voxel estimate") == 1
    assert "Both voxel estimates are above the limit (the limit is 0.81 times the grey peak; 77 of 171 voxels" in t
    assert '"Met" is about the relaxed measure, not about the part.' in t
    assert "have no accuracy band for the peak: the voxel peak was 5 to 87 % below the reference in the measured cases (" in t
    assert "lowest -87.14 % from the 5 public parts: ladle, full_scoop, run type MF, voxel 3.25 mm, variant S2" in t       # F1-8c
    assert "highest -4.82 % from the bench: beam, 16 voxels through the thickness" in t
    assert "2 voxels thick (p10)" in t and "No pass or fail." in t and "options: a lower stress_limit" in t
    t = sr.case_summary("pull_off", fake_case(True, 6.18, 4.08))                  # the 1.2 mm run: the estimates differ in side
    assert "peak: 4.08 to 6.18 MPa (thresholded design 4.08, grey design 6.18); limit 5 MPa" in t
    assert "on different sides of the limit" in t and '"Met" is about the relaxed measure' in t
    t = sr.case_summary("hang", fake_case(True, 0.89, 1.11, lim=1.5, above=0))
    assert "Both voxel estimates are at or below the limit (the limit is 1.7 times the grey peak)" in t
    assert t.count('"Met" is about the relaxed measure, not about the part.') == 1 and "options:" not in t       # F4-29: said for every case with a relaxed row
    assert t.endswith(" option: " + sr.VERIFY_OPTION)                                                            # verify has not run: one option
    t = sr.case_summary("pull_off", fake_case(False, 6.19, None))                 # not met, no thresholded estimate
    assert t.startswith("the optimizer row is NOT met") and "(grey design); limit 5 MPa" in t and "options:" not in t
    assert '"Met" is about the relaxed measure' in t and t.endswith(" option: " + sr.VERIFY_OPTION)
    t = sr.case_summary("x", {"stress_limit_MPa": 5.0, "relaxed": None, "physical": None,
                              "input_changed": {"fact": "the inputs of this run changed"}})
    assert t == "the inputs of this run changed"
    assert sr.summary_log_lines({"cases": {"a": {"summary": "S"}, "b": {}}}) == ["fact: stress a: S"]
    assert sr.summary_log_lines(None) == [] and sr.summary_log_lines({"error": "x"}) == []


def p2_check(value=12.5, verified="final.stl", case="pull_off"):
    return {"file": "verify/verify.json", "field": f"cases.{case}.von_mises.p99_9_by_volume_MPa", "found": value is not None,
            "p2_p99_9_by_volume_MPa": value, "verified_file": verified}


def test_the_summary_ends_with_the_p2_value_and_its_ratio_to_the_limit_when_verify_has_run():
    """F4-29: the stress of the part from `topopt verify` next to "met": the value, its ratio to the limit and its source (file and field)."""
    e = fake_case(True, 6.19, 8.01)
    e["independent_check"] = p2_check()
    t = sr.case_summary("pull_off", e)
    tail = ("The stress of the part from the independent check: P2 p99.9 by volume 12.5 MPa, 2.5 times the limit 5 MPa (tets of final.stl, quadratic elements; "
            "source verify/verify.json, cases.pull_off.von_mises.p99_9_by_volume_MPa).")
    assert tail in t and "option: run verify" not in t and "options: a lower stress_limit, or a finer voxel size" in t and t.endswith("a finer voxel size")
    assert t.index('"Met" is about the relaxed measure') < t.index(tail)                      # the P2 value follows "met"
    e["independent_check"] = p2_check(2.0, "part_only.stl")                                    # below the limit; a part-only mesh
    t = sr.case_summary("pull_off", e)
    assert "P2 p99.9 by volume 2 MPa, 0.4 times the limit 5 MPa (tets of part_only.stl" in t
    e["independent_check"] = p2_check(None)                                                    # found false: the option, not a number
    t = sr.case_summary("pull_off", e)
    assert "options: a lower stress_limit, a finer voxel size, or " + sr.VERIFY_OPTION in t and "P2 p99.9" not in t
    e = fake_case(True, 4.0, 4.5, above=0)                                                     # met, both voxel estimates below the limit: no other options
    e["independent_check"] = p2_check(7.5)
    t = sr.case_summary("pull_off", e)
    assert "P2 p99.9 by volume 7.5 MPa, 1.5 times the limit 5 MPa" in t and "options:" not in t and t.endswith(").")
    assert sr.case_summary("x", {**e, "relaxed": None}) .count("P2") == 0                      # no relaxed row: no "met", no tail


def test_a_met_row_with_the_relaxed_peak_above_the_limit_says_so_in_the_summary():
    """F4-34: the first sentence names the flag; the sentence about "met" and the way to the stress of the part follow."""
    e = fake_case(True, 6.19, 8.01)
    e["relaxed"].update(peak_MPa=6.19, peak_above_limit=True)
    assert "g +0.0005); the relaxed peak is above the limit (`relaxed.peak_above_limit`). " in sr.case_summary("pull_off", e)
    e["relaxed"]["peak_above_limit"] = False
    assert "relaxed.peak_above_limit" not in sr.case_summary("pull_off", e)


def test_a_case_with_changed_inputs_or_no_voxel_still_says_what_met_is():
    t = sr.case_summary("x", {"stress_limit_MPa": 5.0, "relaxed": {"met": True, "peak_MPa": 4.0, "row_g": 0.0}, "physical": None,
                              "input_changed": {"fact": "the inputs of this run changed."}})
    assert t.endswith('the inputs of this run changed. "Met" is about the relaxed measure, not about the part.')
    t = sr.case_summary("x", {"stress_limit_MPa": 5.0, "relaxed": {"met": True, "peak_MPa": 4.0, "row_g": 0.0}, "physical": {"voxels": 0},
                              "independent_check": p2_check(6.0, case="x")})
    assert "No design voxel has rho >= 0.5" in t and '"Met" is about the relaxed measure' in t and "P2 p99.9 by volume 6 MPa, 1.2 times the limit 5 MPa" in t


def test_safety_factor_entries_have_a_range_or_the_status_no_band():
    """p99.9: a range from the measured band; the peak: the number with 'no accuracy band' (REVIEW M6)."""
    member = {"met": True, "p10_voxels": 4.0}
    big = {"voxels": 20000, "p99_9_equals_peak": False}
    e = sr.safety_entry("p99_9", 10.0, 2.0, big, member)
    assert e["value"] == pytest.approx(5.0)
    lo, hi = e["range_estimate"]
    assert lo == pytest.approx(10.0 * 0.41 / 2.0) and hi == pytest.approx(10.0 * 1.57 / 2.0)    # band -59 % .. +57 %
    assert e["status"].startswith("estimate") and "no verdict" in e["status"]
    assert "bench 3 shapes, 1 real part; 5 public parts, 51 cells" in e["status"]
    assert e["member_gate"].startswith("met")
    assert sr.safety_entry("p99_9", 10.0, 2.0, big, {"met": False, "p10_voxels": 2.0})["member_gate"].startswith("not met")
    pk = sr.safety_entry("peak", 10.0, 2.0, big, member)
    assert pk["value"] == pytest.approx(5.0) and pk["range_estimate"] is None and pk["status"] == "no accuracy band"
    assert "below the reference in every measured case" in pk["note"] and "can be lower" in pk["note"]
    assert "lowest -87.14 % from the 5 public parts" in pk["note"] and "highest -4.82 % from the bench" in pk["note"]      # F1-8c
    small = sr.safety_entry("p99_9", 10.0, 2.0, {"voxels": 300, "p99_9_equals_peak": True}, member)
    assert small["range_estimate"] is None and "equals the peak at this voxel count (300 voxels" in small["status"]
    assert sr.safety_entry("peak", 10.0, 0.0, big, member) is None


def test_no_accuracy_data_no_verdict(monkeypatch):
    """While no data is committed for a statistic, the text is 'estimate, accuracy not measured'."""
    monkeypatch.setattr(sr, "ACCURACY", {})
    for stat in ("peak", "p99_9"):
        v = sr.verdict(stat, 1.0, 10.0, {"met": True, "p10_voxels": 4.0})
        assert v == {"status": "estimate, accuracy not measured", "pass": None, "error_band_pct": None}


# --- the accuracy data (B-051) ---------------------------------------------------


def test_accuracy_table_equals_the_data_file():
    doc = json.loads(DATA.read_text())
    rule = {k: v for k, v in doc["verdict_rule"].items() if k != "error_definition"}
    assert rule.keys() == sr.ACCURACY.keys()
    for stat, entry in rule.items():
        for key, value in entry.items():
            got = sr.ACCURACY[stat][key]
            assert got == pytest.approx(value) if isinstance(value, float) else got == value, (stat, key)


def test_rule_follows_from_the_sizes_in_the_file():
    doc = json.loads(DATA.read_text())
    bench = load_bench()
    pp = bench.public_parts()
    public = pp.load()["cells"]         # F1-8c: the band is the union of the bench rows and the public cells
    again = bench.derive_rule(doc["beam"]["sizes"], doc["hole"]["sizes"], doc["cross_check"], public)
    assert again == doc["verdict_rule"]


def test_data_file_covers_the_study():
    doc = json.loads(DATA.read_text())
    assert [r["voxels_through_thickness"] for r in doc["beam"]["sizes"]] == [1, 2, 4, 8, 16]
    assert [r["voxels_across_hole"] for r in doc["hole"]["sizes"]] == [4, 8, 16]
    for r in doc["beam"]["sizes"] + doc["hole"]["sizes"]:
        assert r["active_voxels"] <= doc["guard_active_voxels"] == 100000     # the voxel-count guard
    # the error falls with the voxel size (the sampling depth and the stiffness of hex8)
    peaks = [abs(r["peak_error_pct"]) for r in doc["beam"]["sizes"]]
    assert peaks == sorted(peaks, reverse=True)
    # element error at the voxel centres is far smaller than the surface error: the hex8 itself is close at 8+ voxels
    assert abs(doc["beam"]["sizes"][3]["peak_error_at_voxel_centres_pct"]) < 1.5
    assert set(doc["cross_check"]["cases"]) == {"hang", "pull_off"}


def test_two_rows_of_the_data_are_reproduced():
    """The beam with 4 voxels and the hole with 4 voxels, made again, equal the committed rows (rel 1e-6)."""
    doc = json.loads(DATA.read_text())
    bench = load_bench()
    row = bench.run_beam(4)
    ref = next(r for r in doc["beam"]["sizes"] if r["voxels_through_thickness"] == 4)
    for key in ("peak_MPa", "p99_9_MPa", "peak_error_pct", "p99_9_error_pct"):
        assert row[key] == pytest.approx(ref[key], rel=1e-6), key
    assert row["whole_design"]["peak_MPa"] == pytest.approx(ref["whole_design"]["peak_MPa"], rel=1e-6)
    row, _ = bench.run_hole(4)
    ref = next(r for r in doc["hole"]["sizes"] if r["voxels_across_hole"] == 4)
    for key in ("peak_MPa", "p99_9_MPa", "peak_error_pct", "p99_9_error_pct"):
        assert row[key] == pytest.approx(ref[key], rel=1e-6), key


# --- the section in metrics.json ---------------------------------------------------


def stress_yaml(tmp_path, iterations=24):
    p = make_case(tmp_path, h=1.0, iterations=iterations)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(objective="mass", volume_fraction=0.6, move_limit=0.05,
                         beta_schedule=[[0, 1], [10, 2], [18, 4]])
    d["load_cases"][0]["forces"][0]["force"] = [0, 0, -20.0]
    d["load_cases"][1]["forces"][0]["force"] = [10.0, 30.0, 0]
    d["load_cases"][0].update(stress_limit=10.0)
    d["load_cases"][1].update(allowable_mpa=40.0, safety_factor=4.0, allowable_source="a value of the test")
    p.write_text(yaml.safe_dump(d))
    return p


@pytest.fixture(scope="module")
def stress_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stress_run")
    out = tmp / "r"
    assert main(["run", str(stress_yaml(tmp)), "--out", str(out), "--quiet"]) == 0
    return out


def test_metrics_stress_section(stress_run):
    metrics = json.loads((stress_run / "result" / "metrics.json").read_text())
    s = metrics["stress"]
    assert "error" not in s and s["voxel_size_mm"] == 1.0 and s["parameters"] == {
        "q": 40.0, "relaxation_power": 0.5, "ks_scale": "auto", "step_acceptance": True}
    assert "No statistic has a verdict" in s["note"] and "target that the optimizer tries to meet" in s["note"]
    assert sr.PUBLIC_PARTS_FILE in s["note"] and "not a bound for other parts" in s["note"] and sr.PUBLIC_PARTS_FILE in s["accuracy_data"]   # F1-8
    assert "source of each end" in s["note"] and sr.BENCH_FILE in s["note"] and s["note"].endswith(sr.PART_STRESS_CHECK + ".")      # F1-8c
    assert s["voxel_estimates"]["thresholded_solve_s"] > 0 and "thresholded" in s["voxel_estimates"]["physical_thresholded"]
    # the report says whether the row was met, and gives the facts of step acceptance and of each stage end
    assert s["step_acceptance"]["passes"] == 24 and s["step_acceptance"]["tries_rejected"] >= 0
    assert [e["beta"] for e in s["stage_ends"]] == [1.0, 2.0, 4.0] and s["stage_ends"][-1]["pass"] == 24
    assert s["input_check"] is None                      # the problem of the run itself: no rebuild, no check
    assert set(s["cases"]) == {"down", "side"}
    down, side = s["cases"]["down"], s["cases"]["side"]
    # the limit as given, and its source
    assert (down["stress_limit_MPa"], down["limit_from"], down["allowable_mpa"]) == (10.0, "stress_limit", None)
    assert (side["stress_limit_MPa"], side["limit_from"], side["safety_factor"], side["allowable_mpa"]) == (
        10.0, "allowable_mpa / safety_factor", 4.0, 40.0)
    assert side["allowable_source"] == "a value of the test"
    for c in (down, side):
        rel, phys = c["relaxed"], c["physical"]
        assert rel["label"] == "relaxed, not physical" and rel["ks_MPa"] >= rel["peak_MPa"] > 0
        assert rel["iteration"] == 24
        assert phys["label"] == "full E, rho >= 0.5, design voxels" and phys["voxels"] > 0
        assert phys["peak_MPa"] >= phys["p99_9_by_volume_MPa"] > 0
        assert phys["peak_incl_forced_solid_MPa"] >= phys["peak_MPa"]
        assert rel["met"] == (rel["row_g"] <= 5e-3)
        assert rel["peak_above_limit"] is (rel["peak_MPa"] > c["stress_limit_MPa"] * (1 + rel["peak_above_limit_tol"]))      # F4-34: the test of the log line
        assert rel["peak_above_limit_tol"] == 5e-3 and "stress peak above the limit" in rel["peak_above_limit_means"]
        sf = c["achieved_safety_factor"]
        assert sf["against_limit"]["peak"]["value"] == pytest.approx(10.0 / phys["peak_MPa"])
        assert sf["against_limit"]["peak"]["status"] == "no accuracy band"
        assert sf["against_limit"]["p99_9"]["value"] == pytest.approx(10.0 / phys["p99_9_by_volume_MPa"])
        assert "Voxel estimate, not a check" in sf["text"] and "no accuracy band" in sf["text"]
        # the test part has a few hundred voxels: p99.9 is the peak, the report says so and gives the count
        assert phys["p99_9_equals_peak"] and phys["p99_9_by_volume_MPa"] == phys["peak_MPa"]
        assert f"{phys['voxels']} voxels" in phys["p99_9_note"] and "equals the peak" in sf["text"]
        assert sf["against_limit"]["p99_9"]["range_estimate"] is None
        assert 0 <= phys["voxels_above_limit"] <= phys["voxels"]
        assert phys["peak_over_limit"] == pytest.approx(phys["peak_MPa"] / c["stress_limit_MPa"])
        # R1-1: the key names stay; the new keys put the relaxed result and the voxel estimates together
        assert rel["met_means"].endswith("Not a statement about the part")
        assert c["summary"].startswith("the optimizer row is " + ("met" if rel["met"] else "NOT met"))
        assert "Voxel estimate of the part stress, peak:" in c["summary"] and "No pass or fail." in c["summary"]
        # R1-3: the thresholded design is the second estimate, with its own label; the voxels are the same ones
        thr = c["physical_thresholded"]
        assert thr["label"].startswith("thresholded design") and phys["design"].startswith("grey design")
        assert thr["voxels"] == phys["voxels"] and thr["peak_MPa"] > 0
        assert thr["peak_over_limit"] == pytest.approx(thr["peak_MPa"] / c["stress_limit_MPa"])
        lo, hi = sorted((phys["peak_MPa"], thr["peak_MPa"]))
        assert c["voxel_peak_range_MPa"]["low"] == lo and c["voxel_peak_range_MPa"]["high"] == hi
        assert f"thresholded design {thr['peak_MPa']:.3g}, grey design {phys['peak_MPa']:.3g}" in c["summary"]
        assert phys["volume_above_limit_mm3"] == pytest.approx(phys["voxels_above_limit"] * 1.0 ** 3)
        assert "estimate" in phys["above_limit_note"]
        assert set(c["verdict"]) == {"peak", "p99_9"}
        assert c["verdict"]["peak"]["pass"] is None and c["verdict"]["p99_9"]["pass"] is None   # no verdict
        assert sr.PUBLIC_PARTS_FACT in c["verdict"]["p99_9"]["status"]      # F1-8: the bench band comes with the public-part fact
        band = c["verdict"]["p99_9"]["error_band_pct"]                       # F1-8c: the union, with the source of each end
        assert (band["measured_min_pct"], band["measured_max_pct"]) == (-58.32, 56.91)
        assert (band["sources"]["lower_end"], band["sources"]["upper_end"]) == ("public", "public")
        assert band["data_base"]["public"]["cells"] == 51 and band["data"].endswith(sr.PUBLIC_PARTS_FILE)
        ic = c["independent_check"]
        assert ic["file"] == "verify/verify.json" and ic["found"] is False and ic["p2_p99_9_by_volume_MPa"] is None
        assert ic["p2_over_limit"] is None and ic["p2_over_voxel"] is None
        assert '"Met" is about the relaxed measure, not about the part.' in c["summary"] and c["summary"].endswith(sr.VERIFY_OPTION)       # F4-29
        assert ic["field"].endswith(".von_mises.p99_9_by_volume_MPa")
    assert side["achieved_safety_factor"]["against_allowable"]["p99_9"]["value"] == pytest.approx(
        40.0 / side["physical"]["p99_9_by_volume_MPa"])
    assert "allowable 40" in side["achieved_safety_factor"]["text"]
    assert down["achieved_safety_factor"]["against_allowable"] is None
    # run.json keeps the limits as the user gave them; constraints keep the relaxed numbers per row
    run = json.loads((stress_run / "run.json").read_text())
    assert run["objective"] == "mass" and [c["name"] for c in run["constraints"]] == ["stress:down", "stress:side"]
    assert run["stress"]["cases"][1]["explicit_stress_limit_MPa"] is None
    assert {c["name"] for c in metrics["constraints"]} == {"stress:down", "stress:side"}


def test_stress_limit_is_final_and_allowable_is_kept_for_the_factor(tmp_path):
    p = stress_yaml(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["load_cases"][1].pop("safety_factor")
    d["load_cases"][1].update(stress_limit=8.0)           # final limit 8 MPa, next to allowable 40 MPa (no factor)
    p.write_text(yaml.safe_dump(d))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    side = json.loads((out / "result" / "metrics.json").read_text())["stress"]["cases"]["side"]
    assert side["stress_limit_MPa"] == 8.0 and side["limit_from"] == "stress_limit"
    assert side["allowable_mpa"] == 40.0 and side["safety_factor"] is None
    assert "stress_limit 8 MPa is the final limit" in side["limit_note"] and "allowable_mpa 40 MPa" in side["limit_note"]
    assert side["achieved_safety_factor"]["against_allowable"]["peak"]["value"] == pytest.approx(
        40.0 / side["physical"]["peak_MPa"])


def test_b2_value_is_shown_beside_the_voxel_value(stress_run):
    """With verify/verify.json present, `update_metrics` shows the P2 p99.9 next to the voxel p99.9."""
    before = json.loads((stress_run / "result" / "metrics.json").read_text())
    verify = stress_run / "verify"
    verify.mkdir(exist_ok=True)
    (verify / "verify.json").write_text(json.dumps({"cases": {"down": {"von_mises": {"p99_9_by_volume_MPa": 7.5}}}}))
    section = sr.update_metrics(stress_run)
    ic = section["cases"]["down"]["independent_check"]
    assert ic["found"] and ic["p2_p99_9_by_volume_MPa"] == 7.5
    assert ic["p2_over_voxel"] == pytest.approx(7.5 / ic["voxel_p99_9_by_volume_MPa"])
    assert ic["p2_over_limit"] == pytest.approx(7.5 / section["cases"]["down"]["stress_limit_MPa"])
    assert "P2 p99.9 by volume 7.5 MPa" in section["cases"]["down"]["summary"] and "times the limit 10 MPa" in section["cases"]["down"]["summary"]
    assert section["cases"]["side"]["independent_check"]["found"] is False        # no such case in the file
    on_disk = json.loads((stress_run / "result" / "metrics.json").read_text())
    assert on_disk["stress"]["cases"]["down"]["independent_check"]["found"]
    assert {k: v for k, v in on_disk.items() if k != "stress"} == {k: v for k, v in before.items() if k != "stress"}
    (verify / "verify.json").write_text("not json")                                # a broken file: no value, no error
    assert sr.read_verify(stress_run, "down") is None
    (verify / "verify.json").unlink()
    verify.rmdir()
    sr.update_metrics(stress_run)


def test_b2_value_names_the_file_that_verify_meshed(stress_run):
    """`independent_check` has `verified_file` (`run.verified_file` of `verify.json`) and its note names it (P4b-3, fix round 1, m4)."""
    verify = stress_run / "verify"
    verify.mkdir(exist_ok=True)

    def check(run_block):
        doc = {"cases": {"down": {"von_mises": {"p99_9_by_volume_MPa": 7.5}}}}
        if run_block is not None:
            doc["run"] = run_block
        (verify / "verify.json").write_text(json.dumps(doc))
        return sr.update_metrics(stress_run)["cases"]["down"]["independent_check"]
    try:
        final = check({"verified_file": "final.stl"})
        assert final["verified_file"] == "final.stl" and "meshed final.stl" in final["note"] and "two different models" not in final["note"]
        nominal = check({"verified_file": "nominal.stl"})
        assert nominal["verified_file"] == "nominal.stl" and "meshed nominal.stl" in nominal["note"]
        part = check({"verified_file": "part_only.stl"})
        assert part["verified_file"] == "part_only.stl" and part["found"] and part["p2_p99_9_by_volume_MPa"] == 7.5
        assert "meshed part_only.stl" in part["note"] and "without the bodies of regions.not_printed" in part["note"]
        assert "voxel model is the whole design" in part["note"] and "two different models" in part["note"]
        assert "meshes final.stl" not in part["note"]
        old = check(None)                                                         # a verify.json that has no `run` block: the file is not known
        assert old["verified_file"] is None and "final.stl" not in old["note"] and old["found"]
    finally:
        (verify / "verify.json").unlink()
        verify.rmdir()
        sr.update_metrics(stress_run)
    assert sr.update_metrics(stress_run)["cases"]["down"]["independent_check"]["verified_file"] is None      # no verify.json: no file


def write_verify(folder, cases, **run):
    (folder / "verify").mkdir(exist_ok=True)
    doc = {"cases": {n: {"von_mises": {"p99_9_by_volume_MPa": v}} for n, v in cases.items()}}
    if run:
        doc["run"] = run
    (folder / "verify" / "verify.json").write_text(json.dumps(doc))


def test_a_reader_gets_the_p2_value_next_to_met_from_verify_json_and_no_file_is_written(stress_run, tmp_path):
    """F4-29: the post step writes the text before `topopt verify` ran, and nothing writes it again. A reader (`run_result`, the levers) reads verify.json again."""
    d = tmp_path / "r"
    shutil.copytree(stress_run, d)
    path = d / "result" / "metrics.json"
    text = path.read_text()
    metrics = json.loads(text)
    assert sr.refresh_from_verify(metrics, d) is False                      # no verify.json: the text of the run stays as it is
    assert metrics == json.loads(text)
    write_verify(d, {"down": 7.5}, verified_file="nominal.stl")
    assert sr.refresh_from_verify(metrics, d) is True
    down, side = metrics["stress"]["cases"]["down"], metrics["stress"]["cases"]["side"]
    assert down["independent_check"]["found"] and down["independent_check"]["p2_p99_9_by_volume_MPa"] == 7.5
    assert down["independent_check"]["p2_over_limit"] == pytest.approx(0.75) and down["independent_check"]["verified_file"] == "nominal.stl"
    assert "P2 p99.9 by volume 7.5 MPa, 0.75 times the limit 10 MPa (tets of nominal.stl" in down["summary"] and "has no P2 value" not in down["summary"]
    assert side["independent_check"]["found"] is False and side["summary"] == json.loads(text)["stress"]["cases"]["side"]["summary"]    # no value for this case: as it was
    assert path.read_text() == text                                          # nothing was written
    assert sr.refresh_from_verify(metrics, d) is False                      # the second read changes nothing
    (d / "verify" / "verify.json").write_text("not json")                    # a broken file: no value, the option again, no crash
    assert sr.refresh_from_verify(metrics, d) is True and metrics == json.loads(text)
    assert sr.refresh_from_verify({"verdict": {}}, d) is False and sr.refresh_from_verify(None, d) is False and sr.refresh_from_verify({"stress": {"error": "x"}}, d) is False


def test_run_result_and_the_levers_show_the_p2_value_next_to_met(stress_run, tmp_path):
    """F4-29 on a real run folder: the MCP result and the levers block read verify/verify.json at the call. The row is a mass run with a stress limit."""
    from topopt import levers_run
    from topopt_mcp import server
    d = tmp_path / "r"
    shutil.copytree(stress_run, d)
    cases = json.loads((d / "result" / "metrics.json").read_text())["stress"]["cases"]
    assert sr.VERIFY_OPTION in server.run_result(str(d), preview=False)["metrics"]["stress"]["cases"]["down"]["summary"]
    write_verify(d, {"down": 7.5})
    res = server.run_result(str(d), preview=False)
    down = res["metrics"]["stress"]["cases"]["down"]
    assert down["independent_check"]["found"] and "P2 p99.9 by volume 7.5 MPa" in down["summary"]
    assert json.loads((d / "result" / "metrics.json").read_text())["stress"]["cases"]["down"]["independent_check"]["found"] is False     # the file of the run is as it was
    # a levers entry: the relaxed row of `down` is flagged here only when the numbers say so; force it on a copy of the metrics to see the P2 value in the hit
    def flag(m):
        m["stress"]["cases"]["down"]["relaxed"].update(met=False, peak_above_limit=True)
    data = json.loads((d / "result" / "metrics.json").read_text())
    flag(data)
    (d / "result" / "metrics.json").write_text(json.dumps(data))
    out = levers_run.run_levers(d)
    for fact in ("metrics.stress.cases.<case>.relaxed.met", "metrics.stress.cases.<case>.relaxed.peak_above_limit"):
        e = next(e for e in out["entries"] if e["fact"] == fact)
        w = e["hits"][0]["with"]
        assert w["independent_check.found"] is True and w["independent_check.p2_p99_9_by_volume_MPa"] == 7.5
        assert w["independent_check.p2_over_limit"] == pytest.approx(7.5 / cases["down"]["stress_limit_MPa"])
        assert "found false means verify has not run" in e["flag"]
    (d / "verify" / "verify.json").unlink()
    w = next(e for e in levers_run.run_levers(d)["entries"] if e["fact"] == "metrics.stress.cases.<case>.relaxed.met")["hits"][0]["with"]
    assert w["independent_check.found"] is False and w["independent_check.p2_p99_9_by_volume_MPa"] is None and w["independent_check.p2_over_limit"] is None


def test_section_never_raises(tmp_path):
    out = sr.stress_section(tmp_path)                    # no run.json at all
    assert "error" in out and "the stress report was not made" in out["note"]


def test_run_without_a_stress_row_has_no_section(tmp_path):
    out = tmp_path / "r"
    assert main(["run", str(make_case(tmp_path, h=2.0, iterations=3)), "--out", str(out), "--quiet"]) == 0
    assert "stress" not in json.loads((out / "result" / "metrics.json").read_text())
    assert "error" in sr.stress_section(out)             # asked by hand: a fact, not a crash


# --- the inputs of the run, and the solve that the post-process shares (REVIEW m7, n6) ----------


def test_update_metrics_says_when_the_forces_or_the_material_changed(tmp_path):
    """The rebuilt problem must have the forces and the material of the run, else no physical numbers (m7)."""
    p = stress_yaml(tmp_path, iterations=6)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    original = p.read_text()
    again = sr.update_metrics(out)                         # nothing changed: the check passes, physical numbers
    assert again["input_check"]["changed"] is False and again["cases"]["down"]["physical"]["voxels"] > 0
    assert again["input_check"]["not_compared"] == []      # this run.json holds E, nu and density
    d = yaml.safe_load(original)
    d["load_cases"][0]["forces"][0]["force"] = [0, 0, -40.0]          # the force of case `down` is doubled
    p.write_text(yaml.safe_dump(d))
    sec = sr.update_metrics(out)
    down, side = sec["cases"]["down"], sec["cases"]["side"]
    assert down["physical"] is None and down["achieved_safety_factor"] is None and down["verdict"] is None
    assert "no physical numbers for this case" in down["input_changed"]["fact"]
    assert "forces:" in down["input_changed"]["differences"][0] and "-40.0" in down["input_changed"]["differences"][0]
    assert down["relaxed"]["label"] == "relaxed, not physical"           # the relaxed numbers belong to the run
    assert side["physical"]["voxels"] > 0 and "input_changed" not in side    # the other case is unchanged
    assert sec["input_check"]["changed"] is True
    d = yaml.safe_load(original)
    d["material"] = {"E": 4000.0}
    p.write_text(yaml.safe_dump(d))
    sec = sr.update_metrics(out)
    for c in sec["cases"].values():
        assert c["physical"] is None and any(t.startswith("material.E: run.json has 3500") for t in
                                             c["input_changed"]["differences"])
    p.write_text(original)
    assert sr.update_metrics(out)["cases"]["down"]["physical"]["voxels"] > 0


def test_a_run_json_without_the_material_compares_the_density_and_says_what_it_did_not_compare(tmp_path):
    p = stress_yaml(tmp_path, iterations=4)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run = json.loads((out / "run.json").read_text())
    run["stress"].pop("material")                          # a run.json made before this field existed
    (out / "run.json").write_text(json.dumps(run))
    sec = sr.update_metrics(out)
    assert sec["input_check"]["not_compared"] == ["material.E and material.nu: this run.json does not hold them"]
    assert sec["cases"]["down"]["physical"]["voxels"] > 0


def test_the_cli_run_solves_the_final_stress_once(tmp_path, monkeypatch):
    """The stress section and the refine page use the same solve of the last design (n6)."""
    import topopt.refine.stress as rs
    calls = []
    original = rs.compute_stress

    def counted(problem, field, *a, **k):
        calls.append(sorted(set(np.unique(field).tolist())))
        return original(problem, field, *a, **k)

    monkeypatch.setattr(rs, "compute_stress", counted)
    out = tmp_path / "r"
    assert main(["run", str(stress_yaml(tmp_path, iterations=4)), "--out", str(out), "--quiet"]) == 0
    # the grey design is solved once (the refine page uses the same solve); the thresholded design (R1-3) is the second call
    assert len(calls) == 2 and len(calls[0]) > 2 and set(calls[1]) <= {0, 255}
    assert (out / "result" / "stress.npz").is_file() and (out / "result" / "refine.html").is_file()
    assert "refine failed" not in (out / "run.log").read_text()
    calls.clear()                                          # a run without a stress row: refine solves it once
    out2 = tmp_path / "r2"
    assert main(["run", str(make_case(tmp_path / "plain" if (tmp_path / "plain").mkdir() is None else tmp_path,
                                      h=2.0, iterations=3)), "--out", str(out2), "--quiet"]) == 0
    assert len(calls) == 1


def test_the_accuracy_data_marks_the_rows_where_p99_9_is_the_peak():
    doc = json.loads(DATA.read_text())
    for r in doc["beam"]["sizes"]:
        n = r["whole_design"]["voxels"]
        assert r["whole_design"]["p99_9_equals_peak"] == (n < 5000)
        w = r["whole_design"]
        assert (abs(w["p99_9_MPa"] - w["peak_MPa"]) <= 1e-6 * w["peak_MPa"]) == w["p99_9_equals_peak"]
    for r in doc["hole"]["sizes"]:
        assert r["p99_9_equals_peak"] == (r["design_voxels"] < 5000)
    for c in doc["cross_check"]["cases"].values():
        assert c["p99_9_equals_peak"] is False and c["design_voxels"] >= 5000
    base = doc["verdict_rule"]["p99_9"]["data_base"]
    assert (base["bench"]["shapes"], base["bench"]["real_parts"]) == (3, 1)
    for part in (base, base["bench"]):
        assert part["rows_where_p99_9_equals_the_peak"] + part["rows_where_p99_9_is_a_true_percentile"] == part["rows"]
    pub = base["public"]       # F1-8c: the public cells count too (31 of 51 have p99.9 equal to the peak)
    assert pub["cells_where_p99_9_equals_the_peak"] + pub["cells_where_p99_9_is_a_true_percentile"] == pub["cells"] == 51
    assert base["rows"] == base["bench"]["rows"] + pub["cells"] and pub["cells_where_p99_9_equals_the_peak"] == 31
    assert base["rows_where_p99_9_equals_the_peak"] == base["bench"]["rows_where_p99_9_equals_the_peak"] + 31


def test_the_hook_comparison_is_split_into_effects():
    """REVIEW M3: grey voxels, load patch and load model are separate rows of the data file."""
    split = json.loads(DATA.read_text())["cross_check"]["split"]
    pull = split["cases"]["pull_off"]
    assert pull["error_pct"]["grey_vs_p2"] == pytest.approx(-53.3, abs=0.1)              # as in the accuracy study
    assert pull["error_pct"]["thresholded_vs_p2"] == pytest.approx(-54.5, abs=0.1)       # grey voxels out
    assert pull["load_patch_left_out"]["error_pct"] == pytest.approx(-51.6, abs=0.2)     # load patch out
    assert pull["load_model_like_for_like"]["error_pct"] == pytest.approx(-40.9, abs=0.2)  # nodal spread like the voxel model
    assert pull["load_model_like_for_like"]["stiffness_ratio_voxel_over_p2_traction"] == pytest.approx(1.715, abs=0.01)
    assert pull["p2_element_max_proxy"]["share_of_top_0_1_percent_volume_near_regions"] > 0.7
    assert "stored_values" in pull and "earlier P2 run" in pull["stored_values"]["source"]


# --- the peak scope and the load path (F1 follow-up) ---------------------------------


def dead_region_problem(tmp_path):
    """The cantilever of `make_case` (1 mm voxels) plus a load region 'dead' that no load case uses."""
    from test_optimize import box
    p = make_case(tmp_path, h=1.0)
    box(tmp_path / "dead.stl", (8, 3, 6), (10, 5, 9))
    d = yaml.safe_load(p.read_text())
    d["regions"]["load"]["dead"] = "dead.stl"
    p.write_text(yaml.safe_dump(d))
    return build_problem(load_config(p))


def fake_stress(problem, vm_by_label):
    """A `compute_stress` result for all active voxels with rho >= 0.5. `vm_by_label` is {label: value}."""
    from topopt.optimize import reduced_system
    from topopt.voxelize import DESIGN
    _edof, active, _free = reduced_system(problem)
    label = problem.labels.reshape(-1)[active]
    ijk = np.stack(np.unravel_index(active, problem.shape), axis=1)
    vm = np.array([vm_by_label[int(k)] for k in label], float)
    return {"centers": np.asarray(problem.origin) + problem.h * (ijk + 0.5), "rho": np.ones(len(active)),
            "vm_cases": np.stack([vm, 2 * vm]), "vm_max": 2 * vm, "case_names": ["down", "side"],
            "compliance_cases": np.ones(2), "design": label == DESIGN}, label


def test_peak_scope_leaves_out_only_fixed_voxels_and_the_voxels_of_a_used_load_region(tmp_path):
    """A kept-solid body with no clamp and no load (a load region that no case uses) has real stress: it counts."""
    from topopt.voxelize import DESIGN, FIXED, LOAD
    problem = dead_region_problem(tmp_path)
    stress, label = fake_stress(problem, {DESIGN: 1.0, FIXED: 50.0, LOAD: 30.0})
    rid = problem.load_region_ids.reshape(-1)[np.ravel_multi_index(
        np.floor((stress["centers"] - problem.origin) / problem.h).astype(int).T, problem.shape)]
    names = [lr.name for lr in problem.config.regions.load]
    tip, dead = rid == names.index("tip"), rid == names.index("dead")
    assert tip.any() and dead.any() and (label[tip] == LOAD).all() and (label[dead] == LOAD).all()
    stress["vm_cases"][0][dead] = 7.0                                 # the stress of the dead region, case `down`
    stress["vm_cases"][0][tip] = 40.0                                 # a loaded voxel: left out
    scope = sr.peak_scope(stress, problem)
    assert not scope[label == FIXED].any() and not scope[tip].any()   # fixed and used load voxels: left out
    assert scope[label == DESIGN].all() and scope[dead].all()         # design and the unused load region: counted
    phys = {p["name"]: p for p in sr.physical_statistics(stress, {"down": 5.0, "side": 5.0}, 1.0, scope)}
    down = phys["down"]
    assert down["peak_MPa"] == 7.0 and down["peak_incl_forced_solid_MPa"] == 50.0
    assert down["solid_voxels_without_clamp_or_load"] == int(dead.sum()) and down["voxels"] == int(scope.sum())
    assert down["label"] == sr.PHYSICAL_LABEL + f" and {int(dead.sum())} solid voxels that carry no clamp and no load"
    assert down["voxels_above_limit"] >= int(dead.sum())              # the 7 MPa voxels are above the limit 5
    # the default scope (no `scope`) is the design voxels: the old numbers and the old label
    old = {p["name"]: p for p in sr.physical_statistics(stress, {"down": 5.0, "side": 5.0}, 1.0)}["down"]
    assert old["peak_MPa"] == 1.0 and "solid_voxels_without_clamp_or_load" not in old and old["label"] == sr.PHYSICAL_LABEL
    # with no dead region the scope is the design set
    (tmp_path / "plain").mkdir()
    problem2 = build_problem(load_config(make_case(tmp_path / "plain", h=1.0)))
    stress2, label2 = fake_stress(problem2, {DESIGN: 1.0, FIXED: 50.0, LOAD: 30.0})
    assert (sr.peak_scope(stress2, problem2) == (label2 == DESIGN)).all()


def box_stl(path, lo, hi):
    from test_optimize import box
    box(path, lo, hi)
    return path


def make_snapshot(problem, design_value):
    return np.where(problem.labels == 1, design_value, 0).astype(np.uint8)


def test_load_path_facts_for_the_thresholded_design(tmp_path):
    """The design voxels at rho >= 0.5 join the load region to the plate (or do not)."""
    problem = dead_region_problem(tmp_path)
    solid = sr.load_path_facts(problem, make_snapshot(problem, 255))
    tip = solid["regions"]["tip"]
    assert tip["thresholded_design"]["connected_to_fixed"] and tip["used_by"] == ["down", "side"]
    assert solid["regions"]["dead"]["used_by"] == [] and solid["final_stl"] is None
    assert "6-connected" in solid["rule"]
    void = sr.load_path_facts(problem, make_snapshot(problem, 127))        # rho 0.498: below the threshold 128
    assert not void["regions"]["tip"]["thresholded_design"]["connected_to_fixed"]
    assert void["thresholded_design_bodies"] > 1
    # the thresholded design has one voxel gap (a cut across the arm): not joined
    snap = make_snapshot(problem, 255).reshape(problem.shape)
    snap[14, :, :] = 0                                                    # x 12 to 13 mm: away from the dead region (x 8 to 10)
    cut = sr.load_path_facts(problem, snap)
    assert not cut["regions"]["tip"]["thresholded_design"]["connected_to_fixed"]
    snap = make_snapshot(problem, 255).reshape(problem.shape)
    snap[10:12, :, :] = 0                                                 # a cut across the dead region (x 8 to 10 mm)
    bridged = sr.load_path_facts(problem, snap)                           # its kept-solid voxels are material: they join the sides
    assert bridged["regions"]["tip"]["thresholded_design"]["connected_to_fixed"] is True


def test_load_path_facts_for_final_stl(tmp_path):
    problem = dead_region_problem(tmp_path)
    snap = make_snapshot(problem, 255)
    one = box_stl(tmp_path / "one.stl", (-2, 0, 0), (20, 10, 10))          # plate and tip in one body
    r = sr.load_path_facts(problem, snap, one)["regions"]["tip"]["final_stl"]
    assert r["connected_to_fixed"] and r["touches_a_body"] and r["bodies_with_the_region"] == 1
    plate = box_stl(tmp_path / "plate.stl", (-2, 0, 0), (4, 10, 10))       # no material at the tip
    facts = sr.load_path_facts(problem, snap, plate)
    r = facts["regions"]["tip"]["final_stl"]
    assert not r["connected_to_fixed"] and not r["touches_a_body"] and facts["final_stl"]["bodies"] == 1
    import trimesh
    two = trimesh.util.concatenate([trimesh.load(plate), trimesh.load(box_stl(tmp_path / "tipbox.stl", (14, 0, 0), (20, 10, 10)))])
    two.export(str(tmp_path / "two.stl"))                                  # plate and a floating tip
    facts = sr.load_path_facts(problem, snap, tmp_path / "two.stl")
    r = facts["regions"]["tip"]["final_stl"]
    assert r["touches_a_body"] and not r["connected_to_fixed"] and facts["final_stl"]["bodies"] == 2
    assert facts["final_stl"]["bodies_with_a_fixed_region"] == 1
    bad = tmp_path / "bad.stl"
    bad.write_text("not an stl")
    assert "error" in sr.load_path_facts(problem, snap, bad)["final_stl"]    # an unreadable mesh never stops the report
    assert sr.load_path_facts(problem, snap, tmp_path / "absent.stl")["final_stl"] is None


def test_holds_a_point_stops_at_the_first_chunk_with_a_point_inside(monkeypatch):
    """`_holds_a_point` gives `contains(pts).any()` but tests the points in chunks and stops at the first hit."""
    from topopt import post

    class Stub:
        def __init__(self):
            self.calls = []

        def contains(self, pts):
            self.calls.append(len(pts))
            return pts[:, 0] < 0

    monkeypatch.setattr(post, "CONTAINS_CHUNK", 3)
    pts = np.ones((10, 3))
    stub = Stub()
    assert sr._holds_a_point(stub, pts) is False and stub.calls == [3, 3, 3, 1]      # no point inside: every chunk
    pts[5, 0] = -1.0
    stub = Stub()
    assert sr._holds_a_point(stub, pts) is True and stub.calls == [3, 3]            # point 5 is in the second chunk: stop
    stub = Stub()
    assert sr._holds_a_point(stub, np.zeros((0, 3))) is False and stub.calls == []


def test_final_stl_facts_do_not_depend_on_the_chunk(tmp_path, monkeypatch):
    """Plate and a floating tip (2 bodies): the same facts for chunks of 1, 5 and 300 points, and the one-call value."""
    import trimesh
    from topopt import post
    from topopt.springs import support_mask

    problem = dead_region_problem(tmp_path)
    snap = make_snapshot(problem, 255)
    plate = box_stl(tmp_path / "plate.stl", (-2, 0, 0), (4, 10, 10))
    tipbox = box_stl(tmp_path / "tipbox.stl", (14, 0, 0), (20, 10, 10))
    trimesh.util.concatenate([trimesh.load(plate), trimesh.load(tipbox)]).export(str(tmp_path / "two.stl"))
    bodies = trimesh.load(str(tmp_path / "two.stl"), process=True, force="mesh").split(only_watertight=False)
    sets = {"fixed": sr._region_points(problem, support_mask(problem)),
            "tip": sr._region_points(problem, problem.load_region_ids == 0)}
    assert len(bodies) == 2 and all(len(p) > 5 for p in sets.values())          # more than one chunk at 5
    facts = {}
    for chunk in (1, 5, 300):
        monkeypatch.setattr(post, "CONTAINS_CHUNK", chunk)
        facts[chunk] = sr.load_path_facts(problem, snap, tmp_path / "two.stl")
        for name, pts in sets.items():
            for b in bodies:
                assert sr._holds_a_point(b, pts) == bool(b.contains(pts).any()), (chunk, name)
    assert facts[1] == facts[5] == facts[300]
    assert facts[1]["final_stl"] == {"bodies": 2, "bodies_with_a_fixed_region": 1}
    tip = facts[1]["regions"]["tip"]["final_stl"]
    assert tip == {"connected_to_fixed": False, "touches_a_body": True, "bodies_with_the_region": 1}


def test_the_summary_says_when_the_relaxed_result_meets_the_rows_and_the_part_has_no_load_path(tmp_path):
    """The A2 example at 2.4 mm: both rows met, and no material between the tip and the plate."""
    problem = dead_region_problem(tmp_path)
    facts = sr.load_path_facts(problem, make_snapshot(problem, 0))
    lp = sr.case_load_path(facts, "down")
    assert lp["all_connected"] is False and lp["not_connected"] == ["tip"] and lp["regions"]["tip"]["final_stl"] is None
    assert "dead" not in lp["regions"]                                     # the region that the case does not use
    entry = fake_case(True, 6.19, 8.01)
    entry["load_path"] = lp
    t = sr.case_summary("down", entry, facts)
    assert t.startswith("the optimizer row is met for the relaxed stress (relaxed peak 4.57 MPa, limit 5 MPa, g +0.0005).")
    assert ("But the thresholded part has no material between load region 'tip' and the fixed regions; "
            "this is not a part that carries the load. options:") in t
    assert "settings.filter_radius" in t and "compliance" in t
    assert "Voxel estimate of the part stress, peak: 6.19 to 8.01 MPa" in t              # the old text stays
    entry["relaxed"] = {"met": False, "peak_MPa": 6.0, "row_g": 0.3}
    t = sr.case_summary("down", entry, facts)
    assert "The thresholded part has no material between load region 'tip'" in t and "But the thresholded" not in t
    ok = sr.load_path_facts(problem, make_snapshot(problem, 255))
    entry["load_path"] = sr.case_load_path(ok, "down")
    entry["relaxed"] = {"met": True, "peak_MPa": 4.57, "row_g": 0.0005}
    assert "carries the load" not in sr.case_summary("down", entry, ok)
    assert sr.case_summary("down", entry) == sr.case_summary("down", {**entry, "load_path": None}, ok)   # no facts: no text
    # the mesh case: the thresholded design joins the region, the smoothed mesh does not
    plate = box_stl(tmp_path / "plate.stl", (-2, 0, 0), (4, 10, 10))
    facts = sr.load_path_facts(problem, make_snapshot(problem, 255), plate)
    entry["load_path"] = sr.case_load_path(facts, "down")
    t = sr.case_summary("down", entry, facts)
    assert "But final.stl has no material at load region 'tip' (the thresholded design joins them, the smoothed mesh does not)" in t


def test_the_section_reports_the_load_path_of_a_real_run(tmp_path):
    """A run whose last snapshot has no design material: the rows are the optimizer's, the part has no load path."""
    p = stress_yaml(tmp_path, iterations=6)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    sec = sr.stress_section(out)
    lp = sec["load_paths"]
    assert lp["regions"]["tip"]["thresholded_design"]["connected_to_fixed"] is True           # rho 0.6 at the start
    assert lp["final_stl"]["bodies"] >= 1 and lp["regions"]["tip"]["final_stl"]["connected_to_fixed"] is True
    for c in sec["cases"].values():
        assert c["load_path"]["all_connected"] is True and "carries the load" not in c["summary"]
    from topopt.post import _last_snapshot
    snap_path = _last_snapshot(out)
    labels = np.fromfile(out / "labels.bin", np.uint8)
    snap = np.fromfile(snap_path, np.uint8)
    snap[labels == 1] = 0                                                  # all design voxels void
    snap_path.write_bytes(snap.tobytes())
    sec = sr.update_metrics(out)
    assert sec["load_paths"]["regions"]["tip"]["thresholded_design"]["connected_to_fixed"] is False
    for name, c in sec["cases"].items():
        assert c["load_path"]["not_connected"] == ["tip"]
        assert ("has no material between load region 'tip' and the fixed regions; "
                "this is not a part that carries the load") in c["summary"]
    on_disk = json.loads((out / "result" / "metrics.json").read_text())["stress"]
    assert on_disk["load_paths"]["regions"]["tip"]["thresholded_design"]["connected_to_fixed"] is False
    assert set(on_disk) >= {"load_paths", "cases", "stage_ends"} and "load_path" not in on_disk       # no new top-level key


COVERAGE_BUCKLING_FAILED = {"ran": ["stress"], "failed": [{"name": "buckling", "reason": "RuntimeError: phase 33 error -4"}], "not_run": []}


def test_independent_check_carries_the_coverage_of_a_partial_verify_result(tmp_path):
    """TC-2: the P2 value comes from a verify run whose other sub-checks may have failed. The keys are there only when verify.json has them."""
    v = tmp_path / "verify"
    v.mkdir()
    doc = {"cases": {"down": {"von_mises": {"p99_9_by_volume_MPa": 7.5}}}}
    (v / "verify.json").write_text(json.dumps(doc))
    ic = sr.independent_check(tmp_path, "down", 10.0, 5.0)
    assert ic["found"] and "verify_partial" not in ic and "verify_coverage" not in ic          # an older verify.json: the dict is as it was
    (v / "verify.json").write_text(json.dumps({**doc, "partial": True, "coverage": COVERAGE_BUCKLING_FAILED}))
    ic = sr.independent_check(tmp_path, "down", 10.0, 5.0)
    assert ic["verify_partial"] is True and ic["verify_coverage"] == COVERAGE_BUCKLING_FAILED and ic["p2_p99_9_by_volume_MPa"] == 7.5
    clean = {"ran": ["stress"], "failed": [], "not_run": []}
    (v / "verify.json").write_text(json.dumps({**doc, "partial": False, "coverage": clean}))
    ic = sr.independent_check(tmp_path, "down", 10.0, 5.0)
    assert ic["verify_partial"] is False and ic["verify_coverage"] == clean
    assert "verify_partial" not in sr.independent_check(tmp_path, "side", 10.0, 5.0)           # no P2 value for this case: nothing to qualify
    (v / "verify.json").write_text(json.dumps({**doc, "partial": "yes", "coverage": clean}))
    assert sr.read_verify_coverage(tmp_path) is None                                            # a key of the wrong type is no record


def test_the_summary_says_when_the_verify_result_is_partial():
    e = fake_case(True, 4.0, 4.5, above=0)
    e["independent_check"] = {**p2_check(7.5), "verify_partial": True, "verify_coverage": COVERAGE_BUCKLING_FAILED}
    t = sr.case_summary("pull_off", e)
    assert "That verify result is partial: 1 of 2 checks failed (buckling: RuntimeError: phase 33 error -4); ran: stress." in t
    assert t.index("P2 p99.9 by volume 7.5 MPa") < t.index("That verify result is partial")      # after the number it qualifies
    e["independent_check"]["verify_partial"] = False
    assert "That verify result" not in sr.case_summary("pull_off", e)
    e["independent_check"] = p2_check(7.5)                                                       # an older verify.json: the text as it was
    assert "That verify result" not in sr.case_summary("pull_off", e)
