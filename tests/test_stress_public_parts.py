"""The measurement on 5 public parts (F1-8, Refs #129): `bench/stress_public_parts.json` and the texts that quote it.

The JSON must load with no run folder, and its summary must equal a recomputation from its cells. F1-8c: the band of the voxel
stress is the union of the bench rows and these cells (p99.9: 59 and 57; peak: 88 and 0), derived from the data files, and every
text that gives a band or a range names the source of each end. The texts carry numbers that equal the data (no hand-typed copy
that can drift).
"""

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import spearmanr

from topopt import stress_report as sr

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
DATA = BENCH / "stress_public_parts.json"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pp():
    return load_script("stress_public_parts")


@pytest.fixture(scope="module")
def doc():
    return json.loads(DATA.read_text())


# --- the data file ---------------------------------------------------------------------


def test_the_file_loads_with_no_run_folder_and_has_the_header(pp, doc):
    assert pp.load() == doc and pp.DATA == DATA
    assert doc["format"] == 1 and doc["main"] == "74a1e46" and doc["date"] == "2026-10-25"
    assert "Linux x86_64" in doc["host"] and "not Apple silicon" in doc["host"]
    assert doc["run_folders"].startswith("not in the repository")
    assert doc["issue"].endswith("#129 (F1-8, step 1)") and doc["script"] == "bench/stress_public_parts.py"
    assert set(doc["definitions"]["run_types"]) == {"R", "CS", "MS", "MF"}
    cells = doc["cells"]
    assert len(cells) == doc["summary"]["n_cells"] == 51
    keys = {"part", "case", "run_type", "variant", "runs", "voxel_mm", "voxel_p99_9_MPa", "p2_p99_9_MPa", "p2_over_voxel",
            "voxel_error_pct", "voxels_in_scope", "p99_9_equals_peak", "member_p10_voxels", "thinnest_member_voxels",
            "near_share", "p2_peak_over_voxel_peak", "grayness_last_pass"}
    assert all(set(c) == keys for c in cells)
    assert len({(c["part"], c["case"], c["run_type"], c["variant"]) for c in cells}) == 51      # one row for each cell
    assert set(doc["parts"]) - {"note"} == {c["part"] for c in cells} == set(doc["summary"]["parts"])


def test_each_cell_is_consistent(doc):
    for c in doc["cells"]:
        assert c["voxel_error_pct"] == pytest.approx(100.0 * (1.0 / c["p2_over_voxel"] - 1.0), abs=0.01), c
        if len(c["runs"]) == 1:         # the cell of a repeat holds the mean of two ratios, not the ratio of two means
            assert c["p2_over_voxel"] == pytest.approx(c["p2_p99_9_MPa"] / c["voxel_p99_9_MPa"], rel=1e-3), c
            assert c["voxel_error_pct"] == pytest.approx(100.0 * (c["voxel_p99_9_MPa"] / c["p2_p99_9_MPa"] - 1.0), abs=0.5), c
        assert c["p99_9_equals_peak"] == (c["voxels_in_scope"] < 5000), c
        assert c["p2_peak_over_voxel_peak"] > 1.0 and 0.0 <= c["near_share"] <= 1.0


def test_the_summary_equals_a_recomputation_from_the_cells(pp, doc):
    """The summary counts against the BENCH band alone: the band of the report is the union and contains every cell."""
    bench = sr.ACCURACY["p99_9"]["sources"]["bench"]
    again = pp.summarize(doc["cells"], bench["under_pct"], bench["over_pct"], sr.MIN_MEMBER_VOXELS_EST)
    assert again == doc["summary"]
    s = doc["summary"]
    assert (s["bench_band"]["under_pct"], s["bench_band"]["over_pct"]) == (bench["under_pct"], bench["over_pct"]) == (54, 13)
    assert s["member_gate"]["min_p10_voxels"] == sr.MIN_MEMBER_VOXELS_EST == 3.5
    file_bench = json.loads((BENCH / "stress_accuracy.json").read_text())["verdict_rule"]["p99_9"]["sources"]["bench"]
    assert file_bench == bench and pp.bench_band() == bench
    # the union is another band: no cell is outside it
    acc = sr.ACCURACY["p99_9"]
    assert (acc["under_pct"], acc["over_pct"]) == (59, 57)
    assert all(-acc["under_pct"] <= c["voxel_error_pct"] <= acc["over_pct"] for c in doc["cells"])


def test_resummarize_writes_the_committed_summary_again_with_no_run_folder(pp, tmp_path, monkeypatch):
    out = tmp_path / "stress_public_parts.json"
    out.write_text(DATA.read_text())
    monkeypatch.setattr(sys, "argv", ["stress_public_parts.py", "--resummarize", "--out", str(out)])
    pp.main()
    assert out.read_text() == DATA.read_text()
    broken = json.loads(out.read_text())
    broken["summary"]["bench_band"]["under_pct"] = 59          # a summary that counts against the union is not the committed one
    out.write_text(pp.dumps(broken))
    pp.main()
    assert out.read_text() == DATA.read_text()


def test_the_headline_numbers_by_an_independent_count(doc):
    cells = doc["cells"]
    ratio = [c["p2_over_voxel"] for c in cells]
    err = [100.0 * (1.0 / r - 1.0) for r in ratio]
    s = doc["summary"]
    assert (s["p2_over_voxel"]["min"], s["p2_over_voxel"]["max"]) == (round(min(ratio), 3), round(max(ratio), 3))
    assert s["p2_over_voxel"]["max_over_min"] == round(max(ratio) / min(ratio), 3)
    assert s["p2_over_voxel"]["median"] == pytest.approx(statistics.median(ratio), abs=1e-3)
    q = statistics.quantiles(ratio, n=4, method="inclusive")
    assert (s["p2_over_voxel"]["q25"], s["p2_over_voxel"]["q75"]) == (pytest.approx(q[0], abs=1e-3), pytest.approx(q[2], abs=1e-3))
    assert (s["voxel_error_pct"]["min"], s["voxel_error_pct"]["max"]) == (round(min(err), 1), round(max(err), 1))
    assert s["cells_voxel_above_p2"] == sum(r < 1 for r in ratio) and s["cells_voxel_below_p2"] == sum(r > 1 for r in ratio)
    assert s["cells_voxel_above_p2"] + s["cells_voxel_below_p2"] == 51
    mass = sum(1 for c in cells if c["run_type"] in ("MS", "MF") and c["p2_over_voxel"] < 1)
    assert s["cells_voxel_above_p2_in_mass_objective_runs"] == mass
    out = [e for e in err if e < -54 or e > 13]
    assert s["cells_outside_bench_band"] == {"total": len(out), "voxel_below_the_band": sum(e < -54 for e in err),
                                             "voxel_above_the_band": sum(e > 13 for e in err)}
    gate = [(c, e) for c, e in zip(cells, err) if c["member_p10_voxels"] >= 3.5]
    g = s["member_gate"]
    assert g["cells_meeting"] == len(gate) and g["cells_above_bench_band"] == sum(e > 13 for _, e in gate)
    assert g["cells_below_bench_band"] == sum(e < -54 for _, e in gate)
    assert g["p2_over_voxel"] == {"min": round(min(c["p2_over_voxel"] for c, _ in gate), 3),
                                  "max": round(max(c["p2_over_voxel"] for c, _ in gate), 3)}
    low, high = min(zip(err, cells), key=lambda t: t[0])[1], max(zip(err, cells), key=lambda t: t[0])[1]
    assert (s["voxel_error_pct"]["min_cell"], s["voxel_error_pct"]["max_cell"]) == (
        "hook, pull_off, run type MS, voxel 2.4 mm", "phone_stand, poke, run type MS, voxel 2.45 mm")
    assert (low["part"], low["case"], low["run_type"], low["voxel_mm"]) == ("hook", "pull_off", "MS", 2.4)
    assert (high["part"], high["case"], high["run_type"], high["voxel_mm"]) == ("phone_stand", "poke", "MS", 2.45)
    assert s["voxel_peak"]["cells_below_p2_peak"] == sum(c["p2_peak_over_voxel_peak"] > 1 for c in cells)
    assert s["cells_p99_9_equals_peak"] == sum(c["voxels_in_scope"] < 5000 for c in cells)


def test_the_four_registered_tests_follow_from_the_cells(doc):
    t, cells = doc["tests"], doc["cells"]
    ratio = [c["p2_over_voxel"] for c in cells]
    assert (t["H1"]["result"], t["H2"]["result"], t["H3"]["result"], t["H4"]["result"]) == (False, False, False, True)
    assert t["H1"]["max_over_min"] == doc["summary"]["p2_over_voxel"]["max_over_min"] and max(ratio) / min(ratio) > 1.5
    assert (t["H2"]["median"], t["H2"]["q25"], t["H2"]["q75"]) == tuple(doc["summary"]["p2_over_voxel"][k] for k in ("median", "q25", "q75"))
    assert t["H2"]["median"] == pytest.approx(statistics.median(ratio), abs=1e-3) and statistics.median(ratio) < 1.5
    rho = spearmanr([c["member_p10_voxels"] for c in cells], ratio)[0]
    assert t["H3"]["spearman_rho_p10"] == pytest.approx(rho, abs=2e-2) and rho > -0.6
    top3 = sorted(cells, key=lambda c: -c["p2_over_voxel"])[:3]
    assert [c["part"] for c in top3] == ["hook"] * 3 and {c["case"] for c in top3} == {"pull_off"}
    assert sorted(t["H4"]["top3_near_share"]) == pytest.approx(sorted(c["near_share"] for c in top3), abs=2e-3)
    assert all(c["near_share"] > 0.5 for c in top3)


def test_the_builder_specs_are_next_to_the_data():
    d = BENCH / "stress_public_parts_specs"
    assert sorted(p.name for p in d.iterdir()) == ["phone_stand_MS_spec.yaml", "phone_stand_R.yaml",
                                                   "shelf_bracket_MS_spec.yaml", "shelf_bracket_R.yaml"]
    for part in ("shelf_bracket", "phone_stand"):
        plain, limited = (yaml.safe_load((d / f).read_text()) for f in (f"{part}_R.yaml", f"{part}_MS_spec.yaml"))
        assert plain["name"] == limited["name"] == part and plain["regions"] == limited["regions"]
        assert all("stress_limit" not in c for c in plain["load_cases"])
        assert all(c["stress_limit"] > 0 for c in limited["load_cases"])
        assert limited["settings"]["objective"] == "mass" and "objective" not in plain["settings"]


# --- the texts ---------------------------------------------------------------------------


MEMBER = {"met": True, "p10_voxels": 4.0}
BIG = {"voxels": 20000, "peak_MPa": 3.0, "p99_9_by_volume_MPa": 2.0, "p99_9_equals_peak": False}
CASE = {"stress_limit_MPa": 10.0}

# The source of each end, as the cells say it (the tests below check these strings against the data, not against the code).
P999_LOWER = "lower end -59 % (measured -58.32 %) from the 5 public parts: hook, pull_off, run type MS, voxel 2.4 mm"
P999_UPPER = "upper end +57 % (measured +56.91 %) from the 5 public parts: phone_stand, poke, run type MS, voxel 2.45 mm"
PEAK_LOWER = ("lower end -88 % (measured -87.14 %) from the 5 public parts: ladle, full_scoop, run type MF, voxel 3.25 mm, "
              "variant S2")
PEAK_UPPER = "upper end 0 % (measured -4.82 %) from the bench: beam, 16 voxels through the thickness"
PEAK_MEASURED = ("lowest -87.14 % from the 5 public parts: ladle, full_scoop, run type MF, voxel 3.25 mm, variant S2; "
                 "highest -4.82 % from the bench: beam, 16 voxels through the thickness")
BENCH_ALONE = ("-54 % to +13 % (lower end, measured -53.32 %: the hook, pull_off, voxel 1.2 mm, against P2; upper end, measured "
               "+12.95 %: plate with a hole, 8 voxels across the hole, gross scaling of the reference; data: bench/stress_accuracy.json)")


def label(c):
    """The name of a cell, written here from its fields."""
    return f"{c['part']}, {c['case']}, run type {c['run_type']}, voxel {c['voxel_mm']:g} mm" + (
        "" if c["variant"] == "base" else f", variant {c['variant']}")


def entries():
    p999 = sr.safety_entry("p99_9", 10.0, 2.0, BIG, MEMBER)
    peak = sr.safety_entry("peak", 10.0, 3.0, BIG, MEMBER)
    assert p999 and p999["range_estimate"] and peak
    return p999, peak, {"against_limit": {"p99_9": p999}}


def band_texts():
    """Every text of the stress report that gives the band of p99.9, as a user gets it."""
    p999, _, factors = entries()
    return {
        "reason": sr.ACCURACY["p99_9"]["reason"],
        "verdict status": sr.verdict("p99_9", 1.0, 10.0, MEMBER, 20000)["status"],
        "safety factor status": p999["status"],
        "safety factor text": sr.safety_text(CASE, BIG, factors, MEMBER),
    }


def peak_texts():
    """Every text of the stress report that gives the voxel peak below the reference as a range, as a user gets it."""
    _, peak, factors = entries()
    same = {**BIG, "p99_9_equals_peak": True}
    case = {"name": "pull_off", "stress_limit_MPa": 10.0, "relaxed": None,
            "physical": {"voxels": 20000, "peak_MPa": 3.0, "voxels_above_limit": 5}, "member_check": {"p10_voxels": 4.0}}
    return {
        "reason": sr.ACCURACY["peak"]["reason"],
        "verdict status": sr.verdict("peak", 3.0, 10.0, MEMBER, 20000)["status"],
        "safety factor note": peak["note"],
        "safety factor text": sr.safety_text(CASE, BIG, factors, MEMBER),
        "safety factor text, p99.9 equals the peak": sr.safety_text(CASE, same, None, MEMBER),
        "case summary": sr.case_summary("pull_off", case),
    }


def test_the_fact_in_the_package_is_the_fact_of_the_data_file(pp, doc):
    s = doc["summary"]
    assert sr.PUBLIC_PARTS_FACT == pp.fact_text(s)
    assert sr.PUBLIC_PARTS_FILE == pp.SOURCE == "bench/stress_public_parts.json" and DATA.relative_to(ROOT).as_posix() == pp.SOURCE
    # every number of the fact is a number of the summary (the formatter makes it; this reads it back)
    out = s["cells_outside_bench_band"]["total"]
    for piece in (f"{s['n_parts']} public parts ({s['n_cells']} cells)",
                  f"{s['voxel_error_pct']['min']:+.0f} % ({s['voxel_error_pct']['min_cell']}) to "
                  f"{s['voxel_error_pct']['max']:+.0f} % ({s['voxel_error_pct']['max_cell']})",
                  f"above P2 in {s['cells_voxel_above_p2']} cells",
                  f"up to {s['p2_over_voxel']['max']:.2g} times below P2",
                  f"{out} of the {s['n_cells']} cells were outside the bench band (-54 % to +13 %, bench/stress_accuracy.json)",
                  f"{s['member_gate']['cells_above_bench_band']} of the {s['member_gate']['cells_meeting']} cells",
                  f"above +{s['bench_band']['over_pct']:g} %"):
        assert piece in sr.PUBLIC_PARTS_FACT, piece
    # "outside the band" is about the bench band: the band of the report contains every cell
    assert out == 15 and "outside the bench band" in sr.PUBLIC_PARTS_FACT and "outside the band" not in sr.PUBLIC_PARTS_FACT
    # the words of the fact are true for the cells: both directions, mostly mass objective, a coarse part with thin members
    assert s["cells_voxel_above_p2"] > 0 and s["cells_voxel_below_p2"] > 0
    assert s["cells_voxel_above_p2_in_mass_objective_runs"] > s["cells_voxel_above_p2"] / 2
    worst = max(doc["cells"], key=lambda c: c["p2_over_voxel"])
    same_part = [c for c in doc["cells"] if c["part"] == worst["part"]]
    assert worst["voxel_mm"] == max(c["voxel_mm"] for c in same_part) and worst["thinnest_member_voxels"] < 2.0
    assert worst["voxels_in_scope"] < 1400 and worst["member_p10_voxels"] < sr.MIN_MEMBER_VOXELS_EST
    assert s["member_gate"]["cells_above_bench_band"] > 0           # "did not keep the error inside the bench band"
    assert label(worst) == s["voxel_error_pct"]["min_cell"]         # the worst ratio is the lower end of the error range


def test_the_band_is_the_union_of_the_bench_rows_and_the_public_cells(doc):
    """59 / 57 for p99.9 and 88 / 0 for the peak, from the rows of `bench/stress_accuracy.json` plus the cells (F1-8c)."""
    bench = json.loads((BENCH / "stress_accuracy.json").read_text())
    beam = [r for r in bench["beam"]["sizes"] if r["voxels_through_thickness"] >= 4]
    hole = [r for r in bench["hole"]["sizes"] if r["voxels_across_hole"] >= 4]
    cross = list(bench["cross_check"]["cases"].values())
    b_p999 = ([r["whole_design"]["p99_9_error_pct"] for r in beam] + [r["p99_9_error_pct"] for r in hole]
              + [r["p99_9_error_vs_gross_scaling_pct"] for r in hole] + [c["p99_9_error_pct"] for c in cross])
    b_peak = ([r["whole_design"]["peak_error_pct"] for r in beam] + [r["peak_error_pct"] for r in hole]
              + [c["peak_error_pct"] for c in cross])
    p_p999 = [c["voxel_error_pct"] for c in doc["cells"]]
    p_peak = [100.0 * (1.0 / c["p2_peak_over_voxel_peak"] - 1.0) for c in doc["cells"]]
    edge = lambda e: (math.ceil(max(0.0, -min(e))), math.ceil(max(0.0, max(e))))       # the rounding rule of `derive_rule`
    assert len(b_p999) == 11 and len(p_p999) == 51
    assert edge(b_p999) == (54, 13) and edge(b_peak) == (72, 0)             # the bench alone: the bands of #250
    assert edge(p_p999) == (59, 57) and edge(p_peak) == (88, 0)             # the public cells alone
    assert edge(b_p999 + p_p999) == (59, 57) and edge(b_peak + p_peak) == (88, 0)      # the union
    for stat, rows, band in (("p99_9", b_p999 + p_p999, (59, 57)), ("peak", b_peak + p_peak, (88, 0))):
        acc = sr.ACCURACY[stat]
        assert (acc["under_pct"], acc["over_pct"]) == band and acc["verdict"] is False
        assert acc["measured_min_pct"] == pytest.approx(min(rows), abs=0.006) and acc["measured_max_pct"] == pytest.approx(max(rows), abs=0.006)
        assert (acc["min_voxels"], acc["min_voxels_hole_diameter"]) == (4, 4)
    # the sets apart: the bench ends are those of #250, the public ends are the cells, and each end of the union has its set
    src99, srcpk = sr.ACCURACY["p99_9"]["sources"], sr.ACCURACY["peak"]["sources"]
    assert (src99["bench"]["under_pct"], src99["bench"]["over_pct"], src99["bench"]["measured_min_pct"], src99["bench"]["measured_max_pct"]) == (54, 13, -53.32, 12.95)
    assert (srcpk["bench"]["under_pct"], srcpk["bench"]["over_pct"], srcpk["bench"]["measured_min_pct"], srcpk["bench"]["measured_max_pct"]) == (72, 0, -71.46, -4.82)
    assert (src99["public"]["under_pct"], src99["public"]["over_pct"]) == (59, 57) and (srcpk["public"]["under_pct"], srcpk["public"]["over_pct"]) == (88, 0)
    assert (src99["lower_end"], src99["upper_end"]) == ("public", "public") and (srcpk["lower_end"], srcpk["upper_end"]) == ("public", "bench")
    assert (src99["public"]["parts"], src99["public"]["cells"]) == (5, 51)
    # the cell of each public end is the extreme cell
    by_p999 = sorted(doc["cells"], key=lambda c: c["voxel_error_pct"])
    assert src99["public"]["measured_min_from"] == label(by_p999[0]) and src99["public"]["measured_max_from"] == label(by_p999[-1])
    by_peak = sorted(doc["cells"], key=lambda c: -c["p2_peak_over_voxel_peak"])
    assert srcpk["public"]["measured_min_from"] == label(by_peak[0]) and srcpk["public"]["measured_max_from"] == label(by_peak[-1])
    assert (src99["bench"]["measured_min_from"], src99["bench"]["measured_max_from"]) == (
        "the hook, pull_off, voxel 1.2 mm, against P2", "plate with a hole, 8 voxels across the hole, gross scaling of the reference")
    assert srcpk["bench"]["measured_max_from"] == "beam, 16 voxels through the thickness"
    # the data base counts the public cells too
    db = sr.ACCURACY["p99_9"]["data_base"]
    assert sr.ACCURACY["peak"]["data_base"] == db
    assert db["public"] == {"parts": 5, "cells": 51, "cells_where_p99_9_is_a_true_percentile": 20,
                            "cells_where_p99_9_equals_the_peak": sum(c["p99_9_equals_peak"] for c in doc["cells"])}
    assert db["public"]["cells_where_p99_9_equals_the_peak"] == 31
    assert (db["rows"], db["rows_where_p99_9_is_a_true_percentile"], db["rows_where_p99_9_equals_the_peak"]) == (62, 27, 35)
    assert "the hook is in both sets" in db["note"]


def test_the_range_estimate_of_the_safety_factor_is_0_41_to_1_57_times_ref_over_value():
    entry = sr.safety_entry("p99_9", 10.0, 2.0, BIG, MEMBER)
    assert entry["value"] == pytest.approx(5.0)
    assert entry["range_estimate"] == pytest.approx([2.05, 7.85])          # [10 * 0.41 / 2, 10 * 1.57 / 2]
    assert [x / entry["value"] for x in entry["range_estimate"]] == pytest.approx([0.41, 1.57])       # #250: 0.46 and 1.13
    # the limit 1.155 and the voxel p99.9 1.457 of the bracket mass run: the factor 0.793, the range 0.325 to 1.245; P2 gives 1.13
    lo, hi = sr.safety_entry("p99_9", 1.155, 1.457, {**BIG, "voxels": 6659}, MEMBER)["range_estimate"]
    assert (round(lo, 3), round(hi, 3)) == (0.325, 1.245) and lo < 1.155 / 1.02 < hi
    # the peak has no range, as before
    assert sr.safety_entry("peak", 10.0, 3.0, BIG, MEMBER)["status"] == "no accuracy band"


def test_each_text_that_gives_the_p99_9_band_names_the_source_of_each_end():
    texts = band_texts()
    assert len(texts) == 4
    for name, text in texts.items():
        assert P999_LOWER in text and P999_UPPER in text, name
        assert "data: bench/stress_public_parts.json" in text and "-59 % to +57 %" in text, name
        assert "topopt verify" in text and "not a bound for other parts" in text, name
    assert "no verdict" in texts["safety factor status"] and "bench 3 shapes, 1 real part; 5 public parts, 51 cells" in texts["safety factor status"]
    # the reason gives the bench alone with its own ends, and the data base of both sets
    reason = texts["reason"]
    assert f"The bench alone gave {BENCH_ALONE}" in reason
    assert "4 of the 11 rows and 31 of the 51 cells have fewer than 5000 voxels" in reason
    assert "the member gate does not predict" in reason.lower() and "the hook at 1.2 mm passes it (p10 3.84 voxels)" in reason
    # the safety text gives the band once, also when the limit and the allowable both have a range
    both = sr.safety_text({"stress_limit_MPa": 10.0, "allowable_mpa": 40.0}, BIG, {
        "against_limit": {"p99_9": sr.safety_entry("p99_9", 10.0, 2.0, BIG, MEMBER)},
        "against_allowable": {"p99_9": sr.safety_entry("p99_9", 40.0, 2.0, BIG, MEMBER)}}, MEMBER)
    assert both.count(P999_LOWER) == 1 and both.count("estimated range") == 2
    # a text without a range does not give the p99.9 band (a p99.9 that equals the peak has none)
    same = sr.safety_text(CASE, {**BIG, "p99_9_equals_peak": True}, None, MEMBER)
    assert P999_LOWER not in same and "estimated range" not in same


def test_the_public_fact_is_said_once_in_the_verdict_status_and_the_other_texts_point_to_it():
    texts = band_texts()
    assert sr.PUBLIC_PARTS_FACT in texts["reason"] and sr.PUBLIC_PARTS_FACT in texts["verdict status"]
    assert texts["reason"].count(sr.PUBLIC_PARTS_FACT) == 1
    for name in ("safety factor status", "safety factor text"):
        assert sr.PUBLIC_PARTS_FACT not in texts[name], name
        assert "`verdict.p99_9.status`" in texts[name], name
    assert "`verdict.p99_9.status`" in sr.member_gate_text(MEMBER)


def test_each_text_that_gives_the_peak_range_names_the_source_of_each_end():
    texts = peak_texts()
    assert PEAK_LOWER in texts["reason"] and PEAK_UPPER in texts["reason"] and "-88 % to 0 %" in texts["reason"]
    assert "topopt verify" in texts["reason"] and sr.ACCURACY["peak"]["reason"] in texts["verdict status"]
    for name in ("safety factor note", "safety factor text", "safety factor text, p99.9 equals the peak", "case summary"):
        assert PEAK_MEASURED in texts[name], name
    for name in ("safety factor text", "safety factor text, p99.9 equals the peak"):
        assert "The voxel peak was 5 % to 87 % below the reference in every measured case" in texts[name], name
    assert "the voxel peak was 5 to 87 % below the reference in the measured cases" in texts["case summary"]
    assert "5 to 71 %" not in texts["case summary"] and "-72 %" not in texts["reason"]
    # the note of the limit above: it points to the verdict status, and gives no range
    case_entry = sr.physical_statistics({"case_names": ["a"], "vm_cases": np.array([[1.0, 2.0, 3.0]]), "design": np.array([True] * 3)},
                                        {"a": 2.0})[0]
    assert "`verdict.peak.status`" in case_entry["above_limit_note"] and "can be larger" in case_entry["above_limit_note"]


def test_the_member_gate_text_says_that_the_gate_did_not_keep_the_error_in_the_bench_band():
    for member in (MEMBER, {"met": False, "p10_voxels": 2.0}):
        text = sr.member_gate_text(member)
        assert text.startswith("met") or text.startswith("not met")
        assert "the bench band was measured for 3.5 or more" in text and "did not keep the error inside the bench band" in text
        assert sr.PUBLIC_PARTS_FILE in text and "the source of each end" in text      # the band and its ends are in the verdict status


def test_the_valid_when_text_is_a_condition_of_the_bench_and_no_guarantee():
    t999, tpk = sr.ACCURACY["p99_9"]["valid_when"], sr.ACCURACY["peak"]["valid_when"]
    assert "3.5 voxels or more" in t999 and "not a guarantee" in t999 and sr.PUBLIC_PARTS_FILE in t999
    assert "6 of the 18 cells that met the member gate had an error outside the bench band (-54 % to +13 %, bench/stress_accuracy.json" in t999
    assert "3 of the 18 cells that met the member gate had a voxel peak more than 72 % below the P2 peak" in tpk
    assert "72 % is the lower end of the bench band" in tpk and "not a guarantee" in tpk


def test_rederive_rule_writes_the_committed_rule_again_with_no_bench_run(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("stress_accuracy_bench", BENCH / "stress_accuracy.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    out = tmp_path / "stress_accuracy.json"
    out.write_text((BENCH / "stress_accuracy.json").read_text())
    monkeypatch.setattr(sys, "argv", ["stress_accuracy.py", "--rederive-rule", "--out", str(out)])
    bench.main()
    assert out.read_text() == (BENCH / "stress_accuracy.json").read_text()
    # the package copy is the rule of the file (the whole entry, with its sources and its texts)
    doc = json.loads(out.read_text())
    assert {k: v for k, v in doc["verdict_rule"].items() if k != "error_definition"} == sr.ACCURACY
    # without the public cells the rule is the band of the bench alone, with the text of the bench alone
    alone = bench.derive_rule(doc["beam"]["sizes"], doc["hole"]["sizes"], doc["cross_check"])
    assert "public" not in alone["p99_9"]["reason"] and "public" not in alone["p99_9"]["valid_when"]
    assert "public" not in alone["peak"]["reason"] and "public" not in alone["p99_9"]["data"]
    assert [(alone[k]["under_pct"], alone[k]["over_pct"]) for k in ("p99_9", "peak")] == [(54, 13), (72, 0)]
    assert alone["p99_9"]["sources"]["bench"] == doc["verdict_rule"]["p99_9"]["sources"]["bench"]
    assert set(alone["p99_9"]["sources"]) == {"bench", "lower_end", "upper_end"} and alone["p99_9"]["sources"]["upper_end"] == "bench"
    assert "public" not in alone["p99_9"]["data_base"] and alone["p99_9"]["data_base"]["rows"] == 11
    assert alone["error_definition"] == doc["verdict_rule"]["error_definition"]


def test_the_docs_that_state_the_band_quote_the_data(doc):
    s, g = doc["summary"], doc["summary"]["member_gate"]
    err = f"{s['voxel_error_pct']['min']:+.0f} % to {s['voxel_error_pct']['max']:+.0f} %"
    common = (err, "bench/stress_public_parts.json", "topopt verify", f"{s['n_parts']} public parts", f"{s['n_cells']} cells",
              f"{s['cells_voxel_above_p2']} cells",
              f"up to {s['p2_over_voxel']['max']:.2g} times below P2")
    a99, apk = sr.ACCURACY["p99_9"], sr.ACCURACY["peak"]
    union99 = f"{-a99['under_pct']:+d} % to {a99['over_pct']:+d} %"                 # -59 % to +57 %
    assert union99 == "-59 % to +57 %" and (-apk["under_pct"], apk["over_pct"]) == (-88, 0)
    ends99 = (f"{a99['sources']['public']['measured_min_pct']:.2f} %", f"{a99['sources']['public']['measured_max_pct']:+.2f} %")
    peak_ends = (f"{-apk['measured_max_pct']:.0f} % to {-apk['measured_min_pct']:.0f} %", f"{apk['measured_min_pct']:.2f} %",
                 f"{apk['measured_max_pct']:.2f} %")
    assert ends99 == ("-58.32 %", "+56.91 %") and peak_ends == ("5 % to 87 %", "-87.14 %", "-4.82 %")
    for name in ("docs/reference/problem-and-run.md", "CONTRACTS.md"):   # the stress text of the README is on this page since #294
        text = " ".join((ROOT / name).read_text().split())             # the line breaks of the document do not matter
        for piece in common + (union99, *ends99, *peak_ends, "-54 % to +13 %", "run type MS", "run type MF", "not a bound for other parts"):
            assert piece in text, (name, piece)
    contracts = " ".join((ROOT / "CONTRACTS.md").read_text().split())
    out = s["cells_outside_bench_band"]
    for piece in (f"({s['cells_voxel_above_p2_in_mass_objective_runs']} of the {s['cells_voxel_above_p2']})",
                  f"{out['total']} of the {s['n_cells']} cells were outside the bench band ({out['voxel_below_the_band']} below it, "
                  f"{out['voxel_above_the_band']} above it",
                  f"{g['cells_meeting']} cells met it (P2 / voxel {g['p2_over_voxel']['min']:.2f} to {g['p2_over_voxel']['max']:.2f}), "
                  f"and {g['cells_above_bench_band']} of them were above +{s['bench_band']['over_pct']:g} %",
                  f"all {s['voxel_peak']['cells_below_p2_peak']} cells",
                  "`range_estimate` is `[reference * (1 - 0.59) / p99.9, reference * (1 + 0.57) / p99.9]`, that is 0.41 to 1.57 times",
                  "`verdict.<stat>.error_band_pct` is `{measured_min_pct, measured_max_pct, data, sources, data_base}`",
                  "`lower_end` and `upper_end` say which set", "--resummarize"):
        assert piece in contracts, piece
    # the hook example: the P2 / voxel ratio of pull_off in the hook cells, the band, and the cell of the lower end
    hook = [c["p2_over_voxel"] for c in doc["cells"] if c["part"] == "hook" and c["case"] == "pull_off"]
    assert len(hook) == 4
    example = " ".join((ROOT / "examples" / "hook" / "README.md").read_text().split())
    assert f"was {min(hook):.2g} to {max(hook):.2g} times the voxel p99.9 in four run types" in example
    for piece in (union99, ends99[0], "run type MS", "bench/stress_public_parts.json", "topopt verify"):
        assert piece in example, piece
