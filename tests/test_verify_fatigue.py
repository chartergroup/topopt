"""The Goodman function and record of `topopt.verify.fatigue` (B4a, plan section 5 and 10.1). No extra is needed."""
import math

import pytest

from topopt.config import BoreChecks, BoreSpec, GoodmanMaterial, Cup
from topopt.verify import fatigue as F

SU, SY, SE = 1100.0, 1000.0, 350.0


def test_hand_values():
    n, y = F.goodman_point(100.0, 200.0, SU, SY, SE)
    assert n == 1.0 / (100 / 350 + 200 / 1100) and math.isclose(n, 77 / 36, rel_tol=1e-15) and y == 0.3
    assert F.goodman_point(100.0, 0.0, SU, SY, SE) == (1.0 / (100 / 350), 0.1)     # n = Se/sa = 3.5
    assert math.isclose(F.goodman_point(100.0, 0.0, SU, SY, SE)[0], 3.5, rel_tol=1e-15)
    assert math.isclose(F.goodman_point(0.0, 200.0, SU, SY, SE)[0], 5.5, rel_tol=1e-15)        # n = Su/sm
    n, y = F.goodman_point(100.0, -100.0, SU, SY, SE)
    assert math.isclose(n, 5.133333333333334, rel_tol=1e-15)
    assert y == 0.2                       # (|sm| + sa) / Sy: the plain sum would give 0.0 for a peak magnitude of 200 MPa
    assert F.goodman_point(0.0, 0.0, SU, SY, SE) == (None, 0.0)


def test_a_row_of_the_pipeline_is_reproduced_bit_for_bit():
    """H1025, rigid cup, upper bore, max: sa and sm of goodman.json give its n and its yield ratio (the same operations)."""
    n, y = F.goodman_point(234.07824044834175, 690.8334332103268, SU, SY, SE)
    assert n == 0.7711138490992635 and y == 0.9249116736586686


def test_worst_case_is_the_smallest_n_and_the_first_wins_a_tie():
    r = F.goodman_row({"a": 100.0, "b": 150.0, "c": 150.0}, 200.0, SU, SY, SE)
    assert r["worst_case"] == "b" and r["sa_MPa"] == 150.0 and r["sm_MPa"] == 200.0 and not r["all_n_none"]
    assert [c["case"] for c in r["cases"]] == ["a", "b", "c"] and r["n_goodman"] == r["cases"][1]["n_goodman"]
    assert r["peak_over_yield"] == (200.0 + 150.0) / 1000.0


def test_a_case_with_no_n_is_skipped_in_the_selection():
    # sm = -400: sa/Se + sm/Su is below 0 for sa = 100 (100/350 - 400/1100 < 0) and above 0 for sa = 150
    r = F.goodman_row({"a": 100.0, "b": 150.0, "c": 150.0}, -400.0, SU, SY, SE)
    assert r["cases"][0]["n_goodman"] is None
    assert r["worst_case"] == "b" and r["n_goodman"] is not None and not r["all_n_none"]


def test_all_cases_with_no_n_gives_null_and_the_largest_yield_ratio():
    r = F.goodman_row({"a": 100.0, "b": 120.0, "c": 100.0}, -400.0, SU, SY, SE)
    assert r["n_goodman"] is None and r["all_n_none"]
    assert r["worst_case"] == "b" and r["peak_over_yield"] == (400.0 + 120.0) / 1000.0
    # the first case wins a tie of the yield ratio
    r = F.goodman_row({"a": 100.0, "c": 100.0}, -400.0, SU, SY, SE)
    assert r["worst_case"] == "a"


# --- the record ----------------------------------------------------------------------------------------------------

def stats(vm_p, vm_m, hoop_p=1.0, hoop_m=2.0):
    return {"n_qp": 50, "vm_p99_9_MPa": vm_p, "vm_max_MPa": vm_m, "vm_median_MPa": vm_p / 2, "vm_min_MPa": 0.0,
            "hoop_p99_9_MPa": hoop_p, "hoop_max_MPa": hoop_m, "hoop_median_MPa": 0.5, "hoop_min_MPa": 0.0}


def bore(name, mean=None, interference=None):
    return BoreSpec(name, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 8.0, "yaml", interference, mean)


MATS = (GoodmanMaterial("A", 1100.0, 1000.0, 350.0, (("se_mpa", "src A"),)), GoodmanMaterial("B", 1300.0, 1150.0, 400.0))
SERVICE = {"c1": {"b1": stats(100.0, 150.0), "b2": stats(60.0, 90.0)}, "c2": {"b1": stats(120.0, 140.0), "b2": stats(70.0, 80.0)}}


def test_record_without_press_fit_and_without_mean_stress():
    bc = BoreChecks((bore("b1"), bore("b2")), materials=MATS, goodman=True)
    rec = F.build_record(bc, ["c1", "c2"], SERVICE)
    assert rec["status"] == "done" and rec["kind"] == "goodman" and rec["rule"] == F.RULE and rec["statistics"] == ["p99_9", "max"]
    assert rec["mean_stress"]["from"] == "zero_mean" and rec["mean_stress"]["by_bore"] == {"b1": "zero_mean", "b2": "zero_mean"}
    assert [r["id"] for r in rec["rows"]] == [f"{m}/zero_mean/{b}/{s}" for m in "AB" for b in ("b1", "b2") for s in ("p99_9", "max")]
    r = rec["rows"][0]                                          # A / b1 / p99_9: the worst case is c2 (sa 120)
    assert r["worst_case"] == "c2" and r["sa_MPa"] == 120.0 and r["sm_MPa"] == 0.0 and r["n_goodman"] == 1.0 / (120.0 / 350.0 + 0.0 / 1100.0)
    assert [c["case"] for c in r["cases"]] == ["c1", "c2"]
    assert rec["materials"]["A"] == dict(su_mpa=1100.0, sy_mpa=1000.0, se_mpa=350.0, sources={"se_mpa": "src A"})
    assert rec["service_stress"]["c1"]["b2"]["vm_p99_9_MPa"] == 60.0 and rec["missing_rows"] == []
    assert rec["lowest_n"]["id"] == "A/zero_mean/b1/max" and rec["lowest_n"]["value"] == min(x["n_goodman"] for x in rec["rows"]) == 350.0 / 150.0
    assert any(f.startswith("Rule: ") for f in rec["facts"]) and F.FACT_MIX in rec["facts"]


def test_record_with_a_yaml_mean_stress():
    bc = BoreChecks((bore("b1", mean=200.0), bore("b2")), materials=MATS[:1], goodman=True)
    rec = F.build_record(bc, ["c1", "c2"], SERVICE)
    assert rec["mean_stress"]["from"] == "yaml_mean" and rec["mean_stress"]["by_bore"] == {"b1": "yaml_mean", "b2": "zero_mean"}
    r = rec["rows"][0]
    assert r["id"] == "A/yaml_mean/b1/p99_9" and r["sm_MPa"] == 200.0 and r["n_goodman"] == 1.0 / (120.0 / 350.0 + 200.0 / 1100.0)
    assert rec["rows"][2]["id"] == "A/zero_mean/b2/p99_9" and rec["rows"][2]["sm_MPa"] == 0.0


def test_n_below_one_and_negative_mean_stress_are_facts():
    bc = BoreChecks((bore("b1", mean=-100.0),), materials=MATS[:1], goodman=True)
    svc = {"c1": {"b1": stats(2000.0, 2100.0)}}
    rec = F.build_record(bc, ["c1"], svc)
    assert rec["lowest_n"]["value"] < 1.0
    assert any("n is below 1 in" in f and "not a limit of topopt" in f for f in rec["facts"])
    assert any("sm is below 0 in 2 row(s)" in f for f in rec["facts"]) and F.OPTION_NEGATIVE_SM in rec["options"]


def test_all_null_row_has_a_fact():
    bc = BoreChecks((bore("b1", mean=-400.0),), materials=MATS[:1], goodman=True)
    rec = F.build_record(bc, ["c1"], {"c1": {"b1": stats(100.0, 100.0)}})
    assert all(r["n_goodman"] is None for r in rec["rows"]) and rec["lowest_n"] is None
    assert any("every load case has sa/Se + sm/Su at or below 0" in f for f in rec["facts"])


def test_press_fit_block_without_a_result_has_no_rows_and_names_them():
    """A press fit with no result (a skipped stage, or one that is not built) never gets a row with sm = 0 (REVIEW M4)."""
    bc = BoreChecks((bore("b1", interference=0.1), bore("b2", interference=0.1)), materials=MATS, cups=(Cup("rigid", k_mpa_per_mm=1e6), Cup("al", 7e4, 0.3, 1.0)),
                    goodman=True, press_fit=True)
    rec = F.build_record(bc, ["c1", "c2"], SERVICE, press_reason="not_built", press_options=["B4b adds the press-fit solve."])
    assert rec["status"] == "done" and rec["rows"] == [] and rec["mean_stress"]["from"] == "press_fit"
    assert [(m["variant"], m["bore"], m["reason"]) for m in rec["missing_rows"]] == [("rigid", "b1", "not_built"), ("rigid", "b2", "not_built"),
                                                                                     ("al", "b1", "not_built"), ("al", "b2", "not_built")]
    assert rec["missing_rows"][0]["options"] == ["B4b adds the press-fit solve."]
    assert "B4b adds the press-fit solve." in rec["options"] and any("no Goodman row is given" in f for f in rec["facts"])


def test_press_fit_result_gives_rows_per_variant_and_bore_and_a_missing_bore_is_named():
    bc = BoreChecks((bore("b1", interference=0.1), bore("b2", interference=0.1)), materials=MATS[:1], cups=(Cup("rigid", k_mpa_per_mm=1e6),),
                    goodman=True, press_fit=True)
    hoop = {"rigid": {"b1": {"p99_9": 300.0, "max": 400.0}}}                  # no result for b2
    rec = F.build_record(bc, ["c1", "c2"], SERVICE, hoop=hoop, press_reason="partial")
    assert [r["id"] for r in rec["rows"]] == ["A/rigid/b1/p99_9", "A/rigid/b1/max"]
    assert rec["rows"][0]["sm_MPa"] == 300.0 and rec["rows"][1]["sm_MPa"] == 400.0
    assert [(m["variant"], m["bore"]) for m in rec["missing_rows"]] == [("rigid", "b2")]


def test_dropped_bore_has_no_rows_and_is_named():
    bc = BoreChecks((bore("b1"), bore("b2")), materials=MATS[:1], goodman=True)
    svc = {c: {"b1": v["b1"]} for c, v in SERVICE.items()}
    rec = F.build_record(bc, ["c1", "c2"], svc, dropped=[dict(bore="b2", reason="no_facets", options=["check the geometry"])])
    assert {r["bore"] for r in rec["rows"]} == {"b1"} and rec["missing_rows"] == [dict(variant=None, bore="b2", reason="no_facets", options=["check the geometry"])]


def test_no_bore_gives_a_skipped_record():
    bc = BoreChecks((bore("b1"),), materials=MATS[:1], goodman=True)
    rec = F.build_record(bc, ["c1"], {"c1": {}}, dropped=[dict(bore="b1", reason="no_facets")])
    assert rec["status"] == "skipped" and rec["reason"] == "no_bore" and rec["rows"] == [] and rec["facts"] and rec["options"]
    assert rec["missing_rows"] == [dict(variant=None, bore="b1", reason="no_facets", options=[])]        # the dropped bores are named, as in a done record (review n-4)


def test_the_module_has_no_import_of_an_extra_and_no_part_name():
    import pathlib
    src = pathlib.Path(F.__file__).read_text()
    for word in ("skfem", "gmsh", "cholespy", "numpy", "H1025", "H900", "braking", "upper", "lower", "rigid"):
        assert word not in src, word


def test_error_record():
    rec = F.error_record("ValueError: x")
    assert rec["status"] == "error" and rec["reason"] == "exception" and "ValueError: x" in rec["facts"][0] and rec["rows"] == []


# --- the scatter of the bore statistics (review m-2 of B4a) --------------------------------------------------------

def test_the_record_says_that_n_carries_the_scatter_and_levels_is_the_first_option():
    bc = BoreChecks((bore("b1"), bore("b2")), materials=MATS[:1], goodman=True)
    rec = F.build_record(bc, ["c1", "c2"], SERVICE)
    assert F.FACT_SCATTER in rec["facts"]
    for words in ("p99.9 and the max", "not smoothed", "mesh realization, not only on the mesh size", "median", "CONTRACTS.md section 40.7"):
        assert words in F.FACT_SCATTER, words
    assert rec["options"][0].startswith("`--levels`") and "not a sure cure" in rec["options"][0]


def test_the_levels_option_follows_the_style_of_the_reader():
    from topopt.verify import names
    bc = BoreChecks((bore("b1"),), materials=MATS[:1], goodman=True)
    with names.use_style("mcp"):
        rec = F.build_record(bc, ["c1"], {"c1": {"b1": stats(100.0, 150.0)}})
    assert rec["options"][0].startswith("`levels_mm`")


def test_the_bore_lines_give_the_median_next_to_p99_9_and_max_of_the_bore_own_sample_set():
    """`stats(60, 90)` has the median 30 (`vm_p99_9 / 2`): the line of a bore reads the median of that bore, not another bore's."""
    from topopt.verify.report import bore_lines
    bc = BoreChecks((bore("b1"), bore("b2")), materials=MATS[:1], goodman=True)
    rec = F.build_record(bc, ["c1", "c2"], SERVICE)
    entry = lambda name: dict(name=name, status="ok", geometry_from="yaml", facets=10, area_mm2=5.0, area_over_analytic=0.9, facts=[], options=[],      # noqa: E731
                              sample_set=dict(tets=4, points=50, volume_mm3=1.0, band_mm=1.2))
    lines = bore_lines([entry("b1"), entry("b2")], rec)
    assert lines[0].startswith("bore b1 (yaml): 10 wall facets") and lines[3].startswith("bore b2 (yaml): 10 wall facets")
    assert lines[1] == "  case c1: von Mises at the sample set: p99.9 100, max 150, median 50 MPa"
    assert lines[2] == "  case c2: von Mises at the sample set: p99.9 120, max 140, median 60 MPa"
    assert lines[4] == "  case c1: von Mises at the sample set: p99.9 60, max 90, median 30 MPa"
    assert lines[5] == "  case c2: von Mises at the sample set: p99.9 70, max 80, median 35 MPa"
    # with no Goodman record (a press_fit-only block) or a record that did not finish, no stress line is printed
    assert len(bore_lines([entry("b1")])) == 1 and len(bore_lines([entry("b1")], F.error_record("x"))) == 1


# --- a negative sm also gives n for sm = 0 (review m-6 of B4a) ------------------------------------------------------

def test_a_row_with_a_negative_sm_also_gives_n_at_sm0():
    """The example of the review: sa 150, sm -400, Se 350, Su 1100 gives n 15.4 on the Goodman line, and 2.33 (Se/sa) for sm = 0."""
    r = F.goodman_row({"a": 150.0}, -400.0, SU, SY, SE)
    assert math.isclose(r["n_goodman"], 15.4, rel_tol=1e-3) and r["n_goodman"] == 1.0 / (150.0 / 350.0 + -400.0 / 1100.0)
    assert math.isclose(r["n_goodman_at_sm0"], 350.0 / 150.0, rel_tol=1e-14) and math.isclose(r["n_goodman_at_sm0"], 2.33, abs_tol=5e-3)
    assert r["worst_case_at_sm0"] == "a"
    # the second example of the review: sa 100, sm -100 gives 5.13 against 3.5
    r = F.goodman_row({"a": 100.0}, -100.0, SU, SY, SE)
    assert math.isclose(r["n_goodman"], 5.1333, abs_tol=1e-4) and math.isclose(r["n_goodman_at_sm0"], 3.5, rel_tol=1e-14)


def test_the_value_at_sm0_has_the_worst_case_of_its_own_and_is_there_when_n_is_null():
    r = F.goodman_row({"a": 100.0, "b": 120.0, "c": 100.0}, -400.0, SU, SY, SE)      # every n is null on the Goodman line (as the test above)
    assert r["n_goodman"] is None and r["all_n_none"]
    assert r["worst_case_at_sm0"] == "b" and math.isclose(r["n_goodman_at_sm0"], 350.0 / 120.0, rel_tol=1e-14)
    assert F.goodman_row({"a": 0.0}, -5.0, SU, SY, SE)["n_goodman_at_sm0"] is None       # no stress at all: no n for sm = 0 either


@pytest.mark.parametrize("sm", [0.0, -0.0, 1.0, 200.0])
def test_a_row_with_sm_of_0_or_above_has_neither_key(sm):
    r = F.goodman_row({"a": 150.0}, sm, SU, SY, SE)
    assert "n_goodman_at_sm0" not in r and "worst_case_at_sm0" not in r


def test_the_record_and_the_text_give_n_at_sm0_in_a_row_with_a_negative_mean_stress():
    from topopt.verify.report import goodman_lines
    bc = BoreChecks((bore("b1", mean=-400.0), bore("b2")), materials=MATS[:1], goodman=True)
    rec = F.build_record(bc, ["c1", "c2"], {"c1": {"b1": stats(150.0, 100.0), "b2": stats(150.0, 100.0)},
                                            "c2": {"b1": stats(120.0, 90.0), "b2": stats(120.0, 90.0)}})
    by_id = {r["id"]: r for r in rec["rows"]}
    neg, pos = by_id["A/yaml_mean/b1/p99_9"], by_id["A/zero_mean/b2/p99_9"]
    assert math.isclose(neg["n_goodman"], 15.4, rel_tol=1e-3) and math.isclose(neg["n_goodman_at_sm0"], 350.0 / 150.0, rel_tol=1e-14)
    assert neg["worst_case_at_sm0"] == "c1" and "n_goodman_at_sm0" not in pos
    fact = next(f for f in rec["facts"] if f.startswith("sm is below 0 in 2 row(s)"))
    assert "flat line" in fact and "n_goodman_at_sm0" in fact and "worst_case_at_sm0" in fact
    text = "\n".join(goodman_lines(rec))
    assert "A/yaml_mean/b1/p99_9: worst case c1, sa 150 MPa, sm -400 MPa, n 15.4 (n at sm = 0: 2.333), yield ratio" in text
    assert "A/zero_mean/b2/p99_9: worst case c1, sa 150 MPa, sm 0 MPa, n 2.333, yield ratio" in text


# --- the levels block takes the row ids from the last level (review n-5 of B4a) ------------------------------------------

def test_goodman_n_of_the_levels_block_has_the_row_ids_of_the_last_level():
    from topopt.verify.run import _add_levels_of_the_checks

    def level(rows):
        return {"cases": {"c": {}}, "goodman": {"status": "done", "rows": [{"id": i, "n_goodman": n} for i, n in rows]}}
    coarse, fine = level([("only_coarse", 2.0), ("both", 3.0)]), level([("both", 4.0), ("only_fine", 5.0)])
    out = {"rows": [{}, {}], "relative_change": [{}], "relative_change_first_to_last": {}}
    _add_levels_of_the_checks(out, [coarse, fine])
    assert out["rows"][0]["goodman_n"] == {"both": 3.0, "only_fine": None}          # no "only_coarse"; no value at the coarse level for "only_fine"
    assert out["rows"][1]["goodman_n"] == {"both": 4.0, "only_fine": 5.0}
    assert out["relative_change_first_to_last"]["goodman_n"] == {"both": pytest.approx(1.0 / 3.0), "only_fine": None}
