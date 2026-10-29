"""The print report: the keys `partial` and `coverage` of the `print` section of metrics.json (TC-2, CONTRACTS.md section 44).

The five sub-checks are `bed_fit`, `overhang`, `plate_contact`, `support_volume` and `thinnest_member`. The tests force one failure at a
time with a monkeypatch. A printer type that needs no support does not ask for `support_volume`: it is `not_run` and the section is not
partial. A section that could not be made at all gives `{"ok": false, "error": ...}` with one failed check, `print_report`.
"""
import json
import sys
from pathlib import Path

import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).parent))

from topopt import print_report as pr  # noqa: E402
from topopt.config import PrintSettings  # noqa: E402

COMB = Path(__file__).resolve().parent / "fixtures" / "comb.stl"
FIVE = ["bed_fit", "overhang", "plate_contact", "support_volume", "thinnest_member"]


@pytest.fixture(scope="module")
def comb():
    return trimesh.load(COMB, force="mesh")


def test_a_clean_report_is_not_partial(comb):
    r = pr.build_report(comb, PrintSettings())
    assert r["partial"] is False
    assert r["coverage"] == {"ran": FIVE, "failed": [], "not_run": []}
    assert list(r)[:3] == ["ok", "partial", "coverage"]
    assert "coverage" in r["definitions"] and "partial" in r["definitions"]["coverage"]
    json.dumps(r)


def test_a_failed_thickness_estimate_is_one_failed_check_and_the_other_numbers_stand(comb, monkeypatch):
    clean = pr.build_report(comb, PrintSettings())

    def boom(*a, **k):
        raise ValueError("no rtree")
    monkeypatch.setattr(pr, "thickness_estimate", boom)
    r = pr.build_report(comb, PrintSettings())
    assert r["ok"] is True and r["partial"] is True
    assert r["coverage"]["failed"] == [{"name": "thinnest_member", "reason": "ValueError: no rtree"}]
    assert r["coverage"]["ran"] == FIVE[:-1] and r["coverage"]["not_run"] == []
    thin = r["thinnest_member"]
    assert thin["error"] == "ValueError: no rtree" and thin["thinnest_member_mm"] is None and thin["samples"] == 0
    assert set(thin) - {"error"} == set(clean["thinnest_member"])             # the same keys as a block with no chord
    assert r["configured"] == clean["configured"] and r["options"] == clean["options"]      # no other number changes


def test_a_printer_type_that_needs_no_support_does_not_ask_for_the_support_volume(comb):
    r = pr.build_report(comb, PrintSettings(printer_type="sls", given=("printer_type",)))
    assert r["partial"] is False and r["coverage"]["failed"] == []
    assert r["coverage"]["ran"] == [x for x in FIVE if x != "support_volume"]
    [x] = r["coverage"]["not_run"]
    assert x["name"] == "support_volume" and x["asked"] is False and "no support needed" in x["reason"]


def test_a_mesh_with_no_measurable_chord_is_asked_and_not_run(comb, monkeypatch):
    real = pr.thickness_estimate

    def none_measured(g, *a, **k):
        out = real(g, *a, **k)
        out.update(thinnest_member_mm=None, min_mm=None, samples=0, percentiles_mm=None, area_below_mm2=None, area_below_pct=None)
        return out
    monkeypatch.setattr(pr, "thickness_estimate", none_measured)
    r = pr.build_report(comb, PrintSettings())
    assert r["partial"] is True and r["coverage"]["failed"] == []
    assert r["coverage"]["not_run"] == [{"name": "thinnest_member", "reason": "no chord could be measured on this mesh", "asked": True}]


def test_a_section_that_cannot_be_made_is_a_failed_print_report(comb, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("test fault")
    monkeypatch.setattr(pr, "measure_direction", boom)
    s = pr.section_for_run({"print": {}}, comb)
    assert s["ok"] is False and s["error"] == "RuntimeError: test fault"
    assert s["partial"] is True
    assert s["coverage"] == {"ran": [], "failed": [{"name": "print_report", "reason": "RuntimeError: test fault"}], "not_run": []}
    assert list(s)[:3] == ["ok", "partial", "coverage"]


def test_the_section_of_a_run_has_the_keys(comb):
    s = pr.section_for_run({"print": {}}, comb)
    assert s["ok"] is True and s["partial"] is False and s["coverage"]["ran"] == FIVE


def test_the_text_line_names_a_failure_and_counts_what_was_not_asked(comb, monkeypatch):
    from topopt import check_coverage
    ok = pr.build_report(comb, PrintSettings(printer_type="sls", given=("printer_type",)))
    assert check_coverage.line_of(ok) == "partial: no; 4 of 4 checks ran (bed_fit, overhang, plate_contact, thinnest_member); 1 not asked"
    monkeypatch.setattr(pr, "thickness_estimate", lambda *a, **k: (_ for _ in ()).throw(ValueError("no rtree")))
    bad = pr.build_report(comb, PrintSettings())
    assert check_coverage.line_of(bad) == ("partial: 1 of 5 checks failed (thinnest_member: ValueError: no rtree); "
                                           "ran: bed_fit, overhang, plate_contact, support_volume")


def test_the_print_section_tool_has_the_line_as_its_second_line():
    import subprocess
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / "tools" / "print_section.py"), str(COMB)], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    assert "triangles" in lines[0] and lines[1] == "partial: no; 5 of 5 checks ran (" + ", ".join(FIVE) + ")"
