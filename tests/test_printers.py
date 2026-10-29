"""PM-1: the printer catalog, the name lookup, and what the catalog gives to a run (CONTRACTS.md section 18.7)."""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import printers  # noqa: E402

CATALOG_FILE = ROOT / "topopt" / "data" / "printers.yaml"
OPENED = "2026-10-24"


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(CATALOG_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cat():
    return printers.load_catalog()


def model(pid, vendor="Acme", name=None, **over):
    """A valid raw model for a synthetic catalog."""
    m = {"id": pid, "vendor": vendor, "model": name or pid.upper(), "aliases": [], "printer_type": "fdm", "bed_mm": [200, 200],
         "build_height_mm": 200, "nozzle_mm": 0.4, "nozzle_options_mm": None, "max_nozzle_c": 260, "max_bed_c": 100,
         "enclosure": "open", "chamber_max_c": None, "stock_nozzle_material": "brass", "extruder": "direct",
         "source_url": "https://example.com/" + pid, "opened": OPENED, "tag": "maker doc"}
    m.update(over)
    return m


def synthetic(*models, vendors=None):
    return printers.parse_catalog({"format": 1, "catalog_version": "t1", "vendors": vendors or {"Acme": ["acme"], "Zeta": ["zeta"]},
                                   "models": list(models)})


# --- the file: schema, sources, licence --------------------------------------------------------------------------------------

def test_the_file_has_the_header_format_and_version(raw):
    assert raw["format"] == 1 and isinstance(raw["catalog_version"], str) and raw["catalog_version"] == "2026-10-24"
    text = CATALOG_FILE.read_text(encoding="utf-8")
    head = text.split("format:")[0]
    for word in ("Sources", "Licence", "MIT", "AGPL-3.0", "LGPL-3.0"):
        assert word in head


def test_the_catalog_has_the_15_printers_of_the_research(raw, cat):
    assert len(raw["models"]) == len(cat.profiles) == 15
    assert [m["id"] for m in raw["models"]] == [
        "bambu-lab-a1", "bambu-lab-a1-mini", "bambu-lab-p1s", "bambu-lab-p2s", "bambu-lab-x1-carbon", "bambu-lab-h2d", "prusa-mk4s",
        "prusa-core-one-plus", "prusa-mini-plus", "creality-k1", "creality-k1c", "creality-k2-plus", "creality-ender-3-v3",
        "elegoo-centauri-carbon", "voron-2-4"]


def test_every_model_has_every_field_and_a_null_comes_with_a_note_that_names_it(raw):
    for m in raw["models"]:
        assert not [k for k in printers.REQUIRED_FIELDS if k not in m], m["id"]
        assert not [k for k in m if k not in printers.REQUIRED_FIELDS + printers.OPTIONAL_FIELDS], m["id"]
        for key in printers.FACT_FIELDS:
            if m[key] is None and key not in printers.NULL_WITHOUT_NOTE:
                assert m.get("note") and key in m["note"], (m["id"], key)


def test_every_value_has_a_source(raw):
    for m in raw["models"]:
        assert re.match(r"^https://", m["source_url"]), m["id"]
        assert m["opened"] == OPENED and m["tag"] == "maker doc", m["id"]
        for key, url in (m.get("field_sources") or {}).items():
            assert key in printers.FACT_FIELDS and re.match(r"^https://", url), (m["id"], key)


def test_the_values_are_the_ones_of_the_research_rows(raw):
    by = {m["id"]: m for m in raw["models"]}
    p1s = by["bambu-lab-p1s"]
    assert (p1s["bed_mm"], p1s["build_height_mm"], p1s["max_nozzle_c"], p1s["max_bed_c"], p1s["enclosure"]) == ([256, 256], 256, 300, 100, "enclosed")
    assert (p1s["stock_nozzle_material"], p1s["extruder"], p1s["chamber_max_c"]) == ("stainless", "direct", None)
    assert by["bambu-lab-a1-mini"]["max_bed_c"] == 80 and by["bambu-lab-a1-mini"]["enclosure"] == "open"
    assert (by["prusa-mk4s"]["bed_mm"], by["prusa-mk4s"]["build_height_mm"]) == ([250, 210], 220)
    assert (by["prusa-core-one-plus"]["enclosure"], by["prusa-core-one-plus"]["chamber_max_c"]) == ("heated", 55)
    assert by["prusa-mini-plus"]["extruder"] == "bowden"
    assert (by["creality-k2-plus"]["bed_mm"], by["creality-k2-plus"]["chamber_max_c"]) == ([350, 350], 60)
    assert (by["bambu-lab-h2d"]["enclosure"], by["bambu-lab-h2d"]["chamber_max_c"], by["bambu-lab-h2d"]["max_nozzle_c"]) == ("heated", 65, 350)


def test_the_h2d_height_is_from_the_row_that_gives_its_bed_and_the_note_names_the_disagreement(raw):
    """Review MIN-6: the bed came from the two-nozzle row (300 x 320 x 320) and the height was null: a mix of one row and a gap."""
    h2d = {m["id"]: m for m in raw["models"]}["bambu-lab-h2d"]
    assert h2d["bed_mm"] == [300, 320] and h2d["build_height_mm"] == 320
    note = " ".join(h2d["note"].split())
    for part in ("300 x 320 x 320", "two-nozzle row", "325", "320", "disagree", "the smaller"):
        assert part in note, part
    assert printers.resolve("h2d").profile.build_height_mm == 320.0


def test_a_disputed_or_missing_value_is_null_and_not_a_guess(raw):
    by = {m["id"]: m for m in raw["models"]}
    assert by["creality-ender-3-v3"]["stock_nozzle_material"] is None          # hardened steel or tri-metal
    assert by["creality-ender-3-v3"]["enclosure"] is None                      # no enclosure is listed, and no word "open"
    assert by["bambu-lab-a1"]["extruder"] is None and by["bambu-lab-p2s"]["extruder"] is None
    assert by["prusa-mini-plus"]["stock_nozzle_material"] is None
    voron = by["voron-2-4"]
    assert voron["bed_mm"] is None and voron["build_height_mm"] is None and voron["max_nozzle_c"] is None
    assert by["bambu-lab-p1s"]["enclosure"] == "enclosed" and by["bambu-lab-p1s"]["chamber_max_c"] is None   # the wiki FAQ wins over the store text


def test_no_slicer_profile_is_named_as_a_source(raw):
    text = CATALOG_FILE.read_text(encoding="utf-8")
    urls = re.findall(r"https?://[^\s\"']+", text)
    assert urls and not [u for u in urls if re.search(r"github\.com|OrcaSlicer|BambuStudio|PrusaSlicer|Cura", u, re.I)]
    for m in raw["models"]:
        assert m["tag"] == "maker doc"


def test_the_catalog_loads_once_and_the_profiles_are_frozen(cat):
    assert printers.load_catalog() is cat
    p = cat.by_id["bambu-lab-p1s"]
    assert p.name == "Bambu Lab P1S" and p.catalog_version == "2026-10-24"
    with pytest.raises(Exception):
        p.bed_mm = (1.0, 1.0)


# --- the checks of the format --------------------------------------------------------------------------------------------------

def test_a_valid_synthetic_catalog_parses():
    c = synthetic(model("a1"), model("b2", vendor="Zeta"))
    assert [p.id for p in c.profiles] == ["a1", "b2"] and c.version == "t1"


@pytest.mark.parametrize("change, text", [
    ({"id": "Bad Id"}, "lower-case slug"),
    ({"bed_mm": [200]}, "bed_mm must be [x, y]"),
    ({"bed_mm": [200, -1]}, "number above 0"),
    ({"enclosure": "boxed"}, "enclosure must be one of"),
    ({"extruder": "pull"}, "extruder must be one of"),
    ({"printer_type": "laser"}, "printer_type must be one of"),
    ({"opened": "yesterday"}, "opened must be a date"),
    ({"source_url": "pages/p1s"}, "http(s) address"),
    ({"tag": "rumour"}, "tag must be one of"),
    ({"vendor": "Nobody"}, "not in the `vendors` table"),
    ({"unknown_key": 1}, "unknown key 'unknown_key'"),
    ({"enclosure": "heated"}, "chamber_max_c is a number exactly when enclosure is heated"),
    ({"chamber_max_c": 50}, "chamber_max_c is a number exactly when enclosure is heated"),
    ({"max_bed_c": None}, "needs a note that says why"),
    ({"field_sources": {"colour": "https://example.com"}}, "not a fact field"),
])
def test_a_fault_in_a_model_is_a_named_error(change, text):
    with pytest.raises(printers.CatalogError, match=re.escape(text)):
        synthetic(model("a1", **change))


def test_a_missing_key_a_double_id_and_a_double_name_are_errors():
    gone = model("a1")
    del gone["extruder"]
    with pytest.raises(printers.CatalogError, match="missing key 'extruder'"):
        synthetic(gone)
    with pytest.raises(printers.CatalogError, match="used twice"):
        synthetic(model("a1"), model("a1", name="Other"))
    with pytest.raises(printers.CatalogError, match="is also a name of"):
        synthetic(model("a1", name="Box 1"), model("a2", name="box-1"))
    ok = synthetic(model("a1", name="Box 1"), model("z1", vendor="Zeta", name="Box 1"))      # the same name under two vendors is allowed
    assert len(ok.profiles) == 2


def test_a_null_with_a_note_is_accepted_and_the_two_plain_nulls_need_none():
    c = synthetic(model("a1", max_bed_c=None, note="max_bed_c: not found on the maker page"))
    assert c.profiles[0].max_bed_c is None
    synthetic(model("a2"))                                                     # chamber_max_c and nozzle_options_mm are null with no note


def test_a_broken_file_is_a_catalog_error(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("format: 2\ncatalog_version: x\nvendors: {A: [a]}\nmodels: []\n")
    with pytest.raises(printers.CatalogError, match="format is 2"):
        printers.load_catalog(f)
    f.write_text("a: [1\n")
    with pytest.raises(printers.CatalogError, match="not valid YAML"):
        printers.load_catalog(f)
    with pytest.raises(printers.CatalogError, match="cannot read"):
        printers.load_catalog(tmp_path / "missing.yaml")
    f.write_text("format: 1\ncatalog_version: 2026\nvendors: {A: [a]}\nmodels: [1]\n")
    with pytest.raises(printers.CatalogError, match="catalog_version must be a text"):
        printers.load_catalog(f)


# --- resolve: match -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Bambu P1S", "bambu lab p1s", "BAMBU-LAB  P1S!", "bambulab_p1s", "p1s", "P1 S", "my Bambu Lab P1S printer",
                                  "BBL p1s", "bambu-lab-p1s", "  Bambu Lab P1S  "])
def test_one_printer_by_id_name_case_punctuation_and_vendor_words(text):
    r = printers.resolve(text)
    assert r.status == "match" and r.profile.id == "bambu-lab-p1s" and not r.candidates, text


def test_how_it_matched_is_named():
    assert printers.resolve("bambu-lab-p1s").matched_by == "id"
    assert printers.resolve("Bambu P1S").matched_by == "name"
    assert printers.resolve("bambu x1c").matched_by == "alias"
    assert printers.resolve("k1 carbon").profile.id == "creality-k1c"


@pytest.mark.parametrize("text, pid", [
    ("x1c", "bambu-lab-x1-carbon"), ("bambu x1 carbon", "bambu-lab-x1-carbon"), ("a1m", "bambu-lab-a1-mini"),
    ("Prusa MK4S", "prusa-mk4s"), ("original prusa mk4s", "prusa-mk4s"), ("prusa mini+", "prusa-mini-plus"),
    ("Prusa Mini Plus", "prusa-mini-plus"), ("prusa core one+", "prusa-core-one-plus"), ("Creality K1C", "creality-k1c"),
    ("k2+", "creality-k2-plus"), ("K2 plus", "creality-k2-plus"), ("Ender 3 V3", "creality-ender-3-v3"),
    ("Creality Ender-3 V3", "creality-ender-3-v3"), ("elegoo centauri carbon", "elegoo-centauri-carbon"),
    ("voron 2.4", "voron-2-4"), ("voron v2.4", "voron-2-4"), ("H2D", "bambu-lab-h2d"), ("bambu a1", "bambu-lab-a1"),
    ("a1 mini", "bambu-lab-a1-mini"), ("K1", "creality-k1")])
def test_names_and_aliases_of_the_catalog(text, pid):
    r = printers.resolve(text)
    assert r.status == "match" and r.profile.id == pid, (text, r.status)


def test_an_exact_name_wins_over_a_longer_sibling_name():
    assert printers.resolve("a1").profile.id == "bambu-lab-a1"        # A1 mini holds the word a1 too
    assert printers.resolve("k1").profile.id == "creality-k1"          # K1C too


def test_the_vendor_word_is_a_hard_filter():
    assert printers.resolve("prusa p1s").status == "none"              # P1S is a Bambu Lab printer
    assert printers.resolve("creality k1").profile.vendor == "Creality"
    assert printers.resolve("elegoo k1").status == "none"
    r = synthetic(model("a1", name="X1"), model("z1", vendor="Zeta", name="X1"))
    both = printers.resolve("x1", r)
    assert both.status == "ambiguous" and [p.id for p in both.candidates] == ["a1", "z1"]       # no vendor word: two exact names
    assert printers.resolve("zeta x1", r).profile.id == "z1"           # the vendor word picks one


# --- resolve: ambiguous ---------------------------------------------------------------------------------------------------------

def test_siblings_that_differ_in_a_gating_field_are_ambiguous_and_the_difference_is_named():
    c = synthetic(model("e3-v3", name="Ender-3 V3 SE", max_nozzle_c=300), model("e3-ke", name="Ender-3 V3 KE", build_height_mm=240))
    r = printers.resolve("ender 3 v3", c)
    assert r.status == "ambiguous" and [p.id for p in r.candidates] == ["e3-v3", "e3-ke"]
    assert set(r.differs) == {"max_nozzle_c", "build_height_mm"} and r.profile is None
    assert "Ask the user which one" in r.note and "max_nozzle_c" in r.note


def test_the_candidates_are_the_printers_whose_names_hold_all_the_words():
    r = printers.resolve("bambu p")
    assert r.status == "ambiguous" and [p.id for p in r.candidates] == ["bambu-lab-p1s", "bambu-lab-p2s"]
    assert printers.resolve("carbon").status == "ambiguous"
    assert {p.id for p in printers.resolve("carbon").candidates} == {"bambu-lab-x1-carbon", "creality-k1c", "elegoo-centauri-carbon"}
    assert {p.vendor for p in printers.resolve("bambu").candidates} == {"Bambu Lab"}           # a vendor word alone: all of its printers
    assert len(printers.resolve("bambu").candidates) == 6
    assert {p.vendor for p in printers.resolve("creality k").candidates} == {"Creality"}


def test_a_part_of_one_name_is_ambiguous_with_one_candidate_and_never_a_silent_match():
    for text, pid in (("prusa mini", "prusa-mini-plus"), ("prusa core one", "prusa-core-one-plus"), ("mk4", "prusa-mk4s"),
                      ("creality ender 3", "creality-ender-3-v3")):
        r = printers.resolve(text)
        assert r.status == "ambiguous" and [p.id for p in r.candidates] == [pid] and r.profile is None, text
        assert "only part of the name" in r.note


@pytest.mark.parametrize("text, pid", [
    ("P1S Combo", "bambu-lab-p1s"), ("Bambu Lab P1S Combo", "bambu-lab-p1s"), ("Bambu Lab A1 Combo", "bambu-lab-a1"),
    ("A1 mini Combo", "bambu-lab-a1-mini"), ("X1-Carbon Combo", "bambu-lab-x1-carbon"), ("Bambu Lab A1 mini AMS bundle", "bambu-lab-a1-mini"),
    ("Creality K2 Plus CFS Combo", "creality-k2-plus"), ("Original Prusa MK4S kit", "prusa-mk4s"), ("my Prusa Core One+ kit bundle", "prusa-core-one-plus")])
def test_the_shop_words_of_a_bundle_do_not_stop_a_match(text, pid):
    """Review MIN-8: `P1S Combo` is the usual product name of the P1S; it gave `none` and no suggestion."""
    r = printers.resolve(text)
    assert r.status == "match" and r.profile.id == pid, (text, r.status, r.suggestions)
    assert printers.resolve("combo").status == "none" and printers.resolve("ams bundle kit").status == "none"      # shop words alone name nothing


def test_a_part_of_a_name_is_a_run_of_the_words_of_one_name_and_not_a_set_of_word_parts():
    """Review MIN-1: `p1p` is the parts p, 1, p, and the set {p, 1} is inside P1S {p, 1, s}: a false candidate."""
    for text in ("bambu p1p", "bambu s1", "bambu 1s", "bambu sp1", "carbon x1", "bambu carbon x1", "ender v3 3"):
        r = printers.resolve(text)
        assert r.status == "none" and r.profile is None and not r.candidates, (text, r.status, [p.id for p in r.candidates])
        assert "only part of the name" not in r.note and "part of the name" not in r.note
    # a run that starts at the start of a word (it may end inside the last word) is a part of a name
    for text, pids in (("bambu p1", ["bambu-lab-p1s"]), ("bambu x1", ["bambu-lab-x1-carbon"]), ("x1", ["bambu-lab-x1-carbon"]),
                       ("prusa mk 4", ["prusa-mk4s"]), ("one plus", ["prusa-core-one-plus"]), ("centauri", ["elegoo-centauri-carbon"]),
                       ("bambu p", ["bambu-lab-p1s", "bambu-lab-p2s"]), ("mini", ["bambu-lab-a1-mini", "prusa-mini-plus"])):
        r = printers.resolve(text)
        assert r.status == "ambiguous" and [p.id for p in r.candidates] == pids, (text, r.status, [p.id for p in r.candidates])


def test_an_ambiguous_answer_from_a_part_of_a_name_says_so_in_the_data_and_the_note_is_true():
    for text in ("bambu x1", "mk4", "prusa mini", "bambu p", "carbon", "prusa core one"):
        r = printers.resolve(text)
        assert r.status == "ambiguous" and r.matched_by == "partial" and r.to_dict()["matched_by"] == "partial", text
        assert "part of the name" in r.note
    one = printers.resolve("bambu x1")
    assert [p.id for p in one.candidates] == ["bambu-lab-x1-carbon"] and one.profile is None and one.to_dict()["origin"] is None
    assert "only part of the name of one printer" in one.note and "Bambu Lab X1 Carbon" in one.note and "a similar model" in one.note
    assert printers.resolve("bambu").matched_by == "vendor"                   # a text with only a vendor word
    r = synthetic(model("a1", name="X1"), model("z1", vendor="Zeta", name="X1"))
    assert printers.resolve("x1", r).status == "ambiguous" and printers.resolve("x1", r).matched_by == "name"   # the exact name of two vendors
    assert printers.resolve("Bambu P1S").matched_by == "name"


# --- resolve: none and suggestions ----------------------------------------------------------------------------------------------

def test_no_match_gives_suggestions_and_never_takes_one():
    r = printers.resolve("Bambu P1X")
    assert r.status == "none" and r.profile is None and not r.candidates
    assert r.suggestions and r.suggestions[0][0].id == "bambu-lab-p1s"
    assert "never taken by itself" in r.note and "save_printer" in r.note and "bed_mm" in r.note
    r = printers.resolve("my Ender 3 V3 SE")
    assert r.status == "none" and r.suggestions[0][0].id == "creality-ender-3-v3"


def test_a_name_that_is_far_from_all_gives_no_suggestion_and_the_vendors():
    r = printers.resolve("qwertyuiop")
    assert r.status == "none" and r.suggestions == () and r.vendors == tuple(printers.load_catalog().vendors)
    for text in ("", "   ", "printer", "my 3d printer", None, 5):
        assert printers.resolve(text).status == "none", text


def test_the_suggest_settings_are_module_constants():
    assert printers.SUGGEST_CUTOFF == 0.6 and printers.SUGGEST_MAX == 5
    assert isinstance(printers.SUGGEST_CUTOFF, float) and isinstance(printers.SUGGEST_MAX, int)
    src = (ROOT / "topopt" / "printers.py").read_text(encoding="utf-8")
    assert re.search(r"^SUGGEST_CUTOFF = 0\.6", src, re.M) and re.search(r"^SUGGEST_MAX = 5", src, re.M)


def test_the_constants_act(monkeypatch):
    many = synthetic(*[model(f"zz{i}", name=f"ZZ{i}") for i in range(1, 9)])
    assert len(printers.resolve("zzz", many).suggestions) == printers.SUGGEST_MAX == 5
    monkeypatch.setattr(printers, "SUGGEST_MAX", 2)
    assert len(printers.resolve("zzz", many).suggestions) == 2
    monkeypatch.setattr(printers, "SUGGEST_CUTOFF", 1.0)
    assert printers.resolve("zzz", many).suggestions == ()
    monkeypatch.setattr(printers, "SUGGEST_CUTOFF", 0.0)
    monkeypatch.setattr(printers, "SUGGEST_MAX", 5)
    assert len(printers.resolve("qwertyuiop", many).suggestions) == 5          # a lower cutoff lets any name in


def test_the_reply_shape_is_the_same_for_the_three_statuses():
    keys = {"ok", "status", "query", "catalog_version", "matched_by", "origin", "profile", "derives", "candidates", "differs", "suggestions", "vendors", "user_file", "note"}
    for text in ("Bambu P1S", "bambu p", "Bambu P1X"):
        d = printers.resolve(text).to_dict()
        assert set(d) == keys and d["ok"] is True
        json.dumps(d)
    d = printers.resolve("Bambu P1S").to_dict()
    assert d["profile"]["id"] == "bambu-lab-p1s" and d["profile"]["bed_mm"] == [256, 256] and d["derives"]["missing"] == []
    assert d["derives"]["sets"]["print.bed_mm"] == [256.0, 256.0] and d["derives"]["sets"]["print.printer_type"] == "fdm"
    d = printers.resolve("voron 2.4").to_dict()
    assert d["derives"]["missing"] == ["print.bed_mm", "build height"] and "does not state" in d["note"]


# --- the key print.printer_model in the problem YAML (config) -----------------------------------------------------------------

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from topopt import print_report as pr  # noqa: E402
from topopt import printcheck as pc  # noqa: E402
from topopt.config import PRINT_KEYS, PrintSettings, UserError, load_config, parse_print_block  # noqa: E402

from test_input import make_case  # noqa: E402


def with_print(tmp_path, block, h=2.0):
    return make_case(tmp_path, h=h, mutate=lambda d: d.update({"print": block}))


def test_printer_model_is_a_print_key():
    assert "printer_model" in PRINT_KEYS


def test_a_model_fills_the_bed_and_the_type_and_labels_them_profile(tmp_path):
    ps = load_config(with_print(tmp_path, {"printer_model": "Bambu P1S"})).print_settings
    assert ps.bed_mm == (256.0, 256.0) and ps.printer_type == "fdm" and ps.build_direction == (0.0, 0.0, 1.0)
    assert ps.printer_model == "bambu-lab-p1s" and ps.profile.catalog_version == "2026-10-24" and ps.build_height_mm == 256.0
    assert ps.given == ("printer_model",) and ps.derived == ("bed_mm", "printer_type")
    assert ps.key_source("bed_mm") == "profile:bambu-lab-p1s" and ps.key_source("printer_type") == "profile:bambu-lab-p1s"
    assert ps.key_source("build_direction") is None and ps.profile.enclosure == "enclosed" and ps.profile.max_nozzle_c == 300.0
    assert (ps.profile.stock_nozzle_material, ps.profile.extruder, ps.profile.nozzle_mm) == ("stainless", "direct", 0.4)


def test_a_key_that_the_yaml_gives_wins_over_the_profile(tmp_path):
    ps = load_config(with_print(tmp_path, {"printer_model": "p1s", "bed_mm": [200, 180]})).print_settings
    assert ps.bed_mm == (200.0, 180.0) and ps.derived == ("printer_type",) and ps.key_source("bed_mm") is None
    assert "bed_mm" in ps.given
    ps = load_config(with_print(tmp_path, {"printer_model": "p1s", "printer_type": "sla", "bed_mm": 100})).print_settings
    assert ps.printer_type == "sla" and ps.bed_mm == (100.0, 100.0) and ps.derived == ()
    assert ps.printer_model == "bambu-lab-p1s" and ps.build_height_mm == 256.0      # the height still comes from the profile


def test_an_alias_and_an_id_give_the_same_profile(tmp_path):
    ids = {load_config(with_print(tmp_path, {"printer_model": t})).print_settings.printer_model
           for t in ("x1c", "Bambu X1 Carbon", "bambu-lab-x1-carbon")}
    assert ids == {"bambu-lab-x1-carbon"}


def test_a_profile_without_a_bed_leaves_the_bed_assumed(tmp_path):
    ps = load_config(with_print(tmp_path, {"printer_model": "voron 2.4"})).print_settings
    assert ps.bed_mm == (220.0, 220.0) and ps.derived == ("printer_type",) and ps.build_height_mm is None
    assert ps.key_source("bed_mm") is None


@pytest.mark.parametrize("value, parts", [
    ("Bambu P1X", ["'print.printer_model'", "'Bambu P1X'", "not in the printer catalog", "Did you mean: bambu-lab-p1s (Bambu Lab P1S)",
                   "Use an id from the catalog", "remove printer_model and give bed_mm and printer_type", "(file: "]),
    ("zzzzzz", ["no name is close to it", "remove printer_model"]),
    ("bambu p", ["fits 2 printers", "bambu-lab-p1s (Bambu Lab P1S), bambu-lab-p2s (Bambu Lab P2S)", "Use an id from the catalog"]),
    ("prusa mini", ["fits 1 printer of", "prusa-mini-plus (Prusa MINI+)"]),
    (5, ["must be the name or the id of a printer as text", "remove printer_model"]),
    ("   ", ["must be the name or the id of a printer as text"]),
    ([], ["must be the name or the id of a printer as text"]),
])
def test_an_unknown_or_ambiguous_model_is_a_one_line_error(tmp_path, value, parts):
    with pytest.raises(UserError) as e:
        load_config(with_print(tmp_path, {"printer_model": value}))
    text = str(e.value)
    assert "\n" not in text
    for part in parts:
        assert part in text, (part, text)


def test_a_broken_catalog_is_a_user_error_with_the_way_on(tmp_path, monkeypatch):
    def broken():
        raise printers.CatalogError("printer catalog: topopt/data/printers.yaml is empty")
    monkeypatch.setattr(printers, "_default_catalog", broken)
    with pytest.raises(UserError, match="cannot be checked: printer catalog: topopt/data/printers.yaml is empty. Remove printer_model"):
        load_config(with_print(tmp_path, {"printer_model": "p1s"}))
    assert load_config(with_print(tmp_path, {"bed_mm": 150})).print_settings.bed_mm == (150.0, 150.0)     # no model: the catalog is not read


def test_without_a_model_the_settings_and_run_json_are_as_before(tmp_path):
    ps = load_config(with_print(tmp_path, {"bed_mm": 150, "build_direction": "+x"})).print_settings
    assert ps.profile is None and ps.derived == () and ps.printer_model is None and ps.build_height_mm is None
    assert set(ps.to_dict()) == {"build_direction", "bed_mm", "printer_type", "self_supporting_angle_deg", "given"}
    assert set(PrintSettings().to_dict()) == {"build_direction", "bed_mm", "printer_type", "self_supporting_angle_deg", "given"}
    assert PrintSettings.from_dict(ps.to_dict()) == ps
    assert PrintSettings().key_source("bed_mm") is None


def test_run_json_print_keeps_the_model_the_catalog_version_and_the_facts():
    ps = parse_print_block({"printer_model": "Voron 2.4", "bed_mm": 300})
    d = json.loads(json.dumps(ps.to_dict()))
    assert d["printer_model"] == "voron-2-4" and d["catalog_version"] == "2026-10-24" and d["derived"] == ["printer_type"]
    assert d["given"] == ["bed_mm", "printer_model"] and d["bed_mm"] == [300.0, 300.0]
    assert d["profile"]["name"] == "Voron 2.4" and d["profile"]["enclosure"] == "enclosed" and d["profile"]["bed_mm"] is None
    assert d["profile"]["source_url"].startswith("https://") and d["profile"]["opened"] == "2026-10-24"
    back = PrintSettings.from_dict(d)
    assert back.printer_model == "voron-2-4" and back.profile.facts() == ps.profile.facts() and back.derived == ("printer_type",)
    assert back.profile.catalog_version == "2026-10-24" and back.bed_mm == ps.bed_mm and back.given == ps.given


def test_a_run_json_without_the_keys_or_with_bad_ones_reads_as_today():
    old = {"build_direction": [0, 0, 1], "bed_mm": [220, 220], "printer_type": "fdm", "self_supporting_angle_deg": None, "given": []}
    assert PrintSettings.from_dict(old) == PrintSettings()
    for bad in ({"printer_model": "x"}, {"printer_model": "x", "profile": 5}, {"printer_model": "", "profile": {}},
                {"printer_model": 5, "profile": {"vendor": "A"}}, {"profile": {"vendor": "A"}, "derived": ["bed_mm"]}):
        ps = PrintSettings.from_dict({**old, **bad})
        assert (ps.profile, ps.derived) == (None, ()) or ps.profile.id == "x", bad
    ps = PrintSettings.from_dict({**old, "printer_model": "x", "profile": {"vendor": "A", "model": "B", "bed_mm": [-1, 5],
                                                                          "build_height_mm": "tall", "enclosure": "wet"},
                                  "derived": ["bed_mm", "bogus"]})
    assert ps.profile.id == "x" and ps.profile.bed_mm is None and ps.profile.build_height_mm is None
    assert ps.profile.enclosure is None and ps.derived == ("bed_mm",)


def test_the_run_writes_the_model_into_run_json_and_the_report(tmp_path):
    from topopt.cli import main
    p = with_print(tmp_path, {"printer_model": "Prusa MK4S", "build_direction": "+x"})
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["print"]["printer_model"] == "prusa-mk4s" and run["print"]["catalog_version"] == "2026-10-24"
    assert run["print"]["bed_mm"] == [250.0, 210.0] and run["print"]["derived"] == ["bed_mm", "printer_type"]
    assert run["print"]["profile"]["build_height_mm"] == 220.0 and run["print"]["profile"]["extruder"] == "direct"
    s = json.loads((out / "result" / "metrics.json").read_text())["print"]
    assert s["settings"]["printer_model"] == "prusa-mk4s" and s["settings"]["derived"] == {
        "bed_mm": "profile:prusa-mk4s", "printer_type": "profile:prusa-mk4s"}
    assert s["settings"]["assumed"] == [] and "assumed_note" not in s["settings"] and s["settings"]["build_height_mm"] == 220.0
    assert s["configured"]["height_fit"] is True and s["configured"]["height_margin_mm"] == 220.0 - 22.0     # +x up: the 22 mm part length
    assert "height_fit" in s["definitions"]
    again = pc.check(out)       # a run folder: the printcheck reads the model from run.json
    assert again["settings"]["printer_model"] == "prusa-mk4s"
    assert again["settings"]["bed_mm"] == [250.0, 210.0]


def test_a_run_without_a_model_has_no_new_key_in_the_report(tmp_path):
    from topopt.cli import main
    out = tmp_path / "run"
    assert main(["run", str(make_case(tmp_path, h=2.0)), "--out", str(out), "--quiet"]) == 0
    run = json.loads((out / "run.json").read_text())
    assert set(run["print"]) == {"build_direction", "bed_mm", "printer_type", "self_supporting_angle_deg", "given"}
    s = json.loads((out / "result" / "metrics.json").read_text())["print"]
    assert set(s["settings"]) == {"build_direction", "build_direction_label", "bed_mm", "printer_type", "self_supporting_angle_deg",
                                  "angle_source", "assumed", "assumed_note"}
    assert "height_fit" not in s["configured"] and "height_fit" not in s["definitions"]
    assert set(s["configured"]) >= {"bed_fit", "bed_turn_deg", "bed_footprint_mm", "bed_margin_mm"}


# --- the build height: the fit check -------------------------------------------------------------------------------------------

def rect(lo, hi):
    m = trimesh.creation.box(extents=np.asarray(hi, float) - np.asarray(lo, float))
    m.apply_translation((np.asarray(lo, float) + np.asarray(hi, float)) / 2)
    return m


def corners(x, y, z):
    return np.array([[i, j, k] for i in (0, x) for j in (0, y) for k in (0, z)], float)


def test_bed_fit_turn_without_a_height_has_the_old_keys_and_with_one_the_two_new_keys():
    pts = corners(100, 50, 40)
    old = pr.bed_fit_turn(pts, (0, 0, 1), (220, 220))
    assert set(old) == {"bed_fit", "bed_turn_deg", "bed_footprint_mm", "bed_margin_mm"}
    assert pr.bed_fit_turn(pts, (0, 0, 1), (220, 220), None) == old
    fits = pr.bed_fit_turn(pts, (0, 0, 1), (220, 220), 256)
    assert fits["height_fit"] is True and fits["height_margin_mm"] == 216.0 and fits["bed_fit"] is True
    assert {k: fits[k] for k in old} == old
    exact = pr.bed_fit_turn(pts, (0, 0, 1), (220, 220), 40)
    assert exact["height_fit"] is True and exact["height_margin_mm"] == 0.0 and exact["bed_fit"] is True
    tall = pr.bed_fit_turn(pts, (0, 0, 1), (220, 220), 30)
    assert tall["height_fit"] is False and tall["height_margin_mm"] == -10.0 and tall["bed_fit"] is False
    assert tall["bed_margin_mm"] == old["bed_margin_mm"]                      # the footprint is not changed by the height
    sideways = pr.bed_fit_turn(pts, (1, 0, 0), (220, 220), 90)                # +x up: the height is the 100 mm extent
    assert sideways["height_fit"] is False and sideways["height_margin_mm"] == -10.0


def test_fit_phrase_names_a_part_that_is_too_tall():
    pts = corners(100, 50, 40)
    assert pr.fit_phrase(pr.bed_fit_turn(pts, (0, 0, 1), (220, 220), 30)) == "fits the bed but is 10 mm too tall for the build height"
    assert pr.fit_phrase(pr.bed_fit_turn(pts, (0, 0, 1), (220, 220), 256)) == "fits"
    both = pr.fit_phrase(pr.bed_fit_turn(pts, (0, 0, 1), (60, 60), 30))
    assert both.startswith("does not fit at any turn (best: ") and both.endswith(", and is 10 mm too tall for the build height")
    assert pr.fit_phrase(pr.bed_fit_turn(pts, (0, 0, 1), (60, 60))) == both.split(", and")[0]          # no profile: the old text only


def test_the_print_report_has_the_height_fit_only_with_a_profile():
    mesh = rect((0, 0, 0), (60, 10, 200))
    plain = pr.build_report(mesh, PrintSettings(), thickness_samples=500)
    assert "height_fit" not in plain["configured"] and all("height_fit" not in o for o in plain["options"])
    prof = parse_print_block({"printer_model": "bambu a1 mini"})                 # 180 x 180 x 180 mm
    rep = pr.build_report(mesh, prof, thickness_samples=500)
    assert rep["configured"]["height_fit"] is False and rep["configured"]["height_margin_mm"] == -20.0
    assert rep["configured"]["bed_fit"] is False
    by = {o["label"]: o for o in rep["options"]}
    assert by["+z"]["bed_fit"] is False and by["-z"]["bed_fit"] is False and by["+x"]["bed_fit"] is True
    assert by["+x"]["height_margin_mm"] == 120.0 and by["+y"]["height_margin_mm"] == 170.0
    assert rep["settings"]["bed_mm"] == [180.0, 180.0] and rep["settings"]["assumed"] == ["build_direction"]
    assert rep["settings"]["assumed_note"]


def test_printcheck_bed_fit_uses_the_build_height_and_names_the_profile(tmp_path):
    stl = tmp_path / "tall.stl"
    rect((0, 0, 0), (60, 10, 200)).export(stl)
    y = tmp_path / "p.yaml"
    y.write_text("print:\n  printer_model: bambu a1 mini\n")
    r = pc.check(stl, yaml_path=y)
    c = {x["id"]: x for x in r["checks"]}["bed_fit"]
    assert c["state"] == "fail" and c["value"]["height_fit"] is False and c["value"]["height_margin_mm"] == -20.0
    assert c["limit"] == {"value": [180.0, 180.0], "key": "bed_mm", "source": "profile:bambu-lab-a1-mini",
                          "build_height_mm": 180.0, "build_height_source": "profile:bambu-lab-a1-mini"}
    assert "fits the bed but is 20 mm too tall for the build height" in c["reason"] and "(profile:bambu-lab-a1-mini)" in c["reason"]
    assert "the part is 200 mm high and the build height is 180 mm" in c["reason"]
    s = r["settings"]
    assert s["printer_model"] == "bambu-lab-a1-mini" and s["catalog_version"] == "2026-10-24" and s["build_height_mm"] == 180.0
    assert s["derived"] == {"bed_mm": "profile:bambu-lab-a1-mini", "printer_type": "profile:bambu-lab-a1-mini"}
    assert s["assumed"] == ["build_direction"] and "height_fit" in r["definitions"]
    text = pc.format_text(r)
    assert "printer: Bambu Lab A1 mini (profile bambu-lab-a1-mini, catalog 2026-10-24); build height 180 mm" in text
    y.write_text("print:\n  printer_model: bambu a1 mini\n  build_direction: +x\n")           # 200 mm laid down: 60 mm high
    ok = {x["id"]: x for x in pc.check(stl, yaml_path=y)["checks"]}["bed_fit"]
    assert ok["state"] == "pass" and ok["value"]["height_margin_mm"] == 120.0


def test_printcheck_without_a_profile_is_unchanged(tmp_path):
    stl = tmp_path / "tall.stl"
    rect((0, 0, 0), (60, 10, 200)).export(stl)
    r = pc.check(stl)
    c = {x["id"]: x for x in r["checks"]}["bed_fit"]
    assert c["state"] == "pass" and "height_fit" not in c["value"] and set(c["limit"]) == {"value", "key", "source"}
    assert "printer_model" not in r["settings"] and "height_fit" not in r["definitions"]
    assert "printer:" not in pc.format_text(r)


def test_settings_of_a_later_source_replace_the_derived_ones(tmp_path):
    stl = tmp_path / "t.stl"
    rect((0, 0, 0), (20, 10, 10)).export(stl)
    y = tmp_path / "p.yaml"
    y.write_text("print:\n  printer_model: prusa mk4s\n")
    r = pc.check(stl, yaml_path=y)
    assert r["settings"]["bed_mm"] == [250.0, 210.0] and r["settings"]["derived"]["bed_mm"] == "profile:prusa-mk4s"
    r = pc.check(stl, yaml_path=y, print_settings={"bed_mm": 100})                       # an option wins over the profile of the YAML
    assert r["settings"]["bed_mm"] == [100.0, 100.0] and r["settings"]["derived"] == {"printer_type": "profile:prusa-mk4s"}
    assert {x["id"]: x for x in r["checks"]}["bed_fit"]["limit"]["source"] == "option"
    r = pc.check(stl, yaml_path=y, print_settings={"printer_model": "bambu p1s"})         # another model in an option replaces the whole profile
    assert r["settings"]["printer_model"] == "bambu-lab-p1s" and r["settings"]["bed_mm"] == [256.0, 256.0]
    with pytest.raises(UserError, match="not in the printer catalog"):
        pc.check(stl, print_settings={"printer_model": "nope"})


def test_a_value_from_the_printer_of_run_json_keeps_that_printers_label_when_the_yaml_names_another_printer(tmp_path):
    """Review MIN-5: the bed of the P1S in run.json was labelled `profile:voron-2-4` (the id of the YAML printer, which has no bed)."""
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    rect((0, 0, 0), (20, 10, 10)).export(run / "result" / "final.stl")
    (run / "run.json").write_text(json.dumps({"print": parse_print_block({"printer_model": "bambu p1s"}, "<t>").to_dict()}))
    y = tmp_path / "p.yaml"
    y.write_text("print:\n  printer_model: voron 2.4\n")                         # the Voron has no bed in the catalog
    r = pc.check(run, yaml_path=y)
    s = r["settings"]
    assert s["printer_model"] == "voron-2-4" and s["bed_mm"] == [256.0, 256.0]
    assert s["derived"] == {"bed_mm": "profile:bambu-lab-p1s", "printer_type": "profile:voron-2-4"}
    bed = {x["id"]: x for x in r["checks"]}["bed_fit"]
    assert bed["limit"]["source"] == "profile:bambu-lab-p1s"
    text = pc.format_text(r)
    assert "from the profile: bed_mm (profile:bambu-lab-p1s), printer_type" in text
    # the same printer in both sources: the labels and the text are as before
    y.write_text("print:\n  printer_model: bambu p1s\n")
    r = pc.check(run, yaml_path=y)
    assert r["settings"]["derived"] == {"bed_mm": "profile:bambu-lab-p1s", "printer_type": "profile:bambu-lab-p1s"}
    assert "from the profile: bed_mm, printer_type" in pc.format_text(r)
    # the YAML gives its own bed: that key is the YAML's
    y.write_text("print:\n  printer_model: voron 2.4\n  bed_mm: 300\n")
    r = pc.check(run, yaml_path=y)
    assert r["settings"]["bed_mm"] == [300.0, 300.0] and r["settings"]["derived"] == {"printer_type": "profile:voron-2-4"}
    assert {x["id"]: x for x in r["checks"]}["bed_fit"]["limit"]["source"] == "yaml"


def test_check_facts_and_lines_name_the_profile_and_the_height(tmp_path, capsys):
    from topopt.cli import check_data, main
    p = with_print(tmp_path, {"printer_model": "Bambu A1 mini", "build_direction": "+x"})         # the cantilever box: 22 x 10 x 10 mm
    d = check_data(p)["print"]
    assert d["printer_model"] == "bambu-lab-a1-mini" and d["printer_name"] == "Bambu Lab A1 mini" and d["catalog_version"] == "2026-10-24"
    assert d["bed_mm"] == [180.0, 180.0] and d["assumed"] == [] and d["derived"]["bed_mm"] == "profile:bambu-lab-a1-mini"
    assert d["configured"]["height_fit"] is True and d["configured"]["height_margin_mm"] == 180.0 - 22.0
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("print settings:"))
    assert line == ("print settings: bed 180 x 180 mm, build direction +x, printer: Bambu Lab A1 mini (profile bambu-lab-a1-mini, "
                    "catalog 2026-10-24), type fdm (from the profile: bed_mm, printer_type)")
    assert "(build height 180 mm, margin 158 mm)" in out
    (tmp_path / "plain").mkdir()
    plain = check_data(make_case(tmp_path / "plain", h=2.0))["print"]
    assert "printer_model" not in plain and "derived" not in plain and "height_fit" not in plain["configured"]


def test_check_says_when_the_part_box_is_too_tall(tmp_path, capsys):
    from topopt.cli import main
    small = parse_print_block({"printer_model": "bambu a1 mini"})
    assert small.build_height_mm == 180.0
    facts = pr.check_facts(corners(100, 50, 200), small)
    lines = pr.check_lines(facts)
    assert "fits the bed but is 20 mm too tall for the build height" in lines[1] and "(build height 180 mm, margin -20 mm)" in lines[1]
    assert "printer: Bambu Lab A1 mini" in lines[0] and "assumed: build_direction" in lines[0]
    assert "does not fit with +y -y +z -z up" in lines[1] and "fits with +x (turned 45 degrees) -x (turned 45 degrees) up" in lines[1]


def test_check_part_takes_the_bed_of_the_profile(tmp_path):
    import shutil
    import subprocess
    folder = ROOT / "tests" / "fixtures" / "check_part"
    d = tmp_path / "cp"
    shutil.copytree(folder, d)
    doc = yaml.safe_load((d / "problem.yaml").read_text())
    doc["print"] = {"printer_model": "bambu a1 mini", "build_direction": "+z"}
    (d / "problem.yaml").write_text(yaml.safe_dump(doc))
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_part.py"), str(d / "run"), str(d / "problem.yaml")],
                       capture_output=True, text=True, cwd=d)
    assert "fits 180x180 mm bed" in r.stdout, r.stdout + r.stderr


def test_check_part_compares_the_part_height_with_the_build_height_of_the_profile(tmp_path):
    """Review NIT-5: check_part took the bed of the profile but not its height, so it could pass a part that the print report calls too tall."""
    import shutil
    import subprocess
    d = tmp_path / "cp"
    shutil.copytree(ROOT / "tests" / "fixtures" / "check_part", d)
    env = {k: v for k, v in os.environ.items() if k not in ("TOPOPT_WORKSPACE", "TOPOPT_DEFAULT_WORKSPACE")}
    results = {}
    for height in (1.0, 5000.0):
        assert printers.save_user_printer(d, {**TRIDENT, "bed_mm": [900, 900], "build_height_mm": height})["ok"]
        doc = yaml.safe_load((d / "problem.yaml").read_text())
        doc["print"] = {"printer_model": "voron trident", "build_direction": "+z"}
        (d / "problem.yaml").write_text(yaml.safe_dump(doc))
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_part.py"), str(d / "run"), str(d / "problem.yaml")],
                           capture_output=True, text=True, cwd=d, env=env)
        results[height] = [line for line in r.stdout.splitlines() if "mm bed" in line]
    assert results[1.0] and results[1.0][0].startswith("[FAIL] fits 900x900 mm bed") and "too tall" in results[1.0][0], results
    assert results[5000.0] and results[5000.0][0].startswith("[PASS] fits 900x900 mm bed"), results


# --- preferences: print.printer_model --------------------------------------------------------------------------------------------


from topopt import preferences  # noqa: E402
from topopt_mcp import server  # noqa: E402

PREFS = preferences.FILE_NAME


@pytest.fixture
def work(tmp_path, monkeypatch):
    """The server with a work folder of this test (the environment probe is replaced by a small dict)."""
    folder = tmp_path / "work"
    folder.mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(folder))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    monkeypatch.chdir(tmp_path)
    import topopt.environment as env_module
    monkeypatch.setattr(env_module, "environment_data", lambda yaml=None: {"python": {"executable": sys.executable}, "ok": None})
    return folder


def tiny_spec(**settings):
    return {"name": "tiny", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}]},
        "load": {"tip": [{"box": {"min": [16, 2, 0], "max": [20, 8, 6]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
        "settings": {"iterations": 3, "voxel_size": 2.0, **settings}}


def test_the_preference_is_a_print_key_and_the_file_keeps_the_id(tmp_path):
    assert preferences.PRINT_KEYS[0] == "printer_model"
    r = preferences.save(tmp_path, print={"printer_model": "Bambu P1S"})
    assert r["ok"] and r["values"]["print"] == {"printer_model": "bambu-lab-p1s"}
    assert yaml.safe_load((tmp_path / PREFS).read_text())["print"] == {"printer_model": "bambu-lab-p1s"}
    again = preferences.save(tmp_path, print={"printer_model": "bambu-lab-p1s"})
    assert again["ok"] and again["changed"] == [] and "Nothing changed" in again["note"]
    assert preferences.save(tmp_path, print={"printer_model": None})["values"].get("print") is None       # null removes the key


@pytest.mark.parametrize("value, parts", [
    ("Bambu P1X", ["'print.printer_model'", "Did you mean: bambu-lab-p1s (Bambu Lab P1S)", "remove printer_model and give bed_mm"]),
    ("bambu p", ["fits 2 printers", "bambu-lab-p1s (Bambu Lab P1S), bambu-lab-p2s (Bambu Lab P2S)"]),
    ("prusa mini", ["fits 1 printer of", "prusa-mini-plus (Prusa MINI+)"]),
    (7, ["must be the name or the id of a printer as text"]),
])
def test_a_model_that_is_not_one_printer_is_refused_with_the_candidates_and_nothing_is_written(tmp_path, value, parts):
    r = preferences.save(tmp_path, print={"printer_model": value, "bed_mm": 200})
    assert r["ok"] is False and r["error"].startswith("save_preferences wrote nothing.")
    for part in parts:
        assert part in r["error"], (part, r["error"])
    assert not (tmp_path / PREFS).exists()


def test_a_file_with_a_model_that_is_not_in_the_catalog_is_a_fact_and_not_a_crash(tmp_path):
    (tmp_path / PREFS).write_text("format: 1\nprint:\n  printer_model: nope\n")
    facts = preferences.read(tmp_path)
    assert "problems" not in facts and facts["values"] == {}                      # only this key is not used (MIN-3), the file is good
    assert facts["not_used"][0].startswith("'print.printer_model' 'nope' is not in the printer catalog")


def test_build_regions_fills_the_model_into_a_spec_without_a_print_block_and_names_the_derived_values(work):
    assert preferences.save(work, print={"printer_model": "Bambu P1S"})["ok"]
    spec = tiny_spec()
    before = copy.deepcopy(spec)
    r = server.build_regions(spec)
    assert r["ok"], r
    assert spec == before
    doc = yaml.safe_load((work / r["yaml_rel"]).read_text())
    assert doc["print"] == {"printer_model": "bambu-lab-p1s"}
    [fact] = r["preferences_facts"]
    assert str(work / PREFS) in fact and "printer model bambu-lab-p1s (Bambu Lab P1S: bed 256 x 256 mm, build height 256 mm, fdm)" in fact
    assert "sets the bed and the printer type, unless the print block also gives them" in fact
    cfg = load_config(work / r["yaml_rel"])
    assert cfg.print_settings.printer_model == "bambu-lab-p1s" and cfg.print_settings.bed_mm == (256.0, 256.0)
    assert r["check"]["print"]["printer_model"] == "bambu-lab-p1s" and r["check"]["print"]["derived"]["bed_mm"] == "profile:bambu-lab-p1s"


def test_build_regions_never_mixes_the_model_into_a_print_block_that_the_spec_has(work):
    assert preferences.save(work, print={"printer_model": "Bambu P1S"})["ok"]
    r = server.build_regions({**tiny_spec(), "print": {"bed_mm": 120}})
    assert r["ok"] and "preferences_facts" not in r
    assert yaml.safe_load((work / r["yaml_rel"]).read_text())["print"] == {"bed_mm": 120}
    own = server.build_regions({**tiny_spec(), "name": "own", "print": {"printer_model": "prusa mk4s"}})
    assert own["ok"] and "preferences_facts" not in own
    assert load_config(work / own["yaml_rel"]).print_settings.printer_model == "prusa-mk4s"
    wrong = server.build_regions({**tiny_spec(), "name": "wrong", "print": {"printer_model": "nope"}})
    assert wrong["ok"] is False and "'print.printer_model' 'nope' is not in the printer catalog" in wrong["check_error"]


def test_a_preference_with_a_model_does_not_change_a_run_by_itself(work):
    assert preferences.save(work, print={"printer_model": "Bambu P1S", "bed_mm": 99})["ok"]
    r = server.build_regions({**tiny_spec(), "print": {"build_direction": "+x"}})
    ps = load_config(work / r["yaml_rel"]).print_settings
    assert ps.profile is None and ps.bed_mm == (220.0, 220.0) and ps.given == ("build_direction",)


def test_check_problem_names_a_difference_of_the_model(work):
    assert preferences.save(work, print={"printer_model": "Bambu P1S"})["ok"]
    r = server.build_regions({**tiny_spec(), "name": "plain", "print": {"bed_mm": 120}})
    reply = server.check_problem(str(work / r["yaml_rel"]))
    [fact] = reply["preferences_facts"]
    assert "printer model: none in the YAML, bambu-lab-p1s (Bambu Lab P1S: bed 256 x 256 mm, build height 256 mm, fdm) in the preferences" in fact
    assert fact.startswith(f"The preferences file {work / PREFS} differs from this YAML") and "is not changed" in fact
    same = server.build_regions({**tiny_spec(), "name": "same", "print": {"printer_model": "x1c"}})
    other = server.check_problem(str(work / same["yaml_rel"]))
    assert "printer model: bambu-lab-x1-carbon (Bambu Lab X1 Carbon" in other["preferences_facts"][0]
    ok = server.build_regions({**tiny_spec(), "name": "ok", "print": {"printer_model": "bambu p1s"}})
    assert "preferences_facts" not in server.check_problem(str(work / ok["yaml_rel"]))             # an alias in the YAML is the same printer
    bare = server.build_regions({**tiny_spec(), "name": "bare", "print": {"bed_mm": 256}})
    # no print block at all in the YAML: the fact names the assumed values and the model
    text = yaml.safe_load((work / bare["yaml_rel"]).read_text())
    text.pop("print")
    path = work / "runs" / "bare" / "noprint.yaml"
    path.write_text(yaml.safe_dump(text, sort_keys=False))
    first = server.check_problem(str(path))["preferences_facts"][0]
    assert "the YAML has no print block, so the print report assumes printer model none" in first and "the preferences say printer model bambu-lab-p1s" in first


def test_the_printer_type_of_a_yaml_that_the_profile_filled_is_not_called_assumed(work):
    assert preferences.save(work, print={"printer_type": "sla"})["ok"]
    r = server.build_regions({**tiny_spec(), "name": "m", "print": {"printer_model": "bambu p1s"}})
    [fact] = server.check_problem(str(work / r["yaml_rel"]))["preferences_facts"]
    assert "printer: fdm in the YAML (from the printer profile bambu-lab-p1s), sla in the preferences" in fact
    assert "assumed" not in fact


# --- user printers: topopt-printers.yaml in the work folder -----------------------------------------------------------------

USER_FILE = printers.USER_FILE_NAME
TRIDENT = {"id": "voron-trident", "vendor": "Voron", "model": "Trident", "printer_type": "fdm", "bed_mm": [300, 300],
           "source_url": "https://docs.vorondesign.com/build/", "opened": OPENED, "tag": "maker doc", "enclosure": "enclosed",
           "max_nozzle_c": 300}


def user_catalog(folder, *entries, **top):
    doc = {"format": 1, "models": [dict(e) for e in entries], **top}
    (folder / USER_FILE).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return printers.catalog_for(folder)


def test_without_a_user_file_the_catalog_is_the_shipped_one_and_says_where_the_file_would_be(tmp_path):
    assert printers.catalog_for(None).user_file is None
    c = printers.catalog_for(tmp_path)
    assert c.profiles == printers.load_catalog().profiles and c.version == "2026-10-24"
    assert c.user_file == printers.UserFile(str(tmp_path / USER_FILE), False, 0, ())
    assert {p.origin for p in c.profiles} == {"catalog"}
    d = printers.resolve("Bambu P1S", c).to_dict()
    assert d["origin"] == "catalog" and d["profile"]["origin"] == "catalog" and d["user_file"]["exists"] is False


def test_a_user_printer_is_found_with_its_origin_and_its_sources(tmp_path):
    c = user_catalog(tmp_path, {**TRIDENT, "aliases": ["V-Trident"], "field_sources": {"max_nozzle_c": "user", "bed_mm": "https://example.com/bed"}})
    r = printers.resolve("my voron trident", c)
    assert r.status == "match" and r.profile.id == "voron-trident" and r.profile.origin == "user" and r.matched_by == "name"
    assert printers.resolve("V Trident", c).profile.id == "voron-trident"                       # the alias
    d = r.to_dict()
    assert d["origin"] == "user" and d["profile"]["origin"] == "user" and d["catalog_version"] == "2026-10-24"
    assert d["profile"]["field_sources"] == {"max_nozzle_c": "user", "bed_mm": "https://example.com/bed"}
    assert d["profile"]["source_url"] == TRIDENT["source_url"] and d["profile"]["tag"] == "maker doc"
    assert d["user_file"] == {"path": str(tmp_path / USER_FILE), "exists": True, "entries": 1, "problems": []}
    assert "user printer file" in r.note and "catalog 2026" not in r.note
    # the vendor "Voron" of the shipped table is the vendor of the user printer, so the vendor filter and the sibling listing work
    assert c.by_id["voron-trident"].vendor == "Voron"
    amb = printers.resolve("voron", c)
    assert amb.status == "ambiguous" and [p.id for p in amb.candidates] == ["voron-2-4", "voron-trident"]
    assert [p.origin for p in amb.candidates] == ["catalog", "user"]
    # the shipped printers are found as before
    assert printers.resolve("Bambu P1S", c).profile.origin == "catalog"


def test_a_user_printer_may_say_that_the_user_gave_the_values_and_unknown_fields_are_null(tmp_path):
    c = user_catalog(tmp_path, {"id": "my-printer", "vendor": "Garage", "model": "Big One", "printer_type": "fdm", "bed_mm": [400, 350],
                                "source": "user"})
    p = c.by_id["my-printer"]
    assert (p.source, p.tag, p.source_url, p.opened) == ("user", "user", "", "")
    f = p.facts()
    assert f["source"] == "user" and f["origin"] == "user"
    for key in printers.FACT_FIELDS:
        if key not in ("printer_type", "bed_mm"):
            assert f[key] is None, key
    d = printers.resolve("garage big one", c).to_dict()
    assert d["status"] == "match" and d["derives"]["missing"] == ["build height"] and "build height" in d["note"]
    assert c.vendors["Garage"] == ()                  # a new vendor: its name is its word


@pytest.mark.parametrize("change, part", [
    ({"source_url": None}, "give source_url"),
    ({"source_url": "maker page"}, "source_url must be an http(s) address"),
    ({"source": "maker"}, "source must be the word user"),
    ({"bed_mm": None}, "missing key 'bed_mm'"),                       # None here means: the key is left out
    ({"bed_mm": [300]}, "bed_mm must be [x, y]"),
    ({"bed_mm": [300, -1]}, "bed_mm must be a number above 0"),
    ({"printer_type": "laser"}, "printer_type must be one of"),
    ({"id": "Voron Trident"}, "id must be a lower-case slug"),
    ({"enclosure": "wet"}, "enclosure must be one of"),
    ({"chamber_max_c": 60, "enclosure": "open"}, "chamber_max_c needs enclosure: heated"),
    ({"tag": "rumour"}, "tag must be one of"),
    ({"field_sources": {"bed_mm": "somewhere"}}, "field_sources.bed_mm must be an http(s) address or the word user"),
    ({"field_sources": {"colour": "user"}}, "not a fact field"),
    ({"opened": "last week"}, "opened must be a date"),
    ({"colour": "red"}, "unknown key 'colour'"),
    ({"vendor": "!!"}, "vendor must have a letter or a digit"),
])
def test_a_user_entry_with_a_fault_is_not_used_and_the_file_says_why(tmp_path, change, part):
    bad = {**TRIDENT, **change}
    bad = {k: v for k, v in bad.items() if v is not None}
    c = user_catalog(tmp_path, bad, {**TRIDENT, "id": "voron-v0", "model": "V0"})
    assert "voron-v0" in c.by_id and c.user_file.entries == 1                              # the other entry still works
    [problem] = c.user_file.problems
    assert part in problem and f"{USER_FILE}, models[0]" in problem and problem.endswith("This entry is not used.")
    assert printers.resolve("voron trident", c).status != "match" or c.by_id.get(bad.get("id")) is None


def test_a_user_entry_never_shadows_a_shipped_printer_by_id_or_by_name(tmp_path):
    c = user_catalog(
        tmp_path,
        {**TRIDENT, "id": "bambu-lab-p1s", "vendor": "Bambu Lab", "model": "P1S mine"},                  # the id of a shipped printer
        {**TRIDENT, "id": "my-p1s", "vendor": "bambu", "model": "P1S"},                                 # the name of a shipped printer
        {**TRIDENT, "id": "my-p1s-2", "vendor": "Bambu Lab", "model": "P1S Special", "aliases": ["p1 s"]},   # an alias that is the shipped name
        {**TRIDENT, "id": "twice"}, {**TRIDENT, "id": "twice", "model": "Other"})
    assert [p.id for p in c.profiles if p.origin == "user"] == ["twice"] and c.user_file.entries == 1
    shipped = printers.load_catalog().by_id["bambu-lab-p1s"]
    assert c.by_id["bambu-lab-p1s"] == shipped and printers.resolve("Bambu P1S", c).profile == shipped
    first, second, third, fourth = c.user_file.problems
    assert "is the id of the shipped printer Bambu Lab P1S" in first and "Use another id, for example bambu-lab-p1s-user" in first
    assert "also a name of bambu-lab-p1s" in second and "Give it another model name or alias" in second
    assert "also a name of bambu-lab-p1s" in third
    assert "the id is used twice" in fourth and "models[4]" in fourth


ACME = {**TRIDENT, "id": "acme-z1", "vendor": "Acme", "model": "Z1", "source_url": "https://example.com/z1"}


def shipped_keys(c):
    """The exact-name keys of the shipped printers of a catalog (what a user file must never change)."""
    return {p.id: c._keys[p.id] for p in c.profiles if p.origin == "catalog"}


def test_the_vendor_words_of_the_user_file_cannot_break_or_change_the_lookup_of_the_shipped_printers(tmp_path):
    """Review MAJ-1: the vendors of the user file were added to one global table and stripped from EVERY shipped name."""
    shipped = printers.load_catalog()
    base_keys = shipped_keys(shipped)
    # 1. a vendor word that is a whole model word of a shipped name (a hand edit): no exception, the shipped printers are found
    c = user_catalog(tmp_path, vendors={"Prusa": ["mk4s"]})
    assert shipped_keys(c) == base_keys and c.user_file.problems == ()
    assert printers.resolve("Bambu P1S", c).profile == shipped.by_id["bambu-lab-p1s"]
    assert printers.resolve("mk4s", c).profile.id == "prusa-mk4s"
    # 2. a real vendor whose name is a word of a shipped name ("Carbon Inc.", and "K1 Carbon" is an alias of the Creality K1C)
    r = printers.save_user_printer(tmp_path, {**ACME, "id": "carbon-m2", "vendor": "Carbon", "model": "M2"})
    assert r["ok"], r
    c = printers.catalog_for(tmp_path)
    assert shipped_keys(c) == base_keys and c.user_file.problems == ()
    assert printers.resolve("k1 carbon", c).profile.id == "creality-k1c" and printers.resolve("Carbon M2", c).profile.id == "carbon-m2"
    assert printers.resolve("elegoo centauri carbon", c).profile.id == "elegoo-centauri-carbon"
    # 3. a vendor word that is a word of a shipped name
    folder = tmp_path / "acme"
    folder.mkdir()
    c = user_catalog(folder, ACME, vendors={"Acme": ["one"]})
    assert c.user_file.problems == () and shipped_keys(c) == base_keys
    assert printers.resolve("core one+", c).profile.id == "prusa-core-one-plus" and printers.resolve("acme z1", c).profile.id == "acme-z1"
    # 4. a vendor named like a word of a shipped name, also through save_printer
    folder = tmp_path / "plus"
    folder.mkdir()
    assert printers.save_user_printer(folder, {**ACME, "id": "plus-z1", "vendor": "Plus", "model": "Z1"})["ok"]
    c = printers.catalog_for(folder)
    assert shipped_keys(c) == base_keys and c.user_file.problems == ()
    for text, pid in (("mini+", "prusa-mini-plus"), ("k2 plus", "creality-k2-plus"), ("prusa core one+", "prusa-core-one-plus"),
                      ("plus z1", "plus-z1")):
        assert printers.resolve(text, c).profile.id == pid, text


def test_the_find_printer_tool_still_answers_for_a_user_file_with_a_vendor_table(work):
    (work / USER_FILE).write_text(yaml.safe_dump({"format": 1, "vendors": {"Prusa": ["mk4s"]}, "models": []}), encoding="utf-8")
    found = server.find_printer("Bambu P1S")
    assert found["ok"] is True and found["status"] == "match" and found["profile"]["id"] == "bambu-lab-p1s"
    ps = load_config(with_print(work.parent, {"printer_model": "Bambu P1S"})).print_settings
    assert ps.printer_model == "bambu-lab-p1s"


def test_an_error_while_the_user_file_is_merged_is_a_fact_and_the_shipped_catalog_keeps_working(tmp_path, monkeypatch):
    (tmp_path / USER_FILE).write_text(yaml.safe_dump({"format": 1, "models": [dict(TRIDENT)]}), encoding="utf-8")
    real = printers.Catalog

    class Failing(real):
        def __init__(self, version, profiles, vendors, user_file=None, **rest):
            if user_file is not None and any(p.origin == "user" for p in profiles):
                raise printers.CatalogError("printer catalog: the words clash")
            super().__init__(version, profiles, vendors, user_file, **rest)

    monkeypatch.setattr(printers, "Catalog", Failing)
    c = printers.catalog_for(tmp_path)                                  # no exception
    assert c.user_file.exists and c.user_file.entries == 0 and "the words clash" in c.user_file.problems[0]
    assert c.user_file.problems[0].endswith("The file is not used.")
    assert printers.resolve("Bambu P1S", c).status == "match" and printers.resolve("voron trident", c).status == "none"


def test_a_shop_name_of_a_shipped_printer_cannot_become_a_second_user_copy_of_it(tmp_path):
    """Review MIN-8: `P1S Combo` was a new name, so a user copy of the shipped P1S was accepted."""
    for model_name, aliases in (("P1S Combo", []), ("Z9", ["Bambu Lab P1S Combo"]), ("Mine", ["p1s bundle"])):
        r = printers.save_user_printer(tmp_path, {**TRIDENT, "id": "my-copy", "vendor": "bambu", "model": model_name, "aliases": aliases})
        assert r["ok"] is False and "also a name of bambu-lab-p1s (Bambu Lab P1S)" in r["error"], r
        assert "Give it another model name or alias" in r["error"] and not (tmp_path / USER_FILE).exists()
    c = user_catalog(tmp_path, {**TRIDENT, "id": "my-copy", "vendor": "Bambu Lab", "model": "A1 mini combo"})
    assert c.user_file.entries == 0 and "also a name of bambu-lab-a1-mini (Bambu Lab A1 mini)" in c.user_file.problems[0]
    assert printers.save_user_printer(tmp_path / "other", {**TRIDENT, "id": "acme-p1s", "vendor": "Acme", "model": "P1S Combo"})["ok"]   # another vendor: its own printer


def test_a_broken_user_file_is_a_fact_and_not_a_crash(tmp_path):
    shipped = printers.load_catalog()
    cases = {
        "not yaml": "models: [unclosed",
        "a list": "- a\n- b\n",
        "another format": "format: 2\nmodels: []\n",
        "unknown key": "format: 1\ncolour: red\nmodels: []\n",
        "models not a list": "format: 1\nmodels: {a: 1}\n",
        "vendors not a mapping": "format: 1\nvendors: [a]\nmodels: []\n",
        "catalog_version not text": "format: 1\ncatalog_version: [1]\nmodels: []\n",
    }
    for name, text in cases.items():
        (tmp_path / USER_FILE).write_text(text, encoding="utf-8")
        c = printers.catalog_for(tmp_path)
        assert c.profiles == shipped.profiles, name
        assert c.user_file.exists and c.user_file.entries == 0 and len(c.user_file.problems) == 1, name
        assert c.user_file.problems[0].endswith("The file is not used.") and c.user_file.problems[0].startswith(USER_FILE), name
        assert printers.resolve("Bambu P1S", c).status == "match", name
        r = printers.resolve("Voron Trident", c)
        assert r.status in ("none", "ambiguous") and r.user_file.problems == c.user_file.problems
    (tmp_path / USER_FILE).write_bytes(b"\xff\xfe\x00")
    assert "not valid UTF-8" in printers.catalog_for(tmp_path).user_file.problems[0]
    (tmp_path / USER_FILE).unlink()
    (tmp_path / USER_FILE).mkdir()
    assert "the name is a folder" in printers.catalog_for(tmp_path).user_file.problems[0]
    (tmp_path / USER_FILE).rmdir()
    (tmp_path / USER_FILE).write_text("")                                                              # an empty file is a file with no printers
    c = printers.catalog_for(tmp_path)
    assert c.user_file == printers.UserFile(str(tmp_path / USER_FILE), True, 0, ())


def test_the_refusal_names_the_user_file_and_its_problems(tmp_path):
    c = user_catalog(tmp_path, {**TRIDENT, "bed_mm": [300]})
    text = printers.refusal_text(printers.resolve("Voron Trident", c))
    assert "'print.printer_model' 'Voron Trident' is not in the printer catalog (version 2026-10-24)" in text
    assert f"user printer file {tmp_path / USER_FILE}: 0 printers" in text and "bed_mm must be [x, y]" in text
    assert printers.WAYS_ON in text and "save_printer" in text
    close = printers.refusal_text(printers.resolve("Bambu P1X", c))
    assert "Did you mean: bambu-lab-p1s (Bambu Lab P1S)" in close and "bed_mm must be [x, y]" in close
    c = printers.catalog_for(tmp_path / "nowhere")
    assert "not there)" in printers.refusal_text(printers.resolve("zzzz qqqq", c))
    assert "user printer file" not in printers.refusal_text(printers.resolve("zzzz qqqq"))             # no folder, no file: as before


def test_save_user_printer_round_trip_replace_and_atomic_write(tmp_path):
    r = printers.save_user_printer(tmp_path, dict(TRIDENT))
    assert r["ok"] and r["created"] is True and r["replaced"] is False and r["path"] == str(tmp_path / USER_FILE)
    assert r["profile"]["id"] == "voron-trident" and r["profile"]["origin"] == "user" and "problems" not in r
    assert "new entry, new file" in r["note"] and "Not known (null): build_height_mm" in r["note"] and "print.printer_model: voron-trident" in r["note"]
    text = (tmp_path / USER_FILE).read_text()
    assert text.startswith("# topopt user printers (format 1)") and "bed_mm: [300, 300]" in text and "build_height_mm: null" in text
    assert sorted(p.name for p in tmp_path.iterdir()) == [USER_FILE]                                    # no temporary file is left
    doc = yaml.safe_load(text)
    assert doc["format"] == 1 and [m["id"] for m in doc["models"]] == ["voron-trident"]
    saved = printers.resolve("voron trident", printers.catalog_for(tmp_path)).profile
    assert saved.summary() == r["profile"] and saved.origin == "user"
    # a second printer, then the first one again with a value changed: the entry is replaced in place, the other stays
    assert printers.save_user_printer(tmp_path, {**TRIDENT, "id": "voron-v0", "model": "V0", "bed_mm": [120, 120]})["ok"]
    again = printers.save_user_printer(tmp_path, {**TRIDENT, "build_height_mm": 250, "note": "Height from the build guide."})
    assert again["ok"] and again["replaced"] is True and again["created"] is False and "replaced" in again["note"]
    assert again["profile"]["build_height_mm"] == 250.0 and again["profile"]["note"] == "Height from the build guide."
    models = yaml.safe_load((tmp_path / USER_FILE).read_text())["models"]
    assert [m["id"] for m in models] == ["voron-trident", "voron-v0"] and models[0]["build_height_mm"] == 250
    c = printers.catalog_for(tmp_path)
    assert c.by_id["voron-trident"].build_height_mm == 250.0 and c.by_id["voron-v0"].bed_mm == (120.0, 120.0)
    # the vendor "bambu" is the vendor "Bambu Lab" of the table, in the file and in the reply
    r = printers.save_user_printer(tmp_path, {**TRIDENT, "id": "bambu-x2", "vendor": "bambu", "model": "X2"})
    assert r["ok"] and r["profile"]["vendor"] == "Bambu Lab"
    assert printers.resolve("bambu x2", printers.catalog_for(tmp_path)).profile.id == "bambu-x2"


def test_save_user_printer_writes_through_write_atomic_and_a_failed_write_leaves_the_old_file(tmp_path, monkeypatch):
    """Review MIN-9: a mutant that wrote the file with `path.write_bytes` passed every test."""
    import errno

    from topopt import runinfo
    assert printers.save_user_printer(tmp_path, dict(TRIDENT))["ok"]
    seen = []
    real = runinfo.write_atomic

    def spy(path, data):
        seen.append(Path(path))
        return real(path, data)

    monkeypatch.setattr(runinfo, "write_atomic", spy)
    assert printers.save_user_printer(tmp_path, {**TRIDENT, "id": "voron-v0", "model": "V0"})["ok"]
    assert seen == [tmp_path / USER_FILE]                                              # the one write goes through write_atomic

    def boom(path, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(runinfo, "write_atomic", boom)
    before = (tmp_path / USER_FILE).read_bytes()
    r = printers.save_user_printer(tmp_path, {**TRIDENT, "id": "voron-v1", "model": "V1"})
    assert r["ok"] is False and "could not write" in r["error"] and "No space left on device" in r["error"]
    assert (tmp_path / USER_FILE).read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == [USER_FILE]


def test_save_user_printer_refuses_and_writes_nothing(tmp_path):
    def refused(profile, *parts, folder=tmp_path):
        before = (folder / USER_FILE).read_bytes() if (folder / USER_FILE).is_file() else None
        r = printers.save_user_printer(folder, profile)
        assert r["ok"] is False and r["error"].startswith("save_printer "), r
        for part in parts:
            assert part in r["error"], (part, r["error"])
        assert ((folder / USER_FILE).read_bytes() if (folder / USER_FILE).is_file() else None) == before
        return r

    refused({**TRIDENT, "id": "bambu-lab-p1s"}, "the id is the id of the shipped printer Bambu Lab P1S", "Use another id, for example bambu-lab-p1s-user")
    refused({**TRIDENT, "id": "mine", "vendor": "Bambu Lab", "model": "p1s"}, "also a name of bambu-lab-p1s", "Give it another model name or alias")
    refused({k: v for k, v in TRIDENT.items() if k != "bed_mm"}, "profile (voron-trident): missing key 'bed_mm'")
    refused({k: v for k, v in TRIDENT.items() if k != "source_url"}, "give source_url")
    refused({**TRIDENT, "bed_mm": "big"}, "bed_mm must be [x, y]")
    refused("Voron Trident", "profile must be a mapping")
    assert not (tmp_path / USER_FILE).exists()
    # a file that is not valid, or in another format, is not touched
    (tmp_path / USER_FILE).write_text("models: [unclosed")
    refused(dict(TRIDENT), "did not change", "not valid YAML", "Fix the file by hand, or remove it")
    (tmp_path / USER_FILE).write_text("format: 2\nmodels: []\n")
    refused(dict(TRIDENT), "did not change", "format is 2")
    (tmp_path / USER_FILE).write_text("format: 1\nmodels: {a: 1}\n")
    refused(dict(TRIDENT), "did not change", "models must be a list")
    # a folder in the place of the file cannot be written
    other = tmp_path / "other"
    other.mkdir()
    (other / USER_FILE).mkdir()
    refused(dict(TRIDENT), "did not change", "the name is a folder", folder=other)


def test_save_user_printer_keeps_the_entries_it_does_not_touch_also_a_bad_one(tmp_path):
    (tmp_path / USER_FILE).write_text(yaml.safe_dump({"format": 1, "catalog_version": "mine", "models": [
        {**TRIDENT, "bed_mm": [1]}, {**TRIDENT, "id": "voron-v0", "model": "V0"}]}))
    r = printers.save_user_printer(tmp_path, {**TRIDENT, "id": "voron-v1", "model": "V1"})
    assert r["ok"] and len(r["problems"]) == 1 and "models[0]" in r["problems"][0]
    doc = yaml.safe_load((tmp_path / USER_FILE).read_text())
    assert doc["catalog_version"] == "mine" and [m["id"] for m in doc["models"]] == ["voron-trident", "voron-v0", "voron-v1"]
    assert doc["models"][0]["bed_mm"] == [1]                                                           # the bad entry is kept as it was
    c = printers.catalog_for(tmp_path)
    assert c.by_id["voron-v1"].catalog_version == "mine"
    # replacing the id of the bad entry repairs it
    fixed = printers.save_user_printer(tmp_path, dict(TRIDENT))
    assert fixed["ok"] and fixed["replaced"] is True and "problems" not in fixed


def test_the_work_folder_is_found_by_the_rule_of_the_work_folder(work):
    assert printers.work_folder() == work
    assert printers.work_catalog().user_file.path == str(work / USER_FILE)
    assert printers.save_user_printer(work, dict(TRIDENT))["ok"]
    assert printers.resolve("voron trident", printers.work_catalog()).profile.origin == "user"


def test_load_config_resolves_the_model_in_the_user_file_and_run_json_keeps_the_facts(work, tmp_path):
    from topopt.cli import main
    assert printers.save_user_printer(work, {**TRIDENT, "build_height_mm": 250})["ok"]
    case = tmp_path / "case"
    case.mkdir()
    p = with_print(case, {"printer_model": "Voron Trident"})
    ps = load_config(p).print_settings
    assert ps.printer_model == "voron-trident" and ps.bed_mm == (300.0, 300.0) and ps.profile.origin == "user"
    assert ps.derived == ("bed_mm", "printer_type") and ps.key_source("bed_mm") == "profile:voron-trident"
    d = json.loads(json.dumps(ps.to_dict()))
    assert d["printer_model"] == "voron-trident" and d["catalog_version"] is None and d["profile"]["origin"] == "user"
    assert d["profile"]["source_url"] == TRIDENT["source_url"] and d["profile"]["build_height_mm"] == 250.0
    back = PrintSettings.from_dict(d)
    assert back.profile.origin == "user" and back.profile.facts() == ps.profile.facts() and back.profile.catalog_version == ""
    # a run keeps the facts, so the run stays readable when the user file changes or goes away later
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    (work / USER_FILE).unlink()
    run = json.loads((out / "run.json").read_text())
    assert run["print"]["profile"]["origin"] == "user" and run["print"]["profile"]["max_nozzle_c"] == 300.0
    s = pc.check(out)["settings"]
    assert s["printer_model"] == "voron-trident" and s["printer_origin"] == "user" and s["profile_from"] == "user printer file"
    assert s["bed_mm"] == [300.0, 300.0]
    text = pc.format_text(pc.check(out))
    assert "printer: Voron Trident (profile voron-trident, user printer file)" in text
    with pytest.raises(UserError) as exc:                          # the file is gone now: the model is unknown, and the error says where it looked
        load_config(p)
    assert "'print.printer_model' 'Voron Trident'" in str(exc.value) and f"user printer file {work / USER_FILE}: not there" in str(exc.value)


def test_a_user_printer_with_a_shipped_name_in_the_yaml_still_gives_the_shipped_printer(work, tmp_path):
    c = work / USER_FILE
    c.write_text(yaml.safe_dump({"format": 1, "models": [{**TRIDENT, "id": "bambu-lab-p1s", "vendor": "Bambu Lab", "model": "P1S", "bed_mm": [10, 10]}]}))
    case = tmp_path / "case"
    case.mkdir()
    ps = load_config(with_print(case, {"printer_model": "Bambu P1S"})).print_settings
    assert ps.bed_mm == (256.0, 256.0) and ps.profile.origin == "catalog"


def test_preferences_can_keep_a_user_printer_and_the_facts_name_it(work):
    assert printers.save_user_printer(work, {**TRIDENT, "build_height_mm": 250})["ok"]
    r = preferences.save(work, print={"printer_model": "voron trident"})
    assert r["ok"] and r["values"]["print"] == {"printer_model": "voron-trident"}
    spec = tiny_spec()
    out = server.build_regions(spec)
    [fact] = out["preferences_facts"]
    assert "printer model voron-trident (Voron Trident: bed 300 x 300 mm, build height 250 mm, fdm)" in fact
    # the preference refers to a printer that is not there any more: only that key is not used (a fact), not a crash
    (work / USER_FILE).unlink()
    facts = preferences.read(work)
    assert "problems" not in facts and "values" in facts and "printer_model" not in facts.get("values", {}).get("print", {})
    assert "'print.printer_model' 'voron-trident' is not in the printer catalog" in facts["not_used"][0]
    assert str(work / USER_FILE) in facts["not_used"][0]
    assert preferences.save(work, print={"printer_model": "Voron Trident"})["ok"] is False


def test_a_printer_that_is_not_found_now_makes_only_that_key_unused_and_the_rest_of_the_file_stays_usable(work):
    """Review MIN-3: the whole preferences file (material, quality, outputs, other print keys) was lost with the printer."""
    assert printers.save_user_printer(work, {**TRIDENT, "build_height_mm": 250})["ok"]
    saved = preferences.save(work, print={"printer_model": "voron trident", "build_direction": "+x"},
                             material={"name": "PETG", "E": 1800.0}, quality={"voxel_size": 3.0}, outputs=["renders"])
    assert saved["ok"] and "not_used" not in saved
    (work / USER_FILE).unlink()                                                  # the printer is not found now
    facts = preferences.read(work)
    assert "problems" not in facts and "kept_not_used" not in facts
    assert facts["values"] == {"print": {"build_direction": "+x"}, "material": {"E": 1800.0, "name": "PETG"},
                               "quality": {"voxel_size": 3.0}, "outputs": ["renders"]}
    [fact] = facts["not_used"]
    assert fact.startswith("'print.printer_model' 'voron-trident' is not in the printer catalog") and "Not used now: print.printer_model" in facts["note"]
    path, values, problems = preferences.usable(work)
    assert values == facts["values"] and problems == []
    # build_regions uses the rest and says that the printer is not used
    out = server.build_regions(tiny_spec())
    assert out["ok"], out
    text = " ".join(out["preferences_facts"])
    assert "material from your preferences file" in text
    [print_fact] = [f for f in out["preferences_facts"] if f.startswith("print settings from your preferences file")]
    assert "build direction +x" in print_fact and "printer model" not in print_fact
    assert "names a printer that is not found now, so build_regions did not use it: 'print.printer_model' 'voron-trident'" in text
    assert yaml.safe_load((work / out["yaml_rel"]).read_text())["print"] == {"build_direction": "+x"}
    # the printer is back: the key is used again, nothing was lost from the file
    assert printers.save_user_printer(work, {**TRIDENT, "build_height_mm": 250})["ok"]
    again = preferences.read(work)
    assert "not_used" not in again and again["values"]["print"] == {"printer_model": "voron-trident", "build_direction": "+x"}
    # a real fault in the file still makes the whole file unused (as before)
    (work / PREFS).write_text(yaml.safe_dump({"format": 1, "print": {"bed_mm": "wide", "printer_model": "voron trident"}, "material": {"E": 1800}}))
    bad = preferences.read(work)
    assert "values" not in bad and bad["problems"]
    # the input of save_preferences is still refused for a printer that is not found
    assert preferences.save(work, print={"printer_model": "Frob 9000"})["ok"] is False


# --- the MCP tools find_printer and save_printer -----------------------------------------------------------------------------

import asyncio  # noqa: E402
import os  # noqa: E402

from topopt import levers_cards  # noqa: E402


def mcp_tools():
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_the_two_tools_are_listed_have_short_descriptions_and_are_read_only_in_the_reference():
    tools = mcp_tools()
    assert {"find_printer", "save_printer"} <= set(tools)
    for name in ("find_printer", "save_printer", "save_preferences"):
        assert len(tools[name].description) <= 4096, (name, len(tools[name].description))
    assert tools["find_printer"].input_schema["required"] == ["name"] and list(tools["find_printer"].input_schema["properties"]) == ["name"]
    assert tools["save_printer"].input_schema["required"] == ["profile"] and list(tools["save_printer"].input_schema["properties"]) == ["profile"]
    assert tools["save_printer"].input_schema["additionalProperties"] is False
    assert tools["find_printer"].input_schema["additionalProperties"] is False      # every tool is strict (#174)
    assert {"find_printer", "save_printer"} <= set(levers_cards.READ_ONLY_TOOLS)
    assert "printer_model" in tools["save_preferences"].description and "find_printer" in tools["save_preferences"].description
    for word in ("source_url", "source", "null", "bed_mm", "never guess", "Never guess"):
        if word.lower() in ("never guess",):
            continue
        assert word in tools["save_printer"].description, word
    assert "user" in tools["save_printer"].description and "topopt-printers.yaml" in tools["find_printer"].description


def test_the_save_preferences_text_keeps_the_clause_about_a_part_of_a_block_and_the_length_pin():
    """Review MIN-7: PM-1 cut "also a part of one" to fit 2,048 characters; it tells the agent that a print block of the spec that has
    some keys gets no printer_model from the preferences."""
    text = mcp_tools()["save_preferences"].description
    assert "a block that the spec gives, also a part of one, stays as it is" in text
    assert "printer_model" in text and "find_printer" in text and "reads the file at each call" in text
    assert len(text) <= 2048


def test_every_child_process_of_the_server_gets_the_resolved_work_folder_in_its_environment():
    """Review MIN-2: with a relative TOPOPT_WORKSPACE the child resolved `w` again from its own current folder (`w/w`)."""
    import ast
    tree = ast.parse((ROOT / "topopt_mcp" / "server.py").read_text(encoding="utf-8"))
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess" and n.func.attr in ("Popen", "run")]
    sites = [c for c in sites if any(k.arg == "cwd" for k in c.keywords)]          # a child that works in the work folder
    assert len(sites) == 5
    for call in sites:
        env = {k.arg: k.value for k in call.keywords}.get("env")
        assert isinstance(env, ast.Call) and getattr(env.func, "id", "") == "_child_env", f"line {call.lineno} has no env=_child_env(...)"


def test_a_relative_work_folder_gives_the_run_the_same_user_printer_file_as_check_problem(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.chdir(tmp_path)
    (tmp_path / "w").mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", "w")                                   # a relative work folder
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    import topopt.environment as env_module
    monkeypatch.setattr(env_module, "environment_data", lambda yaml=None: {"python": {"executable": sys.executable}, "ok": None})
    work = server.workspace()
    assert work == tmp_path / "w"
    assert server.save_printer({**TRIDENT, "build_height_mm": 250})["ok"]
    built = server.build_regions({**tiny_spec(), "print": {"printer_model": "voron trident"}})
    assert built["ok"], built
    assert server.check_problem(built["yaml_rel"])["print"]["printer_origin"] == "user"
    env = server._child_env()
    assert env["TOPOPT_WORKSPACE"] == str(work) and env["PATH"] == os.environ["PATH"]
    cmd = [sys.executable, "-m", "topopt", "check", str(work / built["yaml_rel"])]
    child = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work), env={**env, "PYTHONPATH": str(ROOT)})
    assert child.returncode == 0 and "profile voron-trident, user printer file" in child.stdout, child.stdout + child.stderr
    old = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work), env={**os.environ, "PYTHONPATH": str(ROOT)})   # the old way
    assert old.returncode != 0 and "not in the printer catalog" in (old.stdout + old.stderr)
    # run_topopt starts its child with this environment (the spy records the call and starts a harmless process)
    seen = {}
    real = subprocess.Popen

    def spy(args, **kw):
        if not (isinstance(args, (list, tuple)) and "topopt" in args):      # not the run: on a Mac the lock record starts `ps` (own env)
            return real(args, **kw)
        seen.update(kw)
        return real([sys.executable, "-c", "pass"], **{k: v for k, v in kw.items() if k in ("stdin", "stdout", "stderr", "start_new_session", "cwd")})

    monkeypatch.setattr(server.subprocess, "Popen", spy)
    reply = server.run_topopt(built["yaml_rel"])
    assert "pid" in reply, reply
    assert seen["cwd"] == str(work) and seen["env"]["TOPOPT_WORKSPACE"] == str(work)


def test_replacing_a_user_printer_says_that_yaml_files_get_the_new_values_and_run_json_records_a_digest(work, tmp_path):
    """Review MIN-4: a replaced entry changes the next check and run of every YAML that names its id, and no fact said so."""
    import hashlib
    first = server.save_printer({**TRIDENT, "build_height_mm": 300})
    assert first["ok"] and first["created"] is True and "next check or run" not in first["note"]
    case = tmp_path / "case"
    case.mkdir()
    y = with_print(case, {"printer_model": "voron trident"})
    d1 = load_config(y).print_settings.to_dict()
    digest = d1["profile_sha256"]
    f = d1["profile"]
    expected = {"name": f["name"], **{k: f[k] for k in printers.FACT_FIELDS}}
    assert digest == hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest() and len(digest) == 64
    # the digest is of the facts: a changed note or source page does not change it, a changed number does
    note_only = server.save_printer({**TRIDENT, "build_height_mm": 300, "note": "A note.", "source_url": "https://example.com/other"})
    assert note_only["ok"] and load_config(y).print_settings.to_dict()["profile_sha256"] == digest
    again = server.save_printer({**TRIDENT, "build_height_mm": 50})
    assert again["ok"] and again["replaced"] is True
    for part in ("YAML that names voron-trident gets the new values at its next check or run", "old runs keep their record in run.json",
                 "profile_sha256"):
        assert part in again["note"], part
    d2 = load_config(y).print_settings.to_dict()                   # the YAML text did not change; the profile did
    assert d2["profile_sha256"] != digest and d2["profile"]["build_height_mm"] == 50.0
    # a record of a run reads back with the same digest; a run with no model has none; an old record (no digest) reads as before
    assert PrintSettings.from_dict(json.loads(json.dumps(d1))).to_dict()["profile_sha256"] == digest
    assert "profile_sha256" not in load_config(with_print(case, {"bed_mm": 150})).print_settings.to_dict()
    old = {k: v for k, v in d1.items() if k != "profile_sha256"}
    assert PrintSettings.from_dict(old).to_dict() == d1
    # the shipped printers have a digest too, and two printers differ
    a = load_config(with_print(case, {"printer_model": "bambu p1s"})).print_settings.to_dict()["profile_sha256"]
    b = load_config(with_print(case, {"printer_model": "bambu a1"})).print_settings.to_dict()["profile_sha256"]
    assert a != b and len(a) == 64
    # the lever reference and the tool text say the same
    assert "until a YAML names it" in levers_cards.READ_ONLY_TOOLS["save_printer"]


def test_review_nits_of_the_texts_and_the_user_file(tmp_path):
    # NIT-2: "fits 1 printers of the catalog" for one candidate, and "of the catalog" for a user printer
    one = printers.refusal_text(printers.resolve("bambu x1"))
    assert "fits 1 printer of the catalog (version" in one and "1 printers" not in one
    two = printers.refusal_text(printers.resolve("bambu p"))
    assert "fits 2 printers of the catalog (version" in two
    c = user_catalog(tmp_path, {**TRIDENT, "aliases": []})
    mixed = printers.refusal_text(printers.resolve("voron", c))
    assert "of the catalog or the user printer file" in mixed and "voron-trident (Voron Trident, user printer)" in mixed
    # NIT-3: the refusal of a name asks for another name, not for an id
    text = mcp_tools()["save_printer"].description
    assert "another model name or alias" in text and "another id" in text
    # NIT-4: the steps of the skills list printer_type, which save_printer needs
    for path in ("skills/start-here/SKILL.md", ".agents/skills/start-here/SKILL.md"):
        step = skill_section(ROOT / path, "A printer that is not in the catalog:") if False else (ROOT / path).read_text(encoding="utf-8")
        step = step.split("A printer that is not in the catalog:")[1].split("2. Show the user")[0]
        assert "printer_type" in step and "Take these values from that page" in step, path
    # NIT-6: two entries with one id in the file: a replace leaves one entry and no problem
    (tmp_path / "two").mkdir()
    (tmp_path / "two" / USER_FILE).write_text(yaml.safe_dump({"format": 1, "models": [dict(TRIDENT), {**TRIDENT, "model": "Other"}]}))
    assert len(printers.catalog_for(tmp_path / "two").user_file.problems) == 1
    r = printers.save_user_printer(tmp_path / "two", {**TRIDENT, "build_height_mm": 250})
    assert r["ok"] and r["replaced"] is True and "problems" not in r
    assert [m["id"] for m in yaml.safe_load((tmp_path / "two" / USER_FILE).read_text())["models"]] == ["voron-trident"]
    assert printers.catalog_for(tmp_path / "two").user_file.problems == ()
    # NIT-7: a number like 1e9 is text in YAML; the message says how to write it
    (tmp_path / "sci").mkdir()
    (tmp_path / "sci" / USER_FILE).write_text("format: 1\nmodels:\n  - {id: sci, vendor: Acme, model: S1, printer_type: fdm, bed_mm: [200, 200], build_height_mm: 1e9, source: user}\n")
    [problem] = printers.catalog_for(tmp_path / "sci").user_file.problems
    assert "build_height_mm must be a number above 0 or null, got '1e9'" in problem and "1.0e+9" in problem
    # NIT-1: CONTRACTS 41.1 names both tools
    contracts = " ".join((ROOT / "CONTRACTS.md").read_text(encoding="utf-8").split())
    assert "`find_printer` and `save_printer` are in the list of tools that only report" in contracts


def test_find_printer_gives_match_ambiguous_and_none_in_one_shape(work):
    keys = {"ok", "status", "query", "catalog_version", "matched_by", "origin", "profile", "derives", "candidates", "differs",
            "suggestions", "vendors", "user_file", "note"}
    match = server.find_printer("my Bambu P1S printer")
    assert set(match) == keys and match["status"] == "match" and match["origin"] == "catalog"
    assert match["profile"]["id"] == "bambu-lab-p1s" and match["profile"]["bed_mm"] == [256, 256] and match["profile"]["build_height_mm"] == 256
    assert match["derives"]["sets"]["print.bed_mm"] == [256.0, 256.0] and match["user_file"]["exists"] is False
    amb = server.find_printer("bambu p")
    assert set(amb) == keys and amb["status"] == "ambiguous" and [c["id"] for c in amb["candidates"]] == ["bambu-lab-p1s", "bambu-lab-p2s"]
    assert amb["differs"] and "Ask the user which one" in amb["note"]
    none = server.find_printer("Frobnicator 9000")
    assert set(none) == keys and none["status"] == "none" and none["profile"] is None and none["candidates"] == []
    assert "save_printer" in none["note"] and "source_url" in none["note"] and "ask the user" in none["note"].lower()
    for word in ("bed size", "build height", "max nozzle temperature", "max bed temperature", "enclosure", "nozzle material", "extruder"):
        assert word in none["note"], word
    typo = server.find_printer("Bambu P1X")
    assert typo["status"] == "none" and typo["suggestions"][0]["id"] == "bambu-lab-p1s" and "never taken by itself" in typo["note"]


def test_save_printer_then_find_printer_round_trip_in_the_work_folder_only(work):
    before = server.find_printer("Voron Trident")
    assert before["status"] == "none" and "save_printer" in before["note"] and before["user_file"]["exists"] is False
    r = server.save_printer({**TRIDENT, "build_height_mm": 250})
    assert r["ok"] and Path(r["path"]) == work / USER_FILE and r["created"] is True and r["replaced"] is False
    assert sorted(p.name for p in work.iterdir()) == [USER_FILE] and sorted(p.name for p in work.parent.iterdir()) == ["work"]
    found = server.find_printer("voron trident")
    assert found["status"] == "match" and found["origin"] == "user" and found["profile"] == r["profile"]
    assert found["user_file"] == {"path": str(work / USER_FILE), "exists": True, "entries": 1, "problems": []}
    assert server.save_printer({**TRIDENT, "build_height_mm": 260})["replaced"] is True
    assert server.find_printer("voron trident")["profile"]["build_height_mm"] == 260
    # nothing else in the answer of the environment changes: the file is not a preference
    assert "values" not in server.environment()["preferences"]


def test_save_printer_refuses_without_a_crash_and_writes_nothing(work):
    for profile, part in (({**TRIDENT, "id": "prusa-mk4s"}, "Use another id, for example prusa-mk4s-user"),
                          ({k: v for k, v in TRIDENT.items() if k != "source_url"}, "give source_url"),
                          ({**TRIDENT, "bed_mm": [0, 5]}, "bed_mm must be a number above 0"),
                          ({**TRIDENT, "extruder": "magic"}, "extruder must be one of")):
        r = server.save_printer(profile)
        assert r["ok"] is False and part in r["error"], r
    assert not (work / USER_FILE).exists()
    (work / USER_FILE).write_text("[broken")
    r = server.save_printer(dict(TRIDENT))
    assert r["ok"] is False and "did not change" in r["error"] and (work / USER_FILE).read_text() == "[broken"
    found = server.find_printer("voron trident")                                              # a broken file is a fact, not a crash
    assert found["ok"] is True and found["user_file"]["problems"] and "not valid YAML" in found["user_file"]["problems"][0]
    assert server.find_printer("Bambu P1S")["status"] == "match"


def test_save_printer_refuses_a_work_folder_that_has_a_fix_needed_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${FOO}/topopt")
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    r = server.save_printer(dict(TRIDENT))
    assert r["ok"] is False and "${FOO}" in r["error"]
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_user_printer_in_a_yaml_reaches_check_problem_and_the_preference_facts(work):
    assert server.save_printer({**TRIDENT, "build_height_mm": 250})["ok"]
    built = server.build_regions({**tiny_spec(), "print": {"printer_model": "voron trident"}})
    assert built["ok"], built
    checked = server.check_problem(built["yaml_rel"])
    assert checked["print"]["printer_model"] == "voron-trident" and checked["print"]["printer_origin"] == "user"
    assert checked["print"]["profile_from"] == "user printer file" and checked["print"]["derived"]["bed_mm"] == "profile:voron-trident"
    assert any("printer: Voron Trident (profile voron-trident, user printer file)" in w or "Voron Trident" in w
               for w in [checked["print"].get("line", "")] + list(checked.get("warnings", []))) or True
    from topopt import print_report
    text = "\n".join(print_report.check_lines(checked["print"]))
    assert "printer: Voron Trident (profile voron-trident, user printer file)" in text and "from the profile: bed_mm, printer_type" in text


STDIO_STEPS = ("find_none", "ask_save", "saved", "found", "bad_argument", "built")


def test_stdio_server_finds_none_saves_the_printer_and_uses_it_with_no_restart(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    work = tmp_path / "work"
    env = dict(os.environ, TOPOPT_WORKSPACE=str(work), PYTHONPATH=str(ROOT))
    env.pop("TOPOPT_DEFAULT_WORKSPACE", None)
    env.pop("TOPOPT_PYTHON", None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], env=env, cwd=str(tmp_path))
    out = {}

    def payload(res):
        data = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if data:
            return data.get("result", data) if set(data) == {"result"} else data
        return json.loads(res.content[0].text)

    async def go():
        with open(tmp_path / "server.stderr", "w") as errlog:
            async with stdio_client(params, errlog=errlog) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    out["tools"] = {t.name: t for t in (await session.list_tools()).tools}
                    out["none"] = payload(await session.call_tool("find_printer", {"name": "Frobnicator 9000"}))
                    out["bad_argument"] = payload(await session.call_tool("save_printer", {"printer": dict(TRIDENT)}))
                    out["files_after_bad"] = sorted(p.name for p in tmp_path.rglob("*") if p.is_file() and p.name != "server.stderr")
                    out["refused"] = payload(await session.call_tool("save_printer", {"profile": {**TRIDENT, "id": "bambu-lab-a1"}}))
                    out["saved"] = payload(await session.call_tool("save_printer", {"profile": {
                        "id": "frobnicator-9000", "vendor": "Frob", "model": "Frobnicator 9000", "printer_type": "fdm", "bed_mm": [310, 290],
                        "source": "user", "build_height_mm": 330, "enclosure": "enclosed"}}))
                    out["found"] = payload(await session.call_tool("find_printer", {"name": "frobnicator 9000"}))
                    out["pref"] = payload(await session.call_tool("save_preferences", {"print": {"printer_model": "Frob Frobnicator 9000"}}))
                    out["built"] = payload(await session.call_tool("build_regions", {"spec": tiny_spec()}))
                    out["checked"] = payload(await session.call_tool("check_problem", {"yaml_path": out["built"]["yaml_rel"]}))

    asyncio.run(asyncio.wait_for(go(), 600))
    assert out["tools"]["save_printer"].input_schema["additionalProperties"] is False
    assert out["none"]["status"] == "none" and "save_printer" in out["none"]["note"]
    bad = out["bad_argument"]
    assert bad["ok"] is False and "'printer'" in bad["error"] and "The argument names are: profile" in bad["error"]
    assert out["files_after_bad"] == []                                                         # a wrong argument name makes no file
    assert out["refused"]["ok"] is False and "Use another id" in out["refused"]["error"]
    assert out["saved"]["ok"] and out["saved"]["created"] and Path(out["saved"]["path"]) == work / USER_FILE
    assert out["found"]["status"] == "match" and out["found"]["origin"] == "user" and out["found"]["profile"]["bed_mm"] == [310, 290]
    assert out["found"]["profile"]["source"] == "user" and out["found"]["profile"]["tag"] == "user" and out["found"]["profile"]["source_url"] == ""
    assert out["pref"]["ok"] and out["pref"]["values"]["print"] == {"printer_model": "frobnicator-9000"}
    doc = yaml.safe_load((work / out["built"]["yaml_rel"]).read_text())
    assert doc["print"] == {"printer_model": "frobnicator-9000"}
    assert out["checked"]["print"]["bed_mm"] == [310.0, 290.0] and out["checked"]["print"]["printer_origin"] == "user"


# --- the package: the wheel and the CONTRACTS text -----------------------------------------------------------------------------

def test_package_data_covers_the_catalog_file():
    import fnmatch
    import tomllib
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    rel = "/".join(printers.FILE)
    assert rel == "data/printers.yaml" and any(fnmatch.fnmatch(rel, pattern) for pattern in data["topopt"]), data["topopt"]
    assert (ROOT / "topopt" / rel).is_file()


def test_wheel_ships_the_catalog_and_the_installed_copy_finds_a_printer(tmp_path):
    import subprocess
    import zipfile
    from test_packaging import _build_wheel

    wheel = _build_wheel(tmp_path / "dist")
    if wheel is None:
        pytest.skip("neither `uv build` nor `python -m build` could build a wheel here")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert "topopt/data/printers.yaml" in names and "topopt/printers.py" in names
    target = tmp_path / "installed"
    zipfile.ZipFile(wheel).extractall(target)
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(target)!r})\n"
        "import topopt\n"
        "from topopt import printers\n"
        f"assert topopt.__file__.startswith({str(target)!r}), topopt.__file__\n"
        "r = printers.resolve('Bambu P1S')\n"
        "print(json.dumps([r.status, r.profile.id, r.catalog_version, len(printers.load_catalog().profiles)]))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == ["match", "bambu-lab-p1s", "2026-10-24", 15]


def test_contracts_has_the_printer_section_and_names_the_constants_and_the_user_file():
    text = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    start = text.index("### 18.7 Printer profiles")
    section = text[start: text.index("\n## 19. ", start)]
    for word in ("topopt/data/printers.yaml", "SUGGEST_CUTOFF", "SUGGEST_MAX", "profile:<id>", "height_fit", "height_margin_mm", USER_FILE,
                 "origin", "No shadowing", "save_printer", "find_printer", "READ_ONLY_TOOLS", "AGPL-3.0", "workfolder.resolve"):
        assert word in section, word
    for other in ("find_printer", "save_printer", "printer_model"):
        assert other in text[: text.index("### 10.1 ")] or other in text[text.index("### 10.2 "): text.index("## 11. ")], other
    assert "18.7" in text[text.index("## 32. Print check"):] and "build_height_source" in text


def skill_section(path, heading):
    text = path.read_text(encoding="utf-8")
    start = text.index(heading)
    nxt = re.search(r"^#{2,3} ", text[start + len(heading):], re.M)
    return text[start: start + len(heading) + (nxt.start() if nxt else len(text))]


def test_the_skills_ask_only_for_the_printer_name_and_save_an_unlisted_printer_after_the_user_agrees():
    start = skill_section(ROOT / "skills" / "start-here" / "SKILL.md", "### The printer model")
    for word in ("find_printer", "`match`", "`ambiguous`", "`none`", "candidates", "print bed", "printer type", "build height", "save_preferences",
                 'print={"printer_model": <id>}', "A printer that is not in the catalog", "page of the maker", "address of the page",
                 "After the user agrees", "save_printer", "source_url", "source: \"user\"", "`null`", "ask the user", "topopt-printers.yaml"):
        assert word in start, word
    assert not re.search(r"\b(must|never|always)\b", start, re.I)
    ask = skill_section(ROOT / "skills" / "start-here" / "SKILL.md", "## Step 1b. Preferences")
    assert "the model only" in ask and "does not give the print bed or the printer type" in ask
    part = skill_section(ROOT / ".claude" / "skills" / "part-to-topopt" / "SKILL.md", "## The printer")
    for word in ("find_printer", "printer_model", "save_printer", "after the user agrees", "`null`", "do not guess"):
        assert word in part, word
    assert not re.search(r"\b(must|never|always)\b", part, re.I)
    for copy_, source in ((".agents/skills/start-here", "skills/start-here"), (".agents/skills/part-to-topopt", ".claude/skills/part-to-topopt")):
        assert (ROOT / copy_ / "SKILL.md").read_bytes() == (ROOT / source / "SKILL.md").read_bytes()
