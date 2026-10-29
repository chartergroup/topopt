"""PM-2: the filament catalog, the name lookup, and the user filament file (CONTRACTS.md section 18.8)."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import catalogs, compat, materials_catalog as mc, printers  # noqa: E402

CATALOG_FILE = ROOT / "topopt" / "data" / "materials.yaml"
OPENED = "2026-10-24"
FAMILIES = {"ABS", "ASA", "ASA-CF", "PC", "PC-ABS", "PC-PBT", "PC-CF", "CoPA", "PA6-CF", "PA12-CF", "PA6-GF", "PA612-CF", "PAHT-CF", "PA11",
            "PA11-CF", "PETG-CF", "PET-CF", "PLA-CF", "PLA", "PETG", "TPU", "PP-CF", "PP-GF", "PPS-CF", "PPS-GF"}


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(CATALOG_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cat():
    return mc.load_catalog()


@pytest.fixture(scope="module")
def by(raw):
    return {m["id"]: m for m in raw["materials"]}


def row(fid, maker="Acme", product=None, **over):
    """A valid raw row for a synthetic catalog."""
    m = {"id": fid, "family": "PLA", "maker": maker, "product": product or fid.upper(), "aliases": [], "sheet_version": "V1",
         "E_xy_mpa": 2000, "E_z_mpa": 1800, "bending_modulus_xy_mpa": None, "bending_modulus_z_mpa": None, "strength_xy_mpa": 40, "strength_z_mpa": 30, "strength_kind": "not stated",
         "elongation_xy_pct": 5, "density_g_cm3": 1.2, "hdt_18_c": 55, "hdt_045_c": 58, "conditioned": None, "nu": None,
         "nozzle_c": [190, 230], "bed_c": [30, 60], "enclosure": "no", "chamber_c_min": None, "hardened_nozzle": False, "drying": None,
         "direct_drive": None, "abrasive": False, "soft": False, "source_url": "https://example.com/" + fid, "opened": OPENED,
         "tag": "maker TDS"}
    m.update(over)
    return m


def synthetic(*rows, makers=None):
    return mc.parse_catalog({"format": 1, "catalog_version": "t1", "makers": makers or {"Acme": ["acme"], "Zeta": ["zeta"]},
                             "materials": list(rows)})


# --- the file: schema, sources, licence --------------------------------------------------------------------------------------

def test_the_file_has_the_header_format_and_version(raw):
    assert raw["format"] == 1 and isinstance(raw["catalog_version"], str) and raw["catalog_version"] == "2026-10-25"
    text = CATALOG_FILE.read_text(encoding="utf-8")
    head = text.split("\nformat:")[0]
    for word in ("Sources", "Licence", "MIT", "AGPL-3.0", "LGPL-3.0", "no price", "NO data of a slicer profile"):
        assert word in head, word


def test_the_catalog_covers_the_families_and_the_makers_of_the_research(raw, cat):
    assert len(raw["materials"]) == len(cat.profiles) == 69
    assert {m["family"] for m in raw["materials"]} == FAMILIES
    assert set(cat.makers) == {"Polymaker", "Bambu Lab", "Prusament", "Elegoo", "eSun", "AzureFilm"}
    assert {m["maker"] for m in raw["materials"]} == set(cat.makers)
    counts = {f: sum(1 for m in raw["materials"] if m["family"] == f) for f in FAMILIES}
    assert counts["PLA"] == 10 and counts["PETG"] == 5 and counts["ABS"] == 6 and counts["TPU"] == 5 and counts["PC"] == 7


def test_every_row_has_every_field_and_a_null_comes_with_a_note_that_names_it(raw):
    for m in raw["materials"]:
        assert not [k for k in mc.REQUIRED_FIELDS if k not in m], m["id"]
        assert not [k for k in m if k not in mc.REQUIRED_FIELDS + mc.OPTIONAL_FIELDS], m["id"]
        for key in mc.FACT_FIELDS:
            if m[key] is None and key not in mc.NULL_WITHOUT_NOTE:
                assert m.get("note") and key in m["note"], (m["id"], key)


def test_every_row_has_a_source(raw):
    for m in raw["materials"]:
        assert re.match(r"^https://", m["source_url"]), m["id"]
        assert m["opened"] in ("2026-10-22", OPENED, "2026-10-25") and m["tag"] in mc.TAGS, m["id"]
        for key, url in (m.get("field_sources") or {}).items():
            assert key in mc.SOURCED_FIELDS and re.match(r"^https://", url), (m["id"], key)


def test_the_file_has_no_price_and_no_slicer_source(raw):
    text = CATALOG_FILE.read_text(encoding="utf-8")
    body = text.split("\nformat:")[1]
    assert not re.search(r"price|cost|USD|EUR|\$\d", body, re.I)
    assert not [k for m in raw["materials"] for k in m if re.search(r"price|cost", k, re.I)]
    urls = re.findall(r"https?://[^\s\"']+", text)
    assert urls and not [u for u in urls if re.search(r"github\.com|OrcaSlicer|BambuStudio|PrusaSlicer|Cura", u, re.I)]


def test_poisson_ratio_is_null_in_every_row(raw):
    """No data sheet gives it (B-materials 3.4); the engine assumes the value of `Material`, and says so."""
    assert all(m["nu"] is None for m in raw["materials"])


def test_a_sheet_that_gives_a_z_value_never_gives_a_stronger_z_strength(raw):
    for m in raw["materials"]:
        if m["strength_xy_mpa"] is not None and m["strength_z_mpa"] is not None:
            assert m["strength_z_mpa"] <= m["strength_xy_mpa"], m["id"]
        if m["strength_kind"] != "not stated":
            assert m["strength_xy_mpa"] is not None or m["strength_z_mpa"] is not None, m["id"]


def test_the_values_are_the_ones_of_the_research_rows(by):
    abs_ = by["polymaker-polylite-abs"]
    assert (abs_["E_xy_mpa"], abs_["E_z_mpa"], abs_["strength_xy_mpa"], abs_["strength_z_mpa"]) == (2246.6, 2080.9, 33.4, 29.7)
    assert (abs_["density_g_cm3"], abs_["hdt_18_c"], abs_["hdt_045_c"], abs_["enclosure"]) == (1.04, 98, 100, "required")
    assert "1.12" in abs_["note"] and "PDF" in abs_["note"]            # the wiki page disagrees with the PDF and the note says so
    pro = by["polymaker-abs-pro"]
    assert (pro["E_xy_mpa"], pro["E_z_mpa"]) == (2330, 1950) and "GPa" in pro["note"]          # the sheet prints GPa, the row is MPa
    cf = by["polymaker-fiberon-pa6-cf20"]
    assert (cf["E_xy_mpa"], cf["E_z_mpa"], cf["strength_xy_mpa"], cf["strength_z_mpa"]) == (8636.5, 3759.5, 109.3, 54.0)
    assert cf["hardened_nozzle"] is True and cf["abrasive"] is True and cf["nozzle_c"] == [280, 300] and cf["bed_c"] == [40, 50]
    p1s = by["bambu-lab-pla-cf"]
    assert (p1s["E_xy_mpa"], p1s["E_z_mpa"], p1s["hdt_045_c"]) == (2790, 2160, 55) and p1s["hardened_nozzle"] is True
    assert by["bambu-lab-pps-cf"]["chamber_c_min"] == 60 and by["bambu-lab-pps-cf"]["enclosure"] == "required"
    assert by["bambu-lab-tpu-95a-hf"]["soft"] is True and by["bambu-lab-tpu-95a-hf"]["E_xy_mpa"] == 9.8


def test_the_wet_values_of_the_polyamides_are_kept(by):
    wet = by["polymaker-polymide-copa"]["conditioned"]
    assert wet["strength_xy_mpa"] == 34.3 and wet["E_xy_mpa"] == 724 and "water at 60 C for 48 h" in wet["condition"]
    assert by["polymaker-fiberon-pa6-cf20"]["conditioned"]["strength_z_mpa"] == 25.5
    assert by["polymaker-fiberon-pa6-gf25"]["conditioned"]["E_xy_mpa"] == 1793.6
    assert by["bambu-lab-paht-cf"]["conditioned"] is None and "wet" in by["bambu-lab-paht-cf"]["note"]       # the sheet gives dry values only


def test_prusament_xz_is_not_a_z_value(by):
    """The 'vertical xz' bar stands on its edge: its load runs in the layer plane. So Z is null, and the note says why."""
    for fid in ("prusament-asa", "prusament-pc-blend", "prusament-pa11", "prusament-pp-glass-fiber", "prusament-petg", "prusament-pla"):
        m = by[fid]
        assert m["E_z_mpa"] is None and m["strength_z_mpa"] is None and m["strength_kind"] == "yield", fid
        assert "vertical xz" in m["note"] and "layer plane" in m["note"], fid
    assert by["prusament-asa"]["E_xy_mpa"] == 1600 and by["prusament-asa"]["strength_xy_mpa"] == 42


def test_a_value_that_two_sources_state_differently_or_that_has_a_wrong_order_is_null(by):
    pc = by["bambu-lab-pc"]
    assert pc["hdt_18_c"] is None and pc["hdt_045_c"] is None and "wrong order" in pc["note"]
    assert by["polymaker-polyflex-tpu95"]["E_xy_mpa"] is None and "does not say how the bar was printed" in by["polymaker-polyflex-tpu95"]["note"]
    assert by["azurefilm-petg"]["E_xy_mpa"] is None and by["azurefilm-petg"]["strength_xy_mpa"] is None
    assert by["elegoo-abs"]["E_xy_mpa"] is None and "bending modulus" in by["elegoo-abs"]["note"]     # a bending modulus is not a tensile modulus
    assert by["esun-abs-plus"]["E_xy_mpa"] is None and "injection-moulded" in by["esun-abs-plus"]["note"]
    assert by["prusament-petg-carbon-fiber"]["E_xy_mpa"] == 1700 and by["prusament-petg-carbon-fiber"]["hardened_nozzle"] is True


def test_one_product_with_two_sheet_versions_has_two_rows(by):
    new, old = by["bambu-lab-pla-basic"], by["bambu-lab-pla-basic-v2"]
    assert (new["sheet_version"], new["strength_xy_mpa"], new["E_xy_mpa"]) == ("V3.0", 35, 2580)
    assert (old["sheet_version"], old["strength_xy_mpa"], old["E_xy_mpa"]) == ("V2.0", 39, 2680)
    assert "bambu-lab-pla-basic-v2" in new["note"] and "bambu-lab-pla-basic" in old["note"]


def test_the_rows_of_the_material_reference_have_its_numbers(by):
    """`topopt_mcp/materials/reference.md` stays a reference document; its PLA and PETG sheets are rows of the catalog."""
    assert (by["polymaker-polylite-petg"]["strength_xy_mpa"], by["polymaker-polylite-petg"]["strength_z_mpa"]) == (50.8, 42.8)
    assert (by["polymaker-polylite-petg"]["E_xy_mpa"], by["polymaker-polylite-petg"]["E_z_mpa"]) == (2116.8, 1898.7)
    assert (by["polymaker-petg"]["strength_xy_mpa"], by["polymaker-petg"]["E_z_mpa"]) == (47.96, 2202.91)
    assert (by["bambu-lab-petg-basic"]["strength_xy_mpa"], by["bambu-lab-petg-basic"]["strength_z_mpa"]) == (48, 39)
    assert (by["polymaker-polylite-pla"]["strength_xy_mpa"], by["polymaker-polylite-pla"]["E_xy_mpa"]) == (52.3, 3426.9)
    assert (by["azurefilm-pla-prime"]["strength_xy_mpa"], by["azurefilm-pla-prime"]["strength_z_mpa"]) == (27.8, 14.2)
    assert by["azurefilm-pla-prime"]["strength_kind"] == "peak" and by["azurefilm-pla-prime"]["E_z_mpa"] == 1900
    assert (by["bambu-lab-pla-basic-v2"]["strength_xy_mpa"], by["bambu-lab-pla-basic-v2"]["strength_z_mpa"]) == (39, 35)
    reference = (ROOT / "topopt_mcp" / "materials" / "reference.md").read_text(encoding="utf-8")
    assert "Topopt does not read it" in reference or "read-only reference" in reference


# --- fix round 0 (L2): the print needs, the density and the heat deflection that the own sheet of a common filament states -----

SHEET_VALUES = {
    "polymaker-polylite-pla": dict(nozzle_c=[190, 230], bed_c=[25, 60], density_g_cm3=1.17, hdt_18_c=58, hdt_045_c=60, enclosure="no",
                                   drying="55 C, 6 h", elongation_xy_pct=6.3),
    "polymaker-petg": dict(nozzle_c=[240, 260], bed_c=[60, 70], density_g_cm3=1.3, hdt_18_c=65, hdt_045_c=69, enclosure="no",
                           drying="60 C, 6 h", elongation_xy_pct=9.33),
    "polymaker-polylite-petg": dict(hdt_18_c=75, hdt_045_c=78, elongation_xy_pct=8.4),
    "prusament-petg": dict(nozzle_c=[240, 260], bed_c=[70, 90], density_g_cm3=1.27, hdt_18_c=68, hdt_045_c=68, enclosure="no",
                           hardened_nozzle=False, elongation_xy_pct=5.1, tag="maker TDS"),
    "prusament-pla": dict(nozzle_c=[200, 220], bed_c=[40, 60], density_g_cm3=1.24, hdt_18_c=55, hdt_045_c=55, enclosure="no",
                          hardened_nozzle=False, elongation_xy_pct=2.9),
    "prusament-petg-carbon-fiber": dict(nozzle_c=[255, 275], bed_c=[80, 100], density_g_cm3=1.27, hdt_18_c=80, hdt_045_c=96,
                                        hardened_nozzle=True, E_xy_mpa=1700, strength_xy_mpa=47, strength_kind="yield", tag="maker TDS"),
    "azurefilm-petg": dict(nozzle_c=[220, 240], bed_c=[80, 90], density_g_cm3=1.29),
    "azurefilm-pla-prime": dict(density_g_cm3=1.27, elongation_xy_pct=4.3),
    "bambu-lab-petg-basic": dict(nozzle_c=[240, 270], bed_c=[65, 75], density_g_cm3=1.25, hdt_18_c=65, hdt_045_c=69, enclosure="recommended",
                                 hardened_nozzle=False, drying="65 C, 8 h", elongation_xy_pct=11.2),
    "bambu-lab-pla-basic-v2": dict(nozzle_c=[190, 230], bed_c=[35, 45], density_g_cm3=1.24, hdt_18_c=54, hdt_045_c=57, enclosure="no",
                                   hardened_nozzle=False, drying="55 C, 8 h", elongation_xy_pct=12.2),
    "bambu-lab-petg-cf": dict(enclosure="recommended"),
    "bambu-lab-pet-cf": dict(enclosure="no"),
    "bambu-lab-pa6-cf": dict(drying="80 C, 8-12 h"),
    "bambu-lab-pa6-gf": dict(drying="80 C, 8-12 h"),
    "bambu-lab-asa-cf": dict(drying="80 C, 8 h", chamber_c_min=45),
    "prusament-pc-blend-carbon-fiber": dict(enclosure="no"),
    "prusament-pp-carbon-fiber": dict(enclosure="no"),
    "prusament-tpu-95a": dict(drying="60 C, 4 h"),
}


@pytest.mark.parametrize("fid", sorted(SHEET_VALUES))
def test_a_common_filament_has_the_values_that_its_own_sheet_states(by, fid):
    for key, want in SHEET_VALUES[fid].items():
        assert by[fid][key] == want, (fid, key)


def test_no_row_lacks_the_print_temperatures_and_the_sources_of_a_filled_value_are_named(by):
    assert not [fid for fid, m in by.items() if m["nozzle_c"] is None or m["bed_c"] is None]
    # a print need that comes from another page than the sheet names that page
    for fid, field in (("prusament-petg", "enclosure"), ("prusament-pla", "hardened_nozzle"), ("bambu-lab-petg-basic", "enclosure"),
                       ("bambu-lab-pet-cf", "enclosure"), ("prusament-tpu-95a", "drying")):
        assert by[fid]["field_sources"][field].startswith("https://"), (fid, field)
    assert by["prusament-petg-carbon-fiber"]["source_url"].endswith("TDS_Prusament-PETG-CF-Black.zip")


def test_the_p1s_has_no_unknown_for_a_filament_whose_sheet_states_what_the_printer_needs_to_know(cat):
    """Before fix round 0 these eight were `unknown` on the P1S only because the rows left the print needs null."""
    p1s = printers.profile_for("bambu-lab-p1s")
    fit = mc.printer_fit(p1s, cat)
    unknown = {r["id"] for r in fit["groups"]["unknown"]}
    named = {"azurefilm-petg", "bambu-lab-petg-basic", "bambu-lab-pla-basic-v2", "polymaker-petg", "polymaker-polylite-pla",
             "prusament-petg", "prusament-petg-carbon-fiber", "prusament-pla"}
    assert not named & unknown, sorted(named & unknown)


# --- fix round 1 (MA-1): the maker's material table recommends an enclosed printer for PETG and PETG-CF ------------------------------

BAMBU_TABLE = "https://wiki.bambulab.com/en/general/filament-guide-material-table"


@pytest.mark.parametrize("fid", ["bambu-lab-petg-basic", "bambu-lab-petg-cf"])
def test_the_bambu_petg_rows_carry_the_enclosure_recommendation_of_the_maker_table(by, fid):
    m = by[fid]
    assert m["enclosure"] == "recommended"
    assert m["field_sources"]["enclosure"] == BAMBU_TABLE
    assert "an enclosed printer is recommended to prevent low ambient temperatures from reducing the model's interlayer strength" in " ".join(m["note"].split())


def test_the_bambu_rows_agree_with_the_enclosure_column_of_the_maker_table(by):
    """The table: ABS, ASA, PC, PA (and CF/GF) need an enclosed printer; PLA, TPU and PLA-CF take either; PETG and PETG-CF are better enclosed."""
    for fid, want in {"bambu-lab-abs": "required", "bambu-lab-asa": "required", "bambu-lab-pc": "required", "bambu-lab-pa6-cf": "required",
                      "bambu-lab-paht-cf": "required", "bambu-lab-pla-basic": "no", "bambu-lab-pla-cf": "no", "bambu-lab-tpu-95a-hf": "no",
                      "bambu-lab-petg-basic": "recommended", "bambu-lab-petg-cf": "recommended"}.items():
        assert by[fid]["enclosure"] == want, fid


# --- fix round 1 (MI-1): the chamber temperature of the spec boxes, and the rule of the field ---------------------------------------

@pytest.mark.parametrize("fid", ["elegoo-asa-cf", "elegoo-paht-cf"])
def test_the_elegoo_spec_boxes_that_name_a_chamber_of_45_to_60_give_45(by, fid):
    assert by[fid]["enclosure"] == "required" and by[fid]["chamber_c_min"] == 45


def test_chamber_c_min_follows_the_rule_that_the_header_states(raw, by):
    """A chamber temperature is entered for a filament that needs an enclosure (and for Prusament PA11, whose sheet gives the chamber as a
    need). A chamber that a sheet lists only among its print conditions of a filament with no enclosure need is not entered."""
    odd = {fid for fid, m in by.items() if m["chamber_c_min"] is not None and m["enclosure"] != "required"}
    assert odd == {"prusament-pa11"}
    for fid in ("bambu-lab-petg-basic", "bambu-lab-petg-cf", "bambu-lab-pet-cf"):
        assert by[fid]["chamber_c_min"] is None and "chamber_c_min" in by[fid]["note"], fid
    header = " ".join(CATALOG_FILE.read_text(encoding="utf-8").split("format: 1")[0].replace("#", " ").split())
    line = header[header.index("chamber_c_min"):header.index("hardened_nozzle true")]
    assert "enclosure: required" in line and "Prusament PA11" in line and "print conditions" in line, line


def test_the_p1s_gets_a_chamber_line_for_the_two_elegoo_rows_like_for_the_bambu_rows(pc_fc):
    pc, fc = pc_fc
    p1s = pc.by_id["bambu-lab-p1s"]
    for fid in ("elegoo-asa-cf", "elegoo-paht-cf", "bambu-lab-asa-cf", "bambu-lab-paht-cf"):
        assert any(r.check == "chamber" and r.state == "care" for r in compat.check(p1s, fc.by_id[fid]).reasons), fid


@pytest.fixture(scope="module")
def pc_fc():
    return printers.load_catalog(), mc.load_catalog()


# --- fix round 1 (MI-2): a sheet note about printing in an enclosed printer ------------------------------------------------------

PRUSAMENT_TPU_NOTE = ("If you print with a high flow nozzle in an enclosure, leave the enclosure door open. Opening the ventilation grill on "
                      "the Core One is sufficient.")


def test_only_the_rows_whose_sheet_states_a_note_about_an_enclosed_printer_have_one(by):
    """Not invented: the product page of Prusament TPU 95A says it. The general guides of the Bambu wiki (a rule that depends on the bed
    temperature, for the whole PLA, PETG and TPU families) are not the sheet of one product, so no Bambu row has a note."""
    assert {fid for fid, m in by.items() if m.get("enclosed_note")} == {"prusament-tpu-95a"}
    m = by["prusament-tpu-95a"]
    assert m["enclosed_note"] == PRUSAMENT_TPU_NOTE and m["field_sources"]["enclosed_note"] == "https://prusament.com/materials/prusament-tpu-95a/"
    assert "enclosure door is left open" not in " ".join(m["note"].split())      # the old sentence dropped the condition of the page


def test_an_enclosed_note_is_an_optional_text_that_a_user_entry_may_give_and_the_facts_keep(tmp_path):
    a = synthetic(row("a1", enclosed_note="Open the door in summer.")).profiles[0]
    assert a.enclosed_note == "Open the door in summer." and a.facts()["enclosed_note"] == "Open the door in summer."
    assert "enclosed_note" not in synthetic(row("a2")).profiles[0].facts()
    for bad in ("", "  ", 5, ["x"]):
        with pytest.raises(mc.CatalogError, match="enclosed_note must be a non-empty text"):
            synthetic(row("a1", enclosed_note=bad))
    assert mc.FilamentProfile.from_facts("a1", "t1", a.facts()).enclosed_note == a.enclosed_note
    assert mc.FilamentProfile.from_facts("a1", "t1", {"enclosed_note": 5}).enclosed_note is None
    # the printer check depends on it, so the digest sees it; a row without a note keeps its digest
    b = synthetic(row("a1", enclosed_note="Another text.")).profiles[0]
    plain = synthetic(row("a1")).profiles[0]
    assert len({a.facts_sha256(), b.facts_sha256(), plain.facts_sha256()}) == 3
    body = {"name": "Acme A1", **{k: plain.facts()[k] for k in mc.FACT_FIELDS}}
    assert plain.facts_sha256() == hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    saved = mc.save_user_filament(tmp_path, {"id": "acme-x", "maker": "Acme", "product": "X", "family": "PLA", "source": "user",
                                             "enclosed_note": "Open the door in summer."})
    assert saved["ok"], saved
    assert mc.catalog_for(tmp_path).by_id["acme-x"].enclosed_note == "Open the door in summer."


def test_the_names_of_the_rows_are_names_of_the_maker_and_the_explanation_is_in_the_note(by):
    """Fix round 1 (N-1): `PETG (newer product)` is not a name of the maker; the product is `PETG` and the note keeps the explanation."""
    assert by["polymaker-petg"]["product"] == "PETG" and "newer product" in " ".join(by["polymaker-petg"]["note"].split())
    assert not [fid for fid, m in by.items() if "(" in m["product"] and "newer" in m["product"]]


# --- fix round 0 (L3): the bending modulus of the sheet stands in for a tensile modulus that the sheet does not give -----------

ELEGOO_BENDING = {"elegoo-abs": (2484, 2133), "elegoo-asa": (2349, 2219), "elegoo-asa-cf": (3820, 1970), "elegoo-paht-cf": (5089, 2048),
                  "elegoo-pc": (2728, 2530), "elegoo-pet-cf": (5589, 2448), "elegoo-petg-cf": (3040, 2222), "elegoo-pla-cf": (3341, 2474),
                  "elegoo-pla-pro": (2964, 2419), "elegoo-pla-plus": (2979, 2582)}
REASON_BENDING = "the bending modulus of the sheet; the sheet gives no tensile modulus"


@pytest.mark.parametrize("fid, xy_z", sorted(ELEGOO_BENDING.items()))
def test_a_row_with_only_a_bending_modulus_takes_the_lower_of_its_two_values(by, cat, fid, xy_z):
    xy, z = xy_z
    assert (by[fid]["bending_modulus_xy_mpa"], by[fid]["bending_modulus_z_mpa"]) == (xy, z)
    assert by[fid]["E_xy_mpa"] is None and by[fid]["E_z_mpa"] is None
    value, reason = mc.modulus(cat.by_id[fid])
    assert value == min(xy, z) and reason.startswith(REASON_BENDING) and "lower of the XY and Z" in reason


def test_the_bending_modulus_is_only_entered_for_a_row_without_a_tensile_modulus(by):
    for fid, m in by.items():
        if m["bending_modulus_xy_mpa"] is not None or m["bending_modulus_z_mpa"] is not None:
            assert m["E_xy_mpa"] is None and m["E_z_mpa"] is None, fid
    # the rows that have neither: a TPU (hyperelastic), a sheet of injection-moulded bars, a sheet that gives no print direction
    for fid in ("elegoo-tpu-95a", "polymaker-polyflex-tpu95", "polymaker-polyflex-tpu95-hf", "prusament-tpu-95a", "esun-abs-plus", "azurefilm-petg"):
        assert by[fid]["bending_modulus_xy_mpa"] is None and by[fid]["bending_modulus_z_mpa"] is None, fid


def test_the_tensile_modulus_wins_and_one_bending_value_says_which_it_is():
    both = synthetic(row("a1", bending_modulus_xy_mpa=900, bending_modulus_z_mpa=800)).profiles[0]
    assert mc.modulus(both) == (1800, "the lower of the XY and Z modulus")
    xy_only = synthetic(row("a2", E_xy_mpa=None, E_z_mpa=None, bending_modulus_xy_mpa=2100,
                            note="E_xy_mpa, E_z_mpa: bending only; bending_modulus_z_mpa: none")).profiles[0]
    value, reason = mc.modulus(xy_only)
    assert value == 2100 and reason.startswith(REASON_BENDING) and "XY bending modulus" in reason and "no Z value" in reason
    assert "(" not in reason and ")" not in reason, reason             # fix round 1 (N-4): the reason is shown inside brackets, so it has none
    z_only = synthetic(row("a3", E_xy_mpa=None, E_z_mpa=None, bending_modulus_z_mpa=1900,
                           note="E_xy_mpa, E_z_mpa: bending only; bending_modulus_xy_mpa: none")).profiles[0]
    value, reason = mc.modulus(z_only)
    assert value == 1900 and reason.startswith(REASON_BENDING) and "Z bending modulus" in reason and "(" not in reason
    none = synthetic(row("a4", E_xy_mpa=None, E_z_mpa=None, note="E_xy_mpa, E_z_mpa: not stated")).profiles[0]
    assert mc.modulus(none) is None


def test_a_bending_modulus_is_a_number_above_0_and_a_user_entry_may_give_it(tmp_path):
    with pytest.raises(mc.CatalogError, match="bending_modulus_xy_mpa"):
        synthetic(row("a1", bending_modulus_xy_mpa=-5))
    r = mc.save_user_filament(tmp_path, user_entry("acme-bend", E_xy_mpa=None, E_z_mpa=None, bending_modulus_xy_mpa=2200, bending_modulus_z_mpa=2000))
    assert r["ok"], r
    assert r["profile"]["bending_modulus_xy_mpa"] == 2200 and "no tensile modulus" not in r["note"]
    saved = mc.catalog_for(tmp_path).by_id["acme-bend"]
    assert mc.modulus(saved)[0] == 2000
    assert mc.FilamentProfile.from_facts("acme-bend", "v", saved.facts()).bending_modulus_z_mpa == 2000


# --- fix round 0 (L4): a row with no density takes the median of its family, labelled assumed ------------------------------------

def test_the_family_density_is_the_median_of_the_catalog_rows_that_state_one(cat):
    value, reason = mc.family_density(cat.by_id["polymaker-polyflex-tpu95"], cat)
    assert value == 1.19 and reason == ("the median of 4 TPU rows of the catalog (bambu-lab-tpu-95a-hf, elegoo-tpu-95a, "
                                        "polymaker-polyflex-tpu95-hf, prusament-tpu-95a)")
    pla = sorted(p.density_g_cm3 for p in cat.profiles if p.family == "PLA" and p.density_g_cm3 is not None and not p.superseded_by)
    value, reason = mc.family_density(cat.by_id["polymaker-polymax-pla"], cat)
    mid = len(pla) // 2
    assert value == pytest.approx(pla[mid] if len(pla) % 2 else (pla[mid - 1] + pla[mid]) / 2)
    assert reason.startswith(f"the median of {len(pla)} PLA rows of the catalog (") and "polymaker-polymax-pla" not in reason


def test_the_family_density_counts_current_rows_only_and_compares_the_family_without_case(cat):
    """Fix round 1 (N-2): an older sheet of a listed product is not a row of the median (the lists leave it out), and the family `petg`
    of a user filament is the family PETG, as the resolver reads it."""
    value, reason = mc.family_density(cat.by_id["polymaker-polymax-pla"], cat)
    assert "bambu-lab-pla-basic-v2" not in reason and "bambu-lab-pla-basic," in reason
    current = [p for p in cat.profiles if p.family == "PLA" and p.density_g_cm3 is not None and not p.superseded_by]
    assert reason.startswith(f"the median of {len(current)} PLA rows")
    petg = mc.family_density(mc.FilamentProfile(id="u2", maker="U", product="Two", family="PETG", origin="user"), cat)
    low = mc.family_density(mc.FilamentProfile(id="u2", maker="U", product="Two", family="petg", origin="user"), cat)
    assert petg is not None and low[0] == petg[0] and low[1] == petg[1].replace("PETG rows", "petg rows")
    assert mc.family_density(mc.FilamentProfile(id="u2", maker="U", product="Two", family="Pet g", origin="user"), cat) is None


def test_the_family_density_needs_a_row_of_the_same_family_that_states_one_and_leaves_the_user_rows_out():
    one = synthetic(row("a1", family="XX", density_g_cm3=None, note="density_g_cm3: not stated"), row("a2", family="XX", density_g_cm3=1.4))
    assert mc.family_density(one.by_id["a1"], one) == (1.4, "the median of 1 XX row of the catalog (a2)")
    alone = synthetic(row("a1", family="XX", density_g_cm3=None, note="density_g_cm3: not stated"), row("a3", family="YY", density_g_cm3=1.4))
    assert mc.family_density(alone.by_id["a1"], alone) is None
    assert mc.family_density(alone.by_id["a3"], alone) is None or mc.family_density(alone.by_id["a3"], alone)[0] != 1.4
    user = mc.FilamentProfile(id="u1", maker="U", product="One", family="PLA", density_g_cm3=1.9, origin="user")
    mixed = mc.Catalog("t", (*mc.load_catalog().profiles, user), mc.load_catalog().makers)
    assert "u1" not in mc.family_density(mixed.by_id["polymaker-polymax-pla"], mixed)[1]


def with_bare_pla(cat):
    """The shipped catalog and a user filament of the family PLA whose sheet gives neither a density nor a range."""
    bare = mc.FilamentProfile(id="polymaker-nodensity", maker="Polymaker", product="NoDensity PLA", family="PLA", E_xy_mpa=2000.0, origin="user")
    return bare, mc.Catalog(cat.version, (*cat.profiles, bare), cat.makers)


def test_derives_and_the_match_note_show_the_family_density_as_assumed(cat):
    bare, both = with_bare_pla(cat)
    plain = mc.derives(bare)
    assert plain["missing"] == ["material.density"] and plain["assumed"] == {"material.nu": 0.35}          # without a catalog: as before
    d = mc.derives(bare, both)
    assert d["missing"] == ["material.density"] and d["assumed"]["material.nu"] == 0.35
    assert d["assumed"]["material.density"] == mc.family_density(bare, both)[0]
    found = mc.resolve("polymaker nodensity pla", both).to_dict()
    assert found["derives"]["assumed"]["material.density"] == d["assumed"]["material.density"]
    assert "the median of" in found["note"] and "labelled assumed" in found["note"]


# --- fix round 1 (N-3): a sheet that gives a density range gives the middle of it, before the family median -----------------------

RANGES = {"polymaker-polymax-pla": ([1.17, 1.24], 1.205, "1.17-1.24"), "polymaker-polyflex-tpu95": ([1.2, 1.24], 1.22, "1.2-1.24")}


@pytest.mark.parametrize("fid", sorted(RANGES))
def test_a_row_whose_sheet_gives_a_density_range_holds_the_range_and_a_run_takes_the_middle(by, cat, fid):
    rng, mid, text = RANGES[fid]
    assert by[fid]["density_g_cm3"] is None and by[fid]["density_range_g_cm3"] == rng and "density_g_cm3" in by[fid]["note"]
    p = cat.by_id[fid]
    assert mc.sheet_density(p) == (mid, f"the middle of the sheet range {text}")
    d = mc.derives(p, cat)
    assert d["sets"]["material.density"] == mid and "material.density" not in d["assumed"] and "material.density" not in d["missing"]
    note = mc.resolve(fid, cat).to_dict()["note"]
    assert f"density {mid:g} g/cm3 (the middle of the sheet range {text})" in note and "the median of" not in note


def test_the_family_median_is_for_a_sheet_that_gives_no_density_at_all(by):
    assert {fid for fid, m in by.items() if m.get("density_range_g_cm3")} == set(RANGES)
    assert not [fid for fid, m in by.items() if m["density_g_cm3"] is None and not m.get("density_range_g_cm3")]


def test_sheet_density_and_the_range_field_in_the_loader_the_user_file_the_facts_and_the_digest(tmp_path):
    single = synthetic(row("a1")).profiles[0]
    assert mc.sheet_density(single) == (1.2, "the density of the sheet")
    ranged = synthetic(row("a2", density_g_cm3=None, density_range_g_cm3=[1.1, 1.3], note="density_g_cm3: a range")).profiles[0]
    assert ranged.density_range_g_cm3 == (1.1, 1.3) and mc.sheet_density(ranged) == (1.2, "the middle of the sheet range 1.1-1.3")
    assert mc.sheet_density(synthetic(row("a3", density_g_cm3=None, note="density_g_cm3: none")).profiles[0]) is None
    with pytest.raises(mc.CatalogError, match="density_g_cm3 must be null"):
        synthetic(row("a4", density_range_g_cm3=[1.1, 1.3]))
    for bad in ([1.3, 1.1], [1.1], "1.2", [0, 1.3], [1.1, True]):
        with pytest.raises(mc.CatalogError, match="density_range_g_cm3 must be"):
            synthetic(row("a5", density_g_cm3=None, density_range_g_cm3=bad, note="density_g_cm3: a range"))
    assert ranged.facts()["density_range_g_cm3"] == [1.1, 1.3] and "density_range_g_cm3" not in single.facts()
    assert mc.FilamentProfile.from_facts("a2", "t1", ranged.facts()).density_range_g_cm3 == (1.1, 1.3)
    other = synthetic(row("a2", density_g_cm3=None, density_range_g_cm3=[1.1, 1.4], note="density_g_cm3: a range")).profiles[0]
    assert ranged.facts_sha256() != other.facts_sha256()
    saved = mc.save_user_filament(tmp_path, {"id": "acme-r", "maker": "Acme", "product": "R", "family": "PLA", "source": "user",
                                            "E_xy_mpa": 2000, "density_range_g_cm3": [1.1, 1.3]})
    assert saved["ok"] and "density_g_cm3" not in saved["note"].split("Not known")[-1], saved["note"]
    assert mc.catalog_for(tmp_path).by_id["acme-r"].density_range_g_cm3 == (1.1, 1.3)


# --- fix round 0 (L5): the older sheet of a product is marked, and the lists show the current row ----------------------------------

def test_the_older_sheet_of_a_product_names_the_newer_row(by, cat):
    assert by["bambu-lab-pla-basic-v2"]["superseded_by"] == "bambu-lab-pla-basic"
    assert [fid for fid, m in by.items() if "superseded_by" in m] == ["bambu-lab-pla-basic-v2"]
    assert cat.by_id["bambu-lab-pla-basic-v2"].superseded_by == "bambu-lab-pla-basic" and cat.by_id["bambu-lab-pla-basic"].superseded_by is None
    assert cat.by_id["bambu-lab-pla-basic-v2"].facts()["superseded_by"] == "bambu-lab-pla-basic"
    assert "superseded_by" not in cat.by_id["bambu-lab-pla-basic"].facts()
    back = mc.FilamentProfile.from_facts("bambu-lab-pla-basic-v2", "v", cat.by_id["bambu-lab-pla-basic-v2"].facts())
    assert back.superseded_by == "bambu-lab-pla-basic"
    # the digest is of the numbers: marking a row as older changes none
    p = cat.by_id["bambu-lab-pla-basic-v2"]
    assert replace(p, superseded_by=None).facts_sha256() == p.facts_sha256()


@pytest.mark.parametrize("rows, text", [
    ([row("a1", superseded_by="nope")], "'nope' is not a filament of the catalog"),
    ([row("a1", superseded_by="a1")], "cannot be older than itself"),
    ([row("a1", superseded_by=5)], "superseded_by must be the id"),
    ([row("a1", superseded_by="a2"), row("a2", superseded_by="a3"), row("a3")], "is itself superseded"),
])
def test_a_bad_superseded_by_is_a_named_error(rows, text):
    with pytest.raises(mc.CatalogError, match=text):
        synthetic(*rows)


def test_a_user_entry_cannot_say_that_it_is_superseded(tmp_path):
    r = mc.save_user_filament(tmp_path, user_entry("acme-old", superseded_by="polymaker-polylite-abs"))
    assert r["ok"] is False and "superseded_by" in r["error"]


def test_the_lists_show_the_current_row_and_an_exact_id_still_selects_the_old_row(cat):
    old = mc.resolve("bambu-lab-pla-basic-v2")
    assert old.status == "match" and old.profile.id == "bambu-lab-pla-basic-v2" and old.matched_by == "id"
    assert "newer sheet" in old.note and "bambu-lab-pla-basic" in old.note
    assert mc.resolve("bambu lab pla basic").profile.id == "bambu-lab-pla-basic"                 # the product name is the current row
    by_alias = mc.resolve("bambu lab pla basic 2.0")                                              # the name of the old sheet is the old row
    assert by_alias.profile.id == "bambu-lab-pla-basic-v2" and "newer sheet" in by_alias.note
    for text, how in (("PLA", "family"), ("bambu", "vendor"), ("basic", "partial")):
        r = mc.resolve(text)
        ids = {c.id for c in r.candidates}
        assert r.status == "ambiguous" and r.matched_by == how
        assert "bambu-lab-pla-basic-v2" not in ids and "bambu-lab-pla-basic" in ids, text
        assert "older sheet" in r.note.lower() and "bambu-lab-pla-basic-v2" in r.note, text
    assert len(mc.resolve("PLA").candidates) == len([p for p in cat.profiles if p.family == "PLA"]) - 1


def test_printer_materials_lists_the_current_rows_and_counts_the_older_ones(cat):
    fit = mc.printer_fit(printers.profile_for("bambu-lab-p1s"), cat, available=["bambu-lab-pla-basic-v2", "bambu-lab-pla-basic"])
    listed = [r["id"] for g in fit["groups"].values() for r in g]
    assert "bambu-lab-pla-basic-v2" not in listed and "bambu-lab-pla-basic" in listed
    assert sum(fit["counts"].values()) == len(cat.profiles) - 1 == len(listed)
    assert "1 filament with an older sheet" in fit["note"] and "bambu-lab-pla-basic-v2" in fit["note"]
    old, new = fit["available"]
    assert old["id"] == "bambu-lab-pla-basic-v2" and old["state"] in mc.GROUPS and "newer sheet" in old["note"] and "bambu-lab-pla-basic" in old["note"]
    assert new["id"] == "bambu-lab-pla-basic" and "note" not in new
    plain = mc.printer_fit(printers.profile_for("bambu-lab-p1s"), synthetic(row("a1"), row("a2")))
    assert "older sheet" not in plain["note"]


def test_the_catalog_loads_once_and_the_profiles_are_frozen(cat):
    assert mc.load_catalog() is cat
    p = cat.by_id["bambu-lab-abs"]
    assert p.name == "Bambu Lab ABS" and p.catalog_version == "2026-10-25" and p.origin == "catalog"
    with pytest.raises(Exception):
        p.E_xy_mpa = 1.0


def test_the_unquoted_no_of_yaml_is_a_boolean_and_the_loader_reads_it_as_no():
    assert yaml.safe_load("enclosure: no") == {"enclosure": False}
    c = synthetic(row("a1", enclosure=False))
    assert c.profiles[0].enclosure == "no"
    with pytest.raises(mc.CatalogError, match="enclosure must be one of"):
        synthetic(row("a1", enclosure=True))


# --- the checks of the format --------------------------------------------------------------------------------------------------

def test_a_valid_synthetic_catalog_parses():
    c = synthetic(row("a1"), row("b2", maker="Zeta"))
    assert [p.id for p in c.profiles] == ["a1", "b2"] and c.version == "t1" and c.makers == {"Acme": ("acme",), "Zeta": ("zeta",)}


@pytest.mark.parametrize("change, text", [
    ({"id": "Bad Id"}, "lower-case slug"),
    ({"E_xy_mpa": -1}, "number above 0"),
    ({"E_z_mpa": "2e3"}, "number above 0"),
    ({"E_z_mpa": True}, "number above 0"),
    ({"nozzle_c": [230]}, "must be [min, max]"),
    ({"nozzle_c": [260, 200]}, "min not above max"),
    ({"bed_c": [0, 60]}, "number above 0"),
    ({"enclosure": "boxed"}, "enclosure must be one of"),
    ({"strength_kind": "ultimate"}, "strength_kind must be one of"),
    ({"strength_kind": "yield", "strength_xy_mpa": None, "strength_z_mpa": None,
      "note": "strength_xy_mpa, strength_z_mpa: none"}, "has no strength"),
    ({"hardened_nozzle": "yes"}, "hardened_nozzle must be true, false or null"),
    ({"abrasive": 1}, "abrasive must be true, false or null"),
    ({"nu": 0.5}, "nu must be a number above -1 and below 0.5"),
    ({"opened": "yesterday"}, "opened must be a date"),
    ({"source_url": "pages/a1"}, "http(s) address"),
    ({"tag": "rumour"}, "tag must be one of"),
    ({"maker": "Nobody"}, "not in the `makers` table"),
    ({"unknown_key": 1}, "unknown key 'unknown_key'"),
    ({"conditioned": {"condition": "wet"}}, "no wet value"),
    ({"conditioned": {"E_xy_mpa": 5}}, "conditioned.condition must be a non-empty text"),
    ({"conditioned": {"condition": "wet", "colour": 1}}, "unknown key 'colour'"),
    ({"conditioned": [1]}, "conditioned must be a mapping"),
    ({"drying": 5}, "drying must be a non-empty text"),
    ({"E_xy_mpa": None}, "needs a note that names E_xy_mpa"),
    ({"E_xy_mpa": None, "note": "the sheet does not give a modulus"}, "needs a note that names E_xy_mpa"),
    ({"field_sources": {"colour": "https://example.com"}}, "not a fact field"),
    ({"aliases": "x"}, "aliases must be a list"),
])
def test_a_fault_in_a_row_is_a_named_error(change, text):
    with pytest.raises(mc.CatalogError, match=re.escape(text)):
        synthetic(row("a1", **change))


def test_a_missing_key_a_double_id_and_a_double_name_are_errors():
    gone = row("a1")
    del gone["soft"]
    with pytest.raises(mc.CatalogError, match="missing key 'soft'"):
        synthetic(gone)
    with pytest.raises(mc.CatalogError, match="used twice"):
        synthetic(row("a1"), row("a1", product="Other"))
    with pytest.raises(mc.CatalogError, match="is also a name of"):
        synthetic(row("a1", product="Box 1"), row("a2", product="box-1"))
    ok = synthetic(row("a1", product="Box 1"), row("z1", maker="Zeta", product="Box 1"))     # the same name under two makers is allowed
    assert len(ok.profiles) == 2


def test_a_null_with_a_note_that_names_it_is_accepted_and_the_plain_nulls_need_none():
    c = synthetic(row("a1", E_xy_mpa=None, note="E_xy_mpa: the sheet gives a bending modulus only"))
    assert c.profiles[0].E_xy_mpa is None
    plain = synthetic(row("a2", nu=None, chamber_c_min=None, hardened_nozzle=None, drying=None, direct_drive=None, conditioned=None))
    assert plain.profiles[0].hardened_nozzle is None


def test_a_broken_file_is_a_catalog_error(tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text("format: 2\ncatalog_version: x\nmakers: {A: [a]}\nmaterials: []\n")
    with pytest.raises(mc.CatalogError, match="format is 2"):
        mc.load_catalog(f)
    f.write_text("a: [1\n")
    with pytest.raises(mc.CatalogError, match="not valid YAML"):
        mc.load_catalog(f)
    with pytest.raises(mc.CatalogError, match="cannot read"):
        mc.load_catalog(tmp_path / "missing.yaml")
    f.write_text("format: 1\ncatalog_version: 2026\nmakers: {A: [a]}\nmaterials: [1]\n")
    with pytest.raises(mc.CatalogError, match="catalog_version must be a text"):
        mc.load_catalog(f)
    f.write_text("format: 1\ncatalog_version: x\nmakers: {A: [a]}\nmaterials: []\n")
    with pytest.raises(mc.CatalogError, match="non-empty list"):
        mc.load_catalog(f)


def test_a_number_written_as_1e9_gets_the_hint():
    with pytest.raises(mc.CatalogError, match="1e9"):
        synthetic(row("a1", E_xy_mpa="2e3"))


# --- the numbers that a run takes ----------------------------------------------------------------------------------------------

def test_the_modulus_is_the_lower_of_xy_and_z_with_a_reason(cat):
    value, reason = mc.modulus(cat.by_id["polymaker-polylite-abs"])
    assert value == 2080.9 and reason == "the lower of the XY and Z modulus"
    value, reason = mc.modulus(cat.by_id["bambu-lab-pla-tough-plus"])          # the Z modulus is the higher one here
    assert value == 1860 and reason == "the lower of the XY and Z modulus"
    value, reason = mc.modulus(cat.by_id["prusament-asa"])                    # no Z modulus on the sheet
    assert value == 1600 and "XY modulus" in reason and "no Z modulus" in reason
    assert mc.modulus(cat.by_id["esun-abs-plus"]) is None and mc.modulus(cat.by_id["polymaker-polyflex-tpu95"]) is None
    only_z = synthetic(row("a1", E_xy_mpa=None, note="E_xy_mpa: not stated")).profiles[0]
    assert mc.modulus(only_z)[0] == 1800 and "Z modulus" in mc.modulus(only_z)[1]


def test_derives_names_what_is_set_assumed_and_missing(cat):
    d = mc.derives(cat.by_id["polymaker-polylite-abs"])
    assert d == {"sets": {"material.E": 2080.9, "material.density": 1.04}, "assumed": {"material.nu": 0.35}, "missing": []}
    d = mc.derives(cat.by_id["esun-abs-plus"])                  # no tensile and no bending modulus
    assert d["missing"] == ["material.E"] and d["sets"] == {"material.density": 1.06} and d["assumed"] == {"material.nu": 0.35}
    d = mc.derives(cat.by_id["elegoo-abs"])                     # the bending modulus of the sheet stands in for the tensile one
    assert d["missing"] == [] and d["sets"] == {"material.E": 2133, "material.density": 1.05}
    d = mc.derives(cat.by_id["polymaker-polymax-pla"])         # the sheet gives a density range: a run takes the middle (fix round 1)
    assert d["missing"] == [] and d["sets"] == {"material.E": 1983.6, "material.density": 1.205}
    assert mc.derives(synthetic(row("a1", density_g_cm3=None, note="density_g_cm3: none")).profiles[0])["missing"] == ["material.density"]
    with_nu = synthetic(row("a1", nu=0.4)).profiles[0]
    assert mc.derives(with_nu)["sets"]["material.nu"] == 0.4 and mc.derives(with_nu)["assumed"] == {}


def test_the_facts_are_json_and_the_digest_ignores_the_sources(cat):
    p = cat.by_id["polymaker-polymide-copa"]
    facts = p.facts()
    assert json.loads(json.dumps(facts)) == facts
    assert facts["name"] == "Polymaker PolyMide CoPA" and facts["origin"] == "catalog" and facts["conditioned"]["E_xy_mpa"] == 724
    assert facts["nozzle_c"] == [250.0, 270.0] and facts["E_xy_mpa"] == 2703.0 and "source_url" in facts
    a = synthetic(row("a1")).profiles[0]
    b = synthetic(row("a1", source_url="https://example.com/other", opened="2026-01-01", note="another note", tag="third party")).profiles[0]
    c = synthetic(row("a1", E_xy_mpa=2001)).profiles[0]
    assert a.facts_sha256() == b.facts_sha256() != c.facts_sha256()
    body = {"name": "Acme A1", **{k: a.facts()[k] for k in mc.FACT_FIELDS}}
    assert a.facts_sha256() == hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert a.summary()["id"] == "a1" and a.summary()["catalog_version"] == "t1"


def test_a_profile_comes_back_from_its_facts_and_a_bad_record_reads_as_not_stated(cat):
    for fid in ("polymaker-polymide-copa", "prusament-asa", "bambu-lab-pla-tough-plus", "elegoo-abs", "azurefilm-pla-prime"):
        p = cat.by_id[fid]
        back = mc.FilamentProfile.from_facts(p.id, p.catalog_version, p.facts())
        assert back == p, fid
        assert back.facts_sha256() == p.facts_sha256()
    assert mc.FilamentProfile.from_facts("", "v", {}) is None and mc.FilamentProfile.from_facts("a", "v", "x") is None
    bad = mc.FilamentProfile.from_facts("a1", "v", {"E_xy_mpa": -4, "E_z_mpa": "x", "nozzle_c": [5, 1], "enclosure": "boxed",
                                                    "abrasive": "yes", "nu": 7, "conditioned": {"condition": "wet", "E_xy_mpa": 5}})
    assert (bad.E_xy_mpa, bad.E_z_mpa, bad.nozzle_c, bad.enclosure, bad.abrasive, bad.nu) == (None, None, None, None, None, None)
    assert bad.wet["E_xy_mpa"] == 5.0 and bad.origin == "catalog" and bad.product == "a1"


# --- resolve: match -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Bambu ABS", "bambu lab abs", "BAMBU-LAB  ABS!", "bambulab_abs", "bbl abs", "bambu-lab-abs", "  Bambu Lab ABS  ",
                                  "my Bambu ABS"])
def test_one_filament_by_id_name_case_punctuation_and_maker_words(text):
    r = mc.resolve(text)
    assert r.status == "match" and r.profile.id == "bambu-lab-abs" and not r.candidates, text


def test_how_it_matched_is_named():
    assert mc.resolve("bambu-lab-abs").matched_by == "id"
    assert mc.resolve("Bambu ABS").matched_by == "name"
    assert mc.resolve("bambu pla basic 2.0").matched_by == "alias"
    assert mc.resolve("polymaker polylite pla pro").profile.id == "polymaker-polylite-pla-pro"


@pytest.mark.parametrize("text, fid", [
    ("Polymaker PolyLite ABS", "polymaker-polylite-abs"), ("polylite abs", "polymaker-polylite-abs"), ("Polymaker ABS Pro", "polymaker-abs-pro"),
    ("Prusament PETG", "prusament-petg"), ("prusa petg", "prusament-petg"), ("Prusament PC Blend", "prusament-pc-blend"),
    ("Prusament PC Blend Carbon Fiber", "prusament-pc-blend-carbon-fiber"), ("Elegoo PLA Plus", "elegoo-pla-plus"),
    ("Elegoo PLA+", "elegoo-pla-plus"), ("eSun ABS+", "esun-abs-plus"), ("e-sun abs plus", "esun-abs-plus"),
    ("Bambu PLA Tough+", "bambu-lab-pla-tough-plus"), ("Bambu PLA Basic", "bambu-lab-pla-basic"),
    ("Bambu PLA Basic 2.0", "bambu-lab-pla-basic-v2"), ("Fiberon PA6-CF20", "polymaker-fiberon-pa6-cf20"),
    ("bambu pa6-cf", "bambu-lab-pa6-cf"), ("azurefilm pla prime", "azurefilm-pla-prime"), ("Polymaker PolyMax PC", "polymaker-polymax-pc")])
def test_names_and_aliases_of_the_catalog(text, fid):
    r = mc.resolve(text)
    assert r.status == "match" and r.profile.id == fid, (text, r.status, [p.id for p in r.candidates])


def test_the_maker_word_is_a_hard_filter():
    assert mc.resolve("elegoo abs").profile.id == "elegoo-abs"
    assert mc.resolve("prusament abs").status == "none"                  # Prusament has no ABS
    both = mc.resolve("abs")
    assert both.status == "ambiguous"                                    # no maker word: a family
    r = synthetic(row("a1", product="X1"), row("z1", maker="Zeta", product="X1"))
    two = mc.resolve("x1", r)
    assert two.status == "ambiguous" and two.matched_by == "name" and [p.id for p in two.candidates] == ["a1", "z1"]
    assert mc.resolve("zeta x1", r).profile.id == "z1"


# --- resolve: ambiguous ---------------------------------------------------------------------------------------------------------

def test_a_family_name_alone_is_ambiguous_with_every_filament_of_the_family_and_never_a_silent_match():
    """`PLA` is the exact product name of one maker (Prusament PLA). It still names a family, so the user is asked."""
    r = mc.resolve("PLA")
    assert r.status == "ambiguous" and r.matched_by == "family" and r.profile is None
    assert len(r.candidates) == 9 and {p.family for p in r.candidates} == {"PLA"}        # the 10th PLA row is an older sheet: it is counted in the note
    assert "material family" in r.note and "Ask the user which one" in r.note and "E_xy_mpa" in r.differs
    assert r.to_dict()["origin"] is None and r.to_dict()["candidates"][0]["id"] == r.candidates[0].id
    for text in ("petg", "my PETG", "PLA-CF", "pla cf", "PA6-CF", "TPU", "ASA"):
        assert mc.resolve(text).status == "ambiguous" and mc.resolve(text).matched_by == "family", text
    assert mc.resolve("PC-ABS").status == "match"                       # a family with one filament is a name
    assert mc.resolve("CoPA").status == "ambiguous"                      # not the whole name "PolyMide CoPA": a part of it
    assert mc.resolve("Prusament PLA").profile.id == "prusament-pla"     # a maker word makes it a name


def test_the_candidates_are_the_filaments_whose_names_hold_all_the_words():
    r = mc.resolve("polymaker pla")
    assert r.status == "ambiguous" and r.matched_by == "partial"
    assert {p.id for p in r.candidates} == {"polymaker-polylite-pla", "polymaker-polylite-pla-pro", "polymaker-polymax-pla",
                                            "polymaker-polylite-pla-cf"}
    assert {p.maker for p in mc.resolve("elegoo").candidates} == {"Elegoo"} and len(mc.resolve("elegoo").candidates) == 11
    assert mc.resolve("elegoo").matched_by == "vendor"
    r = mc.resolve("polymaker petg")                 # fix round 1 (N-1): the product is called PETG, so this is its exact name
    assert r.status == "match" and r.profile.id == "polymaker-petg" and r.profile.name == "Polymaker PETG"
    assert mc.resolve("polymaker polylite petg").profile.id == "polymaker-polylite-petg"
    assert {p.id for p in mc.resolve("polymaker tpu95").candidates} == {"polymaker-polyflex-tpu95", "polymaker-polyflex-tpu95-hf"}


def test_a_part_of_one_name_is_ambiguous_with_one_candidate_and_never_a_silent_match():
    r = mc.resolve("copa")
    assert r.status == "ambiguous" and [p.id for p in r.candidates] == ["polymaker-polymide-copa"] and r.profile is None
    assert "only part of the name" in r.note and "Polymaker PolyMide CoPA" in r.note


def test_the_ambiguous_note_names_the_facts_on_which_the_candidates_differ():
    c = synthetic(row("a1", product="Box 1 Pro"), row("a2", product="Box 1 Max", E_z_mpa=1500, enclosure="required"))
    r = mc.resolve("acme box 1", c)
    assert r.status == "ambiguous" and set(r.differs) == {"E_z_mpa", "enclosure"} and "E_z_mpa" in r.note


# --- resolve: none and suggestions ----------------------------------------------------------------------------------------------

def test_no_match_gives_suggestions_and_never_takes_one():
    r = mc.resolve("Bambu ABZ")
    assert r.status == "none" and r.profile is None and not r.candidates and r.suggestions
    assert r.suggestions[0][0].id in ("bambu-lab-abs", "bambu-lab-pc")
    assert "never taken by itself" in r.note and "save_material" in r.note and "data sheet" in r.note
    assert "E_xy_mpa" in r.note and "density_g_cm3" in r.note


def test_a_name_that_is_far_from_all_gives_no_suggestion_and_the_makers():
    r = mc.resolve("qwertyuiop")
    assert r.status == "none" and r.suggestions == () and r.makers == tuple(mc.load_catalog().makers)
    for text in ("", "   ", "my", None, 5):
        assert mc.resolve(text).status == "none", text


def test_the_suggest_settings_are_the_ones_of_the_printer_catalog(monkeypatch):
    """One pair of knobs for both catalogs (module constants, not YAML keys)."""
    assert printers.SUGGEST_CUTOFF == 0.6 and printers.SUGGEST_MAX == 5
    src = (ROOT / "topopt" / "materials_catalog.py").read_text(encoding="utf-8")
    assert not re.search(r"^SUGGEST_", src, re.M)
    monkeypatch.setattr(printers, "SUGGEST_MAX", 1)
    assert len(mc.resolve("Bambu ABZ").suggestions) == 1


def test_the_name_matching_is_the_code_of_the_printer_catalog():
    assert issubclass(mc.Catalog, catalogs.NameIndex) and issubclass(printers.Catalog, catalogs.NameIndex)
    assert mc.CatalogError is printers.CatalogError is catalogs.CatalogError
    src = (ROOT / "topopt" / "materials_catalog.py").read_text(encoding="utf-8")
    assert "catalogs.find(" in src and "catalogs.suggest_names(" in src and "import difflib" not in src


# --- the texts of a refusal -----------------------------------------------------------------------------------------------------

def test_a_refusal_names_the_key_the_candidates_or_suggestions_and_the_ways_on():
    text = mc.refusal_text(mc.resolve("polymaker pla"))
    assert text.startswith("'material.filament' 'polymaker pla' fits 4 filaments of the catalog (version 2026-10-25): ")
    assert "polymaker-polylite-pla (Polymaker PolyLite PLA V5.4)" in text and mc.WAYS_ON in text and "\n" not in text
    text = mc.refusal_text(mc.resolve("Bambu ABZ"))
    assert "is not in the filament catalog" in text and "Did you mean: bambu-lab-" in text and "save_material" in text
    text = mc.refusal_text(mc.resolve("qwertyuiop"), key="material.filament")
    assert "no name is close to it" in text and mc.WAYS_ON in text and mc.ADD_ON in text
    assert "(Elegoo ABS)" in mc.refusal_text(mc.resolve("elegoo abs x")) or "Elegoo ABS" in mc.refusal_text(mc.resolve("elegoo ab"))


def test_the_match_note_says_what_is_set_and_that_nu_stays_assumed():
    note = mc.resolve("polymaker polylite abs").note
    assert "E 2080.9 MPa (the lower of the XY and Z modulus)" in note and "density 1.04 g/cm3" in note
    assert "material.nu stays assumed (no data sheet gives it)" in note and "catalog 2026-10-25" in note
    lack = mc.resolve("esun abs plus").note
    assert "does not state material.E" in lack and "E not stated" in lack
    bending = mc.resolve("elegoo abs").note
    assert "E 2133 MPa (the bending modulus of the sheet; the sheet gives no tensile modulus" in bending


def test_the_modulus_error_text_names_the_filament_and_the_way_on(cat):
    text = mc.no_modulus_text(cat.by_id["esun-abs-plus"])
    assert "has no tensile modulus (E_xy_mpa, E_z_mpa) and no bending modulus" in text and "Give E in the material block" in text


def test_the_modulus_error_names_the_null_fields_and_does_not_copy_the_note_of_the_row(cat):
    """Fix round 1 (N-4): the note of a row can be long (azurefilm-petg); the error names the fields and points to `find_material`."""
    for fid in ("azurefilm-petg", "esun-abs-plus", "polymaker-polyflex-tpu95"):
        p = cat.by_id[fid]
        text = mc.no_modulus_text(p)
        assert p.note[100:160] not in text and len(text) < 400 < len(p.note), (fid, len(text), len(p.note))
        assert "E_xy_mpa, E_z_mpa" in text and "bending_modulus_xy_mpa, bending_modulus_z_mpa" in text, text
        assert f"find_material" in text and p.id in text and "Give E in the material block" in text
    only_xy = synthetic(row("a1", E_xy_mpa=None, E_z_mpa=None, note="E_xy_mpa, E_z_mpa: none")).profiles[0]
    assert "has no tensile modulus" in mc.no_modulus_text(only_xy) and "{" not in mc.no_modulus_text(only_xy)


# --- the user filament file -----------------------------------------------------------------------------------------------------

def user_entry(fid="acme-pla", **over):
    e = {"id": fid, "maker": "Acme", "product": "Super PLA", "family": "PLA", "source": "user", "E_xy_mpa": 2500, "density_g_cm3": 1.25}
    e.update(over)
    return e


def test_no_user_file_is_no_problem(tmp_path):
    c = mc.catalog_for(tmp_path)
    assert c.user_file.exists is False and c.user_file.problems == () and len(c.profiles) == len(mc.load_catalog().profiles)
    assert mc.catalog_for(None).user_file is None


def test_save_creates_the_file_and_the_filament_is_live(tmp_path):
    r = mc.save_user_filament(tmp_path, user_entry(enclosure=False, nozzle_c=[200, 220]))
    assert r["ok"] and r["created"] and not r["replaced"] and "new file" in r["note"] and "live now" in r["note"]
    assert r["profile"]["origin"] == "user" and r["profile"]["enclosure"] == "no" and r["profile"]["E_xy_mpa"] == 2500.0
    assert "Not known (null): E_z_mpa" in r["note"]
    c = mc.catalog_for(tmp_path)
    assert c.user_file.entries == 1 and c.user_file.problems == ()
    hit = mc.resolve("acme super pla", c)
    assert hit.status == "match" and hit.profile.origin == "user" and hit.profile.origin_text == "user filament file"
    assert mc.resolve("PLA", c).matched_by == "family" and len(mc.resolve("PLA", c).candidates) == 10       # the user filament is in the family
    text = (tmp_path / mc.USER_FILE_NAME).read_text(encoding="utf-8")
    assert text.startswith("# topopt user filaments (format 1)") and "enclosure: 'no'" in text and "nozzle_c: [200, 220]" in text
    assert "\n- id: acme-pla\n" in text                                  # block style: one fact on a line, a person can edit it


def test_save_replaces_an_entry_of_the_same_id_and_keeps_the_others(tmp_path):
    mc.save_user_filament(tmp_path, user_entry("acme-a"))
    mc.save_user_filament(tmp_path, user_entry("acme-b", product="Other PLA"))
    r = mc.save_user_filament(tmp_path, user_entry("acme-a", E_xy_mpa=2600))
    assert r["ok"] and r["replaced"] and not r["created"] and "gets the new values" in r["note"] and "profile_sha256" in r["note"]
    c = mc.catalog_for(tmp_path)
    assert [p.id for p in c.profiles[-2:]] == ["acme-a", "acme-b"] and c.by_id["acme-a"].E_xy_mpa == 2600.0


def test_save_refuses_a_shipped_id_a_shipped_name_and_a_bad_entry(tmp_path):
    r = mc.save_user_filament(tmp_path, user_entry("bambu-lab-abs"))
    assert not r["ok"] and "never replaces a shipped one" in r["error"] and "bambu-lab-abs-user" in r["error"]
    r = mc.save_user_filament(tmp_path, user_entry("my-abs", maker="Bambu Lab", product="ABS"))
    assert not r["ok"] and "is also a name of bambu-lab-abs" in r["error"]
    r = mc.save_user_filament(tmp_path, {"id": "x1", "maker": "Acme"})
    assert not r["ok"] and "missing key 'product'" in r["error"]
    r = mc.save_user_filament(tmp_path, user_entry(source=None))
    assert not r["ok"] and "give source_url" in r["error"]
    r = mc.save_user_filament(tmp_path, user_entry(E_xy_mpa=-5))
    assert not r["ok"] and "number above 0" in r["error"]
    r = mc.save_user_filament(tmp_path, "not a mapping")
    assert not r["ok"] and "must be a mapping" in r["error"]
    assert not (tmp_path / mc.USER_FILE_NAME).exists()                    # a refusal writes nothing


def test_a_user_entry_can_name_a_data_sheet_instead_of_a_user_word(tmp_path):
    r = mc.save_user_filament(tmp_path, user_entry(source=None, source_url="https://acme.example/pla.pdf", opened="2026-10-25",
                                                   tag="maker TDS", sheet_version="V2"))
    assert r["ok"] and r["profile"]["source_url"] == "https://acme.example/pla.pdf" and r["profile"]["tag"] == "maker TDS"
    assert mc.resolve("acme super pla", mc.catalog_for(tmp_path)).profile.sheet_version == "V2"


def test_a_broken_file_is_a_fact_and_is_not_overwritten(tmp_path):
    path = tmp_path / mc.USER_FILE_NAME
    path.write_text("materials: [oops\n", encoding="utf-8")
    c = mc.catalog_for(tmp_path)
    assert len(c.profiles) == 69 and c.user_file.exists and "not valid YAML" in c.user_file.problems[0] and "The file is not used" in c.user_file.problems[0]
    r = mc.save_user_filament(tmp_path, user_entry())
    assert not r["ok"] and "did not change" in r["error"] and path.read_text(encoding="utf-8") == "materials: [oops\n"
    assert "problems" in mc.resolve("zzzz", c).note
    path.write_text("[1, 2]\n", encoding="utf-8")
    assert "must hold a mapping" in mc.catalog_for(tmp_path).user_file.problems[0]
    path.write_text("format: 7\n", encoding="utf-8")
    assert "format is 7" in mc.catalog_for(tmp_path).user_file.problems[0]
    path.write_text("colour: red\n", encoding="utf-8")
    assert "unknown key 'colour'" in mc.catalog_for(tmp_path).user_file.problems[0]
    path.write_bytes(b"\xff\xfe\x00bad")
    assert "UTF-8" in mc.catalog_for(tmp_path).user_file.problems[0]


def test_a_bad_entry_of_the_file_is_left_out_and_named_and_the_others_work(tmp_path):
    entries = [user_entry("acme-a"), user_entry("acme-bad", E_xy_mpa="lots"), user_entry("bambu-lab-abs", product="Mine"),
               user_entry("acme-a", product="Twice"), user_entry("acme-clash", maker="Bambu Lab", product="ABS")]
    (tmp_path / mc.USER_FILE_NAME).write_text(yaml.safe_dump({"format": 1, "materials": entries}), encoding="utf-8")
    c = mc.catalog_for(tmp_path)
    assert c.user_file.entries == 1 and [p.id for p in c.profiles[-1:]] == ["acme-a"] and len(c.user_file.problems) == 4
    joined = " ".join(c.user_file.problems)
    for part in ("acme-bad", "number above 0", "never replaces a shipped one", "used twice in this file", "must differ from the names"):
        assert part in joined, part
    assert c.by_id["bambu-lab-abs"].origin == "catalog"                  # no shadowing
    assert mc.resolve("bambu abs", c).profile.id == "bambu-lab-abs"


def test_a_file_edited_by_hand_with_an_unquoted_no_and_a_date_is_read(tmp_path):
    (tmp_path / mc.USER_FILE_NAME).write_text(
        "format: 1\nmaterials:\n- id: hand-made\n  maker: Acme\n  product: Hand\n  family: PETG\n  source_url: https://acme.example/h.pdf\n"
        "  opened: 2026-10-25\n  enclosure: no\n  E_xy_mpa: 2100\n", encoding="utf-8")
    c = mc.catalog_for(tmp_path)
    p = c.by_id["hand-made"]
    assert c.user_file.problems == () and p.enclosure == "no" and p.opened == "2026-10-25" and p.E_xy_mpa == 2100.0


def test_a_user_file_changes_no_shipped_name(tmp_path):
    (tmp_path / mc.USER_FILE_NAME).write_text(yaml.safe_dump({"format": 1, "makers": {"Bambu Lab": ["acme"]}, "materials": [user_entry()]}),
                                              encoding="utf-8")
    c = mc.catalog_for(tmp_path)
    assert mc.resolve("bambu abs", c).profile.id == "bambu-lab-abs" and mc.resolve("acme super pla", c).status == "match"


def test_the_user_file_name_and_the_work_folder_follow_the_one_rule(tmp_path, monkeypatch):
    assert mc.USER_FILE_NAME == "topopt-filaments.yaml" and mc.user_file_path(tmp_path) == tmp_path / "topopt-filaments.yaml"
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    mc.save_user_filament(tmp_path, user_entry())
    assert mc.work_folder() == tmp_path and "acme-pla" in mc.work_catalog().by_id


def test_a_write_error_is_a_refusal_with_the_reason(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    r = mc.save_user_filament(blocker / "inside", user_entry())
    assert not r["ok"] and "could not write" in r["error"]


def test_save_user_filament_writes_through_write_atomic_and_a_failed_write_leaves_the_old_file(tmp_path, monkeypatch):
    """Fix round 1 (N-8). Review mutation M16: a mutant that wrote the file with `path.write_bytes` passed every test."""
    import errno

    from topopt import runinfo
    assert mc.save_user_filament(tmp_path, user_entry())["ok"]
    seen = []
    real = runinfo.write_atomic

    def spy(path, data):
        seen.append(Path(path))
        return real(path, data)

    monkeypatch.setattr(runinfo, "write_atomic", spy)
    assert mc.save_user_filament(tmp_path, user_entry("acme-two", product="Two PLA"))["ok"]
    assert seen == [tmp_path / mc.USER_FILE_NAME]                                      # the one write goes through write_atomic

    def boom(path, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(runinfo, "write_atomic", boom)
    before = (tmp_path / mc.USER_FILE_NAME).read_bytes()
    r = mc.save_user_filament(tmp_path, user_entry("acme-three", product="Three PLA"))
    assert r["ok"] is False and "could not write" in r["error"] and "No space left on device" in r["error"]
    assert (tmp_path / mc.USER_FILE_NAME).read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == [mc.USER_FILE_NAME]


def test_a_saved_entry_reads_back_to_the_same_facts(tmp_path):
    entry = user_entry(family="PA6-CF", E_z_mpa=1100, strength_xy_mpa=90, strength_kind="peak", nozzle_c=[270, 290], bed_c=[60, 80],
                       enclosure="required", chamber_c_min=45, hardened_nozzle=True, abrasive=True, soft=False, drying="80 C, 8 h",
                       conditioned={"condition": "wet", "E_xy_mpa": 900}, nu=0.38, aliases=["Super"], note="my spool")
    r = mc.save_user_filament(tmp_path, entry)
    assert r["ok"]
    p = mc.catalog_for(tmp_path).by_id["acme-pla"]
    assert (p.family, p.E_z_mpa, p.strength_kind, p.nozzle_c, p.enclosure, p.chamber_c_min, p.hardened_nozzle, p.nu) == (
        "PA6-CF", 1100.0, "peak", (270.0, 290.0), "required", 45.0, True, 0.38)
    assert p.wet == {"condition": "wet", "strength_xy_mpa": None, "strength_z_mpa": None, "E_xy_mpa": 900.0, "E_z_mpa": None}
    assert p.aliases == ("Super",) and p.note == "my spool" and mc.resolve("acme super", mc.catalog_for(tmp_path)).profile.id == "acme-pla"
    assert mc.FilamentProfile.from_facts(p.id, "", p.facts()) == replace(p, aliases=())       # the record of a run holds no aliases


def test_the_shipped_row_keys_are_the_same_as_the_user_entry_keys():
    """A user entry is the same record as a shipped row (the header of the file says so), so a row can be copied to the user file."""
    row_keys = set(row("a1")) - {"opened", "tag", "source_url"}
    assert row_keys == set(mc.IDENTITY_FIELDS + mc.FACT_FIELDS)
    assert set(mc.entry_dict(synthetic(row("a1")).profiles[0])) >= set(mc.IDENTITY_FIELDS + mc.FACT_FIELDS)
    assert copy.deepcopy(row("a1")) == row("a1")
