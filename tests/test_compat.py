"""PM-2: printer x filament compatibility (CONTRACTS.md section 18.8). A pure function of two profiles."""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import compat, materials_catalog as mc, printers  # noqa: E402


def printer(**over):
    base = dict(id="acme-p1", vendor="Acme", model="P1", printer_type="fdm", bed_mm=(200.0, 200.0), build_height_mm=200.0, nozzle_mm=0.4,
                max_nozzle_c=300.0, max_bed_c=100.0, enclosure="enclosed", chamber_max_c=None, stock_nozzle_material="stainless",
                extruder="direct", catalog_version="t1")
    base.update(over)
    return printers.PrinterProfile(**base)


def filament(**over):
    base = dict(id="acme-f1", maker="Acme", product="F1", family="PLA", sheet_version="V1", E_xy_mpa=2000.0, E_z_mpa=1800.0,
                density_g_cm3=1.2, nozzle_c=(190.0, 230.0), bed_c=(30.0, 60.0), enclosure="no", hardened_nozzle=False, abrasive=False,
                soft=False, catalog_version="t1")
    base.update(over)
    return mc.FilamentProfile(**base)


def states(result):
    return {r.check: r.state for r in result.reasons}


def test_a_plain_filament_on_a_capable_printer_is_ok_and_every_reason_names_both_values():
    r = compat.check(printer(), filament())
    assert r.state == "ok" and set(states(r).values()) == {"ok"}
    assert states(r) == {"nozzle_temperature": "ok", "bed_temperature": "ok", "enclosure": "ok"}
    text = {x.check: x.text for x in r.reasons}
    assert "Acme F1 (catalog t1, sheet V1) needs 190 to 230 C; Acme P1 (catalog t1) reaches 300 C" in text["nozzle_temperature"]
    assert "needs 30 to 60 C" in text["bed_temperature"] and "reaches 100 C" in text["bed_temperature"]
    assert r.printer == "Acme P1 (acme-p1)" and r.filament == "Acme F1 (acme-f1)"


@pytest.mark.parametrize("limit, state", [(180, "no"), (189, "no"), (190, "care"), (229, "care"), (230, "ok"), (300, "ok"), (None, "unknown")])
def test_the_nozzle_temperature_is_the_printer_limit_against_the_range(limit, state):
    r = compat.check(printer(max_nozzle_c=limit), filament())
    assert states(r)["nozzle_temperature"] == state
    assert r.state == state


@pytest.mark.parametrize("limit, state", [(25, "no"), (29, "no"), (30, "care"), (59, "care"), (60, "ok"), (None, "unknown")])
def test_the_bed_temperature_is_the_printer_limit_against_the_range(limit, state):
    assert states(compat.check(printer(max_bed_c=limit), filament()))["bed_temperature"] == state


def test_a_range_of_one_number_has_no_lower_part():
    f = filament(nozzle_c=(230.0, 230.0))
    assert states(compat.check(printer(max_nozzle_c=229), f))["nozzle_temperature"] == "no"
    assert states(compat.check(printer(max_nozzle_c=230), f))["nozzle_temperature"] == "ok"
    assert "needs 230 C" in compat.check(printer(), f).reasons[0].text


def test_a_range_that_the_catalog_does_not_state_is_unknown_and_says_who_lacks_it():
    one = compat.check(printer(), filament(nozzle_c=None)).reasons[0]
    assert one.state == "unknown" and "states no range" in one.text and "reaches 300 C" in one.text
    both = compat.check(printer(max_nozzle_c=None), filament(nozzle_c=None)).reasons[0]
    assert both.state == "unknown" and "neither" in both.text
    other = compat.check(printer(max_nozzle_c=None), filament()).reasons[0]
    assert other.state == "unknown" and "states no maximum" in other.text and "needs 190 to 230 C" in other.text


@pytest.mark.parametrize("need, have, state", [
    ("required", "open", "no"), ("required", "enclosed", "ok"), ("required", "heated", "ok"), ("required", None, "unknown"),
    ("recommended", "open", "care"), ("recommended", "enclosed", "ok"), ("recommended", "heated", "ok"), ("recommended", None, "unknown"),
    ("no", "open", "ok"), ("no", "heated", "ok"), ("no", None, "ok"), (None, "open", "unknown"), (None, None, "unknown"),
    # L1 (fix round 0): an enclosed or heated printer meets any enclosure need, also when the sheet states nothing
    (None, "enclosed", "ok"), (None, "heated", "ok")])
def test_the_enclosure_need_against_the_enclosure_of_the_printer(need, have, state):
    assert states(compat.check(printer(enclosure=have), filament(enclosure=need)))["enclosure"] == state


def test_an_enclosed_printer_and_a_sheet_that_says_nothing_about_an_enclosure_is_ok_and_the_text_says_why():
    r = compat.check(printer(enclosure="enclosed"), filament(enclosure=None))
    text = next(x.text for x in r.reasons if x.check == "enclosure")
    assert r.state == "ok" and "states nothing about an enclosure" in text and "is enclosed" in text
    open_ = next(x for x in compat.check(printer(enclosure="open"), filament(enclosure=None)).reasons if x.check == "enclosure")
    assert open_.state == "unknown" and "states nothing about an enclosure" in open_.text and "is open" in open_.text


@pytest.mark.parametrize("p, state", [
    (dict(enclosure="heated", chamber_max_c=60.0), "ok"), (dict(enclosure="heated", chamber_max_c=45.0), "ok"),
    (dict(enclosure="heated", chamber_max_c=44.0), "no"), (dict(enclosure="heated", chamber_max_c=None), "unknown"),
    (dict(enclosure="enclosed"), "care"), (dict(enclosure="open"), "care"), (dict(enclosure=None), "unknown")])
def test_the_chamber_the_sheet_names_against_the_chamber_heater(p, state):
    r = compat.check(printer(**p), filament(enclosure="required", chamber_c_min=45.0))
    assert states(r)["chamber"] == state
    text = next(x.text for x in r.reasons if x.check == "chamber")
    assert "names a chamber of 45 C or more" in text


def test_a_sheet_with_no_chamber_temperature_has_no_chamber_check():
    assert "chamber" not in states(compat.check(printer(), filament(chamber_c_min=None)))


@pytest.mark.parametrize("stock, state", [("brass", "care"), ("stainless", "care"), ("hardened_steel", "ok"), ("tungsten_carbide", "ok"),
                                          (None, "unknown")])
def test_a_hardened_nozzle_need_against_the_stock_nozzle(stock, state):
    r = compat.check(printer(stock_nozzle_material=stock), filament(hardened_nozzle=True, abrasive=True))
    assert states(r)["nozzle_wear"] == state
    if state == "care":
        assert "a hardened one has to be fitted" in next(x.text for x in r.reasons if x.check == "nozzle_wear")


def test_fibre_with_no_word_of_the_sheet_on_the_nozzle_is_read_as_the_need_of_a_hardened_nozzle():
    r = compat.check(printer(stock_nozzle_material="brass"), filament(hardened_nozzle=None, abrasive=True))
    assert states(r)["nozzle_wear"] == "care" and "holds fibre and the sheet does not say about the nozzle" in r.reasons[-1].text
    assert "nozzle_wear" not in states(compat.check(printer(stock_nozzle_material="brass"), filament(hardened_nozzle=False)))
    assert "nozzle_wear" not in states(compat.check(printer(stock_nozzle_material="brass"), filament(hardened_nozzle=None, abrasive=False)))
    unknown = compat.check(printer(), filament(hardened_nozzle=None, abrasive=None))
    assert states(unknown)["nozzle_wear"] == "unknown"


@pytest.mark.parametrize("extruder, soft, direct_drive, state", [
    ("direct", True, None, "ok"), ("bowden", True, None, "care"), (None, True, None, "unknown"),
    ("bowden", False, True, "care"), ("direct", False, True, "ok"), (None, False, True, "unknown"), ("bowden", True, True, "care")])
def test_a_soft_filament_or_a_direct_drive_need_against_the_extruder(extruder, soft, direct_drive, state):
    r = compat.check(printer(extruder=extruder), filament(soft=soft, direct_drive=direct_drive))
    assert states(r)["extruder"] == state


def test_a_stiff_filament_with_no_direct_drive_need_has_no_extruder_check():
    assert "extruder" not in states(compat.check(printer(extruder="bowden"), filament(soft=False, direct_drive=None)))


@pytest.mark.parametrize("extruder", ["direct", "bowden", None])
def test_a_soft_filament_is_care_because_a_linear_model_does_not_fit_it_on_every_printer(extruder):
    r = compat.check(printer(extruder=extruder), filament(soft=True))
    reason = next(x for x in r.reasons if x.check == "material_model")
    assert reason.state == "care" and "a linear model does not fit this material" in reason.text and "Acme F1" in reason.text
    assert r.state in ("care", "unknown")                                 # never ok; an unknown extruder type is the worse state
    for stiff in (False, None):
        assert "material_model" not in states(compat.check(printer(extruder=extruder), filament(soft=stiff)))


def test_wet_values_make_a_care_line_with_the_dry_and_wet_numbers_and_the_condition():
    wet = (("condition", "annealed, then water at 60 C for 48 h"), ("strength_xy_mpa", 34.3), ("strength_z_mpa", 13.7), ("E_xy_mpa", 724.0),
           ("E_z_mpa", 660.0))
    r = compat.check(printer(), filament(strength_xy_mpa=78.0, conditioned=wet))
    assert states(r)["moisture"] == "care" and r.state == "care"
    text = next(x.text for x in r.reasons if x.check == "moisture")
    for part in ("annealed, then water at 60 C for 48 h", "strength 78 to 34.3 MPa", "modulus 2000 to 724 MPa", "dry values hold for a dry part only"):
        assert part in text, part
    no_dry = compat.check(printer(), filament(conditioned=wet, strength_xy_mpa=None, E_xy_mpa=None))
    assert "gives wet values" in next(x.text for x in no_dry.reasons if x.check == "moisture")


def test_the_drying_text_is_information_and_never_changes_the_state():
    r = compat.check(printer(), filament(drying="55 C, 6 h"))
    assert r.state == "ok" and r.drying == "55 C, 6 h" and r.to_dict()["drying"] == "55 C, 6 h"
    assert "drying" not in states(r)


def test_a_printer_that_is_not_fdm_is_a_no():
    r = compat.check(printer(printer_type="sla"), filament())
    assert r.state == "no" and states(r)["printer_type"] == "no" and "a filament is for an fdm printer" in r.reasons[0].text


def test_the_state_of_the_whole_is_the_worst_state_no_then_unknown_then_care_then_ok():
    assert compat.worst(["ok", "care", "unknown", "no"]) == "no"
    assert compat.worst(["ok", "care", "unknown"]) == "unknown"
    assert compat.worst(["ok", "care"]) == "care" and compat.worst(["ok", "ok"]) == "ok" and compat.worst([]) == "unknown"
    # one check of each state in one answer
    r = compat.check(printer(enclosure="open", max_nozzle_c=None), filament(enclosure="required"))
    assert states(r) == {"nozzle_temperature": "unknown", "bed_temperature": "ok", "enclosure": "no"} and r.state == "no"
    r = compat.check(printer(max_nozzle_c=None), filament())
    assert r.state == "unknown"
    r = compat.check(printer(max_nozzle_c=200), filament())
    assert r.state == "care"


def test_the_line_for_a_person_names_the_state_both_names_and_what_is_open():
    ok = compat.check(printer(), filament()).line()
    assert ok == "ok: Acme F1 (acme-f1) on Acme P1 (acme-p1): the stated values fit"
    line = compat.check(printer(enclosure="open"), filament(enclosure="required")).line()
    assert line.startswith("no: Acme F1 (acme-f1) on Acme P1 (acme-p1): enclosure: ") and "is open" in line and "nozzle temperature" not in line


def test_to_dict_is_json():
    import json
    d = compat.check(printer(enclosure="open"), filament(enclosure="required")).to_dict()
    assert json.loads(json.dumps(d)) == d and d["state"] == "no" and {r["check"] for r in d["reasons"]} <= set(compat.CHECKS)
    assert all(r["state"] in compat.STATES for r in d["reasons"])


def test_a_user_profile_is_named_by_its_file_in_the_reasons():
    r = compat.check(printer(origin="user"), filament(origin="user"))
    assert "user printer file" in r.reasons[0].text and "user filament file" in r.reasons[0].text


# --- the catalog rows --------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pc():
    return printers.load_catalog()


@pytest.fixture(scope="module")
def fc():
    return mc.load_catalog()


def test_known_pairs_of_the_catalogs(pc, fc):
    got = lambda p, f: compat.check(pc.by_id[p], fc.by_id[f])   # noqa: E731
    assert got("bambu-lab-a1", "bambu-lab-pla-basic").state == "ok"
    assert got("bambu-lab-p1s", "bambu-lab-pla-basic").state == "ok"
    assert got("bambu-lab-a1", "bambu-lab-abs").state == "no"                      # ABS needs an enclosure, the A1 is open
    assert states(got("bambu-lab-a1", "bambu-lab-abs"))["enclosure"] == "no"
    abs_p1s = got("bambu-lab-p1s", "bambu-lab-abs")
    assert abs_p1s.state == "care" and states(abs_p1s)["chamber"] == "care" and states(abs_p1s)["enclosure"] == "ok"
    assert got("bambu-lab-h2d", "polymaker-abs-max").state == "ok"                 # chamber 65 C or more against a heater up to 65 C
    assert got("prusa-core-one-plus", "polymaker-abs-max").state == "no"           # the heater reaches 55 C, the sheet names 65 C or more
    assert got("bambu-lab-p1s", "bambu-lab-asa-cf").state == "care"                # stainless stock nozzle, hardened steel needed
    assert got("bambu-lab-p1s", "bambu-lab-pps-cf").state == "no"                  # nozzle 310 C and more, the P1S reaches 300 C
    assert got("prusa-mk4s", "bambu-lab-pps-cf").state == "no"
    assert got("bambu-lab-p1s", "polymaker-polymide-copa").state == "care"
    assert states(got("bambu-lab-p1s", "polymaker-polymide-copa"))["moisture"] == "care"
    assert got("voron-2-4", "bambu-lab-pla-basic").state == "unknown"              # the Voron row states no temperature limit


@pytest.mark.parametrize("have, state", [("enclosed", "care"), ("heated", "care"), ("open", "ok"), (None, "ok")])
def test_a_sheet_note_about_an_enclosed_printer_is_care_with_the_text_of_the_sheet(have, state):
    """Fix round 1 (MI-2). The note speaks of printing in an enclosure, so it counts on an enclosed or heated printer only."""
    f = filament(enclosed_note="Leave the enclosure door open.")
    r = compat.check(printer(enclosure=have), f)
    assert states(r)["enclosure"] == state
    if state == "care":
        text = next(x.text for x in r.reasons if x.check == "enclosure")
        assert 'says: "Leave the enclosure door open."' in text and f"is {have}" in text and "Acme F1 (catalog t1, sheet V1)" in text
        assert r.state == "care"
    assert states(compat.check(printer(enclosure=have), filament()))["enclosure"] == "ok"


def test_the_note_of_the_sheet_wins_over_a_filament_that_requires_an_enclosure_only_as_an_added_care():
    r = compat.check(printer(enclosure="enclosed"), filament(enclosure="required", enclosed_note="Leave the door open."))
    assert states(r)["enclosure"] == "care"
    assert states(compat.check(printer(enclosure="open"), filament(enclosure="required", enclosed_note="Leave the door open.")))["enclosure"] == "no"


def test_prusament_tpu_on_a_heated_printer_names_the_door_of_the_page(pc, fc):
    r = compat.check(pc.by_id["prusa-core-one-plus"], fc.by_id["prusament-tpu-95a"])
    text = next(x.text for x in r.reasons if x.check == "enclosure")
    assert states(r)["enclosure"] == "care" and "leave the enclosure door open" in text
    assert states(compat.check(pc.by_id["prusa-mk4s"], fc.by_id["prusament-tpu-95a"]))["enclosure"] == "ok"      # an open printer has no door
    assert states(compat.check(pc.by_id["bambu-lab-p1s"], fc.by_id["bambu-lab-pla-basic"]))["enclosure"] == "ok"      # no note on the PLA row


@pytest.mark.parametrize("fid", ["bambu-lab-petg-basic", "bambu-lab-petg-cf"])
def test_bambu_petg_on_an_open_printer_is_care_and_on_an_enclosed_printer_ok(pc, fc, fid):
    """The maker's material table recommends an enclosed printer for PETG and PETG-CF (fix round 1, MA-1)."""
    open_ = compat.check(pc.by_id["bambu-lab-a1-mini"], fc.by_id[fid])
    assert states(open_)["enclosure"] == "care" and "recommends an enclosed printer" in next(r.text for r in open_.reasons if r.check == "enclosure")
    assert states(compat.check(pc.by_id["prusa-mk4s"], fc.by_id[fid]))["enclosure"] == "care"
    assert states(compat.check(pc.by_id["bambu-lab-p1s"], fc.by_id[fid]))["enclosure"] == "ok"


def test_a_row_that_states_no_print_needs_is_unknown_not_ok(pc, fc):
    """A row without the print temperatures is `unknown` on the temperature checks, never `ok` (the common rows state them now)."""
    from dataclasses import replace
    bare = replace(fc.by_id["polymaker-polylite-pla"], nozzle_c=None, bed_c=None)
    r = compat.check(pc.by_id["bambu-lab-p1s"], bare)
    assert r.state == "unknown" and states(r)["nozzle_temperature"] == "unknown" and "states no range" in r.reasons[0].text
    assert compat.check(pc.by_id["bambu-lab-p1s"], fc.by_id["polymaker-polylite-pla"]).state == "ok"


def test_every_pair_of_the_catalogs_has_a_state_and_reasons_that_name_a_value_or_a_gap(pc, fc):
    for p in pc.profiles:
        for f in fc.profiles:
            r = compat.check(p, f)
            assert r.state in compat.STATES and r.reasons, (p.id, f.id)
            assert all(x.text.startswith(x.check.replace("_", " ").split()[0]) for x in r.reasons), (p.id, f.id)
            assert all(f.name in x.text or x.check == "printer_type" for x in r.reasons), (p.id, f.id)


# --- the rules of the module -------------------------------------------------------------------------------------------------

def test_check_is_pure_it_reads_no_file_and_changes_no_profile(pc, fc, monkeypatch):
    p, f = pc.by_id["bambu-lab-p1s"], fc.by_id["bambu-lab-asa-cf"]
    before = (p.facts(), f.facts())

    def refuse(*a, **k):
        raise AssertionError("compat.check opened a file")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(Path, "read_text", refuse)
    first = compat.check(p, f)
    assert compat.check(p, f) == first
    assert (p.facts(), f.facts()) == before


def test_the_module_has_no_numeric_threshold_of_its_own():
    """Every check compares two stated values. The only numbers in the source are the ranks 0 to 3 of the four states."""
    tree = ast.parse((ROOT / "topopt" / "compat.py").read_text(encoding="utf-8"))
    numbers = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
    assert numbers <= {0, 1, 2, 3}, numbers
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"__future__", "dataclasses"}, imported


# --- fix round 1 (MI-5): drying is a fact line, never a state -----------------------------------------------------------------------

def test_no_text_lists_drying_among_the_cases_of_care_and_the_code_agrees(pc, fc):
    from topopt_mcp import server
    assert "dry the spool" not in compat.__doc__
    assert "dry the spool" not in server.printer_materials.__doc__
    # the code: the drying text never changes a state, and the long list shows no `drying` key
    for fid in ("bambu-lab-petg-cf", "polymaker-polylite-pla"):
        r = compat.check(pc.by_id["bambu-lab-p1s"], fc.by_id[fid])
        assert r.drying is not None and all("dry" not in x.text.lower() or x.check == "moisture" for x in r.reasons)
    fit = mc.printer_fit(pc.by_id["bambu-lab-p1s"], fc)
    entry = next(x for group in fit["groups"].values() for x in group if x["id"] == "bambu-lab-petg-cf")
    assert "drying" not in entry


# --- fix round 1 (N-6): CONTRACTS says what the code does for a filament that says nothing about the nozzle -------------------------

def test_a_filament_that_says_nothing_about_the_nozzle_is_unknown_only_when_it_does_not_say_if_it_is_abrasive():
    silent = compat.check(printer(), filament(hardened_nozzle=None, abrasive=None))
    assert states(silent)["nozzle_wear"] == "unknown"
    plain = compat.check(printer(), filament(hardened_nozzle=None, abrasive=False))
    assert "nozzle_wear" not in states(plain)
    text = " ".join((ROOT / "CONTRACTS.md").read_text(encoding="utf-8").split())
    assert "A filament that says nothing about the nozzle and does not say whether it is abrasive has `unknown`" in text
    assert "is not marked as abrasive has `unknown`" not in text
