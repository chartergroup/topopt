"""PM-2: the filament in the preferences file (`material.filament`, `filaments.available`) and in `build_regions` (CONTRACTS.md sections 10.2 and 18.8)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import materials_catalog as mc, preferences, regions_builder  # noqa: E402
from topopt.config import UserError, load_config  # noqa: E402
from topopt_mcp import server  # noqa: E402

PREFS = preferences.FILE_NAME
USER_FILE = mc.USER_FILE_NAME
ABS = "polymaker-polylite-abs"
PETG = "polymaker-polylite-petg"
NO_MODULUS = "esun-abs-plus"                      # the catalog gives its density only


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


def tiny_spec(**extra):
    return {"name": "tiny", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}]},
        "load": {"tip": [{"box": {"min": [16, 2, 0], "max": [20, 8, 6]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
        "settings": {"iterations": 3, "voxel_size": 2.0}, **extra}


def written(work, reply):
    return yaml.safe_load((work / reply["yaml_rel"]).read_text())


def add_user_filament(work, **over):
    raw = {"id": "acme-pla", "maker": "Acme", "product": "Super PLA", "family": "PLA", "source": "user",
           "E_xy_mpa": 2500, "E_z_mpa": 2300, "density_g_cm3": 1.25, **over}
    assert mc.save_user_filament(work, raw)["ok"]


# --- the file ----------------------------------------------------------------------------------------------------------------

def test_the_filament_is_a_material_key_and_the_file_keeps_the_id(tmp_path):
    assert "filament" in preferences.MATERIAL_KEYS and preferences.MATERIAL_KEYS.index("filament") < preferences.MATERIAL_KEYS.index("name")
    r = preferences.save(tmp_path, material={"filament": "Polymaker PolyLite ABS", "name": "ABS"})
    assert r["ok"] and r["values"]["material"] == {"filament": ABS, "name": "ABS"}
    assert yaml.safe_load((tmp_path / PREFS).read_text())["material"] == {"filament": ABS, "name": "ABS"}
    again = preferences.save(tmp_path, material={"filament": ABS})
    assert again["ok"] and again["changed"] == [] and again["note"] == "Nothing changed: the file already has these values."
    assert preferences.save(tmp_path, material={"filament": None})["values"]["material"] == {"name": "ABS"}      # null removes the key


@pytest.mark.parametrize("value, expect", [
    ("Frob 9000", "'material.filament' 'Frob 9000' is not in the filament catalog"),
    ("PLA", "'material.filament' 'PLA' fits"),                                 # a family names no product
    (7, "'material.filament' must be the name or the id of a filament as text"),
    ("  ", "'material.filament' must be the name or the id of a filament as text"),
])
def test_a_text_that_is_not_one_filament_is_refused_and_nothing_is_written(tmp_path, value, expect):
    r = preferences.save(tmp_path, material={"filament": value, "E": 2000})
    assert r["ok"] is False and r["error"].startswith("save_preferences wrote nothing.") and expect in r["error"]
    assert not (tmp_path / PREFS).exists()


def test_a_filament_with_no_modulus_can_be_kept_because_the_yaml_may_give_e(tmp_path):
    assert preferences.save(tmp_path, material={"filament": NO_MODULUS})["values"]["material"] == {"filament": NO_MODULUS}


def test_filaments_available_keeps_ids_once_each_in_the_order_given(tmp_path):
    assert preferences.FILAMENTS_KEYS == ("available",) and "filaments" in preferences.SECTIONS
    r = preferences.save(tmp_path, filaments={"available": ["Polymaker PolyLite ABS", PETG, "polylite abs", "bambu pla basic"]})
    assert r["ok"] and r["changed"] == ["filaments"]
    ids = r["values"]["filaments"]["available"]
    assert ids[:2] == [ABS, PETG] and len(ids) == 3 and len(set(ids)) == 3
    assert yaml.safe_load((tmp_path / PREFS).read_text())["filaments"] == {"available": ids}
    assert preferences.read(tmp_path)["values"]["filaments"] == {"available": ids}
    assert "filaments" in preferences.HEADER and "filaments" in preferences.NOTE_NONE
    assert preferences.save(tmp_path, filaments={"available": None})["values"].get("filaments") is None       # null removes the key
    again = preferences.save(tmp_path, filaments={"available": [PETG]})
    assert again["values"]["filaments"] == {"available": [PETG]}
    assert "filaments" not in preferences.save(tmp_path, filaments={})["values"]                               # {} removes the section


@pytest.mark.parametrize("value, expect", [
    ({"available": [ABS, "Frob 9000"]}, "'filaments.available[1]' 'Frob 9000' is not in the filament catalog"),
    ({"available": ["PLA"]}, "'filaments.available[0]' 'PLA' fits"),
    ({"available": [ABS, 5]}, "'filaments.available[1]' must be the name or the id of a filament as text, got 5"),
    ({"available": ABS}, "'filaments.available' must be a list of filament names or ids"),
    ({"have": [ABS]}, "filaments.have: not a key of this section. Allowed keys: available"),
])
def test_a_wrong_list_is_refused_with_the_key_and_the_rule_and_nothing_is_written(tmp_path, value, expect):
    r = preferences.save(tmp_path, filaments=value)
    assert r["ok"] is False and r["error"].startswith("save_preferences wrote nothing.") and expect in r["error"], r
    assert not (tmp_path / PREFS).exists()


def test_a_user_filament_can_be_kept_and_listed(work):
    add_user_filament(work)
    r = preferences.save(work, material={"filament": "acme super pla"}, filaments={"available": ["acme-pla", ABS]})
    assert r["ok"] and r["values"]["material"] == {"filament": "acme-pla"} and r["values"]["filaments"] == {"available": ["acme-pla", ABS]}


def test_a_filament_that_is_not_found_now_makes_only_that_key_unused_and_the_rest_of_the_file_stays_usable(work):
    add_user_filament(work)
    saved = preferences.save(work, material={"filament": "acme-pla", "E": 1800, "name": "PLA"}, filaments={"available": ["acme-pla", ABS]},
                             quality={"voxel_size": 3.0}, outputs=["renders"])
    assert saved["ok"] and "not_used" not in saved
    (work / USER_FILE).unlink()                                                  # the filament is not found now
    facts = preferences.read(work)
    assert "problems" not in facts and "kept_not_used" not in facts
    assert facts["values"] == {"material": {"E": 1800.0, "name": "PLA"}, "filaments": {"available": [ABS]}, "quality": {"voxel_size": 3.0},
                               "outputs": ["renders"]}          # fix round 1 (MI-3): the other id of the list stays
    assert len(facts["not_used"]) == 2
    assert facts["not_used"][0].startswith("'material.filament' 'acme-pla' is not in the filament catalog")
    assert facts["not_used"][1].startswith("'filaments.available[0]' 'acme-pla' is not in the filament catalog")
    assert "Not used now: material.filament, filaments.available" in facts["note"] and "the filament is not found" in facts["note"]
    assert "save_material" in facts["note"] and "save_printer" not in facts["note"]
    path, values, problems = preferences.usable(work)
    assert values == facts["values"] and problems == []
    # build_regions uses the rest and says that the filament is not used
    out = server.build_regions(tiny_spec())
    assert out["ok"], out
    text = " ".join(out["preferences_facts"])
    assert "material from your preferences file" in text and "names a filament that is not found now, so build_regions did not use it" in text
    assert written(work, out)["material"] == {"E": 1800.0, "nu": 0.35, "density": 1.24}
    # the filament is back: the key is used again, nothing was lost from the file
    add_user_filament(work)
    again = preferences.read(work)
    assert "not_used" not in again and again["values"]["material"] == {"filament": "acme-pla", "E": 1800.0, "name": "PLA"}
    # a real fault in the file still makes the whole file unused
    (work / PREFS).write_text(yaml.safe_dump({"format": 1, "material": {"filament": "acme-pla", "E": "stiff"}}))
    assert "values" not in preferences.read(work)


def test_one_filament_that_is_not_found_drops_only_that_id_from_the_available_list(work):
    """Fix round 1 (MI-3): the other ids stay in the list, stay marked in `printer_materials`, and the lost id is named in `not_used`."""
    add_user_filament(work)
    assert preferences.save(work, filaments={"available": [ABS, "acme-pla", "bambu-lab-pla-basic"]})["ok"]
    (work / USER_FILE).unlink()
    facts = preferences.read(work)
    assert facts["values"]["filaments"] == {"available": [ABS, "bambu-lab-pla-basic"]}
    assert len(facts["not_used"]) == 1 and facts["not_used"][0].startswith("'filaments.available[1]' 'acme-pla' is not in the filament catalog")
    assert "Not used now: filaments.available" in facts["note"]
    r = server.printer_materials("Bambu P1S")
    assert r["ok"] and [a["id"] for a in r["available"]] == [ABS, "bambu-lab-pla-basic"], r["available"]
    marked = {x["id"] for group in r["groups"].values() for x in group if x["available"]}
    assert marked == {ABS, "bambu-lab-pla-basic"}
    # every id lost: the list is empty, the key stays, and each lost id is named
    assert preferences.save(work, filaments={"available": ["bambu-lab-pla-basic"]})["ok"]
    add_user_filament(work)
    assert preferences.save(work, filaments={"available": ["acme-pla"]})["ok"]
    (work / USER_FILE).unlink()
    gone = preferences.read(work)
    assert gone["values"]["filaments"] == {"available": []} and len(gone["not_used"]) == 1
    # a save call with a name that is not found is still refused (an input is checked in full)
    refused = preferences.save(work, filaments={"available": ["bambu-lab-pla-basic", "acme-pla"]})
    assert not refused["ok"] and "acme-pla" in refused["error"]


def test_a_printer_and_a_filament_that_are_not_found_are_named_together(work):
    from topopt import printers
    assert printers.save_user_printer(work, {"id": "voron-trident", "vendor": "Voron", "model": "Trident", "printer_type": "fdm",
                                             "bed_mm": [300, 300], "build_height_mm": 250, "source": "user"})["ok"]
    add_user_filament(work)
    assert preferences.save(work, print={"printer_model": "voron trident"}, material={"filament": "acme-pla"})["ok"]
    (work / printers.USER_FILE_NAME).unlink()
    (work / USER_FILE).unlink()
    note = preferences.read(work)["note"]
    assert "(the printer or the filament is not found; the rest of the file is used)" in note
    assert "save_printer" in note and "save_material" in note


def test_the_save_note_names_a_filament_that_is_not_found_now(work):
    add_user_filament(work)
    assert preferences.save(work, material={"filament": "acme-pla"})["ok"]
    (work / USER_FILE).unlink()
    r = preferences.save(work, quality={"voxel_size": 2.5})
    assert r["ok"] and "not_used" in r
    assert "The file has a filament that is not found now (material.filament), so that key is not live; the other values are." in r["note"]


# --- build_regions -----------------------------------------------------------------------------------------------------------

def test_build_regions_fills_the_filament_into_a_spec_without_a_material_block_and_says_what_the_run_takes(work):
    assert preferences.save(work, material={"filament": "Polymaker PolyLite ABS"})["ok"]
    r = server.build_regions(tiny_spec())
    assert r["ok"], r
    assert written(work, r)["material"] == {"filament": ABS}                    # only the filament: the numbers come from the profile
    [fact] = r["preferences_facts"]
    assert f"material from your preferences file ({work / PREFS}): filament {ABS} (Polymaker PolyLite ABS)." in fact
    assert "The spec had no material block." in fact
    assert f"E 2080.9 MPa (filament:{ABS}, the lower of the XY and Z modulus)" in fact
    assert "nu 0.35 (assumed, no data sheet gives it)" in fact and f"density 1.04 g/cm3 (filament:{ABS}, the density of the sheet)" in fact
    cfg = load_config(work / r["yaml_rel"])
    assert cfg.material.E == 2080.9 and cfg.material.density == 1.04 and cfg.filament.profile.id == ABS


def test_the_filament_and_the_numbers_of_the_file_both_go_in_and_the_numbers_of_the_file_win(work):
    assert preferences.save(work, material={"filament": ABS, "E": 1800, "name": "my spool"})["ok"]
    r = server.build_regions(tiny_spec())
    assert r["ok"], r
    assert written(work, r)["material"] == {"filament": ABS, "E": 1800.0}
    [fact] = r["preferences_facts"]
    assert "E 1800 MPa (your preferences file)" in fact and f"density 1.04 g/cm3 (filament:{ABS}" in fact
    assert "A key that the file gives wins over the filament." in fact
    assert load_config(work / r["yaml_rel"]).material.E == 1800.0


def test_a_filament_with_no_modulus_is_not_filled_and_the_fact_says_why(work):
    assert preferences.save(work, material={"filament": NO_MODULUS})["ok"]
    r = server.build_regions(tiny_spec())
    assert r["ok"], r
    assert written(work, r)["material"] == {"E": 3500.0, "nu": 0.35, "density": 1.24}      # the default block: nothing from the file
    [fact] = r["preferences_facts"]
    assert f"The preferences file {work / PREFS} names a filament. The filament of the file is not used:" in fact
    assert "has no tensile modulus" in fact
    # with an E in the file, the filament is used: the file gives the one number that the profile does not have
    assert preferences.save(work, material={"E": 2100})["ok"]
    r = server.build_regions({**tiny_spec(), "name": "tiny2"})
    assert written(work, r)["material"] == {"filament": NO_MODULUS, "E": 2100.0}
    [fact] = r["preferences_facts"]
    assert "E 2100 MPa (your preferences file)" in fact and "density 1.06 g/cm3 (filament:esun-abs-plus" in fact and "is not used" not in fact


def test_build_regions_never_mixes_the_filament_into_a_material_block_that_the_spec_has(work):
    assert preferences.save(work, material={"filament": ABS})["ok"]
    own = server.build_regions(tiny_spec(material={"E": 2500}))
    assert own["ok"] and "preferences_facts" not in own
    assert written(work, own)["material"] == {"E": 2500.0, "nu": 0.35, "density": 1.24}
    other = server.build_regions({**tiny_spec(material={"filament": PETG}), "name": "other"})
    assert other["ok"] and "preferences_facts" not in other
    assert written(work, other)["material"] == {"filament": PETG}


def test_a_spec_can_name_the_filament_and_the_builder_writes_it_with_the_numbers_that_the_spec_gives(work):
    r = server.build_regions(tiny_spec(material={"filament": " Polymaker PolyLite PETG ", "density": 1.3}))
    assert r["ok"], r
    assert written(work, r)["material"] == {"filament": "Polymaker PolyLite PETG", "density": 1.3}
    cfg = load_config(work / r["yaml_rel"])
    assert cfg.filament.profile.id == PETG and cfg.material.density == 1.3 and cfg.filament.source("density") == "yaml"
    assert r["check"] is not None and r["check"]["material"]["filament"] == PETG


def test_a_spec_with_a_filament_that_is_not_found_gets_the_one_line_error_of_the_loader(work):
    r = server.build_regions(tiny_spec(material={"filament": "Frob 9000"}))
    assert r["ok"] is False or r["problems"]
    text = " ".join(r.get("problems") or []) + str(r.get("error", ""))
    assert "'material.filament' 'Frob 9000' is not in the filament catalog" in text


@pytest.mark.parametrize("value", [7, ["abs"], {"id": ABS}, "  "])
def test_a_spec_filament_that_is_not_text_is_a_shape_error(work, value):
    r = server.build_regions(tiny_spec(material={"filament": value}))
    assert r["ok"] is False and "material.filament" in r["error"] and "the name or the id of a filament as text" in r["error"]


def test_a_spec_without_a_filament_writes_the_same_material_block_as_before():
    assert regions_builder.material_block({}) == {"E": 3500.0, "nu": 0.35, "density": 1.24}
    assert regions_builder.material_block({"E": 2000, "nu": 0.3}) == {"E": 2000.0, "nu": 0.3, "density": 1.24}
    assert regions_builder.material_block({"filament": None, "E": 2000}) == {"E": 2000.0, "nu": 0.35, "density": 1.24}


# --- check_problem -----------------------------------------------------------------------------------------------------------

def test_check_problem_names_a_filament_that_differs_and_not_one_that_is_the_same(work):
    assert preferences.save(work, material={"filament": ABS})["ok"]
    none = server.build_regions({**tiny_spec(material={"E": 2500}), "name": "none"})
    [fact] = server.check_problem(str(work / none["yaml_rel"]))["preferences_facts"]
    assert f"material filament: none in the YAML, {ABS} (Polymaker PolyLite ABS) in the preferences" in fact
    assert fact.startswith(f"The preferences file {work / PREFS} differs from this YAML") and "is not changed" in fact
    other = server.build_regions({**tiny_spec(material={"filament": PETG}), "name": "other"})
    [fact] = server.check_problem(str(work / other["yaml_rel"]))["preferences_facts"]
    assert f"material filament: {PETG} in the YAML, {ABS} (Polymaker PolyLite ABS) in the preferences" in fact
    same = server.build_regions({**tiny_spec(material={"filament": "polylite abs"}), "name": "same"})
    assert "preferences_facts" not in server.check_problem(str(work / same["yaml_rel"]))             # an alias in the YAML is the same filament


def test_a_number_that_the_filament_of_the_yaml_gives_is_named_as_such(work):
    assert preferences.save(work, material={"density": 1.5})["ok"]
    r = server.build_regions({**tiny_spec(material={"filament": ABS}), "name": "f"})
    [fact] = server.check_problem(str(work / r["yaml_rel"]))["preferences_facts"]
    assert f"material density: 1.04 g/cm3 from the filament {ABS} of the YAML, 1.5 g/cm3 in the preferences" in fact


def test_the_loader_reads_the_filament_that_build_regions_wrote(work):
    assert preferences.save(work, material={"filament": ABS})["ok"]
    r = server.build_regions(tiny_spec())
    with pytest.raises(UserError) as exc:                                # the filament is gone from the catalog of the YAML: the error is the loader's
        doc = written(work, r)
        doc["material"]["filament"] = "Frob 9000"
        (work / r["yaml_rel"]).write_text(yaml.safe_dump(doc, sort_keys=False))
        load_config(work / r["yaml_rel"])
    assert "'material.filament' 'Frob 9000'" in str(exc.value)
