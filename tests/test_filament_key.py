"""PM-2: the key `material.filament` in the problem YAML, run.json, and `topopt check` (CONTRACTS.md section 18.8)."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from topopt import levers_cards, materials_catalog as mc, printers  # noqa: E402
from topopt.config import (FILAMENT_KEYS, MATERIAL_OPTIONAL, Config, FilamentNotFound, FilamentUse, Material, UserError,  # noqa: E402
                           load_config, parse_filament_block, parse_material_block)
from test_input import make_case  # noqa: E402

USER_FILE = mc.USER_FILE_NAME
ABS = "polymaker-polylite-abs"


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A work folder for the user filament file; the environment is the one of this test."""
    folder = tmp_path / "work"
    folder.mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(folder))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    return folder


def case(tmp_path, material=None, print_block=None, h=2.0, name="case"):
    folder = tmp_path / name
    folder.mkdir(exist_ok=True)

    def mutate(d):
        if material is not None:
            d["material"] = material
        if print_block is not None:
            d["print"] = print_block
    return make_case(folder, h=h, mutate=mutate)


# --- the key in the loader ---------------------------------------------------------------------------------------------------

def test_filament_is_a_material_key():
    assert "filament" in MATERIAL_OPTIONAL and FILAMENT_KEYS == ("E", "nu", "density")


def test_a_filament_fills_e_and_density_and_nu_stays_assumed_with_the_source_of_each(tmp_path):
    cfg = load_config(case(tmp_path, {"filament": "Polymaker PolyLite ABS"}))
    assert cfg.material == Material(E=2080.9, nu=0.35, density=1.04)
    use = cfg.filament
    assert isinstance(use, FilamentUse) and use.profile.id == ABS
    assert use.keys == (("E", 2080.9, f"filament:{ABS}", "the lower of the XY and Z modulus"),
                        ("nu", 0.35, "assumed", "no data sheet gives it"),
                        ("density", 1.04, f"filament:{ABS}", "the density of the sheet"))
    assert use.source("E") == f"filament:{ABS}" and use.source("nu") == "assumed" and use.source("none") is None


def test_the_id_a_name_and_an_alias_give_the_same_filament(tmp_path):
    ids = {load_config(case(tmp_path, {"filament": t}, name=f"c{i}")).filament.profile.id
           for i, t in enumerate((ABS, "polylite abs", "Polymaker PolyLite ABS", "POLYMAKER-polylite_abs!"))}
    assert ids == {ABS}


def test_a_key_that_the_yaml_gives_wins_and_says_so(tmp_path):
    cfg = load_config(case(tmp_path, {"filament": ABS, "E": 3000.0}))
    assert cfg.material == Material(E=3000.0, nu=0.35, density=1.04)
    assert [(k, s) for k, _v, s, _r in cfg.filament.keys] == [("E", "yaml"), ("nu", "assumed"), ("density", f"filament:{ABS}")]
    cfg = load_config(case(tmp_path, {"filament": ABS, "E": 3000.0, "nu": 0.3, "density": 1.1}, name="c2"))
    assert cfg.material == Material(E=3000.0, nu=0.3, density=1.1)
    assert [s for _k, _v, s, _r in cfg.filament.keys] == ["yaml", "yaml", "yaml"]


def test_a_profile_with_a_poisson_ratio_gives_it(tmp_path, work):
    assert mc.save_user_filament(work, {"id": "my-abs", "maker": "Acme", "product": "ABS", "family": "ABS", "source": "user",
                                        "E_xy_mpa": 2200, "nu": 0.38, "density_g_cm3": 1.05})["ok"]
    cfg = load_config(case(tmp_path, {"filament": "my-abs"}))
    assert cfg.material == Material(E=2200.0, nu=0.38, density=1.05) and cfg.filament.source("nu") == "filament:my-abs"


def test_a_filament_with_one_modulus_gives_that_one_and_says_which(tmp_path):
    cfg = load_config(case(tmp_path, {"filament": "prusament asa"}))
    assert cfg.material.E == 1600.0
    reason = cfg.filament.keys[0][3]
    assert "XY modulus" in reason and "no Z modulus" in reason


def test_a_filament_without_a_modulus_is_an_error_unless_the_yaml_gives_e(tmp_path):
    with pytest.raises(UserError) as exc:
        load_config(case(tmp_path, {"filament": "esun abs plus"}))          # the sheet has no tensile and no printed-bar bending modulus
    text = str(exc.value)
    assert "\n" not in text and "'material.filament' esun-abs-plus" in text and "has no tensile modulus" in text and "bending modulus" in text
    assert "Give E in the material block" in text and "(file: " in text and "injection-moulded" not in text      # the note stays in find_material
    cfg = load_config(case(tmp_path, {"filament": "esun abs plus", "E": 2500}, name="c2"))
    assert cfg.material == Material(E=2500.0, nu=0.35, density=1.06) and cfg.filament.source("E") == "yaml"


def test_a_filament_with_only_a_bending_modulus_sets_e_from_it_and_says_so(tmp_path):
    cfg = load_config(case(tmp_path, {"filament": "elegoo abs"}))
    assert cfg.material == Material(E=2133.0, nu=0.35, density=1.05)           # the lower of the XY (2484) and Z (2133) bending modulus
    key, value, source, reason = cfg.filament.keys[0]
    assert (key, value, source) == ("E", 2133.0, "filament:elegoo-abs")
    assert reason == "the bending modulus of the sheet; the sheet gives no tensile modulus; the lower of the XY and Z value"
    # a filament that has no modulus of any kind stays an error
    with pytest.raises(UserError, match="has no tensile modulus"):
        load_config(case(tmp_path, {"filament": "polyflex tpu95"}, name="c3"))


def _median_of_family(family, catalog=None):
    import statistics
    rows = sorted((p.id, p.density_g_cm3) for p in (catalog or mc.load_catalog()).profiles
                  if p.family == family and p.density_g_cm3 is not None and p.origin == "catalog" and not p.superseded_by)
    return float(f"{statistics.median(d for _i, d in rows):g}"), rows


def test_a_filament_without_a_density_takes_the_median_of_its_family_and_labels_it_assumed(tmp_path):
    shipped = mc.load_catalog()
    bare = mc.FilamentProfile(id="polymaker-nodensity", maker="Polymaker", product="NoDensity PLA", family="PLA", E_xy_mpa=1983.6, origin="user")
    tpu = mc.FilamentProfile(id="polymaker-nodensity-tpu", maker="Polymaker", product="NoDensity TPU", family="TPU", E_xy_mpa=30.0, origin="user")
    cat = mc.Catalog(shipped.version, (*shipped.profiles, bare, tpu), shipped.makers)
    mat, use = parse_filament_block({"filament": "polymaker-nodensity"}, filament_catalog=cat)
    median, rows = _median_of_family("PLA")
    assert mat == Material(E=1983.6, nu=0.35, density=median)
    kind = dict((k, (s, r)) for k, _v, s, r in use.keys)
    assert kind["density"][0] == "assumed"
    assert kind["density"][1] == f"the median of {len(rows)} PLA rows of the catalog ({', '.join(i for i, _d in rows)})"
    mat, use = parse_filament_block({"filament": "polymaker-nodensity-tpu", "E": 30}, filament_catalog=cat)
    assert mat.density == 1.19            # the median of the four TPU rows that state a density: 1.16, 1.16, 1.22, 1.25
    reason = dict((k, r) for k, _v, _s, r in use.keys)["density"]
    assert reason == ("the median of 4 TPU rows of the catalog (bambu-lab-tpu-95a-hf, elegoo-tpu-95a, polymaker-polyflex-tpu95-hf, "
                      "prusament-tpu-95a)")


def test_a_sheet_that_gives_a_density_range_gives_the_middle_before_the_family_median_is_asked(tmp_path):
    """Fix round 1 (N-3): the middle of the range of the sheet, labelled as the filament's own value, with the range in the reason."""
    cfg = load_config(case(tmp_path, {"filament": "polymaker polymax pla"}))
    assert cfg.material == Material(E=1983.6, nu=0.35, density=1.205)
    assert dict((k, (s, r)) for k, _v, s, r in cfg.filament.keys)["density"] == (
        "filament:polymaker-polymax-pla", "the middle of the sheet range 1.17-1.24")
    cfg = load_config(case(tmp_path, {"filament": "polymaker polyflex tpu95", "E": 30}, name="c2"))
    assert cfg.material.density == 1.22
    assert dict((k, (s, r)) for k, _v, s, r in cfg.filament.keys)["density"][1] == "the middle of the sheet range 1.2-1.24"


def test_a_filament_without_a_density_and_without_a_family_row_keeps_the_default(tmp_path):
    bare = {"id": "acme-y", "family": "ZZ", "maker": "Acme", "product": "Y", "aliases": [], "sheet_version": "V1", "E_xy_mpa": 2000,
            "E_z_mpa": None, "bending_modulus_xy_mpa": None, "bending_modulus_z_mpa": None, "strength_xy_mpa": None,
            "strength_z_mpa": None, "strength_kind": "not stated", "elongation_xy_pct": None, "density_g_cm3": None, "hdt_18_c": None,
            "hdt_045_c": None, "conditioned": None, "nu": None, "nozzle_c": None, "bed_c": None, "enclosure": None, "chamber_c_min": None,
            "hardened_nozzle": None, "drying": None, "direct_drive": None, "abrasive": None, "soft": None, "source_url": "https://example.com/y",
            "opened": "2026-10-25", "tag": "maker TDS",
            "note": "E_z_mpa, strength_xy_mpa, strength_z_mpa, elongation_xy_pct, density_g_cm3, hdt_18_c, hdt_045_c, nozzle_c, bed_c, "
                    "enclosure, abrasive, soft: none"}
    cat = mc.parse_catalog({"format": 1, "catalog_version": "t1", "makers": {"Acme": ["acme"]}, "materials": [bare]})
    mat, use = parse_filament_block({"filament": "acme y"}, filament_catalog=cat)
    assert mat == Material(E=2000.0, nu=0.35, density=1.24)
    assert dict((k, (s, r)) for k, _v, s, r in use.keys)["density"] == (
        "assumed", "the catalog has no density for this filament; the default of the problem file")
    # a second row of the same family with a density: now the median of that one row
    other = {**bare, "id": "acme-z", "product": "Z", "density_g_cm3": 1.31, "note": bare["note"].replace("density_g_cm3, ", "")}
    cat = mc.parse_catalog({"format": 1, "catalog_version": "t1", "makers": {"Acme": ["acme"]}, "materials": [bare, other]})
    mat, use = parse_filament_block({"filament": "acme y"}, filament_catalog=cat)
    assert mat.density == 1.31
    assert dict((k, r) for k, _v, _s, r in use.keys)["density"] == "the median of 1 ZZ row of the catalog (acme-z)"


@pytest.mark.parametrize("value, parts", [
    ("Bambu ABZ", ["'material.filament'", "'Bambu ABZ'", "not in the filament catalog", "Did you mean: bambu-lab-", "Use an id from the catalog",
                   "remove material.filament and give E and density", "save_material", "(file: "]),
    ("zzzzzz", ["no name is close to it", "remove material.filament"]),
    ("polymaker pla", ["fits 4 filaments", "polymaker-polylite-pla (Polymaker PolyLite PLA V5.4)", "Use an id from the catalog"]),
    ("PLA", ["fits 9 filaments", "prusament-pla (Prusament PLA V1.1 (27-07-2022))"]),
    ("copa", ["fits 1 filament of", "polymaker-polymide-copa"]),
])
def test_an_unknown_or_ambiguous_filament_is_a_one_line_error(tmp_path, value, parts):
    with pytest.raises(FilamentNotFound) as exc:
        load_config(case(tmp_path, {"filament": value}))
    text = str(exc.value)
    assert "\n" not in text
    for part in parts:
        assert part in text, (part, text)


@pytest.mark.parametrize("value", [5, [], "   ", None, {"a": 1}])
def test_a_filament_that_is_not_a_text_is_a_user_error_with_the_way_on(tmp_path, value):
    with pytest.raises(UserError) as exc:
        load_config(case(tmp_path, {"filament": value}))
    assert not isinstance(exc.value, FilamentNotFound)
    assert "must be the name or the id of a filament as text" in str(exc.value) and "remove material.filament" in str(exc.value)


def test_filament_not_found_is_a_user_error_and_a_bad_value_is_not_it(tmp_path):
    assert issubclass(FilamentNotFound, UserError)
    with pytest.raises(UserError) as exc:
        load_config(case(tmp_path, {"filament": 5}))
    assert not isinstance(exc.value, FilamentNotFound)


def test_a_broken_catalog_is_a_user_error_with_the_way_on(tmp_path, monkeypatch):
    def broken():
        raise mc.CatalogError("filament catalog: topopt/data/materials.yaml is empty")
    monkeypatch.setattr(mc, "_default_catalog", broken)
    with pytest.raises(UserError, match="cannot be checked: filament catalog: topopt/data/materials.yaml is empty. Remove filament"):
        load_config(case(tmp_path, {"filament": ABS}))
    assert load_config(case(tmp_path, {"E": 3000.0}, name="c2")).material.E == 3000.0       # no filament: the catalog is not read


def test_the_other_keys_of_the_block_are_still_checked(tmp_path):
    with pytest.raises(UserError, match="unknown key 'colour' in 'material'"):
        load_config(case(tmp_path, {"filament": ABS, "colour": "red"}))
    with pytest.raises(UserError, match="material.E"):
        load_config(case(tmp_path, {"filament": ABS, "E": -5}, name="c2"))
    with pytest.raises(UserError, match="'material' must be a mapping"):
        load_config(case(tmp_path, [1], name="c3"))


def test_parse_material_block_reads_a_filament_and_parse_filament_block_names_the_sources(work):
    assert parse_material_block({"filament": "bambu abs"}) == Material(E=1960.0, nu=0.35, density=1.05)
    mat, use = parse_filament_block({"filament": "bambu abs", "nu": 0.3})
    assert mat.nu == 0.3 and use.profile.id == "bambu-lab-abs" and use.source("nu") == "yaml"
    assert parse_filament_block({"E": 3000}) == (Material(E=3000.0), None)
    assert parse_filament_block(None) == (Material(), None)
    with pytest.raises(FilamentNotFound):
        parse_material_block({"filament": "nope"})
    with pytest.raises(UserError, match="unknown key"):
        parse_material_block({"colour": 1})


def test_a_catalog_of_the_caller_is_the_one_that_is_searched(tmp_path):
    cat = mc.parse_catalog({"format": 1, "catalog_version": "t1", "makers": {"Acme": ["acme"]}, "materials": [
        {"id": "acme-x", "family": "PLA", "maker": "Acme", "product": "X", "aliases": [], "sheet_version": "V1", "E_xy_mpa": 2000, "E_z_mpa": None,
         "bending_modulus_xy_mpa": None, "bending_modulus_z_mpa": None, "strength_xy_mpa": None, "strength_z_mpa": None, "strength_kind": "not stated", "elongation_xy_pct": None, "density_g_cm3": 1.3,
         "hdt_18_c": None, "hdt_045_c": None, "conditioned": None, "nu": None, "nozzle_c": None, "bed_c": None, "enclosure": None,
         "chamber_c_min": None, "hardened_nozzle": None, "drying": None, "direct_drive": None, "abrasive": None, "soft": None,
         "source_url": "https://example.com/x", "opened": "2026-10-25", "tag": "maker TDS",
         "note": "E_z_mpa, strength_xy_mpa, strength_z_mpa, elongation_xy_pct, hdt_18_c, hdt_045_c, nozzle_c, bed_c, enclosure, abrasive, soft: none"}]})
    mat, use = parse_filament_block({"filament": "acme x"}, filament_catalog=cat)
    assert mat == Material(E=2000.0, nu=0.35, density=1.3) and use.profile.catalog_version == "t1"
    with pytest.raises(FilamentNotFound):
        parse_filament_block({"filament": ABS}, filament_catalog=cat)


# --- a YAML without the key is as before ---------------------------------------------------------------------------------------

def test_without_a_filament_nothing_is_added(tmp_path):
    cfg = load_config(case(tmp_path, {"E": 3000.0}))
    assert cfg.filament is None and cfg.material == Material(E=3000.0)
    assert load_config(case(tmp_path, None, name="c2")).filament is None and Config.__dataclass_fields__["filament"].default is None


def test_the_material_numbers_decide_the_problem_key_and_the_checkpoint_not_the_filament(tmp_path):
    """A filament that gives the same numbers as the YAML gives the same problem: the key and the settings of a checkpoint are equal."""
    from topopt.checkpoint import effective_settings
    from topopt.snapshot import problem_key
    a = load_config(case(tmp_path, {"filament": ABS}, name="a"))
    b = load_config(case(tmp_path, {"E": 2080.9, "nu": 0.35, "density": 1.04}, name="b"))
    assert a.material == b.material and a.filament is not None and b.filament is None
    assert problem_key(a, (10, 5, 5), 2.0) == problem_key(b, (10, 5, 5), 2.0)
    assert asdict(a.material) == asdict(b.material)
    assert effective_settings(a) == effective_settings(b) and "filament" not in json.dumps(effective_settings(a))


def test_the_levers_reference_has_a_card_for_the_key():
    card = next(c for c in levers_cards.CARDS if c.key == "material.filament")
    assert card.group == "yaml" and card.kind == "requirement" and card.layer == 1 and card.confidence == "code"
    for part in ("filament:<id>", "material.E", "find_material", "save_material", "topopt-filaments.yaml", "assumed 0.35", "not a stop"):
        assert part in card.effect + " " + card.values, part


# --- run.json and topopt check -------------------------------------------------------------------------------------------------

def test_run_json_keeps_the_filament_its_facts_the_digest_and_the_source_of_each_number(tmp_path):
    from topopt.cli import main
    p = case(tmp_path, {"filament": "Polymaker PolyLite ABS"}, {"printer_model": "bambu p1s"})
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run = json.loads((out / "run.json").read_text())
    f = run["filament"]
    assert f["id"] == ABS and f["origin"] == "catalog" and f["catalog_version"] == "2026-10-25" and f["name"] == "Polymaker PolyLite ABS"
    assert f["facts"]["E_xy_mpa"] == 2246.6 and f["facts"]["E_z_mpa"] == 2080.9 and f["facts"]["source_url"].startswith("https://")
    assert len(f["profile_sha256"]) == 64 and f["profile_sha256"] == mc.load_catalog().by_id[ABS].facts_sha256()
    assert f["material"]["E"] == {"value": 2080.9, "source": f"filament:{ABS}", "reason": "the lower of the XY and Z modulus"}
    assert f["material"]["nu"] == {"value": 0.35, "source": "assumed", "reason": "no data sheet gives it"}
    assert run["mass_design_space_g"] > 0
    back = FilamentUse.from_dict(json.loads(json.dumps(f)))
    assert back.profile.facts() == mc.load_catalog().by_id[ABS].facts() and back.keys == load_config(p).filament.keys
    assert FilamentUse.from_dict("x") is None and FilamentUse.from_dict({"id": "", "facts": {}}) is None
    assert FilamentUse.from_dict({"id": "x", "facts": {"maker": "A"}, "material": {"E": {"value": "bad", "source": "yaml"}}}).keys == ()


def test_a_run_without_a_filament_has_no_filament_key_in_run_json(tmp_path):
    from topopt.cli import main
    out = tmp_path / "run"
    assert main(["run", str(case(tmp_path, {"E": 3000.0})), "--out", str(out), "--quiet"]) == 0
    assert "filament" not in json.loads((out / "run.json").read_text())


def test_check_data_has_the_material_key_only_with_a_filament_and_names_the_printer_fit(tmp_path):
    from topopt.cli import check_data
    plain = check_data(case(tmp_path, {"E": 3000.0}, {"printer_model": "bambu p1s"}))
    assert "material" not in plain
    d = check_data(case(tmp_path, {"filament": "Bambu ABS"}, {"printer_model": "bambu a1"}, name="c2"))["material"]
    assert d["filament"] == "bambu-lab-abs" and d["printer"] == "bambu-lab-a1" and d["profile_from"] == "catalog 2026-10-25"
    assert d["compat"]["state"] == "no" and d["material"]["E"]["value"] == 1960.0
    assert {r["check"]: r["state"] for r in d["compat"]["reasons"]}["enclosure"] == "no"
    no_printer = check_data(case(tmp_path, {"filament": "Bambu ABS"}, None, name="c3"))["material"]
    assert no_printer["compat"] is None and no_printer["printer"] is None


def test_topopt_check_prints_the_material_lines_as_facts_and_does_not_stop(tmp_path, capsys):
    from topopt.cli import main
    assert main(["check", str(case(tmp_path, {"filament": "Bambu ABS"}, {"printer_model": "bambu a1"}))]) == 0      # a "no" is a fact, not a stop
    out = capsys.readouterr().out.splitlines()
    [line] = [x for x in out if x.startswith("material:")]
    assert "Bambu Lab ABS (filament bambu-lab-abs, catalog 2026-10-25, sheet V3.0)" in line
    assert "E 1960 MPa (filament:bambu-lab-abs, the lower of the XY and Z modulus)" in line
    assert "nu 0.35 (assumed, no data sheet gives it)" in line and "density 1.05 g/cm3 (filament:bambu-lab-abs, the density of the sheet)" in line
    [fit] = [x for x in out if x.startswith("material fit:")]
    assert fit == "material fit:   no: Bambu Lab ABS (bambu-lab-abs) on Bambu Lab A1 (bambu-lab-a1)"
    detail = [x for x in out if x.startswith("                no: enclosure:")]
    assert detail and "is open" in detail[0]
    capsys.readouterr()
    assert main(["check", str(case(tmp_path, {"filament": "Bambu ABS"}, None, name="c2"))]) == 0
    assert "material fit:   not checked: the YAML has no print.printer_model" in capsys.readouterr().out


def test_a_yaml_that_names_an_older_sheet_works_and_check_says_that_a_newer_sheet_exists(tmp_path, capsys):
    from topopt.cli import check_data, main
    cfg = load_config(case(tmp_path, {"filament": "bambu-lab-pla-basic-v2"}))
    assert cfg.material == Material(E=2160.0, nu=0.35, density=1.24) and cfg.filament.profile.superseded_by == "bambu-lab-pla-basic"
    d = check_data(case(tmp_path, {"filament": "bambu-lab-pla-basic-v2"}, {"printer_model": "bambu p1s"}, name="c2"))["material"]
    assert d["superseded_by"] == "bambu-lab-pla-basic"
    current = check_data(case(tmp_path, {"filament": "bambu-lab-pla-basic"}, {"printer_model": "bambu p1s"}, name="c3"))["material"]
    assert "superseded_by" not in current
    assert main(["check", str(case(tmp_path, {"filament": "bambu-lab-pla-basic-v2"}, {"printer_model": "bambu p1s"}, name="c4"))]) == 0
    out = capsys.readouterr().out.splitlines()
    [line] = [x for x in out if x.startswith("material sheet:")]
    assert "a newer sheet of this filament exists: bambu-lab-pla-basic" in line
    main(["check", str(case(tmp_path, {"filament": "bambu-lab-pla-basic"}, {"printer_model": "bambu p1s"}, name="c5"))])
    assert "material sheet:" not in capsys.readouterr().out


def test_topopt_check_of_a_yaml_without_a_filament_prints_no_material_line_and_the_rest_is_unchanged(tmp_path, capsys):
    from topopt.cli import main
    main(["check", str(case(tmp_path, {"E": 2080.9, "nu": 0.35, "density": 1.04}, {"printer_model": "bambu p1s"}, name="plain"))])
    plain = capsys.readouterr().out
    main(["check", str(case(tmp_path, {"filament": ABS}, {"printer_model": "bambu p1s"}, name="fil"))])
    with_filament = capsys.readouterr().out
    assert "material" not in plain.replace("material was", "")
    keep = lambda text: [x for x in text.splitlines() if str(tmp_path) not in x and not x.startswith("material")]   # noqa: E731
    assert keep(plain) == keep(with_filament)


def test_a_user_filament_is_found_by_the_loader_and_run_json_keeps_its_facts(work, tmp_path):
    assert mc.save_user_filament(work, {"id": "acme-pla", "maker": "Acme", "product": "Super PLA", "family": "PLA", "source": "user",
                                        "E_xy_mpa": 2500, "E_z_mpa": 2300, "density_g_cm3": 1.25})["ok"]
    p = case(tmp_path, {"filament": "acme super pla"})
    cfg = load_config(p)
    assert cfg.material == Material(E=2300.0, nu=0.35, density=1.25) and cfg.filament.profile.origin == "user"
    d = json.loads(json.dumps(cfg.filament.to_dict()))
    assert d["origin"] == "user" and d["catalog_version"] is None and d["facts"]["source"] == "user"
    (work / USER_FILE).unlink()
    with pytest.raises(UserError) as exc:                             # the file is gone: the filament is unknown, and the error says where it looked
        load_config(p)
    assert "'material.filament' 'acme super pla'" in str(exc.value) and f"user filament file {work / USER_FILE}: not there" in str(exc.value)


def test_the_two_catalogs_do_not_share_a_user_file(work, tmp_path):
    assert printers.USER_FILE_NAME == "topopt-printers.yaml" and USER_FILE == "topopt-filaments.yaml"
    (work / USER_FILE).write_text("materials: [oops\n")
    cfg = load_config(case(tmp_path, {"filament": ABS}, {"printer_model": "bambu p1s"}))          # a broken filament file does not break the printer
    assert cfg.filament.profile.id == ABS and cfg.print_settings.printer_model == "bambu-lab-p1s"
    with pytest.raises(UserError) as exc:
        load_config(case(tmp_path, {"filament": "nobody x"}, name="c2"))
    assert "user filament file has problems" in str(exc.value) and "not valid YAML" in str(exc.value)


def test_the_example_yaml_text_of_the_reference_loads(tmp_path):
    """The text that the card and the contract show: a filament, a printer, and a key of the YAML that wins."""
    raw = yaml.safe_load("material:\n  filament: Bambu PETG Basic\n  density: 1.27\nprint:\n  printer_model: Prusa MK4S\n")
    cfg = load_config(case(tmp_path, raw["material"], raw["print"]))
    assert cfg.material == Material(E=1120.0, nu=0.35, density=1.27) and cfg.print_settings.printer_model == "prusa-mk4s"
