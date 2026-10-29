"""F4-46, round 2: the part builder copies every top-level key of a spec that the run YAML takes, and stops on a key that it does not.

`write_outputs` wrote a fixed list of top-level keys. `limits`, `buckling`, `interfaces`, `acceptance` and `bore_checks` of a spec were dropped
without a word, so a requirement of the user was lost and the run was a run without it. One rule now (the same as for `settings`, see
`tests/test_builder_settings.py`): `regions_builder.copied_top_level` copies each key of `config.TOP_REQUIRED` and `config.TOP_OPTIONAL` except
the named builder-owned keys; `check_spec_keys` stops on any other key, and on a `material` key other than E, nu and density, with the text of
the YAML loader.
"""
import copy
from pathlib import Path

import pytest
import yaml

from test_builder_settings import SPEC, spec_with, written
from topopt import config
from topopt import regions_builder as rb
from topopt.cli import check_data
from topopt.config import UserError, load_config

TOP_RUN_KEYS = [*config.TOP_REQUIRED, *config.TOP_OPTIONAL]
# The top-level keys that the builder dropped before round 2 (probe of main 74a1e46: each was missing from the written YAML), with a block that the
# loader takes. The values come from the public examples and the tests of the loader: limits (CONTRACTS.md section 31), buckling (examples/hook/
# hook_buckling.yaml), interfaces (examples/interfaces/bracket.yaml), acceptance and bore_checks (tests/test_ratchet.py, tests/test_bore_checks_config.py).
SRC = "test source"
DROPPED_TOP = {
    "limits": {"displacement": [{"name": "tip_drop", "region": "tip", "limit_mm": 1.5, "direction": [0, 0, -1], "sense": "max", "cases": ["static"]}]},
    "buckling": {"min_factor": 150},
    "interfaces": [{"name": "pin_bore", "kind": "bore", "point": [28.0, 10.0, 20.0], "axis": [0.0, 0.0, -1.0], "radius_mm": 3.0,
                    "length_mm": 24.0, "stock_mm": 0.4}],
    "acceptance": {"rules": [{"rule": "watertight"}]},
    "bore_checks": {"bores": [{"name": "b1", "point": [5, 5, 12], "axis": [0, 0, -1], "radius_mm": 2.0, "length_mm": 8.0,
                               "radius_mm_source": SRC, "length_mm_source": SRC}],
                    "goodman": {"materials": {"M": {"su_mpa": 1000, "sy_mpa": 900, "se_mpa": 300, "source": SRC}}}},
}
# The keys that the builder copied before, in the order that it wrote them.
COPIED_BEFORE = ("springs", "joints", "assumptions", "mirror", "print")
OLD_TOP_ORDER = ["name", "regions", "load_cases", "material", "settings"]
OLD_YAML = """\
name: bracket
regions:
  design: design.stl
  fixed: [wall.stl]
  keep_out: []
  load: {tip: tip.stl}
load_cases:
- name: static
  weight: 1.0
  forces:
  - region: tip
    force: [0, 0, -20.0]
material: {E: 2100.0, nu: 0.38, density: 1.24}
settings: {voxel_size: 2.0, volume_fraction: 0.3, iterations: 80, penal: 4.0, solver: cholesky,
  min_member_mm: 3.0}
springs:
- region: tip
  stiffness: [10, 10, 10]
joints:
- region: tip
  to: [60, 0, 0]
  section: {kind: round, d: 2}
assumptions: [a one, a two]
mirror:
- {normal: y, at_mm: 8.0}
print: {bed_mm: 180}
"""   # the file that main 74a1e46 writes for OLD_SPEC
COPIED_VALUES = {"springs": [{"region": "tip", "stiffness": [10, 10, 10]}],
                 "joints": [{"region": "tip", "to": [60, 0, 0], "section": {"kind": "round", "d": 2}}],
                 "assumptions": ["a one", "a two"], "mirror": [{"normal": "y", "at_mm": 8.0}], "print": {"bed_mm": 180}}


def old_spec(**over):
    s = spec_with(penal=4.0, solver="cholesky", min_member_mm=3.0)
    s["material"] = {"E": 2100, "nu": 0.38}
    s.update(COPIED_VALUES)
    s.update(over)
    return s


def top_keys(spec, tmp_path, name="out"):
    path, _ = written(tmp_path, spec, name)
    return path, yaml.safe_load(path.read_text())


# ---- acceptance: limits and buckling of a spec reach the YAML -------------------------------------------------------------------
def test_limits_and_buckling_reach_the_yaml_the_loader_and_the_check(tmp_path, capsys):
    spec = spec_with()
    spec["limits"], spec["buckling"] = DROPPED_TOP["limits"], DROPPED_TOP["buckling"]
    r = rb.build_spec(spec, tmp_path / "out")
    assert r["ok"], capsys.readouterr().out
    doc = yaml.safe_load(r["yaml"].read_text())
    assert doc["limits"] == DROPPED_TOP["limits"] and doc["buckling"] == {"min_factor": 150}
    cfg = load_config(r["yaml"])
    (row,) = cfg.limits.displacement
    assert (row.name, row.region, row.limit_mm, row.sense, row.cases) == ("tip_drop", "tip", 1.5, "max", ("static",))
    assert cfg.buckling.min_factor == 150.0
    assert check_data(r["yaml"])["buckling_row"]["min_factor"] == 150.0                # `topopt check` says it
    plain = rb.build_spec(SPEC, tmp_path / "plain")
    assert not {"limits", "buckling"} & set(yaml.safe_load(plain["yaml"].read_text()))   # no key in the spec, no key in the YAML


def test_limits_and_buckling_reach_the_yaml_of_the_tool(tmp_path):
    spec = spec_with()
    spec["limits"], spec["buckling"] = DROPPED_TOP["limits"], DROPPED_TOP["buckling"]
    r = rb.build_in_workspace(spec, tmp_path)
    assert r["ok"], r
    doc = yaml.safe_load((tmp_path / r["yaml_rel"]).read_text())
    assert "limits" in doc and doc["buckling"] == {"min_factor": 150}
    assert r["check"]["buckling_row"]["min_factor"] == 150.0


@pytest.mark.parametrize("key", DROPPED_TOP)
def test_a_top_level_key_that_was_dropped_reaches_the_yaml_and_the_loader(tmp_path, key):
    spec = spec_with()
    spec[key] = DROPPED_TOP[key]
    path, doc = top_keys(spec, tmp_path)
    assert doc[key] == DROPPED_TOP[key]
    cfg = load_config(path)
    read = {"limits": lambda c: c.limits.displacement, "buckling": lambda c: c.buckling.min_factor, "interfaces": lambda c: c.interfaces,
            "acceptance": lambda c: c.acceptance, "bore_checks": lambda c: c.bore_checks}
    assert read[key](cfg)                                                       # the engine reads the block


def test_the_dropped_top_level_keys_are_run_keys_that_the_builder_does_not_own():
    assert set(DROPPED_TOP) <= set(TOP_RUN_KEYS) and set(DROPPED_TOP).isdisjoint(rb.BUILDER_OWNED_TOP_LEVEL)
    assert set(DROPPED_TOP).isdisjoint(COPIED_BEFORE)


# ---- an unknown key stops the build ----------------------------------------------------------------------------------------------
def test_an_unknown_top_level_key_stops_the_build_with_the_text_of_the_loader(tmp_path):
    spec = spec_with()
    spec["limit"] = DROPPED_TOP["limits"]                                       # a typo of `limits`
    with pytest.raises(rb.SpecError) as e:
        rb.build_spec(spec, tmp_path / "out")
    text = str(e.value)
    assert text.startswith("unknown key 'limit' in 'top level'. Known keys: name, regions, load_cases, settings, material")
    assert "limits, buckling" in text and text.count("fixed_axes") == 1         # the keys that the spec takes, also the spec-only key
    assert "Options: use a known key, or remove 'limit'." in text
    assert not (tmp_path / "out").exists()                                      # stopped before any file is written
    (tmp_path / "t").mkdir()
    path, _ = written(tmp_path, SPEC, "t")                                      # the loader's words for the same key in a YAML
    doc = yaml.safe_load(path.read_text())
    doc["limit"] = DROPPED_TOP["limits"]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(UserError) as u:
        load_config(path)
    tail = "Options: use a known key, or remove 'limit'. " + config.LEVERS_HELP
    assert "unknown key 'limit' in 'top level'. Known keys: name, regions, load_cases, settings, material" in str(u.value)
    assert text.endswith(tail) and tail in str(u.value)


def test_an_unknown_top_level_key_is_a_spec_error_on_every_route(tmp_path, capsys):
    spec = spec_with()
    spec["limit"] = DROPPED_TOP["limits"]
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec))
    assert rb.main([str(spec_file), "--out", str(tmp_path / "cli")]) == 1
    printed = capsys.readouterr().out
    assert printed.startswith("SPEC ERROR: unknown key 'limit' in 'top level'.") and printed.count("\n") == 1
    r = rb.build_in_workspace(spec, tmp_path / "ws")
    assert r["ok"] is False and r["error"].startswith("unknown key 'limit' in 'top level'.") and "\n" not in r["error"]
    assert not (tmp_path / "ws" / "runs").exists()
    assert rb.build_in_workspace(yaml.safe_dump(spec), tmp_path / "ws2")["error"] == r["error"]       # a spec as YAML text


def test_a_material_key_that_the_yaml_does_not_take_stops_the_build(tmp_path):
    spec = spec_with()
    spec["material"] = {"E": 2100, "name": "PETG"}
    with pytest.raises(rb.SpecError) as e:
        rb.build_spec(spec, tmp_path / "out")
    assert str(e.value).startswith("unknown key 'name' in 'material'. Known keys: E, nu, density, filament. Options: use a known key, or remove 'name'.")
    assert not (tmp_path / "out").exists()
    with pytest.raises(rb.SpecError):                                           # a direct call of the writer cannot drop it either
        rb.write_outputs(spec, rb.build_all(spec), tmp_path / "direct", 2.0)
    assert not (tmp_path / "direct").exists()
    r = rb.build_in_workspace(spec, tmp_path / "ws")
    assert r["ok"] is False and r["error"].startswith("unknown key 'name' in 'material'.")


def test_a_block_of_the_wrong_type_is_still_a_shape_error_and_not_a_key_error(tmp_path):
    for field in ("material", "settings"):
        spec = spec_with()
        spec[field] = ["E"]
        with pytest.raises(rb.SpecError, match=f"{field} must be a mapping"):
            rb.build_spec(spec, tmp_path / field)


def test_fixed_axes_is_a_spec_only_key(tmp_path):
    spec = spec_with()
    spec["fixed_axes"] = {"wall": ["y"]}
    _, doc = top_keys(spec, tmp_path)
    assert "fixed_axes" not in doc and doc["regions"]["fixed"] == [{"file": "wall.stl", "axes": ["y"]}]
    assert "fixed_axes" in rb.SPEC_ONLY_TOP_LEVEL and "fixed_axes" not in TOP_RUN_KEYS


# ---- a guard against drift (the top level) -----------------------------------------------------------------------------------------
def test_every_top_level_run_key_is_copied_or_builder_owned():
    """Fails when a key of the config is neither copied by the builder nor named in `BUILDER_OWNED_TOP_LEVEL`."""
    sentinel = object()
    for key in TOP_RUN_KEYS:
        if key in rb.BUILDER_OWNED_TOP_LEVEL:
            continue
        assert rb.copied_top_level({key: sentinel}) == {key: sentinel}, (
            f"{key} is a top-level key of the run YAML but the builder does not copy it: make it copy the key, or name it in "
            "regions_builder.BUILDER_OWNED_TOP_LEVEL with the reason")


def test_the_top_level_sets_agree_with_the_config():
    assert set(rb.BUILDER_OWNED_TOP_LEVEL) <= set(TOP_RUN_KEYS), "a builder-owned key that the config does not take is never written"
    assert set(rb.BUILDER_OWNED_TOP_LEVEL) >= set(config.TOP_REQUIRED), "a required key of the run YAML is written by the builder"
    assert not set(rb.SPEC_ONLY_TOP_LEVEL) & set(TOP_RUN_KEYS), "a key that the config takes now must be copied: remove it from SPEC_ONLY_TOP_LEVEL"
    assert set(rb.BUILDER_OWNED_MATERIAL) <= set(config.MATERIAL_OPTIONAL)
    for table in (rb.BUILDER_OWNED_TOP_LEVEL, rb.SPEC_ONLY_TOP_LEVEL, rb.BUILDER_OWNED_MATERIAL):
        assert all(isinstance(why, str) and why.strip() for why in table.values())      # one reason each


def test_every_material_key_is_written_by_the_builder():
    for key in config.MATERIAL_OPTIONAL:
        assert key in rb.BUILDER_OWNED_MATERIAL or rb.copied_material({key: 1}) == {key: 1}


def test_a_top_level_key_that_the_config_gets_later_is_copied_with_no_change_in_the_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOP_OPTIONAL", (*config.TOP_OPTIONAL, "future_block"))
    spec = spec_with()
    spec.update(future_block={"a": 1}, buckling=DROPPED_TOP["buckling"])
    _, doc = top_keys(spec, tmp_path)
    assert doc["future_block"] == {"a": 1} and list(doc)[-2:] == ["buckling", "future_block"]      # after the keys of the config in their order


def test_a_material_key_that_the_config_gets_later_is_copied_with_no_change_in_the_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_OPTIONAL", (*config.MATERIAL_OPTIONAL, "future_property"))
    spec = spec_with()
    spec["material"] = {"E": 2100, "future_property": 3}
    _, doc = top_keys(spec, tmp_path)
    assert doc["material"] == {"E": 2100.0, "nu": 0.35, "density": 1.24, "future_property": 3}


def test_the_writer_writes_every_top_level_run_key_that_is_not_builder_owned(tmp_path):
    every = {k: {"value": k} for k in TOP_RUN_KEYS if k not in rb.BUILDER_OWNED_TOP_LEVEL}
    spec = {**spec_with(), **every}
    path = rb.write_outputs(spec, rb.build_all(spec), tmp_path / "out", 2.0)
    doc = yaml.safe_load(path.read_text())
    assert {k: doc[k] for k in every} == every
    assert set(doc) == set(every) | set(rb.BUILDER_OWNED_TOP_LEVEL)


# ---- a spec without the new keys builds as before --------------------------------------------------------------------------------
def test_the_yaml_of_a_spec_with_the_old_keys_is_byte_equal_to_the_old_one(tmp_path):
    path, _ = top_keys(old_spec(), tmp_path)
    assert path.read_text() == OLD_YAML


def test_the_old_top_level_keys_keep_their_order_whatever_the_order_of_the_spec(tmp_path):
    base = old_spec()
    reversed_spec = {k: v for k, v in base.items() if k not in COPIED_BEFORE}
    for k in reversed(COPIED_BEFORE):
        reversed_spec[k] = base[k]
    assert [k for k in reversed_spec if k in COPIED_BEFORE] == list(reversed(COPIED_BEFORE))
    path, doc = top_keys(reversed_spec, tmp_path)
    assert list(doc) == [*OLD_TOP_ORDER, *COPIED_BEFORE] and path.read_text() == OLD_YAML


def test_an_empty_or_null_old_key_is_not_written_and_an_empty_print_is(tmp_path):
    _, doc = top_keys(old_spec(springs=[], joints=[], assumptions=[], mirror=[], print={}), tmp_path, "empty")
    assert list(doc) == [*OLD_TOP_ORDER, "print"] and doc["print"] == {}
    _, doc = top_keys(old_spec(springs=None, joints=None, assumptions=None, mirror=None, print=None), tmp_path, "null")
    assert list(doc) == OLD_TOP_ORDER
    spec = spec_with()
    spec["limits"] = None                                                       # a null of a new key is the same as no key
    assert "limits" not in top_keys(spec, tmp_path, "nulllimits")[1]


@pytest.mark.parametrize("example", ["hook_spec.yaml", "ladle_blind_spec.yaml"])
def test_an_example_spec_has_the_top_level_keys_it_had(tmp_path, example):
    spec = yaml.safe_load((Path(__file__).resolve().parents[1] / ".claude/skills/part-to-topopt/examples" / example).read_text())
    _, doc = top_keys(spec, tmp_path)
    assert list(doc) == OLD_TOP_ORDER
    material = {"E": 3500.0, "nu": 0.35, "density": 1.24, **{k: float(v) for k, v in (spec.get("material") or {}).items()}}
    assert doc["material"] == material
