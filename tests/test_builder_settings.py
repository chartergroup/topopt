"""F4-46: the part builder copies every `settings` key of a spec that the run YAML takes, and stops on a key that it does not.

The builder copied a fixed list of keys. `settings.objective: mass` (and `stress_*`, `robust`, `eta_eroded`, `gpu_*`,
`part_only_min_film_mm`) was dropped without a word, so a spec for a mass run gave a stiffness run. One rule now:
`regions_builder.copied_settings` copies each key of `config.SETTINGS_REQUIRED` and `config.SETTINGS_OPTIONAL` except the named
builder-owned keys; `check_settings_keys` stops on any other key with the text of the YAML loader.
"""
import copy
from pathlib import Path

import pytest
import yaml

from topopt import config
from topopt import regions_builder as rb
from topopt.cli import check_data
from topopt.config import UserError, load_config

SPEC = {
    "name": "bracket",
    "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [40, 16, 24]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 16, 24]}}]},
        "load": {"tip": [{"box": {"min": [32, 4, 0], "max": [40, 12, 8]}}]},
    },
    "load_cases": [{"name": "static", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -20.0]}]}],
    "settings": {"volume_fraction": 0.3, "voxel_size": 2.0},
}
# The keys that the builder did not copy before F4-46 (probe of main 74a1e46: each was missing from the written YAML), with a value that
# the loader takes and that is not its default. The field of `Settings` has the name of the key.
DROPPED_BEFORE = {
    "objective": "mass", "stress_q": 20.0, "stress_power": 0.6, "stress_scale": 2.5, "stress_step_acceptance": False,
    "robust": "eroded", "eta_eroded": 0.6, "gpu_device": "pocl", "gpu_case": "B", "gpu_tol": 1e-5, "gpu_backward_error": 1e-12,
    "gpu_max_iter": 500, "part_only_min_film_mm": 0.8,
}
# The keys that the builder copied before F4-46, in the order that it wrote them. A built YAML keeps this order.
COPIED_BEFORE = ("filter_radius", "penal", "move_limit", "solver", "max_grid_voxels", "max_active_voxels", "beta_schedule", "case_combine",
                 "mesh_method", "objective_scale", "min_member_mm", "filter_solid_neighbours", "stop_rules", "perimeter_max_mm2")
RUN_KEYS = [*config.KEY_BLOCKS["settings"][0], *config.KEY_BLOCKS["settings"][1]]


def spec_with(**settings):
    s = copy.deepcopy(SPEC)
    s["settings"].update(settings)
    return s


def written(tmp_path, spec, name="out"):
    """Write the STLs and the YAML of `spec` (no check). Return (the path of the YAML, its `settings` block)."""
    path = rb.write_outputs(spec, rb.build_all(spec), tmp_path / name, 2.0)
    return path, yaml.safe_load(path.read_text())["settings"]


# ---- acceptance 1: objective: mass reaches the YAML ---------------------------------------------------------------------------
def test_objective_mass_reaches_the_yaml_and_the_check(tmp_path, capsys):
    out = tmp_path / "out"
    r = rb.build_spec(spec_with(objective="mass"), out)
    assert r["ok"], capsys.readouterr().out
    doc = yaml.safe_load(r["yaml"].read_text())
    assert doc["settings"]["objective"] == "mass"
    assert load_config(r["yaml"]).settings.objective == "mass"                 # the engine reads it
    assert check_data(r["yaml"])["derived"]["objective"] == "mass"             # and `topopt check` says it
    r0 = rb.build_spec(SPEC, tmp_path / "plain")
    assert "objective" not in yaml.safe_load(r0["yaml"].read_text())["settings"]      # no key in the spec, no key in the YAML


def test_objective_mass_reaches_the_yaml_of_the_tool(tmp_path):
    r = rb.build_in_workspace(spec_with(objective="mass"), tmp_path)
    assert r["ok"], r
    assert yaml.safe_load((tmp_path / r["yaml_rel"]).read_text())["settings"]["objective"] == "mass"
    assert r["check"]["derived"]["objective"] == "mass"


# ---- acceptance 2: no settings key is dropped without a word ----------------------------------------------------------------
@pytest.mark.parametrize("key", DROPPED_BEFORE)
def test_a_key_that_was_dropped_reaches_the_yaml_and_the_loader(tmp_path, key):
    path, st = written(tmp_path, spec_with(**{key: DROPPED_BEFORE[key]}))
    assert st[key] == DROPPED_BEFORE[key]
    value = getattr(load_config(path).settings, key)
    assert value == DROPPED_BEFORE[key] or str(value) == str(DROPPED_BEFORE[key])        # gpu_device is a text in `Settings`


def test_the_test_lists_are_run_keys_and_do_not_overlap():
    assert set(DROPPED_BEFORE) <= set(RUN_KEYS) and set(COPIED_BEFORE) <= set(RUN_KEYS)
    assert not set(DROPPED_BEFORE) & set(COPIED_BEFORE) and not (set(DROPPED_BEFORE) | set(COPIED_BEFORE)) & set(rb.BUILDER_OWNED_SETTINGS)


def test_an_unknown_settings_key_stops_the_build_with_the_text_of_the_loader(tmp_path, capsys):
    spec = spec_with(objectve="mass")                                           # a typo
    with pytest.raises(rb.SpecError) as e:
        rb.build_spec(spec, tmp_path / "out")
    text = str(e.value)
    assert text.startswith("unknown key 'objectve' in 'settings'. Known keys: voxel_size, iterations, volume_fraction")
    assert "objective" in text and "target_active, allow_large_grid" in text    # the keys that the spec takes, also the two of the builder
    assert "Options: use a known key, or remove 'objectve'." in text
    assert not (tmp_path / "out").exists()                                      # stopped before any file is written
    # The words of the YAML loader for the same key in a YAML: one wording.
    (tmp_path / "t").mkdir()
    path, _ = written(tmp_path, SPEC, "t")
    doc = yaml.safe_load(path.read_text())
    doc["settings"]["objectve"] = "mass"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(UserError) as u:
        load_config(path)
    tail = "Options: use a known key, or remove 'objectve'. " + config.LEVERS_HELP
    assert "unknown key 'objectve' in 'settings'. Known keys: voxel_size, iterations, volume_fraction" in str(u.value)
    assert text.endswith(tail) and tail in str(u.value)


def test_an_unknown_settings_key_is_a_spec_error_on_every_route(tmp_path, capsys):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec_with(objectve="mass")))
    assert rb.main([str(spec_file), "--out", str(tmp_path / "cli")]) == 1
    printed = capsys.readouterr().out
    assert printed.startswith("SPEC ERROR: unknown key 'objectve' in 'settings'.") and "\n" == printed[-1] and printed.count("\n") == 1
    r = rb.build_in_workspace(spec_with(objectve="mass"), tmp_path / "ws")
    assert r["ok"] is False and r["error"].startswith("unknown key 'objectve' in 'settings'.") and "\n" not in r["error"]
    assert not (tmp_path / "ws" / "runs" / "bracket").exists()
    # a spec text goes the same way
    assert rb.build_in_workspace(yaml.safe_dump(spec_with(objectve="mass")), tmp_path / "ws2")["error"] == r["error"]       # a spec as YAML text


def test_the_two_builder_only_keys_are_read_and_not_written(tmp_path):
    assert rb.copied_settings({"target_active": 5000, "allow_large_grid": True, "voxel_size": 2.0, "iterations": 5,
                               "volume_fraction": 0.3, "objective": "mass"}) == {"objective": "mass"}
    path, st = written(tmp_path, spec_with(target_active=5000, allow_large_grid=True))
    assert list(st) == ["voxel_size", "volume_fraction", "iterations"]
    assert load_config(path).settings.volume_fraction == 0.3


def test_a_value_is_copied_as_given_and_the_loader_refuses_a_bad_one(tmp_path, capsys):
    ok = rb.build_spec(spec_with(objective="mass", stress_q=20.0), tmp_path / "ok")
    assert ok["ok"]
    bad = rb.build_spec(spec_with(objective="weight"), tmp_path / "bad")       # the YAML has it as given, check_yaml says why it is refused
    assert not bad["ok"] and "'settings.objective' must be compliance or mass, got 'weight'" in capsys.readouterr().out
    assert yaml.safe_load(bad["yaml"].read_text())["settings"]["objective"] == "weight"


# ---- decision 3: a guard against drift --------------------------------------------------------------------------------------
def test_every_run_settings_key_is_copied_or_builder_owned():
    """Fails when a key of the config is neither copied by the builder nor named in `BUILDER_OWNED_SETTINGS`."""
    sentinel = object()
    for key in RUN_KEYS:
        if key in rb.BUILDER_OWNED_SETTINGS:
            continue
        assert rb.copied_settings({key: sentinel}) == {key: sentinel}, (
            f"settings.{key} is a run key but the builder does not copy it: make it copy the key, or name it in "
            "regions_builder.BUILDER_OWNED_SETTINGS with the reason")


def test_the_builder_owned_and_builder_only_sets_agree_with_the_config():
    assert set(rb.BUILDER_OWNED_SETTINGS) <= set(RUN_KEYS), "a builder-owned key that the config does not take is never written"
    assert not set(rb.BUILDER_ONLY_SETTINGS) & set(RUN_KEYS), "a key that the config takes now must be copied: remove it from BUILDER_ONLY_SETTINGS"
    assert set(rb.BUILDER_OWNED_SETTINGS) >= set(config.SETTINGS_REQUIRED), "a required key of the run YAML is written by the builder"
    for table in (rb.BUILDER_OWNED_SETTINGS, rb.BUILDER_ONLY_SETTINGS):
        assert all(isinstance(why, str) and why.strip() for why in table.values())      # one reason each


def test_a_key_that_the_config_gets_later_is_copied_with_no_change_in_the_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_OPTIONAL", (*config.SETTINGS_OPTIONAL, "future_key"))
    path, st = written(tmp_path, spec_with(future_key=7, objective="mass"))
    assert st["future_key"] == 7 and st["objective"] == "mass"
    assert list(st)[-2:] == ["objective", "future_key"]                         # after the keys of the config in their order


def test_the_writer_writes_every_run_key_that_is_not_builder_owned(tmp_path):
    every = {k: f"value of {k}" for k in RUN_KEYS if k not in rb.BUILDER_OWNED_SETTINGS}
    path = rb.write_outputs(spec_with(**every), rb.build_all(SPEC), tmp_path / "out", 2.0)
    st = yaml.safe_load(path.read_text())["settings"]
    assert {k: st[k] for k in every} == every
    assert set(st) == set(every) | {"voxel_size", "volume_fraction", "iterations"}


# ---- decision 4: a spec without the new keys builds as before ---------------------------------------------------------------
def test_the_keys_that_were_copied_before_keep_their_order_and_value(tmp_path):
    given = {"filter_radius": 3.0, "penal": 4.0, "move_limit": 0.1, "solver": "cholesky", "max_grid_voxels": 4000000,
             "max_active_voxels": 90000, "beta_schedule": [[0, 1], [30, 2]], "case_combine": "max", "mesh_method": "blur",
             "objective_scale": "first", "min_member_mm": 3.0, "filter_solid_neighbours": "ignore",
             "stop_rules": {"enabled": True, "window": 6}, "perimeter_max_mm2": 4000.0}
    assert list(given) == list(COPIED_BEFORE)
    for order in (list(given), list(reversed(given))):                          # the order of the spec does not matter
        _, st = written(tmp_path, spec_with(**{k: given[k] for k in order}), f"o{order[0]}")
        assert list(st) == ["voxel_size", "volume_fraction", "iterations", *COPIED_BEFORE]
        assert st == {"voxel_size": 2.0, "volume_fraction": 0.3, "iterations": 80, **given}


@pytest.mark.parametrize("example", ["hook_spec.yaml", "ladle_blind_spec.yaml"])
def test_an_example_spec_has_the_settings_block_it_had(tmp_path, example):
    spec = yaml.safe_load((Path(__file__).resolve().parents[1] / ".claude/skills/part-to-topopt/examples" / example).read_text())
    _, st = written(tmp_path, spec)
    assert list(st) == ["voxel_size", "volume_fraction", "iterations"]
    assert st["volume_fraction"] == float(spec["settings"]["volume_fraction"]) and st["iterations"] == spec["settings"]["iterations"]


def test_a_spec_with_a_mass_in_grams_writes_it_in_place_of_the_fraction(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec["settings"] = {"voxel_size": 2.0, "target_mass_g": 12.0, "objective": "compliance"}
    _, st = written(tmp_path, spec)
    assert st == {"voxel_size": 2.0, "target_mass_g": 12.0, "iterations": 80, "objective": "compliance"}
