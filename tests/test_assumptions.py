"""Tests for B-085: the `assumptions` key, the `derived` numbers and run.json."""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from topopt import regions_builder as rb
from topopt.cli import check_data, cmd_check, derived_numbers
from topopt.config import UserError, load_config
from topopt.snapshot import RunWriter

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / ".claude/skills/part-to-topopt/examples"
ASM = ["Shirt mass 0.2 kg, so 1.96 N down", "Thinnest member 2 mm"]


def _model(path):
    """The real optimizer model (AMG setting: no native solver needed)."""
    from topopt.optimize import Model
    from topopt.problem import build_problem
    cfg = load_config(path)
    cfg = dataclasses.replace(cfg, settings=dataclasses.replace(cfg.settings, solver="amg"))
    return Model(build_problem(cfg))


@pytest.fixture(scope="module")
def hook(tmp_path_factory):
    spec = yaml.safe_load((EX / "hook_spec.yaml").read_text())
    spec["assumptions"] = ASM
    out = tmp_path_factory.mktemp("hook")
    r = rb.build_spec(spec, out, say=lambda *a: None)
    return spec, r["yaml"]


def test_spec_key_round_trip(hook):
    _spec, path = hook
    doc = yaml.safe_load(Path(path).read_text())
    assert doc["assumptions"] == ASM
    assert list(doc)[-1] == "assumptions"
    assert list(load_config(path).assumptions) == ASM


def test_no_key_no_line(tmp_path):
    spec = yaml.safe_load((EX / "hook_spec.yaml").read_text())
    spec.pop("assumptions", None)
    r = rb.build_spec(spec, tmp_path, say=lambda *a: None)
    assert "assumptions" not in yaml.safe_load(Path(r["yaml"]).read_text())
    assert load_config(r["yaml"]).assumptions == ()


@pytest.mark.parametrize("bad", ["text", [1], [""], {"a": 1}])
def test_spec_bad_type(tmp_path, bad):
    spec = yaml.safe_load((EX / "hook_spec.yaml").read_text())
    spec["assumptions"] = bad
    with pytest.raises(rb.SpecError, match="assumptions must be a list"):
        rb.build_spec(spec, tmp_path, say=lambda *a: None)


@pytest.mark.parametrize("bad", ["text", [1], [""], {"a": 1}])
def test_config_bad_type_one_line(hook, tmp_path, bad):
    doc = yaml.safe_load(Path(hook[1]).read_text())
    doc["assumptions"] = bad
    p = Path(hook[1]).parent / "bad.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(UserError) as e:
        load_config(p)
    assert "'assumptions' must be a list of non-empty strings" in str(e.value)
    assert "\n" not in str(e.value)


def test_check_data_derived_forces(hook):
    spec, path = hook
    d = check_data(path)
    assert d["assumptions"] == ASM
    dv = d["derived"]
    cases = {c["name"]: c for c in dv["load_cases"]}
    for lc in spec["load_cases"]:
        got = cases[lc["name"]]["forces"]
        assert len(got) == len(lc["forces"])
        for f, g in zip(lc["forces"], got):
            assert g["region"] == f["region"]
            assert g["force_n"] == [float(v) for v in f["force"]]
            assert g["magnitude_n"] == pytest.approx(math.sqrt(sum(v * v for v in f["force"])), abs=1e-3)
    cfg = load_config(path)
    h = cfg.settings.voxel_size
    assert dv["voxel_size_mm"] == h
    assert dv["min_feature_mm"] == max(h, cfg.settings.filter_radius)
    vol = dv["volume"]
    # Independent count: the optimizer's own model (volume = rho.mean() over FREE design voxels,
    # the fixed and load voxels are solid on top).
    m = _model(path)
    n_el, n_free = m.n_el, m.n_design
    assert d["active_voxels"] == n_el and vol["free_design_voxels"] == n_free
    frozen = n_el - n_free
    vf = cfg.settings.volume_fraction
    assert vol["forced_solid_mm3"] == pytest.approx(frozen * h ** 3, rel=1e-3)
    assert vol["active_block_mm3"] == pytest.approx(n_el * h ** 3, rel=1e-3)
    assert vol["target_mass_g"] == pytest.approx(
        (frozen + vf * n_free) * h ** 3 / 1000 * cfg.material.density, rel=1e-3)
    _rho_tilde, rho = m.density(np.full(n_free, vf), 1.0)
    optimizer_g = (frozen + float(rho.mean()) * n_free) * h ** 3 / 1000 * cfg.material.density
    assert vol["target_mass_g"] == pytest.approx(optimizer_g, rel=0.03)
    assert vol["target_mass_g"] < dv["mass_active_block_g"]
    assert dv["grid_shape"] == d["grid"]["shape"]
    assert dv["time_estimate_s"] > 0 and dv["memory_estimate_gib"] > 0


def test_cmd_check_extra_lines_after_existing(hook, capsys):
    class A:
        yaml = str(hook[1])
        json = False
    assert cmd_check(A) == 0
    lines = capsys.readouterr().out.splitlines()
    heads = [ln.split(":")[0] for ln in lines]
    assert heads[:8] == ["name", "grid shape", "active voxels", "free dofs", "iterations", "solver", "solver choice",
                         "time estimate"]                       # P9-4d: `auto` prints what it picked and why, on every machine
    assert heads.index("memory estimate") < heads.index("force")
    for key in ("force", "volume basis", "full block", "min feature", "assumption"):
        assert key in heads
    assert lines[-1] == f"assumption:     {ASM[-1]}"


def test_build_in_workspace_returns_derived(tmp_path):
    spec = yaml.safe_load((EX / "hook_spec.yaml").read_text())
    spec["assumptions"] = ASM
    r = rb.build_in_workspace(spec, tmp_path)
    assert r["ok"]
    assert r["assumptions"] == ASM
    assert r["derived"] == r["check"]["derived"]


def test_derived_from_config_only(hook):
    dv = derived_numbers(load_config(hook[1]))
    assert "volume" not in dv and dv["load_cases"]


def test_run_json_keys(hook, tmp_path):
    from topopt.problem import build_problem
    prob = build_problem(load_config(hook[1]))
    w = RunWriter(tmp_path, quiet=True)
    w.clean()
    m = _model(hook[1])
    n_active = m.n_el
    w.write_run_json(prob, n_active, 1234, "amg")
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["assumptions"] == ASM
    assert run["derived"]["load_cases"][0]["forces"][0]["magnitude_n"] > 0
    assert run["derived"]["volume"]["free_design_voxels"] == m.n_design
    assert run["derived"]["active_voxels"] == n_active == run["n_active"]
    for k in ("format", "name", "shape", "h", "n_active", "load_cases", "regions", "mass_design_space_g"):
        assert k in run   # old keys stay


def test_config_null_assumptions_is_empty(hook):
    doc = yaml.safe_load(Path(hook[1]).read_text())
    doc["assumptions"] = None
    p = Path(hook[1]).parent / "null.yaml"
    p.write_text(yaml.safe_dump(doc))
    assert load_config(p).assumptions == ()


def test_check_json_has_assumptions_and_derived(hook, capsys):
    import json as _json
    class A:
        yaml = str(hook[1])
        json = True
    assert cmd_check(A) == 0
    doc = _json.loads(capsys.readouterr().out)   # one JSON object, no text lines
    assert doc["assumptions"] == ASM and doc["derived"]["load_cases"]
    assert doc["derived"]["time_estimate_s"] == pytest.approx(doc["time_estimate"]["total_s"], abs=1.0)
