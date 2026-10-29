"""Tests for the MCP tool build_regions (topopt/regions_builder.py)."""
import importlib.util
from pathlib import Path

import pytest
import trimesh
import yaml

pytest.importorskip("mcp")
from topopt.cli import check_data  # noqa: E402
from topopt_mcp import server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / ".claude/skills/part-to-topopt/examples"
_s = importlib.util.spec_from_file_location(
    "build_regions_wrapper", ROOT / ".claude/skills/part-to-topopt/build_regions.py")
wrapper = importlib.util.module_from_spec(_s)
_s.loader.exec_module(wrapper)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    return tmp_path / "ws"


@pytest.mark.parametrize("fname", ["hook_spec.yaml", "ladle_blind_spec.yaml"])
def test_examples(work, tmp_path, capsys, fname):
    spec = yaml.safe_load((EX / fname).read_text())
    r = server.build_regions(spec)
    assert r["ok"], r
    assert r["run_dir_rel"] == f"runs/{spec['name']}"
    for f in r["files_rel"]:
        assert (work / f).is_file()
    assert r["yaml_rel"] in r["files_rel"]
    assert any(f.endswith("design.stl") for f in r["files_rel"])
    for f in r["files_rel"]:
        if f.endswith(".stl"):
            assert trimesh.load(work / f).is_watertight
    d = check_data(work / r["yaml_rel"])
    assert d["grid"] == r["check"]["grid"] and d["voxel_counts"] == r["check"]["voxel_counts"]
    assert r["check"]["connectivity"]["components"] == 1
    assert r["builder_ok"] and r["problems"] == []

    # CLI wrapper prints the same lines and picks the same voxel size
    out = tmp_path / "cli"
    assert wrapper.main([str(EX / fname), "--out", str(out)]) == 0
    printed = capsys.readouterr().out.strip().splitlines()
    norm = lambda ls, o: [ln.replace(str(o), "OUT") for ln in ls]
    assert norm(printed, out) == norm(r["report"], work / r["run_dir_rel"])
    doc_cli = yaml.safe_load((out / f"{spec['name']}.yaml").read_text())
    doc_tool = yaml.safe_load((work / r["yaml_rel"]).read_text())
    assert doc_cli == doc_tool
    assert doc_tool["settings"]["voxel_size"] == r["voxel_size"]


def test_yaml_string_and_name(work):
    text = (EX / "hook_spec.yaml").read_text()
    r = server.build_regions(text, name="my_hook")
    assert r["ok"] and r["yaml_rel"] == "runs/my_hook/my_hook.yaml"
    assert (work / "runs/my_hook/my_hook.yaml").is_file()
    again = server.build_regions(text, name="my_hook")
    assert again["overwrote_existing_folder"] and not r["overwrote_existing_folder"]


def test_regions_union_no_xor(work):
    spec = {"name": "u", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [40, 20, 20]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 20]}}]},
        "load": {"tip": [{"box": {"min": [30, 5, 0], "max": [40, 15, 12]}},
                         {"box": {"min": [34, 8, 0], "max": [44, 18, 12]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0,
                        "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
        "settings": {"voxel_size": 2.0}}
    r = server.build_regions(spec)
    assert r["ok"], r
    tip = trimesh.load(work / "runs/u/tip.stl")
    assert tip.is_watertight
    # union volume = 10*10*12 + 10*10*12 - overlap 6*7*12
    assert abs(tip.volume - (1200 + 1200 - 6 * 7 * 12)) < 1.0


@pytest.mark.parametrize("bad", [
    {"regions": {"design": [{"box": {"min": [0, 0, 0], "max": [1, 1, 1]}}]}},          # no load_cases
    {"regions": {"nope": 1}, "load_cases": [{}]},
    {"regions": {"design": [{"torus": {}}], "fixed": {}, "load": {}}, "load_cases": [{}]},
    "a: [unclosed",
    "- just\n- a list",
    42,
])
def test_spec_errors_are_one_line(work, bad):
    r = server.build_regions(bad)
    assert r["ok"] is False and "\n" not in r["error"] and r["error"]


def test_bad_name(work):
    r = server.build_regions(yaml.safe_load((EX / "hook_spec.yaml").read_text()), name="../x")
    assert r["ok"] is False
    assert not (work.parent / "x").exists()


def test_problem_is_fact_not_block(work):
    spec = yaml.safe_load((EX / "hook_spec.yaml").read_text())
    spec["regions"]["load"] = {"far": [{"box": {"min": [500, 0, 0], "max": [510, 10, 10]}}]}
    spec["load_cases"] = [{"name": "a", "weight": 1.0, "forces": [{"region": "far", "force": [0, 0, -1.0]}]}]
    r = server.build_regions(spec)
    assert r["ok"] is False and r["builder_ok"] is False      # P9-8: a build that did not build is not ok
    assert any("separate pieces" in p for p in r["problems"])
    assert (work / r["yaml_rel"]).is_file()                   # the files stay: a problem is a fact, not a block


def test_a_build_with_a_wrong_setting_is_not_ok_and_keeps_every_other_key(work):
    """P9-8 (T10): `volume_fraction: 1.5` fails the check of the written YAML. The reply was `ok: true` with `builder_ok: false`;
    a client that reads `ok` then reported a success. Now `ok` is false and the keys stay as they were (`builder_ok`, `problems`)."""
    spec = yaml.safe_load((EX / "hook_spec.yaml").read_text())
    spec["settings"]["volume_fraction"] = 1.5
    r = server.build_regions(spec)
    assert r["ok"] is False and r["builder_ok"] is False
    assert any("volume_fraction" in p for p in r["problems"])
    assert (work / r["yaml_rel"]).is_file() and r["report"] and r["check"] is None and r["check_error"]
    for key in ("name", "run_dir_rel", "yaml_rel", "files_rel", "other_files_rel", "overwrote_existing_folder", "voxel_size",
                "voxel_auto", "retries", "builder_warnings", "warnings", "assumptions", "derived"):
        assert key in r, key
    from topopt_mcp import replies      # P9-11: the list of the keys of the reply moved to the guide topopt://replies/build_regions
    assert "topopt://replies/build_regions" in server.build_regions.__doc__
    assert "ok is false also when the builder fails (builder_ok false)" in " ".join(server.build_regions.__doc__.split())   # the ok contract stays in the text
    assert "ok (false when builder_ok is false)" in " ".join(replies.guide("build_regions").split())


def test_the_description_says_that_a_stop_rules_mapping_turns_the_rules_on_and_the_budget_ends_the_run():
    """P9-8 review MJ1: `time_budget_s` of a spec is a stop rule. The tool text says so in one clause (key text stays in 2,048 characters)."""
    doc = " ".join(server.build_regions.__doc__.split())
    clause = ("stop_rules (as in the YAML: a mapping turns the stop rules on, and time_budget_s, the user's own time budget, ends the run "
              "at the first pass after it; enabled: false keeps the rules off; the check warns when the estimate is above the budget)")
    assert clause in doc and doc.index(clause) + len(clause) < 2048 and len(server.build_regions.__doc__) <= 4096


def test_a_good_build_is_ok_and_builder_ok_and_the_two_agree(work):
    r = server.build_regions(_hook())
    assert r["ok"] is True and r["builder_ok"] is True and r["problems"] == []


def test_volume_fraction_basis_is_the_free_design_voxels(work):
    """P1 (F3-2): the YAML keeps the spec value, and the reply and the tool text say the same basis as CONTRACTS.md."""
    spec = yaml.safe_load((EX / "hook_spec.yaml").read_text())
    r = server.build_regions(spec)
    assert r["ok"] and r["builder_ok"]
    doc = yaml.safe_load((work / r["yaml_rel"]).read_text())
    assert doc["settings"]["volume_fraction"] == spec["settings"]["volume_fraction"] == 0.2
    vol = r["derived"]["volume"]
    assert vol["volume_fraction"] == 0.2 and vol["volume_fraction_of"].startswith("free design voxels")
    text = "\n".join(r["report"])
    assert f"the {vol['free_design_voxels']} free design voxels" in text
    assert f"= {vol['target_volume_mm3']:.0f} mm3, {vol['target_mass_g']:.1f} g at" in text
    assert "mean density of the FREE design voxels" in " ".join(server.build_regions.__doc__.split())


def test_touch_fact_is_one_warning_in_the_reply(work):
    """P1 (F2-3): the builder and check_problem give the same warning once, and the build is not a problem."""
    spec = {"name": "bridged",
            "regions": {"design": [{"box": {"min": [4, 0, 0], "max": [24, 10, 10]}}],
                        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}]},
                        "solid": {"bridge": [{"box": {"min": [0, 0, 0], "max": [4, 10, 10]}}]},
                        "load": {"tip": [{"box": {"min": [20, 3, 6], "max": [24, 7, 10]}}]}},
            "load_cases": [{"name": "down", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -5.0]}]}],
            "settings": {"volume_fraction": 0.3, "voxel_size": 1.0, "iterations": 5}}
    r = server.build_regions(spec)
    assert r["ok"] and r["builder_ok"] and r["problems"] == []
    touch = [w for w in r["warnings"] if "touches a design voxel" in w]
    assert len(touch) == 1 and touch == [w for w in r["check"]["warnings"] if "touches a design voxel" in w]
    assert "Options:" in touch[0] and r["check"]["connectivity"]["fixed_touches_design"] is False


def test_pieces_that_each_hold_a_support_are_one_warning_and_no_problem(work):
    """P1 review (M2): two walls, each holding its own piece. The reply has the text of `check_problem` once in `warnings`, and no problem."""
    spec = {"name": "two_walls",
            "regions": {"design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}, {"box": {"min": [0, 30, 0], "max": [20, 40, 10]}}],
                        "fixed": {"w1": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}],
                                  "w2": [{"box": {"min": [-4, 30, 0], "max": [0, 40, 10]}}]},
                        "load": {"t1": [{"box": {"min": [16, 3, 6], "max": [20, 7, 10]}}],
                                 "t2": [{"box": {"min": [16, 33, 6], "max": [20, 37, 10]}}]}},
            "load_cases": [{"name": "down", "weight": 1.0,
                            "forces": [{"region": "t1", "force": [0, 0, -5.0]}, {"region": "t2", "force": [0, 0, -5.0]}]}],
            "settings": {"volume_fraction": 0.3, "voxel_size": 2.0, "iterations": 2}}
    r = server.build_regions(spec)
    assert r["ok"] and r["builder_ok"] and r["problems"] == []
    pieces = [w for w in r["warnings"] if "separate pieces" in w]
    assert len(pieces) == 1 and pieces == [w for w in r["check"]["warnings"] if "separate pieces" in w]
    assert "each holds a fixed voxel" in pieces[0] and "stops with this error" not in pieces[0]


def _hook():
    return yaml.safe_load((EX / "hook_spec.yaml").read_text())


@pytest.mark.parametrize("key", ["../../escaped", "/abs/path", "a/b", ".."])
def test_unsafe_region_key(work, tmp_path, capsys, key):
    spec = _hook()
    role = "fixed"
    spec["regions"][role] = {key: next(iter(spec["regions"][role].values()))}
    r = server.build_regions(spec)
    assert r["ok"] is False and "\n" not in r["error"] and repr(key) in r["error"]
    assert not (tmp_path / "escaped.stl").exists()
    assert not (work / "runs").exists() or not any((work / "runs").rglob("*.stl"))
    # CLI gives a SPEC ERROR line
    sp = tmp_path / "s.yaml"
    sp.write_text(yaml.safe_dump(spec))
    assert wrapper.main([str(sp), "--out", str(tmp_path / "o")]) == 1
    assert "SPEC ERROR" in capsys.readouterr().out


def test_run_dir_is_a_file(work):
    (work / "runs").mkdir()
    (work / "runs" / "hook").write_text("x")
    spec = _hook()
    spec["name"] = "hook"
    r = server.build_regions(spec)
    assert r["ok"] is False and "\n" not in r["error"]


def test_rebuild_lists_stale_files(work):
    spec = _hook()
    r1 = server.build_regions(spec)
    assert r1["ok"] and r1["other_files_rel"] == []
    loads = spec["regions"]["load"]
    (nm, items), = list(loads.items())[:1]
    spec["regions"]["load"] = {nm + "_b": items}
    for c in spec["load_cases"]:
        for f in c["forces"]:
            f["region"] = nm + "_b"
    r2 = server.build_regions(spec)
    assert r2["ok"], r2
    assert f"{r2['run_dir_rel']}/{nm}_b.stl" in r2["files_rel"]
    assert f"{r2['run_dir_rel']}/{nm}.stl" not in r2["files_rel"]
    assert f"{r2['run_dir_rel']}/{nm}.stl" in r2["other_files_rel"]
    assert (work / r2["run_dir_rel"] / f"{nm}.stl").exists()


def test_name_live_warns(work):
    r = server.build_regions(_hook(), name="live")
    assert r["ok"] and any("runs/live" in w for w in r["warnings"])


def _small_spec(**settings):
    return {"name": "w", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [40, 20, 20]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 20]}}]},
        "load": {"tip": [{"box": {"min": [36, 8, 8], "max": [40, 12, 12]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0,
                        "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
        "settings": {"voxel_size": 2.0, **settings}}


def test_soft_levels_are_warnings_not_problems(work):
    r = server.build_regions(_small_spec())
    assert r["ok"] and r["builder_ok"] and r["problems"] == []
    assert any("small or thin" in w for w in r["builder_warnings"])
    assert any("small or thin" in w for w in r["warnings"])


def test_ram_block_and_allow_large_grid(work, monkeypatch):
    monkeypatch.setattr("topopt.cli.physical_ram_bytes", lambda: 1_000_000)
    r = server.build_regions(_small_spec())
    assert r["ok"] is False and r["builder_ok"] is False
    assert any("RAM" in p and "allow_large_grid" in p for p in r["problems"])
    r = server.build_regions(_small_spec(allow_large_grid=True))
    assert r["builder_ok"] and r["problems"] == []
    assert r["check"]["memory_estimate"]["ram_bytes"] == 1_000_000
    assert any("memory estimate" in w for w in r["warnings"])
