"""Tests for the workspace tools (list_runs, relative paths, preview) and check_problem."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt_mcp import server  # noqa: E402
from test_optimize import make_case  # noqa: E402


@pytest.fixture(autouse=True)
def ws(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    server._current.clear()
    yield tmp_path
    server._current.clear()


def fake_run(d: Path, state="done", iteration=10, stl=True):
    (d / "result").mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "iteration": iteration}))
    (d / "run.json").write_text(json.dumps({"name": "p", "iterations": 10}))
    if stl:
        import trimesh
        trimesh.creation.box((10, 4, 2)).export(d / "result" / "final.stl")
        (d / "result" / "metrics.json").write_text(json.dumps({"mass_final_g": 1.0}))


def test_list_runs_empty(ws):
    r = server.list_runs()
    assert r == {"workspace": str(ws), "runs": []}


def test_list_runs(ws):
    fake_run(ws / "runs" / "a")
    fake_run(ws / "b", state="running", iteration=3, stl=False)
    (ws / "runs" / "empty").mkdir()
    (ws / "other").mkdir()
    r = server.list_runs()
    by = {x["name"]: x for x in r["runs"]}
    assert set(by) == {"a", "b"}
    assert by["a"]["path"] == "runs/a" and by["a"]["state"] == "done"
    assert by["a"]["final_stl_rel"] == "runs/a/result/final.stl"
    assert by["b"]["path"] == "b" and by["b"]["iteration"] == 3 and by["b"]["final_stl_rel"] is None
    assert by["a"]["run_dir"] == str(ws / "runs" / "a")


def test_run_result_relative_path_and_preview(ws):
    fake_run(ws / "runs" / "a")
    r = server.run_result("runs/a")
    assert r["done"] and r["final_stl_rel"] == "runs/a/result/final.stl"
    assert Path(r["final_stl"]).is_absolute()
    p = r["preview"]
    assert p["kind"] == "ascii-silhouettes" and p["watertight"] is True
    assert abs(p["volume_mm3"] - 80.0) < 0.5
    top = p["views"]["xy_top"]
    assert len(top[0]) > len(top) and set("".join(top)) <= {"#", "."} and "#" in "".join(top)
    assert "preview" not in server.run_result("runs/a", preview=False)


def test_rel_outside_workspace(ws, tmp_path_factory):
    other = tmp_path_factory.mktemp("elsewhere")
    assert server._rel(other / "x") == str(other / "x")
    assert server._rel(ws / "a" / "b") == "a/b"


def test_run_topopt_gives_relative_paths(ws, monkeypatch):
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    (ws / "p.yaml").write_text("name: x\n")
    r = server.run_topopt("p.yaml")
    assert r["yaml_rel"] == "p.yaml" and r["run_dir_rel"] == "runs/live"
    assert r["log_file_rel"] == "runs/live.stdout" and r["workspace"] == str(ws)
    server._current["proc"].wait()


def test_check_problem_data(ws):
    y = make_case(ws, h=2.0)
    r = server.check_problem(y.name)
    assert r["ok"] is True and r["yaml_rel"] == y.name
    g = r["grid"]
    assert g["voxels"] == g["shape"][0] * g["shape"][1] * g["shape"][2]
    vc = r["voxel_counts"]
    assert set(vc) == {"design", "fixed", "load", "keep_out"}
    assert r["active_voxels"] == vc["design"] + vc["fixed"] + vc["load"]
    assert r["forced_solid_voxels"] == vc["fixed"] + vc["load"]
    assert 0 <= r["forced_solid_share"] <= 1
    assert r["free_dofs"] > 0
    assert r["connectivity"]["components"] >= 1
    assert set(r["backends"]) == {"cholesky", "pardiso", "amg"}
    resolved = r["solver"]["resolved"]   # the solver that runs here: the estimate is for it
    assert resolved in r["backends"] and r["solver"]["error"] is None
    assert r["backends"][resolved]["time_estimate"]["total_s"] == r["time_estimate"]["total_s"]
    assert r["time_estimate"]["solver"] == resolved and r["time_estimate"]["amg_total_s"] > 0
    assert isinstance(r["warnings"], list)
    json.dumps(r)  # plain data


def test_check_problem_errors(ws):
    assert server.check_problem("nope.yaml")["ok"] is False
    y = make_case(ws, h=2.0)
    y.write_text(y.read_text() + "bogus: 1\n")
    r = server.check_problem(str(y))
    assert r["ok"] is False and r["error"]


def test_check_data_matches_cli_output(ws, capsys):
    from topopt.cli import main, check_data
    y = make_case(ws, h=2.0)
    d = check_data(y)
    assert main(["check", str(y)]) == 0
    out = capsys.readouterr().out
    assert f"active voxels:  {d['active_voxels']}" in out
    assert f"free dofs:      {d['free_dofs']}" in out


def test_list_runs_newest_first(ws):
    import os
    fake_run(ws / "runs" / "old")
    fake_run(ws / "runs" / "new")
    for name, t in (("old", 1_000_000_000), ("new", 1_600_000_000)):
        for f in ("status.json", "run.json"):
            os.utime(ws / "runs" / name / f, (t, t))
    assert [r["name"] for r in server.list_runs()["runs"]] == ["new", "old"]


def test_list_runs_skips_unreadable_folder(ws):
    import os
    if os.geteuid() == 0:
        pytest.skip("root can read a mode 000 folder")
    fake_run(ws / "runs" / "ok")
    bad = ws / "runs" / "bad"
    bad.mkdir()
    bad.chmod(0)
    try:
        assert [r["name"] for r in server.list_runs()["runs"]] == ["ok"]
    finally:
        bad.chmod(0o755)


def test_non_object_json(ws):
    d = ws / "runs" / "odd"
    d.mkdir(parents=True)
    (d / "status.json").write_text("[1, 2]")
    (d / "run.json").write_text("3")
    r = server.list_runs()["runs"]
    assert r[0]["state"] == "unreadable" and r[0]["iteration"] is None
    assert server.run_status(str(d))["state"] == "none"
    assert server.run_result(str(d))["done"] is False


def test_status_early_returns_have_rel(ws):
    assert server.run_status("runs/none")["run_dir_rel"] == "runs/none"


def test_check_data_warnings(ws):
    # h = 2: the 2 x 2 x 3 mm load box is a few voxels, so the small-region warning shows.
    r = server.check_problem(str(make_case(ws, h=2.0)))
    assert any("load region 'tip' is small or thin" in w for w in r["warnings"])
    assert r["load_regions"][0]["voxels"] < 30


def test_check_problem_with_a_solver_that_this_platform_cannot_run(ws):
    # The facts come back with the solver error: check_problem does not fail on it.
    import yaml
    if sys.platform not in ("linux", "darwin"):
        pytest.skip("needs Linux or macOS")
    y = make_case(ws, h=2.0)
    d = yaml.safe_load(y.read_text())
    d["settings"]["solver"] = "pardiso" if sys.platform == "darwin" else "cholesky"
    y.write_text(yaml.safe_dump(d))
    r = server.check_problem(str(y))
    assert r["ok"] is True and r["free_dofs"] > 0
    assert r["solver"]["resolved"] is None and r["solver"]["error"]
    assert r["time_estimate"]["solver"] is None and r["time_estimate"]["total_s"] > 0
    assert any("cannot start here" in w for w in r["warnings"])
    json.dumps(r)


def test_example_yaml_is_found_in_the_code_folder_by_all_yaml_tools(ws):
    """The work folder has no examples/. The code folder (a checkout, or the plugin copy) has them."""
    rel = "examples/hook/hook.yaml"
    assert not (ws / rel).exists()
    checked = server.check_problem(rel)
    assert checked["ok"] is True and checked["yaml_rel"] == str(REPO / rel)
    env = server.environment(rel)
    assert env["problem"]["ok"] is True
    assert server._resolve_yaml(rel) == REPO / rel
    # A file in the work folder wins over the code folder, and a missing file names the work folder.
    (ws / "examples" / "hook").mkdir(parents=True)
    (ws / rel).write_text("name: x\n")
    assert server._resolve_yaml(rel) == ws / rel
    assert server._resolve_yaml("nope.yaml") == ws / "nope.yaml"
    assert server.check_problem("nope.yaml")["ok"] is False
