"""Tests for the MCP tools run_history, compare_runs and marks_to_hints, and the loop of CONTRACTS.md section 21.

The loop test runs the real optimizer on a small case: run, a marks file, hints, a rerun with the proposed YAML
(started with run_topopt), and compare. It calls the tool functions of the server, as an agent does.
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt_mcp import server  # noqa: E402
from test_feedback import fake_run, make_case, mark, marks_doc, tree_hash  # noqa: E402

NEW_TOOLS = ("run_history", "compare_runs", "marks_to_hints")


@pytest.fixture(autouse=True)
def ws(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    server._current.clear()
    yield tmp_path
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait()
    server._current.clear()


def test_tools_are_registered_with_texts_that_follow_the_freedom_rule():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert set(NEW_TOOLS) <= set(tools)
    for name in NEW_TOOLS:
        text = tools[name].description or ""
        assert 100 < len(text) <= 4096
        assert not re.search(r"\b(must|never|always|forbidden)\b", text, re.I), name   # facts, estimates, options
    assert "Writes nothing" in tools["run_history"].description and "writes nothing" in tools["compare_runs"].description
    assert "NEW folder" in tools["marks_to_hints"].description
    for word in ("run_history", "compare_runs", "marks_to_hints"):      # #380: the last line of the instructions names them; the descriptions say when to use them
        assert word in server.INSTRUCTIONS.splitlines()[-1]
    assert not re.search(r"\b(must|never)\b", server.INSTRUCTIONS, re.I)
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert len(server.INSTRUCTIONS) <= OLD_CLIENT_CUT


def test_run_history_tool(ws):
    ypath = make_case(ws / "in")
    fake_run(ws / "runs" / "a", ypath)
    fake_run(ws / "runs" / "b", ypath, state="running")
    before = tree_hash(ws)
    r = server.run_history()
    assert tree_hash(ws) == before
    assert r["root"] == str(ws) and r["root_rel"] == "." and r["count"] == 2
    by = {x["dir_name"]: x for x in r["runs"]}
    assert by["a"]["path"] == "runs/a" and by["a"]["state"] == "done" and by["a"]["process_alive"] is False
    assert by["a"]["metrics"]["compliance_final"] == 1.0 and by["b"]["state"] == "running"
    assert server.run_history("runs", limit=1)["root_rel"] == "runs" and len(server.run_history("runs", limit=1)["runs"]) == 1
    assert server.run_history("nope")["ok"] is False
    assert server.run_history("runs/a")["count"] == 1
    json.dumps(r)


def test_compare_runs_tool(ws):
    ya = make_case(ws / "in_a")
    yb = make_case(ws / "in_b", vf=0.25)
    fake_run(ws / "runs" / "a", ya, compliance=1.0)
    fake_run(ws / "runs" / "b", yb, compliance=2.0)
    r = server.compare_runs("runs/a", "runs/b")
    assert r["ok"] and r["a"]["run_dir_rel"] == "runs/a" and r["b"]["run_dir_rel"] == "runs/b"
    assert "volume_fraction" in {x["key"] for x in r["setup"]["run_json_diff"]}
    assert r["inputs"]["a"]["labels_vs_disk"]["labels_same"] is True
    assert server.compare_runs("runs/a", "runs/b", check_labels=False)["inputs"]["a"]["labels_vs_disk"]["checked"] is False
    assert server.compare_runs("runs/a", "runs/zzz")["ok"] is False


def test_marks_to_hints_tool(ws):
    ypath = make_case(ws / "in")
    fake_run(ws / "runs" / "a", ypath)
    (ws / "marks.json").write_text(json.dumps(marks_doc(mark(1, "remove", [(10, 3, 6)], mirror=True))))
    before = tree_hash(ws / "in", ws / "runs" / "a")
    r = server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="runs/hints_a")      # the short reply
    assert r["ok"] and r["out_dir_rel"] == "runs/hints_a" and r["yaml_rel"] == "runs/hints_a/bracket.yaml"
    assert r["report_rel"] == "runs/hints_a/hints.json" and r["run_dir_rel"] == "runs/a" and r["detail"] is False
    assert r["marks"][0]["status"] == "applied" and r["proposed_yaml"]["loads"] is True
    assert "option_catalog" not in r and "facts" not in r["marks"][0] and r["report_file"] == "hints.json"
    full = server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="runs/hints_a", overwrite=True, detail=True)
    assert "option_catalog" in full and "facts" in full["marks"][0] and full["out_dir_rel"] == "runs/hints_a"
    assert tree_hash(ws / "in", ws / "runs" / "a") == before
    on_disk = json.loads((ws / "runs" / "hints_a" / "hints.json").read_text())
    assert on_disk["marks"][0]["region_file"] == r["marks"][0]["region_file"] == "hint_remove_1.stl"
    # the default run is the last run that this server started; the marks can be the object or the JSON text
    server._current["run_dir"] = ws / "runs" / "a"
    r = server.marks_to_hints(json.dumps(marks_doc(mark(1, "add", [(10, 3, 6)]))), out_dir="runs/hints_b",
                              options={"add": "skip"}, detail=True)
    assert r["ok"] and r["marks"][0]["status"] == "skipped_by_option" and r["marks_file"] is None
    assert server.marks_to_hints("nope.json", run_dir="runs/a")["ok"] is False
    assert server.marks_to_hints("marks.json", run_dir="runs/zzz")["ok"] is False
    assert server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="runs/hints_a")["ok"] is False   # exists
    assert server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="runs/hints_a", overwrite=True)["ok"]


def test_tool_texts_say_what_the_code_refuses_and_what_the_tools_offer():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    text = tools["marks_to_hints"].description
    for word in ("run folder", "symbolic link", "earlier hints", "input file", "detail", "yaml_path", "first mark wins"):
        assert word in text, word
    from topopt_mcp import replies   # P9-11: the line that explains the keys of the reply moved to the guide topopt://replies/marks_to_hints
    assert "topopt://replies/marks_to_hints" in text
    for word in ("invalid", "volume_target"):
        assert word in replies.guide("marks_to_hints"), word
    assert "limit" in tools["run_history"].description and "default 20" in tools["run_history"].description
    # the documents say the same (CONTRACTS.md section 21, the plan)
    contracts = (REPO / "CONTRACTS.md").read_text()
    for word in ("symbolic link", "earlier hints folder", "first mark wins", "material.E"):
        assert word in contracts, word


def test_run_history_tool_default_limit_and_marks_to_hints_refusals(ws):
    ypath = make_case(ws / "in")
    for i in range(22):
        d = ws / "runs" / f"r{i:02d}"
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"state": "done", "iteration": i}))
        os.utime(d / "status.json", (1_700_000_000 + i, 1_700_000_000 + i))
    r = server.run_history("runs")
    assert r["count"] == 22 and r["listed"] == 20 and any("2 older runs are left out" in n for n in r["notes"])
    assert server.run_history("runs", limit=0)["listed"] == 22
    fake_run(ws / "runs" / "a", ypath)
    (ws / "marks.json").write_text(json.dumps(marks_doc(mark(1, "remove", [(10, 3, 6)]))))
    bad = server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="runs/r00/inside")      # inside a run folder
    assert bad["ok"] is False and bad["refused"]["reason"] == "run_folder" and not (ws / "runs" / "r00" / "inside").exists()
    bad = server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="in")                 # the folder of the YAML
    assert bad["refused"]["reason"] == "input_folder"
    # a moved run: yaml_path (relative to the work folder) names the YAML
    (ws / "moved").mkdir()
    shutil.move(str(ypath.parent), str(ws / "moved" / "in"))
    assert "yaml_path" in server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="runs/h0")["error"]
    ok = server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="runs/h1", yaml_path="moved/in/bracket.yaml")
    assert ok["ok"] and ok["grid"]["labels_match_run"] is True and ok["yaml_rel"] == "runs/h1/bracket.yaml"


def test_marks_to_hints_tool_when_the_proposed_yaml_does_not_load(ws):
    ypath = make_case(ws / "in")
    fake_run(ws / "runs" / "a", ypath)
    doc = marks_doc(mark(1, "remove", [(22.5, 0, 2)], radius=5.0))                              # covers the load region
    r = server.marks_to_hints(doc, run_dir="runs/a", out_dir="runs/stop", options={"protect_regions": False})
    assert r["ok"] is False and "does not load" in r["error"] and r["out_dir_rel"] == "runs/stop"
    assert r["marks"][0]["status"] == "invalid" and r["report_rel"] == "runs/stop/hints.json"
    assert (ws / "runs" / "stop" / "hints.json").is_file()


def wait_done(run_dir, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        st = server.run_status(run_dir)
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.3)
    raise AssertionError(f"run {run_dir} did not finish")


def surface_point(mesh, target):
    v = np.asarray(mesh.vertices)
    return v[int(np.argmin(np.linalg.norm(v - np.asarray(target, float), axis=1)))]


def test_loop_run_marks_hints_rerun_compare(ws, monkeypatch):
    """The whole loop on a small case, through the tool functions. Real optimizer, no dashboard."""
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -m topopt run {{yaml}} --out {{out}} --quiet")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    make_case(ws / "in", iterations=15)
    in_before = tree_hash(ws / "in")
    # 1. run
    started = server.run_topopt("in/bracket.yaml", out_dir="runs/a")
    assert "error" not in started, started
    assert wait_done("runs/a")["state"] == "done"
    res = server.run_result("runs/a", preview=False)
    assert res["done"]
    final = trimesh.load(res["final_stl"], file_type="stl", force="mesh")
    # 2. a marks file, as the Refine page writes it (points on the surface of final.stl, in mm)
    p_remove = surface_point(final, (12, 5, 6))
    p_add = surface_point(final, (17, 0, 11))
    p_smooth = surface_point(final, (8, 0, 0))
    marks = marks_doc(
        mark(1, "remove", [p_remove.round(3)], radius=2.5, mirror=True, note="a hole here"),
        mark(2, "add", [p_add.round(3)], radius=2.0),
        mark(3, "smooth", [p_smooth.round(3)], radius=3.0),
        sha=hashlib.sha256(Path(res["final_stl"]).read_bytes()).hexdigest())
    (ws / "marks.json").write_text(json.dumps(marks))
    # 3. hints: facts and options, a proposed YAML
    hints = server.marks_to_hints("marks.json", run_dir="runs/a", out_dir="runs/a_hints", detail=True)
    assert hints["ok"] and hints["marks_match_final_stl"] is True and hints["proposed_yaml"]["loads"] is True
    status = {e["id"]: e["status"] for e in hints["marks"]}
    assert status == {1: "applied", 2: "applied", 3: "facts_only"}
    assert hints["labels_changed"]["voxels"] > 0 and hints["check"]["active_voxels"] > 0
    assert tree_hash(ws / "in") == in_before                       # the input YAML and STLs stay as they are
    # 4. the rerun with the existing run tool
    again = server.run_topopt(hints["yaml_rel"], out_dir="runs/b")
    assert "error" not in again, again
    assert wait_done("runs/b")["state"] == "done"
    res_b = server.run_result("runs/b", preview=False)
    final_b = trimesh.load(res_b["final_stl"], file_type="stl", force="mesh")
    # the hint was honoured: no vertex of the new part lies inside the keep-out sphere (the exact cut), and
    # the first part has some there
    radius = hints["marks"][0]["region"]["radius_mm"]
    for centre in (p_remove, p_remove * np.array([1, -1, 1])):
        d_a = np.linalg.norm(np.asarray(final.vertices) - centre, axis=1)
        d_b = np.linalg.norm(np.asarray(final_b.vertices) - centre, axis=1)
        assert (d_a < radius - 0.3).sum() > 0 and (d_b < radius - 0.3).sum() == 0
    # 5. compare shows the change
    cmp = server.compare_runs("runs/a", "runs/b")
    assert cmp["ok"]
    regions = {x["key"]: x["status"] for x in cmp["setup"]["region_files"]["diff"]}
    assert regions == {"keep_out:hint_remove_1.stl": "only_in_b", "load:hint_add_2": "only_in_b"}
    lab = cmp["setup"]["labels_diff"]
    assert lab["comparable"] and lab["voxels"] == hints["labels_changed"]["voxels"]
    assert set(lab["transitions"]) == {"design->keep_out", "design->load"}
    rows = {x["key"]: x for x in cmp["metrics"]["rows"]}
    assert rows["compliance_final"]["delta"] != 0 and rows["volume_final_mm3"]["delta"] != 0
    assert cmp["inputs"]["b"]["labels_vs_disk"]["labels_same"] is True
    assert cmp["inputs"]["a"]["labels_vs_disk"]["labels_same"] is True
    # 6. the history lists both runs
    hist = server.run_history("runs")
    assert {r["dir_name"]: r["state"] for r in hist["runs"]} == {"a": "done", "b": "done"}
    print("\nLOOP: compliance_final", rows["compliance_final"]["a"], "->", rows["compliance_final"]["b"],
          "| voxels changed", lab["voxels"], lab["transitions"], "| summary:", cmp["summary"])
