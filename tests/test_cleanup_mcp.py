"""The MCP tool `cleanup_list` (CONTRACTS.md section 29.3): a thin wrapper of topopt.cleanup.cleanup_list.

The tool runs in the server process (about 2 s on a small part) and edits nothing. These tests use the small posts part of
test_cleanup.py: a thin bar with two dead-end posts, so the list has two spur candidates. The server module must not load
topopt.cleanup at import time (the heavy packages load at the first call only).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from test_cleanup import posts_part, run_at, sha  # noqa: E402

from topopt_mcp import server  # noqa: E402

SHORT_KEYS = {"ok", "counts", "candidates", "applicability", "estimate", "routes", "problems", "notes",
              "out_dir_rel", "marks_rel", "report_rel", "run_dir_rel", "detail", "report_file"}
CANDIDATE_KEYS = {"id", "type", "operator", "detector", "confidence", "centre_mm", "detected_volume_mm3", "length_mm",
                  "nearest_frozen_mm"}


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("cleanup_mcp")
    run, ypath = run_at(root, posts_part())
    return root, run, ypath


@pytest.fixture
def ws(case, monkeypatch):
    root, run, ypath = case
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(root))
    monkeypatch.setitem(server._current, "run_dir", None)
    return root, run, ypath


def test_the_tool_is_registered_and_its_text_is_within_the_bound():
    import asyncio
    tools = asyncio.run(server.mcp.list_tools())
    by_name = {t.name: t for t in tools}
    assert "cleanup_list" in by_name
    assert len(by_name["cleanup_list"].description) <= 4096
    assert "29.3" in by_name["cleanup_list"].description
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert "cleanup_list" in server.INSTRUCTIONS and len(server.INSTRUCTIONS) <= OLD_CLIENT_CUT


def test_the_server_does_not_load_the_cleanup_module_at_import():
    code = ("import sys; import topopt_mcp.server; "
            "print('topopt.cleanup' in sys.modules, 'skimage' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[-2:] == ["False", "False"], out.stdout


def test_the_short_reply_has_the_decision_keys_and_relative_paths(ws, tmp_path):
    root, run, ypath = ws
    out = tmp_path / "list_out"
    before = sha(run / "result" / "final.stl")
    rep = server.cleanup_list(run_dir=str(run), out_dir=str(out), yaml_path=str(ypath))
    assert rep["ok"], rep
    assert SHORT_KEYS <= set(rep), SHORT_KEYS - set(rep)
    assert rep["detail"] is False and rep["report_file"] == "cleanup.json"
    assert "marks" not in rep                                              # the full report is in cleanup.json
    assert rep["counts"]["listed"] == len(rep["candidates"]) >= 2
    for cand in rep["candidates"]:
        assert CANDIDATE_KEYS <= set(cand), CANDIDATE_KEYS - set(cand)
    assert {c["detector"] for c in rep["candidates"]} == {"spur"} and {c["operator"] for c in rep["candidates"]} == {"remove cut"}
    # the paths: relative to the work folder when inside it, else absolute
    assert rep["run_dir_rel"] == "runs/r"
    assert Path(rep["marks_rel"]).name == "marks.json" and Path(rep["report_rel"]).name == "cleanup.json"
    assert (out / "marks.json").is_file() and (out / "cleanup.json").is_file()
    marks = json.loads((out / "marks.json").read_text())
    assert marks["version"] == 1 and len(marks["marks"]) == len(rep["candidates"])
    # the call edits nothing: the run folder keeps its final.stl
    assert sha(run / "result" / "final.stl") == before


def test_detail_gives_the_full_report(ws, tmp_path):
    root, run, ypath = ws
    rep = server.cleanup_list(run_dir=str(run), out_dir=str(tmp_path / "d"), yaml_path=str(ypath), detail=True)
    assert rep["ok"] and rep.get("marks") and rep.get("header")
    assert rep["marks_rel"] and rep["out_dir_rel"]
    for m in rep["marks"]:
        assert m["cleanup"]["thresholds_used"] and m["cleanup"]["measured"]


def test_options_reach_the_list_and_the_default_folder_sits_beside_the_run(ws):
    root, run, ypath = ws
    rep = server.cleanup_list(run_dir="runs/r", yaml_path=str(ypath), options={"max_candidates": 1}, overwrite=True)
    assert rep["ok"], rep
    assert len(rep["candidates"]) == 1 and rep["not_listed"] >= 1
    assert rep["out_dir_rel"] == "runs/r-cleanup"
    assert (root / "runs" / "r-cleanup" / "marks.json").is_file()


def test_the_last_run_of_the_server_is_the_default(ws, tmp_path):
    root, run, ypath = ws
    server._current["run_dir"] = run
    try:
        rep = server.cleanup_list(out_dir=str(tmp_path / "last"), yaml_path=str(ypath))
    finally:
        server._current["run_dir"] = None
    assert rep["ok"], rep
    assert rep["run_dir_rel"] == "runs/r"


def test_estimate_only_writes_nothing(ws, tmp_path):
    root, run, ypath = ws
    out = tmp_path / "never"
    rep = server.cleanup_list(run_dir=str(run), out_dir=str(out), yaml_path=str(ypath), estimate_only=True)
    assert rep["ok"] and rep["estimate_only"] is True
    assert rep["estimate"]["detector"]["voxels"] > 0
    assert not out.exists()


def test_a_refusal_is_a_reply_with_the_reason_not_a_crash(ws):
    root, run, ypath = ws
    for bad in (run, run / "result"):
        rep = server.cleanup_list(run_dir=str(run), out_dir=str(bad), yaml_path=str(ypath))
        assert rep["ok"] is False and rep["error"], rep
        assert "marks_rel" not in rep
    gone = server.cleanup_list(run_dir=str(root / "no_such_run"))
    assert gone["ok"] is False and gone["error"]
    # an unknown option name is a problem in the reply (the defaults are used), not a failure
    unknown = server.cleanup_list(run_dir=str(run), options={"no_such_option": 1}, out_dir=str(root / "x"), yaml_path=str(ypath))
    assert unknown["ok"] and "'no_such_option' is not known" in " ".join(unknown["problems"])
