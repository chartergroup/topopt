"""The MCP tool `export_step` (CONTRACTS.md section 42): a thin wrapper of `python -m topopt step ... --json`.

The tool starts the command in a child Python, so OpenCASCADE never loads in the server process (it prints to the C stdout, which is the
protocol channel of the server) and its memory ends with the child. These tests need the extra `step` (`needs_ocp`), except the ones that
fake a Python without it.
"""
import asyncio
import json
import os
import stat
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import needs_ocp  # noqa: E402
from test_step import REPORT_KEYS, make_run, tree  # noqa: E402

from topopt_mcp import server  # noqa: E402


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    monkeypatch.setitem(server._current, "run_dir", None)
    return tmp_path


def test_the_tool_is_registered_and_its_text_is_within_the_bound():
    by_name = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert "export_step" in by_name
    text = by_name["export_step"].description
    assert len(text) <= 4096
    assert "STEP" in text and "estimate_only" in text and "part_only" in text and "tolerance_mm" in text
    low = text.lower()
    assert " must " not in low and "never" not in low
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert len(server.INSTRUCTIONS) <= OLD_CLIENT_CUT


@needs_ocp
def test_the_tool_returns_the_report_of_the_command(ws):
    run = make_run(ws, name="runs/a", run_json={"name": "a", "h": 2.0})
    r = server.export_step(run_dir="runs/a", tolerance_mm=0)
    assert r["ok"] is True and r["exit_code"] == 0
    assert (REPORT_KEYS - {"report_path"}) <= set(r)
    assert r["run_dir_rel"] == "runs/a" and r["output"]["path_rel"] == "runs/a/result/final.step"
    assert r["report_path_rel"] == "runs/a/result/final.step.json"
    assert r["checks_failed"] == [] and r["model"]["faces"] == 6
    saved = json.loads((run / "result" / "final.step.json").read_text())
    assert saved["volume"] == r["volume"] and saved["model"] == r["model"]


@needs_ocp
def test_part_only_estimate_only_schema_and_out(ws):
    import trimesh
    run = make_run(ws, name="runs/b", part_only=trimesh.creation.box((5, 5, 5)))
    before = tree(run)
    e = server.export_step(run_dir=str(run), estimate_only=True)
    assert e["ok"] is True and e["state"] == "estimate" and e["estimate"]["faces"] == 12 and tree(run) == before
    p = server.export_step(run_dir=str(run), part_only=True, tolerance_mm=0, schema="AP242", out="elsewhere.step")
    assert p["ok"] is True and p["mesh"] == "part_only.stl" and p["output"]["schema"] == "AP242"
    assert (ws / "elsewhere.step").is_file() and p["output"]["path_rel"] == "elsewhere.step"
    assert p["volume"]["step_mm3"] == pytest.approx(125.0, rel=1e-12)
    n = server.export_step(run_dir=str(run), tolerance_mm=0, merge_planes=False)
    assert n["model"]["faces"] == 12 and n["merge_planes"] == {"applied": False}


@needs_ocp
def test_a_failed_check_is_data_and_the_file_is_written(ws):
    import trimesh
    box = trimesh.creation.box((10, 20, 30))
    run = make_run(ws, trimesh.Trimesh(box.vertices, box.faces[:-1], process=False), name="runs/open")
    r = server.export_step(run_dir=str(run), tolerance_mm=0)
    assert r["ok"] is True and r["exit_code"] == 3 and "closed" in r["checks_failed"]
    assert (run / "result" / "final.step").is_file()


def test_a_wrong_input_is_a_one_line_error(ws):
    r = server.export_step(run_dir="nothing_here")
    assert r["ok"] is False and "not a folder" in r["error"] and "missing_extra" not in r
    run = make_run(ws, name="runs/c", state="running")
    assert "not finished" in server.export_step(run_dir=str(run))["error"]
    assert "tolerance_mm" in server.export_step(run_dir=str(make_run(ws, name="runs/d")), tolerance_mm=-1)["error"]


def fake_python_without_ocp(tmp_path):
    """An executable that runs the command line `-m topopt step ...` in a Python where OCP cannot be imported."""
    script = tmp_path / "python_without_ocp"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" -c "import sys; sys.modules[\'OCP\'] = None; '
                      f'from topopt.cli import main; sys.exit(main(sys.argv[3:]))" "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@pytest.mark.skipif(sys.platform == "win32", reason="a shell script")
def test_without_the_extra_the_reply_has_the_install_command(ws, monkeypatch):
    run = make_run(ws, name="runs/e")
    monkeypatch.setenv("TOPOPT_PYTHON", str(fake_python_without_ocp(ws)))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    before = tree(run)
    r = server.export_step(run_dir=str(run))
    assert r["ok"] is False and r["missing_extra"] == "step"
    assert r["install_command"] == 'uv pip install -e ".[step]"' and "cadquery-ocp-novtk==" in r["install_command_no_checkout"]
    assert "macOS" in r["error"] and "Linux" in r["error"] and "Traceback" not in r["error"]
    assert tree(run) == before


@needs_ocp
def test_out_that_is_the_input_stl_is_an_error_and_the_stl_stays(ws):
    run = make_run(ws, name="runs/g")
    before = tree(run)
    r = server.export_step(run_dir=str(run), tolerance_mm=0, out=str(run / "result" / "final.stl"))
    assert r["ok"] is False and "the STL that this command reads" in r["error"] and r["exit_code"] == 1
    assert "Give a new file name that ends in .step, or a folder" in r["error"]
    assert tree(run) == before
    r2 = server.export_step(run_dir=str(run), tolerance_mm=0, out="runs/g/status.json")
    assert r2["ok"] is False and r2["exit_code"] == 1 and tree(run) == before


@needs_ocp
def test_the_schema_name_is_read_like_the_command_reads_it(ws):
    run = make_run(ws, name="runs/h")
    r = server.export_step(run_dir=str(run), tolerance_mm=0, schema="ap-242")
    assert r["ok"] is True and r["output"]["schema"] == "AP242" and r["checks_failed"] == []
    bad = server.export_step(run_dir=str(run), tolerance_mm=0, schema="AP999")
    assert bad["ok"] is False and "AP214 or AP242" in bad["error"] and bad["exit_code"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="a shell script")
def test_the_estimate_works_without_the_extra(ws, monkeypatch):
    run = make_run(ws, name="runs/i")
    monkeypatch.setenv("TOPOPT_PYTHON", str(fake_python_without_ocp(ws)))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    before = tree(run)
    r = server.export_step(run_dir=str(run), estimate_only=True)
    assert r["ok"] is True and r["state"] == "estimate" and r["ocp"] is None and r["estimate"]["faces"] == 12
    assert tree(run) == before
    assert server.export_step(run_dir=str(run))["missing_extra"] == "step"      # a call that writes still needs the extra


def fake_python(tmp_path, body):
    """An executable that stands for the Python of the runs: it runs `body` (shell) whatever its arguments are."""
    script = tmp_path / "fake_python"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@pytest.mark.skipif(sys.platform == "win32", reason="a shell script")
@pytest.mark.parametrize("body, code, signal_name, error, first_line", [
    ("echo 'partial output'; echo 'half a line' >&2; kill -SEGV $$", -11, "SIGSEGV", "half a line", "partial output"),
    ("echo 'not json at all'; exit 0", 0, None, "topopt step gave no JSON: ?", "not json at all"),
    ("echo 'Traceback (most recent call last):' >&2; echo '  File x' >&2; echo 'RuntimeError: boom' >&2; exit 2", 2, None, "RuntimeError: boom", None),
    ("echo 'error: line one' >&2; echo 'line two' >&2; exit 1", 1, None, "line one\nline two", None),
    ("exit 7", 7, None, "topopt step stopped with exit code 7", None),
], ids=["killed_by_a_signal", "exit_0_without_json", "internal_error", "usage_error", "other_code"])
def test_a_failure_reply_has_the_exit_code_the_signal_and_the_first_line_of_stdout(ws, monkeypatch, body, code, signal_name, error, first_line):
    run = make_run(ws, name="runs/j")
    monkeypatch.setenv("TOPOPT_PYTHON", str(fake_python(ws, body)))
    r = server.export_step(run_dir=str(run))
    assert r["ok"] is False and r["exit_code"] == code and r["error"] == error
    assert r.get("signal") == signal_name and r.get("stdout_first_line") == first_line
