"""Tests for the MCP server. The fake run stands in for the real optimizer."""
import asyncio
import json
import socket
import os
import signal
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt_mcp import server  # noqa: E402

# The wrapper stays alive after the run, like the real CLI that hosts the dashboard.
FAKE_CMD = f"{sys.executable} {REPO}/tests/serve_after_fake.py {{out}} --spi 0.05 --iterations 10"


@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("name: x\n")
    return p


@pytest.fixture(autouse=True)
def fake_cmd(monkeypatch):
    monkeypatch.setenv("TOPOPT_MCP_CMD", FAKE_CMD)
    server._current.clear()
    yield
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._current.clear()


def wait_done(call, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        st = call()
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def test_missing_yaml(tmp_path):
    r = server.run_topopt(str(tmp_path / "nope.yaml"), str(tmp_path / "o"))
    assert "error" in r and "not found" in r["error"]


def test_status_without_run(tmp_path):
    assert server.run_status(str(tmp_path / "none"))["state"] == "none"
    assert server.run_result(str(tmp_path / "none"))["done"] is False


def test_direct_flow_and_second_run_refused(yaml_file, tmp_path):
    out = tmp_path / "run1"
    r = server.run_topopt(str(yaml_file), str(out))
    assert r["run_dir"] == str(out)
    assert r["dashboard_url"].startswith("http://127.0.0.1:87")  # first free port from 8765
    assert isinstance(r["pid"], int)
    second = server.run_topopt(str(yaml_file), str(tmp_path / "run2"))
    assert "still running" in second["error"]
    assert server.run_result(str(out))["done"] is False
    st = wait_done(lambda: server.run_status(str(out)))
    assert st["state"] == "done", st
    assert st["summary"].startswith("done: iteration 10/10")
    assert "volume" in st["summary"]
    res = server.run_result(str(out))
    assert res["done"] is True
    assert Path(res["final_stl"]).is_absolute() and Path(res["final_stl"]).is_file()
    assert "mass_final_g" in res["metrics"]
    # The old process stays alive (it serves the dashboard), but it is not busy.
    old_pid = r["pid"]
    assert st["process_alive"] is True
    assert "stopped" not in st["summary"]
    # A new run stops the old process and starts.
    again = server.run_topopt(str(yaml_file), str(out))
    assert "pid" in again and again["pid"] != old_pid
    assert not server._pid_alive(old_pid)
    wait_done(lambda: server.run_status(str(out)))


def test_run_result_names_the_nominal_file_only_with_stock(tmp_path):
    """With machining stock (CONTRACTS.md section 23) the result also names result/nominal.stl; without it the keys are not there."""
    (tmp_path / "result").mkdir()
    (tmp_path / "status.json").write_text(json.dumps({"state": "done"}))
    (tmp_path / "result" / "metrics.json").write_text(json.dumps({"mass_final_g": 1.0}))
    (tmp_path / "result" / "final.stl").write_bytes(b"x")
    res = server.run_result(str(tmp_path), preview=False)
    assert res["done"] and "nominal_stl" not in res and "nominal_stl_rel" not in res
    (tmp_path / "result" / "nominal.stl").write_bytes(b"x")
    res = server.run_result(str(tmp_path), preview=False)
    assert Path(res["nominal_stl"]) == (tmp_path / "result" / "nominal.stl").resolve() and res["nominal_stl_rel"].endswith("nominal.stl")


def test_eval_no_wait_ends_the_job_process_after_the_run(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_MCP_CMD")
    monkeypatch.delenv("EVAL_TOPOPT_NO_WAIT", raising=False)
    y, o = tmp_path / "a.yaml", tmp_path / "o"
    assert "--no-wait" not in server._command(y, o, 8765)     # the default: the dashboard stays on after the run
    monkeypatch.setenv("EVAL_TOPOPT_NO_WAIT", "1")
    assert server._command(y, o, 8765, resume=True)[-5:] == ["--dashboard", "--port", "8765", "--no-wait", "--resume"]


def test_port(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_MCP_CMD")
    monkeypatch.delenv("EVAL_TOPOPT_NO_WAIT", raising=False)
    y, o = tmp_path / "a.yaml", tmp_path / "o"
    assert server._command(y, o, 8765)[-3:] == ["--dashboard", "--port", "8765"]
    # The returned URL uses the free port (fake command, no real run).
    monkeypatch.setenv("TOPOPT_PORT", "8899")
    monkeypatch.setenv("TOPOPT_MCP_CMD", FAKE_CMD)
    y.write_text("name: x\n")
    r = server.run_topopt(str(y), str(o))
    assert r["dashboard_url"] == f"http://127.0.0.1:{server._free_port(8899)}/"


def test_busy_port_moves_to_next(monkeypatch):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy = s.getsockname()[1]
    try:
        assert server._free_port(busy) > busy
    finally:
        s.close()


def test_relative_path_uses_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    assert server._resolve("runs/live") == tmp_path / "runs" / "live"
    (tmp_path / "p.yaml").write_text("name: x\n")
    r = server.run_topopt("p.yaml")
    assert r["run_dir"] == str(tmp_path / "runs" / "live")
    assert (tmp_path / "runs").is_dir()


def test_default_out_follows_workspace_then_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert server.default_out() == tmp_path.resolve() / "runs" / "live"
    ws = tmp_path / "ws"
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(ws))
    assert server.default_out() == ws.resolve() / "runs" / "live"
    assert REPO not in server.default_out().parents


def test_child_command_uses_sys_executable(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_MCP_CMD")
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    cmd = server._command(tmp_path / "a.yaml", tmp_path / "o", 8765)
    assert cmd[0] == sys.executable and cmd[1:4] == ["-m", "topopt", "run"]
    monkeypatch.setenv("TOPOPT_PYTHON", "/x/python")
    assert server._command(tmp_path / "a.yaml", tmp_path / "o", 8765)[0] == "/x/python"


def test_other_python_without_topopt_is_reported(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_MCP_CMD")
    monkeypatch.setenv("TOPOPT_PYTHON", str(tmp_path / "no_python"))
    y = tmp_path / "a.yaml"
    y.write_text("name: x\n")
    r = server.run_topopt(str(y), str(tmp_path / "o"))
    assert "cannot start" in r["error"] or "cannot import" in r["error"]


def test_unwritable_run_folder_gives_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    y = tmp_path / "a.yaml"
    y.write_text("name: x\n")
    r = server.run_topopt(str(y), str(blocker / "sub" / "run"))
    assert "error" in r and str(blocker) in r["error"]


def test_main_and_console_script():
    assert callable(server.main)
    import tomllib
    meta = tomllib.loads((REPO / "pyproject.toml").read_text())
    target = meta["project"]["scripts"]["topopt-mcp"]
    assert target == "topopt_mcp.server:main"
    module, func = target.split(":")
    import importlib
    assert callable(getattr(importlib.import_module(module), func))
    assert "topopt_mcp*" in meta["tool"]["setuptools"]["packages"]["find"]["include"]


def test_stdio_round_trip(yaml_file, tmp_path):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    out = tmp_path / "stdio_run"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "topopt_mcp.server"],
        cwd=str(REPO),
        env={"TOPOPT_MCP_CMD": FAKE_CMD},
    )

    def payload(res):
        if getattr(res, "structured_content", None):
            sc = res.structured_content
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        return json.loads(res.content[0].text)

    async def go():
        async with Client(params) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            assert {"run_topopt", "run_status", "run_result"} <= names
            r = payload(await client.call_tool(
                "run_topopt", {"yaml_path": str(yaml_file), "out_dir": str(out)}))
            assert r["run_dir"] == str(out) and r["pid"]
            for _ in range(300):
                st = payload(await client.call_tool("run_status", {"run_dir": str(out)}))
                if st["state"] in ("done", "error"):
                    break
                await asyncio.sleep(0.2)
            assert st["state"] == "done", st
            res = payload(await client.call_tool("run_result", {"run_dir": str(out)}))
            assert res["done"] is True
            assert res["final_stl"].endswith("final.stl")

    try:
        asyncio.run(go())
    finally:
        # The run process is detached: stop its group.
        try:
            os.killpg(int((tmp_path / "stdio_run.pid").read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass


TOOLS = ("run_topopt", "run_status", "run_result", "list_runs", "run_history", "compare_runs", "marks_to_hints",
         "apply_marks", "check_problem", "build_regions", "environment", "render_part", "verify_run", "verify_status",
         "record_brief", "case_status", "ratchet_run", "ratchet_status", "ratchet_approve", "cleanup_list", "stop_run")
# Tools that the instructions do not name: the instructions are at their bound (under 3,000 characters), so the description is the only text.
NOT_IN_INSTRUCTIONS = ("save_preferences",)


def test_instructions_name_tools_and_units():
    text = server.mcp.instructions
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert len(text) <= OLD_CLIENT_CUT
    # #380: the instructions (2048 characters) name the core tools in the flow and the others in the last line; a status or approve tool is named
    # in the description of the tool that starts the job (see MOVED_RULES of tests/test_plugin_manifest.py)
    in_text = [t for t in TOOLS if t not in ("verify_status", "ratchet_status", "ratchet_approve")]
    for t in in_text:
        assert t in text, t
    descriptions = {t.name: t.description for t in asyncio.run(server.mcp.list_tools())}
    assert "verify_status" in descriptions["verify_run"] and "ratchet_status" in descriptions["ratchet_run"] and "ratchet_approve" in descriptions["ratchet_run"]
    assert "mm, N, MPa, g" in text
    assert "assumptions" in text
    assert "must" not in text.lower() and "never" not in text.lower()


def test_every_tool_has_description():
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert set(TOOLS) | set(NOT_IN_INSTRUCTIONS) <= names
    for t in tools:
        assert (t.description or "").strip(), t.name


def test_the_tool_lists_of_the_server_docstring_the_mcp_readme_and_the_clients_page_name_every_tool():
    """Q1-06 and X3: one list of tools. A new tool must show in the docstring, in topopt_mcp/README.md (a bullet) and in docs/clients.md."""
    import re
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert "run_topopt" in names and "verify_status" in names           # the list is not empty or cut
    readme = (REPO / "topopt_mcp" / "README.md").read_text()
    bullets = set(re.findall(r"^- `(\w+)\(", readme, re.M))
    bullets |= set(re.findall(r"^- `\w+\(.*?\)` and `(\w+)\(", readme, re.M))      # two tools in one bullet: `a(...)` and `b(...)`
    assert names <= bullets, sorted(names - bullets)
    docstring = server.__doc__
    assert {n for n in names if not re.search(rf"\b{n}\b", docstring)} == set()
    clients = (REPO / "docs" / "clients.md").read_text()
    assert {n for n in names if f"`{n}`" not in clients} == set()


def test_initialize_result_has_instructions():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(REPO))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                res = await session.initialize()
                return res.instructions

    assert asyncio.run(asyncio.wait_for(go(), 60)) == server.mcp.instructions


def test_run_command_goes_through_topopt_main(monkeypatch, tmp_path):
    """`python -m topopt` ends in cli.main(), which caps the OpenBLAS threads on Linux."""
    monkeypatch.delenv("TOPOPT_MCP_CMD", raising=False)
    cmd = server._command(tmp_path / "a.yaml", tmp_path / "o", 8765)
    assert cmd[1:4] == ["-m", "topopt", "run"]
    assert "from .cli import main" in (REPO / "topopt" / "__main__.py").read_text()
