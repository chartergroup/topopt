"""Tests for the strict argument names of every MCP tool (#174, step 4; CONTRACTS.md section 10, "Strict argument names").

A call with a name that the tool does not have is refused before the tool runs: `{"ok": false, "error": "<tool> did nothing: ..."}`.
A key that a client adds to the schema of every tool for its own use (`CLIENT_CONTROL_KEYS`, today `wait_for_previous` of Gemini CLI) is
dropped first, but only when the tool has no parameter of that name. The check reads the parameter names of the tool, not the schema keyword.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt_mcp import server  # noqa: E402

CONTROL = "wait_for_previous"


def listed_tools():
    return {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}


def call(srv, name, arguments):
    return asyncio.run(srv.call_tool(name, arguments))


def reply_of(result) -> dict:
    return json.loads(result.content[0].text)


def misspelled(tool) -> str:
    """A wrong name for this tool: its first argument name with one letter dropped, or a name of no tool when it has no argument."""
    names = list(tool.input_schema.get("properties", {}))
    return names[0][:-1] if names and len(names[0]) > 2 else "zz_typo"


@pytest.fixture
def work_folder(tmp_path, monkeypatch):
    folder = tmp_path / "work"
    folder.mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(folder))
    return folder


class Recorder:
    """Stands in for the function of a tool: it records each call."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


def test_every_registered_tool_has_additional_properties_false_in_its_input_schema():
    tools = listed_tools()
    assert len(tools) == len(server.TOOL_SURFACE)
    assert [name for name, tool in tools.items() if tool.input_schema.get("additionalProperties") is not False] == []


def test_every_tool_refuses_a_misspelled_name_with_the_valid_names_and_does_not_run(work_folder, monkeypatch):
    for name, tool in listed_tools().items():
        wrong = misspelled(tool)
        recorder = Recorder()
        monkeypatch.setattr(server.mcp._tool_manager.get_tool(name), "fn", recorder)
        reply = reply_of(call(server.mcp, name, {wrong: 1}))
        assert reply["ok"] is False, name
        assert reply["error"].startswith(f"{name} did nothing: it has no argument '{wrong}'."), (name, reply["error"])
        names = list(tool.input_schema.get("properties", {}))
        expected = f"The argument names are: {', '.join(names)}." if names else "It takes no arguments."
        assert expected in reply["error"], (name, reply["error"])
        assert CONTROL not in reply["error"], name
        assert recorder.calls == [], f"{name} ran on a refused call"
    assert list(work_folder.iterdir()) == [], "a refused call made a file or a folder in the work folder"


def test_the_refusal_names_each_unknown_key():
    reply = reply_of(call(server.mcp, "run_status", {"zz_b": 1, "zz_a": 2}))
    assert "no argument 'zz_a', 'zz_b'." in reply["error"]


def test_a_control_key_is_dropped_and_the_tool_runs_as_without_it(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(server.mcp._tool_manager.get_tool("list_runs"), "fn", recorder)
    assert reply_of(call(server.mcp, "list_runs", {CONTROL: True})) == {"ok": True}
    assert recorder.calls == [{}]


def test_a_control_key_is_dropped_for_a_tool_that_has_arguments(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(server.mcp._tool_manager.get_tool("find_features"), "fn", recorder)
    assert reply_of(call(server.mcp, "find_features", {"stl_path": "a.stl", CONTROL: False})) == {"ok": True}
    assert recorder.calls == [{"stl_path": "a.stl"}]


def test_an_unknown_key_next_to_a_control_key_is_refused_and_only_the_unknown_key_is_named(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(server.mcp._tool_manager.get_tool("find_features"), "fn", recorder)
    reply = reply_of(call(server.mcp, "find_features", {"stl_path": "a.stl", CONTROL: True, "overide": 1}))
    assert reply["ok"] is False and "no argument 'overide'." in reply["error"]
    assert CONTROL not in reply["error"]
    assert "The argument names are: stl_path" in reply["error"]
    assert recorder.calls == []


def test_a_tool_parameter_with_the_name_of_a_control_key_gets_its_value():
    fresh = server._TopoptServer("test")

    @fresh.tool()
    def case_status(wait_for_previous: bool = False, other: int = 0) -> dict:
        """A tool with a parameter that has the name of a control key."""
        return {"ok": True, "wait_for_previous": wait_for_previous, "other": other}

    reply = reply_of(call(fresh, "case_status", {CONTROL: True}))
    assert reply == {"ok": True, "wait_for_previous": True, "other": 0}
    refused = reply_of(call(fresh, "case_status", {CONTROL: True, "zz": 1}))
    assert refused["ok"] is False and "no argument 'zz'." in refused["error"]
    assert "The argument names are: wait_for_previous, other." in refused["error"]


@pytest.mark.parametrize("name, arguments", [
    ("save_preferences", {"print": {"min_wall_mm": 1.0}}),
    ("save_printer", {"profile": {"name": "x"}}),
    ("save_material", {"profile": {"name": "x"}}),
])
def test_a_save_tool_that_was_strict_before_runs_with_a_control_key(monkeypatch, name, arguments):
    """The Gemini CLI case for the 3 tools that were strict before #174: a call with `wait_for_previous` was refused."""
    recorder = Recorder()
    monkeypatch.setattr(server.mcp._tool_manager.get_tool(name), "fn", recorder)
    assert reply_of(call(server.mcp, name, {**arguments, CONTROL: True})) == {"ok": True}
    assert len(recorder.calls) == 1 and CONTROL not in recorder.calls[0]       # the SDK also passes the defaults of the other arguments
    assert {key: recorder.calls[0][key] for key in arguments} == arguments


def test_a_tool_registered_under_another_name_is_strict_under_that_name():
    fresh = server._TopoptServer("test")

    @fresh.tool(name="case_status")
    def some_other_function_name(other: int = 0) -> dict:
        """A tool whose name is not the name of its function."""
        return {"ok": True, "other": other}

    assert reply_of(call(fresh, "case_status", {"other": 2})) == {"ok": True, "other": 2}
    refused = reply_of(call(fresh, "case_status", {"zz": 1}))
    assert refused["ok"] is False and refused["error"].startswith("case_status did nothing: it has no argument 'zz'.")


def test_the_refusal_does_not_depend_on_the_schema_keyword(monkeypatch):
    tool = server.mcp._tool_manager.get_tool("find_features")
    schema = dict(tool.parameters)
    schema.pop("additionalProperties")
    monkeypatch.setattr(tool, "parameters", schema)
    assert "additionalProperties" not in tool.parameters
    recorder = Recorder()
    monkeypatch.setattr(tool, "fn", recorder)
    reply = reply_of(call(server.mcp, "find_features", {"stl_path": "a.stl", "zz": 1}))
    assert reply["ok"] is False and "no argument 'zz'." in reply["error"]
    assert recorder.calls == []


def test_strict_arguments_false_turns_the_check_off_for_one_tool():
    fresh = server._TopoptServer("test")

    @fresh.tool(strict_arguments=False)
    def case_status(a: int = 0) -> dict:
        """A tool that is not strict."""
        return {"ok": True, "a": a}

    assert "additionalProperties" not in fresh._tool_manager.get_tool("case_status").parameters
    assert reply_of(call(fresh, "case_status", {"a": 2, "zz": 1})) == {"ok": True, "a": 2}


def test_the_entries_of_the_control_key_list_have_a_source_and_a_read_date():
    source = (ROOT / "topopt_mcp" / "server.py").read_text(encoding="utf-8")
    block = source[source.index("CLIENT_CONTROL_KEYS: tuple"):]
    block = block[:block.index("\n)\n")]
    for key in server.CLIENT_CONTROL_KEYS:
        assert f'"{key}"' in block
    assert "read 20" in block and "https://" in block, "each entry needs its source and its read date in the comment"


# --- the real server over stdio --------------------------------------------------------------------------------------

def run_over_stdio(tmp_path, calls):
    """Start the real server (`python -m topopt_mcp.server`) with a work folder in tmp_path and make each call. Returns the results."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ, TOPOPT_WORKSPACE=str(work), HOME=str(home), PYTHONPATH=str(ROOT), PYTHONSAFEPATH="1")

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(work), env=env)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                return [await session.call_tool(name, arguments) for name, arguments in calls]

    return asyncio.run(asyncio.wait_for(go(), 90)), work


def test_over_stdio_a_misspelled_name_makes_no_run_and_render_part_and_team_guide_refuse(tmp_path):
    yaml_file = tmp_path / "p.yaml"
    yaml_file.write_text("not: read\n")
    results, work = run_over_stdio(tmp_path, [
        ("run_topopt", {"yaml_path": str(yaml_file), "overide_lock": True}),
        ("render_part", {"run_dirr": str(tmp_path)}),
        ("team_guide", {"rol": "fea"}),
        ("team_guide", {"role": "fea", CONTROL: True}),
    ])
    for result, tool in zip(results[:3], ("run_topopt", "render_part", "team_guide")):
        assert not result.is_error
        reply = json.loads(result.content[0].text)
        assert reply["ok"] is False and reply["error"].startswith(f"{tool} did nothing: it has no argument ")
    assert "'overide_lock'" in json.loads(results[0].content[0].text)["error"]
    assert not results[3].is_error and results[3].content[0].text.startswith("You are the FEA engineer of a topopt job.")   # the guide
    assert not (work / "runs").exists() and not list(tmp_path.glob("**/run.json"))
    assert [p.name for p in work.iterdir()] == [], "a refused call made a file in the work folder"
