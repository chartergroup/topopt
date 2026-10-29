"""Tests for the role playbooks (CONTRACTS.md section 25).

One source: topopt_mcp/roles/<id>.md. These tests check that the tool `team_guide`, the MCP prompts, the MCP
resources and the body of agents/<id>-engineer.md carry the same text, and that the wheel ships the files.
"""
import asyncio
import difflib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from helpers import ROLE_TEXT_MAX_CHARS  # noqa: E402
from topopt_mcp import server, team  # noqa: E402

ROLES_DIR = ROOT / "topopt_mcp" / "roles"
IDS = ["lead-design", "cad", "fea", "print", "finish"]
# Words of a talk or a demo day. No role text and no tool text holds them.
TALK_WORDS = re.compile(r"t-shirt|\bshirt\b|run mcp|tuesday|on stage|audience|slides?\b|demo day|live demo", re.I)
HABIT = re.compile(r"\b(must|never)\b", re.I)  # the freedom rule: facts and options, no habitual caps


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 60))


def file_text(role_id: str) -> str:
    """The playbook as the file has it: the body after the header, no outer blank lines."""
    raw = (ROLES_DIR / f"{role_id}.md").read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    return raw[4:].partition("\n---\n")[2].strip()


def agent_body(role_id: str) -> str:
    raw = (ROOT / "agents" / f"{role_id}-engineer.md").read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    return raw[4:].partition("\n---\n")[2].strip()


# --- the role files ----------------------------------------------------------------------------------------------

def test_one_role_file_for_each_plugin_role():
    assert sorted(p.stem for p in ROLES_DIR.glob("*.md")) == sorted(IDS)
    assert list(team.ROLE_IDS) == IDS
    assert sorted(p.stem for p in (ROOT / "agents").glob("*.md")) == sorted(f"{i}-engineer" for i in IDS)


def test_roles_have_a_title_a_one_line_summary_and_a_body():
    roles = team.read_roles(ROLES_DIR)
    assert list(roles) == IDS
    for role in roles.values():
        assert role.title and role.summary and "\n" not in role.summary and len(role.summary) <= 120
        assert role.text == file_text(role.id)
        assert 300 < len(role.text) < ROLE_TEXT_MAX_CHARS   # short playbooks (the plugin test uses the same constant)
        assert "does not exist yet" in role.text    # each role says what the tools cannot do yet
        assert role.uri == f"topopt://roles/{role.id}" and role.agent == f"{role.id}-engineer"


def test_role_texts_follow_the_freedom_rule_and_hold_no_talk_material():
    for role in team.read_roles(ROLES_DIR).values():
        for field in (role.title, role.summary, role.text):
            assert not HABIT.search(field), (role.id, HABIT.search(field).group())
            assert not TALK_WORDS.search(field), (role.id, TALK_WORDS.search(field).group())
        assert "${" not in role.text  # the manifest `prompts` field treats ${...} as a template variable
        assert role.text.rstrip().endswith(tuple(".)")), role.id


def test_parse_role_names_the_problem():
    ok = "---\ntitle: T\nsummary: S\n---\n\nBody.\n"
    assert team.parse_role("x", ok) == team.Role("x", "T", "S", "Body.")
    for raw, word in (("no header", "no header"), ("---\ntitle: T\nsummary: S\nBody", "closing"),
                      ("---\ntitle: T\n---\nBody", "summary"), ("---\nsummary: S\n---\nBody", "title"),
                      ("---\ntitle: T\nsummary: S\n---\n\n\n", "empty")):
        with pytest.raises(team.RoleFilesError, match=word):
            team.parse_role("x", raw)


def test_a_missing_file_is_one_error_with_the_path(tmp_path):
    (tmp_path / "lead-design.md").write_text("---\ntitle: T\nsummary: S\n---\nBody")
    with pytest.raises(team.RoleFilesError, match="cad"):
        team.read_roles(tmp_path)


# --- resolve and guide -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("cad", "cad"), ("CAD", "cad"), (" fea ", "fea"), ("lead-design", "lead-design"), ("lead_design", "lead-design"),
    ("cad-engineer", "cad"), ("topopt:fea-engineer", "fea"), ("lead-design-engineer", "lead-design"),
    ("lead", "lead-design"), ("pri", "print"),
    ("topopt://roles/cad", "cad"), ("TOPOPT://roles/Lead-Design", "lead-design"), (" topopt://roles/print ", "print"),
    ("topopt://roles/", None), ("topopt://roles/welder", None),
    ("", None), (None, None), ("   ", None), ("welder", None), ("c-ad", None), ("engineer", None),
])
def test_resolve_role(name, expected):
    role = team.resolve_role(name)
    assert (role.id if role else None) == expected


def test_guide_gives_the_playbook_of_a_role():
    for role_id in IDS:
        assert team.guide(role_id) == file_text(role_id)


def test_guide_without_a_role_or_with_an_unknown_one_lists_the_roles_and_does_not_raise():
    for name in (None, "", "welder"):
        text = team.guide(name)
        for role in team.load_roles().values():
            assert f"- {role.id}: {role.summary}" in text
        assert "team_guide(role)" in text and "topopt://roles/" in text
        assert not HABIT.search(text)
        assert ("welder" in text) == (name == "welder")  # the first line names a wrong name


# --- the MCP server, in process ----------------------------------------------------------------------------------

def test_team_guide_is_a_tool_with_a_short_description_without_habitual_caps():
    tools = {t.name: t for t in run(server.mcp.list_tools())}
    tool = tools["team_guide"]
    assert 100 < len(tool.description) < 1500
    assert not HABIT.search(tool.description) and not TALK_WORDS.search(tool.description)
    assert tool.input_schema["properties"]["role"]
    assert not tool.input_schema.get("required")  # a call without a role is allowed


def test_existing_tools_are_still_there():
    names = {t.name for t in run(server.mcp.list_tools())}
    assert {"run_topopt", "run_status", "run_result", "list_runs", "check_problem", "build_regions",
            "environment", "render_part"} <= names


def test_instructions_name_team_guide():
    """The one length bound of the instructions is in tests/test_mcp.py. This file adds no second bound."""
    assert "team_guide" in server.INSTRUCTIONS
    assert not HABIT.search(server.INSTRUCTIONS)


def test_tool_prompts_and_resources_equal_the_role_files():
    async def go():
        got = {}
        for role_id in IDS:
            res = await server.mcp.call_tool("team_guide", {"role": role_id})
            assert not res.is_error
            got[role_id] = [res.content[0].text]
        prompts = {p.name: p for p in await server.mcp.list_prompts()}
        resources = {str(r.uri): r for r in await server.mcp.list_resources()}
        assert sorted(prompts) == sorted(IDS)
        # the roles are served as resources; the material reference (CONTRACTS 25.7) is another resource, not a role
        assert sorted(u for u in resources if u.startswith("topopt://roles/")) == sorted(f"topopt://roles/{i}" for i in IDS)
        for role_id in IDS:
            prompt = await server.mcp.get_prompt(role_id)
            assert len(prompt.messages) == 1 and prompt.messages[0].role == "user"
            got[role_id].append(prompt.messages[0].content.text)
            contents = list(await server.mcp.read_resource(f"topopt://roles/{role_id}"))
            assert len(contents) == 1 and contents[0].mime_type == "text/markdown"
            got[role_id].append(contents[0].content)
            assert prompts[role_id].description == resources[f"topopt://roles/{role_id}"].description
            assert prompts[role_id].description == team.load_roles()[role_id].summary
        return got

    for role_id, texts in run(go()).items():
        assert texts == [file_text(role_id)] * 3, role_id


def test_team_guide_tool_without_a_role_returns_the_list_as_a_fact():
    async def go(args):
        res = await server.mcp.call_tool("team_guide", args)
        assert not res.is_error
        return res.content[0].text

    for args in ({}, {"role": None}, {"role": "welder"}, {"role": "Lead"}):
        text = run(go(args))
        if args.get("role") == "Lead":
            assert text == file_text("lead-design")
        else:
            assert all(f"- {i}: " in text for i in IDS)


def test_unreadable_role_files_give_the_cause_as_text_and_the_other_tools_work(monkeypatch):
    """CONTRACTS 25.3: with a broken install, `team_guide` answers with the cause as text (no hidden SDK error)."""
    def broken(folder=None):
        raise team.RoleFilesError("role cad: cannot read /x/roles/cad.md (No such file or directory)")

    team.load_roles.cache_clear()
    monkeypatch.setattr(team, "read_roles", broken)

    async def go():
        out = {}
        for args in ({}, {"role": "cad"}, {"role": "welder"}):
            res = await server.mcp.call_tool("team_guide", args)
            assert not res.is_error, args
            out[str(args)] = res.content[0].text
        names = {t.name for t in await server.mcp.list_tools()}
        return out, names

    try:
        out, names = run(go())
    finally:
        monkeypatch.undo()
        team.load_roles.cache_clear()
    for text in out.values():
        assert "role cad: cannot read /x/roles/cad.md" in text and "The other tools work." in text
    assert "run_topopt" in names and "team_guide" in names
    assert team.guide("cad") == file_text("cad")  # the cache is clean again


# --- the MCP server over stdio, with a real client --------------------------------------------------------------

def test_stdio_server_lists_and_serves_the_role_texts():
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(ROOT))

    async def go():
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            prompts = {p.name for p in (await client.list_prompts()).prompts}
            resources = {str(r.uri) for r in (await client.list_resources()).resources}
            listing = (await client.call_tool("team_guide", {})).content[0].text
            fea = (await client.call_tool("team_guide", {"role": "fea"})).content[0].text
            texts = {}
            for role_id in IDS:
                p = await client.get_prompt(role_id)
                r = await client.read_resource(f"topopt://roles/{role_id}")
                texts[role_id] = (p.messages[0].content.text, r.contents[0].text)
            return tools, prompts, resources, listing, fea, texts

    tools, prompts, resources, listing, fea, texts = run(go())
    assert "team_guide" in tools
    assert prompts == set(IDS) and {u for u in resources if u.startswith("topopt://roles/")} == {f"topopt://roles/{i}" for i in IDS}
    assert fea == file_text("fea")
    assert all(f"- {i}: " in listing for i in IDS)
    for role_id, (prompt, resource) in texts.items():
        assert prompt == resource == file_text(role_id)


# --- the agent files: one source, no drift ----------------------------------------------------------------------

def body_offset(path: Path) -> int:
    """Number of lines before the first line of the body (the header, its closing `---` and the blank lines)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    closing = lines.index("---", 1)
    first = next(i for i in range(closing + 1, len(lines)) if lines[i].strip())
    return first


def drift_message(role_id: str, role: str, agent: str) -> str:
    """The failure text: both files, the lines that differ, and how a line of the body maps to a line of the file."""
    role_file, agent_file = f"topopt_mcp/roles/{role_id}.md", f"agents/{role_id}-engineer.md"
    diff = "\n".join(difflib.unified_diff(role.splitlines(), agent.splitlines(), role_file, agent_file,
                                          lineterm="", n=0))
    return (f"{agent_file} differs from {role_file}. Put the same change in both files (the role file is the source).\n"
            f"The numbers after @@ are body lines (the text after the header). Add {body_offset(ROOT / role_file)} for a "
            f"line of {role_file} and {body_offset(ROOT / agent_file)} for a line of {agent_file}.\n{diff}")


@pytest.mark.parametrize("role_id", IDS)
def test_agent_body_equals_the_role_file(role_id):
    """agents/<id>-engineer.md carries the playbook of topopt_mcp/roles/<id>.md after its own header.

    After a merge that changes one of the two files, make them equal again: the role file is the source,
    so the change goes in topopt_mcp/roles/<id>.md and in the agent file.
    """
    agent, role = agent_body(role_id), file_text(role_id)
    if agent != role:
        pytest.fail(drift_message(role_id, role, agent))


def test_the_drift_message_names_both_files_the_line_and_the_body_numbering():
    message = drift_message("cad", "line one\nline two", "line one\nline two\nan added line")
    assert "agents/cad-engineer.md differs from topopt_mcp/roles/cad.md" in message
    assert "+an added line" in message and "body lines" in message
    offset = body_offset(ROOT / "topopt_mcp" / "roles" / "cad.md")
    # five lines come before body line 1: `---`, title, summary, `---` and a blank line
    assert offset == 5 and f"Add {offset} for a line of topopt_mcp/roles/cad.md" in message


def test_agent_header_keeps_its_own_fields():
    for role_id in IDS:
        raw = (ROOT / "agents" / f"{role_id}-engineer.md").read_text(encoding="utf-8")
        head = raw[4:].partition("\n---\n")[0]
        assert f"name: {role_id}-engineer" in head and "description:" in head
        assert "title:" not in head and "summary:" not in head  # those are role-file fields
        # one YAML comment names the source and the test (KB pattern no-copy-paste-forks)
        assert f"# The body is a copy of topopt_mcp/roles/{role_id}.md. tests/test_team_guide.py checks it." in head.splitlines()


# --- the wheel ships the role files -----------------------------------------------------------------------------

def test_wheel_ships_the_role_files_and_importlib_resources_reads_them(tmp_path):
    from test_packaging import _build_wheel

    wheel = _build_wheel(tmp_path / "dist")
    if wheel is None:
        pytest.skip("neither `uv build` nor `python -m build` could build a wheel here")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert {f"topopt_mcp/roles/{i}.md" for i in IDS} <= names
    assert "topopt_mcp/team.py" in names

    target = tmp_path / "installed"
    zipfile.ZipFile(wheel).extractall(target)
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(target)!r})\n"
        "from importlib import resources\n"
        "import topopt_mcp\n"
        "from topopt_mcp import team\n"
        f"assert topopt_mcp.__file__.startswith({str(target)!r}), topopt_mcp.__file__\n"
        "raw = {r: (resources.files('topopt_mcp') / 'roles' / (r + '.md')).read_text(encoding='utf-8') for r in team.ROLE_IDS}\n"
        "roles = team.read_roles()\n"
        "print(json.dumps({'raw': raw, 'text': {k: v.text for k, v in roles.items()}, 'guide': team.guide('fea')}))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    for role_id in IDS:
        assert data["raw"][role_id] == (ROLES_DIR / f"{role_id}.md").read_text(encoding="utf-8")
        assert data["text"][role_id] == file_text(role_id)
    assert data["guide"] == file_text("fea")
