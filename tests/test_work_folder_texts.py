"""P9-1b: every text that tells a person how to change the work folder uses the words of the surface that started the server.

The Desktop extension sets `TOPOPT_DEFAULT_WORKSPACE` (its manifest is the only launcher that does). A Desktop user has the field
"Work folder", not an env var: no reply, tool description, skill or role may tell that user to set `TOPOPT_WORKSPACE`. Every other start
(the Claude Code plugin, a checkout, Cursor, Codex CLI, ...) gets one text that is true for all of them.

The replies come from the tool functions of `topopt_mcp.server` (and from `topopt.case`). Nothing here starts a solver.
"""
import ast
import json
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import case, workfolder  # noqa: E402
from topopt_mcp import server  # noqa: E402
from test_optimize import make_case  # noqa: E402

DESKTOP_TEXT = "Open Settings > Extensions > topopt, choose the Work folder with the folder button"
OTHER_TEXT = "Start the client in the folder that you want to use as the work folder"
VARIABLE = "TOPOPT_WORKSPACE"


class Surface:
    def __init__(self, name, home, work):
        self.name, self.home, self.work = name, home, work


@pytest.fixture(params=["desktop", "other"])
def surface(request, tmp_path, monkeypatch):
    """The server as the Desktop extension starts it (TOPOPT_DEFAULT_WORKSPACE set, the option empty), and as any other client starts it
    (no variable; the current folder is the work folder: the plugin route)."""
    home = tmp_path / "home"
    work = home / "Documents" / "topopt"
    work.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TOPOPT_WORKSPACE", raising=False)
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    if request.param == "desktop":
        monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", str(work))
        monkeypatch.chdir(tmp_path)                       # the folder of the extension: not the work folder
    else:
        monkeypatch.chdir(work)
    server._current.clear()
    yield Surface(request.param, home, work)
    server._current.clear()


def assert_the_text_of(surface, text):
    """`text` holds the words of the surface and none of the other surface."""
    assert workfolder.change_text() in text
    if surface.name == "desktop":
        assert DESKTOP_TEXT in text and "Cmd+Q" in text and "A new chat or a new Code tab session is not enough" in text
        assert VARIABLE not in text and OTHER_TEXT not in text
    else:
        assert OTHER_TEXT in text and VARIABLE in text
        assert "Settings > Extensions" not in text and "Cmd+Q" not in text


# --- the helper --------------------------------------------------------------------------------------------------

def test_the_surface_is_told_by_the_default_variable_only():
    assert workfolder.is_desktop_extension({}) is False
    assert workfolder.is_desktop_extension({"TOPOPT_WORKSPACE": "/x"}) is False        # a set work folder is no sign of a surface
    assert workfolder.is_desktop_extension({"TOPOPT_DEFAULT_WORKSPACE": " "}) is False
    assert workfolder.is_desktop_extension({"TOPOPT_DEFAULT_WORKSPACE": "/home/x/Documents/topopt"}) is True
    assert workfolder.fix_text({}) == workfolder.FIX_OTHER
    assert workfolder.fix_text({"TOPOPT_DEFAULT_WORKSPACE": "/x"}) == workfolder.FIX_DESKTOP
    assert workfolder.change_text({}) == "To change the work folder: " + workfolder.FIX_OTHER
    assert workfolder.change_text({"TOPOPT_DEFAULT_WORKSPACE": "/x"}) == "To change the work folder: " + workfolder.FIX_DESKTOP


def test_the_two_fix_texts_name_the_ways_that_each_surface_has():
    desktop, other = workfolder.FIX_DESKTOP, workfolder.FIX_OTHER
    # The Desktop extension: the field, the full quit. No env var: the user cannot set one.
    assert "Work folder" in desktop and "Cmd+Q" in desktop and VARIABLE not in desktop
    # The other surfaces: the plugin has no env to edit and starts in the folder where Claude Code runs; the clients with a config
    # have an env for the server. One text names both ways, and the restart.
    assert "Start the client in the folder" in other and f"set {VARIABLE} to a full path" in other and "restart the server" in other


def test_only_the_manifest_of_the_desktop_extension_sets_the_sign():
    """The sign is a variable that the launcher of one surface sets. If another launcher sets it, the two texts are wrong."""
    sign = "TOPOPT_DEFAULT_WORKSPACE"
    manifest = json.loads((REPO / "packaging" / "mcpb" / "manifest.json").read_text())
    assert sign in manifest["server"]["mcp_config"]["env"]
    plugin = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
    assert "env" not in plugin["mcpServers"]["topopt"]                      # the plugin gives the server no env
    assert json.loads((REPO / ".mcp.json").read_text())["mcpServers"]["topopt"]["env"] == {}
    assert set(json.loads((REPO / ".cursor" / "mcp.json").read_text())["mcpServers"]["topopt"]["env"]) == {VARIABLE}
    codex = tomllib.loads((REPO / ".codex" / "config.toml").read_text())
    assert sign not in codex["mcp_servers"]["topopt"]["env"]
    for launcher in ("scripts/topopt-env.sh", ".claude-plugin/marketplace.json"):
        assert sign not in (REPO / launcher).read_text(), launcher


# --- the replies, once for each surface --------------------------------------------------------------------------

@pytest.mark.parametrize("tool, args", [
    (server.run_topopt, ("missing.yaml",)),
    (server.record_brief, ("missing.yaml", "A shelf bracket.")),
    (server.case_status, ("missing.yaml",)),
], ids=["run_topopt", "record_brief", "case_status"])
def test_a_yaml_that_is_not_there_gives_the_two_ways_out_in_the_words_of_the_surface(surface, tool, args):
    error = tool(*args)["error"]
    assert error.startswith("YAML file not found: ") and f"(work folder: {surface.work.resolve()})" in error
    assert "Give an absolute path." in error                                  # the way that works on every surface stays
    assert_the_text_of(surface, error)


def test_a_run_folder_in_the_work_folder_that_cannot_be_made_gives_the_two_ways_out_in_the_words_of_the_surface(surface, tmp_path, monkeypatch):
    """The check says the folder can be made, and the system refuses (on a Mac: the privacy setting of ~/Documents)."""
    (tmp_path / "p.yaml").write_text("name: x\n")
    refused = (surface.work / "runs").resolve()
    real_mkdir = Path.mkdir

    def mkdir(self, *args, **kwargs):
        if self == refused:
            raise PermissionError(1, "Operation not permitted", str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    error = server.run_topopt(str(tmp_path / "p.yaml"))["error"]               # the default run folder: runs/live in the work folder
    assert error.startswith(f"Cannot create the run folder {refused}: ") and "Operation not permitted" in error
    assert "Give out_dir a folder that you can write." in error                # the way that works on every surface stays
    assert_the_text_of(surface, error)
    assert "proc" not in server._current


def test_a_run_folder_outside_the_work_folder_that_cannot_be_made_does_not_offer_another_work_folder(surface, tmp_path):
    """The caller chose that folder: a change of the work folder (and a full quit of Claude) would not help."""
    (tmp_path / "p.yaml").write_text("name: x\n")
    (tmp_path / "blocker").write_text("a file, not a folder\n")
    error = server.run_topopt(str(tmp_path / "p.yaml"), str(tmp_path / "blocker" / "sub" / "run"))["error"]
    assert error.startswith("Cannot create the run folder ") and error.endswith("Give out_dir a folder that you can write.")
    assert "work folder" not in error and VARIABLE not in error and "Settings" not in error
    assert "proc" not in server._current


def test_the_option_for_a_yaml_in_the_code_folder_is_in_the_words_of_the_surface(surface, monkeypatch):
    monkeypatch.setattr(server, "REPO", surface.work)                         # a checkout whose folder is the work folder
    d = surface.work / "examples" / "part"
    d.mkdir(parents=True)
    make_case(d)
    r = server.record_brief("examples/part/c.yaml", "A shelf bracket, 20 N at the tip.", None)
    assert r["recorded"] is True and len(r["options"]) == 1
    assert r["options"][0].startswith("Copy the YAML and the STL files into a work folder, then call record_brief for the copy. ")
    assert_the_text_of(surface, r["options"][0])
    # the same string through topopt.case, with no tool
    again = case.record_brief((d / "c.yaml").resolve(), surface.work, "A shelf bracket, 20 N at the tip.", None, code_folder=surface.work)
    assert again["options"] == r["options"]


def test_the_scratch_workspace_note_is_in_the_words_of_the_surface(surface, tmp_path, monkeypatch):
    scratch = tmp_path / "mac" / "Library" / "Application Support" / "Claude" / "scratch-workspaces" / "id1" / "id1" / "scratch-2026-10-23-ab12cd"
    scratch.mkdir(parents=True)
    if surface.name == "desktop":
        monkeypatch.setenv("TOPOPT_WORKSPACE", str(scratch))                  # a person typed a folder in the scratch workspace
    else:
        monkeypatch.chdir(scratch)
    note = server.list_runs()["work_folder_note"]
    assert "removes this folder when the session ends" in note
    assert_the_text_of(surface, note)
    assert ("Choose a folder that you keep." in note) == (surface.name == "desktop")
    assert ("Open the session in a project folder that you keep." in note) == (surface.name == "other")


def test_a_work_folder_with_its_own_fix_text_says_the_fix_once_in_a_reply(tmp_path, monkeypatch):
    """The work folder cannot be used: `work_folder_note` holds the fix. The error does not repeat it."""
    for name in ("TOPOPT_WORKSPACE", "TOPOPT_DEFAULT_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${FOO}/x")                         # a variable that nobody replaces
    for default in (None, str(tmp_path / "Documents" / "topopt")):             # another client; the Desktop extension
        if default:
            monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", default)
        reply = server.case_status("missing.yaml")
        fix = workfolder.fix_text()
        assert json.dumps(reply).count(fix) == 1, reply
        assert reply["error"].endswith("Give an absolute path.") and fix in reply["work_folder_note"]


# --- no text tells a Desktop user to set the variable ----------------------------------------------------------------

def string_constants(path):
    """`(text, is_docstring)` for each string in the Python file, f-string parts included."""
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, id(node) in docs


def test_the_variable_is_named_in_the_code_by_workfolder_only():
    """A string of the code (not a docstring) that names the variable is a reply text or a text for the agent: only `workfolder.py` has
    them (the name of the variable, and the text for the surfaces that are not the Desktop extension)."""
    found = {}
    for path in sorted([*(REPO / "topopt").rglob("*.py"), *(REPO / "topopt_mcp").rglob("*.py")]):
        hits = [text for text, doc in string_constants(path) if VARIABLE in text and not doc]
        if hits:
            found[path.relative_to(REPO).as_posix()] = hits
    assert set(found) == {"topopt/workfolder.py"}, found
    assert sorted(found["topopt/workfolder.py"]) == sorted([VARIABLE, workfolder.FIX_OTHER])


def test_no_tool_description_and_no_instruction_names_the_variable():
    import asyncio
    tools = asyncio.run(server.mcp.list_tools())
    assert len(tools) >= 20
    for tool in tools:
        assert VARIABLE not in (tool.description or ""), tool.name
    assert VARIABLE not in server.INSTRUCTIONS


def test_the_run_topopt_description_points_to_environment_for_the_work_folder():
    import asyncio
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    text = tools["run_topopt"].description
    assert "relative paths start at the work folder; `environment` shows it" in text
    assert len(text) <= 4096


SKILL_AND_ROLE_FILES = sorted([
    *(REPO / "skills").rglob("*.md"), *(REPO / ".claude" / "skills").rglob("*.md"), *(REPO / ".agents" / "skills").rglob("*.md"),
    *(REPO / "agents").glob("*.md"), *(REPO / "topopt_mcp" / "roles").glob("*.md")])


def test_the_skills_and_the_roles_do_not_tell_the_user_to_set_the_variable():
    assert len(SKILL_AND_ROLE_FILES) >= 8
    for path in SKILL_AND_ROLE_FILES:
        assert VARIABLE not in path.read_text(), path.relative_to(REPO)


def test_the_manifest_text_for_the_user_does_not_name_the_variable():
    manifest = json.loads((REPO / "packaging" / "mcpb" / "manifest.json").read_text())
    shown = json.dumps([manifest["user_config"], manifest["long_description"], manifest["description"]])
    assert VARIABLE not in shown


def section(text, heading, level):
    """The lines under `heading` up to the next heading of the same level or a higher one. A line in a code fence is no heading."""
    out, on, fence = [], False, False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        depth = len(line) - len(line.lstrip("#")) if line.startswith("#") and not fence else 0
        if depth and on and depth <= level:
            break
        if depth and line.strip() == heading:
            on = True
            continue
        if on:
            out.append(line)
    return "\n".join(out)


def test_the_install_page_has_one_section_for_the_work_folder_with_a_part_for_each_surface():
    """#407: the section moved from the README to the install page; the README keeps a short paragraph and a link."""
    readme = (REPO / "README.md").read_text()
    body = section((REPO / "docs" / "reference" / "install.md").read_text(), "## The work folder", 2)
    assert body, "docs/reference/install.md needs the section '## The work folder'"
    parts = {name: section(body, f"### {name}", 3) for name in
             ("Claude Desktop extension", "Claude Code plugin", "The command line", "Another MCP client")}
    assert all(parts.values()), [name for name, text in parts.items() if not text]
    desktop = parts["Claude Desktop extension"]
    assert "Work folder" in desktop and "Cmd+Q" in desktop and VARIABLE not in desktop       # the field; a full quit; no env var
    assert VARIABLE in parts["Another MCP client"]
    assert "Not checked" in body                                                              # what has not run on a Mac
    # the Desktop quick start does not tell a Desktop user to set the variable either
    # (#385: the steps moved from the README to the install page; the README keeps a pointer)
    install = (REPO / "docs" / "reference" / "install.md").read_text()
    readme_desktop = readme.split("<summary><b>Claude Desktop</b></summary>", 1)[1].split("</details>", 1)[0]
    for desktop_steps in (readme_desktop, section(install, "## Claude Desktop", 2)):
        assert desktop_steps and VARIABLE not in desktop_steps
    assert "(docs/reference/install.md#the-work-folder)" in readme
    assert "docs/reference/install.md#the-work-folder" in (REPO / "topopt_mcp" / "README.md").read_text()
    assert "README.md#the-work-folder" not in install and "README.md#the-work-folder" not in (REPO / "topopt_mcp" / "README.md").read_text()
