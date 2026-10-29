"""Tests for the configs of other agent clients (CONTRACTS.md section 34, docs/clients.md).

Shipped config files: `.cursor/mcp.json` (Cursor) and `.codex/config.toml` (Codex CLI).
Snippets in docs/clients.md: `.vscode/mcp.json` (VS Code) and `.gemini/settings.json` (Gemini CLI).
Only Cursor had a run in the real client: one Mac test (2026-10-27, docs/clients.md, section "Cursor").
The tests check what a sandbox can check:

- each config parses and has only the keys that the client documents;
- each config starts the same launcher as the Claude Code plugin;
- a generic MCP client (the `mcp` SDK that the repo already needs) starts the server with the exact command, args and
  env of the config, lists the tools, reads each role text through `team_guide`, and checks the hook example;
- the skills meet the Agent Skills spec, and the copies in `.agents/skills/` equal their sources byte for byte
  (the check is in `scripts/sync_agent_skills.py`, which also makes the copies).

The launch uses the real `scripts/topopt-env.sh`. A fake `uv` makes the environment (a `python` that runs the Python
of the test with the repo under test on `PYTHONPATH`), so the tests need no network and make no `.plugin-env` folder
in the repo.
"""
import asyncio
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt_mcp import team  # noqa: E402  (standard library only)


def _load_helper():
    """scripts/sync_agent_skills.py: it makes the copies and checks them. `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("sync_agent_skills", ROOT / "scripts" / "sync_agent_skills.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sync_agent_skills"] = module
    spec.loader.exec_module(module)
    return module


sync_agent_skills = _load_helper()

LAUNCHER = "scripts/topopt-env.sh"
CLIENTS_DOC = ROOT / "docs" / "clients.md"
HOOK_YAML = "examples/hook/hook.yaml"
EXPECTED_TOOLS = {"run_topopt", "run_status", "run_result", "list_runs", "check_problem", "build_regions",
                  "environment", "render_part", "verify_run", "verify_status", "team_guide", "stop_run"}

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher is a POSIX sh script")

# The keys that each client documents for a stdio server (read on 2026-10-20; the URLs are in the plan).
# The Codex code reads more keys than the docs list (for example `startup_timeout_ms`). The test follows the docs.
CURSOR_KEYS = {"type", "command", "args", "env", "envFile"}                      # cursor.com/docs/mcp
CODEX_KEYS = {"command", "args", "env", "env_vars", "cwd", "startup_timeout_sec", "tool_timeout_sec", "enabled",
              "required", "enabled_tools", "disabled_tools", "default_tools_approval_mode", "experimental_environment",
              "tools"}                    # developers.openai.com/codex/mcp (`tools.<tool>`: approval_mode, output_token_limit)
VSCODE_KEYS = {"type", "command", "args", "cwd", "env", "envFile", "dev", "sandboxEnabled", "sandbox"}  # code.visualstudio.com
GEMINI_KEYS = {"command", "args", "env", "cwd", "timeout", "trust", "includeTools", "excludeTools"}     # geminicli.com
# Variables that the client expands (Cursor and VS Code). Codex and Gemini have none for `command` and `args`.
WORKSPACE_VARS = {"workspaceFolder", "workspaceFolderBasename", "userHome", "pathSeparator", "/"}


# --- reading the configs -----------------------------------------------------------------------------------------

def read_cursor() -> dict:
    data = json.loads((ROOT / ".cursor" / "mcp.json").read_text())
    assert set(data) == {"mcpServers"}
    return data["mcpServers"]["topopt"]


def read_codex() -> dict:
    data = tomllib.loads((ROOT / ".codex" / "config.toml").read_text())
    assert set(data) == {"mcp_servers"}
    return data["mcp_servers"]["topopt"]


def doc_snippet(name: str) -> str:
    """The text of the code block that follows `<!-- config: NAME -->` in docs/clients.md."""
    text = CLIENTS_DOC.read_text()
    m = re.search(r"<!-- config: " + re.escape(name) + r" -->\s*```\w*\n(.*?)\n```", text, re.S)
    assert m, f"docs/clients.md has no snippet for {name}"
    return m.group(1)


def read_vscode() -> dict:
    data = json.loads(doc_snippet(".vscode/mcp.json"))
    assert set(data) == {"servers"}
    return data["servers"]["topopt"]


def read_gemini() -> dict:
    data = json.loads(doc_snippet(".gemini/settings.json"))
    assert set(data) == {"mcpServers"}
    return data["mcpServers"]["topopt"]


READERS = {"cursor": read_cursor, "codex": read_codex, "vscode": read_vscode, "gemini": read_gemini}
SHIPPED = ("cursor", "codex")           # files in the repo; the other two are snippets in docs/clients.md


def expand(text: str, workspace: Path) -> str:
    """What Cursor and VS Code do with `${workspaceFolder}` (the other variables of the list are not used)."""
    return text.replace("${workspaceFolder}", str(workspace))


def variables(entry: dict) -> set[str]:
    found = set()
    for value in [entry["command"], *entry.get("args", []), *entry.get("env", {}).values()]:
        found |= set(re.findall(r"\$\{([^}]+)\}", value))
    return found


# --- schema ------------------------------------------------------------------------------------------------------

def test_cursor_config_has_only_documented_keys_and_stdio_fields():
    entry = read_cursor()
    assert set(entry) <= CURSOR_KEYS
    assert entry["type"] == "stdio"
    assert isinstance(entry["command"], str) and entry["command"]
    assert isinstance(entry["args"], list) and all(isinstance(a, str) for a in entry["args"])
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in entry["env"].items())
    assert variables(entry) <= WORKSPACE_VARS     # `${env:X}` would also be valid; this file needs none


def test_codex_config_has_only_documented_keys_and_types():
    entry = read_codex()
    assert set(entry) <= CODEX_KEYS
    assert isinstance(entry["command"], str) and entry["command"]
    assert isinstance(entry["args"], list) and all(isinstance(a, str) for a in entry["args"])
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in entry["env"].items())
    # Both timeouts are set, because the Codex docs and the Codex code differ on the defaults.
    # Codex reads a number of seconds (a float in the code), so 240 and 240.0 are both valid. A bool is not.
    for key in ("startup_timeout_sec", "tool_timeout_sec"):
        assert isinstance(entry[key], (int, float)) and not isinstance(entry[key], bool) and entry[key] > 0, key
    assert "${" not in json.dumps(entry)          # Codex has no variable expansion in this file


def test_vscode_snippet_has_only_documented_keys_and_stdio_fields():
    entry = read_vscode()
    assert set(entry) <= VSCODE_KEYS
    assert entry["type"] == "stdio" and variables(entry) <= WORKSPACE_VARS


def test_gemini_snippet_has_only_documented_keys():
    entry = read_gemini()
    assert set(entry) <= GEMINI_KEYS
    assert "${" not in entry["command"] and not any("${" in a for a in entry["args"])  # variables work in `env` only
    # `timeout` is in ms. A number below the first start (60 s) would cut it; the snippet does not set one.
    assert "timeout" not in entry


@pytest.mark.parametrize("client", READERS)
def test_every_config_starts_the_launcher_of_the_plugin(client):
    """The plugin runs `sh ${CLAUDE_PLUGIN_ROOT}/scripts/topopt-env.sh mcp`. Each client config runs the same script."""
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())["mcpServers"]["topopt"]
    plugin_script = plugin["args"][0].replace("${CLAUDE_PLUGIN_ROOT}/", "")
    assert plugin_script == LAUNCHER and plugin["command"] == "sh"
    entry = READERS[client]()
    assert entry["command"] == "sh"
    assert expand(entry["args"][0], Path("<root>")).replace("<root>/", "") == LAUNCHER
    assert entry["args"][1:] == plugin["args"][1:] == ["mcp"]
    assert (ROOT / LAUNCHER).is_file()


def test_the_snippets_for_the_shipped_files_equal_the_files():
    """docs/clients.md shows the committed files. A change in one place needs the same change in the other."""
    assert json.loads(doc_snippet(".cursor/mcp.json")) == json.loads((ROOT / ".cursor" / "mcp.json").read_text())
    assert tomllib.loads(doc_snippet(".codex/config.toml")) == tomllib.loads((ROOT / ".codex" / "config.toml").read_text())


def test_codex_start_wait_is_shorter_than_the_codex_timeout():
    """The launcher waits TOPOPT_MCP_WAIT s for the first setup. Codex must wait longer than that."""
    entry = read_codex()
    wait = int(entry["env"]["TOPOPT_MCP_WAIT"])
    assert wait < entry["startup_timeout_sec"]


# --- a generic MCP client starts the server with the exact command, args and env of each config -------------------

FAKE_UV = """#!/bin/sh
# Stand-in for uv. `uv venv ... TARGET` makes TARGET/bin/python. That `python` runs the Python of the test with
# the repo under test first on PYTHONPATH (so an installed copy of topopt_mcp cannot take its place).
# Every other call does nothing.
case "$1" in
  venv)
    for last; do :; done
    mkdir -p "$last/bin"
    cat > "$last/bin/python" <<'PYTHON_END'
#!/bin/sh
PYTHONPATH=@ROOT@${PYTHONPATH:+:$PYTHONPATH} exec @PYTHON@ "$@"
PYTHON_END
    chmod +x "$last/bin/python" ;;
esac
exit 0
"""


@pytest.fixture(scope="module")
def env_dir(tmp_path_factory):
    """A ready environment for the real launcher (`sync` with a fake uv), outside the repo."""
    base = tmp_path_factory.mktemp("client-env")
    bindir = base / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text(FAKE_UV.replace("@ROOT@", shlex.quote(str(ROOT))).replace("@PYTHON@", shlex.quote(sys.executable)))
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    folder = base / "env"
    env = dict(os.environ, PATH=f"{bindir}:{os.environ.get('PATH', '/usr/bin:/bin')}", TOPOPT_ENV_DIR=str(folder),
               HOME=str(base / "home"))
    for key in ("CLAUDE_PLUGIN_DATA", "TOPOPT_MCP_WAIT", "TOPOPT_WORKSPACE"):
        env.pop(key, None)
    done = subprocess.run(["sh", str(ROOT / LAUNCHER), "sync"], env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return folder


def launch_of(client: str, env_dir: Path, tmp_path: Path):
    """The command, args, env and cwd that the client gives to the process. Only TOPOPT_ENV_DIR is added by the test."""
    from mcp.client.stdio import StdioServerParameters
    entry = READERS[client]()
    args = [expand(a, ROOT) for a in entry["args"]]
    env = {k: expand(v, ROOT) for k, v in entry.get("env", {}).items()}
    # Cursor starts the server in a folder that is not the workspace (a forum post of staff, 2025-06-03).
    # The other three start it in the repo root: the relative path of Codex and Gemini needs that.
    start = tmp_path if client == "cursor" else ROOT
    # A `cwd` key of the config sets the folder. A relative `cwd` is taken from the start folder here. Codex says
    # nothing about a relative MCP `cwd`, so this is the simplest rule to test.
    cwd = start / expand(entry["cwd"], ROOT) if "cwd" in entry else start
    env["TOPOPT_ENV_DIR"] = str(env_dir)           # the test adds this one; nothing else
    return StdioServerParameters(command=entry["command"], args=args, env=env, cwd=str(cwd))


def text_of(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def payload_of(result) -> dict:
    sc = getattr(result, "structured_content", None)
    if sc:
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    return json.loads(text_of(result))


async def talk(params, *, check: bool) -> dict:
    from mcp import Client
    found = {}
    async with Client(params) as client:
        found["tools"] = {t.name for t in (await client.list_tools()).tools}
        found["guide"] = {role: text_of(await client.call_tool("team_guide", {"role": role})) for role in team.ROLE_IDS}
        found["list"] = text_of(await client.call_tool("team_guide", {}))
        found["workspace"] = payload_of(await client.call_tool("list_runs", {}))["workspace"]
        if check:
            found["check"] = payload_of(await client.call_tool("check_problem", {"yaml_path": HOOK_YAML}))
    return found


@pytest.mark.parametrize("client", READERS)
def test_stdio_round_trip_with_the_exact_launch_of_the_config(client, env_dir, tmp_path):
    pytest.importorskip("mcp")
    params = launch_of(client, env_dir, tmp_path)
    assert params.command == "sh" and params.args[0].endswith(LAUNCHER) and params.args[1:] == ["mcp"]
    found = asyncio.run(asyncio.wait_for(talk(params, check=client in SHIPPED), 180))
    assert EXPECTED_TOOLS <= found["tools"], sorted(EXPECTED_TOOLS - found["tools"])
    # The work folder is the repo root for each config. Cursor starts the server elsewhere, and its env sets the folder.
    assert Path(found["workspace"]).resolve() == ROOT, found["workspace"]
    roles = team.load_roles()
    for role_id in team.ROLE_IDS:
        assert found["guide"][role_id] == roles[role_id].text, role_id
        assert role_id in found["list"]
    if client in SHIPPED:
        assert found["check"]["ok"] is True, found["check"]
        assert found["check"]["free_dofs"] > 0


# --- Agent Skills spec (https://agentskills.io/specification, read on 2026-10-20) ----------------------------------

SKILL_FOLDERS = {
    "skills/start-here": "start-here",
    ".claude/skills/part-to-topopt": "part-to-topopt",
    ".agents/skills/start-here": "start-here",
    ".agents/skills/part-to-topopt": "part-to-topopt",
}
SPEC_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}


def split_skill(folder: str) -> tuple[dict, str]:
    raw = (ROOT / folder / "SKILL.md").read_text(encoding="utf-8")
    assert raw.startswith("---\n"), f"{folder}: no frontmatter"
    head, sep, body = raw[4:].partition("\n---\n")
    assert sep, f"{folder}: the frontmatter has no closing ---"
    return yaml.safe_load(head), body


@pytest.mark.parametrize("folder", SKILL_FOLDERS)
def test_skill_meets_the_agent_skills_spec(folder):
    front, body = split_skill(folder)
    assert set(front) <= SPEC_FIELDS, set(front) - SPEC_FIELDS
    name = front["name"]
    assert name == SKILL_FOLDERS[folder] == Path(folder).name          # the name equals the folder name
    assert 1 <= len(name) <= 64 and re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name)   # no start or end hyphen, no `--`
    assert isinstance(front["description"], str) and 1 <= len(front["description"].strip()) <= 1024
    if "compatibility" in front:
        assert 1 <= len(front["compatibility"]) <= 500
    if "metadata" in front:
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in front["metadata"].items())
    if "allowed-tools" in front:
        assert isinstance(front["allowed-tools"], str)
    assert body.strip()
    assert len(body.splitlines()) < 500                                  # the spec advises less than 500 lines


# --- one source and byte-for-byte copies -------------------------------------------------------------------------

PAIRS = [tuple(pair) for pair in sync_agent_skills.PAIRS]          # (source, copy)


@pytest.mark.parametrize("source,copy", PAIRS, ids=[copy for _, copy in PAIRS])
def test_agents_skill_folder_is_a_byte_for_byte_copy_of_its_source(source, copy):
    """Files that git tracks (or would track) count. A `.DS_Store` or a `__pycache__` that .gitignore names does not."""
    found = sync_agent_skills.differences(ROOT, (source, copy))
    assert not found, ("\n".join(found) + f"\nFix: python3 scripts/sync_agent_skills.py "
                       f"(it copies each source over its copy; the source, {source}, is the one to edit)")


def test_every_skill_folder_in_agents_has_a_source_and_the_readme_names_it():
    copies = {copy for _, copy in PAIRS}
    on_disk = {f".agents/skills/{p.name}" for p in (ROOT / ".agents" / "skills").iterdir() if p.is_dir()
               and p.name != "__pycache__"}
    assert on_disk == copies, "a folder in .agents/skills/ needs a pair in scripts/sync_agent_skills.py (PAIRS)"
    readme = (ROOT / ".agents" / "README.md").read_text(encoding="utf-8")
    assert "scripts/sync_agent_skills.py" in readme
    for source, copy in PAIRS:
        assert f"`{source}/`" in readme and f"`{copy}/`" in readme, (source, copy)


def _make_pair(root: Path, pair=PAIRS[1]) -> None:
    source, copy = pair
    for folder in (source, copy):
        (root / folder / "examples").mkdir(parents=True)
        (root / folder / "SKILL.md").write_text("---\nname: x\n---\ntext\n")
        (root / folder / "build.py").write_text("print(1)\n")
        (root / folder / "examples" / "a.yaml").write_text("a: 1\n")


def _run_helper(root: Path, *flags: str):
    script = ROOT / "scripts" / "sync_agent_skills.py"
    return subprocess.run([sys.executable, "-I", str(script), "--root", str(root), *flags], capture_output=True, text=True)


def test_sync_helper_check_finds_each_kind_of_difference_and_copy_mode_repairs_it(tmp_path):
    for pair in PAIRS:
        _make_pair(tmp_path, pair)
    assert _run_helper(tmp_path, "--check").returncode == 0
    source, copy = PAIRS[1]
    (tmp_path / copy / "SKILL.md").write_text("changed\n")                   # content
    (tmp_path / copy / "extra.md").write_text("x\n")                          # a file that the source has not
    (tmp_path / copy / "examples" / "a.yaml").unlink()                         # a file that the copy has not
    (tmp_path / source / "link.md").symlink_to("SKILL.md")                     # a symlink
    (tmp_path / copy / "link.md").write_text("x\n")
    done = _run_helper(tmp_path, "--check")
    assert done.returncode == 1
    for word in ("SKILL.md", "extra.md", "examples/a.yaml", "link.md"):
        assert word in done.stdout, done.stdout
    (tmp_path / source / "link.md").unlink()
    done = _run_helper(tmp_path)                                               # copy mode
    assert done.returncode == 0, done.stderr
    assert _run_helper(tmp_path, "--check").returncode == 0
    assert (tmp_path / copy / "SKILL.md").read_bytes() == (tmp_path / source / "SKILL.md").read_bytes()
    assert not (tmp_path / copy / "extra.md").exists() and (tmp_path / copy / "examples" / "a.yaml").is_file()


def test_sync_helper_ignores_junk_files_outside_git(tmp_path):
    _make_pair(tmp_path)
    copy = tmp_path / PAIRS[1][1]
    (copy / ".DS_Store").write_bytes(b"\0")
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "build.cpython-312.pyc").write_bytes(b"\0")
    assert sync_agent_skills.differences(tmp_path, PAIRS[1]) == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_sync_helper_in_git_counts_only_files_that_git_does_not_ignore(tmp_path):
    _make_pair(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text(".DS_Store\nlocal-notes.txt\n")
    copy = tmp_path / PAIRS[1][1]
    (copy / ".DS_Store").write_bytes(b"\0")
    (copy / "local-notes.txt").write_text("mine\n")
    assert sync_agent_skills.differences(tmp_path, PAIRS[1]) == []
    (copy / "other-notes.txt").write_text("mine\n")                           # not ignored: the copy has a file extra
    assert any("other-notes.txt" in line for line in sync_agent_skills.differences(tmp_path, PAIRS[1]))


def test_the_skill_folders_hold_no_symlink():
    """Some tools refuse symlinks, and only Codex documents them."""
    for folder in (".agents", ".cursor", ".codex", "skills", ".claude/skills"):
        for dirpath, dirnames, filenames in os.walk(ROOT / folder):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in dirnames + filenames:
                assert not (Path(dirpath) / name).is_symlink(), Path(dirpath) / name


# --- skill text: paths that resolve in every client, role texts through `team_guide` -----------------------------

# A file path of the repo, written in a skill text. A path starts at the repo root, or (the Agent Skills rule) at the
# folder of the skill. A path after `${...}/` is checked by tests/test_plugin_manifest.py.
PATH_IN_TEXT = re.compile(
    r"(?<![\w$}/.<>-])((?:\.claude|\.agents|skills|scripts|examples|tools|docs|topopt_mcp|topopt)/[\w./-]*\.(?:md|py|sh|yaml|yml|json|toml|stl))")


@pytest.mark.parametrize("folder", SKILL_FOLDERS)
def test_every_path_in_a_skill_text_resolves_from_the_repo_root_or_the_skill_folder(folder):
    texts = [ROOT / folder / "SKILL.md"] + sorted((ROOT / folder / "references").glob("*.md"))   # the files read on demand too (#207)
    for path in texts:
        text = path.read_text(encoding="utf-8")
        assert "CLAUDE_SKILL_DIR" not in text      # only Claude Code fills this in; the copies are read by other clients
        for rel in sorted(set(PATH_IN_TEXT.findall(text))):
            assert (ROOT / rel).is_file() or (ROOT / folder / rel).is_file(), \
                f"{path.relative_to(ROOT)} names {rel}, and the file is in neither the repo root nor the folder of the skill"


PART_FILES = ("SKILL.md", "build_regions.py", "examples/hook_spec.yaml", "examples/ladle_blind_spec.yaml",
              "references/patterns.md", "references/blind-test.md")


@pytest.mark.parametrize("folder", [f for f in SKILL_FOLDERS if f.endswith("part-to-topopt")])
def test_part_to_topopt_names_its_own_files_from_the_skill_folder(folder):
    """Both copies hold the files. The text names them from the folder of the skill, so every client finds them."""
    for rel in PART_FILES:
        assert (ROOT / folder / rel).is_file(), f"{folder}/{rel}"
    text = (ROOT / folder / "SKILL.md").read_text(encoding="utf-8")
    assert "examples/hook_spec.yaml` in the folder of this skill" in text
    assert "skills/part-to-topopt/examples/" not in text      # a repo-root path to a file of the skill: wrong in a copy


def test_start_here_sends_a_client_without_sub_agents_to_team_guide():
    text = (ROOT / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8")
    assert "team_guide" in text
    assert not re.search(r"\b(must|never)\b", text, re.I)   # the freedom rule: facts and options
    for role_id in team.ROLE_IDS:
        assert f"`{role_id}`" in text


def test_start_here_names_where_part_to_topopt_is():
    text = (ROOT / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8")
    assert ".claude/skills/part-to-topopt/SKILL.md" in text and ".agents/skills/part-to-topopt/SKILL.md" in text


def test_start_here_copies_are_equal_and_name_no_must_or_never():
    """The two copies of the skill are one text. The freedom rule: facts and options, no must or never."""
    texts = [(ROOT / folder / "SKILL.md").read_text(encoding="utf-8") for folder in ("skills/start-here", ".agents/skills/start-here")]
    assert texts[0] == texts[1]
    assert not re.search(r"\b(must|never)\b", texts[0], re.I)


def test_the_cursor_user_entry_starts_the_same_launcher_with_full_paths_under_a_name_of_its_own():
    """Mac test, Cursor 3.27.22 (2026-10-27): the project server stayed Disabled, a user entry in ~/.cursor/mcp.json worked.
    The entry of docs/clients.md starts the launcher of the committed file, with full paths (a user entry serves every folder,
    so `${workspaceFolder}` would point at the wrong one), and under another name (the project file wins for the same name)."""
    data = json.loads(doc_snippet("~/.cursor/mcp.json"))
    assert set(data) == {"mcpServers"} and list(data["mcpServers"]) == ["topopt-user"]
    entry, project = data["mcpServers"]["topopt-user"], read_cursor()
    assert set(entry) <= CURSOR_KEYS
    assert (entry["type"], entry["command"]) == (project["type"], project["command"])
    assert entry["args"][1:] == project["args"][1:] and entry["args"][0].startswith("/")
    assert entry["args"][0].endswith("/scripts/topopt-env.sh") and project["args"][0].endswith("/scripts/topopt-env.sh")
    assert set(entry["env"]) == set(project["env"]) == {"TOPOPT_WORKSPACE"} and entry["env"]["TOPOPT_WORKSPACE"].startswith("/")
    assert "${" not in json.dumps(entry)
    text = " ".join(CLIENTS_DOC.read_text().split())
    assert "turn the project server `topopt` off" in text
    checklist = (ROOT / "docs" / "install-test-checklist.md").read_text(encoding="utf-8")
    cursor_steps = checklist.split("## 4. Cursor", 1)[1].split("\n## 5.", 1)[0]
    assert "Disabled" in cursor_steps and set(re.findall(r"`(topopt-[a-z]+)`", cursor_steps)) == set(data["mcpServers"])


# --- #418: the guide for any MCP client (docs/reference/any-mcp-client.md, CONTRACTS.md section 34.2a) ---------------

GUIDE = ROOT / "docs" / "reference" / "any-mcp-client.md"
PLACEHOLDER = "/full/path/to/topopt"
TEMPLATE_NAME = "topopt-local"


def guide_template(name: str) -> str:
    """The text of the code block that follows `<!-- template: NAME -->` in the guide."""
    m = re.search(r"<!-- template: " + re.escape(name) + r" -->\s*```\w*\n(.*?)\n```", GUIDE.read_text(encoding="utf-8"), re.S)
    assert m, f"the guide has no template {name}"
    return m.group(1)


def read_template_json() -> dict:
    data = json.loads(guide_template("any-client.json"))
    assert set(data) == {"mcpServers"} and list(data["mcpServers"]) == [TEMPLATE_NAME]
    return data["mcpServers"][TEMPLATE_NAME]


def read_template_toml() -> dict:
    data = tomllib.loads(guide_template("any-client.toml"))
    assert set(data) == {"mcp_servers"} and list(data["mcp_servers"]) == [TEMPLATE_NAME]
    return data["mcp_servers"][TEMPLATE_NAME]


TEMPLATES = {"json": read_template_json, "toml": read_template_toml}


def guide_text() -> str:
    return " ".join(GUIDE.read_text(encoding="utf-8").split())


@pytest.mark.parametrize("kind", TEMPLATES)
def test_the_templates_have_absolute_paths_and_a_name_that_no_project_file_uses(kind):
    entry = TEMPLATES[kind]()
    assert entry["command"] == "sh"
    assert entry["args"] == [f"{PLACEHOLDER}/{LAUNCHER}", "mcp"]
    assert entry["env"]["TOPOPT_WORKSPACE"] == PLACEHOLDER
    assert "${" not in json.dumps(entry)                      # the clients differ in their variables: full paths only
    # The name is not the name of the repo's project files: the project file of Cursor wins for the same name.
    # No project file of the repo and no snippet of docs/clients.md uses the name (the project file of Cursor wins for the same name).
    for rel in (".cursor/mcp.json", ".codex/config.toml", ".mcp.json", "docs/clients.md"):
        assert TEMPLATE_NAME not in (ROOT / rel).read_text(encoding="utf-8"), rel


def test_the_json_template_has_only_keys_that_cursor_and_vscode_document():
    entry = read_template_json()
    assert set(entry) <= CURSOR_KEYS & VSCODE_KEYS and entry["type"] == "stdio"
    assert all(isinstance(a, str) for a in entry["args"]) and all(isinstance(v, str) for v in entry["env"].values())


def test_the_toml_template_has_only_keys_that_codex_documents_and_both_timeouts():
    entry = read_template_toml()
    assert set(entry) <= CODEX_KEYS
    for key in ("startup_timeout_sec", "tool_timeout_sec"):
        assert isinstance(entry[key], (int, float)) and not isinstance(entry[key], bool) and entry[key] > 0, key
    # export_step holds a call until the file is written: the longest measured call was 613 s (docs/clients.md, Limits).
    assert entry["tool_timeout_sec"] >= 613
    # The template follows the committed Codex file: the same start wait (the status server has no live Codex run yet), shorter than the timeout.
    committed = read_codex()
    assert entry["env"]["TOPOPT_MCP_WAIT"] == committed["env"]["TOPOPT_MCP_WAIT"]
    assert int(entry["env"]["TOPOPT_MCP_WAIT"]) < entry["startup_timeout_sec"]
    assert entry["startup_timeout_sec"] == committed["startup_timeout_sec"]


@pytest.mark.parametrize("kind", TEMPLATES)
def test_the_templates_of_the_guide_start_the_server_as_written(kind, env_dir, tmp_path):
    """The template with the placeholder replaced by the repo root and nothing else changed (the test adds TOPOPT_ENV_DIR)."""
    pytest.importorskip("mcp")
    from mcp.client.stdio import StdioServerParameters
    entry = TEMPLATES[kind]()
    args = [a.replace(PLACEHOLDER, str(ROOT)) for a in entry["args"]]
    env = {k: v.replace(PLACEHOLDER, str(ROOT)) for k, v in entry["env"].items()}
    env["TOPOPT_ENV_DIR"] = str(env_dir)
    # Not the repo root: the full path must not need the start folder.
    params = StdioServerParameters(command=entry["command"], args=args, env=env, cwd=str(tmp_path))
    found = asyncio.run(asyncio.wait_for(talk(params, check=False), 180))
    assert EXPECTED_TOOLS <= found["tools"], sorted(EXPECTED_TOOLS - found["tools"])
    assert Path(found["workspace"]).resolve() == ROOT, found["workspace"]


def test_the_guide_has_its_seven_parts_in_order():
    heads = re.findall(r"^## (\d)\. (.+)$", GUIDE.read_text(encoding="utf-8"), re.M)
    assert [n for n, _ in heads] == list("1234567")
    assert [h.split()[0:2] for _, h in heads][0] == ["What", "to"]


def test_the_numbers_of_the_guide_that_live_in_code_follow_the_code():
    text = guide_text()
    launcher = (ROOT / LAUNCHER).read_text(encoding="utf-8")
    min_uv = re.search(r"^MIN_UV_VERSION=(\S+)", launcher, re.M).group(1)
    assert f"uv {min_uv} or newer" in text
    # The guide names the variables of the two uv versions; the number of the pinned one is in the install doc (tests/test_install_downloads_doc.py).
    assert "`MIN_UV_VERSION`" in text and "`PINNED_UV_VERSION`" in text
    pins = sum(1 for line in (ROOT / "requirements.lock").read_text().splitlines() if re.match(r"^[A-Za-z0-9_.-]+==\S+", line))
    assert f"{pins} packages" in text
    mcp_pin = re.search(r"^mcp==\S+", (ROOT / "requirements.lock").read_text(), re.M).group(0)
    assert f"`{mcp_pin}`" in text                                        # the lock pins mcp: the manual setup needs no extra
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    assert f'`requires-python = "{requires}"`' in text
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["topopt-mcp"] == "topopt_mcp.server:main" and "`topopt-mcp`" in text
    # Step 3 of the launcher is the third command of the manual setup.
    assert "--no-deps --no-build-isolation -e" in launcher and "--no-deps --no-build-isolation -e ." in text
    assert f"{len(asyncio.run(__import__('topopt_mcp.server', fromlist=['mcp']).mcp.list_tools()))} tools" in text


def test_stdio_is_the_only_transport_of_the_server():
    """Part 4 says: stdio is the only transport. `main()` has one `mcp.run(...)` call, with "stdio"."""
    code = (ROOT / "topopt_mcp" / "server.py").read_text(encoding="utf-8")
    runs = re.findall(r"\bmcp\.run\(([^)]*)\)", code)
    assert runs == ['"stdio"'], runs
    text = guide_text()
    assert "Stdio is the only transport" in text and 'mcp.run("stdio")' in text


def test_the_stderr_line_of_part_4_is_the_line_that_the_server_writes(env_dir, tmp_path):
    """Start the real server (the launcher with the test environment), close stdin, and compare its stderr line with the guide."""
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, TOPOPT_ENV_DIR=str(env_dir), TOPOPT_WORKSPACE=str(work), HOME=str(tmp_path / "home"))
    done = subprocess.run(["sh", str(ROOT / LAUNCHER), "mcp"], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert done.stdout == "", done.stdout                                  # stdout carries the protocol only
    line = done.stderr.strip().splitlines()[0]
    assert re.fullmatch(r"\[topopt_mcp\] python: \S+; work folder: \S+ \(from TOPOPT_WORKSPACE\)", line), line
    assert "[topopt_mcp] python: <path of the Python>; work folder: <folder> (from TOPOPT_WORKSPACE)" in guide_text()


def test_the_launcher_commands_that_the_guide_names_exist():
    launcher = (ROOT / LAUNCHER).read_text(encoding="utf-8")
    for command in ("sync", "status", "doctor", "mcp", "python", "topopt"):
        assert re.search(rf"^  {command}\)", launcher, re.M), command
        assert f"`{command}" in guide_text() or f"topopt-env.sh {command}" in guide_text(), command
    for var in ("TOPOPT_ENV_DIR", "TOPOPT_MCP_WAIT", "TOPOPT_WORKSPACE", "UV_OFFLINE", "TOPOPT_UV"):
        assert var in guide_text(), var
    assert 'ENV_DIR=${TOPOPT_ENV_DIR:-${EVAL_TOPOPT_ENV_DIR:-${CLAUDE_PLUGIN_DATA:-$DEFAULT_ENV_DIR}}}' in launcher
    assert re.search(r"`TOPOPT_ENV_DIR`, two internal variables \(`EVAL_TOPOPT_ENV_DIR` is for the eval harness, `CLAUDE_PLUGIN_DATA` for the Claude Code plugin\), the default", guide_text())


def test_the_clients_page_stays_short_and_keeps_its_snippets():
    text = CLIENTS_DOC.read_text(encoding="utf-8")
    # A guard, not a rule: the page was cut from 448 to about 230 lines (#418). If a new fact hits the cap, put its long form in
    # the reference pages and keep a short tip here; raise the number only if the new text is a tip that every client needs.
    assert len(text.splitlines()) < 260
    for name in (".cursor/mcp.json", "~/.cursor/mcp.json", ".codex/config.toml", ".vscode/mcp.json", ".gemini/settings.json"):
        assert f"<!-- config: {name} -->" in text, name


def test_the_guide_is_linked_from_the_readme_the_clients_page_and_the_agent_rules():
    for rel in ("README.md", "docs/clients.md", "AGENTS.md"):
        assert "any-mcp-client.md" in (ROOT / rel).read_text(encoding="utf-8"), rel
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("<summary><b>Generic MCP client</b></summary>", 1)[1].split("</details>", 1)[0]
    assert "docs/reference/any-mcp-client.md" in section           # "Generic MCP client" links the guide
    assert "clients-history" not in CLIENTS_DOC.read_text(encoding="utf-8")     # the page does not link to a file that the public export leaves out


def test_the_guide_does_not_say_that_only_cursor_had_a_person_test_among_all_six_clients():
    """Claude Code and Claude Desktop passed a first-use test on a Mac on 2026-10-27 (docs/reference/install.md)."""
    text = guide_text()
    install = " ".join((ROOT / "docs/reference/install.md").read_text(encoding="utf-8").split())
    assert "Both passed a first-use test on a Mac on 2026-10-27" in install
    assert "Of the four other clients, only Cursor had a person test" in text
    assert "Claude Code and Claude Desktop passed a first-use test on a Mac on 2026-10-27" in text
    assert "Of these clients, only Cursor" not in text


def test_the_guide_tips_name_all_six_clients_and_link_the_details():
    part7 = GUIDE.read_text(encoding="utf-8").split("## 7. Tips per client", 1)[1]
    for client in ("Claude Code", "Claude Desktop", "Cursor", "Codex CLI", "VS Code", "Gemini CLI"):
        bullet = re.search(rf"^- \*\*{re.escape(client)}[^*]*\*\*(.*)$", part7, re.M)
        assert bullet, client
        assert len(bullet.group(1)) < 700, client                      # at most a few lines each
