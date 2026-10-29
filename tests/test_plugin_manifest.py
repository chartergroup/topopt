"""Tests for the Claude Code plugin files: manifests, skills, agents, the server command, the MCP text limits."""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from helpers import ROLE_TEXT_MAX_CHARS  # noqa: E402

PLUGIN_JSON = ROOT / ".claude-plugin" / "plugin.json"
MARKET_JSON = ROOT / ".claude-plugin" / "marketplace.json"

# Claude Code limits (https://code.claude.com/docs/en/mcp, /skills): the text of the server
# instructions and of each tool description is cut at this many characters.
MCP_TEXT_LIMIT = 4096
SKILL_DESCRIPTION_LIMIT = 1536
# Fields that a plugin agent honors. permissionMode, hooks, mcpServers and initialPrompt are ignored.
AGENT_FIELDS = {"name", "description", "model", "effort", "maxTurns", "autoCompactWindow", "tools",
                "disallowedTools", "skills", "memory", "background", "omitClaudeMd", "isolation",
                "color", "experimental"}
ROLES = {"lead-design-engineer", "cad-engineer", "fea-engineer", "print-engineer", "finish-engineer"}


def load_json(path):
    return json.loads(path.read_text())


def split_frontmatter(path):
    text = path.read_text()
    assert text.startswith("---\n"), f"{path}: no frontmatter"
    head, _, body = text[4:].partition("\n---\n")
    return yaml.safe_load(head), body


def plugin_skill_files():
    """SKILL.md files: the default skills/ folder and the folders that `skills` in plugin.json names."""
    plugin = load_json(PLUGIN_JSON)
    found = sorted((ROOT / "skills").glob("*/SKILL.md"))
    for entry in plugin.get("skills", []):
        folder = ROOT / entry
        found += [folder / "SKILL.md"] if (folder / "SKILL.md").exists() else sorted(folder.glob("*/SKILL.md"))
    return found


# --- MCP text limits ---------------------------------------------------------------------------

def test_mcp_instructions_fit_the_cut_of_the_older_claude_code():
    """#380: the whole instructions text is read by Claude Code 2.1.293 and older, which cuts it at 2048 characters (they log "Server
    instructions truncated"). The limit is OLD_CLIENT_CUT (below); a rule that does not fit goes to the description of its tool, to
    the start-here skill, or to a tool result (see MOVED_RULES)."""
    from topopt_mcp.server import INSTRUCTIONS, mcp
    assert len(INSTRUCTIONS) <= OLD_CLIENT_CUT <= MCP_TEXT_LIMIT, len(INSTRUCTIONS)
    assert mcp.instructions == INSTRUCTIONS


def test_mcp_tool_descriptions_are_4096_characters_or_fewer():
    from topopt_mcp.server import mcp
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) >= 7
    for tool in tools:
        assert len(tool.description or "") <= MCP_TEXT_LIMIT, tool.name


# --- the key text is in the first 2048 characters ----------------------------------------------

# Claude Code before v2.1.296 cuts the server instructions and each tool description at this many characters
# (docs: code.claude.com/docs/en/mcp, "Tool search for MCP server authors"; README, section "MCP server").
OLD_CLIENT_CUT = 2048

# For each text over OLD_CLIENT_CUT characters: 2 to 5 short, exact phrases that carry its key text (what it does, what it writes, the
# arguments that change the cost or the effect, the cost, the reply and error contract, the advice to call another tool first).
# The name INSTRUCTIONS is the server instructions; the other names are tools.
KEY_TEXT = {
    "run_topopt": ["Returns at once", "`earlier_checkpoint`", '"awaiting_user"', "`resume.facts`",
        "continue_from: add passes"],
    "run_status": ["run_dir: the run folder", "`stop_request`", "`verdict` (null until the run ends)", "`verdict.facts`",
        'ends with "Facts: ..." for each fact that the run has'],
    "run_result": ["`done: false` and `stopped: true`", "Reply keys: resource topopt://replies/run_result", "not the stress of the part"],
    "stop_run": ["Returns at once; sends no signal", "ends with exit code 5", "resume=true (no allow_changed_inputs)",
        "several live runs: the list is in `runs`"],
    "render_part": ["No display or GL is needed", "Look at it before a run", "when inline is true",
        "Later renders overwrite PNGs with the same names", "winding_consistent"],
    "marks_to_hints": ["Edits no mesh", "The agent decides", "The tool writes only into the new folder",
        "then run_topopt with the proposed YAML and a new out_dir", "about 1.5 KB for each mark"],
    "apply_marks": ["No FE run is made", "is rolled back", "frozen_keep_mm", "marks_match_final_stl false",
        "max_crop_voxels"],
    "check_problem": ["Load and voxelize a problem YAML (no optimizer run; with settings.solver auto it runs the short self-test of gpu-mg", "Use it before run_topopt",
        'a warning that starts with "settings.solver is"', "means run_topopt will stop with the same message",
        "lost_force"],
    "verify_run": ["estimate_only: default True", "call again with estimate_only=false", "a second start is refused",
        "allowable_mpa, percentile", "exit_code 2:"],
    "ratchet_run": ["estimate_only=false", "Only one job goes at a time", "Reply to a start:", "On a wrong input:",
        "rebaseline=true"],
    "build_regions": ["Use it when the user describes a part in words", "Call check_problem only after you edit the",
        "A guard stop names an override", "Writes <work folder>/runs/<name>/", "Use yaml_rel with run_topopt"],
    "environment": ["Changes nothing", "Reply keys: resource topopt://replies/environment", "This tool installs and builds nothing", "you run the command",
        "`gpu_solver` is a fact, not an offer"],
    "print_check": ["Geometry only: no solve, no run", "strict: opt in to the gate", "`exit_code` is then 4",
        "No run stops on it", "A print pass is not a structural pass"],
    "export_step": ["Post-only", "The call is synchronous", "estimate_only first for a large mesh", "613 s (10 minutes)",
        '"missing_extra": "step"'],
    "find_material": ["It reads; it changes nothing", "The user only names the filament", '"ambiguous": `candidates`', '"none": `suggestions`'],
    "save_material": ["It writes only that file", "after the user agrees", "A value that you cannot find stays null", "Needed: id"],
    "cleanup_list": ["no edit, no verdict", "The list judges nothing", "Time: about 2 s", "Codex waits 60 s",
        "A wrong input gives"],
}


# #380: a rule that the instructions had before the cut to 2048 characters, and the place where it is now: (text name, exact phrase).
# The name is a tool (its description; for a description over OLD_CLIENT_CUT the phrase must end inside the first 2048 characters),
# or `replies/<tool>` (the reply key guide of that tool, topopt://replies/<tool>, #174), or a file of the repo (the start-here skill; the copy in .agents/ is the same file, tests/test_client_configs.py).
MOVED_RULES = {
    "the faster backend: the reason and the speed-up first, install only after the user agrees": [
        ("environment", "Say the reason (and the speed-up) first"), ("environment", "If the user agrees, you run the command")],
    "another solver: settings.solver, --solver of topopt run and check, settings.gpu_device": [
        ("check_problem", "Another solver: settings.solver (auto, cholesky, pardiso, amg, gpu-mg)"),
        ("check_problem", "--solver of topopt run and check, not of run_topopt"),
        ("check_problem", "settings.gpu_device (auto, a number from topopt check, or a part of the name)")],
    "a solver backend that cannot run here also stops run_topopt": [("check_problem", "means run_topopt will stop with the same message")],
    "build_regions returns the check and the warnings": [("build_regions", "It already returns `check` and `warnings`")],
    "a look before the first run and after each run": [
        ("render_part", "A look before the first run and after each run finds a wrong region or load fast."),
        ("skills/start-here/SKILL.md", "the regions before the first run (Step 3), the part after each run (Step 5)")],
    "print_check: the gate is opt-in": [("print_check", "strict: opt in to the gate")],
    "refine from feedback: the tools edit no mesh and the agent chooses": [
        ("marks_to_hints", "Edits no mesh"), ("marks_to_hints", "The agent decides"), ("compare_runs", "the files on disk can"),
        ("skills/start-here/SKILL.md", "## Step 6. Refine from feedback")],
    "verify_run: estimate first, then verify_status": [("verify_run", "estimate_only: default True"), ("verify_run", "verify_status shows it")],
    "ratchet_run: estimate first, then ratchet_status; ratchet_approve is the gate": [
        ("ratchet_run", "estimate_only: default True"), ("ratchet_run", "call ratchet_status to watch it"),
        ("ratchet_approve", "The opt-in owner gate"), ("skills/start-here/SKILL.md", "`ratchet_approve` is the opt-in owner gate")],
    "cleanup_list writes a marks file for ratchet_run and edits nothing": [
        ("cleanup_list", "ratchet_run(marks=marks_rel)"), ("cleanup_list", "no edit, no verdict")],
    "the levers resource": [("replies/run_result", "topopt://levers"), ("check_problem", "topopt://levers")],
    "the numbers to state before the run: assumptions and derived": [("skills/start-here/SKILL.md", "The tools return `assumptions` and `derived`")],
    "say early whether and how topology optimization helps": [("skills/start-here/SKILL.md", "Say early whether and how topology optimization helps")],
    "the pass limit in plain words": [("skills/start-here/SKILL.md", 'say it in plain words: "the run stopped at its pass limit", not "not converged"')],
    "team_guide and case_status": [("team_guide", "Playbooks for the roles"), ("case_status", "Show the case of a problem YAML")],
}


def test_each_rule_that_left_the_instructions_is_in_its_new_place():
    texts = mcp_texts()
    missing, late = [], []
    for rule, places in MOVED_RULES.items():
        for place, phrase in places:
            if place.endswith(".md"):
                text = (ROOT / place).read_text(encoding="utf-8")
                text = " ".join(text.split())
            elif place.startswith("replies/"):
                from topopt_mcp import replies
                text = replies.guide(place.split("/", 1)[1])
            else:
                assert place in texts, f"{rule}: no tool {place}"
                text = texts[place]
            at = text.find(phrase)
            if at < 0:
                missing.append((rule, place, phrase))
            elif not place.endswith(".md") and not place.startswith("replies/") and at + len(phrase) > OLD_CLIENT_CUT:
                late.append((rule, place, phrase, at + len(phrase)))
    assert not missing, missing
    assert not late, f"a phrase ends after character {OLD_CLIENT_CUT}: {late}"


def test_the_instructions_name_the_other_tools_and_say_that_their_descriptions_tell_when_to_use_them():
    from topopt_mcp.server import INSTRUCTIONS
    last = INSTRUCTIONS.splitlines()[-1]
    assert last.startswith("Descriptions say when to use:")
    for tool in ("case_status", "run_history", "compare_runs", "marks_to_hints", "apply_marks", "verify_run",
                 "ratchet_run", "cleanup_list", "team_guide", "list_runs"):
        assert tool in last, tool


def mcp_texts():
    """{name: text}: the server instructions (name INSTRUCTIONS) and the description of each tool."""
    from topopt_mcp.server import INSTRUCTIONS, mcp
    texts = {tool.name: tool.description or "" for tool in asyncio.run(mcp.list_tools())}
    texts["INSTRUCTIONS"] = INSTRUCTIONS
    return texts


@pytest.mark.parametrize("name", sorted(KEY_TEXT))
def test_the_key_text_of_each_long_mcp_text_is_in_its_first_2048_characters(name):
    texts = mcp_texts()
    assert name in texts, f"KEY_TEXT has an entry for {name}, and the server has no such text"
    assert 2 <= len(KEY_TEXT[name]) <= 5, f"{name}: use 2 to 5 phrases"
    text = texts[name]
    late = {}
    for phrase in KEY_TEXT[name]:
        at = text.find(phrase)
        assert at >= 0, f"{name}: the phrase {phrase!r} is not in the text (change the phrase or the text)"
        if at + len(phrase) > OLD_CLIENT_CUT:
            late[phrase] = at + len(phrase)
    assert not late, f"{name}: a phrase ends after character {OLD_CLIENT_CUT} (phrase: end position): {late}"


def test_the_address_of_each_reply_key_guide_ends_inside_the_first_2048_characters():
    """#174: the key meanings are in the guide, so a client that cuts the description at 2048 characters must still see where the guide is."""
    from topopt_mcp import replies
    texts = mcp_texts()
    named = [tool for tool in sorted(replies.REPLY_KEYS) if replies.uri(tool) in texts.get(tool, "")]
    assert len(named) >= 8, named
    late = {}
    for tool in named:
        end = texts[tool].index(replies.uri(tool)) + len(replies.uri(tool))
        if end > OLD_CLIENT_CUT:
            late[tool] = end
    assert not late, f"the guide address ends after character {OLD_CLIENT_CUT} (tool: end position): {late}"


def test_every_mcp_text_over_2048_characters_has_a_key_text_entry():
    long_texts = {name: len(text) for name, text in mcp_texts().items() if len(text) > OLD_CLIENT_CUT}
    assert len(long_texts) >= 10, long_texts
    missing = sorted(set(long_texts) - set(KEY_TEXT))
    assert not missing, f"add an entry to KEY_TEXT for {missing}: put the key text in the first {OLD_CLIENT_CUT} characters"


# --- manifests ---------------------------------------------------------------------------------

def test_plugin_manifest_parses_and_has_the_required_fields():
    plugin = load_json(PLUGIN_JSON)
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", plugin["name"])
    assert plugin["description"] and plugin["author"]["name"]
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f'version = "{plugin["version"]}"' in pyproject  # one version for the package and the plugin


def test_plugin_has_no_default_agent_and_no_hooks():
    """`agent` in plugin settings replaces the Claude Code system prompt. It is an option, not a default."""
    plugin = load_json(PLUGIN_JSON)
    assert "settings" not in plugin and "hooks" not in plugin
    assert not (ROOT / "settings.json").exists()
    assert not (ROOT / "hooks").exists()


def test_marketplace_manifest_points_at_the_plugin():
    market = load_json(MARKET_JSON)
    plugin = load_json(PLUGIN_JSON)
    assert market["name"] and market["owner"]["name"]
    assert [p["name"] for p in market["plugins"]] == [plugin["name"]]
    source = market["plugins"][0]["source"]
    assert source in ("./", ".")
    assert (ROOT / source / ".claude-plugin" / "plugin.json").is_file()


def test_the_marketplace_is_named_topopt_and_the_e2e_script_uses_that_name():
    """REL-8 (#313): the marketplace has the name of the repo. Claude Code names the data folder of a plugin
    `<plugin>-<marketplace>`, so the e2e script installs `topopt@topopt` and reads `plugins/data/topopt-topopt`."""
    market = load_json(MARKET_JSON)
    plugin = load_json(PLUGIN_JSON)["name"]
    assert market["name"] == "topopt"
    assert "topopt-demo" not in json.dumps(market)
    script = (ROOT / "scripts" / "plugin-e2e.sh").read_text()
    assert f"claude plugin install {plugin}@{market['name']} " in script
    assert f"\nDATA=$CFG/plugins/data/{plugin}-{market['name']}\n" in script


def test_server_command_points_at_the_installed_copy_and_not_at_dot_mcp_json():
    plugin = load_json(PLUGIN_JSON)
    server = plugin["mcpServers"]["topopt"]  # the name of the dev .mcp.json server: this one replaces it
    assert server["command"] == "sh"
    script = server["args"][0]
    assert script.startswith("${CLAUDE_PLUGIN_ROOT}/")  # the plugin copy, not a checkout path
    assert (ROOT / script.replace("${CLAUDE_PLUGIN_ROOT}/", "")).is_file()
    assert server["args"][1:] == ["mcp"]
    assert ".mcp.json" not in PLUGIN_JSON.read_text()
    assert ".venv" not in PLUGIN_JSON.read_text()


def test_plugin_server_has_the_name_of_the_dev_server():
    """The repo root is the plugin root, so Claude Code also reads .mcp.json in the plugin copy.
    The server in plugin.json has the same name and replaces that entry (it uses the .venv path)."""
    dev = load_json(ROOT / ".mcp.json")["mcpServers"]
    assert set(dev) == set(load_json(PLUGIN_JSON)["mcpServers"])


def test_launcher_inputs_exist():
    """The launcher reads these two files from the plugin root."""
    for name in ("requirements.lock", "pyproject.toml"):
        assert (ROOT / name).is_file()


# --- skills ------------------------------------------------------------------------------------

def test_part_to_topopt_has_one_source_and_ships_in_the_plugin():
    plugin = load_json(PLUGIN_JSON)
    assert plugin["skills"] == ["./.claude/skills/part-to-topopt"]
    assert not (ROOT / "skills" / "part-to-topopt").exists()  # no copy
    assert (ROOT / ".claude/skills/part-to-topopt/SKILL.md").is_file()


def test_skills_have_names_and_short_descriptions():
    files = plugin_skill_files()
    names = {f.parent.name for f in files}
    assert {"start-here", "part-to-topopt"} <= names
    for f in files:
        front, body = split_frontmatter(f)
        assert front["name"] == f.parent.name
        assert 0 < len(front["description"]) + len(front.get("when_to_use", "")) <= SKILL_DESCRIPTION_LIMIT
        assert body.strip()


def test_skill_scripts_exist():
    """Each file that a skill names through ${CLAUDE_SKILL_DIR} or ${CLAUDE_PLUGIN_ROOT} is in the repo."""
    for f in plugin_skill_files():
        text = f.read_text()
        for var, base in (("CLAUDE_SKILL_DIR", f.parent), ("CLAUDE_PLUGIN_ROOT", ROOT)):
            for rel in re.findall(r"\$\{" + var + r"\}/([\w./-]+\.(?:py|sh|yaml))", text):
                assert (base / rel).is_file(), f"{f}: {rel}"


# --- agents ------------------------------------------------------------------------------------

def test_role_agents_use_only_fields_that_plugin_agents_honor():
    files = sorted((ROOT / "agents").glob("*.md"))
    assert {f.stem for f in files} == ROLES
    for f in files:
        front, body = split_frontmatter(f)
        assert front["name"] == f.stem and front["description"]
        assert set(front) <= AGENT_FIELDS, (f.name, set(front) - AGENT_FIELDS)
        assert len(body) < ROLE_TEXT_MAX_CHARS, f"{f.name} is not short"
        assert "does not exist yet" in body  # each role says what it cannot do yet


def test_roles_follow_the_freedom_rule():
    """No habitual must/never in the role prompts. (The rule: facts and options, the agent decides.)"""
    for f in (ROOT / "agents").glob("*.md"):
        assert not re.search(r"\b(must|never)\b", f.read_text(), re.I), f.name


def test_files_named_in_skills_and_agents_exist():
    """A path starts at the repo root, or at the folder of the skill (the Agent Skills rule; `part-to-topopt` examples).
    The files that a skill reads on demand (`references/*.md`, #207) count as texts of the skill."""
    texts = [(f, f.parent) for f in plugin_skill_files() + sorted((ROOT / "agents").glob("*.md"))]
    texts += [(ref, f.parent) for f in plugin_skill_files() for ref in sorted((f.parent / "references").glob("*.md"))]
    for f, folder in texts:
        for rel in re.findall(r"`((?:tools|examples|scripts|topopt)/[\w./-]+\.(?:py|sh|yaml|md))`", f.read_text()):
            assert (ROOT / rel).is_file() or (folder / rel).is_file(), f"{f.relative_to(ROOT)}: {rel}"


def test_agents_find_the_code_folder_through_the_environment_tool():
    """tools/ and CONTRACTS.md are in the code folder. The agents get its path from `environment`."""
    for f in (ROOT / "agents").glob("*.md"):
        text = f.read_text()
        assert "topopt folder" not in text, f.name
        if "tools/check_part.py" in text or "CONTRACTS.md" in text:
            assert "`code_folder`" in text and "`environment`" in text, f.name


def test_skills_keep_timing_numbers_in_one_place():
    """The measured hook times are in the README and in the code. The skills point to `check_problem`."""
    for f in [ROOT / "skills" / "start-here" / "SKILL.md", *sorted((ROOT / "agents").glob("*.md"))]:
        assert not re.search(r"\b(148|406|519)\s*s\b|\b99 s\b", f.read_text()), f


def test_start_here_gives_no_instruction_to_edit_the_plugin_folder():
    text = (ROOT / "skills" / "start-here" / "SKILL.md").read_text()
    assert "`settings.solver: amg` in the YAML" not in text   # the engine runs amg for auto on a Mac without the shim
    assert "copy its folder" in text and "runs/<name>-input/" in text


# --- publishing: text files only, no symlinks --------------------------------------------------

def test_plugin_files_are_text_and_not_symlinks():
    folders = [".claude-plugin", "agents", "skills", "scripts", ".claude/skills"]
    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(ROOT / folder):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in dirnames + filenames:
                path = Path(dirpath) / name
                assert not path.is_symlink(), path
            for name in filenames:
                assert b"\0" not in (Path(dirpath) / name).read_bytes(), name


# --- the real CLI, if there is one -------------------------------------------------------------

@pytest.mark.skipif(shutil.which("claude") is None, reason="the claude command is not installed")
def test_claude_plugin_validate_passes():
    with tempfile.TemporaryDirectory() as config:
        env = dict(os.environ, CLAUDE_CONFIG_DIR=config)
        for target in (ROOT, MARKET_JSON):
            r = subprocess.run(["claude", "plugin", "validate", str(target)], capture_output=True,
                               text=True, env=env, timeout=120)
            out = r.stdout + r.stderr
            assert r.returncode == 0 and "Validation failed" not in out, out
