"""P9-2a: the first start does not look like a failure. The texts that a user meets BEFORE the failure.

The plugin (Claude Code) sets up its own Python on the first start. That takes longer than the 30 s that Claude Code gives an MCP
server to connect. So the launcher answers from a snapshot of the lists, and a tool call gives the setup status until the setup is done
(#370). The extension (Claude Desktop) starts the same launcher (#206), so its first start is the same. Each surface says so where
the user reads it first:

  plugin        `.claude-plugin/plugin.json` `description` and the plugin entry of `.claude-plugin/marketplace.json`
  extension     `packaging/mcpb/manifest.json` `long_description`
  launcher      the line that `scripts/topopt-env.sh mcp` prints when it stops waiting (tests/test_plugin_launcher.py)

The two numbers of the first start are written in one phrase each. Every surface uses these phrases, so a changed number is a
changed test. (The numbers are not new: they were in the README, the skill `start-here`, the manifest and the launcher before.)
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_JSON = ROOT / ".claude-plugin" / "plugin.json"
MARKET_JSON = ROOT / ".claude-plugin" / "marketplace.json"
MANIFEST = ROOT / "packaging" / "mcpb" / "manifest.json"

FIRST_START_TIME = "1 to 3 minutes"
FIRST_START_SIZE = "about 200 MB"


def plugin_description():
    return json.loads(PLUGIN_JSON.read_text())["description"]


def marketplace_plugin_description():
    entries = json.loads(MARKET_JSON.read_text())["plugins"]
    assert [e["name"] for e in entries] == ["topopt"]
    return entries[0]["description"]


def manifest_long_description():
    return json.loads(MANIFEST.read_text())["long_description"]


def sentence_with(text, word):
    """The one sentence of `text` that holds `word`."""
    found = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if word in s]
    assert len(found) == 1, (word, found)
    return found[0]


# --- the plugin (Claude Code) ------------------------------------------------------------------------------------

@pytest.mark.parametrize("read", [plugin_description, marketplace_plugin_description])
def test_plugin_texts_name_the_first_start_the_status_answer_and_that_no_reconnect_is_needed(read):
    """`description` of plugin.json and of the marketplace entry: the time, that `topopt` connects in time with its tools, that a tool call
    gives the setup status, and that the tools work in the same session. No "failed" and no reconnect (the launcher does not fail, #370)."""
    text = read()
    first = sentence_with(text, "first start")
    assert FIRST_START_TIME in first
    assert "connects at once with all its tools" in first or "connects at once with all its tools" in text
    assert "setup status" in text and "same session" in text
    assert text.index(FIRST_START_TIME) < text.index("connects at once") < text.index("setup status") < text.index("same session")   # the order of events
    assert "failed" not in text and "reconnect" not in text and "/mcp" not in text


def test_the_two_plugin_descriptions_give_the_same_first_start_sentence():
    """One sentence, two files: the user reads the same fact in the marketplace list and in the plugin."""
    assert sentence_with(plugin_description(), "first start") == sentence_with(marketplace_plugin_description(), "first start")


# --- the extension (Claude Desktop) ------------------------------------------------------------------------------

def test_extension_text_names_the_first_start_the_status_answer_and_what_to_do_when_the_tools_are_missing():
    """The manifest starts the launcher (#206), so the text tells the same events as the plugin texts: the time, that topopt connects at
    once with its tools, that a tool call gives the setup status, and that the tools then work in the same chat. The advice for a start
    that does not come up stays: off and on (Claude Desktop has no reconnect command)."""
    text = manifest_long_description()
    first = sentence_with(text, FIRST_START_TIME)
    assert FIRST_START_SIZE in first and "first start" in first
    assert text.index(FIRST_START_TIME) < text.index("connects at once with all its tools") < text.index("setup status") \
        < text.index("same chat")                                     # the order of events
    assert "no uv" in text                                            # the launcher gets a uv when the Mac has none
    after = sentence_with(text, "off and on")
    assert "tools are missing" in after and "turn the extension off and on" in after
    assert "uv cache" in text                                         # a stopped start is not lost: the next start goes on
    assert text.index("same chat") < text.index("off and on")
    assert "failed" not in text and "reconnect" not in text


def test_extension_text_keeps_the_work_folder_text_of_p9_1a():
    text = manifest_long_description()
    assert ("Runs, STLs and results go in the work folder: the folder Documents/topopt in your home folder, "
            "or the folder that you choose in the settings of the extension.") in text


def test_extension_text_gives_no_advice_that_belongs_to_the_plugin():
    """Claude Desktop has no `/mcp` command. The advice for it is off and on (or a full quit), not a reconnect."""
    assert "/mcp" not in manifest_long_description()


# --- all surfaces ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [PLUGIN_JSON, MARKET_JSON, MANIFEST], ids=lambda p: p.name)
def test_first_start_texts_have_no_habitual_caps_and_no_talk_material(path):
    """The rule of tests/test_mcpb.py (`test_template_text_has_no_habitual_caps_and_no_talk_material`) and of the role prompts,
    for the three files: facts and options, no habitual must or never, no text of a talk."""
    text = path.read_text()
    assert not re.search(r"\b(must|never)\b", text, re.I)
    assert not re.search(r"t-shirt|\bshirt\b|run mcp|tuesday|on stage|audience|demo day", text, re.I)


def test_the_numbers_of_the_first_start_are_the_same_in_every_text_that_gives_them():
    """One phrase for the time and one for the size, in the README, in `start-here` (and its copy for the other clients), in the
    manifest and in the launcher. Measured figures of one machine (44 s, 63 to 77 s, 205 MB) stay as dated measurements; the phrases
    here are the figure that a user can expect."""
    files = {
        "README.md": (ROOT / "README.md").read_text(),
        "skills/start-here/SKILL.md": (ROOT / "skills" / "start-here" / "SKILL.md").read_text(),
        ".agents/skills/start-here/SKILL.md": (ROOT / ".agents" / "skills" / "start-here" / "SKILL.md").read_text(),
        "packaging/mcpb/manifest.json": manifest_long_description(),
        "scripts/topopt-env.sh": (ROOT / "scripts" / "topopt-env.sh").read_text(),
    }
    for name, text in files.items():
        times = set(re.findall(r"\b\d+ to \d+ minutes\b", text))
        assert times <= {FIRST_START_TIME}, (name, times)
        assert FIRST_START_TIME in text, name
        assert FIRST_START_SIZE in text, name


# --- the table "How long each client waits on the first start" (P9-2b) ------------------------------------------

def readme_wait_table():
    """The rows of the table in the section of the install page (linked from the README), as {first cell: second cell}."""
    page = (ROOT / "docs" / "reference" / "install.md").read_text(encoding="utf-8")
    start = page.index("\n### How long each client waits on the first start\n")
    section = page[start: re.search(r"\n#{2,3} ", page[start + 1:]).start() + start + 1]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 4 and cells[0] not in ("Client", "---"):
            rows[cells[0]] = cells[1]
    return rows


def test_the_wait_table_names_every_client_and_the_waits_match_the_files():
    """The numbers of the table are the numbers of the files that set them: the launcher default and the Codex file."""
    import tomllib
    rows = readme_wait_table()
    assert set(rows) == {"Claude Code (the plugin)", "The launcher", "Codex CLI", "Gemini CLI", "Cursor, VS Code",
                         "Claude Desktop (the extension)"}
    launcher = (ROOT / "scripts" / "topopt-env.sh").read_text()
    default = re.search(r"elif \[ -t 0 \]; then MCP_WAIT=(\d+); else MCP_WAIT=0; fi", launcher).group(1)   # a terminal: 24; a client: 0
    assert rows["The launcher"].startswith("0 s for a client") and f"{default} s in a terminal" in rows["The launcher"]
    codex = tomllib.loads((ROOT / ".codex" / "config.toml").read_text())["mcp_servers"]["topopt"]
    assert f"{codex['startup_timeout_sec']} s" in rows["Codex CLI"]
    assert f"{codex['env']['TOPOPT_MCP_WAIT']} s" in rows["Codex CLI"]
    assert rows["Claude Code (the plugin)"].startswith("30 s")
    assert "not measured" in rows["Claude Desktop (the extension)"]
