"""Guard for the install test checklist (REL-3): docs/install-test-checklist.md.

The page is for a person who installs topopt in one of six clients and reports what happened. These tests read files only.
They check that the page keeps its promises: each client has its section, the page says where its client facts come from,
every repo path that it names exists, the launcher cases that it tells a tester to run exist (and `mcp` is not one of them),
the tools that it expects are in the server, and the other pages point to it.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "install-test-checklist.md"

CLIENT_SECTIONS = (
    "## 2. Claude Code plugin",
    "## 3. Claude Desktop extension and plugin upload",
    "## 4. Cursor",
    "## 5. Codex CLI",
    "## 6. VS Code",
    "## 7. Gemini CLI",
)
# The repo roots that a path on the page can start with.
REPO_PREFIXES = ("scripts/", "docs/", "examples/", "packaging/", ".cursor/", ".codex/", ".claude-plugin/", ".agents/",
                 ".vscode/", ".gemini/", "topopt/", "tests/")
# Files that the tester makes (the snippets are not committed), so they need not exist in the repo.
MADE_BY_THE_TESTER = {".vscode/mcp.json", ".gemini/settings.json", ".gemini/skills/"}
# The doc hosts that the sources table must cover, one for each client.
SOURCE_HOSTS = ("code.claude.com", "claude.com/docs", "cursor.com", "learn.chatgpt.com", "code.visualstudio.com", "geminicli.com")
CORE_TOOLS = ("run_topopt", "run_status", "run_result", "list_runs", "check_problem", "build_regions", "environment",
              "render_part", "verify_run", "verify_status", "team_guide")


def text() -> str:
    assert PAGE.is_file(), "docs/install-test-checklist.md is missing"
    return PAGE.read_text(encoding="utf-8")


def sections() -> dict[str, str]:
    """Heading line -> body, for the `## ` headings."""
    parts = re.split(r"^(## .+)$", text(), flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def test_the_page_says_where_the_client_facts_come_from_and_when():
    head = "\n".join(text().splitlines()[:14])
    assert "analysis of publicly available information" in head
    assert re.search(r"\b2026-\d\d-\d\d\b", head), "the top of the page needs the date of the reading"


@pytest.mark.parametrize("heading", CLIENT_SECTIONS)
def test_every_client_has_a_section_with_numbered_steps(heading):
    found = sections()
    assert heading in found, f"the page needs the section {heading!r}"
    steps = re.findall(r"^\d+\. ", found[heading], flags=re.M)
    assert len(steps) >= 5, f"{heading!r} needs its steps as a numbered list"


def test_the_send_back_section_names_what_to_send():
    body = sections()["## 8. Send back"]
    for item in ("The times", "The first reply of the tool list", "The log lines on a failure", "Screenshots", "Client and version, OS"):
        assert item in body, item


def test_every_repo_path_on_the_page_exists():
    missing = []
    for token in re.findall(r"`([^`\n]+)`", text()):
        word = token.split()[-1] if " " in token else token
        word = word.rstrip(".,:;)")
        if not word.startswith(REPO_PREFIXES) or "<" in word or "*" in word or "..." in word or word in MADE_BY_THE_TESTER:
            continue
        if not (ROOT / word).exists():
            missing.append(word)
    assert not missing, f"the page names files that are not in the repo: {sorted(set(missing))}"


def test_the_page_never_tells_a_tester_to_run_the_mcp_case_of_the_launcher():
    """`mcp` starts the server and waits for a client on stdin. Only sync, status and doctor are for a person."""
    cases = set(re.findall(r"scripts/topopt-env\.sh (\w+)", text()))
    assert cases, "the page needs at least one launcher command"
    assert cases <= {"sync", "status", "doctor"}, cases
    launcher = (ROOT / "scripts" / "topopt-env.sh").read_text(encoding="utf-8")
    for case in cases:
        assert f"sh topopt-env.sh {case}" in launcher, f"the launcher has no case {case}"


def test_the_expected_tools_are_tools_of_the_server():
    server = (ROOT / "topopt_mcp" / "server.py").read_text(encoding="utf-8")
    body = sections()["## 8. Send back"]
    for tool in CORE_TOOLS:
        assert f"`{tool}`" in body, f"the send-back table needs the tool {tool}"
        assert re.search(rf"^def {tool}\(", server, flags=re.M), f"{tool} is not a tool of topopt_mcp/server.py"


def test_the_sources_table_covers_the_docs_of_every_client():
    body = sections()["## 9. Sources"]
    for host in SOURCE_HOSTS:
        assert host in body, f"the sources table needs a page of {host}"
    urls = re.findall(r"https://\S+", body)
    assert len(urls) >= 12
    assert not [u for u in urls if "forum" in u], "a forum post is not an official source"


def owners_named(page):
    return set(re.findall(r"(?<![\w.-])([\w-]+)/topopt(?:-demo)?(?![\w-])", page)) | set(re.findall(r"github\.com/([\w.-]+)/topopt", page))


def test_the_page_does_not_name_the_old_repo_owner():
    assert owners_named(text()) <= {"chartergroup"}, owners_named(text())


def test_the_owner_pattern_finds_each_form_of_a_url():
    for line in ("https://github.com/someone/topopt-demo", "https://raw.githubusercontent.com/someone/topopt-demo/main/scripts/x.sh",
                 "someone/topopt-demo/blob/main/README.md", "the repo someone/topopt.", "`someone/topopt`"):
        assert owners_named(line) == {"someone"}, line
    assert owners_named("https://github.com/chartergroup/topopt/blob/main/README.md") == {"chartergroup"}
    assert not owners_named("the folder ../topopt-demo and topopt")


def test_the_readme_sections_that_the_page_follows_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for client in ("Claude Desktop", "Claude Code", "Generic MCP client"):                 # the closed sections of "### Clients"
        assert f"<summary><b>{client}</b></summary>" in readme, client
    # #385: the long steps moved to the install page; the README keeps the short path and a pointer.
    page = (ROOT / "docs/reference/install.md").read_text(encoding="utf-8")
    for heading in ("## Install the Claude Code plugin", "### What to expect", "## Claude Desktop"):
        assert heading in page, heading
    assert "docs/reference/install.md#claude-desktop" in readme and "docs/reference/install.md#install-the-claude-code-plugin" in readme


def test_the_other_pages_point_to_the_checklist():
    for name in ("README.md", "docs/clients.md"):
        assert "install-test-checklist.md" in (ROOT / name).read_text(encoding="utf-8"), f"{name} needs a link to the checklist"


# --- REL-9: uv is no prerequisite ----------------------------------------------------------------------------------

UV_SENTENCE = ("topopt uses your uv if you have one (uv 0.6.8 or newer); else the setup downloads a pinned uv release and checks its "
               "checksum.")   # 0.6.8: MIN_UV_VERSION of scripts/topopt-env.sh (#390)


@pytest.mark.parametrize("doc", ["README.md", "docs/clients.md", "docs/reference/install.md", "docs/install-test-checklist.md"])
def test_the_docs_say_that_uv_comes_with_the_setup(doc):
    page = (ROOT / doc).read_text(encoding="utf-8")
    assert UV_SENTENCE in re.sub(r"\s+", " ", page).replace("**topopt**", "topopt"), f"{doc} needs the sentence about uv"   # the README writes the name in bold
    assert not re.search(r"only prerequisite", page, re.I), f"{doc} still calls uv a prerequisite"


def test_start_here_names_the_oldest_uv_and_offers_its_update():
    """#390 / #206: the skill names the oldest uv that the launcher uses (MIN_UV_VERSION of the launcher). For a uv on the PATH that
    is too old, step 3 offers the update of that uv, not the install script: a new uv in ~/.local/bin does not help while the old
    one comes first on the PATH."""
    oldest = re.search(r"^MIN_UV_VERSION=(\S+)$", (ROOT / "scripts/topopt-env.sh").read_text(encoding="utf-8"), re.M).group(1)
    for path in ("skills/start-here/SKILL.md", ".agents/skills/start-here/SKILL.md"):
        text = re.sub(r"\s+", " ", (ROOT / path).read_text(encoding="utf-8"))
        assert f"uses the user's `uv` if it is uv {oldest} or newer" in text, path
        step3 = re.search(r" 3\. The row `uv` (.+?) 4\. ", text).group(1)
        assert "`The uv on PATH is too old`" in step3 and "offer the update of that uv" in step3, path
        assert "`brew upgrade uv`, `pip install --upgrade uv`, or `uv self update`" in text, path


@pytest.mark.parametrize("doc", ["docs/reference/install.md", "docs/install-test-checklist.md"])
def test_the_docs_do_not_say_that_the_desktop_extension_skips_the_launcher(doc):
    """Since #389 the extension starts the launcher: it connects at once, so no doc may send the user to turn it off and on until it runs."""
    page = re.sub(r"\s+", " ", (ROOT / doc).read_text(encoding="utf-8"))
    for stale in ("does not use the launcher", "off and on until it runs", "The extension starts the server with `uv run`"):
        assert stale not in page, (doc, stale)


def test_the_client_sections_of_the_page_do_not_ask_for_a_uv_first():
    """The Desktop section keeps its step: the build of the `.mcpb` file runs with uv (the extension itself starts the launcher, #389)."""
    found = sections()
    for heading in CLIENT_SECTIONS:
        if "Claude Desktop" in heading:
            continue
        first = re.findall(r"^1\. (.+)$", found[heading], flags=re.M)[0]
        assert not first.startswith("Install uv"), (heading, first)


# --- #401: the Code tab of Claude Desktop ----------------------------------------------------------------------------

CODE_TAB_LABELS = ("Settings > Plugins", "Add plugins", "+ Add", "Add marketplace", "Add from a repository", "URL",
                   "`chartergroup/topopt#latest`", "Sync", "Discover", "topopt: topology optimization for 3D printing",
                   "Manage marketplaces", "This plugin includes local MCP servers", "Continue", "Disable plugin",
                   "Check my machine with topopt")


def install_section(heading: str) -> str:
    """The body of `heading` on the install page, up to the next heading of the same level or a higher one."""
    page = (ROOT / "docs/reference/install.md").read_text(encoding="utf-8")
    level = len(heading) - len(heading.lstrip("#"))
    body = page.split(f"\n{heading}\n", 1)[1]
    end = re.search(rf"^#{{1,{level}}} ", body, flags=re.M)
    return body[:end.start()] if end else body


def test_the_readme_install_step_names_the_route_of_the_code_tab_with_its_steps():
    """In the Code tab, `/plugin marketplace add` opens Settings > Plugins (Mac test, 2026-10-27). The Claude Code block of the README says
    so, gives the short steps of the app, and links the full steps of the install page."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    step = readme.split("<summary><b>Claude Code</b></summary>", 1)[1].split("</details>", 1)[0]
    assert "In the Code tab of Claude Desktop, the plugin command opens Settings > Plugins." in step
    tab = step.split("In the Code tab of Claude Desktop", 1)[1]
    assert len(re.findall(r"^\d+\. ", tab, flags=re.M)) == 4 and "Add from a repository" in tab and "press Sync" in tab
    lines = [line for line in step.splitlines() if "#the-code-tab-the-plugin" in line]
    assert len(lines) == 1 and "docs/reference/install.md#the-code-tab-the-plugin" in lines[0], lines


def test_the_install_page_has_the_code_tab_steps_with_the_labels_of_the_app():
    steps = install_section("### The Code tab: the plugin")
    assert len(re.findall(r"^\d+\. ", steps, flags=re.M)) >= 5
    missing = [label for label in CODE_TAB_LABELS if label not in steps]
    assert not missing, missing
    assert "anthropics/claude-code#93400" in steps                # press Add right after Sync: the upstream report


def test_the_token_note_holds_only_for_a_private_repo():
    steps = re.sub(r"\s+", " ", install_section("### The Code tab: the plugin"))
    token = [s for s in re.split(r"(?<=\.) ", steps) if "personal access token" in s]
    assert token and all(s.startswith("If the repo is private") for s in token), token
    assert "A public repo needs no token." in steps


def test_the_desktop_texts_say_what_the_mac_test_showed():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    desktop = install_section("## Claude Desktop")
    for text in (readme, desktop):
        assert "Nobody has tested these steps" not in text
    assert "2026-10-27" in desktop and "release 0.4.0" in desktop and "Claude Desktop 3.6556.2" in desktop
    assert "The Code tab test did not record the version of the app." in re.sub(r"\s+", " ", desktop)   # only the chat test had it
