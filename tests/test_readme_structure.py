"""#407: the README is a short entry point: what topopt is, the install for each client, the use, the end. These guards pin the parts that other files link to."""
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def headings(level: int) -> list[str]:
    fence = False
    out = []
    for line in README.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith("#" * level + " "):
            out.append(line[level + 1:].strip())
    return out


def test_the_install_section_has_the_clients_in_the_order_of_the_decision():
    assert headings(2) == ["Install", "Use", "Documentation", "License", "Contributing"]
    assert headings(3) == ["What the setup downloads, and what topopt sends", "Clients", "Other clients"]


def test_the_clients_are_closed_sections_in_the_order_of_the_decision_and_the_old_plugin_anchor_stays():
    clients = README.split("### Clients", 1)[1].split("\n### ", 1)[0]
    summaries = re.findall(r"<details>\s*<summary><b>(.*?)</b></summary>", clients)
    assert summaries == ["Claude Desktop", "Claude Code", "Generic MCP client"]
    # Links from outside the repo (the landing) used the anchor of the old heading "Quick start: Claude Code plugin".
    assert '<a id="quick-start-claude-code-plugin"></a>\n<details>\n<summary><b>Claude Code</b></summary>' in clients
    # A command that the user types is in a code block, not in the text.
    for command in ("/plugin marketplace add chartergroup/topopt#latest\n/plugin install topopt@topopt\n",
                    "claude plugin marketplace update topopt\nclaude plugin update topopt@topopt\n", "/topopt:start-here\n",
                    "sh scripts/topopt-env.sh doctor\n"):
        assert "```\n" + command + "```" in clients, command


def test_each_other_client_is_a_closed_section():
    others = README.split("### Other clients", 1)[1].split("\n## ", 1)[0]
    summaries = re.findall(r"<details>\s*<summary>(.*?)</summary>", others, re.S)
    assert [re.sub(r"<[^>]+>", "", s).split(" (")[0] for s in summaries] == ["Cursor", "Codex CLI", "VS Code", "Gemini CLI"]


def test_the_website_has_one_link_in_the_text_and_the_sketch_links_the_full_size():
    links = re.findall(r"\]\((https://topopt\.ai[^)]*)\)", README) + re.findall(r'href="(https://topopt\.ai[^"]*)"', README)
    assert links.count("https://topopt.ai") == 1
    assert [x for x in links if x != "https://topopt.ai"] == ["https://topopt.ai/assets/early-frame-sketch.png"]
    assert re.search(r'<img src="docs/media/frame-sketch\.png" width="\d+" alt="[^"]+">', README)   # shown small, with a text


def test_the_readme_does_not_link_the_agent_rules_for_a_person():
    assert not re.search(r"\]\([^)]*AGENTS\.md", README)
    assert "(CONTRIBUTING.md)" in README


def test_the_sketch_copy_is_about_1000_px_wide_and_the_media_page_says_how_it_was_made():
    data = (ROOT / "docs" / "media" / "frame-sketch.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1000, 647)
    media = (ROOT / "docs" / "media" / "README.md").read_text(encoding="utf-8")
    assert "frame-sketch.png" in media and "early-frame-sketch.png" in media


def test_the_work_folder_and_the_examples_gallery_are_linked():
    assert "(docs/reference/install.md#the-work-folder)" in README and "(examples/README.md)" in README


def test_the_readme_names_python_3_13_only():
    # topopt supports Python 3.13 only (decided on #469): the badge and the text name no other version.
    assert "![Python 3.13](https://img.shields.io/badge/python-3.13-blue)" in README
    assert re.findall(r"(?<![\d.])3\.1[0-24-9](?!\d)", README) == []


def test_the_readme_and_the_install_doc_say_the_same_about_the_data_that_topopt_sends():
    # Nothing is sent unless the user allows diagnostic data; this version has no such option, and the install doc says so.
    install = (ROOT / "docs" / "reference" / "install.md").read_text(encoding="utf-8")
    assert "<summary>About 200 MB, once. <b>topopt</b> sends no data unless you allow diagnostic data.</summary>" in README
    assert "- **topopt sends no data unless you allow it to send diagnostic data.**" in README
    assert "**What topopt sends: no data, unless you allow diagnostic data.**" in install
    assert "This version has no option for diagnostic data, so it sends nothing." in install
    assert "sends no data." not in README and "sends: no data.**" not in install      # no unqualified claim is left

