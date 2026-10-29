"""The snapshot that the launcher serves while the Python setup is not ready (#370): topopt_mcp/launcher/*.json.

The launcher answers the list requests of the client from these files, so the client sees the real tools from its first answer. The files
are the answers of the real server. They must be made again after a change of a tool, a prompt, a resource or the instructions:
`python -P -m topopt_mcp.snapshot`.

The CI runs this file on macOS too (tests/tiers/platform_files.txt): the lists of the server must be the same there.
"""
import asyncio
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from topopt_mcp import snapshot  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FOLDER = REPO / "topopt_mcp" / "launcher"


def test_the_files_are_the_answers_of_the_real_server(tmp_path):
    out = tmp_path / "made"
    r = subprocess.run([sys.executable, "-P", "-m", "topopt_mcp.snapshot", "--out", str(out)], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert r.returncode == 0, r.stderr
    for name in snapshot.FILE_NAMES:
        committed = (FOLDER / f"{name}.json").read_bytes()
        assert committed == (out / f"{name}.json").read_bytes(), (
            f"the launcher snapshot is out of date: run python -P -m topopt_mcp.snapshot ({name}.json differs)")
    assert sorted(p.name for p in FOLDER.iterdir()) == sorted(f"{n}.json" for n in snapshot.FILE_NAMES)    # no other file, none missing


def test_each_tool_description_is_its_docstring_cleaned_the_same_way_on_every_python():
    """#388: Python 3.13 strips the common indentation of a docstring when it compiles it, and 3.12 keeps it. The server cleans each
    docstring (inspect.cleandoc), so the tool list, and the snapshot made from it on 3.12, is the same on both."""
    from topopt_mcp import server
    tools = asyncio.run(server.mcp.list_tools())
    assert len(tools) >= 30
    for tool in tools:
        text = tool.description or ""
        assert text and text == inspect.cleandoc(text), tool.name      # no common indentation, no blank line at the start or the end
    run_topopt = next(t for t in tools if t.name == "run_topopt")
    assert "Returns at once.\n\nUse after check_problem" in run_topopt.description


@pytest.mark.parametrize("name", snapshot.FILE_NAMES)
def test_each_file_is_one_line_of_json_without_an_inner_newline(name):
    text = (FOLDER / f"{name}.json").read_text(encoding="utf-8")
    assert text.endswith("\n") and "\n" not in text[:-1] and not text.endswith("\n\n")
    json.loads(text)
    assert text.isascii()                                              # the shell copies bytes: no encoding question


def test_the_version_placeholder_is_only_in_the_legacy_initialize_result():
    for name in snapshot.FILE_NAMES:
        text = (FOLDER / f"{name}.json").read_text()
        assert (name == "legacy.initialize") == (snapshot.PLACEHOLDER in text), name
    assert (FOLDER / "legacy.initialize.json").read_text().count(snapshot.PLACEHOLDER) == 1
    assert json.loads((FOLDER / "legacy.initialize.json").read_text())["protocolVersion"] == snapshot.PLACEHOLDER


def test_the_files_hold_no_path_of_this_machine_and_no_time():
    for name in snapshot.FILE_NAMES:
        text = (FOLDER / f"{name}.json").read_text()
        assert str(REPO) not in text and "/tmp/" not in text and "/home/" not in text, name
        assert not re.search(r"\b20\d\d-\d\d-\d\dT\d\d:\d\d", text), name


def test_the_modern_files_carry_what_the_server_sends_and_the_legacy_files_do_not():
    for name in ("discover", "tools", "prompts", "resources", "templates"):
        result = json.loads((FOLDER / f"modern.{name}.json").read_text())
        assert result["resultType"] == "complete" and "_meta" in result, name
    for name in ("tools", "prompts", "resources", "templates"):
        assert "resultType" not in json.loads((FOLDER / f"legacy.{name}.json").read_text()), name
    assert len(json.loads((FOLDER / "modern.tools.json").read_text())["tools"]) == len(json.loads((FOLDER / "legacy.tools.json").read_text())["tools"]) > 20


def test_the_package_includes_the_snapshot_files():
    """pyproject.toml lists the data files of topopt_mcp: the launcher serves them from a plugin copy and from an installed wheel."""
    text = (REPO / "pyproject.toml").read_text()
    assert "launcher/*.json" in text
