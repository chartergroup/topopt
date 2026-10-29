"""Tests for the material reference (CONTRACTS.md section 25.7, item P4b-1).

One source: topopt_mcp/materials/reference.md. The MCP server offers it as the resource topopt://materials/reference.
It is a document. The engine does not read it, no tool returns a number from it, and there is no default allowable
(decision D7; revised on 2026-10-24: the filament catalog is another file, tested in tests/test_materials_catalog.py). These tests check the file, the package data, the resource, and the defaults that must stay None.
The message of `topopt verify` for an unset allowable is tested in tests/test_verify_run.py.
"""
import asyncio
import dataclasses
import datetime
import fnmatch
import inspect
import json
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from topopt_mcp import materials, server  # noqa: E402

URI = "topopt://materials/reference"
REFERENCE = ROOT / "topopt_mcp" / "materials" / "reference.md"
HABIT = re.compile(r"\b(must|never|always)\b", re.I)  # the freedom rule: facts and options, no habitual caps


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 60))


def file_text() -> str:
    return REFERENCE.read_text(encoding="utf-8")


def section(text: str, title: str) -> str:
    """The lines of a `## <title>` section, up to the next `## ` heading."""
    m = re.search(rf"^## {re.escape(title)}.*?$(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"no section {title!r}"
    return m.group(1)


# --- the file and the package data ----------------------------------------------------------------------------------

def test_the_constants_name_the_one_file_and_the_uri():
    assert materials.URI == URI and materials.NAME == "material-reference"
    assert ROOT.joinpath("topopt_mcp", *materials.FILE) == REFERENCE and REFERENCE.is_file()
    assert materials.read_reference() == file_text() == materials.load_reference()
    assert len(materials.DESCRIPTION) <= 300 and "no default allowable" in materials.DESCRIPTION.lower()


def test_package_data_covers_the_reference_file():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    rel = "/".join(materials.FILE)
    assert any(fnmatch.fnmatch(rel, pattern) for pattern in data["topopt_mcp"]), data["topopt_mcp"]


def test_a_missing_or_empty_file_is_a_named_error(monkeypatch, tmp_path):
    monkeypatch.setattr(materials, "FILE", ("materials", "no-such-file.md"))
    with pytest.raises(materials.MaterialFileError, match="no-such-file.md"):
        materials.read_reference()
    (tmp_path / "empty.md").write_text("  \n", encoding="utf-8")

    class Fake:  # an installed package whose file has no text
        def joinpath(self, *parts):
            return tmp_path / "empty.md"

    monkeypatch.setattr(materials.resources, "files", lambda name: Fake())
    with pytest.raises(materials.MaterialFileError, match="empty"):
        materials.read_reference()


def test_a_server_without_the_file_starts_and_serves_no_material_resource(tmp_path):
    """A broken install loses the resource only: the other resources and the tools still work."""
    code = (
        "import asyncio, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from topopt_mcp import materials\n"
        "materials.FILE = ('materials', 'no-such-file.md')\n"
        "from topopt_mcp import server\n"
        "async def go():\n"
        "    uris = sorted(str(r.uri) for r in await server.mcp.list_resources())\n"
        "    tools = sorted(t.name for t in await server.mcp.list_tools())\n"
        "    return uris, tools\n"
        "uris, tools = asyncio.run(go())\n"
        "import json\n"
        "print(json.dumps([uris, 'team_guide' in tools, 'verify_run' in tools]))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert out.returncode == 0, out.stderr
    uris, has_guide, has_verify = json.loads(out.stdout.strip().splitlines()[-1])
    assert URI not in uris and any(u.startswith("topopt://roles/") for u in uris) and has_guide and has_verify
    assert "material reference is not available" in out.stderr and "no-such-file.md" in out.stderr


def test_wheel_ships_the_reference_and_importlib_resources_reads_it(tmp_path):
    from test_packaging import _build_wheel

    wheel = _build_wheel(tmp_path / "dist")
    if wheel is None:
        pytest.skip("neither `uv build` nor `python -m build` could build a wheel here")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert "topopt_mcp/materials/reference.md" in names and "topopt_mcp/materials.py" in names
    target = tmp_path / "installed"
    zipfile.ZipFile(wheel).extractall(target)
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(target)!r})\n"
        "import topopt_mcp\n"
        "from topopt_mcp import materials\n"
        f"assert topopt_mcp.__file__.startswith({str(target)!r}), topopt_mcp.__file__\n"
        "print(json.dumps(materials.read_reference()))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == file_text()


# --- the MCP resource -----------------------------------------------------------------------------------------------

def test_resource_is_listed_and_returns_the_text():
    async def go():
        resources = {str(r.uri): r for r in await server.mcp.list_resources()}
        assert URI in resources
        res = resources[URI]
        contents = list(await server.mcp.read_resource(URI))
        return res, contents

    res, contents = run(go())
    assert res.name == materials.NAME and res.title == materials.TITLE and res.description == materials.DESCRIPTION
    assert res.mime_type == "text/markdown"
    assert len(contents) == 1 and contents[0].mime_type == "text/markdown"
    assert contents[0].content == file_text()


def test_stdio_server_lists_and_serves_the_reference():
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(ROOT))

    async def go():
        async with Client(params) as client:
            resources = {str(r.uri): r for r in (await client.list_resources()).resources}
            read = await client.read_resource(URI)
            return resources, read.contents[0]

    resources, content = run(go())
    assert URI in resources and resources[URI].description == materials.DESCRIPTION
    assert content.text == file_text() and content.mime_type == "text/markdown"


def test_no_tool_returns_a_material_number():
    """Decision D7 (revised on 2026-10-24): the reference is a document. The three tools of the filament catalog are the only tools with a
    material in the name, and the reference stays the one resource with materials in its address."""
    async def go():
        return await server.mcp.list_tools()

    tools = run(go())
    assert tools
    assert {t.name for t in tools if "material" in t.name.lower()} == {"find_material", "save_material", "printer_materials"}
    resources = {str(r.uri) for r in run(server.mcp.list_resources())}
    assert {u for u in resources if u.startswith("topopt://materials/")} == {URI}


def test_the_message_of_verify_and_the_resource_agree():
    from topopt.verify import run as vrun
    assert vrun.MATERIAL_REFERENCE_URI == materials.URI
    assert vrun.MATERIAL_REFERENCE_FILE == "topopt_mcp/" + "/".join(materials.FILE)
    assert materials.URI in server.verify_run.__doc__    # the description of the tool names the resource


# --- the text: sources, dates and the freedom rule ------------------------------------------------------------------

def sheet_numbers(text: str) -> dict:
    """The entries of the section `Sheets`: number -> the text of the entry."""
    body = section(text, "Sheets")
    entries = {}
    for m in re.finditer(r"^(\d+)\. (.*?)(?=^\d+\. |\Z)", body, re.S | re.M):
        entries[int(m.group(1))] = m.group(2)
    return entries


def test_every_table_row_cites_a_sheet_and_every_sheet_has_an_address_and_a_date():
    text = file_text()
    sheets = sheet_numbers(text)
    assert sorted(sheets) == list(range(1, len(sheets) + 1)) and len(sheets) >= 8
    for n, entry in sheets.items():
        assert re.search(r"https://\S+", entry), f"sheet {n} has no address"
    opened = re.findall(r"opened on (\d{4}-\d{2}-\d{2})", text)
    assert opened, "the file says no date"
    for day in opened:
        assert datetime.date.fromisoformat(day) <= datetime.date.today(), day
    cited = set()
    for title in ("PETG: tensile strength of printed bars", "PLA: tensile strength of printed bars"):
        rows = [ln for ln in section(text, title).splitlines() if ln.startswith("| ") and not ln.startswith("| Sheet") and "---" not in ln]
        assert len(rows) >= 3, title
        for row in rows:
            first = row.split("|")[1]
            refs = [int(n) for n in re.findall(r"\[(\d+)\]", first)]
            assert refs and all(n in sheets for n in refs), row
            cited.update(refs)
    assert cited <= set(sheets)


def test_both_materials_have_x_y_and_z_columns_and_the_limits_are_in_the_text():
    text = file_text()
    for title in ("PETG: tensile strength of printed bars", "PLA: tensile strength of printed bars"):
        head = next(ln for ln in section(text, title).splitlines() if ln.startswith("| Sheet"))
        assert "In the layer plane" in head and "Across the layers (Z)" in head
    leaves_out = section(text, "What the numbers leave out").lower()
    for word in ("creep", "temperature", "version", "safety factor", "scatter"):
        assert word in leaves_out, word
    assert "ISO 527" in text


def test_the_text_gives_no_default_no_recommendation_and_no_habitual_caps():
    text = file_text()
    assert not HABIT.search(text), HABIT.search(text).group()
    assert "no default allowable" in text
    assert "decision D7" in text
    # the YAML example holds placeholders and no number: the choice is the reader's
    fence = re.search(r"```yaml\n(.*?)```", text, re.S)
    assert fence
    for line in fence.group(1).splitlines():
        value = line.split(":", 1)[1].strip()
        assert not re.search(r"^\d", value), line


# --- decision D7: the engine default of the allowable stays None ----------------------------------------------------

def test_engine_default_allowable_stays_none():
    from topopt.config import LoadCase, Material, load_config

    case = LoadCase(name="a", weight=1.0, forces=())
    assert case.allowable_mpa is None and case.stress_limit is None and case.safety_factor is None
    assert case.allowable_source is None and case.stress_limit_mpa is None
    cfg = load_config(ROOT / "examples" / "hook" / "hook.yaml")
    for lc in cfg.load_cases:
        assert lc.allowable_mpa is None and lc.stress_limit is None and lc.safety_factor is None, lc.name
    # the Material has stiffness and mass only: no strength, no allowable, no table of materials
    assert {f.name for f in dataclasses.fields(Material)} == {"E", "nu", "density"}
    # the documents say the same: the reference has no default, and the note of verify names the reference
    assert "no default allowable" in materials.read_reference()


def test_verify_options_have_no_default_allowable():
    from topopt.cli import build_parser
    from topopt.verify import run as vrun

    for fn in (vrun.verify_run, vrun.preflight, server.verify_run):
        assert inspect.signature(fn).parameters["allowable_mpa"].default is None, fn
    parser = build_parser()
    args = parser.parse_args(["verify", "somewhere"])
    assert args.allowable_mpa is None
    sub = next(a for a in parser._actions if getattr(a, "choices", None) and "verify" in a.choices)
    assert sub.choices["verify"].get_default("allowable_mpa") is None
    assert "no default allowable" in vrun.no_allowable_fact()["note"]                        # the message says it too
