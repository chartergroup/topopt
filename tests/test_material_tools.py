"""PM-2: the MCP tools find_material, save_material and printer_materials, and the `material` key of check_problem (CONTRACTS.md section 18.8)."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import compat, levers_cards, materials_catalog as mc, preferences, printers  # noqa: E402
from topopt.config import load_config  # noqa: E402
from topopt_mcp import server  # noqa: E402

USER_FILE = mc.USER_FILE_NAME
NEW_TOOLS = ("find_material", "save_material", "printer_materials")
ABS = "polymaker-polylite-abs"
PLA_BASIC = "bambu-lab-pla-basic"

ACME = {"id": "acme-pla-plus", "maker": "Acme", "product": "PLA Plus", "family": "PLA", "source": "user",
        "E_xy_mpa": 2600, "E_z_mpa": 2400, "density_g_cm3": 1.24, "nozzle_c": [200, 230], "bed_c": [45, 60], "enclosure": "no",
        "abrasive": False, "soft": False}


@pytest.fixture
def work(tmp_path, monkeypatch):
    """The server with a work folder of this test (the environment probe is replaced by a small dict)."""
    folder = tmp_path / "work"
    folder.mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(folder))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    monkeypatch.chdir(tmp_path)
    import topopt.environment as env_module
    monkeypatch.setattr(env_module, "environment_data", lambda yaml=None: {"python": {"executable": sys.executable}, "ok": None})
    return folder


def tiny_spec(**extra):
    return {"name": "tiny", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}]},
        "load": {"tip": [{"box": {"min": [16, 2, 0], "max": [20, 8, 6]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
        "settings": {"iterations": 3, "voxel_size": 2.0}, **extra}


def mcp_tools():
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def ids(group):
    return [e["id"] for e in group]


# --- the tools in the list ---------------------------------------------------------------------------------------------------

def test_the_three_tools_are_listed_have_short_descriptions_and_are_read_only_in_the_reference():
    tools = mcp_tools()
    assert set(NEW_TOOLS) <= set(tools)
    for name in (*NEW_TOOLS, "save_preferences", "check_problem", "build_regions"):
        assert len(tools[name].description) <= 4096, (name, len(tools[name].description))
    assert len(tools["save_preferences"].description) <= 2048                       # the pin of PM-1 stays
    assert tools["find_material"].input_schema["required"] == ["name"] and list(tools["find_material"].input_schema["properties"]) == ["name"]
    assert tools["save_material"].input_schema["required"] == ["profile"] and list(tools["save_material"].input_schema["properties"]) == ["profile"]
    assert tools["save_material"].input_schema["additionalProperties"] is False
    assert "required" not in tools["printer_materials"].input_schema and list(tools["printer_materials"].input_schema["properties"]) == ["printer"]
    for name in ("find_material", "printer_materials"):
        assert tools[name].input_schema["additionalProperties"] is False        # every tool is strict (#174)
    assert set(NEW_TOOLS) <= set(levers_cards.READ_ONLY_TOOLS)
    assert {c.key.partition(":")[0] for c in levers_cards.CARDS if c.group == "tool"}.isdisjoint(NEW_TOOLS)
    assert "filaments" in tools["save_preferences"].input_schema["properties"] and "find_material" in tools["save_preferences"].description
    for word in ("source_url", "source", "null", "E_xy_mpa", "density_g_cm3", "topopt-filaments.yaml"):
        assert word in tools["save_material"].description or word in tools["find_material"].description, word
    # the key `material` of check_problem is told in the text of find_material (fix round 1, N-5: check_problem has no room left)
    assert "`material`" in tools["find_material"].description and "check_problem" in tools["find_material"].description
    assert "another id" in tools["save_material"].description and "another product name or alias" in tools["save_material"].description


def test_the_tool_mock_of_the_evals_is_the_list_of_the_server():
    mock = {t["name"]: t for t in json.loads((ROOT / "evals" / "mocks" / "topopt" / "_tools.json").read_text(encoding="utf-8"))["tools"]}
    live = mcp_tools()
    assert set(mock) == set(live)
    for name in (*NEW_TOOLS, "save_preferences", "check_problem", "build_regions"):
        assert inspect.cleandoc(mock[name]["description"]) == inspect.cleandoc(live[name].description), name      # no indent: Python 3.13 strips it from a docstring, 3.12 does not
    assert mock["save_material"]["inputSchema"]["additionalProperties"] is False


def test_the_reference_says_what_each_tool_does_and_contracts_names_them():
    for name in NEW_TOOLS:
        assert levers_cards.READ_ONLY_TOOLS[name]
    assert "until a YAML names it" in levers_cards.READ_ONLY_TOOLS["save_material"]
    contracts = " ".join((ROOT / "CONTRACTS.md").read_text(encoding="utf-8").split())
    assert "`find_material` and `printer_materials` are in the list of tools that only report" in contracts


# --- find_material -----------------------------------------------------------------------------------------------------------

def test_find_material_gives_match_ambiguous_and_none_in_one_shape(work):
    keys = {"ok", "status", "query", "catalog_version", "matched_by", "origin", "profile", "derives", "candidates", "differs",
            "suggestions", "makers", "user_file", "note"}
    match = server.find_material("my Polymaker PolyLite ABS")
    assert set(match) == keys and match["status"] == "match" and match["origin"] == "catalog"
    assert match["profile"]["id"] == ABS and match["profile"]["E_z_mpa"] == 2080.9 and match["profile"]["nu"] is None
    assert match["derives"]["sets"] == {"material.E": 2080.9, "material.density": 1.04} and match["derives"]["assumed"] == {"material.nu": 0.35}
    assert match["user_file"]["exists"] is False
    family = server.find_material("PLA")
    assert set(family) == keys and family["status"] == "ambiguous" and family["matched_by"] == "family" and len(family["candidates"]) > 3
    assert family["profile"] is None and "Ask the user which one" in family["note"]
    none = server.find_material("Frobnicator 9000")
    assert set(none) == keys and none["status"] == "none" and none["profile"] is None and none["candidates"] == []
    assert "save_material" in none["note"] and none["makers"]


def test_find_material_finds_a_user_filament_and_the_file_is_read_at_each_call(work):
    assert server.find_material("acme pla plus")["status"] == "none"
    saved = server.save_material(dict(ACME))
    assert saved["ok"] and saved["created"] is True and Path(saved["path"]) == work / USER_FILE
    found = server.find_material("acme pla plus")                                  # no restart
    assert found["status"] == "match" and found["origin"] == "user" and found["profile"]["E_z_mpa"] == 2400
    assert found["user_file"]["exists"] and found["user_file"]["entries"] == 1


def test_find_material_names_a_broken_user_file_as_a_fact_and_the_catalog_still_works(work):
    (work / USER_FILE).write_text("materials: [oops\n")
    r = server.find_material("polylite abs")
    assert r["ok"] and r["status"] == "match" and r["user_file"]["problems"]


# --- save_material -----------------------------------------------------------------------------------------------------------

def test_save_material_writes_only_the_user_file_and_the_filament_is_live_in_a_yaml(work, tmp_path):
    r = server.save_material(dict(ACME))
    assert r["ok"] and r["created"] and not r["replaced"] and "problems" not in r
    assert sorted(p.name for p in work.iterdir()) == [USER_FILE]                    # no other file, no part file, no folder
    doc = yaml.safe_load((work / USER_FILE).read_text())
    assert [m["id"] for m in doc["materials"]] == ["acme-pla-plus"]
    built = server.build_regions(tiny_spec(material={"filament": "Acme PLA Plus"}))
    assert built["ok"], built
    cfg = load_config(work / built["yaml_rel"])
    assert cfg.material.E == 2400.0 and cfg.material.density == 1.24 and cfg.filament.profile.origin == "user"
    again = server.save_material({**ACME, "E_z_mpa": 2300})
    assert again["ok"] and again["replaced"] is True and again["created"] is False
    assert "next check or run" in again["note"]
    assert load_config(work / built["yaml_rel"]).material.E == 2300.0                # the YAML text did not change; the profile did


@pytest.mark.parametrize("profile, expect", [
    ({**ACME, "id": ABS}, "Use another id"),                                         # the id of a shipped filament
    ({**ACME, "id": "other", "maker": "Polymaker", "product": "PolyLite ABS"}, "another product name or alias"),
    ({**ACME, "E_xy_mpa": -5}, "E_xy_mpa"),
    ({**ACME, "price": 20}, "price"),                                                # the catalog holds no price
    ({k: v for k, v in ACME.items() if k != "source"}, "source"),
    ("not a mapping", "profile must be a mapping"),
])
def test_save_material_refuses_a_wrong_entry_and_writes_nothing(work, profile, expect):
    r = server.save_material(profile)
    assert r["ok"] is False and expect in r["error"], r
    assert list(work.iterdir()) == []


def test_save_material_does_not_write_over_a_file_that_it_cannot_read(work):
    (work / USER_FILE).write_text("materials: [oops\n")
    r = server.save_material(dict(ACME))
    assert r["ok"] is False and "did not change" in r["error"]
    assert (work / USER_FILE).read_text() == "materials: [oops\n"


# --- printer_materials -------------------------------------------------------------------------------------------------------

def test_printer_materials_groups_every_filament_once_with_the_reasons(work):
    r = server.printer_materials("Bambu A1 mini")
    assert r["ok"] and r["printer"]["id"] == "bambu-lab-a1-mini" and r["printer"]["named_by"] == "argument" and r["printer"]["from"].startswith("catalog")
    groups = r["groups"]
    assert list(groups) == ["ok", "care", "no", "unknown"] and r["counts"] == {g: len(groups[g]) for g in groups}
    every = [e["id"] for g in groups.values() for e in g]
    current = [p for p in mc.load_catalog().profiles if not p.superseded_by]         # an older sheet is counted in the note, not listed
    assert len(every) == len(set(every)) == len(current) == sum(r["counts"].values())
    assert len(current) == len(mc.load_catalog().profiles) - 1 and "bambu-lab-pla-basic-v2" not in every
    assert PLA_BASIC in ids(groups["ok"]) and ABS in ids(groups["no"]) and "bambu-lab-pa6-cf" in ids(groups["no"])
    assert all(e["reasons"] == [] for e in groups["ok"])                             # nothing is open for an ok filament
    for state in ("care", "no", "unknown"):
        for e in groups[state]:
            assert e["reasons"] and any(x["state"] == state for x in e["reasons"]), e["id"]
            assert all(x["state"] != "ok" and x["text"] and x["check"] in compat.CHECKS for x in e["reasons"]), e["id"]
            assert e["source"] and e["origin"] == "catalog"
    abs_entry = next(e for e in groups["no"] if e["id"] == ABS)
    [enclosure] = [x for x in abs_entry["reasons"] if x["check"] == "enclosure"]
    assert enclosure["state"] == "no" and "needs an enclosed printer" in enclosure["text"] and "is open" in enclosure["text"]
    assert "Polymaker PolyLite ABS" in enclosure["text"] and "Bambu Lab A1 mini" in enclosure["text"]
    assert "A state is a fact" in r["note"] and "not a stop" in r["note"]


def test_printer_materials_for_an_enclosed_printer_moves_abs_out_of_no(work):
    mini = server.printer_materials("Bambu A1 mini")["groups"]
    p1s = server.printer_materials("Bambu P1S")["groups"]
    assert ABS in ids(mini["no"]) and ABS in ids(p1s["ok"])
    assert PLA_BASIC in ids(mini["ok"]) and PLA_BASIC in ids(p1s["ok"])
    assert "bambu-lab-pa6-cf" in ids(p1s["care"])
    nozzle = [x for e in p1s["care"] if e["id"] == "bambu-lab-pa6-cf" for x in e["reasons"] if x["check"] == "nozzle_wear"]
    assert nozzle and "hardened nozzle" in nozzle[0]["text"] and "stainless" in nozzle[0]["text"]


def test_printer_materials_takes_the_printer_and_the_filaments_from_the_preferences(work):
    server.save_material(dict(ACME))
    assert preferences.save(work, print={"printer_model": "Bambu P1S"}, filaments={"available": [ABS, "acme pla plus", PLA_BASIC]})["ok"]
    r = server.printer_materials()
    assert r["ok"] and r["printer"]["id"] == "bambu-lab-p1s" and r["printer"]["named_by"] == "preferences"
    assert [(a["id"], a["state"]) for a in r["available"]] == [(ABS, "ok"), ("acme-pla-plus", "ok"), (PLA_BASIC, "ok")]
    marked = [e["id"] for g in r["groups"].values() for e in g if e["available"]]
    assert sorted(marked) == sorted([ABS, "acme-pla-plus", PLA_BASIC])
    acme = next(e for e in r["groups"]["ok"] if e["id"] == "acme-pla-plus")
    assert acme["origin"] == "user" and "user" in acme["source"]
    assert r["user_file"]["exists"] and r["user_file"]["entries"] == 1
    other = server.printer_materials("Bambu A1 mini")                                    # the argument wins; the filaments stay
    assert other["printer"]["named_by"] == "argument" and [a["id"] for a in other["available"]] == [ABS, "acme-pla-plus", PLA_BASIC]
    assert dict((a["id"], a["state"]) for a in other["available"])[ABS] == "no"


def test_a_filament_that_is_not_found_now_has_no_state_in_the_available_list(work):
    server.save_material(dict(ACME))
    assert preferences.save(work, filaments={"available": ["acme-pla-plus"]})["ok"]
    (work / USER_FILE).unlink()
    r = server.printer_materials("Bambu P1S")                                             # the list is not used now (a fact in the preferences)
    assert r["ok"] and r["available"] == []
    assert preferences.read(work)["not_used"]


def test_printer_materials_without_a_printer_says_how_to_name_one(work):
    r = server.printer_materials()
    assert r["ok"] is False and "No printer is named" in r["error"] and "find_printer" in r["error"] and "print.printer_model" in r["error"]
    blank = server.printer_materials("   ")
    assert blank["ok"] is False and "No printer is named" in blank["error"]


def test_printer_materials_does_not_pick_a_printer_for_the_user(work):
    none = server.printer_materials("Frobnicator 9000")
    assert none["ok"] is False and none["status"] == "none" and "find_printer" in none["error"] and none["candidates"] == []
    amb = server.printer_materials("bambu p")
    assert amb["ok"] is False and amb["status"] == "ambiguous" and [c["id"] for c in amb["candidates"]] == ["bambu-lab-p1s", "bambu-lab-p2s"]


def test_printer_materials_for_a_user_printer_with_null_fields_gives_unknown_not_a_guess(work):
    assert server.save_printer({"id": "frob-1", "vendor": "Frob", "model": "Frob One", "printer_type": "fdm", "bed_mm": [200, 200], "source": "user"})["ok"]
    r = server.printer_materials("frob one")
    assert r["ok"] and r["counts"]["ok"] == 0 and r["counts"]["unknown"] > 0 and r["counts"]["no"] == 0
    assert r["printer"]["origin"] == "user"
    sample = r["groups"]["unknown"][0]["reasons"]
    assert any("states no maximum" in x["text"] or "does not state" in x["text"] or "states no" in x["text"] for x in sample)


def test_the_reply_of_printer_materials_is_not_longer_than_a_tool_reply_should_be(work):
    assert len(json.dumps(server.printer_materials("Bambu A1 mini"))) < 60_000


# --- check_problem -----------------------------------------------------------------------------------------------------------

def test_check_problem_has_a_material_key_only_with_a_filament(work):
    plain = server.build_regions({**tiny_spec(), "name": "plain", "print": {"printer_model": "bambu a1 mini"}})
    assert plain["ok"] and "material" not in server.check_problem(plain["yaml_rel"])
    with_filament = server.build_regions({**tiny_spec(material={"filament": ABS}), "name": "abs", "print": {"printer_model": "bambu a1 mini"}})
    assert with_filament["ok"], with_filament
    m = server.check_problem(with_filament["yaml_rel"])["material"]
    assert m["filament"] == ABS and m["printer"] == "bambu-lab-a1-mini" and m["compat"]["state"] == "no"
    assert m["material"]["E"]["source"] == f"filament:{ABS}" and m["material"]["nu"]["source"] == "assumed"
    assert any(r["check"] == "enclosure" and r["state"] == "no" for r in m["compat"]["reasons"])
    assert m["profile_sha256"] and m["catalog_version"] == mc.load_catalog().version
    assert with_filament["check"]["material"] == m                                         # build_regions returns the same check
    enclosed = server.build_regions({**tiny_spec(material={"filament": ABS}), "name": "abs2", "print": {"printer_model": "bambu p1s"}})
    assert server.check_problem(enclosed["yaml_rel"])["material"]["compat"]["state"] == "ok"
    nopr = server.build_regions({**tiny_spec(material={"filament": ABS}), "name": "abs3"})
    m3 = server.check_problem(nopr["yaml_rel"])["material"]
    assert m3["printer"] is None and m3["compat"] is None                                    # no printer, no comparison: a fact, not an error


# --- over stdio --------------------------------------------------------------------------------------------------------------

def test_stdio_server_finds_none_saves_the_filament_and_uses_it_with_no_restart(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    work = tmp_path / "work"
    env = dict(os.environ, TOPOPT_WORKSPACE=str(work), PYTHONPATH=str(ROOT))
    env.pop("TOPOPT_DEFAULT_WORKSPACE", None)
    env.pop("TOPOPT_PYTHON", None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], env=env, cwd=str(tmp_path))
    out = {}

    def payload(res):
        data = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if data:
            return data.get("result", data) if set(data) == {"result"} else data
        return json.loads(res.content[0].text)

    async def go():
        with open(tmp_path / "server.stderr", "w") as errlog:
            async with stdio_client(params, errlog=errlog) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    out["tools"] = {t.name: t for t in (await session.list_tools()).tools}
                    out["none"] = payload(await session.call_tool("find_material", {"name": "Acme PLA Plus"}))
                    out["bad_argument"] = payload(await session.call_tool("save_material", {"filament": dict(ACME)}))
                    out["files_after_bad"] = sorted(p.name for p in tmp_path.rglob("*") if p.is_file() and p.name != "server.stderr")
                    out["refused"] = payload(await session.call_tool("save_material", {"profile": {**ACME, "id": ABS}}))
                    out["saved"] = payload(await session.call_tool("save_material", {"profile": dict(ACME)}))
                    out["found"] = payload(await session.call_tool("find_material", {"name": "acme pla plus"}))
                    out["pref"] = payload(await session.call_tool("save_preferences", {
                        "print": {"printer_model": "Bambu P1S"}, "material": {"filament": "Acme PLA Plus"}, "filaments": {"available": ["acme pla plus"]}}))
                    out["fit"] = payload(await session.call_tool("printer_materials", {}))
                    out["built"] = payload(await session.call_tool("build_regions", {"spec": tiny_spec()}))

    asyncio.run(asyncio.wait_for(go(), 600))
    assert set(NEW_TOOLS) <= set(out["tools"]) and out["tools"]["save_material"].input_schema["additionalProperties"] is False
    assert out["none"]["status"] == "none" and "save_material" in out["none"]["note"]
    bad = out["bad_argument"]
    assert bad["ok"] is False and "'filament'" in bad["error"] and "The argument names are: profile" in bad["error"]
    assert out["files_after_bad"] == []                                                         # a wrong argument name makes no file
    assert out["refused"]["ok"] is False and "Use another id" in out["refused"]["error"]
    assert out["saved"]["ok"] and out["saved"]["created"] and Path(out["saved"]["path"]) == work / USER_FILE
    assert out["found"]["status"] == "match" and out["found"]["origin"] == "user" and out["found"]["profile"]["E_xy_mpa"] == 2600
    assert out["pref"]["ok"] and out["pref"]["values"]["material"] == {"filament": "acme-pla-plus"}
    assert out["pref"]["values"]["filaments"] == {"available": ["acme-pla-plus"]}
    assert out["fit"]["ok"] and out["fit"]["printer"]["named_by"] == "preferences" and out["fit"]["available"] == [
        {"id": "acme-pla-plus", "name": "Acme PLA Plus", "state": "ok"}]
    built = out["built"]
    assert built["ok"], built
    assert yaml.safe_load((work / built["yaml_rel"]).read_text())["material"] == {"filament": "acme-pla-plus"}
    assert built["check"]["material"]["compat"]["state"] == "ok" and built["check"]["material"]["material"]["E"]["value"] == 2400.0
    assert len(built["preferences_facts"]) == 2 and "your preferences file" in built["preferences_facts"][0]


def test_the_documents_name_the_tools_and_the_size_of_the_catalog():
    catalog = mc.load_catalog()
    readme = " ".join((ROOT / "docs" / "reference" / "print-step-tet.md").read_text(encoding="utf-8").split())
    assert f"has {len(catalog.profiles)} filaments of six makers" in readme and len(catalog.makers) == 6
    clients = " ".join((ROOT / "docs" / "clients.md").read_text(encoding="utf-8").split())
    count = len(mcp_tools())
    longest = max(mcp_tools(), key=len)
    assert f"The tools: {count}, and the longest name has {len(longest)} characters (`{longest}`)" in clients
    assert f"topopt: {count} tools, longest name {len(longest)} characters (`{longest}`)" in clients
    mcp_readme = (ROOT / "topopt_mcp" / "README.md").read_text(encoding="utf-8")
    for name in NEW_TOOLS:
        assert f"`{name}(" in mcp_readme and f"`{name}`" in clients, name


def test_no_text_says_that_the_tool_has_no_material_table():
    """With the filament catalog the old sentence is false; the true part (no default allowable) stays."""
    files = [ROOT / "CONTRACTS.md", ROOT / "examples" / "hook" / "README.md", ROOT / "examples" / "hook" / "hook_stress.yaml",
             ROOT / "docs" / "reference" / "problem-and-run.md", ROOT / "docs" / "reference" / "verify.md",
             ROOT / "topopt" / "verify" / "fatigue.py", ROOT / "topopt_mcp" / "server.py"]
    for path in files:
        flat = " ".join(path.read_text(encoding="utf-8").split()).lower()
        assert "no material table" not in flat and "no material preset" not in flat, path
    readme = " ".join((ROOT / "examples" / "hook" / "README.md").read_text(encoding="utf-8").split())
    assert "The tool has no default allowable (decision D7)" in readme

def test_printers_module_is_not_changed_by_the_shared_name_matching():
    """The printer catalog keeps its constants after the name matching moved to `topopt.catalogs`."""
    assert printers.SUGGEST_CUTOFF == 0.6 and printers.SUGGEST_MAX == 5


# --- CONTRACTS.md and the skills ---------------------------------------------------------------------------------------------

def test_contracts_has_the_filament_section_and_updates_the_sections_that_it_changes():
    text = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    start = text.index("### 18.8 Filament profiles")
    section = text[start: text.index("\n## 19. ", start)]
    for word in ("topopt/data/materials.yaml", "materials_catalog.resolve", "matched_by: family", "printers.SUGGEST_CUTOFF", "material.filament",
                 "filament:<id>", "assumed", "profile_sha256", "FilamentUse", "topopt-filaments.yaml", "No shadowing", "save_user_filament",
                 "find_material", "save_material", "printer_materials", "READ_ONLY_TOOLS", "topopt.compat.check", "`care`", "`unknown`",
                 "AGPL-3.0", "no price", "lower", "workfolder.resolve", "material fit"):
        assert word in section, word
    for check in compat.CHECKS:
        assert f"`{check}`" in section, check
    assert "filament: polymaker-polylite-pla" in text[text.index("## 2. Problem YAML"): text.index("settings:\n  voxel_size")]
    prefs = text[text.index("### 10.2 "): text.index("## 11. ")]
    for word in ("`filament` (since PM-2)", "`filaments` (since PM-2)", "available", "print, material, quality, outputs, filaments"):
        assert word in prefs, word
    for name in NEW_TOOLS:
        assert f"`{name}(" in text[: text.index("### 10.1 ")], name
    ref = text[text.index("### 25.7 "): text.index("## 26. ")]
    assert "revised on 2026-10-24" in ref and "no default allowable" in ref and "topopt/data/materials.yaml" in ref
    assert "`filament` (added with PM-2" in text[: text.index("## 6. ")] or "`filament` (added with PM-2" in text[: text.index("### 18.8 ")]


def skill_section(path, heading):
    import re
    text = path.read_text(encoding="utf-8")
    start = text.index(heading)
    nxt = re.search(r"^#{2,3} ", text[start + len(heading):], re.M)
    return text[start: start + len(heading) + (nxt.start() if nxt else len(text))]


def test_the_skills_ask_for_the_filament_name_and_save_an_unlisted_filament_after_the_user_agrees():
    import re
    start = skill_section(ROOT / "skills" / "start-here" / "SKILL.md", "### The filaments")
    for word in ("find_material", "printer_materials", "save_material", "`match`", "`ambiguous`", "`none`", "filaments={\"available\"",
                 "After the user agrees", "source_url", "ask the user", "topopt-filaments.yaml", "material.filament"):
        assert word in start, word
    assert not re.search(r"\b(must|never|always)\b", start, re.I)
    part_skill = skill_section(ROOT / ".claude" / "skills" / "part-to-topopt" / "SKILL.md", "## The filament")
    for word in ("find_material", "save_material", "printer_materials", "`filament: <id>`", "`ambiguous`", "`none`", "After the user agrees".lower()):
        assert word in part_skill, word
    assert not re.search(r"\b(must|never|always)\b", part_skill, re.I)
    for name, source in (("start-here", ROOT / "skills" / "start-here" / "SKILL.md"),
                         ("part-to-topopt", ROOT / ".claude" / "skills" / "part-to-topopt" / "SKILL.md")):
        assert (ROOT / ".agents" / "skills" / name / "SKILL.md").read_text(encoding="utf-8") == source.read_text(encoding="utf-8"), name


# --- fix round 1 (MI-4): the texts that the bending modulus (L3) and the family density (L4) made false -----------------------------

def _flat(text):
    return " ".join(text.split())


def test_the_lever_card_of_material_filament_names_the_bending_modulus_and_the_family_density():
    card = next(c for c in levers_cards.CARDS if c.key == "material.filament")
    text = _flat(card.effect + " " + card.values)
    assert "bending modulus" in text and "median" in text and "family" in text
    assert "tensile modulus in the catalog is an error" not in text and "stays the default" not in text


def test_the_part_skill_says_when_a_filament_needs_e_in_the_block():
    section = _flat(skill_section(ROOT / ".claude" / "skills" / "part-to-topopt" / "SKILL.md", "## The filament"))
    assert "A filament with no tensile modulus needs `E` in the block" not in section
    assert "bending modulus" in section and "no modulus at all" in section
    assert "median" in section and "assumed" in section


def test_the_mcp_readme_says_what_a_run_needs_from_a_saved_filament():
    text = _flat((ROOT / "topopt_mcp" / "README.md").read_text(encoding="utf-8"))
    assert "A run needs a tensile modulus and a density from the filament or from the YAML" not in text
    assert "bending modulus" in text and "median density of its family" in text


def test_contracts_counts_the_plain_nulls_of_the_catalog_as_the_code_does():
    text = _flat((ROOT / "CONTRACTS.md").read_text(encoding="utf-8"))
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
    assert f"{words[len(mc.NULL_WITHOUT_NOTE)]} fields are plain nulls that need no note" in text
    start = text.index("fields are plain nulls that need no note:")
    line = text[start: text.index(". ", start)]
    assert all(f"`{name}`" in line for name in mc.NULL_WITHOUT_NOTE), line


def test_check_problem_keeps_the_words_of_main_that_pm2_had_cut_and_stays_within_the_limit():
    """Fix round 1 (N-5): PM-2 cut four things from the text of `check_problem` to make room for its own line. The old text stays.
    Since P9-11 the key lines of the reply are in the guide `topopt://replies/check_problem` that the tool text names, so the text and
    the guide are read together."""
    from topopt_mcp import replies
    text = " ".join((mcp_tools()["check_problem"].description + " " + replies.guide("check_problem")).split())
    for words in ("Read the facts and tell the user the numbers that matter", "it gives the options (iterations, or the schedule scaled to the passes) and the measured price",
                  "recommendation (the offer to install pypardiso on Linux x86_64, else null)", "`topopt check` prints it as one line"):
        assert words in text, words
    assert len(mcp_tools()["check_problem"].description) <= 4096


def test_find_material_says_what_a_range_density_and_a_missing_density_become():
    text = " ".join(mcp_tools()["find_material"].description.split())
    assert "middle" in text and "range" in text and "median" in text


def test_printer_materials_names_a_saved_printer_that_is_not_found_now(work):
    """Fix round 1 (N-7): the error must not say that the file has no `print.printer_model` when it has one that is lost."""
    assert server.save_printer({"id": "frob-1", "vendor": "Frob", "model": "Frob One", "printer_type": "fdm", "bed_mm": [200, 200], "source": "user"})["ok"]
    assert preferences.save(work, print={"printer_model": "frob one"})["ok"]
    assert server.printer_materials()["ok"]
    (work / printers.USER_FILE_NAME).unlink()                                           # the printer is not found now
    r = server.printer_materials()
    assert r["ok"] is False
    assert "has no print.printer_model" not in r["error"]
    assert "frob-1" in r["error"] and "not found now" in r["error"] and "find_printer" in r["error"] and "save_printer" in r["error"]
    assert "\n" not in r["error"]
    # the argument still wins, and a file with no printer at all keeps the old text
    assert server.printer_materials("Bambu P1S")["ok"]
    (work / preferences.FILE_NAME).unlink()
    assert "has no print.printer_model" in server.printer_materials()["error"]


def test_the_print_role_names_the_filament_tools_and_does_not_say_that_material_facts_are_missing():
    """Fix round 1 (N-10): printer and filament fit is a print task. Both copies of the role text say it."""
    for path in (ROOT / "topopt_mcp" / "roles" / "print.md", ROOT / "agents" / "print-engineer.md"):
        text = " ".join(path.read_text(encoding="utf-8").split())
        for word in ("`find_material`", "`printer_materials`", "`save_material`"):
            assert word in text, (path.name, word)
        assert "no process or material fact files" not in text, path.name
        assert "no process fact files" in text, path.name
        assert not re.search(r"\b(must|never|always)\b", text[text.index("`find_material`") - 200: text.index("`save_material`") + 300], re.I)
