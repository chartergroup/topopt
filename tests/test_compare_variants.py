"""Tests for the table form of compare_runs (#209 step 2): `compare_runs(run_a, variants=[...])`.

The runs are recorded runs of `fake_run` (the real RunWriter, no optimizer) with a verify file, a print block and a montage PNG that the
tests write. The two-run form is pinned: its reply has the same keys as before.
"""
import asyncio
import inspect
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import feedback  # noqa: E402
from topopt_mcp import server  # noqa: E402
from test_feedback import fake_run, make_case, tree_hash  # noqa: E402
from test_mcp_strict_arguments import Recorder, call, reply_of  # noqa: E402

PAIR_KEYS = {"ok", "a", "b", "setup", "metrics", "inputs", "notes", "summary", "problems"}
TABLE_KEYS = {"ok", "base", "runs", "vs_base", "notes", "summary", "problems"}
ROW_KEYS = {"name", "dir_name", "run_dir", "state", "mass_final_g", "mass_saved_pct", "stiffness", "stiffness_source", "verdict",
            "verify", "print", "render", "problems"}


@pytest.fixture(autouse=True)
def ws(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    server._current.clear()
    yield tmp_path
    server._current.clear()


def write_metrics(run, **extra):
    path = run / "result" / "metrics.json"
    doc = json.loads(path.read_text())
    doc.update(extra)
    path.write_text(json.dumps(doc))


def png(path: Path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (200, 10, 10)).save(path)


def three_runs(root):
    """base (compliance 1.0), v1 (volume fraction 0.25, compliance 1.25, verify + print + montage), v2 (older: no verify, print, montage)."""
    base = fake_run(root / "runs" / "base", make_case(root / "in_0"), compliance=1.0)
    v1 = fake_run(root / "runs" / "v1", make_case(root / "in_1", vf=0.25), compliance=1.25)
    v2 = fake_run(root / "runs" / "v2", make_case(root / "in_2", vf=0.45), compliance=0.8)
    write_metrics(v1, mass_final_g=1.5, mass_saved_pct=70.0, verdict={"state": "converged", "text": "x"},
                  print={"ok": True, "support_needed": True,
                         "configured": {"bed_fit": True, "overhang_area_mm2": 120.5, "support_volume_mm3": 900.0, "extents_mm": [1, 2, 3]},
                         "thinnest_member": {"thinnest_member_mm": 1.8, "min_mm": 1.1}, "settings": {"printer_type": "fdm"}})
    write_metrics(base, print={"ok": False, "error": "no printer"})
    (v1 / "verify").mkdir()
    (v1 / "verify" / "verify.json").write_text(json.dumps({
        "ok": True, "estimate_only": False, "limit": {"allowable_MPa": 30.0, "percentile": 99.9, "exceeded": True},
        "cases": {"down": {"stiffness_N_per_mm": 19.5, "optimizer_stiffness_N_per_mm": 20.0, "von_mises": {"p99_9_by_volume_MPa": 41.25}}}}))
    (base / "verify").mkdir()
    (base / "verify" / "verify.json").write_text(json.dumps({"ok": True, "estimate_only": True}))
    png(v1 / "renders" / "bracket_montage.png")
    old = v1 / "renders" / "old_montage.png"          # an older montage of the same run: the newest one is picked
    png(old)
    past = time.time() - 7200
    os.utime(old, (past, past))
    return base, v1, v2


# ---- the two-run form is as before ------------------------------------------------------------------
def test_two_run_reply_has_the_same_keys_as_before(ws):
    base, v1, _ = three_runs(ws)
    lib = feedback.compare_runs(base, v1)
    assert set(lib) == PAIR_KEYS
    mcp = server.compare_runs("runs/base", "runs/v1")
    assert set(mcp) == PAIR_KEYS
    assert mcp["a"]["run_dir_rel"] == "runs/base" and mcp["b"]["run_dir_rel"] == "runs/v1"
    assert not {"variants", "runs", "vs_base", "base"} & set(mcp)
    assert not {"variants", "runs", "vs_base", "base"} & set(lib)
    assert set(server.compare_runs("runs/base", "runs/v1", check_labels=False)) == PAIR_KEYS


def test_parameter_order_and_schema_types_of_the_old_call():
    assert list(inspect.signature(server.compare_runs).parameters) == ["run_a", "run_b", "check_labels", "variants"]
    tool = {t.name: t for t in asyncio.run(server.mcp.list_tools())}["compare_runs"]
    props = tool.input_schema["properties"]
    assert list(props)[:3] == ["run_a", "run_b", "check_labels"]
    assert props["run_a"]["type"] == "string"
    assert {x.get("type") for x in props["run_b"]["anyOf"]} == {"string", "null"}
    assert {x.get("type") for x in props["check_labels"]["anyOf"]} == {"boolean", "null"}
    assert props["variants"]["anyOf"][0] == {"items": {"type": "string"}, "type": "array"}
    assert tool.input_schema["required"] == ["run_a"]
    assert tool.input_schema["additionalProperties"] is False


# ---- the table --------------------------------------------------------------------------------------
def test_table_of_three_recorded_runs(ws):
    base, v1, v2 = three_runs(ws)
    before = tree_hash(ws)
    r = server.compare_runs("runs/base", variants=["runs/v1", "runs/v2"])
    assert tree_hash(ws) == before                                    # the tool writes nothing
    assert r["ok"] and set(r) == TABLE_KEYS
    assert [e["run_dir_rel"] for e in r["vs_base"]] == ["runs/v1", "runs/v2"]
    assert [x["dir_name"] for x in r["runs"]] == ["base", "v1", "v2"]  # the base first, then the call order
    assert r["base"]["dir_name"] == "base" and r["base"]["run_dir_rel"] == "runs/base"
    assert [x["run_dir_rel"] for x in r["runs"]] == ["runs/base", "runs/v1", "runs/v2"]
    assert all(set(x) >= ROW_KEYS and "earlier_run" not in x for x in r["runs"])
    b, one, two = r["runs"]
    assert one["mass_final_g"] == 1.5 and one["mass_saved_pct"] == 70.0 and b["mass_final_g"] == 2.0
    assert b["stiffness"] == {"down": 25.0} and one["stiffness"] == {"down": pytest.approx(20.0)} and two["stiffness"] == {"down": pytest.approx(31.25)}
    assert "optimizer" in one["stiffness_source"] and "result/metrics.json" in one["stiffness_source"]
    assert one["state"] == "done" and one["verdict"] == "converged" and b["verdict"] is None
    # verify: done / no result / not run
    assert one["verify"]["status"] == "done" and one["verify"]["limit_exceeded"] is True and one["verify"]["same_mesh"] is None
    assert one["verify"]["cases"] == {"down": {"stiffness_N_per_mm": 19.5, "p99_9_MPa": 41.25}}
    assert b["verify"] == {"status": "no result"} and two["verify"] == {"status": "not run"}
    # print: short keys, an error, or null with a problem
    assert one["print"] == {"ok": True, "support_needed": True, "bed_fit": True, "overhang_area_mm2": 120.5,
                            "support_volume_mm3": 900.0, "thinnest_member_mm": 1.8}
    assert b["print"] == {"ok": False, "error": "no printer"}
    assert two["print"] is None and any("print is not in result/metrics.json" in p for p in two["problems"])
    assert b["verdict"] is None and any("verdict" in p for p in b["problems"])
    # render: the path by the run folder, or null and the pointer to render_part
    assert one["render"] == {"montage_rel": "renders/bracket_montage.png", "newer_than_final": True, "section": False}
    assert b["render"] is None and two["render"] is None
    assert any("render_part with run_dir=" in p and str(v2) in p for p in two["problems"])
    assert r["problems"] == [] and all("v2: " not in p for p in r["problems"])   # a row problem is in its row only
    # vs_base: one per variant, built by the function of the two-run form
    assert len(r["vs_base"]) == 2
    for entry, other in zip(r["vs_base"], (v1, v2)):
        pair = feedback.compare_runs(base, other)
        assert entry["ok"] and entry["run_dir"] == str(other)
        assert entry["setup"] == pair["setup"] and entry["summary"] == pair["summary"] and entry["notes"] == pair["notes"]
        assert set(entry) == {"ok", "run_dir", "run_dir_rel", "setup", "labels_vs_disk", "summary", "notes"}
        assert {x["key"] for x in entry["setup"]["run_json_diff"]} >= {"volume_fraction"}
    assert any("v1" in s and "mass_final_g 2 -> 1.5 (-25.0 %)" in s and "stiffness down 25 -> 20 (-20.0 %)" in s for s in r["summary"])
    assert len(r["summary"]) == 2
    json.dumps(r)


def test_library_table_has_no_workfolder_paths_and_keeps_the_order(ws):
    base, v1, v2 = three_runs(ws)
    r = feedback.compare_runs(base, variants=[str(v2), str(v1)])
    assert r["ok"] and [x["dir_name"] for x in r["runs"]] == ["base", "v2", "v1"]
    assert "run_dir_rel" not in r["runs"][0]


def test_an_old_render_is_marked(ws):
    base, v1, _ = three_runs(ws)
    past = time.time() - 3600
    os.utime(v1 / "renders" / "bracket_montage.png", (past, past))
    r = feedback.compare_runs(base, variants=[str(v1)])
    assert r["runs"][1]["render"]["newer_than_final"] is False


def test_a_refused_run_keeps_the_earlier_run_mark(ws, monkeypatch):
    base, v1, v2 = three_runs(ws)
    monkeypatch.setattr(feedback, "_refused", lambda f, d: d.name == "v2")
    r = feedback.compare_runs(base, variants=[str(v1), str(v2)])
    row = r["runs"][2]
    assert "earlier_run" in row and row["earlier_run"]["mass_final_g"] == 2.0 and row["mass_final_g"] is None
    assert "earlier_run" not in r["runs"][1]
    assert any("v2" in n and "earlier run" in n for n in r["notes"])


def test_a_missing_file_is_null_and_a_problem_not_a_crash(ws):
    base, v1, v2 = three_runs(ws)
    (v2 / "result" / "metrics.json").unlink()
    (v1 / "verify" / "verify.json").write_text("{not json")
    r = feedback.compare_runs(base, variants=[str(v1), str(v2)])
    assert r["ok"]
    two = r["runs"][2]
    assert two["mass_final_g"] is None and two["stiffness"] is None and two["print"] is None
    assert any("mass_final_g is not in" in p for p in two["problems"])
    assert r["runs"][1]["verify"] == {"status": "no result"}
    assert any("verify/verify.json is not valid JSON" in p for p in r["runs"][1]["problems"])


# ---- the refusals -----------------------------------------------------------------------------------
def refusals(ws):
    base, v1, v2 = three_runs(ws)
    return [
        ("run_b and variants", dict(run_b="runs/v1", variants=["runs/v2"])),
        ("neither", dict()),
        ("empty list", dict(variants=[])),
        ("a string", dict(variants="runs/v1")),
        ("a non-string entry", dict(variants=["runs/v1", 3])),
        ("an empty entry", dict(variants=["runs/v1", ""])),
        ("the same entry twice", dict(variants=["runs/v1", "runs/v1"])),
        ("the same folder written two ways", dict(variants=["runs/v1", "runs/../runs/v1/"])),
        ("the base as a variant", dict(variants=["runs/base"])),
        ("a missing folder", dict(variants=["runs/v1", "runs/zzz"])),
    ]


def test_every_refusal_names_the_forms_and_changes_nothing(ws):
    cases = refusals(ws)
    before = tree_hash(ws)
    for label, kw in cases:
        r = server.compare_runs("runs/base", **kw)
        assert r["ok"] is False and set(r) == {"ok", "error"}, label
        assert "compare_runs did nothing" in r["error"], (label, r["error"])
        if label in ("run_b and variants", "neither", "empty list", "a string", "a non-string entry", "an empty entry"):
            assert "(run_a, run_b)" in r["error"] and "variants=[" in r["error"], (label, r["error"])   # the error names both forms
    assert tree_hash(ws) == before
    miss = server.compare_runs("runs/base", variants=["runs/zzz"])
    assert "run folder not found" in miss["error"] and "zzz" in miss["error"]
    dup = server.compare_runs("runs/base", variants=["runs/v1", "runs/v1"])
    assert "named twice" in dup["error"] and "v1" in dup["error"]
    assert server.compare_runs("runs/zzz", variants=["runs/v1"])["ok"] is False


def test_the_library_refuses_the_same_way(ws):
    base, v1, _ = three_runs(ws)
    for kw in (dict(), dict(variants=[]), dict(variants=str(v1)), dict(run_b=str(v1), variants=[str(v1)]), dict(variants=[str(v1), str(v1)])):
        r = feedback.compare_runs(base, **kw)
        assert r["ok"] is False and "compare_runs did nothing" in r["error"], kw
    assert feedback.compare_runs(base, run_b=str(ws / "nope"))["ok"] is False


def test_the_two_run_form_still_accepts_the_same_run_twice(ws):
    base, _, _ = three_runs(ws)
    assert feedback.compare_runs(base, base)["ok"] is True


# ---- the strict check -------------------------------------------------------------------------------
def test_a_misspelled_variants_is_refused_with_the_names_and_the_tool_does_not_run(ws, monkeypatch):
    three_runs(ws)
    recorder = Recorder()
    monkeypatch.setattr(server.mcp._tool_manager.get_tool("compare_runs"), "fn", recorder)
    reply = reply_of(call(server.mcp, "compare_runs", {"run_a": "runs/base", "varients": ["runs/v1"]}))
    assert reply["ok"] is False and "no argument 'varients'" in reply["error"]
    assert "The argument names are: run_a, run_b, check_labels, variants." in reply["error"]
    assert recorder.calls == []
    # the right name passes the check and runs
    assert reply_of(call(server.mcp, "compare_runs", {"run_a": "runs/base", "variants": ["runs/v1"]}))["ok"] is True
    assert recorder.calls == [{"run_a": "runs/base", "run_b": None, "check_labels": None, "variants": ["runs/v1"]}]


# ---- texts, manifest and generated files ------------------------------------------------------------
def listed():
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_the_description_names_both_forms_and_keeps_the_pinned_phrases():
    import re
    text = listed()["compare_runs"].description
    assert 100 < len(text) < 2048 and "writes nothing" in text and "variants" in text and "run_b" in text
    assert "the files on disk can" in text[:2048]
    assert not re.search(r"\b(must|never|always|forbidden)\b", text, re.I)
    assert server.TOOL_SURFACE["compare_runs"].title == "Compare runs"
    assert len(listed()) == 32 and "variants" not in server.INSTRUCTIONS


def test_the_tool_stays_read_only_and_returns_a_dict():
    surface = server.TOOL_SURFACE["compare_runs"]
    assert surface.read_only is True and surface.destructive is False and surface.idempotent is True
    assert inspect.signature(server.compare_runs).return_annotation in (dict, "dict")


def test_the_generated_tool_lists_carry_the_new_argument():
    tools = json.loads((REPO / "evals" / "mocks" / "topopt" / "_tools.json").read_text())["tools"]
    entry = next(t for t in tools if t["name"] == "compare_runs")
    assert entry["inputSchema"]["required"] == ["run_a"] and "variants" in entry["inputSchema"]["properties"]
    assert entry["description"] == listed()["compare_runs"].description
    for name in ("legacy.tools.json", "modern.tools.json"):
        doc = (REPO / "topopt_mcp" / "launcher" / name).read_text()
        assert '"variants"' in doc and "Compare runs" in doc and "Compare two runs" not in doc


def test_contracts_readme_and_the_lever_card_name_the_table_form():
    contracts = (REPO / "CONTRACTS.md").read_text()
    assert "### 21.2 `compare_runs(run_a, run_b=None, check_labels=None, variants=None)`" in contracts
    flat = " ".join(contracts.split())
    assert "**Table form (#209).**" in contracts and "`compare_runs(run_a, run_b=None, check_labels=None, variants=None)`" in flat
    assert "compare_runs(run_a, run_b=None, check_labels=None, variants=None)" in (REPO / "topopt_mcp" / "README.md").read_text()
    from topopt.levers_cards import CARDS
    card = next(c for c in CARDS if c.key == "compare_runs" and c.group == "tool")
    assert "variants" in card.values and "variants" in card.call and "Table form" in card.cost and "host load average" in card.cost


# ---- the review fixes of #209 step 2 ----------------------------------------------------------------
def test_through_mcp_a_string_variants_and_a_non_string_entry_get_the_schema_error(tmp_path):
    """A client call is refused by the SDK check (is_error true) before the tool runs; the library gives {"ok": false} (test above)."""
    from test_mcp_strict_arguments import run_over_stdio
    results, work = run_over_stdio(tmp_path, [
        ("compare_runs", {"run_a": "runs/a", "variants": "runs/b"}),
        ("compare_runs", {"run_a": "runs/a", "variants": ["runs/b", 3]}),
        ("compare_runs", {"run_a": "runs/a", "variants": '["runs/b"]'}),
    ])
    assert results[0].is_error and "variants" in results[0].content[0].text and "valid list" in results[0].content[0].text
    assert results[1].is_error and "variants.1" in results[1].content[0].text
    assert not results[2].is_error and json.loads(results[2].content[0].text)["ok"] is False   # a JSON list text is read as a list: the folder is missing
    assert "not found" in json.loads(results[2].content[0].text)["error"]
    assert list(work.iterdir()) == []


@pytest.mark.parametrize("kw", [dict(run_b=""), dict(run_b="  "), dict(variants=[" "]), dict(variants=["runs/v1", ""])])
def test_a_blank_folder_is_refused_with_the_forms_and_no_folder_is_read(ws, monkeypatch, kw):
    three_runs(ws)
    read = []
    monkeypatch.setattr(feedback, "run_facts", lambda d: read.append(d))
    monkeypatch.setattr(feedback, "_compare_runs", lambda *a, **k: read.append(a))
    for r in (server.compare_runs("runs/base", **kw), feedback.compare_runs(str(ws / "runs" / "base"), **kw)):
        assert r["ok"] is False and "compare_runs did nothing" in r["error"] and "(run_a, run_b)" in r["error"], r
    blank_a = server.compare_runs(" ", run_b="runs/v1")
    assert blank_a["ok"] is False and "(run_a, run_b)" in blank_a["error"] and feedback.compare_runs("", "x")["ok"] is False
    assert read == []


def test_a_check_of_another_final_stl_is_a_problem(ws):
    import hashlib
    base, v1, _ = three_runs(ws)
    doc = json.loads((v1 / "verify" / "verify.json").read_text())
    doc["run"] = {"verified_file": "final.stl", "final_stl_sha256": hashlib.sha256((v1 / "result" / "final.stl").read_bytes()).hexdigest()}
    (v1 / "verify" / "verify.json").write_text(json.dumps(doc))
    row = feedback.compare_runs(base, variants=[str(v1)])["runs"][1]
    assert row["verify"]["same_mesh"] is True and not any("NOT of the mesh" in p for p in row["problems"])
    doc["run"]["final_stl_sha256"] = "0" * 64
    (v1 / "verify" / "verify.json").write_text(json.dumps(doc))
    row = feedback.compare_runs(base, variants=[str(v1)])["runs"][1]
    assert row["verify"]["status"] == "done" and row["verify"]["same_mesh"] is False
    assert any("NOT of the mesh now in result/final.stl" in p and "run verify again" in p for p in row["problems"])


def test_a_stiffness_with_no_value_keeps_its_reason(ws):
    base, v1, _ = three_runs(ws)
    write_metrics(v1, stiffness_cases=[{"name": "down", "N_per_mm": 20.0}, {"name": "twist", "N_per_mm": None, "reason": "the case has a moment"}])
    row = feedback.compare_runs(base, variants=[str(v1)])["runs"][1]
    assert row["stiffness"] == {"down": 20.0, "twist": None}
    assert any("stiffness twist: no value" in p and "the case has a moment" in p for p in row["problems"])
    write_metrics(v1, stiffness_cases=None)
    row = feedback.compare_runs(base, variants=[str(v1)])["runs"][1]
    assert row["stiffness"] is None and row["stiffness_source"] is None


def test_the_newest_whole_montage_is_picked_and_a_section_view_is_marked(ws):
    base, v1, _ = three_runs(ws)
    row = feedback.compare_runs(base, variants=[str(v1)])["runs"][1]
    assert row["render"]["montage_rel"] == "renders/bracket_montage.png"       # not old_montage.png, which is older
    png(v1 / "renders" / "bracket_section-x10_montage.png")                    # the newest file, but a cut view
    row = feedback.compare_runs(base, variants=[str(v1)])["runs"][1]
    assert row["render"] == {"montage_rel": "renders/bracket_montage.png", "newer_than_final": True, "section": False}
    for f in ("bracket_montage.png", "old_montage.png"):
        (v1 / "renders" / f).unlink()
    row = feedback.compare_runs(base, variants=[str(v1)])["runs"][1]
    assert row["render"]["montage_rel"] == "renders/bracket_section-x10_montage.png" and row["render"]["section"] is True


def test_check_labels_is_passed_to_each_pair(ws):
    base, v1, v2 = three_runs(ws)
    on = feedback.compare_runs(base, variants=[str(v1), str(v2)], check_labels=True)
    off = feedback.compare_runs(base, variants=[str(v1), str(v2)], check_labels=False)
    for r, checked in ((on, True), (off, False)):
        for e in r["vs_base"]:
            assert e["labels_vs_disk"]["base"]["checked"] is checked and e["labels_vs_disk"]["variant"]["checked"] is checked


def test_a_refused_row_moves_the_files_of_the_earlier_run_too(ws, monkeypatch):
    base, v1, _ = three_runs(ws)
    monkeypatch.setattr(feedback, "_refused", lambda f, d: d.name == "v1")
    row = feedback.compare_runs(base, variants=[str(v1)])["runs"][1]
    assert row["render"] is None and row["verify"] is None and row["earlier_run"]["render"]["montage_rel"] == "renders/bracket_montage.png"
    assert row["earlier_run"]["verify"]["status"] == "done"
