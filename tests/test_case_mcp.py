"""Tests for record_brief, case_status and the soft gate of run_topopt. The fake run stands in for the optimizer."""
import asyncio
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import workfolder  # noqa: E402
from topopt_mcp import server  # noqa: E402
from test_optimize import box, make_case  # noqa: E402

FAKE_CMD = f"{sys.executable} {REPO}/tests/serve_after_fake.py {{out}} --spi 0.05 --iterations 10"
BRIEF = "A bracket for a 2 kg shelf. The load is 20 N down on the tip."
ASSUMPTIONS = [{"text": "load 20 N at the tip", "origin": "user"}, {"text": "PLA", "origin": "default"}]


@pytest.fixture(autouse=True)
def ws(monkeypatch, tmp_path):
    work = tmp_path / "ws"
    work.mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(work))
    monkeypatch.setenv("TOPOPT_MCP_CMD", FAKE_CMD)
    server._current.clear()
    yield work
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._current.clear()


@pytest.fixture
def part(ws):
    """The YAML and the STL files of a small problem in `runs/part/` of the work folder."""
    d = ws / "runs" / "part"
    d.mkdir(parents=True)
    make_case(d)
    return d


def wait_done(run_dir, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        st = server.run_status(str(run_dir))
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def log_events(part, stem="c"):
    return [json.loads(x) for x in (part / f"{stem}.case-log.jsonl").read_text().splitlines()]


def test_no_case_runs_as_before_and_says_so(ws, part):
    r = server.run_topopt("runs/part/c.yaml")
    assert isinstance(r["pid"], int) and r["run_dir"] == str(ws / "runs" / "live")
    assert r["brief_stated"] is False and r["brief_facts"] == ["no case for this YAML; record_brief makes one"]
    assert not (part / "c.case.json").exists() and not (part / "c.case-log.jsonl").exists()
    assert wait_done(ws / "runs" / "live")["state"] == "done"
    assert server.case_status("runs/part/c.yaml")["exists"] is False


def test_a_current_brief_passes_and_the_run_is_in_the_case(ws, part):
    r = server.record_brief("runs/part/c.yaml", BRIEF, ASSUMPTIONS)
    assert r["recorded"] is True and r["case"] == "runs/part/c.case.json"
    run = server.run_topopt("runs/part/c.yaml")
    assert run["brief_stated"] is True and isinstance(run["pid"], int)
    assert run["workspace"] == str(ws) and run["run_dir_rel"] == "runs/live"
    wait_done(ws / "runs" / "live")
    st = server.case_status("runs/part/c.yaml")
    assert st["exists"] is True and st["brief_stated"] is True and st["brief_stale"] is False
    assert st["brief"]["text"] == BRIEF and st["gate"] == "run_topopt starts"
    assert set(st["inputs"]["files"]) == {"design:design.stl", "fixed:plate.stl", "load:tip.stl"}
    (entry,) = st["runs"]
    assert entry["run_dir_rel"] == "runs/live" and entry["pid"] == run["pid"] and entry["brief"] == 1
    assert entry["state"] == "done" and entry["stale"] is False
    doc = json.loads((part / "c.case.json").read_text())
    assert doc["runs"][0]["run_dir"] == "../live"
    assert [e["event"] for e in log_events(part)] == ["brief_recorded", "run_started"]
    text = (part / "c.case.json").read_text() + (part / "c.case-log.jsonl").read_text()
    for needle in (str(ws), str(ws.parent), "/home/", "/Users/", "/tmp/"):
        assert needle not in text, needle


def test_a_stale_brief_stops_run_topopt_and_keeps_the_old_dashboard(ws, part):
    server.record_brief("runs/part/c.yaml", BRIEF, ASSUMPTIONS)
    first = server.run_topopt("runs/part/c.yaml")
    wait_done(ws / "runs" / "live")
    old_pid, old_proc = first["pid"], server._current["proc"]
    assert server._pid_alive(old_pid)                      # the old process serves the dashboard
    box(part / "design.stl", (0, 0, 0), (20, 10, 12))      # one region STL changes
    stopped = server.run_topopt("runs/part/c.yaml")
    assert stopped["state"] == "awaiting_user" and "error" not in stopped and "pid" not in stopped
    assert stopped["override"] == "proceed_without_brief=true"
    assert any("design.stl" in f and "stale" in f for f in stopped["facts"])
    assert any("record_brief" in o for o in stopped["options"])
    assert server._pid_alive(old_pid) and server._current["proc"] is old_proc   # the gate closed nothing
    st = server.case_status("runs/part/c.yaml")
    assert st["brief_stale"] is True and st["stale_files"] == ["design.stl"]
    assert st["runs"][0]["stale"] is True and st["runs"][0]["stale_files"] == ["design.stl"]
    assert st["gate"].startswith("run_topopt returns awaiting_user")
    # With the override the run starts. The log records it. The reply says that no current brief exists.
    again = server.run_topopt("runs/part/c.yaml", proceed_without_brief=True)
    assert isinstance(again["pid"], int) and again["pid"] != old_pid
    assert again["brief_stated"] is False and any("stale" in f for f in again["brief_facts"])
    wait_done(ws / "runs" / "live")
    events = [e["event"] for e in log_events(part)]
    assert events == ["brief_recorded", "run_started", "gate_returned", "override_used", "run_started"]
    st = server.case_status("runs/part/c.yaml")
    assert [r["brief"] for r in st["runs"]] == [None, 1]
    assert [r["state"] for r in st["runs"]] == ["done", "replaced"]   # one folder holds one run
    # A new brief makes the gate pass again.
    server.record_brief("runs/part/c.yaml", "Second brief.", ASSUMPTIONS)
    assert server.case_status("runs/part/c.yaml")["brief_stated"] is True
    third = server.run_topopt("runs/part/c.yaml")
    assert third["brief_stated"] is True and isinstance(third["pid"], int)


def edit_settings(part, **settings):
    """Change keys of `settings` in the YAML of the case."""
    import yaml as pyyaml
    path = part / "c.yaml"
    data = pyyaml.safe_load(path.read_text())
    data["settings"].update(settings)
    path.write_text(pyyaml.safe_dump(data))


def test_a_method_change_after_the_brief_does_not_stop_run_topopt_and_the_reply_names_it(ws, part):
    """P9-4b: the advice of `environment` (settings.solver) or of a lever card (settings.voxel_size) after record_brief."""
    edit_settings(part, solver="amg")
    server.record_brief("runs/part/c.yaml", BRIEF, ASSUMPTIONS)
    edit_settings(part, solver="gpu-mg")
    run = server.run_topopt("runs/part/c.yaml")
    assert "state" not in run and "error" not in run and isinstance(run["pid"], int)
    assert run["brief_stated"] is True
    assert any("settings.solver" in f and "`amg`" in f and "`gpu-mg`" in f for f in run["brief_facts"]), run["brief_facts"]
    wait_done(ws / "runs" / "live")
    started = [e for e in log_events(part) if e["event"] == "run_started"]
    assert len(started) == 1 and len(started[0]["method_facts"]) == 1                   # the case log keeps the report (review m1)
    assert "settings.solver" in started[0]["method_facts"][0] and started[0]["method_facts"] == [
        f for f in run["brief_facts"] if "settings.solver" in f]
    st = server.case_status("runs/part/c.yaml")
    assert st["brief_stated"] is True and st["brief_stale"] is False and st["gate"] == "run_topopt starts"
    assert any("settings.solver" in f for f in st["facts"])
    assert st["runs"][0]["stale"] is False            # the run started with the new content
    # A second method setting (voxel_size, the advice of a lever card) passes too, with its own fact.
    edit_settings(part, voxel_size=2.0)
    second = server.run_topopt("runs/part/c.yaml")
    assert "state" not in second and isinstance(second["pid"], int) and second["brief_stated"] is True
    assert any("settings.voxel_size" in f and "`1.0`" in f and "`2.0`" in f for f in second["brief_facts"])
    wait_done(ws / "runs" / "live")
    assert [e["event"] for e in log_events(part)] == ["brief_recorded", "run_started", "run_started"]


def test_a_requirement_change_after_the_brief_still_stops_run_topopt_and_names_the_key(ws, part):
    import yaml as pyyaml
    server.record_brief("runs/part/c.yaml", BRIEF, ASSUMPTIONS)
    edit_settings(part, solver="gpu-mg")                       # a method change next to it
    data = pyyaml.safe_load((part / "c.yaml").read_text())
    data["load_cases"][0]["forces"][0]["force"] = [0, 0, -9.0]
    (part / "c.yaml").write_text(pyyaml.safe_dump(data))
    stopped = server.run_topopt("runs/part/c.yaml")
    assert stopped["state"] == "awaiting_user" and "pid" not in stopped and "error" not in stopped
    assert any("load_cases[0].forces[0].force" in f and "requirement" in f for f in stopped["facts"]), stopped["facts"]
    assert any("settings.solver" in f for f in stopped["facts"])
    assert not server._current and not (ws / "runs" / "live").exists()
    assert [e["event"] for e in log_events(part)] == ["brief_recorded", "gate_returned"]


def test_a_case_with_no_brief_stops_run_topopt(ws, part):
    (part / "c.case.json").write_text(json.dumps({"format": 1, "yaml": "c.yaml", "brief": None, "runs": []}))
    r = server.run_topopt("runs/part/c.yaml")
    assert r["state"] == "awaiting_user" and any("no brief" in f for f in r["facts"])
    assert not server._current and not (ws / "runs" / "live").exists()
    assert not (ws / "runs" / "live.pid").exists() and not (ws / "runs" / "live.stdout").exists()
    assert [e["event"] for e in log_events(part)] == ["gate_returned"]


def test_a_yaml_outside_the_work_folder_runs_and_gets_no_case_file(tmp_path, ws):
    outside = tmp_path / "outside"
    outside.mkdir()
    make_case(outside)
    before = sorted(p.name for p in outside.iterdir())
    rec = server.record_brief(str(outside / "c.yaml"), BRIEF, ASSUMPTIONS)
    assert rec["recorded"] is False and rec["facts"] and rec["options"] and "error" not in rec
    st = server.case_status(str(outside / "c.yaml"))
    assert st["exists"] is False and st["facts"] and st["options"]
    r = server.run_topopt(str(outside / "c.yaml"), proceed_without_brief=True)
    assert isinstance(r["pid"], int) and r["brief_stated"] is False
    assert "outside the work folder" in r["brief_facts"][0]
    assert sorted(p.name for p in outside.iterdir()) == before
    wait_done(ws / "runs" / "live")


def test_the_code_folder_example_gets_no_case_file(ws, monkeypatch):
    """A relative path that only the code folder has (an example) is outside the work folder."""
    r = server.record_brief("examples/hook/hook.yaml", BRIEF, None)
    assert r["recorded"] is False and "outside the work folder" in r["facts"][0]
    assert not list((REPO / "examples" / "hook").glob("*.case*"))


def test_a_yaml_in_the_code_folder_is_recorded_with_a_fact_and_an_option(ws, monkeypatch):
    """A source checkout with no TOPOPT_WORKSPACE: the work folder is the code folder. The tool still records."""
    monkeypatch.setattr(server, "REPO", ws)
    d = ws / "examples" / "part"
    d.mkdir(parents=True)
    make_case(d)
    r = server.record_brief("examples/part/c.yaml", BRIEF, ASSUMPTIONS)
    assert r["recorded"] is True and (d / "c.case.json").exists()
    assert any("code folder" in f for f in r["facts"]) and any(workfolder.change_text() in o for o in r["options"])


def test_a_run_folder_outside_the_work_folder_leaves_no_folder_name(tmp_path_factory, ws, part):
    away = tmp_path_factory.mktemp("jdoe-runs")
    server.record_brief("runs/part/c.yaml", BRIEF, ASSUMPTIONS)
    r = server.run_topopt("runs/part/c.yaml", out_dir=str(away / "run1"))
    assert isinstance(r["pid"], int)
    wait_done(away / "run1")
    entry = json.loads((part / "c.case.json").read_text())["runs"][0]
    assert entry["run_dir"] is None and re.fullmatch(r"<outside>/[0-9a-f]{12}", entry["run_dir_name"])
    text = (part / "c.case.json").read_text() + (part / "c.case-log.jsonl").read_text()
    assert "jdoe-runs" not in text and str(away) not in text
    st = server.case_status("runs/part/c.yaml")
    assert st["runs"][0]["run_dir_rel"] is None and "jdoe-runs" not in json.dumps(st)


def test_run_topopt_with_resume_stops_at_the_gate_and_names_both_overrides(ws, part):
    server.record_brief("runs/part/c.yaml", BRIEF, ASSUMPTIONS)
    box(part / "tip.stl", (18, 4, 0), (20, 6, 4))
    r = server.run_topopt("runs/part/c.yaml", resume=True)
    assert r["state"] == "awaiting_user" and "error" not in r and "pid" not in r
    (both,) = [o for o in r["options"] if "allow_changed_inputs=true" in o]
    assert "proceed_without_brief=true" in both and "resume=true" in both
    assert not server._current and not (ws / "runs" / "live.pid").exists()          # the gate started nothing
    # With the override the gate passes; the resume itself has no checkpoint here and refuses, so no run entry is written.
    again = server.run_topopt("runs/part/c.yaml", resume=True, proceed_without_brief=True)
    assert "error" in again and "state" not in again
    assert [e["event"] for e in log_events(part)] == ["brief_recorded", "gate_returned"]
    assert json.loads((part / "c.case.json").read_text())["runs"] == []


def test_the_stub_yaml_works_in_case_mode(ws):
    stub = ws / "x.yaml"
    stub.write_text("name: x\nmaterial: {E: 3500}\n")     # `name` is a method key; `material` is a requirement
    assert server.record_brief("x.yaml", BRIEF, None)["recorded"] is True
    assert server.run_topopt("x.yaml")["brief_stated"] is True
    wait_done(ws / "runs" / "live")
    stub.write_text("name: x\nmaterial: {E: 3000}\n")
    r = server.run_topopt("x.yaml")
    assert r["state"] == "awaiting_user" and any("x.yaml" in f for f in r["facts"])


def test_a_yaml_that_is_not_there_gives_an_error(ws):
    assert "not found" in server.record_brief("nope.yaml", BRIEF, None)["error"]
    assert "not found" in server.case_status("nope.yaml")["error"]


def test_a_yaml_that_is_not_there_names_the_path_and_not_only_the_file_name(ws):
    """Q1-15: a wrong folder in `yaml_path` (`runs/ladle_built/ladle.yaml`; the file is `ladle_built.yaml`) must show in the error."""
    (ws / "runs" / "ladle_built").mkdir(parents=True)
    (ws / "runs" / "ladle_built" / "ladle_built.yaml").write_text("name: x\n")
    wrong = "runs/ladle_built/ladle.yaml"
    for reply in (server.record_brief(wrong, BRIEF, None), server.case_status(wrong)):
        error = reply["error"]
        assert error.startswith("YAML file not found: ") and str(ws / "runs" / "ladle_built" / "ladle.yaml") in error
        assert f"work folder: {ws}" in error
    assert "not found" in server.run_topopt(wrong)["error"] and str(ws / wrong) in server.run_topopt(wrong)["error"]


def test_the_brief_shape_error_is_a_reply(ws, part):
    r = server.record_brief("runs/part/c.yaml", BRIEF, [{"text": "x", "origin": "maybe"}])
    assert r["recorded"] is False and "origin" in r["error"]


def test_tool_texts_have_no_habitual_caps_and_name_the_override():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    for name in ("record_brief", "case_status", "run_topopt"):
        text = tools[name].description
        for word in ("must", "never", "always", "forbidden"):
            assert not re.search(rf"\b{word}\b", text.lower()), (name, word)
    assert "proceed_without_brief" in tools["run_topopt"].description
    assert "awaiting_user" in tools["run_topopt"].description
    for name in ("record_brief", "case_status", "run_topopt"):         # P9-4b: the text says that a method change is not a stop
        assert "method setting" in tools[name].description, name
    assert "proceed_without_brief" in tools["run_topopt"].input_schema["properties"]
    assert set(tools["record_brief"].input_schema["properties"]) == {"yaml_path", "text", "assumptions", "criteria"}
    assert tools["case_status"].input_schema["required"] == ["yaml_path"]


def test_the_tools_work_through_stdio(ws, part):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(REPO),
                                   env={"TOPOPT_MCP_CMD": FAKE_CMD, "TOPOPT_WORKSPACE": str(ws)})

    def payload(res):
        if getattr(res, "structured_content", None):
            sc = res.structured_content
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        return json.loads(res.content[0].text)

    async def go():
        async with Client(params) as client:
            rec = payload(await client.call_tool("record_brief", {
                "yaml_path": "runs/part/c.yaml", "text": BRIEF, "assumptions": ASSUMPTIONS,
                "criteria": [{"text": "stiffness at least 80 N/mm", "origin": "user"}]}))
            assert rec["recorded"] is True and rec["brief"]["criteria"][0]["origin"] == "user"
            st = payload(await client.call_tool("case_status", {"yaml_path": "runs/part/c.yaml"}))
            assert st["brief_stated"] is True and st["brief"]["assumptions"] == ASSUMPTIONS
            box(part / "tip.stl", (18, 4, 0), (20, 6, 4))
            stopped = payload(await client.call_tool("run_topopt", {"yaml_path": "runs/part/c.yaml"}))
            assert stopped["state"] == "awaiting_user" and stopped["override"] == "proceed_without_brief=true"
            bad = payload(await client.call_tool("record_brief", {
                "yaml_path": "runs/part/c.yaml", "text": BRIEF, "assumptions": [{"text": "x"}]}))
            assert bad["recorded"] is False

    asyncio.run(go())
    assert not (ws / "runs" / "live.pid").exists()      # the gate started nothing
