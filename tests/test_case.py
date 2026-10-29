"""Tests for topopt/case.py: the case file, the brief, the stale facts, the log and the gate (CONTRACTS.md section 30)."""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
import yaml as pyyaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import case, runinfo, workfolder  # noqa: E402
from test_optimize import box, make_case  # noqa: E402

BRIEF = "A bracket for a 2 kg shelf. The load is 20 N down on the tip."
ASSUMPTIONS = [{"text": "load 20 N at the tip", "origin": "user"}, {"text": "PLA, E 3500 MPa", "origin": "default"}]


@pytest.fixture
def work(tmp_path):
    """A work folder with one case folder `runs/part/` (the YAML and its STL files, as build_regions writes them)."""
    d = tmp_path / "runs" / "part"
    d.mkdir(parents=True)
    make_case(d)
    return tmp_path


@pytest.fixture
def yml(work):
    return (work / "runs" / "part" / "c.yaml").resolve()


def paths_in(*files):
    """The text of each file: used to look for an absolute path or a user name."""
    return "".join(Path(f).read_text() for f in files if Path(f).exists())


def assert_no_absolute_path(text, tmp_path):
    assert str(tmp_path) not in text
    for needle in ("/home/", "/Users/", "/tmp/", "C:\\Users"):
        assert needle not in text, needle


def record(yml, work, **kw):
    kw.setdefault("text", BRIEF)
    kw.setdefault("assumptions", ASSUMPTIONS)
    return case.record_brief(yml, work, kw.pop("text"), kw.pop("assumptions"), kw.pop("criteria", None))


# --- the hashes ---------------------------------------------------------------------------------

def test_inputs_use_the_one_hash_helper_and_the_one_key_rule(work, yml):
    now = case.current_inputs(yml, work)
    assert case.sha256_file is runinfo.sha256_file and case.region_files is runinfo.region_files
    assert now.yaml_sha256 == runinfo.sha256_file(yml)
    assert set(now.files) == {"design:design.stl", "fixed:plate.stl", "load:tip.stl"}
    assert now.files["design:design.stl"] == runinfo.sha256_file(yml.parent / "design.stl")
    assert now.assumptions == [] and now.facts == []
    assert case.current_inputs(yml, work).record() == now.record()      # the same files give the same record


def test_files_in_other_folders_get_other_keys(work, yml):
    d = yml.parent
    (d / "sub").mkdir()
    box(d / "sub" / "plate.stl", (-3, 0, 0), (0, 10, 10))
    data = pyyaml.safe_load(yml.read_text())
    data["regions"]["fixed"] = ["plate.stl", "sub/plate.stl"]
    yml.write_text(pyyaml.safe_dump(data))
    assert {"fixed:plate.stl", "fixed:sub/plate.stl"} <= set(case.current_inputs(yml, work).files)


# --- record_brief --------------------------------------------------------------------------------

def test_record_brief_makes_the_case_with_relative_paths_only(work, yml):
    r = record(yml, work, criteria=[{"text": "stiffness at least 80 N/mm", "origin": "user"}])
    assert r["recorded"] is True and r["case"] == "runs/part/c.case.json" and r["replaced"] is None
    manifest = yml.parent / "c.case.json"
    doc = json.loads(manifest.read_text())
    assert doc["format"] == 1 and doc["yaml"] == "c.yaml" and doc["runs"] == []
    b = doc["brief"]
    assert b["id"] == 1 and b["mode"] == "state" and b["text"] == BRIEF
    assert b["assumptions"] == ASSUMPTIONS and b["criteria"][0]["origin"] == "user"
    assert b["inputs"]["files"]["design:design.stl"] == runinfo.sha256_file(yml.parent / "design.stl")
    log = [json.loads(x) for x in (yml.parent / "c.case-log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log] == ["brief_recorded"] and log[0]["brief"]["id"] == 1
    assert_no_absolute_path(paths_in(manifest, yml.parent / "c.case-log.jsonl"), work)


def test_record_brief_replaces_the_brief_and_counts(work, yml):
    record(yml, work)
    r = record(yml, work, text="Second brief.")
    assert r["replaced"] == 1 and r["brief"]["id"] == 2
    doc = json.loads((yml.parent / "c.case.json").read_text())
    assert doc["brief"]["text"] == "Second brief." and doc["brief_count"] == 2
    events = [json.loads(x) for x in (yml.parent / "c.case-log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["brief_recorded", "brief_recorded"]


@pytest.mark.parametrize("kw, word", [
    ({"text": "  "}, "text"),
    ({"assumptions": [{"text": "x"}]}, "origin"),
    ({"assumptions": [{"text": "x", "origin": "guess"}]}, "origin"),
    ({"assumptions": [{"text": " ", "origin": "user"}]}, "text"),
    ({"assumptions": ["a string"]}, "assumptions[0]"),
    ({"criteria": [{"origin": "user"}]}, "criteria[0]"),
])
def test_record_brief_checks_the_shape(work, yml, kw, word):
    r = record(yml, work, **kw)
    assert r["recorded"] is False and word in r["error"]
    assert not (yml.parent / "c.case.json").exists()


def test_a_yaml_outside_the_work_folder_gets_no_case_file(tmp_path, yml):
    other = tmp_path / "elsewhere"
    other.mkdir()
    r = case.record_brief(yml, other, BRIEF, ASSUMPTIONS)
    assert r["recorded"] is False and r["case"] is None and r["facts"] and r["options"]
    assert not (yml.parent / "c.case.json").exists() and not (yml.parent / "c.case-log.jsonl").exists()
    st = case.status(yml, other)
    assert st["exists"] is False and st["facts"] == r["facts"] and st["runs"] == []


def test_two_yamls_in_one_folder_have_two_cases(work, yml):
    d = yml.parent
    (d / "d.yaml").write_text(yml.read_text())
    record(yml, work)
    assert case.status(d / "d.yaml", work)["exists"] is False
    case.record_brief(d / "d.yaml", work, "Other.", None)
    assert (d / "c.case.json").exists() and (d / "d.case.json").exists()
    assert json.loads((d / "c.case.json").read_text())["brief"]["text"] == BRIEF


def test_the_stub_yaml_has_a_hash_and_a_fact(work):
    stub = work / "x.yaml"
    stub.write_text("name: x\nmaterial: {E: 3500}\n")      # `name` is a method key; `material` is a requirement
    r = case.record_brief(stub, work, BRIEF, None)
    assert r["recorded"] is True and r["brief"]["inputs"]["files"] == {}
    assert r["brief"]["inputs"]["yaml"] == runinfo.sha256_file(stub)
    assert any("does not load" in f for f in r["facts"])
    assert case.status(stub, work)["brief_stated"] is True
    stub.write_text("name: x\nmaterial: {E: 3000}\n")
    st = case.status(stub, work)
    assert st["brief_stale"] is True and st["stale_files"] == ["x.yaml"]


# --- stale facts ---------------------------------------------------------------------------------

def test_a_changed_stl_makes_the_brief_and_the_runs_stale(work, yml):
    record(yml, work)
    g = case.gate(yml, work)
    case.run_started(g, yml, work, work / "runs" / "live", 4242)
    st = case.status(yml, work)
    assert st["brief_stated"] is True and st["brief_stale"] is False and st["stale_files"] == []
    assert st["runs"][0]["stale"] is False
    box(yml.parent / "design.stl", (0, 0, 0), (20, 10, 12))
    st = case.status(yml, work)
    assert st["brief_stated"] is False and st["brief_stale"] is True
    assert st["stale_files"] == ["design.stl"]
    assert any("design.stl" in f and "stale" in f for f in st["facts"])
    run = st["runs"][0]
    assert run["stale"] is True and run["stale_files"] == ["design.stl"]
    assert any("design.stl" in f and "changed" in f for f in run["facts"])
    assert "awaiting_user" in st["gate"]


def test_a_changed_yaml_content_is_stale_and_a_comment_is_not(work, yml):
    record(yml, work)
    yml.write_text("# a comment\n" + yml.read_text())
    st = case.status(yml, work)
    assert st["brief_stale"] is False and st["brief_stated"] is True
    assert any("comment" in f and "not stale" in f for f in st["facts"])
    data = pyyaml.safe_load(yml.read_text())
    data["settings"]["volume_fraction"] = 0.4          # a `both` key counts as a requirement (a method key: see the P9-4b tests)
    yml.write_text(pyyaml.safe_dump(data))
    st = case.status(yml, work)
    assert st["brief_stale"] is True and st["stale_files"] == ["c.yaml"]


def test_a_missing_stl_is_a_fact_and_not_an_exception(work, yml):
    record(yml, work)
    (yml.parent / "plate.stl").unlink()
    st = case.status(yml, work)       # the YAML does not load now: the files of the brief are still checked
    assert st["brief_stale"] is True and st["stale_files"] == ["plate.stl"]
    assert st["inputs"]["files"]["fixed:plate.stl"] == "missing"
    assert any("does not load" in f for f in st["facts"]) and any("missing" in f for f in st["facts"])
    g = case.gate(yml, work)
    assert g.stop["state"] == "awaiting_user" and any("plate.stl" in f for f in g.stop["facts"])
    assert_no_absolute_path(json.dumps(st["facts"]), work)


def test_a_new_region_file_is_stale(work, yml):
    record(yml, work)
    box(yml.parent / "keep.stl", (5, 0, 0), (8, 10, 10))
    data = pyyaml.safe_load(yml.read_text())
    data["regions"]["keep_out"] = ["keep.stl"]
    yml.write_text(pyyaml.safe_dump(data))
    st = case.status(yml, work)
    assert st["brief_stale"] is True and "keep.stl" in st["stale_files"]
    assert any("keep.stl" in f and "new file" in f for f in st["facts"])


def test_the_brief_and_the_yaml_assumptions_are_compared(work, yml):
    r = record(yml, work)
    assert any("not in the `assumptions` list of the YAML" in f for f in r["facts"])
    data = pyyaml.safe_load(yml.read_text())
    data["assumptions"] = [a["text"] for a in ASSUMPTIONS]
    yml.write_text(pyyaml.safe_dump(data))
    r = record(yml, work)
    assert not any("assumption" in f for f in r["facts"])


# --- atomic write and the log ----------------------------------------------------------------------

def test_an_interrupted_write_leaves_the_last_valid_case_file(work, yml, monkeypatch):
    record(yml, work, text="First.")
    manifest = yml.parent / "c.case.json"
    before = manifest.read_bytes()

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    r = record(yml, work, text="Second.")
    monkeypatch.undo()
    assert r["recorded"] is False and "disk full" in r["error"]
    assert manifest.read_bytes() == before and json.loads(before)["brief"]["text"] == "First."
    assert not (yml.parent / "c.case.json.tmp").exists()
    assert case.status(yml, work)["brief"]["text"] == "First."


def test_a_left_over_tmp_file_is_ignored(work, yml):
    record(yml, work)
    (yml.parent / "c.case.json.tmp").write_text('{"format": 1, "brief": {"id": 9')   # a killed write
    assert case.status(yml, work)["brief"]["id"] == 1
    record(yml, work, text="Next.")
    assert json.loads((yml.parent / "c.case.json").read_text())["brief"]["id"] == 2


def test_a_case_file_that_cannot_be_read_is_moved_aside(work, yml):
    (yml.parent / "c.case.json").write_text("not json {")
    st = case.status(yml, work)
    assert st["exists"] is False and any("cannot be read" in f for f in st["facts"])
    assert case.gate(yml, work).stop["state"] == "awaiting_user"
    r = record(yml, work)
    assert r["recorded"] is True and any("moved to c.case.json.bad1" in f for f in r["facts"])
    assert (yml.parent / "c.case.json.bad1").read_text() == "not json {"
    assert case.status(yml, work)["brief_stated"] is True


def test_the_log_has_one_whole_line_for_each_event(work, yml):
    case.log_event(yml, "one", a=1)
    case.log_event(yml, "two", b=[1, 2])
    lines = (yml.parent / "c.case-log.jsonl").read_text().splitlines()
    assert [json.loads(x)["event"] for x in lines] == ["one", "two"]
    assert all(re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$", json.loads(x)["ts"]) for x in lines)


def test_a_partial_last_log_line_is_closed_before_the_next_event(work, yml):
    log = yml.parent / "c.case-log.jsonl"
    log.write_text('{"ts": "t", "event": "one"}\n{"ts": "t", "event": "tw')   # a write that was cut
    case.log_event(yml, "three")
    rows, bad = [], 0
    for line in log.read_text().splitlines():
        try:
            rows.append(json.loads(line)["event"])
        except ValueError:
            bad += 1
    assert rows == ["one", "three"] and bad == 1


# --- run entries -----------------------------------------------------------------------------------

def test_a_run_entry_has_relative_paths_and_the_inputs_at_the_start(work, yml):
    record(yml, work)
    g = case.gate(yml, work)
    assert g.stop is None and g.reply["brief_stated"] is True and g.brief_id == 1
    out = case.run_started(g, yml, work, work / "runs" / "live", 4242)
    assert out["brief_stated"] is True
    doc = json.loads((yml.parent / "c.case.json").read_text())
    entry = doc["runs"][0]
    assert entry["run_dir"] == "../live" and entry["pid"] == 4242 and entry["brief"] == 1
    assert entry["inputs"] == case.current_inputs(yml, work).record()
    events = [json.loads(x) for x in (yml.parent / "c.case-log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["brief_recorded", "run_started"]
    assert events[1]["run_dir"] == "../live"
    assert_no_absolute_path(paths_in(yml.parent / "c.case.json", yml.parent / "c.case-log.jsonl"), work)


def test_a_run_folder_outside_the_work_folder_is_named_and_not_stored(tmp_path, work, yml):
    record(yml, work)
    outside = tmp_path.parent / "d2-outside-run-folder"
    g = case.gate(yml, work)
    case.run_started(g, yml, work, outside, 1)
    entry = json.loads((yml.parent / "c.case.json").read_text())["runs"][0]
    assert entry["run_dir"] is None and re.fullmatch(r"<outside>/[0-9a-f]{12}", entry["run_dir_name"])
    assert "d2-outside-run-folder" not in paths_in(yml.parent / "c.case.json", yml.parent / "c.case-log.jsonl")
    text = paths_in(yml.parent / "c.case.json", yml.parent / "c.case-log.jsonl")
    assert str(tmp_path.parent) not in text
    run = case.status(yml, work)["runs"][0]
    assert run["run_dir_rel"] is None and any("outside the work folder" in f for f in run["facts"])


def iso(epoch):
    return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")


def write_run(d, pid=None, started=None, state="done", job=True):
    """A run folder as the CLI writes it. `started` is the ISO time in `job.json` (default: now)."""
    started = started or iso(time.time())
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"state": state, "iteration": 7}))
    (d / "run.json").write_text(json.dumps({"name": "p", "iterations": 10}))
    if job:
        (d / "job.json").write_text(json.dumps({"state": "done", "pid": pid, "started_at": started}))


def test_the_run_list_reads_the_run_folder(work, yml):
    record(yml, work)
    live = work / "runs" / "live"
    g = case.gate(yml, work)
    case.run_started(g, yml, work, live, 111)
    assert case.status(yml, work)["runs"][0]["state"] == "missing"
    write_run(live, pid=111)
    run = case.status(yml, work)["runs"][0]
    assert run["state"] == "done" and run["iteration"] == 7 and run["run_dir_rel"] == "runs/live"
    assert run["pid"] == 111 and run["brief"] == 1 and run["stale"] is False
    # another process wrote the folder after this run started (the default folder is reused by each run)
    write_run(live, pid=222, started="2099-01-01T00:00:00+00:00")
    run = case.status(yml, work)["runs"][0]
    assert run["state"] == "replaced"
    assert any("now holds another run" in f and "pid 222" in f for f in run["facts"])
    # the folder still holds an older run: this run has written nothing yet (or it was stopped early)
    write_run(live, pid=333, started="2000-01-01T00:00:00+00:00")
    run = case.status(yml, work)["runs"][0]
    assert run["state"] == "not_written"
    assert any("earlier run (pid 333" in f and "no data of this run yet" in f for f in run["facts"])


def test_a_later_run_in_the_same_folder_replaces_the_earlier_one(work, yml):
    record(yml, work)
    live = work / "runs" / "live"
    for pid in (1, 2, 3):
        case.run_started(case.gate(yml, work), yml, work, live, pid)
    elsewhere = work / "runs" / "other"
    case.run_started(case.gate(yml, work), yml, work, elsewhere, 4)
    write_run(live, job=False)
    write_run(elsewhere, job=False)
    runs = case.status(yml, work)["runs"]
    assert [r["pid"] for r in runs] == [4, 3, 2, 1]          # newest first
    assert [r["state"] for r in runs] == ["done", "done", "replaced", "replaced"]


def test_a_folder_with_older_files_holds_no_data_of_the_run_yet(work, yml):
    record(yml, work)
    live = work / "runs" / "live"
    write_run(live, job=False)
    old = 1_000_000_000
    for name in ("status.json", "run.json"):
        os.utime(live / name, (old, old))
    case.run_started(case.gate(yml, work), yml, work, live, 5)
    run = case.status(yml, work)["runs"][0]
    assert run["state"] == "not_written" and any("no data of this run yet" in f for f in run["facts"])


# --- the gate ----------------------------------------------------------------------------------------

def test_no_case_file_runs_as_before_with_a_fact(work, yml):
    g = case.gate(yml, work)
    assert g.stop is None and g.case is False
    assert g.reply == {"brief_stated": False, "brief_facts": [case.NO_CASE_FACT]}
    assert case.run_started(g, yml, work, work / "runs" / "live", 1) == g.reply
    assert not (yml.parent / "c.case.json").exists() and not (yml.parent / "c.case-log.jsonl").exists()


def test_a_yaml_outside_the_work_folder_runs_as_before(tmp_path, yml):
    g = case.gate(yml, tmp_path / "other")
    assert g.stop is None and g.reply["brief_stated"] is False and "outside the work folder" in g.reply["brief_facts"][0]
    assert not (yml.parent / "c.case.json").exists()


def test_a_case_with_no_brief_returns_awaiting_user(work, yml):
    (yml.parent / "c.case.json").write_text(json.dumps({"format": 1, "yaml": "c.yaml", "brief": None, "runs": []}))
    g = case.gate(yml, work)
    s = g.stop
    assert s["state"] == "awaiting_user" and s["override"] == "proceed_without_brief=true"
    assert "error" not in s and "pid" not in s
    assert any("no brief" in f for f in s["facts"])
    assert any("record_brief" in o for o in s["options"]) and any("proceed_without_brief=true" in o for o in s["options"])
    assert s["case"] == "runs/part/c.case.json" and s["yaml_rel"] == "runs/part/c.yaml"
    log = [json.loads(x) for x in (yml.parent / "c.case-log.jsonl").read_text().splitlines()]
    assert [(e["event"], e["reason"]) for e in log] == [("gate_returned", "no_brief")]
    # the gate itself wrote no run entry
    assert json.loads((yml.parent / "c.case.json").read_text())["runs"] == []


def test_a_stale_brief_returns_awaiting_user_and_names_the_file(work, yml):
    record(yml, work)
    box(yml.parent / "tip.stl", (18, 4, 0), (20, 6, 4))
    s = case.gate(yml, work).stop
    assert s["state"] == "awaiting_user"
    assert any("stale" in f and "tip.stl" in f for f in s["facts"])
    ev = [json.loads(x) for x in (yml.parent / "c.case-log.jsonl").read_text().splitlines()][-1]
    assert ev["event"] == "gate_returned" and ev["reason"] == "stale" and ev["stale_files"] == ["tip.stl"]


def test_the_override_runs_and_the_log_records_it(work, yml):
    record(yml, work)
    box(yml.parent / "tip.stl", (18, 4, 0), (20, 6, 4))
    g = case.gate(yml, work, proceed_without_brief=True)
    assert g.stop is None and g.case is True and g.override == {"reason": "stale", "stale_files": ["tip.stl"]}
    out = case.run_started(g, yml, work, work / "runs" / "live", 77)
    assert out["brief_stated"] is False and any("stale" in f for f in out["brief_facts"])
    events = [json.loads(x) for x in (yml.parent / "c.case-log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["brief_recorded", "override_used", "run_started"]
    assert events[1]["override"] == "proceed_without_brief=true" and events[1]["stale_files"] == ["tip.stl"]
    entry = json.loads((yml.parent / "c.case.json").read_text())["runs"][0]
    assert entry["brief"] is None and entry["pid"] == 77


def test_the_override_with_a_current_brief_is_not_an_override(work, yml):
    record(yml, work)
    g = case.gate(yml, work, proceed_without_brief=True)
    assert g.override is None and g.reply["brief_stated"] is True
    case.run_started(g, yml, work, work / "runs" / "live", 1)
    assert "override_used" not in paths_in(yml.parent / "c.case-log.jsonl")


def test_a_comment_only_change_passes_the_gate_with_a_fact(work, yml):
    record(yml, work)
    yml.write_text("# note\n" + yml.read_text())
    g = case.gate(yml, work)
    assert g.stop is None and g.reply["brief_stated"] is True
    assert any("comment" in f for f in g.reply["brief_facts"])


def test_the_gate_never_raises(work, yml, monkeypatch):
    record(yml, work)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(case, "current_inputs", boom)
    g = case.gate(yml, work)
    assert g.stop is None and g.reply["brief_stated"] is False and "boom" in g.reply["brief_facts"][0]


def test_a_log_that_cannot_be_written_does_not_turn_a_stop_into_a_pass(work, yml, monkeypatch):
    (yml.parent / "c.case.json").write_text(json.dumps({"format": 1, "yaml": "c.yaml", "brief": None, "runs": []}))

    def no_log(*a, **k):
        raise OSError("read-only folder")

    monkeypatch.setattr(case, "log_event", no_log)
    s = case.gate(yml, work).stop
    assert s["state"] == "awaiting_user" and any("log was not written" in f for f in s["facts"])


# --- texts ---------------------------------------------------------------------------------------------

def test_the_texts_of_the_gate_have_no_habitual_caps(work, yml):
    (yml.parent / "c.case.json").write_text(json.dumps({"format": 1, "yaml": "c.yaml", "brief": None, "runs": []}))
    s = case.gate(yml, work).stop
    text = json.dumps(s).lower()
    for word in ("must", "never", "always", "forbidden"):
        assert not re.search(rf"\b{word}\b", text), word
    assert 'proceed_without_brief=true' in text


# === fix round 1 (review of D2) ========================================================================

SECTION_30_STATUS_KEYS = {"yaml_rel", "exists", "case", "log", "brief", "brief_stated", "brief_stale", "stale_files",
                          "inputs", "runs", "gate", "facts", "options"}


def case_text(yml):
    return paths_in(yml.parent / "c.case.json", yml.parent / "c.case-log.jsonl")


def set_region(yml, role, value):
    data = pyyaml.safe_load(yml.read_text())
    data["regions"][role] = value
    yml.write_text(pyyaml.safe_dump(data))


# --- M1: a file outside the work folder leaves no folder name in the case ---------------------------------

def test_an_stl_outside_the_work_folder_gets_a_hash_key_and_no_folder_name(tmp_path_factory, work, yml):
    away = tmp_path_factory.mktemp("jdoe-private")        # a folder named like a user
    box(away / "tip.stl", (18, 4, 0), (20, 6, 3))
    set_region(yml, "load", {"tip": str(away / "tip.stl")})        # an absolute path in the YAML
    r = record(yml, work)
    (key,) = [k for k in r["brief"]["inputs"]["files"] if k.startswith("load:")]
    assert re.fullmatch(r"load:<outside>/[0-9a-f]{12}/tip\.stl", key)
    assert r["brief"]["inputs"]["files"][key] == runinfo.sha256_file(away / "tip.stl")
    assert any("tip.stl (load) is outside the work folder" in f for f in r["facts"])
    case.run_started(case.gate(yml, work), yml, work, work / "runs" / "live", 9)
    st = case.status(yml, work)
    assert st["brief_stale"] is False and st["runs"][0]["stale"] is False
    blob = case_text(yml) + json.dumps(r) + json.dumps(st)
    assert "jdoe-private" not in blob and str(away) not in blob and str(tmp_path_factory.getbasetemp()) not in blob
    box(away / "tip.stl", (18, 4, 0), (20, 6, 4))          # the outside file changes: it is stale, named by its file name
    st = case.status(yml, work)
    assert st["brief_stale"] is True and st["stale_files"] == ["tip.stl"]
    assert "jdoe-private" not in json.dumps(st)
    assert any("tip.stl (load): the file changed" in f for f in st["facts"])
    g = case.gate(yml, work)
    assert g.stop["state"] == "awaiting_user" and "jdoe-private" not in json.dumps(g.stop)


def test_a_symlink_to_a_file_outside_keeps_its_own_name_and_no_target_folder(tmp_path_factory, work, yml):
    away = tmp_path_factory.mktemp("jdoe-store")
    box(away / "d1.stl", (0, 0, 0), (20, 10, 10))
    (yml.parent / "design.stl").unlink()
    (yml.parent / "design.stl").symlink_to(away / "d1.stl")
    r = record(yml, work)
    assert "design:design.stl" in r["brief"]["inputs"]["files"]
    assert any("design.stl (design) is a link to a file outside the work folder" in f for f in r["facts"])
    assert "jdoe-store" not in case_text(yml) + json.dumps(r)


# --- m3: a file is keyed by the path that the YAML names, not by the link target ---------------------------

def test_a_link_to_a_new_target_with_the_same_bytes_is_not_stale(work, yml):
    d = yml.parent
    (d / "store").mkdir()
    box(d / "store" / "d1.stl", (0, 0, 0), (20, 10, 10))
    (d / "design.stl").unlink()
    (d / "design.stl").symlink_to(Path("store") / "d1.stl")
    r = record(yml, work)
    assert "design:design.stl" in r["brief"]["inputs"]["files"]
    (d / "store" / "d2.stl").write_bytes((d / "store" / "d1.stl").read_bytes())
    (d / "design.stl").unlink()
    (d / "design.stl").symlink_to(Path("store") / "d2.stl")
    st = case.status(yml, work)
    assert st["brief_stale"] is False and st["stale_files"] == [] and st["brief_stated"] is True
    box(d / "store" / "d2.stl", (0, 0, 0), (20, 10, 12))     # new bytes behind the same name
    st = case.status(yml, work)
    assert st["stale_files"] == ["design.stl"]
    assert any("design.stl (design): the file changed" in f for f in st["facts"])
    assert not any("store" in f for f in st["facts"])


# --- m1 and m2: the state of a run entry -------------------------------------------------------------

def test_a_later_run_that_wrote_nothing_does_not_replace_the_earlier_one(work, yml):
    record(yml, work)
    live = work / "runs" / "live"
    t0 = time.time() - 300
    case.run_started(case.gate(yml, work), yml, work, live, 2, t0)
    write_run(live, pid=2, started=iso(t0 + 2))                     # run 2 wrote the folder and finished
    case.run_started(case.gate(yml, work), yml, work, live, 3, t0 + 200)    # run 3 stopped before it wrote
    runs = case.status(yml, work)["runs"]
    assert [(r["pid"], r["state"]) for r in runs] == [(3, "not_written"), (2, "done")]
    assert runs[1]["iteration"] == 7 and not any("later run" in f for f in runs[1]["facts"])


def test_a_later_run_that_wrote_nothing_does_not_replace_the_earlier_one_without_job_json(work, yml):
    record(yml, work)
    live = work / "runs" / "live"
    now = time.time()
    case.run_started(case.gate(yml, work), yml, work, live, 2, now - 100)
    write_run(live, job=False)                                         # run files are newer than run 2, older than run 3
    case.run_started(case.gate(yml, work), yml, work, live, 3, now + 100)
    runs = case.status(yml, work)["runs"]
    assert [(r["pid"], r["state"]) for r in runs] == [(3, "not_written"), (2, "done")]


def test_a_later_run_that_wrote_into_the_folder_replaces_the_earlier_one(work, yml):
    record(yml, work)
    live = work / "runs" / "live"
    t0 = time.time() - 300
    case.run_started(case.gate(yml, work), yml, work, live, 2, t0)
    case.run_started(case.gate(yml, work), yml, work, live, 3, t0 + 100)
    write_run(live, pid=3, started=iso(t0 + 101))                   # the job file is the job of run 3
    runs = case.status(yml, work)["runs"]
    assert [(r["pid"], r["state"]) for r in runs] == [(3, "done"), (2, "replaced")]
    assert any("later run of this case (pid 3" in f for f in runs[1]["facts"])


def test_a_reused_pid_with_a_far_start_time_is_another_run(work, yml):
    record(yml, work)
    live, other = work / "runs" / "live", work / "runs" / "other"
    t0 = time.time() - 600
    case.run_started(case.gate(yml, work), yml, work, other, 6, t0)
    write_run(other, pid=6, started=iso(t0 + 3 * 3600), state="done")     # the same pid, three hours later
    run = case.status(yml, work)["runs"][0]
    assert run["state"] == "replaced" and any("now holds another run (pid 6" in f for f in run["facts"])
    write_run(other, pid=6, started=iso(t0 + 4))                              # the same pid, a few seconds later
    run = case.status(yml, work)["runs"][0]
    assert run["state"] == "done" and run["facts"] == []


# --- m4: a YAML that does not load compares each record with its own files ---------------------------------

def test_an_unloadable_yaml_gives_no_false_new_file_for_the_brief(work, yml):
    d = yml.parent
    record(yml, work)                                                         # brief: design, plate, tip
    box(d / "keep.stl", (5, 0, 0), (8, 10, 10))
    set_region(yml, "keep_out", ["keep.stl"])
    case.run_started(case.gate(yml, work, proceed_without_brief=True), yml, work, work / "runs" / "live", 5)
    yml.write_text("regions: [unclosed\n")                                     # the YAML does not load or parse
    st = case.status(yml, work)
    assert st["stale_files"] == ["c.yaml"]                                     # the YAML only; keep.stl is not "new" for the brief
    assert not any("new file of the problem" in f for f in st["facts"])
    g = case.gate(yml, work)
    assert g.stop["state"] == "awaiting_user"
    assert not any("new file of the problem" in f for f in g.stop["facts"])
    ev = [json.loads(x) for x in (d / "c.case-log.jsonl").read_text().splitlines()][-1]
    assert ev["stale_files"] == st["stale_files"]
    assert st["runs"][0]["stale_files"] == ["c.yaml"]                         # the run entry has keep.stl and the same bytes


# --- n1: .bad files get numbers, brief ids go on from the log ----------------------------------------------

def test_a_second_unreadable_case_file_keeps_the_first_and_the_brief_id_goes_on(work, yml):
    manifest = yml.parent / "c.case.json"
    record(yml, work)
    record(yml, work, text="Two.")
    manifest.write_text("broken one")
    r = record(yml, work, text="Three.")
    assert r["brief"]["id"] == 3 and r["replaced"] is None            # the new case file has no earlier brief
    manifest.write_text("broken two")
    r = record(yml, work, text="Four.")
    assert r["brief"]["id"] == 4
    assert (yml.parent / "c.case.json.bad1").read_text() == "broken one"
    assert (yml.parent / "c.case.json.bad2").read_text() == "broken two"
    ids = [e["brief"]["id"] for e in (json.loads(x) for x in (yml.parent / "c.case-log.jsonl").read_text().splitlines())
           if e["event"] == "brief_recorded"]
    assert ids == [1, 2, 3, 4]


# --- n2: an integer and an equal float are one value --------------------------------------------------------

def test_an_integer_and_an_equal_float_in_the_yaml_are_not_a_change(work, yml):
    assert case._content_digest(yml) is not None
    a, b = yml.parent / "a.yaml", yml.parent / "b.yaml"
    a.write_text("v: 1\nw: [2, 3.5]\n")
    b.write_text("v: 1.0\nw: [2.0, 3.5]\n")
    assert case._content_digest(a) == case._content_digest(b)
    b.write_text("v: true\nw: [2.0, 3.5]\n")
    assert case._content_digest(a) != case._content_digest(b)            # a bool is not a number here
    text = yml.read_text()
    assert "voxel_size: 1.0" in text
    record(yml, work)
    yml.write_text(text.replace("voxel_size: 1.0", "voxel_size: 1"))
    st = case.status(yml, work)
    assert st["brief_stale"] is False and any("not stale" in f for f in st["facts"])
    assert "volume_fraction: 0.3" in text
    yml.write_text(text.replace("volume_fraction: 0.3", "volume_fraction: 0.4"))
    assert case.status(yml, work)["brief_stale"] is True


# --- n3: case_status of a YAML outside the work folder has every key -----------------------------------------

def test_case_status_outside_the_work_folder_has_every_key(tmp_path, work, yml):
    outside = case.status(yml, tmp_path / "elsewhere")
    inside = case.status(yml, work)
    assert set(outside) == set(inside) == SECTION_30_STATUS_KEYS
    assert outside["exists"] is False and outside["brief"] is None and outside["case"] is None and outside["log"] is None
    assert outside["runs"] == [] and outside["stale_files"] == [] and outside["inputs"] is None
    assert outside["brief_stated"] is False and outside["brief_stale"] is False
    assert isinstance(outside["gate"], str) and outside["facts"] and outside["options"]


# --- n4: hook.yaml and hook.yml do not share a case --------------------------------------------------------

def test_two_yaml_names_with_one_stem_do_not_share_a_case_file(work, yml):
    d = yml.parent
    record(yml, work)
    other = d / "c.yml"
    other.write_text(yml.read_text())
    manifest, log = d / "c.case.json", d / "c.case-log.jsonl"
    before, before_log = manifest.read_bytes(), log.read_bytes()
    r = case.record_brief(other, work, "Other.", None)
    assert r["recorded"] is False and r["case"] is None and "error" not in r
    assert any("c.case.json is the case of c.yaml, not of c.yml" in f for f in r["facts"])
    assert any("Rename c.yml" in o for o in r["options"])
    st = case.status(other, work)
    assert st["exists"] is False and set(st) == SECTION_30_STATUS_KEYS and st["options"]
    g = case.gate(other, work)
    assert g.stop is None and g.case is False and any("not of c.yml" in f for f in g.reply["brief_facts"])
    assert case.run_started(g, other, work, work / "runs" / "live", 1) == g.reply
    assert manifest.read_bytes() == before and log.read_bytes() == before_log
    assert json.loads(before)["yaml"] == "c.yaml"
    assert case.status(yml, work)["brief_stated"] is True                      # the case of c.yaml is unchanged


# --- n5: the log lists a run start that case.json lost -------------------------------------------------------

def test_case_status_says_when_the_log_has_a_run_that_the_case_file_lost(work, yml):
    record(yml, work)
    manifest = yml.parent / "c.case.json"
    case.run_started(case.gate(yml, work), yml, work, work / "runs" / "live", 1, time.time() - 50)
    assert not any("run start" in f for f in case.status(yml, work)["facts"])
    saved = manifest.read_bytes()
    case.run_started(case.gate(yml, work), yml, work, work / "runs" / "live", 2, time.time() - 40)
    manifest.write_bytes(saved)                                               # a second server wrote an older copy
    st = case.status(yml, work)
    assert [r["pid"] for r in st["runs"]] == [1]
    assert any("1 run start(s)" in f and "does not hold" in f for f in st["facts"])


# --- m5: a case file in the code folder ------------------------------------------------------------------------

def test_a_yaml_in_the_code_folder_is_recorded_with_a_fact_and_an_option(work, yml):
    d = work / "examples" / "part"
    d.mkdir(parents=True)
    y = make_case(d)
    r = case.record_brief(y, work, BRIEF, None, code_folder=work)
    assert r["recorded"] is True and (d / "c.case.json").exists()
    assert any("code folder" in f for f in r["facts"])
    assert any(workfolder.change_text() in o and "work folder" in o for o in r["options"])
    r = case.record_brief(yml, work, BRIEF, None, code_folder=work)         # `runs/` is not a place of the code
    assert r["recorded"] is True and not any("code folder" in f for f in r["facts"]) and r["options"] == []
    assert case.record_brief(y, work, BRIEF, None)["options"] == []         # no code folder given: no fact


def test_git_ignores_the_case_files():
    lines = (REPO / ".gitignore").read_text().splitlines()
    for pattern in ("*.case.json", "*.case-log.jsonl", "*.case.json.bad*"):
        assert pattern in lines


# === refresh onto the integration (D1b resume, region_files rule) =====================================

def test_a_gate_stop_for_a_resume_names_both_overrides_in_one_option(work, yml):
    record(yml, work)
    box(yml.parent / "tip.stl", (18, 4, 0), (20, 6, 4))
    plain = case.gate(yml, work).stop
    assert not any("allow_changed_inputs" in o for o in plain["options"])        # no resume: one override
    s = case.gate(yml, work, resume=True).stop
    assert s["state"] == "awaiting_user" and s["override"] == "proceed_without_brief=true"
    both = [o for o in s["options"] if "proceed_without_brief=true" in o and "allow_changed_inputs=true" in o]
    assert len(both) == 1 and "resume=true" in both[0]
    assert sum("allow_changed_inputs" in o for o in s["options"]) == 1             # one option, not two
    assert any("record_brief" in o for o in s["options"])
    text = json.dumps(s).lower()
    for word in ("must", "never", "always", "forbidden"):
        assert not re.search(rf"\b{word}\b", text), word
    g = case.gate(yml, work, proceed_without_brief=True, resume=True)               # the override still lets the run start
    assert g.stop is None and g.override == {"reason": "stale", "stale_files": ["tip.stl"]}


def test_the_case_and_the_checkpoint_give_the_same_key_for_the_same_file(tmp_path_factory, work, yml, monkeypatch):
    from topopt import checkpoint
    from topopt.config import load_config
    away = tmp_path_factory.mktemp("jdoe-private")
    box(away / "tip.stl", (18, 4, 0), (20, 6, 3))
    set_region(yml, "load", {"tip": str(away / "tip.stl")})                     # outside the work folder
    d = yml.parent
    (d / "store").mkdir()
    (d / "plate.stl").rename(d / "store" / "plate1.stl")
    (d / "plate.stl").symlink_to(Path("store") / "plate1.stl")                  # a link
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(work))                           # the work folder, as the MCP server sets it
    ck = checkpoint.region_hashes(load_config(yml))
    ca = case.current_inputs(yml, work).files
    assert ck == ca
    assert "fixed:plate.stl" in ca and any(k.startswith("load:<outside>/") for k in ca)
    assert "jdoe-private" not in "".join(ca)
