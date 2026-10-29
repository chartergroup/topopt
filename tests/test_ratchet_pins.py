"""Tests for fix round 1 of the ratchet, group 3: the pins of the judge inputs (B1), the folders that no log line names (M1, m9),
the state of a root (m3), the source that sees a new file under an old name (m4), `approve` (m5) and the stored STL (m8).

The runner runs with a fake judge (the facts come from a table), so no solve is needed. `tests/test_ratchet.py` has the helpers.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from ratchet_helpers import BoxCutSource, box_stl, fake_judge  # noqa: E402
from test_feedback import tree_hash  # noqa: E402
from test_ratchet import (  # noqa: E402
    RULES_MASS_STIFF,
    Kill,
    ctx_for,
    mass_stiff_rules,
    passing,
    quiet,
    rows_of,
    run,
    start_run,
)

from topopt import ratchet as rt  # noqa: E402
from topopt.config import UserError  # noqa: E402
from topopt.ratchet_sources import MarksSource, StlFolderSource, judged_refs  # noqa: E402
from topopt.runjob import JobHeld, RunJob  # noqa: E402

TABLE = {"base": passing(1.24), "c001": passing(1.20), "c002": passing(1.10), "c003": passing(1.00), "c004": passing(0.90)}


def yaml_of(tmp_path) -> Path:
    return tmp_path / "in" / "bracket.yaml"


def set_force(tmp_path, value: float) -> None:
    """The reviewer's case: the load of the YAML changes from -5 N to another value between two calls."""
    p = yaml_of(tmp_path)
    doc = yaml.safe_load(p.read_text())
    doc["load_cases"][0]["forces"][0]["force"][2] = value
    p.write_text(yaml.safe_dump(doc, sort_keys=False))


def tree_no_job(root):
    """The hashes of every file of a root except the job files (a refused call rewrites those)."""
    return {k: v for k, v in tree_hash(root).items() if not Path(k).name.startswith("job.")}


def first_call(tmp_path, table=None, **kw):
    kw.setdefault("max_candidates", 1)
    return run(tmp_path, table or TABLE, n=3, **kw)


# --- B1: the inputs of the judge are pinned ----------------------------------------------------------------------------

def test_b1_a_load_that_changed_between_two_calls_stops_the_run_with_the_facts_and_the_options(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    before = (root / "ratchet.jsonl").read_bytes()
    old_sha = rt.inputs_digest(yaml_of(tmp_path))["yaml_sha256"]
    set_force(tmp_path, -1.0)                                         # -5 N -> -1 N: the parent was solved under -5 N
    calls = []
    with pytest.raises(UserError) as caught:
        run(tmp_path, TABLE, n=3, judge=fake_judge(TABLE, calls=calls))
    msg = str(caught.value)
    new_sha = rt.inputs_digest(yaml_of(tmp_path))["yaml_sha256"]
    assert "load_cases[0].forces[0].force[2]" in msg and "-5.0" in msg and "-1.0" in msg      # which key, which values
    assert old_sha[:10] in msg and new_sha[:10] in msg                                       # old and new sha
    assert "new root" in msg and "rebaseline" in msg and "overwrite" in msg                  # the options
    assert calls == [] and (root / "ratchet.jsonl").read_bytes() == before                    # nothing was judged or logged


def test_b1_a_changed_region_file_is_named_and_a_changed_material_too(tmp_path):
    first_call(tmp_path)
    box_stl(tmp_path / "in" / "tip.stl", size=(3.0, 4.0, 5.0), at=(22.5, 0.0, 2.5))           # a region file with other bytes
    with pytest.raises(UserError, match=r"region file .*tip.*changed"):
        run(tmp_path, TABLE, n=3)
    # the YAML key of a material is a leaf like any other: it is named with both values
    doc = yaml.safe_load(yaml_of(tmp_path).read_text())
    doc["material"] = {"E": 1234.0}
    yaml_of(tmp_path).write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(UserError, match=r"material\.E"):
        run(tmp_path, TABLE, n=3)


def test_b1_the_same_inputs_go_on_and_a_state_file_without_pins_falls_back_to_the_log(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    res = run(tmp_path, TABLE, n=3)                                                          # same inputs: goes on
    assert res["exit_code"] == 0 and res["accepted"] == ["c002", "c003"]
    (root / "ratchet.json").unlink()                                                         # the state file is lost
    set_force(tmp_path, -1.0)
    with pytest.raises(UserError, match="state file is missing"):
        run(tmp_path, TABLE, n=3)


def test_b1_rebaseline_judges_the_parent_again_keeps_the_log_and_compares_with_the_new_facts(tmp_path):
    first_call(tmp_path)                                                                     # c000 baseline, c001 -> acc_01
    root = tmp_path / "root"
    old_lines = (root / "ratchet.jsonl").read_bytes()
    set_force(tmp_path, -1.0)
    table = {**TABLE, "c002": passing(1.50), "c003": passing(1.40)}                          # c002 is the re-baseline record
    res = run(tmp_path, table, n=3, rebaseline=True, max_candidates=1)
    rows = rows_of(root)
    assert (root / "ratchet.jsonl").read_bytes().startswith(old_lines)                        # the log only grew
    assert [(r["type"], r["id"]) for r in rows] == [("baseline", "c000"), ("candidate", "c001"), ("baseline", "c002"),
                                                    ("candidate", "c003")]
    rebase = rows[2]
    assert rebase["rebase_of"] == "acc_01" and rebase["source"]["kind"] == "rebaseline" and rebase["promoted"] is None
    assert any("force[2]" in d for d in rebase["source"]["facts"]["differences"])
    assert rows[1]["inputs"]["inputs_sha256"] != rebase["inputs"]["inputs_sha256"] == rows[3]["inputs"]["inputs_sha256"]
    # c003 (mass 1.40) is lighter than the NEW facts of the parent (1.50) and heavier than the old ones (1.20)
    assert rows[3]["decision"] == "ACCEPT" and rows[3]["parent"] == "acc_01" and rows[3]["promoted"] == "acc_02"
    assert res["exit_code"] == 0 and any("re-baseline" in n for n in res["notes"])
    state = json.loads((root / "ratchet.json").read_text())
    assert state["inputs"]["values"]["load_cases[0].forces[0].force[2]"] == -1.0
    again = run(tmp_path, table, n=3, max_candidates=1)                                      # the pins are the new ones: no flag
    assert again["exit_code"] == 0


def test_b1_new_rules_and_a_new_mesh_size_are_refused_and_rebaseline_covers_them_too(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    other = rt.make_rules([{"rule": "watertight"}, {"rule": "mass", "must_fall": True}], **RULES_MASS_STIFF)
    with pytest.raises(UserError, match="rules .* differ"):
        run(tmp_path, TABLE, n=3, rules=other)
    table = {**TABLE, "c002": passing(1.50), "c003": passing(1.40)}
    res = run(tmp_path, table, n=3, rules=other, rebaseline=True, max_candidates=1)
    rows = rows_of(root)
    assert [r["type"] for r in rows] == ["baseline", "candidate", "baseline", "candidate"] and res["exit_code"] == 0
    assert any("rules" in d for d in rows[2]["source"]["facts"]["differences"])
    assert rows[2]["inputs"]["rules_sha256"] == rows[3]["inputs"]["rules_sha256"] != rows[1]["inputs"]["rules_sha256"]
    calls = []
    with pytest.raises(UserError, match="mesh size 2 mm differs"):
        run(tmp_path, table, n=3, rules=other, mesh_size_mm=2.0, judge=fake_judge(table, calls=calls))
    assert calls == []
    run(tmp_path, {**table, "c004": passing(1.45), "c005": passing(1.30)}, n=3, rules=other, mesh_size_mm=2.0, rebaseline=True,
        max_candidates=1, judge=fake_judge({**table, "c004": passing(1.45), "c005": passing(1.30)}, calls=calls))
    assert calls[0][0] == "c004" and calls[0][2] == 2.0                                          # the re-baseline used the new size


def test_b1_rebaseline_with_no_difference_adds_no_record(tmp_path):
    first_call(tmp_path)
    run(tmp_path, TABLE, n=3, rebaseline=True, max_candidates=1)
    assert [r["type"] for r in rows_of(tmp_path / "root")] == ["baseline", "candidate", "candidate"]


def test_b1_a_failed_rebaseline_keeps_the_old_pins_so_that_it_is_asked_for_again(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    set_force(tmp_path, -1.0)
    table = {**TABLE, "c002": ValueError("the solver broke"), "c003": passing(1.50), "c004": passing(1.40)}
    res = run(tmp_path, table, n=3, rebaseline=True)
    assert res["exit_code"] == 2 and res["stopped"] == "rebaseline_failed"
    rows = rows_of(root)
    assert rows[-1]["id"] == "c002" and rows[-1]["errors"]["judge"] and rows[-1]["rebase_of"] == "acc_01"
    state = json.loads((root / "ratchet.json").read_text())
    assert state["state"] == "baseline_failed" and state["inputs"]["values"]["load_cases[0].forces[0].force[2]"] == -5.0
    with pytest.raises(UserError, match="rebaseline"):                                       # still a difference
        run(tmp_path, table, n=3)
    res = run(tmp_path, table, n=3, rebaseline=True, max_candidates=1)
    assert res["exit_code"] == 0 and rows_of(root)[-2]["id"] == "c003" and rows_of(root)[-1]["id"] == "c004"
    assert rows_of(root)[-1]["decision"] == "ACCEPT"                                         # compared with c003 (1.50), not the old 1.20


def test_b1_rebaseline_is_refused_while_a_candidate_waits_for_the_owner(tmp_path):
    run(tmp_path, TABLE, n=3, owner_gate=True)
    set_force(tmp_path, -1.0)
    with pytest.raises(UserError, match="wait for the owner"):
        run(tmp_path, TABLE, n=3, owner_gate=True, rebaseline=True)


def test_b1_the_yaml_that_changes_while_a_candidate_is_judged_stops_the_run_and_logs_nothing_for_it(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    inner = fake_judge(TABLE)

    def judge(shadow_dir, parent_dir, ctx):
        got = inner(shadow_dir, parent_dir, ctx)
        if json.loads((Path(shadow_dir) / "run.json").read_text())["ratchet"]["id"] == "c002":
            set_force(tmp_path, -1.0)                                                         # an edit during the judge call
        return got
    with pytest.raises(rt.RatchetError, match=r"changed while the candidate was judged.*force\[2\]"):
        run(tmp_path, TABLE, n=3, judge=judge)
    assert [r["id"] for r in rows_of(root)] == ["c000", "c001"]
    assert rt.status(root)["state"] == "failed"


def test_b1_approve_needs_the_inputs_of_the_judge_but_a_decline_does_not(tmp_path):
    run(tmp_path, TABLE, n=3, owner_gate=True)
    root = tmp_path / "root"
    set_force(tmp_path, -1.0)
    with pytest.raises(UserError, match=r"other inputs.*force\[2\]"):
        rt.approve(root, say=quiet)
    assert rt.approve(root, decline=True, say=quiet)["owner_approval"] == "declined"


# --- M1, m9: a folder that no log line names -----------------------------------------------------------------------------

def test_m1_the_reviewers_scenario_a_root_without_a_log_but_with_acc_folders_is_refused_and_nothing_is_changed(tmp_path):
    run(tmp_path, TABLE, n=3, max_candidates=2)
    root = tmp_path / "root"
    (root / "ratchet.jsonl").unlink()
    (root / "ratchet.json").unlink()                                                          # "start again": the log is gone
    before = tree_no_job(root)
    with pytest.raises(UserError, match="earlier ratchet"):
        run(tmp_path, TABLE, n=3)
    assert tree_no_job(root) == before
    res = run(tmp_path, TABLE, n=3, overwrite=True, max_candidates=1)
    assert res["exit_code"] == 0 and (root / "earlier_01" / "acc_02" / "run.json").is_file()
    assert (root / "earlier_01" / "acc_01" / "run.json").is_file()                              # moved, not deleted
    new = json.loads((root / "acc_01" / "run.json").read_text())["ratchet"]
    assert new["id"] == "c001"


def test_m1_an_acc_folder_that_the_log_does_not_name_is_not_used_and_not_changed(tmp_path):
    run(tmp_path, TABLE, n=3, max_candidates=1)
    root = tmp_path / "root"
    shutil.copytree(root / "acc_01", root / "acc_02")
    (root / "acc_02" / "from_an_earlier_ratchet.txt").write_text("keep me")
    before = tree_hash(root / "acc_02")
    res = run(tmp_path, TABLE, n=3, max_candidates=1)
    row = rows_of(root)[-1]
    assert row["decision"] == "ACCEPT" and row["promoted"] == "acc_03"                         # the next free number
    assert tree_hash(root / "acc_02") == before and res["unlogged_acc"] == ["acc_02"]
    assert any("acc_02" in n and "no log line names" in n for n in res["notes"])
    assert rt.derive(rows_of(root)).parent == "acc_03" and rt.derive(rows_of(root)).next_acc == 4


def test_m1_the_folders_of_a_judge_that_was_killed_are_moved_to_unlogged_and_not_deleted(tmp_path):
    table = {"base": passing(1.24), "c001": Kill()}
    with pytest.raises(Kill):
        run(tmp_path, table, n=3)
    root = tmp_path / "root"
    (root / "shadow_001" / "evidence.txt").write_text("the judged folder")
    table["c001"] = passing(1.20)
    res = run(tmp_path, table, n=3, max_candidates=1)
    moved = list((root / "unlogged").glob("*/shadow_001"))
    assert len(moved) == 1 and (moved[0] / "evidence.txt").read_text() == "the judged folder"
    assert any("shadow_001" in n and "nothing is deleted" in n for n in res["notes"])
    assert [r["id"] for r in rows_of(root)] == ["c000", "c001"] and (root / "acc_01").is_dir()


def test_m1_a_promotion_never_lands_on_a_folder_of_another_candidate(tmp_path, monkeypatch):
    real_move = rt._move_dir

    def dying_move(src, dst):
        raise Kill()
    monkeypatch.setattr(rt, "_move_dir", dying_move)
    with pytest.raises(Kill):
        run(tmp_path, TABLE, n=3, max_candidates=1)
    monkeypatch.setattr(rt, "_move_dir", real_move)
    root = tmp_path / "root"
    assert rows_of(root)[-1]["promoted"] == "acc_01" and not (root / "acc_01").exists()
    shutil.copytree(root / "acc_00", root / "acc_01")                                          # a folder of another ratchet
    before = (tree_hash(root / "acc_01"), tree_hash(root / "shadow_001"))
    with pytest.raises(UserError, match=r"acc_01 is in .* and so is shadow_001"):
        run(tmp_path, TABLE, n=3)
    assert (tree_hash(root / "acc_01"), tree_hash(root / "shadow_001")) == before             # nothing was changed
    shutil.rmtree(root / "shadow_001")
    with pytest.raises(UserError, match=r"acc_01 .*is not the candidate c001"):
        rt.recover(root, rt.read_state_file(root)["start_run"], rows_of(root), with_labels=False)


def test_m9_a_kill_in_the_archive_loses_nothing_and_the_next_overwrite_finishes_it(tmp_path, monkeypatch):
    run(tmp_path, {**TABLE, "c001": passing(1.30)}, n=3, max_candidates=1)                     # c001 is a REJECT: shadow_001 stays
    root = tmp_path / "root"
    before = sorted(p.name for p in root.iterdir() if not p.name.startswith("job."))
    real, count = os.replace, {"n": 0}

    def dying(src, dst):
        count["n"] += 1
        if count["n"] == 4:                                                                    # after ratchet.jsonl has moved
            raise Kill()
        return real(src, dst)
    monkeypatch.setattr(os, "replace", dying)
    with pytest.raises(Kill):
        rt.archive_root(root)
    monkeypatch.setattr(os, "replace", real)
    assert (root / "earlier_01.tmp").is_dir() and (root / "shadow_001").is_dir() and not (root / "ratchet.jsonl").exists()
    with pytest.raises(UserError, match="archive was not finished"):                          # not a root to start in; nothing deleted
        run(tmp_path, TABLE, n=3)
    assert (root / "shadow_001").is_dir()
    assert rt.archive_root(root) == "earlier_01"
    assert sorted(p.name for p in (root / "earlier_01").iterdir()) == before
    assert not (root / "earlier_01.tmp").exists()


# --- m3: the state of a root comes from the job files, for any runner ------------------------------------------------------

def test_m3_status_with_a_live_runner_of_any_kind_a_dead_one_and_a_final_state_in_the_file(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    state_path = root / "ratchet.json"

    def set_state(name):
        data = json.loads(state_path.read_text())
        data["state"] = name
        state_path.write_text(json.dumps(data))
    set_state("running")                                                                      # a killed runner: no live lock
    st = rt.status(root)
    assert st["state"] == "failed" and st["process_alive"] is False and "no final state" in st["summary"]
    live = RunJob(root)                                                                        # a live runner of the CLI kind
    live.start()
    try:
        st = rt.status(root)
        assert st["state"] == "running" and st["process_alive"] is True and st["pid"] == os.getpid()
        assert st["job"]["state"] == "running"
        set_state("finished")                                                                 # a final state waits for the process
        st = rt.status(root)
        assert st["state"] == "running" and st["file_state"] == "finished" and "still alive" in st["summary"]
    finally:
        live.finish(0)
    assert rt.status(root)["state"] == "finished"
    assert rt.status(root, starting_pid=4242)["state"] == "running"                          # a final file state, a new process
    state_path.unlink()
    assert rt.status(root, starting_pid=4242)["state"] == "starting"


def test_m3_a_refused_second_call_is_a_failed_job_and_not_the_finished_state_of_the_first_call(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    assert rt.status(root)["state"] == "finished"
    set_force(tmp_path, -1.0)
    with pytest.raises(UserError):
        run(tmp_path, TABLE, n=3)
    st = rt.status(root)
    assert st["state"] == "failed" and st["file_state"] == "finished" and st["job"]["state"] == "failed"


# --- L-m1: the text of the error stays in the job record and status shows it --------------------------------------------------

def test_lm1_status_after_a_refused_start_shows_the_error_text_and_not_the_killed_text(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    set_force(tmp_path, -1.0)
    with pytest.raises(UserError) as caught:
        run(tmp_path, TABLE, n=3)
    st = rt.status(root)
    assert st["state"] == "failed" and st["last_error"]["exit_code"] == 1
    assert st["last_error"]["message"] == str(caught.value)                                  # the whole text, with the options
    assert "force[2]" in st["summary"] and "rebaseline" in st["summary"] and "exit code 1" in st["summary"]
    assert "killed" not in st["summary"] and "no exit code" not in st["summary"]
    assert json.loads((root / "job.json").read_text())["error"] == str(caught.value)       # in the job record
    assert (root / "job.FAILED").is_file()


def test_lm1_the_cli_status_prints_the_reason(tmp_path, capsys):
    first_call(tmp_path)
    set_force(tmp_path, -1.0)
    with pytest.raises(UserError):
        run(tmp_path, TABLE, n=3)
    capsys.readouterr()
    assert rt.main(["status", str(tmp_path / "root")]) == 0
    out = capsys.readouterr().out
    assert "failed; the last ratchet process ended with exit code 1" in out and "load_cases[0].forces[0].force[2]" in out


def test_lm1_an_internal_error_and_ctrl_c_are_recorded_too(tmp_path):
    root = tmp_path / "root"

    class Broken:
        kind, needs_labels = "broken", False

        def describe(self):
            return {"kind": self.kind}

        def propose(self, parent_dir, out_dir, history):
            raise ValueError("the source broke")
    r, ypath = start_run(tmp_path)
    kw = dict(root=root, judge=fake_judge(TABLE), ctx=ctx_for(ypath), yaml_path=ypath, say=quiet)
    with pytest.raises(ValueError):
        rt.run_ratchet(r, Broken(), mass_stiff_rules(), **kw)
    st = rt.status(root)
    assert st["state"] == "failed" and st["last_error"]["exit_code"] == 2
    assert st["last_error"]["message"] == "internal error: ValueError: the source broke"
    folder = tmp_path / "cands"
    box_stl(folder / "part_1.stl", size=(2.0, 1.0, 1.0))
    with pytest.raises(KeyboardInterrupt):
        rt.run_ratchet(r, StlFolderSource(folder), mass_stiff_rules(),
                       **dict(kw, judge=fake_judge({"base": passing(1.24), "c001": KeyboardInterrupt()})))
    st = rt.status(root)
    assert st["last_error"] == {"exit_code": 130, "message": "stopped by Ctrl+C"} and "Ctrl+C" in st["summary"]


def test_lm1_a_process_with_no_exit_code_keeps_the_killed_text_and_job_held_writes_nothing(tmp_path):
    first_call(tmp_path)
    root = tmp_path / "root"
    data = json.loads((root / "ratchet.json").read_text())
    data["state"] = "running"
    (root / "ratchet.json").write_text(json.dumps(data))
    info = json.loads((root / "job.json").read_text())                                     # what a kill leaves: a record that says
    for key in ("exit_code", "ended_at", "wall_s", "peak_rss_bytes"):                      # running, a lock of a process that is gone,
        info.pop(key)                                                                      # and no marker
    info["state"] = "running"
    (root / "job.json").write_text(json.dumps(info))
    (root / "job.DONE").unlink()
    (root / "job.lock").write_text(json.dumps({"format": 1, "pid": 2 ** 22 + 12345, "token": info["token"],
                                               "started_at": info["started_at"]}))
    st = rt.status(root)
    assert st["state"] == "failed" and "last_error" not in st
    assert "no exit code" in st["summary"] and "killed" in st["summary"] and "stale lock" in st["summary"]
    # a second runner that gets JobHeld does not write into the record of the process that holds the lock
    live = RunJob(root / "held")
    live.start()
    try:
        before = (root / "held" / "job.json").read_bytes()
        with pytest.raises(JobHeld):
            run(tmp_path, TABLE, n=3, root=root / "held")
        assert (root / "held" / "job.json").read_bytes() == before and "error" not in json.loads(before)
    finally:
        live.finish(0)


def test_m3_status_names_a_promotion_that_the_log_holds_and_the_folders_do_not(tmp_path, monkeypatch):
    def dying_move(src, dst):
        raise Kill()
    monkeypatch.setattr(rt, "_move_dir", dying_move)
    with pytest.raises(Kill):
        run(tmp_path, TABLE, n=3, max_candidates=1)
    st = rt.status(tmp_path / "root")
    assert st["parent"] == "acc_01" and st["promotions_open"] == ["acc_01"] and st["state"] == "failed"
    assert "folder is not in the root yet" in st["summary"]


# --- m4: a new file under an old name is judged ------------------------------------------------------------------------

def test_m4_a_stl_file_with_an_old_name_and_other_content_is_judged_and_the_same_content_is_not(tmp_path):
    run(tmp_path, TABLE, n=2)
    root = tmp_path / "root"
    n_rows = len(rows_of(root))
    same = run(tmp_path, TABLE, n=2)
    assert same["stopped"] == "no more candidates" and len(rows_of(root)) == n_rows
    assert any("nothing new" in n and "same name and the same content" in n for n in same["notes"])
    box_stl(tmp_path / "cands" / "part_1.stl", size=(2.5, 1.0, 1.0))                           # same name, other bytes
    res = run(tmp_path, {**TABLE, "c003": passing(0.80)}, n=2)
    row = rows_of(root)[-1]
    assert row["source"]["ref"] == "part_1.stl" and row["source"]["facts"]["same_ref_as_earlier"] is True
    assert "other content" in row["source"]["note"] and res["exit_code"] == 0


def test_q1_17_the_last_note_of_a_call_that_judged_candidates_says_how_many_and_that_the_source_has_no_more(tmp_path):
    first = run(tmp_path, TABLE, n=2)                                   # a new root: two candidates are judged, then the source is empty
    assert first["stopped"] == "no more candidates"
    (note,) = [n for n in first["notes"] if "no more for this root" in n]
    assert note.startswith("this call judged 2 candidate(s), and the source (2 candidate(s)) has no more for this root: ")
    assert "2 candidate(s) of kind" in note and "nothing new" not in note
    assert "A file with the same name and other content is judged as a new candidate" in note
    second = run(tmp_path, TABLE, n=2)                                  # nothing judged in this call: the old wording stays
    (note2,) = [n for n in second["notes"] if "nothing new for this root" in n]
    assert note2.startswith("the source (2 candidate(s)) has nothing new for this root: ") and "this call judged" not in note2
    box_stl(tmp_path / "cands" / "part_3.stl", size=(2.5, 1.0, 1.0))                           # one more file in the source
    third = run(tmp_path, TABLE, n=3)                                   # judged 1 now, so the note says how many again
    assert any(n.startswith("this call judged 1 candidate(s), and the source (3 candidate(s)) has no more for this root: ")
               for n in third["notes"])


def test_m4_a_marks_file_of_the_same_name_with_other_content_offers_its_marks_again(tmp_path):
    src = object.__new__(MarksSource)
    src.path, src.doc, src.sha256 = Path("m.json"), {"marks": [{"id": 1}, {"id": 2}]}, "new"
    src._apply = lambda parent, out, ref, mark, stale_ok: rt_candidate(ref)

    def rt_candidate(ref):
        from topopt.ratchet_sources import Candidate
        return Candidate("marks", ref, None, None, facts={})
    row = {"type": "candidate", "source": {"ref": "m.json#1"}, "inputs": {"marks_sha256": "old"}}
    assert judged_refs([row]) == {"m.json#1": {"old"}}
    got = src.propose(tmp_path, tmp_path / "out", [row])
    assert got.ref == "m.json#1" and got.facts["same_ref_as_earlier"] is True and "another marks file" in got.note
    done = [row, {"type": "candidate", "source": {"ref": "m.json#1"}, "inputs": {"marks_sha256": "new"}}]
    assert src.propose(tmp_path, tmp_path / "out", done).ref == "m.json#2"
    both = done + [{"type": "candidate", "source": {"ref": "m.json#2"}, "inputs": {"marks_sha256": "new"}}]
    assert src.propose(tmp_path, tmp_path / "out", both) is None


# --- m5: approve recovers and keeps the job record of the run --------------------------------------------------------------

def test_m5_a_second_approve_finishes_a_move_that_a_kill_left_and_the_job_record_of_the_run_stays(tmp_path, monkeypatch):
    run(tmp_path, TABLE, n=3, owner_gate=True)
    root = tmp_path / "root"
    job_before = (root / "job.json").read_bytes()
    real_move = rt._move_dir

    def dying_move(src, dst):
        raise Kill()
    monkeypatch.setattr(rt, "_move_dir", dying_move)
    with pytest.raises(Kill):
        rt.approve(root, say=quiet)
    monkeypatch.setattr(rt, "_move_dir", real_move)
    assert rows_of(root)[-1]["type"] == "approval" and not (root / "acc_01").exists()
    out = rt.approve(root, say=quiet)                                                          # not "no candidate waits"
    assert out["promoted"] == "acc_01" and out["parent"] == "acc_01" and out["recovered"]
    assert (root / "acc_01" / "result" / "final.stl").is_file() and not (root / "shadow_001").exists()
    assert (root / "job.json").read_bytes() == job_before and rt.status(root)["job"]["state"] == "done"
    assert (root / "approve_job" / "job.json").is_file()
    with pytest.raises(UserError, match="no candidate waits"):
        rt.approve(root, say=quiet)


def test_m5_approve_stops_while_a_runner_holds_the_root(tmp_path):
    run(tmp_path, TABLE, n=3, owner_gate=True)
    root = tmp_path / "root"

    live = RunJob(root)
    live.start()
    try:
        with pytest.raises(JobHeld):
            rt.approve(root, say=quiet)
    finally:
        live.finish(0)


def test_m5_approve_takes_the_next_free_acc_number(tmp_path):
    run(tmp_path, TABLE, n=3, owner_gate=True)
    root = tmp_path / "root"
    (root / "acc_01").mkdir()
    (root / "acc_01" / "x.txt").write_text("not ours")
    out = rt.approve(root, say=quiet)
    assert out["promoted"] == "acc_02" and (root / "acc_01" / "x.txt").read_text() == "not ours"


# --- m8: the STL of a candidate is stored once ---------------------------------------------------------------------------

def cut_source():
    return BoxCutSource([("cut1", (0.0, 0.0, 6.0), (4.0, 20.0, 20.0))])


def test_m8_the_stl_of_a_candidate_folder_is_a_hard_link_in_the_judged_folder(tmp_path):
    start, ypath = start_run(tmp_path)
    table = {"base": passing(1.24), "c001": passing(1.20)}
    res = rt.run_ratchet(start, cut_source(), mass_stiff_rules(), root=tmp_path / "root", judge=fake_judge(table),
                         ctx=ctx_for(ypath), yaml_path=ypath, say=quiet)
    root = tmp_path / "root"
    assert res["accepted"] == ["c001"]
    cand, final = root / "cand_001" / "edited.stl", root / "acc_01" / "result" / "final.stl"
    assert cand.stat().st_ino == final.stat().st_ino and cand.read_bytes() == final.read_bytes()


def test_m8_a_file_of_the_user_is_copied_and_a_file_system_without_hard_links_gets_a_copy(tmp_path, monkeypatch):
    run(tmp_path, TABLE, n=1)
    root = tmp_path / "root"
    assert (root / "acc_01" / "result" / "final.stl").stat().st_ino != (tmp_path / "cands" / "part_1.stl").stat().st_ino
    src = tmp_path / "a.stl"
    src.write_bytes(b"solid")

    def no_link(a, b):
        raise OSError("no hard links here")
    monkeypatch.setattr(os, "link", no_link)
    assert rt._copy_atomic(src, tmp_path / "b.stl", link=True) is False
    assert (tmp_path / "b.stl").read_bytes() == b"solid" and not (tmp_path / "b.stl.tmp").exists()


# --- the CLI ---------------------------------------------------------------------------------------------------------------

def test_the_cli_run_has_the_rebaseline_flag(capsys):
    with pytest.raises(SystemExit):
        rt.main(["run", "--help"])
    assert "--rebaseline" in capsys.readouterr().out
