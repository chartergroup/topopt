"""After a refused run, `run_status` and `run_result` say which facts are of the earlier run in the folder (P9-18, CONTRACTS.md section 24.1).

A fresh `topopt run` that an input check refuses keeps the earlier run in its folder (F4-43). Only `status.json` and the job files show the
refused job. `run_status` used to read `history.jsonl`, `verdict.json` and the checkpoint of the folder too, so its summary gave the pass
count and the verdict of the earlier run next to the error of the refused job, and the first option said "continues the run" with no word
about which run. Now the reply has the key `earlier_run` with those facts, the top-level keys of them are null, and the texts name the
earlier run. A folder that is not this case gets the reply of before: the files of the cases below are built here, the detection
(`server._refused_over_earlier_run`) is false for them, and the replies equal the replies of the same code with the detection forced off.
That stays true when another change edits a text of `run_status` or `run_result`.
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("mcp")

import topopt.optimize as optimize_module  # noqa: E402
import topopt.snapshot as snapshot_module  # noqa: E402
from topopt.checkpoint import META  # noqa: E402
from topopt.cli import EARLIER_RUN_KEPT, main  # noqa: E402
from topopt.runjob import RunJob  # noqa: E402
from topopt_mcp import server  # noqa: E402
from test_refused_run_keeps_result import finished, run_args, server_state, status_of, work  # noqa: E402,F401  (fixtures)
from test_user_errors import edit  # noqa: E402

GUARD_TEXT = ("model has 275 active voxels, the limit is 10 (max_active_voxels). Try voxel_size 6.22 or more, or raise "
              "settings.max_active_voxels in the YAML if this machine has the memory (file: /case/c.yaml).")
FIXED_RAM = {"total_gib": 16.0, "available_now_gib": 8.0, "note": "fixed for the test"}      # the room now changes from call to call

# The times of the built folders are fixed, so that the replies do not change from one run to the next.
EARLIER_RUN_AT = 1_792_000_000        # the files of the earlier run
JOB_AT = EARLIER_RUN_AT + 600         # the start of the job that the folder shows


@pytest.fixture(autouse=True)
def clean_server(monkeypatch, tmp_path):
    server._current.clear()
    server._stopped.clear()
    monkeypatch.setattr(server, "_ram_now", lambda: FIXED_RAM)
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))      # `run_dir_rel` is relative to the work folder: a fixed one keeps the replies equal
    yield
    server._current.clear()
    server._stopped.clear()


# --- folders that are built here ------------------------------------------------------------------

def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def put(path, text, at):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (at, at))


def write_earlier_run(folder, *, passes=3, total=3, complete=True):
    """The files of a run that went on to pass `passes` of `total`: run.json, history, verdict (if it ended), checkpoint, snapshot, result."""
    case = folder.parent / "case"
    put(folder / "run.json", json.dumps({"iterations": total, "yaml_path": str(case / "c.yaml")}), EARLIER_RUN_AT)
    rows = [{"iteration": i, "elapsed_s": 0.5 * i, "compliance": 10.0 - i, "volume_fraction": 0.3 - 0.001 * i} for i in range(1, passes + 1)]
    put(folder / "history.jsonl", "".join(json.dumps(r) + "\n" for r in rows), EARLIER_RUN_AT)
    if complete:
        verdict = {"format": 1, "state": "iteration_limit", "text": f"iteration limit reached at pass {passes}. Tests: grayness 0.76 (not met).",
                   "final_iteration": passes, "iterations": total, "converged": False, "feasible": True, "facts": {"iteration": passes}}
        put(folder / "verdict.json", json.dumps(verdict), EARLIER_RUN_AT)
    meta = {"iteration": passes, "n_iterations": total, "complete": complete, "written_at": iso(EARLIER_RUN_AT), "elapsed_s": 1.5,
            "topopt_version": "0.1.0", "format": 1}
    np.savez(folder / "checkpoint.npz", **{META: np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)})
    (folder / "snapshots").mkdir(exist_ok=True)
    (folder / "snapshots" / "it_0000.bin").write_bytes(b"x")
    put(folder / "result" / "final.stl", "solid x\nendsolid x\n", EARLIER_RUN_AT)
    put(folder / "run.log", "topopt run: cantilever\ndone in 1.5 s\n", EARLIER_RUN_AT)


def write_job(folder, *, started=JOB_AT, state="failed", exit_code=1):
    """The job files of a job that ended with a failure. `job.json` is written by hand: a built folder has no process."""
    info = {"format": 1, "state": state, "pid": 4242, "started_at": iso(started), "run_dir": str(folder), "success_codes": [0],
            "exit_code": exit_code, "ended_at": iso(started + 1), "wall_s": 0.7, "peak_rss_bytes": 115642368, "peak_rss_source": "VmHWM",
            "notes": []}
    put(folder / "job.json", json.dumps(info), started + 1)
    put(folder / "job.FAILED", f"exit {exit_code}\n", started + 1)
    put(folder / "job.guard", "", started)


def write_status(folder, message, *, iteration=0, step="setup", at=JOB_AT + 1):
    put(folder / "status.json", json.dumps({"state": "error", "iteration": iteration, "step": step, "elapsed_s": 0.68, "message": message}), at)


def build(folder, kind):
    """A run folder of one of the cases. The names are the cases of the card."""
    folder.mkdir(parents=True)
    if kind in ("refused", "refused_unfinished"):     # the case of the card: a refused job over an earlier run
        if kind == "refused":
            write_earlier_run(folder)
        else:
            write_earlier_run(folder, passes=2, total=6, complete=False)
        write_status(folder, f"{GUARD_TEXT} {EARLIER_RUN_KEPT}")
        write_job(folder)
    elif kind == "leftovers":    # a refused job in an empty folder: what a run leaves when it stops in its setup
        write_status(folder, GUARD_TEXT)
        put(folder / "run.log", "topopt run: cantilever\n", JOB_AT)
        (folder / "snapshots").mkdir()
        write_job(folder)
    elif kind == "failed_resume":    # a resume that failed: the earlier run is the run, its files are its own
        write_earlier_run(folder)
        write_status(folder, "internal error: boom", iteration=3, step="post")
        write_job(folder)
    elif kind == "failed_later_pass":    # a run that went on, failed at pass 3 of 6, and has a checkpoint at pass 2
        write_earlier_run(folder, passes=2, total=6, complete=False)
        write_status(folder, "internal error: boom", iteration=3, step="solve")
        write_job(folder)
    elif kind == "sentence_without_a_run":    # the text of a refused job, but the earlier run has gone: nothing to mark
        write_status(folder, f"{GUARD_TEXT} {EARLIER_RUN_KEPT}")
        write_job(folder)
    elif kind == "job_began_after_the_status":    # a later job (a resume that failed before it wrote a status) began after the refused one
        write_earlier_run(folder)
        write_status(folder, f"{GUARD_TEXT} {EARLIER_RUN_KEPT}", at=JOB_AT + 1)
        write_job(folder, started=JOB_AT + 3600)
    else:
        raise AssertionError(kind)
    return folder


def replies(folder):
    return {"run_status": server.run_status(str(folder)), "run_result": server.run_result(str(folder))}


def detected(folder):
    """The detection on the files of the folder, with the arguments that `run_status` gives it."""
    return server._refused_over_earlier_run(folder, server._read_json(folder / "status.json"), server._job_facts(folder))


def replies_with_detection_off(folder, monkeypatch):
    """The replies of the same code with the detection forced off: the code path of before (no mark, no moved facts)."""
    with monkeypatch.context() as m:
        m.setattr(server, "_refused_over_earlier_run", lambda *args: False)
        return replies(folder)


def check_no_mark(folder, monkeypatch):
    """No mark: the detection is false, the replies have no `earlier_run`, and they equal (keys, order, text) the replies with the detection off."""
    live = replies(folder)
    assert detected(folder) is False
    assert not any("earlier_run" in reply for reply in live.values())
    off = replies_with_detection_off(folder, monkeypatch)
    for tool in live:
        assert json.dumps(live[tool], indent=1, default=str) == json.dumps(off[tool], indent=1, default=str), tool
    return live


NOT_THE_CASE = ["leftovers", "failed_resume", "failed_later_pass", "sentence_without_a_run", "job_began_after_the_status"]
RUN_DATA = {"leftovers": "setup only: no pass is done", "failed_resume": "up to pass 3", "failed_later_pass": "up to pass 2",
            "sentence_without_a_run": "setup only: no pass is done", "job_began_after_the_status": "up to pass 3"}


# --- a folder that is not the case: the reply of before ---------------------------------------------

@pytest.mark.parametrize("kind", NOT_THE_CASE)
def test_a_folder_that_is_not_the_case_gets_the_reply_of_before(kind, tmp_path, monkeypatch):
    live = check_no_mark(build(tmp_path / "run", kind), monkeypatch)
    st = live["run_status"]
    assert st["failure"]["run_data"] == RUN_DATA[kind]
    if kind in ("failed_resume", "failed_later_pass", "job_began_after_the_status"):    # the facts of the folder stay at the top level
        assert st["last_history"] and st["checkpoint"] and st["history_facts"] == []
        assert st["options"][0].startswith("run_topopt with resume=true (yaml_path: checkpoint.yaml_path")
        assert live["run_result"]["checkpoint"] == st["checkpoint"]


# --- the case of the card ------------------------------------------------------------------------

def test_run_status_after_a_refused_run_says_which_facts_are_of_the_earlier_run(tmp_path):
    folder = build(tmp_path / "run", "refused")
    st = server.run_status(str(folder))
    assert st["state"] == "error" and st["status"]["message"].endswith(EARLIER_RUN_KEPT)
    assert st["job"]["state"] == "failed" and st["job"]["exit_code"] == 1
    # one place, near the top, that an agent reads first
    assert list(st)[:5] == ["run_dir", "run_dir_rel", "dashboard_url", "state", "earlier_run"]
    earlier = st["earlier_run"]
    assert "this job did not change it" in earlier["note"] and "continues the earlier run" in earlier["note"]
    assert earlier["last_history"]["iteration"] == 3 and earlier["verdict"]["state"] == "iteration_limit"
    assert (earlier["checkpoint"]["iteration"], earlier["checkpoint"]["complete"]) == (3, True)
    assert earlier["summary"] == "pass 3 of 3, volume 29.7%, verdict iteration_limit, checkpoint at pass 3 of 3 (complete)"
    # the refused job has no data of its own: the data keys are null, as for any job that wrote none
    assert st["last_history"] is None and st["verdict"] is None and st["checkpoint"] is None and st["history_facts"] == []
    # the summary does not give the earlier pass count, volume or verdict as the result of the refused job
    before, marker, after = st["summary"].partition("; earlier run in this folder (this job did not change it; see `earlier_run`): ")
    assert marker and "(iteration 0, 00:00)" in before and "pass 3" not in before and "volume 29" not in before
    assert "verdict:" not in st["summary"] and earlier["verdict"]["text"] not in st["summary"]
    assert after.startswith("pass 3 of 3, volume 29.7%, verdict iteration_limit")
    # the failure facts and the notes name the earlier run
    assert st["failure"]["run_data"] == ("setup only: no pass is done (the passes in the folder, up to pass 3, are of an earlier run: "
                                         "see earlier_run)")
    assert "passes of the earlier run in the folder (up to pass 3)" in st["dashboard_note"] and "saved passes" not in st["dashboard_note"]
    # the resume option continues the earlier run, and says from which pass
    first = st["options"][0]
    assert first.startswith("run_topopt with resume=true (yaml_path: earlier_run.checkpoint.yaml_path, out_dir: run_dir)")
    assert "earlier run" in first and "not the refused job" in first and "checkpoint of the earlier run at pass 3 is complete" in first
    assert "first option: " + first in st["summary"]
    assert earlier["checkpoint"]["yaml_path"].endswith("case/c.yaml")           # the path that the option names exists in the reply
    assert not any("checkpoint.yaml_path" in o.replace("earlier_run.checkpoint.yaml_path", "") for o in st["options"])


def test_an_earlier_run_that_did_not_finish_is_continued_from_its_pass(tmp_path):
    folder = build(tmp_path / "run", "refused_unfinished")
    st = server.run_status(str(folder))
    earlier = st["earlier_run"]
    assert earlier["verdict"] is None and earlier["checkpoint"]["complete"] is False
    assert earlier["summary"] == "pass 2 of 6, volume 29.8%, checkpoint at pass 2 of 6 (not complete)"
    assert st["options"][0].endswith("continues the earlier run (not the refused job) from its checkpoint at pass 2 of 6")
    assert "; checkpoint at pass 2 of 6: run_topopt with resume=true continues the earlier run" in st["summary"]
    assert "continues the run" not in st["summary"] and "first option" not in st["summary"]      # the summary says it once
    assert st["failure"]["run_data"].startswith("setup only: no pass is done (the passes in the folder, up to pass 2,")


def test_run_result_after_a_refused_run_says_it_too(tmp_path):
    folder = build(tmp_path / "run", "refused")
    st, res = server.run_status(str(folder)), server.run_result(str(folder))
    assert res["done"] is False and "metrics" not in res and "final_stl" not in res      # the files of the earlier run are not this job's result
    assert res["message"] == f"The run failed: {GUARD_TEXT} {EARLIER_RUN_KEPT}"
    assert list(res)[:3] == ["done", "state", "earlier_run"]
    for key in ("earlier_run", "summary", "options", "failure", "dashboard_note"):
        assert res[key] == st[key], key
    assert res["checkpoint"] is None
    assert "verdict:" not in res["summary"] and res["options"][0] == st["options"][0] and "earlier run" in res["options"][0]


def test_every_fact_of_the_earlier_run_stays_in_the_reply(tmp_path):
    """`earlier_run` holds what the old reply had at the top level for the same files (the case `failed_resume` has the same earlier run)."""
    old = server.run_status(str(build(tmp_path / "old", "failed_resume")))
    new = server.run_status(str(build(tmp_path / "run", "refused")))["earlier_run"]
    for key in ("last_history", "verdict", "checkpoint", "history_facts"):
        assert new[key] == old[key], key
    assert old["verdict"] and old["last_history"] and old["checkpoint"]                     # the facts are there to compare


def test_a_log_of_the_earlier_run_is_not_the_log_of_the_refused_job(tmp_path):
    """The refused job writes no line to `run.log`. If that file is as new as the job, it is still the log of the earlier run."""
    folder = build(tmp_path / "run", "refused")
    os.utime(folder / "run.log", (JOB_AT + 1, JOB_AT + 1))
    st = server.run_status(str(folder))
    assert "log_tail" not in st and "done in 1.5 s" not in json.dumps(st)
    assert st["failure"]["memory_estimate"]["gib"] is None
    plain = build(tmp_path / "plain", "failed_resume")
    os.utime(plain / "run.log", (JOB_AT + 1, JOB_AT + 1))
    assert "done in 1.5 s" in server.run_status(str(plain))["log_tail"]          # a job that is not refused: its log is run.log


# --- the mark is off ----------------------------------------------------------------------------

def test_a_job_that_runs_in_the_folder_has_no_mark(tmp_path):
    """A new job takes the folder after the refused one. Its record says `running`, and `status.json` is still the refused one.

    The status file is dated after the start of the job here, so that only the state `running` can switch the mark off.
    """
    folder = build(tmp_path / "run", "refused")
    later = (folder / "status.json").stat().st_mtime + 10 * 365 * 86400
    os.utime(folder / "status.json", (later, later))
    job = RunJob(folder)
    job.start()
    try:
        st, res = server.run_status(str(folder)), server.run_result(str(folder))
        assert st["job"]["state"] == "running"
        assert "earlier_run" not in st and "earlier_run" not in res and detected(folder) is False
        assert st["verdict"] is not None and st["last_history"]["iteration"] == 3        # the reply of before
    finally:
        job.finish(1)
    assert "earlier_run" in server.run_status(str(folder))        # the job ended with a failure, and its status is the refused one: the mark is back


def test_the_mark_needs_the_status_of_a_refused_job(tmp_path):
    """The same sentence in a status that is not the refused one (pass above 0, or another step) does not count."""
    for iteration, step in ((2, "solve"), (0, "starting")):
        folder = build(tmp_path / f"run-{iteration}-{step}", "refused")
        write_status(folder, f"{GUARD_TEXT} {EARLIER_RUN_KEPT}", iteration=iteration, step=step)
        assert detected(folder) is False and "earlier_run" not in server.run_status(str(folder))


# --- real files from real runs ----------------------------------------------------------------------

def refuse_through_mcp(work, server_state, monkeypatch):
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -m topopt run {{yaml}} --out {{out}} --quiet")
    edit(work.yaml, lambda d: d["settings"].update(max_active_voxels=10))
    r = server_state.run_topopt(str(work.yaml), str(work.out))
    assert "error" not in r, r
    server_state._current["proc"].wait(timeout=120)
    work.check_kept()


def test_the_mcp_path_of_the_card(work, server_state, monkeypatch):
    """`run_topopt` into a finished folder with a guard error, then `run_status` and `run_result`: real files of a real run."""
    refuse_through_mcp(work, server_state, monkeypatch)
    st, res = server_state.run_status(str(work.out)), server_state.run_result(str(work.out))
    assert st["state"] == "error" and st["process_alive"] is False and st["job"]["exit_code"] == 1
    assert "max_active_voxels" in st["summary"] and EARLIER_RUN_KEPT in st["summary"]
    earlier = st["earlier_run"]
    assert earlier["last_history"]["iteration"] == 3 and earlier["verdict"]["state"] == "iteration_limit"
    assert earlier["checkpoint"]["iteration"] == 3 and earlier["checkpoint"]["yaml_path"].endswith("/case/c.yaml")
    assert st["verdict"] is None and st["checkpoint"] is None and st["last_history"] is None
    before, _, after = st["summary"].partition("earlier run in this folder")
    assert "pass 3" not in before and "iteration limit reached" not in st["summary"] and "pass 3 of 3" in after
    assert st["failure"]["run_data"].startswith("setup only: no pass is done") and "up to pass 3" in st["failure"]["run_data"]
    assert "log_tail" in st and "max_active_voxels" in st["log_tail"]           # the output file of the refused job, not the run.log of the earlier run
    assert "earlier run" in st["options"][0] and "not the refused job" in st["options"][0]
    assert res["done"] is False and "metrics" not in res and res["earlier_run"] == earlier and res["summary"] == st["summary"]


def test_a_refused_run_in_an_empty_folder_has_no_mark(finished, tmp_path, capsys, monkeypatch):
    case = tmp_path / "case"
    shutil.copytree(finished[0], case)
    edit(case / "c.yaml", lambda d: d["settings"].update(max_active_voxels=10))
    out = tmp_path / "run"
    assert main(run_args(case / "c.yaml", out)) == 1
    st = check_no_mark(out, monkeypatch)["run_status"]
    assert st["state"] == "error" and st["failure"]["run_data"] == "setup only: no pass is done" and "earlier run" not in st["summary"]


def test_a_resume_that_failed_has_no_mark(work, monkeypatch):
    def boom(problem, out, plan=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(optimize_module, "run", boom)
    assert main(work.args("--resume")) == 2
    st = check_no_mark(work.out, monkeypatch)["run_status"]
    assert status_of(work.out)["message"] == "internal error: boom"
    assert st["checkpoint"]["iteration"] == 3 and st["last_history"]["iteration"] == 3        # the files of the run that the resume went on with
    assert status_of(work.out)["iteration"] == 0 and status_of(work.out)["step"] == "setup"        # the pass and the step alone do not tell the case
    assert st["failure"]["run_data"] == "up to pass 3" and "(iteration 0/3," in st["summary"] and "volume 29" in st["summary"]
    assert st["options"][0].startswith("run_topopt with resume=true (yaml_path: checkpoint.yaml_path")


def test_a_run_that_failed_at_a_later_pass_has_no_mark(work, monkeypatch):
    """A fresh run that the checks let through deletes the earlier run. It fails at pass 2: the files in the folder are its own."""
    real = snapshot_module.RunWriter.add_history

    def add_history(self, rec):
        real(self, rec)
        if rec["iteration"] == 2:
            raise RuntimeError("boom at pass 2")

    monkeypatch.setattr(snapshot_module.RunWriter, "add_history", add_history)
    assert main(work.args()) == 2
    st = check_no_mark(work.out, monkeypatch)["run_status"]
    assert status_of(work.out)["message"] == "internal error: boom at pass 2" and EARLIER_RUN_KEPT not in json.dumps(st)
    assert st["last_history"]["iteration"] == 2 and st["failure"]["run_data"] == "up to pass 2"
    assert st["options"][0].startswith("run_topopt with resume=true (yaml_path: checkpoint.yaml_path")
