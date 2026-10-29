"""A run that ended with no result: run_status and run_result say what is known and what the user can do (P9-6, CONTRACTS.md section 10).

The run processes are small stand-ins that the test hook TOPOPT_MCP_CMD starts: a shell that kills itself, a shell that exits with an error,
a Python script that takes the job lock, writes some run data and kills itself. The server code is the real one.
"""
import asyncio
import json
import os
import re
import shlex
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt.runjob import INFO_FILE, LOCK_FILE  # noqa: E402
from topopt_mcp import server  # noqa: E402

FAKE_CMD = f"{sys.executable} {REPO}/tests/serve_after_fake.py {{out}} --spi 0.2 --iterations 6"

# The key sets of the replies of a run that did not fail: the new keys are only in the reply of a run that ended with no result.
STATUS_KEYS = {"run_dir", "run_dir_rel", "dashboard_url", "state", "process_alive", "status", "last_history", "job", "history_facts",
               "verdict", "checkpoint", "stop_request", "summary",
               "time_left", "time_left_note"}   # E1-P2 (CONTRACTS.md section 24.5): in every full reply, null when there is no estimate
EARLY_STATUS_KEYS = {"run_dir", "run_dir_rel", "state", "process_alive", "job", "history_facts", "verdict", "checkpoint", "stop_request",
                     "summary"}
# report_md and report_md_rel: the part card of P9-7a (CONTRACTS section 9.2); run_result writes it again before it answers.
DONE_RESULT_KEYS = {"done", "metrics", "final_stl", "final_stl_rel", "run_dir", "run_dir_rel", "workspace", "levers", "preview",
                    "report_md", "report_md_rel"}
FAILURE_KEYS = {"pid", "exit_code", "signal", "run_data", "stopped_by_this_server", "peak_memory", "peak_memory_bytes", "memory_estimate", "ram"}
NEW_KEYS = {"log_tail", "log_tail_file", "failure", "options", "possible_causes", "possible_causes_note", "dashboard_note"}


@pytest.fixture(autouse=True)
def clean_server():
    server._current.clear()
    server._stopped.clear()
    yield
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._current.clear()
    server._stopped.clear()


@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("name: x\n")
    return p


def start(monkeypatch, yaml_file, out, command):
    """Start the run (a stand-in command) through the server. Return the reply of run_topopt."""
    monkeypatch.setenv("TOPOPT_MCP_CMD", command)
    reply = server.run_topopt(str(yaml_file), str(out))
    assert "error" not in reply, reply
    return reply


def start_and_wait(monkeypatch, yaml_file, out, command):
    reply = start(monkeypatch, yaml_file, out, command)
    server._current["proc"].wait(timeout=60)
    return reply


def sh(script):
    return "sh -c " + shlex.quote(script)


def write_script(tmp_path, body, name="stand_in.py"):
    path = tmp_path / name
    path.write_text(body)
    return f"{sys.executable} {path} {{out}}"


# The memory cause says how much RAM is free now. The server reads that number again at each call, so two replies of one run can
# differ in this number and nothing else (seen: 50.1 and 50.2 GiB). A test that compares the causes of two replies masks it.
FREE_RAM = re.compile(r"\d+(?:\.\d+)? GiB available now")


def causes_without_free_ram(causes):
    """The causes as they are, with the number before "GiB available now" masked. Text and order stay as they are."""
    return [FREE_RAM.sub("<free> GiB available now", c) for c in causes]


# A run process that takes the job lock, writes status and history of 3 passes, and kills itself in the middle of pass 3.
MID_RUN = '''
import json, os, signal, sys
from pathlib import Path
from topopt.runjob import RunJob
out = Path(sys.argv[1])
job = RunJob(out)
job.start()
rows = [{"iteration": i, "compliance": 10.0 - i, "volume_fraction": 0.4} for i in (1, 2, 3)]
(out / "history.jsonl").write_text("".join(json.dumps(r) + "\\n" for r in rows))
(out / "run.json").write_text(json.dumps({"iterations": 24}))
(out / "status.json").write_text(json.dumps({"state": "running", "iteration": 3, "step": "solve", "elapsed_s": 1.5, "message": ""}))
print("memory estimate: about 2.5 GiB of 8.0 GiB RAM (the filter is part of it)", flush=True)
print("pass 3 solve", flush=True)
os.kill(os.getpid(), signal.SIGKILL)
'''

# A run process that fails inside Python: the error status, the job record with the exit code and the peak memory, exit 2.
ERROR_EXIT = '''
import json, sys
from pathlib import Path
from topopt.runjob import RunJob
out = Path(sys.argv[1])
job = RunJob(out)
job.start()
rows = [{"iteration": i, "compliance": 10.0 - i, "volume_fraction": 0.4} for i in (1, 2)]
(out / "history.jsonl").write_text("".join(json.dumps(r) + "\\n" for r in rows))
(out / "run.json").write_text(json.dumps({"iterations": 24}))
(out / "status.json").write_text(json.dumps({"state": "error", "iteration": 2, "step": "solve", "elapsed_s": 1.5, "message": "internal error: boom"}))
print("Traceback (most recent call last):", flush=True)
print("RuntimeError: boom", flush=True)
job.finish(2)
sys.exit(2)
'''


# --- a run that dies before it writes run data -------------------------------------------------

def test_early_kill_says_what_is_known_and_what_to_do(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("echo started; kill -9 $$"))
    st = server.run_status(str(out))
    assert st["state"] == "error" and st["process_alive"] is False
    assert st["summary"].startswith("error: the run process was stopped by signal 9 before it wrote any run data (see ")   # the old start stays
    assert "possible cause (the exit code does not say): a stop by the user or another program, or lack of memory" in st["summary"]
    assert "first option: check_problem" in st["summary"]
    assert st["log_tail"] == "started" and st["log_tail_file"].endswith("run.stdout")
    f = st["failure"]
    assert set(f) == FAILURE_KEYS and f["pid"] == server._current["proc"].pid
    assert (f["exit_code"], f["signal"], f["run_data"]) == (-9, "SIGKILL", "none")
    assert f["stopped_by_this_server"] is False
    assert f["peak_memory"] == "not recorded: the process did not reach its end" and f["peak_memory_bytes"] is None
    assert f["memory_estimate"]["gib"] is None and "no 'memory estimate' line" in f["memory_estimate"]["reason"]
    assert set(f["ram"]) == {"total_gib", "available_now_gib", "note"}
    causes = st["possible_causes"]
    assert causes[0].startswith("a stop by the user or another program")   # no positive memory fact: the memory cause follows (F4)
    assert causes[1].startswith("the system stopped it for lack of memory") and "can be lower than the real peak" in causes[1]
    assert "a stop by this server" not in " ".join(causes)        # this server has not stopped it: the stop is not a possible cause
    assert "order of this list is not a ranking" in st["possible_causes_note"]
    opts = st["options"]
    assert opts[0].startswith("check_problem") and "voxel_size" in opts[1] and "solver with less memory" in opts[2]
    assert not any("resume=true" in o for o in opts)      # no checkpoint
    assert "ran inside the run process, which has ended" in st["dashboard_note"] and "saved no pass" in st["dashboard_note"]
    # run_result gives the same facts, not "No result yet"
    res = server.run_result(str(out))
    assert res["done"] is False and res["state"] == "error" and "No result yet" not in json.dumps(res)
    assert res["summary"] == st["summary"] and res["log_tail"] == "started"
    assert {k: v for k, v in res["failure"].items() if k != "ram"} == {k: v for k, v in f.items() if k != "ram"}
    assert res["options"] == st["options"] and "dashboard_note" in res
    # the same causes in the same order; only the free RAM can differ (the server reads it again at each call)
    assert causes_without_free_ram(res["possible_causes"]) == causes_without_free_ram(st["possible_causes"])


def test_two_replies_of_one_run_give_the_same_causes_when_the_free_ram_moves(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("kill -9 $$"))
    calls = []

    def moving_ram():       # a busy host: the room now is 0.1 GiB less at each call
        calls.append(1)
        return {"total_gib": 16.0, "available_now_gib": round(8.0 - 0.1 * len(calls), 1), "note": "n"}

    monkeypatch.setattr(server, "_ram_now", moving_ram)
    st = server.run_status(str(out))
    res = server.run_result(str(out))
    assert len([c for c in st["possible_causes"] if "GiB available now" in c]) == 1       # the memory cause holds the number
    assert res["possible_causes"] != st["possible_causes"]          # "available now" is the room now: it moved between the calls
    assert causes_without_free_ram(res["possible_causes"]) == causes_without_free_ram(st["possible_causes"])


def test_the_mask_of_the_free_ram_hides_that_number_and_nothing_else():
    user = "a stop by the user or another program (a force quit, or `kill`)"
    memory = ("the system stopped it for lack of memory (the log has no memory estimate; this machine has 64.0 GiB RAM, "
              "50.1 GiB available now; an estimate is a model and can be lower than the real peak)")
    base = causes_without_free_ram([user, memory])
    assert base[1].count("<free> GiB available now") == 1 and "50.1" not in " ".join(base)
    assert causes_without_free_ram([user, memory.replace("50.1 GiB available now", "50.2 GiB available now")]) == base
    assert causes_without_free_ram([user, memory.replace("50.1 GiB available now", "9 GiB available now")]) == base
    # every other difference stays visible
    assert causes_without_free_ram([user, memory.replace("64.0 GiB RAM", "32.0 GiB RAM")]) != base            # the total RAM does not move
    assert causes_without_free_ram([user, memory.replace("no memory estimate", "a memory estimate of 3.2 GiB")]) != base
    assert causes_without_free_ram([user, memory.replace("no memory estimate", "a memory estimate of 3.2 GiB")]) != \
        causes_without_free_ram([user, memory.replace("no memory estimate", "a memory estimate of 3.3 GiB")])  # other GiB numbers are not masked
    assert causes_without_free_ram([user + " or a crash", memory]) != base          # another text
    assert causes_without_free_ram([memory, user]) != base                          # another order
    assert causes_without_free_ram([user]) != base                                  # one cause less


def test_the_memory_estimate_comes_from_the_log_of_the_run(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out,
                   sh("echo 'memory estimate: about 3.2 GiB of 15.6 GiB RAM (the filter is part of it)'; kill -9 $$"))
    st = server.run_status(str(out))
    est = st["failure"]["memory_estimate"]
    assert (est["gib"], est["ram_gib"]) == (3.2, 15.6) and est["line"].startswith("memory estimate: about 3.2 GiB") and est["source"].endswith(".stdout")
    assert any("memory estimate is 3.2 GiB" in c for c in st["possible_causes"])
    assert "lack of memory (estimate 3.2 GiB" in st["summary"]


def test_the_memory_estimate_pattern_matches_the_line_that_topopt_run_writes(tmp_path):
    """The reply reads the estimate from the text of `cli.run_memory_lines`: a change of that text must show here."""
    from test_optimize import make_case
    from topopt.cli import estimate_memory_bytes, run_memory_lines
    from topopt.config import load_config
    from topopt.members import filter_size
    from topopt.optimize import reduced_system
    from topopt.problem import build_problem
    prob = build_problem(load_config(make_case(tmp_path, h=2.0)))
    _edof, _active, free = reduced_system(prob)
    fsz = filter_size(float(prob.config.settings.filter_radius), float(prob.h), int((prob.labels == 1).sum()), prob.shape)
    line = run_memory_lines(prob, {"size": fsz})[0]
    m = server._ESTIMATE_LINE.match(line)
    assert m, line
    want = estimate_memory_bytes(int(np.prod(prob.shape)), int(free.size), fsz["held_bytes_estimate"]) / 2**30
    assert abs(float(m.group(1)) - want) <= 0.05 + 1e-9


def test_exit_1_with_a_message_gives_the_message_and_no_memory_cause(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("echo 'ERROR: boom' 1>&2; exit 1"))
    st = server.run_status(str(out))
    assert st["state"] == "error"
    assert "ERROR: boom" in st["log_tail"]
    assert (st["failure"]["exit_code"], st["failure"]["signal"]) == (1, None)
    assert "possible_causes" not in st                                  # no signal: nothing to guess
    assert not any("voxel_size" in o or "less memory" in o for o in st["options"])
    assert st["options"][0].startswith("read `log_tail`")
    assert 'last log line: "ERROR: boom"' in st["summary"]
    assert st["failure"]["peak_memory"].startswith("not recorded")
    res = server.run_result(str(out))
    assert res["done"] is False and "ERROR: boom" in res["log_tail"] and "possible_causes" not in res


def test_a_run_that_fails_in_python_keeps_its_message_and_has_the_recorded_peak(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, write_script(tmp_path, ERROR_EXIT))
    st = server.run_status(str(out))
    assert st["state"] == "error" and "internal error: boom" in st["summary"]
    f = st["failure"]
    assert (f["exit_code"], f["signal"], f["run_data"]) == (2, None, "up to pass 2")
    assert f["peak_memory_bytes"] > 0 and " GB (" in f["peak_memory"] and " GiB; job.json" in f["peak_memory"]     # both units (F11)
    assert "possible_causes" not in st
    assert f"`topopt dashboard {out}` shows the saved passes (up to pass 2)" in st["dashboard_note"]
    assert "RuntimeError: boom" in st["log_tail"]
    res = server.run_result(str(out))
    assert res["message"] == "The run failed: internal error: boom"      # the old text stays; the facts are added
    assert res["failure"]["exit_code"] == 2 and res["done"] is False


def test_a_fault_in_the_failure_facts_is_a_fact_and_the_reply_stays(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("exit 1"))

    def broken(*a, **k):
        raise RuntimeError("no log today")
    monkeypatch.setattr(server, "_run_logs", broken)
    st = server.run_status(str(out))
    assert st["state"] == "error" and st["summary"].startswith("error: the run process ended with exit 1 before it wrote any run data")
    assert st["failure"] == {"error": "the failure facts could not be read: RuntimeError: no log today"}
    assert server.run_result(str(out))["failure"]["error"].startswith("the failure facts could not be read")


# --- a kill in the middle of a run ---------------------------------------------------------------

def test_a_kill_in_the_middle_of_a_run_of_this_server(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, write_script(tmp_path, MID_RUN))
    st = server.run_status(str(out))
    assert st["state"] == "running" and st["process_alive"] is False
    assert "(warning: the process has stopped)" in st["summary"]
    f = st["failure"]
    assert (f["exit_code"], f["signal"], f["run_data"]) == (-9, "SIGKILL", "up to pass 3")
    assert f["peak_memory"] == "not recorded: the process did not reach its end"
    assert (f["memory_estimate"]["gib"], f["memory_estimate"]["ram_gib"]) == (2.5, 8.0)
    assert "pass 3 solve" in st["log_tail"]
    assert f"`topopt dashboard {out}` shows the saved passes (up to pass 3)" in st["dashboard_note"]   # the dashboard ended with the process
    assert any("lack of memory" in c for c in st["possible_causes"])
    res = server.run_result(str(out))
    assert res["done"] is False and res["failure"]["signal"] == "SIGKILL" and res["state"] == "running"
    assert res["message"].startswith("The run process has stopped while the status of the run was `running`: there is no result.")


def write_checkpoint(folder, iteration, iterations, complete):
    meta = {"iteration": iteration, "n_iterations": iterations, "complete": complete, "format": 1}
    np.savez(folder / "checkpoint.npz", meta_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))


def dead_run_folder(tmp_path, yaml_file, *, complete=False, job_extra=None):
    """A run folder of a process that was killed in the middle of pass 6 (24 passes), seen by a server that did not start it."""
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    folder = tmp_path / "old_run"
    folder.mkdir()
    rec = {"format": 1, "pid": p.pid, "process_start": 1.0, "token": "t", "started_at": "2026-10-20T01:02:03+00:00",
           "run_dir": str(folder)}
    (folder / LOCK_FILE).write_text(json.dumps(rec))
    (folder / INFO_FILE).write_text(json.dumps(dict(rec, state="running", success_codes=[0], notes=[], peak_rss_start_bytes=100_000_000,
                                                    **(job_extra or {}))))
    (folder / "run.json").write_text(json.dumps({"iterations": 24, "yaml_path": str(yaml_file)}))
    (folder / "status.json").write_text(json.dumps({"state": "running", "iteration": 6, "step": "solve", "elapsed_s": 9.0, "message": ""}))
    rows = [{"iteration": i, "compliance": 10.0 - i, "volume_fraction": 0.4} for i in range(1, 7)]
    (folder / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_checkpoint(folder, 6 if not complete else 24, 24, complete)
    return folder


def test_a_dead_run_with_a_checkpoint_offers_the_resume_first(tmp_path, yaml_file):
    folder = dead_run_folder(tmp_path, yaml_file)
    st = server.run_status(str(folder))
    assert st["state"] == "running" and st["process_alive"] is False and st["job"]["state"] == "stale_lock"
    assert "checkpoint at pass 6 of 24: run_topopt with resume=true continues the run" in st["summary"]
    assert st["summary"].count("resume=true") == 1                      # the clause does not repeat the resume text
    assert st["options"][0].startswith("run_topopt with resume=true (yaml_path: checkpoint.yaml_path, out_dir: run_dir)")
    assert "checkpoint at pass 6 of 24" in st["options"][0] and st["checkpoint"]["yaml_path"] == str(yaml_file)
    f = st["failure"]
    assert f["peak_memory"] == "not recorded: the process did not reach its end" and f["peak_memory_bytes"] is None
    assert (f["exit_code"], f["signal"], f["run_data"], f["stopped_by_this_server"]) == (None, None, "up to pass 6", None)
    assert "not known" in f["exit_note"]
    assert f["memory_estimate"]["gib"] is None and "no log" in f["memory_estimate"]["reason"]
    causes = " | ".join(st["possible_causes"])         # no exit code, no record of the end: the same list as for a signal
    assert "lack of memory" in causes and "the user or another program" in causes
    assert "a stop by this server" not in causes         # nothing says that this server stopped it (F5)
    assert "log_tail" not in st                         # no log file of this run
    res = server.run_result(str(folder))
    assert res["done"] is False and res["options"][0] == st["options"][0] and res["checkpoint"]["iteration"] == 6


def test_a_complete_checkpoint_offers_the_post_step(tmp_path, yaml_file):
    folder = dead_run_folder(tmp_path, yaml_file, complete=True)
    st = server.run_status(str(folder))
    assert "runs the post step only" in st["options"][0]


def test_the_run_log_of_an_older_run_is_not_the_log_of_this_one(tmp_path, yaml_file):
    folder = dead_run_folder(tmp_path, yaml_file)
    (folder / "run.log").write_text("memory estimate: about 9.9 GiB of 15.6 GiB RAM (x)\n")
    old = time.time() - 10 * 86400                                      # the job started on 2026-10-20: this log is older
    os.utime(folder / "run.log", (old, old))
    st = server.run_status(str(folder))
    assert st["failure"]["memory_estimate"]["gib"] is None and "log_tail" not in st
    now = time.time()
    os.utime(folder / "run.log", (now, now))                             # written after the job started
    st = server.run_status(str(folder))
    assert st["failure"]["memory_estimate"]["gib"] == 9.9 and "memory estimate: about 9.9 GiB" in st["log_tail"]
    assert st["log_tail_file"].endswith("run.log")


# --- a stop by this server -------------------------------------------------------------------------

def test_a_stop_by_this_server_is_a_fact_and_a_possible_cause(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start(monkeypatch, yaml_file, out, f"{sys.executable} -c 'import time; time.sleep(60)'")
    proc = server._current["proc"]
    server._stop(out, proc.pid)
    st = server.run_status(str(out))
    f = st["failure"]
    assert (f["exit_code"], f["signal"], f["stopped_by_this_server"]) == (-15, "SIGTERM", True)
    assert f["stopped_at"].endswith("Z")
    assert len(st["possible_causes"]) == 1 and st["possible_causes"][0].startswith("a stop by this server at ")
    assert "possible cause: a stop by this server" in st["summary"]
    assert not any("voxel_size" in o for o in st["options"])             # the stop explains the exit: no memory option
    assert server.run_result(str(out))["failure"]["stopped_by_this_server"] is True


def test_a_stop_that_ends_in_sigkill_is_not_called_a_memory_stop(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    code = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('MemoryError: Unable to allocate 1.0 GiB', flush=True); time.sleep(60)")   # the log names memory, and this server stops it
    start(monkeypatch, yaml_file, out, f"{sys.executable} -c {shlex.quote(code)}")
    for _ in range(100):
        if "MemoryError" in server._tail(server._log_path(out)):
            break
        time.sleep(0.1)
    server._stop(out, server._current["proc"].pid)                       # SIGTERM is ignored: after 3 s, SIGKILL
    st = server.run_status(str(out))
    assert (st["failure"]["signal"], st["failure"]["stopped_by_this_server"]) == ("SIGKILL", True)
    assert not any("lack of memory" in c for c in st["possible_causes"])
    assert not any("voxel_size" in o or "less memory" in o or "kernel log" in o for o in st["options"])   # the stop explains the exit
    assert "possible_causes_note" not in st                                                           # a known stop is not a guess


def test_the_stop_mark_is_for_one_pid(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start(monkeypatch, yaml_file, out, f"{sys.executable} -c 'import time; time.sleep(60)'")
    old_pid = server._current["proc"].pid
    server._stop(out, old_pid)
    start_and_wait(monkeypatch, yaml_file, out, sh("kill -9 $$"))        # a new run in the same folder, killed from outside
    assert old_pid in server._stopped
    st = server.run_status(str(out))
    assert st["failure"]["signal"] == "SIGKILL" and st["failure"]["stopped_by_this_server"] is False
    assert "a stop by this server" not in " ".join(st["possible_causes"])


# --- the files of an older run in the folder (P9-6 review F6) ---------------------------------------------

def age(folder, seconds=3600):
    """Make every file of the folder older, as the files of a run that ended an hour ago."""
    old = time.time() - seconds
    for f in folder.rglob("*"):
        os.utime(f, (old, old))


def old_run_folder(tmp_path, state, message=""):
    folder = tmp_path / "run"
    (folder / "result").mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps({"state": state, "iteration": 3, "step": state, "elapsed_s": 1.0, "message": message}))
    (folder / "result" / "metrics.json").write_text(json.dumps({"mass_final_g": 1.0, "note": "the OLD run"}))
    (folder / "result" / "final.stl").write_text("solid x\nendsolid x\n")
    age(folder)
    return folder


def test_a_new_run_that_dies_over_an_old_error_status_does_not_return_the_old_message(monkeypatch, yaml_file, tmp_path):
    folder = old_run_folder(tmp_path, "error", "STL file not found: old.stl (the OLD run)")
    start_and_wait(monkeypatch, yaml_file, folder, sh("echo started; kill -9 $$"))
    res = server.run_result(str(folder))
    assert res["done"] is False and res["failure"]["signal"] == "SIGKILL"
    assert "OLD run" not in json.dumps(res) and not res["message"].startswith("The run failed")


def test_a_new_run_that_dies_over_an_old_done_result_does_not_return_the_old_metrics(monkeypatch, yaml_file, tmp_path):
    folder = old_run_folder(tmp_path, "done")
    start_and_wait(monkeypatch, yaml_file, folder, sh("echo started; kill -9 $$"))
    st, res = server.run_status(str(folder)), server.run_result(str(folder), preview=False)
    assert st["state"] == "error" and "failure" in st
    assert res["done"] is False and "metrics" not in res and "final_stl" not in res and res["failure"]["signal"] == "SIGKILL"


def test_a_new_run_that_works_over_an_old_done_result_is_not_done(monkeypatch, yaml_file, tmp_path):
    folder = old_run_folder(tmp_path, "done")
    start(monkeypatch, yaml_file, folder, sh("sleep 30"))
    assert server.run_status(str(folder))["state"] == "starting"
    assert server.run_result(str(folder), preview=False) == {"done": False, "message": "The run is not done yet. Call run_status."}


# --- the facts are true for a run that this session did not start (P9-6 review F1, F2) -------------------------

# A server run that is done and waits (the dashboard), with its own lines in the output file of this server.
DONE_RUN = '''
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
print("memory estimate: about 0.1 GiB of 64.0 GiB RAM (run A of the server)", flush=True)
print("done in 1.0 s (run A of the server)", flush=True)
(out / "status.json").write_text(json.dumps({"state": "done", "iteration": 3, "step": "done", "elapsed_s": 1.0, "message": ""}))
time.sleep(120)
'''

# A run of the CLI: no pid file and no output file of this server. It takes the job lock, writes run.log and is killed in pass 2.
CLI_RUN = '''
import json, os, signal, sys
from pathlib import Path
from topopt.runjob import RunJob
out = Path(sys.argv[1])
for name in ("status.json", "history.jsonl", "run.log"):
    (out / name).unlink(missing_ok=True)
job = RunJob(out)
job.start()
(out / "run.log").write_text("topopt run: x\\nmemory estimate: about 30.0 GiB of 64.0 GiB RAM (the CLI run)\\npass 2 solve\\n")
(out / "history.jsonl").write_text("".join(json.dumps({"iteration": i, "compliance": 1.0, "volume_fraction": 0.4}) + "\\n" for i in (1, 2)))
(out / "run.json").write_text(json.dumps({"iterations": 24}))
(out / "status.json").write_text(json.dumps({"state": "running", "iteration": 2, "step": "solve", "elapsed_s": 3.0, "message": ""}))
os.kill(os.getpid(), signal.SIGKILL)
'''


def test_a_cli_run_after_a_server_run_that_this_server_stopped_has_the_facts_of_its_own_run(monkeypatch, yaml_file, tmp_path):
    import subprocess
    x, y = tmp_path / "runs" / "live", tmp_path / "runs" / "other"
    start(monkeypatch, yaml_file, x, write_script(tmp_path, DONE_RUN, "done_run.py"))
    pid_a = server._current["proc"].pid
    for _ in range(100):
        if (x / "status.json").is_file():
            break
        time.sleep(0.1)
    start(monkeypatch, yaml_file, y, f"{sys.executable} -c 'import time; time.sleep(60)'")      # run_topopt stops the process of run A
    assert pid_a in server._stopped and server._stopped[pid_a]["run_dir"] == str(x)
    age(x.parent, 3600)                                                           # run A ended long before the CLI run starts
    cli_file = tmp_path / "cli_run.py"
    cli_file.write_text(CLI_RUN)
    cli = subprocess.run([sys.executable, str(cli_file), str(x)], cwd=str(tmp_path), capture_output=True)
    assert cli.returncode == -9
    assert int(server._pid_file(x).read_text()) == pid_a                          # the pid file of this server still names run A
    st = server.run_status(str(x))
    f = st["failure"]
    job_pid = st["job"]["pid"]
    assert job_pid != pid_a and f["pid"] == job_pid                                # the pid of the run that ended (F1)
    assert f["stopped_by_this_server"] is None and "stopped_at" not in f           # this server never signalled it
    assert not any("a stop by this server" in c for c in st["possible_causes"])
    assert any("voxel_size" in o for o in st["options"])                           # memory is still a possible cause
    assert "run A" not in st["log_tail"] and "the CLI run" in st["log_tail"] and st["log_tail_file"].endswith("run.log")   # F2
    assert f["memory_estimate"]["gib"] == 30.0 and f["memory_estimate"]["source"].endswith("run.log")


def test_the_output_file_of_this_server_counts_only_if_it_is_not_older_than_the_run(tmp_path, yaml_file):
    folder = dead_run_folder(tmp_path, yaml_file)
    out_file = server._log_path(folder)
    out_file.write_text("memory estimate: about 9.9 GiB of 64.0 GiB RAM (an older run)\n")
    old = time.time() - 10 * 86400                                                 # the job started on 2026-10-20
    os.utime(out_file, (old, old))
    st = server.run_status(str(folder))
    assert "log_tail" not in st and st["failure"]["memory_estimate"]["gib"] is None
    now = time.time()
    os.utime(out_file, (now, now))
    assert server.run_status(str(folder))["failure"]["memory_estimate"]["gib"] == 9.9
    (folder / "run.log").write_text("memory estimate: about 1.5 GiB of 64.0 GiB RAM (the run)\nlater line\n")
    os.utime(folder / "run.log", (now + 5, now + 5))                               # both count: the newer file comes first
    st = server.run_status(str(folder))
    assert st["log_tail_file"].endswith("run.log") and st["failure"]["memory_estimate"]["gib"] == 1.5
    assert st["log_tail"].endswith("later line") and "an older run" not in st["log_tail"]          # the text is of the file that comes first
    os.utime(out_file, (now + 9, now + 9))                                         # the output file is newer now
    st = server.run_status(str(folder))
    assert st["log_tail_file"].endswith(".stdout") and st["failure"]["memory_estimate"]["gib"] == 9.9
    os.utime(out_file, (now + 5, now + 5))                                         # the same time: the output file, which has stderr too
    assert server.run_status(str(folder))["log_tail_file"].endswith(".stdout")


def test_a_run_log_with_no_start_time_to_compare_is_not_used(tmp_path):
    folder = tmp_path / "run"                                                      # an error status of a run with no job record
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"state": "error", "iteration": 0, "step": "setup", "elapsed_s": 0.1, "message": "boom"}))
    (folder / "run.log").write_text("memory estimate: about 9.9 GiB of 8.0 GiB RAM (x)\n")
    st = server.run_status(str(folder))
    assert st["state"] == "error" and "failure" in st
    assert "log_tail" not in st and st["failure"]["memory_estimate"]["gib"] is None          # it cannot be told whether this log is of this run
    server._log_path(folder).write_text("started\n")                                  # the output file of this server counts
    assert server.run_status(str(folder))["log_tail"] == "started"


def test_a_resumed_log_has_two_estimates_and_the_last_one_of_each_kind_counts(monkeypatch, yaml_file, tmp_path):
    def row(gb):
        return f"buckling row: on in the YAML (min_factor 1, KS q 8, 4 modes): memory estimate about {gb} GB with the row (x)"
    out = tmp_path / "run"
    script = "; ".join([f"echo 'memory estimate: about 2.0 GiB of 8.0 GiB RAM (the first run)'", f"echo '{row('3.0')}'",
                        "echo 'resumed'", f"echo 'memory estimate: about 5.5 GiB of 8.0 GiB RAM (the resumed run)'", f"echo '{row('7.5')}'", "kill -9 $$"])
    start_and_wait(monkeypatch, yaml_file, out, sh(script))
    est = server.run_status(str(out))["failure"]["memory_estimate"]
    assert est["gib"] == 5.5 and est["line"].endswith("(the resumed run)")
    assert est["with_buckling_row"]["gb"] == 7.5 and "7.5 GB" in est["with_buckling_row"]["line"]


def test_a_stop_mark_counts_only_for_the_same_folder_and_not_before_the_run_start(tmp_path, yaml_file):
    folder = dead_run_folder(tmp_path, yaml_file)
    pid = json.loads((folder / INFO_FILE).read_text())["pid"]
    begin = server._iso_epoch("2026-10-20T01:02:03+00:00")
    server._stopped[pid] = {"run_dir": str(folder), "at": begin + 60}
    f = server.run_status(str(folder))["failure"]
    assert f["stopped_by_this_server"] is True and f["stopped_at"].endswith("Z")
    server._stopped[pid] = {"run_dir": str(folder), "at": begin - 3600}            # a stop of an earlier process with this pid
    assert server.run_status(str(folder))["failure"]["stopped_by_this_server"] is None
    server._stopped[pid] = {"run_dir": str(tmp_path / "elsewhere"), "at": begin + 60}
    assert server.run_status(str(folder))["failure"]["stopped_by_this_server"] is None


# --- the order of the causes and the options (P9-6 review F4) ---------------------------------------------------

def test_an_estimate_above_the_ram_that_the_run_saw_puts_the_memory_cause_first(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("echo 'memory estimate: about 20.0 GiB of 8.0 GiB RAM (x)'; kill -9 $$"))
    st = server.run_status(str(out))
    assert st["possible_causes"][0].startswith("the system stopped it for lack of memory")
    assert "the memory estimate of the run (20.0 GiB) is above the RAM that the run saw (8.0 GiB)" in st["possible_causes"][0]
    assert st["possible_causes"][1].startswith("a stop by the user or another program")
    assert "lack of memory (estimate of the run 20.0 GiB, above the RAM of the run), or a stop by the user" in st["summary"]
    assert "voxel_size" in st["options"][0] and st["options"][-1].startswith("check_problem")     # the memory options come first


def test_the_buckling_row_estimate_above_the_ram_counts_too(monkeypatch, yaml_file, tmp_path):
    row = ("buckling row: on in the YAML (min_factor 1, KS q 8, 4 modes): memory estimate about 12.0 GB with the row "
           "(the row adds about 10.0 GB; GB = 1e9 bytes). facts")
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh(f"echo 'memory estimate: about 2.0 GiB of 8.0 GiB RAM (x)'; echo '{row}'; kill -9 $$"))
    st = server.run_status(str(out))
    est = st["failure"]["memory_estimate"]
    assert est["gib"] == 2.0 and est["with_buckling_row"]["gb"] == 12.0 and est["with_buckling_row"]["line"] == row
    assert "the memory estimate with the buckling row (11.2 GiB) is above the RAM that the run saw (8.0 GiB)" in st["possible_causes"][0]


def test_the_buckling_row_pattern_matches_the_line_that_topopt_run_writes():
    from topopt.buckling_row import lines
    line = lines({"min_factor": 1.0, "ks_q": 8.0, "modes": 4, "memory_estimate_bytes": 12_000_000_000, "memory_extra_bytes": 10_000_000_000,
                  "facts": ["a fact."]})[0]
    m = server._ROW_LINE.match(line)
    assert m and float(m.group(1)) == 12.0, line


def test_memory_words_in_the_log_put_the_memory_cause_first(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("echo 'MemoryError: Unable to allocate 40.0 GiB for an array'; kill -9 $$"))
    st = server.run_status(str(out))
    assert st["possible_causes"][0].startswith("the system stopped it for lack of memory")
    assert "the message or the log names memory ('MemoryError')" in st["possible_causes"][0]
    assert "lack of memory (the log or the message names memory)" in st["summary"]
    assert "voxel_size" in st["options"][0]


def test_an_exit_with_a_message_that_names_memory_has_the_memory_options_and_no_cause_list(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("echo 'MemoryError: Unable to allocate 40.0 GiB for an array' 1>&2; exit 1"))
    st = server.run_status(str(out))
    assert "possible_causes" not in st
    assert "voxel_size" in st["options"][0] and any(o.startswith("read `log_tail`") for o in st["options"])
    assert not any("kernel log" in o for o in st["options"])        # the kernel did not stop the process: it exited


def test_an_estimate_below_the_ram_keeps_the_stop_by_the_user_first(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("echo 'memory estimate: about 2.0 GiB of 8.0 GiB RAM (x)'; kill -9 $$"))
    st = server.run_status(str(out))
    assert st["possible_causes"][0].startswith("a stop by the user or another program")
    assert "the memory estimate is 2.0 GiB" in st["possible_causes"][1] and "can be lower than the real peak" in st["possible_causes"][1]
    assert st["options"][0].startswith("check_problem") and "voxel_size" in st["options"][1]


def test_the_kernel_log_option_is_for_linux_and_names_the_pid(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, sh("kill -9 $$"))
    monkeypatch.setattr(server, "_ram_now", lambda: {"total_gib": 16.0, "available_now_gib": 8.0, "note": "n"})
    pid = server._current["proc"].pid
    monkeypatch.setattr(sys, "platform", "linux")
    opts = server.run_status(str(out))["options"]
    kernel = [o for o in opts if "kernel log" in o]
    assert len(kernel) == 1 and f"Killed process {pid}" in kernel[0] and "journalctl -k" in kernel[0]
    assert opts.index(kernel[0]) > max(i for i, o in enumerate(opts) if "voxel_size" in o or "less memory" in o)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert not any("kernel log" in o or "journalctl" in o for o in server.run_status(str(out))["options"])


# --- the dashboard note, the exit note, a run that wrote its own message (P9-6 review F3, F10) ----------------------

def test_the_dashboard_note_is_for_every_ended_run_and_points_to_the_saved_passes(tmp_path, yaml_file):
    folder = dead_run_folder(tmp_path, yaml_file)
    note = server.run_status(str(folder))["dashboard_note"]
    assert "ran inside the run process, which has ended" in note and f"`topopt dashboard {folder}` shows the saved passes (up to pass 6)" in note


def test_a_live_process_with_an_error_status_says_that_it_ends_after_the_error(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start(monkeypatch, yaml_file, out, sh("sleep 30"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps({"state": "error", "iteration": 0, "step": "setup", "elapsed_s": 0.1, "message": "boom"}))
    st = server.run_status(str(out))
    assert st["state"] == "error" and st["process_alive"] is True
    assert st["failure"]["exit_code"] is None and st["failure"]["exit_note"] == "the process is still alive; it ends after the error"
    assert "dashboard_note" not in st                                           # the process is alive: the dashboard can answer
    assert server.run_result(str(out)) == {"done": False, "message": "The run is not done yet. Call run_status."}


KILLED_WITH_MESSAGE = '''
import json, os, signal, sys
from pathlib import Path
from topopt.runjob import RunJob
out = Path(sys.argv[1])
job = RunJob(out)
job.start()
(out / "status.json").write_text(json.dumps({"state": "error", "iteration": 0, "step": "setup", "elapsed_s": 0.1,
                                             "message": "stopped by the user (Ctrl+C)"}))
os.kill(os.getpid(), signal.SIGKILL)
'''

JOB_ENDED_NO_STATUS = '''
import sys
from pathlib import Path
from topopt.runjob import RunJob
job = RunJob(Path(sys.argv[1]))
job.start()
job.finish(1)
sys.exit(1)
'''


def test_a_signal_with_a_message_of_the_run_gets_no_cause_list(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start_and_wait(monkeypatch, yaml_file, out, write_script(tmp_path, KILLED_WITH_MESSAGE))
    st = server.run_status(str(out))
    assert st["state"] == "error" and st["failure"]["signal"] == "SIGKILL"
    assert "possible_causes" not in st and "possible_causes_note" not in st
    assert not any("voxel_size" in o for o in st["options"])
    assert st["summary"].startswith("error: stopped by the user (Ctrl+C)")


def test_a_job_that_ended_with_no_status_file_has_the_failure_facts(tmp_path):
    import subprocess
    folder = tmp_path / "run"
    folder.mkdir()
    script = tmp_path / "job_ended.py"
    script.write_text(JOB_ENDED_NO_STATUS)
    assert subprocess.run([sys.executable, str(script), str(folder)], capture_output=True).returncode == 1
    st = server.run_status(str(folder))
    assert st["state"] == "none" and st["summary"].startswith(f"no run found in {folder}") and st["job"]["state"] == "failed"
    f = st["failure"]
    assert (f["exit_code"], f["signal"], f["run_data"]) == (1, None, "none") and f["peak_memory_bytes"] > 0
    assert "saved no pass" in st["dashboard_note"]
    assert st["options"][-1].startswith("check_problem")


def test_a_run_of_an_unknown_origin_with_a_busy_status_and_no_process_is_not_called_ended(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"state": "running", "iteration": 4, "step": "solve", "elapsed_s": 2.0, "message": ""}))
    st = server.run_status(str(folder))
    assert st["state"] == "running" and not (NEW_KEYS & set(st)) and "warning: the process has stopped" not in st["summary"]
    assert server.run_result(str(folder)) == {"done": False, "message": "No result yet. Call run_status."}


def test_a_stopped_status_with_a_dead_job_record_is_not_a_failure(tmp_path, yaml_file):
    folder = dead_run_folder(tmp_path, yaml_file)                                # the job record says stale_lock
    (folder / "status.json").write_text(json.dumps({"state": "stopped", "iteration": 6, "step": "stopped", "elapsed_s": 9.0,
                                                    "message": "stopped by a request"}))
    st = server.run_status(str(folder))
    assert st["state"] == "stopped" and st["job"]["state"] == "stale_lock" and not (NEW_KEYS & set(st))


def test_the_peak_of_an_earlier_job_in_a_stale_record_is_not_the_peak_of_this_run(tmp_path, yaml_file):
    folder = dead_run_folder(tmp_path, yaml_file, job_extra={"peak_rss_bytes": 5_000_000_000})
    f = server.run_status(str(folder))["failure"]
    assert f["peak_memory_bytes"] is None and f["peak_memory"] == "not recorded: the process did not reach its end"


# --- the replies of the runs that did not fail are the same ----------------------------------------

def wait_state(out, states, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        st = server.run_status(str(out))
        if st["state"] in states:
            return st
        time.sleep(0.1)
    raise AssertionError(f"the run did not reach {states}")


def test_the_replies_of_a_running_and_a_done_run_have_no_new_key(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start(monkeypatch, yaml_file, out, FAKE_CMD)
    st = wait_state(out, ("running",))
    assert set(st) == STATUS_KEYS and st["process_alive"] is True
    assert server.run_result(str(out)) == {"done": False, "message": "The run is not done yet. Call run_status."}
    st = wait_state(out, ("done", "error"))
    assert st["state"] == "done" and set(st) == STATUS_KEYS
    res = server.run_result(str(out))
    assert res["done"] is True and set(res) == DONE_RESULT_KEYS


def test_the_replies_of_a_folder_with_no_run_have_no_new_key(tmp_path):
    st = server.run_status(str(tmp_path / "none"))
    assert st["state"] == "none" and set(st) == EARLY_STATUS_KEYS
    assert server.run_result(str(tmp_path / "none")) == {"done": False, "message": "No result yet. Call run_status."}


def test_a_starting_run_has_no_new_key(monkeypatch, yaml_file, tmp_path):
    out = tmp_path / "run"
    start(monkeypatch, yaml_file, out, sh("sleep 30"))
    st = server.run_status(str(out))
    assert st["state"] == "starting" and set(st) == EARLY_STATUS_KEYS
    assert server.run_result(str(out)) == {"done": False, "message": "The run is not done yet. Call run_status."}


def test_a_run_that_a_request_stopped_is_not_a_failure(tmp_path):
    folder = tmp_path / "stopped"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"state": "stopped", "iteration": 4, "step": "stopped", "elapsed_s": 1.0,
                                                    "message": "stopped by a request"}))
    st = server.run_status(str(folder))
    assert st["state"] == "stopped" and not (NEW_KEYS & set(st))
    res = server.run_result(str(folder))
    assert res["stopped"] is True and not (NEW_KEYS & set(res))


# --- the real server over stdio ----------------------------------------------------------------------

def test_stdio_round_trip_of_a_killed_run(yaml_file, tmp_path):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    out = tmp_path / "stdio_run"
    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(REPO),
                                   env={"TOPOPT_MCP_CMD": sh("echo started; kill -9 $$")})

    def payload(res):
        if getattr(res, "structured_content", None):
            sc = res.structured_content
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        return json.loads(res.content[0].text)

    async def go():
        async with Client(params) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert all(len(tools[n].description) < 4096 for n in ("run_status", "run_result"))
            await client.call_tool("run_topopt", {"yaml_path": str(yaml_file), "out_dir": str(out)})
            for _ in range(300):
                st = payload(await client.call_tool("run_status", {"run_dir": str(out)}))
                if st["state"] == "error":
                    break
                await asyncio.sleep(0.1)
            assert st["state"] == "error", st
            res = payload(await client.call_tool("run_result", {"run_dir": str(out)}))
            return st, res

    st, res = asyncio.run(go())
    assert st["log_tail"] == "started" and st["failure"]["signal"] == "SIGKILL" and st["failure"]["stopped_by_this_server"] is False
    assert st["possible_causes"] and st["options"]
    assert res["done"] is False and res["failure"]["signal"] == "SIGKILL" and res["log_tail"] == "started"
