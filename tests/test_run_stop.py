"""Stop a running run on request (P4b-4, F4-11; CONTRACTS.md sections 8, 10 and 24.2).

A request is the file `job.stop` in the run folder. The run reads it at the end of each pass, after it saves the checkpoint, and ends with
the status `stopped`, the job marker `job.STOPPED` and exit code 5. `--resume` then continues it with no override and gives the result of a
run that was not stopped.

The runs are child processes with one thread (`tests/stop_at.py`, like `tests/kill_run.py`): a direct solver then gives the same bytes in each
run. The helper writes the request from inside the run after the checkpoint of a chosen pass, or it holds the run there on a FIFO while the
test calls `topopt stop` or the MCP tool `stop_run`. So no test needs a sleep or depends on timing. AMG is not reproducible, so the tests that
compare bytes need a direct solver.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import checkpoint as C  # noqa: E402
from topopt import runjob  # noqa: E402
from topopt import verdict as V  # noqa: E402
from topopt.cli import build_parser, main  # noqa: E402
from topopt.config import StopRules, UserError  # noqa: E402
from topopt.runjob import (DONE_FILE, FAILED_FILE, INFO_FILE, LOCK_FILE, STOP_FILE, STOPPED_EXIT, STOPPED_FILE,  # noqa: E402
                           JobHeld, RunJob, clear_stop, job_facts, read_stop, request_stop, run_job, stop_reply)

from test_checkpoint import ENV1, clean, edit, finish, history, needs_direct, same_run, tree_state  # noqa: E402
from test_optimize import make_case  # noqa: E402
from test_runjob import dead_pid, put_lock, sleeper  # noqa: E402,F401  (fixtures)

STOP = Path(__file__).with_name("stop_at.py")
N_IT = 24    # the passes of the case: the beta stages are 0 to 9, 10 to 19 and 20 to 24


def stop_run_child(yaml_path, out, point, *extra, timeout=600):
    """`topopt run` in a child process that writes its own request at `point` (see tests/stop_at.py). Return the CompletedProcess."""
    args = ["run", str(yaml_path), "--out", str(out), "--quiet", *extra]
    return subprocess.run([sys.executable, str(STOP), point, *args], env=ENV1, capture_output=True, text=True, timeout=timeout)


def stopped_at(yaml_path, out, k, *extra):
    done = stop_run_child(yaml_path, out, f"pass:{k}", *extra)
    assert done.returncode == STOPPED_EXIT, f"exit {done.returncode}\n" + done.stdout[-1500:] + done.stderr[-2500:]
    return done


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """The cantilever (1 mm voxels, 24 passes), and the run that was not stopped (1 thread)."""
    tmp = tmp_path_factory.mktemp("stop_case")
    yaml_path = make_case(tmp, h=1.0, iterations=N_IT)
    base = tmp / "base"
    finish(yaml_path, base)
    return yaml_path, base


@pytest.fixture
def work(case, tmp_path):
    """A copy of the case in a new folder (the YAML and the STL files): a test can edit it."""
    src = case[0].parent
    dst = tmp_path / "case"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("base", "r*", "*.log"))
    return dst / case[0].name


# --- the file protocol: runjob units -------------------------------------------------------------------

def test_a_request_is_a_json_file_that_is_written_atomically_and_only_once(tmp_path):
    req, created = request_stop(tmp_path, "mcp stop_run", 4242)
    assert created and set(req) == {"format", "requested_at", "by", "target_pid"}
    assert (req["by"], req["target_pid"], req["format"]) == ("mcp stop_run", 4242, 1)
    assert read(tmp_path / STOP_FILE) == req and read_stop(tmp_path) == req
    assert not list(tmp_path.glob("*.tmp"))                              # the temp file is renamed, not left
    before = (tmp_path / STOP_FILE).read_bytes()
    again, created = request_stop(tmp_path, "topopt stop", 99)
    assert not created and again == req and (tmp_path / STOP_FILE).read_bytes() == before   # the first request stays as it is
    assert clear_stop(tmp_path) is True and read_stop(tmp_path) is None and clear_stop(tmp_path) is False


def test_the_target_pid_is_null_when_the_caller_does_not_know_it(tmp_path):
    assert request_stop(tmp_path, "topopt stop")[0]["target_pid"] is None
    clear_stop(tmp_path)
    assert request_stop(tmp_path, "topopt stop", -3)[0]["target_pid"] is None


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "42"])
def test_a_file_that_is_not_a_request_of_topopt_is_still_a_request(tmp_path, text):
    """A person can `touch job.stop`. The file is the request; the dict says that nobody is named."""
    (tmp_path / STOP_FILE).write_text(text)
    req = read_stop(tmp_path)
    assert req is not None and req["by"] is None and req["target_pid"] is None and req["requested_at"] is None
    assert request_stop(tmp_path, "mcp stop_run", 1)[1] is False        # it is a request already


def test_the_job_removes_a_request_for_another_pid_with_a_fact(tmp_path):
    request_stop(tmp_path, "mcp stop_run", os.getpid() + 1_000_003)
    job = RunJob(tmp_path)
    job.start()
    assert read_stop(tmp_path) is None
    assert any(f.startswith("job stop: a request to stop") and "mcp stop_run" in f and f"pid {os.getpid()}" in f for f in job.facts)
    assert read(tmp_path / INFO_FILE)["notes"] == job.facts                      # the fact is in job.json too
    job.finish(0)


def test_the_job_removes_a_request_that_names_no_pid(tmp_path):
    request_stop(tmp_path, "topopt stop")
    job = RunJob(tmp_path)
    job.start()
    assert read_stop(tmp_path) is None and any("(no pid named)" in f for f in job.facts)
    job.finish(0)


def test_a_request_for_this_pid_stays_when_the_job_starts(tmp_path):
    """The case of a request between the spawn of the run process and its lock: the caller knows the pid of the process it started."""
    request_stop(tmp_path, "mcp stop_run", os.getpid())
    job = RunJob(tmp_path)
    job.start()
    assert read_stop(tmp_path)["target_pid"] == os.getpid() and not any("job stop" in f for f in job.facts)
    job.finish(0)


def test_exit_code_5_gives_the_state_stopped_and_the_marker(tmp_path):
    assert run_job(tmp_path, lambda job: STOPPED_EXIT) == 5
    assert (tmp_path / STOPPED_FILE).read_text() == "exit 5\n"
    assert not (tmp_path / DONE_FILE).exists() and not (tmp_path / FAILED_FILE).exists() and not (tmp_path / LOCK_FILE).exists()
    rec = read(tmp_path / INFO_FILE)
    assert rec["state"] == "stopped" and rec["exit_code"] == 5
    facts = job_facts(tmp_path)
    assert facts["state"] == "stopped" and facts["marker"] == STOPPED_FILE and "token" not in facts
    assert facts["summary"].startswith("job: stopped by a request (exit 5, ") and "--resume" in facts["summary"]


def test_a_caller_that_has_5_as_a_success_code_gets_done(tmp_path):
    run_job(tmp_path, lambda job: 5, success_codes=(0, 5))
    assert read(tmp_path / INFO_FILE)["state"] == "done" and (tmp_path / DONE_FILE).exists() and not (tmp_path / STOPPED_FILE).exists()


def test_the_next_job_removes_the_stopped_marker(tmp_path):
    run_job(tmp_path, lambda job: STOPPED_EXIT)
    assert (tmp_path / STOPPED_FILE).read_text().strip() == "exit 5"       # the job wrote it (a build without the marker does not)
    job = RunJob(tmp_path)
    job.start()
    assert not (tmp_path / STOPPED_FILE).exists() and read(tmp_path / INFO_FILE)["state"] == "running"
    job.finish(0)
    assert (tmp_path / DONE_FILE).exists() and not (tmp_path / STOPPED_FILE).exists()


def test_a_lock_that_is_left_after_a_stopped_job_is_a_fact_not_a_stale_lock(tmp_path, dead_pid):
    job = RunJob(tmp_path)
    job.start()
    rec = json.loads((tmp_path / LOCK_FILE).read_text())
    job.finish(STOPPED_EXIT)
    (tmp_path / LOCK_FILE).write_text(json.dumps(dict(rec, pid=dead_pid)))     # the lock of the finished job came back (a person, a crash)
    facts = job_facts(tmp_path)
    assert facts["state"] == "stopped" and any("lock file of the finished job" in n for n in facts["notes"])


def test_the_parser_knows_the_stop_command():
    assert build_parser().parse_args(["stop", "some/run"]).run_dir == "some/run"


# --- the verdict -------------------------------------------------------------------------------------

def rows_of(n=8, beta=1.0):
    return [{"iteration": i, "beta": beta, "compliance": 1.0 / (i + 1), "grayness": 0.5, "change": 0.1, "constraints": [],
             "objective": {"value": 1.0 / (i + 1)}} for i in range(n)]


def test_the_verdict_of_a_request_has_the_state_the_key_and_the_tests():
    rules, sched = StopRules(), ((0, 1.0), (20, 4.0))
    v = V.build_verdict(rows_of(), rules, sched, 24, request={"requested_at": "t0", "by": "mcp stop_run", "target_pid": 9})
    assert v["state"] == "stopped_by_request" and "stopped_by_request" in V.STATES
    assert v["text"].startswith("stopped by a request at pass 7 of 24. Tests: objective change")
    assert v["stop_request"] == {"requested_at": "t0", "by": "mcp stop_run", "iteration": 7}      # no pid, nothing else
    assert v["final_iteration"] == 7 and v["stop"] is None and v["iterations"] == 24


def test_every_other_verdict_keeps_its_bytes_and_has_no_stop_request_key():
    rules, sched = StopRules(), ((0, 1.0), (20, 4.0))
    keys = ["format", "state", "text", "final_iteration", "iterations", "converged", "feasible", "measures", "stop_rules", "stop",
            "resume_points", "notes", "facts"]
    for rows, stop in ((rows_of(8), None), (rows_of(24), None), (rows_of(8), {"rule": "converged", "iteration": 7, "text": "stop rule 'converged' at pass 7"})):
        v = V.build_verdict(rows, rules, sched, 24, stop)
        assert list(v) == keys, v["state"]
        assert json.dumps(V.build_verdict(rows, rules, sched, 24, stop, request=None)) == json.dumps(v)
    assert V.build_verdict([], rules, sched, 24)["state"] == "unfinished"


def test_the_facts_of_the_stop_pass_reach_the_verdict_text():
    rows = rows_of(5)
    rows[-1]["stage_end"] = {"beta": 1.0, "next_beta": 1.0, "volume": {"limit": 0.3, "projected": 0.29, "thresholded": 0.2,
                                                                      "used_projected": 0.97, "used_thresholded": 0.67},
                             "load_path": {"regions": {"tip": True}, "not_joined": [], "bodies": 1, "bodies_with_a_support": 1}}
    v = V.build_verdict(rows, StopRules(), ((0, 1.0),), 24, request={"requested_at": "t0", "by": "x"})
    assert v["facts"]["iteration"] == 4 and v["facts"]["volume"]["used_projected"] == 0.97
    assert "Facts: volume: 0.97 of the allowed volume used" in v["text"]


def test_stage_end_lines_take_the_place_from_the_caller():
    from topopt import stage_facts
    end = {"beta": 1.0, "next_beta": 1.0, "volume": {"limit": 0.3, "projected": 0.29, "thresholded": 0.2, "used_projected": 0.97,
                                                    "used_thresholded": 0.67},
           "load_path": {"regions": {"tip": True}, "not_joined": [], "bodies": 1, "bodies_with_a_support": 1}}
    assert [l.split(": volume")[0] for l in stage_facts.stage_end_lines(end, 7, where="stop at pass 7")][:1] == ["fact: stop at pass 7"]
    assert stage_facts.stage_end_lines(end, 7)[0].startswith("fact: end of the beta 1 stage, pass 7: volume")
    assert stage_facts.stage_end_lines(dict(end, next_beta=None), 7)[0].startswith("fact: last pass 7: volume")


# --- a stop at the end of a pass (CLI) -------------------------------------------------------------------

@needs_direct
@pytest.mark.parametrize("k, stage_end", [(7, False), (9, True)], ids=["mid-stage pass 7", "stage-end pass 9"])
def test_a_stop_writes_the_state_the_checkpoint_the_verdict_and_leaves_the_history_as_it_is(case, tmp_path, k, stage_end):
    yaml_path, base = case
    out = tmp_path / "run"
    done = stopped_at(yaml_path, out, k)
    assert done.stdout == ""                                            # --quiet: the exit code is the answer
    msg = f"stopped by a request at pass {k} of {N_IT}; --resume continues at pass {k + 1}"
    assert read(out / "status.json") | {"elapsed_s": 0} == {"state": "stopped", "iteration": k, "step": "stopped", "elapsed_s": 0, "message": msg}
    # the job: state stopped, exit 5, the marker, no lock, no request left
    job = read(out / INFO_FILE)
    assert (job["state"], job["exit_code"]) == ("stopped", 5) and (out / STOPPED_FILE).read_text() == "exit 5\n"
    assert not (out / LOCK_FILE).exists() and not (out / STOP_FILE).exists()
    assert not (out / DONE_FILE).exists() and not (out / FAILED_FILE).exists()
    assert not runjob.pid_alive(job["pid"])                             # no process is left
    # the checkpoint of pass k: the normal one (an unfinished run), nothing added
    cp = C.describe(out)
    assert (cp["iteration"], cp["complete"], cp["stop_rule"]) == (k, False, None)
    meta = C.read_checkpoint(out).meta
    base_meta = C.read_checkpoint(base).meta
    assert set(meta) == set(base_meta) - {"rows_before"} and meta["stop"] is None   # `rows_before` (P4b-5) is in a complete checkpoint only: the base is complete
    # no post step, no refine step
    assert not (out / "result").exists()
    # history lines and snapshots 0..k: the lines of the run that was not stopped, with no new key
    mine, theirs = history(out), history(base)[: k + 1]
    assert len(mine) == k + 1 and clean(mine) == clean(theirs)
    assert [set(a) for a in mine] == [set(b) for b in theirs]
    assert ("stage_end" in mine[-1]) is stage_end                       # the stop adds no stage_end to a line that has none
    snaps = sorted(p.name for p in (out / "snapshots").iterdir())
    assert snaps == [f"it_{i:04d}.bin" for i in range(k + 1)]
    for name in snaps:
        assert (out / "snapshots" / name).read_bytes() == (base / "snapshots" / name).read_bytes()
    # the verdict: the state, the request, the tests, and the facts of the stop pass
    v = V.read_verdict(out)
    assert v["state"] == "stopped_by_request" and v["final_iteration"] == k and v["iterations"] == N_IT
    assert v["stop_request"]["by"] == "tests/stop_at.py" and v["stop_request"]["iteration"] == k and v["stop"] is None
    assert v["text"].startswith(f"stopped by a request at pass {k} of {N_IT}. Tests: ")
    assert v["facts"]["iteration"] == k and v["facts"]["volume"]["limit"] == 0.3 and "load_path" in v["facts"]
    if stage_end:
        assert v["facts"] == {"iteration": k, **mine[-1]["stage_end"]}          # the facts that the line holds
    # the run log
    log = (out / "run.log").read_text()
    assert f"verdict: stopped by a request at pass {k} of {N_IT}." in log and f"stop: {msg}" in log
    assert f"fact: stop at pass {k}: volume:" in log if not stage_end else f"fact: stop at pass {k}:" not in log
    assert (f"fact: end of the beta 1 stage, pass {k}: volume" in log) is stage_end
    assert "done in" not in log and "refine:" not in log and "last pass" not in log


@needs_direct
def test_a_resume_after_a_stop_needs_no_override_and_equals_the_run_that_was_not_stopped(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    stopped_at(yaml_path, out, 7)
    done = finish(yaml_path, out, "--resume")                           # exit 0, no --allow-changed-inputs
    assert done.returncode == 0
    same_run(base, out)                                                 # history, snapshots, final.stl, preview, metrics, run.json
    assert len(history(out)) == N_IT + 1 and (out / "snapshots" / f"it_{N_IT:04d}.bin").exists()
    job = read(out / INFO_FILE)
    assert job["state"] == "done" and (out / DONE_FILE).exists() and not (out / STOPPED_FILE).exists()
    assert not runjob.pid_alive(job["pid"]) and not (out / LOCK_FILE).exists()
    log = (out / "run.log").read_text()
    assert "resume: the run was stopped by a request at pass 7 (requested by tests/stop_at.py at " in log
    v = V.read_verdict(out)
    assert v["state"] == "iteration_limit" and "stop_request" not in v and v["resume_points"][0]["iteration"] == 7
    assert read(out / "result" / "metrics.json")["verdict"] == v


@needs_direct
def test_two_stops_and_two_resumes_still_equal_the_run_that_was_not_stopped(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    stopped_at(yaml_path, out, 4)
    second = stop_run_child(yaml_path, out, "pass:15", "--resume")          # the second run stops again, at a later pass
    assert second.returncode == STOPPED_EXIT, second.stderr[-2000:]
    assert read(out / "status.json")["iteration"] == 15 and C.describe(out)["iteration"] == 15
    assert [r["iteration"] for r in V.read_verdict(out)["resume_points"]] == [4]
    finish(yaml_path, out, "--resume")
    same_run(base, out)
    assert [r["iteration"] for r in V.read_verdict(out)["resume_points"]] == [4, 15]


@needs_direct
def test_a_stop_at_the_first_pass_has_a_checkpoint_and_resumes(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    stopped_at(yaml_path, out, 0)
    assert C.describe(out)["iteration"] == 0 and len(history(out)) == 1
    assert V.read_verdict(out)["state"] == "stopped_by_request"
    finish(yaml_path, out, "--resume")
    same_run(base, out)


@needs_direct
def test_a_request_after_the_last_pass_stops_before_the_post_step_and_the_resume_runs_the_post_step_only(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    done = stopped_at(yaml_path, out, N_IT)                             # the request comes after the checkpoint of the last pass
    msg = (f"stopped by a request before the post step (the optimizer ended at pass {N_IT} of {N_IT}); "
           "result/ is not made; --resume runs the post step only")
    st = read(out / "status.json")
    assert (st["state"], st["iteration"], st["step"], st["message"]) == ("stopped", N_IT, "stopped", msg)
    assert not (out / "result").exists() and not (out / STOP_FILE).exists() and (out / STOPPED_FILE).exists()
    assert read(out / INFO_FILE)["state"] == "stopped" and not (out / LOCK_FILE).exists()
    cp = C.describe(out)
    assert (cp["iteration"], cp["complete"]) == (N_IT, True)            # the complete checkpoint stays
    assert V.read_verdict(out)["state"] == "iteration_limit"            # the optimizer verdict stays: the request did not touch it
    assert len(history(out)) == N_IT + 1
    # the resume runs no pass: the post step only
    before = (out / "history.jsonl").read_bytes()
    finish(yaml_path, out, "--resume")
    log = (out / "run.log").read_text().split("resume: continue from")[1]
    assert "the post step runs now" in log and "stopped by a request after its last pass" in log
    assert not [l for l in log.splitlines() if l.startswith("it ")] and (out / "history.jsonl").read_bytes() == before
    same_run(base, out)
    assert read(out / INFO_FILE)["state"] == "done"


def test_a_stop_rule_that_ended_the_run_keeps_its_verdict_when_a_request_comes_before_the_post_step(work, tmp_path):
    from test_checkpoint import edit
    edit(work, lambda d: d["settings"].update(stop_rules={"time_budget_s": 1e-6}))     # the budget is spent at the first pass
    out = tmp_path / "run"
    stopped_at(work, out, 0)
    v = V.read_verdict(out)
    assert v["state"] == "stopped_by_rule" and v["stop"]["rule"] == "time_budget" and "stop_request" not in v
    st = read(out / "status.json")
    assert st["state"] == "stopped" and st["message"].startswith("stopped by a request before the post step (the optimizer ended at pass 0")
    assert C.describe(out)["complete"] is True and not (out / "result").exists()


@needs_direct
def test_a_request_during_the_post_step_is_removed_with_a_fact_and_the_run_is_done(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    done = stop_run_child(yaml_path, out, "post")
    assert done.returncode == 0, done.stderr[-2000:]
    assert not (out / STOP_FILE).exists() and (out / DONE_FILE).exists() and not (out / STOPPED_FILE).exists()
    assert read(out / INFO_FILE)["state"] == "done" and read(out / "status.json")["state"] == "done"
    log = (out / "run.log").read_text()
    assert "fact: a request to stop (requested by tests/stop_at.py at " in log and "came after the last pass" in log
    same_run(base, out)


@needs_direct
def test_an_old_request_for_another_pid_is_removed_and_the_run_goes_on(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    done = stop_run_child(yaml_path, out, "lock_other")
    assert done.returncode == 0, done.stderr[-2000:]
    assert not (out / STOP_FILE).exists() and (out / DONE_FILE).exists()
    assert any(n.startswith("job stop: a request to stop") for n in read(out / INFO_FILE)["notes"])
    assert "warning: job stop: a request to stop (requested by tests/stop_at.py" in (out / "run.log").read_text()
    same_run(base, out)


@needs_direct
def test_a_request_that_names_the_pid_of_the_run_stays_and_stops_the_run_at_its_first_pass(case, tmp_path):
    """The request between the spawn and the lock: the caller knew the pid of the process that it started."""
    yaml_path, base = case
    out = tmp_path / "run"
    done = stop_run_child(yaml_path, out, "lock")
    assert done.returncode == STOPPED_EXIT, done.stderr[-2000:]
    assert read(out / "status.json")["iteration"] == 0 and V.read_verdict(out)["state"] == "stopped_by_request"
    assert read(out / INFO_FILE)["notes"] == []                         # nothing was removed
    finish(yaml_path, out, "--resume")
    same_run(base, out)


# --- a stop while the run works: `topopt stop`, held at a pass ---------------------------------------------

def wait_for(path, timeout=300):
    end = time.time() + timeout
    while not Path(path).exists():
        assert time.time() < end, f"{path} did not appear"
        time.sleep(0.05)


def release(out):
    with open(Path(out) / "hold.fifo", "wb") as f:       # the run reads this and goes on
        f.write(b"go")


def held_cli_run(yaml_path, out, k, *extra, env=None):
    """Start `topopt run` as a child that holds at the checkpoint of pass `k`. Return the Popen when the hold is reached."""
    args = [sys.executable, str(STOP), f"hold:{k}", "run", str(yaml_path), "--out", str(out), "--quiet", *extra]
    proc = subprocess.Popen(args, env=env or ENV1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wait_for(Path(out) / "hold.reached")
    return proc


@needs_direct
def test_topopt_stop_asks_the_run_that_works_and_the_run_stops_at_that_pass(case, tmp_path, capsys):
    yaml_path, base = case
    out = tmp_path / "run"
    proc = held_cli_run(yaml_path, out, 6)
    try:
        assert main(["stop", str(out)]) == 0
        said = capsys.readouterr().out
        req = read(out / STOP_FILE)
        assert (req["by"], req["target_pid"]) == ("topopt stop", proc.pid)
        assert "The request is written" in said and f"pid {proc.pid}" in said and "pass 6 at the last status" in said
        assert "does not read job.stop" in said                          # the note about older builds
        before = (out / STOP_FILE).read_bytes()
        assert main(["stop", str(out)]) == 0                             # a second call: already there, not written again
        assert "in the folder already" in capsys.readouterr().out and (out / STOP_FILE).read_bytes() == before
    finally:
        release(out)
    assert proc.wait(timeout=300) == STOPPED_EXIT, proc.stderr.read()[-2000:]
    assert C.describe(out)["iteration"] == 6 and read(out / "status.json")["state"] == "stopped"
    assert V.read_verdict(out)["stop_request"]["by"] == "topopt stop"
    # the run has stopped: nothing to stop. The exit code says that nothing was written.
    assert main(["stop", str(out)]) == 1
    said = capsys.readouterr().out
    assert "has stopped already" in said and "--resume" in said and not (out / STOP_FILE).exists()
    finish(yaml_path, out, "--resume")
    same_run(base, out)


def test_topopt_stop_says_what_it_found_when_it_can_write_nothing(case, tmp_path, capsys):
    yaml_path, base = case
    done = tmp_path / "done"
    shutil.copytree(base, done)
    assert main(["stop", str(done)]) == 1
    said = capsys.readouterr().out
    assert "has ended already" in said and "job: done" in said and not (done / STOP_FILE).exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["stop", str(empty)]) == 1 and "No run in" in capsys.readouterr().out and not (empty / STOP_FILE).exists()
    assert main(["stop", str(tmp_path / "missing")]) == 1 and "is not a folder" in capsys.readouterr().err


# --- the replies of stop_reply, and of the MCP tool ---------------------------------------------------------

@pytest.fixture
def mcp_server(monkeypatch, tmp_path):
    """The MCP module with a clean registry. The work folder is `tmp_path`. A run process that a test starts is ended at the end."""
    from topopt_mcp import server
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    monkeypatch.delenv("TOPOPT_MCP_CMD", raising=False)
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    server._current.clear()
    yield server
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._current.clear()


def live_job(d, status=None):
    """A job that this process holds (a live lock), and the status file of a run in some phase. Return the job; the test ends it."""
    d.mkdir(parents=True, exist_ok=True)
    job = RunJob(d)
    job.start()
    if status is not None:
        (d / "status.json").write_text(json.dumps(status))
    return job


def test_every_reply_has_the_note_and_the_keys(mcp_server, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    r = mcp_server.stop_run(str(empty))
    assert r["state"] == "none" and "No run in" in r["message"] and r["stop_request"] is None and not (empty / STOP_FILE).exists()
    for key in ("state", "run_dir", "run_dir_rel", "pid", "iteration", "phase", "message", "options", "note", "job", "stop_request",
                "found_by", "process_alive"):
        assert key in r, key
    assert r["note"].startswith("A topopt build from before this change does not read job.stop. A run of such a build ignores the request.")
    assert r["found_by"] == "the run_dir argument"


def test_a_live_run_in_its_loop_gets_the_request_with_the_pid_and_the_pass(mcp_server, tmp_path):
    d = tmp_path / "runs" / "a"
    job = live_job(d, {"state": "running", "iteration": 11, "step": "solve", "elapsed_s": 3.0, "message": ""})
    try:
        r = mcp_server.stop_run(str(d))
        assert r["state"] == "stop_requested" and r["pid"] == os.getpid() and r["iteration"] == 11 and r["phase"] == "running"
        assert "stops at the end of the current pass (pass 11 at the last status), after it saves the checkpoint" in r["message"]
        assert read(d / STOP_FILE)["by"] == "mcp stop_run" and read(d / STOP_FILE)["target_pid"] == os.getpid()
        assert r["stop_request"] == read(d / STOP_FILE) and r["note"]
        # a second call: the first request stays, and is named
        before = (d / STOP_FILE).read_bytes()
        again = mcp_server.stop_run(str(d))
        assert again["state"] == "already_requested" and again["stop_request"]["requested_at"] == r["stop_request"]["requested_at"]
        assert "is not written again" in again["message"] and (d / STOP_FILE).read_bytes() == before
    finally:
        job.finish(0)


def test_a_run_of_this_server_that_has_written_no_status_yet_is_starting_and_gets_the_request(tmp_path):
    d = tmp_path / "runs" / "a"
    job = live_job(d)           # the lock is held, status.json and run.json do not exist yet
    try:
        r = stop_reply(d, "test", starting_pid=os.getpid())      # the caller started this process: it is a run that has just begun
        assert r["state"] == "stop_requested" and "starting" in r["message"] and "stops at the end of its first pass" in r["message"]
    finally:
        job.finish(0)


def test_a_live_lock_in_a_folder_that_is_not_a_run_folder_is_not_a_run(mcp_server, tmp_path):
    """M1 of the review: the folder of a ratchet job holds a live job lock, and no status.json or run.json. It reads no job.stop."""
    d = tmp_path / "runs" / "live-ratchet"
    job = live_job(d)
    (d / "ratchet.json").write_text("{}")
    try:
        r = mcp_server.stop_run(str(d))
        assert r["state"] == "none" and r["pid"] == os.getpid() and not (d / STOP_FILE).exists() and r["stop_request"] is None
        assert "not a topopt run" in r["message"] and "ratchet.json is here" in r["message"] and "does not read job.stop" in r["message"]
        assert "Nothing is written" in r["message"] and any("ratchet_status" in o for o in r["options"])
        (d / "ratchet.jsonl").write_text("")
        assert "ratchet.json and ratchet.jsonl are here" in mcp_server.stop_run(str(d))["message"]
        for name in ("ratchet.json", "ratchet.jsonl"):
            (d / name).unlink()
        unknown = mcp_server.stop_run(str(d))      # the same lock with no sign of a ratchet: the kind of the job is not known
        assert unknown["state"] == "none" and "not a topopt run" in unknown["message"] and "kind of the job is not known" in unknown["message"]
        assert not (d / STOP_FILE).exists()
    finally:
        job.finish(0)


def test_the_cli_stop_of_a_ratchet_folder_writes_nothing(tmp_path, capsys):
    d = tmp_path / "runs" / "live-ratchet"
    job = live_job(d)
    (d / "ratchet.json").write_text("{}")
    try:
        assert main(["stop", str(d)]) == 1
        out = capsys.readouterr().out
        assert "not a topopt run" in out and "ratchet.json is here" in out and not (d / STOP_FILE).exists()
    finally:
        job.finish(0)


def test_the_scan_finds_the_run_and_not_the_ratchet_job_next_to_it(mcp_server, tmp_path):
    run, ratchet = tmp_path / "runs" / "live", tmp_path / "runs" / "live-ratchet"
    jr = live_job(run, {"state": "running", "iteration": 3, "step": "solve", "elapsed_s": 1.0, "message": ""})
    jt = live_job(ratchet)
    (ratchet / "ratchet.json").write_text("{}")
    try:
        assert [x["run_dir"] for x in mcp_server._live_runs()] == [str(run)]
        r = mcp_server.stop_run()
        assert r["state"] == "stop_requested" and r["run_dir"] == str(run) and r["found_by"] == "the scan for a live job lock"
        assert (run / STOP_FILE).exists() and not (ratchet / STOP_FILE).exists()
    finally:
        jr.finish(0), jt.finish(0)


def test_the_scan_finds_nothing_when_the_only_live_lock_is_a_ratchet_job(mcp_server, tmp_path):
    ratchet = tmp_path / "runs" / "live-ratchet"
    jt = live_job(ratchet)
    (ratchet / "ratchet.json").write_text("{}")
    try:
        assert mcp_server._live_runs() == []
        r = mcp_server.stop_run()
        assert r["state"] == "none" and r["found_by"] == "the default run folder" and not (ratchet / STOP_FILE).exists()
    finally:
        jt.finish(0)


def test_list_runs_and_the_scan_use_one_rule_for_a_run_folder(mcp_server, tmp_path):
    a, b = tmp_path / "runs" / "with-status", tmp_path / "runs" / "with-run-json"
    ja, jb = live_job(a, {"state": "running", "iteration": 1, "step": "solve", "elapsed_s": 1.0, "message": ""}), live_job(b)
    (b / "run.json").write_text(json.dumps({"format": 1, "name": "x", "shape": [4, 4, 4], "h": 1.0, "origin": [0, 0, 0], "iterations": 4}))
    try:
        listed = {r["run_dir"] for r in mcp_server.list_runs()["runs"]}
        scanned = {x["run_dir"] for x in mcp_server._live_runs()}
        assert listed == scanned == {str(a), str(b)}
        assert runjob.is_run_folder(a) and runjob.is_run_folder(b) and not runjob.is_run_folder(tmp_path)
    finally:
        ja.finish(0), jb.finish(0)


def test_a_status_file_of_an_earlier_run_is_not_the_phase_of_the_run_that_holds_the_lock(mcp_server, tmp_path):
    """After a resume the old `stopped` status stays until the new run writes its own: the new run is starting, not closing."""
    d = tmp_path / "runs" / "a"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": "stopped", "iteration": 7, "step": "stopped", "elapsed_s": 1.0, "message": "m"}))
    old = time.time() - 3600
    os.utime(d / "status.json", (old, old))
    job = live_job(d)
    try:
        r = mcp_server.stop_run(str(d))
        assert r["state"] == "stop_requested" and r["phase"] is None and r["iteration"] is None
    finally:
        job.finish(0)


@pytest.mark.parametrize("phase", ["postprocess", "done"])
def test_a_run_in_the_post_step_gets_no_request_and_the_reply_gives_the_options(mcp_server, tmp_path, phase):
    d = tmp_path / "runs" / "a"
    job = live_job(d, {"state": phase, "iteration": N_IT, "step": phase, "elapsed_s": 9.0, "message": ""})
    try:
        r = mcp_server.stop_run(str(d))
        assert r["state"] == "postprocess" and r["pid"] == os.getpid() and not (d / STOP_FILE).exists()
        assert "post step does not read a request" in r["message"] and "Nothing is written" in r["message"]
        assert any("wait" in o for o in r["options"]) and any(f"end the process yourself (pid {os.getpid()})" in o and "--resume" in o for o in r["options"])
        assert r["stop_request"] is None and r["note"]
    finally:
        job.finish(0)


def test_a_run_that_is_closing_its_job_gets_no_request(mcp_server, tmp_path):
    d = tmp_path / "runs" / "a"
    job = live_job(d, {"state": "stopped", "iteration": 3, "step": "stopped", "elapsed_s": 1.0, "message": "m"})
    try:
        r = mcp_server.stop_run(str(d))
        assert r["state"] == "nothing_to_stop" and "closing its job record" in r["message"] and not (d / STOP_FILE).exists()
    finally:
        job.finish(STOPPED_EXIT)


@pytest.mark.parametrize("code, word", [(0, "has ended already"), (STOPPED_EXIT, "has stopped already"), (2, "has ended already")])
def test_a_job_that_has_ended_has_nothing_to_stop(mcp_server, tmp_path, code, word):
    d = tmp_path / "runs" / "a"
    run_job(d, lambda job: code)
    r = mcp_server.stop_run(str(d))
    assert r["state"] == "nothing_to_stop" and word in r["message"] and r["job"]["state"] in ("done", "stopped", "failed")
    assert not (d / STOP_FILE).exists() and r["stop_request"] is None and r["process_alive"] is False


def test_a_stale_lock_is_gone_with_the_job_summary_and_no_request(mcp_server, tmp_path, dead_pid):
    d = tmp_path / "runs" / "a"
    put_lock(d, dead_pid)
    r = mcp_server.stop_run(str(d))
    assert r["state"] == "gone" and r["job"]["state"] == "stale_lock" and "job: stale lock" in r["message"]
    assert r["checkpoint"] is None and any("no checkpoint" in o for o in r["options"]) and not (d / STOP_FILE).exists()


def test_a_killed_run_is_gone_and_the_reply_names_the_checkpoint_and_the_resume(case, mcp_server, tmp_path):
    yaml_path, base = case
    out = tmp_path / "runs" / "killed"
    from test_checkpoint import killed
    killed(yaml_path, out, "after_checkpoint:6")
    r = mcp_server.stop_run(str(out))
    assert r["state"] == "gone" and r["job"]["state"] == "stale_lock" and not (out / STOP_FILE).exists()
    assert (r["checkpoint"]["iteration"], r["checkpoint"]["complete"]) == (6, False)
    assert any("continues the run at pass 7" in o for o in r["options"]) and any("resume=true replaces the stale lock" in o for o in r["options"])


def test_a_job_record_that_says_running_with_no_lock_is_gone(mcp_server, tmp_path):
    d = tmp_path / "runs" / "a"
    job = live_job(d)
    (d / LOCK_FILE).unlink()
    r = mcp_server.stop_run(str(d))
    assert r["state"] == "gone" and r["job"]["state"] == "unfinished" and not (d / STOP_FILE).exists()


def test_the_default_finds_the_one_live_run_by_the_lock_scan(mcp_server, tmp_path):
    """A run that the CLI started has no registry entry in this server. The job lock finds it."""
    a, b = tmp_path / "runs" / "a", tmp_path / "runs" / "b"
    run_job(a, lambda job: 0)                                             # an ended run: no live lock
    job = live_job(b, {"state": "running", "iteration": 2, "step": "solve", "elapsed_s": 1.0, "message": ""})
    try:
        r = mcp_server.stop_run()
        assert r["state"] == "stop_requested" and r["run_dir"] == str(b) and r["found_by"] == "the scan for a live job lock"
        assert (b / STOP_FILE).exists() and not (a / STOP_FILE).exists()
    finally:
        job.finish(0)


def test_several_live_runs_are_listed_and_nothing_is_written(mcp_server, tmp_path):
    a, b = tmp_path / "runs" / "a", tmp_path / "b"
    status = {"state": "running", "iteration": 2, "step": "solve", "elapsed_s": 1.0, "message": ""}
    ja, jb = live_job(a, status), live_job(b, status)
    try:
        r = mcp_server.stop_run()
        assert r["state"] == "several" and {x["run_dir"] for x in r["runs"]} == {str(a), str(b)} and r["note"]
        assert r["stop_request"] is None and r["run_dir"] is None and "Give run_dir" in r["message"]
        assert not (a / STOP_FILE).exists() and not (b / STOP_FILE).exists()
        assert all(f"stop_run(run_dir=" in o for o in r["options"])
    finally:
        ja.finish(0), jb.finish(0)


def test_with_no_live_run_and_no_folder_the_reply_is_none(mcp_server, tmp_path):
    r = mcp_server.stop_run()
    assert r["state"] == "none" and r["found_by"] == "the default run folder" and r["run_dir"] == str(tmp_path / "runs" / "live")


def sleeper_command(monkeypatch):
    """A run command that does nothing and never writes: a process that is alive and has taken no lock (a run that is starting)."""
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c 'import time; time.sleep(600)'")


def test_a_process_of_this_server_that_has_not_taken_the_lock_gets_the_request_for_its_pid(mcp_server, tmp_path, monkeypatch):
    sleeper_command(monkeypatch)
    yaml_path = make_case(tmp_path, h=1.0, iterations=4)
    out = tmp_path / "runs" / "s"
    started = mcp_server.run_topopt(str(yaml_path), str(out), proceed_without_brief=True)
    assert "error" not in started, started
    pid = started["pid"]
    r = mcp_server.stop_run()
    assert r["state"] == "stop_requested" and r["pid"] == pid and r["found_by"] == "the run that this server started"
    assert "starting; it has not taken the job lock yet" in r["message"]
    assert read(out / STOP_FILE)["target_pid"] == pid and read(out / STOP_FILE)["by"] == "mcp stop_run"
    st = mcp_server.run_status(str(out))
    assert st["state"] == "starting" and st["stop_request"]["target_pid"] == pid


# --- a stop of a run that this server started: the whole way -----------------------------------------------

def wait_state(call, states, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        st = call()
        if st["state"] in states:
            return st
        time.sleep(0.2)
    raise AssertionError(f"the run did not reach {states}: {st['state']}")


@needs_direct
def test_stop_run_stops_a_run_of_this_server_and_run_topopt_resumes_it(case, mcp_server, tmp_path, monkeypatch):
    yaml_path, base = case
    out = tmp_path / "runs" / "live"
    # the run command: the helper holds the run at the checkpoint of pass 6, with the dashboard on (it stays on after the stop)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} {STOP} hold:6 run {{yaml}} --out {{out}} --dashboard --port {{port}}")
    for k, v in ENV1.items():
        monkeypatch.setenv(k, v)
    started = mcp_server.run_topopt(str(yaml_path), str(out), proceed_without_brief=True)
    assert "error" not in started, started
    wait_for(out / "hold.reached")
    r = mcp_server.stop_run()
    assert r["state"] == "stop_requested" and r["pid"] == started["pid"] and r["iteration"] == 6
    assert r["found_by"] == "the run that this server started" and "does not read job.stop" in r["note"]
    st = mcp_server.run_status()
    assert st["stop_request"]["by"] == "mcp stop_run" and "a request to stop is pending" in st["summary"]
    release(out)
    st = wait_state(lambda: mcp_server.run_status(), ("stopped", "error"))
    assert st["state"] == "stopped", st
    # the process of this server is alive (the dashboard stays on, like after a done run); the job is stopped and holds no lock
    assert st["process_alive"] is True and st["job"]["state"] == "stopped" and st["job"]["marker"] == STOPPED_FILE
    assert not (out / LOCK_FILE).exists() and st["stop_request"] is None
    assert st["verdict"]["state"] == "stopped_by_request" and st["checkpoint"]["iteration"] == 6 and st["checkpoint"]["complete"] is False
    assert st["summary"].startswith("stopped: stopped by a request at pass 6 of 24; --resume continues at pass 7")
    assert "verdict: stopped by a request at pass 6 of 24. Tests:" in st["summary"]
    assert "checkpoint at pass 6 of 24: run_topopt with resume=true continues the run" in st["summary"]   # the hint tests the job, not the pid
    # a second stop_run: nothing to stop; the reply says that a process only serves the dashboard
    again = mcp_server.stop_run()
    assert again["state"] == "nothing_to_stop" and again["process_alive"] is True and "only serves the dashboard" in again["message"]
    assert any("continues the run at pass 7" in o for o in again["options"]) and not (out / STOP_FILE).exists()
    # the result of a stopped run
    res = mcp_server.run_result()
    assert res["done"] is False and res["stopped"] is True and "stopped by a request at pass 6 of 24" in res["message"]
    assert "run_topopt with resume=true continues the run" in res["message"]
    # the resume: the real command (no hold), with no override. The server ends the old dashboard process first.
    monkeypatch.delenv("TOPOPT_MCP_CMD")
    old_pid = started["pid"]
    r2 = mcp_server.run_topopt(str(yaml_path), str(out), resume=True, proceed_without_brief=True)
    assert "error" not in r2, r2
    st = wait_state(lambda: mcp_server.run_status(), ("done", "error"))
    assert st["state"] == "done", st
    assert not runjob.pid_alive(old_pid) or old_pid != r2["pid"]
    assert st["checkpoint"]["complete"] is True and "continues the run" not in st["summary"]
    assert st["verdict"]["resume_points"][0]["iteration"] == 6
    assert mcp_server.run_result()["done"] is True
    same_run(base, out)


@needs_direct
def test_a_run_of_this_server_with_no_wait_leaves_no_process(case, mcp_server, tmp_path, monkeypatch):
    yaml_path, base = case
    out = tmp_path / "runs" / "live"
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} {STOP} hold:6 run {{yaml}} --out {{out}} --dashboard --port {{port}} --no-wait")
    for k, v in ENV1.items():
        monkeypatch.setenv(k, v)
    started = mcp_server.run_topopt(str(yaml_path), str(out), proceed_without_brief=True)
    wait_for(out / "hold.reached")
    assert mcp_server.stop_run()["state"] == "stop_requested"
    release(out)
    st = wait_state(lambda: mcp_server.run_status(), ("stopped", "error"))
    assert st["state"] == "stopped", st
    assert mcp_server._current["proc"].wait(timeout=60) == STOPPED_EXIT      # the process has exited, with exit code 5
    st = mcp_server.run_status()
    assert st["process_alive"] is False and st["job"]["state"] == "stopped"
    again = mcp_server.stop_run()
    assert again["state"] == "nothing_to_stop" and again["process_alive"] is False and "only serves the dashboard" not in again["message"]


@needs_direct
def test_stop_run_finds_a_run_that_the_cli_started(case, mcp_server, work, tmp_path):
    yaml_path, base = case
    out = tmp_path / "runs" / "cli"
    proc = held_cli_run(work, out, 5)
    try:
        r = mcp_server.stop_run()                                        # no run_dir, no registry entry of this server
        assert r["state"] == "stop_requested" and r["pid"] == proc.pid and r["found_by"] == "the scan for a live job lock"
        assert r["iteration"] == 5
        st = mcp_server.run_status(str(out))
        assert st["job"]["state"] == "running" and st["stop_request"]["target_pid"] == proc.pid
    finally:
        release(out)
    assert proc.wait(timeout=300) == STOPPED_EXIT
    st = mcp_server.run_status(str(out))
    assert st["state"] == "stopped" and st["job"]["state"] == "stopped" and st["process_alive"] is False
    assert "checkpoint at pass 5 of 24: run_topopt with resume=true continues the run" in st["summary"]


def test_run_status_of_a_run_that_stopped_before_the_post_step_says_that_the_post_step_is_left(case, mcp_server, work, tmp_path):
    out = tmp_path / "runs" / "late"
    stopped_at(work, out, N_IT)
    st = mcp_server.run_status(str(out))
    assert st["state"] == "stopped" and st["verdict"]["state"] == "iteration_limit"
    assert "checkpoint at pass 24 of 24 is complete: run_topopt with resume=true runs the post step only" in st["summary"]
    res = mcp_server.run_result(str(out))
    assert res["done"] is False and res["stopped"] is True and "before the post step" in res["message"]


def test_the_server_text_names_the_tool():
    from topopt_mcp import server
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert "stop_run" in server.INSTRUCTIONS and len(server.INSTRUCTIONS) <= OLD_CLIENT_CUT
    assert "stop_run" in server.__doc__.split("Stdout is")[0]
    import asyncio
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert "stop_run" in tools
    assert "job.stop" in tools["stop_run"].description and "exit code 5" in tools["stop_run"].description
    assert "stopped_by_request" in tools["run_status"].description and "`stopped`" in tools["run_status"].description


# --- the other readers of the states ---------------------------------------------------------------------

def test_the_dashboard_page_shows_a_stopped_pill_and_a_banner_with_the_message():
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node")
    js = (REPO / "topopt/dashboard/js/app.js").read_text()
    fn = js[js.index("function setPill(st)"):js.index("async function pollStatus()")]
    script = (
        "const els = {pill: {}, banner: {hidden: true, className: '', textContent: ''}};\n"
        "const $ = id => els[id];\n" + fn + "\n"
        "const out = [];\n"
        "for (const st of [{state: 'stopped', step: 'stopped', message: 'stopped by a request at pass 6 of 24; --resume continues at pass 7'},\n"
        "                  {state: 'stopped', step: 'stopped', message: ''}, {state: 'error', step: 'solve', message: 'boom'},\n"
        "                  {state: 'done', step: 'done', message: ''}, {state: 'running', step: 'solve', message: ''}]) {\n"
        "  setPill(st);\n"
        "  out.push([els.pill.textContent, els.pill.className, els.banner.hidden, els.banner.className, els.banner.textContent]);\n"
        "}\n"
        "console.log(JSON.stringify(out));\n")
    res = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    stopped, stopped_blank, error, done, running = json.loads(res.stdout)
    assert stopped == ["STOPPED", "pill stopped", False, "stopped", "stopped by a request at pass 6 of 24; --resume continues at pass 7"]
    assert stopped_blank[3:] == ["stopped", "The run was stopped by a request."]
    assert error == ["ERROR", "pill error", False, "", "boom"] and done[2] is True and running[2] is True
    assert ".pill.stopped" in (REPO / "topopt/dashboard/css/style.css").read_text() and "#banner.stopped" in (REPO / "topopt/dashboard/css/style.css").read_text()


def stopped_folder(tmp_path):
    """A small run folder of a run that a request stopped: run.json, status, job files, no result/."""
    d = tmp_path / "stopped"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"format": 1, "name": "x", "shape": [4, 4, 4], "h": 1.0, "origin": [0, 0, 0], "iterations": 24}))
    (d / "status.json").write_text(json.dumps({"state": "stopped", "iteration": 7, "step": "stopped", "elapsed_s": 3.0,
                                               "message": "stopped by a request at pass 7 of 24; --resume continues at pass 8"}))
    run_job(d, lambda job: STOPPED_EXIT)
    return d


def test_run_history_and_list_runs_show_the_state_and_no_missing_result_problem(mcp_server, tmp_path):
    d = stopped_folder(tmp_path)
    shutil.copytree(d, tmp_path / "runs" / "stopped")
    hist = mcp_server.run_history(str(tmp_path / "runs"))
    run = hist["runs"][0]
    assert run["state"] == "stopped" and run["iteration"] == 7 and run["message"].startswith("stopped by a request")
    assert not any("final.stl" in p or "metrics.json" in p for p in run["problems"])
    listed = mcp_server.list_runs()["runs"]
    assert listed[0]["state"] == "stopped" and listed[0]["process_alive"] is False


def test_compare_runs_names_the_state_of_a_stopped_run(mcp_server, tmp_path):
    from topopt import feedback
    d = stopped_folder(tmp_path)
    facts = feedback.run_facts(d)
    assert facts["state"] == "stopped" and facts["iteration"] == 7 and facts["metrics"] is None
    assert not any("final.stl" in p for p in facts["problems"])


def test_a_reader_that_needs_a_result_says_that_the_run_was_stopped_and_how_to_resume(tmp_path):
    d = stopped_folder(tmp_path)
    from topopt import cleanup
    res = cleanup.cleanup_list(d)
    assert res["ok"] is False and "is missing" in res["error"] and "was stopped by a request" in res["error"] and "--resume" in res["error"]
    # a folder that is not a stopped run has no such sentence
    assert runjob.stopped_hint(tmp_path) == ""
    other = tmp_path / "failed"
    other.mkdir()
    (other / "status.json").write_text(json.dumps({"state": "error", "message": "boom"}))
    assert runjob.stopped_hint(other) == ""


def test_the_buckling_command_says_that_the_run_was_stopped(tmp_path, capsys):
    d = stopped_folder(tmp_path)
    assert main(["buckling", str(d)]) == 1
    err = capsys.readouterr().err
    assert "not finished" in err and "was stopped by a request" in err and "--resume" in err


def test_apply_marks_says_that_the_run_was_stopped(tmp_path):
    from topopt import apply_marks
    d = stopped_folder(tmp_path)
    res = apply_marks.apply_marks(d, {"version": 1, "marks": []})
    assert res["ok"] is False and "there is no finished part to edit" in res["error"] and "was stopped by a request" in res["error"]


def test_replay_of_a_stopped_run_ends_as_stopped_and_has_no_post_step(tmp_path):
    from topopt.replay import Replay
    sys.path.insert(0, str(REPO / "tools"))
    import fake_run

    def folder(name, final_status):
        out = tmp_path / name
        fake_run.run(out, spi=0.0, n_it=6)                    # the rows are passes 0 to 6
        rows = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines()]
        for r in rows:
            r["elapsed_s"] = 10.0 + 5.0 * r["iteration"]      # the last row is at 40 s
        (out / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / "status.json").write_text(json.dumps(final_status))     # the run ends at 50 s
        return out

    msg = "stopped by a request at pass 6 of 12; --resume continues at pass 7"
    stopped = folder("stopped", {"state": "stopped", "iteration": 6, "step": "stopped", "elapsed_s": 50.0, "message": msg})
    shutil.rmtree(stopped / "result")                         # a stopped run makes no result/
    finished = folder("finished", {"state": "done", "iteration": 6, "step": "done", "elapsed_s": 50.0, "message": ""})
    clocks = {}
    replays = {name: Replay(d, now=lambda n=name: clocks[n]) for name, d in (("stopped", stopped), ("finished", finished))}
    for name, rp in replays.items():
        clocks[name] = 0.0
        rp.start()
    for name, rp in replays.items():
        clocks[name] = 45.0                                    # after the last row, before the end
        assert rp.status()["state"] == ("running" if name == "stopped" else "postprocess"), name   # a stopped run has no post step
        clocks[name] = 1e6
    st = replays["stopped"].status()
    assert (st["state"], st["iteration"], st["step"], st["message"]) == ("stopped", 6, "stopped", msg)
    assert replays["finished"].status()["state"] == "done"      # a replay of a finished run is unchanged


def test_the_checkpoint_of_a_stopped_run_is_an_unfinished_one_for_the_readers_of_it(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    stopped_at(yaml_path, out, 8)
    cp = C.describe(out)
    assert cp["complete"] is False and cp["iteration"] == 8 and cp["iterations"] == N_IT
    from topopt.feedback import run_history
    runs = run_history(out)["runs"]
    assert runs[0]["state"] == "stopped" and runs[0]["metrics_source"].startswith("history.jsonl")


def test_a_fresh_run_in_the_folder_of_a_stopped_run_says_that_it_deletes_the_checkpoint(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    stopped_at(yaml_path, out, 3)
    (tmp_path / "short").mkdir()
    short = make_case(tmp_path / "short", h=1.0, iterations=2)
    assert main(["run", str(short), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text()
    assert "had a checkpoint at pass 3 of 24" in log and "from a run that did not finish" in log
    assert not (out / STOPPED_FILE).exists() and (out / DONE_FILE).exists()


# --- a request is read by the process it names (m1 of the review) ---------------------------------------------

def test_read_own_stop_gives_the_request_to_the_process_it_names_or_to_any_process_when_it_names_none(tmp_path):
    assert runjob.read_own_stop(tmp_path) is None
    request_stop(tmp_path, "test", os.getpid())
    assert runjob.read_own_stop(tmp_path)["target_pid"] == os.getpid()
    clear_stop(tmp_path)
    request_stop(tmp_path, "test", os.getpid() + 1000003)
    assert runjob.read_own_stop(tmp_path) is None and read_stop(tmp_path)["target_pid"] == os.getpid() + 1000003      # it stays
    clear_stop(tmp_path)
    request_stop(tmp_path, "test")
    assert runjob.read_own_stop(tmp_path) is not None                                       # no pid named: any process takes it
    (tmp_path / STOP_FILE).write_text("")                                                   # `touch job.stop`: also any process
    assert runjob.read_own_stop(tmp_path) is not None


@needs_direct
def test_a_request_that_names_another_process_does_not_stop_this_run_and_stays_in_the_folder(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "run"
    done = stop_run_child(yaml_path, out, "other:5")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-2500:]
    assert read(out / "status.json")["state"] == "done" and read(out / INFO_FILE)["state"] == "done"
    assert (out / STOP_FILE).is_file() and read(out / STOP_FILE)["target_pid"] != read(out / INFO_FILE)["pid"]    # still there, for its owner
    assert not any(line.startswith(("stop:", "verdict: stopped")) for line in (out / "run.log").read_text().splitlines())
    same_run(base, out)                                                                      # the request changed nothing in the run


@needs_direct
def test_with_the_override_the_process_that_the_request_names_stops_and_the_other_runs_on(case, tmp_path, capsys):
    """tools/override_repro.py of the review: A holds at pass 5, B takes the lock (--override-lock) and holds at pass 2, `topopt stop`
    names B (the lock holder). A must not take the request. B stops with exit 5."""
    yaml_path, _ = case
    out = tmp_path / "run"
    a = held_cli_run(yaml_path, out, 5)
    b = subprocess.Popen([sys.executable, str(STOP), "hold:2:b", "run", str(yaml_path), "--out", str(out), "--quiet", "--override-lock"],
                         env=ENV1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for(out / "hold-b.reached")
        assert read(out / LOCK_FILE)["pid"] == b.pid
        assert main(["stop", str(out)]) == 0
        capsys.readouterr()
        assert read(out / STOP_FILE)["target_pid"] == b.pid                                  # the request names the lock holder
        release(out)                                                                         # A goes on: it reads at the end of pass 5
        a_out, a_err = a.communicate(timeout=600)
        assert a.returncode != STOPPED_EXIT, a_err[-1500:]                                   # A did not take the request of B
        assert read(out / STOP_FILE)["target_pid"] == b.pid                                  # it is still in the folder for B
        assert "took the lock over" in a_err
        with open(out / "hold-b.fifo", "wb") as f:                                           # B goes on: it reads at the end of pass 2
            f.write(b"go")
        b_out, b_err = b.communicate(timeout=600)
        assert b.returncode == STOPPED_EXIT, b_err[-2500:]
        assert not (out / STOP_FILE).exists()
        job = read(out / INFO_FILE)
        assert job["state"] == "stopped" and job["pid"] == b.pid and (out / STOPPED_FILE).read_text().strip() == "exit 5"
        assert read(out / "status.json")["state"] == "stopped" and read(out / "status.json")["iteration"] == 2
    finally:
        for proc in (a, b):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


# --- the refusals "a job is still running" name the options that are true (m2 of the review) -------------------

def busy_calls(server, monkeypatch, tmp_path):
    """The three tools that refuse a start while a job works, with the checks before the refusal made to pass."""
    from types import SimpleNamespace
    from topopt import ratchet as rt
    from topopt.verify import run as vrun
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("name: x\n")
    folder = tmp_path / "runs" / "r"
    folder.mkdir(parents=True)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    monkeypatch.setattr(vrun, "preflight", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(rt, "check_report", lambda *a, **k: {})
    return folder, {
        "run_topopt": lambda: server.run_topopt(str(yaml_path), proceed_without_brief=True),
        "verify_run": lambda: server.verify_run(run_dir=str(folder), estimate_only=False),
        "ratchet_run": lambda: server.ratchet_run(run_dir=str(folder), source=f"stl:{tmp_path}", estimate_only=False),
    }


def test_the_refusal_while_an_optimizer_run_works_names_stop_run_and_the_resume(mcp_server, tmp_path, monkeypatch):
    folder, calls = busy_calls(mcp_server, monkeypatch, tmp_path)
    monkeypatch.setattr(mcp_server, "_alive_verify", lambda: None)
    monkeypatch.setattr(mcp_server, "_alive_ratchet", lambda: None)
    monkeypatch.setattr(mcp_server, "_alive_run", lambda: (folder, 4242))
    monkeypatch.setattr(mcp_server, "_is_busy", lambda d: True)
    for name, call in calls.items():
        res = call()
        err = res["error"]
        assert "still running" in err and res["pid"] == 4242, (name, res)
        assert "call stop_run" in err and "after its current pass" in err and "checkpoint is kept" in err and "resume=true" in err, (name, err)
        assert "older build ignores the request" in err, (name, err)


@pytest.mark.parametrize("kind, alive", [("verify", "_alive_verify"), ("ratchet", "_alive_ratchet")])
def test_the_refusal_while_a_verify_or_ratchet_job_works_says_that_stop_run_does_not_stop_it(mcp_server, tmp_path, monkeypatch, kind, alive):
    folder, calls = busy_calls(mcp_server, monkeypatch, tmp_path)
    monkeypatch.setattr(mcp_server, "_alive_verify", lambda: (folder, 777) if kind == "verify" else None)
    monkeypatch.setattr(mcp_server, "_alive_ratchet", lambda: (folder, 777) if kind == "ratchet" else None)
    monkeypatch.setattr(mcp_server, "_alive_run", lambda: None)
    for name, call in calls.items():
        err = call()["error"]
        assert f"{kind} job is still running" in err, (name, err)
        assert f"stop_run does not stop a {kind} job: a {kind} job reads no job.stop" in err, (name, err)
        assert "call stop_run" not in err, (name, err)


# --- a stop and a resume equal a run that was not stopped, also with the rows that keep state (m3 of the review) --

def stop_resume_equals_base(yaml_path, tmp_path, k, expect=()):
    """Stop at pass `k` (a request after the checkpoint), resume with no override, and compare with a run that was not stopped."""
    base, out = tmp_path / "base", tmp_path / "r"
    finish(yaml_path, base)
    stopped_at(yaml_path, out, k)
    assert V.read_verdict(out)["state"] == "stopped_by_request" and read(out / "status.json")["state"] == "stopped"
    assert read(out / "status.json")["iteration"] == k and V.read_verdict(out)["stop_request"]["iteration"] == k      # the pass of the request
    assert C.describe(out)["iteration"] == k and not C.describe(out)["complete"]                                # and its checkpoint
    assert not (out / "result").exists()
    done = finish(yaml_path, out, "--resume")
    same_run(base, out)
    for text in expect:
        assert text in (out / "run.log").read_text(), text
    return base, out, done


@needs_direct
def test_a_stopped_stress_run_with_step_acceptance_resumes_to_the_same_bytes(tmp_path):
    """The stress row keeps a KS scale between passes, and step acceptance carries a design and its solve (`carry_U`)."""
    from test_stress_report import stress_yaml
    yaml_path = stress_yaml(tmp_path, iterations=24)
    base, out, _ = stop_resume_equals_base(yaml_path, tmp_path, 9)
    assert any("step_acceptance" in r for r in history(out))


@needs_direct
def test_a_stopped_run_with_the_perimeter_row_resumes_to_the_same_bytes_and_the_same_block(tmp_path):
    """The stop skips `p_final` of the perimeter block; the resume writes it. The block equals the one of the run that was not stopped."""
    from test_perimeter import cantilever, sub
    yaml_path = cantilever(sub(tmp_path, "case"), perimeter_max_mm2=900.0, iterations=24)
    base, out, _ = stop_resume_equals_base(yaml_path, tmp_path, 12)
    a, b = (json.loads((d / "run.json").read_text())["perimeter"] for d in (base, out))
    assert a == b and "p_final_mm2" in b


@needs_direct
def test_a_stop_with_the_perimeter_row_leaves_the_block_without_the_final_value(tmp_path):
    from test_perimeter import cantilever, sub
    yaml_path = cantilever(sub(tmp_path, "case"), perimeter_max_mm2=900.0, iterations=24)
    out = tmp_path / "r"
    stopped_at(yaml_path, out, 12)
    assert "p_final_mm2" not in read(out / "run.json")["perimeter"]                 # the stop skipped it, as a kill does


@needs_direct
def test_a_stopped_robust_run_resumes_to_the_same_bytes_and_the_same_robust_record(tmp_path):
    """Robust mode: `robust.finish_run` waits for the resume. `robust.json` and the metrics equal the run that was not stopped."""
    yaml_path = make_case(tmp_path, h=2.0, iterations=24)
    edit(yaml_path, lambda d: d["settings"].update(robust="eroded"))
    base, out, _ = stop_resume_equals_base(yaml_path, tmp_path, 12)
    assert (base / "robust.json").is_file() and (out / "robust.json").is_file()
    assert clean(read(base / "robust.json")) == clean(read(out / "robust.json"))


@needs_direct
def test_a_stopped_run_with_a_mirror_resumes_to_the_same_bytes(tmp_path):
    from test_mirror import two_planes, write_yaml
    yaml_path = write_yaml(tmp_path, two_planes(), iterations=24, beta_schedule=[[0, 1], [10, 2], [20, 4]])
    stop_resume_equals_base(yaml_path, tmp_path, 12)


def test_a_body_that_raises_system_exit_5_ends_its_job_as_stopped(tmp_path):
    def body(job):
        raise SystemExit(STOPPED_EXIT)
    with pytest.raises(SystemExit):
        run_job(tmp_path, body)
    job = read(tmp_path / INFO_FILE)
    assert job["state"] == "stopped" and job["exit_code"] == STOPPED_EXIT
    assert (tmp_path / STOPPED_FILE).read_text().strip() == "exit 5" and not (tmp_path / DONE_FILE).exists() and not (tmp_path / FAILED_FILE).exists()
    assert not (tmp_path / LOCK_FILE).exists() and job_facts(tmp_path)["state"] == "stopped"


# --- every run_status reply has the key stop_request (n2 of the review) ------------------------------------------

def test_every_run_status_reply_has_the_key_stop_request(mcp_server, tmp_path, monkeypatch):
    empty = tmp_path / "runs" / "empty"
    empty.mkdir(parents=True)
    none = mcp_server.run_status(str(empty))
    assert none["state"] == "none" and "stop_request" in none and none["stop_request"] is None
    request_stop(empty, "test", 123)
    assert mcp_server.run_status(str(empty))["stop_request"]["target_pid"] == 123           # a request in a folder with no run data is shown
    clear_stop(empty)
    # the early error: the run process of this server ended, and the folder holds no run data
    with monkeypatch.context() as patched:
        patched.setattr(mcp_server, "_own_exit", lambda d: 1)
        early = mcp_server.run_status(str(empty))
    assert early["state"] == "error" and "stop_request" in early and early["stop_request"] is None
    # a run folder with data: the key is there, null without a request
    d = tmp_path / "runs" / "a"
    job = live_job(d, {"state": "running", "iteration": 4, "step": "solve", "elapsed_s": 1.0, "message": ""})
    try:
        assert mcp_server.run_status(str(d))["stop_request"] is None
        request_stop(d, "test", os.getpid())
        assert mcp_server.run_status(str(d))["stop_request"]["by"] == "test"
    finally:
        job.finish(0)


# --- the texts of none, nothing_to_stop and gone (n3, n5 and n7 of the review) ------------------------------------

def test_a_folder_that_does_not_exist_is_named_and_no_folder_is_made(mcp_server, tmp_path):
    for missing in (tmp_path / "runs" / "hoook", tmp_path / "deep" / "er" / "path"):
        r = mcp_server.stop_run(str(missing))
        assert r["state"] == "none" and f"The folder {missing} does not exist" in r["message"] and "no folder is made" in r["message"]
        assert not missing.exists()
    assert not (tmp_path / "deep").exists() and not (tmp_path / "runs").exists()      # not even a parent folder is made


def test_none_and_nothing_to_stop_say_that_a_starting_run_is_not_seen(mcp_server, tmp_path):
    texts = []
    empty = tmp_path / "empty"
    empty.mkdir()
    texts.append(mcp_server.stop_run(str(empty))["message"])                                       # none
    for name, code in (("d", 0), ("s", STOPPED_EXIT), ("f", 2)):
        run_job(tmp_path / name, lambda job, code=code: code)
        texts.append(mcp_server.stop_run(str(tmp_path / name))["message"])                         # nothing_to_stop
    assert len(texts) == 4
    for text in texts:
        assert "has not taken the job lock yet is not seen here: call again" in text, text


def test_the_run_process_of_this_server_that_ended_before_the_lock_gives_gone_with_the_exit_and_the_log(mcp_server, tmp_path):
    d = tmp_path / "runs" / "own"
    d.mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    proc.wait()
    mcp_server._current.update(proc=proc, run_dir=d, started=time.time())
    r = mcp_server.stop_run(str(d))
    log = tmp_path / "runs" / "own.stdout"
    assert r["state"] == "gone" and "ended with exit 3 before it took the job lock" in r["message"] and str(log) in r["message"]
    assert r["job"] is None and not (d / STOP_FILE).exists() and any(str(log) in o for o in r["options"])
    # the same folder when no process of this server was there: the plain none
    mcp_server._current.clear()
    assert mcp_server.stop_run(str(d))["state"] == "none"


# --- two writers of a request (n4 of the review) -------------------------------------------------------------------

def test_request_stop_leaves_no_temp_file_and_a_second_writer_that_wins_is_reported(tmp_path, monkeypatch):
    first, created = request_stop(tmp_path, "writer A", 11)
    assert created is True and sorted(p.name for p in tmp_path.iterdir()) == [STOP_FILE]           # no temp file stays
    clear_stop(tmp_path)
    real_replace = os.replace
    other = {"format": 1, "requested_at": "2026-10-22T20:00:00+00:00", "by": "writer B", "target_pid": 22}

    def replace_then_lose(src, dst):
        real_replace(src, dst)
        Path(dst).write_text(json.dumps(other))        # writer B moves its file after writer A did and before A reads the file back

    monkeypatch.setattr(os, "replace", replace_then_lose)
    request, created = request_stop(tmp_path, "writer A", 11)
    monkeypatch.undo()
    assert created is False and request == other == read_stop(tmp_path)          # A reports what is on disk: B won
    assert sorted(p.name for p in tmp_path.iterdir()) == [STOP_FILE]


def test_the_reply_of_stop_run_for_a_writer_that_lost_is_already_requested(mcp_server, tmp_path, monkeypatch):
    d = tmp_path / "runs" / "a"
    job = live_job(d, {"state": "running", "iteration": 3, "step": "solve", "elapsed_s": 1.0, "message": ""})
    real_replace = os.replace
    other = {"format": 1, "requested_at": "2026-10-22T20:00:00+00:00", "by": "writer B", "target_pid": os.getpid()}

    def replace_then_lose(src, dst):
        real_replace(src, dst)
        Path(dst).write_text(json.dumps(other))

    try:
        with monkeypatch.context() as patched:
            patched.setattr(os, "replace", replace_then_lose)
            r = mcp_server.stop_run(str(d))
        assert r["state"] == "already_requested" and "writer B" in r["message"] and r["stop_request"] == other
    finally:
        job.finish(0)


def test_writers_at_the_same_time_all_return_and_one_request_stays(tmp_path):
    import threading
    start = threading.Barrier(6)
    results, errors = [], []

    def writer(n):
        try:
            start.wait()
            results.append(request_stop(tmp_path, f"writer {n}", n + 1))
        except BaseException as exc:            # the point of the test: no writer raises
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == [] and len(results) == 6
    assert any(created for _, created in results)
    assert read_stop(tmp_path)["by"].startswith("writer ") and sorted(p.name for p in tmp_path.iterdir()) == [STOP_FILE]


def test_a_second_writer_between_the_write_and_the_move_does_not_make_the_first_one_fail(tmp_path, monkeypatch):
    """The review (n4): both writers found no request. With one shared temp name the second move raised FileNotFoundError."""
    real_replace = os.replace
    calls = []

    def replace_after_the_other_writer(src, dst):
        if not calls:
            calls.append(1)
            assert request_stop(tmp_path, "writer B", 22)[1] is True      # writer B runs its whole call between A's write and A's move
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_after_the_other_writer)
    request, created = request_stop(tmp_path, "writer A", 11)
    monkeypatch.undo()
    assert created is True and request["by"] == "writer A" == read_stop(tmp_path)["by"]       # A moved last, and it reports its own file
    assert sorted(p.name for p in tmp_path.iterdir()) == [STOP_FILE]


# --- texts (n6 of the review) ---------------------------------------------------------------------------------------

def test_the_note_the_run_docstring_and_the_readme_use_the_corrected_texts():
    assert runjob.STOP_NOTE.startswith("A topopt build from before this change does not read job.stop. A run of such a build ignores the request.")
    from topopt import optimize
    assert "one file read per pass" in optimize.run.__doc__ and "`stat`" not in optimize.run.__doc__
    readme = " ".join((REPO / "docs" / "reference" / "problem-and-run.md").read_text().split())
    assert "`job.STOPPED`" in readme and "exit code 5" in readme and "`topopt stop`" in readme


# --- CONTRACTS 24.2 follows the code (m4, n1, n8 of the review) -------------------------------------------------------

def stop_block():
    """The block "Stop on request" of CONTRACTS.md section 24.2, with the white space made single."""
    text = (REPO / "CONTRACTS.md").read_text()
    start = text.index("**Stop on request**")
    return " ".join(text[start:text.index("**Verdict.**", start)].split())


def test_the_contract_names_functions_and_texts_that_the_code_has():
    block = stop_block()
    for name in ("request_stop", "read_stop", "read_own_stop", "clear_stop", "stop_reply", "is_run_folder"):
        assert hasattr(runjob, name) and f"`{name}`" in block, name
    assert "`runjob.STOP_NOTE`" in block and "`runjob.NOT_SEEN`" in block and runjob.NOT_SEEN
    # m4: the scan counts run folders only, and a request is read only by the process that it names
    assert "run folders only" in block and "`target_pid` is null or is its own pid" in block
    assert "the history line, the snapshot and the checkpoint of pass k are those of a run that goes on" in block.lower()
    # n8: the second read point also serves a pass at which a stop rule fired; only a job of this build removes a request
    assert "or during a pass at which a stop rule fired" in block and "Only a job of this build removes it" in block
    assert "continues to serve the dashboard" in block


def test_requested_at_is_local_time_with_the_utc_offset_as_the_contract_says(tmp_path):
    from datetime import datetime
    request, _ = runjob.request_stop(tmp_path, "test")
    stamp = datetime.fromisoformat(request["requested_at"])
    assert stamp.utcoffset() is not None                              # an offset, not a naive time
    assert "local time with the UTC offset, ISO (as in `job.json`)" in stop_block()
