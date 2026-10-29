"""Tests of the live time estimate after pass 2 (E1-P2, CONTRACTS.md section 24.5).

`topopt.eta` is a pure function from the history records, the planned passes and the free dofs to "about N s left (up to M s)".
It shows in `run.log` (one line), in `run_status` (`time_left`, and the summary), and in the bar of the dashboard (the key `time_left`
of `/api/status`). The runs here are small CPU runs in child processes (1 thread) and hand-made run folders.
The constants are from the E1 run-time study: the tests pin them, so a change of a number is a deliberate change.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from test_checkpoint import finish, history, killed, log, run_args  # noqa: E402
from test_optimize import make_case  # noqa: E402


@pytest.fixture
def eta():
    from topopt import eta as module     # imported here: a missing module fails each test with one clear error
    return module


def approx(value, abs=0.06):
    """The estimate has one decimal (`left_s`, `up_to_s`), so a value is right to 0.05 s."""
    return pytest.approx(value, abs=abs)


def rec(it, t, **extra):
    return {"iteration": it, "elapsed_s": 10.0 + 3.0 * it, "compliance": 100.0 / (it + 1), "volume_fraction": 0.4, "t_iter_s": t, **extra}


# 5.0 s is the slow pass 0; the passes 1 and 2 take 2.0 s and 4.0 s (mean 3.0 s); later passes take 9.0 s, which the estimate must not use.
RECORDS = [rec(0, 5.0), rec(1, 2.0), rec(2, 4.0), rec(3, 9.0), rec(4, 9.0), rec(5, 9.0)]


# --- the pure function ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_records", [0, 1, 2])
def test_no_estimate_before_pass_2_is_done(eta, n_records):
    est, why = eta.time_left(RECORDS[:n_records], 10, 100_000)
    assert est is None
    assert "no estimate before pass 2 is done" in why
    assert ("no pass is done" if n_records == 0 else f"the last pass done is {n_records - 1}") in why


def test_after_pass_2_the_estimate_is_the_formula(eta):
    est, why = eta.time_left(RECORDS[:3], 10, 100_000)
    assert why is None
    # mean(2.0, 4.0) = 3.0 s; N = 10, k = 2: 8 passes left; 100,000 free dofs: the post step is 7.51 s.
    assert est["left_s"] == approx(3.0 * 8 + 7.51)
    assert est["up_to_s"] == approx((3.0 * 8 + 7.51) * 1.33, abs=0.05)
    assert est["after_pass"] == 2
    b = est["basis"]
    assert b["pass_s"] == 3.0 and b["probe_passes"] == [1, 2] and b["passes_left"] == 8
    assert b["post_s"] == 7.51 and b["post_band"] == "50,000 to 199,999 free dofs" and b["bound_factor"] == 1.33
    assert "stop rule" in b["assumes"] and "earlier" in b["assumes"]
    assert "not_measured" not in b          # 100,000 free dofs: inside the range that E1 measured


def test_the_bound_is_1_33_times_the_point_value(eta):
    est, _ = eta.time_left(RECORDS[:3], 50, 20_000)
    assert est["up_to_s"] == approx(est["left_s"] * 1.33, abs=0.12)
    assert eta.BOUND_FACTOR == 1.33


def test_a_later_pass_keeps_the_passes_1_and_2_and_counts_fewer_passes_left(eta):
    """No new rule for the later passes: the mean stays the one of the passes 1 and 2, the passes left fall by one for each pass."""
    for k in (3, 4, 5):
        est, _ = eta.time_left(RECORDS[:k + 1], 10, 100_000)
        assert est["after_pass"] == k
        assert est["left_s"] == approx(3.0 * (10 - k) + 7.51)
        assert est["basis"]["pass_s"] == 3.0 and est["basis"]["passes_left"] == 10 - k


def test_off_by_one_the_loop_runs_the_passes_0_to_N(eta):
    """`optimize.run` runs `range(first, N + 1)`: after pass k, N - k passes are left (E1 model C: N - 2 after pass 2), not N - 1 - k."""
    est, _ = eta.time_left(RECORDS[:3], 3, 100_000)
    assert est["basis"]["passes_left"] == 1                  # only pass 3 is left in a run of iterations: 3
    est, _ = eta.time_left(RECORDS[:3], 2, 100_000)
    assert est["basis"]["passes_left"] == 0 and est["left_s"] == approx(7.51)   # pass 2 was the last pass: the post step is left
    est, _ = eta.time_left(RECORDS[:6], 3, 100_000)          # more records than planned passes: never a negative count
    assert est["basis"]["passes_left"] == 0


@pytest.mark.parametrize("dofs,band,post", [
    (1_000, "below 50,000 free dofs", 4.43), (49_999, "below 50,000 free dofs", 4.43),
    (50_000, "50,000 to 199,999 free dofs", 7.51), (199_999, "50,000 to 199,999 free dofs", 7.51),
    (200_000, "200,000 free dofs and more", 21.67), (900_000, "200,000 free dofs and more", 21.67)])
def test_the_post_step_is_the_median_of_the_size_band(eta, dofs, band, post):
    assert eta.post_band(dofs) == (band, post)
    est, _ = eta.time_left(RECORDS[:3], 2, dofs)             # no pass left: the estimate is the post step alone
    assert est["left_s"] == approx(post) and est["basis"]["post_band"] == band


def test_a_run_below_10000_free_dofs_says_that_e1_did_not_measure_it(eta):
    est, _ = eta.time_left(RECORDS[:3], 10, 9_999)
    assert "not_measured" in est["basis"] and "10,000" in est["basis"]["not_measured"]
    assert "Experiment E1 (run-time study) did not measure" in eta.log_line(est)      # review n7: "E1" alone is also a series of lever runs
    assert est["basis"]["not_measured"].startswith("experiment E1 (run-time study) measured")
    est, _ = eta.time_left(RECORDS[:3], 10, 10_000)
    assert "not_measured" not in est["basis"] and "did not measure" not in eta.log_line(est)


def test_a_continued_run_counts_the_passes_from_the_start_of_this_process(eta):
    """A process that starts at pass 12 (resume or continuation) is probed with ITS passes 13 and 14, not with the passes 1 and 2 of the file."""
    old = [rec(i, 1.0) for i in range(12)]                   # the passes of the earlier process: 1.0 s each
    mine = [rec(12, 9.0), rec(13, 2.0), rec(14, 4.0), rec(15, 7.0)]
    for k in (13,):                                          # this process has made 2 passes only: no estimate yet
        est, why = eta.time_left(old + mine[:k - 11], 20, 100_000, start=12)
        assert est is None and "no estimate before pass 14 is done" in why and "this process started at pass 12" in why
    est, why = eta.time_left(old + mine[:3], 20, 100_000, start=12)
    assert why is None and est["after_pass"] == 14
    assert est["basis"]["probe_passes"] == [13, 14] and est["basis"]["pass_s"] == 3.0
    assert est["left_s"] == approx(3.0 * 6 + 7.51)    # N - k = 20 - 14
    wrong, _ = eta.time_left(old + mine[:3], 20, 100_000, start=0)   # the same file read from pass 0: the old process gives 1.5 s
    assert wrong["basis"]["pass_s"] == 1.0 and wrong["left_s"] != est["left_s"]


def test_a_process_that_starts_after_the_last_pass_has_no_pass_left_to_run(eta):
    """A resume of a finished run: `first_pass` is N + 1 and no pass runs. The reason must not name a pass that does not exist (review n1)."""
    for start in (6, 7, 40):
        est, why = eta.time_left([rec(i, 2.0) for i in range(6)], 5, 100_000, start=start)
        assert est is None and "no pass is left to run" in why and "no estimate before pass" not in why
    est, why = eta.time_left(RECORDS[:3], 2, 100_000, start=0)         # pass 2 is the last pass: the post step is left (as before)
    assert why is None and est["basis"]["passes_left"] == 0


def test_the_last_record_of_a_pass_wins_and_a_gap_is_a_reason(eta):
    est, _ = eta.time_left([rec(0, 5.0), rec(1, 99.0), rec(1, 2.0), rec(2, 4.0)], 10, 100_000)
    assert est["basis"]["pass_s"] == 3.0
    est, why = eta.time_left([rec(0, 5.0), rec(2, 4.0), rec(3, 4.0)], 10, 100_000)   # pass 1 is not in the file
    assert est is None and "pass 1" in why


@pytest.mark.parametrize("bad", [None, 0, -1.0, "2.0", float("nan"), float("inf"), True])
def test_a_record_without_a_usable_pass_time_gives_a_reason(eta, bad):
    records = [rec(0, 5.0), rec(1, bad), rec(2, 4.0)]
    est, why = eta.time_left(records, 10, 100_000)
    assert est is None and "t_iter_s for pass 1" in why


def test_missing_inputs_give_a_reason_not_an_error(eta):
    assert "iterations" in eta.time_left(RECORDS[:3], None, 100_000)[1]
    assert "n_free_dofs" in eta.time_left(RECORDS[:3], 10, None)[1]
    assert "first pass" in eta.time_left(RECORDS[:3], 10, 100_000, start=None)[1]
    assert eta.time_left(None, 10, 100_000)[0] is None and eta.time_left([], 10, 100_000)[0] is None
    assert eta.time_left(["x", None, {"iteration": "a"}], 10, 100_000)[0] is None


def test_the_text_and_the_log_line_say_it_holds_if_no_stop_rule_or_request_to_stop_ends_the_run_earlier(eta):
    est, _ = eta.time_left(RECORDS[:3], 10, 100_000)
    assert eta.left_text(est) == "about 32 s left (up to 42 s)"
    line = eta.log_line(est)
    assert line.startswith("fact: time: about 32 s left after pass 2 (up to 42 s; ")
    assert "the passes 1 and 2 took 3.00 s on average" in line and "passes left: 8" in line and "post step takes about 7.51 s" in line
    assert "if no stop rule or request to stop ends the run earlier" in line and "\n" not in line
    assert "stop rule or a request to stop" in est["basis"]["assumes"]       # the same two causes in the basis
    assert "conservative" not in line.lower() and "conservative" not in (eta.__doc__ or "").lower()


# --- the readers ----------------------------------------------------------------------------------

def test_rows_of_bytes_skips_a_cut_line_and_a_bad_line(eta):
    raw = (json.dumps(rec(0, 1.0)) + "\n\nnot json\n" + json.dumps([1]) + "\n" + json.dumps(rec(1, 2.0)) + "\n" + '{"iteration": 2, "t_it').encode()
    assert [r["iteration"] for r in eta.rows_of_bytes(raw)] == [0, 1]
    assert eta.rows_of_bytes(None) == [] and eta.rows_of_bytes(b"") == []


def test_read_rows_of_a_folder(eta, tmp_path):
    assert eta.read_rows(tmp_path) == []                     # no file
    (tmp_path / "history.jsonl").write_text("\n".join(json.dumps(r) for r in RECORDS[:2]) + "\n")
    assert [r["iteration"] for r in eta.read_rows(tmp_path)] == [0, 1]


def test_run_time_left_needs_a_run_in_its_passes_and_reads_nothing_otherwise(eta):
    def boom():
        raise AssertionError("the history is read")
    run = {"iterations": 10, "n_free_dofs": 100_000}
    for state in ("starting", "postprocess", "done", "error", "stopped", None):
        est, why = eta.run_time_left(boom, run, {"state": state})
        assert est is None and "not in its passes" in why
    assert eta.run_time_left(boom, run, None)[0] is None
    est, _ = eta.run_time_left(lambda: RECORDS[:3], run, {"state": "running"})
    assert est["left_s"] == approx(31.51)


def test_run_time_left_takes_the_first_pass_from_status_json(eta):
    old = [rec(i, 1.0) for i in range(4)]
    mine = [rec(4, 9.0), rec(5, 2.0), rec(6, 4.0)]
    run = {"iterations": 12, "n_free_dofs": 100_000}
    est, _ = eta.run_time_left(lambda: old + mine, run, {"state": "running", "first_pass": 4})
    assert est["basis"]["probe_passes"] == [5, 6] and est["left_s"] == approx(3.0 * 6 + 7.51)
    est, why = eta.run_time_left(lambda: old + mine, run, {"state": "running", "first_pass": None})
    assert est is None and "checkpoint" in why              # a resume that has not read its checkpoint: the file is still the old process
    est, _ = eta.run_time_left(lambda: old + mine, run, {"state": "running"})    # no key: a new run, probe from pass 0
    assert est["basis"]["probe_passes"] == [1, 2]


# --- status.json: first_pass --------------------------------------------------------------------

def test_the_run_writer_writes_first_pass_only_for_a_process_that_does_not_start_at_pass_0(tmp_path):
    from topopt.snapshot import RunWriter
    w = RunWriter(tmp_path, quiet=True)
    w.status("running", 3, "solve")
    assert "first_pass" not in json.loads((tmp_path / "status.json").read_text())      # a new run: the file is as it was
    w.first_pass = 7
    w.status("running", 8, "solve")
    assert json.loads((tmp_path / "status.json").read_text())["first_pass"] == 7
    w.first_pass = None                                                                 # a resume that has not read its checkpoint
    w.status("starting", 0, "setup")
    st = json.loads((tmp_path / "status.json").read_text())
    assert "first_pass" in st and st["first_pass"] is None


# --- run_status -----------------------------------------------------------------------------------

def make_run(folder, n_records=4, state="running", planned=10, n_free=100_000, first_pass=None, times=(5.0, 2.0, 4.0) + (9.0,) * 20):
    """A run folder with run.json, status.json and history.jsonl, like a run that is at work."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text(json.dumps({"name": "x", "iterations": planned, "n_free_dofs": n_free}))
    status = {"state": state, "iteration": n_records, "step": "solve", "elapsed_s": 20.0, "message": ""}
    if first_pass is not None:
        status["first_pass"] = first_pass
    (folder / "status.json").write_text(json.dumps(status))
    (folder / "history.jsonl").write_text("".join(json.dumps(rec(i, times[i])) + "\n" for i in range(n_records)))
    return folder


@pytest.fixture
def server(monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server as module
    monkeypatch.setattr(module, "_current", {})
    return module


def test_run_status_has_time_left_and_the_summary_says_it(server, tmp_path):
    st = server.run_status(str(make_run(tmp_path / "r", n_records=3)))
    assert st["time_left_note"] is None
    assert st["time_left"]["left_s"] == approx(31.51) and st["time_left"]["after_pass"] == 2
    assert set(st["time_left"]) == {"left_s", "up_to_s", "after_pass", "basis"}
    assert st["summary"].startswith("iteration 3/10, 00:20, volume 40.0%, about 32 s left (up to 42 s)")


def test_run_status_counts_the_passes_left_at_a_later_pass(server, tmp_path):
    st = server.run_status(str(make_run(tmp_path / "r", n_records=5)))     # passes 0 to 4 are done
    assert st["time_left"]["after_pass"] == 4 and st["time_left"]["left_s"] == approx(3.0 * 6 + 7.51)


@pytest.mark.parametrize("n_records", [0, 1, 2])
def test_run_status_before_pass_2_has_null_and_the_reason(server, tmp_path, n_records):
    st = server.run_status(str(make_run(tmp_path / "r", n_records=n_records)))
    assert st["time_left"] is None and "no estimate before pass 2 is done" in st["time_left_note"]
    assert "left" not in st["summary"]


@pytest.mark.parametrize("state", ["done", "error", "stopped", "postprocess"])
def test_run_status_of_a_run_that_is_not_in_its_passes_has_null(server, tmp_path, state):
    st = server.run_status(str(make_run(tmp_path / "r", n_records=6, state=state)))
    assert st["time_left"] is None and f"state: {state}" in st["time_left_note"] and " left" not in st["summary"]


def test_run_status_of_a_process_that_has_stopped_has_null(server, tmp_path):
    d = make_run(tmp_path / "r", n_records=6)
    server._current.update(run_dir=d, started=time.time() - 100)       # this server started the run, and no process is alive
    st = server.run_status(str(d))
    assert st["time_left"] is None and st["time_left_note"] == "the run process has stopped"
    assert "warning: the process has stopped" in st["summary"] and " left" not in st["summary"]


def test_run_status_of_a_resumed_run_counts_from_the_start_of_this_process(server, tmp_path):
    times = (1.0, 1.0, 1.0, 1.0, 9.0, 2.0, 4.0)        # the earlier process: passes 0 to 3; this process starts at pass 4
    d = make_run(tmp_path / "r", n_records=7, planned=12, first_pass=4, times=times)
    st = server.run_status(str(d))
    assert st["time_left"]["basis"]["probe_passes"] == [5, 6] and st["time_left"]["left_s"] == approx(3.0 * 6 + 7.51)
    make_run(d, n_records=6, planned=12, first_pass=4, times=times)    # this process has made 2 passes only
    assert server.run_status(str(d))["time_left"] is None
    d = make_run(tmp_path / "r2", n_records=7, planned=12, times=times)        # the same file with no first_pass: probe from pass 0
    assert server.run_status(str(d))["time_left"]["basis"]["probe_passes"] == [1, 2]


def test_run_status_of_a_resume_of_a_finished_run_says_that_no_pass_is_left(server, tmp_path):
    d = make_run(tmp_path / "r", n_records=6, planned=5, first_pass=6)           # the resume starts at pass 6 of a run of 5 passes
    st = server.run_status(str(d))
    assert st["time_left"] is None and "no pass is left to run" in st["time_left_note"]


def test_run_status_without_run_json_or_a_cut_last_line_gives_a_reason_or_the_estimate(server, tmp_path):
    d = make_run(tmp_path / "r", n_records=3)
    (d / "run.json").unlink()
    st = server.run_status(str(d))
    assert st["time_left"] is None and "iterations" in st["time_left_note"]
    d = make_run(tmp_path / "r3", n_records=3)
    with open(d / "history.jsonl", "a") as f:
        f.write('{"iteration": 3, "t_ite')
    assert server.run_status(str(d))["time_left"]["after_pass"] == 2


def write_job(folder, kind):
    """The job files of a run folder, as `runjob` writes them. `kind`: `running` (a live lock: this test process), `stale_lock` (the lock of a
    process that has ended), `unfinished` (job.json says running, no lock) or `failed` (the job ended with a failure code, no lock)."""
    from topopt import runjob
    pid = os.getpid()
    if kind == "stale_lock":
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        pid = gone.pid
    ident = runjob.process_identity(pid)
    info = {"format": runjob.FORMAT, "state": "failed" if kind == "failed" else "running", "pid": pid, **ident, "token": "tok",
            "started_at": "2026-10-23T12:00:00", "run_dir": str(folder)}
    if kind == "failed":
        info["exit_code"] = 1
    (folder / runjob.INFO_FILE).write_text(json.dumps(info))
    if kind in ("running", "stale_lock"):
        (folder / runjob.LOCK_FILE).write_text(json.dumps({k: info[k] for k in ("format", "pid", "token", "started_at", "run_dir", *ident)}))


JOB_KINDS = [("running", True), ("stale_lock", False), ("unfinished", False), ("failed", False)]      # (kind, the run process is at work)


@pytest.mark.parametrize("kind,at_work", JOB_KINDS)
def test_run_status_of_a_run_that_this_server_did_not_start_follows_the_job_record(server, tmp_path, kind, at_work):
    """Review M1: a CLI run (or a run of an earlier session) that is killed leaves `running` in status.json. The job record says that it ended."""
    d = make_run(tmp_path / "r", n_records=6)
    write_job(d, kind)
    st = server.run_status(str(d))
    assert st["state"] == "running" and st["process_alive"] is False and st["job"]["state"] == ("running" if at_work else kind)
    if at_work:
        assert st["time_left"]["after_pass"] == 5 and st["time_left_note"] is None and " left" in st["summary"]
    else:
        assert st["time_left"] is None and st["time_left_note"] == "the run process has stopped" and "about" not in st["summary"].split("job:")[0]


def test_run_status_of_a_killed_cli_run_has_no_time_left(server, killed_cli_run):
    """The run of the review: a CLI run, SIGKILL after pass 4, `run_status` from a server that did not start it."""
    _yaml, out = killed_cli_run
    assert json.loads((out / "status.json").read_text())["state"] == "running"           # the kill left the state at `running`
    st = server.run_status(str(out))
    assert st["state"] == "running" and st["process_alive"] is False and st["job"]["state"] == "stale_lock"
    assert st["time_left"] is None and st["time_left_note"] == "the run process has stopped"
    assert " left" not in st["summary"] and "stale lock" in st["summary"]


def test_run_status_other_replies_are_as_they_were(server, tmp_path):
    assert server.run_status(str(tmp_path / "none"))["state"] == "none"          # the early replies have no time_left key
    assert "time_left" not in server.run_status(str(tmp_path / "none"))


def test_the_run_status_text_tells_about_time_left_in_one_short_sentence_after_the_key_text():
    pytest.importorskip("mcp")
    from test_plugin_manifest import KEY_TEXT, mcp_texts
    text = mcp_texts()["run_status"]
    at = text.index("`time_left`")
    assert at > max(text.index(p) + len(p) for p in KEY_TEXT["run_status"])       # after the key text
    sentence = text[at:]
    assert sentence.count(". ") <= 1 and len(sentence) < 260, sentence
    assert "`time_left_note`" in sentence and "earlier" in sentence
    assert "while the run is in its passes" in sentence and "stop rule or request to stop" in sentence        # review n2, m1
    assert len(text) < 4096


# --- the dashboard: /api/status -----------------------------------------------------------------

def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


@pytest.fixture
def served(tmp_path):
    from topopt import dashboard_server as ds
    servers = []

    def start(folder, **kw):
        s = ds.start_in_thread(folder, port=0, **kw)
        servers.append(s)
        return s
    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def test_api_status_has_time_left_for_a_run_in_its_passes(served, tmp_path):
    d = make_run(tmp_path / "r", n_records=3)
    base = served(d).url.rstrip("/")
    st = get_json(base + "/api/status")
    assert st["state"] == "running" and st["iteration"] == 3                     # the file's keys are still there
    assert st["time_left"]["left_s"] == approx(31.51) and st["time_left"]["up_to_s"] == approx(41.9, abs=0.1)
    # the next pass is written: the answer follows (the history is read again when it has changed)
    with open(d / "history.jsonl", "a") as f:
        f.write(json.dumps(rec(3, 9.0)) + "\n")
    assert get_json(base + "/api/status")["time_left"]["after_pass"] == 3


def test_api_status_is_the_file_when_there_is_no_estimate(served, tmp_path):
    for n_records, state in ((2, "running"), (6, "done"), (6, "postprocess"), (6, "error")):
        d = make_run(tmp_path / f"r{n_records}{state}", n_records=n_records, state=state)
        base = served(d).url.rstrip("/")
        with urllib.request.urlopen(base + "/api/status") as r:
            assert r.read() == (d / "status.json").read_bytes()                  # the same bytes as before E1-P2
    base = served(tmp_path / "empty").url.rstrip("/")
    try:
        urllib.request.urlopen(base + "/api/status")
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


@pytest.mark.parametrize("kind,at_work", JOB_KINDS)
def test_api_status_of_a_run_with_an_ended_job_is_the_file(served, tmp_path, kind, at_work):
    """Review M1: the viewer of a killed run gets the bytes of status.json, with no `time_left`."""
    d = make_run(tmp_path / "r", n_records=6)
    write_job(d, kind)
    base = served(d).url.rstrip("/")
    st = get_json(base + "/api/status")
    assert ("time_left" in st) is at_work
    if not at_work:
        with urllib.request.urlopen(base + "/api/status") as r:
            assert r.read() == (d / "status.json").read_bytes()


def test_api_status_of_a_killed_cli_run_has_no_time_left(served, killed_cli_run):
    _yaml, out = killed_cli_run
    st = get_json(served(out).url.rstrip("/") + "/api/status")
    assert st["state"] == "running" and "time_left" not in st


def test_api_status_follows_the_job_record_at_each_poll_not_the_cache(served, tmp_path):
    """The process dies with no new line in history.jsonl: the cached estimate must not be sent again."""
    d = make_run(tmp_path / "r", n_records=6)
    write_job(d, "running")
    base = served(d).url.rstrip("/")
    assert get_json(base + "/api/status")["time_left"]["after_pass"] == 5
    write_job(d, "stale_lock")                                                    # the same history, the job record now says the process ended
    assert "time_left" not in get_json(base + "/api/status")


def test_api_status_in_live_mode_reads_no_history_when_it_has_not_changed(served, tmp_path):
    """Review n3: the cache key of a live run is the size and time of the file, so a poll without a change reads nothing of the history."""
    d = make_run(tmp_path / "r", n_records=3)
    s = served(d)
    base = s.url.rstrip("/")
    assert get_json(base + "/api/status")["time_left"]["after_pass"] == 2
    reads, real = [], s.source.read
    s.source.read = lambda rel: (reads.append(rel), real(rel))[1]
    for _ in range(3):
        assert get_json(base + "/api/status")["time_left"]["after_pass"] == 2
    assert "history.jsonl" not in reads and "run.json" not in reads
    with open(d / "history.jsonl", "a") as f:
        f.write(json.dumps(rec(3, 9.0)) + "\n")
    assert get_json(base + "/api/status")["time_left"]["after_pass"] == 3
    assert reads.count("history.jsonl") == 1
    (d / "run.json").write_text(json.dumps({"name": "x", "iterations": 20, "n_free_dofs": 100_000}))     # a continuation changes `iterations`
    assert get_json(base + "/api/status")["time_left"]["basis"]["passes_left"] == 20 - 3


def test_api_status_of_a_resumed_run_uses_first_pass(served, tmp_path):
    times = (1.0, 1.0, 1.0, 1.0, 9.0, 2.0, 4.0)
    d = make_run(tmp_path / "r", n_records=7, planned=12, first_pass=4, times=times)
    st = get_json(served(d).url.rstrip("/") + "/api/status")
    assert st["time_left"]["basis"]["probe_passes"] == [5, 6]


def test_api_status_in_replay_follows_the_replay_clock(served, tmp_path):
    d = make_run(tmp_path / "r", n_records=8, state="done")       # rec(i) ends at 10 + 3 i seconds of the run
    s = served(d, replay=True)
    now = [0.0]
    s.source.replay_state.now = lambda: now[0]
    base = s.url.rstrip("/")
    urllib.request.urlopen(base + "/api/replay/start").read()     # the clock starts at now[0] = 0
    seen = {}
    for name, at in (("pass_1", 10 + 3 * 1 + 0.5), ("pass_2", 10 + 3 * 2 + 0.5), ("pass_4", 10 + 3 * 4 + 0.5)):
        now[0] = at
        seen[name] = get_json(base + "/api/status")
    assert seen["pass_1"]["state"] == "running" and "time_left" not in seen["pass_1"]
    assert seen["pass_2"]["time_left"]["after_pass"] == 2 and seen["pass_4"]["time_left"]["after_pass"] == 4
    assert seen["pass_2"]["mode"] == "replay"


# --- the dashboard page -----------------------------------------------------------------------------

def test_the_bar_has_a_place_for_the_estimate_next_to_the_timer():
    html = (REPO / "topopt/dashboard/index.html").read_text()
    assert re.search(r'<b id="b-time">[^<]*</b><span id="b-left" class="left"></span>', html)       # in the layout from the start: no `hidden`
    css = (REPO / "topopt/dashboard/css/style.css").read_text()
    rule = css[css.index("#bar .left {"):].split("}")[0]
    assert "display: block" in rule and "min-height: 1.2em" in rule and "visibility" not in rule      # review n6: the line is reserved
    assert '#bar .left::after { content: "about 000 s left"' in css                                    # and the width of a value
    js = (REPO / "topopt/dashboard/js/app.js").read_text()
    update = js[js.index("function updateLeft()"):js.index("// ---------------------------------------------------------------- state\nfunction setPill")]
    assert ".hidden" not in update and ".style" not in update                                           # the page never takes the line away
    poll = js[js.index("async function pollStatus()"):js.index("async function pollHistory()")]
    assert "updateLeft();" in poll                      # on each status poll: the value comes with the status
    assert "time_left" in js and "/api/status" in js


def test_the_page_shows_about_n_s_left_from_the_status_and_hides_it_otherwise():
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node")
    js = (REPO / "topopt/dashboard/js/app.js").read_text()
    fn = js[js.index("function leftText(st)"):js.index("// ---------------------------------------------------------------- state\nfunction setPill")]
    est = {"left_s": 92.6, "up_to_s": 123.2, "after_pass": 2, "basis": {"assumes": "all planned passes run; a stop rule can end the run earlier"}}
    script = (
        "const els = {'b-left': {textContent: 'x', title: 'x', hidden: false}};\n"
        "const $ = id => els[id];\n"
        "const S = {status: null};\n" + fn + "\n"
        "const out = [];\n"
        f"for (const st of [null, {{state: 'running'}}, {{state: 'running', time_left: {json.dumps(est)}}}, "
        f"{{state: 'done', time_left: {json.dumps(est)}}}, {{state: 'running', time_left: {{left_s: null}}}}]) {{\n"
        "  S.status = st; updateLeft();\n"
        "  out.push([els['b-left'].textContent, els['b-left'].title, els['b-left'].hidden]);\n"
        "}\n"
        "console.log(JSON.stringify(out));\n")
    res = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    none, no_est, shown, done, bad = json.loads(res.stdout)
    assert none == ["", "", False] and no_est == ["", "", False] and done == ["", "", False] and bad == ["", "", False]    # empty, never hidden (n6)
    assert shown[0] == "about 93 s left" and shown[2] is False
    assert shown[1].startswith("up to 123 s; all planned passes run")


# --- real runs: run.log and status.json -------------------------------------------------------------------

N = 9      # passes of the small runs (0 to 9)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """The cantilever of the tests (1 mm voxels, 9 passes), run once in a child process with 1 thread."""
    root = tmp_path_factory.mktemp("eta_small")
    yaml_path = make_case(root, h=1.0, iterations=N)
    out = root / "run"
    finish(yaml_path, out)
    return yaml_path, out


@pytest.fixture(scope="module")
def killed_cli_run(tmp_path_factory):
    """A CLI run of the test cantilever that dies by SIGKILL after pass 4: `status.json` stays at `running`, `job.lock` stays (a stale lock)."""
    root = tmp_path_factory.mktemp("eta_killed")
    yaml_path = make_case(root, h=1.0, iterations=N)
    out = root / "run"
    killed(yaml_path, out, "after_checkpoint:4")
    return yaml_path, out


TIME_LINE = re.compile(r"^fact: time: about (\d+) s left after pass (\d+) \(up to (\d+) s; the passes (\d+) and (\d+) took ([\d.]+) s on average, "
                       r"passes left: (\d+), the post step takes about ([\d.]+) s; this holds if no stop rule or request to stop ends the run earlier", re.M)


def time_lines(out):
    return TIME_LINE.findall(log(out))


def test_a_run_writes_one_time_line_after_pass_2_and_it_equals_the_function(small):
    from topopt import eta
    _yaml, out = small
    text = log(out)
    assert text.count("fact: time:") == 1 and len(time_lines(out)) == 1            # one line, not one for each pass
    left, after, up_to, a, b, pass_s, left_passes, post = time_lines(out)[0]
    assert (after, a, b, left_passes) == ("2", "1", "2", str(N - 2))
    run = json.loads((out / "run.json").read_text())
    est, _ = eta.time_left(history(out)[:3], run["iterations"], run["n_free_dofs"])
    assert int(left) == round(est["left_s"]) and int(up_to) == round(est["up_to_s"]) and float(pass_s) == approx(est["basis"]["pass_s"], abs=0.006)
    assert float(post) == est["basis"]["post_s"] == 4.43          # a small problem: the band below 50,000 free dofs
    assert "did not measure" in text                              # the small problem is below the 10,000 free dofs that E1 measured
    # the line comes after the line of pass 2 and before the line of pass 3
    lines = text.splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("fact: time:"))
    assert lines[i - 1].lstrip().startswith("it   2") or any(ln.lstrip().startswith("it   2") for ln in lines[max(0, i - 4):i])
    assert not any(ln.lstrip().startswith("it   3") for ln in lines[:i])


def test_a_run_whose_pass_2_is_its_last_pass_writes_no_time_line(tmp_path):
    """Nothing is left to estimate after the last pass, so there is no line. A run of 3 passes still has pass 3 to go."""
    for n_passes, lines in ((1, 0), (2, 0), (3, 1)):
        folder = tmp_path / f"c{n_passes}"
        folder.mkdir()
        out = tmp_path / f"n{n_passes}"
        finish(make_case(folder, h=2.0, iterations=n_passes), out)
        assert len(time_lines(out)) == lines, (n_passes, log(out))


def test_the_final_status_of_a_new_run_has_no_first_pass_and_no_time_left(small):
    _yaml, out = small
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "done" and "first_pass" not in st and "time_left" not in st


def test_a_resumed_process_probes_its_own_passes_and_writes_its_own_line(tmp_path, monkeypatch):
    from topopt.cli import main
    from topopt.snapshot import RunWriter
    yaml_path = make_case(tmp_path, h=1.0, iterations=N)
    out = tmp_path / "run"
    killed(yaml_path, out, "after_checkpoint:3")           # the process dies after pass 3: the resume starts at pass 4
    assert log(out).count("fact: time:") == 1 and time_lines(out)[0][1] == "2"
    seen = []
    real = RunWriter.status

    def spy(self, state=None, iteration=0, step="setup", message=""):
        real(self, state, iteration, step, message)
        seen.append((self.state, self.first_pass, json.loads((self.dir / "status.json").read_text()).get("first_pass", "absent")))
    monkeypatch.setattr(RunWriter, "status", spy)
    assert main(run_args(yaml_path, out, "--resume")) == 0
    assert seen[0][1] is None and seen[0][2] is None       # the setup of a resume: the checkpoint is not read yet
    assert {s[2] for s in seen[1:] if s[0] == "running"} == {4}
    lines = time_lines(out)
    assert [ln[1] for ln in lines] == ["2", "6"]            # the second process: first pass 4, so pass 6, from its passes 5 and 6
    assert (lines[1][3], lines[1][4], lines[1][6]) == ("5", "6", str(N - 6))
    from topopt import eta
    rows = history(out)
    run = json.loads((out / "run.json").read_text())
    est, _ = eta.time_left(rows[:7], run["iterations"], run["n_free_dofs"], start=4)
    assert int(lines[1][0]) == round(est["left_s"]) and est["basis"]["probe_passes"] == [5, 6]


def test_a_continuation_probes_its_own_passes(tmp_path):
    parent_dir = tmp_path / "p"
    parent_dir.mkdir()
    yaml_path = make_case(parent_dir, h=1.0, iterations=3)
    parent = tmp_path / "parent"
    finish(yaml_path, parent)                              # a finished run of 3 passes: its checkpoint is complete (a replay: pass 3 runs again)
    out = tmp_path / "more"
    finish(yaml_path, out, "--continue-from", str(parent), "--more-iterations", "6")
    run = json.loads((out / "run.json").read_text())
    assert run["iterations"] == 9 and run["continued_from"]["iteration"] == 3 and run["continued_from"]["replay"] is True
    lines = time_lines(out)
    assert len(lines) == 1
    # first pass 3 (the replay of the last pass of the parent): the line is at pass 5, from the passes 4 and 5 of this process
    assert (lines[0][1], lines[0][3], lines[0][4], lines[0][6]) == ("5", "4", "5", "4")


# --- the texts: the request to stop, the source, the section number (review m1 to m5) ----------------------------

def test_the_readme_says_what_E1_measured_and_names_a_request_to_stop():
    text = (REPO / "docs" / "reference" / "dashboard-mcp-render.md").read_text()
    at = text.index("about N s left")
    para = text[at:at + 1200]
    assert "no stop rule or request to stop ends the run earlier" in para                       # m1
    assert "In experiment E1 (run-time study; 585 runs of 10,000 free dofs and more on one shared Linux host; no Mac run)" in para   # m3, n7
    assert "section 24.5" in para


def test_the_source_text_names_the_experiment_by_its_date_and_not_one_code_revision(eta):
    source = Path(eta.__file__).read_text()
    assert "run-time study, 2026-10-23" in source and "the code revisions before main bb6e193" in source
    assert "79fe842" not in source      # Review n1: a local research repository; a reader cannot find it
    assert "code revision bb6e193" not in source and "code revision bb6e193" not in (REPO / "CONTRACTS.md").read_text()      # m2
    assert eta.MEASURED.startswith("experiment E1 (run-time study)")                                                          # n7


def test_the_section_of_contracts_is_24_5_and_every_reference_says_so(eta):
    """Review m5: P9-6 has section 24.4."""
    contracts = (REPO / "CONTRACTS.md").read_text()
    assert "### 24.5 The live time estimate after pass 2 (E1-P2)" in contracts and "### 24.4 The live time estimate" not in contracts
    assert "E1-P2, section 24.4" not in contracts and "E1-P2, section 24.5" in contracts
    assert "section 24.5" in eta.__doc__ and "24.4" not in eta.__doc__ and "24.4" not in Path(__file__).read_text().split("import json")[0]
    section = contracts[contracts.index("### 24.5 The live time estimate"):]
    section = section[:section.index("\n## 25.")]
    for needle in ("pardiso", "gpu-mg", "amg", "thin samples", "request to stop", "POST medians pooled over the same rows",
                   "opt-in post check", "replay"):
        assert needle in section, needle                                                                                        # m1, m4, n4, n5, n8
