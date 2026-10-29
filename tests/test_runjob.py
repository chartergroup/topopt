"""Tests for the run helper topopt/runjob.py (CONTRACTS.md section 24.1).

Time, peak memory, the exit code, the done and FAILED markers, the job lock (live holder, dead pid,
reused pid, override), the process start time on both kinds of platform (faked), and the run
facts that run_status shows. The CLI and MCP paths are at the end of the file.
"""
import errno
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from topopt import runjob  # noqa: E402
from topopt.config import UserError  # noqa: E402
from topopt.runjob import (DONE_FILE, FAILED_FILE, GUARD_FILE, INFO_FILE, LOCK_FILE, JobHeld, RunJob,  # noqa: E402
                           exit_code_of, holder_state, job_facts, live_holder, run_job)

needs_proc = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc (Linux)")


def info(d):
    return json.loads((d / INFO_FILE).read_text())


def lock(d):
    return json.loads((d / LOCK_FILE).read_text())


@pytest.fixture
def sleeper():
    """A live process that is not this one. Yields the Popen. The test ends it."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield p
    finally:
        p.kill()
        p.wait()


@pytest.fixture
def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def put_lock(d, pid, process_start="now", token="other", started_at="2026-10-20T01:02:03+00:00"):
    """Write a lock record of another process, and a matching job.json that says running.

    The default is the record that a real run writes (on Linux with the start ticks and the boot id). A number for
    `process_start` gives an older kind of record with an epoch start time only (macOS; the Linux fallback).
    """
    d.mkdir(parents=True, exist_ok=True)
    ident = runjob.process_identity(pid) if process_start == "now" else {"process_start": process_start}
    rec = {"format": 1, "pid": pid, **ident, "token": token, "started_at": started_at, "run_dir": str(d)}
    (d / LOCK_FILE).write_text(json.dumps(rec))
    (d / INFO_FILE).write_text(json.dumps(dict(rec, state="running", success_codes=[0], notes=[])))
    return rec


# --- done and FAILED markers, time, peak memory ------------------------------------------------

def test_success_writes_done_marker_and_record(tmp_path):
    d = tmp_path / "run"
    assert run_job(d, lambda job: 0) == 0
    assert (d / DONE_FILE).read_text() == "exit 0\n" and not (d / FAILED_FILE).exists()
    assert not (d / LOCK_FILE).exists() and not list(d.glob("*.tmp"))
    rec = info(d)
    assert rec["state"] == "done" and rec["exit_code"] == 0 and rec["success_codes"] == [0]
    assert rec["wall_s"] >= 0 and rec["peak_rss_bytes"] > 10e6 and "children_peak_rss_bytes" not in rec
    assert rec["pid"] == os.getpid() and rec["started_at"] and rec["ended_at"] and "token" in rec
    assert 10e6 < rec["peak_rss_start_bytes"]
    assert rec["peak_rss_reset"] or rec["peak_rss_start_bytes"] <= rec["peak_rss_bytes"]   # after a reset the job's own peak can be below the start value (DL1a)
    assert (d / GUARD_FILE).exists()


def test_the_peak_is_not_less_than_the_start_value(tmp_path, monkeypatch):
    # VmHWM of /proc/self/status is not monotonic: a later read can be lower than the start read.
    # This is the path without the reset of the peak (macOS, or a /proc that refuses the write): the start value is the floor.
    reads = iter([(300_000_000, "VmHWM"), (299_000_000, "VmHWM")])
    monkeypatch.setattr(runjob, "peak_rss_info", lambda: next(reads))
    monkeypatch.setattr(runjob.hostinfo, "reset_peak_rss", lambda notes=None: False)
    run_job(tmp_path, lambda job: 0)
    rec = info(tmp_path)
    assert rec["peak_rss_start_bytes"] == 300_000_000 and rec["peak_rss_bytes"] == 300_000_000


def test_wall_time_is_measured_from_t_start(tmp_path):
    t0 = time.time() - 5.0
    run_job(tmp_path, lambda job: 0, t_start=t0)
    assert 5.0 <= info(tmp_path)["wall_s"] < 15.0


def test_non_success_code_writes_failed_marker_with_the_exit_code(tmp_path):
    assert run_job(tmp_path, lambda job: 3) == 3
    assert (tmp_path / FAILED_FILE).read_text() == "exit 3\n" and not (tmp_path / DONE_FILE).exists()
    assert info(tmp_path)["state"] == "failed" and info(tmp_path)["exit_code"] == 3
    assert not (tmp_path / LOCK_FILE).exists()


def test_success_codes_come_from_the_caller(tmp_path):
    run_job(tmp_path, lambda job: 3, success_codes={0, 3})
    assert (tmp_path / DONE_FILE).read_text() == "exit 3\n" and not (tmp_path / FAILED_FILE).exists()
    assert info(tmp_path)["success_codes"] == [0, 3] and info(tmp_path)["state"] == "done"
    run_job(tmp_path, lambda job: 1, success_codes={0, 3})
    assert (tmp_path / FAILED_FILE).exists() and not (tmp_path / DONE_FILE).exists()


@pytest.mark.parametrize("exc, code", [
    (UserError("bad"), 1), (KeyboardInterrupt(), 130), (RuntimeError("boom"), 2),
    (SystemExit(7), 7), (SystemExit(None), 0), (SystemExit("text"), 1), (MemoryError(), 2),
])
def test_exception_gives_the_exit_code_of_main_and_is_raised_again(tmp_path, exc, code):
    def body(job):
        raise exc

    with pytest.raises(type(exc)):
        run_job(tmp_path, body)
    assert exit_code_of(exc) == code
    assert info(tmp_path)["exit_code"] == code
    assert (tmp_path / (DONE_FILE if code == 0 else FAILED_FILE)).exists()
    assert not (tmp_path / LOCK_FILE).exists()


def test_exit_codes_match_cli_main(tmp_path, monkeypatch, capsys):
    """The helper maps an exception as cli.main does. main() and the marker must give one code."""
    from topopt import cli
    for exc in (UserError("x"), KeyboardInterrupt(), ValueError("y")):
        def boom(args, exc=exc):
            raise exc
        monkeypatch.setattr(cli, "cmd_dashboard", boom)
        assert cli.main(["dashboard", str(tmp_path)]) == exit_code_of(exc)
    capsys.readouterr()


def test_none_return_counts_as_exit_0(tmp_path):
    assert run_job(tmp_path, lambda job: None) == 0 and (tmp_path / DONE_FILE).exists()


def test_old_markers_and_record_go_at_the_next_start(tmp_path):
    run_job(tmp_path, lambda job: 1)
    assert (tmp_path / FAILED_FILE).exists()
    seen = {}

    def body(job):
        seen["failed"] = (tmp_path / FAILED_FILE).exists()
        seen["done"] = (tmp_path / DONE_FILE).exists()
        seen["state"] = info(tmp_path)["state"]
        return 0

    run_job(tmp_path, body)
    assert seen == {"failed": False, "done": False, "state": "running"}
    assert (tmp_path / DONE_FILE).exists() and not (tmp_path / FAILED_FILE).exists()


def test_early_finish_then_a_long_wait_keeps_the_first_record(tmp_path):
    def body(job):
        job.finish(0)                 # the work is done; the process only waits after this
        assert (tmp_path / DONE_FILE).exists() and not (tmp_path / LOCK_FILE).exists()
        raise KeyboardInterrupt        # a later exit code does not change the record

    with pytest.raises(KeyboardInterrupt):
        run_job(tmp_path, body)
    assert (tmp_path / DONE_FILE).exists() and not (tmp_path / FAILED_FILE).exists()
    assert info(tmp_path)["exit_code"] == 0


def test_finish_does_not_hide_a_run_when_the_record_cannot_be_written(tmp_path, capsys):
    d = tmp_path / "run"
    job = RunJob(d)
    job.start()
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    job.finish(0)                    # the folder is gone: one line on stderr, no exception
    assert "job record not written" in capsys.readouterr().err


# --- the lock ---------------------------------------------------------------------------------

def test_start_writes_a_lock_with_pid_and_start_time(tmp_path):
    job = RunJob(tmp_path)
    job.start()
    rec = lock(tmp_path)
    assert rec["pid"] == os.getpid() and rec["token"] == job.token and rec["run_dir"] == str(tmp_path)
    assert rec["process_start"] == runjob.process_start_time(os.getpid()) and rec["started_at"]
    if Path("/proc/self/stat").exists():
        assert rec["process_start_ticks"] == runjob._read_proc_stat(os.getpid())[1] and rec["boot_id"] == runjob._boot_id()
    assert info(tmp_path)["state"] == "running" and info(tmp_path)["notes"] == []
    assert info(tmp_path)["peak_rss_source"] in ("VmHWM", "ru_maxrss") and info(tmp_path)["peak_rss_start_bytes"] > 0
    job.finish(0)
    assert not (tmp_path / LOCK_FILE).exists()


def test_live_holder_stops_the_second_start_with_facts(tmp_path, sleeper):
    rec = put_lock(tmp_path, sleeper.pid)
    (tmp_path / "keep.txt").write_text("x")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(JobHeld) as e:
        run_job(tmp_path, lambda job: pytest.fail("the body must not run"))
    msg = str(e.value)
    assert str(sleeper.pid) in msg and rec["started_at"] in msg and str(tmp_path) in msg
    assert "--override-lock" in msg and "override_lock" in msg
    assert isinstance(e.value, UserError)                    # the CLI shows one line and exits 1
    assert e.value.holder["pid"] == sleeper.pid and e.value.holder["run_dir"] == str(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.name != GUARD_FILE} == \
        {k: v for k, v in before.items() if k != GUARD_FILE}          # the folder is untouched
    facts = live_holder(tmp_path)
    assert facts["pid"] == sleeper.pid and facts["started_at"] == rec["started_at"] and facts["message"] == msg


def test_override_takes_the_lock_and_says_so(tmp_path, sleeper):
    put_lock(tmp_path, sleeper.pid)
    seen = {}

    def body(job):
        seen["facts"] = list(job.facts)
        seen["holder"] = lock(tmp_path)["pid"]
        return 0

    run_job(tmp_path, body, override=True)
    assert seen["holder"] == os.getpid()
    assert len(seen["facts"]) == 1 and "override" in seen["facts"][0] and str(sleeper.pid) in seen["facts"][0]
    assert info(tmp_path)["notes"] == seen["facts"]
    assert (tmp_path / DONE_FILE).exists() and sleeper.poll() is None     # the other process is not touched


def test_dead_pid_is_reported_and_replaced(tmp_path, dead_pid):
    put_lock(tmp_path, dead_pid, process_start=12345.0)
    job = RunJob(tmp_path)
    job.start()
    assert len(job.facts) == 1 and "is not running" in job.facts[0] and str(dead_pid) in job.facts[0]
    assert "replaced" in job.facts[0] and lock(tmp_path)["pid"] == os.getpid()
    job.finish(0)


def test_reused_pid_is_reported_and_replaced(tmp_path, sleeper):
    """The pid is alive, but its start time is not the recorded one: another process has the pid now."""
    put_lock(tmp_path, sleeper.pid, process_start=(runjob.process_start_time(sleeper.pid) or time.time()) - 3600)
    assert holder_state(json.loads((tmp_path / LOCK_FILE).read_text())) == "reused"
    assert live_holder(tmp_path) is None
    job = RunJob(tmp_path)
    job.start()
    assert len(job.facts) == 1 and "now another process" in job.facts[0] and str(sleeper.pid) in job.facts[0]
    assert lock(tmp_path)["pid"] == os.getpid()
    job.finish(0)


def test_holder_states(sleeper, dead_pid):
    start = runjob.process_start_time(sleeper.pid)
    assert holder_state({"pid": sleeper.pid, "process_start": start}) == "live"
    assert holder_state({"pid": sleeper.pid, "process_start": start + 0.5}) == "live"     # inside the tolerance
    assert holder_state({"pid": sleeper.pid, "process_start": start + 10}) == "reused"
    assert holder_state({"pid": sleeper.pid}) == "live"                                  # no recorded start: trust the pid
    assert holder_state({"pid": dead_pid, "process_start": 1.0}) == "dead"
    for bad in (None, 0, -4, "12", 1.5):
        assert holder_state({"pid": bad}) == "dead"
    assert holder_state({}) == "dead"


@needs_proc
@pytest.mark.parametrize("step", [0.5, 2.0, -5.0, 3600.0])
def test_a_wall_clock_step_does_not_make_a_live_holder_reused(tmp_path, sleeper, monkeypatch, step):
    """`btime` of /proc/stat moves when the wall clock is stepped (NTP, `date -s`, VM time sync). The start ticks do not."""
    rec = put_lock(tmp_path, sleeper.pid)
    assert isinstance(rec["process_start_ticks"], int) and rec["boot_id"]
    real = runjob._boot_time
    monkeypatch.setattr(runjob, "_boot_time", lambda: real() + step)
    assert holder_state(rec) == "live"
    assert live_holder(tmp_path)["pid"] == sleeper.pid
    with pytest.raises(JobHeld):
        RunJob(tmp_path).start()


@needs_proc
def test_an_old_record_with_an_epoch_start_only_still_follows_the_wall_clock(sleeper, monkeypatch):
    """No ticks in the record (macOS, or a record from an older version): the 1 s compare of epoch times is the rule."""
    rec = {"pid": sleeper.pid, "process_start": runjob.process_start_time(sleeper.pid)}
    assert holder_state(rec) == "live"
    real = runjob._boot_time
    monkeypatch.setattr(runjob, "_boot_time", lambda: real() + 2.0)
    assert holder_state(rec) == "reused"


@needs_proc
def test_other_ticks_or_another_boot_mean_a_reused_pid(sleeper):
    rec = dict(runjob.process_identity(sleeper.pid), pid=sleeper.pid)
    assert holder_state(rec) == "live"
    assert holder_state(dict(rec, process_start_ticks=rec["process_start_ticks"] + 1)) == "reused"
    assert holder_state(dict(rec, process_start_ticks=rec["process_start_ticks"] - 1)) == "reused"
    assert holder_state(dict(rec, boot_id="the-boot-before")) == "reused"          # the machine started again since
    assert holder_state(dict(rec, process_start_ticks="12")) == "live"             # not an int: the epoch compare decides


def test_the_identity_has_no_ticks_without_proc(monkeypatch):
    """A faked macOS: the start time comes from `ps`, and the record has no ticks and no boot id."""
    monkeypatch.setattr(runjob, "_has_procfs", lambda: False)
    monkeypatch.setattr(runjob, "_start_from_ps", lambda pid: 5000.0)
    assert runjob.process_identity(os.getpid()) == {"process_start": 5000.0}


def test_unknown_start_time_trusts_a_live_pid(sleeper, monkeypatch):
    monkeypatch.setattr(runjob, "process_start_time", lambda pid: None)
    assert holder_state({"pid": sleeper.pid, "process_start": 1.0}) == "live"


def test_identity_state_has_four_answers(sleeper, dead_pid, monkeypatch):
    """P9-14: `identity_state` says `unconfirmed` where it can compare nothing. It never says `live` without a compare."""
    ident = runjob.process_identity(sleeper.pid)
    assert runjob.identity_state({"pid": sleeper.pid, **ident}) == "live"
    assert runjob.identity_state({"pid": dead_pid, **runjob.process_identity(os.getpid())}) == "dead"
    assert runjob.identity_state({"pid": sleeper.pid, "process_start": ident["process_start"] + 10}) == "reused"
    # the pid exists, and the record has no start time to compare
    assert runjob.identity_state({"pid": sleeper.pid}) == "unconfirmed"
    assert runjob.identity_state({"pid": sleeper.pid, "process_start": None}) == "unconfirmed"
    # the pid exists, the record has a start time, and the system gives none
    monkeypatch.setattr(runjob, "process_start_time", lambda pid: None)
    assert runjob.identity_state({"pid": sleeper.pid, "process_start": 1.0}) == "unconfirmed"
    for bad in (None, 0, -4, "12", 1.5):
        assert runjob.identity_state({"pid": bad}) == "dead"


@needs_proc
def test_identity_state_with_ticks_and_no_proc_stat_is_unconfirmed(sleeper, monkeypatch):
    """The record has ticks and a boot id, but /proc/<pid>/stat cannot be read now (a mount with hidepid, for example)."""
    rec = {"pid": sleeper.pid, **runjob.process_identity(sleeper.pid)}
    monkeypatch.setattr(runjob, "_read_proc_stat", lambda pid: None)
    monkeypatch.setattr(runjob, "process_start_time", lambda pid: None)
    monkeypatch.setattr(runjob, "pid_alive", lambda pid: True)
    assert runjob.identity_state(rec) == "unconfirmed"


def _holder_state_before_p9_14(rec: dict) -> str:
    """`holder_state` as it was before `identity_state` (revision 636c3d7), copied here as the reference."""
    pid = rec.get("pid")
    if not runjob.pid_alive(pid):
        return "dead"
    ticks, boot_id = rec.get("process_start_ticks"), rec.get("boot_id")
    if isinstance(ticks, int) and isinstance(boot_id, str) and runjob._has_procfs():
        now_boot = runjob._boot_id()
        if now_boot is not None and now_boot != boot_id:
            return "reused"
        stat = runjob._read_proc_stat(pid)
        if stat is not None:
            return "live" if stat[1] == ticks else "reused"
    recorded = rec.get("process_start")
    now = runjob.process_start_time(pid)
    if isinstance(recorded, (int, float)) and now is not None and abs(now - recorded) > runjob.START_TOLERANCE_S:
        return "reused"
    return "live"


def test_holder_state_gives_the_same_answer_as_before(sleeper, dead_pid, monkeypatch):
    """P9-14: `holder_state` is a wrapper of `identity_state` (`unconfirmed` counts as `live`). Live, dead, reused and unconfirmed
    records give the same answers as the function of revision 636c3d7."""
    ident = runjob.process_identity(sleeper.pid)
    start = ident["process_start"]
    records = {
        "live": {"pid": sleeper.pid, **ident},
        "live, epoch start only": {"pid": sleeper.pid, "process_start": start},
        "live, inside the tolerance": {"pid": sleeper.pid, "process_start": start + 0.5},
        "dead": {"pid": dead_pid, **ident},
        "dead, no pid": {},
        "reused, epoch start": {"pid": sleeper.pid, "process_start": start + 10},
        "unconfirmed, no start in the record": {"pid": sleeper.pid},
        "unconfirmed, start None": {"pid": sleeper.pid, "process_start": None},
        "ticks that are not an int": {"pid": sleeper.pid, **dict(ident, process_start_ticks="12")},
    }
    if "process_start_ticks" in ident:
        records["reused, other ticks"] = {"pid": sleeper.pid, **dict(ident, process_start_ticks=ident["process_start_ticks"] + 1)}
        records["reused, other boot"] = {"pid": sleeper.pid, **dict(ident, boot_id="the-boot-before")}
    expected_now = {"live": "live", "live, epoch start only": "live", "live, inside the tolerance": "live", "dead": "dead",
                    "dead, no pid": "dead", "reused, epoch start": "reused", "unconfirmed, no start in the record": "live",
                    "unconfirmed, start None": "live"}
    for name, rec in records.items():
        assert holder_state(rec) == _holder_state_before_p9_14(rec), name
        if name in expected_now:
            assert holder_state(rec) == expected_now[name], name
    # the system gives no start time: the old answer was `live` for a start time that cannot be compared, and it still is
    monkeypatch.setattr(runjob, "process_start_time", lambda pid: None)
    for name, rec in records.items():
        assert holder_state(rec) == _holder_state_before_p9_14(rec), f"{name} (no start time from the system)"
    assert holder_state({"pid": sleeper.pid, "process_start": 1.0}) == "live"
    assert runjob.identity_state({"pid": sleeper.pid, "process_start": 1.0}) == "unconfirmed"


def test_a_lock_of_the_finished_holder_does_not_stop_the_next_run(tmp_path):
    run_job(tmp_path, lambda job: 0)
    assert live_holder(tmp_path) is None
    run_job(tmp_path, lambda job: 0)


def test_a_taken_over_run_writes_nothing_when_it_ends(tmp_path, capsys):
    """Run A holds the lock. B overrides. A ends later: it must not remove B's lock or write markers over B."""
    a = RunJob(tmp_path)
    a.start()
    b = RunJob(tmp_path, override=True)
    b.start()
    assert lock(tmp_path)["token"] == b.token
    a.finish(0)
    assert "took the lock over" in a.facts[-1] and "took the lock over" in capsys.readouterr().err
    assert (tmp_path / LOCK_FILE).exists() and lock(tmp_path)["token"] == b.token
    assert not (tmp_path / DONE_FILE).exists() and info(tmp_path)["state"] == "running"
    b.finish(0)
    assert (tmp_path / DONE_FILE).exists() and not (tmp_path / LOCK_FILE).exists()


def test_a_taken_over_run_that_ends_after_the_other_run_writes_nothing(tmp_path, capsys):
    """B overrides A and ends first (the lock is gone). A ends later: it must not write over the record of B."""
    a = RunJob(tmp_path)
    a.start()
    b = RunJob(tmp_path, override=True)
    b.start()
    b.finish(0)
    assert not (tmp_path / LOCK_FILE).exists() and (tmp_path / DONE_FILE).exists()
    a.finish(2)
    assert "took the lock over" in capsys.readouterr().err
    assert (tmp_path / DONE_FILE).exists() and not (tmp_path / FAILED_FILE).exists()
    assert info(tmp_path)["token"] == b.token and info(tmp_path)["exit_code"] == 0


def test_a_lock_that_somebody_removed_does_not_stop_the_record(tmp_path):
    job = RunJob(tmp_path)
    job.start()
    (tmp_path / LOCK_FILE).unlink()          # a person removes the lock to free the folder
    job.finish(0)
    assert (tmp_path / DONE_FILE).exists() and info(tmp_path)["state"] == "done"


def test_many_starts_at_once_give_one_winner(tmp_path):
    n = 16
    barrier = threading.Barrier(n)
    results = []

    def go():
        job = RunJob(tmp_path)
        barrier.wait()
        try:
            job.start()
            results.append("won")
        except JobHeld:
            results.append("held")

    threads = [threading.Thread(target=go) for _ in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("won") == 1 and results.count("held") == n - 1


def test_many_starts_at_once_on_a_stale_lock_give_one_winner(tmp_path, dead_pid):
    put_lock(tmp_path, dead_pid, process_start=1.0)
    n = 16
    barrier = threading.Barrier(n)
    results = []

    def go():
        job = RunJob(tmp_path)
        barrier.wait()
        try:
            job.start()
            results.append("won")
        except JobHeld:
            results.append("held")

    threads = [threading.Thread(target=go) for _ in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("won") == 1 and results.count("held") == n - 1


@pytest.mark.parametrize("code", [errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL])
def test_a_file_system_without_flock_goes_on_with_a_fact(tmp_path, monkeypatch, code):
    def no_flock(fd, op):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(runjob.fcntl, "flock", no_flock)
    job = RunJob(tmp_path)
    job.start()
    assert job.facts == [runjob.NO_FLOCK_FACT] and info(tmp_path)["notes"] == [runjob.NO_FLOCK_FACT]
    assert lock(tmp_path)["pid"] == os.getpid()
    job.finish(0)
    assert (tmp_path / DONE_FILE).exists() and not (tmp_path / LOCK_FILE).exists()


def test_without_flock_a_live_holder_still_stops_the_run_and_the_override_works(tmp_path, sleeper, monkeypatch):
    def no_flock(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(runjob.fcntl, "flock", no_flock)
    put_lock(tmp_path, sleeper.pid)
    with pytest.raises(JobHeld):                       # the record check still works
        RunJob(tmp_path).start()
    job = RunJob(tmp_path, override=True)
    job.start()
    assert runjob.NO_FLOCK_FACT in job.facts and any("override" in f for f in job.facts)
    job.finish(0)
    assert (tmp_path / DONE_FILE).exists() and sleeper.poll() is None


def test_another_flock_error_is_not_hidden(tmp_path, monkeypatch):
    def broken(fd, op):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(runjob.fcntl, "flock", broken)
    with pytest.raises(OSError, match="Input/output"):
        RunJob(tmp_path).start()


def test_two_processes_start_on_one_folder(tmp_path):
    """Two real processes: the second one gets the fact and exits 1 without a change."""
    code = ("import sys, time\nfrom topopt.runjob import run_job\n"
            "def body(job):\n    print('ready', flush=True)\n    time.sleep(30)\n    return 0\n"
            "run_job(sys.argv[1], body)\n")
    first = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, text=True, cwd=ROOT)
    try:
        assert first.stdout.readline().strip() == "ready"
        r = subprocess.run([sys.executable, "-c", "import sys\nfrom topopt.runjob import run_job\n"
                            "run_job(sys.argv[1], lambda job: 0)", str(tmp_path)],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0 and "JobHeld" in r.stderr and str(first.pid) in r.stderr
        assert lock(tmp_path)["pid"] == first.pid
        facts = job_facts(tmp_path)
        assert facts["state"] == "running" and facts["pid"] == first.pid
    finally:
        first.kill()
        first.wait()
    facts = job_facts(tmp_path)                       # killed: no marker, a lock of a dead process
    assert facts["state"] == "stale_lock" and facts["holder"] == "dead" and facts["pid"] == first.pid
    assert "no done or FAILED marker" in facts["summary"]


# --- process start time and liveness on both kinds of platform ----------------------------------

PROC_STAT = "4242 (my (odd) name) S 1 4242 4242 0 -1 4194560 100 0 0 0 1 2 0 0 20 0 1 0 987654 1000 100 18446744073709551615"


def test_parse_proc_stat_with_brackets_in_the_name():
    assert runjob._parse_proc_stat(PROC_STAT) == ("S", 987654)
    assert runjob._parse_proc_stat("garbage") is None
    assert runjob._parse_proc_stat("1 (x) S 1") is None


def test_parse_ps_lstart():
    a = runjob._parse_lstart("Tue Oct 20 08:09:36 2026\n")
    b = runjob._parse_lstart("Tue Oct  6 08:09:36 2026")          # macOS pads a one-digit day with a space
    assert a is not None and b is not None and a - b == 14 * 86400
    assert runjob._parse_lstart("") is None and runjob._parse_lstart("not a date") is None


def test_process_start_time_picks_proc_or_ps(monkeypatch):
    monkeypatch.setattr(runjob, "_start_from_proc", lambda pid: 111.0)
    monkeypatch.setattr(runjob, "_start_from_ps", lambda pid: 222.0)
    monkeypatch.setattr(runjob, "_has_procfs", lambda: True)          # Linux
    assert runjob.process_start_time(5) == 111.0
    monkeypatch.setattr(runjob, "_has_procfs", lambda: False)         # macOS
    assert runjob.process_start_time(5) == 222.0


def test_ps_branch_gives_a_lock_that_works(tmp_path, monkeypatch, sleeper):
    """A faked macOS: no /proc, the start time comes from ps. The live holder and the reused pid are found."""
    monkeypatch.setattr(runjob, "_has_procfs", lambda: False)
    monkeypatch.setattr(runjob, "_start_from_ps", lambda pid: 5000.0 if pid == sleeper.pid else None)
    put_lock(tmp_path, sleeper.pid, process_start=5000.0)
    with pytest.raises(JobHeld):
        RunJob(tmp_path).start()
    put_lock(tmp_path, sleeper.pid, process_start=1000.0)
    job = RunJob(tmp_path)
    job.start()
    assert "now another process" in job.facts[0]
    job.finish(0)


def test_ps_branch_with_no_ps_gives_none(monkeypatch):
    monkeypatch.setattr(runjob, "_has_procfs", lambda: False)

    def no_ps(*a, **k):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(runjob.subprocess, "run", no_ps)
    assert runjob.process_start_time(os.getpid()) is None


def test_ps_text_from_a_real_ps_if_there_is_one():
    if shutil.which("ps") is None:
        pytest.skip("no ps on this host")
    t = runjob._start_from_ps(os.getpid())
    assert t is not None and abs(t - time.time()) < 365 * 86400 and t <= time.time() + 2
    if runjob._has_procfs():            # Linux with procps: the two sources name one start time (ps has 1 s resolution)
        assert abs(t - runjob._start_from_proc(os.getpid())) <= 2


@needs_proc
def test_process_start_time_from_proc_is_stable_and_plausible():
    a, b = runjob.process_start_time(os.getpid()), runjob.process_start_time(os.getpid())
    assert a == b and 0 < a <= time.time() + 1
    assert abs(runjob._start_from_proc(os.getpid()) - a) < 1e-6


@needs_proc
def test_a_zombie_counts_as_not_alive():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 20
    while time.time() < deadline:                     # the child ends; nobody waits for it: a zombie
        stat = runjob._read_proc_stat(p.pid)
        if stat is not None and stat[0] == "Z":
            break
        time.sleep(0.05)
    try:
        assert runjob.pid_alive(p.pid) is False
    finally:
        p.wait()


def test_pid_alive_basic(sleeper, dead_pid):
    assert runjob.pid_alive(os.getpid()) and runjob.pid_alive(sleeper.pid)
    assert not runjob.pid_alive(dead_pid) and not runjob.pid_alive(0) and not runjob.pid_alive(None)


# --- the facts for run_status -----------------------------------------------------------------

def test_job_facts_none_without_job_files(tmp_path):
    assert job_facts(tmp_path) is None


def test_job_facts_states(tmp_path, sleeper, dead_pid):
    d = tmp_path / "done"
    run_job(d, lambda job: 0)
    f = job_facts(d)
    assert f["state"] == "done" and f["exit_code"] == 0 and f["marker"] == DONE_FILE and "token" not in f
    assert f["summary"].startswith("job: done (exit 0,") and "peak memory" in f["summary"]

    d = tmp_path / "failed"
    run_job(d, lambda job: 3)
    f = job_facts(d)
    assert f["state"] == "failed" and f["exit_code"] == 3 and f["marker"] == FAILED_FILE
    assert "FAILED with exit code 3" in f["summary"]

    d = tmp_path / "running"
    put_lock(d, sleeper.pid)
    f = job_facts(d)
    assert f["state"] == "running" and f["pid"] == sleeper.pid and f["holder"] == "live"
    assert f["summary"].startswith(f"job: running (pid {sleeper.pid}")

    d = tmp_path / "stale"
    put_lock(d, dead_pid, process_start=1.0)
    f = job_facts(d)
    assert f["state"] == "stale_lock" and f["holder"] == "dead" and "is not running" in f["summary"]

    d = tmp_path / "reused"
    put_lock(d, sleeper.pid, process_start=1.0)
    f = job_facts(d)
    assert f["state"] == "stale_lock" and f["holder"] == "reused" and "another process now" in f["summary"]

    d = tmp_path / "unfinished"
    put_lock(d, dead_pid)
    (d / LOCK_FILE).unlink()
    assert job_facts(d)["state"] == "unfinished"

    d = tmp_path / "stalemarker"                   # a new job died after it took the lock; an old marker is left
    run_job(d, lambda job: 0)
    (d / LOCK_FILE).write_text(json.dumps({"pid": dead_pid, "process_start": 1.0, "token": "new job"}))
    f = job_facts(d)
    assert f["state"] == "stale_lock" and f["marker"] == DONE_FILE and "job.DONE is from an earlier job" in f["summary"]

    d = tmp_path / "lockleft"                      # killed after the record, before the lock was removed
    run_job(d, lambda job: 0)
    rec = info(d)
    (d / LOCK_FILE).write_text(json.dumps({"pid": dead_pid, "process_start": 1.0, "token": rec["token"]}))
    f = job_facts(d)
    assert f["state"] == "done" and any("lock file of the finished job" in n for n in f["notes"])


def test_job_facts_survive_a_broken_file(tmp_path):
    (tmp_path / INFO_FILE).write_text("{not json")
    (tmp_path / LOCK_FILE).write_text("[1, 2]")
    assert job_facts(tmp_path) is None


def test_run_status_reports_the_helper_state(tmp_path, sleeper, dead_pid):
    pytest.importorskip("mcp")
    import fake_run
    from topopt_mcp import server
    out = tmp_path / "r"
    fake_run.run(out, spi=0.0, n_it=4)
    run_job(out, lambda job: 0)
    st = server.run_status(str(out))
    assert st["job"]["state"] == "done" and st["job"]["exit_code"] == 0
    assert "job: done (exit 0" in st["summary"] and st["summary"].startswith("done: iteration")

    run_job(out, lambda job: 3)
    st = server.run_status(str(out))
    assert st["job"]["state"] == "failed" and "FAILED with exit code 3" in st["summary"]

    put_lock(out, dead_pid, process_start=1.0)
    st = server.run_status(str(out))
    assert st["job"]["state"] == "stale_lock" and "stale lock" in st["summary"]
    assert str(dead_pid) in st["summary"]

    put_lock(out, sleeper.pid)
    st = server.run_status(str(out))
    assert st["job"]["state"] == "running" and f"pid {sleeper.pid}" in st["summary"]


def test_a_job_record_older_than_the_new_run_is_marked(tmp_path, sleeper):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    out = tmp_path / "r"
    run_job(out, lambda job: 0)
    same = server._job_facts(out, started=time.time() - 100)          # the record is newer than the start: this run's
    assert "earlier_run" not in same and same["summary"].startswith("job: done")
    old = server._job_facts(out, started=time.time() + 5)             # the new run has not written yet
    assert old["earlier_run"] is True and old["summary"].startswith("job (an earlier run; the new run has not written yet): done")
    assert "earlier_run" not in server._job_facts(out)                 # a run that this server did not start
    put_lock(tmp_path / "live", sleeper.pid)
    assert "earlier_run" not in server._job_facts(tmp_path / "live", started=time.time() + 5)   # a live job is current


def test_run_status_with_no_status_file_still_gives_the_job(tmp_path, dead_pid):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    put_lock(tmp_path, dead_pid, process_start=1.0)
    st = server.run_status(str(tmp_path))
    assert st["state"] == "none" and st["job"]["state"] == "stale_lock" and "stale lock" in st["summary"]


def _age(folder, seconds=3600):
    """Make the run files of a folder look old."""
    t = time.time() - seconds
    for name in ("status.json", "job.json", "run.json", "history.jsonl"):
        if (folder / name).exists():
            os.utime(folder / name, (t, t))


def _own_child(server, folder, code=1, kill=False):
    """A run process that this server 'started' for the folder, and that has ended."""
    child = subprocess.Popen([sys.executable, "-c", f"import sys, time\n{'time.sleep(60)' if kill else f'sys.exit({code})'}"])
    if kill:
        child.kill()
    child.wait()
    server._current.clear()
    server._current.update(proc=child, run_dir=folder, started=time.time(), port=8765)
    return child


@pytest.mark.parametrize("status_age", [3600, 0.5])
def test_run_status_says_so_when_the_new_run_process_ended_without_writing(tmp_path, status_age):
    """The folder has an older finished run. The new process (a YAML error, say) ended with exit 1 and wrote nothing.

    The old run can have ended less than a second before the new start (the 0.5 s case): no job record is new, so the
    status file from before the start is old.
    """
    pytest.importorskip("mcp")
    import fake_run
    from topopt_mcp import server
    fold = tmp_path / "run"
    fake_run.run(fold, spi=0.0, n_it=3)
    run_job(fold, lambda job: 0)
    _age(fold)
    t = time.time() - status_age
    os.utime(fold / "status.json", (t, t))
    _own_child(server, fold, code=1)
    try:
        st = server.run_status(str(fold))
    finally:
        server._current.clear()
    assert st["state"] == "error" and st["process_alive"] is False
    assert st["summary"].startswith("error: the run process ended with exit 1 before it wrote any run data")
    assert str(tmp_path / "run.stdout") in st["summary"]
    assert "status" not in st and "last_history" not in st             # the old run is not shown as the current state
    assert st["history_facts"] == []
    job = st["job"]
    assert job["state"] == "no_record" and job["exit_code"] == 1 and "wrote no job record" in job["summary"]
    assert job["earlier_job"]["state"] == "done" and job["log_file"] == str(tmp_path / "run.stdout")


def test_run_status_names_a_signal_and_an_empty_folder(tmp_path):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    fold = tmp_path / "run"
    fold.mkdir()
    _own_child(server, fold, kill=True)
    try:
        st = server.run_status(str(fold))
    finally:
        server._current.clear()
    assert st["state"] == "error" and "was stopped by signal 9 before it wrote any run data" in st["summary"]
    assert st["job"]["state"] == "no_record" and "earlier_job" not in st["job"]


def test_run_status_shows_a_run_that_this_server_started_and_that_wrote_its_files(tmp_path):
    """The process has ended, and its files are newer than its start: this is the normal result, not the error reply."""
    pytest.importorskip("mcp")
    import fake_run
    from topopt_mcp import server
    fold = tmp_path / "run"
    fold.mkdir()
    _own_child(server, fold, code=0)
    server._current["started"] = time.time() - 30
    fake_run.run(fold, spi=0.0, n_it=3)
    run_job(fold, lambda job: 0)
    try:
        st = server.run_status(str(fold))
    finally:
        server._current.clear()
    assert st["state"] == "done" and st["job"]["state"] == "done" and "earlier_run" not in st["job"]


def test_run_status_starting_and_none_replies_have_the_history_facts_key(tmp_path, sleeper):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    fold = tmp_path / "run"
    fold.mkdir()
    server._current.clear()
    assert server.run_status(str(fold))["state"] == "none" and server.run_status(str(fold))["history_facts"] == []
    server._current.update(proc=sleeper, run_dir=fold, started=time.time(), port=8765)
    try:
        st = server.run_status(str(fold))
    finally:
        server._current.clear()
    assert st["state"] == "starting" and st["history_facts"] == []


def test_run_topopt_gives_the_lock_fact_and_the_override(tmp_path, sleeper, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    y = tmp_path / "x.yaml"
    y.write_text("name: x\n")
    out = tmp_path / "run"
    rec = put_lock(out, sleeper.pid)
    r = server.run_topopt(str(y), str(out))
    assert "error" in r and str(sleeper.pid) in r["error"] and rec["started_at"] in r["error"]
    assert r["job"]["pid"] == sleeper.pid and r["run_dir"] == str(out) and "pid" not in r
    r2 = server.run_topopt(str(y), str(out), override_lock=True)
    assert "error" not in r2 and r2["pid"] > 0
    server._current["proc"].wait()
    server._current.clear()


def test_run_topopt_replaces_a_stale_lock_without_the_override(tmp_path, dead_pid, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    y = tmp_path / "x.yaml"
    y.write_text("name: x\n")
    put_lock(tmp_path / "run", dead_pid, process_start=1.0)
    r = server.run_topopt(str(y), str(tmp_path / "run"))
    assert "error" not in r
    server._current["proc"].wait()
    server._current.clear()


def _old_dashboard_process(server, folder):
    """A run of this server that is done and waits for the dashboard (a live process)."""
    import fake_run
    fake_run.run(folder, spi=0.0, n_it=3)
    old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    server._current.clear()
    server._current.update(proc=old, run_dir=folder, started=time.time() - 100, port=8765)
    return old


def test_run_topopt_keeps_the_old_dashboard_when_the_lock_refuses(tmp_path, sleeper, monkeypatch):
    """The lock of the new folder is held by another process. The old dashboard process must stay alive."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    old = _old_dashboard_process(server, tmp_path / "old")
    new = tmp_path / "new"
    put_lock(new, sleeper.pid)
    y = tmp_path / "x.yaml"
    y.write_text("name: x\n")
    try:
        r = server.run_topopt(str(y), str(new))
        assert "error" in r and str(sleeper.pid) in r["error"] and r["job"]["pid"] == sleeper.pid
        time.sleep(0.2)
        assert old.poll() is None                         # not stopped
        assert server._current["proc"] is old
    finally:
        old.kill()
        old.wait()
        server._current.clear()


def test_run_topopt_stops_its_own_old_run_that_holds_the_lock_of_the_same_folder(tmp_path, monkeypatch):
    """The lock holder is the old process of this server (it is in the refine step, say). It is stopped, not a conflict."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    folder = tmp_path / "old"
    old = _old_dashboard_process(server, folder)
    put_lock(folder, old.pid)
    y = tmp_path / "x.yaml"
    y.write_text("name: x\n")
    try:
        r = server.run_topopt(str(y), str(folder))
        assert "error" not in r and r["pid"] != old.pid
        assert old.poll() is not None                     # the old process is stopped
        server._current["proc"].wait()
    finally:
        old.kill()
        old.wait()
        server._current.clear()


def test_the_run_command_has_the_override_flag_only_when_asked(monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.delenv("TOPOPT_MCP_CMD", raising=False)
    y, o = tmp_path / "a.yaml", tmp_path / "o"
    assert "--override-lock" not in server._command(y, o, 8765)
    assert server._command(y, o, 8765, True)[-1] == "--override-lock"


# --- the CLI: `topopt run` goes through the helper ----------------------------------------------

def make_small(tmp_path, **kw):
    from test_optimize import make_case
    return make_case(tmp_path, h=2.0, iterations=3, **kw)


def run_cli(*argv):
    from topopt.cli import main
    return main(list(argv))


def test_cli_run_writes_the_job_files(tmp_path):
    p = make_small(tmp_path)
    out = tmp_path / "r"
    assert run_cli("run", str(p), "--out", str(out), "--quiet") == 0
    rec = info(out)
    assert rec["state"] == "done" and rec["exit_code"] == 0 and rec["wall_s"] > 0 and rec["peak_rss_bytes"] > 50e6
    assert (out / DONE_FILE).read_text() == "exit 0\n" and not (out / FAILED_FILE).exists()
    assert not (out / LOCK_FILE).exists() and not list(out.rglob("*.tmp"))
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "done" and rec["wall_s"] >= status["elapsed_s"]      # the helper counts the refine step too
    log = (out / "run.log").read_text()
    assert "job lock" not in log


def test_cli_guard_error_writes_a_failed_marker(tmp_path, capsys):
    p = make_small(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["settings"]["max_active_voxels"] = 10
    p.write_text(yaml.safe_dump(d))
    out = tmp_path / "r"
    assert run_cli("run", str(p), "--out", str(out), "--quiet") == 1
    assert (out / FAILED_FILE).read_text() == "exit 1\n" and not (out / DONE_FILE).exists()
    assert info(out)["state"] == "failed" and info(out)["exit_code"] == 1
    assert not (out / LOCK_FILE).exists()
    assert "max_active_voxels" in capsys.readouterr().err


def test_cli_internal_error_and_ctrl_c(tmp_path, monkeypatch, capsys):
    import topopt.optimize as optimize
    p = make_small(tmp_path)
    out = tmp_path / "r"

    def boom(*a, **k):
        raise RuntimeError("solver broke")

    monkeypatch.setattr(optimize, "run", boom)
    assert run_cli("run", str(p), "--out", str(out), "--quiet") == 2
    assert (out / FAILED_FILE).read_text() == "exit 2\n"

    def stop(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(optimize, "run", stop)
    assert run_cli("run", str(p), "--out", str(out), "--quiet") == 130
    assert (out / FAILED_FILE).read_text() == "exit 130\n" and info(out)["exit_code"] == 130
    capsys.readouterr()


def test_cli_a_bad_yaml_leaves_the_folder_alone(tmp_path):
    p = make_small(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["settings"]["bogus"] = 1
    p.write_text(yaml.safe_dump(d))
    out = tmp_path / "r"
    out.mkdir()
    (out / "run.json").write_text("old")
    assert run_cli("run", str(p), "--out", str(out), "--quiet") == 1
    assert [x.name for x in out.iterdir()] == ["run.json"]


def test_cli_live_lock_stops_the_run_with_the_facts_and_the_override_runs(tmp_path, sleeper, capsys):
    p = make_small(tmp_path)
    out = tmp_path / "r"
    rec = put_lock(out, sleeper.pid)
    (out / "run.json").write_text("old run")
    assert run_cli("run", str(p), "--out", str(out), "--quiet") == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and len(err.strip().splitlines()) == 1
    assert str(sleeper.pid) in err and rec["started_at"] in err and str(out) in err
    assert (out / "run.json").read_text() == "old run" and lock(out)["pid"] == sleeper.pid
    assert run_cli("run", str(p), "--out", str(out), "--quiet", "--override-lock") == 0
    log = (out / "run.log").read_text()
    assert "warning: job lock: override" in log and str(sleeper.pid) in log
    assert (out / DONE_FILE).exists() and sleeper.poll() is None


def test_cli_dead_lock_is_replaced_and_reported_in_run_log(tmp_path, dead_pid):
    p = make_small(tmp_path)
    out = tmp_path / "r"
    put_lock(out, dead_pid, process_start=1.0)
    assert run_cli("run", str(p), "--out", str(out), "--quiet") == 0
    log = (out / "run.log").read_text()
    assert f"warning: job lock: the process of the lock (pid {dead_pid}" in log and "is not running" in log
    assert info(out)["notes"] and (out / DONE_FILE).exists()


def test_cli_run_goes_on_without_flock_with_and_without_the_override(tmp_path, monkeypatch):
    """A file system with no flock (a network share): the run works, says so, and exits 0. No internal error."""
    def no_flock(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(runjob.fcntl, "flock", no_flock)
    p = make_small(tmp_path)
    out = tmp_path / "r"
    assert run_cli("run", str(p), "--out", str(out), "--quiet") == 0
    assert (out / DONE_FILE).exists() and runjob.NO_FLOCK_FACT in info(out)["notes"]
    assert "warning: " + runjob.NO_FLOCK_FACT in (out / "run.log").read_text()
    assert run_cli("run", str(p), "--out", str(out), "--quiet", "--override-lock") == 0
    assert (out / DONE_FILE).exists() and not (out / LOCK_FILE).exists()


def test_a_summary_names_the_floor_of_the_parent_when_the_peak_is_ru_maxrss():
    done = {"state": "done", "exit_code": 0, "wall_s": 1.0, "peak_rss_bytes": 322e6, "peak_rss_start_bytes": 244e6}
    text = runjob._summary(dict(done))
    assert "peak memory 0.32 GB; 0.24 GB of it was there at the start and can come from the parent process" in text
    assert "parent" not in runjob._summary(dict(done, peak_rss_source="VmHWM"))       # VmHWM has no floor
    assert "parent" not in runjob._summary(dict(done, peak_rss_start_bytes=50e6))     # a small start value
    assert "parent" not in runjob._summary(dict(done, peak_rss_bytes=1.7e9, peak_rss_start_bytes=0.25e9))  # the hook run
    assert runjob._summary({"state": "done", "exit_code": 0, "wall_s": 1.0}).endswith("peak memory ?)")


def test_cli_dashboard_run_writes_the_marker_before_the_wait(tmp_path, monkeypatch):
    """With --dashboard and no --no-wait the process sleeps after the run. The DONE marker is there before that."""
    import topopt.cli as cli
    p = make_small(tmp_path)
    out = tmp_path / "r"
    seen = {}

    real_sleep = time.sleep

    def fake_sleep(sec):
        if sec != 3600:                       # not the wait loop of cmd_run
            return real_sleep(sec)
        seen["done"] = (out / DONE_FILE).exists()
        seen["lock"] = (out / LOCK_FILE).exists()
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)
    assert run_cli("run", str(p), "--out", str(out), "--quiet", "--dashboard", "--port", "8791") == 0
    assert seen == {"done": True, "lock": False}
    assert info(out)["state"] == "done"


class _FakeServer:
    """What `keep_dashboard` uses of a `DashboardServer`: `url`, `socket`, `shutdown()` and `serve_forever()`."""
    url = "http://127.0.0.1:8123/"
    server_address = ("127.0.0.1", 8123)

    def __init__(self):
        import socket
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen()
        self.calls = []
        self.served = threading.Event()

    def shutdown(self):
        self.calls.append("shutdown")

    def serve_forever(self):
        self.calls.append("serve_forever")
        self.served.set()


@pytest.fixture
def fake_server():
    server = _FakeServer()
    yield server
    server.socket.close()


def _wait_probe(monkeypatch, server=None):
    """Fake `os.execv` and the wait loop of `keep_dashboard`. Returns the record of what the function did."""
    import topopt.cli as cli
    seen = {"exec": [], "sleep": []}

    def fake_sleep(sec):
        seen["sleep"].append(sec)
        raise KeyboardInterrupt                    # the user ends the wait

    def fake_execve(path, argv, env):
        # the state of the socket at the moment of the exec: it must pass to the new program, so it must be inheritable
        seen["exec"].append((path, list(argv)))
        seen["env"] = env
        if server is not None:
            seen["inheritable_at_exec"] = os.get_inheritable(server.socket.fileno())
            seen["shutdown_before_exec"] = "shutdown" in server.calls

    monkeypatch.setattr(cli.os, "execve", fake_execve)
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: fake_execve(path, argv, None))   # safe against an older program
    monkeypatch.setattr(cli.time, "sleep", fake_sleep)
    return seen


def test_the_wait_after_a_run_hands_the_process_to_the_small_dashboard(tmp_path, monkeypatch, capsys, fake_server):
    """P4, F4-9 and the review M1. The process that ran the optimizer held about 590 MB (Python heap, numba, libraries) while it
    only served the result. When it is the command line of the user, it becomes `topopt dashboard <run folder> --listen-fd N
    --no-open` (same pid) and one line says how to stop it. The listening socket passes to the new program (it is inheritable at
    the exec), so the port is never closed. The old program stops accepting first."""
    import topopt.cli as cli
    seen = _wait_probe(monkeypatch, fake_server)
    fd = fake_server.socket.fileno()
    assert not os.get_inheritable(fd)
    cli.keep_dashboard(fake_server, tmp_path, own_process=True)
    assert seen["exec"] == [(sys.executable, [sys.executable, "-P", "-m", "topopt", "dashboard", str(tmp_path.resolve()),
                                              "--listen-fd", str(fd), "--no-open"])]
    assert seen["inheritable_at_exec"] is True and seen["shutdown_before_exec"] is True
    text = capsys.readouterr().out
    assert f"pid {os.getpid()}" in text and f"kill {os.getpid()}" in text and "Ctrl+C" in text
    assert "http://127.0.0.1:8123/" in text and "a small process" in text and text.count("\n") == 1   # one line


@pytest.mark.parametrize("error", [OSError(8, "Exec format error"), ValueError("embedded null byte"), TypeError("bad path")])
def test_the_wait_stays_in_this_process_when_a_program_called_main_or_the_exec_fails(tmp_path, monkeypatch, capsys,
                                                                                    fake_server, error):
    import topopt.cli as cli
    seen = _wait_probe(monkeypatch)
    cli.keep_dashboard(fake_server, tmp_path, own_process=False)                 # a call from a program, not the command line
    assert seen["exec"] == [] and seen["sleep"] == [3600] and fake_server.calls == []
    assert "small process" not in capsys.readouterr().out

    seen["sleep"].clear()

    def no_exec(path, argv, env=None):
        raise error

    monkeypatch.setattr(cli.os, "execve", no_exec)
    monkeypatch.setattr(cli.os, "execv", no_exec)
    cli.keep_dashboard(fake_server, tmp_path, own_process=True)
    out = capsys.readouterr().out
    assert seen["sleep"] == [3600] and "the dashboard stays in this process" in out and type(error).__name__ in out
    assert fake_server.served.wait(5)                                            # the server serves again after the failed exec
    assert fake_server.calls == ["shutdown", "serve_forever"]
    assert not os.get_inheritable(fake_server.socket.fileno())                   # and the socket is as it was


def test_the_exec_does_not_put_the_current_folder_first_and_names_the_package_folder(tmp_path, monkeypatch, fake_server):
    """P4 review, L1. `python -m topopt` puts the current folder first on the path, so a `topopt.py` there ran instead of the
    package. The exec uses `-P` and puts the folder of the running package first on PYTHONPATH; an older PYTHONPATH stays after it."""
    import topopt
    import topopt.cli as cli
    package_parent = str(Path(topopt.__file__).resolve().parent.parent)
    seen = _wait_probe(monkeypatch, fake_server)
    monkeypatch.setenv("PYTHONPATH", "/older/one" + os.pathsep + "/older/two")
    cli.keep_dashboard(fake_server, tmp_path, own_process=True)
    argv = seen["exec"][0][1]
    assert argv[:4] == [sys.executable, "-P", "-m", "topopt"]
    assert seen["env"]["PYTHONPATH"].split(os.pathsep) == [package_parent, "/older/one", "/older/two"]
    monkeypatch.delenv("PYTHONPATH")
    cli.keep_dashboard(fake_server, tmp_path, own_process=True)
    assert seen["env"]["PYTHONPATH"] == package_parent                       # no older value: no trailing separator


def test_the_exec_passes_the_dashboard_socket_and_no_other_descriptor(tmp_path, monkeypatch, fake_server):
    """P4 review, L8. `import numba` opens a handle on the Python program with no close-on-exec flag, and the new program kept it
    (fd 4 in the dashboard process). At the exec only the standard streams and the listening socket may pass."""
    import topopt.cli as cli
    stray = os.open(__file__, os.O_RDONLY)
    os.set_inheritable(stray, True)                                           # as a C library leaves it
    seen = {}

    def fake_execve(path, argv, env):
        seen["listen"] = os.get_inheritable(fake_server.socket.fileno())
        seen["stray"] = os.get_inheritable(stray)
        seen["inheritable"] = [fd for fd in range(3, 256) if _is_open(fd) and os.get_inheritable(fd)]

    def _is_open(fd):
        try:
            os.fstat(fd)
            return True
        except OSError:
            return False

    monkeypatch.setattr(cli.os, "execve", fake_execve)
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: fake_execve(path, argv, None))
    monkeypatch.setattr(cli.time, "sleep", lambda sec: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        cli.keep_dashboard(fake_server, tmp_path, own_process=True)
    finally:
        os.close(stray)
    assert seen["listen"] is True and seen["stray"] is False
    assert seen["inheritable"] == [fake_server.socket.fileno()]


@needs_proc
def test_text_in_the_c_stdio_buffer_is_flushed_before_the_exec(tmp_path):
    """P4 review, L8. A library that prints through C leaves the text in the C buffer when stdout is a pipe or a file, and the
    exec drops it. The program here writes through C `puts`, then runs the wait with an exec that ends the process like the
    exec would (`os._exit`, which flushes nothing)."""
    script = tmp_path / "c_text.py"
    script.write_text(
        "import ctypes, os, socket, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import topopt.cli as cli\n"
        "class S:\n"
        "    url = 'http://127.0.0.1:1/'\n"
        "    def __init__(self):\n"
        "        self.socket = socket.socket(); self.socket.bind(('127.0.0.1', 0)); self.socket.listen()\n"
        "    def shutdown(self): pass\n"
        "cli.os.execve = lambda path, argv, env: os._exit(0)\n"
        "ctypes.CDLL(None).puts(b'TEXT FROM C STDIO')\n"
        "cli.keep_dashboard(S(), '.', own_process=True)\n")
    done = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert "TEXT FROM C STDIO" in done.stdout, (done.stdout, done.stderr)
    assert "Dashboard stays on" in done.stdout


def test_a_resumed_run_holds_its_model_only_in_run_folder_so_the_wait_can_drop_it(tmp_path, monkeypatch):
    """P4 review, L7. `cmd_run` built the model of a resume and the lambda for `run_job` kept a reference to it for the whole
    call, so the wait after the run (the fallback when there is no exec) still held it. Now `_run_folder` is the only holder."""
    import gc
    import weakref

    import topopt.cli as cli

    class Model:                                   # stands for the problem of a resumed run
        pass

    model = Model()
    ref = weakref.ref(model)
    holder = [model]
    del model                                      # only `holder` holds it; `_plan_resume` gives it up below
    seen = {}

    def fake_run_folder(args, config, out_dir, t_start, job, problem=None, ck=None, fine=None):   # P4b-5: the parent read (ck); fine pass: the fine plan
        seen["got_it"] = isinstance(problem, Model)
        del problem                                # what the wait does: drop the model
        gc.collect()
        seen["alive_after_drop"] = ref() is not None
        return 0

    monkeypatch.setattr(cli, "_plan_resume", lambda args, config, out_dir, problem=None: (holder.pop(), None))
    monkeypatch.setattr(cli, "_run_folder", fake_run_folder)
    out = tmp_path / "r"
    out.mkdir()
    args = cli.build_parser().parse_args(["run", str(make_small(tmp_path)), "--out", str(out), "--quiet", "--resume"])
    assert cli.cmd_run(args) == 0
    assert seen == {"got_it": True, "alive_after_drop": False}


def test_main_says_that_the_arguments_are_the_command_line_only_when_it_got_none(monkeypatch):
    import types

    import topopt.cli as cli
    got = []

    class FakeParser:
        def parse_args(self, argv):
            return types.SimpleNamespace(func=lambda a: got.append(a.own_process) or 0)

    monkeypatch.setattr(cli, "build_parser", lambda: FakeParser())
    assert cli.main(["run", "x.yaml"]) == 0 and cli.main() == 0
    assert got == [False, True]


class _DashboardRun:
    """A real `python -m topopt run --dashboard` on the small case, in its own session. `poll` starts a thread that asks
    `/api/run` every 10 ms from the start and records each answer."""

    def __init__(self, tmp_path, cwd=None, env=None, python_args=()):
        import socket
        self.tmp = tmp_path
        self.out = tmp_path / "r"
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.logfile = tmp_path / "stdout.txt"
        self.log = open(self.logfile, "wb")
        self.events = []                     # (time, "ok" or the name of the error)
        self.stop = False
        self.proc = subprocess.Popen(
            [sys.executable, *python_args, "-m", "topopt", "run", str(make_small(tmp_path)), "--out", str(self.out), "--quiet",
             "--dashboard", "--port", str(self.port)],
            stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
            env=dict(os.environ, BROWSER="true", **(env or {})), cwd=str(cwd or tmp_path))

    def poll(self):
        import urllib.error
        import urllib.request

        def loop():
            while not self.stop:
                t = time.time()
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/run", timeout=2).read()
                    self.events.append((t, "ok"))
                except urllib.error.URLError as e:
                    self.events.append((t, type(e.reason).__name__))
                except OSError as e:                                              # a reset in the middle of an answer
                    self.events.append((t, type(e).__name__))
                except Exception as e:  # noqa: BLE001  (http.client errors: RemoteDisconnected, IncompleteRead)
                    self.events.append((t, type(e).__name__))
                time.sleep(0.01)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def cmdline(self):
        return Path(f"/proc/{self.proc.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()

    def wait_for_the_small_dashboard(self, deadline):
        while not (self.out / DONE_FILE).exists():
            assert self.proc.poll() is None and time.time() < deadline, self.logfile.read_text()
            time.sleep(0.05)
        self.t_done = time.time()
        cmd = ""
        while "-m topopt dashboard" not in cmd and time.time() < deadline:      # the exec comes a moment after the marker
            assert self.proc.poll() is None, self.logfile.read_text()
            cmd = self.cmdline()
            time.sleep(0.01)
        return cmd

    def end(self):
        import signal
        self.stop = True
        if getattr(self, "thread", None):
            self.thread.join(timeout=10)
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
            return self.proc.wait(timeout=30)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.log.close()


@needs_proc
def test_a_real_run_with_a_dashboard_ends_as_a_small_dashboard_with_the_same_pid(tmp_path):
    """P4, F4-9, end to end. `python -m topopt run --dashboard` runs, writes the DONE marker, and the same process (same pid)
    then is `topopt dashboard` on the same port and serves the result. It ends with Ctrl+C (SIGINT) and the exit code 0."""
    import urllib.request

    run = _DashboardRun(tmp_path)
    try:
        deadline = time.time() + 240
        cmd = run.wait_for_the_small_dashboard(deadline)
        assert "-m topopt dashboard" in cmd and "--listen-fd" in cmd and "--no-open" in cmd
        body = None
        while body is None and time.time() < deadline:
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{run.port}/api/status", timeout=5).read()
            except OSError:
                time.sleep(0.05)
        assert json.loads(body)["state"] == "done"
        rss_kb = int(next(line.split()[1] for line in Path(f"/proc/{run.proc.pid}/status").read_text().splitlines()
                          if line.startswith("VmRSS:")))
    finally:
        code = run.end()
    assert code == 0
    text = run.logfile.read_text()
    assert f"pid {run.proc.pid}" in text and f"kill {run.proc.pid}" in text
    assert rss_kb < 200_000                           # the optimizer process was about 590 MB at this point (measured)


@needs_proc
def test_the_dashboard_port_is_never_closed_at_the_exec(tmp_path):
    """P4 review, M1. A client asks `/api/run` every 10 ms from the start of the run, through the exec. Before, the port was
    closed for 0.2 to 0.5 s (the old program closed its socket, the new one bound it again), and the page saw "connection
    refused". Now the listening socket passes to the new program: no request is refused. (A request in progress at the exec can
    be reset; that is not counted here.)"""
    run = _DashboardRun(tmp_path)
    try:
        run.poll()
        deadline = time.time() + 240
        run.wait_for_the_small_dashboard(deadline)
        t_exec = time.time()
        time.sleep(2)                                                           # many requests after the exec
    finally:
        code = run.end()
    assert code == 0
    first_ok = min(t for t, kind in run.events if kind == "ok")                   # before it the server has not started yet
    assert first_ok < run.t_done
    refused = [(round(t - run.t_done, 3), kind) for t, kind in run.events if kind == "ConnectionRefusedError" and t > first_ok]
    assert refused == [], f"the port was closed: {refused}"
    after = [k for t, k in run.events if t > t_exec]
    assert len(after) > 20 and after.count("ok") >= len(after) - 1, after        # all asked after the exec were answered
    assert any(k == "ok" for t, k in run.events if t > run.t_done)               # also in the window after the marker


@needs_proc
def test_a_topopt_py_in_the_current_folder_does_not_replace_the_package_at_the_exec(tmp_path):
    """P4 review, L1, end to end. The run starts with `-P` (no current folder on the path) and the package on PYTHONPATH, as the
    console script does. The current folder has a `topopt.py` that prints SHADOW. The exec must still start the real package:
    with `-m topopt` and no `-P` it started the shadow file, the process ended and the port closed."""
    import urllib.request

    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "topopt.py").write_text('print("SHADOW topopt.py from the current folder", flush=True)\n')
    run = _DashboardRun(tmp_path, cwd=shadow, python_args=("-P",), env={"PYTHONPATH": str(ROOT)})
    try:
        deadline = time.time() + 240
        cmd = run.wait_for_the_small_dashboard(deadline)
        assert "-P -m topopt dashboard" in cmd
        body = None
        while body is None and time.time() < deadline:
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{run.port}/api/status", timeout=5).read()
            except OSError:
                time.sleep(0.05)
        assert json.loads(body)["state"] == "done"
    finally:
        code = run.end()
    assert code == 0 and "SHADOW" not in run.logfile.read_text()
