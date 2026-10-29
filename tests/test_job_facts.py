"""Tests of the facts that each job records for the data loop (DL1a, CONTRACTS.md sections 5 (`problem_key`) and 24.1 (`job.json`)).

`job.json` has the block `host` at the start, and at the end the CPU time, the context switches, the load, the phases of a run and the
peak at the end of the loop. On Linux the peak of the job is its own (the peak is reset at the start). `run.json` has `problem_key`.
The helper functions are in tests/test_hostinfo.py; here are the record, the run and the readers.
"""
import dataclasses
import json
import os
import re
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

import test_checkpoint as tc  # noqa: E402
from test_optimize import box as make_box  # noqa: E402
from test_optimize import make_case  # noqa: E402
from test_ratchet_mcp import case, kill_job  # noqa: E402,F401  (the fixture `case` of the ratchet MCP tests)

from topopt import hostinfo, ratchet, runjob, snapshot  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.config import load_config  # noqa: E402
from topopt.problem import build_problem  # noqa: E402
from topopt.runjob import INFO_FILE, RunJob, job_facts, run_job  # noqa: E402
from topopt.snapshot import RunWriter, problem_key  # noqa: E402

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux") or not Path("/proc/self/stat").exists(), reason="needs /proc (Linux)")
PROBE = TESTS / "job_probe.py"
HOST_KEYS = {"started_by", "threads", "cpu_count", "cpu_usable", "cpu_model", "ram_total_bytes", "ram_available_bytes", "load_avg",
             "procs_running", "other_jobs", "version", "low_power_mode", "notes"}
END_KEYS = {"cpu_time_s", "cpu_time_children_s", "ctx_switches_voluntary", "ctx_switches_involuntary", "load_avg_end", "procs_running_end",
            "setup_s", "loop_s", "post_s", "peak_rss_loop_bytes"}


def info(d):
    return json.loads((Path(d) / INFO_FILE).read_text())


def run_json(d):
    return json.loads((Path(d) / "run.json").read_text())


def run_cli(y, out, *extra):
    assert main(tc.run_args(y, out, *extra)) == 0


@pytest.fixture(scope="module")
def short(tmp_path_factory):
    """One short CPU run in this process (the 20 x 10 x 10 mm cantilever of tests/test_optimize.py at 2 mm, 3 passes)."""
    d = tmp_path_factory.mktemp("short")
    y = make_case(d, h=2.0, iterations=3)
    run_cli(y, d / "out")
    return dataclasses.make_dataclass("Short", ["dir", "yaml", "out"])(d, y, d / "out")


# --- job.json of a run ------------------------------------------------------------------------

def test_the_job_json_of_a_short_cpu_run_has_the_facts_of_the_host(short):
    rec = info(short.out)
    assert rec["state"] == "done" and set(rec["host"]) == HOST_KEYS
    host = rec["host"]
    assert host["started_by"] == "pytest"
    assert set(host["threads"]) == set(hostinfo.THREAD_VARS) and len(hostinfo.THREAD_VARS) == 6
    assert host["cpu_count"] >= host["cpu_usable"] >= 1 and isinstance(host["cpu_model"], str)
    assert 0 < host["ram_available_bytes"] <= host["ram_total_bytes"] and len(host["load_avg"]) == 3
    assert host["other_jobs"]["count"] == sum(host["other_jobs"]["by_kind"].values()) == len(host["other_jobs"]["processes"])
    assert set(host["other_jobs"]["by_kind"]) == {"run", "verify", "mcp_server", "other"}
    assert all(set(p) == {"pid", "kind"} for p in host["other_jobs"]["processes"])           # pids and kinds only
    assert host["version"]["topopt"] and host["version"]["numpy"] and host["version"]["scipy"]
    assert all(isinstance(n, str) for n in host["notes"])


def test_the_job_json_of_a_short_cpu_run_has_the_facts_of_the_end(short):
    rec = info(short.out)
    assert END_KEYS <= set(rec)
    assert rec["cpu_time_s"] > 0 and rec["cpu_time_s"] >= rec["cpu_time_children_s"] >= 0
    assert isinstance(rec["ctx_switches_voluntary"], int) and isinstance(rec["ctx_switches_involuntary"], int)
    assert len(rec["load_avg_end"]) == 3
    # the phases are parts of the time from the start of the command to the end of the post step, which is before the job record
    assert min(rec["setup_s"], rec["loop_s"], rec["post_s"]) >= 0
    assert rec["setup_s"] + rec["loop_s"] + rec["post_s"] <= rec["wall_s"] + 0.1
    assert 0 < rec["peak_rss_loop_bytes"] <= rec["peak_rss_bytes"]


@linux_only
def test_a_run_on_linux_resets_the_peak_and_has_procs_running(short):
    rec = info(short.out)
    assert rec["peak_rss_reset"] is True and rec["peak_rss_source"] == "VmHWM"
    assert not any("not done" in n for n in rec["host"]["notes"])                 # a fresh run builds its problem inside the job: it asks for the reset
    assert isinstance(rec["host"]["procs_running"], int) and isinstance(rec["procs_running_end"], int)
    assert rec["host"]["version"]["git_sha"] is None or re.fullmatch(r"[0-9a-f]{40,64}", rec["host"]["version"]["git_sha"])


def test_run_status_does_not_carry_the_host_block_but_the_end_numbers(short):
    facts = job_facts(short.out)
    assert "host" not in facts and "token" not in facts
    assert facts["cpu_time_s"] > 0 and facts["peak_rss_reset"] in (True, False) and facts["state"] == "done"
    assert (END_KEYS | {"peak_rss_reset"}) <= set(facts)                         # the 11 keys of the end stay in the reply (CONTRACTS 24.1): they are short


def test_run_json_has_the_problem_key_of_its_problem(short):
    key = run_json(short.out)["problem_key"]
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert key == problem_key_of(short.yaml)


# --- the peak of a job is its own (Linux) --------------------------------------------------------

def rss_now():
    return next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmRSS:")) * 1024


def touch(n_bytes):
    block = bytearray(n_bytes)
    block[::4096] = b"x" * len(block[::4096])
    return block


@linux_only
def test_the_peak_of_a_job_is_its_own_and_not_the_peak_of_the_process_before(tmp_path):
    gone = touch(250 * 2 ** 20)                     # the process before the job (a test process, a long server)
    del gone
    before = rss_now()
    kept = []
    rec = None

    def body(job):
        kept.append(touch(120 * 2 ** 20))           # the job
        return 0

    run_job(tmp_path, body)
    rec = info(tmp_path)
    refused = [n for n in rec["host"]["notes"] if n.startswith("peak_rss_reset:")]
    if refused:                                     # only a write that the system refused (a note names it) is a reason to skip
        pytest.skip(f"this /proc does not allow the reset: {refused}")
    assert rec["peak_rss_reset"] is True            # a silent False is a failure of the reset, not a skip
    assert rec["peak_rss_start_bytes"] > before + 200 * 2 ** 20            # the peak of the process before the job stays as a fact
    assert rec["peak_rss_bytes"] < rec["peak_rss_start_bytes"] - 50 * 2 ** 20     # the job's own peak is lower
    assert rec["peak_rss_bytes"] > before + 100 * 2 ** 20                 # and it holds what the job used
    assert "VmHWM" == rec["peak_rss_source"]


def test_without_the_reset_the_peak_is_not_less_than_the_start_value(tmp_path, monkeypatch):
    monkeypatch.setattr(hostinfo, "reset_peak_rss", lambda notes=None: False)
    run_job(tmp_path, lambda job: 0)
    rec = info(tmp_path)
    assert rec["peak_rss_reset"] is False and rec["peak_rss_bytes"] >= rec["peak_rss_start_bytes"] > 0


@linux_only
def test_a_job_that_keeps_the_peak_has_no_reset_a_reason_and_the_old_rule(tmp_path, monkeypatch):
    called = []
    real = hostinfo.reset_peak_rss
    monkeypatch.setattr(hostinfo, "reset_peak_rss", lambda notes=None: called.append(1) or real(notes))
    run_job(tmp_path / "a", lambda job: 0, keep_peak="the build came first")
    rec = info(tmp_path / "a")
    assert called == [] and rec["peak_rss_reset"] is False and rec["peak_rss_source"] == "VmHWM"
    assert rec["peak_rss_bytes"] >= rec["peak_rss_start_bytes"] > 0              # the old rule: the peak is not below the start value
    assert "peak_rss_reset: not done, the build came first" in rec["host"]["notes"]
    run_job(tmp_path / "b", lambda job: 0)                                       # the default is the reset, with no such note
    assert info(tmp_path / "b")["peak_rss_reset"] is True and not any("not done" in n for n in info(tmp_path / "b")["host"]["notes"])


@linux_only
def test_a_write_that_fails_is_false_with_a_note_and_the_old_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(hostinfo, "PROC_CLEAR_REFS", tmp_path / "no_folder" / "clear_refs")
    run_job(tmp_path / "job", lambda job: 0)
    rec = info(tmp_path / "job")
    assert rec["peak_rss_reset"] is False and rec["peak_rss_bytes"] >= rec["peak_rss_start_bytes"]
    assert any(n.startswith("peak_rss_reset: FileNotFoundError") for n in rec["host"]["notes"])


def test_on_macos_the_record_has_no_reset_and_the_mac_facts(tmp_path, monkeypatch):
    texts = {"ps": "  PID COMMAND\n  77 /usr/bin/python3 -m topopt run a.yaml\n", "pmset": " lowpowermode         1\n"}

    def command(argv):
        if argv[-1] == "machdep.cpu.brand_string":
            return "Apple M2\n"
        return texts[argv[0]]
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(hostinfo, "_command_output", command)
    run_job(tmp_path, lambda job: 0)
    rec = info(tmp_path)
    host = rec["host"]
    assert rec["platform"] == "darwin" and rec["peak_rss_reset"] is False and rec["peak_rss_bytes"] >= rec["peak_rss_start_bytes"]
    assert host["cpu_model"] == "Apple M2" and host["low_power_mode"] is True
    assert host["procs_running"] is None and rec["procs_running_end"] is None
    assert host["notes"].count("procs_running: needs /proc/stat (Linux)") == 1          # once per job, at the start (the end adds no second one)
    assert host["other_jobs"]["by_kind"]["run"] == 1 and host["other_jobs"]["processes"][0]["pid"] == 77
    assert rec["cpu_time_s"] >= 0 and len(rec["load_avg_end"]) == 3     # an empty job uses no CPU time to speak of


# --- the CPU time of a job is its own ------------------------------------------------------------

def own_cpu():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


def spin(cpu_s):
    """Use `cpu_s` seconds of CPU in this process, counted as `getrusage` counts it."""
    end = own_cpu() + cpu_s
    while own_cpu() < end:
        sum(range(20_000))


def test_the_cpu_time_of_a_job_does_not_hold_the_work_of_the_process_before_it(tmp_path):
    spin(0.5)                                                                    # a test, a long server, the start-up of a command
    subprocess.run([sys.executable, "-c", "sum(range(6_000_000))"], check=True)  # a child that the process has waited for
    kids_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    before = own_cpu()
    assert before >= 0.5 and kids_before.ru_utime > 0.05

    def body(job):
        spin(0.1)
        return 0

    run_job(tmp_path, body)
    rec = info(tmp_path)
    assert rec["cpu_time_s"] >= 0.1 - 0.002                                      # the work of the job is in it (0.002 s: the values are rounded to 1 ms)
    assert rec["cpu_time_s"] < 0.4                                               # the 0.5 s before it are not, and nor is the child
    assert rec["cpu_time_children_s"] < kids_before.ru_utime + kids_before.ru_stime


def test_the_cpu_time_of_a_job_holds_its_own_children_and_the_switches_are_its_own(tmp_path):
    def body(job):
        subprocess.run([sys.executable, "-c", "sum(range(6_000_000))"], check=True)
        return 0

    before = resource.getrusage(resource.RUSAGE_SELF)
    run_job(tmp_path, body)
    rec = info(tmp_path)
    whole = resource.getrusage(resource.RUSAGE_SELF).ru_nvcsw
    assert rec["cpu_time_children_s"] > 0.05 and rec["cpu_time_s"] >= rec["cpu_time_children_s"]
    assert 0 <= rec["ctx_switches_voluntary"] <= whole - before.ru_nvcsw          # a difference: the job is inside that window. The count of the whole process is not


# --- the facts never stop a job ------------------------------------------------------------------

def test_a_host_block_that_raises_does_not_stop_the_job(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("facts are broken")
    monkeypatch.setattr(hostinfo, "start_facts", boom)
    monkeypatch.setattr(hostinfo, "end_facts", boom)
    assert run_job(tmp_path, lambda job: 0) == 0
    rec = info(tmp_path)
    assert rec["state"] == "done" and rec["host"]["notes"][0] == "host: RuntimeError: facts are broken"
    assert "end_facts: RuntimeError: facts are broken" in rec["host"]["notes"]
    assert rec["setup_s"] is None and rec["peak_rss_bytes"] > 0


def test_a_refused_start_resets_nothing_and_writes_nothing(tmp_path, monkeypatch):
    holder = RunJob(tmp_path)
    holder.start()                                   # this process holds the lock; the second start needs the override
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    called = []
    monkeypatch.setattr(hostinfo, "reset_peak_rss", lambda notes=None: called.append("reset") or True)
    other = RunJob(tmp_path)
    with pytest.raises(runjob.JobHeld):
        other.start()
    assert called == [] and other.token is None
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before     # the record and the lock of the holder are as they were
    holder.finish(0)


def test_the_host_facts_are_read_before_job_guard_is_taken(tmp_path, monkeypatch):
    """A start fact can wait for a command (macOS). A second start of the folder waits on `job.guard`, so the guard is free while the facts are read."""
    import fcntl
    seen = []

    def facts():
        fd = os.open(tmp_path / runjob.GUARD_FILE, os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen.append("free")
        except BlockingIOError:
            seen.append("held")
        finally:
            os.close(fd)
        return {"notes": []}
    monkeypatch.setattr(hostinfo, "start_facts", facts)
    run_job(tmp_path, lambda job: 0)
    assert seen == ["free"] and info(tmp_path)["host"] == {"notes": []}


def test_a_second_start_does_not_wait_for_the_host_facts_of_the_first(tmp_path, monkeypatch):
    """The first start is slow in its facts (a command that hangs); the second start of the same folder is refused at once."""
    import threading
    slow, release = threading.Event(), threading.Event()

    def facts():
        slow.set()
        release.wait(15)
        return {"notes": []}
    monkeypatch.setattr(hostinfo, "start_facts", facts)
    first = RunJob(tmp_path)
    worker = threading.Thread(target=first.start)
    worker.start()
    assert slow.wait(10)
    try:
        monkeypatch.setattr(hostinfo, "start_facts", lambda: {"notes": []})
        started = time.time()
        second = RunJob(tmp_path)
        second.start()                                # the folder has no lock yet: the second start is not held up by the first one's facts
        assert time.time() - started < 5
        second.finish(0)
    finally:
        release.set()
        worker.join(30)
    first.finish(0)


# --- the phases ----------------------------------------------------------------------------------

class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def at(self, t):
        self.t = t
        return self


def writer(tmp_path, monkeypatch, t_start=1000.0):
    clock = Clock(t_start)
    monkeypatch.setattr(snapshot.time, "time", clock)
    return RunWriter(tmp_path, t_start, quiet=True), clock


def test_phases_are_the_spans_between_the_first_times_of_the_states(tmp_path, monkeypatch):
    out, clock = writer(tmp_path, monkeypatch)
    out.status("starting", 0, "setup")
    clock.at(1003.0)
    out.step(0, "assemble")                      # the first pass: the end of the setup
    clock.at(1004.0)
    out.step(0, "solve")                         # the first time counts: later steps do not move it
    clock.at(1013.25)
    out.status("postprocess", 3, "postprocess")
    clock.at(1018.0)
    out.status("done", 3, "done")
    t = out.phase_times()
    assert t["setup_s"] == 3.0 and t["loop_s"] == 10.25 and t["post_s"] == 4.75 and t["peak_rss_loop_bytes"] > 10e6


def test_a_resume_or_continuation_that_moves_the_start_back_does_not_move_the_phases(tmp_path, monkeypatch):
    out, clock = writer(tmp_path, monkeypatch)
    out.status("starting", 0, "setup")
    out.t_start = clock() - 3600.0               # what `apply_plan` and `apply_continue` do: elapsed_s goes on from the parent (an hour of passes)
    clock.at(1002.0)
    out.step(0, "assemble")
    clock.at(1005.0)
    out.status("postprocess", 3, "postprocess")
    clock.at(1006.0)
    out.status("done", 3, "done")
    assert out.elapsed() == 3600.0 + 6.0         # the clock of the parent goes on
    t = out.phase_times()
    assert (t["setup_s"], t["loop_s"], t["post_s"]) == (2.0, 3.0, 1.0)


def test_a_run_that_failed_has_only_the_phases_it_finished(tmp_path, monkeypatch):
    out, clock = writer(tmp_path, monkeypatch)
    out.status("starting", 0, "setup")
    clock.at(1002.0)
    out.step(1, "solve")
    clock.at(1009.0)
    out.status("error", 1, "solve", "boom")
    t = out.phase_times()
    assert t == {"setup_s": 2.0, "loop_s": None, "post_s": None, "peak_rss_loop_bytes": None}
    early, clock2 = writer(tmp_path / "e", monkeypatch)
    early.dir.mkdir()
    early.status("starting", 0, "setup")
    clock2.at(1001.0)
    early.status("error", 0, "setup", "boom")
    assert early.phase_times() == {"setup_s": None, "loop_s": None, "post_s": None, "peak_rss_loop_bytes": None}


def test_a_run_that_a_request_stopped_has_a_loop_and_no_post_step(tmp_path, monkeypatch):
    out, clock = writer(tmp_path, monkeypatch)
    out.status("starting", 0, "setup")
    clock.at(1001.5)
    out.step(0, "assemble")
    clock.at(1006.5)
    out.status("stopped", 4, "stopped", "stopped by a request")
    t = out.phase_times()
    assert t["setup_s"] == 1.5 and t["loop_s"] == 5.0 and t["post_s"] is None and t["peak_rss_loop_bytes"] > 10e6


def test_the_job_takes_its_phases_from_the_writer_also_when_the_run_fails(tmp_path, monkeypatch):
    out, clock = writer(tmp_path / "run", monkeypatch)
    out.dir.mkdir()

    def body(job):
        job.watch_phases(out)
        out.status("starting", 0, "setup")
        clock.at(1002.0)
        out.step(0, "assemble")
        raise RuntimeError("fails in the loop")

    with pytest.raises(RuntimeError):
        run_job(tmp_path / "run", body)
    rec = info(tmp_path / "run")
    assert rec["state"] == "failed" and rec["setup_s"] == 2.0 and rec["loop_s"] is None and rec["post_s"] is None


def test_a_job_with_no_writer_has_null_phases_and_a_broken_writer_a_note(tmp_path):
    run_job(tmp_path / "a", lambda job: 0)
    rec = info(tmp_path / "a")
    assert (rec["setup_s"], rec["loop_s"], rec["post_s"], rec["peak_rss_loop_bytes"]) == (None, None, None, None)

    class Broken:
        def phase_times(self):
            raise ValueError("no clock")

    def body(job):
        job.watch_phases(Broken())
        return 0
    run_job(tmp_path / "b", body)
    rec = info(tmp_path / "b")
    assert rec["setup_s"] is None and "phases: ValueError: no clock" in rec["host"]["notes"]


# --- problem_key ---------------------------------------------------------------------------------

def problem_key_of(yaml_path):
    """The key as `RunWriter.write_run_json` makes it: from `problem.config` (the loader config with the values that the problem derives),
    not from the loader config itself (DL1a review m9)."""
    prob = build_problem(load_config(yaml_path))
    return problem_key(prob.config, prob.shape, prob.h)


def variant(src_dir, dst_dir, edit=None):
    """A copy of the folder of `make_case`, with the YAML changed by `edit(dict)`. Return the YAML path."""
    import shutil
    shutil.copytree(src_dir, dst_dir)
    path = dst_dir / "c.yaml"
    if edit is not None:
        doc = yaml.safe_load(path.read_text())
        edit(doc)
        path.write_text(yaml.safe_dump(doc))
    return path


def test_the_problem_key_is_the_same_for_a_copied_folder(short, tmp_path):
    copy = variant(short.dir, tmp_path / "somewhere" / "else")
    assert problem_key_of(copy) == run_json(short.out)["problem_key"]


def test_the_problem_key_changes_when_a_force_changes(short, tmp_path):
    def stronger(doc):
        doc["load_cases"][0]["forces"][0]["force"] = [0, 0, -2.5]
    assert problem_key_of(variant(short.dir, tmp_path / "f", stronger)) != run_json(short.out)["problem_key"]


@pytest.mark.parametrize("name, edit", [
    ("volume fraction", lambda d: d["settings"].update(volume_fraction=0.31)),
    ("voxel size", lambda d: d["settings"].update(voxel_size=1.0)),
    ("objective", lambda d: d["settings"].update(objective="mass")),
    ("a load case weight", lambda d: d["load_cases"][1].update(weight=2.0)),
    ("a load case removed", lambda d: d["load_cases"].pop()),
    ("the material", lambda d: d.update(material={"E": 3000.0, "nu": 0.35, "density": 1.24})),
])
def test_the_problem_key_changes_with_the_problem(short, tmp_path, name, edit):
    assert problem_key_of(variant(short.dir, tmp_path / "v", edit)) != run_json(short.out)["problem_key"], name


def test_regions_not_printed_does_not_change_the_key_and_a_renamed_file_keeps_it(short, tmp_path):
    """`regions.not_printed` shapes result/part_only.stl in the post step only; the solver gets the same model (tests/test_part_only.py),
    so it is not in the key. A renamed region file keeps the key too (the content counts, not the name)."""
    plain = problem_key_of(short.yaml)
    marked = variant(short.dir, tmp_path / "a", lambda d: d["regions"].update(not_printed=["plate.stl"]))

    def renamed(doc):
        doc["regions"]["fixed"] = ["clamp.stl"]
        doc["regions"]["not_printed"] = ["clamp.stl"]
    other = variant(short.dir, tmp_path / "b", renamed)
    (other.parent / "plate.stl").rename(other.parent / "clamp.stl")
    other_plain = variant(short.dir, tmp_path / "c", lambda d: d["regions"].update(fixed=["clamp.stl"]))
    (other_plain.parent / "plate.stl").rename(other_plain.parent / "clamp.stl")
    assert load_config(marked).not_printed and load_config(other).not_printed     # the variants do use the key
    assert problem_key_of(marked) == problem_key_of(other) == problem_key_of(other_plain) == plain


def test_the_problem_key_of_a_run_is_the_key_of_its_problem_config(short):
    """The helper and `write_run_json` make the key from the same object: the key in run.json equals the key from `problem.config`."""
    assert problem_key_of(short.yaml) == run_json(short.out)["problem_key"]


def mass_objective(start_density):
    def edit(doc):
        doc["settings"].update(objective="mass", volume_fraction=start_density)
        doc["load_cases"][0]["compliance_limit"] = 50.0
    return edit


@pytest.mark.parametrize("name, edit", [
    ("a load case name", lambda d: d["load_cases"][0].update(name="hang")),
    ("a load region name", lambda d: (d["regions"].update(load={"tip2": "tip.stl"}),
                                      [f.update(region="tip2") for c in d["load_cases"] for f in c["forces"]])),
    ("the way that the cases join", lambda d: d["settings"].update(case_combine="max")),
])
def test_the_problem_key_holds_the_names_of_the_rows_and_the_case_combine(short, tmp_path, name, edit):
    """Part of the key on purpose (CONTRACTS section 5): the rows name the cases and the load regions."""
    assert problem_key_of(variant(short.dir, tmp_path / "n", edit)) != run_json(short.out)["problem_key"], name


def test_the_start_density_of_the_mass_objective_is_part_of_the_key(short, tmp_path):
    a = problem_key_of(variant(short.dir, tmp_path / "a", mass_objective(0.5)))
    b = problem_key_of(variant(short.dir, tmp_path / "b", mass_objective(0.6)))
    assert a != b and a == problem_key_of(variant(short.dir, tmp_path / "c", mass_objective(0.5)))


@pytest.mark.parametrize("name, edit", [
    ("the acceptance rules", lambda d: d.update(acceptance={"rules": [{"rule": "watertight"}]})),
    ("the print block", lambda d: d.update(print={"self_supporting_angle_deg": 40.0})),
])
def test_the_problem_key_does_not_hold_the_checks_of_a_result(short, tmp_path, name, edit):
    """Out of the key on purpose (CONTRACTS section 5): they check a result and do not change the model that the solver gets."""
    assert problem_key_of(variant(short.dir, tmp_path / "k", edit)) == problem_key_of(short.yaml), name


def test_the_text_of_the_allowable_source_is_not_in_the_key_but_the_allowable_is(short, tmp_path):
    def with_source(text):
        return lambda d: d["load_cases"][1].update(allowable_mpa=40.0, safety_factor=4.0, allowable_source=text)
    a = problem_key_of(variant(short.dir, tmp_path / "a", with_source("a value")))
    assert a == problem_key_of(variant(short.dir, tmp_path / "b", with_source("another text")))
    assert a != problem_key_of(short.yaml)                                        # the allowable is a stress row of the model


def test_the_problem_key_that_cannot_be_made_is_null_and_run_log_says_why(tmp_path, monkeypatch):
    def broken(cfg, shape, h):
        raise ValueError("no key today")
    monkeypatch.setattr(snapshot, "problem_key", broken)
    y = make_case(tmp_path, h=2.0, iterations=1)
    run_cli(y, tmp_path / "out")
    assert run_json(tmp_path / "out")["problem_key"] is None and info(tmp_path / "out")["state"] == "done"
    assert "warning: problem_key is null in run.json: ValueError: no key today" in (tmp_path / "out" / "run.log").read_text()


def test_the_problem_key_changes_when_a_region_file_changes(short, tmp_path):
    y = variant(short.dir, tmp_path / "r")
    (y.parent / "tip.stl").unlink()                  # a longer load box: the content of the file differs, its name does not
    make_box(y.parent / "tip.stl", (16, 4, 0), (20, 6, 3))
    assert problem_key_of(y) != run_json(short.out)["problem_key"]


@pytest.mark.parametrize("name, edit", [
    ("iterations", lambda d: d["settings"].update(iterations=99)),
    ("solver", lambda d: d["settings"].update(solver="amg")),
    ("the name", lambda d: d.update(name="another name")),
    ("filter radius", lambda d: d["settings"].update(filter_radius=3.0)),
    ("beta schedule and move limit", lambda d: d["settings"].update(beta_schedule=[[0, 1], [5, 8]], move_limit=0.1)),
])
def test_the_problem_key_does_not_change_with_the_method(short, tmp_path, name, edit):
    assert problem_key_of(variant(short.dir, tmp_path / "m", edit)) == run_json(short.out)["problem_key"], name


def test_the_setup_diff_of_compare_runs_does_not_list_the_hash(short):
    from topopt.feedback import _setup_view
    run = run_json(short.out)
    other = dict(run, problem_key="0" * 64)
    assert "problem_key" not in _setup_view(run) and _setup_view(run) == _setup_view(other)


# --- derived estimates on every path -------------------------------------------------------------

def has_estimates(d):
    der = run_json(d)["derived"]
    return isinstance(der.get("time_estimate_s"), float) and isinstance(der.get("memory_estimate_gib"), float)


def test_a_fresh_run_has_both_estimates(short):
    assert has_estimates(short.out)


@pytest.fixture(scope="module")
def continued(short, tmp_path_factory):
    """`--continue-from` the short run, `--more-iterations 2`: a new run folder with 3 + 2 passes."""
    out = tmp_path_factory.mktemp("continued") / "out"
    run_cli(short.yaml, out, "--continue-from", str(short.out), "--more-iterations", "2")
    return out


@pytest.fixture(scope="module")
def resumed(short, tmp_path_factory):
    """A run that was killed after pass 1 (SIGKILL, in a child process) and then resumed with `--resume`."""
    out = tmp_path_factory.mktemp("resumed") / "out"
    tc.killed(short.yaml, out, "after_checkpoint:1")
    tc.resume(short.yaml, out)
    return out


def phases_fit_the_wall_time(rec):
    """The three phases are parts of the time of THIS command: their sum is not more than `wall_s` (0.1 s is the tolerance of two timer reads)."""
    assert min(rec["setup_s"], rec["loop_s"], rec["post_s"]) >= 0
    assert rec["setup_s"] + rec["loop_s"] + rec["post_s"] <= rec["wall_s"] + 0.1, rec


def test_a_continuation_has_both_estimates(continued):
    assert "continued_from" in run_json(continued) and has_estimates(continued)
    phases_fit_the_wall_time(info(continued))        # not the clock of the parent run (status.json elapsed_s goes on from it, the phases do not)


def test_a_resumed_run_has_both_estimates(resumed):
    assert has_estimates(resumed)


@linux_only
@pytest.mark.parametrize("which", ["continued", "resumed"])
def test_a_resume_and_a_continuation_keep_the_peak_of_the_problem_build(which, request):
    """The plan builds the problem before the job; that build can be the peak of the command, so the job does not reset (DL1a review m5)."""
    rec = info(request.getfixturevalue(which))
    assert rec["peak_rss_source"] == "VmHWM" and rec["peak_rss_reset"] is False
    assert rec["peak_rss_bytes"] >= rec["peak_rss_start_bytes"] > 0
    assert any(n.startswith("peak_rss_reset: not done, the plan of this command") for n in rec["host"]["notes"])


def test_a_continuation_has_the_problem_key_of_its_parent(short, continued):
    run = run_json(continued)
    assert run["iterations"] == 5 and run["problem_key"] == run_json(short.out)["problem_key"]     # 3 + 2 passes, the same problem


def test_a_resumed_run_has_the_problem_key_and_writes_the_job_facts(short, resumed):
    assert run_json(resumed)["problem_key"] == run_json(short.out)["problem_key"]
    rec = info(resumed)
    assert rec["state"] == "done" and set(rec["host"]) == HOST_KEYS and END_KEYS <= set(rec)      # the resumed job writes the facts too
    phases_fit_the_wall_time(rec)


def test_a_run_that_the_mcp_server_starts_has_the_estimates_and_started_by_mcp(tmp_path, monkeypatch, server_started):
    server = server_started
    y = make_case(tmp_path, h=2.0, iterations=2)
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{shlex.quote(sys.executable)} -m topopt run {{yaml}} --out {{out}} --quiet")
    monkeypatch.delenv("PYTEST_CURRENT_TEST")             # the child must not look like a test child
    server._current.clear()
    out = tmp_path / "run"
    r = server.run_topopt(str(y), str(out))
    assert "error" not in r, r
    try:
        rec = wait_for_job(out, timeout=120.0)
    finally:
        server._current["proc"].wait()
        server._current.clear()
    assert rec["host"]["started_by"] == "mcp" and END_KEYS <= set(rec) and rec["peak_rss_loop_bytes"] > 0
    assert has_estimates(out) and re.fullmatch(r"[0-9a-f]{64}", run_json(out)["problem_key"])


# --- readers: old folders and new folders ------------------------------------------------------------

OLD_JOB = {"format": 1, "state": "done", "pid": 4242, "process_start": 1792700000.0, "process_start_ticks": 123456, "boot_id": "b" * 36,
           "token": "t" * 32, "started_at": "2026-10-20T10:00:00+00:00", "run_dir": "x", "success_codes": [0], "platform": "linux", "notes": [],
           "peak_rss_start_bytes": 52_000_000, "peak_rss_source": "VmHWM", "exit_code": 0, "ended_at": "2026-10-20T10:02:00+00:00",
           "wall_s": 120.5, "peak_rss_bytes": 310_000_000}


@pytest.fixture
def old_folder(tmp_path, monkeypatch):
    """A run folder of the old key set: a `job.json` and a `run.json` of before DL1a."""
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    d = tmp_path / "runs" / "old"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps(dict(OLD_JOB, run_dir=str(d))))
    (d / "job.DONE").write_text("exit 0\n")
    (d / "status.json").write_text(json.dumps({"state": "done", "iteration": 3, "step": "done", "elapsed_s": 120.0, "message": ""}))
    (d / "run.json").write_text(json.dumps({"format": 1, "name": "old", "yaml_path": "x.yaml", "started_at": "2026-10-20T10:00:00+00:00",
                                            "iterations": 3, "solver": "pardiso", "derived": {}}))
    return d


def test_the_job_facts_of_an_old_record_read_as_before(old_folder):
    facts = job_facts(old_folder)
    assert facts["state"] == "done" and facts["wall_s"] == 120.5 and "peak memory 0.31 GB" in facts["summary"]
    assert "host" not in facts and "cpu_time_s" not in facts


def test_the_mcp_readers_take_an_old_folder_and_a_new_folder(old_folder, short, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    server._current.clear()
    st = server.run_status(str(old_folder))
    assert st["job"]["state"] == "done" and "host" not in st["job"] and st["state"] == "done"
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(short.dir))
    new = server.run_status(str(short.out))
    assert new["job"]["state"] == "done" and "host" not in new["job"] and new["job"]["cpu_time_s"] > 0
    names = {Path(r["run_dir"]).name for r in server.list_runs()["runs"]}
    assert "out" in names
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(old_folder.parent.parent))
    assert "old" in {Path(r["run_dir"]).name for r in server.list_runs()["runs"]}


def test_the_replay_and_the_lock_reader_take_the_new_record(short):
    from topopt.replay import Replay
    assert len(Replay(short.out).rows) == 4                       # passes 0 to 3
    rec = info(short.out)
    assert runjob.holder_state({"pid": rec["pid"], "process_start_ticks": rec.get("process_start_ticks"), "boot_id": rec.get("boot_id"),
                                "process_start": rec.get("process_start")}) in ("live", "dead", "reused")


# --- who started the job: the MCP server -------------------------------------------------------------------

@pytest.fixture
def server_started(monkeypatch):
    """This test process is an MCP server that has started: `server.main()` ran (its stdio loop is replaced). The server module only; the
    marker of a test (`PYTEST_CURRENT_TEST`) stays, a test removes it when it needs a process that is not a test."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv(hostinfo.STARTED_BY_ENV, "removed at the end of the test")   # so that monkeypatch removes what main() sets
    monkeypatch.delenv(hostinfo.STARTED_BY_ENV)
    monkeypatch.setattr(server.mcp, "run", lambda *a, **k: None)
    server.main()
    return server


def wait_for_job(d, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            rec = info(d)
            if rec.get("state") == "done":
                return rec
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    raise AssertionError(f"no finished job record in {d} after {timeout} s")


def test_the_mcp_server_says_mcp_to_the_run_process_it_starts(tmp_path, monkeypatch, server_started):
    server = server_started
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{shlex.quote(sys.executable)} {shlex.quote(str(PROBE))} {{out}}")
    monkeypatch.delenv("PYTEST_CURRENT_TEST")             # the child must not look like a test child
    server._current.clear()
    y = tmp_path / "x.yaml"
    y.write_text("name: x\n")
    out = tmp_path / "run"
    r = server.run_topopt(str(y), str(out))
    assert "error" not in r, r
    try:
        rec = wait_for_job(out)
    finally:
        server._current["proc"].wait()
        server._current.clear()
    assert rec["host"]["started_by"] == "mcp" and set(rec["host"]) == HOST_KEYS


def test_the_ratchet_process_of_the_mcp_server_gets_mcp_too(case, monkeypatch, server_started):
    server = server_started
    monkeypatch.setenv("TOPOPT_MCP_RATCHET_CMD", f"{shlex.quote(sys.executable)} {shlex.quote(str(PROBE))} {{root}}")
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    res = server.ratchet_run(run_dir=str(case.run), source=f"stl:{case.cands}", estimate_only=False)
    assert res["ok"], res
    rec = wait_for_job(case.root)
    assert rec["host"]["started_by"] == "mcp"


def test_a_job_that_runs_in_the_server_process_says_mcp_too(case, monkeypatch, server_started):
    """`ratchet_approve` runs the approval (`run_job`) in the server process, not in a child: the server's own environment says `mcp`."""
    server = server_started
    assert os.environ.get(hostinfo.STARTED_BY_ENV) == "mcp"             # the one line at the start of the server
    assert server.ratchet_run(run_dir=str(case.run), source=f"stl:{case.cands}", estimate_only=False, stop_before_promote=True)["ok"]
    end = time.time() + 60
    while server.ratchet_status(root=str(case.root)).get("state") != "waiting_for_owner":
        assert time.time() < end, server.ratchet_status(root=str(case.root))
        time.sleep(0.1)
    monkeypatch.delenv("PYTEST_CURRENT_TEST")             # the server is not a test process
    res = server.ratchet_approve(root=str(case.root), decline=True)
    assert res["ok"] and res["owner_approval"] == "declined", res
    rec = info(case.root / ratchet.APPROVE_JOB)
    assert rec["state"] == "done" and rec["pid"] == os.getpid()      # the job of this process
    assert rec["host"]["started_by"] == "mcp"


def test_a_child_of_a_test_is_a_test_run_even_from_the_mcp_server(tmp_path, monkeypatch, server_started):
    server = server_started
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{shlex.quote(sys.executable)} {shlex.quote(str(PROBE))} {{out}}")
    server._current.clear()
    y = tmp_path / "x.yaml"
    y.write_text("name: x\n")
    r = server.run_topopt(str(y), str(tmp_path / "run"))
    assert "error" not in r, r
    try:
        rec = wait_for_job(tmp_path / "run")
    finally:
        server._current["proc"].wait()
        server._current.clear()
    assert rec["host"]["started_by"] == "pytest"
