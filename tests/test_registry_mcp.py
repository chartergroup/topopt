"""Tests for the registry in the MCP server (P9-14): the entry of the server, `environment.other_servers` and the fact in `run_topopt`.

The registry folder of each test is the `TOPOPT_STATE_DIR` that `conftest.py` sets (a folder under tmp_path). The real per-user folder
of the machine is never read or written. Two tests start real servers over stdio.
"""
import asyncio
import json
import os
import signal
import stat
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import environment as topopt_environment  # noqa: E402
from topopt import registry, runjob, workfolder  # noqa: E402
from topopt_mcp import server  # noqa: E402

FAKE_CMD = f"{sys.executable} {REPO}/tests/serve_after_fake.py {{out}} --spi 0.05 --iterations 10"
HOLD_CMD = f"{sys.executable} {REPO}/tests/hold_job_fake.py {{out}}"      # holds the job lock until the file `release` is in the run folder


@pytest.fixture(autouse=True)
def fake_run(monkeypatch):
    """A fake run process instead of the real optimizer; no entry of this process before a test makes one."""
    monkeypatch.setenv("TOPOPT_MCP_CMD", FAKE_CMD)
    server._current.clear()
    monkeypatch.setattr(server, "_registration", None, raising=False)
    yield
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._current.clear()


@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("name: x\n")
    return p


@pytest.fixture
def sleeper():
    """A live process that is not this one: the other server, and the process of its job. The test ends it."""
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


def other_server(pid, tmp_path, run_dirs=(), name="other", run_pid=None, **kw):
    """The entry of another server (a process that is alive) in the registry folder of this test. `run_pid`: the run process that it started in each
    folder of `run_dirs` (the holder of the job lock there); a job counts for an entry only when that process holds its lock."""
    entry = registry.Registration(registry.KIND_MCP_SERVER, pid=pid, work_folder=str(tmp_path / f"{name}-work"),
                                  started_utc="2026-10-23T21:00:00Z", code_folder="/code", python="/py/bin/python", topopt_version="9.9.9", **kw)
    entry.start()
    for d in run_dirs:
        entry.add_run_dir(d, run_pid)
    return entry


def job_in(folder, pid, state="running", estimate_gib=None):
    """A run folder with a job record of the process `pid`, as `topopt run` leaves it."""
    folder.mkdir(parents=True, exist_ok=True)
    rec = {"format": 1, "pid": pid, **runjob.process_identity(pid), "token": "t", "started_at": "2020-01-01T00:00:00+00:00", "run_dir": str(folder)}
    info = dict(rec, state=state, success_codes=[0], notes=[])
    if state == "running":
        (folder / runjob.LOCK_FILE).write_text(json.dumps(rec))
    else:
        info.update(exit_code=0, wall_s=1.0, peak_rss_bytes=1)
        (folder / runjob.DONE_FILE).write_text("exit 0\n")
    (folder / runjob.INFO_FILE).write_text(json.dumps(info))
    if estimate_gib is not None:
        (folder / "run.json").write_text(json.dumps({"derived": {"memory_estimate_gib": estimate_gib}}))


def payload(res):
    data = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if data:
        return data.get("result", data) if set(data) == {"result"} else data
    return json.loads(res.content[0].text)


# --- environment.other_servers, in this process ---------------------------------------------------

def test_environment_has_an_empty_list_when_no_other_server_is_there():
    reply = server.environment()
    assert reply["ok"] is True and reply["other_servers"] == [] and "other_servers_note" not in reply


def test_environment_lists_a_live_server_with_the_job_state_of_its_run_folders(tmp_path, sleeper):
    running, done = tmp_path / "other-runs" / "running", tmp_path / "other-runs" / "done"
    job_in(running, sleeper.pid, "running", estimate_gib=5.5)
    job_in(done, sleeper.pid, "done")
    other_server(sleeper.pid, tmp_path, [running, done], run_pid=sleeper.pid)
    reply = server.environment()
    [item] = reply["other_servers"]
    assert item["pid"] == sleeper.pid and item["state"] == "live" and "note" not in item
    assert item["started_utc"] == "2026-10-23T21:00:00Z" and item["work_folder"] == str(tmp_path / "other-work")
    runs = {Path(r["run_dir"]).name: r for r in item["runs"]}
    assert runs["running"]["job_state"] == "running" and runs["running"]["memory_estimate_gib"] == 5.5 and runs["running"]["pid"] == sleeper.pid
    assert runs["done"]["job_state"] == "done" and runs["done"]["memory_estimate_gib"] is None
    assert "other_servers_note" not in reply


def test_environment_does_not_list_this_server(tmp_path, sleeper):
    mine = registry.Registration(registry.KIND_MCP_SERVER, work_folder=str(tmp_path), started_utc="t", code_folder="c", python="p", topopt_version="v")
    assert mine.start() is None
    other_server(sleeper.pid, tmp_path)
    assert [s["pid"] for s in server.environment()["other_servers"]] == [sleeper.pid]


def test_environment_does_not_list_a_dead_server_and_lists_the_unconfirmed_one_as_may_be_stale(tmp_path, sleeper, dead_pid, monkeypatch):
    other_server(dead_pid, tmp_path)
    unsure = other_server(sleeper.pid, tmp_path)
    for key in ("process_start_ticks", "boot_id"):
        unsure.record.pop(key, None)
    unsure.record["process_start"] = 1.0
    unsure.start()
    monkeypatch.setattr(runjob, "process_start_time", lambda pid: None)
    [item] = server.environment()["other_servers"]
    assert item["pid"] == sleeper.pid and item["state"] == "unconfirmed" and "may be stale" in item["note"]


def test_environment_lists_a_server_that_has_ended_while_its_job_goes_on_and_not_one_whose_jobs_have_ended(tmp_path, sleeper, dead_pid):
    running, done = tmp_path / "other-runs" / "running", tmp_path / "other-runs" / "done"
    job_in(running, sleeper.pid, "running", estimate_gib=5.5)
    job_in(done, sleeper.pid, "done")
    other_server(dead_pid, tmp_path, [done, running], name="gone", run_pid=sleeper.pid)
    reply = server.environment()
    [item] = reply["other_servers"]
    assert item["pid"] == dead_pid and item["state"] == "dead" and "no longer running" in item["note"]
    assert [r["job_state"] for r in item["runs"]] == ["done", "running"] and item["runs"][1]["memory_estimate_gib"] == 5.5
    assert (registry.servers_folder() / f"server-{dead_pid}.json").is_file()
    sleeper.kill()
    sleeper.wait()                                          # the job ends
    assert server.environment()["other_servers"] == []
    assert not (registry.servers_folder() / f"server-{dead_pid}.json").exists()


def test_environment_says_when_the_registry_is_off_for_this_server(tmp_path):
    blocked = tmp_path / "a-file"
    blocked.write_text("x")
    mine = registry.Registration(registry.KIND_MCP_SERVER, root=blocked / "state", work_folder="w", started_utc="t", code_folder="c", python="p",
                                 topopt_version="v")
    assert mine.start()
    server._registration = mine
    reply = server.environment()
    assert reply["ok"] is True and reply["other_servers"] == []
    assert "The registry of topopt servers is off" in reply["other_servers_note"]


def test_environment_gives_a_note_for_a_file_it_cannot_read(tmp_path, sleeper):
    other_server(sleeper.pid, tmp_path)
    (registry.state_root() / "servers" / "server-5.json").write_text("{")
    reply = server.environment()
    assert [s["pid"] for s in reply["other_servers"]] == [sleeper.pid] and "server-5.json" in reply["other_servers_note"]


# --- the entry of this server -------------------------------------------------------------------

def test_register_server_writes_the_entry_and_the_exit_function_removes_it(monkeypatch):
    exit_functions = []
    monkeypatch.setattr(server.atexit, "register", exit_functions.append)
    folder = workfolder.resolve()
    server._register_server(folder)
    path = registry.servers_folder() / f"server-{os.getpid()}.json"
    rec = json.loads(path.read_text())
    assert rec["pid"] == os.getpid() and rec["started_utc"] == server.STARTED_UTC and rec["work_folder"] == str(folder.path)
    assert rec["code_folder"] == str(server.REPO) and rec["python"] == sys.executable and rec["kind"] == "mcp_server" and rec["run_dirs"] == []
    import topopt
    assert rec["topopt_version"] == topopt.__version__
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert exit_functions == [server._registration.remove]
    exit_functions[0]()
    assert not path.exists()


def test_a_failed_entry_write_is_one_line_on_stderr_and_not_a_stop(tmp_path, monkeypatch, capsys):
    blocked = tmp_path / "a-file"
    blocked.write_text("x")
    monkeypatch.setenv("TOPOPT_STATE_DIR", str(blocked / "state"))
    monkeypatch.setattr(server.atexit, "register", lambda fn: None)
    server._register_server(workfolder.resolve())
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[topopt_mcp] The registry of topopt servers is off")
    assert server._registration is not None and server._registration.problem


def test_an_unexpected_error_while_the_entry_is_made_is_one_line_too(monkeypatch, capsys):
    def boom(self):
        raise ValueError("odd")
    monkeypatch.setattr(registry.Registration, "start", boom)
    monkeypatch.setattr(server.atexit, "register", lambda fn: None)
    server._register_server(workfolder.resolve())
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and "ValueError: odd" in err[0]


def test_a_run_start_adds_its_folder_to_the_entry_and_keeps_the_earlier_folders(yaml_file, tmp_path, monkeypatch):
    monkeypatch.setattr(server.atexit, "register", lambda fn: None)
    server._register_server(workfolder.resolve())
    path = registry.servers_folder() / f"server-{os.getpid()}.json"
    assert json.loads(path.read_text())["run_dirs"] == []
    first, second = tmp_path / "run1", tmp_path / "run2"
    started = server.run_topopt(str(yaml_file), str(first))
    assert "error" not in started
    rec = json.loads(path.read_text())
    assert rec["run_dirs"] == [str(first)]
    [process] = rec["run_processes"]                                  # the run process that this server started (P9-14 R2 M1)
    assert process["run_dir"] == str(first) and process["pid"] == started["pid"] and "process_start" in process
    stop = time.time() + 60
    while time.time() < stop and server.run_status(str(first))["state"] not in ("done", "error"):
        time.sleep(0.2)
    started_again = server.run_topopt(str(yaml_file), str(second))
    assert "error" not in started_again
    rec = json.loads(path.read_text())
    assert rec["run_dirs"] == [str(first), str(second)]
    assert [(p["run_dir"], p["pid"]) for p in rec["run_processes"]] == [(str(first), started["pid"]), (str(second), started_again["pid"])]


def test_a_refused_run_adds_no_folder(yaml_file, tmp_path, monkeypatch):
    monkeypatch.setattr(server.atexit, "register", lambda fn: None)
    server._register_server(workfolder.resolve())
    path = registry.servers_folder() / f"server-{os.getpid()}.json"
    assert "error" in server.run_topopt(str(tmp_path / "no-such.yaml"), str(tmp_path / "run1"))
    assert json.loads(path.read_text())["run_dirs"] == []


def test_a_server_that_did_not_make_an_entry_writes_nothing(yaml_file, tmp_path):
    """The tests of the server call `run_topopt` in this process. Without main() there is no entry, and the folder of the user stays as it is."""
    assert server._registration is None
    assert "error" not in server.run_topopt(str(yaml_file), str(tmp_path / "run1"))
    assert not registry.state_root().exists()


# --- run_topopt: the fact about a running job of another server -------------------------------------

def test_run_topopt_gives_a_fact_when_another_server_has_a_running_job_and_the_run_starts(yaml_file, tmp_path, sleeper, monkeypatch):
    other_run = tmp_path / "other-runs" / "r1"
    job_in(other_run, sleeper.pid, "running", estimate_gib=6.5)
    other_server(sleeper.pid, tmp_path, [other_run], run_pid=sleeper.pid)
    monkeypatch.setattr(topopt_environment, "_ram", lambda: {"total_bytes": 16 * 2**30, "available_bytes": 4 * 2**30})
    mine = tmp_path / "mine"
    reply = server.run_topopt(str(yaml_file), str(mine))
    assert "error" not in reply and isinstance(reply["pid"], int) and reply["run_dir"] == str(mine)        # the run starts
    fact = reply["other_server_jobs"]
    assert fact["fact"].startswith("A job that another topopt server started is running now:")
    for text in (f"pid {sleeper.pid} in {other_run}", f"started by server {sleeper.pid}", "memory estimate about 6.5 GiB",
                 "16.0 GiB of RAM and 4.0 GiB free now", "can freeze", "This run has started already"):
        assert text in fact["fact"], text
    [job] = fact["jobs"]
    assert job == {"server_pid": sleeper.pid, "server_state": "live", "work_folder": str(tmp_path / "other-work"), "run_dir": str(other_run),
                   "pid": sleeper.pid, "started_at": "2020-01-01T00:00:00+00:00", "memory_estimate_gib": 6.5}
    assert fact["ram"] == {"total_bytes": 16 * 2**30, "available_bytes": 4 * 2**30}
    options = " ".join(fact["options"])
    assert "wait" in options and "stop the other job" in options and "run both" in options
    assert server.run_status(str(mine))["state"] != "none"                       # the run is there


def test_the_fact_has_no_estimate_when_the_run_folder_has_none_and_names_an_unconfirmed_server(yaml_file, tmp_path, sleeper, monkeypatch):
    other_run = tmp_path / "other-runs" / "r1"
    job_in(other_run, sleeper.pid, "running")                                         # no run.json yet: a run that has just started
    unsure = other_server(sleeper.pid, tmp_path, [other_run], run_pid=sleeper.pid)
    for key in ("process_start_ticks", "boot_id"):
        unsure.record.pop(key, None)
    unsure.record["process_start"] = 1.0
    unsure.start()
    real = runjob.process_start_time
    monkeypatch.setattr(runjob, "process_start_time", lambda pid: None if pid == sleeper.pid else real(pid))
    reply = server.run_topopt(str(yaml_file), str(tmp_path / "mine"))
    fact = reply["other_server_jobs"]
    assert "memory estimate" not in fact["fact"] and "may be stale" in fact["fact"]
    assert fact["jobs"][0]["memory_estimate_gib"] is None and fact["jobs"][0]["server_state"] == "unconfirmed"


def test_the_fact_names_a_server_that_has_ended_and_says_that_it_is_no_longer_running(yaml_file, tmp_path, sleeper, dead_pid, monkeypatch):
    other_run = tmp_path / "other-runs" / "r1"
    job_in(other_run, sleeper.pid, "running", estimate_gib=6.5)
    other_server(dead_pid, tmp_path, [other_run], name="gone", run_pid=sleeper.pid)                        # the server has ended, its run goes on
    monkeypatch.setattr(topopt_environment, "_ram", lambda: {"total_bytes": 16 * 2**30, "available_bytes": 4 * 2**30})
    mine = tmp_path / "mine"
    reply = server.run_topopt(str(yaml_file), str(mine))
    assert "error" not in reply and isinstance(reply["pid"], int) and reply["run_dir"] == str(mine)        # the run starts
    fact = reply["other_server_jobs"]
    assert f"pid {sleeper.pid} in {other_run} (started by server {dead_pid} (no longer running), memory estimate about 6.5 GiB)" in fact["fact"]
    assert "may be stale" not in fact["fact"] and "can freeze" in fact["fact"]
    assert fact["jobs"][0]["server_pid"] == dead_pid and fact["jobs"][0]["server_state"] == "dead"
    assert "stop the other job" in " ".join(fact["options"])


def test_no_fact_when_the_other_job_has_ended_or_its_server_is_gone_or_nobody_else_is_there(yaml_file, tmp_path, sleeper, dead_pid):
    ended = tmp_path / "other-runs" / "ended"
    job_in(ended, sleeper.pid, "done")
    other_server(sleeper.pid, tmp_path, [ended], run_pid=sleeper.pid)                                      # a live server with no running job
    gone_run = tmp_path / "other-runs" / "gone"
    job_in(gone_run, sleeper.pid, "done")
    other_server(dead_pid, tmp_path, [gone_run], name="gone", run_pid=dead_pid)                         # a dead server, and its job has ended: not listed
    reply = server.run_topopt(str(yaml_file), str(tmp_path / "mine"))
    assert "error" not in reply and "other_server_jobs" not in reply


def test_a_server_does_not_report_its_own_run_as_the_job_of_another_server(yaml_file, tmp_path, monkeypatch):
    monkeypatch.setattr(server.atexit, "register", lambda fn: None)
    server._register_server(workfolder.resolve())
    first = tmp_path / "run1"
    assert "other_server_jobs" not in server.run_topopt(str(yaml_file), str(first))


def test_a_run_in_a_folder_that_another_server_also_names_is_the_job_of_this_server_only(yaml_file, tmp_path, sleeper, dead_pid, monkeypatch):
    """P9-14 R2 M1, through `run_topopt`. Another live server used the default run folder before and its job there has ended. This server starts a new
    run in the same folder. A third server reads: the job is listed once, under this server, and the run of the other server is not a running job."""
    monkeypatch.setenv("TOPOPT_MCP_CMD", HOLD_CMD)
    monkeypatch.setattr(server.atexit, "register", lambda fn: None)
    server._register_server(workfolder.resolve())
    shared = tmp_path / "work" / "runs" / "live"
    job_in(shared, dead_pid, "done")
    other_server(sleeper.pid, tmp_path, [shared], run_pid=dead_pid)
    reply = server.run_topopt(str(yaml_file), str(shared))
    assert "error" not in reply and "other_server_jobs" not in reply        # the job that is starting is not a job of the other server
    stop = time.time() + 60
    while time.time() < stop and registry.run_facts(shared)["job_state"] != "running":
        time.sleep(0.1)
    try:
        assert registry.run_facts(shared)["pid"] == reply["pid"]
        others = registry.read_servers(own_pid=0)                           # a third server
        [job] = registry.running_jobs(others.servers)
        assert job["server_pid"] == os.getpid() and job["pid"] == reply["pid"] and job["run_dir"] == str(shared)
        by_pid = {s["pid"]: s for s in others.servers}
        assert by_pid[sleeper.pid]["runs"][0]["job_state"] is None
        assert by_pid[os.getpid()]["runs"][0]["job_state"] == "running"
        assert registry.running_jobs(registry.read_servers(own_pid=os.getpid()).servers) == []      # this server does not see its own job as the job of the other
    finally:
        (shared / "release").write_text("x")
        server._current["proc"].wait(timeout=30)


def test_a_fault_while_the_fact_is_made_does_not_stop_the_run(yaml_file, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("odd")
    monkeypatch.setattr(registry, "read_servers", boom)
    reply = server.run_topopt(str(yaml_file), str(tmp_path / "mine"))
    assert "error" not in reply and "other_server_jobs" not in reply


# --- the texts of the two tools ----------------------------------------------------------------------

def test_the_description_of_environment_names_its_reply_key_and_the_one_of_run_topopt_is_not_changed_by_this_item():
    """`other_server_jobs` has `fact` and `options` and describes itself, so the description of `run_topopt` has no word of P9-14 (R1-3), and the
    budget of 4,096 characters is left to the other items."""
    tools = {t.name: t.description for t in asyncio.run(server.mcp.list_tools())}
    from topopt_mcp import replies
    assert "`other_servers`" in replies.guide("environment") and len(tools["environment"]) <= 4096    # #174: the key lines are in the guide
    assert "other_server" not in tools["run_topopt"] and len(tools["run_topopt"]) <= 4096


# --- real servers over stdio -------------------------------------------------------------------------

def params_for(tmp_path, name, state):
    from mcp.client.stdio import StdioServerParameters
    work = tmp_path / f"{name}-work"
    work.mkdir(exist_ok=True)
    env = dict(os.environ, TOPOPT_WORKSPACE=str(work), TOPOPT_STATE_DIR=str(state), TOPOPT_MCP_CMD=HOLD_CMD, PYTHONPATH=str(REPO))
    return StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], env=env, cwd=str(work))


def test_two_real_servers_see_each_other_and_not_themselves_and_one_leaves_cleanly(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    state = tmp_path / "shared-state"

    async def open_session(stack, name):
        errlog = stack.enter_context(open(tmp_path / f"{name}.stderr", "w"))
        r, w = await stack.enter_async_context(stdio_client(params_for(tmp_path, name, state), errlog=errlog))
        session = await stack.enter_async_context(ClientSession(r, w))
        await session.initialize()
        return session

    async def go():
        seen = {}
        async with AsyncExitStack() as outer:
            a = await open_session(outer, "a")
            async with AsyncExitStack() as inner:
                b = await open_session(inner, "b")
                seen["a"] = payload(await a.call_tool("environment", {}))
                seen["b"] = payload(await b.call_tool("environment", {}))
                seen["files"] = sorted(p.name for p in (state / "servers").iterdir())
                seen["mode"] = stat.S_IMODE((state / "servers" / f"server-{seen['b']['server']['pid']}.json").stat().st_mode)
                seen["dir_mode"] = (stat.S_IMODE(state.stat().st_mode), stat.S_IMODE((state / "servers").stat().st_mode))
            # B has closed its input and ended: a clean exit
            seen["b_file_after_exit"] = (state / "servers" / f"server-{seen['b']['server']['pid']}.json").exists()
            seen["a_after"] = payload(await a.call_tool("environment", {}))
        seen["a_file_after_exit"] = (state / "servers" / f"server-{seen['a']['server']['pid']}.json").exists()
        return seen

    seen = asyncio.run(asyncio.wait_for(go(), 300))
    pid_a, pid_b = seen["a"]["server"]["pid"], seen["b"]["server"]["pid"]
    assert pid_a != pid_b
    assert [s["pid"] for s in seen["a"]["other_servers"]] == [pid_b]               # each sees the other
    assert [s["pid"] for s in seen["b"]["other_servers"]] == [pid_a]               # and not itself
    [item] = seen["a"]["other_servers"]
    assert item["state"] == "live" and item["work_folder"] == seen["b"]["work_folder"]["path"] == str(tmp_path / "b-work")
    assert item["started_utc"] == seen["b"]["server"]["started_utc"] and item["runs"] == [] and item["kind"] == "mcp_server"
    assert item["python"] == sys.executable and item["code_folder"] == str(REPO)
    assert "other_servers_note" not in seen["a"] and "other_servers_note" not in seen["b"]
    assert seen["files"] == [f"server-{min(pid_a, pid_b)}.json", f"server-{max(pid_a, pid_b)}.json"]
    assert seen["mode"] == 0o600 and seen["dir_mode"] == (0o700, 0o700)
    assert seen["b_file_after_exit"] is False                                      # removed at the clean exit, not found dead by a reader
    assert seen["a_after"]["other_servers"] == []
    assert seen["a_file_after_exit"] is False


def test_a_run_goes_on_after_its_server_has_ended_and_another_server_still_knows_it(tmp_path, yaml_file):
    """Server A starts a run and ends cleanly. Its run process goes on. Server B lists A as `dead` with the running job, gives the fact in
    `run_topopt`, and deletes the file of A when the job has ended."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    state = tmp_path / "shared-state"
    run_a, run_b = tmp_path / "a-work" / "runs" / "ra", tmp_path / "b-work" / "runs" / "rb"

    def entry(pid):
        return state / "servers" / f"server-{pid}.json"

    async def open_session(stack, name):
        errlog = stack.enter_context(open(tmp_path / f"{name}.stderr", "w"))
        r, w = await stack.enter_async_context(stdio_client(params_for(tmp_path, name, state), errlog=errlog))
        session = await stack.enter_async_context(ClientSession(r, w))
        await session.initialize()
        return session

    async def wait_for(check, what):
        stop = time.time() + 120
        while time.time() < stop:
            if check():
                return
            await asyncio.sleep(0.1)
        raise AssertionError(what)

    async def go():
        seen = {}
        try:
            async with AsyncExitStack() as outer:
                b = await open_session(outer, "b")
                async with AsyncExitStack() as inner:
                    a = await open_session(inner, "a")
                    seen["pid_a"] = payload(await a.call_tool("environment", {}))["server"]["pid"]
                    seen["a_started"] = payload(await a.call_tool("run_topopt", {"yaml_path": str(yaml_file), "out_dir": str(run_a),
                                                                              "proceed_without_brief": True}))
                    await wait_for(lambda: (run_a / runjob.LOCK_FILE).exists(), "the run of A did not take the job lock")
                    seen["a_file_while_it_runs"] = json.loads(entry(seen["pid_a"]).read_text())
                # A has closed its input and ended: a clean exit. Its run process is in its own session and goes on.
                seen["a_file_after_exit"] = entry(seen["pid_a"]).is_file()
                seen["b_env"] = payload(await b.call_tool("environment", {}))
                seen["b_started"] = payload(await b.call_tool("run_topopt", {"yaml_path": str(yaml_file), "out_dir": str(run_b),
                                                                          "proceed_without_brief": True}))
                (run_a / "release").write_text("go\n")                                    # the job of A ends
                await wait_for(lambda: (run_a / runjob.DONE_FILE).exists() and not (run_a / runjob.LOCK_FILE).exists(), "the run of A did not end")
                seen["a_file_before_read"] = entry(seen["pid_a"]).is_file()
                seen["b_env_after"] = payload(await b.call_tool("environment", {}))
                seen["a_file_after_read"] = entry(seen["pid_a"]).is_file()
        finally:
            for folder in (run_a, run_b):                                                  # no process of this test stays alive
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "release").write_text("go\n")
        await wait_for(lambda: (run_b / runjob.DONE_FILE).exists() and not (run_b / runjob.LOCK_FILE).exists(), "the run of B did not end")
        return seen

    seen = asyncio.run(asyncio.wait_for(go(), 300))
    pid_a = seen["pid_a"]
    assert "error" not in seen["a_started"] and isinstance(seen["a_started"]["pid"], int)
    assert seen["a_file_while_it_runs"]["run_dirs"] == [str(run_a)]
    assert seen["a_file_after_exit"] is True                                            # the clean exit keeps the entry while the run goes on
    [item] = seen["b_env"]["other_servers"]
    assert item["pid"] == pid_a and item["state"] == "dead" and "no longer running" in item["note"]
    assert [(r["run_dir"], r["job_state"]) for r in item["runs"]] == [(str(run_a), "running")]
    assert "error" not in seen["b_started"]
    fact = seen["b_started"]["other_server_jobs"]
    assert f"started by server {pid_a} (no longer running)" in fact["fact"] and str(run_a) in fact["fact"]
    assert [(j["server_pid"], j["server_state"], j["run_dir"]) for j in fact["jobs"]] == [(pid_a, "dead", str(run_a))]
    assert seen["a_file_before_read"] is True                                           # the end of the job deletes nothing by itself
    assert seen["b_env_after"]["other_servers"] == [] and seen["a_file_after_read"] is False       # the next reader deletes the file


def test_a_real_server_goes_on_when_the_registry_folder_cannot_be_made(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    blocked = tmp_path / "a-file"
    blocked.write_text("not a folder")
    errfile = tmp_path / "server.stderr"

    async def go():
        with open(errfile, "w") as errlog:
            async with stdio_client(params_for(tmp_path, "a", blocked / "state"), errlog=errlog) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    return payload(await session.call_tool("environment", {}))

    reply = asyncio.run(asyncio.wait_for(go(), 300))
    assert reply["ok"] is True and reply["other_servers"] == []
    assert "The registry of topopt servers is off" in reply["other_servers_note"]
    lines = [ln for ln in errfile.read_text().splitlines() if "registry of topopt servers" in ln]
    assert len(lines) == 1 and lines[0].startswith("[topopt_mcp] The registry of topopt servers is off")


def test_a_test_server_with_the_default_environment_of_the_sdk_uses_the_test_folder(tmp_path):
    """`conftest.py` adds TOPOPT_STATE_DIR to the default environment of `stdio_client`. Many tests start a server without `env`: it must not write
    its entry in the registry folder of the user."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    state = Path(os.environ["TOPOPT_STATE_DIR"])
    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(REPO))

    async def go():
        async with stdio_client(params, errlog=open(tmp_path / "err", "w")) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                pid = payload(await session.call_tool("environment", {}))["server"]["pid"]
                return pid, (state / "servers" / f"server-{pid}.json").is_file()

    pid, in_test_folder = asyncio.run(asyncio.wait_for(go(), 300))
    assert in_test_folder and not (state / "servers" / f"server-{pid}.json").exists()      # it was there while the server ran, and it left with it
