"""Tests for the registry of live topopt servers (topopt/registry.py, CONTRACTS.md section 43; P9-14).

The folder rule on both platforms, the entry (write, rewrite, remove, modes, atomic write), the states of the other entries
(live, dead, reused, unconfirmed) and the job state of the run folders. Every test uses a folder under tmp_path.
"""
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import registry, runjob  # noqa: E402
from topopt.registry import Registration, read_servers, running_jobs, state_root  # noqa: E402

needs_proc = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc (Linux)")


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
def spawn():
    """A function that starts a live process that is not this one. The test does not end it: the fixture does."""
    procs = []

    def make():
        procs.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]))
        return procs[-1]
    yield make
    for p in procs:
        p.kill()
        p.wait()


@pytest.fixture
def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def reg(root, pid=None, **kw):
    """An entry for the process `pid` (default: this one) in the folder `root`."""
    facts = dict(work_folder="/Users/<user>/topopt", started_utc="2026-10-23T22:00:00Z", code_folder="/code", python="/py/bin/python",
                 topopt_version="9.9.9")
    facts.update(kw)
    return Registration(registry.KIND_MCP_SERVER, root=root, pid=pid, **facts)


def entry_file(root, pid):
    return Path(root) / "servers" / f"server-{pid}.json"


def put_job(folder, pid, state="running", **ident):
    """A run folder with a job record of the process `pid`: the lock and job.json (and the marker for a state that is not running)."""
    folder.mkdir(parents=True, exist_ok=True)
    rec = {"format": 1, "pid": pid, **runjob.process_identity(pid), "token": "t", "started_at": "2020-01-01T00:00:00+00:00",
           "run_dir": str(folder), **ident}
    info = dict(rec, state=state, success_codes=[0], notes=[])
    if state == "running":
        (folder / runjob.LOCK_FILE).write_text(json.dumps(rec))
    else:
        info.update(exit_code=0, wall_s=1.5, peak_rss_bytes=1_000_000_000)
        (folder / runjob.DONE_FILE).write_text("exit 0\n")
    (folder / runjob.INFO_FILE).write_text(json.dumps(info))


# --- the folder rule ---------------------------------------------------------------------------

def test_linux_folder_is_xdg_state_home_or_the_default(tmp_path):
    home = tmp_path / "home"
    assert state_root({}, "linux", home) == home / ".local" / "state" / "topopt"
    assert state_root({"XDG_STATE_HOME": str(tmp_path / "xdg")}, "linux", home) == tmp_path / "xdg" / "topopt"
    for not_valid in ("", "relative/state"):     # the XDG rule: an empty or relative value is not valid and counts as not set
        assert state_root({"XDG_STATE_HOME": not_valid}, "linux", home) == home / ".local" / "state" / "topopt"


def test_macos_folder_is_application_support_and_ignores_xdg(tmp_path):
    home = tmp_path / "home"
    expected = home / "Library" / "Application Support" / "topopt"
    assert state_root({}, "darwin", home) == expected
    assert state_root({"XDG_STATE_HOME": str(tmp_path / "xdg")}, "darwin", home) == expected     # one path for each platform


def passwd_home(monkeypatch, folder):
    """The account database of the test says that the home folder of this user is `folder`."""
    monkeypatch.setattr(registry.pwd, "getpwuid", lambda uid: type("Entry", (), {"pw_dir": str(folder)})())


def test_the_platform_default_is_sys_platform(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_STATE_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    passwd_home(monkeypatch, tmp_path / "h")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert state_root() == tmp_path / "h" / "Library" / "Application Support" / "topopt"
    monkeypatch.setattr(sys, "platform", "linux")
    assert state_root() == tmp_path / "h" / ".local" / "state" / "topopt"


def test_a_fake_home_variable_does_not_move_the_folder(monkeypatch, tmp_path):
    """`~` is the home folder of the account database, not `$HOME`: two processes of one user with different `HOME` see one folder."""
    passwd_home(monkeypatch, tmp_path / "real-home")
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    assert Path.home() == tmp_path / "fake-home"                 # the test is a real test: the variable does move Path.home()
    assert state_root({}, "linux") == tmp_path / "real-home" / ".local" / "state" / "topopt"
    assert state_root({}, "darwin") == tmp_path / "real-home" / "Library" / "Application Support" / "topopt"
    assert state_root({"TOPOPT_STATE_DIR": "~/state"}, "linux") == tmp_path / "real-home" / "state"
    assert state_root({"XDG_STATE_HOME": str(tmp_path / "xdg")}, "linux") == tmp_path / "xdg" / "topopt"      # the variables stay as agreed


def test_the_real_account_database_gives_the_home_folder(monkeypatch, tmp_path):
    import pwd
    try:
        expected = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        pytest.skip("this uid has no entry in the account database")
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    assert state_root({}, "linux") == expected / ".local" / "state" / "topopt"


def test_the_home_folder_falls_back_to_path_home_when_the_account_database_gives_none(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fallback")
    expected = tmp_path / "fallback" / ".local" / "state" / "topopt"

    def no_entry(uid):
        raise KeyError(f"getpwuid(): uid not found: {uid}")       # a container uid with no passwd entry
    monkeypatch.setattr(registry.pwd, "getpwuid", no_entry)
    assert state_root({}, "linux") == expected
    for folder in ("", "relative/home"):                          # an entry with no usable folder
        passwd_home(monkeypatch, folder)
        assert state_root({}, "linux") == expected
    monkeypatch.setattr(registry, "pwd", None)                    # no `pwd` module on this system
    assert state_root({}, "linux") == expected


def test_the_override_wins_on_both_platforms(tmp_path, monkeypatch):
    home = tmp_path / "home"
    over = tmp_path / "over"
    for platform in ("linux", "darwin"):
        assert state_root({"TOPOPT_STATE_DIR": str(over), "XDG_STATE_HOME": str(tmp_path / "xdg")}, platform, home) == over
    assert state_root({"TOPOPT_STATE_DIR": "~/state"}, "linux", home) == home / "state"
    assert state_root({"TOPOPT_STATE_DIR": "  "}, "linux", home) == home / ".local" / "state" / "topopt"     # blanks: not set
    monkeypatch.chdir(tmp_path)
    assert state_root({"TOPOPT_STATE_DIR": "here"}, "linux", home) == tmp_path / "here"                     # a relative path starts at the cwd


def start_with_no_home_folder(monkeypatch):
    """Remove `TOPOPT_STATE_DIR` and every way to a home folder, then start an entry: the registry is off, and the reader does not raise."""
    def no_home():
        raise RuntimeError("Could not determine home directory.")

    def no_entry(uid):
        raise KeyError(f"getpwuid(): uid not found: {uid}")
    monkeypatch.delenv("TOPOPT_STATE_DIR", raising=False)
    monkeypatch.setattr(registry.pwd, "getpwuid", no_entry)
    monkeypatch.setattr(Path, "home", no_home)
    with pytest.raises(registry.RegistryOff):
        state_root({}, "linux")
    r = Registration(registry.KIND_MCP_SERVER, work_folder="w", started_utc="t", code_folder="c", python="p", topopt_version="v")
    problem = r.start()
    assert problem and "registry of topopt servers is off" in problem and "home folder" in problem and r.path is None
    assert "cannot be read" in read_servers().notes[0]       # the reader does not raise


def test_no_home_folder_means_the_registry_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)       # with it set, the folder rule needs no home folder (P9-14 R2 B1)
    start_with_no_home_folder(monkeypatch)


XDG_PROBE = "TOPOPT_TEST_XDG_PROBE"


@pytest.mark.skipif(not os.environ.get(XDG_PROBE), reason="runs inside test_a_runner_that_sets_xdg_state_home_... only")
def test_xdg_probe_inner(monkeypatch):
    """The test above, WITHOUT its own removal of `XDG_STATE_HOME`: only `tests/conftest.py` can keep this test away from `$XDG_STATE_HOME`."""
    start_with_no_home_folder(monkeypatch)


def test_a_runner_that_sets_xdg_state_home_does_not_give_a_test_a_way_to_the_real_folder(tmp_path):
    """P9-14 R2 B1. A test that removes `TOPOPT_STATE_DIR` and starts an entry used to write in `$XDG_STATE_HOME/topopt` when the runner set the
    variable (the real folder of that user). The conftest fixture `state_folder` removes it for every test. Run the inner test in a pytest process
    with `XDG_STATE_HOME` set to a folder of this test: it passes, and the folder stays empty."""
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    env = {**os.environ, "XDG_STATE_HOME": str(xdg), XDG_PROBE: "1"}
    env.pop("TOPOPT_STATE_DIR", None)
    test = f"{Path(__file__)}::test_xdg_probe_inner"
    done = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--basetemp", str(tmp_path / "bt"), test],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and "1 passed" in done.stdout, done.stdout[-2000:] + done.stderr[-2000:]
    assert list(xdg.iterdir()) == []


def test_the_names_do_not_collide_with_the_calibration_file_of_gpu_check(tmp_path, monkeypatch):
    """`topopt gpu-check` writes `gpu-constants.json` at the root of the same `TOPOPT_STATE_DIR`: the registry uses `servers/` and the reserved `locks/`."""
    from topopt.gpu_mg import check
    monkeypatch.setenv("TOPOPT_STATE_DIR", str(tmp_path / "state"))
    assert check.state_dir() == state_root() == tmp_path / "state"       # one folder for both
    assert check.CALIBRATION_FILE not in (registry.SERVERS_DIR, registry.LOCKS_DIR) and not check.CALIBRATION_FILE.startswith("server-")
    r = reg(None)
    assert r.start() is None
    calibration = check.calibration_path()
    calibration.write_text("{}")                                          # the file that gpu-check keeps there
    assert read_servers(own_pid=os.getpid() + 1).notes == []             # the reader does not look at it
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == [check.CALIBRATION_FILE, registry.SERVERS_DIR]
    r.remove()
    assert calibration.read_text() == "{}"                                # the exit of a server leaves it


def test_a_folder_that_cannot_be_written_turns_the_registry_off_with_a_note(tmp_path):
    blocked = tmp_path / "a-file"
    blocked.write_text("not a folder")
    r = reg(blocked / "state")
    problem = r.start()
    assert problem and problem.startswith("The registry of topopt servers is off") and "Other servers do not see this server" in problem
    assert r.problem == problem and not (blocked / "state").exists()
    r.remove()                                                # no error
    assert r.add_run_dir(tmp_path / "run") == r.problem       # a later write is tried again, and fails the same way


@pytest.mark.parametrize("depth", [1, 2])
def test_the_note_names_the_root_folder_and_says_that_a_part_of_the_path_is_a_file(tmp_path, depth):
    """P9-14 R2 n2. A file where a folder must be: the note names the root folder, not only the OS text ("File exists" / "Not a directory")."""
    blocked = tmp_path / "a-file"
    blocked.write_text("not a folder")
    root = blocked.joinpath(*["sub"] * (depth - 1), "state")
    r = reg(root)
    problem = r.start()
    assert f"in {root}" in problem and "a part of the path is a file" in problem and "Other servers do not see this server" in problem, problem
    assert r.path is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a folder with mode 500")
def test_a_read_only_root_turns_the_registry_off(tmp_path):
    root = tmp_path / "state"
    root.mkdir(mode=0o500)
    try:
        r = reg(root)
        problem = r.start()
        assert problem and "registry of topopt servers is off" in problem and "Permission denied" in problem
        assert not (root / "servers").exists()
    finally:
        root.chmod(0o700)


def test_the_registry_comes_back_when_the_folder_can_be_written_again(tmp_path):
    root = tmp_path / "state"
    root.mkdir(mode=0o500)
    try:
        if os.geteuid() == 0:
            pytest.skip("root can write a folder with mode 500")
        r = reg(root)
        assert r.start() and r.problem
    finally:
        root.chmod(0o700)
    assert r.add_run_dir(tmp_path / "run") is None and r.problem is None and entry_file(root, os.getpid()).is_file()


# --- this server's entry: write, rewrite, remove ----------------------------------------------------

def test_start_writes_the_entry_with_the_agreed_keys(tmp_path):
    root = tmp_path / "state"
    r = reg(root)
    assert r.start() is None
    rec = json.loads(entry_file(root, os.getpid()).read_text())
    ident = runjob.process_identity(os.getpid())
    assert rec["format"] == 1 and rec["kind"] == "mcp_server" and rec["pid"] == os.getpid()
    assert {k: rec[k] for k in ident} == ident                       # process_start, and on Linux the ticks and the boot id
    assert rec["started_utc"] == "2026-10-23T22:00:00Z" and rec["work_folder"] == "/Users/<user>/topopt"
    assert rec["topopt_version"] == "9.9.9" and rec["code_folder"] == "/code" and rec["python"] == "/py/bin/python"
    assert rec["run_dirs"] == [] and rec["run_processes"] == []
    assert set(rec) == {"format", "kind", "pid", "started_utc", "work_folder", "topopt_version", "code_folder", "python", "run_dirs", "run_processes", *ident}


def test_the_modes_are_0700_for_folders_and_0600_for_files(tmp_path):
    old = os.umask(0)           # with a umask of 0 only an explicit mode gives 0700 and 0600
    try:
        root = tmp_path / "state"
        r = reg(root)
        assert r.start() is None and r.add_run_dir(tmp_path / "run") is None
    finally:
        os.umask(old)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700 and stat.S_IMODE((root / "servers").stat().st_mode) == 0o700
    assert stat.S_IMODE(entry_file(root, os.getpid()).stat().st_mode) == 0o600
    assert not list((root / "servers").glob("*.tmp"))


def test_a_strange_umask_does_not_change_the_modes(tmp_path):
    """A umask can only take bits away. With 0277 the mode of `mkdir` and of the new file would be too narrow to work: the explicit chmod sets them."""
    old = os.umask(0o277)
    try:
        root = tmp_path / "state"
        r = reg(root)
        assert r.start() is None and r.add_run_dir(tmp_path / "run") is None
    finally:
        os.umask(old)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700 and stat.S_IMODE((root / "servers").stat().st_mode) == 0o700
    assert stat.S_IMODE(entry_file(root, os.getpid()).stat().st_mode) == 0o600


def test_a_folder_that_was_there_is_not_changed(tmp_path):
    root = tmp_path / "state"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    assert reg(root).start() is None
    assert stat.S_IMODE(root.stat().st_mode) == 0o755 and stat.S_IMODE((root / "servers").stat().st_mode) == 0o700


def test_the_locks_folder_is_reserved_and_not_made(tmp_path):
    root = tmp_path / "state"
    r = reg(root)
    r.start()
    r.add_run_dir(tmp_path / "run")
    read_servers(root=root)
    assert registry.LOCKS_DIR == "locks" and registry.SERVERS_DIR == "servers"
    assert sorted(p.name for p in root.iterdir()) == ["servers"]


def test_every_run_folder_is_kept_in_start_order_and_the_file_is_rewritten(tmp_path):
    root = tmp_path / "state"
    r = reg(root)
    r.start()
    for name in ("a", "b", "a", "c"):
        assert r.add_run_dir(tmp_path / name) is None
    rec = json.loads(entry_file(root, os.getpid()).read_text())
    assert rec["run_dirs"] == [str(tmp_path / n) for n in ("a", "b", "c")]       # no duplicate, no cap, no drop of an old folder


def test_remove_deletes_the_entry_and_only_in_the_process_that_made_it(tmp_path, monkeypatch):
    root = tmp_path / "state"
    r = reg(root)
    r.start()
    real = os.getpid()
    with monkeypatch.context() as m:
        m.setattr(os, "getpid", lambda: real + 1)               # a forked child that runs the atexit functions of its parent
        r.remove()
    assert entry_file(root, real).is_file()
    r.remove()
    assert not entry_file(root, real).exists()
    r.remove()                                                  # twice: no error


def test_a_reused_pid_overwrites_the_stale_file_of_the_same_pid(tmp_path):
    root = tmp_path / "state"
    stale = entry_file(root, os.getpid())
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"format": 1, "kind": "mcp_server", "pid": os.getpid(), "work_folder": "/old", "run_dirs": ["/old/run"]}))
    reg(root).start()
    rec = json.loads(stale.read_text())
    assert rec["work_folder"] == "/Users/<user>/topopt" and rec["run_dirs"] == []


def test_the_write_is_atomic_a_failed_move_keeps_the_old_file_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    root = tmp_path / "state"
    r = reg(root)
    r.start()
    before = entry_file(root, os.getpid()).read_text()

    def fail(src, dst):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m:
        m.setattr(os, "replace", fail)
        problem = r.add_run_dir(tmp_path / "run")
    assert problem and "No space left on device" in problem
    assert entry_file(root, os.getpid()).read_text() == before
    assert not list((root / "servers").glob("*.tmp"))


def test_a_reader_never_sees_a_part_file_while_the_entry_is_rewritten(tmp_path):
    root = tmp_path / "state"
    r = reg(root)
    r.start()
    path = entry_file(root, os.getpid())
    stop = threading.Event()
    seen, errors = [], []

    def read():
        while not stop.is_set():
            try:
                seen.append(len(json.loads(path.read_text())["run_dirs"]))
            except (ValueError, KeyError, OSError) as exc:
                errors.append(repr(exc))

    t = threading.Thread(target=read)
    t.start()
    try:
        for n in range(150):
            r.add_run_dir(tmp_path / f"run-{n}")
    finally:
        stop.set()
        t.join()
    assert seen and not errors, errors[:3]
    assert len(json.loads(path.read_text())["run_dirs"]) == 150


def test_calls_in_several_threads_keep_every_folder_and_the_file_whole(tmp_path):
    root = tmp_path / "state"
    r = reg(root)
    r.start()
    errors = []

    def add(n):
        try:
            for k in range(25):
                assert r.add_run_dir(tmp_path / f"run-{n}-{k}") is None
        except Exception as exc:     # a thread must not hide its error
            errors.append(repr(exc))

    threads = [threading.Thread(target=add, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert len(json.loads(entry_file(root, os.getpid()).read_text())["run_dirs"]) == 150
    assert not list((root / "servers").glob("*.tmp"))


def test_write_atomic_mode_is_exact_and_a_leftover_temporary_file_does_not_widen_it(tmp_path):
    from topopt.runinfo import write_atomic
    target = tmp_path / "f.json"
    (tmp_path / "f.json.tmp").write_text("old")                  # a temporary file that an earlier crash left, with a wide mode
    (tmp_path / "f.json.tmp").chmod(0o666)
    old = os.umask(0)
    try:
        write_atomic(target, b"{}", mode=0o600)
    finally:
        os.umask(old)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600 and target.read_bytes() == b"{}" and not (tmp_path / "f.json.tmp").exists()
    write_atomic(tmp_path / "g.txt", b"x")                       # without mode: the default, as before
    assert (tmp_path / "g.txt").read_bytes() == b"x"


# --- the other entries and their states -----------------------------------------------------------------

def test_the_own_entry_is_not_listed(tmp_path):
    root = tmp_path / "state"
    reg(root).start()
    others = read_servers(root=root)
    assert others.servers == [] and others.notes == []
    assert [s["pid"] for s in read_servers(own_pid=os.getpid() + 1, root=root).servers] == [os.getpid()]


def test_a_live_server_is_listed_with_the_keys_of_its_entry(tmp_path, sleeper):
    root = tmp_path / "state"
    r = reg(root, pid=sleeper.pid, work_folder="/Users/<user>/other", started_utc="2026-10-23T21:00:00Z")
    r.start()
    others = read_servers(root=root)
    assert others.notes == []
    [item] = others.servers
    assert item["pid"] == sleeper.pid and item["state"] == "live" and "note" not in item
    assert item["kind"] == "mcp_server" and item["started_utc"] == "2026-10-23T21:00:00Z" and item["work_folder"] == "/Users/<user>/other"
    assert item["topopt_version"] == "9.9.9" and item["code_folder"] == "/code" and item["python"] == "/py/bin/python"
    assert item["runs"] == [] and entry_file(root, sleeper.pid).is_file()


def test_a_dead_pid_is_not_listed_and_its_file_is_deleted(tmp_path, dead_pid):
    root = tmp_path / "state"
    reg(root, pid=dead_pid).start()
    assert entry_file(root, dead_pid).is_file()
    assert read_servers(root=root).servers == []
    assert not entry_file(root, dead_pid).exists()


def test_a_reused_pid_is_not_listed_and_its_file_is_deleted(tmp_path, sleeper):
    """A live pid with another start time: the entry is of a server that ended, and another process has its pid now."""
    root = tmp_path / "state"
    r = reg(root, pid=sleeper.pid)
    r.record.update(process_start=r.record["process_start"] - 3600)
    if "process_start_ticks" in r.record:
        r.record.update(process_start_ticks=r.record["process_start_ticks"] - 99999)
    r.start()
    assert read_servers(root=root).servers == []
    assert not entry_file(root, sleeper.pid).exists()


@needs_proc
def test_a_pid_of_an_earlier_boot_is_reused(tmp_path, sleeper):
    root = tmp_path / "state"
    r = reg(root, pid=sleeper.pid)
    r.record["boot_id"] = "the-boot-before"
    r.start()
    assert read_servers(root=root).servers == []
    assert not entry_file(root, sleeper.pid).exists()


def test_the_unconfirmed_case_is_listed_as_may_be_stale_and_never_as_live(tmp_path, sleeper, monkeypatch):
    """The pid exists, the record has an epoch start only (macOS), and the system gives no start time (`ps` fails)."""
    root = tmp_path / "state"
    r = reg(root, pid=sleeper.pid)
    for key in ("process_start_ticks", "boot_id"):
        r.record.pop(key, None)
    r.record["process_start"] = 1.0
    r.start()
    monkeypatch.setattr(runjob, "process_start_time", lambda pid: None)
    [item] = read_servers(root=root).servers
    assert item["pid"] == sleeper.pid and item["state"] == "unconfirmed" and item["state"] != "live"
    assert "may be stale" in item["note"]
    assert entry_file(root, sleeper.pid).is_file()                 # nothing is deleted that is not confirmed dead or reused


def test_an_entry_with_no_start_time_is_unconfirmed(tmp_path, sleeper):
    root = tmp_path / "state"
    r = reg(root, pid=sleeper.pid)
    for key in ("process_start_ticks", "boot_id"):
        r.record.pop(key, None)
    r.record["process_start"] = None
    r.start()
    [item] = read_servers(root=root).servers
    assert item["state"] == "unconfirmed" and "may be stale" in item["note"]


def test_a_file_that_cannot_be_read_or_parsed_is_a_note_and_the_others_are_read(tmp_path, sleeper, dead_pid):
    root = tmp_path / "state"
    reg(root, pid=sleeper.pid).start()
    folder = root / "servers"
    # P9-14 R2 B2. Hand-made copies of the entry of a dead server, with names that no server writes (`server-{pid}.json` is the only name). int() reads
    # both names as the pid, and the pid in the file is the same, so only the exact name tells them from an entry: they are notes and stay.
    copy = json.dumps(reg(root, pid=dead_pid).record)
    arabic_indic = str(dead_pid).translate(str.maketrans("0123456789", "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"))
    odd_names = [f"server-0{dead_pid}.json", f"server-{arabic_indic}.json", f"server-00{dead_pid}.json"]
    for odd in odd_names:
        (folder / odd).write_text(copy)
    (folder / "server-7.json").write_text("{not json")
    (folder / "server-8.json").write_text("[1, 2]")
    (folder / "server-9.json").write_text(json.dumps({"format": 1, "pid": 10}))            # the pid is not the pid of the name
    (folder / "server-11.json").write_text(json.dumps({"format": 2, "pid": 11}))           # a format of a newer build
    (folder / "server-99999999999999999999999.json").write_text(json.dumps({"format": 1, "pid": 99999999999999999999999}))
    (folder / "server-12.json.tmp").write_text("part")                                      # not an entry
    (folder / "notes.txt").write_text("not an entry")
    (folder / "server-13.json").mkdir()                                                     # a folder with the name of an entry
    others = read_servers(root=root)
    assert [s["pid"] for s in others.servers] == [sleeper.pid]
    joined = " ".join(others.notes)
    for name in ("server-7.json", "server-8.json", "server-9.json", "server-11.json", "server-13.json"):
        assert name in joined, others.notes
    assert "server-12.json.tmp" not in joined and "notes.txt" not in joined
    for odd in odd_names:
        assert odd in joined and (folder / odd).exists(), others.notes
    assert all((folder / n).exists() for n in ("server-7.json", "server-8.json", "server-9.json", "server-11.json"))     # nothing else is deleted


def test_a_link_with_the_name_of_an_entry_is_a_note_and_is_neither_read_through_nor_deleted(tmp_path, sleeper, dead_pid):
    """P9-14 R2 n1. No server writes a link into `servers/`. The reader does not read through it (also not to a live entry) and does not delete it."""
    root = tmp_path / "state"
    reg(root, pid=sleeper.pid).start()
    folder = root / "servers"
    outside = tmp_path / "outside"
    outside.mkdir()
    dead_target = outside / "dead.json"
    dead_target.write_text(json.dumps(reg(root, pid=dead_pid).record))
    live_target = outside / "live.json"
    live_target.write_text(json.dumps(reg(root, pid=os.getppid()).record))
    (folder / f"server-{dead_pid}.json").symlink_to(dead_target)
    (folder / f"server-{os.getppid()}.json").symlink_to(live_target)
    others = read_servers(root=root, own_pid=os.getpid())
    assert [s["pid"] for s in others.servers] == [sleeper.pid]            # the link to a live entry is not listed
    joined = " ".join(others.notes)
    assert f"server-{dead_pid}.json" in joined and f"server-{os.getppid()}.json" in joined and "not a regular file" in joined, others.notes
    assert (folder / f"server-{dead_pid}.json").is_symlink() and (folder / f"server-{os.getppid()}.json").is_symlink()
    assert dead_target.exists() and live_target.exists()


def test_a_missing_folder_is_no_note_and_the_reader_makes_nothing(tmp_path):
    root = tmp_path / "state"
    others = read_servers(root=root)
    assert others.servers == [] and others.notes == [] and not root.exists()


def test_a_folder_that_cannot_be_listed_is_a_note(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    (root / "servers").write_text("a file, not a folder")
    others = read_servers(root=root)
    assert others.servers == [] and "cannot be read" in others.notes[0]


def test_a_file_that_a_new_server_replaced_is_kept(tmp_path, dead_pid, monkeypatch):
    """Between the read and the delete a new server with the same pid writes its entry: the new file stays."""
    root = tmp_path / "state"
    reg(root, pid=dead_pid).start()
    path = entry_file(root, dead_pid)
    real = registry.identity_state

    def replace_then_answer(rec):
        state = real(rec)
        fresh = path.with_name("fresh.tmp")
        fresh.write_text(json.dumps(dict(rec, work_folder="/new")))
        os.replace(fresh, path)
        return state

    monkeypatch.setattr(registry, "identity_state", replace_then_answer)
    assert read_servers(root=root).servers == []
    assert json.loads(path.read_text())["work_folder"] == "/new"


# --- a server that has ended while its job goes on (CONTRACTS.md 43.3, 43.4) -------------------------

def end_job(folder, how, sleeper):
    """The job of `folder` ends: its process dies, or it finishes and writes the marker (it removes the lock)."""
    if how == "killed":
        sleeper.kill()
        sleeper.wait()
    else:
        (folder / runjob.LOCK_FILE).unlink()
        put_job(folder, sleeper.pid, "done")


def test_remove_keeps_the_entry_while_a_run_process_that_this_server_started_is_there(tmp_path, sleeper):
    root = tmp_path / "state"
    running = tmp_path / "runs" / "running"
    put_job(running, sleeper.pid, "running")
    r = reg(root)
    r.start()
    r.add_run_dir(tmp_path / "runs" / "no-job-here")             # a folder with no process: it keeps nothing
    r.add_run_dir(running, sleeper.pid)
    r.remove()
    assert entry_file(root, os.getpid()).is_file()               # the run goes on after this server: the entry stays
    r.remove()
    assert entry_file(root, os.getpid()).is_file()
    sleeper.kill()
    sleeper.wait()                                               # the run process ends
    r.remove()
    assert not entry_file(root, os.getpid()).exists()


def test_a_job_in_a_folder_of_the_entry_does_not_keep_it_when_another_process_holds_it(tmp_path, sleeper):
    """P9-14 R2 M1. The run folder is named by the entry, but the live job there is a job of another process: the entry is not kept for it."""
    root = tmp_path / "state"
    folder = tmp_path / "runs" / "live"
    put_job(folder, sleeper.pid, "running")                      # a job that another server started
    r = reg(root)
    r.start()
    r.add_run_dir(folder)                                        # this server named the folder, but not that process
    r.remove()
    assert not entry_file(root, os.getpid()).exists()


def test_remove_deletes_the_entry_when_every_run_process_has_ended_and_keeps_it_when_unsure(tmp_path, sleeper, dead_pid):
    root = tmp_path / "state"
    done, stale = tmp_path / "runs" / "done", tmp_path / "runs" / "stale"
    put_job(done, dead_pid, "done")
    put_job(stale, dead_pid, "running")
    r = reg(root)
    r.start()
    r.add_run_dir(done, dead_pid)
    r.add_run_dir(stale, dead_pid)
    r.remove()
    assert not entry_file(root, os.getpid()).exists()
    reused = {"run_dir": str(done), "pid": sleeper.pid, **runjob.process_identity(sleeper.pid)}
    reused.update(process_start=reused["process_start"] - 3600)                      # a live pid with another start time: another process
    if "process_start_ticks" in reused:
        reused.update(process_start_ticks=reused["process_start_ticks"] - 99999)
    r.start()
    r.record["run_processes"].append(reused)
    r.start()
    r.remove()
    assert not entry_file(root, os.getpid()).exists()
    r.start()
    r.record["run_processes"].append({"run_dir": str(done), "pid": 2 ** 70})            # a pid that the system refuses: not sure, so the file stays
    r.remove()                                                                          # and nothing is raised
    assert entry_file(root, os.getpid()).is_file()


def test_a_run_process_that_is_there_before_it_takes_the_job_lock_keeps_the_entry(tmp_path, sleeper, dead_pid):
    """P9-14 R2 M1 (gap 1 of CONTRACTS 43.6, closed). The server ends just after the start of a run: the run process has no job lock yet, and the
    folder has no job file. The entry stays while the process is there (at the exit, and for the reader), and goes when the process has ended."""
    folder = tmp_path / "runs" / "starting"
    folder.mkdir(parents=True)
    own = reg(tmp_path / "own-state")                                  # at the clean exit of the server
    own.start()
    own.add_run_dir(folder, sleeper.pid)
    own.remove()
    assert entry_file(tmp_path / "own-state", os.getpid()).is_file()
    root = tmp_path / "state"                                           # for the reader: the server is dead
    gone = reg(root, pid=dead_pid)
    gone.start()
    gone.add_run_dir(folder, sleeper.pid)
    [item] = read_servers(root=root).servers
    assert item["pid"] == dead_pid and item["state"] == "dead" and "still there" in item["note"]
    assert item["runs"][0]["job_state"] is None and running_jobs([item]) == []          # no job yet: no fact of a running job
    assert entry_file(root, dead_pid).is_file()
    sleeper.kill()
    sleeper.wait()                                                      # the run process ends
    assert read_servers(root=root).servers == [] and not entry_file(root, dead_pid).exists()
    own.remove()
    assert not entry_file(tmp_path / "own-state", os.getpid()).exists()


@pytest.mark.parametrize("ended_server", ["dead", "reused"])
def test_a_server_that_has_ended_with_a_live_job_is_listed_with_its_state_and_its_file_is_kept(tmp_path, sleeper, dead_pid, ended_server):
    root = tmp_path / "state"
    folder = tmp_path / "runs" / "r1"
    put_job(folder, os.getpid(), "running")                      # a live process holds the lock (here: this test process)
    (folder / "run.json").write_text(json.dumps({"derived": {"memory_estimate_gib": 4.0}}))
    if ended_server == "dead":
        r = reg(root, pid=dead_pid, work_folder="/Users/<user>/gone")
    else:                                                        # a live pid with another start time
        r = reg(root, pid=sleeper.pid, work_folder="/Users/<user>/gone")
        r.record.update(process_start=r.record["process_start"] - 3600)
        if "process_start_ticks" in r.record:
            r.record.update(process_start_ticks=r.record["process_start_ticks"] - 99999)
    r.start()
    r.add_run_dir(folder, os.getpid())                           # the run process of this server: the holder of the lock
    for _ in range(2):                                           # the reader does not delete it, also not at the second read
        others = read_servers(root=root)
        assert others.notes == []
        [item] = others.servers
        assert item["pid"] == r.pid and item["state"] == ended_server and item["work_folder"] == "/Users/<user>/gone"
        assert "no longer running" in item["note"] and "still there" in item["note"] and "may be stale" not in item["note"]
        assert item["runs"][0]["job_state"] == "running" and item["runs"][0]["memory_estimate_gib"] == 4.0
        assert entry_file(root, r.pid).is_file()
    [job] = running_jobs(others.servers)
    assert job["server_pid"] == r.pid and job["server_state"] == ended_server and job["run_dir"] == str(folder)


def test_when_the_run_process_has_ended_the_next_reader_deletes_the_file_of_the_ended_server(tmp_path, sleeper, dead_pid):
    root = tmp_path / "state"
    folder = tmp_path / "runs" / "r1"
    put_job(folder, sleeper.pid, "running")
    r = reg(root, pid=dead_pid)
    r.start()
    r.add_run_dir(folder, sleeper.pid)
    assert [s["pid"] for s in read_servers(root=root).servers] == [dead_pid]
    sleeper.kill()
    sleeper.wait()
    assert read_servers(root=root).servers == []
    assert not entry_file(root, dead_pid).exists()


def test_a_finished_job_whose_process_is_still_there_keeps_the_entry_until_the_process_ends(tmp_path, sleeper, dead_pid):
    """The rule is about the process: a run that is done but whose process stays on (for example to serve the dashboard) keeps the file of the
    ended server, which is listed with the job state `done` and no fact of a running job. The file goes when the process ends."""
    root = tmp_path / "state"
    folder = tmp_path / "runs" / "r1"
    put_job(folder, sleeper.pid, "done")
    r = reg(root, pid=dead_pid)
    r.start()
    r.add_run_dir(folder, sleeper.pid)
    [item] = read_servers(root=root).servers
    assert item["runs"][0]["job_state"] == "done" and running_jobs([item]) == [] and entry_file(root, dead_pid).is_file()
    sleeper.kill()
    sleeper.wait()
    assert read_servers(root=root).servers == [] and not entry_file(root, dead_pid).exists()


def test_an_ended_server_is_deleted_when_no_run_process_is_there(tmp_path, sleeper, dead_pid):
    root = tmp_path / "state"
    done, stale, nothing = tmp_path / "runs" / "done", tmp_path / "runs" / "stale", tmp_path / "runs" / "empty"
    put_job(done, sleeper.pid, "done")
    put_job(stale, dead_pid, "running")
    nothing.mkdir()
    r = reg(root, pid=dead_pid)
    r.start()
    for folder in (done, stale, nothing, tmp_path / "runs" / "gone"):
        r.add_run_dir(folder, dead_pid)                          # the run processes have ended
    r.add_run_dir(done, 10_000_000)                            # above the largest pid: no such process
    r.record["run_dirs"] += ["relative/run", 42]                 # not read, so they cannot hold a job
    r.record["run_processes"] += ["text", {"run_dir": "x"}, {"pid": sleeper.pid}, {"run_dir": "x", "pid": True}, {"run_dir": 5, "pid": sleeper.pid}]   # not run processes
    r.start()
    assert read_servers(root=root).servers == []
    assert not entry_file(root, dead_pid).exists()


@pytest.mark.parametrize("live_job_is", ["first", "middle", "last"])
def test_one_live_run_process_among_several_keeps_the_entry_wherever_it_is(tmp_path, sleeper, dead_pid, live_job_is):
    root = tmp_path / "state"
    folders = [tmp_path / "runs" / "a", tmp_path / "runs" / "b", tmp_path / "runs" / "c"]
    live = folders["first middle last".split().index(live_job_is)]
    for folder in folders:
        put_job(folder, sleeper.pid if folder == live else dead_pid, "running" if folder == live else "done")
    r = reg(root, pid=dead_pid)
    r.start()
    for folder in folders:
        r.add_run_dir(folder, sleeper.pid if folder == live else dead_pid)
    [item] = read_servers(root=root).servers
    assert [run["job_state"] for run in item["runs"]] == ["running" if f == live else "done" for f in folders]
    assert entry_file(root, dead_pid).is_file()
    own = reg(tmp_path / "own-state")                              # the same rule at the clean exit of a server
    own.start()
    for folder in folders:
        own.add_run_dir(folder, sleeper.pid if folder == live else dead_pid)
    own.remove()
    assert entry_file(tmp_path / "own-state", os.getpid()).is_file()


# --- one run folder, several servers (CONTRACTS.md 43.3, 43.4; P9-14 R2 M1) -------------------------

def test_add_run_dir_names_the_run_process_with_its_identity_and_keeps_every_process(tmp_path, sleeper):
    root = tmp_path / "state"
    a, b = tmp_path / "a", tmp_path / "b"
    r = reg(root)
    r.start()
    r.add_run_dir(a, sleeper.pid)
    r.add_run_dir(a, sleeper.pid)                                 # the same process again: no duplicate
    r.add_run_dir(a)                                              # no pid: the folder only
    r.add_run_dir(b, os.getpid())
    rec = json.loads(entry_file(root, os.getpid()).read_text())
    assert rec["run_dirs"] == [str(a), str(b)]
    assert rec["run_processes"] == [{"run_dir": str(a), "pid": sleeper.pid, **runjob.process_identity(sleeper.pid)},
                                    {"run_dir": str(b), "pid": os.getpid(), **runjob.process_identity(os.getpid())}]
    assert {"process_start"} <= set(rec["run_processes"][0])      # an identity, and no run state
    assert not {"state", "job_state", "summary"} & set(rec["run_processes"][0])


def test_same_process_compares_the_pid_and_the_identity_keys_that_both_records_have():
    ticks = {"pid": 5, "process_start": 100.0, "process_start_ticks": 7, "boot_id": "b"}
    assert registry.same_process(ticks, dict(ticks))
    assert not registry.same_process(ticks, dict(ticks, pid=6))
    assert not registry.same_process(ticks, dict(ticks, process_start_ticks=8))                   # a pid that another process took
    assert not registry.same_process(ticks, dict(ticks, boot_id="c"))                              # a pid of an earlier boot
    assert registry.same_process(ticks, {"pid": 5, "process_start": 100.5})                        # only one record has ticks: the start times
    assert not registry.same_process(ticks, {"pid": 5, "process_start": 103.0})
    no_ticks = {"pid": 5, "process_start": 100.0}
    assert registry.same_process(no_ticks, {"pid": 5, "process_start": 100.0 + runjob.START_TOLERANCE_S})
    assert not registry.same_process(no_ticks, {"pid": 5, "process_start": 100.0 + runjob.START_TOLERANCE_S + 0.5})
    assert registry.same_process({"pid": 5}, {"pid": 5, "process_start": None})                    # nothing to compare: the pid decides
    assert not registry.same_process({"pid": "5"}, {"pid": "5"}) and not registry.same_process({}, {})


@pytest.mark.parametrize("server_a", ["ended", "live"])
def test_two_entries_name_one_folder_and_the_job_counts_only_for_the_server_that_started_it(tmp_path, dead_pid, spawn, server_a):
    """The probe `probe_reuse_folder.py` of the review. Server A started a run in the folder X, and it has ended. Server B starts a new job in X
    (the default run folder `runs/live` is the same for the chats of one user). The job of B must not be reported as a job of A."""
    root = tmp_path / "state"
    x = tmp_path / "work" / "runs" / "live"
    a_server = dead_pid if server_a == "ended" else spawn().pid
    b_server, b_run = spawn().pid, spawn()
    a = reg(root, pid=a_server, work_folder="/Users/<user>/chat-a")
    a.start()
    a.add_run_dir(x, dead_pid)                                    # the run of A has ended long ago
    b = reg(root, pid=b_server, work_folder="/Users/<user>/chat-b")
    b.start()
    b.add_run_dir(x, b_run.pid)
    put_job(x, b_run.pid, "running")                              # the job in X is the job of B
    # a third server (this process) reads
    others = read_servers(root=root)
    jobs = running_jobs(others.servers)
    assert [j["server_pid"] for j in jobs] == [b_server] and [j["pid"] for j in jobs] == [b_run.pid]           # once, under B
    by_pid = {s["pid"]: s for s in others.servers}
    assert by_pid[b_server]["state"] == "live" and by_pid[b_server]["runs"][0]["job_state"] == "running"
    if server_a == "ended":
        assert a_server not in by_pid and not entry_file(root, a_server).exists()                           # A has nothing that goes on
    else:
        run_of_a = by_pid[a_server]["runs"][0]
        assert run_of_a["job_state"] is None and run_of_a["pid"] is None and "not a job that this server started" in run_of_a["summary"]
    # server B reads: its own job is not the job of A
    assert running_jobs(read_servers(own_pid=b_server, root=root).servers) == []
    # the run of A has the same folder and a new job of A: it counts for A, and no longer for B
    a_run = spawn()
    a.add_run_dir(x, a_run.pid)
    put_job(x, a_run.pid, "running")
    assert [j["server_pid"] for j in running_jobs(read_servers(root=root).servers)] == [a_server]


# --- the job state of the run folders ------------------------------------------------------------

def test_the_runs_of_a_server_have_the_job_state_of_their_own_folder(tmp_path, sleeper):
    root = tmp_path / "state"
    running, done, empty = tmp_path / "runs" / "running", tmp_path / "runs" / "done", tmp_path / "runs" / "empty"
    put_job(running, sleeper.pid, "running")
    (running / "run.json").write_text(json.dumps({"derived": {"memory_estimate_gib": 7.5}}))
    put_job(done, sleeper.pid, "done")
    (done / "run.json").write_text(json.dumps({"derived": {"memory_estimate_gib": 3.0}}))
    empty.mkdir()
    r = reg(root, pid=sleeper.pid)                                       # the stand-in is the server and the process of every job here
    r.start()
    for folder in (running, done, empty, tmp_path / "runs" / "gone"):
        r.add_run_dir(folder, sleeper.pid)
    r.record["run_dirs"].append("relative/run")                      # not an absolute path: not read
    r.record["run_dirs"].append(42)                                  # not a text: not read
    r.start()
    [item] = read_servers(root=root).servers
    states = {Path(run["run_dir"]).name: run["job_state"] for run in item["runs"]}
    assert states == {"running": "running", "done": "done", "empty": None, "gone": None}
    by_name = {Path(run["run_dir"]).name: run for run in item["runs"]}
    assert by_name["running"]["pid"] == sleeper.pid and by_name["running"]["memory_estimate_gib"] == 7.5
    assert by_name["running"]["summary"].startswith("job: running")
    assert by_name["done"]["memory_estimate_gib"] is None and by_name["done"]["summary"].startswith("job: done")
    assert "no job record" in by_name["empty"]["summary"]
    assert [j["run_dir"] for j in running_jobs([item])] == [str(running)]


def test_the_estimate_of_an_earlier_run_in_the_same_folder_is_not_used(tmp_path, sleeper):
    """`run.json` older than the running job is the file of the run before: the new run has not written its own yet."""
    root = tmp_path / "state"
    folder = tmp_path / "runs" / "again"
    put_job(folder, sleeper.pid, "running", started_at="2025-01-01T00:00:00+00:00")
    (folder / "run.json").write_text(json.dumps({"derived": {"memory_estimate_gib": 9.0}}))
    old = 1_700_000_000                                                  # a run.json of November 2023
    os.utime(folder / "run.json", (old, old))
    r = reg(root, pid=sleeper.pid)
    r.start()
    r.add_run_dir(folder, sleeper.pid)
    [item] = read_servers(root=root).servers
    assert item["runs"][0]["job_state"] == "running" and item["runs"][0]["memory_estimate_gib"] is None
    new = 1_900_000_000                                                  # written after the job started
    os.utime(folder / "run.json", (new, new))
    assert read_servers(root=root).servers[0]["runs"][0]["memory_estimate_gib"] == 9.0


def test_a_job_of_a_dead_process_is_not_running(tmp_path, sleeper, dead_pid):
    """The state of a run is the state of its job lock. A server that lives can list a folder whose job has died."""
    root = tmp_path / "state"
    folder = tmp_path / "runs" / "stale"
    put_job(folder, dead_pid, "running")
    r = reg(root, pid=sleeper.pid)
    r.start()
    r.add_run_dir(folder, dead_pid)
    [item] = read_servers(root=root).servers
    assert item["runs"][0]["job_state"] == "stale_lock" and running_jobs([item]) == []


def test_running_jobs_names_the_server_and_its_state(tmp_path, sleeper, monkeypatch):
    root = tmp_path / "state"
    folder = tmp_path / "runs" / "r1"
    put_job(folder, sleeper.pid, "running")
    r = reg(root, pid=sleeper.pid, work_folder="/Users/<user>/w")
    r.start()
    r.add_run_dir(folder, sleeper.pid)
    [job] = running_jobs(read_servers(root=root).servers)
    assert job["server_pid"] == sleeper.pid and job["server_state"] == "live" and job["work_folder"] == "/Users/<user>/w"
    assert job["run_dir"] == str(folder) and job["job_state"] == "running"
