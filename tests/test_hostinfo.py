"""Tests for topopt/hostinfo.py: the facts of the machine and of the process for the job record (DL1a, CONTRACTS.md section 24.1).

Each helper is tested on this host and with its source made to fail (the value is None and a note says why). The macOS branches run on
any host: the tests change `sys.platform` and replace the one function that runs a command (`_command_output`).
"""
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import hostinfo  # noqa: E402

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux") or not Path("/proc/self/stat").exists(), reason="needs /proc (Linux)")


def with_notes(fn, *args, **kw):
    notes: list = []
    return fn(*args, notes=notes, **kw), notes


@pytest.fixture
def linux(tmp_path, monkeypatch):
    """This process takes the Linux branches of `hostinfo` on any host: `sys.platform` is linux, and the /proc files are files of the test.

    A test that sets only some of them, or none, does not run the same on a Mac (no /proc, `sysctl` and `ps` instead): DL1a review M2.
    """
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "stat").write_text("cpu  1 2 3\nprocs_running 3\nprocs_blocked 0\n")
    (proc / "cpuinfo").write_text("processor\t: 0\nmodel name\t: Test CPU @ 3.00GHz\n")
    (proc / "self" / "clear_refs").write_text("")
    monkeypatch.setattr(sys, "platform", "linux")
    for name, path in (("PROC_DIR", proc), ("PROC_STAT", proc / "stat"), ("PROC_CPUINFO", proc / "cpuinfo"),
                       ("PROC_CLEAR_REFS", proc / "self" / "clear_refs")):
        monkeypatch.setattr(hostinfo, name, path)
    return proc


# --- each helper on this host ------------------------------------------------------------------

def test_the_counts_and_the_model_of_this_host():
    (count, usable), notes = with_notes(hostinfo.cpu_counts)
    assert count == os.cpu_count() and 1 <= usable <= count and notes == []
    model, notes = with_notes(hostinfo.cpu_model)
    assert isinstance(model, str) and model and notes == []


def test_ram_total_and_available_are_the_numbers_of_the_environment_tool():
    (total, avail), notes = with_notes(hostinfo.ram)
    from topopt.environment import _ram
    assert total == _ram()["total_bytes"] > 0 and 0 < avail <= total and notes == []


def test_the_load_average_is_three_numbers():
    load, notes = with_notes(hostinfo.load_avg)
    assert len(load) == 3 and all(isinstance(v, float) and v >= 0 for v in load) and notes == []


@linux_only
def test_procs_running_counts_at_least_the_reader():
    n, notes = with_notes(hostinfo.procs_running)
    assert isinstance(n, int) and n >= 1 and notes == []


def test_the_thread_variables_are_the_six_of_the_libraries_and_none_when_unset(monkeypatch):
    from topopt.cli import THREAD_VARS as CAPPED      # the variables that `cap_blas_threads` reads: all are recorded
    for name in hostinfo.THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "4")
    env = hostinfo.thread_env()
    assert set(env) == {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "GOTO_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS"}
    assert set(CAPPED) <= set(env)
    assert env["OMP_NUM_THREADS"] == "3" and env["VECLIB_MAXIMUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] is None and env["NUMBA_NUM_THREADS"] is None and env["GOTO_NUM_THREADS"] is None


def test_the_versions_of_the_package_and_of_numpy_and_scipy():
    import numpy
    import scipy
    from topopt import __version__
    v, notes = with_notes(hostinfo.versions)
    assert set(v) == {"topopt", "git_sha", "numpy", "scipy", "pypardiso"}
    assert v["topopt"] == __version__ and v["numpy"] == numpy.__version__ and v["scipy"] == scipy.__version__
    assert v["pypardiso"] is None or isinstance(v["pypardiso"], str)
    assert v["git_sha"] is None or re.fullmatch(r"[0-9a-f]{40,64}", v["git_sha"])
    assert (v["git_sha"] is None) == (not (ROOT / ".git").exists()) or any(n.startswith("git_sha:") for n in notes)   # no SHA: no checkout, or a note


def test_a_package_that_is_not_installed_is_none_without_a_note():
    v, notes = with_notes(hostinfo.package_version, "no_such_package_dl1a")
    assert v is None and notes == []


def test_the_version_of_a_package_that_is_not_loaded_comes_from_the_metadata(monkeypatch):
    from importlib import metadata
    monkeypatch.delitem(sys.modules, "pypardiso", raising=False)
    monkeypatch.setattr(metadata, "version", lambda name: "9.8.7")
    assert hostinfo.package_version("pypardiso") == "9.8.7"


def git(cwd, *args):
    """Real git as the reference (a test only): no user config, no signing, a fixed author."""
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "LC_ALL": "C"}
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-C", str(cwd), *args], capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@needs_git
def test_the_git_sha_of_this_checkout_is_what_git_says():
    """This repo is a clone, or a linked worktree (`.git` is a file): the files give the same commit as `git rev-parse HEAD`."""
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    sha, notes = with_notes(hostinfo.git_sha)
    assert sha == git(ROOT, "rev-parse", "HEAD") and notes == []


@needs_git
def test_the_commit_from_the_files_equals_git_for_a_branch_a_detached_head_packed_refs_and_a_linked_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("a")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "one")
    first = git(repo, "rev-parse", "HEAD")
    assert hostinfo.head_commit(repo) == first                               # HEAD is a branch, a loose ref file
    (repo / "a.txt").write_text("b")
    git(repo, "commit", "-q", "-am", "two")
    assert hostinfo.head_commit(repo) == git(repo, "rev-parse", "HEAD") != first
    git(repo, "pack-refs", "--all")
    assert not (repo / ".git" / "refs" / "heads" / "main").exists()          # the ref is only in packed-refs now
    assert hostinfo.head_commit(repo) == git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "--detach", first)
    assert hostinfo.head_commit(repo) == first                               # a detached HEAD holds the commit itself
    git(repo, "checkout", "-q", "main")
    wt = tmp_path / "linked"
    git(repo, "worktree", "add", "-q", "-b", "side", str(wt), first)
    assert (wt / ".git").is_file() and hostinfo.head_commit(wt) == first == git(wt, "rev-parse", "HEAD")
    git(wt, "pack-refs", "--all")                                            # the branch of a linked worktree is in the common folder
    assert hostinfo.head_commit(wt) == first
    git(wt, "checkout", "-q", "--detach", "main")
    assert hostinfo.head_commit(wt) == git(repo, "rev-parse", "main")


def test_started_by_is_pytest_in_a_test_mcp_when_the_server_says_so_else_cli(monkeypatch):
    assert hostinfo.started_by() == "pytest"                       # this test is running under pytest
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert hostinfo.started_by() == "cli"
    monkeypatch.setenv(hostinfo.STARTED_BY_ENV, "mcp")
    assert hostinfo.started_by() == "mcp"
    monkeypatch.setenv(hostinfo.STARTED_BY_ENV, "something else")  # only the one value is a fact
    assert hostinfo.started_by() == "cli"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")                 # a test that starts the MCP server still makes a test run
    monkeypatch.setenv(hostinfo.STARTED_BY_ENV, "mcp")
    assert hostinfo.started_by() == "pytest"


def own_cpu():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


def spin(cpu_s):
    """Use `cpu_s` seconds of CPU in this process, counted as `getrusage` counts it (`time.process_time` can differ by a millisecond)."""
    end = own_cpu() + cpu_s
    while own_cpu() < end:
        sum(range(20_000))


def test_the_cpu_time_is_the_difference_to_the_start_and_covers_the_children(monkeypatch):
    monkeypatch.setattr(hostinfo, "procs_running", lambda notes=None: 3)           # this test is about the CPU time, and a Mac has no /proc/stat
    spin(0.3)                                                                      # work before the job: it is not in the CPU time
    subprocess.run([sys.executable, "-c", "sum(range(4_000_000))"], check=True)    # a child before the job
    start = hostinfo.usage()
    assert set(start) == {"cpu_s", "cpu_children_s", "ctx_voluntary", "ctx_involuntary"}
    spin(0.2)
    subprocess.run([sys.executable, "-c", "sum(range(4_000_000))"], check=True)    # a child that the process waits for, in the job
    notes: list = []
    end = hostinfo.end_facts(start, notes)
    assert notes == []
    assert end["cpu_time_children_s"] > 0 and end["cpu_time_s"] >= end["cpu_time_children_s"] + 0.2 - 0.002   # 0.002 s: the values are rounded to 1 ms
    whole = sum(u.ru_utime + u.ru_stime for u in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))
    assert end["cpu_time_s"] < whole - 0.3                                         # the 0.3 s of work before the start are not in it
    assert isinstance(end["ctx_switches_voluntary"], int) and isinstance(end["ctx_switches_involuntary"], int)
    assert end["ctx_switches_voluntary"] >= 0 and end["ctx_switches_involuntary"] >= 0
    assert len(end["load_avg_end"]) == 3
    assert set(end) == {"cpu_time_s", "cpu_time_children_s", "ctx_switches_voluntary", "ctx_switches_involuntary",
                        "load_avg_end", "procs_running_end"}


def test_a_start_that_was_not_read_gives_null_cpu_fields_and_a_note(monkeypatch):
    monkeypatch.setattr(hostinfo, "procs_running", lambda notes=None: 3)
    notes: list = []
    end = hostinfo.end_facts(None, notes)
    assert end["cpu_time_s"] is None and end["cpu_time_children_s"] is None and end["ctx_switches_voluntary"] is None
    assert len(notes) == 1 and notes[0].startswith("cpu_time:") and len(end["load_avg_end"]) == 3


@linux_only
def test_the_reset_lowers_the_peak_memory_of_the_process_to_its_memory_now():
    def hwm():
        return next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:")) * 1024
    notes: list = []
    done = hostinfo.reset_peak_rss(notes)           # first, so that the peak of the earlier tests of this process is not in the numbers
    if notes:                                       # only a write that the system refused (a note names it) is a reason to skip
        pytest.skip(f"this /proc does not allow the write: {notes}")
    assert done is True                             # a silent False is a failure of the reset, not a skip
    base = hwm()
    block = bytearray(150 * 2 ** 20)
    block[::4096] = b"x" * len(block[::4096])       # touch every page, so that the memory is really used
    high = hwm()
    del block
    assert high > base + 100 * 2 ** 20 and hwm() >= high - 10 * 2 ** 20     # freeing does not lower the peak (VmHWM can read 1.5 MB lower: PR #34)
    assert hostinfo.reset_peak_rss(notes) is True and notes == []
    assert hwm() < high - 100 * 2 ** 20                                # the reset does


# --- a source that fails: the value is None and a note says why --------------------------------

def test_a_cpuinfo_that_is_missing_or_has_no_model_line(tmp_path, monkeypatch, linux):
    monkeypatch.setattr(hostinfo, "PROC_CPUINFO", tmp_path / "nothing")
    model, notes = with_notes(hostinfo.cpu_model)
    assert model is None and len(notes) == 1 and notes[0].startswith("cpu_model: FileNotFoundError")
    arm = tmp_path / "cpuinfo"
    arm.write_text("processor\t: 0\nCPU part\t: 0xd0c\n")           # an arm64 Linux file has no `model name`
    monkeypatch.setattr(hostinfo, "PROC_CPUINFO", arm)
    model, notes = with_notes(hostinfo.cpu_model)
    assert model is None and notes[0].startswith("cpu_model: LookupError") and "model name" in notes[0]


def test_a_cpu_count_that_is_none_has_a_note_and_one_that_raises_has_only_its_own(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)       # `os.cpu_count()` gives None, it does not raise, when the system does not tell
    (count, usable), notes = with_notes(hostinfo.cpu_counts)
    assert count is None and notes == ["cpu_count: os.cpu_count() gives no number"] and usable >= 1

    def boom():
        raise RuntimeError("no cores today")
    monkeypatch.setattr(os, "cpu_count", boom)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)      # Linux: the usable count does not use os.cpu_count
    monkeypatch.setattr("topopt.cli.cgroup_limits", lambda *a, **k: {"cpu_cores": None})     # and no CPU quota lowers it
    (count, usable), notes = with_notes(hostinfo.cpu_counts)
    assert count is None and usable == 2 and [n.split(":")[0] for n in notes] == ["cpu_count"] and "no cores today" in notes[0]


def test_without_sched_getaffinity_the_usable_count_reads_os_cpu_count_and_both_notes_show(monkeypatch):
    """macOS has no `os.sched_getaffinity`: `cli.host_cores` then asks `os.cpu_count`, so a failure of it shows in both numbers."""
    def boom():
        raise RuntimeError("no cores today")
    monkeypatch.setattr(os, "cpu_count", boom)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    (count, usable), notes = with_notes(hostinfo.cpu_counts)
    assert count is None and usable is None and [n.split(":")[0] for n in notes] == ["cpu_count", "cpu_usable"]


def test_a_cpu_count_and_a_usable_count_that_fail(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no cores today")
    monkeypatch.setattr(os, "cpu_count", boom)
    monkeypatch.setattr("topopt.cli.usable_cores", boom)
    (count, usable), notes = with_notes(hostinfo.cpu_counts)
    assert count is None and usable is None
    assert [n.split(":")[0] for n in notes] == ["cpu_count", "cpu_usable"] and "no cores today" in notes[0]


def test_ram_that_fails_or_has_no_available_number(monkeypatch):
    import topopt.environment as env

    def boom():
        raise OSError("no meminfo")
    monkeypatch.setattr(env, "_ram", boom)
    (total, avail), notes = with_notes(hostinfo.ram)
    assert total is None and avail is None and notes == ["ram: OSError: no meminfo"]
    monkeypatch.setattr(env, "_ram", lambda: {"total_bytes": 8 * 2 ** 30, "available_bytes": None})
    (total, avail), notes = with_notes(hostinfo.ram)
    assert total == 8 * 2 ** 30 and avail is None and notes == ["ram_available: the system gives no available RAM"]
    monkeypatch.setattr(env, "_ram", lambda: {"total_bytes": None, "available_bytes": 2 ** 30})
    (total, avail), notes = with_notes(hostinfo.ram)
    assert total is None and avail == 2 ** 30 and notes == ["ram_total: the system gives no total RAM"]


def test_a_load_average_that_fails(monkeypatch):
    def boom():
        raise OSError("no loadavg")
    monkeypatch.setattr(os, "getloadavg", boom)
    load, notes = with_notes(hostinfo.load_avg)
    assert load is None and notes == ["load_avg: OSError: no loadavg"]


def test_a_proc_stat_that_is_missing_or_has_no_procs_running_line(tmp_path, monkeypatch, linux):
    monkeypatch.setattr(hostinfo, "PROC_STAT", tmp_path / "nothing")
    n, notes = with_notes(hostinfo.procs_running)
    assert n is None and notes[0].startswith("procs_running: FileNotFoundError")
    short = tmp_path / "stat"
    short.write_text("cpu  1 2 3\nbtime 1\n")
    monkeypatch.setattr(hostinfo, "PROC_STAT", short)
    n, notes = with_notes(hostinfo.procs_running)
    assert n is None and notes[0].startswith("procs_running: LookupError")


SHA_A, SHA_B, SHA_C = "a" * 40, "b" * 40, "c" * 64


def fake_repo(tmp_path, head, refs=None, packed=None):
    """A `.git` folder made by hand. `refs`: `{ref: text}` as loose files. `packed`: the text of `packed-refs`, or None."""
    g = tmp_path / ".git"
    g.mkdir()
    (g / "HEAD").write_text(head)
    for ref, text in (refs or {}).items():
        (g / ref).parent.mkdir(parents=True, exist_ok=True)
        (g / ref).write_text(text)
    if packed is not None:
        (g / "packed-refs").write_text(packed)
    return g


def test_the_commit_is_read_from_the_files_of_each_layout(tmp_path):
    def folder(name):
        d = tmp_path / name
        d.mkdir()
        return d
    loose, det, packed, both = folder("loose"), folder("det"), folder("packed"), folder("both")
    fake_repo(loose, "ref: refs/heads/main\n", {"refs/heads/main": SHA_A + "\n"})
    assert hostinfo.head_commit(loose) == SHA_A
    fake_repo(det, SHA_C + "\n")                                                # detached HEAD, a SHA-256 repository
    assert hostinfo.head_commit(det) == SHA_C
    text = f"# pack-refs with: peeled fully-peeled sorted \n{SHA_B} refs/heads/other\n{SHA_A} refs/heads/main\n^{SHA_C}\n{SHA_C} refs/tags/v1\n"
    fake_repo(packed, "ref: refs/heads/main\n", packed=text)
    assert hostinfo.head_commit(packed) == SHA_A                                 # only in packed-refs; a comment and a peeled line are skipped
    fake_repo(both, "ref: refs/heads/main\n", {"refs/heads/main": SHA_B + "\n"}, packed=f"{SHA_A} refs/heads/main\n")
    assert hostinfo.head_commit(both) == SHA_B                                   # a loose ref is newer than packed-refs


def test_the_commit_of_a_linked_worktree_and_of_a_gitdir_file_with_a_relative_path(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    fake_repo(main, "ref: refs/heads/main\n", {"refs/heads/side": SHA_B + "\n"}, packed=f"{SHA_A} refs/heads/main\n")
    admin = main / ".git" / "worktrees" / "w"                                    # what `git worktree add` makes
    admin.mkdir(parents=True)
    (admin / "HEAD").write_text("ref: refs/heads/side\n")
    (admin / "commondir").write_text("../..\n")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {admin}\n")                          # an absolute path, as git writes it
    assert hostinfo.head_commit(linked) == SHA_B                                 # the branch is in the common folder
    (admin / "HEAD").write_text("ref: refs/heads/main\n")
    assert hostinfo.head_commit(linked) == SHA_A                                 # the common packed-refs
    (admin / "refs" / "worktree").mkdir(parents=True)
    (admin / "refs" / "worktree" / "x").write_text(SHA_C + "\n")
    (admin / "HEAD").write_text("ref: refs/worktree/x\n")
    assert hostinfo.head_commit(linked) == SHA_C                                 # a ref of this worktree only
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".git").write_text("gitdir: ../main/.git\n")                         # a relative path is relative to the folder of the file
    (main / ".git" / "HEAD").write_text("ref: refs/heads/side\n")
    assert hostinfo.head_commit(sub) == SHA_B


def test_a_folder_that_is_not_a_checkout_is_none_without_a_note(tmp_path):
    sha, notes = with_notes(hostinfo.git_sha, root=tmp_path)
    assert sha is None and notes == []                                           # an installed wheel: a known state, as Low Power Mode on Linux


@pytest.mark.parametrize("name, make, error", [
    ("no HEAD file", lambda t: (t / ".git").mkdir(), "FileNotFoundError"),
    ("HEAD that is not a commit or a ref", lambda t: fake_repo(t, "garbage\n"), "ValueError"),
    ("a ref that is nowhere", lambda t: fake_repo(t, "ref: refs/heads/gone\n", packed=f"{SHA_A} refs/heads/main\n"), "LookupError"),
    ("a ref that is nowhere, no packed-refs", lambda t: fake_repo(t, "ref: refs/heads/gone\n"), "LookupError"),
    ("a loose ref with no commit", lambda t: fake_repo(t, "ref: refs/heads/main\n", {"refs/heads/main": "not a hash\n"}), "ValueError"),
    ("a .git file that is not a gitdir line", lambda t: (t / ".git").write_text("hello\n"), "ValueError"),
    ("a .git file that points nowhere", lambda t: (t / ".git").write_text(f"gitdir: {t / 'nowhere'}\n"), "FileNotFoundError"),
])
def test_a_git_folder_that_cannot_be_read_is_none_with_a_note(tmp_path, name, make, error):
    make(tmp_path)
    sha, notes = with_notes(hostinfo.git_sha, root=tmp_path)
    assert sha is None and len(notes) == 1 and notes[0].startswith(f"git_sha: {error}"), (name, notes)


def test_the_git_sha_runs_no_command(monkeypatch):
    """The start of a job does not wait for git: no process starts (no timeout, no `xcode-select` on a Mac)."""
    def no_process(*a, **k):
        raise AssertionError("a command was started")
    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(hostinfo, "_command_output", no_process)
    sha, notes = with_notes(hostinfo.git_sha)
    assert notes == [] and (sha is None) == (not (ROOT / ".git").exists())


def test_a_package_version_that_fails_is_none_with_a_note(monkeypatch):
    from importlib import metadata

    def boom(name):
        raise ValueError("broken metadata")
    monkeypatch.delitem(sys.modules, "pypardiso", raising=False)
    monkeypatch.setattr(metadata, "version", boom)
    v, notes = with_notes(hostinfo.package_version, "pypardiso")
    assert v is None and notes == ["version_pypardiso: ValueError: broken metadata"]


def test_getrusage_that_fails_gives_null_cpu_fields_and_a_note(monkeypatch, linux):
    start = hostinfo.usage()
    def boom(who):
        raise OSError("no rusage")
    monkeypatch.setattr(resource, "getrusage", boom)
    notes: list = []
    assert hostinfo.usage(notes) is None and notes == ["getrusage: OSError: no rusage"]
    notes = []
    end = hostinfo.end_facts(start, notes)
    assert end["cpu_time_s"] is None and end["cpu_time_children_s"] is None and end["ctx_switches_voluntary"] is None
    assert notes == ["getrusage: OSError: no rusage"] and len(end["load_avg_end"]) == 3     # the other fields are still read


def test_a_reset_that_cannot_write_is_false_with_a_note(tmp_path, monkeypatch, linux):
    monkeypatch.setattr(hostinfo, "PROC_CLEAR_REFS", tmp_path / "no_such_folder" / "clear_refs")
    ok, notes = with_notes(hostinfo.reset_peak_rss)
    assert ok is False and len(notes) == 1 and notes[0].startswith("peak_rss_reset: FileNotFoundError")


def test_the_start_facts_have_every_key_when_every_source_fails(tmp_path, monkeypatch, linux):
    def boom(*a, **k):
        raise OSError("source is gone")
    monkeypatch.setattr(hostinfo, "PROC_DIR", tmp_path / "no_proc")           # cpuinfo, stat, the process list
    monkeypatch.setattr(hostinfo, "PROC_CPUINFO", tmp_path / "no_proc" / "cpuinfo")
    monkeypatch.setattr(hostinfo, "PROC_STAT", tmp_path / "no_proc" / "stat")
    monkeypatch.setattr(os, "getloadavg", boom)
    monkeypatch.setattr("topopt.environment._ram", boom)
    monkeypatch.setattr("topopt.cli.usable_cores", boom)
    f = hostinfo.start_facts()
    assert set(f) == {"started_by", "threads", "cpu_count", "cpu_usable", "cpu_model", "ram_total_bytes", "ram_available_bytes", "load_avg",
                      "procs_running", "other_jobs", "version", "low_power_mode", "notes"}
    assert f["cpu_usable"] is None and f["cpu_model"] is None and f["ram_total_bytes"] is None and f["load_avg"] is None
    assert f["procs_running"] is None and f["other_jobs"] is None
    said = {n.split(":")[0] for n in f["notes"]}
    assert {"cpu_usable", "cpu_model", "ram", "load_avg", "procs_running", "other_jobs"} <= said
    json.dumps(f)                                                              # the block is JSON


# --- the other topopt processes ------------------------------------------------------------------

@pytest.mark.parametrize("argv, kind", [
    (["/venv/bin/python", "-m", "topopt", "run", "x.yaml", "--out", "o"], "run"),
    (["python3.12", "-m", "topopt", "verify", "runs/a"], "verify"),
    (["python", "-m", "topopt", "ratchet", "run"], "other"),
    (["/venv/bin/python", "-m", "topopt.ratchet", "run", "/w/runs/a", "--source", "stl:/w/cands", "--root", "/w/ratchet"], "other"),   # the real command of ratchet_run (topopt_mcp/server.py)
    (["python", "-m", "topopt.ratchet", "run"], "other"),
    (["python", "-m", "topopt.verify.memory", "run"], "other"),
    (["python", "-m", "topopt.__main__", "run", "x.yaml"], "run"),
    (["python", "-m", "topopt", "dashboard", "runs/a"], "other"),
    (["python", "-m", "topopt"], "other"),
    (["python", "-m", "topopt.cli", "run", "x.yaml"], "run"),
    (["python", "-mtopopt", "run", "x.yaml"], "run"),
    (["python", "-u", "-X", "dev", "-m", "topopt", "run", "x.yaml"], "run"),
    (["python", "-W", "ignore", "-m", "topopt", "verify", "d"], "verify"),
    (["python", "-m", "topopt", "--quiet", "run", "x.yaml"], "run"),
    (["python", "-m", "topopt_mcp.server"], "mcp_server"),
    (["python", "-m", "topopt_mcp"], "mcp_server"),
    (["/venv/bin/python", "/venv/bin/topopt", "run", "x.yaml"], "run"),
    (["/opt/homebrew/Cellar/python@3.12/3.12.4/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python",
      "-m", "topopt", "run", "x.yaml"], "run"),                  # macOS: a framework Python shows its launcher `Python`, with a capital letter
    (["/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python", "-m", "topopt_mcp.server"], "mcp_server"),
    (["Python3", "-m", "pytest"], None),
    (["/venv/bin/python", "/venv/bin/topopt-mcp"], "mcp_server"),
    (["/venv/bin/python", "/src/topopt_mcp/server.py"], "mcp_server"),
    (["/venv/bin/topopt", "verify", "d"], "verify"),
    (["topopt-mcp"], "mcp_server"),
    (["vim", "topopt/cli.py"], None),
    (["rg", "topopt", "run"], None),
    (["ls", "/home/me/topopt", "run"], None),
    (["bash", "-c", "topopt run x.yaml"], None),
    (["python", "-c", "import topopt"], None),
    (["python", "-m", "pytest", "tests/test_topopt.py"], None),
    (["python", "tools/bench_gpu_mg.py", "topopt", "run"], None),
    (["python"], None),
    ([], None),
])
def test_the_kind_of_a_process_comes_from_its_arguments(argv, kind):
    assert hostinfo.classify_command(argv) == kind


def fake_proc(tmp_path, rows):
    """A /proc folder with `<pid>/cmdline` files. `rows`: `{pid: argv or bytes}`."""
    proc = tmp_path / "fake_proc"
    proc.mkdir()
    for pid, argv in rows.items():
        d = proc / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(argv if isinstance(argv, bytes) else b"\0".join(a.encode() for a in argv) + b"\0")
    return proc


def test_the_scan_of_proc_counts_other_topopt_processes_by_pid_and_kind_only(tmp_path, monkeypatch, linux):
    proc = fake_proc(tmp_path, {
        101: ["python", "-m", "topopt", "run", "secret_part.yaml", "--out", "private_run"],
        102: ["python", "-m", "topopt", "verify", "private_run"],
        103: ["python", "-m", "topopt_mcp.server"],
        104: ["python", "-m", "topopt", "ratchet", "status"],
        105: ["vim", "topopt/cli.py"],
        106: b"",                                   # a kernel thread or a zombie: an empty command line
        107: ["python", "-m", "topopt", "run", "me.yaml"],       # the caller
        108: b"\xff\xfe python -m topopt run \xff",             # not UTF-8
    })
    (proc / "self").mkdir()
    (proc / "not_a_pid").mkdir()
    (proc / "109").mkdir()                          # a process that ended during the scan: no cmdline
    monkeypatch.setattr(hostinfo, "PROC_DIR", proc)
    found, notes = with_notes(hostinfo.topopt_processes, exclude_pid=107)
    assert notes == []
    assert found["count"] == 4 and found["by_kind"] == {"run": 1, "verify": 1, "mcp_server": 1, "other": 1}
    assert [(p["pid"], p["kind"]) for p in found["processes"]] == [(101, "run"), (102, "verify"), (103, "mcp_server"), (104, "other")]
    text = json.dumps(found)
    assert "secret_part" not in text and "private_run" not in text and "yaml" not in text      # pids and kinds only
    assert hostinfo.topopt_processes(exclude_pid=999)["count"] == 5                              # the caller is not special unless named


@linux_only
def test_the_scan_sees_a_real_process_and_leaves_out_its_own_pid(tmp_path):
    script = tmp_path / "bin" / "topopt"
    script.parent.mkdir()
    script.write_text("import time\ntime.sleep(120)\n")
    mcp = tmp_path / "topopt_mcp" / "server.py"
    mcp.parent.mkdir()
    mcp.write_text("import time\ntime.sleep(120)\n")
    procs = [subprocess.Popen([sys.executable, str(script), "run", "x.yaml"]), subprocess.Popen([sys.executable, str(mcp)])]
    try:
        deadline = time.time() + 20
        want = {procs[0].pid: "run", procs[1].pid: "mcp_server"}
        while time.time() < deadline:
            found = hostinfo.topopt_processes()
            seen = {p["pid"]: p["kind"] for p in found["processes"]}
            if all(seen.get(pid) == kind for pid, kind in want.items()):
                break
            time.sleep(0.1)
        assert all(seen.get(pid) == kind for pid, kind in want.items()), seen
        assert os.getpid() not in seen
    finally:
        for p in procs:
            p.kill()
            p.wait()


def test_the_scan_that_fails_is_none_with_a_note(monkeypatch, linux):
    def boom():
        raise PermissionError("hidepid")
    monkeypatch.setattr(hostinfo, "_commands_proc", boom)
    found, notes = with_notes(hostinfo.topopt_processes)
    assert found is None and notes == ["other_jobs: PermissionError: hidepid"]
    monkeypatch.setattr(sys, "platform", "freebsd14")
    found, notes = with_notes(hostinfo.topopt_processes)
    assert found is None and notes[0].startswith("other_jobs: OSError: no process list on freebsd14")


# --- the macOS branches ----------------------------------------------------------------------

PS_TEXT = """  PID COMMAND
    1 /sbin/launchd
  501 /Applications/Safari.app/Contents/MacOS/Safari
  612 /Users/me/proj/.venv/bin/python -m topopt run /Users/me/proj/hook.yaml --out /Users/me/runs/hook
  613 /Users/me/proj/.venv/bin/python -m topopt verify /Users/me/runs/old
  700 /Users/me/proj/.venv/bin/python -m topopt_mcp.server
  701 /Users/me/proj/.venv/bin/python /Users/me/proj/.venv/bin/topopt dashboard /Users/me/runs/hook
  702 vim topopt/cli.py
  703 /opt/homebrew/Cellar/python@3.12/3.12.4/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python -m topopt run /Users/me/b.yaml
"""
PMSET_TEXT = """System-wide power settings:
Currently in use:
 standby              1
 powernap             1
 lowpowermode         1
 displaysleep         10
"""


@pytest.fixture
def mac(monkeypatch):
    """This process looks like a Mac: `sys.platform` is darwin and the commands give the text of macOS."""
    monkeypatch.setattr(sys, "platform", "darwin")
    calls = []

    def command(argv):
        calls.append(argv)
        if argv[0] == "ps":
            return PS_TEXT
        if argv[0] == "pmset":
            return PMSET_TEXT
        if argv[-1] == "machdep.cpu.brand_string":
            return "Apple M2\n"
        raise AssertionError(argv)
    monkeypatch.setattr(hostinfo, "_command_output", command)
    return calls


def test_macos_cpu_model_comes_from_sysctl(mac):
    model, notes = with_notes(hostinfo.cpu_model)
    assert model == "Apple M2" and notes == [] and mac[0][1:] == ["-n", "machdep.cpu.brand_string"]


def test_macos_cpu_model_that_fails_is_none_with_a_note(monkeypatch, mac):
    def boom(argv):
        raise FileNotFoundError("sysctl")
    monkeypatch.setattr(hostinfo, "_command_output", boom)
    model, notes = with_notes(hostinfo.cpu_model)
    assert model is None and notes == ["cpu_model: FileNotFoundError: sysctl"]


def test_macos_other_processes_come_from_ps(mac):
    found, notes = with_notes(hostinfo.topopt_processes, exclude_pid=613)
    assert notes == [] and mac[0] == ["ps", "-axo", "pid=,command="]
    assert found["by_kind"] == {"run": 2, "verify": 0, "mcp_server": 1, "other": 1}    # 703: a framework Python, `Python` with a capital letter
    assert [p["pid"] for p in found["processes"]] == [612, 700, 701, 703]
    assert "hook.yaml" not in json.dumps(found)


def test_macos_has_no_procs_running_and_no_peak_reset(mac):
    n, notes = with_notes(hostinfo.procs_running)
    assert n is None and notes == ["procs_running: needs /proc/stat (Linux)"]
    start = hostinfo.usage()
    notes = []
    end = hostinfo.end_facts(start, notes)
    assert end["procs_running_end"] is None and notes == []                      # the note is at the start of the job, not again at the end
    notes = []
    assert hostinfo.reset_peak_rss(notes) is False and notes == []             # expected on a Mac, so no note


def test_macos_low_power_mode_comes_from_pmset(monkeypatch, mac):
    on, notes = with_notes(hostinfo.low_power_mode)
    assert on is True and notes == [] and mac[0] == ["pmset", "-g"]
    monkeypatch.setattr(hostinfo, "_command_output", lambda argv: PMSET_TEXT.replace("lowpowermode         1", "lowpowermode         0"))
    off, notes = with_notes(hostinfo.low_power_mode)
    assert off is False and notes == []
    monkeypatch.setattr(hostinfo, "_command_output", lambda argv: "System-wide power settings:\n standby 1\n")    # a desktop Mac
    none, notes = with_notes(hostinfo.low_power_mode)
    assert none is None and notes[0].startswith("low_power_mode: LookupError")
    monkeypatch.setattr(hostinfo, "_command_output", lambda argv: (_ for _ in ()).throw(subprocess.TimeoutExpired("pmset", 5)))
    none, notes = with_notes(hostinfo.low_power_mode)
    assert none is None and notes[0].startswith("low_power_mode: TimeoutExpired")


def test_low_power_mode_does_not_apply_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    low, notes = with_notes(hostinfo.low_power_mode)
    assert low is None and notes == []


def test_the_start_facts_of_a_mac(mac):
    f = hostinfo.start_facts()
    assert f["cpu_model"] == "Apple M2" and f["low_power_mode"] is True
    assert f["procs_running"] is None and "procs_running: needs /proc/stat (Linux)" in f["notes"]
    assert f["other_jobs"]["count"] == 5
    json.dumps(f)


def test_a_command_that_does_not_end_is_stopped_at_the_timeout(monkeypatch):
    monkeypatch.setattr(hostinfo, "COMMAND_TIMEOUT_S", 0.3)
    started = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        hostinfo._command_output([sys.executable, "-c", "import time; time.sleep(60)"])
    assert time.time() - started < 20                                          # it did not wait for the sleep


def test_a_command_gives_its_text_and_a_code_other_than_0_is_an_error():
    assert hostinfo._command_output([sys.executable, "-c", "print('ok')"]).strip() == "ok"
    with pytest.raises(RuntimeError, match="code 3"):
        hostinfo._command_output([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert hostinfo._command_output([sys.executable, "-c", "import sys; print(sys.stdin.read() == '')"]).strip() == "True"   # no terminal input
