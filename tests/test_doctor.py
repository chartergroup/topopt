"""`topopt doctor` and the checkout facts (CONTRACTS.md section 8, L1b).

The git tests build small repositories in a temporary folder (a bare "origin", a seed clone, a work clone), so they run
on every machine that has the `git` command. Nothing here loads an OpenCL driver.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from topopt import cli, doctor

REPO = Path(__file__).resolve().parent.parent
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs the git command")


def git(cwd, *args, check=True):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                           "-C", str(cwd), *args], check=check, capture_output=True, text=True, env=env).stdout


def commit(repo, name, text="x"):
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", f"add {name}")


@pytest.fixture
def repos(tmp_path):
    """`origin` (bare), `seed` (pushes to it) and `work` (a clone, no fetch since the clone)."""
    origin, seed, work = tmp_path / "origin.git", tmp_path / "seed", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    git(seed, "remote", "add", "origin", str(origin))
    commit(seed, "a.txt")
    git(seed, "push", "-q", "origin", "main")
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    return types_ns(origin=origin, seed=seed, work=work)


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


# --- checkout facts -------------------------------------------------------------------------------------------------


@needs_git
def test_a_fresh_clone_has_no_fetch_time_and_is_level_with_origin(repos):
    c = doctor.checkout_facts(repos.work)
    assert c["git"] is True and c["kind"] == "git checkout" and c["branch"] == "main" and c["detached"] is False
    assert c["head"] and c["head_date"] and c["default_branch_ref"] == "origin/main"
    assert (c["ahead"], c["behind"]) == (0, 0) and c["dirty"] is False and c["changed_files"] == 0
    assert c["last_fetch_utc"] is None and any("FETCH_HEAD" in n for n in c["notes"])
    assert "as of the clone (no fetch since)" in doctor.checkout_origin_line(c) and "never" not in doctor.checkout_origin_line(c)
    assert c["update_command"] == f"git -C {repos.work} pull --ff-only"
    assert c["fetch_command"] == f"git -C {repos.work} fetch origin"
    json.dumps(c)


@needs_git
def test_behind_is_counted_as_of_the_last_fetch_and_nothing_is_fetched(repos):
    commit(repos.seed, "b.txt")
    git(repos.seed, "push", "-q", "origin", "main")
    git(repos.work, "fetch", "-q", "origin")
    c = doctor.checkout_facts(repos.work)
    assert (c["ahead"], c["behind"]) == (0, 1)
    assert c["last_fetch_utc"] and 0 <= c["last_fetch_age_s"] < 120
    # origin moves again: doctor must not fetch, so the numbers and the fetch time stay as they were
    commit(repos.seed, "c.txt")
    git(repos.seed, "push", "-q", "origin", "main")
    fetch_head = repos.work / ".git" / "FETCH_HEAD"
    before = fetch_head.stat().st_mtime_ns
    c2 = doctor.checkout_facts(repos.work)
    assert c2["behind"] == 1 and fetch_head.stat().st_mtime_ns == before
    assert c2["last_fetch_utc"] == c["last_fetch_utc"]


@needs_git
def test_the_fetch_age_is_the_age_of_fetch_head(repos):
    git(repos.work, "fetch", "-q", "origin")
    three_days_ago = time.time() - 3 * 86400
    os.utime(repos.work / ".git" / "FETCH_HEAD", (three_days_ago, three_days_ago))
    c = doctor.checkout_facts(repos.work)
    assert abs(c["last_fetch_age_s"] - 3 * 86400) < 60
    assert doctor.human_age(c["last_fetch_age_s"]) == "3 days"
    assert "3 days ago" in doctor.checkout_origin_line(c)


@needs_git
def test_ahead_dirty_and_untracked_are_counted(repos):
    commit(repos.work, "local.txt")
    (repos.work / "a.txt").write_text("changed")
    (repos.work / "new.txt").write_text("new")
    c = doctor.checkout_facts(repos.work)
    assert c["ahead"] == 1 and c["behind"] == 0
    assert c["dirty"] is True and c["changed_files"] == 1 and c["untracked_files"] == 1
    line = doctor.checkout_code_line(c)
    assert "uncommitted changes: 1 file(s)" in line and "1 untracked" in line and "branch main" in line


@needs_git
def test_detached_head_has_no_branch_and_the_update_command_switches_first(repos):
    git(repos.work, "checkout", "-q", "--detach")
    c = doctor.checkout_facts(repos.work)
    assert c["detached"] is True and c["branch"] is None
    assert "switch main && git" in c["update_command"] and any("detached" in n for n in c["notes"])
    assert "detached HEAD" in doctor.checkout_code_line(c)


def add_worktree(repos, name="wt2", branch="feature"):
    wt = repos.work.parent / name
    git(repos.work, "worktree", "add", "-q", "-b", branch, str(wt))
    assert (wt / ".git").is_file()                              # a worktree has a .git file, not a folder
    return wt


def fetch_head_of(path):
    """The FETCH_HEAD file that a `git fetch` in `path` writes: the git folder of that worktree."""
    return Path(git(path, "rev-parse", "--absolute-git-dir").strip()) / "FETCH_HEAD"


def age(path, seconds):
    then = time.time() - seconds
    os.utime(path, (then, then))


@needs_git
def test_a_worktree_sees_a_fetch_that_was_made_in_the_main_clone(repos):
    git(repos.work, "fetch", "-q", "origin")
    wt = add_worktree(repos)
    assert not (wt / ".git" / "FETCH_HEAD").exists()
    c = doctor.checkout_facts(wt)
    assert c["git"] is True and c["branch"] == "feature"
    assert c["last_fetch_utc"] is not None and c["last_fetch_age_s"] < 120


@needs_git
def test_a_fetch_inside_a_linked_worktree_is_the_last_fetch(repos):
    """git writes FETCH_HEAD in the git folder of the worktree where `git fetch` ran (review M1)."""
    wt = add_worktree(repos)
    assert doctor.checkout_facts(wt)["last_fetch_utc"] is None            # nothing was fetched yet
    git(wt, "fetch", "-q", "origin")
    assert fetch_head_of(wt).is_file() and fetch_head_of(wt) != repos.work / ".git" / "FETCH_HEAD"
    assert not (repos.work / ".git" / "FETCH_HEAD").exists()               # the common file was not written
    for root in (wt, repos.work):                                          # the refs are shared: both see the fetch
        c = doctor.checkout_facts(root)
        assert c["last_fetch_utc"] is not None and c["last_fetch_age_s"] < 120, root
        assert not any("FETCH_HEAD" in n for n in c["notes"])


@needs_git
def test_the_newest_fetch_of_all_worktrees_wins(repos):
    git(repos.work, "fetch", "-q", "origin")
    wt = add_worktree(repos)
    git(wt, "fetch", "-q", "origin")
    age(repos.work / ".git" / "FETCH_HEAD", 7 * 86400)                     # the main clone fetched a week ago
    assert doctor.checkout_facts(repos.work)["last_fetch_age_s"] < 120      # the worktree fetched just now
    age(fetch_head_of(wt), 2 * 86400)
    c = doctor.checkout_facts(wt)
    assert abs(c["last_fetch_age_s"] - 2 * 86400) < 60                      # now the worktree one is the newest


@needs_git
def test_the_update_command_runs_in_the_worktree_that_has_the_default_branch(repos):
    """`git switch main` fails when main is checked out in another worktree (review m2): the command runs there."""
    git(repos.work, "fetch", "-q", "origin")
    wt = add_worktree(repos)
    c = doctor.checkout_facts(wt)
    assert c["update_command"] == f"git -C {repos.work} pull --ff-only"
    assert any("main is checked out in another worktree" in n and f"git -C {wt} merge origin/main" in n for n in c["notes"])
    assert "switch" not in c["update_command"]


@needs_git
def test_the_update_command_switches_to_main_when_no_worktree_has_it(repos):
    git(repos.work, "fetch", "-q", "origin")
    git(repos.work, "switch", "-q", "-c", "other")                        # main is now checked out nowhere
    wt = add_worktree(repos, branch="feature")
    c = doctor.checkout_facts(wt)
    assert c["update_command"] == f"git -C {wt} switch main && git -C {wt} pull --ff-only"
    assert any("branch feature, not main" in n for n in c["notes"])


@needs_git
def test_a_branch_that_tracks_the_default_branch_is_updated_with_a_plain_pull(repos):
    """A local branch with another name that tracks origin/main (review n10): no `switch main`."""
    git(repos.work, "switch", "-q", "-c", "master", "--track", "origin/main")
    c = doctor.checkout_facts(repos.work)
    assert c["branch"] == "master" and c["update_command"] == f"git -C {repos.work} pull --ff-only"
    assert not any("not main" in n for n in c["notes"])
    # with a commit of its own the branch cannot move forward by pull: the command is the one for another branch
    commit(repos.work, "own.txt")
    c = doctor.checkout_facts(repos.work)
    assert c["ahead"] == 1 and c["update_command"] == f"git -C {repos.work} switch main && git -C {repos.work} pull --ff-only"


@needs_git
def test_the_default_branch_falls_back_to_origin_main_without_origin_head(repos):
    git(repos.work, "remote", "set-head", "origin", "-d")
    assert doctor.checkout_facts(repos.work)["default_branch_ref"] == "origin/main"


@needs_git
def test_a_repository_without_origin_says_so(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, "a.txt")
    c = doctor.checkout_facts(tmp_path)
    assert c["git"] is True and c["head"] and c["default_branch_ref"] is None
    assert c["ahead"] is None and c["behind"] is None and any("no origin/main" in n.lower() or "origin/main" in n for n in c["notes"])
    assert doctor.checkout_origin_line(c) is None


def test_a_folder_without_git_is_a_plugin_copy_or_a_zip(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    c = doctor.checkout_facts(tmp_path)
    assert c["git"] is False and c["has_pyproject"] is True and c["kind"].startswith("source folder without git")
    assert c["head"] is None and c["update_command"] is None and c["fetch_command"] is None
    assert any("git clone https://github.com/chartergroup/topopt.git" in n for n in c["notes"])


def test_an_installed_package_has_no_tools(tmp_path):
    (tmp_path / "topopt").mkdir()
    c = doctor.checkout_facts(tmp_path)
    assert c["kind"].startswith("installed package") and c["has_tools"] is False and c["has_pyproject"] is False


def test_a_broken_git_folder_or_a_missing_git_command_is_a_note_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_is_macos", lambda: False)     # on a Mac `xcode-select -p` runs first; the `mac_without_tools` tests cover that
    (tmp_path / ".git").mkdir()                                  # not a repository
    c = doctor.checkout_facts(tmp_path)
    assert c["git"] is True and c["head"] is None and any("git does not answer" in n for n in c["notes"])

    def no_git(*a, **k):
        raise FileNotFoundError(2, "No such file or directory: 'git'")
    monkeypatch.setattr(subprocess, "run", no_git)
    c = doctor.checkout_facts(tmp_path)
    assert c["head"] is None and any("git does not answer" in n for n in c["notes"])


def render_checkout(c):
    return doctor.render({"time_utc": "t", "topopt_version": "0", "ok": True, "problems": [], "checkout": c, "environment": None,
                          "environment_error": "e", "imports": [], "known": {}, "offers": [], "next_commands": []})


def test_the_text_of_a_git_that_does_not_answer_has_no_made_up_fact(tmp_path):
    (tmp_path / ".git").mkdir()                                  # not a repository: git does not answer
    c = doctor.checkout_facts(tmp_path)
    text = render_checkout(c)
    assert "git does not answer" in text
    for wrong in ("detached HEAD", "git None", "date unknown", "update: None", "code:", "update:"):
        assert wrong not in text, wrong
    assert "detached HEAD" not in doctor.checkout_code_line(c) and "date unknown" not in doctor.checkout_code_line(c)


def test_a_known_detached_head_is_still_said():
    c = {"git": True, "head": "abc1234", "head_date": "2026-10-20T10:00:00+00:00", "branch": None, "detached": True,
         "dirty": False, "changed_files": 0, "untracked_files": 0}
    assert doctor.checkout_code_line(c) == "git abc1234 (2026-10-20), detached HEAD, no uncommitted changes"


def mac_without_tools(monkeypatch, tools_rc):
    """macOS with `/usr/bin/git`; `xcode-select -p` ends with `tools_rc`. Returns the commands that ran."""
    ran = []
    monkeypatch.setattr(doctor, "_is_macos", lambda: True)
    monkeypatch.setattr(doctor, "_which_git", lambda: "/usr/bin/git")

    def fake_run(cmd, *a, **k):
        ran.append(cmd[0])
        if cmd[0] == "xcode-select":
            assert cmd == ["xcode-select", "-p"]                # the one command that is not git: it only reads
        return tools_rc, "", ""
    monkeypatch.setattr(doctor, "_run", fake_run)
    return ran


def test_on_a_mac_without_the_command_line_tools_no_git_command_runs(tmp_path, monkeypatch):
    """`/usr/bin/git` is a stub there: it opens an install dialog (review m8). `xcode-select -p` decides first."""
    (tmp_path / ".git").mkdir()
    ran = mac_without_tools(monkeypatch, tools_rc=2)
    c = doctor.checkout_facts(tmp_path)
    assert ran == ["xcode-select"]                                # no git command was started
    assert c["git"] is True and c["head"] is None
    assert any("xcode-select --install" in n and "git facts are skipped" in n for n in c["notes"])


def test_on_a_mac_with_the_tools_git_runs_as_always(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    ran = mac_without_tools(monkeypatch, tools_rc=0)
    doctor.checkout_facts(tmp_path)
    assert ran[0] == "xcode-select" and "git" in ran[1:]


def test_another_git_than_the_stub_needs_no_tools_check(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    ran = mac_without_tools(monkeypatch, tools_rc=2)
    monkeypatch.setattr(doctor, "_which_git", lambda: "/opt/homebrew/bin/git")      # not the stub
    doctor.checkout_facts(tmp_path)
    assert "xcode-select" not in ran and "git" in ran


@needs_git
def test_every_git_command_is_read_only(repos, monkeypatch):
    monkeypatch.setattr(doctor, "_is_macos", lambda: False)     # on a Mac `xcode-select -p` runs first; a test below checks that command
    real = subprocess.run
    seen = []
    wt = add_worktree(repos)

    def spy(cmd, *a, **k):
        seen.append(cmd[3] if cmd[0] == "git" else cmd[0])
        return real(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", spy)
    doctor.checkout_facts(repos.work)
    doctor.checkout_facts(wt)
    assert seen and set(seen) <= {"rev-parse", "log", "symbolic-ref", "status", "rev-list", "worktree"}
    assert "fetch" not in seen and "pull" not in seen


# --- the import test ------------------------------------------------------------------------------------------------


def row(statement, dist="pytest", use="core", **kw):
    return doctor.import_check(statement, statement, dist, use, **kw)


def test_an_import_that_works():
    r = row("json", dist="pytest")
    assert r["status"] == "ok" and r["version"] and r["seconds"] is not None and not doctor.is_problem(r)


def test_a_package_that_is_not_installed(tmp_path):
    r = row("no_such_module_xyz", dist="no-such-dist-xyz", cwd=str(tmp_path))
    assert r["status"] == "not installed" and r["version"] is None
    assert doctor.is_problem(r) is True                                          # a core package
    assert doctor.is_problem(row("no_such_module_xyz", dist="no-such-dist-xyz", use="extra gpu")) is False   # an optional one


def test_a_cut_install_has_metadata_and_no_files(tmp_path):
    """M-2: the dist-info folder is there, the package files are not. The row says `cut` and names the metadata version."""
    info = tmp_path / "cutpkg-1.2.3.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: cutpkg\nVersion: 1.2.3\n")
    r = row("cutpkg", dist="cutpkg", use="extra gpu", code_folder=str(tmp_path))
    assert r["status"] == "cut" and r["version"] == "1.2.3" and "install was cut" in r["detail"]
    assert doctor.is_problem(r) is True                                          # installed and broken: a problem, even if optional


def test_an_import_that_raises_another_error_is_failed(tmp_path):
    (tmp_path / "badlib.py").write_text("raise ImportError('libfoo.so.1: cannot open shared object file')\n")
    r = row("badlib", dist="badlib", use="extra pardiso", code_folder=str(tmp_path))
    assert r["status"] == "failed" and "libfoo.so.1" in r["detail"] and doctor.is_problem(r)


def test_a_missing_module_under_a_package_is_failed_not_cut(tmp_path):
    (tmp_path / "needs_other.py").write_text("import module_that_is_not_there_xyz\n")
    r = row("needs_other", dist="needs_other", use="extra gpu", code_folder=str(tmp_path))
    assert r["status"] == "failed" and "module_that_is_not_there_xyz" in r["detail"]


def fake_server(tmp_path):
    """A package `fakeserver.server` that needs the package `fakedep` (the way `topopt_mcp.server` needs `mcp`)."""
    (tmp_path / "fakeserver").mkdir()
    (tmp_path / "fakeserver" / "__init__.py").write_text("")
    (tmp_path / "fakeserver" / "server.py").write_text("import fakedep\n")


def test_a_server_row_without_its_package_is_not_installed_and_no_problem(tmp_path):
    """Review M2: a venv without the `mcp` extra is a valid install. The row `MCP server` says `not installed`, as `mcp` does."""
    fake_server(tmp_path)
    r = doctor.import_check("Fake server", "fakeserver.server", "fakedep", "extra mcp", code_folder=str(tmp_path), cwd=str(tmp_path))
    assert r["status"] == "not installed" and "fakedep" in r["detail"] and not doctor.is_problem(r)


def test_a_server_row_with_the_metadata_of_its_package_is_cut_and_shows_no_version(tmp_path):
    fake_server(tmp_path)
    info = tmp_path / "fakedep-4.5.6.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakedep\nVersion: 4.5.6\n")
    args = ("Fake server", "fakeserver.server", "fakedep", "extra mcp")
    r = doctor.import_check(*args, code_folder=str(tmp_path), cwd=str(tmp_path), show_version=False)
    assert r["status"] == "cut" and r["version"] is None and "fakedep 4.5.6" in r["detail"] and doctor.is_problem(r)
    assert doctor.import_check(*args, code_folder=str(tmp_path), cwd=str(tmp_path))["version"] == "4.5.6"


def test_the_mcp_server_row_has_no_version_of_its_own():
    row_ = next(r for r in doctor.IMPORT_CHECKS if r[0] == "MCP server")
    assert row_[2] == "mcp" and row_[4] is False


@pytest.mark.skipif(sys.platform == "win32", reason="signals")
def test_a_crash_and_a_timeout(tmp_path):
    (tmp_path / "crashlib.py").write_text("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    (tmp_path / "slowlib.py").write_text("import time\ntime.sleep(30)\n")
    crash = row("crashlib", dist="crashlib", use="extra gpu", code_folder=str(tmp_path))
    assert crash["status"] == "crashed" and "signal" in crash["detail"]
    slow = row("slowlib", dist="slowlib", use="extra gpu", code_folder=str(tmp_path), timeout=1)
    assert slow["status"] == "timeout" and doctor.is_problem(slow)


def test_the_import_list_names_the_core_packages_and_the_optional_ones():
    names = {r[0] for r in doctor.IMPORT_CHECKS}
    assert {"numpy", "scipy", "numba", "PyYAML", "scikit-image", "trimesh", "manifold3d"} <= names
    assert {"pyopencl", "pypardiso", "gmsh", "cholespy", "scikit-fem", "mcp", "OpenCASCADE (OCP)"} <= names
    core = [r for r in doctor.IMPORT_CHECKS if r[3] == "core"]
    text = (REPO / "pyproject.toml").read_text().split("[project.optional-dependencies]")[0]
    for _name, _stmt, dist, _use in core:                                        # the core list follows pyproject.toml
        assert dist.lower().replace("_", "-") in text.lower(), dist


# --- the environments and the clients -------------------------------------------------------------------------------


def write_clients(root, secret="Bearer SECRET-TOKEN-123"):
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": {"topopt": {
        "type": "stdio", "command": ".venv/bin/python", "args": ["-m", "topopt_mcp.server"],
        "env": {"AUTH": secret}, "headers": {"Authorization": secret}}}}))
    (root / ".cursor").mkdir()
    (root / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {"topopt": {
        "command": "sh", "args": ["${workspaceFolder}/scripts/topopt-env.sh", "mcp"]}}}))
    (root / ".codex").mkdir()
    (root / ".codex" / "config.toml").write_text(
        '[mcp_servers.topopt]\ncommand = "sh"\nargs = ["scripts/topopt-env.sh", "mcp"]\n[mcp_servers.topopt.env]\nK = "' + secret + '"\n')
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"mcpServers": {"topopt": {
        "command": "sh", "args": ["${CLAUDE_PLUGIN_ROOT}/scripts/topopt-env.sh", "mcp"]}}}))


def test_each_client_config_says_which_environment_it_starts(tmp_path):
    write_clients(tmp_path)
    rows = {r["file"]: r["environment"] for r in doctor.client_configs(tmp_path)}
    assert rows == {".mcp.json": ".venv", ".claude-plugin/plugin.json": "launcher",
                    ".cursor/mcp.json": "launcher", ".codex/config.toml": "launcher"}


def test_a_secret_in_a_client_config_is_never_in_the_report(tmp_path):
    write_clients(tmp_path)
    known = doctor.known_environments(tmp_path, probe=lambda p: {"runs": True, "python_version": "3.12.0", "topopt_from": None, "error": None})
    assert "SECRET-TOKEN-123" not in json.dumps(known)
    text = doctor.render({"time_utc": "t", "topopt_version": "0", "ok": True, "problems": [], "checkout": {}, "environment": None,
                          "environment_error": "x", "imports": [], "known": known, "offers": [], "next_commands": []})
    assert "SECRET-TOKEN-123" not in text and ".mcp.json" in text


def test_a_client_config_that_cannot_be_read_is_a_row_not_a_crash(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / ".mcp.json").write_text("{ not json")
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text("[mcp_servers.other]\ncommand='x'\n")
    rows = {r["file"]: r["environment"] for r in doctor.client_configs(tmp_path)}
    assert rows == {".mcp.json": "unreadable", ".codex/config.toml": "no topopt server"}


def test_the_launcher_env_dir_follows_the_rule_of_the_script(tmp_path, monkeypatch):
    for var in ("TOPOPT_ENV_DIR", "EVAL_TOPOPT_ENV_DIR", "CLAUDE_PLUGIN_DATA"):
        monkeypatch.delenv(var, raising=False)
    assert doctor.launcher_env_dir(tmp_path) == tmp_path / ".plugin-env"
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "plugin-data"))
    assert doctor.launcher_env_dir(tmp_path) == tmp_path / "plugin-data"
    monkeypatch.setenv("EVAL_TOPOPT_ENV_DIR", str(tmp_path / "eval"))
    assert doctor.launcher_env_dir(tmp_path) == tmp_path / "eval"
    monkeypatch.setenv("TOPOPT_ENV_DIR", str(tmp_path / "mine"))
    assert doctor.launcher_env_dir(tmp_path) == tmp_path / "mine"
    monkeypatch.setenv("TOPOPT_ENV_DIR", "relative/dir")
    assert doctor.launcher_env_dir(tmp_path) == Path(os.path.abspath("relative/dir"))


def fake_python(path, version="3.12.1", topopt_from="/code/topopt/__init__.py"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho PY={version}\necho TOPOPT={topopt_from}\n")
    path.chmod(0o755)


def test_known_environments_with_a_venv_and_a_launcher_env(tmp_path, monkeypatch):
    write_clients(tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "topopt-env.sh").write_text("#!/bin/sh\n")      # a plugin copy or a checkout has the launcher
    monkeypatch.delenv("CLAUDE_PLUGIN_DATA", raising=False)
    monkeypatch.setenv("TOPOPT_ENV_DIR", str(tmp_path / "launcher-env"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))                           # no plugin data folder is found
    fake_python(tmp_path / ".venv" / "bin" / "python")
    known = doctor.known_environments(tmp_path)
    by_role = {e["role"]: e for e in known["environments"]}
    assert by_role["other"]["is_this"] is True                                   # this test Python is neither of them
    venv, launcher = by_role[".venv"], by_role["launcher"]
    assert venv["exists"] and venv["runs"] is True and venv["python_version"] == "3.12.1"
    assert venv["topopt_from"] == "/code" and venv["used_by"] == ["Claude Code or VS Code in the repo folder"]
    assert launcher["exists"] is False and launcher["runs"] is None and "not made yet" in launcher["error"]
    assert launcher["used_by"] == ["Claude Code plugin", "Cursor", "Codex"]
    assert launcher["found_by"] == "TOPOPT_ENV_DIR"


def test_a_python_that_does_not_run_is_reported(tmp_path):
    bad = tmp_path / "bin" / "python"
    bad.parent.mkdir()
    bad.write_text("#!/bin/sh\necho 'dyld: Library not loaded' >&2\nexit 3\n")
    bad.chmod(0o755)
    r = doctor._python_probe(bad)
    assert r["runs"] is False and "Library not loaded" in r["error"]


def test_the_probe_of_this_python():
    r = doctor._python_probe(Path(sys.executable))
    assert r["runs"] is True and r["python_version"] == platform.python_version() and r["topopt_from"]


# --- the commands -----------------------------------------------------------------------------------------------------


def base_data(**kw):
    data = {"checkout": {"code_folder": "/code dir", "git": True, "has_pyproject": True, "behind": 0, "default_branch_ref": "origin/main",
                         "update_command": "git -C '/code dir' pull --ff-only", "fetch_command": "git -C '/code dir' fetch origin"},
            "imports": [], "offers": [], "known": {"environments": [{"is_this": True, "role": ".venv", "dir": "/code dir/.venv"}]}}
    data.update(kw)
    return data


def test_next_commands_for_a_cut_install_in_the_dev_venv():
    cut = {"name": "scipy", "use": "core", "status": "cut", "detail": "x"}
    cmds = doctor.next_commands(base_data(imports=[cut]))
    assert cmds[0]["command"].startswith("uv pip install --reinstall --python ") and "[dev,mcp]" in cmds[0]["command"]
    missing = {"name": "numba", "use": "core", "status": "not installed", "detail": "x"}
    first = doctor.next_commands(base_data(imports=[missing]))[0]["command"]
    assert "--reinstall" not in first and first.startswith("uv pip install --python ")


def test_next_commands_for_a_broken_launcher_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("TOPOPT_ENV_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_DATA", raising=False)
    cut = {"name": "numpy", "use": "core", "status": "cut", "detail": "x"}
    data = base_data(imports=[cut], known={"environments": [{"is_this": True, "role": "launcher", "dir": str(tmp_path / "env" / "venv")}]})
    data["checkout"]["code_folder"] = str(tmp_path)
    items = doctor.next_commands(data)
    marker = tmp_path / "env" / "venv" / ".topopt-step2"
    # the command of `sh scripts/topopt-env.sh doctor`: the marker makes sync install the packages again
    assert items[0]["command"] == f"touch {marker} && sh {tmp_path / 'scripts' / 'topopt-env.sh'} sync"
    assert "rm -rf" not in " ".join(c["command"] for c in items) and "200 MB" not in items[0]["why"]
    assert "needs the network" in items[0]["why"] and "UV_OFFLINE=1" in items[0]["why"]      # uv --reinstall also refreshes
    assert "uv cache" in items[0]["why"] and "no download" not in items[0]["why"].lower()


def test_next_commands_for_a_broken_wheel_install_has_no_command_to_guess():
    data = base_data(imports=[{"name": "numpy", "use": "core", "status": "not installed", "detail": "x"}])
    data["checkout"] = {"code_folder": "/site-packages", "git": False, "has_pyproject": False}
    (item,) = doctor.next_commands(data)
    assert item["command"] is None and "installed package, not a checkout" in item["why"]
    assert "numpy" in doctor.render({"time_utc": "t", "topopt_version": "0", "ok": False, "problems": ["x"], "checkout": data["checkout"],
                                     "environment": None, "environment_error": "e", "imports": [], "known": {}, "offers": [],
                                     "next_commands": [item]})


def test_the_launcher_command_names_a_custom_env_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_ENV_DIR", str(tmp_path / "my env"))
    cmd = doctor._launcher_command(tmp_path, "sync")
    assert cmd.startswith("TOPOPT_ENV_DIR=") and "'" in cmd and cmd.endswith(" sync") and "topopt-env.sh" in cmd


def test_next_commands_when_level_with_origin_give_the_fetch():
    cmds = doctor.next_commands(base_data())
    assert [c["command"] for c in cmds] == ["git -C '/code dir' fetch origin"]


def test_next_commands_when_behind_origin_and_with_offers():
    data = base_data(offers=[{"action": "install_pyamg", "command": "uv pip install pyamg"},
                             {"action": "install_pypardiso", "install_command": "uv pip install pypardiso"}])
    data["checkout"]["behind"] = 12
    cmds = doctor.next_commands(data)
    assert cmds[0]["command"] == "git -C '/code dir' pull --ff-only" and "12 commits behind origin/main" in cmds[0]["why"]
    assert "as of the last fetch" in cmds[0]["why"]
    assert "this fetches first" in cmds[0]["why"]
    assert not any(c["command"] == "git -C '/code dir' fetch origin" for c in cmds)     # the update fetches: no second fetch
    assert [c["command"] for c in cmds][-2:] == ["uv pip install pyamg", "uv pip install pypardiso"]
    assert not any(" #" in c["command"] for c in cmds)                          # zsh does not read a comment in an interactive shell


# --- the report --------------------------------------------------------------------------------------------------------


def not_installed_row(name, use):
    return {"name": name, "import": name, "package": name, "use": use, "version": None, "status": "not installed",
            "detail": f"ModuleNotFoundError: No module named '{name}'", "seconds": 0.1}


def test_a_venv_without_the_mcp_extra_is_a_valid_install_with_the_install_option(monkeypatch, capsys):
    """Review M2: exit code 0; the rows say `not installed (optional)`; the report gives the install command."""
    rows = [{"name": "numpy", "import": "numpy", "package": "numpy", "use": "core", "version": "2", "status": "ok",
             "detail": "", "seconds": 0.1},
            not_installed_row("mcp", "extra mcp"), not_installed_row("MCP server", "extra mcp"),
            not_installed_row("gmsh", "extra tet")]
    monkeypatch.setattr(doctor, "import_checks", lambda **k: [dict(r) for r in rows])
    d = doctor.collect()
    assert d["ok"] is True and d["problems"] == []
    by = {r["name"]: r for r in d["imports"]}
    assert by["mcp"]["install_command"].endswith("[mcp]'") and by["gmsh"]["install_command"].endswith("[tet]'")
    text = doctor.render(d)
    assert "not installed (optional): mcp (extra mcp), MCP server (extra mcp), gmsh (extra tet)" in text
    assert "To add an optional package" in text and text.count("    mcp: uv pip install --python ") == 1
    assert "PROBLEM" not in text
    assert doctor.run() == 0


def test_the_pardiso_extra_is_offered_only_where_it_can_install(monkeypatch):
    monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
    assert doctor._extra_fits_here("extra pardiso") is False and doctor._extra_fits_here("extra mcp") is True
    monkeypatch.setattr(doctor.platform, "machine", lambda: "x86_64")
    assert doctor._extra_fits_here("extra pardiso") is sys.platform.startswith("linux")
    assert doctor._extra_fits_here("core") is False


def test_a_python_that_runs_but_cannot_import_topopt_says_so():
    env = {"role": ".venv", "name": "checkout .venv", "python": "/c/.venv/bin/python", "runs": True, "python_version": "3.12.1",
           "error": "ModuleNotFoundError: No module named 'topopt'", "is_this": False, "used_by": []}
    text = doctor.render({"time_utc": "t", "topopt_version": "0", "ok": True, "problems": [], "checkout": {}, "environment": None,
                          "environment_error": "e", "imports": [], "known": {"environments": [env], "clients": []},
                          "offers": [], "next_commands": []})
    assert "runs (Python 3.12.1), but topopt does not import: ModuleNotFoundError: No module named 'topopt'" in text


def test_a_long_problem_line_is_one_item_with_a_hanging_indent():
    long_line = "scipy (core): failed. " + "ImportError: cannot import name 'x' from 'y' " * 5
    text = doctor.render({"time_utc": "t", "topopt_version": "0", "ok": False, "problems": [long_line], "checkout": {},
                          "environment": None, "environment_error": "e", "imports": [], "known": {}, "offers": [],
                          "next_commands": []})
    block = text.split("\n\nProblems\n", 1)[1].split("\n\nOffers", 1)[0].splitlines()
    assert len(block) > 1 and block[0].startswith("  - ") and all(ln.startswith("    ") and not ln.startswith("  - ") for ln in block[1:])


def test_a_wheel_install_lists_no_launcher_environment(tmp_path):
    (tmp_path / "topopt").mkdir()                                # no pyproject.toml, no scripts/topopt-env.sh
    roles = [e["role"] for e in doctor.known_environments(tmp_path)["environments"]]
    assert "launcher" not in roles and ".venv" not in roles
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "topopt-env.sh").write_text("#!/bin/sh\n")
    assert "launcher" in [e["role"] for e in doctor.known_environments(tmp_path)["environments"]]


def test_the_yaml_line_is_called_the_problem_file(tmp_path, capsys):
    assert cli.main(["doctor", str(tmp_path / "no" / "such.yaml")]) == 0
    out = capsys.readouterr().out
    assert "status: ok, no problem found" in out and "problem file " in out and "such.yaml: error:" in out


def test_collect_keeps_going_when_environment_data_fails(monkeypatch):
    from topopt import environment

    def broken(*a, **k):
        raise RuntimeError("scipy is broken")
    monkeypatch.setattr(environment, "environment_data", broken)
    d = doctor.collect(imports=False)
    assert d["environment"] is None and "RuntimeError: scipy is broken" in d["environment_error"]
    assert d["ok"] is False and any("environment_data() failed" in p for p in d["problems"])
    assert d["checkout"]["code_folder"] and d["known"]["environments"]          # the other parts still ran
    text = doctor.render(d)
    assert "not available: RuntimeError: scipy is broken" in text and "Problems" in text


def test_doctor_on_this_machine():
    d = doctor.collect()
    assert d["ok"], d["problems"]
    assert d["environment"]["python"]["executable"] == sys.executable and d["checkout"] == d["environment"]["checkout"]
    rows = {r["name"]: r for r in d["imports"]}
    assert rows["numpy"]["status"] == "ok" and rows["scipy"]["status"] == "ok" and rows["numba"]["status"] == "ok"
    assert d["offers"] == d["environment"]["recommendations"]
    json.dumps(d, default=str)


def test_the_text_report_has_the_sections_in_order():
    text = doctor.render(doctor.collect(imports=False))
    heads = ["Checkout", "Machine", "Packages", "Environments", "Offers", "Next commands"]
    positions = [text.index("\n" + h) for h in heads]
    assert positions == sorted(positions)
    assert text.startswith("topopt doctor (read-only") and "not tested" in text


def test_the_command_prints_json_and_the_exit_code_follows_the_problems(capsys, monkeypatch):
    assert cli.main(["doctor", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["ok"] is True and d["problems"] == [] and {"checkout", "imports", "known", "offers", "next_commands"} <= set(d)
    monkeypatch.setattr(doctor, "import_checks", lambda **k: [
        {"name": "scipy", "import": "scipy", "package": "scipy", "use": "core", "version": "1.0", "status": "cut",
         "detail": "the install was cut", "seconds": 0.1}])
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "status: 1 problem(s)" in out and "PROBLEM scipy (core): cut" in out and "Problems" in out


def test_the_parser_has_the_two_commands():
    ap = cli.build_parser()
    a = ap.parse_args(["doctor", "case.yaml", "--json", "--timeout-s", "5"])
    assert (a.yaml, a.json, a.timeout_s, a.func) == ("case.yaml", True, 5.0, cli.cmd_doctor)
    b = ap.parse_args(["gpu-check", "--out", "r", "--device", "1", "--skip-bench", "--skip-tests", "--step-timeout-s", "60"])
    assert (b.out, b.device, b.skip_bench, b.skip_tests, b.step_timeout_s) == ("r", "1", True, True, 60.0)
    assert ap.parse_args(["gpu-check"]).device is None


# --- light start ------------------------------------------------------------------------------------------------------


def run_python(code, **env):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, **env}, cwd=str(REPO))


def test_importing_the_cli_and_doctor_loads_no_heavy_package():
    """`topopt doctor` must start where numpy or scipy is broken: the import of the command loads none of them."""
    r = run_python("import sys, topopt.cli, topopt.doctor; "
                   "print(sorted(m for m in ('numpy', 'scipy', 'numba', 'pyopencl', 'mcp', 'skimage', 'trimesh') if m in sys.modules))")
    assert r.returncode == 0 and r.stdout.strip() == "[]", r.stderr


def test_doctor_runs_in_an_environment_with_a_broken_scipy(tmp_path):
    """A scipy that raises at import: the command still starts, shows the break as a line and exits with 1."""
    fake = tmp_path / "fake"
    (fake / "scipy").mkdir(parents=True)
    (fake / "scipy" / "__init__.py").write_text("raise ImportError('scipy is broken for this test')\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(fake), str(REPO)])}
    r = subprocess.run([sys.executable, "-m", "topopt", "doctor", "--json"], capture_output=True, text=True, env=env,
                       cwd=str(tmp_path))
    assert r.returncode == 1, r.stderr
    d = json.loads(r.stdout)
    assert d["ok"] is False
    assert any("scipy" in p and "scipy is broken for this test" in p for p in d["problems"])
    rows = {x["name"]: x for x in d["imports"]}
    assert rows["scipy"]["status"] == "failed" and rows["numpy"]["status"] == "ok"
    assert d["next_commands"]                                                       # and it says what to run


# --- progress lines while the command runs (#424) -------------------------------------------------------------------------

def _slow_rows(monkeypatch, order):
    """Make `import_check` end the rows in `order` (the names of rows), each after a short sleep that grows with its place."""
    names = [row[0] for row in doctor.IMPORT_CHECKS]
    delay = {n: 0.05 * order.index(n) for n in order}

    def fake(name, statement, dist, use, **kw):
        time.sleep(delay.get(name, 0.0))
        return {"name": name, "import": statement, "package": dist, "use": use, "version": "1", "status": "ok",
                "detail": "", "seconds": delay.get(name, 0.0)}

    monkeypatch.setattr(doctor, "import_check", fake)
    return names


def test_import_checks_call_progress_once_per_row_before_they_return_and_keep_the_row_order(monkeypatch):
    names = [row[0] for row in doctor.IMPORT_CHECKS]
    _slow_rows(monkeypatch, list(reversed(names)))          # the last row ends first
    lines = []
    rows = doctor.import_checks(progress=lines.append)
    assert [r["name"] for r in rows] == names               # the order of the rows is the order of IMPORT_CHECKS
    n = len(names)
    assert len(lines) == n
    for k, line in enumerate(lines, 1):
        assert f"[{k}/{n}]" in line and "ok" in line
    shown = [line.split("] ", 1)[1].split(":")[0].split(" (extra ")[0] for line in lines]
    assert sorted(shown) == sorted(names)
    assert shown[0] != names[0]     # the lines come in the order of the ends, not of the rows


def test_import_checks_call_progress_while_a_later_row_still_runs(monkeypatch):
    seen_at_the_end = {}

    def fake(name, statement, dist, use, **kw):
        if name == IMPORT_LAST:
            for _ in range(200):                              # wait until the first line was printed
                if seen_at_the_end.get("first"):
                    break
                time.sleep(0.01)
            seen_at_the_end["when_the_last_row_ended"] = seen_at_the_end.get("first")
        return {"name": name, "import": statement, "package": dist, "use": use, "version": None, "status": "ok",
                "detail": "", "seconds": 0.0}

    IMPORT_LAST = doctor.IMPORT_CHECKS[-1][0]
    monkeypatch.setattr(doctor, "import_check", fake)
    doctor.import_checks(progress=lambda line: seen_at_the_end.setdefault("first", line))
    assert seen_at_the_end["when_the_last_row_ended"]         # a line was out before the slow last row ended


def test_no_progress_by_default(monkeypatch, capsys):
    _slow_rows(monkeypatch, [])
    rows = doctor.import_checks()
    assert len(rows) == len(doctor.IMPORT_CHECKS)
    d = doctor.collect(imports=True)
    assert d["imports"] and capsys.readouterr() == ("", "")


def test_collect_says_what_it_does_before_the_machine_facts(monkeypatch):
    lines = []
    order = []
    import topopt.environment as environment
    monkeypatch.setattr(environment, "environment_data", lambda *a, **k: order.append("environment") or {"checkout": None})
    _slow_rows(monkeypatch, [])
    doctor.collect(progress=lambda t: (lines.append(t), order.append("line")))
    assert order[0] == "line" and "self-test" in lines[0] and "gpu-mg" in lines[0]
    from topopt.gpu_mg.probe import PROBE_TIMEOUT_S
    assert f"{PROBE_TIMEOUT_S:g} s" in lines[0]               # the number is the constant of the probe, not a copy
    assert any(f"up to {doctor.IMPORT_TIMEOUT_S:g} s each" in t for t in lines)


@pytest.mark.parametrize("as_json", [False, True])
def test_run_writes_progress_to_stderr_and_the_report_to_stdout(monkeypatch, capsys, as_json):
    _slow_rows(monkeypatch, [])
    assert doctor.run(as_json=as_json) == 0
    out, err = capsys.readouterr()
    n = len(doctor.IMPORT_CHECKS)
    assert f"[{n}/{n}]" in err and "[1/" in err and "reading the machine" in err
    assert "[1/" not in out and "reading the machine" not in out          # stdout keeps the report only
    if as_json:
        assert json.loads(out)["imports"]
    else:
        assert "Packages" in out


def test_progress_lines_of_optional_packages_name_the_extra(monkeypatch):
    _slow_rows(monkeypatch, [])
    lines = []
    doctor.import_checks(progress=lines.append)
    by_name = {line.split("] ", 1)[1].split(":")[0]: line for line in lines}
    assert "pypardiso (extra pardiso): ok" in by_name["pypardiso (extra pardiso)"]
    assert by_name["numpy"].split("] ", 1)[1].startswith("numpy: ok")        # a core row has no extra


def _doctor_process(tmp_path, stderr):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, "-P", "-m", "topopt", "doctor", "--json", "--timeout-s", "30"], stdout=subprocess.PIPE,
                          stderr=stderr, env=env, cwd=str(tmp_path), timeout=120)


def test_a_closed_stderr_changes_neither_stdout_nor_the_exit_code(tmp_path):
    """`topopt doctor --json 2>&-`: the progress has no place to go. stdout stays one JSON object (a closed stderr must not
    make print() fall back to stdout)."""
    code = ("import subprocess, sys, os\n"
            "r = subprocess.run(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,\n"
            "                   preexec_fn=lambda: os.close(2), env={**os.environ, 'PYTHONPATH': %r}, cwd=%r)\n"
            "sys.stdout.buffer.write(r.stdout); sys.exit(r.returncode)\n" % (str(REPO), str(tmp_path)))
    r = subprocess.run([sys.executable, "-c", code, sys.executable, "-P", "-m", "topopt", "doctor", "--json", "--timeout-s", "30"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert r.returncode in (0, 1), r.stderr
    assert json.loads(r.stdout)["imports"]


def test_a_stderr_pipe_with_no_reader_changes_neither_stdout_nor_the_exit_code(tmp_path):
    reader, writer = os.pipe()
    os.close(reader)
    try:
        r = _doctor_process(tmp_path, writer)
    finally:
        os.close(writer)
    assert r.returncode in (0, 1)                        # not 120 (a flush of a broken stderr at the exit)
    assert json.loads(r.stdout)["imports"]


def test_the_launcher_line_names_the_limits_of_doctor():
    text = (REPO / "scripts" / "topopt-env.sh").read_text()
    line = next(ln for ln in text.splitlines() if "printf 'topopt doctor (Python): reading the machine" in ln)
    from topopt.gpu_mg.probe import PROBE_TIMEOUT_S
    assert f"up to {doctor.IMPORT_TIMEOUT_S} s each" in line      # the number of IMPORT_TIMEOUT_S
    assert f"up to {PROBE_TIMEOUT_S:g} s," in line                  # the number of the self-test of gpu-mg
    assert "a minute" not in line                                   # no time claim without a measurement


# --- the skill text and the contract --------------------------------------------------------------------------------------


def test_start_here_runs_the_doctor_first_and_names_the_gpu_facts():
    text = (REPO / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8")
    step0 = text.split("## Step 0")[1].split("## Step 1")[0]
    assert 'topopt-env.sh" doctor' in step0 and "topopt doctor" in step0
    assert step0.index("doctor") < step0.index("status")                      # the doctor comes before the status command
    assert "`ImportError`" in step0 and "marker" in step0                    # the cut-setup sentence follows the launcher repair
    assert "the command that the doctor prints" in step0 and "delete the folder" not in step0     # one repair, as the L1a doctor
    assert "when the uv cache is empty" in step0                             # the 200 MB is a fact only for an empty cache
    assert "needs the network" in step0 and "UV_OFFLINE=1" in step0 and "no download" not in step0.lower()
    step1 = text.split("## Step 1")[1].split("## Step 2")[0]
    for word in ("`gpu_solver`", "to_use_the_gpu", "solver: gpu-mg", "--solver gpu-mg", "gpu-check", "`checkout`", "last fetch", "one machine"):
        assert word in step1, word
    for word in ("use_gpu_mg", "install_gpu_extra", "experimental", "opt-in", "opt in"):      # gpu-mg is a main value: no offer, no opt-in
        assert word not in step1, word
    assert "run_topopt` has no solver argument" in step1


def test_contracts_section_10_names_every_key_of_the_environment_data():
    from topopt.environment import environment_data
    text = (REPO / "CONTRACTS.md").read_text(encoding="utf-8")
    section = text.split("- `environment(yaml_path")[1].split("## 11.")[0]
    for key in environment_data():
        assert f"`{key}`" in section, key
    for key in doctor.checkout_facts():
        assert f"`{key}`" in section, key
    for action in ("use_gpu_mg", "install_gpu_extra"):       # the offers of the opt-in time are gone: the fact `gpu_solver` replaced them
        assert action not in section and action not in text.split("## 20.")[1].split("## 21.")[0]


def test_contracts_section_8_has_the_two_commands_and_the_exit_codes():
    section = (REPO / "CONTRACTS.md").read_text(encoding="utf-8").split("## 8. CLI")[1].split("## 9.")[0]
    for word in ("topopt doctor", "topopt gpu-check", "--timeout-s", "--step-timeout-s", "--skip-bench", "--device", "Exit code of `doctor`",
                 "FETCH_HEAD", "cut", "not run", "gpu-check.md", "gpu-check.json"):
        assert word in section, word
    assert "has never run on a real GPU" not in (REPO / "CONTRACTS.md").read_text(encoding="utf-8")


