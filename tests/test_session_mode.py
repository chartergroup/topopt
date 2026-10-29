"""Tests for `scripts/dev/session_mode.py`: the maintainer or contributor line that Claude Code shows at session start.

The script looks for a clone of the private knowledge base (KB) in this order: the environment variable TOPOPT_KB, the
sibling clone `<repo root>/../topopt-kb`, the folder in the file `<config folder>/topopt/kb-path`. The KB counts as
found only when `maintainer/START.md` is a readable file in that folder. Every test builds its own fake KB and repo
folders in `tmp_path`: nothing reads the real home folder or the real environment.

When the KB is not found, the script asks `gh` whether the user is an active member of the organization (the only network
call). The tests put a fake `gh` first on a PATH that they build, so no test calls the real `gh` or the network.
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "dev" / "session_mode.py"

spec = importlib.util.spec_from_file_location("session_mode", SCRIPT)
session_mode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session_mode)


def make_kb(folder: Path) -> Path:
    (folder / "maintainer").mkdir(parents=True)
    (folder / "maintainer" / "START.md").write_text("# start\n", encoding="utf-8")
    return folder


@pytest.fixture
def world(tmp_path):
    """A repo folder with a sibling place for a KB, an empty home, and an empty environment."""
    repo = tmp_path / "work" / "topopt"
    repo.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    return repo, home, {}


CONTRIBUTOR = [session_mode.CONTRIBUTOR_LINE, session_mode.CLONE_HINT]      # the output when the KB is not found


def fake_gh(folder: Path, body: str) -> Path:
    """A folder with an executable `gh` (a shell script). It records its arguments in `args.txt` beside it."""
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "gh"
    script.write_text('#!/bin/sh\necho "$@" > "${0%/*}/args.txt"\n' + body + "\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return folder


def with_tools(folder: Path) -> str:
    """A PATH with the fake `gh` first, and the system tools after it (a fake script may call `sleep`)."""
    return f"{folder}:/usr/bin:/bin"


def lines_of(world, **env):
    repo, home, base_env = world
    return session_mode.mode_lines(repo, {**base_env, **env}, home)


def test_no_kb_anywhere_is_contributor_and_the_second_line_tells_a_maintainer_how_to_get_the_kb(world):
    lines = lines_of(world)
    assert lines == CONTRIBUTOR
    assert lines[0].startswith("Mode: contributor.") and "CONTRIBUTING.md" in lines[0]
    assert lines[1] == ("If you are a member of the chartergroup GitHub organization, clone the knowledge base next to this repo "
                        "(gh repo clone chartergroup/topopt-kb ../topopt-kb) or set TOPOPT_KB, then start a new session.")


def test_the_maintainer_line_has_no_clone_hint(world):
    repo, _, _ = world
    make_kb(repo.parent / "topopt-kb")
    (line,) = lines_of(world)
    assert "gh repo clone" not in line and "Mode: contributor" not in line


def test_the_sibling_clone_is_found(world):
    repo, _, _ = world
    kb = make_kb(repo.parent / "topopt-kb")
    lines = lines_of(world)
    assert len(lines) == 1
    assert lines[0].startswith(f"Mode: maintainer. Knowledge base: {kb}")
    assert f"Read {kb / 'maintainer' / 'START.md'} before you plan or change code." in lines[0]


def test_the_environment_variable_wins_over_the_sibling(world, tmp_path):
    repo, _, _ = world
    make_kb(repo.parent / "topopt-kb")
    mine = make_kb(tmp_path / "elsewhere" / "kb")
    (line,) = lines_of(world, TOPOPT_KB=str(mine))
    assert f"Knowledge base: {mine} " in line or f"Knowledge base: {mine}." in line
    assert str(repo.parent / "topopt-kb") not in line


def test_the_path_file_is_found_and_the_environment_variable_wins_over_it(world, tmp_path):
    repo, home, _ = world
    from_file = make_kb(tmp_path / "from-file")
    (home / ".config" / "topopt").mkdir(parents=True)
    (home / ".config" / "topopt" / "kb-path").write_text(f"\n  {from_file}  \nignored second line\n", encoding="utf-8")
    (line,) = lines_of(world)
    assert line.startswith(f"Mode: maintainer. Knowledge base: {from_file}")
    from_env = make_kb(tmp_path / "from-env")
    (line,) = lines_of(world, TOPOPT_KB=str(from_env))
    assert line.startswith(f"Mode: maintainer. Knowledge base: {from_env}")


def test_the_path_file_lives_below_xdg_config_home_when_that_is_set(world, tmp_path):
    repo, home, _ = world
    kb = make_kb(tmp_path / "kb-xdg")
    config = tmp_path / "xdg"
    (config / "topopt").mkdir(parents=True)
    (config / "topopt" / "kb-path").write_text(str(kb), encoding="utf-8")
    (line,) = lines_of(world, XDG_CONFIG_HOME=str(config))
    assert f"Knowledge base: {kb}" in line
    assert lines_of(world) == CONTRIBUTOR          # without the variable the file is not in ~/.config


def test_a_relative_path_in_the_path_file_starts_at_the_folder_of_the_file(world, tmp_path):
    repo, home, _ = world
    (home / ".config" / "topopt").mkdir(parents=True)
    kb = make_kb(home / ".config" / "topopt" / "kb-copy")
    (home / ".config" / "topopt" / "kb-path").write_text("kb-copy\n", encoding="utf-8")
    (line,) = lines_of(world)
    assert f"Knowledge base: {kb}" in line


def test_a_folder_without_the_start_page_is_not_a_kb(world):
    repo, _, _ = world
    (repo.parent / "topopt-kb" / "maintainer").mkdir(parents=True)          # the folder exists, the page does not
    assert lines_of(world) == CONTRIBUTOR
    (repo.parent / "topopt-kb" / "maintainer" / "START.md").mkdir()          # a folder in the place of the page
    assert lines_of(world) == CONTRIBUTOR


def test_a_broken_link_in_the_place_of_the_start_page_is_not_a_kb(world, tmp_path):
    repo, _, _ = world
    (repo.parent / "topopt-kb" / "maintainer").mkdir(parents=True)
    (repo.parent / "topopt-kb" / "maintainer" / "START.md").symlink_to(tmp_path / "gone.md")
    assert lines_of(world) == CONTRIBUTOR


def test_an_unreadable_start_page_is_not_a_kb_and_the_user_sees_which_setting_failed(world, tmp_path):
    kb = make_kb(tmp_path / "locked")
    page = kb / "maintainer" / "START.md"
    page.chmod(0)
    try:
        if os.access(page, os.R_OK):
            pytest.skip("this process can read a file with mode 0 (it runs as root)")
        lines = lines_of(world, TOPOPT_KB=str(kb))
    finally:
        page.chmod(0o644)
    assert lines[0] == session_mode.CONTRIBUTOR_LINE and lines[-1] == session_mode.CLONE_HINT
    assert len(lines) == 3 and lines[1].startswith("Note: TOPOPT_KB names ") and "is not readable" in lines[1]
    assert str(page) in lines[1]


def test_a_bad_environment_variable_falls_through_to_the_next_place(world, tmp_path):
    repo, _, _ = world
    kb = make_kb(repo.parent / "topopt-kb")
    (line,) = lines_of(world, TOPOPT_KB=str(tmp_path / "does-not-exist"))
    assert f"Knowledge base: {kb}" in line


def test_a_missing_kb_that_the_user_named_in_the_path_file_gets_a_note(world, tmp_path):
    _, home, _ = world
    (home / ".config" / "topopt").mkdir(parents=True)
    (home / ".config" / "topopt" / "kb-path").write_text(str(tmp_path / "gone"), encoding="utf-8")
    lines = lines_of(world)
    assert lines[0] == session_mode.CONTRIBUTOR_LINE and lines[-1] == session_mode.CLONE_HINT and len(lines) == 3
    assert "kb-path" in lines[1] and "is not readable" in lines[1]


def test_an_empty_environment_variable_counts_as_not_set(world):
    assert lines_of(world, TOPOPT_KB="  ") == CONTRIBUTOR


def test_the_date_of_the_last_commit_is_shown_when_the_kb_is_a_git_clone(world):
    repo, _, _ = world
    kb = make_kb(repo.parent / "topopt-kb")
    git = ["git", "-C", str(kb), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    try:
        subprocess.run([*git, "init", "-q"], check=True, capture_output=True)
        subprocess.run([*git, "add", "."], check=True, capture_output=True)
        subprocess.run([*git, "commit", "-q", "-m", "x", "--date=2026-10-23T12:00:00+0000"], check=True, capture_output=True,
                       env=dict(os.environ, GIT_COMMITTER_DATE="2026-10-23T12:00:00+0000"))
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"git is not usable here: {error}")
    (line,) = lines_of(world)
    assert "(last commit 2026-10-23)" in line


def test_no_date_when_the_kb_is_not_a_git_clone(world):
    repo, _, _ = world
    make_kb(repo.parent / "topopt-kb")
    (line,) = lines_of(world)
    assert "last commit" not in line and line.startswith("Mode: maintainer. Knowledge base: ")


# --- a maintainer without the KB clone: the only network call, through `gh` ---------------------------------------------

def test_an_active_member_without_the_kb_gets_the_maintainer_line_and_the_clone_line(world, tmp_path):
    bin_dir = fake_gh(tmp_path / "bin", 'echo active')
    lines = lines_of(world, PATH=with_tools(bin_dir))
    assert lines == [session_mode.MEMBER_LINE, session_mode.MEMBER_CLONE_LINE]
    assert lines[0] == "Mode: maintainer (knowledge base not found)."
    assert "gh repo clone chartergroup/topopt-kb ../topopt-kb" in lines[1] and "TOPOPT_KB" in lines[1]
    assert lines[1].endswith("then start a new session after the clone.")
    assert (bin_dir / "args.txt").read_text().split() == ["api", "user/memberships/orgs/chartergroup", "--jq", ".state"]


def test_a_member_whose_own_setting_failed_also_sees_the_note(world, tmp_path):
    bin_dir = fake_gh(tmp_path / "bin", 'echo active')
    lines = lines_of(world, PATH=with_tools(bin_dir), TOPOPT_KB=str(tmp_path / "gone"))
    assert lines[0] == session_mode.MEMBER_LINE and lines[-1] == session_mode.MEMBER_CLONE_LINE
    assert len(lines) == 3 and lines[1].startswith("Note: TOPOPT_KB names ")


@pytest.mark.parametrize("body", [
    'echo pending',                                   # invited, not yet a member
    'echo "gh: Not Found (HTTP 404)" >&2; exit 1',    # not a member
    'echo "To get started with GitHub CLI, please run: gh auth login" >&2; exit 4',     # not logged in
    'echo active; exit 1',                            # a failed call is never "active", whatever it printed
    'echo',                                           # empty answer
    'echo "ACTIVE-ish"',
], ids=["pending", "not-a-member", "not-logged-in", "failed-exit", "empty", "other-text"])
def test_every_answer_but_active_is_the_contributor_output(world, tmp_path, body):
    bin_dir = fake_gh(tmp_path / "bin", body)
    assert lines_of(world, PATH=with_tools(bin_dir)) == CONTRIBUTOR


def test_no_gh_on_the_path_is_the_contributor_output(world, tmp_path):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    assert lines_of(world, PATH=str(empty)) == CONTRIBUTOR
    assert lines_of(world) == CONTRIBUTOR             # no PATH at all in the environment: no call


def test_a_gh_that_hangs_is_stopped_after_the_timeout(world, tmp_path, monkeypatch):
    monkeypatch.setattr(session_mode, "GH_TIMEOUT_S", 0.3)
    bin_dir = fake_gh(tmp_path / "bin", 'exec sleep 30')
    started = time.monotonic()
    assert lines_of(world, PATH=with_tools(bin_dir)) == CONTRIBUTOR
    assert 0.25 <= time.monotonic() - started < 10          # it waited for the timeout, and it did not wait for the sleep


def test_a_gh_that_hangs_with_a_child_process_is_stopped_too(world, tmp_path, monkeypatch):
    """The child of the script holds the pipe open. The whole process group is killed, so the call still ends."""
    monkeypatch.setattr(session_mode, "GH_TIMEOUT_S", 0.3)
    bin_dir = fake_gh(tmp_path / "bin", 'sleep 30 & wait')
    started = time.monotonic()
    assert lines_of(world, PATH=with_tools(bin_dir)) == CONTRIBUTOR
    assert 0.25 <= time.monotonic() - started < 10          # it waited for the timeout, and it did not wait for the sleep


def test_a_gh_that_cannot_run_is_the_contributor_output(world, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text("not a program\n", encoding="utf-8")
    (bin_dir / "gh").chmod(0o755)                     # no shebang and no binary: the exec fails or the shell fails
    assert lines_of(world, PATH=with_tools(bin_dir)) == CONTRIBUTOR


def test_when_the_kb_is_found_gh_is_not_called(world, tmp_path):
    repo, _, _ = world
    make_kb(repo.parent / "topopt-kb")
    bin_dir = fake_gh(tmp_path / "bin", 'echo active')
    (line,) = lines_of(world, PATH=with_tools(bin_dir))
    assert line.startswith("Mode: maintainer. Knowledge base: ")
    assert not (bin_dir / "args.txt").exists()        # no network call when the KB is found


def test_the_timeout_is_one_named_constant_and_the_hook_waits_longer_than_it():
    assert session_mode.GH_TIMEOUT_S == 10                  # a plain constant, no override from the environment
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("GH_TIMEOUT_S = ") == 1
    assert not any("GH_TIMEOUT" in line and "environ" in line for line in text.splitlines())
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    hook = settings["hooks"]["SessionStart"][0]["hooks"][0]
    assert hook["timeout"] >= session_mode.GH_TIMEOUT_S + 3  # the hook limit must not cut the gh call and the git call short


def run_script(env_extra, home, cwd):
    """Run the script as the hook does. The environment is clean, so nothing of the real home folder, KB or `gh` leaks in."""
    empty = Path(home) / "empty-bin"
    empty.mkdir(exist_ok=True)
    env = {"PATH": str(empty), "HOME": str(home), **env_extra}
    return subprocess.run([sys.executable, "-I", str(SCRIPT)], capture_output=True, text=True, env=env, cwd=cwd, timeout=60)


def test_the_script_prints_the_contributor_line_and_exits_with_0(tmp_path):
    done = run_script({}, tmp_path, tmp_path)
    assert done.returncode == 0 and done.stderr == ""
    # the sibling of the repo folder under test could hold a real KB clone on a maintainer machine: accept both modes
    assert done.stdout.startswith(("Mode: contributor.", "Mode: maintainer."))
    if done.stdout.startswith("Mode: contributor."):
        assert done.stdout.rstrip("\n").endswith(session_mode.CLONE_HINT)


def test_the_script_in_a_folder_tree_with_a_kb_names_the_kb(tmp_path):
    repo = tmp_path / "src" / "topopt"
    (repo / "scripts" / "dev").mkdir(parents=True)
    copy = repo / "scripts" / "dev" / "session_mode.py"
    copy.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    kb = make_kb(tmp_path / "src" / "topopt-kb")
    env = {"PATH": str(tmp_path), "HOME": str(tmp_path / "nohome")}      # no git and no gh on this PATH: no date, no call
    done = subprocess.run([sys.executable, "-I", str(copy)], capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60)
    assert done.returncode == 0
    assert done.stdout.startswith(f"Mode: maintainer. Knowledge base: {kb}")


def test_any_error_inside_the_lookup_prints_the_contributor_line_and_exits_with_0(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise ValueError("broken")
    monkeypatch.setattr(session_mode, "mode_lines", boom)
    assert session_mode.main() == 0
    assert capsys.readouterr().out.splitlines() == CONTRIBUTOR


def test_no_home_folder_is_known(monkeypatch):
    def no_home():
        raise RuntimeError("no home")
    monkeypatch.setattr(session_mode.Path, "home", staticmethod(no_home))
    assert session_mode.user_home() == Path(os.devnull)


def test_the_hook_in_the_project_settings_runs_this_script_and_cannot_fail_the_session():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    (group,) = settings["hooks"]["SessionStart"]
    (hook,) = group["hooks"]
    assert hook["type"] == "command"
    assert "scripts/dev/session_mode.py" in hook["command"] and "$CLAUDE_PROJECT_DIR" in hook["command"]
    assert "|| echo " in hook["command"]            # no python3 on the machine: the session still gets the line
    assert "Mode: contributor." in hook["command"] and session_mode.CLONE_HINT.split(" (")[0] in hook["command"]
    # every source that empties the context of the model: a new start, a resume, /clear and a compaction (not "fork": it keeps the context)
    assert set(group["matcher"].split("|")) == {"startup", "resume", "clear", "compact"}


def test_the_script_runs_on_the_oldest_python_that_a_contributor_can_have():
    """macOS ships Python 3.9. The file has no syntax or import that needs a newer one."""
    import ast
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
              {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules <= {"__future__", "os", "shutil", "signal", "subprocess", "sys", "pathlib"}
    assert "from __future__ import annotations" in SCRIPT.read_text(encoding="utf-8")
