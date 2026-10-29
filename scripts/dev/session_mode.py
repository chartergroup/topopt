#!/usr/bin/env python3
"""Print the mode in which an agent works in this repo: maintainer or contributor.

Claude Code runs this script at session start (the SessionStart hook in `.claude/settings.json`).
A client without hooks (Codex, Cursor, Gemini CLI, ...) runs it by hand from the repo root:

    python3 scripts/dev/session_mode.py

maintainer:  a clone of the private knowledge base (KB) is on this machine, and `maintainer/START.md` in it is readable.
contributor: anything else. This is the normal mode of someone who has no access to the KB.

The script looks for the KB in this order. The first folder that passes wins:
  1. the folder in the environment variable TOPOPT_KB
  2. the sibling clone <repo root>/../topopt-kb
  3. the folder in the first line of the file $XDG_CONFIG_HOME/topopt/kb-path (default ~/.config/topopt/kb-path)

Three outcomes:
  - The KB is found: `Mode: maintainer. Knowledge base: <path> ...`. One line. No network call.
  - The KB is not found, and `gh` says that you are an active member of the organization chartergroup:
    `Mode: maintainer (knowledge base not found).` and a line that tells you to clone the KB and start a new session.
    This is the only network call. It runs only when the KB is not found, and it stops after GH_TIMEOUT_S seconds.
  - Anything else (no `gh`, not logged in, not a member, a timeout, any error): the contributor line, a note for each
    folder that you set yourself (1 or 3) and that did not pass, and the clone hint.
It uses the standard library only (Python 3.8 or newer) and never fails or hangs: any error prints the contributor
output and exits with 0.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

START_PAGE = Path("maintainer") / "START.md"
ENV_NAME = "TOPOPT_KB"
PATH_FILE = Path("topopt") / "kb-path"          # below the user's config folder
SIBLING = "topopt-kb"                           # a clone next to the repo folder
ORG = "chartergroup"
GH_TIMEOUT_S = 10                               # a slow network must not block the session start for long; one call per session start, only when the KB is not found

CONTRIBUTOR_LINE = ("Mode: contributor. To change this repo, follow CONTRIBUTING.md. To make a part, use AGENTS.md. "
                    "The private knowledge base and the board are for maintainers only.")
CLONE_HINT = ("If you are a member of the chartergroup GitHub organization, clone the knowledge base next to this repo "
              "(gh repo clone chartergroup/topopt-kb ../topopt-kb) or set TOPOPT_KB, then start a new session.")
MEMBER_LINE = "Mode: maintainer (knowledge base not found)."
MEMBER_CLONE_LINE = ("Clone the knowledge base next to this repo (gh repo clone chartergroup/topopt-kb ../topopt-kb) "
                     "or set TOPOPT_KB, then start a new session after the clone.")


def is_readable(path: Path) -> bool:
    """True if `path` is a file that this process can open and read."""
    try:
        with open(path, "rb") as handle:
            handle.read(1)
        return path.is_file()
    except OSError:
        return False


def config_dir(env: "dict[str, str]", home: Path) -> Path:
    base = env.get("XDG_CONFIG_HOME", "").strip()
    return Path(base) if base and os.path.isabs(base) else home / ".config"


def candidates(repo_root: Path, env: "dict[str, str]", home: Path) -> "list[tuple[str, Path]]":
    """(label, folder) in the order of the lookup. A label that starts with `sibling` is not set by the user."""
    found = []
    from_env = env.get(ENV_NAME, "").strip()
    if from_env:
        found.append((ENV_NAME, Path(os.path.abspath(os.path.expanduser(from_env)))))
    found.append(("sibling clone", Path(os.path.abspath(repo_root.parent / SIBLING))))
    path_file = config_dir(env, home) / PATH_FILE
    try:
        first = next((line.strip() for line in path_file.read_text(encoding="utf-8").splitlines() if line.strip()), "")
    except (OSError, UnicodeError):
        first = ""
    if first:
        folder = Path(os.path.expanduser(first))
        if not folder.is_absolute():
            folder = path_file.parent / folder
        found.append((str(path_file), Path(os.path.abspath(folder))))
    return found


def find_kb(repo_root: Path, env: "dict[str, str]", home: Path) -> "tuple[Path | None, list[str]]":
    """The first KB folder whose `maintainer/START.md` is readable, and notes on the folders the user set that failed."""
    notes = []
    for label, folder in candidates(repo_root, env, home):
        if is_readable(folder / START_PAGE):
            return folder, []
        if not label.startswith("sibling"):
            notes.append(f"Note: {label} names {folder}, and {folder / START_PAGE} is not readable.")
    return None, notes


def last_commit_date(kb: Path) -> "str | None":
    """The date (YYYY-MM-DD) of the last commit of the KB clone, or None if git cannot tell. No network."""
    try:
        done = subprocess.run(["git", "-C", str(kb), "log", "-1", "--format=%cs"], capture_output=True, text=True,
                              timeout=3, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    except (OSError, subprocess.SubprocessError):
        return None
    text = done.stdout.strip()
    return text if done.returncode == 0 and text else None


def is_org_member(env: "dict[str, str]") -> bool:
    """True if `gh` says that the logged-in user is an active member of ORG. False for every other result.

    The only network call of this script. It makes no prompt, stops after GH_TIMEOUT_S seconds, and kills the
    process group of `gh` on a timeout. The `gh` that is used is the first one on the PATH in `env`.
    """
    try:
        gh = shutil.which("gh", path=env.get("PATH", ""))
        if not gh:
            return False
        child = subprocess.Popen([gh, "api", f"user/memberships/orgs/{ORG}", "--jq", ".state"], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                 env=dict(env, GH_PROMPT_DISABLED="1", GH_NO_UPDATE_NOTIFIER="1", NO_COLOR="1"),
                                 start_new_session=True)
        try:
            out, _ = child.communicate(timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except (OSError, AttributeError):
                child.kill()
            try:
                child.communicate(timeout=1)
            except Exception:
                pass
            return False
        return child.returncode == 0 and (out or "").strip() == "active"
    except Exception:                                   # no gh, not logged in, no network, anything: not a member
        return False


def mode_lines(repo_root: Path, env: "dict[str, str]", home: Path) -> "list[str]":
    kb, notes = find_kb(repo_root, env, home)
    if kb is None:
        if is_org_member(env):
            return [MEMBER_LINE, *notes, MEMBER_CLONE_LINE]
        return [CONTRIBUTOR_LINE, *notes, CLONE_HINT]
    date = last_commit_date(kb)
    since = f" (last commit {date})" if date else ""
    return [f"Mode: maintainer. Knowledge base: {kb}{since}. Read {kb / START_PAGE} before you plan or change code."]


def user_home() -> Path:
    try:
        return Path.home()
    except (RuntimeError, KeyError, OSError):           # no home folder is known: the lookup 3 finds nothing
        return Path(os.devnull)


def main() -> int:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        lines = mode_lines(repo_root, dict(os.environ), user_home())
    except Exception:                                   # the session must start: any error means contributor
        lines = [CONTRIBUTOR_LINE, CLONE_HINT]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
