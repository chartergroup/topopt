"""`topopt doctor`: what this install is, what is wrong with it, and the commands that fix it (CONTRACTS.md section 8).

    topopt doctor [problem.yaml] [--json]

Read-only. It installs, builds, fetches and deletes nothing. It gives facts first, then the offers
(`environment_data()["recommendations"]`), then the commands for the user to run. It reports:

- the checkout: the code folder, git or not, the revision, the branch, uncommitted changes, and how far the
  checkout is ahead or behind `origin/<default branch>` AS OF THE LAST FETCH (the age of `FETCH_HEAD` is shown);
- the machine and the solver backends (`topopt.environment.environment_data`; a failure there is a line of the
  report, not a crash);
- an import test of the core packages and of the optional ones, each in a child Python, so a cut install shows as
  a line of the report (the way `tools/platform_probe.py` does it);
- the known environments (this one, the `.venv` of the checkout, the environment of `scripts/topopt-env.sh`) and
  which client uses which.

`checkout_facts` is also called by `environment_data` (the MCP tool `environment`): one function, no copy.

Light module: it imports only the standard library at the top. `import topopt.cli` loads only PyYAML today, and
the command must start in an environment with a broken scipy. `topopt.environment` is imported inside `collect`.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
from pathlib import Path

from .uv_command import uv_command

REPO_URL = "https://github.com/chartergroup/topopt.git"
# Seconds for a read-only helper command (git, a version probe of another Python). Recorded default, not a user setting.
QUICK_TIMEOUT_S = 20
# Seconds for the import test of one package in a child Python (the value of tools/platform_probe.py). `--timeout-s` sets it.
IMPORT_TIMEOUT_S = 90

# (name, import statement, distribution, use[, show the version]). The core rows are the dependencies of pyproject.toml.
# The row "MCP server" tests topopt_mcp.server; the version of its distribution (mcp) is the version of the row `mcp`.
IMPORT_CHECKS = (
    ("numpy", "numpy", "numpy", "core"),
    ("scipy", "scipy.ndimage, scipy.sparse.linalg", "scipy", "core"),
    ("numba", "numba", "numba", "core"),
    ("PyYAML", "yaml", "PyYAML", "core"),
    ("scikit-image", "skimage", "scikit-image", "core"),
    ("trimesh", "trimesh", "trimesh", "core"),
    ("manifold3d", "manifold3d", "manifold3d", "core"),
    ("pyamg", "pyamg", "pyamg", "core"),
    ("rtree", "rtree", "rtree", "core"),
    ("networkx", "networkx", "networkx", "core"),
    ("Pillow", "PIL", "Pillow", "core"),
    ("mcp", "mcp", "mcp", "extra mcp"),
    ("MCP server", "topopt_mcp.server", "mcp", "extra mcp", False),
    ("pyopencl", "pyopencl", "pyopencl", "extra gpu"),
    ("pypardiso", "pypardiso", "pypardiso", "extra pardiso"),
    ("gmsh", "gmsh", "gmsh", "extra tet"),
    ("cholespy", "cholespy", "cholespy", "extra verify"),
    ("scikit-fem", "skfem", "scikit-fem", "extra verify"),
    ("OpenCASCADE (OCP)", "OCP", "cadquery-ocp-novtk", "extra step"),
)

_VERSION_CODE = """
import importlib.metadata as m, sys
try:
    v = m.version(sys.argv[1])
except Exception:
    v = ''
print('VERSION=' + v, flush=True)
import {statement}
"""

# The file in the venv of the launcher that makes `scripts/topopt-env.sh sync` install the packages again.
LAUNCHER_MARKER = ".topopt-step2"

# What a client config file is, and where it lives in the checkout (the plugin manifest is a client config too).
CLIENT_FILES = (
    (".mcp.json", "Claude Code or VS Code in the repo folder"),
    (".claude-plugin/plugin.json", "Claude Code plugin"),
    (".cursor/mcp.json", "Cursor"),
    (".codex/config.toml", "Codex"),
)


def _q(text) -> str:
    return shlex.quote(str(text))


def utc_text(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human_age(seconds: float) -> str:
    """`5 min`, `3 h`, `2 days`: a short age for the text report."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 129600:
        return f"{seconds / 3600:.0f} h"
    return f"{seconds / 86400:.0f} days"


def _run(cmd, timeout=QUICK_TIMEOUT_S, cwd=None, env=None):
    """`(returncode, stdout, stderr)` of a command, or `(None, "", text)` when it did not start or took too long."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, "", f"took more than {timeout} s"
    except OSError as exc:
        return None, "", f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout, done.stderr


# --- the checkout -------------------------------------------------------------------------------------------------


def _git_env() -> dict:
    # GIT_OPTIONAL_LOCKS=0: `git status` does not write the index. LC_ALL=C: the same words on every machine.
    return {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"}


def checkout_facts(root=None) -> dict:
    """Facts about the code folder: git or not, revision, branch, changes, ahead or behind origin as of the last fetch.

    Read-only: the git commands only read, and nothing is fetched. `behind` counts the commits of the LAST FETCH, so
    `last_fetch_utc` and `last_fetch_age_s` (the newest time of the `FETCH_HEAD` files of the repository: git writes
    one in the git folder of the worktree where `git fetch` ran) say how old that is. A folder without `.git` (a plugin copy, a ZIP, a wheel install)
    gives `git: false` and no git fact. The function never raises.
    """
    root = Path(root).resolve() if root else Path(__file__).resolve().parent.parent
    git = (root / ".git").exists()
    has_pyproject = (root / "pyproject.toml").is_file()
    out = {
        "code_folder": str(root),
        "kind": ("git checkout" if git else "source folder without git (a plugin copy or a ZIP download)"
                 if has_pyproject else "installed package (no pyproject.toml and no tools/ folder here)"),
        "git": git, "has_tools": (root / "tools").is_dir(), "has_pyproject": has_pyproject,
        "head": None, "head_date": None, "branch": None, "detached": None,
        "dirty": None, "changed_files": None, "untracked_files": None,
        "default_branch_ref": None, "ahead": None, "behind": None,
        "last_fetch_utc": None, "last_fetch_age_s": None,
        "fetch_command": None, "update_command": None, "notes": [],
    }
    if not git:
        out["notes"].append("This folder has no git data. Get the new version the way you got this one (update the plugin, "
                            f"or clone again: git clone {REPO_URL}).")
        return out
    try:
        _fill_git_facts(root, out)
    except Exception as exc:    # a report must not fail because of one odd git state
        out["notes"].append(f"git facts stopped: {type(exc).__name__}: {exc}")
    return out


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _which_git():
    return shutil.which("git")


def _git_unusable():
    """On macOS, `/usr/bin/git` is a stub when the Command Line Tools are missing: it opens an install dialog. Return a note
    that says so (and then no git command runs), else None. `xcode-select -p` only reads."""
    if not _is_macos() or _which_git() != "/usr/bin/git":
        return None
    rc, _, _ = _run(["xcode-select", "-p"])
    if rc == 0:
        return None
    return ("git is /usr/bin/git, and the Command Line Developer Tools are not installed (`xcode-select -p` fails). "
            "A git command would open an install dialog, so the git facts are skipped. To install the tools: "
            "xcode-select --install")


def _fetch_times(root: Path, git) -> list:
    """The modification times of the `FETCH_HEAD` files that a fetch of this repository could have written.

    git writes `FETCH_HEAD` in the git folder of the worktree where `git fetch` ran, so a linked worktree has its own file
    (`<common>/worktrees/<name>/FETCH_HEAD`) beside the common one. The remote-tracking refs are shared, so the newest
    of all these files is the last fetch.
    """
    paths = []
    rc, text, _ = git("rev-parse", "--git-path", "FETCH_HEAD")
    if rc == 0 and text.strip():
        paths.append(root / text.strip())      # an absolute path stays as it is
    rc, text, _ = git("rev-parse", "--git-common-dir")
    if rc == 0 and text.strip():
        common = root / text.strip()
        paths.append(common / "FETCH_HEAD")
        paths.extend(sorted(common.glob("worktrees/*/FETCH_HEAD")))
    times = []
    for path in paths:
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            pass
    return times


def _worktree_of(branch: str, git):
    """The folder of the worktree that has `branch` checked out, or None (`git worktree list --porcelain`, read-only)."""
    rc, text, _ = git("worktree", "list", "--porcelain")
    if rc != 0:
        return None
    folder = None
    for line in text.splitlines():
        if line.startswith("worktree "):
            folder = line[len("worktree "):]
        elif folder and line == f"branch refs/heads/{branch}":
            return folder
    return None


def _fill_git_facts(root: Path, out: dict) -> None:
    unusable = _git_unusable()
    if unusable:
        out["notes"].append(unusable)
        return
    env = _git_env()

    def git(*args):
        return _run(["git", "-C", str(root), *args], env=env)

    rc, text, err = git("rev-parse", "--short", "HEAD")
    if rc is None or rc != 0:
        out["notes"].append(f"git does not answer: {err.strip().splitlines()[-1] if err.strip() else 'no output'}")
        return
    out["head"] = text.strip()
    rc, text, _ = git("log", "-1", "--format=%cI")
    out["head_date"] = text.strip() or None if rc == 0 else None
    rc, text, _ = git("symbolic-ref", "-q", "--short", "HEAD")
    out["detached"] = rc != 0
    out["branch"] = text.strip() if rc == 0 else None
    rc, text, _ = git("status", "--porcelain=v1")
    if rc == 0:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        out["untracked_files"] = sum(1 for ln in lines if ln.startswith("??"))
        out["changed_files"] = len(lines) - out["untracked_files"]
        out["dirty"] = out["changed_files"] > 0
    # The default branch: what origin/HEAD points to, else origin/main or origin/master.
    ref = None
    rc, text, _ = git("symbolic-ref", "-q", "--short", "refs/remotes/origin/HEAD")
    if rc == 0 and text.strip():
        ref = text.strip()
    else:
        for cand in ("origin/main", "origin/master"):
            if git("rev-parse", "--verify", "-q", f"refs/remotes/{cand}")[0] == 0:
                ref = cand
                break
    out["default_branch_ref"] = ref
    if ref:
        rc, text, _ = git("rev-list", "--left-right", "--count", f"HEAD...{ref}")
        if rc == 0 and len(text.split()) == 2:
            out["ahead"], out["behind"] = (int(n) for n in text.split())
    else:
        out["notes"].append("There is no origin/main or origin/master: ahead and behind are not known.")
    times = _fetch_times(root, git)
    if times:
        out["last_fetch_utc"], out["last_fetch_age_s"] = utc_text(max(times)), max(0.0, time.time() - max(times))
    else:
        out["notes"].append("There is no FETCH_HEAD file: no fetch since the clone. The numbers are as of the clone.")
    out["fetch_command"] = f"git -C {_q(root)} fetch origin"
    base = (ref or "origin/main").split("/", 1)[-1]
    pull = f"git -C {_q(root)} pull --ff-only"
    rc, text, _ = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    tracks_default = bool(ref) and rc == 0 and text.strip() == ref
    if out["branch"] == base or (tracks_default and out["ahead"] == 0):
        out["update_command"] = pull        # the branch is, or follows, the default branch: pull moves it forward
        return
    # `behind` counts against the default branch, so the update command follows that branch. Another worktree can hold it
    # (git refuses `switch` then); the command runs in that worktree. `git switch` refuses if a change would be lost.
    where = _worktree_of(base, git)
    merge = f"merge origin/{base} into it (git -C {_q(root)} merge origin/{base})"
    if where and Path(where).resolve() != root:
        out["update_command"] = f"git -C {_q(where)} pull --ff-only"
        out["notes"].append(f"{base} is checked out in another worktree, {where}. The update command runs there. "
                            f"To bring this branch up to date, {merge}.")
        return
    out["update_command"] = f"git -C {_q(root)} switch {base} && {pull}"
    if out["detached"]:
        out["notes"].append(f"HEAD is detached (not on a branch). The update command switches to {base} first.")
    else:
        out["notes"].append(f"This checkout is on the branch {out['branch']}, not {base}. The update command switches to {base} "
                            f"first. To keep this branch, {merge} instead.")


# --- the import test ----------------------------------------------------------------------------------------------


def _top_names(statement: str) -> set:
    """The top-level module names of an import statement: `scipy.ndimage, scipy.sparse.linalg` gives `{scipy}`."""
    return {part.strip().split(".")[0] for part in statement.split(",") if part.strip()}


def import_check(name: str, statement: str, dist: str, use: str, python: str = None,
                 timeout: float = IMPORT_TIMEOUT_S, cwd: str = None, code_folder: str = None,
                 show_version: bool = True) -> dict:
    """Import `statement` in a child Python. Return one row.

    `status`: `ok`; `not installed` (the module is missing and the metadata has no such package);
    `cut` (the module is missing but the metadata lists the package: an install that was cut, M-2);
    `failed` (the import raises another error, for example a missing library); `crashed` (a signal); `timeout`.
    The module that is missing can be the package itself (`dist`) and not a name in `statement`: `topopt_mcp.server`
    needs `mcp`, so a venv without the `mcp` extra gives `not installed` for that row, as for the row `mcp`.
    `show_version` False: the row has no version (the version of `dist` is the version of another row).
    """
    row = _import_check(name, statement, dist, use, python, timeout, cwd, code_folder)
    if not show_version:
        row["version"] = None
    return row


def _import_check(name, statement, dist, use, python, timeout, cwd, code_folder) -> dict:
    python = python or sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if code_folder:    # the child sees the code the way this process sees it
        env["PYTHONPATH"] = os.pathsep.join([code_folder] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    row = {"name": name, "import": statement, "package": dist, "use": use, "version": None,
           "status": "ok", "detail": "", "seconds": None}
    t0 = time.perf_counter()
    rc, out, err = _run([python, "-B", "-c", _VERSION_CODE.format(statement=statement), dist],
                        timeout=timeout, cwd=cwd, env=env)
    row["seconds"] = round(time.perf_counter() - t0, 2)
    version = next((ln[len("VERSION="):] for ln in out.splitlines() if ln.startswith("VERSION=")), "")
    row["version"] = version or None
    if rc == 0:
        return row
    lines = [ln for ln in err.strip().splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    if rc is None:
        row["status"] = "timeout" if "took more than" in err else "failed"
        row["detail"] = err.strip()
    elif rc < 0:
        row.update(status="crashed", detail=f"stopped by signal {-rc}" + (f": {last}" if last else ""))
    else:
        missing = re.search(r"No module named '([^']+)'", last) if "ModuleNotFoundError" in last else None
        gone = missing.group(1).split(".")[0] if missing else None
        if gone and (gone in _top_names(statement) or gone == dist.lower().replace("-", "_")):
            if row["version"]:
                row.update(status="cut", detail=f"the files are missing, but the metadata lists {dist} {row['version']}: "
                                                f"the install was cut. {last}")
            else:
                row.update(status="not installed", detail=last)
        else:
            row.update(status="failed", detail=last or f"exit code {rc}")
    return row


def _progress_line(row: dict, k: int, n: int) -> str:
    """The progress line of one finished import check: `  [3/19] numba: ok (1.2 s)`.
    A row of an optional package names its extra: `  [15/19] pypardiso (extra pardiso): not installed (0.1 s)`."""
    seconds = row.get("seconds")
    took = f" ({seconds:g} s)" if seconds is not None else ""
    use = row.get("use")
    extra = f" ({use})" if use and use != "core" else ""
    return f"  [{k}/{n}] {row['name']}{extra}: {row['status']}{took}"


def import_checks(python: str = None, timeout: float = IMPORT_TIMEOUT_S, code_folder: str = None, progress=None) -> list:
    """The rows of `IMPORT_CHECKS`, each tested in its own child Python (a few at a time).

    The rows come back in the order of `IMPORT_CHECKS`. `progress` (a callable that takes one line of text, or None) is called
    once for each row as soon as that row ends, so in the order of the ends: `[k/N] name: status (seconds s)`.
    """
    with tempfile.TemporaryDirectory(prefix="topopt-doctor-") as tmp:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(import_check, *row[:4], python=python, timeout=timeout, cwd=tmp, code_folder=code_folder,
                                   show_version=row[4] if len(row) > 4 else True)
                       for row in IMPORT_CHECKS]
            if progress is not None:
                for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    progress(_progress_line(future.result(), k, len(futures)))
            return [f.result() for f in futures]


def _extra_fits_here(use: str) -> bool:
    """Is the extra named by `use` (`extra pardiso`) one that this machine can install? pardiso: Linux x86_64 only."""
    if not use.startswith("extra "):
        return False
    if use == "extra pardiso":
        return sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64")
    return True


def is_problem(row: dict) -> bool:
    """A core package that does not import, or an optional one that is installed and does not import."""
    if row["status"] == "ok":
        return False
    if row["use"] == "core":
        return True
    return row["status"] != "not installed"


# --- the known environments and the clients -----------------------------------------------------------------------


def launcher_env_dir(root: Path) -> Path:
    """The environment folder of `scripts/topopt-env.sh`: $TOPOPT_ENV_DIR, else $EVAL_TOPOPT_ENV_DIR, else $CLAUDE_PLUGIN_DATA,
    else <repo>/.plugin-env."""
    chosen = (os.environ.get("TOPOPT_ENV_DIR") or os.environ.get("EVAL_TOPOPT_ENV_DIR")
              or os.environ.get("CLAUDE_PLUGIN_DATA"))
    return Path(os.path.abspath(chosen)) if chosen else root / ".plugin-env"


def _python_probe(python: Path, timeout=QUICK_TIMEOUT_S) -> dict:
    """Does this Python run? Which version, and which topopt folder does it import? (a child process)"""
    code = ("import sys; print('PY=' + sys.version.split()[0]); "
            "import topopt; print('TOPOPT=' + topopt.__file__)")
    rc, out, err = _run([str(python), "-B", "-c", code], timeout=timeout, cwd=tempfile.gettempdir())
    py = next((ln[3:] for ln in out.splitlines() if ln.startswith("PY=")), None)
    tp = next((ln[7:] for ln in out.splitlines() if ln.startswith("TOPOPT=")), None)
    lines = [ln for ln in err.strip().splitlines() if ln.strip()]
    return {"runs": py is not None, "python_version": py,
            "topopt_from": str(Path(tp).parent.parent) if tp else None,
            "error": None if rc == 0 else (lines[-1] if lines else err.strip() or f"exit code {rc}")}


def _client_environment(command: str, args) -> str:
    """Which environment a client config starts: `launcher` (scripts/topopt-env.sh), `.venv`, or `other`."""
    text = " ".join([str(command or "")] + [str(a) for a in (args or [])])
    if "topopt-env.sh" in text:
        return "launcher"
    if ".venv" in text:
        return ".venv"
    return "other"


def client_configs(root: Path) -> list:
    """For each client config file of the checkout: which environment its `topopt` server starts.

    Only the command and its arguments are read, never the `env` or the `headers` of a config.
    """
    rows = []
    for rel, label in CLIENT_FILES:
        path = root / rel
        if not path.is_file():
            continue
        row = {"file": rel, "client": label, "environment": "unreadable"}
        try:
            if rel.endswith(".toml"):
                server = tomllib.loads(path.read_text(encoding="utf-8")).get("mcp_servers", {}).get("topopt", {})
            else:
                server = json.loads(path.read_text(encoding="utf-8")).get("mcpServers", {}).get("topopt", {})
            row["environment"] = _client_environment(server.get("command"), server.get("args")) if server else "no topopt server"
        except (OSError, ValueError, AttributeError):
            pass
        rows.append(row)
    return rows


def known_environments(root=None, probe=_python_probe) -> dict:
    """This environment, the `.venv` of the checkout and the environment of the launcher: path, runs or not, who uses it."""
    root = Path(root).resolve() if root else Path(__file__).resolve().parent.parent
    source = (root / "pyproject.toml").is_file()
    clients = client_configs(root) if source else []
    used = {kind: [c["client"] for c in clients if c["environment"] == kind] for kind in ("launcher", ".venv")}
    this_prefix = Path(sys.prefix).resolve()
    cands = []
    if source:
        cands.append({"role": ".venv", "name": "checkout .venv", "dir": root / ".venv", "found_by": "the checkout folder"})
    ldir = launcher_env_dir(root)
    if (root / "scripts" / "topopt-env.sh").is_file():      # an installed package (a wheel) has no launcher
        cands.append({"role": "launcher", "name": "launcher environment (scripts/topopt-env.sh)", "dir": ldir / "venv",
                      "found_by": "TOPOPT_ENV_DIR" if os.environ.get("TOPOPT_ENV_DIR") else
                      "CLAUDE_PLUGIN_DATA" if os.environ.get("CLAUDE_PLUGIN_DATA") else "the default <repo>/.plugin-env"})
        for found in sorted((Path.home() / ".claude" / "plugins" / "data").glob("*topopt*/venv")):
            if found != ldir / "venv":
                cands.append({"role": "launcher", "name": "launcher environment of the Claude Code plugin",
                              "dir": found, "found_by": "a search in ~/.claude/plugins/data"})
    rows, this_known = [], False
    for c in cands:
        python = c["dir"] / "bin" / "python"
        is_this = c["dir"].exists() and c["dir"].resolve() == this_prefix
        this_known = this_known or is_this
        row = {"role": c["role"], "name": c["name"], "dir": str(c["dir"]), "python": str(python), "found_by": c["found_by"],
               "exists": python.exists(), "is_this": is_this, "runs": None, "python_version": None, "topopt_from": None,
               "error": None, "used_by": used.get(c["role"], [])}
        if is_this:
            row.update(runs=True, python_version=sys.version.split()[0], topopt_from=str(root))
        elif row["exists"]:
            row.update(probe(python))
        else:
            row["error"] = "the Python does not exist" if c["dir"].exists() else "the folder does not exist (not made yet)"
        rows.append(row)
    if not this_known:
        rows.insert(0, {"role": "other", "name": "this Python (not one of the known environments)", "dir": str(this_prefix),
                        "python": sys.executable, "found_by": "sys.executable", "exists": True, "is_this": True,
                        "runs": True, "python_version": sys.version.split()[0], "topopt_from": str(root),
                        "error": None, "used_by": []})
    return {"environments": rows, "clients": clients}


# --- the commands -------------------------------------------------------------------------------------------------


def _launcher_command(root: Path, verb: str) -> str:
    """The command that runs the launcher for the SAME environment folder (the `again` function of the script)."""
    script = _q(root / "scripts" / "topopt-env.sh")
    default = root / ".plugin-env"
    ldir = launcher_env_dir(root)
    prefix = "" if ldir == default else f"TOPOPT_ENV_DIR={_q(ldir)} "
    return f"{prefix}sh {script} {verb}"


def next_commands(data: dict) -> list:
    """The commands for the user, in order, each with the reason. Nothing here runs them."""
    cmds = []
    checkout = data.get("checkout") or {}
    root = Path(checkout.get("code_folder") or ".")
    py = _q(sys.executable)
    uv = uv_command()    # the uv that the launcher downloaded, if it did (TOPOPT_UV), else uv
    envs = (data.get("known") or {}).get("environments", [])
    this = next((e for e in envs if e.get("is_this")), {})
    broken = [r for r in data.get("imports", []) if is_problem(r)]
    if broken:
        names = ", ".join(r["name"] for r in broken)
        reinstall = any(r["status"] in ("cut", "failed", "crashed") for r in broken)
        if this.get("role") == "launcher":
            # The marker file makes `sync` install the packages again (uv skips a package that has its .dist-info). It is the
            # command that `sh scripts/topopt-env.sh doctor` prints. uv `--reinstall` also refreshes, so the repair needs
            # the network even when the uv cache is warm; with UV_OFFLINE=1 it works offline if the cache has every wheel.
            marker = Path(this["dir"]) / LAUNCHER_MARKER
            cmds.append({"why": f"{names}: the environment of the launcher is broken. The marker file makes sync install the "
                                f"packages again. This needs the network, also with a warm uv cache; with UV_OFFLINE=1 in front "
                                f"of the command, uv works offline when its cache has every wheel",
                         "command": f"touch {_q(marker)} && {_launcher_command(root, 'sync')}"})
        elif checkout.get("has_pyproject"):
            flag = "--reinstall " if reinstall else ""
            cmds.append({"why": f"{names}: " + ("install again the packages whose files are broken" if reinstall
                                                else "install the missing packages"),
                         "command": f"{uv} pip install {flag}--python {py} -e {_q(str(root) + '[dev,mcp]')}"})
        else:     # an installed package (a wheel): the source it came from is not known here
            cmds.append({"why": f"{names}: this is an installed package, not a checkout. Install topopt again, with "
                                f"`{uv} pip install --reinstall --python {py}` and the source you installed it from",
                         "command": None})
    if checkout.get("git"):
        if checkout.get("behind"):
            cmds.append({"why": f"the checkout is {checkout['behind']} commits behind {checkout['default_branch_ref']} "
                                f"(as of the last fetch): update the code (this fetches first)",
                         "command": checkout["update_command"]})
            if this.get("role") == "launcher":
                cmds.append({"why": "then refresh the packages of the launcher environment (the stamp changes with the lock)",
                             "command": _launcher_command(root, "sync")})
            elif checkout.get("has_pyproject"):
                cmds.append({"why": "then refresh the packages of this environment (the lock may have changed)",
                             "command": f"{uv} pip install --python {py} -e {_q(str(root) + '[dev,mcp]')}"})
        if not checkout.get("behind"):
            cmds.append({"why": "see the newest code: the numbers above are as of the last fetch",
                         "command": checkout["fetch_command"]})
    for offer in data.get("offers", []):
        command = offer.get("command") or offer.get("install_command")
        if command:
            cmds.append({"why": f"offer {offer['action']} (only after you agree)", "command": command})
    return cmds


# --- the report ---------------------------------------------------------------------------------------------------


def _environment_progress_text() -> str:
    """The line before `environment_data`: what it does, and the limit of the self-test of gpu-mg (`PROBE_TIMEOUT_S`)."""
    try:
        from .gpu_mg.probe import PROBE_TIMEOUT_S
        limit = f"up to {PROBE_TIMEOUT_S:g} s"
    except Exception:    # a broken install: say the fact without the number
        limit = "with a time limit"
    return (f"topopt doctor: reading the machine and the solver backends (on a GPU, until a self-test has passed on "
            f"this device and driver, this runs the self-test of gpu-mg, {limit}).")


def collect(yaml_path=None, imports: bool = True, import_timeout_s: float = IMPORT_TIMEOUT_S, progress=None) -> dict:
    """All the facts of `topopt doctor` as one dict. It never raises; a failing part becomes a line in `problems`.

    `progress`: a callable that takes one line of text, or None (the default: no line). See `import_checks`."""
    from . import __version__
    data = {"topopt_version": __version__, "time_utc": utc_text(time.time()), "environment": None, "environment_error": None,
            "checkout": None, "imports": [], "known": None, "offers": [], "problems": [], "next_commands": [], "ok": True}
    if progress is not None:
        progress(_environment_progress_text())
    try:
        from .environment import environment_data
        data["environment"] = environment_data(yaml_path)
        data["checkout"] = data["environment"].get("checkout")
        data["offers"] = list(data["environment"].get("recommendations") or [])
    except Exception as exc:    # a broken environment: say it and go on with the other parts
        data["environment_error"] = f"{type(exc).__name__}: {exc}"
        data["problems"].append(f"environment_data() failed: {data['environment_error']}")
    if data["checkout"] is None:
        data["checkout"] = checkout_facts()
    root = data["checkout"]["code_folder"]
    if imports:
        if progress is not None:
            progress(f"topopt doctor: checking the imports of {len(IMPORT_CHECKS)} packages, each in its own Python "
                     f"(up to {import_timeout_s:g} s each).")
        data["imports"] = import_checks(timeout=import_timeout_s, code_folder=root, progress=progress)
        for row in data["imports"]:
            if is_problem(row):
                data["problems"].append(f"{row['name']} ({row['use']}): {row['status']}. {row['detail']}".strip())
            elif row["status"] == "not installed" and data["checkout"].get("has_pyproject") and _extra_fits_here(row["use"]):
                extra = row["use"].split(" ", 1)[1]
                row["install_command"] = f"{uv_command()} pip install --python {_q(sys.executable)} -e {_q(root + '[' + extra + ']')}"
    try:
        data["known"] = known_environments(root)
    except Exception as exc:
        data["known"] = {"environments": [], "clients": [], "error": f"{type(exc).__name__}: {exc}"}
    data["ok"] = not data["problems"]
    data["next_commands"] = next_commands(data)
    return data


def checkout_code_line(c: dict) -> str:
    """One line: `git 98bb5fb (2026-10-20), branch main, no uncommitted changes`. For a git checkout."""
    where = (f"branch {c['branch']}" if c.get("branch") else "detached HEAD" if c.get("detached") is True
             else "branch not known")
    changes = ("no uncommitted changes" if c.get("dirty") is False else
               f"uncommitted changes: {c['changed_files']} file(s)" if c.get("dirty") else "changes: not known")
    if c.get("untracked_files"):
        changes += f", {c['untracked_files']} untracked"
    date = f" ({c['head_date'][:10]})" if c.get("head_date") else ""
    return f"git {c.get('head')}{date}, {where}, {changes}"


def checkout_origin_parts(c: dict):
    """`(ref, text)`: `("origin/main", "0 commit(s) ahead, 12 behind, as of the last fetch (<time>, 3 days ago)")`, or None."""
    if not c.get("default_branch_ref") or c.get("behind") is None:
        return None
    counts = f"{c['ahead']} commit(s) ahead, {c['behind']} behind"
    if not c.get("last_fetch_utc"):
        return c["default_branch_ref"], f"{counts}, as of the clone (no fetch since)"
    return (c["default_branch_ref"],
            f"{counts}, as of the last fetch ({c['last_fetch_utc']}, {human_age(c['last_fetch_age_s'])} ago)")


def checkout_origin_line(c: dict) -> str | None:
    """One line: `origin/main: 0 commit(s) ahead, 12 behind, as of the last fetch (<time>, 3 days ago)`, or None."""
    parts = checkout_origin_parts(c)
    return f"{parts[0]}: {parts[1]}" if parts else None


def _wrap(text: str, indent: str, width: int = 100, later: str = None) -> list:
    """Wrap `text`; the first line starts with `indent`, the next lines with `later` (default: spaces as wide as `indent`)."""
    later = " " * len(indent) if later is None else later
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=later) or [indent.rstrip()]


def _gib(n) -> str:
    return f"{n / 2**30:.1f} GiB" if n else "unknown"


def render(d: dict) -> str:
    """The text report: short sections, facts first, then Offers and Next commands."""
    L = ["topopt doctor (read-only: it installs, builds, fetches and deletes nothing)",
         f"time: {d['time_utc']}   topopt {d['topopt_version']}"]
    L.append("status: ok, no problem found" if d["ok"] else f"status: {len(d['problems'])} problem(s), listed under Problems")
    c = d["checkout"] or {}
    L += ["", "Checkout", f"  folder:       {c.get('code_folder')} ({c.get('kind')})"]
    if c.get("git") and c.get("head"):       # git did not answer: the note says why, and no fact is made up
        L.append(f"  code:         {checkout_code_line(c)}")
        origin = checkout_origin_parts(c)
        if origin:
            L.append(f"  {origin[0] + ':':<14}{origin[1]}")
        if c.get("update_command"):
            L.append(f"  update:       {c['update_command']}")
    for note in c.get("notes", []):
        L += _wrap(note, "  note: ")
    env = d.get("environment")
    L += ["", "Machine"]
    if env:
        p, pl = env["python"], env["platform"]
        L.append(f"  {pl['name']}, release {pl['release']}, Python {p['version']} at {p['executable']}")
        ram = env.get("ram", {})
        L.append(f"  RAM {_gib(ram.get('total_bytes'))} (available {_gib(ram.get('available_bytes'))}), "
                 f"{env.get('cores', {}).get('usable')} usable core(s)")
        auto = env.get("auto_solver")
        backs = ", ".join(f"{n} {'yes' if b['available'] else 'no'}" for n, b in env["backends"].items())
        L.append(f"  solver for settings.solver auto: {auto}; backends: {backs}")
        why = (env.get("auto_choice") or {}).get("line")
        if why:
            L += _wrap(why, "  ")
        oc = env.get("opencl") or {}
        if oc.get("pyopencl"):
            devs = "; ".join(f"{x['index']}: {x['name']} ({x['kind']}" + (f", same as {x['duplicate_of']}" if x.get("duplicate_of") is not None else "") + ")"
                             for x in oc.get("devices", [])) or "no OpenCL device"
            L.append(f"  OpenCL: pyopencl {oc['pyopencl']}, {devs}")
        else:
            L.append(f"  OpenCL: {oc.get('problem') or 'not known'}")
        if env.get("platform", {}).get("hint"):
            L += _wrap(env["platform"]["hint"], "  note: ")
        prob = env.get("problem")
        if prob:
            L.append(f"  problem file {prob.get('yaml')}: " + (f"{prob['free_dofs']} free dofs, solver {prob['solver']['resolved']}, "
                                                           f"time estimate " + ("none (no constants for this device)" if prob['time_estimate']['total_s'] is None
                                                                else f"{prob['time_estimate']['total_s']:.0f} s")
                                                           if prob.get("ok") else f"error: {prob.get('error')}"))
    else:
        L.append(f"  not available: {d['environment_error']}")
    rows = d.get("imports", [])
    L += ["", "Packages (each import runs in a child Python)"]
    if rows:
        ok = [f"{r['name']} {r['version'] or ''}".strip() for r in rows if r["status"] == "ok"]
        L += _wrap("ok: " + ", ".join(ok), "  ") if ok else []
        for r in rows:
            if is_problem(r):
                L += _wrap(f"PROBLEM {r['name']} ({r['use']}): {r['status']}. {r['detail']}", "  ")
        absent = [f"{r['name']} ({r['use']})" for r in rows if r["status"] == "not installed" and r["use"] != "core"]
        if absent:
            L += _wrap("not installed (optional): " + ", ".join(absent), "  ")
            added = {}
            for r in rows:
                if r.get("install_command"):
                    added.setdefault(r["use"], r["install_command"])
            if added:
                L.append("  To add an optional package, run its command (only after you agree):")
                L += [f"    {use.split(' ', 1)[1]}: {command}" for use, command in added.items()]
    else:
        L.append("  not tested")
    known = d.get("known") or {}
    L += ["", "Environments"]
    for e in known.get("environments", []):
        state = ("runs" + (f" (Python {e['python_version']})" if e.get("python_version") else "") if e.get("runs")
                 else f"does not run: {e.get('error')}")
        if e.get("runs") and e.get("error"):       # the Python runs, but `import topopt` fails in it
            state += f", but topopt does not import: {e['error']}"
        mark = " <- this one" if e.get("is_this") else ""
        L.append(f"  {e['name']}: {e['python']}  {state}{mark}")
        if e.get("topopt_from") and not e.get("is_this"):
            L.append(f"      topopt code from: {e['topopt_from']}")
        if e.get("used_by"):
            L += _wrap("used by: " + ", ".join(e["used_by"]), "      ")
    for cl in known.get("clients", []):
        L.append(f"  client {cl['file']} ({cl['client']}): starts the {cl['environment']} environment")
    if d["problems"]:
        L += ["", "Problems"]
        for line in d["problems"]:
            L += _wrap(line, "  - ", 100, later="    ")
    L += ["", "Offers (facts and options; run a command only after you agree)"]
    if d["offers"]:
        for n, o in enumerate(d["offers"], 1):
            L.append(f"  {n}. {o['action']}")
            L += _wrap(o.get("reason", ""), "     ")
            L.append(f"     command: {o.get('command') or o.get('install_command')}")
    else:
        L.append("  none")
    L += ["", "Next commands"]
    if d["next_commands"]:
        for n, item in enumerate(d["next_commands"], 1):
            L += _wrap(f"{n}. {item['why']}", "  ")
            if item["command"]:
                L.append(f"     {item['command']}")
    else:
        L.append("  none")
    L += ["", "Send this text back if you ask for help. It has no secret except folder names."]
    return "\n".join(L)


def _stderr_printer():
    """A printer of progress lines for `collect`: one line to stderr, flushed. A stderr that is closed (`sys.stderr` is None) or
    broken (a closed pipe) must change neither stdout nor the exit code: the printer stops at the first failure and never
    writes to stdout."""
    state = {"on": sys.stderr is not None}

    def emit(text: str) -> None:
        if not state["on"]:
            return
        try:
            try:
                fd = sys.stderr.fileno()
            except (OSError, ValueError, AttributeError):    # no file number (a test capture): the normal way
                print(text, file=sys.stderr, flush=True)
                return
            # A write to the file number, with no buffer of Python: a buffer that holds a line that could not be written
            # makes the exit code of the interpreter 120 when it is flushed again at the end.
            os.write(fd, (text + "\n").encode("utf-8", "replace"))
        except (OSError, ValueError):
            state["on"] = False
    return emit


def run(yaml_path=None, as_json: bool = False, import_timeout_s: float = IMPORT_TIMEOUT_S) -> int:
    """`topopt doctor`: print the report. Exit code 0 when no problem line was found, 1 when there is one."""
    # The progress goes to stderr and is flushed at each line: stdout keeps the report (or the JSON) and nothing else.
    data = collect(yaml_path, import_timeout_s=import_timeout_s, progress=_stderr_printer())
    print(json.dumps(data, indent=1, default=str) if as_json else render(data))
    return 0 if data["ok"] else 1
