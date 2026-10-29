"""Facts of the machine and of the process for the job record (CONTRACTS.md section 24.1, DL1a).

`topopt.runjob` puts these facts into `job.json`: `start_facts()` into the key `host` when a job starts, and `end_facts()` next to
`wall_s` when it ends. The aim is data for later analysis of run time and memory: a run that is slow because the CPU was busy with
other jobs looks different from a slow run on a quiet host, and the folder did not record the difference before.

Every helper here never raises. On an error the value is `None`, and (when the caller gives a list `notes`) one line says why.
A fact that does not exist on a platform is `None` without a note when it is expected (for example Low Power Mode on Linux, or the
git commit of a folder that is not a checkout), and `None` with a note when a Linux-only source is missing on macOS (`procs_running`,
once at the start of a job; the list of processes needs `ps`).
The reads are cheap: files of /proc and of `.git`, and `os.getloadavg()`. The git commit is read from the files (`head_commit`), with no
`git` command. The commands that remain (`ps`, `sysctl`, `pmset`; macOS only) run with the same timeout as the other read-only commands
of the code (`COMMAND_TIMEOUT_S`). Nothing here reads a command line of another process into the record: the process scan keeps the pid
and the kind of each process only.

Public: `start_facts()`, `usage()`, `end_facts()`, `topopt_processes()` (item P9-14 reuses it), `classify_command()`, `started_by()`,
`reset_peak_rss()`, `cpu_model()`, `cpu_counts()`, `ram()`, `load_avg()`, `procs_running()`, `versions()`, `git_sha()`, `head_commit()`,
`low_power_mode()`, `thread_env()`.

The module uses the standard library at import time. `topopt.cli` and `topopt.environment` are imported inside the functions that
reuse their code (the core and RAM rules), because they import `topopt.runjob` in the other direction.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

COMMAND_TIMEOUT_S = 5   # seconds for a read-only command: the value of `runjob._start_from_ps`, `solver_choice` (sysctl) and `environment` (vm_stat)
STARTED_BY_ENV = "TOPOPT_STARTED_BY"   # `topopt_mcp.server.main` sets it to `mcp` in its own environment: its child processes inherit it, and its own jobs see it
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
               "GOTO_NUM_THREADS",         # `cli.cap_blas_threads` reads it: if it is set, the cap does nothing
               "VECLIB_MAXIMUM_THREADS")   # Apple Accelerate (the macOS numpy and scipy wheels and the Cholesky shim)
PACKAGES = ("numpy", "scipy", "pypardiso")
KINDS = ("run", "verify", "mcp_server", "other")   # the kinds of a topopt process (`classify_command`)

# The files that Linux gives. They are names here, so that a test can point them to a file of its own.
PROC_DIR = Path("/proc")
PROC_STAT = PROC_DIR / "stat"
PROC_CPUINFO = PROC_DIR / "cpuinfo"
PROC_CLEAR_REFS = PROC_DIR / "self" / "clear_refs"


def _guard(notes, name: str, fn):
    """`fn()`, or None if it raises. The reason goes to `notes` (a list, or None for no notes) as `<name>: <ErrorType>: <text>`."""
    try:
        return fn()
    except Exception as exc:   # a fact must never stop a job
        if notes is not None:
            notes.append(f"{name}: {type(exc).__name__}: {exc}")
        return None


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _command_output(argv: list[str]) -> str:
    """Text of a read-only command. Raises `OSError`, `subprocess.SubprocessError` or `RuntimeError` (a code other than 0). A test replaces this."""
    done = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S, stdin=subprocess.DEVNULL,
                          env={**os.environ, "LC_ALL": "C"})
    if done.returncode != 0:
        raise RuntimeError(f"`{' '.join(argv)}` ended with code {done.returncode}")
    return done.stdout


# --- threads, cores, RAM ----------------------------------------------------------------------

def thread_env() -> dict:
    """`{variable: text or None}` for the six thread variables of `THREAD_VARS`. None: the variable is not set (the library uses its default).

    `topopt.cli.main` sets `OPENBLAS_NUM_THREADS` itself on Linux (`cli.cap_blas_threads`: min(4, usable cores), if none of the
    three OpenBLAS variables is set), before numpy loads. So the value read here is the OpenBLAS count of the run. The code has
    no other thread setting: MKL (PARDISO), numba and OpenMP use the variable, or their own default (all usable cores). On a Mac the
    OpenBLAS variable is usually None (no cap there); the Accelerate variable `VECLIB_MAXIMUM_THREADS` is the one that can be set.
    """
    return {name: os.environ.get(name) for name in THREAD_VARS}


def cpu_counts(notes=None) -> tuple:
    """`(cpu_count, usable)`. `cpu_count` is `os.cpu_count()`. `usable` is `cli.usable_cores()`: the CPU affinity
    (`os.sched_getaffinity` where it exists), lowered by the CPU limit of a container when there is one. Each is None on an error."""
    count = _guard(notes, "cpu_count", os.cpu_count)
    if count is None and notes is not None and not any(n.startswith("cpu_count:") for n in notes):
        notes.append("cpu_count: os.cpu_count() gives no number")   # it returns None, it does not raise, when the system does not tell

    def usable():
        from .cli import usable_cores
        return int(usable_cores())
    return count, _guard(notes, "cpu_usable", usable)


def _cpu_model_linux() -> str:
    for line in PROC_CPUINFO.read_text(errors="replace").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    raise LookupError(f"no 'model name' line in {PROC_CPUINFO}")


def _cpu_model_macos() -> str:
    from .solver_choice import SYSCTL_PATHS
    last: Exception = OSError("sysctl not found")
    for exe in SYSCTL_PATHS:   # the full path first: a short PATH has no /usr/sbin
        try:
            text = _command_output([exe, "-n", "machdep.cpu.brand_string"]).strip()
        except FileNotFoundError as exc:
            last = exc
            continue
        if not text:
            raise LookupError("sysctl machdep.cpu.brand_string gave no text")
        return text
    raise last


def cpu_model(notes=None) -> str | None:
    """The CPU model text. Linux: the first `model name` line of /proc/cpuinfo. macOS: `sysctl -n machdep.cpu.brand_string`."""
    if _is_macos():
        return _guard(notes, "cpu_model", _cpu_model_macos)
    return _guard(notes, "cpu_model", _cpu_model_linux)


def ram(notes=None) -> tuple:
    """`(total_bytes, available_bytes)` for this process: the host value, or the container limit if it is lower.

    This is `topopt.environment._ram()`, the function that the tool `environment` uses (and `check` reads the same total).
    Each is None when the system does not tell.
    """
    def read():
        from .environment import _ram
        return _ram()
    facts = _guard(notes, "ram", read)
    if not facts:
        return None, None
    total, avail = facts.get("total_bytes"), facts.get("available_bytes")
    if notes is not None:
        if total is None:
            notes.append("ram_total: the system gives no total RAM")
        if avail is None:
            notes.append("ram_available: the system gives no available RAM")
    return total, avail


# --- load -------------------------------------------------------------------------------------

def load_avg(notes=None) -> list | None:
    """`[1 min, 5 min, 15 min]` load average, 2 decimals, or None."""
    return _guard(notes, "load_avg", lambda: [round(v, 2) for v in os.getloadavg()])


def _procs_running_linux() -> int:
    for line in PROC_STAT.read_text().splitlines():
        if line.startswith("procs_running "):
            return int(line.split()[1])
    raise LookupError(f"no 'procs_running' line in {PROC_STAT}")


def procs_running(notes=None) -> int | None:
    """The number of tasks that run or wait to run now (`procs_running` of /proc/stat; it counts the reader). Linux only.

    The load average lags behind by minutes; this number does not. macOS has no such count: None, with a note.
    """
    if not _is_linux():
        if notes is not None:
            notes.append("procs_running: needs /proc/stat (Linux)")
        return None
    return _guard(notes, "procs_running", _procs_running_linux)


# --- other topopt processes -------------------------------------------------------------------

_PYTHON = re.compile(r"^(python|pypy)[0-9.]*(\.exe)?$", re.IGNORECASE)   # ignore case: a framework Python of macOS is `.../Python.app/Contents/MacOS/Python`
_CLI_MODULES = ("topopt", "topopt.cli", "topopt.__main__")   # the modules that run the command line `topopt <command>`


def _basename(text: str) -> str:
    return text.replace("\\", "/").rsplit("/", 1)[-1]


def _kind_of_module(module: str, args: list[str]) -> str | None:
    if module in _CLI_MODULES:
        return _kind_of_subcommand(args)   # only the command line has the words `run` and `verify`
    if module == "topopt_mcp" or module.startswith("topopt_mcp."):
        return "mcp_server"
    if module.startswith("topopt."):   # another module of the package, `python -m topopt.ratchet run ...` for example: its first word is not a topopt command
        return "other"
    return None


def _kind_of_script(script: str, args: list[str]) -> str | None:
    parts = script.replace("\\", "/").split("/")
    name, parent = parts[-1], (parts[-2] if len(parts) > 1 else "")
    if name in ("topopt-mcp", "topopt_mcp") or parent == "topopt_mcp":
        return "mcp_server"
    return _kind_of_subcommand(args) if name == "topopt" else None


def _kind_of_subcommand(args: list[str]) -> str:
    sub = next((a for a in args if not a.startswith("-")), None)
    return sub if sub in ("run", "verify") else "other"


def classify_command(argv: list[str]) -> str | None:
    """The kind of a process from its argument list: `run`, `verify`, `mcp_server`, `other`, or None for a process that is not topopt.

    A topopt process is a Python (`python -m topopt <command>`, `python -m topopt_mcp.server`, or `python <path>/topopt <command>`
    for the console script) or a console script started by its own name (`topopt`, `topopt-mcp`). `run` and `verify` are the commands
    `topopt run` and `topopt verify` (`python -m topopt`, `topopt.cli` or `topopt.__main__`, or the console script); any other command of
    `topopt` (ratchet, dashboard, check, ...) is `other`, and so is any other module of the package (`python -m topopt.ratchet run ...`, the
    process that the MCP server starts for a ratchet job).
    The program must be the first word. A text that only has the word `topopt` somewhere (an editor, `rg`) is not a topopt process.
    """
    if not argv:
        return None
    prog, rest = _basename(argv[0]), argv[1:]
    if prog in ("topopt", "topopt-mcp"):
        return _kind_of_script(argv[0], rest)
    if not _PYTHON.match(prog):
        return None
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg == "-m":
            return _kind_of_module(rest[i + 1], rest[i + 2:]) if i + 1 < len(rest) else None
        if arg.startswith("-m") and not arg.startswith("--"):
            return _kind_of_module(arg[2:], rest[i + 1:])
        if arg in ("-c", "-"):
            return None
        if arg.startswith("-"):
            i += 2 if arg in ("-W", "-X") else 1   # these two options take a value
            continue
        return _kind_of_script(arg, rest[i + 1:])
    return None


def _commands_proc() -> list:
    """`[(pid, argv)]` of the live processes, from /proc/<pid>/cmdline. A process that ends during the scan is skipped."""
    out = []
    for entry in os.listdir(PROC_DIR):
        if not entry.isdigit():
            continue
        try:
            raw = (PROC_DIR / entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
        if argv:
            out.append((int(entry), argv))
    return out


def _commands_ps() -> list:
    """`[(pid, argv)]` from `ps -axo pid=,command=` (macOS). `ps` joins the words with a space, so a path with a space is split."""
    out = []
    for line in _command_output(["ps", "-axo", "pid=,command="]).splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid.isdigit() and command.strip():
            out.append((int(pid), command.split()))
    return out


def topopt_processes(notes=None, exclude_pid: int | None = None) -> dict | None:
    """The live topopt processes on this machine, without their command lines: `{"count", "by_kind", "processes"}`, or None.

    `processes` is a list of `{"pid", "kind"}`; `by_kind` has the four kinds of `KINDS` (a count, 0 for none); `count` is their sum.
    The process of the caller (`exclude_pid`, default `os.getpid()`) is not counted. A parent that is a topopt process (the MCP
    server of a run that it started) is counted. Linux reads /proc/<pid>/cmdline; macOS runs `ps -axo pid=,command=`.
    Another system gives None with a note.
    """
    me = os.getpid() if exclude_pid is None else exclude_pid

    def scan():
        if _is_linux() and PROC_DIR.is_dir():
            rows = _commands_proc()
        elif _is_macos():
            rows = _commands_ps()
        else:
            raise OSError(f"no process list on {sys.platform}")
        found = []
        for pid, argv in sorted(rows):
            kind = None if pid == me else classify_command(argv)
            if kind is not None:
                found.append({"pid": pid, "kind": kind})
        by_kind = {k: sum(1 for p in found if p["kind"] == k) for k in KINDS}
        return {"count": len(found), "by_kind": by_kind, "processes": found}
    return _guard(notes, "other_jobs", scan)


# --- who started the job ----------------------------------------------------------------------

def started_by() -> str:
    """`pytest` (a test is running: `PYTEST_CURRENT_TEST`), `mcp` (the MCP server started the process and said so in
    `TOPOPT_STARTED_BY`), else `cli`. A test comes first: a test that starts the MCP server still makes a test run."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return "pytest"
    if os.environ.get(STARTED_BY_ENV) == "mcp":
        return "mcp"
    return "cli"


# --- versions ---------------------------------------------------------------------------------

def package_version(name: str, notes=None) -> str | None:
    """The version of an installed package, or None when it is not installed (no note: that is not an error).

    A package that the process has loaded already gives its `__version__` (no search of the install folders). Else the
    version comes from `importlib.metadata`.
    """
    def read():
        from importlib import metadata
        mod = sys.modules.get(name)
        loaded = getattr(mod, "__version__", None)
        if isinstance(loaded, str) and loaded:
            return loaded
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return _guard(notes, f"version_{name}", read)


_COMMIT = re.compile(r"[0-9a-f]{40,64}")


def _git_dir(folder: Path) -> Path | None:
    """The git folder of a checkout: `<folder>/.git` when it is a folder, or the path in the file `.git` (`gitdir: <path>`: a linked
    worktree or a submodule). None when `folder` has no `.git`: it is not a checkout (an installed wheel)."""
    dot = folder / ".git"
    if dot.is_dir():
        return dot
    if not dot.is_file():
        return None
    text = dot.read_text(errors="replace").strip()
    if not text.startswith("gitdir:"):
        raise ValueError(f"{dot} is a file that does not start with 'gitdir:'")
    return (folder / text.partition(":")[2].strip()).resolve()   # a relative path is relative to the folder; an absolute path stays


def _commit_of_ref(ref: str, gitdir: Path, common: Path) -> str:
    """The commit of a ref (`refs/heads/main`): the loose file of the git folder, then of the common folder (a linked worktree keeps
    the refs of the branches there), then `packed-refs` of the common folder."""
    for base in (gitdir, common):
        loose = base / ref
        if loose.is_file():
            sha = loose.read_text().strip()
            if not _COMMIT.fullmatch(sha):
                raise ValueError(f"{loose} does not hold a commit")
            return sha
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            sha, _, name = line.partition(" ")
            if name.strip() == ref and _COMMIT.fullmatch(sha):   # a comment line (#) and a peeled line (^) do not match
                return sha
    raise LookupError(f"the ref {ref} is not a loose ref or in packed-refs")


def head_commit(folder) -> str | None:
    """The commit that HEAD names in the git folder of `folder`, read from the files (`HEAD`, the ref file, `commondir`, `packed-refs`).

    No git command runs: nothing to wait for, no timeout, and on a Mac no stub of a missing developer tool. None when `folder` is not a
    checkout. Raises `OSError`, `ValueError` or `LookupError` for a git folder that cannot be read (the caller notes it).
    """
    gitdir = _git_dir(Path(folder))
    if gitdir is None:
        return None
    head = (gitdir / "HEAD").read_text().strip()
    if _COMMIT.fullmatch(head):   # a detached HEAD
        return head
    if not head.startswith("ref:"):
        raise ValueError(f"{gitdir / 'HEAD'} is neither a commit nor a ref")
    common_file = gitdir / "commondir"   # a linked worktree: the folder that holds the branches
    common = (gitdir / common_file.read_text().strip()).resolve() if common_file.is_file() else gitdir
    return _commit_of_ref(head.partition(":")[2].strip(), gitdir, common)


def git_sha(notes=None, root=None) -> str | None:
    """The commit of the code folder, if it is a git checkout: what HEAD names, read from the files of the `.git` folder (`head_commit`).

    None with no note when the folder is not a checkout (an installed wheel: that is a known state, as the missing Low Power Mode on Linux).
    None with a note when the files cannot be read. `root` is the folder that holds the `topopt` package (default: the folder of this
    file's package). No command runs, so the start of a job does not wait for git (DL1a review m7).
    """
    folder = Path(__file__).resolve().parent.parent if root is None else Path(root)
    return _guard(notes, "git_sha", lambda: head_commit(folder))


def versions(notes=None) -> dict:
    """`{"topopt", "git_sha", "numpy", "scipy", "pypardiso"}`. A package that is not installed is None."""
    from . import __version__
    out = {"topopt": __version__, "git_sha": git_sha(notes)}
    for name in PACKAGES:
        out[name] = package_version(name, notes)
    return out


# --- macOS power state ------------------------------------------------------------------------

_LOW_POWER = re.compile(r"^\s*lowpowermode\s+([01])\s*$", re.MULTILINE)


def low_power_mode(notes=None) -> bool | None:
    """macOS: True when Low Power Mode is on (the `lowpowermode` line of `pmset -g`). None on Linux (it does not apply, no note), and None
    with a note on macOS when the command fails or has no such line (a desktop Mac).

    The thermal state is not read: `pmset -g` does not print it (that needs a second command, `pmset -g therm`).
    """
    if not _is_macos():
        return None

    def read():
        found = _LOW_POWER.search(_command_output(["pmset", "-g"]))
        if found is None:
            raise LookupError("no 'lowpowermode' line in the output of `pmset -g`")
        return found.group(1) == "1"
    return _guard(notes, "low_power_mode", read)


# --- own peak memory --------------------------------------------------------------------------

def reset_peak_rss(notes=None) -> bool:
    """Linux: write `5` to /proc/self/clear_refs, which sets the peak resident memory of this process (`VmHWM`) to its memory now.

    After it, `VmHWM` is the peak of this process since this call, not since the process began. Returns True when the write worked.
    macOS has no such reset (`ru_maxrss` only grows): False, no note. A write that fails (a restricted /proc) gives False and a note.
    The value `5` resets the peak only: the other values of the file clear page flags and cost time.
    """
    if not _is_linux():
        return False

    def write():
        with open(PROC_CLEAR_REFS, "w") as f:
            f.write("5")
        return True
    return bool(_guard(notes, "peak_rss_reset", write))


# --- the facts ---------------------------------------------------------------------------------

def start_facts() -> dict:
    """The `host` block of `job.json`: the facts at the start of a job. See CONTRACTS.md section 24.1 for every key.

    The keys are always there. A fact that could not be read is None, and `notes` says why.
    """
    notes: list = []
    cpu_count, cpu_usable = cpu_counts(notes)
    total, avail = ram(notes)
    return {
        "started_by": started_by(),
        "threads": thread_env(),
        "cpu_count": cpu_count, "cpu_usable": cpu_usable, "cpu_model": cpu_model(notes),
        "ram_total_bytes": total, "ram_available_bytes": avail,
        "load_avg": load_avg(notes), "procs_running": procs_running(notes),
        "other_jobs": topopt_processes(notes),
        "version": versions(notes),
        "low_power_mode": low_power_mode(notes),
        "notes": notes,
    }


def usage(notes=None) -> dict | None:
    """The CPU time and the context switches of this process, and of the children that it has waited for, up to now (`getrusage`).

    A job reads it at its start and again at its end (`end_facts`): the difference is the work of the job. The process before the job, or
    another job of the same process, is not in it. None, with a note, when the call fails.
    """
    def read():
        import resource
        own, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        return {"cpu_s": own.ru_utime + own.ru_stime, "cpu_children_s": kids.ru_utime + kids.ru_stime,
                "ctx_voluntary": int(own.ru_nvcsw), "ctx_involuntary": int(own.ru_nivcsw)}
    return _guard(notes, "getrusage", read)


def end_facts(start: dict | None, notes=None) -> dict:
    """The keys that a finished job adds next to `wall_s`: CPU time, context switches, the load again.

    `start` is the `usage()` that the job read when it began. The CPU time, the CPU time of the children and the context switches are
    the DIFFERENCE between now and `start`: the work of the job, not of the process from its start (a test process, a long server, or the
    start-up of a command are not in it). `cpu_time_s` is the user and system time of this process and of the children that it has waited
    for during the job; `cpu_time_children_s` is the part of the children. The context switches are those of this process.
    They are None, with a note, when `start` is None (the read at the start failed) or the read now fails.
    """
    now = usage(notes)
    out = {"cpu_time_s": None, "cpu_time_children_s": None, "ctx_switches_voluntary": None, "ctx_switches_involuntary": None}
    if now is not None and start is None and notes is not None:
        notes.append("cpu_time: the CPU time was not read at the start of the job, so there is no difference")
    if now is not None and start is not None:
        kids = now["cpu_children_s"] - start["cpu_children_s"]
        out.update(cpu_time_s=round(now["cpu_s"] - start["cpu_s"] + kids, 3), cpu_time_children_s=round(kids, 3),
                   ctx_switches_voluntary=now["ctx_voluntary"] - start["ctx_voluntary"],
                   ctx_switches_involuntary=now["ctx_involuntary"] - start["ctx_involuntary"])
    out["load_avg_end"] = load_avg(notes)
    out["procs_running_end"] = procs_running(notes if _is_linux() else None)   # macOS: the note "needs /proc/stat" is at the start only
    return out
