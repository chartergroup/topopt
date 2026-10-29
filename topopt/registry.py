"""The registry of live topopt servers on this machine (CONTRACTS.md section 43).

Two topopt servers on one machine (two chats, a chat and a Code tab, two clients) do not know about each other. Each server writes one
small file in a folder of the user. A server that reads the folder knows the other servers and the run folders that they started.
The registry holds no run state: the state of a run comes from the job files of its run folder (`topopt.runjob`).

The folder is one path for each platform (two processes with different environments must see the same folder):

- Linux: `$XDG_STATE_HOME/topopt`, else `~/.local/state/topopt`.
- macOS: `~/Library/Application Support/topopt`.
- The env var `TOPOPT_STATE_DIR` replaces the folder (tests, unusual setups).
- `~` is the home folder of the account database, not the env var `HOME`. Every topopt process of a user must see the same `XDG_STATE_HOME` and
  `TOPOPT_STATE_DIR`, or the registry is split.

The server entries are in `<root>/servers/server-<pid>.json`. `<root>/locks/` is reserved for a later machine-wide lock: this module
does not make it. Folders have the mode 0700 and files the mode 0600. When the folder cannot be made or written, the registry is off:
`Registration.problem` says why. There is no other folder to fall back to.

A run process lives on after the server that started it ends. So the entry names, for each run that the server started, the run folder and the
identity of the run process (`run_processes`). The entry of a server stays while one of these processes is still there (`Registration.remove`),
and the reader lists a server that has ended (`dead` or `reused`) as long as that is so. The file goes when the processes have ended. A job in a
run folder counts for an entry only when the process that holds its job lock is a process that this server started: two servers can name one folder
(the default run folder is the same for the chats of one user). The job is read from the job files of its run folder (`topopt.runjob`): the registry
holds no run state, only the identity of the process.

The module uses the standard library, `topopt.runinfo` and `topopt.runjob`. It makes nothing at import. It works on macOS and on Linux.
"""
from __future__ import annotations

import json
import os
import re
import stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pwd      # the account database; not on every system
except ImportError:     # pragma: no cover
    pwd = None

from .runinfo import write_atomic
from .runjob import START_TOLERANCE_S, _epoch, identity_state, job_facts, process_identity

ENV = "TOPOPT_STATE_DIR"
XDG_ENV = "XDG_STATE_HOME"
SERVERS_DIR = "servers"
LOCKS_DIR = "locks"        # reserved for the machine-wide lock; nothing here makes it
FORMAT = 1
KIND_MCP_SERVER = "mcp_server"
DIR_MODE = 0o700
FILE_MODE = 0o600
ENDED_STATES = ("dead", "reused")     # the answers of `identity_state` for a server process that is gone
THERE_STATES = ("live", "unconfirmed")      # the answers of `identity_state` for a process that is still there
_ENTRY = re.compile(r"server-(\d+)\.json")       # a name that looks like an entry; only the exact name `server-{pid}.json` of the pid in the file is read

MAY_BE_STALE = ("may be stale: the process with this pid exists, but the system gives no start time to compare, so this server "
                "cannot be confirmed")
ENDED = ("no longer running: the process of this server has ended, or its pid is another process now. A process that it started for a run is "
         "still there, so the entry stays until that process has ended")


class RegistryOff(OSError):
    """The folder of the registry cannot be found, made or written. The text says why."""


# --- the folder ----------------------------------------------------------------------------

def _home(home=None) -> Path:
    """The home folder of the user for the `~` of the rules below: the one of the account database (`pwd.getpwuid(os.getuid()).pw_dir`), not
    the env var `HOME`. A client can start a server with another `HOME`, and the two processes must see the same folder. `Path.home()` is the fallback
    when the system has no `pwd` module, the uid has no entry (a container user), or the entry has no usable folder."""
    if home is not None:
        return Path(home)
    if pwd is not None:
        try:
            folder = pwd.getpwuid(os.getuid()).pw_dir
        except (KeyError, OSError):
            folder = ""
        if folder and os.path.isabs(folder):
            return Path(folder)
    try:
        return Path.home()
    except (RuntimeError, KeyError, OSError) as exc:    # no home folder can be found
        raise RegistryOff(f"no home folder can be found ({exc})") from exc


def state_root(env=None, platform: str | None = None, home=None) -> Path:
    """The root folder of the registry. It makes nothing. `RegistryOff` when the rule needs a home folder and there is none.

    `env` (default `os.environ`), `platform` (a `sys.platform` text) and `home` are for a test.
    `TOPOPT_STATE_DIR` wins (a leading `~` is replaced by the home folder, see `_home`; a relative path starts at the current folder). An empty text means not set.
    Else macOS (`darwin`) uses `~/Library/Application Support/topopt` and any other system uses `$XDG_STATE_HOME/topopt`. As the
    XDG rule says, an empty or relative `XDG_STATE_HOME` is not valid and counts as not set: `~/.local/state/topopt`.
    """
    env = os.environ if env is None else env
    given = (env.get(ENV) or "").strip()
    if given:
        path = Path(given)
        if given == "~" or given.startswith("~/"):
            path = _home(home) / given[2:]
        return path if path.is_absolute() else Path(os.path.abspath(path))
    if (sys.platform if platform is None else platform) == "darwin":
        return _home(home) / "Library" / "Application Support" / "topopt"
    xdg = env.get(XDG_ENV) or ""
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / "topopt"
    return _home(home) / ".local" / "state" / "topopt"


def servers_folder(root=None) -> Path:
    return (state_root() if root is None else Path(root)) / SERVERS_DIR


def _make_private(folder: Path) -> None:
    """Make one folder with the mode 0700. A folder that is there already is not changed (it may be a folder of the user's choice)."""
    try:
        folder.mkdir(mode=DIR_MODE)
    except FileExistsError:
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} is there and is not a folder") from None
        return
    os.chmod(folder, DIR_MODE)     # the umask can only take bits away from the mode of mkdir


def _make_folders(root: Path) -> Path:
    root.parent.mkdir(parents=True, exist_ok=True)    # the parents are folders of the system or of the user (~/.local/state)
    _make_private(root)
    servers = root / SERVERS_DIR
    _make_private(servers)
    return servers


# --- the run folders of an entry -------------------------------------------------------------

def _run_folders(run_dirs) -> list[str]:
    """The folders of `run_dirs` that can be read: absolute texts. A relative path would be read from the wrong folder."""
    return [d for d in run_dirs if isinstance(d, str) and os.path.isabs(d)] if isinstance(run_dirs, list) else []


def _run_processes(rec) -> list[dict]:
    """The run processes that an entry names: dicts with a text `run_dir` and an int `pid` (and the keys of `process_identity`). Anything else is skipped."""
    items = rec.get("run_processes") if isinstance(rec, dict) else None
    return [p for p in items if isinstance(p, dict) and isinstance(p.get("run_dir"), str) and type(p.get("pid")) is int] \
        if isinstance(items, list) else []


def same_process(a: dict, b: dict) -> bool:
    """True when two records name the same process: the same pid, and the identity keys of both agree. On Linux the records have the raw start ticks and
    the boot id (`runjob.process_identity`); else the epoch start times must differ by `START_TOLERANCE_S` at most. When a record has nothing to compare,
    the pid decides (the limit of `identity_state` too: a pid that is reused by a process of the same start time)."""
    if type(a.get("pid")) is not int or a.get("pid") != b.get("pid"):
        return False
    ticks_a, ticks_b = a.get("process_start_ticks"), b.get("process_start_ticks")
    if type(ticks_a) is int and type(ticks_b) is int and isinstance(a.get("boot_id"), str) and isinstance(b.get("boot_id"), str):
        return (ticks_a, a["boot_id"]) == (ticks_b, b["boot_id"])
    start_a, start_b = a.get("process_start"), b.get("process_start")
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (start_a, start_b)):
        return abs(start_a - start_b) <= START_TOLERANCE_S
    return True


def has_live_run_process(rec) -> bool:
    """True when a run process that the entry names is still there (`identity_state` is `live` or `unconfirmed`), also before it has taken the job lock.

    This is a process, not run state: the registry reads no job file for it. `identity_state` can raise for a pid that the system refuses
    (the reader turns that into a note).
    """
    return any(identity_state(p) in THERE_STATES for p in _run_processes(rec))


# --- this server ---------------------------------------------------------------------------

class Registration:
    """The entry of this process. `start()` writes it, `add_run_dir()` rewrites it, `remove()` deletes it.

    No method raises for a file problem: a problem is returned as a text, and `problem` keeps the text of the last write
    (None when the last write worked). A server that cannot write its entry goes on without it.
    """

    def __init__(self, kind: str, *, work_folder, started_utc: str, code_folder, python: str, topopt_version: str,
                 root=None, pid: int | None = None):
        self.pid = os.getpid() if pid is None else pid
        self.root = root
        self.record: dict = {
            "format": FORMAT, "kind": kind, "pid": self.pid, **process_identity(self.pid),
            "started_utc": started_utc, "work_folder": str(work_folder), "topopt_version": topopt_version,
            "code_folder": str(code_folder), "python": python, "run_dirs": [], "run_processes": [],
        }
        self.problem: str | None = None
        self._path: Path | None = None
        self._lock = threading.Lock()      # two tool calls in two threads must not write the temporary file together

    @property
    def path(self) -> Path | None:
        """The file of this entry, or None while it is not known (the folder rule failed)."""
        return self._path

    def _write(self) -> str | None:
        tmp = None
        root = None
        try:
            root = state_root() if self.root is None else Path(self.root)
            self._path = _make_folders(root) / f"server-{self.pid}.json"
            tmp = self._path.with_name(self._path.name + ".tmp")
            write_atomic(self._path, json.dumps(self.record, indent=1).encode(), mode=FILE_MODE)
        except OSError as exc:
            if tmp is not None:
                tmp.unlink(missing_ok=True)       # nothing is left behind if the move did not happen
            where = f" in {root}" if root is not None else ""
            cause = exc.strerror or str(exc)
            if isinstance(exc, (FileExistsError, NotADirectoryError)):
                cause = f"a part of the path is a file: {cause}"
            self.problem = (f"The registry of topopt servers is off: the entry of this server cannot be written{where} "
                            f"({cause}). Other servers do not see this server.")
            return self.problem
        self.problem = None
        return None

    def start(self) -> str | None:
        """Write the entry. None, or the text of the problem (the registry is off for this server)."""
        with self._lock:
            return self._write()

    def add_run_dir(self, run_dir, run_pid: int | None = None) -> str | None:
        """Add a run folder that this server started a run in, and rewrite the entry. Every folder stays in the list, in start order.

        `run_pid` is the pid of the run process that this server started there. The entry then also names that process (`run_processes`: the folder,
        the pid and `process_identity`; an identity, not run state). A job counts for this entry only when that process holds its job lock, and the
        entry stays while that process is there.
        """
        text = str(run_dir)
        process = None if run_pid is None else {"run_dir": text, "pid": run_pid, **process_identity(run_pid)}
        with self._lock:
            if text not in self.record["run_dirs"]:
                self.record["run_dirs"].append(text)
            if process is not None and process not in self.record["run_processes"]:
                self.record["run_processes"].append(process)
            return self._write()

    def remove(self) -> None:
        """Delete the entry, unless a run process that this server started is still there. It never raises.

        A run process lives on after its server. While one is there (also before it has taken its job lock), the entry stays, so that the other
        servers know it (they list this server as `dead`). The next reader deletes the file when the processes have ended. It does nothing in a
        forked child (the pid is not the pid of the entry).
        """
        if os.getpid() != self.pid or self._path is None:
            return
        try:
            if has_live_run_process({"run_processes": list(self.record["run_processes"])}):
                return
        except Exception:
            return              # not sure: keep the file. The next reader deletes it when it is stale.
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass


# --- the other servers ---------------------------------------------------------------------

@dataclass
class Others:
    """`servers`: one dict for each listed entry of another server: `live`, `unconfirmed`, or `dead` / `reused` with a run process that is still there (see `describe`).
    `notes`: what was not read."""
    servers: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _memory_estimate_gib(run_dir: Path, job_started_at):
    """`derived.memory_estimate_gib` of the `run.json` in the folder, or None. The run writes it after it has taken the job lock.

    A `run.json` that is older than the job (the start time of the job is cut to whole seconds, so it is never later than the first write of the
    run) is from an earlier run in the same folder: None.
    """
    path = run_dir / "run.json"
    try:
        started = _epoch(job_started_at)
        if started is not None and path.stat().st_mtime < started:
            return None
        derived = json.loads(path.read_text()).get("derived")
        value = derived.get("memory_estimate_gib") if isinstance(derived, dict) else None
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def run_facts(run_dir, processes=None) -> dict:
    """The job in one run folder: `run_dir`, `job_state` (see `runjob.job_facts`; None if the folder has no job record), `pid`,
    `started_at`, `summary` and `memory_estimate_gib` (of a running job, from its `run.json`; None for any other job, for a run
    that has not written `run.json` yet, and when `run.json` is from an earlier run). The state comes from the job files, not from the registry.

    `processes`: the run processes that one entry names for this folder (`run_processes`). When it is given, the job counts for that entry only if
    it is the job of one of them (`same_process`); a job of another process (another server used the same folder) is not the job of the entry: the
    facts say so and `job_state` is None, so that it is never read as a job of this server.
    """
    d = Path(run_dir)
    try:
        job = job_facts(d)
    except Exception as exc:     # a fact, not an error: the folder is odd
        return {"run_dir": str(d), "job_state": None, "pid": None, "started_at": None,
                "summary": f"the job record cannot be read ({type(exc).__name__}: {exc})", "memory_estimate_gib": None}
    if job and processes is not None and not any(same_process(job, p) for p in processes):
        return {"run_dir": str(d), "job_state": None, "pid": None, "started_at": None,
                "summary": f"the job record in this folder (pid {job.get('pid')}) is not a job that this server started", "memory_estimate_gib": None}
    state = job["state"] if job else None
    return {"run_dir": str(d), "job_state": state, "pid": job.get("pid") if job else None,
            "started_at": job.get("started_at") if job else None,
            "summary": job["summary"] if job else "no job record can be read in this folder",
            "memory_estimate_gib": _memory_estimate_gib(d, job.get("started_at")) if state == "running" else None}


def describe(rec: dict, state: str) -> dict:
    """One entry as the reply of `environment` shows it (`other_servers`). `state` is the answer of `identity_state`.
    `unconfirmed` adds the note "may be stale". `dead` and `reused` (a server that has ended, listed because its run goes on) add the note
    "no longer running"."""
    processes = _run_processes(rec)
    item = {"pid": rec["pid"], "state": state, "kind": rec.get("kind"), "started_utc": rec.get("started_utc"),
            "work_folder": rec.get("work_folder"), "topopt_version": rec.get("topopt_version"),
            "code_folder": rec.get("code_folder"), "python": rec.get("python"),
            "runs": [run_facts(d, [p for p in processes if p["run_dir"] == d]) for d in _run_folders(rec.get("run_dirs"))]}
    if state == "unconfirmed":
        item["note"] = MAY_BE_STALE
    elif state in ENDED_STATES:
        item["note"] = ENDED
    return item


def _remove_if_unchanged(path: Path, before: os.stat_result) -> None:
    """Delete an entry of a process that is gone. A new server with the same pid may have replaced the file since it was read: then keep it."""
    try:
        now = os.lstat(path)
        if stat.S_ISREG(now.st_mode) and (now.st_ino, now.st_mtime_ns) == (before.st_ino, before.st_mtime_ns):
            path.unlink()
    except OSError:
        pass


def read_servers(own_pid: int | None = None, root=None) -> Others:
    """The entries of the other servers. It never raises.

    The own pid is skipped. A process that is `live` or `unconfirmed` is listed (`unconfirmed`: the pid exists, but there is no start
    time to compare; it is never called live). A server that is confirmed `dead`, or whose pid is `reused`, is listed with that state
    as long as a run process that it started is still there (its run went on after it); else its file is deleted and it is not listed.
    Nothing else is deleted. A file that cannot be read or parsed is a note in `notes`, and the other files are read. A folder that is
    not there is no note: no server has written to it. The folder is not made.
    """
    out = Others()
    own = os.getpid() if own_pid is None else own_pid
    try:
        folder = servers_folder(root)
        names = sorted(os.listdir(folder))
    except FileNotFoundError:
        return out
    except OSError as exc:
        out.notes.append(f"The registry of topopt servers cannot be read ({exc.strerror or exc}).")
        return out
    for name in names:
        if _ENTRY.fullmatch(name) is None or name == f"server-{own}.json":
            continue
        path = folder / name
        try:
            before = os.lstat(path)
            if not stat.S_ISREG(before.st_mode):      # a folder, a link: never read through it, never delete it
                out.notes.append(f"{name}: not read (it is not a regular file).")
                continue
            rec = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            out.notes.append(f"{name}: not read ({getattr(exc, 'strerror', None) or exc}).")
            continue
        # The name must be the one that `Registration` writes for the pid in the file: `server-{pid}.json`. A name with a leading zero or with
        # non-ASCII digits is not, so such a file is a note and is never deleted.
        if not isinstance(rec, dict) or type(rec.get("pid")) is not int or name != f"server-{rec['pid']}.json":
            out.notes.append(f"{name}: not read (it is not an entry of that pid).")
            continue
        if rec.get("format") != FORMAT:
            out.notes.append(f"{name}: not read (format {rec.get('format')!r}; this build reads format {FORMAT}).")
            continue
        try:
            state = identity_state(rec)
            if state in ENDED_STATES and not has_live_run_process(rec):
                _remove_if_unchanged(path, before)
            else:
                out.servers.append(describe(rec, state))
        except Exception as exc:     # for example a pid that is too big for the system: a fact, not an error
            out.notes.append(f"{name}: not read ({type(exc).__name__}: {exc}).")
    return out


def running_jobs(servers: list) -> list:
    """The runs of the listed servers that have a running job: a list of `{server_pid, server_state, work_folder, **run_facts}`.
    `server_state` is `dead` or `reused` when the server has ended and its run goes on."""
    return [{"server_pid": s["pid"], "server_state": s["state"], "work_folder": s["work_folder"], **r}
            for s in servers for r in s["runs"] if r["job_state"] == "running"]
