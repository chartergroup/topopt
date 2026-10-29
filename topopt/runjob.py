"""Run helper: job lock, time, peak memory and done/FAILED markers (CONTRACTS.md section 24.1).

One run process owns one run folder. This module writes these files in that folder:

    job.lock      the pid, the process start time and a token of the process that runs now
    job.json      state, pid, start and end time, wall time, peak memory, exit code, notes
    job.DONE      the run ended with an exit code that is a success code (text: `exit 0`)
    job.FAILED    the run ended with any other exit code (text: `exit N`)
    job.STOPPED   the run ended at a request to stop (text: `exit 5`; `STOPPED_EXIT`)
    job.stop      a request to stop the run (JSON). `request_stop` writes it, the run reads it at the end of a pass.
    job.guard     an empty file. A short `flock` on it makes "read the lock, then take it" one step.

`job.json` also holds the facts of the machine and of the process for the data loop (DL1a): the block `host` at the start
(`topopt.hostinfo.start_facts`), and the CPU time, the context switches, the load and the phases of a run at the end. CONTRACTS.md
section 24.1 lists every key. A fact that cannot be read is null, and `host.notes` says why.

Use `run_job(run_dir, body)`. It takes the lock, calls `body(job)`, records the end and removes the lock.
The body can call `job.finish(code)` early when its work is done and only waits after that.

The module needs `topopt.config`, `topopt.runinfo` and `topopt.hostinfo` only (`hostinfo` imports the other code inside its functions).
It works on macOS and on Linux. The platform branches are the unit of `ru_maxrss` in `runinfo` and the host facts in `hostinfo`.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import hostinfo
from .config import UserError
from .runinfo import peak_rss_info, write_atomic

LOCK_FILE = "job.lock"
INFO_FILE = "job.json"
DONE_FILE = "job.DONE"
FAILED_FILE = "job.FAILED"
STOPPED_FILE = "job.STOPPED"
STOP_FILE = "job.stop"
GUARD_FILE = "job.guard"
FORMAT = 1
STOP_FORMAT = 1
STOPPED_EXIT = 5      # the exit code of `topopt run` for a run that a request stopped (0, 1, 2 and 130 are taken; verify uses 3)
START_TOLERANCE_S = 1.0   # two epoch start times that differ by more than this are two processes (macOS, old records)
# `flock` errors that mean "this file system has no flock" (a network share, some FUSE mounts)
NO_FLOCK_ERRNOS = (errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS)
NO_FLOCK_FACT = ("job lock: the file system of the run folder gives no flock. The lock record is still checked, "
                 "but two starts at the same moment can both take the lock.")


class JobHeld(UserError):
    """A live process holds the job lock of the run folder. `holder` has the facts."""

    def __init__(self, message: str, holder: dict):
        super().__init__(message)
        self.holder = holder


# --- process facts -------------------------------------------------------------------------

def _iso(epoch: float | None = None) -> str:
    when = datetime.now() if epoch is None else datetime.fromtimestamp(epoch)
    return when.astimezone().isoformat(timespec="seconds")


def _parse_proc_stat(text: str):
    """(state letter, start time in clock ticks since boot) from the text of /proc/<pid>/stat, or None."""
    try:
        rest = text[text.rindex(")") + 2:].split()   # the name in the first field can hold spaces and ")"
        return rest[0], int(rest[19])
    except (ValueError, IndexError):
        return None


def _read_proc_stat(pid: int):
    try:
        return _parse_proc_stat(Path(f"/proc/{pid}/stat").read_text())
    except OSError:
        return None


def _boot_time() -> float | None:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _has_procfs() -> bool:
    return Path("/proc/self/stat").exists()


def _boot_id() -> str | None:
    """The id of this boot of the machine (Linux), or None."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _start_from_proc(pid: int) -> float | None:
    stat, boot = _read_proc_stat(pid), _boot_time()
    if stat is None or boot is None:
        return None
    return boot + stat[1] / os.sysconf("SC_CLK_TCK")


def _parse_lstart(text: str) -> float | None:
    """Epoch seconds from the `ps -o lstart=` text, for example `Tue Oct 20 08:09:36 2026`."""
    try:
        return time.mktime(time.strptime(" ".join(text.split()), "%a %b %d %H:%M:%S %Y"))
    except ValueError:
        return None


def _start_from_ps(pid: int) -> float | None:
    """macOS has no /proc. `ps` gives the start time (second resolution)."""
    try:
        r = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
                           timeout=hostinfo.COMMAND_TIMEOUT_S, env={**os.environ, "LC_ALL": "C"}, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_lstart(r.stdout) if r.returncode == 0 else None


def process_start_time(pid: int) -> float | None:
    """Start time of a process in epoch seconds, or None if this system does not say.

    Linux: /proc/<pid>/stat and the boot time. Other systems: `ps -o lstart=`. This is for text and for macOS:
    on Linux the boot time moves when the wall clock is stepped, so `holder_state` compares the raw ticks there.
    """
    return _start_from_proc(pid) if _has_procfs() else _start_from_ps(pid)


def process_identity(pid: int) -> dict:
    """What tells this process from another process with the same pid, for the lock record.

    `process_start`: epoch seconds, or None. On Linux also `process_start_ticks` (field 22 of /proc/<pid>/stat: clock
    ticks since boot, a clock step does not move it) and `boot_id` (a pid of an earlier boot is another process).
    """
    ident: dict = {"process_start": process_start_time(pid)}
    if _has_procfs():
        stat, boot_id = _read_proc_stat(pid), _boot_id()
        if stat is not None and boot_id is not None:
            ident.update(process_start_ticks=stat[1], boot_id=boot_id)
    return ident


def pid_alive(pid) -> bool:
    """True if a process with this pid exists. A zombie (ended, not waited for) counts as ended where /proc says it."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if _has_procfs():
        stat = _read_proc_stat(pid)
        if stat is not None and stat[0] in ("Z", "X"):
            return False
    return True


def identity_state(rec: dict) -> str:
    """State of the process that a record names (`pid` and the keys of `process_identity`): `live`, `dead`, `reused` or `unconfirmed`.

    `dead`: no such process. `reused`: the pid exists, but it is another process. On Linux the record has the raw start ticks and
    the boot id: another boot id or other ticks mean another process, and a wall-clock step changes neither. Without them (macOS,
    an old record) the epoch start times are compared with a tolerance of `START_TOLERANCE_S`. `live`: the pid exists and the
    start time is the recorded one. `unconfirmed`: the pid exists, but there is nothing to compare: the system gives no start
    time, or the record has none. A reader that must not say "live" without proof (the registry of servers) uses this function.
    """
    pid = rec.get("pid")
    if not pid_alive(pid):
        return "dead"
    ticks, boot_id = rec.get("process_start_ticks"), rec.get("boot_id")
    if isinstance(ticks, int) and isinstance(boot_id, str) and _has_procfs():
        now_boot = _boot_id()
        if now_boot is not None and now_boot != boot_id:
            return "reused"
        stat = _read_proc_stat(pid)
        if stat is not None:
            return "live" if stat[1] == ticks else "reused"
    recorded = rec.get("process_start")
    now = process_start_time(pid)
    if isinstance(recorded, (int, float)) and now is not None:
        return "reused" if abs(now - recorded) > START_TOLERANCE_S else "live"
    return "unconfirmed"


def holder_state(rec: dict) -> str:
    """State of the process that a lock record names: `live`, `dead` or `reused`.

    This is `identity_state`, where `unconfirmed` counts as `live`: if the system gives no start time, a live pid is a live holder
    (a lock must not be taken from a process that may still hold it).
    """
    state = identity_state(rec)
    return "live" if state == "unconfirmed" else state


# --- files ---------------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@contextlib.contextmanager
def _guard(run_dir: Path):
    """Hold an exclusive `flock` on job.guard. The kernel frees it when the process ends.

    Gives True when the `flock` is held. A file system with no `flock` (`NO_FLOCK_ERRNOS`) gives False: the caller goes on
    with the record check only. Any other error is raised.
    """
    fd = os.open(run_dir / GUARD_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            held = True
        except OSError as exc:
            if exc.errno not in NO_FLOCK_ERRNOS:
                raise
            held = False
        yield held
    finally:
        os.close(fd)


def _held_message(run_dir: Path, rec: dict) -> str:
    return (f"Another run holds the job lock of {run_dir}: pid {rec.get('pid')}, started {rec.get('started_at')}. "
            "Wait for it, use another run folder, or give the override (--override-lock, or "
            "override_lock=true in run_topopt) to run anyway.")


def live_holder(run_dir) -> dict | None:
    """Facts about a live process that holds the job lock of this folder, else None.

    A lock of a dead process or of a reused pid gives None. `run_topopt` calls this before it starts a run.
    """
    d = Path(run_dir)
    rec = _read_json(d / LOCK_FILE)
    if rec is None or holder_state(rec) != "live":
        return None
    return {"pid": rec.get("pid"), "started_at": rec.get("started_at"), "run_dir": str(d),
            "message": _held_message(d, rec)}


def exit_code_of(exc: BaseException) -> int:
    """The exit code that `topopt.cli.main` gives for an exception: 1 user error, 130 Ctrl+C, 2 other.

    A run that a request stopped is not an exception here: `topopt run` ends its job with `STOPPED_EXIT` itself."""
    if isinstance(exc, SystemExit):
        code = exc.code
        return 0 if code is None else code if isinstance(code, int) else 1
    if isinstance(exc, KeyboardInterrupt):
        return 130
    if isinstance(exc, UserError):
        return 1
    return 2


# --- the request to stop -------------------------------------------------------------------

def read_stop(run_dir) -> dict | None:
    """The request to stop in a run folder (`job.stop`), or None if there is none.

    The file is the request: a file that is not a topopt request (a person made it with `touch`, or it does not parse) is still
    a request, and the dict has `by: None`, `requested_at: None` and `target_pid: None`. One open per call, no waiting.
    """
    try:
        text = (Path(run_dir) / STOP_FILE).read_text()
    except FileNotFoundError:
        return None
    except OSError:
        text = ""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return {"format": None, "requested_at": None, "by": None, "target_pid": None}
    return data


def request_stop(run_dir, by: str, target_pid: int | None = None) -> tuple[dict, bool]:
    """Ask the run in a folder to stop at the end of its current pass. Return `(request, created)`.

    The request is JSON: `format`, `requested_at`, `by` (who asked: `mcp stop_run`, `topopt stop`), `target_pid` (the pid of the run if the caller
    knows it, else None). It is written atomically. If a request is there already, nothing is written: the old request comes back with
    `created` False. No signal is sent and nothing waits. The run reads the request after the checkpoint of a pass (`optimize.run`), and once
    before the post step.
    """
    old = read_stop(run_dir)
    if old is not None:
        return old, False
    Path(run_dir).mkdir(parents=True, exist_ok=True)   # a run that is starting may not have made its folder yet
    request = {"format": STOP_FORMAT, "requested_at": _iso(), "by": by,
               "target_pid": int(target_pid) if isinstance(target_pid, int) and target_pid > 0 else None}
    data = json.dumps(request, indent=1).encode()
    path = Path(run_dir) / STOP_FILE
    # Two writers can both find no request. Each writes its own temp file (a name unique to the writer, in the same folder) and moves it
    # to job.stop; the last move wins. A shared temp name would let the second move fail with FileNotFoundError.
    tmp = path.with_name(f"{STOP_FILE}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)       # nothing is left behind if the move did not happen
    on_disk = read_stop(run_dir)          # the reply comes from the file, not from what this writer meant to write
    if on_disk is None:                   # the run read the request and removed it already
        return request, True
    return on_disk, on_disk == request


def read_own_stop(run_dir) -> dict | None:
    """The request in the folder if it is for this process, else None. The run reads its request with this function.

    A request is for this process when `target_pid` is null (the writer named no pid, or the file is not a topopt request) or is this pid.
    A request that names another pid stays in the folder for that process: with `--override-lock` two processes can work in one folder, and
    `stop_run` names the one that holds the lock (the newer one).
    """
    request = read_stop(run_dir)
    if request is None:
        return None
    target = request.get("target_pid")
    return request if target is None or target == os.getpid() else None


def clear_stop(run_dir) -> bool:
    """Remove the request to stop. True if there was one."""
    path = Path(run_dir) / STOP_FILE
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def stopped_hint(run_dir) -> str:
    """A sentence for a reader that finds no result in a run folder: it names a stop by request and the resume. Empty for any other run.

    The status `stopped` is the sign (`topopt run` writes it when a request ended the run). The reader keeps its own text; this adds the
    fact that `Run topopt run to the end` would start a new run and delete the checkpoint, and that `--resume` continues the stopped run.
    """
    status = _read_json(Path(run_dir) / "status.json")
    if not status or status.get("state") != "stopped":
        return ""
    return (f" The run was stopped by a request ({status.get('message') or 'status stopped'}). `topopt run <yaml> --out {run_dir} --resume` "
            "continues it from its checkpoint; a run without --resume deletes that checkpoint.")


def is_run_folder(run_dir) -> bool:
    """True if the folder is a run folder: it holds `status.json` or `run.json`. This is the rule of the MCP tool `list_runs`.

    A folder of another kind of job (a ratchet root, for example) holds a job lock but neither file. `topopt run` writes `status.json` at once
    after it takes the lock, so a run that holds the lock is a run folder a moment later.
    The hidden folders where `RunWriter.clean` keeps an earlier run aside (F4-47) hold these files, and they are not run folders: they are inside one.
    A list of the folders of a work folder that is a run folder itself would show them as runs.
    """
    from .snapshot import ASIDE_DIR, ASIDE_MOVING
    d = Path(run_dir)
    if d.name in (ASIDE_DIR, ASIDE_MOVING):
        return False
    return (d / "status.json").is_file() or (d / "run.json").is_file()


def other_job_markers(run_dir) -> list[str]:
    """The names of the files in a folder that show a job that is not a topopt run (today: the ratchet log and state file)."""
    from .ratchet import LOG_FILE, STATE_FILE
    return [name for name in (STATE_FILE, LOG_FILE) if (Path(run_dir) / name).is_file()]


def stop_text(request: dict) -> str:
    """One phrase for a request: who asked, when, and for which pid. For facts and log lines."""
    who = request.get("by") or "an unknown requester"
    when = request.get("requested_at") or "an unknown time"
    pid = request.get("target_pid")
    return f"requested by {who} at {when}" + (f" for pid {pid}" if pid else " (no pid named)")


# --- the job -------------------------------------------------------------------------------

class RunJob:
    """The job of one run process in one run folder. Use `run_job`, or `start()` and `finish(code)`."""

    def __init__(self, run_dir, *, success_codes=(0,), override: bool = False, t_start: float | None = None, keep_peak: str | None = None,
                 record: dict | None = None):
        self.dir = Path(run_dir)
        self.record = dict(record or {})   # keys that the caller adds to job.json (`fine_pass`: the first run of `--fine-pass`, for `stop_reply`)
        self.keep_peak = keep_peak     # a reason (text) not to reset the peak memory at the start; None: reset (Linux)
        self.success_codes = frozenset(int(c) for c in success_codes)
        self.override = override
        self.t_start = time.time() if t_start is None else t_start
        self.facts: list[str] = []     # what start() found: a replaced lock, an override
        self.token: str | None = None
        self.finished = False
        self._record: dict = {}
        self._peak_floor = 0        # the peak at the start, or (after the reset) the memory in use right after it: the end peak is not below it
        self._phase_source = None   # the run writer that has the times of the phases (`watch_phases`)
        self._usage_start: dict | None = None   # `getrusage` at the start: the CPU time of the job is the difference to the end (`hostinfo.end_facts`)

    def start(self) -> None:
        """Take the lock. Raise `JobHeld` if a live process holds it and there is no override.

        A lock of a dead process or of a reused pid is replaced, and `facts` says so. Old markers go.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        # The facts of the host come BEFORE the guard: a start fact can wait for a command (macOS: sysctl, ps, pmset, vm_stat, each with a
        # timeout), and a second start of this folder must not wait on `job.guard` for that. A start that is refused below reads them for
        # nothing (about 7 ms) and writes nothing. The peak read and its reset stay inside, after the lock check.
        try:
            host = hostinfo.start_facts()
        except Exception as exc:   # the helpers do not raise; a fact must still never stop a job
            host = {"notes": [f"host: {type(exc).__name__}: {exc}"]}
        with _guard(self.dir) as flock_held:
            if not flock_held:
                self.facts.append(NO_FLOCK_FACT)
            old = _read_json(self.dir / LOCK_FILE)
            if old is not None:
                state = holder_state(old)
                who = f"pid {old.get('pid')}, started {old.get('started_at')}"
                if state == "live":
                    if not self.override:
                        raise JobHeld(_held_message(self.dir, old),
                                      {"pid": old.get("pid"), "started_at": old.get("started_at"),
                                       "run_dir": str(self.dir)})
                    self.facts.append(f"job lock: override. The process of the lock ({who}) is alive. "
                                      "This run takes the lock. Both runs write to the folder and can damage each other's files.")
                elif state == "reused":
                    self.facts.append(f"job lock: the pid of the lock ({who}) is now another process. "
                                      "The lock is replaced.")
                else:
                    self.facts.append(f"job lock: the process of the lock ({who}) is not running. "
                                      "The lock is replaced.")
            self.token = uuid.uuid4().hex
            pid = os.getpid()
            request = read_stop(self.dir)
            if request is not None and request.get("target_pid") != pid:
                # A request for another run (or for no pid) is old: this run did not exist when it was written. A request that names
                # this pid stays: it can have come between the spawn and the lock (`stop_run` knows the pid of a process that it started).
                clear_stop(self.dir)
                self.facts.append(f"job stop: a request to stop ({stop_text(request)}) was in the folder. It does not name this run "
                                  f"(pid {pid}), so it is removed.")
            ident = process_identity(pid)
            peak, source = peak_rss_info()
            # Linux: set VmHWM to the memory in use now, so that `peak_rss_bytes` at the end is the peak of this job (the process before
            # the job, a test process for example, is not in it). The peak before the reset stays in `peak_rss_start_bytes`.
            reset = source == "VmHWM" and self.keep_peak is None and hostinfo.reset_peak_rss(host["notes"])
            if source == "VmHWM" and self.keep_peak is not None:   # a reset was possible and the caller said no: the record says why
                host["notes"].append(f"peak_rss_reset: not done, {self.keep_peak}")
            self._peak_floor = peak_rss_info()[0] if reset else peak
            self._usage_start = hostinfo.usage(host["notes"])   # the CPU time and the switches of the job are the difference from here
            self._record = {
                "format": FORMAT, "state": "running", "pid": pid, **ident,
                "token": self.token, "started_at": _iso(self.t_start), "run_dir": str(self.dir),
                "success_codes": sorted(self.success_codes), "platform": sys.platform, "notes": list(self.facts),
                # The peak at the start, and where the number comes from. `ru_maxrss` can start at the peak of the parent.
                "peak_rss_start_bytes": peak, "peak_rss_source": source, "peak_rss_reset": bool(reset),
                "host": host, **self.record,
            }
            write_atomic(self.dir / INFO_FILE, json.dumps(self._record, indent=1).encode())
            for name in (DONE_FILE, FAILED_FILE, STOPPED_FILE):
                (self.dir / name).unlink(missing_ok=True)
            lock = {k: self._record[k] for k in ("format", "pid", "token", "started_at", "run_dir", *ident)}
            write_atomic(self.dir / LOCK_FILE, json.dumps(lock, indent=1).encode())

    def watch_phases(self, writer) -> None:
        """Take the times of the phases of a run from its run writer when the job ends (`RunWriter.phase_times`). A job with no
        writer (a ratchet, for example) has null phases."""
        self._phase_source = writer

    def _phase_facts(self, notes: list) -> dict:
        out = {"setup_s": None, "loop_s": None, "post_s": None, "peak_rss_loop_bytes": None}
        if self._phase_source is not None:
            try:
                out.update(self._phase_source.phase_times())
            except Exception as exc:   # the record must be written
                notes.append(f"phases: {type(exc).__name__}: {exc}")
        return out

    def finish(self, exit_code: int) -> None:
        """Write job.json and the marker, then remove the lock. A second call does nothing.

        `STOPPED_EXIT` (and not a success code) gives the state `stopped` and the marker `job.STOPPED`: the run did not finish, a request
        stopped it. A success code gives `done` and `job.DONE`. Any other code gives `failed` and `job.FAILED`.
        If another run has taken the lock over (an override), this run writes nothing and says so in `facts`.
        If the lock file is gone and the record is this run's own (a person removed the lock), this run writes its record.
        A problem with the files does not stop the run: it prints one line to stderr.
        """
        if self.finished or self.token is None:
            return
        self.finished = True
        ok = int(exit_code) in self.success_codes
        stopped = int(exit_code) == STOPPED_EXIT and not ok
        state, marker = ("stopped", STOPPED_FILE) if stopped else ("done", DONE_FILE) if ok else ("failed", FAILED_FILE)
        # VmHWM can read a little lower than an earlier read of the same process (up to about 1.5 MB on Linux),
        # so the peak is never less than the floor: the start value, or (after the reset) the memory in use right after it.
        notes: list = []
        phases = self._phase_facts(notes)
        try:
            ending = hostinfo.end_facts(self._usage_start, notes)
        except Exception as exc:   # as at the start: no fact stops the record
            ending = {}
            notes.append(f"end_facts: {type(exc).__name__}: {exc}")
        info = dict(self._record, state=state, exit_code=int(exit_code),
                    ended_at=_iso(), wall_s=round(time.time() - self.t_start, 2),
                    peak_rss_bytes=max(peak_rss_info()[0], self._peak_floor, phases["peak_rss_loop_bytes"] or 0),
                    **ending, **phases)
        if notes:
            info["host"] = dict(info.get("host") or {}, notes=[*((info.get("host") or {}).get("notes") or []), *notes])
        try:
            with _guard(self.dir):
                lock, current = _read_json(self.dir / LOCK_FILE), _read_json(self.dir / INFO_FILE)
                # Another run owns the folder if it holds the lock, or if its record is there and the lock is gone
                # (it ended first). A lock that a person removed leaves our own record: that is free, write.
                other = lock.get("token") if lock is not None else (current or {}).get("token")
                if other is not None and other != self.token:
                    self.facts.append("job lock: another run took the lock over. No job record and no marker are written.")
                    print("topopt: " + self.facts[-1], file=sys.stderr, flush=True)
                    return
                write_atomic(self.dir / INFO_FILE, json.dumps(info, indent=1).encode())
                write_atomic(self.dir / marker, f"exit {int(exit_code)}\n".encode())
                (self.dir / LOCK_FILE).unlink(missing_ok=True)
        except OSError as exc:
            print(f"topopt: job record not written: {exc}", file=sys.stderr, flush=True)


def run_job(run_dir, body, *, success_codes=(0,), override: bool = False, t_start: float | None = None, keep_peak: str | None = None,
            record: dict | None = None) -> int:
    """Run `body(job)` under the job lock of `run_dir`. Return its exit code (an int).

    `keep_peak` is a reason (text) for NOT resetting the peak memory of the process at the start (Linux, CONTRACTS.md section 24.1). Give it
    when the caller did work before the job that can be the peak of the whole command (`topopt run --resume` builds the problem first). The
    record then has `peak_rss_reset` false, the reason as a note in `host.notes`, and the old rule: `peak_rss_bytes` is not less than the start value.
    `record` is a dict of keys for job.json (the first run of `--fine-pass` adds `fine_pass`: `first_run`, `fine_run_dir`, `forced`).

    The marker is `job.DONE` if the exit code is in `success_codes` (default {0}). It is `job.STOPPED` for `STOPPED_EXIT`. Else it is `job.FAILED`.
    An exception ends the job with `exit_code_of(exc)`, and then goes on to the caller.
    `body` can call `job.finish(code)` before it returns. The first call counts.
    """
    job = RunJob(run_dir, success_codes=success_codes, override=override, t_start=t_start, keep_peak=keep_peak, record=record)
    job.start()
    try:
        code = body(job)
    except BaseException as exc:
        job.finish(exit_code_of(exc))
        raise
    code = 0 if code is None else int(code)
    job.finish(code)
    return code


# --- the facts for run_status ---------------------------------------------------------------

def _gb(nbytes) -> str:
    return f"{nbytes / 1e9:.2f} GB" if isinstance(nbytes, (int, float)) else "?"


def job_facts(run_dir) -> dict | None:
    """The state of the job in a run folder as facts, or None if the folder has no job files.

    `state` is one of:
      running     a live process holds the lock
      done        the job ended with a success code (job.DONE)
      stopped     a request to stop ended the run at a pass (job.STOPPED, exit code 5); the checkpoint is there, `--resume` continues
      failed      the job ended with another exit code (job.FAILED); `exit_code` has it
      stale_lock  the lock is there, but its process is not running (`holder` says dead or reused); `marker` is None
                  or the marker of an earlier job
      unfinished  job.json says running, but there is no lock
    The dict also has `pid`, `started_at`, `wall_s`, `peak_rss_bytes`, `notes` and one `summary` line.
    """
    d = Path(run_dir)
    info, lock = _read_json(d / INFO_FILE), _read_json(d / LOCK_FILE)
    if info is None and lock is None:
        return None
    out = dict(info or {})
    out.pop("token", None)
    out.pop("host", None)   # the facts of the data loop (DL1a) are in job.json for analysis; the reply for a person or an agent stays short
    out["run_dir"] = str(d)
    out["marker"] = next((m for m in (DONE_FILE, FAILED_FILE, STOPPED_FILE) if (d / m).exists()), None)
    state = (info or {}).get("state", "unfinished")
    if lock is not None:
        holder = holder_state(lock)
        out["holder"] = holder
        if holder == "live":
            state = "running"
            out.update(pid=lock.get("pid"), started_at=lock.get("started_at"))
        elif state in ("done", "failed", "stopped") and (info or {}).get("token") == lock.get("token"):
            out.setdefault("notes", []).append("a lock file of the finished job is left in the folder")
        else:
            state = "stale_lock"
            out.update(pid=lock.get("pid"), started_at=lock.get("started_at"))
    elif state == "running":
        state = "unfinished"
    out["state"] = state
    out["summary"] = _summary(out)
    return out


def _peak_text(f: dict) -> str:
    """`peak memory X GB`, with a note when the number can carry the peak of the parent process (`ru_maxrss`)."""
    peak, start = f.get("peak_rss_bytes"), f.get("peak_rss_start_bytes")
    text = f"peak memory {_gb(peak)}"
    if (f.get("peak_rss_source") != "VmHWM" and isinstance(peak, (int, float)) and isinstance(start, (int, float))
            and peak > 0 and start >= 0.25 * peak):
        text += f"; {_gb(start)} of it was there at the start and can come from the parent process"
    return text


def _summary(f: dict) -> str:
    state, rc = f["state"], f.get("exit_code")
    tail = f"{f.get('wall_s')} s, {_peak_text(f)}"
    if state == "running":
        return f"job: running (pid {f.get('pid')}, started {f.get('started_at')})"
    if state == "done":
        return f"job: done (exit {rc}, {tail})"
    if state == "stopped":
        return f"job: stopped by a request (exit {rc}, {tail}); the checkpoint is kept and --resume continues"
    if state == "failed":
        return f"job: FAILED with exit code {rc} ({tail})"
    if state == "stale_lock":
        why = "is another process now" if f.get("holder") == "reused" else "is not running"
        marker = "there is no done or FAILED marker" if f.get("marker") is None else f"{f['marker']} is from an earlier job"
        return (f"job: stale lock. The process of the lock (pid {f.get('pid')}, started {f.get('started_at')}) {why}, "
                f"and {marker}")
    return "job: job.json says running, but no lock is held"


# --- the reply of `topopt stop` and `stop_run` -------------------------------------------------

STOP_NOTE = ("A topopt build from before this change does not read job.stop. A run of such a build ignores the request.")


def _epoch(text) -> float | None:
    """Epoch seconds of an ISO time text (`job.json` `started_at`), or None."""
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


NOT_SEEN = "A run that is starting and has not taken the job lock yet is not seen here: call again if you expect one."


def _current_status(d: Path, job: dict | None) -> dict | None:
    """`status.json` of the folder, or None. A status file that is older than the job that holds the folder is from an earlier run: None.

    The start time of the job is cut to whole seconds, so it is never later than the first write of the run: a current file is not dropped.
    """
    status = _read_json(d / "status.json")
    if status is None or job is None or job.get("state") != "running":
        return status
    started = _epoch(job.get("started_at"))
    try:
        if started is not None and (d / "status.json").stat().st_mtime < started:
            return None
    except OSError:
        return None
    return status


def _fine_pass_follows(d: Path, job: dict) -> bool:
    """True if the job of `d` is the first run of `--fine-pass` and the fine pass follows it: the part needs a finer voxel (`run.json` `part_voxel`,
    written before the post step) or `--fine-voxel` was given (it wins)."""
    fine = job.get("fine_pass")
    if not isinstance(fine, dict) or not fine.get("first_run"):
        return False
    part = (_read_json(d / "run.json") or {}).get("part_voxel")
    return bool(part) and bool(part.get("recommend") or fine.get("forced"))


def stop_reply(run_dir, by: str, starting_pid: int | None = None) -> dict:
    """Ask the run in a folder to stop, if it can be asked. Return the reply as facts and options. It returns at once and sends no signal.

    `by` goes into the request (`topopt stop`, `mcp stop_run`). `starting_pid` is the pid of a process that the caller started for this folder
    and that has not taken the job lock yet (the MCP server knows it): the request then names that pid, and `job.stop` stays in the folder when the
    run takes the lock (`RunJob.start`). `state` is one of:
      stop_requested     a live run (a lock holder, or `starting_pid`) in its optimizer loop, or before it: `job.stop` is written with the pid
      already_requested  the same, but a request is in the folder already: it is not written again
      postprocess        a live run in the post step, the refine step or the end of its job: nothing is written (no pass is left to read it),
                         except for the first run of `--fine-pass` when a fine pass follows: `job.stop` is written in the folder, and the driver
                         reads it before it starts the fine pass (`cli._stop_before_fine`): state `stop_requested` (or `already_requested`)
      nothing_to_stop    the job ended (done, stopped, failed), or it is closing: nothing is written
      gone               a lock whose process is not running, or a record that says running and has no lock: nothing is written
      none               the folder has no job files and no live process of the caller
    The reply has `run_dir`, `pid`, `iteration` and `phase` (from status.json when it is the file of this run), `message`, `options`,
    `note` (`STOP_NOTE`), `job` (the job facts, or None), and `stop_request` (the request in the folder, or None).
    """
    d = Path(run_dir)
    job = job_facts(d)
    request = read_stop(d)
    state = job["state"] if job else None
    status = _current_status(d, job)
    phase, it = (status or {}).get("state"), (status or {}).get("iteration")
    reply = {"run_dir": str(d), "pid": None, "iteration": it, "phase": phase, "message": "", "options": [],
             "note": STOP_NOTE, "job": job, "stop_request": request, "checkpoint": None}

    def done(state_name, message, options=()):
        reply.update(state=state_name, message=message, options=list(options), stop_request=read_stop(d))
        return reply

    def ask(pid, where):
        request, created = request_stop(d, by, pid)       # the request that is on disk; `created` is False when another writer's request won
        text = stop_text(request)
        at = f"the current pass (pass {it} at the last status)" if isinstance(it, int) and phase == "running" else "its first pass"
        reply["pid"] = pid
        if not created:
            return done("already_requested", f"A request to stop is in the folder already ({text}). It is not written again. The run (pid {pid}, "
                        f"{where}) stops at the end of {at}, after it saves the checkpoint.",
                        ["wait: run_status shows the state `stopped` when the run has stopped",
                         "remove job.stop yourself to take the request back"])
        return done("stop_requested", f"The request is written ({text}). The run (pid {pid}, {where}) stops at the end of {at}, after it saves "
                    "the checkpoint of that pass. It does not stop sooner: a pass can take a long time.",
                    ["wait: run_status shows the state `stopped` when the run has stopped, and `topopt run ... --resume` continues it",
                     "remove job.stop to take the request back, before the run reads it"])

    def ask_before_fine(pid, fine, phase):
        """The first run of `--fine-pass` is in its post or refine step and the fine pass follows: the request waits in this folder for the driver."""
        request, created = request_stop(d, by, pid)
        text = stop_text(request)
        reply["pid"] = pid
        where = "in the post step" if phase == "postprocess" else "done (its refine step may still run)"
        head = ("A request to stop is in the folder already" if not created else "The request is written")
        return done("stop_requested" if created else "already_requested",
                    f"The first run of --fine-pass (pid {pid}) is {where}: all passes are done, and the post step does not read a request. {head} ({text}). "
                    f"The process reads it before it starts the fine pass in {fine.get('fine_run_dir')}, ends without starting it (exit code {STOPPED_EXIT}) and says "
                    "so in the log. The first run is finished and keeps its result.",
                    ["wait: the process ends with exit code 5 and the log of the first run says that the fine pass did not start",
                     "remove job.stop to take the request back, before the process reads it",
                     f"to make the fine pass later: `topopt run <yaml> --out {fine.get('fine_run_dir')} --fine-from {d}`"])

    if state == "running" and starting_pid is None and not is_run_folder(d):
        pid = job.get("pid")
        reply["pid"] = pid
        markers = other_job_markers(d)
        what = (f"{' and '.join(markers)} {'is' if len(markers) == 1 else 'are'} here: the job is a ratchet job, and a ratchet job does not read job.stop"
                if markers else "the folder has no ratchet.json and no ratchet.jsonl, so the kind of the job is not known")
        return done("none", f"No topopt run in {d}. A live process (pid {pid}) holds the job lock there, but the folder has no status.json and no run.json, "
                    f"so the job is not a topopt run ({what}). Nothing is written. A topopt run that has only just started writes status.json "
                    "at once after it takes the lock: call again if this folder should hold one.",
                    ["ratchet_status shows a ratchet job" if markers else "call stop_run again if this folder should hold a topopt run that has just started"])
    if state == "running":
        pid = job.get("pid")
        reply["pid"] = pid
        if phase in ("postprocess", "done") and _fine_pass_follows(d, job):
            return ask_before_fine(pid, job["fine_pass"], phase)
        if phase in ("postprocess", "done"):
            return done("postprocess", f"The run (pid {pid}) is in the post step or the refine step (status {phase}). All passes are done, and the "
                        "post step does not read a request. Nothing is written.",
                        ["wait for the run to end: run_status shows the state",
                         f"end the process yourself (pid {pid}); `topopt run ... --resume` then repeats the post step only, from the complete checkpoint"])
        if phase in ("stopped", "error"):
            return done("nothing_to_stop", f"The run (pid {pid}) has ended its work (status {phase}) and is closing its job record. Nothing is written.",
                        ["call run_status again"])
        return ask(pid, "running" if phase == "running" else "starting")
    if starting_pid is not None:
        return ask(starting_pid, "starting; it has not taken the job lock yet")
    from .checkpoint import describe
    cp = describe(d)
    reply["checkpoint"] = cp
    resume = []
    if cp:
        again = f"continues the run at pass {cp['iteration'] + 1}" if not cp["complete"] else "repeats the post step only"
        resume = [f"`topopt run <yaml> --out {d} --resume` {again}"]
    if state in ("stale_lock", "unfinished"):
        if cp:
            resume.append("run_topopt with resume=true replaces the stale lock, with a fact, and does the same")
        else:
            resume = ["no checkpoint is in the folder: a new run starts from the beginning"]
        return done("gone", f"No live run: {job['summary']}. Nothing is written.", resume)
    if state == "stopped":
        return done("nothing_to_stop", f"The run has stopped already ({job['summary']}). Nothing is written. {NOT_SEEN}", resume)
    if state in ("done", "failed"):
        return done("nothing_to_stop", f"The run has ended already ({job['summary']}). Nothing is written. {NOT_SEEN}",
                    resume if state == "failed" else [])
    if not d.is_dir():
        return done("none", f"The folder {d} does not exist, so there is no run in it. Nothing is written, and no folder is made. {NOT_SEEN}", [])
    return done("none", f"No run in {d}: the folder has no job files and no live process that was started for it. Nothing is written. {NOT_SEEN}", [])
