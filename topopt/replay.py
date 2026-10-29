"""Replay of a finished run (CONTRACTS.md section 7).

`Replay` shows the files of a finished run as if the run was live.
It uses a replay clock. The clock starts at 0 when `start()` is called.
`RunSource` calls `Replay.read()` and `Replay.snapshot_iterations()`.
Nothing else reads the run directory in replay mode.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

# The steps of one iteration, in order. Each gets an equal share of the time.
STEPS = ("assemble", "solve", "sensitivity", "filter", "update", "snapshot")

_IT_LINE = re.compile(r"^\s*it\s+(\d+)\b")
_SNAP_FILE = re.compile(r"^snapshots/it_(\d+)\.bin$")


class Replay:
    def __init__(self, run_dir, speed: float = 1.0, badge: bool = True, now=time.monotonic):
        self.run_dir = Path(run_dir)
        self.speed = float(speed) if speed and speed > 0 else 1.0
        self.badge = badge
        self.now = now  # a function that returns seconds. Tests can replace it.
        self._lock = threading.Lock()
        self._t0 = None  # None means: not started
        self._started_at = None
        self._load()

    # -- load the finished run once --
    def _raw(self, rel):
        try:
            return (self.run_dir / rel).read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    def _load(self):
        self.rows = []
        for line in (self._raw("history.jsonl") or b"").decode("utf-8", "replace").splitlines():
            line = line.strip()
            if line:
                try:
                    self.rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self.rows.sort(key=lambda r: r["elapsed_s"])
        # Log lines. Each line belongs to the last "it" line before it.
        self.log = []  # (iteration the line belongs to, text)
        cur = -1
        for text in (self._raw("run.log") or b"").decode("utf-8", "replace").splitlines():
            m = _IT_LINE.match(text)
            if m:
                cur = int(m.group(1))
            self.log.append((cur, text))
        self.last_it = self.rows[-1]["iteration"] if self.rows else 0
        # The end of the run: the largest of the final status and metrics times.
        ends = [0.0]
        self.stopped = None   # the final status of a run that a request to stop ended (P4b-4): it has no post step and no result
        for rel in ("status.json", "result/metrics.json"):
            raw = self._raw(rel)
            if raw:
                try:
                    d = json.loads(raw)
                    ends.append(float(d.get("elapsed_s", d.get("run_time_s", 0.0))))
                    ends.append(float(d.get("run_time_s", 0.0)))
                    if rel == "status.json" and d.get("state") == "stopped":
                        self.stopped = d
                except (ValueError, TypeError):
                    pass
        self.end_s = max(ends + [self.rows[-1]["elapsed_s"] if self.rows else 0.0])

    # -- clock --
    def start(self):
        """Start the clock at 0. Call it again to restart."""
        with self._lock:
            self._t0 = self.now()
            self._started_at = datetime.now().astimezone().isoformat(timespec="microseconds")

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def clock(self) -> float:
        """Replay time in seconds. -1 before start."""
        t0 = self._t0
        if t0 is None:
            return -1.0
        return (self.now() - t0) * self.speed

    # -- what the clock shows --
    def _shown_rows(self, clock):
        return [r for r in self.rows if r["elapsed_s"] <= clock]

    def shown_iteration(self, clock=None) -> int:
        """Last iteration with a history line at the clock. -1 if none."""
        clock = self.clock() if clock is None else clock
        rows = self._shown_rows(clock)
        return rows[-1]["iteration"] if rows else -1

    def status(self) -> dict | None:
        clock = self.clock()
        if clock < 0:
            return None
        rows = self._shown_rows(clock)
        if clock >= self.end_s and self.stopped is not None:   # the run was stopped by a request: it ends as stopped, not as done
            st = {"state": "stopped", "iteration": self.last_it, "step": "stopped",
                  "elapsed_s": round(self.end_s, 2), "message": str(self.stopped.get("message") or "")}
        elif clock >= self.end_s:
            st = {"state": "done", "iteration": self.last_it, "step": "done",
                  "elapsed_s": round(self.end_s, 2), "message": ""}
        elif not rows:
            st = {"state": "starting", "iteration": 0, "step": "setup",
                  "elapsed_s": round(clock, 2), "message": ""}
        elif len(rows) == len(self.rows) and self.stopped is not None:   # a stopped run has no post step
            st = {"state": "running", "iteration": self.last_it, "step": STEPS[-1],
                  "elapsed_s": round(clock, 2), "message": ""}
        elif len(rows) == len(self.rows):
            st = {"state": "postprocess", "iteration": self.last_it, "step": "postprocess",
                  "elapsed_s": round(clock, 2), "message": ""}
        else:
            a = rows[-1]["elapsed_s"]
            b = self.rows[len(rows)]["elapsed_s"]
            frac = (clock - a) / (b - a) if b > a else 0.0
            step = STEPS[min(len(STEPS) - 1, max(0, int(frac * len(STEPS))))]
            st = {"state": "running", "iteration": rows[-1]["iteration"], "step": step,
                  "elapsed_s": round(clock, 2), "message": ""}
        if self.badge:
            st["mode"] = "replay"
        return st

    def snapshot_iterations(self, real: list[int]) -> list[int]:
        shown = self.shown_iteration()
        return [i for i in real if i <= shown]

    def read(self, rel: str):
        """Return the bytes of a file at the clock, or None."""
        clock = self.clock()
        if clock < 0:
            return None  # not started: "Waiting for a run..."
        if rel == "status.json":
            return json.dumps(self.status()).encode()
        if rel == "run.json":
            raw = self._raw(rel)
            if raw is None:
                return None
            info = json.loads(raw)
            info["started_at"] = self._started_at  # fresh value, so the page resets
            return json.dumps(info).encode()
        if rel == "history.jsonl":
            rows = self._shown_rows(clock)
            return "".join(json.dumps(r) + "\n" for r in rows).encode()
        if rel == "run.log":
            shown = self.shown_iteration(clock)
            done = clock >= self.end_s
            lines = [t for it, t in self.log
                     if it <= shown and (done or it < self.last_it or not self._is_tail(it, t))]
            return ("\n".join(lines) + "\n").encode() if lines else b""
        if rel.startswith("result/"):
            return self._raw(rel) if clock >= self.end_s else None
        m = _SNAP_FILE.match(rel)
        if m:
            return self._raw(rel) if int(m.group(1)) <= self.shown_iteration(clock) else None
        return self._raw(rel)  # labels.bin and other files

    def _is_tail(self, it, text) -> bool:
        """True for a line after the last "it" line (for example "done in ...")."""
        return it == self.last_it and not _IT_LINE.match(text)
