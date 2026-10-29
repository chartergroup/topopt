"""Dashboard server for topopt (CONTRACTS.md section 7).

Standard library only for HTTP. The mesh endpoint needs numpy and skimage.
All run files are read through `RunSource`. In replay mode, `RunSource`
asks `topopt.replay.Replay`, which shows only the files up to the replay clock.
"""
from __future__ import annotations

import errno
import json
import mimetypes
import re
import socket
import struct
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# All paths are relative to the package, so a wheel and a source checkout
# find the same files. `PKG_ROOT` is the folder that holds the `topopt` package.
PKG_DIR = Path(__file__).resolve().parent
PKG_ROOT = PKG_DIR.parent
STATIC_DIR = PKG_DIR / "dashboard"
SOURCE_REAL = PKG_DIR / "optimize.py"
SOURCE_FAKE = STATIC_DIR / "fake_optimize.py"

_MARKER = re.compile(r"^\s*#\s*\[(step:(\w+)|end)\]\s*$")
_SNAP = re.compile(r"^it_(\d{4,})\.bin$")

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")


# --------------------------------------------------------------------------
# File access. One place for all reads of the run directory.
# --------------------------------------------------------------------------
class RunSource:
    """Read files of one run directory.

    In replay mode, `self.replay_state` (a `topopt.replay.Replay`) filters
    every read by the replay clock. Nothing else reads the run directory.
    """

    def __init__(self, run_dir, replay: bool = False, speed: float = 1.0, badge: bool = True):
        self.run_dir = Path(run_dir)
        self.replay = replay
        self.speed = speed
        self.replay_state = None
        if replay:
            from .replay import Replay
            self.replay_state = Replay(self.run_dir, speed=speed, badge=badge)

    def read(self, rel: str):
        """Return the bytes of a file, or None if it does not exist."""
        if self.replay_state is not None:
            return self.replay_state.read(rel)
        path = self.run_dir / rel
        try:
            return path.read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    def exists(self, rel: str) -> bool:
        """True if a file exists (in replay mode: and is shown at the clock)."""
        if self.replay_state is not None:
            return self.replay_state.read(rel) is not None
        return (self.run_dir / rel).is_file()

    def snapshot_iterations(self) -> list[int]:
        """Return the sorted iteration numbers that have a snapshot file."""
        snap_dir = self.run_dir / "snapshots"
        found = []
        try:
            for p in snap_dir.iterdir():
                m = _SNAP.match(p.name)
                if m:
                    found.append(int(m.group(1)))
        except FileNotFoundError:
            return []
        found.sort()
        if self.replay_state is not None:
            return self.replay_state.snapshot_iterations(found)
        return found

    def snapshot_bytes(self, it: int):
        return self.read(f"snapshots/it_{it:04d}.bin")

    def run_info(self):
        raw = self.read("run.json")
        return json.loads(raw) if raw else None


# --------------------------------------------------------------------------
# Mesh from density
# --------------------------------------------------------------------------
def pack_mesh(verts, faces) -> bytes:
    """Pack a mesh in the format of CONTRACTS.md section 5."""
    import numpy as np

    v = np.ascontiguousarray(verts, dtype="<f4")
    f = np.ascontiguousarray(faces, dtype="<u4")
    return struct.pack("<II", len(v), len(f)) + v.tobytes() + f.tobytes()


def density_to_mesh(density, origin, h, level=0.5):
    """Marching cubes on the padded density. Return (verts, faces) in mm.

    `density` is a float array (nx, ny, nz) with values 0..1.
    An empty field gives empty arrays.
    """
    import numpy as np
    from skimage.measure import marching_cubes

    padded = np.pad(density.astype(np.float32), 1, mode="constant")
    if padded.max() <= level or padded.min() >= level:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    verts, faces, _n, _v = marching_cubes(padded, level=level)
    # Padded index p is voxel p - 1. The voxel centre is at +0.5 voxel.
    verts = np.asarray(origin, dtype=np.float64) + h * (verts - 1.0 + 0.5)
    # skimage winds the faces inward for this field. Flip to get outward normals.
    faces = faces[:, ::-1]
    return verts.astype(np.float32), faces.astype(np.uint32)


def adaptive_level(density, labels, vf, floor=0.05):
    """Find the iso level for the preview mesh.

    The level is the density value above which the design voxels fill the
    volume fraction `vf`. It is never above 0.5 and never below `floor`.
    Then a gray start field still shows a shape.
    """
    import numpy as np

    design = density[labels == 1]
    if design.size == 0:
        return 0.5
    q = float(np.quantile(design, 1.0 - min(max(vf, 0.0), 1.0)))
    # Go a little below q, so a uniform field does not sit on the level.
    return min(0.5, max(floor, q - 0.002))


class MeshCache:
    """Make preview meshes from snapshots. Keep the result by iteration.

    The cache is cleared when `started_at` of run.json changes (a new run)
    or when run.json is gone.
    """

    def __init__(self, source: RunSource, max_items: int = 200):
        self.source = source
        self.max_items = max_items
        self._cache: dict[int, bytes] = {}
        self._stamp = None
        self._lock = threading.Lock()

    def get(self, it: int):
        info = self.source.run_info()
        stamp = None if info is None else info.get("started_at")
        with self._lock:
            if stamp != self._stamp:
                self._cache.clear()
                self._stamp = stamp
            hit = self._cache.get(it)
        if hit is not None:
            return hit
        raw = self.source.snapshot_bytes(it)
        if info is None or raw is None:
            return None
        import numpy as np

        shape = tuple(info["shape"])
        n = shape[0] * shape[1] * shape[2]
        arr = np.frombuffer(raw, dtype=np.uint8)
        lab_raw = self.source.read("labels.bin")
        if arr.size != n or lab_raw is None or len(lab_raw) != n:
            return None
        density = arr.reshape(shape).astype(np.float32) / 255.0
        labels = np.frombuffer(lab_raw, dtype=np.uint8).reshape(shape)
        level = adaptive_level(density, labels, info.get("volume_fraction", 0.5))
        verts, faces = density_to_mesh(density, info["origin"], info["h"], level)
        data = pack_mesh(verts, faces)
        with self._lock:
            if stamp == self._stamp:
                if len(self._cache) >= self.max_items:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[it] = data
        return data


# --------------------------------------------------------------------------
# Source of the optimizer for the code panel
# --------------------------------------------------------------------------
def parse_source(path: Path) -> dict:
    """Read a source file. Remove marker lines. Find the step blocks.

    A block goes from its marker to the next marker or `# [end]`.
    Line numbers are 1-based in the shown text.
    """
    lines: list[str] = []
    steps: dict[str, list[int]] = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _MARKER.match(raw)
        if m:
            if current is not None and current in steps:
                steps[current][1] = len(lines)
            if m.group(2):
                current = m.group(2)
                if current not in steps:
                    steps[current] = [len(lines) + 1, len(lines) + 1]
                else:
                    current = None  # a repeated name: keep the first block
            else:
                current = None
            continue
        lines.append(raw)
        if current is not None:
            steps[current][1] = len(lines)
    # A block that has no line after its marker is empty: make it one line.
    for name, (a, b) in list(steps.items()):
        if b < a:
            steps[name] = [a, a]
    try:
        shown = str(path.relative_to(PKG_ROOT))
    except ValueError:
        shown = str(path)
    return {"path": shown, "lines": lines, "steps": steps}


def static_missing_message() -> str:
    return (f"dashboard files not found in {STATIC_DIR}; reinstall topopt from a "
            "build that ships topopt/dashboard, or use an editable install (pip install -e .)")


def source_path() -> Path:
    return SOURCE_REAL if SOURCE_REAL.exists() else SOURCE_FAKE


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "topopt-dashboard/1"
    protocol_version = "HTTP/1.1"

    # The server object gives: source, meshes.
    def log_message(self, fmt, *args):  # quiet
        pass

    # -- helpers --
    def _send(self, code, body: bytes, ctype: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _missing(self, what="not found"):
        self._json({"error": what}, 404)

    def _file(self, rel, ctype):
        data = self.server.source.read(rel)
        if data is None:
            return self._missing(rel)
        self._send(200, data, ctype)

    # -- routes --
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self._route(urlparse(self.path).path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # keep the server alive
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:
                pass

    def _route(self, path: str):
        src: RunSource = self.server.source
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path == "/refine":  # the markup page of the run (result/refine.html)
            return self._refine()
        if not path.startswith("/api/"):
            return self._static(path.lstrip("/"))
        name = path[len("/api/"):]
        if name == "run":
            return self._file("run.json", "application/json")
        if name == "status":
            return self._status()
        if name == "history":
            return self._history()
        if name == "labels":
            return self._file("labels.bin", "application/octet-stream")
        if name == "final":
            return self._file("result/preview.bin", "application/octet-stream")
        if name == "metrics":
            return self._file("result/metrics.json", "application/json")
        if name == "refine":  # does the refine page exist? (the Refine button)
            return self._json({"exists": src.exists("result/refine.html")})
        if name == "part_file":  # where the finished part is: the result panel names the file (CONTRACTS.md section 7)
            run_dir = src.run_dir.resolve()
            return self._json({"file": "result/final.stl", "run_dir": str(run_dir), "run_name": run_dir.name,
                               "exists": src.exists("result/final.stl")})
        if name == "replay/start" and src.replay_state is not None:
            src.replay_state.start()
            return self._json({"started": True})
        if name == "log":  # extra endpoint: the last lines of run.log
            raw = src.read("run.log") or b""
            tail = b"\n".join(raw.splitlines()[-30:])
            return self._send(200, tail, "text/plain; charset=utf-8")
        if name == "source":
            return self._json(parse_source(source_path()))
        if name.startswith("density/"):
            it = self._int(name[len("density/"):])
            if it is None:
                return self._missing("bad iteration")
            data = src.snapshot_bytes(it)
            if data is None:
                return self._missing("no snapshot")
            return self._send(200, data, "application/octet-stream")
        if name.startswith("mesh/"):
            arg = name[len("mesh/"):]
            if arg == "latest":
                its = src.snapshot_iterations()
                if not its:
                    return self._missing("no snapshot")
                it = its[-1]
            else:
                it = self._int(arg)
                if it is None:
                    return self._missing("bad iteration")
            data = self.server.meshes.get(it)
            if data is None:
                return self._missing("no snapshot")
            return self._send(200, data, "application/octet-stream",
                              {"X-Iteration": str(it)})
        return self._missing()

    def _status(self):
        """`status.json`; for a run in its passes with an estimate, also the key `time_left` (E1-P2, `topopt.eta`). The bar shows it."""
        src: RunSource = self.server.source
        raw = src.read("status.json")
        if raw is None:
            return self._missing("status.json")
        try:
            st = json.loads(raw)
        except ValueError:
            return self._send(200, raw, "application/json")   # not our file to repair: the page gets what the run wrote
        est = self.server.time_left(st) if isinstance(st, dict) else None
        if est is None:
            return self._send(200, raw, "application/json")   # no estimate: the bytes of the file, as before
        self._json({**st, "time_left": est})

    def _refine(self):
        from topopt.refine.page_scripts import inline_old_page

        data = self.server.source.read("result/refine.html")
        if data is None:
            return self._send(404, b"No refine page yet. Run: topopt refine <run_dir>\n",
                              "text/plain; charset=utf-8")
        self._send(200, inline_old_page(data), "text/html; charset=utf-8")   # a page of an older topopt: no CDN load (#396)

    @staticmethod
    def _int(text):
        return int(text) if text.isdigit() else None

    def _history(self):
        raw = self.server.source.read("history.jsonl")
        if raw is None:
            return self._json([])
        rows = []
        for line in raw.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a part line at the end
        self._json(rows)

    def _static(self, rel: str):
        base = STATIC_DIR.resolve()
        if not (base / "index.html").is_file():
            return self._json({"error": static_missing_message()}, 404)
        target = (base / rel).resolve()
        if base != target and base not in target.parents:
            return self._missing()
        if not target.is_file():
            return self._missing()
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/json":
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)


def _is_listening(sock) -> bool:
    """True when `sock` listens. An `OSError` means that it is not a usable socket (the caller names the fd).

    Linux answers `SO_ACCEPTCONN`. macOS does not: it fails with `ENOPROTOOPT` ("Protocol not available"). Then the test is
    indirect: a stream socket that is bound is taken as the socket that `topopt run --dashboard` listened on, and `listen()` on it
    sets the queue length again (it does no harm on a socket that listens, and it makes a bound socket that did not listen
    listen). A socket that is not bound (no port) does not listen, and `listen()` is not called on it (it would bind a free
    port and start to listen). A socket for which `listen()` fails (`OSError`, for example a connected one) does not listen.
    """
    try:
        return bool(sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN))
    except OSError as exc:
        if exc.errno != errno.ENOPROTOOPT:
            raise
    if sock.type != socket.SOCK_STREAM:
        return False
    name = sock.getsockname()
    if not isinstance(name, tuple) or len(name) < 2 or not name[1]:
        return False
    try:
        sock.listen(DashboardServer.request_queue_size)
    except OSError:                       # for example a socket that is connected: Linux says "does not listen" for it
        return False
    return True


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, run_dir, port=8765, replay=False, speed=1.0, badge=True, listen_fd=None):
        if not (STATIC_DIR / "index.html").is_file():
            print("warning: " + static_missing_message(), file=sys.stderr, flush=True)
        self.source = RunSource(run_dir, replay=replay, speed=speed, badge=badge)
        self.meshes = MeshCache(self.source)
        self._eta_cache = None   # (key, estimate): the history is read again only when it has changed
        if listen_fd is None:
            super().__init__(("127.0.0.1", port), Handler)
            return
        # A listening socket that the process before `os.execv` left open (`topopt run --dashboard`, P4 review M1): use it and do
        # not bind. The port is never closed, so a request that comes during the exec waits in the backlog.
        super().__init__(("127.0.0.1", 0), Handler, bind_and_activate=False)
        self.socket.close()
        try:
            self.socket = socket.socket(fileno=listen_fd)
            listening = _is_listening(self.socket)
        except OSError as exc:
            raise OSError(f"file descriptor {listen_fd} is not an open socket ({exc})") from exc
        if not listening:
            raise OSError(f"file descriptor {listen_fd} is a socket that does not listen")
        self.server_address = self.socket.getsockname()
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    def time_left(self, status: dict):
        """The live estimate (`topopt.eta`) for the run of this server, or None. It reads `history.jsonl` and `run.json` through `RunSource`.

        The reason that a run has none is not sent here: `run_status` gives it. A state other than `running` costs no read. A live run is
        also left without an estimate when its job record says that the process has ended (a kill leaves `running` in `status.json`):
        that test reads the two small job files at each poll and is not cached. The estimate is kept until the history, `run.json` or the
        `first_pass` of the status changes; in live mode a change is a new size or time of the file (`stat`), so a poll without a change
        reads nothing of the history. In replay mode the replay builds the bytes at each read, and the key is made from them.
        """
        from . import eta
        if status.get("state") != "running":
            return None
        src = self.source
        if not src.replay:
            from .runjob import job_facts
            if eta.job_ended(job_facts(src.run_dir)):
                return None
            hist = run_raw = None
            key = (self._stat_key("history.jsonl"), self._stat_key("run.json"), status.get("first_pass", 0))
        else:
            hist, run_raw = src.read("history.jsonl"), src.read("run.json")
            key = (len(hist or b""), hash(run_raw), status.get("first_pass", 0))
        cached = self._eta_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        if not src.replay:
            hist, run_raw = src.read("history.jsonl"), src.read("run.json")
        try:
            run = json.loads(run_raw) if run_raw else None
        except ValueError:
            run = None
        est, _why = eta.run_time_left(lambda: eta.rows_of_bytes(hist), run, status)
        self._eta_cache = (key, est)
        return est

    def _stat_key(self, rel: str):
        """`(size, mtime_ns)` of a file of the run folder, or None: a change of the file with no read of it."""
        try:
            st = (self.source.run_dir / rel).stat()
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


def make_server(run_dir, port=0, replay=False, speed=1.0, badge=True, listen_fd=None) -> DashboardServer:
    """Make a server. Port 0 picks a free port. `listen_fd`: use this open listening socket (no bind; `port` is not used)."""
    return DashboardServer(run_dir, port, replay, speed, badge, listen_fd)


def start_in_thread(run_dir, port=8765, open_browser=False, replay=False, speed=1.0,
                    badge=True):
    """Start the server in a daemon thread. Return the server object."""
    server = make_server(run_dir, port, replay, speed, badge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if open_browser:
        webbrowser.open(server.url)
    return server


def _enter_listener(replay_state):
    """Start (or restart) the replay clock each time the user presses Enter."""
    import sys

    def loop():
        while True:
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                return
            if line == "":  # no terminal input (EOF): stop quietly
                return
            replay_state.start()
            print("Replay started.", flush=True)

    threading.Thread(target=loop, daemon=True).start()


def serve(run_dir, port=8765, open_browser=False, replay=False, speed=1.0, badge=True, listen_fd=None):
    """Run the server in the calling thread until Ctrl+C."""
    server = make_server(run_dir, port, replay, speed, badge, listen_fd)
    print(f"Dashboard: {server.url}", flush=True)
    if open_browser:
        webbrowser.open(server.url)
    if replay:
        _enter_listener(server.source.replay_state)
        print("Replay ready: press Enter to start", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="topopt dashboard")
    ap.add_argument("run_dir")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-badge", action="store_true", help="hide the REPLAY label")
    ap.add_argument("--listen-fd", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    serve(args.run_dir, args.port, args.open, args.replay, args.speed, not args.no_badge, args.listen_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
