"""Test the dashboard server against a fake run directory."""
import json
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fake_run  # noqa: E402
from topopt import dashboard_server as ds  # noqa: E402


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    fake_run.run(out, spi=0.0, n_it=6)
    return out


@pytest.fixture(scope="module")
def base(run_dir):
    server = ds.start_in_thread(run_dir, port=0)
    yield server.url.rstrip("/")
    server.shutdown()
    server.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def check_mesh(body):
    nv, nt = struct.unpack("<II", body[:8])
    assert len(body) == 8 + 12 * nv + 12 * nt
    return nv, nt


def test_index_and_static(base):
    code, hdr, body = get(base + "/")
    assert code == 200 and hdr["Content-Type"].startswith("text/html")
    assert b"topopt" in body
    code, hdr, _ = get(base + "/js/app.js")
    assert code == 200 and "javascript" in hdr["Content-Type"]
    code, hdr, _ = get(base + "/css/style.css")
    assert code == 200 and hdr["Content-Type"].startswith("text/css")
    code, hdr, _ = get(base + "/vendor/three.module.min.js")
    assert code == 200 and "javascript" in hdr["Content-Type"]


def test_no_path_escape(base):
    assert get(base + "/../topopt/dashboard_server.py")[0] == 404
    assert get(base + "/%2e%2e/CLAUDE.md")[0] == 404


def test_json_endpoints(base):
    code, hdr, body = get(base + "/api/run")
    run = json.loads(body)
    assert code == 200 and hdr["Content-Type"] == "application/json"
    assert run["shape"] == list(fake_run.SHAPE)
    status = json.loads(get(base + "/api/status")[2])
    assert status["state"] == "done"
    hist = json.loads(get(base + "/api/history")[2])
    assert [r["iteration"] for r in hist] == list(range(7))
    metrics = json.loads(get(base + "/api/metrics")[2])
    assert metrics["mass_final_g"] > 0


def test_binary_endpoints(base):
    nx, ny, nz = fake_run.SHAPE
    code, hdr, body = get(base + "/api/labels")
    assert code == 200 and hdr["Content-Type"] == "application/octet-stream"
    assert len(body) == nx * ny * nz
    code, _, body = get(base + "/api/density/3")
    assert code == 200 and len(body) == nx * ny * nz
    assert get(base + "/api/density/99")[0] == 404
    assert get(base + "/api/density/abc")[0] == 404


def test_mesh(base):
    code, hdr, body = get(base + "/api/mesh/3")
    assert code == 200 and hdr["Content-Type"] == "application/octet-stream"
    nv, nt = check_mesh(body)
    assert nv > 0 and nt > 0
    code, hdr, latest = get(base + "/api/mesh/latest")
    assert code == 200 and hdr["X-Iteration"] == "6"
    check_mesh(latest)
    assert get(base + "/api/mesh/99")[0] == 404
    # The second call comes from the cache and gives the same bytes.
    assert get(base + "/api/mesh/3")[2] == body


def test_final(base):
    code, _, body = get(base + "/api/final")
    assert code == 200
    nv, nt = check_mesh(body)
    assert nv > 0 and nt > 0


def test_part_file_names_the_final_stl_of_the_run_folder(base, run_dir):
    """P9-8: the result panel shows where the part is: the run folder name and result/final.stl (the full path is its tooltip)."""
    code, hdr, body = get(base + "/api/part_file")
    info = json.loads(body)
    assert code == 200 and hdr["Content-Type"] == "application/json"
    assert info == {"file": "result/final.stl", "run_dir": str(run_dir.resolve()), "run_name": run_dir.name, "exists": True}
    assert (run_dir / info["file"]).is_file()


def test_part_file_says_when_the_run_has_no_final_stl_yet(tmp_path):
    out = tmp_path / "live"
    fake_run.run(out, spi=0.0, n_it=2)
    (out / "result" / "final.stl").unlink()
    server = ds.start_in_thread(out, port=0)
    try:
        info = json.loads(get(server.url.rstrip("/") + "/api/part_file")[2])
    finally:
        server.shutdown()
        server.server_close()
    assert info["exists"] is False and info["run_name"] == "live" and info["file"] == "result/final.stl"


def test_source(base):
    code, hdr, body = get(base + "/api/source")
    src = json.loads(body)
    assert code == 200 and hdr["Content-Type"] == "application/json"
    assert not any(ds._MARKER.match(ln) for ln in src["lines"])
    for name in ("assemble", "solve", "sensitivity", "filter", "update", "snapshot"):
        a, b = src["steps"][name]
        assert 1 <= a <= b <= len(src["lines"])


def test_parse_source_blocks(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("a = 1\n# [step:solve]\nb = 2\nc = 3\n# [step:filter]\nd = 4\n# [end]\ne = 5\n")
    out = ds.parse_source(p)
    assert out["lines"] == ["a = 1", "b = 2", "c = 3", "d = 4", "e = 5"]
    assert out["steps"] == {"solve": [2, 3], "filter": [4, 4]}


def test_missing_files_before_run(tmp_path):
    server = ds.start_in_thread(tmp_path, port=0)
    try:
        url = server.url.rstrip("/")
        for ep in ("run", "status", "labels", "final", "metrics", "mesh/latest"):
            assert get(f"{url}/api/{ep}")[0] == 404, ep
        assert json.loads(get(url + "/api/history")[2]) == []
        assert get(url + "/api/nothing")[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_mesh_is_outward(run_dir):
    import numpy as np

    src = ds.RunSource(run_dir)
    body = ds.MeshCache(src).get(6)
    nv, nt = struct.unpack("<II", body[:8])
    v = np.frombuffer(body, "<f4", 3 * nv, 8).reshape(-1, 3).astype(float)
    f = np.frombuffer(body, "<u4", 3 * nt, 8 + 12 * nv).reshape(-1, 3)
    vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6
    assert vol > 0


def test_gray_start_shows_a_shape(base):
    # Iteration 0 is a uniform gray field. The adaptive level still gives a mesh.
    code, _, body = get(base + "/api/mesh/0")
    assert code == 200
    nv, nt = check_mesh(body)
    assert nv > 0 and nt > 0


def test_adaptive_level():
    import numpy as np

    lab = np.ones((4, 4, 4), np.uint8)
    flat = np.full((4, 4, 4), 0.2, np.float32)
    assert 0.05 <= ds.adaptive_level(flat, lab, 0.25) < 0.2
    sharp = np.zeros((4, 4, 4), np.float32)
    sharp[:2] = 1.0
    assert ds.adaptive_level(sharp, lab, 0.5) <= 0.5
    assert ds.adaptive_level(np.zeros((4, 4, 4), np.float32), lab, 0.25) == 0.05


def test_load_case_forces_in_run_json(base):
    run = json.loads(get(base + "/api/run")[2])
    for lc in run["load_cases"]:
        f = lc["forces"][0]
        assert len(f["force"]) == 3 and len(f["center"]) == 3


def test_new_run_clears_mesh_cache(tmp_path):
    fake_run.run(tmp_path, spi=0.0, n_it=2)
    src = ds.RunSource(tmp_path)
    cache = ds.MeshCache(src)
    assert cache.get(2) is not None
    assert len(cache._cache) == 1
    run = json.loads((tmp_path / "run.json").read_text())
    run["started_at"] = "2099-01-01T00:00:00.000+00:00"
    (tmp_path / "run.json").write_text(json.dumps(run))
    assert cache.get(1) is not None
    assert list(cache._cache) == [1]  # the old entry is gone
    (tmp_path / "run.json").unlink()
    assert cache.get(1) is None
    assert cache._cache == {}


# --- a listening socket that comes from the process before `os.execv` (P4 review, M1) ---------------------------------


def test_a_server_can_use_an_open_listening_socket_and_does_not_bind(run_dir):
    """`topopt run --dashboard` ends as `topopt dashboard --listen-fd N`: the new program takes the socket that the old one
    left open. The port is that of the socket, and the server answers on it. (`os.dup` stands for the fd that the exec passes.)"""
    import os
    import socket
    import threading

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = ds.make_server(run_dir, port=1, listen_fd=os.dup(sock.fileno()))        # `port` is not used
    assert server.server_address == ("127.0.0.1", port) and server.url == f"http://127.0.0.1:{port}/"
    # a client that comes before the server runs waits in the backlog and gets its answer when the server starts
    results = []
    client = threading.Thread(target=lambda: results.append(get(f"http://127.0.0.1:{port}/api/status")))
    client.start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client.join(timeout=10)
    assert results and results[0][0] == 200 and json.loads(results[0][2])["state"] == "done"
    server.shutdown()
    server.server_close()
    sock.close()


def test_a_listen_fd_that_is_not_a_listening_socket_is_an_error_with_the_fd_in_the_text(run_dir):
    import os
    import socket

    r, w = os.pipe()
    with pytest.raises(OSError, match=f"file descriptor {r} is not an open socket"):
        ds.make_server(run_dir, listen_fd=r)
    os.close(w)
    plain = socket.socket()                       # a socket that does not listen
    with pytest.raises(OSError, match="does not listen"):
        ds.make_server(run_dir, listen_fd=os.dup(plain.fileno()))
    plain.close()


def _without_so_acceptconn(monkeypatch):
    """Make `getsockopt(SOL_SOCKET, SO_ACCEPTCONN)` fail as it does on macOS (`ENOPROTOOPT`: 42 there, 92 on Linux)."""
    import errno
    import socket

    real = socket.socket.getsockopt

    def getsockopt(self, level, name, *rest):
        if level == socket.SOL_SOCKET and name == socket.SO_ACCEPTCONN:
            raise OSError(errno.ENOPROTOOPT, "Protocol not available")
        return real(self, level, name, *rest)

    monkeypatch.setattr(socket.socket, "getsockopt", getsockopt)


def test_a_system_without_so_acceptconn_still_takes_a_listening_socket(run_dir, monkeypatch):
    """macOS gives `ENOPROTOOPT` for `SO_ACCEPTCONN`: the hand-over must still work, and the server must answer on the port."""
    import os
    import socket
    import threading

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    _without_so_acceptconn(monkeypatch)
    server = ds.make_server(run_dir, listen_fd=os.dup(sock.fileno()))
    assert server.server_address == ("127.0.0.1", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    status, _headers, body = get(f"http://127.0.0.1:{port}/api/status")
    assert status == 200 and json.loads(body)["state"] == "done"
    server.shutdown()
    server.server_close()
    sock.close()


def test_a_system_without_so_acceptconn_takes_a_bound_socket_that_does_not_listen_and_makes_it_listen(run_dir, monkeypatch):
    """The fallback calls `listen()`: a bound stream socket that did not listen is taken, and a client gets its answer."""
    import os
    import socket
    import threading

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))                   # bound, no listen()
    port = sock.getsockname()[1]
    _without_so_acceptconn(monkeypatch)
    server = ds.make_server(run_dir, listen_fd=os.dup(sock.fileno()))
    assert server.server_address == ("127.0.0.1", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    status, _headers, body = get(f"http://127.0.0.1:{port}/api/status")
    assert status == 200 and json.loads(body)["state"] == "done"
    server.shutdown()
    server.server_close()
    sock.close()


def test_a_system_without_so_acceptconn_says_does_not_listen_when_listen_fails(run_dir, monkeypatch):
    """A bound socket for which `listen()` raises `OSError` (a connected one on macOS) gives the Linux error, not "not an open socket"."""
    import os
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _without_so_acceptconn(monkeypatch)

    def refuse(self, *args):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(socket.socket, "listen", refuse)
    with pytest.raises(OSError, match="does not listen"):
        ds.make_server(run_dir, listen_fd=os.dup(sock.fileno()))
    sock.close()


def test_a_system_without_so_acceptconn_gives_the_same_errors_for_a_bad_fd(run_dir, monkeypatch):
    import os
    import socket

    _without_so_acceptconn(monkeypatch)
    r, w = os.pipe()
    with pytest.raises(OSError, match=f"file descriptor {r} is not an open socket"):
        ds.make_server(run_dir, listen_fd=r)
    os.close(w)
    plain = socket.socket()                       # not bound, not listening: it must stay so (no port is taken)
    with pytest.raises(OSError, match="does not listen"):
        ds.make_server(run_dir, listen_fd=os.dup(plain.fileno()))
    assert plain.getsockname()[1] == 0
    plain.close()
    udp = socket.socket(type=socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    with pytest.raises(OSError, match="does not listen"):
        ds.make_server(run_dir, listen_fd=os.dup(udp.fileno()))
    udp.close()


def test_the_dashboard_command_takes_a_hidden_listen_fd_and_names_a_bad_one(run_dir, capsys):
    import os

    from topopt.cli import build_parser, main

    args = build_parser().parse_args(["dashboard", str(run_dir), "--listen-fd", "7"])
    assert args.listen_fd == 7
    assert "--listen-fd" not in build_parser().format_help() and "--listen-fd" not in capsys.readouterr().out
    r, w = os.pipe()
    assert main(["dashboard", str(run_dir), "--listen-fd", str(r), "--no-open"]) == 1
    assert f"--listen-fd {r}" in capsys.readouterr().err
    os.close(w)


def test_the_page_resets_the_view_only_when_run_json_is_missing_not_on_a_network_error():
    """P4 review, M1. `watchRun` asked `getJSON("/api/run")`, which gives null for a network error too, so a server that restarts
    (the exec of `topopt run --dashboard`) cleared the scene. Now `getRun` tells HTTP 404 from the rest, and only 404 resets."""
    js = (ROOT / "topopt/dashboard/js/app.js").read_text()
    get_run = js[js.index("async function getRun()"):js.index("async function getBuf(")]
    assert "r.status === 404" in get_run and "catch (e) { return { missing: false }; }" in get_run
    watch = js[js.index("async function watchRun()"):js.index("$(\"view\").classList.add(\"waiting\");\nrequestAnimationFrame")]
    assert 'getJSON("/api/run")' not in watch and "await getRun()" in watch
    assert "if (missing && S.run) resetView();" in watch
