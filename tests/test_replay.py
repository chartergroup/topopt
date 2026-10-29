"""Test replay mode: the dashboard shows a finished run on a replay clock."""
import json
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fake_run  # noqa: E402
from topopt import dashboard_server as ds  # noqa: E402

N_IT = 6
END_S = 60.0


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("replay_run")
    fake_run.run(out, spi=0.0, n_it=N_IT)
    # Set clean times: iteration i ends at 10 + 5*i s. The run ends at 60 s.
    rows = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines()]
    for r in rows:
        r["elapsed_s"] = 10.0 + 5.0 * r["iteration"]
    (out / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    st = json.loads((out / "status.json").read_text())
    st["elapsed_s"] = END_S
    (out / "status.json").write_text(json.dumps(st))
    m = json.loads((out / "result" / "metrics.json").read_text())
    m["run_time_s"] = END_S
    (out / "result" / "metrics.json").write_text(json.dumps(m))
    return out


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture()
def rp(run_dir):
    """A replay server with a clock the test controls."""
    server = ds.start_in_thread(run_dir, port=0, replay=True, speed=1.0)
    clock = Clock()
    server.source.replay_state.now = clock
    yield server, clock
    server.shutdown()
    server.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def jget(base, path):
    code, _h, body = get(base + path)
    return code, (json.loads(body) if code == 200 else None)


def base_of(server):
    return server.url.rstrip("/")


def start(server, clock):
    clock.t = 100.0  # the clock value at the start does not matter
    assert get(base_of(server) + "/api/replay/start")[0] == 200


def test_before_start_nothing_leaks(rp):
    server, _clock = rp
    b = base_of(server)
    for path in ("/api/run", "/api/status", "/api/labels", "/api/final", "/api/metrics",
                 "/api/mesh/latest", "/api/mesh/0", "/api/density/0"):
        assert get(b + path)[0] == 404, path
    assert jget(b, "/api/history") == (200, [])
    assert get(b + "/api/log")[2] == b""
    assert get(b + "/")[0] == 200  # the page itself is served


def test_iteration_grows_with_clock(rp):
    server, clock = rp
    b = base_of(server)
    start(server, clock)
    st = jget(b, "/api/status")[1]
    assert st["state"] == "starting" and st["mode"] == "replay"
    assert jget(b, "/api/run")[1]["name"] == "fake_hook"
    assert get(b + "/api/labels")[0] == 200
    last = -1
    for t, expect in ((9.0, -1), (10.0, 0), (17.0, 1), (30.0, 4), (40.0, 6)):
        clock.t = 100.0 + t
        hist = jget(b, "/api/history")[1]
        shown = hist[-1]["iteration"] if hist else -1
        assert shown == expect and shown >= last
        last = shown
    st = jget(b, "/api/status")[1]
    assert st["state"] == "postprocess" and st["iteration"] == N_IT


def test_no_future_data(rp):
    server, clock = rp
    b = base_of(server)
    start(server, clock)
    clock.t = 100.0 + 17.0  # iteration 1 is shown. Iteration 2 ends at 20 s.
    assert [r["iteration"] for r in jget(b, "/api/history")[1]] == [0, 1]
    assert get(b + "/api/density/1")[0] == 200
    assert get(b + "/api/density/2")[0] == 404
    assert get(b + "/api/mesh/2")[0] == 404
    code, hdr, body = get(b + "/api/mesh/latest")
    assert code == 200 and hdr["X-Iteration"] == "1"
    nv, nt = struct.unpack("<II", body[:8])
    assert len(body) == 8 + 12 * nv + 12 * nt
    log = get(b + "/api/log")[2].decode()
    assert "it   1" in log and "it   2" not in log
    assert get(b + "/api/final")[0] == 404
    assert get(b + "/api/metrics")[0] == 404


def test_steps_cycle_in_an_iteration(rp):
    server, clock = rp
    b = base_of(server)
    start(server, clock)
    seen = []
    for t in (10.4, 11.2, 12.1, 12.9, 13.7, 14.6):  # between rows 0 and 1
        clock.t = 100.0 + t
        st = jget(b, "/api/status")[1]
        assert st["state"] == "running" and st["iteration"] == 0
        seen.append(st["step"])
    assert seen == ["assemble", "solve", "sensitivity", "filter", "update", "snapshot"]


def test_result_only_at_the_end(rp):
    server, clock = rp
    b = base_of(server)
    start(server, clock)
    clock.t = 100.0 + 59.0
    assert get(b + "/api/final")[0] == 404
    assert get(b + "/api/metrics")[0] == 404
    assert "done" not in get(b + "/api/log")[2].decode().splitlines()[-1:]
    clock.t = 100.0 + 60.0
    assert get(b + "/api/final")[0] == 200
    code, m = jget(b, "/api/metrics")
    assert code == 200 and m["run_time_s"] == END_S
    st = jget(b, "/api/status")[1]
    assert st["state"] == "done" and st["step"] == "done" and st["elapsed_s"] == END_S
    assert get(b + "/api/log")[2].decode().strip().endswith("done")


def test_restart_gives_new_started_at(rp):
    server, clock = rp
    b = base_of(server)
    start(server, clock)
    clock.t = 100.0 + 30.0
    first = jget(b, "/api/run")[1]["started_at"]
    time.sleep(0.01)
    start(server, clock)  # restart: the clock goes back to 0
    second = jget(b, "/api/run")[1]["started_at"]
    assert second != first
    assert jget(b, "/api/history")[1] == []
    assert get(b + "/api/density/3")[0] == 404
    assert jget(b, "/api/status")[1]["state"] == "starting"


def test_speed_and_real_clock(run_dir):
    server = ds.start_in_thread(run_dir, port=0, replay=True, speed=200.0)
    try:
        b = base_of(server)
        assert get(b + "/api/replay/start")[0] == 200
        deadline = time.time() + 5
        while time.time() < deadline:
            if jget(b, "/api/status")[1]["state"] == "done":
                break
            time.sleep(0.05)
        assert jget(b, "/api/status")[1]["state"] == "done"
        assert get(b + "/api/final")[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_no_badge(run_dir):
    server = ds.start_in_thread(run_dir, port=0, replay=True, badge=False)
    try:
        b = base_of(server)
        get(b + "/api/replay/start")
        assert "mode" not in jget(b, "/api/status")[1]
    finally:
        server.shutdown()
        server.server_close()


def test_live_server_has_no_replay_endpoint(run_dir):
    server = ds.start_in_thread(run_dir, port=0)
    try:
        b = base_of(server)
        assert get(b + "/api/replay/start")[0] == 404
        assert "mode" not in jget(b, "/api/status")[1]
    finally:
        server.shutdown()
        server.server_close()
