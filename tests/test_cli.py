"""Tests for the command line."""

import json
import socket

import pytest
import yaml

from topopt.cli import estimate_seconds, main
from test_optimize import make_case


def test_estimate_anchor():
    per_it, total = estimate_seconds(109_548, 80)
    assert abs(per_it - 1.30) < 1e-9
    assert total > 80 * 1.3


def test_check(tmp_path, capsys):
    assert main(["check", str(make_case(tmp_path, h=2.0))]) == 0
    out = capsys.readouterr().out
    assert "grid shape" in out and "active voxels" in out and "free dofs" in out
    assert "time estimate" in out


def test_missing_file(tmp_path, capsys):
    assert main(["check", str(tmp_path / "nope.yaml")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and len(err.strip().splitlines()) == 1


def test_bad_yaml_one_line(tmp_path, capsys):
    p = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["bogus"] = 1
    p.write_text(yaml.safe_dump(d))
    assert main(["run", str(p), "--out", str(tmp_path / "r")]) == 1
    err = capsys.readouterr().err
    lines = err.strip().splitlines()
    assert len(lines) == 1 and "bogus" in lines[0] and "Traceback" not in err
    assert not (tmp_path / "r" / "run.json").exists()   # old run is not touched on a bad YAML


def test_guard_error_exit_1(tmp_path, capsys):
    p = make_case(tmp_path, h=1.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["max_active_voxels"] = 10
    p.write_text(yaml.safe_dump(d))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 1
    assert "max_active_voxels" in capsys.readouterr().err
    assert json.loads((out / "status.json").read_text())["state"] == "error"


def test_run_and_cleanup(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=3)
    out = tmp_path / "r"
    (out / "snapshots").mkdir(parents=True)
    (out / "snapshots" / "it_0099.bin").write_bytes(b"old")
    (out / "result").mkdir()
    (out / "result" / "final.stl").write_bytes(b"old")
    (out / "history.jsonl").write_text("old\n")
    (out / "notes.txt").write_text("keep me")
    (out / "other").mkdir()
    (out / "other" / "x.txt").write_text("keep me too")
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert (out / "notes.txt").read_text() == "keep me"
    assert (out / "other" / "x.txt").read_text() == "keep me too"
    assert not (out / "snapshots" / "it_0099.bin").exists()
    assert len(list((out / "snapshots").iterdir())) == 4
    assert len((out / "history.jsonl").read_text().splitlines()) == 4
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "done" and status["step"] == "done"
    assert (out / "run.log").read_text().count("done in") == 1


def test_busy_port(tmp_path, capsys):
    p = make_case(tmp_path, h=2.0, iterations=2)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert main(["run", str(p), "--out", str(tmp_path / "r"), "--dashboard",
                     "--port", str(port), "--no-wait"]) == 0
    finally:
        s.close()
    out = capsys.readouterr().out
    assert f"Dashboard port {port} is busy" in out
    assert f"http://127.0.0.1:{port}/" not in out  # the next free port is used


def test_dashboard_command_on_a_busy_port_is_a_user_error_and_names_the_port_option(tmp_path, capsys):
    """Q1-04: no traceback and no "internal error": exit code 1 (a user error), the busy port and the `--port` option."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert main(["dashboard", str(tmp_path), "--port", str(port), "--no-open"]) == 1
    finally:
        s.close()
    err = capsys.readouterr().err
    assert f"error: dashboard port {port} is busy" in err and "--port N" in err
    assert "Traceback" not in err and "internal error" not in err
