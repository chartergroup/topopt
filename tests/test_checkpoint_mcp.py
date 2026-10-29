"""MCP side of checkpoint and resume (D1b): `run_topopt(resume=...)`, `run_status` with `checkpoint` and `verdict`."""
import os
import signal
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt_mcp import server  # noqa: E402

from test_checkpoint import edit, killed, tree_state  # noqa: E402
from test_optimize import make_case  # noqa: E402


@pytest.fixture(autouse=True)
def clean_server(monkeypatch):
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    server._current.clear()
    yield
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._current.clear()


@pytest.fixture
def half(tmp_path):
    """A run that was killed after pass 6 (24 passes planned): `(yaml, folder)`."""
    yaml_path = make_case(tmp_path, h=1.0, iterations=24)
    out = tmp_path / "run"
    killed(yaml_path, out, "after_checkpoint:6")
    return yaml_path, out


def wait_done(call, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        st = call()
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.3)
    raise AssertionError("run did not finish")


def test_the_command_has_the_flags_in_order(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_MCP_CMD", raising=False)
    y, o = tmp_path / "a.yaml", tmp_path / "o"
    base = server._command(y, o, 8765)
    assert "--resume" not in base and "--stop-rules" not in base and "--no-stop-rules" not in base
    cmd = server._command(y, o, 8765, True, True, True, False)
    assert cmd[len(base):] == ["--override-lock", "--resume", "--allow-changed-inputs", "--no-stop-rules"]
    assert server._command(y, o, 8765, stop_rules=True)[len(base):] == ["--stop-rules"]
    assert server._command(y, o, 8765, resume=True)[len(base):] == ["--resume"]


def test_a_resume_refusal_has_facts_options_and_the_checkpoint_and_writes_nothing(half, monkeypatch):
    yaml_path, out = half
    edit(yaml_path, lambda d: d["settings"].update(volume_fraction=0.35))
    before = tree_state(out)
    r = server.run_topopt(str(yaml_path), str(out), resume=True)
    assert "settings.volume_fraction: 0.3 -> 0.35" in r["error"] and r["run_dir"] == str(out)
    assert r["resume"]["facts"] == ["settings.volume_fraction: 0.3 -> 0.35"]
    assert any("allow_changed_inputs=true" in o for o in r["resume"]["options"])
    assert r["resume"]["checkpoint"]["iteration"] == 6 and not r["resume"]["checkpoint"]["complete"]
    assert "pid" not in r and tree_state(out) == before and not server._pid_file(out).exists()


def test_a_resume_without_a_checkpoint_is_refused(tmp_path):
    yaml_path = make_case(tmp_path, h=1.0, iterations=6)
    (tmp_path / "run").mkdir()
    r = server.run_topopt(str(yaml_path), str(tmp_path / "run"), resume=True)
    assert "no checkpoint" in r["error"] and r["resume"]["checkpoint"] is None
    assert any("without --resume" in o for o in r["resume"]["options"])


def test_a_refusal_does_not_stop_the_old_dashboard_process(half, monkeypatch):
    yaml_path, out = half
    edit(yaml_path, lambda d: d["settings"].update(volume_fraction=0.35))
    stopped = []
    monkeypatch.setattr(server, "_alive_run", lambda: (out.parent / "other", 12345))
    monkeypatch.setattr(server, "_is_busy", lambda d: False)
    monkeypatch.setattr(server, "_stop", lambda *a: stopped.append(a))
    r = server.run_topopt(str(yaml_path), str(out), resume=True)
    assert "error" in r and "resume" in r and stopped == []


def test_the_override_starts_the_run_with_the_flags(half, monkeypatch):
    yaml_path, out = half
    edit(yaml_path, lambda d: d["settings"].update(volume_fraction=0.35))
    seen = []
    monkeypatch.setattr(server, "_command", lambda *a: seen.append(a) or [sys.executable, "-c", "pass"])
    r = server.run_topopt(str(yaml_path), str(out), resume=True, allow_changed_inputs=True, stop_rules=True)
    assert "error" not in r and seen[0][3:] == (False, True, True, True)
    server._current["proc"].wait()


def test_run_status_of_a_killed_run_says_how_to_resume(half):
    yaml_path, out = half
    st = server.run_status(str(out))
    cp = st["checkpoint"]
    assert (cp["iteration"], cp["iterations"], cp["complete"]) == (6, 24, False)
    assert cp["yaml_path"] == str(yaml_path) and st["verdict"] is None
    assert "checkpoint at pass 6 of 24: run_topopt with resume=true continues the run" in st["summary"]


def test_the_early_replies_have_the_verdict_and_checkpoint_keys(tmp_path):
    st = server.run_status(str(tmp_path / "none"))
    assert st["state"] == "none" and st["verdict"] is None and st["checkpoint"] is None


def test_a_resume_through_the_server_ends_with_the_verdict_and_no_resume_hint(half):
    yaml_path, out = half
    r = server.run_topopt(str(yaml_path), str(out), resume=True)
    assert "error" not in r, r
    st = wait_done(lambda: server.run_status(str(out)))
    assert st["state"] == "done", st
    assert st["verdict"]["state"] == "iteration_limit" and st["verdict"]["resume_points"][0]["iteration"] == 6
    assert st["checkpoint"]["complete"] is True and "continues the run" not in st["summary"]
    assert "verdict: iteration limit reached at pass 24" in st["summary"]
    res = server.run_result(str(out))
    assert res["done"] is True and res["metrics"]["verdict"] == st["verdict"]
    log = (out / "run.log").read_text()
    assert "resume: continue from checkpoint.npz, iteration 6 of 24" in log


def test_the_resume_hint_is_not_shown_while_a_cli_run_is_alive(half, monkeypatch):
    """Review n3: a run that the CLI started has no pid file of this server. The job state `running` says that it is alive."""
    yaml_path, out = half
    monkeypatch.setattr(server, "_job_facts", lambda *a: {"state": "running", "summary": "job: running (pid 1)"})
    st = server.run_status(str(out))
    assert st["checkpoint"]["iteration"] == 6 and "continues the run" not in st["summary"]
    monkeypatch.setattr(server, "_job_facts", lambda *a: {"state": "stale_lock", "summary": "job: stale lock"})
    assert "continues the run" in server.run_status(str(out))["summary"]


def test_a_run_without_resume_says_that_it_deletes_the_checkpoint_of_an_unfinished_run(half, monkeypatch):
    """Review m8: the reply has `earlier_checkpoint`, and the run starts."""
    yaml_path, out = half
    monkeypatch.setattr(server, "_command", lambda *a: [sys.executable, "-c", "pass"])
    r = server.run_topopt(str(yaml_path), str(out))
    server._current["proc"].wait()
    assert "error" not in r and r["earlier_checkpoint"]["iteration"] == 6 and r["earlier_checkpoint"]["iterations"] == 24
    note = r["earlier_checkpoint"]["note"]
    assert "checkpoint at pass 6 of 24" in note and "This run deletes it" in note and "resume=true" in note
    server._current.clear()
    r = server.run_topopt(str(yaml_path), str(out), resume=True)          # a resume does not delete it: no such key
    server._current["proc"].wait()
    assert "error" not in r and "earlier_checkpoint" not in r
