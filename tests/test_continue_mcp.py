"""MCP side of a continuation (P4b-5): `run_topopt(continue_from=..., more_iterations=...)` (CONTRACTS.md section 24.3).

The reply of a refusal has `error`, `run_dir` and `continue` (`facts`, `options`, `checkpoint` of the parent). The reply of a run that starts has
`continue` (the pass of the checkpoint, the pass count, whether the pass runs again, the step settings and the facts). Every refusal reads only:
no folder, no pid file, and the parent does not change.
"""
import inspect
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt_mcp import server  # noqa: E402

from test_checkpoint import edit, finish, history, log, needs_direct, same_run  # noqa: E402
from test_continue import DROP, parent_state, read_json  # noqa: E402
from test_optimize import make_case  # noqa: E402

K, TOTAL = 12, 24


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


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """The cantilever (24 passes), a finished parent of 12 passes (made by the CLI, 1 thread) and the path of the run of all 24 passes.
    The run of all 24 passes is made by the fixture `full_run`, for the test that reads it; the other tests do not need it.
    `full_run` can run after the other tests and reads `yaml_path`: a test edits a copy of the YAML, never `yaml_path` itself."""
    root = tmp_path_factory.mktemp("cont_mcp")
    (root / "case").mkdir()
    yaml_path = make_case(root / "case", h=1.0, iterations=TOTAL)
    base = root / "base"
    edit(yaml_path, lambda d: d["settings"].update(iterations=K))
    parent = root / "parent"
    finish(yaml_path, parent)
    edit(yaml_path, lambda d: d["settings"].update(iterations=TOTAL))
    return yaml_path, base, parent


@pytest.fixture(scope="module")
def full_run(case):
    """The run of all 24 passes (1 thread) in the folder `case[1]`."""
    finish(case[0], case[1])
    return case[1]


def wait_done(call, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        st = call()
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.3)
    raise AssertionError("run did not finish")


def no_op_command(seen):
    def command(*args, **kwargs):
        seen.append((args, kwargs))
        return [sys.executable, "-c", "pass"]
    return command


def test_the_parameters_are_optional_and_the_description_names_them():
    params = inspect.signature(server.run_topopt).parameters
    assert params["continue_from"].default is None and params["more_iterations"].default is None
    doc = server.run_topopt.__doc__
    assert "continue_from" in doc and "more_iterations" in doc and "K + more_iterations" in doc


def test_the_command_has_the_continue_flags_after_the_override_and_before_the_stop_rules(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_MCP_CMD", raising=False)
    y, o, p = tmp_path / "a.yaml", tmp_path / "o", tmp_path / "parent"
    base = server._command(y, o, 8765)
    assert "--continue-from" not in base and "--more-iterations" not in base
    cmd = server._command(y, o, 8765, True, False, True, False, continue_from=p, more_iterations=7)
    assert cmd[len(base):] == ["--override-lock", "--allow-changed-inputs", "--continue-from", str(p), "--more-iterations", "7", "--no-stop-rules"]
    assert server._command(y, o, 8765, continue_from=p, more_iterations=3)[len(base):] == ["--continue-from", str(p), "--more-iterations", "3"]


def test_a_run_that_starts_has_the_plan_in_continue_and_the_flags_in_the_command(case, tmp_path, monkeypatch):
    yaml_path, _base, parent = case
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    before = parent_state(parent)
    out = tmp_path / "new"
    r = server.run_topopt(str(yaml_path), str(out), continue_from=str(parent), more_iterations=5)
    server._current["proc"].wait()
    assert "error" not in r, r
    c = r["continue"]
    assert (c["parent"], c["iteration"], c["parent_iterations"], c["iterations"], c["more_iterations"], c["pass_runs_again"]) == (
        str(parent), K, K, K + 5, 5, True)
    assert c["step_settings"]["first_pass"] == K and c["step_settings"]["last_pass"] == K + 5
    assert any(f.startswith("continue: a new run folder from") for f in c["facts"]) and any("pass count of this run is 17" in f for f in c["facts"])
    # The server does not resolve `auto` (the run process can be another Python): the reply shows the setting, and says who picks the solver.
    assert c["step_settings"]["solver"] == "auto"
    assert "continue: solver auto (the run process picks the solver; run.json of the run names it)." in c["facts"]
    args, kwargs = seen[0]
    assert args[3:] == (False, False, False, None) and kwargs == {"continue_from": parent, "more_iterations": 5}
    assert parent_state(parent) == before                                     # the plan reads the parent only


def test_a_parent_that_runs_is_named_and_the_reply_says_that_the_run_reads_the_checkpoint_again(case, tmp_path, monkeypatch):
    """Fix round 1, finding 1: the pre-check and the run process are two processes, so the pass of the reply can be older than the run's."""
    yaml_path, _base, parent = case
    running = tmp_path / "running"
    shutil.copytree(parent, running)
    (running / "job.lock").write_text(json.dumps({"pid": os.getpid()}))                   # a live holder: the parent runs
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    before = parent_state(running)
    r = server.run_topopt(str(yaml_path), str(tmp_path / "new"), continue_from=str(running), more_iterations=5)
    server._current["proc"].wait()
    assert "error" not in r, r
    facts = r["continue"]["facts"]
    assert any(f.startswith("continue: the parent is running now") and f"checkpoint that it read, at pass {K}" in f for f in facts), facts
    assert any(f.startswith("continue: the numbers in this reply come from a read before the run process starts") and "reads its checkpoint again" in f
               and "name the pass that the run read" in f for f in facts), facts
    assert parent_state(running) == before                                                # no refusal, and the parent is only read
    r = server.run_topopt(str(yaml_path), str(tmp_path / "new2"), continue_from=str(parent), more_iterations=5)       # the finished parent: no such fact
    server._current["proc"].wait()
    assert not any("reads its checkpoint again" in f or "is running now" in f for f in r["continue"]["facts"])


@pytest.mark.parametrize("kwargs, text", [
    ({"continue_from": "P"}, "continue_from and more_iterations go together"),
    ({"more_iterations": 5}, "continue_from and more_iterations go together"),
    ({"continue_from": "P", "more_iterations": 5, "resume": True}, "resume goes on in out_dir"),
    ({"continue_from": "P", "more_iterations": 0}, "must be a whole number of 1 or more"),
    ({"continue_from": "P", "more_iterations": -2}, "must be a whole number of 1 or more"),
    ({"continue_from": "P", "more_iterations": 2.5}, "must be a whole number of 1 or more"),
    ({"continue_from": "P", "more_iterations": "abc"}, "must be a whole number of 1 or more"),
    ({"continue_from": "P", "more_iterations": True}, "must be a whole number of 1 or more"),
    ({"continue_from": "P", "more_iterations": "\u00b2"}, "must be a whole number of 1 or more"),
    ({"continue_from": "P", "more_iterations": "--5"}, "must be a whole number of 1 or more")])
def test_a_wrong_mix_or_number_is_an_error_that_writes_nothing(case, tmp_path, monkeypatch, kwargs, text):
    yaml_path, _base, parent = case
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    out = tmp_path / "new"
    before = parent_state(parent)
    r = server.run_topopt(str(yaml_path), str(out), **{k: (str(parent) if v == "P" else v) for k, v in kwargs.items()})
    assert text in r["error"] and "more_iterations" in r["error"] and "--more-iterations" not in r["error"]
    assert "pid" not in r and "continue" not in r and not out.exists() and seen == [] and not server._pid_file(out).exists()
    assert parent_state(parent) == before


def test_a_new_folder_that_is_the_parent_or_inside_it_is_refused(case, monkeypatch):
    yaml_path, _base, parent = case
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    before = parent_state(parent)
    r = server.run_topopt(str(yaml_path), str(parent), continue_from=str(parent), more_iterations=5)
    assert "--out is the parent run folder" in r["error"] and "pid" not in r
    r = server.run_topopt(str(yaml_path), str(parent / "more"), continue_from=str(parent), more_iterations=5)
    assert "is inside the parent run folder" in r["error"] and not (parent / "more").exists()
    assert seen == [] and parent_state(parent) == before and not server._pid_file(parent).exists()


def test_a_refusal_has_facts_options_and_the_checkpoint_and_writes_nothing(case, tmp_path, monkeypatch):
    yaml_path, _base, parent = case
    work = tmp_path / "case"
    shutil.copytree(yaml_path.parent, work)
    edit(work / yaml_path.name, lambda d: d["settings"].update(volume_fraction=0.35))
    out = tmp_path / "new"
    before = parent_state(parent)
    r = server.run_topopt(str(work / yaml_path.name), str(out), continue_from=str(parent), more_iterations=5)
    assert "settings.volume_fraction: 0.3 -> 0.35" in r["error"] and r["run_dir"] == str(out)
    assert r["continue"]["facts"] == ["settings.volume_fraction: 0.3 -> 0.35"]
    assert any("allow_changed_inputs=true" in o for o in r["continue"]["options"])
    assert any("run the YAML as a new run in a new folder" in o for o in r["continue"]["options"])
    assert r["continue"]["checkpoint"]["iteration"] == K and r["continue"]["checkpoint"]["complete"] is True
    assert "pid" not in r and not out.exists() and not server._pid_file(out).exists() and parent_state(parent) == before


def test_a_parent_with_no_checkpoint_is_refused(case, tmp_path):
    yaml_path, _base, _parent = case
    empty, out = tmp_path / "empty", tmp_path / "new"
    empty.mkdir()
    r = server.run_topopt(str(yaml_path), str(out), continue_from=str(empty), more_iterations=5)
    assert "no checkpoint" in r["error"] and r["continue"]["checkpoint"] is None and not out.exists()
    assert list(empty.iterdir()) == []


def test_a_refusal_does_not_stop_the_old_dashboard_process(case, tmp_path, monkeypatch):
    yaml_path, _base, parent = case
    stopped = []
    monkeypatch.setattr(server, "_alive_run", lambda: (tmp_path / "other", 12345))
    monkeypatch.setattr(server, "_is_busy", lambda d: False)
    monkeypatch.setattr(server, "_stop", lambda *a: stopped.append(a))
    r = server.run_topopt(str(yaml_path), str(parent), continue_from=str(parent), more_iterations=5)
    assert "error" in r and stopped == []


def test_the_override_starts_the_run_with_the_flags(case, tmp_path, monkeypatch):
    yaml_path, _base, parent = case
    work = tmp_path / "case"
    shutil.copytree(yaml_path.parent, work)
    edit(work / yaml_path.name, lambda d: d["settings"].update(volume_fraction=0.35))
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    r = server.run_topopt(str(work / yaml_path.name), str(tmp_path / "new"), continue_from=str(parent), more_iterations=5,
                          allow_changed_inputs=True, stop_rules=True)
    server._current["proc"].wait()
    assert "error" not in r and seen[0][0][3:] == (False, False, True, True)
    assert any("the inputs changed since the checkpoint: settings.volume_fraction" in f for f in r["continue"]["facts"])


@needs_direct
def test_a_continuation_through_the_server_ends_with_the_verdict_and_the_numbers_of_the_full_run(case, full_run, tmp_path):
    """The server does not set one thread: the numbers are close to the full run (rtol 1e-6), not equal byte for byte."""
    yaml_path, base, parent = case
    before = parent_state(parent)
    out = tmp_path / "new"
    r = server.run_topopt(str(yaml_path), str(out), continue_from=str(parent), more_iterations=TOTAL - K)
    assert "error" not in r, r
    assert r["continue"]["iterations"] == TOTAL and r["continue"]["pass_runs_again"] is True
    st = wait_done(lambda: server.run_status(str(out)))
    assert st["state"] == "done", st
    assert st["checkpoint"]["complete"] is True and st["checkpoint"]["continued_from"]["iteration"] == K
    assert st["verdict"]["state"] == read_json(base / "verdict.json")["state"]
    same_run(base, out, exact=False, rtol=1e-6, run_json_drop=DROP)
    assert len(history(out)) == TOTAL + 1 and "continue: a new run folder from" in log(out)
    res = server.run_result(str(out))
    assert res["done"] is True
    assert parent_state(parent) == before
