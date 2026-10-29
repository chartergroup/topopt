"""F4-48 (#260): the child processes of the MCP server start without the work folder on their module path.

`python -m` and `python -c` put the current folder first on `sys.path`. The server starts its children in the work folder, so a user
file there (`timeit.py`, `json.py`, `random.py`) took the place of a standard module and broke every run (numba imports `timeit`).
`_child_env` now sets `PYTHONSAFEPATH=1` (CONTRACTS.md section 10.1), and the launcher starts the server with `python -P`.

Safety of these tests. A shadow file is a stand-in for a module. It only writes a marker file: no import, no process, no loop
(`test_a_shadow_file_only_writes_its_marker_file` checks the text). A shadow file lives in a folder of pytest (`tmp_path`) and this
process never changes its current folder to it.
"""
import ast
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt_mcp import server  # noqa: E402

from test_optimize import make_case  # noqa: E402

SERVER_SOURCE = REPO / "topopt_mcp" / "server.py"
LAUNCHER = REPO / "scripts" / "topopt-env.sh"


@pytest.fixture(autouse=True)
def clean_server(monkeypatch):
    for name in ("TOPOPT_PYTHON", "TOPOPT_MCP_CMD", "TOPOPT_DEFAULT_WORKSPACE", "PYTHONSAFEPATH"):
        monkeypatch.delenv(name, raising=False)
    server._current.clear()
    yield
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._current.clear()


def shadow(folder: Path, name: str, marker: Path) -> Path:
    """A harmless stand-in for the module `name` in `folder`: it only writes the marker file."""
    path = folder / f"{name}.py"
    path.write_text(f"open({str(marker)!r}, 'w').write('imported')\n", encoding="utf-8")
    return path


def work_folder_with(tmp_path: Path, name: str):
    """`(work folder, marker)`: a work folder that holds a shadow file for the module `name`. The marker is outside the work folder."""
    work = tmp_path / "work"
    work.mkdir()
    marker = tmp_path / f"imported-{name}.txt"
    shadow(work, name, marker)
    return work, marker


def wait_done(call, proc, timeout=180):
    """The state of the run when it is `done` or `error`, or when its process has ended (the last state then)."""
    end = time.time() + timeout
    while time.time() < end:
        st = call()
        if st["state"] in ("done", "error"):
            return st
        if proc.poll() is not None:
            return call()
        time.sleep(0.3)
    raise AssertionError("run did not finish")


# --- the shadow file and the premise ------------------------------------------------------------------------------

def test_a_shadow_file_only_writes_its_marker_file(tmp_path):
    path = shadow(tmp_path, "timeit", tmp_path / "marker.txt")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom, ast.While, ast.For, ast.FunctionDef))]
    called = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert called == {"open", "write"}


def test_the_shadow_file_takes_the_place_of_the_module_without_the_safe_path_and_does_not_with_it(tmp_path):
    """The premise of the tests below: in the work folder a plain `python -c` imports the shadow file, and the safe path stops it."""
    work, marker = work_folder_with(tmp_path, "timeit")
    plain = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    off = subprocess.run([sys.executable, "-c", "import timeit"], cwd=str(work), env=plain, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=60)
    assert marker.exists() and off.returncode == 0                       # the shadow file was the module `timeit`
    marker.unlink()
    safe = subprocess.run([sys.executable, "-c", "import timeit"], cwd=str(work), env={**plain, "PYTHONSAFEPATH": "1"},
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert not marker.exists() and safe.returncode == 0                  # the module of the standard library


# --- the environment of a child -----------------------------------------------------------------------------------

def test_the_child_environment_has_the_safe_path_and_keeps_the_other_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/some/code")
    monkeypatch.setenv("PYTHONSAFEPATH", "")                             # an empty value of the user does not switch it on
    env = server._child_env(TOPOPT_VERIFY_STYLE="mcp")
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["TOPOPT_WORKSPACE"] == str(tmp_path) and env["TOPOPT_VERIFY_STYLE"] == "mcp"
    assert env["PYTHONPATH"] == "/some/code" and env["PATH"] == os.environ["PATH"]


def test_every_subprocess_call_of_the_server_passes_the_child_environment():
    """A new `subprocess` call in the server that forgets `env=_child_env(...)` starts a Python child with the work folder on its path."""
    source = SERVER_SOURCE.read_text(encoding="utf-8")
    assert "from subprocess import" not in source and "os.system(" not in source and "os.popen(" not in source
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess"
             and n.func.attr in ("Popen", "run", "call", "check_call", "check_output")]
    assert len(calls) >= 6                                               # the five starts in the work folder and the import probe
    for call in calls:
        env = {k.arg: k.value for k in call.keywords}.get("env")
        assert isinstance(env, ast.Call) and getattr(env.func, "id", "") == "_child_env", f"line {call.lineno} has no env=_child_env(...)"


def test_the_import_probe_of_another_python_gets_the_safe_path(tmp_path):
    """`_python_problem` runs `<python> -c "import topopt"` in the current folder of the server. The fake Python tells its environment."""
    fake = tmp_path / "fakepy"
    fake.write_text('#!/bin/sh\necho "safe path: [$PYTHONSAFEPATH]" >&2\nexit 3\n', encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    message = server._python_problem(str(fake))
    assert message is not None and message.endswith("(safe path: [1])"), message


# --- end to end: a user file in the work folder does not replace a module in a child ------------------------------

@pytest.mark.parametrize("name", ["timeit", "json", "random"])
def test_the_environment_probe_does_not_import_a_user_file_of_the_work_folder(tmp_path, monkeypatch, name):
    """The probe is `<TOPOPT_PYTHON> -m topopt.environment` with the work folder as its current folder (and the gmsh probe below it)."""
    work, marker = work_folder_with(tmp_path, name)
    wrapper = tmp_path / "otherpy"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(work))
    monkeypatch.setenv("TOPOPT_PYTHON", str(wrapper))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    reply = server.environment()
    assert reply["ok"], reply
    assert reply["run_interpreter"]["probe_error"] is None, reply["run_interpreter"]
    assert reply["run_interpreter"]["report_is_for"] == sys.executable          # the data came from the probe, not from this server
    assert not marker.exists()


def test_a_run_through_run_topopt_does_not_import_a_user_timeit_of_the_work_folder(tmp_path, monkeypatch):
    """The command is the real one (`python -m topopt run ...`). Before F4-48 it failed at the start: `cannot import name 'default_timer'`."""
    work, marker = work_folder_with(tmp_path, "timeit")
    yaml_path = make_case(work, h=2.0, iterations=3)
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(work))
    monkeypatch.setenv("PYTHONPATH", str(REPO))                          # the package comes from PYTHONPATH, never from the work folder
    monkeypatch.setenv("EVAL_TOPOPT_NO_WAIT", "1")                       # the run process ends with the run (no dashboard wait)
    monkeypatch.setenv("BROWSER", "true")                                # the dashboard of the child opens no browser
    out = work / "run"
    started = server.run_topopt(str(yaml_path), str(out))
    assert "error" not in started, started
    st = wait_done(lambda: server.run_status(str(out)), server._current["proc"])
    log = out.parent / f"{out.name}.stdout"                               # the output of the run process, for the message of a failure
    assert st["state"] == "done", (st["state"], log.read_text(errors="replace")[-600:] if log.exists() else "")
    server._current["proc"].wait(timeout=60)
    assert not marker.exists()


# --- the launcher -------------------------------------------------------------------------------------------------

def launcher_exec_arguments() -> dict:
    """`{case: arguments after exec "$PY"}` for the `mcp`, `python` and `topopt` cases of scripts/topopt-env.sh."""
    text = LAUNCHER.read_text(encoding="utf-8")
    # the `mcp` case is `if ensure_for_server; then exec "$PY" ...; fi` since P9-2b (main), the other two are `exec "$PY" ... ;;`
    return dict(re.findall(r'^  (mcp|python|topopt)\)\n(?:    .*\n)*?    (?:if ensure_for_server; then )?exec "\$PY" (.*?)(?: ;;|; fi)', text, re.M))


def test_the_launcher_starts_the_server_and_the_command_line_with_the_safe_path_and_the_python_case_as_it_was():
    found = launcher_exec_arguments()
    assert found == {"mcp": '-P -m topopt_mcp.server "$@"', "python": '"$@"', "topopt": '-P -m topopt "$@"'}
    assert 'exec "$PY" -m' not in LAUNCHER.read_text(encoding="utf-8")
