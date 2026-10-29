"""#448: the run process ends natively in the post step (libgomp ends it with exit 1 after a failed thread start, or a SIGSEGV in native code).

Python cannot catch such an end, so `status.json` stays at `postprocess`. run_status and run_result must still say that the process has ended
with no result: the failure facts (exit code or signal, the last log lines) of P9-6 (CONTRACTS.md section 24.4), and never the result of an
older run in the folder. The run is a real `topopt run` of a small case; a stand-in for `post._load_paths` ends the process in the post step.
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from test_optimize import make_case  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt_mcp import server  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="os._exit and SIGSEGV in a child: POSIX only")

LIBGOMP = "libgomp: Thread creation failed: Resource temporarily unavailable"
CHILD = f"""import os, resource, signal, sys
sys.path.insert(0, {str(REPO)!r})
from topopt import post
from topopt.cli import main

yaml_path, out, how = sys.argv[1:4]


def end(*a, **k):    # the post step ends the process the way native code does: no Python exception, no handler runs
    if how == "sigsegv":
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.kill(os.getpid(), signal.SIGSEGV)
    sys.stderr.write({LIBGOMP!r} + "\\n")
    sys.stderr.flush()
    os._exit(1)


post._load_paths = end
sys.exit(main(["run", yaml_path, "--out", out, "--quiet"]))
"""


@pytest.fixture(autouse=True)
def clean_server():
    server._current.clear()
    server._stopped.clear()
    yield
    server._current.clear()
    server._stopped.clear()


@pytest.fixture
def case(tmp_path):
    child = tmp_path / "ends_in_post.py"
    child.write_text(CHILD)
    return make_case(tmp_path, h=2.0, iterations=3), child


def run_by_server(monkeypatch, yaml_path, child, out, how):
    """`run_topopt` starts the run (the test hook TOPOPT_MCP_CMD) and the test waits for its end. Return the exit code."""
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{shlex.quote(sys.executable)} {shlex.quote(str(child))} {{yaml}} {{out}} {how}")
    reply = server.run_topopt(str(yaml_path), str(out))
    assert "error" not in reply, reply
    return server._current["proc"].wait(timeout=300)


def assert_ended_in_post(st, out):
    """The facts that every reply of such a run has: the status file still says postprocess, the process is gone, the job record is stale."""
    assert json.loads((out / "status.json").read_text())["state"] == "postprocess"
    assert st["state"] == "postprocess" and st["process_alive"] is False and st["job"]["state"] == "stale_lock"
    assert "(warning: the process has stopped)" in st["summary"]
    assert st["failure"]["run_data"] == "up to pass 3" and st["options"]
    assert st["failure"]["peak_memory"].startswith("not recorded: the process ") and st["failure"]["peak_memory_bytes"] is None


def assert_no_result(out):
    res = server.run_result(str(out), preview=False)
    assert res["done"] is False and "final_stl" not in res and "metrics" not in res
    assert res["message"].startswith("The run process has stopped while the status of the run was `postprocess`: there is no result.")
    assert res["failure"] and res["options"]
    return res


@pytest.mark.parametrize("how,code,sig", [("exit", 1, None), ("sigsegv", -11, "SIGSEGV")], ids=["libgomp-exit-1", "SIGSEGV"])
def test_a_native_end_in_the_post_step_of_a_server_run_is_a_failure_with_its_cause(monkeypatch, case, tmp_path, how, code, sig):
    yaml_path, child = case
    out = tmp_path / "run"
    assert run_by_server(monkeypatch, yaml_path, child, out, how) == code
    st = server.run_status(str(out))
    assert_ended_in_post(st, out)
    assert (st["failure"]["exit_code"], st["failure"]["signal"]) == (code, sig)
    ended = "did not reach its end" if sig else "wrote no job record of its end"     # a signal, or an exit of the process itself
    assert st["failure"]["peak_memory"] == f"not recorded: the process {ended}"
    if how == "exit":    # the output file of this server has the line that the process printed last: it is the newest log
        assert st["log_tail_file"] == str(out) + ".stdout" and LIBGOMP in st["log_tail"]
        assert "possible_causes" not in st                    # an exit with a message: the message is the fact
    else:                # a SIGSEGV prints nothing: run.log is the newest log
        assert st["log_tail_file"] == str(out / "run.log") and st["log_tail"]
        assert any("crash in native code" in c for c in st["possible_causes"])
    assert_no_result(out)


def test_a_native_end_in_the_post_step_over_an_older_done_result_does_not_return_it(monkeypatch, case, tmp_path):
    yaml_path, child = case
    out = tmp_path / "run"
    assert main(["run", str(yaml_path), "--out", str(out), "--quiet"]) == 0
    assert server.run_result(str(out), preview=False)["done"] is True
    assert run_by_server(monkeypatch, yaml_path, child, out, "exit") == 1
    st = server.run_status(str(out))
    assert_ended_in_post(st, out)
    assert not (out / "result" / "metrics.json").exists()      # the new run removed the older result before its passes
    assert_no_result(out)


def test_a_native_end_in_the_post_step_of_a_cli_run_is_a_failure_too(case, tmp_path):
    """A run of `topopt run` in a terminal: this server did not start it, so it knows no exit code. The job record of the dead process says
    that the process has ended, and the summary says so."""
    yaml_path, child = case
    out = tmp_path / "run"
    done = subprocess.run([sys.executable, str(child), str(yaml_path), str(out), "exit"], capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 1 and LIBGOMP in done.stderr
    st = server.run_status(str(out))
    assert_ended_in_post(st, out)
    assert st["failure"]["exit_code"] is None and "not known" in st["failure"]["exit_note"]
    assert st["failure"]["peak_memory"] == "not recorded: the process did not reach its end"
    assert st["log_tail_file"] == str(out / "run.log")             # the terminal had the stderr line; run.log has the lines of the run
    assert_no_result(out)
