"""The MCP tools `verify_run` and `verify_status`: a detached child with a status file (CONTRACTS.md section 16).

Most tests use a fake child (`tests/serve_verify_fake.py`, env var TOPOPT_MCP_VERIFY_CMD), so they need no gmsh.
The last test runs the real command through the server (needs gmsh and scikit-fem).
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import needs_gmsh, needs_skfem, recorded_hook_run  # noqa: E402
from topopt_mcp import server  # noqa: E402

RECORDING = recorded_hook_run()
HOOK_YAML = REPO / "examples" / "hook" / "hook.yaml"
FAKE = f"{sys.executable} {REPO}/tests/serve_verify_fake.py {{run_dir}}"
TAG = " --job topopt-verify"          # the server tests the command line of a pid for "topopt" (a reused pid is not ours)


def fake(mode, step):
    return f"{FAKE} {mode} --step {step}{TAG}"

pytestmark = pytest.mark.skipif(not (RECORDING / "result" / "final.stl").is_file(),
                                reason="needs the recorded hook run recordings/hook_*")


def copy_run(dst: Path) -> Path:
    (dst / "result").mkdir(parents=True)
    shutil.copy(RECORDING / "run.json", dst / "run.json")
    for name in ("final.stl", "metrics.json"):
        shutil.copy(RECORDING / "result" / name, dst / "result" / name)
    return dst


def fake_extras():
    return {"gmsh": {"state": "ok", "version": "4.15.2", "detail": "", "fix": ""},
            "scikit-fem": {"state": "ok", "version": "x"}, "cholespy": {"state": "ok", "version": None}}


def kill_job():
    proc = server._verify.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._verify.clear()


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    from topopt.verify import run as vrun
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TOPOPT_MCP_VERIFY_CMD", fake("ok", 0.25))
    monkeypatch.setattr(vrun, "extras_facts", fake_extras)          # the preflight needs no gmsh when the child is a fake
    server._verify.clear()
    d = copy_run(tmp_path / "runs" / "hook")
    yield d
    kill_job()


def start(d, **kw):
    return server.verify_run(run_dir=str(d), estimate_only=False, yaml_path=str(HOOK_YAML), **kw)


def wait_for(d, states, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        res = server.verify_status(run_dir=str(d))
        if res["state"] in states:
            return res
        time.sleep(0.1)
    raise AssertionError(f"no state in {states} after {timeout} s; last: {res}")


def test_estimate_only_is_the_default_and_has_the_seconds(run_dir):
    res = server.verify_run(run_dir=str(run_dir), yaml_path=str(HOOK_YAML))
    assert res["ok"] and res["estimate_only"] and not (run_dir / "verify").exists()
    est = res["estimates_before_meshing"][0]
    assert est["seconds_by_backend"]["pardiso"] > 0 and est["estimated_dofs"] > 10_000
    assert server._alive_verify() is None                       # nothing was started


def test_a_real_run_returns_at_once_and_verify_status_shows_the_states_and_the_result(run_dir, monkeypatch):
    import topopt.verify.factor as factor
    from helpers import fake_factor_availability
    # A host where auto takes cholespy, a factor with a time model. With the Accelerate shim, auto takes cholesky, which has none
    # (the next test).
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("cholespy"))
    monkeypatch.setenv("TOPOPT_MCP_VERIFY_CMD", fake("ok", 1.0))      # the call waits 1 s for an early failure
    t0 = time.time()
    res = start(run_dir)
    assert time.time() - t0 < 5.0                                # the call does not wait for the job
    assert res["ok"] and res["started"] and res["pid"] > 0 and res["estimated_seconds_total"] > 0
    assert res["status_file"].endswith("verify/status.json") and Path(res["log_file"]).name == "hook.verify.stdout"
    assert res["estimates_before_meshing"][0]["estimated_dofs"] > 0 and res["yaml_vs_run"]["matches"]
    seen = set()
    end = time.time() + 30
    while time.time() < end:
        st = server.verify_status(run_dir=str(run_dir), result=False)
        seen.add(st["state"])
        if st["state"] == "done":
            break
        time.sleep(0.05)
    assert st["state"] == "done" and st["exit_code"] == 0 and "result" not in st
    assert {"meshing", "solving"} & seen                       # the job was seen while it worked
    assert server.verify_status()["run_dir"] == str(run_dir)                    # no run_dir: the folder of the last verify job
    full = server.verify_status(run_dir=str(run_dir))
    assert full["result"]["ok"] and full["result_matches_final_stl"] is True
    assert full["files_rel"]["mesh"].endswith("verify/mesh.npz") and full["verify_json_rel"].endswith("verify/verify.json")
    assert full["elapsed_s"] > 0 and full["process_alive"] in (True, False)
    assert server._alive_verify() is None or wait_for(run_dir, {"done"})


def test_the_accelerate_factor_has_no_time_estimate_so_the_total_is_none(run_dir, monkeypatch):
    """CONTRACTS.md section 16, time model: no measurement of `cholesky` on a P2 system. The job starts; the total is None, not 0."""
    import topopt.verify.factor as factor
    from helpers import fake_factor_availability
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("cholesky", "cholespy", "superlu"))
    res = start(run_dir)
    assert res["ok"] and res["started"] and res["solver"] == "cholesky" and res["estimated_seconds_total"] is None
    e = res["estimates_before_meshing"][0]
    assert e["estimated_seconds"] is None and e["seconds_by_backend"]["cholesky"] is None and e["seconds_by_backend"]["cholespy"] > 0
    wait_for(run_dir, {"done"})


def test_only_one_job_at_a_time_and_the_optimizer_run_sees_it(run_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_MCP_VERIFY_CMD", fake("ok", 1.0))
    first = start(run_dir)
    assert first["ok"]
    second = start(run_dir)
    assert second["ok"] is False and second["exit_code"] == 2 and second["pid"] == first["pid"]
    assert "verify job is still running" in second["error"] and "verify_status" in second["error"]
    # an optimizer run is refused too (one multi-GB job at a time)
    yaml = tmp_path / "x.yaml"
    yaml.write_text("name: x\n")
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    other = server.run_topopt(str(yaml))
    assert "verify job is still running" in other["error"] and other["pid"] == first["pid"]
    # the lock file of the work folder is what a restarted server sees
    lock = json.loads((tmp_path / server.VERIFY_LOCK).read_text())
    assert lock["pid"] == first["pid"] and lock["run_dir"] == str(run_dir)
    server._verify.clear()                                      # "restart": no Popen handle
    assert server._alive_verify() == (run_dir, first["pid"])
    wait_for(run_dir, {"done"}, 30)
    server._verify["proc"] = None
    assert server._alive_verify() is None and not (tmp_path / server.VERIFY_LOCK).exists()
    third = start(run_dir)
    assert third["ok"] and third["pid"] != first["pid"]
    wait_for(run_dir, {"done"})


def test_an_optimizer_run_in_progress_blocks_a_verify_job(run_dir, tmp_path, monkeypatch):
    """The busy rule of `run_topopt` (status.json in starting, running or postprocess) is the one that is used."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    other = tmp_path / "runs" / "live"
    other.mkdir(parents=True)
    (other / "status.json").write_text(json.dumps({"state": "running"}))
    server._current.update(proc=proc, run_dir=other, started=time.time() - 100)
    try:
        res = start(run_dir)
        assert res["ok"] is False and "optimizer run is still running" in res["error"] and res["pid"] == proc.pid
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        server._current.clear()


def test_a_job_that_fails_shows_its_error(run_dir, monkeypatch):
    monkeypatch.setenv("TOPOPT_MCP_VERIFY_CMD", fake("error", 0.6))
    assert start(run_dir)["ok"]
    st = wait_for(run_dir, {"error"})
    assert st["error"] == "fake failure" and st["exit_code"] == 2 and "result" not in st
    assert st["summary"].startswith("error: fake failure")


def test_a_job_that_ends_at_once_with_an_error_is_reported_by_the_call_itself(run_dir, monkeypatch):
    monkeypatch.setenv("TOPOPT_MCP_VERIFY_CMD", fake("error", 0.0))
    res = start(run_dir)
    assert res["ok"] is False and res["exit_code"] == 2 and "ended at once with code 2" in res["error"]
    assert Path(res["log_file"]).name == "hook.verify.stdout"


def test_a_job_that_is_killed_shows_failed_with_the_log_tail(run_dir, monkeypatch):
    monkeypatch.setenv("TOPOPT_MCP_VERIFY_CMD", fake("die", 0.6))
    first = start(run_dir)
    assert first["ok"]
    st = wait_for(run_dir, {"failed"})
    assert "stopped with no final status" in st["summary"] and st["process_alive"] is False and "log_tail" in st
    assert st["status"]["state"] == "meshing"                   # the last thing the job wrote
    assert start(run_dir)["ok"]                                 # the dead job does not block a new one
    kill_job()


def test_verify_status_of_an_empty_folder_and_of_a_result_with_no_status_file(run_dir, tmp_path):
    assert server.verify_status(run_dir=str(run_dir))["state"] == "none"
    (run_dir / "verify").mkdir()
    (run_dir / "verify" / "verify.json").write_text(json.dumps({"ok": True, "exit_code": 3, "files": {"verify": "verify.json"}}))
    st = server.verify_status(run_dir=str(run_dir))
    assert st["state"] == "done" and st["exit_code"] == 3 and "earlier run" in st["summary"]


def test_verify_status_carries_partial_and_coverage_before_the_large_result(run_dir):
    """TC-2 (CONTRACTS.md section 44): the record is a top-level key of the reply, also with result=false, and `summary` says it."""
    (run_dir / "verify").mkdir()
    cov = {"ran": ["stress"], "failed": [{"name": "buckling", "reason": "RuntimeError: phase 33 error -4"}], "not_run": []}
    path = run_dir / "verify" / "verify.json"
    path.write_text(json.dumps({"ok": True, "exit_code": 0, "partial": True, "coverage": cov, "files": {"verify": "verify.json"}}))
    for want_result in (True, False):
        st = server.verify_status(run_dir=str(run_dir), result=want_result)
        assert st["state"] == "done" and st["partial"] is True and st["coverage"] == cov
        assert st["summary"] == ("a result from an earlier run (no status file); "
                                 "partial: 1 of 2 checks failed (buckling: RuntimeError: phase 33 error -4); ran: stress")
        assert ("result" in st) is want_result
        assert list(st).index("coverage") < list(st).index("verify_json")
    path.write_text(json.dumps({"ok": True, "exit_code": 0, "partial": False, "coverage": {"ran": ["stress"], "failed": [], "not_run": []},
                                "files": {"verify": "verify.json"}}))
    st = server.verify_status(run_dir=str(run_dir))
    assert st["partial"] is False and "partial" not in st["summary"]                      # a clean result keeps its summary
    path.write_text(json.dumps({"ok": True, "exit_code": 3, "files": {"verify": "verify.json"}}))
    st = server.verify_status(run_dir=str(run_dir))
    assert "partial" not in st and "coverage" not in st                                    # a result of an older version has neither key


def test_a_result_for_another_final_stl_is_marked(run_dir):
    assert start(run_dir)["ok"]
    wait_for(run_dir, {"done"})
    assert server.verify_status(run_dir=str(run_dir))["result_matches_final_stl"] is True
    stl = run_dir / "result" / "final.stl"
    stl.write_bytes(stl.read_bytes() + b" ")                    # another file
    st = server.verify_status(run_dir=str(run_dir))
    assert st["result_matches_final_stl"] is False


def test_a_result_for_a_run_with_stock_matches_nominal_stl_and_not_final_stl(run_dir):
    nominal = run_dir / "result" / "nominal.stl"
    shutil.copy(run_dir / "result" / "final.stl", nominal)      # a run with machining stock has both files
    assert start(run_dir)["ok"]
    wait_for(run_dir, {"done"})
    st = server.verify_status(run_dir=str(run_dir))
    assert st["result"]["run"]["verified_file"] == "nominal.stl" and st["result_matches_final_stl"] is True
    final = run_dir / "result" / "final.stl"
    final.write_bytes(final.read_bytes() + b" ")                # the as-printed mesh changes: the result is still for nominal.stl
    assert server.verify_status(run_dir=str(run_dir))["result_matches_final_stl"] is True
    nominal.write_bytes(nominal.read_bytes() + b" ")
    assert server.verify_status(run_dir=str(run_dir))["result_matches_final_stl"] is False


def test_stale_and_foreign_locks_are_removed(run_dir, tmp_path):
    lock = tmp_path / server.VERIFY_LOCK
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock.write_text(json.dumps({"pid": dead.pid, "run_dir": str(run_dir)}))
    assert server._alive_verify() is None and not lock.exists()
    sleeper = subprocess.Popen(["sleep", "30"])                 # alive, but not a topopt process (a reused pid)
    time.sleep(0.2)
    try:
        lock.write_text(json.dumps({"pid": sleeper.pid, "run_dir": str(run_dir)}))
        assert server._cmdline_matches(sleeper.pid) in (False, True)
        if Path("/proc").is_dir():
            assert server._cmdline_matches(sleeper.pid) is False
            assert server._alive_verify() is None and not lock.exists()
    finally:
        sleeper.kill()
        sleeper.wait()
    mine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "topopt-verify"])
    try:
        end = time.time() + 5
        while Path("/proc").is_dir() and time.time() < end and \
                b"topopt-verify" not in Path(f"/proc/{mine.pid}/cmdline").read_bytes():
            time.sleep(0.02)                                    # the child has not run exec yet
        assert server._cmdline_matches(mine.pid) is True
        lock.write_text(json.dumps({"pid": mine.pid, "run_dir": str(run_dir)}))
        assert server._alive_verify() == (run_dir, mine.pid)
        (run_dir / "verify").mkdir()
        (run_dir / "verify" / "status.json").write_text(json.dumps({"state": "done"}))      # it said done: not busy
        assert server._alive_verify() is None
    finally:
        mine.kill()
        mine.wait()


def test_errors_of_the_preflight_come_back_at_once_with_mcp_names(run_dir):
    refused = start(run_dir, max_gb=0.01)
    assert refused["ok"] is False and refused["exit_code"] == 2 and refused["facts"]["when"] == "before meshing"
    assert "max_gb" in refused["error"] and "force" in refused["error"] and "--" not in refused["error"]
    assert server._alive_verify() is None and not (run_dir / "verify").exists()          # nothing started
    bad = start(run_dir, mesh_size_mm=-1.0)
    assert bad["ok"] is False and bad["exit_code"] == 1 and "mesh_size_mm" in bad["error"]
    assert start(run_dir, solver="amg")["exit_code"] == 1
    gone = server.verify_run(run_dir=str(run_dir / "nope"), estimate_only=False)
    assert gone["ok"] is False and gone["exit_code"] == 1


def test_options_go_to_the_child_as_command_line_flags():
    args = server._verify_args(1.5, None, 0.6, 5.0, 99.0, "pardiso", 8.0, True, "/x/hook.yaml", 3.0, False)
    assert args == ["--mesh-size", "1.5", "--region-band-mm", "0.6", "--allowable-mpa", "5.0", "--solver", "pardiso",
                    "--max-gb", "8.0", "--yaml", "/x/hook.yaml", "--near-mm", "3.0", "--percentile", "99.0", "--force",
                    "--no-sensitivity"]
    assert server._verify_args(None, [2.0, 1.5], None, None, 99.9, None, None, False, None, 2.0, True) == [
        "--near-mm", "2.0", "--levels", "2,1.5"]
    cmd = server._verify_command(Path("/r"), ["--force"])
    assert cmd[-3:] == ["verify", "/r", "--force"] or os.environ.get("TOPOPT_MCP_VERIFY_CMD")


def test_the_instructions_and_the_tool_list_name_both_tools():
    import asyncio
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"verify_run", "verify_status"} <= names
    assert "verify_run" in server.mcp.instructions      # #380: verify_status is named in the description of verify_run (checked below)
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert len(server.mcp.instructions) <= OLD_CLIENT_CUT
    doc = next(t.description for t in asyncio.run(server.mcp.list_tools()) if t.name == "verify_run")
    for word in ("estimate_only", "verify_status", "near_regions", "load_band_sensitivity", "background"):
        assert word in doc
    assert "--max-gb" not in doc and "--force" not in doc and "--solver" not in doc      # MCP parameters, not CLI flags


@needs_gmsh
@needs_skfem
def test_the_real_command_runs_as_a_detached_child_and_the_server_never_loads_gmsh(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("TOPOPT_MCP_VERIFY_CMD", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    server._verify.clear()
    d = copy_run(tmp_path / "runs" / "hook")
    gmsh_before = "gmsh" in sys.modules                              # another test in the same process can load it
    try:
        res = start(d, load_sensitivity=False)
        assert res["ok"] and res["started"], res
        st = wait_for(d, {"done", "error", "failed"}, timeout=180)
        assert st["state"] == "done", st
        out = st["result"]
        assert out["ok"] and set(out["cases"]) == {"hang", "pull_off"} and out["exit_code"] == 0
        assert out["memory"]["peak_rss_bytes"] > 1e8                # the peak of this job (a process of its own)
        assert st["result_matches_final_stl"] is True
        assert (d / "verify" / "mesh.npz").is_file()
        assert gmsh_before or "gmsh" not in sys.modules             # the server did not load libgmsh
    finally:
        kill_job()


def test_no_direct_factor_is_exit_code_1_with_the_facts_for_estimate_only_and_a_real_run(run_dir, monkeypatch):
    """A real run is refused before the job starts (exit_code 1, a UserError). An estimate states the fact and goes on."""
    import topopt.verify.factor as factor
    from helpers import fake_factor_availability
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability())
    res = start(run_dir)
    assert res["ok"] is False and res["exit_code"] == 1 and "started" not in res
    assert "no direct linear solver can run here" in res["error"] and "Options" in res["error"]
    assert "cholespy" in res["error"] and "superlu" in res["error"]
    assert not (run_dir / "verify").exists() and server._alive_verify() is None             # no job, no folder
    assert res["facts"]["requested"] == "auto" and set(res["facts"]["factors"]) == set(factor.FACTOR_NAMES)   # the facts, not only text
    assert not res["facts"]["factors"]["cholespy"]["available"] and "note" in res["facts"]["factors"]["cholespy"]
    named = start(run_dir, solver="pardiso")
    assert named["ok"] is False and named["exit_code"] == 1 and "solver 'pardiso' cannot run here" in named["error"]
    assert named["facts"]["requested"] == "pardiso"
    est = server.verify_run(run_dir=str(run_dir), yaml_path=str(HOOK_YAML))                  # estimate_only is the default
    assert est["ok"] and est["backend"]["solver"] is None and "no direct linear solver" in est["backend"]["problem"]


def test_auto_runs_on_superlu_and_the_estimate_names_it(run_dir, monkeypatch):
    import topopt.verify.factor as factor
    from helpers import fake_factor_availability
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))
    est = server.verify_run(run_dir=str(run_dir), yaml_path=str(HOOK_YAML))
    assert est["ok"] and est["backend"]["solver"] == "superlu" and list(est["backend"]["skipped"]) == ["cholesky", "pardiso", "cholespy"]
    e = est["estimates_before_meshing"][0]
    assert e["seconds_by_backend"]["superlu"] == e["estimated_seconds"] > 0 and "SuperLU model" in e["estimate_model"]
    started = start(run_dir)
    assert started["ok"] and started["started"] and started["estimated_seconds_total"] > 0
