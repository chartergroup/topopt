"""The MCP tools `ratchet_run`, `ratchet_status` and `ratchet_approve`: a detached child with a lock file (CONTRACTS.md section 36).

The rules in these tests need `printcheck` only, so the real child runs in about a second and needs no gmsh. The guard tests use a
child that only sleeps (env var TOPOPT_MCP_RATCHET_CMD).
"""
import asyncio
import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("mcp")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from ratchet_helpers import stl_folder  # noqa: E402
from test_feedback import fake_run, make_case, tree_hash  # noqa: E402

from topopt_mcp import server  # noqa: E402

TOOLS = ("ratchet_run", "ratchet_status", "ratchet_approve")
PRINT_RULES = {"rules": [{"rule": "watertight"}, {"rule": "bodies"}, {"rule": "mass", "must_fall": True}]}
SLEEPER = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)' topopt-ratchet-fake"


def kill_job():
    proc = server._ratchet.get("proc")
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._ratchet.clear()


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.delenv("TOPOPT_MCP_RATCHET_CMD", raising=False)
    server._ratchet.clear()
    ypath = make_case(tmp_path / "in")
    doc = yaml.safe_load(ypath.read_text())
    doc["acceptance"] = PRINT_RULES
    ypath.write_text(yaml.safe_dump(doc, sort_keys=False))
    run = fake_run(tmp_path / "runs" / "a", ypath)
    stl_folder(tmp_path / "cands", 3)
    yield SimpleNamespace(tmp=tmp_path, run=run, yaml=ypath, cands=tmp_path / "cands", root=tmp_path / "runs" / "a-ratchet")
    kill_job()


def start(c, **kw):
    kw.setdefault("source", f"stl:{c.cands}")
    return server.ratchet_run(run_dir=str(c.run), estimate_only=False, **kw)


def wait_for(c, states, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        res = server.ratchet_status(root=str(c.root))
        if res.get("state") in states:
            return res
        time.sleep(0.1)
    raise AssertionError(f"no state in {states} after {timeout} s; last: {res}")


def test_the_three_tools_are_registered_and_the_instructions_name_them():
    names = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert set(TOOLS) <= set(names)
    # #380: the instructions (2048 characters) name ratchet_run in their last line; the status and the approve tool are named in the
    # description of ratchet_run (MOVED_RULES of tests/test_plugin_manifest.py)
    assert "ratchet_run" in server.mcp.instructions.splitlines()[-1]
    doc = " ".join(names["ratchet_run"].description.split())
    assert "ratchet_status" in doc and "ratchet_approve" in doc
    for word in ("estimate_only", "ratchet_status", "stop_before_promote", "marks:FILE", "stl:FOLDER", "detached process"):
        assert word in doc, word
    assert "owner" in names["ratchet_approve"].description
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert len(server.mcp.instructions) <= OLD_CLIENT_CUT
    assert "must" not in server.mcp.instructions.lower() and "never" not in server.mcp.instructions.lower()


def test_estimate_only_is_the_default_it_starts_nothing_and_it_has_the_rules_and_the_count(case):
    res = server.ratchet_run(run_dir=str(case.run), source=f"stl:{case.cands}")
    assert res["ok"] and res["estimate_only"] is True and res["root"] == str(case.root)
    chk = res["check"]
    assert [r["rule"] for r in chk["rules"]] == ["watertight", "bodies", "mass"] and chk["ready"] is True
    assert chk["estimate"]["candidates"] == 3 and chk["estimate"]["verify_calls"] == 0
    assert not case.root.exists() and server._alive_ratchet() is None and not (case.tmp / server.RATCHET_LOCK).exists()
    via_marks = server.ratchet_run(run_dir=str(case.run), marks=str(case.tmp / "nowhere.json"))
    assert via_marks["ok"] is False and via_marks["exit_code"] == 1 and "nowhere.json" in via_marks["error"]


def test_a_missing_or_a_wrong_source_and_a_yaml_without_rules_are_errors_with_the_way_out(case):
    res = server.ratchet_run(run_dir=str(case.run))
    assert res["ok"] is False and res["exit_code"] == 1 and "source" in res["error"] and "stl:FOLDER" in res["error"]
    res = server.ratchet_run(run_dir=str(case.run), source="folder")
    assert res["ok"] is False and res["exit_code"] == 1 and "not understood" in res["error"]
    plain = fake_run(case.tmp / "runs" / "plain", make_case(case.tmp / "plain_in"))
    res = server.ratchet_run(run_dir=str(plain), source=f"stl:{case.cands}")
    assert res["ok"] is False and res["exit_code"] == 1 and "acceptance" in res["error"]
    assert server.ratchet_status(root=str(case.tmp / "nowhere"))["ok"] is False
    assert server.ratchet_status()["error"].startswith("no ratchet root")
    assert server.ratchet_approve()["error"].startswith("no ratchet root")


def test_a_real_run_returns_at_once_and_ratchet_status_shows_the_log(case):
    before = tree_hash(case.run)
    t0 = time.time()
    res = start(case)
    assert time.time() - t0 < 10.0
    assert res["ok"] and res["started"] and res["pid"] > 0 and res["root"] == str(case.root)
    assert Path(res["log_file"]).name == "a-ratchet.stdout" and Path(res["log_file"]).parent == case.root.parent
    st = wait_for(case, {"finished", "failed", "baseline_failed", "waiting_for_owner"})
    assert st["state"] == "finished", st
    assert st["counts"] == {"candidates": 3, "ACCEPT": 1, "REJECT": 2} and st["parent"] == "acc_01"
    assert [r["decision"] for r in st["last"]][-3:] == ["ACCEPT", "REJECT", "REJECT"]
    assert st["last"][-1]["failed"] == ["mass"] and "mass" in st["last"][-1]["why"][0]
    assert st["job"]["state"] == "done" and st["pending"] == [] and st["ok"] is True
    assert (case.root / "ratchet.jsonl").is_file() and (case.root / "acc_01" / "result" / "final.stl").is_file()
    assert tree_hash(case.run) == before
    assert server.ratchet_status()["root"] == str(case.root)               # no root: the last ratchet of this server
    assert len(server.ratchet_status(root=str(case.root), last=1)["last"]) == 1
    # the child ended: the guard is free, and a second call goes on with the same root (nothing is left to judge)
    deadline = time.time() + 20
    while server._alive_ratchet() and time.time() < deadline:
        time.sleep(0.1)
    assert server._alive_ratchet() is None
    again = start(case)
    assert again["ok"], again
    assert wait_for(case, {"finished"})["counts"]["candidates"] == 3


def test_the_owner_gate_through_the_tools(case):
    res = start(case, stop_before_promote=True)
    assert res["ok"]
    st = wait_for(case, {"waiting_for_owner", "finished", "failed"})
    assert st["state"] == "waiting_for_owner" and st["pending"] == ["c001"] and st["counts"]["ACCEPT"] == 1
    assert st["parent"] == "acc_00" and not (case.root / "acc_01").exists()
    bad = server.ratchet_approve(root=str(case.root), id="c009")
    assert bad["ok"] is False and bad["exit_code"] == 1 and "does not wait" in bad["error"]
    ok = server.ratchet_approve(root=str(case.root))
    assert ok["ok"] and ok["owner_approval"] == "given" and ok["promoted"] == "acc_01" and ok["id"] == "c001"
    assert (case.root / "acc_01" / "result" / "final.stl").is_file()
    assert server.ratchet_status(root=str(case.root))["pending"] == []
    nothing = server.ratchet_approve(root=str(case.root))
    assert nothing["ok"] is False and "no candidate waits" in nothing["error"]


def test_decline_drops_the_candidate(case):
    assert start(case, stop_before_promote=True)["ok"]
    wait_for(case, {"waiting_for_owner"})
    res = server.ratchet_approve(root=str(case.root), decline=True)
    assert res["ok"] and res["owner_approval"] == "declined" and res["promoted"] is None and res["parent"] == "acc_00"
    assert not (case.root / "acc_01").exists()


def test_the_options_reach_the_child(case, monkeypatch):
    seen = []
    real = server._ratchet_command

    def spy(run_dir, root, source, args):
        seen.append((str(run_dir), str(root), source, list(args)))
        return shlex.split(SLEEPER)
    monkeypatch.setattr(server, "_ratchet_command", spy)
    res = start(case, yaml_path=str(case.yaml), max_candidates=2, mesh_size_mm=1.5, stop_before_promote=True, overwrite=True,
                override_lock=True, root=str(case.tmp / "elsewhere"))
    assert res["ok"] and res["root"] == str(case.tmp / "elsewhere")
    run_dir, root, source, args = seen[0]
    assert (run_dir, root, source) == (str(case.run), str(case.tmp / "elsewhere"), f"stl:{case.cands}")
    assert args == ["--yaml", str(case.yaml), "--max-candidates", "2", "--mesh-size", "1.5", "--stop-before-promote",
                    "--overwrite", "--override-lock"]
    monkeypatch.setattr(server, "_ratchet_command", real)
    res = server._ratchet_command(case.run, case.root, "marks:/x/m.json", ["--overwrite"])
    assert res[1:4] == ["-m", "topopt.ratchet", "run"] and res[-1] == "--overwrite" and "--source" in res
    assert server._ratchet_source(None, "m.json") == f"marks:{case.tmp / 'm.json'}"           # a relative path starts at the work folder
    assert server._ratchet_source("stl:cands", None) == f"stl:{case.tmp / 'cands'}"


def test_only_one_job_at_a_time_and_every_other_start_sees_the_ratchet(case, monkeypatch):
    from topopt.verify import run as vrun
    monkeypatch.setenv("TOPOPT_MCP_RATCHET_CMD", SLEEPER)
    first = start(case)
    assert first["ok"] and first["started"]
    assert server._alive_ratchet() == (case.root, first["pid"])
    second = start(case)
    assert second["ok"] is False and second["exit_code"] == 2 and second["pid"] == first["pid"]
    assert "ratchet job is still running" in second["error"] and "ratchet_status" in second["error"]
    # an optimizer run and a verify job are refused too (one multi-GB job at a time)
    yaml_path = case.tmp / "x.yaml"
    yaml_path.write_text("name: x\n")
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    other = server.run_topopt(str(yaml_path))
    assert "ratchet job is still running" in other["error"] and other["pid"] == first["pid"]
    monkeypatch.setattr(vrun, "preflight", lambda *a, **k: SimpleNamespace())
    ver = server.verify_run(run_dir=str(case.run), estimate_only=False)
    assert ver["ok"] is False and "ratchet job is still running" in ver["error"] and ver["pid"] == first["pid"]
    # the estimate is not a job: it is allowed while a ratchet runs
    assert server.ratchet_run(run_dir=str(case.run), source=f"stl:{case.cands}")["estimate_only"] is True
    # ratchet_status sees the live process before the root has any file
    st = server.ratchet_status(root=str(case.root))
    assert st["state"] == "starting" and st["process_alive"] is True and st["pid"] == first["pid"]
    # approve is refused while the job runs
    busy = server.ratchet_approve(root=str(case.root))
    assert busy["ok"] is False and "still running" in busy["error"]
    # the lock file of the work folder is what a restarted server sees
    lock = json.loads((case.tmp / server.RATCHET_LOCK).read_text())
    assert lock["pid"] == first["pid"] and lock["root"] == str(case.root) and lock["run_dir"] == str(case.run)
    child = server._ratchet["proc"]
    server._ratchet.clear()                                       # "restart": no Popen handle
    assert server._alive_ratchet() == (case.root, first["pid"])
    kill_job_by_pid(first["pid"])
    # A child that has ended stays a zombie until its parent waits for it. /proc shows the zombie (Linux), but macOS has no /proc
    # and the pid check still sees the process. A job of an earlier server has no parent here (init waits for it), so the test
    # waits for the child, as the parent of a real earlier job would.
    child.wait(timeout=10)
    deadline = time.time() + 10
    while server._alive_ratchet() and time.time() < deadline:
        time.sleep(0.1)
    assert server._alive_ratchet() is None and not (case.tmp / server.RATCHET_LOCK).exists()   # a dead process frees the lock


def kill_job_by_pid(pid):
    os.killpg(pid, signal.SIGKILL)


@pytest.mark.parametrize("final", ["finished", "waiting_for_owner", "baseline_failed"])
def test_a_final_state_shows_only_after_the_process_has_ended(case, monkeypatch, final):
    """The state file is written before the process exits. While the process lives the status says `running` (the next
    call is refused as busy), and it keeps the file state as a fact. When the process is gone the file state shows."""
    monkeypatch.setenv("TOPOPT_MCP_RATCHET_CMD", SLEEPER)
    first = start(case)
    assert first["ok"]
    case.root.mkdir(parents=True)
    (case.root / "ratchet.json").write_text(json.dumps({"format": 1, "state": final}))
    st = server.ratchet_status(root=str(case.root))
    assert st["state"] == "running" and st["file_state"] == final and st["process_alive"] is True
    assert final in st["summary"] and "still alive" in st["summary"]
    kill_job_by_pid(first["pid"])
    deadline = time.time() + 10
    while server._alive_ratchet() and time.time() < deadline:
        time.sleep(0.1)
    st = server.ratchet_status(root=str(case.root))
    assert st["state"] == final and st["process_alive"] is False and "file_state" not in st


def test_a_process_that_stopped_with_no_final_state_is_failed_in_the_status(case):
    case.root.mkdir(parents=True)
    (case.root / "ratchet.json").write_text(json.dumps({"format": 1, "state": "running"}))
    (case.root.parent / "a-ratchet.stdout").write_text("line 1\nTraceback: the child died\n")
    st = server.ratchet_status(root=str(case.root))
    assert st["ok"] and st["state"] == "failed" and st["process_alive"] is False
    assert "the child died" in st["log_tail"] and "goes on" in st["summary"]


def test_a_child_that_ends_at_once_with_an_error_shows_its_output(case, monkeypatch):
    monkeypatch.setenv("TOPOPT_MCP_RATCHET_CMD", f"{shlex.quote(sys.executable)} -c 'import sys; print(\"bad input here\"); sys.exit(2)' topopt")
    res = start(case)
    assert res["ok"] is False and res["exit_code"] == 2 and "bad input here" in res["error"]


# --- fix round 1 -------------------------------------------------------------------------------------------------------

def test_m3_a_runner_of_the_cli_kind_that_holds_the_job_lock_shows_as_running_in_ratchet_status(case):
    """The reviewer's probe (e5): a CLI process holds the job lock of the root; this server did not start it."""
    from topopt.runjob import RunJob
    case.root.mkdir(parents=True)
    (case.root / "ratchet.json").write_text(json.dumps({"format": 1, "state": "running", "parent": "acc_00"}))
    job = RunJob(case.root)
    job.start()
    try:
        st = server.ratchet_status(root=str(case.root))
        assert st["state"] == "running" and st["process_alive"] is True and st["pid"] == os.getpid()
        assert st["job"]["state"] == "running"
    finally:
        job.finish(0)
    assert server.ratchet_status(root=str(case.root))["state"] == "failed"              # the file said running, the process is gone


def test_n5_a_root_in_a_folder_that_does_not_exist_yet_starts(case):
    root = case.tmp / "new" / "deeper" / "a-ratchet"
    res = start(case, root=str(root))
    assert res["ok"] and res["started"], res
    end = time.time() + 60
    st = {}
    while time.time() < end:
        st = server.ratchet_status(root=str(root))
        if st.get("state") in ("finished", "failed", "baseline_failed"):
            break
        time.sleep(0.1)
    assert st["state"] == "finished", st
    assert (root.parent / "a-ratchet.stdout").is_file()


def test_b1_a_changed_yaml_is_refused_at_the_start_and_rebaseline_goes_on(case):
    first = start(case)
    assert first["ok"]
    assert wait_for(case, {"finished"})["counts"]["candidates"] == 3
    deadline = time.time() + 20
    while server._alive_ratchet() and time.time() < deadline:
        time.sleep(0.1)
    doc = yaml.safe_load(case.yaml.read_text())
    doc["load_cases"][0]["forces"][0]["force"][2] = -1.0                                 # -5 N -> -1 N
    case.yaml.write_text(yaml.safe_dump(doc, sort_keys=False))
    refused = start(case)                                                                # a new child, which stops at the check
    deadline = time.time() + 30
    st = {}
    while time.time() < deadline:
        st = server.ratchet_status(root=str(case.root))
        if st["state"] == "failed" or not refused["ok"]:
            break
        time.sleep(0.1)
    text = refused.get("error", "") + st.get("log_tail", "")                            # the child may end before or after the 1 s wait
    assert "force[2]" in text and "rebaseline" in text, (refused, st)
    assert refused["ok"] is False or st["state"] == "failed"
    st = server.ratchet_status(root=str(case.root))                                       # the reason stays for a later reader (L-m1)
    assert st["state"] == "failed" and st["file_state"] == "finished" and st["job"]["state"] == "failed"
    assert st["last_error"]["exit_code"] == 1 and "force[2]" in st["last_error"]["message"]
    assert "force[2]" in st["summary"] and "killed" not in st["summary"]
    deadline = time.time() + 20
    while server._alive_ratchet() and time.time() < deadline:
        time.sleep(0.1)
    go = start(case, rebaseline=True)
    assert go["ok"], go
    end = time.time() + 60
    while time.time() < end:
        st = server.ratchet_status(root=str(case.root))
        if st["state"] == "finished" and st["process_alive"] is False:
            break
        time.sleep(0.1)
    assert st["state"] == "finished", st
    kinds = [json.loads(line)["type"] for line in (case.root / "ratchet.jsonl").read_text().splitlines()]
    assert kinds.count("baseline") == 2 and kinds.count("candidate") == 3                # the log grew by the re-baseline record


def test_the_run_tool_names_the_rebaseline_option():
    doc = " ".join({t.name: t for t in asyncio.run(server.mcp.list_tools())}["ratchet_run"].description.split())
    assert "rebaseline" in doc and "overwrite" in doc


def test_m6_the_client_guide_names_the_real_number_of_tools():
    """The guide had 11 tools when the server had 21 (review m6). A change of the tool list must change docs/clients.md."""
    names = [t.name for t in asyncio.run(server.mcp.list_tools())]
    text = " ".join((REPO / "docs" / "clients.md").read_text().split())
    longest = max(names, key=len)
    assert f"The tools: {len(names)}, and the longest name has {len(longest)} characters (`{longest}`)" in text
    assert f"twice ({2 * len(names)})" in text and f"topopt has {len(names)} " in text
    assert f"topopt: {len(names)} tools, longest name {len(longest)} characters (`{longest}`)" in text
    for name in names:
        assert f"`{name}`" in text, name
