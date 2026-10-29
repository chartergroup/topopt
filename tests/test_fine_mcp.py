"""MCP side of the fine pass (#318): `run_topopt(fine_from=..., fine_pass=..., fine_voxel=..., fine_passes=..., fine_tol=...)` (CONTRACTS.md section 24.6).

The reply of a refusal has `error` and, for a parent that cannot start a fine pass, `fine` (`facts`, `options`). The reply of a run that starts has `fine`
(the parent, its pass and voxel, the fine voxel, the maximum, the tolerance and the facts). `fine_pass=true` names both folders. Every refusal reads only:
no folder, no pid file, and the parent does not change. The cantilever is the one of tests/test_fine_cli.py (275 elements at 2 mm).
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

from topopt import fine_pass as F  # noqa: E402
from topopt.cli import fine_folder, main  # noqa: E402
from topopt_mcp import server  # noqa: E402

from test_checkpoint import edit, run_args  # noqa: E402
from test_continue import parent_state  # noqa: E402
from test_optimize import make_case  # noqa: E402

PASSES = 6


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
    """The cantilever at 2 mm voxels with 6 passes and its finished run: `(yaml, parent folder)`."""
    root = tmp_path_factory.mktemp("fine_mcp")
    (root / "case").mkdir()
    yaml_path = make_case(root / "case", h=2.0, iterations=PASSES)
    parent = root / "parent"
    assert main(run_args(yaml_path, parent)) == 0
    return yaml_path, parent


def wait_for(call, states=("done", "error"), timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        st = call()
        if st.get("state") in states:
            return st
        time.sleep(0.3)
    raise AssertionError("the run did not finish")


def no_op_command(seen):
    def command(*args, **kwargs):
        seen.append((args, kwargs))
        return [sys.executable, "-c", "pass"]
    return command


def read_json(path):
    return json.loads(Path(path).read_text())


# --- the texts ---------------------------------------------------------------------------------------------

def test_the_parameters_are_optional_and_the_texts_name_them():
    """The text of run_topopt names fine_from, fine_pass and the guide; the guide has the detail (the room in the text goes to the guidance that main had)."""
    from topopt_mcp import replies
    params = inspect.signature(server.run_topopt).parameters
    assert [params[n].default for n in ("fine_from", "fine_pass", "fine_voxel", "fine_passes", "fine_tol")] == [None, False, None, None, None]
    doc = server.run_topopt.__doc__
    assert "fine_from" in doc and "fine_pass" in doc and replies.uri("run_topopt") in doc
    guide = replies.guide("run_topopt")
    for part in ("fine_from", "fine_pass", "fine_voxel", "fine_tol", "fine_passes", "maximum", "`fine_run_dir`", "<out_dir>-fine", "tolerance_from",
                 "settings.fine_pass.objective_change", "`fine_run`", "`fine_from`", "stop_run()", "continue_from"):
        assert part in guide, part
    assert "fine_pass" in server.check_problem.__doc__ and "fine_run" in server.run_status.__doc__ and "fine pass" in server.stop_run.__doc__


def test_the_guidance_that_main_had_stays_in_the_texts():
    """The fine pass text moved to the guide; the sentences of the texts at f63b576 are back (the review of e87ce2c cut them for room)."""
    ins, doc = server.INSTRUCTIONS, server.run_topopt.__doc__
    # #380: the solver sentences of the instructions moved to the descriptions of environment and check_problem (the cut of 2048 characters)
    assert "Say the reason (and the speed-up) first" in server.environment.__doc__
    assert "not of run_topopt" in server.check_problem.__doc__ and "a part of the name" in server.check_problem.__doc__
    from test_plugin_manifest import OLD_CLIENT_CUT
    assert len(ins) <= OLD_CLIENT_CUT
    for part in ("the reply names its pid and start time", "(bit-equal with one thread and a direct solver)", "The lost force is in check_problem (before)",
                 "the stiffness of such a case is null with a reason", "no upper limit", "names both overrides in one option",
                 "so you can show them"):
        assert part in doc, part
    from topopt_mcp import replies
    assert "The list is there only if there is a warning." in replies.guide("run_result")      # #174: the key lines moved to the guide


def test_the_command_has_the_fine_flags_after_the_stop_rules(monkeypatch, tmp_path):
    monkeypatch.delenv("TOPOPT_MCP_CMD", raising=False)
    y, o, p = tmp_path / "a.yaml", tmp_path / "o", tmp_path / "parent"
    base = server._command(y, o, 8765)
    assert not [a for a in base if a.startswith("--fine")]
    cmd = server._command(y, o, 8765, True, False, True, False, fine_from=p, fine_voxel=0.5, fine_passes=9, fine_tol=0.002)
    assert cmd[len(base):] == ["--override-lock", "--allow-changed-inputs", "--no-stop-rules", "--fine-from", str(p), "--fine-voxel", "0.5",
                               "--fine-passes", "9", "--fine-tol", "0.002"]
    assert server._command(y, o, 8765, fine_pass=True)[len(base):] == ["--fine-pass"]


# --- a fine pass from a parent folder ----------------------------------------------------------------

def test_a_run_that_starts_has_the_plan_in_fine_and_the_flags_in_the_command(case, tmp_path, monkeypatch):
    yaml_path, parent = case
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    before = parent_state(parent)
    out = tmp_path / "fine"
    r = server.run_topopt(str(yaml_path), str(out), fine_from=str(parent), fine_voxel=1.0, fine_passes=4)
    server._current["proc"].wait()
    assert "error" not in r, r
    f = r["fine"]
    assert (f["parent"], f["iteration"], f["parent_voxel_mm"], f["voxel_mm"], f["voxel_from"]) == (str(parent), PASSES, 2.0, 1.0, "fine_voxel")
    assert (f["max_passes"], f["max_from"], f["tolerance"], f["tolerance_from"], f["window"], f["source"]) == (4, "fine_passes", F.FINE_OBJECTIVE_CHANGE, "default", 5, "x")
    assert f["schedule_offset"] == PASSES
    assert f["facts"][0].startswith("fine pass:") and any("the maximum" in x or "at most" in x for x in f["facts"]), f["facts"]
    args, kwargs = seen[0]
    assert kwargs == {"fine_from": Path(parent).resolve(), "fine_pass": False, "fine_voxel": 1.0, "fine_passes": 4, "fine_tol": None}
    assert "fine_run_dir" not in r and parent_state(parent) == before         # the plan reads the parent only


def test_the_plan_of_a_run_with_the_voxel_of_the_part_has_one_voxel_and_the_test_on_the_grid(case, tmp_path, monkeypatch):
    yaml_path, parent = case
    monkeypatch.setattr(server, "_command", no_op_command([]))
    r = server.run_topopt(str(yaml_path), str(tmp_path / "fine"), fine_from=str(parent), fine_passes=1)
    server._current["proc"].wait()
    assert "error" not in r, r
    f = r["fine"]
    assert f["voxel_from"] == "part_voxel" and f["voxel_mm"] == f["part_voxel_mm"] and f["limiting_role"] in ("fixed", "load")
    assert f["limiting"] in ("plate", "tip") and f["limiting_thickness_mm"] == pytest.approx(2.0, abs=1e-3)
    assert set(f["grid_check"]) == {"voxel_mm", "start_mm", "steps", "region", "ok", "skipped"} and f["grid_check"]["voxel_mm"] == f["voxel_mm"]
    assert not {"part_voxel_level", "part_voxel_bands"} & set(f)
    assert any("the voxel that the part needs" in x for x in f["facts"][:1])


def test_the_plan_of_a_run_names_the_source_of_the_tolerance(case, tmp_path, monkeypatch):
    yaml_path, parent = case
    shutil.copytree(Path(yaml_path).parent, tmp_path / "case2")
    path = tmp_path / "case2" / Path(yaml_path).name
    edit(path, lambda d: d["settings"].update(fine_pass={"objective_change": 0.02}))
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    got = {}
    for name, kwargs in (("yaml", {}), ("option", {"fine_tol": 0.05})):
        r = server.run_topopt(str(path), str(tmp_path / name), fine_from=str(parent), fine_voxel=1.0, fine_passes=4, **kwargs)
        server._current["proc"].wait()
        assert "error" not in r, r
        got[name] = (r["fine"]["tolerance"], r["fine"]["tolerance_from"])
        server._current.clear()
    assert got == {"yaml": (0.02, "yaml"), "option": (0.05, "option")}


def test_a_fine_pass_through_the_server_ends_with_the_record_in_run_json_and_run_result(case, tmp_path):
    yaml_path, parent = case
    before = parent_state(parent)
    out = tmp_path / "fine"
    r = server.run_topopt(str(yaml_path), str(out), fine_from=str(parent), fine_voxel=1.0, fine_passes=3, fine_tol=0.5)
    assert "error" not in r, r
    st = wait_for(lambda: server.run_status(str(out)))
    assert st["state"] == "done", st
    run = read_json(out / "run.json")
    assert run["fine_from"]["voxel_mm"] == 1.0 and run["fine_from"]["tolerance"] == 0.5 and run["fine_from"]["max_passes"] == 3
    res = server.run_result(str(out), preview=False)
    assert res["fine_from"]["run_dir"] == str(Path(parent).resolve()) and res["fine_from"]["tolerance"] == 0.5
    assert res["fine_pass"]["recommend"] is True and res["fine_pass"]["run_voxel_mm"] == pytest.approx(1.0)
    assert "fine_run" not in st                                              # only the first folder of fine_pass=true has it
    assert parent_state(parent) == before


@pytest.mark.parametrize("kwargs, text", [
    ({"fine_voxel": 1.0}, "go with fine_from or fine_pass"),
    ({"fine_passes": 3}, "go with fine_from or fine_pass"),
    ({"fine_tol": 0.01}, "go with fine_from or fine_pass"),
    ({"fine_from": "P", "fine_pass": True}, "fine_from and fine_pass do not go together"),
    ({"fine_from": "P", "resume": True}, "fine_from does not go with resume"),
    ({"fine_from": "P", "continue_from": "P", "more_iterations": 2}, "fine_from does not go with continue_from"),
    ({"fine_from": "P", "more_iterations": 2}, "continue_from and more_iterations go together"),
    ({"fine_pass": True, "resume": True}, "fine_pass does not go with resume"),
    ({"fine_pass": True, "allow_changed_inputs": True}, "fine_pass does not go with allow_changed_inputs"),
    ({"fine_from": "P", "fine_voxel": 0}, "fine_voxel"),
    ({"fine_from": "P", "fine_voxel": -1.0}, "fine_voxel"),
    ({"fine_from": "P", "fine_passes": 0}, "fine_passes"),
    ({"fine_from": "P", "fine_passes": 2.5}, "fine_passes"),
    ({"fine_from": "P", "fine_tol": 0}, "fine_tol"),
    ({"fine_from": "P", "fine_tol": "abc"}, "fine_tol")])
def test_a_wrong_mix_or_number_is_an_error_that_writes_nothing(case, tmp_path, monkeypatch, kwargs, text):
    yaml_path, parent = case
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    out = tmp_path / "new"
    before = parent_state(parent)
    r = server.run_topopt(str(yaml_path), str(out), **{k: (str(parent) if v == "P" else v) for k, v in kwargs.items()})
    assert text in r["error"] and "--fine" not in r["error"]
    assert "pid" not in r and "fine" not in r and not out.exists() and seen == [] and not server._pid_file(out).exists()
    assert parent_state(parent) == before


def test_an_equal_voxel_a_new_folder_in_the_parent_and_a_parent_without_a_checkpoint_are_refused(case, tmp_path, monkeypatch):
    yaml_path, parent = case
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    before = parent_state(parent)
    out = tmp_path / "new"
    r = server.run_topopt(str(yaml_path), str(out), fine_from=str(parent), fine_voxel=2.0)
    assert "continue_from=PARENT with more_iterations=N" in r["error"] and "fine_voxel=H" in r["error"] and "--" not in r["error"]
    assert "pid" not in r and not out.exists()
    r = server.run_topopt(str(yaml_path), str(parent), fine_from=str(parent), fine_voxel=1.0)
    assert "parent run folder" in r["error"] and "pid" not in r
    r = server.run_topopt(str(yaml_path), str(parent / "more"), fine_from=str(parent), fine_voxel=1.0)
    assert "inside the parent run folder" in r["error"] and not (parent / "more").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    r = server.run_topopt(str(yaml_path), str(out), fine_from=str(empty), fine_voxel=1.0)
    assert "error" in r and not out.exists() and list(empty.iterdir()) == []
    assert seen == [] and parent_state(parent) == before and not server._pid_file(parent).exists()


def test_a_changed_input_is_a_refusal_with_facts_and_options_and_the_override_starts_the_run(case, tmp_path, monkeypatch):
    yaml_path, parent = case
    work = tmp_path / "case"
    shutil.copytree(yaml_path.parent, work)
    edit(work / yaml_path.name, lambda d: d["settings"].update(volume_fraction=0.35))
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    out = tmp_path / "new"
    before = parent_state(parent)
    r = server.run_topopt(str(work / yaml_path.name), str(out), fine_from=str(parent), fine_voxel=1.0)
    assert "settings.volume_fraction" in r["error"] and r["run_dir"] == str(out)
    assert any("settings.volume_fraction" in f for f in r["fine"]["facts"]) and any("allow_changed_inputs=true" in o for o in r["fine"]["options"])
    assert "pid" not in r and not out.exists() and not server._pid_file(out).exists() and seen == [] and parent_state(parent) == before
    r = server.run_topopt(str(work / yaml_path.name), str(out), fine_from=str(parent), fine_voxel=1.0, allow_changed_inputs=True)
    server._current["proc"].wait()
    assert "error" not in r and seen[0][0][3:] == (False, False, True, None)
    assert any("settings.volume_fraction" in f for f in r["fine"]["facts"])


def test_a_refusal_does_not_stop_the_old_dashboard_process(case, tmp_path, monkeypatch):
    yaml_path, parent = case
    stopped = []
    monkeypatch.setattr(server, "_alive_run", lambda: (tmp_path / "other", 12345))
    monkeypatch.setattr(server, "_is_busy", lambda d: False)
    monkeypatch.setattr(server, "_stop", lambda *a: stopped.append(a))
    r = server.run_topopt(str(yaml_path), str(parent), fine_from=str(parent), fine_voxel=1.0)
    assert "error" in r and stopped == []


# --- a continuation of a fine pass ---------------------------------------------------------------------

def test_a_continuation_of_a_fine_pass_through_the_server_keeps_the_fine_voxel_and_both_records(case, tmp_path):
    yaml_path, parent = case
    first, cont = tmp_path / "fine", tmp_path / "cont"
    assert "error" not in server.run_topopt(str(yaml_path), str(first), fine_from=str(parent), fine_voxel=1.0, fine_passes=3)
    assert wait_for(lambda: server.run_status(str(first)))["state"] == "done"
    r = server.run_topopt(str(yaml_path), str(cont), continue_from=str(first), more_iterations=2)       # the YAML is the one of the first run (2 mm)
    assert "error" not in r, r
    assert wait_for(lambda: server.run_status(str(cont)))["state"] == "done"
    run = read_json(cont / "run.json")
    assert run["h"] == 1.0 and run["iterations"] == 5 and run["continued_from"]["run_dir"] == str(first.resolve())
    assert run["fine_from"] == read_json(first / "run.json")["fine_from"]
    assert server.run_result(str(cont), preview=False)["fine_from"]["voxel_mm"] == 1.0


# --- the one command ---------------------------------------------------------------------------------------

def test_fine_pass_true_names_both_folders_and_puts_the_flag_in_the_command(case, tmp_path, monkeypatch):
    yaml_path, _parent = case
    seen = []
    monkeypatch.setattr(server, "_command", no_op_command(seen))
    out = tmp_path / "first"
    r = server.run_topopt(str(yaml_path), str(out), fine_pass=True, fine_passes=3)
    server._current["proc"].wait()
    assert "error" not in r, r
    assert r["run_dir"] == str(out) and r["fine_run_dir"] == str(fine_folder(out)) and r["fine"]["first_run_dir"] == str(out)
    assert r["fine"]["fine_run_dir"] == r["fine_run_dir"] and r["fine_run_dir_rel"].endswith("first-fine")
    assert seen[0][1] == {"fine_from": None, "fine_pass": True, "fine_voxel": None, "fine_passes": 3, "fine_tol": None}
    assert server._current["fine_run_dir"] == fine_folder(out)


def test_the_server_is_busy_while_the_fine_pass_is_to_come(tmp_path):
    first, fine = tmp_path / "first", tmp_path / "first-fine"
    first.mkdir()
    fine.mkdir()
    (first / "status.json").write_text(json.dumps({"state": "done"}))
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": True}}))
    server._current.update(run_dir=first, fine_run_dir=fine, started=time.time() - 100)
    assert server._is_busy(first) is True                                    # the first run is done, the fine pass has not written its status
    assert server._fine_run_block(first)["state"] == "not started" and server._fine_run_block(first)["run_dir"] == str(fine)
    (fine / "status.json").write_text(json.dumps({"state": "running"}))
    assert server._is_busy(first) is True and server._fine_run_block(first)["state"] == "running"
    (fine / "status.json").write_text(json.dumps({"state": "done"}))
    assert server._is_busy(first) is False and server._fine_run_block(first)["state"] == "done"
    (fine / "status.json").unlink()                                          # the part needs no finer voxel: the process ends after the first run
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": False}}))
    assert server._is_busy(first) is False
    block = server._fine_run_block(first)
    assert block["state"] == "none" and "no fine pass" in block["note"]
    server._current.update(run_dir=tmp_path / "other")
    assert server._fine_run_block(first) is None and server._fine_pending(first) is False


def test_the_fine_pass_of_one_command_runs_through_the_server(case, tmp_path):
    """The real command: the first run at 2 mm, then the fine pass at 1 mm (3 passes at most) in the sibling folder."""
    yaml_path, _parent = case
    out = tmp_path / "first"
    r = server.run_topopt(str(yaml_path), str(out), fine_pass=True, fine_voxel=1.0, fine_passes=3)
    assert "error" not in r, r
    fine = Path(r["fine_run_dir"])
    # While the fine pass is to come, a second run_topopt does not stop the process.
    st = wait_for(lambda: server.run_status(str(fine)))
    assert st["state"] == "done", st
    first = server.run_status(str(out))
    assert first["state"] == "done" and first["fine_run"]["state"] == "done" and first["fine_run"]["run_dir"] == str(fine)
    assert read_json(out / "run.json")["part_voxel"]["recommend"] is True and "fine_from" not in read_json(out / "run.json")
    rec = read_json(fine / "run.json")["fine_from"]
    assert rec["run_dir"] == str(out.resolve()) and rec["voxel_mm"] == 1.0 and rec["max_passes"] == 3 and rec["parent_voxel_mm"] == 2.0
    res = server.run_result(str(out), preview=False)
    assert res["fine_pass"]["recommend"] is True and "fine_from" not in res
    assert server.run_result(str(fine), preview=False)["fine_from"]["voxel_mm"] == 1.0


def test_a_given_fine_voxel_keeps_the_server_busy_and_the_fine_pass_pending_although_the_part_needs_no_finer_one(tmp_path):
    first, fine = tmp_path / "first", tmp_path / "first-fine"
    first.mkdir()
    fine.mkdir()
    (first / "status.json").write_text(json.dumps({"state": "done"}))
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": False}}))
    server._current.update(run_dir=first, fine_run_dir=fine, started=time.time() - 100, fine_voxel=None)
    assert server._is_busy(first) is False and server._fine_run_block(first)["state"] == "none"       # no fine_voxel: no fine pass, as before
    server._current.update(fine_voxel=0.5)                                   # run_topopt(fine_pass=True, fine_voxel=0.5): the fine pass follows
    assert server._is_busy(first) is True and server._fine_pending(first) is True
    assert server._fine_run_block(first)["state"] == "not started"
    (fine / "status.json").write_text(json.dumps({"state": "running"}))
    assert server._is_busy(first) is True and server._fine_run_block(first)["state"] == "running"
    (fine / "status.json").write_text(json.dumps({"state": "done"}))
    assert server._is_busy(first) is False and server._fine_run_block(first)["state"] == "done"
    (fine / "status.json").unlink()
    (first / "run.json").write_text(json.dumps({}))                          # the first run never reached its end (no part_voxel block): no fine pass
    assert server._fine_pending(first) is False


def test_the_fine_pass_of_one_command_with_a_fine_voxel_runs_through_the_server_although_the_part_needs_no_finer_one(tmp_path):
    """The real command: the run voxel (0.7 mm) is the voxel of the part, and `fine_voxel=0.65` is finer, so the fine pass runs at 0.65 mm as asked."""
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=0.7, iterations=2)
    out = tmp_path / "first"
    r = server.run_topopt(str(yaml_path), str(out), fine_pass=True, fine_voxel=0.65, fine_passes=1)
    assert "error" not in r, r
    assert "fine_voxel is given and finer than the run" in r["fine"]["facts"][0] and server._current["fine_voxel"] == 0.65
    fine = Path(r["fine_run_dir"])
    st = wait_for(lambda: server.run_status(str(fine)))
    assert st["state"] == "done", st
    first = server.run_status(str(out))
    assert first["state"] == "done" and first["fine_run"]["state"] == "done" and first["fine_run"]["run_dir"] == str(fine)
    assert read_json(out / "run.json")["part_voxel"]["recommend"] is False                  # the part needs no finer voxel, and the fine pass ran
    rec = read_json(fine / "run.json")["fine_from"]
    assert rec["voxel_mm"] == 0.65 and rec["voxel_from"] == "fine_voxel" and rec["parent_voxel_mm"] == 0.7
    fact = "--fine-voxel 0.65 mm is finer, so the fine pass runs at 0.65 mm as asked."
    assert fact in (fine / "run.log").read_text() and fact in (out / "run.log").read_text()


# --- the end state of the fine run, and stop_run during fine_pass=true -------------------------------------

class Ended:
    """A run process that has ended with this exit code."""
    pid = 0

    def __init__(self, code):
        self.code = code

    def poll(self):
        return self.code


def sleeper():
    """A harmless live process in its own process group (the fixture `clean_server` kills the group of `_current["proc"]` at the end of a test)."""
    import subprocess
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)


def write_status(folder, state, **more):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "status.json").write_text(json.dumps({"state": state, **more}))


def test_the_fine_run_state_is_the_real_end_state_after_the_process_has_ended(tmp_path):
    first, fine = tmp_path / "first", tmp_path / "first-fine"
    write_status(first, "done")
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": True}}))
    server._current.update(run_dir=first, fine_run_dir=fine, started=time.time() - 100, proc=Ended(1))
    refused = server._fine_run_block(first)
    assert refused["state"] == "refused" and "ended with exit 1" in refused["message"] and str(tmp_path / "first.stdout") in refused["log"]
    assert "guard" in refused["message"]
    write_status(fine, "error", message="model has 9 active voxels")
    failed = server._fine_run_block(first)
    assert failed["state"] == "failed" and failed["message"] == "model has 9 active voxels"
    (fine / "status.json").unlink()
    write_status(first, "error")                                              # the first run failed: there is no fine pass, and that is not a refusal
    none = server._fine_run_block(first)
    assert none["state"] == "none" and "no fine pass" in none["note"]
    write_status(first, "done")
    server._current["proc"] = sleeper()                                      # the process is alive: the fine pass is still to come
    assert server._fine_run_block(first)["state"] == "not started"


def test_a_fine_pass_refused_by_a_guard_of_the_server_run_shows_refused(case, tmp_path):
    """The real command: `fine_pass=true` with a fine voxel that the guard `max_active_voxels` refuses. The CLI of the one command refuses before the
    first run: the process ends with exit 1, nothing is written, and `fine_run.state` says `refused`."""
    yaml_path, _parent = case
    shutil.copytree(Path(yaml_path).parent, tmp_path / "case2")
    path = tmp_path / "case2" / Path(yaml_path).name
    edit(path, lambda d: d["settings"].update(max_active_voxels=500))
    out = tmp_path / "first"
    r = server.run_topopt(str(path), str(out), fine_pass=True, fine_voxel=0.7)
    assert "error" not in r, r
    server._current["proc"].wait(timeout=120)
    assert server._current["proc"].poll() == 1 and not out.exists() and not Path(r["fine_run_dir"]).exists()       # nothing is written
    assert "stops before the first run" in Path(str(out) + ".stdout").read_text()
    block = server._fine_run_block(out)
    assert block["state"] == "refused" and "exit 1" in block["message"] and block["log"].endswith("first.stdout")


def test_stop_run_with_no_run_dir_stops_the_fine_pass_when_it_runs(tmp_path):
    """scen6 of the review, kept fast: the first run is 2 passes at 2 mm; the fine pass has a maximum of 400 passes and a tolerance that never settles."""
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=2.0, iterations=2)
    out = tmp_path / "first"
    r = server.run_topopt(str(yaml_path), str(out), fine_pass=True, fine_voxel=1.0, fine_passes=400, fine_tol=1e-12)
    assert "error" not in r, r
    fine = Path(r["fine_run_dir"])
    end = time.time() + 120
    while time.time() < end:
        st = read_json(fine / "status.json") if (fine / "status.json").exists() else {}
        if st.get("state") == "running" and st.get("iteration", 0) >= 2:
            break
        time.sleep(0.2)
    else:
        raise AssertionError("the fine pass did not run")
    assert server._is_busy(out) is True
    reply = server.stop_run()
    assert reply["state"] == "stop_requested" and reply["run_dir"] == str(fine) and "fine pass" in reply["found_by"], reply
    assert (fine / "job.stop").is_file() and not (out / "job.stop").exists()
    st = wait_for(lambda: server.run_status(str(fine)), states=("stopped", "error", "done"), timeout=120)
    assert st["state"] == "stopped", st
    assert server.run_status(str(out))["fine_run"]["state"] == "stopped"


def test_stop_run_with_no_run_dir_stops_the_first_run_and_no_fine_pass_starts(tmp_path):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=2.0, iterations=400)
    edit(yaml_path, lambda d: d["settings"].update(stop_rules=False))
    out = tmp_path / "first"
    r = server.run_topopt(str(yaml_path), str(out), fine_pass=True, fine_voxel=1.0)
    assert "error" not in r, r
    end = time.time() + 120
    while time.time() < end:
        st = read_json(out / "status.json") if (out / "status.json").exists() else {}
        if st.get("state") == "running" and st.get("iteration", 0) >= 2:
            break
        time.sleep(0.2)
    else:
        raise AssertionError("the first run did not run")
    reply = server.stop_run()
    assert reply["state"] == "stop_requested" and reply["run_dir"] == str(out), reply
    st = wait_for(lambda: server.run_status(str(out)), states=("stopped", "error", "done"), timeout=120)
    assert st["state"] == "stopped", st
    server._current["proc"].wait(timeout=60)
    assert not Path(r["fine_run_dir"]).exists()                                # the fine pass did not start
    assert server._fine_run_block(out)["state"] == "none"


def test_stop_run_between_the_two_runs_writes_the_request_in_the_fine_folder_and_says_that_the_fine_pass_will_not_start(tmp_path):
    first, fine = tmp_path / "first", tmp_path / "first-fine"
    write_status(first, "done")
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": True}}))
    proc = sleeper()
    server._current.update(run_dir=first, fine_run_dir=fine, started=time.time() - 100, proc=proc)
    reply = server.stop_run()
    assert reply["state"] == "stop_requested" and reply["run_dir"] == str(fine) and reply["pid"] == proc.pid, reply     # not `none`: a state and the folder
    assert "has not started" in reply["message"] and "the fine pass did not start" in reply["message"] and "exit code 5" in reply["message"]
    request = read_json(fine / "job.stop")
    assert request["target_pid"] == proc.pid and request["by"] == "mcp stop_run" and not (first / "job.stop").exists()    # for the process that makes both runs
    again = server.stop_run()
    assert again["state"] == "already_requested" and again["run_dir"] == str(fine)


def test_stop_run_in_the_post_step_of_the_first_run_prevents_the_fine_pass(tmp_path):
    first, fine = tmp_path / "first", tmp_path / "first-fine"
    write_status(first, "postprocess", iteration=6)
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": True}}))
    proc = sleeper()
    server._current.update(run_dir=first, fine_run_dir=fine, started=time.time() - 100, proc=proc)
    reply = server.stop_run()          # before: the state `postprocess` and nothing written, and the fine pass started anyway
    assert reply["state"] == "stop_requested" and reply["run_dir"] == str(fine) and "in its post step" in reply["message"], reply
    assert read_json(fine / "job.stop")["target_pid"] == proc.pid and not (first / "job.stop").exists()


def test_the_window_before_the_fine_pass_exists_only_when_the_part_needs_a_fine_pass(tmp_path):
    first, fine = tmp_path / "first", tmp_path / "first-fine"
    write_status(first, "postprocess", iteration=6)
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": False}}))
    server._current.update(run_dir=first, fine_run_dir=fine, started=time.time() - 100, proc=sleeper())
    assert server._fine_not_started(first) is None                     # no fine pass to prevent: the stop of the post step is the one of any run
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": True}}))
    assert server._fine_not_started(first) == (fine, "postprocess")
    write_status(first, "running", iteration=3)
    assert server._fine_not_started(first) is None                     # the first run still works: the first folder is the target
    write_status(first, "done")
    write_status(fine, "running", iteration=1)
    assert server._fine_not_started(first) is None                     # the fine run works: the fine folder is the target (as before)


def test_the_fine_run_state_is_stopped_when_a_stop_request_came_after_the_first_run(tmp_path):
    first, fine = tmp_path / "first", tmp_path / "first-fine"
    write_status(first, "done")
    (first / "run.json").write_text(json.dumps({"part_voxel": {"recommend": True}}))
    server._current.update(run_dir=first, fine_run_dir=fine, started=time.time() - 100, proc=Ended(5))
    block = server._fine_run_block(first)
    assert block["state"] == "stopped" and "a stop request came after the first run: the fine pass did not start" in block["note"]
    server._current.update(proc=Ended(1))
    assert server._fine_run_block(first)["state"] == "refused"                    # any other exit before the fine run started is still a refusal


def test_the_text_of_stop_run_names_the_window_before_the_fine_pass():
    doc = server.stop_run.__doc__
    assert "From the post step of the first run until the fine pass has started" in doc and len(doc) < 4096