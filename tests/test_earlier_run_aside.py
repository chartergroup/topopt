"""A fresh run keeps the earlier run in its folder until it has saved its first pass (F4-47, CONTRACTS.md section 24.1).

F4-43 moved the checks of the input before `RunWriter.clean()`. A refusal that needs the model or the first solve still came after the delete.
Now:

- a refusal that needs only the settings and the problem comes before the delete (robust mode with a stress row or the buckling row, a row of
  `limits` that cannot work), and `topopt check` says the same text as a warning;
- for the rest, `clean()` moves the earlier run into the hidden folder `.earlier-run`, the first saved pass deletes it, and a refusal, an error
  or Ctrl+C before that moves the files back. A kill in that window leaves the folder, and the next run moves the files back first.
"""
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import topopt.checkpoint as checkpoint_module
import topopt.optimize as optimize_module
from topopt import robust
from topopt.cli import EARLIER_RUN_KEPT, check_data, check_run_inputs, main
from topopt.config import load_config
from topopt.snapshot import RUN_DIRS, RUN_FILES, RunWriter, old_run_files
from test_optimize import box
from test_refused_run_keeps_result import finished, normal_log, run_args, server_state, status_of, work  # noqa: F401  (fixtures)
from test_user_errors import edit

ASIDE_DIR = ".earlier-run"      # the name is part of the contract (CONTRACTS.md section 24.1): a literal here, so a run of this file on the code of F4-43 still collects
ZERO = "load case 'tiny' applies no load"
STOPPED = "a run stopped before it saved its first pass"


def tiny_case(case, d):
    """A third load case with a force of 1e-300 N and a compliance limit: its compliance is exactly 0.0 at pass 0 (F4-37, `zero_compliance_refusal`)."""
    d["load_cases"].append({"name": "tiny", "forces": [{"region": "tip", "force": [0, 0, 1e-300]}], "compliance_limit": 50.0})


def full_state(folder):
    """Every file of the folder with its bytes (SHA-256): also `status.json` and the job files."""
    import hashlib
    folder = Path(folder)
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


def hidden(folder):
    return sorted(p.name for p in Path(folder).iterdir() if p.name.startswith(".earlier-run"))


def window_status(work, err):
    """`status.json` of a run that the optimizer stopped before its first saved pass: error, the text of the CLI, and the sentence about the kept run."""
    st = status_of(work.out)
    assert st["state"] == "error" and st["iteration"] == 0
    assert st["message"] == err.removeprefix("error: ").strip() and st["message"].endswith(EARLIER_RUN_KEPT)
    return st


# --- step 1: the refusals that need only the settings and the problem come before the delete -----------------------

def robust_with_a_stress_row(case, d):
    d["settings"]["robust"] = "eroded"
    d["load_cases"][0]["stress_limit"] = 30.0


def test_robust_mode_with_a_stress_row_is_refused_before_the_delete_and_check_says_it(work, capsys):
    edit(work.yaml, lambda d: robust_with_a_stress_row(work.case, d))
    said = robust.stress_stop_text(1) + ". The run stops at the start"
    assert main(["check", str(work.yaml)]) == 0                        # the check reports a refusal of the run as a warning (as it does for the buckling row)
    out = capsys.readouterr().out
    assert f"warning: {said}" in out
    data = check_data(work.yaml)
    assert said in data["warnings"] and said in data["derived"]["robust"]["facts"]      # `check_problem` returns these keys
    err = work.refused(capsys)                                         # the earlier run is byte-equal, and the status is the one of F4-43
    assert said in err and err.rstrip().endswith(EARLIER_RUN_KEPT)
    assert hidden(work.out) == []


def test_robust_mode_with_the_buckling_row_is_refused_before_the_delete(work, capsys):
    edit(work.yaml, lambda d: (d["settings"].update(robust="eroded"), d.update(buckling={"min_factor": 2.0})))
    assert "buckling" in work.refused(capsys)                          # the solver rule or the robust rule speaks first: both name the row
    assert hidden(work.out) == []


def test_a_row_of_limits_that_cannot_work_is_refused_before_the_delete_and_check_says_it(work, capsys):
    def far_region(d):
        box(work.case / "far.stl", (100, 100, 100), (102, 102, 102))
        d["regions"]["measure"] = {"far": "far.stl"}
        d["limits"] = {"displacement": [{"region": "far", "limit_mm": 1.0}]}

    edit(work.yaml, far_region)
    words = "measure region 'far' has no voxel inside the grid"
    assert main(["check", str(work.yaml)]) == 1                         # `check_data` binds the rows: an error, as before this change
    assert words in capsys.readouterr().err
    assert words in work.refused(capsys)
    assert hidden(work.out) == []


def test_check_run_inputs_builds_the_table_of_voxels_only_for_a_row_of_limits(work, monkeypatch):
    """The check of the rows of `limits` needs the free dofs. A problem with no such row pays nothing for it."""
    called = []
    real = optimize_module.reduced_system
    monkeypatch.setattr(optimize_module, "reduced_system", lambda problem: (called.append(1), real(problem))[1])
    check_run_inputs(load_config(work.yaml))
    assert called == []
    edit(work.yaml, lambda d: (d["regions"].update(measure={"mid": "mid.stl"}), d.update(limits={"displacement": [{"region": "mid", "limit_mm": 1.0}]}),
                               box(work.case / "mid.stl", (8, 0, 0), (12, 10, 10))))
    check_run_inputs(load_config(work.yaml))
    assert called == [1]


# --- step 2: a refusal, an error or Ctrl+C in the window gives the earlier run back ----------------------------

def test_zero_compliance_at_pass_0_keeps_the_earlier_run(work, capsys):
    """`zero_compliance_refusal` needs the first solve, so it stays in the window. The earlier run is moved aside and then back."""
    edit(work.yaml, lambda d: tiny_case(work.case, d))
    assert main(work.args()) == 1
    err = capsys.readouterr().err
    assert ZERO in err and len(err.strip().splitlines()) == 1
    work.check_kept()                                                  # every file but status.json and the job files: the same bytes
    window_status(work, err)
    assert hidden(work.out) == []
    assert (work.out / "job.FAILED").read_text() == "exit 1\n" and not (work.out / "job.DONE").exists()


def test_the_window_refusal_has_the_step_where_it_stopped(work, capsys):
    edit(work.yaml, lambda d: tiny_case(work.case, d))
    assert main(work.args()) == 1
    st = status_of(work.out)
    assert (st["state"], st["iteration"]) == ("error", 0) and st["step"] in ("solve", "sensitivity", "filter")


def test_an_internal_error_and_ctrl_c_in_the_first_solve_keep_the_earlier_run(work, monkeypatch):
    seen = []

    def boom(self, K):
        seen.append(sorted(p.name for p in (work.out / ASIDE_DIR).iterdir()))     # the earlier run is aside while the solve runs
        raise RuntimeError("boom")

    def interrupt(self, K):
        raise KeyboardInterrupt

    monkeypatch.setattr(optimize_module.Model, "solve", boom)
    assert main(work.args()) == 2
    work.check_kept()
    assert status_of(work.out)["message"] == "internal error: boom. " + EARLIER_RUN_KEPT
    assert {"run.json", "checkpoint.npz", "result", "snapshots", "history.jsonl"} <= set(seen[0]) and hidden(work.out) == []
    monkeypatch.setattr(optimize_module.Model, "solve", interrupt)
    assert main(work.args()) == 130
    work.check_kept()
    assert status_of(work.out)["message"] == "stopped by the user (Ctrl+C). " + EARLIER_RUN_KEPT
    assert (work.out / "job.FAILED").read_text() == "exit 130\n" and hidden(work.out) == []


def a_module_that_a_ctrl_c_stops():
    """`topopt.optimize` as a Ctrl+C finds it in the middle of a slow import: the names are not there yet."""
    module = types.ModuleType("topopt.optimize")

    def stop(name):
        raise KeyboardInterrupt

    module.__getattr__ = stop
    return module


@pytest.mark.parametrize("how", ["import-error", "ctrl-c"])
def test_a_stop_in_the_import_of_the_optimizer_comes_before_the_delete_and_keeps_the_earlier_run(work, monkeypatch, how):
    """The import of `RunStopped` and `run` is the first line of `_run_folder` that can stop, and it comes before `clean()`. It was the first line of
    the `try` after it: a stop there left `RunStopped` unbound, `except RunStopped` raised an `UnboundLocalError` instead of the error, and the handlers
    that give the earlier run back never ran."""
    monkeypatch.setitem(sys.modules, "topopt.optimize", None if how == "import-error" else a_module_that_a_ctrl_c_stops())
    code, text = (2, "internal error: ") if how == "import-error" else (130, "stopped by the user (Ctrl+C). ")
    assert main(work.args()) == code
    work.check_kept()
    st = status_of(work.out)
    assert (st["state"], st["iteration"], st["step"]) == ("error", 0, "setup")
    assert st["message"].startswith(text) and st["message"].endswith(EARLIER_RUN_KEPT) and "UnboundLocalError" not in st["message"]
    assert hidden(work.out) == [] and (work.out / "job.FAILED").read_text() == f"exit {code}\n"


def test_a_stop_in_the_import_of_the_optimizer_in_an_empty_folder_says_error_and_no_kept_run(work, tmp_path, monkeypatch):
    out = tmp_path / "fresh"
    monkeypatch.setitem(sys.modules, "topopt.optimize", a_module_that_a_ctrl_c_stops())
    assert main(run_args(work.yaml, out)) == 130
    st = status_of(out)
    assert (st["state"], st["message"]) == ("error", "stopped by the user (Ctrl+C)")      # nothing is kept: no sentence about a kept run
    assert hidden(out) == [] and old_run_files(out) == []


@pytest.mark.parametrize("how", ["ctrl-c", "oserror"])
def test_a_stop_in_the_status_line_after_the_clean_gives_the_earlier_run_back(work, monkeypatch, how):
    """After `clean()` the earlier run is aside. The status line `starting` comes before the old place of the `try`; so do the first warning lines."""
    real = RunWriter.status
    stop = KeyboardInterrupt() if how == "ctrl-c" else OSError(28, "No space left on device")

    def status(self, state=None, *args, **kwargs):
        if state == "starting":
            raise stop
        return real(self, state, *args, **kwargs)

    monkeypatch.setattr(RunWriter, "status", status)
    code, text = (130, "stopped by the user (Ctrl+C). ") if how == "ctrl-c" else (2, "internal error: [Errno 28] No space left on device. ")
    assert main(work.args()) == code
    work.check_kept()
    st = status_of(work.out)
    assert (st["state"], st["iteration"], st["step"]) == ("error", 0, "setup") and st["message"] == text + EARLIER_RUN_KEPT
    assert hidden(work.out) == [] and (work.out / "job.FAILED").read_text() == f"exit {code}\n"


def test_a_stop_in_a_warning_line_after_the_clean_gives_the_earlier_run_back(work, monkeypatch):
    """The first log lines (`warning: ...` for the facts of the job and for a recovered folder) come before the old place of the `try` too."""
    monkeypatch.setattr(RunWriter, "log", lambda self, line: (_ for _ in ()).throw(OSError(28, "No space left on device")))
    monkeypatch.setattr(checkpoint_module, "describe", lambda folder: {"complete": False, "iteration": 1, "iterations": 3, "written_at": "x"})
    assert main(work.args()) == 2
    work.check_kept()
    st = status_of(work.out)
    assert st["state"] == "error" and st["message"] == "internal error: [Errno 28] No space left on device. " + EARLIER_RUN_KEPT
    assert hidden(work.out) == []


def continue_args(work, parent):
    return work.args("--continue-from", str(parent), "--more-iterations", "2")


def test_a_continuation_into_a_folder_with_an_earlier_run_keeps_it_when_the_first_solve_fails(work, finished, tmp_path, monkeypatch):
    """`--continue-from` calls `clean()` on NEW and then copies the history and the snapshots of the parent into it: all of that is in the window."""
    parent = tmp_path / "parent"
    shutil.copytree(finished[1], parent)

    def boom(self, K):
        assert (work.out / ASIDE_DIR / "result" / "final.stl").is_file() and (work.out / "history.jsonl").is_file()      # the copy of the parent is there
        raise RuntimeError("boom")

    monkeypatch.setattr(optimize_module.Model, "solve", boom)
    assert main(continue_args(work, parent)) == 2
    work.check_kept()
    assert status_of(work.out)["message"] == "internal error: boom. " + EARLIER_RUN_KEPT
    assert hidden(work.out) == [] and not (parent / ASIDE_DIR).exists()


def test_a_continuation_that_goes_on_deletes_the_earlier_run_of_new_at_its_first_saved_pass(work, finished, tmp_path):
    parent = tmp_path / "parent"
    shutil.copytree(finished[1], parent)
    assert main(continue_args(work, parent)) == 0
    assert hidden(work.out) == [] and status_of(work.out)["state"] == "done"
    assert len((work.out / "history.jsonl").read_text().splitlines()) == 6      # the 4 lines of the parent and 2 new passes


def test_two_window_refusals_in_a_row_keep_the_earlier_run(work, capsys):
    edit(work.yaml, lambda d: tiny_case(work.case, d))
    for _ in range(2):
        assert main(work.args()) == 1
        capsys.readouterr()
        work.check_kept()
        assert hidden(work.out) == []


def test_a_move_back_that_fails_says_where_the_earlier_run_is_and_the_next_run_moves_it_back(work, capsys, monkeypatch):
    edit(work.yaml, lambda d: tiny_case(work.case, d))

    def fails(self):
        raise OSError("the disk is gone")

    with monkeypatch.context() as m:
        m.setattr(RunWriter, "restore_earlier", fails)
        assert main(work.args()) == 1
    err = capsys.readouterr().err
    assert "could not be moved back (the disk is gone)" in err and f"{ASIDE_DIR}/" in err and EARLIER_RUN_KEPT not in err
    assert hidden(work.out) == [ASIDE_DIR]
    edit(work.yaml, lambda d: d["settings"].update(max_active_voxels=10))     # the next run: refused by a check, after it made the folder right
    assert main(work.args()) == 1
    err = capsys.readouterr().err
    assert STOPPED in err and err.rstrip().endswith("is back in place, and the files of the stopped run are deleted.")
    work.check_kept()
    assert hidden(work.out) == []


# --- the window is the time until the first saved pass ------------------------------------------------------

def test_a_run_that_goes_on_deletes_the_earlier_run_when_its_first_pass_is_saved(work, tmp_path, monkeypatch):
    seen = []
    real = checkpoint_module.Checkpointer.save

    def spy(self, *args, **kwargs):
        d = self.out.dir
        before = sorted(os.listdir(d / ASIDE_DIR)) if (d / ASIDE_DIR).is_dir() else None
        top = sorted(p.name for p in d.iterdir() if p.name in RUN_FILES or p.name in RUN_DIRS)
        real(self, *args, **kwargs)
        seen.append((before, top, (d / ASIDE_DIR).exists()))

    monkeypatch.setattr(checkpoint_module.Checkpointer, "save", spy)
    assert main(work.args()) == 0
    monkeypatch.undo()
    assert len(seen) == 4                                               # passes 0 to 3
    first = seen[0]
    assert {"checkpoint.npz", "result", "history.jsonl", "snapshots", "verdict.json"} <= set(first[0])      # before the first save: the earlier run is aside
    assert "result" not in first[1] and "verdict.json" not in first[1] and "checkpoint.npz" not in first[1]      # and the folder holds only the new run
    assert first[2] is False                                            # after it: gone
    assert all(before is None and not after for before, _top, after in seen[1:])
    empty = tmp_path / "empty"                                          # at the end the folder is the one of a run in an empty folder
    assert main(run_args(work.yaml, empty)) == 0
    assert sorted(p.name for p in work.out.iterdir()) == sorted(p.name for p in empty.iterdir())
    assert normal_log(work.out) == normal_log(empty)


@pytest.mark.parametrize("leftovers", [False, True], ids=["empty-folder", "only-what-a-refused-run-leaves"])
def test_a_folder_with_no_earlier_run_never_has_the_hidden_folder(work, tmp_path, monkeypatch, leftovers):
    out = tmp_path / "fresh"
    if leftovers:                                                       # status.json, run.log and an empty snapshots/: no earlier run (F4-43)
        (out / "snapshots").mkdir(parents=True)
        (out / "status.json").write_text("{}")
        (out / "run.log").write_text("topopt run: x\n")
        assert old_run_files(out) == []
    seen = []
    real = checkpoint_module.Checkpointer.save

    def spy(self, *args, **kwargs):
        seen.append(hidden(self.out.dir))
        real(self, *args, **kwargs)
        seen.append(hidden(self.out.dir))

    monkeypatch.setattr(checkpoint_module.Checkpointer, "save", spy)
    assert main(run_args(work.yaml, out)) == 0
    assert len(seen) == 8 and all(h == [] for h in seen) and hidden(out) == []


def test_clean_with_an_earlier_run(work):
    """Every name of the earlier run is aside, the folder is ready for a new run, and `first_pass_saved` deletes the folder (twice is fine)."""
    w = RunWriter(work.out, quiet=True)
    names = sorted(set(old_run_files(work.out)) | {"status.json", "run.log"})
    w.clean()
    assert w.earlier_aside and sorted(os.listdir(work.out / ASIDE_DIR)) == names
    assert sorted(p.name for p in work.out.iterdir() if not p.name.startswith("job.")) == [ASIDE_DIR, "snapshots"]
    assert old_run_files(work.out) == []
    w.first_pass_saved()
    w.first_pass_saved()
    assert hidden(work.out) == [] and not w.earlier_aside
    assert w.restore_earlier() is False


def test_clean_without_an_earlier_run_moves_nothing(tmp_path):
    w = RunWriter(tmp_path / "r", quiet=True)
    w.clean()
    assert not w.earlier_aside and hidden(tmp_path / "r") == [] and (tmp_path / "r" / "snapshots").is_dir()
    assert w.restore_earlier() is False and (tmp_path / "r" / "snapshots").is_dir()


def test_a_clean_that_cannot_move_a_name_leaves_the_folder_as_it_was(work, monkeypatch):
    before = full_state(work.out)
    real, calls = os.replace, []

    def flaky(src, dst, *args, **kwargs):
        calls.append(src)
        if len(calls) == 4:
            raise OSError("no right to move this")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError, match="no right to move this"):
        RunWriter(work.out, quiet=True).clean()
    monkeypatch.undo()
    assert full_state(work.out) == before and hidden(work.out) == []


# --- a kill in the window: the next run makes the folder right first -----------------------------------------

class Killed(BaseException):
    """The process is gone: nothing after this point runs."""


def kill_at(monkeypatch, n):
    """From the `n`-th call on, `os.replace` and `os.rename` raise `Killed`."""
    calls = []
    real_replace, real_rename = os.replace, os.rename

    def guard(real):
        def call(*args, **kwargs):
            calls.append(1)
            if len(calls) >= n:
                raise Killed
            return real(*args, **kwargs)
        return call

    monkeypatch.setattr(os, "replace", guard(real_replace))
    monkeypatch.setattr(os, "rename", guard(real_rename))


def new_run_files(d):
    """What a new run has written before its first saved pass."""
    d = Path(d)
    (d / "status.json").write_text('{"state": "running"}')
    (d / "run.json").write_text("{}")
    (d / "run.log").write_text("topopt run: new\n")
    (d / "snapshots").mkdir(exist_ok=True)
    (d / "snapshots" / "it_0000.bin").write_bytes(b"new")


@pytest.mark.parametrize("n", range(1, 14))
@pytest.mark.parametrize("where", ["move-aside", "move-back"])
def test_a_kill_at_any_move_leaves_a_folder_that_recover_earlier_run_makes_right(work, monkeypatch, where, n):
    """The kill stops the process before its n-th `os.replace` or `os.rename`. The state that is left is read by `recover_earlier_run`: the earlier
    run is back, byte for byte (with its `status.json`), and no hidden folder is left. n beyond the last move: no kill happens."""
    from topopt.snapshot import recover_earlier_run
    before = full_state(work.out)
    w = RunWriter(work.out, quiet=True)
    with monkeypatch.context() as m:
        try:
            if where == "move-aside":
                kill_at(m, n)
                w.clean()
                new_run_files(work.out)
            else:
                w.clean()
                new_run_files(work.out)
                kill_at(m, n)
                w.restore_earlier()
        except Killed:
            pass
    recover_earlier_run(work.out)
    assert full_state(work.out) == before
    assert hidden(work.out) == []


def test_recover_earlier_run_does_nothing_in_a_folder_that_a_run_ended_on_its_own(work):
    from topopt.snapshot import has_aside, recover_earlier_run
    before = full_state(work.out)
    assert recover_earlier_run(work.out) is None and full_state(work.out) == before
    assert recover_earlier_run(work.out / "no-such-folder") is None
    assert not has_aside(work.out)


def test_recover_earlier_run_drops_the_folder_of_a_run_that_saved_its_first_pass(work):
    """A kill between the first checkpoint and the delete of the folder: the earlier run is given up already, the new run's files stay."""
    from topopt.snapshot import recover_earlier_run
    RunWriter(work.out, quiet=True).clean()
    new_run_files(work.out)
    (work.out / "checkpoint.npz").write_bytes(b"new checkpoint")
    now = full_state(work.out)
    now = {k: v for k, v in now.items() if not k.startswith(ASIDE_DIR)}
    assert recover_earlier_run(work.out) is None
    assert full_state(work.out) == now and hidden(work.out) == []


KILL = """
import os, sys
import topopt.optimize as optimize
import topopt.snapshot as snapshot
if sys.argv[1] == "solve":      # the first solve: the earlier run is aside, no pass is saved
    optimize.Model.solve = lambda self, K: os._exit(9)
else:                           # the first pass is saved, the folder is not deleted yet
    snapshot.RunWriter.first_pass_saved = lambda self: os._exit(9)
from topopt.cli import main
sys.exit(main(sys.argv[2:]))
"""


def killed_run(work, where, tmp_path):
    """A `topopt run` in a child process that the first solve (or the first save) kills: no handler runs. Return the state of the folder."""
    r = subprocess.run([sys.executable, "-c", KILL, where, *work.args()], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 9, r.stderr
    return full_state(work.out)


def test_a_run_killed_in_the_window_leaves_the_earlier_run_aside(work, tmp_path):
    from topopt.snapshot import has_aside
    left = killed_run(work, "solve", tmp_path)
    assert hidden(work.out) == [ASIDE_DIR] and has_aside(work.out)
    assert f"{ASIDE_DIR}/result/final.stl" in left and f"{ASIDE_DIR}/checkpoint.npz" in left      # the earlier run is whole
    assert "checkpoint.npz" not in left and "result/final.stl" not in left                        # and the folder holds the killed run only


def test_the_next_fresh_run_that_a_check_refuses_finds_the_earlier_run_back_and_keeps_it(work, tmp_path, capsys):
    killed_run(work, "solve", tmp_path)
    edit(work.yaml, lambda d: d["settings"].update(max_active_voxels=10))
    assert main(work.args()) == 1
    err = capsys.readouterr().err
    assert "max_active_voxels" in err and f"{EARLIER_RUN_KEPT} Before this run, {STOPPED} (the process was killed, or the host went down, or the files could not move back)" in err
    work.check_kept()
    assert hidden(work.out) == []
    assert status_of(work.out)["message"] == err.removeprefix("error: ").strip()


def test_the_next_fresh_run_that_goes_on_moves_the_earlier_run_back_first_and_logs_it(work, tmp_path, monkeypatch):
    killed_run(work, "solve", tmp_path)
    seen = []
    real = checkpoint_module.Checkpointer.save

    def spy(self, *args, **kwargs):
        if not seen:                                                    # before the first save: moved back, then aside again, by the new run
            seen.append(sorted(os.listdir(self.out.dir / ASIDE_DIR)))
        real(self, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module.Checkpointer, "save", spy)
    assert main(work.args()) == 0
    monkeypatch.undo()
    assert {"checkpoint.npz", "result", "history.jsonl"} <= set(seen[0])      # the aside folder of this run holds the EARLIER run, not the killed one's files
    log = (work.out / "run.log").read_text()
    assert log.count("warning: " + STOPPED) == 1 and log.index("warning: " + STOPPED) < log.index("topopt run: cantilever")
    assert status_of(work.out)["state"] == "done" and hidden(work.out) == []


def test_the_next_resume_moves_the_earlier_run_back_and_resumes_it(work, tmp_path):
    """The killed run never saved a pass, so there is nothing of it to resume. `--resume` makes the folder right, then resumes the earlier run."""
    killed_run(work, "solve", tmp_path)
    history = (work.out / ASIDE_DIR / "history.jsonl").read_bytes()
    assert main(work.args("--resume")) == 0
    assert hidden(work.out) == [] and (work.out / "history.jsonl").read_bytes() == history      # the passes of the earlier run are the ones that stay
    log = (work.out / "run.log").read_text()
    assert log.count("warning: " + STOPPED) == 1 and "resume" in log
    assert status_of(work.out)["state"] == "done" and (work.out / "result" / "final.stl").is_file()


def test_a_run_killed_after_its_first_save_is_resumed_and_its_folder_is_not_given_back(work, tmp_path):
    """The first checkpoint is on disk: the earlier run is given up. The next `--resume` continues the new run and deletes the leftover folder."""
    killed_run(work, "save", tmp_path)
    assert hidden(work.out) == [ASIDE_DIR] and (work.out / "checkpoint.npz").is_file()
    assert main(work.args("--resume")) == 0
    assert hidden(work.out) == []
    log = (work.out / "run.log").read_text()
    assert STOPPED not in log and status_of(work.out)["state"] == "done"
    assert len((work.out / "history.jsonl").read_text().splitlines()) == 4


# --- readers of the run folder -------------------------------------------------------------------------------

def test_readers_see_one_run_and_the_new_names_in_the_window(work, tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt.feedback import run_history
    from topopt_mcp import server
    runs = tmp_path / "ws" / "runs"
    runs.mkdir(parents=True)
    shutil.copytree(work.out, runs / "r1")
    RunWriter(runs / "r1", quiet=True).clean()
    new_run_files(runs / "r1")
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path / "ws"))
    assert [r["path"] for r in server.list_runs()["runs"]] == ["runs/r1"]
    assert run_history(tmp_path / "ws")["count"] == 1
    assert run_history(runs / "r1")["count"] == 1
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(runs / "r1"))        # the work folder is the run folder itself: the hidden folder is a child of it
    assert server.list_runs()["runs"] == []
    assert not (runs / "r1" / "result").exists() and (runs / "r1" / ASIDE_DIR / "result" / "final.stl").is_file()      # `topopt verify` reads result/ by name


# --- the MCP tool --------------------------------------------------------------------------------------------

def test_run_topopt_into_a_finished_folder_with_a_refusal_in_the_window(work, server_state, monkeypatch):
    server = server_state
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -m topopt run {{yaml}} --out {{out}} --quiet")
    edit(work.yaml, lambda d: tiny_case(work.case, d))
    r = server.run_topopt(str(work.yaml), str(work.out))
    assert "error" not in r, r
    server._current["proc"].wait(timeout=120)
    work.check_kept()
    assert hidden(work.out) == []
    st = server.run_status(str(work.out))
    assert st["state"] == "error" and st["process_alive"] is False
    assert ZERO in st["summary"] and EARLIER_RUN_KEPT in st["summary"]
    assert st["status"]["message"].endswith(EARLIER_RUN_KEPT)
    assert st["job"]["state"] == "failed" and st["job"]["exit_code"] == 1
    res = server.run_result(str(work.out))
    assert res["done"] is False and EARLIER_RUN_KEPT in res["message"] and "metrics" not in res
