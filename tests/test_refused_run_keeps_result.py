"""A run that an input check refuses keeps the earlier run in its run folder (F4-43, CONTRACTS.md section 24.1).

A fresh `topopt run` used to call `RunWriter.clean()` before `build_problem` and the other checks of the input. A voxel guard or a
region error then removed the last good result of the folder, and nothing said so. Now the checks come first when the folder holds an
earlier run. A refusal writes `status.json` (state `error`) and the job files, and nothing else. The error text has one more sentence.
A folder with no earlier run behaves as before.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import trimesh

import topopt.problem as problem_module
from topopt.cli import EARLIER_RUN_KEPT, main
from topopt.snapshot import RUN_DIRS, RUN_FILES, old_run_files
from test_optimize import box, make_case
from test_user_errors import edit, no_dylib, one_line_error


def kept_state(folder):
    """Every file of the folder with the SHA-256 of its bytes, except `status.json` and the `job.*` files of the folder (a refused job writes those)."""
    folder = Path(folder)
    state = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}
    return {rel: digest for rel, digest in state.items() if rel != "status.json" and not rel.startswith("job.")}


def status_of(folder):
    return json.loads((Path(folder) / "status.json").read_text())


def run_args(yaml_path, out, *extra):
    return ["run", str(yaml_path), "--out", str(out), "--quiet", *extra]


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A finished run of the cantilever of `make_case` (2 mm voxels, 3 passes): `(case folder, run folder)`. Tests copy both."""
    tmp = tmp_path_factory.mktemp("refused_run")
    case = tmp / "case"
    case.mkdir()
    yaml_path = make_case(case, h=2.0, iterations=3)
    out = tmp / "run"
    assert main(run_args(yaml_path, out)) == 0
    return case, out


class Work:
    """A copy of the case and of the finished run folder, and the state of the run folder before the test."""

    def __init__(self, finished, tmp_path):
        case, out = finished
        self.case, self.out = tmp_path / "case", tmp_path / "run"
        shutil.copytree(case, self.case)
        shutil.copytree(out, self.out)
        self.yaml = self.case / "c.yaml"
        self.before = kept_state(self.out)

    def args(self, *extra):
        return run_args(self.yaml, self.out, *extra)

    def refused(self, capsys):
        """Start the run in the folder. It is refused (exit 1, one line on stderr). Return the line. The earlier run is kept, and the status says error."""
        assert main(self.args()) == 1
        err = one_line_error(capsys)
        self.check_kept()                      # first: this is what the card asks for
        self.check_error_status(err)
        return err

    def check_kept(self):
        """The earlier run is in the folder as it was: every file but `status.json` and the job files has the same bytes."""
        assert kept_state(self.out) == self.before
        assert {"run.json", "history.jsonl", "verdict.json", "checkpoint.npz", "result/final.stl"} <= set(self.before)   # the fixture has them
        assert any(name.startswith("snapshots/") for name in self.before)

    def check_error_status(self, err):
        """`status.json` says error with the text of the CLI, the job record is the one of the refused job (exit 1)."""
        st = status_of(self.out)
        assert (st["state"], st["iteration"], st["step"]) == ("error", 0, "setup")
        assert st["message"] == err.removeprefix("error: ").strip()
        assert st["message"].endswith(EARLIER_RUN_KEPT)
        assert (self.out / "job.FAILED").read_text() == "exit 1\n" and not (self.out / "job.DONE").exists()
        job = json.loads((self.out / "job.json").read_text())
        assert job["state"] == "failed" and job["exit_code"] == 1


@pytest.fixture
def work(finished, tmp_path):
    return Work(finished, tmp_path)


# --- the checks that the card names -------------------------------------------------------------

def test_voxel_guard_keeps_the_earlier_result(work, capsys):
    edit(work.yaml, lambda d: d["settings"].update(max_active_voxels=10))
    err = work.refused(capsys)
    assert "max_active_voxels" in err and f"(file: {work.yaml.resolve()})" in err
    assert err.rstrip().endswith(EARLIER_RUN_KEPT)


def test_the_new_sentence_is_one_sentence_after_the_old_text(work, capsys):
    edit(work.yaml, lambda d: d["settings"].update(max_grid_voxels=10))
    err = work.refused(capsys).strip().removeprefix("error: ")
    old_text, sentence = err[:-len(EARLIER_RUN_KEPT)], EARLIER_RUN_KEPT
    assert old_text.endswith("). ") and sentence == err[len(old_text):]       # the text of the guard and its `(file: ...)` stay as they were
    assert "max_grid_voxels" in old_text and sentence.count(". ") == 0 and sentence.endswith(").") and "kept" in sentence


def tiny_region(case, d):
    """A load region with no voxel at this voxel size (the region error of `build_problem`)."""
    box(case / "tiny.stl", (5.1, 5.1, 5.1), (5.3, 5.3, 5.3))
    d["regions"]["load"] = {"tip": "tiny.stl"}
    d["settings"]["voxel_size"] = 1.0


def open_region(case, d):
    """A load region STL that is not closed (the STL error of `build_problem`)."""
    trimesh.Trimesh(vertices=[[18, 4, 0], [20, 4, 0], [20, 6, 3]], faces=[[0, 1, 2]], process=False).export(case / "open.stl")
    d["regions"]["load"] = {"tip": "open.stl"}


@pytest.mark.parametrize("break_region, words", [(tiny_region, ("region load 'tip' has no voxel", "Use a smaller voxel_size")),
                                                 (open_region, ("open.stl", "watertight"))])
def test_region_error_keeps_the_earlier_result(work, capsys, break_region, words):
    edit(work.yaml, lambda d: break_region(work.case, d))
    err = work.refused(capsys)
    assert all(w in err for w in words), err


# --- the checks after the build (they were after the delete, too) ---------------------------------

def far_support(case, d):
    """The fixed plate does not touch the design: a piece of active voxels with no support (`floating_error`)."""
    box(case / "plate.stl", (-12, 0, 0), (-10, 10, 10))


def buckling_row_with_amg(case, d):
    """The buckling row needs a direct solver (`buckling_row_refusal`)."""
    d["settings"]["solver"] = "amg"
    d["buckling"] = {"min_factor": 2.0}


@pytest.mark.parametrize("break_input, words", [(far_support, "no support"), (buckling_row_with_amg, "buckling")])
def test_check_after_the_build_keeps_the_earlier_result(work, capsys, break_input, words):
    edit(work.yaml, lambda d: break_input(work.case, d))
    assert words in work.refused(capsys)


def test_backend_check_keeps_the_earlier_result(work, capsys, monkeypatch):
    """`backend_problem`: an explicit `cholesky` on a Mac without the library (the platform is faked, as in tests/test_user_errors.py)."""
    no_dylib(monkeypatch)
    edit(work.yaml, lambda d: d["settings"].update(solver="cholesky"))
    assert "settings.solver: amg" in work.refused(capsys)


def test_an_internal_error_and_ctrl_c_keep_the_earlier_result_too(work, monkeypatch):
    """The two other outcomes of the checks: the status says it, the earlier run stays, and the exit code is the one of before."""
    def boom(config):
        raise RuntimeError("boom")

    def interrupt(config):
        raise KeyboardInterrupt

    monkeypatch.setattr(problem_module, "build_problem", boom)
    assert main(work.args()) == 2
    work.check_kept()
    st = status_of(work.out)
    assert st["state"] == "error" and st["message"] == "internal error: boom. " + EARLIER_RUN_KEPT
    monkeypatch.setattr(problem_module, "build_problem", interrupt)
    assert main(work.args()) == 130
    work.check_kept()
    st = status_of(work.out)
    assert st["state"] == "error" and st["message"] == "stopped by the user (Ctrl+C). " + EARLIER_RUN_KEPT
    assert (work.out / "job.FAILED").read_text() == "exit 130\n"


def test_two_refused_runs_in_a_row_keep_the_earlier_result(work, capsys):
    edit(work.yaml, lambda d: d["settings"].update(max_active_voxels=10))
    for _ in range(2):
        work.refused(capsys)


# --- a folder with no earlier run: as before ----------------------------------------------------

def test_empty_folder_and_a_guard_error_is_as_before(finished, tmp_path, capsys):
    case = tmp_path / "case"
    shutil.copytree(finished[0], case)
    edit(case / "c.yaml", lambda d: d["settings"].update(max_active_voxels=10))
    out = tmp_path / "run"
    assert main(run_args(case / "c.yaml", out)) == 1
    err = one_line_error(capsys)
    assert "max_active_voxels" in err and "earlier run" not in err and EARLIER_RUN_KEPT not in err
    st = status_of(out)
    assert (st["state"], st["iteration"], st["step"]) == ("error", 0, "setup")
    assert st["message"] == err.removeprefix("error: ").strip()
    assert (out / "run.log").read_text() == "topopt run: cantilever\n"      # the line of the run that started, then the error
    assert (out / "snapshots").is_dir() and not list((out / "snapshots").iterdir())
    assert sorted(p.name for p in out.iterdir()) == ["job.FAILED", "job.guard", "job.json", "run.log", "snapshots", "status.json"]
    assert old_run_files(out) == []


def test_what_a_refused_run_leaves_is_not_an_earlier_run(finished, tmp_path, capsys):
    """A second refused run in the same folder behaves as the first: it has nothing to keep, so it says nothing about a kept run."""
    case = tmp_path / "case"
    shutil.copytree(finished[0], case)
    edit(case / "c.yaml", lambda d: d["settings"].update(max_active_voxels=10))
    out = tmp_path / "run"
    for _ in range(2):
        assert main(run_args(case / "c.yaml", out)) == 1
        assert EARLIER_RUN_KEPT not in one_line_error(capsys)
        assert status_of(out)["message"].endswith(")")
    assert old_run_files(out) == []


def test_old_run_files(tmp_path):
    d = tmp_path / "r"
    assert old_run_files(d) == []                                  # no folder
    d.mkdir()
    assert old_run_files(d) == []
    for name in ("status.json", "run.log", "job.json", "notes.txt"):
        (d / name).write_text("x")
    (d / "snapshots").mkdir()
    for name in RUN_DIRS:
        (d / name).mkdir(exist_ok=True)
    assert old_run_files(d) == []                                  # what a run leaves when it stops in its setup, and files that are not its own
    (d / "run.json").write_text("{}")
    (d / "result").joinpath("final.stl").write_text("x")
    (d / "verify").joinpath("deep").mkdir()
    assert old_run_files(d) == ["run.json", "result", "verify"]
    assert set(old_run_files(d)) <= set(RUN_FILES) | set(RUN_DIRS)
    for name in RUN_FILES:
        (d / name).write_text("x")
    (d / "snapshots" / "it_0000.bin").write_bytes(b"x")
    assert old_run_files(d) == [n for n in RUN_FILES if n not in ("status.json", "run.log")] + list(RUN_DIRS)


# --- a run that goes on: the old files go, the log is the same ----------------------------------

def normal_log(folder):
    """The lines of `run.log`: the run folder is a name, every number is a `#` (times, sizes and counts differ from run to run)."""
    text = (Path(folder) / "run.log").read_text().replace(str(folder), "<run>")
    return [re.sub(r"\d+(\.\d+)?(e[-+]?\d+)?", "#", line) for line in text.splitlines()]


def test_a_run_that_goes_on_deletes_the_old_files_and_logs_the_same_lines(work, finished, tmp_path):
    (work.out / "snapshots" / "it_0099.bin").write_bytes(b"old")
    (work.out / "notes.txt").write_text("keep me")
    assert main(work.args()) == 0
    assert not (work.out / "snapshots" / "it_0099.bin").exists()
    assert (work.out / "notes.txt").read_text() == "keep me"
    assert status_of(work.out)["state"] == "done" and (work.out / "job.DONE").exists()
    assert (work.out / "run.log").read_text().count("done in") == 1
    empty = tmp_path / "empty"                          # the same run in a folder with no earlier run (the order of before)
    assert main(run_args(work.yaml, empty)) == 0
    assert normal_log(work.out) == normal_log(empty)
    assert normal_log(work.out)[0] == "topopt run: cantilever" and "refine: <run>/result/refine.html (# s)" in normal_log(work.out)


def test_the_build_comes_before_the_delete_only_when_the_folder_holds_an_earlier_run(work, monkeypatch):
    """With an earlier run the build sees its snapshots. Without one the folder is cleaned first, as before: `snapshots/` is new and empty."""
    seen = []
    real = problem_module.build_problem

    def spy(config):
        folder = work.out if not seen else work.out.parent / "empty"
        seen.append(len(list((folder / "snapshots").iterdir())) if (folder / "snapshots").is_dir() else None)
        return real(config)

    monkeypatch.setattr(problem_module, "build_problem", spy)
    assert main(work.args()) == 0
    assert main(run_args(work.yaml, work.out.parent / "empty")) == 0
    assert seen == [4, 0]            # the 3 passes of the finished run make 4 snapshots; the new folder has the empty `snapshots/` that `clean` makes


# --- the MCP tool --------------------------------------------------------------------------------

@pytest.fixture
def server_state():
    pytest.importorskip("mcp")
    from topopt_mcp import server
    server._current.clear()
    yield server
    proc = server._current.get("proc")
    if proc is not None and proc.poll() is None:
        import os
        import signal
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._current.clear()


def test_run_topopt_into_a_finished_folder_with_a_guard_error(work, server_state, monkeypatch):
    server = server_state
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -m topopt run {{yaml}} --out {{out}} --quiet")
    edit(work.yaml, lambda d: d["settings"].update(max_active_voxels=10))
    r = server.run_topopt(str(work.yaml), str(work.out))
    assert "error" not in r, r
    server._current["proc"].wait(timeout=120)
    work.check_kept()
    st = server.run_status(str(work.out))
    assert st["state"] == "error" and st["process_alive"] is False
    assert "max_active_voxels" in st["summary"] and EARLIER_RUN_KEPT in st["summary"]
    assert st["status"]["message"].endswith(EARLIER_RUN_KEPT)
    assert st["job"]["state"] == "failed" and st["job"]["exit_code"] == 1
    res = server.run_result(str(work.out))
    assert res["done"] is False and EARLIER_RUN_KEPT in res["message"]
    assert "metrics" not in res                          # the files of the earlier run are there, and the tool does not give them as the result of this run
