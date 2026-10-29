"""After a refused run, the rows of `list_runs`, `run_history` and `compare_runs` say which keys are of the earlier run (P9-19, CONTRACTS.md section 24.1).

A fresh `topopt run` that an input check refuses keeps the earlier run in its folder (F4-43). Only `status.json` and the job files show the
refused job. `run_status` says which facts are of the earlier run (P9-18). The rows of the three list tools still gave the earlier run's
`iterations`, `final_stl_rel`, `yaml_path`, `settings` and `metrics` next to `state: error`, and an agent could report the mass saving of the
earlier run as the result of the refused job. Now such a row has the key `earlier_run` (after `state`): a note and the earlier run's keys.
At the top of the row those keys are `null`. A folder that is not this case gets the row of before: the folders of the cases below are built
here, the rule (`topopt.earlier_run.refused_over_earlier_run`) is false for them, and the rows equal the rows of the same code with the rule
forced off. The key lists of a plain row are recorded from the start commit of P9-19 (`ba7af78`), so that another change cannot move them.
"""
import copy
import json
import os
import shutil

import pytest

pytest.importorskip("mcp")

from topopt import feedback  # noqa: E402
from topopt.cli import EARLIER_RUN_KEPT, main  # noqa: E402
from topopt.runjob import RunJob  # noqa: E402
from topopt_mcp import server  # noqa: E402
from test_refused_run_keeps_result import finished, run_args, status_of, work  # noqa: E402,F401  (fixtures)
from test_refused_run_reply import (EARLIER_RUN_AT, GUARD_TEXT, NOT_THE_CASE, build, clean_server, iso, put,  # noqa: E402,F401  (clean_server: autouse)
                                    write_earlier_run, write_status)
from test_user_errors import edit  # noqa: E402

# --- the keys of a plain row, recorded from the start commit ba7af78 (a finished run) -------------------------------------------------------
LIST_KEYS = ["name", "path", "run_dir", "state", "iteration", "iterations", "problem", "process_alive", "modified", "mtime", "final_stl_rel"]
HISTORY_KEYS = ["name", "dir_name", "run_dir", "state", "iteration", "iterations", "message", "started_at", "modified", "mtime", "yaml_path",
                "yaml_exists", "yaml_newer_than_run", "settings", "metrics", "metrics_source", "problems", "path", "process_alive"]
HISTORY_DETAIL_KEYS = ["name", "dir_name", "run_dir", "state", "iteration", "iterations", "step", "message", "started_at", "modified", "mtime",
                       "yaml", "settings", "metrics", "metrics_source", "last_history", "files", "problems", "path", "process_alive"]
COMPARE_KEYS = ["name", "dir_name", "run_dir", "state", "iteration", "iterations", "started_at", "modified", "yaml_path", "run_dir_rel"]

# --- which key is of whom: every key of a row is in one of the two lists, so a new key in a row fails here until it is sorted ------------------
# Of the folder or of the job that the status file shows (they stay at the top of the row):
FOLDER_LIST = {"name", "path", "run_dir", "state", "iteration", "process_alive", "modified", "mtime"}
FOLDER_HISTORY = {"dir_name", "run_dir", "state", "iteration", "message", "modified", "mtime", "problems", "path", "process_alive"}
FOLDER_HISTORY_DETAIL = FOLDER_HISTORY | {"step"}
FOLDER_COMPARE = {"dir_name", "run_dir", "state", "iteration", "modified", "run_dir_rel"}


# --- folders that are built here ----------------------------------------------------------------------------------

def enrich(folder):
    """What a real run leaves and `write_earlier_run` does not: the problem name, the start time, the settings, the YAML file, `metrics.json`."""
    run_json = folder / "run.json"
    if not run_json.is_file():
        return folder
    run = json.loads(run_json.read_text())
    run.update(name="cantilever", started_at=iso(EARLIER_RUN_AT - 5), h=2.0, volume_fraction=0.3, solver="cholmod", n_active=275)
    put(run_json, json.dumps(run), EARLIER_RUN_AT)
    put(folder.parent / "case" / "c.yaml", "name: cantilever\n", EARLIER_RUN_AT - 100)
    metrics = {"compliance_initial": 59.1, "compliance_final": 6.77, "mass_design_space_g": 89.1, "mass_final_g": 13.8, "mass_saved_pct": 84.5,
               "volume_final_mm3": 11104.3, "run_time_s": 7.3, "iterations": 3}
    put(folder / "result" / "metrics.json", json.dumps(metrics), EARLIER_RUN_AT)
    return folder


def built(tmp_path, name, kind):
    return enrich(build(tmp_path / name, kind))


def finished_folder(tmp_path, name="done"):
    """A finished run: the earlier run, and a status `done` (no refused job)."""
    folder = tmp_path / name
    folder.mkdir(parents=True)
    write_earlier_run(folder)
    put(folder / "status.json", json.dumps({"state": "done", "iteration": 3, "step": "done", "elapsed_s": 1.5, "message": ""}), EARLIER_RUN_AT)
    return enrich(folder)


def running_job_folder(tmp_path):
    """A refused folder in which a new job runs: its record says `running`, and `status.json` is still the refused one (dated after the job)."""
    folder = built(tmp_path, "running", "refused")
    later = (folder / "status.json").stat().st_mtime + 10 * 365 * 86400
    os.utime(folder / "status.json", (later, later))
    return folder


# --- the rows -------------------------------------------------------------------------------------------------------

def list_row(folder):
    return next(r for r in server.list_runs()["runs"] if r["name"] == folder.name)


def history_row(folder, detail=False):
    return next(r for r in server.run_history(detail=detail)["runs"] if r["dir_name"] == folder.name)


def compare(folder, other, folder_is="b"):
    pair = (other, folder) if folder_is == "b" else (folder, other)
    return server.compare_runs(str(pair[0]), str(pair[1]), check_labels=False)


def all_rows(folder):
    return {"list_runs": list_row(folder), "run_history": history_row(folder), "run_history_detail": history_row(folder, detail=True)}


def detection_off(monkeypatch):
    """The rule forced off for every reader: the code path of before. On the start commit there is no such module, and nothing is patched."""
    try:
        import topopt.earlier_run as module
    except ImportError:
        return
    monkeypatch.setattr(module, "refused_over_earlier_run", lambda *args, **kw: False)


def unmark(row):
    """The row of before: the keys of `earlier_run` back at the top, the block gone."""
    row = copy.deepcopy(row)
    block = row.pop("earlier_run")
    row.update({k: v for k, v in block.items() if k != "note"})
    return row


def text(obj):
    return json.dumps(obj, indent=1, default=str)


def check_no_mark(folder, monkeypatch):
    """No mark: the rows have no `earlier_run`, and they equal (keys, order, values) the rows of the same code with the rule forced off."""
    live = all_rows(folder)
    assert not any("earlier_run" in row for row in live.values())
    with monkeypatch.context() as m:
        detection_off(m)
        off = all_rows(folder)
    for tool in live:
        assert text(live[tool]) == text(off[tool]), tool
    return live


# --- the case of the card -----------------------------------------------------------------------------------------

def test_list_runs_row_names_the_keys_of_the_earlier_run(tmp_path):
    row = list_row(built(tmp_path, "run", "refused"))
    assert list(row)[:5] == ["name", "path", "run_dir", "state", "earlier_run"]                # near the top, after `state`
    assert (row["state"], row["iteration"]) == ("error", 0)                                    # the refused job
    block = row["earlier_run"]
    assert "this job did not change it" in block["note"] and "`state` and `iteration` are of the refused job" in block["note"]
    assert {k: v for k, v in block.items() if k != "note"} == {"iterations": 3, "problem": "cantilever", "final_stl_rel": "run/result/final.stl"}
    for key in ("iterations", "problem", "final_stl_rel"):
        assert row[key] is None, key
        assert f"`{key}`" in block["note"]                                                       # the note names the keys of the block


def test_run_history_row_names_the_keys_of_the_earlier_run(tmp_path):
    folder = built(tmp_path, "run", "refused")
    row = history_row(folder)
    assert list(row)[:5] == ["name", "dir_name", "run_dir", "state", "earlier_run"]
    assert (row["state"], row["iteration"], row["dir_name"]) == ("error", 0, "run")
    assert row["message"].endswith(EARLIER_RUN_KEPT)                                            # the text of the refused job stays at the top
    block = row["earlier_run"]
    assert "this job did not change it" in block["note"] and "`state`, `iteration` and `message` are of the refused job" in block["note"]
    # the YAML of the earlier run is marked as such: not at the top of the row, where an agent reads it as the YAML of the job
    assert block["yaml_path"].endswith("/case/c.yaml") and block["yaml_exists"] is True and block["yaml_newer_than_run"] is False
    for key in ("yaml_path", "yaml_exists", "yaml_newer_than_run", "name", "iterations", "started_at", "settings", "metrics", "metrics_source"):
        assert row[key] is None, key
        assert f"`{key}`" in block["note"]
    assert (block["name"], block["iterations"], block["started_at"]) == ("cantilever", 3, iso(EARLIER_RUN_AT - 5))
    assert block["metrics"]["mass_saved_pct"] == 84.5 and block["metrics"]["run_time_s"] == 7.3 and block["metrics_source"] == "result/metrics.json"
    assert block["settings"]["n_active"] == 275 and block["settings"]["volume_fraction"] == 0.3
    assert row["problems"] == []


def test_run_history_detail_row_names_the_keys_of_the_earlier_run(tmp_path):
    row = history_row(built(tmp_path, "run", "refused"), detail=True)
    assert list(row)[:5] == ["name", "dir_name", "run_dir", "state", "earlier_run"]
    assert (row["state"], row["iteration"], row["step"]) == ("error", 0, "setup")
    block = row["earlier_run"]
    for key in ("yaml", "last_history", "files", "settings", "metrics"):
        assert row[key] is None and block[key], key
    assert block["last_history"]["iteration"] == 3 and block["yaml"]["path"].endswith("/case/c.yaml")
    assert block["files"]["snapshots"] == 1 and block["files"]["result/final.stl"] is True


def test_every_fact_of_the_earlier_run_stays_in_the_rows(tmp_path, monkeypatch):
    """Put the keys of `earlier_run` back and the row is the row of before, key for key, in the same order."""
    folder = built(tmp_path, "run", "refused")
    live = all_rows(folder)
    with monkeypatch.context() as m:
        detection_off(m)
        off = all_rows(folder)
    for tool in live:
        assert "earlier_run" in live[tool] and "earlier_run" not in off[tool], tool
        assert unmark(live[tool]) == off[tool], tool
        assert [k for k in live[tool] if k != "earlier_run"] == list(off[tool]), tool           # the order of the other keys stays
    assert off["run_history"]["metrics"]["mass_saved_pct"] == 84.5                               # the old row did say it, at the top, next to `error`


def test_the_keys_of_a_row_are_sorted_into_the_job_and_the_earlier_run(tmp_path, monkeypatch):
    """Each key of a plain row is of the folder or the job (it stays at the top) or of the earlier run (it moves). A new key must be sorted."""
    folder = built(tmp_path, "run", "refused")
    with monkeypatch.context() as m:
        detection_off(m)
        off = all_rows(folder)
    live = all_rows(folder)
    cases = {"list_runs": (FOLDER_LIST, server.LIST_RUNS_EARLIER_KEYS),
             "run_history": (FOLDER_HISTORY, feedback.HISTORY_EARLIER_KEYS),
             "run_history_detail": (FOLDER_HISTORY_DETAIL, feedback.HISTORY_EARLIER_KEYS_DETAIL)}
    for tool, (kept, moved) in cases.items():
        assert set(off[tool]) == kept | set(moved) and not kept & set(moved), tool
        assert all(live[tool][k] is None for k in moved), tool
        assert set(live[tool]["earlier_run"]) == set(moved) | {"note"}, tool


def test_a_real_refused_run_marks_the_rows(work, capsys):
    """The files of a real run and of a real refusal (the case of the card): the finished 3-pass run, then a run with a guard error."""
    edit(work.yaml, lambda d: d["settings"].update(max_active_voxels=10))
    assert main(work.args()) == 1
    work.check_kept()
    assert status_of(work.out)["message"].endswith(EARLIER_RUN_KEPT)
    rows = {"list_runs": list_row(work.out), "run_history": history_row(work.out), "run_history_detail": history_row(work.out, detail=True)}
    for tool, row in rows.items():
        assert row["state"] == "error" and row["iteration"] == 0 and "earlier_run" in row, tool
    block = rows["run_history"]["earlier_run"]
    assert block["iterations"] == 3 and block["settings"]["iterations"] == 3 and block["metrics"]["compliance_final"] is not None
    assert block["yaml_path"].endswith("/c.yaml") and block["yaml_path"] != str(work.yaml.resolve())   # the YAML of the earlier run, not the refused one
    assert str(work.yaml.resolve()) in rows["run_history"]["message"]                          # the refusal names its own file
    assert rows["list_runs"]["earlier_run"]["final_stl_rel"].endswith("run/result/final.stl") and rows["list_runs"]["final_stl_rel"] is None
    assert rows["run_history"]["metrics"] is None and rows["run_history"]["yaml_path"] is None


# --- compare_runs -------------------------------------------------------------------------------------------------

def test_compare_runs_names_the_earlier_run_of_a_refused_side(tmp_path):
    folder, other = built(tmp_path, "run", "refused"), finished_folder(tmp_path)
    res = compare(folder, other)
    assert res["ok"] is True
    assert "earlier_run" not in res["a"] and res["a"]["state"] == "done" and res["a"]["name"] == "cantilever"      # a plain run: as before
    b = res["b"]
    assert list(b)[:5] == ["name", "dir_name", "run_dir", "state", "earlier_run"] and (b["state"], b["iteration"]) == ("error", 0)
    assert b["name"] is None and b["iterations"] is None and b["started_at"] is None and b["yaml_path"] is None
    assert b["earlier_run"]["name"] == "cantilever" and b["earlier_run"]["yaml_path"].endswith("/case/c.yaml")
    assert "At the top of this entry they are null" in b["earlier_run"]["note"]
    assert res["notes"][0].startswith("Run B is a folder where an input check refused a job and kept an earlier run (see b.earlier_run)")
    assert res["summary"][0].startswith("run B: the folder holds an earlier run that a refused job did not change")
    # the setup, the metrics and the inputs are the facts of the earlier run, and the notes say so
    assert {r["key"]: r["b"] for r in res["metrics"]["rows"]}["mass_saved_pct"] == 84.5
    assert "setup, the metrics and the inputs of run B in this reply are of that earlier run" in res["notes"][0]


def test_compare_runs_with_the_refused_folder_as_a(tmp_path):
    folder, other = built(tmp_path, "run", "refused"), finished_folder(tmp_path)
    res = compare(folder, other, folder_is="a")
    assert "earlier_run" in res["a"] and "earlier_run" not in res["b"]
    assert res["notes"][0].startswith("Run A is a folder") and res["summary"][0].startswith("run A: ")
    assert not any(line.startswith("run B: the folder holds") for line in res["summary"])


def test_compare_runs_with_two_refused_folders_names_both(tmp_path):
    one, two = built(tmp_path, "one", "refused"), built(tmp_path, "two", "refused")
    res = compare(one, two, folder_is="a")      # `compare` puts the folder first: one is A, two is B
    assert "earlier_run" in res["a"] and "earlier_run" in res["b"]
    assert res["notes"][0].startswith("Run A is ") and res["notes"][1].startswith("Run B is ")
    assert res["summary"][0].startswith("run A: ") and res["summary"][1].startswith("run B: ")


def test_every_fact_of_a_refused_side_stays_in_compare_runs(tmp_path, monkeypatch):
    folder, other = built(tmp_path, "run", "refused"), finished_folder(tmp_path)
    live = compare(folder, other)
    with monkeypatch.context() as m:
        detection_off(m)
        off = compare(folder, other)
    assert unmark(live["b"]) == off["b"] and live["a"] == off["a"]
    assert [k for k in live["b"] if k != "earlier_run"] == list(off["b"])
    assert live["metrics"] == off["metrics"] and live["setup"] == off["setup"] and live["inputs"] == off["inputs"]
    assert live["notes"][1:] == off["notes"] and live["summary"][1:] == off["summary"]           # only the first line of each list is new
    assert set(off["b"]) == FOLDER_COMPARE | set(feedback.COMPARE_EARLIER_KEYS)                 # a new key must be sorted


# --- the mark is off ----------------------------------------------------------------------------------------------

def test_the_keys_of_a_plain_row_are_the_keys_of_the_start_commit(tmp_path):
    """Recorded from `ba7af78`: the keys and the order of the rows of a finished run."""
    folder = finished_folder(tmp_path)
    rows = all_rows(folder)
    assert list(rows["list_runs"]) == LIST_KEYS
    assert list(rows["run_history"]) == HISTORY_KEYS
    assert list(rows["run_history_detail"]) == HISTORY_DETAIL_KEYS
    res = compare(folder, finished_folder(tmp_path, "done2"))
    assert list(res["a"]) == list(res["b"]) == COMPARE_KEYS
    assert rows["run_history"]["metrics"]["mass_saved_pct"] == 84.5 and rows["list_runs"]["iterations"] == 3
    assert [r for r in res["notes"] if r.startswith("Run ")] == [] and not any("refused job did not change" in line for line in res["summary"])


@pytest.mark.parametrize("kind", NOT_THE_CASE)
def test_a_folder_that_is_not_the_case_gets_the_rows_of_before(kind, tmp_path, monkeypatch):
    """No earlier run, a resume that failed, a run that failed at a later pass, a status that is older than the job, the text with no run."""
    rows = check_no_mark(built(tmp_path, "run", kind), monkeypatch)
    assert list(rows["list_runs"]) == LIST_KEYS and list(rows["run_history"]) == HISTORY_KEYS
    if kind in ("failed_resume", "failed_later_pass", "job_began_after_the_status"):       # the folder is the run: its facts are its own
        assert rows["list_runs"]["iterations"] and rows["run_history"]["settings"] and rows["run_history"]["metrics"]["mass_saved_pct"] == 84.5
    if kind in ("leftovers", "sentence_without_a_run"):                                      # no run.json: there is no run to give facts of
        assert rows["run_history"]["settings"] is None and rows["list_runs"]["iterations"] is None


def test_a_finished_run_gets_the_rows_of_before(tmp_path, monkeypatch):
    rows = check_no_mark(finished_folder(tmp_path), monkeypatch)
    assert rows["list_runs"]["state"] == "done" and list(rows["list_runs"]) == LIST_KEYS


def test_a_run_that_failed_at_a_later_pass_in_a_real_folder_has_no_mark(work, monkeypatch):
    """A fresh run that the checks let through deletes the earlier run and fails at pass 2: the files are its own."""
    import topopt.snapshot as snapshot_module
    real = snapshot_module.RunWriter.add_history

    def add_history(self, rec):
        real(self, rec)
        if rec["iteration"] == 2:
            raise RuntimeError("boom at pass 2")

    monkeypatch.setattr(snapshot_module.RunWriter, "add_history", add_history)
    assert main(work.args()) == 2
    rows = check_no_mark(work.out, monkeypatch)
    assert rows["list_runs"]["iterations"] == 3 and rows["run_history"]["metrics"]["compliance_last"] is not None and rows["run_history"]["settings"]["iterations"] == 3


def test_a_resume_that_failed_in_a_real_folder_has_no_mark(work, monkeypatch):
    import topopt.optimize as optimize_module

    def boom(problem, out, plan=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(optimize_module, "run", boom)
    assert main(work.args("--resume")) == 2
    rows = check_no_mark(work.out, monkeypatch)
    assert rows["list_runs"]["iterations"] == 3 and rows["run_history"]["metrics"] is not None and rows["run_history"]["settings"]


def test_a_refused_run_in_an_empty_folder_has_no_mark(finished, tmp_path, capsys, monkeypatch):
    case = tmp_path / "case"
    shutil.copytree(finished[0], case)
    edit(case / "c.yaml", lambda d: d["settings"].update(max_active_voxels=10))
    out = tmp_path / "run"
    assert main(run_args(case / "c.yaml", out)) == 1
    rows = check_no_mark(out, monkeypatch)
    assert rows["run_history"]["state"] == "error" and "earlier run" not in rows["run_history"]["message"]
    assert rows["list_runs"]["final_stl_rel"] is None and rows["run_history"]["settings"] is None


def test_a_job_that_runs_in_a_folder_with_an_earlier_run_has_no_mark(tmp_path, monkeypatch):
    """A new job takes the folder after the refused one. Its record says `running`, and `status.json` is still the refused one."""
    folder = running_job_folder(tmp_path)
    job = RunJob(folder)
    job.start()
    try:
        rows = check_no_mark(folder, monkeypatch)
        assert rows["list_runs"]["iterations"] == 3 and rows["run_history"]["metrics"]["mass_saved_pct"] == 84.5      # the rows of before
        res = compare(folder, finished_folder(tmp_path))
        assert "earlier_run" not in res["b"]
    finally:
        job.finish(1)
    assert all("earlier_run" in row for row in all_rows(folder).values())      # the job ended with a failure, its status is the refused one: the mark is back


def test_the_mark_needs_the_status_of_a_refused_job(tmp_path, monkeypatch):
    """The same sentence in a status that is not the refused one (pass above 0, or another step) does not count."""
    for iteration, step in ((2, "solve"), (0, "starting")):
        folder = built(tmp_path, f"run-{iteration}-{step}", "refused")
        write_status(folder, f"{GUARD_TEXT} {EARLIER_RUN_KEPT}", iteration=iteration, step=step)
        check_no_mark(folder, monkeypatch)


def test_a_fault_in_the_rule_gives_the_row_of_before_not_a_lost_row(tmp_path, monkeypatch):
    folder = built(tmp_path, "run", "refused")
    with monkeypatch.context() as m:
        detection_off(m)
        off = all_rows(folder)
    import topopt.earlier_run as module

    def broken(*args, **kw):
        raise OSError("the disk is gone")

    monkeypatch.setattr(module, "refused_over_earlier_run", broken)
    rows = all_rows(folder)
    assert not any("earlier_run" in row for row in rows.values())
    for tool in rows:
        assert text(rows[tool]) == text(off[tool]), tool
    res = compare(folder, finished_folder(tmp_path))
    assert res["ok"] is True and "earlier_run" not in res["b"]


# --- the helper ---------------------------------------------------------------------------------------------------

def test_mark_row_moves_the_keys_and_keeps_the_rest():
    from topopt.earlier_run import mark_row
    row = {"a": 1, "state": "error", "b": 2, "c": 3, "iteration": 0}
    marked = mark_row(row, ("b", "c", "missing"))
    assert row == {"a": 1, "state": "error", "b": 2, "c": 3, "iteration": 0}                  # the argument is not changed
    assert list(marked) == ["a", "state", "earlier_run", "b", "c", "iteration"]
    assert (marked["b"], marked["c"], marked["a"], marked["iteration"]) == (None, None, 1, 0)
    assert {k: v for k, v in marked["earlier_run"].items() if k != "note"} == {"b": 2, "c": 3}
    assert "`b` and `c`" in marked["earlier_run"]["note"] and "missing" not in marked["earlier_run"]["note"]
    assert "`state` and `iteration` are of the refused job" in marked["earlier_run"]["note"]
    assert list(mark_row({"x": 1}, ("x",))) == ["x", "earlier_run"]                              # a row with no `state`: the block is last
