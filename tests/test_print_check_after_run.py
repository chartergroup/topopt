"""The print check at the end of every run (CONTRACTS.md section 32, "After every run"): `result/print_check.json`, its states in the run
summary and in `run_result`, and the rule that it reports and gates nothing."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fake_run  # noqa: E402
import test_part_only as T  # noqa: E402
from topopt import post, printcheck  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt_mcp import server  # noqa: E402

FILE = "print_check.json"


def report_of(run_dir) -> dict:
    return json.loads((Path(run_dir) / "result" / FILE).read_text())


@pytest.fixture(scope="module")
def bar(tmp_path_factory):
    """One small real run (the bar of test_part_only: voxel 1 mm, 8 passes) through the CLI, and its log."""
    folder = tmp_path_factory.mktemp("pc_bar")
    yml = T.write_case(folder)
    out = folder / "run"
    assert main(["run", str(yml), "--out", str(out), "--quiet"]) == 0
    return {"folder": folder, "yaml": yml, "out": out}


@pytest.fixture()
def fake(tmp_path):
    fake_run.run(tmp_path, spi=0.0, n_it=4)
    (tmp_path / "result" / "metrics.json").unlink()
    return tmp_path


# -- the post step ------------------------------------------------------------------------------------------------------
def test_the_post_step_writes_the_report(fake):
    post.finish(fake)
    rep = report_of(fake)
    assert rep["ok"] is True and [c["id"] for c in rep["checks"]] == [
        "mesh_closed", "bodies", "closed_voids", "bed_fit", "overhang", "support_volume", "plate_contact", "wall_thickness"]
    assert rep["input"]["run_dir"] == str(fake.resolve()) and rep["gate"]["state"] == "off"
    assert json.loads((fake / "status.json").read_text())["state"] == "done"
    assert not list(fake.rglob("*.tmp"))


def test_the_report_is_the_report_of_a_direct_call(fake):
    post.finish(fake)
    direct = printcheck.check(fake, contain_errors=True)
    drop = ("timing_s", "elapsed_s")
    assert {k: v for k, v in report_of(fake).items() if k not in drop} == {k: v for k, v in direct.items() if k not in drop}


def test_a_check_that_raises_gives_ok_false_and_the_run_ends_done(fake, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("the check broke")
    monkeypatch.setattr(printcheck, "check", boom)
    metrics = post.finish(fake)
    assert metrics["triangles"] > 0 and (fake / "result" / "final.stl").is_file()
    assert report_of(fake) == {"ok": False, "error": "RuntimeError: the check broke"}
    assert json.loads((fake / "status.json").read_text())["state"] == "done"


def test_a_report_that_cannot_be_written_does_not_raise(fake, monkeypatch):
    post.finish(fake)
    monkeypatch.setattr(Path, "replace", lambda self, target: (_ for _ in ()).throw(OSError("disk full")))
    rec = printcheck.write_for_run(fake)
    assert rec["ok"] is False and "disk full" in rec["error"]


def test_a_report_that_json_cannot_write_gives_ok_false_and_the_run_ends_done(fake, monkeypatch):
    real = printcheck.check
    monkeypatch.setattr(printcheck, "check", lambda *a, **k: {**real(*a, **k), "extra": np.int64(1)})
    post.finish(fake)
    rep = report_of(fake)
    assert rep["ok"] is False and rep["error"].startswith("TypeError") and "int64" in rep["error"]
    assert json.loads((fake / "status.json").read_text())["state"] == "done"
    assert not list(fake.rglob("*.tmp"))


def test_a_failed_write_leaves_no_old_report_and_no_tmp_file(fake, monkeypatch):
    post.finish(fake)
    assert server.run_result(str(fake), preview=False)["print_check"]["ok"] is True
    monkeypatch.setattr(Path, "replace", lambda self, target: (_ for _ in ()).throw(OSError("disk full")))
    post.finish(fake)                                                      # a second post step, as a post-only resume runs it
    assert not (fake / "result" / FILE).exists() and not list(fake.rglob("*.tmp"))
    assert "print_check" not in server.run_result(str(fake), preview=False)
    assert json.loads((fake / "status.json").read_text())["state"] == "done"


def test_write_done_false_still_gets_the_report_and_keeps_the_status(fake):
    post.finish(fake, write_done=False)
    assert report_of(fake)["ok"] is True
    assert json.loads((fake / "status.json").read_text())["state"] == "postprocess"


def test_an_old_report_is_replaced(fake):
    (fake / "result").mkdir(exist_ok=True)
    (fake / "result" / FILE).write_text(json.dumps({"ok": False, "error": "old"}))
    post.finish(fake)
    assert report_of(fake)["ok"] is True


# -- the short form ---------------------------------------------------------------------------------------------------
def test_summary_names_the_failed_checks_and_the_unknown_checks_that_have_a_limit():
    rep = {"ok": True, "partial": True, "summary": {"pass": 4, "fail": 1, "unknown": 3, "failed": ["bodies"]},
           "checks": [{"id": "bodies", "state": "fail", "limit": {"value": 1}},
                      {"id": "overhang", "state": "unknown", "limit": {"value": 5.0}},
                      {"id": "plate_contact", "state": "unknown", "limit": {"value": None}},
                      {"id": "wall_thickness", "state": "unknown", "limit": {"value": 1.2}},
                      {"id": "bed_fit", "state": "pass", "limit": {"value": [220, 220]}}]}
    s = printcheck.summary_of(rep)
    assert s == {"ok": True, "pass": 4, "fail": 1, "unknown": 3, "failed": ["bodies"],
                 "unknown_with_limit": ["overhang", "wall_thickness"], "partial": True}
    assert printcheck.summary_line(s) == ("print check: 4 pass, 1 fail, 3 unknown; failed: bodies; "
                                          "unknown with a limit: overhang, wall_thickness (print_check.json)")


def test_summary_of_a_failed_report_and_of_no_report():
    s = printcheck.summary_of({"ok": False, "error": "X: y"})
    assert s == {"ok": False, "error": "X: y"} and "not made (X: y)" in printcheck.summary_line(s)
    assert printcheck.summary_of(None) is None and printcheck.summary_line(None) is None


# -- run_result and the CLI summary -------------------------------------------------------------------------------------
def test_run_result_shows_the_states(fake):
    post.finish(fake)
    pc = server.run_result(str(fake), preview=False)["print_check"]
    assert pc["ok"] is True and pc["pass"] + pc["fail"] + pc["unknown"] == 8
    assert pc["failed"] == [] and Path(pc["file"]) == (fake / "result" / FILE).resolve() and pc["file_rel"].endswith(FILE)


def test_run_result_of_a_run_from_before_has_no_key(fake):
    post.finish(fake)
    (fake / "result" / FILE).unlink()
    assert "print_check" not in server.run_result(str(fake), preview=False)


def test_the_run_log_has_the_states_and_the_run_ends_done(bar):
    log = (bar["out"] / "run.log").read_text().splitlines()
    line = [x for x in log if "print check:" in x]
    assert len(line) == 1 and line[0].rstrip().endswith("(print_check.json)")
    assert json.loads((bar["out"] / "status.json").read_text())["state"] == "done"
    assert report_of(bar["out"])["ok"] is True


def test_a_post_only_resume_with_a_changed_print_key_gives_a_new_report(bar):
    before = report_of(bar["out"])
    assert next(c for c in before["checks"] if c["id"] == "plate_contact")["state"] == "unknown"
    doc = yaml.safe_load(bar["yaml"].read_text())
    doc["print"] = {"min_plate_contact_mm2": 1.0e9}
    keyed = bar["yaml"].with_name("keyed.yaml")
    keyed.write_text(yaml.safe_dump(doc, sort_keys=False))
    copy = bar["folder"] / "copy"
    shutil.copytree(bar["out"], copy)
    assert main(["run", str(keyed), "--out", str(copy), "--resume", "--allow-changed-inputs", "--quiet"]) == 0
    after = report_of(copy)
    plate = next(c for c in after["checks"] if c["id"] == "plate_contact")
    assert plate["state"] == "fail" and plate["limit"]["value"] == 1.0e9
    assert after["summary"]["failed"] == ["plate_contact"]
    assert json.loads((copy / "status.json").read_text())["state"] == "done"      # a failed check does not fail the run
    assert "failed: plate_contact" in [x for x in (copy / "run.log").read_text().splitlines() if "print check:" in x][-1]
    assert server.run_result(str(copy), preview=False)["print_check"]["failed"] == ["plate_contact"]
