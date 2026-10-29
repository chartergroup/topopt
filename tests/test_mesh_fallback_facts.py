"""#445: a fallback of the mesh step in any round of the volume calibration is a fact (CONTRACTS.md section 19, "Fallbacks of the mesh step").

`build` takes a fallback when a step raises an error that is no resource error. The mesh facts keep only the kept round, so a fallback in an
earlier round left no fact. Now `mesh.fallbacks` lists every fallback of every round, the log has one warning line for each, and the part card
and `run_result` carry a note. The tests inject the error into the call of one round on a small plate case (a deck on a design space).
"""
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import cli, part_card, post, smooth_sdf  # noqa: E402
from topopt.snapshot import RunWriter  # noqa: E402

from test_mesh_resource_errors import FromFn, fixed_run  # noqa: E402

BAD = ValueError("the snap broke here")
KEYS = {"pass", "round", "build", "step", "error_type", "error", "kept"}


@pytest.fixture(scope="module")
def normal(tmp_path_factory):
    d = tmp_path_factory.mktemp("normal")
    run_dir = fixed_run(d)
    m = post.finish(run_dir)
    return m, (run_dir / "result" / "final.stl").read_bytes()


def test_a_normal_run_has_an_empty_list_and_no_note(normal):
    m, _stl = normal
    assert m["mesh"]["fallbacks"] == []
    assert part_card.mesh_fallback_lines(m) == []


def test_a_snap_failing_in_the_first_round_only_is_in_the_facts(tmp_path, monkeypatch, caplog, normal):
    m0, _stl0 = normal
    rounds = len(m0["mesh"]["calibration"])
    assert rounds >= 2      # round 0 is then not the kept round: the old facts of the kept round show nothing of it
    run_dir = fixed_run(tmp_path)
    FromFn(monkeypatch, "snap_into_regions", BAD, "build", nth=1)
    with caplog.at_level(logging.WARNING):
        m = post.finish(run_dir)
    mesh = m["mesh"]
    assert mesh["snap"].get("error") is None          # the old fact of the kept round shows nothing: this is the gap of #445
    [e] = mesh["fallbacks"]
    assert set(e) == KEYS
    assert (e["pass"], e["round"], e["build"], e["step"], e["kept"]) == (1, 0, "calibration", "snap", False)
    assert e["error_type"] == "ValueError" and e["error"] == "the snap broke here"
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "mesh fallback" in r.getMessage()]
    assert len(lines) == 1 and "round 0 of pass 1" in lines[0] and "step snap" in lines[0] and "ValueError: the snap broke here" in lines[0]


def test_a_fallback_in_the_last_round_is_the_kept_round(tmp_path, monkeypatch, normal):
    last = len(normal[0]["mesh"]["calibration"]) - 1
    FromFn(monkeypatch, "snap_into_regions", BAD, "build", nth=last + 1)
    m = post.finish(fixed_run(tmp_path))["mesh"]
    [e] = m["fallbacks"]
    assert e["round"] == last and e["kept"] is True and e["step"] == "snap"
    assert m["snap"]["error"].startswith("ValueError")      # the old fact is still there


def test_the_build_without_the_move_is_in_the_list(tmp_path, monkeypatch):
    real = smooth_sdf._manifold_of

    def fail_without_move(*a, **k):
        frame = sys._getframe(1)
        if frame.f_code.co_name == "build" and frame.f_locals.get("snap") is False:   # `build(..., snap=False)`: the build after the calibration
            raise ValueError("the boolean broke here")
        return real(*a, **k)
    monkeypatch.setattr(smooth_sdf, "_manifold_of", fail_without_move)
    m = post.finish(fixed_run(tmp_path))["mesh"]
    [e] = m["fallbacks"]
    assert (e["build"], e["step"], e["round"], e["kept"]) == ("without_move", "exact", None, False)
    assert e["error"] == "the boolean broke here"
    assert "the build without the move" in part_card.mesh_fallback_lines({"mesh": m})[0]


def test_each_failing_step_of_a_round_has_its_entry(tmp_path, monkeypatch):
    FromFn(monkeypatch, "snap_into_regions", BAD, "build")
    FromFn(monkeypatch, "cut_to_envelope", TypeError("no cut"), "build")
    m = post.finish(fixed_run(tmp_path))["mesh"]
    got = sorted({(e["step"], e["error_type"]) for e in m["fallbacks"]})
    assert ("snap", "ValueError") in got and ("envelope_cut", "TypeError") in got
    assert len(m["fallbacks"]) == 2 * len(m["calibration"]) and {e["pass"] for e in m["fallbacks"]} == {1}
    lines = part_card.mesh_fallback_lines({"mesh": m})      # the same step and error in each round: one line, with the rounds
    rounds = ", ".join(str(k) for k in range(len(m["calibration"])))
    assert len(lines) == 2 and all(f"rounds {rounds} of pass 1 ({len(m['calibration']) - 1} is the kept round)" in x for x in lines)


def test_a_failed_float32_clean_up_is_in_the_list_with_one_log_line(tmp_path, monkeypatch, caplog):
    FromFn(monkeypatch, "_manifold_of", BAD, "_cleanup_float32", nth=1)        # the clean-up of round 0 keeps the mesh as it was
    with caplog.at_level(logging.WARNING):
        m = post.finish(fixed_run(tmp_path))["mesh"]
    [e] = m["fallbacks"]
    assert (e["pass"], e["round"], e["build"], e["step"], e["error_type"], e["error"]) == (1, 0, "calibration", "cleanup", "ValueError",
                                                                                         "the snap broke here")
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "the snap broke here" in r.getMessage()]
    assert len(lines) == 1 and lines[0].startswith("mesh fallback in round 0 of pass 1, step cleanup")


def test_the_run_log_has_the_fallback_line(tmp_path, monkeypatch):
    """CLI: `cli.finish` writes the lines of the card to `run.log` (Acceptance 1 of #445), once, with "warning: " before them."""
    run_dir = fixed_run(tmp_path)
    FromFn(monkeypatch, "snap_into_regions", BAD, "build", nth=1)
    cli.finish(RunWriter(run_dir, quiet=True), run_dir)
    log_lines = [x for x in (run_dir / "run.log").read_text().splitlines() if "took a fallback" in x]
    assert log_lines == ["warning: " + x for x in part_card.mesh_fallback_lines(json.loads((run_dir / "result" / "metrics.json").read_text()))]
    assert len(log_lines) == 1 and log_lines[0].startswith("warning: the mesh step took a fallback in round 0 of pass 1, step snap: ValueError: the snap broke here.")


@pytest.mark.parametrize("mesh", [{"fallbacks": []}, {}], ids=["empty", "before-445"])
def test_the_run_log_has_no_fallback_line_without_a_fallback(tmp_path, monkeypatch, mesh):
    (tmp_path / "result").mkdir()
    (tmp_path / "result" / "metrics.json").write_text(json.dumps({"mesh": mesh}))
    monkeypatch.setattr(post, "finish", lambda *a, **k: None)
    cli.finish(RunWriter(tmp_path, quiet=True), tmp_path)
    assert "fallback" not in ((tmp_path / "run.log").read_text() if (tmp_path / "run.log").exists() else "")
