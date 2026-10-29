"""Tests of the places that write or name the part card: the run (post step), `topopt report` and `run_result` (CONTRACTS.md section 9.2)."""
import asyncio
import json
from pathlib import Path

from topopt import part_card, runinfo
from topopt.cli import main
from test_optimize import make_case
from part_card_helpers import add_verify, run_dir, work  # noqa: F401  (`work` is a fixture that `run_dir` needs)


# --- the post step writes the card -----------------------------------------------------------------------

def test_the_run_writes_the_card_with_no_agent_call(tmp_path, capsys):
    p = make_case(tmp_path, h=2.0, iterations=3)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out)]) == 0
    card = out / "result" / "report.md"
    assert card.is_file()
    text = card.read_text()
    m = json.loads((out / "result" / "metrics.json").read_text())
    assert text.startswith("# Part card: cantilever\n") and f"{m['mass_final_g']:.10g} g" in text
    assert "| down |" in text and "| side |" in text
    assert "part card: result/report.md" in capsys.readouterr().out
    assert "part card: result/report.md" in (out / "run.log").read_text()
    assert card.read_bytes() == part_card.build(out).encode()     # the run folder has no later fact: the card is current


def test_the_card_of_a_run_with_the_buckling_check_has_the_load_factor(tmp_path):
    """Review M1: the check runs after the post step, so the card is written after the check, not inside `finish`."""
    import yaml
    p = make_case(tmp_path, h=2.0, iterations=3)
    doc = yaml.safe_load(Path(p).read_text())
    doc["buckling"] = {"check": True}
    Path(p).write_text(yaml.safe_dump(doc, sort_keys=False))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    m = json.loads((out / "result" / "metrics.json").read_text())
    assert m["buckling"]["status"] == "done"
    text = (out / "result" / "report.md").read_text()
    assert "- Buckling (linear, voxel estimate, not a verdict): lowest load factor " in text
    assert "Buckling: not run" not in text and "no buckling result" not in text
    assert "Buckling: no load factors" not in text
    assert (out / "result" / "report.md").read_bytes() == part_card.build(out).encode()     # nothing came after the card
    assert (out / "run.log").read_text().count("part card: result/report.md") == 1           # one call


def test_a_card_that_fails_to_write_does_not_fail_the_run(tmp_path, monkeypatch):
    real = runinfo.write_atomic

    def refuse(path, data):
        if Path(path).name == part_card.CARD_FILE:
            raise OSError(28, "No space left on device")
        return real(path, data)

    monkeypatch.setattr(runinfo, "write_atomic", refuse)
    p = make_case(tmp_path, h=2.0, iterations=3)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert json.loads((out / "status.json").read_text())["state"] == "done"
    assert (out / "result" / "metrics.json").is_file() and (out / "result" / "final.stl").is_file()
    assert not (out / "result" / "report.md").exists()
    log = [l for l in (out / "run.log").read_text().splitlines() if "part card" in l]
    assert len(log) == 1 and log[0].startswith("warning: part card not written: OSError: ")


def test_a_card_that_fails_for_any_reason_in_the_run_log_line_is_still_one_line(tmp_path, monkeypatch):
    def boom(run_dir, log=None):
        raise RuntimeError("boom")        # even a fault in the helper itself must not fail the run

    monkeypatch.setattr(part_card, "write_logged", boom)
    p = make_case(tmp_path, h=2.0, iterations=3)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert [l for l in (out / "run.log").read_text().splitlines() if "part card" in l] == [
        "warning: part card not written: RuntimeError: boom"]


# --- the command ------------------------------------------------------------------------------------------

def test_the_report_command_writes_the_card_again_with_the_facts_now(run_dir, capsys):
    assert main(["report", str(run_dir)]) == 0
    assert capsys.readouterr().out.strip() == str(run_dir / "result" / "report.md")
    assert "Not run: the run folder has no `verify/verify.json`" in (run_dir / "result" / "report.md").read_text()
    add_verify(run_dir)
    assert main(["report", str(run_dir)]) == 0
    assert "Checked file: `final.stl`" in (run_dir / "result" / "report.md").read_text()


def test_the_report_command_stops_with_one_line_for_a_folder_with_no_result(tmp_path, capsys):
    assert main(["report", str(tmp_path / "none")]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    (tmp_path / "empty").mkdir()
    assert main(["report", str(tmp_path / "empty")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: empty has no result/metrics.json") and len(err.strip().splitlines()) == 1


# --- run_result -------------------------------------------------------------------------------------------------

def test_run_result_names_the_card_and_writes_it_again(run_dir):
    from topopt_mcp import server
    res = server.run_result(str(run_dir), preview=False)
    assert res["done"] is True
    assert Path(res["report_md"]) == (run_dir / "result" / "report.md").resolve() and res["report_md_rel"].endswith("result/report.md")
    assert "report_md_note" not in res
    assert "Not run: the run folder has no `verify/verify.json`" in Path(res["report_md"]).read_text()
    add_verify(run_dir)
    res = server.run_result(str(run_dir), preview=False)
    assert "Checked file: `final.stl`" in Path(res["report_md"]).read_text()
    stamp = Path(res["report_md"]).stat().st_mtime_ns
    server.run_result(str(run_dir), preview=False)
    assert Path(res["report_md"]).stat().st_mtime_ns == stamp         # no new fact: the file is not touched


def test_run_result_names_an_older_card_when_the_write_fails(run_dir, monkeypatch):
    from topopt_mcp import server
    part_card.write(run_dir)
    add_verify(run_dir)
    monkeypatch.setattr(runinfo, "write_atomic", lambda path, data: (_ for _ in ()).throw(OSError(30, "Read-only file system")))
    res = server.run_result(str(run_dir), preview=False)
    assert res["done"] is True and res["report_md_rel"].endswith("result/report.md")
    assert res["report_md_note"].startswith("the part card could not be written again: OSError: ")
    assert "Checked file" not in Path(res["report_md"]).read_text()          # the older card, as it was


def test_run_result_writes_nothing_into_a_shipped_recording(run_dir, tmp_path, monkeypatch):
    """Review m1: `<code folder>/recordings/<name>/` is committed. Another folder of the code folder still gets the card."""
    import shutil
    from topopt_mcp import server
    code = tmp_path / "code"
    rec = code / "recordings" / "hook"
    shutil.copytree(run_dir, rec)
    inside = code / "runs" / "mine"            # a user's run inside a clone of the code
    shutil.copytree(run_dir, inside)
    monkeypatch.setattr(server, "REPO", code)
    before = sorted(p.relative_to(rec).as_posix() for p in rec.rglob("*"))
    res = server.run_result(str(rec), preview=False)
    assert res["done"] is True
    assert sorted(p.relative_to(rec).as_posix() for p in rec.rglob("*")) == before and not (rec / "result" / "report.md").exists()
    assert "report_md" not in res and "topopt report" in res["report_md_note"] and "recording" in res["report_md_note"]
    res = server.run_result(str(inside), preview=False)
    assert Path(res["report_md"]) == (inside / "result" / "report.md").resolve() and (inside / "result" / "report.md").is_file()
    assert "report_md_note" not in res
    (rec / "result" / "report.md").write_text("an older card\n")      # a recording that has a card: its path is named, the file is not touched
    res = server.run_result(str(rec), preview=False)
    assert Path(res["report_md"]) == (rec / "result" / "report.md").resolve() and "topopt report" in res["report_md_note"]
    assert (rec / "result" / "report.md").read_text() == "an older card\n"


def test_run_result_has_no_card_key_when_there_is_no_card_and_the_write_fails(run_dir):
    from topopt_mcp import server
    (run_dir / "result" / "report.md").mkdir()
    res = server.run_result(str(run_dir), preview=False)
    assert res["done"] is True and "report_md" not in res and "report_md_rel" not in res
    assert "report_md_note" in res


def test_run_result_of_a_run_with_no_result_has_no_card(tmp_path):
    from topopt_mcp import server
    (tmp_path / "status.json").write_text(json.dumps({"state": "error", "message": "x"}))
    res = server.run_result(str(tmp_path), preview=False)
    assert res["done"] is False and "report_md" not in res


def test_the_run_result_description_names_the_card_and_keeps_the_key_text():
    from topopt_mcp.server import mcp
    from test_plugin_manifest import KEY_TEXT, MCP_TEXT_LIMIT, OLD_CLIENT_CUT
    from topopt_mcp import replies
    description = next(t.description for t in asyncio.run(mcp.list_tools()) if t.name == "run_result")
    assert len(description) <= OLD_CLIENT_CUT       # #174: the key lines moved to the guide
    assert "Reply keys: resource topopt://replies/run_result" in description
    for phrase in KEY_TEXT["run_result"]:
        assert 0 <= description.find(phrase) and description.find(phrase) + len(phrase) <= OLD_CLIENT_CUT, phrase
    text = replies.guide("run_result")
    assert len(text) <= MCP_TEXT_LIMIT
    assert "`report_md_rel`" in text
    assert "`result/report.md`, the one-page part card" in text
    assert not text.rstrip().endswith("part card, written again by this call.")        # not the last line: a later branch appends there


def test_run_result_has_mesh_fallbacks_only_when_the_mesh_step_took_one(run_dir):
    """#445: the key `mesh_fallbacks` is the lines of `mesh.fallbacks`; a normal run has no such key."""
    from topopt_mcp import server
    assert "mesh_fallbacks" not in server.run_result(str(run_dir), preview=False)
    f = run_dir / "result" / "metrics.json"
    m = json.loads(f.read_text())
    m["mesh"]["fallbacks"] = [{"pass": 1, "round": 1, "build": "calibration", "step": "snap", "error_type": "ValueError", "error": "the snap broke here", "kept": False}]
    f.write_text(json.dumps(m))
    res = server.run_result(str(run_dir), preview=False)
    assert res["mesh_fallbacks"] == part_card.mesh_fallback_lines(m) and len(res["mesh_fallbacks"]) == 1
    assert "round 1 of pass 1, step snap: ValueError: the snap broke here" in res["mesh_fallbacks"][0]
