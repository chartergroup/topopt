"""The print check: the keys `partial` and `coverage` and the line of text (TC-2, CONTRACTS.md section 44).

A check that stops with an exception is a record with the state `unknown` and the key `error` when the caller sets `contain_errors`
(the CLI and the MCP tool do); the other checks go on. The tests force one failure at a time with a monkeypatch and look at the
records, the text, the exit code and the gate. A check that the inputs do not ask for is `not_run` and does not make the report partial.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_printcheck import COMB, box, by_id, plate, save, states  # noqa: E402

from topopt import print_report as pr  # noqa: E402
from topopt import printcheck as pc  # noqa: E402
from topopt.cli import main  # noqa: E402

ALL = ["mesh_closed", "bodies", "closed_voids", "bed_fit", "overhang", "support_volume", "plate_contact", "wall_thickness"]


def boom(*_a, **_k):
    raise ValueError("boom")


def cli(capsys, *argv):
    code = main(["printcheck", *map(str, argv)])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_a_clean_report_is_not_partial_and_an_unknown_with_no_limit_is_a_fact_not_a_defect():
    r = pc.check(COMB)
    assert states(r)["overhang"] == "unknown"                       # no max_overhang_area_mm2: the value is a fact
    assert r["partial"] is False
    assert r["coverage"] == {"ran": ALL, "failed": [], "not_run": []}
    assert list(r)[:3] == ["ok", "partial", "coverage"]
    json.dumps(r)


def test_a_check_that_raises_is_a_failed_check_and_the_others_go_on(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "check_overhang", boom)
    p = save(plate(2.0), tmp_path / "p.stl")
    r = pc.check(p, print_settings={"min_wall_mm": 1.0}, strict=True, contain_errors=True)
    assert r["partial"] is True
    assert r["coverage"]["failed"] == [{"name": "overhang", "reason": "ValueError: boom"}]
    assert r["coverage"]["ran"] == [x for x in ALL if x != "overhang"] and r["coverage"]["not_run"] == []
    rec = by_id(r)["overhang"]
    assert rec["state"] == "unknown" and rec["error"] == "ValueError: boom" and "the check stopped" in rec["reason"]
    assert by_id(r)["wall_thickness"]["state"] == "pass"           # the other numbers stand
    assert r["summary"]["unknown"] >= 1 and "overhang" in r["summary"]["unknown_ids"]
    assert r["gate"]["state"] == "pass" and r["gate"]["exit_code"] == 0      # an unknown never sets the gate


def test_without_contain_errors_the_exception_passes_to_the_caller(tmp_path, monkeypatch):
    """The ratchet calls `check(strict=True)` and reads an exception as a failure of the tool: that stays."""
    monkeypatch.setattr(pc, "check_overhang", boom)
    with pytest.raises(ValueError, match="boom"):
        pc.check(save(plate(2.0), tmp_path / "p.stl"), strict=True)


def test_a_stage_that_stops_names_every_check_that_needs_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "measure_direction", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no rays")))
    r = pc.check(save(plate(2.0), tmp_path / "p.stl"), contain_errors=True)
    names = [x["name"] for x in r["coverage"]["failed"]]
    assert names == ["bed_fit", "overhang", "support_volume", "plate_contact", "wall_thickness"]
    assert {x["reason"] for x in r["coverage"]["failed"]} == {"build-direction numbers: RuntimeError: no rays"}
    assert r["coverage"]["ran"] == ["mesh_closed", "bodies", "closed_voids"] and r["partial"] is True
    assert r["input"]["triangles"] > 0 and r["input"]["total_area_mm2"] is None
    assert r["settings"]["build_direction_label"] == "+z"           # the settings do not need the failed stage
    assert "area not measured" in pc.format_text(r)


def test_a_failure_of_the_thickness_estimate_leaves_the_other_seven_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "thickness_estimate", boom)
    r = pc.check(save(plate(2.0), tmp_path / "p.stl"), contain_errors=True)
    assert r["coverage"]["failed"] == [{"name": "wall_thickness", "reason": "thickness estimate: ValueError: boom"}]
    assert r["coverage"]["ran"] == ALL[:-1]


def test_a_failure_of_the_mesh_health_names_the_checks_that_read_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "mesh_health", boom)
    r = pc.check(save(plate(2.0), tmp_path / "p.stl"), contain_errors=True)
    assert [x["name"] for x in r["coverage"]["failed"]] == ["mesh_closed", "bodies", "closed_voids", "wall_thickness"]
    assert r["coverage"]["ran"] == ["bed_fit", "overhang", "support_volume", "plate_contact"]
    assert r["input"]["triangles"] > 0 and r["input"]["total_area_mm2"] > 0      # the input section needs no number of the failed stage


def test_a_check_that_the_inputs_do_not_ask_for_is_not_run_and_the_report_is_not_partial(tmp_path):
    r = pc.check(save(plate(2.0), tmp_path / "p.stl"), print_settings={"printer_type": "sls"})
    assert r["partial"] is False and r["coverage"]["failed"] == []
    assert r["coverage"]["ran"] == [x for x in ALL if x != "support_volume"]
    [skipped] = r["coverage"]["not_run"]
    assert skipped["name"] == "support_volume" and skipped["asked"] is False and "needs no support" in skipped["reason"]
    assert pc.format_text(r).splitlines()[1] == "partial: no; 7 of 7 checks ran (" + ", ".join(r["coverage"]["ran"]) + "); 1 not asked"


def test_a_check_with_a_limit_that_cannot_decide_is_asked_and_not_run(tmp_path):
    """A limit above the chord cap: the check has a limit and gives no verdict. The gate rule stays: an unknown never sets it."""
    r = pc.check(save(plate(2.0), tmp_path / "p.stl"), print_settings={"min_wall_mm": 40.0}, strict=True)
    assert states(r)["wall_thickness"] == "unknown"
    assert r["partial"] is True and r["coverage"]["failed"] == []
    [x] = r["coverage"]["not_run"]
    assert x["name"] == "wall_thickness" and x["asked"] is True and "chord cap" in x["reason"]
    assert r["gate"]["state"] == "pass"


def test_the_text_has_the_line_near_the_top(tmp_path, monkeypatch):
    p = save(plate(2.0), tmp_path / "p.stl")
    clean = pc.format_text(pc.check(p)).splitlines()
    assert clean[0].startswith("print check: ") and clean[1] == "partial: no; 8 of 8 checks ran (" + ", ".join(ALL) + ")"
    monkeypatch.setattr(pc, "check_overhang", boom)
    broken = pc.format_text(pc.check(p, contain_errors=True)).splitlines()
    assert broken[1] == ("partial: 1 of 8 checks failed (overhang: ValueError: boom); ran: "
                         + ", ".join(x for x in ALL if x != "overhang"))


def test_the_command_contains_errors_and_keeps_the_exit_code(tmp_path, monkeypatch, capsys):
    """The report shows the other checks, and the exit code stays 2 (an internal error), with and without `--strict`."""
    p = save(plate(2.0), tmp_path / "p.stl")
    code, out, err = cli(capsys, p, "--strict")
    assert code == 0 and err == "" and out.splitlines()[1].startswith("partial: no; 8 of 8 checks ran")
    monkeypatch.setattr(pc, "check_support_volume", boom)
    for flags in (["--strict"], []):
        code, out, err = cli(capsys, p, *flags)
        assert code == 2
        assert err == "internal error: 1 check(s) stopped: support_volume; see `error` in their records\n"
        assert out.splitlines()[1].startswith("partial: 1 of 8 checks failed (support_volume: ValueError: boom)")
        assert "unknown  support_volume" in out
    code, out, err = cli(capsys, p, "--json")
    rep = json.loads(out)
    assert code == 2 and rep["partial"] is True and rep["coverage"]["failed"] == [{"name": "support_volume", "reason": "ValueError: boom"}]


def test_the_mcp_tool_contains_errors_too(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    p = save(plate(2.0), tmp_path / "p.stl")
    clean = server.print_check(path=str(p))
    assert clean["ok"] is True and clean["partial"] is False and clean["coverage"]["ran"] == ALL
    monkeypatch.setattr(pc, "check_plate_contact", boom)
    r = server.print_check(path=str(p))
    assert r["ok"] is True and r["partial"] is True
    assert r["coverage"]["failed"] == [{"name": "plate_contact", "reason": "ValueError: boom"}]
