"""Tests for the part card `result/report.md` (topopt/part_card.py, CONTRACTS.md section 9.2).

The fixtures in tests/fixtures/part_card/ are copies of real files: `run.json`, `metrics.json` and two rows of `history.jsonl` of a
real run of examples/hook/hook_short.yaml (CPU, 2026-10-23), and a `verify.json` of a real `topopt verify` run (another hook run,
cut down to the keys that the card reads). `older_build/` is the sample of a run from an older build (the files of an old recording of the hook; the mesh is empty).
"""
import json
import re
import shutil

import pytest

from topopt import case, part_card
from part_card_helpers import (FIX, HEADINGS, ROOT, STL_BYTES, add_pngs, add_verify, run_dir, section, sha, work)  # noqa: F401


# --- the sections ------------------------------------------------------------------------------

def test_every_section_with_data(run_dir, work):
    add_verify(run_dir)
    add_pngs(run_dir)
    (run_dir / "result" / "nominal.stl").write_bytes(STL_BYTES)
    case.record_brief(work / "hook_short.yaml", work, "A wall hook for a shirt of 200 g.",
                      assumptions=[{"text": "PLA, printed upright", "origin": "inferred"}],
                      criteria=[{"text": "Hangs 200 g", "origin": "user"}])
    text = part_card.build(run_dir)
    assert [m for m in re.findall(r"^## (.*)$", text, re.M)] == HEADINGS
    assert text.startswith("# Part card: tshirt_hook_short\n")
    part = section(text, "The part")
    assert "Problem file: hook_short.yaml" in part and "Started: 2026-10-23T21:28:18+00:00" in part
    assert "Run time: 59.6 s" in part and "solver pardiso" in part
    assert "The run stopped at its pass limit (pass 40 of 40)." in part      # state iteration_limit, in plain words
    assert "The objective had settled: no. Every constraint row is met: yes." in part
    mass = section(text, "Mass and volume")
    assert "89.1 g and 71856 mm3" in mass and "25.72 g and 20738.6 mm3" in mass and "71.1 %" in mass
    assert "Density used for the mass: 1.24 g/cm3" in mass and "20586 triangles, watertight: yes, method sdf" in mass
    assert "Machining stock" not in mass
    stiff = section(text, "Stiffness")
    assert "| hang | tip: [0, 0, -1.96] N (size 1.96 N) | 1294.703 | 0.002967 |" in stiff
    assert "| pull_off | tip: [20.89, 5.21, -20.89] N (size 29.9988 N) | 1316.773 | 0.6834 |" in stiff
    assert "59.08 at the start, 1.378 at the end (N mm)" in stiff
    brief = section(text, "The brief")
    assert "Requirement: A wall hook for a shirt of 200 g." in brief
    assert "PLA, printed upright (origin: inferred)" in brief and "Hangs 200 g (origin: user)" in brief
    prn = section(text, "Print facts")
    assert "printer fdm, build direction +z, bed 220 x 220 mm, self-supporting angle 45 degrees" in prn
    assert "Assumed, not in the YAML: build_direction, bed_mm, printer_type" in prn
    assert "Overhang area: 2413.743 mm2 (19.2 % of the surface)" in prn and "Thinnest member (an estimate): 2.818 mm" in prn
    ver = section(text, "Verify (independent check)")
    assert "Checked file: `final.stl`; mesh size 2.195 mm" in ver and "(same SHA-256)" in ver
    assert "| hang | 0.07019 | 0.4079 |" in ver and "Margin (allowable / p99.9)" in ver and "202.3" in ver
    assert "Your allowable stress: 14.2 MPa at the percentile 99.9; exceeded: no" in ver
    assert "None. No load force sits on fixed nodes." in section(text, "Lost force")
    img = section(text, "Images")
    assert "![final_montage.png](../renders/final_montage.png)" in img
    assert "[final_iso.png](../renders/final_iso.png)" in img and "[final_front.png](../renders/final_front.png)" in img
    files = section(text, "Files")
    assert "[result/final.stl](final.stl): the as-printed mesh in mm (with machining stock)" in files
    assert "[result/nominal.stl](nominal.stl)" in files and "[run.log]" not in files        # run.log is not in this folder
    assert "refine" not in files          # the refine page is made after the post step: the card does not list it
    assert "[verify/verify.json](../verify/verify.json)" in files
    nc = section(text, "Not checked")
    assert "Buckling: not run" in nc and "Fatigue: not checked" in nc and "The printed part: not checked" in nc
    assert "Stress" not in nc and "Independent check" not in nc          # verify ran: stress is checked
    src = section(text, "Sources")
    assert "Read: result/metrics.json, run.json, verify/verify.json, history.jsonl, hook_short.case.json" in src
    assert str(run_dir.parent.parent) not in text and not re.search(r"(^|[\s(`])/(app|home|Users|tmp)/", text)     # no absolute path


def test_every_section_with_no_data(tmp_path):
    d = tmp_path / "bare"
    (d / "result").mkdir(parents=True)
    (d / "result" / "metrics.json").write_text(json.dumps({"mass_final_g": 1.0}))
    text = part_card.build(d)
    assert [m for m in re.findall(r"^## (.*)$", text, re.M)] == HEADINGS
    assert text.startswith("# Part card: bare\n")        # no run.json: the folder name
    part = section(text, "The part")
    assert "Name: not known (run.json is missing)" in part and "Problem file: not known" in part
    assert "- Run time: not known\n" in part and "- Model: voxel size not known, solver not known" in part
    assert f"The verdict is {part_card.NOT_KNOWN}." in part
    mass = section(text, "Mass and volume")
    assert "(before): not known\n" in mass and "(after): 1 g and not known\n" in mass
    assert "not known at the start, not known at the end" in section(text, "Stiffness")
    assert section(text, "The brief").strip().startswith("- Brief: not known (run.json is missing")
    assert "Not run: `metrics.json` has no print block" in section(text, "Print facts")
    assert "Not run: the run folder has no `verify/verify.json`" in section(text, "Verify (independent check)")
    assert "None. The run has no stress row and no buckling result." in section(text, "Other checks that ran")
    assert "not known: `run.json` has no `lost_force`" in section(text, "Lost force")
    assert "No image yet. `render_part`" in section(text, "Images")
    assert "[result/metrics.json](metrics.json)" in section(text, "Files")
    nc = section(text, "Not checked")
    for word in ("Stress: not checked", "Buckling: not run", "Fatigue: not checked", "The printed part: not checked",
                 "Print limits: not applied", "Independent check of the stiffness and the stress on a tet mesh: not run"):
        assert word in nc
    assert "Not found: run.json, verdict.json, verify/verify.json, history.jsonl" in section(text, "Sources")


def test_no_brief_with_a_case_file_that_has_none_and_the_yaml_assumptions(run_dir, work):
    run = json.loads((run_dir / "run.json").read_text())
    run["assumptions"] = ["Shirt mass 0.2 kg, so 1.96 N down"]
    (run_dir / "run.json").write_text(json.dumps(run))
    brief = section(part_card.build(run_dir), "The brief")
    assert "No brief. There is no case file beside the problem YAML. `record_brief`" in brief
    assert "The YAML lists these assumptions (the origin is not recorded):" in brief and "Shirt mass 0.2 kg" in brief


def test_the_brief_says_when_the_run_started_with_another_brief(run_dir, work):
    yaml = work / "hook_short.yaml"
    case.record_brief(yaml, work, "First text.")
    case.record_brief(yaml, work, "Second text.", assumptions=[{"text": "a", "origin": "default"}])
    manifest = case.manifest_path(yaml)
    doc = json.loads(manifest.read_text())
    doc["runs"].append({"run_dir": "runs/hook", "started_at": "2026-10-23T21:28:06+00:00", "pid": 1, "brief": 1, "inputs": {}})
    manifest.write_text(json.dumps(doc))
    brief = section(part_card.build(run_dir), "The brief")
    assert "Requirement: Second text." in brief and "This run started with brief 1, not with this one." in brief
    assert "a (origin: default)" in brief and "Criteria: none recorded" in brief
    doc["runs"][-1]["brief"] = None
    manifest.write_text(json.dumps(doc))
    assert "This run started without a brief." in section(part_card.build(run_dir), "The brief")


def test_the_verdict_in_plain_words_for_each_state(run_dir):
    metrics_file = run_dir / "result" / "metrics.json"
    metrics = json.loads(metrics_file.read_text())
    said = {}
    for state, extra in (("converged", {"converged": True, "feasible": True}), ("stopped_by_rule", {}),
                         ("stopped_by_request", {}), ("unfinished", {}), ("surprise", {})):
        metrics["verdict"] = {**metrics["verdict"], "state": state, **extra}
        metrics_file.write_text(json.dumps(metrics))
        said[state] = section(part_card.build(run_dir), "The part").split("- Result: ")[1].split("\n")[0]
    assert said["converged"].startswith("The run converged at the last pass (pass 40 of 40)")
    assert said["stopped_by_rule"].startswith("A stop rule ended the run before its pass limit")
    assert said["stopped_by_request"].startswith("A request to stop ended the run")
    assert said["unfinished"].startswith("The run did not reach its last pass")
    assert said["surprise"] == f"The verdict is {part_card.NOT_KNOWN}."


def test_the_numbers_of_the_card_are_the_numbers_of_metrics_json(run_dir):
    m = json.loads((run_dir / "result" / "metrics.json").read_text())
    text = part_card.build(run_dir)
    for key in ("mass_design_space_g", "mass_final_g", "mass_saved_pct", "volume_final_mm3", "run_time_s", "triangles"):
        assert f"{m[key]:.10g}" in text, key
    for s in m["stiffness_cases"]:
        assert f"{s['N_per_mm']:.10g}" in text
    assert f"{m['print']['thinnest_member']['thinnest_member_mm']:.10g}" in text


def test_a_null_stiffness_shows_its_reason(run_dir):
    f = run_dir / "result" / "metrics.json"
    m = json.loads(f.read_text())
    m["stiffness_cases"][0] = {"name": "hang", "N_per_mm": None, "reason": "the case has loads of kind moment"}
    f.write_text(json.dumps(m))
    assert "not known (the case has loads of kind moment)" in section(part_card.build(run_dir), "Stiffness")


def test_a_stress_row_a_buckling_result_and_the_lost_force(run_dir):
    f = run_dir / "result" / "metrics.json"
    m = json.loads(f.read_text())
    m["stress"] = {"cases": {"hang": {"summary": "relaxed row met; voxel peak 0.9 MPa"}}}
    m["buckling"] = {"status": "done", "lowest_load_factor": {"case": "pull_off", "value": 120.9},
                     "cases": [{"name": "hang", "load_factors": [1276.37, 1604.65]}, {"name": "pull_off", "load_factors": [120.9]}]}
    f.write_text(json.dumps(m))
    run = json.loads((run_dir / "run.json").read_text())
    run["lost_force"] = {"cases": [{"name": "hang", "total_magnitude_n": 1.96, "lost_magnitude_n": 0.49, "lost_percent": 25.0}]}
    (run_dir / "run.json").write_text(json.dumps(run))
    text = part_card.build(run_dir)
    other = section(text, "Other checks that ran")
    assert "Stress row of the optimizer (voxel estimates, not a tet check)" in other and "hang: relaxed row met; voxel peak 0.9 MPa" in other
    assert "lowest load factor 120.9 (case pull_off); lowest of each case: hang 1276.37, pull_off 120.9" in other
    assert "Case hang: 0.49 N of 1.96 N is lost" in section(text, "Lost force")
    nc = section(text, "Not checked")
    assert "Buckling" not in nc and "Stress on a tet mesh: not checked. Only the stress row" in nc and "Stress: not checked" not in nc
    m["buckling"] = {"status": "skipped", "reason": "solver_not_direct", "facts": ["the solver is amg"]}
    f.write_text(json.dumps(m))
    text = part_card.build(run_dir)
    assert "Buckling check: skipped: the solver is amg" in section(text, "Other checks that ran")
    assert "Buckling: no load factors, the check is skipped (see the section Other checks that ran)." in section(text, "Not checked")


def test_a_force_with_a_moment_and_a_pressure_in_a_case(run_dir):
    run = json.loads((run_dir / "run.json").read_text())
    run["load_cases"][0]["forces"] = [{"region": "tip", "force": [0, 0, 0], "kind": "moment", "moment": [0, 5, 0]},
                                      {"region": "face", "kind": "pressure", "pressure_mpa": 0.5, "area_mm2": 12.0}]
    (run_dir / "run.json").write_text(json.dumps(run))
    row = [l for l in section(part_card.build(run_dir), "Stiffness").splitlines() if l.startswith("| hang")][0]
    assert "tip: [0, 0, 0] N (moment, moment [0, 5, 0] N mm); face: pressure 0.5 MPa on 12 mm2" in row


def test_the_print_report_that_failed_and_a_printer_that_needs_no_support(run_dir):
    f = run_dir / "result" / "metrics.json"
    m = json.loads(f.read_text())
    m["print"] = {"ok": False, "error": "RuntimeError: boom"}
    f.write_text(json.dumps(m))
    assert "The print report failed: RuntimeError: boom." in section(part_card.build(run_dir), "Print facts")
    m["print"] = json.loads((FIX / "metrics.json").read_text())["print"]
    m["print"]["support_needed"] = False
    m["print"]["support_note"] = "sls: no support needed"
    f.write_text(json.dumps(m))
    assert "Support: not needed for this printer type (sls: no support needed)" in section(part_card.build(run_dir), "Print facts")


# --- the verify facts ----------------------------------------------------------------------------

def test_verify_of_another_mesh_is_named(run_dir):
    add_verify(run_dir, stl_sha="0" * 64)
    assert "is NOT of the mesh now in `result/final.stl` (another SHA-256): run verify again." in section(
        part_card.build(run_dir), "Verify (independent check)")


def test_verify_without_an_allowable_gives_no_pass_or_fail(run_dir):
    v = add_verify(run_dir)
    del v["limit"]
    (run_dir / "verify" / "verify.json").write_text(json.dumps(v))
    ver = section(part_card.build(run_dir), "Verify (independent check)")
    assert "No allowable stress was given, so there is no pass or fail." in ver and "Margin" not in ver


def test_verify_that_holds_no_result_or_cannot_be_read(run_dir):
    add_verify(run_dir, estimate_only=True)
    assert "holds no result (ok: yes, estimate only: yes)" in section(part_card.build(run_dir), "Verify (independent check)")
    assert "Independent check of the stiffness and the stress on a tet mesh: not run" in section(part_card.build(run_dir), "Not checked")
    (run_dir / "verify" / "verify.json").write_text("{not json")
    text = part_card.build(run_dir)
    assert "`verify/verify.json` is there and cannot be read." in section(text, "Verify (independent check)")
    assert "Not readable: verify/verify.json" in section(text, "Sources")


def test_verify_of_part_only_names_that_file(run_dir):
    (run_dir / "result" / "part_only.stl").write_bytes(b"solid p\nendsolid p\n")
    v = add_verify(run_dir, stl_sha=sha(b"solid p\nendsolid p\n"))
    v["run"]["verified_file"] = "part_only.stl"
    v["run"]["final_stl_sha256"] = sha(b"solid p\nendsolid p\n")
    (run_dir / "verify" / "verify.json").write_text(json.dumps(v))
    ver = section(part_card.build(run_dir), "Verify (independent check)")
    assert "Checked file: `part_only.stl`" in ver and "mesh now in `result/part_only.stl` (same SHA-256)" in ver


# --- images, files ----------------------------------------------------------------------------------

def test_images_are_listed_in_name_order_and_the_montage_is_shown(run_dir):
    add_pngs(run_dir)
    (run_dir / "renders" / "a b.png").write_bytes(b"\x89PNG")
    lines = [l for l in section(part_card.build(run_dir), "Images").splitlines() if l.startswith("- ")]
    assert lines[:4] == ["- [a b.png](../renders/a%20b.png)", "- [final_front.png](../renders/final_front.png)",
                         "- [final_iso.png](../renders/final_iso.png)", "- ![final_montage.png](../renders/final_montage.png)"]


def test_an_image_older_than_the_final_mesh_may_be_from_an_earlier_result(run_dir):
    import os
    add_pngs(run_dir)
    stl = run_dir / "result" / "final.stl"
    old = run_dir / "renders" / "final_front.png"
    t = stl.stat().st_mtime_ns
    os.utime(old, ns=(t - 5_000_000_000, t - 5_000_000_000))        # made before the final mesh
    os.utime(run_dir / "renders" / "final_iso.png", ns=(t, t))      # the same time is not older
    os.utime(run_dir / "renders" / "final_montage.png", ns=(t + 1, t + 1))
    lines = [l for l in section(part_card.build(run_dir), "Images").splitlines() if l.startswith("- ")]
    note = " (may be from an earlier result: the file is older than `result/final.stl`)"
    assert lines[:3] == ["- [final_front.png](../renders/final_front.png)" + note, "- [final_iso.png](../renders/final_iso.png)",
                         "- ![final_montage.png](../renders/final_montage.png)"]
    stl.unlink()                                                      # no final mesh to compare with: no note
    assert "may be from" not in section(part_card.build(run_dir), "Images")


def test_the_design_space_line_names_the_other_active_regions(run_dir):
    mass = section(part_card.build(run_dir), "Mass and volume")
    assert "- The design space and its other active regions (fixed, load, solid), all solid (before): 89.1 g and 71856 mm3" in mass


# --- write: deterministic, untouched when equal, refreshed ------------------------------------------

def test_the_same_folder_gives_the_same_bytes(run_dir):
    add_verify(run_dir)
    first = part_card.write(run_dir)
    data = first.read_bytes()
    stamp = first.stat().st_mtime_ns
    assert part_card.build(run_dir).encode() == data
    assert part_card.write(run_dir) == first and first.read_bytes() == data
    assert first.stat().st_mtime_ns == stamp          # equal bytes: the file is not written again
    copy = run_dir.parent / "copy"
    shutil.copytree(run_dir, copy)
    assert part_card.build(copy).replace("# Part card: tshirt_hook_short", "") == data.decode().replace("# Part card: tshirt_hook_short", "")
    assert not list((run_dir / "result").glob("*.tmp"))


def test_the_second_write_has_the_facts_that_came_later(run_dir):
    path = part_card.write(run_dir)
    before = path.read_text()
    assert "Not run: the run folder has no `verify/verify.json`" in before and "No image yet." in before
    add_verify(run_dir)
    add_pngs(run_dir)
    (run_dir / "result" / "final.step").write_bytes(b"ISO-10303-21;")
    part_card.write(run_dir)
    after = path.read_text()
    assert after != before
    assert "Checked file: `final.stl`" in after and "../renders/final_montage.png" in after and "[result/final.step](final.step)" in after
    assert "Not run: the run folder has no" not in after


def test_a_folder_with_no_result_has_no_card(tmp_path):
    with pytest.raises(part_card.NoResult):
        part_card.build(tmp_path)
    with pytest.raises(part_card.NoResult):
        part_card.write(tmp_path)
    assert part_card.write_logged(tmp_path) is None


def test_a_section_that_fails_does_not_stop_the_card(run_dir, monkeypatch):
    def boom(f):
        raise KeyError("x")
    monkeypatch.setattr(part_card, "SECTIONS", tuple((t, boom if t == "Print facts" else s) for t, s in part_card.SECTIONS))
    text = part_card.build(run_dir)
    assert "This section could not be written: KeyError: 'x'" in section(text, "Print facts")
    assert "25.72 g" in text


def test_a_card_that_cannot_be_written_is_one_log_line(run_dir):
    (run_dir / "result" / part_card.CARD_FILE).mkdir()          # a folder in the place of the file
    lines = []
    assert part_card.write_logged(run_dir, lines.append) is None
    assert len(lines) == 1 and lines[0].startswith("warning: part card not written: ")
    with pytest.raises(OSError):
        part_card.write(run_dir)


# --- the old recording ---------------------------------------------------------------------------------

def test_a_recorded_run_of_an_older_build_gives_a_card_with_not_known(tmp_path):
    rec = FIX / "older_build"
    text = part_card.build(rec)             # the folder is read only: build writes nothing
    m = json.loads((rec / "result" / "metrics.json").read_text())
    assert f"{m['mass_final_g']:.10g} g" in text and f"{m['mass_design_space_g']:.10g} g" in text
    assert "Name: tshirt_hook" in section(text, "The part") and f"The verdict is {part_card.NOT_KNOWN}." in section(text, "The part")
    assert "Not run: `metrics.json` has no print block" in section(text, "Print facts")
    assert "| hang | tip: [0, 0, -1.96] N | 1260.501 |" in section(text, "Stiffness")
    assert "[result/final.stl](final.stl)" in section(text, "Files")
    assert not (rec / "result" / "report.md").exists()


def test_a_run_that_asked_for_a_mass_has_the_fact_line_of_volume_basis(run_dir):
    from topopt.volume_basis import mass_asked_line
    assert "asked (settings.target_mass_g)" not in section(part_card.build(run_dir), "Mass and volume")   # volume_fraction: no line
    p = run_dir / "result" / "metrics.json"
    m = json.loads(p.read_text())
    m["mass_asked_g"] = 30.0
    p.write_text(json.dumps(m))
    mass = section(part_card.build(run_dir), "Mass and volume")
    line = mass_asked_line(m)
    assert line and line.startswith("fact: final mesh 25.72 g against 30 g asked")
    assert f"- {line}" in mass                            # the same text as the run log and run_status


def test_a_mesh_fallback_is_a_line_of_the_card_and_a_normal_run_has_none(run_dir):
    """#445: `mesh.fallbacks` of metrics.json is a line under "Other checks that ran"."""
    f = run_dir / "result" / "metrics.json"
    m = json.loads(f.read_text())
    assert "Mesh fallback" not in part_card.build(run_dir)
    m["mesh"]["fallbacks"] = []
    f.write_text(json.dumps(m))
    assert "Mesh fallback" not in part_card.build(run_dir)
    m["mesh"]["fallbacks"] = [{"pass": 1, "round": 1, "build": "calibration", "step": "snap", "error_type": "ValueError", "error": "the snap broke here", "kept": False},
                              {"pass": 2, "round": 3, "build": "calibration", "step": "plane_clip", "error_type": "KeyError", "error": "x", "kept": True},
                              {"pass": 2, "round": None, "build": "without_move", "step": "exact", "error_type": "ValueError", "error": "y", "kept": False}]
    f.write_text(json.dumps(m))
    other = section(part_card.build(run_dir), "Other checks that ran")
    assert "- Mesh fallback: the mesh step took a fallback in round 1 of pass 1, step snap: ValueError: the snap broke here. The part can differ" in other
    assert "round 3 of pass 2 (the kept round), step plane_clip: KeyError: x." in other
    assert "in the build without the move, step exact: ValueError: y." in other
    assert other.count("Option: check the part with `topopt verify` (MCP: verify_run), or report the error in an issue.") == 3


def test_the_same_fallback_in_several_rounds_is_one_card_line():
    """#445 review F5: a step that fails with the same error in each round of a pass is one line; another error, step or pass is a line of its own."""
    def e(p, k, step="snap", error="no snap", kept=False):
        return {"pass": p, "round": k, "build": "calibration", "step": step, "error_type": "ValueError", "error": error, "kept": kept}
    lines = part_card.mesh_fallback_lines({"mesh": {"fallbacks": [e(1, 0), e(1, 1), e(1, 1, "flush"), e(1, 2), e(2, 0), e(2, 1, error="other"),
                                                                    e(2, 2, kept=True)]}})
    assert [x.split(", step ")[0] for x in lines] == ["the mesh step took a fallback in rounds 0, 1, 2 of pass 1", "the mesh step took a fallback in round 1 of pass 1",
                                                      "the mesh step took a fallback in rounds 0, 2 of pass 2 (2 is the kept round)",
                                                      "the mesh step took a fallback in round 1 of pass 2"]
    assert [x.split(", step ")[1].split(".")[0] for x in lines] == ["snap: ValueError: no snap", "flush: ValueError: no snap", "snap: ValueError: no snap",
                                                                    "snap: ValueError: other"]
