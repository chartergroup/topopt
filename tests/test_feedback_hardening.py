"""Tests for the round-1 fixes of topopt/feedback.py: write guards, overlapping hints, what compare can see,
the short reply, the history limit (CONTRACTS.md section 21). They use the helpers of test_feedback.py.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import feedback  # noqa: E402
from topopt.config import load_config  # noqa: E402
from topopt.voxelize import DESIGN, KEEP_OUT, LOAD, inside_mask, voxelize  # noqa: E402
from test_feedback import (box, fake_run, make_case, mark, marks_doc, region_mesh, tree_hash)  # noqa: E402


@pytest.fixture
def case(tmp_path):
    ypath = make_case(tmp_path / "in")
    run = fake_run(tmp_path / "runs" / "a", ypath)
    return type("Case", (), {"root": tmp_path, "yaml": ypath, "run": run, "inp": ypath.parent})


def call(case, *marks, out_dir=None, **kw):
    f = case.root / "marks_h.json"
    f.write_text(json.dumps(marks_doc(*marks)))
    return feedback.make_hints(kw.pop("run", case.run), f, out_dir, kw.pop("options", None), **kw)


M1 = mark(1, "remove", [(10, 3, 6)], radius=2.0)


# ---- M1: where the tool writes ----------------------------------------------------------------------
def test_out_dir_refusals_change_no_file_anywhere(case):
    root = case.root
    other = fake_run(root / "runs" / "other", case.yaml)                       # another run folder
    victim = root / "examples" / "hook"                                        # another problem, same file names
    shutil.copytree(case.inp, victim)
    (victim / "bracket.yaml").write_text((victim / "bracket.yaml").read_text().replace("iterations: 12", "iterations: 99"))
    stl_case = root / "stl_case"                                               # region STLs in a sub-folder
    (stl_case / "stl").mkdir(parents=True)
    for f in ("design.stl", "plate.stl", "tip.stl"):
        shutil.copy(case.inp / f, stl_case / "stl" / f)
    doc = yaml.safe_load(case.yaml.read_text())
    doc["regions"] = {"design": "stl/design.stl", "fixed": ["stl/plate.stl"], "load": {"tip": "stl/tip.stl"}}
    (stl_case / "bracket.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    run_stl = fake_run(root / "runs" / "r_stl", stl_case / "bracket.yaml")
    afile = root / "a_file"
    afile.write_text("x")
    link = root / "a_link"
    link.symlink_to(root / "empty_target", target_is_directory=True)
    (root / "empty_target").mkdir()
    watched = [case.inp, case.run, other, victim, stl_case, run_stl, afile, root / "empty_target"]
    before = tree_hash(*watched)
    cases = [
        (case.run, other, "run_folder"),                                       # another run folder
        (case.run, other / "sub_hints", "run_folder"),                         # a new folder inside another run
        (case.run, case.run / "hints", "run_folder"),                          # inside the run itself
        (case.run, victim, None),                                              # another problem's folder
        (run_stl, stl_case / "stl", "input_folder"),                           # the STL folder of the YAML
        (case.run, case.inp, "input_folder"),                                  # the folder of the YAML
        (case.run, afile, "not_a_folder"),
        (case.run, link, "symbolic_link"),                                     # a symbolic link as out_dir
    ]
    for run, out, reason in cases:
        for overwrite in (False, True):
            r = call(case, M1, out_dir=out, run=run, overwrite=overwrite)
            assert r["ok"] is False and "internal error" not in r["error"], (out, overwrite, r)
            assert r["refused"]["path"] and r["refused"]["options"], r
            if reason:
                assert r["refused"]["reason"] == reason, (out, overwrite, r["refused"])
            else:                                                              # a folder with files, not a hints folder
                assert r["refused"]["reason"] == ("not_a_hints_folder" if overwrite else "not_empty")
    assert tree_hash(*watched) == before and not (root / "runs" / "other" / "sub_hints").exists()
    assert "iterations: 99" in (victim / "bracket.yaml").read_text()


def test_overwrite_takes_only_an_earlier_hints_folder(case):
    out = case.root / "h"
    assert call(case, M1, out_dir=out)["ok"]
    (out / "stray.txt").write_text("old")
    r = call(case, M1, out_dir=out, overwrite=True)                            # an earlier hints folder: accepted
    assert r["ok"] and any("stray.txt" in n for n in r["notes"])               # a file that stays is a fact
    assert (out / "stray.txt").read_text() == "old"                            # and it is not deleted
    data = json.loads((out / "hints.json").read_text())
    data["format"] = 2                                                         # another format: not an earlier folder
    (out / "hints.json").write_text(json.dumps(data))
    r = call(case, M1, out_dir=out, overwrite=True)
    assert not r["ok"] and r["refused"]["reason"] == "not_a_hints_folder"
    empty = case.root / "empty"
    empty.mkdir()
    assert call(case, M1, out_dir=empty)["ok"]                                 # an empty folder is fine


def test_symbolic_link_at_a_target_is_refused_not_followed(case):
    out = case.root / "h"
    assert call(case, M1, out_dir=out)["ok"]
    precious = case.root / "precious.stl"
    precious.write_bytes(b"precious data")
    (out / "design.stl").unlink()
    (out / "design.stl").symlink_to(precious)
    before = tree_hash(case.inp, case.run, precious)
    r = call(case, M1, out_dir=out, overwrite=True)
    assert not r["ok"] and r["refused"]["reason"] == "symbolic_link" and r["refused"]["path"].endswith("design.stl")
    assert "symbolic link" in r["error"]
    assert precious.read_bytes() == b"precious data" and (out / "design.stl").is_symlink()   # not followed, not removed
    assert tree_hash(case.inp, case.run, precious) == before
    (out / "design.stl").unlink()
    shutil.copy(case.inp / "design.stl", out / "design.stl")
    (out / "hints.json").unlink()
    (out / "hints.json").symlink_to(case.root / "nowhere.json")               # a dangling link at hints.json
    (out / "hints2.json").write_text(json.dumps({"format": 1}))              # (an earlier folder needs a hints.json)
    r = call(case, M1, out_dir=out, overwrite=True)
    assert not r["ok"] and r["refused"]["reason"] in ("symbolic_link", "not_a_hints_folder")
    assert not (case.root / "nowhere.json").exists()


def test_out_dir_is_checked_before_the_voxelization(case, monkeypatch):
    import topopt.voxelize as vox

    def boom(*a, **k):
        raise AssertionError("voxelize was called")
    monkeypatch.setattr(vox, "voxelize", boom)
    r = call(case, M1, out_dir=case.inp)
    assert r["refused"]["reason"] == "input_folder"                            # a refusal, not the AssertionError
    r = call(case, M1, out_dir=case.run / "x")
    assert r["refused"]["reason"] == "run_folder"
    r = call(case, M1, out_dir=case.root / "fine")                             # a good folder gets to the voxelization
    assert "voxelize was called" in r["error"]


def test_a_write_error_is_a_user_error_with_the_cause(case, monkeypatch):
    def same(*a, **k):
        raise shutil.SameFileError("same file")
    monkeypatch.setattr(shutil, "copyfile", same)
    r = call(case, M1, out_dir=case.root / "h")
    assert not r["ok"] and "cannot write" in r["error"] and "SameFileError" in r["error"]
    assert "internal error" not in r["error"]


def test_unique_name_ignores_case():
    assert feedback._unique_name("Design.stl", {"design.stl"}) == "Design_2.stl"
    assert feedback._unique_name("a.stl", {"A.STL", "a_2.stl"}) == "a_3.stl"
    assert feedback._unique_name("b.stl", {"a.stl"}) == "b.stl"


# ---- M2: marks that overlap -------------------------------------------------------------------------
def labels_of(path):
    o, h, s, lab, _ = voxelize(load_config(path))
    return o, h, s, lab


def test_a_mark_inside_an_earlier_add_mark_has_no_effect_and_the_yaml_loads(case):
    big = mark(1, "add", [(14, 0, 6)], radius=3.0)
    small = mark(2, "add", [(14.5, 0, 6)], radius=1.5)
    r = call(case, big, small, out_dir=case.root / "two")
    assert r["ok"] and r["proposed_yaml"]["loads"] is True and r["problems"] == []
    e1, e2 = r["marks"]
    assert (e1["status"], e2["status"]) == ("applied", "no_effect")
    assert "earlier marks 1" in e2["reason"] and "first mark wins" in e2["reason"]
    assert e2["facts"]["overlap_earlier_hints"]["marks"] == [1]
    assert r["summary"] == {"add:applied": 1, "add:no_effect": 1, "proposed_yaml": "loads"}
    solo = call(case, big, out_dir=case.root / "solo")
    assert np.array_equal(labels_of(r["proposed_yaml"]["path"])[3], labels_of(solo["proposed_yaml"]["path"])[3])
    assert [lr.name for lr in load_config(r["proposed_yaml"]["path"]).regions.load] == ["tip", "hint_add_1"]


def test_a_remove_mark_and_an_add_mark_at_one_place_the_first_wins(case):
    at = (14, 0, 6)
    r = call(case, mark(1, "remove", [at], radius=3.0), mark(2, "add", [at], radius=2.0), out_dir=case.root / "ra")
    assert r["ok"] and [e["status"] for e in r["marks"]] == ["applied", "no_effect"]
    assert r["marks"][1]["facts"]["overlap_earlier_hints"]["marks"] == [1] and r["problems"] == []
    # the other order: the add region is solid, the remove region is cut around it
    r2 = call(case, mark(1, "add", [at], radius=2.0), mark(2, "remove", [at], radius=3.0), out_dir=case.root / "ar")
    assert r2["ok"] and [e["status"] for e in r2["marks"]] == ["applied", "applied"]
    f2 = r2["marks"][1]["facts"]
    assert f2["overlap_earlier_hints"]["marks"] == [1] and f2["overlap_earlier_hints"]["voxels"] > 0
    assert "the regions of the earlier marks 1" in f2["applied_region"]["cut_to"][-1]
    o, h, s, lab = labels_of(r2["proposed_yaml"]["path"])
    solid = inside_mask(region_mesh(r2, r2["marks"][0]), o, h, s)
    assert (lab[solid] == LOAD).all() and (lab == KEEP_OUT).sum() == f2["applied_region"]["design_voxels"]


def test_marks_that_overlap_in_part_are_cut_and_counted_on_the_labels_that_hold_the_earlier_ones(case):
    a = mark(1, "add", [(12, 0, 6)], radius=3.0)
    b = mark(2, "add", [(15, 0, 6)], radius=3.0)
    c = mark(3, "remove", [(13.5, 0, 6)], radius=4.0)
    r = call(case, a, b, c, out_dir=case.root / "three")
    assert r["ok"] and [e["status"] for e in r["marks"]] == ["applied"] * 3
    e1, e2, e3 = r["marks"]
    assert e2["facts"]["overlap_earlier_hints"]["marks"] == [1] and e3["facts"]["overlap_earlier_hints"]["marks"] == [1, 2]
    assert e2["facts"]["overlap_earlier_hints"]["voxels"] > 0
    counted = sum(e["facts"]["applied_region"]["design_voxels"] for e in r["marks"])
    assert counted == r["labels_changed"]["voxels"]                            # no voxel is counted twice
    o, h, s, lab0 = labels_of(case.yaml)
    _, _, _, lab1 = labels_of(r["proposed_yaml"]["path"])
    assert ((lab0 == DESIGN) & (lab1 == LOAD)).sum() == (e1["facts"]["applied_region"]["design_voxels"]
                                                          + e2["facts"]["applied_region"]["design_voxels"])


def test_a_region_that_the_loader_stops_on_gives_ok_false_and_names_the_mark(case):
    """A keep-out over the whole load region (protect_regions off): the loader stops. It is a fact, not a crash."""
    tip = mark(1, "remove", [(22.5, 0, 2)], radius=5.0)
    r = call(case, tip, mark(2, "remove", [(8, 3, 6)], radius=1.5), out_dir=case.root / "stop",
             options={"protect_regions": False})
    assert r["ok"] is False and r["proposed_yaml"]["loads"] is False
    assert "load region 'tip' has no voxel left" in r["proposed_yaml"]["error"] and r["error"].startswith("the proposed YAML")
    assert any("does not load" in p for p in r["problems"]) and r["summary"]["proposed_yaml"].startswith("does not load")
    e1, e2 = r["marks"]
    assert e1["status"] == "invalid" and "loader stops" in e1["reason"] and e1["loader_error"]
    assert e2["status"] == "applied"                                           # the other mark is not blamed
    on_disk = json.loads((Path(r["out_dir"]) / "hints.json").read_text())
    assert on_disk["ok"] is False and on_disk["marks"][0]["status"] == "invalid"
    assert feedback.short_report(r)["ok"] is False and feedback.short_report(r)["error"]


# ---- M3: what compare can see -----------------------------------------------------------------------
def edit_yaml(path, fn):
    doc = yaml.safe_load(path.read_text())
    fn(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    old = 1_700_000_000
    os.utime(path, (old, old))                                                 # an old mtime: the newer check is blind


@pytest.mark.parametrize("key,fn,seen", [
    ("settings.filter_radius", lambda d: d["settings"].__setitem__("filter_radius", 3.0), True),
    ("material.density", lambda d: d["material"].__setitem__("density", 2.0), True),
    ("settings.case_combine", lambda d: d["settings"].__setitem__("case_combine", "max"), True),
    ("material.E", lambda d: d["material"].__setitem__("E", 7000.0), False),
    ("settings.penal", lambda d: d["settings"].__setitem__("penal", 4.0), False),
    ("settings.move_limit", lambda d: d["settings"].__setitem__("move_limit", 0.05), False),
])
def test_compare_sees_the_values_that_run_json_records_and_says_what_it_does_not_record(tmp_path, key, fn, seen):
    doc = {"material": {"E": 3500.0, "density": 1.24}}
    ypath = make_case(tmp_path / "in")
    d = yaml.safe_load(ypath.read_text())
    d.update(doc)
    ypath.write_text(yaml.safe_dump(d, sort_keys=False))
    run = fake_run(tmp_path / "runs" / "a", ypath)
    edit_yaml(ypath, fn)
    r = feedback.compare_runs(run, run)
    chk = r["inputs"]["a"]
    keys = [x["key"] for x in chk["yaml_vs_run_json"]["differs"]]
    assert (key in keys) is seen, (key, keys)
    facts = " ".join(chk["facts"])
    assert "does not record material.E, material.nu, settings.penal, settings.beta_schedule, settings.move_limit" in facts
    if seen:
        assert any(key in w for w in chk["warnings"]) and any(key in s for s in r["summary"])
        assert "Compared with run.json, equal:" in facts or key in " ".join(chk["warnings"])
    assert chk["yaml_vs_run_json"]["not_recorded"] == list(feedback.NOT_IN_RUN_JSON)
    assert any("does not record material.E" in n for n in r["notes"])           # also a note of the compare result
    # the filter radius is compared only when the YAML sets it (round 2)
    assert ("settings.filter_radius" in chk["yaml_vs_run_json"]["compared"]) is (key == "settings.filter_radius")


def test_compare_does_not_call_the_defaults_a_difference(tmp_path):
    ypath = make_case(tmp_path / "in")                                         # no filter_radius, no material key
    run = fake_run(tmp_path / "runs" / "a", ypath)
    r = feedback.compare_runs(run, run)
    assert r["inputs"]["a"]["yaml_vs_run_json"]["differs"] == []
    assert "settings.filter_radius" not in r["inputs"]["a"]["yaml_vs_run_json"]["compared"]   # derived, not compared
    assert "material.density" in r["inputs"]["a"]["yaml_vs_run_json"]["compared"]


def test_compare_target_volume_iterations_and_noise(tmp_path):
    ya = make_case(tmp_path / "in_a", iterations=12)
    yb = make_case(tmp_path / "in_b", iterations=20)
    box(tmp_path / "in_b" / "hole.stl", (8, -1, 4), (10, 1, 8))
    edit_yaml(yb, lambda d: d["regions"].__setitem__("keep_out", ["hole.stl"]))
    os.utime(yb, None)
    a, b = fake_run(tmp_path / "runs" / "a", ya, compliance=1.0), fake_run(tmp_path / "runs" / "b", yb, compliance=1.0 + 1e-13)
    # nit 4: lost_force is derived data, not a setup key
    rj = json.loads((b / "run.json").read_text())
    rj["lost_force"] = {"max_lost_percent": 3.0}
    (b / "run.json").write_text(json.dumps(rj))
    r = feedback.compare_runs(a, b)
    diff_keys = [x["key"] for x in r["setup"]["run_json_diff"]]
    assert "derived.volume.target_volume_mm3" in diff_keys and "derived.volume.target_mass_g" in diff_keys
    assert not any(k.startswith("lost_force") for k in diff_keys)
    assert any("target volumes differ" in n for n in r["notes"]) and any("iteration counts differ" in n for n in r["notes"])
    # nit 3: a change of 1e-13 percent is not a summary line
    assert not any(s.startswith("compliance_final:") for s in r["summary"]), r["summary"]
    assert not any("(-0.0 %)" in s or "(+0.0 %)" in s for s in r["summary"])


# ---- m1, m2: the report on disk and the short reply --------------------------------------------------
def strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)
    elif isinstance(obj, str):
        yield obj


def test_hints_json_holds_no_absolute_path_but_the_reply_does(case):
    marks_file = case.root / "m.json"
    marks_file.write_text(json.dumps(marks_doc(M1, mark(2, "remove", [(1, 0, 6)], radius=3.0))))
    out = case.root / "h"
    r = feedback.make_hints(case.run, marks_file, out)
    text = (out / "hints.json").read_text()
    on_disk = json.loads(text)
    assert str(case.root) not in text and not any(s.startswith("/") for s in strings(on_disk))
    assert r["out_dir"] == str(out) and r["run_dir"] == str(case.run)         # the reply has them
    assert on_disk["out_dir"] == "." and (out / on_disk["run_dir"]).resolve() == case.run
    assert on_disk["marks_file"] == "m.json" and len(on_disk["marks_sha256"]) == 64
    assert on_disk["source"]["yaml_path"] == "bracket.yaml" and on_disk["proposed_yaml"]["path"] == "bracket.yaml"
    assert on_disk["source"]["yaml_sha256"] == r["source"]["yaml_sha256"]
    assert on_disk["check"]["yaml"] == "bracket.yaml"
    assert any(("cut" in str(m) or "plate" in str(m)) for m in on_disk["marks"][1]["facts"]["overlap_regions"]["fixed"])


def many_marks(n):
    rng = np.random.default_rng(3)
    out = []
    for i in range(n):
        x, y, z = rng.uniform(2, 22), rng.uniform(-5, 5), rng.uniform(1, 11)
        pts = [(x, y, z)] if i % 2 == 0 else [(x + 0.5 * k, y, z) for k in range(12)]
        out.append(mark(i + 1, ("remove", "add", "smooth")[i % 3], pts, radius=1.0, mirror=bool(i % 4 == 0)))
    return out


def test_short_reply_is_small_and_the_full_report_is_on_disk(case):
    r3 = call(case, M1, mark(2, "add", [(14, -3, 8)], radius=1.5), mark(3, "smooth", [(8, 0, 12)]),
              out_dir=case.root / "h3")
    s3 = feedback.short_report(r3)
    assert s3["detail"] is False and s3["report_file"] == "hints.json" and "option_catalog" not in s3
    assert [m["id"] for m in s3["marks"]] == [1, 2, 3] and s3["marks"][0]["status"] == "applied"
    assert {"voxels_touched", "design_voxels", "region_file"} <= set(s3["marks"][0]) and "facts" not in s3["marks"][0]
    assert s3["check"]["time_estimate"]["solver"] and "warnings" in s3["check"] and s3["labels_changed"]["voxels"] > 0
    assert s3["volume_target"]["before"]["target_volume_mm3"] and s3["volume_target"]["after"]["target_volume_mm3"]
    size3, full3 = len(json.dumps(s3)), len(json.dumps(r3))
    r40 = call(case, *many_marks(40), out_dir=case.root / "h40")
    size40, full40 = len(json.dumps(feedback.short_report(r40))), len(json.dumps(r40))
    print(f"\nREPLY SIZES: 3 marks short {size3} B full {full3} B | 40 marks short {size40} B full {full40} B")
    assert size3 < full3 / 2 and size3 < 6000
    assert size40 < full40 / 4 and size40 < 16000
    assert (case.root / "h40" / "hints.json").stat().st_size > size40          # the full report is on disk
    err = {"ok": False, "error": "x"}
    assert feedback.short_report(err) == err


# ---- m3, nit 9: the history limit and caps ----------------------------------------------------------
def test_history_default_limit_count_and_caps(tmp_path):
    for i in range(25):
        d = tmp_path / "runs" / f"r{i:02d}"
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"state": "done", "iteration": i}))
        os.utime(d / "status.json", (1_700_000_000 + i, 1_700_000_000 + i))
    r = feedback.run_history(tmp_path)
    assert r["count"] == 25 and r["listed"] == 20 and len(r["runs"]) == 20
    assert r["runs"][0]["dir_name"] == "r24" and r["runs"][-1]["dir_name"] == "r05"           # the newest 20
    assert any("5 older runs are left out" in n for n in r["notes"])
    assert len(feedback.run_history(tmp_path, limit=0)["runs"]) == 25
    assert len(feedback.run_history(tmp_path, limit=None)["runs"]) == 25
    assert len(feedback.run_history(tmp_path, limit=3)["runs"]) == 3
    r = feedback.run_history(tmp_path, depth=99)
    assert r["depth"] == feedback.HISTORY_MAX_DEPTH and any("depth 99 is not used" in n for n in r["notes"])
    old = feedback.HISTORY_MAX_FOLDERS
    feedback.HISTORY_MAX_FOLDERS = 10
    try:
        r = feedback.run_history(tmp_path, limit=0)
    finally:
        feedback.HISTORY_MAX_FOLDERS = old
    assert r["count"] <= 10 and any("stopped after 10 folders" in n for n in r["notes"])
    assert feedback.run_history(tmp_path / "runs" / "r03")["count"] == 1                       # a run folder as root


# ---- m4: the target volume in the hint report -------------------------------------------------------
def test_hint_report_gives_the_target_volume_before_and_after(case):
    r = call(case, mark(1, "remove", [(10, 3, 6)], radius=2.5, mirror=True), out_dir=case.root / "v")
    v = r["volume_target"]
    assert v["before"]["target_volume_mm3"] > 0 and v["after"]["target_volume_mm3"] > 0
    assert v["before"]["target_volume_mm3"] != v["after"]["target_volume_mm3"]
    assert v["after"]["target_volume_mm3"] == r["check"]["derived"]["volume"]["target_volume_mm3"]
    assert "free design voxels" in v["note"]
    r = call(case, M1, out_dir=case.root / "v2", check=False)
    assert r["volume_target"]["after"] is None and r["volume_target"]["before"] is not None


# ---- m5: every key under regions --------------------------------------------------------------------
def test_every_key_under_regions_is_copied_rewritten_and_compared(tmp_path):
    base = tmp_path / "in"
    (base / "sub").mkdir(parents=True)
    for n in ("design.stl", "sub/s1.stl", "s2.stl", "extra.stl"):
        box(base / n, (0, 0, 0), (1, 1, 1))
    regions = {"design": "design.stl", "solid": ["sub/s1.stl", "s2.stl"], "other": {"cap": "extra.stl"}, "text": "no_file.stl"}
    paths = feedback._region_paths(regions, base)
    assert [p.name for p in paths] == ["design.stl", "s1.stl", "s2.stl", "extra.stl"]     # the missing file is skipped
    copies = {str(p): p.name for p in paths}
    assert feedback._rewrite_regions(regions, base, copies) == {
        "design": "design.stl", "solid": ["s1.stl", "s2.stl"], "other": {"cap": "extra.stl"}, "text": "no_file.stl"}
    items = feedback._region_files(base / "x.yaml", {"regions": regions}, with_hash=True)
    assert [i["key"] for i in items] == ["design", "solid:s1.stl", "solid:s2.stl", "other:cap", "text"]
    assert items[1]["sha256"] and items[4]["exists"] is False


# ---- m6: does the region touch the material of final.stl --------------------------------------------
def test_add_region_facts_say_if_it_touches_the_material_of_final_stl(case):
    on = call(case, mark(1, "add", [(14, 0, 6)], radius=2.0), out_dir=case.root / "on")
    f = on["marks"][0]["facts"]
    assert f["touches_final_material"] is True and f["final_material_in_region_mm3"] > 10
    box(case.run / "result" / "final.stl", (0, -6, 0), (4, 6, 12))             # the material is far from the mark
    off = call(case, mark(1, "add", [(14, 0, 6)], radius=2.0), out_dir=case.root / "off")
    f = off["marks"][0]["facts"]
    assert f["touches_final_material"] is False and f["final_material_in_region_mm3"] == 0.0
    (case.run / "result" / "final.stl").unlink()
    none = call(case, mark(1, "add", [(14, 0, 6)], radius=2.0), out_dir=case.root / "none")
    assert none["marks"][0]["facts"]["touches_final_material"] is None
    text = " ".join(o["text"] for o in feedback.MARK_OPTIONS["add"])
    assert "island" in text and "fixed region" in text and "final_material_in_region_mm3" in text


# ---- m7, nit 8: the YAML of a moved run -------------------------------------------------------------
def test_yaml_path_argument_for_a_moved_run(case):
    moved = case.root / "moved"
    shutil.move(str(case.inp), str(moved))
    r = call(case, M1, out_dir=case.root / "h")
    assert not r["ok"] and "yaml_path" in r["error"] and "not on disk" in r["error"]
    r = call(case, M1, out_dir=case.root / "h", yaml_path=moved / "bracket.yaml")
    assert r["ok"] and r["grid"]["labels_match_run"] is True and r["proposed_yaml"]["loads"] is True
    assert any("yaml_path was given" in n for n in r["notes"])
    assert r["source"]["yaml_path"] == str(moved / "bracket.yaml")
    r = call(case, M1, out_dir=case.root / "h2", yaml_path=moved / "nothing.yaml")
    assert not r["ok"] and "is not a file" in r["error"]
    box(moved / "design.stl", (0, -6, 0), (24, 6, 10))                          # other STL: the labels check says so
    r = call(case, M1, out_dir=case.root / "h3", yaml_path=moved / "bracket.yaml")
    assert r["grid"]["labels_match_run"] is False and any("do not give the labels" in p for p in r["problems"])


def test_a_relative_yaml_path_in_run_json_starts_at_the_run_folder(case, monkeypatch, tmp_path_factory):
    rj = json.loads((case.run / "run.json").read_text())
    rj["yaml_path"] = os.path.relpath(case.yaml, case.run)
    (case.run / "run.json").write_text(json.dumps(rj))
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))                    # the folder of the process is not the run folder
    assert feedback.run_facts(case.run)["yaml"]["exists"] is True
    assert feedback.compare_runs(case.run, case.run)["inputs"]["a"]["yaml"]["exists"] is True
    assert call(case, M1, out_dir=case.root / "h")["ok"]


# ---- nit 7: a mirror copy outside the grid ----------------------------------------------------------
def test_mirror_copy_outside_the_grid_box_is_a_fact_for_the_mark(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    box(d / "design.stl", (0, 0, 0), (24, 12, 12))                            # not symmetric about y = 0
    box(d / "plate.stl", (-2, 0, 0), (0, 12, 12))
    box(d / "tip.stl", (21, 4, 0), (24, 8, 4))
    doc = {"name": "shifted", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
           "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -5.0]}]}],
           "settings": {"voxel_size": 1.0, "volume_fraction": 0.35, "iterations": 5}}
    (d / "shifted.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    run = fake_run(tmp_path / "runs" / "a", d / "shifted.yaml")
    f = tmp_path / "m.json"
    f.write_text(json.dumps(marks_doc(mark(1, "remove", [(10, 10, 6)], radius=1.5, mirror=True))))
    r = feedback.make_hints(run, f, tmp_path / "h")
    e = r["marks"][0]
    assert e["status"] == "applied" and e["facts"]["mirror_copy"]["outside_grid_box"] is True
    assert e["facts"]["mirror_copy"]["voxels_touched"] == 0 and r["grid"]["design_symmetric_about_y0"] is False
    assert feedback.short_report(r)["marks"][0]["mirror_copy_outside_grid_box"] is True


# ---- round 2 (A3a review, M3): the filter radius and the filter facts -------------------------------------
def set_run_json(run, fn):
    f = Path(run) / "run.json"
    rj = json.loads(f.read_text())
    fn(rj)
    f.write_text(json.dumps(rj))


def set_metrics(run, fn):
    f = Path(run) / "result" / "metrics.json"
    m = json.loads(f.read_text())
    fn(m)
    f.write_text(json.dumps(m))


def test_a_derived_filter_radius_is_not_compared_with_the_default_rule(tmp_path):
    ypath = make_case(tmp_path / "in")                                         # no settings.filter_radius
    run = fake_run(tmp_path / "runs" / "a", ypath)
    h = json.loads((run / "run.json").read_text())["h"]
    # the engine derived another radius (A3a: 1.877 x min_member_mm), which is not 1.5 x voxel_size
    set_run_json(run, lambda rj: rj["derived"].__setitem__("filter_radius_mm", 1.877 * 2.0))
    assert abs(1.877 * 2.0 - 1.5 * h) > 0.1
    r = feedback.compare_runs(run, run)
    vr = r["inputs"]["a"]["yaml_vs_run_json"]
    assert vr["differs"] == [] and "settings.filter_radius" not in vr["compared"]
    assert vr["not_compared"] == [{"key": "settings.filter_radius", "run_json_key": "derived.filter_radius_mm",
                                   "run_json": 1.877 * 2.0, "reason": "derived by the engine; not compared"}]
    facts = " ".join(r["inputs"]["a"]["facts"])
    assert "settings.filter_radius is not in the YAML: derived by the engine; not compared" in facts
    assert "derived.filter_radius_mm = 3.754" in facts
    assert not any("filter_radius" in w for w in r["inputs"]["a"]["warnings"])
    assert not any("filter_radius" in x for x in r["summary"])


def test_an_explicit_filter_radius_is_still_compared(tmp_path):
    ypath = make_case(tmp_path / "in")
    edit_yaml(ypath, lambda d: d["settings"].__setitem__("filter_radius", 2.5))
    run = fake_run(tmp_path / "runs" / "a", ypath)
    r = feedback.compare_runs(run, run)
    vr = r["inputs"]["a"]["yaml_vs_run_json"]
    assert "settings.filter_radius" in vr["compared"] and vr["differs"] == [] and vr["not_compared"] == []
    edit_yaml(ypath, lambda d: d["settings"].__setitem__("filter_radius", 3.0))           # changed after the run
    r = feedback.compare_runs(run, run)
    assert [x["key"] for x in r["inputs"]["a"]["yaml_vs_run_json"]["differs"]] == ["settings.filter_radius"]
    assert any("settings.filter_radius" in w for w in r["inputs"]["a"]["warnings"])
    # and a run.json radius that another rule gave is a difference when the YAML sets the radius
    set_run_json(run, lambda rj: rj["derived"].__setitem__("filter_radius_mm", 9.0))
    r = feedback.compare_runs(run, run)
    x = r["inputs"]["a"]["yaml_vs_run_json"]["differs"][0]
    assert x["key"] == "settings.filter_radius" and x["yaml"] == 3.0 and x["run_json"] == 9.0


def test_filter_facts_of_run_json_and_metrics_json_are_copied_generically(tmp_path):
    ya, yb = make_case(tmp_path / "in_a"), make_case(tmp_path / "in_b")
    a = fake_run(tmp_path / "runs" / "a", ya)
    b = fake_run(tmp_path / "runs" / "b", yb)
    block_a = {"radius_mm": 1.5, "radius_source": "default", "solid_neighbours": "density_one", "min_member_mm": None,
               "facts": ["a long sentence that follows from the other keys"], "size": {"build_peak_gib_estimate": 0.1},
               "min_member_note": "a text"}
    block_b = dict(block_a, radius_mm=3.754, radius_source="min_member_mm", solid_neighbours="ignore",
                   min_member_mm=2.0, size={"build_peak_gib_estimate": 0.4})
    for run, blk, radius in ((a, block_a, 1.5), (b, block_b, 3.754)):
        set_run_json(run, lambda rj, blk=blk, radius=radius: (
            rj["derived"].__setitem__("filter", blk), rj["derived"].__setitem__("filter_radius_mm", radius),
            rj["derived"].__setitem__("filter_future_key", radius * 2)))   # a key that the code does not know
        set_metrics(run, lambda m, blk=blk: m.__setitem__("filter", blk))
    r = feedback.compare_runs(a, b, check_labels=False)
    keys = {x["key"]: x for x in r["setup"]["run_json_diff"]}
    for k in ("derived.filter.radius_mm", "derived.filter.radius_source", "derived.filter.solid_neighbours",
              "derived.filter.min_member_mm", "derived.filter_radius_mm", "derived.filter_future_key"):
        assert k in keys, (k, sorted(keys))
    assert keys["derived.filter.solid_neighbours"]["a"] == "density_one" and keys["derived.filter.solid_neighbours"]["b"] == "ignore"
    assert not any(k.startswith(("derived.filter.facts", "derived.filter.size", "derived.filter.min_member_note"))
                   for k in keys)
    # the block of metrics.json: side by side, and the differences in `other`
    m = r["metrics"]
    assert m["filter"]["a"]["filter.solid_neighbours"] == "density_one" and m["filter"]["b"]["filter.radius_mm"] == 3.754
    assert "filter.facts" not in m["filter"]["a"] and not any(k.startswith("filter.size") for k in m["filter"]["a"])
    differs = {x["key"]: x for x in m["filter"]["differs"]}
    assert differs["filter.solid_neighbours"]["b"] == "ignore" and differs["filter.radius_mm"]["a"] == 1.5
    assert not any(x["key"].startswith("filter") for x in m["other"])           # `other` is solver and watertight only
    assert any(s.startswith("filter differs:") and "solid_neighbours" in s for s in r["summary"]), r["summary"]
    # equal blocks: nothing differs, the block is still shown
    r = feedback.compare_runs(a, a, check_labels=False)
    assert r["metrics"]["filter"]["a"] == r["metrics"]["filter"]["b"] and r["metrics"]["filter"]["differs"] == []
    assert not any(s.startswith("filter differs") for s in r["summary"])
    # a run without the block (an older run): the keys have `missing_in`
    set_metrics(a, lambda m: m.pop("filter"))
    set_run_json(a, lambda rj: [rj["derived"].pop(k) for k in ("filter", "filter_future_key")])
    r = feedback.compare_runs(a, b, check_labels=False)
    assert r["metrics"]["filter"]["a"] is None and r["metrics"]["filter"]["b"]
    assert r["metrics"]["filter"]["differs"] and all(x.get("missing_in") == "a" for x in r["metrics"]["filter"]["differs"])
    assert any(x["key"] == "derived.filter.radius_mm" and x.get("missing_in") == "a" for x in r["setup"]["run_json_diff"])
