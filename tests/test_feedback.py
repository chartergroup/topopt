"""Tests for topopt/feedback.py: run history, compare, marks as hints (CONTRACTS.md section 21).

Most tests use a fake run (made with the real run writer, no optimizer): it is fast and each test can change it.
The loop test at the end runs the real optimizer through the MCP tool functions.
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import feedback  # noqa: E402
from topopt.config import load_config  # noqa: E402
from topopt.problem import build_problem  # noqa: E402
from topopt.snapshot import RunWriter  # noqa: E402
from topopt.voxelize import DESIGN, FIXED, KEEP_OUT, LOAD, inside_mask, voxelize  # noqa: E402

H = 1.0


def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def make_case(d, h=H, iterations=12, vf=0.35, keep_out=()):
    """A cantilever, symmetric about y = 0: design x 0..24, y -6..6, z 0..12; a plate at x -2..0; a tip load."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    box(d / "design.stl", (0, -6, 0), (24, 6, 12))
    box(d / "plate.stl", (-2, -6, 0), (0, 6, 12))
    box(d / "tip.stl", (21, -2, 0), (24, 2, 4))
    doc = {"name": "bracket",
           "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
           "load_cases": [{"name": "down", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -5.0]}]}],
           "settings": {"voxel_size": h, "volume_fraction": vf, "iterations": iterations,
                        "beta_schedule": [[0, 1], [6, 2], [10, 4]]}}
    if keep_out:
        doc["regions"]["keep_out"] = list(keep_out)
    p = d / "bracket.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def fake_run(run_dir, yaml_path, state="done", compliance=1.0, started_at=None):
    """A run folder that the real RunWriter wrote, with a short history, a box as final.stl and metrics.json."""
    run_dir = Path(run_dir)
    prob = build_problem(load_config(yaml_path))
    w = RunWriter(run_dir, quiet=True)
    w.clean()
    w.write_labels(prob.labels)
    w.write_run_json(prob, int(((prob.labels >= 1) & (prob.labels <= 3)).sum()), 100, "amg", started_at)
    for it in range(3):
        w.add_history({"iteration": it, "elapsed_s": it + 0.5, "compliance": compliance * (3 - it),
                       "volume_fraction": 0.4 - 0.01 * it})
    w.status(state, 3, "done" if state == "done" else "loop")
    if state == "done":
        (run_dir / "result").mkdir(exist_ok=True)
        box(run_dir / "result" / "final.stl", (0, -6, 0), (24, 6, 12))
        metrics = {"mass_design_space_g": 5.0, "mass_final_g": 2.0, "mass_saved_pct": 60.0, "volume_final_mm3": 1600.0,
                   "compliance_initial": compliance * 3, "compliance_final": compliance,
                   "stiffness_cases": [{"name": "down", "N_per_mm": 25.0 / compliance}], "run_time_s": 4.2,
                   "iterations": 3, "solver": "amg", "watertight": True, "triangles": 12}
        (run_dir / "result" / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


def tree_hash(*roots):
    out = {}
    for r in roots:
        r = Path(r)
        files = [r] if r.is_file() else sorted(p for p in r.rglob("*") if p.is_file())
        for p in files:
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def mark(mid, kind, points, radius=2.5, mirror=False, **kw):
    pts = [list(map(float, p)) for p in points]
    return {"id": mid, "type": kind, "points": pts, "normals": [[0, 0, 1]] * len(pts), "radius_mm": radius,
            "amount_mm": None if kind == "smooth" else 1, "strength": 1.5 if kind == "smooth" else None,
            "mirror": mirror, "note": kw.get("note", "")}


def marks_doc(*marks, sha=None):
    return {"version": 1, "source_stl": "/x/final.stl", "source_sha256": sha, "units": "mm",
            "created": "2026-10-20T00:00:00Z", "marks": list(marks)}


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def path_dist(P, path):
    path = np.asarray(path, float)
    if len(path) == 1:
        return np.linalg.norm(P - path[0], axis=1)
    return np.min([seg_dist(P, a, b) for a, b in zip(path[:-1], path[1:])], axis=0)


@pytest.fixture
def case(tmp_path):
    ypath = make_case(tmp_path / "in")
    run = fake_run(tmp_path / "runs" / "a", ypath)
    return type("Case", (), {"root": tmp_path, "yaml": ypath, "run": run, "inp": ypath.parent})


# ---- history ----------------------------------------------------------------------------------------
def test_history_empty_folder(tmp_path):
    r = feedback.run_history(tmp_path)
    assert r["ok"] and r["runs"] == [] and r["count"] == 0
    assert feedback.run_history(tmp_path / "nope")["ok"] is False


def test_history_facts_and_writes_nothing(case):
    before = tree_hash(case.root)
    r = feedback.run_history(case.root)
    assert tree_hash(case.root) == before                       # a read-only tool
    assert r["count"] == 1
    run = r["runs"][0]
    assert run["state"] == "done" and run["name"] == "bracket" and run["dir_name"] == "a"
    assert run["yaml_path"] == str(case.yaml) and run["yaml_exists"] and run["yaml_newer_than_run"] is False
    assert run["settings"]["voxel_size_mm"] == H and run["settings"]["volume_fraction"] == 0.35
    assert run["settings"]["solver"] == "amg" and run["settings"]["load_cases"] == ["down"]
    m = run["metrics"]
    assert m["compliance_final"] == 1.0 and m["mass_final_g"] == 2.0 and m["volume_final_mm3"] == 1600.0
    assert m["run_time_s"] == 4.2 and m["volume_fraction_final"] == pytest.approx(0.38)
    assert run["metrics_source"] == "result/metrics.json" and run["problems"] == []
    full = feedback.run_history(case.root, detail=True)["runs"][0]
    assert full["files"]["result/final.stl"] and full["files"]["snapshots"] == 0
    assert full["settings"]["grid_shape"] == [26, 12, 12]
    json.dumps(r)


def test_history_newest_first_depth_limit_and_skips(tmp_path):
    ypath = make_case(tmp_path / "in")
    for name, t in (("old", 1_000_000_000), ("new", 1_600_000_000)):
        d = fake_run(tmp_path / "runs" / name, ypath)
        for f in d.iterdir():
            if f.is_file():
                os.utime(f, (t, t))
        os.utime(d / "result" / "metrics.json", (t, t))
    fake_run(tmp_path / ".hidden" / "x", ypath)
    fake_run(tmp_path / "a" / "b" / "c" / "deep", ypath)
    (tmp_path / "link").symlink_to(tmp_path / "runs", target_is_directory=True)
    r = feedback.run_history(tmp_path)
    assert [x["dir_name"] for x in r["runs"]] == ["new", "old"]    # not the hidden, deep or linked folders
    assert feedback.run_history(tmp_path, limit=1)["runs"][0]["dir_name"] == "new"
    assert feedback.run_history(tmp_path, depth=4)["count"] == 3
    assert feedback.run_history(tmp_path / "runs" / "old")["count"] == 1       # a run folder as the root


def test_history_partial_files_are_facts(case):
    run = case.run
    (run / "result" / "metrics.json").unlink()
    with open(run / "history.jsonl", "a") as f:
        f.write('{"iteration": 3, "elapsed_s": 9')                     # a cut last line
    r = feedback.run_history(case.root)["runs"][0]
    assert r["state"] == "done" and r["metrics_source"].startswith("history.jsonl")
    assert r["metrics"]["compliance_last"] == 1.0
    text = " | ".join(r["problems"])
    assert "result/metrics.json is missing" in text and "history.jsonl: the last line cannot be read" in text
    (run / "status.json").write_text("{not json")
    (run / "run.json").unlink()
    r = feedback.run_history(case.root)["runs"][0]
    assert r["state"] == "unreadable" and r["settings"] is None and r["yaml_path"] is None
    assert any("run.json is missing" in p for p in r["problems"])
    assert any("status.json is not valid JSON" in p for p in r["problems"])


def test_history_running_run_has_no_metrics_problem(tmp_path):
    ypath = make_case(tmp_path / "in")
    fake_run(tmp_path / "runs" / "r", ypath, state="running")
    r = feedback.run_history(tmp_path)["runs"][0]
    assert r["state"] == "running" and r["problems"] == []
    assert r["metrics"]["compliance_last"] == 1.0                      # from the last history row


def test_history_yaml_newer_than_run_and_missing(case):
    future = time.time() + 3600
    os.utime(case.yaml, (future, future))
    r = feedback.run_history(case.root)["runs"][0]
    assert r["yaml_newer_than_run"] is True
    os.utime(case.inp / "plate.stl", (future, future))
    assert feedback.run_history(case.root, detail=True)["runs"][0]["yaml"]["newer_region_files"] == ["plate.stl"]
    case.yaml.unlink()
    r = feedback.run_history(case.root)["runs"][0]
    assert r["yaml_exists"] is False and r["state"] == "done"


# ---- compare ----------------------------------------------------------------------------------------
def make_pair(tmp_path):
    """Run A, and run B with a smaller volume fraction and one more keep-out region."""
    ya = make_case(tmp_path / "in_a")
    yb = make_case(tmp_path / "in_b", vf=0.25, keep_out=["hole.stl"])
    box(yb.parent / "hole.stl", (10, -1, 4), (14, 1, 8))
    a = fake_run(tmp_path / "runs" / "a", ya, compliance=1.0)
    b = fake_run(tmp_path / "runs" / "b", yb, compliance=1.25)
    return a, b, ya, yb


def test_compare_setup_and_metrics_with_sources(tmp_path):
    a, b, ya, yb = make_pair(tmp_path)
    before = tree_hash(tmp_path)
    r = feedback.compare_runs(a, b)
    assert tree_hash(tmp_path) == before
    assert r["ok"]
    rows = {x["key"]: x for x in r["metrics"]["rows"]}
    assert rows["compliance_final"]["delta"] == pytest.approx(0.25)
    assert rows["compliance_final"]["delta_pct"] == pytest.approx(25.0)
    assert r["metrics"]["source_a"] == "result/metrics.json"
    setup = {x["key"]: x for x in r["setup"]["run_json_diff"]}
    assert setup["volume_fraction"]["a"] == 0.35 and setup["volume_fraction"]["b"] == 0.25
    assert setup["regions.keep_out"]["b"] == ["hole.stl"]
    assert all("run.json" in x["source"] for x in setup.values())
    assert "derived.volume.free_design_voxels" not in setup           # numbers that follow from others are left out
    diff = {x["key"]: x for x in r["setup"]["yaml_diff"]["diff"]}
    assert diff["settings.volume_fraction"]["b"] == 0.25 and "not proven" in r["setup"]["yaml_diff"]["source"]
    regions = {x["key"]: x for x in r["setup"]["region_files"]["diff"]}
    assert regions["keep_out:hole.stl"]["status"] == "only_in_b" and r["setup"]["region_files"]["same_count"] == 3
    lab = r["setup"]["labels_diff"]
    assert lab["comparable"] and lab["voxels"] > 0 and lab["transitions"] == {"design->keep_out": lab["voxels"]}
    assert lab["bbox_mm"]["min"] == [10.0, -1.0, 4.0] and lab["bbox_mm"]["max"] == [14.0, 1.0, 8.0]
    assert any("compliance_final" in s for s in r["summary"]) and any("keep_out:hole.stl" in s for s in r["summary"])
    assert any("volume fractions differ" in n for n in r["notes"])
    json.dumps(r)


def test_compare_same_changed_stl_hash(tmp_path):
    a, b, ya, yb = make_pair(tmp_path)
    shutil.copyfile(ya.parent / "design.stl", yb.parent / "design.stl")
    box(yb.parent / "plate.stl", (-2, -6, 0), (0, 6, 11))               # another plate
    r = feedback.compare_runs(a, b)
    status = {x["key"]: x["status"] for x in r["setup"]["region_files"]["diff"]}
    assert status["fixed:plate.stl"] == "changed" and "design" not in status


def test_compare_says_when_the_files_on_disk_may_not_be_the_ones_that_ran(tmp_path):
    a, b, ya, yb = make_pair(tmp_path)
    base = feedback.compare_runs(a, b)
    for side in ("a", "b"):
        facts = " ".join(base["inputs"][side]["facts"])
        assert "Compared with run.json, equal:" in facts and "same labels as labels.bin" in facts
        assert "does not record material.E" in facts and "no hash" in facts and base["inputs"][side]["labels_vs_disk"]["labels_same"] is True
    # 1. the YAML file is newer than the run
    future = time.time() + 3600
    os.utime(ya, (future, future))
    r = feedback.compare_runs(a, b)
    assert r["inputs"]["a"]["yaml"]["newer_than_run"] is True
    assert any("YAML file is newer than the run" in f for f in r["inputs"]["a"]["facts"])
    lines = [s for s in r["summary"] if s.startswith("run A:")]
    assert len(lines) == 1 and "the YAML is newer than the run" in lines[0]        # one line for each run
    assert "Still true: the YAML values equal run.json and the voxel labels equal labels.bin" in lines[0]
    # 2. the YAML was edited after the run: a value differs from run.json
    doc = yaml.safe_load(ya.read_text())
    doc["settings"]["volume_fraction"] = 0.5
    doc["load_cases"][0]["forces"][0]["force"] = [0, 0, -9.0]
    ya.write_text(yaml.safe_dump(doc, sort_keys=False))
    r = feedback.compare_runs(a, b)
    keys = {x["key"]: x for x in r["inputs"]["a"]["yaml_vs_run_json"]["differs"]}
    assert keys["settings.volume_fraction"]["yaml"] == 0.5 and keys["settings.volume_fraction"]["run_json"] == 0.35
    assert "load_cases[0].forces" in keys
    # 3. an STL changed: the labels that the files give differ from labels.bin
    box(ya.parent / "design.stl", (0, -6, 0), (24, 6, 10))
    r = feedback.compare_runs(a, b)
    lv = r["inputs"]["a"]["labels_vs_disk"]
    assert lv["checked"] and lv["labels_same"] is False
    assert any("not the files that ran" in f for f in r["inputs"]["a"]["facts"])
    # 4. the switch
    r = feedback.compare_runs(a, b, check_labels=False)
    assert r["inputs"]["a"]["labels_vs_disk"] == {"checked": False, "reason": "not requested (check_labels=false)"}
    # 5. the YAML is gone
    ya.unlink()
    r = feedback.compare_runs(a, b)
    assert r["ok"] and any("not on disk" in f for f in r["inputs"]["a"]["facts"])
    assert r["setup"]["yaml_diff"]["diff"] is None                     # no YAML diff when a file is missing
    assert r["setup"]["region_files"]["diff"] is None and any("not compared" in n for n in r["notes"])


def test_compare_same_yaml_for_both_runs_and_partial_runs(tmp_path):
    ypath = make_case(tmp_path / "in")
    a = fake_run(tmp_path / "runs" / "a", ypath, compliance=1.0)
    b = fake_run(tmp_path / "runs" / "b", ypath, compliance=0.5)
    r = feedback.compare_runs(a, b)
    assert r["setup"]["yaml_diff"]["same_file"] is True and r["setup"]["run_json_diff"] == []
    assert any("same YAML file" in n for n in r["notes"])
    assert r["setup"]["labels_diff"]["voxels"] == 0 and "the voxel labels of the two runs are equal" in r["summary"]
    (b / "run.json").unlink()
    (b / "labels.bin").unlink()
    (b / "result" / "metrics.json").unlink()
    r = feedback.compare_runs(a, b)                                     # partial run: facts, no crash
    assert r["ok"] and r["setup"]["run_json_diff"] is None and r["setup"]["labels_diff"]["comparable"] is False
    assert any("run.json is missing" in p for p in r["problems"])
    assert feedback.compare_runs(a, tmp_path / "nope")["ok"] is False


def test_compare_grids_that_differ(tmp_path):
    ya = make_case(tmp_path / "in_a")
    yb = make_case(tmp_path / "in_b", h=1.5)
    a, b = fake_run(tmp_path / "runs" / "a", ya), fake_run(tmp_path / "runs" / "b", yb)
    r = feedback.compare_runs(a, b)
    assert r["setup"]["labels_diff"]["comparable"] is False and "grids differ" in r["setup"]["labels_diff"]["reason"]
    assert any("voxel sizes differ" in n for n in r["notes"])


# ---- marks as hints: geometry -----------------------------------------------------------------------
def hints(case, *marks, **kw):
    f = case.root / "marks.json"
    f.write_text(json.dumps(marks_doc(*marks)))
    r = feedback.make_hints(case.run, f, kw.pop("out_dir", case.root / "hints"), kw.pop("options", None), **kw)
    assert r["ok"], r
    return r


def region_mesh(r, entry):
    return trimesh.load(str(Path(r["out_dir"]) / entry["region_file"]), file_type="stl", force="mesh")


def test_region_geometry_matches_the_marks(case):
    sphere = mark(1, "remove", [(10, 3, 6)], radius=2.5, mirror=True)           # a point, mirrored
    stroke = mark(2, "add", [(14, -3.5, 8), (17, -3.5, 8), (19, -3.5, 10)], radius=1.5)   # a stroke
    single = mark(3, "remove", [(16, 0, 9)], radius=2.0)                        # a point, not mirrored
    r = hints(case, sphere, stroke, single)
    e1, e2, e3 = r["marks"]
    assert [e["status"] for e in r["marks"]] == ["applied"] * 3
    assert [e["option_used"] for e in r["marks"]] == ["keep_out", "solid", "keep_out"]
    # a sphere: every vertex is at the radius from the centre; the mirror copy is a second body at y -> -y
    m1 = region_mesh(r, e1)
    bodies = m1.split(only_watertight=False)
    assert len(bodies) == 2
    centres = sorted(b.bounds.mean(axis=0).tolist() for b in bodies)
    assert centres[0] == pytest.approx([10, -3, 6], abs=H / 10) and centres[1] == pytest.approx([10, 3, 6], abs=H / 10)
    for b, c in zip(sorted(bodies, key=lambda b: b.bounds.mean(axis=0)[1]), ([10, -3, 6], [10, 3, 6])):
        d = np.linalg.norm(np.asarray(b.vertices) - c, axis=1)
        assert abs(d - 2.5).max() < H                                        # within one voxel (it is far tighter)
        assert abs(d - 2.5).max() < 0.02
    # a stroke: every vertex is at the radius from the polyline
    m2 = region_mesh(r, e2)
    d = path_dist(np.asarray(m2.vertices), [(14, -3.5, 8), (17, -3.5, 8), (19, -3.5, 10)])
    assert abs(d - 1.5).max() < H and abs(d - 1.5).max() < 0.02 and len(m2.split(only_watertight=False)) == 1
    # a point without mirror: one body, centred on the mark
    m3 = region_mesh(r, e3)
    assert len(m3.split(only_watertight=False)) == 1
    assert m3.bounds.mean(axis=0) == pytest.approx([16, 0, 9], abs=H / 10)
    assert e1["facts"]["applied_region"]["cut_to"] == [] and e1["facts"]["applied_region"]["cut_volume_pct"] == 0.0
    assert e1["region"]["kind"] == "sphere" and e2["region"]["kind"] == "capsule chain"
    assert e1["region"]["radius_mm"] == 2.5 and e1["region"]["mirror"] is True
    # watertight and the loader reads them
    for e in (e1, e2, e3):
        assert region_mesh(r, e).is_watertight


def test_proposed_yaml_loads_and_labels_differ_only_near_the_marks(case):
    sphere = mark(1, "remove", [(10, 3, 6)], radius=2.5, mirror=True)
    stroke = mark(2, "add", [(14, -3.5, 8), (17, -3.5, 8), (19, -3.5, 10)], radius=1.5)
    r = hints(case, sphere, stroke)
    new_yaml = Path(r["proposed_yaml"]["path"])
    assert new_yaml.parent == Path(r["out_dir"]) and r["proposed_yaml"]["loads"] is True
    cfg0, cfg1 = load_config(case.yaml), load_config(new_yaml)                  # the normal config path
    assert cfg1.settings == cfg0.settings and cfg1.load_cases == cfg0.load_cases and cfg1.material == cfg0.material
    assert [lr.name for lr in cfg1.regions.load] == ["tip", "hint_add_2"] and len(cfg1.regions.keep_out) == 1
    o0, h0, s0, lab0, _ = voxelize(cfg0)
    o1, h1, s1, lab1, _ = voxelize(cfg1)
    assert (h1, s1) == (h0, s0) and np.allclose(o1, o0)                         # the grid does not move
    diff = np.argwhere(lab0 != lab1)
    assert len(diff) == r["labels_changed"]["voxels"] > 0
    centres = o0 + h0 * (diff + 0.5)
    near = np.minimum(path_dist(centres, [(10, 3, 6)]) - 2.5, path_dist(centres, [(10, -3, 6)]) - 2.5)
    near = np.minimum(near, path_dist(centres, [(14, -3.5, 8), (17, -3.5, 8), (19, -3.5, 10)]) - 1.5)
    assert near.max() <= H                                                       # only near the marks
    assert set(r["labels_changed"]["transitions"]) == {"design->keep_out", "design->load"}
    assert (lab1[lab0 == FIXED] == FIXED).all() and ((lab1 == LOAD) | (lab0 != LOAD)).all()
    assert r["check"]["active_voxels"] > 0 and r["check_error"] is None


def test_the_report_names_the_c3b_keys_as_not_used(case):
    """The keys that the list command (CONTRACTS 29.3) adds to a mark are not read by the hints: the report says so (review n4)."""
    m = mark(1, "remove", [(10, 3, 6)])
    m.update({"remove_mode": "shave", "cleanup": {"detector": "spur"}})
    r = hints(case, m, mark(2, "smooth", [(8, 0, 12)]))
    assert {"smooth_mode", "remove_mode", "cleanup"} <= set(r["values_not_used"])
    assert any("smooth_mode, remove_mode, cleanup" in n for n in r["notes"])


def test_inputs_and_run_folder_are_unchanged(case):
    before = tree_hash(case.inp, case.run)
    r = hints(case, mark(1, "remove", [(10, 3, 6)], mirror=True), mark(2, "add", [(14, -3.5, 8)], radius=1.5),
              mark(3, "smooth", [(8, 0, 12)]))
    assert tree_hash(case.inp, case.run) == before
    assert r["source"]["yaml_sha256"] == hashlib.sha256(case.yaml.read_bytes()).hexdigest()
    out = Path(r["out_dir"])
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["bracket.yaml", "design.stl", "plate.stl", "tip.stl", "hint_remove_1.stl", "hint_add_2.stl", "hints.json"])
    assert json.loads((out / "hints.json").read_text())["marks"][0]["id"] == 1  # the report is on disk
    assert not any(p.name.startswith("hint") for p in case.inp.iterdir())


def test_zero_change_proposal_gives_equal_labels_and_copies_stl_files_flat(tmp_path):
    """Region files in other folders with equal names are copied flat; nothing changes without a hint."""
    d = tmp_path / "in"
    make_case(d)
    (d / "k1").mkdir()
    (d / "k2").mkdir()
    box(d / "k1" / "hole.stl", (8, -1, 4), (10, 1, 8))
    box(d / "k2" / "hole.stl", (14, -1, 4), (16, 1, 8))
    doc = yaml.safe_load((d / "bracket.yaml").read_text())
    doc["regions"]["keep_out"] = ["k1/hole.stl", "k2/hole.stl"]
    (d / "bracket.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    run = fake_run(tmp_path / "runs" / "a", d / "bracket.yaml")
    f = tmp_path / "m.json"
    f.write_text(json.dumps(marks_doc(mark(1, "smooth", [(5, 0, 6)]))))
    r = feedback.make_hints(run, f)                                      # default out_dir, no change possible
    assert r["ok"] and r["out_dir"] == str(tmp_path / "runs" / "a-hints")
    assert sorted(r["files"]["copied_region_files"]) == ["design.stl", "hole.stl", "hole_2.stl", "plate.stl", "tip.stl"]
    new = yaml.safe_load((Path(r["out_dir"]) / "bracket.yaml").read_text())
    assert new["regions"]["keep_out"] == ["hole.stl", "hole_2.stl"]
    assert r["labels_changed"]["voxels"] == 0 and r["labels_changed"]["grid_unchanged"] is True
    assert r["marks"][0]["status"] == "facts_only"
    r2 = feedback.make_hints(run, f)                                     # a second call: a new folder
    assert r2["out_dir"] == str(tmp_path / "runs" / "a-hints-2")


def test_voxel_facts_and_overlaps(case):
    """A mark on the fixed plate: the facts name the overlap; the keep-out hint is cut by the plate."""
    on_plate = mark(1, "remove", [(1.0, 0, 6)], radius=2.5)
    r = hints(case, on_plate)
    e = r["marks"][0]
    f = e["facts"]
    cfg = load_config(case.yaml)
    o, h, shape, lab, _ = voxelize(cfg)
    raw = trimesh.creation.icosphere(subdivisions=4, radius=2.5)
    raw.apply_translation((1.0, 0, 6))
    mask = inside_mask(raw, o, h, shape)
    assert f["voxels_touched"] == pytest.approx(int(mask.sum()), abs=int(0.03 * mask.sum()))   # polygon sphere vs ico
    assert f["voxels_by_label"]["fixed"] > 0 and f["voxels_by_label"]["design"] > 0
    assert f["overlap_regions"]["fixed"]["plate.stl"] == f["voxels_by_label"]["fixed"]
    assert f["overlap_regions"]["design"] == f["voxels_by_label"]["design"]
    assert f["design_voxels_share_pct"] > 0 and f["outside_grid_box_pct"] == 0.0
    assert f["applied_region"]["cut_to"] == ["the fixed and load regions"]
    assert r["labels_changed"]["transitions"] == {"design->keep_out": f["applied_region"]["design_voxels"]}
    # without protection the plate voxels in the region become keep-out: a fact in the labels
    r2 = hints(case, on_plate, out_dir=case.root / "hints2", options={"protect_regions": False})
    assert r2["labels_changed"]["transitions"]["fixed->keep_out"] == f["voxels_by_label"]["fixed"]
    assert r2["marks"][0]["facts"]["applied_region"]["cut_to"] == []


def test_mark_types_and_statuses(case):
    """Smooth, a mark outside the box, an add inside the fixed plate, a bad mark: facts, no crash, no silent drop."""
    r = hints(case,
              mark(1, "smooth", [(8, 0, 12), (12, 0, 12)], radius=3),
              mark(2, "remove", [(100, 0, 6)]),
              mark(3, "add", [(-1, 0, 6)], radius=0.8),
              {"id": 4, "type": "twist", "points": [[1, 1, 1]], "radius_mm": 2},
              mark(5, "remove", [(10, 0, 6)], radius=2.0))
    st = {e["id"]: e for e in r["marks"]}
    assert st[1]["status"] == "facts_only" and st[1]["option_used"] == "none" and st[1]["facts"]["voxels_touched"] > 0
    assert {o["id"] for o in st[1]["options"]} == {"none", "filter_radius", "mesh_edit"}
    assert "no key" in st[1]["reason"] and st[1]["facts"]["strength"] == 1.5
    assert st[2]["status"] == "no_effect" and "outside the bounds" in st[2]["reason"]
    assert st[3]["status"] == "no_effect" and "design space" in st[3]["reason"]
    assert st[4]["status"] == "unreadable" and "type" in st[4]["reason"]
    assert st[5]["status"] == "applied"
    assert r["summary"] == {"smooth:facts_only": 1, "remove:no_effect": 1, "add:no_effect": 1,
                            "mark:unreadable": 1, "remove:applied": 1, "proposed_yaml": "loads"}
    cat = r["option_catalog"]
    assert {o["id"] for o in cat["remove"]} >= {"keep_out", "skip", "volume_fraction"}
    assert all(o["text"] for t in cat.values() for o in t)
    assert [e["status"] for e in r["marks"] if e["status"] == "applied"] == ["applied"]
    assert r["files"]["region_files"] == ["hint_remove_5.stl"]


def test_a_run_with_an_asked_mass_gets_options_that_name_the_mass_key(case):
    """Review n5: a YAML with `settings.target_mass_g` has no `settings.volume_fraction`; the option text names the key that the user can edit."""
    plain = hints(case, mark(1, "remove", [(10, 3, 6)], radius=2.5), mark(2, "add", [(14, -3, 8)], radius=2.0))["option_catalog"]
    for mtype, verb in (("remove", "Lower"), ("add", "Raise")):
        text = next(o["text"] for o in plain[mtype] if o["id"] == "volume_fraction")
        assert text.startswith(f"{verb} settings.volume_fraction.") and "target_mass_g" not in text
    path = case.run / "run.json"
    doc = json.loads(path.read_text())
    doc.setdefault("derived", {})["target_mass_asked_g"] = 25.0
    path.write_text(json.dumps(doc))
    cat = hints(case, mark(1, "remove", [(10, 3, 6)], radius=2.5), mark(2, "add", [(14, -3, 8)], radius=2.0),
                out_dir=case.root / "hints_mass")["option_catalog"]
    for mtype, verb in (("remove", "Lower"), ("add", "Raise")):
        opt = next(o for o in cat[mtype] if o["id"] == "volume_fraction")
        assert opt["text"].startswith(f"{verb} settings.target_mass_g (the asked mass; the YAML of this run has no settings.volume_fraction).")
        assert "everywhere, not at the mark" in opt["text"] and opt["implemented"] is False
    assert {o["id"] for o in cat["remove"]} == {o["id"] for o in plain["remove"]}      # the ids do not change
    assert feedback.mark_options({"derived": {}}) is feedback.MARK_OPTIONS and feedback.mark_options(None) is feedback.MARK_OPTIONS
    assert "target_mass_g" not in " ".join(o["text"] for t in feedback.MARK_OPTIONS.values() for o in t)      # the shared catalog is not changed


def test_options(case):
    marks = [mark(1, "remove", [(10, 3, 6)], radius=2.5), mark(2, "add", [(14, -3, 8)], radius=2.0),
             mark(3, "remove", [(16, 0, 9)], radius=2.0)]
    r = hints(case, *marks, options={"radius_scale": 0.5, "per_mark": {"3": "skip"}})
    e1, e2, e3 = r["marks"]
    assert e1["facts"]["radius_used_mm"] == 1.25 and e1["region"]["radius_mm"] == 1.25
    assert e3["status"] == "skipped_by_option" and e3["option_used"] == "skip"
    assert e1["option_used"] == "keep_out" and [o["used"] for o in e1["options"]][:2] == [True, False]
    assert np.abs(np.linalg.norm(region_mesh(r, e2).vertices - [14, -3, 8], axis=1) - 1.0).max() < 0.02
    r = hints(case, *marks, out_dir=case.root / "h2", options={"remove": "skip", "add": "skip"})
    assert {e["status"] for e in r["marks"]} == {"skipped_by_option"} and r["labels_changed"]["voxels"] == 0
    assert r["files"]["region_files"] == []
    r = hints(case, *marks, out_dir=case.root / "h3", options={"remove": "solid", "radius_scale": -1, "bogus": 1})
    assert len(r["problems"]) >= 3 and any("bogus" in p for p in r["problems"])
    r = hints(case, mark(1, "add", [(10, 3, 6)]), out_dir=case.root / "h4", options={"per_mark": {"1": "keep_out"}})
    assert r["marks"][0]["option_used"] == "solid" and "option_note" in r["marks"][0]["facts"]


def test_marks_input_and_file_checks(case):
    good = marks_doc(mark(1, "remove", [(10, 0, 6)]))
    bad = case.root / "x.json"
    bad.write_text("not json")
    assert "not valid JSON" in feedback.make_hints(case.run, bad, case.root / "o1")["error"]
    assert "not a marks file" in feedback.make_hints(case.run, {"version": 2, "marks": []}, case.root / "o2")["error"]
    assert "cannot read the marks file" in feedback.make_hints(case.run, case.root / "none.json")["error"]
    # the JSON text and the dict work like the file
    r = feedback.make_hints(case.run, json.dumps(good), case.root / "o3")
    assert r["ok"] and r["marks_file"] is None and r["marks"][0]["status"] == "applied"
    # a hash that does not fit final.stl, and units
    bad_sha = dict(good, source_sha256="0" * 64, units="inch")
    r = feedback.make_hints(case.run, bad_sha, case.root / "o4")
    assert r["marks_match_final_stl"] is False
    assert any("source_sha256" in p for p in r["problems"]) and any("units" in p for p in r["problems"])
    ok = dict(good, source_sha256=hashlib.sha256((case.run / "result" / "final.stl").read_bytes()).hexdigest())
    assert feedback.make_hints(case.run, ok, case.root / "o5")["marks_match_final_stl"] is True
    # a mark that is far from the surface of final.stl is a fact
    far = marks_doc(mark(1, "remove", [(12, 0, 6)]))
    e = feedback.make_hints(case.run, far, case.root / "o6")["marks"][0]
    assert e["facts"]["surface_distance_mm"] == [6.0] and "more than two voxels" in e["facts"]["surface_note"]


def test_out_dir_rules_and_missing_inputs(case):
    m = marks_doc(mark(1, "remove", [(10, 0, 6)]))
    r = feedback.make_hints(case.run, m, case.run / "hints")
    assert not r["ok"] and r["refused"]["reason"] == "run_folder" and "inside the run folder" in r["error"]
    r = feedback.make_hints(case.run, m, case.inp)
    assert not r["ok"] and r["refused"]["reason"] == "input_folder" and "holds an input file" in r["error"]
    out = case.root / "new"
    assert feedback.make_hints(case.run, m, out)["ok"]
    r = feedback.make_hints(case.run, m, out)
    assert not r["ok"] and r["refused"]["reason"] == "not_empty" and "holds files" in r["error"]
    assert feedback.make_hints(case.run, m, out, overwrite=True)["ok"]          # an earlier hints folder
    assert "run folder not found" in feedback.make_hints(case.root / "no", m)["error"]
    case.yaml.unlink()
    r = feedback.make_hints(case.run, m, case.root / "o2")
    assert "YAML of the run is not on disk" in r["error"] and "yaml_path" in r["error"]
    (case.run / "run.json").unlink()
    assert "run.json" in feedback.make_hints(case.run, m, case.root / "o3")["error"]


def test_hint_that_breaks_the_problem_is_a_fact_not_a_crash(case):
    """A keep-out wall across the whole beam cuts the part in two: the proposed YAML loads, the check says so."""
    wall = mark(1, "remove", [(12, y, z) for y in (-6, 0, 6) for z in (0, 6, 12)], radius=3.0)
    r = hints(case, wall)
    assert r["proposed_yaml"]["loads"] is True
    assert any("separate pieces" in w for w in r["check"]["warnings"])
    assert r["check"]["connectivity"]["components"] > 1


def test_surface_facts_without_final_stl(case):
    (case.run / "result" / "final.stl").unlink()
    r = hints(case, mark(1, "remove", [(10, 0, 6)]))
    assert "surface_distance_mm" not in r["marks"][0]["facts"] and r["marks_match_final_stl"] is None
    assert any("final.stl is missing" in p for p in r["problems"])


def test_report_is_plain_json_and_cli(case, capsys):
    f = case.root / "marks.json"
    f.write_text(json.dumps(marks_doc(mark(1, "remove", [(10, 3, 6)], mirror=True))))
    assert feedback.main(["hints", str(case.run), str(f), "--out", str(case.root / "cli"), "--radius-scale", "0.8",
                          "--no-check", "--detail"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["marks"][0]["facts"]["radius_used_mm"] == 2.0 and r["check"] is None
    assert feedback.main(["hints", str(case.run), str(f), "--out", str(case.root / "cli2")]) == 0   # the short reply
    r = json.loads(capsys.readouterr().out)
    assert r["detail"] is False and r["marks"][0]["radius_used_mm"] == 2.5 and "option_catalog" not in r
    assert feedback.main(["history", str(case.root)]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1
    assert feedback.main(["compare", str(case.run), str(case.run)]) == 0
    assert json.loads(capsys.readouterr().out)["setup"]["labels_diff"]["voxels"] == 0
    assert feedback.main(["hints", str(case.root / "no"), str(f)]) == 1


def test_module_does_not_import_heavy_packages_at_import_time():
    import subprocess
    code = ("import sys, topopt.feedback;"
            "bad=[m for m in ('trimesh','manifold3d','scipy','numba','mcp') if m in sys.modules];"
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    assert out == "[]", out


def test_check_hints_script_passes_and_catches_a_wrong_region(case, capsys):
    """tools/check_hints.py reads the hints folder from disk and checks it (with --make it hashes before and after)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_hints", REPO / "tools" / "check_hints.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    f = case.root / "marks.json"
    f.write_text(json.dumps(marks_doc(mark(1, "remove", [(10, 3, 6)], mirror=True),
                                      mark(2, "add", [(14, -3.5, 8), (17, -3.5, 8)], radius=1.5))))
    out = case.root / "hints"
    assert script.main([str(case.run), str(f), str(out), "--make"]) == 0
    text = capsys.readouterr().out
    assert "RESULT: OK" in text and "centre and radius within one voxel" in text and "both copies present" in text
    assert "[FAIL]" not in text and "unchanged" in text
    # a region that is 2 mm away from its mark is found
    mesh = trimesh.load(str(out / "hint_remove_1.stl"), file_type="stl", force="mesh")
    mesh.apply_translation((2.0, 0, 0))
    mesh.export(out / "hint_remove_1.stl")
    assert script.main([str(case.run), str(f), str(out)]) == 1
    assert "[FAIL] mark 1 (remove): centre and radius within one voxel" in capsys.readouterr().out


def test_region_that_reaches_outside_the_input_regions_does_not_move_the_grid(case):
    """The origin is the minimum of all region bounds. A hint region is cut to the box, so the grid stays."""
    r = hints(case, mark(1, "remove", [(1.0, 0, 6)], radius=3.5), mark(2, "add", [(22, 5.0, 1.0)], radius=3.0))
    assert r["marks"][0]["facts"]["outside_grid_box_pct"] > 0 and r["marks"][1]["facts"]["outside_grid_box_pct"] > 0
    assert r["labels_changed"]["grid_unchanged"] is True and r["proposed_yaml"]["loads"] is True
    o0, h0, s0, _l, _i = voxelize(load_config(case.yaml))
    o1, h1, s1, _l, _i = voxelize(load_config(r["proposed_yaml"]["path"]))
    assert (h1, tuple(s1)) == (h0, tuple(s0)) and np.array_equal(o0, o1)
