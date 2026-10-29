"""Tests for topopt/apply_marks.py and the MCP tool apply_marks (CONTRACTS.md section 29).

Each test makes a small run folder (the real RunWriter, no optimizer) and puts a finished part into result/final.stl:
the part is built in the test from boxes and cylinders, so no binary fixture is in the repository. The marks are small
dicts, as the Refine page exports them.
"""
import asyncio
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import manifold3d as m3d
import numpy as np
import pytest
import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import apply_marks as am  # noqa: E402
from topopt import feedback  # noqa: E402
from topopt_mcp import server  # noqa: E402
from test_feedback import fake_run, make_case, mark, marks_doc, tree_hash  # noqa: E402

M = m3d.Manifold
FAST = {"pitch_mm": 0.35}        # a coarser crop grid than the default (voxel size / 4): tests that check files and statuses


def cube(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return M.cube(tuple(hi - lo)).translate(tuple(lo))


def cyl_z(x, y, z0, z1, r, seg=48):
    return M.cylinder(z1 - z0, r, r, seg).translate((x, y, z0))


def cyl_x(y, z, x0, x1, r, seg=48):
    return M.cylinder(x1 - x0, r, r, seg).rotate((0, 90, 0)).translate((x0, y, z))


def to_stl(man, path):
    am._mesh_of(man).export(str(path))


# The case of test_feedback: design x 0..24, y -6..6, z 0..12; a fixed plate x -2..0; a load tip x 21..24, y -2..2, z 0..4.
PLATE = lambda: cube((-2, -6, 0), (0, 6, 12))                      # noqa: E731
BAR = lambda: cube((0, -4, 0), (24, 4, 8))                         # noqa: E731


def make_run(root, part, name="a"):
    """A run folder whose result/final.stl is `part` (a manifold3d solid)."""
    ypath = make_case(Path(root) / "in")
    run = fake_run(Path(root) / "runs" / name, ypath)
    to_stl(part, run / "result" / "final.stl")
    return run, ypath


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def doc_for(run, *marks, sha_ok=True):
    return marks_doc(*marks, sha=sha(Path(run) / "result" / "final.stl") if sha_ok else None)


def area(mesh):
    return float(mesh.area)


def load_edited(rep):
    return trimesh.load(str(Path(rep["out_dir"]) / "edited.stl"), file_type="stl", force="mesh")


def spur_part():
    """Plate + bar + a spur on top (axis z, radius 1.2, z 8..11) at x=12, y=2.5 and its mirror image at y=-2.5."""
    return PLATE() + BAR() + cyl_z(12, 2.5, 8, 11, 1.2) + cyl_z(12, -2.5, 8, 11, 1.2)


@pytest.fixture
def spur(tmp_path):
    run, ypath = make_run(tmp_path, spur_part())
    return type("Case", (), {"root": tmp_path, "run": run, "yaml": ypath})


def plane_of(*a):
    planes, _facts = am._planes_for(*a)
    assert len(planes) == 1
    return planes[0]


def test_plane_order_option_mark_file_run_default():
    run = {"mirror": [{"axis": "z", "at_mm": 3.0}]}
    base = {"mirror_plane": None}
    assert plane_of({}, {}, {}, base)["source"].startswith("default")
    pl = plane_of({}, {}, run, base)
    assert (pl["axis"], pl["at_mm"]) == ("z", 3.0) and "run.json" in pl["source"]
    pl = plane_of({}, {"mirror_plane": {"axis": "x", "at_mm": 1.0}}, run, base)
    assert (pl["axis"], pl["at_mm"]) == ("x", 1.0) and "marks file" in pl["source"]
    pl = plane_of({"mirror_plane": {"axis": "y", "at_mm": 2.0}}, {"mirror_plane": {"axis": "x"}}, run, base)
    assert (pl["axis"], pl["at_mm"]) == ("y", 2.0) and "mark" in pl["source"]
    opt = {"mirror_plane": am._plane_from({"axis": "x", "at_mm": -1.0}, "call option mirror_plane")}
    pl = plane_of({"mirror_plane": {"axis": "y"}}, {}, run, opt)
    assert (pl["axis"], pl["at_mm"]) == ("x", -1.0)
    pts = np.array([[1.0, 2.0, 3.0]])
    assert am._mirror_points(pts, {"axis_index": 1, "at_mm": 0.5}).tolist() == [[1.0, -1.0, 3.0]]


def test_planes_of_run_read_the_a4a_layout():
    """M3: run.json mirror is {planes: [{axis, at_mm, ...}], design_voxels, ...} (CONTRACTS 26.1). Up to 3 planes."""
    a4a = {"mirror": {"planes": [{"axis": "y", "at_mm": 5.0, "align": "auto", "parity": "face", "k": 10, "plane_index": 5},
                                 {"axis": "x", "at_mm": 20.0, "align": "auto", "parity": "centre", "k": 3, "plane_index": 1}],
                      "design_voxels": 1000, "load_asymmetry": [], "end": {}}}
    planes, facts = am._planes_of_run(a4a)
    assert [(p["axis"], p["at_mm"]) for p in planes] == [("y", 5.0), ("x", 20.0)] and not facts
    assert all("run.json" in p["source"] for p in planes)
    one = {"mirror": {"planes": [{"axis": "y", "at_mm": 5.0}]}}
    pl = plane_of({}, {}, one, {"mirror_plane": None})
    assert (pl["axis"], pl["at_mm"]) == ("y", 5.0) and pl["source"] == "run.json (mirror)"
    # the old forms stay: a list, one dict, the key `normal`, a 3-vector
    assert am._planes_of_run({"mirror": [{"normal": "Z", "at_mm": 1}]})[0][0]["axis"] == "z"
    assert am._planes_of_run({"mirror": {"axis": "x", "at_mm": 2}})[0][0]["at_mm"] == 2.0
    assert am._planes_of_run({"mirror": [{"normal": [0, -1, 0], "at_mm": 4}]})[0][0]["axis"] == "y"
    # no plane, a repeated axis, an unreadable entry: a fact, not a guess
    assert am._planes_of_run({"mirror": None}) == ([], [])
    planes, facts = am._planes_of_run({"mirror": {"planes": [{"axis": "y", "at_mm": 1}, {"axis": "y", "at_mm": 2}, {"foo": 1}]}})
    assert len(planes) == 1 and len(facts) == 2 and "axis of an earlier plane" in facts[0]


def test_mirror_images_form_the_group_of_the_reflections():
    path = np.array([[10.0, 2.0, 5.0], [10.0, 3.0, 5.0]])
    py = {"axis": "y", "axis_index": 1, "at_mm": 0.0}
    px = {"axis": "x", "axis_index": 0, "at_mm": 20.0}
    assert len(am._mirror_images(path, [py])) == 2
    four = am._mirror_images(path, [py, px])
    assert len(four) == 4
    assert {(round(q[0, 0], 6), round(q[0, 1], 6)) for q in four} == {(10, 2), (10, -2), (30, 2), (30, -2)}
    on = am._mirror_images(np.array([[10.0, 0.0, 5.0]]), [py])
    assert len(on) == 1                                  # a point on the plane is its own image
    groups, gaps = am._image_groups(four, reach=3.0)     # 4 mm and 20 mm apart: only the two across y (4 mm < 6 mm) join
    assert sorted(len(g) for g in groups) == [2, 2] and len(gaps) == 6


def test_options_are_checked_and_named():
    opt, probs = am._norm_options({"pitch_mm": -1, "bogus": 1, "parent_check": "maybe", "smooth_mode": "x",
                                   "per_mark": {"3": {"radius_mm": 2, "zzz": 1}}, "skip": [1, "a"]})
    assert opt["pitch_mm"] is None and opt["parent_check"] == "report" and opt["smooth_mode"] == "auto"
    assert opt["per_mark"] == {3: {"radius_mm": 2}} and opt["skip"] == [1]
    text = " ".join(probs)
    for word in ("pitch_mm", "bogus", "parent_check", "smooth_mode", "zzz", "skip"):
        assert word in text


# ---- helpers for the geometry checks ------------------------------------------------------------------------
def far_surface_unchanged(before, after, centres, dist, n=4000, tol=1e-4):
    """Points on the surface of each mesh that lie farther than `dist` from every centre are on the other surface."""
    rng = np.random.default_rng(3)
    centres = np.asarray(centres, float)
    for a, b in ((before, after), (after, before)):
        pts, _ = trimesh.sample.sample_surface(a, n, seed=int(rng.integers(1 << 30)))
        far = np.min(np.linalg.norm(pts[:, None, :] - centres[None], axis=2), axis=1) > dist
        assert far.sum() > 100
        _c, d, _t = trimesh.proximity.closest_point(b, pts[far])
        assert d.max() < tol, d.max()


def top_z(mesh, x, y):
    loc, _i, _t = mesh.ray.intersects_location([[x, y, 30.0]], [[0, 0, -1]])
    return float(loc[:, 2].max())


# ---- remove ----------------------------------------------------------------------------------------------
def test_remove_cuts_the_spur_and_leaves_the_rest(spur):
    run, root = spur.run, spur.root
    before = tree_hash(root / "in", run)
    mesh0 = trimesh.load(str(run / "result" / "final.stl"), force="mesh")
    mk = mark(1, "remove", [[12, 2.5, 10]], radius=2.5)
    rep = am.apply_marks(run, doc_for(run, mk))
    assert rep["ok"] and rep["marks"][0]["status"] == "applied"
    e = rep["marks"][0]
    assert e["modes"] == ["cut"] and e["parent_check"]["within_tolerance"] is True
    pcs = e["steps"][0]["parent_check"]                       # m3: the skeleton test of lib2.parent_check is in the check
    assert {"min_drop_mm", "mean_drop_mm", "max_local_drop_mm", "n_points"} <= set(pcs["skeleton"])
    assert e["parent_check"]["skeleton_max_local_drop_mm"] <= pcs["tolerance_mm"]
    t = rep["totals"]
    assert t["volume_after_mm3"] < t["volume_before_mm3"] - 8                      # the spur (13.6 mm3) is cut
    assert t["change_mm3"] == pytest.approx(e["edit"]["part_volume_change_mm3"], abs=0.01)
    assert t["bodies_after"] == 1 and t["closed_after"] is True
    # the files: a new folder beside the run, the run and the inputs are as they were
    out = Path(rep["out_dir"])
    assert out == run.parent / "a-marks" and (out / "edited.stl").is_file() and (out / "apply_marks.json").is_file()
    assert tree_hash(root / "in", run) == before
    # the part on disk: one closed body after the float32 reload; the spur is gone, the other spur stays
    ed = load_edited(rep)
    assert ed.is_watertight and ed.is_winding_consistent and len(trimesh.graph.connected_components(ed.face_adjacency)) == 1
    assert rep["totals"]["reload_from_disk"]["closed"] and ed.volume == pytest.approx(t["volume_after_mm3"], abs=0.01)
    assert top_z(ed, 12, 2.5) < 8.6 and top_z(mesh0, 12, 2.5) > 10.9
    assert top_z(ed, 12, -2.5) > 10.9
    # nothing changes outside the edit zone
    reach = t["edit_reach_mm"] + 2.5 * rep["grid"]["pitch_mm"] + 0.1
    far_surface_unchanged(mesh0, ed, [[12, 2.5, 10]], reach)
    assert t["triangles_replaced"]["triangles_kept_bit_equal"] > 0
    # the report names the defaults and the values that it does not use
    assert "normals" in rep["values_not_used"] and "amount_mm (the cut mode takes no depth)" in e["values_not_used"]
    assert "cleanup" in rep["values_not_used"]           # the facts of a list candidate are read by nobody here (review n4)
    assert rep["pipeline_defaults"]["remove"]["parent_tol_mm"]["value"] == 0.2
    assert "lib2.py" in rep["pipeline_defaults"]["remove"]["parent_tol_mm"]["anchor"]
    assert json.loads((out / "apply_marks.json").read_text())["format"] == am.APPLY_FORMAT


def test_remove_mirror_overlap_is_one_site(spur):
    mk = mark(1, "remove", [[12, 2.5, 10]], radius=2.5, mirror=True)
    rep = am.apply_marks(spur.run, doc_for(spur.run, mk))
    e = rep["marks"][0]
    assert len(e["steps"]) == 1 and e["mirror"]["combined"] is True and "one site" in e["steps"][0]["label"]
    assert e["mirror"]["plane"]["axis"] == "y" and e["mirror"]["plane"]["source"].startswith("default")
    ed = load_edited(rep)
    assert top_z(ed, 12, 2.5) < 8.6 and top_z(ed, 12, -2.5) < 8.6               # both spurs are gone


def test_mirror_far_apart_is_two_steps_and_the_plane_can_move(tmp_path):
    part = PLATE() + BAR() + cyl_z(12, 3.5, 8, 11, 1.0)
    run, _ = make_run(tmp_path, part)
    mk = mark(1, "remove", [[12, 3.5, 10]], radius=1.0, mirror=True)
    rep = am.apply_marks(run, doc_for(run, mk))
    e = rep["marks"][0]
    assert e["mirror"]["combined"] is False and len(e["steps"]) == 2
    assert e["steps"][0]["status"] == "applied" and e["steps"][1]["status"] == "no_effect"          # no spur at y = -3
    assert "not on the part" in e["steps"][1]["reason"] or "no solid" in e["steps"][1]["reason"]
    assert e["status"] == "partly_applied"
    # the plane of the call: y = 3.5 maps the mark onto itself
    rep2 = am.apply_marks(run, doc_for(run, mk), options={"mirror_plane": {"axis": "y", "at_mm": 3.5}},
                          out_dir=tmp_path / "o2")
    e2 = rep2["marks"][0]
    assert len(e2["steps"]) == 1 and e2["mirror"]["plane"]["at_mm"] == 3.5
    assert e2["mirror"]["plane"]["source"] == "call option mirror_plane"


def test_remove_on_a_flat_face_is_a_pit_and_the_report_says_so(tmp_path):
    """A flat face has no feature that sticks out: the cut digs a pit. The parent check does not see it (it compares the
    medial axis), so the report carries the volume and the thickness, and says that the check does not see a pit."""
    run, _ = make_run(tmp_path, PLATE() + BAR())
    rep = am.apply_marks(run, doc_for(run, mark(1, "remove", [[12, 0, 8]], radius=2.5)))
    e = rep["marks"][0]
    assert e["status"] == "applied" and e["modes"] == ["cut"]
    assert e["edit"]["removed_mm3"] > 10 and e["edit"]["part_volume_change_mm3"] < -10            # the pit is in the volume
    assert e["parent_check"]["within_tolerance"] is True                                         # and the check does not see it
    th = e["thickness_near_mm"]
    assert th["after_median"] < th["before_median"] - 0.3                                        # the member is thinner there
    assert any("does not see a pit" in n for n in rep["notes"])


# ---- add -------------------------------------------------------------------------------------------------
def groove_part(x0=11.25, x1=12.75):
    """Plate + bar with a groove (1.5 wide, 1.5 deep, along z) in the side y = -4."""
    return PLATE() + (BAR() - cube((x0, -4.5, -1), (x1, -2.5, 9)))


def test_add_fills_the_groove_and_uses_the_page_amount(tmp_path):
    run, _ = make_run(tmp_path, groove_part())
    mesh0 = trimesh.load(str(run / "result" / "final.stl"), force="mesh")
    rep = am.apply_marks(run, doc_for(run, mark(1, "add", [[12, -4, 4]], radius=3.0)))
    e = rep["marks"][0]
    assert e["status"] == "applied" and e["edit"]["added_mm3"] > 5 and e["edit"]["removed_mm3"] < 0.5
    assert rep["totals"]["change_mm3"] > 5 and rep["totals"]["closed_after"] and rep["totals"]["bodies_after"] == 1
    step = e["steps"][0]
    assert step["operator"]["neck_offset_is_default"] is False and step["operator"]["neck_offset_mm"] == 1.0   # the page: 1 mm
    ed = load_edited(rep)
    assert ed.is_watertight and ed.volume > mesh0.volume + 5
    # beyond the changed nodes and the zone ring (ZONE_NODES nodes) the surface is the old one; inside the ring it is the
    # marching cubes surface of the same field, which can be 0.01 to 0.1 mm off the smoothed mesh of the run
    far_surface_unchanged(mesh0, ed, [[12, -4, 4]], rep["totals"]["edit_reach_mm"] + 2.5 * rep["grid"]["pitch_mm"] + 0.1)


def test_add_without_amount_names_the_pipeline_default(tmp_path):
    run, _ = make_run(tmp_path, groove_part())
    mk = mark(1, "add", [[12, -4, 4]], radius=3.0)
    mk["amount_mm"] = None
    rep = am.apply_marks(run, doc_for(run, mk))
    e = rep["marks"][0]
    names = {d["name"]: d for d in e["defaults_used"]}
    assert names["neck_offset_mm"]["value"] == 0.5 and "apply_marks3.py" in names["neck_offset_mm"]["anchor"]
    assert e["steps"][0]["operator"]["neck_offset_is_default"] is True
    assert rep["pipeline_defaults"]["add"]["close_mm"]["anchor"].startswith("apply_marks3.py")


def test_frozen_clip_is_a_number(tmp_path):
    """A groove 2.25 mm from the plate: the weight of the edit is 0 within 3 mm and 1 beyond 6 mm."""
    run, _ = make_run(tmp_path, groove_part(2.25, 3.75))
    rep = am.apply_marks(run, doc_for(run, mark(1, "add", [[3, -4, 4]], radius=3.0)))
    e = rep["marks"][0]
    assert e["status"] == "applied"
    ed_ = e["edit"]
    assert ed_["clipped_by_frozen_mm3"] > 5 and ed_["clipped_share_pct"] > 50
    assert ed_["planned_mm3"] == pytest.approx(ed_["added_mm3"] + ed_["removed_mm3"] + ed_["clipped_by_frozen_mm3"], abs=0.02)
    fz = e["steps"][0]["frozen"]
    assert fz["nearest_frozen_to_edit_mm"] >= fz["frozen_keep_mm"] - 0.3 and fz["nearest_frozen_to_mark_mm"] == pytest.approx(3.0, abs=0.7)
    t = rep["totals"]
    assert t["frozen_by_labels"]["clipped_by_frozen_mm3"] == ed_["clipped_by_frozen_mm3"]
    assert t["frozen_exact"]["done"] and t["frozen_exact"]["unchanged"]
    # a smaller keep distance clips less (an option, not a cap)
    rep2 = am.apply_marks(run, doc_for(run, mark(1, "add", [[3, -4, 4]], radius=3.0)), out_dir=tmp_path / "o2",
                          options={"frozen_keep_mm": 1.0, "frozen_full_mm": 2.0})
    assert rep2["marks"][0]["edit"]["clipped_by_frozen_mm3"] < ed_["clipped_by_frozen_mm3"]
    assert rep2["pipeline_defaults"]["add"]["frozen_keep_mm"]["overridden_by_option"] is True


def test_frozen_exact_check_restores_an_edit_inside_a_frozen_region(tmp_path):
    run, _ = make_run(tmp_path, PLATE() + BAR())
    man0 = am._manifold(trimesh.load(str(run / "result" / "final.stl"), force="mesh"))
    man1 = man0 - cube((-1.5, -1, 4), (-0.5, 1, 5))                   # a hole in the fixed plate
    run_json = json.loads((run / "run.json").read_text())
    F, ff = am._load_frozen(run, run_json, None)
    st = type("S", (), {"bodies": 1, "frozen_solid": F, "frozen_facts": ff, "entries": []})()
    facts, restored, failed = am._frozen_exact(st, man0, man1)
    assert facts["done"] and facts["unchanged"] is False and facts["removed_from_frozen_mm3"] == pytest.approx(2.0, abs=0.01)
    assert facts["restored"] is True and restored is not None and failed is False
    assert abs(float(restored[0].volume) - float(man0.volume())) < 0.01
    facts, restored, failed = am._frozen_exact(st, man0, man0)
    assert facts["unchanged"] is True and restored is None and failed is False
    F2, ff2 = am._load_frozen(run, run_json, tmp_path / "missing.yaml")
    assert F2 is None and ff2["done"] is False and "not on disk" in ff2["reason"]
    st2 = type("S", (), {"bodies": 1, "frozen_solid": None, "frozen_facts": ff2, "entries": []})()
    assert am._frozen_exact(st2, man0, man0)[0]["done"] is False


# ---- smooth ----------------------------------------------------------------------------------------------
def smooth_mark(points, radius, strength=1.5, mid=1):
    m = mark(mid, "smooth", points, radius=radius)
    m["strength"] = strength
    return m


def pimple_part():
    pim = None
    for i in range(10):
        c = cyl_z(6 + 1.2 * i, 0.0, 8, 8.4, 0.35, 12)
        pim = c if pim is None else pim + c
    return PLATE() + BAR() + pim


def test_smooth_one_point_is_iso_and_the_surface_gets_smoother(tmp_path):
    run, _ = make_run(tmp_path, pimple_part())
    mesh0 = trimesh.load(str(run / "result" / "final.stl"), force="mesh")
    rep = am.apply_marks(run, doc_for(run, mark(1, "smooth", [[12, 0, 8]], radius=4.0)))
    e = rep["marks"][0]
    assert e["status"] == "applied" and e["modes"] == ["iso"]
    assert "one point" in e["steps"][0]["mode_reason"]
    ed = load_edited(rep)
    assert ed.is_watertight and ed.area < mesh0.area - 2.0                 # the pimples get lower: less surface
    assert 0.5 < mesh0.volume - ed.volume < 6.0
    assert e["steps"][0]["operator"]["strength_used"] == 1.5 and e["steps"][0]["operator"]["sigma_mm"] == 1.5
    reach = rep["totals"]["edit_reach_mm"] + 2.5 * rep["grid"]["pitch_mm"] + 0.1
    far_surface_unchanged(mesh0, ed, [[12, 0, 8]], reach)                      # nothing changes outside the zone (GOAL item 4)
    assert rep["totals"]["triangles_replaced"]["triangles_kept_bit_equal"] > 0


def strut_part():
    strut = cyl_x(0, 6, 0, 24, 2.0)
    for xx in np.arange(4, 22, 3.0):
        strut = strut + cyl_x(0, 6, xx, xx + 1.0, 2.35)                       # rings: a rough strut
    return PLATE() + strut


def test_smooth_stroke_is_fair_and_tube_is_asked_for_only(tmp_path):
    """M4: auto is iso for a point and fair for a stroke. A tube rebuilds the member, so it is an explicit mode."""
    run, _ = make_run(tmp_path, strut_part())
    mk = mark(1, "smooth", [[8, 0, 6], [12, 0, 6], [16, 0, 6]], radius=3.0)
    rep = am.apply_marks(run, doc_for(run, mk))
    e = rep["marks"][0]
    assert e["status"] == "applied" and e["modes"] == ["fair"] and e["steps"][0]["mode_reason"].startswith("auto: a stroke")
    rep2 = am.apply_marks(run, doc_for(run, mk), out_dir=tmp_path / "o2", options={"smooth_mode": "tube"})
    e2 = rep2["marks"][0]
    assert e2["status"] == "applied" and e2["modes"] == ["tube"] and e2["steps"][0]["mode_reason"] == "asked: smooth_mode tube"
    assert e2["steps"][0]["operator"]["arc_mm"] > 20
    assert "tube rebuilds the member" in e2["edit_note"] and "radius_mm is not used" in e2["edit_note"]
    assert e2["edit"]["edit_reach_mm"] > 3 and e2["edit"]["planned_mm3"] > 0           # the reach of a tube is a fact
    assert "smooth_mode tube does not use it" in " ".join(rep2["values_used"]["smooth"])
    ed = load_edited(rep2)
    assert ed.is_watertight and len(trimesh.graph.connected_components(ed.face_adjacency)) == 1
    # the rings are gone: the radius along the strut is the same at x = 10 (a ring) and x = 12 (no ring)
    r10 = np.hypot(*(ed.vertices[(abs(ed.vertices[:, 0] - 10.5) < 0.2)][:, 1:] - [0, 6]).T).max()
    r12 = np.hypot(*(ed.vertices[(abs(ed.vertices[:, 0] - 12.5) < 0.2)][:, 1:] - [0, 6]).T).max()
    assert abs(r10 - r12) < 0.15 and r10 < 2.3
    # radius_mm does not change a tube: the same edit for a small and a large radius
    mk2 = mark(1, "smooth", [[8, 0, 6], [12, 0, 6], [16, 0, 6]], radius=1.0)
    rep3 = am.apply_marks(run, doc_for(run, mk2), out_dir=tmp_path / "o3", options={"smooth_mode": "tube"})
    assert rep3["marks"][0]["edit"]["part_volume_change_mm3"] == pytest.approx(e2["edit"]["part_volume_change_mm3"], abs=0.5)
    rep4 = am.apply_marks(run, doc_for(run, mk), out_dir=tmp_path / "o4", options={"smooth_mode": "fair"})
    assert rep4["marks"][0]["modes"] == ["fair"] and rep4["marks"][0]["steps"][0]["mode_reason"] == "asked: smooth_mode fair"


def test_per_mark_mode_keys_order_and_problems(tmp_path):
    """C3b: a mark can carry smooth_mode and remove_mode. The order: per_mark, then the key of the mark, then the call option."""
    run, _ = make_run(tmp_path, strut_part())
    pts = [[8, 0, 6], [12, 0, 6], [16, 0, 6]]
    plain = mark(1, "smooth", pts, radius=3.0)
    keyed = {**plain, "smooth_mode": "tube"}
    # no key: the old behaviour (auto, a stroke is fair); the entry shows the mode
    rep0 = am.apply_marks(run, doc_for(run, plain), out_dir=tmp_path / "o0")
    assert rep0["marks"][0]["values"]["smooth_mode"] == "auto" and rep0["marks"][0]["modes"] == ["fair"]
    # the key is used, also when the call has another global option
    rep1 = am.apply_marks(run, doc_for(run, keyed), out_dir=tmp_path / "o1", options={"smooth_mode": "iso"})
    e1 = rep1["marks"][0]
    assert e1["values"]["smooth_mode"] == "tube" and e1["modes"] == ["tube"] and not rep1["problems"]
    # per_mark beats the key
    rep2 = am.apply_marks(run, doc_for(run, keyed), out_dir=tmp_path / "o2", options={"per_mark": {1: {"smooth_mode": "fair"}}})
    assert rep2["marks"][0]["values"]["smooth_mode"] == "fair" and rep2["marks"][0]["modes"] == ["fair"]
    # the same key with the same fields as the call option gives the same edit
    rep3 = am.apply_marks(run, doc_for(run, plain), out_dir=tmp_path / "o3", options={"smooth_mode": "tube"})
    assert rep3["marks"][0]["edit"]["part_volume_change_mm3"] == e1["edit"]["part_volume_change_mm3"]
    # a value that is not a mode: a fact, and the call option is used
    rep4 = am.apply_marks(run, doc_for(run, {**plain, "smooth_mode": "bogus"}), out_dir=tmp_path / "o4",
                          options={"smooth_mode": "fair"})
    assert rep4["marks"][0]["values"]["smooth_mode"] == "fair"
    assert any("mark 1: smooth_mode 'bogus' is not one of" in x and "the call option 'fair' is used" in x for x in rep4["problems"])
    # a key of the other type: a fact, and it is not used
    rep5 = am.apply_marks(run, doc_for(run, {**plain, "remove_mode": "shave"}), out_dir=tmp_path / "o5")
    assert rep5["marks"][0]["values"]["remove_mode"] is None and rep5["marks"][0]["values"]["smooth_mode"] == "auto"
    assert any("remove_mode 'shave' is for a remove mark, and this mark is smooth" in x for x in rep5["problems"])
    # a remove mark with remove_mode shave in the file is a shave; the call option cut does not change it
    run2, _ = make_run(tmp_path / "r2", PLATE() + BAR())
    rm = {**mark(1, "remove", [[12, 0, 8]], radius=2.0), "remove_mode": "shave", "amount_mm": 0.6}
    rep6 = am.apply_marks(run2, doc_for(run2, rm), out_dir=tmp_path / "o6", options={"remove_mode": "cut", **FAST})
    assert rep6["marks"][0]["values"]["remove_mode"] == "shave"
    # _norm_mark returns the raw keys and keeps the old ones
    mk, errs = feedback._norm_mark(keyed, 0)
    assert not errs and mk["smooth_mode"] == "tube" and mk["remove_mode"] is None and mk["radius_mm"] == 3.0


def test_radius_and_strength_change_the_size_of_a_smooth_edit(tmp_path):
    """M4 / pitfalls/mark-values-ignored: a value of the mark that the tool reads must change the result."""
    run, _ = make_run(tmp_path, pimple_part())
    vol = {}
    for i, (r, st_) in enumerate(((2.0, 1.0), (4.0, 1.0), (4.0, 0.4), (4.0, 2.5))):
        mk = smooth_mark([[12, 0, 8]], r, st_)
        rep = am.apply_marks(run, doc_for(run, mk), out_dir=tmp_path / f"o{i}", options=FAST)
        e = rep["marks"][0]
        assert e["status"] == "applied" and e["modes"] == ["iso"]
        vol[(r, st_)] = (e["edit"]["planned_mm3"], e["edit"]["edit_reach_mm"])
    assert vol[(4.0, 1.0)][0] > 2 * vol[(2.0, 1.0)][0] and vol[(4.0, 1.0)][1] > vol[(2.0, 1.0)][1] + 1.0   # radius_mm
    assert vol[(4.0, 2.5)][0] > vol[(4.0, 1.0)][0] > vol[(4.0, 0.4)][0] + 0.2                             # strength
    # a stroke (fair): radius_mm and strength change the edit too
    run2, _ = make_run(tmp_path / "s", strut_part())
    path = [[8, 0, 6], [12, 0, 6], [16, 0, 6]]
    res = {}
    for i, (r, st_) in enumerate(((1.5, 1.0), (4.0, 1.0), (4.0, 0.3))):
        rep = am.apply_marks(run2, doc_for(run2, smooth_mark(path, r, st_)), out_dir=tmp_path / f"f{i}",
                             options=FAST)
        e = rep["marks"][0]
        assert e["status"] == "applied" and e["modes"] == ["fair"]
        res[(r, st_)] = (abs(e["edit"]["part_volume_change_mm3"]), e["edit"]["edit_reach_mm"])
    assert res[(4.0, 1.0)][1] > res[(1.5, 1.0)][1]
    assert res[(4.0, 1.0)][0] != pytest.approx(res[(4.0, 0.3)][0], abs=0.05)


def test_auto_does_not_pick_a_tube_and_a_tube_on_a_web_falls_back(tmp_path):
    """The reviewer's case: a stroke on the flat top face of a square bar was a "round strut" for the old pick. Now auto
    gives fair. A tube that is asked for on a wide thin plate (a section that is not bounded) cannot be made: iso, with
    the reason."""
    run, _ = make_run(tmp_path, PLATE() + BAR())
    mk = mark(1, "smooth", [[8, -2, 8], [12, 0, 8], [16, 2, 8]], radius=3.0)
    rep = am.apply_marks(run, doc_for(run, mk), options=FAST)
    assert "tube" not in (rep["marks"][0].get("modes") or []) and rep["marks"][0]["status"] in ("applied", "no_effect")
    if rep["marks"][0]["status"] == "applied":
        assert rep["marks"][0]["modes"] == ["fair"]
    run2, _ = make_run(tmp_path / "w", PLATE() + cube((0, -6, 0), (24, 6, 2)))
    mk2 = mark(1, "smooth", [[8, 0, 2], [12, 0, 2], [16, 0, 2]], radius=3.0)
    rep2 = am.apply_marks(run2, doc_for(run2, mk2), options={**FAST, "smooth_mode": "tube"})
    e = rep2["marks"][0]
    assert "tube" not in (e.get("modes") or [])
    why = e["steps"][0]["mode_reason"]
    assert "smooth_mode tube was asked, and the tube could not be made" in why and "not bounded" in why


# ---- no doubling, rollback, enforce -----------------------------------------------------------------------
def test_shave_mirror_overlap_is_not_edited_twice(tmp_path):
    """Marks at y = 1 and its mirror image y = -1 (radius 3) overlap. One site: the deepest cut is amount x weight (1 mm)."""
    run, _ = make_run(tmp_path, PLATE() + BAR())
    opts = {"remove_mode": "shave"}
    one = am.apply_marks(run, doc_for(run, mark(1, "remove", [[12, 1, 8]], radius=3.0, mirror=True)), options=opts)
    assert len(one["marks"][0]["steps"]) == 1
    z_one = {y: top_z(load_edited(one), 12, y) for y in (-1, 0, 1)}
    assert z_one[1] == pytest.approx(7.0, abs=0.15) and z_one[-1] == pytest.approx(7.0, abs=0.15)
    assert z_one[0] == pytest.approx(7.25, abs=0.15)                      # weight 0.75 at the middle, not 1.5
    # two marks that are not mirrors of each other are two edits: the middle is cut twice (a fact of the file)
    two = am.apply_marks(run, doc_for(run, mark(1, "remove", [[12, 1, 8]], radius=3.0),
                                      mark(2, "remove", [[12, -1, 8]], radius=3.0)), options=opts, out_dir=tmp_path / "o2")
    assert top_z(load_edited(two), 12, 0) < 6.9


def test_a_step_that_splits_the_part_is_rolled_back(tmp_path):
    neck = cube((0, -1, 3), (24, 1, 5))
    run, _ = make_run(tmp_path, PLATE() + neck + cube((20, -2, 0), (24, 2, 8)))
    rep = am.apply_marks(run, doc_for(run, mark(1, "remove", [[10, 0, 4]], radius=3.0)),
                         options={"remove_mode": "shave", "per_mark": {1: {"amount_mm": 1.5}}})
    e = rep["marks"][0]
    assert rep["ok"] and e["status"] == "rolled_back" and e["steps"][0]["rollback"] == "splits_part"
    assert "splits the part" in e["reason"] and rep["totals"]["change_mm3"] == 0.0
    assert rep["files"]["edited_stl"] is None and not (Path(rep["out_dir"]) / "edited.stl").exists()
    assert "route_note" in e and "rerun_with_hints" in e["route_note"]
    assert any(n.startswith("No mark changed the part") for n in rep["notes"])


def test_parent_check_enforce_rolls_back_a_cut_that_fails_it(spur, monkeypatch):
    fake = {"tolerance_mm": 0.2, "n_points": 1, "max_drop_mm": 0.9, "n_over_tolerance": 3, "removed_in_parent_mm3": 1.0,
            "max_depth_into_parent_mm": 0.5, "within_tolerance": False,
            "skeleton": {"n_points": 0, "min_drop_mm": 0.0, "mean_drop_mm": 0.0, "max_local_drop_mm": 0.0}}
    real = am._parent_check
    monkeypatch.setattr(am, "_parent_check", lambda *a, **k: fake)
    mk = mark(1, "remove", [[12, 2.5, 10]], radius=2.5)
    rep = am.apply_marks(spur.run, doc_for(spur.run, mk))                       # report: the edit is made, the check is a fact
    assert rep["marks"][0]["status"] == "applied" and rep["marks"][0]["parent_check"]["within_tolerance"] is False
    rep2 = am.apply_marks(spur.run, doc_for(spur.run, mk), options={"parent_check": "enforce"}, out_dir=spur.root / "o2")
    e = rep2["marks"][0]
    assert e["status"] == "rolled_back" and e["steps"][0]["rollback"] == "parent_check" and "load path" in e["reason"]
    assert real is not am._parent_check


# ---- stale marks, guards, options, errors -------------------------------------------------------------------
def test_stale_marks_are_a_fact_and_allow_stale_applies_them(spur):
    run, root = spur.run, spur.root
    mk = mark(1, "remove", [[12, 2.5, 10]], radius=2.5)
    bad = marks_doc(mk, sha="0" * 64)
    before = tree_hash(root / "in", run)
    rep = am.apply_marks(run, bad)
    assert rep["ok"] is False and rep["marks_match_final_stl"] is False and rep["refused"]["reason"] == "stale_marks"
    assert any("allow_stale" in o for o in rep["refused"]["options"])
    assert not (run.parent / "a-marks").exists() and tree_hash(root / "in", run) == before    # nothing was written
    rep = am.apply_marks(run, bad, allow_stale=True, options=FAST)
    assert rep["ok"] and rep["marks_match_final_stl"] is False and rep["allow_stale"] is True
    assert rep["marks"][0]["status"] == "applied" and any("source_sha256" in p for p in rep["problems"])
    assert short_ok(rep)["marks_match_final_stl"] is False
    # a marks file without a hash is not stale: the fact is None
    rep = am.apply_marks(run, marks_doc(mk), out_dir=root / "o3", options=FAST)
    assert rep["ok"] and rep["marks_match_final_stl"] is None


def short_ok(rep):
    return am.short_report(rep)


def test_voxel_guard_is_a_fact_with_options(spur):
    mk = mark(1, "remove", [[12, 2.5, 10]], radius=2.5)
    rep = am.apply_marks(spur.run, doc_for(spur.run, mk), options={"max_crop_voxels": 1000})
    assert rep["ok"]
    e = rep["marks"][0]
    s = e["steps"][0]
    assert e["status"] == "guard" and s["status"] == "guard" and "above the guard" in s["reason"]
    assert s["facts"]["crop_nodes"] > 1000 and s["facts"]["pitch_that_fits_mm"] > s["facts"]["pitch_mm"]
    assert s["facts"]["memory_estimate_mb"] == round(s["facts"]["crop_nodes"] * am.BYTES_PER_NODE / 1e6) + am.FIXED_MB
    assert "plus about 300 MB" in s["facts"]["memory_estimate_note"]                  # m10: the fixed part of the process
    assert any("pitch_mm" in o for o in s["options"]) and any("fewer" in o for o in s["options"])
    assert rep["files"]["edited_stl"] is None
    # the option the guard names passes the guard (a grid of 2 mm may still not fit the part: the step says so)
    rep2 = am.apply_marks(spur.run, doc_for(spur.run, mk), out_dir=spur.root / "o2",
                          options={"max_crop_voxels": 1000, "pitch_mm": s["facts"]["pitch_that_fits_mm"]})
    assert rep2["marks"][0]["status"] != "guard"
    # the guard on all boxes of one call
    two = doc_for(spur.run, mk, mark(2, "remove", [[12, -2.5, 10]], radius=2.5))
    one_box = am.apply_marks(spur.run, doc_for(spur.run, mk), out_dir=spur.root / "o4")["totals"]["crop_nodes_total"]
    rep3 = am.apply_marks(spur.run, two, out_dir=spur.root / "o3", options={"max_total_voxels": int(one_box * 1.5)})
    assert [m["status"] for m in rep3["marks"]] == ["applied", "guard"]
    assert "steps of this call" in rep3["marks"][1]["reason"]


def test_skip_per_mark_radius_scale_unreadable_and_page_default_radius(spur):
    a = mark(1, "remove", [[12, 2.5, 10]], radius=2.5)
    b = mark(2, "remove", [[12, -2.5, 10]], radius=2.5)
    c = mark(3, "remove", [[12, 0, 8]], radius=2.5)
    del c["radius_mm"]
    bad = {"id": 4, "type": "remove", "points": []}
    rep = am.apply_marks(spur.run, doc_for(spur.run, a, b, c, bad),
                         options={"skip": [2], "radius_scale": 2.0, "per_mark": {"3": {"skip": True}}})
    st = {m["id"]: m for m in rep["marks"]}
    assert st[1]["status"] == "applied" and st[1]["values"]["radius_used_mm"] == 5.0
    assert st[2]["status"] == "skipped_by_option" and st[3]["status"] == "skipped_by_option"
    assert st[3]["defaults_used"][0]["name"] == "radius_mm" and st[3]["defaults_used"][0]["value"] == 4.0
    assert st[4]["status"] == "unreadable" and "points" in st[4]["reason"]
    assert rep["summary"]["remove:applied"] == 1 and rep["summary"]["remove:skipped_by_option"] == 2
    # per_mark radius
    rep2 = am.apply_marks(spur.run, doc_for(spur.run, a), out_dir=spur.root / "o2", options={"per_mark": {1: {"radius_mm": 1.5}}})
    assert rep2["marks"][0]["values"]["radius_used_mm"] == 1.5


def test_out_dir_rules_and_overwrite(spur):
    run, root = spur.run, spur.root
    doc = doc_for(run, mark(1, "remove", [[12, 2.5, 10]], radius=2.5))
    r = am.apply_marks(run, doc, out_dir=run / "inside")
    assert r["ok"] is False and r["refused"]["reason"] == "run_folder"
    r = am.apply_marks(run, doc, out_dir=root / "in")
    assert r["ok"] is False and r["refused"]["reason"] in ("input_folder", "not_empty")
    out = root / "mine"
    assert am.apply_marks(run, doc, out_dir=out, options=FAST)["ok"]
    r = am.apply_marks(run, doc, out_dir=out, options=FAST)
    assert r["ok"] is False and r["refused"]["reason"] == "not_empty"
    assert am.apply_marks(run, doc, out_dir=out, overwrite=True, options=FAST)["ok"]
    other = root / "other"
    other.mkdir()
    (other / "x.txt").write_text("x")
    r = am.apply_marks(run, doc, out_dir=other, overwrite=True)
    assert r["ok"] is False and r["refused"]["reason"] == "not_an_apply_folder" and (other / "x.txt").read_text() == "x"
    # the default folder gets a number when it exists
    assert Path(am.apply_marks(run, doc, options=FAST)["out_dir"]).name == "a-marks"
    assert Path(am.apply_marks(run, doc, options=FAST)["out_dir"]).name == "a-marks-2"
    link = root / "link"
    link.symlink_to(out)
    assert am.apply_marks(run, doc, out_dir=link)["refused"]["reason"] == "symbolic_link"


def test_errors_are_facts_not_crashes(tmp_path, spur):
    assert am.apply_marks(tmp_path / "nope", marks_doc())["ok"] is False
    r = am.apply_marks(spur.run, {"version": 2, "marks": []})
    assert r["ok"] is False and "not a marks file" in r["error"]
    (spur.run / "labels.bin").unlink()
    r = am.apply_marks(spur.run, doc_for(spur.run, mark(1, "remove", [[12, 2.5, 10]])))
    assert r["ok"] is False and "labels.bin" in r["error"]
    shutil.rmtree(spur.run / "result")
    r = am.apply_marks(spur.run, marks_doc(mark(1, "remove", [[12, 2.5, 10]])))
    assert r["ok"] is False and "final.stl" in r["error"]


def test_a_mark_off_the_part_has_no_effect_and_nothing_is_written(spur):
    rep = am.apply_marks(spur.run, doc_for(spur.run, mark(1, "add", [[200, 200, 200]], radius=2.0),
                                           mark(2, "smooth", [[12, 0, 40]], radius=2.0)))
    assert [m["status"] for m in rep["marks"]] == ["no_effect", "no_effect"]
    assert "outside the part" in rep["marks"][0]["reason"]
    assert rep["files"]["edited_stl"] is None and rep["totals"]["change_mm3"] == 0.0
    assert not (Path(rep["out_dir"]) / "edited.stl").exists() and (Path(rep["out_dir"]) / "apply_marks.json").is_file()


def test_order_is_remove_then_smooth_then_add(tmp_path, monkeypatch):
    part = groove_part() + cyl_z(12, 2.5, 8, 11, 1.2)
    run, _ = make_run(tmp_path, part)
    ran = []
    real = am._edit_step
    monkeypatch.setattr(am, "_edit_step", lambda st, site: (ran.append((site["kind"], site["mark_id"])), real(st, site))[1])
    rep = am.apply_marks(run, doc_for(run, mark(1, "add", [[12, -4, 4]], radius=3.0),
                                      mark(2, "smooth", [[6, 0, 8]], radius=2.0),
                                      mark(3, "remove", [[12, 2.5, 10]], radius=2.5)))
    assert ran == [("remove", 3), ("smooth", 2), ("add", 1)]
    start = {m["id"]: m["steps"][0]["volume"]["before_mm3"] for m in rep["marks"]}
    assert start[3] > start[1] + 8                                      # remove ran first: it saw the original volume
    assert "remove, then smooth, then add" in " ".join(rep["notes"])


# ---- fix round 1: the review of C3a (M1 to M4, m2, m7, m9, m12) -----------------------------------------------------
LOWPLATE = lambda: cube((-2, -4, 0), (0, 4, 8))            # noqa: E731  the part is lower than its design domain (z 0..12)


def thin_part(r=0.6):
    """Plate + left block + right block (holds the load tip) joined by a thin strut."""
    return PLATE() + cube((0, -4, 0), (8, 4, 8)) + cube((16, -4, 0), (24, 4, 8)) + cyl_x(0, 4, 8, 16, r, 24)


def dangling_part():
    """A bar with a thin neck (r 0.5) and a 6 x 6 x 3 block on it (108 mm3) that touches no fixed or load region."""
    return PLATE() + BAR() + cyl_z(12, 0, 8, 10, 0.5, 24) + cube((9, -3, 10), (15, 3, 13))


def test_changed_nodes_count_only_what_can_move_the_surface():
    """M1: a change of the field far from the zero level does not move the surface and is not a changed node."""
    p = 0.3
    z = np.arange(40) * p - 6.0
    Tc = np.broadcast_to(z[None, None, :], (3, 3, 40)).astype(np.float32).copy()          # a flat surface at z = 0
    far = Tc.copy()
    far[..., 0:8] += 0.5
    far[..., 32:] -= 0.5
    assert not am.changed_nodes(Tc, far, p).any()
    near = Tc.copy()
    near[..., 18:22] -= 0.2
    assert am.changed_nodes(Tc, near, p)[..., 18:22].any()
    flip = Tc.copy()
    flip[..., 25] = -1.0                                                                  # a sign change counts
    assert am.changed_nodes(Tc, flip, p)[..., 25].all()


def test_crop_edge_text_names_the_face_how_near_and_the_options():
    crop = am.Crop([0, 0, 0], (30, 30, 30), 0.3)
    chg = np.zeros(crop.shape, bool)
    chg[1:5, 10:20, 10:20] = True
    z = np.zeros(crop.shape, np.float32)
    *_x, why = am._splice(None, z, z, crop, chg)
    assert "face x-" in why and "from it" in why and "radius_mm" in why and "rerun_with_hints" in why


def test_a_part_that_does_not_fill_its_domain_takes_marks_on_its_top(tmp_path):
    """M1: the design domain is higher than the part (z 0..12, the part z 0..8). The crop margin is at least the reach of
    the operator and only nodes near the surface count as changed, so a mark on the top face is not rolled back."""
    cases = {
        "iso": (LOWPLATE() + BAR(), mark(1, "smooth", [[12, 0, 8]], radius=4.0), {}),
        "fair": (LOWPLATE() + BAR(), mark(1, "smooth", [[8, 0, 8], [16, 0, 8]], radius=3.0), {"smooth_mode": "fair"}),
        "shave": (LOWPLATE() + BAR(), mark(1, "remove", [[12, 0, 8]], radius=3.0), {"remove_mode": "shave"}),
        "side": (LOWPLATE() + BAR(), mark(1, "smooth", [[12, 4, 4]], radius=4.0), {}),
        "add": (LOWPLATE() + (BAR() - cube((11.25, -1, 7), (12.75, 1, 9))), mark(1, "add", [[12, 0, 7.5]], radius=3.0), {}),
        "cut": (LOWPLATE() + BAR() + cyl_z(12, 0, 7, 9.5, 1.0), mark(1, "remove", [[12, 0, 9.2]], radius=2.0), {}),
    }
    for name, (part, mk, opts) in cases.items():
        run, _ = make_run(tmp_path / name, part)
        rep = am.apply_marks(run, doc_for(run, mk), options={**FAST, **opts})
        e = rep["marks"][0]
        assert e["status"] == "applied", (name, e["status"], e.get("reason"))
        assert rep["totals"]["closed_after"] and rep["totals"]["bodies_after"] == 1, name
        assert "crop_edge" not in json.dumps(rep["marks"]), name


@pytest.mark.parametrize("name,mk,opts", [
    ("cut", mark(1, "remove", [[12, 0, 4.6]], radius=2.0), {}),
    ("shave", mark(1, "remove", [[12, 0, 4.6]], radius=2.0), {"remove_mode": "shave"}),
    ("iso", mark(1, "smooth", [[12, 0, 4.6]], radius=2.0), {})])
def test_a_step_that_severs_a_thin_strut_is_rolled_back_with_the_volumes(tmp_path, name, mk, opts):
    """M2: the coarse box mesh has no vertex near the load tip, so the old vertex test dropped the whole right block (512 mm3)
    as a loose body and the frozen restore failed. Now a body that holds a fixed or load voxel is not dropped, a body
    above the sliver limit is not dropped, and the step is rolled back with the volume of each body."""
    run, _ = make_run(tmp_path, thin_part())
    rep = am.apply_marks(run, doc_for(run, mk), options=opts)
    e = rep["marks"][0]
    assert e["status"] == "rolled_back" and e["steps"][0]["rollback"] == "splits_part", (e["status"], e.get("reason"))
    bodies = e["steps"][0]["facts"]["bodies"]
    assert bodies[0]["volume_mm3"] == pytest.approx(800, abs=10) and bodies[1]["volume_mm3"] == pytest.approx(513, abs=8)
    assert all(b["holds_fixed_or_load"] for b in bodies[:2]) and "holds fixed or load" in e["reason"]
    assert rep["totals"]["change_mm3"] == 0.0 and rep["files"]["edited_stl"] is None


def test_a_dangling_body_is_dropped_only_below_the_sliver_limit(tmp_path, monkeypatch):
    """M2: a dangling block of 108 mm3 that touches no fixed or load region is not dropped silently. The option
    loose_body_mm3 names the volume that may go; the slivers of the boolean (no volume) are dropped and reported."""
    run, _ = make_run(tmp_path, dangling_part())
    mk = mark(1, "remove", [[12, 0, 9]], radius=1.5)
    opts = {"remove_mode": "shave", **FAST}
    rep = am.apply_marks(run, doc_for(run, mk), options=opts)
    e = rep["marks"][0]
    assert e["status"] == "rolled_back" and e["steps"][0]["rollback"] == "splits_part"
    loose = [b for b in e["steps"][0]["facts"]["bodies"] if not b["holds_fixed_or_load"] and b["volume_mm3"] > 50]
    assert len(loose) == 1 and loose[0]["volume_mm3"] == pytest.approx(108.0, abs=1.0)
    assert e["steps"][0]["facts"]["loose_body_limit_mm3"] < 1.0
    rep2 = am.apply_marks(run, doc_for(run, mk), options={**opts, "loose_body_mm3": 200}, out_dir=tmp_path / "o2")
    e2 = rep2["marks"][0]
    assert e2["status"] == "applied" and e2["dropped_bodies"]["total_volume_mm3"] == pytest.approx(108.0, abs=1.0)
    assert rep2["totals"]["bodies_after"] == 1
    line = am.short_report(rep2)["marks"][0]
    assert line["dropped_bodies"]["count"] >= 1                                      # the mark line of the short reply
    # a smooth on the neck. With the sewn seam (the default) the seam makes no sliver bodies at all. With the sewn seam
    # switched off the plain boolean leaves slivers (no volume): they are dropped, and the report says how many
    sm = doc_for(run, mark(1, "smooth", [[12, 0.5, 9]], radius=2.0))
    rep3 = am.apply_marks(run, sm, options=FAST, out_dir=tmp_path / "o3")
    e3 = rep3["marks"][0]
    assert e3["status"] == "applied" and "dropped_bodies" not in e3 and rep3["totals"]["bodies_after"] == 1
    assert e3["steps"][0]["seam"]["method"] == "sewn"
    monkeypatch.setattr(am, "_graft", lambda *a, **k: (None, None, "the sewn seam is switched off in this test"))
    rep4 = am.apply_marks(run, sm, options=FAST, out_dir=tmp_path / "o4")
    e4 = rep4["marks"][0]
    assert e4["status"] == "applied" and e4["steps"][0]["seam"]["method"] == "boolean"
    assert e4["dropped_bodies"]["total_volume_mm3"] < e4["dropped_bodies"]["limit_mm3"]


def carving_operator(monkeypatch, radius):
    """An operator that carves a sphere (centre x = -1, in the fixed plate, a frozen region): what a clip that fails would do."""
    def fake(cx):
        pts = cx.crop.points().reshape(cx.crop.shape + (3,))
        dist = np.linalg.norm(pts - np.array([-1.0, 0.0, 6.0]), axis=-1)
        Tn = np.maximum(cx.Tc, (radius - dist)).astype(np.float32)
        return am.OpResult(lambda frozen: Tn, "iso", {"fake": True})
    monkeypatch.setattr(am, "op_iso", fake)


def test_a_change_inside_a_frozen_region_is_undone_in_the_step(tmp_path, monkeypatch):
    """M2: the exact frozen check runs in each step; the part of the edit inside the regions of the YAML is undone."""
    run, _ = make_run(tmp_path, PLATE() + BAR())
    carving_operator(monkeypatch, 0.9)                                    # a cavity of 3.05 mm3 inside the plate
    rep = am.apply_marks(run, doc_for(run, mark(1, "smooth", [[-1, 0, 6]], radius=3.0)), options=FAST)
    e = rep["marks"][0]
    assert e["status"] == "applied", (e["status"], e.get("reason"))
    assert e["steps"][0]["frozen_restored_mm3"] == pytest.approx(3.05, abs=0.4)
    assert e["edit"]["restored_in_frozen_mm3"] > 2.5 and "frozen region" in e["frozen_note"]
    assert rep["totals"]["restored_mm3"] == e["edit"]["restored_in_frozen_mm3"]
    fe = rep["totals"]["frozen_exact"]
    assert fe["unchanged"] is True and fe["restored_by_steps_mm3"] > 2.5
    assert abs(rep["totals"]["change_mm3"]) < 0.05 and rep["totals"]["bodies_after"] == 1
    ed = load_edited(rep)
    assert ed.is_watertight and len(trimesh.graph.connected_components(ed.face_adjacency)) == 1
    assert ed.volume == pytest.approx(float(trimesh.load(str(run / "result" / "final.stl"), force="mesh").volume), abs=0.05)


def test_an_undo_that_leaves_a_cavity_is_rolled_back_and_the_report_says_cavity(tmp_path, monkeypatch):
    """A carve that reaches the bar: the undo refills the plate, and the part of the carve that is in the bar is a cavity."""
    run, _ = make_run(tmp_path, PLATE() + BAR())
    carving_operator(monkeypatch, 1.5)
    rep = am.apply_marks(run, doc_for(run, mark(1, "smooth", [[-1, 0, 6]], radius=3.0)), options=FAST)
    e = rep["marks"][0]
    assert e["status"] == "rolled_back" and e["steps"][0]["rollback"] == "splits_part"
    assert "a cavity inside the part" in e["reason"]
    assert any(b["cavity"] and not b["holds_fixed_or_load"] for b in e["steps"][0]["facts"]["bodies"])


def test_a_frozen_restore_that_fails_rolls_the_step_back(tmp_path, monkeypatch):
    run, _ = make_run(tmp_path, PLATE() + BAR())
    carving_operator(monkeypatch, 0.9)
    real = am._reload_facts
    monkeypatch.setattr(am, "_reload_facts", lambda mesh: ({**real(mesh)[0], "closed": False}, mesh))
    rep = am.apply_marks(run, doc_for(run, mark(1, "smooth", [[-1, 0, 6]], radius=3.0)), options=FAST)
    e = rep["marks"][0]
    assert e["status"] == "rolled_back" and e["steps"][0]["rollback"] == "frozen_restore_failed"
    assert "frozen_keep_mm" in e["reason"] and "rerun_with_hints" in e["reason"]
    assert rep["totals"]["change_mm3"] == 0.0 and rep["files"]["edited_stl"] is None


def test_the_a4a_layout_with_a_plane_at_y5_is_read(tmp_path):
    """M3: run.json mirror is {planes: [{axis, at_mm, ...}], ...} (CONTRACTS 26.1); the plane at y = 5 maps y 2 to 8."""
    part = PLATE() + cube((0, -4, 0), (24, 10, 8)) + cyl_z(12, 2.0, 8, 11, 1.2) + cyl_z(12, 8.0, 8, 11, 1.2)
    run, _ = make_run(tmp_path, part)
    rj = json.loads((run / "run.json").read_text())
    rj["mirror"] = {"planes": [{"axis": "y", "at_mm": 5.0, "align": "auto", "parity": "face", "k": 10, "plane_index": 5}],
                    "design_voxels": 1000, "load_asymmetry": [], "end": {}}
    (run / "run.json").write_text(json.dumps(rj))
    rep = am.apply_marks(run, doc_for(run, mark(1, "remove", [[12, 2.0, 10]], radius=2.5, mirror=True)), options=FAST)
    e = rep["marks"][0]
    assert e["status"] == "applied" and e["mirror"]["plane"] == {"axis": "y", "at_mm": 5.0, "source": "run.json (mirror)"}
    ed = load_edited(rep)
    assert top_z(ed, 12, 2.0) < 8.6 and top_z(ed, 12, 8.0) < 8.6                    # the image at y = 8 is cut too
    # the default plane (y = 0) would have put the image at y = -2, where there is no spur
    rj["mirror"] = None
    (run / "run.json").write_text(json.dumps(rj))
    rep2 = am.apply_marks(run, doc_for(run, mark(1, "remove", [[12, 2.0, 10]], radius=2.5, mirror=True)), options=FAST,
                          out_dir=tmp_path / "o2")
    assert top_z(load_edited(rep2), 12, 8.0) > 10.9


def test_two_planes_give_the_mark_its_images_in_the_group_of_the_reflections(tmp_path):
    spur4 = PLATE() + BAR() + cyl_z(6, 2.5, 8, 11, 1.2) + cyl_z(6, -2.5, 8, 11, 1.2) \
        + cyl_z(18, 2.5, 8, 11, 1.2) + cyl_z(18, -2.5, 8, 11, 1.2)
    run, _ = make_run(tmp_path, spur4)
    rj = json.loads((run / "run.json").read_text())
    rj["mirror"] = {"planes": [{"axis": "y", "at_mm": 0.0}, {"axis": "x", "at_mm": 12.0}]}
    (run / "run.json").write_text(json.dumps(rj))
    rep = am.apply_marks(run, doc_for(run, mark(1, "remove", [[6, 2.5, 10]], radius=2.5, mirror=True)), options=FAST)
    e = rep["marks"][0]
    assert e["status"] == "applied" and len(e["mirror"]["planes"]) == 2 and e["mirror"]["images"] == 4
    assert e["mirror"]["sites"] == 2 and e["mirror"]["images_per_site"] == [2, 2] and len(e["steps"]) == 2
    ed = load_edited(rep)
    assert all(top_z(ed, x, y) < 8.6 for x in (6, 18) for y in (2.5, -2.5))          # all four spurs are cut
    assert any("group of the reflections" in n for n in rep["notes"])


def test_the_neck_threshold_is_shared_by_the_mirror_images(tmp_path):
    """m8: the pipeline counts both mirror sides against the 0.4 g threshold; a site that holds one of two images gets half."""
    part = PLATE() + (BAR() - cube((11.25, -4.5, -1), (12.75, -2.5, 9)) - cube((11.25, 2.5, -1), (12.75, 4.5, 9)))
    run, _ = make_run(tmp_path, part)
    rep = am.apply_marks(run, doc_for(run, mark(1, "add", [[12, -4, 4]], radius=3.0, mirror=True)), options=FAST)
    e = rep["marks"][0]
    assert len(e["steps"]) == 2 and e["mirror"]["images_per_site"] == [1, 1]
    for stp in e["steps"]:
        assert stp["operator"]["neck_min_mm3_used"] == pytest.approx(0.5 * am.DEFAULTS["neck_min_mm3"][0], abs=0.01)


ABS_PATH = re.compile(r"(?<![\w.:/])/(?:[^\s\"',;()\[\]]+/)+[^\s\"',;()\[\]]+|(?:\.\./){3,}")


def test_the_report_file_has_no_absolute_path(tmp_path):
    """m2: a path inside a sentence (the YAML of a moved run is gone) and a path that climbs to the root are scrubbed."""
    part = PLATE() + BAR() + cyl_z(12, 2.5, 8, 11, 1.2)
    run, ypath = make_run(tmp_path / "w", part)
    doc = doc_for(run, mark(1, "remove", [[12, 2.5, 10]], radius=2.5))
    rep = am.apply_marks(run, doc, options=FAST)
    text = (Path(rep["out_dir"]) / "apply_marks.json").read_text()
    assert not ABS_PATH.findall(text), ABS_PATH.findall(text)
    moved = tmp_path / "m" / "a"
    shutil.copytree(run, moved)
    shutil.rmtree(tmp_path / "w" / "in")
    rep2 = am.apply_marks(moved, doc, options=FAST)
    f = Path(rep2["out_dir"]) / "apply_marks.json"
    assert not ABS_PATH.findall(f.read_text()), ABS_PATH.findall(f.read_text())
    fe = json.loads(f.read_text())["totals"]["frozen_exact"]
    assert fe["done"] is False and "not on disk" in fe["reason"] and ypath.name in fe["reason"]
    assert str(tmp_path) not in f.read_text()
    out, root = Path("/x/y/o"), Path("/x/y")
    assert am._scrub_absolute("see /x/y/in/b.yaml.", out, root) == "see ../in/b.yaml."
    assert am._scrub_absolute("the file /app" + "/data/in/b.yaml is gone", out, root) == "the file b.yaml is gone"
    assert am._scrub_absolute("a/b and/or 1/2 and https://x.org/y", out, root) == "a/b and/or 1/2 and https://x.org/y"


@pytest.mark.parametrize("amount", [1.0, 2.0, 3.0])
def test_shave_moves_the_surface_in_by_amount_at_the_mark(tmp_path, amount):
    """m9: the weight comes from the distance in the tangent plane, so the surface at the mark moves in by amount_mm (to
    within a node), and the facts give the depth that was reached."""
    run, _ = make_run(tmp_path, PLATE() + BAR())
    rep = am.apply_marks(run, doc_for(run, mark(1, "remove", [[12, 0, 8]], radius=4.0)),
                         options={"remove_mode": "shave", "per_mark": {1: {"amount_mm": amount}}})
    e = rep["marks"][0]
    assert e["status"] == "applied"
    depth = 8.0 - top_z(load_edited(rep), 12.01, 0.013)
    assert depth == pytest.approx(amount, abs=rep["grid"]["pitch_mm"])               # within one node: the field is exact at the surface layer only
    op = e["steps"][0]["operator"]
    assert op["depth_reached_mm"] == pytest.approx(depth, abs=0.1) and "half a node" in op["depth_note"]
    assert op["depth_at_mark_mm"] == pytest.approx(amount, abs=0.01)                  # the planned depth is the amount


def test_odd_rays_are_recast_and_the_columns_that_stay_odd_roll_the_step_back(tmp_path, monkeypatch):
    """m12: a ray that hits the surface an odd number of times is cast again with a small shift; a column that stays odd is
    counted, and above max_odd_rays the step is rolled back with the fact."""
    box = trimesh.creation.box((6, 6, 6))
    faces = np.asarray(box.faces)
    top = np.nonzero(box.face_normals[:, 2] > 0.9)[0][0]
    open_box = trimesh.Trimesh(box.vertices, np.delete(faces, top, axis=0), process=False)
    crop = am.Crop([-4, -4, -4], (30, 30, 30), 0.3)
    st: dict = {}
    am.signed_distance(open_box, crop, stats=st)
    assert st["odd_rays"] > 0 and st["rays"] > 0                                     # a hole: the columns under it stay odd
    clean: dict = {}
    am.signed_distance(box, crop, stats=clean)
    assert clean["odd_rays"] == 0
    # in a call: the step is rolled back above the limit, and applied at or below it
    run, _ = make_run(tmp_path, PLATE() + BAR())
    real = am.signed_distance

    def odd3(mesh, crop, cap=am.BAND_MM, stats=None):
        out = real(mesh, crop, cap=cap, stats=stats)
        if stats is not None:
            stats["odd_rays"] = 3
        return out
    monkeypatch.setattr(am, "signed_distance", odd3)
    mk = mark(1, "remove", [[12, 0, 8]], radius=3.0)
    rep = am.apply_marks(run, doc_for(run, mk), options={"remove_mode": "shave"})
    e = rep["marks"][0]
    assert e["status"] == "rolled_back" and e["steps"][0]["rollback"] == "odd_rays" and "max_odd_rays" in e["reason"]
    rep2 = am.apply_marks(run, doc_for(run, mk), options={"remove_mode": "shave", "max_odd_rays": 5}, out_dir=tmp_path / "o2")
    assert rep2["marks"][0]["status"] == "applied" and rep2["marks"][0]["steps"][0]["odd_rays"] == 3


def test_the_arguments_after_marks_are_keyword_only():
    """m13: the module function takes run_dir and marks, then keywords (the MCP tool has marks first: the family order)."""
    import inspect
    kinds = {n: p.kind for n, p in inspect.signature(am.apply_marks).parameters.items()}
    assert [n for n, k in kinds.items() if k == inspect.Parameter.POSITIONAL_OR_KEYWORD] == ["run_dir", "marks"]
    assert all(k == inspect.Parameter.KEYWORD_ONLY for n, k in kinds.items() if n not in ("run_dir", "marks"))
    with pytest.raises(TypeError):
        am.apply_marks("a", {}, "out")


def test_the_pinch_repair_moves_vertices_into_their_own_solid_in_passes():
    """m4: as f32_fix of the pipeline: of the vertices with one float32 position the first stays and the j-th other moves
    2e-4 x j mm against its normal (into the solid); up to 6 passes."""
    box = trimesh.creation.box((4, 4, 4))
    v = np.vstack([box.vertices, box.vertices[:1]])
    f = np.asarray(box.faces).copy()
    one = np.nonzero((f == 0).any(axis=1))[0]
    f[one[0]][f[one[0]] == 0] = len(box.vertices)                       # a second vertex index at the position of vertex 0
    pinched = trimesh.Trimesh(v, f, process=False)
    out, passes, left = am._separate_pinched(pinched)
    assert out is not None and passes >= 1 and left == 0
    assert np.allclose(out.vertices[:len(box.vertices)][1:], box.vertices[1:].astype(np.float32).astype(float))   # the rest stays
    moved = out.vertices[-1] - pinched.vertices[-1]
    assert 0 < np.linalg.norm(moved) < 1e-3 and float(moved @ np.sign(pinched.vertices[-1])) < 0       # inward at a corner
    assert np.allclose(out.vertices[0], pinched.vertices[0])                                            # rank 0 stays
    assert am._separate_pinched(box)[0] is None


def test_the_private_helpers_of_feedback_that_apply_marks_reads_exist():
    """The helpers of feedback.py (#28) are read as fb.<name> at call time: a rename shows here, not in a rare path."""
    src = (REPO / "topopt" / "apply_marks.py").read_text()
    names = sorted(set(re.findall(r"\bfb\.(\w+)", src)))
    assert len(names) >= 18, names
    missing = [n for n in names if not hasattr(feedback, n)]
    assert not missing, missing


# ---- B-094: the stress near a mark ----------------------------------------------------------------------------
def test_stress_facts_near_a_mark_come_from_stress_npz(spur):
    centers = np.array([[12, 2.5, 8.5], [12.5, 2.5, 9], [20, 0, 4], [3, 0, 4]], float)
    vm = np.array([0.8, 1.6, 5.0, 0.2])
    np.savez(spur.run / "result" / "stress.npz", centers=centers, rho=np.ones(4, np.float32), vm_max=vm,
             case_names=np.array(["down"]), vm_case_0=vm, E=np.float64(3500), nu=np.float64(0.35), h=np.float64(1.0))
    rep = am.apply_marks(spur.run, doc_for(spur.run, mark(1, "remove", [[12, 2.5, 10]], radius=2.5)))
    sn = rep["marks"][0]["stress_near_mark"]
    assert sn["voxels_near"] == 2 and sn["near_max_mpa"] == 1.6 and sn["global_max_mpa"] == 5.0
    assert sn["near_max_by_case_mpa"] == {"down": 1.6} and "before the edit" in sn["unit"] and "voxel estimate" in sn["unit"]
    assert am.short_report(rep)["marks"][0]["stress_near_max_mpa"] == 1.6
    # no file: a note, no error
    (spur.run / "result" / "stress.npz").unlink()
    rep = am.apply_marks(spur.run, doc_for(spur.run, mark(1, "remove", [[12, 2.5, 10]], radius=2.5)), out_dir=spur.root / "o2")
    assert rep["ok"] and rep["marks"][0]["stress_near_mark"] is None and any("stress.npz" in n for n in rep["notes"])


# ---- the short report, the script and the MCP tool ------------------------------------------------------------
def test_short_report_and_script(spur, capsys):
    mk = mark(1, "remove", [[12, 2.5, 10]], radius=2.5)
    rep = am.apply_marks(spur.run, doc_for(spur.run, mk))
    sh = am.short_report(rep)
    line = sh["marks"][0]
    assert line["status"] == "applied" and line["parent_check_within_tolerance"] is True and line["volume_change_mm3"] < -8
    assert line["edit_reach_mm"] > 2 and "parent_check_pass" not in line         # N5: the reach of each mark; N2: no verdict-like key
    assert sh["detail"] is False and sh["report_file"] == "apply_marks.json" and {r["id"] for r in sh["routes"]} >= {"use_edited_stl", "rerun_with_hints"}
    assert "verdict" not in json.dumps(sh).lower() and "KEEP" not in json.dumps(sh) and "REJECT" not in json.dumps(sh)
    assert len(json.dumps(sh)) < 5000
    marks_file = spur.root / "marks.json"
    marks_file.write_text(json.dumps(doc_for(spur.run, mk)))
    assert am.main([str(spur.run), str(marks_file), "--out", str(spur.root / "cli"), "--pitch", "0.3", "--mirror-plane", "y=0"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["marks"][0]["status"] == "applied" and (spur.root / "cli" / "edited.stl").is_file()
    assert am.main([str(spur.run), str(marks_file), "--out", str(spur.root / "cli")]) == 1           # not empty, no --overwrite


@pytest.fixture
def ws(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    server._current.clear()
    yield tmp_path
    server._current.clear()


def test_mcp_tool_text_follows_the_freedom_rule():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    text = tools["apply_marks"].description or ""
    assert 100 < len(text) <= 4096
    assert not re.search(r"\b(must|never|always|forbidden)\b", text, re.I)
    assert "NEW folder" in text and "no verdict" in text
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert "apply_marks" in server.INSTRUCTIONS and len(server.INSTRUCTIONS) <= OLD_CLIENT_CUT
    assert not re.search(r"\b(must|never)\b", server.INSTRUCTIONS, re.I)   # #380: the refine paragraph moved to the descriptions; the whole text is checked


def test_mcp_tool_on_the_fixture_run(ws):
    run, _ = make_run(ws, spur_part())
    marks = doc_for(run, mark(1, "remove", [[12, 2.5, 10]], radius=2.5, mirror=True))
    (ws / "marks.json").write_text(json.dumps(marks))
    before = tree_hash(ws / "in", run)
    r = server.apply_marks("marks.json", str(run), options=FAST)
    assert r["ok"] and r["marks"][0]["status"] == "applied" and r["detail"] is False
    assert r["out_dir_rel"] == "runs/a-marks" and r["edited_rel"] == "runs/a-marks/edited.stl" and r["report_rel"].endswith("apply_marks.json")
    assert r["totals"]["bodies_after"] == 1 and r["totals"]["reload_from_disk"]["closed"] is True
    sq = r["totals"]["surface_quality"]                                  # C3a-fix: the facts of topopt verify, before and after
    assert sq["after"]["closed_manifold"] is True and sq["before"]["fold_edges"] == 0 and sq["verify_edge_limit_mm"] == 0.05
    assert tree_hash(ws / "in", run) == before
    json.dumps(r)
    # JSON text, detail, the default run (the last run of the server is none: the newest run folder of the work folder)
    full = server.apply_marks(json.dumps(marks), str(run), out_dir="runs/b", detail=True, options=FAST)
    assert full["ok"] and "option_catalog" in full and full["marks"][0]["steps"][0]["crop"]["nodes"] > 0
    stale = server.apply_marks(json.dumps(marks_doc(mark(1, "remove", [[12, 2.5, 10]]), sha="1" * 64)), str(run), out_dir="runs/c")
    assert stale["ok"] is False and stale["refused"]["reason"] == "stale_marks"
    assert server.apply_marks("nope.json", str(run))["ok"] is False
