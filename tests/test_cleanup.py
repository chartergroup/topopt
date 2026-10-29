"""Tests for topopt/cleanup.py: the list of cleanup candidates as a marks file (CONTRACTS.md section 29.3).

Each test makes a small run folder (the real RunWriter, no optimizer) and puts a synthetic finished part into result/final.stl.
The parts are built here from boxes and cylinders, so no binary fixture is in the repository. A run of the list takes 1 to 2 s.

The fixtures (all on the cantilever of `test_feedback.make_case`: a plate at x -2..0, a tip load at x 21..24):
- POSTS: a thin bar (z 0..4) with a tall post at x 8 (z 4..14) and a short post at x 16 (z 4..12). Both are dead-end spurs.
- WEB: the thin bar with posts at x 5 and 16. The post at x 5 is part of the web; the bar is listed as a flat member (FAIR).
- BUMP: the thick bar with a post on top (a spur, confidence high) and a thin fin below it (a bump, near the frozen plate: low).
- BODY: the thick bar with a loose cube 1.5 mm above it (a body that holds no fixed and no load voxel), inside the design box.

Phase test (REVIEW Q8): the outcome of each fixture is asserted at three sub-voxel shifts of the whole case (the part AND its
regions move), so a one-ulp difference on another platform cannot flip it. The sizes are asserted with a tolerance, because the
skeleton and the opening residue move with the lattice phase; the kinds, the operators and the counts are exact.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import manifold3d as m3d
import numpy as np
import pytest
import trimesh
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import apply_marks as am  # noqa: E402
from topopt import cleanup as cl  # noqa: E402
from topopt import feedback as fb  # noqa: E402
from topopt import ratchet as rt  # noqa: E402
from test_apply_marks import BAR, PLATE, cube, cyl_z, to_stl  # noqa: E402
from test_feedback import box, fake_run, make_case, tree_hash  # noqa: E402

M = m3d.Manifold
PHASES = [(0.0, 0.0, 0.0), (0.13, 0.0, 0.07), (0.31, 0.0, 0.19)]       # sub-voxel shifts of the case (pitch 0.5)


def THIN():
    return cube((0, -4, 0), (24, 4, 4))


def posts_part():
    return PLATE() + THIN() + cyl_z(8, 0, 4, 14, 2.5) + cyl_z(16, 0, 4, 12, 2.5)


def web_part():
    return PLATE() + THIN() + cyl_z(5, 0, 4, 12, 2.5) + cyl_z(16, 0, 4, 12, 2.5)


def bump_part():
    return PLATE() + BAR() + cube((4, -1, -4), (16, 1, 0)) + cyl_z(12, 0, 8, 16, 2.5)


def body_part():
    return PLATE() + BAR() + cube((4, -2, 9.5), (8, 2, 11.5))


def cyl_y(x, z, r, y0, y1):
    return M.cylinder(y1 - y0, r, r, 48).rotate((-90, 0, 0)).translate((x, y0, z))


def rods_part():
    """The thick bar with a round rod on each side (r 1.2, 8 mm long): round, so the cut of a mark leaves a clean seam."""
    return PLATE() + BAR() + cyl_y(12, 4, 1.2, 3.9, 12.0) + cyl_y(8, 4, 1.2, -12.0, -3.9)


def run_at(root, part, shift=(0.0, 0.0, 0.0), mirror_planes=None):
    """A run folder whose final.stl is `part`; the part and the three region STLs of the case are moved by `shift`."""
    root = Path(root)
    ypath = make_case(root / "in")
    for n in ("design", "plate", "tip"):
        p = root / "in" / f"{n}.stl"
        mesh = trimesh.load(str(p), force="mesh")
        mesh.apply_translation(shift)
        mesh.export(str(p))
    run = fake_run(root / "runs" / "r", ypath)
    to_stl(part.translate(tuple(shift)), run / "result" / "final.stl")
    if mirror_planes:
        rj = json.loads((run / "run.json").read_text())
        rj["mirror"] = {"planes": mirror_planes}
        (run / "run.json").write_text(json.dumps(rj))
    return run, ypath


def listing(run, ypath, out, **options):
    rep = cl.cleanup_list(run, out_dir=out, yaml_path=ypath, overwrite=True, options=options or None)
    assert rep.get("ok"), rep.get("error")
    return rep


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def posts(tmp_path_factory):
    run, y = run_at(tmp_path_factory.mktemp("posts"), posts_part())
    return run, y


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    return run_at(tmp_path_factory.mktemp("web"), web_part())


@pytest.fixture(scope="module")
def bump(tmp_path_factory):
    return run_at(tmp_path_factory.mktemp("bump"), bump_part())


# --- the tables and the module ----------------------------------------------------------------------------------------

def test_tables_have_an_anchor_for_every_number_and_the_agreed_defaults():
    for name, table in (("THRESHOLDS", cl.THRESHOLDS), ("METHOD", cl.METHOD)):
        for key, (value, unit, anchor) in table.items():
            assert isinstance(value, (int, float)) and unit and len(anchor) > 10, (name, key)
            assert "none in the pipeline" in anchor or any(s in anchor for s in (".py", "pipeline")), (name, key, anchor)
    assert not set(cl.THRESHOLDS) & set(cl.METHOD)                             # a name is a limit or a constant, never both
    assert "near_junction_factor" not in cl.THRESHOLDS and "near_junction_factor" not in cl.METHOD     # C8
    assert cl.THRESHOLDS["max_candidates"][0] == 10 and cl.THRESHOLDS["pitch_mm"][0] == 0.5
    # the mass limits of the pipeline are volumes at 0.0078 g/mm3 (geometric sizes)
    assert cl.THRESHOLDS["spur_min_volume_mm3"][0] == pytest.approx(0.4 / 0.0078)
    assert cl.THRESHOLDS["member_min_volume_mm3"][0] == pytest.approx(0.8 / 0.0078)
    assert cl.THRESHOLDS["bump_min_final_volume_mm3"][0] == pytest.approx(0.2 / 0.0078)
    assert cl.THRESHOLDS["spur_min_length_mm"][0] == 5.0 and cl.THRESHOLDS["opening_radius_mm"][0] == 1.5
    assert cl.THRESHOLDS["web_flatness"][0] == am.DEFAULTS["web_flatness"][0]            # read from apply_marks, not copied
    assert cl.RATCHET_FILES == (rt.LOG_FILE, rt.STATE_FILE)


def test_the_module_loads_no_heavy_package_and_nothing_on_the_run_path_imports_it():
    code = ("import sys; import topopt.cleanup; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('vtk', 'gmsh', 'pyopencl', 'skfem', 'topopt_mcp')); "
            "print(bad)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
    importers = []
    for path in list((REPO / "topopt").rglob("*.py")) + list((REPO / "topopt_mcp").rglob("*.py")):
        if path.name == "cleanup.py" and path.parent.name == "topopt":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [(node.module or "")] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any(n == "topopt.cleanup" or n == "cleanup" or n.endswith(".cleanup") for n in names):
                importers.append(str(path.relative_to(REPO)))
    # the run path (optimizer, post, cli, ...) never imports it; only the MCP server may, inside a function (checked in test_cleanup_mcp)
    assert set(importers) <= {"topopt_mcp/server.py"}, importers
    for name in ("optimize.py", "post.py", "cli.py", "mma.py", "snapshot.py", "runjob.py"):
        assert "cleanup" not in (REPO / "topopt" / name).read_text().replace("cleanup_", "").replace("_cleanup", ""), name


# --- the list of a spur ---------------------------------------------------------------------------------------------------

def test_spur_list_file_has_the_facts_of_each_candidate(posts, tmp_path):
    run, y = posts
    before = tree_hash(run)
    rep = listing(run, y, tmp_path / "out")
    assert tree_hash(run) == before                                            # the run folder stays as it is
    marks_file = tmp_path / "out" / "marks.json"
    doc = json.loads(marks_file.read_text())
    assert (doc["version"], doc["units"]) == (1, "mm") and doc["source_sha256"] == sha(run / "result" / "final.stl")
    assert [m["id"] for m in doc["marks"]] == [1, 2] and rep["counts"]["listed"] == 2 and rep["counts"]["found"] == 2
    big, small = doc["marks"]                                                  # remove marks: by volume, the larger first
    for m, post_mm3, x in ((big, np.pi * 2.5 ** 2 * 10, 8.0), (small, np.pi * 2.5 ** 2 * 8, 16.0)):
        c = m["cleanup"]
        assert (m["type"], m["remove_mode"], m["mirror"]) == ("remove", "cut", False) and "smooth_mode" not in m
        assert m["radius_mm"] == 0.4331                                       # pitch x sqrt(3) / 2, rounded up to 1e-4 
        assert c["detector"] == "spur" and c["operator"] == "remove cut" and c["source"].startswith("s5_candidates.py")
        assert abs(c["location"]["centre_mm"][0] - x) < 0.5
        me = c["measured"]
        assert me["detected_volume_mm3"] == pytest.approx(post_mm3, rel=0.25)  # the post plus a collar of the bar under it
        assert me["removal_reach_mm"] == 2.0 and me["components"] == 1 and "component" not in c and me["points"] == len(m["points"])
        assert me["length_mm"] >= 5.0 and me["voxels"] * 0.5 ** 3 == pytest.approx(me["detected_volume_mm3"])
        assert c["thresholds_used"]["spur_min_length_mm"]["limit"] == 5.0
        assert c["thresholds_used"]["spur_min_volume_mm3"]["value"] == pytest.approx(me["detected_volume_mm3"], abs=0.01)
        assert c["location"]["near_frozen"] is (c["confidence"] == "low") and c["stress"] is None
    h = doc["cleanup"]
    assert h["pitch_mm"] == 0.5 and h["pitch_is_default"] is True and h["kinds"] == ["spur", "bump", "member"]
    assert h["thresholds"]["spur_min_length_mm"]["overridden"] is False and "real_change" in h
    for k in ("origin_mm",):
        assert all(abs(v / 0.5 - round(v / 0.5)) < 1e-6 for v in h["grid"][k])  # the lattice is a multiple of the pitch (C2)
    # no absolute path and no time in the marks file; the report is portable too
    text = marks_file.read_text() + (tmp_path / "out" / "cleanup.json").read_text()
    assert str(tmp_path) not in text and str(run) not in text
    assert "elapsed" not in marks_file.read_text() and "created" not in marks_file.read_text()
    rj = json.loads((tmp_path / "out" / "cleanup.json").read_text())
    assert rj["paths_are_relative_to"].startswith("the folder of this file") and rj["out_dir"] == "."
    assert not Path(rj["run_dir"]).is_absolute()
    assert rep["files"] == {"marks": "marks.json", "report": "cleanup.json"} and rep["marks_sha256"] == sha(marks_file)


def test_the_list_is_deterministic_and_overwrite_rewrites_the_same_bytes(posts, tmp_path):
    run, y = posts
    a = listing(run, y, tmp_path / "a")
    first = (tmp_path / "a" / "marks.json").read_bytes()
    listing(run, y, tmp_path / "a")                                            # overwrite=True: an earlier cleanup folder
    assert (tmp_path / "a" / "marks.json").read_bytes() == first
    listing(run, y, tmp_path / "b")                                            # another folder: no path, no time in the file
    assert (tmp_path / "b" / "marks.json").read_bytes() == first
    assert a["marks_sha256"] == hashlib.sha256(first).hexdigest()
    assert not list((tmp_path / "a").glob("*.tmp"))                            # the atomic write leaves no temporary file


def test_set_overrides_unknown_and_method_names_and_the_budget(posts, tmp_path):
    run, y = posts
    rep = listing(run, y, tmp_path / "a", spur_min_length_mm=9.5, max_candidates=1)
    th = rep["options"]
    assert th["spur_min_length_mm"] == {**th["spur_min_length_mm"], "value": 9.5, "overridden": True}
    assert th["max_candidates"]["overridden"] and th["pitch_mm"]["overridden"] is False
    assert rep["counts"]["listed"] == 0                                        # both posts have a skeleton shorter than 9.5 mm
    rep = listing(run, y, tmp_path / "b", max_candidates=1)
    assert (rep["counts"]["listed"], rep["counts"]["not_listed"]) == (1, 1)
    assert "budget" in rep["not_listed"][0]["why_not_listed"] and "max_candidates 1" in rep["not_listed"][0]["why_not_listed"]
    rep = listing(run, y, tmp_path / "c", max_candidates=None)                 # all of them
    assert rep["counts"]["listed"] == 2
    rep = listing(run, y, tmp_path / "d", pitch_mm=-1, near_junction_factor=2, grid_pad_mm=3, nonsense=1, order="alphabet", kinds="rock")
    text = " | ".join(rep["problems"])
    assert "pitch_mm=-1" in text and "not settable" in text and "'near_junction_factor' is not known" in text
    assert "'nonsense' is not known" in text and "order='alphabet'" in text and "kinds='rock'" in text
    assert rep["header"]["pitch_mm"] == 0.5 and rep["counts"]["listed"] == 2   # the defaults were used
    bad = cl.cleanup_list(run, out_dir=tmp_path / "e", yaml_path=y, options=[1, 2])
    assert bad["ok"] and "options must be an object" in " ".join(bad["problems"])


# --- the other detectors --------------------------------------------------------------------------------------------------

def test_a_thin_bump_far_from_the_plate_is_high_and_near_it_low(bump, tmp_path):
    run, y = bump
    rep = listing(run, y, tmp_path / "out")
    by = {m["cleanup"]["detector"]: m for m in rep["marks"]}
    assert set(by) == {"spur", "bump"} and rep["counts"]["by_operator"] == {"remove cut": 2}
    spur, fin = by["spur"]["cleanup"], by["bump"]["cleanup"]
    assert spur["confidence"] == "high" and spur["location"]["nearest_frozen_mm"] >= 6.0
    assert fin["confidence"] == "low" and fin["location"]["near_frozen"] and fin["location"]["nearest_frozen_mm"] < 6.0   # still listed
    assert fin["measured"]["detected_volume_mm3"] == pytest.approx(84.0, rel=0.1)       # the fin: 12 x 2 x 4 mm minus the collar
    assert fin["thresholds_used"]["bump_extent_min_mm"]["limit"] == 5.0 and fin["source"].startswith("s4_bumps.py")
    assert by["bump"]["remove_mode"] == "cut" and by["bump"]["radius_mm"] == 0.4331


def test_a_flat_web_is_a_fair_member_and_a_flatness_limit_makes_it_a_smooth_tube(web, tmp_path):
    run, y = web
    rep = listing(run, y, tmp_path / "a", member_min_cv=0.12)                 # a margin below the CV of the web (0.15 to 0.18)
    kinds = sorted((m["cleanup"]["detector"], m["cleanup"]["operator"]) for m in rep["marks"])
    assert kinds == [("member", "smooth fair"), ("spur", "remove cut")]
    fair = next(m for m in rep["marks"] if m["cleanup"]["detector"] == "member")
    assert (fair["type"], fair["smooth_mode"], fair["radius_mm"]) == ("smooth", "fair", cl.METHOD["fair_zone_mm"][0])
    assert fair["cleanup"]["measured"]["flatness"] >= cl.THRESHOLDS["web_flatness"][0] and fair["cleanup"]["measured"]["dead_ends"] == 0
    assert fair["cleanup"]["thresholds_used"]["member_min_cv"]["limit"] == 0.12 and fair["remove_mode"] is None
    assert [m["id"] for m in rep["marks"]] == [1, 2] and rep["marks"][0]["type"] == "remove"      # remove first, then the smooth marks
    rep = listing(run, y, tmp_path / "b", member_min_cv=0.12, web_flatness=9.0)
    tube = next(m for m in rep["marks"] if m["cleanup"]["detector"] == "member")
    assert (tube["type"], tube["smooth_mode"], tube["cleanup"]["operator"]) == ("smooth", "tube", "smooth tube")
    assert tube["radius_mm"] == pytest.approx(tube["cleanup"]["measured"]["mean_radius_mm"], abs=1e-3)
    doc = json.loads((tmp_path / "b" / "marks.json").read_text())
    assert len(next(m for m in doc["marks"] if m["type"] == "smooth")["points"]) >= 4      # the resampled path of the member


def test_a_candidate_in_the_frozen_region_is_not_listed(tmp_path):
    part = PLATE() + BAR() + cyl_z(12, 0, 8, 16, 2.5) + cyl_z(23.9, 0, 0, 1, 1) + M.cylinder(10, 1.5, 1.5, 32).rotate((0, 90, 0)).translate((23.9, 0, 2))
    run, y = run_at(tmp_path / "case", part)                                  # a rod that leaves the tip region at x 24
    rep = listing(run, y, tmp_path / "out")
    assert rep["counts"]["frozen_excluded"] >= 1 and rep["counts"]["listed"] == 1
    assert all(m["cleanup"]["location"]["centre_mm"][0] < 20 for m in rep["marks"])      # only the post at x 12


def tall_run(root, part, keep_out=None):
    """A run on a case with a taller design box (z 0..40) and, when `keep_out` is a box (lo, hi), a keep-out region of that box."""
    root = Path(root)
    d = root / "in"
    ypath = make_case(d, keep_out=["ko.stl"] if keep_out else ())
    box(d / "design.stl", (0, -6, 0), (24, 6, 40))
    if keep_out:
        box(d / "ko.stl", *keep_out)
    run = fake_run(root / "runs" / "r", ypath)
    to_stl(part, run / "result" / "final.stl")
    return run, ypath


def test_a_candidate_whose_tip_enters_a_frozen_region_is_not_listed_by_the_inside_rule_alone(tmp_path):
    """Review m3. The rule is `fd_min <= 0` (a voxel of the candidate inside a frozen region), next to the share rule (more than 5 % of the
    voxels within 3 mm). A long needle spur on the bar, with its tip 1.5 mm inside a keep-out box, has a share far below 5 %, so the
    share rule cannot exclude it; with `frozen_near_mm` at 0.01 mm only the voxels in or next to the box count. The control has no box."""
    needle = PLATE() + BAR() + M.cylinder(27.5, 2.5, 0.5, 48).translate((12, 0, 8))        # z 8 to 35.5: a cone, base radius 2.5
    ko = ((10, -3, 34), (14, 3, 38))
    free_run, free_y = tall_run(tmp_path / "free", needle)
    free = listing(free_run, free_y, tmp_path / "free_out")
    spur = [m for m in free["marks"] if m["cleanup"]["detector"] == "spur"]
    assert len(spur) == 1 and spur[0]["cleanup"]["location"]["centre_mm"][2] > 15           # the control lists the needle
    run, y = tall_run(tmp_path / "case", needle, ko)
    for name, opts in (("default", {}), ("tiny_near", {"frozen_near_mm": 0.01})):
        rep = listing(run, y, tmp_path / name, **opts)
        assert rep["counts"]["frozen_excluded"] > free["counts"]["frozen_excluded"], name            # the needle is excluded
        assert not [m for m in rep["marks"] if m["cleanup"]["location"]["centre_mm"][2] > 15], name


def test_a_candidate_order_by_stress_uses_the_verify_fields(posts, tmp_path):
    run, y = posts
    work, _ = run_at(tmp_path / "w", posts_part())
    final = work / "result" / "final.stl"
    # a field with one tetra at each post: high stress at x 8, low at x 16, degenerate tetras (the 4 corners are one point)
    pts = np.array([[8.0, 0.0, 8.0], [16.0, 0.0, 8.0], [2.0, 0.0, 2.0], [20.0, 0.0, 2.0]]).T
    tets = np.array([[0, 1, 2, 3]] * 4)
    vdir = work / "verify"
    vdir.mkdir()
    np.savez(vdir / "fields.npz", p=pts, t=tets, vm_elem_max_down=np.array([10.0, 1.0, 5.0, 5.0]))
    (vdir / "verify.json").write_text(json.dumps({"files": {"fields": "fields.npz"}, "run": {"final_stl_sha256": sha(final)}}))
    pipe = listing(work, y, tmp_path / "p")
    assert [round(m["cleanup"]["location"]["centre_mm"][0]) for m in pipe["marks"]] == [8, 16]          # by volume
    assert all(m["cleanup"]["stress"] is not None for m in pipe["marks"]) and pipe["header"]["stress"]["cases"] == ["down"]
    st = listing(work, y, tmp_path / "s", order="stress")
    ratios = [m["cleanup"]["stress"]["ratio"] for m in st["marks"]]
    assert ratios == sorted(ratios) and [round(m["cleanup"]["location"]["centre_mm"][0]) for m in st["marks"]] == [16, 8]   # low stress first
    assert st["header"]["order"].startswith("bodies; remove, smooth, fair")
    (vdir / "verify.json").write_text(json.dumps({"files": {"fields": "fields.npz"}, "run": {"final_stl_sha256": "0" * 64}}))
    other = listing(work, y, tmp_path / "o")                                   # fields for another final.stl: no stress facts, a note
    assert other["header"]["stress"] is None and any("another final.stl" in n for n in other["notes"])
    no = listing(run, y, tmp_path / "n")
    assert no["header"]["stress"] is None and any("fields.npz is missing" in n for n in no["notes"])


# --- the lattice, the walk, the folders ------------------------------------------------------------------------------------

@pytest.mark.parametrize("pad", [2.0, 3.0, 4.0])
def test_the_lattice_is_anchored_so_the_pad_and_whole_pitch_shifts_do_not_change_the_list(posts, tmp_path, monkeypatch, pad):
    run, y = posts
    monkeypatch.setitem(cl.METHOD, "grid_pad_mm", (pad, "mm", cl.METHOD["grid_pad_mm"][2]))
    rep = listing(run, y, tmp_path / "a")
    ref = [(m["cleanup"]["location"]["centre_mm"], m["cleanup"]["measured"]["detected_volume_mm3"], m["cleanup"]["measured"]["points"]) for m in rep["marks"]]
    assert all(abs(v / 0.5 - round(v / 0.5)) < 1e-6 for v in rep["header"]["grid"]["origin_mm"])
    shifted, y2 = run_at(tmp_path / "s", posts_part(), shift=(1.0, -0.5, 1.5))        # whole pitches: the same lattice cells
    rep2 = listing(shifted, y2, tmp_path / "b")
    got = [(m["cleanup"]["location"]["centre_mm"], m["cleanup"]["measured"]["detected_volume_mm3"], m["cleanup"]["measured"]["points"]) for m in rep2["marks"]]
    assert [g[1:] for g in got] == [r[1:] for r in ref]
    for g, r in zip(got, ref):
        assert np.allclose(np.array(g[0]) - np.array(r[0]), [1.0, -0.5, 1.5], atol=2e-3)
    # the origin moves by the same whole pitches (the pad is not the same, but the cells are)
    d = np.array(rep2["header"]["grid"]["origin_mm"]) - np.array(rep["header"]["grid"]["origin_mm"])
    assert np.allclose(d / 0.5, np.round(d / 0.5))


def test_the_mirror_plane_is_a_lattice_plane_even_at_an_odd_position(tmp_path):
    for at in (0.0, 0.13, -0.37, 2.5):
        origin, shape = cl._lattice(np.array([[-2.0, -6.0, 0.0], [24.0, 6.0, 12.0]]), 0.5, [{"axis_index": 1, "at_mm": at}])
        m = (at - origin[1]) / 0.5
        assert abs(m - round(m)) < 1e-9, at                                    # the plane is a voxel face
        assert origin[1] <= -6.0 - cl.METHOD["grid_pad_mm"][0] + 1e-9 and np.isclose(origin[[0, 2]] / 0.5, np.round(origin[[0, 2]] / 0.5)).all()


def test_the_walk_has_one_walk_for_each_component_and_consecutive_points_are_neighbours():
    rng = np.random.default_rng(3)
    cloud = np.unique(rng.integers(0, 6, size=(160, 3)), axis=0)               # a ragged set; maybe several components
    far = cloud + np.array([40, 0, 0])
    idx = np.vstack([cloud, far[:20]])                                          # a second component far away
    walks = cl._walk_voxels(idx)
    dense = np.zeros(idx.max(0) + 1, bool)
    dense[tuple(idx.T)] = True
    _lab, ncomp = ndi.label(dense, structure=np.ones((3, 3, 3)))               # the 26-connected components
    assert len(walks) == ncomp >= 2
    assert set(np.concatenate(walks).tolist()) == set(range(len(idx)))         # every voxel is on a walk
    for w in walks:                                                             # no step is longer than a 26-neighbour step: no jump
        assert (np.linalg.norm(np.diff(idx[w], axis=0), axis=1) <= np.sqrt(3.0) + 1e-9).all()
    one = np.argwhere(np.ones((3, 4, 2), bool))
    (rows,) = cl._walk_voxels(one)
    assert (np.linalg.norm(np.diff(one[rows], axis=0), axis=1) <= np.sqrt(3.0) + 1e-9).all()


def fake_candidate(idx):
    """A spur candidate on a given voxel set: the fields that `_make_marks` reads."""
    idx = np.asarray(idx)
    return {"detector": "spur", "operator": "remove", "idx": idx, "volume": len(idx) * 0.125, "length": 5.0, "rmean": 1.0, "cv": None,
            "ratio": None, "flat": None, "dead_ends": 1, "frozen_share": 0.0, "fd_min": 20.0, "centre": np.zeros(3),
            "bbox": np.zeros((2, 3)), "rules": {}, "source": "test"}


def test_a_voxel_set_of_two_components_gives_two_marks_and_no_jump_segment():
    """The case of the review (m2): a line of 11 voxels and one voxel 2 voxels beyond its end. At pitch 0.5 the old walk joined them
    from the first voxel with a chord of 6.0 mm, and `apply_marks` cuts along a chord. Now there is one mark for each component."""
    line = np.array([[x, 0, 0] for x in range(11)])
    lone = np.array([[12, 0, 0]])
    opt = {k: v[0] for k, v in cl.THRESHOLDS.items()}
    marks = cl._make_marks(fake_candidate(np.vstack([line, lone])), 5, np.zeros(3), 0.5, opt, 1.24, False)
    assert [m["id"] for m in marks] == [5, 6] and [m["type"] for m in marks] == ["remove", "remove"]
    assert [m["cleanup"]["component"] for m in marks] == [{"index": 1, "of": 2, "voxels": 11, "detected_volume_mm3": 1.375},
                                                          {"index": 2, "of": 2, "voxels": 1, "detected_volume_mm3": 0.125}]
    assert all(m["cleanup"]["measured"]["components"] == 2 and "walk_jumps" not in m["cleanup"]["measured"] for m in marks)
    assert "Part 1 of 2" in marks[0]["note"] and "Part 2 of 2" in marks[1]["note"] and marks[0]["cleanup"]["reason"] == marks[0]["note"]
    for m in marks:                                                             # no chord between the parts
        P = np.array(m["points"])
        assert (np.linalg.norm(np.diff(P, axis=0), axis=1) <= 0.5 * np.sqrt(3.0) + 1e-3).all()
    centres = {tuple(np.round((v + 0.5) * 0.5, 4)) for v in np.vstack([line, lone])}      # origin 0: (index + 0.5) x pitch
    assert {tuple(q) for m in marks for q in m["points"]} == centres
    one = cl._make_marks(fake_candidate(line), 1, np.zeros(3), 0.5, opt, 1.24, False)    # one component: one mark, no part fact
    assert len(one) == 1 and "component" not in one[0]["cleanup"] and one[0]["cleanup"]["measured"]["components"] == 1


def test_the_points_of_a_mark_are_one_walk(posts, tmp_path):
    run, y = posts
    rep = listing(run, y, tmp_path / "o")
    for m in json.loads((tmp_path / "o" / "marks.json").read_text())["marks"]:
        P = np.array(m["points"])
        assert (np.linalg.norm(np.diff(P, axis=0), axis=1) <= 0.5 * np.sqrt(3.0) + 1e-3).all()
    assert all(m["cleanup"]["measured"]["components"] == 1 for m in rep["marks"]) and rep["counts"]["marks"] == rep["counts"]["listed"]


def test_the_default_folder_and_the_refused_folders(posts, tmp_path):
    run, y = posts
    work, wy = run_at(tmp_path / "case", posts_part())
    rep = cl.cleanup_list(work, yaml_path=wy)
    assert Path(rep["out_dir"]) == work.parent / "r-cleanup" and (work.parent / "r-cleanup" / "marks.json").is_file()
    rep2 = cl.cleanup_list(work, yaml_path=wy)                                 # the next call: a new folder, the first is not touched
    assert Path(rep2["out_dir"]) == work.parent / "r-cleanup-2" and (work.parent / "r-cleanup" / "marks.json").is_file()
    # refusals: each names the fact and the options, and nothing is written
    cases = {
        "run_folder": work / "inside",
        "input_folder": tmp_path / "case" / "in",
        "not_empty": tmp_path / "case" / "runs",
    }
    for reason, folder in cases.items():
        before = tree_hash(tmp_path / "case")
        bad = cl.cleanup_list(work, out_dir=folder, yaml_path=wy)
        assert bad["ok"] is False and bad["refused"]["reason"] == reason and bad["refused"]["options"], reason
        assert tree_hash(tmp_path / "case") == before and not (folder / "marks.json").exists()
    afile = tmp_path / "afile"
    afile.write_text("x")
    assert cl.cleanup_list(work, out_dir=afile, yaml_path=wy)["refused"]["reason"] == "not_a_folder"
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "somewhere", target_is_directory=True)
    assert cl.cleanup_list(work, out_dir=link, yaml_path=wy)["refused"]["reason"] == "symbolic_link"
    held = tmp_path / "held"
    held.mkdir()
    (held / "other.txt").write_text("keep")
    bad = cl.cleanup_list(work, out_dir=held, yaml_path=wy, overwrite=True)    # overwrite is for an earlier cleanup folder only
    assert bad["refused"]["reason"] == "not_a_cleanup_folder" and (held / "other.txt").read_text() == "keep"
    assert cl.cleanup_list(work, out_dir=held, yaml_path=wy)["refused"]["reason"] == "not_empty"


def test_the_folder_of_an_acc_folder_is_beside_the_root_and_a_folder_in_the_root_is_refused(tmp_path):
    work, wy = run_at(tmp_path / "case", posts_part())
    root = tmp_path / "root"
    (root / "acc_00").mkdir(parents=True)
    for item in ("run.json", "labels.bin"):
        (root / "acc_00" / item).write_bytes((work / item).read_bytes())
    (root / "acc_00" / "result").mkdir()
    (root / "acc_00" / "result" / "final.stl").write_bytes((work / "result" / "final.stl").read_bytes())
    (root / "ratchet.json").write_text("{}")
    rep = cl.cleanup_list(root / "acc_00", yaml_path=wy)
    assert rep["ok"], rep.get("error")
    assert Path(rep["out_dir"]) == tmp_path / "root-cleanup-acc_00"            # beside the root; the root keeps its own files
    assert sorted(p.name for p in root.iterdir()) == ["acc_00", "ratchet.json"]
    for inside in (root / "cleanup", root / "acc_00" / "x", root):
        bad = cl.cleanup_list(root / "acc_00", out_dir=inside, yaml_path=wy)
        assert bad["ok"] is False and bad["refused"]["reason"] in ("ratchet_root", "run_folder"), inside
    assert sorted(p.name for p in root.iterdir()) == ["acc_00", "ratchet.json"]


def test_the_guards_refuse_before_any_work_and_the_estimate_writes_nothing(posts, tmp_path, monkeypatch):
    run, y = posts
    before = tree_hash(tmp_path)
    bad = cl.cleanup_list(run, out_dir=tmp_path / "v", yaml_path=y, options={"max_voxels": 5000})
    assert bad["ok"] is False and bad["refused"]["reason"] == "voxel_guard" and "Nothing was computed" in bad["error"]
    assert any("pitch_mm" in o for o in bad["refused"]["options"]) and bad["facts"]["detector_estimate"]["pitch_that_fits_mm"] > 0.5
    bad = cl.cleanup_list(run, out_dir=tmp_path / "m", yaml_path=y, options={"max_gb": 0.05})
    assert bad["ok"] is False and bad["refused"]["reason"] == "memory_guard" and "max_gb" in " ".join(bad["refused"]["options"])
    assert not (tmp_path / "v").exists() and not (tmp_path / "m").exists() and tree_hash(tmp_path) == before
    est = cl.cleanup_list(run, yaml_path=y, estimate_only=True)
    assert est["ok"] and est["estimate_only"] and est["out_dir"] is None and est["files"] == {"marks": None, "report": None}
    d = est["estimate"]["detector"]
    assert d["kind"] == "model" and d["voxels"] > 0 and d["memory_estimate_mb"] > 0 and "measured" in d
    assert not Path(str(run) + "-cleanup").exists() and tree_hash(tmp_path) == before
    import topopt.voxelize as vx

    def boom(*a, **k):
        raise MemoryError("test")
    monkeypatch.setattr(vx, "inside_mask", boom)
    out = cl.cleanup_list(run, out_dir=tmp_path / "x", yaml_path=y)
    assert out["ok"] is False and "out of memory" in out["error"] and "pitch_mm" in out["error"]


def test_missing_inputs_are_facts_not_crashes(tmp_path):
    assert "run folder not found" in cl.cleanup_list(tmp_path / "none")["error"]
    (tmp_path / "empty").mkdir()
    assert "run.json" in cl.cleanup_list(tmp_path / "empty")["error"]
    run, y = run_at(tmp_path / "case", posts_part())
    (run / "result" / "final.stl").unlink()
    assert "final.stl" in cl.cleanup_list(run, yaml_path=y)["error"] and not Path(str(run) + "-cleanup").exists()


# --- mirror twins ---------------------------------------------------------------------------------------------------------

PLANE_Y = [{"axis": "y", "at_mm": 0.0}]


def pair_part():
    return PLATE() + BAR() + cyl_z(12, 2.2, 8, 18, 1.7) + cyl_z(12, -2.2, 8, 18, 1.7)


def quad_part():
    return PLATE() + BAR() + sum((cyl_z(x, s * 2.2, 8, 18, 1.7) for x in (8, 16) for s in (1, -1)), M())


def test_two_mirror_twins_are_one_mark_with_the_facts_of_the_other(tmp_path):
    for pitch in (0.5, 0.6):                                                    # the E5 layout changed the list with the pitch
        run, y = run_at(tmp_path / f"p{pitch}", pair_part(), mirror_planes=PLANE_Y)
        rep = listing(run, y, tmp_path / f"o{pitch}", pitch_mm=pitch)
        assert (rep["counts"]["found"], rep["counts"]["after_twin_merge"], rep["counts"]["twins_merged"]) == (2, 1, 1)
        m = rep["marks"][0]
        assert m["mirror"] is True and rep["header"]["mirror"]["pairs_merged"] == 1
        tw = m["cleanup"]["mirror"]["twins"]
        assert len(tw) == 1 and tw[0]["twin_overlap_voxels"] > 0 and 0 < tw[0]["twin_iou"] <= 1 and tw[0]["detector"] == "spur"
        assert m["cleanup"]["location"]["centre_mm"][1] * tw[0]["centre_mm"][1] < 0       # on the other side of the plane
        run2, y2 = run_at(tmp_path / f"q{pitch}", pair_part())                              # no plane: two marks, mirror false
        rep2 = listing(run2, y2, tmp_path / f"n{pitch}", pitch_mm=pitch)
        assert rep2["counts"]["listed"] == 2 and not any(m["mirror"] for m in rep2["marks"]) and rep2["header"]["mirror"] is None


def test_a_spur_on_the_plane_is_one_mark_without_the_mirror_flag(tmp_path):
    part = PLATE() + BAR() + cyl_z(12, 0, 8, 18, 2.0)
    run, y = run_at(tmp_path / "c", part, mirror_planes=PLANE_Y)
    rep = listing(run, y, tmp_path / "o")
    assert rep["counts"]["listed"] == 1 and rep["marks"][0]["mirror"] is False
    mir = rep["marks"][0]["cleanup"]["mirror"]
    assert mir["twins"] == [] and mir["on_mirror_plane_voxels"] > 0


def test_two_planes_make_orbits_of_four(tmp_path):
    planes = PLANE_Y + [{"axis": "x", "at_mm": 12.0}]
    run, y = run_at(tmp_path / "c", quad_part(), mirror_planes=planes)
    rep = listing(run, y, tmp_path / "o")
    assert (rep["counts"]["found"], rep["counts"]["listed"], rep["counts"]["twins_merged"]) == (4, 1, 3)
    assert rep["marks"][0]["mirror"] is True and len(rep["marks"][0]["cleanup"]["mirror"]["twins"]) == 3
    one, y1 = run_at(tmp_path / "d", quad_part(), mirror_planes=PLANE_Y)                    # one plane: two orbits of two
    assert listing(one, y1, tmp_path / "p")["counts"]["listed"] == 2


# --- bodies ---------------------------------------------------------------------------------------------------------------

def test_a_loose_body_is_the_only_kind_in_the_file_until_the_kinds_option_says_otherwise(tmp_path):
    part = body_part() + cyl_z(16, 0, 8, 16, 2.5)                              # a loose cube above the bar, and a post on the bar
    run, y = run_at(tmp_path / "c", part)
    rep = listing(run, y, tmp_path / "a")
    b = rep["bodies"]
    assert (b["count"], b["holding_fixed_or_load"], b["cavities"]) == (2, 1, 0)
    loose = [x for x in b["list"] if x["removable"]]
    assert len(loose) == 1 and loose[0]["volume_mm3"] == pytest.approx(4 * 2 * 4) and not loose[0]["holds_fixed_or_load"]
    assert [m["cleanup"]["detector"] for m in rep["marks"]] == ["body"] and rep["header"]["kinds"] == ["body"]
    m = rep["marks"][0]
    assert (m["type"], m["remove_mode"]) == ("remove", "cut") and m["cleanup"]["source"].startswith("apply_marks._body_facts")
    assert m["cleanup"]["measured"]["detected_volume_mm3"] == pytest.approx(32.0, rel=0.1)
    assert rep["counts"]["not_listed"] == 1 and "removable body exists" in rep["not_listed"][0]["why_not_listed"]
    assert rep["header"]["applicability"]["note"] is None                      # a body mark is not a strut: no strut note
    mixed = listing(run, y, tmp_path / "b", kinds="body,spur")
    assert sorted(m["cleanup"]["detector"] for m in mixed["marks"]) == ["body", "spur"] and "mixed_kinds_note" in mixed["header"]
    assert mixed["header"]["applicability"]["note"] == cl.NOTE_STRUTS
    only = listing(run, y, tmp_path / "c2", kinds="spur")
    assert [m["cleanup"]["detector"] for m in only["marks"]] == ["spur"] and "mixed_kinds_note" not in only["header"]


def test_a_loose_body_outside_the_run_grid_is_a_fact_with_options_and_has_no_mark(tmp_path):
    """Review m4. `apply_marks` cannot cut a body that lies outside the grid of the run: it reads the frozen distance at the nearest grid
    voxel (a wrong `no_effect` reason) or fails in the marching cubes (an internal ValueError). The list makes no mark for such a body."""
    post = cyl_z(16, 0, 8, 16, 2.5)
    for name, cube_box in (("beyond_tip", ((30, -2, 0), (34, 2, 4))), ("beside_bar", ((10, 8, 0), (14, 12, 4)))):    # the grid ends at x 24 and y 6
        run, y = run_at(tmp_path / name, PLATE() + BAR() + post + cube(*cube_box))
        rep = listing(run, y, tmp_path / (name + "_out"))
        out = [b for b in rep["bodies"]["list"] if b["outside_run_grid"]]
        assert len(out) == 1 and out[0]["removable"] and out[0]["volume_mm3"] == pytest.approx(64.0), name
        assert [m["cleanup"]["detector"] for m in rep["marks"]] == ["spur"], name     # no body mark; the post is listed as usual
        assert rep["counts"]["outside_run_grid"] == 1 and rep["counts"]["not_listed"] == 1, name
        (nl,) = rep["not_listed"]
        assert nl["detector"] == "body" and nl["outside_run_grid"] is True and nl["why_not_listed"].startswith("outside_run_grid"), name
        assert len(nl["options"]) == 2 and "stl:" in nl["options"][0] and "design box" in nl["options"][1], name
        assert any("outside the grid of the run" in n for n in rep["notes"]), name
    # only a body with no common point with the grid is outside: one that crosses the grid edge keeps its mark, and apply_marks cuts it
    run, y = run_at(tmp_path / "cross", PLATE() + BAR() + cube((23, -2, 9.5), (27, 2, 11.5)))
    rep = listing(run, y, tmp_path / "cross_out")
    assert [m["cleanup"]["detector"] for m in rep["marks"]] == ["body"] and rep["counts"]["outside_run_grid"] == 0
    assert not any(b["outside_run_grid"] for b in rep["bodies"]["list"])
    applied = am.apply_marks(run, str(tmp_path / "cross_out" / "marks.json"), out_dir=tmp_path / "cross_am")
    assert applied["ok"] and applied["totals"]["bodies_after"] == 1 and applied["totals"]["change_mm3"] < -30


def test_a_cavity_is_not_a_candidate_and_a_part_without_support_has_no_candidate(tmp_path):
    hollow = PLATE() + BAR() - cube((8, -1, 3), (12, 1, 5))                    # a closed cavity inside the bar
    run, y = run_at(tmp_path / "c", hollow)
    rep = listing(run, y, tmp_path / "a")
    assert rep["bodies"]["cavities"] == 1 and rep["bodies"]["count"] == 2 and not any(x["removable"] for x in rep["bodies"]["list"])
    assert rep["marks"] == [] and any("No candidate was found" in n for n in rep["notes"])
    assert json.loads((tmp_path / "a" / "marks.json").read_text())["marks"] == []        # an empty list is a valid marks file
    away, y2 = run_at(tmp_path / "d", cube((40, 40, 40), (44, 44, 44)))                 # the part touches no fixed and no load voxel
    rep = listing(away, y2, tmp_path / "b")
    assert rep["marks"] == [] and rep["bodies"]["holding_fixed_or_load"] == 0 and "would remove the whole part" in rep["bodies"]["note"]


# --- applicability --------------------------------------------------------------------------------------------------------

def test_the_applicability_block_gives_facts_and_a_fixed_sentence_and_no_trigger(posts, bump, tmp_path):
    rep = listing(*posts, tmp_path / "a")
    a = rep["header"]["applicability"]
    assert a["note"] == cl.NOTE_STRUTS and "skeleton gives sheet fragments" in a["note"]       # fixed sentence (C12)
    assert a["members_in_table"] >= 1 and a["branches"] >= 1 and a["skeleton_points"] > 0 and 0 < a["member_volume_share"] <= 1.0
    assert 0 <= a["dead_end_member_share"] <= 1 and 0 <= a["flat_member_volume_share"] <= 1
    assert not any("trigger" in k or "threshold" in k or "limit" in k for k in a)             # no number decides when the note appears
    nb = listing(*bump, tmp_path / "b", kinds="bump")
    assert [m["cleanup"]["detector"] for m in nb["marks"]] == ["bump"] and nb["header"]["applicability"]["note"] is None


# --- the judge estimate with a root (review M1) ----------------------------------------------------------------------------

def test_the_judge_estimate_counts_the_marks_of_this_list_and_takes_only_the_baseline_state_from_the_root(tmp_path, monkeypatch):
    """`ratchet.check_report` subtracts the candidates that the log of the root already holds, so a new list of 3 marks in a root with
    3 old candidates gave 0 candidates and 0 s. The count is the count of the list; the root gives only 'baseline done or not'."""
    states = {}

    def fake_check_report(run_dir, yaml_path, source=None, root=None):
        rep = {"yaml": "x.yaml", "ready": True, "rules": [], "tools": {"printcheck": {"ready": True, "seconds": 2.0}, "verify": None}}
        if states["estimate"] is not None:
            rep["estimate"] = states["estimate"]
        return rep

    monkeypatch.setattr(rt, "check_report", fake_check_report)
    opt = {k: v[0] for k, v in cl.THRESHOLDS.items()}

    def judge_estimate(count):
        return cl._judge_estimate(tmp_path, tmp_path / "x.yaml", tmp_path / "marks.json", count, tmp_path / "root", opt)

    states["estimate"] = {"candidates": 0, "judge_calls": 0}        # the root holds 3 judged candidates and its baseline
    je = judge_estimate(3)
    assert (je["candidates"], je["judge_calls"], je["baseline_included"], je["seconds_total_model"]) == (3, 3, False, 6.0)
    states["estimate"] = {"candidates": 0, "judge_calls": 1}        # the root has no baseline yet: one call more
    je = judge_estimate(3)
    assert (je["candidates"], je["judge_calls"], je["baseline_included"], je["seconds_total_model"]) == (3, 4, True, 8.0)
    states["estimate"] = None                                        # no marks file (estimate_only): the baseline is counted
    je = judge_estimate(10)
    assert (je["candidates"], je["judge_calls"]) == (10, 11)
    assert judge_estimate(None)["candidates"] is None                # max_candidates null: all; the count is not known


# --- the phase test -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,make,options,expect", [
    ("posts", posts_part, {}, [("spur", "remove cut", 203), ("spur", "remove cut", 170)]),
    ("web", web_part, {"member_min_cv": 0.12}, [("spur", "remove cut", 165), ("member", "smooth fair", 255)]),
    ("bump", bump_part, {}, [("spur", "remove cut", 173), ("bump", "remove cut", 84)]),
    ("body", body_part, {}, [("body", "remove cut", 32)]),
])
def test_phase_the_same_kinds_operators_and_counts_at_three_shifts(tmp_path, name, make, options, expect):
    seen = []
    for k, shift in enumerate(PHASES):
        run, y = run_at(tmp_path / f"{name}{k}", make(), shift=shift)
        rep = listing(run, y, tmp_path / f"o{k}", **options)
        got = [(m["cleanup"]["detector"], m["cleanup"]["operator"], m["cleanup"]["measured"]["detected_volume_mm3"]) for m in rep["marks"]]
        assert [g[:2] for g in got] == [e[:2] for e in expect], (shift, got)               # exact: kinds and operators and counts
        for g, e in zip(got, expect):
            assert g[2] == pytest.approx(e[2], rel=0.15), (shift, g, e)                    # sizes: the lattice phase moves them a little
        seen.append([m["cleanup"]["confidence"] for m in rep["marks"]])
    assert seen[0] == seen[1] == seen[2], seen                                              # the near-frozen confidence is stable too


# --- the marks file in the rest of the system ------------------------------------------------------------------------------

def test_the_marks_file_is_read_by_the_marks_reader_and_the_per_mark_keys_by_apply_marks(web, tmp_path):
    run, y = web
    listing(run, y, tmp_path / "o", member_min_cv=0.12)
    doc = fb.read_marks(str(tmp_path / "o" / "marks.json"))
    assert doc["version"] == 1 and len(doc["marks"]) == 2
    for i, raw in enumerate(doc["marks"]):
        nm, problems = fb._norm_mark(raw, i)
        assert nm is not None and not problems, problems
        assert nm["remove_mode"] == raw.get("remove_mode") and nm["smooth_mode"] == raw.get("smooth_mode")
        problems = []
        opt = {"smooth_mode": "auto", "remove_mode": "shave"}              # the call options differ from the marks, so the mark wins
        got_s = am._mode_of(raw["id"], raw["type"], "smooth_mode", {}, nm, am.SMOOTH_MODES, opt, problems)
        got_r = am._mode_of(raw["id"], raw["type"], "remove_mode", {}, nm, am.REMOVE_MODES, opt, problems)
        assert problems == []
        assert (got_s, got_r) == (("auto", "cut") if raw["type"] == "remove" else (raw["smooth_mode"], "shave"))


def test_the_short_report_and_the_script(posts, tmp_path, capsys):
    run, y = posts
    rep = listing(run, y, tmp_path / "o")
    short = cl.short_report(rep)
    assert short["ok"] and short["detail"] is False and short["report_file"] == "cleanup.json" and "more" in short
    assert [c["id"] for c in short["candidates"]] == [1, 2] and short["counts"] == rep["counts"]
    assert {"detector", "operator", "confidence", "centre_mm", "detected_volume_mm3", "nearest_frozen_mm"} <= set(short["candidates"][0])
    assert len(json.dumps(short)) < len(json.dumps(rep)) / 4                      # the short reply is short
    assert cl.short_report({"ok": False, "error": "x"}) == {"ok": False, "error": "x"}
    code = cl.main(["list", str(run), "--yaml", str(y), "--out", str(tmp_path / "cli"), "--set", "spur_min_length_mm=4", "--max-candidates", "1"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["counts"]["listed"] == 1 and (tmp_path / "cli" / "marks.json").is_file()
    code = cl.main(["list", str(run), "--yaml", str(y), "--estimate"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["estimate_only"] and not (tmp_path / "cli-2").exists()
    code = cl.main(["list", str(tmp_path / "missing")])
    cap = capsys.readouterr()
    assert code == 1 and "error:" in cap.err and json.loads(cap.out)["ok"] is False
    assert cl.main(["list", str(run), "--set", "oops"]) == 1 and "NAME=VALUE" in capsys.readouterr().err
