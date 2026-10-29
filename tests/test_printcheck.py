"""Tests for the print check (CONTRACTS.md section 32): `topopt printcheck`, the MCP tool `print_check`, the limit
keys of the `print` block, the states and the strict gate.

The meshes are made in the tests (no binary file is committed). The known answers are worked out by hand in the
comments. The comb fixture (`tests/fixtures/comb.stl`, text) has the V2 numbers 160 (overhang), 200 (plate contact)
and 640 (support volume) mm2 / mm3 for the build direction +z.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from topopt import printcheck as pc
from topopt import print_report as pr
from topopt.cli import main
from topopt.config import PRINT_LIMIT_KEYS, PrintSettings, UserError, load_config, parse_print_block

from test_input import make_case

ROOT = Path(__file__).resolve().parent.parent
COMB = ROOT / "tests" / "fixtures" / "comb.stl"


# --- meshes made in the tests ----------------------------------------------------------------------------------------


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    return m


def plate(t, size=12.0, edge=0.5):
    """A size x size x t plate with small triangles (edges 0.5 mm or less), so the mesh can resolve thin limits."""
    m = box((0, 0, 0), (size, size, t))
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=edge)
    return trimesh.Trimesh(v, f, process=True)


def merge(*meshes):
    return trimesh.Trimesh(np.vstack([m.vertices for m in meshes]),
                           np.vstack([m.faces + sum(len(x.vertices) for x in meshes[:i])
                                      for i, m in enumerate(meshes)]), process=False)


def save(mesh, path):
    mesh.export(path)
    return path


def by_id(report):
    return {c["id"]: c for c in report["checks"]}


def states(report):
    return {c["id"]: c["state"] for c in report["checks"]}


def make_run(tmp_path, mesh, h=None, print_block=None, name="run"):
    """A folder with result/final.stl and a small run.json (no metrics.json, no YAML)."""
    run = tmp_path / name
    (run / "result").mkdir(parents=True)
    mesh.export(run / "result" / "final.stl")
    doc = {"format": 1, "name": "t"}
    if h is not None:
        doc["h"] = h
    if print_block is not None:
        doc["print"] = print_block
    (run / "run.json").write_text(json.dumps(doc))
    return run


# --- the print block: limit keys ----------------------------------------------------------------------------------


def test_limit_keys_parse_and_name_their_source():
    s = parse_print_block({"min_wall_mm": 1.2, "max_closed_void_mm3": 0, "max_bodies": 2, "bed_mm": 180})
    assert s.limit("min_wall_mm") == 1.2 and s.limit_source("min_wall_mm") == "yaml"
    assert s.limit("max_closed_void_mm3") == 0.0 and s.limit_source("max_closed_void_mm3") == "yaml"
    assert s.limit("max_bodies") == 2 and s.limit_source("max_bodies") == "yaml"
    # a key that the YAML did not give: the default is "assumed", no default is "none"
    d = PrintSettings()
    assert (d.limit("max_bodies"), d.limit_source("max_bodies")) == (1, "assumed")
    assert (d.limit("min_wall_area_pct"), d.limit_source("min_wall_area_pct")) == (0.5, "assumed")
    for key in ("min_wall_mm", "max_closed_void_mm3", "max_overhang_area_mm2", "max_support_volume_mm3",
                "min_plate_contact_mm2"):
        assert (d.limit(key), d.limit_source(key)) == (None, "none")
    assert set(s.given) == {"min_wall_mm", "max_closed_void_mm3", "max_bodies", "bed_mm"}


@pytest.mark.parametrize("block, text", [
    ({"min_wall_mm": 0}, "print.min_wall_mm"),
    ({"min_wall_mm": -1}, "print.min_wall_mm"),
    ({"min_wall_mm": "thin"}, "print.min_wall_mm"),
    ({"max_bodies": 0}, "print.max_bodies"),
    ({"max_bodies": 1.5}, "print.max_bodies"),
    ({"min_wall_area_pct": 101}, "print.min_wall_area_pct"),
    ({"max_closed_void_mm3": -0.1}, "print.max_closed_void_mm3"),
    ({"max_overhang_area_mm2": True}, "print.max_overhang_area_mm2"),
    ({"min_wall": 1.2}, "unknown key 'min_wall'"),
])
def test_bad_limit_key_is_a_user_error(block, text):
    with pytest.raises(UserError, match=text):
        parse_print_block(block)


def test_run_json_print_dict_is_unchanged_without_limits_and_round_trips_with_them():
    # No limit key: the dict has the four V2 keys and `given` (a run folder keeps its old run.json).
    assert set(PrintSettings().to_dict()) == {"build_direction", "bed_mm", "printer_type",
                                              "self_supporting_angle_deg", "given"}
    s = parse_print_block({"min_wall_mm": 1.5, "max_bodies": 3})
    d = s.to_dict()
    assert d["limits"] == {"min_wall_mm": 1.5, "max_bodies": 3}
    back = PrintSettings.from_dict(json.loads(json.dumps(d)))
    assert back == s
    # a bad limit in run.json is dropped, not an error (the same rule as the V2 keys)
    d["limits"] = {"min_wall_mm": "x", "max_bodies": -2, "max_closed_void_mm3": 5}
    assert PrintSettings.from_dict(d).limits == (("max_closed_void_mm3", 5),)


def test_yaml_with_limit_keys_loads(tmp_path):
    p = make_case(tmp_path, h=2.0, mutate=lambda d: d.update({"print": {"min_wall_mm": 1.2, "bed_mm": 150}}))
    ps = load_config(p).print_settings
    assert ps.limit("min_wall_mm") == 1.2 and "min_wall_mm" in ps.given
    assert ps.to_dict()["limits"] == {"min_wall_mm": 1.2}


# --- known answers: wall thickness ---------------------------------------------------------------------------------


@pytest.mark.parametrize("t, want_state", [(1.0, "fail"), (2.0, "pass")])
def test_plate_thickness_against_the_limit(tmp_path, t, want_state):
    """A 12 x 12 x t mm plate. Every chord from a big face goes through the plate: t mm. The four rim faces have
    chords of 12 mm (the plate is 12 mm wide). The big faces hold 2 x 144 mm2 of 288 + 48 t mm2: 85.7 % for t = 1
    and 75 % for t = 2. The limit is 1.5 mm: a 1 mm plate fails (85.7 % of the area is thinner), a 2 mm plate passes."""
    p = save(plate(t), tmp_path / "plate.stl")
    r = pc.check(p, print_settings={"min_wall_mm": 1.5})
    c = by_id(r)["wall_thickness"]
    assert c["state"] == want_state, c["reason"]
    assert c["value"]["thinnest_member_mm"] == pytest.approx(t, rel=0.01)
    assert c["limit"]["value"] == 1.5 and c["limit"]["source"] == "option" and c["limit"]["key"] == "min_wall_mm"
    if t == 1.0:
        assert c["value"]["area_below_limit_pct"] == pytest.approx(100 * 288 / 336, abs=1.5)
        assert c["value"]["area_below_limit_mm2"] == pytest.approx(288, rel=0.02)
    else:
        assert c["value"]["area_below_limit_pct"] == 0.0 and c["value"]["area_below_limit_mm2"] == 0.0
    assert c["estimate"] is True


def test_plate_thickness_has_no_limit_and_gives_unknown(tmp_path):
    p = save(plate(1.0), tmp_path / "plate.stl")
    c = by_id(pc.check(p))["wall_thickness"]
    assert c["state"] == "unknown" and c["limit"]["value"] is None and c["limit"]["source"] == "none"
    assert "min_wall_mm" in c["reason"] and c["value"]["thinnest_member_mm"] == pytest.approx(1.0, rel=0.01)


def test_coarse_mesh_cannot_resolve_the_limit_so_it_is_unknown_not_pass(tmp_path):
    """The same 2 mm plate with 12 triangles: the median edge is 12 mm, so a limit of 1.5 mm is under 3 x the
    resolution. The measure does not pass it. With resolution_mm 0 (exact CAD) it passes. A thin plate still fails."""
    p = save(box((0, 0, 0), (12, 12, 2)), tmp_path / "coarse.stl")
    c = by_id(pc.check(p, print_settings={"min_wall_mm": 1.5}))["wall_thickness"]
    assert c["state"] == "unknown"
    assert c["value"]["resolution"]["resolution_mm"] > 5 and "pitfalls/mesh-coarser-than-members" in c["reason"]
    assert c["value"]["area_below_limit_pct"] == 0.0     # the measure sees no thin wall; it still cannot say pass
    exact = by_id(pc.check(p, print_settings={"min_wall_mm": 1.5}, resolution_mm=0))["wall_thickness"]
    assert exact["state"] == "pass" and exact["value"]["resolution"]["source"] == "option"
    thin = save(box((0, 0, 0), (12, 12, 1)), tmp_path / "coarse_thin.stl")
    assert by_id(pc.check(thin, print_settings={"min_wall_mm": 1.5}))["wall_thickness"]["state"] == "fail"


def test_run_folder_gives_its_voxel_size_as_the_resolution(tmp_path):
    """run.json has `h` and the `print` block of the run. A 2 mm plate with a 1.5 mm limit: a grid of 1.2 mm
    resolves members from 3.6 mm, so the state is unknown; a grid of 0.3 mm resolves from 0.9 mm: pass."""
    block = {"build_direction": [0, 0, 1], "bed_mm": [220, 220], "printer_type": "fdm",
             "self_supporting_angle_deg": None, "given": ["min_wall_mm"], "limits": {"min_wall_mm": 1.5}}
    coarse = pc.check(make_run(tmp_path, plate(2.0), h=1.2, print_block=block, name="coarse"))
    fine = pc.check(make_run(tmp_path, plate(2.0), h=0.3, print_block=block, name="fine"))
    c, f = by_id(coarse)["wall_thickness"], by_id(fine)["wall_thickness"]
    assert c["limit"]["value"] == 1.5 and c["limit"]["source"] == "run.json"      # the limit came from run.json
    assert c["state"] == "unknown" and c["value"]["resolution"]["voxel_size_mm"] == 1.2
    assert c["value"]["resolution"]["resolves_from_mm"] == pytest.approx(3.6)
    assert f["state"] == "pass" and f["value"]["resolution"]["resolution_mm"] == pytest.approx(0.3)
    assert coarse["input"]["kind"] == "run" and "run.json" in coarse["settings"]["from"]


def test_thickness_limit_above_the_chord_cap_is_unknown(tmp_path):
    p = save(box((0, 0, 0), (40, 40, 40)), tmp_path / "big.stl")
    c = by_id(pc.check(p, print_settings={"min_wall_mm": 20}, resolution_mm=0))["wall_thickness"]
    assert c["state"] == "unknown" and "chord cap" in c["reason"]


# --- known answers: closed voids, bodies, mesh health ---------------------------------------------------------------


def cavity_box(outer=20.0, inner=6.0):
    """A solid cube with a sealed cube-shaped cavity in the middle. The cavity wall has inward faces."""
    o = box((0, 0, 0), (outer,) * 3)
    lo = (outer - inner) / 2
    i = box((lo,) * 3, (lo + inner,) * 3)
    i.invert()
    return merge(o, i)


def test_sealed_cavity_is_a_closed_void_with_its_volume(tmp_path):
    """A 20 mm cube (8000 mm3) with a sealed 6 mm cube cavity (216 mm3): one body, one void of 216 mm3, a solid
    volume of 7784 mm3. A cavity wall is a closed shell with a negative volume, so it is a void and not a body."""
    p = save(cavity_box(), tmp_path / "cavity.stl")
    r = pc.check(p)
    c, bodies, closed = by_id(r)["closed_voids"], by_id(r)["bodies"], by_id(r)["mesh_closed"]
    assert c["value"]["count"] == 1 and c["value"]["total_mm3"] == pytest.approx(216.0)
    assert c["value"]["largest_mm3"] == pytest.approx(216.0)
    assert bodies["value"]["count"] == 1 and bodies["value"]["shells"] == 2 and bodies["state"] == "pass"
    assert closed["state"] == "pass" and closed["value"]["volume_mm3"] == pytest.approx(7784.0)
    # no limit: a fact with state unknown, and the process note
    assert c["state"] == "unknown" and c["limit"]["value"] is None
    assert "FDM prints a closed void as a hollow" in c["reason"]
    # the limit is the largest void: 100 fails, 300 passes, 0 fails
    assert by_id(pc.check(p, print_settings={"max_closed_void_mm3": 100}))["closed_voids"]["state"] == "fail"
    assert by_id(pc.check(p, print_settings={"max_closed_void_mm3": 216}))["closed_voids"]["state"] == "pass"
    assert by_id(pc.check(p, print_settings={"max_closed_void_mm3": 0}))["closed_voids"]["state"] == "fail"
    lp = by_id(pc.check(p, print_settings={"printer_type": "lpbf"}))["closed_voids"]
    assert lp["state"] == "unknown" and "powder" in lp["reason"]       # the process default is a note, not a limit


def test_no_void_passes_without_a_limit(tmp_path):
    p = save(box((0, 0, 0), (10, 10, 10)), tmp_path / "cube.stl")
    c = by_id(pc.check(p))["closed_voids"]
    assert c["state"] == "pass" and c["value"]["count"] == 0 and c["value"]["total_mm3"] == 0.0


def test_two_separate_bodies(tmp_path):
    """Two 10 mm cubes 10 mm apart: 2 bodies of 1000 mm3. The default limit is 1 (assumed), so it fails."""
    p = save(merge(box((0, 0, 0), (10, 10, 10)), box((20, 0, 0), (30, 10, 10))), tmp_path / "two.stl")
    r = pc.check(p)
    b = by_id(r)["bodies"]
    assert b["state"] == "fail" and b["value"]["count"] == 2
    assert [x["volume_mm3"] for x in b["value"]["largest"]] == [pytest.approx(1000.0)] * 2
    assert b["limit"] == {"value": 1, "key": "max_bodies", "source": "assumed"}
    assert by_id(r)["mesh_closed"]["state"] == "pass"
    assert by_id(pc.check(p, print_settings={"max_bodies": 2}))["bodies"]["state"] == "pass"
    # the inside-out relation: shells = bodies + voids
    assert b["value"]["shells"] == 2


def test_open_mesh_is_not_closed_and_the_thickness_is_unknown(tmp_path):
    """A 10 mm cube without its top face (2 triangles removed): 4 open edges. The chord needs a closed solid."""
    m = box((0, 0, 0), (10, 10, 10))
    keep = m.triangles_center[:, 2] < 9.9
    p = save(trimesh.Trimesh(m.vertices, m.faces[keep], process=False), tmp_path / "open.stl")
    r = pc.check(p, print_settings={"min_wall_mm": 1.0}, resolution_mm=0)
    c = by_id(r)["mesh_closed"]
    assert c["state"] == "fail" and c["value"]["open_edges"] == 4 and c["value"]["watertight"] is False
    assert by_id(r)["wall_thickness"]["state"] == "unknown" and "not closed" in by_id(r)["wall_thickness"]["reason"]


def test_non_manifold_edge(tmp_path):
    """Two unit cubes that touch along one edge: that edge has 4 faces."""
    p = save(merge(box((0, 0, 0), (1, 1, 1)), box((1, 1, 0), (2, 2, 1))), tmp_path / "edge.stl")
    c = by_id(pc.check(p))["mesh_closed"]
    assert c["state"] == "fail" and c["value"]["nonmanifold_edges"] == 1 and c["value"]["open_edges"] == 0
    assert "more than 2 faces" in c["reason"]


def test_float32_reload_shows_a_pinch_that_the_float64_mesh_does_not_have(tmp_path):
    """Two unit cubes 1e-9 mm apart. In float64 each is closed and nothing touches, so the mesh in memory is
    watertight. The STL holds float32 (1.2e-7 mm steps near 1), so the faces meet: 4 edges with 4 faces."""
    a, b = box((0, 0, 0), (1, 1, 1)), box((1 + 1e-9, 0, 0), (2, 1, 1))
    m = merge(a, b)
    assert m.is_watertight
    r = pc.check(save(m, tmp_path / "pinch.stl"))
    c = by_id(r)["mesh_closed"]
    assert c["state"] == "fail" and c["value"]["nonmanifold_edges"] == 4
    assert "float32 reload" in c["reason"]


def test_inside_out_mesh_fails_closed_and_keeps_one_body(tmp_path):
    m = box((0, 0, 0), (10, 10, 10))
    m.invert()
    r = pc.check(save(m, tmp_path / "inverted.stl"))
    assert by_id(r)["mesh_closed"]["state"] == "fail" and "inward" in by_id(r)["mesh_closed"]["reason"]
    assert by_id(r)["bodies"]["value"]["count"] == 1 and by_id(r)["closed_voids"]["value"]["count"] == 0
    assert "warning" in r


def test_a_collapsed_face_is_a_zero_area_fact_and_not_a_body(tmp_path):
    """A face whose corners are one point (the float32 rounding does this) has no edge and no volume. It is counted as a
    zero-area face. It is not a body, and it does not open the mesh."""
    m = box((0, 0, 0), (10, 10, 10))
    flap = np.vstack([m.faces, [[0, 0, 1]]])
    r = pc.check(save(trimesh.Trimesh(m.vertices, flap, process=False), tmp_path / "collapsed.stl"))
    c = by_id(r)
    assert c["mesh_closed"]["state"] == "pass" and c["mesh_closed"]["value"]["zero_area_faces"] == 1
    assert c["mesh_closed"]["value"]["triangles"] == 13 and "zero-area face" in c["mesh_closed"]["reason"]
    assert c["bodies"]["value"]["count"] == 1 and c["bodies"]["state"] == "pass"


def test_a_file_with_no_real_face_is_a_user_error(tmp_path):
    m = box((0, 0, 0), (10, 10, 10))
    p = save(trimesh.Trimesh(m.vertices, [[0, 0, 1], [2, 2, 2]], process=False), tmp_path / "none.stl")
    with pytest.raises(UserError, match="no face with three different corners"):
        pc.check(p)


# --- fix round 1 (the review of E1a): inside-out bodies, the wall at the limit, thin shells, damaged files ----------------


def test_a_separate_small_body_written_inside_out_is_a_body_and_not_a_void(tmp_path):
    """Review M1. A 20 mm cube and a 6 mm cube 20 mm apart; the small one has inward normals. It lies in no other shell,
    so it is not a cavity. 2 bodies (the limit is 1: fail), no void, and mesh_closed fails with 1 shell inside out."""
    small = box((40, 0, 0), (46, 6, 6))
    small.invert()
    r = pc.check(save(merge(box((0, 0, 0), (20, 20, 20)), small), tmp_path / "sep_inverted_small.stl"), strict=True)
    c = by_id(r)
    assert c["bodies"]["value"]["count"] == 2 and c["bodies"]["state"] == "fail"
    assert c["closed_voids"]["value"]["count"] == 0 and c["closed_voids"]["state"] == "pass"
    assert c["mesh_closed"]["state"] == "fail" and c["mesh_closed"]["value"]["inverted_shells"] == 1
    assert "1 shell(s) inside out" in c["mesh_closed"]["reason"]
    assert [b["inside_out"] for b in c["bodies"]["value"]["largest"]] == [False, True]
    assert "inside out" in c["bodies"]["reason"]
    assert r["gate"]["state"] == "fail" and set(r["gate"]["failed"]) == {"mesh_closed", "bodies"}


def test_the_big_body_written_inside_out_does_not_make_the_small_body_a_void(tmp_path):
    """Review M1, the other way round: the big cube is inverted, the small one is right. The total volume is negative, so
    the mesh is turned; the small body then has a negative volume and lies in no other shell: a body, not a void."""
    big = box((0, 0, 0), (20, 20, 20))
    big.invert()
    r = pc.check(save(merge(big, box((40, 0, 0), (46, 6, 6))), tmp_path / "sep_inverted_big.stl"))
    c = by_id(r)
    assert c["bodies"]["value"]["count"] == 2 and c["closed_voids"]["value"]["count"] == 0
    assert c["mesh_closed"]["state"] == "fail"


def test_a_cavity_in_a_mesh_that_is_inside_out_all_over_is_still_a_void(tmp_path):
    """Every normal of a cube with a cavity is turned, and the part is far from the origin. The turned mesh is one body
    with one void of 216 mm3; mesh_closed fails (inward), and no shell is inside out relative to the others."""
    m = cavity_box()
    m.invert()
    m.apply_translation([100.0, -50.0, 30.0])
    r = pc.check(save(m, tmp_path / "all_inverted.stl"))
    c = by_id(r)
    assert c["bodies"]["value"]["count"] == 1 and c["closed_voids"]["value"]["total_mm3"] == pytest.approx(216.0)
    assert c["mesh_closed"]["state"] == "fail" and "inward" in c["mesh_closed"]["reason"]
    assert c["mesh_closed"]["value"]["inverted_shells"] == 0


def test_a_body_in_a_void_in_a_body_and_a_cavity_off_centre(tmp_path):
    """A 30 mm cube with a 20 mm cavity (not in the middle) and a 6 mm cube floating in the cavity: 2 bodies and 1 void
    of 8000 mm3, a closed mesh, and a volume of 27000 - 8000 + 216 mm3."""
    outer = box((0, 0, 0), (30, 30, 30))
    void = box((2, 3, 4), (22, 23, 24))
    void.invert()
    r = pc.check(save(merge(outer, void, box((10, 10, 10), (16, 16, 16))), tmp_path / "nested.stl"))
    c = by_id(r)
    assert c["bodies"]["value"]["count"] == 2 and c["closed_voids"]["value"]["total_mm3"] == pytest.approx(8000.0)
    assert c["mesh_closed"]["state"] == "pass" and c["mesh_closed"]["value"]["volume_mm3"] == pytest.approx(19216.0)


def test_a_shell_with_no_area_is_not_a_body(tmp_path):
    """Review m2. A cube and a triangle with three different but collinear corners (area 0). The triangle is a shell, but it
    has no area, so it is not a body: 1 body. It is a fact in mesh_closed (a zero-area face, a shell with no area, 3 open
    edges), and mesh_closed fails because of the open edges."""
    m = box((0, 0, 0), (10, 10, 10))
    verts = np.vstack([m.vertices, [[30, 0, 0], [35, 0, 0], [40, 0, 0]]])
    faces = np.vstack([m.faces, [[8, 9, 10]]])
    r = pc.check(save(trimesh.Trimesh(verts, faces, process=False), tmp_path / "sliver.stl"))
    c = by_id(r)
    assert c["bodies"]["value"]["count"] == 1 and c["bodies"]["state"] == "pass" and c["bodies"]["value"]["shells"] == 2
    assert c["mesh_closed"]["value"]["degenerate_shells"] == 1 and c["mesh_closed"]["value"]["zero_area_faces"] == 1
    assert "shell(s) with no area" in c["mesh_closed"]["reason"] and c["mesh_closed"]["state"] == "fail"


def test_a_wall_exactly_at_the_limit_passes_at_any_position_and_turn(tmp_path):
    """Review M2. A 1.2 mm plate at z 0.1 to 1.3 and a 2.0 mm plate turned 37 degrees about [1, 1, 0] read a few 1e-6 mm short
    of the wall after the float32 rounding. They are at the limit, not under it: pass. A wall that is 0.01 mm under the
    limit still fails (the tolerance is 1e-4 mm)."""
    offset = plate(1.2)
    offset.apply_translation([0, 0, 0.1])
    c = by_id(pc.check(save(offset, tmp_path / "offset.stl"), print_settings={"min_wall_mm": 1.2},
                       resolution_mm=0))["wall_thickness"]
    assert c["state"] == "pass" and c["value"]["area_below_limit_pct"] == 0.0, c["reason"]
    turned = plate(2.0)
    turned.apply_transform(trimesh.transformations.rotation_matrix(np.radians(37), [1, 1, 0]))
    c = by_id(pc.check(save(turned, tmp_path / "turned.stl"), print_settings={"min_wall_mm": 2.0},
                       resolution_mm=0))["wall_thickness"]
    assert c["state"] == "pass" and c["value"]["area_below_limit_pct"] == 0.0, c["reason"]
    c = by_id(pc.check(save(offset, tmp_path / "offset2.stl"), print_settings={"min_wall_mm": 1.21},
                       resolution_mm=0))["wall_thickness"]
    assert c["state"] == "fail" and c["limit"]["chord_tolerance_mm"] == pc.CHORD_TOL_MM


def test_the_wall_share_gives_its_sampling_error(tmp_path):
    """Review n10. The share f has a sampling error of about sqrt(f (1 - f) / samples) (60,000 samples). A 1 mm plate
    against 1.5 mm: f = 85.7 % of the area. The error is in the value (in points), and the text says it."""
    c = by_id(pc.check(save(plate(1.0), tmp_path / "p.stl"), print_settings={"min_wall_mm": 1.5}))["wall_thickness"]
    f, n = c["value"]["area_below_limit_pct"] / 100, c["value"]["samples"]
    assert c["value"]["area_below_limit_sampling_error_pct"] == pytest.approx(100 * np.sqrt(f * (1 - f) / n), abs=1e-3)
    assert "sampling error about" in c["reason"]


def test_a_share_near_the_tolerance_says_that_another_sample_can_change_the_state():
    from topopt.print_report import _Geometry, thickness_estimate

    m = plate(2.0)
    thin, chords = thickness_estimate(_Geometry(m), return_chords=True)
    chords = chords.copy()
    chords[: int(0.005 * len(chords))] = 0.1               # 0.5 % of the samples are thin: at the tolerance
    settings = parse_print_block({"min_wall_mm": 1.0})
    res = pc.resolution_of({"run": None}, 0, 0.5)
    c = pc.check_wall_thickness(thin, chords, 100.0, settings, res, True, {"min_wall_mm": "option"})
    assert c["state"] == "pass" and "within 2 sampling errors of the tolerance" in c["reason"]


# --- the limit keys accept a number in exponent form (review m3) ----------------------------------------------------


def test_exponent_numbers_work_in_the_yaml_the_options_and_the_cli(tmp_path, capsys):
    for text in ("1e5", "1.0e5", "100000", "1.0e+5"):
        assert dict(parse_print_block({"max_support_volume_mm3": text}).limits)["max_support_volume_mm3"] == 1e5
    assert dict(parse_print_block({"min_wall_area_pct": "5e-1"}).limits)["min_wall_area_pct"] == 0.5
    y = tmp_path / "p.yaml"
    y.write_text("print:\n  max_support_volume_mm3: 1e5\n  min_wall_mm: 1.2e0\n")
    r = pc.check(COMB, yaml_path=y)
    assert r["settings"]["limits"]["max_support_volume_mm3"] == {"value": 100000.0, "source": "yaml"}
    code, out, _ = run_cli(capsys, COMB, "--set", "max_support_volume_mm3=1e5", "--json")
    assert code == 0 and json.loads(out)["settings"]["limits"]["max_support_volume_mm3"]["value"] == 100000.0
    with pytest.raises(UserError, match="must be a number"):
        parse_print_block({"max_support_volume_mm3": "a lot"})
    with pytest.raises(UserError, match="must be an integer"):
        parse_print_block({"max_bodies": "1e0"})


def test_a_hand_edited_run_json_with_a_bad_limit_gives_the_default():
    """Review n8: `from_dict` uses the rules of the YAML (max_bodies an integer of 1 or more, min_wall_mm above 0)."""
    ps = PrintSettings.from_dict({"limits": {"max_bodies": 0, "min_wall_mm": 0, "min_wall_area_pct": 101,
                                             "max_closed_void_mm3": -1, "max_overhang_area_mm2": 5.5,
                                             "min_plate_contact_mm2": True}})
    assert dict(ps.limits) == {"max_overhang_area_mm2": 5.5}
    assert ps.limit("max_bodies") == 1 and ps.limit_source("max_bodies") == "assumed"
    assert dict(PrintSettings.from_dict({"limits": {"max_bodies": 2, "min_wall_mm": 0.8}}).limits) == \
        {"min_wall_mm": 0.8, "max_bodies": 2}
    assert dict(PrintSettings.from_dict({"limits": {"max_bodies": 1.5}}).limits) == {}


# --- damaged files (review m6, n5) ------------------------------------------------------------------------------------


def binary_stl(tri):
    """The bytes of a binary STL for triangles `tri` of shape (n, 3, 3)."""
    tri = np.asarray(tri, dtype="<f4")
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = tri
    return b"\0" * 80 + len(tri).to_bytes(4, "little") + rec.tobytes()


def test_files_that_are_not_an_stl_get_a_message_that_says_so(tmp_path):
    (tmp_path / "random.stl").write_bytes(np.random.default_rng(1).bytes(5000))
    (tmp_path / "text.stl").write_text("hello, this is a text file\n" * 20)
    (tmp_path / "empty.stl").write_bytes(b"")
    cube = box((0, 0, 0), (10, 10, 10))
    good = binary_stl(cube.triangles)
    (tmp_path / "cut.stl").write_bytes(good[: 84 + 50 * 5])
    (tmp_path / "tail.stl").write_bytes(good + b"x")
    for name, word in (("random", "not an STL file"), ("text", "not an STL file"), ("empty", "is empty"),
                       ("cut", "truncated or broken"), ("tail", "truncated or broken")):
        with pytest.raises(UserError, match=word) as e:
            pc.check(tmp_path / f"{name}.stl")
        assert "charset_normalizer" not in str(e.value) and "ModuleNotFoundError" not in str(e.value)
    (tmp_path / "good.stl").write_bytes(good)
    assert pc.check(tmp_path / "good.stl")["input"]["triangles"] == 12


def test_a_binary_stl_whose_header_starts_with_solid_is_read(tmp_path):
    """Some exporters write `solid` in the 80-byte header of a binary file: the size decides."""
    cube = box((0, 0, 0), (10, 10, 10))
    data = b"solid exporter" + binary_stl(cube.triangles)[14:]
    (tmp_path / "solid.stl").write_bytes(data)
    assert by_id(pc.check(tmp_path / "solid.stl"))["mesh_closed"]["state"] == "pass"


def test_a_non_finite_coordinate_is_a_user_error_and_not_a_report_with_nan(tmp_path, capsys):
    tri = box((0, 0, 0), (10, 10, 10)).triangles.copy()
    tri[3, 1, 0] = np.nan
    (tmp_path / "nan.stl").write_bytes(binary_stl(tri))
    with pytest.raises(UserError, match="non-finite coordinates"):
        pc.check(tmp_path / "nan.stl")
    code, out, err = run_cli(capsys, tmp_path / "nan.stl", "--json")
    assert code == 1 and out == "" and err.startswith("error:") and "NaN" not in out


def test_the_text_at_resolution_zero_says_the_mesh_is_exact(tmp_path):
    p = save(plate(2.0), tmp_path / "p.stl")
    c = by_id(pc.check(p, print_settings={"min_wall_mm": 1.5}, resolution_mm=0))["wall_thickness"]
    assert c["state"] == "pass" and "option says the mesh is exact" in c["reason"]
    assert "0 mm (3 x 0 mm" not in c["reason"]


def test_the_resolution_rule_is_worded_as_a_rule_of_this_check(tmp_path):
    """Review m5. A 12-triangle 2 mm plate measures 2.0 mm, but the median edge is large: the state is unknown, and the text
    says that this is the rule of this check, not a fact of the mesh. A run folder adds that thin members were
    under-resolved in the analysis."""
    p = save(box((0, 0, 0), (12, 12, 2)), tmp_path / "coarse.stl")
    c = by_id(pc.check(p, print_settings={"min_wall_mm": 1.5}))["wall_thickness"]
    assert c["state"] == "unknown" and "by the rule of this check" in c["reason"]
    assert "pitfalls/mesh-coarser-than-members" in c["reason"] and "cannot resolve" not in c["reason"]
    assert "under-resolved in the analysis" not in c["reason"]            # a bare STL has no analysis
    run = make_run(tmp_path, plate(2.0), h=1.2)
    c = by_id(pc.check(run, print_settings={"min_wall_mm": 1.5}))["wall_thickness"]
    assert c["state"] == "unknown" and "under-resolved in the analysis of the run" in c["reason"]
    assert "3.6 mm" in c["reason"]
    nl = by_id(pc.check(run))["wall_thickness"]
    assert "The rule of this check: a pass needs a limit of at least 3 x the resolution" in nl["reason"]


def test_an_unknown_by_the_rule_says_what_a_pass_needs_for_a_run_folder(tmp_path):
    """P4, F4-3. A 2 mm plate on a 1.2 mm grid against a 1.5 mm limit: the measure finds no thin wall, and the rule holds
    the pass back. The result says the resolution that a pass needs as a number, the voxel count that this costs, and
    that the measured share is within the tolerance. The number is the rule turned round: the state at the resolution
    that the result names is a pass."""
    run = make_run(tmp_path, plate(2.0), h=1.2)
    c = by_id(pc.check(run, print_settings={"min_wall_mm": 1.5}))["wall_thickness"]
    need = c["value"]["pass_needs"]
    assert c["state"] == "unknown"
    assert need == {"resolution_mm_at_most": pytest.approx(0.5),
                    "voxel_count_factor": pytest.approx((1.2 / 0.5) ** 3, abs=0.1)}
    r = c["reason"]
    assert "A pass needs a resolution of at most 0.5 mm (the limit divided by 3)" in r
    assert "For this run folder that is a voxel size of 0.5 mm or less." in r
    assert "That is about 13.8 times the voxels of the 1.2 mm grid" in r
    assert "A run at that voxel size makes a new shape. The state is pass there if its measured share stays within the tolerance" in r
    assert "The measured share of this STL is within the tolerance, so the state is then pass" in r
    assert ".." not in r
    # the rule turned round: the resolution that the result names gives the pass
    at_target = by_id(pc.check(run, print_settings={"min_wall_mm": 1.5},
                               resolution_mm=need["resolution_mm_at_most"]))["wall_thickness"]
    assert at_target["state"] == "pass" and "pass_needs" not in at_target["value"]
    above = by_id(pc.check(run, print_settings={"min_wall_mm": 1.5}, resolution_mm=need["resolution_mm_at_most"] * 1.1))
    assert above["wall_thickness"]["state"] == "unknown"                  # a little above the target is no pass


def test_pass_needs_for_a_bare_stl_and_for_an_option_resolution(tmp_path):
    p = save(box((0, 0, 0), (12, 12, 2)), tmp_path / "coarse.stl")      # 12 triangles: the median edge is the resolution
    c = by_id(pc.check(p, print_settings={"min_wall_mm": 1.5}))["wall_thickness"]
    need = c["value"]["pass_needs"]
    assert c["state"] == "unknown" and "voxel_count_factor" not in need and need["resolution_mm_at_most"] == 0.5
    assert "a mesh with a median edge of 0.5 mm or less (now" in c["reason"] and "For this run folder" not in c["reason"]
    assert "exact CAD" in c["reason"]
    o = by_id(pc.check(p, print_settings={"min_wall_mm": 1.5}, resolution_mm=1.0))["wall_thickness"]
    assert o["state"] == "unknown" and o["value"]["pass_needs"]["resolution_mm_at_most"] == 0.5
    assert "Give resolution_mm of 0.5 or less (now 1)" in o["reason"]
    assert "The measured share of this STL is within the tolerance, so the state is then pass" in o["reason"]


@pytest.mark.parametrize("limit, plate_mm, h, expect", [
    (1.5, 2.0, 1.2, 0.5), (2.0, 3.0, 1.2, 0.6666), (0.8, 1.2, 0.5, 0.2666), (1.4, 2.0, 1.0, 0.4666), (0.5, 1.0, 0.4, 0.1666)])
def test_the_named_resolution_is_rounded_down_so_that_it_gives_the_pass(tmp_path, limit, plate_mm, h, expect):
    """P4 review, M2. `limit / 3` was rounded to the nearest 4 decimals: 0.6667 for a limit of 2.0 mm, and at 0.6667 the state is
    `unknown` again (3 x 0.6667 is over 2.0). The value is rounded down. The key and the text have the same value, and the
    state at that value is a pass for each limit."""
    run = make_run(tmp_path, plate(plate_mm), h=h)
    c = by_id(pc.check(run, print_settings={"min_wall_mm": limit}))["wall_thickness"]
    named = c["value"]["pass_needs"]["resolution_mm_at_most"]
    assert named == pytest.approx(expect, abs=1e-9)
    assert f"a resolution of at most {expect:g} mm" in c["reason"] and f"voxel size of {expect:g} mm or less" in c["reason"]
    at = by_id(pc.check(run, print_settings={"min_wall_mm": limit}, resolution_mm=named))["wall_thickness"]
    assert at["state"] == "pass", (limit, named)


def test_the_text_never_names_a_larger_number_than_the_key():
    """`_num` rounds to 4 significant digits and can round up (10.3366 -> 10.34). The text uses the stored value."""
    res = pc.resolution_of({"run": None}, None, 1.0)
    needs, text = pc._pass_needs(31.0099, res)
    assert needs["resolution_mm_at_most"] == 10.3366 and "at most 10.3366 mm" in text and "10.34" not in text


def test_the_run_folder_branch_is_decided_like_resolution_of(tmp_path):
    """P4 review, L6. A voxel size with more than 4 decimals that rounds up (0.12346 -> 0.1235) took the wrong branch: the text
    spoke of a median edge. `resolution_of` takes the voxel size when it is not under the median edge; so does the text."""
    inp = {"run": {"h": 0.12346}}
    res = pc.resolution_of(inp, None, 0.05)
    assert res["resolution_mm"] == 0.1235 and res["voxel_size_mm"] == 0.12346
    needs, text = pc._pass_needs(0.375, res)
    assert "For this run folder that is a voxel size of 0.125 mm or less" in text and "voxel_count_factor" in needs
    res = pc.resolution_of(inp, None, 0.2)                                    # the median edge is larger: it sets the resolution
    needs, text = pc._pass_needs(0.9, res)
    assert "a mesh with a median edge of 0.3 mm or less" in text and "voxel_count_factor" not in needs


def test_pass_needs_is_only_in_the_unknown_by_the_rule(tmp_path):
    """A fail (a measured thin wall, at any resolution), a pass and a missing limit do not carry the key."""
    thin = save(box((0, 0, 0), (12, 12, 1)), tmp_path / "thin.stl")
    assert "pass_needs" not in by_id(pc.check(thin, print_settings={"min_wall_mm": 1.5}))["wall_thickness"]["value"]
    ok = save(plate(2.0), tmp_path / "ok.stl")
    assert "pass_needs" not in by_id(pc.check(ok, print_settings={"min_wall_mm": 1.5}, resolution_mm=0))["wall_thickness"]["value"]
    assert "pass_needs" not in by_id(pc.check(ok))["wall_thickness"]["value"]


# --- known answers: the V2 comb fixture (160 / 200 / 640) ---------------------------------------------------------


def test_comb_has_the_v2_numbers_and_the_limits_set_the_state():
    r = pc.check(COMB)
    c = by_id(r)
    assert c["overhang"]["value"]["area_mm2"] == pytest.approx(160.0)
    assert c["plate_contact"]["value"]["area_mm2"] == pytest.approx(200.0)
    assert c["support_volume"]["value"]["volume_mm3"] == pytest.approx(640.0)
    assert c["wall_thickness"]["value"]["thinnest_member_mm"] == pytest.approx(2.0, rel=0.01)   # the arms
    # no limit key: the state is unknown, the limit is empty
    for cid in ("overhang", "support_volume", "plate_contact"):
        assert c[cid]["state"] == "unknown" and c[cid]["limit"]["value"] is None and c[cid]["limit"]["source"] == "none"
    assert c["bed_fit"]["state"] == "pass" and c["bed_fit"]["limit"]["source"] == "assumed"
    assert c["bed_fit"]["limit"]["value"] == [220.0, 220.0]
    # the limit is a bar on the number: at the number it passes (<= and >=), beside it it fails
    def state(key, value, cid):
        return by_id(pc.check(COMB, print_settings={key: value}))[cid]["state"]
    assert state("max_overhang_area_mm2", 160, "overhang") == "pass"
    assert state("max_overhang_area_mm2", 159.9, "overhang") == "fail"
    assert state("max_support_volume_mm3", 640, "support_volume") == "pass"
    assert state("max_support_volume_mm3", 639, "support_volume") == "fail"
    assert state("min_plate_contact_mm2", 200, "plate_contact") == "pass"
    assert state("min_plate_contact_mm2", 201, "plate_contact") == "fail"
    assert state("bed_mm", 10, "bed_fit") == "fail"        # the part is 20 x 10 x 10
    # a limit of the arm thickness (2 mm): 1.5 passes, 3 fails (the arms are 2 mm thick); resolution 0 = exact CAD
    arm = lambda limit: by_id(pc.check(COMB, print_settings={"min_wall_mm": limit}, resolution_mm=0))["wall_thickness"]
    assert arm(1.5)["state"] == "pass" and arm(3.0)["state"] == "fail"


def test_overhang_with_no_overhang_passes_with_no_limit(tmp_path):
    p = save(box((0, 0, 0), (10, 10, 10)), tmp_path / "cube.stl")
    c = by_id(pc.check(p))
    assert c["overhang"]["state"] == "pass" and c["overhang"]["value"]["area_mm2"] == 0.0
    assert c["support_volume"]["state"] == "pass"


def test_sls_needs_no_support(tmp_path):
    r = pc.check(COMB, print_settings={"printer_type": "sls"})
    c = by_id(r)["support_volume"]
    assert c["state"] == "pass" and c["value"]["support_needed"] is False and c["value"]["volume_mm3"] is None


def test_the_v2_numbers_are_the_v2_code_and_equal_build_report(tmp_path):
    """The numbers come from print_report (not written again): they equal `build_report` for the same mesh."""
    p = save(cavity_box(), tmp_path / "cavity.stl")
    mesh = trimesh.load(p, force="mesh")
    for path, settings in ((COMB, PrintSettings()), (p, parse_print_block({"build_direction": "-x"}))):
        mesh = trimesh.load(path, force="mesh")
        want = pr.build_report(mesh, settings)["configured"]
        c = by_id(pc.check(path, print_settings={"build_direction": list(settings.build_direction)}))
        assert c["overhang"]["value"]["area_mm2"] == want["overhang_area_mm2"]
        assert c["plate_contact"]["value"]["area_mm2"] == want["plate_contact_area_mm2"]
        assert c["support_volume"]["value"]["volume_mm3"] == want["support_volume_mm3"]
        assert c["bed_fit"]["value"]["bed_margin_mm"] == want["bed_margin_mm"]


# --- states, the gate and the report ---------------------------------------------------------------------------------


def test_every_check_has_a_value_a_limit_with_a_source_a_state_and_a_reason(tmp_path):
    r = pc.check(save(cavity_box(), tmp_path / "c.stl"), print_settings={"min_wall_mm": 1.0})
    assert [c["id"] for c in r["checks"]] == ["mesh_closed", "bodies", "closed_voids", "bed_fit", "overhang",
                                              "support_volume", "plate_contact", "wall_thickness"]
    for c in r["checks"]:
        assert set(c) == {"id", "state", "reason", "value", "limit", "method", "estimate"}
        assert c["state"] in ("pass", "fail", "unknown") and c["reason"] and c["method"]
        assert set(c["limit"]) >= {"value", "key", "source"}
        assert c["limit"]["source"] in ("yaml", "run.json", "option", "assumed", "none", "definition")
    s = r["summary"]
    assert s["pass"] + s["fail"] + s["unknown"] == len(r["checks"])
    assert s["failed"] == [c["id"] for c in r["checks"] if c["state"] == "fail"]
    assert r["ok"] is True and set(r["definitions"]) >= {c["id"] for c in r["checks"]}
    json.dumps(r)                                          # plain JSON types only


def test_default_only_reports_and_the_strict_gate_is_opt_in(tmp_path):
    p = save(merge(box((0, 0, 0), (10, 10, 10)), box((20, 0, 0), (30, 10, 10))), tmp_path / "two.stl")
    r = pc.check(p)
    assert r["summary"]["failed"] == ["bodies"]
    assert r["gate"]["state"] == "off" and r["gate"]["exit_code"] == 0 and r["gate"]["strict"] is False
    s = pc.check(p, strict=True)
    g = s["gate"]
    assert g["state"] == "fail" and g["exit_code"] == pc.GATE_EXIT_CODE == 4 and g["failed"] == ["bodies"]
    assert "max_bodies" in g["override"] and "without --strict" in g["override"] and "No run stops" in g["override"]
    assert 3 not in (g["exit_code"],)                      # 3 is the exit code of `topopt verify --allowable-mpa`
    # the override named in the text is true: the limit set in the print block makes the gate pass
    assert pc.check(p, strict=True, print_settings={"max_bodies": 2})["gate"]["state"] == "pass"


def test_unknown_never_sets_the_gate(tmp_path):
    """The comb has four unknown checks (no limit) and no fail: the strict gate passes."""
    r = pc.check(COMB, strict=True)
    assert r["summary"]["unknown"] == 4 and r["summary"]["fail"] == 0
    assert r["gate"]["state"] == "pass" and r["gate"]["exit_code"] == 0 and r["gate"]["override"] is None
    # a coarse mesh gives unknown for the thickness, and the strict gate still passes
    p = save(box((0, 0, 0), (12, 12, 2)), tmp_path / "coarse.stl")
    s = pc.check(p, strict=True, print_settings={"min_wall_mm": 1.5})
    assert by_id(s)["wall_thickness"]["state"] == "unknown" and s["gate"]["state"] == "pass"


def test_an_open_mesh_fails_the_strict_gate_and_names_the_mesh_fix(tmp_path):
    m = box((0, 0, 0), (10, 10, 10))
    keep = m.triangles_center[:, 2] < 9.9
    p = save(trimesh.Trimesh(m.vertices, m.faces[keep], process=False), tmp_path / "open.stl")
    g = pc.check(p, strict=True)["gate"]
    assert g["state"] == "fail" and g["failed"] == ["mesh_closed"] and "repair the mesh" in g["override"]
    assert "mesh_closed has no print key" in g["override"] and "run without --strict" in g["override"]


def test_input_kinds_final_nominal_and_edited(tmp_path):
    """A bare STL with any name works: no metrics.json, no YAML. A run folder works through result/final.stl."""
    part = cavity_box()
    final = save(part, tmp_path / "final.stl")
    nominal = save(part, tmp_path / "nominal.stl")                 # a C2-style file name
    edited_dir = tmp_path / "edits"
    edited_dir.mkdir()
    edited = save(part, edited_dir / "edited.stl")                 # a C3a-style file in a folder with no metrics.json
    run = make_run(tmp_path, part)
    reports = [pc.check(x) for x in (final, nominal, edited, run)]
    assert [r["input"]["kind"] for r in reports] == ["stl", "stl", "stl", "run"]
    for r in reports:
        assert r["ok"] and states(r) == states(reports[0])
        assert by_id(r)["closed_voids"]["value"]["total_mm3"] == pytest.approx(216.0)
    assert reports[0]["input"]["sha256"] == hashlib.sha256(final.read_bytes()).hexdigest()
    assert not (edited_dir / "metrics.json").exists()              # nothing is written


def test_the_check_writes_nothing_into_a_run_folder(tmp_path):
    run = make_run(tmp_path, plate(2.0), h=0.3)
    def snapshot():
        return {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file()}
    before = snapshot()
    pc.check(run, strict=True)
    assert snapshot() == before


def test_settings_come_from_run_json_then_yaml_then_options(tmp_path):
    run = make_run(tmp_path, plate(2.0), h=0.3, print_block={"bed_mm": [100, 100], "given": ["bed_mm"],
                                                              "limits": {"min_wall_mm": 1.0}})
    y = tmp_path / "p.yaml"
    y.write_text("print:\n  min_wall_mm: 1.5\n  printer_type: sla\n")     # a YAML with only a print block works
    r = pc.check(run, yaml_path=y, print_settings={"min_wall_mm": 1.8})
    s = r["settings"]
    assert s["from"] == ["defaults", "run.json", "yaml p.yaml", "options"]
    assert s["bed_mm"] == [100.0, 100.0] and s["printer_type"] == "sla"
    assert s["limits"]["min_wall_mm"] == {"value": 1.8, "source": "option"}          # the last source that gave it
    assert by_id(r)["bed_fit"]["limit"] == {"value": [100.0, 100.0], "key": "bed_mm", "source": "run.json"}
    assert by_id(r)["wall_thickness"]["limit"]["source"] == "option" and "(option)" not in by_id(r)["bed_fit"]["reason"]
    assert pc.check(run, yaml_path=y)["settings"]["limits"]["min_wall_mm"]["source"] == "yaml"
    assert pc.check(run)["settings"]["limits"]["min_wall_mm"]["source"] == "run.json"
    only_yaml = pc.check(run, yaml_path=y)["settings"]
    assert only_yaml["limits"]["min_wall_mm"]["value"] == 1.5


def test_wrong_inputs_are_user_errors(tmp_path):
    with pytest.raises(UserError, match="file not found"):
        pc.check(tmp_path / "none.stl")
    with pytest.raises(UserError, match="no result/final.stl"):
        pc.check(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(UserError, match="not an STL"):
        pc.check(tmp_path / "a.txt")
    (tmp_path / "bad.stl").write_bytes(b"this is not an stl")
    with pytest.raises(UserError, match="is not an STL file"):
        pc.check(tmp_path / "bad.stl")
    with pytest.raises(UserError, match="YAML file not found"):
        pc.check(COMB, yaml_path=tmp_path / "none.yaml")
    with pytest.raises(UserError, match="unknown key"):
        pc.check(COMB, print_settings={"min_wal": 1})
    with pytest.raises(UserError, match="resolution_mm"):
        pc.check(COMB, resolution_mm=-1)


def test_the_module_needs_no_optional_package():
    """Geometry only: after a full check no GPL or optional package is loaded."""
    code = ("import sys; from topopt import printcheck as p; p.check(sys.argv[1]); "
            "bad = [m for m in ('gmsh', 'pypardiso', 'cholespy', 'skfem', 'pyopencl', 'mcp') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code, str(COMB)], check=True, cwd=str(ROOT))


# --- the command ---------------------------------------------------------------------------------------------------


def run_cli(capsys, *argv):
    code = main(["printcheck", *map(str, argv)])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_exits_0_for_any_check_result(tmp_path, capsys):
    p = save(merge(box((0, 0, 0), (10, 10, 10)), box((20, 0, 0), (30, 10, 10))), tmp_path / "two.stl")
    code, out, err = run_cli(capsys, p)
    assert code == 0 and "fail     bodies" in out and "gate: off" in out and err == ""
    assert "summary: " in out and "failed: bodies" in out


def test_cli_strict_exit_code_and_override_text(tmp_path, capsys):
    p = save(merge(box((0, 0, 0), (10, 10, 10)), box((20, 0, 0), (30, 10, 10))), tmp_path / "two.stl")
    code, out, _ = run_cli(capsys, p, "--strict")
    assert code == 4 and "gate: FAIL (strict), exit code 4" in out and "max_bodies" in out and "without --strict" in out
    code, out, _ = run_cli(capsys, p, "--strict", "--set", "max_bodies=2")
    assert code == 0 and "gate: pass (strict)" in out


def test_cli_exit_code_1_only_for_an_input_that_cannot_be_read(tmp_path, capsys):
    for argv in ([tmp_path / "none.stl"], [tmp_path], [COMB, "--set", "nokey=1"], [COMB, "--set", "min_wall_mm"],
                 [COMB, "--set", "min_wall_mm=-2"], [COMB, "--yaml", tmp_path / "none.yaml"]):
        code, out, err = run_cli(capsys, *argv)
        assert code == 1 and out == "" and err.startswith("error: ") and err.count("\n") == 1, (argv, err)


def test_cli_json_is_the_report(tmp_path, capsys):
    code, out, _ = run_cli(capsys, COMB, "--json", "--set", "min_wall_mm=1.5", "--set", "bed_mm=[100, 80]",
                           "--resolution-mm", "0")
    r = json.loads(out)
    assert code == 0 and r["ok"] and by_id(r)["wall_thickness"]["state"] == "pass"
    assert r["settings"]["bed_mm"] == [100.0, 80.0] and r["settings"]["from"] == ["defaults", "options"]


def test_cli_text_names_the_check_the_state_and_the_reason(capsys):
    code, out, _ = run_cli(capsys, COMB, "--set", "max_overhang_area_mm2=100")
    assert code == 0
    for word in ("mesh_closed", "closed_voids", "bed_fit", "support_volume", "plate_contact", "wall_thickness"):
        assert word in out
    assert "fail     overhang" in out and "the limit is 100 mm2 (option)" in out


# --- the MCP tool ------------------------------------------------------------------------------------------------


def strip_time(report):
    r = json.loads(json.dumps(report))
    r.pop("elapsed_s", None)
    r.pop("timing_s", None)
    return r


@pytest.mark.parametrize("extra, tool_args", [
    ([], {}),
    (["--strict"], {"strict": True}),
    (["--set", "min_wall_mm=1.5", "--set", "max_overhang_area_mm2=100", "--strict"],
     {"print_settings": {"min_wall_mm": 1.5, "max_overhang_area_mm2": 100}, "strict": True}),
    (["--resolution-mm", "0", "--set", "min_wall_mm=1.5"], {"resolution_mm": 0, "print_settings": {"min_wall_mm": 1.5}}),
])
def test_mcp_tool_returns_the_same_report_as_the_cli(capsys, extra, tool_args):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    code, out, _ = run_cli(capsys, COMB, "--json", *extra)
    result = server.print_check(path=str(COMB), **tool_args)
    assert strip_time(result) == strip_time(json.loads(out))
    assert result["gate"]["exit_code"] == code


def test_mcp_tool_run_folder_yaml_and_errors(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    make_run(tmp_path, plate(2.0), h=0.3, name="runs/one")
    y = tmp_path / "p.yaml"
    y.write_text("print: {min_wall_mm: 1.5}\n")
    r = server.print_check(path="runs/one", yaml_path="p.yaml")           # relative paths start at the work folder
    assert r["ok"] and by_id(r)["wall_thickness"]["state"] == "pass" and r["input"]["kind"] == "run"
    assert server.print_check(path="runs/none") == {
        "ok": False, "error": f"file not found: {tmp_path / 'runs' / 'none'}"}
    bad = server.print_check(path="runs/one", print_settings={"nokey": 1})
    assert bad["ok"] is False and "unknown key" in bad["error"]
    # no path: the last run; with none, a plain error as data
    assert server.print_check()["ok"] is False
    monkeypatch.setitem(server._current, "run_dir", tmp_path / "runs" / "one")
    assert server.print_check()["ok"] is True


def test_mcp_tool_is_listed_with_a_description_and_the_instructions_name_it():
    pytest.importorskip("mcp")
    import asyncio
    from topopt_mcp import server
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert "print_check" in tools and "gate" in tools["print_check"].description
    assert len(tools["print_check"].description) <= 4096
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert "print_check" in server.INSTRUCTIONS and len(server.INSTRUCTIONS) <= OLD_CLIENT_CUT
    assert "must" not in server.INSTRUCTIONS.lower() and "never" not in server.INSTRUCTIONS.lower()
    for key in PRINT_LIMIT_KEYS:
        assert key in tools["print_check"].description
