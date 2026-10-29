"""`find_features` (CONTRACTS.md section 45): the holes, bosses and flat faces of an STL, and the read-only MCP tool.

The parts are built in the tests with manifold3d (a plate with holes, a boss, a counterbore) and numpy. The wall vertices of such a cylinder
lie exactly on the radius. The STL file keeps float32 and the reply rounds a length to 1e-4 mm, so a radius, a position or a length is tested
to 1e-4 mm (`TOL_MM`); a unit vector to 1e-5 (`TOL_UNIT`). The engine module is imported inside the tests, so a run on a commit without it
fails each test on its own.
"""
import asyncio
import inspect
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from manifold3d import Manifold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOL_MM = 1e-4
TOL_UNIT = 1e-5
HABIT = re.compile(r"\b(must|never|always|conservative)\b", re.I)


def ff():
    from topopt import features
    return features


# --- parts ---------------------------------------------------------------------------------------------------------------------------


def cylinder(r, h, seg, at):
    return Manifold.cylinder(h, r, r, seg).translate(at)


def arrays(m):
    mesh = m.to_mesh64()
    return np.asarray(mesh.vert_properties)[:, :3].copy(), np.asarray(mesh.tri_verts).astype(np.int64)


def plate(seg=64, holes=((10, 10, 3.0), (30, 10, 2.0)), size=(40, 20, 4)):
    """A plate with through holes along z: x, y, radius. The first hole has the radius 3, the second 2."""
    m = Manifold.cube(size)
    for x, y, r in holes:
        m = m - cylinder(r, size[2] + 2, seg, (x, y, -1))
    return m


def write_stl(m_or_arrays, path):
    V, F = m_or_arrays if isinstance(m_or_arrays, tuple) else arrays(m_or_arrays)
    trimesh.Trimesh(V, F, process=False).export(str(path))
    return path


def polygon_area(r, seg):
    return 0.5 * seg * r * r * math.sin(2.0 * math.pi / seg)


def rotation(ax, ay, az):
    a, b, c = (math.radians(v) for v in (ax, ay, az))
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def angle_between_lines(a, b):
    """The angle in degrees between two lines (the sign of a direction does not count). Stable near 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.asin(min(1.0, float(np.linalg.norm(np.cross(a / np.linalg.norm(a), b / np.linalg.norm(b)))))))


def by_radius(entries):
    return {round(e["radius_mm"]): e for e in entries}


# --- the plate with two through holes (the card's test part) ----------------------------------------------------------------------


def test_plate_with_two_through_holes(tmp_path):
    r = ff().find_features(write_stl(plate(64), tmp_path / "plate.stl"))
    assert r["ok"] is True
    assert [h["id"] for h in r["holes"]] == ["H1", "H2"]
    assert [h["radius_mm"] for h in r["holes"]] == pytest.approx([2.0, 3.0], abs=TOL_MM)      # the order: radius, smallest first
    small, big = r["holes"]
    for hole, (x, y), radius in ((small, (30, 10), 2.0), (big, (10, 10), 3.0)):
        assert hole["radius_mm"] == pytest.approx(radius, abs=TOL_MM)
        assert hole["diameter_mm"] == pytest.approx(2 * radius, abs=2 * TOL_MM)
        assert abs(hole["axis"][2]) == pytest.approx(1.0, abs=TOL_UNIT) and np.linalg.norm(hole["axis"]) == pytest.approx(1.0, abs=TOL_UNIT)
        assert hole["length_mm"] == pytest.approx(4.0, abs=TOL_MM)
        assert hole["depth"] == "through" and hole["ends"] == ["open", "open"]
        assert hole["point"][:2] == pytest.approx([x, y], abs=TOL_MM) and hole["end_point"][:2] == pytest.approx([x, y], abs=TOL_MM)
        assert sorted([hole["point"][2], hole["end_point"][2]]) == pytest.approx([0.0, 4.0], abs=TOL_MM)
        assert hole["closed_ring"] is True and hole["arc_deg"] == pytest.approx(360.0)
        assert hole["segments"] == 64 and hole["faces"] == 128
        assert hole["coaxial_with"] == []
    assert small["area_mm2"] == pytest.approx(2 * math.pi * 2.0 * 4.0, rel=2e-3)                 # the wall: 64 flat strips, 0.1 % under the cylinder
    assert big["fit"]["rms_mm"] <= 1e-5 and big["fit"]["max_mm"] <= 1e-5 and big["fit"]["normal_deviation_deg"] < 1e-3
    assert r["bosses"] == []
    assert r["mesh"]["watertight"] is True and r["mesh"]["faces"] == 532 and r["mesh"]["volume_mm3"] == pytest.approx(3036.899, abs=1e-2)
    assert r["mesh"]["bounds_mm"] == [[0.0, 0.0, 0.0], [40.0, 20.0, 4.0]]


def test_the_reply_is_json_safe_and_the_unit_vectors_are_unit_vectors(tmp_path):
    r = ff().find_features(write_stl(plate(32), tmp_path / "plate.stl"))
    json.dumps(r, allow_nan=False)
    for item in r["holes"] + r["bosses"]:
        assert np.linalg.norm(item["axis"]) == pytest.approx(1.0, abs=TOL_UNIT)
    for face in r["flat_faces"]:
        assert np.linalg.norm(face["normal"]) == pytest.approx(1.0, abs=TOL_UNIT)


def test_a_plain_box_has_no_hole_and_six_flat_faces(tmp_path):
    r = ff().find_features(write_stl(Manifold.cube((10, 20, 30)), tmp_path / "box.stl"))
    assert r["holes"] == [] and r["bosses"] == []
    faces = r["flat_faces"]
    assert len(faces) == 6
    got = {tuple(f["normal"]): f for f in faces}
    expect = {(-1.0, 0.0, 0.0): (600.0, 0.0), (1.0, 0.0, 0.0): (600.0, 10.0), (0.0, -1.0, 0.0): (300.0, 0.0),
              (0.0, 1.0, 0.0): (300.0, 20.0), (0.0, 0.0, -1.0): (200.0, 0.0), (0.0, 0.0, 1.0): (200.0, 30.0)}
    assert set(got) == set(expect)                                  # six outward normals
    for normal, (area, offset) in expect.items():
        assert got[normal]["area_mm2"] == pytest.approx(area, abs=1e-3)
        assert got[normal]["offset_mm"] == pytest.approx(offset, abs=TOL_MM)       # normal . x of a point of the plane
    assert [f["area_mm2"] for f in faces] == sorted((f["area_mm2"] for f in faces), reverse=True)     # the order: the largest first
    assert [f["id"] for f in faces] == [f"F{i}" for i in range(1, 7)]
    assert faces[0]["centroid"] == pytest.approx([0.0, 10.0, 15.0], abs=TOL_MM) or faces[0]["centroid"] == pytest.approx([10.0, 10.0, 15.0], abs=TOL_MM)


def test_flat_faces_of_the_plate_are_the_face_areas_minus_the_holes(tmp_path):
    seg = 64
    r = ff().find_features(write_stl(plate(seg), tmp_path / "plate.stl"))
    assert len(r["flat_faces"]) == 6
    hole_area = polygon_area(3.0, seg) + polygon_area(2.0, seg)
    top = [f for f in r["flat_faces"] if f["normal"] == [0.0, 0.0, 1.0]]
    bottom = [f for f in r["flat_faces"] if f["normal"] == [0.0, 0.0, -1.0]]
    assert len(top) == 1 and len(bottom) == 1
    for face in (top[0], bottom[0]):
        assert face["area_mm2"] == pytest.approx(800.0 - hole_area, abs=2e-3)
    assert top[0]["offset_mm"] == pytest.approx(4.0, abs=TOL_MM) and bottom[0]["offset_mm"] == pytest.approx(0.0, abs=TOL_MM)
    sides = sorted(f["area_mm2"] for f in r["flat_faces"] if abs(f["normal"][2]) < 0.5)
    assert sides == pytest.approx([80.0, 80.0, 160.0, 160.0], abs=1e-3)


# --- blind hole, boss, stepped hole -------------------------------------------------------------------------------------------------


def test_a_blind_hole_is_blind_and_has_the_right_length(tmp_path):
    m = Manifold.cube((40, 20, 4)) - cylinder(3.0, 10, 32, (10, 10, 1.5))        # 2.5 mm deep from the top face
    r = ff().find_features(write_stl(m, tmp_path / "blind.stl"))
    assert len(r["holes"]) == 1
    h = r["holes"][0]
    assert h["depth"] == "blind" and h["ends"] == ["open", "closed"]
    assert h["length_mm"] == pytest.approx(2.5, abs=TOL_MM) and h["radius_mm"] == pytest.approx(3.0, abs=TOL_MM)
    assert h["axis"] == pytest.approx([0.0, 0.0, -1.0], abs=TOL_UNIT)                    # from the open end (z = 4) to the closed end (z = 1.5)
    assert h["point"] == pytest.approx([10.0, 10.0, 4.0], abs=TOL_MM) and h["end_point"] == pytest.approx([10.0, 10.0, 1.5], abs=TOL_MM)
    bottom = [f for f in r["flat_faces"] if f["normal"] == [0.0, 0.0, 1.0] and abs(f["offset_mm"] - 1.5) < 1e-3]
    assert len(bottom) == 1 and bottom[0]["area_mm2"] == pytest.approx(polygon_area(3.0, 32), abs=1e-3)       # the bottom is a flat face


def test_a_boss_on_a_plate_is_not_a_hole(tmp_path):
    m = Manifold.cube((40, 20, 4)) + cylinder(3.0, 8, 32, (20, 10, 4))
    r = ff().find_features(write_stl(m, tmp_path / "boss.stl"))
    assert r["holes"] == []
    assert len(r["bosses"]) == 1
    b = r["bosses"][0]
    assert b["id"] == "B1" and b["radius_mm"] == pytest.approx(3.0, abs=TOL_MM) and b["length_mm"] == pytest.approx(8.0, abs=TOL_MM)
    assert abs(b["axis"][2]) == pytest.approx(1.0, abs=TOL_UNIT) and "depth" not in b
    assert b["point"][:2] == pytest.approx([20.0, 10.0], abs=TOL_MM)


def test_a_stepped_hole_is_two_cylinders_that_name_each_other(tmp_path):
    m = Manifold.cube((40, 20, 4)) - cylinder(2.0, 10, 32, (20, 10, -1)) - cylinder(4.0, 10, 32, (20, 10, 2.5))
    r = ff().find_features(write_stl(m, tmp_path / "counterbore.stl"))
    small, big = r["holes"]
    assert (small["radius_mm"], big["radius_mm"]) == pytest.approx((2.0, 4.0), abs=TOL_MM)
    assert small["depth"] == "through" and small["length_mm"] == pytest.approx(2.5, abs=TOL_MM)
    assert big["depth"] == "blind" and big["length_mm"] == pytest.approx(1.5, abs=TOL_MM) and big["ends"] == ["open", "closed"]
    assert small["coaxial_with"] == [big["id"]] and big["coaxial_with"] == [small["id"]]
    assert any("shares its axis" in fact for fact in r["facts"])


# --- the orientation, the size of the facets ---------------------------------------------------------------------------------------


def test_a_plate_rotated_by_a_general_rotation_has_the_same_holes_with_the_axis_rotated(tmp_path):
    V, F = arrays(plate(32))
    R = rotation(20.0, 35.0, 50.0)                                   # no axis of the plate stays an axis of the room
    t = np.array([12.5, -7.25, 30.0])
    r = ff().find_features(write_stl((V @ R.T + t, F), tmp_path / "rotated.stl"))
    assert len(r["holes"]) == 2 and len(r["flat_faces"]) == 6
    expect_axis = R @ np.array([0.0, 0.0, 1.0])
    expect_centre = {2: R @ np.array([30.0, 10.0, 2.0]) + t, 3: R @ np.array([10.0, 10.0, 2.0]) + t}
    for hole in r["holes"]:
        assert hole["depth"] == "through"
        assert angle_between_lines(hole["axis"], expect_axis) < 1e-3                       # the file is float32: 1e-3 degrees is 2e-5 rad
        centre = 0.5 * (np.array(hole["point"]) + np.array(hole["end_point"]))
        assert np.linalg.norm(centre - expect_centre[round(hole["radius_mm"])]) < 1e-3
        assert hole["radius_mm"] == pytest.approx(round(hole["radius_mm"]), abs=TOL_MM)
    # the six outward normals of the box are rotated
    got = sorted(tuple(np.round(f["normal"], 4)) for f in r["flat_faces"])
    want = sorted(tuple(np.round(R @ np.array(n, float), 4)) for n in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
    assert got == want


@pytest.mark.parametrize("seg", [16, 24, 128])
def test_a_coarse_and_a_fine_circle_are_both_found_with_the_radius_of_the_wall_vertices(seg, tmp_path):
    r = ff().find_features(write_stl(plate(seg), tmp_path / "plate.stl"))
    assert len(r["holes"]) == 2
    for hole in r["holes"]:
        assert hole["radius_mm"] == pytest.approx(round(hole["radius_mm"]), abs=TOL_MM)       # the wall vertices are on the radius
        assert hole["segments"] == seg
        sag = hole["radius_mm"] * (1.0 - math.cos(math.pi / seg))
        assert hole["chord_sag_mm"] == pytest.approx(sag, abs=1e-6)
        assert hole["depth"] == "through" and hole["length_mm"] == pytest.approx(4.0, abs=TOL_MM)


@pytest.mark.parametrize("radius", [1.0, 5.0, 20.0])
def test_a_radius_from_1_to_20_mm_with_the_facets_of_the_interface_rule(radius, tmp_path):
    seg = max(16, 2 * math.ceil(math.pi / math.acos(1 - 0.002 / radius) / 2))                 # CONTRACTS.md section 23: chord sag of 0.002 mm
    w = 2 * radius + 10
    m = Manifold.cube((w, w, 6)) - cylinder(radius, 8, seg, (w / 2, w / 2, -1))
    r = ff().find_features(write_stl(m, tmp_path / "p.stl"))
    assert len(r["holes"]) == 1
    h = r["holes"][0]
    assert h["radius_mm"] == pytest.approx(radius, abs=TOL_MM) and h["length_mm"] == pytest.approx(6.0, abs=TOL_MM)
    assert h["chord_sag_mm"] <= 0.002 + 1e-6


def test_a_part_far_from_the_origin_keeps_its_numbers(tmp_path):
    V, F = arrays(plate(32))
    r = ff().find_features(write_stl((V + np.array([1000.0, -2000.0, 500.0]), F), tmp_path / "far.stl"))      # float32 has 6e-5 mm there
    assert len(r["holes"]) == 2
    for hole in r["holes"]:
        assert hole["radius_mm"] == pytest.approx(round(hole["radius_mm"]), abs=2e-4)
    assert r["holes"][1]["point"][:2] == pytest.approx([1010.0, -1990.0], abs=2e-4)


# --- what is not a hole -------------------------------------------------------------------------------------------------------------


def test_a_tapered_hole_a_sphere_and_a_notch_are_not_listed_as_holes(tmp_path):
    tapered = Manifold.cube((40, 20, 10)) - Manifold.cylinder(12, 3.0, 3.0 + 12 * math.tan(math.radians(3.0)), 64).translate((20, 10, -1))
    r = ff().find_features(write_stl(tapered, tmp_path / "taper.stl"))
    assert r["holes"] == [] and any("neither a cylinder nor flat" in fact for fact in r["facts"])           # a cone: the facts count the region
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    r = ff().find_features(sphere)
    assert r["holes"] == [] and r["bosses"] == [] and r["flat_faces"] == []                       # the facets of a sphere are no flat faces
    notch = Manifold.cube((40, 20, 4)) - cylinder(2.5, 8, 32, (20, 0, -2))                       # a half round on the edge: 180 degrees
    r = ff().find_features(write_stl(notch, tmp_path / "notch.stl"))
    assert [h["closed_ring"] for h in r["holes"]] == [False] and r["holes"][0]["arc_deg"] == pytest.approx(180.0, abs=1e-2)
    assert ff().find_features(write_stl(notch, tmp_path / "notch.stl"), arc_min_deg=200.0)["holes"] == []
    assert any("arc_min_deg" in fact for fact in ff().find_features(write_stl(notch, tmp_path / "notch.stl"), arc_min_deg=200.0)["facts"])


def test_the_angle_between_wall_facets_is_limited_by_smooth_angle_deg():
    # 12 segments: exactly 30 degrees between the wall facets, the default limit (the angle epsilon lets it pass); 10 segments: 36 degrees.
    # The meshes are used in memory: a float32 file moves an angle by about 1e-6 rad, as much as the epsilon.
    as_mesh = lambda m: trimesh.Trimesh(*arrays(m), process=False)      # noqa: E731
    assert len(ff().find_features(as_mesh(plate(12)))["holes"]) == 2
    p10 = as_mesh(plate(10))
    assert ff().find_features(p10)["holes"] == []
    assert len(ff().find_features(p10, smooth_angle_deg=40.0)["holes"]) == 2


def test_a_tapered_wall_needs_both_the_fit_limit_and_the_normal_limit(tmp_path):
    thick = 10.0
    cone = Manifold.cube((40, 20, thick)) - Manifold.cylinder(thick + 2, 3.0, 3.0 + (thick + 2) * math.tan(math.radians(3.0)), 64).translate((20, 10, -1))
    path = write_stl(cone, tmp_path / "taper3.stl")
    assert ff().find_features(path)["holes"] == []                                              # 3 degrees: not a cylinder at the defaults
    assert ff().find_features(path, fit_residual_frac=0.1)["holes"] == []                         # the circle fit passes, the normals are 3 degrees off the plane
    found = ff().find_features(path, fit_residual_frac=0.1, axis_normal_deg=5.0)["holes"]
    assert len(found) == 1 and found[0]["fit"]["normal_deviation_deg"] == pytest.approx(3.0, abs=0.01)


def test_one_face_off_the_plane_across_the_axis_rejects_the_wall(tmp_path):
    """A mouth chamfer of 0.03 mm at 20 degrees joins the wall (the angle is under smooth_angle_deg) and is too small to move the mean of the
    normals, so the scatter test passes and the test of each face rejects the region. `axis_normal_deg` 25 accepts it."""
    from manifold3d import CrossSection
    R, t, h = 3.0, 4.0, 0.03
    profile = [(0.0, -1.0), (R, -1.0), (R, t - h), (R + h * math.tan(math.radians(20.0)), t), (R + 1.0, t + 1.0), (0.0, t + 1.0)]
    cut = Manifold.revolve(CrossSection([profile]), 64).translate((20, 10, 0))
    path = write_stl(Manifold.cube((40, 20, t)) - cut, tmp_path / "chamfer.stl")
    r = ff().find_features(path)
    assert r["holes"] == [] and r["method"]["counts"]["rejected_normal"] == 1
    assert len(ff().find_features(path, axis_normal_deg=25.0)["holes"]) == 1


def curved_panel(radius, width=40.0, length=40.0, facet=4.0, thick=2.0):
    """A slab whose top is an arc of `radius` (in x) with a facet of 4 mm: the facet angle is about 0.46 degrees for 500 mm."""
    from manifold3d import CrossSection
    n = int(round(width / facet))
    half = width / radius / 2.0
    ang = np.linspace(-half, half, n + 1)
    top = [(radius * math.sin(a), radius * (math.cos(a) - 1.0)) for a in ang]
    outline = (top + [(top[-1][0], top[-1][1] - thick), (top[0][0], top[0][1] - thick)])[::-1]      # counter-clockwise
    return Manifold.extrude(CrossSection([outline]), length)


def test_a_gently_curved_face_is_flat_only_when_flat_angle_deg_is_large_enough(tmp_path):
    path = write_stl(curved_panel(500.0), tmp_path / "panel.stl")                                # the top turns by 2.3 degrees from its middle to its edge
    tops = lambda r: [f for f in r["flat_faces"] if f["normal"][1] > 0.9]               # noqa: E731  (the arc is in x, the top faces +y; the flat bottom faces -y)
    default = ff().find_features(path)
    assert tops(default) == [] and any("not flat" in fact for fact in default["facts"])
    loose = ff().find_features(path, flat_angle_deg=5.0)
    assert len(tops(loose)) == 1 and tops(loose)[0]["area_mm2"] == pytest.approx(40.0 * 40.0, rel=1e-3)      # the arc is 40 mm long, the slab 40 mm deep
    assert loose["method"]["numbers"]["flat_angle_deg"] == 5.0


def test_the_order_does_not_depend_on_the_order_of_the_faces(tmp_path):
    V, F = arrays(plate(32, holes=((10, 10, 3.0), (30, 10, 2.0), (20, 5, 1.0), (20, 15, 1.0))))
    a = ff().find_features(trimesh.Trimesh(V, F, process=False))
    rng = np.random.default_rng(1)
    order = rng.permutation(len(F))
    perm = rng.permutation(len(V))
    inv = np.argsort(perm)
    b = ff().find_features(trimesh.Trimesh(V[perm], inv[F[order]], process=False))
    for key in ("holes", "flat_faces"):
        assert [(e["id"], e["radius_mm"] if key == "holes" else e["area_mm2"]) for e in a[key]] == [(e["id"], e["radius_mm"] if key == "holes" else e["area_mm2"]) for e in b[key]]
    assert [h["radius_mm"] for h in a["holes"]] == [1.0, 1.0, 2.0, 3.0]
    assert a["holes"][0]["point"][1] == 5.0 and a["holes"][1]["point"][1] == 15.0                  # the same radius: the centre decides


# --- meshes that are not clean --------------------------------------------------------------------------------------------------------


def test_a_hole_near_a_limit_of_the_method_is_named_in_the_facts():
    V, F = arrays(plate(64, holes=((20, 10, 3.0),)))
    centre = np.array([20.0, 10.0])
    rel = V[:, :2] - centre
    wall = np.abs(np.linalg.norm(rel, axis=1) - 3.0) < 1e-6
    noise = np.random.default_rng(1).normal(0.0, 0.01, int(wall.sum()))                          # radial noise of 1 percent of the radius
    V = V.copy()
    V[wall, :2] = centre + rel[wall] * (1.0 + noise)[:, None]
    r = ff().find_features(trimesh.Trimesh(V, F, process=False))
    assert len(r["holes"]) == 1 and 0.005 < r["holes"][0]["fit"]["rms_mm"] / r["holes"][0]["radius_mm"] < 0.01
    assert any(fact.startswith("H1 is near a limit of the method") and "fit RMS" in fact for fact in r["facts"])
    clean = ff().find_features(trimesh.Trimesh(*arrays(plate(64)), process=False))
    assert not any("near a limit" in fact for fact in clean["facts"])


def test_a_mesh_that_is_not_watertight_gives_facts_and_no_exception(tmp_path):
    V, F = arrays(plate(32))
    top = [i for i, f in enumerate(F) if np.allclose(V[f][:, 2], 4.0) and np.allclose(np.cross(V[f][1] - V[f][0], V[f][2] - V[f][0])[:2], 0.0)]
    F = np.delete(F, top[0], axis=0)                                                              # one triangle of the top face is missing
    r = ff().find_features(write_stl((V, F), tmp_path / "open.stl"))
    assert r["mesh"]["watertight"] is False and r["mesh"]["volume_mm3"] is None
    assert any("not watertight" in fact for fact in r["facts"])
    assert len(r["holes"]) == 2                                                                   # the walls are closed rings: still found
    assert len(r["flat_faces"]) >= 5


def test_a_mesh_with_inward_normals_is_read_with_outward_normals_and_says_so(tmp_path):
    V, F = arrays(plate(32))
    good = ff().find_features(write_stl((V, F), tmp_path / "good.stl"))
    flipped = ff().find_features(write_stl((V, F[:, ::-1]), tmp_path / "flipped.stl"))
    assert flipped["mesh"]["normals_flipped"] is True and good["mesh"]["normals_flipped"] is False
    assert any("inward normals" in fact for fact in flipped["facts"])
    assert [h["depth"] for h in flipped["holes"]] == ["through", "through"] and flipped["bosses"] == []
    assert sorted(f["normal"] for f in flipped["flat_faces"]) == sorted(f["normal"] for f in good["flat_faces"])


def test_a_mesh_object_with_a_vertex_for_each_corner_is_joined_before_the_search():
    V, F = arrays(plate(32))
    soup = trimesh.Trimesh(vertices=V[F].reshape(-1, 3), faces=np.arange(3 * len(F)).reshape(-1, 3), process=False)      # no vertex is shared
    r = ff().find_features(soup)
    assert r["mesh"]["vertices"] == len(V) and r["mesh"]["watertight"] is True
    assert [h["radius_mm"] for h in r["holes"]] == pytest.approx([2.0, 3.0], abs=TOL_MM) and len(r["flat_faces"]) == 6


def test_a_file_that_is_missing_or_not_an_stl_gives_a_clear_error(tmp_path):
    from topopt.config import UserError
    with pytest.raises(UserError, match="file not found"):
        ff().find_features(tmp_path / "nothing.stl")
    text = tmp_path / "notes.stl"
    text.write_text("this is not a mesh\n")
    with pytest.raises(UserError, match="not an STL file"):
        ff().find_features(text)
    (tmp_path / "empty.stl").write_bytes(b"")
    with pytest.raises(UserError, match="empty"):
        ff().find_features(tmp_path / "empty.stl")
    with pytest.raises(UserError, match="is a folder"):
        ff().find_features(tmp_path)


# --- the numbers ----------------------------------------------------------------------------------------------------------------------


def test_each_number_is_a_constant_a_keyword_default_and_a_key_of_the_reply(tmp_path):
    f = ff()
    sig = inspect.signature(f.find_features)
    assert set(f.OPTION_NAMES) == {n for n in sig.parameters if n != "stl"}
    for name in f.OPTION_NAMES:
        assert sig.parameters[name].default == getattr(f, name.upper()), name
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    r = f.find_features(write_stl(plate(32), tmp_path / "plate.stl"))
    assert set(r["method"]["numbers"]) >= set(f.OPTION_NAMES)
    assert all(r["method"]["numbers"][n] == getattr(f, n.upper()) for n in r["method"]["numbers"])


def test_a_changed_number_changes_the_result_and_the_reported_value(tmp_path):
    path = write_stl(plate(16), tmp_path / "plate16.stl")                                           # 22.5 degrees between the wall facets
    assert len(ff().find_features(path)["holes"]) == 2
    r = ff().find_features(path, smooth_angle_deg=15.0)
    assert r["holes"] == [] and r["method"]["numbers"]["smooth_angle_deg"] == 15.0
    r = ff().find_features(write_stl(plate(32), tmp_path / "p32.stl"), flat_min_area_mm2=200.0)
    assert [f["area_mm2"] for f in r["flat_faces"]] == pytest.approx([r["flat_faces"][0]["area_mm2"]] * 2)           # only the two big faces
    assert r["method"]["counts"]["flat_regions_not_listed"] == 4 and any("flat_min_area_mm2" in fact for fact in r["facts"])


@pytest.mark.parametrize("name,value", [("smooth_angle_deg", -1.0), ("smooth_angle_deg", 90.0), ("flat_angle_deg", 0.0), ("fit_residual_frac", float("nan")),
                                        ("axis_normal_deg", "2"), ("arc_min_deg", 400.0), ("flat_min_area_mm2", -1.0), ("flat_soft_share_max", True)])
def test_a_number_that_is_not_valid_is_an_error_with_its_name(name, value, tmp_path):
    from topopt.config import UserError
    with pytest.raises(UserError, match=name):
        ff().find_features(write_stl(plate(16), tmp_path / "p.stl"), **{name: value})


# --- the MCP tool ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    return tmp_path


def tree(folder):
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(folder.rglob("*"))}


def tools():
    from topopt_mcp import server
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_the_tool_reads_a_relative_path_of_the_work_folder_and_changes_nothing(ws):
    from topopt_mcp import server
    (ws / "parts").mkdir()
    write_stl(plate(32), ws / "parts" / "plate.stl")
    before = tree(ws)
    r = server.find_features("parts/plate.stl")
    assert r["ok"] is True and set(r) >= {"mesh", "holes", "bosses", "flat_faces", "method", "facts", "timing_s"}
    assert len(r["holes"]) == 2 and len(r["flat_faces"]) == 6
    assert r["mesh"]["source_rel"] == "parts/plate.stl" and r["mesh"]["source"] == str(ws / "parts" / "plate.stl")
    json.dumps(r, allow_nan=False)
    assert tree(ws) == before                                                                          # read only: the same files, bytes and times
    assert "work_folder_note" not in r
    absolute = server.find_features(str(ws / "parts" / "plate.stl"))
    assert [h["id"] for h in absolute["holes"]] == [h["id"] for h in r["holes"]]


def test_the_tool_answers_a_wrong_input_in_the_form_of_the_other_tools(ws):
    from topopt_mcp import server
    missing = server.find_features("nothing.stl")
    assert missing["ok"] is False and "file not found" in missing["error"] and "Give an absolute path" in missing["error"]
    (ws / "notes.stl").write_text("not a mesh")
    bad = server.find_features("notes.stl")
    assert bad["ok"] is False and "not an STL file" in bad["error"] and set(bad) <= {"ok", "error", "work_folder_note"}
    folder = server.find_features(".")
    assert folder["ok"] is False and "is a folder" in folder["error"]
    assert not (ws / "runs").exists()                                                                  # a read makes no folder


def test_the_tool_is_registered_with_one_argument_and_a_text_within_the_bound():
    t = tools()["find_features"]
    assert list(t.input_schema["properties"]) == ["stl_path"] and t.input_schema["required"] == ["stl_path"]
    text = t.description
    assert len(text) <= 4096 and len(text) < 2048                                                     # under the cut of the old clients: no key-text entry is needed
    assert not HABIT.search(text), HABIT.search(text).group()
    for phrase in ("Read only", "in mm", "holes:", "bosses:", "flat_faces:", "method:", "`bore`", "no repair"):
        assert phrase in text, phrase


def test_the_mock_of_the_tools_equals_the_server():
    mock = json.loads((REPO / "evals" / "mocks" / "topopt" / "_tools.json").read_text())
    got = {t["name"]: t for t in mock["tools"]}
    for name, tool in tools().items():
        entry = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert name in got, name
        assert inspect.cleandoc(got[name]["description"]) == inspect.cleandoc(entry["description"]), name      # no indent: Python 3.13 strips it from a docstring, 3.12 does not
        assert got[name]["inputSchema"] == entry["inputSchema"], name
    assert set(got) == set(tools())
    assert "find_features" in got


def test_the_tool_is_in_the_list_of_the_tools_that_only_report():
    from topopt import levers_cards
    assert "find_features" in levers_cards.READ_ONLY_TOOLS
