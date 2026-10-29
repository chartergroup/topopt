"""Flat faces of a region that are not on a voxel face (P1, F3-3; `voxelize.plane_facts`, CONTRACTS.md section 3).

A voxel belongs to a region when its centre is inside the STL. So a plane that is not on a voxel face moves: the voxelized
region ends at the nearest voxel face, and it is larger or smaller than the STL by the distance to that face. The tools state
it as a fact with the options. There is no snap and no threshold: on a face means within floating-point tolerance.
"""
import numpy as np
import pytest
import trimesh
import yaml

from test_optimize import box
from topopt.cli import check_data
from topopt.config import load_config
from topopt.problem import build_problem
from topopt.regions_builder import build_spec


def write_case(tmp_path, tip=((18.8, 4, 0), (20, 6, 4)), h=2.0, keep_out=None, solid=None, design=((0, 0, 0), (20, 10, 10)),
               keep_boxes=()):
    """Design x 0..20, y 0..10, z 0..10; clamp plate x -4..0; a load box `tip`. At h = 2 the grid origin is (-4, 0, 0), so every
    plane is on a voxel face except those of `tip` that are not a multiple of 2 mm from the origin. `design` and `keep_boxes`
    (a list of (lo, hi) boxes, region names `ko0`, `ko1` ...) change the other regions. Return the YAML path."""
    box(tmp_path / "design.stl", *design)
    box(tmp_path / "plate.stl", (-4, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", *tip)
    d = {"name": "c", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 2}}
    if keep_out is not None:
        keep_out.export(tmp_path / "bore.stl")
        d["regions"]["keep_out"] = ["bore.stl"]
    for i, kb in enumerate(keep_boxes):
        box(tmp_path / f"ko{i}.stl", *kb)
        d["regions"].setdefault("keep_out", []).append(f"ko{i}.stl")
    if solid is not None:                                                  # a solid region: (lo, hi) of a box
        box(tmp_path / "block.stl", *solid)
        d["regions"]["solid"] = ["block.stl"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_a_region_off_the_grid_by_0_4_voxel_is_a_fact_and_a_region_on_the_grid_is_not(tmp_path):
    """The tip starts at x = 18.8: 11.4 voxels from the origin (-4), so 0.4 voxel from the face at 11. The design and the plate are on the grid."""
    facts = build_problem(load_config(write_case(tmp_path))).region_facts["planes"]
    assert [(r["role"], r["name"]) for r in facts] == [("load", "tip")]
    [q] = facts[0]["planes"]                                               # the other five planes of the tip are on the grid
    assert (q["axis"], q["side"], q["at_mm"], q["effect"]) == ("x", "min", 18.8, "larger")
    assert q["offset_voxels"] == pytest.approx(0.4, abs=1e-4) and q["offset_mm"] == pytest.approx(0.8, abs=1e-3)
    assert q["from_origin_voxels"] == pytest.approx(11.4, abs=1e-4)


def test_check_data_gives_the_fact_and_the_options(tmp_path):
    d = check_data(write_case(tmp_path))
    assert d["region_planes"][0]["name"] == "tip"
    [w] = [w for w in d["warnings"] if "off the voxel grid" in w]
    assert w.startswith("region planes off the voxel grid: 1 of 1 change the part (voxel 2 mm; grid origin (-4, 0, 0) mm, "
                        "the min corner of all regions, so it moves when such a region moves).")
    assert ": load 'tip' x min 18.8 mm: 0.80 mm larger." in w
    assert ("Options: move or resize the region so that the plane is on a voxel face. Or choose a voxel size that divides "
            "the distance from the grid origin. Or extend a keep-out region past the part.") in w
    assert w.endswith("`check.region_planes`") and max(len(s.split()) for s in w.split(". ")) < 60      # short sentences


def test_all_planes_are_one_line_and_each_has_the_part_effect_in_the_facts(tmp_path):
    """Two regions with planes off the grid give ONE line with one item each. Every plane is in `region_planes` with `changes_part`."""
    d = check_data(write_case(tmp_path, solid=((8.6, 2, 2), (12, 6, 6))))
    lines = [w for w in d["warnings"] if "off the voxel grid" in w]
    assert len(lines) == 1 and "2 of 2 change the part" in lines[0]
    assert "solid 'block' x min 8.6 mm: 0.60 mm larger" in lines[0] and "load 'tip' x min 18.8 mm: 0.80 mm larger" in lines[0]
    assert {(r["role"], r["name"]) for r in d["region_planes"]} == {("solid", "block"), ("load", "tip")}
    assert all(q["changes_part"] is True and q["changed_voxels"] > 0 for r in d["region_planes"] for q in r["planes"])


def test_a_plane_under_a_region_of_higher_priority_or_outside_the_part_does_not_change_the_part(tmp_path):
    """The design starts at x = -1.3, inside the clamp plate (fixed wins), and a keep-out box is outside the part. Both are off the grid.
    The facts keep these planes with `changes_part` False and no voxel; the line leaves them out. A keep-out box inside the design changes the part."""
    p = write_case(tmp_path, tip=((18, 4, 0), (20, 6, 4)), design=((-1.3, 0, 0), (20, 10, 10)),
                   keep_boxes=[((22.3, 2, 2), (25.3, 6, 6))])
    d = check_data(p)
    by = {(r["name"], q["axis"], q["side"]): q for r in d["region_planes"] for q in r["planes"]}
    assert by[("design", "x", "min")]["changes_part"] is False and by[("design", "x", "min")]["changed_voxels"] == 0   # under the plate
    assert by[("ko0", "x", "min")]["changes_part"] is False and by[("ko0", "x", "max")]["changes_part"] is False       # beyond the design
    assert not any("off the voxel grid" in w for w in d["warnings"])                                                   # nothing is left for the line
    (tmp_path / "in").mkdir()
    inside = check_data(write_case(tmp_path / "in", tip=((18, 4, 0), (20, 6, 4)), keep_boxes=[((7.3, 2, 2), (10, 6, 6))]))
    [q] = [q for r in inside["region_planes"] for q in r["planes"]]
    assert (q["axis"], q["side"], q["changes_part"]) == ("x", "min", True) and q["changed_voxels"] > 0
    [w] = [w for w in inside["warnings"] if "off the voxel grid" in w]
    assert "keep-out 'ko0' x min 7.3 mm" in w


def test_a_region_on_the_grid_gives_no_fact(tmp_path):
    d = check_data(write_case(tmp_path, tip=((18, 4, 0), (20, 6, 4))))
    assert d["region_planes"] == [] and not any("off the voxel grid" in w for w in d["warnings"])


@pytest.mark.parametrize("tip", [
    ((14.4, 3.3, 1.3), (21.2, 6.9, 5.7)),       # six planes off the grid; some make the region larger, some smaller
    ((13.1, 4.0, 0.0), (16.7, 8.0, 3.2)),
    ((15.0, 3.0, 1.0), (19.0, 7.0, 5.0)),       # all six planes are half a voxel off: the voxel centre is on the plane
])
def test_the_effect_is_what_the_voxelization_does(tmp_path, tip):
    """On each axis the predicted growth (the signed offsets of the two planes) equals the voxelized length minus the STL length."""
    prob = build_problem(load_config(write_case(tmp_path, tip=tip)))
    in_tip = prob.load_region_ids == 0
    facts = {(q["axis"], q["side"]): q for r in prob.region_facts["planes"] for q in r["planes"]}
    assert facts                                                            # the test is empty without a plane
    for a, ax in enumerate("xyz"):
        others = tuple(i for i in range(3) if i != a)
        idx = np.nonzero(in_tip.any(axis=others))[0]
        measured = (idx.max() - idx.min() + 1) * prob.h - (tip[1][a] - tip[0][a])
        predicted = sum((1 if q["effect"] == "larger" else -1) * q["offset_mm"]
                        for (qax, _side), q in facts.items() if qax == ax)
        assert predicted == pytest.approx(measured, abs=1e-3), ax


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("side", ["min", "max"])
@pytest.mark.parametrize("delta", [-0.0003, 0.0002, 0.0006, 0.002])
def test_each_side_follows_the_ray_jitter_at_a_half_voxel_tie(tmp_path, axis, side, delta):
    """The ray of a column in x and y is shifted by 0.000731 (x) or 0.000419 (y) of a voxel, so a plane that is `delta` voxels above a
    half-voxel tie can still belong to the region of the face below it. `delta` 0.0002 is inside the band of x and y, 0.0006 is inside
    the band of x only. One plane is off the grid in each case. The face that the fact names is the face that the voxelization makes.
    (The sum of both sides on one axis cannot see a wrong side: two wrong sides sum to the same number.)"""
    lo, hi = [[12.0, 2.0, 2.0], [18.0, 6.0, 6.0]]            # at h = 2 and origin (-4, 0, 0) every plane is on a voxel face
    origin = (-4.0, 0.0, 0.0)[axis]
    tie = {("min", 0): 8.5, ("max", 0): 11.5, ("min", 1): 1.5, ("max", 1): 2.5}[(side, axis)]   # voxels from the origin
    (lo if side == "min" else hi)[axis] = origin + (tie + delta) * 2.0
    prob = build_problem(load_config(write_case(tmp_path, tip=(tuple(lo), tuple(hi)))))
    [q] = [q for r in prob.region_facts["planes"] for q in r["planes"]]
    assert (q["axis"], q["side"]) == ("xyz"[axis], side)
    idx = np.nonzero((prob.load_region_ids == 0).any(axis=tuple(i for i in range(3) if i != axis)))[0]
    made = idx.min() if side == "min" else idx.max() + 1          # the voxel face where the voxelized region ends
    above = (side == "min") == (q["effect"] == "smaller")          # the face is above the plane: min plane smaller, max plane larger
    said = q["from_origin_voxels"] + (q["offset_voxels"] if above else -q["offset_voxels"])
    assert said == pytest.approx(made, abs=2e-3)


def test_a_plane_on_the_grid_up_to_float32_rounding_is_on_the_grid(tmp_path):
    """An STL stores float32, and float32(0.7) is not 0.7. Planes at multiples of 0.1 mm are on a grid of 0.1 mm."""
    box(tmp_path / "design.stl", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    box(tmp_path / "plate.stl", (-0.2, 0.0, 0.0), (0.0, 1.0, 1.0))
    box(tmp_path / "tip.stl", (0.7, 0.4, 0.3), (1.0, 0.6, 0.9))
    d = {"name": "f", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
         "settings": {"voxel_size": 0.1, "volume_fraction": 0.3, "iterations": 2}}
    (tmp_path / "f.yaml").write_text(yaml.safe_dump(d))
    assert float(np.float32(0.7)) != 0.7
    assert build_problem(load_config(tmp_path / "f.yaml")).region_facts["planes"] == []


def test_a_plane_of_a_solid_region_is_named_with_the_role_solid(tmp_path):
    """The facts pair each mesh with its role and name. A solid block (x 8.6 to 12, the others on the grid) gives one plane, role `solid`, name = file stem."""
    prob = build_problem(load_config(write_case(tmp_path, tip=((18, 4, 0), (20, 6, 4)), solid=((8.6, 2, 2), (12, 6, 6)))))
    [r] = prob.region_facts["planes"]
    assert (r["role"], r["name"]) == ("solid", "block")
    [q] = r["planes"]
    assert (q["axis"], q["side"], q["at_mm"]) == ("x", "min", 8.6) and q["offset_voxels"] == pytest.approx(0.3, abs=1e-3)


def test_a_curved_side_has_no_plane_to_shift(tmp_path):
    """A cylinder along x: its two caps are flat faces, its side is not. The side planes (y, z) are off the grid and give no fact."""
    cyl = trimesh.creation.cylinder(radius=1.7, height=6.0, sections=40)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))     # axis x
    cyl.apply_translation((9.3, 5.0, 5.0))                                                 # caps at x 6.3 and 12.3; sides at 5 +- 1.7
    facts = build_problem(load_config(write_case(tmp_path, tip=((18, 4, 0), (20, 6, 4)), keep_out=cyl))).region_facts["planes"]
    [bore] = [r for r in facts if r["name"] == "bore"]
    assert {(q["axis"], q["side"]) for q in bore["planes"]} == {("x", "min"), ("x", "max")}


SPEC = {
    "name": "wall_part",
    "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}]},
        "load": {"tip": [{"box": {"min": [18.8, 4, 0], "max": [20, 6, 4]}}]},
    },
    "load_cases": [{"name": "down", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
    "settings": {"volume_fraction": 0.3, "voxel_size": 2.0, "iterations": 2},
}


def build_lines(spec, out):
    lines = []
    r = build_spec(spec, out, say=lambda *a: lines.append(" ".join(map(str, a))))
    return r, lines


def test_the_builder_reports_the_region_and_check_gives_the_same_line(tmp_path):
    r, lines = build_lines(SPEC, tmp_path / "o")
    assert r["ok"]
    planes = [ln for ln in lines if "off the voxel grid" in ln]
    assert len(planes) == 1 and planes[0].startswith("note: region planes off the voxel grid: 1 of 1 change the part")   # a note, not a WARNING
    assert planes[0][len("note: "):] in check_data(r["yaml"])["warnings"]                    # one text in both tools


def test_the_builder_is_silent_for_a_region_on_the_grid(tmp_path):
    on_grid = dict(SPEC, regions=dict(SPEC["regions"], load={"tip": [{"box": {"min": [18, 4, 0], "max": [20, 6, 4]}}]}))
    _r, lines = build_lines(on_grid, tmp_path / "o")
    assert not any("off the voxel grid" in ln for ln in lines)


def test_the_mcp_texts_name_the_key_of_the_facts():
    """P1 review (L3): the `check_problem` docstring lists its keys, so it names `region_planes`; `build_regions` points to it."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    from topopt_mcp import replies      # P9-11: the line that explains the keys of the reply moved to the guide topopt://replies/check_problem
    guide = replies.guide("check_problem")
    assert "topopt://replies/check_problem" in server.check_problem.__doc__
    assert "region_planes" in guide and "check.region_planes" in server.build_regions.__doc__
    assert "changes_part" in guide
