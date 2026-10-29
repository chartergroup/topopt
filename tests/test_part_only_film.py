"""Tests of `settings.part_only_min_film_mm`: the thin film at the contact faces of `result/part_only.stl` (P4b-6, CONTRACTS.md section 9.1).

The cases:
- built meshes (fast): a block with pockets behind the contact face of a removed body B. The wall behind the face is 0.1 mm over the first
  pocket and 0.4 mm over the second. The grid of the face of B lies on the corners of the pockets, so the numbers below are exact.
- the bar of `tests/test_part_only.py` (voxel 1 mm, 8 passes): one run without the key, a resume that adds the key, and post-only copies
  with other values (the post step reads the value from `run.json`, as it does after a resume).
- the hook (`examples/hook/hook_short.yaml` with `not_printed`, voxel 2 mm, 40 passes): one run with the key.

Numbers that the tests use (measured on Linux x86_64, 2026-10-23). Bounds in the tests are test bounds, not limits of the tool.
- bar, setting off: contact area plate 122.3175 mm2 (294 triangles), tip 18.8623 (62), boss 69.4615 (263); thinnest wall 0.00047 mm at the plate.
- bar, 0.05 mm: thin area of the plate 22.28 mm2, film removed 0.5856 mm3, thinnest wall that stays 0.0561 mm.
- hook, 0.25 mm: thin area of the tip 182 of 485.5 mm2, film removed 45.75 mm3, thinnest wall that stays 0.2502 mm.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import test_part_only as T  # noqa: E402
from topopt import checkpoint, part_only, post, smooth_sdf  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.config import UserError, load_config  # noqa: E402
from topopt.optimize import problem_facts  # noqa: E402
from topopt.snapshot import RunWriter  # noqa: E402
from topopt.tet.surface import surface_report  # noqa: E402

KEY = "part_only_min_film_mm"
PLATE = "fixed:plate.stl"
REAL_SNAP = getattr(smooth_sdf, "snap_into_regions", None)   # None: a checkout from before F4-27b


def no_snap(verts, faces, surfaces, band_mm):
    """`snap_into_regions` that moves nothing: the smooth surface as the marching cubes and Taubin leave it, before F4-27b."""
    return np.asarray(verts, np.float64), {"moved": 0}


@pytest.fixture(scope="module", autouse=True)
def the_smooth_surface_as_it_was_before_the_snap():
    """The film at the contact faces of a removed body comes from the smooth surface of the `sdf` mesher, which lies a little off the exact
    region face. Since F4-27b the mesher moves the vertices just outside a region face into the region (CONTRACTS.md section 19), and that
    takes most of the film away on the bar (the contact area with a wall under the voxel size: 27.9 mm2 against 100 mm2 and more; see the
    last test). The tests below are about the film rule of `part_only` and need a film, so they run the mesher without the move.

    The same holds for the cut at the planes that the design space shares with a region (F4-50, section 19). The sides of the plate in
    this case are flush with the sides of the design space, and the thin raised patches over them are most of the film (the thin area of
    the plate at 0.05 mm: 22 mm2 without the cut, 0.01 mm2 with it). So the post step runs without the design space file, as before F4-50
    (`design_path` gives None, and the mesher finds no plane). `raising=False`: a checkout from before F4-50 has no `design_path`."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(smooth_sdf, "snap_into_regions", no_snap, raising=False)   # raising=False: a checkout from before F4-27b has nothing to replace
        mp.setattr(post, "design_path", lambda run: None, raising=False)
        yield


def with_key(path: Path, value, name="keyed.yaml") -> Path:
    """A copy of the YAML at `path` with `settings.part_only_min_film_mm: value` (in the same folder, so the region files are found)."""
    doc = yaml.safe_load(path.read_text())
    doc["settings"][KEY] = value
    out = path.parent / name
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


def file_report(path) -> dict:
    """The closed-manifold audit of the mesher (`topopt.tet.surface`) on the file as written."""
    mesh = trimesh.load(path, file_type="stl")
    return surface_report(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64))


def metrics_of(out: Path) -> dict:
    return json.loads((out / "result" / "metrics.json").read_text())["part_only"]


def volume_of(out: Path) -> float:
    return float(trimesh.load(out / "result" / "part_only.stl", file_type="stl").volume)


def post_copy(src: Path, dest: Path, value) -> dict:
    """Copy a finished run to `dest`, set the film value in `run.json` (None: remove it) and repeat the post step. Return the block."""
    shutil.copytree(src, dest)
    run = json.loads((dest / "run.json").read_text())
    run["part_only"].pop("min_film_mm", None)
    if value is not None:
        run["part_only"]["min_film_mm"] = value
    (dest / "run.json").write_text(json.dumps(run))
    return post.finish(dest)["part_only"]


# -- independent measures on the written file --------------------------------------------------------------------------
def contact_rows(mesh, lo, hi, tol=1e-5):
    """The triangles of the file that lie on the surface of the box `lo`..`hi` and look into it: the contact faces of that body."""
    tv = mesh.vertices[mesh.faces]
    sdf = np.stack([T.box_sdf(tv[:, i, :], lo, hi) for i in range(3)], axis=1)
    on = (np.abs(sdf) < tol).all(axis=1)
    into = T.box_sdf(tv.mean(axis=1) + 1e-3 * mesh.face_normals, lo, hi) < 0
    return np.flatnonzero(on & into)


def pointwise_wall(mesh, rows, k=8):
    """`(wall, area)` of k sample points on each triangle: the distance along the inward normal to the first hit of another triangle."""
    rng = np.random.default_rng(1)
    tv = mesh.vertices[mesh.faces[rows]]
    w = rng.dirichlet([1, 1, 1], size=(len(rows), k))
    pts = np.einsum("rkj,rjc->rkc", w, tv).reshape(-1, 3)
    start = np.repeat(rows, k)
    hit_tri, hit_ray, where = mesh.ray.intersects_id(pts, -mesh.face_normals[start], multiple_hits=True, return_locations=True)
    keep = hit_tri != start[hit_ray]
    hit_ray, where = hit_ray[keep], where[keep]
    wall = np.full(len(pts), np.inf)
    np.minimum.at(wall, hit_ray, np.linalg.norm(where - pts[hit_ray], axis=1))
    return wall, np.repeat(mesh.area_faces[rows] / k, k)


def thin_area_of_plate(out: Path, t: float) -> float:
    mesh = trimesh.load(out / "result" / "part_only.stl", file_type="stl")
    wall, area = pointwise_wall(mesh, contact_rows(mesh, *T.BOXES["plate"]))
    return float(area[wall < t].sum())


# -- built meshes -------------------------------------------------------------------------------------------------------
def film_case(folder: Path, pockets):
    """`(verts, faces, bodies, material)`: a block x 0..14, y and z 0.5..9.5, with a box pocket for each `(lo, hi)` in `pockets`, and the removed
    body B: a box x -2..0.37, y and z -1..11, cut into 1.5 mm triangles (the corners lie at -1 + 1.5 k, so at the corners of the pockets).
    The contact face is the plane x = 0.37; a pocket that starts at x = 0.47 leaves a film of 0.1 mm between the face and the pocket."""
    block = T.cube((0, 0.5, 0.5), (14, 9.5, 9.5))
    for lo, hi in pockets:
        block = block - T.cube(lo, hi)
    v, f = smooth_sdf._from_manifold(block)
    b = trimesh.creation.box(extents=(2.37, 12, 12), transform=trimesh.transformations.translation_matrix((-0.815, 5, 5)))
    for _ in range(3):
        b = b.subdivide()
    folder.mkdir(parents=True, exist_ok=True)
    b.export(folder / "b.stl")
    return v, f, [("fixed:b", folder / "b.stl")], T.field()


# pocket 1: film 0.1 mm thick over y 3.5..5.0, z 3.5..8.0 (4.5 x 1.5 = 6.75 mm2); pocket 2: film 0.4 mm over y 6.5..8.0, same z.
# Both pockets are open at the back of the block (x = 14), so the air behind a film has no second wall.
OPEN_POCKETS = [((0.47, 3.5, 3.5), (15, 5.0, 8.0)), ((0.77, 6.5, 3.5), (15, 8.0, 8.0))]
FILM_1, FILM_2 = 6.75, 6.75                                   # mm2
FACE = 81.0                                                   # the contact face: 9 x 9 mm
OFF_VOLUME = 13.63 * 81.0 - 13.53 * 6.75 - 13.23 * 6.75      # the block behind the face (x 0.37..14) without the two pockets


def cut_value(case, value):
    v, f, bodies, material = case
    vv, ff, facts = part_only.cut(v, f, bodies, material, value)
    return vv, ff, facts, smooth_sdf.stl_facts(trimesh.Trimesh(vv, ff, process=False).export(file_type="stl"))


def test_the_thin_wall_is_cut_and_the_thicker_wall_and_the_rim_keep_the_exact_face(tmp_path):
    case = film_case(tmp_path, OPEN_POCKETS)
    _, _, off, got_off = cut_value(case, None)
    assert got_off["ok"] and got_off["volume_mm3"] == pytest.approx(OFF_VOLUME, abs=1e-2)
    contact = off["contact"]["fixed:b"]                                   # the facts are there with the setting off
    assert contact["area_mm2"] == pytest.approx(FACE, abs=1e-3) and contact["thinnest_wall_mm"] == pytest.approx(0.1, abs=1e-5)
    assert contact["area_below_voxel_mm2"] == pytest.approx(FILM_1 + FILM_2, abs=1e-3)      # voxel size 1.0: both films are below it
    assert "thin_area_mm2" not in contact and "film_removed_mm3" not in off and "pinches_left" not in off

    _, _, on, got = cut_value(case, 0.3)                                  # 0.1 mm is thinner, 0.4 mm is not
    c = on["contact"]["fixed:b"]
    assert on["pinches_left"] == 0                                         # the cut leaves no vertex where the surface touches itself
    assert c["thin_area_mm2"] == pytest.approx(FILM_1, abs=1e-3) and c["thin_triangles"] == c["cut_triangles"] == 6
    assert c["film_removed_mm3"] == on["film_removed_mm3"] == pytest.approx(0.1 * FILM_1, abs=1e-4)
    assert c["area_after_mm2"] == pytest.approx(FACE - FILM_1, abs=1e-3)   # the exact face stays: the rim and the 0.4 mm film
    assert c["thinnest_wall_after_mm"] == pytest.approx(0.4, abs=1e-5) and c["thin_area_after_mm2"] == 0.0
    assert got["ok"] and got["bodies"] == 1
    assert got["volume_mm3"] == pytest.approx(got_off["volume_mm3"] - 0.1 * FILM_1, abs=1e-3)

    _, _, both, got_both = cut_value(case, 1.0)                           # both films are thinner than the value
    assert both["contact"]["fixed:b"]["thin_area_mm2"] == pytest.approx(FILM_1 + FILM_2, abs=1e-3)
    assert both["contact"]["fixed:b"]["thin_area_mm2"] == both["contact"]["fixed:b"]["area_below_voxel_mm2"]   # a value equal to the voxel size
    assert both["film_removed_mm3"] == pytest.approx(0.1 * FILM_1 + 0.4 * FILM_2, abs=1e-3)
    assert got_both["ok"] and got_both["volume_mm3"] == pytest.approx(got_off["volume_mm3"] - 0.1 * FILM_1 - 0.4 * FILM_2, abs=1e-3)


def test_a_value_below_every_wall_changes_nothing_and_a_value_just_above_a_wall_cuts_it(tmp_path):
    case = film_case(tmp_path, OPEN_POCKETS)
    v0, f0, off, _ = cut_value(case, None)
    for value in (0.05, 0.099):                                           # below the thinnest wall (0.1 mm)
        v1, f1, low, _ = cut_value(case, value)
        assert np.array_equal(v0, v1) and np.array_equal(f0, f1)          # the same arrays, so the same file
        assert low["contact"]["fixed:b"]["thin_triangles"] == 0 and low["film_removed_mm3"] == 0.0
    _, _, just, _ = cut_value(case, 0.101)                                # a wall is thin if it is thinner than the value
    assert just["contact"]["fixed:b"]["thin_triangles"] == 6 and just["film_removed_mm3"] > 0


def test_the_depth_of_the_cut_stops_in_the_middle_of_the_air_gap(tmp_path):
    """A pocket of 0.2 mm behind the film (open to the side) and the solid block behind it: a value of 1.0 mm must not cut into the block."""
    case = film_case(tmp_path, [((0.47, 3.5, 3.5), (0.67, 10, 8.0))])
    _, _, small, got_small = cut_value(case, 0.3)
    _, _, large, got_large = cut_value(case, 1.0)
    assert got_large["volume_mm3"] == pytest.approx(got_small["volume_mm3"], abs=1e-6)       # the gap ends the prism, not the value
    assert large["film_removed_mm3"] == small["film_removed_mm3"] > 0
    v, f, _, _ = cut_value(case, 1.0)
    part = trimesh.Trimesh(v, f, process=False)
    wall_behind = np.array([[0.9, y, z] for y in (4.0, 5.5, 7.0) for z in (4.0, 6.0, 7.5)])   # x 0.67..1.37 is block behind the pocket
    assert part.contains(wall_behind).all() and not part.contains(wall_behind - [0.6, 0, 0]).any()


def test_off_the_centre_ray_a_prism_cuts_other_material_up_to_its_depth_and_the_volume_fact_counts_it(tmp_path):
    """Review F3 (measured behavior, the texts say it now). A film of 0.1 mm over an open pocket, and a rib 0.3 mm wide that stands 0.1 mm behind
    the film at the side of the pocket (y 3.5..3.8, x 0.57..14): off the centre ray of every film triangle. The depth of the prism comes from the
    centre ray (the gap behind the film is 13.5 mm), so the prism of value t cuts t mm into the rib, and `film_removed_mm3` counts the rib."""
    pocket = [((0.47, 3.8, 3.5), (15, 5.0, 8.0)), ((0.47, 3.5, 3.5), (0.57, 3.8, 8.0))]       # the pocket y 3.5..5.0 without the rib
    case = film_case(tmp_path, pocket)
    rib = ((0.57, 3.5, 3.5), (14, 3.8, 8.0))

    def rib_volume(v, f):
        return float((smooth_sdf._manifold_of(v, f) ^ T.cube(*rib)).volume())

    v0, f0, off, _ = cut_value(case, None)[:4]
    intact = rib_volume(v0, f0)
    assert intact == pytest.approx(0.3 * 4.5 * 13.43, abs=1e-3)
    for value, lost in ((0.3, 0.135), (1.0, 1.08)):
        v, f, on, _ = cut_value(case, value)[:4]
        assert intact - rib_volume(v, f) == pytest.approx(lost, abs=2e-3)                       # the rib loses up to t mm at the film footprint
        assert on["contact"]["fixed:b"]["thin_area_mm2"] == pytest.approx(FILM_1, abs=1e-3)       # the film itself: 6.75 mm2 x 0.1 mm
        assert on["film_removed_mm3"] == pytest.approx(0.1 * FILM_1 + lost, abs=2e-3)             # the fact counts the film and the rib
    assert 1.08 > 0.1 * FILM_1                                                                   # at 1.0 the rib loss is more than the film


def test_the_texts_say_that_the_depth_limit_holds_on_the_centre_ray_only():
    """Review F3. No text says that a prism never holds other material of the part."""
    text = {name: " ".join((ROOT / name).read_text().split()) for name in ("CONTRACTS.md", "docs/reference/problem-and-run.md", "topopt/part_only.py")}
    for name, t in text.items():
        assert "never reaches past the middle of the air gap" not in t and "holds no other material of the part" not in t, name
        assert "behind the wall, so that the prism holds" not in t and "but not past the middle of the air gap behind the wall." not in t, name
        assert "centre ray" in t, name
    for name in ("docs/reference/problem-and-run.md", "topopt/part_only.py"):
        assert "`film_removed_mm3` is all the material that the cut removes" in text[name], name
    contracts = text["CONTRACTS.md"]
    assert "Off the centre ray the prism cuts any material of the part under the triangle, up to its depth" in contracts
    assert "So `film_removed_mm3` is all the material that the cut removes: the film and, at an edge or off the centre ray, other material" in contracts
    assert "It is all the material that the cut removes: the film and, at the edge of a film or off the centre ray of a triangle, other material" in contracts


@pytest.mark.parametrize("film", [1e-5, 5e-5])
def test_a_film_thinner_than_any_offset_is_measured_because_the_start_triangle_is_excluded_by_index(tmp_path, film):
    """Review F9. The brief: the start triangle of a ray is excluded by its index, not by an offset. A film of 1e-5 mm (a test number) lies
    below every offset one would pick (the P4b-2 script used 1e-4 mm): a rule with an offset skips the back wall of the film and measures the
    wall behind the pocket (13.5 mm). The rule with the index finds the film."""
    pocket = ((0.37 + film, 3.5, 3.5), (15, 5.0, 8.0))
    case = film_case(tmp_path, [pocket])
    _, _, off, _ = cut_value(case, None)
    assert off["contact"]["fixed:b"]["thinnest_wall_mm"] == pytest.approx(film, abs=2e-6)       # float32 coordinates near 0.37: 3e-8
    assert off["contact"]["fixed:b"]["area_below_voxel_mm2"] == pytest.approx(FILM_1, abs=1e-3)
    _, _, on, got = cut_value(case, 0.3)
    c = on["contact"]["fixed:b"]
    assert c["thin_area_mm2"] == pytest.approx(FILM_1, abs=1e-3) and c["thin_triangles"] == 6
    assert c["thin_area_after_mm2"] == 0.0 and c["area_after_mm2"] == pytest.approx(FACE - FILM_1, abs=1e-3) and got["ok"]


def test_contracts_says_what_the_reviews_nits_found():
    """Review F5, F6, F7, F8: the sentences of CONTRACTS that the characterization tests above stand on."""
    text = " ".join((ROOT / "CONTRACTS.md").read_text().split())
    for sentence in ("the checkpoint keeps the inputs of the first run, and the next resume compares against them",
                     "to remove the key again needs no override, and a change from 0.05 to 0.1 shows `settings.part_only_min_film_mm: <not set> -> 0.1`",
                     "A face that two removed bodies share (two bodies with the same mesh) belongs to one of them",
                     "The other body has area 0, 0 triangles and `thinnest_wall_mm` `null`",
                     "The ray search of `trimesh` has its own tolerances", "the step adds none",
                     "`the film rule failed (<Type>: <message>). Option: remove settings.part_only_min_film_mm to get the exact cut`"):
        assert sentence in text, sentence


def test_a_value_that_removes_every_wall_is_an_error_with_the_value_and_the_options(tmp_path):
    case = film_case(tmp_path, OPEN_POCKETS)
    with pytest.raises(ValueError) as e:
        part_only.cut(*case[:3], case[3], 1000.0)
    assert "settings.part_only_min_film_mm of 1000 mm" in str(e.value) and "Option: a smaller value, or remove the key" in str(e.value)


def test_a_face_that_two_bodies_share_belongs_to_the_first_one(tmp_path):
    """Review F6. Two removed bodies with the same mesh: the union keeps one source for each face, the first body in the list."""
    v, f, bodies, material = film_case(tmp_path, OPEN_POCKETS)
    twin = tmp_path / "b2.stl"
    twin.write_bytes(bodies[0][1].read_bytes())
    for value in (None, 0.3):
        _, _, facts = part_only.cut(v, f, bodies + [("solid:b2", twin)], material, value)
        first, second = facts["contact"]["fixed:b"], facts["contact"]["solid:b2"]
        assert first["area_mm2"] == pytest.approx(FACE, abs=1e-3) and first["triangles"] == 52 and first["thinnest_wall_mm"] == pytest.approx(0.1, abs=1e-5)
        assert second["area_mm2"] == 0.0 and second["triangles"] == 0 and second["thinnest_wall_mm"] is None
        assert facts["overlap_mm3"]["fixed:b"] == facts["overlap_mm3"]["solid:b2"] > 0                  # both overlap the part: the volume facts are not shared
    assert facts["contact"]["fixed:b"]["thin_area_mm2"] == pytest.approx(FILM_1, abs=1e-3) and facts["contact"]["solid:b2"]["thin_area_mm2"] == 0.0


def test_a_body_without_a_contact_face_has_no_thinnest_wall(tmp_path):
    v, f, bodies, material = film_case(tmp_path, OPEN_POCKETS)
    far = tmp_path / "far.stl"
    trimesh.creation.box(extents=(1, 1, 1), transform=trimesh.transformations.translation_matrix((30, 5, 5))).export(far)
    for value in (None, 0.3):
        _, _, facts = part_only.cut(v, f, bodies + [("keep_out:far", far)], material, value)
        assert facts["contact"]["keep_out:far"]["thinnest_wall_mm"] is None
        assert facts["contact"]["keep_out:far"]["area_mm2"] == 0.0 and facts["contact"]["keep_out:far"]["triangles"] == 0


def reference_cut(verts, faces, bodies, material):
    """The `cut` of the base (f545ee7), copied: the arrays that the new `cut` must give with the setting off."""
    import manifold3d as m3d

    from topopt.interfaces import CLEAN_TOL_MM

    body = smooth_sdf._manifold_of(np.asarray(verts, np.float64), np.asarray(faces, np.int64))
    overlap, cutter = {}, None
    for label, path in bodies:
        man = smooth_sdf._load_manifold(path)
        overlap[label] = round(float((body ^ man).volume()), 4)
        cutter = man if cutter is None else cutter + man
    out = body - cutter
    parts = out.decompose() if not out.is_empty() else []
    keep, dropped = [], []
    for p in parts:
        v, f = smooth_sdf._from_manifold(p)
        (keep if len(parts) == 1 or material.held_by(v, f) else dropped).append(p)
    v, f = smooth_sdf._from_manifold(m3d.Manifold.compose(keep) if len(keep) > 1 else keep[0])
    v, f = smooth_sdf._orient(v, f)
    v = v.astype(np.float32).astype(np.float64)
    v, f, clean = smooth_sdf._cleanup_float32(v, f, CLEAN_TOL_MM)
    return v, f, overlap, {"bodies": len(dropped), "volume_mm3": round(float(sum(p.volume() for p in dropped)), 4)}, clean


def test_with_the_setting_off_the_arrays_are_those_of_the_base_cut(tmp_path):
    """Behavior that held before P4b-6: the original ids in the boolean do not change the output arrays. (This test passes on the base.)"""
    v, f, bodies, material = film_case(tmp_path, OPEN_POCKETS)
    new_v, new_f, facts = part_only.cut(v, f, bodies, material)
    ref_v, ref_f, overlap, dropped, clean = reference_cut(v, f, bodies, material)
    assert np.array_equal(new_v, ref_v) and np.array_equal(new_f, ref_f)
    assert facts["overlap_mm3"] == overlap and facts["leftovers_dropped"] == dropped and facts["clean"] == clean


def test_a_fault_in_the_measure_of_the_contact_faces_does_not_stop_the_file_when_the_setting_is_off(tmp_path, monkeypatch):
    """The facts are extra with the setting off: the file of the base is still written, and the block says what was not measured."""
    v, f, bodies, material = film_case(tmp_path / "case", OPEN_POCKETS)
    ref_v, ref_f, *_ = reference_cut(v, f, bodies, material)

    def boom(*args, **kwargs):
        raise RuntimeError("no rays today")

    monkeypatch.setattr(part_only, "Walls", boom)
    new_v, new_f, facts = part_only.cut(v, f, bodies, material)
    assert np.array_equal(new_v, ref_v) and np.array_equal(new_f, ref_f)
    assert facts["contact"] == {} and "RuntimeError: no rays today" in facts["contact_error"]
    with pytest.raises(RuntimeError, match="no rays today"):                  # with the value the rule needs the walls
        part_only.cut(v, f, bodies, material, 0.3)
    block, out = make_block(tmp_path, 1.0, None, "boom")
    assert block["applied"] and (out / "part_only.stl").is_file() and block["contact"] == {}
    assert len(block["notes"]) == 1 and "the contact faces were not measured (RuntimeError: no rays today)" in block["notes"][0]


def test_a_fault_with_the_setting_on_names_the_option_and_a_fault_of_the_files_does_not(tmp_path, monkeypatch):
    """Review F8. A fault in the measure or the rule with the setting on gives the option "remove the key"; a missing region file does not
    (removing the key would not help)."""
    option = "Option: remove settings.part_only_min_film_mm to get the exact cut"
    v, f, bodies, material = film_case(tmp_path / "case", OPEN_POCKETS)

    def boom(*args, **kwargs):
        raise RuntimeError("no rays today")

    monkeypatch.setattr(part_only, "Walls", boom)
    with pytest.raises(RuntimeError) as e:
        part_only.cut(v, f, bodies, material, 0.3)
    assert "the film rule failed (RuntimeError: no rays today). " + option in str(e.value)
    block, out = make_block(tmp_path, 1.0, 0.3, "boom")
    assert block["applied"] is False and block["error"].endswith(option) and "no rays today" in block["error"] and not (out / "part_only.stl").exists()
    assert part_only.log_lines(block) == [f"part_only: not made ({block['error']})"]
    monkeypatch.undo()

    def boom_rule(*args, **kwargs):
        raise ValueError("no prisms today")

    monkeypatch.setattr(part_only, "_drop_film", boom_rule)
    block, _ = make_block(tmp_path, 1.0, 0.3, "rule")
    assert block["applied"] is False and "the film rule failed (ValueError: no prisms today)" in block["error"] and block["error"].endswith(option)
    monkeypatch.undo()

    off, _ = make_block(tmp_path, 1.0, None, "off")                                        # the setting off: no film rule, no option
    assert off["applied"] is True
    monkeypatch.setattr(part_only, "bodies_of_run", lambda run: [("fixed:b.stl", tmp_path / "gone.stl")])
    block, _ = make_block(tmp_path, 1.0, 0.3, "gone")
    assert block["applied"] is False and block["error"].startswith("FileNotFoundError: region file not found") and "Option" not in block["error"]
    monkeypatch.undo()
    huge, _ = make_block(tmp_path, 1.0, 1000.0, "huge")                                    # "removes every piece" keeps its own option, once
    assert huge["error"].count("Option") == 1 and "the film rule failed" not in huge["error"]


def make_block(tmp_path, h, value, tag):
    """The block of `part_only.make` for the built case (`film_case`) with voxel size `h` and the film value (None: off)."""
    v, f, _bodies, _material = film_case(tmp_path / "case", OPEN_POCKETS)
    labels = np.zeros((15, 11, 11), np.uint8)
    labels[:14, :10, :10] = post.DESIGN
    rho = (labels > 0).astype(np.float32)
    run = {"yaml_path": str(tmp_path / "case" / "x.yaml"),
           "part_only": {"file": "part_only.stl", "removed": ["fixed:b.stl"], "entries": [{"role": "fixed", "name": "b.stl", "file": "b.stl"}]}}
    if value is not None:
        run["part_only"]["min_film_mm"] = value
    out = tmp_path / f"out_{tag}"
    out.mkdir()
    final = {"volume_mm3": float(smooth_sdf._manifold_of(v, f).volume())}
    return part_only.make(run, v, f, rho, labels, (0.0, 0.0, 0.0), h, 1.0, final, "sdf", out), out


def test_the_note_and_the_log_line_follow_the_setting(tmp_path):
    """Voxel size 1.0 (a wall of 0.1 mm is thinner: a note) and 0.05 (it is not: no note); with the value: a log line, and no note."""
    off, _ = make_block(tmp_path, 1.0, None, "a")
    assert off["applied"] and len(off["notes"]) == 1
    note = off["notes"][0]
    assert "above the measured wall" not in note                              # review F1: the thinnest wall is not a value to use
    assert "The thinnest wall is the lower end of the film, not a value to use" in note
    assert "settings.part_only_min_film_mm: <the wall thickness that you want to cut off, in mm>" in note
    assert "in a new YAML file in the folder of the old one, and resume the finished run" in note
    assert "thin_area_mm2, cut_area_mm2 and thin_area_after_mm2" in note
    assert "fixed:b.stl: thinnest wall 0.1 mm, 13.5 mm2 of contact face with a wall below the voxel size" in note and "voxel size (1 mm)" in note
    small, _ = make_block(tmp_path, 0.2, None, "a2")                          # voxel size 0.2: only the 0.1 mm film is below it
    assert "fixed:b.stl: thinnest wall 0.1 mm, 6.75 mm2 of contact face with a wall below the voxel size" in small["notes"][0]
    assert "min_film_mm" not in off and "film_removed_mm3" not in off
    assert not any("film rule" in line for line in part_only.log_lines(off))
    assert "notes" not in make_block(tmp_path, 0.05, None, "b")[0]            # the wall is not thinner than the voxel size

    on, _ = make_block(tmp_path, 1.0, 0.3, "c")
    assert on["applied"] and on["min_film_mm"] == 0.3 and on["film_removed_mm3"] == pytest.approx(0.1 * FILM_1, abs=1e-4) and "notes" not in on
    line = [x for x in part_only.log_lines(on) if "film rule" in x]
    assert len(line) == 1 and "settings.part_only_min_film_mm 0.3 mm" in line[0] and "0.6750 mm3 of material removed" in line[0]
    assert "thin 6.750 mm2 (6 triangle(s)); cut 6.750 mm2 (6 triangle(s), with the repair of pinch vertices)" in line[0]
    assert "cut off" not in line[0]                                                       # review F2: the line gave the thin area as "cut off"
    assert "the thinnest wall that stays is 0.4 mm" in line[0]


def test_a_value_that_removes_every_wall_gives_a_fact_and_no_file(tmp_path):
    """`make` does not stop the run: the block says why there is no file. final.stl is not its business."""
    block, out = make_block(tmp_path, 1.0, 1000.0, "huge")
    assert block["applied"] is False and block["file"] is None and block["min_film_mm"] == 1000.0
    assert "settings.part_only_min_film_mm of 1000 mm" in block["error"] and "Option: a smaller value" in block["error"]
    assert not (out / "part_only.stl").exists()
    assert part_only.log_lines(block) == [f"part_only: not made ({block['error']})"]


# -- loader, plan, checkpoint --------------------------------------------------------------------------------------------
def test_the_setting_is_off_by_default_and_reads_a_number(tmp_path):
    p = T.write_case(tmp_path, T.ALL_KINDS)
    assert load_config(p).settings.part_only_min_film_mm is None
    assert load_config(with_key(p, None)).settings.part_only_min_film_mm is None            # null is off
    assert load_config(with_key(p, 0.05)).settings.part_only_min_film_mm == 0.05
    assert load_config(with_key(p, 3)).settings.part_only_min_film_mm == 3.0                 # an integer, no upper limit
    assert load_config(with_key(p, "1e-2")).settings.part_only_min_film_mm == 0.01           # YAML 1.1 reads 1e-2 as text


@pytest.mark.parametrize("value", [0, 0.0, -0.1, "thin", "", float("inf"), float("-inf"), float("nan"), "inf", [0.1], True, {"a": 1}])
def test_a_bad_value_is_a_loader_error_with_the_fact_and_the_options(tmp_path, value):
    with pytest.raises(UserError) as e:
        load_config(with_key(T.write_case(tmp_path, T.ALL_KINDS), value))
    text = str(e.value)
    assert "settings.part_only_min_film_mm" in text and "must be" in text                         # the fact: key and value
    assert "Options: write a number above 0 in mm, or remove the key (or write null) to keep the film" in text
    assert "result/part_only.stl" in text


def test_the_key_without_not_printed_gives_a_fact_not_an_error(tmp_path, capsys):
    plain = T.write_case(tmp_path, None, "plain.yaml")
    keyed = with_key(plain, 0.05, "nokeyset.yaml")
    facts = [x for x in problem_facts(load_config(keyed)) if "part_only_min_film_mm" in x]
    assert len(facts) == 1 and facts[0].startswith("fact: settings.part_only_min_film_mm is 0.05 mm")
    assert "regions.not_printed is not set" in facts[0] and "no part_only.stl is made" in facts[0] and "options:" in facts[0]
    assert main(["check", str(keyed)]) == 0                                                      # `topopt check` shows it, and does not stop
    assert "fact: settings.part_only_min_film_mm is 0.05 mm" in capsys.readouterr().out
    assert not [x for x in problem_facts(load_config(plain)) if "part_only_min_film_mm" in x]
    with_list = with_key(T.write_case(tmp_path, T.ALL_KINDS, "listed.yaml"), 0.05, "listed_keyed.yaml")
    assert not [x for x in problem_facts(load_config(with_list)) if "part_only_min_film_mm" in x]


def test_the_plan_of_the_run_carries_the_value_only_when_it_is_set(tmp_path):
    p = T.write_case(tmp_path, T.ALL_KINDS)
    assert "min_film_mm" not in RunWriter._part_only_plan(load_config(p))
    assert "min_film_mm" not in RunWriter._part_only_plan(load_config(with_key(p, None)))
    assert RunWriter._part_only_plan(load_config(with_key(p, 0.05)))["min_film_mm"] == 0.05


def test_the_effective_settings_have_the_key_only_when_it_is_set_and_a_change_is_a_normal_change(tmp_path):
    p = T.write_case(tmp_path, T.ALL_KINDS)
    off, null = load_config(p), load_config(with_key(p, None, "null.yaml"))
    on, on2 = load_config(with_key(p, 0.05, "on.yaml")), load_config(with_key(p, 0.1, "on2.yaml"))
    assert KEY not in checkpoint.effective_settings(off)["settings"] and checkpoint.effective_settings(off) == checkpoint.effective_settings(null)
    assert checkpoint.effective_settings(on)["settings"][KEY] == 0.05
    assert checkpoint.effective_settings(on) != checkpoint.effective_settings(off)
    assert KEY not in "".join(checkpoint.SOFT_KEYS)                                         # like settings.mesh_method: not a soft key
    changes, notes = checkpoint.diff_inputs(checkpoint.input_hashes(off), checkpoint.input_hashes(on))
    assert changes == [f"settings.{KEY}: <not set> -> 0.05"] and notes == []
    changes, notes = checkpoint.diff_inputs(checkpoint.input_hashes(on), checkpoint.input_hashes(on2))
    assert changes == [f"settings.{KEY}: 0.05 -> 0.1"] and notes == []


# -- the bar --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bar_off(tmp_path_factory):
    """One run of the bar without the key (a run that the base can make too)."""
    folder = tmp_path_factory.mktemp("film_bar")
    off_yaml = T.write_case(folder, T.ALL_KINDS, "off.yaml")
    off = folder / "off"
    assert main(["run", str(off_yaml), "--out", str(off), "--quiet"]) == 0
    return {"folder": folder, "off": off, "off_yaml": off_yaml}


@pytest.fixture(scope="module")
def bar(bar_off):
    """A resume of a copy of that run with the key 0.05 (the path of a user whose run has finished)."""
    on = bar_off["folder"] / "on"
    shutil.copytree(bar_off["off"], on)
    keyed = with_key(bar_off["off_yaml"], 0.05)
    assert main(["run", str(keyed), "--out", str(on), "--resume", "--allow-changed-inputs", "--quiet"]) == 0
    return {**bar_off, "on": on, "keyed": keyed, "t": 0.05}


@pytest.fixture(scope="module")
def bar_at(bar):
    """`bar_at(value)`: the run folder of a post-only copy of the bar with that value (cached)."""
    cache = {}

    def at(value):
        if value not in cache:
            dest = bar["folder"] / f"at_{value}"
            cache[value] = (dest, post_copy(bar["off"], dest, value))
        return cache[value]
    return at


def test_the_setting_off_has_the_facts_and_a_note_with_the_measured_wall(bar):
    block = metrics_of(bar["off"])
    assert block["applied"] and "min_film_mm" not in block and "film_removed_mm3" not in block
    assert list(block["contact"]) == T.LABELS                                    # one record for each removed body, in the order of the list
    for label, rec in block["contact"].items():
        assert {"area_mm2", "triangles", "thinnest_wall_mm", "area_below_voxel_mm2"} <= set(rec), label
        assert not any(k.startswith(("thin_", "cut_", "film_")) for k in rec), label  # no film keys without the setting
        assert 0.0 <= rec["area_below_voxel_mm2"] <= rec["area_mm2"], label
    assert block["contact"][PLATE]["area_below_voxel_mm2"] > 100 and block["contact"]["keep_out:hole.stl"]["area_below_voxel_mm2"] == 0.0   # voxel 1 mm
    assert block["contact"][PLATE]["thinnest_wall_mm"] < 0.01 and block["contact"]["keep_out:hole.stl"]["thinnest_wall_mm"] > 1.0
    notes = " ".join(block["notes"])
    assert "above the measured wall" not in notes and "not a value to use" in notes
    assert "settings.part_only_min_film_mm: <the wall thickness that you want to cut off, in mm>" in notes
    plate = block["contact"][PLATE]
    assert f"{PLATE}: thinnest wall {plate['thinnest_wall_mm']:g} mm, {plate['area_below_voxel_mm2']:g} mm2 of contact face with a wall below" in notes
    assert "load:tip" in notes and "solid:boss.stl" in notes
    assert "keep_out:hole.stl 1.5" not in notes                                  # a wall above the voxel size is not named
    run = json.loads((bar["off"] / "run.json").read_text())["part_only"]
    assert "min_film_mm" not in run and run["contact"] == block["contact"]
    assert not any("film rule" in x for x in (bar["off"] / "run.log").read_text().splitlines())


def test_the_contact_facts_are_the_triangles_of_the_file_that_lie_on_the_removed_bodies(bar):
    """An independent measure: the triangles of the file on the surface of each box and looking into it (no ids). Setting off."""
    block = metrics_of(bar["off"])
    mesh = trimesh.load(bar["off"] / "result" / "part_only.stl", file_type="stl")
    for name in ("plate", "tip", "boss"):
        rows = contact_rows(mesh, *T.BOXES[name])
        label = [x for x in T.LABELS if name in x][0]
        assert float(mesh.area_faces[rows].sum()) == pytest.approx(block["contact"][label]["area_mm2"], rel=1e-3), name
        assert len(rows) == block["contact"][label]["triangles"], name
    assert block["contact"][PLATE]["area_mm2"] > 95.3                                # the plate face (95.26 mm2) and the film around the plate


def test_the_snap_of_the_region_faces_takes_most_of_the_film_of_the_bar_plate_away(bar_off, tmp_path, monkeypatch):
    """F4-27b. The same post step with the real `snap_into_regions` (the module runs without it): the contact area of the plate with a wall under
    the voxel size is less than half of the area without it, and the thinnest wall is 30 times thicker (measured: 27.9 mm2 against 100 mm2 and
    more; 0.017 mm against 0.0005 mm). The film rule is for the film that stays, and for films of other origins."""
    assert REAL_SNAP is not None, "this checkout has no snap_into_regions"
    off = metrics_of(bar_off["off"])["contact"][PLATE]
    monkeypatch.setattr(smooth_sdf, "snap_into_regions", REAL_SNAP)
    snapped = post_copy(bar_off["off"], tmp_path / "snapped", None)["contact"][PLATE]
    assert snapped["area_below_voxel_mm2"] < 0.5 * off["area_below_voxel_mm2"]
    assert snapped["thinnest_wall_mm"] > 10 * off["thinnest_wall_mm"]


def test_the_plain_file_of_the_real_bar_is_the_file_of_the_base_cut(bar_off, tmp_path, monkeypatch):
    """Behavior that held before P4b-6 (this test passes on the base): the post step with the base `cut` writes the same bytes."""
    def base_cut(verts, faces, bodies, material, min_film_mm=None):
        v, f, overlap, dropped, clean = reference_cut(verts, faces, bodies, material)
        return v, f, {"overlap_mm3": overlap, "leftovers_dropped": dropped, "contact": {}, "clean": clean}

    monkeypatch.setattr(part_only, "cut", base_cut)
    post_copy(bar_off["off"], tmp_path / "base", None)
    assert (tmp_path / "base" / "result" / "part_only.stl").read_bytes() == (bar_off["off"] / "result" / "part_only.stl").read_bytes()


def test_the_resume_that_adds_the_key_repeats_the_post_step_with_the_film_rule(bar):
    block = metrics_of(bar["on"])
    assert block["applied"] and block["min_film_mm"] == bar["t"] and block["film_removed_mm3"] > 0
    run = json.loads((bar["on"] / "run.json").read_text())["part_only"]
    assert run["min_film_mm"] == bar["t"] and run["contact"] == block["contact"]
    log = (bar["on"] / "run.log").read_text()
    line = [x for x in log.splitlines() if "film rule" in x]
    assert len(line) == 1 and f"part_only: film rule (settings.part_only_min_film_mm {bar['t']:g} mm): thin " in line[0] and "; cut " in line[0]
    assert "cut off" not in line[0]
    assert (bar["off"] / "result" / "final.stl").read_bytes() == (bar["on"] / "result" / "final.stl").read_bytes()    # the optimization is the same
    assert (bar["off"] / "result" / "part_only.stl").read_bytes() != (bar["on"] / "result" / "part_only.stl").read_bytes()


def test_a_second_resume_compares_with_the_inputs_of_the_first_run(bar, tmp_path, capsys):
    """Review F5. The resume that added the key ran no pass, so the checkpoint still holds the inputs of the first run: removing the key needs no
    override and gives the file of the run without the key; a change from 0.05 to 0.1 shows `<not set> -> 0.1`."""
    back = tmp_path / "back"
    shutil.copytree(bar["on"], back)                                                         # a run folder that was resumed with the key 0.05
    assert main(["run", str(bar["off_yaml"]), "--out", str(back), "--resume", "--quiet"]) == 0   # the YAML without the key, no override
    assert (back / "result" / "part_only.stl").read_bytes() == (bar["off"] / "result" / "part_only.stl").read_bytes()
    assert "min_film_mm" not in metrics_of(back)
    capsys.readouterr()
    again = tmp_path / "again"
    shutil.copytree(bar["on"], again)
    other = with_key(bar["off_yaml"], 0.1, "k01.yaml")
    assert main(["run", str(other), "--out", str(again), "--resume", "--quiet"]) != 0
    err = capsys.readouterr()
    assert f"settings.{KEY}: <not set> -> 0.1" in err.err + err.out and f"{KEY}: 0.05 -> 0.1" not in err.err + err.out


def test_the_resume_needs_the_override_and_changes_nothing_without_it(bar, tmp_path, capsys):
    copy = tmp_path / "copy"
    shutil.copytree(bar["off"], copy)
    before = {p.name: p.read_bytes() for p in sorted((copy / "result").iterdir())}
    assert main(["run", str(bar["keyed"]), "--out", str(copy), "--resume", "--quiet"]) != 0
    err = capsys.readouterr()
    text = err.err + err.out
    assert f"settings.{KEY}: <not set> -> 0.05" in text and "--allow-changed-inputs" in text and "Resume stopped" in text
    assert {p.name: p.read_bytes() for p in sorted((copy / "result").iterdir())} == before             # nothing was written


@pytest.mark.parametrize("value", [0.05, 0.2])
def test_the_thin_contact_area_is_gone_and_the_file_is_one_closed_manifold(bar, bar_at, value):
    out = bar["on"] if value == bar["t"] else bar_at(value)[0]
    block = metrics_of(out)
    for label, rec in block["contact"].items():
        assert rec["thin_area_after_mm2"] == 0.0, label                              # no contact triangle is thinner than the value
        if rec["thinnest_wall_after_mm"] is not None:
            assert rec["thinnest_wall_after_mm"] >= value, label
        assert rec["cut_triangles"] >= rec["thin_triangles"] and rec["area_after_mm2"] <= rec["area_mm2"]
    assert block["contact"][PLATE]["thin_area_mm2"] > 1.0                               # the film of the plate was there (bar: 22 mm2 at 0.05)
    assert block["contact"]["keep_out:hole.stl"]["thin_triangles"] == 0                 # a wall above the value is not touched
    rep = file_report(out / "result" / "part_only.stl")
    assert rep["closed_manifold"] and rep["bodies"] == 1 and rep["nonmanifold_vertices"] == 0 and rep["nonmanifold_edges"] == 0
    assert block["watertight"] and block["bodies"] == 1
    assert block["pinches_left"] == rep["nonmanifold_vertices"] == 0                    # review F4: the fact agrees with the audit of the mesher
    assert "notes" not in block or "settings.part_only_min_film_mm:" not in " ".join(block["notes"])  # the note is for the setting off
    assert "notes" not in block or "touches itself" not in " ".join(block["notes"])


def test_the_repair_of_pinch_vertices_is_what_keeps_the_file_a_manifold(bar, bar_at, tmp_path, monkeypatch):
    """At 0.1 mm the thin triangles alternate with thicker ones; without the repair the cut leaves vertices with two fans of faces."""
    rep = file_report(bar_at(0.1)[0] / "result" / "part_only.stl")
    assert rep["closed_manifold"] and rep["nonmanifold_vertices"] == 0
    monkeypatch.setattr(part_only, "_pinches", lambda man: np.zeros((0, 3), np.float32))
    post_copy(bar["off"], tmp_path / "norepair", 0.1)
    assert not file_report(tmp_path / "norepair" / "result" / "part_only.stl")["closed_manifold"]       # measured: 26 vertices with two fans


def test_a_pinch_that_the_repair_cannot_see_is_a_fact_and_a_note_with_the_options(bar, tmp_path, monkeypatch):
    """Review F4. If the repair loop cannot see the pinches (here: `_pinches` answers "none" inside `_drop_film` only), the cut at 0.1 mm leaves
    vertices with two fans of faces (measured: 26 on the bar plate without the repair). The block says how many, and the note gives the options."""
    import sys

    real = part_only._pinches

    def blind_in_the_loop(man):
        callers = (sys._getframe(1).f_code.co_name, sys._getframe(2).f_code.co_name)         # direct calls: a generator would shift the frames
        return np.zeros((0, 3), np.float32) if "_drop_film" in callers else real(man)

    monkeypatch.setattr(part_only, "_pinches", blind_in_the_loop)
    block = post_copy(bar["off"], tmp_path / "blind", 0.1)
    rep = file_report(tmp_path / "blind" / "result" / "part_only.stl")
    assert not rep["closed_manifold"] and rep["nonmanifold_vertices"] > 0
    assert block["pinches_left"] > 0 and block["pinches_left"] == rep["nonmanifold_vertices"]      # measured: 26 and 26
    note = [x for x in block["notes"] if "touches itself" in x]
    assert len(note) == 1 and f"leaves {block['pinches_left']} place(s)" in note[0] and "topopt.tet refuses a surface like this" in note[0]
    assert "Options: another value of settings.part_only_min_film_mm, or remove the key to get the exact cut" in note[0]
    assert any("touches itself at a vertex" in x for x in part_only.log_lines(block))


def test_the_root_face_of_the_bar_stays_the_plane_of_the_plate(bar):
    """The contact face with the plate is still mostly the plate plane (the bound of the test of P4b-2: more than 50 mm2), and it is exact."""
    part = trimesh.load(bar["on"] / "result" / "part_only.stl", file_type="stl")
    plane = float(np.float32(T.PLATE_X))
    v = np.asarray(part.vertices)
    near = np.abs(v[:, 0] - T.PLATE_X) < 1e-3
    assert np.abs(v[near, 0] - plane).max() <= np.spacing(np.float32(T.PLATE_X))             # the vertices on the plane are on it
    face = near[part.faces].all(axis=1)
    assert (part.face_normals[face][:, 0] < -0.999).all() and part.area_faces[face].sum() > 50
    assert metrics_of(bar["on"])["contact"][PLATE]["area_after_mm2"] > 50
    g = np.linspace(0.25, 9.75, 20)
    yz = np.array([(y, z) for y in g for z in g])
    assert not part.contains(np.column_stack([np.full(len(yz), T.PLATE_X - 0.01), yz])).any()   # no material in the plate


def test_the_volume_of_the_file_falls_by_the_film_and_the_pieces_that_the_film_cut_frees(bar):
    """Independent of the facts of the contact: the volumes of the two files (setting off, and 0.05) differ by the film volume plus the
    volume of the loose pieces that the second pass of the loose-piece rule drops."""
    off, on = metrics_of(bar["off"]), metrics_of(bar["on"])
    more = on["leftovers_dropped"]["volume_mm3"] - off["leftovers_dropped"]["volume_mm3"]
    assert volume_of(bar["off"]) - volume_of(bar["on"]) == pytest.approx(on["film_removed_mm3"] + more, abs=3e-3)    # measured: 0.7210 against 0.7210
    assert on["film_removed_mm3"] == pytest.approx(sum(r["film_removed_mm3"] for r in on["contact"].values()), rel=0.05)


@pytest.mark.parametrize("value", [0.05, 0.1, 0.2])
def test_a_point_by_point_measure_finds_at_most_half_of_the_thin_area_that_the_file_had(bar, bar_at, value):
    """The rule measures the wall at the centre of each contact triangle, so a wedge of film inside a larger triangle can stay. Measured on
    the plate with 8 points on each triangle: the area with a wall below the value falls to 36 % (0.05), 11 % (0.1) and 6 % (0.2)."""
    out = bar["on"] if value == bar["t"] else bar_at(value)[0]
    before = thin_area_of_plate(bar["off"], value)
    after = thin_area_of_plate(out, value)
    assert before > 10 and after < 0.5 * before


def test_the_facts_are_in_metrics_run_json_and_the_log(bar):
    out = bar["on"]
    block = metrics_of(out)
    run = json.loads((out / "run.json").read_text())["part_only"]
    run.pop("entries")
    for key in ("min_film_mm", "film_removed_mm3", "contact", "applied", "volume_after_mm3"):
        assert run[key] == block[key], key
    rec = block["contact"][PLATE]
    assert set(rec) == {"area_mm2", "triangles", "thinnest_wall_mm", "area_below_voxel_mm2", "thin_area_mm2", "thin_triangles", "cut_area_mm2", "cut_triangles",
                        "film_removed_mm3", "area_after_mm2", "thinnest_wall_after_mm", "thin_area_after_mm2"}
    assert rec["thin_triangles"] > 0 and rec["thin_area_mm2"] > 0 and rec["film_removed_mm3"] > 0
    assert 50 < rec["area_after_mm2"] < rec["area_mm2"]                              # measured again on the result: 96.9 of 122.3 mm2
    line = [x for x in (out / "run.log").read_text().splitlines() if "film rule" in x]
    assert len(line) == 1 and f"{block['film_removed_mm3']:.4f} mm3 of material removed" in line[0] and "the thinnest wall that stays is" in line[0]
    assert line[0] in part_only.log_lines(block)


def test_the_log_line_gives_the_thin_area_and_the_larger_cut_area_with_the_repair(bar_at):
    """Review F2. On the bar at 0.1 mm the repair of pinch vertices takes more than the thin triangles: thin 54.65 mm2, cut 84.08 mm2 (plate)."""
    block = bar_at(0.1)[1]
    rec = list(block["contact"].values())
    thin, cut = sum(c["thin_area_mm2"] for c in rec), sum(c["cut_area_mm2"] for c in rec)
    assert cut > 1.2 * thin > 0                                                              # measured: 54.65 and 84.08 at the plate
    line = [x for x in part_only.log_lines(block) if "film rule" in x]
    assert len(line) == 1
    assert f"thin {thin:.3f} mm2 ({sum(c['thin_triangles'] for c in rec)} triangle(s)); cut {cut:.3f} mm2 ({sum(c['cut_triangles'] for c in rec)} triangle(s)," in line[0]
    assert f"{thin:.3f}" != f"{cut:.3f}" and "cut off" not in line[0]


def test_the_mcp_result_carries_the_new_facts(bar):
    server = pytest.importorskip("topopt_mcp.server")
    res = server.run_result(str(bar["on"]), preview=False)
    block = res["metrics"]["part_only"]
    assert block["min_film_mm"] == bar["t"] and block["contact"][PLATE]["thin_area_after_mm2"] == 0.0
    off = server.run_result(str(bar["off"]), preview=False)["metrics"]["part_only"]
    assert off["contact"][PLATE]["thinnest_wall_mm"] is not None and "min_film_mm" not in off


def test_the_run_result_text_names_the_new_facts_the_setting_and_the_way_to_apply_it():
    """The description that a client reads (the limit of 4096 characters is tested in test_plugin_manifest.py)."""
    import asyncio

    server = pytest.importorskip("topopt_mcp.server")
    from topopt_mcp import replies
    text = next(t.description for t in asyncio.run(server.mcp.list_tools()) if t.name == "run_result")
    text += "\n" + replies.guide("run_result")   # the lines of the keys moved to the guide (#174)
    for word in ("settings.part_only_min_film_mm", "contact", "thinnest_wall_mm", "min_film_mm", "film_removed_mm3", "resume=true",
                 "allow_changed_inputs=true", "copy the run folder", "area_below_voxel_mm2", "thin_area_mm2", "cut_area_mm2", "thin_area_after_mm2",
                 "not a value to use", "NEW file in the folder of the old YAML", "change the value if the film is still there"):
        assert word in text, word
    assert "above the measured wall" not in text


def test_the_texts_do_not_lead_to_a_value_just_above_the_thinnest_wall():
    """Review F1. The thinnest wall is the lower end of the film. The note, the skill, the README, CONTRACTS and the docstrings say so, give the
    way to find a value (write one, resume, read the three areas), and say that the YAML is a new file in the folder of the old one."""
    for name in ("skills/start-here/SKILL.md", ".agents/skills/start-here/SKILL.md", "docs/reference/problem-and-run.md", "CONTRACTS.md", "topopt/part_only.py", "topopt_mcp/server.py"):
        text = (ROOT / name).read_text()
        assert "above the measured wall" not in text and "above the thinnest wall" not in text, name
    skill = (ROOT / "skills" / "start-here" / "SKILL.md").read_text()
    bullet = next(x for x in skill.splitlines() if x.startswith("- To cut the thin film at the contact faces"))
    for word in ("The thinnest wall is the lower end of the film, not a value to use", "area_below_voxel_mm2", "NEW YAML file in the folder of the old YAML",
                 "resume=true", "allow_changed_inputs=true", "thin_area_mm2", "cut_area_mm2", "thin_area_after_mm2"):
        assert word in bullet, word
    readme = (ROOT / "docs/reference/problem-and-run.md").read_text()
    for word in ("The thinnest wall is the lower end of the film, not a value to use", "area_below_voxel_mm2", "as a NEW file in the folder of the old YAML"):
        assert word in readme, word


# -- the hook ---------------------------------------------------------------------------------------------------------------
HOOK = ROOT / "examples" / "hook"


@pytest.fixture(scope="module")
def hook(tmp_path_factory):
    """The hook of `examples/hook/hook_short.yaml` (voxel 2 mm, 40 passes) with the tip, the plate and one screw hole not printed, and the key 0.25."""
    folder = tmp_path_factory.mktemp("film_hook")
    for name in ("design.stl", "hook_tip.stl", "screw_hole_bottom.stl", "screw_hole_top.stl", "wall_plate.stl"):
        shutil.copy(HOOK / name, folder / name)
    doc = yaml.safe_load((HOOK / "hook_short.yaml").read_text())
    doc["regions"]["not_printed"] = ["wall_plate.stl", "hook_tip.stl", "screw_hole_top.stl"]
    doc["settings"][KEY] = 0.25
    path = folder / "hook.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    out = folder / "run"
    assert main(["run", str(path), "--out", str(out), "--quiet"]) == 0
    off_block = post_copy(out, folder / "off", None)
    return {"on": out, "off": folder / "off", "off_block": off_block, "t": 0.25}


def test_the_film_around_the_removed_tip_is_gone(hook):
    block = metrics_of(hook["on"])
    assert block["applied"] and block["min_film_mm"] == hook["t"] and block["watertight"] and block["bodies"] == 1
    tip, plate = block["contact"]["load:tip"], block["contact"]["fixed:wall_plate.stl"]
    assert tip["thinnest_wall_mm"] < 0.05 and tip["thin_area_mm2"] > 100                    # the film of P4b-3: walls of 0.03 to 0.19 mm
    for label, rec in block["contact"].items():
        assert rec["thin_area_after_mm2"] == 0.0 and rec["thinnest_wall_after_mm"] >= hook["t"], label
    assert tip["area_after_mm2"] < 0.6 * tip["area_mm2"] and plate["area_after_mm2"] > 0.85 * plate["area_mm2"]
    assert block["film_removed_mm3"] > 10                                                   # measured: 45.75 mm3 of 10,838
    rep = file_report(hook["on"] / "result" / "part_only.stl")
    assert rep["closed_manifold"] and rep["nonmanifold_vertices"] == 0 and rep["bodies"] == 1
    off = hook["off_block"]
    assert off["contact"]["load:tip"]["thinnest_wall_mm"] == tip["thinnest_wall_mm"] and "min_film_mm" not in off
    below = off["contact"]["load:tip"]["area_below_voxel_mm2"]                             # the scale of the whole film (voxel 2 mm): 292.6 of 485.5 mm2
    assert tip["thin_area_mm2"] < below <= tip["area_mm2"] and tip["area_below_voxel_mm2"] == below
    assert below > 200 and tip["thin_area_mm2"] < 0.8 * below                              # a value of 0.25 mm takes 62 % of it; the thinnest wall alone takes ~0
    note = " ".join(off["notes"])
    assert f"load:tip: thinnest wall {tip['thinnest_wall_mm']:g} mm, {below:g} mm2 of contact face with a wall below the voxel size" in note
    more = block["leftovers_dropped"]["volume_mm3"] - off["leftovers_dropped"]["volume_mm3"]
    assert volume_of(hook["off"]) - volume_of(hook["on"]) == pytest.approx(block["film_removed_mm3"] + more, abs=3e-3)
