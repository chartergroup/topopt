"""The `stl` kind of the `build_regions` spec (CONTRACTS.md section 45.1): a region from the user's own STL, with plugged holes.

The parts are built in the tests with manifold3d (a plate with two through holes, a plate with a blind hole, a drill point, a chamfer) and written
as float32 STL files, the way a user's file is. The wall of such a hole is a regular polygon, so the void of a hole is a prism of that polygon: the
volume that a plug adds is known to the last digit (`polygon_area` times the length). A volume is compared to 1e-3 mm3: the file keeps float32,
so a vertex moves by about 1e-6 mm, and the plate has about 3,000 mm3.
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
import yaml
from manifold3d import Manifold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import recorded_hook_run  # noqa: E402

VOL_TOL = 1e-3
HABIT = re.compile(r"\b(must|never|always|conservative)\b", re.I)


def rb():
    from topopt import regions_builder
    return regions_builder


# --- parts ---------------------------------------------------------------------------------------------------------------------------


def cyl(r, h, seg, at):
    return Manifold.cylinder(h, r, r, seg).translate(at)


def plate(seg=32, holes=((10, 10, 3.0), (30, 10, 2.0)), size=(40, 20, 4)):
    """A plate with through holes along z: x, y, radius. `find_features` orders the holes by radius, so H1 is the one with the radius 2."""
    m = Manifold.cube(size)
    for x, y, r in holes:
        m = m - cyl(r, size[2] + 2, seg, (x, y, -1))
    return m


def write_stl(m, path, flip=False):
    mesh = m.to_mesh()
    V, F = np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)
    trimesh.Trimesh(V, F[:, ::-1] if flip else F, process=False).export(str(path))
    return path


def prism(r, seg, length):
    """The volume of the void of a hole whose wall is a regular polygon of `seg` sides with its vertices on the radius `r`."""
    return 0.5 * seg * r * r * math.sin(2.0 * math.pi / seg) * length


def box_roles():
    """The fixed and load regions of the existing kinds, at the two ends of a plate of 40 x 20 x 4 mm."""
    return {"fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 4]}}]},
            "load": {"tip": [{"box": {"min": [40, 5, 0], "max": [44, 15, 4]}}]}}


def spec_of(design, name="p", **extra):
    reg = {"design": design, **box_roles()}
    reg.update(extra)
    return {"name": name, "regions": reg,
            "load_cases": [{"name": "a", "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
            "settings": {"voxel_size": 1.0}}


def said(lines):
    """A `say` function for `build_spec`: it joins its arguments to one line, as `build_in_workspace` does."""
    return lambda *a: lines.append(" ".join(map(str, a)))


def volume_of(path):
    m = trimesh.load(str(path), force="mesh")
    assert m.is_watertight
    return float(m.volume)


def holes_of(path):
    from topopt.features import find_features
    return find_features(path)["holes"]


@pytest.fixture
def parts(tmp_path):
    """A folder with the card's test plate: two through holes (H1 radius 2, H2 radius 3)."""
    write_stl(plate(), tmp_path / "plate.stl")
    return tmp_path


# --- the card: a YAML that check_problem accepts, and the report names the plugged holes ---------------------------------------------------


def test_the_card_a_plate_as_the_design_gives_a_yaml_that_check_accepts(parts):
    from topopt.cli import check_data
    out = parts / "out"
    lines = []
    r = rb().build_spec(spec_of([{"stl": {"path": "plate.stl"}}]), out, say=said(lines), base=parts)
    assert r["ok"], lines
    data = check_data(r["yaml"])                                                      # no UserError: the YAML is a valid problem
    assert data["voxel_counts"]["design"] > 0
    assert volume_of(out / "design.stl") == pytest.approx(plate().volume(), abs=VOL_TOL)      # the design STL has the volume of the plate
    doc = yaml.safe_load(r["yaml"].read_text())
    assert doc["regions"]["design"] == "design.stl"
    assert re.search(r"^stl regions\.design\[0\]: plate\.stl, 276 faces, closed solid, size 40 x 20 x 4 mm, volume 3037\.7 mm3$", lines[0]), lines[0]


def test_the_card_the_report_names_the_plugged_holes(parts):
    out = parts / "out"
    lines = []
    r = rb().build_spec(spec_of([{"stl": {"path": "plate.stl", "plug_holes": ["H1"]}}]), out, say=said(lines), base=parts)
    assert r["ok"], lines
    assert lines[0].endswith("plugged H1 (d 4 mm, 4 mm long, through); open H2"), lines[0]
    assert not [ln for ln in lines if ln.startswith("WARNING")]


# --- the plug: complete, and not past the faces of the part ---------------------------------------------------------------------------------


def built(parts, items, name="p", **roles):
    """Build the spec with `build_spec` and return the output folder, which holds the region STLs."""
    out = parts / "out"
    r = rb().build_spec(spec_of(items, name, **roles), out, say=lambda *_: None, base=parts)
    assert r["ok"]
    return out


def test_a_plugged_hole_adds_the_volume_of_the_hole_and_no_hole_is_left_at_its_place(parts):
    out = built(parts, [{"stl": {"path": "plate.stl", "plug_holes": ["H1"]}}])
    reference = plate(holes=((10, 10, 3.0),))                                        # a part built with no hole at the place of H1
    assert volume_of(out / "design.stl") == pytest.approx(reference.volume(), abs=VOL_TOL)
    assert volume_of(out / "design.stl") == pytest.approx(plate().volume() + prism(2.0, 32, 4.0), abs=VOL_TOL)
    left = holes_of(out / "design.stl")
    assert [(h["id"], h["radius_mm"], h["point"][:2]) for h in left] == [("H1", 3.0, [10.0, 10.0])]      # only the other hole
    box = trimesh.load(str(out / "design.stl"), force="mesh")
    assert np.allclose(box.bounds, [[0, 0, 0], [40, 20, 4]], atol=1e-5)               # nothing outside the faces of the part


def test_plug_all_leaves_no_hole_and_the_part_is_the_full_box(parts):
    out = built(parts, [{"stl": {"path": "plate.stl", "plug_holes": "all"}}])
    assert volume_of(out / "design.stl") == pytest.approx(40 * 20 * 4, abs=VOL_TOL)
    assert holes_of(out / "design.stl") == []
    from topopt.features import find_features
    assert len(find_features(out / "design.stl")["flat_faces"]) == 6                  # the caps join the plate faces: still six faces


@pytest.mark.parametrize("seg", [16, 24, 64, 128])
def test_the_plug_fills_the_whole_hole_for_any_number_of_segments(tmp_path, seg):
    """A cylinder of the plug with its own segments would leave a skin at the facets of the wall (chord sag); the hull of the wall vertices leaves none."""
    write_stl(plate(seg, holes=((20, 10, 5.0),)), tmp_path / "p.stl")
    out = built(tmp_path, [{"stl": {"path": "p.stl", "plug_holes": ["H1"]}}])
    assert volume_of(out / "design.stl") == pytest.approx(40 * 20 * 4, abs=VOL_TOL)
    assert holes_of(out / "design.stl") == []


def test_a_rotated_plate_far_from_the_origin_is_plugged_the_same(tmp_path):
    rot = plate().rotate((20.0, 35.0, 50.0)).translate((120.5, -87.25, 300.0))
    write_stl(rot, tmp_path / "r.stl")
    from topopt.features import find_features
    assert [h["id"] for h in find_features(tmp_path / "r.stl")["holes"]] == ["H1", "H2"]
    shapes = rb().build_all(spec_of([{"stl": {"path": "r.stl", "plug_holes": ["H1"]}}]), tmp_path)
    out = tmp_path / "design.stl"
    rb().to_trimesh(shapes["design"][1]).export(str(out))
    # The reference is the same part with no hole at the place of H1, rotated, moved and written as float32 in the same way. At 300 mm a float32 step is
    # 3e-5 mm, so the volume of any such file differs from the exact volume by about 3e-3 mm3 (measured 2.7e-3 for the reference); the two files differ by 1e-4.
    write_stl(plate(holes=((10, 10, 3.0),)).rotate((20.0, 35.0, 50.0)).translate((120.5, -87.25, 300.0)), tmp_path / "ref.stl")
    assert volume_of(out) == pytest.approx(volume_of(tmp_path / "ref.stl"), abs=1e-3)
    assert [h["radius_mm"] for h in holes_of(out)] == [3.0]


def test_a_blind_hole_is_plugged_and_keeps_its_bottom(tmp_path):
    blind = Manifold.cube((30, 20, 6)) - cyl(4.0, 5, 32, (15, 10, 3.0))              # a flat bottom at z = 3: 3 mm deep from the top at z = 6
    write_stl(blind, tmp_path / "b.stl")
    from topopt.features import find_features
    (hole,) = find_features(tmp_path / "b.stl")["holes"]
    assert hole["depth"] == "blind" and hole["length_mm"] == pytest.approx(3.0)
    out = built(tmp_path, [{"stl": {"path": "b.stl", "plug_holes": ["H1"]}}],
                fixed={"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 6]}}]}, load={"tip": [{"box": {"min": [30, 5, 0], "max": [34, 15, 6]}}]})
    assert volume_of(out / "design.stl") == pytest.approx(30 * 20 * 6, abs=VOL_TOL)      # the full block: the bottom stays, no material outside
    assert holes_of(out / "design.stl") == []
    box = trimesh.load(str(out / "design.stl"), force="mesh")
    assert np.allclose(box.bounds, [[0, 0, 0], [30, 20, 6]], atol=1e-5)


def test_a_drill_point_stays_and_the_report_gives_a_fact_about_the_cavity(tmp_path):
    seg, r = 32, 4.0
    part = Manifold.cube((30, 20, 8)) - cyl(r, 5, seg, (15, 10, 4.0)) - Manifold.cylinder(1.0, 0.0, r, seg).translate((15, 10, 3.0))   # wall z 4 to 8, cone z 3 to 4
    write_stl(part, tmp_path / "d.stl")
    lines = []
    out = tmp_path / "out"
    spec = spec_of([{"stl": {"path": "d.stl", "plug_holes": ["H1"]}}], fixed={"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 8]}}]},
                   load={"tip": [{"box": {"min": [30, 5, 0], "max": [34, 15, 8]}}]})
    rb().build_spec(spec, out, say=said(lines), base=tmp_path)
    cone = prism(r, seg, 1.0) / 3.0                                                  # the cone is a pyramid on the polygon of the wall
    assert volume_of(out / "design.stl") == pytest.approx(30 * 20 * 8 - cone, abs=VOL_TOL)     # the wall is filled, the cone stays as a closed cavity
    warn = [ln for ln in lines if ln.startswith("WARNING")]
    assert len(warn) == 1 and "inner void" in warn[0] and f"{cone:.3g}" in warn[0] and "drill point" in warn[0], warn


def test_a_chamfer_at_the_mouth_stays_a_dimple(tmp_path):
    seg, r = 64, 3.0
    part = Manifold.cube((30, 20, 6)) - cyl(r, 8, seg, (15, 10, -1)) - Manifold.cylinder(0.8, r, r + 0.8, seg).translate((15, 10, 5.2))
    write_stl(part, tmp_path / "c.stl")
    from topopt.features import find_features
    assert [h["id"] for h in find_features(tmp_path / "c.stl")["holes"]] == ["H1"]
    out = built(tmp_path, [{"stl": {"path": "c.stl", "plug_holes": "all"}}],
                fixed={"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 6]}}]}, load={"tip": [{"box": {"min": [30, 5, 0], "max": [34, 15, 6]}}]})
    dimple = Manifold.cube((30, 20, 6)) - Manifold.cylinder(0.8, r, r + 0.8, seg).translate((15, 10, 5.2))
    assert volume_of(out / "design.stl") == pytest.approx(dimple.volume(), abs=VOL_TOL)      # the wall is filled; the chamfer is not part of the wall


def test_a_notch_is_an_arc_and_the_plug_fills_it_to_the_chord(tmp_path):
    notch = Manifold.cube((30, 20, 4)) - cyl(4.0, 6, 64, (15, 1.0, -1))              # a notch in the edge y = 0: the wall covers about 209 degrees
    write_stl(notch, tmp_path / "n.stl")
    from topopt.features import find_features
    (hole,) = find_features(tmp_path / "n.stl")["holes"]
    assert hole["closed_ring"] is False and hole["arc_deg"] > 180.0
    lines = []
    out = tmp_path / "out"
    rb().build_spec(spec_of([{"stl": {"path": "n.stl", "plug_holes": "all"}}], fixed={"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 4]}}]},
                            load={"tip": [{"box": {"min": [30, 5, 0], "max": [34, 15, 4]}}]}), out, say=said(lines), base=tmp_path)
    assert volume_of(out / "design.stl") == pytest.approx(30 * 20 * 4, abs=VOL_TOL)    # here the chord of the arc lies in the face y = 0: the notch is full
    assert any(f"H1 is an arc of {hole['arc_deg']:g} deg" in ln and "chord" in ln for ln in lines), lines


def test_many_holes_are_plugged_in_one_item(tmp_path):
    holes = tuple((6 + 8 * i, 6 + 8 * j, 2.0) for i in range(5) for j in range(2))
    write_stl(plate(32, holes, size=(46, 20, 4)), tmp_path / "m.stl")
    out = built(tmp_path, [{"stl": {"path": "m.stl", "plug_holes": "all"}}],
                fixed={"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 4]}}]}, load={"tip": [{"box": {"min": [46, 5, 0], "max": [50, 15, 4]}}]})
    assert volume_of(out / "design.stl") == pytest.approx(46 * 20 * 4, abs=VOL_TOL)
    assert holes_of(out / "design.stl") == []
    notes = []
    rb().build_all(spec_of([{"stl": {"path": "m.stl", "plug_holes": "all"}}]), tmp_path, notes)
    assert notes[0].endswith("; plugged H1, H2, H3, H4, H5, H6, H7, H8, H9, H10 (d 4 mm, 4 mm long, through); open none"), notes[0]      # equal holes in one group
    with pytest.raises(rb().SpecError, match=r"Known ids: H1, H2, H3, H4, H5, H6, H7, H8, H9, H10 \(d 4 mm, 4 mm long, through\)$"):
        rb().build_all(spec_of([{"stl": {"path": "m.stl", "plug_holes": ["H11"]}}]), tmp_path)


# --- the kind works like the other kinds: union, subtract, every role -------------------------------------------------------------------


def test_a_growth_box_next_to_the_stl_item_is_united_with_it(parts):
    out = built(parts, [{"stl": {"path": "plate.stl"}}, {"box": {"min": [0, 0, 4], "max": [40, 20, 10]}}], load={"tip": [{"box": {"min": [40, 5, 0], "max": [44, 15, 4]}}]})
    assert volume_of(out / "design.stl") == pytest.approx(plate().volume() + 40 * 20 * 6, abs=VOL_TOL)      # material may be added above the plate
    out2 = built(parts, [{"stl": {"path": "plate.stl", "plug_holes": "all"}}, {"box": {"min": [20, 5, 2], "max": [60, 15, 9]}}], name="q",
                 load={"tip": [{"box": {"min": [60, 5, 0], "max": [64, 15, 4]}}]})
    expect = 40 * 20 * 4 + 40 * 10 * 7 - 20 * 10 * 2                                # the plate (full) plus the box, less the part of the box inside the plate
    assert volume_of(out2 / "design.stl") == pytest.approx(expect, abs=VOL_TOL)


def test_the_stl_item_works_in_every_role(parts):
    block = parts / "block.stl"
    write_stl(Manifold.cube((10, 10, 4)).translate((100, 0, 0)), block)
    item = [{"stl": {"path": "block.stl"}}]
    for role in ("fixed", "load", "solid", "keep_out"):
        spec = {"regions": {"design": [{"stl": {"path": "plate.stl"}}], role: {"blk": item}, **({} if role == "fixed" else {"fixed": {"w": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 4]}}]}}),
                            **({} if role == "load" else {"load": {"tip": [{"box": {"min": [40, 5, 0], "max": [44, 15, 4]}}]}})}}
        shapes = rb().build_all(spec, parts)
        assert shapes["blk"][0] == role and shapes["blk"][1].volume() == pytest.approx(400.0, abs=VOL_TOL), role
        assert shapes["design"][1].volume() == pytest.approx(plate().volume(), abs=VOL_TOL)


def test_the_stl_item_as_a_keep_out_and_inside_subtract(parts):
    spec = spec_of([{"box": {"min": [0, 0, 0], "max": [40, 20, 10]}}, {"subtract": {"stl": {"path": "plate.stl"}}}],
                   keep_out={"cut": [{"stl": {"path": "plate.stl", "plug_holes": ["H1"]}}]})
    shapes = rb().build_all(spec, parts)
    assert shapes["design"][1].volume() == pytest.approx(40 * 20 * 10 - plate().volume(), abs=VOL_TOL)      # the box minus the plate (the holes stay in the cutter)
    assert shapes["cut"][0] == "keep_out"
    assert shapes["cut"][1].volume() == pytest.approx(plate().volume() + prism(2.0, 32, 4.0), abs=VOL_TOL)    # a keep-out with H1 plugged
    shapes = rb().build_all(spec_of([{"box": {"min": [0, 0, 0], "max": [40, 20, 10]}}, {"subtract": {"stl": {"path": "plate.stl", "plug_holes": "all"}}}]), parts)
    assert shapes["design"][1].volume() == pytest.approx(40 * 20 * 10 - 40 * 20 * 4, abs=VOL_TOL)             # plug_holes works inside subtract too


def test_a_relative_path_starts_at_the_work_folder_for_the_tool_function(tmp_path, monkeypatch):
    work, elsewhere = tmp_path / "ws", tmp_path / "elsewhere"
    (work / "parts").mkdir(parents=True)
    elsewhere.mkdir()
    write_stl(plate(), work / "parts" / "plate.stl")
    monkeypatch.chdir(elsewhere)                                                       # not the work folder, not the folder of a spec
    r = rb().build_in_workspace(spec_of([{"stl": {"path": "parts/plate.stl", "plug_holes": ["H1"]}}]), work)
    assert r["ok"], r
    assert any("plugged H1" in ln for ln in r["report"])
    assert volume_of(work / "runs" / "p" / "design.stl") == pytest.approx(plate().volume() + prism(2.0, 32, 4.0), abs=VOL_TOL)
    assert r["check"] is not None


def test_a_relative_path_starts_at_the_folder_of_the_spec_for_the_cli(tmp_path, monkeypatch, capsys):
    folder, elsewhere = tmp_path / "s", tmp_path / "elsewhere"
    folder.mkdir()
    elsewhere.mkdir()
    write_stl(plate(), folder / "plate.stl")
    (folder / "spec.yaml").write_text(yaml.safe_dump(spec_of([{"stl": {"path": "plate.stl", "plug_holes": ["H1"]}}])))
    monkeypatch.chdir(elsewhere)
    assert rb().main([str(folder / "spec.yaml"), "--out", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert "plugged H1" in printed and "RESULT: OK" in printed
    assert volume_of(tmp_path / "out" / "design.stl") == pytest.approx(plate().volume() + prism(2.0, 32, 4.0), abs=VOL_TOL)
    (folder / "bad.yaml").write_text(yaml.safe_dump(spec_of([{"stl": {"path": "missing.stl"}}])))
    assert rb().main([str(folder / "bad.yaml"), "--out", str(tmp_path / "out2")]) == 1
    assert "the folder of the spec file" in capsys.readouterr().out


def test_a_caller_with_no_base_folder_reads_a_relative_path_from_the_current_folder(parts, monkeypatch):
    monkeypatch.chdir(parts)
    shapes = rb().build_all(spec_of([{"stl": {"path": "plate.stl"}}]))
    assert shapes["design"][1].volume() == pytest.approx(plate().volume(), abs=VOL_TOL)
    with pytest.raises(rb().SpecError, match="the current folder"):
        rb().build_all(spec_of([{"stl": {"path": "nothing.stl"}}]))


# --- a mesh or a spec that cannot be used ---------------------------------------------------------------------------------------------------


def error_of(parts, item, **roles):
    with pytest.raises(rb().SpecError) as e:
        rb().build_spec(spec_of(item, **roles), parts / "out", say=lambda *_: None, base=parts)
    return str(e.value)


def test_a_file_that_is_not_there_names_the_key_path_the_file_and_the_cause(parts):
    text = error_of(parts, [{"stl": {"path": "nothing.stl"}}])
    assert text.startswith("regions.design[0].stl.path: nothing.stl: file not found") and "\n" not in text
    assert str(parts) in text and "the folder of the spec file" in text or "the current folder" in text
    keyed = error_of(parts, [{"box": {"min": [0, 0, 0], "max": [1, 1, 1]}}], fixed={"w": [{"box": {"min": [0, 0, 0], "max": [1, 1, 1]}}, {"stl": {"path": "x.stl"}}]})
    assert keyed.startswith("regions.fixed.w[1].stl.path: x.stl:")
    sub = error_of(parts, [{"box": {"min": [0, 0, 0], "max": [1, 1, 1]}}, {"subtract": {"stl": {"path": "x.stl"}}}])
    assert sub.startswith("regions.design[1].subtract.stl.path: x.stl:")


def test_a_text_file_an_empty_file_and_a_folder_are_not_stl_files(parts):
    (parts / "notes.stl").write_text("this is not a mesh")
    (parts / "empty.stl").write_bytes(b"")
    (parts / "folder.stl").mkdir()
    for name, cause in (("notes.stl", "not an STL file"), ("empty.stl", "is empty"), ("folder.stl", "this is a folder")):
        text = error_of(parts, [{"stl": {"path": name}}])
        assert text.startswith(f"regions.design[0].stl.path: {name}: ") and cause in text and "\n" not in text, text


def test_an_open_mesh_and_inward_faces_are_refused_with_no_repair(parts):
    mesh = trimesh.load(str(parts / "plate.stl"), force="mesh", process=False)
    trimesh.Trimesh(mesh.vertices, mesh.faces[:-1], process=False).export(str(parts / "open.stl"))
    text = error_of(parts, [{"stl": {"path": "open.stl"}}])
    assert text.startswith("regions.design[0].stl.path: open.stl: the mesh is not a closed solid") and "open edges" in text and "does not repair" in text
    write_stl(plate(), parts / "inward.stl", flip=True)
    text = error_of(parts, [{"stl": {"path": "inward.stl"}}])
    assert "the faces point inward" in text and text.startswith("regions.design[0].stl.path: inward.stl:")
    mixed = mesh.faces.copy()
    mixed[0] = mixed[0][::-1]
    trimesh.Trimesh(mesh.vertices, mixed, process=False).export(str(parts / "mixed.stl"))
    assert "winding is not consistent" in error_of(parts, [{"stl": {"path": "mixed.stl"}}])


def test_an_unknown_hole_id_lists_the_known_ids_with_their_size(parts):
    text = error_of(parts, [{"stl": {"path": "plate.stl", "plug_holes": ["H1", "H9"]}}])
    assert text.startswith("regions.design[0].stl.plug_holes: 'H9' is not a hole of plate.stl")
    assert "H1 (d 4 mm, 4 mm long, through)" in text and "H2 (d 6 mm, 4 mm long, through)" in text and "\n" not in text
    write_stl(Manifold.cube((10, 10, 4)), parts / "solid.stl")
    assert "none (find_features lists no hole" in error_of(parts, [{"stl": {"path": "solid.stl", "plug_holes": ["H1"]}}])


@pytest.mark.parametrize("item,where,expect", [
    ({"stl": "plate.stl"}, "regions.design[0].stl", "must be a mapping with a path"),
    ({"stl": {"path": 5}}, "regions.design[0].stl.path", "must be a file name (text)"),
    ({"stl": {"path": ["a.stl"]}}, "regions.design[0].stl.path", "must be a file name (text)"),
    ({"stl": {"path": "plate.stl", "plug_holes": "H1"}}, "regions.design[0].stl.plug_holes", "must be a list of hole ids"),
    ({"stl": {"path": "plate.stl", "plug_holes": [1]}}, "regions.design[0].stl.plug_holes", "must be a list of hole ids"),
    ({"stl": {"path": "plate.stl", "plug_holes": {"H1": True}}}, "regions.design[0].stl.plug_holes", "must be a list of hole ids"),
    ({"stl": {"path": "plate.stl", "plug": ["H1"]}}, "regions.design[0].stl.plug", "is not a key of an stl item"),
    ({"stl": {"plug_holes": ["H1"]}}, "regions.design[0].stl.path", "is missing"),
])
def test_a_wrong_type_or_key_names_the_key_path_before_the_build(parts, item, where, expect):
    spec = spec_of([item])
    with pytest.raises(rb().SpecError) as e:                                           # the shape walk: nothing is read from the disk
        rb().check_spec_shape(spec)
    assert str(e.value).startswith(where) and expect in str(e.value), str(e.value)
    again = rb().build_in_workspace(spec, parts)                                       # the MCP entry: the same line
    assert again["ok"] is False and again["error"].startswith(where) and expect in again["error"] and "\n" not in again["error"]


def test_the_mcp_entry_gives_the_error_form_for_a_missing_file(parts):
    r = rb().build_in_workspace(spec_of([{"stl": {"path": "nothing.stl"}}]), parts)
    assert r["ok"] is False and set(r) == {"ok", "error"}
    assert r["error"].startswith("regions.design[0].stl.path: nothing.stl: file not found") and "the work folder" in r["error"]


def test_a_region_named_stl_with_a_list_still_works(parts):
    shapes = rb().build_all(spec_of([{"stl": {"path": "plate.stl"}}], keep_out={"stl": [{"box": {"min": [1, 1, 1], "max": [2, 2, 2]}}]}), parts)
    assert shapes["stl"][0] == "keep_out" and shapes["stl"][1].volume() == pytest.approx(1.0)


def test_an_empty_plug_list_plugs_nothing_and_a_repeated_id_plugs_once(parts):
    lines = []
    out = parts / "out"
    rb().build_spec(spec_of([{"stl": {"path": "plate.stl", "plug_holes": []}}]), out, say=said(lines), base=parts)
    assert "plugged" not in lines[0] and volume_of(out / "design.stl") == pytest.approx(plate().volume(), abs=VOL_TOL)
    lines.clear()
    rb().build_spec(spec_of([{"stl": {"path": "plate.stl", "plug_holes": ["H1", "H1"]}}]), out, say=said(lines), base=parts)
    assert lines[0].count("H1 (") == 1 and volume_of(out / "design.stl") == pytest.approx(plate().volume() + prism(2.0, 32, 4.0), abs=VOL_TOL)


# --- the numbers of the wall test ----------------------------------------------------------------------------------------------------------


def test_the_wall_tolerance_comes_from_the_rounding_digits_of_the_reply():
    from topopt import features
    b = rb()
    assert b.WALL_ROUND_MM == 2 * 10 ** -features.MM_DIGITS and b.WALL_AXIS_PER_MM == 10 ** -features.UNIT_DIGITS


def test_the_wall_vertices_are_the_same_for_a_wide_range_of_tolerance(tmp_path):
    """Measured: the wall vertices of a rotated plate (float32 file, 64 segments) lie up to 6e-5 mm off the cylinder of the reply. The set that the plug
    uses is the same from the default tolerance (2e-4 mm) up to 50 times more (1e-2 mm: the size of a small chamfer); a tolerance 10 times smaller loses vertices."""
    from scipy.spatial import cKDTree
    from topopt.features import find_features
    from topopt.printcheck import load_stl_float32
    write_stl(plate(64).rotate((20.0, 35.0, 50.0)).translate((12.5, -7.25, 30.0)), tmp_path / "t.stl")
    V, _F = load_stl_float32(tmp_path / "t.stl")
    tree = cKDTree(V)
    holes = find_features(tmp_path / "t.stl")["holes"]
    b = rb()
    sizes = {}
    for scale in (0.1, 1.0, 10.0, 50.0):
        old = b.WALL_ROUND_MM
        b.WALL_ROUND_MM = old * scale
        try:
            sizes[scale] = [len(b._wall_vertices(V, tree, h)[0]) for h in holes]
        finally:
            b.WALL_ROUND_MM = old
    assert sizes[1.0] == sizes[10.0] == sizes[50.0] == [128, 128]                      # two rings of 64 vertices
    assert sum(sizes[0.1]) < 256


def test_a_wall_vertex_that_the_plug_missed_is_named_in_a_warning(tmp_path, monkeypatch):
    """The check after the plug uses the looser tolerance of the cylinder test of find_features. With a tolerance of the plug that is too small
    (a quarter of the default: 238 of 256 wall vertices at 64 segments, see the test above), wall faces stay in the part and the report says so."""
    write_stl(plate(64).rotate((20.0, 35.0, 50.0)).translate((12.5, -7.25, 30.0)), tmp_path / "t.stl")
    monkeypatch.setattr(rb(), "WALL_ROUND_MM", rb().WALL_ROUND_MM * 0.25)
    notes = []
    rb().build_all(spec_of([{"stl": {"path": "t.stl", "plug_holes": ["H1"]}}]), tmp_path, notes)
    warn = [ln for ln in notes if ln.startswith("WARNING")]
    assert warn and "H1" in warn[0] and "faces of its wall stay in the part" in warn[0], notes
    monkeypatch.undo()
    notes.clear()
    rb().build_all(spec_of([{"stl": {"path": "t.stl", "plug_holes": ["H1"]}}]), tmp_path, notes)
    assert not [ln for ln in notes if ln.startswith("WARNING")], notes                  # the default tolerance: no warning


def test_a_real_mesh_whose_hole_is_not_an_exact_prism_gives_one_clear_error(tmp_path):
    """Real data: the recorded hook (an optimizer result, 85,610 faces) has one recess that find_features lists (R 4.5 mm, 0.56 mm deep). Its wall has
    vertices at 84 different depths and no flat floor, so the hull of the wall vertices is not the void: wall faces stay and the surface pinches when it is
    written as float32. The builder says so with the key path, in place of the generic 'region is not a closed solid' of the writer."""
    recorded = recorded_hook_run() / "result" / "final.stl"
    if not recorded.is_file():
        pytest.skip("the recorded hook is not in this checkout")
    item = [{"stl": {"path": str(recorded)}}]
    assert rb().build_all(spec_of(item), tmp_path)["design"][1].volume() == pytest.approx(21362.804, abs=VOL_TOL)      # the mesh alone is a valid region
    from topopt.features import find_features
    assert [h["id"] for h in find_features(recorded)["holes"]] == ["H1"]
    with pytest.raises(rb().SpecError) as e:
        rb().build_all(spec_of([{"stl": {"path": str(recorded), "plug_holes": ["H1"]}}]), tmp_path)
    text = str(e.value)
    assert text.startswith("regions.design[0].stl.plug_holes: the plugged mesh is not a closed solid (H1: wall faces stay in the part)") and "\n" not in text, text


def test_a_plugged_mesh_that_is_not_closed_is_refused_with_the_key_path(parts, monkeypatch):
    """The plumbing of the same error on the test plate: a writer test that fails is named with the key path of the item."""
    monkeypatch.setattr(rb(), "to_trimesh", lambda m: trimesh.Trimesh())
    with pytest.raises(rb().SpecError, match=r"^regions\.design\[0\]\.stl\.plug_holes: the plugged mesh is not a closed solid\. "):
        rb().build_all(spec_of([{"stl": {"path": "plate.stl", "plug_holes": ["H1"]}}]), parts)


# --- the MCP tool and its texts ------------------------------------------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    return tmp_path


def tools():
    from topopt_mcp import server
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_the_tool_builds_a_spec_with_an_stl_item_and_returns_the_plugged_holes(ws):
    from topopt_mcp import server
    (ws / "parts").mkdir()
    write_stl(plate(), ws / "parts" / "plate.stl")
    r = server.build_regions(spec_of([{"stl": {"path": "parts/plate.stl", "plug_holes": ["H1"]}}, {"box": {"min": [0, 0, 4], "max": [40, 20, 6]}}]))
    assert r["ok"] is True and r["builder_ok"] and r["check"] is not None, r
    assert r["report"][0].startswith("stl regions.design[0]: parts/plate.stl, 276 faces, closed solid") and "plugged H1 (d 4 mm, 4 mm long, through); open H2" in r["report"][0]
    json.dumps(r, allow_nan=False)
    assert volume_of(ws / r["run_dir_rel"] / "design.stl") == pytest.approx(plate().volume() + prism(2.0, 32, 4.0) + 40 * 20 * 2, abs=VOL_TOL)
    bad = server.build_regions(spec_of([{"stl": {"path": "parts/plate.stl", "plug_holes": ["H7"]}}]))
    assert bad["ok"] is False and "Known ids: H1" in bad["error"] and set(bad) == {"ok", "error"}


def test_the_texts_name_the_kind_and_stay_within_the_bound():
    texts = {n: t.description for n, t in tools().items()}
    build, find = texts["build_regions"], texts["find_features"]
    assert len(build) < 4096 and len(find) < 2048                                       # the bounds of test_load_types and test_find_features
    assert not HABIT.search(build) and not HABIT.search(find), (HABIT.search(build), HABIT.search(find))
    assert "stl" in build and "find_features" in build
    for word in ("plug_holes", "path", "work folder", "stable"):
        assert word in find, word
    assert "interfaces" not in find.split("plug_holes")[1][:400]                        # the kind of the YAML block `interfaces` is another thing


def test_the_mock_of_the_tools_equals_the_server():
    mock = json.loads((REPO / "evals" / "mocks" / "topopt" / "_tools.json").read_text())
    got = {t["name"]: t for t in mock["tools"]}
    for name, tool in tools().items():
        entry = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert inspect.cleandoc(got[name]["description"]) == inspect.cleandoc(entry["description"]), name      # no indent: Python 3.13 strips it from a docstring, 3.12 does not
        assert got[name]["inputSchema"] == entry["inputSchema"], name
    assert set(got) == set(tools())
