"""A frozen ring round each screw hole of an `stl` item, and the `screw_ring` region item (CONTRACTS.md section 45.2).

The parts are built in the tests with manifold3d and written as float32 STL files, the way a user's file is. The test plate of the card has two
through holes along z: H1 (d 4 mm, at x = 30) and H2 (d 6 mm, at x = 10), 32 segments, 40 x 20 x 4 mm. `find_features` orders the holes by radius.

The thickness t of the ring is a method setting: the default is 0.5 x d, at least 2 voxel sizes of the written YAML, and the
spec key `ring_mm` sets it. A ring volume is compared to the exact annulus within 1 percent: the ring wall is a polygon with a chord sag of 0.002 mm.
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

HABIT = re.compile(r"\b(must|never|always|conservative)\b", re.I)
LEAD_LINE = "screw ring regions.design[0].stl: "


def rb():
    from topopt import regions_builder
    return regions_builder


# --- parts ------------------------------------------------------------------------------------------------------------------------------------


def cyl(r, h, seg, at):
    return Manifold.cylinder(h, r, r, seg).translate(at)


def plate(seg=32, holes=((10, 10, 3.0), (30, 10, 2.0)), size=(40, 20, 4)):
    m = Manifold.cube(size)
    for x, y, r in holes:
        m = m - cyl(r, size[2] + 2, seg, (x, y, -1))
    return m


def write_stl(m, path):
    mesh = m.to_mesh()
    trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False).export(str(path))
    return path


def annulus(r_in, r_out, length):
    return math.pi * (r_out ** 2 - r_in ** 2) * length


def volume_of(path):
    m = trimesh.load(str(path), force="mesh")
    assert m.is_watertight
    return float(m.volume)


def holes_of(path):
    from topopt.features import find_features
    return find_features(path)["holes"]


def box_roles():
    return {"fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 4]}}]},
            "load": {"tip": [{"box": {"min": [40, 5, 0], "max": [44, 15, 4]}}]}}


def spec_of(design, name="p", voxel=1.0, **roles):
    reg = {"design": design, **box_roles()}
    reg.update(roles)
    return {"name": name, "regions": reg,
            "load_cases": [{"name": "a", "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
            "settings": {"voxel_size": voxel}}


def ring_spec(voxel=1.0, **item):
    """The plate as the design with screw holes H1 and H2, the wall of H2 as the fixed region and a pull on the ring of H1."""
    design = [{"stl": {"path": "plate.stl", "screw_holes": ["H1", "H2"], **item}}]
    spec = spec_of(design, voxel=voxel, fixed={"wall": [{"screw_ring": {"path": "plate.stl", "hole": "H2"}}]},
                   load={"pull": [{"screw_ring": {"path": "plate.stl", "hole": "H1"}}]})
    spec["load_cases"] = [{"name": "pull", "forces": [{"region": "pull", "force": [20.0, 0.0, 0.0]}]}]
    return spec


def said(lines):
    return lambda *a: lines.append(" ".join(map(str, a)))


@pytest.fixture
def parts(tmp_path):
    """A folder with the test plate of the card."""
    write_stl(plate(), tmp_path / "plate.stl")
    return tmp_path


def build(parts, spec, **kw):
    """`(result, report lines, yaml document)` of `build_spec` on the folder `parts`."""
    lines = []
    r = rb().build_spec(spec, parts / "out", say=said(lines), base=parts, **kw)
    return r, lines, yaml.safe_load(Path(r["yaml"]).read_text())


def tubes(doc):
    return {e["name"]: e for e in doc["interfaces"]}


# --- the interface entries: one `tube` for each screw hole, with the values of the hole ---------------------------------------------------------


def test_each_screw_hole_gets_one_tube_with_the_values_of_the_hole(parts):
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1", "H2"]}}]))
    assert r["ok"], lines
    found = {h["id"]: h for h in holes_of(parts / "plate.stl")}
    assert list(tubes(doc)) == ["design_plate_H1", "design_plate_H2"]
    for hid in ("H1", "H2"):
        e, h = tubes(doc)[f"design_plate_{hid}"], found[hid]
        assert e["kind"] == "tube"
        assert e["point"] == h["point"] and e["axis"] == h["axis"] and e["length_mm"] == h["length_mm"] == 4.0
        assert e["inner_radius_mm"] == h["radius_mm"]
        assert e["radius_mm"] == pytest.approx(h["radius_mm"] + 0.5 * h["diameter_mm"], abs=1e-9)        # t = 0.5 x d
    assert tubes(doc)["design_plate_H1"]["point"] == [30.0, 10.0, 0.0] and tubes(doc)["design_plate_H2"]["point"] == [10.0, 10.0, 0.0]


def test_the_written_yaml_is_a_valid_problem_with_the_tubes(parts):
    from topopt.cli import check_data
    from topopt.config import load_config
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": "all"}}]))
    assert r["ok"] and r["kind"] == "ok", lines
    cfg = load_config(r["yaml"])
    assert [(i.name, i.kind, i.op) for i in cfg.interfaces] == [("design_plate_H1", "tube", "union"), ("design_plate_H2", "tube", "union")]
    data = check_data(r["yaml"])
    names = [f["name"] for f in data["interfaces"]]
    assert names == ["design_plate_H1", "design_plate_H1 (hole)", "design_plate_H2", "design_plate_H2 (hole)"]
    assert all(f["voxels"] > 0 and f["held"] > 0 for f in data["interfaces"])


def test_a_part_without_screw_holes_writes_no_interfaces(parts):
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "plug_holes": ["H1"]}}]))
    assert "interfaces" not in doc
    assert not [ln for ln in lines if ln.startswith("screw ring")]


def test_all_screw_holes_are_every_hole_and_a_list_is_a_subset(parts):
    _, _, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H2"]}}]))
    assert list(tubes(doc)) == ["design_plate_H2"]
    _, _, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": "all"}}]))
    assert list(tubes(doc)) == ["design_plate_H1", "design_plate_H2"]


def test_a_plugged_hole_and_a_screw_hole_in_one_item(parts):
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "plug_holes": ["H1"], "screw_holes": ["H2"]}}]))
    assert r["ok"], lines
    assert list(tubes(doc)) == ["design_plate_H2"]
    assert lines[0].endswith("plugged H1 (d 4 mm, 4 mm long, through); open H2"), lines[0]


def test_a_blind_hole_gets_a_ring_along_its_depth(tmp_path):
    block = Manifold.cube((30, 20, 6)) - cyl(4.0, 5, 32, (15, 10, 3.0))              # a flat bottom at z = 3: 3 mm deep from the top at z = 6
    write_stl(block, tmp_path / "b.stl")
    (h,) = holes_of(tmp_path / "b.stl")
    assert h["depth"] == "blind" and h["length_mm"] == pytest.approx(3.0) and h["axis"] == [0.0, 0.0, -1.0]        # from the open end to the bottom
    spec = spec_of([{"stl": {"path": "b.stl", "screw_holes": ["H1"]}}], fixed={"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 6]}}]},
                   load={"tip": [{"box": {"min": [30, 5, 0], "max": [34, 15, 6]}}]})
    r, lines, doc = build(tmp_path, spec)
    assert r["ok"], lines
    (e,) = doc["interfaces"]
    assert (e["point"], e["axis"], e["length_mm"], e["inner_radius_mm"], e["radius_mm"]) == ([15.0, 10.0, 6.0], [0.0, 0.0, -1.0], 3.0, 4.0, 8.0)


def test_a_rotated_part_far_from_the_origin_gets_the_fit_values_of_its_hole(tmp_path):
    part = plate().rotate((20.0, 35.0, 50.0)).translate((300.0, -200.0, 120.0))
    write_stl(part, tmp_path / "r.stl")
    found = {h["id"]: h for h in holes_of(tmp_path / "r.stl")}
    box = part.bounding_box()
    lo, hi = np.array(box[:3]), np.array(box[3:])
    spec = spec_of([{"stl": {"path": "r.stl", "screw_holes": "all"}}],
                   fixed={"wall": [{"box": {"min": (lo - 3).tolist(), "max": (lo + 1).tolist()}}]}, load={"tip": [{"box": {"min": (hi - 1).tolist(), "max": (hi + 3).tolist()}}]})
    r, lines, doc = build(tmp_path, spec)
    for hid in ("H1", "H2"):
        e, h = tubes(doc)[f"design_r_{hid}"], found[hid]
        assert (e["point"], e["axis"], e["length_mm"], e["inner_radius_mm"]) == (h["point"], h["axis"], h["length_mm"], h["radius_mm"])       # the reply, not a rounded text
    assert not [ln for ln in lines if "the ring of" in ln], lines                         # no "outside the part" note: the rounding of a far part is below the noise


def test_the_interface_of_the_spec_comes_first_and_the_names_are_unique(parts):
    own = {"name": "pin", "kind": "bore", "point": [20.0, 10.0, 0.0], "axis": [0, 0, 1], "radius_mm": 1.0, "length_mm": 4.0}
    spec = spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"]}}, {"stl": {"path": "plate.stl", "screw_holes": ["H1"]}}],
                   fixed={"w": [{"stl": {"path": "plate.stl", "screw_holes": ["H1"]}}]})
    spec["interfaces"] = [own]
    r, lines, doc = build(parts, spec)
    assert [e["name"] for e in doc["interfaces"]] == ["pin", "design_plate_H1", "design_plate_H1_2", "w_plate_H1"]
    assert doc["interfaces"][0] == own
    from topopt.config import load_config
    assert len(load_config(r["yaml"]).interfaces) == 4


def test_an_interfaces_key_of_the_spec_alone_is_copied_and_must_be_a_list_of_mappings(parts):
    own = {"name": "pin", "kind": "bore", "point": [20.0, 10.0, 0.0], "axis": [0, 0, 1], "radius_mm": 1.0, "length_mm": 4.0}
    spec = spec_of([{"stl": {"path": "plate.stl"}}])
    spec["interfaces"] = [own]
    _, _, doc = build(parts, spec)
    assert doc["interfaces"] == [own]
    spec["interfaces"] = "pin"
    with pytest.raises(rb().SpecError, match=r"interfaces must be a list of mappings"):
        rb().build_spec(spec, parts / "o2", say=lambda *_: None, base=parts)


# --- every error: one line with the key path ------------------------------------------------------------------------------------------------


def err(parts, items, **roles):
    with pytest.raises(rb().SpecError) as e:
        rb().build_all(spec_of(items, **roles), parts)
    return str(e.value)


@pytest.mark.parametrize("value", [5, "H1", {"H1": 1}, [1], [None], [""]])
def test_screw_holes_of_the_wrong_type_is_a_spec_error_with_the_key_path(parts, value):
    with pytest.raises(rb().SpecError, match=r"^regions\.design\[0\]\.stl\.screw_holes must be a list of hole ids from find_features"):
        rb().check_spec_shape(spec_of([{"stl": {"path": "plate.stl", "screw_holes": value}}]))


def test_an_unknown_hole_id_lists_the_known_ids(parts):
    text = err(parts, [{"stl": {"path": "plate.stl", "screw_holes": ["H1", "H9"]}}])
    assert text == "regions.design[0].stl.screw_holes: 'H9' is not a hole of plate.stl. Known ids: H1 (d 4 mm, 4 mm long, through), H2 (d 6 mm, 4 mm long, through)"


@pytest.mark.parametrize("plug, screw", [(["H1"], ["H1"]), ("all", ["H1"]), (["H1"], "all"), ("all", "all")])
def test_a_hole_in_plug_holes_and_in_screw_holes_is_a_spec_error(parts, plug, screw):
    text = err(parts, [{"stl": {"path": "plate.stl", "plug_holes": plug, "screw_holes": screw}}])
    assert re.match(r"^regions\.design\[0\]\.stl\.screw_holes: H\d(, H\d)? is also in plug_holes\. A hole is filled or it takes a screw", text), text


def test_a_notch_cannot_take_a_ring_and_the_error_says_why(tmp_path):
    write_stl(Manifold.cube((30, 20, 4)) - cyl(4.0, 6, 64, (15, 1.0, -1)), tmp_path / "n.stl")
    (h,) = holes_of(tmp_path / "n.stl")
    assert h["closed_ring"] is False
    text = err(tmp_path, [{"stl": {"path": "n.stl", "screw_holes": ["H1"]}}])
    assert text.startswith(f"regions.design[0].stl.screw_holes: H1 is an arc of {h['arc_deg']:g} deg, not a closed ring (a notch)"), text
    assert err(tmp_path, [{"stl": {"path": "n.stl", "screw_holes": "all"}}]).startswith("regions.design[0].stl.screw_holes: H1 is an arc")


def test_a_cavity_cannot_take_a_ring_and_the_error_says_why(tmp_path):
    write_stl(Manifold.cube((30, 20, 10)) - cyl(3.0, 4, 32, (15, 10, 3.0)), tmp_path / "c.stl")
    assert holes_of(tmp_path / "c.stl")[0]["depth"] == "closed"
    text = err(tmp_path, [{"stl": {"path": "c.stl", "screw_holes": ["H1"]}}])
    assert text.startswith("regions.design[0].stl.screw_holes: H1 is a cavity inside the part (both ends closed)"), text


def test_a_screw_ring_is_for_the_part_and_not_for_a_cutter(parts):
    item = {"stl": {"path": "plate.stl", "screw_holes": ["H1"]}}
    assert err(parts, [{"box": {"min": [0, 0, 0], "max": [40, 20, 4]}}], keep_out={"hole": [item]}).startswith("regions.keep_out.hole[0].stl.screw_holes: a screw ring is a frozen shape")
    text = err(parts, [{"box": {"min": [0, 0, 0], "max": [40, 20, 4]}}, {"subtract": item}])
    assert text.startswith("regions.design[1].subtract.stl.screw_holes: a screw ring is a frozen shape") and "subtract" in text, text


def test_a_missing_file_gives_the_error_of_the_stl_kind(parts):
    text = err(parts, [{"stl": {"path": "nothing.stl", "screw_holes": ["H1"]}}])
    assert text.startswith("regions.design[0].stl.path: nothing.stl: file not found")


# --- the report: one line for each screw hole ------------------------------------------------------------------------------------------------


def test_the_report_has_one_line_for_each_screw_hole_with_d_t_and_why(parts):
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1", "H2"]}}]))
    assert lines[0].startswith("stl regions.design[0]: plate.stl, 276 faces, closed solid")
    assert lines[1] == LEAD_LINE + "H1 (d 4 mm), t 2 mm (default 0.5 x d); tube interface 'design_plate_H1'"
    assert lines[2] == LEAD_LINE + "H2 (d 6 mm), t 3 mm (default 0.5 x d); tube interface 'design_plate_H2'"
    assert lines[3].startswith("wrote ")
    assert not [ln for ln in lines if ln.startswith("WARNING")]


def test_a_voxel_of_1_5_mm_raises_the_thin_ring_to_two_voxels_and_the_line_says_so(parts):
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1", "H2"]}}], voxel=1.5))
    assert lines[1] == (LEAD_LINE + "H1 (d 4 mm), t 3 mm (raised to 2 voxels; 0.5 x d is 2 mm and the voxel is 1.5 mm); tube interface 'design_plate_H1'"), lines[1]
    assert lines[2] == LEAD_LINE + "H2 (d 6 mm), t 3 mm (default 0.5 x d); tube interface 'design_plate_H2'"       # 0.5 x d is 3 mm = 2 voxels: no raise
    assert tubes(doc)["design_plate_H1"]["radius_mm"] == 5.0 and tubes(doc)["design_plate_H1"]["inner_radius_mm"] == 2.0


def test_the_ring_leaving_the_part_is_a_note_with_the_volume(tmp_path):
    write_stl(plate(holes=((3, 10, 2.0), (30, 10, 2.0))), tmp_path / "e.stl")             # H1 at x = 3: the ring (outer radius 4) reaches x = -1
    spec = spec_of([{"stl": {"path": "e.stl", "screw_holes": ["H1"]}}])
    r, lines, doc = build(tmp_path, spec)
    note = [ln for ln in lines if ln.startswith("note: regions.design[0].stl: the ring of H1")]
    assert len(note) == 1, lines
    segment = 16 * math.acos(0.75) - 3 * math.sqrt(7)                                    # the circle of radius 4 beyond the line x = 0, 3 mm from its centre
    got = float(re.search(r"has ([0-9.e+-]+) mm3 outside the part", note[0]).group(1))
    assert got == pytest.approx(segment * 4.0, rel=0.02), note
    assert "adds this material outside your part" in note[0]


def test_a_ring_inside_the_part_gives_no_note_and_no_warning_for_any_default(parts):
    for voxel in (0.5, 1.0, 2.0):
        r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": "all"}}], voxel=voxel))
        assert not [ln for ln in lines if "the ring of" in ln or ln.startswith("WARNING: regions.design")], (voxel, lines)


def test_a_ring_in_a_keep_out_region_is_a_warning_with_the_volume(parts):
    keep = {"clear": [{"box": {"min": [24, 4, -1], "max": [36, 16, 5]}}]}               # holds the whole ring of H1
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"]}}], keep_out=keep))
    warn = [ln for ln in lines if ln.startswith("WARNING: regions.design[0].stl: the ring of H1")]
    assert len(warn) == 1, lines
    assert "overlaps the keep_out region 'clear'" in warn[0] and "The keep-out region wins (CONTRACTS.md section 23)" in warn[0]
    got = float(re.search(r"by ([0-9.e+-]+) mm3", warn[0]).group(1))
    assert got == pytest.approx(annulus(2.0, 4.0, 4.0), rel=0.01)


# --- the default rule and the voxel floor -----------------------------------------------------------------------------------------------------


def test_the_two_numbers_of_the_default_rule():
    assert rb().RING_PER_DIAMETER == 0.5 and rb().RING_MIN_VOXELS == 2


def test_ring_thickness_is_half_the_diameter_raised_to_two_voxels():
    h = {"diameter_mm": 6.0}
    assert rb().ring_thickness(h, None, None) == (3.0, "default 0.5 x d")                 # the voxel size is not known yet
    assert rb().ring_thickness(h, None, 1.0) == (3.0, "default 0.5 x d")
    assert rb().ring_thickness(h, None, 1.5) == (3.0, "default 0.5 x d")                  # exactly 2 voxels
    assert rb().ring_thickness(h, None, 2.0) == (4.0, "raised to 2 voxels")
    assert rb().ring_thickness(h, 0.4, 2.0) == (0.4, "ring_mm from the spec")             # the key is used as it is, also under the floor


def test_the_voxel_size_of_the_written_yaml_decides_the_floor_and_a_fixed_voxel_is_used_as_given(parts):
    for voxel, t in ((0.5, 2.0), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0)):                   # H1: d = 4 mm
        r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"]}}], voxel=voxel))
        e = tubes(doc)["design_plate_H1"]
        assert e["radius_mm"] - e["inner_radius_mm"] == pytest.approx(t, abs=1e-9), (voxel, e)
        assert doc["settings"]["voxel_size"] == voxel


def test_with_an_auto_voxel_the_floor_uses_the_voxel_of_the_yaml(parts):
    spec = spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"]}}])
    spec["settings"]["voxel_size"] = "auto"
    r, lines, doc = build(parts, spec)
    h = doc["settings"]["voxel_size"]
    e = tubes(doc)["design_plate_H1"]
    assert e["radius_mm"] - e["inner_radius_mm"] == pytest.approx(max(2.0, 2 * h), abs=1e-9)


# --- ring_mm: one group (the key is a method setting; the tests of the key are only in this group) -------------------------------------------


def test_ring_mm_as_a_number_sets_the_thickness_of_every_screw_hole(parts):
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1", "H2"], "ring_mm": 1.2}}], voxel=0.5))
    for hid, rad in (("H1", 2.0), ("H2", 3.0)):
        assert tubes(doc)[f"design_plate_{hid}"]["radius_mm"] == pytest.approx(rad + 1.2)
    assert lines[1].endswith("t 1.2 mm (ring_mm from the spec); tube interface 'design_plate_H1'"), lines[1]
    assert lines[2].endswith("t 1.2 mm (ring_mm from the spec); tube interface 'design_plate_H2'"), lines[2]


def test_ring_mm_as_a_mapping_sets_one_hole_and_the_others_keep_the_default(parts):
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1", "H2"], "ring_mm": {"H1": 2.5}}}]))
    assert tubes(doc)["design_plate_H1"]["radius_mm"] == pytest.approx(4.5)
    assert tubes(doc)["design_plate_H2"]["radius_mm"] == pytest.approx(6.0)
    assert "t 2.5 mm (ring_mm from the spec)" in lines[1] and "t 3 mm (default 0.5 x d)" in lines[2]


def test_ring_mm_under_two_voxels_is_used_and_gets_a_warning(parts):
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"], "ring_mm": 0.6}}]))
    assert tubes(doc)["design_plate_H1"]["radius_mm"] == pytest.approx(2.6)
    warn = [ln for ln in lines if ln.startswith("WARNING")]
    assert len(warn) == 1 and warn[0].startswith("WARNING: regions.design[0].stl.ring_mm: H1: t 0.6 mm is under 2 voxels (the voxel is 1 mm)"), warn


@pytest.mark.parametrize("value, path", [(0, "ring_mm"), (-1.0, "ring_mm"), (True, "ring_mm"), ("thick", "ring_mm"), ([2], "ring_mm"), ({}, "ring_mm"),
                                         ({"H1": 0}, "ring_mm.H1"), ({"H1": "x"}, "ring_mm.H1"), ({"H1": None}, "ring_mm.H1")])
def test_ring_mm_of_the_wrong_shape_is_a_spec_error_with_the_key_path(parts, value, path):
    with pytest.raises(rb().SpecError, match=rf"^regions\.design\[0\]\.stl\.{re.escape(path)} must be a "):
        rb().check_spec_shape(spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"], "ring_mm": value}}]))


def test_ring_mm_needs_screw_holes_and_names_only_screw_holes(parts):
    with pytest.raises(rb().SpecError, match=r"^regions\.design\[0\]\.stl\.ring_mm is set, but the item has no screw_holes"):
        rb().check_spec_shape(spec_of([{"stl": {"path": "plate.stl", "ring_mm": 2}}]))
    with pytest.raises(rb().SpecError, match=r"^regions\.design\[0\]\.stl\.ring_mm\.H2: H2 is not in screw_holes \(H1\)"):
        rb().check_spec_shape(spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"], "ring_mm": {"H2": 2}}}]))
    text = err(parts, [{"stl": {"path": "plate.stl", "screw_holes": "all", "ring_mm": {"H7": 2}}}])
    assert text.startswith("regions.design[0].stl.ring_mm.H7: H7 is not a screw hole of this item"), text


# --- the screw_ring item: the same ring as a fixed, load or solid region ------------------------------------------------------------------------


def test_the_ring_item_is_the_ring_of_the_hole_in_a_fixed_region(parts):
    r, lines, doc = build(parts, ring_spec(voxel=0.5))
    assert r["ok"], lines
    wall = parts / "out" / "wall.stl"
    assert volume_of(wall) == pytest.approx(annulus(3.0, 6.0, 4.0), rel=0.01)           # H2: r 3, t = 0.5 x d = 3
    assert volume_of(parts / "out" / "pull.stl") == pytest.approx(annulus(2.0, 4.0, 4.0), rel=0.01)
    m = trimesh.load(str(wall), force="mesh")
    assert np.allclose(m.bounds, [[4.0, 4.0, 0.0], [16.0, 16.0, 4.0]], atol=0.01)
    assert doc["regions"]["fixed"] == ["wall.stl"] and doc["regions"]["load"] == {"pull": "pull.stl"}
    assert lines[3] == "screw ring regions.fixed.wall[0].screw_ring: H2 of plate.stl (d 6 mm), t 3 mm (default 0.5 x d)", lines[3]
    assert lines[4] == "screw ring regions.load.pull[0].screw_ring: H1 of plate.stl (d 4 mm), t 2 mm (default 0.5 x d)", lines[4]


def test_the_ring_item_and_the_tube_interface_hold_the_same_voxels(parts):
    from topopt.cli import check_data
    r, lines, doc = build(parts, ring_spec(voxel=0.5))
    data = check_data(r["yaml"])
    by = {f["name"]: f for f in data["interfaces"]}
    assert data["voxel_counts"]["fixed"] == by["design_plate_H2"]["voxels"] > 0           # the fixed ring is the ring of the interface of H2
    assert data["voxel_counts"]["load"] == by["design_plate_H1"]["voxels"] > 0
    assert by["design_plate_H2"]["held"] == 0 and by["design_plate_H1"]["held"] == by["design_plate_H1"]["voxels"]      # the fixed label wins on its own voxels
    assert data["voxel_counts"]["keep_out"] == by["design_plate_H1 (hole)"]["voxels"] + by["design_plate_H2 (hole)"]["voxels"]


def test_the_ring_item_has_its_own_ring_mm_and_a_hole_that_is_not_a_hole_is_an_error(parts):
    spec = spec_of([{"stl": {"path": "plate.stl"}}], fixed={"wall": [{"screw_ring": {"path": "plate.stl", "hole": "H2", "ring_mm": 1.5}}]})
    r, lines, doc = build(parts, spec)
    assert volume_of(parts / "out" / "wall.stl") == pytest.approx(annulus(3.0, 4.5, 4.0), rel=0.01)
    assert lines[1] == "screw ring regions.fixed.wall[0].screw_ring: H2 of plate.stl (d 6 mm), t 1.5 mm (ring_mm from the spec)", lines[1]
    assert "interfaces" not in doc                                                       # a ring item alone writes no tube interface
    text = err(parts, [{"stl": {"path": "plate.stl"}}], fixed={"wall": [{"screw_ring": {"path": "plate.stl", "hole": "H9"}}]})
    assert text == ("regions.fixed.wall[0].screw_ring.hole: 'H9' is not a hole of plate.stl. Known ids: H1 (d 4 mm, 4 mm long, through), "
                    "H2 (d 6 mm, 4 mm long, through)")


@pytest.mark.parametrize("item, expect", [
    ({"path": "plate.stl"}, r"^regions\.fixed\.wall\[0\]\.screw_ring\.hole is missing\. Example: "),
    ({"hole": "H1"}, r"^regions\.fixed\.wall\[0\]\.screw_ring\.path is missing\. Example: "),
    ({"path": "plate.stl", "hole": "H1", "plug_holes": []}, r"^regions\.fixed\.wall\[0\]\.screw_ring\.plug_holes is not a key of a screw_ring item\. Keys: path, hole, ring_mm"),
    ({"path": "plate.stl", "hole": ["H1"]}, r"^regions\.fixed\.wall\[0\]\.screw_ring\.hole must be one hole id from find_features"),
    ({"path": 5, "hole": "H1"}, r"^regions\.fixed\.wall\[0\]\.screw_ring\.path must be a file name"),
    ({"path": "plate.stl", "hole": "H1", "ring_mm": -2}, r"^regions\.fixed\.wall\[0\]\.screw_ring\.ring_mm must be a number in mm above 0"),
    ("H1", r"^regions\.fixed\.wall\[0\]\.screw_ring must be a mapping with a path and a hole"),
])
def test_a_screw_ring_item_of_the_wrong_shape_is_a_spec_error_with_the_key_path(parts, item, expect):
    with pytest.raises(rb().SpecError, match=expect):
        rb().check_spec_shape(spec_of([{"stl": {"path": "plate.stl"}}], fixed={"wall": [{"screw_ring": item}]}))


def test_a_screw_ring_is_for_fixed_load_and_solid_only(parts):
    ring = {"screw_ring": {"path": "plate.stl", "hole": "H1"}}
    box = {"box": {"min": [0, 0, 0], "max": [40, 20, 4]}}
    assert err(parts, [ring]).startswith("regions.design[0].screw_ring: a screw ring is a frozen shape, and this item is in design")
    assert err(parts, [box], keep_out={"k": [ring]}).startswith("regions.keep_out.k[0].screw_ring: a screw ring is a frozen shape, and this item is in keep_out")
    assert err(parts, [box, {"subtract": ring}]).startswith("regions.design[1].subtract.screw_ring: a screw ring is a frozen shape, and this item is inside subtract")
    shapes = rb().build_all(spec_of([box], solid={"s": [ring]}), parts)
    assert shapes["s"][0] == "solid"


def test_a_screw_ring_of_a_notch_or_a_cavity_is_a_spec_error(tmp_path):
    write_stl(Manifold.cube((30, 20, 4)) - cyl(4.0, 6, 64, (15, 1.0, -1)), tmp_path / "n.stl")
    text = err(tmp_path, [{"box": {"min": [0, 0, 0], "max": [30, 20, 4]}}], fixed={"w": [{"screw_ring": {"path": "n.stl", "hole": "H1"}}]})
    assert text.startswith("regions.fixed.w[0].screw_ring.hole: H1 is an arc of ") and "not a closed ring" in text


def test_the_ring_item_is_built_again_with_the_floor_of_the_voxel_size(parts):
    r, lines, doc = build(parts, ring_spec(voxel=1.5))
    assert volume_of(parts / "out" / "pull.stl") == pytest.approx(annulus(2.0, 5.0, 4.0), rel=0.01)       # H1: 0.5 x d = 2 mm is raised to 3 mm = 2 voxels
    assert tubes(doc)["design_plate_H1"]["radius_mm"] == 5.0                              # the interface has the same ring
    ring_lines = [ln for ln in lines if ln.startswith("screw ring regions.load")]
    assert ring_lines == ["screw ring regions.load.pull[0].screw_ring: H1 of plate.stl (d 4 mm), t 3 mm (raised to 2 voxels; 0.5 x d is 2 mm and the voxel is 1.5 mm)"]


def test_a_retry_with_a_bigger_voxel_builds_the_ring_item_again(parts, monkeypatch):
    calls = []
    real = rb().check_yaml

    def check(path, say=print, allow_large_grid=False):
        calls.append(path)
        if len(calls) == 1:
            return False, "time"                                                          # the first check fails for time: the auto voxel grows by 8 percent
        return real(path, say, allow_large_grid)

    monkeypatch.setattr(rb(), "check_yaml", check)
    monkeypatch.setattr(rb(), "auto_voxel", lambda *a, **k: 1.0)
    spec = ring_spec()
    spec["settings"]["voxel_size"] = "auto"
    r, lines, doc = build(parts, spec)
    assert r["retries"] == 1 and r["voxel_size"] == 1.08
    pull = [ln for ln in lines if "regions.load.pull[0].screw_ring" in ln]
    assert len(pull) == 2 and "t 2 mm (default 0.5 x d)" in pull[0] and "t 2.16 mm (raised to 2 voxels" in pull[1], pull
    assert volume_of(parts / "out" / "pull.stl") == pytest.approx(annulus(2.0, 4.16, 4.0), rel=0.01)
    assert tubes(doc)["design_plate_H1"]["radius_mm"] == pytest.approx(4.16)


def test_two_thicknesses_for_one_hole_are_a_warning(parts):
    spec = spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H2"], "ring_mm": 1.5}}], fixed={"wall": [{"screw_ring": {"path": "plate.stl", "hole": "H2", "ring_mm": 2.5}}]})
    r, lines, doc = build(parts, spec)
    warn = [ln for ln in lines if "two thicknesses" in ln]
    assert len(warn) == 1 and warn[0].startswith("WARNING: the ring of H2 of plate.stl has two thicknesses: 1.5 mm at regions.design[0].stl, 2.5 mm at regions.fixed.wall[0].screw_ring"), warn


# --- the voxel labels: a frozen ring round each hole, and the hole stays free (the card) -----------------------------------------------------------


def labels_of(r):
    from topopt.config import load_config
    from topopt.problem import build_problem
    prob = build_problem(load_config(r["yaml"]))
    ix, iy, iz = np.indices(prob.shape)
    centres = [prob.origin[k] + (g + 0.5) * float(prob.h) for k, g in enumerate((ix, iy, iz))]
    return prob, centres


def test_the_labels_hold_a_frozen_ring_round_each_hole_and_the_hole_is_keep_out(parts):
    from topopt.voxelize import DESIGN, FIXED, KEEP_OUT, LOAD
    r, lines, doc = build(parts, ring_spec(voxel=0.5))
    prob, (cx, cy, cz) = labels_of(r)
    h = float(prob.h)
    for hid, (x, y), r_in, t in (("H1", (30.0, 10.0), 2.0, 2.0), ("H2", (10.0, 10.0), 3.0, 3.0)):
        rho = np.hypot(cx - x, cy - y)
        ring = (rho >= r_in + h) & (rho <= r_in + t - h)                                  # inside the ring by one voxel
        hole = rho <= r_in - h
        assert ring.sum() > 100 and hole.sum() > 20, hid
        assert set(np.unique(prob.labels[ring])) <= {LOAD, FIXED}, hid                     # frozen: no design voxel in the ring
        assert (prob.labels[hole] == KEEP_OUT).all(), hid                                  # the hole is free of material
        outside = (rho > r_in + t + 2 * h) & (rho < r_in + t + 4 * h)
        assert (prob.labels[outside] == DESIGN).any(), hid                                 # the design space starts behind the ring


def test_without_the_ring_the_hole_has_no_label_and_the_ring_is_design_space(parts):
    from topopt.voxelize import DESIGN, KEEP_OUT
    r, lines, doc = build(parts, spec_of([{"stl": {"path": "plate.stl"}}], voxel=0.5))
    prob, (cx, cy, cz) = labels_of(r)
    rho = np.hypot(cx - 30.0, cy - 10.0)
    ring = (rho >= 2.5) & (rho <= 3.5)
    assert (prob.labels[ring] == DESIGN).all() and not (prob.labels == KEEP_OUT).any()


def test_a_short_run_on_the_plate_keeps_the_holes_exact_in_final_stl(parts):
    """The acceptance of the card: a run keeps a frozen ring round each screw hole, and `final.stl` shows the holes (radius against the YAML)."""
    from topopt.cli import main
    spec = ring_spec(voxel=1.0)
    spec["settings"]["iterations"] = 12
    r, lines, doc = build(parts, spec)
    assert r["ok"], lines
    out = parts / "run"
    assert main(["run", str(r["yaml"]), "--out", str(out), "--quiet"]) == 0
    final = out / "result" / "final.stl"
    final_holes = holes_of(final)
    for rad, centre in ((2.0, [30.0, 10.0]), (3.0, [10.0, 10.0])):
        near = [h for h in final_holes if h["point"][:2] == pytest.approx(centre, abs=0.01)]
        assert len(near) == 1 and near[0]["depth"] == "through", (centre, final_holes)      # the hole is in final.stl, through the plate
        assert near[0]["radius_mm"] == pytest.approx(rad, abs=0.002)                        # the radius of the YAML (section 23: the sag of the facets)
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    measured = [m for item in metrics["interfaces"]["items"] for m in item["as_printed"]["measured"].values()]
    assert measured and all(m["within_tolerance"] for m in measured)
    assert metrics["interfaces"]["applied"] is True and metrics["watertight"] is True


# --- the tool and its texts --------------------------------------------------------------------------------------------------------------------


def test_the_tool_reply_has_the_report_lines_the_check_and_the_warnings(parts):
    from topopt.regions_builder import build_in_workspace
    spec = spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"], "ring_mm": 0.6}}], voxel=0.5)
    res = build_in_workspace(spec, parts)
    assert res["ok"], res
    assert any(ln == "screw ring regions.design[0].stl: H1 (d 4 mm), t 0.6 mm (ring_mm from the spec); tube interface 'design_plate_H1'" for ln in res["report"])
    assert any("is under 2 voxels" in w for w in res["builder_warnings"]) and any("is under 2 voxels" in w for w in res["warnings"])
    assert [f["name"] for f in res["check"]["interfaces"]] == ["design_plate_H1", "design_plate_H1 (hole)"]
    bad = build_in_workspace(spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H9"]}}]), parts)
    assert bad["ok"] is False and bad["error"].startswith("regions.design[0].stl.screw_holes: 'H9' is not a hole of plate.stl")
    worse = build_in_workspace(spec_of([{"stl": {"path": "plate.stl", "screw_holes": ["H1"], "ring_mm": -1}}]), parts)
    assert worse["ok"] is False and worse["error"].startswith("regions.design[0].stl.ring_mm must be a number in mm above 0")


def tool_texts():
    from topopt_mcp.server import mcp
    return {t.name: t.description for t in asyncio.run(mcp.list_tools())}


def test_the_texts_name_the_keys_and_stay_within_their_bounds():
    texts = tool_texts()
    assert len(texts["build_regions"]) <= 4096 and len(texts["find_features"]) <= 2048
    assert "stl and screw_ring (see find_features)" in texts["build_regions"]
    for word in ("screw_holes", "ring_mm", "screw_ring", "0.5 x d", "at least 2 voxels"):
        assert word in texts["find_features"], word
    for name in ("build_regions", "find_features"):
        assert not HABIT.search(texts[name]), HABIT.search(texts[name])


def test_the_mock_of_the_tools_has_the_new_texts():
    mock = {t["name"]: t["description"] for t in json.loads((REPO / "evals" / "mocks" / "topopt" / "_tools.json").read_text())["tools"]}
    texts = tool_texts()
    for name in ("find_features", "build_regions"):          # no indent: Python 3.13 strips it from a docstring, 3.12 does not
        assert inspect.cleandoc(mock[name]) == inspect.cleandoc(texts[name]), name
    assert "screw_holes" in mock["find_features"]


# --- the skill route "my own STL", and the eval case that sends an STL request through it ------------------------------------------------------

SKILL_FILES = (".claude/skills/part-to-topopt/SKILL.md", ".agents/skills/part-to-topopt/SKILL.md")


def skill_text():
    return (REPO / SKILL_FILES[0]).read_text(encoding="utf-8")


def test_the_skill_has_two_entries_and_the_two_copies_are_equal():
    a, b = ((REPO / f).read_text(encoding="utf-8") for f in SKILL_FILES)
    assert a == b
    head = a.split("---", 2)[1]
    assert "gives an STL of their own part" in head and "Do not use for a part that already has STL files" not in head
    assert "Do not use when the user already has the region STL files and the problem YAML" in head
    assert "## Two entries" in a and "## Your own STL" in a
    start = (REPO / "skills/start-here/SKILL.md").read_text(encoding="utf-8")
    assert start == (REPO / ".agents/skills/start-here/SKILL.md").read_text(encoding="utf-8")
    assert 'section "Your own STL"' in start and "`find_features`" in start


def test_the_skill_names_each_key_of_the_route_and_the_new_rows():
    text = skill_text()
    route = text.split("## Your own STL", 1)[1].split("\n## ", 1)[0]
    for word in ("find_features", "screw_holes", "plug_holes", "ring_mm", "screw_ring", "mesh.size_mm", "metres or inches", "0.5 x the hole diameter",
                 "at least 2 voxels", "method setting", "keep-out region wins"):
        assert word in route, word
    assert "| `stl` |" in text and "| `screw_ring` |" in text
    assert "(for example `box`, `stl` or `screw_ring`)" in text
    assert "The helper does not write `interfaces`" not in text
    assert "ring_mm: ... under 2 voxels" in text and "a screw ring is a frozen shape" in text


def test_the_example_spec_of_the_skill_route_builds(tmp_path):
    """The YAML block of the route is a valid spec: it builds on a bracket with four holes (ids H1 to H4 by radius)."""
    route = skill_text().split("## Your own STL", 1)[1].split("\n## ", 1)[0]
    spec = yaml.safe_load(re.search(r"```yaml\n(.*?)```", route, re.S).group(1))
    part = Manifold.cube((60, 30, 10))
    for x, y, r in ((8, 8, 1.5), (8, 22, 2.0), (52, 15, 2.5), (30, 15, 3.0)):
        part = part - cyl(r, 12, 32, (x, y, -1))
    write_stl(part, tmp_path / "bracket.stl")
    lines = []
    r = rb().build_spec(spec, tmp_path / "out", say=said(lines), base=tmp_path)
    assert r["ok"], lines
    names = [e["name"] for e in yaml.safe_load(Path(r["yaml"]).read_text())["interfaces"]]
    assert names == ["design_bracket_H2", "design_bracket_H3"]
    assert any("plugged H4" in ln for ln in lines) and sum(ln.startswith("screw ring") for ln in lines) == 4      # two interfaces and two ring items


EVAL = REPO / "evals" / "stl-own-part"


def test_the_eval_case_sends_an_stl_request_through_the_new_route():
    prompt = (EVAL / "prompt.md").read_text(encoding="utf-8")
    head, body = prompt.split("---\n", 2)[1], prompt.split("---\n", 2)[2]
    front = yaml.safe_load(head)
    assert front["tags"] == ["mock", "flow", "files"] and "Skill" in front["allowed_tools"]
    assert "files/bracket.stl" in body and "30 N along +x" in body and "Do not run" in body
    case = yaml.safe_load((EVAL / "case.yaml").read_text(encoding="utf-8"))
    assert case["name"] == "stl-own-part" and case["context"]["scaffold_script"] == "stage-files.sh"
    assert (EVAL / "stage-files.sh").is_file() and (EVAL / "files" / "bracket.stl").is_file()
    tools = {t["name"] for t in json.loads((REPO / "evals/mocks/topopt/_tools.json").read_text())["tools"]}
    graders = {f.name: f.read_text(encoding="utf-8") for f in (EVAL / "graders").glob("*.md")}
    assert set(graders) == {"find-before-build.md", "no-run.md", "spec-stl.md", "result.md"}
    for text in graders.values():
        for tool in re.findall(r"mcp__plugin_topopt_topopt__(\w+)", text):
            assert tool in tools, tool
    assert "before: mcp__plugin_topopt_topopt__find_features" in graders["find-before-build.md"]
    assert "after: mcp__plugin_topopt_topopt__build_regions" in graders["find-before-build.md"]


def test_the_mock_of_find_features_of_the_eval_case_is_the_real_answer_for_its_file():
    from topopt.features import find_features
    real = find_features(EVAL / "files" / "bracket.stl")
    mock = json.loads((EVAL / "mocks" / "topopt" / "fixtures" / "find_features.json").read_text())
    assert [(h["id"], h["diameter_mm"], h["point"], h["axis"], h["length_mm"], h["depth"]) for h in mock["holes"]] == [
        (h["id"], h["diameter_mm"], h["point"], h["axis"], h["length_mm"], h["depth"]) for h in real["holes"]]
    assert mock["mesh"]["size_mm"] == real["mesh"]["size_mm"] == [40.0, 20.0, 4.0] and mock["ok"] is True
    assert [h["diameter_mm"] for h in mock["holes"]] == [4.0, 6.0]                                  # the small hole is H1, as the prompt and the graders say
    assert "/Users/me/" in mock["mesh"]["source"]                                                      # a placeholder path, not the path of this machine
    assert str(REPO) not in json.dumps(mock)
    assert "type: fixed" in (EVAL / "mocks/topopt/find_features.md").read_text()
