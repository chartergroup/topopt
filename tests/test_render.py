"""Tests for the render tool (CONTRACTS.md section 15): library, CLI and MCP tool, on synthetic meshes."""
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import render as R  # noqa: E402
from topopt.cli import main as cli_main  # noqa: E402
from topopt.config import UserError  # noqa: E402


def box(path, extents=(20, 10, 10), center=(0, 0, 0), flip=False, open_top=False):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    if flip:
        m.invert()
    if open_top:                      # remove the two faces with the largest z: an open surface
        keep = np.ones(len(m.faces), bool)
        zc = m.triangles_center[:, 2]
        keep[zc > zc.max() - 1e-6] = False
        m = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    m.export(path)
    return path


def pixels(png):
    return np.asarray(Image.open(png).convert("RGB"), float)


def count_color(png, rgb, tol=8):
    """Pixels with the color of `rgb` at any brightness (shading changes the brightness only)."""
    p = pixels(png).reshape(-1, 3)
    c = np.array(rgb, float)
    mx = p.max(1)
    ok = mx >= 40
    scaled = p[ok] * (c.max() / mx[ok])[:, None]
    return int((np.abs(scaled - c).max(1) < tol).sum())


def view_png(facts, name="iso"):
    return next(v["path"] for v in facts["views"] if v["name"] == name)


def test_png_sizes_and_facts(tmp_path):
    a = box(tmp_path / "a.stl")
    f = R.render(str(a))
    assert [v["name"] for v in f["views"]] == ["iso", "front", "right", "top"]
    assert f["out_dir"] == str(tmp_path / "renders")
    m = f["montage"]
    w, h = Image.open(m["path"]).size
    assert (w, h) == (m["width"], m["height"]) and max(w, h) <= 1568
    assert m["bytes"] == Path(m["path"]).stat().st_size
    for v in f["views"]:
        assert Image.open(v["path"]).size == (v["width"], v["height"]) == (640, 480)
    mesh = f["meshes"][0]
    assert mesh["role"] == "part" and mesh["triangles"] == 12 and mesh["watertight"] is True
    assert mesh["bbox_mm"]["size"] == [20.0, 10.0, 10.0]
    assert f["back_face_share"] == 0.0 and f["missing_files"] == [] and f["section"] is None
    assert f["font"] and f["elapsed_s"] >= 0
    json.dumps(f)                      # the facts are JSON


def test_big_size_is_scaled_to_the_long_edge_limit(tmp_path):
    f = R.render(str(box(tmp_path / "a.stl")), size="1000x800")
    assert max(f["montage"]["width"], f["montage"]["height"]) <= 1568
    assert f["tile_size"][0] < 1000


def test_default_folder_and_out_dir(tmp_path):
    a = box(tmp_path / "a.stl")
    before = a.read_bytes()
    f = R.render(str(a), views="front", out_dir=str(tmp_path / "x" / "y"))
    assert Path(f["montage"]["path"]).parent == tmp_path / "x" / "y"
    assert a.read_bytes() == before and not (tmp_path / "renders").exists()


@pytest.mark.parametrize("role", R.ROLES)
def test_each_role_color_appears(tmp_path, role):
    p = box(tmp_path / f"{role}.stl")
    f = R.render([(str(p), role)], views=["front"])
    assert count_color(view_png(f, "front"), R.ROLE_COLORS[role]) > 500


def test_design_space_is_a_ghost_and_the_part_stays_visible(tmp_path):
    design = box(tmp_path / "design.stl", extents=(60, 40, 40))
    part = box(tmp_path / "part.stl", extents=(20, 10, 10))
    keep = box(tmp_path / "keep.stl", extents=(10, 10, 60), center=(0, 0, 0))
    f = R.render([(str(part), "part"), (str(design), "design"), (str(keep), "keep-out")], views=["front", "iso"])
    drawn = {m["role"]: m["drawn"] for m in f["meshes"]}
    assert drawn == {"part": "solid", "design": "ghost", "keep-out": "ghost"}
    for name in ("front", "iso"):
        assert count_color(view_png(f, name), R.ROLE_COLORS["part"]) > 300
    # a design space alone is drawn solid
    alone = R.render([(str(design), "design")], views=["front"], out_dir=str(tmp_path / "alone"))
    assert alone["meshes"][0]["drawn"] == "solid"


def test_flipped_box_gives_magenta_and_a_fact(tmp_path):
    good = R.render(str(box(tmp_path / "g.stl")), out_dir=str(tmp_path / "g"))
    bad = R.render(str(box(tmp_path / "b.stl", flip=True)), out_dir=str(tmp_path / "b"))
    assert count_color(view_png(good), R.BACK_COLOR) == 0 and good["back_face_share"] == 0.0
    assert count_color(view_png(bad), R.BACK_COLOR) > 500
    assert bad["back_face_share"] > 0.9
    assert bad["meshes"][0]["inside_out"] is True and good["meshes"][0]["inside_out"] is False


def test_section_gives_orange_cut_faces(tmp_path):
    f = R.render(str(box(tmp_path / "a.stl")), section="y=0", views=["iso", "front"])
    assert {k: f["section"][k] for k in ("axis", "value", "keeps", "applied")} == {
        "axis": "y", "value": 0.0, "keeps": "y >= 0", "applied": True}
    assert count_color(view_png(f, "iso"), R.CUT_COLOR) > 300
    assert f["meshes"][0]["section"]["applied"] is True
    assert "section-y0" in f["montage"]["path"]            # a section does not overwrite the whole-part render
    # the bbox fact is the mesh as loaded, not the cut mesh
    assert f["meshes"][0]["bbox_mm"]["size"] == [20.0, 10.0, 10.0]


def test_section_that_cannot_run_is_a_fact_and_the_render_completes(tmp_path):
    f = R.render(str(box(tmp_path / "open.stl", open_top=True)), section="x=0")
    assert f["meshes"][0]["watertight"] is False
    assert f["section"]["applied"] is False
    assert "not a valid closed solid" in f["meshes"][0]["section"]["note"]
    assert f["notes"] and Path(f["montage"]["path"]).is_file()


def test_same_inputs_give_the_same_png_bytes(tmp_path):
    a = box(tmp_path / "a.stl")
    f1 = R.render(str(a), out_dir=str(tmp_path / "r1"), section="y=0")
    f2 = R.render(str(a), out_dir=str(tmp_path / "r2"), section="y=0")
    for p1, p2 in zip([f1["montage"]] + f1["views"], [f2["montage"]] + f2["views"]):
        assert Path(p1["path"]).read_bytes() == Path(p2["path"]).read_bytes()


def make_run(tmp_path, with_fixed=True):
    """A run folder: result/final.stl; run.json points to a YAML folder that holds the region STLs."""
    spec = tmp_path / "spec"
    spec.mkdir()
    box(spec / "design.stl", extents=(60, 40, 40))
    if with_fixed:
        box(spec / "plate.stl", extents=(4, 40, 40), center=(-30, 0, 0))
    box(spec / "tip.stl", extents=(6, 6, 6), center=(25, 0, 0))
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", extents=(30, 10, 10))
    (run / "run.json").write_text(json.dumps({
        "yaml_path": str(spec / "p.yaml"),
        "regions": {"design": "design.stl", "fixed": ["plate.stl"], "keep_out": ["hole.stl"],
                    "load": {"tip": "tip.stl"}}}))
    return run


def test_run_folder_with_a_missing_region_file(tmp_path):
    run = make_run(tmp_path, with_fixed=False)
    f = R.render(str(run))
    assert f["input"]["kind"] == "run" and f["out_dir"] == str(run / "renders")
    assert [m["role"] for m in f["meshes"]] == ["part", "design", "load"]
    assert {(m["role"], m["file"]) for m in f["missing_files"]} == {("fixed", "plate.stl"), ("keep-out", "hole.stl")}
    assert Path(f["montage"]["path"]).name == "final_montage.png"
    assert count_color(view_png(f, "front"), R.ROLE_COLORS["part"]) > 300
    p = pixels(view_png(f, "front"))      # a strong ghost: a red tint, so test "reddish"
    assert int(((p[..., 0] > 1.15 * p[..., 1]) & (p[..., 0] > 1.15 * p[..., 2])).sum()) > 300


def test_run_folder_without_run_json_draws_the_part_alone(tmp_path):
    run = make_run(tmp_path)
    (run / "run.json").unlink()
    f = R.render(str(run))
    assert [m["role"] for m in f["meshes"]] == ["part"] and f["notes"]


def test_user_errors(tmp_path):
    a = box(tmp_path / "a.stl")
    with pytest.raises(UserError, match="not found"):
        R.render(str(tmp_path / "nope.stl"))
    with pytest.raises(UserError, match="no result/final.stl"):
        R.render(str(tmp_path))
    with pytest.raises(UserError, match="AXIS=VALUE"):
        R.render(str(a), section="w=1")
    with pytest.raises(UserError, match="unknown view"):
        R.render(str(a), views="sideways")
    with pytest.raises(UserError, match="unknown role"):
        R.render([(str(a), "wall")])
    with pytest.raises(UserError, match="WxH"):
        R.render(str(a), size="big")
    with pytest.raises(UserError, match="roles for"):
        R.render([str(a), str(a), str(a)], roles=["part", "fixed"])


def test_custom_view(tmp_path):
    f = R.render(str(box(tmp_path / "a.stl")), views="low:1,1,-0.5;top")
    assert [v["name"] for v in f["views"]] == ["low", "top"]


# --- views text (F4-10): one custom view, or custom views mixed with names, with commas only --------

def test_one_custom_view_with_commas_is_one_view():
    """`--views 'fl:-0.5,-1,0.25'` was split at the commas into 'fl:-0.5', '-1' and '0.25' (P4, F4-10)."""
    from topopt.render_input import parse_views
    assert parse_views("fl:-0.5,-1,0.25") == [("fl", (-0.5, -1.0, 0.25))]


def test_custom_views_mixed_with_names_in_a_comma_list():
    from topopt.render_input import parse_views
    got = parse_views("iso,fl:-0.5,-1,0.25,top,under:1,-1,-0.8")
    assert got == [("iso", (1, -1, 0.8)), ("fl", (-0.5, -1.0, 0.25)), ("top", (0, 0, 1)), ("under", (1.0, -1.0, -0.8))]


def test_semicolon_lists_and_old_forms_give_the_same_views():
    from topopt.render_input import parse_views
    assert parse_views("low:1,1,-0.5;top") == [("low", (1.0, 1.0, -0.5)), ("top", (0, 0, 1))]
    assert parse_views("a:1,0,0;b:0,1,0") == [("a", (1.0, 0.0, 0.0)), ("b", (0.0, 1.0, 0.0))]
    assert parse_views("iso,front") == [("iso", (1, -1, 0.8)), ("front", (0, -1, 0))]
    assert parse_views(["low:1,1,-0.5", "top"]) == [("low", (1.0, 1.0, -0.5)), ("top", (0, 0, 1))]


def test_a_list_item_may_hold_a_comma_list_too():
    """The MCP tool takes a list: an item with commas is read like a CLI text."""
    from topopt.render_input import parse_views
    assert parse_views(["fl:-0.5,-1,0.25"]) == [("fl", (-0.5, -1.0, 0.25))]
    assert parse_views(["iso", "fl:-0.5,-1,0.25"]) == [("iso", (1, -1, 0.8)), ("fl", (-0.5, -1.0, 0.25))]
    assert [n for n, _ in parse_views(["iso,fl:-0.5,-1,0.25"])] == ["iso", "fl"]


def test_a_custom_view_with_too_few_numbers_is_still_a_clear_error():
    from topopt.render_input import parse_views
    for bad in ("fl:-0.5,-1", "fl:1,iso,front", "fl:0,0,0", "fl:a,b,c", "fl:"):
        with pytest.raises(UserError, match="use NAME:dx,dy,dz"):
            parse_views(bad)


def test_render_part_view_items_with_commas(tmp_path):
    """The MCP tool: a list item that holds a custom view alone, or a comma list with names."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    a = box(tmp_path / "a.stl")
    for views, names in ((["fl:-0.5,-1,0.25"], ["fl"]), (["iso,fl:-0.5,-1,0.25"], ["iso", "fl"]),
                         (["front", "fl:-0.5,-1,0.25"], ["front", "fl"])):
        d = json.loads(server.render_part(stl_paths=[str(a)], views=views, size="320x240", inline=False,
                                          out_dir=str(tmp_path / ("o" + str(len(names)) + views[0][:3])))[0].text)
        assert d.get("ok", True) and [v["name"] for v in d["views"]] == names, d


def test_cli_one_custom_view_with_commas(tmp_path, capsys):
    a = box(tmp_path / "a.stl")
    rc = cli_main(["render", str(a), "--views", "fl:-0.5,-1,0.25", "--size", "320x240", "--out", str(tmp_path / "o")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and [v["name"] for v in out["views"]] == ["fl"]
    assert out["views"][0]["direction"] == [-0.5, -1.0, 0.25]


# --- section: kept side, default views, removed meshes ---------------------------------------

AXES = "xyz"


def two_box_scene(tmp_path):
    """A part and a region on each side of the plane x=0 (the part is centered, regions at +-30)."""
    return [(str(box(tmp_path / "part.stl", extents=(20, 10, 10))), "part"),
            (str(box(tmp_path / "wall.stl", extents=(4, 20, 20), center=(-30, 0, 0))), "fixed"),
            (str(box(tmp_path / "tip.stl", extents=(4, 6, 6), center=(30, 0, 0))), "load")]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("op", [">=", "<="])
def test_cut_face_is_visible_for_each_axis_and_side(tmp_path, axis, op):
    f = R.render(two_box_scene(tmp_path), section=f"{axis}{op}0")
    assert [v["name"] for v in f["views"]][:2] == ["cut", "iso-cut"] and len(f["views"]) == 4
    by = {v["name"]: v for v in f["views"]}
    for name in ("cut", "iso-cut"):
        assert by[name]["cut_face_visible_px"] > 300, name
        assert count_color(by[name]["path"], R.CUT_COLOR, tol=3) > 300, name     # flat, full orange
    assert f["section"]["cut_face_visible_px"] == sum(v["cut_face_visible_px"] for v in f["views"])
    assert f["section"]["views_with_cut"] >= 2 and not f["notes"]
    assert f["section"]["keeps"] == f"{axis} {op} 0"


def test_default_section_views_by_axis():
    from topopt.render_input import section_views
    names = lambda s: [n for n, _ in section_views(s)]
    assert names("x=0") == ["cut", "iso-cut", "front", "top"]
    assert names("y=0") == ["cut", "iso-cut", "right", "top"]
    assert names("z=0") == ["cut", "iso-cut", "front", "right"]
    assert dict(section_views("x=1"))["cut"] == (-1.0, 0.0, 0.0)       # looks from the removed side
    assert dict(section_views("x<=1"))["cut"] == (1.0, 0.0, 0.0)


def test_given_views_replace_the_section_defaults_and_a_missing_cut_is_a_note(tmp_path):
    f = R.render(two_box_scene(tmp_path), section="z=0", views=["front", "right"])
    assert f["section"]["cut_face_visible_px"] == 0
    assert any("cut:0,0,-1" in n for n in f["notes"])
    with pytest.raises(UserError, match="needs a section"):
        R.render(two_box_scene(tmp_path), views="cut")
    g = R.render(two_box_scene(tmp_path), section="y=0", views="cut,front", out_dir=str(tmp_path / "g"))
    assert [v["name"] for v in g["views"]] == ["cut", "front"]


def test_a_mesh_on_the_removed_side_is_not_drawn(tmp_path):
    f = R.render(two_box_scene(tmp_path), section="x=0")          # the wall at x=-30 is removed
    by = {m["file"]: m for m in f["meshes"]}
    assert by["wall.stl"]["drawn"] == "removed"
    assert by["wall.stl"]["section"] == {"applied": True, "kept_triangles": 0,
                                         "note": "no material on the kept side: the mesh is not drawn"}
    assert by["tip.stl"]["drawn"] == "ghost" and by["part.stl"]["section"]["kept_triangles"] > 0
    assert count_color(view_png(f, "cut"), R.ROLE_COLORS["fixed"], tol=20) == 0     # it does not cover the cut
    with pytest.raises(UserError, match="leaves no material"):
        R.render([(str(tmp_path / "wall.stl"), "part")], section="x=0")


def test_inside_out_mesh_is_flipped_for_the_section(tmp_path):
    p = box(tmp_path / "io.stl", flip=True)
    f = R.render(str(p), section="y=0")
    assert f["meshes"][0]["inside_out"] is True and f["meshes"][0]["section"]["applied"] is True
    assert "inside out" in f["meshes"][0]["section"]["note"]
    assert f["views"][0]["cut_face_visible_px"] > 300
    # the kept half is a box of 20 x 5 x 10 mm: cut faces fill the cut view
    assert f["section"]["applied"] is True


def test_section_syntax_and_wrong_values():
    from topopt.render_input import parse_section
    assert parse_section("x=30") == parse_section("x>=30") == parse_section(" X >= 30 ") == (0, 30.0, 1)
    assert parse_section("z<=-2.5") == parse_section("z<-2.5") == (2, -2.5, -1)
    for bad in ("x=nan", "y=inf", "y<=-inf", "w=1", "x", "x=", "x==1", "x=1,2"):
        with pytest.raises(UserError, match="section must be"):
            parse_section(bad)


# --- regions, scale, limits, files --------------------------------------------------------------

def test_regions_option(tmp_path):
    scene = two_box_scene(tmp_path)
    g = R.render(scene, views=["front"], out_dir=str(tmp_path / "g"))
    assert g["regions"] == "ghost" and {m["role"]: m["drawn"] for m in g["meshes"]} == {
        "part": "solid", "fixed": "ghost", "load": "ghost"}
    s = R.render(scene, regions="solid", views=["front"], out_dir=str(tmp_path / "s"))
    assert {m["role"]: m["drawn"] for m in s["meshes"]} == {"part": "solid", "fixed": "solid", "load": "solid"}
    o = R.render(scene, regions="off", views=["front"], out_dir=str(tmp_path / "o"))
    assert {m["role"]: m["drawn"] for m in o["meshes"]} == {"part": "solid", "fixed": "off", "load": "off"}
    assert count_color(view_png(o, "front"), R.ROLE_COLORS["fixed"], tol=20) == 0
    assert count_color(view_png(s, "front"), R.ROLE_COLORS["fixed"], tol=3) > 100       # solid: the true color
    nopart = R.render(scene[1:], views=["front"], out_dir=str(tmp_path / "n"))
    assert nopart["regions"] == "solid"
    with pytest.raises(UserError, match="regions must be"):
        R.render(scene, regions="half")
    with pytest.raises(UserError, match="needs a mesh with the role part"):
        R.render(scene[1:], regions="off")


def test_legend_says_how_each_role_is_drawn(tmp_path):
    f = R.render(two_box_scene(tmp_path), views=["front"])
    assert {e["label"]: e["style"] for e in f["legend"]} == {
        "part": "solid", "fixed": "ghost", "load": "ghost", "back face": "solid"}
    s = R.render(two_box_scene(tmp_path), views=["front"], regions="solid", out_dir=str(tmp_path / "s"))
    assert {e["label"]: e["style"] for e in s["legend"]}["fixed"] == "solid"
    assert count_color(s["montage"]["path"], R.ROLE_COLORS["fixed"], tol=3) > 100


def test_one_scale_for_all_tiles_and_a_scale_bar(tmp_path):
    tiny = R.render(str(box(tmp_path / "t.stl", extents=(2, 2, 2))), out_dir=str(tmp_path / "t"))
    big = R.render(str(box(tmp_path / "b.stl", extents=(300, 200, 1.5))), out_dir=str(tmp_path / "b"))
    assert tiny["scale"]["bar_mm"] <= 1 and big["scale"]["bar_mm"] >= 50
    assert tiny["scale"]["px_per_mm"] > 50 * big["scale"]["px_per_mm"]
    # one scale: a thin edge-on view of the flat plate stays thin (not stretched to fill the tile)
    front = pixels(view_png(big, "front")).min(axis=2) < 200
    ys = np.nonzero(front[60:420, 40:600].any(axis=1))[0]       # without the label and the scale bar
    assert ys.max() - ys.min() < 12


def test_max_triangles_and_memory_estimate(tmp_path):
    a = box(tmp_path / "a.stl")
    f = R.render(str(a), views=["front"])
    assert f["triangles_drawn"] == 12 and 440 <= f["memory_estimate_mb"] <= 460
    with pytest.raises(UserError, match="more than 5 triangles"):
        R.render(str(a), max_triangles=5)
    with pytest.raises(UserError, match="max_triangles"):
        R.render(str(a), max_triangles=0)


def test_winding_not_consistent_gives_no_volume(tmp_path):
    m = trimesh.creation.box(extents=(20, 10, 10))
    F = m.faces.copy()
    F[0] = F[0][::-1]                                         # one flipped face: closed, winding not consistent
    trimesh.Trimesh(m.vertices, F, process=False).export(tmp_path / "w.stl")
    f = R.render(str(tmp_path / "w.stl"), views=["front"])
    mesh = f["meshes"][0]
    assert mesh["watertight"] is True and mesh["winding_consistent"] is False
    assert mesh["volume_mm3"] is None and mesh["inside_out"] is None
    assert any("winding is not consistent" in n for n in f["notes"])
    good = R.render(str(box(tmp_path / "g.stl")), views=["front"], out_dir=str(tmp_path / "g"))
    assert good["meshes"][0]["winding_consistent"] is True and good["meshes"][0]["volume_mm3"] == 2000.0


def test_unwritable_out_dir_is_a_user_error(tmp_path):
    a = box(tmp_path / "a.stl")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(UserError, match="cannot make the output folder"):
        R.render(str(a), out_dir=str(blocker / "sub"))


def test_view_names_are_file_name_safe(tmp_path):
    a = box(tmp_path / "a.stl")
    for bad in ("../../x:1,0,0", "a/b:1,0,0", "a b:1,0,0"):
        with pytest.raises(UserError, match="letters, digits"):
            R.render(str(a), views=[bad, "top"])
    with pytest.raises(UserError, match="finite"):
        R.render(str(a), views=["v:nan,0,1"])
    f = R.render(str(a), views=["low-1:1,1,-0.5", "top_2:0,0,1"])
    assert [v["name"] for v in f["views"]] == ["low-1", "top_2"]


# --- fonts ---------------------------------------------------------------------------------

def test_load_font_falls_back_to_the_pillow_default(monkeypatch):
    monkeypatch.delenv("TOPOPT_FONT", raising=False)
    monkeypatch.setattr(R, "FONT_CHAIN", ("/no/such/font.ttf", "/System/Library/Fonts/Missing.ttc"))
    font = R.load_font(18)
    assert R.font_name() == R.PILLOW_DEFAULT
    assert font.getbbox("topopt")[2] > 0


def test_load_font_uses_the_first_file_that_opens(monkeypatch):
    seen = []

    def fake_truetype(path, size):
        seen.append(path)
        if path == "second.ttf":
            return "FONT"
        raise OSError("cannot open resource")

    monkeypatch.delenv("TOPOPT_FONT", raising=False)
    monkeypatch.setattr(R.ImageFont, "truetype", fake_truetype)
    monkeypatch.setattr(R, "FONT_CHAIN", ("first.ttf", "second.ttf", "third.ttf"))
    assert R.load_font(18) == "FONT" and R.font_name() == "second.ttf" and seen == ["first.ttf", "second.ttf"]
    monkeypatch.setenv("TOPOPT_FONT", "mine.ttf")
    R.load_font(18)
    assert seen[2] == "mine.ttf"


def test_render_works_with_no_system_font(tmp_path, monkeypatch):
    monkeypatch.delenv("TOPOPT_FONT", raising=False)
    monkeypatch.setattr(R, "FONT_CHAIN", ())
    f = R.render(str(box(tmp_path / "a.stl")), views=["front"])
    assert f["font"] == R.PILLOW_DEFAULT


def test_labels_are_at_least_16_px():
    assert R.LABEL_PX >= 16


# --- numba cache ---------------------------------------------------------------------------

def test_render_works_with_a_numba_cache_folder_in_a_temp_dir(tmp_path):
    a = box(tmp_path / "a.stl")
    cache = tmp_path / "numba-cache"
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache))
    code = f"from topopt.render import render; print(render({str(a)!r}, views='front')['ok'])"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip() == "True", r.stderr
    assert any(cache.iterdir())


# --- lazy imports ----------------------------------------------------------------------------

HEAVY = ("topopt.render", "numba", "PIL", "manifold3d", "trimesh", "numpy")


def run_py(code):
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_import_cli_and_server_do_not_load_render_or_its_packages():
    out = run_py("import sys, topopt.cli, topopt_mcp.server; "
                 f"print([m for m in {HEAVY!r} if m in sys.modules])")
    assert out == "[]", out


def test_wrong_cli_input_stops_before_the_render_packages_load():
    out = run_py("import sys; from topopt.cli import main; "
                 "rc = main(['render', 'none.stl', '--section', 'q=1']); "
                 f"print(rc, [m for m in {HEAVY!r} if m in sys.modules])")
    assert out == "1 []", out


# --- CLI -----------------------------------------------------------------------------------

def test_cli_prints_facts_json(tmp_path, capsys):
    a = box(tmp_path / "a.stl")
    b = box(tmp_path / "b.stl", center=(30, 0, 0))
    rc = cli_main(["render", str(a), str(b), "--role", "part", "--role", "load", "--views", "iso,top",
                   "--size", "320x240", "--out", str(tmp_path / "o")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and [m["role"] for m in out["meshes"]] == ["part", "load"]
    assert out["tile_size"] == [320, 240] and (tmp_path / "o" / "a_montage.png").is_file()


def test_cli_user_error_exits_1(tmp_path, capsys):
    assert cli_main(["render", str(tmp_path / "none.stl")]) == 1
    assert "error: file not found" in capsys.readouterr().err


def test_cli_run_folder(tmp_path, capsys):
    run = make_run(tmp_path)
    assert cli_main(["render", str(run), "--views", "front", "--size", "320x240"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["meshes"]) == 4 and out["missing_files"][0]["file"] == "hole.stl"


# --- MCP tool ------------------------------------------------------------------------------

def test_render_part_result_types_inline_on_and_off(tmp_path):
    pytest.importorskip("mcp")
    from mcp.types import ImageContent, TextContent
    from topopt_mcp import server
    a = box(tmp_path / "a.stl")
    on = server.render_part(stl_paths=[str(a)], views=["front"], size="320x240")
    assert [type(c) for c in on] == [TextContent, ImageContent]
    assert on[1].mime_type == "image/png"
    facts = json.loads(on[0].text)
    assert base64.b64decode(on[1].data) == Path(facts["montage"]["path"]).read_bytes() and facts["inline"] is True
    off = server.render_part(stl_paths=[str(a)], views=["front"], size="320x240", inline=False)
    assert [type(c) for c in off] == [TextContent] and json.loads(off[0].text)["inline"] is False


def test_render_part_error_is_data(tmp_path):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    r = server.render_part(stl_paths=[str(tmp_path / "none.stl")])
    d = json.loads(r[0].text)
    assert len(r) == 1 and d["ok"] is False and "not found" in d["error"]


def test_render_part_without_a_run_says_so(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    server._current.clear()
    d = json.loads(server.render_part()[0].text)
    assert d == {"ok": False, "error": "no run yet: give run_dir or stl_paths"}


def test_render_part_text_is_compact_json(tmp_path):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    a = box(tmp_path / "a.stl")
    text = server.render_part(stl_paths=[str(a)], views=["front"], size="320x240", inline=False)[0].text
    assert "\n" not in text and ": " not in text and json.loads(text)["ok"] is True


def test_render_part_over_stdio(tmp_path):
    """The real result shape: a client calls the tool through the stdio server."""
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    run = make_run(tmp_path)

    async def call(args):
        params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(REPO),
                                       env=dict(os.environ, TOPOPT_WORKSPACE=str(tmp_path)))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                return [await session.call_tool("render_part", a) for a in args]

    on, off = asyncio.run(asyncio.wait_for(call([
        {"run_dir": "run", "views": ["front", "top"], "size": "320x240"},
        {"run_dir": "run", "views": ["front"], "size": "320x240", "inline": False}]), 240))
    assert [c.type for c in on.content] == ["text", "image"] and on.content[1].mime_type == "image/png"
    assert [c.type for c in off.content] == ["text"]
    facts = json.loads(on.content[0].text)
    assert facts["out_dir"] == str(run / "renders") and facts["montage"]["width"] == 640
    assert facts["out_dir_rel"] == "run/renders" and facts["montage_rel"] == "run/renders/final_montage.png"


def test_tool_texts_have_no_habitual_caps():
    """The freedom rule: facts and options. No `must` or `never` in the render text."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    text = tools["render_part"].description
    assert not re.search(r"\b(must|never|always)\b", text, re.I)


# --- packaging -----------------------------------------------------------------------------

def test_pillow_pin_in_pyproject_matches_the_lock():
    pyproject = (REPO / "pyproject.toml").read_text()
    lock = (REPO / "requirements.lock").read_text()
    pin = re.search(r'"Pillow==([\d.]+)"', pyproject)
    assert pin, "Pillow must be a main dependency, pinned with =="
    assert f"pillow=={pin.group(1)}" in lock
