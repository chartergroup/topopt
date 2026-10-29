"""Force arrows in the render of a problem YAML (P9-16; CONTRACTS.md section 15).

The cantilever of `write_case`: design x 0..40, y 0..10, z 0..10; plate x -2..0; load region `tip` x 36..40, y 3..7, z 0..3
(volume centroid 38, 5, 1.5); load region `side` x 20..24, y 0..2, z 2..8 (centroid 22, 1, 5).
"""
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import render as R  # noqa: E402
from topopt.cli import main as cli_main  # noqa: E402
from topopt.config import UserError  # noqa: E402
from topopt.render_input import check_inputs  # noqa: E402

DOWN = {"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}
PULL = {"name": "pull", "forces": [{"region": "side", "force": [4.0, 0, 0]}]}
TWIST = {"name": "twist", "forces": [{"region": "tip", "moment": [0, 0, 100.0]}]}
PUSH = {"name": "push", "pressures": [{"region": "side", "value_mpa": 0.5}]}
VIEWS = dict(views="front,top", size="480x360")


def box(path, lo, hi, open_top=False):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    if open_top:                                   # drop the two faces with the largest z: an open surface
        keep = m.triangles_center[:, 2] < m.triangles_center[:, 2].max() - 1e-6
        m = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    m.export(path)


def write_case(tmp_path, cases, open_tip=False):
    box(tmp_path / "design.stl", (0, 0, 0), (40, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", (36, 3, 0), (40, 7, 3), open_top=open_tip)
    box(tmp_path / "side.stl", (20, 0, 2), (24, 2, 8))
    d = {"name": "cantilever",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl", "side": "side.stl"}},
         "load_cases": cases,
         "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": 4}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def render(tmp_path, cases, out="r", **kw):
    return R.render(str(write_case(tmp_path, cases, open_tip=kw.pop("open_tip", False))),
                    out_dir=str(tmp_path / out), **{**VIEWS, **kw})


def png(facts, name):
    return next(v["path"] for v in facts["views"] if v["name"] == name)


def mask(path, rgb, tol=10):
    """Pixels with the color `rgb` at any brightness (shading changes the brightness only)."""
    p = np.asarray(Image.open(path).convert("RGB"), float)
    mx = p.max(2)
    scaled = p * (max(rgb) / np.where(mx >= 40, mx, np.inf))[..., None]
    return np.abs(scaled - np.array(rgb, float)).max(2) < tol


def tip_side(m, axis):
    """+1 when the wide part (the head) of the arrow in the mask `m` is at the high index of `axis` (0: rows, 1: columns), else -1.

    The head starts at the base of the cone, near the tip, and its base is the widest part of the arrow.
    """
    width = m.sum(1 - axis)
    idx = np.nonzero(width)[0]
    wide = np.nonzero(width >= 0.8 * width.max())[0]
    return int(np.sign(wide.mean() - (idx.min() + idx.max()) / 2))


def hexcolor(i):
    return "#%02x%02x%02x" % R.CASE_COLORS[i % len(R.CASE_COLORS)]


# --- the arrows ------------------------------------------------------------------------------------

def test_a_force_along_minus_z_gives_an_arrow_that_points_down_and_a_force_along_x_one_that_points_right(tmp_path):
    f = render(tmp_path, [DOWN, PULL])
    down, pull = f["load_arrows"]["cases"]
    [a] = down["arrows"]
    assert (down["case"], a["region"], a["kind"], a["force_n"], a["size_n"]) == ("down", "tip", "force", [0.0, 0.0, -2.0], 2.0)
    assert a["point_mm"] == pytest.approx([38, 5, 1.5], abs=1e-3) and a["point_source"] == "region centroid"
    assert a["arrow_tip_mm"] == a["point_mm"] and a["color"] == hexcolor(0) == down["color"]
    assert a["arrow_tail_mm"][:2] == a["point_mm"][:2] and a["arrow_tail_mm"][2] > a["arrow_tip_mm"][2]   # the tail is above the tip
    assert np.linalg.norm(np.subtract(a["arrow_tip_mm"], a["arrow_tail_mm"])) == pytest.approx(f["load_arrows"]["length_mm"], abs=2e-3)
    [b] = pull["arrows"]
    assert b["point_mm"] == pytest.approx([22, 1, 5], abs=1e-3) and b["color"] == hexcolor(1) != a["color"]
    assert b["arrow_tail_mm"][0] < b["arrow_tip_mm"][0] and b["arrow_tail_mm"][1:] == b["point_mm"][1:]
    m_down = mask(png(f, "front"), R.CASE_COLORS[0])
    assert m_down.sum() > 200 and tip_side(m_down, 0) == 1          # the front view: the head is below the tail
    m_pull = mask(png(f, "top"), R.CASE_COLORS[1])
    assert m_pull.sum() > 200 and tip_side(m_pull, 1) == 1          # the top view (x to the right): the head is right of the tail


def test_the_arrow_is_not_drawn_the_other_way(tmp_path):
    """The same check with the force reversed: the head is above (front view) and left (top view)."""
    f = render(tmp_path, [{"name": "up", "forces": [{"region": "tip", "force": [0, 0, 2.0]}]},
                          {"name": "back", "forces": [{"region": "side", "force": [-4.0, 0, 0]}]}])
    assert tip_side(mask(png(f, "front"), R.CASE_COLORS[0]), 0) == -1
    assert tip_side(mask(png(f, "top"), R.CASE_COLORS[1]), 1) == -1


def test_a_remote_force_starts_at_its_at_point(tmp_path):
    far = {"name": "far", "forces": [{"region": "tip", "force": [0, 0, -1.0], "at": [10, 5, 9]}]}
    f = render(tmp_path, [far, DOWN])
    [a] = f["load_arrows"]["cases"][0]["arrows"]
    assert a["kind"] == "remote_force" and a["point_source"] == "at"
    assert a["point_mm"] == [10.0, 5.0, 9.0] and a["arrow_tip_mm"] == [10.0, 5.0, 9.0]
    at_tip = f["load_arrows"]["cases"][1]["arrows"][0]
    assert at_tip["point_source"] == "region centroid" and at_tip["point_mm"] == pytest.approx([38, 5, 1.5], abs=1e-3)
    far_px, near_px = (mask(png(f, "front"), R.CASE_COLORS[i]) for i in (0, 1))
    assert far_px.sum() > 200 and near_px.sum() > 200
    assert np.nonzero(far_px.any(0))[0].mean() < np.nonzero(near_px.any(0))[0].min()      # x = 10 is left of x = 38 in the front view


def test_a_moment_and_a_pressure_have_no_arrow_and_the_facts_say_so(tmp_path):
    f = render(tmp_path, [TWIST, PUSH])
    la = f["load_arrows"]
    assert la["drawn"] is True and la["arrow_count"] == 0
    twist, push = la["cases"]
    assert twist["arrows"] == [] and twist["color"] is None
    assert twist["not_drawn"] == [{"region": "tip", "kind": "moment", "moment_nmm": [0.0, 0.0, 100.0],
                                   "reason": "a moment has no arrow; only a force has one"}]
    assert push["arrows"] == [] and push["not_drawn"][0]["kind"] == "pressure" and push["not_drawn"][0]["value_mpa"] == 0.5
    assert any("'twist' has no force arrow" in n for n in f["notes"]) and any("'push' has no force arrow" in n for n in f["notes"])
    for c in range(len(R.CASE_COLORS)):
        for v in ("front", "top"):
            assert not mask(png(f, v), R.CASE_COLORS[c]).any()
    assert not any(e["style"] == "arrow" for e in f["legend"])


def test_a_force_with_a_moment_gets_an_arrow_and_the_facts_say_the_moment_is_not_drawn(tmp_path):
    f = render(tmp_path, [{"name": "both", "forces": [{"region": "tip", "force": [0, 0, -2.0], "moment": [0, 50.0, 0]}]}])
    [a] = f["load_arrows"]["cases"][0]["arrows"]
    assert a["kind"] == "force_moment" and a["moment_nmm"] == [0.0, 50.0, 0.0] and a["moment_drawn"] is False


def test_a_case_without_an_arrow_does_not_shift_the_colors_of_the_next_cases(tmp_path):
    """The color follows the case number, so a case without an arrow does not shift the colors of the next cases."""
    f = render(tmp_path, [TWIST, DOWN])
    assert f["load_arrows"]["cases"][1]["color"] == hexcolor(1)
    assert mask(png(f, "front"), R.CASE_COLORS[1]).sum() > 200


def test_the_option_off_draws_no_arrow_and_gives_the_render_without_arrows(tmp_path):
    cases = [DOWN, PULL]
    on = render(tmp_path, cases, out="on")
    off = render(tmp_path, cases, out="off", loads=False)
    none = render(tmp_path, [TWIST], out="none")                  # a YAML whose render has no arrow at all
    assert off["load_arrows"]["drawn"] is False and "loads is false" in off["load_arrows"]["note"]
    assert none["load_arrows"]["arrow_count"] == 0
    for i in range(2):
        assert mask(png(on, "front"), R.CASE_COLORS[i]).any() or mask(png(on, "top"), R.CASE_COLORS[i]).any()
        for v in ("front", "top"):
            assert not mask(png(off, v), R.CASE_COLORS[i]).any()
    assert not any(e["style"] == "arrow" for e in off["legend"])
    for a, b in zip(off["views"] + [off["montage"]], none["views"] + [none["montage"]]):    # the same bytes: the old render
        assert Path(a["path"]).read_bytes() == Path(b["path"]).read_bytes()
    assert Path(on["montage"]["path"]).read_bytes() != Path(off["montage"]["path"]).read_bytes()


def test_the_same_input_gives_the_same_png_bytes_with_arrows(tmp_path):
    a = render(tmp_path, [DOWN, PULL], out="a")
    b = render(tmp_path, [DOWN, PULL], out="b")
    assert a["load_arrows"] == b["load_arrows"]
    for x, y in zip(a["views"] + [a["montage"]], b["views"] + [b["montage"]]):
        assert Path(x["path"]).read_bytes() == Path(y["path"]).read_bytes()


def test_the_legend_and_the_header_name_each_case_with_an_arrow(tmp_path):
    f = render(tmp_path, [DOWN, TWIST, PULL])
    assert [(e["label"], e["style"]) for e in f["legend"] if e["style"] == "arrow"] == [("force: down", "arrow"), ("force: pull", "arrow")]
    assert f["load_arrows"]["arrow_count"] == 2
    # the swatch of the legend has the color of the case (the legend is below the tiles of the montage)
    sheet = np.asarray(Image.open(f["montage"]["path"]).convert("RGB"), float)
    for i in (0, 2):                                              # the case number: down is 0, pull is 2
        assert (np.abs(sheet[-80:] - np.array(R.CASE_COLORS[i], float)).max(2) < 1).any()


def test_forces_of_different_size_get_the_same_length_and_a_note_says_so(tmp_path):
    f = render(tmp_path, [DOWN, PULL])
    la = f["load_arrows"]
    lens = {round(float(np.linalg.norm(np.subtract(a["arrow_tip_mm"], a["arrow_tail_mm"]))), 2)
            for c in la["cases"] for a in c["arrows"]}
    assert len(lens) == 1 and {a["size_n"] for c in la["cases"] for a in c["arrows"]} == {2.0, 4.0}
    assert "does not show the size of the force" in la["length_rule"]
    assert any("the forces differ in size (2 N to 4 N)" in n and "does not show the size" in n for n in f["notes"])
    same = render(tmp_path, [DOWN, {"name": "down2", "forces": [{"region": "side", "force": [0, 0, -2.0]}]}], out="same")
    assert not any("differ in size" in n for n in same["notes"])


def test_more_cases_than_colors_repeat_the_colors_and_a_note_says_so(tmp_path):
    n = len(R.CASE_COLORS) + 1
    cases = [{"name": f"c{i}", "forces": [{"region": "tip", "force": [0, 0, -1.0 - i]}]} for i in range(n)]
    f = render(tmp_path, cases, views="front", size="320x240")
    assert f["load_arrows"]["cases"][n - 1]["color"] == f["load_arrows"]["cases"][0]["color"]
    assert any("the colors repeat" in x for x in f["notes"])


def test_a_load_region_that_is_not_a_closed_solid_gives_the_bbox_center_and_a_note(tmp_path):
    f = render(tmp_path, [DOWN], open_tip=True)
    [a] = f["load_arrows"]["cases"][0]["arrows"]
    assert a["point_source"] == "region bbox center" and a["point_mm"] == pytest.approx([38, 5, 1.5], abs=1e-3)
    assert any("not a closed solid" in n and "bounding box" in n for n in f["notes"])


def test_the_arrows_do_not_change_the_facts_of_the_meshes(tmp_path):
    on = render(tmp_path, [DOWN], out="on")
    off = render(tmp_path, [DOWN], out="off", loads=False)
    assert on["meshes"] == off["meshes"] and on["triangles_drawn"] == off["triangles_drawn"]


# --- the option and the other inputs ---------------------------------------------------------------

def test_loads_has_no_effect_for_an_stl_and_a_note_says_so(tmp_path):
    box(tmp_path / "a.stl", (0, 0, 0), (20, 10, 10))
    f = R.render(str(tmp_path / "a.stl"), loads=False, views="front", out_dir=str(tmp_path / "o"))
    assert f["load_arrows"] is None and any(n.startswith("loads has no effect") for n in f["notes"])
    g = R.render(str(tmp_path / "a.stl"), views="front", out_dir=str(tmp_path / "o2"))
    assert g["load_arrows"] is None and not any("loads" in n for n in g["notes"])


def test_loads_must_be_true_or_false(tmp_path):
    with pytest.raises(UserError, match="loads must be true or false"):
        check_inputs(str(write_case(tmp_path, [DOWN])), loads="off")
    check_inputs(str(write_case(tmp_path, [DOWN])), loads=False)
    check_inputs(str(write_case(tmp_path, [DOWN])), loads=True)


def test_the_cli_has_loads_and_no_loads(tmp_path, capsys):
    p = write_case(tmp_path, [DOWN])
    assert cli_main(["render", str(p), "--views", "front", "--size", "320x240", "--out", str(tmp_path / "a")]) == 0
    assert json.loads(capsys.readouterr().out)["load_arrows"]["arrow_count"] == 1
    assert cli_main(["render", str(p), "--no-loads", "--views", "front", "--size", "320x240", "--out", str(tmp_path / "b")]) == 0
    assert json.loads(capsys.readouterr().out)["load_arrows"]["drawn"] is False
    assert cli_main(["render", str(p), "--loads", "--views", "front", "--size", "320x240", "--out", str(tmp_path / "c")]) == 0
    assert json.loads(capsys.readouterr().out)["load_arrows"]["arrow_count"] == 1


def test_the_mcp_tool_draws_the_arrows_and_takes_loads_false(tmp_path):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    p = write_case(tmp_path, [DOWN])
    on = json.loads(server.render_part(yaml_path=str(p), views=["front"], size="320x240", inline=False, out_dir=str(tmp_path / "a"))[0].text)
    assert on["ok"] and on["load_arrows"]["arrow_count"] == 1 and on["load_arrows"]["cases"][0]["arrows"][0]["force_n"] == [0.0, 0.0, -2.0]
    off = json.loads(server.render_part(yaml_path=str(p), views=["front"], size="320x240", inline=False, loads=False,
                                        out_dir=str(tmp_path / "b"))[0].text)
    assert off["load_arrows"]["drawn"] is False
    bad = json.loads(server.render_part(yaml_path=str(p), views=["front"], inline=False, loads="maybe")[0].text)
    assert bad["ok"] is False


# --- the arrow shape and the colors ----------------------------------------------------------------

@pytest.mark.parametrize("tip", [(0, 0, 0), (3, -2, 5), (-4, 0, 1)])
def test_the_arrow_mesh_is_a_closed_solid_with_outward_winding(tip):
    tail = np.array([0.0, 0.0, 10.0])
    V, F = R.arrow_mesh(tail, tip)
    m = trimesh.Trimesh(V, F, process=False)
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0
    assert np.allclose(V[0], tail) and np.allclose(V[-1], tip)        # the center of the end of the shaft and the apex
    length = np.linalg.norm(np.subtract(tip, tail))
    assert np.linalg.norm(V[1:1 + R.ARROW_SEGMENTS] - tail, axis=1).max() == pytest.approx(R.ARROW_SHAFT_RADIUS * length)


def test_the_case_colors_differ_from_every_color_of_the_roles_and_from_each_other():
    used = list(R.ROLE_COLORS.values()) + [R.BACK_COLOR, R.CUT_COLOR]
    for c in R.CASE_COLORS:
        assert min(np.linalg.norm(np.subtract(c, u)) for u in used) >= 60
    for i, c in enumerate(R.CASE_COLORS):
        for d in R.CASE_COLORS[i + 1:]:
            assert np.linalg.norm(np.subtract(c, d)) >= 60


# --- the tool text -----------------------------------------------------------------------------------

def render_part_text():
    from topopt_mcp import server
    return {t.name: t.description for t in asyncio.run(server.mcp.list_tools())}["render_part"]


def test_the_render_part_text_offers_the_force_arrows_after_the_key_text():
    pytest.importorskip("mcp")
    text = render_part_text()
    key = text.find("Look at it before a run")
    at = text.find("force arrow")
    assert key >= 0 and at > key                                    # the key text of P7-7 keeps its place; the arrows come after it
    assert "check the directions" in text and "loads: false" in text
    assert len(text) <= 4096
    assert not re.search(r"\b(must|never|always)\b", text, re.I)
