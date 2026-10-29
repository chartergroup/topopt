"""What a cut takes from a frozen region, a bounded face, and the render of a problem YAML (H1 PR 2; CONTRACTS.md sections 23 and 15).

A `face` cuts the whole half-space on the side of its normal inside the grid box, and the label KEEP_OUT wins over LOAD and FIXED.
So a face through a sleeve takes the sleeve. `voxelize.taken_facts` counts it before the run, and a cut that takes a region deeper
than a machining trim is a warning. `radius_mm` bounds a face to a disc.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from test_optimize import box
from topopt import interfaces as I
from topopt.cli import buckling_off_line, check_data, main
from topopt.config import Interface, UserError, load_config
from topopt.voxelize import KEEP_OUT, LOAD, TRIM_MIN_MM, taken_lines, taken_warnings, voxelize

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "interfaces"


def write_case(tmp_path, collar=(24, 31.5), face=None, extra=None, h=2.0):
    """Design x 0..30, a clamp plate x -2..0, and a load region `sleeve` x `collar` (a bar that reaches past the plane x = 30).

    `face` is the dict of a face interface (default: the plane x = 30, normal +x, no stock); `extra` is more interfaces. Return the YAML path."""
    box(tmp_path / "design.stl", (0, 0, 0), (30, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "sleeve.stl", (collar[0], 2, 2), (collar[1], 8, 8))
    ifaces = [dict({"name": "end", "kind": "face", "point": [30, 5, 5], "normal": [1, 0, 0]}, **(face or {}))] + (extra or [])
    d = {"name": "c",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"sleeve": "sleeve.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "sleeve", "force": [0, 0, -2.0]}]}],
         "interfaces": ifaces,
         "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 2}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


# -- check A: what a cut takes ------------------------------------------------------------------------------------
def test_a_face_that_cuts_a_sleeve_is_a_warning(tmp_path):
    """The bar reaches 30 mm past the plane: the face takes most of it, far more than a trim."""
    d = check_data(write_case(tmp_path, collar=(24, 60)))
    [f] = d["interface_taken"]
    assert (f["cut"], f["cut_kind"], f["region"], f["region_role"]) == ("end", "face", "sleeve", "load")
    assert f["voxels"] > 0.7 * f["region_voxels"] and f["share"] == pytest.approx(f["voxels"] / f["region_voxels"], abs=1e-4)
    assert f["depth_mm"] == pytest.approx(30.0, abs=1e-6)       # the STL vertices: 60 - 30
    assert f["trim_mm"] == TRIM_MIN_MM and f["warn"] is True
    [w] = [w for w in d["warnings"] if w.startswith("interface 'end'")]
    assert "(face, cut) takes" in w and "of load region 'sleeve'" in w and "up to 30 mm past the plane" in w
    assert "A face cuts everything on the side of its normal" in w and "'radius_mm'" in w
    assert [l for l in taken_lines(d["interface_taken"])] and taken_warnings(d["interface_taken"]) == [w]


def test_a_face_that_trims_a_collar_end_is_a_fact_only(tmp_path):
    """The collar reaches 1.5 mm past the plane: a machining trim. The count is a fact, there is no warning."""
    d = check_data(write_case(tmp_path, collar=(24, 31.5)))
    [f] = d["interface_taken"]
    assert f["voxels"] > 0 and f["depth_mm"] == pytest.approx(1.5, abs=1e-6) and f["warn"] is False
    assert not [w for w in d["warnings"] if "takes" in w and "voxels" in w]
    [line] = taken_lines(d["interface_taken"])
    assert line.startswith("interface cut:  end (face) takes ") and "up to 1.5 mm past the plane" in line


def test_the_stock_widens_the_trim_limit(tmp_path):
    """Limit = max(stock + voxel size, 2 mm). A collar 3.4 mm past the plane warns without stock, and not with stock 1.5 at voxel 2 (3.5 mm)."""
    d = check_data(write_case(tmp_path, collar=(24, 33.4)))
    assert d["interface_taken"][0]["warn"] is True and d["interface_taken"][0]["trim_mm"] == 2.0
    d = check_data(write_case(tmp_path, collar=(24, 33.4), face={"stock_mm": 1.5}))
    f = d["interface_taken"][0]
    assert f["trim_mm"] == 3.5 and f["depth_mm"] == pytest.approx(3.4) and f["warn"] is False


def test_a_bore_in_a_boss_is_a_fact_only(tmp_path):
    """A coaxial bore takes the middle of a boss. That is what a bore is for: no depth, no warning."""
    boss = {"name": "boss", "kind": "boss", "point": [10, 5, -1], "axis": [0, 0, 1], "radius_mm": 3.0, "length_mm": 12.0}
    bore = {"name": "bore", "kind": "bore", "point": [10, 5, 12], "axis": [0, 0, -1], "radius_mm": 1.5, "length_mm": 14.0}
    d = check_data(write_case(tmp_path, collar=(24, 30), extra=[boss, bore], h=1.0))
    [f] = [f for f in d["interface_taken"] if f["cut"] == "bore"]
    assert (f["cut_kind"], f["region"], f["region_role"]) == ("bore", "boss", "interface")
    assert f["voxels"] > 0 and f["depth_mm"] is None and f["trim_mm"] is None and f["warn"] is False
    assert not [w for w in d["warnings"] if "bore" in w and "takes" in w]
    assert "(a hole removes material)" in taken_lines([f])[0]


def test_a_keep_out_region_inside_a_fixed_region(tmp_path):
    """A keep-out slab of 6 mm inside a thick fixed block is 3 mm deep: a warning. A slab of 2 mm is 1 mm deep: a fact only."""
    for ko, warn, depth in (((-8, -2), True, 3.0), ((-5, -3), False, 1.0)):
        sub = tmp_path / str(ko[0])
        sub.mkdir()
        p = write_case(sub, collar=(24, 30), extra=[])
        box(sub / "plate.stl", (-12, 0, 0), (0, 10, 10))
        box(sub / "ko.stl", (ko[0], 0, 0), (ko[1], 10, 10))
        d = yaml.safe_load(p.read_text())
        d["regions"]["keep_out"] = ["ko.stl"]
        d["interfaces"] = []
        p.write_text(yaml.safe_dump(d))
        got = check_data(p)
        [f] = got["interface_taken"]
        assert (f["cut"], f["cut_kind"], f["region"], f["region_role"]) == ("ko", "keep-out region", "plate", "fixed")
        assert f["depth_mm"] == pytest.approx(depth) and f["trim_mm"] == 2.0 and f["warn"] is warn
        ws = [w for w in got["warnings"] if w.startswith("keep-out region 'ko' takes")]
        assert bool(ws) is warn
        assert not ws or "of fixed region 'plate'" in ws[0] and "into the keep-out region" in ws[0]


# -- check B: a bounded face -----------------------------------------------------------------------------------------
def test_a_bounded_face_does_not_cut_away_from_its_disc(tmp_path):
    """radius_mm 1.5 about the axis: the disc on the axis is cut, the rest of the bar past the plane stays LOAD."""
    cfg = load_config(write_case(tmp_path, collar=(24, 60), face={"radius_mm": 1.5}, h=1.0))
    org, h, shape, labels, _ids = voxelize(cfg)

    def at(x, y, z):
        i, j, k = (np.floor((np.array([x, y, z]) - org) / h)).astype(int)
        return labels[i, j, k]

    assert at(40.5, 5.5, 5.5) == KEEP_OUT and at(55.5, 4.5, 4.5) == KEEP_OUT   # on the axis, past the plane
    assert at(40.5, 2.5, 2.5) == LOAD and at(40.5, 7.5, 7.5) == LOAD            # in the bar, outside the disc
    assert at(27.5, 5.5, 5.5) == LOAD                                           # before the plane
    d = check_data(write_case(tmp_path, collar=(24, 60), face={"radius_mm": 1.5}, h=1.0))
    [f] = d["interface_taken"]
    assert f["bounded"] is True and f["share"] < 0.2          # the disc takes a small part of the bar
    [w] = [w for w in d["warnings"] if w.startswith("interface 'end'")]   # it still cuts the bar deeply: the same warning
    assert "A face with 'radius_mm' cuts the disc of that radius" in w and "make 'radius_mm' smaller" in w


def test_a_radius_that_clears_the_sleeve_removes_the_warning(tmp_path):
    """A face with radius_mm 0.1 takes a sliver on the axis only: it does not take the sleeve, but depth is still a fact."""
    d = check_data(write_case(tmp_path, collar=(24, 31.5), face={"radius_mm": 4.0}))
    assert all(not f["warn"] for f in d["interface_taken"])


def test_the_face_cutter_is_a_disc_and_the_old_half_space_is_unchanged():
    box_ = (np.array([0.0, 0.0, 0.0]), np.array([20.0, 20.0, 20.0]))
    point, n = np.array([10.0, 10.0, 10.0]), np.array([1.0, 0.0, 0.0])
    cyl = I._cutter_of_face(point, n, 0.0, box_, radius=3.0)
    v = np.asarray(cyl.to_mesh().vert_properties)[:, :3]
    assert v[:, 0].min() == pytest.approx(10.0) and v[:, 0].max() == pytest.approx(20.0 + I.BOX_MARGIN_MM)
    assert np.hypot(v[:, 1] - 10.0, v[:, 2] - 10.0).max() <= 3.0 + 1e-5
    assert cyl.volume() == pytest.approx(np.pi * 9.0 * (10.0 + I.BOX_MARGIN_MM), rel=0.01)
    assert I._cutter_of_face(point, n, 0.0, box_, radius=3.0).volume() > 0
    assert I._cutter_of_face(point, n, 0.0, (np.array([0.0] * 3), np.array([5.0] * 3)), radius=3.0).is_empty()   # the box lies before the plane
    # the half-space: the formula of before this change
    import manifold3d as m3d
    lo, hi = box_
    m = I.BOX_MARGIN_MM + 0.5
    old = m3d.Manifold.cube(tuple(hi - lo + 2 * m)).translate(tuple(lo - m)).trim_by_plane(tuple(n), float(np.dot(point, n) + 0.5))
    new = I._cutter_of_face(point, n, 0.5, box_)
    a, b = I._arrays(old), I._arrays(new)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_shapes_of_a_bounded_face_with_stock():
    """The as-printed disc starts at the plane moved by the stock; the label disc is the same; the nominal disc starts at the plane."""
    spec = Interface("end", "face", "cut", 0.5, (10.0, 10.0, 10.0), (1.0, 0.0, 0.0), 4.0)
    sh = I.Shapes([spec], (np.zeros(3), np.full(3, 20.0)))
    x0 = {w: float(sh.arrays(spec, w)[0][:, 0].min()) for w in (I.AS_PRINTED, I.NOMINAL, I.LABEL)}
    assert x0[I.AS_PRINTED] == pytest.approx(10.5) and x0[I.LABEL] == pytest.approx(10.5) and x0[I.NOMINAL] == pytest.approx(10.0)
    assert sh.manifold(spec, I.AS_PRINTED).volume() == pytest.approx(np.pi * 16.0 * (9.5 + I.BOX_MARGIN_MM), rel=0.01)


def test_measure_of_a_bounded_face(tmp_path):
    """A recess of radius 4 in a cube: the measure finds the plane inside the disc only, and the coverage is about 1."""
    import manifold3d as m3d
    from topopt.post import _to_manifold
    spec = Interface("end", "face", "cut", 0.0, (10.0, 10.0, 10.0), (1.0, 0.0, 0.0), 4.0)
    cube = m3d.Manifold.cube((20.0, 20.0, 20.0))
    cutter = I._cutter_of_face(spec.point, spec.axis, 0.0, (np.zeros(3), np.full(3, 20.0)), spec.radius_mm)
    me = (cube - cutter).to_mesh()
    V, F = np.asarray(me.vert_properties, np.float64)[:, :3], np.asarray(me.tri_verts, np.int64)
    body = _to_manifold(V, F)
    got = I.measure(spec, "as_printed", V, F, body=body, box=(np.zeros(3), np.full(3, 20.0)))["position_mm"]
    assert got["found"] and got["within_tolerance"]
    assert got["coverage"] == pytest.approx(1.0, abs=0.02)
    assert got["area_mm2"] == pytest.approx(np.pi * 16.0, rel=0.01)


def test_config_reads_radius_mm_of_a_face(tmp_path):
    p = write_case(tmp_path, face={"radius_mm": 12.5})
    [spec] = load_config(p).interfaces
    assert spec.radius_mm == 12.5
    back = Interface.from_dict(json.loads(json.dumps(spec.to_dict(tmp_path))), tmp_path)
    assert back == spec
    for bad in (0, -1, "x", True):
        d = yaml.safe_load(p.read_text())
        d["interfaces"][0]["radius_mm"] = bad
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(UserError, match="radius_mm"):
            load_config(p)
    d = yaml.safe_load(p.read_text())
    d["interfaces"] = [{"name": "b", "kind": "bore", "point": [1, 1, 1], "axis": [0, 0, 1], "radius_mm": 1.0, "length_mm": 2.0}]
    d["interfaces"][0]["radius_mm"] = 1.0
    p.write_text(yaml.safe_dump(d))
    assert load_config(p).interfaces[0].radius_mm == 1.0       # a bore keeps its own radius_mm


# -- without the new key: the same labels and shapes as before ---------------------------------------------------------
LABELS = {   # sha256 of `labels` and `load_ids` of voxelize, measured with the code of origin/main (d43f580)
    "bracket": ("9336f9af66b6848d3e1b9605f98a31b2ca881c4ff8416e8606677f1325ee4b7e",
                "c831f806ea8faeae7f92755c494700482845afa3696a8ac2b9ef0a6cfb5f42f4"),
    "bracket_nostock": ("31122648ab08c71b73a8b4767e2d6d601d04025b9af712d4bb43bc66b61b1d69",
                        "c831f806ea8faeae7f92755c494700482845afa3696a8ac2b9ef0a6cfb5f42f4"),
}


@pytest.mark.parametrize("name", sorted(LABELS))
def test_the_interface_examples_give_the_same_labels_without_radius_mm(name):
    cfg = load_config(EXAMPLE / f"{name}.yaml")
    assert all(i.radius_mm is None for i in cfg.interfaces if i.kind == "face")
    _org, _h, _shape, labels, ids = voxelize(cfg)
    assert (hashlib.sha256(labels.tobytes()).hexdigest(), hashlib.sha256(ids.tobytes()).hexdigest()) == LABELS[name]


def test_the_bracket_face_shapes_are_the_old_half_space():
    cfg = load_config(EXAMPLE / "bracket.yaml")
    org, h, shape, _labels, _ids = voxelize(cfg)
    box_ = (org, org + h * np.asarray(shape, float))
    [face] = [i for i in cfg.interfaces if i.kind == "face"]
    import manifold3d as m3d
    n = np.asarray(face.axis, float)
    for stock in (0.0, face.stock_mm):
        lo, hi = box_
        m = I.BOX_MARGIN_MM + stock
        old = m3d.Manifold.cube(tuple(hi - lo + 2 * m)).translate(tuple(lo - m)).trim_by_plane(tuple(n), float(np.dot(face.point, n) + stock))
        new = I.build_shape(face, stock, box_)
        a, b = I._arrays(old), I._arrays(new)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_the_examples_have_no_cut_warning():
    d = check_data(EXAMPLE / "bracket.yaml")
    assert not [w for w in d["warnings"] if w.startswith(("interface '", "keep-out region '"))]


# -- run.json and the run log ------------------------------------------------------------------------------------------
def test_run_json_and_the_check_json_hold_the_facts(tmp_path, capsys):
    p = write_case(tmp_path, collar=(24, 60))
    assert main(["check", str(p), "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["interface_taken"][0]["warn"] is True
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["interface_taken"] == got["interface_taken"]
    log = (out / "run.log").read_text()
    assert "interface cut:  end (face) takes" in log and "warning: interface 'end' (face, cut) takes" in log


def test_topopt_check_prints_the_lines(tmp_path, capsys):
    assert main(["check", str(write_case(tmp_path, collar=(24, 60)))]) == 0
    text = capsys.readouterr().out
    assert "warning: interface 'end' (face, cut) takes" in text
    assert "interface cut:  end (face) takes" in text


# -- the buckling line ---------------------------------------------------------------------------------------------------
def test_the_buckling_off_line(tmp_path, capsys):
    p = write_case(tmp_path, collar=(24, 31.5))
    d = check_data(p)
    info = d["buckling_off"]
    assert info["memory_estimate_bytes"] > 0 and info["max_seconds"] == 120.0
    line = buckling_off_line(info)
    assert line.startswith("buckling: off (check and row). A part with members in compression can buckle: buckling.check true gives the load factors (memory estimate about ")
    assert "the time budget of each eigen solve is 120 s" in line
    assert main(["check", str(p)]) == 0
    assert line in capsys.readouterr().out
    on = yaml.safe_load(p.read_text())
    on["buckling"] = {"check": True}
    p.write_text(yaml.safe_dump(on))
    assert "buckling_off" not in check_data(p)       # on: the existing line, no new one


# -- render of a problem YAML and the --role comma list -------------------------------------------------------------------
def test_render_of_a_yaml_writes_pngs(tmp_path):
    from topopt.render import render
    p = write_case(tmp_path, collar=(24, 60), extra=[
        {"name": "boss", "kind": "boss", "point": [10, 5, -1], "axis": [0, 0, 1], "radius_mm": 3.0, "length_mm": 12.0}])
    facts = render(str(p), views="iso,front", out_dir=str(tmp_path / "r"))
    assert facts["ok"] and facts["input"]["kind"] == "yaml"
    roles = {f["role"]: f["drawn"] for f in facts["meshes"]}
    assert roles == {"design": "ghost", "fixed": "solid", "load": "solid", "interface": "solid", "cut": "ghost"}
    assert {l["label"] for l in facts["legend"]} >= {"union shape", "cut shape"}
    for v in facts["views"]:
        assert Path(v["path"]).is_file() and Path(v["path"]).stat().st_size > 0
    assert Path(facts["montage"]["path"]).name == "c_montage.png"
    face = [f for f in facts["meshes"] if f["role"] == "cut"][0]
    assert face["bbox_mm"]["max"][0] <= 60.0 + 1e-3        # clipped to the grid box: not the margin of the exact step


def test_the_cli_renders_a_yaml(tmp_path, capsys):
    p = write_case(tmp_path)
    assert main(["render", str(p), "--views", "front", "--out", str(tmp_path / "o")]) == 0
    assert json.loads(capsys.readouterr().out)["input"]["kind"] == "yaml"
    assert (tmp_path / "o" / "c_front.png").is_file()
    assert main(["render", str(p), "--role", "part", "--out", str(tmp_path / "o")]) == 1      # roles do not apply to a YAML
    assert "roles do not apply" in capsys.readouterr().err


def test_role_comma_list_and_repeated_flags(tmp_path, capsys):
    box(tmp_path / "a.stl", (0, 0, 0), (4, 4, 4))
    box(tmp_path / "b.stl", (5, 0, 0), (9, 4, 4))
    outs = []
    for args in (["--role", "design,fixed"], ["--role", "design", "--role", "fixed"], ["--role", "design", "--role", "fixed,"]):
        assert main(["render", str(tmp_path / "a.stl"), str(tmp_path / "b.stl"), *args, "--views", "front", "--out", str(tmp_path / "o")]) == 0
        outs.append([m["role"] for m in json.loads(capsys.readouterr().out)["meshes"]])
    assert outs == [["design", "fixed"]] * 3
    with pytest.raises(SystemExit):
        main(["render", str(tmp_path / "a.stl"), "--role", "design,bogus"])
    assert "unknown role 'bogus'" in capsys.readouterr().err


def test_the_mcp_tool_renders_a_yaml_and_takes_a_role_comma_list(tmp_path):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    p = write_case(tmp_path, collar=(24, 60))
    d = json.loads(server.render_part(yaml_path=str(p), views=["front"], size="320x240", inline=False,
                                      out_dir=str(tmp_path / "o"))[0].text)
    assert d["ok"] and d["input"]["kind"] == "yaml" and "cut" in {m["role"] for m in d["meshes"]}
    a, b = tmp_path / "a.stl", tmp_path / "b.stl"
    box(a, (0, 0, 0), (4, 4, 4))
    box(b, (5, 0, 0), (9, 4, 4))
    d = json.loads(server.render_part(stl_paths=[str(a), str(b)], roles=["design,fixed"], views=["front"], size="320x240",
                                      inline=False, out_dir=str(tmp_path / "o2"))[0].text)
    assert [m["role"] for m in d["meshes"]] == ["design", "fixed"]
    d = json.loads(server.render_part(yaml_path=str(p), roles=["part"], inline=False)[0].text)
    assert d["ok"] is False and "roles do not apply" in d["error"]
