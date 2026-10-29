"""Tests of the exact interfaces and the machining stock (topopt/interfaces.py, CONTRACTS.md section 23).

The measurements in these tests read the STL files back from disk and compute the bore radius, the boss radius and the
face position with their own numpy code. They do not call `interfaces.measure`.
"""
import json
import logging
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_final_stl  # noqa: E402
from test_optimize import box, make_case  # noqa: E402
from test_post import load_stl, make_hole_run  # noqa: E402
from test_smooth_sdf import _pinch_fixture  # noqa: E402
from topopt import interfaces as I  # noqa: E402
from topopt import post, smooth_sdf  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.config import Interface, UserError, load_config  # noqa: E402
from topopt.voxelize import voxelize  # noqa: E402

EXAMPLE = ROOT / "examples" / "interfaces"
SAG_TOL = I.SAG_MM + 2 * I.FEATURE_TOL_MM   # the sag of the facets, the float32 resolution of the file and the rounding


# -- loader ----------------------------------------------------------------------------------------------------
def _config_with(tmp_path, entries, extra=None):
    """A config file from `make_case` with the `interfaces` list `entries`. Return its path."""
    p = make_case(tmp_path, h=2.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    d["interfaces"] = entries
    if extra:
        d.update(extra)
    p.write_text(yaml.safe_dump(d))
    return p


BORE = {"name": "b", "kind": "bore", "point": [5, 5, 12], "axis": [0, 0, -1], "radius_mm": 2.0, "length_mm": 8.0}
BOSS = {"name": "s", "kind": "boss", "point": [5, 5, -1], "axis": [0, 0, 1], "radius_mm": 3.0, "length_mm": 6.0}
TUBE = {"name": "t", "kind": "tube", "point": [5, 5, -1], "axis": [0, 0, 1], "radius_mm": 4.0, "inner_radius_mm": 2.0,
        "length_mm": 6.0}
FACE = {"name": "f", "kind": "face", "point": [1, 0, 0], "normal": [-1, 0, 0]}


def test_no_interfaces_by_default(tmp_path):
    assert load_config(make_case(tmp_path)).interfaces == ()


def test_defaults_and_kinds(tmp_path):
    box(tmp_path / "tab.stl", (0, 0, 0), (1, 1, 1))
    entries = [dict(BORE), dict(BOSS), dict(TUBE, stock_mm=0.25), dict(FACE, normal=[-3, 0, 0]),
               {"name": "u", "kind": "stl", "op": "union", "file": "tab.stl"},
               {"name": "c", "kind": "stl", "op": "cut", "file": "tab.stl"}]
    cfg = load_config(_config_with(tmp_path, entries))
    got = {i.name: i for i in cfg.interfaces}
    assert list(got) == ["b", "s", "t", "f", "u", "c"]                       # the order of the YAML
    assert [got[n].op for n in got] == ["cut", "union", "union", "cut", "union", "cut"]
    assert [got[n].stock_mm for n in got] == [0.0, 0.0, 0.25, 0.0, 0.0, 0.0]   # the default stock is 0
    assert got["f"].axis == (-1.0, 0.0, 0.0)                                  # the normal is a unit vector
    assert got["u"].path == tmp_path / "tab.stl"


def test_run_dict_round_trip(tmp_path):
    box(tmp_path / "tab.stl", (0, 0, 0), (1, 1, 1))
    entries = [dict(BORE, stock_mm=0.1), dict(BOSS), dict(TUBE), dict(FACE, stock_mm=0.3),
               {"name": "u", "kind": "stl", "op": "union", "file": "tab.stl"}]
    cfg = load_config(_config_with(tmp_path, entries))
    stored = json.loads(json.dumps([i.to_dict(tmp_path) for i in cfg.interfaces]))
    assert stored[-1]["file"] == "tab.stl"      # relative to the YAML folder: the run folder has no absolute path
    back = [Interface.from_dict(d, tmp_path) for d in stored]
    assert back == list(cfg.interfaces)


BAD = [
    ("kind", {"kind": "hole"}, "must be one of bore, boss, tube, face, stl"),
    ("unknown key", {"inner_radius_mm": 1.0}, "unknown key 'inner_radius_mm'"),
    ("takes list", {"colour": "red"}, "takes: name, kind, point, axis, radius_mm, length_mm, stock_mm"),
    ("missing radius", {"radius_mm": None}, "missing key 'radius_mm'"),
    ("negative stock", {"stock_mm": -0.1}, "stock_mm"),
    ("stock over radius", {"stock_mm": 2.0}, "not less than radius_mm"),
    ("radius zero", {"radius_mm": 0.0}, "radius_mm"),
    ("zero axis", {"axis": [0, 0, 0]}, "zero vector"),
    ("point shape", {"point": [1, 2]}, "list of 3 numbers"),
    ("name type", {"name": 3}, "non-empty string"),
]


@pytest.mark.parametrize("label,change,text", BAD, ids=[b[0] for b in BAD])
def test_bad_bore_entry_is_a_loader_error(tmp_path, label, change, text):
    e = dict(BORE)
    for k, v in change.items():
        if v is None:
            e.pop(k)
        else:
            e[k] = v
    with pytest.raises(UserError, match=text):
        load_config(_config_with(tmp_path, [e]))


def test_other_loader_errors(tmp_path):
    box(tmp_path / "tab.stl", (0, 0, 0), (1, 1, 1))
    stl = {"name": "u", "kind": "stl", "op": "union", "file": "tab.stl"}
    cases = [
        ([dict(TUBE, inner_radius_mm=4.5)], "not less than radius_mm"),
        ([dict(TUBE, stock_mm=2.5)], "not less than inner_radius_mm"),
        ([dict(BORE), dict(BORE, kind="boss")], "duplicate interface name 'b'"),
        ([dict(stl, stock_mm=0.1)], "STL solid has no stock"),
        ([dict(stl, op="add")], "must be union or cut"),
        ([dict(stl, file="missing.stl")], "missing.stl"),
        ([{k: v for k, v in stl.items() if k != "op"}], "missing key 'op'"),
        ([dict(FACE, normal=[0, 0, 0])], "zero vector"),
        ([dict(FACE, inner_radius_mm=1.0)], "unknown key 'inner_radius_mm'"),   # a face takes radius_mm (the bound of the cut), not the other cylinder keys
        ([dict(FACE, length_mm=1.0)], "unknown key 'length_mm'"),
        ("bore", "must be a list of mappings"),
        (["bore"], "must be a mapping"),
    ]
    for entries, text in cases:
        with pytest.raises(UserError, match=text):
            load_config(_config_with(tmp_path, entries))


# -- shapes ----------------------------------------------------------------------------------------------------
def test_segments_keep_the_sag():
    for r in (0.05, 0.3, 1.0, 2.6, 5.9, 25.0, 120.0):
        n = I.segments(r)
        assert n % 2 == 0 and n >= I.MIN_SEGMENTS
        assert I.facet_sag(r) <= I.SAG_MM + 1e-12 or r <= I.SAG_MM
        assert I.facet_sag(r) == pytest.approx(r * (1 - math.cos(math.pi / n)))


def _poly_area(r):
    n = I.segments(r)
    return 0.5 * n * r * r * math.sin(2 * math.pi / n)


def test_built_in_shapes_are_exact():
    box_ = ((0.0, 0.0, 0.0), (20.0, 10.0, 10.0))
    bore = Interface("b", "bore", "cut", 0.3, (5, 5, 9), (0, 0, -1), 2.0, None, 8.0)
    boss = Interface("s", "boss", "union", 0.3, (5, 5, 0), (0, 0, 1), 3.0, None, 6.0)
    tube = Interface("t", "tube", "union", 0.3, (5, 5, 0), (0, 0, 1), 4.0, 2.0, 6.0)
    face = Interface("f", "face", "cut", 0.5, (2, 0, 0), (-1, 0, 0))
    vol = lambda spec, stock: I.build_shape(spec, stock, box_).volume()   # noqa: E731
    assert vol(bore, 0.0) == pytest.approx(_poly_area(2.0) * 8.0, rel=1e-9)
    assert vol(bore, 0.3) == pytest.approx(_poly_area(1.7) * 8.0, rel=1e-9)   # the as-printed bore is smaller
    assert vol(boss, 0.3) == pytest.approx(_poly_area(3.3) * 6.0, rel=1e-9)   # the as-printed boss is larger
    assert vol(tube, 0.0) == pytest.approx((_poly_area(4.0) - _poly_area(2.0)) * 6.0, rel=1e-9)
    assert vol(tube, 0.3) == pytest.approx((_poly_area(4.3) - _poly_area(1.7)) * 6.0, rel=1e-9)
    m = I.BOX_MARGIN_MM
    # the face cutter: everything at x < 2 - stock, up to the margin around the grid box
    for stock in (0.0, 0.5):
        assert vol(face, stock) == pytest.approx((2 - stock + m + stock) * (10 + 2 * (m + stock)) * (10 + 2 * (m + stock)), rel=1e-9)
    # the cylinder wall lies between the facet radius and the nominal radius
    arr = I.Shapes([boss], box_).arrays(boss, I.AS_PRINTED)
    rho = np.linalg.norm(arr[0][:, :2] - [5, 5], axis=1)
    assert np.allclose(rho[np.abs(rho) > 1e-9], 3.3, atol=1e-6)


def _block(lo, hi):
    m = trimesh.creation.box(bounds=[lo, hi])
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def _step(specs, verts, faces, frozen=None, box_=((-5, -5, -5), (25, 25, 25))):
    ctx = I.Context(specs, tuple(np.asarray(b, float) for b in box_), I.Shapes(specs, box_), any(
        s.kind != "stl" and s.stock_mm > 0 for s in specs), frozen)
    v, f, res = I.apply_step(ctx, verts, faces)
    return ctx, v, f, res


def test_exact_step_volumes_on_a_block():
    """Union of a boss, cut of a bore and a face on a block: the volume is the exact sum."""
    v, f = _block((0, 0, 0), (20, 10, 10))
    boss = Interface("s", "boss", "union", 0.0, (10, 5, 10), (0, 0, 1), 3.0, None, 4.0)         # on the top face
    bore = Interface("b", "bore", "cut", 0.0, (10, 5, 15), (0, 0, -1), 1.0, None, 12.0)         # through boss and block
    face = Interface("f", "face", "cut", 0.0, (2, 0, 0), (-1, 0, 0))
    ctx, v2, f2, res = _step([boss, bore, face], v, f)
    assert res.applied and ctx.has_stock is False and list(res.variants) == ["as_printed"]
    want = 20 * 10 * 10 + _poly_area(3.0) * 4.0 - _poly_area(1.0) * 11.0 - 2 * 10 * 10     # the bore runs from z 14 down to z 3
    got = trimesh.Trimesh(v2, f2, process=False)
    assert got.is_watertight and got.volume == pytest.approx(want, rel=1e-6)
    assert sum(res.deltas["as_printed"].values()) == pytest.approx(want - 2000, rel=1e-6)
    assert res.deltas["as_printed"]["b"] == pytest.approx(-_poly_area(1.0) * 11.0, rel=1e-6)


def test_exact_step_stock_variants_on_a_block():
    v, f = _block((0, 0, 0), (20, 10, 10))
    boss = Interface("s", "boss", "union", 0.4, (10, 5, 10), (0, 0, 1), 3.0, None, 4.0)
    bore = Interface("b", "bore", "cut", 0.4, (10, 5, 15), (0, 0, -1), 1.5, None, 12.0)
    face = Interface("f", "face", "cut", 0.5, (2, 0, 0), (-1, 0, 0))
    ctx, v2, f2, res = _step([boss, bore, face], v, f)
    assert ctx.has_stock and set(res.variants) == {"as_printed", "nominal"}
    ap = trimesh.Trimesh(*res.variants["as_printed"], process=False)
    nom = trimesh.Trimesh(*res.variants["nominal"], process=False)
    assert ap.is_watertight and nom.is_watertight
    assert nom.volume == pytest.approx(20 * 10 * 10 + _poly_area(3.0) * 4 - _poly_area(1.5) * 11 - 2 * 100, rel=1e-6)
    assert ap.volume == pytest.approx(20 * 10 * 10 + _poly_area(3.4) * 4 - _poly_area(1.1) * 11 - 1.5 * 100, rel=1e-6)
    assert np.allclose(v2, res.variants["as_printed"][0])                 # the step returns the as-printed mesh


def test_stl_solids_are_exact_union_and_cut(tmp_path):
    v, f = _block((0, 0, 0), (20, 10, 10))
    box(tmp_path / "tab.stl", (18, 2, 2), (26, 8, 8))     # sticks out of the end face by 6 mm
    box(tmp_path / "slot.stl", (5, -1, 4), (8, 11, 6))    # a slot through the block along y
    specs = [Interface("tab", "stl", "union", 0.0, path=tmp_path / "tab.stl"),
             Interface("slot", "stl", "cut", 0.0, path=tmp_path / "slot.stl")]
    ctx, v2, f2, res = _step(specs, v, f)
    got = trimesh.Trimesh(v2, f2, process=False)
    assert got.is_watertight and got.volume == pytest.approx(2000 + 6 * 6 * 6 - 3 * 10 * 2, rel=1e-9)
    assert not ctx.has_stock and res.deltas["as_printed"]["tab"] == pytest.approx(216.0)
    assert res.deltas["as_printed"]["slot"] == pytest.approx(-60.0)


def test_a_failing_exact_step_keeps_the_mesh(monkeypatch, caplog):
    v, f = _block((0, 0, 0), (20, 10, 10))
    boss = Interface("s", "boss", "union", 0.0, (10, 5, 10), (0, 0, 1), 3.0, None, 4.0)
    monkeypatch.setattr(I, "_variant", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no boolean today")))
    with caplog.at_level(logging.WARNING, logger="topopt.interfaces"):
        _ctx, v2, f2, res = _step([boss], v, f)
    assert not res.applied and "no boolean today" in res.error and v2 is v and f2 is f
    assert any("interface step failed" in r.message for r in caplog.records)


def test_a_body_that_no_frozen_shape_touches_is_removed(tmp_path, caplog):
    """A bore that cuts a block in two leaves a piece that no fixed or load region holds. The step removes it and says so."""
    import manifold3d as m3d

    box(tmp_path / "fixed.stl", (-2, 0, 0), (0, 10, 10))
    v, f = _block((0, 0, 0), (20, 10, 10))
    bore = Interface("b", "bore", "cut", 0.0, (10, 20, 5), (0, -1, 0), 6.0, None, 30.0)   # r 6 at z 5: cuts the block through
    with caplog.at_level(logging.WARNING, logger="topopt.interfaces"):
        _c, va, fa, ra = _step([bore], v, f, frozen=[tmp_path / "fixed.stl"])
        _c, vb, fb, rb = _step([bore], v, f, frozen=None)
    pieces = (m3d.Manifold.cube((20, 10, 10)) - I.build_shape(bore, 0.0)).decompose()
    right = [p for p in pieces if np.asarray(p.to_mesh().vert_properties)[:, 0].min() > 1.0]
    assert len(pieces) == 2 and len(right) == 1
    assert ra.dropped["as_printed"]["bodies"] == 1 and ra.dropped["as_printed"]["volume_mm3"] == pytest.approx(right[0].volume(), abs=1e-3)
    assert len(trimesh.Trimesh(va, fa, process=False).split(only_watertight=False)) == 1
    assert rb.dropped["as_printed"]["bodies"] == 0                     # without the region files nothing is removed
    assert len(trimesh.Trimesh(vb, fb, process=False).split(only_watertight=False)) == 2
    assert any("no frozen shape touches" in r.message for r in caplog.records)


def test_a_body_of_the_method_mesh_stays(tmp_path):
    """A body that the mesh method already had (an island away from the interfaces) is not removed: the same run with and
    without an interface gives the same bodies away from the interface."""
    box(tmp_path / "fixed.stl", (-2, 0, 0), (0, 10, 10))
    v, f = _block((0, 0, 0), (20, 10, 10))
    v2, f2 = _block((38, 3, 3), (42, 7, 7))         # a second body in the method mesh, 18 mm from the block
    bodies = trimesh.util.concatenate([trimesh.Trimesh(v, f), trimesh.Trimesh(v2, f2)])
    bore = Interface("b", "bore", "cut", 0.0, (5, 5, 12), (0, 0, -1), 1.0, None, 4.0)   # a blind hole: no new body
    far = Interface("far", "bore", "cut", 0.0, (40, 5, 12), (0, 0, -1), 1.0, None, 4.0)  # a hole in the island, no split
    for specs in ([bore], [bore, far]):
        _c, va, fa, ra = _step(specs, bodies.vertices, bodies.faces, frozen=[tmp_path / "fixed.stl"])
        assert ra.dropped["as_printed"]["bodies"] == 0
        assert len(trimesh.Trimesh(va, fa, process=False).split(only_watertight=False)) == 2


# -- labels ------------------------------------------------------------------------------------------------------
def _part(tmp_path, entries, h=1.0, iterations=2, vf=0.3, keep_out=None):
    """A 24 x 14 x 14 part with a clamped plate and a load box at the tip, and the interfaces `entries`.

    `keep_out` is a box `(lo, hi)` that is written as the keep-out region `pocket.stl`.
    """
    box(tmp_path / "design.stl", (0, 0, 0), (24, 14, 14))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 14, 14))
    box(tmp_path / "tip.stl", (22, 5, 0), (24, 9, 3))
    d = {"name": "part", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": vf, "iterations": iterations}}
    if keep_out is not None:
        box(tmp_path / "pocket.stl", *keep_out)
        d["regions"]["keep_out"] = ["pocket.stl"]
    if entries is not None:
        d["interfaces"] = entries
    p = tmp_path / "part.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_a_bore_gives_the_same_labels_as_a_keep_out_region(tmp_path):
    """The label code is the existing one: the as-printed bore as an STL keep-out region gives the same labels."""
    bore = dict(BORE, point=[10.3, 7.2, 14.0], radius_mm=2.7, length_mm=14.0, stock_mm=0.3)   # z 0..14: as a region, no wider grid
    cfg = load_config(_part(tmp_path, [bore]))
    org, h, shape, labels, ids = voxelize(cfg)
    spec = cfg.interfaces[0]
    cutter = I.Shapes([spec], (org, org + h * np.asarray(shape))).trimesh(spec, I.LABEL)
    assert cutter is not None
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(tmp_path / "design.stl", other)
    shutil.copy(tmp_path / "plate.stl", other)
    shutil.copy(tmp_path / "tip.stl", other)
    cutter.export(other / "bore.stl")
    d = yaml.safe_load((tmp_path / "part.yaml").read_text())
    d.pop("interfaces")
    d["regions"]["keep_out"] = ["bore.stl"]
    (other / "part.yaml").write_text(yaml.safe_dump(d))
    org2, h2, shape2, labels2, ids2 = voxelize(load_config(other / "part.yaml"))
    assert np.array_equal(org, org2) and shape == shape2
    assert np.array_equal(labels, labels2) and np.array_equal(ids, ids2)
    assert (labels == post.KEEP_OUT).sum() > 100
    # and the radius is the as-printed radius: 2.7 - 0.3 = 2.4 mm
    c = org + h * (np.indices(shape).reshape(3, -1).T + 0.5)
    rho = np.linalg.norm(c[:, :2] - [10.3, 7.2], axis=1)
    ko = (labels == post.KEEP_OUT).reshape(-1)
    inside = rho < 2.4
    near = np.abs(rho - 2.4) < 0.01
    assert (ko[~near] == inside[~near]).all()


def test_a_boss_is_load_without_a_load_case(tmp_path):
    boss = dict(BOSS, point=[12.2, 7.1, 6.0], axis=[0, 0, 1], radius_mm=3.3, length_mm=11.0, stock_mm=0.4)   # ends at z 17
    cfg = load_config(_part(tmp_path, [boss]))
    org, h, shape, labels, ids = voxelize(cfg)
    plain = voxelize(load_config(_part(tmp_path, None)))
    c = org + h * (np.indices(shape).reshape(3, -1).T + 0.5)
    rho = np.linalg.norm(c[:, :2] - [12.2, 7.1], axis=1)
    lab = labels.reshape(-1)
    ids_f = ids.reshape(-1)
    inside = (rho < 3.3) & (c[:, 2] >= 6.0) & (c[:, 2] <= 17.0)          # the nominal radius: the labels have no stock
    near = np.abs(rho - 3.3) < 0.01
    on = (lab == post.LOAD)
    tip = (c[:, 0] > 22) & (c[:, 1] > 5) & (c[:, 1] < 9) & (c[:, 2] < 3)       # the load box of the part
    assert (on[~near & ~tip] == inside[~near & ~tip]).all()
    assert (ids_f[on & ~tip] == -1).all()                                  # no load case: it is solid, it takes no force
    assert (ids_f[tip & on] == 0).all() and (ids_f[tip & on] == 0).any()   # the tip still has its force
    assert (lab[inside & ~near] != post.DESIGN).all()                      # the optimizer cannot remove the boss
    # the same labels as a load region that no load case uses (section 2: a forced-solid body), made from the nominal boss
    other = tmp_path / "other"
    other.mkdir()
    for n in ("design", "plate", "tip"):
        shutil.copy(tmp_path / f"{n}.stl", other)
    spec = cfg.interfaces[0]
    I.Shapes([spec], ((0, 0, 0), (1, 1, 1))).trimesh(spec, I.LABEL).export(other / "boss.stl")
    d = yaml.safe_load((tmp_path / "part.yaml").read_text())   # (`_part(tmp_path, None)` above wrote the plain part)
    d["regions"]["load"]["boss"] = "boss.stl"
    (other / "part.yaml").write_text(yaml.safe_dump(d))
    org2, _h2, shape2, labels2, ids2 = voxelize(load_config(other / "part.yaml"))
    assert np.array_equal(org, org2) and shape == shape2 and np.array_equal(labels, labels2)
    # a union shape also widens the grid: this boss reaches z = 17, over the design box at z = 14
    assert org[2] + h * shape[2] >= 17.0 - 1e-9
    assert labels.shape[2] > plain[3].shape[2]


def test_label_priority_keeps_a_bore_over_a_boss(tmp_path):
    boss = dict(BOSS, point=[12.0, 7.0, 0.0], radius_mm=3.0, length_mm=10.0)
    bore = dict(BORE, point=[12.0, 7.0, 10.0], radius_mm=1.6, length_mm=10.0)
    org, h, _shape, labels, _ids = voxelize(load_config(_part(tmp_path, [boss, bore])))
    ix, iy, iz = (int((12.5 - org[0]) / h), int((7.5 - org[1]) / h), int((5.5 - org[2]) / h))
    assert labels[ix, iy, iz] == post.KEEP_OUT       # near the axis: the bore wins over the boss
    assert labels[ix + 2, iy, iz] == post.LOAD       # in the wall (2.55 mm from the axis): boss, no bore


# -- one small run, every claim measured on the files ---------------------------------------------------------------
def _run_example(tmp_path_factory, yaml_name, **settings):
    """The example part (a coarser grid, 12 iterations, volume fraction 0.5: a thicker design that no smoothing splits) after `topopt run`. Return the run folder."""
    src = tmp_path_factory.mktemp("example_src")
    for f in EXAMPLE.iterdir():
        if f.suffix in (".stl", ".yaml"):
            shutil.copy(f, src)
    d = yaml.safe_load((src / yaml_name).read_text())
    d["settings"].update({"voxel_size": 1.5, "iterations": 12, "volume_fraction": 0.5, **settings})
    (src / yaml_name).write_text(yaml.safe_dump(d))
    out = tmp_path_factory.mktemp("example_run") / "out"
    assert main(["run", str(src / yaml_name), "--out", str(out), "--quiet"]) == 0
    return out


@pytest.fixture(scope="module")
def bracket(tmp_path_factory):
    """`bracket.yaml` (stock on the bore, the boss and the face) after a run."""
    return _run_example(tmp_path_factory, "bracket.yaml")


@pytest.fixture(scope="module")
def bracket_nofill(tmp_path_factory):
    """`bracket.yaml` as `bracket`, with the envelope fill off. For the test of the cut count: the fill is 4.46 % of the target on this part (just under the cap), and its
    mass-neutral payment moves the global level by -0.157 mm by design, so the shift of that test would not tell an uncounted cut."""
    return _run_example(tmp_path_factory, "bracket.yaml", envelope_fill="off")


@pytest.fixture(scope="module")
def bracket0(tmp_path_factory):
    """`bracket_nostock.yaml` (the same part, no stock) after a run."""
    return _run_example(tmp_path_factory, "bracket_nostock.yaml")


def _variant(src: Path, dst: Path, method=None, stock=None):
    """Copy a run folder, set the mesh method and the stock of every built-in interface, and post-process the copy."""
    shutil.copytree(src, dst)
    r = json.loads((dst / "run.json").read_text())
    if method:
        r["mesh_method"] = method
    if stock is not None:
        for it in r["interfaces"]:
            if "stock_mm" in it:
                it["stock_mm"] = stock
    (dst / "run.json").write_text(json.dumps(r))
    return post.finish(dst)


def _reload(path):
    return trimesh.load(path, file_type="stl")


def _closed_one_body(m):
    """The facts of a mesh from disk, computed here (not with `stl_facts`)."""
    tri = m.vertices[m.faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return m.is_watertight and m.is_winding_consistent and m.volume > 0 and len(m.split(only_watertight=False)) == 1 \
        and int((area == 0).sum()) == 0


def _wall(m, axis_xy, r, z0, z1, below):
    """Mean distance to the axis of the vertices within `below` mm under the radius `r`, between the heights `z0` and `z1`."""
    v = m.vertices
    rho = np.linalg.norm(v[:, :2] - axis_xy, axis=1)
    sel = (rho >= r - below) & (rho <= r + 1e-4) & (v[:, 2] >= z0) & (v[:, 2] <= z1)
    assert sel.sum() > 50, f"no wall near r = {r}"
    return float(rho[sel].mean())


def _interface_measures(final, nominal, bore_r, boss_r, face_x, stock):
    """Check each built-in interface of the example against the YAML, in both meshes. Everything from the vertices."""
    axis = np.array([28.0, 10.0])
    for m, s in ((final, stock), (nominal, 0.0)):
        if m is None:
            continue
        # bore (axis z at x 28, y 10): the wall of the bore has the vertices at the smallest distance from the axis
        # (its facets run the full height, so the vertices are on the two ends of the boss)
        v = m.vertices
        rho = np.linalg.norm(v[:, :2] - axis, axis=1)
        r_bore = bore_r - s
        assert rho.min() == pytest.approx(r_bore, abs=SAG_TOL)
        assert _wall(m, axis, r_bore, -3, 19, 0.01) == pytest.approx(r_bore, abs=SAG_TOL)
        assert _wall(m, axis, boss_r + s, -3, 19, 0.01) == pytest.approx(boss_r + s, abs=SAG_TOL)
        # face: the lowest vertex in x is the machined face (the part has no material at x < face_x - stock)
        assert v[:, 0].min() == pytest.approx(face_x - (0.5 if s else 0.0), abs=1e-4)


@pytest.mark.parametrize("method", ["sdf", "blur"])
def test_exact_interfaces_in_both_methods(bracket, tmp_path, method):
    m = _variant(bracket, tmp_path / "v", method=method)
    final = _reload(tmp_path / "v" / "result" / "final.stl")
    nominal = _reload(tmp_path / "v" / "result" / "nominal.stl")
    assert m["mesh"]["method"] == method and m["interfaces"]["applied"]
    for mesh in (final, nominal):                                         # the reload check, measured here
        assert _closed_one_body(mesh)
    _interface_measures(final, nominal, 3.0, 5.5, 2.0, 0.4)
    # the stock on each interface: the files show the YAML values
    axis = np.array([28.0, 10.0])
    stock_bore = _wall(nominal, axis, 3.0, -3, 19, 0.01) - _wall(final, axis, 2.6, -3, 19, 0.01)
    stock_boss = _wall(final, axis, 5.9, -3, 19, 0.01) - _wall(nominal, axis, 5.5, -3, 19, 0.01)
    stock_face = nominal.vertices[:, 0].min() - final.vertices[:, 0].min()
    assert stock_bore == pytest.approx(0.4, abs=2 * SAG_TOL)
    assert stock_boss == pytest.approx(0.4, abs=2 * SAG_TOL)
    assert stock_face == pytest.approx(0.5, abs=2e-4)
    assert final.volume > nominal.volume                                  # the as-printed part keeps the stock


@pytest.mark.parametrize("method", ["sdf", "blur"])
def test_no_stock_means_no_nominal_file(bracket0, tmp_path, method):
    m = _variant(bracket0, tmp_path / "v", method=method)
    res = tmp_path / "v" / "result"
    assert not (res / "nominal.stl").exists() and not (bracket0 / "result" / "nominal.stl").exists()
    rec = m["interfaces"]
    assert rec["stock"] is False and rec["nominal"] is None and rec["files"]["nominal.stl"] is None
    assert all("nominal" not in it for it in rec["items"]) and "stock_volume_mm3" not in rec
    final = _reload(res / "final.stl")
    assert _closed_one_body(final)
    _interface_measures(final, None, 3.0, 5.5, 2.0, 0.0)
    out = check_final_stl.check_run(tmp_path / "v")   # (the blur method does not hold the fixed regions exactly, so not out["ok"])
    assert out["interfaces"]["ok"] and out["interfaces"]["nominal_mesh"] is None and out["interfaces"]["problems"] == []
    assert m["mass_final_g"] == rec["as_printed"]["mass_g"] and rec["items"][1]["as_printed"]["measured"]["radius_mm"]["deviation_mm"] == pytest.approx(0, abs=SAG_TOL)


def test_a_stale_nominal_file_is_removed(bracket, tmp_path):
    shutil.copytree(bracket, tmp_path / "v")
    r = json.loads((tmp_path / "v" / "run.json").read_text())
    for it in r["interfaces"]:
        it["stock_mm"] = 0.0
    (tmp_path / "v" / "run.json").write_text(json.dumps(r))
    assert (tmp_path / "v" / "result" / "nominal.stl").exists()           # from the first post-process
    post.finish(tmp_path / "v")
    assert not (tmp_path / "v" / "result" / "nominal.stl").exists()
    out = check_final_stl.check_run(tmp_path / "v")
    assert out["interfaces"]["ok"] and out["interfaces"]["items"][0]["nominal"] is None


def test_run_json_and_metrics_record(bracket):
    run = json.loads((bracket / "run.json").read_text())
    assert [i["kind"] for i in run["interfaces"]] == ["boss", "bore", "face"]
    assert run["interfaces"][0]["stock_mm"] == 0.4 and "path" not in run["interfaces"][0]
    met = json.loads((bracket / "result" / "metrics.json").read_text())
    rec = met["interfaces"]
    assert rec["applied"] and rec["stock"] and rec["error"] is None and rec["sag_mm"] == I.SAG_MM
    assert rec["files"] == {"final.stl": "as_printed", "nominal.stl": "nominal"}
    final, nom = _reload(bracket / "result" / "final.stl"), _reload(bracket / "result" / "nominal.stl")
    # the existing mass keys refer to final.stl (as-printed); the record has both meshes
    assert met["volume_final_mm3"] == pytest.approx(final.volume, abs=0.1)
    assert met["mass_final_g"] == rec["as_printed"]["mass_g"] == pytest.approx(final.volume / 1000 * 1.24, abs=0.01)
    assert rec["as_printed"]["volume_mm3"] == pytest.approx(final.volume, abs=0.01)
    assert rec["nominal"]["volume_mm3"] == pytest.approx(nom.volume, abs=0.01)
    assert rec["stock_volume_mm3"] == pytest.approx(final.volume - nom.volume, abs=0.01)
    for part in ("as_printed", "nominal"):
        f = rec[part]["stl_roundtrip"]
        assert f["ok"] and f["bodies"] == 1 and f["zero_area_faces"] == 0 and f["open_edges"] == 0
    items = {i["name"]: i for i in rec["items"]}
    assert set(items) == {"bushing_seat", "pin_bore", "mounting_face"}
    boss, bore, face = items["bushing_seat"], items["pin_bore"], items["mounting_face"]
    assert (boss["kind"], bore["kind"], face["kind"]) == ("boss", "bore", "face")
    assert (boss["stock_mm"], bore["stock_mm"], face["stock_mm"]) == (0.4, 0.4, 0.5)
    assert boss["as_printed"]["volume_change_mm3"] > boss["nominal"]["volume_change_mm3"] > 0   # a bigger boss adds more
    assert bore["as_printed"]["volume_change_mm3"] > bore["nominal"]["volume_change_mm3"]       # both negative: less removed
    assert face["as_printed"]["volume_change_mm3"] > face["nominal"]["volume_change_mm3"]
    assert bore["as_printed"]["volume_change_mm3"] < 0 and face["nominal"]["volume_change_mm3"] < 0
    assert boss["as_printed"]["measured"]["radius_mm"]["measured_mm"] == pytest.approx(5.9, abs=SAG_TOL)
    assert bore["nominal"]["measured"]["radius_mm"]["measured_mm"] == pytest.approx(3.0, abs=SAG_TOL)
    assert face["as_printed"]["measured"]["position_mm"]["measured_mm"] == pytest.approx(0.5, abs=1e-4)
    assert bore["stock_measured_mm"]["radius_mm"] == pytest.approx(0.4, abs=2 * SAG_TOL)
    assert all(f["within_tolerance"] for it in rec["items"] for v in ("as_printed", "nominal") for f in it[v]["measured"].values())
    assert rec["t_s"]["total"] >= 0 and set(rec["t_s"]) >= {"base", "as_printed", "nominal", "total", "measure"}
    # the cutters are keep-out shapes of the region checks
    assert set(met["mesh"]["checks"]["keep_out"]) == {"interface:pin_bore", "interface:mounting_face"}
    assert all(v["intersect_mm3"] < 1e-3 for v in met["mesh"]["checks"]["keep_out"].values())


def test_the_calibration_counts_a_cut_over_a_united_region(bracket_nofill):
    """The face machines the outer 2 mm of the fixed plate. The exact union of the plate STL puts that part back before the cut,
    so the volume that the calibration measures must include the cut. Without it the shift is -0.2 mm or more and thins every member."""
    bracket = bracket_nofill                                             # the test is about counting a cut: the fill is off (see the fixture)
    met = json.loads((bracket / "result" / "metrics.json").read_text())["mesh"]
    cal = met["calibration"]
    assert abs(cal[-1]["error"]) < 5e-4                                  # the calibration reaches its target volume
    assert abs(cal[-1]["delta_mm"]) < 0.1                                # and with a small shift (the unit is mm)
    assert cal[-1]["final_mm3"] == pytest.approx(cal[-1]["target_mm3"], rel=5e-4)


def test_the_labels_of_the_run_hold_the_shapes(bracket):
    """The optimizer saw the boss as LOAD, the bore and the face as KEEP_OUT, and the density is 0 inside the cutters."""
    run = json.loads((bracket / "run.json").read_text())
    labels = np.frombuffer((bracket / "labels.bin").read_bytes(), np.uint8).reshape(tuple(run["shape"]))
    org, h = np.asarray(run["origin"]), run["h"]
    c = org + h * (np.indices(labels.shape).reshape(3, -1).T + 0.5)
    rho = np.linalg.norm(c[:, :2] - [28.0, 10.0], axis=1).reshape(labels.shape)
    z = c[:, 2].reshape(labels.shape)
    x = c[:, 0].reshape(labels.shape)
    assert (labels[(rho < 2.5) & (z > 0) & (z < 16)] == post.KEEP_OUT).all()           # the as-printed bore, r = 2.6
    assert (labels[(rho > 3.4) & (rho < 5.4) & (z > 0) & (z < 16)] == post.LOAD).all()   # the boss wall, nominal r = 5.5
    assert (labels[x < 1.4] == post.KEEP_OUT).all()                                     # beyond the as-printed face, x < 1.5


def test_check_tool_reports_and_fails(bracket, tmp_path, capsys):
    res = check_final_stl.check_run(bracket)
    ic = res["interfaces"]
    assert res["ok"] and ic["ok"] and ic["problems"] == []
    assert ic["nominal_mesh"]["ok"] and ic["nominal_mesh"]["bodies"] == 1 and ic["nominal_mesh"]["zero_area_faces"] == 0
    assert res["mesh"]["bodies"] == 1 and res["mesh"]["zero_area_faces"] == 0
    items = {i["name"]: i for i in ic["items"]}
    assert items["pin_bore"]["as_printed"]["radius_mm"]["measured_mm"] == pytest.approx(2.6, abs=SAG_TOL)
    assert items["pin_bore"]["stock_measured_mm"]["radius_mm"] == pytest.approx(0.4, abs=2 * SAG_TOL)
    assert items["mounting_face"]["stock_measured_mm"]["position_mm"] == pytest.approx(0.5, abs=2e-4)
    assert items["bushing_seat"]["nominal"]["radius_mm"]["deviation_mm"] == pytest.approx(0.0, abs=SAG_TOL)
    assert "interface:pin_bore" in res["keep_out"]
    assert check_final_stl.main([str(bracket)]) == 0
    text = capsys.readouterr().out
    assert "interface pin_bore (bore, as_printed) radius_mm: measured" in text and "stock in the files" in text and text.strip().endswith("OK")
    # a tolerance under the sag of the facets fails the boss
    assert not check_final_stl.check_run(bracket, tol_interface_mm=1e-9)["ok"]
    assert check_final_stl.main([str(bracket), "--tol-interface-mm", "1e-9"]) == 1
    # the as-printed file is replaced by the finished one: the bore is 0.4 mm too large
    bad = tmp_path / "swapped"
    shutil.copytree(bracket, bad)
    shutil.copy(bad / "result" / "nominal.stl", bad / "result" / "final.stl")
    r = check_final_stl.check_run(bad)
    assert not r["ok"] and any("pin_bore (as_printed)" in p for p in r["interfaces"]["problems"])
    # a missing nominal.stl with stock in the run
    gone = tmp_path / "gone"
    shutil.copytree(bracket, gone)
    (gone / "result" / "nominal.stl").unlink()
    r = check_final_stl.check_run(gone)
    assert not r["ok"] and any("nominal.stl is not there" in p for p in r["interfaces"]["problems"])


def test_the_check_tool_sees_a_filled_bore(bracket, tmp_path):
    """A final.stl whose bore is filled is still one closed body. The keep-out check and the post-time record find it."""
    bad = tmp_path / "filled"
    shutil.copytree(bracket, bad)
    mesh = _reload(bad / "result" / "final.stl")
    spec = Interface("pin", "bore", "cut", 0.0, (28.0, 10.0, 20.0), (0.0, 0.0, -1.0), 2.7, None, 24.0)   # fills the bore (r 2.6)
    solid = smooth_sdf._manifold_of(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)) + I.build_shape(spec, 0.0)
    v, f = smooth_sdf._from_manifold(solid)
    trimesh.Trimesh(v, f, process=False).export(bad / "result" / "final.stl")
    r = check_final_stl.check_run(bad)
    assert r["mesh"]["ok"] and r["mesh"]["bodies"] == 1                  # a good mesh by the reload check ...
    assert not r["ok"] and r["keep_out"]["interface:pin_bore"]["intersect_mm3"] > 10     # ... but with material in the bore
    assert any("pin_bore (as_printed) radius_mm: found at post time, not found" in p for p in r["interfaces"]["problems"])


def test_no_interfaces_leaves_no_trace(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=3)
    out = tmp_path / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert "interfaces" not in json.loads((out / "run.json").read_text())
    met = json.loads((out / "result" / "metrics.json").read_text())
    assert "interfaces" not in met and not (out / "result" / "nominal.stl").exists()
    assert "interfaces" not in check_final_stl.check_run(out)
    (out / "result" / "nominal.stl").write_bytes(b"old")        # a file of an older post-process is removed
    post.finish(out)
    assert not (out / "result" / "nominal.stl").exists()


def test_the_default_path_does_not_load_the_module(tmp_path):
    """A run without interfaces never imports `topopt.interfaces` (checked in a fresh Python)."""
    import subprocess

    p = make_case(tmp_path, h=2.0, iterations=2)
    code = ("import sys; from topopt.cli import main; "
            f"assert main(['run', {str(p)!r}, '--out', {str(tmp_path / 'o')!r}, '--quiet']) == 0; "
            "assert 'topopt.interfaces' not in sys.modules, 'imported'")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-600:]


def test_a_hard_case_reloads_clean(tmp_path):
    """The pinch case of the float32 reload (a face at 2.975 voxels, a hole near the plate), with the hole as an interface bore."""
    run, labels, _rho, org, h = _pinch_fixture(tmp_path)
    x0, y0, z0 = (float(c) for c in org)
    run["regions"]["keep_out"] = []                       # the hole is now the interface `hole`
    run["region_files"] = {"fixed": ["plate.stl"], "keep_out": [], "load": {"t": "load.stl"}}
    run["interfaces"] = [
        {"name": "hole", "kind": "bore", "op": "cut", "stock_mm": 0.2, "point": [x0 + 20 * h - 0.11, y0 + 30 * h, z0 + 12 * h],
         "axis": [0, -1, 0], "radius_mm": 3.5, "length_mm": 60.0},
        {"name": "top", "kind": "face", "op": "cut", "stock_mm": 0.11, "point": [0, 0, z0 + 24 * h - 0.25], "normal": [0, 0, 1]},
        {"name": "post", "kind": "boss", "op": "union", "stock_mm": 0.3, "point": [x0 + 27 * h - 1, y0 + 12 * h, z0 + 12 * h],
         "axis": [1, 0, 0], "radius_mm": 4.0, "length_mm": 5.0},
    ]
    (tmp_path / "run.json").write_text(json.dumps(run))
    m = post.finish(tmp_path)
    assert m["interfaces"]["applied"]
    for name in ("final.stl", "nominal.stl"):
        assert _closed_one_body(load_stl(tmp_path / "result" / name)), name
    for part in ("as_printed", "nominal"):
        f = m["interfaces"][part]["stl_roundtrip"]
        assert f["ok"] and f["bodies"] == 1 and f["zero_area_faces"] == 0 and f["open_edges"] == 0 and f["nonmanifold_edges"] == 0
    assert all(f["within_tolerance"] for it in m["interfaces"]["items"] for v in ("as_printed", "nominal")
               for f in it[v]["measured"].values())
    assert check_final_stl.check_run(tmp_path)["interfaces"]["ok"]


# -- fix round 1: tube hole, keep-out regions, floating shapes, coverage ------------------------------------------------
def _material_in(path, spec, stock, shrink=0.003):
    """Volume (mm3) of the file inside the cylinder `spec` (radius - stock - `shrink`): measured with manifold3d on the file."""
    import manifold3d as m3d

    from topopt.post import _to_manifold

    m = _reload(path)
    body = _to_manifold(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))
    r = spec.radius_mm - stock - shrink
    cyl = I._cylinder(spec.point, spec.axis, r, None, spec.length_mm)
    return float((body ^ cyl).volume())


TUBE_RUN = {"name": "sleeve", "kind": "tube", "point": [12.0, 7.0, 0.0], "axis": [0, 0, 1], "radius_mm": 4.0, "inner_radius_mm": 2.0,
            "length_mm": 14.0, "stock_mm": 0.3}


@pytest.fixture(scope="module")
def tube_run(tmp_path_factory):
    """A tube through the design volume, one run (a coarse grid, 8 iterations)."""
    tmp = tmp_path_factory.mktemp("tube")
    p = _part(tmp, [TUBE_RUN], iterations=8, vf=0.5)
    assert main(["run", str(p), "--out", str(tmp / "out"), "--quiet"]) == 0
    return tmp / "out", load_config(p)


def test_a_tube_run_keeps_its_hole(tube_run):
    """A tube is a ring and a hole. The hole is a KEEP_OUT label, it is free of material in both files, and both walls have their stock."""
    out, cfg = tube_run
    spec = cfg.interfaces[0]
    run = json.loads((out / "run.json").read_text())
    labels = np.frombuffer((out / "labels.bin").read_bytes(), np.uint8).reshape(tuple(run["shape"]))
    org, h = np.asarray(run["origin"]), run["h"]
    c = org + h * (np.indices(labels.shape).reshape(3, -1).T + 0.5)
    rho = np.linalg.norm(c[:, :2] - [12.0, 7.0], axis=1).reshape(labels.shape)
    z = c[:, 2].reshape(labels.shape)
    inside = (z > 0) & (z < 14)
    assert (labels[(rho < 1.6) & inside] == post.KEEP_OUT).all() and ((rho < 1.6) & inside).any()    # the hole, as-printed r 1.7
    assert (labels[(rho > 2.1) & (rho < 3.9) & inside] == post.LOAD).all()                           # the ring, nominal 2.0 to 4.0
    assert (labels[(rho > 1.75) & (rho < 1.95) & inside] == post.DESIGN).all() or not ((rho > 1.75) & (rho < 1.95) & inside).any()
    final, nominal = _reload(out / "result" / "final.stl"), _reload(out / "result" / "nominal.stl")
    assert _closed_one_body(final) and _closed_one_body(nominal)
    hole_as = I.Interface("h", "bore", "cut", 0.0, spec.point, spec.axis, spec.inner_radius_mm - spec.stock_mm, None, spec.length_mm)
    hole_nom = I.Interface("h", "bore", "cut", 0.0, spec.point, spec.axis, spec.inner_radius_mm, None, spec.length_mm)
    assert _material_in(out / "result" / "final.stl", hole_as, 0.0) <= 1e-3        # free of material inside r = 1.7
    assert _material_in(out / "result" / "nominal.stl", hole_nom, 0.0) <= 1e-3     # and inside r = 2.0
    axis = np.array([12.0, 7.0])
    assert _wall(final, axis, 1.7, -0.1, 14.1, 0.01) == pytest.approx(1.7, abs=SAG_TOL)
    assert _wall(nominal, axis, 2.0, -0.1, 14.1, 0.01) == pytest.approx(2.0, abs=SAG_TOL)
    assert _wall(final, axis, 4.3, -0.1, 14.1, 0.01) == pytest.approx(4.3, abs=SAG_TOL)
    assert _wall(nominal, axis, 4.0, -0.1, 14.1, 0.01) == pytest.approx(4.0, abs=SAG_TOL)
    met = json.loads((out / "result" / "metrics.json").read_text())
    item = met["interfaces"]["items"][0]
    assert set(item["as_printed"]["measured"]) == {"radius_mm", "inner_radius_mm"}
    assert item["stock_measured_mm"]["radius_mm"] == pytest.approx(0.3, abs=2 * SAG_TOL)
    assert item["stock_measured_mm"]["inner_radius_mm"] == pytest.approx(0.3, abs=2 * SAG_TOL)
    assert all(f["within_tolerance"] and f["coverage"] > 0.9 for v in ("as_printed", "nominal") for f in item[v]["measured"].values())
    assert met["mesh"]["checks"]["keep_out"]["interface:sleeve (hole)"]["intersect_mm3"] < 1e-3
    assert check_final_stl.check_run(out)["interfaces"]["ok"]
    assert set(I.cut_meshes(cfg, ((0, 0, 0), (24, 14, 14)))) == {"sleeve (hole)"} and set(I.add_meshes(cfg)) == {"sleeve"}


def test_a_filled_tube_hole_is_not_within_tolerance(tube_run, tmp_path):
    """Fill the hole of the tube in final.stl except the last 10 percent of its length: the wall is still found, the coverage is low."""
    out, cfg = tube_run
    spec = cfg.interfaces[0]
    shutil.copytree(out, tmp_path / "bad")
    mesh = _reload(tmp_path / "bad" / "result" / "final.stl")
    plug = I.Interface("p", "bore", "cut", 0.0, spec.point, spec.axis, 1.701, None, 0.9 * spec.length_mm)   # a hair over the wall: no slivers
    solid = smooth_sdf._manifold_of(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)) + I.build_shape(plug, 0.0)
    v, f = smooth_sdf._from_manifold(solid)
    trimesh.Trimesh(v, f, process=False).export(tmp_path / "bad" / "result" / "final.stl")
    res = check_final_stl.check_run(tmp_path / "bad")
    feat = next(i for i in res["interfaces"]["items"])["as_printed"]["inner_radius_mm"]
    assert feat["found"] and abs(feat["deviation_mm"]) < SAG_TOL        # a patch of wall is there, at the right radius ...
    assert feat["coverage"] < 0.2 and not feat["within_tolerance"]      # ... but it is 10 percent of the wall
    assert not res["interfaces"]["ok"] and any("covers" in p and "percent of its expected area" in p for p in res["interfaces"]["problems"])


def test_the_measure_looks_at_the_normal_side():
    """A bore wall faces the axis, a boss wall faces away, a face cut leaves the normal on the side of the cutter."""
    import manifold3d as m3d

    block = m3d.Manifold.cube((20, 10, 10))
    bore = Interface("b", "bore", "cut", 0.0, (10, 5, 12), (0, 0, -1), 2.0, None, 14.0)
    boss = Interface("s", "boss", "union", 0.0, (10, 5, 10), (0, 0, 1), 2.0, None, 4.0)
    face = Interface("f", "face", "cut", 0.0, (2, 0, 0), (-1, 0, 0))
    holed = I._arrays(block - I.build_shape(bore, 0.0))
    posted = I._arrays(block + I.build_shape(boss, 0.0))
    assert I._measure_radius(*holed, bore.point, bore.axis, 2.0, 14.0, outward=False)["found"]
    assert not I._measure_radius(*holed, bore.point, bore.axis, 2.0, 14.0, outward=True)["found"]      # a hole is not a boss
    assert I._measure_radius(*posted, boss.point, boss.axis, 2.0, 4.0, outward=True)["found"]
    assert not I._measure_radius(*posted, boss.point, boss.axis, 2.0, 4.0, outward=False)["found"]
    cut = I._arrays(block - I.build_shape(face, 0.0, ((0, 0, 0), (20, 10, 10))))    # the part at x >= 2: its face at x = 2 faces -x
    assert I._measure_plane(*cut, face.point, face.axis, 0.0)["found"]
    gone = I._arrays(m3d.Manifold.cube((12, 10, 10)).translate((-10, 0, 0)))          # the removed side: its face at x = 2 faces +x
    assert not I._measure_plane(*gone, face.point, face.axis, 0.0)["found"]


def test_the_coverage_of_a_correct_part_is_one():
    import manifold3d as m3d

    block = m3d.Manifold.cube((20, 10, 10))
    bore = Interface("b", "bore", "cut", 0.0, (10, 5, 12), (0, 0, -1), 2.0, None, 14.0)
    boss = Interface("s", "boss", "union", 0.0, (15, 5, 10), (0, 0, 1), 1.5, None, 4.0)
    face = Interface("f", "face", "cut", 0.0, (2, 0, 0), (-1, 0, 0))
    box_ = ((0, 0, 0), (20, 10, 10))
    body = block + I.build_shape(boss, 0.0)
    body = body - I.build_shape(bore, 0.0)
    body = body - I.build_shape(face, 0.0, box_)
    v, f = I._arrays(body)
    mb = I.mesh_body(v, f)
    for spec in (bore, boss, face):
        for feat in I.measure(spec, I.AS_PRINTED, v, f, body=mb, box=box_).values():
            assert feat["found"] and feat["within_tolerance"] and feat["coverage"] == pytest.approx(1.0, abs=0.01)
    # a bore in empty space has no expected area: not found, no coverage, and it is no failure of the coverage test
    far = Interface("far", "bore", "cut", 0.0, (50, 5, 12), (0, 0, -1), 2.0, None, 14.0)
    feat = I.measure(far, I.AS_PRINTED, v, f, body=mb, box=box_)["radius_mm"]
    assert not feat["found"] and feat["coverage"] is None and feat["expected_area_mm2"] < I.AREA_MIN_MM2


def test_a_cut_that_takes_part_of_a_boss_wall_does_not_lower_its_coverage():
    """A counterbore across the top of a boss removes the wall there: the rest of the wall is the whole wall that is there to find."""
    import manifold3d as m3d

    block = m3d.Manifold.cube((20, 10, 10))
    boss = Interface("s", "boss", "union", 0.0, (15, 5, 8), (0, 0, 1), 1.5, None, 6.0)     # z 8 to 14: 2 mm in the block
    cbore = Interface("c", "bore", "cut", 0.0, (15, 5, 14), (0, 0, -1), 2.5, None, 3.0)      # z 11 to 14, wider than the boss
    box_ = ((0, 0, 0), (20, 10, 14))
    cut = I.build_shape(cbore, 0.0)
    body = (block + I.build_shape(boss, 0.0)) - cut
    v, f = I._arrays(body)
    mb = I.mesh_body(v, f)
    plain = I.measure(boss, I.AS_PRINTED, v, f, body=mb, box=box_)["radius_mm"]
    assert plain["found"] and plain["coverage"] < 0.6        # the cut took the top half of the wall, and it is counted as missing
    right = I.measure(boss, I.AS_PRINTED, v, f, body=mb, box=box_, removed=cut)["radius_mm"]
    assert right["found"] and right["within_tolerance"] and right["coverage"] == pytest.approx(1.0, abs=0.02)


KEEP_BOX = ((9.0, 8.0, 3.0), (15.0, 16.0, 7.0))     # a slot in the side of the design volume (y 8 to 16), through the boss below
BOSS_OVER = {"name": "seat", "kind": "boss", "point": [12.0, 5.0, 0.0], "axis": [0, 0, 1], "radius_mm": 4.5, "length_mm": 12.0}


@pytest.mark.parametrize("method", ["sdf", "blur"])
def test_a_boss_over_a_keep_out_region_is_cut_back(tmp_path, caplog, method):
    """The label rule: a keep-out region wins over a union interface. The exact step cuts the region out of the union."""
    p = _part(tmp_path, [BOSS_OVER], iterations=6, vf=0.5, keep_out=KEEP_BOX)
    with caplog.at_level(logging.WARNING):
        assert main(["run", str(p), "--out", str(tmp_path / "out"), "--quiet", "--mesh-method", method]) == 0
    assert any("union interface 'seat' overlaps keep-out region pocket.stl in" in r.message and "voxels" in r.message for r in caplog.records)
    met = json.loads((tmp_path / "out" / "result" / "metrics.json").read_text())
    rec = met["interfaces"]
    assert rec["applied"] and rec["keep_out_cut"]["as_printed"]["removed_mm3"] > 1.0     # the boss did reach into the pocket
    res = check_final_stl.check_run(tmp_path / "out")
    assert res["keep_out"]["pocket.stl"]["intersect_mm3"] <= 1e-3
    assert _closed_one_body(_reload(tmp_path / "out" / "result" / "final.stl"))
    if method == "sdf":
        assert met["mesh"]["checks"]["keep_out"]["pocket.stl"]["intersect_mm3"] <= 1e-3
    # the pocket is empty in the labels too
    run = json.loads((tmp_path / "out" / "run.json").read_text())
    labels = np.frombuffer((tmp_path / "out" / "labels.bin").read_bytes(), np.uint8).reshape(tuple(run["shape"]))
    org, h = np.asarray(run["origin"]), run["h"]
    ix, iy, iz = (int((12.5 - org[0]) / h), int((9.5 - org[1]) / h), int((5.5 - org[2]) / h))     # inside the boss and the pocket
    assert labels[ix, iy, iz] == post.KEEP_OUT


def test_a_missing_keep_out_file_is_a_fact(tmp_path, caplog):
    """If the keep-out STL is not there, the exact step cannot cut it back. It says so."""
    p = _part(tmp_path, [BOSS_OVER], iterations=3, vf=0.5, keep_out=KEEP_BOX)
    assert main(["run", str(p), "--out", str(tmp_path / "out"), "--quiet"]) == 0
    (tmp_path / "pocket.stl").unlink()
    with caplog.at_level(logging.WARNING, logger="topopt.interfaces"):
        met = post.finish(tmp_path / "out")
    assert met["interfaces"]["applied"] and "keep_out_cut" not in met["interfaces"]
    assert any("keep-out region file(s) pocket.stl not found" in r.message for r in caplog.records)


FLOATER = {"name": "floater", "kind": "boss", "point": [30.0, 7.0, 3.0], "axis": [0, 0, 1], "radius_mm": 2.0, "length_mm": 6.0}


def test_a_floating_union_shape_is_a_fact_in_check_and_an_error_in_run(tmp_path, capsys):
    """A boss that touches no design, fixed or load voxel makes the stiffness matrix singular: check warns, run stops before the solve."""
    p = _part(tmp_path, [BOSS, FLOATER], iterations=3, vf=0.5)    # BOSS (x 5, y 5) is in the design box, FLOATER is 6 mm outside
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert "warning: interface 'floater' (boss) holds" in out and "no fixed voxel" in out and "Options: move the shape" in out
    assert "interface:      floater (boss, union):" in out and "interface:      s (boss, union):" in out
    assert "interface 's'" not in out.split("interface:")[0]       # the attached boss gets no warning
    assert main(["run", str(p), "--out", str(tmp_path / "out"), "--quiet"]) == 1
    err = capsys.readouterr().err
    assert "interface 'floater' (boss) holds" in err and "stiffness matrix is singular" in err
    assert not (tmp_path / "out" / "result").exists() or not (tmp_path / "out" / "result" / "final.stl").exists()
    assert main(["check", str(_part(tmp_path, [BOSS], iterations=3))]) == 0
    assert "no fixed voxel" not in capsys.readouterr().out


def test_the_error_of_an_empty_design_names_the_interfaces(tmp_path):
    """A face with the normal on the part side takes every design voxel. The error names the face and the count."""
    face = {"name": "flipped", "kind": "face", "point": [-1.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]}
    with pytest.raises(UserError) as exc:
        voxelize(load_config(_part(tmp_path, [face])))
    text = str(exc.value)
    assert "no free voxel left" in text and "flipped (cut)" in text and "Check the `normal` of a face" in text


def test_a_body_next_to_a_union_shape_is_judged(tmp_path):
    """The sdf step keeps an island next to a union shape (it pins the shape). The exact union can leave it floating: the step judges it."""
    box(tmp_path / "fixed.stl", (-2, 0, 0), (0, 10, 10))
    v, f = _block((0, 0, 0), (20, 10, 10))
    v2, f2 = _block((23.0, 4, 4), (23.9, 6, 6))        # an island 0.05 mm from the boss wall below (the boss ends at x = 22.95)
    bodies = trimesh.util.concatenate([trimesh.Trimesh(v, f), trimesh.Trimesh(v2, f2)])
    boss = Interface("s", "boss", "union", 0.0, (20.0, 5, 5), (1, 0, 0), 1.5, None, 2.95)   # sticks out of the end face
    for near, want in ((0.0, 2), (0.5, 1)):
        specs = [boss]
        ctx = I.Context(specs, ((-5, -5, -5), (30, 25, 25)), I.Shapes(specs, ((-5, -5, -5), (30, 25, 25))), False,
                        [tmp_path / "fixed.stl"], near_mm=near)
        _v, va, fa = None, *I.apply_step(ctx, bodies.vertices, bodies.faces)[:2]
        assert len(trimesh.Trimesh(va, fa, process=False).split(only_watertight=False)) == want


def test_a_hand_edited_run_json_is_normalized_and_checked(tmp_path):
    d = dict(BORE, op="cut", axis=[0, 0, -3.0], stock_mm=0.0)
    spec = Interface.from_dict(d)
    assert spec.axis == (0.0, 0.0, -1.0)
    unit = Interface.from_dict(dict(d, axis=[0.0, 0.6, 0.8]))
    assert unit.axis == (0.0, 0.6, 0.8)                                  # a unit vector stays as it is
    with pytest.raises(UserError, match="'axis' must not be the zero vector"):
        Interface.from_dict(dict(d, axis=[0, 0, 0]))
    # a union shape outside the grid box (the loader makes the grid hold it; a hand edit does not) is a fact
    run = {"yaml_path": str(tmp_path / "x.yaml"), "regions": {}, "interfaces": [
        {"name": "out", "kind": "boss", "op": "union", "stock_mm": 0.0, "point": [60.0, 5.0, 5.0], "axis": [0, 0, 1], "radius_mm": 2.0, "length_mm": 3.0}]}
    ctx = I.from_run(run, (0, 0, 0), 1.0, (20, 10, 10))
    assert any("interface 'out' lies partly outside the grid box" in n for n in ctx.notes)
