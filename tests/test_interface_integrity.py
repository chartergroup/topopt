"""What stays of a frozen region, a surface that must be complete, and a stock with no effect (H1 PR 3; CONTRACTS.md section 23).

Check C: `interfaces.integrity` gives the share of each frozen region (fixed, load, solid, union interface) that the finished mesh keeps,
after the bores and tube holes are taken out of the region. Check D: `full_surface: true` on a bore, boss, tube or bounded face gives
`full_area_mm2` and `full_coverage`, and one test (0.98). Check E: `stock_mm` with no effect is a warning.
The fixtures are small meshes made with the exact step, so no optimizer runs for them.
"""
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from test_interface_cuts import write_case
from test_interfaces import EXAMPLE, _block, _run_example
from test_optimize import box
from topopt import interfaces as I
from topopt.cli import main
from topopt.config import Interface, UserError, load_config

BOX = ((-5.0, -5.0, -5.0), (65.0, 25.0, 25.0))


def _make(specs, verts, faces, regions=(), taken=(), box_=BOX):
    """Run the exact step on a mesh and write the outputs of a run. Return `(record, folder-less context, result)`."""
    box_ = tuple(np.asarray(b, float) for b in box_)
    ctx = I.Context(list(specs), box_, I.Shapes(specs, box_), any(s.kind != "stl" and s.stock_mm > 0 for s in specs),
                    regions=list(regions), taken=list(taken))
    v, f, res = I.apply_step(ctx, verts, faces)
    assert res.applied, res.error
    return ctx, v, f, res


def _outputs(tmp_path, ctx, v, f, res):
    data = trimesh.Trimesh(v, f, process=False).export(file_type="stl")
    return I.write_outputs(res, tmp_path, data, 1.24)


def _mesh_of(man):
    v, f = I._arrays(man)
    return v, f


def _cube(lo, hi):
    import manifold3d as m3d
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


# -- check C: what stays of a frozen region ------------------------------------------------------------------
def test_a_region_cut_by_an_unbounded_face_warns(tmp_path, caplog):
    """A bar x 24..60 is a load region. A face at x = 30 cuts the half-space past it: 6 of 36 mm stay."""
    bar = tmp_path / "bar.stl"
    box(bar, (24, 2, 2), (60, 8, 8))
    v, f = _block((0, 0, 0), (60, 10, 10))
    face = Interface("end", "face", "cut", 0.0, (30, 5, 5), (1, 0, 0))
    taken = [{"cut": "end", "cut_kind": "face", "region": "bar", "region_role": "load", "warn": True}]
    ctx, v2, f2, res = _make([face], v, f, regions=[("load", "bar", bar)], taken=taken)
    with caplog.at_level(logging.WARNING, logger="topopt.interfaces"):
        rec = _outputs(tmp_path, ctx, v2, f2, res)
    [r] = rec["integrity"]
    assert (r["region"], r["role"]) == ("bar", "load")
    assert r["kept"] == pytest.approx(6 / 36, abs=1e-3) and r["warn"] is True
    assert r["volume_mm3"] == pytest.approx(36 * 36, rel=1e-6) and r["lost_mm3"] == pytest.approx(30 * 36, rel=1e-3)
    assert r["cuts"] == [{"name": "end", "kind": "face", "role": "loss"}]
    [w] = rec["warnings"]
    assert w.startswith("load region 'bar' keeps 16.7 % of its volume in final.stl (the limit is 97 %): 1080 of 1296 mm3 are missing.")
    assert "end (face, loss)" in w and "Options: bound the cut (a face takes 'radius_mm')" in w
    keys, sentence = I.verdict_extra(rec)
    assert keys["region_integrity"] == [{"region": "bar", "role": "load", "kept": r["kept"], "threshold": 0.97}]
    assert "bar 0.167 of its volume (threshold 0.97: not met)" in sentence and "interface_tests" not in keys


def test_a_region_that_the_mesh_keeps_has_no_warning(tmp_path):
    bar = tmp_path / "bar.stl"
    box(bar, (24, 2, 2), (28, 8, 8))
    v, f = _block((0, 0, 0), (60, 10, 10))
    face = Interface("end", "face", "cut", 0.0, (30, 5, 5), (1, 0, 0))
    ctx, v2, f2, res = _make([face], v, f, regions=[("load", "bar", bar)])
    rec = _outputs(tmp_path, ctx, v2, f2, res)
    assert rec["integrity"][0]["kept"] == pytest.approx(1.0, abs=1e-4) and "warnings" not in rec
    assert I.verdict_extra(rec) == ({}, "")


def test_a_boss_with_a_coaxial_bore_does_not_warn(tmp_path):
    """The bore takes the middle of the boss: that is a hole the YAML asks for, not a loss. The hole is taken out of the boss first."""
    v, f = _block((0, 0, 0), (20, 10, 10))
    boss = Interface("s", "boss", "union", 0.0, (10, 5, 10), (0, 0, 1), 3.0, None, 4.0)
    bore = Interface("b", "bore", "cut", 0.0, (10, 5, 15), (0, 0, -1), 1.5, None, 12.0)
    ctx, v2, f2, res = _make([boss, bore], v, f)
    rec = _outputs(tmp_path, ctx, v2, f2, res)
    [r] = rec["integrity"]
    assert (r["region"], r["role"]) == ("s", "interface")
    assert r["kept"] == pytest.approx(1.0, abs=1e-4) and r["warn"] is False and "warnings" not in rec
    ring = (np.pi * 9.0 - np.pi * 2.25) * 4.0       # the boss minus the hole, about (polygons)
    assert r["volume_mm3"] == pytest.approx(ring, rel=0.01)


def test_a_fixed_region_with_a_hole_inside_it_keeps_its_share(tmp_path):
    """A fixed plate with a bore through it: kept is 1 after the hole is taken out; a keep-out cut across it would not be."""
    plate = tmp_path / "plate.stl"
    box(plate, (0, 0, 0), (20, 10, 10))
    v, f = _block((0, 0, 0), (20, 10, 10))
    bore = Interface("b", "bore", "cut", 0.0, (10, 5, 12), (0, 0, -1), 2.0, None, 14.0)
    ctx, v2, f2, res = _make([bore], v, f, regions=[("fixed", "plate", plate)])
    [r] = _outputs(tmp_path, ctx, v2, f2, res)["integrity"]
    assert r["role"] == "fixed" and r["kept"] == pytest.approx(1.0, abs=1e-4)
    # the same mesh, measured with no hole subtracted, would lose the cylinder
    ctx0 = I.Context([], ctx.box, I.Shapes([], ctx.box), False, regions=[("fixed", "plate", plate)])
    [r0] = I.integrity(ctx0, v2, f2)
    assert r0["kept"] == pytest.approx(1 - np.pi * 4.0 * 10 / 2000, abs=0.01) and r0["warn"] is True


def test_a_region_file_that_cannot_be_read_is_a_fact(tmp_path):
    bad = tmp_path / "bad.stl"
    trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False).export(bad)       # one triangle: not a closed manifold
    v, f = _block((0, 0, 0), (20, 10, 10))
    ctx = I.Context([], BOX, I.Shapes([], BOX), False, regions=[("load", "bad", bad)])
    [r] = I.integrity(ctx, v, f)
    assert r["kept"] is None and r["warn"] is False and r["note"].startswith("not measured")


# -- check D: a surface that must be complete ----------------------------------------------------------------------
def _socket(tmp_path, x_lo, full=True, stock=0.0, hole=False):
    """A block x `x_lo`..30, 20 mm thick (z 0..10) with a bore of radius 3 at x = 10 through its thickness. `hole`: the block has the hole already."""
    body = _cube((x_lo, 0, 0), (30, 20, 10))
    if hole:
        body = body - I._cylinder((10, 10, 12), (0, 0, -1), 3.0, None, 14.0)
    v, f = _mesh_of(body)
    bore = Interface("sock", "bore", "cut", stock, (10, 10, 10), (0, 0, -1), 3.0, None, 10.0, full_surface=full)
    ctx, v2, f2, res = _make([bore], v, f, box_=((-5, -5, -5), (35, 25, 15)))
    return _outputs(tmp_path, ctx, v2, f2, res)


def test_full_surface_on_a_whole_socket_is_met(tmp_path):
    rec = _socket(tmp_path, x_lo=0)
    [feat] = [f for f in rec["items"][0]["as_printed"]["measured"].values()]
    assert feat["full_area_mm2"] == pytest.approx(2 * np.pi * 3.0 * 10.0, abs=0.01)
    assert feat["full_coverage"] == pytest.approx(1.0, abs=0.005)
    [t] = rec["full_surface_tests"]
    assert (t["interface"], t["variant"], t["threshold"], t["met"]) == ("sock", "as_printed", 0.98, True)
    assert "warnings" not in rec
    keys, sentence = I.verdict_extra(rec)
    assert keys["interface_tests"] == rec["full_surface_tests"]
    assert sentence == " Interface tests (full surface): sock 1 of its full area (threshold 0.98: met)."


def test_full_surface_on_a_half_cut_socket_is_not_met(tmp_path, caplog):
    """The block ends at the bore axis (x = 10): half of the wall has no material. `coverage` says 1; `full_coverage` says 0.5."""
    with caplog.at_level(logging.WARNING, logger="topopt.interfaces"):
        rec = _socket(tmp_path, x_lo=10)
    feat = rec["items"][0]["as_printed"]["measured"]["radius_mm"]
    assert feat["coverage"] == pytest.approx(1.0, abs=0.02)             # the existing number does not count the missing wall
    assert feat["full_coverage"] == pytest.approx(0.5, abs=0.01)
    [t] = rec["full_surface_tests"]
    assert t["met"] is False and t["full_coverage"] == pytest.approx(0.5, abs=0.01)
    [w] = rec["warnings"]
    assert w.startswith("interface 'sock' has full_surface, and the as_printed mesh shows 0.5 of its full area (the limit is 0.98): ")
    assert "Options: change the region or the interface" in w and w not in caplog.messages     # cli.finish writes it to run.log, once
    assert "sock 0.5 of its full area (threshold 0.98: not met)" in I.verdict_extra(rec)[1]


def test_no_full_surface_gives_the_old_keys(tmp_path):
    rec = _socket(tmp_path, x_lo=10, full=False)
    feat = rec["items"][0]["as_printed"]["measured"]["radius_mm"]
    assert "full_area_mm2" not in feat and "full_coverage" not in feat
    assert "full_surface" not in rec["items"][0] and "full_surface_tests" not in rec and "warnings" not in rec


def test_full_surface_of_a_tube_and_a_bounded_face(tmp_path):
    """A tube has two surfaces (the lowest counts); a bounded face has the area of its disc."""
    v, f = _mesh_of(_cube((0, 0, 0), (20, 20, 10)))
    tube = Interface("t", "tube", "union", 0.0, (10, 10, 10), (0, 0, 1), 4.0, 2.0, 5.0, full_surface=True)
    face = Interface("e", "face", "cut", 0.0, (10, 10, 5), (0, 0, 1), 3.0, full_surface=True)
    ctx, v2, f2, res = _make([tube, face], v, f, box_=((-5, -5, -5), (25, 25, 20)))
    rec = _outputs(tmp_path, ctx, v2, f2, res)
    items = {i["name"]: i for i in rec["items"]}
    assert set(items["t"]["as_printed"]["measured"]) == {"radius_mm", "inner_radius_mm"}
    assert items["e"]["as_printed"]["measured"]["position_mm"]["full_area_mm2"] == pytest.approx(np.pi * 9.0, abs=0.01)
    tests = {t["interface"]: t for t in rec["full_surface_tests"]}
    assert set(tests) == {"t", "e"} and len(tests["t"]["surfaces"]) == 2
    assert tests["e"]["met"] is True and tests["e"]["full_coverage"] == pytest.approx(1.0, abs=0.02)


# -- check E: a stock with no effect -----------------------------------------------------------------------------------
def test_a_stock_with_no_effect_warns(tmp_path, caplog):
    """The block has the hole at the finished radius already (the region starts at the finished size): the as-printed bore of r - 0.3 finds no wall."""
    with caplog.at_level(logging.WARNING, logger="topopt.interfaces"):
        rec = _socket(tmp_path, x_lo=0, full=False, stock=0.3, hole=True)
    ap = rec["items"][0]["as_printed"]["measured"]["radius_mm"]
    nom = rec["items"][0]["nominal"]["measured"]["radius_mm"]
    assert ap.get("area_mm2", 0.0) < 0.05 * nom["area_mm2"] and nom["area_mm2"] == pytest.approx(2 * np.pi * 3.0 * 10.0, rel=0.01)
    [w] = rec["warnings"]
    assert w.startswith("interface 'sock': stock_mm 0.3 has no effect: the as-printed part has ")
    assert f"mm2 of this surface against {nom['area_mm2']:.0f} mm2 finished. The region next to it starts at the finished size." in w
    assert w.endswith("Make the region reach the as-printed size, or use a 'tube' interface, which adds the stock")


def test_a_stock_that_acts_has_no_warning(tmp_path):
    """The block is solid there: both bores cut a wall, the as-printed wall (r = 2.7) has 90 % of the area of the finished one."""
    rec = _socket(tmp_path, x_lo=0, full=False, stock=0.3, hole=False)
    ap = rec["items"][0]["as_printed"]["measured"]["radius_mm"]["area_mm2"]
    nom = rec["items"][0]["nominal"]["measured"]["radius_mm"]["area_mm2"]
    assert ap / nom == pytest.approx(0.9, abs=0.02)
    assert "warnings" not in rec and I.stock_effect_warnings(rec["items"]) == []


def test_no_stock_means_no_check_e(tmp_path):
    assert I.stock_effect_warnings([{"name": "x", "kind": "bore", "stock_mm": 0.0}]) == []
    assert I.stock_effect_warnings([{"name": "x", "kind": "stl", "stock_mm": 0.0}]) == []


# -- the loader ------------------------------------------------------------------------------------------------------------
def _cfg(tmp_path, entry):
    p = write_case(tmp_path, extra=[entry])
    d = yaml.safe_load(p.read_text())
    d["interfaces"] = [entry]
    p.write_text(yaml.safe_dump(d))
    return p


BORE = {"name": "b", "kind": "bore", "point": [5, 5, 12], "axis": [0, 0, -1], "radius_mm": 2.0, "length_mm": 8.0}
BOSS = {"name": "s", "kind": "boss", "point": [5, 5, -1], "axis": [0, 0, 1], "radius_mm": 3.0, "length_mm": 6.0}
TUBE = {"name": "t", "kind": "tube", "point": [5, 5, -1], "axis": [0, 0, 1], "radius_mm": 4.0, "inner_radius_mm": 2.0, "length_mm": 6.0}
FACE = {"name": "f", "kind": "face", "point": [1, 0, 0], "normal": [-1, 0, 0]}


@pytest.mark.parametrize("entry", [BORE, BOSS, TUBE, dict(FACE, radius_mm=5.0)], ids=["bore", "boss", "tube", "bounded face"])
def test_full_surface_is_allowed_where_the_surface_can_be_measured(tmp_path, entry):
    [spec] = load_config(_cfg(tmp_path, dict(entry, full_surface=True))).interfaces
    assert spec.full_surface is True
    d = spec.to_dict(tmp_path)
    assert d["full_surface"] is True and Interface.from_dict(json.loads(json.dumps(d)), tmp_path) == spec
    [off] = load_config(_cfg(tmp_path, entry)).interfaces
    assert off.full_surface is False and "full_surface" not in off.to_dict(tmp_path)      # run.json: the key only when true
    [off2] = load_config(_cfg(tmp_path, dict(entry, full_surface=False))).interfaces
    assert off2 == off


def test_full_surface_on_a_face_without_radius_is_a_loader_error(tmp_path):
    with pytest.raises(UserError) as e:
        load_config(_cfg(tmp_path, dict(FACE, full_surface=True)))
    msg = str(e.value)
    assert "'interfaces[0].full_surface' is true on a face with no 'radius_mm'" in msg and "cuts a half-space" in msg
    assert "Options: add 'radius_mm'" in msg


def test_full_surface_on_an_stl_solid_is_a_loader_error(tmp_path):
    box(tmp_path / "lug.stl", (0, 0, 0), (1, 1, 1))
    entry = {"name": "lug", "kind": "stl", "file": "lug.stl", "op": "union", "full_surface": True}
    with pytest.raises(UserError, match=r"'interfaces\[0\]\.full_surface' is true on an STL solid"):
        load_config(_cfg(tmp_path, entry))
    entry["full_surface"] = False       # false is a no-op
    assert load_config(_cfg(tmp_path, entry)).interfaces[0].kind == "stl"


@pytest.mark.parametrize("bad", ["yes", 1, 0, None, [True]])
def test_full_surface_must_be_a_bool(tmp_path, bad):
    with pytest.raises(UserError, match=r"'interfaces\[0\]\.full_surface' must be true or false, got"):
        load_config(_cfg(tmp_path, dict(BORE, full_surface=bad)))


# -- a run: the log, metrics.json, the verdict and run_result ------------------------------------------------------------------
def test_a_run_writes_the_integrity_record_the_log_and_the_verdict(tmp_path):
    """The bar of `write_case` reaches 30 mm past the face: the load region `sleeve` keeps little of its volume."""
    p = write_case(tmp_path, collar=(24, 60))
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    met = json.loads((out / "result" / "metrics.json").read_text())
    rec = met["interfaces"]
    by = {r["region"]: r for r in rec["integrity"]}
    assert set(by) == {"plate", "sleeve"} and by["plate"]["kept"] == pytest.approx(1.0, abs=1e-3) and by["plate"]["warn"] is False
    assert by["sleeve"]["warn"] is True and by["sleeve"]["kept"] < 0.4 and by["sleeve"]["cuts"] == [{"name": "end", "kind": "face", "role": "loss"}]
    [w] = rec["warnings"]
    assert w.startswith("load region 'sleeve' keeps ") and "of its volume in final.stl (the limit is 97 %)" in w
    assert (out / "run.log").read_text().count("warning: " + w) == 1
    v = met["verdict"]
    assert v["region_integrity"][0]["region"] == "sleeve" and "Frozen regions that keep less than the limit: sleeve " in v["text"]
    assert v["state"] in ("converged", "iteration_limit") and "interface_tests" not in v
    assert (out / "verdict.json").read_text() and "region_integrity" not in json.loads((out / "verdict.json").read_text())   # the run's own file is unchanged
    pytest.importorskip("mcp")
    from topopt_mcp import server
    got = server.run_result(str(out), preview=False)
    assert got["interface_warnings"] == [w]


def test_a_run_with_full_surface_has_the_test_in_the_verdict(tmp_path):
    p = write_case(tmp_path, collar=(24, 31.5), face={"radius_mm": 5.0}, extra=[
        {"name": "seat", "kind": "bore", "point": [20, 5, 12], "axis": [0, 0, -1], "radius_mm": 1.5, "length_mm": 14.0, "full_surface": True}])
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    met = json.loads((out / "result" / "metrics.json").read_text())
    [t] = met["verdict"]["interface_tests"]
    assert t["interface"] == "seat" and t["threshold"] == 0.98 and t["met"] is (t["full_coverage"] >= 0.98)
    assert "Interface tests (full surface): seat " in met["verdict"]["text"]
    assert json.loads((out / "run.json").read_text())["interfaces"][1]["full_surface"] is True


# -- no new key, no change ------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bracket(tmp_path_factory):
    return _run_example(tmp_path_factory, "bracket.yaml")


def test_the_bracket_example_changes_only_by_the_additive_keys(bracket):
    """`bracket.yaml` has no `full_surface`: run.json, the items and the verdict keep their keys; the record gains `integrity` only.

    The meshes were compared byte for byte with the code of the commit before this change (1d5dc51) when this test was written
    (final.stl and nominal.stl; metrics.json equal apart from `integrity` and the time of that step).
    """
    run = json.loads((bracket / "run.json").read_text())
    assert all("full_surface" not in i for i in run["interfaces"])
    met = json.loads((bracket / "result" / "metrics.json").read_text())
    rec = met["interfaces"]
    assert set(rec) == {"applied", "error", "count", "stock", "sag_mm", "t_s", "files", "as_printed", "nominal", "stock_volume_mm3",
                        "clean", "bodies_removed", "label_volume_error_pct", "items", "integrity"}
    assert "full_surface_tests" not in rec
    # the machined face `mounting_face` trims the clamp plate within the check A limit: a trim, not a loss
    assert "warnings" not in rec
    for it in rec["items"]:
        assert "full_surface" not in it
        for v in ("as_printed", "nominal"):
            for feat in it[v]["measured"].values():
                assert "full_area_mm2" not in feat and "full_coverage" not in feat
    assert not {"interface_tests", "region_integrity"} & set(met["verdict"])
    by = {r["region"]: r for r in rec["integrity"]}
    assert set(by) == {"plate", "tip", "bushing_seat"} and not by["plate"]["warn"] and by["plate"]["cuts"] == [{"name": "mounting_face", "kind": "face", "role": "trim"}]
    assert all(not r["warn"] and r["kept"] > 0.99 for r in by.values())


# -- a trim within the check A limit is not a loss ---------------------------------------------------------------------------
def _trim_case(tmp_path, bar_end):
    """A bar x 24..`bar_end` is a load region; a face at x = 30 cuts past it. `taken` is what check A says for the pair."""
    bar = tmp_path / "bar.stl"
    box(bar, (24, 2, 2), (bar_end, 8, 8))
    v, f = _block((0, 0, 0), (60, 10, 10))
    face = Interface("end", "face", "cut", 0.0, (30, 5, 5), (1, 0, 0))
    depth = bar_end - 30.0
    taken = [{"cut": "end", "cut_kind": "face", "region": "bar", "region_role": "load", "depth_mm": depth, "trim_mm": 2.0,
              "warn": depth > 2.0}]
    ctx, v2, f2, res = _make([face], v, f, regions=[("load", "bar", bar)], taken=taken)
    return _outputs(tmp_path, ctx, v2, f2, res)


def test_a_region_trimmed_within_the_limit_gives_no_warning(tmp_path):
    rec = _trim_case(tmp_path, 31.5)       # 1.5 mm past the plane: a trim (the raw share would be 0.8)
    [r] = rec["integrity"]
    assert r["kept"] == pytest.approx(1.0, abs=1e-4) and r["warn"] is False and "warnings" not in rec
    assert r["cuts"] == [{"name": "end", "kind": "face", "role": "trim"}]


def test_the_same_face_cutting_deep_still_warns(tmp_path):
    rec = _trim_case(tmp_path, 33.0)       # 3 mm past the plane: check A warned, so it stays a loss
    [r] = rec["integrity"]
    assert r["kept"] == pytest.approx(6 / 9, abs=1e-3) and r["warn"] is True
    assert r["cuts"] == [{"name": "end", "kind": "face", "role": "loss"}] and len(rec["warnings"]) == 1
