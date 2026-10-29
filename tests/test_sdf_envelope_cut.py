"""F4-51: the result of the `sdf` mesh step stays inside the envelope (CONTRACTS.md section 19, "Cut to the envelope").

The envelope is the room that the user gave: the design space, the fixed, load and solid regions, and the union pieces of the interfaces at their
as-printed size. The smooth surface goes past a face of the design space by up to 0.6 of a voxel (the voxel layer at the face, the low-pass and
the level shifts); 25 of 28 finished runs of the series had a vertex more than 0.1 mm past. The cut at the shared planes (F4-50) acts only on a
plane that a frozen region shares with the design space, over the face of that region. The generated case below has a design space whose side
faces no region touches.
"""
import dataclasses
import logging

import numpy as np
import pytest
import trimesh

from topopt import post, smooth_sdf as ss

H = 1.0
NX, NY, NZ = 24, 12, 10
Y0, Y1 = 2.24, 9.76          # the side faces of the design space; the voxels of the bar reach 0.24 mm past them (y 2 to 10)


def box(lo, hi):
    return trimesh.creation.box(bounds=[lo, hi])


def bar_case(tmp_path, y0=Y0, y1=Y1, rows=None):
    """A bar between a fixed foot and a load block. The design space has the side faces `y = y0` and `y = y1`, which no region touches.

    The design space is wide in x and z, so only the side faces are near the bar (its ends and its top and bottom are free to grow).
    `rows`: the coarse rows `(j0, j1)` of the bar in y; the default is the cells whose centre is inside the side faces.
    Return `(args, run)`: the arguments of the mesh step and the run record for `post._mesh_method`.
    """
    j0, j1 = rows or (int(np.ceil(y0 / H - 0.5)), int(np.floor(y1 / H - 0.5)) + 1)
    labels = np.full((NX, NY, NZ), post.EMPTY, np.uint8)
    labels[:, j0:j1, :] = post.DESIGN
    labels[:4, 3:9, 1:9] = post.FIXED
    labels[20:, 3:9, 1:9] = post.LOAD
    rho = np.zeros((NX, NY, NZ), np.float32)
    rho[:, j0:j1, :] = 1.0
    box([-3, y0, -3], [NX * H + 3, y1, NZ * H + 3]).export(tmp_path / "design.stl")
    box([-1, 3, 1], [4, 9, 9]).export(tmp_path / "foot.stl")                # sticks out of the design space at x = 0: no face in a design plane
    box([20, 3, 1], [25, 9, 9]).export(tmp_path / "tip.stl")
    run = {"yaml_path": str(tmp_path / "x.yaml"),
           "regions": {"design": "design.stl", "fixed": ["foot.stl"], "keep_out": [], "solid": [], "load": {"tip": "tip.stl"}}}
    return (rho, labels, np.zeros(3), H), run


def mesh_step(tmp_path, **kw):
    args, run = bar_case(tmp_path, **kw)
    logging.disable(logging.WARNING)
    try:
        v, f, info, rec = post._mesh_method(run, *args)
    finally:
        logging.disable(logging.NOTSET)
    return v, f, info, rec


def shells(verts, faces):
    return len(trimesh.Trimesh(verts, faces, process=False).split(only_watertight=False))


# -- the case --
def test_the_result_stays_inside_a_design_face_that_no_region_touches(tmp_path):
    """The symptom, through the post step. Before F4-51 the bar is 0.24 mm past each side face (y 2.0 to 10.0 against 2.24 to 9.76). The field
    is limited to the envelope (#319): the surface stays inside, the cut finds nothing past, and the calibration runs once."""
    v, f, info, rec = mesh_step(tmp_path)
    assert v[:, 1].min() >= Y0 - 1e-5 and v[:, 1].max() <= Y1 + 1e-5
    assert shells(v, f) == 1 and info["watertight"]
    ec = rec["envelope_cut"]
    assert ec["cut"] is False and ec["vertices_beyond"] == 0 and ec["passes"] == 1 and ec["t_s"]["cut_pass"] == 0.0   # main d68bdc5: 2232 past, 2 passes
    assert ec["field_limit"] is True and "field_limit_note" not in ec and ec["t_s"]["field_limit"] > 0.0
    assert abs(rec["calibration"][-1]["error"]) < 5e-4 and len(rec["calibration"]) <= 4     # the cut alone: 6 builds in each of 2 passes
    assert rec["plane_clip"]["planes"] == 0


def test_the_cut_alone_keeps_the_result_inside_a_design_face(tmp_path, sdf_cut_only):
    """Without the limit of the field (F4-51 as before #319), the bar is cut to the side faces."""
    v, f, info, rec = mesh_step(tmp_path)
    assert v[:, 1].min() >= Y0 - 1e-5 and v[:, 1].max() <= Y1 + 1e-5
    assert shells(v, f) == 1 and info["watertight"]
    ec = rec["envelope_cut"]
    assert ec["field_limit"] is False and "envelope_limit is off" in ec["field_limit_note"]
    assert ec["cut"] is True and ec["vertices_beyond"] > 0 and ec["volume_removed_mm3"] > 0 and ec["largest_excess_mm"] > 0.2
    assert rec["plane_clip"]["planes"] == 0                      # no region shares a plane with the design space: only the envelope cut acts


def test_the_cut_is_paid_by_the_volume_calibration(tmp_path, sdf_cut_only):
    v, f, _info, rec = mesh_step(tmp_path)
    rounds = rec["calibration"]
    assert all(r["envelope_cut_removed_mm3"] > 0 for r in rounds)
    last = rounds[-1]
    assert abs(ss._signed_volume(v, f) - last["final_mm3"]) < 1e-3 * last["final_mm3"]
    assert abs(last["error"]) < abs(rounds[0]["error"]) / 3 and abs(last["error"]) < 2e-2    # the level shift gave back what the cut took
    assert rounds[-1]["delta_mm"] > rounds[0]["delta_mm"]


def sdf_step(tmp_path, params=None, **kw):
    """`make_mesh_sdf` on the bar case with the envelope fill off and the field not limited to the envelope (the case of #348: the cut takes
    about half of each level change); `params` replaces fields of SdfParams. Return (verts, faces, info)."""
    args, _run = bar_case(tmp_path, **kw)
    rho, labels, org, h = args
    p = dataclasses.replace(ss.SdfParams(envelope_fill="off", envelope_limit=False), **(params or {}))
    logging.disable(logging.WARNING)
    try:
        return ss.make_mesh_sdf(rho, labels, org, h, up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[str(tmp_path / "foot.stl")],
                                load_stls=[str(tmp_path / "tip.stl")], keep_out_stls=[], design_stl=str(tmp_path / "design.stl"), params=p)
    finally:
        logging.disable(logging.NOTSET)


def test_the_bar_meets_the_sharp_volume_within_0_1_percent(tmp_path):
    """#348: the cut takes about half of each level change, so the plain update left the bar 1.15 % light after `calibrate_steps` rounds
    (4 builds). A run that still misses then gets more rounds, with the measured slope of the final volume to the target."""
    plain = ss.SdfParams().calibrate_steps
    v, f, info = sdf_step(tmp_path)
    vt, rounds = info["volume_thresholded_mm3"], info["calibration"]
    assert abs(abs(ss._signed_volume(v, f)) - vt) < 1e-3 * vt and abs(rounds[-1]["error"]) < 1e-3
    assert plain + 1 < len(rounds) <= ss.SdfParams().calibrate_steps_fill + 1
    _v, _f, old = sdf_step(tmp_path, {"calibrate_steps_fill": plain})        # the round limit before #348
    assert len(old["calibration"]) == plain + 1 and old["calibration"][-1]["error"] < -1e-2


def test_a_run_that_meets_the_target_within_the_plain_rounds_keeps_its_bytes(tmp_path):
    """The bar inside a wider design space: nothing is cut, and the plain update meets the target. The extra rounds of #348 do not start."""
    plain = ss.SdfParams().calibrate_steps
    v1, f1, info1 = sdf_step(tmp_path, y0=0.0, y1=float(NY), rows=(2, 10))
    v0, f0, info0 = sdf_step(tmp_path, {"calibrate_steps_fill": plain}, y0=0.0, y1=float(NY), rows=(2, 10))
    assert len(info1["calibration"]) <= plain + 1 and abs(info1["calibration"][-1]["error"]) < 5e-4
    assert np.array_equal(v0, v1) and np.array_equal(f0, f1)


def test_a_run_that_misses_less_than_the_bar_of_348_after_the_plain_rounds_keeps_its_bytes(tmp_path):
    """The extra rounds start only at a miss of `extra_rounds_from` (0.1 %, the bar of #348) or more. With the bar above the miss of the bar
    case (1.15 % after 3 rounds), the run ends at the plain limit and has the bytes of the round limit before #348."""
    assert ss.SdfParams().extra_rounds_from == 1e-3
    plain = ss.SdfParams().calibrate_steps
    v1, f1, info1 = sdf_step(tmp_path, {"extra_rounds_from": 0.02})
    v0, f0, _info0 = sdf_step(tmp_path, {"calibrate_steps_fill": plain})
    rounds = info1["calibration"]
    assert len(rounds) == plain + 1 and -0.02 < rounds[-1]["error"] < -1e-2 and "next_round_error" not in rounds[-1]
    assert np.array_equal(v0, v1) and np.array_equal(f0, f1)


def test_an_extra_round_that_misses_more_ends_the_loop_with_the_mesh_and_the_facts_of_the_plain_limit(tmp_path):
    """The slope 0.2 (the lower clamp) moves the target of the first extra round by 5 times the miss, so that round misses more than round 3
    (+2.06 % against -1.15 %). The loop ends, and round 3 is the result: the bytes of the round limit before #348. The snap fact is measured
    at the level of round 3 too (#348 review: it was measured at the level of the dropped round)."""
    plain = ss.SdfParams().calibrate_steps
    over = {"fill_slope_min": 0.2, "fill_slope_max": 0.2}
    v1, f1, info1 = sdf_step(tmp_path, over)
    v0, f0, info0 = sdf_step(tmp_path, {**over, "calibrate_steps_fill": plain})
    rounds = info1["calibration"]
    assert len(rounds) == plain + 1 and rounds[-1]["next_round_error"] > -rounds[-1]["error"] > 1e-3
    assert [{k: r[k] for k in r if k != "next_round_error"} for r in rounds] == info0["calibration"]
    assert np.array_equal(v0, v1) and np.array_equal(f0, f1)
    assert info1["snap"]["moved"] and info1["snap"]["volume_change_mm3"] == info0["snap"]["volume_change_mm3"]


def test_the_fact_names_the_faces_with_vertices_past_and_the_volume_taken(tmp_path, sdf_cut_only):
    _v, _f, _info, rec = mesh_step(tmp_path)
    ec = rec["envelope_cut"]
    assert ec["passes"] == 2 and ec["tol_mm"] > 0 and ec["regions"] == 2 and ec["interface_pieces"] == 0
    assert "reason" not in ec and "error" not in ec
    planes = {(tuple(e["normal"]), round(e["offset_mm"], 4)): e for e in ec["faces"] if e["kind"] == "plane"}
    assert ((0.0, -1.0, 0.0), round(-Y0, 4)) in planes and ((0.0, 1.0, 0.0), round(Y1, 4)) in planes
    for e in ec["faces"]:
        assert set(e) >= {"kind", "vertices", "largest_excess_mm", "volume_removed_mm3"}
        assert e["kind"] == "curved" or set(e) >= {"normal", "offset_mm"}
    assert sum(e["vertices"] for e in ec["faces"]) == ec["vertices_beyond"]
    assert sum(e["volume_removed_mm3"] for e in ec["faces"]) == pytest.approx(ec["volume_removed_mm3"], rel=1e-3)
    assert max(e["largest_excess_mm"] for e in ec["faces"]) == pytest.approx(ec["largest_excess_mm"])
    side = planes[((0.0, -1.0, 0.0), round(-Y0, 4))]
    assert side["largest_excess_mm"] > 0.2 and side["volume_removed_mm3"] > 0


def test_the_fact_gives_the_removed_percent_and_the_time_cost(tmp_path, sdf_cut_only):
    """The price of the cut for the user: the share of the volume that the cut took, and the seconds of the mesh step and of its second pass."""
    _v, _f, _info, rec = mesh_step(tmp_path)
    ec = rec["envelope_cut"]
    first = rec["calibration"][-1]["final_mm3"] + ec["volume_removed_mm3"]       # the mesh of the last round before its cut
    assert 0.0 < ec["volume_removed_pct"] < 100.0
    assert ec["volume_removed_pct"] == pytest.approx(100.0 * ec["volume_removed_mm3"] / first, rel=2e-2)
    assert ec["t_s"]["mesh_step"] > 0.0 and 0.0 < ec["t_s"]["cut_pass"] <= ec["t_s"]["mesh_step"]


def test_the_step_logs_one_warning_when_it_cuts(tmp_path, caplog, sdf_cut_only):
    args, run = bar_case(tmp_path)
    with caplog.at_level(logging.WARNING, logger="topopt.smooth_sdf"):
        post._mesh_method(run, *args)
    lines = [r.getMessage() for r in caplog.records if "to the envelope" in r.getMessage()]
    assert len(lines) == 1 and "removed" in lines[0] and "volume calibration counts the cut" in lines[0] and " %)" in lines[0]


def test_nothing_past_the_envelope_gives_the_same_arrays_and_a_zero_fact(tmp_path):
    """The design space is wider than the bar: the result does not reach its faces. The mesh is the one without the design file."""
    args, run = bar_case(tmp_path, y0=0.0, y1=float(NY), rows=(2, 10))      # the bar fills y 2 to 10 of the design space y 0 to 12
    rho, labels, org, h = args
    logging.disable(logging.WARNING)
    try:
        v1, f1, _i1, rec1 = post._mesh_method({**run, "envelope_fill": "off"}, rho, labels, org, h)   # by design: this test compares with the mesh of a run without the design file, and the envelope fill (F4-55) grows the field of a part that has one
        v0, f0, _i0, rec0 = post._mesh_method({**run, "regions": {**run["regions"], "design": None}}, rho, labels, org, h)
    finally:
        logging.disable(logging.NOTSET)
    ec = rec1["envelope_cut"]
    assert np.array_equal(v0, v1) and np.array_equal(f0, f1)
    assert ec["cut"] is False and ec["vertices_beyond"] == 0 and ec["largest_excess_mm"] == 0.0 and ec["volume_removed_mm3"] == 0.0
    assert ec["volume_removed_pct"] == 0.0 and ec["t_s"]["cut_pass"] == 0.0 and ec["t_s"]["mesh_step"] > 0.0
    assert ec["faces"] == [] and ec["passes"] == 1 and "reason" in ec
    assert not any("envelope_cut_removed_mm3" in r for r in rec1["calibration"])
    assert rec0["envelope_cut"]["cut"] is False and rec0["envelope_cut"]["faces"] == []


def test_without_a_design_file_the_mesh_is_not_cut_and_the_fact_says_why(tmp_path):
    args, run = bar_case(tmp_path)
    no_design = {**run, "regions": {**run["regions"], "design": None}}
    logging.disable(logging.WARNING)
    try:
        v, f, _info, rec = post._mesh_method(no_design, *args)
    finally:
        logging.disable(logging.NOTSET)
    ec = rec["envelope_cut"]
    assert ec["cut"] is False and ec["vertices_beyond"] == 0 and "design space STL" in ec["reason"]
    assert ec["field_limit"] is False and "envelope is not known" in ec["field_limit_note"]
    assert y_past(v) > 0.2                                        # the bar is past the faces of the design space, as before F4-51


def y_past(v):
    return max(Y0 - v[:, 1].min(), v[:, 1].max() - Y1)


def test_a_missing_design_file_and_an_open_one_are_reasons_not_errors(tmp_path):
    args, run = bar_case(tmp_path)
    (tmp_path / "design.stl").unlink()
    logging.disable(logging.WARNING)
    try:
        _v, _f, _info, rec = post._mesh_method(run, *args)
        open_box = box([0, Y0, 0], [NX, Y1, NZ])
        open_box.update_faces(np.arange(len(open_box.faces)) != 0)
        open_box.export(tmp_path / "design.stl")
        _v, _f, _info, rec2 = post._mesh_method(run, *args)
    finally:
        logging.disable(logging.NOTSET)
    assert rec["envelope_cut"]["cut"] is False and "not there" in rec["envelope_cut"]["reason"]
    assert rec2["envelope_cut"]["cut"] is False and "not a closed mesh" in rec2["envelope_cut"]["reason"]


def test_regions_without_stl_files_give_no_envelope(tmp_path):
    """The labels give the shape of the regions then: their exact faces are not known, so the envelope is not either."""
    args, run = bar_case(tmp_path)
    (tmp_path / "foot.stl").unlink()
    logging.disable(logging.WARNING)
    try:
        _v, _f, _info, rec = post._mesh_method(run, *args)
    finally:
        logging.disable(logging.NOTSET)
    assert rec["envelope_cut"]["cut"] is False and "no STL file" in rec["envelope_cut"]["reason"]


def test_a_union_piece_of_an_interface_is_in_the_envelope(tmp_path):
    """A boss goes out of the design space through the side face y = Y0. With its as-printed shape in the envelope the join stays; without
    it, the cut takes the boss back to the face."""
    args, run = bar_case(tmp_path)
    boss = trimesh.creation.cylinder(radius=1.5, height=4.0, sections=32)
    boss.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    boss.apply_translation([12.0, Y0, 5.0])                        # y from Y0 - 2 to Y0 + 2
    kw = dict(up=post.UPSAMPLE, pad=post.PAD, load_stls=[tmp_path / "tip.stl"], fixed_stls=[tmp_path / "foot.stl"],
              design_stl=tmp_path / "design.stl", extra_frozen=[boss])
    logging.disable(logging.WARNING)
    try:
        v1, f1, i1 = ss.make_mesh_sdf(*args, **kw, envelope_extra=[boss])
        v0, f0, i0 = ss.make_mesh_sdf(*args, **kw, envelope_extra=[])
    finally:
        logging.disable(logging.NOTSET)
    assert v1[:, 1].min() < Y0 - 1.5 and i1["envelope_cut"]["interface_pieces"] == 1
    assert v0[:, 1].min() >= Y0 - 1e-5 and i0["envelope_cut"]["interface_pieces"] == 0
    assert shells(v1, f1) == 1 and shells(v0, f0) == 1


def test_the_post_step_gives_the_as_printed_union_pieces_to_the_mesh_step(tmp_path, monkeypatch):
    from topopt import interfaces as itf
    from topopt.config import Interface

    args, run = bar_case(tmp_path)
    boss = Interface("s", "boss", "union", 0.3, (12, Y0, 5), (0, 1, 0), 1.5, None, 2.0)
    box_ = (np.zeros(3), np.array([NX, NY, NZ], float))
    ctx = itf.Context([boss], box_, itf.Shapes([boss], box_), True)
    printed, nominal = ctx.union_meshes(), ctx.union_meshes(itf.NOMINAL)
    assert len(printed) == 1 and printed[0].bounds[1][0] - printed[0].bounds[0][0] == pytest.approx(2 * 1.8, abs=0.01)   # radius + stock
    assert nominal[0].bounds[1][0] - nominal[0].bounds[0][0] == pytest.approx(2 * 1.5, abs=0.01)
    got = {}
    real = ss.make_mesh_sdf

    def spy(*a, **k):
        got.update(k)
        return real(*a, **k)
    monkeypatch.setattr(ss, "make_mesh_sdf", spy)
    logging.disable(logging.WARNING)
    try:
        post._mesh_method(run, *args, ctx=ctx)
    finally:
        logging.disable(logging.NOTSET)
    assert len(got["envelope_extra"]) == 1
    assert got["envelope_extra"][0].bounds[1][0] - got["envelope_extra"][0].bounds[0][0] == pytest.approx(2 * 1.8, abs=0.01)   # the label shape has none
    assert len(got["extra_frozen"]) == 1 and got["extra_frozen"][0].bounds[1][0] - got["extra_frozen"][0].bounds[0][0] == pytest.approx(2 * 1.5, abs=0.01)


# -- the envelope and the cut --
def slab(lo=(0, 2, 2), hi=(10, 8, 8)):
    """A box mesh (10 x 6 x 6). Its corners are away from the edges of the design boxes below, so each vertex has one face as the nearest."""
    c = box(list(lo), list(hi))
    return np.asarray(c.vertices, np.float64), np.asarray(c.faces, np.int64)


def test_the_envelope_is_the_union_of_the_design_space_the_regions_and_the_pieces():
    design = box([0, 0, 0], [10, 10, 10])
    env = ss.make_envelope(design, [box([-3, 2, 2], [0, 6, 6])], [box([10, 4, 4], [12, 6, 6])])
    assert env["man"].volume() == pytest.approx(1000.0 + 3 * 4 * 4 + 2 * 2 * 2)
    assert env["regions"] == 1 and env["pieces"] == 1 and env["tol"] > 0 and env["tm"].is_watertight
    kinds = [e["kind"] for e in env["faces"]]
    assert kinds == ["plane"] * 6 + ["curved"]                      # six faces of the box, then the curved entry for the rest
    assert (env["face_of_tri"] >= 0).all() and (env["face_of_tri"] < len(env["faces"])).all()


def test_cut_to_envelope_cuts_the_whole_mesh_to_the_envelope_and_keeps_a_closed_body():
    v, f = slab()
    env = ss.make_envelope(box([0, 0, 0], [8, 10, 10]), [], [])
    out_v, out_f, facts = ss.cut_to_envelope(v, f, env)
    m = trimesh.Trimesh(out_v, out_f, process=False)
    assert m.is_watertight and m.volume == pytest.approx(288.0) and m.bounds[1][0] == pytest.approx(8.0)
    assert facts["cut"] is True and facts["vertices_beyond"] == 4 and facts["largest_excess_mm"] == pytest.approx(2.0)
    assert facts["volume_removed_mm3"] == pytest.approx(72.0)
    assert facts["faces"] == [{"kind": "plane", "normal": [1.0, 0.0, 0.0], "offset_mm": 8.0, "vertices": 4,
                               "largest_excess_mm": pytest.approx(2.0), "volume_removed_mm3": pytest.approx(72.0)}]
    assert "reason" not in facts


def test_a_mesh_inside_the_envelope_comes_back_as_the_same_arrays():
    v, f = slab()
    env = ss.make_envelope(box([-1, -1, -1], [11, 11, 11]), [], [])
    out_v, out_f, facts = ss.cut_to_envelope(v, f, env)
    assert out_v is v and out_f is f
    assert facts == {"cut": False, "vertices_beyond": 0, "largest_excess_mm": 0.0, "volume_removed_mm3": 0.0, "volume_removed_pct": 0.0,
                     "tol_mm": env["tol"], "faces": [], "reason": facts["reason"]}
    assert "no vertex" in facts["reason"]


def test_a_vertex_within_the_rounding_of_the_files_is_not_past():
    v, f = slab()
    env = ss.make_envelope(box([0, 0, 0], [10 - 0.5 * ss._plane_tol(box([0, 0, 0], [10, 10, 10])), 10, 10]), [], [])
    out_v, out_f, facts = ss.cut_to_envelope(v, f, env)
    assert out_v is v and facts["vertices_beyond"] == 0 and facts["cut"] is False


def test_a_dry_run_counts_the_vertices_and_does_not_cut():
    v, f = slab()
    env = ss.make_envelope(box([0, 0, 0], [8, 10, 10]), [], [])
    out_v, out_f, facts = ss.cut_to_envelope(v, f, env, dry=True)
    assert out_v is v and out_f is f
    assert facts["cut"] is False and facts["vertices_beyond"] == 4 and facts["largest_excess_mm"] == pytest.approx(2.0)
    assert facts["volume_removed_mm3"] == 0.0 and facts["faces"] == []


def test_a_region_that_goes_past_the_design_space_widens_the_envelope():
    v, f = slab()
    env = ss.make_envelope(box([0, 0, 0], [8, 10, 10]), [box([8, 3, 3], [12, 7, 7])], [])
    out_v, out_f, facts = ss.cut_to_envelope(v, f, env)
    m = trimesh.Trimesh(out_v, out_f, process=False)
    assert m.is_watertight and m.volume == pytest.approx(8 * 36 + 2 * 4 * 4)                 # the region is in the envelope: its part of the slab stays
    assert facts["vertices_beyond"] == 4 and facts["largest_excess_mm"] == pytest.approx(np.sqrt(2.0))   # the corners (x = 10, y and z at 2 or 8)


def test_a_curved_face_of_the_design_space_is_one_curved_entry():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    env = ss.make_envelope(sphere, [], [])
    assert [e["kind"] for e in env["faces"]] == ["curved"]
    c = box([0, 0, 0], [10, 10, 10])
    c.apply_translation([-5, -5, -5])
    out_v, out_f, facts = ss.cut_to_envelope(np.asarray(c.vertices, float), np.asarray(c.faces, np.int64), env)
    m = trimesh.Trimesh(out_v, out_f, process=False)
    assert m.is_watertight and np.linalg.norm(m.vertices, axis=1).max() <= 5.0 + 1e-4
    assert facts["cut"] is True and [e["kind"] for e in facts["faces"]] == ["curved"] and facts["faces"][0]["vertices"] == 8
    assert facts["faces"][0]["volume_removed_mm3"] == pytest.approx(facts["volume_removed_mm3"])
    assert facts["largest_excess_mm"] == pytest.approx(5.0 * np.sqrt(3.0) - 5.0, abs=0.1)


def test_the_volume_of_a_piece_at_an_edge_is_split_between_the_faces_and_adds_up():
    v, f = slab((1, 1, 2), (10, 10, 8))
    env = ss.make_envelope(box([0, 0, 0], [8, 8, 10]), [], [])                        # the slab is past x = 8 and y = 8: one piece along the edge
    _v, _f, facts = ss.cut_to_envelope(v, f, env)
    names = sorted(tuple(e["normal"]) for e in facts["faces"])
    assert names == [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    assert facts["volume_removed_mm3"] == pytest.approx(9 * 9 * 6 - 7 * 7 * 6)
    assert sum(e["volume_removed_mm3"] for e in facts["faces"]) == pytest.approx(facts["volume_removed_mm3"])
    assert all(e["volume_removed_mm3"] > 0 for e in facts["faces"])


def test_a_cut_that_leaves_nothing_is_an_error_not_an_empty_mesh():
    v, f = slab()
    far = box([100, 100, 100], [101, 101, 101])
    env = ss.make_envelope(far, [], [])
    with pytest.raises(ValueError, match="no faces"):
        ss.cut_to_envelope(v, f, env)


def test_an_error_of_the_cut_leaves_the_mesh_and_is_a_fact(tmp_path, monkeypatch, sdf_cut_only):
    args, run = bar_case(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("test")
    monkeypatch.setattr(ss, "cut_to_envelope", boom)
    logging.disable(logging.WARNING)
    try:
        v, f, _info, rec = post._mesh_method(run, *args)
    finally:
        logging.disable(logging.NOTSET)
    assert rec["envelope_cut"]["cut"] is False and rec["envelope_cut"]["error"] == "RuntimeError: test"
    assert y_past(v) > 0.2 and shells(v, f) == 1


# -- the limit of the field to the envelope (#319) --
def test_envelope_sdf_is_the_exact_distance_near_the_surface():
    """`stl_sdf` takes the distance to random samples of the surface, which next to the face is too large by a part of a fine voxel. The SDF of
    the limit has the distance to the triangles near the surface, so its zero level lies on the face: at a face of the box, within 2 layers of
    voxels past the layer next to the face (2.24 fine voxels inside the low x face here; one layer ends at 1.24). Farther away it is the clip
    value, with the sign of `stl_sdf` (the same ray-cast mask)."""
    pitch, shape, origin_f, band = 0.5, (44, 34, 34), np.zeros(3), 3.0
    lo, hi = np.array([2.13, 3.07, 2.71]), np.array([19.31, 13.77, 14.29])
    m = box(lo, hi)
    e = ss.envelope_sdf(m, origin_f, pitch, shape, band)
    s = ss.stl_sdf(m, origin_f, pitch, shape, band)
    p = origin_f + pitch * (np.indices(shape).reshape(3, -1).T + 0.5)
    outside = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)
    exact = np.where(outside > 0, outside, -np.minimum(p - lo, hi - p).min(1)).reshape(shape)   # the signed distance to the box
    near = np.abs(exact) < 2.5 * pitch
    assert near.sum() > 1000 and np.abs(e[near] - exact[near]).max() < 1e-5
    assert np.abs(s[near] - exact[near]).max() > 0.1 * pitch
    assert np.array_equal(e < 0, s < 0)
    clip = np.float32((band + 1.0) * pitch)
    assert np.abs(e).max() == clip and np.all(np.abs(e[np.abs(exact) >= clip]) == clip)


def test_the_triangle_distance_is_exact_on_thin_fans_and_a_flat_triangle():
    """The exact distance of `envelope_sdf` (`_triangle_distance`) against the distance to every triangle. The caps of the cylinder are fans
    of thin triangles that meet in the middle, its sides are long thin triangles, and one triangle has no area. `k` 4 makes most points go
    round again with more pieces."""
    m = trimesh.creation.cylinder(radius=1.0, height=30.0, sections=120)
    flat = np.array([[[0.0, 0.0, 20.0], [1.0, 0.0, 20.0], [2.0, 0.0, 20.0]]])
    tri = np.concatenate([np.asarray(m.triangles, np.float64), flat])
    pts = np.random.default_rng(5).uniform([-3.0, -3.0, -17.0], [3.0, 3.0, 22.0], (1500, 3))
    got = ss._triangle_distance(tri, pts, 0.5, k=4)
    n = len(m.faces)
    q = trimesh.triangles.closest_point(np.tile(np.asarray(m.triangles), (len(pts), 1, 1)), np.repeat(pts, n, axis=0))
    ref = np.linalg.norm(q - np.repeat(pts, n, axis=0), axis=1).reshape(len(pts), n).min(1)
    t = np.clip(pts[:, 0] / 2.0, 0.0, 1.0)                                                   # the flat triangle is the segment (0..2, 0, 20)
    ref = np.minimum(ref, np.linalg.norm(pts - np.column_stack([2.0 * t, 0.0 * t, 20.0 + 0.0 * t]), axis=1))
    assert np.abs(got - ref).max() < 1e-9

def test_a_triangle_of_no_area_or_a_sliver_gives_the_distance_to_its_edges():
    """A triangle whose vertices lie on a line, one with 2 equal vertices and a sliver 1e-12 mm wide: the distance and the closest point
    are those of the segment (0, 0, 0) to (2, 0, 0) (#319 review: the region tests gave the distance to the nearest vertex, 2.0 too far)."""
    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1e-12, 0.0]]])
    pts = np.array([[0.5, 0.3, 0.4], [1.5, -0.6, 0.8], [-1.0, 0.0, 0.0], [3.0, 0.0, 4.0]])
    want = np.array([0.5, 1.0, 1.0, np.sqrt(17.0)])
    foot = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    for t in tri:
        n = len(pts)
        d, q = ss._point_triangle_distance(pts, np.tile(t[0], (n, 1)), np.tile(t[1], (n, 1)), np.tile(t[2], (n, 1)), point=True)
        assert d == pytest.approx(want, abs=1e-9) and q == pytest.approx(foot, abs=1e-9)


def test_the_triangle_distance_finds_a_large_triangle_whose_centre_is_far():
    """A point 1 mm over a large triangle, whose centre is 3.4 mm away, and small triangles 1.5 mm away, whose centres are nearer than that.
    The nearest pieces are the small ones, so the point goes round again with more pieces until no piece can be nearer (the proof)."""
    big = [[-1.0, -1.0, 0.0], [9.0, -1.0, 0.0], [-1.0, 9.0, 0.0]]
    small = [[[x - 0.05, y, 2.5], [x + 0.05, y, 2.5], [x, y + 0.05, 2.5]] for x in (-0.2, 0.0, 0.2) for y in (-0.2, 0.0, 0.2)]
    tri = np.array([big] + small)
    pts = np.array([[0.0, 0.0, 1.0], [0.1, -0.1, 1.0], [-0.1, 0.1, 0.9]])
    d = ss._triangle_distance(tri, pts, 20.0, k=4)                  # pieces of up to 20 mm: the large triangle is one piece
    assert d == pytest.approx([1.0, 1.0, 0.9], abs=1e-12)

def test_the_local_window_counts_only_the_solid_inside_the_room():
    """The solid past the envelope is cut away, so the local level shift of `smooth_vp` counts only the solid inside it (`room`): near a face
    that cuts the top layer of a bar, the level rises (the field is lower, the member gets its volume back there); far from it, nothing changes."""
    shape = (48, 24, 24)
    solid = np.zeros(shape, bool)
    solid[4:44, 8:16, 8:16] = True
    room = np.ones(shape, bool)
    room[28:, :, 15:] = False                                       # the face z = 15 for x >= 28 cuts the top layer of the bar
    p = ss.SdfParams()
    phi = ss.binary_sdf(solid, 1.0)
    f_room, _c = ss.smooth_vp(phi, solid, p.sigma * 2.0, 1.0, p, room=room)
    f_free, _c = ss.smooth_vp(phi, solid, p.sigma * 2.0, 1.0, p)
    assert f_room[36, 8, 11] < f_free[36, 8, 11] - 0.05                # at the side of the bar under the face
    assert f_room[6, 8, 11] == pytest.approx(f_free[6, 8, 11], abs=1e-3)


def test_the_mesh_record_of_the_post_step_has_the_key(tmp_path):
    _v, _f, _info, rec = mesh_step(tmp_path)
    assert "envelope_cut" in rec and rec["envelope_cut"]["field_limit"] is True and "cut" in rec["envelope_cut"]
