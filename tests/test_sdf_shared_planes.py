"""F4-50: the result of the `sdf` mesh step stays inside the planes that the design space shares with a frozen region (CONTRACTS.md section 19).

The footbridge case that showed it (run b2 of the REL-6 series: a solid deck that is flush with the side planes y = -18 and y = 18 of
the design space) had thin raised patches on the deck face, up to 0.56 mm high, and the members that touch the side plane went past it by up
to 0.6 mm. The cause: the frozen voxels are a staircase (the exact face lies up to half a fine voxel inside the last voxel that
the field holds), the low-pass moves the zero level by a little more, and the snap takes only the vertices within 0.35 fine voxels of
the face into the region. The exact union with the region does not take material away.

The generated case below is a small copy of that mechanism: a deck plate flush with the side planes, a face that is not on the fine grid.
The tests of the planes use boxes by hand. The case that made 3 shells in the first report of the same series (run b1: a bar between
two fixed feet whose inner faces are not on the grid) is a void turned inside out: it was fixed with F4-49 (`_drop_voids`).
"""
import logging

import numpy as np
import pytest
import trimesh

from topopt import post, smooth_sdf as ss

H = 1.0
NX, NY, NZ = 24, 12, 14
ZD, ZT = 10.0, 13.0    # the top of the design space, the top of the deck
FACE = 2.24            # the side plane y = FACE: the deck voxels reach 0.24 mm past it (the fine pitch is 0.5 mm, the band 0.175 mm)


def box(lo, hi):
    return trimesh.creation.box(bounds=[lo, hi])


def plate_case(tmp_path, face=FACE, with_design=True, tab=False):
    """A design space with a deck plate on top, flush with its side planes `y = face` and `y = NY - face`. Return `(args, kwargs, planes)`.

    `tab`: a second load region, a tab on the deck that goes 1.5 mm past the plane `y = NY - face` (x 20 to 24, z 10 to 13)."""
    y0, y1 = face, NY * H - face
    labels = np.full((NX, NY, NZ), post.EMPTY, np.uint8)
    j0, j1 = int(np.ceil(y0 / H - 0.5)), int(np.floor(y1 / H - 0.5)) + 1     # the coarse cells whose centre is inside the planes
    labels[:, j0:j1, :int(ZD)] = post.DESIGN
    labels[:, j0:j1, int(ZD):int(ZT)] = post.LOAD
    rho = np.zeros((NX, NY, NZ), np.float32)
    rho[:, j0:j1, int(ZD):int(ZT)] = 1.0
    rho[8:12, j0:j1, :int(ZD)] = 1.0                                          # a wall of design that touches both planes
    design = tmp_path / "design.stl"
    box([0, y0, 0], [NX * H, y1, ZD]).export(design)
    deck = tmp_path / "deck.stl"
    box([0, y0, ZD], [NX * H, y1, ZT]).export(deck)
    loads = [deck]
    if tab:
        jt = int(np.floor((y1 + 1.5) / H - 0.5))                                                  # the coarse cells of the tab
        labels[20:24, j1:jt + 1, int(ZD):int(ZT)] = post.LOAD
        rho[20:24, j1:jt + 1, int(ZD):int(ZT)] = 1.0
        box([20, y1, ZD], [24, y1 + 1.5, ZT]).export(tmp_path / "tab.stl")
        loads.append(tmp_path / "tab.stl")
    kw = dict(up=post.UPSAMPLE, pad=post.PAD, load_stls=loads, **({"design_stl": design} if with_design else {}))
    return (rho, labels, np.zeros(3), H), kw, (y0, y1)


def run_case(tmp_path, **kw):
    logging.disable(logging.WARNING)
    try:
        args, kwargs, planes = plate_case(tmp_path, **kw)
        return ss.make_mesh_sdf(*args, **kwargs), planes
    finally:
        logging.disable(logging.NOTSET)


def shells(verts, faces):
    return len(trimesh.Trimesh(verts, faces, process=False).split(only_watertight=False))


# -- the case --
def test_without_the_design_space_the_mesh_goes_past_the_planes_by_more_than_the_band(tmp_path):
    """The cause: the smooth surface, snapped, is still past the plane. Without `design_stl` the step is the one of F4-49."""
    (v, f, info), (y0, y1) = run_case(tmp_path, with_design=False)
    band = info["snap"]["band_mm"]
    assert band == pytest.approx(0.35 * H / post.UPSAMPLE)
    assert y0 - v[:, 1].min() > 2 * band and v[:, 1].max() - y1 > 2 * band    # 0.5 mm past the planes, the band is 0.175 mm


def test_the_mesh_stays_inside_the_planes_that_the_design_space_shares_with_the_deck(tmp_path):
    (v, f, info), (y0, y1) = run_case(tmp_path)
    assert v[:, 1].min() >= y0 - 1e-6 and v[:, 1].max() <= y1 + 1e-6          # the deck face is flat: no vertex past the plane
    assert v[:, 0].min() >= -1e-6 and v[:, 0].max() <= NX * H + 1e-6           # the end faces x = 0 and x = 24 are shared too
    assert shells(v, f) == 1 and info["watertight"]
    deck = v[v[:, 2] >= ZD - 1e-6]                                              # the vertices at the height of the deck
    assert len(deck) > 0 and deck[:, 1].min() >= y0 - 1e-6 and deck[:, 1].max() <= y1 + 1e-6


def test_the_cut_is_a_fact_and_a_warning(tmp_path, caplog, sdf_cut_only):
    args, kwargs, (y0, y1) = plate_case(tmp_path)
    with caplog.at_level(logging.WARNING, logger="topopt.smooth_sdf"):
        v, f, info = ss.make_mesh_sdf(*args, **kwargs)
    pc = info["plane_clip"]
    assert pc["planes"] == 4 and pc["passes"] == 2                              # y = y0, y = y1, x = 0, x = 24; the calibration ran again with the cut
    assert pc["vertices_beyond"] > 0 and pc["volume_removed_mm3"] > 0 and pc["largest_excess_mm"] > 0.3
    clipped = {(tuple(c["normal"]), round(c["offset_mm"], 4)) for c in pc["clipped"]}
    assert ((0.0, -1.0, 0.0), round(-y0, 4)) in clipped and ((0.0, 1.0, 0.0), round(y1, 4)) in clipped
    assert any("cut the smooth mesh at" in r.getMessage() for r in caplog.records)
    assert pc["skipped_for_interfaces"] == 0 and "error" not in pc and "note" not in pc


def test_a_region_that_goes_past_the_plane_in_another_part_stays(tmp_path, sdf_cut_only):
    """A tab of the deck (a load region) goes 1.5 mm past the plane y = y1. The tab is not cut, the rest of the deck face is flat (the cut at
    the plane; with the field limited to the envelope of #319 nothing is past the plane, see `test_the_field_limit_keeps_the_tab`)."""
    args, kwargs, (y0, y1) = plate_case(tmp_path, tab=True)
    logging.disable(logging.WARNING)
    try:
        v, f, info = ss.make_mesh_sdf(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)
    m = trimesh.Trimesh(v, f, process=False)
    assert info["plane_clip"]["planes"] == 4 and info["plane_clip"]["vertices_beyond"] > 0
    assert m.contains(np.array([[22.0, y1 + 0.75, 11.5]])).all() and v[:, 1].max() == pytest.approx(y1 + 1.5, abs=1e-5)
    away = v[v[:, 0] < 19.0]
    assert away[:, 1].max() <= y1 + 1e-6 and shells(v, f) == 1


def test_the_field_limit_keeps_the_tab(tmp_path):
    """#319: the tab is in the envelope (a load region), so the limit keeps it, and the rest of the deck face is flat without a cut."""
    args, kwargs, (y0, y1) = plate_case(tmp_path, tab=True)
    logging.disable(logging.WARNING)
    try:
        v, f, info = ss.make_mesh_sdf(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)
    m = trimesh.Trimesh(v, f, process=False)
    assert info["envelope_cut"]["field_limit"] is True and info["plane_clip"]["vertices_beyond"] == 0
    assert m.contains(np.array([[22.0, y1 + 0.75, 11.5]])).all() and v[:, 1].max() == pytest.approx(y1 + 1.5, abs=1e-5)
    away = v[v[:, 0] < 19.0]
    assert away[:, 1].max() <= y1 + 1e-6 and shells(v, f) == 1


def _inside_by_less_than(v, y0, y1, band):
    """The vertices just under the deck (1 mm) that lie inside a side plane by more than 1e-5 mm and less than `band`."""
    v = np.asarray(v)
    under = (v[:, 2] > ZD - 1.0) & (v[:, 2] < ZD - 1e-4)
    d = np.minimum(v[:, 1] - y0, y1 - v[:, 1])
    return int((under & (d > 1e-5) & (d < band)).sum())


def test_a_wall_that_ends_a_hair_inside_a_shared_plane_ends_on_it(tmp_path, monkeypatch):
    """#319: with the side planes at 2.9 and 9.1 the cells of the wall end 0.1 mm inside them (the band is 0.175 mm). With the limit of the
    field nothing stands past a plane, so the cut at the planes does not make the side of the wall flush, and the exact union with the deck
    leaves a strip on the deck face next to the plane (run b2 of the series: 0.04 mm wide, and `topopt verify` folded it at every mesh size
    from 4.75 to 5.25 mm). The vertices within the band move onto the plane."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (v, f, info), (y0, y1) = run_case(tmp_path / "a", face=2.9)
    band = info["snap"]["band_mm"]
    flush = info["envelope_cut"]["flush"]
    assert flush["vertices"] > 0 and flush["largest_move_mm"] <= band and flush["turned_faces"] == 0
    assert _inside_by_less_than(v, y0, y1, band) == 0 and info["watertight"] and shells(v, f) == 1
    monkeypatch.setattr(ss, "flush_to_planes", lambda verts, faces, planes, band, on: (np.asarray(verts, np.float64), {"vertices": 0}))
    (v, f, info), _ = run_case(tmp_path / "b", face=2.9)
    assert _inside_by_less_than(v, y0, y1, band) > 0                   # the cause

def test_a_plane_that_an_interface_shape_goes_past_is_not_cut(tmp_path):
    """A united interface shape (a boss) is part of the field. The plane that it crosses is not cut; the other planes are."""
    args, kwargs, (y0, y1) = plate_case(tmp_path)
    boss = trimesh.creation.cylinder(radius=1.5, height=4.0, sections=32)                      # axis along z: turn it to y
    boss.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    boss.apply_translation([16.0, y0, 5.0])                                                     # y from y0 - 2 to y0 + 2
    logging.disable(logging.WARNING)
    try:
        v, f, info = ss.make_mesh_sdf(*args, **kwargs, extra_frozen=[boss])
    finally:
        logging.disable(logging.NOTSET)
    pc = info["plane_clip"]
    assert pc["planes"] == 4 and pc["skipped_for_interfaces"] == 1
    assert v[:, 1].max() <= y1 + 1e-6                                                           # the plane y = y1 is cut
    assert v[:, 1].min() < y0 - 1.0                                                             # the boss is there; the plane y = y0 is not cut
    assert all(c["normal"][1] != -1.0 for c in pc["clipped"])


def without_envelope(monkeypatch):
    """F4-51 cuts the result to the envelope, also where this cut leaves it (`left`). The tests that need the plane cut alone run without it."""
    def refuse(*_a, **_k):
        raise RuntimeError("the envelope is off in this test")
    monkeypatch.setattr(ss, "make_envelope", refuse)


def test_a_deck_that_is_flush_on_the_grid_has_no_patch_and_the_mesh_stays_as_it_was(tmp_path, monkeypatch):
    """The deck faces lie on voxel boundaries (face 2.0): no patch over a deck face. The wall that touches both planes is 0.23 mm past
    them, away from the deck faces. Nothing is cut (the same arrays as without the design space); the facts list the planes in `left`."""
    without_envelope(monkeypatch)
    args, kwargs, (y0, y1) = plate_case(tmp_path, face=2.0)
    logging.disable(logging.WARNING)
    try:
        v0, f0, _i0 = ss.make_mesh_sdf(*args, **{k: w for k, w in kwargs.items() if k != "design_stl"})
        v1, f1, i1 = ss.make_mesh_sdf(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)
    pc = i1["plane_clip"]
    assert np.array_equal(v0, v1) and np.array_equal(f0, f1)
    assert not any("plane_clip_removed_mm3" in r for r in i1["calibration"])
    assert pc["planes"] == 4 and pc["clipped"] == [] and pc["vertices_beyond"] == 0 and pc["volume_removed_mm3"] == 0.0
    assert pc["passes"] == 1                                                    # no patch: the calibration ran once, as before F4-50
    assert {tuple(q["normal"]) for q in pc["left"]} == {(0.0, -1.0, 0.0), (0.0, 1.0, 0.0)}
    assert all(q["vertices"] > 0 and q["largest_excess_mm"] > 0.1 for q in pc["left"])
    assert y0 - v1[:, 1].min() > 0.1                                                           # the wall is past the plane, and stays


def test_the_calibration_sees_the_cut(tmp_path, monkeypatch):
    """The cut is made inside the build, so the volume of the file is the volume of the last round."""
    without_envelope(monkeypatch)
    (v, f, info), _planes = run_case(tmp_path)
    assert all(r["plane_clip_removed_mm3"] > 0 for r in info["calibration"] if "plane_clip_removed_mm3" in r)
    assert "plane_clip_removed_mm3" in info["calibration"][0]                   # round 0 (no level shift) has the patches
    last = info["calibration"][-1]
    assert abs(ss._signed_volume(v, f) - last["final_mm3"]) < 1e-3 * last["final_mm3"]
    assert abs(last["error"]) < 5e-3


def test_a_missing_design_file_leaves_the_mesh_as_it_was_and_says_so(tmp_path):
    args, kwargs, _planes = plate_case(tmp_path, with_design=False)
    logging.disable(logging.WARNING)
    try:
        v0, f0, i0 = ss.make_mesh_sdf(*args, **kwargs)
        v1, f1, i1 = ss.make_mesh_sdf(*args, **kwargs, design_stl=tmp_path / "nothing.stl")
    finally:
        logging.disable(logging.NOTSET)
    assert np.array_equal(v0, v1) and np.array_equal(f0, f1)
    assert i1["plane_clip"]["planes"] == 0 and "design space STL file is not there" in i1["plane_clip"]["note"]


def test_a_design_file_that_is_not_a_closed_mesh_gives_no_plane(tmp_path):
    args, kwargs, _planes = plate_case(tmp_path, with_design=False)
    open_box = box([0, 2.24, 0], [24, 9.76, 10])
    open_box.update_faces(np.arange(len(open_box.faces)) != 0)                                    # one face missing: not closed
    open_box.export(tmp_path / "open.stl")
    logging.disable(logging.WARNING)
    try:
        v0, f0, _i0 = ss.make_mesh_sdf(*args, **kwargs)
        v1, f1, i1 = ss.make_mesh_sdf(*args, **kwargs, design_stl=tmp_path / "open.stl")
    finally:
        logging.disable(logging.NOTSET)
    assert np.array_equal(v0, v1) and i1["plane_clip"]["planes"] == 0 and "not a closed mesh" in i1["plane_clip"]["note"]


# -- the planes --
def planes_of(design, *regions):
    return sorted((tuple(int(round(x)) for x in n), round(d, 6)) for n, d, _boxes in ss.shared_planes(design, list(regions)))


def whole(n, d):
    """A plane for `clip_to_planes` whose region face box is the whole plane."""
    return (np.array(n, float), float(d), np.array([[-1e3, 1e3, -1e3, 1e3]]))


def face_box(n, lo, hi):
    """The box `(u_min, u_max, w_min, w_max)` of the corners `lo`, `hi` (3D points in the plane with the unit normal `n`)."""
    u, w = ss._plane_frame(np.array(n, float))
    pts = np.array([lo, hi], float)
    return (pts @ u).min(), (pts @ u).max(), (pts @ w).min(), (pts @ w).max()


def test_the_planes_of_a_flush_deck():
    design = box([0, 2, 0], [24, 10, 10])
    deck = box([0, 2, 10], [24, 10, 13])
    # the sides and the ends are shared; the face z = 10 is not (the faces turn away from each other), the top of the deck is not a face of the design
    assert planes_of(design, deck) == sorted([((-1, 0, 0), 0.0), ((1, 0, 0), 24.0), ((0, -1, 0), -2.0), ((0, 1, 0), 10.0)])


def test_a_region_that_sticks_out_shares_no_plane():
    design = box([0, 2, 0], [24, 10, 10])
    wide = box([0, 1, 10], [24, 11, 13])
    assert planes_of(design, wide) == sorted([((-1, 0, 0), 0.0), ((1, 0, 0), 24.0)])


def test_a_plane_that_does_not_bound_the_design_space_is_not_shared():
    """The top of the low step (z = 5) is a face of the design space, but the design space goes on above it."""
    design = trimesh.util.concatenate([box([0, 0, 0], [10, 10, 5]), box([10, 0, 0], [20, 10, 10])])
    plate = box([0, 0, 0], [10, 10, 5])
    got = planes_of(design, plate)
    assert ((0, 0, 1), 5.0) not in got
    assert ((0, 0, -1), 0.0) in got and ((-1, 0, 0), 0.0) in got and ((0, -1, 0), 0.0) in got and ((0, 1, 0), 10.0) in got


def test_a_mesh_with_inverted_faces_gives_the_same_planes():
    design = box([0, 2, 0], [24, 10, 10])
    deck = box([0, 2, 10], [24, 10, 13])
    flipped = deck.copy()
    flipped.invert()
    assert planes_of(design, flipped) == planes_of(design, deck)


def test_no_plane_without_a_region_face_in_it():
    design = box([0, 0, 0], [10, 10, 10])
    inside = box([2, 2, 2], [4, 4, 4])
    assert ss.shared_planes(design, [inside]) == []


def test_each_plane_has_the_boxes_of_the_region_faces_in_it():
    design = box([0, 2, 0], [24, 10, 10])
    two = [box([0, 2, 10], [10, 10, 13]), box([14, 2, 10], [24, 10, 13])]       # two decks with a gap: two pieces in the plane y = 10
    got = {(tuple(int(round(x)) for x in n), round(d, 6)): b for n, d, b in ss.shared_planes(design, two)}
    boxes = got[((0, 1, 0), 10.0)]
    assert boxes.shape == (2, 4)
    extents = sorted(sorted([round(b[1] - b[0], 6), round(b[3] - b[2], 6)]) for b in boxes)
    assert extents == [[3.0, 10.0], [3.0, 10.0]]
    u, w = ss._plane_frame(np.array([0.0, 1.0, 0.0]))
    assert np.allclose(sorted((x[0], x[1]) for x in [face_box((0, 1, 0), (0, 10, 10), (10, 10, 13)),
                                                        face_box((0, 1, 0), (14, 10, 10), (24, 10, 13))]),
                       sorted((b[0], b[1]) for b in boxes))
    assert abs(float(np.dot(np.cross(u, w), [0, 1, 0])) - 1.0) < 1e-12                 # (u, w, n) is a right-handed frame


def test_plane_tol_follows_the_float32_resolution_of_the_files():
    small, large = box([0, 0, 0], [1, 1, 1]), box([0, 0, 0], [1000, 1000, 1000])
    assert ss._plane_tol(small) < ss._plane_tol(large)
    assert ss._plane_tol(large) == pytest.approx(2 * np.sqrt(3) * np.spacing(np.float32(1000.0)))


# -- the cut --
def cube_mesh():
    cube = box([0, 0, 0], [10, 10, 10])
    return np.asarray(cube.vertices, np.float64), np.asarray(cube.faces, np.int64)


def test_flush_to_planes_moves_the_vertices_within_the_band_onto_the_plane():
    m = box([0.0, 0.1, 0.0], [10.0, 5.0, 3.0])
    v, f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    planes = [(np.array([0.0, -1.0, 0.0]), 0.0, np.zeros((0, 4)))]               # the plane y = 0, outward to -y
    yes, no = (lambda q: np.ones(len(q), bool)), (lambda q: np.zeros(len(q), bool))
    new, facts = ss.flush_to_planes(v, f, planes, 0.2, yes)
    side = v[:, 1] < 1.0
    assert np.array_equal(new[side, 1], np.zeros(int(side.sum()))) and np.array_equal(new[side][:, [0, 2]], v[side][:, [0, 2]])
    assert np.array_equal(new[~side], v[~side])
    assert facts["vertices"] == int(side.sum()) and facts["largest_move_mm"] == pytest.approx(0.1) and facts["turned_faces"] == 0
    for band, on in ((0.05, yes), (0.2, no)):                                    # farther than the band; the foot is not on the envelope
        same, facts = ss.flush_to_planes(v, f, planes, band, on)
        assert np.array_equal(same, v) and facts["vertices"] == 0
    # a vertex in a frozen region stays where it is (#319 review: a member pulled back onto the plane left a loose body at the region)
    new, facts = ss.flush_to_planes(v, f, planes, 0.2, yes, inside=lambda q: q[:, 0] < 5.0)
    keep = side & (v[:, 0] < 5.0)
    assert np.array_equal(new[keep], v[keep]) and np.array_equal(new[side & ~keep, 1], np.zeros(int((side & ~keep).sum())))
    assert facts["vertices"] == int((side & ~keep).sum()) > 0

def test_clip_to_planes_cuts_at_the_plane_and_keeps_a_closed_body():
    v, f = cube_mesh()
    out_v, out_f, facts = ss.clip_to_planes(v, f, [whole([1.0, 0.0, 0.0], 8.0)], tol=1e-5)
    m = trimesh.Trimesh(out_v, out_f, process=False)
    assert m.is_watertight and m.volume == pytest.approx(800.0) and m.bounds[1][0] == pytest.approx(8.0)
    assert facts["planes"] == 1 and facts["vertices_beyond"] == 4 and facts["largest_excess_mm"] == pytest.approx(2.0)
    assert facts["volume_removed_mm3"] == pytest.approx(200.0) and facts["left"] == []
    assert facts["clipped"] == [{"normal": [1.0, 0.0, 0.0], "offset_mm": 8.0, "vertices": 4, "over_face": 4,
                                 "largest_excess_mm": pytest.approx(2.0)}]


def test_a_vertex_in_a_region_is_not_a_reason_to_cut():
    v, f = cube_mesh()
    plane = [whole([1.0, 0.0, 0.0], 8.0)]
    out_v, out_f, facts = ss.clip_to_planes(v, f, plane, tol=1e-5, inside=lambda p: np.ones(len(p), bool))
    assert out_v is v and out_f is f and facts["vertices_beyond"] == 0 and facts["left"] == []
    out_v, out_f, facts = ss.clip_to_planes(v, f, plane, tol=1e-5, inside=lambda p: p[:, 1] < 5.0)   # 2 of the 4 vertices at x = 10
    assert facts["vertices_beyond"] == 2 and trimesh.Trimesh(out_v, out_f, process=False).bounds[1][0] == pytest.approx(8.0)


def test_only_a_part_over_the_face_of_a_region_is_a_reason_to_cut():
    """The plane x = 8 is shared with a region whose face is y 6 to 10. The 2 vertices at y = 10 are over it; the 2 at y = 0 are not.
    The cut takes the whole mesh back to the plane, so that the side face stays flat."""
    v, f = cube_mesh()
    plane = [([1.0, 0.0, 0.0], 8.0, np.array([face_box((1, 0, 0), (8, 6, 0), (8, 10, 10))]))]
    plane = [(np.array(n, float), d, b) for n, d, b in plane]
    out_v, out_f, facts = ss.clip_to_planes(v, f, plane, tol=1e-5)
    assert facts["clipped"][0]["vertices"] == 4 and facts["clipped"][0]["over_face"] == 2 and facts["left"] == []
    assert trimesh.Trimesh(out_v, out_f, process=False).bounds[1][0] == pytest.approx(8.0)


def test_a_plane_that_the_mesh_goes_past_only_away_from_the_region_faces_is_left_alone():
    v, f = cube_mesh()
    plane = [(np.array([1.0, 0.0, 0.0]), 8.0, np.array([face_box((1, 0, 0), (8, 3, 0), (8, 6, 10))]))]   # the face: y 3 to 6
    out_v, out_f, facts = ss.clip_to_planes(v, f, plane, tol=1e-5, margin=0.5)
    assert out_v is v and out_f is f
    assert facts["clipped"] == [] and facts["vertices_beyond"] == 0 and facts["volume_removed_mm3"] == 0.0
    assert facts["left"] == [{"normal": [1.0, 0.0, 0.0], "offset_mm": 8.0, "vertices": 4, "largest_excess_mm": pytest.approx(2.0)}]
    out_v, out_f, facts = ss.clip_to_planes(v, f, plane, tol=1e-5, margin=4.0)                             # the margin reaches y = 0 and y = 10
    assert facts["margin_mm"] == 4.0 and facts["clipped"][0]["over_face"] == 4 and facts["left"] == []
    assert trimesh.Trimesh(out_v, out_f, process=False).bounds[1][0] == pytest.approx(8.0)


def test_a_dry_run_finds_the_facts_and_does_not_cut():
    v, f = cube_mesh()
    out_v, out_f, facts = ss.clip_to_planes(v, f, [whole([1.0, 0.0, 0.0], 8.0)], tol=1e-5, dry=True)
    assert out_v is v and out_f is f
    assert facts["clipped"][0]["over_face"] == 4 and facts["vertices_beyond"] == 4 and facts["volume_removed_mm3"] == 0.0


def test_a_mesh_that_does_not_go_past_a_plane_comes_back_as_the_same_arrays():
    v, f = cube_mesh()
    out_v, out_f, facts = ss.clip_to_planes(v, f, [whole([1.0, 0.0, 0.0], 10.0), whole([0.0, 0.0, -1.0], 0.0)], tol=1e-5)
    assert out_v is v and out_f is f
    assert facts == {"planes": 2, "margin_mm": 0.0, "vertices_beyond": 0, "largest_excess_mm": 0.0, "volume_removed_mm3": 0.0,
                     "clipped": [], "left": []}
    out_v, out_f, facts = ss.clip_to_planes(v, f, [], tol=0.0)
    assert out_v is v and facts["planes"] == 0


def test_a_cut_that_leaves_nothing_is_an_error_not_an_empty_mesh():
    v, f = cube_mesh()
    with pytest.raises(ValueError, match="no faces"):
        ss.clip_to_planes(v, f, [whole([1.0, 0.0, 0.0], -1.0)], tol=1e-5)


# -- the run record and the post step --
def test_the_design_path_of_a_run(tmp_path):
    assert post.design_path({"yaml_path": str(tmp_path / "x.yaml"), "regions": {"design": "d.stl"}}) == tmp_path / "d.stl"
    run = {"yaml_path": str(tmp_path / "x.yaml"), "regions": {"design": "d.stl"}, "region_files": {"design": "sub/d2.stl"}}
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d2.stl").write_bytes(b"")
    assert post.design_path(run) == tmp_path / "sub" / "d2.stl"
    assert post.design_path({"yaml_path": str(tmp_path / "x.yaml")}) is None


def post_step(tmp_path, face=FACE):
    """The mesh step of a run folder (`post._mesh_method`), with the design space and the deck as STL files. Return `(verts, faces, rec, planes)`."""
    args, _kwargs, planes = plate_case(tmp_path, face)
    run = {"yaml_path": str(tmp_path / "x.yaml"), "regions": {"design": "design.stl", "fixed": [], "keep_out": [], "solid": [],
                                                              "load": {"deck": "deck.stl"}}}
    logging.disable(logging.WARNING)
    try:
        v, f, _info, rec = post._mesh_method(run, *args)
    finally:
        logging.disable(logging.NOTSET)
    return v, f, rec, planes


def test_the_post_step_keeps_the_deck_face_flat(tmp_path):
    """The symptom, through the post step (it fails on the code before F4-50: the deck face is 0.3 mm past the plane, the band of the snap is 0.175 mm)."""
    v, f, rec, (y0, y1) = post_step(tmp_path)
    deck = v[v[:, 2] >= ZD - 1e-6]
    assert deck[:, 1].min() >= y0 - 1e-6 and deck[:, 1].max() <= y1 + 1e-6
    assert v[:, 1].min() >= y0 - 1e-6 and v[:, 1].max() <= y1 + 1e-6
    assert shells(v, f) == 1


def test_the_post_step_gives_the_design_file_to_the_mesh_step(tmp_path, monkeypatch, sdf_cut_only):
    got = {}
    real = ss.make_mesh_sdf

    def spy(*a, **k):
        got.update(k)
        return real(*a, **k)
    monkeypatch.setattr(ss, "make_mesh_sdf", spy)
    v, f, rec, _planes = post_step(tmp_path)
    assert got["design_stl"] == tmp_path / "design.stl"
    assert rec["plane_clip"]["planes"] == 4 and rec["plane_clip"]["vertices_beyond"] > 0


# -- the facts of the file --
def test_the_file_facts_list_the_volume_of_each_body():
    one = box([0, 0, 0], [10, 10, 10])
    two = trimesh.util.concatenate([one, box([20, 0, 0], [25, 5, 5])])
    got = ss.stl_facts(two.export(file_type="stl"))
    assert got["bodies"] == 2 and got["body_volumes_mm3"] == [1000.0, 125.0]
    assert "body_volumes_mm3" not in ss.stl_facts(one.export(file_type="stl"))


# -- the b1 geometry of the same series: two feet with faces off the grid --
def test_a_bar_between_two_feet_with_faces_off_the_grid_is_one_shell(tmp_path):
    """On main before F4-49 the case at face 3.56 gave 3 shells (the two voids at the inner faces, turned inside out)."""
    ny = nz = 16
    foot, span, patch = 4, 16, 12
    nx = 2 * foot + span
    labels = np.full((nx, ny, nz), post.DESIGN, np.uint8)
    labels[:foot] = post.FIXED
    labels[nx - foot:] = post.FIXED
    rho = np.zeros((nx, ny, nz), np.float32)
    rho[:foot] = 1.0
    rho[nx - foot:] = 1.0
    y0 = z0 = (ny - patch) // 2
    rho[foot:nx - foot, y0:y0 + patch, z0:z0 + patch] = 1.0
    c0 = nx // 2 - 1
    labels[c0:c0 + 2, y0:y0 + patch, z0:z0 + patch] = post.LOAD
    face = 3.56
    box([-1, -1, -1], [face, ny + 1, nz + 1]).export(tmp_path / "foot_a.stl")
    box([nx - face, -1, -1], [nx + 1, ny + 1, nz + 1]).export(tmp_path / "foot_b.stl")
    box([c0, y0, z0], [c0 + 2, y0 + patch, z0 + patch]).export(tmp_path / "load.stl")
    logging.disable(logging.WARNING)
    try:
        v, f, info = ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=post.UPSAMPLE, pad=post.PAD,
                                      fixed_stls=[tmp_path / "foot_a.stl", tmp_path / "foot_b.stl"], load_stls=[tmp_path / "load.stl"])
    finally:
        logging.disable(logging.NOTSET)
    facts = ss.stl_roundtrip(v, f)
    assert facts["ok"] and facts["bodies"] == 1, facts
    assert shells(v, f) == 1 and trimesh.Trimesh(v, f, process=False).volume > 0
