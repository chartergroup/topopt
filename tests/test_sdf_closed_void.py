"""A closed void of the design in the `sdf` mesh step (CONTRACTS.md section 19, steps 5, 6 and 7).

The case of the issue: the slanted bar of `tests/test_sdf_envelope_surface.py` with a closed void of 28 x 4 x 5 mm, 3 mm above the bottom
face and 2 mm under the top face (14.5 % of the target). Before 0.5.1 the calibration flipped between about +30 % and -25 %: the wall of
the void was removed as a body that does not touch a frozen region, which filled the void and took its volume off the target, until the
level shift opened the skin. Now the void stays a cavity: the mesh meets the target, and the skin keeps its thickness.
"""
import logging

import numpy as np
import pytest
import trimesh

from test_sdf_envelope_surface import slanted_case
from topopt import post, printcheck, smooth_sdf as ss

VOID = (slice(10, 38), slice(6, 10), slice(3, 8))   # coarse voxels; the voxel size of the case is 1 mm, the top face is at z 10 mm


def void_bar(tmp_path, mode, void=True):
    (rho, labels, org, h), _run, _ = slanted_case(tmp_path)
    if void:
        rho[VOID] = 0.0
    logging.disable(logging.WARNING)
    try:
        return ss.make_mesh_sdf(rho, labels, org, h, up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[tmp_path / "foot.stl"],
                                load_stls=[tmp_path / "tip.stl"], design_stl=tmp_path / "design.stl", params=ss.SdfParams(envelope_fill=mode))
    finally:
        logging.disable(logging.NOTSET)


def skin_mm(verts, faces):
    """The thickness of the skin over the void on vertical lines through its middle: the top face minus the roof of the cavity."""
    m = trimesh.Trimesh(verts, faces, process=False)
    out = []
    for x in np.linspace(VOID[0].start + 3, VOID[0].stop - 3, 5):
        loc, _r, _t = m.ray.intersects_location([[x, 8.0, -50.0]], [[0.0, 0.0, 1.0]], multiple_hits=True)
        z = np.sort(loc[:, 2])
        out.append(z.max() - z[z < VOID[2].stop + 1.0].max())
    return np.array(out)


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_a_closed_void_keeps_the_target_and_stays_a_cavity(tmp_path, mode):
    v, f, info = void_bar(tmp_path, mode)
    last = info["calibration"][-1]
    assert abs(last["error"]) < 5e-4 and info["watertight"]                       # the stop rule of step 7; main: -25 % (off), -57 % (auto)
    assert abs(ss._signed_volume(v, f)) == pytest.approx(info["volume_thresholded_mm3"], rel=1e-3)
    assert all(np.sign(r["error"]) != 0 for r in info["calibration"])
    assert max(abs(r["error"]) for r in info["calibration"]) < 0.1                 # no round near the +-30 % of the flip
    assert info["voids_filled"] == {"count": 0, "volume_mm3": 0.0}
    kept = info["voids_kept"]
    assert kept["count"] == 1 and 400.0 < kept["volume_mm3"] < 700.0              # the sharp void is 560 mm3; the smoothing rounds it
    health = printcheck.mesh_health(np.asarray(v, np.float64), np.asarray(f, np.int64))
    assert len(health["bodies"]) == 1 and len(health["voids"]) == 1 and health["inverted_shells"] == 0   # a cavity, not a second body
    assert ss.stl_roundtrip(v, f)["ok"]


def test_the_skin_over_the_void_keeps_its_thickness_within_one_fine_voxel(tmp_path):
    v, f, _info = void_bar(tmp_path, "off")
    pitch = 1.0 / post.UPSAMPLE
    skin = skin_mm(v, f)
    assert skin.min() >= 2.0 - pitch, skin                                         # main: the skin was cut through


def test_with_the_fill_the_skin_pays_the_shift_of_the_fill_as_any_member(tmp_path):
    """The fill moves material to the skin of the design space, and the level shift pays for it from every member (`envelope_fill.note`):
    the wall over the cavity loses it from both of its faces. It stays, and the fact says the cost."""
    v, f, info = void_bar(tmp_path, "auto")
    assert info["envelope_fill"]["ran"] and info["envelope_fill"]["note"] == ss.FILL_COST_NOTE
    assert info["calibration"][-1]["delta_mm"] < 0.0
    assert skin_mm(v, f).min() > 1.0


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_a_flat_void_one_voxel_high_stays_a_cavity(tmp_path, mode):
    """Two fine layers tie for the deepest voxel of a flat void, and a voxel centre is a quarter of a voxel off its mid-plane: the
    smoothed wall missed it, the void was filled, and with the fill the calibration flipped (+2.1 % after 9 rounds)."""
    (rho, labels, org, h), _run, _ = slanted_case(tmp_path)
    rho[10:38, 6:10, 5:6] = 0.0
    logging.disable(logging.WARNING)
    try:
        _v, _f, info = ss.make_mesh_sdf(rho, labels, org, h, up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[tmp_path / "foot.stl"],
                                        load_stls=[tmp_path / "tip.stl"], design_stl=tmp_path / "design.stl",
                                        params=ss.SdfParams(envelope_fill=mode))
    finally:
        logging.disable(logging.NOTSET)
    assert info["voids_kept"]["count"] == 1 and info["voids_filled"]["count"] == 0
    assert abs(info["calibration"][-1]["error"]) < 5e-4 and info["watertight"]


def test_part_only_keeps_the_cavity_of_the_part(tmp_path):
    """`part_only`: manifold3d splits a body with a cavity into the body and the wall of the cavity. The wall goes with its piece: the
    part keeps its cavity, and no loose piece is dropped."""
    from topopt import part_only

    block, cav = trimesh.creation.box(bounds=[[0, 0, 0], [20, 10, 10]]), trimesh.creation.box(bounds=[[4, 3, 3], [8, 7, 7]])
    v, f = _join((np.asarray(block.vertices, np.float64), np.asarray(block.faces, np.int64)),
                 (np.asarray(cav.vertices, np.float64), np.asarray(cav.faces, np.int64)[:, ::-1]))   # 20 x 10 x 10 mm minus 4^3
    trimesh.creation.box(bounds=[[16, -1, -1], [22, 11, 11]]).export(tmp_path / "clamp.stl")   # a removed body at the far end
    rho = np.ones((20, 10, 10), np.float32)
    rho[4:8, 3:7, 3:7] = 0.0
    mat = part_only.FieldMaterial(rho, np.full((20, 10, 10), post.DESIGN, np.uint8), (0.0, 0.0, 0.0), 1.0)
    vv, ff, facts = part_only.cut(v, f, [("not_printed:clamp", tmp_path / "clamp.stl")], mat)
    assert abs(ss._signed_volume(vv, ff)) == pytest.approx(16 * 100 - 64)          # before: 1600, the cavity filled
    assert facts["leftovers_dropped"] == {"bodies": 0, "volume_mm3": 0.0}           # before: a "body" of -64 mm3
    health = printcheck.mesh_health(np.asarray(vv, np.float64), np.asarray(ff, np.int64))
    assert len(health["bodies"]) == 1 and len(health["voids"]) == 1


def test_a_bar_with_no_void_keeps_no_void(tmp_path):
    _v, _f, info = void_bar(tmp_path, "off", void=False)
    assert info["voids_kept"] == {"count": 0, "volume_mm3": 0.0} and info["voids_filled"]["count"] == 0


# -- the helpers, on small shapes made by hand --
def _shell(half, center=(0.0, 0.0, 0.0), inward=False):
    m = trimesh.creation.box(extents=(2 * half,) * 3)
    m.apply_translation(center)
    f = np.asarray(m.faces, np.int64)
    return np.asarray(m.vertices, np.float64), (f[:, ::-1] if inward else f)


def _join(*shells):
    verts, faces, off = [], [], 0
    for v, f in shells:
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    return np.vstack(verts), np.vstack(faces)


def _shell_volumes(v, f):
    lab = ss._split_faces(f, len(v))
    return sorted(ss._signed_volume(v, f[lab == k]) for k in np.unique(lab))


def test_keep_touching_keeps_a_void_with_its_body():
    v, f = _join(_shell(5.0), _shell(1.0, inward=True), _shell(2.0, center=(30.0, 0.0, 0.0)))
    touches = lambda ids: bool(np.abs(v[ids]).max() > 4.0 and (v[ids, 0] < 20.0).all())   # only the big shell touches
    out_v, out_f, n, removed = ss._keep_touching(v, f, touches)
    assert n == 1 and removed == pytest.approx(64.0)               # the far body; the void is not a body (main: removed as one, 8 mm3 more)
    assert _shell_volumes(out_v, out_f) == pytest.approx([-8.0, 1000.0])


def test_keep_touching_keeps_the_void_of_a_body_in_the_cavity_of_a_removed_body():
    """A hollow body that does not touch holds, in its cavity, a body with a void that touches: the void goes with its innermost body."""
    v, f = _join(_shell(10.0), _shell(7.5, inward=True), _shell(5.0), _shell(2.5, inward=True))
    out_v, out_f, n, removed = ss._keep_touching(v, f, lambda ids: bool(np.abs(v[ids]).max() < 6.0))
    assert n == 1 and removed == pytest.approx(8000.0 - 3375.0)                    # the material of the hollow body
    assert _shell_volumes(out_v, out_f) == pytest.approx([-125.0, 1000.0])


def test_keep_touching_removes_a_void_with_the_body_around_it():
    v, f = _join(_shell(5.0), _shell(3.0, center=(30.0, 0.0, 0.0)), _shell(1.0, center=(30.0, 0.0, 0.0), inward=True))
    out_v, out_f, n, removed = ss._keep_touching(v, f, lambda ids: bool((v[ids, 0] < 20.0).all()))
    assert n == 1 and removed == pytest.approx(216.0 - 8.0)                        # the material of the far body
    assert _shell_volumes(out_v, out_f) == pytest.approx([1000.0])


def test_drop_voids_keeps_the_voids_that_keep_names():
    v, f = _join(_shell(10.0), _shell(1.0, center=(-5.0, 0.0, 0.0), inward=True), _shell(2.0, center=(5.0, 0.0, 0.0), inward=True))
    out_v, out_f, facts = ss._drop_voids(v, f, keep=lambda verts, shell: bool(verts[np.unique(shell), 0].mean() > 0.0))
    assert facts == {"count": 1, "volume_mm3": pytest.approx(8.0), "kept": {"count": 1, "volume_mm3": pytest.approx(64.0)}}
    assert _shell_volumes(out_v, out_f) == pytest.approx([-64.0, 8000.0])


def test_orient_keeps_a_cavity_turned_inward_and_still_turns_bodies_outward():
    v, f = _join(_shell(5.0), _shell(1.0, inward=True))
    out_v, out_f = ss._orient(v, f)
    assert _shell_volumes(out_v, out_f) == pytest.approx([-8.0, 1000.0])
    v, f = _join(_shell(5.0, inward=True), _shell(1.0))                           # the whole mesh inside out: it turns over
    assert _shell_volumes(*ss._orient(v, f)) == pytest.approx([-8.0, 1000.0])
    v, f = _join(_shell(5.0), _shell(2.0, center=(30.0, 0.0, 0.0), inward=True))  # a body written inside out, in no other shell
    assert _shell_volumes(*ss._orient(v, f)) == pytest.approx([64.0, 1000.0])


def test_closed_cavities_finds_an_enclosed_void_and_not_an_open_pocket():
    S = np.zeros((20, 20, 20), bool)
    S[2:18, 2:18, 2:18] = True
    S[6:10, 6:12, 6:9] = False                                                     # closed: 4 x 6 x 3 voxels
    S[12:16, 12:16, 12:18] = False                                                 # open at the top face of the block
    cav = ss.closed_cavities(S, np.array([-1.0, 0.0, 0.0]), 0.5)
    assert cav["count"] == 1 and cav["volume_mm3"] == pytest.approx(72 * 0.125)
    (x, y, z), = cav["points"]
    assert 2.0 < x < 4.0 and 3.0 < y < 6.0 and 3.0 < z < 4.5                      # inside the closed void (world, origin x -1)
    assert ss.closed_cavities(np.ones((5, 5, 5), bool), np.zeros(3), 1.0)["count"] == 0


def test_the_cavity_reaches_final_stl_through_the_whole_post_step(tmp_path):
    """The run folder of `test_smooth_sdf` (STL plate, load and keep-out hole) with a closed void in its block: `final.stl` has one body and
    one cavity, the volume meets the target, and the fact is in `metrics.json`."""
    from test_smooth_sdf import _hole_run_with_stls

    run, labels = _hole_run_with_stls(tmp_path)
    shape = tuple(run["shape"])
    rho = np.frombuffer((tmp_path / "snapshots" / "it_0001.bin").read_bytes(), np.uint8).reshape(shape).copy()
    rho[8:14, 4:8, 15:20] = 0                                                      # 6 x 4 x 5 voxels of 1.2 mm, away from the hole and the load
    (tmp_path / "snapshots" / "it_0001.bin").write_bytes(rho.tobytes())
    m = post.finish(tmp_path)
    rec = m["mesh"]
    assert rec["method"] == "sdf" and abs(rec["volume_error_pct"]) < 0.1
    assert rec["voids_kept"]["count"] == 1 and rec["voids_filled"]["count"] == 0
    stl = trimesh.load(tmp_path / "result" / "final.stl", force="mesh")
    stl.merge_vertices()
    health = printcheck.mesh_health(np.asarray(stl.vertices, np.float64), np.asarray(stl.faces, np.int64))
    assert len(health["bodies"]) == 1 and len(health["voids"]) == 1 and health["inverted_shells"] == 0
    assert health["voids"][0]["volume_mm3"] == pytest.approx(rec["voids_kept"]["volume_mm3"], rel=0.02)
