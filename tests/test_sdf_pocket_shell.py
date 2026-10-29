"""F4-49: the result STL of the `sdf` mesh step must not hold a second closed shell inside the part (CONTRACTS.md section 19).
Tests of `smooth_sdf._drop_voids`, of the fact `voids_filled`, and of the generated case that made the shell on main (`8dc2ec2`).

The case is small and made in the test. A fixed plate has an exact face that is not on the voxel grid. The last fixed voxel
centre is 3.5, and the face is at 3.56, so one fine voxel between the plate and the first design voxel is empty in the sharp
field. The smoothing makes a thin slit of 0.1 to 0.4 mm there. Where the slit is narrower than the band of the snap, the snap
closes it and the exact union with the plate seals the rest: a thin closed void between the plate and the design. The cleanup
step (`_orient`) turns a void inside out, so the file holds a thin second body with outward normals inside the material.
"""
import dataclasses
import logging

import numpy as np
import pytest
import trimesh

from topopt import post, smooth_sdf as ss

FACE = 3.56    # the exact plate face (mm); the voxel size is 1 mm and the fixed voxels end at 4 mm
PATCH = 12     # the design block touches the plate over PATCH x PATCH voxels (a bar in the middle of the face)


def pocket_case(tmp_path, face=FACE, patch=PATCH, ny=16, nz=16, fixed_n=4, depth=6, tip=2):
    """A fixed plate (STL, exact face at `face`), a block of design that joins a load block at its far end. Return the arguments of the mesh step."""
    nx = fixed_n + depth + tip
    labels = np.full((nx, ny, nz), post.DESIGN, np.uint8)
    labels[:fixed_n] = post.FIXED
    rho = np.zeros((nx, ny, nz), np.float32)
    rho[:fixed_n] = 1.0
    y0, z0 = (ny - patch) // 2, (nz - patch) // 2
    rho[fixed_n:, y0:y0 + patch, z0:z0 + patch] = 1.0
    labels[fixed_n + depth:, y0:y0 + patch, z0:z0 + patch] = post.LOAD
    plate = tmp_path / "plate.stl"
    trimesh.creation.box(bounds=[[-1.0, -1.0, -1.0], [face, ny + 1.0, nz + 1.0]]).export(plate)
    load = tmp_path / "load.stl"
    trimesh.creation.box(bounds=[[fixed_n + depth, y0, z0], [nx, y0 + patch, z0 + patch]]).export(load)
    return rho, labels, plate, load


def run_case(tmp_path, **kw):
    logging.disable(logging.WARNING)
    try:
        rho, labels, plate, load = pocket_case(tmp_path, **kw)
        return ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=post.UPSAMPLE, pad=post.PAD,
                                fixed_stls=[plate], load_stls=[load])
    finally:
        logging.disable(logging.NOTSET)


def bodies(verts, faces):
    """The closed shells of the mesh as the STL file holds them: (signed volume, box) for each, the largest first."""
    parts = trimesh.Trimesh(verts, faces, process=False).split(only_watertight=False)
    return sorted(((float(p.volume), p.bounds) for p in parts), key=lambda t: -abs(t[0]))


def test_the_pocket_comes_from_the_snap(tmp_path, monkeypatch):
    """With the move of the vertices switched off, the same case gives one shell: the move that closes the rim of the slit is the cause."""
    monkeypatch.setattr(ss, "snap_into_regions", lambda verts, faces, surfaces, band: (np.asarray(verts, np.float64), {
        "moved": 0, "in_band": 0, "of": len(verts), "target_outside": 0, "largest_move_mm": 0.0, "turned_faces": 0,
        "halving_rounds": 0, "band_mm": float(band), "depth_mm": float(band)}))
    v, f, _info = run_case(tmp_path)
    assert len(bodies(v, f)) == 1


def test_the_result_has_one_closed_shell(tmp_path):
    v, f, info = run_case(tmp_path)
    facts = ss.stl_roundtrip(v, f)
    assert facts["ok"] and facts["bodies"] == 1, facts
    assert len(bodies(v, f)) == 1
    assert bodies(v, f)[0][0] > 0   # the one shell turns outward


def test_the_filled_void_is_a_fact_and_a_warning(tmp_path, caplog):
    """The level of the third round seals the slit. The round limit of before #348 keeps that level; the extra rounds keep a lower one."""
    rho, labels, plate, load = pocket_case(tmp_path)
    plain = dataclasses.replace(ss.SdfParams(), calibrate_steps_fill=ss.SdfParams().calibrate_steps)
    with caplog.at_level(logging.WARNING, logger="topopt.smooth_sdf"):
        v, f, info = ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=post.UPSAMPLE, pad=post.PAD,
                                      fixed_stls=[plate], load_stls=[load], params=plain)
    voids = info["voids_filled"]
    assert voids["count"] == 1 and 5.0 < voids["volume_mm3"] < 60.0   # 23 mm3 on the reference host: 0.2 to 0.5 mm over a square patch
    assert any("filled 1 closed void" in r.getMessage() for r in caplog.records)
    assert info["watertight"]


def test_the_extra_rounds_keep_the_best_round_when_the_volume_jumps_with_the_level(tmp_path):
    """#348: the void of the slit appears and goes with the level, so the volume jumps by about 0.9 % and the rounds circle round the target
    (+0.88, -0.58, -0.19, +0.88 %). The first extra round that misses more ends the loop, and the best round is the result."""
    rho, labels, plate, load = pocket_case(tmp_path)
    logging.disable(logging.WARNING)
    try:
        v, f, info = ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[plate], load_stls=[load])
    finally:
        logging.disable(logging.NOTSET)
    rounds, plain = info["calibration"], ss.SdfParams().calibrate_steps
    errors = [abs(r["error"]) for r in rounds[plain:]]
    assert len(rounds) > plain + 1 and errors == sorted(errors, reverse=True)       # each kept extra round misses less than the one before
    assert abs(rounds[-1]["next_round_error"]) >= abs(rounds[-1]["error"])          # the round after the kept one missed more
    assert abs(ss._signed_volume(v, f)) == pytest.approx(rounds[-1]["final_mm3"], rel=2e-4)
    assert not any("next_round_error" in r for r in rounds[:-1])
    # The kept round is the result in full: a run whose round limit ends at that round has the same mesh and the same snap fact (the build
    # without the move is at the level of the kept round; #348 review: -46.19 mm3 at the level of the dropped round against -23.84 mm3).
    logging.disable(logging.WARNING)
    try:
        v2, f2, info2 = ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[plate], load_stls=[load],
                                         params=dataclasses.replace(ss.SdfParams(), calibrate_steps_fill=len(rounds) - 1))
    finally:
        logging.disable(logging.NOTSET)
    assert np.array_equal(v, v2) and np.array_equal(f, f2)
    assert info["snap"]["moved"] and info["snap"]["volume_change_mm3"] == info2["snap"]["volume_change_mm3"]


def test_the_volume_of_the_last_calibration_round_is_the_volume_of_the_one_shell(tmp_path):
    """The calibration measures the mesh after the fill, so the file volume and the volume of the last round are the same number."""
    v, f, info = run_case(tmp_path)
    (vol, _box), = bodies(v, f)
    assert vol == pytest.approx(info["calibration"][-1]["final_mm3"], rel=2e-4)   # the float32 clean-up changes the volume by a few ppm


def test_a_run_without_a_void_has_the_fact_with_zeros_and_the_same_mesh(tmp_path):
    """No slit, no void: the fact is zero and the clean-up step leaves the arrays as they are (the same objects)."""
    v, f, info = run_case(tmp_path, face=3.9)
    assert info["voids_filled"] == {"count": 0, "volume_mm3": 0.0}
    assert len(bodies(v, f)) == 1
    verts, faces = np.asarray(v, np.float64), np.asarray(f, np.int64)
    out_v, out_f, facts = ss._drop_voids(verts, faces)
    assert out_v is verts and out_f is faces and facts["count"] == 0


# -- _drop_voids on small meshes made by hand --
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


def test_drop_voids_removes_a_cavity_and_keeps_the_body():
    v, f = _join(_shell(5.0), _shell(1.0, inward=True))
    assert ss._signed_volume(v, f) == pytest.approx(1000.0 - 8.0)
    out_v, out_f, facts = ss._drop_voids(v, f)
    assert facts == {"count": 1, "volume_mm3": pytest.approx(8.0)}
    assert len(out_f) == 12 and ss._signed_volume(out_v, out_f) == pytest.approx(1000.0)
    assert len(out_v) == 8 and out_f.max() < len(out_v)


def test_drop_voids_keeps_a_body_inside_a_cavity_and_a_second_body():
    v, f = _join(_shell(10.0), _shell(5.0, inward=True), _shell(1.0), _shell(2.0, center=(30.0, 0.0, 0.0)))
    out_v, out_f, facts = ss._drop_voids(v, f)
    assert facts["count"] == 1 and facts["volume_mm3"] == pytest.approx(1000.0)
    assert ss._signed_volume(out_v, out_f) == pytest.approx(8000.0 + 8.0 + 64.0)
    assert len(np.unique(ss._split_faces(out_f, len(out_v)))) == 3


def test_drop_voids_returns_the_same_arrays_when_there_is_nothing_to_fill():
    v, f = _join(_shell(5.0), _shell(2.0, center=(30.0, 0.0, 0.0)))
    out_v, out_f, facts = ss._drop_voids(v, f)
    assert out_v is v and out_f is f and facts == {"count": 0, "volume_mm3": 0.0}
    v, f = _shell(5.0)
    out_v, out_f, facts = ss._drop_voids(v, f)
    assert out_v is v and out_f is f and facts["count"] == 0
    out = ss._drop_voids(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    assert out[2]["count"] == 0


def test_drop_voids_does_not_empty_a_mesh_whose_faces_all_turn_inward():
    v, f = _join(_shell(5.0, inward=True), _shell(2.0, center=(30.0, 0.0, 0.0), inward=True))
    out_v, out_f, facts = ss._drop_voids(v, f)
    assert out_v is v and out_f is f and facts["count"] == 0


def test_the_marching_cubes_turn_outward_from_the_material():
    """`_drop_voids` reads the sign of the volume: a solid from `_march` must be positive and a hole in it negative."""
    x, y, z = np.meshgrid(*[np.arange(24, dtype=np.float32)] * 3, indexing="ij")
    r = np.sqrt((x - 11.5) ** 2 + (y - 11.5) ** 2 + (z - 11.5) ** 2)
    verts, faces = ss._march(r - 8.0)
    assert ss._signed_volume(verts, faces) > 0
    verts, faces = ss._march(np.maximum(r - 9.0, 4.0 - r))   # a ball of radius 9 with a hole of radius 4 in the middle
    lab = ss._split_faces(faces, len(verts))
    vols = [ss._signed_volume(verts, faces[lab == k]) for k in np.unique(lab)]
    assert sorted(v > 0 for v in vols) == [False, True]
