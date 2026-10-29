"""Tests of the SDF mesh step (topopt/smooth_sdf.py, CONTRACTS.md section 19)."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_final_stl  # noqa: E402
from test_optimize import make_case  # noqa: E402
from test_post import load_stl, make_hole_run  # noqa: E402
from topopt import post, smooth_sdf as ss  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.config import UserError, load_config  # noqa: E402


def _rods_field(shape=(60, 40, 40)):
    """Two parallel rods (thin and thick) and a fixed base plate. Return (rho, labels)."""
    labels = np.full(shape, post.DESIGN, np.uint8)
    labels[:3] = post.FIXED
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    rho = np.zeros(shape, np.float32)
    rho[((y - 10) ** 2 + (z - 20) ** 2) < 1.6 ** 2] = 1.0   # thin rod: 3 voxels across
    rho[((y - 28) ** 2 + (z - 20) ** 2) < 5.0 ** 2] = 1.0   # thick rod
    rho[:3] = 1.0
    rho[:3, :, :][:, :, :] = 1.0
    return rho, labels


def test_thin_member_keeps_its_volume():
    rho, labels = _rods_field()
    h = 1.0
    v, f, info = ss.make_mesh_sdf(rho, labels, [0, 0, 0], h, up=2, pad=post.PAD)
    assert info["watertight"] and info["sources"]["frozen"] == "labels"
    thr = info["volume_thresholded_mm3"]
    assert abs(info["volume_smoothed_mm3"] - thr) / thr < 0.01
    # the thin rod is still there: material at its axis, in the middle of the rod length
    mesh = trimesh.Trimesh(v, f, process=False)
    assert mesh.contains(np.array([[30.0, 10.5, 20.5], [30.0, 28.5, 20.5]])).all()


def test_steps_are_smoothed():
    """A shallow slope has long terraces. Marching cubes of the sharp field shows them, the SDF step removes them."""
    shape = (50, 12, 40)
    labels = np.full(shape, post.DESIGN, np.uint8)
    labels[:3] = post.FIXED
    x, _y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    rho = (z < 8 + 0.25 * x).astype(np.float32)
    rho[:3] = 1.0

    def rough(v, f):
        m = trimesh.Trimesh(v, f, process=False)
        c = m.triangles_center
        sel = (c[:, 0] > 15) & (c[:, 0] < 45) & (c[:, 1] > 3) & (c[:, 1] < 11) & (m.face_normals[:, 2] > 0.3)
        n = m.face_normals[sel]
        return np.arctan2(n[:, 0], n[:, 2]).std()

    v0, f0, i0 = post.make_mesh(rho, labels, np.zeros(3), 1.2, blur=0.0, passes=0)
    v1, f1, i1 = ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.2, up=2, pad=post.PAD)
    assert i1["watertight"]
    assert rough(v1, f1) < 0.5 * rough(v0, f0)
    vol0 = post._signed_volume(v0, f0)
    assert abs(post._signed_volume(v1, f1) - vol0) / vol0 < 0.05


def test_sdf_volume_error_not_larger_than_blur():
    rho, labels = _rods_field()
    h = 1.0
    thr = ss.thresholded_volume(rho, labels, 2, post.PAD) * h ** 3
    v, f, _i = post.make_mesh(rho, labels, np.zeros(3), h)
    err_blur = abs(post._signed_volume(v, f) - thr) / thr
    v, f, _i = ss.make_mesh_sdf(rho, labels, np.zeros(3), h, up=2, pad=post.PAD)
    err_sdf = abs(post._signed_volume(v, f) - thr) / thr
    assert err_sdf <= max(err_blur, 5e-4)


# -- exact shapes, on a run with STL files (the hole field of test_post) --
def _hole_run_with_stls(tmp_path, *, hole_stl=True, fixed_stl=True):
    run, labels = make_hole_run(tmp_path)
    h, org = run["h"], np.asarray(run["origin"])
    regions = {"design": "design.stl", "fixed": [], "keep_out": [], "load": {}}
    if fixed_stl:
        trimesh.creation.box(bounds=[org + h * np.array([24, 10, 10]), org + h * np.array([27, 14, 14])]).export(tmp_path / "load.stl")
        regions["load"] = {"t": "load.stl"}
        # A plate that is a little thicker than the 3 fixed voxels, with an exact (not voxel) face.
        top = org + h * np.array([3.3, 0, 0])
        box = trimesh.creation.box(bounds=[org + [0, 0, 0], [top[0], org[1] + 24 * h, org[2] + 24 * h]])
        box.export(tmp_path / "plate.stl")
        regions["fixed"] = ["plate.stl"]
    if hole_stl:
        cx, cz, radius = org[0] + 20 * h, org[2] + 12 * h, 3.5
        cyl = trimesh.creation.cylinder(radius=radius, height=60, sections=128)
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        cyl.apply_translation([cx, org[1] + 12 * h, cz])
        cyl.export(tmp_path / "hole.stl")
        regions["keep_out"] = ["hole.stl"]
    run["yaml_path"] = str(tmp_path / "x.yaml")
    run["regions"] = regions
    (tmp_path / "run.json").write_text(json.dumps(run))
    return run, labels


def test_exact_regions_and_reload(tmp_path):
    run, _ = _hole_run_with_stls(tmp_path)
    m = post.finish(tmp_path)
    rec = m["mesh"]
    assert rec["method"] == "sdf" and rec["sources"] == {"frozen": "stl", "keep_out": "stl"}
    assert m["watertight"] is True
    assert abs(rec["volume_error_pct"]) < 0.1   # the tolerance of the plan (measured on the hook: 0.02 %)
    # The file as a slicer reads it (float32): one watertight body.
    mesh = load_stl(tmp_path / "result" / "final.stl")
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert len(mesh.split(only_watertight=False)) == 1
    assert rec["stl_roundtrip"]["watertight"] and rec["stl_roundtrip"]["bodies"] == 1
    # Numbers of the check: the keep-out hole is empty, the plate is held exactly.
    res = check_final_stl.check_run(tmp_path)
    assert res["ok"], res
    assert max(v["intersect_mm3"] for v in res["keep_out"].values()) < 1e-3
    assert max(v["uncovered_mm3"] for v in res["regions"].values()) < 1e-3
    assert rec["checks"]["fixed"]["fixed:plate.stl"]["uncovered_mm3"] < 1e-3
    # The exact plate face (x = origin + 3.3 h) is on the mesh, to float32 accuracy.
    org, h = np.asarray(run["origin"]), run["h"]
    face_x = org[0] + 3.3 * h
    on_face = np.abs(mesh.vertices[:, 0] - face_x) < 1e-4
    assert on_face.sum() > 20


def test_old_path_misses_the_exact_plate(tmp_path):
    """The check can tell the two methods apart: blur does not hold the exact STL shape."""
    _hole_run_with_stls(tmp_path)
    run = json.loads((tmp_path / "run.json").read_text())
    run["mesh_method"] = "blur"
    (tmp_path / "run.json").write_text(json.dumps(run))
    m = post.finish(tmp_path)
    assert m["mesh"]["method"] == "blur"
    res = check_final_stl.check_run(tmp_path)
    assert res["regions"]["fixed:plate.stl"]["uncovered_mm3"] > 1e-3 and not res["ok"]


def test_blur_is_bit_equal_to_the_old_functions(tmp_path):
    run, labels = _hole_run_with_stls(tmp_path)
    run["mesh_method"] = "blur"
    (tmp_path / "run.json").write_text(json.dumps(run))
    post.finish(tmp_path)
    got = (tmp_path / "result" / "final.stl").read_bytes()
    snap = np.frombuffer(post._last_snapshot(tmp_path).read_bytes(), np.uint8).reshape(labels.shape)
    rho = post.build_field(snap.astype(np.float32) / 255.0, labels)
    v, f, _ = post.make_mesh(rho, labels, np.asarray(run["origin"], np.float64), run["h"])
    v, f, _ = post.subtract_keep_out(v, f, post.keep_out_paths(run))
    want = trimesh.Trimesh(v, f, process=False).export(file_type="stl")
    assert got == want


def test_labels_are_used_without_stl_files(tmp_path):
    run, _ = make_hole_run(tmp_path)   # no regions key: no STL files
    m = post.finish(tmp_path)
    assert m["mesh"]["sources"]["frozen"] == "labels"
    lc = m["mesh"]["checks"]["labels"]   # numbers against the label voxel boxes
    assert lc["fixed_label_mm3"] > 0 and lc["fixed_missing_mm3"] < 0.05 * lc["fixed_label_mm3"]
    assert lc["keep_out_label_mm3"] > 0 and lc["keep_out_inside_mm3"] < 0.05 * lc["keep_out_label_mm3"]
    res = check_final_stl.check_run(tmp_path)   # the tool gives the same numbers from the file
    assert res["labels"] is not None and res["regions"] is None
    mesh = load_stl(tmp_path / "result" / "final.stl")
    assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1


# -- guard --
def test_guard_gives_a_fact_and_the_old_path(tmp_path, monkeypatch, caplog):
    make_hole_run(tmp_path)
    monkeypatch.setenv("TOPOPT_SDF_MAX_VOXELS", "1000")
    m = post.finish(tmp_path)
    rec = m["mesh"]
    assert rec["method"] == "blur" and rec["requested"] == "sdf"
    fb = rec["fallback"]
    assert fb["ok"] is False and fb["voxels"] > fb["limit"] == 1000
    assert any("blur" in o for o in fb["options"]) and any("voxel_size" in o for o in fb["options"])
    assert "SDF mesh step skipped" in caplog.text
    assert m["watertight"] is True


def test_grid_fact_numbers():
    f = ss.grid_fact((100, 100, 100), 1.0, limit=2_000_000)
    assert f["ok"] and f["voxels"] == 1_000_000
    g = ss.grid_fact((400, 400, 400), 1.0, limit=2_000_000)
    assert not g["ok"] and "voxel_size" in g["message"]


def test_surface_triangles_counts_the_solid_void_faces_of_the_coarse_field():
    rho = np.zeros((4, 3, 3), np.float32)
    assert ss.surface_triangles(rho, 2) == 0
    rho[1, 1, 1] = 1.0                       # one solid voxel: 6 faces, 2 x 2^2 triangles each
    assert ss.surface_triangles(rho, 2) == 48
    rho[2, 1, 1] = 0.6                       # two voxels side by side: 10 faces
    assert ss.surface_triangles(rho, 2) == 80 and ss.surface_triangles(rho, 3) == 180
    rho[:] = 1.0                             # the outside counts as void: 2 x (3 x 3 + 4 x 3 + 4 x 3) faces
    assert ss.surface_triangles(rho, 2) == 8 * 66


def test_the_estimate_has_a_term_for_the_triangles_and_covers_the_measured_peaks():
    """#429: the mesh stage grows with the triangles; on a lattice the estimate of the voxels alone was a quarter of the peak."""
    f = ss.grid_fact((100, 100, 100), 1.0, limit=2_000_000, triangles=50_000)
    assert f["ok"] and f["est_triangles"] == 50_000
    assert f["est_peak_gib"] == round((10 ** 6 * ss.BYTES_PER_VOXEL + 50_000 * ss.BYTES_PER_TRIANGLE) / 2 ** 30, 2)
    assert ss.grid_fact((100, 100, 100), 1.0)["est_triangles"] == 0
    # measured with the queries of #429: fine voxels, solid/void faces of the coarse field, the rise of the step (MiB), the largest ratio
    for voxels, faces, mib, high in ((6_228_992, 235_282, 2947, 1.1), (14_257_152, 566_824, 6973, 1.1),   # the generated lattice
                                     (9_598_688, 52_038, 975, 1.4), (22_077_440, 92_512, 1999, 1.4)):    # the hook refined 3 and 4 times
        est = voxels * ss.BYTES_PER_VOXEL + 8 * faces * ss.BYTES_PER_TRIANGLE
        assert mib * 2 ** 20 <= est <= high * mib * 2 ** 20


def test_the_step_estimates_the_triangles_from_the_coarse_field():
    rho, labels = _rods_field()
    v, f, info = ss.make_mesh_sdf(rho, labels, [0, 0, 0], 1.0, up=2, pad=post.PAD)
    g = info["grid"]
    assert g["est_triangles"] == ss.surface_triangles(rho, 2) > len(f)    # the flat faces of the plate end with few triangles
    assert g["est_peak_gib"] == round((g["voxels"] * ss.BYTES_PER_VOXEL + g["est_triangles"] * ss.BYTES_PER_TRIANGLE) / 2 ** 30, 2)


# -- setting and flag --
def test_setting_and_flag(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=3)
    assert load_config(p).settings.mesh_method == "sdf"
    import yaml
    d = yaml.safe_load(p.read_text())
    d["settings"]["mesh_method"] = "blur"
    p.write_text(yaml.safe_dump(d))
    assert load_config(p).settings.mesh_method == "blur"
    d["settings"]["mesh_method"] = "foo"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="mesh_method"):
        load_config(p)


def test_run_uses_flag_over_yaml(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=3)
    out1, out2 = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(p), "--out", str(out1), "--quiet", "--mesh-method", "blur"]) == 0
    assert json.loads((out1 / "run.json").read_text())["mesh_method"] == "blur"
    assert json.loads((out1 / "result" / "metrics.json").read_text())["mesh"]["method"] == "blur"
    assert main(["run", str(p), "--out", str(out2), "--quiet"]) == 0
    assert json.loads((out2 / "run.json").read_text())["mesh_method"] == "sdf"
    m = json.loads((out2 / "result" / "metrics.json").read_text())
    assert m["mesh"]["method"] == "sdf" and m["watertight"]
    res = check_final_stl.check_run(out2)   # 3 iterations: the design may be in two bodies, so "ok" is not asserted
    assert res["mesh"]["watertight"] and res["mesh"]["bodies"] == m["mesh"]["stl_roundtrip"]["bodies"]
    assert all(v["uncovered_mm3"] < 1e-3 for v in res["regions"].values())   # the load box is kept


# -- float32 reload (pitfall float32-stl-reload-pinch) --
def _pinch_fixture(tmp_path):
    """A plate whose face is off the grid and a hole near the plate: the exact union gives near-coincident float32 vertices."""
    h = 1.2
    run, labels = make_hole_run(tmp_path, h=h)
    org = np.asarray(run["origin"])
    plate = trimesh.creation.box(bounds=[org, [org[0] + 2.975 * h, org[1] + 24 * h, org[2] + 24 * h]])
    plate.export(tmp_path / "plate.stl")
    trimesh.creation.box(bounds=[org + h * np.array([24, 10, 10]), org + h * np.array([27, 14, 14])]).export(tmp_path / "load.stl")
    cyl = trimesh.creation.cylinder(radius=3.5, height=60, sections=128)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cyl.apply_translation([org[0] + 20 * h - 0.11, org[1] + 12 * h, org[2] + 12 * h])
    cyl.export(tmp_path / "hole.stl")
    run["yaml_path"] = str(tmp_path / "x.yaml")
    run["regions"] = {"design": "design.stl", "fixed": ["plate.stl"], "keep_out": ["hole.stl"], "load": {"t": "load.stl"}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    snap = np.frombuffer(post._last_snapshot(tmp_path).read_bytes(), np.uint8).reshape(labels.shape)
    rho = post.build_field(snap.astype(np.float32) / 255.0, labels)
    return run, labels, rho, org, h


def test_float32_reload_is_clean_after_the_fix(tmp_path):
    """Without the clean-up this case reloads with non-manifold edges, an extra zero-volume body and zero-area faces (276,000 fine voxels)."""
    run, labels, rho, org, h = _pinch_fixture(tmp_path)
    args = dict(up=2, pad=post.PAD, fixed_stls=[tmp_path / "plate.stl"], load_stls=[tmp_path / "load.stl"],
                keep_out_stls=[tmp_path / "hole.stl"])
    v0, f0, _ = ss.make_mesh_sdf(rho, labels, org, h, params=ss.SdfParams(simplify_tol=0), **args)
    bad = ss.stl_roundtrip(v0, f0)
    if bad["ok"] and bad["bodies"] == 1 and bad["zero_area_faces"] == 0:
        pytest.skip("this platform does not reproduce the pinch without the clean-up")
    v1, f1, info = ss.make_mesh_sdf(rho, labels, org, h, **args)
    good = ss.stl_roundtrip(v1, f1)
    assert good["ok"] and good["bodies"] == 1 and good["zero_area_faces"] == 0
    assert good["open_edges"] == 0 and good["nonmanifold_edges"] == 0
    assert abs(good["volume_mm3"] - bad["volume_mm3"]) < 0.01   # the clean-up does not change the shape
    rc = ss.region_checks(v1, f1, {"fixed:plate.stl": tmp_path / "plate.stl", "load:t": tmp_path / "load.stl"},
                          {"hole.stl": tmp_path / "hole.stl"})
    assert all(x["uncovered_mm3"] < 1e-3 for x in rc["fixed"].values()) and rc["keep_out"]["hole.stl"]["intersect_mm3"] < 1e-3


def test_watertight_flag_comes_from_the_reloaded_file(tmp_path, monkeypatch, caplog):
    """If the reloaded file is not a closed manifold, metrics.json does not say watertight: true, and one warning gives the counts."""
    _pinch_fixture(tmp_path)
    monkeypatch.setattr(ss, "_cleanup_float32", lambda v, f, tol, log_skip=True: (v, f, {"skipped": True}))   # the mesh step passes log_skip (#445)
    m = post.finish(tmp_path)
    rt = m["mesh"]["stl_roundtrip"]
    if rt["ok"]:
        pytest.skip("this platform does not reproduce the pinch without the clean-up")
    assert m["watertight"] is False and m["mesh"]["watertight"] is False
    assert "not a closed manifold" in caplog.text and "zero-area faces" in caplog.text


# -- other shapes with goal numbers (review R1, m-9) --
def _rods_part(h=1.0):
    """A fixed plate, a load block, and square rods of 2, 3 and 5 voxels (straight) and a 2-voxel diagonal between them."""
    shape = (60, 64, 30)
    lab = np.full(shape, post.DESIGN, np.uint8)
    lab[:3] = post.FIXED
    lab[52:57, :, 8:22] = post.LOAD
    lab[57:] = post.EMPTY
    rho = np.zeros(shape, np.float32)
    members, y0, z0 = {}, 4, 13
    for s in (2, 3, 5):
        rho[3:52, y0:y0 + s, z0:z0 + s] = 1
        members[s] = (y0, s, z0)
        y0 += s + 7
    return post.build_field(rho, lab), lab, members


def _section_volumes(v, f, members, h, x0=8, x1=48):
    import manifold3d as m3d

    from topopt.post import _to_manifold

    body = _to_manifold(v, f)
    out = {}
    for s, (y0, _s, z0) in members.items():
        lo = [x0 * h, (y0 - 3) * h, (z0 - 3) * h]
        hi = [x1 * h, (y0 + s + 3) * h, (z0 + s + 3) * h]
        box = m3d.Manifold.cube([a - b for a, b in zip(hi, lo)], False).translate(lo)
        out[s] = (body ^ box).volume()
    return out


def test_rods_of_2_3_and_5_voxels_survive_at_the_default_sigma():
    rho, lab, members = _rods_part()
    v, f, info = ss.make_mesh_sdf(rho, lab, np.zeros(3), 1.0, up=2, pad=post.PAD)
    assert info["extra_bodies_removed"] == 0 and info["extra_bodies_volume_mm3"] == 0
    vols = _section_volumes(v, f, members, 1.0)
    for s, vol in vols.items():
        voxel_vol = s * s * 40.0   # the voxel count of the rod between x = 8 and 48
        assert 0.85 * voxel_vol < vol < 1.15 * voxel_vol, (s, vol, voxel_vol)
    assert info["local_loss"]["deep_lost_mm3"] < 0.02 * info["local_loss"]["sharp_mm3"]
    assert ss.stl_roundtrip(v, f)["ok"]


def test_removed_bodies_are_reported_and_leave_the_target(caplog):
    """At the old sigma 1.5 h the 2-voxel rod breaks at the joints (review M-2). The step must say so, and not pay with other rods."""
    rho, lab, members = _rods_part()
    v, f, info = ss.make_mesh_sdf(rho, lab, np.zeros(3), 1.0, up=2, pad=post.PAD, params=ss.SdfParams(sigma=1.5))
    assert info["extra_bodies_removed"] >= 1 and info["extra_bodies_volume_mm3"] > 0
    assert "removed" in caplog.text and "do not touch a frozen region" in caplog.text
    last = info["calibration"][-1]
    assert last["target_mm3"] == pytest.approx(info["volume_thresholded_mm3"] - info["extra_bodies_volume_mm3"], rel=1e-6)
    # the 3-voxel rod keeps its voxel volume (it was +17 % when the calibration paid for the lost rod)
    vols = _section_volumes(v, f, members, 1.0)
    assert vols[3] < 1.10 * 9 * 40.0


def test_mixed_sources_keep_the_load_block(tmp_path):
    """Frozen shape from labels, keep-out from an STL: a load block that the design does not reach stays (review m-1)."""
    shape = (40, 30, 20)
    lab = np.full(shape, post.DESIGN, np.uint8)
    lab[:3] = post.FIXED
    lab[30:34, 10:20, 5:15] = post.LOAD
    rho = np.zeros(shape, np.float32)
    rho[3:20, 10:20, 5:15] = 1
    rho = post.build_field(rho, lab)
    cyl = trimesh.creation.cylinder(radius=2.0, height=40, sections=64)
    cyl.apply_translation([10.0, 5.0, 10.0])
    cyl.export(tmp_path / "ko.stl")
    v, f, info = ss.make_mesh_sdf(rho, lab, np.zeros(3), 1.0, up=2, pad=post.PAD, keep_out_stls=[tmp_path / "ko.stl"])
    assert info["sources"] == {"frozen": "labels", "keep_out": "stl"}
    assert info["extra_bodies_removed"] == 0
    tm = trimesh.Trimesh(v, f, process=False)
    parts = tm.split(only_watertight=False)
    assert len(parts) == 2   # the plate with its stub, and the load block
    block = min(parts, key=lambda q: abs(q.volume))
    assert abs(block.volume) == pytest.approx(4 * 10 * 10, rel=0.02)   # the load block keeps its label shape


# -- region files in run.json (review m-2) --
def test_region_files_in_a_subfolder_and_a_moved_run(tmp_path):
    (tmp_path / "parts").mkdir()
    trimesh.creation.box(extents=[1, 1, 1]).export(tmp_path / "parts" / "plate.stl")
    run = {"yaml_path": str(tmp_path / "x.yaml"),
           "regions": {"design": "d.stl", "fixed": ["plate.stl"], "keep_out": [], "load": {}},
           "region_files": {"design": "d.stl", "fixed": ["parts/plate.stl"], "keep_out": [], "load": {}}}
    fixed, _load, _keep = post.region_paths(run)
    assert fixed["plate.stl"] == tmp_path / "parts" / "plate.stl"     # a path that the file name alone would not find
    # An old run folder (no region_files) falls back to the YAML folder plus the name: the file is not there.
    old = {k: v for k, v in run.items() if k != "region_files"}
    assert not post.region_paths(old)[0]["plate.stl"].is_file()
    # A moved run: the YAML folder is gone, but an absolute stored path still finds the file.
    moved = {**run, "yaml_path": "/nonexistent/x.yaml",
             "region_files": {"fixed": [str(tmp_path / "parts" / "plate.stl")]}}
    assert post.region_paths(moved)[0]["plate.stl"].is_file()


def test_run_json_has_region_files(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=2)
    out = tmp_path / "o"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    rj = json.loads((out / "run.json").read_text())
    assert rj["region_files"]["fixed"] == ["plate.stl"] and rj["region_files"]["load"] == {"tip": "tip.stl"}
    assert post.region_paths(rj)[0]["plate.stl"] == tmp_path / "plate.stl"


# -- fallback and guard text (review m-4, n-1) --
def test_any_error_of_the_sdf_path_gives_the_old_path(tmp_path, monkeypatch, caplog):
    make_hole_run(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("the field cannot be made")   # a resource error (MemoryError) stops the step: tests/test_mesh_resource_errors.py
    monkeypatch.setattr(ss, "make_mesh_sdf", boom)
    m = post.finish(tmp_path)
    assert m["mesh"]["method"] == "blur" and m["mesh"]["requested"] == "sdf"
    assert "RuntimeError" in m["mesh"]["fallback"]["error"] and "resource_error" not in m["mesh"]["fallback"]
    assert "SDF mesh step failed" in caplog.text and m["watertight"] is True


def test_bad_env_value_gives_a_fact_and_an_option(monkeypatch, caplog):
    monkeypatch.setenv("TOPOPT_SDF_MAX_VOXELS", "abc")
    f = ss.grid_fact((100, 100, 100), 1.0)
    assert f["ok"] and f["limit"] == ss.MAX_SDF_VOXELS
    assert "not a number" in f["env_note"] and "Option" in f["env_note"] and "not a number" in caplog.text


def test_options_are_given_only_when_the_step_cannot_run():
    ok = ss.grid_fact((100, 100, 100), 3.25, limit=2_000_000)
    assert ok["ok"] and "options" not in ok and "voxel_size" not in ok["message"]
    over = ss.grid_fact((400, 400, 400), 1.0, limit=2_000_000)
    assert not over["ok"] and len(over["options"]) == 3


# -- check tool (review m-10) and the record --
def test_check_fails_when_the_missing_volume_is_large_even_if_the_reach_is_small(tmp_path, monkeypatch):
    _hole_run_with_stls(tmp_path)
    post.finish(tmp_path)
    assert check_final_stl.check_run(tmp_path)["ok"]
    real = ss.region_checks

    def film(*a, **k):
        out = real(*a, **k)
        key = next(iter(out["fixed"]))
        out["fixed"][key] = {"uncovered_mm3": 5.0, "max_outside_mm": 1e-5, "region_mm3": 100.0}   # a thin film
        return out
    monkeypatch.setattr(ss, "region_checks", film)
    assert not check_final_stl.check_run(tmp_path)["ok"]


def test_record_has_the_new_keys(tmp_path):
    _hole_run_with_stls(tmp_path)
    rec = post.finish(tmp_path)["mesh"]
    assert rec["fallback"] is None and rec["extra_bodies_volume_mm3"] == 0
    assert rec["voids_filled"] == {"count": 0, "volume_mm3": 0.0}   # F4-49: the fact is there, with zeros when no void was filled
    assert {"deep_lost_mm3", "parts", "largest_mm3", "sharp_mm3"} <= set(rec["local_loss"])
    assert rec["watertight"] is True and rec["stl_roundtrip"]["open_edges"] == 0
    assert set(rec["checks"]["fixed"]) == {"fixed:plate.stl", "load:t"}
    assert rec["params"]["sigma"] == 1.0
    assert rec["snap"]["moved"] >= 0 and "error" not in rec["snap"] and rec["snap"]["band_mm"] == pytest.approx(0.35 * 1.2 / post.UPSAMPLE)


# -- the snap of the vertices at the region faces (F4-27b) --
def _cube(half=1.0):
    return trimesh.creation.box(extents=(2 * half,) * 3)


def test_snap_moves_a_vertex_outside_the_face_one_band_deep_and_leaves_the_others():
    """The cube [-1, 1]^3 is the region, the band is 0.1 mm. Moved: a vertex 0.04 above the top face (to 0.1 under it, straight down) and a
    vertex 0.03 beside an edge and 0.03 above it (0.042 from the edge; it goes along the diagonal, 0.1 deep from the edge point). Not moved: a vertex inside the cube, a vertex 0.15 above
    the face (farther than the band), a vertex on the face (no outside, no direction)."""
    verts = np.array([[0.2, 0.1, 1.04], [1.03, 0.2, 1.03], [0.0, 0.0, 0.95], [0.2, 0.1, 1.15], [0.3, 0.2, 1.0]])
    new, facts = ss.snap_into_regions(verts, np.zeros((0, 3), np.int64), [_cube()], 0.1)
    np.testing.assert_allclose(new[0], [0.2, 0.1, 0.9], atol=1e-12)
    np.testing.assert_allclose(new[1], [1.0 - 0.1 / np.sqrt(2), 0.2, 1.0 - 0.1 / np.sqrt(2)], atol=1e-12)
    np.testing.assert_array_equal(new[2:], verts[2:])
    assert facts["moved"] == 2 and facts["in_band"] == 4 and facts["of"] == 5      # the vertex on the face is in the band, and does not move
    assert facts["largest_move_mm"] == pytest.approx(0.03 * np.sqrt(2) + 0.1, abs=1e-9) and facts["band_mm"] == 0.1 and facts["depth_mm"] == 0.1
    assert facts["turned_faces"] == 0 and facts["halving_rounds"] == 0


def test_snap_leaves_a_vertex_on_the_face_and_takes_the_direction_of_a_vertex_just_off_it():
    """A vertex within 1e-8 mm of the face counts as on it and stays, with no division by a length that is nearly 0 and no direction of
    noise (the cube [-1, 1]^3, band 0.1 mm). A vertex 1e-6 mm above the face has a direction that is good to 1e-9, and goes one band under the face."""
    verts = np.array([[0.2, 0.3, 1.0], [0.2, 0.3, 1.0 + 1e-12], [0.2, 0.3, 1.0 + 1e-10], [0.2, 0.3, 1.0 - 1e-10], [0.2, 0.3, 1.0 + 1e-6]])
    new, facts = ss.snap_into_regions(verts, np.zeros((0, 3), np.int64), [_cube()], 0.1)
    assert np.isfinite(new).all()
    np.testing.assert_array_equal(new[:4], verts[:4])
    np.testing.assert_allclose(new[4], [0.2, 0.3, 0.9], atol=1e-9)
    assert facts["moved"] == 1 and facts["in_band"] == 5 and facts["target_outside"] == 0


def test_snap_halves_the_move_of_a_face_that_it_would_turn_over():
    """A thin face on the face of the region: its three vertices are all within the band, and the full move puts them in a line (no area).
    The move is halved until the normal of the face is again the one before; no vertex goes the full way."""
    verts = np.array([[0.2, 0.5, 1.02], [0.5, 0.5, 1.06], [0.8, 0.5, 1.03]])
    faces = np.array([[0, 1, 2]])
    new, facts = ss.snap_into_regions(verts, faces, [_cube()], 0.1)
    n0 = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    n1 = np.cross(new[1] - new[0], new[2] - new[0])
    assert n0 @ n1 > 0
    assert facts["turned_faces"] == 1 and facts["halving_rounds"] >= 1 and facts["moved"] == 3
    assert (new[:, 2] < verts[:, 2]).all() and (new[:, 2] > 0.9).all()


def test_snap_takes_the_nearer_of_two_surfaces_and_survives_a_closest_point_that_is_not_finite(monkeypatch):
    other = _cube()
    other.apply_translation((2.07, 0, 0))                                          # its face x = 1.07 is 0.04 from the vertex, the cube face x = 1 is 0.03
    verts = np.array([[1.03, 0.2, 0.1], [1.05, 0.2, 0.4]])
    new, facts = ss.snap_into_regions(verts, np.zeros((0, 3), np.int64), [_cube(), other], 0.1)
    np.testing.assert_allclose(new[0], [0.9, 0.2, 0.1], atol=1e-12)                  # inside the first cube, one band under its face
    assert facts["moved"] == 2
    import trimesh.proximity as proximity
    monkeypatch.setattr(proximity, "closest_point", lambda mesh, pts: (np.full((len(pts), 3), np.nan), np.zeros(len(pts)), np.zeros(len(pts), np.int64)))
    new, facts = ss.snap_into_regions(verts, np.zeros((0, 3), np.int64), [_cube()], 0.1)
    np.testing.assert_array_equal(new, verts)
    assert facts["moved"] == 0


def _prism(poly, triangles, height=1.0):
    """A closed prism of a 2D triangulation, made like the region surfaces of `make_mesh_sdf`: through manifold3d (outward faces)."""
    from topopt.post import _to_manifold
    tm = trimesh.creation.extrude_triangulation(np.asarray(poly, float), np.asarray(triangles), height)
    return trimesh.Trimesh(*ss._from_manifold(_to_manifold(tm.vertices, tm.faces)), process=False)


def _in_polygon(poly, x, y):
    """Even-odd test in 2D, independent of trimesh."""
    c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            c = not c
    return c


def test_snap_does_not_move_the_vertices_inside_a_region_with_a_notch_narrower_than_90_degrees():
    """The prism has a notch with an opening of about 33 degrees: at its tip (0, 0) is a reflex edge with an interior angle of 327 degrees, and
    the normals of its two faces have a negative dot product. There the side by the normal of ONE face is wrong for a vertex inside the solid
    below the tip (on the first version, 2,513 of 4,000 such points moved, through the tip, to the other side of the notch). The side of every
    point comes from the whole solid now: no point inside moves, and every point in the notch (outside) goes into the solid."""
    notch = [(-1, -1), (1, -1), (1, 1), (0.3, 1), (0, 0), (-0.3, 1), (-1, 1)]
    tm = _prism(notch, [(4, 5, 6), (4, 6, 0), (4, 0, 1), (4, 1, 2), (4, 2, 3)])
    assert tm.is_watertight and tm.volume > 0
    band = 0.1
    rng = np.random.default_rng(1)
    angle, radius = rng.uniform(-np.pi, 0, 4000), rng.uniform(0.005, band, 4000)
    below = np.c_[radius * np.cos(angle), radius * np.sin(angle), np.full(4000, 0.5)]           # round the tip, on the side of the solid
    below = below[[_in_polygon(notch, x, y) for x, y, _z in below]]
    assert len(below) > 3000
    new, facts = ss.snap_into_regions(below, np.zeros((0, 3), np.int64), [tm], band)
    np.testing.assert_array_equal(new, below)
    assert facts["moved"] == 0 and facts["in_band"] == len(below)
    height = rng.uniform(0.005, band, 2000)
    inside_notch = np.c_[rng.uniform(-0.3, 0.3, 2000) * height, height, np.full(2000, 0.5)]    # in the gap of the notch, between its walls
    inside_notch = inside_notch[[not _in_polygon(notch, x, y) for x, y, _z in inside_notch]]
    assert len(inside_notch) > 1000
    new, facts = ss.snap_into_regions(inside_notch, np.zeros((0, 3), np.int64), [tm], band)
    assert facts["moved"] == len(inside_notch) and facts["target_outside"] == 0
    assert all(_in_polygon(notch, x, y) for x, y, _z in new)


def test_snap_leaves_a_vertex_whose_target_is_not_inside_a_region_thinner_than_one_band():
    """The slab is 0.05 mm thick, the band 0.1 mm. A vertex 0.04 mm above it would go one band under its top face, 0.05 mm beyond the far face
    (on the first version it did). It stays, and the fact counts it. A slab of 0.15 mm takes a vertex that is 0.04 mm above it inside."""
    thin = _prism([(-1, -1), (1, -1), (1, 1), (-1, 1)], [(0, 1, 2), (0, 2, 3)], 0.05)
    thin.apply_translation((0, 0, -0.025))
    vertex = np.array([[0.1, 0.2, 0.065]])
    new, facts = ss.snap_into_regions(vertex, np.zeros((0, 3), np.int64), [thin], 0.1)
    np.testing.assert_array_equal(new, vertex)
    assert facts["moved"] == 0 and facts["in_band"] == 1 and facts["target_outside"] == 1
    thick = _prism([(-1, -1), (1, -1), (1, 1), (-1, 1)], [(0, 1, 2), (0, 2, 3)], 0.15)
    thick.apply_translation((0, 0, -0.075))
    new, facts = ss.snap_into_regions(np.array([[0.1, 0.2, 0.115]]), np.zeros((0, 3), np.int64), [thick], 0.1)
    np.testing.assert_allclose(new[0], [0.1, 0.2, -0.025], atol=1e-7)     # the surface is float32
    assert facts["moved"] == 1 and facts["target_outside"] == 0


def _seam_inputs(tmp_path):
    from test_tet_clean_retry import SEAM_H, seam_inputs
    rho, labels, regions = seam_inputs(tmp_path, 8.0)
    return (lambda: ss.make_mesh_sdf(rho, labels, np.zeros(3), SEAM_H, up=post.UPSAMPLE, pad=post.PAD, **regions))


def test_a_failed_exact_step_keeps_the_smooth_mesh_where_the_marching_cubes_left_it(tmp_path, monkeypatch):
    """The snap is part of the exact step: if that step fails, the mesh is the smooth mesh, not moved (as before F4-27b), and the fact says so."""
    from test_tet_clean_retry import no_snap
    make = _seam_inputs(tmp_path)

    def boom(*a, **k):
        raise ValueError("no manifold today")
    monkeypatch.setattr(ss, "_manifold_of", boom)
    v1, f1, i1 = make()
    monkeypatch.setattr(ss, "snap_into_regions", no_snap)
    v0, f0, _i0 = make()
    assert i1["exact"]["union"] is False and i1["exact"]["error"] == "no manifold today"
    assert i1["snap"] == {"error": "the exact boolean step failed, so the mesh was not moved: ValueError: no manifold today"}
    np.testing.assert_array_equal(v1, v0)
    np.testing.assert_array_equal(f1, f0)


def test_a_failed_snap_leaves_the_mesh_unmoved_and_the_exact_step_runs(tmp_path, monkeypatch):
    from test_tet_clean_retry import no_snap
    make = _seam_inputs(tmp_path)
    monkeypatch.setattr(ss, "snap_into_regions", no_snap)
    v0, f0, i0 = make()

    def boom(*a, **k):
        raise RuntimeError("closest point failed")
    monkeypatch.setattr(ss, "snap_into_regions", boom)
    v1, f1, i1 = make()
    assert i1["snap"] == {"error": "RuntimeError: closest point failed"} and i1["exact"]["union"] is True and i1["exact"]["cut"] is True
    np.testing.assert_array_equal(v1, v0)
    np.testing.assert_array_equal(f1, f0)


def test_surfaces_of_the_regions_that_cannot_be_made_leave_the_mesh_unmoved(tmp_path, monkeypatch, caplog):
    from test_tet_clean_retry import no_snap
    make = _seam_inputs(tmp_path)
    monkeypatch.setattr(ss, "snap_into_regions", no_snap)
    v0, f0, _i0 = make()
    monkeypatch.undo()
    real, calls = ss._from_manifold, []

    def first_call_fails(man):
        calls.append(1)
        if len(calls) == 1:   # the first call makes the surface of the fixed and load regions
            raise ValueError("no surface today")
        return real(man)
    monkeypatch.setattr(ss, "_from_manifold", first_call_fails)
    with caplog.at_level("WARNING", logger="topopt.smooth_sdf"):
        v1, f1, i1 = make()
    assert i1["snap"] == {"error": "the exact surfaces of the regions were not made: ValueError: no surface today"}
    assert "the smooth mesh is not moved to the region faces" in caplog.text
    np.testing.assert_array_equal(v1, v0)
    np.testing.assert_array_equal(f1, f0)


@pytest.mark.parametrize("how", ["exact_step", "build"])
def test_volume_change_error_says_why_the_number_is_missing(tmp_path, monkeypatch, how):
    """`volume_change_mm3` is the volume of the last build minus the volume of one more build without the move. When that build fails, the key
    `volume_change_error` has the reason: the exact step of that build (a mesh without booleans is not a volume to compare with; on the first
    version the number was then wrong, and no key said why) or the build itself. The mesh and the other facts do not change."""
    make = _seam_inputs(tmp_path)
    v0, f0, i0 = make()
    assert "volume_change_mm3" in i0["snap"] and "volume_change_error" not in i0["snap"]
    builds, real_taubin, real_manifold = [], ss.taubin_pinned, ss._manifold_of

    def counting(*a, **k):
        builds.append(1)
        return real_taubin(*a, **k)
    monkeypatch.setattr(ss, "taubin_pinned", counting)
    make()
    last = len(builds)           # the build without the move is the last one
    builds.clear()

    def in_last_build(real, what):
        def wrapped(*a, **k):
            if len(builds) == last:
                raise what
            return real(*a, **k)
        return wrapped
    if how == "exact_step":
        monkeypatch.setattr(ss, "_manifold_of", in_last_build(real_manifold, ValueError("no manifold today")))
        reason = "the exact boolean step of the build without the move failed: no manifold today"
    else:
        monkeypatch.setattr(ss, "taubin_pinned", lambda *a, **k: (builds.append(1), in_last_build(real_taubin, RuntimeError("taubin today"))(*a, **k))[1])
        reason = "RuntimeError: taubin today"
    v1, f1, i1 = make()
    assert i1["snap"]["volume_change_error"] == reason and "volume_change_mm3" not in i1["snap"]
    assert {k: i1["snap"][k] for k in ("moved", "in_band", "of", "largest_move_mm")} == {k: i0["snap"][k] for k in ("moved", "in_band", "of", "largest_move_mm")}
    np.testing.assert_array_equal(v1, v0)
    np.testing.assert_array_equal(f1, f0)


def test_the_snap_moves_vertices_into_the_keep_out_slot_on_a_real_mesh_and_no_vertex_inside_every_region_moves(tmp_path, monkeypatch):
    """The real mesh of the seam part, with the plate and the seat (`fixed`, `load`) and the slot (`keep_out`): the smooth surface lies on the
    material side of the slot face, and the snap puts the vertices there one band into the slot (the exact cut then removes them). The unit
    tests above use a cube as the surface; this one shows the move on a real mesh, for the keep-out group."""
    make = _seam_inputs(tmp_path)
    seen, real = [], ss.snap_into_regions

    def spy(verts, faces, surfaces, band):
        out = real(verts, faces, surfaces, band)
        seen.append((np.array(verts), out[0], surfaces))
        return out
    monkeypatch.setattr(ss, "snap_into_regions", spy)
    _v, _f, info = make()
    verts, new, surfaces = seen[-1]
    assert len(surfaces) == 2                                             # the frozen union, then the keep-out union
    moved = np.flatnonzero((new != verts).any(1))
    assert len(moved) == info["snap"]["moved"] > 0
    sd = trimesh.proximity.signed_distance
    outside = [sd(tm, verts[moved]) < 0 for tm in surfaces]
    into_slot = outside[1] & (sd(surfaces[1], new[moved]) > 0)            # outside the slot before, inside it after
    assert into_slot.sum() > 0, "no vertex went into the keep-out slot"
    assert (outside[0] | outside[1]).all(), "a vertex that was inside every region moved"   # (a halved move can stay outside: no check after)


def test_a_part_with_no_stl_region_is_not_touched_by_the_snap(monkeypatch):
    """The voxel labels give the shape of the regions: no exact boolean, no snap, no fact. The function is never called."""
    def never(*a, **k):
        raise AssertionError("the snap ran on a part with no STL region")
    rho, labels = _rods_field()
    v0, f0, i0 = ss.make_mesh_sdf(rho, labels, [0, 0, 0], 1.0, up=2, pad=post.PAD)
    monkeypatch.setattr(ss, "snap_into_regions", never)
    v1, f1, i1 = ss.make_mesh_sdf(rho, labels, [0, 0, 0], 1.0, up=2, pad=post.PAD)
    assert i0["snap"] is None and i1["snap"] is None and i1["sources"]["frozen"] == "labels"
    assert trimesh.Trimesh(v0, f0, process=False).export(file_type="stl") == trimesh.Trimesh(v1, f1, process=False).export(file_type="stl")


def test_the_snap_runs_for_a_keep_out_stl_alone_and_is_in_the_record(tmp_path):
    """`mesh.snap` of metrics.json: a fact dict with an STL region (here the keep-out hole only), `null` without one."""
    _hole_run_with_stls(tmp_path, fixed_stl=False)
    rec = post.finish(tmp_path)["mesh"]
    assert rec["sources"] == {"frozen": "labels", "keep_out": "stl"}
    assert rec["snap"] is not None and "error" not in rec["snap"] and rec["snap"]["in_band"] > 0 and rec["snap"]["of"] > 0
    other = tmp_path / "labels_only"
    other.mkdir()
    make_hole_run(other)
    assert post.finish(other)["mesh"]["snap"] is None


# -- the global level shift when the volume does not follow it (X1) --
def _frozen_only_field(shape=(30, 20, 20)):
    """A fixed block and no free material: the frozen shape alone holds the surface."""
    labels = np.full(shape, post.DESIGN, np.uint8)
    labels[:3] = post.FIXED
    return np.zeros(shape, np.float32), labels


def test_design_with_no_free_material_gives_a_mesh_and_a_fact(caplog):
    """The marching-cubes volume of a frozen-only design is the same at every level shift (slope 0). The secant used to divide by it."""
    rho, labels = _frozen_only_field()
    v, f, info = ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=2, pad=post.PAD)
    assert info["watertight"] and info["extra_bodies_removed"] == 0
    rounds = info["calibration"]
    assert len(rounds) == 2   # the second round found the slope 0 and stopped: more rounds would build the same mesh
    assert "stopped" not in rounds[0] and "slope 0 mm2" in rounds[1]["stopped"]
    assert rounds[1]["final_mm3"] == rounds[0]["final_mm3"] and rounds[1]["mc_mm3"] == rounds[0]["mc_mm3"]
    for r in rounds:
        assert all(np.isfinite(r[k]) for k in ("delta_mm", "mc_mm3", "final_mm3", "target_mm3", "error"))
    assert abs(rounds[-1]["error"]) < 0.01   # the residual stays in the record: it is a fact, not hidden
    assert abs(post._signed_volume(v, f) - 3 * 20 * 20) < 0.02 * 3 * 20 * 20   # the block, to marching-cubes accuracy
    assert "calibration stopped in round 1" in caplog.text


def test_run_with_no_free_material_keeps_the_sdf_method(tmp_path):
    """A run stopped early (density 0.3 everywhere, below the 0.5 cut) with a plate off the voxel grid reached the divide by zero.
    `post` took the error as a failure of the SDF step and used `blur`."""
    run, labels = make_hole_run(tmp_path, h=1.2)
    org, h = np.asarray(run["origin"]), run["h"]
    trimesh.creation.box(bounds=[org, [org[0] + 2.975 * h, org[1] + 24 * h, org[2] + 24 * h]]).export(tmp_path / "plate.stl")
    trimesh.creation.box(bounds=[org + h * np.array([24, 10, 10]), org + h * np.array([27, 14, 14])]).export(tmp_path / "load.stl")
    run["yaml_path"] = str(tmp_path / "x.yaml")
    run["regions"] = {"design": "design.stl", "fixed": ["plate.stl"], "keep_out": [], "load": {"t": "load.stl"}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    rho = np.where(labels == post.DESIGN, 0.3, 0.0).astype(np.float32)   # the start density of a run with volume fraction 0.3
    for name in ("it_0000.bin", "it_0001.bin"):
        (tmp_path / "snapshots" / name).write_bytes((rho * 255).astype(np.uint8).tobytes())
    m = post.finish(tmp_path)
    rec = m["mesh"]
    assert rec["method"] == "sdf" and rec["fallback"] is None and m["watertight"] is True
    assert rec["calibration"][-1]["stopped"].startswith("the mesh volume stays")
    assert np.isfinite(rec["volume_error_pct"])
    mesh = load_stl(tmp_path / "result" / "final.stl")
    assert mesh.is_watertight and mesh.volume > 0
    assert len(mesh.split(only_watertight=False)) == 2   # the plate and the load block: no material joins them (a fact)


@pytest.mark.parametrize("taken", [1.0, 2.0])
def test_volume_target_that_is_not_above_zero_is_a_clear_error(monkeypatch, taken):
    """The bodies that were removed take the whole sharp volume (`taken` = 1) or more: no relative error exists.
    The step says so (the caller then uses `blur` and records the text); it used to divide by zero or return a record with an error of -2."""
    rho, labels = _rods_field()
    v_thr = ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=2, pad=post.PAD)[2]["volume_thresholded_mm3"]
    real = ss._keep_touching
    monkeypatch.setattr(ss, "_keep_touching", lambda verts, faces, touches: (*real(verts, faces, touches)[:2], 1, taken * v_thr))
    with pytest.raises(ValueError, match="volume target is not above 0"):
        ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=2, pad=post.PAD)


def test_secant_stops_when_the_shifted_target_is_not_above_zero(caplog):
    """A union shape outside the grid adds its volume to the measured volume only. The target for the next round goes below 0,
    and the relative error has no meaning there. The step stops with a fact, and does not build the same mesh again and again."""
    rho, labels = _rods_field()
    far = trimesh.creation.box(extents=[50.0, 50.0, 50.0])
    far.apply_translation([500.0, 0.0, 0.0])
    v, f, info = ss.make_mesh_sdf(rho, labels, np.zeros(3), 1.0, up=2, pad=post.PAD, extra_frozen=[far])
    rounds = info["calibration"]
    assert rounds[0]["final_mm3"] > 2 * rounds[0]["target_mm3"]   # the shape outside the grid is in the measured volume
    assert len(rounds) == 2 and rounds[1]["stopped"].startswith("the marching-cubes volume target of this round is -")
    assert "no relative error" in rounds[1]["stopped"] and "calibration stopped in round 1" in caplog.text
    assert info["watertight"] and np.isfinite(v).all()
