"""Test post-process (topopt/post.py) on fake runs and on a field with a hole."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fake_run  # noqa: E402
from topopt import post  # noqa: E402

KEYS = {
    "mass_design_space_g", "mass_final_g", "mass_saved_pct", "volume_final_mm3",
    "compliance_initial", "compliance_final", "stiffness_cases", "run_time_s",
    "iterations", "watertight", "triangles",
}


def read_preview(path):
    raw = Path(path).read_bytes()
    nv, nt = struct.unpack_from("<II", raw, 0)
    assert len(raw) == 8 + 12 * nv + 12 * nt
    verts = np.frombuffer(raw, "<f4", 3 * nv, 8).reshape(nv, 3)
    faces = np.frombuffer(raw, "<u4", 3 * nt, 8 + 12 * nv).reshape(nt, 3)
    return verts, faces


def load_stl(path):
    import trimesh

    return trimesh.load(path, file_type="stl")


@pytest.fixture(scope="module")
def fake_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    fake_run.run(out, spi=0.0, n_it=8)
    (out / "result" / "metrics.json").unlink()  # finish() must write it again
    return out


@pytest.fixture(scope="module")
def fake_metrics(fake_dir):
    return post.finish(fake_dir)


def test_fake_run_files(fake_dir, fake_metrics):
    res = fake_dir / "result"
    assert (res / "final.stl").stat().st_size > 84
    assert KEYS <= set(fake_metrics)
    on_disk = json.loads((res / "metrics.json").read_text())
    assert on_disk["triangles"] == fake_metrics["triangles"]
    assert isinstance(fake_metrics["watertight"], bool)
    assert isinstance(fake_metrics["triangles"], int)
    assert len(fake_metrics["stiffness_cases"]) == 2
    for c in fake_metrics["stiffness_cases"]:
        assert set(c) == {"name", "N_per_mm"}


def test_fake_status_done(fake_dir, fake_metrics):
    st = json.loads((fake_dir / "status.json").read_text())
    assert st["state"] == "done" and st["step"] == "done"
    assert not list(fake_dir.rglob("*.tmp"))


def test_fake_mesh_quality(fake_dir, fake_metrics):
    mesh = load_stl(fake_dir / "result" / "final.stl")
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume > 0
    assert fake_metrics["watertight"] is True
    assert abs(mesh.volume - fake_metrics["volume_final_mm3"]) / mesh.volume < 1e-3
    assert fake_metrics["triangles"] == len(mesh.faces)
    assert 0 < fake_metrics["mass_final_g"] < fake_metrics["mass_design_space_g"]


def test_preview_header(fake_dir, fake_metrics):
    verts, faces = read_preview(fake_dir / "result" / "preview.bin")
    assert len(faces) == fake_metrics["triangles"]
    assert faces.max() < len(verts)


def test_mesh_covers_fixed_region(fake_dir, fake_metrics):
    from scipy.ndimage import binary_erosion

    run = json.loads((fake_dir / "run.json").read_text())
    shape = tuple(run["shape"])
    labels = np.frombuffer((fake_dir / "labels.bin").read_bytes(), np.uint8).reshape(shape)
    core = binary_erosion(labels == post.FIXED, iterations=1)
    idx = np.argwhere(core)[::7]
    pts = np.asarray(run["origin"]) + run["h"] * (idx + 0.5)
    mesh = load_stl(fake_dir / "result" / "final.stl")
    assert mesh.contains(pts).all()


def test_missing_forces_gives_null(tmp_path):
    fake_run.run(tmp_path, spi=0.0, n_it=3)
    run = json.loads((tmp_path / "run.json").read_text())
    for c in run["load_cases"]:
        c.pop("forces", None)
    (tmp_path / "run.json").write_text(json.dumps(run))
    m = post.finish(tmp_path)
    assert all(c["N_per_mm"] is None for c in m["stiffness_cases"])


# -- a field with a hole and a loose island --
def make_hole_run(out: Path, h=1.2):
    shape = (40, 24, 24)
    labels = np.full(shape, post.DESIGN, np.uint8)
    labels[:3] = post.FIXED
    # Body is x < 27. Load box at its end. Empty gap x 27..30. Island x 31..35.
    labels[24:27, 10:14, 10:14] = post.LOAD
    labels[18:22, :, 10:14] = post.KEEP_OUT  # hole along y: 4 x 4 voxels (4.8 mm)
    labels[27:] = post.EMPTY
    labels[31:35, 4:8, 4:8] = post.DESIGN
    rho = np.ones(shape, np.float32)
    rho[labels == post.KEEP_OUT] = 0
    rho[labels == post.EMPTY] = 0
    rho[31:35, 4:8, 4:8] = 1
    (out / "snapshots").mkdir(parents=True)
    n_active = int((labels > 0).sum())
    run = {"format": 1, "name": "hole", "shape": list(shape), "origin": [10.0, -5.0, 3.0],
           "h": h, "n_active": n_active, "n_free_dofs": 1, "iterations": 1,
           "volume_fraction": 0.5,
           "load_cases": [{"name": "a", "weight": 1.0,
                           "forces": [{"region": "t", "force": [0.0, 0.0, -3.0], "center": [0, 0, 0]},
                                      {"region": "t", "force": [4.0, 0.0, 0.0], "center": [0, 0, 0]}]}],
           "mass_design_space_g": n_active * h ** 3 / 1000 * 1.24}
    (out / "run.json").write_text(json.dumps(run))
    (out / "labels.bin").write_bytes(labels.tobytes())
    (out / "snapshots" / "it_0000.bin").write_bytes((rho * 255).astype(np.uint8).tobytes())
    (out / "snapshots" / "it_0001.bin").write_bytes((rho * 255).astype(np.uint8).tobytes())
    rows = [{"iteration": i, "elapsed_s": 1.0 + i, "compliance": 10.0 / (i + 1),
             "compliance_cases": [10.0 / (i + 1)]} for i in range(2)]
    (out / "history.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run, labels


def test_hole_stays_open_and_island_removed(tmp_path):
    run, labels = make_hole_run(tmp_path)
    m = post.finish(tmp_path)
    mesh = load_stl(tmp_path / "result" / "final.stl")
    assert mesh.is_watertight and mesh.volume > 0
    assert len(mesh.split(only_watertight=False)) == 1  # island is gone
    org, h = np.asarray(run["origin"]), run["h"]
    # Points on the hole axis (all y) must be outside the solid.
    ys = np.linspace(1, 22, 8)
    cx, cz = 20.0 * h, 12.0 * h
    pts = np.array([[org[0] + cx, org[1] + y * h, org[2] + cz] for y in ys])
    assert not mesh.contains(pts).any()
    # The solid next to the hole is present.
    solid = np.array([[org[0] + 14 * h, org[1] + 12 * h, org[2] + 12 * h]])
    assert mesh.contains(solid).all()
    # Island bounds: no vertex beyond the island x range with y < 6 voxels.
    assert mesh.vertices[:, 0].max() < org[0] + 27 * h + 0.25 * h
    # Stiffness: F = (4, 0, -3), |F|^2 = 25, c = 10 / 2 = 5 at the last line.
    assert m["stiffness_cases"][0]["N_per_mm"] == pytest.approx(25.0 / 5.0)
    assert m["compliance_initial"] == pytest.approx(10.0)
    assert m["run_time_s"] >= 2.0


def test_blur_removes_voxel_steps():
    """A sloped boundary gives steps without blur. The blur must remove most of them."""
    import trimesh

    shape = (50, 12, 40)
    labels = np.full(shape, post.DESIGN, np.uint8)
    labels[:3] = post.FIXED
    x, _y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    rho = (z < 8 + 0.25 * x).astype(np.float32)  # shallow slope: long terraces
    rho[:3] = 1.0
    org = np.zeros(3)

    def roughness(blur):
        v, f, info = post.make_mesh(rho, labels, org, 1.2, blur=blur)
        assert info["watertight"]
        m = trimesh.Trimesh(v, f, process=False)
        c = m.triangles_center
        sel = (c[:, 0] > 15) & (c[:, 0] < 45) & (c[:, 1] > 3) & (c[:, 1] < 11) & (m.face_normals[:, 2] > 0.3)
        n = m.face_normals[sel]
        ang = np.arctan2(n[:, 0], n[:, 2])  # slope angle in the x-z plane
        return ang.std(), post._signed_volume(v, f)

    r0, v0 = roughness(0.0)
    r1, v1 = roughness(post.BLUR_SIGMA)
    assert r1 < 0.5 * r0
    assert abs(v1 - v0) / v0 < 0.05


def test_keep_out_cut_is_exact(tmp_path):
    import trimesh

    run, labels = make_hole_run(tmp_path)
    h, org = run["h"], np.asarray(run["origin"])
    cx, cz, radius = org[0] + 20 * h, org[2] + 12 * h, 3.5
    cyl = trimesh.creation.cylinder(radius=radius, height=60, sections=128)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cyl.apply_translation([cx, org[1] + 12 * h, cz])
    cyl.export(tmp_path / "hole.stl")
    run["yaml_path"] = str(tmp_path / "x.yaml")
    run["regions"] = {"keep_out": ["hole.stl"]}
    (tmp_path / "run.json").write_text(json.dumps(run))
    m = post.finish(tmp_path)
    mesh = load_stl(tmp_path / "result" / "final.stl")
    assert mesh.is_watertight and mesh.volume > 0 and m["watertight"]
    assert len(mesh.split(only_watertight=False)) == 1
    v = mesh.vertices
    d = np.hypot(v[:, 0] - cx, v[:, 2] - cz)
    near = d < 3.7  # hole wall vertices only
    assert near.any()
    assert abs(d[near].min() - radius) < 0.05  # the hole has the STL radius
    assert (np.abs(d - radius) < 0.05).sum() > 50  # many wall vertices on the cylinder
    assert m["triangles"] == len(mesh.faces)


def test_missing_keep_out_stl_is_skipped(tmp_path, caplog):
    run, _ = make_hole_run(tmp_path)
    run["yaml_path"] = str(tmp_path / "x.yaml")
    run["regions"] = {"keep_out": ["nope.stl"]}
    (tmp_path / "run.json").write_text(json.dumps(run))
    m = post.finish(tmp_path)
    assert m["watertight"] is True
    assert "keep-out STL missing" in caplog.text


# --- contains_chunks: the chunked test of points equals the one-call test (S1) -------------------------------


def four_bodies():
    """Four closed bodies that do not touch (three boxes and a sphere, 12 mm apart), and the points of one test set.

    The points are 40 random ones over the whole scene (inside some bodies, outside the others) and the centre of each body.
    """
    import trimesh

    parts = []
    for i, ext in enumerate(([4.0, 4.0, 4.0], [3.0, 2.0, 5.0], [6.0, 1.0, 1.0])):
        m = trimesh.creation.box(extents=ext)
        m.apply_translation([12.0 * i, 0.0, 0.0])
        parts.append(m)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    sphere.apply_translation([36.0, 0.0, 0.0])
    parts.append(sphere)
    union = trimesh.util.concatenate(parts)
    lo, hi = union.bounds
    pts = np.vstack([np.random.default_rng(4).uniform(lo - 1.0, hi + 1.0, (40, 3)), [[12.0 * i, 0, 0] for i in range(4)]])
    return parts, union, pts


@pytest.mark.parametrize("chunk", [1, 2, 7, 44, 1000, None])
def test_contains_chunks_equal_the_one_call_result_on_several_bodies(chunk):
    parts, union, pts = four_bodies()
    bodies = union.split(only_watertight=False)
    assert len(bodies) == 4
    inside_counts = []
    for b in bodies:
        ref = np.asarray(b.contains(pts), bool)
        got = np.concatenate(list(post.contains_chunks(b, pts, chunk=chunk)))
        assert got.dtype == bool and got.shape == ref.shape and np.array_equal(got, ref)
        inside_counts.append(int(ref.sum()))
    assert all(c >= 1 for c in inside_counts) and any(c < len(pts) for c in inside_counts)   # points inside and outside, so not vacuous
    verts, faces = union.vertices, union.faces
    assert np.array_equal(post.points_inside(verts, faces, pts), np.asarray(union.contains(pts), bool))


def test_contains_chunks_sizes_default_and_odd_input(monkeypatch):
    class Stub:
        def __init__(self):
            self.calls = []

        def contains(self, pts):
            self.calls.append(len(pts))
            return pts[:, 0] < 0

    pts = np.ones((10, 3))
    stub = Stub()
    assert len(list(post.contains_chunks(stub, pts, chunk=4))) == 3 and stub.calls == [4, 4, 2]
    stub = Stub()
    monkeypatch.setattr(post, "CONTAINS_CHUNK", 3)
    list(post.contains_chunks(stub, pts))
    assert stub.calls == [3, 3, 3, 1]                                       # the default is the module constant
    stub = Stub()
    assert list(post.contains_chunks(stub, np.zeros((0, 3)))) == [] and stub.calls == []
    import trimesh

    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    assert post.points_inside(box.vertices, box.faces, np.zeros((0, 3))).shape == (0,)
    with pytest.raises(ValueError):                                         # a shape that is not (n, 3): the error of trimesh
        post.points_inside(box.vertices, box.faces, [0.0, 0.0, 0.0])
