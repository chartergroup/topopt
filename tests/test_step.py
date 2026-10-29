"""Tests for `topopt step` (CONTRACTS.md section 42): a faceted STEP file from a finished run.

The tests that write a STEP file need OpenCASCADE (the extra `step`) and are skipped without it (`needs_ocp`). The tests of the
missing-extra message, of the numbers and of the import rule run without OpenCASCADE: they use a child Python in which OCP cannot be
imported, or they do not touch OCP.

Parts: the 10 x 20 x 30 mm box of `tests/fixtures/check_part/run` (12 triangles, 6000 mm3, so the volume is exact), boxes built here
(a hollow box, two boxes, an inside-out box, an open box), and a block with rounded edges and a hole, made here with marching cubes on a
signed distance field at 0.5 mm (`part_mesh`: 4,712 triangles, flat faces, a round hole and rounded edges, like the mesh of a run).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from helpers import needs_ocp
from topopt import step
from topopt.cli import main
from topopt.config import UserError

REPO = Path(__file__).resolve().parent.parent
BOX_RUN = REPO / "tests" / "fixtures" / "check_part" / "run"      # final.stl: a box of 10 x 20 x 30 mm
REPORT_KEYS = {"format", "state", "run_dir", "mesh", "input", "settings", "simplify", "ocp", "output", "merge_planes", "build", "model",
               "volume", "deviation", "invalid_parts", "checks", "checks_failed", "options", "timing_s", "cpu_s", "peak_rss_bytes", "peak_rss_mb_after_phase",
               "report_path"}
CHECKS = {"read_back", "topology", "before_write", "valid_solid", "closed", "units", "schema"}


def make_run(root, mesh=None, name="run", run_json=None, state="done", part_only=None):
    """A run folder with `result/final.stl` (a trimesh, bytes, or a path), and optionally run.json, status.json and part_only.stl."""
    run = Path(root) / name
    (run / "result").mkdir(parents=True)
    def put(target, data):
        if isinstance(data, (str, Path)):
            shutil.copy(data, target)
        elif isinstance(data, bytes):
            target.write_bytes(data)
        else:
            target.write_bytes(data.export(file_type="stl"))
    put(run / "result" / "final.stl", mesh if mesh is not None else BOX_RUN / "result" / "final.stl")
    if part_only is not None:
        put(run / "result" / "part_only.stl", part_only)
    if run_json is not None:
        (run / "run.json").write_text(json.dumps(run_json))
    if state:
        (run / "status.json").write_text(json.dumps({"state": state, "iteration": 1, "step": state, "elapsed_s": 1.0, "message": ""}))
    return run


def tree(folder):
    """{relative path: sha256} of every file under the folder."""
    out = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file():
            out[p.relative_to(folder).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def hollow_box():
    outer, inner = trimesh.creation.box((30, 30, 30)), trimesh.creation.box((10, 10, 10))
    inner.invert()
    return trimesh.util.concatenate([outer, inner])      # a cavity: 27000 - 1000 = 26000 mm3


def two_boxes():
    a, b = trimesh.creation.box((10, 20, 30)), trimesh.creation.box((10, 10, 10))
    b.apply_translation((50, 0, 0))
    return trimesh.util.concatenate([a, b])              # 6000 + 1000 mm3


def inside_out_box():
    box = trimesh.creation.box((10, 20, 30))
    box.invert()
    return box


def open_box():
    box = trimesh.creation.box((10, 20, 30))
    return trimesh.Trimesh(box.vertices, box.faces[:-1], process=False)    # one triangle missing: three open edges


def inverted(mesh):
    out = mesh.copy()
    out.invert()
    return out


def hollow_inverted():
    return inverted(hollow_box())      # the outer shell turned inward, the cavity turned outward: 26000 mm3


def one_body_inverted():
    a, b = trimesh.creation.box((10, 20, 30)), trimesh.creation.box((10, 10, 10))
    b.apply_translation((50, 0, 0))
    return trimesh.util.concatenate([a, inverted(b)])      # 6000 + 1000 mm3, the second box inside out


def zero_area_box():
    """A closed box with one zero-area triangle (A, B, M with M the midpoint of AB): a T-junction closed by a cap."""
    import numpy as np
    box = trimesh.creation.box((10, 20, 30))
    box.merge_vertices()
    V, F = np.asarray(box.vertices, float).tolist(), np.asarray(box.faces).tolist()
    owner = {(f[k], f[(k + 1) % 3]): i for i, f in enumerate(F) for k in range(3)}
    a, b = next((a, b) for (a, b) in owner if (b, a) in owner)
    t2 = owner[(b, a)]
    V.append(((np.array(V[a]) + np.array(V[b])) / 2.0).tolist())
    m = len(V) - 1
    d = next(x for x in F[t2] if x not in (a, b))
    F[t2] = [b, m, d]
    F.extend([[m, a, d], [b, a, m]])      # the cap has the edges B->A, A->M and M->B: the mesh stays closed
    return trimesh.Trimesh(np.array(V), np.array(F), process=False)


def turned_triangle_box():
    import numpy as np
    box = trimesh.creation.box((10, 20, 30))
    faces = np.asarray(box.faces).copy()
    faces[0] = faces[0][::-1]        # one triangle turned: the edge it shares is run twice in the same direction
    return trimesh.Trimesh(box.vertices, faces, process=False)


def part_mesh():
    """A block of 40 x 24 x 8 mm with rounded edges and a round hole of 5 mm radius, as a marching-cubes mesh (0.5 mm cells, 4,712 triangles)."""
    from skimage import measure
    h = 0.5
    xs, ys, zs = (np.arange(-a, a + h, h) for a in (24, 16, 8))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    q = np.stack([np.abs(X) - 20, np.abs(Y) - 12, np.abs(Z) - 4], -1)
    box = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - 2.0
    sdf = np.maximum(box, -(5.0 - np.hypot(X - 8, Y)))
    v, f, _, _ = measure.marching_cubes(-sdf, 0.0, spacing=(h, h, h))
    return trimesh.Trimesh(v + np.array([xs[0], ys[0], zs[0]]), f[:, ::-1])      # turned, so that the normals point out


@pytest.fixture(scope="module")
def hook_runs(tmp_path_factory):
    """The marching-cubes part (`part_mesh`, cells of 0.5 mm) exported twice with a tolerance of 0.12 mm: with and without merged planes.
    One export each for the module."""
    root = tmp_path_factory.mktemp("part")
    mesh = part_mesh()
    merged = step.export_step(make_run(root, mesh, "merged"), tolerance_mm=0.12)
    plain = step.export_step(make_run(root, mesh, "plain"), tolerance_mm=0.12, merge_planes=False)
    return merged, plain


# --- without OpenCASCADE -------------------------------------------------------------------------------------------------------


def child(code, *args, env=None):
    """Run `code` in a child Python (the repo on the path) and return the CompletedProcess."""
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO), **(env or {})})


NO_OCP = "import sys; sys.modules['OCP'] = None; from topopt.cli import main; sys.exit(main(sys.argv[1:]))"


def test_without_the_extra_one_message_with_the_install_command_and_exit_code_1(tmp_path):
    run = make_run(tmp_path)
    done = child(NO_OCP, "step", str(run))
    assert done.returncode == 1
    assert "Traceback" not in done.stderr and "Traceback" not in done.stdout
    assert done.stderr.startswith("error: ")
    assert 'uv pip install -e ".[step]"' in done.stderr and "cadquery-ocp-novtk==" in done.stderr
    assert "macOS" in done.stderr and "Linux" in done.stderr
    assert "not installed" in done.stderr
    assert tree(run) == tree(make_run(tmp_path, name="same"))      # nothing was written


def test_the_options_that_write_stop_without_the_extra_and_the_estimate_does_not(tmp_path):
    run = make_run(tmp_path, part_only=trimesh.creation.box((5, 5, 5)))
    for extra in (["--json"], ["--part-only"], ["--out", str(tmp_path / "x.step")]):
        done = child(NO_OCP, "step", str(run), *extra)
        assert done.returncode == 1 and "optional extra `step`" in done.stderr, extra
    before = tree(run)
    text = child(NO_OCP, "step", str(run), "--estimate-only")
    assert text.returncode == 0 and "estimate only, nothing is written" in text.stdout and "Traceback" not in text.stderr
    report = json.loads(child(NO_OCP, "step", str(run), "--estimate-only", "--json").stdout)
    assert report["state"] == "estimate" and report["ocp"] is None and report["estimate"]["faces"] == 12
    assert tree(run) == before


def test_an_extra_that_does_not_load_says_so():
    text = step.install_text("ImportError: libfoo.so: cannot open shared object file")
    assert "installed but does not load" in text and "libfoo.so" in text and 'uv pip install -e ".[step]"' in text


def test_the_core_does_not_import_ocp():
    code = ("import sys, topopt, topopt.cli, topopt.step, topopt.doctor; "
            "print(any(m == 'OCP' or m.startswith('OCP.') for m in sys.modules))")
    done = child(code)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_the_mcp_server_does_not_import_ocp():
    pytest.importorskip("mcp")
    done = child("import sys, topopt_mcp.server; print(any(m == 'OCP' or m.startswith('OCP.') for m in sys.modules))")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_the_extra_is_declared_with_a_pin():
    import tomllib
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert extras["step"] == [f"{step.OCP_DISTRIBUTION}=={step.OCP_PIN}"]


def info_of(folder, run=None):
    return {"run_dir": Path(folder), "run": run or {}}


def test_the_tolerance_default_comes_from_the_voxel_size(tmp_path):
    assert step.resolve_tolerance(None, info_of(tmp_path, {"h": 2.0})) == (
        pytest.approx(0.1), "default: the voxel size of the run (2 mm, run.json h) / 20")
    assert step.resolve_tolerance("0.3", info_of(tmp_path, {"h": 2.0})) == (0.3, "option")
    assert step.resolve_tolerance(0, info_of(tmp_path, {"h": 2.0})) == (0.0, "option")


def test_a_folder_with_no_voxel_size_gets_no_simplification_and_the_reason(tmp_path):
    for run in ({}, {"h": "x"}, {"h": 0}, {"h": -1.0}, {"h": True}):
        tol, why = step.resolve_tolerance(None, info_of(tmp_path, run))
        assert tol == 0.0 and "no voxel size" in why and "no simplification" in why, run
    assert not hasattr(step, "DEFAULT_TOLERANCE_MM")


def test_the_voxel_size_is_read_from_the_verify_report_when_run_json_has_none(tmp_path):
    (tmp_path / "verify").mkdir()
    (tmp_path / "verify" / "verify.json").write_text(json.dumps({"run": {"name": "x", "voxel_size_mm": 1.2}}))
    assert step.voxel_size(info_of(tmp_path)) == (1.2, "verify/verify.json run.voxel_size_mm")
    tol, why = step.resolve_tolerance(None, info_of(tmp_path))
    assert tol == pytest.approx(0.06) and "verify/verify.json" in why
    assert step.voxel_size(info_of(tmp_path, {"h": 2.0})) == (2.0, "run.json h")      # run.json wins
    (tmp_path / "verify" / "verify.json").write_text("not json")
    assert step.voxel_size(info_of(tmp_path)) == (None, None)


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "abc", "", "-0.0001"])
def test_a_bad_tolerance_is_a_user_error(value, tmp_path):
    with pytest.raises(UserError, match="tolerance_mm"):
        step.resolve_tolerance(value, info_of(tmp_path))


def test_a_bad_schema_is_a_user_error():
    with pytest.raises(UserError, match="AP214 or AP242"):
        step.resolve_schema("AP203")
    assert step.resolve_schema("ap-242") == "AP242" and step.resolve_schema(None) == "AP214"


def test_input_errors_come_before_the_extra_is_needed(tmp_path):
    with pytest.raises(UserError, match="not a folder"):
        step.export_step(tmp_path / "nothing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UserError, match="no result/final.stl"):
        step.export_step(empty)
    run = make_run(tmp_path, state="running")
    with pytest.raises(UserError, match="not finished"):
        step.export_step(run)
    run2 = make_run(tmp_path, name="r2")
    with pytest.raises(UserError, match="part_only.stl.*regions.not_printed"):
        step.export_step(run2, part_only=True)


def shells_of(mesh):
    import numpy as np
    mesh = mesh.copy()
    mesh.merge_vertices()
    return step.mesh_shells(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))


@pytest.mark.parametrize("mesh, kinds, flips, volume", [
    (hollow_box, ["body", "void"], [False, False], 26000.0),
    (two_boxes, ["body", "body"], [False, False], 7000.0),
    (inside_out_box, ["body"], [True], 6000.0),
    (hollow_inverted, ["body", "void"], [True, True], 26000.0),         # both shells turned: the cavity is still a cavity
    (one_body_inverted, ["body", "body"], [False, True], 7000.0),      # one body inside out: its volume still counts
])
def test_the_mesh_shells_classify_after_the_orientation_and_orient_bodies_out_and_voids_in(mesh, kinds, flips, volume):
    sh = shells_of(mesh())
    assert sorted(sh["kind"]) == sorted(kinds) and sorted(sh["flip"].tolist()) == sorted(flips)
    assert sh["volume_mm3"] == pytest.approx(volume, rel=1e-12)
    assert not sh["not_manifold"].any()
    for s, kind in enumerate(sh["kind"]):      # a void is held by a body; these meshes have no body inside another shell
        assert (sh["kind"][sh["parent"][s]] == "body") if kind == "void" else sh["parent"][s] == -1


def test_a_body_inside_a_cavity_is_a_body_and_a_cavity_in_it_is_a_void():
    import numpy as np
    meshes = [trimesh.creation.box((40, 40, 40)), inverted(trimesh.creation.box((30, 30, 30))), trimesh.creation.box((20, 20, 20)),
              inverted(trimesh.creation.box((10, 10, 10)))]     # a hollow box with a hollow island in the cavity
    sh = step.mesh_shells(*(lambda m: (np.asarray(m.vertices, float), np.asarray(m.faces)))(trimesh.util.concatenate(meshes)))
    by_volume = {round(abs(v)): s for s, v in enumerate(sh["volume"])}
    assert [sh["kind"][by_volume[x]] for x in (64000, 27000, 8000, 1000)] == ["body", "void", "body", "void"]
    assert sh["volume_mm3"] == pytest.approx(64000 - 27000 + 8000 - 1000, rel=1e-12)
    # each void is held by the shell with the smallest volume that holds it: the cavity by the hollow box, the island's cavity by the island
    assert sh["parent"][by_volume[27000]] == by_volume[64000] and sh["parent"][by_volume[1000]] == by_volume[8000]


def test_many_separate_bodies_are_found_one_by_one_and_an_open_mesh_has_a_positive_volume():
    boxes = []
    for i in range(12):
        for j in range(10):
            b = trimesh.creation.box((1, 1, 1))
            b.apply_translation((3 * i, 3 * j, 0))
            boxes.append(b)
    sh = shells_of(trimesh.util.concatenate(boxes))
    assert sh["count"] == 120 and set(sh["kind"]) == {"body"} and not sh["flip"].any() and (sh["parent"] == -1).all()
    assert sh["volume_mm3"] == pytest.approx(120.0, rel=1e-12)
    open_inverted = shells_of(inverted(open_box()))      # its winding is unknown: it counts as a body, whatever the winding is
    assert open_inverted["not_manifold"].all() and not open_inverted["flip"].any() and open_inverted["volume_mm3"] > 0


def test_shells_that_touch_along_an_edge_are_marked_not_manifold():
    a, b = trimesh.creation.box((10, 10, 10)), trimesh.creation.box((10, 10, 10))
    b.apply_translation((10, 10, 0))      # the two boxes share one edge: that edge has 4 faces
    sh = shells_of(trimesh.util.concatenate([a, b]))
    assert sh["count"] == 2 and sh["not_manifold"].tolist() == [True, True]


def test_the_text_of_a_report_and_the_exit_code():
    report = {"state": "written", "run_dir": "/r", "mesh": "final.stl",
              "input": {"triangles": 12, "volume_mm3": 6000.0, "closed": True, "open_edges": 0, "nonmanifold_edges": 0},
              "settings": {"tolerance_mm": 0.0, "tolerance_source": "option"},
              "simplify": {"applied": False, "note": "tolerance 0"}, "merge_planes": {"applied": True, "faces_before": 12, "faces_after": 6},
              "output": {"path": "/r/result/final.step", "bytes": 4096, "schema": "AP214", "unit": "mm"},
              "model": {"solids": 1, "faces": 6, "planar_faces": 6, "edges": 12, "vertices": 8},
              "volume": {"step_mm3": 6000.0, "stl_mm3": 6000.0, "step_vs_stl_pct": 0.0},
              "deviation": {"step_to_stl": {"max_mm": 0.0, "mean_mm": 0.0}, "stl_to_step": {"max_mm": 0.0, "mean_mm": 0.0}},
              "checks": {"closed": {"state": "fail", "reason": "3 edge(s) with 1 face"}}, "checks_failed": ["closed"],
              "options": ["option: --no-merge-planes"], "timing_s": {"total": 1.0}, "cpu_s": 1.0, "peak_rss_bytes": 5e8,
              "report_path": "/r/result/final.step.json"}
    text = step.format_text(report)
    assert "result/final.step" in text and "FAILED closed: 3 edge(s) with 1 face" in text and "--no-merge-planes" in text
    assert "report: result/final.step.json" in text
    assert step.exit_code(report) == step.EXIT_CHECK_FAILED == 3
    assert step.exit_code({**report, "checks_failed": []}) == 0


def test_the_text_of_the_flagged_parts():
    text = step._invalid_text({"vertices": 0, "edges": 0, "faces": 1, "statuses": ["BadOrientationOfSubshape"],
                               "smallest_face_area_mm2": 1.08e-8, "first_faces": []})
    assert text == "; flagged: 1 face(s) (BadOrientationOfSubshape); the smallest flagged face has 1.08e-08 mm2"
    assert step._invalid_text(None) == ""
    assert "the solid or its shell" in step._invalid_text({"vertices": 0, "edges": 0, "faces": 0, "statuses": [],
                                                           "smallest_face_area_mm2": None, "first_faces": []})


def test_the_estimate_grows_with_the_faces_and_says_where_it_comes_from():
    small, large = step.estimate_costs(10_000, 20_000), step.estimate_costs(100_000, 200_000)
    for key in ("step_bytes", "seconds", "peak_rss_bytes"):
        assert 0 < small[key] < large[key]
    assert small["step_bytes"] == pytest.approx(10_000 * step.ESTIMATE_STEP_BYTES_PER_FACE)
    assert "measured runs" in small["basis"] and "CONTRACTS.md section 42.7" in small["basis"]
    assert "the data of the fit" in small["basis"] and "too low" in small["basis"]      # the error is in-sample; the worst errors are low
    text = step.format_text({"state": "estimate", "run_dir": "/r", "mesh": "final.stl",
                             "input": {"triangles": 20_000, "volume_mm3": 1.0, "closed": True, "open_edges": 0, "nonmanifold_edges": 0},
                             "settings": {"tolerance_mm": 0.1, "tolerance_source": "option"},
                             "simplify": {"applied": True, "triangles_before": 20_000, "triangles_after": 10_000}, "estimate": small})
    assert "estimate only, nothing is written" in text and "10,000" in text and "basis:" in text


def test_the_options_depend_on_the_state_of_the_run():
    checks = {"closed": {"state": "fail", "reason": "3 edge(s) with 1 face"}}
    plain = step._options(["closed"], checks, merged=False, tolerance_mm=0.0)
    assert plain[0].startswith("check closed failed") and "written as it is" in plain[1]
    assert any("tolerance above 0" in line for line in plain) and any("leave out --no-merge-planes" in line for line in plain)
    merged = step._options(["closed"], checks, merged=True, tolerance_mm=0.05)
    assert any("different --tolerance-mm" in line for line in merged) and any("one face for each triangle" in line for line in merged)
    assert step._options([], checks, merged=True, tolerance_mm=0.05) == []


# --- the box: an exact volume ---------------------------------------------------------------------------------------------------


@needs_ocp
def test_box_round_trip_exact_volume_and_merged_planes(tmp_path):
    merged = step.export_step(make_run(tmp_path, name="m"), tolerance_mm=0)
    plain = step.export_step(make_run(tmp_path, name="p"), tolerance_mm=0, merge_planes=False)
    assert merged["model"]["faces"] == 6 and merged["model"]["planar_faces"] == 6      # the 12 triangles join into the 6 faces of the box
    assert plain["model"]["faces"] == 12 and plain["merge_planes"] == {"applied": False}
    assert merged["merge_planes"] == {"applied": True, "faces_before": 12, "faces_after": 6}
    for r in (merged, plain):
        assert r["state"] == "written" and r["checks_failed"] == []
        assert set(r["checks"]) == CHECKS and all(c["state"] == "pass" for c in r["checks"].values())
        assert r["volume"]["step_mm3"] == pytest.approx(6000.0, rel=1e-12) and r["volume"]["stl_mm3"] == pytest.approx(6000.0, rel=1e-12)
        assert r["volume"]["roundtrip_error"] < 1e-12
        assert r["model"]["solids"] == 1 and r["model"]["free_edges"] == 0 and r["model"]["overshared_edges"] == 0
        assert r["build"]["in_memory"] == {"valid": True, "solids": 1, "smallest_solid_mm3": pytest.approx(6000.0, rel=1e-12),
                                           "invalid_parts": None}
        assert r["deviation"]["step_to_stl"]["max_mm"] < 1e-6 and r["deviation"]["stl_to_step"]["max_mm"] < 1e-6
        assert r["output"]["unit"] == "mm" and r["output"]["schema"] == "AP214"
        assert r["simplify"]["applied"] is False and r["settings"]["tolerance_source"] == "option"


@needs_ocp
def test_the_step_file_is_a_step_file_with_the_run_name(tmp_path):
    run = make_run(tmp_path, run_json={"name": "my_hook", "h": 2.0})
    r = step.export_step(run, tolerance_mm=0)
    text = Path(r["output"]["path"]).read_text()
    assert text.startswith("ISO-10303-21;") and "FILE_SCHEMA" in text and "MANIFOLD_SOLID_BREP" in text
    name = r["output"]["product_name"]      # the name that the file holds: OpenCASCADE adds a number (the count of products of the process)
    assert re.fullmatch(r"my_hook \d+", name) and f"PRODUCT('{name}'" in text and r["output"]["notes"] == []
    assert "PCURVE" not in text      # the planes carry no pcurves (half the size)
    assert r["output"]["bytes"] == Path(r["output"]["path"]).stat().st_size


@needs_ocp
def test_the_report_keys_and_the_report_file(tmp_path):
    run = make_run(tmp_path)
    r = step.export_step(run, tolerance_mm=0)
    assert REPORT_KEYS <= set(r)
    saved = json.loads((run / "result" / "final.step.json").read_text())
    assert saved == {k: v for k, v in r.items() if k != "report_path"}
    assert r["report_path"] == str(run / "result" / "final.step.json") and not (run / "result" / "step.json").exists()
    assert set(r["input"]) >= {"path", "bytes", "sha256", "triangles", "vertices", "volume_mm3", "closed", "bodies", "voids"}
    assert set(r["settings"]) == {"tolerance_mm", "tolerance_source", "merge_planes", "schema", "part_only"}
    assert set(r["timing_s"]) >= {"read_stl", "simplify", "build", "check_before_write", "write", "read_back", "check", "deviation", "total"}
    assert set(r["deviation"]) == {"method", "step_to_stl", "stl_to_step"} and r["deviation"]["method"] == "sampled"
    assert set(r["deviation"]["step_to_stl"]) == {"max_mm", "mean_mm", "points"}


@needs_ocp
def test_the_run_folder_keeps_every_file_except_the_new_outputs(tmp_path):
    run = make_run(tmp_path, run_json={"name": "x", "h": 2.0}, part_only=BOX_RUN / "result" / "final.stl")
    (run / "result" / "metrics.json").write_text("{}")
    before = tree(run)
    step.export_step(run, tolerance_mm=0)
    step.export_step(run, tolerance_mm=0, part_only=True)
    step.export_step(run, tolerance_mm=0, estimate_only=True)
    after = tree(run)
    assert set(after) - set(before) == {"result/final.step", "result/part_only.step", "result/final.step.json", "result/part_only.step.json"}
    assert {k: v for k, v in after.items() if k in before} == before
    assert not [p for p in run.rglob("*") if p.name.endswith(".tmp")]


@needs_ocp
def test_estimate_only_writes_nothing_and_counts_the_faces(tmp_path):
    run = make_run(tmp_path, run_json={"name": "x", "h": 2.0})
    before = tree(run)
    r = step.export_step(run, estimate_only=True)
    assert r["state"] == "estimate" and r["estimate"]["faces"] == 12 and "output" not in r
    assert {"faces", "step_bytes", "seconds", "peak_rss_bytes", "basis"} <= set(r["estimate"])
    assert r["settings"]["tolerance_mm"] == pytest.approx(0.1)
    assert tree(run) == before


@needs_ocp
def test_a_triangle_turned_back_is_flagged_in_memory_and_the_flagged_parts_are_a_lookup(tmp_path):
    import numpy as np
    mesh = turned_triangle_box()
    ocp = step._import_ocp()
    shape, _ = step.build_brep(ocp, np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
    analyzer = ocp.BRepCheck_Analyzer(shape)
    assert not analyzer.IsValid()
    flagged = step.invalid_parts(ocp, analyzer, shape)
    assert set(flagged) == {"vertices", "edges", "faces", "statuses", "smallest_face_area_mm2", "first_faces"}
    run = make_run(tmp_path, mesh)
    r = step.export_step(run, tolerance_mm=0)
    # The reader of the file turns the faces that point the wrong way (OCP 7.9.3, the pinned wheel), so the file reads as a valid solid
    # and only the check of the model in memory sees the turned triangle.
    assert r["checks_failed"] == ["before_write"] and r["input"]["closed"] is False and (run / "result" / "final.step").is_file()
    assert r["build"]["in_memory"]["valid"] is False and r["build"]["in_memory"]["invalid_parts"] is not None
    assert "not valid" in r["checks"]["before_write"]["reason"] and "flagged:" in r["checks"]["before_write"]["reason"]
    assert r["checks"]["valid_solid"]["state"] == "pass"
    assert r["invalid_parts"] is None and step.export_step(make_run(tmp_path, name="ok"), tolerance_mm=0)["invalid_parts"] is None


@needs_ocp
def test_out_is_a_file_or_an_existing_folder(tmp_path):
    run = make_run(tmp_path)
    target = tmp_path / "elsewhere"
    target.mkdir()
    a = step.export_step(run, tolerance_mm=0, out=target)
    b = step.export_step(run, tolerance_mm=0, out=target / "my.step")
    assert Path(a["output"]["path"]) == target / "final.step" and Path(b["output"]["path"]) == target / "my.step"
    assert (target / "final.step").is_file() and (target / "my.step").is_file()
    assert not (run / "result" / "final.step").exists() and not (run / "result" / "final.step.json").exists()      # the run folder keeps no file
    assert json.loads((target / "final.step.json").read_text())["output"]["path"] == str(target / "final.step")
    assert json.loads((target / "my.step.json").read_text())["output"]["path"] == str(target / "my.step")
    with pytest.raises(UserError, match="folder of --out does not exist"):
        step.export_step(run, tolerance_mm=0, out=tmp_path / "no" / "such" / "x.step")


@needs_ocp
def test_part_only_wraps_part_only_stl(tmp_path):
    small = trimesh.creation.box((5, 5, 5))
    run = make_run(tmp_path, part_only=small)
    r = step.export_step(run, tolerance_mm=0, part_only=True)
    assert r["mesh"] == "part_only.stl" and Path(r["output"]["path"]).name == "part_only.step"
    assert r["volume"]["step_mm3"] == pytest.approx(125.0, rel=1e-12) and r["settings"]["part_only"] is True
    assert "part only" in r["output"]["product_name"]


@needs_ocp
def test_ap242_schema(tmp_path):
    r = step.export_step(make_run(tmp_path), tolerance_mm=0, schema="AP242")
    assert r["output"]["schema"] == "AP242" and r["checks"]["schema"]["state"] == "pass" and r["checks_failed"] == []
    assert "242" in Path(r["output"]["path"]).read_text()[:2000]


# --- bodies, voids, inside-out and open meshes ---------------------------------------------------------------------------------


@needs_ocp
def test_a_cavity_is_an_inner_shell_of_one_solid(tmp_path):
    r = step.export_step(make_run(tmp_path, hollow_box()), tolerance_mm=0)
    assert r["build"]["bodies"] == 1 and r["build"]["voids"] == 1
    assert r["model"]["solids"] == 1 and r["model"]["shells"] == 2
    assert r["volume"]["step_mm3"] == pytest.approx(26000.0, rel=1e-12) and r["checks_failed"] == []


@needs_ocp
def test_two_bodies_are_two_solids(tmp_path):
    r = step.export_step(make_run(tmp_path, two_boxes()), tolerance_mm=0)
    assert r["model"]["solids"] == 2 and r["input"]["bodies"] == 2
    assert r["volume"]["step_mm3"] == pytest.approx(7000.0, rel=1e-12) and r["checks_failed"] == []


@needs_ocp
def test_an_inside_out_body_is_turned_right_side_out(tmp_path):
    r = step.export_step(make_run(tmp_path, inside_out_box()), tolerance_mm=0)
    assert r["build"]["inside_out_bodies"] == 1
    assert r["volume"]["step_mm3"] == pytest.approx(6000.0, rel=1e-12) and r["checks_failed"] == []


@needs_ocp
def test_an_open_mesh_is_a_failed_check_and_the_file_is_still_written(tmp_path, capsys):
    run = make_run(tmp_path, open_box())
    code = main(["step", str(run), "--tolerance-mm", "0"])
    out = capsys.readouterr().out
    assert code == step.EXIT_CHECK_FAILED
    report = json.loads((run / "result" / "final.step.json").read_text())
    assert {"topology", "before_write", "valid_solid", "closed"} == set(report["checks_failed"])
    assert "FAILED closed" in out and "FAILED valid_solid" in out and "NOT closed" in out
    assert (run / "result" / "final.step").is_file() and report["input"]["closed"] is False and report["input"]["open_edges"] == 3
    assert any("written as it is" in line for line in report["options"])


# --- the hook: a real mesh --------------------------------------------------------------------------------------------------------


@needs_ocp
def test_the_part_is_a_valid_closed_solid(hook_runs):
    for r in hook_runs:
        assert r["checks_failed"] == [], r["options"]
        assert r["model"]["solids"] == 1 and r["model"]["free_edges"] == 0 and r["model"]["planar_faces"] == r["model"]["faces"]
        assert r["input"]["triangles"] == len(part_mesh().faces) == 4712 and r["simplify"]["applied"] is True
        assert r["simplify"]["triangles_after"] < r["input"]["triangles"] / 4
        assert r["volume"]["roundtrip_error"] < 1e-9


@needs_ocp
def test_the_part_volume_and_distance_are_within_the_measured_deviation(hook_runs):
    for r in hook_runs:
        tol = r["settings"]["tolerance_mm"]
        d = r["deviation"]
        # Measured on meshes of runs (the hook) and on `part_mesh`: the largest distance is 1.4 to 2 times the simplification tolerance.
        assert d["step_to_stl"]["max_mm"] <= 2.5 * tol and d["stl_to_step"]["max_mm"] <= 2.5 * tol
        assert d["step_to_stl"]["mean_mm"] < 0.5 * tol and d["stl_to_step"]["mean_mm"] < 0.5 * tol
        assert abs(r["volume"]["step_vs_stl_pct"]) < 2.0       # measured: -1.03 percent on the hook at 0.12 mm, -0.26 at 0.06 mm; -0.53 on `part_mesh`


@needs_ocp
def test_the_part_has_fewer_faces_with_merged_planes(hook_runs):
    merged, plain = hook_runs
    assert merged["merge_planes"]["applied"] and not plain["merge_planes"]["applied"]
    assert merged["model"]["faces"] < plain["model"]["faces"] == plain["simplify"]["triangles_after"]
    assert merged["merge_planes"]["faces_after"] == merged["model"]["faces"]
    assert merged["output"]["bytes"] < plain["output"]["bytes"]


# --- the command ---------------------------------------------------------------------------------------------------------------


@needs_ocp
def test_the_command_prints_the_text_and_exit_code_0(tmp_path, capsys):
    run = make_run(tmp_path, run_json={"name": "t", "h": 2.0})
    code = main(["step", str(run)])
    cap = capsys.readouterr()
    assert code == 0
    assert "result/final.step" in cap.out and "checks:" in cap.out and "distance to the STL" in cap.out and "report: result/final.step.json" in cap.out
    assert "topopt step: reading" in cap.err      # the progress goes to stderr
    assert (run / "result" / "final.step").is_file()


@needs_ocp
def test_the_json_option_prints_the_report(tmp_path, capsys):
    run = make_run(tmp_path)
    assert main(["step", str(run), "--json", "--no-merge-planes", "--tolerance-mm", "0"]) == 0
    cap = capsys.readouterr()
    report = json.loads(cap.out)
    assert report["model"]["faces"] == 12 and report["settings"]["merge_planes"] is False
    assert cap.err == ""
    assert report == {**json.loads((run / "result" / "final.step.json").read_text()), "report_path": report["report_path"]}


@needs_ocp
def test_the_command_estimate_only(tmp_path, capsys):
    run = make_run(tmp_path)
    assert main(["step", str(run), "--estimate-only"]) == 0
    out = capsys.readouterr().out
    assert "estimate only, nothing is written" in out and "12 triangles" in out
    assert not (run / "result" / "final.step").exists() and not (run / "result" / "final.step.json").exists()


def test_a_usage_error_and_an_input_error_are_exit_code_1(tmp_path, capsys):
    assert main(["step", str(make_run(tmp_path)), "--tolerance-mm", "-1"]) == 1
    assert "tolerance_mm" in capsys.readouterr().err
    assert main(["step", str(make_run(tmp_path, name="s")), "--schema", "AP999", "--estimate-only"]) == 1
    assert "AP214 or AP242" in capsys.readouterr().err
    assert main(["step", str(tmp_path / "missing")]) == 1
    assert "not a folder" in capsys.readouterr().err


# --- review round 1: the target of --out, write problems, an empty result ---------------------------------------------------------


@needs_ocp
@pytest.mark.parametrize("where, text", [
    ("result/final.stl", "the STL that this command reads"),
    ("run.json", "existing file that is not a .step or .stp file"),
    ("status.json", "existing file that is not a .step or .stp file"),
    ("result/final.step.json", "existing file that is not a .step or .stp file"),      # the report of an earlier export
], ids=["input_stl", "run_json", "status_json", "report_of_an_earlier_export"])
def test_out_does_not_replace_the_input_stl_or_a_file_that_is_not_a_step_file(tmp_path, where, text):
    run = make_run(tmp_path, run_json={"name": "x", "h": 2.0})
    step.export_step(run, tolerance_mm=0)
    before = tree(run)
    with pytest.raises(UserError, match=text) as e:
        step.export_step(run, tolerance_mm=0, out=run / where)
    assert "Give a new file name that ends in .step, or a folder" in str(e.value)      # the options
    assert tree(run) == before


@needs_ocp
def test_the_command_says_it_in_one_message_and_exit_code_1_for_a_target_that_is_not_a_step_file(tmp_path, capsys):
    run = make_run(tmp_path)
    before = tree(run)
    assert main(["step", str(run), "--out", str(run / "result" / "final.stl")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err and "final.stl" in err and tree(run) == before


@needs_ocp
def test_out_replaces_a_step_file_in_either_suffix_and_makes_a_new_file_of_any_name(tmp_path):
    run = make_run(tmp_path)
    for name in ("old.step", "old.stp", "OLD.STEP"):
        (tmp_path / name).write_text("an earlier STEP file")
        r = step.export_step(run, tolerance_mm=0, out=tmp_path / name)
        assert Path(r["output"]["path"]).read_text().startswith("ISO-10303-21;") and r["checks_failed"] == []
        assert Path(r["report_path"]) == tmp_path / (name + ".json")
    assert step.export_step(run, tolerance_mm=0, out=tmp_path / "new.model")["output"]["path"] == str(tmp_path / "new.model")


def test_the_report_has_the_name_of_the_step_file_plus_json():
    assert step.report_path_of(Path("/r/result/final.step")) == Path("/r/result/final.step.json")
    assert step.report_path_of(Path("/x/hook.stp")) == Path("/x/hook.stp.json")


@needs_ocp
def test_a_folder_without_write_access_is_a_user_error_before_the_work(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("this user can write to a folder with mode 500 (root)")
        monkeypatch.setattr("topopt.printcheck.load_stl_float32", lambda *a, **k: pytest.fail("the work started"))
        with pytest.raises(UserError, match="no write access to the folder") as e:
            step.export_step(run, tolerance_mm=0, out=locked / "x.step")
        assert str(locked) in str(e.value) and "x.step.json" in str(e.value)
        assert main(["step", str(run), "--out", str(locked / "x.step")]) == 1
        err = capsys.readouterr().err
        assert "Traceback" not in err and str(locked) in err
    finally:
        locked.chmod(0o700)


@needs_ocp
def test_a_run_folder_without_write_access_is_fine_when_out_is_elsewhere(tmp_path):
    run = make_run(tmp_path)
    before = tree(run)
    folders = [run, run / "result"]
    for f in folders:
        f.chmod(0o500)
    try:
        r = step.export_step(run, tolerance_mm=0, out=tmp_path / "elsewhere.step")
        assert r["checks_failed"] == [] and Path(r["report_path"]) == tmp_path / "elsewhere.step.json"
    finally:
        for f in folders:
            f.chmod(0o700)
    assert tree(run) == before


@needs_ocp
def test_a_write_error_of_the_step_file_and_of_the_report_are_user_errors_with_the_path(tmp_path, monkeypatch):
    run = make_run(tmp_path)

    def cannot_write(*a, **k):
        raise RuntimeError("OpenCASCADE could not write the file")
    with monkeypatch.context() as m:
        m.setattr(step, "write_step", cannot_write)
        with pytest.raises(UserError, match=r"STEP file could not be written to .*final\.step: OpenCASCADE could not write") as e:
            step.export_step(run, tolerance_mm=0)
        assert "Traceback" not in str(e.value)

    def denied(*a, **k):
        raise PermissionError(13, "Permission denied")
    with monkeypatch.context() as m:
        m.setattr("topopt.runinfo.write_atomic", denied)
        with pytest.raises(UserError, match=r"STEP file is written \(.*final\.step\), but the report could not be written to .*final\.step\.json"):
            step.export_step(run, tolerance_mm=0)


@needs_ocp
def test_a_tolerance_that_removes_every_triangle_gives_an_error_with_the_options_and_writes_nothing(tmp_path, capsys):
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.02)      # a 40 mm sphere written in metres
    run = make_run(tmp_path, sphere, run_json={"name": "metres", "h": 1.0})      # the default tolerance is 0.05 mm
    before = tree(run)
    with pytest.raises(UserError) as e:
        step.export_step(run)
    text = str(e.value)
    assert "removed every triangle" in text and "not in mm" in text and "--tolerance-mm" in text and "No file is written" in text
    assert main(["step", str(run)]) == 1 and "removed every triangle" in capsys.readouterr().err
    with pytest.raises(UserError, match="removed every triangle"):
        step.export_step(run, estimate_only=True)
    assert tree(run) == before
    assert step.export_step(run, tolerance_mm=0)["build"]["bodies"] == 1      # the option that the text names works


# --- review round 1: exact checks, orientation, units, names -------------------------------------------------------------------


@needs_ocp
@pytest.mark.parametrize("mesh, volume, expect", [
    (hollow_inverted, 26000.0, {"inside_out_bodies": 1, "inside_out_voids": 1, "bodies": 1, "voids": 1}),
    (one_body_inverted, 7000.0, {"inside_out_bodies": 1, "inside_out_voids": 0, "bodies": 2, "voids": 0}),
], ids=["hollow_inverted", "one_body_inverted"])
def test_a_mesh_with_mixed_orientation_gives_the_right_volumes_and_solids(tmp_path, mesh, volume, expect):
    r = step.export_step(make_run(tmp_path, mesh()), tolerance_mm=0)
    assert r["checks_failed"] == [], r["options"]
    assert {k: r["build"][k] for k in expect} == expect
    assert r["volume"]["stl_mm3"] == pytest.approx(volume, rel=1e-12) and r["volume"]["wrapped_mm3"] == pytest.approx(volume, rel=1e-12)
    assert r["volume"]["step_mm3"] == pytest.approx(volume, rel=1e-12) and abs(r["volume"]["step_vs_stl_pct"]) < 1e-10
    assert r["model"]["solids"] == expect["bodies"] and r["model"]["shells"] == expect["bodies"] + expect["voids"]
    assert r["build"]["in_memory"]["smallest_solid_mm3"] > 0


@needs_ocp
def test_a_cavity_that_is_written_outward_is_turned_inward(tmp_path):
    # the outer box is right side out and the cavity too (its faces point out of the empty space the wrong way round for a cavity)
    a, b = trimesh.creation.box((30, 30, 30)), trimesh.creation.box((10, 10, 10))
    r = step.export_step(make_run(tmp_path, trimesh.util.concatenate([a, b])), tolerance_mm=0)
    assert r["build"]["voids"] == 1 and r["build"]["inside_out_voids"] == 1 and r["checks_failed"] == []
    assert r["volume"]["step_mm3"] == pytest.approx(26000.0, rel=1e-12) and r["model"]["shells"] == 2


@needs_ocp
def test_an_inside_out_body_is_turned_and_the_model_in_memory_has_a_positive_volume(tmp_path):
    r = step.export_step(make_run(tmp_path, inside_out_box()), tolerance_mm=0)
    assert r["build"]["inside_out_bodies"] == 1 and r["checks_failed"] == []
    assert r["build"]["in_memory"]["smallest_solid_mm3"] == pytest.approx(6000.0, rel=1e-12)
    assert "(positive)" in r["checks"]["before_write"]["reason"]


class ShimStatic:
    """Stands in for `Interface_Static`: `values` gives another value for a setting, `reject` makes a setting return False."""

    def __init__(self, real, values=None, reject=()):
        self._real, self._values, self._reject = real, values or {}, set(reject)

    def SetCVal_s(self, name, value):
        if name in self._reject:
            return False
        return self._real.SetCVal_s(name, self._values.get(name, value))

    def __getattr__(self, name):
        return getattr(self._real, name)


def with_static(monkeypatch, **kwargs):
    from types import SimpleNamespace
    real = step._import_ocp()
    ns = SimpleNamespace(**{**vars(real), "Interface_Static": ShimStatic(real.Interface_Static, **kwargs)})
    monkeypatch.setattr(step, "_import_ocp", lambda: ns)


@needs_ocp
@pytest.mark.parametrize("setting, name", [("INCH", "INCH"), ("FT", "FOOT"), ("M", "m")])
def test_a_file_in_another_unit_fails_the_units_check_although_the_reader_gives_the_same_volume(tmp_path, monkeypatch, setting, name):
    with_static(monkeypatch, values={"write.step.unit": setting})
    r = step.export_step(make_run(tmp_path), tolerance_mm=0)
    assert r["output"]["unit"] == name
    assert r["checks_failed"] == ["units"] and name in r["checks"]["units"]["reason"]
    assert r["volume"]["step_mm3"] == pytest.approx(6000.0, rel=1e-9)      # the reader converts to mm: no other check sees the unit
    assert r["checks"]["topology"]["state"] == r["checks"]["valid_solid"]["state"] == "pass"


@needs_ocp
def test_a_setting_that_opencascade_does_not_take_is_a_note_of_the_report(tmp_path, monkeypatch):
    with_static(monkeypatch, reject=["write.step.product.name"])
    r = step.export_step(make_run(tmp_path, run_json={"name": "my_hook", "h": 2.0}), tolerance_mm=0)
    assert any("did not take write.step.product.name = 'my_hook'" in n for n in r["output"]["notes"])
    assert not r["output"]["product_name"].startswith("my_hook")      # the report names the product name that the file holds
    assert "note: OpenCASCADE did not take" in step.format_text(r)


@needs_ocp
def test_the_product_name_is_the_one_in_the_file_and_a_name_outside_ascii_is_a_note(tmp_path):
    name = "крюк 'q' \\ x"
    r = step.export_step(make_run(tmp_path, run_json={"name": name, "h": 2.0}), tolerance_mm=0)
    assert re.fullmatch(re.escape(name) + r" \d+", r["output"]["product_name"]) and r["checks_failed"] == []
    assert any("outside ASCII" in n for n in r["output"]["notes"])
    ascii_name = step.export_step(make_run(tmp_path, name="a", run_json={"name": "plain", "h": 2.0}), tolerance_mm=0)
    assert ascii_name["output"]["notes"] == []


@needs_ocp
def test_the_reason_of_closed_names_the_triangles_that_were_left_out(tmp_path):
    r = step.export_step(make_run(tmp_path, zero_area_box()), tolerance_mm=0)
    assert r["input"]["closed"] is True and r["input"]["zero_area_faces"] == 1 and r["build"]["degenerate_triangles_left_out"] == 1
    assert "closed" in r["checks_failed"]
    assert "1 triangle(s) with no area were left out of the model, and the free edges can come from them" in r["checks"]["closed"]["reason"]
    clean = step.export_step(make_run(tmp_path, name="clean"), tolerance_mm=0)
    assert "left out" not in clean["checks"]["closed"]["reason"]


# --- review round 1: tests that fail when a check is taken out (the surviving mutants of the review) -------------------------------


def write_first_solid_only(monkeypatch):
    """The file holds the first solid of the model only: a body is lost on the way to the file."""
    original = step.write_step

    def write_first(ocp, shape, path, schema, product):
        solids = ocp.TopTools_IndexedMapOfShape()
        ocp.TopExp.MapShapes_s(shape, ocp.TopAbs_SOLID, solids)
        return original(ocp, solids.FindKey(1), path, schema, product)
    monkeypatch.setattr(step, "write_step", write_first)


@needs_ocp
def test_a_body_lost_on_the_way_to_the_file_fails_topology_and_the_distance_sees_it_in_the_other_direction(tmp_path, monkeypatch):
    write_first_solid_only(monkeypatch)
    r = step.export_step(make_run(tmp_path, two_boxes()), tolerance_mm=0)
    # the read-back is checked on the file: one solid in the file, two built (a check on the model in memory would not see it)
    assert r["checks_failed"] == ["topology"] and r["model"]["solids"] == 1 and r["build"]["bodies"] == 2
    assert "1 solid(s)" in r["checks"]["topology"]["reason"] and "2 solid(s)" in r["checks"]["topology"]["reason"]
    d = r["deviation"]
    assert d["step_to_stl"]["max_mm"] < 1e-6      # every point of the file is on the STL
    assert d["stl_to_step"]["max_mm"] > 30.0      # the lost box is more than 30 mm from the box that stayed (a one-way distance misses this)


@needs_ocp
def test_a_void_lost_on_the_way_to_the_file_fails_topology(tmp_path, monkeypatch):
    original = step.write_step

    def write_outer_shell_only(ocp, shape, path, schema, product):
        shells = ocp.TopTools_IndexedMapOfShape()
        ocp.TopExp.MapShapes_s(shape, ocp.TopAbs_SHELL, shells)
        solid = ocp.TopoDS_Solid()
        builder = ocp.BRep_Builder()
        builder.MakeSolid(solid)
        builder.Add(solid, shells.FindKey(1))
        return original(ocp, solid, path, schema, product)
    monkeypatch.setattr(step, "write_step", write_outer_shell_only)
    r = step.export_step(make_run(tmp_path, hollow_box()), tolerance_mm=0)
    assert r["checks_failed"] == ["topology"] and r["model"]["shells"] == 1 and r["build"]["voids"] == 1
    assert r["volume"]["step_mm3"] == pytest.approx(27000.0, rel=1e-12)      # the file is a full box


@needs_ocp
def test_the_analyzer_on_the_file_is_what_valid_solid_reports(tmp_path, monkeypatch):
    import numpy as np
    mesh = turned_triangle_box()
    ocp = step._import_ocp()
    broken, _ = step.build_brep(ocp, np.asarray(mesh.vertices, float), np.asarray(mesh.faces))      # a closed shell the analyzer flags
    monkeypatch.setattr(step, "read_back", lambda o, path: (broken, {"read": True, "unit": "mm"}))
    r = step.export_step(make_run(tmp_path), tolerance_mm=0)
    assert r["checks_failed"] == ["valid_solid"] and r["model"]["solids"] == 1 and r["model"]["free_edges"] == 0
    assert "not valid" in r["checks"]["valid_solid"]["reason"] and r["invalid_parts"] is not None
    assert r["checks"]["before_write"]["state"] == "pass"      # the model in memory was fine: the flag comes from the read-back


@needs_ocp
def test_part_only_has_its_own_report_next_to_its_file(tmp_path):
    run = make_run(tmp_path, part_only=trimesh.creation.box((5, 5, 5)))
    a = step.export_step(run, tolerance_mm=0)
    b = step.export_step(run, tolerance_mm=0, part_only=True)
    assert Path(a["report_path"]) == run / "result" / "final.step.json" and Path(b["report_path"]) == run / "result" / "part_only.step.json"
    assert json.loads(Path(a["report_path"]).read_text())["mesh"] == "final.stl"
    assert json.loads(Path(b["report_path"]).read_text())["mesh"] == "part_only.stl"


@needs_ocp
def test_the_schema_name_is_the_same_in_the_command_and_the_api(tmp_path, capsys):
    run = make_run(tmp_path)
    assert main(["step", str(run), "--schema", "ap-242", "--tolerance-mm", "0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["output"]["schema"] == "AP242"
    assert step.export_step(run, tolerance_mm=0, schema="Ap242")["output"]["schema"] == "AP242"
    assert step.export_step(run, tolerance_mm=0, schema="ap214")["output"]["schema"] == "AP214"


@needs_ocp
def test_a_unit_that_opencascade_does_not_know_is_a_note_and_the_units_check_sees_the_old_value(tmp_path, monkeypatch):
    step._import_ocp().Interface_Static.SetCVal_s("write.step.unit", "MM")      # the parameters belong to the process: start from mm
    with_static(monkeypatch, values={"write.step.unit": "IN"})      # "IN" is not a unit name of OpenCASCADE ("INCH" is): the call returns False
    r = step.export_step(make_run(tmp_path), tolerance_mm=0)
    assert any("did not take write.step.unit = 'MM'" in n for n in r["output"]["notes"])
    assert r["checks_failed"] == [] and r["output"]["unit"] == "mm"


@needs_ocp
def test_a_solid_with_its_faces_turned_inward_in_memory_fails_before_write_though_the_analyzer_says_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(step, "_merged", lambda ocp, shape: shape.Reversed())      # the model in memory is inside out
    r = step.export_step(make_run(tmp_path), tolerance_mm=0)
    assert r["build"]["in_memory"]["valid"] is True      # the analyzer does not see an inside-out solid
    assert r["build"]["in_memory"]["smallest_solid_mm3"] == pytest.approx(-6000.0, rel=1e-12)
    assert r["checks"]["before_write"]["state"] == "fail" and "not positive" in r["checks"]["before_write"]["reason"]
