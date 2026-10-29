"""Tests of `topopt verify --part-only` (P4b-3, CONTRACTS.md sections 9.1, 16 and 17): the check of `result/part_only.stl`.

The case is a bar (design box x 0..20, y 0..10, z 0..10) with four bodies in `regions.not_printed`: a fixed plate on the left whose
face lies at x = 0.37 (off the voxel faces), a load box at the tip (a notch in the corner), a solid box (a notch in the top face) and a
keep-out box (a hole through the bar). `topopt run` makes the run (run.json, the record of the cut, the YAML). Then the fixture replaces
`final.stl` and `part_only.stl` by exact boxes (the same cut, `topopt.part_only.cut`, of boxes). The reason: the part_only.stl of an
optimizer run keeps a thin film and slivers around the removed bodies (CONTRACTS.md section 9.1, known limits) and does not always mesh;
the boxes mesh at every size, and the contact areas are known numbers: plate 10 x 10 = 100, tip 6 + 6 + 6 + 4 = 22, boss
2 x 11.4 + 2 x 10.2 + 12.92 = 56.12, hole 2 x (2.2 + 3.4) x 10 = 112 mm2. The tests compare with numbers that they compute from the boxes
(`box_sdf`) and from the mesh that verify wrote (`mesh.npz`, `fields.npz`), not with the code that is tested.

Bounds that the tests use (test bounds, not limits of the tool; measured on this case, Linux x86_64, 2026-10-23):
- an area against its exact value: float32 of the box corners, 1e-7 relative. The test allows 1e-5.
- the resultant of the traction against the YAML force: 1e-16 relative. The test allows 1e-9.
- the equilibrium residual of the P2 solve: 1e-13 relative. The test allows 1e-8.
- a node of the contact face against the plane of the plate: below 1e-6 mm. The test allows 1e-4 mm.
- the stiffness against the Euler-Bernoulli cantilever of the full section: 0.885 at mesh size 1.5 mm and 0.87 at 2.0 mm (the notches,
  the hole and the shear lower it). The test allows 0.6 to 1.0.

The hook case (the last test) is checked for its boundary conditions only, not for a stiffness value: the film that stays around the
removed tip decides the stiffness of that part (0.5 N/mm against 1300 N/mm of the optimizer, measured), and no film rule is decided.
Its equilibrium residual is 1e-9 to 4e-9 (the soft system is ill-conditioned); the test allows 1e-6.
"""
import contextlib
import hashlib
import inspect
import io
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import needs_gmsh, needs_skfem  # noqa: E402
from test_optimize import box  # noqa: E402
from test_part_only import PLATE_X, TIMING, box_sdf, load_stl  # noqa: E402
from topopt import post  # noqa: E402
from topopt.cli import main  # noqa: E402

BOXES = {                                        # name -> (lo, hi): the region files of the bar
    "plate": ((-2, 0, 0), (PLATE_X, 10, 10)),
    "tip": ((18, 4, 0), (20, 6, 3)),
    "boss": ((6.3, 3.1, 7.0), (9.7, 6.9, 10)),
    "hole": ((12.2, 3.3, -1), (14.4, 6.7, 11)),
}
CONTACT = {"fixed:plate.stl": 100.0, "load:tip": 22.0, "solid:boss.stl": 56.12, "keep_out:hole.stl": 112.0}      # mm2, from the boxes
KEYED = ["plate.stl", "tip.stl", "boss.stl", "hole.stl"]
LABELS = list(CONTACT)
MESH = ["--mesh-size", "1.5", "--solver", "superlu"]
VOLATILE = TIMING | {"memory", "memory_limit", "timing_s", "peak_rss_bytes", "seconds", "seconds_gmsh", "seconds_surface",
                     "run_dir", "final_stl"}


# -- helpers -------------------------------------------------------------------------------------------------------
def verify(src: Path, dst: Path, *argv) -> SimpleNamespace:
    """Copy the run folder `src` (without its verify folder) to `dst`, run `topopt verify` there, return code, text and verify.json."""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("verify"))
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(dst), *map(str, argv)])
    vj = dst / "verify" / "verify.json"
    return SimpleNamespace(dir=dst, code=code, out=out.getvalue(), err=err.getvalue(),
                           res=json.loads(vj.read_text()) if vj.is_file() else None)


def scrub(obj):
    """The object without the keys that change from run to run (times, memory, the path of the run folder)."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_run_json(run_dir: Path, edit) -> None:
    p = run_dir / "run.json"
    doc = json.loads(p.read_text())
    edit(doc)
    p.write_text(json.dumps(doc))


def region(res, role: str, name: str) -> dict:
    return next(r for r in res["regions"][role] if r["region"] == name)


def removed(res, label: str) -> dict:
    return next(r for r in res["part_only"]["removed"] if r["label"] == label)


def mesh_of(run_dir: Path):
    """`(tets, vertices, p2_nodes)` of the mesh that verify wrote: the P1 tets and vertices, and the P2 nodes (vertices and edge midpoints)."""
    z = np.load(run_dir / "verify" / "mesh.npz")
    f = np.load(run_dir / "verify" / "fields.npz")
    return z["tets"], z["nodes"], f["node_xyz"].T.astype(np.float64)


def boundary_facets(tets):
    """The faces of the tets that belong to one tet: the boundary of the mesh."""
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    _, idx, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[idx[cnt == 1]]


def facet_area(points, facets) -> np.ndarray:
    a, b, c = (points[facets[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def write_bar(folder: Path, not_printed=None, name="bar.yaml") -> Path:
    """The region files of the bar in `folder` and a YAML. `not_printed` None: the key is not there."""
    folder.mkdir(parents=True, exist_ok=True)
    box(folder / "design.stl", (0, 0, 0), (20, 10, 10))
    for n, (lo, hi) in BOXES.items():
        box(folder / f"{n}.stl", lo, hi)
    regions = {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}, "solid": ["boss.stl"], "keep_out": ["hole.stl"]}
    if not_printed is not None:
        regions["not_printed"] = not_printed
    doc = {"name": "bar", "regions": regions,
           "load_cases": [{"name": "down", "weight": 3.0, "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
           "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": 8, "beta_schedule": [[0, 1], [4, 2]]}}
    p = folder / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def make_exact_files(folder: Path, run_dir: Path) -> None:
    """Replace final.stl and part_only.stl of the run by exact boxes, and the volumes in the record of the cut by their volumes."""
    import trimesh

    from topopt import part_only, smooth_sdf
    result = run_dir / "result"
    box(folder / "bar_only.stl", (0, 0, 0), (20, 10, 10))
    bar_and_plate = smooth_sdf._load_manifold(folder / "bar_only.stl") + smooth_sdf._load_manifold(folder / "plate.stl")
    v, f = smooth_sdf._from_manifold(bar_and_plate - smooth_sdf._load_manifold(folder / "hole.stl"))     # the printed bar has its hole
    trimesh.Trimesh(v.astype(np.float32).astype(np.float64), f, process=False).export(result / "final.stl")
    final = trimesh.load(result / "final.stl")
    bodies = [(label, folder / name) for label, name in zip(LABELS, KEYED)]
    v, f, facts = part_only.cut(final.vertices, final.faces, bodies, None)
    trimesh.Trimesh(v, f, process=False).export(result / "part_only.stl")
    cut = trimesh.load(result / "part_only.stl")
    assert final.is_watertight and cut.is_watertight and len(cut.split(only_watertight=False)) == 1
    edit_run_json(run_dir, lambda d: d["part_only"].update(
        volume_before_mm3=round(float(final.volume), 3), volume_after_mm3=round(float(cut.volume), 3),
        removed_mm3=round(float(final.volume - cut.volume), 3), leftovers_dropped=facts["leftovers_dropped"], bodies=1, watertight=True,
        shortest_edge_mm=part_only.shortest_edge_mm((result / "part_only.stl").read_bytes())))


# -- fixtures --------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bar(tmp_path_factory):
    """Two runs of the bar: with the four bodies in `regions.not_printed` (final.stl and part_only.stl made exact), and without the key."""
    folder = tmp_path_factory.mktemp("p4b3_bar")
    keyed_yaml = write_bar(folder, KEYED, "keyed.yaml")
    plain_yaml = write_bar(folder, None, "plain.yaml")
    for y, name in ((keyed_yaml, "keyed"), (plain_yaml, "plain")):
        assert main(["run", str(y), "--out", str(folder / name), "--quiet"]) == 0
    assert (folder / "keyed" / "result" / "part_only.stl").is_file()
    assert not (folder / "plain" / "result" / "part_only.stl").exists()
    make_exact_files(folder, folder / "keyed")
    return SimpleNamespace(folder=folder, keyed=folder / "keyed", plain=folder / "plain", keyed_yaml=keyed_yaml)


@pytest.fixture(scope="module")
def part(bar, tmp_path_factory):
    """`topopt verify --part-only` on the keyed bar (one real mesh and solve)."""
    got = verify(bar.keyed, tmp_path_factory.mktemp("p4b3_po") / "run", "--part-only", *MESH)
    assert got.code == 0, got.err
    return got


@pytest.fixture(scope="module")
def noflag(bar, tmp_path_factory):
    """`topopt verify` on the keyed bar without the option (the run has part_only.stl), and on a copy without the file."""
    base = tmp_path_factory.mktemp("p4b3_nf")
    with_file = verify(bar.keyed, base / "with_file", *MESH)
    shutil.copytree(bar.keyed, base / "src_no_file", ignore=shutil.ignore_patterns("verify"))
    (base / "src_no_file" / "result" / "part_only.stl").unlink()
    no_file = verify(base / "src_no_file", base / "no_file", *MESH)
    return SimpleNamespace(with_file=with_file, no_file=no_file)


# -- the file that is meshed, and what verify.json records ---------------------------------------------------------------
@needs_gmsh
@needs_skfem
def test_the_option_meshes_part_only_stl_and_verify_json_records_the_file_and_its_hash(part):
    res, stl = part.res, part.dir / "result" / "part_only.stl"
    assert res["ok"] and res["exit_code"] == 0
    assert res["run"]["verified_file"] == "part_only.stl" and res["run"]["final_stl_sha256"] == sha256(stl)
    assert res["mesh"]["stl_sha256"] == sha256(stl)
    assert res["stl"]["volume_mm3"] == pytest.approx(load_stl(stl).volume, rel=1e-6)
    po = res["part_only"]
    assert po["on"] is True and po["file"] == "part_only.stl" and po["sha256"] == sha256(stl)
    assert [r["label"] for r in po["removed"]] == LABELS
    assert [r["role"] for r in po["removed"]] == ["fixed", "load", "solid", "keep_out"]
    assert po["run_record"]["applied"] is True and po["run_record"]["removed"] == LABELS      # the record of the cut, from run.json
    assert po["yaml_vs_run"]["matches"] is True
    assert "mesh file: part_only.stl" in part.out


@needs_gmsh
@needs_skfem
def test_the_summary_has_one_line_per_removed_body(part):
    lines = [x for x in part.out.splitlines() if x.startswith("removed body ")]
    assert len(lines) == 4
    assert lines[0].startswith("removed body fixed:plate.stl: clamp; ")
    assert lines[1].startswith("removed body load:tip: traction; ")
    assert lines[2].startswith("removed body solid:boss.stl: no boundary condition; ")
    assert lines[3].startswith("removed body keep_out:hole.stl: no boundary condition; ")
    for x in lines:
        assert "at the bands 0, 0.5x and 1x" in x and "contact area of the cut" in x and "selected over contact" in x


# -- removed fixed body: the clamp ---------------------------------------------------------------------------------------
@needs_gmsh
@needs_skfem
def test_the_clamp_is_on_the_nodes_that_the_region_rule_selects_at_the_contact_face(part):
    res = part.res
    tets, verts, nodes = mesh_of(part.dir)
    band = res["band_mm"]
    lo, hi = BOXES["plate"]
    rule_nodes = box_sdf(nodes, lo, hi) <= band                  # the rule: signed distance to the region >= -band (box_sdf: negative inside)
    rule_verts = box_sdf(verts, lo, hi) <= band
    fact = removed(res, "fixed:plate.stl")
    assert fact["boundary_condition"] == "clamp"
    assert fact["nodes"] == int(rule_nodes.sum()) == res["regions"]["fixed_nodes_total"]      # no node is clamped for another body
    assert fact["vertices"] == int(rule_verts.sum()) and fact["vertices"] > 0
    assert region(res, "fixed", "plate")["nodes"] == fact["nodes"]
    # the contact face: the nodes on the plane of the plate (x = float32(0.37), the face that the cut made) are all clamped
    on_plane = np.abs(nodes[:, 0] - np.float32(PLATE_X)) <= 1e-4
    assert on_plane.sum() > 0 and rule_nodes[on_plane].all()
    # and the rule takes nothing that is farther than the band from the body
    assert nodes[rule_nodes, 0].max() <= PLATE_X + band + 1e-6
    assert (nodes[~rule_nodes, 0] > PLATE_X + band - 1e-6).all()
    assert verts[rule_verts, 0].max() <= PLATE_X + band + 1e-6
    # the facets, their area and the counts at three bands
    facets = boundary_facets(tets)
    sel = rule_verts[facets].all(axis=1)
    assert fact["facets"] == int(sel.sum())
    assert fact["area_mm2"] == pytest.approx(float(facet_area(verts, facets[sel]).sum()), rel=1e-9)
    counts = fact["counts_by_band"]
    assert list(counts) == ["0", "0.5x", "1x"]
    assert counts["0"]["nodes"] <= counts["0.5x"]["nodes"] <= counts["1x"]["nodes"] == fact["nodes"]
    assert counts["0.5x"]["nodes"] == int((box_sdf(nodes, lo, hi) <= 0.5 * band).sum())
    on_or_in = box_sdf(nodes, np.float32(lo).astype(float), np.float32(hi).astype(float))       # the region file holds float32 corners
    assert int((on_or_in <= -1e-6).sum()) <= counts["0"]["nodes"] <= int((on_or_in <= 1e-6).sum())     # band 0: in the body or on its face
    assert counts["0"]["facets"] <= counts["0.5x"]["facets"] <= counts["1x"]["facets"] == fact["facets"]
    # the contact face (the plane of the plate, 10 x 10 mm) is in the selection, and the comparison with the contact area
    plane_facets = (np.abs(verts[facets][:, :, 0] - np.float32(PLATE_X)) <= 1e-4).all(axis=1)
    assert plane_facets.sum() > 0 and sel[plane_facets].all()
    assert float(facet_area(verts, facets[plane_facets]).sum()) == pytest.approx(CONTACT["fixed:plate.stl"], rel=1e-5)
    assert fact["contact"]["area_mm2"] == pytest.approx(CONTACT["fixed:plate.stl"], rel=1e-5)
    assert fact["area_mm2"] >= fact["contact"]["area_mm2"] * (1 - 1e-5)
    assert fact["contact"]["selected_area_over_contact"] == pytest.approx(fact["area_mm2"] / fact["contact"]["area_mm2"], rel=1e-9)


# -- removed load body: the traction and the resultant -----------------------------------------------------------------------
@needs_gmsh
@needs_skfem
def test_the_load_is_a_traction_on_the_facets_that_the_rule_selects_and_the_resultant_is_the_yaml_force(part):
    res = part.res
    tets, verts, _ = mesh_of(part.dir)
    lo, hi = BOXES["tip"]
    facets = boundary_facets(tets)
    sel = (box_sdf(verts, lo, hi) <= res["band_mm"])[facets].all(axis=1)            # all 3 vertices of a facet within the band
    area = float(facet_area(verts, facets[sel]).sum())
    fact = removed(res, "load:tip")
    assert fact["boundary_condition"] == "traction"
    assert fact["facets"] == int(sel.sum()) > 0
    assert fact["area_mm2"] == pytest.approx(area, rel=1e-9)
    assert region(res, "load", "tip")["facets"] == fact["facets"]
    assert fact["contact"]["area_mm2"] == pytest.approx(CONTACT["load:tip"], rel=1e-5)
    assert fact["area_mm2"] >= fact["contact"]["area_mm2"] * (1 - 1e-5)                          # the notch faces are in the selection
    force = res["cases"]["down"]["forces"][0]
    assert force["area_mm2"] == pytest.approx(area, rel=1e-9)
    assert force["traction_MPa"] == pytest.approx([0.0, 0.0, -2.0 / area], rel=1e-9)            # the YAML force over the area
    assert force["resultant_N"] == pytest.approx([0.0, 0.0, -2.0], abs=1e-9)                     # the check of the resultant stays
    assert force["resultant_relative_error"] < 1e-9


@needs_gmsh
@needs_skfem
def test_the_equilibrium_of_the_part_only_solve_passes(part):
    case = part.res["cases"]["down"]
    eq = case["equilibrium"]
    assert eq["force_relative_residual"] < 1e-8 and eq["moment_relative_residual"] < 1e-8
    assert eq["applied_force_N"] == pytest.approx([0.0, 0.0, -2.0], abs=1e-9)
    assert eq["reaction_force_N"] == pytest.approx([0.0, 0.0, 2.0], abs=1e-8)                  # the clamp takes the whole load
    assert case["linear_solve"]["relative_residual"] < 1e-8
    assert case["stiffness_N_per_mm"] > 0 and case["max_displacement_mm"] > 0
    assert case["stiffness_N_per_mm"] == pytest.approx(4.0 / case["compliance_Nmm"], rel=1e-9)         # |F|^2 over the compliance, |F| = 2 N
    # the physical result: a bar clamped at x = 0.37 and loaded at its end is as stiff as the cantilever of its section, a little less
    # (the notches, the hole and the shear lower it)
    cantilever = 3.0 * part.res["material"]["E_MPa"] * (10.0 * 10.0 ** 3 / 12.0) / (20.0 - PLATE_X) ** 3
    assert 0.6 < case["stiffness_N_per_mm"] / cantilever < 1.0


# -- removed solid or keep-out body: a fact and no condition ---------------------------------------------------------------------
@needs_gmsh
@needs_skfem
def test_a_removed_solid_or_keep_out_body_gives_the_fact_and_no_boundary_condition(part):
    res = part.res
    for label in ("solid:boss.stl", "keep_out:hole.stl"):
        fact = removed(res, label)
        assert fact["boundary_condition"] == "none" and "no boundary condition" in fact["note"]
        assert fact["nodes"] > 0 and fact["facets"] > 0                                         # the facts are measured
        assert fact["contact"]["area_mm2"] == pytest.approx(CONTACT[label], rel=1e-5)
        assert any(x.startswith(f"removed body {label}: no boundary condition") for x in part.out.splitlines())
    assert res["regions"]["fixed_nodes_total"] == removed(res, "fixed:plate.stl")["nodes"]     # the two bodies clamped nothing
    assert [r["region"] for r in res["regions"]["fixed"]] == ["plate"]
    assert [r["region"] for r in res["regions"]["load"]] == ["tip"]
    note = res["part_only"]["singular_note"]
    assert "solid:boss.stl, keep_out:hole.stl" in note and "singular" in note and "This solve did not stop" in note
    assert any(x.startswith("note: solid:boss.stl, keep_out:hole.stl: no boundary condition") for x in part.out.splitlines())


@needs_gmsh
@needs_skfem
def test_the_contact_areas_are_the_known_areas_of_the_faces_where_the_bodies_touched_the_part(part):
    for label, area in CONTACT.items():
        assert removed(part.res, label)["contact"]["area_mm2"] == pytest.approx(area, rel=1e-5)


@needs_gmsh
@needs_skfem
def test_the_time_of_the_contact_areas_is_a_timing_key_only_with_the_option(part, noflag):
    t = part.res["timing_s"]                                                           # fix round 1, n7: the booleans run after regions_s is set
    assert isinstance(t["part_only_contact_s"], float) and 0.0 <= t["part_only_contact_s"] <= t["verify_total_s"]
    assert "part_only_contact_s" not in noflag.with_file.res["timing_s"]               # no option: no such key (verify.json is as before)


@needs_gmsh
@needs_skfem
def test_with_several_levels_the_contact_areas_are_measured_once_and_the_reference_level_has_the_time(bar, tmp_path, monkeypatch):
    from topopt.verify import part_only as vpo
    calls = []
    real = vpo.contact_areas
    monkeypatch.setattr(vpo, "contact_areas", lambda *a, **k: calls.append(1) or real(*a, **k))
    got = verify(bar.keyed, tmp_path / "run", "--part-only", "--levels", "2.0,1.5", "--solver", "superlu")
    assert got.code == 0, got.err
    assert len(calls) == 1                                                             # the cache serves the second level
    assert got.res["timing_s"]["part_only_contact_s"] >= 0.0                           # the top-level timing_s is the one of the finest level
    assert len(got.res["levels"]["rows"]) == 2


# -- without the option ---------------------------------------------------------------------------------------------------------
@needs_gmsh
@needs_skfem
def test_without_the_option_the_result_is_the_one_of_a_run_without_the_file_and_one_fact_line_names_the_option(noflag):
    got, bare = noflag.with_file, noflag.no_file
    assert got.code == 0 and bare.code == 0
    assert got.res["run"]["verified_file"] == "final.stl"
    assert "part_only" not in got.res and "notes" not in got.res                      # verify.json is as it is today
    assert json.dumps(scrub(got.res), sort_keys=True).replace(str(got.dir), "RUN") == \
        json.dumps(scrub(bare.res), sort_keys=True).replace(str(bare.dir), "RUN")
    # the folder of the test is out of the text: a line such as "wrote <folder>/verify/verify.json" holds part_only when the test folder is named so
    got_out, bare_out = got.out.replace(str(got.dir), "RUN"), bare.out.replace(str(bare.dir), "RUN")
    lines = [x for x in got_out.splitlines() if "--part-only" in x]
    assert len(lines) == 1 and lines[0].startswith("note: this run has result/part_only.stl")
    assert "Verify meshes final.stl. Set --part-only to mesh part_only.stl." in lines[0]       # fix round 1, n2: it names the file
    assert not [x for x in bare_out.splitlines() if "part_only" in x]                   # a run without the file: no such line
    assert "removed body" not in got_out and "mesh file: part_only.stl" not in got_out     # no line for the removed bodies


# -- the option on a run without the file ---------------------------------------------------------------------------------------
def test_the_option_on_a_run_without_the_file_is_a_user_error_with_the_fact_and_the_options(bar, tmp_path):
    got = verify(bar.plain, tmp_path / "run", "--part-only", "--estimate-only")
    assert got.code == 1 and got.res is None and not (got.dir / "verify").exists()
    assert "no result/part_only.stl" in got.err and "run.json has no part_only block" in got.err
    assert "regions.not_printed" in got.err and "without --part-only" in got.err and "CONTRACTS.md section 9.1" in got.err


def test_a_deleted_file_and_a_cut_that_failed_have_their_own_fact(bar, tmp_path):
    shutil.copytree(bar.keyed, tmp_path / "deleted", ignore=shutil.ignore_patterns("verify"))
    (tmp_path / "deleted" / "result" / "part_only.stl").unlink()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(tmp_path / "deleted"), "--part-only", "--estimate-only"])
    assert code == 1 and "deleted or moved" in err.getvalue() and "without --part-only" in err.getvalue()
    shutil.copytree(bar.keyed, tmp_path / "failed", ignore=shutil.ignore_patterns("verify"))
    edit_run_json(tmp_path / "failed", lambda d: d["part_only"].update(applied=False, error="the cut left nothing"))
    (tmp_path / "failed" / "result" / "part_only.stl").unlink()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(tmp_path / "failed"), "--part-only", "--estimate-only"])
    assert code == 1 and "the post step did not make it (the cut left nothing)" in err.getvalue()


# -- the estimate, and the stock --------------------------------------------------------------------------------------------------
def test_the_estimate_with_the_option_uses_part_only_stl_and_carries_the_block(bar, tmp_path):
    from topopt.verify import part_only as vpart_only
    from topopt.verify import verify_run
    shutil.copytree(bar.keyed, tmp_path / "run", ignore=shutil.ignore_patterns("verify"))
    stl = tmp_path / "run" / "result" / "part_only.stl"
    res = verify_run(tmp_path / "run", estimate_only=True, part_only=True, mesh_size=1.5, solver="superlu")
    assert res["ok"] and res["estimate_only"] and not (tmp_path / "run" / "verify").exists()
    po = res["part_only"]
    assert po["on"] is True and po["file"] == "part_only.stl" and po["sha256"] == sha256(stl)
    assert [r["label"] for r in po["removed"]] == LABELS
    assert [(r["role"], r["name"]) for r in po["removed"]] == [("fixed", "plate.stl"), ("load", "tip"), ("solid", "boss.stl"), ("keep_out", "hole.stl")]
    assert res["run"]["verified_file"] == "part_only.stl" and res["run"]["final_stl_sha256"] == sha256(stl)
    assert res["stl"]["volume_mm3"] == pytest.approx(load_stl(stl).volume, rel=1e-6)
    assert "notes" not in res                                                        # the option is set: no line that names it
    plain = verify_run(tmp_path / "run", estimate_only=True, mesh_size=1.5, solver="superlu")
    assert "part_only" not in plain and plain["run"]["verified_file"] == "final.stl"
    assert plain["notes"] == [vpart_only.available_note("final.stl", True)] and "--part-only" in plain["notes"][0]
    assert "Verify would mesh final.stl." in plain["notes"][0] and "meshed" not in plain["notes"][0]      # an estimate meshes nothing (fix round 1, n2)
    assert plain["stl"]["volume_mm3"] > res["stl"]["volume_mm3"]                      # final.stl holds the bodies, part_only.stl does not


def test_machining_stock_is_a_fact_and_verify_makes_no_second_cut(bar, tmp_path, capsys):
    from topopt.verify import part_only as vpart_only
    from topopt.verify import verify_run
    shutil.copytree(bar.keyed, tmp_path / "run", ignore=shutil.ignore_patterns("verify"))
    r = tmp_path / "run" / "result"
    shutil.copy(r / "final.stl", r / "nominal.stl")                                   # a run with stock has nominal.stl
    res = verify_run(tmp_path / "run", estimate_only=True, part_only=True, mesh_size=1.5, solver="superlu")
    assert res["part_only"]["notes"] == [vpart_only.STOCK_NOTE]
    assert "keeps the stock" in vpart_only.STOCK_NOTE and "no second cut" in vpart_only.STOCK_NOTE
    assert res["run"]["verified_file"] == "part_only.stl"
    capsys.readouterr()
    assert main(["verify", str(tmp_path / "run"), "--part-only", "--estimate-only"]) == 0
    assert "machining stock" in capsys.readouterr().out
    # without the option the run is verified as before: nominal.stl
    plain = verify_run(tmp_path / "run", estimate_only=True, mesh_size=1.5, solver="superlu")
    assert plain["run"]["verified_file"] == "nominal.stl" and "part_only" not in plain
    assert "Verify would mesh nominal.stl." in plain["notes"][0]                         # the note names the file that is meshed
    shutil.copytree(bar.keyed, tmp_path / "run2", ignore=shutil.ignore_patterns("verify"))
    res2 = verify_run(tmp_path / "run2", estimate_only=True, part_only=True, mesh_size=1.5, solver="superlu")
    assert "notes" not in res2["part_only"]                                             # no stock: no stock line


def test_a_difference_between_the_yaml_and_the_cut_is_a_fact_and_the_run_goes_on(bar, tmp_path):
    from topopt.verify import verify_run
    shutil.copytree(bar.keyed, tmp_path / "run", ignore=shutil.ignore_patterns("verify"))
    doc = yaml.safe_load(bar.keyed_yaml.read_text())
    doc["regions"]["not_printed"] = ["plate.stl", "tip.stl"]
    other = bar.folder / "fewer.yaml"
    other.write_text(yaml.safe_dump(doc, sort_keys=False))
    res = verify_run(tmp_path / "run", estimate_only=True, part_only=True, yaml_path=other, mesh_size=1.5, solver="superlu")
    po = res["part_only"]
    assert res["ok"] and po["yaml_vs_run"]["matches"] is False
    assert [r["label"] for r in po["removed"]] == ["fixed:plate.stl", "load:tip"]                 # the removed bodies are the ones of the YAML
    assert any("regions.not_printed" in d for d in po["yaml_vs_run"]["differences"])


# -- a part that nothing holds ------------------------------------------------------------------------------------------------------------
@needs_gmsh
@needs_skfem
def test_a_removed_fixed_body_that_selects_nothing_leaves_the_part_without_support_and_the_error_has_the_facts(bar, tmp_path):
    """The fixed plate is the only support. A YAML whose plate lies away from the part: the rule finds no node, the part can move."""
    box(bar.folder / "plate_far.stl", (-12, 0, 0), (-11, 10, 10))
    doc = yaml.safe_load(bar.keyed_yaml.read_text())
    doc["regions"]["fixed"] = ["plate_far.stl"]
    doc["regions"]["not_printed"] = ["plate_far.stl", "tip.stl", "boss.stl", "hole.stl"]
    far = bar.folder / "far.yaml"
    far.write_text(yaml.safe_dump(doc, sort_keys=False))
    got = verify(bar.keyed, tmp_path / "run", "--yaml", far, "--part-only", *MESH)
    assert got.code == 1 and got.res is None
    assert "has no fixed node" in got.err and "singular" in got.err
    assert "result/part_only.stl" in got.err and "fixed:plate_far.stl" in got.err
    assert "without --part-only" in got.err and "regions.not_printed" in got.err                     # the options
    assert "fixed:plate_far.stl selects no node within the band of 0.5 mm, so it clamps nothing" in got.err      # the fact: it clamps nothing


@needs_gmsh
@needs_skfem
def test_a_removed_load_body_that_selects_no_facet_is_an_error_with_the_facts_and_the_options(bar, tmp_path):
    box(bar.folder / "tip_far.stl", (30, 4, 0), (32, 6, 3))
    doc = yaml.safe_load(bar.keyed_yaml.read_text())
    doc["regions"]["load"] = {"tip": "tip_far.stl"}
    doc["regions"]["not_printed"] = ["plate.stl", "tip_far.stl", "boss.stl", "hole.stl"]
    far = bar.folder / "far_tip.yaml"
    far.write_text(yaml.safe_dump(doc, sort_keys=False))
    got = verify(bar.keyed, tmp_path / "run", "--yaml", far, "--part-only", *MESH)
    assert got.code == 1 and got.res is None
    assert "load region 'tip' touches no body of the final mesh" in got.err                  # the fact of today
    assert "The mesh is result/part_only.stl" in got.err and "load:tip" in got.err           # and what the file is
    assert "without --part-only" in got.err and "regions.not_printed" in got.err


def test_a_singular_solve_carries_the_file_and_the_removed_bodies():
    from topopt.verify.factor import SingularSystem
    from topopt.verify.run import _singular_with_mesh
    body = SimpleNamespace(count=1, facts=[])
    exc = _singular_with_mesh(SingularSystem("singular", dict(reason="singular_matrix", solver="superlu", detail="x", dofs=12)), body,
                              np.arange(4), extra="CONTEXT OF PART ONLY.",
                              extra_facts=dict(part_only=dict(file="part_only.stl", removed=LABELS)))
    assert "CONTEXT OF PART ONLY." in str(exc)
    assert exc.facts["part_only"] == dict(file="part_only.stl", removed=LABELS) and exc.facts["fixed_nodes"] == 4


# -- gaps that the mutants of the review show (fix round 1, m3) ---------------------------------------------------------------------------
def variant_yaml(bar, name: str, edit) -> Path:
    """A copy of the YAML of the keyed bar; `edit(regions)` changes its `regions` block."""
    doc = yaml.safe_load(bar.keyed_yaml.read_text())
    edit(doc["regions"])
    path = bar.folder / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


@needs_gmsh
@needs_skfem
def test_a_removed_load_body_and_a_removed_fixed_body_are_selected_with_the_band_of_the_other_regions(bar, tmp_path):
    """The load body lies 0.3 mm above the top face. At band 0 it selects no facet, at the band (0.5 mm) it selects the facets below it."""
    lo, hi = (15, 3, 10.3), (17.5, 7, 12)
    box(bar.folder / "tip_gap.stl", lo, hi)

    def edit(r):
        r["load"] = {"tip": "tip_gap.stl"}
        r["not_printed"] = ["plate.stl", "tip_gap.stl", "boss.stl", "hole.stl"]
    got = verify(bar.keyed, tmp_path / "run", "--yaml", variant_yaml(bar, "gap.yaml", edit), "--part-only", *MESH)
    assert got.code == 0, got.err
    res = got.res
    tets, verts, _ = mesh_of(got.dir)
    near = box_sdf(verts, lo, hi)
    assert not (near <= 0.0).any()                                                      # at band 0 the body selects nothing
    facets = boundary_facets(tets)
    sel = (near <= res["band_mm"])[facets].all(axis=1)
    assert sel.sum() > 0
    tip = removed(res, "load:tip")
    assert tip["band_mm"] == res["band_mm"] == removed(res, "fixed:plate.stl")["band_mm"]      # one band for every region
    assert tip["boundary_condition"] == "traction" and tip["facets"] == int(sel.sum())
    assert tip["area_mm2"] == pytest.approx(float(facet_area(verts, facets[sel]).sum()), rel=1e-9)
    assert tip["contact"]["area_mm2"] == pytest.approx(0.0, abs=1e-6)                     # the body touches nothing: the facts say so
    assert res["cases"]["down"]["forces"][0]["resultant_N"] == pytest.approx([0.0, 0.0, -2.0], abs=1e-9)


@needs_gmsh
@needs_skfem
def test_a_removed_load_body_that_no_case_uses_gives_no_traction_and_says_why(bar, tmp_path):
    box(bar.folder / "tip2.stl", (15, 3, 10), (17.5, 7, 12))                           # sits on the top face; no load case names it

    def edit(r):
        r["load"] = {"tip": "tip.stl", "tip2": "tip2.stl"}
        r["not_printed"] = [*KEYED, "tip2.stl"]
    got = verify(bar.keyed, tmp_path / "run", "--yaml", variant_yaml(bar, "unused_load.yaml", edit), "--part-only", *MESH)
    assert got.code == 0, got.err
    res = got.res
    fact = removed(res, "load:tip2")
    assert fact["boundary_condition"] == "none" and "no load case uses this region" in fact["note"]
    assert fact["contact"]["area_mm2"] == pytest.approx(10.0, rel=1e-5) and fact["facets"] > 0       # the facts are measured
    assert [r["region"] for r in res["regions"]["load"]] == ["tip"] and len(res["cases"]["down"]["forces"]) == 1
    line = next(x for x in got.out.splitlines() if x.startswith("removed body load:tip2"))
    assert line.startswith("removed body load:tip2: no traction (no load case uses this region); ")
    assert "load:tip2" not in res["part_only"]["singular_note"]                          # it never held the part: no singular note for it


@needs_gmsh
@needs_skfem
def test_a_removed_fixed_body_that_selects_nothing_is_a_fact_when_another_fixed_region_holds_the_part(bar, tmp_path):
    box(bar.folder / "plate_far.stl", (-12, 0, 0), (-11, 10, 10))

    def edit(r):                                                                           # the plate stays in the YAML and is not removed
        r["fixed"] = ["plate.stl", "plate_far.stl"]
        r["not_printed"] = ["plate_far.stl", "tip.stl", "boss.stl", "hole.stl"]
    got = verify(bar.keyed, tmp_path / "run", "--yaml", variant_yaml(bar, "two_fixed.yaml", edit), "--part-only", *MESH)
    assert got.code == 0, got.err                                                          # the solve runs
    res = got.res
    far = removed(res, "fixed:plate_far.stl")
    assert far["boundary_condition"] == "none" and far["nodes"] == 0 and far["empty"] is True and "clamps nothing" in far["note"]
    assert "fixed:plate_far.stl" in res["part_only"]["singular_note"]
    plate = region(res, "fixed", "plate")
    assert [r["region"] for r in res["regions"]["fixed"]] == ["plate"]
    assert res["regions"]["fixed_nodes_total"] == plate["nodes"] > 0                       # the plate that is not removed holds the part
    case = res["cases"]["down"]
    assert case["equilibrium"]["force_relative_residual"] < 1e-8 and case["stiffness_N_per_mm"] > 0
    assert case["equilibrium"]["reaction_force_N"] == pytest.approx([0.0, 0.0, 2.0], abs=1e-8)


@needs_gmsh
@needs_skfem
def test_the_same_stl_check_sees_a_switch_of_the_file_in_one_folder(bar, tmp_path):
    shutil.copytree(bar.keyed, tmp_path / "run", ignore=shutil.ignore_patterns("verify"))
    run, mesh = tmp_path / "run", ["--mesh-size", "2.0", "--solver", "superlu"]
    seen, switch = [], []
    for flags in ([], ["--part-only"], []):                                                 # final.stl, part_only.stl, final.stl
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            assert main(["verify", str(run), *flags, *mesh]) == 0, err.getvalue()
        res = json.loads((run / "verify" / "verify.json").read_text())
        seen.append((res["run"]["verified_file"], (res.get("previous_verify") or {}).get("same_final_stl")))
        switch.append([x for x in out.getvalue().splitlines() if "earlier verify.json" in x])
    assert seen == [("final.stl", None), ("part_only.stl", False), ("final.stl", False)]
    # the note names the thing that differs: the mesh file (it is not always a final.stl; fix round 1, n1)
    assert switch[0] == []
    assert switch[1] == switch[2] == ["note: the earlier verify.json was for another mesh file (run.verified_file) (it was removed)"]


# -- two removed bodies with one label (fix round 1, m1) -----------------------------------------------------------------------------------
def two_plates_yaml(bar) -> Path:
    """A YAML with two fixed plates of one file name in two folders: the plate of the bar (contact 100 mm2) and `b/plate.stl`, which
    touches the right end of the part along 4 mm of its width (y 0..4, below the notch of the tip: contact 4 x 10 = 40 mm2).
    Both are in `regions.not_printed`."""
    (bar.folder / "b").mkdir(exist_ok=True)
    box(bar.folder / "b" / "plate.stl", (20, 0, 0), (22, 4, 10))
    doc = yaml.safe_load(bar.keyed_yaml.read_text())
    doc["regions"]["fixed"] = ["plate.stl", "b/plate.stl"]
    doc["regions"]["not_printed"] = ["plate.stl", "b/plate.stl", "tip.stl", "boss.stl", "hole.stl"]
    two = bar.folder / "two_plates.yaml"
    two.write_text(yaml.safe_dump(doc, sort_keys=False))
    return two


@needs_gmsh
@needs_skfem
def test_two_removed_bodies_with_one_label_have_their_own_contact_area_and_selection(bar, tmp_path):
    got = verify(bar.keyed, tmp_path / "run", "--yaml", two_plates_yaml(bar), "--part-only", *MESH)
    assert got.code == 0, got.err
    plates = [r for r in got.res["part_only"]["removed"] if r["label"] == "fixed:plate.stl"]
    assert [Path(r["file"]).parent.name for r in plates] == [bar.folder.name, "b"]              # two entries, in the order of the YAML
    assert plates[0]["contact"]["area_mm2"] == pytest.approx(100.0, rel=1e-5)               # not the area of the other plate
    assert plates[1]["contact"]["area_mm2"] == pytest.approx(40.0, rel=1e-5)
    assert plates[0]["nodes"] != plates[1]["nodes"] and all(r["boundary_condition"] == "clamp" for r in plates)
    assert got.res["regions"]["fixed_nodes_total"] == plates[0]["nodes"] + plates[1]["nodes"]     # the two plates share no node
    lines = [x for x in got.out.splitlines() if x.startswith("removed body fixed:plate.stl")]
    assert len(lines) == 2 and "contact area of the cut 100.0 mm2" in lines[0] and "contact area of the cut 40.0 mm2" in lines[1]


@needs_gmsh
@needs_skfem
def test_the_facts_of_a_part_with_no_support_list_the_removed_fixed_bodies_by_body(bar, tmp_path):
    """Two removed fixed plates of one file name, both away from the part: the error lists the two files with their own counts."""
    from topopt.verify import verify_run
    from topopt.verify.errors import UnsolvableMesh
    far = []
    for name, lo, hi in (("far_a", (-12, 0, 0), (-11, 10, 10)), ("far_b", (30, 0, 0), (31, 10, 10))):
        (bar.folder / name).mkdir(exist_ok=True)
        box(bar.folder / name / "plate.stl", lo, hi)
        far.append(str((bar.folder / name / "plate.stl").resolve()))
    doc = yaml.safe_load(bar.keyed_yaml.read_text())
    doc["regions"]["fixed"] = ["far_a/plate.stl", "far_b/plate.stl"]
    doc["regions"]["not_printed"] = ["far_a/plate.stl", "far_b/plate.stl", "tip.stl", "boss.stl", "hole.stl"]
    yml = bar.folder / "far_two.yaml"
    yml.write_text(yaml.safe_dump(doc, sort_keys=False))
    shutil.copytree(bar.keyed, tmp_path / "run", ignore=shutil.ignore_patterns("verify"))
    with pytest.raises(UnsolvableMesh) as e:
        verify_run(tmp_path / "run", yaml_path=yml, part_only=True, mesh_size=1.5, solver="superlu")
    assert e.value.facts["part_only"]["fixed_selected_nodes"] == [dict(label="fixed:plate.stl", file=far[0], nodes=0),
                                                                 dict(label="fixed:plate.stl", file=far[1], nodes=0)]
    assert f"fixed:plate.stl ({far[0]}) selects no node" in str(e.value) and f"fixed:plate.stl ({far[1]}) selects no node" in str(e.value)


def test_contact_areas_and_selected_nodes_are_listed_by_body_not_by_label(tmp_path):
    from topopt.verify import part_only as vpart_only
    a, b = tmp_path / "a" / "plate.stl", tmp_path / "b" / "plate.stl"
    a.parent.mkdir()
    b.parent.mkdir()
    box(a, (-2, 0, 0), (0, 10, 10))
    box(b, (20, 0, 0), (22, 5, 10))
    box(tmp_path / "part.stl", (0, 0, 0), (20, 10, 10))
    areas = vpart_only.contact_areas(tmp_path / "part.stl", [a, b])
    assert [x["area_mm2"] for x in areas] == [pytest.approx(100.0), pytest.approx(50.0)]
    label = "fixed:plate.stl"
    text, nodes = vpart_only.selection_text([(label, a, dict(nodes=10, band_mm=0.5)), (label, b, dict(nodes=0, band_mm=0.5))])
    assert nodes == [dict(label=label, file=str(a), nodes=10), dict(label=label, file=str(b), nodes=0)]
    assert str(a) in text and str(b) in text                                                # the text tells the two bodies apart
    assert "selects 10 nodes" in text and "selects no node" in text


# -- the names and the help ----------------------------------------------------------------------------------------------------------
def test_the_file_name_is_the_one_of_the_post_step_and_the_option_has_a_cli_and_an_mcp_name():
    from topopt.verify import names
    from topopt.verify import part_only as vpart_only
    assert vpart_only.FILE == post.PART_ONLY_FILE == "part_only.stl"
    with names.use_style("cli"):
        assert names.opt("part_only") == "--part-only"
    with names.use_style("mcp"):
        assert names.opt("part_only") == "part_only"


def test_the_cli_help_describes_the_option(capsys):
    with pytest.raises(SystemExit) as e:
        main(["verify", "--help"])
    assert e.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--part-only" in text and "regions.not_printed" in text and "part_only.stl" in text
    assert "clamp" in text and "contact" in text
    assert "instead of nominal.stl or final.stl" in text                                  # the default file has stock: nominal.stl (fix round 1, n3)
    assert "instead of `nominal.stl` or `final.stl`" in " ".join((ROOT / "topopt_mcp" / "README.md").read_text().split())


# -- the MCP tool ---------------------------------------------------------------------------------------------------------------------------
def _fake_extras():
    return {"gmsh": {"state": "ok", "version": "4.15.2", "detail": "", "fix": ""},
            "scikit-fem": {"state": "ok", "version": "x"}, "cholespy": {"state": "ok", "version": None}}


@pytest.fixture
def mcp(bar, tmp_path, monkeypatch):
    """The server module, a workspace with a copy of the keyed run and one of the plain run, and a fake child for `verify_run`."""
    server = pytest.importorskip("topopt_mcp.server")
    from topopt.verify import run as vrun
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TOPOPT_MCP_VERIFY_CMD",
                       f"{sys.executable} {ROOT}/tests/serve_verify_fake.py {{run_dir}} ok --step 0.25 --job topopt-verify")
    monkeypatch.setattr(vrun, "extras_facts", _fake_extras)                       # the preflight needs no gmsh when the child is a fake
    server._verify.clear()
    keyed, plain = tmp_path / "runs" / "keyed", tmp_path / "runs" / "plain"
    shutil.copytree(bar.keyed, keyed, ignore=shutil.ignore_patterns("verify"))
    shutil.copytree(bar.plain, plain, ignore=shutil.ignore_patterns("verify"))
    seen = []
    real = server._verify_command
    monkeypatch.setattr(server, "_verify_command", lambda run_dir, args: (seen.append(list(args)), real(run_dir, args))[1])
    yield SimpleNamespace(server=server, keyed=keyed, plain=plain, args=seen)
    proc = server._verify.get("proc")
    if proc is not None and proc.poll() is None:
        import signal
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    server._verify.clear()


def test_the_mcp_server_imports_when_docstrings_are_stripped(tmp_path):
    """`python -OO` removes the docstrings; `_tight_doc` must not fail on `fn.__doc__` of None (fix round 1, m2)."""
    pytest.importorskip("mcp")
    import subprocess
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-OO", "-c", "import topopt_mcp.server"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-600:]


def test_the_mcp_option_becomes_the_flag_of_the_child_command(mcp):
    args = (None, None, None, None, 99.9, None, None, False, None, 2.0, True)           # the defaults of verify_run
    without = mcp.server._verify_args(*args)
    assert "--part-only" not in without
    assert mcp.server._verify_args(*args, part_only=True) == without + ["--part-only"]


def test_mcp_estimate_with_part_only_carries_the_block(mcp):
    res = mcp.server.verify_run(run_dir=str(mcp.keyed), part_only=True, mesh_size_mm=1.5)
    assert res["ok"] and res["estimate_only"]
    po = res["part_only"]
    assert po["file"] == "part_only.stl" and po["sha256"] == sha256(mcp.keyed / "result" / "part_only.stl")
    assert [r["label"] for r in po["removed"]] == LABELS and "notes" not in res


def test_mcp_start_reply_has_the_part_only_block_and_the_child_gets_the_flag(mcp):
    res = mcp.server.verify_run(run_dir=str(mcp.keyed), estimate_only=False, part_only=True, mesh_size_mm=1.5)
    assert res["ok"] and res["started"]
    assert mcp.args and "--part-only" in mcp.args[-1] and mcp.args[-1][mcp.args[-1].index("--mesh-size") + 1] == "1.5"
    po = res["part_only"]
    assert po["on"] is True and po["file"] == "part_only.stl" and po["sha256"] == sha256(mcp.keyed / "result" / "part_only.stl")
    assert [r["label"] for r in po["removed"]] == LABELS and po["yaml_vs_run"]["matches"] is True
    assert "notes" not in res and res["yaml_vs_run"]["matches"] is True


def test_mcp_start_reply_without_the_option_names_the_parameter_in_one_note(mcp):
    res = mcp.server.verify_run(run_dir=str(mcp.keyed), estimate_only=False, mesh_size_mm=1.5)
    assert res["ok"] and res["started"] and "part_only" not in res
    assert "--part-only" not in mcp.args[-1]
    from topopt.verify import names
    from topopt.verify import part_only as vpart_only
    with names.use_style("mcp"):
        assert res["notes"] == [vpart_only.available_note("final.stl", False)]
    assert "Set part_only to mesh part_only.stl" in res["notes"][0]
    assert "--part-only" not in res["notes"][0]                                    # the MCP text names the parameter, not the CLI flag


def test_mcp_option_on_a_run_without_the_file_is_an_error_with_the_options(mcp):
    res = mcp.server.verify_run(run_dir=str(mcp.plain), part_only=True, estimate_only=False)
    assert res["ok"] is False and res["exit_code"] == 1
    assert "no result/part_only.stl" in res["error"] and "regions.not_printed" in res["error"]
    assert "without part_only" in res["error"] and "--part-only" not in res["error"]
    assert not mcp.args                                                          # no child was started
    est = mcp.server.verify_run(run_dir=str(mcp.plain), part_only=True)
    assert est["ok"] is False and est["exit_code"] == 1 and "without part_only" in est["error"]


def test_the_mcp_tool_description_names_the_parameter_and_stays_within_the_limit(mcp):
    doc = inspect.getdoc(mcp.server.verify_run)
    assert "part_only" in doc and "regions.not_printed" in doc and "part_only.stl" in doc
    assert len(doc) <= 4096


def test_the_verify_status_description_names_the_file_after_part_only(mcp):
    """`result.run.verified_file` can be part_only.stl; the text of the tool must say so (fix round 1, m5)."""
    doc = inspect.getdoc(mcp.server.verify_status)
    assert "verified_file" in doc and "part_only.stl" in doc and "part_only=true" in doc
    assert len(doc) <= 4096


# -- the hook, boundary conditions only ----------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hook(tmp_path_factory):
    """The short hook (`examples/hook/hook_short.yaml`) with the wall plate, the tip and the top screw hole as not printed bodies."""
    pytest.importorskip("skfem")
    from helpers import tet_probe
    probe = tet_probe()
    if not probe.ok:
        pytest.skip(f"needs gmsh (the 'tet' extra): {probe.state}: {probe.detail}")
    src = ROOT / "examples" / "hook"
    folder = tmp_path_factory.mktemp("p4b3_hook")
    for f in src.glob("*.stl"):
        shutil.copy(f, folder / f.name)
    doc = yaml.safe_load((src / "hook_short.yaml").read_text())
    doc["regions"]["not_printed"] = ["wall_plate.stl", "hook_tip.stl", "screw_hole_top.stl"]
    yml = folder / "hook_np.yaml"
    yml.write_text(yaml.safe_dump(doc, sort_keys=False))
    assert main(["run", str(yml), "--out", str(folder / "run"), "--quiet"]) == 0
    got = verify(folder / "run", folder / "v", "--part-only", "--mesh-size", "3", "--solver", "superlu")
    return SimpleNamespace(folder=folder, got=got, doc=doc)


@pytest.mark.slow
@needs_gmsh
@needs_skfem
def test_hook_part_only_boundary_conditions_clamp_on_the_contact_face_load_resultant_and_equilibrium(hook):
    """Only boundary-condition facts. The stiffness of this mesh is not asserted: the film around the removed tip decides it."""
    got = hook.got
    assert got.code == 0, got.err
    res = got.res
    assert res["part_only"]["on"] is True and [r["label"] for r in res["part_only"]["removed"]] == [
        "fixed:wall_plate.stl", "load:tip", "keep_out:screw_hole_top.stl"]
    _, verts, nodes = mesh_of(got.dir)
    lo, hi = np.array([0.0, -17.5, -30.0]), np.array([3.5, 17.5, 36.0])              # the wall plate: a box (examples/hook/wall_plate.stl)
    plate = removed(res, "fixed:wall_plate.stl")
    assert plate["boundary_condition"] == "clamp"
    rule = box_sdf(nodes, lo, hi) <= res["band_mm"]
    assert plate["nodes"] == int(rule.sum()) == res["regions"]["fixed_nodes_total"] > 0
    assert plate["vertices"] == int((box_sdf(verts, lo, hi) <= res["band_mm"]).sum())
    on_plane = np.abs(nodes[:, 0] - 3.5) <= 1e-3                                       # the wall side of the part: the plane x = 3.5 mm
    assert on_plane.sum() > 0 and rule[on_plane].all()
    assert removed(res, "load:tip")["boundary_condition"] == "traction" and removed(res, "load:tip")["facets"] > 0
    hole = removed(res, "keep_out:screw_hole_top.stl")
    assert hole["boundary_condition"] == "none" and "no boundary condition" in hole["note"]
    assert res["regions"]["fixed_nodes_total"] == plate["nodes"]                       # the hole clamped nothing
    for name, force in {"hang": [0.0, 0.0, -1.96], "pull_off": [20.89, 5.21, -20.89]}.items():
        case = res["cases"][name]
        assert case["forces"][0]["resultant_N"] == pytest.approx(force, abs=1e-9)
        assert case["forces"][0]["resultant_relative_error"] < 1e-9
        eq = case["equilibrium"]                                  # 1e-9 to 4e-9 measured: the soft film makes the system ill-conditioned
        assert eq["force_relative_residual"] < 1e-6 and eq["moment_relative_residual"] < 1e-6
