"""Tests of `regions.not_printed` and `result/part_only.stl` (P4b-2, CONTRACTS.md section 9.1).

The case is a bar: the design box x 0..20, a fixed plate on the left whose face lies at x = 0.37 (inside the design box, off the voxel
faces), a load box at the tip, a solid box inside the design and a keep-out box (a hole). The main run lists all four kinds as not
printed. The volumes of the boxes are exact numbers, so the volume test does not use the code that it tests.

Accuracy that these tests use (measured on this case, Linux x86_64, 2026-10-22):
- the volume of the file against `final - removed boxes - loose pieces`: 5e-5 mm3 of 463 mm3 (6e-8 of the volume; the loose-piece
  volume in the metrics is rounded to 1e-4 mm3). The test allows 1e-5 of the volume of final.stl (0.008 mm3). This is a test bound, not a limit of the tool.
- a vertex of the contact face against the plane of the plate: 0 (the plane is float32 of 0.37). The test allows one float32 step.
"""
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_optimize import box  # noqa: E402
from topopt import part_only, post, smooth_sdf  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.config import NotPrinted, UserError, load_config  # noqa: E402
from topopt.snapshot import RunWriter  # noqa: E402

PLATE_X = 0.37                                   # the face of the plate: off the voxel faces (h 1)
BOXES = {                                        # name -> (lo, hi): the region files of the case
    "plate": ((-2, 0, 0), (PLATE_X, 10, 10)),
    "tip": ((18, 4, 0), (20, 6, 3)),
    "boss": ((6.3, 3.1, 3.1), (9.7, 6.9, 6.9)),
    "hole": ((12.2, 3.3, -1), (14.4, 6.7, 11)),
}
VOLUME = {n: float(np.prod(np.subtract(hi, lo))) for n, (lo, hi) in BOXES.items()}
ALL_KINDS = ["plate.stl", "tip.stl", "boss.stl", "hole.stl"]
LABELS = ["fixed:plate.stl", "load:tip", "solid:boss.stl", "keep_out:hole.stl"]
TIMING = {"run_time_s", "t_s", "elapsed_s", "started_at", "t_iter_s"}


def write_case(folder: Path, not_printed=None, name="c.yaml") -> Path:
    """The files of the bar case in `folder` and a YAML. `not_printed` None: the key is not there."""
    folder.mkdir(parents=True, exist_ok=True)
    box(folder / "design.stl", (0, 0, 0), (20, 10, 10))
    for n, (lo, hi) in BOXES.items():
        box(folder / f"{n}.stl", lo, hi)
    regions = {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}, "solid": ["boss.stl"],
               "keep_out": ["hole.stl"]}
    if not_printed is not None:
        regions["not_printed"] = not_printed
    doc = {"name": "bar", "regions": regions,
           "load_cases": [{"name": "down", "weight": 3.0, "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
           "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": 8, "beta_schedule": [[0, 1], [4, 2]]}}
    p = folder / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def clean(obj):
    """The object without the timing keys."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k not in TIMING}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def box_sdf(points, lo, hi):
    """Signed distance of points to a box (negative inside)."""
    q = np.maximum(np.asarray(lo, float) - points, points - np.asarray(hi, float))
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def load_stl(path):
    return trimesh.load(path, file_type="stl")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two runs of the same bar: without the key, and with all four kinds listed."""
    folder = tmp_path_factory.mktemp("bar")
    plain_yaml = write_case(folder, None, "plain.yaml")
    keyed_yaml = write_case(folder, ALL_KINDS, "keyed.yaml")
    plain, keyed = folder / "plain", folder / "keyed"
    assert main(["run", str(plain_yaml), "--out", str(plain), "--quiet"]) == 0
    assert main(["run", str(keyed_yaml), "--out", str(keyed), "--quiet"]) == 0
    return {"folder": folder, "plain": plain, "keyed": keyed, "keyed_yaml": keyed_yaml}


# -- loader ----------------------------------------------------------------------------------------------------------
def test_without_the_key_nothing_is_listed(tmp_path):
    assert load_config(write_case(tmp_path, None)).not_printed == ()
    assert load_config(write_case(tmp_path, [], "empty.yaml")).not_printed == ()      # an empty list is the same as no key


@pytest.mark.parametrize("entry,role,name", [("plate.stl", "fixed", "plate.stl"), ("tip.stl", "load", "tip"),
                                             ("boss.stl", "solid", "boss.stl"), ("hole.stl", "keep_out", "hole.stl")])
def test_every_kind_of_region_can_be_listed(tmp_path, entry, role, name):
    cfg = load_config(write_case(tmp_path, [entry]))
    assert cfg.not_printed == (NotPrinted(role, name, (tmp_path / entry).resolve()),)
    assert cfg.not_printed[0].label == f"{role}:{name}"          # the form of check_regions: no name hides another


def test_the_order_of_the_list_is_kept_and_a_repeated_file_counts_once(tmp_path):
    cfg = load_config(write_case(tmp_path, ["hole.stl", "plate.stl", "./plate.stl", "sub/../hole.stl"]))
    assert [n.label for n in cfg.not_printed] == ["keep_out:hole.stl", "fixed:plate.stl"]


def test_a_file_that_two_kinds_name_gives_two_entries(tmp_path):
    p = write_case(tmp_path, ["tip.stl"])
    doc = yaml.safe_load(p.read_text())
    doc["regions"]["solid"] = ["tip.stl"]                         # the old form of a body that is also a load region
    p.write_text(yaml.safe_dump(doc))
    assert [n.label for n in load_config(p).not_printed] == ["load:tip", "solid:tip.stl"]


def test_the_match_is_by_resolved_path_not_by_file_name(tmp_path):
    """Two fixed files with one file name in two folders: the entry names one of them."""
    p = write_case(tmp_path, ["sub/plate.stl"])
    (tmp_path / "sub").mkdir(exist_ok=True)
    box(tmp_path / "sub" / "plate.stl", (-3, 0, 0), (-2.5, 10, 10))
    doc = yaml.safe_load(p.read_text())
    doc["regions"]["fixed"] = ["plate.stl", "sub/plate.stl"]
    p.write_text(yaml.safe_dump(doc))
    got = load_config(p).not_printed
    assert got == (NotPrinted("fixed", "plate.stl", (tmp_path / "sub" / "plate.stl").resolve()),)
    doc["regions"]["not_printed"] = ["plate.stl"]
    p.write_text(yaml.safe_dump(doc))
    assert load_config(p).not_printed[0].path == (tmp_path / "plate.stl").resolve()


def test_an_entry_that_matches_no_region_is_an_error_that_names_the_entry_and_the_region_files(tmp_path):
    box(tmp_path / "other.stl", (30, 0, 0), (31, 1, 1))
    p = write_case(tmp_path, ["plate.stl", "other.stl"])
    with pytest.raises(UserError) as e:
        load_config(p)
    text = str(e.value)
    assert "regions.not_printed[1]" in text and "other.stl" in text and "not a file of any region" in text
    for name in ("plate.stl", "tip.stl", "boss.stl", "hole.stl"):          # every region file is named
        assert name in text
    assert f"(file: {p.resolve()})" in text and "\n" not in text


def test_the_design_region_is_named_as_such(tmp_path):
    with pytest.raises(UserError, match=r"regions.not_printed\[0\].*design region: the part itself"):
        load_config(write_case(tmp_path, ["design.stl"]))


@pytest.mark.parametrize("value,text", [("plate.stl", "must be a list of STL files"), ([3], "must be a non-empty string"),
                                        (["gone.stl"], "STL file not found")])
def test_a_bad_value_is_a_loader_error(tmp_path, value, text):
    with pytest.raises(UserError, match=text):
        load_config(write_case(tmp_path, value))


def test_the_inputs_of_the_checkpoint_do_not_change_with_the_key(tmp_path):
    """The bodies are not in `Regions`: the region hashes and the effective settings of a run are the same with and without the key."""
    from topopt import checkpoint

    a, b = load_config(write_case(tmp_path, None, "a.yaml")), load_config(write_case(tmp_path, ALL_KINDS, "b.yaml"))
    assert checkpoint.region_hashes(a) == checkpoint.region_hashes(b)
    assert checkpoint.effective_settings(a) == checkpoint.effective_settings(b)
    assert checkpoint.input_hashes(a)["regions"] == checkpoint.input_hashes(b)["regions"]


# -- run.json, metrics.json, run.log ---------------------------------------------------------------------------------
def test_the_start_of_the_run_writes_the_plan(runs):
    cfg = load_config(runs["keyed_yaml"])
    plan = RunWriter._part_only_plan(cfg)
    assert plan == {"file": "part_only.stl", "removed": LABELS,
                    "entries": [{"role": r, "name": n, "file": f} for (r, n), f in
                                zip([("fixed", "plate.stl"), ("load", "tip"), ("solid", "boss.stl"), ("keep_out", "hole.stl")], ALL_KINDS)]}


def test_run_json_and_metrics_carry_the_same_facts(runs):
    out = runs["keyed"]
    metrics = json.loads((out / "result" / "metrics.json").read_text())["part_only"]
    run = json.loads((out / "run.json").read_text())["part_only"]
    entries = run.pop("entries")
    assert run == metrics and [e["role"] for e in entries] == ["fixed", "load", "solid", "keep_out"]
    assert metrics["applied"] is True and metrics["file"] == "part_only.stl" and metrics["removed"] == LABELS
    assert metrics["mesh_method"] == "sdf" and metrics["watertight"] is True and metrics["bodies"] == 1
    assert set(metrics["overlap_mm3"]) == set(LABELS)
    assert metrics["leftovers_dropped"]["bodies"] >= 0
    assert metrics["stl_roundtrip"]["ok"] and metrics["stl_roundtrip"]["volume_mm3"] == metrics["volume_after_mm3"]
    assert metrics["volume_before_mm3"] == json.loads((out / "result" / "metrics.json").read_text())["mesh"]["stl_roundtrip"]["volume_mm3"]
    assert list(json.loads((out / "result" / "metrics.json").read_text()))[-1] == "verdict"      # the key order of section 5 holds


def test_the_log_says_what_was_cut(runs):
    log = (runs["keyed"] / "run.log").read_text()
    assert "part_only: 4 body(ies) cut out (fixed:plate.stl, load:tip, solid:boss.stl, keep_out:hole.stl)" in log
    assert "1 body(ies) in result/part_only.stl" in log


def test_a_keep_out_region_shares_no_volume_with_the_mesh(runs):
    block = json.loads((runs["keyed"] / "result" / "metrics.json").read_text())["part_only"]
    assert block["overlap_mm3"]["keep_out:hole.stl"] == 0.0 and block["overlap_mm3"]["fixed:plate.stl"] > 0


# -- the mesh --------------------------------------------------------------------------------------------------------
def test_the_overlap_of_each_removed_box_is_its_volume(runs):
    """The sdf step unites each fixed, load and solid body exactly, so the mesh holds the whole box."""
    block = json.loads((runs["keyed"] / "result" / "metrics.json").read_text())["part_only"]
    for label, n in (("fixed:plate.stl", "plate"), ("load:tip", "tip"), ("solid:boss.stl", "boss")):
        assert block["overlap_mm3"][label] == pytest.approx(VOLUME[n], abs=1e-3), label


def test_the_file_is_one_closed_body_and_final_stl_is_unchanged(runs):
    part, final = load_stl(runs["keyed"] / "result" / "part_only.stl"), load_stl(runs["keyed"] / "result" / "final.stl")
    assert part.is_watertight and part.is_winding_consistent and part.volume > 0
    assert len(part.split(only_watertight=False)) == 1
    assert (runs["keyed"] / "result" / "final.stl").read_bytes() == (runs["plain"] / "result" / "final.stl").read_bytes()
    assert part.volume < final.volume


def test_the_volume_is_final_minus_the_removed_bodies_and_the_loose_pieces(runs):
    out = runs["keyed"]
    part, final = load_stl(out / "result" / "part_only.stl"), load_stl(out / "result" / "final.stl")
    block = json.loads((out / "result" / "metrics.json").read_text())["part_only"]
    removed = sum(VOLUME[n] for n in ("plate", "tip", "boss"))                       # the hole is already empty in final.stl
    expected = final.volume - removed - block["leftovers_dropped"]["volume_mm3"]
    assert part.volume == pytest.approx(expected, abs=1e-5 * final.volume)           # measured: 5e-5 mm3
    assert block["leftovers_dropped"]["volume_mm3"] < 2e-3 * final.volume            # measured: 0.58 mm3 of 761 (7.6e-4)
    assert block["volume_before_mm3"] == pytest.approx(final.volume, abs=2e-3)
    assert block["volume_after_mm3"] == pytest.approx(part.volume, abs=2e-3)
    assert block["removed_mm3"] == pytest.approx(block["volume_before_mm3"] - block["volume_after_mm3"], abs=2e-3)


def test_the_contact_face_is_the_plane_of_the_plate(runs):
    """Every vertex near the plate plane is on it, within one float32 step: not at a voxel face (h 1: off by up to 0.5 mm)."""
    part = load_stl(runs["keyed"] / "result" / "part_only.stl")
    plane = float(np.float32(PLATE_X))
    v = np.asarray(part.vertices)
    near = np.abs(v[:, 0] - PLATE_X) < 1e-3
    assert near.sum() > 20
    assert np.abs(v[near, 0] - plane).max() <= np.spacing(np.float32(PLATE_X))
    face = near[part.faces].all(axis=1)
    assert face.sum() > 20 and (part.face_normals[face][:, 0] < -0.999).all()        # the faces look into the plate
    assert part.area_faces[face].sum() > 50                                         # the root of the bar: most of the 100 mm2
    # no material in the plate, material next to it wherever final.stl has some
    final = load_stl(runs["keyed"] / "result" / "final.stl")
    g = np.linspace(0.25, 9.75, 20)
    yz = np.array([(y, z) for y in g for z in g])
    inside_plate = np.column_stack([np.full(len(yz), PLATE_X - 0.01), yz])
    next_to = np.column_stack([np.full(len(yz), PLATE_X + 0.01), yz])
    assert not part.contains(inside_plate).any()
    assert (part.contains(next_to) == final.contains(next_to)).all() and final.contains(next_to).any()


def test_every_point_is_in_the_part_if_it_is_in_final_and_in_no_removed_body(runs):
    part, final = load_stl(runs["keyed"] / "result" / "part_only.stl"), load_stl(runs["keyed"] / "result" / "final.stl")
    rng = np.random.default_rng(7)
    pts = rng.uniform([-2.5, -0.5, -0.5], [20.5, 10.5, 10.5], size=(1500, 3))
    sdf = np.column_stack([box_sdf(pts, *BOXES[n]) for n in BOXES])
    safe = np.abs(sdf).min(axis=1) > 0.1                  # the loose skin of a removed body lies within microns of its surface
    expect = final.contains(pts) & (sdf > 0).all(axis=1)
    got = part.contains(pts)
    assert safe.sum() > 1000 and expect[safe].sum() > 50
    assert (got[safe] == expect[safe]).all()


def test_the_plain_run_has_no_trace_and_a_stale_file_is_removed(runs, tmp_path):
    out = runs["plain"]
    assert "part_only" not in json.loads((out / "run.json").read_text())
    assert "part_only" not in json.loads((out / "result" / "metrics.json").read_text())
    assert not (out / "result" / "part_only.stl").exists() and "part_only" not in (out / "run.log").read_text()
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    (copy / "result" / "part_only.stl").write_bytes(b"old")           # a file of an older post-process of this folder
    post.finish(copy)
    assert not (copy / "result" / "part_only.stl").exists()


def test_with_the_key_every_other_output_is_the_same_as_without_it(runs):
    """The key changes final.stl, the snapshots, the labels, the history and every other number of metrics.json and run.json by nothing."""
    a, b = runs["plain"], runs["keyed"]
    for f in ("result/final.stl", "result/preview.bin", "labels.bin"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    names = sorted(p.name for p in (a / "snapshots").iterdir())
    assert names == sorted(p.name for p in (b / "snapshots").iterdir())
    for n in names:
        assert (a / "snapshots" / n).read_bytes() == (b / "snapshots" / n).read_bytes(), n

    def lines(d):
        return [clean(json.loads(x)) for x in (d / "history.jsonl").read_text().splitlines()]
    assert lines(a) == lines(b)
    ma, mb = (json.loads((d / "result" / "metrics.json").read_text()) for d in (a, b))
    assert mb.pop("part_only") and clean(ma) == clean(mb)
    ra, rb = (json.loads((d / "run.json").read_text()) for d in (a, b))
    assert rb.pop("part_only")
    assert ra.pop("yaml_path").endswith("plain.yaml") and rb.pop("yaml_path").endswith("keyed.yaml")      # the two YAML files differ by one key
    assert clean(ra) == clean(rb)


# -- the MCP result --------------------------------------------------------------------------------------------------
def test_the_mcp_result_carries_the_facts_and_the_file(runs):
    server = pytest.importorskip("topopt_mcp.server")
    res = server.run_result(str(runs["keyed"]), preview=False)
    assert res["done"] and res["metrics"]["part_only"]["removed"] == LABELS
    assert Path(res["part_only_stl"]) == (runs["keyed"] / "result" / "part_only.stl").resolve()
    assert res["part_only_stl_rel"].endswith("part_only.stl") and Path(res["final_stl"]).name == "final.stl"
    plain = server.run_result(str(runs["plain"]), preview=False)
    assert plain["done"] and "part_only_stl" not in plain and "part_only_stl_rel" not in plain and "part_only" not in plain["metrics"]


# -- faults are facts, not a stop -------------------------------------------------------------------------------------
def test_a_missing_region_file_gives_a_fact_and_no_file(runs, tmp_path, caplog):
    copy = tmp_path / "copy"
    shutil.copytree(runs["keyed"], copy)
    run = json.loads((copy / "run.json").read_text())
    run["part_only"]["entries"][0]["file"] = "gone.stl"
    (copy / "run.json").write_text(json.dumps(run))
    with caplog.at_level(logging.WARNING, logger="topopt.part_only"):
        metrics = post.finish(copy)
    block = metrics["part_only"]
    assert block["applied"] is False and block["file"] is None and "gone.stl" in block["error"] and block["removed"] == LABELS
    assert not (copy / "result" / "part_only.stl").exists()
    assert json.loads((copy / "status.json").read_text())["state"] == "done"            # the run is finished
    assert load_stl(copy / "result" / "final.stl").is_watertight
    assert part_only.log_lines(block) == [f"part_only: not made ({block['error']})"]
    assert any("part_only.stl not made" in r.message for r in caplog.records)


def test_blur_says_what_its_voxel_bodies_do(runs, tmp_path):
    """The blur method holds a removed body as a voxel shape; the cut still runs, the notes and the loose pieces say it."""
    copy = tmp_path / "copy"
    shutil.copytree(runs["keyed"], copy)
    run = json.loads((copy / "run.json").read_text())
    run["mesh_method"] = "blur"
    (copy / "run.json").write_text(json.dumps(run))
    block = post.finish(copy)["part_only"]
    assert block["applied"] and block["mesh_method"] == "blur" and "Option: settings.mesh_method: sdf" in " ".join(block["notes"])
    assert load_stl(copy / "result" / "part_only.stl").is_watertight
    assert block["leftovers_dropped"]["bodies"] > 0                                         # the voxel steps outside the exact bodies


# -- the cut on built meshes -----------------------------------------------------------------------------------------
def cube(lo, hi):
    import manifold3d as m3d

    return m3d.Manifold.cube(tuple(np.subtract(hi, lo))).translate(tuple(lo))


def arrays(man):
    return smooth_sdf._from_manifold(man)


def field(shape=(15, 11, 11), inside=((0, 14), (0, 10), (0, 10))):
    """A field with h 1 and origin 0: material (design, density 1) in the box `inside`, empty elsewhere."""
    labels = np.zeros(shape, np.uint8)
    labels[inside[0][0]:inside[0][1], inside[1][0]:inside[1][1], inside[2][0]:inside[2][1]] = post.DESIGN
    return part_only.FieldMaterial((labels > 0).astype(np.float32), labels, (0.0, 0.0, 0.0), 1.0)


def reload(v, f):
    import trimesh as tm

    data = tm.Trimesh(v, f, process=False).export(file_type="stl")
    return smooth_sdf.stl_facts(data)


def test_the_cut_drops_the_thin_skin_that_the_plain_boolean_leaves(tmp_path):
    """A skin of 0.005 mm around the removed box B: the plain cut leaves 2 thin sheets that meet the part along an edge each."""
    a = cube((0, 0, 0), (10, 10, 10))
    skin = cube((10, -0.005, 0), (14, 10.005, 10))
    removed = cube((10, 0, 0), (14, 10, 10))
    box(tmp_path / "b.stl", (10, 0, 0), (14, 10, 10))
    v, f = arrays(a + skin)
    plain = (smooth_sdf._manifold_of(v, f) - removed)
    assert len(plain.decompose()) == 3                                              # the part and two sheets
    vv, ff, facts = part_only.cut(v, f, [("fixed:b", tmp_path / "b.stl")], field())
    assert facts["leftovers_dropped"] == {"bodies": 2, "volume_mm3": pytest.approx(2 * 4 * 10 * 0.005, abs=1e-6)}
    got = reload(vv, ff)
    assert got["ok"] and got["bodies"] == 1 and got["nonmanifold_edges"] == 0 and got["volume_mm3"] == pytest.approx(1000.0, abs=1e-3)
    assert facts["overlap_mm3"] == {"fixed:b": pytest.approx(400.0, abs=1e-3)}


def test_a_piece_that_only_a_removed_body_held_stays(tmp_path):
    """Two members joined by the removed body: both are part of the print (the rule of the interface step would drop one of them)."""
    bar = cube((0, 0, 0), (14, 10, 10))
    box(tmp_path / "mid.stl", (5, 0, 0), (9, 10, 10))
    v, f = arrays(bar)
    vv, ff, facts = part_only.cut(v, f, [("solid:mid", tmp_path / "mid.stl")], field())
    got = reload(vv, ff)
    assert got["ok"] and got["bodies"] == 2 and facts["leftovers_dropped"]["bodies"] == 0
    assert got["volume_mm3"] == pytest.approx(1000.0, abs=1e-3)


def test_a_cut_that_leaves_no_material_is_an_error(tmp_path):
    bar = cube((0, 0, 0), (14, 10, 10))
    box(tmp_path / "all.stl", (-1, -1, -1), (15, 11, 11))
    v, f = arrays(bar)
    with pytest.raises(ValueError, match="leaves no material"):
        part_only.cut(v, f, [("fixed:all", tmp_path / "all.stl")], field())


def test_a_body_with_a_tiny_volume_but_a_field_voxel_stays(tmp_path):
    """The test for 'holds material' is the voxel centre of the field, not the size of the body."""
    mat = field()
    v, f = arrays(cube((2, 2, 2), (3, 3, 3)))                  # the centre (2.5, 2.5, 2.5) is a field voxel
    assert mat.held_by(v, f)
    v, f = arrays(cube((2.6, 2.6, 2.6), (2.9, 2.9, 2.9)))      # no centre inside: a piece of one voxel
    assert not mat.held_by(v, f)
    v, f = arrays(cube((20, 2, 2), (21, 3, 3)))                # outside the field
    assert not mat.held_by(v, f) and len(mat.points_in([20, 2, 2], [21, 3, 3])) == 0


def test_a_load_label_or_a_fixed_label_with_density_is_material():
    labels = np.zeros((4, 4, 4), np.uint8)
    labels[1, 1, 1], labels[2, 2, 2], labels[1, 2, 1], labels[2, 1, 1] = post.FIXED, post.LOAD, post.KEEP_OUT, post.DESIGN
    rho = np.ones(labels.shape, np.float32)
    rho[2, 1, 1] = 0.4                                          # a design voxel at or below 0.5 is not material
    mat = part_only.FieldMaterial(rho, labels, (0.0, 0.0, 0.0), 1.0)
    assert mat.mask.sum() == 2 and mat.mask[1, 1, 1] and mat.mask[2, 2, 2]


def test_the_log_lines_for_a_cut_with_pieces_dropped():
    block = {"applied": True, "removed": ["fixed:a.stl"], "volume_before_mm3": 10.0, "volume_after_mm3": 6.0, "bodies": 2,
             "file": "part_only.stl", "watertight": False, "leftovers_dropped": {"bodies": 3, "volume_mm3": 0.25},
             "notes": ["a note"]}
    assert part_only.log_lines(block) == [
        "part_only: 1 body(ies) cut out (fixed:a.stl); volume 10.0 -> 6.0 mm3, 2 body(ies) in result/part_only.stl, NOT one closed manifold",
        "part_only: dropped 3 loose piece(s) of 0.25 mm3 that hold no material of the field", "part_only: a note"]
    assert part_only.log_lines(None) == [] and part_only.log_lines({}) == []
