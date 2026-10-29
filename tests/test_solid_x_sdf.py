"""A3a x C1/C2 merge gate (patch `a3a-x-c1c2`): a solid region (`regions.solid`, CONTRACTS.md section 22a) is a frozen body
of the mesh step, as the same STL in `regions.load` (the old form) is.

The mesh step takes the frozen shape from the files in `run.json` `region_files`. A solid file that is not listed there makes the
SDF step (CONTRACTS.md section 19) and the exact interface step (section 23, `_drop_floating`) treat the solid body as a free island:
they cut it back to the voxels or remove it. The old form is the reference: the run of a solid box as `regions.load` with no force
has the same density field (the bit-equal snapshots of the A3a tests), so the two final meshes must be the same file.
"""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from test_optimize import box
from test_solid_filter import CAP, write_case
from topopt import interfaces as I
from topopt import post
from topopt.cli import main

OFF = ((8.4, 0.6, 0.6), (12.6, 9.4, 4.6))      # off the voxel faces (h 2): the exact shape differs from the voxels
ISLAND = ((15.4, 6.6, 6.6), (19.6, 9.4, 9.6))  # off the voxel faces, and no fixed or load region touches it
BORE = {"name": "hole", "kind": "bore", "point": [4, 5, 12], "axis": [0, 0, -1], "radius_mm": 1.0, "length_mm": 14}


def _run(tmp_path, name, solids, as_load, interfaces=None, **settings):
    d = tmp_path / name
    d.mkdir()
    p = write_case(d, solids=solids, as_load=as_load, iterations=6, **settings)
    if interfaces:
        c = yaml.safe_load(p.read_text())
        c["interfaces"] = interfaces
        p.write_text(yaml.safe_dump(c, sort_keys=False))
    out = d / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    return out


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_solid_files_are_in_run_json_and_in_the_region_paths(tmp_path):
    out = _run(tmp_path, "a", [OFF], as_load=False)
    run = json.loads((out / "run.json").read_text())
    assert [Path(q).name for q in run["region_files"]["solid"]] == ["solid0.stl"]
    fixed, load, _keep = post.region_paths(run)
    assert list(load) == ["tip", "solid:solid0.stl"] and Path(load["solid:solid0.stl"]).name == "solid0.stl"   # after the loads
    assert list(fixed) == ["plate.stl"]


def test_a_run_without_solid_has_the_same_region_paths(tmp_path):
    out = _run(tmp_path, "a", [], as_load=False)
    run = json.loads((out / "run.json").read_text())
    assert run["region_files"]["solid"] == []
    fixed, load, keep = post.region_paths(run)
    assert list(fixed) == ["plate.stl"] and list(load) == ["tip"] and keep == {}


@pytest.mark.parametrize("method", ["sdf", "blur"])
def test_the_exact_solid_shape_is_in_the_final_stl_like_the_old_form(tmp_path, method):
    a = _run(tmp_path, "solid", [OFF], as_load=False, mesh_method=method)
    b = _run(tmp_path, "old", [OFF], as_load=True, mesh_method=method)
    ma = trimesh.load(a / "result" / "final.stl")
    mb = trimesh.load(b / "result" / "final.stl")
    assert _sha(a / "result" / "final.stl") == _sha(b / "result" / "final.stl")
    assert ma.volume == pytest.approx(mb.volume, rel=1e-9)
    if method == "sdf":
        box_mm3 = (12.6 - 8.4) * (9.4 - 0.6) * (4.6 - 0.6)         # about 147.8: the solid box, kept as the exact shape
        ca = json.loads((a / "result" / "metrics.json").read_text())["mesh"]["checks"]
        assert ca["fixed"]["load:solid:solid0.stl"]["region_mm3"] == pytest.approx(box_mm3, rel=1e-6)
        assert ca["fixed"]["load:solid:solid0.stl"]["uncovered_mm3"] < 1e-3


def test_a_solid_island_survives_the_sdf_step_and_the_interface_step(tmp_path, caplog):
    """The island touches no fixed and no load region. The bore makes the exact interface step run `_drop_floating`.

    Since C2 round 1 the step judges only the pieces that its booleans split off, so the interface part of this test also passes
    without the patch (the bore does not split the island). The SDF part needs the patch. The split test below needs it for the
    interface step too.
    """
    with caplog.at_level(logging.WARNING):
        a = _run(tmp_path, "solid", [ISLAND], as_load=False, interfaces=[BORE], filter_solid_neighbours="ignore")
        b = _run(tmp_path, "old", [ISLAND], as_load=True, interfaces=[BORE], filter_solid_neighbours="ignore")
    ma = trimesh.load(a / "result" / "final.stl")
    assert ma.contains([[17.5, 8.0, 8.1]])[0]                                  # the centre of the island
    assert len(ma.split(only_watertight=False)) == 3                           # plate body, tip body, solid island
    assert _sha(a / "result" / "final.stl") == _sha(b / "result" / "final.stl")
    assert not any("interface step removed" in r.message for r in caplog.records)


@pytest.mark.parametrize("method", ["sdf", "blur"])
def test_a_cut_that_splits_a_solid_body_keeps_both_pieces(tmp_path, caplog, method):
    """The exact interface step judges the pieces that its booleans split off (`_drop_floating`). A slab cut through the solid
    island makes two pieces. Both touch the solid file, so both stay. Without the solid file in `frozen_paths` the step
    would remove the smaller piece (no fixed or load file touches it)."""
    d = tmp_path / "split"
    d.mkdir()
    p = write_case(d, solids=[ISLAND], iterations=6, filter_solid_neighbours="ignore", mesh_method=method)
    box(d / "slab.stl", (17.3, -1.0, 4.0), (17.7, 11.0, 12.0))
    c = yaml.safe_load(p.read_text())
    c["interfaces"] = [{"name": "slab", "kind": "stl", "op": "cut", "file": "slab.stl"}]
    p.write_text(yaml.safe_dump(c, sort_keys=False))
    with caplog.at_level(logging.WARNING):
        assert main(["run", str(p), "--out", str(d / "out"), "--quiet"]) == 0
    mesh = trimesh.load(d / "out" / "result" / "final.stl")
    inside = mesh.contains([[16.4, 8.0, 8.1], [18.6, 8.0, 8.1], [17.5, 8.0, 8.1]])
    assert list(inside) == [True, True, False]                                 # both pieces, and the slab is empty
    assert not any("interface step removed" in r.message for r in caplog.records)


def test_from_run_lists_the_solid_files_as_frozen_paths(tmp_path):
    out = _run(tmp_path, "solid", [ISLAND], as_load=False, interfaces=[BORE], filter_solid_neighbours="ignore")
    run = json.loads((out / "run.json").read_text())
    ctx = I.from_run(run, (0, 0, 0), 2.0, (10, 5, 5))
    assert sorted(Path(q).name for q in ctx.frozen_paths) == ["plate.stl", "solid0.stl", "tip.stl"]


def test_solid_facts_count_the_voxels_that_an_interface_cut_removes(tmp_path):
    """Minor 10 of the review: after the merge a cut interface is a KEEP_OUT label. The solid facts add up."""
    from topopt.config import load_config
    from topopt.problem import build_problem

    d = tmp_path / "c"
    d.mkdir()
    p = write_case(d, solids=[CAP])
    c = yaml.safe_load(p.read_text())
    c["interfaces"] = [{"name": "hole", "kind": "bore", "point": [10, 5, 12], "axis": [0, 0, -1], "radius_mm": 2.5, "length_mm": 14}]
    p.write_text(yaml.safe_dump(c, sort_keys=False))
    f = build_problem(load_config(p)).region_facts["solid"]
    assert f["in_keep_out"] > 0                                              # the bore cuts the solid cap
    assert f["stl_voxels"] == f["voxels"] + f["in_fixed"] + f["in_load"] + f["in_keep_out"]
