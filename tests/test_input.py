"""Tests for the input layer: config, voxelize, loads, problem."""

import time

import numpy as np
import pytest
import trimesh
import yaml

from topopt.config import UserError, load_config
from topopt.problem import build_problem
from topopt.voxelize import DESIGN, EMPTY, FIXED, KEEP_OUT, LOAD, voxelize


def box(path, lo, hi):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def make_case(tmp_path, h=1.0, extra_regions=None, mutate=None):
    """Cantilever: design x 0..20, fixed plate x -2..0, load box at the tip."""
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", (18, 4, 0), (20, 6, 3))
    d = {
        "name": "cantilever",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
        "load_cases": [
            {"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]},
            {"name": "side", "weight": 2.0, "forces": [{"region": "tip", "force": [1.0, 3.0, 0]}]},
        ],
        "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 10},
    }
    if extra_regions:
        d["regions"].update(extra_regions)
    if mutate:
        mutate(d)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


# --- config --------------------------------------------------------------


def test_defaults(tmp_path):
    cfg = load_config(make_case(tmp_path, h=2.0))
    assert cfg.settings.filter_radius == 3.0
    assert cfg.settings.penal == 3.0
    assert cfg.settings.beta_schedule[-1] == (65, 8.0)
    assert cfg.material.E == 3500.0
    assert cfg.load_cases[0].weight == 1.0
    assert cfg.regions.design.is_absolute() and cfg.regions.design.is_file()
    with pytest.raises(Exception):
        cfg.name = "x"  # frozen


@pytest.mark.parametrize("mutate,text", [
    (lambda d: d.update(bogus=1), "unknown key 'bogus'"),
    (lambda d: d["settings"].update(speed=1), "unknown key 'speed'"),
    (lambda d: d["settings"].pop("iterations"), "missing key 'iterations'"),
    (lambda d: d.pop("load_cases"), "missing key 'load_cases'"),
    (lambda d: d["regions"].update(design="nope.stl"), "nope.stl"),
    (lambda d: d["load_cases"][0]["forces"][0].update(region="zzz"), "'zzz'"),
    (lambda d: d["load_cases"][1].update(name="down"), "duplicate load case name 'down'"),
    (lambda d: d["settings"].update(voxel_size="big"), "voxel_size"),
    (lambda d: d["settings"].update(volume_fraction=1.5), "volume_fraction"),
    (lambda d: d["settings"].update(iterations=2.5), "iterations"),
    (lambda d: d["settings"].update(iterations=True), "iterations"),
    (lambda d: d["load_cases"][0]["forces"][0].update(force=[1, 2]), "3 numbers"),
    (lambda d: d["load_cases"][0].update(weight=0), "weight"),
    (lambda d: d["settings"].update(solver="mkl"), "solver"),
    (lambda d: d["regions"].update(fixed="plate.stl"), "list"),
])
def test_yaml_errors(tmp_path, mutate, text):
    p = make_case(tmp_path, mutate=mutate)
    with pytest.raises(UserError) as e:
        load_config(p)
    assert text in str(e.value)
    assert "\n" not in str(e.value)


def test_missing_yaml_file(tmp_path):
    with pytest.raises(UserError):
        load_config(tmp_path / "none.yaml")


# --- voxelize ------------------------------------------------------------


def test_labels_and_grid(tmp_path):
    p = build_problem(load_config(make_case(tmp_path)))
    assert p.shape == (22, 10, 10)
    assert np.allclose(p.origin, [-2, 0, 0])
    assert p.h == 1.0
    lab = p.labels
    assert lab.dtype == np.uint8
    assert (lab[:2] == FIXED).all()                       # plate x -2..0
    assert (lab[2:20] != FIXED).all()
    assert (lab[20:22, 4:6, 0:3] == LOAD).all()           # tip x 18..20 -> i 20..22
    assert (lab == LOAD).sum() == 2 * 2 * 3
    assert (lab == DESIGN).sum() == 20 * 10 * 10 - 12
    assert (lab == KEEP_OUT).sum() == 0 and (lab == EMPTY).sum() == 0
    # load ids: -1 outside load voxels, 0 inside
    assert ((p.load_region_ids == 0) == (lab == LOAD)).all()
    assert p.load_region_ids.dtype == np.int16


def test_fixed_covered_error(tmp_path):
    box(tmp_path / "ko.stl", (-2, 0, 0), (3, 10, 10))
    cfg = load_config(make_case(tmp_path, extra_regions={"keep_out": ["ko.stl"]}))
    with pytest.raises(UserError) as e:
        build_problem(cfg)
    assert "plate.stl" in str(e.value)


def test_priority_keep_out_partial(tmp_path):
    box(tmp_path / "ko.stl", (-1, 0, 0), (3, 10, 10))
    p = build_problem(load_config(make_case(tmp_path, extra_regions={"keep_out": ["ko.stl"]})))
    assert (p.labels[0:1] == FIXED).all()                  # x -2..-1 stays fixed
    assert (p.labels[1:5] == KEEP_OUT).all()               # keep_out beats fixed and design


def test_priority_fixed_over_load_over_design(tmp_path):
    # Fixed plate overlaps the load box in one voxel column.
    box(tmp_path / "plate2.stl", (19, 4, 0), (20, 6, 3))
    cfg = load_config(make_case(
        tmp_path, extra_regions={"fixed": ["plate.stl", "plate2.stl"]}))
    p = build_problem(cfg)
    assert (p.labels[21:22, 4:6, 0:3] == FIXED).all()      # fixed wins over load
    assert (p.labels[20:21, 4:6, 0:3] == LOAD).all()
    assert (p.labels == LOAD).sum() == 6
    # Load nodes that are fixed nodes keep their load entry.
    assert np.isclose(p.forces[0].reshape(-1, 3).sum(axis=0)[2], -2.0)


def test_sphere_volume_and_parity(tmp_path):
    ball = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    ball.export(tmp_path / "ball.stl")
    box(tmp_path / "plate.stl", (-12, -2, -2), (-10, 2, 2))
    box(tmp_path / "tip.stl", (9, -1, -1), (10, 1, 1))
    p = make_case(tmp_path, mutate=lambda d: d["regions"].update(design="ball.stl"))
    cfg = load_config(p)
    origin, h, shape, labels, ids = voxelize(cfg)
    n = int((labels == DESIGN).sum() + (labels == LOAD).sum() + (labels == FIXED).sum())
    sphere_vol = ball.volume
    assert abs(n - sphere_vol) / sphere_vol < 0.06 + 0.02  # tolerates the plate, tip voxels


def test_an_open_stl_says_what_is_open_and_how_to_close_it(tmp_path):
    """Q1-16: the error counts the open edges and names tools that close the file (the repo has none of its own)."""
    from topopt.voxelize import load_mesh
    m = trimesh.creation.box(extents=(5, 5, 5))
    m.faces = m.faces[:-2]                                  # two triangles off: one square hole of 4 edges
    m.export(tmp_path / "bad.stl")
    with pytest.raises(UserError) as e:
        load_mesh(tmp_path / "bad.stl")
    text = str(e.value)
    assert text.startswith(f"STL is not watertight (not closed): {tmp_path / 'bad.stl'}. It has 4 open edge(s)")
    assert "0 non-manifold edge(s)" in text
    assert "MeshLab" in text and "Blender" in text and "trimesh.repair.fill_holes" in text and "\n" not in text
    assert "a hole of 3 or 4 edges" in text and "fill_holes(mesh, use_fan=True)" in text and "a larger convex hole" in text    # Q1 R1 n2
    assert "closes small holes" not in text
    assert trimesh.repair.fill_holes(trimesh.load(str(tmp_path / "bad.stl")))         # the named Python route does close this hole


def test_the_fill_holes_text_is_true_for_the_pinned_trimesh():
    """Q1 R1 n2: the default of `fill_holes` closes a hole of 3 or 4 edges; `use_fan=True` closes a larger convex one (trimesh 5.1.0)."""
    def sphere_with_hole(n_faces):                          # the faces around one vertex are removed: a hole of n_faces edges
        m = trimesh.creation.icosphere(subdivisions=2)
        around = np.where((m.faces == 0).any(axis=1))[0]
        assert len(around) == 5
        keep = np.ones(len(m.faces), dtype=bool)
        keep[around[:n_faces]] = False
        return trimesh.Trimesh(m.vertices.copy(), m.faces[keep], process=False)
    assert trimesh.repair.fill_holes(sphere_with_hole(2)) and trimesh.repair.fill_holes(sphere_with_hole(1))     # holes of 4 and 3 edges
    closed = sphere_with_hole(5)                            # a hole of 5 edges: the default leaves it open
    assert not trimesh.repair.fill_holes(closed) and not closed.is_watertight
    assert trimesh.repair.fill_holes(closed, use_fan=True) and closed.is_watertight


def test_not_watertight(tmp_path):
    m = trimesh.creation.box(extents=(5, 5, 5))
    m.faces = m.faces[:-2]                                  # open the box
    m.export(tmp_path / "bad.stl")
    cfg = load_config(make_case(tmp_path, mutate=lambda d: d["regions"].update(design="bad.stl")))
    with pytest.raises(UserError) as e:
        build_problem(cfg)
    assert "bad.stl" in str(e.value) and "watertight" in str(e.value)


def test_guard_grid_voxels(tmp_path):
    def m(d):
        d["settings"]["max_grid_voxels"] = 1000
    cfg = load_config(make_case(tmp_path, mutate=m))
    with pytest.raises(UserError) as e:
        build_problem(cfg)
    msg = str(e.value)
    assert "2200" in msg and "1000" in msg and "voxel_size" in msg


def test_guard_active_voxels(tmp_path):
    def m(d):
        d["settings"]["max_active_voxels"] = 500
    cfg = load_config(make_case(tmp_path, mutate=m))
    with pytest.raises(UserError) as e:
        build_problem(cfg)
    assert "active voxels" in str(e.value) and "500" in str(e.value)


def test_guard_before_alloc(tmp_path, monkeypatch):
    # The grid guard must fire before any mask is made.
    import topopt.voxelize as vx
    called = []
    monkeypatch.setattr(vx, "inside_mask", lambda *a, **k: called.append(1))
    cfg = load_config(make_case(tmp_path, mutate=lambda d: d["settings"].update(max_grid_voxels=10)))
    with pytest.raises(UserError):
        build_problem(cfg)
    assert not called


def test_empty_region(tmp_path):
    box(tmp_path / "tiny.stl", (5.1, 5.1, 5.1), (5.3, 5.3, 5.3))   # no voxel centre inside
    cfg = load_config(make_case(
        tmp_path, extra_regions={"load": {"tip": "tiny.stl"}}))
    with pytest.raises(UserError) as e:
        build_problem(cfg)
    assert "tip" in str(e.value)


# --- loads and indices ---------------------------------------------------


def test_node_dof_formulas_and_forces(tmp_path):
    p = build_problem(load_config(make_case(tmp_path)))
    nx, ny, nz = p.shape
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    assert p.forces.shape == (2, 3 * n_nodes)
    assert p.case_names == ["down", "side"]
    assert np.allclose(p.case_weights, [1.0, 2.0])

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    # Load nodes: voxels i 20..21, j 4..5, k 0..2 -> nodes i 20..22, j 4..6, k 0..3
    load_nodes = [nid(i, j, k) for i in range(20, 23) for j in range(4, 7) for k in range(0, 4)]
    assert len(load_nodes) == 36
    for c, f in enumerate([(0, 0, -2.0), (1.0, 3.0, 0.0)]):
        F = p.forces[c].reshape(-1, 3)
        assert np.allclose(F.sum(axis=0), f, rtol=1e-12, atol=1e-15)
        nz_nodes = np.flatnonzero(np.abs(F).sum(axis=1) > 0) if any(f) else []
        for d in range(3):
            if f[d] != 0:
                idx = np.flatnonzero(p.forces[c, d::3])
                assert sorted(idx) == sorted(load_nodes)
                assert np.allclose(p.forces[c, [3 * n + d for n in load_nodes]], f[d] / 36)
    # Fixed dofs: plate voxels i 0..1 -> nodes i 0..2, all j 0..10, k 0..10
    expect = sorted(3 * nid(i, j, k) + d for i in range(3) for j in range(11)
                    for k in range(11) for d in range(3))
    assert p.fixed_dofs.dtype == np.int64
    assert p.fixed_dofs.tolist() == expect
    assert np.all(np.diff(p.fixed_dofs) > 0)


def test_timing_2m_voxels(tmp_path):
    h = 1.0
    box(tmp_path / "design.stl", (0, 0, 0), (200, 100, 100))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 100, 100))
    box(tmp_path / "tip.stl", (198, 40, 0), (200, 60, 10))
    ball = trimesh.creation.icosphere(subdivisions=4, radius=15.0)
    ball.apply_translation((100, 50, 50))
    ball.export(tmp_path / "ko.stl")
    d = {
        "name": "big",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"], "keep_out": ["ko.stl"],
                    "load": {"tip": "tip.stl"}},
        "load_cases": [{"name": "a", "forces": [{"region": "tip", "force": [0, 0, -1]}]}],
        "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 10,
                     "max_active_voxels": 5000000},
    }
    (tmp_path / "big.yaml").write_text(yaml.safe_dump(d))
    cfg = load_config(tmp_path / "big.yaml")
    t0 = time.perf_counter()
    prob = build_problem(cfg)
    dt = time.perf_counter() - t0
    n = int(np.prod(prob.shape))
    print(f"\nVOXELIZE TIMING: {n} voxels in grid, build_problem {dt:.2f} s")
    assert n > 2_000_000
    assert (prob.labels == KEEP_OUT).sum() > 10000
    assert dt < 15.0   # loose limit; the target is under 5 s


def test_the_settings_section_of_the_hook_readme_gives_the_values_of_the_shipped_yaml():
    """Q1-11: voxel size, active voxels and the design-space volume in examples/hook/README.md are those of examples/hook/hook.yaml."""
    from pathlib import Path
    from topopt.cli import check_data
    hook = Path(__file__).resolve().parent.parent / "examples" / "hook"
    cfg = load_config(hook / "hook.yaml")
    text = (hook / "README.md").read_text()
    settings = text[text.index("## Settings"):text.index("## Active voxel counts")]
    assert f"`voxel_size`: {cfg.settings.voxel_size:g} mm" in settings
    assert f"{check_data(hook / 'hook.yaml')['active_voxels']:,} active voxels" in settings
    design = trimesh.load(str(hook / "design.stl"))
    assert f"{design.volume / 1000:.1f} cm3" in settings
