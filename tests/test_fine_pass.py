"""Tests of `topopt/fine_pass.py` (#318, CONTRACTS.md section 24.6): the voxel of the part, the settings of a fine pass, and the mapping.

The runs of the CLI are in tests/test_fine_cli.py. Here no run starts. The measure of the thickness of a region (`inscribed_radius`) works on the
mesh, so its cost depends on the number of faces and not on the size of the region: a box takes 0.03 s, a rod of 64 sections 0.1 s, a sphere of
1280 faces 1.2 s (measured 2026-10-26, one thread, CPU time). The test of the voxel on the grid of the fine pass (`check_voxel`) voxelizes the
regions at the fine voxel, over their own boxes only.
"""
from pathlib import Path

import dataclasses
import math
import time

import numpy as np
import pytest
import trimesh
import yaml

from topopt import fine_pass as F
from topopt.config import UserError, load_config
from topopt.optimize import Model, beta_at
from topopt.problem import build_problem
from topopt.voxelize import DESIGN, FIXED, LOAD, load_mesh

from test_optimize import box, make_case

BIG = 1_000_000   # voxels of a run grid for the check of the guards


def case_config(tmp_path, h=2.0, edit=None, iterations=10, sub=None):
    """The cantilever of tests/test_optimize.py (plate 2 mm thick, tip load 2 x 2 x 3 mm) with `edit(dict)` applied to the YAML.

    `sub` is the name of a folder of `tmp_path` for the files (a test that makes more than one case).
    """
    if sub is not None:
        tmp_path = tmp_path / sub
        tmp_path.mkdir()
    path = make_case(tmp_path, h=h, iterations=iterations)
    if edit is not None:
        data = yaml.safe_load(path.read_text())
        edit(data)
        path.write_text(yaml.safe_dump(data))
    return load_config(path)


# --- the voxel of the part ----------------------------------------------------------------------------

def test_round_down_to_step():
    assert F.round_down_to_step(0.69) == pytest.approx(0.65)
    assert F.round_down_to_step(0.7) == pytest.approx(0.7)       # a value that is a multiple of the step up to the float error stays
    assert F.round_down_to_step(0.7000000001) == pytest.approx(0.7)
    assert F.round_down_to_step(0.01) == F.STEP_MM               # never below one step


def slab(thickness, tilt_deg=0.0, width=30.0):
    mesh = trimesh.creation.box(extents=(width, width, thickness))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(tilt_deg), (1, 0, 0)))
    return mesh


def convex_inscribed_diameter(mesh):
    """The exact diameter of the largest sphere inside a CONVEX mesh: a linear program over the planes of its faces (the reference of the tests)."""
    from scipy.optimize import linprog

    normals = np.asarray(mesh.face_normals, float)
    offsets = np.einsum("ij,ij->i", normals, mesh.triangles[:, 0, :])
    res = linprog([0, 0, 0, -1.0], A_ub=np.hstack([normals, np.ones((len(normals), 1))]), b_ub=offsets, bounds=[(None, None)] * 3 + [(0, None)])
    return 2.0 * float(res.x[3])


# The tolerance of the measure, as measured on 2026-10-26 (see the comment above `THICK_SAMPLES`): it is never above the exact value (the sphere that it
# finds lies inside), and below it by 0.00 % for boxes and slabs (tilted too), 0.01 % for a rod of 64 sections, 0.03 % for the wall of a tube, 0.12 % for a
# sphere of 5120 faces, 0.34 % for one of 1280 faces and 0.55 % for a cone. The tests allow a little more than the measured value.
@pytest.mark.parametrize("thickness", [1.0, 2.0, 3.5])
@pytest.mark.parametrize("tilt", [0.0, 20.0, 45.0])
def test_the_thickness_of_a_slab_is_exact_also_when_the_slab_is_tilted(thickness, tilt):
    t = F.thickness_mm(slab(thickness, tilt))
    assert thickness * (1 - 1e-4) <= t <= thickness * (1 + 1e-9)


@pytest.mark.parametrize("diameter", [1.0, 2.0, 3.5])
def test_the_thickness_of_a_rod_is_its_inscribed_diameter_also_when_the_rod_is_tilted(diameter):
    for tilt in (0.0, 45.0):
        rod = trimesh.creation.cylinder(radius=diameter / 2, height=30, sections=64)
        rod.apply_transform(trimesh.transformations.rotation_matrix(np.radians(tilt), (1, 1, 0)))
        exact = convex_inscribed_diameter(rod)
        assert exact == pytest.approx(diameter * math.cos(math.pi / 64), rel=1e-9)
        t = F.thickness_mm(rod)
        assert exact * (1 - 1e-3) <= t <= exact * (1 + 1e-9)       # measured 0.01 % low


@pytest.mark.parametrize("wall", [1.0, 3.5])
def test_the_thickness_of_the_wall_of_a_tube(wall):
    tube = trimesh.creation.annulus(r_min=10.0, r_max=10.0 + wall, height=20, sections=96)      # not convex: the sphere is in the wall
    t = F.thickness_mm(tube)
    assert wall * (1 - 1e-3) <= t <= wall * (1 + 1e-9)          # measured 0.03 % low


@pytest.mark.parametrize("make, rel", [
    (lambda: trimesh.creation.icosphere(subdivisions=3, radius=1.75), 0.006),          # 1280 faces, measured 0.34 % low
    (lambda: trimesh.creation.cone(radius=3.0, height=30, sections=64), 0.008)])      # measured 0.55 % low
def test_the_thickness_of_a_curved_convex_body_is_within_the_measured_tolerance_and_never_above_the_exact_value(make, rel):
    mesh = make()
    exact = convex_inscribed_diameter(mesh)
    t = F.thickness_mm(mesh)
    assert exact * (1 - rel) <= t <= exact * (1 + 1e-9)


def test_a_region_of_two_bodies_gives_the_thickest_body():
    plate = trimesh.creation.box(extents=(200, 200, 2))
    ball = trimesh.creation.icosphere(subdivisions=3, radius=2.5)
    ball.apply_translation((0, 0, 20))
    both = trimesh.util.concatenate([plate, ball])
    exact = convex_inscribed_diameter(ball)
    assert exact * 0.995 <= F.thickness_mm(both) <= exact * (1 + 1e-9)          # the ball (D 5) is thicker than the plate (2); measured 0.34 % low
    assert F.thickness_mm(both) == F.thickness_mm(ball)       # the body keeps the face order of the mesh: 2.2 % low on macOS arm64 with `split`
    small = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    small.apply_translation((0, 0, 20))
    assert F.thickness_mm(trimesh.util.concatenate([plate, small])) == pytest.approx(2.0, rel=1e-4)    # the plate is thicker than the small ball (D 1)


def test_the_measure_costs_the_same_for_a_small_plate_and_a_large_one():
    """The 200 x 200 x 2 mm plate (41 million voxels at 16 across, the case of the review) costs 0.02 s: the cost depends on the faces, not on the size."""
    started = time.process_time()
    large = F.thickness_mm(trimesh.creation.box(extents=(200, 200, 2)))
    cost = time.process_time() - started
    assert large == pytest.approx(2.0, rel=1e-9) and F.thickness_mm(trimesh.creation.box(extents=(20, 20, 2))) == pytest.approx(2.0, rel=1e-9)
    assert cost < 5.0       # seconds, bounded (measured 0.02 s)


def test_a_flat_region_has_no_thickness(tmp_path):
    box(tmp_path / "flat.stl", (0, 0, 0), (5, 5, 1e-9))
    try:
        t = F.thickness_mm(load_mesh(tmp_path / "flat.stl"))
    except Exception:             # the loader or the measure may refuse the flat mesh
        return
    assert t < 1e-6


def test_the_region_rule_is_the_rule_of_the_region_check():
    from topopt import region_rules
    assert F.region_across_needed() == region_rules.REGION_ACROSS == pytest.approx(2.8)


def test_the_band_and_the_two_levels_of_the_old_measure_are_gone(tmp_path):
    for name in ("PROBE_ACROSS", "part_voxel_levels", "thickness_band", "band_wide", "band_text", "_probe_count"):
        assert not hasattr(F, name), name
    res = F.part_voxel(case_config(tmp_path, h=2.0))
    assert not {"voxel_high_mm", "level", "probe_voxels_max", "band_wide", "levels"} & set(res)
    for f in res["features"]:
        assert not {"probe_mm", "thickness_low_mm", "thickness_high_mm"} & set(f)
    assert not {"voxel_high_mm", "level", "probe_voxels_max"} & set(F.part_voxel_block(case_config(tmp_path, h=2.0, sub="b")))


def plate_config(tmp_path, max_grid_voxels=50_000_000):
    """A 200 x 200 x 2 mm fixed plate under a design space: 41 million voxels at 16 across. The run is at 2 mm (the grid is 100 x 100 x 30)."""
    box(tmp_path / "design.stl", (0, 0, 2), (200, 200, 60))
    box(tmp_path / "plate.stl", (0, 0, 0), (200, 200, 2))
    box(tmp_path / "pad.stl", (90, 90, 58), (110, 110, 60))
    d = {"name": "plate", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"pad": "pad.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "pad", "force": [0, 0, -10.0]}]}],
         "settings": {"voxel_size": 2.0, "volume_fraction": 0.2, "iterations": 5, "max_grid_voxels": max_grid_voxels}}
    (tmp_path / "plate.yaml").write_text(yaml.safe_dump(d))
    return load_config(tmp_path / "plate.yaml")


def test_a_thin_and_wide_plate_costs_seconds_for_the_measure_and_for_the_test_on_the_fine_grid(tmp_path):
    cfg = plate_config(tmp_path)
    started = time.time()
    measured = F.part_voxel(cfg)
    verified = F.part_voxel(cfg, verify=True)               # the test voxelizes the plate on the grid of the fine pass: 285 x 285 x 3 voxels
    assert time.time() - started < 15.0                     # seconds, not minutes (measured about 1 s)
    plate = next(f for f in measured["features"] if f["name"] == "plate")
    assert plate["thickness_mm"] == pytest.approx(2.0, abs=1e-6) and measured["voxel_mm"] == pytest.approx(0.7)
    assert measured["checked"] is None and verified["checked"]["ok"] and not verified["checked"]["skipped"]
    assert verified["voxel_mm"] <= measured["voxel_mm"] + 1e-9


def test_the_test_on_the_fine_grid_is_skipped_for_a_grid_that_the_run_refuses(tmp_path):
    cfg = plate_config(tmp_path, max_grid_voxels=100_000)    # the box of the plate at 0.7 mm is 286 x 286 x 3 = 245,000 voxels
    res = F.part_voxel(cfg, verify=True)
    assert res["checked"]["skipped"] is True and res["checked"]["steps"] == 0 and res["voxel_mm"] == pytest.approx(0.7)
    assert "skipped" in F.check_fact(res) and "settings.max_grid_voxels" in F.check_fact(res)


def test_the_voxel_of_the_part_is_set_by_the_thinnest_region(tmp_path):
    res = F.part_voxel(case_config(tmp_path, h=2.0))
    lim = res["limiting"]
    assert lim["role"] in ("fixed", "load") and lim["thickness_mm"] == pytest.approx(2.0, abs=1e-6)
    assert res["voxel_mm"] == pytest.approx(0.7) and lim["across_needed"] == pytest.approx(2.8)      # 2.0 / 2.8 = 0.714, rounded down to 0.05
    assert lim["across_at_run"] == pytest.approx(1.0, abs=0.01)
    assert {f["role"] for f in res["features"]} == {"fixed", "load"}
    assert res["checked"] is None and res["run_voxel_mm"] == 2.0


def test_min_member_mm_sets_the_voxel_with_three_voxels_across(tmp_path):
    res = F.part_voxel(case_config(tmp_path, h=2.0, edit=lambda d: d["settings"].update(min_member_mm=1.5)))
    lim = res["limiting"]
    assert (lim["role"], lim["across_needed"]) == ("min_member_mm", 3.0)
    assert res["voxel_mm"] == pytest.approx(0.5) and lim["voxel_mm"] == pytest.approx(0.5)


def test_a_thin_keep_out_region_sets_the_voxel(tmp_path):
    def edit(d):
        box(tmp_path / "ko.stl", (8, 3, 3), (12, 4.4, 7))          # 1.4 mm thick, inside the design space
        d["regions"]["keep_out"] = ["ko.stl"]
    res = F.part_voxel(case_config(tmp_path, h=2.0, edit=edit))
    assert res["limiting"]["role"] == "keep_out" and res["voxel_mm"] == pytest.approx(0.5)      # 1.4 / 2.8 = 0.5


def test_recommend_is_true_when_the_voxel_is_below_the_run_voxel_by_a_step_or_more(tmp_path):
    assert F.part_voxel(case_config(tmp_path, h=2.0, sub="a"))["recommend"] is True
    equal = F.part_voxel(case_config(tmp_path, h=0.7, sub="b"))           # 0.7 is the voxel of the part: no recommendation
    assert equal["recommend"] is False and equal["voxel_mm"] == pytest.approx(0.7)
    assert F.part_voxel(case_config(tmp_path, h=0.72, sub="c"))["recommend"] is False      # below by 0.02 mm: less than a step
    assert F.part_voxel(case_config(tmp_path, h=0.75, sub="d"))["recommend"] is True       # below by one step
    assert F.part_voxel(case_config(tmp_path, h=0.5, sub="e"))["recommend"] is False       # the run voxel is finer than the part needs


def test_hook_short_at_2_mm_gives_1_25_mm_which_is_2_8_voxels_across_the_wall_plate():
    """The wall plate of the hook is exactly 3.5 mm thick, and 3.5 / 2.8 = 1.25 mm. (The two-level measure of the review gave 1.2 mm, the lower end of a band;
    the exact measure gives the rule with equality.) The test on the grid of the fine pass finds 3 rows in the plate there, so no step down."""
    cfg = load_config(Path(__file__).resolve().parent.parent / "examples" / "hook" / "hook_short.yaml")
    cfg = dataclasses.replace(cfg, settings=dataclasses.replace(cfg.settings, voxel_size=2.0))
    res = F.part_voxel(cfg)
    wall = next(f for f in res["features"] if f["name"] == "wall_plate")
    assert wall["thickness_mm"] == pytest.approx(3.5, abs=1e-6) and res["limiting"]["name"] == "wall_plate"
    assert res["voxel_mm"] == pytest.approx(1.25) and res["recommend"] is True
    assert wall["thickness_mm"] / res["voxel_mm"] == pytest.approx(F.region_across_needed())
    assert F.part_voxel(cfg, verify=True)["checked"] == {"voxel_mm": 1.25, "start_mm": 1.25, "steps": 0, "region": None, "ok": True, "skipped": False}


def thin_slab_config(tmp_path):
    """The cantilever with a load region that is a slab 1.97 mm thick: 1.97 / 2.8 = 0.7036, which the rounding puts at 0.7 mm. The slab sits 0.6 voxel
    above a voxel boundary of the grid at 0.7 mm, so the grid holds only 2 rows of it: `thick` is 1.0 and the region check fails (it needs 1.9)."""
    cfg = case_config(tmp_path, h=2.0)
    origin = F.grid_origin(cfg, [load_mesh(path) for path in cfg.regions.fixed], 0.7)      # the fixed plate (x from -2) sets the corner of the grid
    lo = origin[0] + 0.7 * 4 + 0.6 * 0.7
    box(tmp_path / "slab.stl", (lo, 2, 2), (lo + 1.97, 7, 7))
    data = yaml.safe_load(cfg.yaml_path.read_text())
    data["regions"]["load"] = {"tip": "slab.stl"}
    cfg.yaml_path.write_text(yaml.safe_dump(data))
    return load_config(cfg.yaml_path), load_mesh(tmp_path / "slab.stl"), origin


def test_the_voxel_goes_down_a_step_when_the_grid_of_the_fine_pass_holds_too_few_voxels_across_a_region(tmp_path):
    cfg, mesh, origin = thin_slab_config(tmp_path)
    assert F.region_thick(mesh, origin, 0.7, 10**9) < 1.9 <= F.region_thick(mesh, origin, 0.65, 10**9)    # the case: 0.7 fails, 0.65 passes
    measured = F.part_voxel(cfg)
    assert measured["voxel_mm"] == pytest.approx(0.7) and measured["limiting"]["name"] == "tip"
    verified = F.part_voxel(cfg, verify=True)
    assert verified["voxel_mm"] == pytest.approx(0.65) and verified["recommend"] is True
    assert verified["checked"] == {"voxel_mm": 0.65, "start_mm": 0.7, "steps": 1, "region": "tip", "ok": True, "skipped": False}
    text = F.check_fact(verified)
    assert "needs a voxel of 0.7 mm" in text and "region 'tip' has too few voxels across" in text and "down 1 step of 0.05 mm to 0.65 mm" in text


def test_the_test_on_the_grid_stops_after_a_few_steps_and_says_so(tmp_path):
    cfg, mesh, origin = thin_slab_config(tmp_path)
    flat = trimesh.creation.box(extents=(5.0, 5.0, 0.04))            # thinner than a voxel at every step: no voxel centre inside, thick is 0 at all voxels
    flat.apply_translation((10.0, 5.0, 5.0))
    res = F.check_voxel(cfg, [("load", "flat", flat)], 0.7, ("load", "flat"))
    assert res["ok"] is False and res["steps"] == F.CHECK_STEPS and res["region"] == "flat"
    assert res["voxel_mm"] == pytest.approx(0.7 - F.CHECK_STEPS * F.STEP_MM)
    assert "did not help" in F.check_fact({"checked": res, "limiting": {"role": "load", "name": "flat", "thickness_mm": 0.04}})


def test_the_block_of_the_part_voxel_has_the_fact_the_estimate_and_never_raises(tmp_path):
    cfg = case_config(tmp_path, h=2.0)
    check = {"grid": {"voxel_mm": 2.0, "voxels": 275, "shape": [11, 5, 5]}, "active_voxels": 275, "free_design_voxels": 249, "free_dofs": 1080,
             "solver": {"resolved": "cholesky"}}
    block = F.part_voxel_block(cfg, check)
    assert block["recommend"] and block["max_passes"] == 10 and block["tolerance"] == F.FINE_OBJECTIVE_CHANGE and block["voxel_mm"] == pytest.approx(0.7)
    assert set(block["limiting"]) == {"role", "name", "thickness_mm", "across_at_run", "across_needed"}
    est = block["estimate"]
    assert est["voxel_ratio"] == pytest.approx((2.0 / 0.7) ** 3, abs=0.01) and est["max_passes"] == 10
    assert est["free_dofs"] == round(1080 * (2.0 / 0.7) ** 3) and est["time_s_max"] > 0 and est["memory_bytes"] > 0
    text = block["fact"]
    assert text.startswith("fine pass: the part needs a voxel of 0.7 mm or less (the ")
    assert "1 voxels across at the voxel of this run, 2.8 needed" in text
    assert "This run uses 2 mm" in text and "Estimate at 0.7 mm: about" in text
    assert "voxel between" not in text and "band" not in text and "probe" not in text and "lower end" not in text
    assert "tests the voxel on its grid and goes down in steps of 0.05 mm" in text
    assert "--fine-from" in text and "--fine-pass" in text and "fine_from" in text
    assert "the loss has settled" in text and "at most 0.001" in text and "at the maximum of passes" in text
    assert "The first run is extra time" in text and "estimates" in text and "faster" not in text      # no claim of a speed-up
    broken = dataclasses.replace(cfg, regions=dataclasses.replace(cfg.regions, fixed=[cfg.yaml_path.parent / "missing.stl"]))
    bad = F.part_voxel_block(broken, check)
    assert bad["recommend"] is False and "error" in bad and "fact" not in bad


def test_no_fact_when_the_run_voxel_is_fine_enough(tmp_path):
    block = F.part_voxel_block(case_config(tmp_path, h=0.7, iterations=3))
    assert block["recommend"] is False and "fact" not in block and "estimate" not in block


# --- settings.fine_pass in the YAML -------------------------------------------------------------------

def fine_pass_config(tmp_path, value="absent"):
    return case_config(tmp_path, edit=None if value == "absent" else (lambda d: d["settings"].update(fine_pass=value)))


def test_the_default_tolerance_is_in_the_config_module_and_the_yaml_key_is_absent_by_default(tmp_path):
    from topopt import config
    assert F.FINE_OBJECTIVE_CHANGE is config.FINE_OBJECTIVE_CHANGE == 1e-3
    assert fine_pass_config(tmp_path).settings.fine_pass == config.FinePassSettings(objective_change=None)
    assert F.tolerance_of(fine_pass_config(tmp_path)) == (1e-3, "default")


def test_the_yaml_gives_the_tolerance_and_the_option_wins(tmp_path):
    cfg = fine_pass_config(tmp_path, {"objective_change": 5e-4})
    assert cfg.settings.fine_pass.objective_change == 5e-4
    assert F.tolerance_of(cfg) == (5e-4, "yaml")
    assert F.tolerance_of(cfg, 2e-3) == (2e-3, "option")                          # --fine-tol wins over the YAML
    assert F.tolerance_of(cfg, "0.01") == (0.01, "option")
    with pytest.raises(UserError, match="settings.fine_pass.objective_change"):    # a wrong option names the YAML key as the other source
        F.tolerance_of(cfg, 0)


def test_a_text_number_and_an_empty_mapping_are_read(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert fine_pass_config(tmp_path, {"objective_change": "5e-4"}).settings.fine_pass.objective_change == 5e-4     # YAML 1.1 loads 5e-4 as text
    assert fine_pass_config(tmp_path / "a", {}).settings.fine_pass.objective_change is None
    assert fine_pass_config(tmp_path / "b", None).settings.fine_pass.objective_change is None


@pytest.mark.parametrize("value, text", [
    (3, "'settings.fine_pass' must be a mapping"), ([1e-3], "'settings.fine_pass' must be a mapping"), ("fast", "'settings.fine_pass' must be a mapping"),
    ({"objective_change": 0}, "'settings.fine_pass.objective_change' must be a finite number above 0"),
    ({"objective_change": -1e-3}, "'settings.fine_pass.objective_change' must be a finite number above 0"),
    ({"objective_change": "abc"}, "'settings.fine_pass.objective_change' must be a finite number above 0"),
    ({"objective_change": True}, "'settings.fine_pass.objective_change' must be a finite number above 0"),
    ({"objective_change": None}, "'settings.fine_pass.objective_change' must be a finite number above 0"),
    ({"objective_change": float("inf")}, "'settings.fine_pass.objective_change' must be a finite number above 0"),
    ({"objective_change": float("nan")}, "'settings.fine_pass.objective_change' must be a finite number above 0"),
    ({"objective_change": [1e-3]}, "'settings.fine_pass.objective_change' must be a finite number above 0"),
    ({"tolerance": 1e-3}, "unknown key 'tolerance' in 'settings.fine_pass'"),
    ({"objective_change": 1e-3, "passes": 4}, "unknown key 'passes' in 'settings.fine_pass'")])
def test_a_wrong_fine_pass_setting_is_refused_with_the_key_path_and_an_example(tmp_path, value, text):
    with pytest.raises(UserError) as err:
        fine_pass_config(tmp_path, value)
    assert text in str(err.value) and "Example: settings: {fine_pass: {objective_change: 0.001}}" in str(err.value)


def test_the_key_is_no_change_of_the_inputs_and_not_in_the_effective_settings(tmp_path):
    from topopt import checkpoint
    (tmp_path / "a").mkdir()
    a, b = fine_pass_config(tmp_path), fine_pass_config(tmp_path / "a", {"objective_change": 1e-2})
    assert checkpoint.effective_settings(a) == checkpoint.effective_settings(b) and "fine_pass" not in checkpoint.effective_settings(b)
    changes, notes = checkpoint.diff_inputs(checkpoint.input_hashes(a), checkpoint.input_hashes(b))
    assert not [c for c in changes if "fine_pass" in c] and any("settings.fine_pass.objective_change" in n for n in notes)


# --- the guards of the fine grid ---------------------------------------------------------------------

CHECK = {"grid": {"voxel_mm": 2.0, "voxels": 275, "shape": [11, 5, 5]}, "active_voxels": 275, "free_design_voxels": 249, "free_dofs": 1080, "solver": {"resolved": "cholesky"}}


def test_the_fine_grid_is_tested_against_the_three_guards_and_the_finest_voxel_that_passes_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr("topopt.cli.physical_ram_bytes", lambda: 64 * 2**30)
    cfg = case_config(tmp_path, h=2.0, edit=lambda d: d["settings"].update(max_active_voxels=2000, max_grid_voxels=5_000))
    res = F.fine_grid_guards(cfg, CHECK, 0.7)
    assert res["ok"] is False and [f["guard"] for f in res["failed"]] == ["settings.max_active_voxels", "settings.max_grid_voxels"]
    assert res["failed"][0]["value"] == round(275 * (2.0 / 0.7) ** 3) and res["failed"][1]["limit"] == 5_000
    v = res["voxel_pass_mm"]
    assert v is not None and 0.7 < v < 2.0 and round(v / F.STEP_MM) * F.STEP_MM == pytest.approx(v)
    assert F.fine_grid_guards(cfg, CHECK, v)["ok"] and not F.fine_grid_guards(cfg, CHECK, v - 2 * F.STEP_MM)["ok"]   # close to the finest one that passes
    assert 275 * (2.0 / v) ** 3 <= 2000 / 1.03 ** 3 * 1.01                               # with the margin of 3 % of the voxel (`suggest_voxel_size`)
    assert f"--fine-voxel {v:g} (MCP fine_voxel) is the finest voxel that passes" in res["text"] and "raise settings.max_active_voxels, settings.max_grid_voxels" in res["text"]
    assert F.fine_grid_guards(cfg, CHECK, v)["text"] == ""


def test_no_voxel_finer_than_the_run_passes_when_the_run_grid_is_at_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("topopt.cli.physical_ram_bytes", lambda: 64 * 2**30)
    cfg = case_config(tmp_path, h=2.0, edit=lambda d: d["settings"].update(max_active_voxels=280))
    res = F.fine_grid_guards(cfg, CHECK, 1.9)
    assert res["ok"] is False and res["voxel_pass_mm"] is None and "No voxel finer than the voxel of this run passes" in res["text"]


def test_the_guards_are_in_the_block_when_the_part_needs_a_finer_voxel(tmp_path, monkeypatch):
    monkeypatch.setattr("topopt.cli.physical_ram_bytes", lambda: 64 * 2**30)
    cfg = case_config(tmp_path, h=2.0, edit=lambda d: d["settings"].update(max_active_voxels=2000, max_grid_voxels=10**9))
    check = dict(CHECK, grid={"voxel_mm": 2.0, "voxels": BIG, "shape": [100, 100, 100]})
    block = F.part_voxel_block(cfg, check)
    assert block["guards"]["ok"] is False and block["guards"]["text"] in block["fact"] and block["fact"].count("fails settings.max_active_voxels") == 1
    (tmp_path / "x").mkdir()
    ok = F.part_voxel_block(case_config(tmp_path / "x", h=2.0, edit=lambda d: d["settings"].update(max_grid_voxels=10**9)), check)
    assert ok["guards"]["ok"] is True and ok["guards"]["failed"] == [] and "fails" not in ok["fact"]


def test_the_error_of_a_fine_grid_names_fine_voxel_and_not_the_voxel_size_of_the_yaml():
    old = ("model has 9000 active voxels, the limit is 2000 (max_active_voxels). Try voxel_size 1.1 or more, or raise settings.max_active_voxels in the "
           "YAML if this machine has the memory")
    new = str(F.name_fine_option(UserError(old)))
    assert "Try --fine-voxel 1.1 or more (MCP fine_voxel), or raise settings.max_active_voxels" in new and "voxel_size" not in new
    grid = "grid would have 9e9 voxels at voxel_size 0.7, the limit is 5000000 (max_grid_voxels). Use a larger voxel_size, or raise settings.max_grid_voxels"
    new = str(F.name_fine_option(UserError(grid)))
    assert "at the fine voxel 0.7 mm" in new and "Use a larger --fine-voxel (MCP fine_voxel)" in new and "voxel_size" not in new
    other = UserError("region design has no voxel at voxel_size 0.3 and the voxel_size is odd")
    assert "not settings.voxel_size" in str(F.name_fine_option(other))
    plain = UserError("a region has no thickness")
    assert F.name_fine_option(plain) is plain


# --- the settings of a fine pass ----------------------------------------------------------------------

def test_the_shifted_schedule_has_the_beta_of_the_pass_in_the_parent_chain():
    sched = ((0, 1.0), (30, 2.0), (50, 4.0), (65, 8.0))
    for offset in (0, 1, 29, 30, 49, 50, 64, 65, 80):
        shifted = F.shift_schedule(sched, offset)
        assert shifted[0][0] == 0
        for j in range(0, 40):
            assert beta_at(shifted, j) == beta_at(sched, offset + j), (offset, j)
    with pytest.raises(ValueError):
        F.shift_schedule(sched, -1)


def test_the_fine_settings_set_voxel_maximum_schedule_and_the_rule(tmp_path):
    cfg = case_config(tmp_path, h=2.0, iterations=10)
    fine = F.with_fine_settings(cfg, 0.7, 25, 10, 5e-4)
    st = fine.settings
    assert (st.voxel_size, st.iterations) == (0.7, 25)
    assert st.filter_radius == pytest.approx(1.05)                               # the default radius is 1.5 voxels and follows the voxel
    assert st.beta_schedule == F.shift_schedule(cfg.settings.beta_schedule, 10)
    assert st.stop_rules.enabled and st.stop_rules.objective_change == 5e-4 and st.stop_rules.window == cfg.settings.stop_rules.window
    assert cfg.settings.voxel_size == 2.0 and not cfg.settings.stop_rules.enabled   # the config of the caller does not change


def test_a_filter_radius_that_the_yaml_gave_stays_the_same(tmp_path):
    cfg = case_config(tmp_path, h=2.0, edit=lambda d: d["settings"].update(filter_radius=3.5))
    assert F.with_fine_settings(cfg, 0.7, 5, 0).settings.filter_radius == 3.5
    cfg = case_config(tmp_path, h=2.0, edit=lambda d: d["settings"].update(min_member_mm=2.0), sub="m")
    radius = cfg.settings.filter_radius
    assert F.with_fine_settings(cfg, 0.7, 5, 0).settings.filter_radius == radius


def test_the_options_are_checked_with_options_in_the_error():
    assert F.check_fine_passes(7) == 7 and F.check_fine_passes("12") == 12
    for bad in (0, -3, "x", "1.5", True, None, 2.5):
        with pytest.raises(UserError, match="--fine-passes .*maximum.*Options"):
            F.check_fine_passes(bad)
    assert F.check_fine_voxel("0.65") == 0.65 and F.check_fine_voxel(1) == 1.0
    for bad in (0, -1, "x", float("nan"), float("inf"), True):
        with pytest.raises(UserError, match="--fine-voxel .*Options"):
            F.check_fine_voxel(bad)
    assert F.check_fine_tol("5e-4") == 5e-4 and F.check_fine_tol(1) == 1.0
    for bad in (0, -1e-3, "x", float("nan"), True):
        with pytest.raises(UserError, match="--fine-tol .*Options"):
            F.check_fine_tol(bad)


def test_the_default_tolerance_is_one_named_constant_with_its_evidence():
    assert F.FINE_OBJECTIVE_CHANGE == 1e-3 and not hasattr(F, "FINE_PASSES_DEFAULT")
    from topopt import config
    source = open(config.__file__).read()        # the one default is next to the loader of settings.fine_pass, with its evidence as conditions
    assert source.count("FINE_OBJECTIVE_CHANGE = ") == 1 and "FINE_OBJECTIVE_CHANGE = " not in open(F.__file__).read()
    comment = source[:source.index("FINE_OBJECTIVE_CHANGE = ")]
    for words in ("599.72", "+0.46 %", "596.95", "stops at pass 21", "5e-4", "1e-4", "pass 15", "2026-10-26"):
        assert words in comment, words
    for words in ("phase", "drone-arm", "/runs/"):
        assert words not in comment[comment.index("Evidence"):], words


# --- the mapping --------------------------------------------------------------------------------------

def make_parent(shape=(10, 10, 10), h=1.0, origin=(0.0, 0.0, 0.0), field=None, labels=None, rho=None, iteration=5):
    labels = np.full(shape, DESIGN, np.uint8) if labels is None else labels
    x = np.full(shape, np.nan) if field is None else field.copy()
    return F.Parent(run_dir=None, shape=shape, origin=np.asarray(origin, float), h=h, labels=labels, rho=rho, x=x, iteration=iteration,
                    volume_fraction=0.3, facts=[])


def centres_of(shape, h, origin):
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    return np.asarray(origin, float) + h * (ijk + 0.5)


def linear(c):
    return 0.1 + 0.05 * c[:, 0] + 0.02 * c[:, 1] + 0.01 * c[:, 2]


def test_a_linear_field_is_mapped_exactly_inside_the_parent_grid():
    h = 1.0
    ijk = np.stack(np.meshgrid(*[np.arange(10)] * 3, indexing="ij"), axis=-1)
    parent = make_parent(field=linear((ijk.reshape(-1, 3) + 0.5) * h).reshape(10, 10, 10))
    points = centres_of((16, 16, 16), 0.5, (1.0, 1.0, 1.0))                     # a finer grid inside the parent, voxel 0.5, shifted origin
    values, info = F.map_field(parent, points, "x")
    assert np.allclose(values, linear(points), atol=1e-12)
    assert info["no_source"] == 0 and info["source"] == "x"


def test_two_grids_with_another_origin_are_mapped_by_place():
    ijk = np.stack(np.meshgrid(*[np.arange(10)] * 3, indexing="ij"), axis=-1)
    parent = make_parent(origin=(0.3, -0.2, 0.0), field=linear((ijk.reshape(-1, 3) + 0.5) + np.array([0.3, -0.2, 0.0])).reshape(10, 10, 10))
    points = np.array([[3.3, 2.8, 4.0], [5.05, 5.05, 5.05], [7.9, 6.1, 2.2]])
    values, _ = F.map_field(parent, points, "x")
    assert np.allclose(values, linear(points), atol=1e-12)


def test_a_hole_of_the_parent_takes_the_value_of_the_design_around_it():
    labels = np.full((10, 10, 10), DESIGN, np.uint8)
    labels[4:6, 4:6, :] = 0                                   # a hole through the parent
    parent = make_parent(labels=labels, field=np.full((10, 10, 10), 0.4))
    values, info = F.map_field(parent, np.array([[5.0, 5.0, 5.0], [4.2, 5.0, 5.0], [2.5, 2.5, 5.0]]), "x")
    # (5, 5, 5) has only hole voxels around it: no source, the nearest design voxel; (4.2, 5, 5) has design voxels on one side only: less weight
    assert np.allclose(values, 0.4) and info["no_source"] == 1 and info["weak_source"] >= 1


def test_a_point_far_from_any_source_takes_the_nearest_source_voxel():
    field = np.full((10, 10, 10), 0.25)
    field[9, 9, 9] = 0.9
    parent = make_parent(field=field)
    values, info = F.map_field(parent, np.array([[30.0, 30.0, 30.0], [-20.0, 5.0, 5.0]]), "x")
    assert values[0] == pytest.approx(0.9) and values[1] == pytest.approx(0.25) and info["no_source"] == 2


def test_the_x_source_reads_only_design_voxels_and_the_rho_source_counts_the_frozen_ones_as_density_one():
    labels = np.full((6, 6, 6), DESIGN, np.uint8)
    labels[0, :, :] = FIXED
    labels[5, :, :] = LOAD
    rho = np.full((6, 6, 6), 0.5)
    rho[0], rho[5] = 1.0, 1.0
    parent = make_parent((6, 6, 6), labels=labels, field=np.where(labels == DESIGN, 0.2, np.nan), rho=rho)
    point = np.array([[0.5, 3.0, 3.0]])                        # the centre of a fixed voxel
    assert F.map_field(parent, point, "x")[0][0] == pytest.approx(0.2)         # the fixed voxels are no sources of `x`
    assert F.map_field(parent, point, "rho")[0][0] == pytest.approx(1.0)       # they are sources of the density, with density 1


def test_a_source_that_the_parent_does_not_have_is_refused_with_options():
    parent = make_parent(rho=None)
    parent.x = None
    with pytest.raises(UserError, match="no checkpoint.*Options"):
        F.map_field(parent, np.zeros((1, 3)), "x")
    with pytest.raises(UserError, match="no snapshot.*Options"):
        F.map_field(make_parent(field=np.zeros((10, 10, 10))), np.zeros((1, 3)), "rho")
    with pytest.raises(UserError, match="must be one of"):
        F.map_field(make_parent(field=np.zeros((10, 10, 10))), np.zeros((1, 3)), "u")


def test_the_source_is_x_with_a_checkpoint_and_the_density_without_one_with_a_fact():
    assert F.choose_source(make_parent(field=np.zeros((10, 10, 10)))) == ("x", [])
    parent = make_parent(rho=np.zeros((10, 10, 10)))
    parent.x = None
    source, facts = F.choose_source(parent)
    assert source == "rho" and "filters it twice" in facts[0] and "854 against 726" in facts[0]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    cfg = load_config(make_case(tmp_path_factory.mktemp("fit"), h=2.0))
    return Model(build_problem(cfg))


def test_the_fit_reaches_the_volume_target(tiny_model):
    x = np.random.default_rng(3).uniform(0.2, 0.8, tiny_model.n_design)
    for target in (0.2, 0.3, 0.45):
        fitted, shift, mean, before = F.fit_volume(tiny_model, x, 2.0, target)
        assert mean == pytest.approx(target, abs=1e-6) and np.all((fitted >= 0) & (fitted <= 1))
        assert shift == pytest.approx(shift) and abs(mean - target) < abs(before - target) + 1e-6


def test_the_fit_reports_a_target_it_cannot_reach(tiny_model):
    x = np.full(tiny_model.n_design, 0.5)
    fitted, shift, mean, _ = F.fit_volume(tiny_model, x, 2.0, 5.0)           # a mean density above 1
    assert shift == 1.0 and mean < 5.0 and np.all(fitted == 1.0)
    fitted, shift, mean, _ = F.fit_volume(tiny_model, x, 2.0, -1.0)
    assert shift == -1.0 and mean > -1.0 and np.all(fitted == 0.0)


def test_the_start_design_has_the_volume_target_and_the_facts(tiny_model):
    problem = tiny_model.problem
    centres = tiny_model.design_centers()
    parent = make_parent((11, 5, 5), h=2.0, origin=problem.origin, field=np.full((11, 5, 5), 0.5),
                         labels=np.asarray(problem.labels).copy())
    lines, record = [], {}
    x = F.start_design(parent, tiny_model, 2.0, "x", log=lines.append, record=record)
    assert x.shape == (tiny_model.n_design,) and centres.shape == (tiny_model.n_design, 3)
    assert record["volume"]["mean_after"] == pytest.approx(problem.config.settings.volume_fraction, abs=1e-6)
    assert record["map"]["points"] == tiny_model.n_design and record["map"]["no_source"] == 0
    assert any("start design from pass 5" in line for line in lines) and any("fitted to the volume target" in line for line in lines)
    assert math.isclose(record["origin_shift_mm"], 0.0, abs_tol=1e-9)


def test_the_fine_start_maps_once_and_lets_go_of_the_parent(tiny_model):
    problem = tiny_model.problem
    parent = make_parent((11, 5, 5), h=2.0, origin=problem.origin, field=np.full((11, 5, 5), 0.5), labels=np.asarray(problem.labels).copy())
    start = F.FineStart(parent, {"max_passes": 4}, 2.0, "x", extra_lines=["fine pass: first line"])
    x = start(tiny_model)
    assert x.shape == (tiny_model.n_design,) and start.parent is None
    assert parent.labels is None and parent.x is None and parent.rho is None and parent.ck is None     # also the plan of the run lets go of them
    assert parent.iteration == 5 and parent.h == 2.0 and parent.shape == (11, 5, 5)                      # the numbers stay
    assert start.lines[0] == "fine pass: first line" and any("start design" in line for line in start.lines[1:])
    assert start.record["volume"]["target"] == pytest.approx(problem.config.settings.volume_fraction)


def test_the_end_line_says_how_a_fine_pass_ended():
    rec = {"max_passes": 12, "tolerance": 1e-3, "window": 5, "stop_rule": "settled"}
    settled = {"state": "stopped_by_rule", "stop": {"rule": "settled", "iteration": 7}}
    assert "the loss settled at pass 7 (the maximum is 12 passes)" in F.end_line(settled, rec)
    limit = {"state": "iteration_limit", "measures": {"objective_change": {"value": 0.02, "passes_needed": 6, "passes_at_beta": 12, "beta": 8.0}}}
    line = F.end_line(limit, rec)
    assert "the maximum of 12 passes ended the fine pass before the loss settled" in line and "0.02 over the last 5 passes" in line
    assert "higher --fine-passes" in line and "--fine-tol" in line
    short = {"state": "iteration_limit", "measures": {"objective_change": {"value": None, "passes_needed": 6, "passes_at_beta": 3, "beta": 8.0}}}
    assert "not judged: it needs 6 passes in a row at one beta" in F.end_line(short, rec)
    assert F.end_line({"state": "stopped_by_request"}, rec) is None
    assert F.end_line({"state": "stopped_by_rule", "stop": {"rule": "time_budget", "iteration": 3}}, rec) is None
    # all stop rules off: the fine pass runs to its maximum and the line says so (a continuation counts its own passes)
    off = dict(rec, stop_rule=None)
    assert "all stop rules were off (--no-stop-rules), so the fine pass ran its 12 passes" in F.end_line({"state": "iteration_limit"}, off)
    assert "ran its 20 passes" in F.end_line({"state": "iteration_limit"}, off, 20)
    assert F.end_line({"state": "stopped_by_request"}, off) is None
