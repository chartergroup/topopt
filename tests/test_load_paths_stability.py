"""Tests of the two facts that every run has since I6: `metrics.load_paths` and `metrics.threshold_stability` (CONTRACTS.md section 5).

`metrics.load_paths` is the result of `stress_report.load_path_facts` with the `final.stl` part, for every run (before I6 only a run with a stress
row had it, as `metrics.stress.load_paths`). `metrics.threshold_stability` is the voxel volume and the body count of the thresholded design at the
levels 0.4, 0.5 and 0.6 (`stage_facts.STABILITY_LEVELS` and `THRESHOLD`). Both are facts only: they change no result, no verdict and no exit code.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from test_optimize import make_case
from test_stress_report import stress_yaml
from topopt import post, stage_facts as sf, stress_report as sr, verdict
from topopt.cli import main
from topopt.refine.build import problem_from_run
from topopt.voxelize import DESIGN, FIXED, LOAD


def read(folder, name):
    return json.loads((Path(folder) / name).read_text())


def counting_run(tmp, yaml_path):
    """Run a case and count the calls of `stress_report.load_path_facts` that ask for `final.stl` (the stage-end facts of the optimizer ask for none)."""
    out, calls = tmp / "run", []
    real_fn = sr.load_path_facts

    def counting(problem, snap, final_stl=None):
        calls.append(final_stl)
        return real_fn(problem, snap, final_stl)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sr, "load_path_facts", counting)
        assert main(["run", str(yaml_path), "--out", str(out), "--quiet"]) == 0
    return out, [c for c in calls if c is not None]


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """One real short run (cantilever, voxel 2 mm, 12 passes) and the calls that asked for `final.stl`."""
    tmp = tmp_path_factory.mktemp("lp_real")
    return counting_run(tmp, make_case(tmp, h=2.0, iterations=12))


@pytest.fixture(scope="module")
def stress_real(tmp_path_factory):
    """A short run with a stress row (the stress section also reads the load path) and the calls that asked for `final.stl`."""
    tmp = tmp_path_factory.mktemp("lp_stress")
    return counting_run(tmp, stress_yaml(tmp))


# --- stage_facts.threshold_stability: the levels, the cut and the counts -----------------------------------------------------------

def line(codes, bytes_):
    """A grid of one row: `codes` are the labels along x, `bytes_` the snapshot bytes of the design voxels (the other voxels hold 255)."""
    labels = np.array(codes, np.uint8).reshape(-1, 1, 1)
    snap = np.full(labels.shape, 255, np.uint8)
    snap[labels == DESIGN] = bytes_
    return labels, snap


def test_the_levels_are_a_constant_next_to_the_threshold():
    assert sf.STABILITY_LEVELS == (0.4, 0.6) and sf.THRESHOLD == 0.5
    assert sf.stability_levels() == [0.4, 0.5, 0.6]
    assert sf.STABILITY_LEVELS[0] < sf.THRESHOLD < sf.STABILITY_LEVELS[1]


def test_each_level_gives_the_voxel_volume_and_the_body_count_of_its_thresholded_design():
    """Fixed end, seven design voxels, load end. The design bytes 200, 153, 152, 128, 127, 102, 102 cut at 102 (0.4), 128 (0.5) and 153 (0.6)."""
    labels, snap = line([FIXED] + [DESIGN] * 7 + [LOAD], [200, 153, 152, 128, 127, 102, 102])
    out = sf.threshold_stability(labels, snap, 2.0)
    assert "error" not in out and out["rule"] == sf.STABILITY_RULE
    assert [(e["level"], e["volume_mm3"], e["bodies"]) for e in out["levels"]] == [
        (0.4, 9 * 8.0, 1),      # all 9 voxels solid: one body
        (0.5, 6 * 8.0, 2),      # fixed + 4 design voxels, and the load voxel alone
        (0.6, 4 * 8.0, 2)]      # fixed + 2 design voxels, and the load voxel alone
    volumes = [e["volume_mm3"] for e in out["levels"]]
    assert volumes == sorted(volumes, reverse=True)       # 0.4 >= 0.5 >= 0.6


def test_a_byte_one_below_the_cut_is_void_and_the_cut_byte_is_solid_at_each_level():
    labels, snap = line([FIXED] + [DESIGN] * 6 + [LOAD], [101, 102, 127, 128, 152, 153])
    levels = {e["level"]: e for e in sf.threshold_stability(labels, snap, 1.0)["levels"]}
    # design voxels at or above the cut: 0.4 -> 102 (5 voxels), 0.5 -> 128 (3), 0.6 -> 153 (1); the fixed and the load voxel are solid at every level
    assert [levels[v]["volume_mm3"] for v in (0.4, 0.5, 0.6)] == [7.0, 5.0, 3.0]


def test_the_byte_cut_is_the_float_rule_of_the_snapshot_for_every_byte():
    """A snapshot stores round(255 rho). The cut of each level keeps the voxels with byte / 255 >= level, the rule of `thresholded_fraction`."""
    labels = np.full((256, 1, 1), DESIGN, np.uint8)
    snap = np.arange(256, dtype=np.uint8).reshape(256, 1, 1)
    for entry in sf.threshold_stability(labels, snap, 1.0)["levels"]:
        assert entry["volume_mm3"] == float(sum(b / 255 >= entry["level"] for b in range(256))), entry
    half = next(e for e in sf.threshold_stability(labels, snap, 1.0)["levels"] if e["level"] == sf.THRESHOLD)
    assert half["volume_mm3"] == round(sf.thresholded_fraction(np.arange(256) / 255.0) * 256, 1)


def test_the_forced_solid_voxels_count_at_every_level_and_a_void_design_has_their_bodies_only():
    labels, snap = line([FIXED, FIXED, DESIGN, DESIGN, LOAD], [0, 0])
    for entry in sf.threshold_stability(labels, snap, 3.0)["levels"]:
        assert (entry["volume_mm3"], entry["bodies"]) == (3 * 27.0, 2), entry


def test_a_wrong_input_gives_an_error_and_no_exception():
    labels, _ = line([FIXED, DESIGN, LOAD], [200])
    out = sf.threshold_stability(labels, np.zeros(5, np.uint8), 1.0)
    assert list(out) == ["error"] and out["error"]


# --- the post step: every run has both keys ---------------------------------------------------------------------------------------

def test_every_run_has_the_load_paths_of_the_final_mesh(real):
    out, calls = real
    m = read(out, "result/metrics.json")
    lp = m["load_paths"]
    assert "error" not in lp and lp["final_stl"]["bodies"] >= 1 and lp["final_stl_rule"]
    tip = lp["regions"]["tip"]
    assert tip["thresholded_design"]["connected_to_fixed"] is True and tip["final_stl"]["connected_to_fixed"] is True
    run = read(out, "run.json")
    again = sr.load_path_facts(problem_from_run(out, run), np.frombuffer(post._last_snapshot(out).read_bytes(), np.uint8), out / "result" / "final.stl")
    assert lp == again       # the same call as the stress section made before I6
    assert len(calls) == 1   # one check of final.stl for the run


def test_the_three_levels_are_in_every_run_and_the_volumes_are_in_order(real):
    out, _ = real
    m = read(out, "result/metrics.json")
    ts = m["threshold_stability"]
    assert "error" not in ts and [e["level"] for e in ts["levels"]] == [0.4, 0.5, 0.6]
    v = [e["volume_mm3"] for e in ts["levels"]]
    assert v[0] >= v[1] >= v[2] > 0
    run = read(out, "run.json")
    labels = np.frombuffer((out / "labels.bin").read_bytes(), np.uint8).reshape(run["shape"])
    snap = np.frombuffer(post._last_snapshot(out).read_bytes(), np.uint8).reshape(run["shape"])
    solid = ((labels == DESIGN) & (snap >= 128)) | (labels == FIXED) | (labels == LOAD)
    assert v[1] == round(float(solid.sum()) * run["h"] ** 3, 1)                      # the 0.5 cut is the thresholded design of the load path facts
    assert ts["levels"][1]["bodies"] == m["load_paths"]["thresholded_design_bodies"]  # and has its body count


def test_both_facts_change_no_verdict(real):
    out, _ = real
    m = read(out, "result/metrics.json")
    assert "threshold_stability" not in json.dumps(m["verdict"]) and "load_paths" not in json.dumps(m["verdict"])
    assert list(m)[-1] == "verdict"            # the verdict stays the last key
    # The run wrote `verdict.json` before the post step: a post step that changed the verdict (a new test, a new key) fails this line.
    assert m["verdict"] == verdict.read_verdict(out)


# --- a part that is cut: the final mesh does not join the load region -------------------------------------------------------------

def cut_copy(source, tmp_path):
    """A copy of the finished run in which a slab of three voxel layers of the design is void in the last snapshot (a part cut in two)."""
    out = tmp_path / "cut"
    shutil.copytree(source, out)
    run = read(out, "run.json")
    shape = tuple(run["shape"])
    labels = np.frombuffer((out / "labels.bin").read_bytes(), np.uint8).reshape(shape)
    path = post._last_snapshot(out)
    snap = np.frombuffer(path.read_bytes(), np.uint8).reshape(shape).copy()
    mid = shape[0] // 2
    slab = labels[mid - 1:mid + 2] == DESIGN
    snap[mid - 1:mid + 2][slab] = 0
    path.write_bytes(snap.tobytes())
    return out


@pytest.mark.parametrize("problem_passed", [False, True])
def test_a_cut_part_says_that_final_stl_does_not_join_the_load_region(real, tmp_path, problem_passed):
    out = cut_copy(real[0], tmp_path)
    problem = problem_from_run(out, read(out, "run.json")) if problem_passed else None
    m = post.finish(out, problem)
    tip = m["load_paths"]["regions"]["tip"]
    assert tip["thresholded_design"]["connected_to_fixed"] is False
    assert tip["final_stl"]["connected_to_fixed"] is False and tip["final_stl"]["touches_a_body"] is True
    assert m["load_paths"]["final_stl"]["bodies"] >= 2
    intact = read(real[0], "result/metrics.json")["threshold_stability"]["levels"]
    for cut, whole in zip(m["threshold_stability"]["levels"], intact):
        assert cut["level"] == whole["level"] and cut["bodies"] >= 2 and cut["volume_mm3"] < whole["volume_mm3"]
    assert read(out, "result/metrics.json")["load_paths"] == m["load_paths"]


def test_a_run_without_final_stl_has_the_thresholded_part_and_no_mesh_part(real, tmp_path):
    out = real[0]
    run = read(out, "run.json")
    problem = problem_from_run(out, run)
    snap = np.frombuffer(post._last_snapshot(out).read_bytes(), np.uint8).reshape(run["shape"])
    lp = post._load_paths(out, run, problem, snap, tmp_path / "absent.stl")
    assert lp["final_stl"] is None and "final_stl" not in lp["regions"]["tip"]
    assert lp["regions"]["tip"]["thresholded_design"]["connected_to_fixed"] is True


def test_a_fault_in_the_load_path_never_stops_the_post_step(real, tmp_path):
    out = tmp_path / "gone"
    shutil.copytree(real[0], out)
    run = read(out, "run.json")
    run["yaml_path"] = str(tmp_path / "no_such_file.yaml")        # the problem cannot be rebuilt
    (out / "run.json").write_text(json.dumps(run))
    m = post.finish(out)       # the mesh step reads the region files next to the YAML too: it falls back to the voxel labels, so the mass is not compared here
    assert "error" in m["load_paths"] and "YAML file of the run is missing" in m["load_paths"]["error"]
    assert "error" not in m["threshold_stability"] and len(m["threshold_stability"]["levels"]) == 3
    assert m["mass_final_g"] > 0 and (out / "result" / "final.stl").is_file() and read(out, "status.json")["state"] == "done"
    assert read(out, "result/metrics.json")["load_paths"] == m["load_paths"]


# --- a run with a stress row: one check of final.stl --------------------------------------------------------------------------------

def test_a_run_with_a_stress_row_checks_final_stl_once_and_the_stress_section_has_the_same_facts(stress_real):
    out, calls = stress_real
    assert len(calls) == 1
    m = read(out, "result/metrics.json")
    assert "error" not in m["stress"] and m["stress"]["load_paths"] == m["load_paths"]
    assert m["load_paths"]["final_stl"]["bodies"] >= 1


def test_the_stress_section_uses_the_load_paths_that_it_is_given(stress_real, monkeypatch):
    out, _ = stress_real
    given = read(out, "result/metrics.json")["load_paths"]
    monkeypatch.setattr(sr, "load_path_facts", lambda *a, **k: pytest.fail("the stress section made a second check of final.stl"))
    section = sr.stress_section(out, load_paths=given)
    assert "error" not in section and section["load_paths"] == given
    assert section["cases"]["down"]["load_path"]["all_connected"] is True
