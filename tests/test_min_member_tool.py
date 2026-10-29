"""Tests for tools/calibrate_min_member.py: the thickness measure and the case. No optimizer run."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from topopt.config import load_config
from topopt.problem import build_problem

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibrate_min_member", ROOT / "tools" / "calibrate_min_member.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_local_thickness_of_a_plate_is_within_a_quarter_voxel(m):
    mat = np.zeros((24, 24, 24), bool)
    mat[:, :, 8:8 + m] = True
    t = cal.local_thickness(mat, 1.0)[mat]
    assert abs(float(np.median(t)) - m) <= 0.26


def test_local_thickness_scales_with_the_voxel_size():
    mat = np.zeros((16, 16, 16), bool)
    mat[:, :, 5:9] = True
    assert cal.local_thickness(mat, 2.0).max() == pytest.approx(2.0 * cal.local_thickness(mat, 1.0).max())


def test_interior_mask_leaves_the_plate_side_alone():
    m = cal.interior_mask((10, 8, 6), 2)
    assert m[0, 4, 3] and m[7, 4, 3] and not m[8, 4, 3]      # x: only the free end is cut
    assert not m[5, 1, 3] and not m[5, 6, 3] and not m[5, 4, 1] and not m[5, 4, 4]


def test_summarize_numbers():
    s = cal.summarize(np.array([1.0, 2.0, 3.0, 4.0]), 2.5)
    assert s["voxels"] == 4 and s["min_mm"] == 1.0 and s["median_mm"] == 2.5 and s["share_below_request"] == 0.5
    assert cal.summarize(np.empty(0), 1.0) == {"voxels": 0}


def test_case_goes_through_the_normal_config_path_and_the_guards(tmp_path):
    p = cal.write_case(tmp_path, {"min_member_mm": 2.5})
    cfg = load_config(p)
    assert cfg.settings.min_member_mm == 2.5 and cfg.settings.filter_radius_source == "min_member_mm"
    prob = build_problem(cfg)                       # the voxel-count guards run here
    assert prob.shape == (50, 24, 16) and int((prob.labels == 1).sum()) < cfg.settings.max_active_voxels
    assert cfg.settings.max_grid_voxels == 5000000 and cfg.settings.max_active_voxels == 100000   # the defaults


def test_measure_splits_the_design_material_into_interior_and_face_layer():
    shape = (20, 14, 12)
    density = np.zeros(shape)
    density[2:18, 1:13, 1:11] = 1.0                       # a block that fills the design box
    labels = np.ones(shape, np.uint8)
    m = cal.measure(density, labels, 1.0, 3.0)
    assert m["all"]["voxels"] == m["interior"]["voxels"] + m["face_layer"]["voxels"]
    assert m["face_layer_share_of_material"] == pytest.approx(m["face_layer"]["voxels"] / m["all"]["voxels"], abs=1e-3)
    assert m["interior"]["median_mm"] > m["face_layer"]["min_mm"] > 0


def test_peak_rss_unit_follows_the_platform(monkeypatch):
    """`ru_maxrss` is KiB on Linux and bytes on macOS: 512 MiB reads as 512.0 on both."""
    monkeypatch.setattr(cal.sys, "platform", "linux")
    assert cal.peak_rss_mib(512 * 1024) == 512.0
    monkeypatch.setattr(cal.sys, "platform", "darwin")
    assert cal.peak_rss_mib(512 * 2 ** 20) == 512.0
    assert cal.peak_rss_mib() > 0                     # the live value, on this platform
