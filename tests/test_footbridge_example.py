"""The footbridge example (`examples/footbridge/`, #296): our own parametric bridge, made by `make_footbridge.py` from named round
dimensions. The files in the folder are the ones that the generator makes, the YAML sets the settings that the README numbers are for,
the short YAML differs from it only in the grid, and the folder holds no third-party file."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from topopt.config import load_config

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "examples" / "footbridge"


@pytest.fixture(scope="module")
def made():
    spec = importlib.util.spec_from_file_location("footbridge_make", FOLDER / "make_footbridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.build()


def test_the_region_files_are_the_ones_of_the_generator(made):
    _mod, meshes = made
    assert sorted(f"{n}.stl" for n in meshes) == sorted(p.name for p in FOLDER.glob("*.stl"))
    for name, m in meshes.items():
        f = trimesh.load(FOLDER / f"{name}.stl")
        assert f.is_watertight, name
        assert f.volume == pytest.approx(m.volume, rel=1e-6), name
        assert np.allclose(f.bounds, m.bounds, atol=1e-4), name


def test_the_regions_have_the_places_that_the_readme_gives(made):
    mod, meshes = made
    box = {name: np.round(m.bounds, 3).tolist() for name, m in meshes.items()}
    assert box["design"] == [[0, -15.5, 0], [280, 15.5, 86.5]]          # 2.5 mm inside the skin, 1 mm into it
    assert box["deck_skin"] == [[0, -18, 85.5], [280, 18, 88]]
    assert box["walk_left"] == [[0, -18, 88], [140, 18, 91]] and box["walk_right"] == [[140, -18, 88], [280, 18, 91]]
    assert box["feet_a"] == [[0, -18, 0], [24, 18, 30]] and box["feet_b"] == [[256, -18, 0], [280, 18, 30]]
    assert box["clearance"] == [[40, -22, -5], [240, 22, 56]]
    assert mod.SKIN_Z0 == mod.DECK_TOP - mod.SKIN


def test_the_footbridge_yaml_sets_the_settings_that_its_readme_numbers_name():
    s = load_config(FOLDER / "footbridge.yaml").settings
    assert (s.voxel_size, s.min_member_mm, s.iterations, s.volume_fraction, s.penal, s.filter_solid_neighbours) == \
        (1.1, 2.3, 200, 0.1, 4.0, "ignore")
    assert s.max_active_voxels >= 394348                       # the active voxels of the grid (README, "Time and memory")


def test_the_short_yaml_is_the_same_problem_on_a_coarser_grid():
    full, short = (yaml.safe_load((FOLDER / f"{n}.yaml").read_text(encoding="utf-8")) for n in ("footbridge", "footbridge_short"))
    assert short["name"] == "footbridge_short"
    assert {k: v for k, v in full.items() if k not in ("name", "settings")} == \
        {k: v for k, v in short.items() if k not in ("name", "settings")}
    changed = {k for k in full["settings"].keys() | short["settings"].keys() if full["settings"].get(k) != short["settings"].get(k)}
    assert changed == {"voxel_size", "min_member_mm", "iterations", "max_active_voxels"}
    s = load_config(FOLDER / "footbridge_short.yaml").settings
    assert (s.voxel_size, s.min_member_mm, s.iterations) == (2.1, 4.4, 80)
    # the member is about 2.1 voxels across in both
    assert s.min_member_mm / s.voxel_size == pytest.approx(full["settings"]["min_member_mm"] / full["settings"]["voxel_size"], rel=0.01)


def test_the_example_holds_no_third_party_file():
    assert not (FOLDER / "COPYING").exists()
    assert "footbridge" not in (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
