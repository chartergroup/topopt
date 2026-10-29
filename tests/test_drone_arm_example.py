"""The drone-arm example (`examples/drone_arm/`, #296, #374): our own parametric arm, made by `make_arm.py` from named round
dimensions. The files in the folder are the ones that the generator makes, the YAML sets the settings that the README numbers
are for, and the folder holds no third-party file (no arm from the drawing of another frame)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

from topopt.config import load_config

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "examples" / "drone_arm"


@pytest.fixture(scope="module")
def made():
    spec = importlib.util.spec_from_file_location("drone_arm_make_arm", FOLDER / "make_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meshes, area = mod.build()
    return mod, meshes, area


def test_the_region_files_are_the_ones_of_the_generator(made):
    _mod, meshes, _area = made
    assert sorted(f"{n}.stl" for n in meshes) == sorted(p.name for p in FOLDER.glob("*.stl"))
    for name, m in meshes.items():
        f = trimesh.load(FOLDER / f"{name}.stl")
        assert f.is_watertight, name
        assert f.volume == pytest.approx(m.volume, rel=1e-6), name
        assert np.allclose(f.bounds, m.bounds, atol=1e-4), name


def test_the_mass_target_is_the_plain_plate_arm(made):
    """`target_mass_g` is the mass of `arm_original.stl` (the plate arm) in the material of the YAML: the comparison at about the same
    mass."""
    mod, meshes, area = made
    cfg = load_config(FOLDER / "drone_arm.yaml")
    vol = meshes["arm_original"].volume
    assert vol == pytest.approx(area * (mod.PLATE_Z[1] - mod.PLATE_Z[0]), rel=1e-3)
    assert cfg.settings.target_mass_g == round(vol * cfg.material.density / 1000.0, 3)


def test_the_drone_arm_yaml_sets_the_settings_that_its_readme_numbers_name():
    """The README numbers of the drone arm are for an explicit `envelope_fill: on` and the eroded robust design (method settings). If the
    YAML loses one of them, the numbers no longer belong to it."""
    s = load_config(FOLDER / "drone_arm.yaml").settings
    assert (s.envelope_fill, s.robust, s.eta_eroded, s.iterations, s.voxel_size) == ("on", "eroded", 0.7, 200, 0.5)
    assert "envelope_fill: on" in (FOLDER / "drone_arm.yaml").read_text(encoding="utf-8")   # set in the file, not taken from the default
    ref = load_config(FOLDER / "drone_arm_original.yaml")
    assert ref.regions.design.name == "arm_original.stl" and ref.settings.volume_fraction == 1.0


def test_the_example_holds_no_third_party_file():
    assert not (FOLDER / "COPYING").exists()
    assert "drone_arm" not in (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    text = " ".join(p.read_text(encoding="utf-8").lower() for p in [FOLDER / "README.md", *FOLDER.glob("*.yaml"), FOLDER / "make_arm.py"])
    for word in ("tbs", "source_one", "source one", ".dxf", "gpl"):
        assert word not in text, word
