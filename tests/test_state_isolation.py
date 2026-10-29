"""The per-user state folder of topopt in the tests (P2a review m2): `tests/conftest.py` gives every test a folder of its own."""
import os
from pathlib import Path

from topopt.gpu_mg import check as gpu_check


def test_a_test_does_not_see_the_state_folder_of_the_user(state_folder):
    """A result of `check` with gpu-mg depends on `gpu-constants.json` of the state folder, so a test must not read the file of the user."""
    assert os.environ[gpu_check.STATE_VARIABLE] == str(state_folder)
    assert gpu_check.state_dir() == state_folder and gpu_check.state_dir() != Path.home() / ".topopt"
    assert not state_folder.exists()                                   # not made until a test writes in it
    assert gpu_check.load_calibrations() == ([], None)
    assert gpu_check.calibration_path() == state_folder / "gpu-constants.json"


def test_a_test_that_asks_for_no_fixture_has_a_state_folder_of_its_own_too():
    """The fixture is autouse: a test in any file (springs, joints, robust, the builder) is cut off from `~/.topopt` without asking."""
    folder = os.environ.get(gpu_check.STATE_VARIABLE)
    assert folder and Path(folder) != Path.home() / ".topopt"
    assert gpu_check.state_dir() == Path(folder) and gpu_check.load_calibrations() == ([], None)
