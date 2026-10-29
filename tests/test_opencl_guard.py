"""Tests of the OpenCL guard in tests/conftest.py (review m2 and m3). No OpenCL, no pyopencl import, no driver load.

The guard keeps the tests of gpu-mg from making a context on a GPU. It reads the ICD files (the list of
OpenCL drivers) before any driver loads. These tests use fake `.icd` files in a temp folder.
"""
import os
import sys
from types import SimpleNamespace

import pytest

import conftest


@pytest.fixture(autouse=True)
def keep_the_auto_gpu_variable(monkeypatch):
    """`pytest_sessionstart` sets TOPOPT_AUTO_GPU. Some tests here call it by hand, so the variable goes back to its value after each test."""
    monkeypatch.setenv(conftest.AUTO_GPU_VARIABLE, os.environ.get(conftest.AUTO_GPU_VARIABLE, "0"))


def icd(folder, name, library):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(library + "\n")
    return folder


def test_the_vendor_folder_is_read_and_a_driver_that_is_not_pocl_is_found(tmp_path):
    folder = icd(tmp_path / "vendors", "intel.icd", "libigdrcl.so")
    text = conftest._icd_problem({"OCL_ICD_VENDORS": str(folder)}, "linux")
    assert text is not None and "not pocl" in text and "libigdrcl.so" in text


def test_a_value_of_ocl_icd_filenames_does_not_hide_the_vendor_folder(tmp_path):
    """The loader inside the pyopencl wheel ignores OCL_ICD_FILENAMES (review m2): the folder counts anyway."""
    folder = icd(tmp_path / "vendors", "intel.icd", "libigdrcl.so")
    env = {"OCL_ICD_VENDORS": str(folder), "OCL_ICD_FILENAMES": "/x/libpocl.so"}
    assert "libigdrcl.so" in conftest._icd_problem(env, "linux")


def test_a_driver_named_only_in_ocl_icd_filenames_counts_for_a_system_loader(tmp_path):
    env = {"OCL_ICD_VENDORS": str(tmp_path / "empty"), "OCL_ICD_FILENAMES": "/x/libpocl.so:/x/libigdrcl.so"}
    assert "libigdrcl.so" in conftest._icd_problem(env, "linux")


def test_the_pyopencl_home_libs_folder_is_read(tmp_path):
    home = tmp_path / "home"
    icd(home / ".libs", "other.icd", "libother_vendor.so")
    pocl_only = icd(tmp_path / "vendors", "pocl.icd", "libpocl-1234.so")
    env = {"OCL_ICD_VENDORS": str(pocl_only), "PYOPENCL_HOME": str(home)}
    assert "libother_vendor.so" in conftest._icd_problem(env, "linux")


def test_a_pocl_only_list_is_fine_and_the_check_does_not_import_pyopencl(tmp_path):
    folder = icd(tmp_path / "vendors", "pocl.icd", "libpocl-21062cec.so")
    before = "pyopencl" in sys.modules
    assert conftest._icd_problem({"OCL_ICD_VENDORS": str(folder), "PYOPENCL_HOME": str(tmp_path / "no")}, "linux") is None
    assert ("pyopencl" in sys.modules) == before          # the guard only reads files


def test_the_default_vendor_folder_is_read_when_the_variable_is_not_set(monkeypatch):
    seen = []
    monkeypatch.setattr(conftest.glob, "glob", lambda pattern: seen.append(pattern) or [])
    conftest._icd_names({"PYOPENCL_HOME": "/nowhere"}, "linux")
    assert any(p.startswith("/etc/OpenCL/vendors") for p in seen) and any("/nowhere/.libs" in p for p in seen)


def test_the_static_check_is_for_linux_only(tmp_path):
    folder = icd(tmp_path / "vendors", "intel.icd", "libigdrcl.so")
    assert conftest._icd_problem({"OCL_ICD_VENDORS": str(folder)}, "darwin") is None


def make_items():
    def item(name, marked):
        added = []
        return SimpleNamespace(name=name, added=added, add_marker=added.append,
                               get_closest_marker=lambda m: object() if marked and m == "opencl_context" else None)
    return [item("makes_context", True), item("plain", False)]


def test_on_a_mac_the_tests_with_a_context_are_skipped_with_one_reason():
    config = SimpleNamespace(_opencl_skip="a non-CPU OpenCL device is visible: Apple M1; set TOPOPT_TEST_REAL_GPU=1")
    items = make_items()
    conftest.pytest_collection_modifyitems(config, items)
    assert len(items[0].added) == 1 and items[1].added == []
    mark = items[0].added[0]
    assert mark.name == "skip" and "TOPOPT_TEST_REAL_GPU" in mark.kwargs["reason"]


def test_nothing_is_skipped_when_the_guard_found_no_problem():
    items = make_items()
    conftest.pytest_collection_modifyitems(SimpleNamespace(_opencl_skip=None), items)
    assert all(not i.added for i in items)


@pytest.mark.parametrize("system,exits", [("linux", True), ("darwin", False)])
def test_a_visible_gpu_stops_a_linux_session_and_skips_on_a_mac(monkeypatch, system, exits):
    monkeypatch.delenv(conftest.REAL_GPU_VARIABLE, raising=False)
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(conftest.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(conftest, "_opencl_problem", lambda: "a non-CPU OpenCL device is visible: Test GPU")
    session = SimpleNamespace(config=SimpleNamespace())
    if exits:
        with pytest.raises(pytest.exit.Exception) as e:
            conftest.pytest_sessionstart(session)
        assert e.value.returncode == 3 and "TOPOPT_TEST_REAL_GPU=1" in str(e.value)
    else:
        conftest.pytest_sessionstart(session)
        assert "TOPOPT_TEST_REAL_GPU=1" in session.config._opencl_skip


def test_the_opt_in_variable_allows_a_gpu(monkeypatch):
    monkeypatch.setenv(conftest.REAL_GPU_VARIABLE, "1")
    monkeypatch.setattr(conftest, "_opencl_problem", lambda: "a non-CPU OpenCL device is visible: Test GPU")
    session = SimpleNamespace(config=SimpleNamespace())
    conftest.pytest_sessionstart(session)
    assert session.config._opencl_skip is None


def test_the_test_session_has_the_auto_gpu_variable_unless_the_real_gpu_is_allowed():
    """The live session: `auto` stays on the CPU rule in the tests of a machine with an OpenCL GPU (a value that the user set stays)."""
    if os.environ.get(conftest.REAL_GPU_VARIABLE) == "1":
        pytest.skip(f"{conftest.REAL_GPU_VARIABLE}=1: the session does not set {conftest.AUTO_GPU_VARIABLE}")
    assert os.environ.get(conftest.AUTO_GPU_VARIABLE) is not None


@pytest.fixture
def quiet_guard(monkeypatch):
    """A session start where the guard finds no problem."""
    monkeypatch.setattr(conftest, "_opencl_problem", lambda: None)
    return SimpleNamespace(config=SimpleNamespace())


def test_the_session_start_sets_the_auto_gpu_variable_to_0(monkeypatch, quiet_guard):
    monkeypatch.delenv(conftest.REAL_GPU_VARIABLE, raising=False)
    monkeypatch.delenv(conftest.AUTO_GPU_VARIABLE)
    conftest.pytest_sessionstart(quiet_guard)
    assert os.environ[conftest.AUTO_GPU_VARIABLE] == "0"


def test_the_session_start_keeps_a_value_that_the_user_set(monkeypatch, quiet_guard):
    monkeypatch.delenv(conftest.REAL_GPU_VARIABLE, raising=False)
    monkeypatch.setenv(conftest.AUTO_GPU_VARIABLE, "1")
    conftest.pytest_sessionstart(quiet_guard)
    assert os.environ[conftest.AUTO_GPU_VARIABLE] == "1"


def test_the_session_start_does_not_set_the_auto_gpu_variable_when_the_real_gpu_is_allowed(monkeypatch, quiet_guard):
    monkeypatch.setenv(conftest.REAL_GPU_VARIABLE, "1")
    monkeypatch.delenv(conftest.AUTO_GPU_VARIABLE)
    conftest.pytest_sessionstart(quiet_guard)
    assert conftest.AUTO_GPU_VARIABLE not in os.environ
