"""Tests of the choice logic of the solver gpu-mg (CONTRACTS.md section 20).

These tests need no OpenCL: the devices, the platform and the pyopencl import are faked. They run on
every platform and without pyopencl. The solver itself is tested in `test_gpu_mg.py`.
"""
import json
import platform
import sys
from pathlib import Path

import pytest
import yaml

from helpers import in_process_probe
from test_optimize import make_case
from topopt import cli, solver_choice
from topopt.cli import main
from topopt.config import UserError, load_config
from topopt.gpu_mg import check as gpu_check, device
from topopt.gpu_mg.device import DeviceInfo

REPO = Path(__file__).resolve().parent.parent


def dev(index=0, name="Test GPU", kind="GPU", fp64=False, platform_name="Test Platform", **kw):
    return DeviceInfo(index=index, platform=platform_name, name=name, kind=kind, fp64=fp64,
                      version="OpenCL 1.2", max_work_group_size=256, max_mem_alloc=2**30,
                      global_mem=2**32, local_mem=32768, constant_mem=65536, compute_units=8, **kw)


@pytest.fixture
def fake_devices(monkeypatch):
    """`fake_devices([dev, ...])` replaces the device list and says that pyopencl loads."""
    def set_devices(devices):
        in_process_probe(monkeypatch)       # the product asks a child process; the fakes act in this process
        monkeypatch.setattr(device, "pyopencl_problem", lambda: None)
        monkeypatch.setattr(device, "enumerate_devices", lambda: list(devices))
        return devices
    return set_devices


# --- the choice of the device ---------------------------------------------------------------


def test_auto_takes_the_gpu_before_the_cpu():
    cpu, gpu = dev(0, "Big CPU", "CPU", True), dev(1, "Small GPU", "GPU", False)
    assert device.pick_device("auto", [cpu, gpu]) is gpu
    assert device.pick_device(None, [cpu]) is cpu


def test_device_by_number_and_by_name_part():
    a, b = dev(0, "Alpha CPU", "CPU"), dev(1, "Beta GPU", "GPU")
    assert device.pick_device("1", [a, b]) is b
    assert device.pick_device("alpha", [a, b]) is a
    with pytest.raises(UserError, match=r"settings.gpu_device is 5.*2 OpenCL devices.*Options"):
        device.pick_device("5", [a, b])
    with pytest.raises(UserError, match=r"settings.gpu_device is 'nope'.*Alpha CPU.*Options"):
        device.pick_device("nope", [a, b])


def test_no_device_is_a_fact_with_options():
    with pytest.raises(UserError) as e:
        device.pick_device("auto", [])
    text = str(e.value)
    assert "no OpenCL device" in text and "settings.solver" in text
    assert "'auto'" in text and "'amg'" in text


# --- the precision case ---------------------------------------------------------------------


def test_case_auto_depends_on_fp64():
    assert device.choose_case(dev(fp64=False), "auto") == "C"
    assert device.choose_case(dev(fp64=True), "auto") == "B"
    assert device.choose_case(dev(fp64=True), None) == "B"


def test_case_c_can_be_forced_and_case_b_needs_fp64():
    assert device.choose_case(dev(fp64=True), "C") == "C"
    assert device.choose_case(dev(fp64=True), "B") == "B"
    with pytest.raises(UserError, match=r"no 64-bit float.*Options.*'C'"):
        device.choose_case(dev(fp64=False), "B")
    with pytest.raises(UserError, match="settings.gpu_case must be"):
        device.choose_case(dev(), "A")


def test_feature_problems_are_facts():
    assert device.feature_problems(dev()) == []
    bad = dev(available=False)
    bad.local_mem = 1024
    text = "; ".join(device.feature_problems(bad))
    assert "not available" in text and "local memory is 1024" in text


def test_select_stops_on_a_device_without_a_feature(fake_devices):
    bad = dev()
    bad.local_mem = 1024
    fake_devices([bad])
    with pytest.raises(UserError, match=r"cannot run gpu-mg.*local memory.*Options"):
        device.select("auto", "auto")


# --- the errors of the solver rule -----------------------------------------------------------


def test_missing_pyopencl_is_a_fact_with_options(monkeypatch):
    monkeypatch.setattr(device, "pyopencl_problem", lambda: ("missing", "No module named 'pyopencl'"))
    with pytest.raises(UserError) as e:
        solver_choice.resolve_solver("gpu-mg")
    text = str(e.value)
    assert "needs pyopencl" in text and "not installed" in text
    assert "[gpu]" in text or "pyopencl==" in text            # the install command of the gpu extra
    assert "'auto'" in text and "'cholesky'" in text and "'pardiso'" in text and "'amg'" in text
    assert "pyopencl" in cli.backend_problem("gpu-mg")


def test_pyopencl_that_does_not_load_is_a_fact(monkeypatch):
    monkeypatch.setattr(device, "pyopencl_problem", lambda: ("load_failed", "OSError: libOpenCL.so.1"))
    with pytest.raises(UserError, match=r"does not load: OSError: libOpenCL.so.1.*gpu extra.*Or set"):
        solver_choice.resolve_solver("gpu-mg")


def test_no_opencl_device_stops_the_rule(fake_devices):
    fake_devices([])
    with pytest.raises(UserError, match="no OpenCL device"):
        solver_choice.resolve_solver("gpu-mg")


def test_rule_gives_gpu_mg_when_a_device_exists(fake_devices):
    fake_devices([dev()])
    assert solver_choice.resolve_solver("gpu-mg") == "gpu-mg"
    assert "gpu-mg" in solver_choice.SOLVER_SETTINGS and solver_choice.EXPERIMENTAL_SOLVERS == ()      # gpu-mg is a main value now


@pytest.mark.parametrize("plat,machine", [("darwin", "arm64"), ("linux", "x86_64"), ("linux", "aarch64")])
def test_the_cpu_rule_of_auto_never_gives_gpu_mg(monkeypatch, fake_devices, plat, machine):
    """`resolve_solver("auto")` is the CPU rule on every platform, even with a fast GPU in the list. The pick that can be gpu-mg is `run_solver("auto")`
    (tests/test_solver_choice.py)."""
    fake_devices([dev(fp64=True), dev(1, "Other GPU")])
    monkeypatch.setattr(sys, "platform", plat)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(solver_choice, "accelerate_problem", lambda: None)
    monkeypatch.setattr(solver_choice, "pypardiso_available", lambda: True)
    assert solver_choice.resolve_solver("auto") in ("cholesky", "pardiso", "amg")


def test_backend_problem_checks_the_device_and_case_of_the_problem(fake_devices):
    from topopt.config import Settings
    fake_devices([dev(fp64=False)])
    ok = Settings(1.0, 0.5, 1, 1.0, solver="gpu-mg")
    assert cli.backend_problem("gpu-mg", None, ok) is None
    bad_case = Settings(1.0, 0.5, 1, 1.0, solver="gpu-mg", gpu_case="B")
    assert "no 64-bit float" in cli.backend_problem("gpu-mg", None, bad_case)
    bad_dev = Settings(1.0, 0.5, 1, 1.0, solver="gpu-mg", gpu_device="9")
    assert "settings.gpu_device is 9" in cli.backend_problem("gpu-mg", None, bad_dev)


def test_the_rule_checks_only_that_a_device_exists_and_the_problem_checks_the_configured_one(fake_devices):
    """`resolve_solver("gpu-mg")` does not look at settings.gpu_device or settings.gpu_case (review m1).

    The `auto` device here is a GPU that reports "not available". A user who set the working CPU device
    gets no stop from the rule and none from the check of the problem. The `auto` setting still stops, with the fact.
    """
    from topopt.config import Settings
    broken_gpu = dev(0, "Broken GPU", "GPU", True, available=False)
    cpu = dev(1, "Working CPU", "CPU", True)
    fake_devices([broken_gpu, cpu])
    assert solver_choice.resolve_solver("gpu-mg") == "gpu-mg"
    assert cli.backend_problem("gpu-mg", None, Settings(1.0, 0.5, 1, 1.0, solver="gpu-mg", gpu_device="1")) is None
    text = cli.backend_problem("gpu-mg", None, Settings(1.0, 0.5, 1, 1.0, solver="gpu-mg"))
    assert "Broken GPU" in text and "not available" in text and "gpu_device" in text
    assert device.select("working", "auto")[0].name == "Working CPU"


def test_the_rule_does_not_check_the_case(fake_devices):
    """A device without 64-bit floats and gpu_case B: the rule says gpu-mg, the problem check says the fact."""
    from topopt.config import Settings
    fake_devices([dev(fp64=False)])
    assert solver_choice.resolve_solver("gpu-mg") == "gpu-mg"
    assert "no 64-bit float" in cli.backend_problem("gpu-mg", None, Settings(1.0, 0.5, 1, 1.0, solver="gpu-mg", gpu_case="B"))


def test_duplicate_devices_are_marked_in_the_facts(fake_devices):
    a = dev(0, "pocl CPU", "CPU", True, platform_name="Portable Computing Language")
    b = dev(1, "pocl CPU", "CPU", True, platform_name="Portable Computing Language")
    c = dev(2, "Other", "GPU", False)
    fake_devices([a, b, c])
    f = device.facts()
    assert [d["duplicate_of"] for d in f["devices"]] == [None, 0, None]
    lines = cli.opencl_lines(f)
    assert "same platform, name and version as device 0" in lines[2] and "same platform" not in lines[1]


def test_device_list_is_asked_once_for_each_process(monkeypatch):
    """Review m1: the rule, the problem check and the model do not ask the OpenCL drivers three times."""
    device.forget_devices()
    calls = []
    monkeypatch.setattr(device, "import_pyopencl", lambda: None)
    monkeypatch.setattr(device, "_list_devices", lambda: calls.append(1) or [dev()])
    for _ in range(4):
        assert device.enumerate_devices()[0].name == "Test GPU"
    assert len(calls) == 1
    first = device.enumerate_devices()
    first[0].fp64 = True                                    # a caller may change its copy
    assert device.enumerate_devices()[0].fp64 is False
    device.forget_devices()
    device.enumerate_devices()
    assert len(calls) == 2
    device.forget_devices()


# --- the memory and time lines of check (review Q7) -------------------------------------------------------


def _check_json(tmp_path, capsys, *flags):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = make_case(tmp_path, h=2.0, iterations=2)
    assert main(["check", str(p), "--json", *flags]) == 0
    return json.loads(capsys.readouterr().out)


def test_check_uses_the_gpu_mg_memory_number_for_the_warning(tmp_path, monkeypatch, fake_devices, capsys):
    """gpu-mg 1 GB, Cholesky 5 GB, RAM 2 GB: gpu-mg fits, the Cholesky model would not."""
    fake_devices([dev(0, "Fast CPU", "CPU", True)])
    monkeypatch.setattr(gpu_check, "estimate_memory", lambda *a, **k: {"bytes": 1_000_000_000, "typical_bytes": 600_000_000, "build_bytes": 400_000_000,
                                                                       "loop_bytes": 500_000_000, "mesh_bytes": 1_000_000_000, "phase": "mesh"})
    monkeypatch.setattr(cli, "estimate_memory_bytes", lambda voxels, dofs=0, filter_bytes=0, bordered_bytes=0: 5_000_000_000)
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: 2_000_000_000)
    d = _check_json(tmp_path / "a", capsys, "--solver", "gpu-mg")
    me = d["memory_estimate"]
    assert me["model"] == "gpu-mg" and me["bytes"] == 1_000_000_000 and me["cholesky_bytes"] == 5_000_000_000
    assert not [w for w in d["warnings"] if w.startswith("the memory estimate")]
    d = _check_json(tmp_path / "b", capsys, "--solver", "amg")           # the Cholesky model for another solver
    assert d["memory_estimate"]["bytes"] == 5_000_000_000 and "model" not in d["memory_estimate"]
    assert [w for w in d["warnings"] if w.startswith("the memory estimate")]
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: 500_000_000)   # now gpu-mg does not fit either
    d = _check_json(tmp_path / "c", capsys, "--solver", "gpu-mg")
    warn = [w for w in d["warnings"] if w.startswith("the memory estimate")]
    assert len(warn) == 1 and "gpu-mg model, measured on pocl" in warn[0]


def test_check_text_for_gpu_mg_has_the_memory_line_and_its_own_time_model(tmp_path, monkeypatch, fake_devices, capsys):
    """P2a: the time lines are the gpu-mg model with its band and its constants; the Cholesky number is not shown as the time."""
    monkeypatch.setenv("TOPOPT_STATE_DIR", str(tmp_path / "state"))
    fake_devices([dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")])
    # `cli.opencl_facts` returns None when pyopencl is not installed (the CI legs have no `gpu` extra), and then the device is "not known".
    monkeypatch.setattr(cli, "opencl_facts", lambda settings=None, solver_setting=None: device.facts(getattr(settings, "gpu_device", "auto")))
    p = make_case(tmp_path, h=2.0, iterations=2)
    assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
    out = capsys.readouterr().out
    assert "none for gpu-mg" not in out and "no time model" not in out
    assert "time model:     gpu-mg, linear in the free dofs: band " in out
    assert "time constants: no constants for this device (Fast GPU, case B); run `topopt gpu-check` to measure them" in out
    assert "memory estimate: about" in out
    assert "memory model:   gpu-mg, measured on pocl with the device buffers in host memory; the cholesky model for comparison: about" in out
    assert out.count("experimental") == 0 and "\ngpu-mg models: the time estimate is about" in out and "warning: gpu-mg models" not in out
    d = json.loads((lambda: (main(["check", str(p), "--solver", "gpu-mg", "--json"]), capsys.readouterr().out)[1])())
    te = d["time_estimate"]
    assert te["model"] == "gpu-mg" and te["basis"] == "reference" and te["low_s"] <= te["total_s"] <= te["high_s"]
    main(["check", str(p)])
    assert "time model" not in capsys.readouterr().out                  # the default solver is not changed


def test_the_constants_of_the_gpu_mg_memory_model_are_pinned():
    """F4-36: the fitted numbers of CONTRACTS.md section 20. A new fit changes them here and in the table of tests/test_gpu_mg_memory.py."""
    assert (gpu_check.GPU_MG_BASE_BYTES, gpu_check.GPU_MG_BYTES_PER_DOF, gpu_check.GPU_MG_JOINT_BYTES_PER_DOF) == (860_000_000, 730, 1000)
    assert (gpu_check.GPU_MG_MESH_BASE_BYTES, gpu_check.GPU_MG_MESH_BYTES_PER_DOF23, gpu_check.GPU_MG_MESH_BYTES_PER_GRID_VOXEL) == (500_000_000, 370_000, 220)
    assert gpu_check.GPU_MG_LAYOUT_FACTOR == 3.2


def test_check_json_and_text_for_gpu_mg_show_the_two_phases_of_the_memory_model(tmp_path, monkeypatch, fake_devices, capsys):
    """F4-36: `memory_estimate` has the loop phase, the mesh phase and the number without the layout factor; the number is made from the dofs and the grid."""
    monkeypatch.setenv("TOPOPT_STATE_DIR", str(tmp_path / "state"))
    fake_devices([dev(0, "Fast CPU", "CPU", True)])
    d = _check_json(tmp_path / "a", capsys, "--solver", "gpu-mg")
    me = d["memory_estimate"]
    fsz = d["derived"]["filter"]["size"]            # the filter is in the build phase and in the loop phase
    want = gpu_check.estimate_memory(d["free_dofs"], d["grid"]["voxels"], d["time_estimate"]["gpu_mg"]["case"], 0,
                                     fsz["held_bytes_estimate"], fsz["build_peak_bytes_estimate"])
    assert {k: me[k] for k in want} == want
    assert me["bytes"] == max(me["build_bytes"], me["loop_bytes"], me["mesh_bytes"]) >= me["typical_bytes"] >= me["loop_bytes"]
    assert me["mesh_bytes"] > gpu_check.GPU_MG_MESH_BASE_BYTES + gpu_check.GPU_MG_MESH_BYTES_PER_GRID_VOXEL * d["grid"]["voxels"] - 1
    assert main(["check", str(next((tmp_path / "a").glob("*.yaml"))), "--solver", "gpu-mg"]) == 0
    out = capsys.readouterr().out
    assert "mesh step of the post-process" in out and "like the hook example" in out and "memory estimate: about" in out


def test_the_loop_phase_of_the_gpu_mg_memory_model_is_linear_and_the_mesh_phase_grows_with_the_free_dofs_to_the_power_two_thirds():
    """F4-36: the peak is the larger of two phases. The loop phase is linear in the free dofs; the mesh step of the post-process follows the surface."""
    loop = [gpu_check.estimate_memory(n)["loop_bytes"] for n in (100_000, 200_000, 300_000)]
    assert (loop[2] - loop[1]) == pytest.approx(loop[1] - loop[0], abs=2) and loop[1] - loop[0] == pytest.approx(100_000 * gpu_check.GPU_MG_BYTES_PER_DOF, abs=2)
    mesh = [gpu_check.estimate_memory(n)["mesh_bytes"] for n in (100_000, 800_000)]
    surface = gpu_check.GPU_MG_LAYOUT_FACTOR * gpu_check.GPU_MG_MESH_BYTES_PER_DOF23
    assert mesh[1] - mesh[0] == pytest.approx(surface * (800_000 ** (2 / 3) - 100_000 ** (2 / 3)), rel=1e-6)    # 8 times the dofs: 4 times the surface
    a, b, c = (gpu_check.estimate_memory_bytes(n) for n in (100_000, 200_000, 300_000))
    assert 0 < a < b < c


# --- the YAML ---------------------------------------------------------------------------------


def _yaml_with(tmp_path, **settings):
    p = make_case(tmp_path, h=2.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(settings)
    p.write_text(yaml.safe_dump(d))
    return p


def test_yaml_accepts_gpu_mg_and_the_gpu_keys(tmp_path):
    st = load_config(_yaml_with(tmp_path, solver="gpu-mg", gpu_device=1, gpu_case="C")).settings
    assert (st.solver, st.gpu_device, st.gpu_case) == ("gpu-mg", "1", "C")
    st = load_config(_yaml_with(tmp_path, solver="gpu-mg", gpu_device="Apple M1")).settings
    assert (st.gpu_device, st.gpu_case) == ("Apple M1", "auto")
    assert load_config(_yaml_with(tmp_path)).settings.gpu_device == "auto"


def test_yaml_checks_the_gpu_keys(tmp_path):
    with pytest.raises(UserError, match=r"settings.gpu_case' must be auto, B, C, got 'D'"):
        load_config(_yaml_with(tmp_path, gpu_case="D"))
    with pytest.raises(UserError, match="settings.gpu_device"):
        load_config(_yaml_with(tmp_path, gpu_device=True))
    with pytest.raises(UserError, match=r"settings.solver. must be auto, cholesky, pardiso, amg or gpu-mg, got .gpu."):
        load_config(_yaml_with(tmp_path, solver="gpu"))


# --- the two numbers of gpu-mg that are settings (G2) -------------------------------------------------


def test_gpu_tol_and_gpu_backward_error_have_their_defaults_and_are_parsed(tmp_path):
    from topopt.config import GPU_BACKWARD_ERROR_DEFAULT, GPU_TOL_DEFAULT
    st = load_config(_yaml_with(tmp_path)).settings
    assert (st.gpu_tol, st.gpu_backward_error) == (1e-6, 1e-15) == (GPU_TOL_DEFAULT, GPU_BACKWARD_ERROR_DEFAULT)
    st = load_config(_yaml_with(tmp_path, solver="gpu-mg", gpu_tol=1e-8, gpu_backward_error=0)).settings
    assert (st.gpu_tol, st.gpu_backward_error) == (1e-8, 0.0) and isinstance(st.gpu_backward_error, float)
    st = load_config(_yaml_with(tmp_path, gpu_tol=0.5, gpu_backward_error=9.9e-7)).settings
    assert (st.gpu_tol, st.gpu_backward_error) == (0.5, 9.9e-7)
    st = load_config(_yaml_with(tmp_path, gpu_tol="1e-7", gpu_backward_error="1e-13")).settings    # YAML 1.1 reads 1e-7 as text
    assert (st.gpu_tol, st.gpu_backward_error) == (1e-7, 1e-13)
    assert load_config(_yaml_with(tmp_path, gpu_tol=1e-300)).settings.gpu_tol == 1e-300      # any number above 0 is a number


@pytest.mark.parametrize("value", [0, 0.0, -1e-6, 1, 1.0, 2, 10.0, "abc", "", True, None, [1e-6], float("nan"), float("inf")])
def test_gpu_tol_refuses_what_is_not_a_number_above_0_and_below_1(tmp_path, value):
    with pytest.raises(UserError, match=r"settings\.gpu_tol' must be"):
        load_config(_yaml_with(tmp_path, gpu_tol=value))


@pytest.mark.parametrize("value", [-1e-14, -1.0, 1e-6, 1e-5, 1, 1.0, "abc", "", True, None, [1e-14], float("nan"), float("inf")])
def test_gpu_backward_error_refuses_what_is_not_a_number_from_0_up_to_but_not_1e_6(tmp_path, value):
    with pytest.raises(UserError, match=r"settings\.gpu_backward_error' must be"):
        load_config(_yaml_with(tmp_path, gpu_backward_error=value))


def test_the_two_gpu_numbers_are_checked_for_any_solver_and_a_wrong_key_is_still_unknown(tmp_path):
    with pytest.raises(UserError, match=r"settings\.gpu_tol' must be"):
        load_config(_yaml_with(tmp_path, solver="pardiso", gpu_tol=0))
    with pytest.raises(UserError, match="unknown key 'gpu_tolerance'"):
        load_config(_yaml_with(tmp_path, gpu_tolerance=1e-6))


def test_a_checkpoint_of_before_G2_still_matches_and_a_changed_number_is_a_change(tmp_path):
    """The effective settings leave out each number at its default (like `perimeter_max_mm2` when unset). A value that is not the
    default is a change of the inputs, and a resume names it."""
    from topopt import checkpoint as C
    base = C.input_hashes(load_config(_yaml_with(tmp_path)))
    assert "gpu_tol" not in base["effective"]["settings"] and "gpu_backward_error" not in base["effective"]["settings"]
    same = C.input_hashes(load_config(_yaml_with(tmp_path, gpu_tol=1e-6, gpu_backward_error=1e-15)))     # written out = default
    assert same["effective"] == base["effective"] and same["effective_sha256"] == base["effective_sha256"]
    new = C.input_hashes(load_config(_yaml_with(tmp_path, solver="gpu-mg", gpu_tol=1e-8)))
    assert new["effective"]["settings"]["gpu_tol"] == 1e-8 and "gpu_backward_error" not in new["effective"]["settings"]
    changes, _ = C.diff_inputs(base, C.input_hashes(load_config(_yaml_with(tmp_path, gpu_tol=1e-8, gpu_backward_error=0))))
    assert "settings.gpu_tol: <not set> -> 1e-08" in changes and "settings.gpu_backward_error: <not set> -> 0" in changes


# --- the CLI ------------------------------------------------------------------------------------


def test_cli_solver_flag_choices():
    parser = cli.build_parser()
    assert parser.parse_args(["run", "x.yaml", "--out", "o", "--solver", "gpu-mg"]).solver == "gpu-mg"
    assert parser.parse_args(["check", "x.yaml", "--solver", "amg"]).solver == "amg"
    assert parser.parse_args(["run", "x.yaml", "--out", "o"]).solver is None
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "x.yaml", "--out", "o", "--solver", "gpu"])


def test_run_with_missing_pyopencl_stops_before_the_solve(tmp_path, monkeypatch, capsys):
    """The flag --solver gpu-mg without pyopencl: exit 1, one-line fact with options, status error. No fallback."""
    monkeypatch.setattr(device, "pyopencl_problem", lambda: ("missing", "No module named 'pyopencl'"))
    p = make_case(tmp_path, h=2.0, iterations=2)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet", "--solver", "gpu-mg"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: solver 'gpu-mg' needs pyopencl") and "Or set settings.solver" in err
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "error"
    assert not (out / "history.jsonl").exists()          # no iteration ran with another solver


def test_check_shows_the_opencl_devices(tmp_path, monkeypatch, fake_devices, capsys):
    fake_devices([dev(0, "Fast CPU", "CPU", True), dev(1, "Small GPU", "GPU", False)])
    monkeypatch.setattr(cli, "opencl_facts", lambda settings=None, solver_setting=None: device.facts())
    p = make_case(tmp_path, h=2.0, iterations=2)
    assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
    out = capsys.readouterr().out
    assert "solver:         gpu-mg" in out
    assert "opencl device:  1: Small GPU (GPU, platform Test Platform, OpenCL 1.2), 64-bit float no" in out
    assert "device 1 (case C)" in out
    assert "\ngpu-mg models: " in out and "warning: gpu-mg models" not in out


def test_check_json_has_the_opencl_facts(tmp_path, monkeypatch, fake_devices, capsys):
    fake_devices([dev(0, "Fast CPU", "CPU", True)])
    monkeypatch.setattr(cli, "opencl_facts", lambda settings=None, solver_setting=None: device.facts())
    p = make_case(tmp_path, h=2.0, iterations=2)
    assert main(["check", str(p), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["opencl"]["devices"][0]["fp64"] is True and d["opencl"]["auto_case"] == "B"
    assert d["solver"]["resolved"] != "gpu-mg" and d["solver"]["cpu_solver"] == d["solver"]["resolved"]      # a CPU device does not count for auto


def test_environment_report_has_the_opencl_facts(monkeypatch, fake_devices):
    from topopt.environment import environment_data
    monkeypatch.delenv("TOPOPT_AUTO_GPU", raising=False)      # the test session sets it to 0: then the facts ask no device list
    fake_devices([dev(0, "Small GPU", "GPU", False)])
    data = environment_data()
    assert data["opencl"]["devices"][0]["name"] == "Small GPU"
    assert data["opencl"]["devices"][0]["fp64"] is False
    assert data["solver_setting"]["gpu-mg"] == {"resolved": "gpu-mg", "error": None} and data["experimental_solvers"] == {}
    assert data["auto_solver"] != "gpu-mg"                                      # the faked GPU cannot pass the self-test
    assert set(data["solver_setting"]) == set(solver_choice.SOLVER_SETTINGS)


def test_environment_report_without_pyopencl(monkeypatch):
    from topopt.environment import environment_data
    in_process_probe(monkeypatch)
    monkeypatch.delenv("TOPOPT_AUTO_GPU", raising=False)      # the test session sets it; `auto` then does not look at pyopencl
    monkeypatch.setattr(device, "pyopencl_problem", lambda: ("missing", "No module named 'pyopencl'"))
    data = environment_data()
    assert data["opencl"]["pyopencl"] is None and "missing" in data["opencl"]["problem"]
    assert "[gpu]" in data["opencl"]["install_command"] or "pyopencl" in data["opencl"]["install_command"]
    assert "needs pyopencl" in data["solver_setting"]["gpu-mg"]["error"] and data["experimental_solvers"] == {}
    gpu = data["gpu_solver"]
    assert gpu["used"] is False and gpu["to_use_the_gpu"].startswith("pyopencl is not installed. Install it with: ") and "pyopencl" in gpu["why"]


def test_install_command_names_the_gpu_extra():
    assert "[gpu]" in device.install_command()
    text = (REPO / "pyproject.toml").read_text()
    assert "gpu = [" in text and "pyopencl==" in text
    core = text.split("[project.optional-dependencies]")[0]
    # pyopencl is a main dependency on an Apple silicon Mac and on Linux x86_64 (one line, with that marker). pocl is a test extra only.
    lines = [l for l in core.splitlines() if l.strip().startswith('"pyopencl==')]
    assert len(lines) == 1 and "platform_system == 'Darwin' and platform_machine == 'arm64'" in lines[0], lines
    assert "platform_system == 'Linux' and platform_machine == 'x86_64'" in lines[0]
    assert "pocl" not in core


