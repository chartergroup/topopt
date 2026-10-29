"""The time model of the solver gpu-mg (P2a, CONTRACTS.md section 20).

The model is in `topopt/gpu_mg/check.py`. These tests need no OpenCL: the devices are faked, and the per-user state folder is
a folder of the test (`TOPOPT_STATE_DIR`). The numbers of the fixtures are rows of the research of 2026-10-22 (not in the repo):
runs of the user tests on an Intel Arc iGPU, case B. A row has no name of the run or of the part, only its size and its settings.
"""
import json
import math
import os
import re
from pathlib import Path

import pytest
import yaml

from test_gpu_mg_device import dev, fake_devices  # noqa: F401  (the fixture and the device factory of the choice tests)
from test_optimize import make_case
from topopt import cli, regions_builder as rb
from topopt.cli import (backend_estimates, check_data, derived_numbers, estimate_seconds, fit_voxel_size, main)
from topopt.config import load_config
from topopt.gpu_mg import check as gpu_check, device

REPO = Path(__file__).resolve().parent.parent

# (row, free dofs, load cases, planned iterations, penal, volume fraction, gpu_tol, wall s, mid band without sharing (the research number),
#  and the low and high of the band when no other gpu-mg run was active (conc 0), else None)
ROWS = [
    ("170k-4cases", 169_920, 4, 70, None, 0.23214, 1e-8, 76.04, 130.15183679999998, (79.55407871999999, 214.48143360000003)),
    ("170k-1case", 169_920, 1, 100, None, 0.2999, 1e-8, 54.7, 95.4040464, (71.77208256, 134.7906528)),
    ("170k-1case-penal3", 169_920, 1, 80, 3.0, 0.22407, 1e-8, 81.26, 62.90846064, None),
    ("408k-1case-penal4", 408_240, 1, 90, 4.0, 0.054852, 1e-8, 282.82, 197.47580004000002, None),
    ("788k-1case-penal4", 787_800, 1, 100, 4.0, 0.0629, 1e-8, 315.95, 406.78076430000004, (351.99833400000006, 473.737068)),
    ("503k-7cases", 503_502, 7, 100, None, 0.15, 1e-8, 445.48, 861.02231796, None),
    ("176k-5cases", 175_680, 5, 200, None, 0.15, 1e-8, 364.34, 463.0207248, (240.87494592, 833.2636895999999)),
    ("609k-1case-penal5", 609_336, 1, 80, 5.0, 0.11247, 1e-8, 359.92, 265.05607719600005, None),
]


INTEL = {"index": 0, "platform": "Intel(R) OpenCL Graphics", "name": "Intel(R) Arc(TM) Graphics", "vendor": "Intel(R) Corporation",
         "kind": "GPU", "fp64": True, "version": "OpenCL 3.0"}
CPU = {"index": 0, "platform": "Portable Computing Language", "name": "pthread-Test CPU", "vendor": "GenuineIntel", "kind": "CPU", "fp64": True,
       "version": "OpenCL 3.0 PoCL"}
APPLE = {"index": 0, "platform": "Apple", "name": "Apple M4", "vendor": "Apple", "kind": "GPU", "fp64": False, "version": "OpenCL 1.2"}


def _private_names():
    import re
    return re.compile("|".join(["[a-z0-9]+/[a-z0-9]*_" + "run[0-9]*", "impl" + "/research", "IMPL" + r"\.md", "tests" + "-[a-z]+(?![a-z-])", "runs" + r"\.csv"]))


def test_the_fixtures_and_the_model_have_no_user_test_names_and_no_internal_path():
    """P2a review B1: a row is named by its size and settings. The research is not in the repo, so no file here names its folders."""
    import re
    bad = _private_names()
    for path in (Path(__file__), REPO / "topopt" / "gpu_mg" / "check.py", REPO / "topopt" / "gpu_check.py"):
        assert not bad.findall(path.read_text(encoding="utf-8")), path.name
    assert all(re.fullmatch(r"\d+k-\d+cases?(-penal\d)?", row[0]) for row in ROWS)


def test_the_name_pattern_finds_a_folder_name_in_each_form():
    folder = "tests" + "-xyz"
    for line in (f"the folder {folder}/ holds", f"the folder `{folder}`", f"it is in {folder}.", f"{folder}", f"{folder}/a.py"):
        assert _private_names().search(line), line
    for line in ("tests-and-limits.md", "tests/test_x.py"):
        assert not _private_names().search(line), line


# --- the formula against the research -----------------------------------------------------------------------------------


@pytest.mark.parametrize("row,n,cases,N,penal,vf,tol,wall,blind,band", ROWS, ids=[r[0] for r in ROWS])
def test_the_model_reproduces_the_research_rows(row, n, cases, N, penal, vf, tol, wall, blind, band):
    """The mid band without sharing is the number of the research, to the last digit; the band is the low and high of the isolated runs."""
    e = gpu_check.estimate_seconds_gpu_mg(n, cases, N, volume_fraction=vf, penal=penal, gpu_tol=tol, device=INTEL)
    assert e.basis == "table" and e.device_name == "Intel(R) Arc(TM) Graphics" and e.case == "B"
    assert e.total_s == pytest.approx(blind, rel=1e-9)
    if band:
        assert (e.low_s, e.high_s) == (pytest.approx(band[0], rel=1e-9), pytest.approx(band[1], rel=1e-9))
    assert e.low_s < e.total_s < e.high_s
    assert 0.35 < e.total_s / wall < 2.2                  # the research range of the first number, for these rows


def test_the_cholesky_number_was_far_off_for_the_same_rows():
    """What the model replaces: the Cholesky number for 787,800 free dofs is 15,000 s for a run that took 316 s."""
    assert estimate_seconds(787_800, 100, "cholesky")[1] > 14_000
    assert estimate_seconds(787_800, 100, "gpu-mg")[1] < 600


def test_the_iteration_bands_and_the_tolerance_factor():
    assert gpu_check.iteration_band(1.9) == (50, 65, 80) and gpu_check.iteration_band(2.5) == (100, 145, 200)
    assert gpu_check.iteration_band(4.99) == (100, 145, 200) and gpu_check.iteration_band(5.0) == (150, 200, 300)
    assert gpu_check.iteration_band(None) == (60, 150, 300)
    assert gpu_check.stiffness_contrast(0.0629, 4.0) == pytest.approx(4.7785, abs=1e-3)       # the research lgK of the 788k row
    assert gpu_check.stiffness_contrast(0.2, None) is None
    # P2a review m1: the bands hold for gpu_tol 1e-8 to 1e-6 with no factor (the default 1e-6 included)
    for tol in (1e-8, 3e-8, 1e-7, 5e-7, 1e-6, None):
        assert gpu_check.tol_factor(tol) == 1.0, tol
    # outside the range the rule of thumb of the pocl runs: 15 percent of the count at 1e-6 for each decade
    assert gpu_check.tol_factor(1e-4) == pytest.approx(0.7) and 0.57 <= gpu_check.tol_factor(1e-4) <= 0.77       # the pocl range
    assert gpu_check.tol_factor(1e-10) == pytest.approx((1 + 0.15 * 4) / (1 + 0.15 * 2))                        # over its value at 1e-8: 1.0 at 1e-8
    assert gpu_check.tol_factor(1e-6 * 1.001) == pytest.approx(1.0, abs=1e-3) and gpu_check.tol_factor(1e-8 * 0.999) == pytest.approx(1.0, abs=1e-3)
    assert gpu_check.tol_factor(0.99) > 0                                         # the factor stays positive for any tolerance of the config


def test_the_default_tolerance_gives_the_numbers_of_the_research_and_the_text_names_the_range():
    kw = dict(volume_fraction=0.1, penal=4.0, device=INTEL)
    default = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, **kw)
    assert default.total_s == gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, gpu_tol=1e-8, **kw).total_s       # was 1.3 times lower at 1e-6
    assert default.iterations_per_block == (100, 145, 200) and default.tol_factor == 1.0
    assert "gpu_tol 1e-06 is from 1e-08 to 1e-06, where the bands hold: no factor" in " ".join(default.assumptions)
    loose = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, gpu_tol=1e-4, **kw)
    text = " ".join(loose.assumptions)
    assert loose.iterations_per_block == pytest.approx((70, 101.5, 140)) and loose.total_s < default.total_s
    assert "gpu_tol 0.0001 is outside 1e-08 to 1e-06: the iterations are scaled by 0.70" in text
    assert "pocl runs on synthetic designs, not from real runs" in text


def test_the_text_says_that_the_band_is_not_a_limit():
    """The band is the spread of the iteration counts of the fit runs: 19 of 49 runs inside, 14 below, 16 above (15 of them shared the GPU)."""
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 1, 80, volume_fraction=0.1, penal=4.0, device=INTEL)
    text = " ".join(e.assumptions)
    assert "not a limit" in text and "of the 49 runs of the fit the time was inside the band for 19, below it for 14 and above it for 16" in text
    assert "(15 of those shared the GPU)" in text and gpu_check.FIT_INSIDE + gpu_check.FIT_BELOW + gpu_check.FIT_ABOVE == gpu_check.FIT_RUNS


def test_the_assumptions_say_that_springs_joints_and_the_first_pass_cap_are_not_in_the_fit():
    """P2a review (merge items with P2b): the model is from runs with none of them; the text says so for every device."""
    line = "the springs and joints kernels of gpu-mg, the first-pass cap of 1000 iterations (P2b) and per-axis supports (P2c) are not in the fit"
    for device in (INTEL, APPLE, CPU, None):
        e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=device)
        assert line in e.assumptions and line in e.as_dict()["assumptions"]
    d = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=APPLE).as_dict()
    warn = gpu_check.apply({"solver": {"resolved": "gpu-mg"}, "memory_estimate": {"ram_bytes": 0, "bytes": 1}, "free_dofs": 1000, "warnings": [],
                            "time_estimate": {"gpu_mg": d, "total_s": d["total_s"]}})["warnings"]
    assert line in [w for w in warn if w.startswith("gpu-mg models: ")][0]


def test_the_void_stiffness_is_the_one_of_the_optimizer():
    from topopt.optimize import E_MIN
    assert gpu_check.VOID_STIFFNESS == E_MIN


@pytest.mark.parametrize("cases,blocks,columns", [(1, 1, 1), (2, 1, 2), (3, 2, 2), (4, 2, 2), (7, 4, 2)])
def test_two_load_cases_share_a_launch(cases, blocks, columns):
    e = gpu_check.estimate_seconds_gpu_mg(100_000, cases, 50, volume_fraction=0.2, penal=3.0, device=INTEL)
    assert (e.blocks, e.columns_per_launch) == (blocks, columns)
    k = gpu_check.TIME_CONSTANTS["intel-arc-igpu"]
    its = e.iterations_per_block[1]
    per_pass = blocks * its * (k.c1_s if columns == 1 else k.c2_s) * 100_000 + k.host_s_per_dof * 100_000
    assert e.s_per_iteration == pytest.approx(per_pass)
    assert e.total_s == pytest.approx(k.t0_s + k.t0_s_per_dof * 100_000 + 51 * per_pass)


def test_the_model_is_linear_in_the_free_dofs_and_the_voxel_aim_inverts_it():
    a = gpu_check.estimate_seconds_gpu_mg(200_000, 2, 80, volume_fraction=0.2, penal=3.0, device=INTEL).total_s
    b = gpu_check.estimate_seconds_gpu_mg(400_000, 2, 80, volume_fraction=0.2, penal=3.0, device=INTEL).total_s
    fixed = gpu_check.fixed_seconds()
    assert (b - fixed) / (a - fixed) == pytest.approx(2.0)                        # the power of the model is 1
    inputs = gpu_check.TimeInputs(n_cases=2, volume_fraction=0.2, penal=3.0, device=INTEL, case="B")
    n = gpu_check.free_dofs_for_seconds(120.0, 80, inputs)
    assert gpu_check.estimate_seconds_gpu_mg(n, 2, 80, volume_fraction=0.2, penal=3.0, device=INTEL).total_s == pytest.approx(120.0, abs=0.01)
    assert gpu_check.GPU_MG_DOF_POWER == 1.0 and cli.DOF_POWER == 2.4


def test_sharing_and_a_calibration_scale_the_estimate():
    base = gpu_check.estimate_seconds_gpu_mg(300_000, 1, 80, volume_fraction=0.1, penal=4.0, device=INTEL)
    shared = gpu_check.estimate_seconds_gpu_mg(300_000, 1, 80, volume_fraction=0.1, penal=4.0, device=INTEL, k_other=1.0)
    assert shared.total_s > base.total_s and "1 other gpu-mg runs" in " ".join(shared.assumptions)
    assert "no other gpu-mg run" in " ".join(base.assumptions)
    c = gpu_check.constants_for(INTEL, "B")
    slow = gpu_check.estimate_seconds_gpu_mg(300_000, 1, 80, volume_fraction=0.1, penal=4.0, device=INTEL,
                                             constants=gpu_check.Constants(**{**c.__dict__, "c1_s": 2 * c.c1_s, "host_s_per_dof": 0.0}))
    host = c.host_s_per_dof * 300_000
    assert slow.s_per_iteration == pytest.approx(2 * (base.s_per_iteration - host))       # twice the device part, no host part


# --- the device class, and a device with no row ---------------------------------------------------------------------------


def test_the_intel_arc_igpu_has_a_row_and_other_devices_have_none():
    assert gpu_check.device_class(INTEL, "B") == "intel-arc-igpu" and gpu_check.device_class(INTEL, None) == "intel-arc-igpu"
    assert gpu_check.device_class(INTEL, "C") is None                              # the row is for case B
    assert gpu_check.device_class({**INTEL, "name": "Intel(R) Arc(TM) A770 Graphics"}, "B") is None   # a card, not the iGPU
    assert gpu_check.device_class({**INTEL, "kind": "CPU"}, "B") is None
    assert gpu_check.device_class(APPLE, "C") is None and gpu_check.device_class(None, None) is None
    assert gpu_check.device_case(APPLE) == "C" and gpu_check.device_case(INTEL) == "B" and gpu_check.device_case(APPLE, "B") == "B"
    assert gpu_check.device_case(None) is None


def test_a_device_with_no_row_says_so_and_names_the_reference():
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=APPLE)
    assert e.basis == "reference"
    ref = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=INTEL, case="B")
    assert e.total_s == pytest.approx(ref.total_s)                                 # the numbers of the known device, named as such
    text = gpu_check.basis_text(e.as_dict())
    assert "no constants for this device (Apple M4, case C); run `topopt gpu-check` to measure them" in text
    assert "constants of the Intel Arc iGPU as a reference" in text and "faster or slower" in text
    assert "the iteration bands are from case B runs" in e.assumptions
    unknown = gpu_check.basis_text(gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80).as_dict())
    assert "the device is not known here" in unknown and "topopt gpu-check" in unknown


def test_the_reference_is_the_right_order_for_the_first_real_gpu_of_another_vendor():
    """The M4 rows of `environment.GPU_MG_MEASURED` (hook, 2 load cases, 20 iterations, measured at gpu_tol 1e-8, case C): the reference
    number (the Intel Arc iGPU constants) is 1.4 to 2.1 times the Mac number. It is on the safe side, and the text says that it is a reference."""
    from topopt.environment import GPU_MG_MEASURED
    ratios = []
    for row in GPU_MG_MEASURED["rows"]:
        e = gpu_check.estimate_seconds_gpu_mg(row["free_dofs"], 2, 20, volume_fraction=0.2, penal=3.0, gpu_tol=1e-8, device=APPLE)
        ratios.append(e.s_per_iteration / row["gpu_mg_s"])
    assert [round(r, 1) for r in ratios] == [1.4, 1.6, 2.1], ratios


def test_a_cpu_device_gets_no_reference_number_and_the_text_says_so():
    """P2a review M2: the Intel numbers are 2.5 to 5 times too low for a CPU OpenCL device (the pocl bench), so no seconds are given."""
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=CPU)
    assert e.basis == "none" and e.total_s is None and e.low_s is None and e.high_s is None and e.s_per_iteration is None
    d = e.as_dict()
    assert d["total_s"] is None and d["low_s"] is None and d["s_per_iteration"] is None and d["basis"] == "none"
    assert d["iterations_per_block"]["mid"] == 65 and "pthread-Test CPU" in d["device"]          # the iteration band is not seconds: it stays
    assert "no constants for this device (CPU device); run `topopt gpu-check` to measure them" in gpu_check.basis_text(d)
    assert "no seconds are given for this device" in " ".join(e.assumptions)
    json.dumps(d)
    assert gpu_check.free_dofs_for_seconds(120.0, 80, gpu_check.TimeInputs(device=CPU)) is None
    # the same problem on a GPU of another vendor still gets the reference number
    assert gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=APPLE).total_s > 0
    # the device that is not known at all is not a CPU: it keeps the labelled reference
    assert gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=None).basis == "reference"


def test_a_cpu_device_with_a_calibration_has_numbers():
    """`topopt gpu-check --device N` on a CPU device saves its constants: then there is a measured number, and the basis says so."""
    calibrated(c_s=90e-9, host=1.5e-6, device=CPU, case="B")
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=CPU)
    assert e.basis == "calibration" and e.total_s > 0 and e.low_s < e.high_s


def test_the_callers_take_none_for_a_cpu_device():
    gpu = gpu_check.TimeInputs(n_cases=2, volume_fraction=0.2, penal=3.0, device=CPU, case="B")
    assert estimate_seconds(300_000, 80, "gpu-mg", gpu=gpu) == (None, None)
    est = backend_estimates(300_000, 80, 2, 90_000, gpu=gpu)["gpu-mg"]
    assert est["total_s"] is None and est["low_s"] is None and "stress_total_s" not in est and est["gpu_mg"]["basis"] == "none"
    assert rb.target_active(0.37, 80, "gpu-mg", gpu) == rb.CAP_ACTIVE                          # no time goal: the cap of the aim decides


def test_check_for_gpu_mg_on_a_cpu_device_gives_no_seconds(tmp_path, monkeypatch, fake_devices, capsys):
    fake_devices([dev(0, "Fast CPU", "CPU", True)])
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, h=1.0, iterations=40, cases=2, stop_rules={"time_budget_s": 1.0})   # any number would be above this: there must be no time warning at all
    d = check_data(str(p), solver_setting="gpu-mg")
    te = d["time_estimate"]
    assert te["total_s"] is None and te["low_s"] is None and te["basis"] == "none" and te["model"] == "gpu-mg" and te["s_per_iteration"] is None
    assert not [w for w in d["warnings"] if w.startswith("the gpu-mg time estimate")]
    one = [w for w in d["warnings"] if w.startswith("gpu-mg models: ")]
    assert len(one) == 1 and "there is no time estimate for this device" in one[0] and "no constants for this device (CPU device)" in one[0]
    assert "about" not in one[0].split("Constants:")[0]
    assert d["derived"]["time_estimate_s"] is None and d["derived"]["time_estimate_band_s"] is None
    assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
    out = capsys.readouterr().out
    assert "time estimate:  none: no constants for this device (CPU device); run `topopt gpu-check` to measure them" in out
    assert "time model:     gpu-mg, linear in the free dofs: no seconds for this device" in out and "s/iteration" not in out.split("memory estimate")[0]
    assert main(["check", str(p), "--solver", "gpu-mg", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["time_estimate"]["total_s"] is None and j["time_estimate"]["basis"] == "none" and j["time_estimate"]["gpu_mg"]["total_s"] is None


def test_check_yaml_of_the_builder_gives_no_seconds_for_a_cpu_device(tmp_path, monkeypatch, fake_devices):
    fake_devices([dev(0, "Fast CPU", "CPU", True)])
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, h=1.0, iterations=80, cases=2, solver="gpu-mg", stop_rules={"time_budget_s": 1.0})
    lines = []
    ok, kind = rb.check_yaml(p, lambda *a: lines.append(" ".join(str(x) for x in a)))
    text = "\n".join(lines)
    assert ok and "time estimate: none for gpu-mg on this device" in text and "no constants for this device (CPU device)" in text
    assert "WARNING: the gpu-mg estimate" not in text and "band " not in text.split("time basis")[0]


def test_the_doctor_says_none_for_a_time_estimate_with_no_seconds():
    from topopt import doctor
    d = doctor.collect(imports=False)
    d["environment"]["problem"] = {"ok": True, "yaml": "x.yaml", "free_dofs": 100, "solver": {"resolved": "gpu-mg"}, "time_estimate": {"total_s": None}}
    assert "time estimate none (no constants for this device)" in doctor.render(d)
    d["environment"]["problem"]["time_estimate"]["total_s"] = 47.4
    assert "time estimate 47 s" in doctor.render(d)


# --- the calibration of `topopt gpu-check` -------------------------------------------------------------------------------------


def bench_report():
    """A trimmed report of `tools/bench_gpu_mg.py` on pocl (the sample `tests/data/bench_gpu_mg_pocl_sample.json`), with the gpu-mg run only. `n_load_cases` is the key that
    the bench writes since P2a (the hook has 2 load cases)."""
    d = json.loads((REPO / "tests/data/bench_gpu_mg_pocl_sample.json").read_text())
    d["runs"] = [dict(r, n_load_cases=2) for r in d["runs"] if r["solver"] == "gpu-mg"]
    return d


def test_the_calibration_comes_from_the_bench_report_of_gpu_check():
    rep = bench_report()
    rec, why = gpu_check.calibration_from_bench(rep, "9.9", "a test")
    assert why is None and rec["case"] == "B" and rec["columns_per_launch"] == 2 and rec["topopt_version"] == "9.9"
    run = rep["runs"][0]
    n = run["free_dofs"]
    per = [r["solve_s"] / (sum(i["executed"] for i in r["iterations"]) * n) for r in run["gpu_mg"]["per_iteration"]]
    assert rec["c_s_per_dof_iteration"] == pytest.approx(sorted(per)[len(per) // 2])          # the median of the optimizer iterations
    assert rec["host_s_per_dof"] == pytest.approx((run["median_iter_s"] - run["gpu_mg"]["median_s"]["solve_s"]) / n)
    assert rec["device"]["name"] == rep["opencl"]["device"]["name"] and len(rec["points"]) == 1
    assert 1e-9 < rec["c_s_per_dof_iteration"] < 1e-6                                           # nanoseconds for each free dof and iteration


def test_two_sizes_are_combined_by_their_free_dofs():
    rep = bench_report()
    small = rep["runs"][0]
    big = json.loads(json.dumps(small))
    big.update(h=1.0, free_dofs=2 * small["free_dofs"], tag="big")
    for r in big["gpu_mg"]["per_iteration"]:
        r["solve_s"] *= 4.0                       # twice the dofs and twice the cost for each dof: c is 2 times the one of the small size
    big["median_iter_s"], big["gpu_mg"]["median_s"]["solve_s"] = small["median_iter_s"], small["gpu_mg"]["median_s"]["solve_s"]
    rep["runs"] = [small, big]
    rec, _ = gpu_check.calibration_from_bench(rep)
    c1 = rec["points"][0]["c_s"]
    assert rec["points"][1]["c_s"] == pytest.approx(2 * c1)
    assert rec["c_s_per_dof_iteration"] == pytest.approx((c1 * 1 + 2 * c1 * 2) / 3)


def test_a_bench_report_that_cannot_calibrate_says_why():
    rep = bench_report()
    del rep["runs"][0]["n_load_cases"]
    assert gpu_check.calibration_from_bench(rep)[0] is None and "no load-case count" in gpu_check.calibration_from_bench(rep)[1]
    rep = bench_report()
    rep["runs"][0]["status"] = "error"
    assert "nothing to calibrate" in gpu_check.calibration_from_bench(rep)[1]
    rep = bench_report()
    rep["opencl"] = {"device": None}
    assert "no OpenCL device" in gpu_check.calibration_from_bench(rep)[1]


def calibrated(c_s=40e-9, host=1.0e-6, device=APPLE, case="C", columns=2):
    rec = {"time_utc": "2026-10-23T08:00:00Z", "topopt_version": "t", "device": {k: device[k] for k in ("platform", "name", "vendor", "kind", "fp64")},
           "case": case, "columns_per_launch": columns, "c_s_per_dof_iteration": c_s, "host_s_per_dof": host, "points": [], "source": "test"}
    gpu_check.save_calibration(rec)
    return rec


def test_the_calibration_text_says_what_was_measured_and_what_was_not(state_folder):
    """P2a review m3: it said "c and the host part are measured". The other column count comes from the Intel ratio, and the host part is
    measured at the sizes of the bench only."""
    rec = calibrated(c_s=30e-9, host=1.0e-6, device=APPLE, case="C", columns=2)
    rec = {**rec, "points": [{"free_dofs": 205_000}, {"free_dofs": 115_836}, {"free_dofs": "x"}, "junk"]}
    gpu_check.save_calibration(rec)
    k = gpu_check.constants_for(APPLE, "C")
    assert k.basis == "calibration"
    for text in (k.source, k.short):
        assert "c is measured for 2 columns for each launch; the value for 1 column is c / 1.52, the ratio of the Intel Arc iGPU fit, not measured on this device" in text
        assert "The host part is measured at 115,836 to 205,000 free dofs only, and is used for every size" in text
    assert "c and the host part are measured" not in k.source
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=APPLE)
    assert "measured by `topopt gpu-check`" in gpu_check.basis_text(e.as_dict()) and "115,836 to 205,000 free dofs only" in gpu_check.basis_text(e.as_dict())
    # one column measured: the other count is c times the ratio; one size, and no usable points
    gpu_check.save_calibration({**rec, "columns_per_launch": 1, "points": [{"free_dofs": 170_000}], "time_utc": "2026-10-24T08:00:00Z"})
    assert "the value for 2 columns is c * 1.52" in gpu_check.constants_for(APPLE, "C").short
    assert "measured at 170,000 free dofs only" in gpu_check.constants_for(APPLE, "C").short
    gpu_check.save_calibration({**rec, "points": "none", "time_utc": "2026-10-25T08:00:00Z"})
    assert "measured at the sizes of the bench only" in gpu_check.constants_for(APPLE, "C").short


def test_the_calibration_file_is_read_and_a_missing_file_gives_the_reference(state_folder):
    assert not state_folder.exists()
    assert gpu_check.load_calibrations() == ([], None)                         # missing: no error
    assert gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=APPLE).basis == "reference"
    calibrated()
    assert (state_folder / "gpu-constants.json").is_file() and gpu_check.calibration_path() == state_folder / "gpu-constants.json"
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=APPLE)
    assert e.basis == "calibration"
    ref = gpu_check.TIME_CONSTANTS["intel-arc-igpu"]
    per_pass = 1 * e.iterations_per_block[1] * 40e-9 * 300_000 + 1.0e-6 * 300_000
    assert e.s_per_iteration == pytest.approx(per_pass)                           # c of two columns and the host part are the measured ones
    assert e.total_s == pytest.approx(ref.t0_s + ref.t0_s_per_dof * 300_000 + 81 * per_pass)   # the start and end are those of the reference
    text = gpu_check.basis_text(e.as_dict())
    assert "constants of this machine" in text and "`topopt gpu-check` on 2026-10-23 08:00:00 UTC" in text
    one = gpu_check.estimate_seconds_gpu_mg(300_000, 1, 80, volume_fraction=0.2, penal=3.0, device=APPLE)
    assert one.constants.c1_s == pytest.approx(40e-9 / gpu_check.TWO_COLUMN_RATIO)    # one column from two columns, by the ratio of the Intel fit


def test_a_calibration_is_for_its_device_and_its_case():
    calibrated()
    assert gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=APPLE, case="B").basis == "reference"             # another case
    assert gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device={**APPLE, "name": "Apple M5"}).basis == "reference"   # another device
    assert gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=INTEL).basis == "table"                            # the row of the table
    calibrated(c_s=10e-9, device=INTEL, case="B")                                                                      # a measurement wins over the table
    assert gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=INTEL).basis == "calibration"


def test_the_newest_record_of_a_device_replaces_the_older_one(state_folder):
    calibrated(c_s=40e-9)
    calibrated(c_s=30e-9)
    calibrated(c_s=20e-9, device=INTEL, case="B")
    records, problem = gpu_check.load_calibrations()
    assert problem is None and sorted(r["c_s_per_dof_iteration"] for r in records) == [20e-9, 30e-9]
    assert not list(state_folder.glob("*.tmp"))


def test_the_calibration_write_is_flushed_with_a_name_of_its_own_and_replaces_the_file(state_folder, monkeypatch):
    """P2a review m4: the pattern of `checkpoint.write_checkpoint`: unique tmp in the same folder, flush, `os.replace`, flush of the folder."""
    events = []
    real_replace = os.replace
    monkeypatch.setattr(gpu_check, "_flush", lambda fd: events.append("file"))
    monkeypatch.setattr(gpu_check, "_flush_folder", lambda path: events.append(("folder", Path(path))))

    def replace(src, dst):
        events.append(("replace", Path(src).name, Path(dst).name))
        real_replace(src, dst)
    monkeypatch.setattr(os, "replace", replace)
    calibrated()
    calibrated(c_s=30e-9)
    names = [e[1] for e in events if e[0] == "replace"]
    assert len(names) == 2 and names[0] != names[1] and all(n.startswith("gpu-constants.json.") and n.endswith(".tmp") for n in names)
    assert names[0] != "gpu-constants.json.tmp"                                  # not the fixed name of the first version
    assert events[:3] == ["file", ("replace", names[0], "gpu-constants.json"), ("folder", state_folder)]    # file, then rename, then folder
    assert sorted(p.name for p in state_folder.iterdir()) == ["gpu-constants.json"]       # no tmp file is left
    assert (state_folder / "gpu-constants.json").stat().st_size > 0 and gpu_check.load_calibrations()[1] is None


def test_a_write_that_fails_leaves_the_old_file_and_no_tmp_file(state_folder, monkeypatch):
    calibrated(c_s=40e-9)
    before = (state_folder / "gpu-constants.json").read_text()

    def broken(src, dst):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        calibrated(c_s=10e-9)
    assert (state_folder / "gpu-constants.json").read_text() == before
    assert sorted(p.name for p in state_folder.iterdir()) == ["gpu-constants.json"]


@pytest.mark.parametrize("text,why", [("{not json", "is not a calibration file"), ('{"schema": 7, "calibrations": []}', "schema 7"),
                                      ('{"schema": 1}', "KeyError")])
def test_a_calibration_file_that_cannot_be_read_is_ignored_with_a_note(state_folder, text, why):
    state_folder.mkdir(parents=True)   # alone, no other test has made the state folder yet
    (state_folder / "gpu-constants.json").write_text(text)
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=APPLE)
    assert e.basis == "reference"
    note = e.constants.problem
    assert note and why in note and "it is ignored" in note and "Note: " in gpu_check.basis_text(e.as_dict())


BAD_RECORDS = {      # the three crashes of the review, and more: field that is named, record
    "host_s_per_dof": {"host_s_per_dof": None},
    "columns_per_launch": {"columns_per_launch": "two"},
    "device": {"device": "Apple M4"},
    "device.name": {"device": {"platform": "Apple", "name": ""}},
    "case": {"case": 7},
    "c_s_per_dof_iteration": {"c_s_per_dof_iteration": float("nan")},
    "c_s_per_dof_iteration_zero": {"c_s_per_dof_iteration": 0},
    "host_negative": {"host_s_per_dof": -1e-6},
    "columns_three": {"columns_per_launch": 3},
    "c_text": {"c_s_per_dof_iteration": "40e-9"},
}


def write_file(state_folder, records, raw=None):
    state_folder.mkdir(parents=True, exist_ok=True)
    path = state_folder / "gpu-constants.json"
    path.write_bytes(raw if raw is not None else json.dumps({"schema": 1, "calibrations": records}).encode())
    return path


@pytest.mark.parametrize("name", list(BAD_RECORDS))
def test_a_bad_record_is_dropped_with_a_note_that_names_the_file_and_the_field(state_folder, name):
    """P2a review M1: one bad record in the file stopped `check`, `check_problem`, the builder and every run (TypeError, ValueError, AttributeError)."""
    good = {"time_utc": "2026-10-23T08:00:00Z", "topopt_version": "t", "device": {k: APPLE[k] for k in ("platform", "name", "vendor", "kind", "fp64")},
            "case": "C", "columns_per_launch": 2, "c_s_per_dof_iteration": 40e-9, "host_s_per_dof": 1.0e-6, "points": [], "source": "test"}
    other = {**good, "device": {**good["device"], "name": "Another GPU"}}
    path = write_file(state_folder, [{**good, **BAD_RECORDS[name]}, other])
    records, note = gpu_check.load_calibrations()
    assert len(records) == 1 and records[0]["device"]["name"] == "Another GPU"            # the good record of the file is kept
    assert str(path) in note and "1 of 2 records ignored" in note and "record 0:" in note
    field = {"host_negative": "host_s_per_dof", "c_s_per_dof_iteration_zero": "c_s_per_dof_iteration", "columns_three": "columns_per_launch",
             "c_text": "c_s_per_dof_iteration"}.get(name, name)
    assert f"`{field}" in note
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=APPLE)          # no exception
    assert e.basis == "reference" and path.name in e.constants.problem and "Note: " in gpu_check.basis_text(e.as_dict())


@pytest.mark.parametrize("raw", [b"\xff\xfe\x00 not utf-8", b"[1, 2]", b'{"schema": 1, "calibrations": {"a": 1}}', b'{"schema": 1, "calibrations": [1, "x", null]}', b""])
def test_a_file_that_is_not_text_json_or_a_list_of_records_is_ignored_with_a_note(state_folder, raw):
    path = write_file(state_folder, None, raw)
    records, note = gpu_check.load_calibrations()
    assert records == [] and str(path) in note
    assert gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, device=APPLE).basis == "reference"


def test_the_record_guard_in_constants_for_falls_back_when_a_bad_record_gets_through(state_folder, monkeypatch):
    """The second guard: even if a record that `record_problem` would drop reaches `constants_for`, there is no exception."""
    bad = {"device": APPLE, "case": "C", "columns_per_launch": 2, "c_s_per_dof_iteration": 40e-9, "host_s_per_dof": None}
    monkeypatch.setattr(gpu_check, "find_calibration", lambda device, case: (bad, None))
    k = gpu_check.constants_for(APPLE, "C")
    assert k.basis == "reference" and "the record of this device cannot be used (TypeError" in k.problem and "gpu-constants.json" in k.problem
    k = gpu_check.constants_for(INTEL, "B")
    assert k.basis == "table" and k.problem                                          # the table row wins over a bad record
    bad.update(host_s_per_dof=1e-6, columns_per_launch="two")
    assert gpu_check.constants_for(APPLE, "C").basis == "reference"


def test_a_bad_record_does_not_stop_check_derived_numbers_or_the_builder(tmp_path, monkeypatch, fake_devices, state_folder, capsys):
    """`derived_numbers` (run.json, `RunWriter.write_run_json`) called the model with no guard: one bad record stopped every gpu-mg run."""
    fake_devices([dev(0, "Apple M4", "GPU", False, platform_name="Apple", vendor="Apple")])
    _facts(monkeypatch)
    good = {"time_utc": "t", "device": {k: APPLE[k] for k in ("platform", "name")}, "case": "C", "columns_per_launch": 2,
            "c_s_per_dof_iteration": 40e-9, "host_s_per_dof": 1e-6}
    path = write_file(state_folder, [{**good, "host_s_per_dof": None}, {**good, "device": "Apple M4"}, {**good, "columns_per_launch": "two"}])
    p = _gpu_case(tmp_path, h=1.0, iterations=40, cases=2)
    cfg = load_config(p)
    d = check_data(str(p), solver_setting="gpu-mg")
    assert d["time_estimate"]["basis"] == "reference" and "3 of 3 records ignored" in d["time_estimate"]["note"] and str(path) in d["time_estimate"]["note"]
    der = derived_numbers(cfg, tuple(d["grid"]["shape"]), d["grid"]["voxel_mm"], d["active_voxels"], d["free_design_voxels"], d["free_dofs"], "gpu-mg", None)
    assert der["time_estimate_s"] > 0
    assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
    out = capsys.readouterr().out
    assert "Note: " in out and "3 of 3 records ignored" in out
    spec = {"load_cases": [{"name": "a"}], "settings": {"solver": "gpu-mg"}}
    assert rb.spec_time_model(spec, 0.1)[0] == "gpu-mg"                                  # the builder takes the model too
    ok, kind = rb.check_yaml(p, lambda *a: None)
    assert ok


# --- the places that use the model ------------------------------------------------------------------------------------------------


def _gpu_inputs(**kw):
    return gpu_check.TimeInputs(**{"n_cases": 2, "volume_fraction": 0.2, "penal": 3.0, "device": INTEL, "case": "B", **kw})


def test_the_cholesky_path_is_unchanged():
    per_it, total = estimate_seconds(109_548, 80)
    assert per_it == pytest.approx(1.30) and total == pytest.approx(10 + 1.30 * 81 + 15)
    assert estimate_seconds(200_000, 80, "pardiso") == estimate_seconds(200_000, 80, "cholesky")
    assert set(backend_estimates(200_000, 80)) == {"cholesky", "pardiso", "amg"}
    # fit_voxel_size: the old exponent 1 / (3 * 2.4) and the old fixed part 25 s
    assert fit_voxel_size(1.0, 25 + 4 * 215.0, 240.0) == math.ceil(1.0 * 4 ** (1 / 7.2) * 1.02 * 100) / 100
    assert fit_voxel_size(1.0, 100.0, 240.0, solver="amg") is None
    assert rb.target_active(0.05, 80) == rb.CAP_ACTIVE and 30000 < rb.target_active(0.37, 80) < 40000


def test_estimate_seconds_has_a_gpu_mg_branch_that_uses_the_model():
    gpu = _gpu_inputs()
    per_it, total = estimate_seconds(300_000, 80, "gpu-mg", gpu=gpu)
    e = gpu_check.estimate_seconds_gpu_mg(300_000, 2, 80, volume_fraction=0.2, penal=3.0, device=INTEL, case="B")
    assert (per_it, total) == (e.s_per_iteration, e.total_s) and total < estimate_seconds(300_000, 80, "cholesky")[1] / 10
    assert estimate_seconds(300_000, 80, "gpu-mg")[1] > 0                              # no inputs: one load case, the reference constants
    # stress rows and robust mode are added with the terms of the other solvers
    plain = estimate_seconds(300_000, 80, "gpu-mg", gpu=gpu)[1]
    assert estimate_seconds(300_000, 80, "gpu-mg", 2, 90_000, gpu=gpu)[1] > plain
    assert estimate_seconds(300_000, 80, "gpu-mg", robust_mode=True, gpu=gpu)[1] > plain


def test_backend_estimates_has_the_gpu_mg_entry_only_with_inputs():
    est = backend_estimates(300_000, 80, gpu=_gpu_inputs())
    g = est["gpu-mg"]
    assert set(est) == {"cholesky", "pardiso", "amg", "gpu-mg"} and g["low_s"] < g["total_s"] < g["high_s"]
    assert g["total_s"] == estimate_seconds(300_000, 80, "gpu-mg", gpu=_gpu_inputs())[1]
    assert g["gpu_mg"]["basis"] == "table" and g["gpu_mg"]["iterations_per_block"]["mid"] == pytest.approx(65)
    with_rows = backend_estimates(300_000, 80, 2, 90_000, gpu=_gpu_inputs())["gpu-mg"]
    assert with_rows["stress_total_s"] > 0 and with_rows["stress_rows"] == 2


def test_fit_voxel_size_takes_the_power_of_the_model_in_use():
    h, fixed = 1.0, gpu_check.fixed_seconds()
    total, limit = fixed + 8 * 100.0, fixed + 100.0                         # 8 times the time above the fixed part
    assert fit_voxel_size(h, total, limit, solver="gpu-mg") == math.ceil(8 ** (1 / 3) * 1.02 * 100) / 100       # the power 1: h grows as the cube root
    old = fit_voxel_size(h, 25 + 8 * 215.0, 240.0)                                        # the power 2.4 and the fixed part 25 s
    assert old == math.ceil(8 ** (1 / 7.2) * 1.02 * 100) / 100 and fit_voxel_size(h, total, limit, solver="gpu-mg") > old
    assert fit_voxel_size(h, fixed + 50.0, limit, solver="gpu-mg") is None


def test_the_voxel_aim_uses_the_gpu_mg_model():
    gpu = _gpu_inputs()
    cap = rb.CAP_ACTIVE
    assert rb.target_active(0.37, 80, "gpu-mg", gpu) == cap                # the time goal does not bind: the cap of the aim does
    assert rb.target_active(0.37, 80) < cap                                # for the direct solvers it does (the old model)
    # a time goal that binds for gpu-mg: the free dofs, and so the active voxels, grow in proportion to the time above the fixed part
    n1 = gpu_check.free_dofs_for_seconds(gpu_check.fixed_seconds() + 20.0, 80, gpu)
    n2 = gpu_check.free_dofs_for_seconds(gpu_check.fixed_seconds() + 40.0, 80, gpu)
    assert n2 == pytest.approx(2 * n1, rel=1e-3)


def test_the_builder_aim_for_gpu_mg_follows_the_spec(monkeypatch, fake_devices):
    fake_devices([dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")])
    spec = {"load_cases": [{"name": "a"}, {"name": "b"}, {"name": "c"}], "settings": {"solver": "gpu-mg", "penal": 4.0}}
    solver, gpu = rb.spec_time_model(spec, 0.1)
    assert solver == "gpu-mg" and (gpu.n_cases, gpu.penal, gpu.volume_fraction) == (3, 4.0, 0.1) and gpu.device["name"] == "Fast GPU"
    assert rb.spec_time_model({"load_cases": [], "settings": {"solver": "auto"}}, 0.1) == ("cholesky", None)
    assert rb.spec_time_model({"load_cases": []}, 0.1) == ("cholesky", None)
    fake_devices([])                                                       # gpu-mg cannot run here: the builder says so in check_yaml and uses the old model
    monkeypatch.setattr(device, "forget_devices", lambda: None)
    assert rb.spec_time_model(spec, 0.1) == ("cholesky", None)


def _gpu_case(tmp_path, h=2.0, iterations=2, cases=1, **settings):
    p = make_case(tmp_path, h=h, iterations=iterations)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(settings)
    if cases > 1:
        d["load_cases"] = [dict(d["load_cases"][0], name=f"c{i}") for i in range(cases)]
    p.write_text(yaml.safe_dump(d))
    return p


def _facts(monkeypatch):
    monkeypatch.setattr(cli, "opencl_facts", lambda settings=None, solver_setting=None: device.facts(getattr(settings, "gpu_device", "auto")))


def test_derived_numbers_and_run_json_use_the_gpu_mg_model(tmp_path, monkeypatch, fake_devices):
    """`derived.time_estimate_s` (run.json, check_problem) for solver gpu-mg is the gpu-mg number, with the band; not the Cholesky one."""
    fake_devices([dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")])
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, cases=3, penal=4.0, iterations=40)
    d = check_data(str(p), solver_setting="gpu-mg")
    cfg = load_config(p)
    gpu = gpu_check.TimeInputs.from_config(cfg, d["opencl"])
    assert (gpu.n_cases, gpu.penal, gpu.gpu_tol, gpu.case) == (3, 4.0, 1e-6, "B") and gpu.device["name"] == "Fast GPU"
    der = d["derived"]
    expect = estimate_seconds(d["free_dofs"], 40, "gpu-mg", gpu=gpu)[1]
    assert der["time_estimate_s"] == round(expect, 1) == round(d["time_estimate"]["total_s"], 1)
    assert der["time_estimate_band_s"][0] <= der["time_estimate_s"] <= der["time_estimate_band_s"][1]      # rounded to 0.1 s
    chol = derived_numbers(cfg, (11, 5, 5), 2.0, 275, 200, d["free_dofs"], "cholesky")
    assert "time_estimate_band_s" not in chol and chol["time_estimate_s"] != der["time_estimate_s"]
    # the run writer gets the same number: `write_run_json` calls derived_numbers with the solver of the run
    again = derived_numbers(cfg, tuple(d["grid"]["shape"]), d["grid"]["voxel_mm"], d["active_voxels"], d["free_design_voxels"], d["free_dofs"], "gpu-mg", None)
    assert again["time_estimate_s"] == der["time_estimate_s"]


TIME_MODEL_KEYS = {"model", "basis", "label", "device", "case", "penal", "volume_fraction", "lgK", "gpu_tol", "n_cases", "constants_source"}


def test_derived_has_a_time_model_that_names_the_basis_and_the_inputs(tmp_path, monkeypatch, fake_devices, state_folder):
    """P2a review M3: `derived.time_estimate_s` alone was a bare number; `derived.time_model` says whose constants and which inputs."""
    p = _gpu_case(tmp_path, cases=3, penal=4.0, iterations=40, volume_fraction=0.3, gpu_tol=1e-7)
    cfg = load_config(p)
    _facts(monkeypatch)
    got = {}
    for label, devs in (("reference", [dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")]),
                        ("table", [dev(0, "Intel(R) Arc(TM) Graphics", "GPU", True, platform_name="Intel(R) OpenCL Graphics", vendor="Intel(R) Corporation")]),
                        ("none", [dev(0, "Fast CPU", "CPU", True)])):
        fake_devices(devs)
        d = check_data(str(p), solver_setting="gpu-mg")
        tm = d["derived"]["time_model"]
        assert set(tm) == TIME_MODEL_KEYS and tm["basis"] == label and tm["model"] == "gpu-mg"
        assert (tm["penal"], tm["volume_fraction"], tm["gpu_tol"], tm["n_cases"], tm["device"]) == (4.0, 0.3, 1e-7, 3, devs[0].name)
        assert tm["lgK"] == pytest.approx(-math.log10(1e-6 + 0.3 ** 4), abs=0.005)
        assert tm["constants_source"] and tm["label"] == d["time_estimate"]["gpu_mg"]["label"]
        got[label] = tm
    assert got["reference"]["label"] == "reference: Intel Arc iGPU constants" and got["table"]["label"] == "Intel Arc iGPU constants"
    assert got["none"]["label"] == "no constants for this device"
    # a calibration: the basis and the source are those of the record
    fake_devices([dev(0, "Apple M4", "GPU", False, platform_name="Apple", vendor="Apple")])
    calibrated(c_s=5e-9, host=0.5e-6, device=APPLE, case="C")
    tm = check_data(str(p), solver_setting="gpu-mg")["derived"]["time_model"]
    assert tm["basis"] == "calibration" and tm["label"] == "constants of this machine" and "gpu-check" in tm["constants_source"] and tm["case"] == "C"
    # the other solvers have no time model, and JSON takes it
    assert "time_model" not in check_data(str(p), solver_setting="amg")["derived"]
    json.dumps(tm)


def test_the_run_json_has_the_time_model_of_the_run(tmp_path, monkeypatch, fake_devices):
    from topopt.problem import build_problem
    from topopt.snapshot import RunWriter
    fake_devices([dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")])
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, cases=2, penal=3.0, iterations=40)
    prob = build_problem(load_config(p))
    out = tmp_path / "run"
    out.mkdir()
    RunWriter(out, 0.0, quiet=True).write_run_json(prob, 10, 1234, "gpu-mg")
    der = json.loads((out / "run.json").read_text())["derived"]
    assert set(der["time_model"]) == TIME_MODEL_KEYS and der["time_model"]["basis"] == "reference" and der["time_model"]["n_cases"] == 2
    assert der["time_model"]["penal"] == 3.0 and der["time_estimate_s"] > 0


def test_the_first_time_line_of_check_names_the_basis_for_gpu_mg(tmp_path, monkeypatch, fake_devices, capsys, state_folder):
    """The first line used to be a bare number; the label of the constants came three lines later."""
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, h=1.0, iterations=40, cases=2)
    first = lambda: [l for l in capsys.readouterr().out.splitlines() if l.startswith("time estimate:")]      # noqa: E731
    for devs, tail in (([dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")], " (reference: Intel Arc iGPU constants)"),
                       ([dev(0, "Intel(R) Arc(TM) Graphics", "GPU", True, platform_name="Intel(R) OpenCL Graphics", vendor="Intel(R) Corporation")],
                        " (Intel Arc iGPU constants)")):
        fake_devices(devs)
        assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
        (line,) = first()
        assert " in total" + tail + " (" in line and "s/iteration, about " in line      # E1-P3: the bounds text follows the basis (up to ... or no up-to value)
    fake_devices([dev(0, "Apple M4", "GPU", False, platform_name="Apple", vendor="Apple")])
    calibrated(c_s=5e-9, host=0.5e-6, device=APPLE, case="C")
    assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
    (line,) = first()
    assert " in total (constants of this machine) (" in line
    # the other solvers keep their line
    assert main(["check", str(p), "--solver", "amg"]) == 0
    (line,) = first()
    assert re.search(r" in total \((up to about|no up-to value)", line) and "constants" not in line      # E1-P3: the bounds text; a value on Linux x86_64


def test_check_data_for_gpu_mg_keeps_the_time_warnings_with_the_gpu_mg_number(tmp_path, monkeypatch, fake_devices):
    fake_devices([dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")])
    _facts(monkeypatch)
    fits = []
    monkeypatch.setattr(cli, "fit_voxel_size", lambda *a, **k: fits.append((a, k)) or 1.7)
    p = _gpu_case(tmp_path, h=1.0, iterations=80, stop_rules={"time_budget_s": 9.0})       # the budget of the user: the only one
    d = check_data(str(p), solver_setting="gpu-mg")
    te = d["time_estimate"]
    assert te["model"] == "gpu-mg" and te["total_s"] > 9.0 and te["gpu_mg"]["basis"] == "reference"
    warn = [w for w in d["warnings"] if w.startswith("the gpu-mg time estimate")]
    assert len(warn) == 1 and f"{te['total_s']:.0f} s (band {te['low_s']:.0f} to {te['high_s']:.0f} s)" in warn[0]
    assert "above your time budget of 9 s (settings.stop_rules.time_budget_s)" in warn[0] and "demo" not in warn[0]
    assert "; the stop rules are on in this YAML (a stop_rules mapping turns them on), so the run stops at the first pass that ends after 9 s of run time" in warn[0]   # MJ1
    assert warn[0].index("rule time_budget") < warn[0].index("voxel_size about")
    assert "voxel_size about 1.7 mm would fit" in warn[0] and fits[-1] == ((1.0, te["total_s"], 9.0, "gpu-mg"), {})       # the fit gets the budget and the model of the solver
    assert "compact part" not in warn[0] and not [w for w in d["warnings"] if w.startswith("the time estimate is")]
    assert d["time_estimate"]["limit_s"] == 9.0                    # `limit_s` is the budget of the user
    (tmp_path / "nb").mkdir()
    d0 = check_data(str(_gpu_case(tmp_path / "nb", h=1.0, iterations=80)), solver_setting="gpu-mg")      # no budget in the YAML
    assert d0["time_estimate"]["limit_s"] is None and not [w for w in d0["warnings"] if "your time budget" in w]
    assert not [w for w in d["warnings"] if "no time model" in w or "Cholesky" in w and "time" in w]
    one = [w for w in d["warnings"] if w.startswith("gpu-mg models: ")]
    assert len(one) == 1 and "experimental" not in one[0] and "no other gpu-mg run on the device" in one[0]
    assert "stiffness contrast" in one[0] and "penal 3" in one[0] and "topopt gpu-check" in one[0]


def test_the_check_of_another_solver_is_not_changed_by_the_gpu_mg_model(tmp_path, monkeypatch, fake_devices):
    fake_devices([dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")])
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, h=1.0, iterations=80)
    d = check_data(str(p), solver_setting="amg")
    assert "model" not in d["time_estimate"] and "gpu_mg" not in d["time_estimate"] and "gpu-mg" not in d["backends"]
    assert d["time_estimate"]["total_s"] == pytest.approx(estimate_seconds(d["free_dofs"], 80, "amg")[1])


def test_the_mac_without_a_row_gets_a_clear_check(tmp_path, monkeypatch, fake_devices, capsys):
    """An Apple GPU has no row: the check says so, shows the reference number as such and does not show the Cholesky number as the time."""
    fake_devices([dev(0, "Apple M4", "GPU", False, platform_name="Apple", vendor="Apple")])
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, h=1.0, iterations=40, cases=2)
    assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
    out = capsys.readouterr().out
    assert "time constants: no constants for this device (Apple M4, case C); run `topopt gpu-check` to measure them." in out
    assert "as a reference" in out and "the iteration bands are from case B runs" in out
    assert main(["check", str(p), "--solver", "gpu-mg", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    te, chol = d["time_estimate"], d["backends"]["cholesky"]["time_estimate"]["total_s"]
    assert te["basis"] == "reference" and te["total_s"] != round(chol) and te["gpu_mg"]["device"] == "Apple M4"
    assert te["gpu_mg"]["case"] == "C" and "freeze_fact" not in d["opencl"]


def test_the_check_uses_a_calibration_and_names_its_source(tmp_path, monkeypatch, fake_devices, capsys):
    fake_devices([dev(0, "Apple M4", "GPU", False, platform_name="Apple", vendor="Apple")])
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, h=1.0, iterations=40, cases=2)
    assert main(["check", str(p), "--solver", "gpu-mg", "--json"]) == 0
    before = json.loads(capsys.readouterr().out)["time_estimate"]
    calibrated(c_s=5e-9, host=0.5e-6, device=APPLE, case="C")          # a device that is 4 times faster than the reference
    assert main(["check", str(p), "--solver", "gpu-mg", "--json"]) == 0
    after = json.loads(capsys.readouterr().out)["time_estimate"]
    assert after["basis"] == "calibration" and after["total_s"] < before["total_s"] and "measured on this machine" in after["gpu_mg"]["constants"]["source"]
    assert "constants of this machine, measured by `topopt gpu-check` on 2026-10-23" in after["note"]
    assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
    assert "time constants: constants of this machine, measured by `topopt gpu-check`" in capsys.readouterr().out


def test_check_yaml_of_the_builder_uses_the_gpu_mg_model(tmp_path, monkeypatch, fake_devices, capsys):
    fake_devices([dev(0, "Fast GPU", "GPU", True, platform_name="Some Platform", vendor="Some Vendor")])
    _facts(monkeypatch)
    p = _gpu_case(tmp_path, h=1.0, iterations=80, cases=2, solver="gpu-mg", stop_rules={"time_budget_s": 9})
    lines = []
    ok, kind = rb.check_yaml(p, lambda *a: lines.append(" ".join(str(x) for x in a)))
    text = "\n".join(lines)
    assert ok
    assert "time estimate: gpu-mg model about " in text and "band " in text and "does not change it" in text and "conservative" not in text.split("memory")[0]
    assert "time basis: no constants for this device (Fast GPU, case B)" in text
    assert ("WARNING: the gpu-mg estimate is above your time budget of 9 s (settings.stop_rules.time_budget_s); the stop rules are on in this YAML "
            "(a stop_rules mapping turns them on), so the run stops at the first pass that ends after 9 s of run time (rule time_budget; the rule "
            "converged is on too) and the part is the design of that pass; --no-stop-rules (MCP run_topopt(stop_rules=false)) turns the rules off "
            "for one run. A voxel_size of about ") in text      # MJ1: the same words as `check`
    assert "demo" not in text
    lines.clear()
    (tmp_path / "other").mkdir()
    p2 = _gpu_case(tmp_path / "other", h=1.0, iterations=80)           # the default solver: the old lines
    rb.check_yaml(p2, lambda *a: lines.append(" ".join(str(x) for x in a)))
    other = "\n".join(lines)
    assert "time estimate: about " in other and "(the model); scaled by the fill " in other and "gpu-mg" not in other
    assert "conservative" not in other and "demo" not in other and "time budget" not in other      # no budget of the user: no budget warning
