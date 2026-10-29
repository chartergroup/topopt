"""F4-36: the memory model of `topopt check` for gpu-mg (CONTRACTS.md section 20, "Memory model of `check`").

The tables are the measured peaks (`ru_maxrss` of the whole `topopt run`, MB of 10^6 bytes) of runs on pocl, one Linux x86_64 host
(Intel Core Ultra 9 185H, shared), 8 compute units, revision 74a1e46 of main, 2026-10-25. `N` is the free dofs and `V` the voxels of the grid. The rule of the fit: the estimate is not below any measured peak
(`typical_bytes` for the `plain` class, `bytes` for all rows) and the 15 runs of the old fit stay below both.
No OpenCL is needed: the model is a function of numbers (the tests of `check` fake the device list).
"""
from pathlib import Path

import pytest

from test_gpu_mg_x_merge import fake_cpu_device  # noqa: F401  (a fixture: a faked device list, so `check` can resolve gpu-mg with no OpenCL)
from topopt.gpu_mg import check as gpu_check

REPO = Path(__file__).resolve().parent.parent

# (run, N, V, precision case, joints, peak MB, class). `plain`: the region shapes of the hook example or simpler. `stress`: the hook with an
# extra keep-out shape whose bounding box covers the whole part (`big_`, `far3`) or a small keep-out shape with 80 or 320 triangles (`dense_`).
MEASURED = [
    ("sk_L100_b4_h2.0", 15363, 143312, "B", 0, 588.9, "plain"),
    ("hook2.4_i10", 16434, 14280, "B", 0, 610.5, "plain"),
    ("hook2.4_i30", 16434, 14280, "B", 0, 607.3, "plain"),
    ("hook2.4_i80", 16434, 14280, "B", 0, 620.6, "plain"),
    ("hook_stress_h2.4", 16434, 14280, "B", 0, 664.6, "plain"),
    ("sk_L100_b4_h1.5", 36276, 347900, "B", 0, 838.7, "plain"),
    ("sk_h1.5_dense2", 36276, 347900, "B", 0, 797.7, "plain"),
    ("hook1.8_i30", 36516, 33300, "B", 0, 782.1, "plain"),
    ("padJ0_h1.5_B", 51912, 20240, "B", 0, 569.3, "plain"),
    ("padJ0_h1.5_B_dense2", 51912, 21252, "B", 0, 513.2, "plain"),
    ("padJ1_h1.5_B", 55881, 20240, "B", 1, 574.3, "plain"),
    ("padJ2_h1.5_B", 58590, 21252, "B", 2, 604.1, "plain"),
    ("padJ2_h1.5_C", 58590, 21252, "C", 2, 660.2, "plain"),
    ("hook1.5_i30", 63291, 57024, "B", 0, 975.9, "plain"),
    ("hook1.5_i30_rep", 63291, 57024, "B", 0, 970.0, "plain"),
    ("hook1.5_i80", 63291, 57024, "B", 0, 1012.7, "plain"),
    ("hook_blur_h1.5", 63291, 57024, "B", 0, 684.9, "plain"),
    ("hook_c6_h1.5", 63291, 57024, "B", 0, 978.8, "plain"),
    ("hook_fr6_h1.5", 63291, 57024, "B", 0, 928.3, "plain"),
    ("sk_L100_b4_h1.0", 85602, 1146496, "B", 0, 1160.9, "plain"),
    ("hook1.2_i30", 115836, 112200, "B", 0, 1237.4, "plain"),
    ("sk_L100_b4_h0.8", 153984, 2247700, "B", 0, 1804.2, "plain"),
    ("padJ0_h1.0_B", 171021, 67320, "B", 0, 743.8, "plain"),
    ("padJ2_h1.0_B", 189621, 70686, "B", 2, 701.2, "plain"),
    ("padJ2_h1.0_C", 189621, 70686, "C", 2, 827.5, "plain"),
    ("hook1.0_i30", 205485, 187110, "B", 0, 1654.8, "plain"),
    ("sk_L100_b4_h0.7", 239583, 3374552, "B", 0, 2532.0, "plain"),
    ("sk_L100_b4_h0.6", 365352, 5358852, "B", 0, 1482.3, "plain"),
    ("hook0.8_i15", 386142, 372504, "B", 0, 2361.0, "plain"),
    ("hook0.7_i12", 580863, 551000, "B", 0, 2905.4, "plain"),
    ("hook0.6_i12", 905010, 876150, "B", 0, 3805.3, "plain"),
    ("big_h2.4_f0.9", 16434, 14280, "B", 0, 738.0, "stress"),
    ("dense_h2.4_s2", 16434, 14280, "B", 0, 919.3, "stress"),
    ("padJ2_h1.5_B_far3", 58590, 605613, "B", 2, 919.7, "stress"),
    ("padJ2_h1.5_C_far3", 58590, 605613, "C", 2, 932.5, "stress"),
    ("big_h1.5_f0.9", 63291, 57024, "B", 0, 1465.1, "stress"),
    ("big_h1.5_f0.9_rep", 63291, 57024, "B", 0, 1463.8, "stress"),
    ("big_h1.5_f1.5", 63291, 177870, "B", 0, 1703.7, "stress"),
    ("big_h1.5_f2", 63291, 421872, "B", 0, 1737.6, "stress"),
    ("big_h1.5_f2.5", 63291, 830720, "B", 0, 1796.6, "stress"),
    ("big_h1.5_f3.5", 63291, 2247784, "B", 0, 1989.1, "stress"),
    ("big_h1.5_f5", 63291, 6537960, "B", 0, 1578.8, "stress"),
    ("dense_h1.5_s1", 63291, 57024, "B", 0, 1599.3, "stress"),
    ("dense_h1.5_s2", 63291, 57024, "B", 0, 2086.4, "stress"),
    ("big_h1.0_f0.9", 205485, 187110, "B", 0, 2650.7, "stress"),
    ("dense_h1.0_s2", 205485, 187110, "B", 0, 4003.1, "stress"),
    ("big_h0.8_f0.9", 386142, 372504, "B", 0, 3949.7, "stress"),
]

# The 15 runs of the old fit (hook, case B, 2026-10-20, revision 5b32a83, before the SDF mesh step):
# (voxel mm, N, V, peak MB, script)
OLD_HOOK = [
    (2.4, 16434, 14280, 397.4, "bench"), (1.8, 36516, 33300, 492.5, "bench"), (1.5, 63291, 57024, 592.9, "bench"), (1.2, 115836, 112200, 487.7, "bench"),
    (1.0, 205485, 187110, 579.8, "bench"), (0.8, 386142, 372504, 833.0, "bench"), (0.7, 580863, 551000, 1104.9, "bench"), (0.6, 905010, 876150, 1196.8, "bench"),
    (2.4, 16434, 14280, 438.3, "phases"), (1.5, 63291, 57024, 663.3, "phases"), (1.2, 115836, 112200, 591.5, "phases"), (1.0, 205485, 187110, 678.8, "phases"),
    (0.8, 386142, 372504, 904.6, "phases"), (0.7, 580863, 551000, 1152.7, "phases"), (0.6, 905010, 876150, 1442.1, "phases"),
]


def test_the_constants_of_the_memory_model_are_the_fitted_ones():
    """The numbers of the fit (CONTRACTS.md section 20); a change of one is a new fit and needs a new table above."""
    assert (gpu_check.GPU_MG_BASE_BYTES, gpu_check.GPU_MG_BYTES_PER_DOF, gpu_check.GPU_MG_JOINT_BYTES_PER_DOF) == (860_000_000, 730, 1000)
    assert (gpu_check.GPU_MG_MESH_BASE_BYTES, gpu_check.GPU_MG_MESH_BYTES_PER_DOF23, gpu_check.GPU_MG_MESH_BYTES_PER_GRID_VOXEL) == (500_000_000, 370_000, 220)
    assert gpu_check.GPU_MG_LAYOUT_FACTOR == 3.2


def test_the_estimate_of_the_hook_at_1_0_mm_is_above_its_measured_peak():
    """Main at 74a1e46 gave 815 MB for 205,485 free dofs; the run peaked at 1,655 MB (the mesh step of the post-process, not the loop)."""
    assert gpu_check.estimate_memory_bytes(205_485) >= 1_654.8e6
    assert gpu_check.estimate_memory_bytes(905_010) >= 3_805.3e6      # the hook at 0.6 mm: 3.8 GB against 1.5 GB of the old estimate


@pytest.mark.parametrize("run,n,v,case,joints,peak_mb,kind", MEASURED, ids=[m[0] for m in MEASURED])
def test_the_estimate_is_not_below_any_measured_peak(run, n, v, case, joints, peak_mb, kind):
    m = gpu_check.estimate_memory(n, v, case, joints)
    assert m["bytes"] >= peak_mb * 1e6, (run, m["bytes"] / 1e6, peak_mb)
    assert m["typical_bytes"] <= m["bytes"]
    if kind == "plain":
        assert m["typical_bytes"] >= peak_mb * 1e6, (run, m["typical_bytes"] / 1e6, peak_mb)


@pytest.mark.parametrize("mm,n,v,peak_mb,script", OLD_HOOK, ids=[f"{o[0]}-{o[4]}" for o in OLD_HOOK])
def test_the_old_hook_rows_stay_below_the_estimate(mm, n, v, peak_mb, script):
    """The rule of the old fit holds: the 15 peaks of the first fit are below both numbers (without the grid: the same N alone is enough)."""
    m = gpu_check.estimate_memory(n, v, "B", 0)
    assert peak_mb * 1e6 <= m["typical_bytes"] <= m["bytes"]
    assert peak_mb * 1e6 <= gpu_check.estimate_memory_bytes(n)


def test_the_lug_rows_of_the_card():
    """The two-joint part (2 tube joints, no fixed region; pocl, revision 133e5a1): the card has N and V at 1.0 mm. At 2.2 mm V is not in the card;
    the estimate is above 2.6 GB for any V because the mesh term alone is (N = 75,489)."""
    assert gpu_check.estimate_memory_bytes(600_771, 6_450_000, "B", 2) >= 3.7e9
    assert gpu_check.estimate_memory_bytes(600_771, 6_450_000, "C", 2) >= 3.5e9
    assert gpu_check.estimate_memory_bytes(75_489, 0, "B", 2) >= 2.6e9
    assert gpu_check.estimate_memory_bytes(75_489, 0, "C", 2) >= 2.5e9
    assert gpu_check.estimate_memory(600_771, 6_450_000, "B", 2)["typical_bytes"] >= 3.7e9    # also without the layout factor, at 1.0 mm


def test_the_phase_is_the_loop_for_a_small_part_and_the_mesh_step_for_a_large_one():
    small, large = gpu_check.estimate_memory(5_000, 5_000, "B", 0), gpu_check.estimate_memory(600_000, 600_000, "B", 0)
    assert small["phase"] == "loop" and small["bytes"] == small["loop_bytes"] and small["typical_bytes"] == small["loop_bytes"]
    assert large["phase"] == "mesh" and large["bytes"] == large["mesh_bytes"] > large["loop_bytes"]


def test_the_columns_of_case_c_are_counted_in_the_loop_phase_only():
    b, c, unknown = (gpu_check.estimate_memory(190_000, 70_000, case, 2) for case in ("B", "C", None))
    assert c["loop_bytes"] - b["loop_bytes"] == 190_000 * gpu_check.GPU_MG_JOINT_BYTES_PER_DOF
    assert unknown["loop_bytes"] == c["loop_bytes"]                           # no device: the case is not known, the larger number
    assert c["mesh_bytes"] == b["mesh_bytes"]
    assert gpu_check.estimate_memory(190_000, 70_000, "C", 0)["loop_bytes"] == b["loop_bytes"]         # no joint: no column


def test_the_grid_voxels_are_in_the_mesh_phase_and_the_layout_factor_is_on_its_surface_term_only():
    base = gpu_check.estimate_memory(100_000, 0, "B", 0)
    grid = gpu_check.estimate_memory(100_000, 1_000_000, "B", 0)
    assert grid["mesh_bytes"] - base["mesh_bytes"] == 1_000_000 * gpu_check.GPU_MG_MESH_BYTES_PER_GRID_VOXEL
    assert grid["loop_bytes"] == base["loop_bytes"]
    surface = gpu_check.GPU_MG_MESH_BYTES_PER_DOF23 * 100_000 ** (2 / 3)
    assert base["mesh_bytes"] - base["typical_bytes"] == pytest.approx((gpu_check.GPU_MG_LAYOUT_FACTOR - 1) * surface, abs=2)


def _check_dict(free_dofs, voxels, case="B", joints=0, ram=64_000_000_000):
    """The part of the dict of `cli.check_data` that `gpu_check.apply` reads."""
    g = {"case": case, "low_s": 1.0, "high_s": 2.0, "basis": "reference", "total_s": 1.5, "assumptions": ["test"]}
    return {"solver": {"resolved": "gpu-mg"}, "free_dofs": free_dofs, "grid": {"voxels": voxels}, "extended": {"joints": [{}] * joints},
            "memory_estimate": {"bytes": 1, "ram_bytes": ram}, "time_estimate": {"gpu_mg": g}, "warnings": [], "opencl": None}


def test_check_gives_a_memory_number_above_the_measured_peak_of_a_run_with_joints_and_a_high_ratio_of_voxels_to_dofs(monkeypatch):
    """The path of `topopt check`: `apply` writes `memory_estimate`. Main at 74a1e46 gave 0.67 GB for the hook at 1.5 mm (measured 0.98 GB) and
    0.82 GB for 189,621 dofs with 2 joints in case C (measured 0.83 GB); with a grid of 605,613 voxels the pad with 2 joints measured 0.93 GB."""
    monkeypatch.setattr(gpu_check, "basis_text", lambda g: "test")
    for run, n, v, case, joints, peak_mb in (("hook1.5_i30", 63291, 57024, "B", 0, 975.9), ("padJ2_h1.0_C", 189621, 70686, "C", 2, 827.5),
                                             ("padJ2_h1.5_C_far3", 58590, 605613, "C", 2, 932.5), ("hook0.6_i12", 905010, 876150, "B", 0, 3805.3)):
        me = gpu_check.apply(_check_dict(n, v, case, joints))["memory_estimate"]
        assert me["bytes"] >= peak_mb * 1e6 and me["bytes"] == max(me["loop_bytes"], me["mesh_bytes"]), (run, me["bytes"] / 1e6, peak_mb)


def test_check_warns_when_the_memory_number_is_above_the_ram_and_names_the_typical_number(monkeypatch):
    monkeypatch.setattr(gpu_check, "basis_text", lambda g: "test")
    d = gpu_check.apply(_check_dict(600_771, 6_450_000, "B", 2, ram=8_000_000_000))
    warn = [w for w in d["warnings"] if w.startswith("the memory estimate")]
    assert len(warn) == 1 and "like the hook example" in warn[0]
    assert d["memory_estimate"]["bytes"] > 8_000_000_000 and 4.5e9 < d["memory_estimate"]["typical_bytes"] < 4.6e9        # 10.35 GB and 4.55 GB


# --- the filter in the model (P9-4d scope item, issue #329; the FFT filter of #333): the build phase and the filter held in the loop -------------


def test_the_filter_is_in_the_build_phase_and_in_the_loop_phase():
    """A part with a large filter: 529,287 free dofs, the filter alone holds 6.53 GiB and needs 14.0 GiB at its build (the case of issue #329)."""
    held, built = int(6.53 * 2**30), int(14.0 * 2**30)
    without = gpu_check.estimate_memory(529_287, 3_000_000, "B", 0)
    m = gpu_check.estimate_memory(529_287, 3_000_000, "B", 0, held, built)
    assert m["build_bytes"] == gpu_check.GPU_MG_BASE_BYTES + built and m["phase"] == "build" and m["bytes"] == m["build_bytes"] > without["bytes"]
    assert m["bytes"] >= gpu_check.GPU_MG_BASE_BYTES + built and m["typical_bytes"] >= gpu_check.GPU_MG_BASE_BYTES + built
    assert m["loop_bytes"] == without["loop_bytes"] + held and m["mesh_bytes"] == without["mesh_bytes"]      # the mesh phase has no filter term
    assert gpu_check.estimate_memory_bytes(529_287, 3_000_000, "B", 0, held, built) == m["bytes"]


def test_a_small_filter_keeps_the_number_of_the_phases_of_the_hook():
    """The hook at 1.2 mm: the mesh step sets the peak, so a filter of 100 MiB held and 220 MiB at the build does not change `bytes`."""
    n, v = 115_836, 112_200
    old = gpu_check.estimate_memory(n, v, "B", 0)
    new = gpu_check.estimate_memory(n, v, "B", 0, 100 * 2**20, 220 * 2**20)
    assert old["phase"] == "mesh" and new["phase"] == "mesh"
    assert new["bytes"] == old["bytes"] and new["typical_bytes"] == old["typical_bytes"] and new["mesh_bytes"] == old["mesh_bytes"]
    assert new["loop_bytes"] - old["loop_bytes"] == 100 * 2**20


def test_without_the_filter_numbers_the_phases_are_the_old_ones():
    for n, v in ((5_000, 5_000), (115_836, 112_200), (905_010, 876_150)):
        m = gpu_check.estimate_memory(n, v, "B", 0)
        assert m["build_bytes"] == gpu_check.GPU_MG_BASE_BYTES <= m["loop_bytes"]
        assert m["phase"] == ("mesh" if m["mesh_bytes"] > m["loop_bytes"] else "loop")
        assert m["bytes"] == max(m["loop_bytes"], m["mesh_bytes"])


def _case_with_a_large_filter(tmp_path, radius_mm, h=2.0):
    import yaml
    from test_optimize import make_case
    p = make_case(tmp_path, h=h, iterations=2)
    d = yaml.safe_load(p.read_text())
    d["settings"]["filter_radius"] = radius_mm
    p.write_text(yaml.safe_dump(d))
    return p


def test_check_counts_the_filter_and_warns_with_the_filter_line_and_its_option(tmp_path, monkeypatch, fake_cpu_device):
    """A small grid with a filter radius of 20 voxels. The matrix of this radius had 8.4 million entries and set the peak at its build (1.3 GB);
    the FFT filter (#333) needs 56 B a cell of its padded grid and 24 B a design voxel, so the build phase stays below the loop. On a machine whose RAM holds the model
    without the filter and not with it, the warning has the memory line and the filter line with its option."""
    from topopt import cli
    monkeypatch.setattr(gpu_check, "basis_text", lambda g: "test")
    p = _case_with_a_large_filter(tmp_path, 40.0)
    d = cli.check_data(p, solver_setting="gpu-mg")
    me, fsz = d["memory_estimate"], d["derived"]["filter"]["size"]
    assert fsz["radius_voxels"] == 20.0
    assert fsz["held_bytes_estimate"] == int(fsz["fft_cells"] * 56 + fsz["design_voxels"] * 24) < 100_000_000
    assert me["model"] == "gpu-mg" and me["phase"] == "loop" and me["build_bytes"] == gpu_check.GPU_MG_BASE_BYTES + fsz["build_peak_bytes_estimate"]
    old = gpu_check.estimate_memory(d["free_dofs"], d["grid"]["voxels"], d["time_estimate"]["gpu_mg"]["case"], 0)
    assert me["loop_bytes"] == old["loop_bytes"] + fsz["held_bytes_estimate"] and me["bytes"] == me["loop_bytes"]
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: old["bytes"])
    d = cli.check_data(p, solver_setting="gpu-mg")
    warn = [w for w in d["warnings"] if w.startswith(("the memory estimate", "the filter needs"))]
    assert len(warn) == 2 and "the gpu-mg model" in warn[0]
    assert warn[1].startswith("the filter needs about") and "Option: a larger voxel_size" in warn[1]
    assert "the Cholesky" not in " ".join(warn)


def test_check_keeps_the_number_of_a_small_filter_problem(tmp_path, monkeypatch, fake_cpu_device):
    """A part with a small filter: the loop sum has its filter term (a few hundred kB here, counted twice at most), and nothing else changes."""
    from topopt import cli
    monkeypatch.setattr(gpu_check, "basis_text", lambda g: "test")
    d = cli.check_data(_case_with_a_large_filter(tmp_path, 3.0), solver_setting="gpu-mg")
    me, fsz = d["memory_estimate"], d["derived"]["filter"]["size"]
    old = gpu_check.estimate_memory(d["free_dofs"], d["grid"]["voxels"], d["time_estimate"]["gpu_mg"]["case"], 0)
    assert me["phase"] == old["phase"] == "loop" and me["mesh_bytes"] == old["mesh_bytes"]
    assert me["bytes"] - old["bytes"] == fsz["held_bytes_estimate"] < 1_000_000


def test_check_of_the_hook_keeps_its_number(monkeypatch, fake_cpu_device):
    """No change for the hook example: its mesh step sets the peak, so the filter terms (a few MB: the FFT filter) do not move `bytes`."""
    from topopt import cli
    monkeypatch.setattr(gpu_check, "basis_text", lambda g: "test")
    d = cli.check_data(REPO / "examples" / "hook" / "hook.yaml", solver_setting="gpu-mg")
    me = d["memory_estimate"]
    old = gpu_check.estimate_memory(d["free_dofs"], d["grid"]["voxels"], d["time_estimate"]["gpu_mg"]["case"], 0)
    assert me["phase"] == old["phase"] == "mesh" and me["bytes"] == old["bytes"] and me["typical_bytes"] == old["typical_bytes"]
    assert me["build_bytes"] < me["bytes"] and me["loop_bytes"] - old["loop_bytes"] == d["derived"]["filter"]["size"]["held_bytes_estimate"]


def test_the_run_start_line_of_gpu_mg_counts_the_filter(tmp_path, monkeypatch):
    """`cli.run_memory_lines` gives the same number as `check` for solver gpu-mg, with the filter line when the filter is the cause."""
    from topopt import cli
    from topopt.config import load_config
    from topopt.members import filter_report
    from topopt.optimize import reduced_system
    from topopt.problem import build_problem
    prob = build_problem(load_config(_case_with_a_large_filter(tmp_path, 40.0)))
    rep = filter_report(prob.config.settings, prob.h, int((prob.labels == 1).sum()), prob.shape)
    free = reduced_system(prob)[2]
    want = gpu_check.problem_memory(int(free.size), int(prob.shape[0] * prob.shape[1] * prob.shape[2]), "B", 0, rep["size"])
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: want["without_filter"])     # holds the run without the filter, not with it
    lines = cli.run_memory_lines(prob, rep, solver="gpu-mg", case="B")
    assert want["phase"] == "loop" and f"memory estimate: about {cli.gib(want['bytes'])} of {cli.gib(want['without_filter'])} RAM" in lines[0]
    assert "the gpu-mg model, set by its loop phase" in lines[0] and "the filter is part of it" in lines[0]
    assert lines[1].startswith("warning: the memory estimate is") and lines[2].startswith("warning: the filter needs about")
