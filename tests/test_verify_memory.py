"""The memory model of `topopt verify` and the measured peak (CONTRACTS.md section 16, "Memory model")."""
import subprocess
import sys

import pytest

from topopt.verify import memory

# Measured with `mesh_stl` (1 thread): shape, size, volume, area, dofs. The hook rows are the final.stl of the recorded run.
MEASURED = [
    ("hook", 3.0, 21362.8, 12147.6, 32463), ("hook", 2.0, 21362.8, 12147.6, 70992),
    ("hook", 1.5, 21362.8, 12147.6, 130845), ("hook", 1.2, 21362.8, 12147.6, 208065),
    ("hook", 1.0, 21362.8, 12147.6, 343860), ("hook", 0.9, 21362.8, 12147.6, 400650),
    ("hook", 0.8, 21362.8, 12147.6, 624525),
    ("box", 2.0, 8000.0, 2800.0, 18486), ("box", 1.0, 8000.0, 2800.0, 101397),
    ("sphere", 2.0, 14106.6, 2824.1, 23106), ("sphere", 1.0, 14106.6, 2824.1, 145194),
    ("plate", 1.0, 5400.0, 4140.0, 104340),
]


def test_model_is_7_5_gb_per_million_dofs_times_1_25():
    assert memory.rule_bytes(1e6) == pytest.approx(7.5e9)
    assert memory.estimate_bytes(1e6) == pytest.approx(7.5e9 * 1.25)
    assert memory.estimate_bytes(136512) == pytest.approx(1.28e9, rel=0.01)       # the hook at the default size
    assert memory.estimate_bytes(0) == 0


# SuperLU (E3): shape, volume, area, dofs, peak GB of the verify process (1 thread, one Linux x86_64 host). The hook rows are
# `topopt verify` runs of the recorded hook at 2.5 to 0.8 mm; the others are `mesh_stl` meshes of a cube, a sphere, a plate and a bar.
SUPERLU_PEAK = [
    ("hook", 21362.8, 12147.6, 45129, 0.50), ("hook", 21362.8, 12147.6, 70632, 0.64), ("hook", 21362.8, 12147.6, 134466, 1.08),
    ("hook", 21362.8, 12147.6, 213387, 1.66), ("hook", 21362.8, 12147.6, 359193, 3.05), ("hook", 21362.8, 12147.6, 419934, 3.94),
    ("hook", 21362.8, 12147.6, 662454, 6.82),
    ("cube", 27000.0, 5400.0, 23742, 0.49), ("cube", 27000.0, 5400.0, 57384, 1.35),
    ("sphere", 14106.6, 2824.1, 27270, 0.86), ("sphere", 14106.6, 2824.1, 53331, 1.75),
    ("plate", 12000.0, 5800.0, 63432, 0.92), ("plate", 12000.0, 5800.0, 169464, 3.57), ("plate", 12000.0, 5800.0, 181668, 3.60),
    ("bar", 8000.0, 3400.0, 59142, 0.89), ("bar", 8000.0, 3400.0, 105081, 1.80), ("bar", 8000.0, 3400.0, 230829, 5.54),
]


def test_pre_mesh_dof_estimate_is_within_20_percent_of_the_measured_meshes():
    for name, h, V, A, dofs in MEASURED:
        est = memory.estimate_dofs(V, A, h)
        assert 0.8 < est / dofs < 1.2, (name, h, est, dofs)


def test_dof_estimate_falls_with_the_cube_of_the_size_for_a_compact_part():
    a = memory.estimate_dofs(8000.0, 0.0, 2.0)
    b = memory.estimate_dofs(8000.0, 0.0, 1.0)
    assert b / a == pytest.approx(8.0, rel=0.01)


def test_peak_rss_is_in_bytes_on_this_platform():
    """A child that holds 300 MB reports between 0.3 and 1.5 GB: not 1000 times too small (the Linux unit trap)."""
    code = ("import numpy as np; a = np.ones(300_000_000 // 8); a.sum(); "
            "from topopt.verify.memory import peak_rss_bytes; print(peak_rss_bytes())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    peak = int(out.strip().splitlines()[-1])
    # On Linux a child keeps the peak RSS of its parent (ru_maxrss survives fork and exec, also after the parent freed the memory).
    # So the upper bound grows with the peak of the pytest process, which depends on the tests that ran before this one.
    import resource
    parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    assert 0.3e9 < peak < 1.5e9 + parent_peak


def test_peak_rss_unit_switch_for_both_platforms(monkeypatch):
    import resource
    import types
    monkeypatch.setattr(resource, "getrusage", lambda who: types.SimpleNamespace(ru_maxrss=5_000_000))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert memory.peak_rss_bytes() == 5_000_000                    # macOS: bytes
    assert memory.children_peak_rss_bytes() == 5_000_000
    monkeypatch.setattr(sys, "platform", "linux")
    assert memory.peak_rss_bytes() == 5_000_000 * 1024             # Linux: kilobytes


def test_limit_is_the_smaller_of_the_ram_and_max_gb(monkeypatch):
    monkeypatch.setattr(memory, "available_ram_bytes", lambda: 16_000_000_000)
    assert memory.limits(None)["limit_bytes"] == 16_000_000_000
    assert memory.limits(None)["limit_source"] == "available RAM"
    assert memory.limits(8.0)["limit_bytes"] == 8_000_000_000
    assert memory.limits(8.0)["limit_source"] == "--max-gb"
    assert memory.limits(64.0)["limit_bytes"] == 16_000_000_000
    monkeypatch.setattr(memory, "available_ram_bytes", lambda: None)
    assert memory.limits(None)["limit_bytes"] is None              # RAM not known: no refusal from the RAM
    assert memory.limits(2.0)["limit_bytes"] == 2_000_000_000


def test_limit_source_is_the_ram_when_the_ram_is_below_max_gb_in_both_styles(monkeypatch):
    from topopt.verify import names
    monkeypatch.setattr(memory, "available_ram_bytes", lambda: 4_000_000_000)    # below max_gb: the RAM is the limit
    for style in ("cli", "mcp"):
        with names.use_style(style):
            lim = memory.limits(8.0)
            assert lim["limit_source"] == "available RAM" and lim["limit_bytes"] == 4_000_000_000
            assert lim["available_bytes"] == 4_000_000_000 and lim["max_gb_bytes"] == 8_000_000_000


# --------------------------------------------------------------------------- #
# The time model (review M4): the estimate of the seconds, against the 8 runs that it was fitted on
# --------------------------------------------------------------------------- #

# The hook at 1.5 to 0.8 mm (Gmsh+Netgen, 2 load cases, one busy Linux host): backend, dofs, with load bands, total seconds.
TIMED = [("pardiso", 134466, False, 12.44), ("pardiso", 213387, False, 11.65), ("pardiso", 359193, False, 16.52),
         ("pardiso", 419934, False, 18.54), ("pardiso", 662454, False, 30.79), ("cholespy", 134466, False, 10.06),
         ("cholespy", 359193, False, 46.67), ("cholespy", 419934, False, 69.59),
         ("pardiso", 134466, True, 14.1), ("pardiso", 359193, True, 38.6), ("pardiso", 662454, True, 68.1),
         ("cholespy", 359193, True, 84.9)]


def test_time_estimate_is_within_a_factor_of_the_measured_runs():
    for backend, dofs, sens, seconds in TIMED:
        est = memory.estimate_seconds(dofs, backend, 2, sensitivity=sens)
        assert 0.6 < est / seconds < 1.5, (backend, dofs, sens, est, seconds)
    # the mean absolute relative error is the number that the plan quotes
    err = [abs(memory.estimate_seconds(d, b, 2, sensitivity=n) / s - 1) for b, d, n, s in TIMED]
    assert sum(err) / len(err) < 0.25


def test_time_estimate_grows_with_dofs_cases_and_the_load_bands():
    f = memory.estimate_seconds
    assert f(100_000, "pardiso", 2) < f(400_000, "pardiso", 2) < f(1_000_000, "pardiso", 2)
    assert f(300_000, "cholespy", 2) > f(300_000, "pardiso", 2)                    # cholespy is slower on a large system
    assert f(300_000, "pardiso", 4) > f(300_000, "pardiso", 2)
    assert f(300_000, "pardiso", 2, sensitivity=True) > f(300_000, "pardiso", 2, sensitivity=False)


def test_there_is_no_time_estimate_without_a_measurement():
    assert memory.estimate_seconds(300_000, "cholesky", 2) is None                  # Accelerate on P2: not measured
    assert memory.estimate_seconds(300_000, "unknown", 2) is None


def test_limit_source_names_the_option_in_the_style_of_the_reader(monkeypatch):
    from topopt.verify import names
    monkeypatch.setattr(memory, "available_ram_bytes", lambda: 16_000_000_000)   # above max_gb: not the RAM of this host
    assert memory.limits(8.0)["limit_source"] == "--max-gb"
    with names.use_style("mcp"):
        assert memory.limits(8.0)["limit_source"] == "max_gb"
        assert names.opt("force") == "force" and names.opt("region_band") == "region_band_mm"
    assert names.opt("force") == "--force"


# --------------------------------------------------------------------------- #
# SuperLU: the memory and time model (E3)
# --------------------------------------------------------------------------- #

def test_compactness_is_dimensionless_and_ranks_the_shapes():
    c = memory.compactness
    assert c(21362.8, 12147.6) == pytest.approx(memory.HOOK_COMPACTNESS, abs=5e-4)           # the hook
    sphere = c(4 / 3 * 3.141592653589793 * 10.0 ** 3, 4 * 3.141592653589793 * 10.0 ** 2)
    assert sphere == pytest.approx(0.0940, abs=5e-4)                                          # the most compact shape
    assert c(27000.0, 5400.0) == pytest.approx(c(27000.0 * 8, 5400.0 * 4))                    # a scale does not change C
    assert c(21362.8, 12147.6) < c(12000.0, 5800.0) < c(8000.0, 3400.0) < c(27000.0, 5400.0) < sphere


def test_superlu_memory_is_the_base_plus_the_dof_term_plus_the_fill():
    c, n = 0.05, 2.0e6
    expect = 0.50 + 1.63 * 2.0 + 65000.0 * c ** 2 * 2.0 ** 2
    assert memory.rule_bytes(n, "superlu", c) == pytest.approx(expect * 1e9)
    assert memory.estimate_bytes(n, "superlu", c) == pytest.approx(expect * 1e9)              # the central value, no margin
    assert memory.superlu_fill_gb(n, c) == pytest.approx(65000.0 * c ** 2 * 4.0)
    assert memory.superlu_fill_gb(n) == memory.superlu_fill_gb(n, memory.HOOK_COMPACTNESS)    # no shape: the hook


def test_superlu_fill_grows_with_the_square_of_the_dofs_and_with_the_compactness():
    f = memory.superlu_fill_gb
    assert f(400_000, 0.016) == pytest.approx(4 * f(200_000, 0.016))
    assert f(200_000, 0.068) / f(200_000, 0.016) == pytest.approx((0.068 / 0.016) ** 2)       # a cube against a hook: 18 times
    assert memory.estimate_bytes(300_000, "superlu", 0.068) > 5 * memory.estimate_bytes(300_000, "superlu", 0.016)


def test_the_other_factors_keep_the_rule_of_7_5_gb_per_million_dofs():
    for backend in (None, "cholesky", "pardiso", "cholespy"):
        assert memory.rule_bytes(1e6, backend, 0.05) == pytest.approx(7.5e9)
        assert memory.estimate_bytes(1e6, backend, 0.05) == pytest.approx(7.5e9 * 1.25)
    assert memory.model_text("pardiso") == "7.5 GB per million dofs"
    assert "SuperLU model" in memory.model_text("superlu", 0.0123) and "C = 0.012" in memory.model_text("superlu", 0.0123)
    for backend in (None, "cholesky", "pardiso", "cholespy"):
        assert memory.memory_range_bytes(1e6, backend, 0.05) is None and memory.notes(1e6, backend, 0.05, 1e9) == []


def test_superlu_memory_range_holds_the_peak_of_every_measured_mesh():
    """The range is the measured spread of model / peak (0.60 to 1.45 on 17 meshes of five shapes): every peak is inside."""
    ratios = {}
    for name, V, A, dofs, peak in SUPERLU_PEAK:
        c = memory.compactness(V, A)
        lo, mid, hi = memory.memory_range_bytes(dofs, "superlu", c)
        assert mid == pytest.approx(memory.estimate_bytes(dofs, "superlu", c), abs=1)   # the headline is the central value
        assert lo < peak * 1e9 * 1.0005 and peak * 1e9 < hi * 1.0005, (name, dofs, lo / 1e9, peak, hi / 1e9)
        ratios.setdefault(name, []).append(mid / 1e9 / peak)
    assert len(SUPERLU_PEAK) == 17
    flat = [x for r in ratios.values() for x in r]
    assert (min(flat), max(flat)) == pytest.approx(memory.SUPERLU_MODEL_OVER_PEAK, abs=0.006)     # the constants are the data
    assert min(ratios["plate"]) < 0.7 and max(ratios["cube"]) > 1.4          # the shape that is under, and the one that is over
    assert all(0.9 < x < 1.35 for x in ratios["hook"])                        # the hook (the part of the demo): near the peak


def test_superlu_range_and_the_notes_for_a_run_that_passes_the_guard_with_the_upper_end_above_the_limit():
    hook = 0.016
    lo, mid, hi = memory.memory_range_bytes(662_000, "superlu", hook)           # the hook at 0.8 mm: central 8.9 GB
    assert lo == pytest.approx(mid / 1.45) and hi == pytest.approx(mid / 0.60) and lo < mid < hi
    assert memory.notes(662_000, "superlu", hook, 20e9) == []                   # the whole range is below the limit
    assert memory.notes(662_000, "superlu", hook, 3e9) == []                    # the guard refuses it (central above): no "run starts"
    (text,) = memory.notes(662_000, "superlu", hook, 12e9)
    assert "upper end" in text and "run starts" in text and "--mesh-size" in text and "--solver NAME" in text
    assert memory.gb_text(hi) in text and memory.gb_text(mid) in text
    assert memory.notes(662_000, "superlu", hook, None) == []                   # no limit known: nothing to compare with


def test_superlu_int32_fact_comes_with_a_large_modelled_fill_only():
    assert memory.int32_fact(300_000, 0.016) == ""
    big = memory.int32_fact(1_500_000, 0.016)                                   # modelled fill 37 GB
    assert "32-bit indices" in big and "not measured" in big and "37 GB" in big
    (text,) = memory.notes(1_500_000, "superlu", 0.016, 200e9)
    assert big in text and "--mesh-size" in text and "pypardiso" in text and "cholespy" in text
    assert "47 GB" in memory.int32_fact(1_000_000, 0.027)                       # a plate of 1M dofs: 47 GB of modelled fill


def test_superlu_time_estimate_for_the_hook_without_the_load_bands():
    # the hook at 2.5 to 0.8 mm, 2 load cases, no load bands: dofs, total seconds of `topopt verify` on SuperLU
    for dofs, seconds in [(70632, 7.6), (134466, 11.9), (213387, 20.1), (359193, 48.4), (419934, 64.8), (662454, 150.9)]:
        est = memory.estimate_seconds(dofs, "superlu", 2, sensitivity=False)
        assert 0.6 < est / seconds < 1.4, (dofs, est, seconds)


def test_superlu_time_grows_fast_with_the_compactness_and_has_a_model_for_every_shape():
    f = memory.estimate_seconds
    assert f(100_000, "superlu", 2, True, 0.068) > 4 * f(100_000, "superlu", 2, True, 0.016)
    assert f(300_000, "superlu", 2, True) > f(300_000, "pardiso", 2, True)       # slower than PARDISO at the hook
    assert f(300_000, "superlu", 2, True) == f(300_000, "superlu", 2, True, memory.HOOK_COMPACTNESS)
    assert f(300_000, "superlu", 2, True) > f(300_000, "superlu", 2, False)
