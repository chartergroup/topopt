"""#378: the design facts and options in the text of a gpu-mg solve that is not accepted (CONTRACTS.md section 20).

A case C solve of the phone dock of #378 stopped while the compliance of a load that its per-axis supports do not take rose 4.5 times in 4 passes:
the design came near a mechanism for that load. The text of a failed solve names the load case whose compliance rose the most (with its
`compliance_limit`) and the per-axis supports, and it adds the options that follow from them. The tests of part 1 need no OpenCL.
"""
import math
from types import SimpleNamespace

import pytest

from test_gpu_mg_axes import axes_case, fresh, full_clamps
from topopt.config import load_config
from topopt.gpu_mg.model import compliance_rise, partial_supports, rise_sentence

# history.jsonl of the #378 repro (roller_c, case C, passes 0 to 6, rounded): phone_rests (limit 0.05) and dock_insert (limit 10)
ROLLER_C = [[0.506, 1.02], [0.0891, 0.216], [0.0504, 0.214], [0.0484, 0.298], [0.0476, 0.443], [0.0478, 0.628], [0.048, 0.969]]
CASES = [("phone_rests", 0.05), ("dock_insert", 10.0)]


def rows_of(values):
    return [{"iteration": i, "compliance": sum(v), "compliance_cases": v} for i, v in enumerate(values)]


# --- part 1: the facts (no OpenCL) ----------------------------------------------------------------------------------


def test_the_rise_is_the_load_case_whose_compliance_rose_the_most_from_its_lowest_pass_to_the_last():
    rise = compliance_rise(rows_of(ROLLER_C), CASES)
    assert {k: rise[k] for k in ("case", "low", "low_pass", "last", "last_pass", "limit")} == {
        "case": "dock_insert", "low": 0.214, "low_pass": 2, "last": 0.969, "last_pass": 6, "limit": 10.0}
    assert math.isclose(rise["factor"], 0.969 / 0.214)
    assert rise_sentence(rise) == ("The compliance of load case 'dock_insert' rose from 0.214 at pass 2 to 0.969 at pass 6 (4.53 times; its "
                                   "compliance_limit is 10), the largest rise of the load cases in the passes so far. ")


def test_no_rise_without_two_passes_or_when_no_compliance_rose():
    assert compliance_rise([], CASES) is None
    assert compliance_rise(rows_of(ROLLER_C[:1]), CASES) is None
    assert compliance_rise(rows_of(ROLLER_C[:3]), CASES) is None                    # both fall up to pass 2
    assert rise_sentence(None) == ""


def test_a_run_with_one_load_case_and_rows_without_a_value_or_with_a_value_that_is_not_finite():
    rows = [{"iteration": 0, "compliance": 3.0}, {"iteration": 1, "compliance": 1.0}, {"iteration": 2, "compliance": float("nan")},
            {"iteration": 3, "compliance": 2.0}, {"iteration": 4}, {"note": "no iteration", "compliance": 9.0}]
    rise = compliance_rise(rows, [("only", None)])
    assert (rise["case"], rise["low_pass"], rise["last_pass"], rise["factor"], rise["limit"]) == ("only", 1, 3, 2.0, None)
    assert "compliance_limit" not in rise_sentence(rise) and "(2 times)" in rise_sentence(rise)
    # a case past the values of a row (an older row with fewer cases) uses the rows that have it
    assert compliance_rise([{"iteration": 0, "compliance_cases": [1.0]}, {"iteration": 1, "compliance_cases": [1.0, 2.0]}],
                           [("a", None), ("b", None)]) is None


@pytest.mark.parametrize("kind,want", [("roller", [("plate", "x"), ("floor_y", "y"), ("floor_z", "z")]), ("slider", [("plate", "yz"), ("stop", "x")])])
def test_the_per_axis_supports_of_a_config(tmp_path, kind, want):
    assert partial_supports(load_config(axes_case(fresh(tmp_path, "a"), kind))) == want
    assert partial_supports(load_config(axes_case(fresh(tmp_path, "b"), kind, mutate=full_clamps))) == []


def test_a_config_without_regions_has_no_per_axis_support():
    """The self-test of `auto` makes its problem with `regions=None` (`gpu_mg.selftest.run`); its solver is a `GpuSolver` too."""
    assert partial_supports(SimpleNamespace(regions=None)) == []


def test_a_fixed_entry_that_names_all_three_axes_is_a_full_clamp(tmp_path):
    def all_axes(d):
        d["regions"]["fixed"][0]["axes"] = ["x", "y", "z"]
    assert partial_supports(load_config(axes_case(tmp_path, "slider", mutate=all_axes))) == [("stop", "x")]


# --- part 2: the text of a real solver (OpenCL) -----------------------------------------------------------------------

opencl = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1


@pytest.fixture(scope="module")
def dev():
    pytest.importorskip("pyopencl")
    from topopt.config import UserError
    from topopt.gpu_mg import device
    try:
        return device.select("auto", "auto")[0]
    except UserError as e:
        pytest.skip(f"no usable OpenCL device: {e}")


def two_cases(d):
    """`mutate`: a second load case `side` with a compliance limit."""
    d["load_cases"].append({"name": "side", "compliance_limit": 10.0, "forces": [{"region": "tip", "force": [0.0, 1.0, 0.0]}]})


def text_of(tmp_path, mutate, history=None):
    """The text of a failed case C solve of a solver made for the roller case (a run folder with `history` rows, or no run writer)."""
    from topopt.gpu_mg.model import GpuModel, GpuParams
    from topopt.problem import build_problem
    from topopt.snapshot import RunWriter
    tmp_path.mkdir(parents=True, exist_ok=True)
    prob = build_problem(load_config(axes_case(fresh(tmp_path, "case"), "roller", h=2.0, solver="gpu-mg", gpu_case="C", mutate=mutate)))
    out = None
    if history is not None:
        out = RunWriter(tmp_path / "run", quiet=True)
        out.clean()
        for row in history:
            out.add_history(row)
    m = GpuModel(prob, out, GpuParams(min_free=300))
    try:
        return m.solver._failure_text(1e-5, 1e-9, "up to 300 PCG iterations in each of 4 refinement steps", True)
    finally:
        m.solver.free()


@opencl
@pytest.mark.usefixtures("dev")
def test_the_text_names_the_rise_and_the_per_axis_supports_and_their_options(tmp_path):
    text = text_of(tmp_path, two_cases, rows_of(ROLLER_C))                         # `side` (the second case) rises here
    assert ("The compliance of load case 'side' rose from 0.214 at pass 2 to 0.969 at pass 6 (4.53 times; its compliance_limit is 10), the largest "
            "rise of the load cases in the passes so far. The part has per-axis supports: plate (x), floor_y (y), floor_z (z). Options: ") in text
    assert ("; a lower compliance_limit for load case 'side' (now 10); full clamps in place of the per-axis supports (plate, floor_y, floor_z), "
            "if the part is held that way in use; ") in text
    assert text.endswith("or 'amg' (with 'auto' a gpu-mg failure does not end the run: it goes on with the CPU solver from the last checkpoint)")
    assert text.index("FP64 rounding floor") < text.index("The compliance of load case") < text.index("Options: ")       # the facts come first


@opencl
@pytest.mark.usefixtures("dev")
def test_without_a_limit_a_history_or_per_axis_supports_the_text_has_no_such_fact_or_option(tmp_path):
    no_limit = text_of(tmp_path / "a", None, rows_of([[v[1]] for v in ROLLER_C]))
    assert "rose from 0.214 at pass 2 to 0.969 at pass 6 (4.53 times), the largest" in no_limit and "a lower compliance_limit" not in no_limit
    no_history = text_of(tmp_path / "b", two_cases)                                  # no run writer: no history to read
    assert "The compliance of load case" not in no_history and "The part has per-axis supports" in no_history
    missing = text_of(tmp_path / "c", two_cases, [])                                 # a run folder without history.jsonl
    assert "The compliance of load case" not in missing

    def clamps_and_cases(d):
        full_clamps(d)
        two_cases(d)
    clamped = text_of(tmp_path / "d", clamps_and_cases, rows_of(ROLLER_C))
    assert "per-axis" not in clamped and "full clamps" not in clamped and "a lower compliance_limit for load case 'side'" in clamped
    assert clamped.count("Options: ") == 1
