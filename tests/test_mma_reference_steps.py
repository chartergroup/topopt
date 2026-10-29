"""The own MMA code (`topopt.mma_sub`) against recorded reference steps.

The file `tests/data/mma_reference_steps.npz` holds the outputs of a reference MMA implementation for fixed inputs
(who recorded it, when and how: `tests/data/mma_reference_steps.md`). The test runs the own code only and compares
its numbers with the recorded ones. It imports no other MMA code and needs no optional package.

What the file holds:

- 31 random problems (n 1 to 3000, m 1 to 6, 8 to 20 steps each). The inputs of a step are made again from the seed
  (`random_step`); the fixture keeps a digest of them, and a test fails with a clear message if numpy gives
  another random stream. The chain is made from the OWN outputs (the next x is the clipped own x, the next asymptotes are
  the own asymptotes): the reference was given the same arguments at every step, so the test replays the same chain.
  For n up to 64 the fixture keeps the whole x, low and upp of each step; for a larger n it keeps 64 fixed
  indices and the sum, the minimum and the maximum of the whole array.
- 5 steps of the `Mma` class (n 40) and the first step after a recorded state (n 60, two rows).
- 9 steps of two short real runs of the T-shirt hook, both with n = 955 design voxels (voxel size 4 mm in place of 2 mm and 2.4 mm):
  `examples/hook/hook_short.yaml` (the run of the golden file: one row, 40 passes) and `examples/hook/hook_stress.yaml` (two stress
  rows, 30 passes, a shorter beta schedule). The arguments of `mma_update` were taken from the own code during the run; the
  number in the name of a step is its call number in the run. Each step stands alone and is stored in full.
  Steps 1 and 2 use the start asymptotes; step 17 of the short run is the first pass of a new beta stage; step 36 has a
  row that is not met; the stress run has a row at the price limit (lam = c) and up to 146 Newton iterations.

Tolerances (test tolerances, not user settings). The measurement when the fixture was made, on these problems:
the largest differences of the own code against the reference were

    asymptotes low, upp   0 (bit equal), also when the chain follows the own outputs
    x                     2e-8 on the random problems; 1.3e-7 on the real steps (step 36 of the short run)
    lam                   1.3e-8 (abs; 8e-4 of the tolerance)      z 9e-13      y 4e-14

An earlier comparison of 2754 steps on 160 random problems found a larger x difference once: 1.6e-7, with move 0.03,
where the two Newton paths took another line-search step and both ended 1.4e-3 from the exact solution of the
subproblem, because the end value of the barrier parameter is 1e-7. So the limit for x is 1e-6.

Why the tolerances also hold on another platform. The fixture was recorded on Linux x86_64 (OpenBLAS); CI also runs
this file on macOS arm64 (Accelerate). The test does not need bit equality:

- The asymptotes are made by element-wise operations only (+, -, *, comparisons, min, max). IEEE 754 gives the same bits on
  every platform for the same inputs; so for a step with recorded inputs (the real steps, the recorded state) the limit is
  1e-14. In a chain the x of step k comes from the own solver, which uses small dense solves and sums over n (BLAS, LAPACK,
  pairwise sums). They differ between platforms by round-off, about 1e-16 relative per operation. The asymptotes of the next
  step move by the same amount. Measured as a stand-in for another platform (2026-10-26): the replay of this file with
  the OpenBLAS kernels HASWELL, ZEN, SANDYBRIDGE, NEHALEM and PRESCOTT (the last three have no FMA), 1 and 8 threads, and numpy
  without AVX2 and FMA moved the chained asymptotes by at most 7e-15, and the largest difference to the recorded x and lam did
  not change in three digits. So CHAIN_ASYMPTOTE_ATOL (1e-9) is more than 1e5 times that spread. It is not wider because a
  change of 1.4e-7 (relative) in a constant of the asymptote rule (ASYDECR 0.7 to 0.7000001) must still fail the test.
- x, lam and z come from the Newton iteration of the subproblem. Round-off changes its path by about 1e-13. The path
  can change by a whole step only if a decision (a line-search step, the test of the residual against epsilon) sits at its
  edge; then the end point moves by up to the size of the end value of epsilon. That is inside the limit of x (1e-6)
  by a factor of 8 or more of the largest difference ever measured (1.3e-7).
- The sum, the minimum and the maximum of a large array are tested as a mean: the mean of x may differ by X_ATOL, which every
  element is allowed to differ by. This covers the sum over 3000 elements with another order of additions.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from topopt import checkpoint as C      
from topopt import mma_sub as S         
from topopt.mma import Mma, MmaState    

FIXTURE = Path(__file__).parent / "data" / "mma_reference_steps.npz"

X_ATOL = 1e-6      # about 8 times the largest difference ever measured (1.3e-7, a real step); typical 1e-12
LAM_RTOL, LAM_ATOL = 1e-6, 1e-8
Z_ATOL = 1e-7      # also used for y
ASYMPTOTE_ATOL = 1e-14         # a step with recorded inputs: element-wise operations give the same bits on every platform
CHAIN_ASYMPTOTE_ATOL = 1e-9    # a chained step: the x of the step before has round-off of the platform (see the docstring)


@pytest.fixture(scope="module")
def ref():
    """The recorded numbers: a dict of arrays, and `meta` as a dict."""
    with np.load(FIXTURE) as z:
        data = {k: z[k] for k in z.files}
    data["meta"] = json.loads(str(data["meta"]))
    return data


def random_step(rng, n, m, kind):
    f0 = float(rng.uniform(0.5, 2.0))
    if kind == "topopt":      # like a compliance run: df0 < 0 and rows with dg > 0
        return (f0, -np.abs(rng.normal(size=n)) * rng.uniform(0.1, 3), rng.uniform(-0.3, 0.3, size=m),
                np.abs(rng.normal(size=(m, n))) * rng.uniform(0.2, 2) / n * 50)
    return (f0, rng.normal(size=n) * rng.uniform(0.1, 3), rng.normal(size=m) * rng.uniform(0.1, 2),
            rng.normal(size=(m, n)) * rng.uniform(0.2, 2))


def digest(f0, df0, g, dg):
    """A few numbers of the inputs of a step. They tell the test that numpy gave the same random problem as at the recording."""
    return [float(f0), float(np.sum(df0)), float(np.sum(g)), float(np.sum(dg))]


def check_inputs(recorded, f0, df0, g, dg, where):
    np.testing.assert_allclose(digest(f0, df0, g, dg), recorded, rtol=1e-12, atol=1e-12,
                               err_msg="the random inputs differ from the recorded ones (another numpy random stream?); "
                                       "record the fixture again: " + where)


# --- the random problems -------------------------------------------------------------------------------------------------------------

CASES = {}   # id -> dict(n, m, steps, seed, kw): the same table as the recording (31 problems)


def _add(cid, n, m, steps, seed, **kw):
    CASES[cid] = dict(n=n, m=m, steps=steps, seed=seed, kw=kw)


for _n in (10, 200, 3000):
    _add(f"one_row_n{_n}_a", _n, 1, 10, _n)
    _add(f"one_row_n{_n}_b", _n, 1, 10, _n + 1, kind="mixed", xmin=0.02, on_bounds=True)
for _n in (10, 200, 3000):
    for _m in (2, 5):
        _add(f"rows{_m}_n{_n}_a", _n, _m, 10, _n * _m, c_diff=True)
        _add(f"rows{_m}_n{_n}_b", _n, _m, 10, _n * _m + 1, kind="mixed", xmin=0.02, on_bounds=True)
for _n in (10, 200):
    for _m in (2, 4):
        _add(f"bound_n{_n}_m{_m}", _n, _m, 10, _n + _m, a_bound=True, c_diff=True, kind="mixed")
for _move in (0.01, 0.12, 0.5, 1.0):
    _add(f"move_{_move}", 300, 2, 8, 17, move=_move)
_add("quadratic_d", 300, 2, 8, 18, d_val=1.0, kind="mixed")
for _n, _m in ((1, 1), (2, 2), (3, 6)):
    _add(f"few_vars_n{_n}_m{_m}", _n, _m, 8, 40 + _n + _m, kind="mixed")
_add("twenty_steps", 50, 1, 20, 3)

REAL_STEPS = ["short_01", "short_02", "short_17", "short_36", "stress_01", "stress_02", "stress_14", "stress_23", "stress_26"]


def replay(ref, cid):
    """Run the own code along the chain of the problem `cid` and compare each step with the recorded one.

    Returns the largest differences (a dict) so that a measurement can read them.
    """
    spec = CASES[cid]
    n, m, steps, seed, kw = spec["n"], spec["m"], spec["steps"], spec["seed"], spec["kw"]
    a_bound, c_diff = kw.get("a_bound", False), kw.get("c_diff", False)
    xmin, xmax, kind = kw.get("xmin", 0.0), kw.get("xmax", 1.0), kw.get("kind", "topopt")
    on_bounds, move, d_val = kw.get("on_bounds", False), kw.get("move", 0.2), kw.get("d_val", 0.0)
    full = ref["meta"]["cases"][cid]["full"]
    idx = slice(None) if full else ref[f"{cid}__idx"]
    rng = np.random.default_rng(seed)
    x = rng.uniform(xmin, xmax, size=n)
    if on_bounds:
        pick = rng.random(n)
        x[pick < 0.3], x[pick > 0.7] = xmin, xmax
    a = np.zeros(m)
    if a_bound:
        a[: max(1, m // 2)] = 1.0
    c = rng.choice([100.0, 1000.0, 5000.0], size=(m, 1)).reshape(m) if c_diff else np.full(m, 1000.0)
    d = np.full(m, d_val)
    old1 = old2 = x.copy()
    low, upp = np.zeros(n), np.ones(n)
    worst = dict(x=0.0, low=0.0, upp=0.0, lam=0.0, z=0.0, y=0.0, stat_x=0.0, stat_low=0.0, stat_upp=0.0)
    for k in range(steps):
        it = k + 1
        where = f"{cid}: n={n} m={m} step {it}"
        f0, df0, g, dg = random_step(rng, n, m, kind)
        check_inputs(ref[f"{cid}__inp"][k], f0, df0, g, dg, where)
        res = S.mma_update(it, x, xmin, xmax, old1, old2, f0, df0, g, dg, low, upp, 1.0, a, c, d, move)
        r_x, r_low, r_upp = ref[f"{cid}__x"][k], ref[f"{cid}__low"][k], ref[f"{cid}__upp"][k]
        r_lam, r_y, r_z = ref[f"{cid}__lam"][k], ref[f"{cid}__y"][k], float(ref[f"{cid}__z"][k])
        for name, own, rec in (("x", res.x[idx], r_x), ("low", res.low[idx], r_low), ("upp", res.upp[idx], r_upp)):
            worst[name] = max(worst[name], float(np.max(np.abs(own - rec))))
        worst["lam"] = max(worst["lam"], float(np.max(np.abs(res.lam - r_lam))))
        worst["z"] = max(worst["z"], abs(res.z - r_z))
        worst["y"] = max(worst["y"], float(np.max(np.abs(res.y - r_y))))
        np.testing.assert_allclose(res.low[idx], r_low, rtol=0, atol=CHAIN_ASYMPTOTE_ATOL, err_msg="low " + where)
        np.testing.assert_allclose(res.upp[idx], r_upp, rtol=0, atol=CHAIN_ASYMPTOTE_ATOL, err_msg="upp " + where)
        np.testing.assert_allclose(res.x[idx], r_x, rtol=0, atol=X_ATOL, err_msg="x " + where)
        np.testing.assert_allclose(res.lam, r_lam, rtol=LAM_RTOL, atol=LAM_ATOL, err_msg="lam " + where)
        assert res.z == pytest.approx(r_z, abs=Z_ATOL), "z " + where
        np.testing.assert_allclose(res.y, r_y, rtol=0, atol=Z_ATOL, err_msg="y " + where)
        if not full:    # the whole array: its sum (as a mean), its minimum and its maximum
            for name, own, tol in (("x", res.x, X_ATOL), ("low", res.low, CHAIN_ASYMPTOTE_ATOL),
                                   ("upp", res.upp, CHAIN_ASYMPTOTE_ATOL)):
                s, lo, hi = ref[f"{cid}__stat_{name}"][k]
                worst[f"stat_{name}"] = max(worst[f"stat_{name}"], abs(own.sum() - s) / n, abs(own.min() - lo), abs(own.max() - hi))
                assert abs(own.sum() - s) <= tol * n, f"sum of {name} {where}"
                assert abs(own.min() - lo) <= tol and abs(own.max() - hi) <= tol, f"min or max of {name} {where}"
        old2, old1, x = old1, x.copy(), np.clip(res.x, xmin, xmax)
        low, upp = res.low, res.upp
    return worst


@pytest.mark.parametrize("n", [10, 200, 3000])
def test_one_row(ref, n):
    replay(ref, f"one_row_n{n}_a")
    replay(ref, f"one_row_n{n}_b")


@pytest.mark.parametrize("n", [10, 200, 3000])
@pytest.mark.parametrize("m", [2, 5])
def test_several_rows(ref, n, m):
    replay(ref, f"rows{m}_n{n}_a")
    replay(ref, f"rows{m}_n{n}_b")


@pytest.mark.parametrize("n", [10, 200])
@pytest.mark.parametrize("m", [2, 4])
def test_bound_form_with_a_equal_to_one(ref, n, m):
    replay(ref, f"bound_n{n}_m{m}")


@pytest.mark.parametrize("move", [0.01, 0.12, 0.5, 1.0])
def test_other_move_limits(ref, move):
    replay(ref, f"move_{move}")


def test_quadratic_price_d(ref):
    replay(ref, "quadratic_d")


@pytest.mark.parametrize("n, m", [(1, 1), (2, 2), (3, 6)])
def test_many_rows_and_few_variables_use_the_other_reduction(ref, n, m):
    replay(ref, f"few_vars_n{n}_m{m}")


def test_asymptotes_follow_the_reference_over_twenty_steps(ref):
    replay(ref, "twenty_steps")


# --- the Mma class and a recorded state ------------------------------------------------------------------------------------------------

def test_the_fixture_holds_the_cases_of_this_file(ref):
    """The table of the problems in this file is the table of the recording, and every array is there."""
    meta = ref["meta"]
    assert meta["format"] == 1
    assert {cid: {k: v for k, v in d.items() if k != "full"} for cid, d in meta["cases"].items()} == json.loads(json.dumps(CASES))
    assert sorted(k.split("__")[0].removeprefix("real_") for k in ref if k.startswith("real_") and k.endswith("__x")) == sorted(REAL_STEPS)
    assert sorted(meta["real"]) == sorted(f"real_{k}" for k in REAL_STEPS)
    for cid, d in meta["cases"].items():
        assert ref[f"{cid}__x"].shape == (d["steps"], d["n"] if d["full"] else len(ref[f"{cid}__idx"]))


def test_mma_class_follows_the_recorded_steps(ref):
    """`Mma` with one row gives the x of the recorded call (a0 1, a 0, c 1000, d 0) and keeps lam and the asymptotes."""
    n = 40
    rng = np.random.default_rng(0)
    mma = Mma(n, 1, 0.2)
    x = np.full(n, 0.4)
    for k in range(5):
        f0, df0 = float(rng.uniform(0.5, 1.5)), rng.normal(size=n)
        g, dg = float(rng.normal()), rng.normal(size=n)
        np.testing.assert_allclose(digest(f0, df0, g, dg), ref["mma_class__inp"][k], rtol=1e-12, atol=1e-12,
                                   err_msg="the random inputs differ from the recorded ones; record the fixture again")
        x_new = mma.step(x, f0, df0, np.array([g]), dg.reshape(1, n))
        np.testing.assert_allclose(x_new, ref["mma_class__x"][k], rtol=0, atol=X_ATOL)
        np.testing.assert_allclose(mma.lam, ref["mma_class__lam"][k], rtol=LAM_RTOL, atol=LAM_ATOL)
        np.testing.assert_allclose(mma.state.low.reshape(-1), ref["mma_class__low"][k], rtol=0, atol=CHAIN_ASYMPTOTE_ATOL)
        np.testing.assert_allclose(mma.state.upp.reshape(-1), ref["mma_class__upp"][k], rtol=0, atol=CHAIN_ASYMPTOTE_ATOL)
        x = x_new
    assert mma.state.it == 5 and mma.sub_converged is True and mma.newton > 0


def test_a_recorded_state_continues_in_the_mma_class(ref):
    """A checkpoint of main holds a state (x, xold1, xold2, low, upp, it) made by an earlier version. `Mma` goes on from it.

    The state is the state after four steps of the reference implementation (the recording made them alone). The fixture
    holds the state and the inputs of step 5, and the reference result of step 5.
    """
    n, m = 60, 2
    mma = Mma(n, m, 0.2)
    x = ref["continue__state_x"]
    mma.state = MmaState(n, it=4, x=x.copy(), xold1=ref["continue__state_xold1"].copy(), xold2=ref["continue__state_xold2"].copy(),
                         low=ref["continue__state_low"].reshape(n, 1).copy(), upp=ref["continue__state_upp"].reshape(n, 1).copy())
    new = mma.step(x, float(ref["continue__f0"]), ref["continue__df0"], ref["continue__g"], ref["continue__dg"])
    np.testing.assert_allclose(new, ref["continue__x"], rtol=0, atol=X_ATOL)
    np.testing.assert_allclose(mma.state.low.reshape(n), ref["continue__low"], rtol=0, atol=ASYMPTOTE_ATOL)
    np.testing.assert_allclose(mma.state.upp.reshape(n), ref["continue__upp"], rtol=0, atol=ASYMPTOTE_ATOL)
    assert mma.state.it == 5


def test_the_default_parameters_are_those_of_the_note(ref):
    """The 2007 note and the reference implementation agree on every default; the constants of `mma_sub` are the note's."""
    defaults = ref["meta"]["reference"]["defaults"]
    assert defaults["asyinit"] == S.ASYINIT == C.MMA_ASYINIT
    assert defaults["asydecr"] == S.ASYDECR and defaults["asyincr"] == S.ASYINCR
    assert defaults["asymin"] == S.ASYMIN and defaults["asymax"] == S.ASYMAX
    assert defaults["albefa"] == S.ALBEFA and defaults["raa0"] == S.RAA0


# --- the test has teeth ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cid, key, index, change, text", [
    ("one_row_n10_a", "x", (2, 0), 1e-5, "x one_row_n10_a"),             # a whole array: x is 10 times the limit away
    ("one_row_n10_a", "low", (3, 4), 1e-6, "low one_row_n10_a"),         # an asymptote of a chain
    ("rows2_n3000_a", "stat_x", (1, 0), 3000 * 2 * X_ATOL, "sum of x"),  # the sum of a large array (the mean is 2 limits away)
    ("rows2_n200_a", "lam", (4, 1), 1e-3, "lam rows2_n200_a"),
])
def test_a_changed_reference_fails_the_comparison(ref, cid, key, index, change, text):
    """The comparison is not vacuous: one recorded number moved beyond its tolerance gives a failure that names the problem."""
    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ref.items()}
    bad[f"{cid}__{key}"][index] += change
    with pytest.raises(AssertionError, match=text):
        replay(bad, cid)
    replay(ref, cid)        # the real fixture still passes


# --- steps from real runs ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", REAL_STEPS)
def test_a_step_of_a_real_run(ref, key):
    """The arguments of one `mma_update` call of a short real run; the recorded result of the reference for the same arguments."""
    p = f"real_{key}__"
    args = [ref[p + "in_" + nm] for nm in ("it", "x", "xmin", "xmax", "xold1", "xold2", "f0", "df0", "f", "dg", "low", "upp",
                                          "a0", "a", "c", "d", "move")]
    args[0] = int(args[0])
    for i in (2, 3, 6, 12, 16):     # the scalars
        args[i] = float(args[i])
    res = S.mma_update(*args)
    where = f"{key} (n={ref['meta']['real']['real_' + key]['n']})"
    np.testing.assert_allclose(res.low, ref[p + "low"], rtol=0, atol=ASYMPTOTE_ATOL, err_msg="low " + where)
    np.testing.assert_allclose(res.upp, ref[p + "upp"], rtol=0, atol=ASYMPTOTE_ATOL, err_msg="upp " + where)
    np.testing.assert_allclose(res.x, ref[p + "x"], rtol=0, atol=X_ATOL, err_msg="x " + where)
    np.testing.assert_allclose(res.lam, ref[p + "lam"], rtol=LAM_RTOL, atol=LAM_ATOL, err_msg="lam " + where)
    assert res.z == pytest.approx(float(ref[p + "z"]), abs=Z_ATOL), "z " + where
    np.testing.assert_allclose(res.y, ref[p + "y"], rtol=0, atol=Z_ATOL, err_msg="y " + where)
    assert res.converged
