"""Tests of `topopt.mma_sub`, the own MMA code (written from the 2007 note of Svanberg).

These tests need no other MMA code. The comparison with recorded reference steps is in `test_mma_reference_steps.py`.

Equation numbers are the numbers of the 2007 note ([S07]). The interior-point method ends at the barrier
parameter epsilon = 1e-7, so a solution has an error of the order of 1e-7 (times the inverse of the curvature):
the tests of known solutions use 1e-5 for x and for the multipliers.
"""

import subprocess
import sys

import numpy as np
import pytest

from test_optimize import make_case
from topopt import mma_sub as S

SOLVE_TOL = 1e-5     # error of a subproblem solution at epsilon = 1e-7


def sub(alpha, beta, low, upp, p0, q0, P, Q, b, a0=1.0, a=None, c=None, d=None):
    """A Subproblem from lists. Defaults: a = 0, c = 1000, d = 0 for each row."""
    P, Q = np.atleast_2d(np.asarray(P, float)), np.atleast_2d(np.asarray(Q, float))
    m = P.shape[0]
    return S.Subproblem(*(np.asarray(v, float) for v in (alpha, beta, low, upp, p0, q0)), P, Q,
                        np.asarray(b, float), float(a0),
                        np.zeros(m) if a is None else np.asarray(a, float),
                        np.full(m, 1000.0) if c is None else np.asarray(c, float),
                        np.zeros(m) if d is None else np.asarray(d, float))


# --- asymptotes and move bounds ------------------------------------------------------------------


def test_first_two_calls_use_the_start_asymptotes_whatever_the_old_ones_are():
    x = np.array([0.2, 0.5, 0.9])
    for it in (1, 2):
        low, upp = S.update_asymptotes(it, x, x * 0 + 7, x * 0 - 3, x * 0 + 99, x * 0 - 99, 0.0, 1.0)
        np.testing.assert_array_equal(low, x - 0.5)      # (3.11)
        np.testing.assert_array_equal(upp, x + 0.5)
    low, upp = S.update_asymptotes(1, x, x, x, x, x, 0.02, 2.02)     # the range is 2: 0.5 * 2 = 1
    np.testing.assert_allclose(low, x - 1.0, atol=1e-15)
    np.testing.assert_allclose(upp, x + 1.0, atol=1e-15)


def test_asymptote_rules_from_call_3_on_oscillation_growth_and_clips():
    # one column for each case. Range 1 (xmin 0, xmax 1).
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    xold1 = np.array([0.4, 0.6, 0.5, 0.5, 0.5])
    xold2 = np.array([0.3, 0.5, 0.2, 0.5, 0.5])
    low0 = np.array([0.0, 0.0, 0.0, 0.49999, -20.0])
    upp0 = np.array([1.0, 1.0, 1.0, 0.50001, 30.0])
    low, upp = S.update_asymptotes(3, x, xold1, xold2, low0, upp0, 0.0, 1.0)
    # 0: (0.5-0.4)*(0.4-0.3) > 0, so gamma = 1.2: l = 0.5 - 1.2*(0.4-0) = 0.02, u = 0.5 + 1.2*(1-0.4) = 1.22   (3.12), (3.13)
    # 1: (0.5-0.6)*(0.6-0.5) < 0, so gamma = 0.7: l = 0.5 - 0.7*0.6 = 0.08, u = 0.5 + 0.7*0.4 = 0.78
    # 2: x = xold1, the product is 0, so gamma = 1: l = 0.5 - 0.5 = 0, u = 0.5 + 0.5 = 1
    # 3: the same gamma, but l = 0.49999 is above x - 0.01 and u = 0.50001 is below x + 0.01: both are clipped   (3.14)
    # 4: l = -20 is below x - 10 and u = 30 is above x + 10: both are clipped
    np.testing.assert_allclose(low, [0.02, 0.08, 0.0, 0.49, -9.5], atol=1e-14)
    np.testing.assert_allclose(upp, [1.22, 0.78, 1.0, 0.51, 10.5], atol=1e-14)


def test_asymptotes_follow_over_four_calls():
    """Calls 1 to 4 with a design that moves on in one direction (0.5, 0.6, 0.7), then turns back (0.65)."""
    xs = [0.5, 0.6, 0.7, 0.65]
    low = upp = np.zeros(1)
    seen = []
    for k, x in enumerate(xs, start=1):
        old1 = xs[max(k - 2, 0)]
        old2 = xs[max(k - 3, 0)]
        low, upp = S.update_asymptotes(k, np.array([x]), np.array([old1]), np.array([old2]), low, upp, 0.0, 1.0)
        seen.append((float(low[0]), float(upp[0])))
    # calls 1 and 2: x -/+ 0.5                                                                     (3.11)
    assert seen[0] == pytest.approx((0.0, 1.0)) and seen[1] == pytest.approx((0.1, 1.1))
    # call 3: (0.7-0.6)*(0.6-0.5) > 0, gamma 1.2: l = 0.7 - 1.2*(0.6-0.1) = 0.1, u = 0.7 + 1.2*(1.1-0.6) = 1.3   (3.12)
    assert seen[2] == pytest.approx((0.1, 1.3))
    # call 4: (0.65-0.7)*(0.7-0.6) < 0, gamma 0.7: l = 0.65 - 0.7*(0.7-0.1) = 0.23, u = 0.65 + 0.7*(1.3-0.7) = 1.07
    assert seen[3] == pytest.approx((0.23, 1.07))


def test_move_bounds_are_the_largest_of_three_and_the_smallest_of_three():
    # (3.6), (3.7). x = 0.5 between the asymptotes 0 and 1: the move limit 0.2 is the tightest.
    a, b = S.move_bounds(np.array([0.5]), np.array([0.0]), np.array([1.0]), 0.0, 1.0, 0.2)
    np.testing.assert_allclose([a[0], b[0]], [0.3, 0.7])
    # x = 0.1 with the lower asymptote close: 0.1 of the way (0.1*0.1 = 0.01 above l) is tighter than the move limit
    a, b = S.move_bounds(np.array([0.1]), np.array([0.0]), np.array([1.0]), 0.0, 1.0, 0.5)
    np.testing.assert_allclose([a[0], b[0]], [0.01, 0.6])
    # x on the bound: the bound is kept, the other side moves
    a, b = S.move_bounds(np.array([1.0]), np.array([0.5]), np.array([1.5]), 0.0, 1.0, 0.2)
    np.testing.assert_allclose([a[0], b[0]], [0.8, 1.0])
    # a bigger range: the move limit is a part of (xmax - xmin)
    a, b = S.move_bounds(np.array([1.0]), np.array([-4.0]), np.array([6.0]), 0.0, 2.0, 0.25)
    np.testing.assert_allclose([a[0], b[0]], [0.5, 1.5])


# --- the convex approximation --------------------------------------------------------------------------


def test_p_and_q_values_and_the_first_order_match():
    x = np.array([0.5, 0.5, 0.5])
    low, upp = x - 0.5, x + 0.5
    df0 = np.array([2.0, -3.0, 0.0])
    dg = np.array([[1.0, 1.0, -1.0], [-2.0, 0.5, 0.0]])
    f = np.array([0.3, -0.2])
    p0, q0, P, Q, b = S.approximation(x, low, upp, 0.0, 1.0, f, df0, dg)
    # (3.3), (3.4): variable 0 of the objective, df = 2 > 0: p = 0.5^2 * (1.001*2 + 1e-5), q = 0.5^2 * (0.001*2 + 1e-5)
    assert p0[0] == pytest.approx(0.25 * (1.001 * 2 + 1e-5), rel=1e-14)
    assert q0[0] == pytest.approx(0.25 * (0.001 * 2 + 1e-5), rel=1e-14)
    # df = -3 < 0: the roles turn round
    assert p0[1] == pytest.approx(0.25 * (0.001 * 3 + 1e-5), rel=1e-14)
    assert q0[1] == pytest.approx(0.25 * (1.001 * 3 + 1e-5), rel=1e-14)
    # df = 0: both get the floor
    assert p0[2] == pytest.approx(0.25 * 1e-5, rel=1e-14) and q0[2] == p0[2]
    # the approximation has the value and the gradient of the function at x   (3.5), (4.10)
    ux, xl = upp - x, x - low
    np.testing.assert_allclose(P @ (1 / ux) + Q @ (1 / xl) - b, f, atol=1e-14)
    np.testing.assert_allclose(P / ux**2 - Q / xl**2, dg, atol=1e-14)
    np.testing.assert_allclose(p0 / ux**2 - q0 / xl**2, df0, atol=1e-14)


def test_the_floor_of_p_and_q_uses_the_range():
    x = np.array([1.0])
    p0, q0, *_ = S.approximation(x, x - 2.0, x + 2.0, 0.0, 4.0, np.zeros(1), np.zeros(1), np.zeros((1, 1)))
    assert p0[0] == pytest.approx(4.0 * 1e-5 / 4.0)      # (u-x)^2 * raa0 / (xmax - xmin) = 4 * 1e-5 / 4


# --- the subproblem: known solutions ---------------------------------------------------------------


def one_var(b, c=1000.0, beta=1.5):
    """minimize 1/x subject to 1/(2-x) - y <= b, 0.2 <= x <= beta; asymptotes l = 0, u = 2."""
    return sub([0.2], [beta], [0.0], [2.0], [0.0], [1.0], [[1.0]], [[0.0]], [b], c=[c])


@pytest.mark.parametrize("system", [None, "dual", "primal"])
def test_one_variable_the_constraint_is_active(system):
    # 1/(2-x) <= 1 gives x <= 1. The objective 1/x falls, so x = 1. The multiplier: -1/x^2 + lam/(2-x)^2 = 0, lam = 1.
    r = S.solve_subproblem(one_var(1.0), system=system)
    assert r.converged
    assert r.x[0] == pytest.approx(1.0, abs=SOLVE_TOL) and r.lam[0] == pytest.approx(1.0, abs=SOLVE_TOL)
    assert r.y[0] == pytest.approx(0.0, abs=SOLVE_TOL) and r.z == pytest.approx(0.0, abs=SOLVE_TOL)


def test_one_variable_the_move_bound_is_active_and_the_row_is_free():
    r = S.solve_subproblem(one_var(10.0, beta=1.5))
    assert r.x[0] == pytest.approx(1.5, abs=SOLVE_TOL) and r.lam[0] == pytest.approx(0.0, abs=SOLVE_TOL)
    assert r.eta[0] == pytest.approx(1 / 1.5**2, abs=SOLVE_TOL)   # the multiplier of the bound is the slope -d(1/x)/dx


def test_one_variable_the_row_cannot_be_met_and_the_price_decides():
    # 1/(2-x) <= 0.4 needs x <= -0.5: not possible with x >= 0.2. With the price 1000 the row loses: x stays
    # at 0.2 (the objective 1/x + 1000*y rises with x there), y = 1/1.8 - 0.4, and lam = c.
    r = S.solve_subproblem(one_var(0.4, c=1000.0))
    assert r.x[0] == pytest.approx(0.2, abs=SOLVE_TOL)
    assert r.y[0] == pytest.approx(1 / 1.8 - 0.4, abs=SOLVE_TOL) and r.lam[0] == pytest.approx(1000.0, rel=1e-5)
    # with the price 1 the balance -1/x^2 + 1/(2-x)^2 = 0 gives x = 1, y = 1/(2-1) - 0.4 = 0.6, lam = c = 1
    r = S.solve_subproblem(one_var(0.4, c=1.0))
    assert r.x[0] == pytest.approx(1.0, abs=SOLVE_TOL) and r.y[0] == pytest.approx(0.6, abs=SOLVE_TOL)
    assert r.lam[0] == pytest.approx(1.0, rel=1e-5)


def test_one_variable_with_a_quadratic_price_d():
    # minimize 1/x + c*y + 0.5*d*y^2 with 1/(2-x) - y <= 0.4, c = 1, d = 2: stationarity in y: c + d*y = lam.
    # In x: -1/x^2 + lam/(2-x)^2 = 0 with lam = 1 + 2*y and y = 1/(2-x) - 0.4. Solve by bisection here.
    from scipy.optimize import brentq
    f = lambda x: -1 / x**2 + (1 + 2 * (1 / (2 - x) - 0.4)) / (2 - x) ** 2   # noqa: E731
    x_star = brentq(f, 0.2, 1.5, xtol=1e-14)
    sp = sub([0.2], [1.5], [0.0], [2.0], [0.0], [1.0], [[1.0]], [[0.0]], [0.4], c=[1.0], d=[2.0])
    r = S.solve_subproblem(sp)
    assert r.x[0] == pytest.approx(x_star, abs=SOLVE_TOL)
    assert r.y[0] == pytest.approx(1 / (2 - x_star) - 0.4, abs=SOLVE_TOL)


@pytest.mark.parametrize("system", [None, "dual", "primal"])
def test_two_variables_with_a_hand_solution(system):
    # minimize 1/x1 + 4/x2 subject to 1/(2-x1) + 1/(2-x2) <= 2. Lagrange: x_j = 2/(1 + sqrt(lam/q_j)); the row gives
    # sqrt(lam) = 1.5, so lam = 2.25, x1 = 0.8 and x2 = 8/7.
    sp = sub([0.1, 0.1], [1.9, 1.9], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [1.0, 4.0], [[1.0, 1.0]], [[0.0, 0.0]], [2.0])
    r = S.solve_subproblem(sp, system=system)
    np.testing.assert_allclose(r.x, [0.8, 8 / 7], atol=SOLVE_TOL)
    assert r.lam[0] == pytest.approx(2.25, abs=SOLVE_TOL)


@pytest.mark.parametrize("n, system", [(1, None), (3, None), (3, "primal")])
def test_bound_form_a_equals_one_gives_the_minimax_and_z(n, system):
    # minimize z subject to h1(x) <= z and h2(x) <= z, with h1 = sum 1/x_j and h2 = sum 1/(2-x_j). Symmetric: x_j = 1,
    # z = n. The multipliers: lam1 + lam2 = a0 = 1 (the row for z), lam1 = lam2 = 0.5.
    ones = np.ones(n)
    sp = sub(0.2 * ones, 1.8 * ones, 0 * ones, 2 * ones, 0 * ones, 0 * ones, [0 * ones, ones], [ones, 0 * ones],
             [0.0, 0.0], a=[1.0, 1.0])
    r = S.solve_subproblem(sp, system=system)
    np.testing.assert_allclose(r.x, ones, atol=SOLVE_TOL)
    assert r.z == pytest.approx(n, abs=SOLVE_TOL * n)
    np.testing.assert_allclose(r.lam, [0.5, 0.5], atol=SOLVE_TOL)


def random_sub(rng, n, m, a=False):
    x = rng.uniform(0.1, 0.9, n)
    low, upp = x - rng.uniform(0.3, 1.0, n), x + rng.uniform(0.3, 1.0, n)
    alpha, beta = np.maximum(0.0, x - 0.2), np.minimum(1.0, x + 0.2)
    p0, q0, P, Q, b = S.approximation(x, low, upp, 0.0, 1.0, rng.normal(size=m) * 0.3, rng.normal(size=n),
                                      rng.normal(size=(m, n)))
    return S.Subproblem(alpha, beta, low, upp, p0, q0, P, Q, b, 1.0,
                        (rng.random(m) < 0.5).astype(float) if a else np.zeros(m),
                        rng.choice([100.0, 1000.0], size=m), np.zeros(m))


def kkt_errors(sp, r):
    """The largest violation of the KKT conditions (5.7a)-(5.7n) of the subproblem (5.1) at the solution r."""
    ux, xl = sp.upp - r.x, r.x - sp.low
    pl, ql = sp.p0 + sp.P.T @ r.lam, sp.q0 + sp.Q.T @ r.lam
    stationary = pl / ux**2 - ql / xl**2 - r.xi + r.eta                                    # (5.7a)
    g = sp.P @ (1 / ux) + sp.Q @ (1 / xl)
    primal = g - sp.a * r.z - r.y - sp.b                                                    # (5.7d): <= 0
    return dict(
        stationary_x=np.abs(stationary).max(),
        stationary_y=np.abs(sp.c + sp.d * r.y - r.lam - r.mu).max(),                        # (5.7b)
        stationary_z=abs(sp.a0 - r.zeta - r.lam @ sp.a),                                    # (5.7c)
        primal_violation=max(0.0, primal.max()),
        slack=np.abs(r.lam * primal).max(),                                                 # (5.7e)
        slack_bounds=max(np.abs(r.xi * (r.x - sp.alpha)).max(), np.abs(r.eta * (sp.beta - r.x)).max(),
                         np.abs(r.mu * r.y).max(), abs(r.zeta * r.z)),                       # (5.7f)-(5.7i)
        outside=max(0.0, (sp.alpha - r.x).max(), (r.x - sp.beta).max(), -r.y.min(), -r.z, -r.lam.min(),
                    -r.xi.min(), -r.eta.min(), -r.mu.min(), -r.zeta))
    # (5.7j)-(5.7n): the signs


@pytest.mark.parametrize("n, m, a", [(8, 1, False), (8, 3, False), (20, 4, True), (3, 5, False), (4, 4, True)])
def test_kkt_conditions_hold_at_the_solution_of_random_problems(n, m, a):
    rng = np.random.default_rng(1000 * n + m)
    for _ in range(5):
        sp = random_sub(rng, n, m, a)
        r = S.solve_subproblem(sp)
        assert r.converged
        e = kkt_errors(sp, r)
        assert e["outside"] == 0.0
        assert e["stationary_x"] < 1e-6 and e["stationary_y"] < 1e-6 and e["stationary_z"] < 1e-6, e
        assert e["primal_violation"] < 1e-6 and e["slack_bounds"] < 3e-7, e
        assert e["slack"] < 3e-7 * (1.0 + r.lam.max()), e    # lam * (residual of 5.9d): the multiplier scales it


@pytest.mark.parametrize("n, m", [(10, 2), (30, 5), (5, 2)])
def test_both_reductions_of_the_newton_system_give_the_same_solution(n, m):
    rng = np.random.default_rng(n * 7 + m)
    for a in (False, True):
        sp = random_sub(rng, n, m, a)
        dual = S.solve_subproblem(sp, system="dual")
        primal = S.solve_subproblem(sp, system="primal")
        np.testing.assert_allclose(dual.x, primal.x, atol=1e-8)
        np.testing.assert_allclose(dual.lam, primal.lam, rtol=1e-6, atol=1e-8)


def test_the_solver_takes_the_smaller_system():
    """n > m takes (5.20) and m >= n takes (5.22): the results of the automatic choice are those of the forced one."""
    rng = np.random.default_rng(5)
    big, small = random_sub(rng, 6, 2), random_sub(rng, 2, 3)
    np.testing.assert_array_equal(S.solve_subproblem(big).x, S.solve_subproblem(big, system="dual").x)
    np.testing.assert_array_equal(S.solve_subproblem(small).x, S.solve_subproblem(small, system="primal").x)


# --- mma_update: the whole iteration --------------------------------------------------------------------


def run_steps(rng, n, m, steps, move=0.2, a=None, c=None, bounds=(0.0, 1.0)):
    xmin, xmax = bounds
    x = rng.uniform(xmin, xmax, n)
    old1 = old2 = x
    low = upp = np.zeros(n)
    out = []
    for it in range(1, steps + 1):
        df0, dg, g = -np.abs(rng.normal(size=n)), np.abs(rng.normal(size=(m, n))) / n * 30, rng.uniform(-0.3, 0.3, m)
        res = S.mma_update(it, x, xmin, xmax, old1, old2, 1.0, df0, g, dg, low, upp, 1.0,
                           np.zeros(m) if a is None else a, np.full(m, 1000.0) if c is None else c, np.zeros(m), move)
        out.append((x, res))
        old2, old1, x, low, upp = old1, x, res.x, res.low, res.upp
    return out


@pytest.mark.parametrize("m", [1, 3])
def test_the_new_design_stays_in_the_move_bounds_and_the_state_is_finite(m):
    rng = np.random.default_rng(m)
    for x, res in run_steps(rng, 200, m, steps=6, move=0.15):
        assert res.converged and np.isfinite(res.x).all() and np.isfinite(res.low).all() and np.isfinite(res.upp).all()
        assert (res.x >= 0.0).all() and (res.x <= 1.0).all()
        assert np.abs(res.x - x).max() <= 0.15 + 1e-12      # the move limit is a part of the range
        assert res.low.shape == res.upp.shape == res.x.shape == (200,) and res.lam.shape == (m,)
        assert (res.low < x).all() and (res.upp > x).all()


def test_a_row_that_is_met_by_a_large_margin_has_a_multiplier_of_zero():
    rng = np.random.default_rng(3)
    n = 50
    x = np.full(n, 0.5)
    df0 = -np.abs(rng.normal(size=n)) - 0.1
    res = S.mma_update(1, x, 0.0, 1.0, x, x, 1.0, df0, np.array([-5.0]), np.full((1, n), 0.01), x, x, 1.0,
                       np.zeros(1), np.array([1000.0]), np.zeros(1), 0.2)
    assert res.lam[0] < 1e-6 and res.z == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(res.x, 0.7)      # every variable goes to the upper move bound: the gradient of the objective is < 0


@pytest.mark.parametrize("m", [1, 2, 4])
def test_two_variables_with_equal_data_get_equal_bits(m):
    """The mirror test needs this: a variable and its image have the same data and must get the same bits,
    for all steps (the asymptote rules too). The sums over the variables are shared scalars."""
    rng = np.random.default_rng(10 + m)
    half = 37
    sym = lambda v: np.concatenate([v, v[..., ::-1]], axis=-1)    # noqa: E731  the image of i is n - 1 - i
    n = 2 * half
    x = sym(rng.uniform(0.0, 1.0, half))
    x[: half // 3] = x[n - half // 3:] = 0.0      # some variables on the bound
    old1, old2 = x, x
    low = upp = np.zeros(n)
    for it in range(1, 7):
        df0 = sym(-np.abs(rng.normal(size=half)))
        dg = sym(np.abs(rng.normal(size=(m, half))) / n * 30)
        res = S.mma_update(it, x, 0.0, 1.0, old1, old2, 1.0, df0, rng.uniform(-0.3, 0.3, m), dg, low, upp, 1.0,
                           np.zeros(m), np.full(m, 1000.0), np.zeros(m), 0.2)
        for v in (res.x, res.low, res.upp):
            np.testing.assert_array_equal(v, v[::-1])
        old2, old1, x, low, upp = old1, x, res.x, res.low, res.upp


def test_a_permutation_of_the_variables_gives_a_permuted_result_up_to_round_off():
    """Bit equality under a permutation is not possible: a sum over the variables adds in another order.
    The element-wise parts (the asymptotes) stay bit equal; x agrees to round-off times the conditioning."""
    rng = np.random.default_rng(8)
    n, m = 300, 2
    perm = rng.permutation(n)
    x = rng.uniform(0.0, 1.0, n)
    df0, dg, g = -np.abs(rng.normal(size=n)), np.abs(rng.normal(size=(m, n))) / n * 30, np.array([0.1, -0.1])
    rows = (np.zeros(m), np.full(m, 1000.0), np.zeros(m), 0.2)
    base = S.mma_update(1, x, 0.0, 1.0, x, x, 1.0, df0, g, dg, x, x, 1.0, *rows)
    xp = x[perm]
    permuted = S.mma_update(1, xp, 0.0, 1.0, xp, xp, 1.0, df0[perm], g, dg[:, perm], xp, xp, 1.0, *rows)
    np.testing.assert_allclose(permuted.x, base.x[perm], atol=1e-10)
    np.testing.assert_array_equal(permuted.low, base.low[perm])
    np.testing.assert_array_equal(permuted.upp, base.upp[perm])


def test_xmin_and_xmax_may_be_arrays_and_a_range_other_than_one():
    rng = np.random.default_rng(2)
    n = 20
    lo, hi = np.full(n, 0.02), np.full(n, 3.02)
    x = rng.uniform(0.5, 2.5, n)
    args = (1, x, lo, hi, x, x, 1.0, -np.abs(rng.normal(size=n)), np.array([0.1]), np.abs(rng.normal(size=(1, n))), x, x,
            1.0, np.zeros(1), np.array([1000.0]), np.zeros(1), 0.1)
    as_arrays = S.mma_update(*args)
    as_numbers = S.mma_update(*(args[:2] + (0.02, 3.02) + args[4:]))
    np.testing.assert_array_equal(as_arrays.x, as_numbers.x)
    assert (np.abs(as_arrays.x - x) <= 0.1 * 3.0 + 1e-12).all()      # move is a part of xmax - xmin = 3
    np.testing.assert_allclose(as_arrays.low, x - 1.5, atol=1e-14)      # the start asymptotes: 0.5 * (xmax - xmin) = 1.5
    np.testing.assert_allclose(as_arrays.upp, x + 1.5, atol=1e-14)


# --- the core needs no optional package -------------------------------------------------------------------

OPTIONAL_PACKAGES = ("gmsh", "cholespy", "skfem", "pyopencl", "pypardiso", "OCP", "mcp")   # the modules of the extras


def test_a_short_run_and_the_imports_work_with_every_optional_package_blocked(tmp_path):
    """A subprocess where `import <name>` fails for each module of an extra: the imports of the optimizer, an Mma step and a
    short run need none of them. So a plain install (the core dependencies only) runs, and the MMA code needs no package."""
    p = make_case(tmp_path, h=2.0, iterations=3)
    out = tmp_path / "run"
    code = (
        "import sys\n"
        f"for name in {OPTIONAL_PACKAGES!r}:\n"
        "    sys.modules[name] = None\n"                 # `import <name>` now raises ImportError
        "import topopt.optimize, topopt.formulation, topopt.mma, topopt.cli\n"
        "from topopt.mma import Mma\n"
        "import numpy as np\n"
        "m = Mma(5, 1)\n"
        "x = m.step(np.full(5, 0.5), 1.0, -np.ones(5), np.array([0.1]), np.ones((1, 5)))\n"
        "assert x.shape == (5,)\n"
        "rc = topopt.cli.main(['run', sys.argv[1], '--out', sys.argv[2], '--quiet'])\n"
        "assert rc == 0, rc\n"
        f"loaded = [k for k in sys.modules if k.split('.')[0] in {OPTIONAL_PACKAGES!r} and sys.modules[k] is not None]\n"
        "assert not loaded, loaded\n")
    r = subprocess.run([sys.executable, "-c", code, str(p), str(out)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "result" / "final.stl").exists() or any(out.rglob("final.stl"))


def test_a_subproblem_that_stops_early_gives_a_log_warning(monkeypatch, caplog):
    """A guard of the subproblem (not a part of the method) ends the solve: the step is used, and the log says so."""
    import functools
    import logging
    from topopt import mma_sub
    from topopt.mma import Mma
    monkeypatch.setattr(mma_sub, "solve_subproblem", functools.partial(mma_sub.solve_subproblem, max_newton=1))
    mma = Mma(4, 1)
    x = np.full(4, 0.5)
    with caplog.at_level(logging.WARNING, logger="topopt.mma"):
        new = mma.step(x, 1.0, -np.ones(4), np.array([0.1]), np.ones((1, 4)))
    assert mma.sub_converged is False and mma.newton == 1
    assert np.all((new >= 0.0) & (new <= 1.0))
    assert "the subproblem stopped after 1 Newton iterations" in caplog.text
