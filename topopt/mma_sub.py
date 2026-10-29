"""The MMA update, written from the published note of Svanberg (2007) (clean room, MIT).

Source of this file: [S07] K. Svanberg (2007), "MMA and GCMMA - two methods for nonlinear
optimization" (a technical note of the author, KTH Stockholm). Equation numbers in the comments
below are the numbers of this note: (1.1) the problem, (3.1)-(3.14) the MMA subproblem and the
asymptote rules, (5.1)-(5.22) the primal-dual interior-point method for the subproblem, section 5.4
the line search, section 5.5 the algorithm. The note describes the method of [S87] K. Svanberg
(1987), "The method of moving asymptotes - a new method for structural optimization", Int. J.
Numer. Meth. Engng 24:359-373 (the origin of the method; it is cited here and was not read).
No other MMA code was used: the file is not a port of any program.

One call of `mma_update` is one outer iteration of MMA. It makes the convex approximation at the
design `x`, and solves the subproblem

    minimize   g0(x) + a0*z + sum_i (c_i*y_i + 0.5*d_i*y_i^2)
    subject to g_i(x) - a_i*z - y_i <= b_i,   alpha_j <= x_j <= beta_j,   y >= 0, z >= 0

with the primal-dual interior-point method of [S07] section 5. The system for the Newton direction
is reduced to the size m + 1 when m < n (5.20), and to the size n + 1 otherwise (5.22).

Design rule for the mirror test: each value that belongs to ONE variable (asymptotes, bounds,
p and q, the Newton direction in x) is made by element-wise operations only. A sum over the
variables gives a scalar (or a vector of the m rows) that all variables share. So two variables
with the same data get the same bits, for any number of rows (a design and its mirror image stay
equal bit for bit). A permutation of the variables gives a permuted result up to the round-off of
the sums over the variables, not bit for bit. For this reason a sum over the rows (the m rows of
one variable) is a loop over the rows, not a matrix product: a BLAS kernel may treat the tail of a
vector in another way than the body.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# --- constants of the method; each one is a value of [S07] ----------------------------------------

ASYINIT = 0.5     # (3.11): the first two asymptotes are x -/+ ASYINIT * (xmax - xmin); the note calls it `asyinit`
ASYDECR = 0.7     # (3.13): the asymptotes move closer when x oscillates; the note calls it `asydecr`
ASYINCR = 1.2     # (3.13): the asymptotes move away when x moves in one direction; `asyincr`
ASYMIN = 0.01     # (3.14): lowest distance of an asymptote from x, times (xmax - xmin)
ASYMAX = 10.0     # (3.14): highest distance of an asymptote from x, times (xmax - xmin)
ALBEFA = 0.1      # (3.6), (3.7): a step stays this part of the way away from an asymptote; `albefa`
RAA0 = 1e-5       # (3.3), (3.4): the term that keeps p and q above zero; `raa0`
PQ_NEAR = 1.001   # (3.3), (3.4): weight of the positive part of the gradient in p, and of the negative part in q
PQ_FAR = 0.001    # (3.3), (3.4): weight of the other part (the negative part in p, the positive part in q)
EPS_START = 1.0   # 5.5: the first value of the barrier parameter epsilon
EPS_END = 1e-7    # 5.5: the end value of epsilon ("say eps <= 1e-7")
EPS_FACTOR = 0.1  # 5.5 step 4: epsilon is multiplied by this when the residual is small
EPS_TEST = 0.9    # 5.5 step 4: the residual (max norm) must be below this part of epsilon
EDGE = 0.01       # 5.4: a step stops at this part of the distance to a bound of a positive variable

# --- guards that are NOT from the note (they are limits of this code, not parts of the method) --------------------------------

MAX_NEWTON = 1000        # Newton iterations in one subproblem. The note gives no limit; real runs use 20-60, random hard ones up to 410.
MAX_HALVINGS = 60        # halvings of the step in the line search of 5.4 (the note says one always exists)


class Subproblem(NamedTuple):
    """The data of the subproblem (5.1): the convex approximation, the bounds and the price vectors."""

    alpha: np.ndarray   # (n,) lower move bound
    beta: np.ndarray    # (n,) upper move bound
    low: np.ndarray     # (n,) lower asymptote l
    upp: np.ndarray     # (n,) upper asymptote u
    p0: np.ndarray      # (n,) p of the objective row
    q0: np.ndarray      # (n,)
    P: np.ndarray       # (m, n) p of the constraint rows
    Q: np.ndarray       # (m, n)
    b: np.ndarray       # (m,) right hand side b_i = -r_i
    a0: float
    a: np.ndarray       # (m,)
    c: np.ndarray       # (m,)
    d: np.ndarray       # (m,)


class Solution(NamedTuple):
    """The optimal point of the subproblem and the multipliers (names of (5.9))."""

    x: np.ndarray
    y: np.ndarray
    z: float
    lam: np.ndarray
    xi: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    zeta: float
    s: np.ndarray
    newton: int         # Newton iterations used
    converged: bool     # False when a guard ended the solve before epsilon reached its end value


class MmaResult(NamedTuple):
    """The result of one MMA iteration. `low` and `upp` are the asymptotes that the next iteration needs."""

    x: np.ndarray
    y: np.ndarray
    z: float
    lam: np.ndarray
    low: np.ndarray
    upp: np.ndarray
    newton: int
    converged: bool


# --- asymptotes and move bounds ---------------------------------------------------------------------


def update_asymptotes(it, x, xold1, xold2, low, upp, xmin, xmax, *, asyinit=ASYINIT, asydecr=ASYDECR,
                      asyincr=ASYINCR, asymin=ASYMIN, asymax=ASYMAX):
    """The asymptotes l and u of iteration `it` (1 for the first call). [S07] (3.11)-(3.14).

    `low` and `upp` are the asymptotes of the iteration before. They have no effect for it <= 2.
    """
    span = xmax - xmin
    if it <= 2:
        # (3.11)
        return x - asyinit * span, x + asyinit * span
    # (3.13): the sign of (x(k) - x(k-1)) * (x(k-1) - x(k-2)) tells if x oscillates (< 0) or moves on (> 0)
    sign = (x - xold1) * (xold1 - xold2)
    gamma = np.where(sign < 0.0, asydecr, np.where(sign > 0.0, asyincr, 1.0))
    # (3.12)
    new_low = x - gamma * (xold1 - low)
    new_upp = x + gamma * (upp - xold1)
    # (3.14): a violated limit is replaced by the limit itself
    new_low = np.minimum(new_low, x - asymin * span)
    new_low = np.maximum(new_low, x - asymax * span)
    new_upp = np.maximum(new_upp, x + asymin * span)
    new_upp = np.minimum(new_upp, x + asymax * span)
    return new_low, new_upp


def move_bounds(x, low, upp, xmin, xmax, move, *, albefa=ALBEFA):
    """The bounds alpha and beta of one step. [S07] (3.6), (3.7), with `move` in place of the 0.5 there."""
    span = xmax - xmin
    alpha = np.maximum(np.maximum(xmin, low + albefa * (x - low)), x - move * span)
    beta = np.minimum(np.minimum(xmax, upp - albefa * (upp - x)), x + move * span)
    return alpha, beta


# --- the convex approximation ----------------------------------------------------------------------------


def _pq(df, ux, xl, span, raa0):
    """p and q of one or more rows from the gradient `df` (shape (n,) or (m, n)). [S07] (3.3), (3.4)."""
    pos = np.maximum(df, 0.0)
    neg = np.maximum(-df, 0.0)
    floor = raa0 / span
    p = (ux * ux) * (PQ_NEAR * pos + PQ_FAR * neg + floor)
    q = (xl * xl) * (PQ_FAR * pos + PQ_NEAR * neg + floor)
    return p, q


def approximation(x, low, upp, xmin, xmax, f, df0, dg, *, raa0=RAA0):
    """p, q of the objective and the rows, and b. `f` holds the row values f_i(x), i = 1..m.

    The row i of the subproblem is sum_j (P_ij/(u_j-x_j) + Q_ij/(x_j-l_j)) - b_i <= ..., with
    b_i = -r_i and r_i from (3.5). Returns (p0, q0, P, Q, b).
    """
    ux = upp - x
    xl = x - low
    span = xmax - xmin
    p0, q0 = _pq(df0, ux, xl, span, raa0)
    P, Q = _pq(dg, ux, xl, span, raa0)
    # (3.5): r_i = f_i - sum_j (p_ij/(u_j-x_j) + q_ij/(x_j-l_j)); the subproblem has b_i = -r_i
    b = P @ (1.0 / ux) + Q @ (1.0 / xl) - f
    return p0, q0, P, Q, b


# --- the primal-dual interior-point method for the subproblem ([S07] section 5) -------------------


def _rows_sum(base, M, w):
    """base + sum_i w_i * M[i], element by element (a loop over the rows, see the module text)."""
    out = base.copy()
    tmp = np.empty_like(base)
    for i in range(M.shape[0]):
        np.multiply(M[i], w[i], out=tmp)
        out += tmp
    return out


class _Point:
    """The quantities that depend on x and lam only: used by the residual and the Newton step."""

    __slots__ = ("ux", "xl", "iux", "ixl", "pl", "ql", "dpsi", "g")

    def __init__(self, sp: Subproblem, x, lam):
        self.ux = sp.upp - x
        self.xl = x - sp.low
        self.iux = 1.0 / self.ux
        self.ixl = 1.0 / self.xl
        self.pl = _rows_sum(sp.p0, sp.P, lam)       # (5.5) p_j(lambda)
        self.ql = _rows_sum(sp.q0, sp.Q, lam)       # (5.5) q_j(lambda)
        self.dpsi = self.pl * self.iux * self.iux - self.ql * self.ixl * self.ixl    # (5.8)
        self.g = sp.P @ self.iux + sp.Q @ self.ixl  # (5.2) g_i(x), i = 1..m


def _residual(sp: Subproblem, w, pt: _Point, eps):
    """The left hand sides of the relaxed KKT conditions (5.9a)-(5.9i), and their two norms."""
    x, y, z, lam, xi, eta, mu, zeta, s = w
    parts = (
        pt.dpsi - xi + eta,                      # (5.9a)
        sp.c + sp.d * y - lam - mu,              # (5.9b)
        np.array([sp.a0 - zeta - lam @ sp.a]),   # (5.9c)
        pt.g - sp.a * z - y + s - sp.b,          # (5.9d)
        xi * (x - sp.alpha) - eps,               # (5.9e)
        eta * (sp.beta - x) - eps,               # (5.9f)
        mu * y - eps,                            # (5.9g)
        np.array([zeta * z - eps]),              # (5.9h)
        lam * s - eps,                           # (5.9i)
    )
    total = 0.0
    biggest = 0.0
    for v in parts:
        total += float(v @ v)
        biggest = max(biggest, float(v.max()), -float(v.min()))
    return np.sqrt(total), biggest


def _direction(sp: Subproblem, w, pt: _Point, eps, system=None):
    """The Newton direction (dx, dy, dz, dlam, dxi, deta, dmu, dzeta, ds). [S07] 5.3.

    `system` picks the reduced system: "dual" is (5.20), of the size m + 1; "primal" is (5.22), of the size
    n + 1; None takes the smaller one, as the note advises.
    """
    x, y, z, lam, xi, eta, mu, zeta, s = w
    n, m = x.size, lam.size
    ixa = 1.0 / (x - sp.alpha)        # <x-alpha>^-1
    ibx = 1.0 / (sp.beta - x)         # <beta-x>^-1
    iux, ixl = pt.iux, pt.ixl
    # (5.11) Psi, (5.12) G
    psi = 2.0 * pt.pl * iux * iux * iux + 2.0 * pt.ql * ixl * ixl * ixl
    G = sp.P * (iux * iux) - sp.Q * (ixl * ixl)
    # (5.15)
    Dx = psi + ixa * xi + ibx * eta
    Dy = sp.d + mu / y
    Dlam = s / lam
    tx = pt.dpsi - eps * ixa + eps * ibx
    ty = sp.c + sp.d * y - lam - eps / y
    tz = sp.a0 - lam @ sp.a - eps / z
    tl = pt.g - sp.a * z - y - sp.b + eps / lam
    # (5.18)
    Dly = Dlam + 1.0 / Dy
    tly = tl + ty / Dy
    iDx = 1.0 / Dx
    if system is None:
        system = "dual" if m < n else "primal"
    if system == "dual":
        # (5.20): a system of the size m + 1
        GD = G * iDx
        K = np.empty((m + 1, m + 1))
        K[:m, :m] = GD @ G.T
        K[np.arange(m), np.arange(m)] += Dly
        K[:m, m] = sp.a
        K[m, :m] = sp.a
        K[m, m] = -zeta / z
        rhs = np.empty(m + 1)
        rhs[:m] = tly - GD @ tx
        rhs[m] = tz
        sol = np.linalg.solve(K, rhs)
        dlam, dz = sol[:m], float(sol[m])
        # (5.19)
        dx = -(_rows_sum(tx, G, dlam)) * iDx
    else:
        # (5.22): a system of the size n + 1
        iDly = 1.0 / Dly
        K = np.empty((n + 1, n + 1))
        K[:n, :n] = (G.T * iDly) @ G
        K[np.arange(n), np.arange(n)] += Dx
        K[:n, n] = -(G.T * iDly) @ sp.a
        K[n, :n] = K[:n, n]
        K[n, n] = zeta / z + float(sp.a @ (iDly * sp.a))
        rhs = np.empty(n + 1)
        rhs[:n] = -tx - G.T @ (iDly * tly)
        rhs[n] = -tz + float(sp.a @ (iDly * tly))
        sol = np.linalg.solve(K, rhs)
        dx, dz = sol[:n], float(sol[n])
        # (5.21)
        dlam = iDly * (G @ dx - sp.a * dz + tly)
    dy = (dlam - ty) / Dy                   # (5.16)
    # (5.13)
    dxi = -(xi * ixa) * dx - xi + eps * ixa
    deta = (eta * ibx) * dx - eta + eps * ibx
    dmu = -(mu / y) * dy - mu + eps / y
    dzeta = -(zeta / z) * dz - zeta + eps / z
    ds = -(s / lam) * dlam - s + eps / lam
    return dx, dy, dz, dlam, dxi, deta, dmu, dzeta, ds


def _max_step(sp: Subproblem, w, dw):
    """The largest t <= 1 that keeps every inequality (5.9j)-(5.9n) with the margin EDGE. [S07] 5.4."""
    x, y, z, lam, xi, eta, mu, zeta, s = w
    dx, dy, dz, dlam, dxi, deta, dmu, dzeta, ds = dw
    t = 1.0
    keep = 1.0 - EDGE
    # a value v with step dv stays above EDGE * v when t * dv >= -keep * v, so t <= keep * v / (-dv) if dv < 0
    pairs = ((x - sp.alpha, dx), (sp.beta - x, -dx), (y, dy), (np.array([z]), np.array([dz])), (lam, dlam),
             (xi, dxi), (eta, deta), (mu, dmu), (np.array([zeta]), np.array([dzeta])), (s, ds))
    for v, dv in pairs:
        down = dv < 0.0
        if down.any():
            t = min(t, float((keep * v[down] / -dv[down]).min()))
    return t


def solve_subproblem(sp: Subproblem, *, eps_end=EPS_END, max_newton=MAX_NEWTON, system=None) -> Solution:
    """The primal-dual interior-point method of [S07] 5.5 for the subproblem (5.1). `system`: see `_direction`."""
    n, m = sp.alpha.size, sp.b.size
    # 5.5: the simple starting point
    x = 0.5 * (sp.alpha + sp.beta)
    y = np.ones(m)
    z = 1.0
    lam = np.ones(m)
    zeta = 1.0
    s = np.ones(m)
    xi = np.maximum(1.0, 1.0 / (x - sp.alpha))
    eta = np.maximum(1.0, 1.0 / (sp.beta - x))
    mu = np.maximum(1.0, 0.5 * sp.c)
    # 5.5: epsilon runs through EPS_START, EPS_START * EPS_FACTOR, ... down to eps_end. The stages are counted, so that a
    # round-off in the repeated product (0.1**7 is not exactly 1e-7) cannot add or drop a stage.
    last_stage = int(round(np.log(eps_end / EPS_START) / np.log(EPS_FACTOR)))
    stage = 0
    eps = EPS_START
    w = (x, y, z, lam, xi, eta, mu, zeta, s)
    pt = _Point(sp, x, lam)
    norm2, _ = _residual(sp, w, pt, eps)
    newton = 0
    converged = False
    while newton < max_newton:
        newton += 1
        dw = _direction(sp, w, pt, eps, system)               # step 1
        t = _max_step(sp, w, dw)                      # 5.4: the largest t
        tau = t                                       # step 2: t, t/2, t/4, ... until the residual falls
        for _ in range(MAX_HALVINGS + 1):
            new = tuple(v + tau * dv for v, dv in zip(w, dw))
            new_pt = _Point(sp, new[0], new[3])
            new_norm2, new_inf = _residual(sp, new, new_pt, eps)
            if new_norm2 < norm2:
                break
            tau *= 0.5
        else:
            break                                     # no step makes the residual smaller: stop with w
        w, pt = new, new_pt                           # step 3
        norm2 = new_norm2
        if new_inf < EPS_TEST * eps:                  # step 4
            if stage >= last_stage:
                converged = True
                break
            stage += 1
            eps = EPS_START * EPS_FACTOR ** stage
            norm2, _ = _residual(sp, w, pt, eps)
    x, y, z, lam, xi, eta, mu, zeta, s = w
    return Solution(x, y, z, lam, xi, eta, mu, zeta, s, newton, converged)


# --- one MMA iteration ----------------------------------------------------------------------------------


def mma_update(it, x, xmin, xmax, xold1, xold2, f0, df0, f, dg, low, upp, a0, a, c, d, move, *,
               asyinit=ASYINIT, asydecr=ASYDECR, asyincr=ASYINCR, asymin=ASYMIN, asymax=ASYMAX,
               albefa=ALBEFA, raa0=RAA0) -> MmaResult:
    """One outer iteration of MMA: the new design for the rows `f_i(x) <= 0`.

    it          : the number of this call, 1 for the first. The first two calls use the start asymptotes.
    x, xold1, xold2 : (n,) the design now and the designs of the two calls before
    xmin, xmax  : bounds of x: a number or an (n,) array
    f0, df0     : the objective value (not used by the method: the constant of (3.2) drops out) and (n,) gradient
    f, dg       : (m,) the row values and the (m, n) gradients
    low, upp    : (n,) the asymptotes of the call before (no effect for it <= 2)
    a0, a, c, d : the coefficients of (1.1): a0 (number), and (m,) arrays a, c, d
    move        : the move limit, a part of (xmax - xmin)
    """
    del f0   # the value of f0 is a constant of the subproblem: it does not change the minimizer
    x = np.asarray(x, dtype=float)
    new_low, new_upp = update_asymptotes(it, x, xold1, xold2, low, upp, xmin, xmax, asyinit=asyinit,
                                         asydecr=asydecr, asyincr=asyincr, asymin=asymin, asymax=asymax)
    alpha, beta = move_bounds(x, new_low, new_upp, xmin, xmax, move, albefa=albefa)
    p0, q0, P, Q, b = approximation(x, new_low, new_upp, xmin, xmax, f, df0, dg, raa0=raa0)
    sol = solve_subproblem(Subproblem(alpha, beta, new_low, new_upp, p0, q0, P, Q, b, float(a0), a, c, d))
    return MmaResult(sol.x, sol.y, sol.z, sol.lam, new_low, new_upp, sol.newton, sol.converged)
