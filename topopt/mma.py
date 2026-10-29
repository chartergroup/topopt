"""MMA update with m constraint rows. All state lives in one `MmaState` object.

`Mma.step` makes one MMA iteration with `topopt.mma_sub.mma_update` (own code, written from the
2007 note of Svanberg; see that module). The settings follow the earlier pipeline of the same author
(`run_unattended_perim.py` `mma_step`, `run_final_pass.py`):
a0 = 1, a = 0, c = 1000 and d = 0 for each row, the other MMA parameters at the
default values of the 2007 note (`asyinit` 0.5, `asydecr` 0.7, `asyincr` 1.2, `asymin` 0.01, `asymax` 10,
`albefa` 0.1, `raa0` 1e-5). `c` can differ for each row (`step(..., c=...)`): the default is 1000.
The only addition is the vector `a` for the
bound (min-max) form.

The subproblem is

    minimize   f0(x) + a0*z + sum_i (c_i*y_i + 0.5*d_i*y_i^2)
    subject to f_i(x) - a_i*z - y_i <= 0,   xmin <= x <= xmax,   z >= 0, y >= 0

Give each row `g_i <= 0` with values of order 1 (the price c = 1000 is for
rows of that size). A row that cannot be met stays positive: MMA treats it as
a soft constraint (y_i > 0). It does not raise an error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .mma_sub import mma_update

log = logging.getLogger("topopt.mma")

# MMA settings. Same as the pipeline calls (see the plan document).
A0 = 1.0          # cost of z in the objective
C_PENALTY = 1000.0   # price of the violation y_i of one row
D_QUAD = 0.0      # quadratic price of y_i: 0 (the pipeline passes np.zeros((m, 1)))


@dataclass
class MmaState:
    """The whole state of the optimizer between two steps.

    A checkpoint needs exactly these fields. `it` is the number of finished
    steps. `mma_update` gets `it + 1`, so the first call has number 1 (the asymptotes
    reset at numbers 1 and 2, [S07] (3.11)). `xold1` and `xold2` are the designs of
    the two last steps. `low` and `upp` are the asymptotes. `f0_ref` is the
    value that scales the objective (the pipeline calls it c0 or mass_base). It
    stays None until the first evaluation sets it. The loop sets it to None again at the
    first pass of each beta stage (`settings.objective_scale: stage`, the default;
    `optimize.objective_scale_resets`), and the next evaluation takes the objective of that
    pass. `x` is the design after the last step (None before the first step).

    A2 added two fields, so that a resume can restore them (KB pitfall
    resume-differs-from-straight-run):

    - `rows`: state of the rows that keep a value between passes, as `{row name: dict}`
      (JSON numbers; a value of the first level can also be a numpy array: the checkpoint stores it
      in the `.npz`, `checkpoint.pack_rows`). The stress row keeps its KS scale `c` and the beta of
      the stage in which `c` was set; the buckling row also keeps the start vector `v0` of its
      eigen solve (`Formulation.respond` copies them in and out).
    - `move`: the move limit with which step acceptance starts the next pass (None: step
      acceptance is not used, or has not run yet). It halves after a rejected step and grows after
      an accepted step (`topopt.optimize.StepAcceptance`). `Mma.step` does not read it.
    """

    n: int
    it: int = 0
    x: np.ndarray | None = None
    xold1: np.ndarray | None = None
    xold2: np.ndarray | None = None
    low: np.ndarray = field(default=None)
    upp: np.ndarray = field(default=None)
    f0_ref: float | None = None
    rows: dict = field(default_factory=dict)
    move: float | None = None

    def __post_init__(self):
        if self.low is None:
            self.low = np.zeros((self.n, 1))
        if self.upp is None:
            self.upp = np.ones((self.n, 1))


class Mma:
    """MMA for n variables in [xmin, xmax] and m constraint rows `g_i(x) <= 0`.

    Call `step(x, f0, df0, g, dg)` once per iteration. It returns the new design.
    `state` holds all that must be saved to resume a run. `lam` and `z` keep the
    multipliers and the bound variable of the last step (for reports only). `newton` and
    `sub_converged` keep the number of Newton iterations of the last subproblem and whether
    it ended at the end value of the barrier parameter (for tests and reports only).
    """

    def __init__(self, n: int, m: int = 1, move: float = 0.2, xmin: float = 0.0,
                 xmax: float = 1.0):
        if n < 1 or m < 1:
            raise ValueError(f"MMA needs n >= 1 variables and m >= 1 rows, got n={n}, m={m}")
        if not 0.0 < move <= 1.0:
            raise ValueError(f"move limit must be in (0, 1], got {move!r}")
        self.n, self.m, self.move, self.xmin, self.xmax = n, m, float(move), xmin, xmax
        self.state = MmaState(n)
        self.lam = None
        self.z = None
        self.newton = None
        self.sub_converged = None

    def step(self, x, f0, df0, g, dg, a0: float = A0, a=None, c=None, move: float | None = None):
        """One MMA update.

        x    : (n,) design now
        f0   : objective value (scaled), df0 : (n,) its gradient with respect to x
        g    : (m,) row values, feasible when <= 0
        dg   : (m, n) row gradients with respect to x
        a0, a: coefficients of z. Default a0 = 1 and a = 0 (z stays 0). The bound
               form uses a_i = 1 for the rows that bound the objective.
        c    : price of a violation, one value for each row (m,) or one number.
               Default 1000 for every row. A row with a small price gives way first when
               the rows cannot all be met.
        move : move limit for this step (default: `self.move`). Step acceptance passes a smaller
               value after a rejected step.
        """
        n, m, st = self.n, self.m, self.state
        x = np.asarray(x, dtype=float).reshape(-1)
        g = np.asarray(g, dtype=float).reshape(-1)
        dg = np.asarray(dg, dtype=float).reshape(-1, n) if np.size(dg) else np.empty((0, n))
        df0 = np.asarray(df0, dtype=float).reshape(-1)
        if x.size != n or df0.size != n:
            raise ValueError(f"x and df0 must have {n} values, got {x.size} and {df0.size}")
        if g.size != m or dg.shape != (m, n):
            raise ValueError(f"MMA has {m} rows: g needs {m} values and dg shape ({m}, {n}), "
                             f"got {g.size} values and shape {dg.shape}")
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(dg)) and np.isfinite(f0)
                and np.all(np.isfinite(df0))):
            raise ValueError("MMA got a value or gradient that is not finite (nan or inf)")
        a_vec = np.zeros((m, 1)) if a is None else np.asarray(a, dtype=float).reshape(m, 1)
        c_vec = np.full((m, 1), C_PENALTY) if c is None else np.broadcast_to(
            np.asarray(c, dtype=float).reshape(-1, 1), (m, 1)).copy()
        if not np.all(c_vec > 0):
            raise ValueError(f"the price c of a row must be > 0, got {c_vec.reshape(-1).tolist()}")
        if st.xold1 is None:
            st.xold1 = st.xold2 = x.copy()
        move = self.move if move is None else float(move)
        if not 0.0 < move <= 1.0:
            raise ValueError(f"move limit must be in (0, 1], got {move!r}")
        res = mma_update(
            st.it + 1, x, float(self.xmin), float(self.xmax), st.xold1, st.xold2, f0, df0, g, dg,
            st.low.reshape(n), st.upp.reshape(n), a0, a_vec.reshape(m), c_vec.reshape(m),
            np.full(m, D_QUAD), move)
        low, upp = res.low.reshape(n, 1), res.upp.reshape(n, 1)
        st.low, st.upp = low, upp
        st.xold2, st.xold1 = st.xold1, x.copy()
        st.it += 1
        self.lam, self.z = res.lam, res.z
        self.newton, self.sub_converged = res.newton, res.converged
        if not res.converged:
            log.warning("MMA step %d: the subproblem stopped after %d Newton iterations, before the barrier parameter "
                        "reached its end value; the step uses the last point", st.it, res.newton)
        st.x = np.clip(res.x, self.xmin, self.xmax)
        return st.x.copy()
