"""The optimization problem for MMA: one objective and m constraints.

This is the API to add a constraint. A constraint is a function of a
`DesignPoint` that returns a `Response`: a value and the gradient with respect
to the physical density of the design voxels. The library does the chain rule
through the Heaviside projection and the density filter, scales the row and
hands it to `topopt.mma.Mma`. Example: the mean density of a set of voxels
(`voxels`, an index array) must stay at or below 0.5:

    def box_mean(point):                        # mean density in a set of voxels
        grad = np.zeros(point.n_design)
        grad[voxels] = 1.0 / len(voxels)
        return Response(float(point.rho[voxels].mean()), grad)

    constraints = [volume_constraint(model, 0.3), Constraint("box", box_mean, limit=0.5)]
    form = Formulation(weighted_compliance(), constraints)

`examples/custom_constraint.py` runs this end to end with `optimize.minimize`.
CONTRACTS.md section 13 gives the YAML keys and the history fields.

A row that uses `U` (an adjoint, `DesignPoint.stiffness_derivative`) returns `wrt="rho_stiff"` and reads
`point.rho_stiff`: `stiffness_derivative` is the slope with respect to `rho_stiff`, which equals `rho` in a
default run and is the eroded density in robust mode (settings.robust, CONTRACTS.md section 22b). A row that
uses only the density (a volume, a box mean) keeps `wrt="rho"` and reads `point.rho`.

Sign rule: a row is met when `value <= limit`. To get `value >= limit`, return
the negative value and the negative of the gradient, with the negative limit.

Prices. MMA treats every row as a soft constraint: a row that cannot be met is
violated, and the price `Constraint.price` (default 1000) says how hard MMA
works to avoid that. When rows cannot all be met, the row with the lower price
gives way first. The volume row has price 1000. The built-in compliance limits
have price 100 and the form "log" (MMA gets ln(value/limit)), so a limit that
cannot be met does not pull the volume off its target: the volume is the
budget. A custom row stays linear with price 1000 unless you set `form` and
`price`. The form "log" is for responses that stay above 0.

Rows that are linear copies of each other (the same function twice) are legal.
The dual system of MMA is then nearly singular: measured on the test cantilever,
condition number 1.6e14 against 6e7 for one row. The MMA code raised no warning there. The 40-iteration run
differed from the single-row run by 1.8e-7 in compliance (relative) and 3.6e-5 in the largest density (measured 2026-10-26 with
`topopt.mma_sub`; the MMA code before REL-7 gave 3.8e-3 for the density). No guard is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .mma import C_PENALTY, MmaState

WRT = ("rho", "rho_stiff", "rho_tilde", "x")   # what a gradient is taken with respect to
FORMS = ("sum", "max")
ROW_FORMS = ("linear", "log")
NOT_MET = 5e-3   # g above this value counts as "not met" in the run log (the history keeps all)
HELD = "skipped"   # the `status` of a row entry at a pass that the row did not evaluate: it holds the last value, a result


def no_result(entry: dict) -> bool:
    """True if the history entry of a row has no result at its pass. A fact the verdict and the end lines read.

    A row that reports `evaluated` (the buckling row) has no result when it did not evaluate and its `status` is not `skipped`
    (it failed), or when it holds no value at all (`last_eval` is None). A skipped pass with a known value is a result: the value
    is held. A row with no `evaluated` key always has a result.
    """
    if entry.get("evaluated") is not False:
        return False
    return entry.get("status") != HELD or entry.get("last_eval") is None


def no_result_text(entry: dict) -> str:
    """The words for a row with no result: its status and the pass of the value that it holds."""
    n = entry.get("last_eval")
    held = f"the value held is from pass {n}" if n is not None else "no pass gave a result, the value held is the start value"
    return f"no result at the last pass (status {entry.get('status')}; {held})"


# --- values the user code returns and receives ------------------------------


@dataclass
class Response:
    """A value and its gradient.

    `grad` has one entry for each design voxel (`n_design` values). `wrt` says
    with respect to what it is:
      "rho"       the physical density (after filter and projection). The default.
      "rho_stiff" the density that the stiffness sees. It is the same as "rho", except in robust
                  mode (settings.robust, CONTRACTS.md section 22b): there it is the eroded density.
                  The compliance rows use it. Use it for a response that comes from U.
      "rho_tilde" the filtered density (before the projection).
      "x"         the design variable. The library does no chain rule then.

    The true gradient is `grad / divisor`. The library divides after the chain rule.
    It is for a gradient with one common factor, such as a mean over n voxels
    (`grad` all ones, `divisor` n). Most functions leave `divisor` at 1.

    `info` (optional) is a dict of JSON numbers. A constraint adds it to its entry in the history
    (`constraints[].<key>`). The stress row uses it for the numbers in MPa.
    """

    value: float
    grad: np.ndarray
    wrt: str = "rho"
    divisor: float = 1.0
    info: dict | None = None     # extra numbers for the history entry of a constraint (A2: stress rows)


def _as_response(result, name, n_design) -> Response:
    """Accept a Response or a pair `(value, grad)`. Check it. One clear message."""
    if isinstance(result, tuple) and len(result) == 2:
        result = Response(result[0], result[1])
    if not isinstance(result, Response):
        raise TypeError(f"'{name}' must return a Response(value, grad) or a pair (value, grad), "
                        f"got {type(result).__name__}")
    grad = np.asarray(result.grad, dtype=float).reshape(-1)
    if result.wrt not in WRT:
        raise ValueError(f"'{name}': wrt must be one of {', '.join(WRT)}, got {result.wrt!r}")
    if grad.size != n_design:
        raise ValueError(f"'{name}': the gradient needs {n_design} values (one for each design "
                         f"voxel), got {grad.size}")
    value = float(result.value)
    if not (np.isfinite(value) and np.all(np.isfinite(grad))):
        raise ValueError(f"'{name}': the value or the gradient is not finite (nan or inf)")
    if not (np.isfinite(result.divisor) and result.divisor > 0):
        raise ValueError(f"'{name}': divisor must be > 0, got {result.divisor!r}")
    return Response(value, grad, result.wrt, float(result.divisor), result.info)


class DesignPoint:
    """What is known at one design: the input of every objective and constraint function.

    model        the `optimize.Model`
    x, beta      design variables and Heaviside beta
    rho_tilde    filtered density of the design voxels
    rho          physical density of the design voxels (`n_design` values). The nominal design:
                 the volume, the snapshots and the final mesh use it
    rho_stiff    the density of the stiffness (`n_design` values). The same array as `rho`, except in
                 robust mode: there it is the eroded density, and `U` is the solution for it
    U            displacements of the free dofs, shape (n_free, n_cases)
    case_names, weights
    compliance_cases   c_i = f_i . u_i for each load case
    compliance         weighted sum of the cases
    iteration, last    the pass number and whether it is the last pass of the run (`run` and `minimize` set them).
                 A point built by hand has `iteration` None and `last` False. A row that skips passes (A6b) needs them
    """

    def __init__(self, model, x, beta, rho_tilde, rho, U, rho_stiff=None, iteration=None, last=False):
        self.model, self.x, self.beta = model, x, beta
        self.iteration, self.last = iteration, bool(last)
        self._K = None
        self.rho_tilde, self.rho, self.U = rho_tilde, rho, U
        self.rho_stiff = model.stiffness_density(rho_tilde, rho, beta) if rho_stiff is None else rho_stiff
        self.n_design = model.n_design
        self.case_names = list(model.problem.case_names)
        self.weights = model.weights
        self.compliance_cases = np.einsum("ij,ij->j", model.f_red, U)
        self.compliance = float(self.weights @ self.compliance_cases)
        self._weighted_slope = None
        self._case_slopes = None

    @property
    def n_cases(self) -> int:
        return len(self.case_names)

    def case_index(self, case) -> int:
        """Index of a load case from its name or its index."""
        if isinstance(case, str):
            if case not in self.case_names:
                raise ValueError(f"no load case '{case}' (known: {', '.join(self.case_names)})")
            return self.case_names.index(case)
        i = int(case)
        if not 0 <= i < self.n_cases:
            raise ValueError(f"load case index {i} is out of range 0..{self.n_cases - 1}")
        return i

    def weighted_compliance_slope(self) -> np.ndarray:
        """d(sum_i w_i c_i)/d(rho_stiff) for the design voxels."""
        if self._weighted_slope is None:
            self._weighted_slope = self.model.compliance(self.U, self.rho_stiff)[1]
        return self._weighted_slope

    def compliance_slopes(self) -> np.ndarray:
        """d(c_i)/d(rho_stiff) for each case: shape (n_design, n_cases). No weights."""
        if self._case_slopes is None:
            m = self.model
            dE = m.dE_drho(self.rho_stiff)
            self._case_slopes = np.column_stack(
                [-dE * m.element_energy(self.U[:, i], self.U[:, i]) for i in range(self.n_cases)])
        return self._case_slopes

    def stiffness_matrix(self):
        """The reduced stiffness matrix of this design: `model.stiffness(rho_stiff)`, assembled once for the point.

        The run holds a factor of it (`model.solver`), not the matrix. A response that needs the matrix (the buckling row) asks
        for it here, so a run without such a row assembles nothing more. Do not change the matrix.
        """
        if self._K is None:
            self._K = self.model.stiffness(self.rho_stiff)
        return self._K

    def solve_adjoint(self, rhs) -> np.ndarray:
        """Solve K lam = rhs with the stiffness of this design.

        `rhs` is a vector (n_free,) or a matrix (n_free, k) in the numbering of the free
        dofs (`model.to_free(full_vector)` maps a full-grid vector). Use it for the adjoint
        load of a response that is not a compliance (stress, a point displacement, ...).
        It does not change the warm start of the main solve.
        """
        return self.model.solver.solve(rhs, warm=False)

    def stiffness_derivative(self, lam, u) -> np.ndarray:
        """d(lam^T K(rho) u)/d(rho_stiff) of the design voxels, with lam and u held fixed.

        `lam` and `u` are vectors of the free dofs. For a response r(u, rho) with the
        adjoint K lam = dr/du, the full gradient is dr/drho = (explicit part)
        - stiffness_derivative(lam, u). A compliance has lam = u = U[:, i]. The slope is with respect to
        `rho_stiff`: return such a response with `wrt="rho_stiff"`. It is the same as "rho" except in
        robust mode, where it makes the chain rule use the eroded projection.
        """
        m = self.model
        return m.dE_drho(self.rho_stiff) * m.element_energy(lam, u)


# --- objective and constraints ----------------------------------------------


@dataclass
class Constraint:
    """The row `value(rho) <= limit`, in the dimensionless form MMA needs.

    evaluate  function `DesignPoint -> Response` (or a pair `(value, grad)`)
    limit     the bound on the value (default 0)
    scale     g = (value - limit) / scale. Default |limit|, or 1 if the limit is 0.
              Choose it so that g is of order 1 near the limit.
    form      "linear" (default): MMA gets the row g. "log": MMA gets ln(value / limit)
              (value and limit must be above 0). The report keeps the linear g.
    price     price of a violation in MMA (default 1000, see the module text)
    hint      text for the run log when the row is not met at the end
    cause     the sentence before the hint when only rows with a hint are not met. Default: the
              compliance-limit text of A1 ("... this limit does not fit at this volume")
    xmin      the lowest design variable the row wants (a `Formulation` without its own `xmin`
              takes the largest value of its rows). The stress row asks for 0.02.
    stage_aware   True: the run log and the history record when a beta stage ends with this
              row not met (a fact, the schedule does not change)
    detail    function `entry -> text`: more words for the log line of a row that is not met
    facts     function `entry -> list of text`: more lines at the end of the run (any state of the row)
    stage_start   function `entry -> list of text`: facts at the first pass of each beta stage (the
              stress row says when its KS scale `c` is far below 1)
    measure   function `DesignPoint -> float`: the value of the row at a design, with no gradient and
              no change of the state of the row. Step acceptance (`topopt.optimize.StepAcceptance`) uses
              it for the rows with `stage_aware`: it solves at the trial design and reads this value.
              A stage-aware row with no `measure` is not checked by step acceptance.
    rival     `(clause, option)` or None: a row with a higher price than a compliance limit can take
              material from it. Under the mass objective, when this row is met and a compliance limit
              is not, the summary line of the run log adds the clause as a possible cause and the
              option to its options. A row with no such effect leaves it None.

A row function can keep a value between passes (the stress row keeps its KS scale `c`). It then has
`get_state() -> dict` and `set_state(dict or None)` (None: the start state). `Formulation.respond` copies
that dict to and from `MmaState.rows[<row name>]` at every pass, so the state is part of the optimizer
state: a resume or a restored `MmaState` sets it again. A value of the dict is JSON data or a numpy array
(the checkpoint stores an array in its `.npz`: `checkpoint.pack_rows`); a row never changes an array in place. A state with no entry for the row leaves the row
object as it is (a script that evaluates one row at several designs with a new `MmaState` each time keeps
the value of the row).

    Row: g = value/scale - limit/scale, a row is met when g <= 0. The history
    keeps `value`, `limit`, `g` and `violation = max(g, 0)` of every iteration.
    """

    name: str
    evaluate: Callable[[DesignPoint], Response]
    limit: float = 0.0
    scale: float | None = None
    form: str = "linear"
    price: float = C_PENALTY
    hint: str | None = None
    cause: str | None = None
    xmin: float | None = None
    stage_aware: bool = False
    detail: Callable[[dict], str] | None = None
    facts: Callable[[dict], list] | None = None
    stage_start: Callable[[dict], list] | None = None
    measure: Callable[[DesignPoint], float] | None = None
    rival: tuple[str, str] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a constraint needs a name (a non-empty string)")
        if not np.isfinite(self.limit):
            raise ValueError(f"constraint '{self.name}': the limit must be finite, got {self.limit!r}")
        if self.scale is None:
            self.scale = abs(self.limit) if self.limit != 0 else 1.0
        if not (np.isfinite(self.scale) and self.scale > 0):
            raise ValueError(f"constraint '{self.name}': the scale must be > 0, got {self.scale!r}")
        if self.form not in ROW_FORMS:
            raise ValueError(f"constraint '{self.name}': form must be linear or log, got {self.form!r}")
        if self.form == "log" and not self.limit > 0:
            raise ValueError(f"constraint '{self.name}': the log form needs a limit above 0, "
                             f"got {self.limit!r}")
        if not (np.isfinite(self.price) and self.price > 0):
            raise ValueError(f"constraint '{self.name}': the price must be > 0, got {self.price!r}")


@dataclass
class Objective:
    """What MMA minimizes.

    terms   function `DesignPoint -> list of Response` (one Response is also allowed)
    form    "sum": minimize the sum of the terms.
            "max": minimize the largest term (bound form: MMA gets one row for each
                   term and the variable z, which bounds all terms).
    n_terms number of terms the function returns (the library needs it before the first call)

    The objective is divided by a scale, `MmaState.f0_ref`: its value at the first pass of the run, and again its value at the
    first pass of each beta stage (`settings.objective_scale`: `stage`, the default; `first` keeps the scale of pass 0, see
    CONTRACTS.md section 13). The history keeps the physical value, not the scaled one.
    The "max" form needs terms above 0: MMA bounds them with z, and z >= 0. A term at or
    below 0 stops the run with a message. With one term, "max" and "sum" are the same
    problem: use "sum" (`worst_case_compliance(1)` does).
    """

    name: str
    terms: Callable[[DesignPoint], list]
    form: str = "sum"
    n_terms: int = 1

    def __post_init__(self):
        if self.form not in FORMS:
            raise ValueError(f"objective form must be 'sum' or 'max', got {self.form!r}")
        if self.n_terms < 1:
            raise ValueError("an objective needs at least one term")


def weighted_compliance() -> Objective:
    """sum_i w_i c_i: the objective of the default run."""
    def terms(p: DesignPoint):
        return [Response(p.compliance, p.weighted_compliance_slope(), wrt="rho_stiff")]
    return Objective("weighted_compliance", terms, "sum", 1)


def worst_case_compliance(n_cases: int) -> Objective:
    """max_i w_i c_i, in the bound form. The weights scale the cases before the max.

    With one case this is the weighted compliance (the "sum" form): the problem is the same and
    the plain form takes the same MMA path as a run without the key.
    """
    if n_cases == 1:
        return weighted_compliance()
    def terms(p: DesignPoint):
        slopes = p.compliance_slopes()
        return [Response(float(p.weights[i] * p.compliance_cases[i]), p.weights[i] * slopes[:, i],
                         wrt="rho_stiff") for i in range(p.n_cases)]
    return Objective("worst_case_compliance", terms, "max", n_cases)


def mass_objective() -> Objective:
    """The mass of the part in g: design voxels at their density plus the forced-solid voxels.

    The density of the material and the voxel size come from the model of the design point
    (`material.density`, `problem.h`). The objective is divided by its scale (`MmaState.f0_ref`, see `Objective`). Use it with
    rows that hold material: stress rows (`topopt.ks_stress.stress_ks_constraint`) or compliance
    limits. With no such row, nothing stops the mass from going to the lowest density.
    """
    def terms(p: DesignPoint):
        m = p.model
        voxel_g = m.problem.h ** 3 / 1000.0 * m.problem.config.material.density
        n_solid = m.n_el - m.n_design
        return [Response(float((p.rho.sum() + n_solid) * voxel_g), np.full(p.n_design, voxel_g))]
    return Objective("mass", terms, "sum", 1)


def volume_constraint(n_design: int, fraction: float) -> Constraint:
    """mean(rho) of the design voxels <= `fraction`. g = mean/fraction - 1."""
    def evaluate(p: DesignPoint):
        return Response(float(p.rho.mean()), np.ones(n_design), divisor=float(n_design))
    return Constraint("volume", evaluate, limit=fraction)


COMPLIANCE_PRICE = 100.0   # below the 1000 of the volume row: the volume is the budget
COMPLIANCE_HINT = ("options: raise settings.volume_fraction, relax the limit, or use the mass "
                   "objective (A2)")
# Under the mass objective the volume and the mass objective are not options (there is no volume budget).
MASS_CAUSE = "the other rows are met, and this limit is not met at the end of the run"


def compliance_constraint(case, limit: float, name: str | None = None) -> Constraint:
    """c_i <= limit for one load case (a name or an index). Unit N*mm.

    The report has g = c_i/limit - 1. MMA gets ln(c_i/limit) with price 100 (module text).
    """
    def evaluate(p: DesignPoint):
        i = p.case_index(case)
        return Response(float(p.compliance_cases[i]), p.compliance_slopes()[:, i], wrt="rho_stiff")
    return Constraint(name or f"compliance:{case}", evaluate, limit=limit, form="log",
                      price=COMPLIANCE_PRICE, hint=COMPLIANCE_HINT)


# --- the formulation --------------------------------------------------------


@dataclass
class Responses:
    """Raw output of `Formulation.respond`: responses before the chain rule."""

    f0_ref: float
    terms: list[Response]
    cons: list[Response]


@dataclass
class Evaluation:
    """Everything MMA needs at one design, and what the history reports.

    f0, df0 (n,)   scaled objective and its gradient with respect to x. In the bound
                   form f0 = 0 and df0 = 0: the objective is z.
    g (m,), dg (m, n)   rows as MMA gets them (bound rows first, then the constraints in
                   order). A "log" row is ln(value/limit); the report has the linear g.
    a0, a (m,)     coefficients of z for `Mma.step`
    c (m,)         prices of the rows for `Mma.step`
    objective      physical value of the objective (for "max": the largest term)
    constraints    list of dicts name, value, limit, g, violation
    """

    f0: float
    df0: np.ndarray
    g: np.ndarray
    dg: np.ndarray
    a0: float
    a: np.ndarray
    f0_ref: float
    objective: float
    c: np.ndarray = None
    constraints: list = field(default_factory=list)


class Formulation:
    """One objective and a list of constraints. Makes the rows for `Mma`.

    Rows: with the "max" form, one row for each objective term comes first. Then the
    constraints in the given order. `n_rows` is the `m` of `Mma`.
    `xmin` is the lower bound of the design variables for `Mma`. Default (None): the largest
    `xmin` that a row asks for (the stress row asks for 0.02), else 0.
    """

    def __init__(self, objective: Objective, constraints=(), xmin: float | None = None):
        self.objective = objective
        self.constraints = list(constraints)
        self.xmin = (max([c.xmin for c in self.constraints if c.xmin is not None], default=0.0)
                     if xmin is None else xmin)
        self.stage_facts = any(c.stage_aware for c in self.constraints)
        names = [c.name for c in self.constraints]
        if len(set(names)) != len(names):
            raise ValueError(f"constraint names must be unique, got {names}")
        self.n_bound = objective.n_terms if objective.form == "max" else 0
        self.n_rows = self.n_bound + len(self.constraints)
        if self.n_rows < 1:
            raise ValueError("MMA needs one constraint row or more: add a constraint "
                             "(for example volume_constraint) or use the 'max' objective form")

    def respond(self, point: DesignPoint, state: MmaState) -> Responses:
        """Evaluate the objective and the constraints. Set `state.f0_ref` when it is None: at the first call, and at the first pass
        of a beta stage (the loop sets it to None there: `optimize.objective_scale_resets`)."""
        n = point.n_design
        raw = self.objective.terms(point)
        raw = raw if isinstance(raw, list) else [raw]
        if len(raw) != self.objective.n_terms:
            raise ValueError(f"objective '{self.objective.name}' returned {len(raw)} terms, "
                             f"n_terms is {self.objective.n_terms}")
        terms = [_as_response(r, self.objective.name, n) for r in raw]
        cons = []
        for c in self.constraints:
            keeper = c.evaluate if hasattr(c.evaluate, "set_state") else None
            if keeper is not None and c.name in state.rows:
                keeper.set_state(state.rows[c.name])       # the state is the master copy (resume, restore)
            cons.append(_as_response(c.evaluate(point), c.name, n))
            if keeper is not None:
                state.rows[c.name] = keeper.get_state()
        if state.f0_ref is None:
            value = (terms[0].value if self.objective.form == "sum" and len(terms) == 1
                     else self._objective_value(terms))
            state.f0_ref = abs(value) or 1.0
        return Responses(state.f0_ref, terms, cons)

    def measure(self, point: DesignPoint, names=None) -> dict:
        """Linear g of the rows that have `measure`, at a design, with no gradient: `{name: g}`.

        `names` limits the rows (default: all rows with `measure`). `g = value/scale - limit/scale`,
        the number that the history keeps. A row function does not change its state here.
        """
        out = {}
        for c in self.constraints:
            if c.measure is None or (names is not None and c.name not in names):
                continue
            out[c.name] = float(c.measure(point)) / c.scale - c.limit / c.scale
        return out

    def _objective_value(self, terms) -> float:
        values = [t.value for t in terms]
        return float(sum(values)) if self.objective.form == "sum" else float(max(values))

    def chain_rule(self, point: DesignPoint, resp: Responses) -> Evaluation:
        """Gradients with respect to x for all responses, then the scaled rows."""
        responses = resp.terms + resp.cons
        D = point.model.gradient_wrt_x([r.grad for r in responses], [r.wrt for r in responses],
                                       point.rho_tilde, point.beta)
        D = D / np.array([r.divisor for r in responses])[:, None]
        nt = len(resp.terms)
        ref, a0 = resp.f0_ref, 1.0
        obj_value = self._objective_value(resp.terms)
        if self.objective.form == "sum":
            f0 = obj_value / ref
            df0 = D[:nt].sum(axis=0) / ref
            g, dg, a = [], [], []
        else:   # bound form: minimize z, with the rows term_i / ref - z <= 0
            if min(t.value for t in resp.terms) <= 0:
                raise ValueError(f"objective '{self.objective.name}': the 'max' form needs terms "
                                 f"above 0, got {[t.value for t in resp.terms]}")
            f0, df0 = 0.0, np.zeros(point.n_design)
            g = [t.value / ref for t in resp.terms]
            dg = [D[i] / ref for i in range(nt)]
            a = [1.0] * nt
        prices = [C_PENALTY] * len(g)
        report = []
        for c, r, d in zip(self.constraints, resp.cons, D[nt:]):
            gi = r.value / c.scale - c.limit / c.scale      # linear g: what the report shows
            if c.form == "log":
                if not r.value > 0:
                    raise ValueError(f"constraint '{c.name}': the log form needs a value above 0, "
                                     f"got {r.value!r}")
                g.append(float(np.log(r.value / c.limit)))
                dg.append(d / r.value)
            else:
                g.append(gi)
                dg.append(d / c.scale)
            a.append(0.0)
            prices.append(c.price)
            report.append({"name": c.name, "value": r.value, "limit": c.limit, "g": float(gi),
                           "violation": float(max(gi, 0.0)), **(r.info or {})})
        return Evaluation(f0, df0, np.array(g), np.array(dg).reshape(-1, point.n_design), a0,
                          np.array(a), ref, obj_value, np.array(prices), report)

    def evaluate(self, point: DesignPoint, state: MmaState) -> Evaluation:
        """`respond` and `chain_rule` in one call."""
        return self.chain_rule(point, self.respond(point, state))

    def report(self, ev: Evaluation) -> dict:
        """The fields this formulation adds to a history record."""
        return {"objective": {"name": self.objective.name, "form": self.objective.form,
                              "value": ev.objective},
                "constraints": ev.constraints}

    def stage_note(self, previous, beta, it, current=()) -> dict | None:
        """A fact about a beta stage: `{iteration, from_beta, to_beta, row, max_g, feasible, notes}`.

        The note exists only when a row has `stage_aware`, at the first pass (`previous` is None:
        `from_beta`, `row`, `max_g` and `feasible` are None) and at each pass where the beta changes.
        `previous` is the history record of the pass before. `max_g` is the largest linear g of the
        last design of the old stage; `feasible` is `max_g <= NOT_MET`. A row with no result at that design
        (`no_result`) is named instead, `feasible` is false, and the note has `no_result` (text). The schedule goes by
        iteration, so a stage can end with a row not met (KB pitfall stage-advance-while-infeasible).
        `current` is the list of constraint entries of this pass: the rows with `stage_start` add
        their `notes` (a list of text). A mass run whose old stage ended grey (the `grey` fact of the `stage_end` of `previous`, F1-3a)
        adds the key `grey` (`grayness`, `grayness_max`): "all rows met" then holds for the grey design, and `stage_lines` says so.
        """
        if not self.stage_facts or (previous is not None and previous["beta"] == beta):
            return None
        by_name = {c.name: c for c in self.constraints}
        notes = [t for e in current for t in (by_name[e["name"]].stage_start(e)
                                              if by_name[e["name"]].stage_start else [])]
        note = {"iteration": int(it), "from_beta": None, "to_beta": float(beta), "row": None,
                "max_g": None, "feasible": None, "notes": notes}
        if previous is not None:
            cons = previous["constraints"]
            unjudged = next((c for c in cons if no_result(c)), None)    # a row with no result is not judged (not met)
            worst = unjudged or max(cons, key=lambda c: c["g"])
            note.update(from_beta=previous["beta"], row=worst["name"], max_g=worst["g"],
                        feasible=unjudged is None and bool(worst["g"] <= NOT_MET))
            if unjudged is not None:
                note["no_result"] = no_result_text(unjudged)
            grey = (previous.get("stage_end") or {}).get("grey")
            if grey and "grayness" in grey:
                note["grey"] = {"grayness": grey["grayness"], "grayness_max": grey["grayness_max"]}
        return note

    @staticmethod
    def stage_lines(note) -> list[str]:
        """The run log lines for a stage note."""
        if note["from_beta"] is None:
            lines = [f"beta stage start: beta {note['to_beta']:g} at pass {note['iteration']}"]
        else:
            grey = note.get("grey")
            state = ("all rows met (a stress row is the relaxed measure of the optimizer, not the stress of the part)"
                     + ("" if not grey else f"; the design of the old stage is grey (grayness {grey['grayness']:.3g}, grayness_max {grey['grayness_max']:.3g}), "
                                            "so \"met\" holds for this grey design and not for the thresholded part")
                     if note["feasible"] else
                     f"NOT feasible: {note['row']} not judged, {note['no_result']}" if note.get("no_result") else
                     f"NOT feasible: {note['row']} not met (limit g {NOT_MET:g})")
            lines = [f"beta stage {note['from_beta']:g} -> {note['to_beta']:g} at pass {note['iteration']}: "
                     f"largest g of the old stage {note['max_g']:+.4f} ({note['row']}), {state}; "
                     f"the schedule goes by iteration count"]
        return lines + ["  fact: " + t for t in note["notes"]]

    def log_extra(self, ev) -> str:
        """Extra words for the log line of a pass. Empty for the default compliance run."""
        if self.objective.name == "weighted_compliance" and not self.stage_facts:
            return ""
        worst = max(ev.constraints, key=lambda c: c["g"]) if ev.constraints else None
        text = f"  obj={ev.objective:.5g}"
        return text + (f"  max_g={worst['g']:+.4f} ({worst['name']})" if worst else "")

    def unmet(self, constraints) -> list[str]:
        """Log lines for the constraints that are not met at the end. A fact, not an error.

        A row is not met when its linear g is above `NOT_MET`. When every row that is not met
        has a hint, one more line gives it: the cause is then the row and not the volume.
        A row with `detail` adds its words to its line, a row with `facts` adds lines in any state.
        A row with no result at the last pass (`no_result`) has its own line: it is not judged, so it is not met.
        """
        by_name = {c.name: c for c in self.constraints}
        bad = [c for c in constraints if c["g"] > NOT_MET and not no_result(c)]
        lines = [f"constraint not judged: {c['name']}  {no_result_text(c)}; the verdict counts it as not met"
                 for c in constraints if no_result(c)]
        for c in bad:
            if c["limit"] != 0:
                rel = f"{100 * (c['value'] - c['limit']) / abs(c['limit']):+.2f}% of the limit"
            else:
                rel = f"{c['value'] - c['limit']:+.5g} above the limit"
            line = (f"constraint not met: {c['name']}  value {c['value']:.5g}  "
                    f"limit {c['limit']:.5g}  ({rel})")
            row = by_name.get(c["name"])
            if row is not None and row.detail is not None:
                line += "; " + row.detail(c)
            lines.append(line)
        for c in constraints:
            row = by_name.get(c["name"])
            if row is not None and row.facts is not None:
                lines += row.facts(c)
        rows = [by_name[c["name"]] for c in bad if c["name"] in by_name]
        hints = {r.hint for r in rows}
        causes = {r.cause for r in rows}
        if bad and None not in hints and len(hints) == 1 and len(rows) == len(bad) and not any(no_result(c) for c in constraints):
            cause = causes.pop() if len(causes) == 1 and None not in causes else None
            hint = hints.pop()
            if cause is None and hint == COMPLIANCE_HINT and self.objective.name == "mass":
                cause, hint = self._mass_summary({c["name"] for c in bad})
            lines.append(f"  {cause or 'the other rows are met, so this limit does not fit at this volume'}; {hint}")
        return lines

    def _mass_summary(self, unmet_names) -> tuple[str, str]:
        """Cause and options of a compliance limit that is not met under the mass objective.

        There is no volume budget in a mass run, so the text has no volume and no mass-objective option. A met row with a
        `rival` (a higher price) is named as a possible cause, and its option comes first.
        """
        rivals = [c.rival for c in self.constraints if c.rival is not None and c.name not in unmet_names]
        options = [option for _clause, option in rivals] + ["relax the limit"]
        listed = options[0] if len(options) == 1 else ", ".join(options[:-1]) + ", or " + options[-1]
        return "; ".join([MASS_CAUSE] + [clause for clause, _option in rivals]), f"options: {listed}"
