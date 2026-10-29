"""The perimeter bound `settings.perimeter_max_mm2` as a constraint row (A4c, CONTRACTS.md section 26.3).

`P` is the surface of the physical density, counted in voxel faces of area `h^2`. The faces that count are
the faces that touch a design voxel, and the faces between a solid voxel and a void voxel (a constant):

  design voxel             its physical density `rho` (the nominal one, also in robust mode)
  FIXED and LOAD voxel     density 1 (this includes regions.solid, interface unions and spring regions)
  EMPTY, KEEP_OUT voxel    density 0
  outside the grid         density 0

One face with the density jump `d` gives `h^2 f(d)` with the smooth, normalised jump

  f(d) = (sqrt(d^2 + eps^2) - eps) / (sqrt(1 + eps^2) - eps),     f(0) = 0, f(1) = 1, eps = EPS.

So a binary jump is exactly one face (one design voxel in void: `P = 6 h^2`) and the term has a slope at `d = 0`. The row is
`P <= bound`, linear form, price 1000 (the volume row). The gradient is with respect to `rho`: `Model.gradient_wrt_x` does the
projection, the filter and the mirror average (so these need no code here).

`FaceList` holds the faces as two index arrays into `[rho_1 ... rho_n, 0, 1]`. The memory follows the number of design voxels.
`p_min_mm2` is the smallest `P` of any design with the density 0 or 1 in every voxel: a minimum cut of the design graph. A bound
below it cannot be met by such a design, and `PerimeterRow.bind` refuses it with the facts and the options. A gray design can have
a smooth `P` a little below it (measured: 0.043 % on one example, at a near-void design), so the refusal is a floor for the
designs that are printed, and not a proof for every gray density.
"""

from __future__ import annotations

import math

import numpy as np

from .config import UserError
from .formulation import COMPLIANCE_PRICE, NOT_MET, Constraint, Response
from .mma import C_PENALTY
from .voxelize import DESIGN, FIXED, LOAD

EPS = 1e-3          # smoothing of the face jump, in density. It shapes the gradient; it is not a limit (section 26.3)
REL_TOL = 1e-9      # a bound this close to `p_min` is not below it (round-off only; it is not a design limit)
ROW_NAME = "perimeter"
UNIT = "mm2"
UNIT_TEXT = ("P is voxel face area in mm2 (a staircase surface; in one measured problem about 1.24 times the area of "
             "final.stl)")
HINT = "options: raise settings.perimeter_max_mm2, or relax the other rows (volume_fraction, compliance and stress limits)"
CAUSE = "the other rows are met, and the perimeter bound is not met at the end of the run"
# The price of the row (the default of a linear row) is above the price of a compliance limit, so a tight bound can take
# material from a limit. `PRICE_FACT` is in `problem_facts` of a mass run; `RIVAL` is in the summary of the run log.
PRICES = f"the perimeter row has price {C_PENALTY:g} and a compliance limit has price {COMPLIANCE_PRICE:g}"
PRICE_FACT = PRICES[0].upper() + PRICES[1:] + ", so a tight bound can leave a compliance limit not met"
RIVAL = (PRICES + ", so a tight perimeter bound can be a cause", "raise settings.perimeter_max_mm2")


def below_floor(bound, p_min) -> bool:
    """True when `bound` is below `p_min` by more than round-off: the bound that `PerimeterRow.bind` refuses."""
    return bound < p_min and not math.isclose(bound, p_min, rel_tol=REL_TOL)


def shown_p_min(p_min) -> str:
    """`p_min` for a text, at 0.1 mm2, as a number that `PerimeterRow.bind` accepts as a bound.

    It is the nearest value at 0.1 mm2, or the next one up when the nearest one is below `p_min` (display rounding: for
    example 6779.52 shows as 6779.6). One form for each text that shows `p_min` or tells the user to raise the bound to it.
    """
    value = round(float(p_min), 1)
    if below_floor(value, p_min):
        value = math.ceil(p_min * 10) / 10
    return f"{value:.1f}"


class FaceList:
    """The voxel faces that count in `P`, for one label array. `value_grad(rho)` gives `P` (mm2) and its gradient.

    `rho` has one entry for each design voxel, in the order of `np.flatnonzero(labels == DESIGN)` (the order of `x`).
    `fixed_part_mm2` is the part of `P` that no design changes: the faces between a solid voxel and a void voxel
    (void: EMPTY, KEEP_OUT or outside the grid) with no design voxel on either side. It is in `P`.
    """

    def __init__(self, labels, h, eps=EPS):
        lab = np.asarray(labels)
        self.h, self.eps = float(h), float(eps)
        self.norm = math.sqrt(1.0 + self.eps ** 2) - self.eps      # f before the divide, at d = 1
        design = lab == DESIGN
        solid = (lab == FIXED) | (lab == LOAD)
        n = int(design.sum())
        self.n_design = n
        slot = np.full(lab.shape, n, dtype=np.int32)               # slot n: void (density 0)
        slot[solid] = n + 1                                        # slot n + 1: solid (density 1)
        slot[design] = np.arange(n, dtype=np.int32)                # C order, as the order of x
        ii, jj, const = [], [], 0
        for ax in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[ax], hi[ax] = slice(0, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            d_lo, d_hi = design[lo], design[hi]
            touch = d_lo | d_hi
            ii.append(slot[lo][touch])
            jj.append(slot[hi][touch])
            const += int((~d_lo & ~d_hi & (solid[lo] != solid[hi])).sum())   # solid next to void, no design voxel
            for plane in (0, lab.shape[ax] - 1):                   # the face between a border voxel and the outside (void)
                idx = [slice(None)] * 3
                idx[ax] = plane
                idx = tuple(idx)
                d = design[idx]
                ii.append(slot[idx][d])
                jj.append(np.full(int(d.sum()), n, dtype=np.int32))
                const += int(solid[idx].sum())
        self.ii, self.jj = np.concatenate(ii), np.concatenate(jj)
        self.n_const_faces = const
        self.n_design_faces = int(self.ii.size)
        self.n_design_solid_faces = int(((self.ii == n + 1) | (self.jj == n + 1)).sum())
        self.n_design_void_faces = int(((self.ii == n) | (self.jj == n)).sum())

    @property
    def fixed_part_mm2(self) -> float:
        return self.h ** 2 * self.n_const_faces

    def _extended(self, rho) -> np.ndarray:
        e = np.empty(self.n_design + 2)
        e[:self.n_design] = rho
        e[self.n_design], e[self.n_design + 1] = 0.0, 1.0
        return e

    def value(self, rho) -> float:
        """`P` in mm2 (the fixed part included)."""
        e = self._extended(rho)
        d = e[self.ii] - e[self.jj]
        return self.h ** 2 * (float((np.sqrt(d * d + self.eps ** 2) - self.eps).sum()) / self.norm + self.n_const_faces)

    def value_grad(self, rho):
        """`(P, dP/drho)`: `P` in mm2 and its gradient with respect to the density of the design voxels."""
        n = self.n_design
        e = self._extended(rho)
        d = e[self.ii] - e[self.jj]
        r = np.sqrt(d * d + self.eps ** 2)
        total = float((r - self.eps).sum()) / self.norm + self.n_const_faces
        s = d / r
        g = np.bincount(self.ii, weights=s, minlength=n + 2) - np.bincount(self.jj, weights=s, minlength=n + 2)
        return self.h ** 2 * total, (self.h ** 2 / self.norm) * g[:n]

    def min_cut_faces(self) -> int:
        """The smallest number of faces of the cut that separates the solid voxels from the void voxels.

        The design graph has one node for each design voxel and the nodes `source` (solid) and `sink` (void). A
        design-design face is an edge of capacity 1 (both ways). A design-solid face is an edge from the source, and a
        design-void face is an edge to the sink, each of capacity 1. The minimum cut is the smallest number of faces
        that a design with densities 0 and 1 can have, apart from the constant faces. The smooth `P` of a gray
        design is at most the exact value of its total variation, and the total variation of any gray design is at
        least this number (co-area formula).
        """
        n = self.n_design
        if n == 0:
            return 0
        from scipy.sparse import csr_matrix            # scipy is a main dependency; the import keeps `config` light
        from scipy.sparse.csgraph import maximum_flow
        ii, jj = self.ii.astype(np.int64), self.jj.astype(np.int64)
        source, sink = n, n + 1
        rows, cols = [], []
        both = (ii < n) & (jj < n)
        rows += [ii[both], jj[both]]
        cols += [jj[both], ii[both]]
        for a, b in ((ii, jj), (jj, ii)):
            to_sink = (a < n) & (b == n)
            rows.append(a[to_sink])
            cols.append(np.full(int(to_sink.sum()), sink, dtype=np.int64))
            from_source = (a < n) & (b == n + 1)
            rows.append(np.full(int(from_source.sum()), source, dtype=np.int64))
            cols.append(a[from_source])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n + 2, n + 2))
        graph.sum_duplicates()                          # several faces of one voxel join into one edge of larger capacity
        return int(maximum_flow(graph, source, sink).flow_value)

    def p_min_mm2(self) -> float:
        """The smallest `P` (mm2) of any design with the density 0 or 1 in every voxel, the fixed part included."""
        return self.h ** 2 * (self.min_cut_faces() + self.n_const_faces)


def start_density(model) -> np.ndarray:
    """The physical density of the start design: `x = volume_fraction` everywhere, at the first beta of the schedule.

    It is the chain of `Model.density` (mirror average, filter, projection) without the record of the mirror deviation
    that `Model.density` keeps for the end of a run. The pass 0 of a run computes the same density.
    """
    from . import symmetry
    from .filter import filtered_density, heaviside_project
    from .optimize import ETA
    st = model.problem.config.settings
    x = np.full(model.n_design, st.volume_fraction)
    rho_tilde = filtered_density(model.F, model.b, symmetry.average(model, x))
    return heaviside_project(rho_tilde, float(st.beta_schedule[0][1]), ETA)


def refusal_text(bound, fixed, p_min) -> str:
    """The `UserError` text for a bound below the smallest `P`: the facts and the options.

    `p_min` is shown rounded as `shown_p_min` says, so the value in the advice is a bound that `bind` accepts.
    """
    shown = shown_p_min(p_min)
    return (f"settings.perimeter_max_mm2 is {bound:g} mm2, and no design with density 0 or 1 can have a perimeter below "
            f"{shown} mm2 (the minimum cut of the design space, rounded up to 0.1 mm2). The solid regions alone (fixed, "
            f"load, solid and spring voxels) have {fixed:.1f} mm2 of surface to void, and any design adds at least "
            f"{p_min - fixed:.1f} mm2 more. {UNIT_TEXT}. Options: raise settings.perimeter_max_mm2 to {shown} mm2 or "
            f"more; reduce the solid regions; make the design space bigger")


class PerimeterRow:
    """The row `perimeter` of a config for one problem (or model). It builds its face list at the first use.

    `bind(problem)` builds the list, finds `p_min_mm2` and raises a `UserError` for a bound below it. `bind_model(model)` does
    the same for a model, and adds `p_base_mm2` (P of the start design) and sends the facts to `log`. A row that
    is not bound yet binds to the model of its first `DesignPoint`. A config without the key has `active` False and no row.
    """

    def __init__(self, config, log=None):
        self.bound = config.settings.perimeter_max_mm2
        self.log = log
        self.faces = None
        self.p_min = self.p_base = None
        self._bound_to = None

    @property
    def active(self) -> bool:
        return self.bound is not None

    def bind(self, problem, key=None) -> dict:
        """Build the face list for `problem`, check the bound and return the numbers (`numbers()`)."""
        faces = FaceList(problem.labels, problem.h)
        p_min = faces.p_min_mm2()
        if below_floor(self.bound, p_min):
            raise UserError(refusal_text(self.bound, faces.fixed_part_mm2, p_min))
        self.faces, self.p_min, self._bound_to = faces, p_min, key
        return self.numbers()

    def bind_model(self, model) -> dict:
        """Bind to `model`, set `p_base_mm2`, send the facts to `log` (if any) and return the numbers."""
        numbers = self.bind(model.problem, key=model)
        self.p_base = self.faces.value(start_density(model))
        numbers = self.numbers()
        if self.log is not None:
            for line in self.lines(numbers):
                self.log("fact: " + line)
        return numbers

    def faces_for(self, model) -> FaceList:
        if self._bound_to is not model:
            self.bind_model(model)
        return self.faces

    def numbers(self) -> dict:
        """The numbers of the problem: bound, fixed part, `p_min_mm2`, the ratio, `p_base_mm2` (None before a model is bound)."""
        return {"max_mm2": self.bound, "eps": EPS, "fixed_part_mm2": self.faces.fixed_part_mm2,
                "p_min_mm2": self.p_min, "p_base_mm2": self.p_base, "bound_over_p_min": self.bound / self.p_min if self.p_min else None}

    def block(self, p_final=None) -> dict:
        """The `perimeter` block of `run.json`. `p_final_mm2` is added at the end of the run."""
        n = self.numbers()
        out = {k: n[k] for k in ("max_mm2", "eps", "fixed_part_mm2", "p_min_mm2", "p_base_mm2")}
        if p_final is not None:
            out["p_final_mm2"] = float(p_final)
        return out

    @staticmethod
    def lines(n) -> list:
        """Fact lines (no prefix): the bound against the numbers of the problem. All are facts; none blocks a run."""
        ratio = n["bound_over_p_min"]
        ratio_text = f"{ratio:.3f}" if ratio is not None else "not defined (the smallest possible P is 0)"
        text = (f"perimeter: bound {n['max_mm2']:g} mm2; fixed part {n['fixed_part_mm2']:.1f} mm2 (solid voxels next to "
                f"void); smallest possible {shown_p_min(n['p_min_mm2'])} mm2 (minimum cut, for designs with density 0 or 1, "
                f"rounded up to 0.1 mm2); bound / smallest = {ratio_text}")
        if n.get("p_base_mm2") is not None:
            text += f"; P of the start design {n['p_base_mm2']:.1f} mm2 ({n['p_base_mm2'] / n['max_mm2']:.3f} of the bound)"
        return [text + f". {UNIT_TEXT}"]

    def constraint(self) -> Constraint:
        """The row: `P <= bound`, linear form, price 1000 (the default), gradient with respect to `rho`."""
        def evaluate(point):
            P, grad = self.faces_for(point.model).value_grad(point.rho)
            return Response(P, grad)

        def detail(entry):
            if not self.p_min:
                return "the smallest possible P is 0 mm2"
            return (f"the bound is {self.bound / self.p_min:.3f} times the smallest possible P "
                    f"({shown_p_min(self.p_min)} mm2)")

        def facts(entry):
            state = "met" if entry["g"] <= NOT_MET else "not met"
            return [f"fact: perimeter: P of the start design {self.p_base:.1f} mm2, P of the last design "
                    f"{entry['value']:.1f} mm2, bound {entry['limit']:g} mm2 ({state}); smallest possible "
                    f"{shown_p_min(self.p_min)} mm2"]

        return Constraint(ROW_NAME, evaluate, limit=self.bound, hint=HINT, cause=CAUSE, detail=detail, facts=facts,
                          rival=RIVAL)

    def constraints(self) -> list:
        return [self.constraint()] if self.active else []


def check_facts(config, problem):
    """The numbers of the bound for `topopt check` and the start of `topopt run`: no model, no filter.

    None without the key. Raises `UserError` for a bound below `p_min_mm2`. `p_base_mm2` is None (it needs the filter).
    """
    row = PerimeterRow(config)
    return row.bind(problem) if row.active else None
