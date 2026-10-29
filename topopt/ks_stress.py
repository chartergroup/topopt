"""Stress row for MMA: relaxed von Mises stress, KS aggregate (A2, CONTRACTS.md section 17).

The measure is the one of the earlier pipeline of the same author (revision d43bcf3, `pp_core_final.py`), on the hex8
centre strain of the demo:

    sigma_e = rho_e^s * D0 * eps_e(u)      relaxed stress, s = 0.5, D0 with the full modulus E
    vm_e    = von Mises(sigma_e)           floor 1e-12 under the root (MPa^2)
    z_e     = vm_e / limit
    S       = zmax + ln(sum exp(q (z_e - zmax))) / q          KS, log-sum-exp with a max shift, q = 40
    row     = c * S - 1  <=  0

`c` is the KS scale. The default (`"auto"`) sets `c = zmax / S` at the first design of each beta
stage, so the row equals the relaxed peak at that moment (`update_ks_scales`, `pp_core_final.py`
602-608). `c = 1` makes the row a bound: `S >= zmax`, so `S <= 1` gives `zmax <= 1`. The limit itself
never changes (rule R21 of the pipeline: a limit is not relaxed to get a pass).

The stress is the relaxed measure of the optimizer. It is not the stress of a printed part
(KB pitfall relaxed-stress-not-physical). `topopt.stress_report` gives the physical numbers.

The aggregation functions (`relaxed_von_mises`, `ks_aggregate`, `ks_scale`) work on plain arrays. The
tests compare them with a fixture that the pipeline code made (`tests/data/ks_pipeline_fixture.json`).
The equivalence holds at this layer only: the pipeline uses P1 tets, the demo uses hex8 voxels.
"""

from __future__ import annotations

import numpy as np

from .fe import isotropic_D, strain_matrix_centre
from .formulation import NOT_MET, Constraint, Response

Q_DEFAULT = 40.0       # KS sharpness (pipeline: `PrintConfig.ks_q`)
POWER_DEFAULT = 0.5    # relaxation power s (pipeline: `PrintConfig.stress_power`)
SCALE_DEFAULT = "auto"
RHO_FLOOR = 1e-12      # floor of rho under the power (pipeline: `relaxed_vm`, `_stress_nominal`)
VM2_FLOOR = 1e-12      # floor of vm^2 in MPa^2 (pipeline: `relaxed_vm`)
XMIN_STRESS = 0.02     # lowest x with a stress row (pipeline: `XMIN` of `run_final_pass.py`)
PEAK_ABOVE_TOL = NOT_MET   # a relaxed peak counts as above the limit when it is more than this part of the limit above it. It is the
                           # tolerance of "not met" (g above NOT_MET): with the KS scale c = 1, a peak above the limit by more than
                           # this is a row that is not met, so a row that is met with such a peak has c < 1.

STRESS_HINT = ("options: relax the stress limit (a decision for the user, the tool does not change it), "
               "add material (a larger design space), run more iterations, or use a finer voxel size (the default "
               "filter radius is 1.5 voxels, so its size in mm changes with the voxel size)")
STRESS_CAUSE = "this stress limit is not met by the design that MMA reached"


# --- aggregation on plain arrays ---------------------------------------------


def relaxed_von_mises(eps, D0, rho, power=POWER_DEFAULT):
    """Relaxed von Mises stress of elements. `eps` (n, 6) strain, `rho` (n,) density.

    Returns `(vm, dvm_ds, fac)`: von Mises (n,), its derivative with respect to the six stress
    components (n, 6) and the factor `max(rho, 1e-12)^power` (n,). Same formulas as `relaxed_vm`
    of the pipeline (`pp_core_final.py` 404-414).
    """
    fac = np.maximum(rho, RHO_FLOOR) ** power
    s = fac[:, None] * (eps @ D0.T)
    sx, sy, sz, txy, tyz, tzx = s.T
    vm2 = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2)
    vm = np.sqrt(np.maximum(vm2, VM2_FLOOR))
    dvm_ds = np.column_stack((2 * sx - sy - sz, 2 * sy - sx - sz, 2 * sz - sx - sy,
                              6 * txy, 6 * tyz, 6 * tzx)) / (2 * vm[:, None])
    return vm, dvm_ds, fac


def ks_aggregate(vm, limit, q=Q_DEFAULT):
    """KS aggregate of the stress ratio `z = vm / limit`. Returns `(S, w, zmax)`.

    `S = zmax + ln(sum exp(q (z - zmax))) / q`, `w` are the softmax weights (they sum to 1),
    `zmax = max z`. Same formulas as `_stress_nominal` (`pp_core_final.py` 466-470).
    """
    z = np.asarray(vm, dtype=float) / limit
    a = float(z.max())
    ez = np.exp(q * (z - a))
    S = a + np.log(ez.sum()) / q
    return float(S), ez / ez.sum(), a


def ks_scale(zmax, S):
    """The KS scale `c = zmax / S` (`update_ks_scales`, `pp_core_final.py` 602-608)."""
    return float(zmax / S)


def adjoint_strain_load(w, limit, fac, dvm_ds, D0):
    """The derivative of S with respect to the strain row of each element: `(w / limit) fac (dvm/dsigma @ D0)`.

    Shape (n, 6). The adjoint load is `sum_e B_e^T` of it. Same line as `g6` of the pipeline (`pp_core_final.py` 472).
    """
    return (w / limit)[:, None] * fac[:, None] * (dvm_ds @ D0)


def explicit_rho_term(w, vm, limit, rho, power=POWER_DEFAULT):
    """The part of dS/drho that does not go through the displacements: `w vm / limit s / rho`."""
    return w * vm / limit * power / np.maximum(rho, RHO_FLOOR)


# --- the model layer (hex8, adjoint) -----------------------------------------


def _bc(model):
    if getattr(model, "_ks_bc", None) is None:
        model._ks_bc = strain_matrix_centre(model.problem.h)
        model._ks_D0 = isotropic_D(model.E, model.problem.config.material.nu)
    return model._ks_bc, model._ks_D0


def element_strain(model, u):
    """Strain (n_design, 6) at the centre of each design voxel. `u` is a vector of the free dofs."""
    B, _ = _bc(model)
    u_all = np.zeros(model.n_dof_all)
    u_all[model.free_dofs] = u
    return u_all[model.edof_design] @ B.T


class StressRow:
    """The function of one stress row: `DesignPoint -> Response`. Keeps the KS scale `c`.

    case    load case (name or index)
    limit   stress limit in MPa (> 0)
    q, power   KS sharpness and relaxation power
    scale   "auto": `c = zmax / S` at the first design of each beta stage. A number: that `c`, kept.
    The value is `c * S` (1 = at the limit). `info` of the response holds the MPa numbers.

    State: `c` and the beta of the stage in which `c` was set. `get_state` and `set_state` copy them
    to and from `MmaState.rows` (`Formulation.respond` does it at every pass), so a resume or a restored
    optimizer state sets them again. `measure(point)` gives the value with no adjoint and no change of `c`.
    """

    def __init__(self, case, limit, q=Q_DEFAULT, power=POWER_DEFAULT, scale=SCALE_DEFAULT):
        if not (np.isfinite(limit) and limit > 0):
            raise ValueError(f"the stress limit must be > 0 (MPa), got {limit!r}")
        if not (np.isfinite(q) and q > 0):
            raise ValueError(f"q must be > 0, got {q!r}")
        if not (np.isfinite(power) and 0 < power <= 1):
            raise ValueError(f"the relaxation power must be in (0, 1], got {power!r}")
        if scale != "auto" and not (isinstance(scale, (int, float)) and np.isfinite(scale) and scale > 0):
            raise ValueError(f"the KS scale must be 'auto' or a number > 0, got {scale!r}")
        self.case, self.limit, self.q, self.power, self.scale = case, float(limit), float(q), float(power), scale
        self.c = None if scale == "auto" else float(scale)
        self._beta = None

    def get_state(self) -> dict:
        """`{"c": KS scale or None, "beta": beta of the stage that set it or None}`."""
        return {"c": self.c, "beta": self._beta}

    def set_state(self, state) -> None:
        """Set `c` and the stage beta from `get_state()`. None: the start state of a new run."""
        if state is None:
            self.c, self._beta = (None if self.scale == "auto" else float(self.scale)), None
        else:
            self.c, self._beta = state["c"], state["beta"]

    def _aggregate(self, point):
        """Strain, relaxed von Mises and the KS aggregate at a design point (no adjoint)."""
        model = point.model
        u = point.U[:, point.case_index(self.case)]
        _, D0 = _bc(model)
        vm, dvm_ds, fac = relaxed_von_mises(element_strain(model, u), D0, point.rho, self.power)
        S, w, zmax = ks_aggregate(vm, self.limit, self.q)
        return u, vm, dvm_ds, fac, S, w, zmax

    def measure(self, point) -> float:
        """The row value `c * S` at a design, with the `c` of the last evaluation. No adjoint solve.

        It does not set `c`. Use it after `__call__` of the same stage (step acceptance does).
        """
        if self.c is None:
            raise ValueError("StressRow.measure needs c: call the row once at the stage start first")
        return float(self.c * self._aggregate(point)[4])

    def __call__(self, point) -> Response:
        model = point.model
        u, vm, dvm_ds, fac, S, w, zmax = self._aggregate(point)
        _, D0 = _bc(model)
        changed = False
        if self.scale == "auto" and (self.c is None or point.beta != self._beta):
            self.c, changed = ks_scale(zmax, S), True    # stage start: the row equals the peak
        self._beta = point.beta
        c = self.c
        # adjoint: K lam = dS/du. dS/du = sum_e B_e^T (w_e / limit) fac_e D0^T dvm/dsigma_e
        B, _ = _bc(model)
        g6 = adjoint_strain_load(w, self.limit, fac, dvm_ds, D0)
        g24 = g6 @ B
        rhs_full = np.bincount(model.edof_design.ravel(), weights=g24.ravel(), minlength=model.n_dof_all)
        lam = point.solve_adjoint(model.to_free(rhs_full))
        grad = c * (explicit_rho_term(w, vm, self.limit, point.rho, self.power)
                    - point.stiffness_derivative(lam, u))
        info = {"stress_limit_MPa": self.limit, "ks_MPa": S * self.limit,
                "relaxed_peak_MPa": zmax * self.limit, "c": c, "c_changed": changed, "q": self.q}
        return Response(float(c * S), grad, info=info)


def _detail(entry) -> str:
    """The text that the run log adds to a stress row (the numbers are relaxed, not physical)."""
    return (f"relaxed peak {entry['relaxed_peak_MPa']:.4g} MPa, KS {entry['ks_MPa']:.4g} MPa, "
            f"stress limit {entry['stress_limit_MPa']:.4g} MPa, c {entry['c']:.4g} "
            f"(relaxed measure of the optimizer, not physical)")


def peak_above_limit(peak, limit) -> bool:
    """True when the relaxed peak is above the stress limit by more than `PEAK_ABOVE_TOL` of the limit (both in MPa).

    The one comparison of the run log line `stress peak above the limit` (`_facts`) and of the flag
    `metrics.stress.cases.<case>.relaxed.peak_above_limit` (`stress_report.relaxed_numbers`): they cannot disagree.
    The test does not look at the state of the row: the peak can be above the limit with the row met (`c < 1`)
    or not met.
    """
    return bool(peak > limit * (1.0 + PEAK_ABOVE_TOL))


def _facts(entry) -> list[str]:
    """Extra facts at the end of a run. The relaxed peak can sit above the limit with the row met:
    `c < 1` makes the row a calibrated estimate of the peak and not a bound."""
    peak, lim = entry["relaxed_peak_MPa"], entry["stress_limit_MPa"]
    if peak_above_limit(peak, lim):
        return [f"stress peak above the limit: {entry['name']}  relaxed peak {peak:.4g} MPa, "
                f"limit {lim:.4g} MPa ({100 * (peak - lim) / lim:+.2f}%); the KS row is c*S = "
                f"{entry['value']:.4g} with c = {entry['c']:.4g} (settings.stress_scale 1 gives a bound: "
                f"KS >= peak)"]
    return []


C_NOTE = 0.5   # a stage start with c below this value gets a fact in the log


def _stage_start(entry) -> list[str]:
    """At the first pass of a beta stage: say when the KS scale is far below 1.

    `c = zmax / KS` is small when the relaxed peak is far below the limit at the stage start, because
    the log-sum-exp has a gap above the peak of up to ln(n)/q. The row `c*S - 1` then allows a peak
    above the limit later (measured on the test cantilever: c 0.155, peak 185x the limit).
    """
    if not (entry.get("c_changed") and entry["c"] < C_NOTE):
        return []
    return [f"{entry['name']}: KS scale c = {entry['c']:.3f} at this stage start (relaxed peak "
            f"{entry['relaxed_peak_MPa']:.4g} MPa, KS {entry['ks_MPa']:.4g} MPa, limit "
            f"{entry['stress_limit_MPa']:.4g} MPa). The row c*S - 1 allows a relaxed peak above the limit "
            f"while the stress is far below it. options: start the stage closer to the limit, or "
            f"settings.stress_scale 1 (KS is then a bound of the peak)"]


def stress_ks_constraint(case, limit, q=Q_DEFAULT, power=POWER_DEFAULT, scale=SCALE_DEFAULT,
                         name=None) -> Constraint:
    """The stress row of one load case: relaxed von Mises, KS aggregate, `c*S - 1 <= 0`.

    `limit` is in MPa and never changes during a run. `scale` is "auto" (the default: `c = zmax/KS`
    at the first design of each beta stage) or a number such as 1. The row asks for `xmin` 0.02
    (a `Formulation` without its own `xmin` uses it) and records beta stage ends.
    The history entry of the row has the extra keys `stress_limit_MPa`, `ks_MPa`,
    `relaxed_peak_MPa`, `c`, `c_changed` and `q`.
    """
    row = StressRow(case, limit, q, power, scale)
    return Constraint(name or f"stress:{case}", row, limit=1.0, scale=1.0, hint=STRESS_HINT,
                      cause=STRESS_CAUSE, xmin=XMIN_STRESS, stage_aware=True, detail=_detail,
                      facts=_facts, stage_start=_stage_start, measure=row.measure)
