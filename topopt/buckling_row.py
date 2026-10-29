"""The buckling row of the optimizer (A6b, CONTRACTS.md section 39).

Opt-in: the YAML key `buckling.min_factor` turns it on, and `topopt.optimize.formulation_from_config` builds one row
for each load case, named `buckling:<case>`. Without the key this module is not imported.

The row asks that the lowest linear buckling load factor of the load case is at least `min_factor` (F). The factor is the
multiplier on the load case as the YAML gives it. For the density `rho_stiff` of the pass it solves

    (-Kg) phi_j = mu_j K phi_j,    phi_j^T K phi_j = 1,    mu_j = 1 / lambda_j

for the `modes` largest `mu` (ARPACK, regular inverse mode, with the factor of K that the run holds: `topopt.buckling`).
The row value is the KS aggregate of `z_j = F mu_j`:

    S = a + ln(sum_j exp(q (z_j - a))) / q,    a = max z,    limit 1, scale 1, so g = S - 1.

`S >= max z`, so `S <= 1` gives `lambda_1 >= F`. `S <= max z + ln(k)/q` (the KS offset). A `mu` of zero or below
(no buckling load) enters the sum as it is: the row needs no special case for it.

The gradient of `S` with respect to `rho_stiff` (the weights are `w_j = F p_j`, `p_j` the softmax weights) has three parts:

    explicit   - p rho^(p-1) sigma_e . Q_e        Q_e,c = sum_j w_j phi_je^T G_c phi_je     (Kg depends on rho directly)
    K part     - dE_e  sum_j w_j mu_j phi_je^T ke phi_je
    stress     + dE_e lam_e^T ke u_e              K lam = s,  s = sum_e Bc^T D0 (rho_e^p Q_e)   (Kg depends on u)

The weights of all modes add up in `s`, so one adjoint solve for each load case is enough (not one for each mode).
The sum over `e` in `Q` and `s` runs over all active voxels (solid ones too): their stress depends on rho.
The KS sum over a complete cluster of repeated modes does not depend on the basis in the cluster. The row asks for one mode
more than it uses (`k + 1`) and records when the gap at the cut is small (`cluster_cut`, `ks_share_last`).

With joints (CONTRACTS.md section 38) the run holds the factor of the bordered matrix, whose grid block inverse is `S^-1`, `S = K + Kb - C D^-1 C^T`.
The row uses `S` where this text says `K` (the eigen solve, the K-normalisation, the adjoint and both residuals): `Geometry.border` condenses the
matrix of the pass. `dS/drho = dK/drho` because `Kb`, `C` and `D` do not depend on the density, so the three gradient parts do not change.
The beams have no geometric stiffness in this model: the factors are for buckling of the part, not of the beams.

The row does not run at every pass: `evaluation_reason` is the skip rule of the pipeline (`run_final_pass.py`: BUCK_SKIP_FACTOR,
BUCK_EVERY). A skipped pass returns the last value and a zero gradient. The row keeps its state in `get_state` / `set_state`
(the optimizer state `MmaState.rows`, so a resume sets it again).

Warm start (#155, W2). After an evaluation with a result the row keeps `v0 = sum_j c_j / sqrt(c_j^T K c_j)` over all columns `c_j` that the
eigen solve returned (`modes + 1`), and gives it as the start vector of the next evaluation of the same row. This takes 13 percent fewer
solves on the hook and gives the same factors (the solver tolerance is the same). The first evaluation, an evaluation after a state
of other settings (`min_factor`, `ks_q`, `modes`, `tol`), and an evaluation whose kept vector has the wrong length or is not finite start
cold (the fixed random vector of `buckling_modes`). `v0` is part of the row state, so it is in the checkpoint (an array of the `.npz`,
`topopt.checkpoint`), and a resume or a continuation does the same solves as the run that was not stopped.

The factor check is the adjoint residual `||K lam - s|| / ||s||` against `buckling.RESIDUAL_FACT`. It tests the factor that the
gradient uses (the residual of U does not: it reads the U that the pass solved). A failure is "no usable result": the last value,
a zero gradient, and the row is unknown again. After `KNOWN_EVALUATIONS` failures in a row (a case with no compression never
converges) the row stops evaluating at every pass: it evaluates at a stage start, at the last pass and every `EVERY` passes, until
one evaluation gives a result. A pass with no result at the end of the run is "not judged" in the verdict (`formulation.no_result`).
It is a voxel estimate on the gray density, not a verdict.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import buckling as bk
from . import buckling_voxel as bv
from .config import BUCKLING_KS_Q, UserError
from .fe import isotropic_D, strain_matrix_centre
from .formulation import HELD, Constraint, Response, no_result, no_result_text
from .joints import condensed_border

SKIP_FACTOR = 1.5          # evaluate while the last lowest factor is at most this times min_factor (pipeline: BUCK_SKIP_FACTOR)
EVERY = 5                  # else every this many passes (pipeline: BUCK_EVERY)
KNOWN_EVALUATIONS = 2      # the row is known after this many evaluations with a result
KS_Q_DEFAULT = BUCKLING_KS_Q   # KS sharpness when `buckling.ks_q` is not set (the pipeline value, 40)
ROW_TOL = bk.TOL           # the eigen tolerance of the row (#155: a looser value is not decided yet). It is in the row state

FACTOR_MISMATCH = "factor_mismatch"
CHECK_SEED = 0             # the seed of the random vector v of the factor check after a failed eigen solve (b = K v, solve, residual)
SKIPPED = HELD
STATUSES = (bk.OK, bk.NO_POSITIVE, bk.NOT_CONVERGED, bk.TIME_BUDGET, FACTOR_MISMATCH, SKIPPED)

# The keys that the row adds to its entry in `history.jsonl` (`constraints[]`). An evaluated entry and a skipped entry have all of them.
INFO_KEYS = (
    "evaluated", "reason", "status", "min_factor", "ks_q", "modes",
    "lowest_factor", "last_eval", "load_factors", "near_repeated", "min_rel_gap", "cluster_cut", "ks_share_last",
    "ks_offset", "low_density_energy_share",
    "n_pass", "n_eval", "n_fail",
    "solves", "eigen_seconds", "kg_seconds", "adjoint_seconds",
    "residual_u", "residual_adjoint", "residual_max",
)
# The keys that describe the last evaluation with a result. A skipped entry repeats them (the end facts read the entry).
CARRIED = ("lowest_factor", "last_eval", "load_factors", "near_repeated", "min_rel_gap", "cluster_cut", "ks_share_last",
           "ks_offset", "low_density_energy_share")

BUCKLING_HINT = ("options: lower buckling.min_factor (a decision for the user: the tool does not change it), add material "
                 "(a larger design space or a higher volume_fraction), run more iterations, or use a finer voxel_size "
                 "(the voxel estimate reads high for thin members, CONTRACTS section 39)")
BUCKLING_CAUSE = "this buckling limit is not met by the design that MMA reached"


# --- the aggregate, on plain arrays ------------------------------------------


def ks_modes(mu, min_factor: float, q: float):
    """KS aggregate of `z_j = min_factor * mu_j`: `(S, p, w)`.

    `S = a + ln(sum exp(q (z - a))) / q` with `a = max z`; `p` the softmax weights (sum 1); `w = dS/dmu = min_factor * p`.
    `S >= max z` and `S <= max z + ln(k)/q`. The shift by `a` makes `exp` safe for any `z`.
    """
    z = float(min_factor) * np.asarray(mu, dtype=float)
    a = float(z.max())
    e = np.exp(float(q) * (z - a))
    total = float(e.sum())
    return a + math.log(total) / float(q), e / total, float(min_factor) * e / total


def k_normalise(phi, K) -> np.ndarray:
    """A copy of the columns of `phi` scaled so that `phi_j^T K phi_j = 1` (the gradient formula needs it; the pipeline does the same)."""
    phi = np.array(phi, dtype=float)
    for j in range(phi.shape[1]):
        phi[:, j] /= math.sqrt(float(phi[:, j] @ (K @ phi[:, j])))
    return phi


def evaluation_reason(state: dict, point, min_factor: float) -> str | None:
    """Why the row evaluates at this pass, or None (skip). The first reason found, in this order:

    `no_pass_number` (a point built by hand: `point.iteration` is None), `unknown` (fewer than `KNOWN_EVALUATIONS` evaluations
    with a result since the row was last unknown, and fewer than `KNOWN_EVALUATIONS` failures in a row), `stage_start` (the beta
    differs from the last pass), `last_pass`, `near` (the last lowest factor is at most `SKIP_FACTOR * min_factor`), `every`
    (`EVERY` passes since the last evaluation).
    After `KNOWN_EVALUATIONS` failures in a row (the back-off) the row has no known lowest factor and no `near` reason: it evaluates for
    `stage_start`, `last_pass`, and `every` (`EVERY` passes since the last evaluation, with a result or not), until one evaluation gives a result.
    """
    if point.iteration is None:
        return "no_pass_number"
    failing = state.get("streak", 0) >= KNOWN_EVALUATIONS
    if state["n_known"] < KNOWN_EVALUATIONS and not failing:
        return "unknown"
    if state["beta"] is None or point.beta != state["beta"]:
        return "stage_start"
    if point.last:
        return "last_pass"
    if failing:
        tried = state.get("last_try")
        return "every" if tried is None or point.iteration - tried >= EVERY else None
    low = state["last_lowest"]
    if low is not None and low <= SKIP_FACTOR * float(min_factor):
        return "near"
    if state["last_eval"] is None or point.iteration - state["last_eval"] >= EVERY:
        return "every"
    return None


# --- the gradient parts ------------------------------------------------------


class Geometry:
    """Constants of one model for the row: the table of all active voxels, D0, Bc and the six blocks `G_c` stacked for one product."""

    def __init__(self, model):
        cfg = model.problem.config
        self.model = model
        self.edof = model.edof                       # all active voxels (`Model.__init__` keeps it)
        self.h = float(model.problem.h)
        self.E, self.nu = float(model.E), float(cfg.material.nu)
        self.D0 = isotropic_D(self.E, self.nu)
        self.Bc = strain_matrix_centre(self.h)
        G = bv.hex8_kg_blocks(self.h)                # (6, 24, 24): Kg_e = sum_c sigma_c G_c
        self.Gs = np.ascontiguousarray(G.transpose(1, 0, 2).reshape(24, 6 * 24))   # one product gives all six: (pe @ Gs)[e, c, :]
        self.n_el = int(self.edof.shape[0])
        joints = model.problem.joints
        self.border = None if joints is None else condensed_border(joints, model.free_dofs)   # joints (section 38): S = K - C D^-1 C^T


def gradient_parts(model, geo: Geometry, rho_all, u_all, mu, phi, w):
    """The explicit part, the K part and the adjoint load `s` of `d(sum_j w_j mu_j)/d rho` (module text).

    `rho_all` (n_el): the density of all active voxels (solid voxels 1). `u_all`: the displacement of all grid dofs (one load case).
    `mu` (m,), `phi` (n_free, m) K-normalised, `w` (m,) the weights `dS/dmu_j`. The stress part is `+stiffness_derivative(lam, u)` with
    `K lam = s`: the caller solves for `lam`, so that a test can leave the part out.
    Returns `(explicit, kpart, s, e_first)`: `explicit` and `kpart` for the design voxels, `s` on the free dofs,
    `e_first` (n_el) = `phi_0^T ke phi_0` of the first mode (unit modulus) for the energy share.
    """
    p, edof, ke = model.penal, geo.edof, model.ke
    sig = bv.element_stress(u_all, edof, geo.E, geo.nu, geo.h)      # (n_el, 6), full modulus
    Q = np.zeros((geo.n_el, 6))
    kterm = np.zeros(geo.n_el)
    e_first = None
    pa = np.zeros(model.n_dof_all)
    for j in range(mu.size):
        pa[model.free_dofs] = phi[:, j]
        pe = pa[edof]                                               # (n_el, 24)
        Qj = np.einsum("ecj,ej->ec", (pe @ geo.Gs).reshape(-1, 6, 24), pe)
        ej = np.einsum("ei,ei->e", pe @ ke, pe)
        if j == 0:
            e_first = ej
        Q += w[j] * Qj
        kterm += w[j] * mu[j] * ej
    weight = rho_all ** p
    s_el = (weight[:, None] * Q) @ geo.D0 @ geo.Bc                  # Bc^T D0 (w_e Q_e): D0 is symmetric
    s = np.bincount(edof.ravel(), weights=s_el.ravel(), minlength=model.n_dof_all)[model.free_dofs]
    dpos = model.design_pos
    rho_d = rho_all[dpos]
    explicit = -p * rho_d ** (p - 1.0) * np.einsum("ec,ec->e", sig[dpos], Q[dpos])
    kpart = -model.dE_drho(rho_d) * kterm[dpos]
    return explicit, kpart, s, e_first


# --- texts -------------------------------------------------------------------


def refusal(solver_name: str, avail: dict | None = None) -> str:
    """The text of the refusal when the solver of the run has no direct factor of K (facts and options).

    The options come from `cli.backend_availability()` (`avail` replaces it in tests), as `buckling_voxel.skip_decision` does:
    a direct solver that is available here is an option; one that is not has its note (the install command on Linux x86_64, the
    shim build command on Apple silicon). The option "remove buckling.min_factor and use the check" appears only when a direct
    solver is available here, because the check skips without one.
    """
    if avail is None:
        from .cli import backend_availability
        avail = backend_availability()
    from . import solver_choice
    direct_here = [d for d in bv.DIRECT_SOLVERS if avail.get(d, {}).get("available")]
    options = []
    for d in bv.DIRECT_SOLVERS:
        info = avail.get(d, {})
        if info.get("available"):
            options.append(f"Set settings.solver: {d} in the YAML.")
        else:
            options.append(f"{d}: {info.get('note', 'not known')}")
    if "pardiso" not in direct_here and solver_choice.is_linux_x86_64():
        options.append(solver_choice.LICENSE_NOTE)
    if direct_here:
        options.append("Remove buckling.min_factor and use the check after the run (buckling.check: true): it gives the "
                       "load factors of the final design, and the optimizer does not hold them.")
    else:
        options.append("No direct solver is available here, so the check after the run (buckling.check) skips too.")
    return (f"buckling.min_factor needs a direct factor of K with a solve call at every evaluated pass. The solver of this run is "
            f"{solver_name}: it has no such factor (amg is an inexact solve, gpu-mg has no matrix), and the optimizer has no other "
            f"direct factor. The run is not started. options: " + " ".join(f"({i}) {o}" for i, o in enumerate(options, 1)))


def model_refusal(model) -> str | None:
    """The refusal text for a built model that the row cannot use (a solver with no direct factor of K), or None."""
    if getattr(model, "solver_name", None) not in bv.DIRECT_SOLVERS:
        return refusal(str(getattr(model, "solver_name", None)))
    return None


def config_refusal(config, avail: dict | None = None) -> str | None:
    """The refusal text for a YAML whose buckling row cannot run with the solver of the run, or None (also None without the row).

    A YAML with `joints` is not refused: the row uses the condensed stiffness S (CONTRACTS.md section 38). The setting string `gpu-mg` is refused at once (`cli.backend_problem` fails without OpenCL, and it would speak first).
    Another setting is resolved as `settings.solver` is at the start of a run: `amg` (also what `auto` gives without PARDISO or the
    Accelerate shim) is refused. A platform that cannot run the named solver is not a refusal of the row: `backend_problem` has the message.
    """
    if config.buckling.min_factor is None:
        return None
    from .solver_choice import resolve_solver
    setting = config.settings.solver
    if setting == "gpu-mg":
        return refusal("gpu-mg", avail)
    try:
        solver = resolve_solver(setting)
    except UserError:
        return None
    return None if solver in bv.DIRECT_SOLVERS else refusal(solver, avail)


def evaluation_range(iterations: int, schedule, n_cases: int) -> tuple[int, int]:
    """`(low, high)`: the number of evaluations (one eigen solve for one load case) in a run of `iterations` iterations (passes 0 to `iterations`).

    The high count is every pass for every case. The low count is the skip rule when the row stays far from its limit: the passes that
    always run (the first `KNOWN_EVALUATIONS`, each start of a beta stage in `schedule`, the last pass) and, between two of them, one
    evaluation for each `EVERY` passes. The `near` reason adds evaluations, so a run is between the two counts.
    """
    last = int(iterations)
    forced = sorted({p for p in (*range(KNOWN_EVALUATIONS), *(int(i) for i, _ in schedule), last) if p <= last})
    low = len(forced) + sum((b - a - 1) // EVERY for a, b in zip(forced, forced[1:]))
    return n_cases * low, n_cases * (last + 1)


def time_facts(cfg) -> str:
    """Facts about the time of the row, for `topopt check`: the count of evaluations, the measured cost on the hook, the worst case."""
    low, high = evaluation_range(cfg.settings.iterations, cfg.settings.beta_schedule, len(cfg.load_cases))
    return (f"The time estimate of the run does not include the row. The row evaluates {low} to {high} times in this run (one eigen solve "
            f"for one load case; {len(cfg.load_cases)} load cases, {cfg.settings.iterations + 1} passes: the low count is the skip rule when the "
            f"row stays far from its limit, the high count is every pass; passes 0 and 1 always run, CONTRACTS section 39). One evaluation took "
            f"9.3 s for `hang` and 12.2 s for `pull_off` on the hook at 1.2 mm (115,836 free dofs; one machine, 4 threads, E3), which is 5.5 "
            f"to 7.2 times the median pass without the row. The time of another model is not estimated. An evaluation that fails costs up "
            f"to buckling.max_seconds ({cfg.buckling.max_seconds:g} s) for each load case.")


def preflight(cfg, solver, grid_voxels: int, free_dofs: int, ram_bytes=None, run_bytes=None, avail=None) -> dict:
    """What `topopt check` and the start of a run say about the row: will it run, the memory, and facts with options.

    `solver` is the resolved solver name, or None when this platform cannot run the solver of the YAML. `run_bytes` is the memory
    estimate of the run (with the filter) that the caller prints; the default is the estimate without the filter.
    The row adds K (a second copy, assembled by the design point), `Kg` and the Lanczos vectors: the estimate of the A6a check
    (`buckling_voxel.estimate_bytes`) for `modes + 1` pairs, without its run part. It is a fact: the run goes on above the budget.
    Return a dict with `will_run`, `reason`, `memory_extra_bytes`, `memory_estimate_bytes`, `budget_bytes`, `evaluations_low`,
    `evaluations_high` (`evaluation_range`), `facts`, `options` and `warnings` (text lines).
    """
    from .cli import estimate_memory_bytes
    b = cfg.buckling
    base = estimate_memory_bytes(int(grid_voxels), int(free_dofs))
    extra = bv.estimate_bytes(int(grid_voxels), int(free_dofs), b.modes + 1) - base
    total = (base if run_bytes is None else int(run_bytes)) + extra
    budget = bv.memory_budget(b.max_gb) if ram_bytes is None else (int(b.max_gb * 1e9) if b.max_gb is not None else ram_bytes)
    out = {"min_factor": b.min_factor, "ks_q": BUCKLING_KS_Q if b.ks_q is None else b.ks_q, "modes": b.modes,
           "max_seconds": b.max_seconds, "solver": solver, "will_run": True, "reason": None,
           "memory_extra_bytes": int(extra), "memory_estimate_bytes": int(total), "budget_bytes": budget,
           "evaluations_low": evaluation_range(cfg.settings.iterations, cfg.settings.beta_schedule, len(cfg.load_cases))[0],
           "evaluations_high": evaluation_range(cfg.settings.iterations, cfg.settings.beta_schedule, len(cfg.load_cases))[1],
           "facts": [time_facts(cfg)],
           "options": [], "warnings": []}
    if cfg.has_joints():        # joints (section 38): the row runs with S; the beams have no Kg
        from .joints import BUCKLING_BEAMS_FACT
        out["facts"].append(BUCKLING_BEAMS_FACT)
    if solver is None and cfg.settings.solver == "gpu-mg":      # the run speaks with the refusal text: the check says the same
        out.update(will_run=False, reason=out["reason"] or "solver_not_direct")
        out["warnings"].append("buckling row: " + refusal("gpu-mg", avail))
    elif solver is None:
        out.update(will_run=False, reason=out["reason"] or "solver_unavailable")
        out["warnings"].append("buckling row: it is on in the YAML, and the solver of this run cannot start here, so the row cannot run "
                               "either. Set settings.solver in the YAML to a solver that runs on this machine (a direct one: "
                               + ", ".join(bv.DIRECT_SOLVERS) + ").")
    elif solver not in bv.DIRECT_SOLVERS:
        out.update(will_run=False, reason=out["reason"] or "solver_not_direct")
        out["warnings"].append("buckling row: " + refusal(solver, avail))
    if budget is not None and total > budget:
        src = f"buckling.max_gb = {b.max_gb:g}" if b.max_gb is not None else "the RAM of this machine"
        out["warnings"].append(
            f"buckling row: the memory estimate with the row is {bv.gb(total)} (the run {bv.gb(total - extra)} and the row {bv.gb(extra)}), "
            f"above the budget of {bv.gb(budget)} ({src}; GB = 1e9 bytes). The run goes on. options: raise buckling.max_gb if this "
            f"machine has the memory; use a larger voxel_size (this needs a new run); or remove buckling.min_factor")
    return out


def lines(info: dict) -> list[str]:
    """The text lines of `preflight` for `topopt check` and the start of a run (the warnings are separate)."""
    return [f"buckling row: on in the YAML (min_factor {info['min_factor']:g}, KS q {info['ks_q']:g}, {info['modes']} modes): memory estimate "
            f"about {bv.gb(info['memory_estimate_bytes'])} with the row (the row adds about {bv.gb(info['memory_extra_bytes'])}; GB = 1e9 bytes). "
            + " ".join(info["facts"])]


def _detail(entry) -> str:
    low = entry.get("lowest_factor")
    now = f"lowest factor {low:.5g}" if low is not None else "no positive factor"
    return (f"{now} (pass {entry.get('last_eval')}) against buckling.min_factor {entry['min_factor']:g}; KS row value "
            f"{entry['value']:.5g}, KS offset up to {entry['ks_offset']:.3g} over {entry['modes']} modes (voxel estimate "
            f"on the gray density of the optimizer)" if entry.get("ks_offset") is not None else
            f"{now} against buckling.min_factor {entry['min_factor']:g} (voxel estimate on the gray density)")


NO_COMPRESSION = ("The case may have no compression: a case with no compressive stress has no buckling load, and the eigen solve "
                  "then does not converge in the limit of its iterations (raising buckling.max_seconds does not help). The row is "
                  "on for every load case: there is no key to choose the cases. options: remove buckling.min_factor, or look at "
                  "the load case")
OUT_OF_TIME = ("options: raise buckling.max_seconds (a pass that fails costs up to that time for each case), ask for fewer "
               "buckling.modes, or remove buckling.min_factor")
FAULT = "This is a fault of the tool, not of the design. options: remove buckling.min_factor, and report it with the YAML"


def _remedy(status) -> str:
    """The text after a pass with no result, by status. None (the status of the last failure is not known): all three."""
    if status == FACTOR_MISMATCH:
        return FAULT
    if status == bk.NOT_CONVERGED:
        return NO_COMPRESSION
    if status == bk.TIME_BUDGET:
        return OUT_OF_TIME
    return f"If the status is {bk.NOT_CONVERGED}: {NO_COMPRESSION}. If the status is {bk.TIME_BUDGET}: {OUT_OF_TIME}. If it is {FACTOR_MISMATCH}: {FAULT}"


def _facts(entry) -> list[str]:
    """Lines at the end of the run, any state of the row. They read the history entry (the counters are in it)."""
    name, lines = entry["name"], []
    low, n_pass, n_eval, n_fail = entry.get("lowest_factor"), entry["n_pass"], entry["n_eval"], entry["n_fail"]
    if entry.get("last_eval") is None:
        head = f"fact: buckling row {name}: no evaluation gave a result"
    elif low is None:
        head = f"fact: buckling row {name}: no positive buckling factor at pass {entry['last_eval']}"
    else:
        head = f"fact: buckling row {name}: lowest linear buckling factor {low:.5g} at pass {entry['last_eval']}"
    lines.append(f"{head} (min_factor {entry['min_factor']:g}, row value {entry['value']:.5g}, g {entry['g']:+.4f}); "
                 f"{n_eval} of {n_pass} passes were evaluated, {n_fail} without a result. A voxel estimate on the gray density of "
                 f"the optimizer: it reads high for thin members (CONTRACTS section 39); buckling.check tests the printed design")
    if no_result(entry):
        lines.append(f"fact: {name}: {no_result_text(entry)}. The verdict does not judge the row: it counts it as not met, so "
                     f"`feasible` is false. {_remedy(entry['status'])}")
    elif not entry["evaluated"] and entry["reason"] == SKIPPED:
        lines.append(f"fact: {name}: the last pass was not evaluated (a stop rule ended the run before the skip rule could "
                     f"know); the value is from pass {entry.get('last_eval')}")
    share = entry.get("low_density_energy_share")
    if share is not None and share >= bv.FALSE_MODE_SHARE:
        lines.append(f"fact: {name}: {100 * share:.1f} percent of the strain energy of the lowest mode is in voxels with a density "
                     f"below 0.5 (gray or void voxels). It can be a false mode of a thin or loose gray piece. options: look at "
                     f"the design near the mode; `{bk.RERUN} --density printed` shows the factors of the printed design. "
                     f"(The share is about 1 in the first passes of every run: read it for the final design)")
    if entry.get("cluster_cut"):
        lines.append(f"fact: {name}: the factors of mode {entry['modes']} and the next mode are within the near-repeated level "
                     f"({bk.NEAR_REPEATED_REL:g} relative), so the KS sum may hold only part of a cluster. ks_share_last "
                     f"{entry['ks_share_last']:.3g}, row g {entry['g']:+.4f}. The gradient error of a cut cluster was 0.23 to 1.8 "
                     f"times this share on a square column: it depends on the basis that ARPACK picks inside the cluster. A share near "
                     f"0 means that the cut mode has no weight in the sum. The flag sees only the modes that ARPACK returned. "
                     f"options: raise buckling.modes")
    if n_fail and not no_result(entry):      # a row with no result at the end has the status and the options in its own fact
        lines.append(f"fact: {name}: {n_fail} evaluations gave no usable result (not converged, the time budget buckling.max_seconds, or "
                     f"a factor that does not match the matrix). Each such pass keeps the last value and a zero gradient. "
                     f"{_remedy(None)}")
    return lines


# --- the row -----------------------------------------------------------------


def _start_state(min_factor, ks_q, modes, tol=None) -> dict:
    return {"min_factor": float(min_factor), "ks_q": float(ks_q), "modes": int(modes), "tol": float(ROW_TOL if tol is None else tol),
            "n_known": 0, "last_lowest": None,
            "last_value": None, "last_eval": None, "beta": None, "n_pass": 0, "n_eval": 0, "n_fail": 0, "last_info": None,
            "streak": 0, "last_try": None, "fail_status": None, "logged": False, "v0": None}


SETTINGS_KEYS = ("min_factor", "ks_q", "modes", "tol")      # a saved state of other values for these makes the row unknown (and drops `v0`)


class BucklingRow:
    """The response function of the row `buckling:<case>`: `DesignPoint -> Response` (see the module text).

    `skip=False` evaluates at every pass (for scripts and tests). `log` (a function `text -> None`, the log of the run) gets one fact
    line at the first failed evaluation (the state keeps that it was said, so a resume does not repeat it). The object keeps its state in `get_state` / `set_state`.
    """

    def __init__(self, case, min_factor, q=KS_Q_DEFAULT, modes=bk.DEFAULT_MODES, max_seconds=bk.DEFAULT_MAX_SECONDS,
                 skip=True, log=None, name=None):
        if not (np.isfinite(min_factor) and min_factor > 0):
            raise ValueError(f"buckling min_factor must be a number above 0, got {min_factor!r}")
        if not (np.isfinite(q) and q > 0):
            raise ValueError(f"the KS sharpness q must be a number above 0, got {q!r}")
        if int(modes) < 1:
            raise ValueError(f"buckling modes must be 1 or more, got {modes!r}")
        self.case, self.min_factor, self.q, self.modes = case, float(min_factor), float(q), int(modes)
        self.max_seconds, self.skip, self.log = max_seconds, bool(skip), log
        self.tol = float(ROW_TOL)                    # the one place that the row reads its eigen tolerance from
        self.name = name or f"buckling:{case}"
        self._geo = None
        self._checked = False
        self.st = _start_state(self.min_factor, self.q, self.modes, self.tol)

    # -- state (MmaState.rows)
    def get_state(self) -> dict:
        """A copy of the state. The start vector `v0` (an array, or None) is shared, not copied: the row never changes it in place."""
        st = dict(self.st)
        st["last_info"] = None if st["last_info"] is None else dict(st["last_info"])
        return st

    def set_state(self, state) -> None:
        """Set the state from `get_state()`. None: the start state of a new run. A state of other settings makes the row unknown
        and drops the start vector (`SETTINGS_KEYS`)."""
        fresh = _start_state(self.min_factor, self.q, self.modes, self.tol)
        if state is None:
            self.st = fresh
            return
        st = dict(state)
        st["last_info"] = None if state.get("last_info") is None else dict(state["last_info"])
        for key in ("streak", "last_try", "fail_status", "logged", "v0"):      # a state of an earlier run of this branch has no such keys
            st.setdefault(key, fresh[key])
        st.setdefault("tol", bk.TOL)       # a state from before #155 has no `tol`: its solves ran at bk.TOL (so a later ROW_TOL is a change)
        if any(st[key] != fresh[key] for key in SETTINGS_KEYS):
            st.update({key: fresh[key] for key in SETTINGS_KEYS}, n_known=0, streak=0, last_try=None, fail_status=None, v0=None)
        self.st = st

    # -- one pass
    def __call__(self, point) -> Response:
        model = point.model
        self._check_solver(model)
        st = self.st
        st["n_pass"] += 1
        ci = point.case_index(self.case)
        reason = evaluation_reason(st, point, self.min_factor) if self.skip else "no_skip"
        if reason is None:
            st["beta"] = point.beta
            return Response(self._last_value(), np.zeros(point.n_design), wrt="rho_stiff",
                            info=self._info(False, SKIPPED, st["fail_status"] or SKIPPED))   # in a back-off: no value is known
        return self._evaluate(point, ci, reason)

    def _last_value(self) -> float:
        return 0.0 if self.st["last_value"] is None else float(self.st["last_value"])

    def _start_vector(self, n: int):
        """The kept start vector for an eigen problem of size `n`, or None (a cold start: `buckling_modes` uses its fixed random vector).

        None when no vector is kept, or when the kept one has another length (a different model), is not finite, or is zero. Such a vector is dropped.
        """
        v = self.st.get("v0")
        if v is None:
            return None
        v = np.asarray(v, dtype=float)
        if v.ndim != 1 or v.shape[0] != n or not np.all(np.isfinite(v)) or not v.any():
            self.st["v0"] = None
            return None
        return v

    def _check_solver(self, model):
        if not self._checked:
            text = model_refusal(model)
            if text:
                raise UserError(text)
            self._checked = True

    def _evaluate(self, point, ci, reason) -> Response:
        model, st = point.model, self.st
        geo = self._geo if self._geo is not None and self._geo.model is model else Geometry(model)
        self._geo = geo
        t0 = time.perf_counter()
        K = point.stiffness_matrix()
        if geo.border is not None:      # joints (section 38): the solver of the run inverts S, so S is the stiffness of the eigen problem
            K = geo.border.condense(K)
        u, f = point.U[:, ci], model.f_red[:, ci]
        res_u = float(np.linalg.norm(K @ u - f) / (np.linalg.norm(f) or 1.0))
        u_all = np.zeros(model.n_dof_all)
        u_all[model.free_dofs] = u
        rho_all = np.ones(geo.n_el)
        rho_all[model.design_pos] = point.rho_stiff
        Kg = bv.hex8_kg(model.assembler, geo.edof, u_all, geo.h, geo.E, geo.nu, rho_all ** model.penal)
        t_kg = time.perf_counter() - t0
        res = bk.buckling_modes(Kg, K, model.solver, k=self.modes + 1, tol=self.tol, maxiter=bk.MAXITER,
                                max_seconds=self.max_seconds, v0=self._start_vector(K.shape[0]))
        cost = {"solves": int(res["solves"]), "eigen_seconds": float(res["seconds"]), "kg_seconds": t_kg,
                "adjoint_seconds": 0.0, "residual_u": res_u, "residual_adjoint": None, "residual_max": None}
        if res["status"] != bk.OK or res["mu"].size == 0:
            # A wrong factor can stop the eigen solve before the adjoint residual below shows it: one solve with the factor tells.
            # The right-hand side is K v for a fixed random v, not the load: the residual of the load is `residual_u`, which is high on
            # a valid near-singular design (1e-6 and more), while K v gives the backward error of the factor (1e-16 with the right one).
            t1 = time.perf_counter()
            b = K @ np.random.default_rng(CHECK_SEED).standard_normal(K.shape[0])
            res_check = float(np.linalg.norm(K @ point.solve_adjoint(b) - b) / (np.linalg.norm(b) or 1.0))
            cost.update(adjoint_seconds=time.perf_counter() - t1, residual_adjoint=res_check)
            if not res_check <= bk.RESIDUAL_FACT:
                return self._failure(point, reason, FACTOR_MISMATCH, cost, check=True)
            return self._failure(point, reason, bk.NOT_CONVERGED if res["status"] == bk.OK else res["status"], cost)
        k_used = min(self.modes, int(res["mu"].size))
        mu, lam_all = res["mu"], res["lam"]
        phi_all = k_normalise(res["phi"], K)                   # the formula needs phi^T K phi = 1. All returned columns: the start vector sums them
        phi = phi_all[:, :k_used]
        mu = mu[:k_used]
        S, p, w = ks_modes(mu, self.min_factor, self.q)
        t1 = time.perf_counter()
        explicit, kpart, s, e_first = gradient_parts(model, geo, rho_all, u_all, mu, phi, w)
        lam = point.solve_adjoint(s)
        res_adj = float(np.linalg.norm(K @ lam - s) / (np.linalg.norm(s) or 1.0))
        cost.update(adjoint_seconds=time.perf_counter() - t1, residual_adjoint=res_adj, residual_max=float(res["residual"].max()))
        if not res_adj <= bk.RESIDUAL_FACT:                    # the factor of `model.solver` is not the factor of this K
            return self._failure(point, reason, FACTOR_MISMATCH, cost)
        grad = explicit + kpart + point.stiffness_derivative(lam, u)
        finite = lam_all[np.isfinite(lam_all)]
        gap_flag, min_gap, _pairs = bk.near_repeated(finite)
        shown = lam_all[:k_used]
        shown = shown[np.isfinite(shown)]
        cut = bool(lam_all.size > k_used and np.isfinite(lam_all[k_used]) and np.isfinite(lam_all[k_used - 1])
                   and (lam_all[k_used] - lam_all[k_used - 1]) / lam_all[k_used - 1] < bk.NEAR_REPEATED_REL)
        en = model.element_E(point.rho_stiff) * e_first        # strain energy of the first mode in each voxel, with E_min
        total = float(en.sum())
        share = float(en[rho_all < bv.THRESHOLD].sum() / total) if total > 0 else 0.0
        carried = {"lowest_factor": float(shown[0]) if shown.size else None,
                   "last_eval": point.iteration, "load_factors": [float(v) for v in shown],
                   "near_repeated": bool(gap_flag), "min_rel_gap": None if min_gap is None else float(min_gap),
                   "cluster_cut": cut, "ks_share_last": float(p[k_used - 1]), "ks_offset": math.log(k_used) / self.q,
                   "low_density_energy_share": share}
        v0 = np.zeros(phi_all.shape[0])        # the start vector of the next evaluation: the sum of the K-normalised columns, added in order
        for j in range(phi_all.shape[1]):
            v0 += phi_all[:, j]
        st.update(n_known=min(st["n_known"] + 1, KNOWN_EVALUATIONS), last_lowest=carried["lowest_factor"], last_value=float(S),
                  last_eval=point.iteration, beta=point.beta, n_eval=st["n_eval"] + 1, last_info=carried,
                  streak=0, last_try=point.iteration, fail_status=None, v0=v0)
        status = bk.OK if shown.size else bk.NO_POSITIVE
        return Response(float(S), grad, wrt="rho_stiff", info=self._info(True, reason, status, cost))

    def _failure(self, point, reason, status, cost, check=False) -> Response:
        st = self.st
        st.update(n_known=0, n_fail=st["n_fail"] + 1, beta=point.beta, streak=st["streak"] + 1, last_try=point.iteration,
                  fail_status=status)
        if self.log is not None and not st["logged"]:        # the flag is in the state: a resume does not say it again
            st["logged"] = True
            self.log(self._first_failure_text(point, status, cost, check))
        return Response(self._last_value(), np.zeros(point.n_design), wrt="rho_stiff", info=self._info(False, reason, status, cost))

    def _first_failure_text(self, point, status, cost, check=False) -> str:
        where = f"at pass {point.iteration}" if point.iteration is not None else "at this pass"
        head = (f"fact: {self.name}: no usable buckling result {where} (status {status}, {cost['eigen_seconds']:.1f} s of eigen solve). "
                f"The row keeps its last value with a zero gradient and is unknown again, so the next passes evaluate. After "
                f"{KNOWN_EVALUATIONS} failures in a row it evaluates only at a stage start, at the last pass and every {EVERY} passes, "
                "until one evaluation gives a result. ")
        if status == FACTOR_MISMATCH:
            what = "The residual of a check solve with the factor after the failed eigen solve" if check else "The adjoint residual"
            return (head + f"{what} is {cost['residual_adjoint']:.3g}, above {bk.RESIDUAL_FACT:g}: the factor of K that the "
                    f"run holds does not match the matrix of this pass. {FAULT}")
        return head + _remedy(status)

    def _info(self, evaluated, reason, status, cost=None) -> dict:
        st = self.st
        info = {k: None for k in INFO_KEYS}
        info.update(evaluated=bool(evaluated), reason=reason, status=status, min_factor=self.min_factor, ks_q=self.q, modes=self.modes,
                    n_pass=st["n_pass"], n_eval=st["n_eval"], n_fail=st["n_fail"],
                    solves=0, eigen_seconds=0.0, kg_seconds=0.0, adjoint_seconds=0.0, load_factors=[])
        if st["last_info"] is not None:
            info.update({k: st["last_info"][k] for k in CARRIED})
        if cost is not None:
            info.update(cost)
        return info


def buckling_ks_constraint(case, min_factor, q=KS_Q_DEFAULT, modes=bk.DEFAULT_MODES, max_seconds=bk.DEFAULT_MAX_SECONDS,
                           name=None, skip=True, log=None) -> Constraint:
    """The buckling row of one load case: `KS_q(min_factor * mu_j) - 1 <= 0` over the `modes` lowest factors.

    `case` is a load case name or index. The row has no step-acceptance `measure` (it would cost one eigen solve for each trial).
    It is `stage_aware`: the run log and the history record a beta stage that ends with the row not met.
    The history entry of the row has the extra keys `INFO_KEYS`.
    """
    row = BucklingRow(case, min_factor, q, modes, max_seconds, skip, log, name)
    return Constraint(row.name, row, limit=1.0, scale=1.0, hint=BUCKLING_HINT, cause=BUCKLING_CAUSE, stage_aware=True,
                      detail=_detail, facts=_facts)
