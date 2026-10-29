"""Robust erosion projection, low-cost form (CONTRACTS.md section 22b).

`settings.robust: eroded` projects the one filtered field `rho_tilde` twice with the Heaviside
function of `filter.py`:

    nominal  rho_n = P(rho_tilde; beta, eta_n)   eta_n = 0.5 (`optimize.ETA`)
    eroded   rho_e = P(rho_tilde; beta, eta_e)   eta_e > eta_n

The volume, the snapshots and the final mesh use the nominal design. The stiffness and so the
compliance use the eroded design. A member that is too thin for the filter radius does not reach
the threshold eta_e. It has no stiffness in the eroded design, so the compliance gives it no use.
The optimizer drops it, or makes it thicker.

This is the form of the pipeline (`pp_core_final.py`: `densities` at lines 341-347, the mass on the
nominal design and K on the eroded design at lines 495-503, the module text at lines 1-8): one eroded
and one nominal design, with the volume on the nominal design. The form in the literature, the
low-cost form of Wang, Lazarov and Sigmund (2011), puts the volume on the dilated design (to our
knowledge). There is no dilated design here. Each pass has one solve (the eroded design). One more
solve at the end gives the nominal compliance of the final design. The pipeline also runs a second
factorisation for the nominal stress rows. The demo has no stress row, so it has no such cost.

Which numbers are which design (`post.finish` and the run log follow this):
the history curve, the compliance limits and the `unmet` lines are the eroded design (the optimizer
minimises it); `compliance_final` and `stiffness_cases` of `metrics.json` are the nominal design, the
part that the user prints. The eroded values are in the `robust` block of `metrics.json`.

This module has no import from the rest of the package at load time, so `config.py` can use it.
It loads numpy and the projection only inside the functions that need them: `topopt.cli` must not
load numpy at import. Every function returns facts (numbers and text). None of them stops a run: a
solver that has no parity test is a warning fact with options (`solver_fact`).
"""

from __future__ import annotations

import math
import time

ETA_NOMINAL = 0.5          # the threshold of the nominal design (the constant `optimize.ETA`)
ETA_ERODED_DEFAULT = 0.55  # eta_e of `settings.robust: eroded` when `settings.eta_eroded` is not set. Calibration: the plan document, section 6.
WALL_ETA_DEFAULT = 0.7     # eta_e of `settings.robust: wall` when `settings.eta_eroded` is not set (CONTRACTS.md section 22b)
MODES = ("off", "eroded", "wall")
# The solvers that have a parity test (tests/test_robust.py: the compliance of a robust fixture equals
# the direct solve of the same eroded system by `scipy.sparse.linalg.spsolve`; gpu-mg: the same in
# tests/test_gpu_mg_robust.py, which needs an OpenCL device). The loop passes the eroded density to
# `Model.stiffness`, so any model that builds its element stiffness from the density it is given works by
# construction. A solver that is not in this list gets a warning fact (`solver_fact`) and the run goes on.
# The PR of a new solver adds its name and its parity test.
ROBUST_SOLVERS = ("cholesky", "pardiso", "amg", "gpu-mg")
MIN_MEMBER_VOXELS = 2.0    # a member needs 2 voxels or more to carry load (the same number as members.py)
EROSION_VISIBLE = 0.4      # the eroded density at rho_tilde 0.5 must be below this to have a visible effect

# Time. Measured on the hook at 1.2 mm, 1 thread, PARDISO (plan document, section 7):
# the time of a pass is the same in both modes within the noise of the shared host (the ratio
# of the summed pass times was 0.98 and 0.99 in two pairs of runs), and the final nominal solve
# is one more solve. `time_factor` is the estimate of the total.
PASS_TIME_FACTOR = 1.0     # time of a robust pass over the time of a default pass (measured)
EXTRA_SOLVES = 1           # the nominal solve at the end

NOTE = ("volume, snapshots and final mesh use the nominal design; stiffness, compliance and "
        "compliance limits use the eroded design")


# --- the parameters and the second projection ---------------------------------------------------


class Robust:
    """The two thresholds of a run, and the time spent in the work that only robust mode does."""

    def __init__(self, eta_n: float, eta_e: float, source: str = "default", mode: str = "eroded"):
        self.eta_n, self.eta_e, self.source, self.mode = float(eta_n), float(eta_e), source, mode
        self.projection_s = 0.0    # second projection and its slope (all passes)
        self.solve_s = 0.0         # the solve of the nominal design at the end

    @property
    def seconds(self) -> float:
        """Seconds spent in the robust part: the second projection, its slope, the nominal solve."""
        return self.projection_s + self.solve_s

    def eroded(self, rho_tilde, beta):
        """The eroded density: the filtered density projected at eta_e."""
        from .filter import heaviside_project
        t0 = time.perf_counter()
        rho = heaviside_project(rho_tilde, beta, self.eta_e)
        self.projection_s += time.perf_counter() - t0
        return rho

    def eroded_slope(self, rho_tilde, beta):
        """d(eroded density)/d(rho_tilde)."""
        from .filter import heaviside_deriv
        t0 = time.perf_counter()
        d = heaviside_deriv(rho_tilde, beta, self.eta_e)
        self.projection_s += time.perf_counter() - t0
        return d


def eta_eroded_of(st, mode: str | None = None) -> tuple[float, str]:
    """The threshold eta_e of a `Settings` in robust mode `mode` (default: `st.robust`) and where it came from: `settings`
    (settings.eta_eroded), `wall_rule` (the default 0.7 of `robust: wall`) or `default` (the default 0.55 of `robust: eroded`)."""
    if st.eta_eroded is not None:
        return float(st.eta_eroded), "settings"
    if (mode or st.robust) == "wall":
        return WALL_ETA_DEFAULT, "wall_rule"
    return ETA_ERODED_DEFAULT, "default"


def solver_fact(solver: str | None, mode: str = "eroded") -> str | None:
    """The warning fact for a solver that has no parity test in robust mode, or None.

    It is a fact, not a refusal: by construction the step works for every solver that builds its
    stiffness from the density it is given.
    """
    if solver is None or solver in ROBUST_SOLVERS:
        return None
    return (f"settings.robust is {mode}, and the solver {solver!r} has no parity test for it. The parity test "
            f"covers these solvers: {', '.join(ROBUST_SOLVERS)}. For {solver!r} the step works by construction "
            f"(the loop gives the eroded density to the stiffness), but no test shows that its compliance equals "
            f"a direct solve. Options: use a tested solver (settings.solver: {', '.join(ROBUST_SOLVERS)}); run "
            f"topopt verify after the run, when your version has it, to compare the printed part with the "
            f"optimizer; or add {solver!r} to robust.ROBUST_SOLVERS together with a parity test "
            f"(tests/test_robust.py)")


def stress_stop_text(n_rows: int, mode: str = "eroded") -> str:
    """The text for robust mode with stress rows (CONTRACTS.md section 22b, PR #22). One text for the stop and for `check`."""
    return (f"settings.robust is {mode}, and {n_rows} load case(s) have a stress limit. Robust mode does not "
            f"support a stress row yet: the row reads the nominal density with the displacements of the eroded "
            f"design, so its gradient mixes two densities and is wrong, with no error (measured: 4e-3 to 1e-1 "
            f"relative error against 1e-5 for a right gradient). Options: set settings.robust to off; or remove the "
            f"stress limits (stress_limit, allowable_mpa) of the load cases. Two ways to support it are not built: "
            f"(a) the stress of the eroded design, one solve for each pass; (b) a second solve of the nominal design "
            f"for the stress rows, about twice the time for each pass")


def buckling_stop_text(mode: str = "eroded") -> str:
    """The text for robust mode with the buckling row (CONTRACTS.md section 39). One text for the stop and for `check`."""
    return (f"settings.robust is {mode}, and buckling.min_factor is set. The buckling row is not built for robust mode: it would hold "
            "the load factor of the eroded design (the density of the stiffness), not of the nominal design that is printed, and no "
            "test or measurement covers this. Options: set settings.robust to off; or remove buckling.min_factor and use the check "
            "after the run (buckling.check: true, density printed), which tests the nominal printed design")


def from_settings(st, stress_rows: int = 0, buckling_row: bool = False):
    """The `Robust` object of a `Settings`, or None when robust mode is off.

    Raises `UserError` for robust mode with `stress_rows` > 0 (the number of load cases with a stress limit):
    the step cannot work, because the gradient of a stress row is wrong in robust mode. It also raises with
    `buckling_row` (the buckling row, section 39, is on).
    """
    if st.robust == "off":
        return None
    if stress_rows:
        from .config import UserError
        raise UserError(stress_stop_text(stress_rows, st.robust) + ". The run stops at the start")
    if buckling_row:
        from .config import UserError
        raise UserError(buckling_stop_text(st.robust) + ". The run stops at the start")
    eta, source = eta_eroded_of(st)
    return Robust(ETA_NOMINAL, eta, source, st.robust)


# --- numbers about the projection (pure Python) -------------------------------------------------


def projected(value: float, beta: float, eta: float) -> float:
    """The Heaviside projection of one number (the formula of `filter.heaviside_project`)."""
    a, b = math.tanh(beta * eta), math.tanh(beta * (1.0 - eta))
    return (a + math.tanh(beta * (value - eta))) / (a + b)


def plate_limit_per_radius(eta: float) -> float:
    """Thickness of the thinnest wall, in filter radii, whose centre reaches the filtered value `eta`.

    Continuum value for the hat filter w = R - d (infinite plate). With u = t / (2 R) the centre
    value is 2u - 2u^3 + u^4. It is 0.5328 for eta 0.5 (the rule of `members.py`), 0.5948 for 0.55.
    A thinner wall stays below `eta` and has no stiffness in the design that is projected at `eta`.
    """
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 2 * mid - 2 * mid ** 3 + mid ** 4 < eta:
            lo = mid
        else:
            hi = mid
    return 2.0 * 0.5 * (lo + hi)


# --- facts, estimate, text ----------------------------------------------------------------------


def time_factor(iterations: int) -> float:
    """Estimate of the run time in robust mode over the default run: the time of a pass is the same
    (`PASS_TIME_FACTOR`, measured), and the nominal solve at the end is `EXTRA_SOLVES` more solves."""
    n = int(iterations) + 1
    return (PASS_TIME_FACTOR * n + EXTRA_SOLVES) / n


def report(st, h: float, solver: str | None = None, stress_rows: int = 0, buckling_row: bool = False,
           print_wall_mm: float | None = None) -> dict:
    """The facts about robust mode for a `Settings`. The dict is `derived.robust` of `check` and `run.json`.

    `solver` is the solver that runs (`solver_choice.resolve_solver`), or None if it is not known.
    `stress_rows` is the number of load cases with a stress limit: a fact (the run stops at the start).
    `buckling_row` is true when `buckling.min_factor` is set: a fact too (the run stops at the start).
    `print_wall_mm` is `print.min_wall_mm` when the YAML gives it: with `robust` not `wall` it is a fact (`wall_fact`).
    `facts` are things that the author may want to change, each with options. None of them stops a run.
    """
    on = st.robust != "off"
    d: dict = {"mode": st.robust, "on": on, "facts": []}
    if st.robust != "wall" and print_wall_mm is not None:
        d["facts"].append(wall_fact(st, h, float(print_wall_mm)))
    if not on:
        if st.eta_eroded is not None:
            d["facts"].append(
                f"settings.eta_eroded {st.eta_eroded:g} is set, and settings.robust is off, so it is not "
                f"used. Options: set settings.robust to eroded or wall, or remove settings.eta_eroded.")
        return d
    h, R = float(h), float(st.filter_radius)
    eta_e, eta_source = eta_eroded_of(st)
    limit = plate_limit_per_radius(eta_e) * R
    beta_max = max(b for _, b in st.beta_schedule)
    at_half = projected(0.5, beta_max, eta_e)
    d.update({
        "eta_n": ETA_NOMINAL, "eta_e": eta_e, "eta_e_source": eta_source,
        "solves_per_step": 1, "extra_solves": EXTRA_SOLVES, "note": NOTE,
        "wall_limit_mm": round(limit, 3), "wall_limit_voxels": round(limit / h, 2),
        "wall_limit_nominal_mm": round(plate_limit_per_radius(ETA_NOMINAL) * R, 3),
        "beta_max": float(beta_max), "eroded_at_nominal_threshold": round(at_half, 4),
        "time_factor": round(time_factor(st.iterations), 4),
    })
    if st.robust == "wall":
        d["padded_filter"] = True   # the density filter is padded with void (filter.make_filter, pad_void)
        from .members import nominal_width_per_radius
        d["nominal_wall_mm"] = round(nominal_width_per_radius(eta_e) * R, 3)   # the nominal width of the thinnest wall that survives the erosion
    if stress_rows:
        d["facts"].append(stress_stop_text(stress_rows, st.robust) + ". The run stops at the start")
    if buckling_row:
        d["facts"].append(buckling_stop_text(st.robust) + ". The run stops at the start")
    d["solver"], d["solver_tested"] = solver, None if solver is None else solver in ROBUST_SOLVERS   # None: not known
    fact = solver_fact(solver, st.robust)
    if fact:
        d["facts"].append(fact)
    req = st.min_member_mm if st.robust != "wall" else None   # the wall rule sets the radius: min_member_mm is only reported there (members.wall_min_member_fact)
    if req is not None and float(req) < MIN_MEMBER_VOXELS * h:
        d["facts"].append(
            f"robust mode with min_member_mm {float(req):g} mm, which is {float(req) / h:.2g} voxels at "
            f"voxel_size {h:g} mm: the eroded threshold cannot tell members of less than 2 voxels apart, so "
            f"the grid sets the member size. Options: voxel_size {float(req) / 3.0:.3g} mm or less gives 3 "
            f"voxels across; min_member_mm {MIN_MEMBER_VOXELS * h:g} mm or more gives 2 voxels at this "
            f"grid; or keep both and accept the grid limit.")
    elif limit < MIN_MEMBER_VOXELS * h:
        d["facts"].append(
            f"the eroded threshold {eta_e:g} keeps a wall of {limit:.3g} mm ({limit / h:.2g} voxels) or more "
            f"at filter radius {R:g} mm (continuum estimate). A limit below 2 voxels is at the grid scale, so "
            f"robust mode has little effect on the member size. Options: "
            + ("a smaller voxel_size (the wall rule sets the radius from print.min_wall_mm), or settings.filter_radius set by hand."
               if st.robust == "wall" else
               "set settings.min_member_mm (it derives the filter radius), raise settings.filter_radius, or use a smaller voxel_size."))
    if at_half > EROSION_VISIBLE:
        d["facts"].append(
            f"the highest beta of settings.beta_schedule is {beta_max:g}. At this beta the eroded density at "
            f"rho_tilde 0.5 is {at_half:.2f} (0.5 means no erosion), so robust mode has little effect. "
            f"Options: raise the last beta (the default schedule ends at 8), or run more iterations at the last beta.")
    if req is not None and st.filter_radius_source == "min_member_mm" and limit > float(req):
        per_member = 1.0 / plate_limit_per_radius(eta_e)
        d["radius_for_request_mm"] = round(per_member * float(req), 3)
        d["facts"].append(
            f"the radius rule of min_member_mm (filter_radius = {R / float(req):.4g} x min_member_mm) is for the "
            f"threshold {ETA_NOMINAL:g}. With eta_e {eta_e:g} the eroded design keeps walls of about {limit:.3g} mm "
            f"or more (continuum estimate), against the request {float(req):g} mm. Option: set settings.filter_radius "
            f"to {per_member:.4g} x min_member_mm = {per_member * float(req):.3g} mm, which makes the wall limit of the "
            f"eroded design equal to the request (not calibrated by a run), or keep the rule and accept the thicker members.")
    return d


def wall_fact(st, h: float, wall_mm: float) -> str:
    """The fact for `print.min_wall_mm` in the YAML with `settings.robust` not `wall`: the optimizer does not aim at this wall.

    It gives the radius that the wall rule would derive and the measured price of the rule (CONTRACTS.md section 22b, evidence).
    A fact with options: it stops nothing.
    """
    from .members import wall_radius
    eta, _ = eta_eroded_of(st, "wall")
    r = wall_radius(wall_mm, eta, h)
    if st.filter_radius_source == "settings":   # settings.filter_radius wins over the rule in the wall mode
        R = float(st.filter_radius)
        radius_text = (f"it keeps the filter radius settings.filter_radius {R:g} mm (an explicit radius wins) and gives a nominal wall of about "
                       f"{r['nominal_width_per_radius'] * R:.3g} mm at eta_e {eta:g} (continuum value); without settings.filter_radius the rule "
                       f"derives {r['radius_mm']:.4g} mm (R = max(w / L, 1.5 voxels), w {wall_mm:g} mm, L {r['nominal_width_per_radius']:.4g} of R)")
    else:
        radius_text = (f"it derives the filter radius {r['radius_mm']:.4g} mm (R = max(w / L, 1.5 voxels), w {wall_mm:g} mm, eta_e {eta:g}, "
                       f"L {r['nominal_width_per_radius']:.4g} of R)")
    eta_text = "" if eta == WALL_ETA_DEFAULT else f" The measured numbers below are for eta_e {WALL_ETA_DEFAULT:g}; settings.eta_eroded {eta:g} is used."
    return (f"print.min_wall_mm is {wall_mm:g} mm, and settings.robust is {st.robust}: the optimizer does not aim at this wall; "
            f"the print check measures it after the run. settings.robust: wall would aim at it: {radius_text}, "
            f"pads the density filter with void so that a wall at a face of the design space is as thick as an inner wall, and "
            f"projects the stiffness at eta_e.{eta_text} Measured on the hook example (voxel 0.5 mm, 564,298 active voxels, wall limit 1.2 mm, "
            f"80 passes, gpu-mg on one Intel Arc iGPU, eta_e 0.7, radius 1.521 mm): the share of the surface area with a wall under 1.2 mm fell from "
            f"4.145 to 0.000 percent, the compliance rose by 7.9 percent, the run time was 1.07 times, and the peak memory was 4.86 GiB against "
            f"5.51 GiB for the default run (radius 0.75 mm); the cause of the lower memory was not measured. Options: set settings.robust to wall; "
            f"or keep this setting and read the wall check after the run (topopt printcheck)")


def lines(rep: dict) -> list[str]:
    """Text lines for `topopt check` and the run log, from `report`. The facts are not in these lines."""
    if not rep.get("on"):
        return []
    src = {"default": "default", "wall_rule": "default of the wall rule"}.get(rep["eta_e_source"], "settings.eta_eroded")
    if rep["mode"] == "wall":
        return [
            f"robust: wall rule (the eroded form), eta_n {rep['eta_n']:g}, eta_e {rep['eta_e']:g} ({src}); {rep['note']}; one solve "
            f"for each pass and {rep['extra_solves']} more at the end",
            f"robust: the density filter is padded with void: a wall at a face of the design space is as thick as an inner wall. "
            f"The eroded design keeps a wall of {rep['wall_limit_mm']:g} mm ({rep['wall_limit_voxels']:g} voxels) or more; "
            f"that wall is {rep['nominal_wall_mm']:g} mm wide in the nominal design; continuum estimate. "
            f"Estimated time: {rep['time_factor']:.3f} times the default run",
        ]
    return [
        f"robust: eroded form, eta_n {rep['eta_n']:g}, eta_e {rep['eta_e']:g} ({src}); {rep['note']}; one solve "
        f"for each pass and {rep['extra_solves']} more at the end",
        f"robust: the eroded design keeps a wall of {rep['wall_limit_mm']:g} mm ({rep['wall_limit_voxels']:g} "
        f"voxels) or more (the nominal design: {rep['wall_limit_nominal_mm']:g} mm); continuum estimate. "
        f"Estimated time: {rep['time_factor']:.3f} times the default run",
    ]


# --- the end of a run ---------------------------------------------------------------------------


def finish_run(model, x, beta, c_cases, out, rows=None) -> None:
    """End of `optimize.run`: one solve of the nominal final design, then the summary of the robust part.

    `x` and `beta` are the design and the beta of the last pass, `c_cases` its (eroded) compliance of
    each case. The summary goes to `robust.json` of the run folder (`RunWriter.write_robust`) and to the
    run log. Nothing happens when robust mode is off. `rows` is the `MotionSet` of the A5 rows (section 31):
    their physical values at the nominal design go to the summary as `rows_nominal`.
    """
    rb = model.robust
    if rb is None:
        return
    import numpy as np
    t0 = time.perf_counter()
    _rho_tilde, rho_n = model.density(x, beta)
    U = model.solve(model.stiffness(rho_n))
    c_nominal = np.einsum("ij,ij->j", model.f_red, U)
    rb.solve_s = time.perf_counter() - t0
    w = model.weights
    ce, cn = float(w @ np.asarray(c_cases, float)), float(w @ c_nominal)
    summary = {
        "on": True, "mode": rb.mode, "eta_n": rb.eta_n, "eta_e": rb.eta_e, "eta_e_source": rb.source,
        "compliance_eroded": ce, "compliance_nominal": cn,
        "compliance_cases_eroded": [float(v) for v in c_cases],
        "compliance_cases_nominal": [float(v) for v in c_nominal],
        "robust_s": round(rb.seconds, 3),
        "robust_parts_s": {"projection": round(rb.projection_s, 3), "nominal_solve": round(rb.solve_s, 3)},
        "run_s": round(out.elapsed(), 1),
    }
    if rows is not None and rows.rows:
        summary["rows_nominal"] = rows.physical_values(U)   # A5 rows (section 31): the nominal design, like compliance_nominal
    if model.kb is not None:       # springs (section 35): the reactions of the nominal design, the one that compliance_final describes
        summary["reactions"] = model.reaction_facts(U, rho_n)
    out.write_robust(summary)
    for line in summary_lines(summary):
        out.log(line)
    if "reactions" in summary:
        from .springs import reaction_lines
        for line in reaction_lines(summary["reactions"]):
            out.log(line)


def summary_lines(s: dict) -> list[str]:
    """The run log lines of a robust run (CLI summary), from the summary dict of `finish_run`."""
    up = (s["compliance_eroded"] / s["compliance_nominal"] - 1.0) * 100.0 if s["compliance_nominal"] > 0 else float("nan")
    return [
        f"robust: eta_n {s['eta_n']:g}, eta_e {s['eta_e']:g}; compliance nominal {s['compliance_nominal']:.6g}, "
        f"eroded {s['compliance_eroded']:.6g} (the eroded value is {up:.1f} percent higher; the optimizer "
        f"minimised the eroded one). The curve above (history) is the eroded design; compliance_final and the "
        f"stiffness of metrics.json are the nominal design, the part that you print",
        f"robust: {s['robust_s']:.2f} s in the robust part (projection {s['robust_parts_s']['projection']:.2f} s, "
        f"solve of the nominal design {s['robust_parts_s']['nominal_solve']:.2f} s) of {s['run_s']:.1f} s run time "
        f"(the run time before the post-process)",
    ]


def final_lines(out_dir) -> list[str]:
    """The last line of a robust run for the CLI: every record of the goal on one line (empty for a default run).

    It reads `robust.json` of the run folder and the total run time of `result/metrics.json` (post-process included).
    """
    import json
    from pathlib import Path
    d = Path(out_dir)
    try:
        s = json.loads((d / "robust.json").read_text())
        total = json.loads((d / "result" / "metrics.json").read_text())["run_time_s"]
    except (OSError, ValueError, KeyError):
        return []
    return [f"robust: summary: on, eta_n {s['eta_n']:g}, eta_e {s['eta_e']:g}; compliance nominal "
            f"{s['compliance_nominal']:.6g}, eroded {s['compliance_eroded']:.6g}; {s['robust_s']:.2f} s in the robust "
            f"part; total run time {total:g} s; compliance_final is the nominal design, the history curve the eroded design"]


NOTE_FINAL = ("compliance_final and stiffness_cases are the nominal design (the part that you print). "
              "compliance_eroded and stiffness_cases_eroded are the eroded design that the optimizer minimised. "
              "compliance_initial and the history curve are the eroded design")
NOTE_INCOMPLETE = ("the run ended before the nominal solve, so compliance_final and stiffness_cases are the "
                   "eroded values of the last history line")


def metrics_block(run: dict, summary: dict | None, run_time_s: float, stiffness_eroded: list | None = None) -> dict:
    """The `robust` block of `result/metrics.json`.

    `run` is `run.json`, `summary` is `robust.json` (None if the file is missing), `run_time_s` the
    total run time of the metrics, `stiffness_eroded` the N/mm of each case for the eroded design (made
    by `post.case_stiffness`, the same formula as `stiffness_cases`). A default run gives `{"on": false}`.
    """
    rep = (run.get("derived") or {}).get("robust") or {}
    if not rep.get("on"):
        return {"on": False}
    block = {k: rep[k] for k in ("on", "mode", "eta_n", "eta_e", "eta_e_source")}
    if rep.get("padded_filter"):   # robust: wall (section 22b); the other modes keep their keys
        block["padded_filter"] = True
    if summary is None:   # the run ended before the summary
        return {**block, "complete": False, "note": NOTE_INCOMPLETE, "run_time_s": run_time_s}
    block.update({k: summary[k] for k in ("compliance_nominal", "compliance_eroded", "compliance_cases_nominal",
                                          "compliance_cases_eroded", "robust_s", "robust_parts_s")})
    block["stiffness_cases_eroded"] = stiffness_eroded
    block["note"] = NOTE_FINAL
    block["run_time_s"] = run_time_s
    block["complete"] = True
    return block
