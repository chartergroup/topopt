"""Compliance minimisation with SIMP, density filter, Heaviside and MMA.

`run()` is the main loop. The marker comments in it name the code blocks that
the dashboard shows. Helpers are below `run()` and in `filter.py`.

`minimize()` is the same loop for scripts. It takes a `Formulation` (an objective
and a list of constraints, see formulation.py) and returns a `Result`.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from . import checkpoint
from . import eta
from . import fine_pass
from . import robust
from . import solver_fallback
from . import stage_facts
from .config import UserError
from .fe import Assembler, edof_table, hex8_ke, rigid_body_modes, triplets_aligned
from .filter import filtered_density, heaviside_deriv, heaviside_project, make_filter, simp_E, simp_dEdrho
from .formulation import (Formulation, DesignPoint, compliance_constraint, mass_objective, volume_constraint,
                          weighted_compliance, worst_case_compliance)
from .ks_stress import stress_ks_constraint
from .loads import no_load_lines, repeated_load_cases, repeated_load_lines, zero_compliance_refusal
from .mma import Mma, MmaState  # noqa: F401  (Mma and MmaState are part of the API of this module)
from .perimeter import PRICE_FACT as PERIMETER_PRICE_FACT, ROW_NAME as PERIMETER_ROW, PerimeterRow
from .rows_motion import motion_set
from .runjob import read_own_stop
from .solver import LinearSolver
from .solver_choice import resolve_solver
from .verdict import beta_stage_fact, build_verdict, stop_decision, write_verdict
from . import symmetry
from .joints import support_matrix
from .springs import reaction_facts, reaction_lines
from .voxelize import DESIGN, FIXED, LOAD

E_MIN = 1e-6   # stiffness of void (relative)
ETA = 0.5      # Heaviside threshold


class RunStopped(Exception):
    """A request to stop (`runjob.request_stop`, the file `job.stop`) ended the run (CONTRACTS.md section 24.2). It is not a `UserError`.

    `run()` raises it after the checkpoint of the last pass is saved. `iteration` is that pass, `n_iterations` the pass count of the run,
    `request` the request (`runjob.read_own_stop`). `complete` is True when the request came after the last pass of the optimizer (the
    checkpoint is the complete one): `topopt run --resume` then runs the post step only. Else the resume runs pass `iteration + 1` next.
    `topopt.cli` catches it, writes the status `stopped` and ends the job with `runjob.STOPPED_EXIT`.
    """

    def __init__(self, iteration: int, n_iterations: int, request: dict | None, complete: bool = False):
        self.iteration, self.n_iterations, self.request, self.complete = int(iteration), int(n_iterations), request, bool(complete)
        super().__init__(self.message)

    @property
    def message(self) -> str:
        if self.complete:
            return (f"stopped by a request before the post step (the optimizer ended at pass {self.iteration} of {self.n_iterations}); "
                    "result/ is not made; --resume runs the post step only")
        return (f"stopped by a request at pass {self.iteration} of {self.n_iterations}; "
                f"--resume continues at pass {self.iteration + 1}")


def run(problem, out, resume=None, solver=None, x0=None):
    """Optimise the design. `out` writes the run folder.

    `solver` is the name of the solver of this part of the run: "gpu-mg", or a CPU solver ("cholesky", "pardiso", "amg"). `cli._run_folder` gives
    the solver that `solver_choice.run_solver` chose, and `solver_fallback.run_with_fallback` gives the CPU solver when gpu-mg failed. None:
    the value of `settings.solver` ("auto" is then the CPU rule, as before GPU first).

    `x0` is the start design of a new run (the fine pass, `topopt/fine_pass.py`): an array with one value for each design voxel, or a
    function that gets the `Model` and returns such an array. Without `x0` the start design is an even density at `volume_fraction`.
    Give `x0` or `resume`, not both: a resume restores the design of its checkpoint.
    A `fine_pass.FineStart` as `x0` makes the run a fine pass (CONTRACTS.md section 24.6): its `record` goes to run.json and to each checkpoint
    (a resume of the run reads it from the checkpoint), its `lines` go to the run log, and the stop rule `settled` is on: the run ends when its
    loss has settled (`verdict.stop_decision`), or at the last pass, which is then the maximum.

    `resume` (a `checkpoint.ResumePlan`) continues a run from its checkpoint (CONTRACTS.md section 24.2). Each pass
    writes a checkpoint, and the end of the run writes `verdict.json`. A stop rule (off by default) ends the run
    at a pass that then only measures, like the last pass.

    A request to stop (`job.stop`, see `runjob.request_stop`) is read at the end of each pass that is not the last pass and was not ended by a stop
    rule, after the checkpoint of the pass is saved: one file read per pass, no wait. The run then writes `verdict.json` with the state
    `stopped_by_request`, frees the solver and raises `RunStopped`. The history line, the snapshot and the checkpoint of that pass are the ones
    of a run that goes on: a resumed run equals a run that was not stopped. The stage facts of the pass that no stage end made go to the run log
    and to the verdict, not to the history. `robust.finish_run` and the perimeter end value wait for the resume.
    """
    cfg = problem.config.settings
    n_iter = cfg.iterations
    # [step:setup]
    out.status("starting", 0, "setup")
    m = make_model(problem, out, resume, solver)      # grid, filter, assembler, solver
    out.write_labels(problem.labels)
    x = np.full(m.n_design, cfg.volume_fraction)   # start with an even density
    if x0 is not None:                        # a new run from a given design (the fine pass). It is made before run.json: the record has its numbers
        if resume is not None:
            raise ValueError("give x0 or resume, not both: a resume restores the design of its checkpoint")
        x = np.array(x0(m) if callable(x0) else x0, dtype=np.float64)
        if x.shape != (m.n_design,):
            raise ValueError(f"x0 has the shape {x.shape}, the model has {m.n_design} design voxels")
    fine = getattr(x0, "record", None) if x0 is not None else getattr(resume, "fine_from", None)   # #318: the record of a fine pass, or None
    out.write_run_json(problem, m.n_el, m.n_free, m.solver_name,
                       started_at=checkpoint.started_at(out.dir) if resume else None,
                       continued_from=getattr(resume, "continued_from", None),   # P4b-5: a continuation (and the resume of one) says where it began
                       fine_from=fine)
    out.log(f"model: {m.n_el} elements, {m.n_free} free dofs, {m.n_design} design voxels, "
            f"solver {m.solver_name}")
    for line in getattr(x0, "lines", ()):    # a fine pass: the facts of the start design
        out.log(line)
    form = formulation_from_config(problem.config, m.n_design, m, out.log)   # objective, volume, case limits
    if x0 is not None:
        x = np.clip(x, form.xmin, 1.0)    # a start design below the lower bound of the rows (the stress row asks for 0.02) would start MMA outside its box
    if form.perimeter.active:                 # A4c (section 26.3): the numbers of the bound, known before the first pass
        out.write_perimeter(form.perimeter.block())
    mma = Mma(m.n_design, form.n_rows, cfg.move_limit, xmin=form.xmin)
    rule = StepAcceptance(m, form, mma, cfg.move_limit, cfg.stress_step_acceptance)   # stress rows only
    ev, prev, first, records = None, None, 0, []   # `records`: the history records of the run
    ckpt = checkpoint.Checkpointer(out, m, mma, rule, problem.config,
                                   checkpoint.input_hashes(problem.config, problem, m.solver_name), resume, fine_from=fine if resume is None else None)
    stop = None
    react = {}   # springs (section 35): the reactions of the last pass
    beta = c = c_cases = vol = gray = change = None   # the loop sets them. A resume of a finished run runs no pass (see below)
    if resume is not None:                    # the one place where a saved state goes in
        x, prev, first, records = ckpt.restore()
        out.first_pass = first   # status.json says where THIS process starts (E1-P2); the live time estimate counts its passes from here
        stop = None if getattr(resume, "replay", False) else resume.ck.meta.get("stop")   # a replay (P4b-5) runs the pass of a stop rule again: the rule is judged in the loop
        if first > n_iter:                    # no pass is left: the code after the loop (robust.finish_run solves the nominal design)
            out.step(resume.ck.iteration, "snapshot")   # runs as it does after the last pass, so the run writer has that pass and step
        if prev is not None:                  # the values of the last pass, for code after the loop that reads them (a resume that runs
            beta, c, c_cases = prev.get("beta"), prev.get("compliance"), prev.get("compliance_cases")   # no pass leaves them unset)
            vol, gray, change = prev.get("volume_fraction"), prev.get("grayness"), prev.get("change")
            react = prev.get("reactions") or {}
    if out.solver_parts is None:    # P9-4d: which solver ran which passes. A resume starts from the parts of the checkpoint; a restart after a fallback has them in `out`
        out.solver_parts = copy.deepcopy(resume.ck.meta.get("solver_parts") or []) if resume is not None else []
    if first <= n_iter:     # a resume of a finished run runs no pass: no part starts, so `solver` of run.json stays the solver of the last part that ran a pass
        solver_fallback.part_start(out.solver_parts, m.solver_name, m.device_name, first)
    out.write_solver_record()
    reach = problem.load_reach or None   # F4-37: for each case, why its load reaches a free dof or not (`loads.reach_kinds`)
    for line in problem_facts(problem.config, continued_from=getattr(resume, "continued_from", None), reach=reach, fine=fine is not None):   # facts about the problem, before the first pass
        out.log(line)
    asked = None   # the request to stop that ended the run at a pass (read at the end of the pass)
    for it in range(first, n_iter + 1):
        t0 = time.time()
        beta = beta_at(cfg.beta_schedule, it)
        # [step:assemble]
        out.step(it, "assemble")
        rho_tilde, rho = m.density(x, beta)   # filter, then Heaviside
        rho_stiff = m.stiffness_density(rho_tilde, rho, beta)   # rho itself; the eroded density in robust mode
        U = rule.solved(x, beta)              # step acceptance solved this design already: no assembly, no solve
        K = m.stiffness(rho_stiff) if U is None else None   # SIMP stiffness of each element
        # [step:solve]
        out.step(it, "solve")
        if U is None:
            U = m.solve(K)                    # one column for each load case
        # [step:sensitivity]
        out.step(it, "sensitivity")
        point = m.design_point(x, beta, rho_tilde, rho, U, rho_stiff, iteration=it, last=it == n_iter)   # c = f . u for each case
        if objective_scale_resets(cfg.objective_scale, cfg.beta_schedule, it):   # a new beta stage: respond() sets the scale of the objective again
            mma.state.f0_ref = None
        rows_before = copy.deepcopy(mma.state.rows)   # P4b-5: the rows at the start of the pass; a complete checkpoint keeps them (a continuation runs this pass again)
        refusal = zero_compliance_refusal(problem.config, point.compliance_cases, reach) if reach else None   # F4-37: c is 0.0 or subnormal for a case with a load
        if refusal:                           # the log form, the max form and the buckling row stop on it with an internal error: say why instead
            raise UserError(refusal)
        resp = form.respond(point, mma.state)   # objective and limits, and their slopes
        c, c_cases = point.compliance, point.compliance_cases
        # [step:filter]
        out.step(it, "filter")
        ev = form.chain_rule(point, resp)     # back through filter
        vol = float(rho.mean())
        gray = float(np.mean(4 * rho * (1 - rho)))
        stop = None
        if cfg.stop_rules.enabled:            # the stop rules: off by default (CONTRACTS.md section 24.2)
            records.append({"iteration": it, "beta": beta, "objective": {"value": ev.objective},
                            "grayness": gray, "constraints": ev.constraints})
            stop = stop_decision(records, cfg.stop_rules, cfg.beta_schedule, out.elapsed(), settled=fine is not None)   # a fine pass: also the rule `settled`
            records.pop()
        # [step:update]
        out.step(it, "update")
        change = 0.0
        step = None
        if it < n_iter and stop is None:      # the last pass, and the pass of a stop rule, only measure
            if rule.active:                   # a stress row: reject a step that makes it worse
                x_new, step = rule.step(x, ev, beta)
            else:
                x_new = mma.step(x, ev.f0, ev.df0, ev.g, ev.dg, ev.a0, ev.a, ev.c)
            change = float(np.abs(x_new - x).max())
        # [step:snapshot]
        out.step(it, "snapshot")
        snap = m.density_bytes(rho)
        out.write_snapshot(it, snap)
        record = {
            "iteration": it, "elapsed_s": round(out.elapsed(), 2), "compliance": c,
            "compliance_cases": [float(v) for v in c_cases], "volume_fraction": vol,
            "beta": beta, "change": change, "grayness": gray,
            "t_iter_s": round(time.time() - t0, 3), **form.report(ev)}
        if step is not None:
            record["step_acceptance"] = step
        if problem.joints is not None and getattr(m.solver, "method", None) == "amg":     # joints (section 38): CG iterations of the main solve (gpu-mg has no `method`)
            record["amg_iterations"] = [int(n) for n in m.solver.last_iters]
        note = form.stage_note(prev, beta, it, ev.constraints)   # a fact at each beta stage start
        if note:
            record["stage"] = note
            for line in form.stage_lines(note):
                out.log(line)
        if m.kb is not None and m.robust is None and (it == n_iter or stop is not None):   # springs (section 35): the last pass,
            react = m.reaction_facts(U, rho_stiff)                 # also when a stop rule ends the run; robust mode: `finish_run`
            record["reactions"] = react
        next_beta = None if it == n_iter or stop is not None else beta_at(cfg.beta_schedule, it + 1)
        if next_beta is None or next_beta != beta:   # the last pass of a beta stage, or of the run: the facts of that design (P3a)
            record["stage_end"] = stage_facts.stage_end(problem, snap, rho, ev.constraints, beta, next_beta)
        out.add_history(record)
        records.append(record)
        prev = record
        out.log(f"it {it:3d}  beta={beta:g}  c={c:.5g}  vol={vol:.4f}  "
                f"change={change:.3f}  t={time.time() - t0:.2f}s{form.log_extra(ev)}")
        for line in rule.lines(it, step):
            out.log(line)
        for line in stage_facts.stage_end_lines(record.get("stage_end"), it):
            out.log(line)
        if stop is not None:
            out.log(stop["text"])
        if it == first + eta.FIRST_PASS_WITH_ESTIMATE and it < n_iter and stop is None:   # E1-P2: one line, once for each process
            est, _why = eta.time_left(records, n_iter, m.n_free, start=first)
            if est is not None:
                out.log(eta.log_line(est))
        if it < n_iter and stop is None:
            x = x_new
        ckpt.save(it, x, prev, complete=it == n_iter or stop is not None, stop=stop, rows_before=rows_before)   # the state of the next pass
        # [end]
        if stop is not None:
            break
        if it < n_iter:                       # a request to stop (P4b-4): after the checkpoint, so the stopped run has the state of pass `it`
            asked = read_own_stop(out.dir)    # a request for another process stays in the folder for it
            if asked is not None:
                break
    last_rows = ev.constraints if ev is not None else prev["constraints"]
    for line in rule.summary() + form.unmet(last_rows):
        out.log(line)
    if asked is None:
        if form.perimeter.active:             # A4c: P of the last design goes into the block of run.json
            out.write_perimeter(form.perimeter.block(p_final=next(c["value"] for c in last_rows if c["name"] == PERIMETER_ROW)))
        for line in reaction_lines(react) if react else []:
            out.log(line)
    if ckpt.summary_line():
        out.log(ckpt.summary_line())
    if asked is not None:
        raise_stopped(problem, out, m, ckpt, records, asked, (snap, rho, ev, beta))
    out.write_solver_record()
    verdict = build_verdict(records, cfg.stop_rules, cfg.beta_schedule, n_iter, stop, ckpt.resume_points, notes=solver_fallback.verdict_notes(out.solver_parts))
    write_verdict(out.dir, verdict)
    if fine is not None:                      # #318: say how the fine pass ended: the loss settled, or the maximum came first
        end = fine_pass.end_line(verdict, fine if cfg.stop_rules.enabled else dict(fine, stop_rule=None), n_iter)   # a continuation with --no-stop-rules has the rules off
        if end:
            out.log(end)
    out.log("verdict: " + verdict["text"])
    robust.finish_run(m, x, beta, c_cases, out, form.motion)   # robust mode: nominal compliance and A5 rows of the final design, summary
    m.solver.free()
    return x


def raise_stopped(problem, out, m, ckpt, records, asked, last_pass):
    """The end of `run()` for a request to stop at the end of the last pass in `records` (P4b-4). Does not return: it raises `RunStopped`.

    `last_pass` is `(snap, rho, ev, beta)` of that pass: the density bytes, the density, the `Evaluation` and the Heaviside beta.
    The pass is not the last pass, and no stage ended at it, or it is a stage end. If its history record has no `stage_end`, the facts of the
    design of the pass are made now (`stage_facts.stage_end`, no solve), go to the run log as `fact: stop at pass N: ...` and into the
    verdict `facts`. They do not go into the history: the line is on disk already, and a resumed run writes the same line as a run that was not stopped.
    """
    cfg = problem.config.settings
    n_iter = cfg.iterations
    snap, rho, ev, beta = last_pass
    last = records[-1]
    it = int(last["iteration"])
    end = last.get("stage_end")
    if end is None:
        end = stage_facts.stage_end(problem, snap, rho, ev.constraints, beta, beta_at(cfg.beta_schedule, it + 1))
        for line in stage_facts.stage_end_lines(end, it, where=f"stop at pass {it}"):
            out.log(line)
        rows = records[:-1] + [{**last, "stage_end": end}]    # a copy: the record that is on disk and in the checkpoint stays as it is
    else:
        rows = records
    out.write_solver_record()
    verdict = build_verdict(rows, cfg.stop_rules, cfg.beta_schedule, n_iter, None, ckpt.resume_points, request=asked,
                            notes=solver_fallback.verdict_notes(out.solver_parts))
    write_verdict(out.dir, verdict)
    out.log("verdict: " + verdict["text"])
    m.solver.free()
    raise RunStopped(it, n_iter, asked)


def beta_at(schedule, it):
    """Heaviside beta for iteration `it`. `schedule` is ((iteration, beta), ...)."""
    beta = schedule[0][1]
    for start, b in schedule:
        if it >= start:
            beta = b
    return float(beta)


def objective_scale_resets(mode, schedule, it):
    """True if the objective scale is set again at pass `it` (`settings.objective_scale`, CONTRACTS.md section 13).

    Mode "stage": at the first pass of each beta stage after pass 0, that is when the beta of the pass is not the beta of the pass before.
    Mode "first": never; the scale of pass 0 stays for the whole run. Pass 0 sets the scale in both modes (`Formulation.respond`).
    The pass number and the schedule alone decide, so a run that resumes from a checkpoint (also one at the first pass of a stage)
    has the same scale as a run that was not stopped.
    """
    return mode == "stage" and it > 0 and beta_at(schedule, it) != beta_at(schedule, it - 1)


def reduced_system(problem):
    """Active elements and free dofs: `(edof, active_index, free_dofs)`."""
    active = (problem.labels >= DESIGN) & (problem.labels <= LOAD)
    edof, active_index = edof_table(problem.shape, active)
    free_dofs = np.setdiff1d(np.unique(edof), problem.fixed_dofs)
    return edof, active_index, free_dofs


def make_model(problem, out=None, resume=None, solver=None):
    """The `Model` for the solver of this part: the matrix-free `GpuModel` for "gpu-mg", else `Model`.

    `solver` is the name of the solver ("gpu-mg", "cholesky", "pardiso", "amg") or None for the value of `settings.solver`.
    `resume` is the `checkpoint.ResumePlan` of a resumed run, else None. Only `GpuModel` reads it: its timing file goes on
    after the records of the first part."""
    name = solver or problem.config.settings.solver
    if name == "gpu-mg":
        from .gpu_mg.model import GpuModel     # lazy: pyopencl is loaded by the engine only
        return GpuModel(problem, out, resume=resume)
    return Model(problem, solver=name)


def make_solver(setting, free_dofs, shape, h, joints=None):
    """Make the LinearSolver for `settings.solver`. Return `(solver, name)`.

    `name` is the solver that really runs: cholesky, pardiso or amg (see
    `solver_choice.resolve_solver`). Only AMG gets the rigid-body nullspace. With `joints` (a `joints.JointSet`,
    section 38) the solver is a `BorderedSolver`: it takes the grid matrix and vectors and borders them itself.
    "gpu-mg" has no `LinearSolver`: `make_model` builds its model, and this function stops with a
    `UserError` for a caller that needs a matrix and a factor. The stress step of the post run does not
    call this function for "gpu-mg": it uses `gpu_mg.model.solve_design`.
    """
    name = resolve_solver(setting)
    if name == "gpu-mg":
        raise UserError(
            "this step assembles and factors the stiffness matrix, and settings.solver 'gpu-mg' is matrix-free. "
            "The matrix and its factor need the memory that gpu-mg avoids. Options: run this step with "
            "settings.solver set to 'auto', 'cholesky' (Apple silicon), 'pardiso' (Linux x86_64) or 'amg', or go "
            "without this step")
    nullspace = rigid_body_modes(free_dofs, shape, h) if name == "amg" else None
    if joints is not None:
        return joints.make_solver(name, free_dofs, shape, h, nullspace, triplets_aligned(free_dofs)), name
    return LinearSolver(name, nullspace=nullspace, block3=triplets_aligned(free_dofs)), name


# --- step acceptance for the stress rows (A2, CONTRACTS.md section 17) ------------------------------

STEP_TOL = 1e-3     # a stress row may rise by this much in g (above the larger of its old g and 0)
STEP_TRIES = 6      # tries in one pass: the first, then up to 5 more with the move limit halved
STEP_HALVE = 0.5    # the move limit after a rejected try
STEP_GROW = 1.5     # the move limit after an accepted step, up to the move limit of the settings


class StepAcceptance:
    """Take an MMA step only if the stress rows do not get worse. For rows with `stage_aware`.

    MMA keeps a convex model of the KS row. When the peak moves to other voxels, the model is too
    optimistic, and the row goes up while MMA predicts that it goes down (measured on the hook at
    1.2 mm: the design collapsed; at 2.4 mm the row swung around the limit). This rule checks the step:

    1. MMA gives a trial design. One solve at the trial design gives the values of the stress rows
       (`Constraint.measure`, no adjoint).
    2. Accepted: each stress row has `g_new <= max(g_old, 0) + STEP_TOL` (linear g, the same beta).
    3. Rejected: restore the MMA state (a copy of `MmaState`), halve the move limit, and try again,
       up to `STEP_TRIES` tries. After the last try the step is taken, and the pass records this as a fact.
    4. After an accepted step the move limit grows by `STEP_GROW`, up to the limit of the settings.
    5. The solve of the accepted trial design is the solve of the next pass (if beta is the same).

    The rule is a heuristic, not GCMMA (GCMMA raises the convexity terms of the rows that the model
    under-predicts, and has a proof of convergence; it needs a change of the MMA call). A run with no
    stress row does not use the rule: `active` is False and `minimize` and `run` call `Mma.step` as before.
    The move limit for the next pass is kept in `MmaState.move`.
    """

    def __init__(self, model, formulation, mma, move_max, enabled=True):
        self.model, self.form, self.mma, self.move_max = model, formulation, mma, float(move_max)
        self.names = ([c.name for c in formulation.constraints if c.stage_aware and c.measure is not None]
                      if enabled else [])
        self.carry = None
        self.n_passes = self.n_rejected = self.n_rejected_passes = self.n_forced = 0

    @property
    def active(self) -> bool:
        return bool(self.names)

    def solved(self, x, beta):
        """Displacements `U` of the accepted trial design if `x` is that design and beta is the same, else None."""
        carry, self.carry = self.carry, None
        if carry is not None and carry[0] is x and carry[1] == beta:
            return carry[2]
        return None

    def step(self, x, ev, beta):
        """Return `(x_new, info)`. `ev` is the `Evaluation` of the pass, `beta` its Heaviside beta.

        `info` goes into the history record as `step_acceptance`: `tries`, `accepted`, `move` (of the last try),
        `move_next`, `rejected` (a list of `{move, g}`: the move limit and the g of the stress rows at the
        rejected trial design), `g_old` (the g of the rows now).
        """
        mma, model = self.mma, self.model
        move = mma.state.move if mma.state.move is not None else self.move_max
        g_old = {e["name"]: e["g"] for e in ev.constraints if e["name"] in self.names}
        rejected = []
        for attempt in range(STEP_TRIES):
            saved = copy.deepcopy(mma.state)
            x_new = mma.step(x, ev.f0, ev.df0, ev.g, ev.dg, ev.a0, ev.a, ev.c, move=move)
            rho_tilde, rho = model.density(x_new, beta)
            rho_stiff = model.stiffness_density(rho_tilde, rho, beta)   # the eroded density in robust mode (section 22b)
            U = model.solve(model.stiffness(rho_stiff))
            g_new = self.form.measure(model.design_point(x_new, beta, rho_tilde, rho, U, rho_stiff), self.names)
            ok = all(g_new[n] <= max(g_old[n], 0.0) + STEP_TOL for n in self.names)
            if ok or attempt == STEP_TRIES - 1:
                break
            rejected.append({"move": move, "g": g_new})
            mma.state = saved
            move *= STEP_HALVE
        move_next = min(self.move_max, move * STEP_GROW) if ok else move
        mma.state.move = move_next
        self.carry = (x_new, beta, U)
        self.n_passes += 1
        self.n_rejected += len(rejected) + (0 if ok else 1)
        self.n_rejected_passes += bool(rejected) or not ok
        self.n_forced += not ok
        info = {"tries": attempt + 1, "accepted": bool(ok), "move": move, "move_next": move_next,
                "g_old": g_old, "rejected": rejected}
        if not ok:
            info["g_taken"] = g_new   # the step was taken with these values: the rows are worse than before
        return x_new, info

    def lines(self, it, info) -> list[str]:
        """Run log lines (facts) for the pass `it`. Empty when the first try was accepted.

        Each rejected try is listed with its move limit and the rows that made it fail (g at the trial
        design); `g_old` is the g of those rows now. The history keeps the g of all rows.
        """
        if not info or (info["accepted"] and not info["rejected"]):
            return []

        def worse(g):
            return ", ".join(f"{n} {v:+.4f}" for n, v in g.items()
                             if v > max(info["g_old"][n], 0.0) + STEP_TOL)

        tries = [(r["move"], r["g"]) for r in info["rejected"]]
        if not info["accepted"]:
            tries.append((info["move"], info["g_taken"]))
        now = ", ".join(f"{n} {v:+.4f}" for n, v in info["g_old"].items())
        line = (f"fact: step acceptance, pass {it}: {len(tries)} of {info['tries']} tries rejected, "
                f"a stress row would get worse (g now: {now}). Rejected (move limit: row g at the trial "
                f"design): " + "; ".join(f"{m:.4g}: {worse(g)}" for m, g in tries))
        if info["accepted"]:
            return [line + f". Accepted with move limit {info['move']:.4g}"]
        return [line + f". All {STEP_TRIES} tries were rejected and the step with move limit "
                       f"{info['move']:.4g} was taken"]

    def summary(self) -> list[str]:
        """One fact line at the end of the run, or nothing without a stress row."""
        if not self.active:
            return []
        return [f"fact: step acceptance (stress rows): {self.n_rejected} tries rejected in "
                f"{self.n_rejected_passes} of {self.n_passes} passes; {self.n_forced} passes took the step "
                f"after {STEP_TRIES} rejected tries (settings.stress_step_acceptance: false turns the rule off)"]


class Model:
    """The finite element model and the maps between x, rho and the system."""

    def __init__(self, problem, solver=None):
        cfg = problem.config
        st = cfg.settings
        self.problem = problem
        self.device_name = None       # the OpenCL device of gpu-mg (`GpuModel` sets it), else None: run.json `solver_parts`
        self.solver_setting = solver or st.solver    # the value for `make_solver`: a CPU solver name, or "auto" (the CPU rule)
        self.E = cfg.material.E
        self.penal = st.penal
        self.weights = problem.case_weights
        self.robust = robust.from_settings(   # None unless settings.robust is eroded (section 22b); a stress or buckling row stops it
            st, sum(1 for c in cfg.load_cases if c.stress_limit_mpa is not None),
            buckling_row=cfg.buckling.min_factor is not None)
        edof, self.active_index, self.free_dofs = reduced_system(problem)
        self.n_el, self.n_free = edof.shape[0], self.free_dofs.size
        label = problem.labels.reshape(-1)[self.active_index]
        self.design_pos = np.flatnonzero(label == DESIGN)   # place among the elements
        self.solid_pos = np.flatnonzero((label == FIXED) | (label == LOAD))
        self.n_design = self.design_pos.size
        self.edof_design = edof[self.design_pos]
        self.n_dof_all = 3 * int(np.prod(np.array(problem.shape) + 1))
        self.ke = hex8_ke(1.0, cfg.material.nu, problem.h)    # unit stiffness (E = 1)
        self.assembler = self._make_assembler(edof)
        # The filter as FFT operators (#333): no stored matrix, the numbers of `build_filter` to rounding. Section 22b: the wall rule
        # pads the filter with void.
        self.F, self.FT, self.b = make_filter(problem.labels, problem.h, st.filter_radius,
                                             st.filter_solid_neighbours, pad_void=st.robust == "wall")
        self.f_red = np.ascontiguousarray(problem.forces[:, self.free_dofs].T)
        self.solver, self.solver_name = self._make_solver(edof)
        self.kb = None
        self.edof = edof                       # the table of all active voxels: springs, joints and the buckling row (A6b) read it
        if (problem.springs is not None or problem.joints is not None) and self.assembler is not None:   # springs (section 35) and
            self.kb = support_matrix(problem, self.assembler, self.free_dofs)   # bonds (38): Kb in the pattern of K.
            # gpu-mg has no assembler: `GpuModel` sets `kb` to the support terms of its solver (P2b)

    # Hooks for the solver backend. `GpuModel` (topopt/gpu_mg/model.py) replaces both.
    def _make_assembler(self, edof):
        return Assembler(edof, self.ke, self.free_dofs)

    def _make_solver(self, edof):
        return make_solver(self.solver_setting, self.free_dofs, self.problem.shape, self.problem.h, self.problem.joints)

    def density(self, x, beta):
        """Physical density of the design voxels: mirror average, filter, then Heaviside.

        The mirror average is the identity without a `mirror` key (section 26.1).
        """
        x_avg = symmetry.average(self, x)
        rho_tilde = filtered_density(self.F, self.b, x_avg)
        rho = heaviside_project(rho_tilde, beta, ETA)
        symmetry.record(self, x_avg, rho)   # the mirror deviation of the averaged design and of rho: a fact for the run report
        return rho_tilde, rho

    def stiffness_density(self, rho_tilde, rho, beta):
        """The density that the stiffness sees: `rho` itself, or the eroded density in robust mode.

        `rho` is the nominal density from `density`. The loop passes the result to `stiffness`.
        """
        return rho if self.robust is None else self.robust.eroded(rho_tilde, beta)

    def stiffness(self, rho):
        """Reduced stiffness matrix. Solid elements have full stiffness. `rho`: see `stiffness_density`."""
        K = self.assembler.K(self.element_E(rho))
        if self.kb is not None:
            self.kb.add_to(K)           # springs and the bonds of joints: K + Kb in place, same pattern (sections 35, 38)
        return K

    def element_E(self, rho):
        """Young's modulus of each active element for the density `rho` of the design voxels (SIMP)."""
        Ee = np.full(self.n_el, self.E)
        Ee[self.design_pos] = self.E * simp_E(rho, self.penal, 1.0, E_MIN)
        return Ee

    def reaction_facts(self, U, rho):
        """Reactions of the solution `U` of the stiffness density `rho` (springs and joints; sections 35, 38)."""
        return reaction_facts(self.problem, self.edof, self.element_E(rho), self.ke, self.free_dofs, U)

    def solve(self, K):
        """Displacements of the free dofs for all load cases."""
        self.solver.factor(K)
        return self.solver.solve(self.f_red)

    def compliance(self, U, rho):
        """Compliance of each case, and the weighted slope d(sum w c)/d(rho). `rho` is the density of the stiffness."""
        c = np.einsum("ij,ij->j", self.f_red, U)
        dE = self.E * simp_dEdrho(rho, self.penal, 1.0, E_MIN)
        energy = np.zeros(self.n_design)
        u_all = np.zeros(self.n_dof_all)
        for i in range(U.shape[1]):
            u_all[self.free_dofs] = U[:, i]
            ue = u_all[self.edof_design]
            energy += self.weights[i] * np.einsum("ij,ij->i", ue @ self.ke, ue)
        return c, -dE * energy

    def chain_rule(self, dc_drho, rho_tilde, beta):
        """Slopes of compliance and volume ratio with respect to x.

        `dc_drho` is the slope with respect to the density of the stiffness. In robust mode the compliance
        goes through the eroded projection and the volume through the nominal one.
        """
        dproj = heaviside_deriv(rho_tilde, beta, ETA)
        dproj = heaviside_deriv(rho_tilde, beta, ETA)
        dstiff = dproj if self.robust is None else self.robust.eroded_slope(rho_tilde, beta)
        dc, dv = self.FT @ (dc_drho * dstiff), self.FT @ dproj / self.n_design
        return symmetry.average(self, dc, transpose=True), symmetry.average(self, dv, transpose=True)

    def gradient_wrt_x(self, grads, wrts, rho_tilde, beta):
        """Chain rule for k responses: array (k, n_design) of d(response)/dx.

        `grads[i]` is the slope of response i with respect to `wrts[i]`: "rho" (physical
        density), "rho_stiff" (the density of the stiffness: the eroded one in robust mode, else
        the same as "rho"), "rho_tilde" (filtered density) or "x" (the design variable).
        """
        G = np.column_stack(grads)
        out = np.empty((G.shape[1], self.n_design))
        if self.robust is None:   # one density: "rho_stiff" is "rho"
            wrts = ["rho" if w == "rho_stiff" else w for w in wrts]
        for kind in ("rho", "rho_stiff", "rho_tilde", "x"):
            idx = [i for i, w in enumerate(wrts) if w == kind]
            if not idx:
                continue
            part = G[:, idx]
            if kind == "rho":
                part = part * heaviside_deriv(rho_tilde, beta, ETA)[:, None]
            elif kind == "rho_stiff":
                part = part * self.robust.eroded_slope(rho_tilde, beta)[:, None]
            if kind != "x":
                part = self.FT @ part
            out[idx] = part.T
        return symmetry.average(self, out, transpose=True)   # mirror: the transpose of the average in `density`

    def dE_drho(self, rho):
        """Slope of the stiffness of a design voxel with respect to its density (SIMP)."""
        return self.E * simp_dEdrho(rho, self.penal, 1.0, E_MIN)

    def design_centers(self):
        """Centres of the design voxels in mm: array (n_design, 3), same order as x."""
        flat = self.active_index[self.design_pos]
        ijk = np.stack(np.unravel_index(flat, self.problem.shape), axis=1)
        return np.asarray(self.problem.origin) + self.problem.h * (ijk + 0.5)

    def to_free(self, full_vector):
        """Free-dof part of a vector on all dofs of the grid (3 dofs for each node)."""
        return np.asarray(full_vector)[..., self.free_dofs]

    def element_energy(self, v, u):
        """v_e^T Ke u_e for each design voxel. `u` and `v` are vectors of the free dofs."""
        u_all = np.zeros(self.n_dof_all)
        v_all = np.zeros(self.n_dof_all)
        u_all[self.free_dofs] = u
        v_all[self.free_dofs] = v
        return np.einsum("ij,ij->i", v_all[self.edof_design], u_all[self.edof_design] @ self.ke)

    def design_point(self, x, beta, rho_tilde, rho, U, rho_stiff=None, iteration=None, last=False):
        """The `DesignPoint` for objective and constraint functions.

        `rho_stiff` is the density that `U` was solved with (`stiffness_density`). The loop passes it so that the
        eroded projection runs once for each pass; without it the point computes it again.
        `point.x` is the design that the model uses: the mirror average of `x` (`x` itself without a `mirror` key).
        A response with `wrt="x"` reads it, and its gradient gets the transposed average in `gradient_wrt_x`.
        `iteration` and `last` are the pass number and the last-pass flag (a row that skips passes reads them).
        """
        return DesignPoint(self, symmetry.average(self, x), beta, rho_tilde, rho, U, rho_stiff, iteration, last)

    def evaluate(self, x, beta):
        """Weighted compliance, its slope, mean density and its slope (for tests).

        In robust mode the compliance is the eroded one and the density is the nominal one.
        """
        rho_tilde, rho = self.density(x, beta)
        rho_stiff = self.stiffness_density(rho_tilde, rho, beta)
        U = self.solve(self.stiffness(rho_stiff))
        c_cases, dc_drho = self.compliance(U, rho_stiff)
        dc_dx, dv_dx = self.chain_rule(dc_drho, rho_tilde, beta)
        return float(self.weights @ c_cases), dc_dx, float(rho.mean()), dv_dx

    def density_bytes(self, rho):
        """Density of the full grid as bytes: 0 is 0.0 and 255 is 1.0."""
        u8 = np.zeros(self.problem.labels.size, dtype=np.uint8)
        u8[self.active_index[self.solid_pos]] = 255
        u8[self.active_index[self.design_pos]] = np.rint(np.clip(rho, 0, 1) * 255)
        return u8


# --- the problem from the YAML, and the script loop ------------------------


def formulation_from_config(config, n_design, model=None, log=None) -> Formulation:
    """Objective and constraints of a YAML problem (CONTRACTS.md sections 13, 17 and 31).

    Objective: weighted compliance, or the worst case with `settings.case_combine: max`; or the
    mass with `settings.objective: mass`.
    Constraints, in this order: the volume (compliance objective only), the compliance limit of
    each case that has one, the stress row of each case that has a limit, the buckling row of each
    case (only with `buckling.min_factor`, section 39), then the displacement
    and alignment rows of `limits` (A5). With `model`, the checks of the A5 rows run now (a
    `UserError` stops a row that cannot work), a buckling row on a model without a direct solver
    stops the run now, and `log` gets the facts of the rows. Without a model, the
    rows check themselves at the first evaluation.
    A mass run with no stress limit, no compliance limit, no buckling row and no A5 row has no row at all. It still
    gets the volume row (a ceiling at `settings.volume_fraction`), so that MMA has a row. An A5 row
    on a probe region does not hold material under the mass objective either: the run has these
    rows and no ceiling. A buckling row holds material in a case with a compression: a mass run
    with only buckling rows has these rows and no ceiling. `problem_facts` says both in the run log.
    The perimeter row of `settings.perimeter_max_mm2` (A4c, CONTRACTS.md section 26.3) comes last, after the volume
    fallback: it does not hold material either. With `model` the row binds now (a bound below the smallest
    possible P is a `UserError`); without a model it binds at its first evaluation.
    """
    st = config.settings
    if st.objective == "mass":
        objective, constraints = mass_objective(), []
    else:
        objective = (worst_case_compliance(len(config.load_cases)) if st.case_combine == "max"
                     else weighted_compliance())
        constraints = [volume_constraint(n_design, st.volume_fraction)]
    constraints += [compliance_constraint(i, c.compliance_limit, f"compliance:{c.name}")
                    for i, c in enumerate(config.load_cases) if c.compliance_limit is not None]
    constraints += [stress_ks_constraint(i, c.stress_limit_mpa, st.stress_q, st.stress_power,
                                         st.stress_scale, f"stress:{c.name}")
                    for i, c in enumerate(config.load_cases) if c.stress_limit_mpa is not None]
    constraints += buckling_rows(config, model, log)
    motion = motion_set(config, model, log)
    constraints += motion.constraints()
    if not constraints:
        constraints = [volume_constraint(n_design, st.volume_fraction)]
    perimeter = PerimeterRow(config, log)     # A4c: after the fallback, so a mass run with no other row keeps its volume ceiling
    if perimeter.active and model is not None:
        perimeter.bind_model(model)           # the face list, p_min (a `UserError` for a bound below it) and P of the start design
    constraints += perimeter.constraints()
    form = Formulation(objective, constraints)
    form.motion = motion   # the A5 rows (`rows_motion.MotionSet`), for the end report; it has no row without `limits`
    form.perimeter = perimeter   # the A4c row (`perimeter.PerimeterRow`), for the end report; `active` is False without the key
    return form


def buckling_rows(config, model=None, log=None) -> list:
    """The buckling row of each load case, or no row without `buckling.min_factor` (then `topopt.buckling_row` is not imported).

    With `model`, a solver without a direct factor of K (amg, gpu-mg) does too.
    """
    b = config.buckling
    if b.min_factor is None:
        return []
    from . import buckling_row
    if model is not None:
        text = buckling_row.model_refusal(model)
        if text:
            raise UserError(text)
    q = buckling_row.KS_Q_DEFAULT if b.ks_q is None else b.ks_q
    return [buckling_row.buckling_ks_constraint(i, b.min_factor, q, b.modes, b.max_seconds, f"buckling:{c.name}", log=log)
            for i, c in enumerate(config.load_cases)]


def _a5_row_on_a_load_region(config) -> bool:
    """True if a displacement or alignment row (A5) has a load region as its region (the load moves it)."""
    load = {r.name for r in config.regions.load}
    return (any(L.region in load for L in config.limits.displacement)
            or any(L.region_a in load or L.region_b in load for L in config.limits.alignment))


STRESS_SCALE_OPTIONS = ("options: 1 (the KS row is then a bound of the relaxed peak: the peak stays at or below the limit), "
                        "or auto (the default: the scale follows the design at the start of each beta stage)")


def stress_scale_facts(c: float) -> list[str]:
    """The fact for a `settings.stress_scale` number that is not 1 (F4-28): the row `c*S - 1 <= 0` asks for KS `S <= 1/c`, and `S` is at or above the relaxed peak.

    A number above 1 holds the relaxed peak below the limit (tighter than the user's limit); a number below 1 allows a relaxed peak up to `1/c` times the
    limit with the row met (looser). Both move the stress limit of the user, a requirement, and the two texts say so in the same words. Exactly 1 and `auto` give no
    fact: 1 is the limit as the user wrote it, and `auto` is the default.
    """
    if c > 1.0:
        return [f"fact: settings.stress_scale {c:g} is above 1. The row c*S - 1 <= 0 then asks for KS S <= {1.0 / c:.3g}, so the relaxed peak is held "
                f"to at most {1.0 / c:.3g} times the limit (stricter than the limit). This tightens the stress limit of the user, a requirement: "
                f"the part needs more material, and the row can be not met with the part below the limit. " + STRESS_SCALE_OPTIONS]
    if c < 1.0:
        return [f"fact: settings.stress_scale {c:g} is below 1. The row c*S - 1 <= 0 then asks for KS S <= {1.0 / c:.3g}, and KS is at or above the relaxed "
                f"peak, so the row allows a relaxed peak up to {1.0 / c:.3g} times the limit and is met there (looser than the limit). This loosens the "
                f"stress limit of the user, a requirement: the part can carry more than the limit while the row says met. Only the user can allow it "
                f"(the tool does not change the limit). " + STRESS_SCALE_OPTIONS]
    return []


def problem_facts(config, continued_from=None, reach=None, fine=False) -> list[str]:
    """Facts about a YAML problem for the run log (and `topopt check`). No fact blocks a run.

    - A mass run with no stress limit, no compliance limit and no A5 row on a load region: no row holds material
      (an A5 row on a probe region does not: the optimizer can remove the material between the load and the probe).
    - A mass run with a `max` or `min` displacement row: the row is also met when the region moves the other way.
    - `case_combine: max` with the mass objective: the key has no effect.
    - `perimeter_max_mm2` with the mass objective: the perimeter row does not hold material (A4c).
    - `allowable_source` in a case with no stress limit: the text is not used.
    - `safety_factor` below 1: the limit is above the allowable.
    - `stress_scale` above 1 (the row asks for a peak below the limit, `S <= 1/c`) and below 1 (the row allows a peak up to `1/c` times the limit): the
      number moves the stress limit of the user, a requirement (F4-28). `auto` and exactly 1 give no fact: `auto` is the default (the run log has the
      stage-start fact when the scale is far below 1), and 1 makes the row a bound of the peak at the limit as the user wrote it.
    - `settings.stress_step_acceptance: false` with a stress limit: the optimizer takes every MMA step.
    - `buckling.ks_q` without `buckling.min_factor`: no row exists. `buckling.modes` of 1 with the row: a repeated
      factor gives a wrong gradient. A mass run with the buckling row as the only row that holds material: a case
      with no compression has no buckling load.
    - A load case that repeats an earlier case, or is its exact opposite (F4-5, `loads.repeated_load_cases`): the rows that
      give the same value, the rows that follow the sign, the effect of the weights and limits, the cost and the options.
    - A load case that applies no load (U21, `loads.no_load_lines`): every force and moment is exactly 0 and no pressure is other than 0.
      What follows (nothing in the objective, a stress limit met by any design), the cost and the options. With `reach`
      (`Problem.load_reach`, F4-37) also a case whose load vector is exactly 0.0 on every free dof (its nodal forces cancel, or all its force lands on fixed dofs).
    - `settings.iterations` below the start of the last stage of `settings.beta_schedule` (F4-18, `verdict.beta_stage_fact`): the run cannot
      reach the last beta. The options and the price that was measured on the hook. `continued_from` (the run.json record of a continuation)
      changes the names in that fact: the pass count of a continuation is K + `more_iterations`, not `settings.iterations`.
    - `settings.part_only_min_film_mm` without `regions.not_printed` (P4b-6): no `part_only.stl` is made, so the key has no effect.
    - A design space STL whose largest size times 25.4 still fits the longer side of the print bed (TC-3, `unit_fact`): the file may be in
      inches or in metres. The sizes after x 25.4 and x 1000, the options. No number is added: the bed is `print.bed_mm`.
    """
    from .unit_fact import unit_fact_lines   # light module; the same text goes to the `warnings` of `check_data` and to the error of the loader

    st, lines = config.settings, []
    bk_on = config.buckling.min_factor is not None
    if config.buckling.ks_q is not None and not bk_on:
        lines.append("fact: buckling.ks_q is set, and buckling.min_factor is not set, so there is no buckling row and "
                     "ks_q has no effect. options: set buckling.min_factor (a multiplier on the load case) to turn "
                     "the row on, or remove buckling.ks_q")
    if bk_on and config.buckling.modes == 1:
        lines.append("fact: buckling.modes is 1 with the buckling row. At a repeated lowest factor (two modes with the "
                     "same factor) the KS over one mode has a wrong gradient: it follows one vector of the cluster. "
                     "options: use buckling.modes of 2 or more (the default is 6)")
    if st.objective == "mass":
        if not any(c.stress_limit_mpa is not None or c.compliance_limit is not None for c in config.load_cases):
            lim = config.limits
            if bk_on and not _a5_row_on_a_load_region(config):
                lines.append("fact: settings.objective is mass, and the buckling row is the only row that holds material "
                             "(no load case has a stress_limit or a compliance_limit). A load case with no compression has "
                             "no buckling load, so the row holds no material for that case and the mass can go to the "
                             "lowest density. The run goes on with the buckling rows and no volume ceiling. options: add "
                             "stress_limit (MPa) or compliance_limit (N*mm) to a load case, or use objective compliance "
                             "with a volume_fraction")
            elif bk_on:
                pass
            elif not (lim.displacement or lim.alignment):
                lines.append("fact: settings.objective is mass, and no load case has a stress_limit (or "
                             "allowable_mpa) or a compliance_limit. No row holds material, so the mass goes "
                             "to the lowest density. The run goes on with a volume ceiling at settings.volume_fraction. "
                             "options: add stress_limit (MPa) or compliance_limit (N*mm) to a load case, or use "
                             "objective compliance with a volume_fraction")
            elif not _a5_row_on_a_load_region(config):
                lines.append("fact: settings.objective is mass, and no load case has a stress_limit (or "
                             "allowable_mpa) or a compliance_limit, and no displacement or alignment limit has a "
                             "load region. No row holds material: a limit on a probe region does not, because the "
                             "optimizer can remove the material between the load and the probe, so the mass goes "
                             "to the lowest density. The run goes on with the displacement and alignment rows only. "
                             "options: add stress_limit (MPa) or compliance_limit (N*mm) to a load case, give a "
                             "limit a load region, or use objective compliance with a volume_fraction")
        for L in config.limits.displacement:
            if L.sense != "abs":
                lines.append(f"fact: settings.objective is mass, and displacement limit '{L.name}' has the sense "
                             f"{L.sense}. The row is also met when the region moves the other way, so it may not "
                             f"hold material. options: use the sense abs (the default) to limit the motion both "
                             f"ways, or add a stress_limit (MPa) or compliance_limit (N*mm) to a load case")
        if st.case_combine == "max":
            lines.append("fact: settings.case_combine has no effect with the mass objective")
        if st.perimeter_max_mm2 is not None:
            lines.append("fact: settings.objective is mass and settings.perimeter_max_mm2 is set. The perimeter row limits "
                         "the surface and not the material: in the problems that were measured the empty design had the smallest surface, so this "
                         "row does not hold material. A bound below the surface that the other rows give can make the optimizer add material. "
                         + PERIMETER_PRICE_FACT)
    has_stress = False
    for c in config.load_cases:
        if c.stress_limit_mpa is None:
            if c.allowable_source:
                lines.append(f"fact: load case '{c.name}' has allowable_source but no stress limit "
                             f"(no stress_limit, no allowable_mpa): there is no stress row, and the text is not used")
            continue
        has_stress = True
        if c.safety_factor is not None and c.safety_factor < 1.0:
            lines.append(f"fact: load case '{c.name}': safety_factor {c.safety_factor:g} is below 1, so the "
                         f"stress limit {c.stress_limit_mpa:g} MPa is above the allowable {c.allowable_mpa:g} MPa")
    if has_stress and isinstance(st.stress_scale, float):
        lines += stress_scale_facts(st.stress_scale)
    if has_stress and not st.stress_step_acceptance:
        lines.append("fact: settings.stress_step_acceptance is false: the optimizer takes every MMA step, "
                     "also a step that makes a stress row worse (the measured runs at 1.2 mm collapsed "
                     "and the 2.4 mm runs swung around the limit without this rule)")
    lines += repeated_load_lines(repeated_load_cases(config))   # F4-5: the same text goes to the warnings of `check_data`
    lines += no_load_lines(config, reach)   # U21 and F4-37: a load case that applies no load; the same text goes to the warnings of `check_data`
    if st.part_only_min_film_mm is not None and not config.not_printed:
        lines.append(f"fact: settings.part_only_min_film_mm is {st.part_only_min_film_mm:g} mm, and regions.not_printed is not set, so no "
                     "part_only.stl is made and the setting has no effect. options: list the region files that are not printed in "
                     "regions.not_printed, or remove settings.part_only_min_film_mm")
    if not fine:   # a fine pass has a maximum of passes, not a plan: its rule 'settled' does not need the last beta stage, and its facts name the schedule
        lines += beta_stage_fact(st.iterations, st.beta_schedule, continued_from)   # F4-18: the planned passes end before the last beta stage (the same text goes to `check_data`)
    lines += unit_fact_lines(config)   # TC-3: the design space may be in inches or in metres (the same text goes to `check_data`)
    return lines


@dataclass
class Result:
    """Output of `minimize`.

    x, rho        final design variables and physical density of the design voxels
    history       one dict for each pass (iteration 0 to `iterations`), like history.jsonl
    constraints   the constraint entries (name, value, limit, g, violation) of the last pass
    unmet         log lines of the constraints with g above `NOT_MET`
    mma           the `Mma` object. `mma.state` is the whole MMA state
    stages        the beta stage notes (`Formulation.stage_note`): only with a stage-aware row (A2)
    step_facts    log lines of step acceptance (rejected steps, one summary line): only with a stress row (A2)
    """

    x: np.ndarray
    rho: np.ndarray
    history: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    unmet: list = field(default_factory=list)
    mma: Mma | None = None
    stages: list = field(default_factory=list)
    step_facts: list = field(default_factory=list)


def minimize(model, formulation, *, x0=None, iterations=None, beta_schedule=None,
             move_limit=None, on_iteration=None, step_acceptance=None, objective_scale=None) -> Result:
    """Run MMA on `formulation` with the finite element `model`. For scripts.

    Defaults for `iterations`, `beta_schedule`, `move_limit` and the start design
    (an even density at the volume fraction) come from the YAML settings of the model.
    `on_iteration(record, point)` runs after each pass: `record` is the history dict,
    `point` the `DesignPoint`. Same steps as `run`, without a run folder.
    The last pass only measures. The caller frees the solver with `model.solver.free()`.
    `step_acceptance` (default: `settings.stress_step_acceptance`, which is True) turns the step
    acceptance of the stress rows on or off (`StepAcceptance`). A run with no stress row does not change.
    `objective_scale` (default: `settings.objective_scale`, which is "stage") is "stage" or "first": see `objective_scale_resets`.
    """
    st = model.problem.config.settings
    n_iter = st.iterations if iterations is None else iterations
    schedule = st.beta_schedule if beta_schedule is None else beta_schedule
    move = st.move_limit if move_limit is None else move_limit
    scale_mode = st.objective_scale if objective_scale is None else objective_scale
    x = np.full(model.n_design, st.volume_fraction) if x0 is None else np.asarray(x0, float).copy()
    mma = Mma(model.n_design, formulation.n_rows, move, xmin=formulation.xmin)
    rule = StepAcceptance(model, formulation, mma, move,
                          st.stress_step_acceptance if step_acceptance is None else step_acceptance)
    history, ev, rho, step_lines = [], None, None, []
    for it in range(n_iter + 1):
        t0 = time.time()
        beta = beta_at(schedule, it)
        rho_tilde, rho = model.density(x, beta)
        rho_stiff = model.stiffness_density(rho_tilde, rho, beta)   # eroded density in robust mode
        U = rule.solved(x, beta)
        if U is None:
            U = model.solve(model.stiffness(rho_stiff))
        point = model.design_point(x, beta, rho_tilde, rho, U, rho_stiff, iteration=it, last=it == n_iter)
        if objective_scale_resets(scale_mode, schedule, it):
            mma.state.f0_ref = None
        ev = formulation.evaluate(point, mma.state)
        step = None
        if it == n_iter:
            x_new = x
        elif rule.active:
            x_new, step = rule.step(x, ev, beta)
        else:
            x_new = mma.step(x, ev.f0, ev.df0, ev.g, ev.dg, ev.a0, ev.a, ev.c)
        record = {"iteration": it, "compliance": point.compliance,
                  "compliance_cases": [float(v) for v in point.compliance_cases],
                  "volume_fraction": float(rho.mean()), "beta": beta,
                  "change": float(np.abs(x_new - x).max()),
                  "t_iter_s": round(time.time() - t0, 3), **formulation.report(ev)}
        if step is not None:
            record["step_acceptance"] = step
            step_lines += rule.lines(it, step)
        note = formulation.stage_note(history[-1] if history else None, beta, it, ev.constraints)
        if note:
            record["stage"] = note
        history.append(record)
        if on_iteration is not None:
            on_iteration(record, point)
        x = x_new
    return Result(x, rho, history, ev.constraints, formulation.unmet(ev.constraints), mma,
                  [r["stage"] for r in history if "stage" in r], step_lines + rule.summary())
