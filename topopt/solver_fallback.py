"""GPU first, CPU fallback during a run (CONTRACTS.md section 20, "Fallback during a run").

`settings.solver: auto` picks gpu-mg on a usable GPU (`solver_choice.run_solver`). When gpu-mg fails during the run, the run goes on with the CPU
solver of the platform rule (`SolverChoice.fallback`). `run_with_fallback` does this for `cli._run_folder`:

1. It calls `optimize.run` with the solver that `run_solver` chose.
2. A `GpuFailure` (a device error, a device that is out of memory, a solve that is not accepted, an engine that cannot be made) ends that call. The
   device buffers are released. Any other error goes up as it is: an input error, `RunStopped`, Ctrl+C, a host `MemoryError`, an internal error.
3. The memory estimate of the CPU solver is compared with the RAM (`cli.estimate_memory_bytes`, the number of `topopt check`). If the estimate is above
   the RAM, the run stops with a `UserError` that names the facts and the options. The checkpoint stays. The estimate has one number for all CPU solvers
   (there is none for amg), so the check cannot look for another CPU solver that fits.
4. The run goes on from the last checkpoint with the CPU solver: the same run folder, `checkpoint.plan_resume` and `checkpoint.apply_plan`, then
   `optimize.run` again. This is not a model swap inside the loop: the restart reads the same state as a `--resume` does. A failure in the setup or in pass
   0 (no checkpoint yet) starts again at pass 0. A continuation (`--continue-from`) with no checkpoint in its own folder yet goes on with the same plan.
   `history.jsonl`, the snapshots and `gpu_mg.jsonl` keep the passes of the GPU part. The failed pass wrote no history line.
5. The record: `run.json` `solver` (the solver of the last part), `solver_setting` (`settings.solver`) and `solver_parts`; the same `solver_parts` in each
   checkpoint; one line in `run.log`; one note in `verdict.json`.

An explicit `settings.solver: gpu-mg`, and `auto` that picked a CPU solver, never fall back: the first stops with the error, as before.

`solver_parts` is a list with one dict for each part of the run (newest last):
`{"solver", "device", "first_pass", "last_pass", "end"}`. `device` is the OpenCL device name for gpu-mg, else None. `first_pass` is None for a part
that stopped in its setup. `last_pass` is the last pass that the part saved in a checkpoint. `end` is None for a part that ran to the end of the run, else
`{"kind", "text", "at_utc"}` (the kind of the `GpuFailure` and its text). A run with one solver has one part with `end` None. A resume, and a continuation, go on in the last
part when the solver and the device are the same.

This module imports no pyopencl, and `optimize` and `cli` only inside functions (`optimize` imports this module).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .config import UserError
from .gpu_mg.device import GpuFailure

TEXT_LIMIT = 600      # characters of the failure text that go into the record and the log line (the text of a device error is one line; the rest is cut)


def cut_text(text: str) -> str:
    """`text` in one line, at most `TEXT_LIMIT` characters. A longer text ends at its last full sentence inside the limit (a cut in the middle of a word
    would read as a fault); a text with no sentence end inside the limit is cut at the limit."""
    text = " ".join(str(text).split())
    if len(text) <= TEXT_LIMIT:
        return text
    head = text[:TEXT_LIMIT]
    end = head.rfind(". ")
    return head[:end + 1] if end > 0 else head


# --- the record of the parts ---------------------------------------------------------------------------------------

def part_start(parts: list, solver: str, device: str | None, first_pass: int) -> list:
    """A part of the run with `solver` starts at pass `first_pass`. Return `parts`.

    The last part goes on when it is the same solver and device and has no end. That is a resume, and a continuation (`--continue-from`) too: its `parts` start
    as the parts of the parent checkpoint, and a continuation with the same solver equals the run of full length, `run.json` included (CONTRACTS.md section 24).
    A new part starts when the solver or the device is another, or when the last part ended with a failure.
    """
    last = parts[-1] if parts else None
    if (last is not None and last["end"] is None and last["solver"] == solver and last.get("device") == device
            and last["first_pass"] is not None):
        return parts
    parts.append({"solver": solver, "device": device, "first_pass": int(first_pass), "last_pass": None, "end": None})
    return parts


def part_end(parts: list, solver: str, device: str | None, kind: str, text: str) -> list:
    """The part with `solver` ended with a failure of `kind`. Return `parts`.

    The last part is the one that ends. When it is another solver, or it has ended already, the part did not get as far as a pass (the setup failed): a part
    with `first_pass` None is added. `last_pass` is not changed here: `Checkpointer.save` sets it after each pass that the part saved.
    """
    end = {"kind": kind, "text": cut_text(text), "at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    last = parts[-1] if parts else None
    if last is None or last["end"] is not None or last["solver"] != solver:
        last = {"solver": solver, "device": device, "first_pass": None, "last_pass": None, "end": None}
        parts.append(last)
    last["end"] = end
    return parts


def _passes(part: dict) -> str:
    first, last = part.get("first_pass"), part.get("last_pass")
    if first is None or last is None:
        return "no pass"
    return f"pass {first}" if first == last else f"passes {first} to {last}"


def verdict_notes(parts) -> list[str]:
    """The notes for `verdict.json` (`notes`): one sentence when the run changed solver or a part ended with a failure, else an empty list."""
    if not parts or (len(parts) < 2 and parts[-1]["end"] is None):
        return []
    bits = []
    for p in parts:
        where = f" on {p['device']}" if p.get("device") else ""
        if p["end"] is not None:
            bits.append(f"{p['solver']}{where} ran {_passes(p)} and stopped ({p['end']['kind']})")
        else:
            bits.append(f"{p['solver']}{where} ran {_passes(p)}")
    return ["solver: " + "; ".join(bits) + ". The cause is in run.json (`solver_parts`) and in run.log."]


# --- the run -------------------------------------------------------------------------------------------------------

def reason_text(failure: GpuFailure) -> str:
    """The text of a failure for the log line and the record: one line, without the options (they are for a run with settings.solver gpu-mg)."""
    text = " ".join(str(failure).split())
    return cut_text(text.split(" Options:")[0]).rstrip(" .") if text else type(failure).__name__


def cpu_memory_problem(problem, cpu: str, choice=None) -> str | None:
    """The text of the stop when the CPU solver `cpu` does not fit in the RAM by the estimate of `topopt check`, else None.

    The estimate is `cli.estimate_memory_bytes` (grid, free dofs, filter, the bordered copy of K for joints). The RAM is `cli.physical_ram_bytes`
    (the host RAM, or the cgroup limit if it is lower). With no RAM size there is nothing to compare. The text also gives the gpu-mg estimate of the same run.
    """
    from . import cli
    from .members import filter_report
    from .optimize import reduced_system
    from .voxelize import DESIGN
    ram = cli.physical_ram_bytes()
    if not ram:
        return None
    edof, _active, free = reduced_system(problem)
    grid = int(math.prod(problem.shape))
    rep = filter_report(problem.config.settings, problem.h, int((problem.labels == DESIGN).sum()), problem.shape)
    need = cli.estimate_memory_bytes(grid, int(free.size), rep["size"]["held_bytes_estimate"], cli.bordered_bytes(problem, edof, free))
    if need <= ram:
        return None
    gpu = ""
    try:
        from .gpu_mg.check import problem_memory
        joints = 0 if problem.joints is None else len(problem.joints.joints)
        gpu = f" (the estimate for gpu-mg was {cli.gib(problem_memory(int(free.size), grid, getattr(choice, 'case', None), joints, rep['size'])['bytes'])})"
    except Exception:       # the gpu-mg number is a comparison only
        gpu = ""
    return f"the memory estimate of the CPU solver {cpu} is {cli.gib(need)}{gpu} and this machine has {cli.gib(ram)} RAM"


def run_with_fallback(problem, out, plan, choice, x0=None):
    """Run the optimizer (`optimize.run`) with the solver of `choice`; when gpu-mg fails and `choice` is an `auto` pick, go on with the CPU solver.

    `plan` is the `checkpoint.ResumePlan` of a resume or a continuation, or None. `choice` is the `solver_choice.SolverChoice` of the run. `x0` is None,
    or a function with no argument that makes the start design of a new run (the fine pass: `lambda: fine_pass.fine_start(setup)`). It is called for
    each start at pass 0, so a restart at pass 0 after a failure gets a new start (a `FineStart` keeps its log lines); a restart from a checkpoint
    reads the design and the fine-pass record from the checkpoint. Return what `optimize.run` returns. An error that is not a `GpuFailure`, and a
    `GpuFailure` of a run that is not an `auto` pick, go up as they are.
    """
    from . import optimize     # at call time, not at import: `optimize` imports this module
    use = {"solver": choice.solver} if choice.auto_gpu else {}      # only `auto` on a GPU names the solver: every other run keeps the old call

    def start(the_plan, **kw):
        if the_plan is not None:
            return optimize.run(problem, out, the_plan, **kw)
        return optimize.run(problem, out, **kw) if x0 is None else optimize.run(problem, out, x0=x0(), **kw)

    try:
        return start(plan, **use)
    except GpuFailure as failure:
        if not choice.auto_gpu:
            raise
        next_plan = switch(problem, out, plan, choice, failure)
    return start(next_plan, solver=choice.fallback)


def switch(problem, out, plan, choice, failure: GpuFailure):
    """Prepare the restart with the CPU solver after `failure`. Return the plan for `optimize.run` (None: start again at pass 0).

    It releases the device buffers, writes the record and the run-log line, checks the memory of the CPU solver (a `UserError` stops the run here), and, when
    the run folder has a checkpoint, plans and applies the resume of it.
    """
    from . import checkpoint
    cpu, gpu = choice.fallback, choice.solver
    if failure.solver is not None:
        try:
            failure.solver.free()
        except Exception:       # the buffers of a device that failed: nothing more to do, the process ends them
            pass
    where = "in the setup" if out.cur_step == "setup" else f"at pass {out.cur_iteration}"
    ck = checkpoint.describe(out.dir)
    saved = None if ck is None else int(ck["iteration"])
    device = (choice.device or {}).get("name")
    part_end(_parts(out), gpu, device, failure.kind, reason_text(failure))
    out.write_solver_record()
    reason = f"{failure.kind}: {reason_text(failure)}"
    from_parent = saved is None and plan is not None and plan.cont is not None      # a continuation with no checkpoint of its own yet: its plan is the start
    restart = (f"from the checkpoint of pass {saved}" if saved is not None else
               f"from the checkpoint of pass {plan.ck.iteration} of the parent run" if from_parent else "from pass 0")
    try:
        memory = cpu_memory_problem(problem, cpu, choice)
    except Exception as exc:        # the check must not hide the failure that started this: say that it did not run
        out.log(f"solver: the memory check of the CPU solver {cpu} did not run ({type(exc).__name__}: {exc}); the run goes on")
        memory = None
    if memory:
        stop = (f"solver gpu-mg stopped {where} ({reason}), and the run cannot go on with {cpu}: {memory}. "
                + (f"The checkpoint of pass {saved} stays in {out.dir}. " if saved is not None else "The run has no checkpoint. ")
                + "Options: resume with settings.solver set to 'amg' (amg makes no factor; give --allow-changed-inputs, MCP: allow_changed_inputs=true), "
                  "a larger settings.voxel_size, or a machine with more RAM")
        out.log("solver: " + stop)
        raise UserError(stop) from failure
    out.log(f"solver: gpu-mg stopped {where} ({reason}); the run goes on with {cpu} {restart}")
    if saved is not None:
        next_plan = checkpoint.plan_resume(out.dir, problem.config, problem, cpu)     # `auto` on both sides: the other solver is a note, not a change
        t_start = out.t_start
        checkpoint.apply_plan(next_plan, out, reason="solver_fallback")
        out.t_start = t_start       # the time of the pass that failed was run time: `apply_plan` set the time of the checkpoint
        return next_plan
    return plan if plan is not None and plan.cont is not None else None      # a continuation with no checkpoint of its own yet goes on with its plan


def stress_solver(config, solver: str | None = None):
    """The solver of a one-off stress solve (`refine/stress.compute_stress`) and its CPU fallback: `(name, fallback)`. `fallback` is None for no fallback.

    An explicit `settings.solver` is the solver (as before; gpu-mg has no fallback). For `auto`: `solver` is the solver that the run ended on
    (`run.json` `solver`, the solver of its last part). A CPU solver there is the solver of the stress solve: a run that went on with the CPU solver after
    a failure of gpu-mg does not try gpu-mg again in its post step. Else (gpu-mg, or no `solver`) the rule of a run holds: `solver_choice.run_solver`,
    that is gpu-mg on a usable GPU that passes the self-test (with the CPU rule as the fallback), else the CPU rule.
    """
    from .solver_choice import resolve_solver, run_solver
    setting = config.settings.solver
    if setting != "auto":
        return resolve_solver(setting), None
    if solver is not None and solver != "gpu-mg":
        return solver, None
    choice = run_solver(setting, config)
    return choice.solver, choice.fallback


def _parts(out) -> list:
    """The `solver_parts` list of the run writer (made when it is None)."""
    if out.solver_parts is None:
        out.solver_parts = []
    return out.solver_parts
