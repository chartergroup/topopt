"""`GpuModel`: the finite element `Model` with the matrix-free multigrid solver (settings.solver: gpu-mg).

The model does not assemble or factor the global stiffness matrix. `stiffness(rho)` returns the
stiffness factor of each element (a vector), and `solve(Ee)` makes the multigrid numbers for that
design on the CPU, uploads them to the OpenCL device, and runs the PCG. The rest of the model
(filter, density, compliance, chain rule) is the code of `Model`.

`m.solver` is a `GpuSolver`. It has `solve(B, warm=True)` and `free()` like `LinearSolver`, so
`DesignPoint.solve_adjoint` works. It solves with the operator of the last `Model.solve`, as the
factor of `LinearSolver` does. It has no `factor`.

`solve_design` is the one-off solve for the stress step of the post run (`refine/stress.py`): it
uses the same solver and makes no global matrix.

A device error (`pyopencl.Error`) during an upload or a solve ends as a `GpuFailure` (a `UserError`) with the facts
and the options, and the device buffers are released. The other failures of the solver are a `GpuFailure` too: a solve that is not accepted
(`SolveFailure`) and an engine that cannot be made (`GpuSolver.__init__`). `GpuFailure.kind` says which. A run with `settings.solver: auto` goes
on with the CPU solver after one (`topopt/solver_fallback.py`); a run with `settings.solver: gpu-mg` stops with the text.

The run folder gets `gpu_mg.jsonl`, one line for each solve. A new run starts a new file. A resume (D1b) goes on after the
lines of the first part: it cuts the file to the iteration of the checkpoint, like `history.jsonl`, and appends.

Two settings decide when a solve is done (G2, CONTRACTS.md section 20): `settings.gpu_tol` (the relative residual of each solve
in the optimizer loop) and `settings.gpu_backward_error` (a solve whose normwise backward error is at most this number is
accepted although its residual is above the target: the FP64 residual of any solver has a floor). A solve that fails writes its
record (the residual history and the backward error) to `gpu_mg.jsonl` before the `GpuFailure`.

A third setting, `settings.gpu_max_iter` (P2b), caps the PCG iterations of the FIRST pass of a case B solve (the FP64 outer PCG).
A restart step of case B and a refinement step of case C keep their own cap (`GpuParams.max_iter`, 300). Case C has no first pass.

Deflation of the rigid modes and of the bonded patches (P2d, `deflation.py`). Every case C solve with joints (a border with `k` > 0) removes two kinds of
column from the inner PCG and solves them in FP64 on the host: a method setting with no switch. The rigid columns are the rigid modes of the bodies of the
part (D3). The harmonic columns (D3h) are the rigid motions of the bonded patch of each joint, extended elastically into the part: `set_design` solves the
extension for each design with an FP64 PCG on the host and the V-cycle of the device as preconditioner, starting from the extension of the previous design.
Its tolerance is `deflation.EXTENSION_TOL`; its chunks are `GpuParams.max_iter` iterations long and it stops on stagnation as the refinement loop does.
`run.log`, the record of the design and `GpuSolver.facts()` say that it is on, the columns, the extension (iterations, time, warm or cold, the largest
final residual) and the memory. A device that cannot run it, or a design whose matrix `E` is not positive definite in FP64, runs without it (the code path
of before P2d), and `run.log` and the record say why.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..checkpoint import cut_history
from ..config import ALL_AXES, GPU_BACKWARD_ERROR_DEFAULT, GPU_MAX_ITER_DEFAULT, UserError
from ..filter import simp_E
from ..optimize import E_MIN, Model
from ..runinfo import read_history, write_atomic
from . import device as devmod
from . import kernels
from .border import grid_border
from .deflation import Deflation, DeflationError
from .device import GpuFailure
from .hierarchy import HierarchyError, HostMG, MIN_FREE_DOFS

SOLVER_NAME = "gpu-mg"
TIMING_FILE = "gpu_mg.jsonl"

# Facts about the floor of the backward error (CONTRACTS.md section 20). The first holds for parts without joints. With joints the host check has a
# rounding noise of its own (F1-7): in a rigid motion of the part the joint term of the operator is the difference of two terms that are far larger than
# the load. Since F1-7 that noise shows in the relative residual and not in the backward error (whose denominator holds the two large terms).
# These are measured facts of the revisions named in section 20, not limits.
FLOOR_FACT = "a solve at the FP64 rounding floor has a backward error of about 4e-17 to 2e-16"
JOINT_FLOOR_FACT = ("With joints the host check of the result subtracts two terms that are far larger than the load in a rigid motion of the part, so its "
                    "rounding shows in the relative residual: about 1.6 unit roundoffs of |Kb u| (5e-8 of the load on the two-joint part, one host). The backward "
                    "error of that rounding is inside the floor of parts without joints (4e-17 to 6e-17 on the two-joint part). The answer of a direct solver "
                    "(pardiso) on the same two-joint part had a backward error of 6e-16 to 4e-15 (one host). ")


@dataclass(frozen=True)
class GpuParams:
    """Numeric parameters. Recorded defaults of the G1b measurements, not user settings."""

    tol: float = 1e-8            # relative residual of the solve (FP64 true residual in case C). The loop of a run uses settings.gpu_tol (G2)
    backward_error: float = GPU_BACKWARD_ERROR_DEFAULT   # a solve at this normwise backward error is accepted. 0: off. settings.gpu_backward_error
    inner_tol: float = 1e-2      # case C: relative residual of each inner PCG on the device
    max_refine: int = 12         # case C: refinement steps
    max_iter_first: int = GPU_MAX_ITER_DEFAULT   # case B: PCG iterations of the FIRST pass of a solve (the FP64 outer PCG). settings.gpu_max_iter (P2b). Case C has no first pass
    max_iter: int = 300          # PCG iterations of each restart step (case B) and of each refinement step (case C): a recorded default, not a setting
    lmax_iters: int = 12         # power iterations for the Chebyshev bound
    min_free: int = MIN_FREE_DOFS
    nr_block: int = 2            # load cases for each kernel launch


class SolveFailure(GpuFailure):
    """A solve that did not reach its target (kind `not_accepted`). `its` is the record entry of the solve (residual history, backward error)."""

    def __init__(self, text, its):
        super().__init__(text, "not_accepted")
        self.its = its


def _top(values):
    """The largest finite value of a list (or array) as a float, or None when there is none."""
    vals = [float(v) for v in np.asarray(values, dtype=float).reshape(-1) if np.isfinite(v)]
    return max(vals) if vals else None


def _accepted_by_backward_error(rec) -> int:
    """How many solves of a `gpu_mg.jsonl` record were accepted by the backward error."""
    return sum(1 for it in rec.get("iterations", []) if it.get("accepted_by") == "backward_error")


def compliance_rise(rows, cases):
    """The load case whose compliance rose the most in the passes of `rows` (the records of `history.jsonl`, oldest first), from its lowest
    value to the last pass (#378). `cases` is a list of `(name, compliance_limit or None)` in the order of `compliance_cases`. Return
    `{"case", "low", "low_pass", "last", "last_pass", "limit", "factor"}`, or None when there are fewer than two passes or no compliance rose.
    It is a fact for the text of a failed solve: on a part with a per-axis support (#378, the phone dock) the compliance of a load that the
    support does not take rose 4.5 times in 4 passes before case C stopped, while the design came near a mechanism for that load."""
    series = []
    for r in rows:
        values = r.get("compliance_cases") or ([r["compliance"]] if "compliance" in r else None)
        if values is not None and "iteration" in r:
            series.append((int(r["iteration"]), [float(v) for v in values]))
    if len(series) < 2:
        return None
    best = None
    for c, (name, limit) in enumerate(cases):
        vals = [(it, v[c]) for it, v in series if c < len(v) and np.isfinite(v[c])]
        if len(vals) < 2:
            continue
        low_pass, low = min(vals, key=lambda p: p[1])
        last_pass, last = vals[-1]
        if low > 0 and last > low and (best is None or last / low > best["factor"]):
            best = {"case": name, "low": low, "low_pass": low_pass, "last": last, "last_pass": last_pass, "limit": limit,
                    "factor": last / low}
    return best


def rise_sentence(rise) -> str:
    """The sentence of `compliance_rise` for the text of a failed solve, or "" for None."""
    if rise is None:
        return ""
    lim = "" if rise["limit"] is None else f"; its compliance_limit is {rise['limit']:g}"
    return (f"The compliance of load case '{rise['case']}' rose from {rise['low']:.3g} at pass {rise['low_pass']} to {rise['last']:.3g} at pass "
            f"{rise['last_pass']} ({rise['factor']:.3g} times{lim}), the largest rise of the load cases in the passes so far. ")


def partial_supports(config) -> list[tuple[str, str]]:
    """The per-axis supports of a config: `(name of the fixed region, the axes it fixes)` for each `fixed` entry that is not a full clamp.
    A config without regions (the problem of the self-test, `selftest.run`) has none."""
    regions = getattr(config, "regions", None)
    if regions is None or not regions.fixed_axes:
        return []
    return [(Path(p).stem, "".join("xyz"[a] for a in ax)) for p, ax in zip(regions.fixed, regions.fixed_axes) if tuple(ax) != ALL_AXES]


def border_local_memory_problem(border, nr, case, dev) -> str | None:
    """The text of the stop when the joint kernel needs more local memory than the device has, else None (P2b review N4).

    The kernel `bord_w` keeps `D^-1 C^T x` of one launch in local memory: `nr` columns times the border dofs of the joints (6 for each joint),
    in the type of the outer PCG (64-bit in case B, 32-bit in case C). The device size is `CL_DEVICE_LOCAL_MEM_SIZE`; 0 means that the device
    gave no size, and then there is nothing to compare. Springs only (no border dofs) need none of it.
    """
    nk = 0 if border is None else int(border.k)
    if not nk or not dev.local_mem:
        return None
    isz = max(np.dtype(t).itemsize for t in kernels.kernel_dtype(case))
    need = nr * nk * isz
    if need <= dev.local_mem:
        return None
    return (f"solver 'gpu-mg': the {nk // 6} joints have {nk} border dofs, and the joint kernel keeps {nr} load case column(s) of them in the local "
            f"memory of one work-group: {nr} x {nk} x {isz} bytes = {need} bytes (case {case}). The OpenCL device {dev.name} has {dev.local_mem} "
            f"bytes of local memory. Options: use fewer joints, or {devmod.solver_options_text()}")


class GpuSolver:
    """Solver object of `GpuModel`. See the module text."""

    def __init__(self, free_dofs, edof, active_index, problem, dev, case, n_columns, params: GpuParams, out=None,
                 resume=None):
        from . import engine     # imports pyopencl: after `device.select` has checked it
        cl = self._cl = engine.cl
        self.dev, self.case, self.params, self.out = dev, case, params, out
        self.eroded_design = problem.config.settings.robust != "off"      # robust mode (A3b): the solved design is the eroded design
        # for the text of a failed solve (#378); a problem made by hand (the self-test, a test) may have no load cases or regions
        self._cases = [(c.name, c.compliance_limit) for c in getattr(problem.config, "load_cases", ())]
        self._partial = partial_supports(problem.config)
        self.n_free = int(free_dofs.size)
        try:
            self.host = HostMG(problem.shape, edof, active_index, problem.fixed_dofs, problem.h,
                               problem.config.material.nu, min_free=params.min_free,
                               border=grid_border(problem))       # springs and joints: Kb - C D^-1 C^T on every level (P2b)
        except HierarchyError as e:
            raise GpuFailure(f"solver 'gpu-mg': {e}. Options: {devmod.solver_options_text()}", "setup") from e
        dense = self.host.coarsest_dense_bytes(4)
        if dev.max_mem_alloc and dense > dev.max_mem_alloc:
            raise GpuFailure(
                f"solver 'gpu-mg': the dense inverse of the coarsest level needs {dense / 2**20:.0f} MB in one "
                f"buffer, and the largest buffer of the device {dev.name} is {dev.max_mem_alloc / 2**20:.0f} MB "
                f"(coarsest level: {3 * self.host.levels[-1].nn} dofs, {self.host.n_levels} levels). "
                f"Options: {devmod.solver_options_text()}", "setup")
        self.nr = max(1, min(params.nr_block, n_columns))
        local_text = border_local_memory_problem(self.host.border, self.nr, case, dev)
        if local_text:
            raise GpuFailure(local_text, "setup")
        self.defl = None            # the rigid modes (P2d): case C with joints only
        if case == "C" and self.host.border is not None and self.host.border.k:
            self.defl = Deflation(self.host)       # the bodies and the modes depend on the topology: made once
        t0 = time.perf_counter()
        try:
            self.engine = engine.GpuMG(self.host, dev.obj, case=case, nr=self.nr, deflation=self.defl)
        except cl.Error as e:
            raise GpuFailure(
                f"solver 'gpu-mg': the OpenCL device {dev.name} could not build or allocate the program "
                f"(case {case}, {self.n_free} free dofs): {' '.join(str(e).split())[:400]}. Options: set "
                f"settings.gpu_case or settings.gpu_device, or {devmod.solver_options_text()}", "setup") from e
        self.t_build = time.perf_counter() - t0
        self.defl_off = self.engine.defl_off       # why a deflation that the part needs is not in use (a text), or None
        if self.engine.defl is None:
            self.defl = None
        self._defl_logged = False
        self._ext_logged = False    # the line "the extension stopped above its tolerance" came (once for each run)
        self._defl_why = None       # why the last design ran without the deflation (the text of its `DeflationError`), or None
        L0 = self.host.levels[0]
        self.fmap = L0.lookup[free_dofs // 3].astype(np.int64) * 3 + free_dofs % 3
        if not (np.all(self.fmap >= 0) and (L0.fix.reshape(-1)[self.fmap] == 0).all()
                and self.fmap.size == int((L0.fix.reshape(-1) == 0).sum())):
            raise RuntimeError("gpu-mg: the free dofs of the model do not match the multigrid level 0 "
                               "(internal consistency error)")
        self.nn0 = L0.nn
        self.records: list[dict] = []
        self._pending: dict = {}
        self._design_open = False       # between set_design and end_design
        self._ready = False
        self._freed = False
        self._first = True
        self._n_design = 0              # design solves so far
        self._n_lines = 0               # records so far (design and adjoint)
        self._fh = None
        if out is not None:
            path = out.dir / TIMING_FILE
            if resume is None:
                self._fh = open(path, "w")      # a new run starts a new file
            else:
                # A continuation (P4b-5) of a complete checkpoint runs the pass of the checkpoint again: `kept_iteration` is the pass before it.
                self._keep_first_part(path, getattr(resume, "kept_iteration", resume.ck.iteration), out,
                                      getattr(resume, "kept_complete", bool(resume.ck.meta.get("complete"))))
                self._fh = open(path, "a")      # a resume goes on after the records of the first part (D1b)
        self._be_logged = any(_accepted_by_backward_error(r) for r in self.records)   # one run.log line for each run (a resume: the first part counts)
        self._summary_done = False
        self._n_checks = 0              # pass-1 true-residual checks so far (item 4)

    def _keep_first_part(self, path, last_iteration: int, out, complete: bool = False):
        """A resume: cut the timing file to the checkpoint iteration, like `history.jsonl`, and number the next records after it.

        The records of the passes after the checkpoint stay out: the resumed run solves those passes again. A line that
        does not parse (a partial last line) goes too. `line` and `solve` go on, so the file reads like the file of a run
        that was not stopped. A missing file is a fact: a new file starts.
        """
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)       # a kill inside an earlier cut leaves it
        cut = cut_history(path.parent, last_iteration, TIMING_FILE)
        final = 0
        if cut["present"]:
            lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
            self.records = [json.loads(line) for line in lines]
            if complete:       # the loop is over: a trial line of the last pass is the nominal solve of `robust.finish_run`, and this
                while self.records and self.records[-1]["kind"] == "trial" and self.records[-1]["iteration"] == last_iteration:
                    self.records.pop()      # run makes it again (the last pass has no update step, so it has no other trial line)
                    final += 1
                if final:
                    write_atomic(path, b"".join(lines[:len(lines) - final]))
        self._n_lines = 1 + max((r.get("line", -1) for r in self.records), default=-1)
        self._n_design = 1 + max((r.get("solve", -1) for r in self.records), default=-1)
        if not cut["present"]:
            out.log(f"resume: {TIMING_FILE} is missing: a new file starts")
            return
        plural = lambda n: f"{n} line" + ("" if n == 1 else "s")      # noqa: E731
        last = max((r["iteration"] for r in self.records), default=None)
        parts = [f"keeps {plural(len(self.records))} up to iteration {last}" if self.records else "keeps no line"]
        if cut["dropped"]:
            lo, hi = min(cut["dropped"]), max(cut["dropped"])
            parts.append(f"drops {plural(len(cut['dropped']))} for " + (f"iteration {lo}" if lo == hi else f"iterations {lo} to {hi}"))
        if final:
            parts.append(f"drops {plural(final)} of the solve at the end of the run (the checkpoint is from the last pass: the resumed "
                         "run makes it again)")
        if cut["bad_lines"]:
            parts.append(f"drops {cut['bad_lines']} line(s) that do not parse (a partial last line, or a damaged line)")
        if cut["newline_added"]:
            parts.append("adds the newline that the last kept line lacked (the kill came before it)")
        if last is None or last < last_iteration:
            parts.append("the file ends " + (f"at iteration {last}, " if last is not None else "with no line, ")
                         + f"before the checkpoint iteration {last_iteration}: the lines in between are not in the file")
        out.log("resume: " + TIMING_FILE + " " + ", ".join(parts) + "; the lines of this run follow")

    # ------------------------------------------------------------ one design
    def _iteration(self):
        """The optimizer iteration of the run writer (None without a run writer)."""
        it = getattr(self.out, "cur_iteration", None)
        return None if it is None else int(it)

    def _design_kind(self) -> str:
        """"main" for a design solve in the solve step of the loop, "trial" for one in another step
        (the step acceptance of A2 solves trial designs in the update step)."""
        step = getattr(self.out, "cur_step", None)
        return "main" if step in (None, "solve") else "trial"

    def _device_failure(self, e, doing: str):
        """Release the engine and raise the `GpuFailure` for an OpenCL error (facts and options). Its kind is `device_memory` when the device is out of memory."""
        text = " ".join(f"{type(e).__name__}: {e}".split())[:400]
        it = self._iteration()
        kind = "device_memory" if devmod.is_memory_error(self._cl, e) else "device_error"
        self.free()
        raise GpuFailure(
            f"solver 'gpu-mg': the OpenCL device {self.dev.name} reported an error when it {doing} "
            f"(case {self.case}, {self.n_free} free dofs, optimizer iteration {'?' if it is None else it}): {text}. "
            f"The device buffers were released. Options: set settings.gpu_case to 'C', set another "
            f"settings.gpu_device, or {devmod.solver_options_text()}", kind, solver=self) from e

    def set_design(self, Ee):
        """Make the numbers for the element stiffnesses `Ee` and upload them."""
        if self._freed:
            raise RuntimeError("gpu-mg has no design: free() was called")
        t0 = time.perf_counter()
        T = self.host.setup_host(Ee, lmax_iters=self.params.lmax_iters, reuse_power_vector=not self._first)
        self._first = False
        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        try:
            self.engine.upload_numerics(self.host)
        except self._cl.Error as e:
            self._device_failure(e, "copied the numbers of the design to the device")
        t_upload = time.perf_counter() - t0
        defl = self._set_deflation(T)       # D3h: after the numbers are on the device (the extension uses the V-cycle), before the first solve
        t_setup += T.get("defl_ext", 0.0) + T.get("defl", 0.0)
        t_upload += T.get("defl_upload", 0.0)
        self._ready = True
        self._design_open = True
        self._pending = {"kind": self._design_kind(), "iteration": self._iteration(), "setup_s": t_setup,
                         "setup_parts_s": {k: round(v, 5) for k, v in T.items()},
                         "upload_s": t_upload, "solves": 0, "solve_s": 0.0, "map_s": 0.0,
                         "iterations": []}
        if defl is not None:
            self._pending["deflation"] = defl

    def _set_deflation(self, T):
        """The deflation for the design that `setup_host` made and `upload_numerics` copied (P2d): the harmonic columns (the extension of the patch
        motions, D3h), the host numbers `AZ` and `E`, and the copy of `Z`, `AZ`, `E^-1` to the device. Return the fact for the record of the design (None
        when the part has no deflation to make: case B, no joints). The times go into `T["defl_ext"]`, `T["defl"]` and `T["defl_upload"]`. A design whose
        matrix `E` is not positive definite runs without the deflation; the next design tries again."""
        from . import engine
        if self.defl is None and self.defl_off is None:
            return None
        if self.defl is None:
            return {"on": False, "why": self.defl_off}
        d, g = self.defl, self.engine
        try:
            ext = None
            if d.n_harm:
                ext = d.extend(self.host, g.precondition, g.nr, self.params.max_iter, engine.STAGNATION_FALL)
                T["defl_ext"] = ext["seconds"]
            T["defl"] = d.set_design(self.host)
        except DeflationError as e:
            g.use_deflation(False)
            self._defl_why = str(e)
            if self.out is not None:
                self.out.log(f"gpu-mg: deflation of the rigid modes and patches is off for this design (optimizer iteration {self._iteration()}): {e}. "
                             "The inner PCG of this design runs as before; the next design tries the deflation again")
            return {"on": False, "why": str(e)}
        except self._cl.Error as e:
            self._device_failure(e, "ran the V-cycle for the harmonic columns of the deflation")
        g.use_deflation(True)
        self._defl_why = None
        t0 = time.perf_counter()
        try:
            g.upload_deflation()
        except self._cl.Error as e:
            self._device_failure(e, "copied the deflation columns of the design to the device")
        T["defl_upload"] = time.perf_counter() - t0
        fact = {"on": True, "bodies": d.nbodies, "columns": d.dk, "rigid_columns": d.n_rigid, "harmonic_columns": d.n_harm, "active_columns": d.k}
        if ext is not None:
            fact["extension"] = {**ext, "seconds": round(ext["seconds"], 5), "rel_max": float(f"{ext['rel_max']:.3g}")}
        self._log_extension(ext)
        return fact

    def _log_extension(self, ext):
        """`run.log` lines for the extension of the harmonic columns: the first design (cold), and the first design that stopped above the tolerance."""
        if self.out is None:
            return
        d = self.defl
        if not self._defl_logged:
            self._defl_logged = True
            host = f"host part ({d.k} products with the FP64 operator, matrix E) {d.t_design:.3f} s"
            if ext is None:
                self.out.log(f"gpu-mg: deflation of the first design: no harmonic column (no bonded patch has a free motion); {host}; the same part runs once "
                             "for each design")
            else:
                self.out.log(
                    f"gpu-mg: deflation of the first design: extension of {ext['columns']} harmonic column(s) of {len(d.patches)} bonded patch(es), "
                    f"{'warm' if ext['warm'] else 'cold'} start, {ext['iterations']} PCG iterations in {ext['chunks']} chunk(s) of up to "
                    f"{self.params.max_iter} ({ext['seconds']:.1f} s), largest relative residual {ext['rel_max']:.1e} against {ext['tol']:g}; {host}. "
                    "The same runs for each design, warm from the previous one")
        if ext is not None and not ext["converged"] and not self._ext_logged:
            self._ext_logged = True
            self.out.log(
                f"gpu-mg: the extension of the harmonic columns stopped above its tolerance in the design of optimizer iteration {self._iteration()}: "
                f"relative residual {ext['rel_max']:.1e} against {ext['tol']:g}, stopped on {', '.join(sorted(set(ext['stopped'])))}. The columns are still "
                "valid for the deflation (any columns are; only the speed changes). This line comes once for each run; the count comes at the end of the run")

    # ------------------------------------------------------------ solve
    def solve(self, B, warm: bool = True):
        """Solve K X = B for the operator of the last `set_design`. `warm` has no effect (cold start).

        A call after `end_design` is an adjoint solve (`DesignPoint.solve_adjoint`): it gets its own
        record, with kind "adjoint". A solve that fails writes its record before the `UserError` (G2).
        """
        if not self._ready:
            raise RuntimeError("gpu-mg has no design: call Model.solve (set_design) first, "
                               "or free() was called")
        B = np.asarray(B, dtype=np.float64)
        one = B.ndim == 1
        B = B.reshape(self.n_free, -1)
        k = B.shape[1]
        X = np.zeros((self.n_free, k))
        live = [j for j in range(k) if np.any(B[:, j])]
        rec = self._pending if self._design_open else {
            "kind": "adjoint", "iteration": self._iteration(), "setup_s": 0.0, "upload_s": 0.0, "solves": 0,
            "solve_s": 0.0, "map_s": 0.0, "iterations": []}
        for s in range(0, len(live), self.nr):
            chunk = live[s:s + self.nr]
            cols = chunk + [chunk[0]] * (self.nr - len(chunk))   # a short block repeats a live column
            t0 = time.perf_counter()
            Bn = np.zeros((self.nr, self.nn0 * 3))
            for i, j in enumerate(cols):
                Bn[i][self.fmap] = B[:, j]
            t_map = time.perf_counter() - t0
            t0 = time.perf_counter()
            try:
                Xn, its = self._solve_block(Bn.reshape(self.nr, self.nn0, 3))
            except self._cl.Error as e:
                self._device_failure(e, "solved the linear system")
            except SolveFailure as e:       # the record first (the residual history is the fact that the user needs), then the error
                rec["solves"] += 1
                rec["solve_s"] += time.perf_counter() - t0
                rec["map_s"] += t_map
                rec["iterations"].append(e.its)
                self._note_solve(rec, e.its)
                rec["failed"] = True
                self._close_record(rec)
                e.solver = self         # the fallback of `settings.solver: auto` releases the device buffers of this solver
                raise
            t_solve = time.perf_counter() - t0
            t0 = time.perf_counter()
            for i, j in enumerate(chunk):
                X[:, j] = Xn[i].reshape(-1)[self.fmap]
            t_map += time.perf_counter() - t0
            rec["solves"] += 1
            rec["solve_s"] += t_solve
            rec["map_s"] += t_map
            rec["iterations"].append(its)
            self._note_solve(rec, its)
        if not self._design_open and rec["solves"]:
            self._write(rec)
        return X[:, 0] if one else X

    def _note_solve(self, rec, its):
        """Put the backward-error facts of one solve into the record, and say once for each run that the rule accepted a solve."""
        if not its.get("failed"):       # a failed solve was accepted by nothing: the record has no `accepted_by` from it
            rec.setdefault("accepted_by", "tol")
        if its.get("accepted_by") == "backward_error":
            rec["accepted_by"] = "backward_error"
            if self.out is not None and not self._be_logged:
                self._be_logged = True
                p = self.params
                self.out.log(
                    f"gpu-mg: the backward-error rule accepted a solve (optimizer iteration {self._iteration()}): the relative residual "
                    f"reached {its['true_rel']:.2e}, the target is {p.tol:g} (settings.gpu_tol), the backward error is {its['eta']:.2e} "
                    f"(limit {p.backward_error:g}, settings.gpu_backward_error). " + self._floor_note()
                    + " This line comes once for each run; the count of such solves comes at the end of the run")
        for key in ("eta", "true_rel"):
            if its.get(key) is not None:
                rec[key] = max(rec.get(key, 0.0), its[key])

    def _floor_note(self) -> str:
        """The fact about the floor of the backward error for the run.log line (a fact, no claim about the design)."""
        bord = self.host.border
        if bord is not None and bord.k:
            return ("On the designs without joints measured so far a solve at the FP64 floor has a backward error of about 4e-17; this run has joints, "
                    "and the host check of a part that joints hold has a rounding noise of its own in the relative residual (about 1.6 unit roundoffs "
                    "of |Kb u|; 5e-8 of the load on the two-joint part), so the residual reached can be that noise and not the error of the solve "
                    "(CONTRACTS.md section 20).")
        return "On the designs measured so far a solve at the FP64 floor has a backward error of about 4e-17."

    def _close_record(self, rec):
        """Write the record of a solve that failed (a design record is closed here, as `end_design` does)."""
        if self._design_open:
            self._pending = {}
            self._design_open = False
            self._write(rec)
            self._n_design += 1
        else:
            self._write(rec)

    def _solve_block(self, Bn):
        """Solve one block of `nr` columns. Return `(X, its)`; `its` is the entry of the record. Case C: the refinement loop. Case B:
        the 64-bit PCG, and when it does not reach the tolerance in `max_iter_first` iterations (settings.gpu_max_iter), the same
        refinement loop (a restart, each step capped at `max_iter`) from its result. A PCG that converges in the first pass gives the same result as without the loop.

        G2: the target is `params.tol`; with `params.backward_error` > 0 a solve is also accepted at that backward error, and a
        first pass that stopped on its recurrence residual gets a check of the true residual (`engine.check_solution`)
        and goes on in the refinement loop when the true residual is above the target and the backward error above the limit.
        A solve that is not accepted raises `SolveFailure` (a `UserError`) with the entry of the record.
        """
        from . import engine
        pr, g, limit = self.params, self.engine, self.params.backward_error
        if self.case == "C":
            X, info = engine.solve_refined(g, self.host, Bn, pr.tol, pr.inner_tol, maxref=pr.max_refine,
                                           maxit=pr.max_iter, backward_error=limit, last_step_margin=engine.LAST_STEP_MARGIN)
            its = {"inner": info["inner_iters"], "refine_steps": info["steps"]}
            used = f"up to {pr.max_iter} PCG iterations in each of {info['steps']} refinement steps"
        else:
            X, first = g.pcg(Bn, pr.tol, maxit=pr.max_iter_first)
            its = {"pcg": first["iters"], "executed": first["executed"]}
            used = f"{first['executed']} PCG iterations (the first pass is capped at {pr.max_iter_first}, settings.gpu_max_iter)"
            if first["converged"]:
                if limit <= 0:
                    its["accepted_by"] = "tol"      # the rule is off: the recurrence residual decides, as at d19c7fb
                    return X, its
                chk = engine.check_solution(self.host, Bn, X, pr.tol, limit)
                self._n_checks += 1
                its.update(true_rel=_top(chk["rel"]), check_s=round(chk["seconds"], 5))
                if not np.all(np.isnan(chk["eta"])):
                    its["eta"] = _top(chk["eta"])
                if chk["ok"]:
                    its["accepted_by"] = chk["accepted_by"]
                    return X, its
                its["pass1_true_rel"], its["pass1_eta"] = its.pop("true_rel"), its.pop("eta", None)
                used += f" (stopped on the recurrence residual; the true residual was {its['pass1_true_rel']:.2e})"
            X, info = engine.solve_refined(g, self.host, Bn, pr.tol, pr.inner_tol, maxref=pr.max_refine,
                                           maxit=pr.max_iter, u0=X, backward_error=limit, last_step_margin=engine.LAST_STEP_MARGIN)
            its.update({"inner": info["inner_iters"], "refine_steps": info["steps"]})
            used += f", then {info['steps']} restart steps of up to {pr.max_iter} PCG iterations"
        # the end of the refinement loop: the true residual of the result
        reached = _top(info["rel_hist"][-1])
        its["true_rel"] = reached
        etas = [v for v in (info["eta_hist"][-1] or []) if v is not None]
        if etas:
            its["eta"] = max(etas)
        if info["converged"]:
            its["accepted_by"] = info["accepted_by"]
            return X, its
        its.update(failed=True, rel_hist=[_top(v) for v in info["rel_hist"]],
                   eta_hist=[None if e is None else _top([v for v in e if v is not None] or [None]) for e in info["eta_hist"]],
                   stagnated=info["stagnated"], nonfinite=info["nonfinite"])
        raise SolveFailure(self._failure_text(reached, its.get("eta"), used, info["stagnated"], info["nonfinite"]), its)

    def _deflation_clause(self) -> str:
        """The text for the failure of a case C joint solve: was the rigid motion removed from the inner PCG (P2d)? A fact, no claim about the cause."""
        if self.defl is not None and self.engine.defl_active:
            d = self.defl
            nb, ext = d.nbodies, d.ext
            text = (f"; the inner PCG deflated the rigid modes ({nb} {'body' if nb == 1 else 'bodies'}, {d.n_rigid} columns)"
                    + (f" and {d.n_harm} harmonic columns of {len(d.patches)} bonded patch(es) (the patch motions extended into the part)" if d.n_harm else "")
                    + " and the host solved them in FP64")
            if ext is not None and not ext["converged"]:
                text += (f"; the extension of the harmonic columns stopped above its tolerance in the last design (residual {ext['rel_max']:.1e} against "
                         f"{ext['tol']:g}, {', '.join(sorted(set(ext['stopped'])))})")
            return text
        why = self.defl_off or self._defl_why or "it was switched off for this engine"
        return f"; the deflation of the rigid modes was not used: {why}"

    def _failure_text(self, reached, eta, used, stagnated, nonfinite=False) -> str:
        """The error of a solve that was not accepted: the facts, then the levers as options (no order)."""
        pr = self.params
        be = (f"backward error {eta:.2e} (limit {pr.backward_error:g})" if eta is not None
              else "the backward error was not computed" + ("" if pr.backward_error > 0 else " (the rule is off)"))
        stag = ("; the loop stopped early: the residual fell by less than 10 % over two steps" if stagnated else "")
        stag += ("; the iteration gave a value that is not finite (the PCG broke down)" if nonfinite else "")
        bord = self.host.border
        if bord is not None and bord.k and self.case == "C":       # joints in 32-bit arithmetic (P2b)
            soft = bord.facts.get("beam_over_bond_min")
            stag += ("; the operator of a joint run is a difference of two terms of the size of the bond stiffness in a rigid motion of the body, "
                     "and case C computes it in 32-bit float, which rounds at 6e-8 of the bond term"
                     + (f"; the beam is {soft:.1e} of the bond stiffness on its softest mode" if soft is not None else "")
                     + self._deflation_clause())
        robust = "; robust mode: the solved design is the eroded design" if self.eroded_design else ""
        rule = (f"a larger settings.gpu_backward_error (now {pr.backward_error:g}; 0 turns the rule off)" if pr.backward_error > 0
                else "a settings.gpu_backward_error above 0 (now 0: the rule is off)")
        case = "settings.gpu_case 'B' (this device has 64-bit float); " if self.case == "C" and self.dev.fp64 else ""
        beam = "a stiffer beam (a larger section, a shorter beam); " if bord is not None and bord.k and self.case == "C" else ""
        cap = (f"a larger settings.gpu_max_iter (the cap of the first pass, now {pr.max_iter_first}); " if self.case == "B" else "")
        if eta is None:
            floor = ""
        elif bord is not None and bord.k:
            floor = JOINT_FLOOR_FACT
        else:
            floor = f"On the designs without joints measured so far {FLOOR_FACT}; a larger value means that the iteration did not converge. "
        # The design facts of #378: a load case whose compliance rose (the design comes near a mechanism for it), and the per-axis supports.
        rise = self._compliance_rise()
        design = rise_sentence(rise)
        if self._partial:
            design += "The part has per-axis supports: " + ", ".join(f"{n} ({ax})" for n, ax in self._partial) + ". "
        limit = (f"a lower compliance_limit for load case '{rise['case']}' (now {rise['limit']:g}); "
                 if rise is not None and rise["limit"] is not None else "")
        clamps = (f"full clamps in place of the per-axis supports ({', '.join(n for n, _ in self._partial)}), if the part is held that way in use; "
                  if self._partial else "")
        return (f"solver 'gpu-mg': the solve did not reach the relative residual {pr.tol:g} (target; settings.gpu_tol in the loop). "
                f"Reached {'nan' if reached is None else format(reached, '.2e')}, {be}; case {self.case}, device {self.dev.name}, {used}{stag}{robust}. "
                f"{floor}{design}Options: a larger settings.gpu_tol; {rule}; {cap}a stiffer start design "
                f"(a lower settings.penal or a higher settings.volume_fraction); {limit}{clamps}{beam}{case}or {devmod.solver_options_text()} "
                f"(with 'auto' a gpu-mg failure does not end the run: it goes on with the CPU solver from the last checkpoint)")

    def _compliance_rise(self):
        """`compliance_rise` of the passes in `history.jsonl` of the run folder, or None (no run writer, no file, or a read error)."""
        if self.out is None:
            return None
        try:
            return compliance_rise(read_history(Path(self.out.dir) / "history.jsonl", strict=False).rows, self._cases)
        except (OSError, ValueError, TypeError, KeyError):     # a missing or odd file: the text goes without the fact (a MemoryError goes up)
            return None

    # ------------------------------------------------------------ records
    def _write(self, rec):
        """Append a record of a solve to the list and to the run file."""
        design = self._n_design - 1 if rec["kind"] == "adjoint" else self._n_design   # the design that the solve used
        rec = {"line": self._n_lines, "kind": rec["kind"], "iteration": rec["iteration"], "solve": design,
               "case": self.case, **{k: v for k, v in rec.items() if k not in ("kind", "iteration")}}
        rec["total_s"] = round(rec["setup_s"] + rec["upload_s"] + rec["solve_s"] + rec["map_s"], 5)
        for k in ("setup_s", "upload_s", "solve_s", "map_s"):
            rec[k] = round(rec[k], 5)
        self._n_lines += 1
        self.records.append(rec)
        if self._fh is not None:
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        return rec

    def end_design(self):
        """Close the record of the design: append it to the list and to the run file."""
        rec = self._pending
        self._pending = {}
        self._design_open = False
        if not rec:
            return None
        out = self._write(rec)
        self._n_design += 1
        return out

    def facts(self) -> dict:
        g = self.engine
        return {"device": self.dev.facts(), "case": self.case, "levels": self.host.n_levels,
                "level_free_dofs": [int(3 * L.nn - L.fix.sum()) for L in self.host.levels],
                "work_group": g.wg, "work_group_vector": g.wgv, "columns_per_launch": self.nr,
                "device_bytes": int(g.nbytes), "device_bytes_parts": dict(g.nbytes_parts),
                "build_s": round(self.t_build, 3), "params": self.params.__dict__,
                "border": None if self.host.border is None else self.host.border.summary(),
                "deflation": self.deflation_facts()}

    def deflation_facts(self) -> dict | None:
        """Facts about the deflation (P2d), or None when the part has none to make (case B, no joints). `on`: the engine holds it. `extension`: the facts of
        the last extension of the harmonic columns (None before the first design), `extension_total`: the sums over the designs so far. `device_bytes` and
        `host_bytes` are the memory of `Z`, `AZ` and the rest."""
        if self.defl is None and self.defl_off is None:
            return None
        if self.defl is None:
            return {"on": False, "why": self.defl_off}
        d = self.defl
        out = {"on": True, **d.summary(), "work_group": self.engine.wd, "device_bytes": int(self.engine.nbytes_parts.get("defl", 0)),
               "extension_tol": d.ext["tol"] if d.ext is not None else None, "extension": d.ext, "extension_total": dict(d.ext_total)}
        return out

    def free(self):
        """Release the device buffers and close the run file. A second call does nothing."""
        self._ready = False
        self._design_open = False
        self._say_count()
        eng = getattr(self, "engine", None)
        if eng is not None and not self._freed:
            self._freed = True
            eng.release()
            if self.defl is not None:        # the host arrays of the deflation too (357.5 MiB at 1.0 mm on the two-joint part)
                self.defl.release()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _say_count(self):
        """One line in `run.log` at the end of the run (once): how many solves the backward-error rule accepted. Nothing if none."""
        if self._summary_done:
            return
        self._summary_done = True
        d, tot = self.defl, None if self.defl is None else self.defl.ext_total
        if d is not None and tot["designs"] and self.out is not None:
            self.out.log(f"gpu-mg: extension of the harmonic columns over the run: {tot['designs']} design(s), {tot['warm']} warm, {tot['iterations']} PCG "
                         f"iterations, {tot['seconds']:.1f} s, {tot['above_tol']} above the tolerance {d.ext['tol']:g}, largest final relative residual "
                         f"{tot['rel_max']:.1e}")
        n = sum(_accepted_by_backward_error(r) for r in self.records)
        if n and self.out is not None:
            total = sum(len(r.get("iterations", [])) for r in self.records)
            p = self.params
            self.out.log(f"gpu-mg: {n} of {total} solves were accepted by the backward error (settings.gpu_backward_error "
                         f"{p.backward_error:g}); the others by the relative residual {p.tol:g} (settings.gpu_tol)")

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class GpuModel(Model):
    """The `Model` of the solver gpu-mg. `out` is the run writer (or None): the model logs facts to it.

    `resume` (a `checkpoint.ResumePlan`, or None for a new run) is the state that `optimize.run` has: with it the timing
    file goes on, without it a new file starts.
    """

    def __init__(self, problem, out=None, params: GpuParams | None = None, resume=None):
        self._out = out
        st = problem.config.settings
        self._params = params or GpuParams(tol=st.gpu_tol, backward_error=st.gpu_backward_error,
                                           max_iter_first=st.gpu_max_iter)   # G2, P2b: the loop takes the settings
        self._resume = resume
        super().__init__(problem)
        self.device_name = self.solver.dev.name
        if problem.springs is not None or problem.joints is not None:
            self.kb = self.solver.host.border      # not None: the loop and robust.finish_run then write the reactions (the solver has the support terms inside)

    def checkpoint_meta(self) -> dict:
        """The stop numbers of the loop and the first-pass cap, for `meta_json` of a checkpoint (G2, P2b): `checkpoint.plan_resume` compares
        them on a resume."""
        return {"gpu_mg": {"tol": self._params.tol, "backward_error": self._params.backward_error,
                           "max_iter": self._params.max_iter_first}}

    # hooks of Model.__init__
    def _make_assembler(self, edof):
        return None              # nothing assembles the global matrix

    def _make_solver(self, edof):
        st = self.problem.config.settings
        try:
            dev, case = devmod.select(st.gpu_device, st.gpu_case)
        except GpuFailure:
            raise
        except UserError as e:       # the device is gone, or it lacks a feature of the engine: the engine cannot be made (setup)
            raise GpuFailure(str(e), "setup") from e
        solver = GpuSolver(self.free_dofs, edof, self.active_index, self.problem, dev, case, self.f_red.shape[1], self._params,
                           self._out, self._resume)
        if self._out is not None:
            f = solver.facts()
            self._out.log(
                f"gpu-mg: device {dev.name} ({dev.kind}, platform {dev.platform}, {dev.version}), case {case} "
                f"({'FP32 V-cycle and FP64 outer PCG on the device' if case == 'B' else 'FP32 on the device, FP64 refinement on the CPU'})"
                f", {f['levels']} levels, work-group {f['work_group']}/{f['work_group_vector']}, "
                f"device buffers {f['device_bytes'] / 2**20:.0f} MB, tolerance {self._params.tol:g}, "
                f"backward error {self._params.backward_error:g}, "
                + (f"PCG cap {self._params.max_iter_first} in the first pass (settings.gpu_max_iter) and {self._params.max_iter} in a restart step"
                   if case == "B" else f"PCG cap {self._params.max_iter} in a refinement step (settings.gpu_max_iter does not apply to case C)"))
            b = f["border"]
            if b:      # springs and joints (P2b): the terms are in every level
                soft = b.get("beam_over_bond_min")
                self._out.log(
                    f"gpu-mg: support terms on all {f['levels']} levels (Galerkin products of the transfer): {b['springs']} spring region(s), "
                    f"{b['joints']} joint(s) ({b['border_dofs']} border dofs), bonded nodes on each level {b['nodes']}"
                    + (f"; the beam is {soft:.1e} of the bond stiffness on its softest mode" if soft is not None else ""))
            d = f["deflation"]
            if d is not None and d["on"]:       # P2d: a method setting, on for every case C solve with joints
                harm = (f" and {d['harmonic_columns']} harmonic column(s) {d['columns_per_patch']} of {d['patches']} bonded patch(es) (the rigid motions of the "
                        f"patch extended elastically into the part: an FP64 PCG on the host with the device V-cycle as preconditioner, for each design, "
                        f"warm from the previous design; chunks of {self._params.max_iter} iterations; the line of the first design gives the tolerance that the "
                        "extension used)"
                        if d["harmonic_columns"] else "")
                self._out.log(
                    f"gpu-mg: deflation of the rigid modes in the inner PCG is on (case C with joints, a method setting): {d['bodies']} bod"
                    f"{'y' if d['bodies'] == 1 else 'ies'}, {d['rigid_columns']} rigid mode column(s) {d['columns_per_body']}{harm}; {d['columns']} columns in all; "
                    f"the host part takes {d['topology_s']:.3f} s once (bodies, modes, QR) and then, for each design, {d['columns']} products with the FP64 operator; "
                    f"the device holds {d['device_bytes'] / 2**20:.1f} MB for the deflation (Z, AZ, E^-1 and the work buffers of its three kernels), the host "
                    f"{d['host_bytes'] / 2**20:.1f} MB (Z and AZ in 64-bit and 32-bit float, the prescribed values, the harmonic columns); the columns are solved in FP64 "
                    "on the host and removed from the inner PCG")
            elif d is not None:
                self._out.log(f"gpu-mg: deflation of the rigid modes in the inner PCG is off (case C with joints): {d['why']}. The inner PCG runs as "
                              "before; a joint with a soft beam can stop it (see the options of the error)")
        return solver, SOLVER_NAME

    # replaces stiffness and solve of Model
    def stiffness(self, rho):
        """The stiffness factor of each element (a vector). No matrix is assembled."""
        Ee = np.full(self.n_el, self.E)
        Ee[self.design_pos] = self.E * simp_E(rho, self.penal, 1.0, E_MIN)
        return Ee

    def solve(self, Ee):
        """Displacements of the free dofs for all load cases, for the element stiffnesses `Ee`."""
        self.solver.set_design(Ee)
        U = self.solver.solve(self.f_red)
        self.solver.end_design()
        return U


def solve_design(problem, edof, active_index, free_dofs, Ee, f_red, params: GpuParams | None = None, log=None):
    """Displacements of the free dofs for the element stiffnesses `Ee` and the loads `f_red` (n_free, k).

    One design, no run writer and no global matrix: the solve of the stress step in `refine/stress.py`
    when `settings.solver` is gpu-mg. The device buffers are released before the function returns.

    Each call builds a new `GpuSolver`, so the extension of the harmonic columns (case C with joints, P2d-D3h) starts cold: it has no design before it. It is
    the larger part of the cost of the call. `log` (a function of one line, or None) gets one line with its iterations and seconds when the call made an
    extension; the stress report gives it a writer of `run.log`. No other output: no file and no key of a JSON file.
    """
    st = problem.config.settings
    try:
        dev, case = devmod.select(st.gpu_device, st.gpu_case)
    except GpuFailure:
        raise
    except UserError as e:       # the device is gone, or it lacks a feature of the engine (setup): `settings.solver: auto` goes on with the CPU solver
        raise GpuFailure(str(e), "setup") from e
    # The tolerance of this one-off solve stays `GpuParams.tol` (1e-8); the backward-error rule and the first-pass cap are the settings (G2, P2b).
    params = params or GpuParams(backward_error=st.gpu_backward_error, max_iter_first=st.gpu_max_iter)
    solver = GpuSolver(free_dofs, edof, active_index, problem, dev, case, f_red.shape[1], params)
    try:
        solver.set_design(Ee)
        ext = solver.defl.ext if solver.defl is not None else None
        if log is not None and ext is not None:       # before the solve: the cost is paid also when the solve fails
            log(f"gpu-mg: extension of the {ext['columns']} harmonic column(s) of the deflation for this one solve, {'warm' if ext['warm'] else 'cold'} start: "
                f"{ext['iterations']} PCG iterations in {ext['chunks']} chunk(s), {ext['seconds']:.1f} s, largest relative residual {ext['rel_max']:.1e} "
                f"against {ext['tol']:g}")
        return solver.solve(f_red)
    finally:
        solver.free()
