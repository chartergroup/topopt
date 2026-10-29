"""Command line: `topopt run | dashboard | refine | check | render | verify | printcheck | step | report | doctor | gpu-check` (CONTRACTS.md sections 8, 9.2, 15, 16, 20, 32 and 41)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from . import __version__
from . import robust
from .config import ENVELOPE_FILL_DEFAULT, ENVELOPE_FILL_VALUES, UserError, load_config
from .estimate_bounds import derived_keys, stress_runs_text
from .region_rules import MIN_REGION_THICK, MIN_REGION_VOXELS  # noqa: F401  (one place for the two numbers, with regions_builder)
from .members import filter_lines, filter_report, filter_size, min_feature_line, size_text
from .solver_choice import (AMG_TIME_FACTOR, EXPERIMENTAL_SOLVERS, SOLVER_SETTINGS, platform_hint, platform_name,
                            recommendation, resolve_solver, run_solver, shim_build_command, shim_warning_line,
                            warning_line)
from .volume_basis import volume_target  # noqa: F401  (I7: the rule moved to topopt/volume_basis.py, with its inverse; `cli.volume_target` stays)

# Time model from CONTRACTS.md section 11: 1.30 s per iteration at 109,548 free dofs.
# It is fitted to one M1 run with the Accelerate Cholesky solver.
REF_DOFS, REF_S, DOF_POWER = 109_548, 1.30, 2.4
SETUP_S, POST_S = 10.0, 15.0


THREAD_VARS = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")

# Stress rows (A2). One adjoint solve of a stress row reuses the factor of the pass. Measured on the Linux build host
# (PARDISO, hook, uniform density 0.3, median of 5): 0.04 to 0.21 of one factor + solve of the 2 cases (STRESS_K_DIRECT is
# the high end). With AMG every adjoint is a full AMG solve (K = 1, not measured). The KS aggregate, the strain and the
# gradients cost about 0.5 microseconds per design voxel and row (STRESS_AGG_S, an estimate from the same runs).
# Step acceptance adds a trial solve for each rejected try: 131 of 220 passes at 2.4 mm and 94 of 160 at 1.2 mm
# (REVIEW) gave 0.6 tries per pass; STEP_TRIES_PER_PASS is the value measured again in FIXES-R1.
STRESS_K_DIRECT, STRESS_K_AMG = 0.2, 1.0
STRESS_AGG_S = 0.5e-6
STEP_TRIES_PER_PASS = 0.6
CHECK_FACT_LINES = ("gpu-mg models: ",)      # lines of `d["warnings"]` that `topopt check` prints as they are, without "warning: "
# The high end of the time range for a run with stress rows (A2 R1-2). It is a fixed round number, not a bound. It was chosen from
# three PARDISO runs on Linux x86_64 against the model total of this file: the hook example at 2.4 mm took 44 to 62 s against 32 s,
# and the hook at 1.2 mm with stress rows took 731 s (8 threads) and 845 s (4 threads) against 509 s. In those runs the rejected
# tries of the step acceptance made a pass 50 to 55 % longer than a pass with one try. Experiment E1 (run-time study, 2026-10-23;
# one shared Linux host) measured more: 53 finished PARDISO runs with stress rows in 6 problems had a median of 2.23 times
# the model total, and 70 % took more than 1.6 times it. `tshirt_hook_stress` (36 runs) had a median of 2.63 times, a part with two tube joints
# (14 runs) a median of 0.46 times. The numbers are in `estimate_bounds.STRESS_RUNS`. The factor stays 1.6 (a change of it is a
# change of the time range and of the budget warning of `settings.stop_rules.time_budget_s`).
# The model has no fixed cost for each pass (older than A2), so a small problem is under-estimated more.
STRESS_RUN_TIME_FACTOR = 1.6


def stress_seconds(per_it: float, n_rows: int, n_design: int, solver: str = "cholesky",
                   step_acceptance: bool = True) -> float:
    """Estimate of the extra seconds in one pass for `n_rows` stress rows.

    `per_it * k * n_rows` (adjoint solves) + `n_rows * STRESS_AGG_S * n_design` (KS and gradients)
    + `STEP_TRIES_PER_PASS * per_it` (rejected tries of the step acceptance). `per_it` is the model time of
    one pass without stress rows. An estimate: the model has no fixed cost per pass, so a small problem is under-estimated.
    """
    if n_rows <= 0:
        return 0.0
    k = STRESS_K_AMG if solver in ("amg", "gpu-mg") else STRESS_K_DIRECT     # no factor to reuse: every adjoint is a full solve (not measured)
    extra = per_it * k * n_rows + n_rows * STRESS_AGG_S * n_design
    return extra + (STEP_TRIES_PER_PASS * per_it if step_acceptance else 0.0)


def gpu_mg_estimate(n_free: int, iterations: int, gpu=None, stress_rows: int = 0, n_design: int = 0,
                    step_acceptance: bool = True, robust_mode: bool = False):
    """The time estimate of the solver gpu-mg as a `gpu_mg.check.GpuEstimate` (the mid number, the low and high band, the assumptions).

    `gpu` is a `gpu_mg.check.TimeInputs` (the load cases, `volume_fraction`, `penal`, `gpu_tol` and the device); without it the
    model has one load case, no contrast and no device (the reference constants). The model is linear in the free dofs (P2a).
    Stress rows and robust mode are added with the terms of the other solvers (`stress_seconds`, `robust.time_factor`).
    """
    from .gpu_mg.check import TimeInputs, estimate_seconds_gpu_mg
    gpu = gpu or TimeInputs()
    extra = (lambda per_it: stress_seconds(per_it, stress_rows, n_design, "gpu-mg", step_acceptance)) if stress_rows else None
    return estimate_seconds_gpu_mg(
        n_free, gpu.n_cases, iterations, volume_fraction=gpu.volume_fraction, penal=gpu.penal, gpu_tol=gpu.gpu_tol,
        device=gpu.device, case=gpu.case, k_other=gpu.k_other,
        robust_factor=robust.time_factor(iterations) if robust_mode else 1.0, extra_s_per_pass=extra)


def estimate_seconds(n_free: int, iterations: int, solver: str = "cholesky", stress_rows: int = 0,
                     n_design: int = 0, step_acceptance: bool = True, robust_mode: bool = False, gpu=None):
    """Return `(s_per_iteration, total_s)` for the run.

    The factors for the other solvers are rough:

    - "cholesky": the M1 fit above.
    - "pardiso": the same model. The hook example took 99 s on a 16 cores / 22 threads Linux x86_64 machine, and the model says
      145 s (148 s on the M1 Mac). The 99 s is the best of 62 runs of the default hook on a shared Linux host, not a typical run:
      minimum 98.8 s, median 462 s, 90th percentile 743 s, maximum 1,469 s (experiment E1, run-time study, 2026-10-23).
      For PARDISO at 50,000 free dofs and more the finished runs of E1 (126 runs, 32 problems) took a median of 2.19 times the
      model total. `estimate_bounds` holds the measured "up to" values.
    - "amg": `AMG_TIME_FACTOR` (4.1) times the time per iteration. The setup and
      the post-process time (`SETUP_S`, `POST_S`) do not change with the solver.
    - "gpu-mg": its own model, linear in the free dofs (`gpu_mg_estimate`; `gpu` has the load cases, the contrast and the device).
      The mid number of the band is returned (`(None, None)` for a CPU OpenCL device: it has no constants).

    `stress_rows` (the number of load cases with a stress limit) and `n_design` add the estimate of
    `stress_seconds` to each pass. Without stress rows the result is the same as before A2.
    `robust_mode` (settings.robust is eroded or wall) adds the solve of the nominal design at the end
    (`robust.time_factor`, from measured data). The time per iteration stays the same.
    """
    if solver == "gpu-mg":
        est = gpu_mg_estimate(n_free, iterations, gpu, stress_rows, n_design, step_acceptance, robust_mode)
        return est.s_per_iteration, est.total_s
    per_it = REF_S * (n_free / REF_DOFS) ** DOF_POWER
    if solver == "amg":
        per_it *= AMG_TIME_FACTOR
    per_it += stress_seconds(per_it, stress_rows, n_design, solver, step_acceptance)
    total = SETUP_S + per_it * (iterations + 1) * (robust.time_factor(iterations) if robust_mode else 1.0) + POST_S
    return per_it, total


def backend_estimates(n_free: int, iterations: int, stress_rows: int = 0, n_design: int = 0,
                      step_acceptance: bool = True, robust_mode: bool = False, gpu=None) -> dict:
    """Time estimate of one problem for each backend: `{name: {s_per_iteration, total_s, note}}`.

    These are model values for all backends. Only the cholesky model is fitted to a Mac
    (see `estimate_seconds`). With stress rows, each entry also has `stress_s_per_iteration` (the part of
    `s_per_iteration` that the stress rows add: an estimate) and `stress_total_s`.
    With `gpu` (`gpu_mg.check.TimeInputs`) the dict has the entry "gpu-mg" too: its own model, with `low_s`, `high_s` (the band),
    and `gpu_mg` (the facts of the estimate: constants, iterations, assumptions). For a CPU OpenCL device the numbers are None (no constants).
    """
    notes = {"cholesky": "model calibrated on Apple Accelerate Cholesky (Apple silicon)",
             "pardiso": "same model as cholesky (the hook example: 99 s on a 16 cores / 22 threads Linux machine, model 145 s; "
                        "the 99 s is the best of 62 runs on a shared host, the median was 462 s: experiment E1, run-time study, 2026-10-23)",
             "amg": f"the time per iteration is {AMG_TIME_FACTOR:g}x the cholesky model (the factor is "
                    "from the hook example; setup and post-process time do not change)"}
    out = {}
    for name, note in notes.items():
        per_it, total = estimate_seconds(n_free, iterations, name, stress_rows, n_design, step_acceptance,
                                         robust_mode=robust_mode)
        out[name] = {"s_per_iteration": per_it, "total_s": total, "note": note}
        if stress_rows:
            base_it, base_total = estimate_seconds(n_free, iterations, name, robust_mode=robust_mode)
            out[name].update(stress_s_per_iteration=per_it - base_it, stress_total_s=total - base_total,
                             stress_rows=stress_rows)
    if gpu is not None:
        est = gpu_mg_estimate(n_free, iterations, gpu, stress_rows, n_design, step_acceptance, robust_mode)
        out["gpu-mg"] = {"s_per_iteration": est.s_per_iteration, "total_s": est.total_s, "low_s": est.low_s, "high_s": est.high_s,
                         "note": "the gpu-mg model: linear in the free dofs, with a band from the PCG iterations (see gpu_mg/check.py)",
                         "gpu_mg": est.as_dict()}
        if stress_rows and est.total_s is not None:     # a device with no constants has no seconds (P2a review M2)
            base = gpu_mg_estimate(n_free, iterations, gpu, 0, n_design, step_acceptance, robust_mode)
            out["gpu-mg"].update(stress_s_per_iteration=est.s_per_iteration - base.s_per_iteration,
                                 stress_total_s=est.total_s - base.total_s, stress_rows=stress_rows)
    return out


CGROUP_ROOT = Path("/sys/fs/cgroup")
_NO_LIMIT = 2 ** 60   # cgroup v1 shows "no limit" as a huge number


def _read_text(path: Path):
    try:
        return path.read_text().strip()
    except (OSError, ValueError):
        return None


def _stat_bytes(path: Path, key: str):
    """The value of the line `key` in a cgroup `memory.stat` file, or None when the file, the line or the number is missing."""
    text = _read_text(path)
    for line in (text or "").splitlines():
        name, _, val = line.partition(" ")
        if name == key:
            try:
                v = int(val)
            except ValueError:
                return None
            return v if v >= 0 else None
    return None


def cgroup_limits(root: Path | None = None) -> dict:
    """Limits of the Linux cgroup (container) as facts: `{"memory_bytes", "memory_used_bytes",
    "memory_current_bytes", "inactive_file_bytes", "cpu_cores", "source", "facts"}`. A value is None when there is
    no limit (or the value is not known). `source` is "v2", "v1" or None.

    Reads cgroup v2 (memory.max, memory.current, memory.stat, cpu.max) and cgroup v1 (memory.limit_in_bytes,
    memory.usage_in_bytes, memory.stat, cpu.cfs_quota_us, cpu.cfs_period_us) under `root`.

    `memory_used_bytes` is the working set, the rule of cAdvisor and the kubelet (Docker subtracts the same value): the raw
    count (`memory_current_bytes`: memory.current, or memory.usage_in_bytes in v1) minus the inactive file pages
    (`inactive_file_bytes`: `inactive_file` of memory.stat in v2, `total_inactive_file` in v1), and not below 0. The raw count
    includes the page cache, and the kernel takes the inactive part back when a process needs memory. When memory.stat has
    no such line, the working set is the raw count and `facts` says so (a list of texts; empty when nothing is missing).
    When the inactive pages are more than the raw count (the two files are read at different times), the working set is 0
    and `facts` says so; Docker shows the raw count in that case.
    """
    root = CGROUP_ROOT if root is None else Path(root)
    out = {"memory_bytes": None, "memory_used_bytes": None, "memory_current_bytes": None, "inactive_file_bytes": None,
           "cpu_cores": None, "source": None, "facts": []}

    def num(text):
        try:
            v = int(text)
        except (TypeError, ValueError):
            return None   # None, "max" or garbage
        return v if 0 < v < _NO_LIMIT else None

    def used(current_file: Path, stat_file: Path, key: str):
        current, inactive = num(_read_text(current_file)), _stat_bytes(stat_file, key)
        out["memory_current_bytes"], out["inactive_file_bytes"] = current, inactive
        if current is None:
            return
        if inactive is None:
            out["memory_used_bytes"] = current
            room = " So the room can look smaller than it is." if out["memory_bytes"] else ""   # a room needs a limit
            out["facts"].append(
                f"{stat_file.name} has no readable {key} line, so the used memory is {current_file.name} with no change. "
                f"It counts the page cache, which the kernel takes back when a process needs memory.{room}")
        elif inactive > current:
            out["memory_used_bytes"] = 0
            out["facts"].append(
                f"{current_file.name} is {current} bytes and {key} in {stat_file.name} is {inactive} bytes, which is more. "
                f"The two files are read one after the other, and the kernel can update {stat_file.name} later than "
                f"{current_file.name}. The used memory is set to 0, as cAdvisor does.")
        else:
            out["memory_used_bytes"] = current - inactive

    mem, cpu = _read_text(root / "memory.max"), _read_text(root / "cpu.max")
    if mem is not None or cpu is not None:
        out["source"] = "v2"
        out["memory_bytes"] = num(mem)
        used(root / "memory.current", root / "memory.stat", "inactive_file")
        parts = (cpu or "").split()
        if len(parts) == 2 and num(parts[0]) and num(parts[1]):
            out["cpu_cores"] = int(parts[0]) / int(parts[1])
        return out
    lim = _read_text(root / "memory" / "memory.limit_in_bytes")
    quota = _read_text(root / "cpu" / "cpu.cfs_quota_us") or _read_text(root / "cpu,cpuacct" / "cpu.cfs_quota_us")
    if lim is not None or quota is not None:
        out["source"] = "v1"
        out["memory_bytes"] = num(lim)
        used(root / "memory" / "memory.usage_in_bytes", root / "memory" / "memory.stat", "total_inactive_file")
        period = _read_text(root / "cpu" / "cpu.cfs_period_us") or _read_text(root / "cpu,cpuacct" / "cpu.cfs_period_us")
        if num(quota) and num(period):
            out["cpu_cores"] = int(quota) / int(period)
    return out


def host_cores() -> int:
    """CPU cores that this process may use, by the CPU affinity only (the host view)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:   # macOS has no sched_getaffinity
        return os.cpu_count() or 1


def usable_cores() -> int:
    """CPU cores for this run: the affinity, and the cgroup CPU quota if it is lower (at least 1)."""
    n = host_cores()
    cores = cgroup_limits()["cpu_cores"]
    return max(1, min(n, int(cores))) if cores else n


def blas_thread_cap() -> int:
    """The OpenBLAS thread count that `cap_blas_threads` sets on Linux: min(4, usable cores)."""
    return min(4, usable_cores())


def cap_blas_threads() -> None:
    """Linux only: limit the OpenBLAS threads to 4, if the user did not set a limit.

    Many OpenBLAS threads next to the MKL and numba threads slow the run down
    (hook example with PARDISO on 16 cores / 22 threads: 149 s with the default,
    99 s with 4 threads; two single runs. Experiment E1 found that identical hook runs on a shared Linux host take from 99 s
    to 1,469 s). OpenBLAS reads the variable when numpy loads, so `main()`
    calls this first. OpenBLAS also reads GOTO_NUM_THREADS and OMP_NUM_THREADS: if
    the user set any of the three, nothing changes. Nothing happens on macOS.
    """
    if sys.platform != "linux" or any(v in os.environ for v in THREAD_VARS):
        return
    if "numpy" in sys.modules:   # too late: OpenBLAS has read its settings
        return
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_thread_cap())


def time_budget_s(settings) -> float | None:
    """The time budget of the user: `settings.stop_rules.time_budget_s` of the YAML, or None when the YAML sets none.

    There is no other budget. Without one, `topopt check` and the part builder give the time estimate as a fact and
    no budget warning.
    """
    return settings.stop_rules.time_budget_s


def budget_words(budget: float) -> str:
    """The words of a budget warning for the budget of the user (the same in `topopt check` and in the part builder)."""
    return f"your time budget of {budget:g} s (settings.stop_rules.time_budget_s)"


def budget_effect(rules) -> str:
    """What the time budget does in a run, as a fact: one text for every budget warning (`topopt check` and the part builder).

    `time_budget_s` is a stop rule, not a note. A `stop_rules` mapping turns the stop rules on, so with `enabled` true the
    rule `time_budget` ends the run at the first pass that ends after the budget (the clock of the rule is the run time at
    the end of a pass: it counts the setup and the passes, not the post step; CONTRACTS.md section 11). With `enabled: false`
    the budget is set and the rules are off, so the run does not stop at it. `rules` is `settings.stop_rules`.
    """
    if rules.enabled:
        return (f"the stop rules are on in this YAML (a stop_rules mapping turns them on), so the run stops at the first pass "
                f"that ends after {rules.time_budget_s:g} s of run time (rule time_budget; the rule converged is on too) and the part "
                "is the design of that pass; --no-stop-rules (MCP run_topopt(stop_rules=false)) turns the rules off for one run")
    return ("the stop rules are off in this YAML (enabled: false), so the run does not stop at this budget; "
            "--stop-rules (MCP run_topopt(stop_rules=true)) turns them on for one run")


def fit_voxel_size(h: float, total_s: float, limit_s: float, solver: str | None = "cholesky"):
    """Voxel size (mm, about) where the estimate is `limit_s` (the time budget of the user); None if it fits at `h`.

    Free dofs grow as h^-3 and the time of the loop as dofs^p: p is `DOF_POWER` for the direct solvers and 1 for gpu-mg
    (`solver`), whose model is linear and whose fixed part is not `SETUP_S` + `POST_S`. The result is a rough guide.
    """
    if solver == "gpu-mg":
        from .gpu_mg.check import GPU_MG_DOF_POWER as power, fixed_seconds
        fixed = fixed_seconds()
    else:
        power, fixed = DOF_POWER, SETUP_S + POST_S
    var, room = total_s - fixed, limit_s - fixed
    if var <= room or room <= 0:
        return None
    return math.ceil(h * (var / room) ** (1.0 / (3 * power)) * 1.02 * 100.0) / 100.0


def host_ram_bytes():
    """Physical RAM of the host in bytes, or None when the system does not tell."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def physical_ram_bytes():
    """RAM that this process may use, in bytes: the host RAM, or the cgroup limit if it is lower."""
    host = host_ram_bytes()
    limit = cgroup_limits()["memory_bytes"]
    if host is None:
        return limit
    return min(host, limit) if limit else host


# Memory model: constants fitted to the peak RSS of the Cholesky runs that were measured (Apple
# Accelerate, METIS): 65,067 dofs 0.96 GB, 129,360 dofs 2.05 GB, 253,575 dofs 4.12 GB. The
# fit is within about 12 percent on all six rows and a little high at the top. The sparse
# factor grows as dofs^(4/3) (nested dissection in 3D). The AMG backend uses less memory.
# The grid term (labels, masks, ray-cast arrays) is a rough guess.
GRID_BYTES_PER_VOXEL, FACTOR_BYTES_PER_DOF43, BASE_BYTES = 40, 270, 300_000_000


def estimate_memory_bytes(grid_voxels: int, free_dofs: int = 0, filter_bytes: int = 0, bordered_bytes: int = 0) -> int:
    """Rough peak memory of a build (grid part), the filter, and a Cholesky solve (factor part).

    The constants are fitted to the measured peak RSS of Apple Cholesky runs. AMG uses less.
    `filter_bytes` is the memory of the FFT filter (`members.filter_size`, `held_bytes_estimate`; 0 for
    "not known": the old estimate). It is held next to the factor, and its setup peak is not larger,
    so the run is the peak (#333: before it, the build of the filter matrix could be the peak).
    `bordered_bytes` (CONTRACTS.md section 38) is the bordered copy of `K` of a run with joints
    (`joints.bordered_copy_bytes`); 0 for a run without joints (the old number).
    """
    base = BASE_BYTES + GRID_BYTES_PER_VOXEL * grid_voxels
    factor = FACTOR_BYTES_PER_DOF43 * free_dofs ** (4 / 3) if free_dofs else 0
    return int(base + factor + filter_bytes + bordered_bytes)


def bordered_bytes(problem, edof, free_dofs) -> int:
    """The bordered copy of `K` in bytes for a problem with joints (CONTRACTS.md section 38); 0 without joints."""
    if problem.joints is None:
        return 0
    from .joints import bordered_copy_bytes, stiffness_nnz
    return bordered_copy_bytes(stiffness_nnz(edof, free_dofs))


def memory_warnings(mem: int, ram, fsz: dict | None, mem_without_filter: int) -> list[str]:
    """Facts about memory, as text lines, for `check` and the start of `run`. Nothing here stops a run.

    One line when the whole estimate is above the RAM. One more, with options, when the filter
    is the cause (it alone is above the RAM, or it takes the estimate above the RAM).
    """
    if not ram or mem <= ram:
        return []
    lines = [f"the memory estimate is {gib(mem)} and this machine has {gib(ram)} RAM: a run may swap or crash"]
    if fsz and (fsz["build_peak_bytes_estimate"] > ram or mem_without_filter <= ram):
        lines.append(f"the filter needs about {size_text(fsz['held_bytes_estimate'])} (estimate: an FFT on a padded grid of "
                     f"{fsz['fft_cells']:,} cells) and this machine has {gib(ram)} RAM: a run may swap or crash. "
                     f"Option: a larger voxel_size")
    return lines


def gib(nbytes) -> str:
    """Bytes as text in GiB (1024^3), for example '16.0 GiB'."""
    return f"{nbytes / 2**30:.1f} GiB"


# Warning thresholds of `check_data` (the values of the part-to-topopt skill check). The two region numbers are in `region_rules`.
MAX_FORCED_SOLID = 0.60


def backend_availability() -> dict:
    """Return which linear solvers can run here: `{name: {"available": bool, "note": str}}`.

    The names are the names of `solver_choice.resolve_solver`: cholesky, pardiso, amg.
    pypardiso is only asked on Linux x86_64 (never on macOS).
    """
    import importlib.util
    from .solver_choice import (PYPARDISO_PIN, accelerate_problem, install_command, is_linux_x86_64,
                                is_macos_arm64, is_x86_mac, platform_name, pypardiso_problem,
                                shim_build_command, x86_mac_text)
    mac = is_macos_arm64()
    shim = accelerate_problem() if mac else None   # a stray dylib on another platform does not count
    chol = mac and shim is None
    amg = importlib.util.find_spec("pyamg") is not None
    if chol:
        chol_note = "Apple Accelerate sparse Cholesky"
    elif mac:
        chol_note = ("libaccel_chol.dylib not found" if shim[0] == "missing"
                     else f"libaccel_chol.dylib is built but it does not load ({shim[1]})")
        chol_note += f" (Apple silicon only; build with: {shim_build_command()})"
    elif is_x86_mac():
        chol_note = f"Apple silicon only: {x86_mac_text()}"
    else:
        chol_note = f"Apple silicon only (this platform is {platform_name()})"
    if not is_linux_x86_64():   # pypardiso is never imported on other platforms
        pardiso, pnote = False, f"runs on Linux x86_64 only (this platform is {platform_name()})"
    else:
        problem = pypardiso_problem()   # the check of solver_choice: missing, or installed but not loading
        if problem is None:
            pardiso, pnote = True, f"Intel MKL PARDISO ({PYPARDISO_PIN})"
        elif problem[0] == "missing":
            pardiso, pnote = False, f"pypardiso is not installed. Install it with: {install_command()}"
        else:
            pardiso, pnote = False, (f"pypardiso is installed but it does not load: {problem[1]}. "
                                     f"Install the extra again with: {install_command()}")
    return {
        "cholesky": {"available": chol, "note": chol_note},
        "pardiso": {"available": pardiso, "note": pnote},
        "amg": {"available": amg,
                "note": "PyAMG-preconditioned CG" if amg else "pyamg is not installed"},
    }


# The backend that each solver (the name that `resolve_solver` returns) needs. An unknown solver
# or backend name is not a problem here (config validation owns unknown values).
_SOLVER_NEEDS = {"cholesky": ["cholesky"], "pardiso": ["pardiso"], "amg": ["amg"]}


def backend_problem(setting: str, backends=None, settings=None):
    """One line that says why the chosen solver cannot run here, with the options; None if it can.

    First the platform rule (`resolve_solver`): the text of its `UserError` if the platform cannot
    run the solver that `setting` names. Then the backend of the solver that `setting` means here
    (the Cholesky library, pypardiso, pyamg). For "gpu-mg", `settings` (the `Settings` of the
    problem) gives the device and the precision case to check.
    """
    if setting not in SOLVER_SETTINGS + EXPERIMENTAL_SOLVERS:
        return None   # config validation owns unknown values
    try:
        solver = resolve_solver(setting)
    except UserError as e:
        return str(e)
    if solver == "gpu-mg":
        from .gpu_mg.device import gpu_problem
        return gpu_problem(getattr(settings, "gpu_device", "auto"), getattr(settings, "gpu_case", "auto"))
    backends = backend_availability() if backends is None else backends
    for name in _SOLVER_NEEDS.get(solver, []):
        info = backends.get(name)
        if info is None or info["available"]:
            continue
        opts = [f"set settings.solver: {o} in the YAML"
                for o, oi in backends.items() if o != name and oi["available"]]
        if name == "cholesky":
            opts.append(f"build the library with: {shim_build_command()} (Apple silicon only)")
        if name == "amg" or not backends.get("amg", {}).get("available", True):
            opts.append("install pyamg")
        return (f"settings.solver is {setting!r} and the {name} backend cannot run here: "
                f"{info['note']}. Options: " + ", or ".join(dict.fromkeys(opts)))
    return None


def buckling_row_refusal(config) -> str | None:
    """The text of the stop when the buckling row (`buckling.min_factor`) cannot run with the solver of this run, or None.

    The row needs a direct factor of K. The imports of the row load only when the YAML turns it on.
    """
    if config.buckling.min_factor is None:
        return None
    from . import buckling_row
    return buckling_row.config_refusal(config)


def target_mass_line(problem) -> str:
    """The run log line of a run with `settings.target_mass_g` (I7): the asked mass, the derived fraction and the numbers it comes from."""
    from .voxelize import DESIGN, FIXED, LOAD
    cfg = problem.config
    st, lab = cfg.settings, problem.labels
    n_free = int((lab == DESIGN).sum())
    n_active = n_free + int(((lab == FIXED) | (lab == LOAD)).sum())
    vol = volume_target(st.volume_fraction, float(problem.h), cfg.material.density, n_active, n_free)
    return (f"target mass: {st.target_mass_g:g} g asked (settings.target_mass_g); settings.volume_fraction {st.volume_fraction:.6g} = "
            f"({vol['target_volume_mm3']:.1f} mm3 target - {vol['forced_solid_mm3']:.1f} mm3 forced solid) / {vol['free_design_mm3']:.1f} mm3 "
            f"free design volume, at {cfg.material.density:g} g/cm3")


def with_file(e: UserError, yaml_path) -> UserError:
    """Return `e`, or a copy with the suffix `(file: <yaml>)` when the message names the file in no way."""
    msg = str(e)
    shown = Path(yaml_path).expanduser().resolve()
    if "(file:" in msg or str(shown) in msg:   # the message has the path already
        return e
    return UserError(f"{msg} (file: {shown})")


EARLIER_RUN_KEPT = ("The earlier run in this folder is kept: its run files, snapshots/ and result/ are not changed "
                    "(status.json and the job files now show this run).")


def with_kept_run(text: str, recovered: str | None = None) -> str:
    """`text` and one more sentence: the run that this text is about changed nothing of the earlier run in its folder (`EARLIER_RUN_KEPT`).

    `topopt run` adds it to the error of a run that an input check refused in a folder that holds an earlier run (CONTRACTS.md section 24.1).
    `recovered` is the sentence of `snapshot.recover_earlier_run`, if this run made the folder right first (F4-47): it goes last.
    """
    text = text.rstrip()
    text = f"{text}{'' if text.endswith(('.', '!', '?')) else '.'} {EARLIER_RUN_KEPT}"
    return text if not recovered else f"{text} Before this run, {recovered}"


def kept_run_text(out, text: str, recovered: str | None = None) -> str:
    """The error text of a run that ended before it saved its first pass (F4-47). Move the earlier run back first (`RunWriter.restore_earlier`).

    The text has the sentence `EARLIER_RUN_KEPT` if `RunWriter.clean` had moved an earlier run aside, else it is `text`. If the move back fails, the
    text says where the earlier run is: the next `topopt run` in the folder moves it back (`snapshot.recover_earlier_run`).
    """
    from .snapshot import ASIDE_DIR, ASIDE_MOVING
    try:
        moved = out.restore_earlier()
    except OSError as e:
        text = text.rstrip()
        return (f"{text}{'' if text.endswith(('.', '!', '?')) else '.'} The earlier run in this folder could not be moved back ({e}). Its files are "
                f"in {ASIDE_DIR}/ and {ASIDE_MOVING}/ of the run folder, and the next `topopt run` in this folder moves them back.")
    return with_kept_run(text, recovered) if moved else text


VOLUME_FRACTION_FROM_MASS = ("derived from settings.target_mass_g: (target volume - forced-solid volume) / free design volume "
                             "(CONTRACTS.md section 3)")


def derived_numbers(cfg, grid_shape=None, h=None, n_active=None, n_free_vox=None,
                    free_dofs=None, solver="cholesky", extended=None, reach=None, solver_starts=True, solver_setting=None) -> dict:
    """The numbers an author states before a run (units mm, N, g). All are facts, none blocks a run.

    Without the grid arguments, only the numbers that the YAML gives are returned (forces,
    voxel size, filter radius, minimum feature, target fraction). With the grid, the volume
    basis, the masses and the time and memory estimates are added. `reach` (`Problem.load_reach`, F4-37) gives
    `load_cases[].load_reaches_free_dofs` and `load_cases[].load_reach`: null without it. `solver_starts` is False when
    `topopt check` finds that no solver can start: `solver` is then only the model of the estimates,
    and the up-to facts say that no run takes them (E1-P3).
    """
    st = cfg.settings
    h = float(st.voxel_size if h is None else h)
    cases = []
    entries = (extended or {}).get("entries")   # nodal data of the moment, remote force and pressure loads
    for ci, c in enumerate(cfg.load_cases):
        fs = [{"region": f.region, "force_n": [float(v) for v in f.force],
               "magnitude_n": round(math.sqrt(sum(v * v for v in f.force)), 4)} for f in c.forces]
        total = [sum(f.force[i] for f in c.forces) for i in range(3)]
        case = {"name": c.name, "weight": c.weight, "forces": fs}
        if cfg.uses_extended_loads():   # kind, moment, at and pressures (CONTRACTS.md section 26.2)
            from .loads import derived_extended
            case["pressures"], p_force = derived_extended(c, fs, entries[ci] if entries else None)
            total = [total[i] + p_force[i] for i in range(3)]
        cases.append({**case, "total_magnitude_n": round(math.sqrt(sum(v * v for v in total)), 4),
                      "applies_load": c.applies_load,            # U21: false if every force and moment is exactly 0 and no pressure is other than 0
                      "load_reaches_free_dofs": reach[ci] == "free" if reach else None,   # F4-37: false if the load vector is exactly 0.0 on every free dof (null: no grid)
                      "load_reach": reach[ci] if reach else None,   # F4-37 (review m1): why: free, nodal_zero, fixed or fixed_net (null: no grid); the words of the fact read it
                      "compliance_limit": c.compliance_limit,    # N*mm, null if none
                      "stress_limit_mpa": c.stress_limit_mpa,    # MPa, null if no stress row (A2)
                      "stress_limit_from": c.stress_limit_from,
                      "allowable_mpa": c.allowable_mpa, "safety_factor": c.safety_factor,
                      "allowable_source": c.allowable_source})
    d = {
        "load_cases": cases,
        "voxel_size_mm": h,
        "filter_radius_mm": float(st.filter_radius),
        "min_feature_mm": max(h, float(st.filter_radius)),
        "filter": filter_report(st, h, n_free_vox, grid_shape),   # radius, solid rule, min_member_mm (section 22a)
        "volume_fraction": st.volume_fraction,   # None for a config with settings.target_mass_g that no problem built yet (I7)
        "density_g_cm3": cfg.material.density,
        "iterations": st.iterations,
        "case_combine": st.case_combine,
        "objective": st.objective,
        "robust": robust.report(st, h, solver,   # mode, thresholds, facts (section 22b)
                                sum(1 for c in cfg.load_cases if c.stress_limit_mpa is not None),
                                buckling_row=cfg.buckling.min_factor is not None,
                                print_wall_mm=cfg.print_settings.limit("min_wall_mm")),
    }
    d["min_feature_note"] = d["filter"]["min_feature_note"]   # the same text as in `metrics.json` (section 22a)
    asked = {}
    if st.target_mass_g is not None:   # I7: the mass that the user asked for, and where the fraction comes from. A YAML with volume_fraction has none of these keys
        asked = {"target_mass_asked_g": float(st.target_mass_g), "target_mass_origin": "settings.target_mass_g"}
        d.update(asked)
        d["volume_fraction_origin"] = (VOLUME_FRACTION_FROM_MASS if st.volume_fraction is not None else
                                       "settings.target_mass_g (derived when the problem is built: it needs the voxel grid)")
    if grid_shape is not None and n_active is not None:
        vox_mm3 = h ** 3
        block = n_active * vox_mm3
        d["grid_shape"] = [int(v) for v in grid_shape]
        d["active_voxels"] = int(n_active)
        dens = cfg.material.density
        d["volume"] = {
            "active_block_mm3": round(block, 2),
            "volume_fraction": st.volume_fraction,
            "volume_fraction_of": "free design voxels (engine value; forced-solid voxels are "
                                  "solid on top)",
            **asked,   # I7: target_mass_asked_g and target_mass_origin; `target_mass_g` below is the mass of the derived fraction (the same number, to 3 decimals)
        }
        if n_free_vox is not None:
            d["volume"].update(volume_target(st.volume_fraction, h, dens, n_active, n_free_vox))
        d["mass_active_block_g"] = round(block / 1000.0 * dens, 3)
        if free_dofs is not None:
            n_stress = sum(1 for c in cfg.load_cases if c.stress_limit_mpa is not None)
            if solver == "gpu-mg":     # its own model: the load cases, the contrast and the device matter (P2a)
                from .gpu_mg.check import TimeInputs
                est = gpu_mg_estimate(free_dofs, st.iterations, TimeInputs.from_config(cfg, opencl_facts(st, solver_setting)), n_stress,
                                      int(n_free_vox or 0), st.stress_step_acceptance, robust_mode=st.robust != "off")
                total = est.total_s     # None: a CPU OpenCL device has no constants, so no seconds
                d["time_model"] = est.time_model()     # the basis and the inputs of the seconds below (P2a review M3)
                d["time_estimate_band_s"] = None if total is None else [round(est.low_s, 1), round(est.high_s, 1)]   # gpu-mg only: low and high
            else:
                per_it, total = estimate_seconds(
                    free_dofs, st.iterations, solver, n_stress,
                    int(n_free_vox or 0), st.stress_step_acceptance, robust_mode=st.robust != "off")
            d["time_estimate_s"] = None if total is None else round(total, 1)
            fbytes = d["filter"].get("size", {}).get("held_bytes_estimate", 0)   # the filter is part of it
            d["memory_estimate_gib"] = round(
                estimate_memory_bytes(int(math.prod(grid_shape)), int(free_dofs), fbytes) / 2**30, 2)
            # E1-P3: the measured "up to" value next to each estimate, facts only (CONTRACTS.md section 5.1). Nothing compares with it.
            d.update(derived_keys(solver if solver_starts else None, platform_name(), d["time_estimate_s"], d["memory_estimate_gib"], n_stress,
                                 free_dofs=int(free_dofs)))
    return d


def opencl_facts(settings=None, solver_setting=None):
    """The OpenCL platform and device facts when pyopencl is installed, else None.

    The drivers of the machine list the devices (no context is made). With `settings.solver: gpu-mg` (the user asked for the GPU) the list is made in
    this process. Else a child process makes it, with a time limit (`gpu_mg.probe`): this process makes no OpenCL call and does not import pyopencl,
    and `problem` of the facts says so when the list did not finish.
    `solver_setting` is the setting that the check runs with (`check --solver`); without it `settings.solver` holds.
    `settings` (the `Settings` of the problem) gives `gpu_device`, the device that the facts call "chosen".
    """
    import importlib.util
    if importlib.util.find_spec("pyopencl") is None:
        return None
    device = getattr(settings, "gpu_device", "auto")
    if (solver_setting or getattr(settings, "solver", None)) == "gpu-mg":
        from .gpu_mg.device import facts
        return facts(device)
    from .gpu_mg import probe
    return probe.facts(device, getattr(settings, "gpu_case", "auto"))


def _with_gpu_mg_facts(fn):
    """Decorator of `check_data`: `gpu_mg.check.apply` changes the memory and time facts of the solver gpu-mg (its own memory
    and time models). The code of the other solvers is not changed."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from .gpu_mg import check as gpu_check      # light module: no pyopencl
        return gpu_check.apply(fn(*args, **kwargs))
    return wrapper


@_with_gpu_mg_facts
def check_data(yaml_path, solver_setting: str | None = None, config=None) -> dict:
    """Load and voxelize a problem (no optimizer run). Return the facts as a dict. With `auto` it runs the short self-test of gpu-mg in a child process with a time limit; a pass is cached for each device and driver version.

    `config` (#318) is a config that replaces the one of the YAML: the config of a fine pass (`topopt check --fine-from`), with its voxel. The key
    `fine_pass` is then None. Else the key holds the voxel that the part needs (`fine_pass.part_voxel_block`, CONTRACTS.md section 24.6).

    Raises `UserError` when the YAML or an STL is wrong or a size guard stops the build.
    A solver that the platform cannot run does not raise: `solver["resolved"]` is None,
    `solver["error"]` has the text, and `cmd_check` raises it after it prints the facts.
    The `warnings` list holds facts that may matter. Nothing in it blocks a run.
    `solver_setting` replaces `settings.solver` for the solver rule only (for example "auto").
    """
    cfg = load_config(yaml_path) if config is None else config   # first: a bad YAML must not need the heavy imports
    import numpy as np
    from scipy import ndimage
    from .filter import solid_pull
    from .optimize import reduced_system
    from .loads import extended_facts, lost_force, lost_force_warning, no_load_lines, repeated_load_cases, repeated_load_lines
    from .problem import build_problem, with_volume
    from .springs import near_singular_lines, support_mask
    from .unit_fact import unit_fact_lines
    from .joints import near_singular_lines as joint_near_singular_lines
    from .voxelize import DESIGN, FIXED, KEEP_OUT, LOAD

    st = cfg.settings
    setting = solver_setting or st.solver
    solver_error = None
    choice = None
    try:
        choice = run_solver(setting, cfg)    # `auto`: gpu-mg on a GPU that passes the self-test, else the CPU rule; the facts say why
        solver = choice.solver
    except UserError as e:   # the platform cannot run this solver: give the facts, with the error
        solver, solver_error = None, str(e)
    try:
        prob = build_problem(cfg)
    except UserError as e:
        raise with_file(e, cfg.yaml_path)
    cfg = with_volume(cfg, prob)   # I7: with settings.target_mass_g the volume fraction is the one the problem derived
    st = cfg.settings
    _edof, active_index, free = reduced_system(prob)
    lab = prob.labels
    warnings: list[str] = []
    lost = lost_force(cfg, prob.load_region_ids, prob.fixed_dofs, prob.forces, (prob.extended or {}).get("entries"))
    lost_warning = lost_force_warning(lost)
    if lost_warning:
        warnings.append(lost_warning)
    try:   # A5: a row that cannot work stops here, before a run; the other facts are warnings
        from .rows_motion import check_rows
        warnings += [f"limits: {t}" for t in check_rows(cfg, prob, free)]
    except UserError as e:
        raise with_file(e, cfg.yaml_path)
    iface = None
    if cfg.interfaces:   # CONTRACTS.md section 23: the voxels of each interface, and a union piece that cannot work
        from . import interfaces

        iface = interfaces.voxel_facts(cfg, prob.labels, prob.origin, prob.h, prob.shape, support_mask(prob))
        floating = interfaces.floating_message(iface)
        if floating:
            warnings.append(floating + " (`topopt run` stops with this error)")

    grid_voxels = int(np.prod(prob.shape))
    counts = {"design": int((lab == DESIGN).sum()), "fixed": int((lab == FIXED).sum()),
              "load": int((lab == LOAD).sum()), "keep_out": int((lab == KEEP_OUT).sum())}
    n_active = int(active_index.size)
    n_free_vox = counts["design"]
    n_frozen = n_active - n_free_vox
    share = n_frozen / n_active if n_active else 0.0
    if share > MAX_FORCED_SOLID:
        warnings.append(f"forced-solid voxels are {100 * share:.0f} percent of the active voxels "
                        f"(a share above {100 * MAX_FORCED_SOLID:.0f} percent is a warning level): "
                        "little is left to optimize")

    load_regions = []
    for i, lr in enumerate(cfg.regions.load):
        m = prob.load_region_ids == i
        n = int(m.sum())
        thick = float(ndimage.distance_transform_edt(np.pad(m, 1)).max()) if n else 0.0
        load_regions.append({"name": lr.name, "voxels": n,
                             "thickness_voxels": round(2 * thick - 1, 1) if n else 0.0})
        if n < MIN_REGION_VOXELS or thick < MIN_REGION_THICK:
            warnings.append(f"load region '{lr.name}' is small or thin ({n} voxels, about "
                            f"{2 * thick - 1:.1f} voxels across): it can be cut off and removed")

    solid_regions = prob.region_facts.get("solid")   # None without regions.solid
    for r in (solid_regions or {}).get("regions", []):
        if r["voxels"] < MIN_REGION_VOXELS or (r["thickness_voxels"] + 1) / 2 < MIN_REGION_THICK:
            warnings.append(f"solid region '{r['name']}' is small or thin ({r['voxels']} voxels, about "
                            f"{r['thickness_voxels']:.1f} voxels across): it can be cut off and removed")

    active = (lab == DESIGN) | (lab == FIXED) | (lab == LOAD)
    # V2: does the design-space box fit the print bed? A fact for the configured direction and the
    # six axis directions, not a refusal (CONTRACTS.md section 18).
    from .print_report import check_facts, mask_box_corners
    print_facts = check_facts(mask_box_corners(active, prob.origin, prob.h), cfg.print_settings)
    comp, n_comp = ndimage.label(active)
    sizes = sorted(np.bincount(comp.ravel())[1:].tolist(), reverse=True) if n_comp else []
    from .connectivity import (RUN_STOPS, held_pieces_message, message as pieces_message, support_touch, touch_message,
                               unsupported_pieces)
    touch = support_touch(prob)      # one rule with the part builder: a support (FIXED, a spring or a joint region) that shares a face with the design
    touches, held_by = touch["touches"], touch["held_by"]     # A7b: a spring or a joint region holds a piece
    loose = pieces_message(unsupported_pieces(prob))   # a piece with no support: `topopt run` stops (CONTRACTS.md section 3)
    if loose:
        warnings.append(loose + RUN_STOPS)
    elif n_comp > 1:   # every piece holds a support: the run works (the same text as the part builder)
        warnings.append(held_pieces_message(sizes, held_by))
    try:   # A4c: a perimeter bound below the smallest possible P cannot be met: it stops here, before a run. None without the key
        from .perimeter import check_facts as perimeter_facts
        perimeter = perimeter_facts(cfg, prob)
    except UserError as e:     # after the connectivity facts: a refused bound must not hide a piece with no support
        raise with_file(UserError(f"{loose}{RUN_STOPS}\n{e}") if loose else e, cfg.yaml_path)
    if not touches:   # a fact: a support that reaches the design through solid voxels holds the part (the run works)
        msg = touch_message(touch)
        if msg:
            warnings.append(msg)
    from .voxelize import plane_lines, taken_warnings
    taken = prob.region_facts.get("taken") or []   # CONTRACTS.md section 23: what the cuts take from the frozen regions
    warnings += taken_warnings(taken)
    region_planes = prob.region_facts.get("planes") or []   # P1: flat faces that are not on a voxel face (the region is larger or smaller than the STL)
    warnings += plane_lines(region_planes, float(prob.h), prob.origin)
    warnings += unit_fact_lines(cfg)   # TC-3: the design space may be in inches or in metres; the text of `problem_facts` (`topopt check` prints it from there, once)
    repeated = repeated_load_cases(cfg)   # F4-5: a load case that repeats an earlier case or is its exact opposite
    warnings += repeated_load_lines(repeated)   # the text of `problem_facts` (`topopt check` prints it from there, once)
    reach = prob.load_reach or None   # F4-37: why the load of each case reaches a free dof or not
    warnings += no_load_lines(cfg, reach)   # U21: a load case that applies no load; the text of `problem_facts` (`topopt check` prints it from there, once)
    warnings += near_singular_lines((prob.extended or {}).get("springs"), cfg.material.E, prob.h)     # section 35
    warnings += joint_near_singular_lines((prob.extended or {}).get("joints"), cfg.material.E, prob.h)     # section 38

    # The time estimate is for the solver that runs. Without a solver (the platform cannot run
    # the named one), it uses the model of the fast solvers, and `amg_total_s` shows amg.
    n_stress = sum(1 for c in cfg.load_cases if c.stress_limit_mpa is not None)
    opencl = opencl_facts(st, setting)
    gpu = None
    if solver == "gpu-mg":     # the gpu-mg model needs the load cases, the contrast and the device (P2a)
        from .gpu_mg.check import TimeInputs
        gpu = TimeInputs.from_config(cfg, opencl)
    est = backend_estimates(int(free.size), st.iterations, n_stress, int(n_free_vox), st.stress_step_acceptance,
                            robust_mode=st.robust != "off", gpu=gpu)
    per_it, total = (est[solver if solver in est else "cholesky"][k] for k in ("s_per_iteration", "total_s"))
    stress_est = None
    if n_stress:
        e = est[solver if solver in est else "cholesky"]
        stress_est = {"rows": n_stress, "s_per_iteration": e.get("stress_s_per_iteration"), "total_s": e.get("stress_total_s"),
                      "step_acceptance": st.stress_step_acceptance, "label": "estimate"}
    amg_s = est["amg"]["total_s"]
    rec = None if solver is None else recommendation(setting, solver, {
        "amg": amg_s, "pardiso": est["pardiso"]["total_s"]})
    range_s = [total, total * STRESS_RUN_TIME_FACTOR] if n_stress and total is not None else None   # a run with stress rows: low and high
    budget = time_budget_s(st)     # the user's own budget (settings.stop_rules.time_budget_s), or None: then there is no budget warning
    if budget is not None and range_s and range_s[1] > budget:
        fit = fit_voxel_size(float(prob.h), range_s[1], budget, solver)
        warnings.append(f"the time estimate for a run with stress rows is {range_s[0]:.0f} to {range_s[1]:.0f} s, "
                        f"above {budget_words(budget)} at the high end (the high end is the model times "
                        f"{STRESS_RUN_TIME_FACTOR:g}, a fixed factor; {stress_runs_text()}); "
                        + budget_effect(st.stop_rules)
                        + (f"; voxel_size about {fit} mm would fit" if fit else "")
                        + ("; or install pypardiso (see recommendation)" if rec else ""))
    elif budget is not None and total is not None and total > budget and not range_s:
        fit = fit_voxel_size(float(prob.h), total, budget, solver)
        if solver == "gpu-mg":     # the middle of the gpu-mg band; the fill of the part does not change a time that is linear in the dofs
            what = (f"the gpu-mg time estimate is {total:.0f} s (band {est['gpu-mg']['low_s']:.0f} to {est['gpu-mg']['high_s']:.0f} s), "
                    f"above {budget_words(budget)}")
        else:
            what = f"the time estimate is {total:.0f} s, above {budget_words(budget)}"
        warnings.append(what + "; " + budget_effect(st.stop_rules)
                        + (f"; voxel_size about {fit} mm would fit" if fit else "")
                        + ("; or install pypardiso (see recommendation)" if rec else ""))
    ram = physical_ram_bytes()
    fsz = filter_size(float(st.filter_radius), float(prob.h), n_free_vox, prob.shape)
    bord = bordered_bytes(prob, _edof, free)    # joints (section 38): the bordered copy of K; 0 without joints
    mem = estimate_memory_bytes(grid_voxels, int(free.size), fsz["held_bytes_estimate"], bord)   # the filter is part of it
    warnings += memory_warnings(mem, ram, fsz, estimate_memory_bytes(grid_voxels, int(free.size), 0, bord))
    warnings += filter_report(st, prob.h)["facts"]   # min_member_mm facts with options (section 22a)
    if solver_error:
        warnings.append(f"settings.solver is {setting!r} and it cannot start here: {solver_error}")
    warnings += robust.report(st, prob.h, solver, n_stress,   # robust mode facts with options (section 22b)
                              buckling_row=cfg.buckling.min_factor is not None,
                              print_wall_mm=cfg.print_settings.limit("min_wall_mm"))["facts"]

    backends = backend_availability()
    for name, info in backends.items():
        info["time_estimate"] = est.get(name)
    problem = backend_problem(setting, backends, st) if solver else None   # the platform error is in `solver_error`
    if solver == "gpu-mg" and not problem and prob.joints is not None:     # P2b, P2d, P2d-D3h: the beam over the bond stiffness for case C, and what case C does with the rigid modes and the harmonic columns of the bonded patches
        from .gpu_mg.border import check_line as gpu_mg_joint_line
        warnings.append(gpu_mg_joint_line(prob.joints))
    if problem:
        warnings.append(f"settings.solver is {setting!r} and it cannot start here: {problem}"
                        if solver == "gpu-mg" else problem)
    shim_line = None if solver is None else shim_warning_line(setting, solver)
    if shim_line:   # a Mac without the Accelerate shim: `auto` runs amg, with the build command
        warnings.append(shim_line)
    bk_info = None
    if cfg.buckling.check:   # the YAML turns the buckling check on: say now if it cannot run (CONTRACTS.md 33)
        from .buckling_voxel import preflight
        bk_info = preflight(cfg, solver, grid_voxels, n_active, int(free.size), ram)
        if not bk_info["will_run"]:
            warnings.append(f"buckling check: it is on in the YAML, but it will be skipped "
                            f"({bk_info['reason']}). " + " ".join(bk_info["facts"])
                            + " Options: " + " ".join(bk_info["options"]))

    row_info = None
    if cfg.buckling.min_factor is not None:   # the YAML turns the buckling row on (CONTRACTS.md 39): memory, and a refusal as a warning
        from . import buckling_row
        row_info = buckling_row.preflight(cfg, solver, grid_voxels, int(free.size), ram, run_bytes=mem, avail=backends)
        warnings += row_info["warnings"]

    from .verdict import beta_stage_fact
    if config is None:   # the plan of a fine pass (#318) has a maximum of passes, not a plan of passes
        warnings += beta_stage_fact(st.iterations, st.beta_schedule)   # F4-18: the text of `problem_facts` (`topopt check` prints it from there, once)
    off_info = None
    if not cfg.buckling.check and cfg.buckling.min_factor is None:   # both off: say what the check would cost (a fact; CONTRACTS.md 33)
        from .buckling_voxel import preflight as bk_preflight
        pf = bk_preflight(cfg, solver, grid_voxels, n_active, int(free.size), ram)
        off_info = {"check": False, "row": False, "modes": cfg.buckling.modes, "max_seconds": cfg.buckling.max_seconds,
                    "memory_estimate_bytes": pf["memory_estimate_bytes"], "would_run": pf["will_run"], "reason": pf["reason"]}

    result = {
        "name": cfg.name,
        "yaml": str(Path(yaml_path).resolve()),
        "grid": {"shape": [int(v) for v in prob.shape], "voxel_mm": float(prob.h),
                 "origin_mm": [round(float(v), 3) for v in prob.origin], "voxels": grid_voxels},
        "voxel_counts": counts,
        "interfaces": iface,
        "interface_taken": taken,   # what each cut takes from each frozen region (section 23); [] if no cut takes a voxel
        "load_regions": load_regions,
        "solid_regions": solid_regions,   # counts (section 22a), or null without regions.solid
        "solid_pull": solid_pull(lab, float(prob.h), float(st.filter_radius), st.filter_solid_neighbours,
                                 pad_void=st.robust == "wall"),
        "active_voxels": n_active,
        "free_design_voxels": n_free_vox,
        "forced_solid_voxels": n_frozen,
        "forced_solid_share": round(share, 4),
        "free_dofs": int(free.size),
        "iterations": st.iterations,
        "volume_fraction": st.volume_fraction,
        "connectivity": {"components": int(n_comp), "component_sizes": sizes[:5],
                         "fixed_touches_design": touches},
        "region_planes": region_planes,   # P1: the flat faces off the voxel grid, by region (CONTRACTS.md section 3); [] if none
        "repeated_load_cases": repeated,  # F4-5: the pairs of load cases with the same load or the exact opposite (section 13); [] if none
        "solver_setting": setting,
        "solver": {"setting": setting, "resolved": solver, "platform": platform_name(),
                   "cpu_solver": choice.cpu if choice is not None else None,     # the CPU rule for `setting` (for `auto`: the fallback), never gpu-mg for `auto`; CI compares it
                   "error": solver_error,
                   **({"choice": choice.as_dict()} if choice is not None else {})},   # GPU first: what the run uses first, the CPU fallback and the facts of `auto`
        "backends": backends,
        "opencl": opencl,
        "time_estimate": {"solver": solver, "s_per_iteration": per_it, "total_s": total,
                          "limit_s": budget, "amg_total_s": amg_s, "stress": stress_est,
                          **({"range_s": range_s, "range_factor": STRESS_RUN_TIME_FACTOR} if range_s else {}),
                          **({"gpu_mg": est["gpu-mg"]["gpu_mg"]} if gpu is not None else {})},     # `gpu_mg.check.apply` reads it
        "memory_estimate": {"bytes": mem, "ram_bytes": ram,
                            "filter_build_peak_bytes": fsz["build_peak_bytes_estimate"],
                            "filter_held_bytes": fsz["held_bytes_estimate"],
                            **({"bordered_copy_bytes": bord} if bord else {}),     # joints (section 38)
                            "note": "rough model: grid arrays plus the sparse Cholesky factor plus the FFT filter"},
        "recommendation": rec,
        "hint": platform_hint(),
        "snapshot_disk_mb": round((st.iterations + 1) * grid_voxels / 1e6, 1),
        "print": print_facts,
        **({"material": cfg.filament.check_facts(cfg.print_settings.profile)} if cfg.filament is not None else {}),   # section 18.8: only with material.filament
        "warnings": warnings,
        "lost_force": lost,
        "mirror": prob.mirror,   # facts of the `mirror` key, or None (CONTRACTS.md section 26.1)
        "perimeter": perimeter,  # facts of `settings.perimeter_max_mm2` (bound, fixed part, p_min, ratio), or None (section 26.3)
        "assumptions": list(cfg.assumptions),
        "derived": derived_numbers(cfg, prob.shape, float(prob.h), n_active, n_free_vox, int(free.size),
                                   solver or "cholesky", prob.extended, reach, solver_starts=solver is not None, solver_setting=setting),
        **({"extended": extended_facts(prob)} if prob.extended else {}),
    }
    from . import fine_pass
    result["fine_pass"] = None if config is not None else fine_pass.part_voxel_block(cfg, result)   # #318: the voxel that the part needs; `fact` when the run voxel is too coarse
    if bk_info is not None:
        result["buckling"] = bk_info    # only when the YAML turns the check on
    if row_info is not None:
        result["buckling_row"] = row_info    # only when the YAML turns the row on (`buckling.min_factor`)
    if off_info is not None:
        result["buckling_off"] = off_info    # only when the check and the row are both off (the default)
    return result


def buckling_off_line(info: dict) -> str:
    """The `topopt check` line when the buckling check and the buckling row are both off (CONTRACTS.md section 33): the memory estimate and the time budget.

    The tool has a memory estimate of the check (`buckling_voxel.estimate_bytes`) and a time budget for the eigen solve of each load case
    (`buckling.max_seconds`). It has no model of the time of the check, so the line gives the budget and says so.
    """
    from .buckling_voxel import gb
    skip = ""
    if not info["would_run"]:
        skip = (" With the solver of this run the check would skip: it needs a direct solver (cholesky or pardiso)."
                if info["reason"] == "solver_not_direct" else f" With these settings the check would skip (reason: {info['reason']}).")
    return ("buckling: off (check and row). A part with members in compression can buckle: buckling.check true gives the load factors "
            f"(memory estimate about {gb(info['memory_estimate_bytes'])}; the time budget of each eigen solve is {info['max_seconds']:g} s; "
            "the tool has no estimate of the time)." + skip)


def problem_facts(cfg, reach=None, fine=False) -> list[str]:
    """Facts about the problem that no key says (a mass run with no limit). See `optimize.problem_facts`.
    `reach` is `Problem.load_reach` (F4-37), or None where there is no grid. `fine`: the plan of a fine pass (#318)."""
    from .optimize import problem_facts as facts
    return facts(cfg, reach=reach, fine=fine)


def opencl_lines(cl) -> list[str]:
    """Lines for `topopt check`: the OpenCL devices (pyopencl installed), none otherwise."""
    if not cl:
        return []
    if cl.get("problem") and not cl["devices"]:
        return [f"opencl:         {cl['problem']} (pyopencl {cl['pyopencl']})"]
    out = [f"opencl:         pyopencl {cl['pyopencl']}, {len(cl['devices'])} device(s); "
           f"settings.gpu_device auto takes device {cl['auto_device']} (case {cl['auto_case']})"]
    for dv in cl["devices"]:
        same = dv.get("duplicate_of")
        out.append(f"opencl device:  {dv['index']}: {dv['name']} ({dv['kind']}, platform {dv['platform']}, "
                   f"{dv['version']}), 64-bit float {'yes' if dv['fp64'] else 'no'}"
                   + (f" (same platform, name and version as device {same}: the OpenCL driver list names it twice)"
                      if same is not None else ""))
    return out


def reach_of(d: dict) -> list | None:
    """`load_reach` of each case of `check_data` (F4-37), for the facts of `problem_facts`."""
    reach = [c["load_reach"] for c in d["derived"]["load_cases"]]
    return None if None in reach else reach


def _fine_check_setup(args, cfg):
    """`topopt check --fine-from PARENT` (#318): the `fine_pass.FineSetup` of the plan, or None without `--fine-from`. Nothing is written.

    The plan is made as `topopt run --fine-from` makes it: the parent is read, the voxel, the maximum and the tolerance are chosen, and the
    inputs are compared with the checkpoint of the parent (a refusal is an error with options, as the run would give it).
    """
    from . import fine_pass
    from .checkpoint import ResumeRefused, plan_fine
    parent = getattr(args, "fine_from", None)
    _fine_options_error(args, parent is not None)
    if parent is None:
        return None
    if getattr(args, "solver", None):
        cfg = dataclasses.replace(cfg, settings=dataclasses.replace(cfg.settings, solver=args.solver))
    try:
        setup = fine_pass.resolve_fine(cfg, parent, None, getattr(args, "fine_voxel", None), getattr(args, "fine_passes", None),
                                       getattr(args, "fine_tol", None))
        if setup.parent.ck is not None:
            setup.facts += plan_fine(setup.parent.ck, setup.config, None, None, getattr(args, "allow_changed_inputs", False), setup.parent_dir)
    except ResumeRefused:
        raise
    except UserError as e:
        raise with_file(e, args.yaml)
    return setup


def fine_plan(setup) -> dict:
    """The plan of a fine pass as data (`topopt check --fine-from --json`, key `fine_from`): the record without the numbers of a mapping, and the facts."""
    from . import fine_pass
    source, source_facts = fine_pass.choose_source(setup.parent)
    return {**fine_pass.fine_record(setup, source), "facts": [fine_pass.plan_line(setup), *setup.facts, *source_facts]}


def fine_plan_lines(setup) -> list[str]:
    """The lines of `topopt check --fine-from`: what the fine pass would do. The numbers below them are those of the fine voxel."""
    return fine_plan(setup)["facts"]


def cmd_check(args) -> int:
    cfg = load_config(args.yaml)   # a bad YAML ends here, before the heavy imports of check_data
    fine = _fine_check_setup(args, cfg)   # #318: `--fine-from` shows the plan of a fine pass (the config has the fine voxel), or None
    try:
        d = check_data(args.yaml, solver_setting=getattr(args, "solver", None), config=None if fine is None else fine.config)
    except UserError as e:
        from . import fine_pass
        raise with_file(e if fine is None else fine_pass.name_fine_option(e), args.yaml)
    if fine is not None:
        cfg = fine.config
    te, sv, rec, hint = d["time_estimate"], d["solver"], d["recommendation"], d["hint"]
    g = d["grid"]["shape"]
    # The platform error (`sv["error"]`) is raised at the end, so its warning is not printed twice.
    warnings = [w for w in d["warnings"]
                if w.startswith(("the time estimate is", "the gpu-mg time estimate", "the time estimate for a run with stress rows",
                                 "the memory estimate", "the filter needs", "solver auto -> amg", "limits: ",
                                 "buckling check:", "buckling row:", "interface '", "keep-out region '",
                                 "gpu-mg models: ", "joints on gpu-mg: "))
                or w.endswith(" (`topopt run` stops with this error)")     # connectivity.RUN_STOPS: a piece with no support
                or (w.startswith("settings.solver is") and not sv["error"])]
    me = d["memory_estimate"]
    bounds = d["derived"]["estimate_bounds"]      # E1-P3: the same dict as `run.json` `derived`
    if getattr(args, "json", False):
        # All the facts of `check_data` (this branch), with the fields of the contract: the time
        # estimate has rounded numbers and `amg_total_s`.
        none = lambda v, n=None: None if v is None else round(v, n) if n is not None else round(v)      # noqa: E731  (a device with no constants: None)
        te_json = {**te, "s_per_iteration": none(te["s_per_iteration"], 3),
                   "total_s": none(te["total_s"]), "amg_total_s": round(te["amg_total_s"])}
        if te.get("stress"):
            te_json["stress"] = {**te["stress"], "s_per_iteration": none(te["stress"]["s_per_iteration"], 4),
                                 "total_s": none(te["stress"]["total_s"], 1)}
        if te.get("range_s"):
            te_json["range_s"] = [round(v) for v in te["range_s"]]
        print(json.dumps({**d, "grid_shape": g, "voxel_size_mm": d["grid"]["voxel_mm"],
                          "time_estimate": te_json, "facts": problem_facts(cfg, reach_of(d), fine is not None),
                          **({} if fine is None else {"fine_from": fine_plan(fine)})}, indent=2, default=str))
    else:
        if fine is not None:
            for line in fine_plan_lines(fine):
                print(line)
        elif d.get("fine_pass") and d["fine_pass"].get("fact"):
            print(d["fine_pass"]["fact"])   # #318: the part needs a finer voxel than this run has. One line, only when it is so
        print(f"name:           {d['name']}")
        print(f"grid shape:     {g[0]} x {g[1]} x {g[2]}  (voxel {d['grid']['voxel_mm']} mm)")
        print(f"active voxels:  {d['active_voxels']}")
        print(f"free dofs:      {d['free_dofs']}")
        print(f"iterations:     {d['iterations']}")
        if sv["resolved"] is None:
            print(f"solver:         none  (settings.solver {sv['setting']} cannot start: {sv['error']})")
            print(f"time estimate:  {te['s_per_iteration']:.2f} s/iteration, about {te['total_s']:.0f} s in total "
                  f"with the model of the fast solvers; about {te['amg_total_s']:.0f} s with amg ({bounds['time']['line']})")
        else:
            print(f"solver:         {sv['resolved']}  (settings.solver {sv['setting']}, platform {sv['platform']})")
            why = (sv.get("choice") or {}).get("line")
            if why:     # `auto`: what it picked and why (the GPU and its self-test, or the reason for the CPU solver)
                print(f"solver choice:  {why[len('solver: '):]}")
            if te["total_s"] is None:     # gpu-mg on a CPU OpenCL device: no constants, so no seconds
                print("time estimate:  none: no constants for this device (CPU device); run `topopt gpu-check` to measure them")
            else:
                basis = f" ({te['gpu_mg']['label']})" if te.get("model") == "gpu-mg" else ""     # whose constants: not a bare number (P2a review M3)
                print(f"time estimate:  {te['s_per_iteration']:.2f} s/iteration, about {te['total_s']:.0f} s in total{basis} "
                      f"({bounds['time']['line']})")
        if te.get("stress") and te["stress"]["total_s"] is None:
            se = te["stress"]
            print(f"stress rows:    {se['rows']} row{'s' if se['rows'] != 1 else ''}; no seconds for this device (see the time estimate above)")
        elif te.get("stress"):
            se = te["stress"]
            print(f"stress rows:    {se['rows']} row{'s' if se['rows'] != 1 else ''}, in the time estimate above (estimate): about "
                  f"{se['s_per_iteration']:.3f} s/iteration and {se['total_s']:.0f} s in total for the adjoint solves, the KS "
                  f"aggregate" + (" and the rejected tries of the step acceptance" if se["step_acceptance"] else ""))
            if te.get("range_s"):
                lo, hi = te["range_s"]
                print(f"time range:     {lo:.0f} to {hi:.0f} s for this run with stress rows (estimate). Low: the model above. "
                      f"High: the model times {te['range_factor']:g}, a fixed factor, not a bound. {stress_runs_text(sentence=True)}")
        print(f"memory estimate: about {gib(me['bytes'])}"
              + (f" of {gib(me['ram_bytes'])} RAM" if me["ram_bytes"] else " (RAM size not known here)")
              + f" (the filter is part of it: about {size_text(me['filter_held_bytes'])}"
              + (f"; the bordered copy of K for the joints: about {size_text(me['bordered_copy_bytes'])}" if me.get("bordered_copy_bytes") else "")
              + f") ({bounds['memory']['line']})")
        print(f"estimate bounds: measured in {bounds['time']['basis']}. Facts only: no warning or refusal uses them")
        if me.get("model") == "gpu-mg":
            from .gpu_mg.check import lines as gpu_mg_lines
            for line in gpu_mg_lines(d):
                print(line)
        for line in opencl_lines(d["opencl"]):
            print(line)
        bk = d.get("buckling")
        if bk and bk["will_run"]:
            from .buckling_voxel import gb
            print(f"buckling check: on in the YAML ({bk['density']} design, {bk['modes']} modes): memory estimate "
                  f"about {gb(bk['memory_estimate_bytes'])} (GB = 1e9 bytes)")
        if d.get("buckling_row"):
            from . import buckling_row
            for line in buckling_row.lines(d["buckling_row"]):
                print(line)
        if d.get("buckling_off"):
            print(buckling_off_line(d["buckling_off"]))
        if hint:
            print("hint: " + hint)
        if rec:
            print(recommendation_line(rec))
        for w in warnings:
            print(w if w.startswith(CHECK_FACT_LINES) else f"warning: {w}")      # a fact line of the solver gpu-mg has no "warning:" prefix (gpu-mg is the main mode)
        from .print_report import check_lines
        for line in check_lines(d["print"]):   # V2: before the derived lines, so the assumptions stay last
            print(line)
        if d.get("material"):   # section 18.8: the filament and its fit to the printer; facts, never a stop
            from .materials_catalog import check_lines as material_lines
            for line in material_lines(d["material"]):
                print(line)
        for f in d.get("interfaces") or []:
            what = ("forced solid, no load case" if f["op"] == "union" else "keep-out")
            print(f"interface:      {f['name']} ({f['kind']}, {f['op']}): {f['voxels']} voxels, {f['held']} hold the label ({what})")
        from .voxelize import taken_lines
        for line in taken_lines(d.get("interface_taken")):
            print(line)
        print_derived(d)
        from .loads import extended_lines
        for line in extended_lines(d.get("extended")):
            print(line)
        for line in lost_force_lines(d["lost_force"]):
            print(line)
        for line in problem_facts(cfg, reach_of(d), fine is not None):
            print(line)
        from .symmetry import summary_lines
        for line in summary_lines(d["mirror"]):
            print(f"mirror:         {line}")
        if d.get("perimeter"):   # A4c (section 26.3): facts, no threshold
            from .perimeter import PerimeterRow
            for line in PerimeterRow.lines(d["perimeter"]):
                print(line.replace("perimeter: ", "perimeter:      ", 1))
    if sv["error"]:
        raise with_file(UserError(sv["error"]), d["yaml"])   # main() prints it and exits with 1
    return 0


def lost_force_lines(lf) -> list[str]:
    """Text lines about the lost force and moment; an empty list when nothing is lost."""
    out = []
    for k in lf["cases"]:
        if k["lost_magnitude_n"] > 0:
            p = k["lost_percent"]
            of_sum = f"{k['lost_percent_of_sum']:.1f} percent of the sum of region forces"
            if p is None and "lost_moment_nmm" in k:   # a case with a moment and no resultant force
                out.append(f"lost force:     {k['name']}: {k['lost_magnitude_n']:.4g} N of nodal force on fixed nodes, "
                           f"dropped by the solver (the case has no resultant force, a moment; "
                           f"{k['lost_percent_of_sum']:.1f} percent of the sum of the nodal force sizes)")
            else:
                pct = (f"{p:.1f} percent" if p is not None else "total about 0") + f", {of_sum}"
                out.append(f"lost force:     {k['name']}: {k['lost_magnitude_n']:.4g} N of "
                           f"{k['total_magnitude_n']:.4g} N ({pct}) on fixed nodes, dropped by the solver")
        if k.get("lost_moment_magnitude_nmm", 0) > 0:
            p = k["lost_moment_percent"]
            out.append(f"lost moment:    {k['name']}: {k['lost_moment_magnitude_nmm']:.4g} N mm about the origin "
                       f"({'applied moment about 0' if p is None else f'{p:.1f} percent of the applied moment'}) "
                       "on fixed nodes, dropped by the solver")
    return out


def print_derived(d) -> None:
    """Print the derived numbers and the assumptions of `check_data` (extra lines)."""
    dv = d["derived"]
    for c in dv["load_cases"]:
        for f in c["forces"]:
            if f.get("kind", "force") != "force":   # a moment or a remote force: see the `load:` lines
                continue
            v = ", ".join(f"{x:g}" for x in f["force_n"])
            print(f"force:          {c['name']} (weight {c['weight']:g}) on '{f['region']}': "
                  f"[{v}] N, {f['magnitude_n']:g} N")
    for c in dv["load_cases"]:
        if c["stress_limit_mpa"] is not None:
            extra = (f", allowable {c['allowable_mpa']:g} MPa" if c["allowable_mpa"] is not None else "") + (
                f", safety factor {c['safety_factor']:g}" if c["safety_factor"] is not None else "") + (
                f" ({c['allowable_source']})" if c["allowable_source"] else "")
            print(f"stress limit:   {c['name']}: {c['stress_limit_mpa']:g} MPa from {c['stress_limit_from']}{extra}")
    if dv["objective"] != "compliance":
        print(f"objective:      {dv['objective']} (settings.volume_fraction is the start density)")
    vol = dv["volume"]
    if "target_mass_asked_g" in dv:   # I7: the user's mass and the fraction that the engine derived from it
        print(f"target mass:    {dv['target_mass_asked_g']:g} g asked ({dv['target_mass_origin']}); "
              f"settings.volume_fraction {dv['volume_fraction']:.6g} is derived from it")
    if "target_mass_g" in vol:
        print(f"volume basis:   all active voxels {vol['active_block_mm3']:.0f} mm3 "
              f"({vol['forced_solid_mm3']:.0f} mm3 forced solid, {vol['free_design_mm3']:.0f} mm3 free); "
              f"fraction {dv['volume_fraction']:g} applies to the free part; "
              f"{'start' if dv['objective'] == 'mass' else 'target'} "
              f"{vol['target_volume_mm3']:.0f} mm3 = {vol['target_mass_g']:.1f} g at "
              f"{dv['density_g_cm3']:g} g/cm3")
    print(f"full block:     {dv['mass_active_block_g']:.1f} g (all active voxels solid)")
    print(f"min feature:    {min_feature_line(dv['filter'])}")
    from .filter import solid_pull_line
    from .voxelize import solid_lines
    for line in filter_lines(dv["filter"]):
        print(f"filter:         {line}")
    for line in dv["filter"]["facts"]:
        print(f"warning: {line}")
    for line in solid_lines(d.get("solid_regions")):
        print(line)
    pull = solid_pull_line(d.get("solid_pull"))
    if pull:
        print(pull)
    for a in d["assumptions"]:
        print(f"assumption:     {a}")
    for line in robust.lines(dv["robust"]):   # robust mode (section 22b)
        print(f"robust:         {line[len('robust: '):]}")
    for line in dv["robust"]["facts"]:
        print(f"warning: {line}")


def start_dashboard(run_dir, port, tries=20):
    """Start the dashboard thread on the first free port from `port`.

    Return the server, or None if all the ports are busy.
    """
    from . import dashboard_server
    for p in range(port, port + tries):
        try:
            server = dashboard_server.start_in_thread(run_dir, p, open_browser=True)
        except OSError:
            continue
        if p != port:
            print(f"Dashboard port {port} is busy. Port {p} is used.", flush=True)
        print(f"Dashboard: {server.url}", flush=True)
        return server
    print(f"Dashboard ports {port} to {port + tries - 1} are busy. No dashboard.", flush=True)
    return None


def recommendation_line(rec: dict) -> str:
    """The offer of `solver_choice.recommendation` as one text line for the human output of `topopt check`.

    The JSON output (`check --json`) and the MCP tools keep the dict with its keys."""
    speedup = rec.get("speedup_factor")
    return (f"recommendation: {rec['reason']} Command ({rec['action']}, extra '{rec['extra']}'): {rec['install_command']}"
            + (f" (about {speedup:g} times faster for this problem)" if speedup else "")
            + f". {rec['license_note']}")


def run_memory_lines(problem, rep: dict, reduced=None, solver=None, case=None) -> list[str]:
    """Log lines at the start of `topopt run`: the memory estimate with the filter, and the
    warnings of `memory_warnings`. `rep` is `filter_report` with the size. They are facts: the run goes on.
    `reduced` is `optimize.reduced_system(problem)` if the caller has it (it builds the table of all active voxels).
    `solver` is the resolved solver. gpu-mg is matrix-free: it makes no factor and no bordered copy of `K` (P2b), so its number is the gpu-mg model of
    `gpu_mg.check.estimate_memory` (the largest of the filter setup, the loop and the mesh step; the same number as `topopt check`). `case` is the
    precision case of the device (None: it is found from `settings.gpu_device` and `settings.gpu_case`)."""
    from .optimize import reduced_system
    _edof, _active, free = reduced_system(problem) if reduced is None else reduced
    grid = int(math.prod(problem.shape))
    fsz = rep["size"]
    ram = physical_ram_bytes()
    gpu_text = ""
    if solver == "gpu-mg":
        from .gpu_mg.check import problem_memory
        st = problem.config.settings
        if case is None:
            from .gpu_mg import device as devmod
            try:
                case = devmod.select(st.gpu_device, st.gpu_case)[1]
            except UserError:       # the run stops at the device with the fact; the memory line needs no device
                case = None
        joints = 0 if problem.joints is None else len(problem.joints.joints)
        m = problem_memory(int(free.size), grid, case, joints, fsz)
        mem, without, bord = m["bytes"], m["without_filter"], 0
        gpu_text = f"; the gpu-mg model, set by its {m['phase']} phase"
    else:
        bord = bordered_bytes(problem, _edof, free)
        mem = estimate_memory_bytes(grid, int(free.size), fsz["held_bytes_estimate"], bord)
        without = estimate_memory_bytes(grid, int(free.size), 0, bord)
    lines = [f"memory estimate: about {gib(mem)}" + (f" of {gib(ram)} RAM" if ram else " (RAM size not known here)")
             + f" (the filter is part of it: about {size_text(fsz['held_bytes_estimate'])}"
             + (f"; the bordered copy of K for the joints: about {size_text(bord)}" if bord else "") + gpu_text + ")"]
    lines += ["warning: " + w for w in memory_warnings(mem, ram, fsz, without)]
    return lines


def buckling_row_lines(config, problem, solver, rep: dict, reduced=None) -> list[str]:
    """Log lines at the start of `topopt run` for the buckling row (CONTRACTS.md section 39): the memory estimate with the row,
    the note on the time, and the warnings. They are facts: the run goes on. No line without `buckling.min_factor`.
    `reduced` is `optimize.reduced_system(problem)` if the caller has it: the row does not build the table a second time."""
    if config.buckling.min_factor is None:
        return []
    from . import buckling_row
    from .optimize import reduced_system
    _edof, _active, free = reduced_system(problem) if reduced is None else reduced
    grid = int(math.prod(problem.shape))
    run_bytes = estimate_memory_bytes(grid, int(free.size), rep["size"]["held_bytes_estimate"])
    info = buckling_row.preflight(config, solver, grid, int(free.size), physical_ram_bytes(), run_bytes=run_bytes)
    return buckling_row.lines(info) + ["warning: " + w for w in info["warnings"]]


def cmd_run(args) -> int:
    from . import checkpoint
    from .runjob import run_job
    t_start = time.time()
    config = load_config(args.yaml)
    config = with_stop_rules(config, getattr(args, "stop_rules", None))
    if getattr(args, "mesh_method", None):   # the flag wins over settings.mesh_method
        config = dataclasses.replace(config, settings=dataclasses.replace(config.settings, mesh_method=args.mesh_method))
    if getattr(args, "envelope_fill", None):   # the flag wins over settings.envelope_fill
        config = dataclasses.replace(config, settings=dataclasses.replace(config.settings, envelope_fill=args.envelope_fill))
    if getattr(args, "solver", None):    # --solver replaces settings.solver for this run
        config = dataclasses.replace(config, settings=dataclasses.replace(config.settings, solver=args.solver))
    out_dir = Path(args.out)
    resume, allow = getattr(args, "resume", False), getattr(args, "allow_changed_inputs", False)
    if getattr(args, "fine_pass", False):   # #318: the first run in --out, then the fine pass in the sibling folder (CONTRACTS.md section 24.6)
        _fine_pass_arguments(args)
        return _run_then_fine(args, out_dir)
    fine = _fine_arguments(args, config, out_dir)   # #318: a FineSetup (the parent is read once) or None; the config has the fine voxel and maximum
    if fine is not None:
        config = fine.config
        if getattr(args, "fine_forced_fact", None):   # --fine-pass with a --fine-voxel that the part does not need: the fact is in the log of the fine run too
            fine.facts.insert(0, args.fine_forced_fact)
    continuing = _continue_arguments(args, config, out_dir)   # P4b-5: (parent, more, iterations, ck) or None; the config has K + N passes
    if continuing is not None:   # a parent that is a fine pass (#318): its voxel, schedule and stop rule first, then the pass count of the continuation
        config = checkpoint.with_iterations(
            checkpoint.config_for_continuation(continuing[3], config, getattr(args, "stop_rules", None)), continuing[2])
    if allow and not resume and continuing is None and fine is None:
        raise UserError("--allow-changed-inputs needs --resume, --continue-from or --fine-from")
    if resume and not out_dir.is_dir():
        raise UserError(f"--resume: the run folder {out_dir} does not exist. Give the folder of the run that you want to continue.")
    if resume:
        config = checkpoint.config_of_fine(out_dir, config, getattr(args, "stop_rules", None))   # #318: the run of the folder is a fine pass: its voxel, maximum, schedule and stop rules are in the checkpoint; --stop-rules / --no-stop-rules wins
        config = checkpoint.config_of_continuation(out_dir, config)   # P4b-5: the run of the folder is a continuation: its pass count is in the checkpoint (after the fine pass)
    # A continuation or a fine pass that cannot go on stops here, before the new folder exists: a refusal makes no folder. The parent is only read.
    # The one read of the checkpoint of the parent (`continuing[3]`) goes to both plans: a parent that runs can write a later pass in between.
    if fine is not None:
        held = [_plan_fine(args, fine, out_dir)]
    else:
        held = [_plan_continue(args, config, out_dir, ck=continuing[3])[0]] if continuing is not None else [None]
    held_ck = [continuing[3] if continuing is not None else None]
    continuing = None   # `held_ck` keeps the one read for `_run_folder`; no other reference stays in this frame
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "second_of_fine", False) and _stop_before_fine(Path(args.fine_from), out_dir, getattr(args, "quiet", False)):
        from .runjob import STOPPED_EXIT   # --fine-pass: a request came while the fine grid was planned; the fine pass does not start
        return STOPPED_EXIT
    # A resume that cannot go on stops here, before the job files: a refusal must not change the record of the run in the folder
    # (a finished run keeps its done marker). This read-only plan is made again under the lock.
    if resume:   # F4-47: a folder that a killed run left with the earlier run aside is made right under the lock (`_run_folder`), then planned
        from .snapshot import has_aside
        held = [None if has_aside(out_dir) else _plan_resume(args, config, out_dir)[0]]
    # Job lock, time, peak memory, done or FAILED marker (CONTRACTS.md section 24.1). A held lock stops here.
    # `held.pop()` hands the model of a resume to `_run_folder` and leaves no other reference in this frame or in the lambda:
    # the wait after the run drops the model (P4 review, L7).
    # A resume and a continuation built the problem above, before the job. The voxel build of a fine grid can be the peak of the whole command
    # (a 0.5 mm grid crashed a 16 GB Mac), so the job does not reset the peak memory: its peak is the larger of the start and the end read.
    keep_peak = None if held[0] is None else ("the plan of this command (a resume, a continuation or a fine pass) built the problem before the job, "
                                              "and that build can be the peak of the command")
    record = ({"fine_pass": {"first_run": True, "fine_run_dir": str(fine_folder(out_dir)), "forced": getattr(args, "fine_forced_voxel", None) is not None}}
              if getattr(args, "first_of_fine", False) else None)   # job.json marks the first run of --fine-pass: `stop_reply` writes a request in its post step
    return run_job(out_dir, lambda job: _run_folder(args, config, out_dir, t_start, job, held.pop(), held_ck.pop(), fine),
                   override=getattr(args, "override_lock", False), t_start=t_start, keep_peak=keep_peak, record=record)


def _continue_arguments(args, config, out_dir):
    """`--continue-from PARENT --more-iterations N` (P4b-5): return `(parent, more, iterations, ck)`, or None if the run is not a continuation.

    The two options go together, and they do not go with `--resume` (a resume goes on in `--out`; a continuation starts a new run in `--out`).
    `iterations` is the pass count of the new run: the pass `K` of the parent checkpoint plus `N`. `ck` is the one read of that checkpoint:
    the plans use it, and nothing in this process reads the checkpoint of the parent again (a parent that runs can write a later pass).
    Raise `UserError` with options for a wrong combination, a value of N below 1, or a new folder that is the parent or inside it.
    """
    from . import checkpoint
    parent, more = getattr(args, "continue_from", None), getattr(args, "more_iterations", None)
    if parent is None and more is None:
        return None
    if parent is None or more is None:
        raise UserError("--continue-from and --more-iterations go together. --continue-from names the run folder to continue, and "
                        "--more-iterations N says how many passes to add. Options: give both, for example "
                        "`--continue-from runs/hook --more-iterations 20`.")
    if getattr(args, "resume", False):
        raise UserError("--resume goes on in the folder of --out, and --continue-from starts a new run in --out from another run folder. "
                        "Options: give --resume only, to go on in --out; or --continue-from only, to add passes to a finished run in a new folder.")
    n = checkpoint.check_more_iterations(more)
    parent = Path(parent)
    if not parent.is_dir():
        raise UserError(f"--continue-from: {parent} is not a folder. Give the run folder that you want to continue (the --out of the earlier `topopt run`).")
    checkpoint.check_new_folder(parent, out_dir)
    ck = checkpoint.read_checkpoint(parent, checkpoint.CONTINUE_RESTART_OPTION)   # raises ResumeRefused with the reason for each file
    return parent, n, ck.iteration + n, ck


FINE_SUFFIX = "-fine"   # `--fine-pass`: the fine pass goes to the sibling folder `<--out>-fine` (a folder inside the parent is refused: `check_new_folder`)


def fine_folder(out_dir) -> Path:
    """The folder of the fine pass of the one-command form `--fine-pass`: a sibling of `--out`, named `<name>-fine`."""
    out_dir = Path(out_dir).resolve()   # `--out .` has no name: the sibling is built from the resolved path
    return out_dir.with_name(out_dir.name + FINE_SUFFIX)


def _fine_options_error(args, has_fine: bool) -> None:
    """`--fine-voxel`, `--fine-passes` and `--fine-tol` go with `--fine-from` or `--fine-pass` only. Raise `UserError` with options."""
    given = [f"--{n.replace('_', '-')}" for n in ("fine_voxel", "fine_passes", "fine_tol") if getattr(args, n, None) is not None]
    if given and not has_fine:
        raise UserError(f"{', '.join(given)} {'goes' if len(given) == 1 else 'go'} with --fine-from or --fine-pass: they set the fine pass. "
                        "Options: add --fine-pass to run the first run and then the fine pass; or --fine-from PARENT to run the fine pass "
                        "from a finished run folder; or remove the option.")


def _fine_pass_arguments(args) -> None:
    """`--fine-pass` (#318): the first run in `--out`, then the fine pass in `<--out>-fine`. Raise `UserError` with options for a wrong mix."""
    for name, flag in (("resume", "--resume"), ("continue_from", "--continue-from"), ("more_iterations", "--more-iterations"),
                       ("fine_from", "--fine-from"), ("allow_changed_inputs", "--allow-changed-inputs")):
        if getattr(args, name, None):
            raise UserError(f"--fine-pass does not go with {flag}. --fine-pass makes a new first run in --out and then the fine pass in "
                            f"{fine_folder(args.out)}. Options: remove {flag}; or run the two steps by hand: `topopt run <yaml> --out <first> "
                            "--resume`, then `topopt run <yaml> --out <fine> --fine-from <first>`.")


def _fine_arguments(args, config, out_dir):
    """`--fine-from PARENT` (#318): return the `fine_pass.FineSetup`, or None if the run is not a fine pass. The parent is read once and only read.

    `--fine-voxel`, `--fine-passes` and `--fine-tol` go with it. It does not go with `--resume`, `--continue-from` or `--more-iterations`
    (a fine pass is a new run in a new folder). Raise `UserError` with options for a wrong mix, a parent that did not finish, a new folder that
    is the parent or inside it, or a fine voxel equal to the one of the parent.
    """
    from . import fine_pass
    parent = getattr(args, "fine_from", None)
    _fine_options_error(args, parent is not None)
    if parent is None:
        return None
    for name, flag in (("resume", "--resume"), ("continue_from", "--continue-from"), ("more_iterations", "--more-iterations")):
        if getattr(args, name, None):
            raise UserError(f"--fine-from does not go with {flag}. A fine pass is a new run in --out, on a finer voxel, from the design of the "
                            f"finished run in PARENT. Options: remove {flag}; --resume alone goes on in --out; --continue-from with "
                            "--more-iterations adds passes on the same grid.")
    return fine_pass.resolve_fine(config, parent, out_dir, getattr(args, "fine_voxel", None), getattr(args, "fine_passes", None),
                                  getattr(args, "fine_tol", None), early_stop=getattr(args, "stop_rules", None) is not False)


def _plan_fine(args, fine, out_dir, problem=None):
    """`--fine-from`: build the problem of the fine voxel and compare the inputs with the checkpoint of the parent. Write nothing.

    Return the problem. The facts of the comparison go to `fine.facts`, the lines of the run log. A fine pass that cannot go on raises
    `ResumeRefused` (a `UserError`) and the folder `out_dir` is not made.
    """
    from . import fine_pass
    from .checkpoint import ResumeRefused, plan_fine
    from .problem import build_problem
    config = fine.config
    try:
        problem = build_problem(config) if problem is None else problem   # input errors first, then the platform check
        row_msg = buckling_row_refusal(config)
        if row_msg:
            raise UserError(row_msg)
        problem_msg = backend_problem(config.settings.solver, None, config.settings)
        if problem_msg:
            raise UserError(problem_msg)
        solver = run_solver(config.settings.solver, config).solver      # the solver that the run takes first (`auto` on a GPU: gpu-mg), as the plans of --resume and --continue-from
        if fine.parent.ck is None:
            fine.facts.append("fine pass: the parent has no checkpoint, so its inputs are not compared with this run.")
        else:
            fine.facts += plan_fine(fine.parent.ck, config, problem, solver, getattr(args, "allow_changed_inputs", False), fine.parent_dir)
        return problem
    except ResumeRefused:
        raise
    except UserError as e:
        raise with_file(fine_pass.name_fine_option(e), config.yaml_path)   # the guards of the build say voxel_size: a fine pass has --fine-voxel


def _fine_pass_precheck(args) -> str | None:
    """`--fine-pass` (#318): test the planned fine pass BEFORE the first run starts, so that no first run is wasted. Return a note or None.

    The check builds the grid of the first run (as `topopt check` does) and takes the fine voxel: `--fine-voxel` when given, else the voxel that the
    part needs (`part_voxel` with `verify`: the same call that the fine pass will make, so the two agree).
    1. A fine voxel equal to or coarser than the voxel of the first run is a `UserError` with options, and nothing is written. (The fine pass would
       be refused after the first run: the same voxel adds nothing, `--continue-from` adds passes on the same grid.)
    2. The fine grid at that voxel is tested against `settings.max_active_voxels`, `settings.max_grid_voxels` and the memory estimate. A failure is a
       `UserError` with the fact and the options.
    3. Without `--fine-voxel`, a part that needs no finer voxel has no fine pass: the first run goes on and ends with the fact (exit code 0). The
       return value is the note that says this at the start of the first run.
    4. A `--fine-voxel` that is finer than the run voxel wins (tools inform, the user decides; `--fine-from ... --fine-voxel` forces it too): the fine
       pass runs at that voxel also when the part needs no finer one. The return value is the fact `forced_fine_fact` ("the part needs about X mm;
       --fine-voxel F is finer, so the fine pass runs at F as asked"), and the grid of that voxel goes through the guards (step 2). Else None.
    A bad YAML is the error that the first run would give.
    """
    from . import fine_pass
    cfg = load_config(args.yaml)
    try:
        d = check_data(args.yaml, solver_setting=getattr(args, "solver", None))
    except UserError as e:
        raise with_file(e, args.yaml)
    block = d.get("fine_pass") or {}
    run_h = float(d["grid"]["voxel_mm"])
    given = getattr(args, "fine_voxel", None)
    note = None
    if given is not None:
        voxel = fine_pass.check_fine_voxel(given)
        if voxel >= run_h - fine_pass.SAME_VOXEL:
            part = block.get("voxel_mm")
            raise UserError(
                f"--fine-pass: --fine-voxel {voxel:g} mm is {'the same as' if abs(voxel - run_h) < fine_pass.SAME_VOXEL else 'coarser than'} the voxel of the "
                f"first run, {run_h:g} mm (settings.voxel_size). The fine pass must have a finer voxel than the first run, so --fine-pass stops before "
                "the first run: no first run is made for a fine pass that cannot start. "
                f"Options: give --fine-voxel a value below {run_h:g}"
                + (f" (the part needs about {float(part):g} mm: `topopt check` prints it)" if part else "")
                + "; remove --fine-voxel to use the voxel that the part needs; remove --fine-pass to make the first run only.")
        if not block.get("recommend"):
            note = forced_fine_fact(block, run_h, voxel)   # the part needs no finer voxel: the voxel that the user gave wins
    else:
        if not block.get("recommend"):
            return ("fine pass: --fine-pass is on, but " + _no_fine_pass_reason(block, run_h)
                    + " There will be no fine pass: the run ends after the first run, with exit code 0.")
        try:
            voxel = float(fine_pass.part_voxel(cfg, verify=True)["voxel_mm"])   # the voxel that the fine pass will take: tested on the grid of the fine pass
        except UserError as e:
            raise with_file(e, args.yaml)
    guards = fine_pass.fine_grid_guards(cfg, d, voxel, block.get("max_passes"))
    if guards and not guards["ok"]:
        raise UserError(guards["text"].replace("A fine pass at this voxel stops when it builds the grid.",
                                               "--fine-pass stops before the first run: no first run is made for a fine pass that cannot start."))
    return note


def forced_fine_fact(block: dict, run_h: float, voxel: float) -> str:
    """The fact of a `--fine-voxel` that is finer than the run voxel when the part needs no finer voxel (the block of `part_voxel_block`).

    It goes to the start of the first run and to the log of the fine run of `--fine-pass`: the voxel that the user gave wins, and the text says what
    the measure of the part found.
    """
    return (f"fine pass: {_no_fine_pass_reason(block, run_h, about=True)} --fine-voxel {voxel:g} mm is finer, "
            f"so the fine pass runs at {voxel:g} mm as asked.")


def _no_fine_pass_reason(block: dict, run_h: float, about: bool = False) -> str:
    """Why the part has no fine pass: the measured voxel is not below the voxel of the run by a step, or the measure failed (the block of `part_voxel_block`)."""
    from . import fine_pass
    if block.get("error"):
        return f"the voxel of the part could not be measured ({block['error']})."
    lim = block.get("limiting") or {}
    return (f"the part needs {'about' if about else 'a voxel of'} {block.get('voxel_mm', 0):g} mm (the {str(lim.get('role', 'feature')).replace('_', '-')} "
            f"'{lim.get('name', '?')}': {lim.get('thickness_mm', 0):g} mm thick), which is not finer than the voxel of this run, {run_h:g} mm, "
            f"by {fine_pass.STEP_MM:g} mm or more.")


def _run_then_fine(args, out_dir) -> int:
    """`--fine-pass` (#318): the normal run in `out_dir`, then the fine pass from it in `fine_folder(out_dir)`. Return the exit code.

    The first run ends with the exit code of a run. When it is not 0 (an error, a request to stop) there is no fine pass. When the part voxel is
    not smaller than the voxel of the run (`run.json` `part_voxel.recommend` is false), the first run has logged the fact and this function ends
    with exit code 0, unless `--fine-voxel` was given: the pre-check refused a voxel that is not finer than the run, so a given voxel is finer, it wins,
    and the fine pass runs at it (the fact `forced_fine_fact` is at the start of both logs). The dashboard of the first run is released at its
    end, so that the fine pass starts its own on the same port.
    A request to stop that comes after the first run has read its last request (its post step, its refine step, the gap before the fine pass, and the
    plan of the fine grid) stops the fine pass from starting: `_stop_before_fine` reads it in the first folder and in the fine folder, and the exit
    code is `STOPPED_EXIT`.
    """
    from .runjob import STOPPED_EXIT
    note = _fine_pass_precheck(args)   # the fine voxel and the fine grid against the guards, before the first run writes anything
    forced = getattr(args, "fine_voxel", None)   # a given voxel passed the pre-check: it is finer than the run, and the user decides
    first = argparse.Namespace(**vars(args))
    first.fine_pass, first.first_of_fine, first.fine_note = False, True, note   # the note goes to the log at the start of the first run
    first.fine_forced_voxel = None if forced is None else float(forced)   # the first run logs that the fine pass follows although the part needs no finer voxel
    first.fine_voxel = first.fine_passes = first.fine_tol = None   # these set the fine pass; the first run does not take them
    code = cmd_run(first)
    if code != 0:
        return code
    try:
        block = json.loads((out_dir / "run.json").read_text()).get("part_voxel") or {}
    except (OSError, ValueError):
        block = {}
    if not block.get("recommend") and forced is None:
        return 0
    second = argparse.Namespace(**vars(args))
    second.fine_pass, second.fine_from, second.out = False, str(out_dir.resolve()), str(fine_folder(out_dir))
    second.fine_forced_fact = note if forced is not None else None   # the note of the pre-check is the fact `forced_fine_fact` when the part needs no finer voxel
    second.second_of_fine = True   # cmd_run reads the request to stop once more, after the plan of the fine grid and before the job starts
    if _stop_before_fine(out_dir, Path(second.out), getattr(args, "quiet", False)):
        return STOPPED_EXIT
    if not getattr(args, "quiet", False):
        print(f"fine pass: the first run is in {out_dir}. The fine pass goes to {second.out}.", flush=True)
    return cmd_run(second)


def _stop_before_fine(first_dir: Path, fine_dir: Path, quiet: bool = False) -> bool:
    """`--fine-pass`: True if a request to stop for this process is in the first run folder or in the fine run folder, so that the fine pass does not start.

    The first run does not read a request in its post step and its refine step, and between the two runs no job holds a lock. A request
    that comes in these windows stays: in the first folder (`_run_folder` keeps it for a first run of `--fine-pass`), or in the fine folder
    (the MCP tool `stop_run` writes it there, with the pid of this process). The request is removed, a fact is printed and added to the log of the
    first run, and an empty fine folder that the request made is removed. The first run is finished and keeps its result.
    """
    from .runjob import clear_stop, read_own_stop, stop_text
    for folder in (Path(first_dir), Path(fine_dir)):
        request = read_own_stop(folder)
        if request is None:
            continue
        clear_stop(folder)
        text = (f"fine pass: a stop request came after the first run: the fine pass did not start. The first run in {first_dir} is finished and keeps "
                f"its result. The request ({stop_text(request)}) is removed. The fine pass can start later: `topopt run <yaml> --out {fine_dir} "
                f"--fine-from {first_dir}`.")
        try:
            with open(Path(first_dir) / "run.log", "a") as f:
                f.write(text + "\n")
        except OSError:
            pass
        if not quiet:
            print(text, flush=True)
        try:
            Path(fine_dir).rmdir()    # the folder that a request made: only when it is empty
        except OSError:
            pass
        return True
    return False


def cmd_stop(args) -> int:
    """`topopt stop <run_dir>`: ask the run in the folder to stop at the end of its current pass (CONTRACTS.md section 24.2).

    It writes `<run_dir>/job.stop` and returns at once: no signal, no wait. The run saves the checkpoint of the pass, ends with status `stopped`
    and exit code 5, and `topopt run ... --resume` continues it. Exit code 0: a request is in the folder (written now, or there before). Exit code 1:
    nothing was written (the run is in its post step, it has ended, it is gone, or the folder has no run); the text says why and gives options.
    """
    from .runjob import stop_reply
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise UserError(f"{run_dir} is not a folder. Give the run folder (the --out of `topopt run`).")
    reply = stop_reply(run_dir, by="topopt stop")
    print(reply["message"])
    for option in reply["options"]:
        print(f"  option: {option}")
    print(reply["note"])
    return 0 if reply["state"] in ("stop_requested", "already_requested") else 1


def with_stop_rules(config, enabled):
    """The config with the stop rules turned on or off (`topopt run --stop-rules` / `--no-stop-rules`). None: as the YAML says."""
    if enabled is None:
        return config
    import dataclasses
    st = config.settings
    return dataclasses.replace(config, settings=dataclasses.replace(
        st, stop_rules=dataclasses.replace(st.stop_rules, enabled=bool(enabled))))


def _plan_resume(args, config, out_dir, problem=None):
    """`--resume`: read the checkpoint, build the problem and compare the inputs. Write nothing to the run folder.

    Return `(problem, plan)`. A resume that cannot go on raises `ResumeRefused` (a `UserError`): the folder stays as it is.
    `problem` is the one that an earlier call built (the voxelization is not made twice).
    """
    from .checkpoint import ResumeRefused, plan_resume
    from .problem import build_problem
    try:
        problem = build_problem(config) if problem is None else problem   # input errors first, then the platform check
        row_msg = buckling_row_refusal(config)   # the buckling row needs a direct solver: before `backend_problem` (gpu-mg needs OpenCL)
        if row_msg:
            raise UserError(row_msg)
        problem_msg = backend_problem(config.settings.solver, None, config.settings)
        if problem_msg:
            raise UserError(problem_msg)
        solver = run_solver(config.settings.solver, config).solver
        return problem, plan_resume(out_dir, config, problem, solver, getattr(args, "allow_changed_inputs", False))
    except ResumeRefused:
        raise
    except UserError as e:
        raise with_file(e, config.yaml_path)


def _plan_continue(args, config, out_dir, problem=None, ck=None):
    """`--continue-from`: build the problem and compare the inputs with the checkpoint of the parent. Write nothing, in the parent or in `--out`.

    Return `(problem, plan)`. A continuation that cannot go on raises `ResumeRefused` (a `UserError`). `config` has the pass count of the continuation.
    `ck` is the one read of the checkpoint (`_continue_arguments`); `cmd_run` and `_run_folder` give the same read to their plans.
    """
    from .checkpoint import ResumeRefused, plan_continue
    from .problem import build_problem
    try:
        problem = build_problem(config) if problem is None else problem   # input errors first, then the platform check
        row_msg = buckling_row_refusal(config)
        if row_msg:
            raise UserError(row_msg)
        problem_msg = backend_problem(config.settings.solver, None, config.settings)
        if problem_msg:
            raise UserError(problem_msg)
        solver = run_solver(config.settings.solver, config).solver
        return problem, plan_continue(args.continue_from, config, problem, solver, args.more_iterations,
                                      getattr(args, "allow_changed_inputs", False), new_dir=out_dir, ck=ck)
    except ResumeRefused:
        raise
    except UserError as e:
        raise with_file(e, config.yaml_path)


def check_run_inputs(config, problem=None):
    """Every check of the input that stops a fresh run before its first pass. Return `(problem, choice)`: `choice` is the `SolverChoice` of the run
    (`solver_choice.run_solver`: the solver that runs first, and for `auto` the GPU facts and the CPU fallback). Write nothing, log nothing.

    The order is the order of the run: `build_problem` (the voxel guards, the region files and the load regions; skipped if the caller has the
    problem), a union piece with no fixed voxel, a piece of active voxels with no support (CONTRACTS.md section 3), a perimeter bound below
    the smallest possible perimeter, the buckling row with its solver, the platform and the backend, the solver that runs, robust mode with a
    stress row or the buckling row (F4-47), and last the rows of `limits` (A5; they need the voxel grid only, not the model).
    A refusal is a `UserError`. The run log lines that come from these steps are written by `_run_folder` after the folder is cleaned.
    """
    from .problem import build_problem
    problem = build_problem(config) if problem is None else problem   # input errors first, then the platform check
    if config.interfaces:   # a union piece with no fixed voxel makes the stiffness matrix singular: stop before the solve
        from . import interfaces
        from .springs import support_mask

        floating = interfaces.floating_message(
            interfaces.voxel_facts(config, problem.labels, problem.origin, problem.h, problem.shape,
                                   support_mask(problem)))
        if floating:
            raise UserError(floating)
    from .connectivity import floating_error
    loose = floating_error(problem)   # a piece of active voxels with no support: the matrix is singular (CONTRACTS.md section 3)
    if loose:
        raise UserError(loose)
    from .perimeter import check_facts as perimeter_facts   # after the loose pieces: fixing one changes the labels and so P_min
    perimeter_facts(config, problem)   # A4c: a perimeter bound below the smallest possible P stops here, before the model is built
    row_msg = buckling_row_refusal(config)   # the buckling row needs a direct solver: before `backend_problem` (gpu-mg needs OpenCL)
    if row_msg:
        raise UserError(row_msg)
    problem_msg = backend_problem(config.settings.solver, None, config.settings)   # the platform rule, then the backend
    if problem_msg:
        raise UserError(problem_msg)   # the run stops here, before the solve
    choice = run_solver(config.settings.solver, config)   # `auto` on a GPU that passes the self-test: gpu-mg, with the CPU solver as the fallback
    robust.from_settings(config.settings, sum(1 for c in config.load_cases if c.stress_limit_mpa is not None),   # F4-47: robust mode with a stress row
                         buckling_row=config.buckling.min_factor is not None)   # or the buckling row: `Model` makes the same call and stops
    if config.limits.displacement or config.limits.alignment:   # F4-47: a row of `limits` that cannot work (A5): `optimize.run` binds it to the model
        from .optimize import reduced_system
        from .rows_motion import check_rows
        check_rows(problem.config, problem, reduced_system(problem)[2])
    return problem, choice


def _run_folder(args, config, out_dir, t_start, job, problem=None, ck=None, fine=None) -> int:
    """The run, after the job lock is taken. `problem` is the one that the read-only plan of `cmd_run` built (a resume, a continuation, a fine pass).
    `ck` is the checkpoint of the parent that `cmd_run` read (a continuation): this plan uses that read, not a new one.
    `fine` (#318) is the `fine_pass.FineSetup` of a fine pass: the run starts from the design of its parent (`optimize.run(x0=...)`).

    Returns 0 for a finished run, and `STOPPED_EXIT` (5) for a run that a request to stop ended (`topopt stop`, MCP `stop_run`). A request is read
    at the end of each pass (`optimize.run`) and once before the post step; a stopped run has status `stopped`, the job marker `job.STOPPED`,
    no `result/`, and a checkpoint that `--resume` continues. The dashboard stays on after a stop, like after a finished run.
    """
    from .checkpoint import config_of_continuation, config_of_fine, describe
    from .runjob import STOPPED_EXIT, clear_stop, read_own_stop, stop_text
    from .snapshot import RunWriter, old_run_files, recover_earlier_run
    recovered = recover_earlier_run(out_dir)   # F4-47: a run that a kill stopped before its first saved pass left the earlier run aside: move it back
    if getattr(args, "resume", False):
        if recovered:   # `cmd_run` read the checkpoint of the folder before the earlier run was back: a continuation has its pass count there
            config = config_of_continuation(out_dir, config_of_fine(out_dir, config, getattr(args, "stop_rules", None)))
        problem, plan = _plan_resume(args, config, out_dir, problem)
    elif getattr(args, "continue_from", None) is not None:
        problem, plan = _plan_continue(args, config, out_dir, problem, ck)   # P4b-5: the parent is only read, its checkpoint once
    else:
        problem, plan = None, None
    fresh = plan is None or plan.cont is not None   # a new run, and a continuation, start in a cleaned folder (they delete the files of an old run)
    server = start_dashboard(out_dir, args.port) if args.dashboard else None
    out = RunWriter(out_dir, t_start, quiet=args.quiet)
    if plan is not None:
        out.first_pass = None   # a resume or a continuation: status.json says null until `optimize.run` has read the checkpoint (E1-P2)
    job.watch_phases(out)   # DL1a: the job record takes setup_s, loop_s, post_s and the peak at the end of the loop from the writer
    earlier = describe(out_dir) if fresh else None   # a fresh run deletes the checkpoint of a run that did not finish
    choice = None   # the `SolverChoice` of the run (`check_run_inputs`)
    kept = old_run_files(out_dir) if fresh else []
    def before_clean(text):   # the status text of a stop before the delete: the earlier run, if there is one, is in place (F4-43)
        return with_kept_run(text, recovered) if kept else text
    # Every check of the input comes before the delete, so that a refusal keeps an earlier run (F4-43). The build stays here, inside the job: the job
    # record and its peak memory cover it. A folder with no earlier run loses nothing by a refusal, so it keeps the old order (`starting` in status.json
    # at once, the checks after the clean).
    try:
        # F4-47: bind the names of `.optimize` here, before `out.clean()`. A Ctrl+C or an ImportError in the import of the `try` below would leave
        # `RunStopped` unbound: the clause `except RunStopped` then raises an `UnboundLocalError` that replaces the error, and the handlers that move
        # the earlier run back never run.
        from .optimize import RunStopped
        if kept:
            problem, choice = check_run_inputs(config, problem)
    except UserError as e:
        e = UserError(before_clean(str(with_file(e, config.yaml_path))))   # status.json only: no line goes to the old run.log
        out.status("error", 0, "setup", str(e))
        raise e
    except KeyboardInterrupt:
        out.status("error", 0, "setup", before_clean("stopped by the user (Ctrl+C)"))
        raise
    except Exception as e:
        out.status("error", 0, "setup", before_clean(f"internal error: {e}"))
        raise
    if fresh:
        out.clean()   # F4-47: with an earlier run in the folder, this moves it aside. The first saved pass deletes it (`Checkpointer.save`)
    stopped = None   # a RunStopped: a request to stop ended the run (P4b-4)
    try:   # F4-47: the lines up to `topopt run:` are inside the `try` too: after `out.clean()` an error or Ctrl+C there gives the earlier run back
        out.status("starting", 0, "setup")
        for fact in job.facts:
            out.log("warning: " + fact)
        if recovered:
            out.log("warning: " + recovered)
        if earlier and not earlier["complete"]:
            out.log(f"warning: the run folder had a checkpoint at pass {earlier['iteration']} of {earlier['iterations']} (written "
                    f"{earlier['written_at']}) from a run that did not finish. This run deletes it. `topopt run ... --resume` continues "
                    "that run instead.")
        out.log(f"topopt run: {config.name}")
        if getattr(args, "fine_note", None):   # --fine-pass (#318): what the pre-check found about the fine pass, at the start of the first run
            out.log(args.fine_note)
        if plan is not None and plan.cont is not None:
            from .checkpoint import apply_continue
            apply_continue(plan, out)   # copy history and snapshots from the parent (it is only read), state the facts
        elif plan is not None:
            from .checkpoint import apply_plan
            apply_plan(plan, out)   # cut history and snapshots to the checkpoint, state the facts
        if choice is None:   # no earlier run was in the folder: the checks run here, after the clean, as before F4-43
            problem, choice = check_run_inputs(config, problem)
        solver = choice.solver
        if choice.line():      # `auto`: what it picked and why (the GPU, the self-test, the fallback; or the reason for the CPU solver)
            out.log(choice.line())
        from .springs import near_singular_lines
        from .joints import near_singular_lines as joint_near_singular_lines
        hint = warning_line(config.settings.solver, solver)
        if hint:
            out.log("warning: " + hint)
        arch_hint = platform_hint()
        if arch_hint:
            out.log("warning: " + arch_hint)
        from .filter import solid_pull, solid_pull_line
        from .loads import extended_facts, extended_lines, lost_force
        from .voxelize import DESIGN, solid_lines, taken_lines, taken_warnings
        if problem.config.settings.target_mass_g is not None:   # I7: the volume fraction is derived from the mass; one line says how
            out.log(target_mass_line(problem))
        rep = filter_report(config.settings, problem.h, int((problem.labels == DESIGN).sum()), problem.shape)
        pull = solid_pull_line(solid_pull(problem.labels, problem.h, config.settings.filter_radius,
                                          config.settings.filter_solid_neighbours,
                                          pad_void=config.settings.robust == "wall"))
        for line in (filter_lines(rep) + ["min feature " + min_feature_line(rep)]
                     + ["warning: " + f for f in rep["facts"]]
                     + solid_lines(problem.region_facts.get("solid")) + ([pull] if pull else [])
                     + taken_lines(problem.region_facts.get("taken"))
                     + ["warning: " + w for w in taken_warnings(problem.region_facts.get("taken"))]):
            out.log(line)
        from .optimize import reduced_system
        reduced = reduced_system(problem)                  # one table of the active voxels for both memory lines
        for line in run_memory_lines(problem, rep, reduced, solver, choice.case):
            out.log(line)
        for line in buckling_row_lines(config, problem, solver, rep, reduced):   # the buckling row (section 39): none without it
            out.log(line)
        out.lost_force = lost_force(config, problem.load_region_ids, problem.fixed_dofs, problem.forces,
                                    (problem.extended or {}).get("entries"))
        for line in lost_force_lines(out.lost_force):
            out.log(line.replace("lost force:     ", "warning: lost force: ", 1)
                    .replace("lost moment:    ", "warning: lost moment: ", 1))
        for line in extended_lines(extended_facts(problem)):
            out.log(line)
        for line in near_singular_lines((problem.extended or {}).get("springs"), config.material.E, problem.h):
            out.log("warning: " + line)       # section 35: the same text as `check`
        for line in joint_near_singular_lines((problem.extended or {}).get("joints"), config.material.E, problem.h):
            out.log("warning: " + line)       # section 38: the same text as `check`
        rrep = robust.report(config.settings, problem.h, solver,   # robust mode (section 22b)
                             sum(1 for c in config.load_cases if c.stress_limit_mpa is not None),
                             buckling_row=config.buckling.min_factor is not None,
                             print_wall_mm=config.print_settings.limit("min_wall_mm"))
        for line in robust.lines(rrep) + ["warning: " + f for f in rrep["facts"]]:
            out.log(line)
        from .symmetry import end_lines, summary_lines
        for line in summary_lines(problem.mirror):
            out.log("mirror: " + line)
        from .solver_fallback import run_with_fallback
        fine_x0 = None
        if fine is not None:   # a fine pass: the mapped design of the parent is the start; the record goes to run.json
            from .fine_pass import fine_start
            fine_x0 = lambda: fine_start(fine)   # noqa: E731  (one new start for each start at pass 0)
        run_with_fallback(problem, out, plan, choice, fine_x0)   # `optimize.run`; with `auto` on a GPU, a failure of gpu-mg goes on with the CPU solver (P9-4d)
        if problem.mirror:   # the end values of the mirror deviation: run.json and the log
            out.write_mirror_end(problem.mirror)
            for line in end_lines(problem.mirror):
                out.log("mirror: " + line)
        part_block = write_part_voxel(out, config, problem, solver, reduced, args)   # #318: the voxel that the part needs, in run.json and (a recommendation) in the log
        asked = read_own_stop(out_dir)   # the one read before the post step: also for a request that came during the last pass
        if asked is not None:
            from .verdict import read_verdict
            ended = (read_verdict(out_dir) or {}).get("final_iteration")   # the last pass that ran (a stop rule can end the run early)
            raise RunStopped(config.settings.iterations if ended is None else ended, config.settings.iterations, asked, complete=True)
        out.status("postprocess", config.settings.iterations, "postprocess")
        out.problem = problem   # the post-process uses it for the stress section (A2): no second voxelization
        out.stress_cache = {}   # the stress section puts the voxel stress here; refine uses it (no second solve)
        check_on = buckling_wanted(problem, getattr(args, "buckling", None))
        if check_on:   # `done` comes after the check, so the post-process must not write it
            finish(out, out_dir, write_done=False)
            run_buckling(out, out_dir, problem)   # opt-in; run_time_s of metrics.json is final
        else:
            finish(out, out_dir)   # the call of the run without the check: unchanged
        write_part_card(out, out_dir)   # after the last fact of the run (the buckling check too), before `done`
        out.status("done", config.settings.iterations, "done")
        out.log(f"done in {out.elapsed():.1f} s")
        for line in robust.final_lines(out_dir):   # robust mode: the record of the run on one line
            out.log(line)
    except RunStopped as e:   # before the clauses below: it is not an error
        stopped = e
    except UserError as e:   # the three handlers: before the first saved pass, `kept_run_text` moves the earlier run back first (F4-47)
        e = with_file(e, config.yaml_path)
        text = kept_run_text(out, str(e), recovered)
        e = e if text == str(e) else UserError(text)
        out.status("error", out.cur_iteration, out.cur_step, str(e))
        raise e
    except KeyboardInterrupt:
        out.status("error", out.cur_iteration, out.cur_step, kept_run_text(out, "stopped by the user (Ctrl+C)", recovered))
        raise
    except Exception as e:
        out.status("error", out.cur_iteration, out.cur_step, kept_run_text(out, f"internal error: {e}", recovered))
        raise
    if stopped is None:
        run_refine(out, out_dir, problem, server)   # after "done": the run time stays the same
        leftover = read_own_stop(out_dir)   # a request for this process that came during the post step: nothing reads it any more
        if leftover is not None and _fine_follows(args, part_block):
            out.log(f"fact: a request to stop ({stop_text(leftover)}) came after the last pass. The post step does not read requests and the "
                    "run is finished. --fine-pass reads the request before it starts the fine pass, so the fine pass does not start.")   # `_stop_before_fine` removes it
        elif leftover is not None:
            clear_stop(out_dir)
            out.log(f"fact: a request to stop ({stop_text(leftover)}) came after the last pass. The post step does not read requests and "
                    "the run is finished, so the request is removed.")
        code = 0
    else:   # the checkpoint is saved; no post step and no refine step (P4b-4). The process still serves the dashboard, like a done run.
        out.status("stopped", stopped.iteration, "stopped", stopped.message)
        clear_stop(out_dir)
        out.log(f"stop: {stopped.message}. The request ({stop_text(stopped.request)}) is removed.")
        code = STOPPED_EXIT
    job.finish(code)   # DONE or STOPPED marker and time record now: the wait below can last until a signal
    if server is not None and getattr(args, "first_of_fine", False):   # #318: the fine pass starts its own dashboard on the same port
        server.shutdown()
        server.server_close()
        server = None
    if server is not None and not args.no_wait:
        # The run is over: the model, the solver tables and the stress cache are not needed. Drop them, so that the wait does not hold them.
        problem = reduced = plan = choice = solver = rep = rrep = pull = None
        out.problem = out.stress_cache = None
        keep_dashboard(server, out_dir, own_process=bool(getattr(args, "own_process", False)))
    return code


def _fine_follows(args, part_block) -> bool:
    """True for the first run of `--fine-pass` when the fine pass follows it: the part needs a finer voxel, or `--fine-voxel` was given (it wins)."""
    return bool(getattr(args, "first_of_fine", False)
                and ((part_block or {}).get("recommend") or getattr(args, "fine_forced_voxel", None) is not None))


def write_part_voxel(out, config, problem, solver, reduced, args) -> dict:
    """After the last pass (#318, CONTRACTS.md section 24.6): measure the voxel that the part needs, write the block `part_voxel` to run.json,
    and log the fact line when the voxel of this run is too coarse for the thinnest feature. A fact for the user: a failure of the measure goes to
    the block (`error`) and never stops the run.

    With `--fine-pass` (the first run of the one-command form) the log also says what comes next: the fine pass (also at a `--fine-voxel` that is
    finer than the run when the part needs no finer one), or no fine pass and why.
    """
    from . import fine_pass
    _edof, active_index, free = reduced
    from .voxelize import DESIGN
    check = {"grid": {"voxel_mm": float(problem.h), "voxels": int(math.prod(problem.shape)), "shape": [int(v) for v in problem.shape]},
             "active_voxels": int(active_index.size),
             "free_design_voxels": int((problem.labels == DESIGN).sum()), "free_dofs": int(free.size), "solver": {"resolved": solver}}
    block = fine_pass.part_voxel_block(config, check)
    out.write_part_voxel(block)
    if block["recommend"]:
        out.log(block["fact"])
    if getattr(args, "first_of_fine", False):
        forced = getattr(args, "fine_forced_voxel", None)
        if block["recommend"]:
            out.log(f"fine pass: --fine-pass is on. The fine pass starts next, in {fine_folder(args.out)}, from this run.")
        elif forced is not None:   # the user gave a voxel that is finer than the run: it wins, although the part needs no finer one
            out.log(f"fine pass: --fine-pass is on. The fine pass starts next, in {fine_folder(args.out)}, from this run, at {forced:g} mm as --fine-voxel says.")
        elif block.get("error"):
            out.log(f"fine pass: --fine-pass is on, but the voxel of the part could not be measured ({block['error']}). There is no fine pass.")
        else:
            lim = block["limiting"]
            out.log(f"fine pass: --fine-pass is on, but there is no fine pass. The voxel of this run, {block['run_voxel_mm']:g} mm, is not coarser "
                    f"than the voxel that the part needs, {block['voxel_mm']:g} mm, by a step of {fine_pass.STEP_MM:g} mm or more "
                    f"(the {lim['role'].replace('_', '-')} '{lim['name']}': {lim['thickness_mm']:g} mm thick). The run ends here.")
    return block


def keep_dashboard(server, out_dir, own_process: bool = False) -> None:
    """Keep the dashboard on after the run until the user stops the process (P4, F4-9).

    This is by design: the dashboard shows the result (the final screen, the refine page).
    The process that ran the optimizer holds the Python heap, the compiled numba code and the libraries of the run.
    That is about 590 MB for the 2 mm hook, and 1.7 MB of it are arrays. The server needs none of it.
    So when this process is the command line of the user (`own_process`: `main` got no argument list), it becomes the small
    dashboard of the run folder (`topopt dashboard`, same pid and process group).
    `os.execve` replaces the program and frees all of that memory.
    The listening socket stays open across the exec (`--listen-fd`), so the port is never closed.
    A request that comes during the exec waits in the backlog (P4 review, M1).
    If the exec does not work, the server stays in this process, as before.
    Ctrl+C or `kill <pid>` ends it either way.
    """
    pid = os.getpid()
    small = own_process and os.name == "posix"
    print(f"Dashboard stays on at {server.url} (pid {pid}"
          + (", a small process that only shows the result" if small else "")
          + f"). Press Ctrl+C to stop it, or run: kill {pid}", flush=True)
    if small:
        hand_over_dashboard(server, out_dir)   # does not return, unless the exec fails
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def flush_all_output() -> None:
    """Flush the Python streams and the C stdio buffers. The exec drops what is in a buffer.

    Libraries that print through C (numba, MKL, gmsh) leave text in the C buffers.
    `fflush(NULL)` flushes all of them (POSIX). On any error it does nothing.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        import ctypes
        ctypes.CDLL(None).fflush(None)
    except Exception:  # noqa: BLE001  (a missing libc symbol must not stop the dashboard)
        pass


def close_on_exec_except(keep: int) -> None:
    """Mark every open file descriptor above 2, except `keep`, as not inheritable (close on exec).

    Importing numba opens a handle on the Python program, with no close-on-exec flag. The new program would keep it.
    Errors are ignored. The descriptor of the listing itself is closed when the listing ends.
    """
    try:
        fds = [int(name) for name in os.listdir("/dev/fd")]      # Linux and macOS
    except (OSError, ValueError):
        return
    for fd in fds:
        if fd > 2 and fd != keep:
            try:
                os.set_inheritable(fd, False)
            except OSError:
                pass


def hand_over_dashboard(server, out_dir) -> None:
    """`os.execve` into `python -P -m topopt dashboard <run folder> --listen-fd <fd of server> --no-open`. Return only on a failure.

    The old process stops accepting first (`server.shutdown()`), so that it does not take a request that the exec would drop.
    A request that is in progress at the exec can still be reset. The page then asks again.
    """
    import gc
    gc.collect()
    fd = server.socket.fileno()
    # `-m` puts the current folder first on the path, so a `topopt.py` there would run instead of this package. `-P` does not
    # put it there, and PYTHONPATH names the folder of the package that runs now (an older PYTHONPATH stays after it).
    package_parent = str(Path(__file__).resolve().parent.parent)
    older = os.environ.get("PYTHONPATH")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([package_parent] + ([older] if older else [])))
    cmd = [sys.executable, "-P", "-m", "topopt", "dashboard", str(Path(out_dir).resolve()), "--listen-fd", str(fd),
           "--no-open"]
    flush_all_output()
    try:
        server.shutdown()
        close_on_exec_except(fd)        # a handle of a C library (numba opens one on the Python program) must not pass on
        os.set_inheritable(fd, True)    # the socket stays open and bound across the exec: the port is never closed
        os.execve(cmd[0], cmd, env)
    except (OSError, ValueError, TypeError) as e:   # a failed exec leaves this program as it was: serve again from here
        os.set_inheritable(fd, False)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"note: the dashboard stays in this process ({type(e).__name__}: {e})", flush=True)


def finish(out, out_dir, write_done=True):
    """Post-process. If topopt.post is missing, log one warning and go on.

    `write_done=False`: the post-process leaves the status at `postprocess` (a step follows).
    """
    try:
        from .post import finish as post_finish
    except ImportError:
        out.log("warning: topopt.post is not available, no final mesh")
        return
    post_finish(out_dir, getattr(out, "problem", None), getattr(out, "stress_cache", None), write_done=write_done)
    try:   # a run with a stress row: the relaxed result and the voxel estimates of the part stress in one fact (A2 R1-1)
        from .stress_report import summary_log_lines
        metrics = json.loads((Path(out_dir) / "result" / "metrics.json").read_text())
        from .volume_basis import mass_asked_line
        asked_line = mass_asked_line(metrics)   # I7: the final mesh against the asked mass, a run with settings.target_mass_g only
        if asked_line:
            out.log(asked_line)
        grey = metrics.get("grey_design")   # F1-3a: a grey mass run: the body count of final.stl next to the one of the thresholded part
        if grey and grey.get("text"):
            out.log("fact: " + grey["text"])
        for line in summary_log_lines(metrics.get("stress")):
            out.log(line)
        for w in (metrics.get("interfaces") or {}).get("warnings") or []:   # CONTRACTS.md section 23: what stays, full_surface, stock with no effect
            out.log("warning: " + w)
        if metrics.get("part_only"):   # regions.not_printed (CONTRACTS.md section 9.1): what the cut did, in the log and on the screen
            from .part_only import log_lines
            for line in log_lines(metrics["part_only"]):
                out.log(line)
    except (OSError, ValueError, ImportError):
        pass
    try:   # #445: a fallback of the mesh step in any round of the volume calibration; a block of its own, so no earlier line can drop it
        from .part_card import mesh_fallback_lines
        for line in mesh_fallback_lines(json.loads((Path(out_dir) / "result" / "metrics.json").read_text())):
            out.log("warning: " + line)
    except Exception:  # noqa: BLE001  (a log line must never fail a finished run)
        pass
    try:   # the print check of the post step (section 32): its states on one line; a block of its own, so no earlier line can drop it
        from .printcheck import REPORT_FILE, summary_line, summary_of
        line = summary_line(summary_of(json.loads((Path(out_dir) / "result" / REPORT_FILE).read_text())))
        if line:
            out.log(line)
    except Exception:  # noqa: BLE001  (a log line must never fail a finished run)
        pass


def write_part_card(out, out_dir) -> None:
    """The one-page part card `result/report.md` (CONTRACTS.md section 9.2). The run body calls it once, after the post step and the
    buckling check, so the card has every fact of the run. It only reads files; a failure is one log line and the run goes on."""
    try:
        from .part_card import write_logged
        card = write_logged(out_dir, out.log)
        if card is not None:
            out.log(f"part card: {card.relative_to(Path(out_dir)).as_posix()}")
    except Exception as e:  # noqa: BLE001  (the run is done: the card must never fail it)
        out.log("warning: part card not written: " + " ".join(f"{type(e).__name__}: {e}".split()))


def buckling_wanted(problem, flag=None) -> bool:
    """Is the linear buckling check on? The YAML block `buckling` (`check: true`) decides, and `flag`
    (`--buckling` true, `--no-buckling` false) overrides it. None: no flag was given."""
    return bool(problem.config.buckling.check if flag is None else flag)


def run_buckling(out, out_dir, problem) -> None:
    """The linear buckling check of the final design (CONTRACTS.md section 33). Never stop the run.

    The caller has decided that it is on (`buckling_wanted`). The result is the key `buckling` of
    `result/metrics.json`. A failure logs one warning and goes on. Ctrl+C during the check writes the
    `buckling` record `status: error, reason: interrupted`, logs one line and goes on: the run stays a
    finished run (`done`). A kill leaves the status `postprocess` with the message `buckling check`;
    `topopt buckling <run_dir>` repairs that.
    """
    out.status("postprocess", problem.config.settings.iterations, "postprocess", "buckling check")
    try:
        from .buckling_voxel import run_check
        run_check(out_dir, problem, log=out.log)
    except KeyboardInterrupt:
        from . import buckling as bk
        from .buckling_voxel import error_result
        bk.write_metrics_key(out_dir, error_result(problem.config.buckling.density,
                                                   "interrupted by the user (Ctrl+C) during the check",
                                                   "interrupted"))
        out.log("warning: buckling check interrupted (Ctrl+C). The run is finished. "
                f"`{bk.RERUN}` repeats the check.")
    except Exception as e:  # the run is done: a check problem must not break it
        out.log("warning: buckling check failed: " + " ".join(f"{type(e).__name__}: {e}".split()))


def run_refine(out, out_dir, problem, server) -> None:
    """Make the refine page after the run is done. Never stop the run.

    On a failure, log one warning line and go on. With a dashboard server,
    open the page in the browser (the real port of the server).
    """
    t0 = time.time()
    try:
        from .refine import build_refine
        path = build_refine(out_dir, problem=problem, stress=(getattr(out, "stress_cache", None) or {}).get("stress"))
    except Exception as e:  # the run is done: a refine problem must not break it
        out.log("warning: refine failed: " + " ".join(f"{type(e).__name__}: {e}".split()))
        return
    out.log(f"refine: {path} ({time.time() - t0:.1f} s)")
    if server is not None:
        try:
            webbrowser.open(server.url + "refine")
        except Exception:  # no browser: the page is still on disk
            pass


def cmd_refine(args) -> int:
    from .refine import build_refine
    t0 = time.time()
    try:
        path = build_refine(args.run_dir, yaml_path=args.yaml, say=lambda line: print(line, file=sys.stderr))
    except UserError as e:
        yaml_path = args.yaml
        if not yaml_path:
            try:
                import json
                from .runinfo import run_yaml_path
                yaml_path = run_yaml_path(json.loads((Path(args.run_dir) / "run.json").read_text()), args.run_dir)
            except (OSError, ValueError, AttributeError):
                pass
        if yaml_path:
            raise with_file(e, yaml_path)
        raise
    print(path)
    print(f"refine page ready in {time.time() - t0:.1f} s", file=sys.stderr)
    return 0


def _buckling_number(name, text, integer=False):
    """A number of the `topopt buckling` command: a UserError for a bad value."""
    try:
        v = int(text) if integer else float(text)
    except (TypeError, ValueError):
        raise UserError(f"{name} must be {'an integer' if integer else 'a number'}, got {text!r}")
    if not math.isfinite(v) or v <= 0:
        raise UserError(f"{name} must be above 0, got {text!r}")
    return v


def cmd_buckling(args) -> int:
    """`topopt buckling <run_dir>`: repeat the linear buckling check of a finished run (CONTRACTS.md section 33).

    It rebuilds the problem from the YAML named in run.json, reads the last snapshot, and writes the key
    `buckling` of `result/metrics.json` like `topopt run --buckling` does. Nothing else of the run changes,
    except a status that a kill left at `postprocess` with the message `buckling check`: that becomes `done`.
    Exit code 0: the check ran (`status: done`). 3: it wrote a fact and no load factors (skipped or error).
    """
    from .snapshot import RunWriter, _atomic_write
    run_dir = Path(args.run_dir)
    if not (run_dir / "run.json").is_file():
        raise UserError(f"{run_dir} is not a run folder (no run.json)")
    if not (run_dir / "result" / "metrics.json").is_file():
        from .runjob import stopped_hint
        raise UserError(f"the run in {run_dir} has no result/metrics.json: it is not finished. "
                        "Run `topopt run` to the end first." + stopped_hint(run_dir))
    status = {}
    try:
        status = json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError):
        pass
    state, message = status.get("state"), status.get("message", "")
    repair = state == "postprocess" and message == "buckling check"
    if state in ("starting", "running") or (state == "postprocess" and not repair):
        raise UserError(f"the run in {run_dir} is not finished (state {state!r}). Wait until it is done.")
    overrides = {}
    if args.modes is not None:
        overrides["modes"] = int(_buckling_number("--modes", args.modes, integer=True))
    if args.density is not None:
        overrides["density"] = args.density
    if args.max_seconds is not None:
        overrides["max_seconds"] = _buckling_number("--max-seconds", args.max_seconds)
    if args.max_gb is not None:
        overrides["max_gb"] = _buckling_number("--max-gb", args.max_gb)
    from .buckling_voxel import run_check
    from .refine.build import problem_from_run
    run = json.loads((run_dir / "run.json").read_text())
    try:
        problem = problem_from_run(run_dir, run, args.yaml)
    except UserError as e:
        from .runinfo import run_yaml_path
        yaml_path = args.yaml or run_yaml_path(run, run_dir)
        if yaml_path:
            raise with_file(e, yaml_path)
        raise
    from .verify.report import yaml_note_lines
    from .verify.run import compare_yaml_problem
    compared = compare_yaml_problem(run, problem)    # the YAML against run.json: facts, as for `topopt verify`
    out = RunWriter(run_dir)     # its log() appends to run.log and prints
    out.log("topopt buckling: repeat the check" + (f" ({', '.join(f'{k} {v}' for k, v in overrides.items())})"
                                                   if overrides else ""))
    for line in yaml_note_lines(compared):
        out.log(line)
    result = run_check(run_dir, problem, log=out.log, yaml_vs_run=compared, **overrides)
    if repair:
        _atomic_write(run_dir / "status.json", json.dumps({**status, "state": "done", "step": "done",
                                                           "message": ""}).encode())
        out.log("topopt buckling: the status was postprocess (buckling check): it is done now")
    return 0 if result["status"] == "done" else 3


def cmd_render(args) -> int:
    """Render STL files or a run folder into PNG views and a montage (CONTRACTS.md section 15)."""
    from .render_input import check_inputs      # light: a wrong input stops before numba and Pillow load
    sources = args.inputs
    opts = dict(roles=args.role or None, views=args.views, section=args.section, size=args.size,
                regions=args.regions, max_triangles=args.max_triangles, loads=args.loads)
    check_inputs(sources, **opts)
    from .render import render
    facts = render(sources, out_dir=args.out, **opts)
    print(json.dumps(facts, indent=2))
    return 0


def _parse_set(items) -> dict:
    """`--set KEY=VALUE` (repeatable) as a dict for the `print` block. VALUE is read as YAML: 1.2, [0, 0, 1], fdm."""
    import yaml
    out = {}
    for item in items or []:
        key, sep, text = item.partition("=")
        if not sep or not key.strip():
            raise UserError(f"--set needs KEY=VALUE, got {item!r}")
        try:
            out[key.strip()] = yaml.safe_load(text)
        except yaml.YAMLError:
            raise UserError(f"--set {key.strip()}: cannot read the value {text!r}")
    return out


def cmd_printcheck(args) -> int:
    """Print check of an STL file or a run folder (CONTRACTS.md section 32). Geometry only: no solve.

    It reports. The exit code is 0 for any check result, and 1 for an input that cannot be read. With `--strict`
    a failed check gives the exit code `printcheck.GATE_EXIT_CODE`. A check that stops with an exception is a record with the state
    `unknown` and the key `error`, and the other checks go on (`partial` and `coverage`, CONTRACTS.md section 44). The command prints
    the whole report and then exits with 2, the code of an internal error, as it did when such a check stopped the command.
    """
    from . import printcheck
    report = printcheck.check(args.path, yaml_path=args.yaml, print_settings=_parse_set(args.set),
                              resolution_mm=args.resolution_mm, strict=args.strict, contain_errors=True)
    print(json.dumps(report, indent=1) if args.json else printcheck.format_text(report))
    stopped = [c["id"] for c in report["checks"] if c.get("error")]
    if stopped:      # TC-2: the other checks are in the report; the exit code of an internal error stays (an `unknown` never sets the gate)
        print(f"internal error: {len(stopped)} check(s) stopped: {', '.join(stopped)}; see `error` in their records", file=sys.stderr)
        return 2
    return report["gate"]["exit_code"]


def cmd_report(args) -> int:
    """`topopt report <run_dir>`: write the one-page part card `result/report.md` again with the facts that the run folder has now
    (CONTRACTS.md section 9.2). Reads files only: no pass, no solve. Prints the path of the card.

    Exit code 0: the card is written (or it was the same). 1: the folder is not a run folder with a result, or the file cannot be written.
    """
    from . import part_card
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise UserError(f"{args.run_dir} is not a run folder (no such folder). Give the folder that `topopt run --out` wrote")
    try:
        path = part_card.write(run_dir)
    except part_card.NoResult as e:
        raise UserError(f"{e}. Run `topopt run` first, or wait until the run is done") from None
    except OSError as e:
        raise UserError(f"the part card cannot be written: {e}") from None
    print(path)
    return 0


def cmd_levers(args) -> int:
    """`topopt levers [QUERY ...]`: print the lever cards (CONTRACTS.md section 41). The text is the one of the MCP resource `topopt://levers`.

    No query: all cards. Query words: the cards that match a key path, a tool name or a symptom word. Exit code 1 when no card matches.
    `--run DIR`: the facts of that run folder that the engine flags, each with its cards (section 41.10); no query. Exit code 1 for a folder that is not there.
    """
    from . import levers
    query = [q for q in args.query if q.strip()]
    if args.run:
        if query:
            raise UserError("`topopt levers --run` takes no key or word: it prints the cards of the facts of that run. "
                            "Use `topopt levers <key or word>` for the cards that match a word")
        if not Path(args.run).is_dir():
            raise UserError(f"{args.run} is not a run folder (no such folder). Give the folder that `topopt run --out` wrote")
        from . import levers_run
        print(levers_run.format_text(levers_run.run_levers(args.run)), end="")
        return 0
    if query and not levers.find(query):
        raise UserError(f"no lever card matches {', '.join(repr(q) for q in query)}. Options: `topopt levers` prints all cards; "
                        "give a key path such as settings.penal, a tool name such as verify_run, or a word that a user says, such as film")
    print(levers.reference(query or None), end="")
    return 0


def cmd_step(args) -> int:
    """Write a faceted STEP file of a finished run (CONTRACTS.md section 42). Post-only: no pass runs, no input file changes.

    Exit code 0: the file is written and every check of the read-back passed. 3: the file is written and a check failed (a fact; the
    text says which). 1: bad input, or the extra `step` (OpenCASCADE) is not installed (the text has the install command).
    """
    from . import step

    def say(msg):
        print(msg, file=sys.stderr, flush=True)

    report = step.export_step(args.run_dir, part_only=args.part_only, out=args.out, tolerance_mm=args.tolerance_mm,
                              merge_planes=not args.no_merge_planes, schema=args.schema, estimate_only=args.estimate_only,
                              log=None if args.json else say)
    print(json.dumps(report, indent=1) if args.json else step.format_text(report))
    return step.exit_code(report)


def cmd_doctor(args) -> int:
    """Facts about this install and the commands that fix it (CONTRACTS.md section 8). Read-only; exit code 1 on a problem."""
    from . import doctor     # inside the command: `import topopt.cli` loads only PyYAML, and doctor must start in a broken venv
    return doctor.run(args.yaml, as_json=args.json, import_timeout_s=args.timeout_s)


def cmd_gpu_check(args) -> int:
    """Check gpu-mg on this machine and write one report (CONTRACTS.md sections 8 and 20)."""
    from . import gpu_check
    return gpu_check.run(out=args.out, device=args.device, skip_tests=args.skip_tests, skip_bench=args.skip_bench,
                         step_timeout_s=args.step_timeout_s)


def cmd_dashboard(args) -> int:
    """Serve a run folder until Ctrl+C. A busy port is an error that the user can fix (exit code 1), not an internal error."""
    import errno
    from . import dashboard_server
    try:
        dashboard_server.serve(args.run_dir, args.port, open_browser=not args.no_open,
                               replay=args.replay, speed=args.speed,
                               badge=not args.no_badge, listen_fd=args.listen_fd)
    except OSError as e:
        if args.listen_fd is not None:
            raise UserError(f"--listen-fd {args.listen_fd}: {e}")
        if e.errno != errno.EADDRINUSE:
            raise
        raise UserError(f"dashboard port {args.port} is busy: another dashboard or program uses it. Options: open that "
                        f"dashboard if it shows this run folder (it follows a new run), or give a free port with "
                        f"--port N") from None
    return 0


def _verify_number(name: str, text):
    """A float option of `topopt verify`, read as text so that a bad value is a UserError (exit code 1)."""
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        raise UserError(f"{name} must be a number, got {text!r}") from None


def cmd_verify(args) -> int:
    """Check the final STL of a run with quadratic tets (CONTRACTS.md section 16). Exit codes: 0, 3, 1, 2."""
    from .verify.errors import VerifyError
    from .verify.report import estimate_lines, plain_lines, summary_lines
    from .verify.run import DEFAULT_NEAR_MM, parse_levels, verify_run

    def say(msg):
        print(msg, file=sys.stderr, flush=True)

    try:
        res = verify_run(
            args.run_dir, mesh_size=_verify_number("--mesh-size", args.mesh_size),
            levels=parse_levels(args.levels) if args.levels else None,
            region_band_mm=_verify_number("--region-band-mm", args.region_band_mm),
            allowable_mpa=_verify_number("--allowable-mpa", args.allowable_mpa),
            percentile=_verify_number("--percentile", args.percentile),
            solver=args.solver, max_gb=_verify_number("--max-gb", args.max_gb), force=args.force,
            estimate_only=args.estimate_only, yaml_path=args.yaml,
            near_region_mm=(_verify_number("--near-mm", args.near_mm) if args.near_mm is not None else DEFAULT_NEAR_MM),
            load_sensitivity=not args.no_sensitivity, buckling=args.buckling, bore_checks=args.bore_checks,
            part_only=args.part_only, gate=args.gate, log=say)
    except VerifyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for line in (estimate_lines(res) if res.get("estimate_only") else summary_lines(res)):
        print(line)
    if not res.get("estimate_only"):        # an estimate has no stress: nothing to say in plain words
        for line in plain_lines(res):
            print("plain: " + line)
    return int(res["exit_code"])


def _usage_error(parser, msg):
    """Print a usage error of `topopt verify` and exit with code 1 (the contract: 1 is bad input or usage)."""
    parser.print_usage(sys.stderr)
    print(f"{parser.prog}: error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _role_arg(text):
    """The type of `--role`: a role or a comma list of roles, as a list (the flag extends one list)."""
    from .render_input import role_list
    try:
        return role_list(text)
    except UserError as e:
        raise argparse.ArgumentTypeError(str(e))


def build_parser():
    ap = argparse.ArgumentParser(prog="topopt", description="3D topology optimization.")
    ap.add_argument("--version", action="version", version=f"topopt {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="optimise a problem file")
    r.add_argument("yaml")
    r.add_argument("--out", required=True, help="run folder")
    r.add_argument("--dashboard", action="store_true", help="start the dashboard and open the browser")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--quiet", action="store_true", help="write run.log but print nothing")
    r.add_argument("--mesh-method", choices=["sdf", "blur"], default=None,
                   help="final mesh: sdf (volume-preserving SDF smoothing, default) or blur (the old Gaussian path)")
    r.add_argument("--envelope-fill", choices=list(ENVELOPE_FILL_VALUES), default=None,
                   help="sdf mesh: grow the result to the design space surface: auto (not when the growth is a large share of the mass), on (also above the limit) or off; "
                        f"the default is {ENVELOPE_FILL_DEFAULT}; wins over settings.envelope_fill")
    r.add_argument("--solver", choices=SOLVER_SETTINGS + EXPERIMENTAL_SOLVERS, default=None,
                   help="use this value for settings.solver in this run (auto: gpu-mg on a GPU that passes the self-test, else the CPU solver)")
    r.add_argument("--no-wait", action="store_true",
                   help="with --dashboard: exit when done (default: keep the dashboard on)")
    r.add_argument("--buckling", action=argparse.BooleanOptionalAction, default=None,
                   help="linear buckling check of the final design after the post-process (default: the YAML "
                        "block buckling, which is off)")
    r.set_defaults(func=cmd_run)
    r.add_argument("--override-lock", action="store_true",
                   help="run even if a live process holds the job lock of the run folder")
    r.add_argument("--resume", action="store_true",
                   help="continue the run of --out from its last checkpoint (the YAML and the region files are read again)")
    r.add_argument("--allow-changed-inputs", action="store_true",
                   help="with --resume or --continue-from: continue also if the YAML, a region file or a setting changed since the checkpoint")
    r.add_argument("--continue-from", metavar="PARENT", default=None,
                   help="add passes to the run of the folder PARENT in a new run folder --out (needs --more-iterations). The run starts from the "
                        "last checkpoint of PARENT, which is only read and does not change. settings.iterations of the YAML is not used")
    r.add_argument("--more-iterations", metavar="N", default=None,
                   help="with --continue-from: how many passes to add, a whole number of 1 or more with no upper limit. The new run has "
                        "K + N passes, where K is the pass of the checkpoint of PARENT")
    r.add_argument("--stop-rules", action=argparse.BooleanOptionalAction, default=None,
                   help="turn the stop rules on or off (default: settings.stop_rules of the YAML, off). A fine pass has the rule 'settled' on by default; "
                        "--no-stop-rules turns every stop rule off, and the fine pass runs to its maximum")
    r.add_argument("--fine-from", metavar="PARENT", default=None,
                   help="a fine pass: a new run in --out on a finer voxel, from the design of the finished run folder PARENT, which is only read. "
                        "The voxel is the one that the part needs (see `topopt check`), or --fine-voxel. The run ends when its loss has settled, "
                        "or after --fine-passes passes. settings.voxel_size and settings.iterations of the YAML are not used")
    r.add_argument("--fine-pass", action="store_true",
                   help="the normal run in --out, then the fine pass from it in the sibling folder <--out>-fine (when the part needs a finer "
                        "voxel than the run has, or --fine-voxel is finer than the run; else the run ends after the first run, with exit code 0)")
    r.add_argument("--fine-voxel", metavar="H", default=None,
                   help="with --fine-from or --fine-pass: the voxel of the fine pass in mm (default: the voxel that the part needs). A voxel equal to "
                        "the one of the parent is refused; a coarser voxel is allowed, with a fact")
    r.add_argument("--fine-passes", metavar="N", default=None,
                   help="with --fine-from or --fine-pass: the MAXIMUM of passes of the fine pass, a whole number of 1 or more (default: the pass "
                        "count of the parent run). The fine pass ends earlier when its loss has settled")
    r.add_argument("--fine-tol", metavar="T", default=None,
                   help="with --fine-from or --fine-pass: the tolerance at which the loss has settled, a number above 0: the relative change of "
                        "the objective between two passes over the window of settings.stop_rules (default 0.001). A smaller number runs more passes")
    sp = sub.add_parser("stop", help="ask the run in a run folder to stop at the end of its current pass (it saves a checkpoint; --resume continues)")
    sp.add_argument("run_dir", help="the run folder (the --out of `topopt run`)")
    sp.set_defaults(func=cmd_stop)
    d = sub.add_parser("dashboard", help="show a run folder in the browser")
    d.add_argument("run_dir")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--replay", action="store_true")
    d.add_argument("--speed", type=float, default=1.0, help="replay clock speed")
    d.add_argument("--no-badge", action="store_true", help="replay: hide the REPLAY label")
    d.add_argument("--no-open", action="store_true", help="do not open the browser")
    d.add_argument("--listen-fd", type=int, default=None, help=argparse.SUPPRESS)   # `topopt run --dashboard` hands its socket over
    d.set_defaults(func=cmd_dashboard)
    f = sub.add_parser("refine", help="make result/refine.html (final part and stress) for a finished run")
    f.add_argument("run_dir")
    f.add_argument("--yaml", metavar="FILE", help="the problem YAML, if the path in run.json is not valid")
    f.set_defaults(func=cmd_refine)
    b = sub.add_parser("buckling", help="repeat the linear buckling check of a finished run (writes metrics.json)")
    b.add_argument("run_dir")
    b.add_argument("--modes", metavar="N", help="how many of the lowest load factors for each case (default: the "
                   "YAML block buckling, else 6)")
    b.add_argument("--density", choices=["printed", "gray"], help="printed (the design at the print threshold 0.5) "
                   "or gray (the optimizer density)")
    b.add_argument("--max-seconds", metavar="S", help="time budget of the eigen solve of one case, in seconds")
    b.add_argument("--max-gb", metavar="G", help="memory budget of the check in GB (1e9 bytes; default: the RAM)")
    b.add_argument("--yaml", metavar="FILE", help="the problem YAML, if the path in run.json is not valid")
    b.set_defaults(func=cmd_buckling)
    c = sub.add_parser("check", help="load and voxelize, no optimizer run; show the solver and the time estimate")
    c.add_argument("yaml")
    c.add_argument("--json", action="store_true", help="print one JSON object, with the solver recommendation")
    c.add_argument("--solver", choices=SOLVER_SETTINGS + EXPERIMENTAL_SOLVERS, default=None,
                   help="show the facts for this value of settings.solver")
    c.add_argument("--fine-from", metavar="PARENT", default=None,
                   help="show the plan of a fine pass from the finished run folder PARENT (voxel, counts, estimates, refusals); nothing is written")
    c.add_argument("--fine-voxel", metavar="H", default=None, help="with --fine-from: the voxel of the fine pass in mm")
    c.add_argument("--fine-passes", metavar="N", default=None, help="with --fine-from: the maximum of passes of the fine pass")
    c.add_argument("--fine-tol", metavar="T", default=None, help="with --fine-from: the tolerance at which the loss has settled")
    c.add_argument("--allow-changed-inputs", action="store_true", help="with --fine-from: show the plan also if an input changed since the parent run")
    c.set_defaults(func=cmd_check)
    n = sub.add_parser("render", help="render STL files or a run folder to PNG views and a montage (no GL)")
    n.add_argument("inputs", nargs="+", help="one run folder, one problem YAML (the regions and the interface shapes, before a run), or STL files")
    n.add_argument("--role", action="extend", type=_role_arg, metavar="ROLE[,ROLE...]",
                   help="role of an STL: part, design, fixed, load or keep-out (default: part). Give one role for all files, or one for each "
                        "file, as a comma list (--role design,fixed,load) or as repeated flags (--role design --role fixed)")
    n.add_argument("--views", default=None, help="comma list of: iso front back right left top bottom "
                   "(default iso,front,right,top; with --section: cut,iso-cut and two side views); "
                   "NAME:dx,dy,dz for your own (alone or in the list: fl:-0.5,-1,0.25 or iso,fl:-0.5,-1,0.25,top; "
                   "';' also separates)")
    n.add_argument("--section", default=None, metavar="AXIS=VALUE",
                   help="cut at the plane, in mm: x=30 (or x>=30) keeps x >= 30; 'x<=30' keeps x <= 30")
    n.add_argument("--regions", default=None, choices=["ghost", "solid", "off"],
                   help="how the non-part meshes are drawn (default: ghost with a part, else solid)")
    n.add_argument("--max-triangles", type=int, default=None, metavar="N",
                   help="stop when the meshes have more triangles than N (default: no limit)")
    n.add_argument("--loads", action=argparse.BooleanOptionalAction, default=None,
                   help="a problem YAML: draw one force arrow for each force of each load case (default); --no-loads draws none")
    n.add_argument("--size", default=None, metavar="WxH", help="one tile in pixels (default 640x480)")
    n.add_argument("--out", default=None, help="output folder (default: <run>/renders, or renders beside the first STL)")
    n.set_defaults(func=cmd_render)

    v = sub.add_parser("verify", help="check the final STL of a run with quadratic tets (P2 stress)")
    v.add_argument("run_dir")
    v.add_argument("--mesh-size", metavar="MM", help="mesh size in mm (default: the size of the tet mesher, 2.5 times the mean edge of the STL, about 1.3 to 2.2 voxels for an sdf STL; the report prints it)")
    v.add_argument("--levels", metavar="MM,MM,...", help="several mesh sizes; the finest is the reference level")
    v.add_argument("--region-band-mm", metavar="MM", help="band of the region rule (default: 0.5 voxel)")
    v.add_argument("--allowable-mpa", metavar="X", help="your allowable stress: exit code 3 if it is exceeded at the percentile. With --gate it "
                   "replaces the stress limit of every load case in the YAML; the gate record shows the replaced limit, and a value above it "
                   "loosens the requirement (only with the consent of the user)")
    v.add_argument("--percentile", metavar="P", default="99.9", help="percentile by volume for the limit (default 99.9)")
    v.add_argument("--solver", metavar="NAME", help="auto (default), cholesky, pardiso, cholespy or superlu")
    v.add_argument("--max-gb", metavar="X", help="refuse to start above this memory estimate in GB")
    v.add_argument("--force", action="store_true", help="run although the memory estimate is above the limit")
    v.add_argument("--estimate-only", action="store_true", help="print the memory estimates and stop, no meshing")
    v.add_argument("--yaml", metavar="FILE", help="the problem YAML, if the path in run.json is not valid")
    v.add_argument("--near-mm", metavar="MM", help="distance from a fixed or load region for the stress numbers "
                   "without the region patches (default 2)")
    v.add_argument("--no-sensitivity", action="store_true",
                   help="do not solve the load at a band of 0 and of 2 times the band (saves time)")
    v.add_argument("--buckling", action=argparse.BooleanOptionalAction, default=None,
                   help="linear buckling load factors of the tet model, from the factor of the static solve "
                        "(default: buckling.check in the YAML of the run, which is off)")
    v.add_argument("--bore-checks", action=argparse.BooleanOptionalAction, default=None,
                   help="the Goodman and press-fit checks at the bores of the block bore_checks of the YAML "
                        "(default: on when the YAML has the block; --no-bore-checks turns them off)")
    v.add_argument("--gate", action="store_true",
                   help="a hard gate (opt-in): the key gate of verify.json has met for the stress of each case (stress_limit or allowable_mpa of the "
                        "case at --percentile; --allowable-mpa replaces it for all cases), the lowest buckling load factor (buckling.min_factor; this "
                        "turns the buckling check on) and the rows of limits.displacement and limits.alignment. A check that could not run fails the "
                        "gate. Exit code 3 if the gate fails. The limits are read as limits of the tet value, not of the voxel value (topopt levers verify_run:gate)")
    v.add_argument("--part-only", action="store_true",
                   help="mesh result/part_only.stl (the part without the bodies of regions.not_printed) instead of nominal.stl or final.stl: a removed "
                        "fixed body is clamped at its contact face, a removed load body takes its force there, a removed solid or "
                        "keep-out body gives no boundary condition (needs a run with regions.not_printed)")
    v.error = lambda msg: _usage_error(v, msg)   # a usage error is exit code 1 (argparse gives 2)
    v.set_defaults(func=cmd_verify)

    p = sub.add_parser("printcheck", help="check an STL or a run folder for printing: facts, limits and a state for "
                       "each check (no solve)")
    p.add_argument("path", help="an STL file, or a run folder (it has result/final.stl)")
    p.add_argument("--yaml", default=None, help="a problem YAML: its print block gives the settings and limits "
                   "(a run folder already has the print block of its run)")
    p.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                   help="a key of the print block, for example min_wall_mm=1.2 or bed_mm=[250,210]; repeat it; wins over the YAML")
    p.add_argument("--resolution-mm", type=float, default=None, metavar="MM",
                   help="the size of the smallest member that the mesh holds: the voxel size of the run, or 0 for exact CAD "
                   "(default: the voxel size in run.json, else the median mesh edge)")
    p.add_argument("--strict", action="store_true",
                   help="opt in to the gate: a failed check gives the exit code 4 (unknown never does)")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(func=cmd_printcheck)

    lv = sub.add_parser("levers", help="print the lever cards: what each YAML key and engine tool does, what it costs and which facts point to it "
                                       "(with a key path or a word, such as film, only the matching cards)")
    lv.add_argument("query", nargs="*", help="a key path (settings.penal), a tool name (verify_run) or a word that a user says; "
                                             "a word matches the symptoms too")
    lv.add_argument("--run", metavar="RUN_DIR", default=None,
                    help="a run folder: print the facts of that run that the engine flags (a verdict test not met, a warning, ...), each with the cards "
                         "that name it, cheapest layer first. Reads files only")
    lv.set_defaults(func=cmd_levers)
    rp = sub.add_parser("report", help="write the one-page part card result/report.md of a finished run again, with the facts the run folder "
                        "has now (verify, buckling, renders). Reads files only")
    rp.add_argument("run_dir", help="the run folder (it has result/metrics.json)")
    rp.set_defaults(func=cmd_report)
    st = sub.add_parser("step", help="write a faceted STEP file of a finished run (extra step: OpenCASCADE). No pass runs; the run folder keeps "
                        "all its files")
    st.add_argument("run_dir", help="the run folder (it has result/final.stl)")
    st.add_argument("--part-only", action="store_true", help="wrap result/part_only.stl (the part without the bodies of regions.not_printed) "
                    "instead of result/final.stl; the file is result/part_only.step")
    st.add_argument("--out", metavar="PATH", help="the STEP file, or an existing folder for it (default: result/final.step in the run folder). "
                    "An existing file is replaced only when it is a .step or .stp file. The report goes next to it, as <file>.json")
    st.add_argument("--tolerance-mm", metavar="X", help="simplify the mesh with manifold3d before the faces are made: the largest move of the "
                    "surface that a merge of triangles may make, in mm; 0 = no simplification (default: the voxel size of the run / 20; 0 when the run folder holds no voxel size)")
    st.add_argument("--no-merge-planes", action="store_true", help="keep one face for each triangle; by default coplanar faces join into one planar face")
    st.add_argument("--schema", default="AP214", help="the STEP schema, AP214 or AP242 (default AP214)")
    st.add_argument("--estimate-only", action="store_true", help="simplify the mesh, print the number of faces and the estimate of file size, time "
                    "and memory, and stop; nothing is written")
    st.add_argument("--json", action="store_true", help="print the report as JSON (the same as the report file next to the STEP file)")
    st.error = lambda msg: _usage_error(st, msg)   # a usage error is exit code 1 (argparse gives 2)
    st.set_defaults(func=cmd_step)

    dr = sub.add_parser("doctor", help="what this install is, what is wrong with it, and the commands that fix it "
                        "(read-only: installs, builds and fetches nothing)")
    dr.add_argument("yaml", nargs="?", default=None, help="optional problem YAML: adds the time estimate for it")
    dr.add_argument("--json", action="store_true", help="print one JSON object")
    dr.add_argument("--timeout-s", type=float, default=90.0, metavar="S",
                    help="seconds for the import test of one package (default 90)")
    dr.set_defaults(func=cmd_doctor)

    gc = sub.add_parser("gpu-check", help="check the GPU solver gpu-mg on this machine: parity, real-device tests, "
                        "a short bench, one report (needs a checkout: tools/ is not in a wheel)")
    gc.add_argument("--out", default=None, metavar="DIR", help="report folder (default: runs/gpu-check)")
    gc.add_argument("--device", default=None, metavar="D",
                    help="an OpenCL device: its number in the list that `topopt check` prints, or a part of its name "
                         "(default: the first device that is not a CPU; give a CPU device for a smoke test)")
    gc.add_argument("--skip-tests", action="store_true", help="do not run the pytest step")
    gc.add_argument("--skip-bench", action="store_true", help="do not run the short bench (it takes minutes)")
    gc.add_argument("--step-timeout-s", type=float, default=None, metavar="S",
                    help="stop a step that runs longer than S seconds (default: no limit)")
    gc.set_defaults(func=cmd_gpu_check)
    return ap


def main(argv=None) -> int:
    cap_blas_threads()   # first: before any import of numpy
    args = build_parser().parse_args(argv)
    args.own_process = argv is None   # the arguments are the command line of this process (not a call from a program)
    try:
        return args.func(args)
    except UserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # internal error
        traceback.print_exc()
        print(f"internal error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
