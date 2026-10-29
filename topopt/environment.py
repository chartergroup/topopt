"""Facts about the machine and the solver backends (CONTRACTS.md section 10, tool `environment`).

`environment_data(yaml_path=None)` returns a dict of facts, estimates and options. It reads
the machine and starts no optimizer run (with `auto` on a GPU it runs the short self-test of gpu-mg in a child process, at most 60 s: seconds the first time on a device and driver, then the cached pass). It installs and builds nothing: when a faster backend is
possible, the dict holds the offer as data (`recommendations`) and the user decides.
On macOS it reads the free RAM with the read-only command `vm_stat`. With an x86_64 Python it also runs
`sysctl -n sysctl.proc_translated`, to tell Rosetta from an Intel Mac.

`python -m topopt.environment [problem.yaml]` prints the dict as JSON. The MCP tool uses it to ask
the Python that runs the optimizer (TOPOPT_PYTHON), which can differ from the Python of the server.

`checkout` holds the facts about the code folder (git or not, revision, dirty, ahead or behind origin as of the
last fetch). They come from `topopt.doctor.checkout_facts`, the function that `topopt doctor` uses. `gpu_solver` says if
`settings.solver: auto` uses the GPU solver gpu-mg here, and if not, why and what would make it usable (facts, no offer). The slow checks
(the import test of every package in a child Python) are in `topopt.doctor` only.
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from . import __version__
from .cli import THREAD_VARS as CAP_VARS
from .cli import blas_thread_cap, cgroup_limits, host_cores, host_ram_bytes, usable_cores
from .config import UserError, load_config
from .doctor import checkout_facts
from .uv_command import uv_command
from .solver_choice import (AMG_TIME_FACTOR, EXPERIMENTAL_SOLVERS, SOLVER_SETTINGS, accelerate_problem, is_linux_x86_64,
                            is_macos_arm64, platform_name, recommendation, resolve_solver, run_solver,
                            shim_build_command)

PACKAGES = ("numpy", "scipy", "pyamg", "pypardiso", "mkl", "scikit-image", "trimesh",
            "manifold3d", "mcp", "pyopencl")
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _package_versions() -> dict:
    """`{name: version or None}`. It reads the metadata and imports nothing."""
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


PYAMG_PIN = "pyamg==5.3.0"


def parse_vm_stat(txt: str):
    """Free RAM in bytes from the text of macOS `vm_stat` (free + inactive + speculative pages)."""
    size = int(txt.split("page size of")[1].split()[0])
    pages = {}
    for line in txt.splitlines()[1:]:
        key, _, val = line.partition(":")
        val = val.strip().rstrip(".")
        if val.isdigit():
            pages[key.strip()] = int(val)
    return (pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
            + pages.get("Pages speculative", 0)) * size


def _available_ram_bytes():
    """RAM that is free now on the host (Linux: MemAvailable; macOS: `vm_stat`), or None."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "darwin":
        try:
            txt = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            return parse_vm_stat(txt)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None
    return None


def _ram() -> dict:
    """RAM facts: the host values, the cgroup (container) limit, and the values for this process.

    In a cgroup the room is the limit minus the working set (`cgroup_limits`: memory.current minus the inactive file
    pages; this is the rule of cAdvisor and the kubelet, and Docker subtracts the same value). The raw count
    (`cgroup_current_bytes`) and the inactive file pages (`cgroup_inactive_file_bytes`) are facts too. `cgroup_facts`
    says what is missing when memory.stat has no inactive file line (the room is then limit minus the raw count),
    and when the inactive file pages are more than the raw count.
    """
    cg = cgroup_limits()
    host_total, host_avail = host_ram_bytes(), _available_ram_bytes()
    total, avail, room = host_total, host_avail, None
    if cg["memory_bytes"]:
        total = min(host_total, cg["memory_bytes"]) if host_total else cg["memory_bytes"]
        if cg["memory_used_bytes"] is not None:
            room = max(0, cg["memory_bytes"] - cg["memory_used_bytes"])
            avail = min(host_avail, room) if host_avail is not None else room
    return {"total_bytes": total, "available_bytes": avail,
            "host_total_bytes": host_total, "host_available_bytes": host_avail,
            "cgroup_limit_bytes": cg["memory_bytes"], "cgroup_source": cg["source"],
            "cgroup_used_bytes": cg["memory_used_bytes"], "cgroup_current_bytes": cg["memory_current_bytes"],
            "cgroup_inactive_file_bytes": cg["inactive_file_bytes"],
            "cgroup_room_bytes": room, "cgroup_facts": list(cg["facts"]),
            "note": "total_bytes and available_bytes are for this process: the host value, "
                    "or the cgroup (container) limit if it is lower. In a cgroup, available_bytes is the smaller of the "
                    "host value and the room: the limit minus the working set (cgroup_used_bytes = cgroup_current_bytes "
                    "minus the inactive file pages). The kernel takes the inactive page cache back when a process "
                    "needs memory, so it is not counted as used."}


def _cores() -> dict:
    cg = cgroup_limits()
    return {"logical": os.cpu_count(), "affinity": host_cores(),
            "cgroup_cpu_limit": cg["cpu_cores"], "usable": usable_cores()}


def _threads() -> dict:
    env = {k: os.environ[k] for k in THREAD_VARS if k in os.environ}
    out = {"env": env}
    if sys.platform == "linux":
        # The same rule as cli.cap_blas_threads: OpenBLAS reads the first of the variables in
        # cli.THREAD_VARS that is set. If one is set, the cap does nothing. A value that is not a
        # number stays as it is.
        value = next((os.environ[k] for k in CAP_VARS if k in os.environ), None)
        out["openblas_threads_in_run"] = value if value is not None else blas_thread_cap()
        out["note"] = ("topopt run limits OpenBLAS to min(4, usable cores) threads on Linux when "
                       "none of OPENBLAS_NUM_THREADS, GOTO_NUM_THREADS and OMP_NUM_THREADS is set. "
                       "Set one of them to use another number.")
    else:
        out["openblas_threads_in_run"] = None
        out["note"] = "topopt does not limit threads on this platform."
    return out


def _mac_facts() -> dict:
    """The facts about an x86_64 Python on a Mac: `mac_python` (None, "arm64", "rosetta", "intel", "unknown") and `hint`.

    `hint` is the text of `solver_choice.platform_hint()` (None on every other platform). On a Mac with an
    x86_64 Python it runs the read-only command `sysctl -n sysctl.proc_translated`.
    """
    from .solver_choice import mac_python_kind, platform_hint
    return {"mac_python": mac_python_kind(), "hint": platform_hint()}


def _gpu_mg_entry(opencl: dict) -> dict:
    """The entry of `gpu-mg` in `solver_setting`, from the OpenCL facts of the probe (a child process): this process asks no OpenCL driver.

    `resolved` is "gpu-mg" when the facts hold a device list, as `resolve_solver("gpu-mg")` does. Else `error` is the `problem` text of the facts.
    With `TOPOPT_AUTO_GPU=0` the device list was not asked: `error` says that, and that an explicit `settings.solver: gpu-mg` asks the drivers in the
    run (the variable does not change it), not that gpu-mg cannot run.
    """
    if opencl.get("devices"):
        return {"resolved": "gpu-mg", "error": None}
    from .gpu_mg.device import no_device_error, solver_options_text
    problem = opencl.get("problem") or ""
    install = opencl.get("install_command")
    if problem.startswith("missing"):       # the texts of `device.import_pyopencl`
        return {"resolved": None, "error": f"solver 'gpu-mg' needs pyopencl, and it is not installed. Install the gpu extra: {install}. Or {solver_options_text()}"}
    if problem.startswith("load_failed"):
        return {"resolved": None, "error": (f"solver 'gpu-mg' needs pyopencl, and it is installed but it does not load: {problem.split(': ', 1)[-1]}. "
                                            f"Check the OpenCL library of this machine, or install the gpu extra again: {install}. Or {solver_options_text()}")}
    if problem.startswith("not asked"):
        return {"resolved": None, "error": (f"solver 'gpu-mg' is not checked here: the OpenCL device list was {problem}. An explicit "
                                            "`settings.solver: gpu-mg` still asks the OpenCL drivers when the run starts")}
    if problem and problem != "no OpenCL device":       # a child that did not finish
        return {"resolved": None, "error": f"solver 'gpu-mg' cannot run here: {problem}"}
    return {"resolved": None, "error": str(no_device_error())}


def _resolution(opencl: dict | None = None) -> dict:
    """What each value of `settings.solver` means on this platform: a solver name or the error text.

    `auto` is what a run uses first (`solver_choice.run_solver`: gpu-mg on a GPU that passes the self-test, else the CPU rule). It also holds
    `cpu_solver` (the CPU rule, which is the fallback), `fallback`, `why` (the fact line of the choice) and the whole `choice`.
    `gpu-mg` comes from the `opencl` facts of the probe (`_opencl_facts()` when not given), not from `resolve_solver` (that lists the devices in this process).
    """
    out = {}
    for setting in SOLVER_SETTINGS:
        try:
            if setting == "auto":
                choice = run_solver("auto")
                out[setting] = {"resolved": choice.solver, "error": None, "cpu_solver": choice.cpu, "fallback": choice.fallback,
                                "why": choice.line(), "choice": choice.as_dict()}
            elif setting == "gpu-mg":
                out[setting] = _gpu_mg_entry(_opencl_facts() if opencl is None else opencl)
            else:
                out[setting] = {"resolved": resolve_solver(setting), "error": None}
        except UserError as e:
            out[setting] = {"resolved": None, "error": str(e)}
    return out


def _experimental(opencl: dict | None = None) -> dict:
    """What each experimental value of `settings.solver` means here: a solver name or the error text. No value is experimental now: the dict is empty."""
    out = {}
    for setting in EXPERIMENTAL_SOLVERS:
        try:
            if setting == "gpu-mg":
                out[setting] = _gpu_mg_entry(_opencl_facts() if opencl is None else opencl)
            else:
                out[setting] = {"resolved": resolve_solver(setting), "error": None}
        except UserError as e:
            out[setting] = {"resolved": None, "error": str(e)}
    return out


def _opencl_facts() -> dict:
    """OpenCL platform, device and 64-bit float facts for the GPU solver gpu-mg (never raises). A child process lists the devices, with a time limit."""
    from .gpu_mg import probe
    return probe.facts()


def _shim_offer(backends: dict) -> dict | None:
    """The offer to build the Accelerate shim: Apple silicon, library missing or not loading. Data only."""
    if not is_macos_arm64() or backends["cholesky"]["available"]:
        return None
    problem = accelerate_problem()
    state = ("is not built" if problem is None or problem[0] == "missing"
             else f"is built but it does not load ({problem[1]})")
    return {
        "action": "build_accelerate_shim",
        "command": shim_build_command(),
        "needs": "Xcode command line tools (xcrun clang)",
        "reason": (f"The Accelerate shim {state}. Until it works, settings.solver auto runs amg, "
                   f"which needs no shim and is about {AMG_TIME_FACTOR:g}x slower per iteration "
                   f"(an estimate from timing runs, which give 3 to 5 times; `check_problem` gives the "
                   f"time for this problem). "
                   f"The shim is a small C file that links Apple Accelerate."),
        "license_note": "Apple Accelerate is part of macOS. No extra package.",
    }


def _pyamg_offer() -> dict:
    py = shlex.quote(sys.executable)
    return {"action": "install_pyamg", "command": f"{uv_command()} pip install --python {py} {PYAMG_PIN}",
            "reason": "pyamg is the solver that runs on every platform. It is not installed.",
            "license_note": "pyamg is MIT licensed."}


# What the first real-GPU run measured (a MacBook Air M4, 2026-10-21, main cd349f9, tools/bench_gpu_mg.py with
# 20 iterations on the hook). `gpu_solver.measured` gives these as the facts of ONE machine. They are not a promise for
# another machine, and no rule of the code depends on them (no grid size threshold: the agent decides).
GPU_MG_MEASURED = {
    "machine": "MacBook Air M4, 16 GB",
    "what": "the hook example, median seconds per optimizer iteration, 20 iterations, case C",
    "rows": [   # voxel mm, free dofs, default solver s/iter (Accelerate Cholesky), gpu-mg s/iter
        {"voxel_mm": 1.2, "free_dofs": 115836, "default_s": 0.765, "gpu_mg_s": 0.287},
        {"voxel_mm": 1.0, "free_dofs": 205485, "default_s": 2.235, "gpu_mg_s": 0.439},
        {"voxel_mm": 0.8, "free_dofs": 386142, "default_s": 9.241, "gpu_mg_s": 0.658},
    ],
    "finest": {"voxel_mm": 0.6, "free_dofs": 905010, "default_memory_estimate_gib": 22.5, "ram_gib": 16,
               "gpu_mg_s": 1.919, "gpu_mg_peak_mb": 2415.5},
    "compliance_agreement": "1e-10 or better",
}


def _python_command(*args: str) -> str:
    """`<this python> -m topopt ...` as a shell string, for a command in a fact."""
    return " ".join([shlex.quote(sys.executable), "-m", "topopt", *args])


def _gpu_solver(opencl: dict, choice: dict | None) -> dict:
    """The facts about the GPU solver gpu-mg on this machine, from the pick of `auto` (`choice`, the dict of `SolverChoice.as_dict`) and the `opencl` facts.

    Data only: it replaces the two offers of the time when gpu-mg was opt-in. Keys: `used` (True when `settings.solver: auto` runs gpu-mg here),
    `device`, `case` and `self_test` (the facts of the pick, or None), `why` (the fact line of the pick: the device and the self-test, or the reason for
    the CPU solver), `to_use_the_gpu` (what would make a GPU usable, or None when it is used or when the reason says it all), `test_command`
    (`topopt gpu-check`) and `measured` (`GPU_MG_MEASURED`: the figures of one machine, labelled with it).
    """
    choice = choice or {}
    used = choice.get("solver") == "gpu-mg"
    return {"used": used, "device": choice.get("device"), "case": choice.get("case"), "self_test": choice.get("self_test"),
            "why": choice.get("line"), "to_use_the_gpu": None if used else _gpu_hint(opencl),
            "test_command": _python_command("gpu-check"), "measured": GPU_MG_MEASURED}


def _gpu_hint(opencl: dict) -> str | None:
    """What would make the GPU solver usable, from the `opencl` facts. None when the facts name no step (the reason of the pick says why)."""
    problem = opencl.get("problem") or ""
    if problem.startswith("not asked"):       # TOPOPT_AUTO_GPU=0: the device list was not asked; the reason of the pick says why
        return None
    if opencl.get("pyopencl") is None:
        if problem.startswith("missing"):
            return f"pyopencl is not installed. Install it with: {opencl.get('install_command')}"
        return f"pyopencl is installed and does not load ({problem}). Fix that first: it is a fault of the OpenCL library or its driver, not of topopt"
    devices = opencl.get("devices") or []
    from .gpu_mg.probe import PROBLEM_STARTS
    if problem.startswith(PROBLEM_STARTS):      # the child process of the probe did not give a device list
        return (f"The child process that lists the OpenCL devices gave no list ({problem}). Check the GPU driver. `TOPOPT_AUTO_GPU=0` keeps `auto` "
                "on the CPU solver with no wait, and `settings.solver: gpu-mg` asks for the GPU in this process")
    if not devices:
        return ("OpenCL shows no device. A GPU driver with OpenCL support makes one visible (on Linux the driver gives an ICD file; "
                "an Apple silicon Mac has OpenCL in macOS)")
    if all(d.get("kind") != "GPU" for d in devices):
        names = ", ".join(f"{d['name']} ({d['kind']})" for d in devices)
        return (f"OpenCL shows no GPU, only: {names}. `auto` uses a GPU only, because the speed of gpu-mg is measured on GPUs. "
                "`settings.solver: gpu-mg` runs on such a device when you ask for it")
    return None


def _problem_facts(yaml_path) -> dict:
    """Time estimate per backend for one problem file. Errors come back as data. One build only."""
    from .cli import check_data
    note, setting = None, None
    try:
        cfg = load_config(yaml_path)
        try:
            resolve_solver(cfg.settings.solver)
        except UserError as e:
            # The named solver cannot run here. The estimates stay useful: use auto for them.
            note, setting = str(e), "auto"
        d = check_data(yaml_path, solver_setting=setting)
    except UserError as e:
        return {"ok": False, "yaml": str(yaml_path), "error": str(e)}
    return {
        "ok": True, "yaml": d["yaml"], "name": d["name"],
        "free_dofs": d["free_dofs"], "iterations": d["iterations"],
        "settings_solver": d["solver_setting"] if note is None else None,
        "solver_setting_problem": note,
        "solver": d["solver"],
        "time_estimate_s": {name: (info["time_estimate"] or {}).get("total_s")
                            for name, info in d["backends"].items()},
        "time_estimate": d["time_estimate"],
        "memory_estimate": d["memory_estimate"],
        "backend_warnings": [w for w in d["warnings"] if w.startswith("settings.solver is")],
        "recommendation": d["recommendation"],
    }


def _tet_facts() -> dict:
    """The state of the optional tet mesher (CONTRACTS.md section 14), as plain data.

    The probe runs in a child Python, so this process never loads libgmsh.
    """
    try:
        from .tet.gmsh_probe import probe_subprocess
        return probe_subprocess().as_dict()
    except Exception as exc:        # never fail the whole report
        return {"state": "load_failed", "version": None,
                "detail": f"the probe did not run: {type(exc).__name__}: {exc}", "fix": ""}


def environment_data(yaml_path=None) -> dict:
    """Return the facts about this machine and the solver backends.

    With `yaml_path`, `problem` holds the time estimate for each backend. Estimates are model
    values (CONTRACTS.md section 11), not promises. `recommendations` is a list of offers
    (reason, speed-up, command, license note). Nothing is installed or built here.
    """
    from .cli import backend_availability
    backends = backend_availability()
    resolution = _resolution()      # `auto` first: its child answer also serves the OpenCL facts below (one child, one time limit)
    opencl = _opencl_facts()
    auto = resolution["auto"]["resolved"]
    for name, info in backends.items():
        info["resolved_by"] = [s for s, r in resolution.items() if r.get("cpu_solver", r["resolved"]) == name]   # `auto` counts with its CPU solver (the fallback)
    recs = []
    problem = _problem_facts(yaml_path) if yaml_path is not None else None
    if problem and problem.get("ok"):
        est = {k: v for k, v in problem["time_estimate_s"].items() if v is not None}
        rec = recommendation("auto", auto, est) if auto else None
    else:
        rec = recommendation("auto", auto) if auto else None
    if rec:
        recs.append(rec)
    shim = _shim_offer(backends)
    if shim:
        recs.append(shim)
    if not backends["amg"]["available"]:
        recs.append(_pyamg_offer())
    return {
        "topopt_version": __version__,
        # The folder with topopt/ and, in a checkout or a plugin, examples/ and tools/ (check_part.py).
        "code_folder": str(Path(__file__).resolve().parent.parent),
        # Git or not, revision, dirty, ahead or behind origin as of the last fetch. Read-only; it never fetches.
        "checkout": checkout_facts(),
        "python": {"version": platform.python_version(), "executable": sys.executable,
                   "implementation": platform.python_implementation()},
        "platform": {"name": platform_name(), "system": sys.platform, "machine": platform.machine(),
                     "release": platform.release(), "apple_silicon": is_macos_arm64(),
                     "linux_x86_64": is_linux_x86_64(), **_mac_facts()},
        "packages": _package_versions(),
        "tet": _tet_facts(),
        "backends": backends,
        "solver_setting": resolution,
        "experimental_solvers": _experimental(opencl),
        "opencl": opencl,
        "auto_solver": auto,
        "auto_choice": resolution["auto"].get("choice"),
        "gpu_solver": _gpu_solver(opencl, resolution["auto"].get("choice")),
        "ram": _ram(),
        "cores": _cores(),
        "threads": _threads(),
        "problem": problem,
        "recommendations": recs,
        "note": ("Facts and estimates only. Nothing was installed or built. A recommendation "
                 "is an offer: tell the user the reason and the speed-up, and run its command "
                 "only after the user agrees."),
    }


if __name__ == "__main__":
    print(json.dumps(environment_data(sys.argv[1] if len(sys.argv) > 1 else None), default=str))
