"""Which linear solver runs on this platform (CONTRACTS.md section 4).

`resolve_solver(setting)` turns the value of `settings.solver` into the name
of the solver that really runs: "cholesky", "pardiso", "amg" or "gpu-mg". The platform
rule is only here. `run`, `check` and `refine` all call it.

The rule for "auto":

- macOS on Apple silicon with the Accelerate shim built and loading: "cholesky" (Apple Accelerate).
  Without the shim (not built, or built but it does not load), "amg", and `topopt run` prints a
  warning with the build command. An explicit `solver: cholesky` stays "cholesky": the run then
  stops with the build command.
- Linux x86_64 with pypardiso installed: "pardiso" (Intel MKL PARDISO).
- All other cases: "amg" (PyAMG-CG, needs the rigid-body nullspace). That includes an x86_64
  Python on a Mac (an Intel Mac, or Rosetta on Apple silicon): `platform_hint()` says so, and an
  explicit `solver: cholesky` is a user error there.

"gpu-mg" (matrix-free multigrid on an OpenCL device) is the main solver where the machine has a GPU. `resolve_solver("auto")`
is the CPU rule above and never gives it: `run_solver("auto")` (below) gives gpu-mg on a GPU that passes the self-test, and the CPU rule is
the fallback. `resolve_solver("gpu-mg")` checks that pyopencl loads and that an OpenCL device exists. The check of
`settings.gpu_device` and `settings.gpu_case` is in `cli.backend_problem` and in `gpu_mg.device.select`.

This module needs no numpy. pypardiso is imported only inside
`import_pypardiso()`, and never on macOS.
"""
from __future__ import annotations

import ctypes
import importlib
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import UserError
from .uv_command import uv_command

SOLVER_SETTINGS = ("auto", "cholesky", "pardiso", "amg", "gpu-mg")
# Solvers that a user must ask for, and that `auto` never gives. None now: gpu-mg is in SOLVER_SETTINGS (CONTRACTS.md section 20). The name stays
# because `environment` has the key `experimental_solvers` (empty).
EXPERIMENTAL_SOLVERS: tuple[str, ...] = ()
PYPARDISO_PIN = "pypardiso==0.4.7"

# Rough time factor of AMG against PARDISO. It is the ratio of the total run times of
# the hook example (examples/hook/hook.yaml, 80 iterations), measured on one Linux x86_64
# machine (16 cores, 22 threads) with OPENBLAS_NUM_THREADS=4: 406 s with amg, 99 s with
# pardiso. The time estimate (`cli.estimate_seconds`) applies the factor to the
# per-iteration time only. Other machines differ. Cholesky on an M1 Mac needed 148 s
# for the same run. These are single runs. In experiment E1 (run-time study,
# 2026-10-23) the 99 s is the best of 62 runs of the default hook with pardiso on a shared
# Linux host: minimum 98.8 s, median 462 s, 90th percentile 743 s, maximum 1,469 s.
AMG_TIME_FACTOR = 4.1
HOOK_AMG_S, HOOK_PARDISO_S = 406, 99
HOOK_MACHINE = "16 cores / 22 threads"
HOOK_PARDISO_NOTE = ("single runs; the pardiso time is the best of 62 runs of the default hook on a shared Linux host, "
                     "the median was 462 s (experiment E1)")

# What `pip install pypardiso` pulls in (checked in a fresh venv, 2026-10-19).
LICENSE_NOTE = (
    "pypardiso pulls Intel packages, about 456 MB in a fresh venv. "
    "mkl, tbb and tcmlib: Intel Simplified Software License. "
    "intel-openmp and intel-cmplr-lib-ur: Intel EULA for Developer Tools "
    "(more limits, for example no use in a hosted service). "
    "umf: Apache-2.0. "
    "The mkl version is not pinned: pypardiso only needs mkl != 2024.0.")


def platform_name() -> str:
    """For example "linux x86_64" or "darwin arm64"."""
    return f"{sys.platform} {platform.machine()}"


def is_macos_arm64() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_linux_x86_64() -> bool:
    return sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")


SYSCTL_PATHS = ("/usr/sbin/sysctl", "sysctl")   # the full path first: a short PATH has no /usr/sbin


def _sysctl_proc_translated() -> str | None:
    """Text of the read-only command `sysctl -n sysctl.proc_translated`, or None if it did not run.

    macOS gives "1" to a process that runs under Rosetta and "0" to a native one (macOS 11 or newer, also on
    an Intel Mac). On older macOS the key does not exist: the command fails (exit code 1) and the text is "".
    It is a function of its own: a test replaces it.
    """
    for exe in SYSCTL_PATHS:
        try:
            done = subprocess.run([exe, "-n", "sysctl.proc_translated"], capture_output=True,
                                  text=True, timeout=5)
        except OSError:
            continue   # no such file: try the next name
        except subprocess.SubprocessError:
            return None
        return done.stdout.strip() if done.returncode == 0 else ""
    return None


_KIND_CACHE: dict = {}


def mac_python_kind() -> str | None:
    """What kind of Python runs on a Mac: None (not a Mac), "arm64", "rosetta", "intel" or "unknown".

    "rosetta": an x86_64 Python on an Apple silicon Mac (macOS translates it). "intel": an x86_64
    Python that is not translated, so an Intel Mac. "unknown": an x86_64 Python on a Mac, and the
    check `sysctl.proc_translated` did not run. Both x86_64 cases take the same code path.
    The check runs once for each process (the answer cannot change while Python runs).
    """
    if sys.platform != "darwin":
        return None
    if platform.machine() == "arm64":
        return "arm64"
    key = (platform.machine(), _sysctl_proc_translated)   # the function is in the key: a test can replace it
    if key not in _KIND_CACHE:
        translated = _sysctl_proc_translated()
        _KIND_CACHE[key] = "unknown" if translated is None else ("rosetta" if translated == "1" else "intel")
    return _KIND_CACHE[key]


def is_x86_mac() -> bool:
    """True for an x86_64 Python on a Mac (an Intel Mac, or Rosetta on Apple silicon)."""
    return mac_python_kind() in ("rosetta", "intel", "unknown")


# The functions that the shim must export (the ones `solver._accel_lib` uses).
ACCEL_SYMBOLS = ("af_factor", "af_solve", "af_free", "af_factor_ex", "af_refactor")


def accelerate_library_path() -> Path:
    """Where the Accelerate shim (the library `topopt/native/build.sh` builds) must be."""
    return Path(__file__).resolve().parent / "native" / "libaccel_chol.dylib"


def _load_library(path: Path):
    return ctypes.CDLL(str(path))   # a function of its own: a test replaces it


def accelerate_problem() -> tuple[str, str] | None:
    """Why the Accelerate shim cannot run, or None if it loads. Only asked on Apple silicon.

    Return `("missing", text)` if the library file is not there, or `("load_failed", text)` if it is
    there but it does not load or lacks a function (for example, a build from an old source).
    """
    path = accelerate_library_path()
    if not path.is_file():
        return "missing", f"{path} not found"
    try:
        lib = _load_library(path)
        for name in ACCEL_SYMBOLS:
            getattr(lib, name)
    except (OSError, AttributeError) as e:
        return "load_failed", f"{type(e).__name__}: {e}"
    return None


def shim_build_command() -> str:
    """The command that builds the Accelerate shim. The path is quoted for a shell."""
    return f"sh {shlex.quote(str(accelerate_library_path().parent / 'build.sh'))}"


def import_pypardiso():
    """Import and return the module pypardiso. The only import of it in the package."""
    return importlib.import_module("pypardiso")


def pypardiso_problem() -> tuple[str, str] | None:
    """Why pypardiso does not load, or None if it loads. Only asked on Linux x86_64.

    Return `("missing", text)` if the package is not installed, or
    `("load_failed", text)` if it is installed but it fails to load (for example,
    the MKL library is not found). `text` is the text of the exception.
    """
    try:
        import_pypardiso()
    except ModuleNotFoundError as e:
        if e.name in (None, "pypardiso"):
            return "missing", str(e)
        return "load_failed", f"{type(e).__name__}: {e}"   # a module under pypardiso is missing
    except Exception as e:  # for example the MKL library does not load
        return "load_failed", f"{type(e).__name__}: {e}"
    return None


def pypardiso_available() -> bool:
    """True if pypardiso (and the MKL library under it) loads. Only asked on Linux x86_64."""
    return pypardiso_problem() is None


def install_command() -> str:
    """The command to install the `pardiso` extra into the Python that runs topopt now."""
    py = shlex.quote(sys.executable)
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").is_file():
        return f"{uv_command()} pip install --python {py} -e {shlex.quote(str(root) + '[pardiso]')}"
    return f"{uv_command()} pip install --python {py} {shlex.quote(PYPARDISO_PIN)}"


def resolve_solver(setting: str) -> str:
    """Return the solver that runs for `settings.solver == setting`.

    Raise `UserError` if the platform cannot run the solver that the user named.
    """
    if setting == "amg":
        return "amg"
    if setting == "gpu-mg":
        from .gpu_mg.device import runtime_problem     # light: pyopencl loads only inside this call
        problem = runtime_problem()      # pyopencl loads and a device exists; the device and case of the YAML: `backend_problem`
        if problem:
            raise UserError(problem)
        return "gpu-mg"
    if setting == "cholesky":
        if sys.platform != "darwin":
            raise UserError(
                f"solver 'cholesky' needs Apple Accelerate (macOS), and this platform is "
                f"{platform_name()}. In settings.solver use 'pardiso' (Linux x86_64, "
                f"needs the pardiso extra), 'amg' (all platforms, slower) or 'auto'")
        if not is_macos_arm64():   # an x86_64 Python on a Mac: the shim is for Apple silicon
            raise UserError(
                f"solver 'cholesky' needs the Apple Accelerate shim, and topopt supports it on Apple "
                f"silicon only: {x86_mac_text()}. In settings.solver use 'amg' (all platforms, about "
                f"{AMG_TIME_FACTOR:g}x slower per iteration, an estimate) or 'auto'"
                + (". For Accelerate, run a native arm64 Python (the hint of `topopt check` has the steps)"
                   if mac_python_kind() == "rosetta" else ""))
        return "cholesky"
    if setting == "pardiso":
        if not is_linux_x86_64():
            raise UserError(
                f"solver 'pardiso' runs on Linux x86_64 only, and this platform is "
                f"{platform_name()}. In settings.solver use "
                + ("" if is_x86_mac() else "'cholesky' (macOS), ")
                + "'amg' (all platforms) or 'auto'")
        problem = pypardiso_problem()
        if problem and problem[0] == "missing":
            raise UserError(
                f"solver 'pardiso' needs pypardiso, and it is not installed. Install it with: "
                f"{install_command()} ({LICENSE_NOTE}) Or set settings.solver to "
                f"'amg' (all platforms, about {AMG_TIME_FACTOR:g}x slower) or 'auto'")
        if problem:
            raise UserError(
                f"solver 'pardiso' needs pypardiso, and it is installed but it does not load: "
                f"{problem[1]}. Check that the MKL library is found (the mkl package, "
                f"the variable PYPARDISO_MKL_RT) or install the extra again with: "
                f"{install_command()}. Or set settings.solver to "
                f"'amg' (all platforms, about {AMG_TIME_FACTOR:g}x slower) or 'auto'")
        return "pardiso"
    if setting == "auto":
        if is_macos_arm64():
            # No import of pypardiso here. Without a working shim, `auto` does not stop: it runs AMG.
            return "cholesky" if accelerate_problem() is None else "amg"
        if is_linux_x86_64() and pypardiso_available():
            return "pardiso"
        return "amg"
    raise UserError(f"'settings.solver' must be one of {', '.join(SOLVER_SETTINGS + EXPERIMENTAL_SOLVERS)}, "
                    f"got {setting!r}")


@dataclass(frozen=True)
class SolverChoice:
    """The solver that a RUN uses first, and the facts that say why (`run_solver`).

    `setting` is the value of `settings.solver`. `solver` is the name that runs first: "gpu-mg", "cholesky", "pardiso" or "amg".
    `fallback` is the CPU solver that takes over when gpu-mg fails during a run: it is set only when `setting` is "auto" and `solver` is
    "gpu-mg"; else None. `cpu` is the solver of the CPU rule for `setting` ("auto") or the solver itself.
    `device` (the facts of the `DeviceInfo`), `case` and `self_test` (the dict of `gpu_mg.device.self_test`) are set when `solver` is
    "gpu-mg" and `setting` is "auto". `reason` says why `auto` did not pick gpu-mg (a clause), or is None.
    """

    setting: str
    solver: str
    cpu: str
    fallback: str | None = None
    device: dict | None = None
    case: str | None = None
    self_test: dict | None = None
    reason: str | None = None

    @property
    def auto_gpu(self) -> bool:
        """True when `auto` picked gpu-mg: a run can fall back to `fallback`."""
        return self.fallback is not None

    def line(self) -> str | None:
        """The fact line for `topopt check`, the run log and the environment report: what `auto` picked and why. None for an explicit solver."""
        if self.setting != "auto":
            return None
        if not self.auto_gpu:
            return f"solver: auto -> {self.solver} ({self.reason})" if self.reason else None
        t, d = self.self_test or {}, self.device or {}
        test = ("; self-test passed: gpu-mg against a direct solve on a small cantilever, relative difference of the displacements "
                f"{t['rel_diff']:.1e} (limit {t['max_diff']:g}) at gpu tolerance {t['tol']:g}, "
                f"{('cached result of ' + str(t.get('tested_at'))) if t.get('cached') else format(t['seconds'], '.1f') + ' s'}"
                f"{'; the self-test cache was not used: ' + t['cache_note'] if t.get('cache_note') else ''}") if t else ""
        return (f"solver: auto -> gpu-mg on {d.get('name')} ({d.get('kind')}, platform {d.get('platform')}, case {self.case}){test}; "
                f"if gpu-mg fails during the run, the run goes on with {self.fallback} from the last checkpoint")

    def as_dict(self) -> dict:
        """The choice as plain data (`topopt check --json`, the environment report)."""
        return {"setting": self.setting, "solver": self.solver, "cpu_solver": self.cpu, "fallback": self.fallback, "device": self.device,
                "case": self.case, "self_test": self.self_test, "reason": self.reason, "line": self.line()}


def run_solver(setting: str, config=None) -> SolverChoice:
    """The solver that a RUN uses first for `settings.solver == setting`, and why (CONTRACTS.md section 4 and section 20).

    `resolve_solver` stays the rule for the matrix solvers: it turns "auto" into the CPU rule (cholesky, pardiso or amg). This function
    decides what a run uses: an explicit value gives what `resolve_solver` gives (a `UserError` where the platform cannot run it), and
    "auto" gives gpu-mg when all of these hold, else the CPU rule with the reason as a fact:

    1. The YAML has no feature that gpu-mg cannot run (today the buckling row, which needs a direct factor of the matrix).
    2. pyopencl loads, and the device that `settings.gpu_device` takes is a GPU (`gpu_mg.device.counts_as_gpu`; pocl and other CPU devices do not
       count) with the features of the engine, and `settings.gpu_case` fits it (`gpu_mg.device.auto_gpu`).
    3. The self-test passes (`gpu_mg.device.self_test`, cached for the process and, for a pass, on the drive).

    Steps 2 and 3 run in a child process with a time limit (`gpu_mg.probe`); at the limit `auto` takes the CPU rule with a reason that names it.

    `config` (the `Config` of the problem) gives `gpu_device`, `gpu_case` and the buckling row; without it the defaults hold. A choice of
    gpu-mg by "auto" has `fallback`: the CPU solver for a run that has to leave gpu-mg (`topopt/solver_fallback.py`). Never raises for "auto".
    """
    cpu = resolve_solver(setting)
    if setting != "auto":
        return SolverChoice(setting, cpu, cpu)
    reason = None
    if config is not None and config.buckling.min_factor is not None:
        reason = "the buckling row (buckling.min_factor) needs a direct factor of the stiffness matrix, and gpu-mg has none"
    if reason is None:
        from .gpu_mg import probe      # light: the parent makes no OpenCL call and does not import pyopencl; a child process does (time limit)
        st = getattr(config, "settings", None)
        answer = probe.auto_pick(getattr(st, "gpu_device", "auto"), getattr(st, "gpu_case", "auto"))
        reason, dev, case, test = answer["reason"], answer["device"], answer["case"], answer["self_test"]
        if dev is not None:
            if test["passed"]:
                return SolverChoice(setting, "gpu-mg", cpu, fallback=cpu, device=dev, case=case, self_test=test)
            what = (f"error {test['error']}" if test["error"] else
                    f"the relative difference of the displacements is {test['rel_diff']:.2e}, above the limit {test['max_diff']:g}")
            reason = (f"the self-test of gpu-mg on {dev['name']} (GPU, platform {dev['platform']}, case {case}) did not pass: {what} "
                      f"(a small cantilever, gpu tolerance {test['tol']:g}, {test['seconds']:.1f} s)")
    return SolverChoice(setting, cpu, cpu, reason=reason)


def x86_mac_text() -> str:
    """One clause that says what runs: an x86_64 Python on a Mac, and which kind of Mac."""
    kind = mac_python_kind()
    if kind == "rosetta":
        return "this x86_64 Python runs under Rosetta on an Apple silicon Mac"
    if kind == "intel":
        return "this x86_64 Python runs on an Intel Mac"
    return f"this Python runs as {platform.machine()} on a Mac"


def platform_hint() -> str | None:
    """A hint for an x86_64 Python on a Mac (Rosetta or an Intel Mac), else None.

    Accelerate is for Apple silicon. With an x86_64 Python, "auto" picks "amg" and an explicit
    "cholesky" is a user error. The run is best effort: it uses the lock for macOS x86_64, which is older
    than the main lock, and nobody tested it on an Intel Mac. The time model is fitted to a native M1 run
    with Cholesky, so the estimate is not calibrated. The text says which kind of Mac it is. On Apple
    silicon it also gives the commands to a native arm64 Python (the fast path).
    """
    kind = mac_python_kind()
    if kind in (None, "arm64"):
        return None
    best_effort = (f"This is best effort: topopt was tested on Linux only, not on an Intel Mac, and the "
                   f"lock for macOS x86_64 (requirements-macos-x86_64.lock) is older than the main lock "
                   f"(its numpy, numba, llvmlite and cryptography pins are lower). The time estimate is not calibrated for this case: the "
                   f"model is fitted to a native M1 run with Cholesky, and amg uses a factor of "
                   f"{AMG_TIME_FACTOR:g} measured on one Linux machine.")
    if kind == "intel":
        return (f"Python runs as {platform.machine()} on an Intel Mac. The Accelerate shim is for "
                f"Apple silicon, so solver auto gives amg, and solver cholesky is a user error. "
                f"{best_effort}")
    uv = uv_command()
    which_uv = 'file "$(command -v uv)"' if uv == "uv" else f"file {uv}"    # a uv that the launcher downloaded has no name on PATH
    steps = ("open the Terminal without Rosetta (Get Info on the app, clear 'Open using Rosetta'); "
             f"check uv with: {which_uv} (it must say arm64, else install uv again from "
             f"that Terminal); remove this environment with: rm -rf {shlex.quote(sys.prefix)} ; "
             "then start topopt again, and the setup makes an arm64 environment")
    if kind == "rosetta":
        who = "Apple silicon runs this Python under Rosetta. A native arm64 Python is the fast path"
    else:   # "unknown": the check of Rosetta did not run
        who = ("The check for Rosetta did not run. On an Apple silicon Mac, a native arm64 Python is the "
               "fast path (an Intel Mac has none)")
    return (f"Python runs as {platform.machine()} on this Mac. {who} (Apple Accelerate, solver cholesky): "
            f"{steps}. With this Python, solver auto gives amg, and solver cholesky is a user error. "
            f"{best_effort}")


def recommendation(setting: str, resolved: str, estimate_s: dict | None = None) -> dict | None:
    """The offer to fix the solver, as a dict for an agent. None if there is no offer.

    There is an offer only on Linux x86_64, when `auto` ends in "amg" because
    pypardiso is missing (`action` "install_pypardiso") or installed but it does
    not load (`action` "fix_pypardiso_load"). `estimate_s` can hold the run time
    estimates (`{"amg": s, "pardiso": s}`) of the problem.
    """
    if setting != "auto" or resolved != "amg" or not is_linux_x86_64():
        return None
    problem = pypardiso_problem()
    if problem is None:
        return None
    if estimate_s:
        amg_s, pd_s = round(estimate_s["amg"]), round(estimate_s["pardiso"])
        speedup = round(amg_s / pd_s, 1)
        reason_est = (f"For this problem the estimate is {amg_s} s with amg and "
                      f"{pd_s} s with pardiso. ")
    else:
        speedup = AMG_TIME_FACTOR
        reason_est = ""
    hook = (f"The hook example took {HOOK_AMG_S} s with amg and {HOOK_PARDISO_S} s with "
            f"pardiso on a {HOOK_MACHINE} machine ({HOOK_PARDISO_NOTE}).")
    if problem[0] == "missing":
        rec = {
            "action": "install_pypardiso",
            "extra": "pardiso",
            "install_command": install_command(),
            "reason": ("On Linux x86_64 the solver is AMG without pypardiso. PARDISO is faster. "
                       + reason_est + hook),
        }
    else:
        rec = {
            "action": "fix_pypardiso_load",
            "extra": "pardiso",
            "error": problem[1],
            "install_command": install_command(),
            "reason": (f"pypardiso is installed, but it does not load ({problem[1]}), so the "
                       f"solver is AMG. Check that the MKL library is found (the mkl package, "
                       f"the variable PYPARDISO_MKL_RT), or install the extra again. "
                       + reason_est + hook),
        }
    rec["speedup_factor"] = speedup
    rec["license_note"] = LICENSE_NOTE
    if estimate_s:
        rec["estimate_s"] = {"without_pardiso": amg_s, "with_pardiso": pd_s}
    return rec


def shim_warning_line(setting: str, resolved: str) -> str | None:
    """One line for `topopt run` and `topopt check`: `auto` gave AMG on a Mac because the shim cannot run."""
    if setting != "auto" or resolved != "amg" or not is_macos_arm64():
        return None
    problem = accelerate_problem()
    if problem is None:
        return None
    why = ("the Accelerate shim is not built" if problem[0] == "missing"
           else f"the Accelerate shim is built but it does not load ({problem[1]})")
    return (f"solver auto -> amg on {platform_name()}: {why}, about {AMG_TIME_FACTOR:g}x slower per "
            f"iteration (an estimate); build it with: {shim_build_command()} (needs the Xcode command "
            f"line tools). Continuing with amg.")


def warning_line(setting: str, resolved: str) -> str | None:
    """One line for `topopt run`, if `auto` gave AMG because a faster solver is missing."""
    shim = shim_warning_line(setting, resolved)
    if shim:
        return shim
    rec = recommendation(setting, resolved)
    if rec is None:
        return None
    speed = (f"pardiso is faster; the factor depends on the problem (hook: {HOOK_AMG_S} s -> "
             f"{HOOK_PARDISO_S} s, {HOOK_PARDISO_NOTE}; see speedup_factor of `topopt check`).")
    if rec["action"] == "install_pypardiso":
        return (f"solver auto -> amg on {platform_name()}: pypardiso is not installed. {speed} "
                f"Install: {rec['install_command']}. {LICENSE_NOTE} Continuing with amg.")
    return (f"solver auto -> amg on {platform_name()}: pypardiso is installed but it does not "
            f"load ({rec['error']}). Check the MKL library, or install again: "
            f"{rec['install_command']}. Continuing with amg.")
