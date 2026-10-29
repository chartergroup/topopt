"""OpenCL facts and the errors of the gpu-mg solver (CONTRACTS.md section 20).

This module is light: it imports pyopencl only inside the functions, and only when they run.
`resolve_solver("gpu-mg")`, `topopt check` and the environment report use it. The engine
(`engine.py`, which needs pyopencl at import) is loaded when the run starts.

An error of an explicit `settings.solver: gpu-mg` is a fact with options. It never falls back to another solver.
`settings.solver: auto` picks gpu-mg on a usable GPU (`auto_gpu`, `self_test`) and falls back to the CPU solver of the
platform rule when gpu-mg fails during a run (`topopt/solver_fallback.py`).
"""
from __future__ import annotations

import dataclasses
import importlib
import functools
import importlib.util
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import GPU_CASES, UserError
from ..uv_command import uv_command

PYOPENCL_PIN = "pyopencl==2026.1.4"

# The self-test of `settings.solver: auto` (`self_test`, `selftest.py`): one small gpu-mg solve at this relative residual, against a
# direct solve on the host. `auto` picks gpu-mg only when the relative difference of the displacements is at most SELFTEST_MAX_DIFF.
# Constants for now: they are not settings fields.
SELFTEST_TOL = 1e-8
SELFTEST_MAX_DIFF = 1e-6

# What the engine needs from a device (facts for the feature check). Recorded defaults.
MIN_LOCAL_MEM = 4096            # bytes: the reduction scratch of the largest work-group
MIN_CONSTANT_MEM = 16384        # bytes: the element matrix and the transfer weights


AUTO_GPU_VARIABLE = "TOPOPT_AUTO_GPU"       # the value "0" keeps `settings.solver: auto` on the CPU rule; an explicit gpu-mg is not changed

GPU_FAILURE_KINDS = ("device_error", "device_memory", "not_accepted", "setup")
# The status codes of an OpenCL error that say the device is out of memory (`pyopencl.status_code`). pyopencl also raises its own class `MemoryError` for them.
_MEMORY_STATUS_NAMES = ("MEM_OBJECT_ALLOCATION_FAILURE", "OUT_OF_RESOURCES", "OUT_OF_HOST_MEMORY")


class GpuFailure(UserError):
    """The gpu-mg solver failed on the device. A run with `settings.solver: auto` goes on with the CPU solver (`topopt/solver_fallback.py`).

    `kind` says what failed (the fallback writes it into the record):
      device_error   an OpenCL error in an upload or a solve
      device_memory  an OpenCL error that says the device is out of memory (`MEM_OBJECT_ALLOCATION_FAILURE`, `OUT_OF_RESOURCES`, `OUT_OF_HOST_MEMORY`)
      not_accepted   a solve that did not reach its target (the residual target and the backward-error rule both missed): `model.SolveFailure`
      setup          the engine cannot be made for this model on this device (the device, the hierarchy, the largest buffer, the local memory of the
                     joint kernel, the build of the program)
    `solver` is the `GpuSolver` that raised the error, when it exists: the fallback releases its device buffers. The text is the text of the
    `UserError`; a run with the explicit setting gpu-mg stops with it, as before. A host `MemoryError`, an error of the input, a request
    to stop and an internal error (`RuntimeError`) are not a `GpuFailure`.
    """

    def __init__(self, text, kind: str, solver=None):
        super().__init__(text)
        if kind not in GPU_FAILURE_KINDS:
            raise ValueError(f"GpuFailure kind must be one of {', '.join(GPU_FAILURE_KINDS)}, got {kind!r}")
        self.kind, self.solver = kind, solver


def is_memory_error(cl, error) -> bool:
    """True when the OpenCL error `error` (a `pyopencl.Error`) says that the device is out of memory. `cl` is the module pyopencl."""
    if isinstance(error, getattr(cl, "MemoryError", ())):
        return True
    try:
        code = error.code
    except Exception:         # an error made by hand has no status record
        return False
    return code in {getattr(cl.status_code, name, None) for name in _MEMORY_STATUS_NAMES} - {None}


@dataclass
class DeviceInfo:
    """Facts about one OpenCL device."""

    index: int
    platform: str
    name: str
    kind: str                   # "GPU", "CPU", "ACCELERATOR" or "OTHER"
    fp64: bool
    version: str = ""
    vendor: str = ""
    driver: str = ""            # `driver_version`: with the device name and version it keys the cached self-test (`selftest_path`)
    max_work_group_size: int = 0
    max_mem_alloc: int = 0
    global_mem: int = 0
    local_mem: int = 0
    constant_mem: int = 0
    compute_units: int = 0
    available: bool = True
    compiler: bool = True
    obj: object = field(default=None, repr=False, compare=False)

    def label(self) -> str:
        return (f"{self.index}: {self.name} ({self.kind}, platform {self.platform}, {self.version or 'OpenCL ?'}, "
                f"64-bit float {'yes' if self.fp64 else 'no'})")

    def facts(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "obj"}


def solver_options_text() -> str:
    return ("set settings.solver to 'auto', 'cholesky' (Apple silicon), 'pardiso' (Linux x86_64, needs the "
            "pardiso extra) or 'amg'")


def install_command() -> str:
    """The command to install the `gpu` extra into the Python that runs topopt now."""
    py = shlex.quote(sys.executable)
    root = Path(__file__).resolve().parent.parent.parent
    if (root / "pyproject.toml").is_file():
        return f"{uv_command()} pip install --python {py} -e {shlex.quote(str(root) + '[gpu]')}"
    return f"{uv_command()} pip install --python {py} {shlex.quote(PYOPENCL_PIN)}"


def pyopencl_problem() -> tuple[str, str] | None:
    """Why pyopencl does not load, or None. `("missing", text)` or `("load_failed", text)`."""
    if importlib.util.find_spec("pyopencl") is None:
        return "missing", "No module named 'pyopencl'"
    try:
        importlib.import_module("pyopencl")
    except ModuleNotFoundError as e:
        if e.name in (None, "pyopencl"):
            return "missing", str(e)
        return "load_failed", f"{type(e).__name__}: {e}"   # a module under pyopencl is missing
    except Exception as e:   # for example the OpenCL library does not load
        return "load_failed", f"{type(e).__name__}: {e}"
    return None


def import_pyopencl():
    """Return the module pyopencl, or raise a `UserError` with the fact and the options."""
    problem = pyopencl_problem()
    if problem is None:
        return importlib.import_module("pyopencl")
    if problem[0] == "missing":
        raise UserError(f"solver 'gpu-mg' needs pyopencl, and it is not installed. Install the gpu extra: "
                        f"{install_command()}. Or {solver_options_text()}")
    raise UserError(f"solver 'gpu-mg' needs pyopencl, and it is installed but it does not load: {problem[1]}. "
                    f"Check the OpenCL library of this machine, or install the gpu extra again: "
                    f"{install_command()}. Or {solver_options_text()}")


def _kind(cl, dev_type) -> str:
    for name in ("GPU", "CPU", "ACCELERATOR"):
        if dev_type & getattr(cl.device_type, name):
            return name
    return "OTHER"


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


_DEVICE_CACHE: list[DeviceInfo] | None = None


_SELFTEST_CACHE: dict = {}


def forget_devices() -> None:
    """Drop the cached device list and the cached self-tests: the next `enumerate_devices` asks the OpenCL drivers again."""
    global _DEVICE_CACHE
    _DEVICE_CACHE = None
    _SELFTEST_CACHE.clear()
    from . import probe
    probe.forget()


def _list_devices() -> list[DeviceInfo]:
    cl = import_pyopencl()
    try:
        platforms = cl.get_platforms()
    except cl.Error:    # no OpenCL platform (no driver, or an empty ICD list)
        return []
    out: list[DeviceInfo] = []
    for p in platforms:
        try:
            devices = p.get_devices()
        except cl.Error:
            continue
        for d in devices:
            fp64 = _safe(lambda: d.double_fp_config != 0, False)
            out.append(DeviceInfo(
                index=len(out), platform=str(_safe(lambda: p.name, "?")).strip(),
                name=str(_safe(lambda: d.name, "?")).strip(), kind=_kind(cl, d.type), fp64=bool(fp64),
                version=str(_safe(lambda: d.version, "")).strip(),
                vendor=str(_safe(lambda: d.vendor, "")).strip(),
                driver=str(_safe(lambda: d.driver_version, "")).strip(),
                max_work_group_size=int(_safe(lambda: d.max_work_group_size, 0)),
                max_mem_alloc=int(_safe(lambda: d.max_mem_alloc_size, 0)),
                global_mem=int(_safe(lambda: d.global_mem_size, 0)),
                local_mem=int(_safe(lambda: d.local_mem_size, 0)),
                constant_mem=int(_safe(lambda: d.max_constant_buffer_size, 0)),
                compute_units=int(_safe(lambda: d.max_compute_units, 0)),
                available=bool(_safe(lambda: d.available, True)),
                compiler=bool(_safe(lambda: d.compiler_available, True)), obj=d))
    return out


def enumerate_devices() -> list[DeviceInfo]:
    """All OpenCL devices of all platforms. No context is made. An empty list if there is none.

    The drivers are asked once for each process. Each call returns new `DeviceInfo` objects, so
    a caller can change them. `forget_devices()` asks again.
    """
    global _DEVICE_CACHE
    import_pyopencl()     # a UserError if pyopencl is missing or does not load, also with a cached list
    if _DEVICE_CACHE is None:
        _DEVICE_CACHE = _list_devices()
    return [dataclasses.replace(d) for d in _DEVICE_CACHE]


def no_device_error() -> UserError:
    return UserError("solver 'gpu-mg' found no OpenCL device on this machine. Install an OpenCL driver "
                     "(on Linux, pocl gives a CPU device; on a Mac, OpenCL is part of macOS), or "
                     + solver_options_text())


def pick_device(spec, devices: list[DeviceInfo]) -> DeviceInfo:
    """The device for `settings.gpu_device`: "auto", the number in the list, or a part of the name.

    "auto" takes the first GPU, else the first accelerator, else the first CPU device.
    Raise a `UserError` with the list of the devices when there is none or when `spec` fits none.
    """
    listing = "; ".join(d.label() for d in devices) or "none"
    if not devices:
        raise no_device_error()
    spec = "auto" if spec is None else str(spec).strip()
    if spec.lower() == "auto":
        for kind in ("GPU", "ACCELERATOR", "CPU"):
            for d in devices:
                if d.kind == kind:
                    return d
        return devices[0]
    if spec.isdigit():
        if int(spec) < len(devices):
            return devices[int(spec)]
        raise UserError(f"settings.gpu_device is {spec}, and there are {len(devices)} OpenCL devices ({listing}). "
                        f"Options: use a number from this list, a part of a device name, or 'auto'")
    hits = [d for d in devices if spec.lower() in f"{d.platform} {d.name}".lower()]
    if not hits:
        raise UserError(f"settings.gpu_device is {spec!r}, and no OpenCL device name has this text "
                        f"(devices: {listing}). Options: use a number from this list, a part of a "
                        f"device name, or 'auto'")
    return hits[0]


def feature_problems(dev: DeviceInfo) -> list[str]:
    """Facts about what the device lacks for the engine. Empty if it has all."""
    out = []
    if not dev.available:
        out.append("the device reports that it is not available")
    if not dev.compiler:
        out.append("the device has no OpenCL compiler")
    if dev.local_mem and dev.local_mem < MIN_LOCAL_MEM:
        out.append(f"local memory is {dev.local_mem} bytes, and the engine needs {MIN_LOCAL_MEM}")
    if dev.constant_mem and dev.constant_mem < MIN_CONSTANT_MEM:
        out.append(f"constant memory is {dev.constant_mem} bytes, and the engine needs {MIN_CONSTANT_MEM}")
    return out


def choose_case(dev: DeviceInfo, requested="auto") -> str:
    """The precision case on `dev`. "auto": B if the device has 64-bit floats, else C.

    A forced "B" on a device without 64-bit floats is a `UserError`; a forced "C" always works.
    """
    requested = "auto" if requested is None else str(requested)
    if requested not in GPU_CASES:
        raise UserError(f"settings.gpu_case must be one of {', '.join(GPU_CASES)}, got {requested!r}")
    if requested == "auto":
        return "B" if dev.fp64 else "C"
    if requested == "B" and not dev.fp64:
        raise UserError(f"settings.gpu_case is 'B' (a 64-bit outer PCG on the device), and the device "
                        f"{dev.name} reports no 64-bit float support. Options: set settings.gpu_case to "
                        f"'C' or 'auto' (32-bit on the device, 64-bit refinement on the CPU), or choose "
                        f"another settings.gpu_device")
    return requested


def select(device="auto", case="auto") -> tuple[DeviceInfo, str]:
    """The device and the case for a run. Raise a `UserError` (fact and options) if none is usable."""
    dev = pick_device(device, enumerate_devices())
    problems = feature_problems(dev)
    if problems:
        raise UserError(f"the OpenCL device {dev.name} cannot run gpu-mg: {'; '.join(problems)}. Options: choose "
                        f"another settings.gpu_device, or {solver_options_text()}")
    return dev, choose_case(dev, case)


def counts_as_gpu(dev: DeviceInfo) -> bool:
    """True when `settings.solver: auto` may use `dev`: a device of the kind GPU.

    A CPU device (pocl) and an accelerator do not count: the speed of gpu-mg is measured on GPUs only. An explicit
    `settings.solver: gpu-mg` runs on any device. It is a function of its own: a test replaces it.
    """
    return dev.kind == "GPU"


def auto_gpu(device="auto", case="auto") -> tuple[tuple[DeviceInfo, str] | None, str | None]:
    """The device and the precision case that `settings.solver: auto` may use, or the reason why there is none.

    Return `((dev, case), None)` or `(None, reason)`. Never raises. The reason is one clause for a fact line. It covers: pyopencl that does
    not load, no OpenCL device, a `gpu_device` that fits none, a device that is not a GPU (`counts_as_gpu`), a device that lacks a feature
    of the engine, and a `gpu_case` that the device cannot run. The self-test (`self_test`) comes after this.

    The environment variable `TOPOPT_AUTO_GPU=0` is the first check, before pyopencl loads: then there is no device for `auto`. A subprocess
    gets the same answer, because it has the same environment. Any other value, or no value, gives the normal rule.
    """
    if os.environ.get(AUTO_GPU_VARIABLE) == "0":
        return None, f"{AUTO_GPU_VARIABLE}=0 is set in the environment"
    problem = pyopencl_problem()
    if problem is not None:
        return None, f"pyopencl is not usable here ({problem[0]}: {problem[1]})"
    try:
        devices = enumerate_devices()
    except Exception as e:      # a driver fault must not stop a run that has the CPU solvers
        return None, f"the OpenCL drivers gave an error ({type(e).__name__}: {e})"
    if not devices:
        return None, "no OpenCL device on this machine"
    try:
        dev = pick_device(device, devices)
    except UserError as e:
        return None, str(e)
    if not counts_as_gpu(dev):
        return None, (f"the OpenCL device {dev.name} is a {dev.kind} device, not a GPU. `auto` uses a GPU only; "
                      f"settings.solver: gpu-mg runs on it when you ask for it")
    problems = feature_problems(dev)
    if problems:
        return None, f"the GPU {dev.name} cannot run gpu-mg: {'; '.join(problems)}"
    try:
        return (dev, choose_case(dev, case)), None
    except UserError as e:
        return None, str(e)


SELFTEST_FILE = "gpu-selftest.json"
SELFTEST_SCHEMA = 1


def selftest_path() -> Path:
    """The file of the cached passed self-tests: `gpu-selftest.json` in the state folder (the folder of `gpu-constants.json`)."""
    from .check import state_dir        # light module
    return state_dir() / SELFTEST_FILE


def code_hash() -> str:
    """The sha256 of the bytes of every `*.py` file in the folder of this package (`topopt/gpu_mg/`), sorted by name. A change of the gpu-mg code tests again."""
    return _code_hash(str(Path(__file__).resolve().parent))


@functools.lru_cache(maxsize=None)
def _code_hash(folder: str) -> str:
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(folder).glob("*.py"), key=lambda p: p.name):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def selftest_key(dev: DeviceInfo, case: str) -> dict:
    """What a cached pass must match, field by field: the device (platform, name, version, driver), the case, the two limits of the test, the
    topopt version and the hash of the gpu-mg code (`code_hash`: new kernel code tests again). The index of the device in the list is not a field."""
    from .. import __version__
    return {"code": code_hash(), "platform": dev.platform, "device": dev.name, "device_version": dev.version, "driver": dev.driver, "case": case,
            "tol": SELFTEST_TOL, "max_diff": SELFTEST_MAX_DIFF, "topopt": __version__}


def _read_selftest_records() -> tuple[list, str | None]:
    """`(records, note)`: the records of the cache file, and why it was not used (None when it is good or absent). Never raises."""
    path = selftest_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], None
    except OSError as e:
        return [], f"{path} cannot be read ({type(e).__name__}: {e})"
    try:
        data = json.loads(text)
        records = data["passes"]
        if data.get("schema") != SELFTEST_SCHEMA or not isinstance(records, list):
            raise ValueError("unknown schema")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return [], f"{path} is not a cache file of this version ({type(e).__name__}: {e})"
    return [r for r in records if isinstance(r, dict) and isinstance(r.get("key"), dict) and isinstance(r.get("result"), dict)], None


def _write_selftest_record(key: dict, result: dict) -> str | None:
    """Add the pass `result` for `key` to the cache file (it replaces the record of the same key). Return why it was not written, or None.

    The write is the pattern of `check.save_calibration`: a temporary file in the same folder, flushed, then `os.replace`. No lock: two
    processes at the same time can lose one record, and the next process tests again. Never raises.
    """
    import tempfile
    from .check import _flush, _flush_folder
    path = selftest_path()
    records, _bad = _read_selftest_records()        # a bad file is replaced
    keep = [r for r in records if r["key"] != key]
    text = json.dumps({"schema": SELFTEST_SCHEMA, "passes": keep + [{"key": key, "result": result}]}, indent=1)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                _flush(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        _flush_folder(path.parent)
    except OSError as e:
        return f"{path} cannot be written ({type(e).__name__}: {e})"
    return None


def self_test(dev: DeviceInfo, case: str) -> dict:
    """The self-test of `auto` on `dev` in `case`: see `selftest.run`. Cached for the process, and a pass is cached on the drive.

    The result is a dict with `passed`, `rel_diff`, `seconds`, `free_dofs`, `tol`, `max_diff`, `device`, `case`, `error`, `cached` (True when
    the result is the record of an earlier process), `tested_at` (the UTC time of a pass) and `cache_note` (why the cache file was not used,
    or None). The file `gpu-selftest.json` holds a pass for each `selftest_key`. A failure, an error or a timeout is never written, so the
    next process tests again. A file that cannot be read or written is no error: the test runs, and `cache_note` says why.
    It never raises for a failure of the device or of the solve.
    """
    key = (dev.platform, dev.name, dev.index, dev.version, dev.driver, case)
    if key not in _SELFTEST_CACHE:
        file_key = selftest_key(dev, case)
        records, note = _read_selftest_records()
        hit = next((r for r in records if r["key"] == file_key and r["result"].get("passed") is True), None)
        if hit is not None:
            _SELFTEST_CACHE[key] = {**hit["result"], "cached": True, "cache_note": None}
        else:
            from . import selftest      # heavy: loads the engine (pyopencl) and numba
            result = selftest.run(dev, case, SELFTEST_TOL, SELFTEST_MAX_DIFF)
            result = {**result, "cached": False, "tested_at": None, "cache_note": note}
            if result["passed"]:
                result["tested_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                wrote = _write_selftest_record(file_key, {k: v for k, v in result.items() if k not in ("cached", "cache_note")})
                result["cache_note"] = "; ".join(t for t in (note, wrote) if t) or None
            _SELFTEST_CACHE[key] = result
    return dict(_SELFTEST_CACHE[key])


def runtime_problem() -> str | None:
    """One line that says why gpu-mg cannot start on this machine at all, with the options; None if it can.

    This is the check of `resolve_solver("gpu-mg")`: pyopencl loads and one OpenCL device exists.
    It does not look at `settings.gpu_device` or `settings.gpu_case`: `gpu_problem` does.
    """
    try:
        if not enumerate_devices():
            raise no_device_error()
    except UserError as e:
        return str(e)
    return None


def gpu_problem(device="auto", case="auto") -> str | None:
    """One line that says why gpu-mg cannot run here with this device and case, with the options; None if it can."""
    try:
        select(device, case)
    except UserError as e:
        return str(e)
    return None


def mark_duplicates(devices: list[dict]) -> list[dict]:
    """Add `duplicate_of` to the device dicts: the index of the first device with the same platform, name and version."""
    first: dict = {}
    for d in devices:
        key = (d["platform"], d["name"], d["version"])
        d["duplicate_of"] = first.get(key)
        first.setdefault(key, d["index"])
    return devices


def facts(device="auto") -> dict:
    """The OpenCL facts for `topopt check` and the environment report. Never raises.

    `pyopencl` is the version or None. `problem` is the text when pyopencl does not load.
    `devices` lists the devices (`duplicate_of` is set for a device that has the same platform, name
    and version as an earlier one). `auto_device` is the index that "auto" takes, `auto_case` the
    case that "auto" gives for it. `chosen_device` is the index that `device` (the value of
    `settings.gpu_device`) takes, or None.
    """
    empty = {"platforms": [], "devices": [], "auto_device": None, "auto_case": None, "chosen_device": None}
    problem = pyopencl_problem()
    if problem is not None:
        return {"pyopencl": None, "problem": f"{problem[0]}: {problem[1]}", "install_command": install_command(),
                **empty}
    version = _safe(lambda: importlib.import_module("pyopencl").VERSION_TEXT, "unknown")
    try:
        devices = enumerate_devices()
    except Exception as e:   # a driver fault must not break the report
        return {"pyopencl": version, "problem": f"{type(e).__name__}: {e}",
                "install_command": install_command(), **empty}
    auto = chosen = None
    try:
        auto = pick_device("auto", devices) if devices else None
    except UserError:
        pass
    try:
        chosen = pick_device(device, devices) if devices else None
    except UserError:     # a bad value: the run stops with the text of `pick_device`
        pass
    return {"pyopencl": version, "problem": None if devices else "no OpenCL device",
            "install_command": install_command(),
            "platforms": sorted({d.platform for d in devices}),
            "devices": mark_duplicates([d.facts() for d in devices]),
            "auto_device": auto.index if auto else None,
            "auto_case": ("B" if auto.fp64 else "C") if auto else None,
            "chosen_device": chosen.index if chosen else None}
