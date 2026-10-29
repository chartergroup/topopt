"""The gpu-mg part of `topopt check` (CONTRACTS.md section 20): the memory model, the time model and the warnings.

This module is light (no pyopencl, no numpy). `cli.check_data` is wrapped so that `apply` runs on its result,
and `cli.cmd_check` prints `lines`. The code of the other solvers in `cli.py` is not changed by this.

The time model (P2a) is `estimate_seconds_gpu_mg`. It is linear in the free dofs (the Cholesky model grows as dofs^2.4):

    t_total = T0 + (N + 1) * t_pass * robust
    t_pass  = B * I * c_NR * n * S + h * n                      (seconds)
    T0      = T0_FIXED_S + T0_S_PER_DOF * n                      (start: read, voxelize, filter; end: post-process, STL)
    B       = ceil(load cases / 2) when there is more than one load case, else 1   (2 columns for each launch)
    c_NR    = seconds for each free dof and PCG iteration of one block (one column, or two columns)
    I       = PCG iterations of one block, mean over a run (the band by stiffness contrast, times the gpu_tol factor)
    S       = 1 + share * (other gpu-mg runs on the same device)     (time slicing; 0 other runs in `check`)
    h       = host part of one optimizer pass for each free dof

The constants are in ONE table, keyed by device class (`TIME_CONSTANTS`). A device with no row gets no constants of its own:
`topopt gpu-check` measures them (c_NR and h) and writes them to `gpu-constants.json` in the per-user state folder (`state_dir`);
`check` reads that file first. A device with neither is shown with the constants of the reference row, and the text says so.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Memory model of the solver gpu-mg: the peak resident memory (RSS) of the whole `topopt run` is the largest of three phases.
# MEASURED ON POCL (the OpenCL CPU device of the Linux host), where the device buffers are host memory and are in these numbers (about
# 0.3 kB for each free dof), and where pocl compiles the kernels with LLVM (a part of the base). On a GPU with its own memory the device
# part is not in the host memory. See CONTRACTS.md section 20, "Memory model of `check`".
#
# The loop phase (start, hierarchy, optimizer passes): LOOP = BASE + PER_DOF * N, plus JOINT_PER_DOF * N for case C with joints (the
# columns of the deflation, 0.95 to 0.97 kB for each free dof in the pair of runs B and C of the same part), plus the FFT filter that the run
# holds (`filter_held_bytes`, `members.filter_size`). The runs that gave BASE and PER_DOF had a filter too (then a sparse matrix), so the sum can count a
# small filter part twice (the number can be higher than a measured peak); the FFT filter follows the padded grid and the design voxels
# (56 B a padded cell plus 24 B a design voxel, #333).
# The build phase (the filter is set up before the model): BUILD = BASE + `filter_build_peak_bytes` (`members.filter_size`). It is the
# CPU model without the factor: gpu-mg makes none. Without the filter numbers (0, "not known") the loop phase is the old number and the build phase is BASE.
# The mesh phase (the post-process after the last pass: the SDF mesh with `snap_into_regions`, whose `trimesh` closest-point queries made
# one array for all pairs of a vertex and a nearby face until #429; they now take 8,192 points at a time, so this fit is high): MESH = MESH_BASE + LAYOUT * MESH_PER_DOF23 * N ** (2/3) + MESH_PER_GRID_VOXEL * V,
# with N the free dofs and V the voxels of the grid. The number of mesh vertices follows N ** (2/3) (the surface), and the hook fits
# 387 MB + 0.365 MB * N ** (2/3) within 3 percent over 16,434 to 905,010 free dofs (8 sizes).
# `typical_bytes` is the fit with LAYOUT 1; `bytes` uses GPU_MG_LAYOUT_FACTOR, which covers the stress layouts that were measured (a keep-out
# shape whose bounding box covers the whole part, and a small keep-out shape with 320 triangles): at the same N and V the mesh phase was
# up to 2.4 times that of the plain hook, and its surface term up to 2.7 times the typical fit (a fit of all 47 rows alone gives 2.95 times).
# The region layout is not known to `check`; the factor is a measured cover, not a bound.
# Measured cases: 47 runs on 2026-10-25 (49 were recorded; a 4-pass harness test and a case-C run that stopped early are not in the fit),
# one Linux x86_64 host (Intel Core Ultra 9 185H, shared), pocl CPU device, 8 compute units, revision 74a1e46 of main (hook at 8 sizes, a
# sparse frame at 6 sizes, a block with 0 to 2 joints in case B and C, the hook with a far keep-out shape, with dense keep-out shapes,
# 6 load cases, a larger filter radius and the blur mesh method), 13 to 81 passes;
# the 15 runs of the old fit (2026-10-20, 610 MB + 1,000 bytes for each free dof, hook before the SDF mesh step) stay below both lines.
# The fit is an upper envelope with 5 percent over every measured peak of its class.
GPU_MG_BASE_BYTES, GPU_MG_BYTES_PER_DOF = 860_000_000, 730          # the base also stands in the build phase (the process before the filter)
GPU_MG_JOINT_BYTES_PER_DOF = 1000
GPU_MG_MESH_BASE_BYTES, GPU_MG_MESH_BYTES_PER_DOF23, GPU_MG_MESH_BYTES_PER_GRID_VOXEL = 500_000_000, 370_000, 220
GPU_MG_LAYOUT_FACTOR = 3.2

# Warnings of `check_data` that are about the models of the other solvers. For gpu-mg the memory one is replaced
# (the gpu-mg model has its own). The time warnings stay: they are made with the gpu-mg number (P2a).
_REPLACED_WARNINGS = ("the memory estimate", "the filter needs")       # the filter line is made again with the gpu-mg numbers (`apply`)


# ---- the time model ------------------------------------------------------------------------------------------------

# The stiffness of void of `optimize.E_MIN`; this module stays light, so it has its own copy (a test checks that they are equal).
VOID_STIFFNESS = 1e-6
# The power of the free dofs in the model: it is linear (the fit of 14 isolated runs gave n^0.998). `cli.fit_voxel_size` and
# `regions_builder.target_active` take the power of the model that runs, so a voxel size for gpu-mg is not a Cholesky guess.
GPU_MG_DOF_POWER = 1.0
# The bands below were fitted on runs at `gpu_tol` 1e-8 (56 of the 58 runs) and 1e-6 (2 runs), and the research says that they hold from 1e-8 to
# 1e-6, so there is NO factor in that range (P2a review m1). The one real pair (the same problem at 1e-8 and at 1e-6) gave 0.92 for the mean iterations;
# the earlier factor of 0.77 at 1e-6 came from pocl runs on synthetic designs. Outside the range the count is scaled by the rule of thumb of those pocl
# runs (52,920 dofs, first-pass cap 2000): one decade of `gpu_tol` changes the iterations by 15 percent of the count at 1e-6 (1e-4: 0.57 to 0.77;
# 1e-8: 1.31 of 1e-6). Iteration counts do not depend on the device. Looser than 1e-6: `1 + 0.15 log10(1e-6 / tol)`. Tighter than 1e-8: the same
# rule, over its value at 1e-8, so that the factor is 1.0 at 1e-8 and goes on from there.
TOL_SLOPE_PER_DECADE = 0.15
BAND_TOL_TIGHT, BAND_TOL_LOOSE = 1e-8, 1e-6
# The iteration bands by stiffness contrast `lgK = -log10(VOID_STIFFNESS + volume_fraction ** penal)`: the mean PCG iterations
# of one block over a whole run (low, mid, high), for `gpu_tol` 1e-8 to 1e-6. Case B on one Intel Arc iGPU, 49 finished runs. A contrast that is not
# known (no penal) gets the band of all runs. The count is the dominant unknown: 22 to about 2,000 were seen for similar dof counts.
ITERATION_BANDS = (       # (lgK upper limit, (low, mid, high))
    (2.5, (50, 65, 80)),
    (5.0, (100, 145, 200)),
    (math.inf, (150, 200, 300)),
)
ITERATION_BAND_UNKNOWN = (60, 150, 300)
# Two columns for each launch cost this many times one column (the fit of 14 and 12 isolated runs: 15.3 and 23.3 ns). It is used to
# turn a calibration of one column count into the other (the hook has 2 load cases, so `gpu-check` measures two columns).
FIT_RUNS, FIT_INSIDE, FIT_BELOW, FIT_ABOVE, FIT_ABOVE_SHARED = 49, 19, 14, 16, 15     # the band against the wall time of the fit runs (research of 2026-10-22, not in the repo)
TWO_COLUMN_RATIO = 1.52


@dataclass(frozen=True)
class DeviceConstants:
    """The constants of one device class and precision case. Seconds, and seconds for each free dof."""

    key: str                # the key of the table
    label: str              # for the text
    case: str               # the precision case of the measurement
    c1_s: float             # one PCG iteration of one block of ONE column, for each free dof
    c2_s: float             # the same for a block of TWO columns
    host_s_per_dof: float   # the host part of one optimizer pass (setup, upload, map, filter, MMA, snapshot), for each free dof
    t0_s: float             # start and end of the job: the fixed part
    t0_s_per_dof: float     # ... and the part for each free dof
    share: float            # each other gpu-mg run on the same device adds this share of the isolated time
    source: str             # where the numbers come from (for the JSON)
    short: str              # the same in a few words (for the text of `check`)


TIME_CONSTANTS = {
    "intel-arc-igpu": DeviceConstants(
        key="intel-arc-igpu", label="Intel Arc iGPU", case="B", c1_s=15.3e-9, c2_s=23.3e-9, host_s_per_dof=2.0e-6,
        t0_s=8.1, t0_s_per_dof=80e-6, share=0.65,
        source=("a fit of test runs on one Linux host: 49 finished runs "
                "(170k to 790k free dofs), 14 isolated runs for c, case B; the host part is 1.2 us (170k dofs) to 2.4 us "
                "(1.35M dofs) for each free dof"),
        short="a fit of 49 finished test runs, one Linux host"),
}
REFERENCE_KEY = "intel-arc-igpu"        # the row that a device with no constants is shown with, named as a reference


def device_class(device: dict | None, case: str | None = None) -> str | None:
    """The key of `TIME_CONSTANTS` for an OpenCL device (the dict of `DeviceInfo.facts()`) and a precision case, or None.

    The row is matched by what the driver reports: a GPU of an Intel platform named "Intel(R) Arc(TM) Graphics" (the iGPU; an Arc
    card has a model name), and the case of the measurement.
    """
    if not device or device.get("kind") != "GPU":
        return None
    name = re.sub(r"\((r|tm)\)", "", str(device.get("name", "")).lower()).split()
    if "intel" not in f"{device.get('platform', '')} {device.get('vendor', '')}".lower() or name != ["intel", "arc", "graphics"]:
        return None
    key = "intel-arc-igpu"
    return key if case in (None, TIME_CONSTANTS[key].case) else None


def device_case(device: dict | None, gpu_case: str | None = "auto") -> str | None:
    """The precision case that a run on `device` uses: `settings.gpu_case`, or B with 64-bit float and C without ("auto"). None without a device."""
    if gpu_case in ("B", "C"):
        return gpu_case
    return None if not device else ("B" if device.get("fp64") else "C")


def chosen_device(opencl: dict | None) -> dict | None:
    """The device that `settings.gpu_device` takes, from the `opencl` facts of `device.facts()`, or None."""
    if not opencl or opencl.get("chosen_device") is None:
        return None
    return next((d for d in opencl.get("devices", []) if d.get("index") == opencl["chosen_device"]), None)


def device_name(device: dict | None) -> str:
    return str(device.get("name")).strip() if device and device.get("name") else "not known"


# ---- the per-user state folder and the calibration file --------------------------------------------------------------

STATE_VARIABLE = "TOPOPT_STATE_DIR"
CALIBRATION_FILE = "gpu-constants.json"      # not `gpu-check.json`: that is the report file of `topopt gpu-check`
CALIBRATION_SCHEMA = 1


def state_dir() -> Path:
    """The per-user state folder of topopt: the folder in `TOPOPT_STATE_DIR`, else `~/.topopt`. It is not made until a file is written."""
    env = os.environ.get(STATE_VARIABLE)
    return Path(env).expanduser() if env else Path.home() / ".topopt"


def calibration_path() -> Path:
    return state_dir() / CALIBRATION_FILE


def _number(value, positive: bool) -> bool:
    """True for a finite int or float (not a bool, not a text); `positive`: above 0, else 0 or more."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return value > 0 if positive else value >= 0


def record_problem(record) -> str | None:
    """The first field of a calibration record that the model cannot use (named, with the reason), or None when the record is good.

    Every field that `constants_for`, `same_device` and `find_calibration` read: `device` (an object with the texts `platform` and
    `name`), `case` (a text), `columns_per_launch` (1 or 2), `c_s_per_dof_iteration` (a finite number above 0) and `host_s_per_dof` (a
    finite number, 0 or more). `time_utc` is a text when it is there.
    """
    if not isinstance(record, dict):
        return "the record is not an object"
    device = record.get("device")
    if not isinstance(device, dict):
        return "field `device` is not an object"
    for key in ("platform", "name"):
        if not isinstance(device.get(key), str) or not device[key].strip():
            return f"field `device.{key}` is not a text"
    if not isinstance(record.get("case"), str):
        return "field `case` is not a text"
    cols = record.get("columns_per_launch")
    if isinstance(cols, bool) or cols not in (1, 2):
        return "field `columns_per_launch` is not 1 or 2"
    if not _number(record.get("c_s_per_dof_iteration"), True):
        return "field `c_s_per_dof_iteration` is not a finite number above 0"
    if not _number(record.get("host_s_per_dof"), False):
        return "field `host_s_per_dof` is not a finite number of 0 or more"
    if record.get("time_utc") is not None and not isinstance(record["time_utc"], str):
        return "field `time_utc` is not a text"
    return None


def load_calibrations() -> tuple[list[dict], str | None]:
    """`(records, problem)` of `gpu-constants.json` in the state folder: only the records that `record_problem` accepts.

    A missing file gives `([], None)`. A file that cannot be read, is not JSON (or not UTF-8 text), is not a file of this schema or has no
    list of records gives `([], text)`. A bad record is dropped and the text names the file and the field of each one (P2a review M1): a bad
    file never stops `check`, `check_problem`, the builder or a run.
    """
    path = calibration_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return [], None
    except OSError as exc:
        return [], f"{path} could not be read ({exc.strerror or exc}); it is ignored"
    try:
        doc = json.loads(raw.decode("utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("the top level is not an object")
        if doc.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError(f"schema {doc.get('schema')!r}, not {CALIBRATION_SCHEMA}")
        listed = doc["calibrations"]
        if not isinstance(listed, list):
            raise ValueError("`calibrations` is not a list")
    except (ValueError, KeyError) as exc:       # a UnicodeDecodeError and a JSONDecodeError are ValueErrors
        return [], f"{path} is not a calibration file of this version ({type(exc).__name__}: {exc}); it is ignored"
    records, dropped = [], []
    for i, rec in enumerate(listed):
        why = record_problem(rec)
        if why is None:
            records.append(rec)
        else:
            dropped.append(f"record {i}: {why}")
    note = (f"{path}: {len(dropped)} of {len(listed)} record{'s' if len(listed) != 1 else ''} ignored (" + "; ".join(dropped) + ")") if dropped else None
    return records, note


def same_device(record_device: dict | None, device: dict | None) -> bool:
    """A record is for a device when the platform and the name that the driver reports are the same."""
    if not isinstance(record_device, dict) or not isinstance(device, dict):
        return False
    return all(str(record_device.get(k, "")).strip() == str(device.get(k, "")).strip() for k in ("platform", "name"))


def find_calibration(device: dict | None, case: str | None) -> tuple[dict | None, str | None]:
    """`(record, problem)`: the newest calibration of this device and case in `gpu-constants.json`, else None."""
    records, problem = load_calibrations()
    hits = [r for r in records if same_device(r.get("device"), device) and r.get("case") == case]
    return (max(hits, key=lambda r: str(r.get("time_utc", ""))) if hits else None), problem


def _flush(fd: int) -> None:
    """Flush a file to the drive: `fcntl.F_FULLFSYNC` where it exists (macOS: `fsync` does not flush the drive cache), else `os.fsync`.

    The same as `checkpoint._flush` (that module needs numpy, and this one stays light, so the 12 lines are copied).
    """
    try:
        import fcntl
        full = getattr(fcntl, "F_FULLFSYNC", None)
    except ImportError:   # no fcntl on this system
        full = None
    if full is not None:
        try:
            fcntl.fcntl(fd, full)
            return
        except OSError:
            pass   # this file system does not do it (for example a network drive): `fsync` is the next best
    os.fsync(fd)


def _flush_folder(path: Path) -> None:
    """After the rename: flush the folder, so that the new name is on the drive too. A system that cannot do it is skipped."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        _flush(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_calibration(record: dict) -> Path:
    """Add `record` to `gpu-constants.json` in the state folder (it replaces the record of the same device and case). Return the path.

    The write is the pattern of `checkpoint.write_checkpoint`: a temporary file with a name of its own in the same folder (two
    `gpu-check` runs do not write the same file), flushed to the drive, then `os.replace`, then a flush of the folder. A kill at any point
    leaves the old file or the new one, never a part of a file. Like `checkpoint.py`, it takes no lock: two runs at the same time can lose
    the record of the one that wrote first (the next `gpu-check` writes it again).

    Raises `OSError` when the folder cannot be written; the caller says so and goes on.
    """
    path = calibration_path()
    records, _ = load_calibrations()
    keep = [r for r in records if not (same_device(r.get("device"), record["device"]) and r.get("case") == record["case"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"schema": CALIBRATION_SCHEMA, "calibrations": keep + [record]}, indent=1)
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
    return path


def _median(values: list[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def _device_iterations(row: dict) -> int:
    """The PCG iterations of one optimizer iteration summed over its blocks: `device_iterations`, or from `iterations` (a bench file of an older version)."""
    if row.get("device_iterations") is not None:
        return int(row["device_iterations"])
    return sum(int(its.get("executed", 0)) + int(sum(its.get("inner", []))) for its in row.get("iterations", []))


def calibration_from_bench(report: dict, topopt_version: str | None = None, source: str | None = None) -> tuple[dict | None, str | None]:
    """`(record, why_not)`: the constants of this device from the report of `tools/bench_gpu_mg.py` (the bench step of `gpu-check`).

    For each gpu-mg run that finished: `c` is the median over the optimizer iterations of `solve_s / (device iterations * free dofs)`, the
    seconds of the device for one PCG iteration of one block and one free dof; `host` is `(median seconds of an iteration - median
    `solve_s`) / free dofs`. The columns for each launch are 2 when the problem has more than one load case, else 1
    (`GpuParams.nr_block`). The sizes are combined by their free dofs. A run needs `n_load_cases` (the bench writes it since P2a).
    """
    dev = (report.get("opencl") or {}).get("device")
    if not dev:
        return None, "the bench report names no OpenCL device"
    points = []
    for run in report.get("runs", []):
        g = run.get("gpu_mg")
        if run.get("solver") != "gpu-mg" or run.get("status") != "ok" or not g or not run.get("free_dofs"):
            continue
        if not run.get("n_load_cases"):
            return None, "the bench report has no load-case count (a bench of an older version): run `topopt gpu-check` again"
        n = int(run["free_dofs"])
        per = [r["solve_s"] / (its * n) for r in g.get("per_iteration", []) if (its := _device_iterations(r))]
        if not per or not run.get("median_iter_s"):
            continue
        host = max(0.0, run["median_iter_s"] - g["median_s"]["solve_s"]) / n
        points.append({"voxel_mm": run["h"], "free_dofs": n, "n_load_cases": int(run["n_load_cases"]),
                       "c_s": _median(per), "host_s_per_dof": host})
    if not points:
        return None, "no gpu-mg run of the bench finished, so there is nothing to calibrate"
    cols = {2 if p["n_load_cases"] > 1 else 1 for p in points}
    if len(cols) > 1:
        return None, "the gpu-mg runs of the bench have different columns for each launch"
    weight = sum(p["free_dofs"] for p in points)
    case = next((r["gpu_mg"]["case"] for r in report["runs"] if r.get("solver") == "gpu-mg" and r.get("gpu_mg")), None)
    record = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topopt_version": topopt_version,
        "device": {k: dev.get(k) for k in ("platform", "name", "vendor", "kind", "fp64")},
        "case": case,
        "columns_per_launch": cols.pop(),
        "c_s_per_dof_iteration": sum(p["c_s"] * p["free_dofs"] for p in points) / weight,
        "host_s_per_dof": sum(p["host_s_per_dof"] * p["free_dofs"] for p in points) / weight,
        "points": points,
        "source": source or "topopt gpu-check, bench step",
    }
    return record, None


# ---- the estimate --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeInputs:
    """What the model needs from the problem and the device. `from_config` makes it from a loaded problem."""

    n_cases: int = 1
    volume_fraction: float | None = None
    penal: float | None = None
    gpu_tol: float | None = None
    device: dict | None = None      # the facts of the chosen OpenCL device (`DeviceInfo.facts()`), or None
    case: str | None = None         # the precision case; None: from the device
    k_other: float = 0.0            # other gpu-mg runs on the same device at the same time

    @classmethod
    def from_config(cls, cfg, opencl: dict | None = None) -> "TimeInputs":
        """The inputs of a loaded `Config`; `opencl` is the `opencl` facts of `cli.opencl_facts` (None: the device is not known)."""
        st = cfg.settings
        dev = chosen_device(opencl)
        return cls(n_cases=max(1, len(cfg.load_cases)), volume_fraction=st.volume_fraction, penal=st.penal,
                   gpu_tol=st.gpu_tol, device=dev, case=device_case(dev, st.gpu_case))


def stiffness_contrast(volume_fraction: float | None, penal: float | None) -> float | None:
    """`lgK = -log10(1e-6 + volume_fraction ** penal)`: the decades between solid and a uniform start design. None when `penal` is not known."""
    if volume_fraction is None or penal is None or volume_fraction < 0:
        return None
    return -math.log10(VOID_STIFFNESS + volume_fraction ** penal)


def tol_factor(gpu_tol: float | None) -> float:
    """The PCG iterations at `gpu_tol` over the iterations of the bands: 1.0 for `gpu_tol` from 1e-8 to 1e-6 (the default 1e-6 too).

    Outside that range the rule of thumb of the pocl runs (see `TOL_SLOPE_PER_DECADE`): looser than 1e-6 it is
    `1 + 0.15 log10(1e-6 / gpu_tol)`, tighter than 1e-8 the same value over its value at 1e-8.
    """
    if gpu_tol is None:
        from ..config import GPU_TOL_DEFAULT
        gpu_tol = GPU_TOL_DEFAULT
    if BAND_TOL_TIGHT <= gpu_tol <= BAND_TOL_LOOSE:
        return 1.0
    at = lambda tol: 1.0 + TOL_SLOPE_PER_DECADE * math.log10(BAND_TOL_LOOSE / tol)      # noqa: E731
    return at(gpu_tol) if gpu_tol > BAND_TOL_LOOSE else at(gpu_tol) / at(BAND_TOL_TIGHT)


def iteration_band(lgk: float | None) -> tuple[int, int, int]:
    """The (low, mid, high) mean PCG iterations of one block over a run at `gpu_tol` 1e-8 to 1e-6, for the contrast `lgk`."""
    if lgk is None:
        return ITERATION_BAND_UNKNOWN
    return next(band for limit, band in ITERATION_BANDS if lgk < limit)


@dataclass(frozen=True)
class Constants:
    """The constants that one estimate used, and where they come from (`basis`: "calibration", "table", "reference", or "none" for a CPU device: no seconds)."""

    c1_s: float
    c2_s: float
    host_s_per_dof: float
    t0_s: float
    t0_s_per_dof: float
    share: float
    basis: str
    label: str
    source: str
    short: str
    problem: str | None = None


def _measured_sizes(record: dict) -> tuple[int, int] | None:
    """The smallest and the largest free-dof count of the bench points of a calibration record, or None (a record with no usable points)."""
    points = record.get("points")
    sizes = [int(p["free_dofs"]) for p in points if isinstance(p, dict) and _number(p.get("free_dofs"), True)] if isinstance(points, list) else []
    return (min(sizes), max(sizes)) if sizes else None


def calibration_facts_text(record: dict, two: bool) -> str:
    """What the record measured and what it did not, in two sentences (P2a review m3)."""
    have, other = (2, 1) if two else (1, 2)
    derived = f"c / {TWO_COLUMN_RATIO}" if two else f"c * {TWO_COLUMN_RATIO}"
    sizes = _measured_sizes(record)
    where = f"{sizes[0]:,} to {sizes[1]:,} free dofs" if sizes and sizes[0] != sizes[1] else (f"{sizes[0]:,} free dofs" if sizes else "the sizes of the bench")
    return (f"c is measured for {have} column{'s' if have > 1 else ''} for each launch; the value for {other} column{'s' if other > 1 else ''} is "
            f"{derived}, the ratio of the Intel Arc iGPU fit, not measured on this device. The host part is measured at {where} only, and is used for every size")


def constants_for(device: dict | None, case: str | None) -> Constants:
    """The constants for a device and a case: the calibration of `gpu-constants.json`, else the table row, else the reference row.

    A record that the model cannot use never raises: `load_calibrations` drops a bad record, and this function falls back to the table
    or the reference row (with the text of the problem in `problem`) if a record still fails when it is used.
    """
    ref = TIME_CONSTANTS[REFERENCE_KEY]
    record, problem = find_calibration(device, case)
    if record is not None:
        try:
            c = float(record["c_s_per_dof_iteration"])
            host = float(record["host_s_per_dof"])
            two = int(record["columns_per_launch"]) == 2
            if not (math.isfinite(c) and c > 0 and math.isfinite(host) and host >= 0):
                raise ValueError("c or the host part is not a finite number")
            c1, c2 = (c / TWO_COLUMN_RATIO, c) if two else (c, c * TWO_COLUMN_RATIO)
            when = str(record.get("time_utc") or "an unknown time").replace("T", " ").replace("Z", " UTC")
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
            problem = ((problem + "; ") if problem else "") + (f"{calibration_path()}: the record of this device cannot be used "
                                                              f"({type(exc).__name__}: {exc}); it is ignored")
        else:
            facts = calibration_facts_text(record, two)
            return Constants(c1, c2, host, ref.t0_s, ref.t0_s_per_dof, ref.share, "calibration",
                             f"{device_name(device)}, case {case}", f"measured on this machine by `topopt gpu-check` on {when} ({calibration_path()}). {facts}. "
                             f"The start and end of the job and the sharing are those of the {ref.label}",
                             f"measured by `topopt gpu-check` on {when} ({calibration_path()}). {facts}", problem)
    key = device_class(device, case)
    if key is not None:
        row = TIME_CONSTANTS[key]
        return Constants(row.c1_s, row.c2_s, row.host_s_per_dof, row.t0_s, row.t0_s_per_dof, row.share, "table", row.label, row.source, row.short, problem)
    # A CPU OpenCL device (pocl) is not a GPU: the constants of a GPU are 2.5 to 5 times too low for it (the pocl bench of the repo, hook 1.2 mm),
    # so it gets no number at all (P2a review M2). The numbers of `ref` are in the record only because the type needs them; `basis` "none" says
    # that `estimate_seconds_gpu_mg` does not use them.
    basis = "none" if device and device.get("kind") == "CPU" else "reference"
    return Constants(ref.c1_s, ref.c2_s, ref.host_s_per_dof, ref.t0_s, ref.t0_s_per_dof, ref.share, basis, ref.label, ref.source, ref.short, problem)


@dataclass(frozen=True)
class GpuEstimate:
    """The result of `estimate_seconds_gpu_mg`: the middle number, the band, and what it assumed."""

    s_per_iteration: float | None   # one optimizer pass, mid band; None when `basis` is "none" (a CPU device: no seconds)
    total_s: float | None           # mid band; None as above
    low_s: float | None
    high_s: float | None
    iterations_per_block: tuple     # (low, mid, high), after the gpu_tol factor
    contrast_lgk: float | None
    tol_factor: float
    gpu_tol: float
    blocks: int
    columns_per_launch: int
    n_free: int
    device_name: str
    case: str | None
    constants: Constants
    k_other: float = 0.0
    assumptions: list = field(default_factory=list)
    n_cases: int = 1                # the inputs of the estimate, kept for `time_model` (P2a review M3)
    volume_fraction: float | None = None
    penal: float | None = None

    @property
    def basis(self) -> str:
        return self.constants.basis

    @property
    def label(self) -> str:
        """The basis in a few words, for the first line of `check` and for `derived.time_model`."""
        return basis_label(self.constants.basis, self.constants.label)

    def time_model(self) -> dict:
        """What the seconds of `derived.time_estimate_s` depend on: the basis of the constants and the inputs of the model."""
        c = self.constants
        return {"model": "gpu-mg", "basis": c.basis, "label": self.label, "device": self.device_name, "case": self.case,
                "penal": self.penal, "volume_fraction": self.volume_fraction,
                "lgK": None if self.contrast_lgk is None else round(self.contrast_lgk, 2),
                "gpu_tol": self.gpu_tol, "n_cases": self.n_cases, "constants_source": c.short}

    def as_dict(self) -> dict:
        c = self.constants
        whole = lambda v: None if v is None else round(v)      # noqa: E731  (whole seconds, rounded once; None for a device with no constants)
        return {
            "model": "gpu-mg", "total_s": whole(self.total_s), "low_s": whole(self.low_s), "high_s": whole(self.high_s),
            "s_per_iteration": None if self.s_per_iteration is None else round(self.s_per_iteration, 4),
            "iterations_per_block": {"low": round(self.iterations_per_block[0], 1), "mid": round(self.iterations_per_block[1], 1),
                                    "high": round(self.iterations_per_block[2], 1)},
            "contrast_lgk": None if self.contrast_lgk is None else round(self.contrast_lgk, 2),
            "gpu_tol": self.gpu_tol, "gpu_tol_factor": round(self.tol_factor, 3), "blocks": self.blocks,
            "columns_per_launch": self.columns_per_launch, "free_dofs": self.n_free, "other_gpu_runs": self.k_other,
            "device": self.device_name, "case": self.case, "n_cases": self.n_cases, "penal": self.penal, "volume_fraction": self.volume_fraction,
            "basis": c.basis, "label": self.label, "constants": {"for": c.label, "source": c.source, "source_short": c.short, "c_one_column_ns": round(c.c1_s * 1e9, 2),
                                            "c_two_columns_ns": round(c.c2_s * 1e9, 2), "host_us_per_dof": round(c.host_s_per_dof * 1e6, 2),
                                            "start_end_s": c.t0_s, "start_end_us_per_dof": round(c.t0_s_per_dof * 1e6, 1),
                                            "calibration_problem": c.problem},
            "assumptions": list(self.assumptions)}


def estimate_seconds_gpu_mg(n_free: int, n_cases: int, iterations: int, *, volume_fraction: float | None = None,
                            penal: float | None = None, gpu_tol: float | None = None, device: dict | None = None,
                            case: str | None = None, k_other: float = 0.0, robust_factor: float = 1.0,
                            extra_s_per_pass=None, constants: Constants | None = None) -> GpuEstimate:
    """The run time of a gpu-mg run (the model of this module's text): the mid number and the low and high band.

    `n_free` free dofs, `n_cases` load cases, `iterations` optimizer iterations. `volume_fraction` and `penal` give the stiffness
    contrast of the start design (without `penal` the band of all runs is used), `gpu_tol` (default `settings.gpu_tol`) the factor of
    the tolerance. `device` and `case` choose the constants (see `constants_for`); `constants` replaces them. `k_other` is the number of
    other gpu-mg runs on the same device (0 in `check`). `robust_factor` multiplies the passes (`robust.time_factor`); `extra_s_per_pass`
    is a function of the seconds of one pass that returns the extra seconds (the stress rows of `cli.stress_seconds`).
    """
    case = case or device_case(device)
    k = constants or constants_for(device, case)
    blocks, nr = (1, 1) if n_cases <= 1 else (math.ceil(n_cases / 2), 2)
    if gpu_tol is None:
        from ..config import GPU_TOL_DEFAULT
        gpu_tol = GPU_TOL_DEFAULT
    lgk = stiffness_contrast(volume_fraction, penal)
    factor = tol_factor(gpu_tol)
    band = tuple(i * factor for i in iteration_band(lgk))
    c_n = (k.c1_s if nr == 1 else k.c2_s) * n_free * (1.0 + k.share * k_other)
    t0 = k.t0_s + k.t0_s_per_dof * n_free

    def run(its: float) -> tuple[float, float]:
        per_pass = blocks * its * c_n + k.host_s_per_dof * n_free
        if extra_s_per_pass is not None:
            per_pass += extra_s_per_pass(per_pass)
        return per_pass, t0 + per_pass * (iterations + 1) * robust_factor

    if k.basis == "none":       # a device with no constants and no reference (a CPU device): no seconds
        per_mid = mid_s = lo_s = hi_s = None
    else:
        (_, lo_s), (per_mid, mid_s), (_, hi_s) = run(band[0]), run(band[1]), run(band[2])
    name = device_name(device)
    assumptions = [f"device: {name}" + (f", case {case}" if case else ""),
                   f"{k_other:g} other gpu-mg runs on the device at the same time" if k_other else "no other gpu-mg run on the device at the same time",
                   (f"stiffness contrast of the start design {lgk:.1f} decades (volume_fraction {volume_fraction:g}, penal {penal:g})" if lgk is not None
                    else "stiffness contrast not known (no penal): the band of all runs"),
                   f"PCG iterations for each block {band[0]:.0f} / {band[1]:.0f} / {band[2]:.0f} (low / mid / high; the count depends on how hard "
                   f"the design gets: 22 to about 2,000 were seen), gpu_tol {gpu_tol:g}",
                   (f"gpu_tol {gpu_tol:g} is from {BAND_TOL_TIGHT:g} to {BAND_TOL_LOOSE:g}, where the bands hold: no factor"
                    if factor == 1.0 else
                    f"gpu_tol {gpu_tol:g} is outside {BAND_TOL_TIGHT:g} to {BAND_TOL_LOOSE:g}: the iterations are scaled by {factor:.2f}, a rule of thumb "
                    f"(15 percent for each decade) from pocl runs on synthetic designs, not from real runs"),
                   f"the band is the spread of the iteration counts of the fit runs, not a limit: of the {FIT_RUNS} runs of the fit the time was "
                   f"inside the band for {FIT_INSIDE}, below it for {FIT_BELOW} and above it for {FIT_ABOVE} ({FIT_ABOVE_SHARED} of those shared the GPU)"]
    assumptions.append("the springs and joints kernels of gpu-mg, the first-pass cap of 1000 iterations (P2b) and per-axis supports (P2c) are not in the fit")
    if extra_s_per_pass is not None or robust_factor != 1.0:
        assumptions.append("the extra solves of stress rows and robust mode are added with the model of the other solvers: the fit does not hold them")
    if k.basis == "none":
        assumptions.append("no seconds are given for this device: it is a CPU device, and the constants of a GPU do not fit it")
    if case == "C":
        assumptions.append("the iteration bands are from case B runs")
    return GpuEstimate(per_mid, mid_s, lo_s, hi_s, band, lgk, factor, gpu_tol, blocks, nr, int(n_free), name, case, k, k_other, assumptions,
                       n_cases=max(1, int(n_cases)), volume_fraction=volume_fraction, penal=penal)


def fixed_seconds() -> float:
    """The part of the model time that does not grow with the free dofs (for `cli.fit_voxel_size`)."""
    return TIME_CONSTANTS[REFERENCE_KEY].t0_s


def free_dofs_for_seconds(goal_s: float, iterations: int, inputs: "TimeInputs | None" = None, **kw) -> int | None:
    """The free dofs at which the mid estimate is `goal_s`: the inverse of the model (it is linear), for `regions_builder.target_active`.

    None when the device has no constants and no reference (a CPU device): the model has no seconds to invert.
    """
    inputs = inputs or TimeInputs()
    args = dict(volume_fraction=inputs.volume_fraction, penal=inputs.penal, gpu_tol=inputs.gpu_tol, device=inputs.device, case=inputs.case,
                k_other=inputs.k_other, **kw)
    probe = 1_000_000
    e = estimate_seconds_gpu_mg(probe, inputs.n_cases, iterations, **args)
    if e.total_s is None:
        return None
    slope = (e.total_s - fixed_seconds()) / probe
    return max(0, int((goal_s - fixed_seconds()) / slope))


# ---- the check -----------------------------------------------------------------------------------------------------


def estimate_memory(free_dofs: int, grid_voxels: int = 0, case: str | None = None, joints: int = 0,
                    filter_held_bytes: int = 0, filter_build_bytes: int = 0) -> dict:
    """The memory model of gpu-mg (see the constants above): the peak RSS of a `topopt run` from the free dofs, the voxels of the grid and the filter.

    `case` is the precision case of the device ("B", "C" or None when the device is not known); case C (and an unknown case) with `joints` adds the columns
    of the deflation to the loop phase. `filter_held_bytes` and `filter_build_bytes` are `filter_held_bytes` and `filter_build_peak_bytes` of the check
    (`members.filter_size`: the FFT filter that the run holds, and its peak while it is set up); 0 when they are not known.
    The phases: `build` = base + filter build peak; `loop` = base + per free dof (+ joints) + filter held; `mesh` = the mesh step of the post-process.
    Returns `bytes` (the largest phase, with the layout factor), `typical_bytes` (the same with layout factor 1), `build_bytes`, `loop_bytes`,
    `mesh_bytes` (with the layout factor) and `phase` (`"build"`, `"loop"` or `"mesh"`: the phase that sets `bytes`; a tie goes to loop, then mesh).
    """
    n = max(0, int(free_dofs))
    held, built = max(0, int(filter_held_bytes)), max(0, int(filter_build_bytes))
    build = GPU_MG_BASE_BYTES + built
    loop = GPU_MG_BASE_BYTES + GPU_MG_BYTES_PER_DOF * n + held
    if joints and case != "B":
        loop += GPU_MG_JOINT_BYTES_PER_DOF * n
    mesh_fixed = GPU_MG_MESH_BASE_BYTES + GPU_MG_MESH_BYTES_PER_GRID_VOXEL * max(0, int(grid_voxels))
    surface = GPU_MG_MESH_BYTES_PER_DOF23 * n ** (2 / 3)
    mesh = mesh_fixed + GPU_MG_LAYOUT_FACTOR * surface
    phase = max((("loop", loop), ("mesh", mesh), ("build", build)), key=lambda item: item[1])[0]
    return {"bytes": int(max(build, loop, mesh)), "typical_bytes": int(max(build, loop, mesh_fixed + surface)),
            "build_bytes": int(build), "loop_bytes": int(loop), "mesh_bytes": int(mesh), "phase": phase}


def estimate_memory_bytes(free_dofs: int, grid_voxels: int = 0, case: str | None = None, joints: int = 0,
                          filter_held_bytes: int = 0, filter_build_bytes: int = 0) -> int:
    """Peak memory of a `topopt run` with solver gpu-mg in bytes (`estimate_memory(...)["bytes"]`): measured on pocl (host memory)."""
    return estimate_memory(free_dofs, grid_voxels, case, joints, filter_held_bytes, filter_build_bytes)["bytes"]


def problem_memory(free_dofs: int, grid_voxels: int, case: str | None, joints: int, fsz: dict | None) -> dict:
    """`estimate_memory` for a problem, with the filter numbers of `members.filter_size` (`fsz`; None: not known) and without them (`without_filter`).

    Returns the dict of `estimate_memory` plus `without_filter`: the `bytes` of the model with both filter terms at 0 (the number that
    `cli.memory_warnings` needs to tell if the filter is the cause).
    """
    fsz = fsz or {}
    m = estimate_memory(free_dofs, grid_voxels, case, joints, int(fsz.get("held_bytes_estimate", 0)), int(fsz.get("build_peak_bytes_estimate", 0)))
    return {**m, "without_filter": estimate_memory_bytes(free_dofs, grid_voxels, case, joints)}


def basis_label(basis: str, label: str) -> str:
    """The basis of the constants in a few words (the first `time estimate` line of `check`, `derived.time_model.label`)."""
    return {"calibration": "constants of this machine", "table": f"{label} constants",
            "reference": f"reference: {label} constants", "none": "no constants for this device"}[basis]


def basis_text(g: dict) -> str:
    """One sentence that says where the constants of the estimate come from."""
    c = g["constants"]
    if g["basis"] == "calibration":
        text = f"constants of this machine, {c['source_short']}"
    elif g["basis"] == "table":
        text = f"constants of the {c['for']}, case {g['case']} ({c['source_short']})"
    elif g["basis"] == "none":
        text = (f"no constants for this device (CPU device); run `topopt gpu-check` to measure them. No seconds are given for it: "
                f"the constants of the {c['for']} are not for a CPU device ({g['device']})")
    else:
        where = f"({g['device']}, case {g['case'] or 'not known'})" if g["device"] != "not known" else "(the device is not known here)"
        text = (f"no constants for this device {where}; run `topopt gpu-check` to measure them. The numbers use the constants of the "
                f"{c['for']} as a reference ({c['source_short']}); this device can be faster or slower")
    return text + (f". Note: {c['calibration_problem']}" if c["calibration_problem"] else "")


def apply(d: dict) -> dict:
    """Change the dict of `cli.check_data` in place; return it.

    For any solver but gpu-mg the dict is not changed. For gpu-mg: the memory estimate becomes the gpu-mg model (the Cholesky number stays as
    `cholesky_bytes`), the time estimate gets `model: "gpu-mg"`, the band (`low_s`, `high_s`) and its assumptions (the
    time numbers are those of the gpu-mg model: `check_data` makes them), the warning about the Cholesky memory model is
    replaced, and one warning gives the time and memory facts of the gpu-mg models.
    """
    gpu = (d.get("solver") or {}).get("resolved") == "gpu-mg"
    if not gpu:
        return d
    from ..cli import gib     # late: cli imports this module only for a gpu-mg result
    me = d["memory_estimate"]
    ram, cholesky = me["ram_bytes"], me["bytes"]
    g = d["time_estimate"]["gpu_mg"]
    joints = len((d.get("extended") or {}).get("joints") or [])
    fsz = ((d.get("derived") or {}).get("filter") or {}).get("size")        # the FFT filter: its padded grid and bytes (`members.filter_size`)
    m = problem_memory(d["free_dofs"], (d.get("grid") or {}).get("voxels", 0), g.get("case"), joints, fsz)
    mem = m["bytes"]
    d["memory_estimate"] = {
        "bytes": mem, "typical_bytes": m["typical_bytes"], "build_bytes": m["build_bytes"], "loop_bytes": m["loop_bytes"], "mesh_bytes": m["mesh_bytes"],
        "phase": m["phase"],
        "ram_bytes": ram, "model": "gpu-mg", "cholesky_bytes": cholesky,
        **{k: me[k] for k in ("filter_build_peak_bytes", "filter_held_bytes") if k in me},   # the filter does not depend on the solver (A3a)
        "note": ("model of the peak memory of topopt run with gpu-mg, measured on pocl (a CPU OpenCL device: the device buffers are host memory): "
                 "the largest of the build phase (the filter is set up), the loop phase (with the filter held) and the mesh phase of "
                 "the post-process; `bytes` has the layout factor of the mesh phase, `typical_bytes` has none; cholesky_bytes is the model of the "
                 "Cholesky solver, for comparison")}
    d["time_estimate"] = {**d["time_estimate"], "model": "gpu-mg", "low_s": g["low_s"], "high_s": g["high_s"],
                          "basis": g["basis"], "note": basis_text(g)}
    warnings = [w for w in d["warnings"] if not w.startswith(_REPLACED_WARNINGS)]
    if ram and mem > ram:
        from ..cli import memory_warnings
        warnings.append(f"the memory estimate is {gib(mem)} and this machine has {gib(ram)} RAM: a run may swap or "
                        f"crash (the gpu-mg model, measured on pocl; for a layout like the hook example it is {gib(m['typical_bytes'])})")
        warnings += memory_warnings(mem, ram, fsz, m["without_filter"])[1:]       # the filter line with its options, when the filter is the cause
    estimate = (f"the time estimate is about {g['total_s']:.0f} s, in a band of {g['low_s']:.0f} to {g['high_s']:.0f} s: a model that is linear "
                f"in the free dofs" if g["total_s"] is not None else
                "there is no time estimate for this device: the model is linear in the free dofs, and it has no constants for a CPU device")
    warnings.append(
        f"gpu-mg models: {estimate}. Constants: {basis_text(g)}. It assumes: " + "; ".join(g["assumptions"]) + ". "
        f"The memory estimate is the gpu-mg model, the largest of three phases ({gib(m['build_bytes'])} while the filter is set up, {gib(m['loop_bytes'])} "
        f"for the loop and {gib(m['mesh_bytes'])} for the mesh step of the post-process, with a layout factor of {GPU_MG_LAYOUT_FACTOR:g} on its "
        f"surface term; {gib(m['typical_bytes'])} without the factor, as for the hook example), "
        "measured on pocl, where the device buffers are host memory (about 0.3 kB for each free dof). A GPU "
        "with its own memory holds the device buffers on the device")
    d["warnings"] = warnings
    return d


def lines(d: dict) -> list[str]:
    """Lines for `topopt check` after the memory line, for a gpu-mg result."""
    from ..cli import gib
    me, te = d["memory_estimate"], d["time_estimate"]
    g = te["gpu_mg"]
    it = g["iterations_per_block"]
    band = (f"band {g['low_s']:.0f} to {g['high_s']:.0f} s around the number above " if g["total_s"] is not None
            else "no seconds for this device ")
    return [f"memory model:   gpu-mg, measured on pocl with the device buffers in host memory; the cholesky model for "
            f"comparison: about {gib(me['cholesky_bytes'])} (gpu-mg builds no factor). The peak is the largest of the filter setup "
            f"({gib(me['build_bytes'])}), the loop phase ({gib(me['loop_bytes'])}, with the filter) and the mesh step of the post-process "
            f"({gib(me['mesh_bytes'])}, set by the layout of the regions; {gib(me['typical_bytes'])} for a layout like the hook example)",
            f"time model:     gpu-mg, linear in the free dofs: {band}"
            f"(PCG iterations for each block {it['low']:.0f} / {it['mid']:.0f} / {it['high']:.0f}"
            + (f", stiffness contrast {g['contrast_lgk']:.1f} decades" if g["contrast_lgk"] is not None else ", stiffness contrast not known")
            + f"; {g['other_gpu_runs']:g} other gpu-mg runs)",
            f"time constants: {basis_text(g)}"]
