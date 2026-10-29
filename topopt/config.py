"""Read and check the problem YAML (CONTRACTS.md section 2).

`load_config(path)` returns a frozen `Config`. Every user error raises
`UserError` with one clear line.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from . import materials_catalog, printers
from .members import radius_from_min_member, wall_radius
from .robust import ETA_NOMINAL, MODES as ROBUST_MODES, WALL_ETA_DEFAULT
from .symmetry import ALIGNS, MAX_PLANES, MIRROR_HELP, axis_of_normal


class UserError(Exception):
    """An error the user can fix (bad YAML, bad file, guard). One line."""


class PrinterNotFound(UserError):
    """`print.printer_model` names no printer that can be found now (not in the catalog, not in the user printer file, ambiguous, or the
    catalog cannot be read). The text is not a fault of the value itself: the preferences file keeps such a key and does not use it."""


class FilamentNotFound(UserError):
    """`material.filament` names no filament that can be found now (not in the catalog, not in the user filament file, ambiguous, or the
    catalog cannot be read). The text is not a fault of the value itself: the preferences file keeps such a key and does not use it."""


# The values of `settings.solver`. "auto" runs gpu-mg on a GPU that passes the self-test and the CPU rule elsewhere
# (CONTRACTS.md section 20).
SOLVER_VALUES = ("auto", "cholesky", "pardiso", "amg", "gpu-mg")
GPU_CASES = ("auto", "B", "C")
OBJECTIVE_SCALES = ("stage", "first")   # settings.objective_scale (CONTRACTS.md section 13)
OBJECTIVE_SCALE_DEFAULT = "stage"       # the default of the field. `checkpoint.effective_settings` leaves the key out at this value (a checkpoint of a build before P3a resumes)
# The three numbers of gpu-mg that are settings (G2 and P2b, CONTRACTS.md section 20). Each default is also the default of the field.
GPU_TOL_DEFAULT = 1e-6              # settings.gpu_tol: relative residual of each solve in the optimizer loop (0 < value < 1)
GPU_BACKWARD_ERROR_DEFAULT = 1e-15  # settings.gpu_backward_error: normwise backward error that accepts a solve (0 <= value < the limit below). The floor of eta is 3.5e-17 to 4.4e-17 on the measured designs without joints; with joints the host check has a noise of its own (CONTRACTS.md section 20)
GPU_BACKWARD_ERROR_LIMIT = 1e-6     # settings.gpu_backward_error must be below this (a backward error of 1e-6 is not a rounding floor)
GPU_MAX_ITER_DEFAULT = 1000         # settings.gpu_max_iter: PCG iterations of the first pass of a gpu-mg solve (an integer, 1 or more). A restart step keeps its own cap


@dataclass(frozen=True)
class LoadRegion:
    name: str
    path: Path


ALL_AXES = (0, 1, 2)   # a full clamp fixes x, y and z


@dataclass(frozen=True)
class Regions:
    design: Path
    fixed: tuple[Path, ...]
    keep_out: tuple[Path, ...]
    load: tuple[LoadRegion, ...]  # in YAML order
    solid: tuple[Path, ...] = ()  # frozen at density 1, no clamp, no load (CONTRACTS.md section 22a)
    # Axes (0, 1, 2 for x, y, z) that each `fixed` entry fixes, parallel to `fixed`. `()` means that
    # every entry is a full clamp (no `{file, axes}` entry in the YAML). CONTRACTS.md section 26.2.
    fixed_axes: tuple[tuple[int, ...], ...] = ()
    measure: tuple["MeasureRegion", ...] = ()   # A5: named probe regions (a node set, no label)


# --- regions.not_printed (CONTRACTS.md section 9.1) --------------------------------------------------
NOT_PRINTED_ROLES = ("fixed", "load", "solid", "keep_out")


@dataclass(frozen=True)
class NotPrinted:
    """A region file that `regions.not_printed` names: a body that is in the problem but not in the print.

    `role` is the region kind that holds the file (`fixed`, `load`, `solid` or `keep_out`). `name` is the file name, or the
    region name for a `load` region. `path` is the resolved path, the key of the match. A file that two kinds hold gives two
    entries.
    """
    role: str
    name: str
    path: Path

    @property
    def label(self) -> str:
        """`<role>:<name>`, the form that `check_regions` and `metrics.json` use, so that no name hides another."""
        return f"{self.role}:{self.name}"


@dataclass(frozen=True)
class LoadForce:
    region: str
    force: tuple[float, float, float]  # N. (0, 0, 0) for a pure moment.
    # CONTRACTS.md section 26.2. None: the key is not in the YAML.
    moment: tuple[float, float, float] | None = None   # N*mm, about the node centroid of the region
    at: tuple[float, float, float] | None = None       # mm, the point where `force` acts (a remote force)


@dataclass(frozen=True)
class PressureLoad:
    region: str          # a key of regions.load
    value_mpa: float     # N/mm2. Positive pushes into the body, negative pulls (CONTRACTS.md section 26.2)


@dataclass(frozen=True)
class SpringSpec:
    """An elastic support on a load region (CONTRACTS.md section 35). Exactly one of the two stiffness fields is set."""

    region: str                                      # a key of regions.load
    stiffness: tuple[float, float, float] | None = None   # N/mm, the total for the region, global axes x, y, z
    per_area: tuple[float, float, float] | None = None    # N/mm3, on each exposed face


# Keys of each beam section kind of a joint (CONTRACTS.md section 38), and the zero-vector test of a direction.
SECTION_KEYS = {"round": ("d",), "tube": ("od", "id"), "rect": ("b", "h"), "props": ("A", "Iy", "Iz", "J")}
DEGENERATE = 1e-9    # a vector shorter than this is the zero vector (not a physical limit): the one test of `_vec3`, the build direction, the interface vectors and the joints


@dataclass(frozen=True)
class SectionSpec:
    """The beam section of a joint: `kind` and the values in the order of `SECTION_KEYS[kind]` (mm, mm2, mm4)."""

    kind: str
    values: tuple[float, ...]

    def as_dict(self) -> dict:
        return {"kind": self.kind, **dict(zip(SECTION_KEYS[self.kind], self.values))}


@dataclass(frozen=True)
class JointSpec:
    """A beam from a load region to a fixed far point (CONTRACTS.md section 38)."""

    region: str                                       # a key of regions.load
    to: tuple[float, float, float]                    # mm, the fixed far point
    section: SectionSpec
    origin: tuple[float, float, float] | None = None  # mm, the reference point; None: the area centroid of the coupled faces
    orient: tuple[float, float, float] | None = None  # the section y axis (unit vector); None: the global axis least parallel to the beam
    bond_stiffness_per_area: float | None = None      # N/mm3; None: c E / h
    E: float | None = None                            # MPa, the beam; None: the run material
    nu: float | None = None


@dataclass(frozen=True)
class LoadCase:
    name: str
    weight: float
    forces: tuple[LoadForce, ...]
    compliance_limit: float | None = None  # N*mm. The case compliance must stay at or below it.
    pressures: tuple[PressureLoad, ...] = ()   # CONTRACTS.md section 26.2
    stress_limit: float | None = None      # MPa. The final limit. It takes no safety_factor (a loader error).
    safety_factor: float | None = None     # only with allowable_mpa: limit = allowable_mpa / safety_factor
    allowable_mpa: float | None = None     # MPa. Given by the user or the agent. No default, no preset.
    allowable_source: str | None = None    # free text: where the allowable comes from

    @property
    def stress_limit_mpa(self) -> float | None:
        """The limit of the stress row in MPa, or None (no stress row for this case).

        `stress_limit` is the final limit. Else `allowable_mpa / safety_factor` (factor 1 if only
        `allowable_mpa` is given). The loader refuses `safety_factor` together with `stress_limit`,
        and a `safety_factor` without `allowable_mpa`. With `stress_limit` and `allowable_mpa` (no factor),
        the limit is `stress_limit` and the report uses `allowable_mpa` for the achieved safety factor.
        """
        if self.stress_limit is not None:
            return self.stress_limit
        if self.allowable_mpa is not None:
            return self.allowable_mpa / (1.0 if self.safety_factor is None else self.safety_factor)
        return None

    @property
    def stress_limit_from(self) -> str | None:
        """"stress_limit" or "allowable_mpa / safety_factor", or None without a stress row."""
        if self.stress_limit is not None:
            return "stress_limit"
        return None if self.allowable_mpa is None else "allowable_mpa / safety_factor"

    @property
    def applies_load(self) -> bool:
        """False if the case applies no load (U21, CONTRACTS.md section 13): every component of every `force` and `moment` is exactly
        0.0, and no `pressures` entry has a value other than 0.0. No tolerance. `weight` is the share of the case in the objective
        and is never a load (the engine has no body force). A moment with a zero `force` is a load, so the sum of the nodal forces
        (`derived.total_magnitude_n`) is no test: it is 0.0 for a pure moment."""
        return (any(v != 0.0 for f in self.forces for v in (*f.force, *(f.moment or ())))
                or any(p.value_mpa != 0.0 for p in self.pressures))


@dataclass(frozen=True)
class Material:
    E: float = 3500.0
    nu: float = 0.35
    density: float = 1.24


# The material numbers that `material.filament` can fill, and the words for the source of a number (CONTRACTS.md section 18.8).
FILAMENT_KEYS = ("E", "nu", "density")
SOURCE_YAML, SOURCE_ASSUMED = "yaml", "assumed"
NU_ASSUMED_REASON = materials_catalog.NU_REASON


@dataclass(frozen=True)
class FilamentUse:
    """The filament of `material.filament` and where each material number came from.

    `keys` has one entry for each of `FILAMENT_KEYS`: `(key, value, source, reason)`. `source` is `yaml` (the YAML gave the key, and it wins),
    `filament:<id>` (the profile gave it) or `assumed` (the default of `Material`, or the median density of the family of the
    filament); `reason` is None or the words that say why.
    """
    profile: "materials_catalog.FilamentProfile"
    keys: tuple[tuple[str, float, str, str | None], ...] = ()

    def source(self, key: str) -> str | None:
        """The source of a material number, or None for a key that is not listed."""
        return next((s for k, _v, s, _r in self.keys if k == key), None)

    def to_dict(self) -> dict:
        """The form stored in run.json (key `filament`). A run without `material.filament` has no such key."""
        p = self.profile
        return {"id": p.id, "name": p.name, "origin": p.origin, "catalog_version": p.catalog_version or None,
                "facts": p.facts(), "profile_sha256": p.facts_sha256(),     # two runs can be compared: the same digest, the same filament facts
                "material": {k: {"value": v, "source": s, **({"reason": r} if r else {})} for k, v, s, r in self.keys}}

    def check_facts(self, printer=None) -> dict:
        """The `material` key of `topopt check` and `check_problem`: the filament, where each number comes from, and (with a printer
        profile) the answer of `topopt.compat.check`. A fact for a person: it never stops a run."""
        from . import compat      # lazy: compat is a leaf, and a run without a filament never loads it

        p = self.profile
        return {"filament": p.id, "name": p.name, "origin": p.origin, "profile_from": p.origin_text, "catalog_version": p.catalog_version or None,
                "sheet_version": p.sheet_version or None, "tag": p.tag, "source_url": p.source_url, "profile_sha256": p.facts_sha256(),
                "material": self.to_dict()["material"], "printer": printer.id if printer is not None else None,
                "compat": compat.check(printer, p).to_dict() if printer is not None else None,
                **({"superseded_by": p.superseded_by} if p.superseded_by else {})}       # an older sheet: the id of the newer row

    @classmethod
    def from_dict(cls, d) -> "FilamentUse | None":
        """Read the `filament` key of run.json. A bad value reads as not stated; None if the key is no filament record."""
        if not isinstance(d, dict):
            return None
        profile = materials_catalog.FilamentProfile.from_facts(d.get("id"), d.get("catalog_version"), d.get("facts"))
        if profile is None:
            return None
        raw = d.get("material")
        raw = raw if isinstance(raw, dict) else {}
        keys = []
        for k in FILAMENT_KEYS:
            e = raw.get(k)
            if isinstance(e, dict) and isinstance(e.get("value"), (int, float)) and not isinstance(e["value"], bool) \
                    and isinstance(e.get("source"), str):
                reason = e.get("reason")
                keys.append((k, float(e["value"]), e["source"], reason if isinstance(reason, str) else None))
        return cls(profile, tuple(keys))


@dataclass(frozen=True)
class StopRules:
    """Thresholds of the convergence verdict and of the stop rules (CONTRACTS.md section 24.2).

    The verdict uses the thresholds in every run. The run stops at a rule only when `enabled` is true
    (`settings.stop_rules`, or `topopt run --stop-rules`). Defaults are not tuned for one part.
    """
    enabled: bool = False
    objective_change: float = 1e-4   # relative change of the objective between two passes
    window: int = 5                  # passes in a row that must stay at or below `objective_change`
    grayness_max: float = 0.1        # mean(4 rho (1 - rho)) at the last pass
    time_budget_s: float | None = None   # stop at the first pass that ends after this many seconds of run time


# The tolerance of the plateau stop `settled` of a fine pass (#318, CONTRACTS.md section 24.6): the largest relative change of the objective
# between two passes in the last `window` passes at one beta (`verdict.measures.objective_change`; the window is `settings.stop_rules.window`, 5 by
# default). A fine pass ends when this change is at or below the tolerance and the constraints are met. This is the one default in code; the
# YAML key `settings.fine_pass.objective_change` and `--fine-tol` (MCP `fine_tol`) override it.
# Evidence (offline replay of the history of two measured parts, window 5; compliance against the run from zero at the fine voxel; measured
# 2026-10-26, AMG solver, shared Linux host): a 5-inch drone arm at 0.7 mm, beta 8 held: 1e-3 stops at pass 21 with the weighted compliance
# 599.72, +0.46 % against 596.95 of the 80 pass run from zero; 5e-4 stops at pass 30; 1e-4 does not stop in 30 passes. examples/hook/hook_short.yaml
# at 1.2 mm, beta 8 held: 1e-3 stops at pass 15 (1.39 against 1.36 of the 40 pass run from zero); with beta 4 to 8, 1e-3 does not stop in 16 passes
# (the beta step restarts the window), so the maximum ends it. The tolerance is a default for the first fine passes of a part, not a limit.
FINE_OBJECTIVE_CHANGE = 1e-3
FINE_PASS_KEYS = ("objective_change",)
FINE_PASS_EXAMPLE = "settings: {fine_pass: {objective_change: 0.001}}"


@dataclass(frozen=True)
class FinePassSettings:
    """`settings.fine_pass` (CONTRACTS.md section 24.6): the settings of a fine pass made from this YAML (`--fine-from`, `--fine-pass`).

    `objective_change` is None when the YAML does not give it: a fine pass then uses `FINE_OBJECTIVE_CHANGE`. `--fine-tol` wins over both.
    The key has no effect on the passes of the run that reads this YAML.
    """
    objective_change: float | None = None


ENVELOPE_FILL_VALUES = ("auto", "on", "off")
ENVELOPE_FILL_DEFAULT = "off"   # the one place that spells the default of settings.envelope_fill (F4-55); one of ENVELOPE_FILL_VALUES


@dataclass(frozen=True)
class Settings:
    voxel_size: float
    # The mean density of the FREE design voxels. With `target_mass_g` it is None until `topopt.problem.build_problem` derives it from the voxel
    # grid (CONTRACTS.md section 3); `Problem.config.settings.volume_fraction` is then the derived number. Read it from `problem.config`.
    volume_fraction: float | None
    iterations: int
    filter_radius: float
    penal: float = 3.0
    beta_schedule: tuple[tuple[int, float], ...] = ((0, 1.0), (30, 2.0), (50, 4.0), (65, 8.0))
    move_limit: float = 0.2
    solver: str = "auto"
    case_combine: str = "sum"   # sum | max: how the load cases join in the objective (CONTRACTS.md section 13)
    max_active_voxels: int = 100000
    max_grid_voxels: int = 5000000
    objective: str = "compliance"   # compliance | mass (CONTRACTS.md section 17). mass: `volume_fraction` is the start density.
    stress_q: float = 40.0          # KS sharpness of the stress rows
    stress_power: float = 0.5       # relaxation power s of the stress rows
    stress_scale: float | str = "auto"   # KS scale c: "auto" (zmax / KS at each beta stage start) or a number
    stress_step_acceptance: bool = True  # reject an MMA step that makes a stress row worse (CONTRACTS.md section 17)
    stop_rules: StopRules = StopRules()  # verdict thresholds and optional stop rules (CONTRACTS.md section 24.2)
    # A3a (CONTRACTS.md section 22a): the filter at frozen voxels, and the minimum member size.
    filter_solid_neighbours: str = "density_one"   # density_one | ignore
    min_member_mm: float | None = None   # derives filter_radius, unless filter_radius is set
    filter_radius_source: str = "default"   # default | settings | min_member_mm | wall (where filter_radius came from)
    # A3b (CONTRACTS.md section 22b): the robust erosion form. Volume on the nominal design, compliance on the eroded one.
    robust: str = "off"   # off | eroded | wall (wall: the eroded form with the wall of print.min_wall_mm, a void-padded filter and eta_e 0.7; section 22b)
    eta_eroded: float | None = None   # threshold of the eroded design, 0.5 < eta < 1. None: robust.ETA_ERODED_DEFAULT (0.55; 0.7 with robust: wall)
    wall_mm: float | None = None   # robust: wall only: the wall w in mm, a copy of print.min_wall_mm. A record: not in the effective settings (the radius and eta_e are)
    mesh_method: str = "sdf"    # sdf | blur: how the final mesh is smoothed (CONTRACTS.md section 19)
    envelope_fill: str = ENVELOPE_FILL_DEFAULT  # off | auto | on (default off, see ENVELOPE_FILL_DEFAULT). sdf only: grow the sharp field to the design space surface before the smoothing (CONTRACTS.md section 19)
    gpu_device: str = "auto"   # solver gpu-mg only: "auto", the number of the device, or a part of its name
    gpu_case: str = "auto"     # solver gpu-mg only: "auto", "B" or "C" (CONTRACTS.md section 20)
    gpu_tol: float = GPU_TOL_DEFAULT   # solver gpu-mg only: relative residual of each solve in the loop (G2, CONTRACTS.md section 20)
    gpu_backward_error: float = GPU_BACKWARD_ERROR_DEFAULT   # solver gpu-mg only: a solve at this backward error is accepted. 0: off (G2)
    gpu_max_iter: int = GPU_MAX_ITER_DEFAULT   # solver gpu-mg only: PCG iterations of the first pass of a solve (case B; P2b, CONTRACTS.md section 20)
    perimeter_max_mm2: float | None = None   # A4c (CONTRACTS.md section 26.3): bound on the voxel surface area P, mm2. None: no row
    objective_scale: str = OBJECTIVE_SCALE_DEFAULT   # stage | first: the objective is divided by its value at the first pass of each beta stage, or of the run (section 13)
    part_only_min_film_mm: float | None = None   # P4b-6 (CONTRACTS.md section 9.1): wall thickness in mm below which the film at a contact face of part_only.stl is cut off. None: off
    target_mass_g: float | None = None   # I7 (CONTRACTS.md section 3): the mass in g that the user asks for, a user requirement. It sets the allowed material: gives `volume_fraction` (exclusive with the key). The final mesh is a result (metrics.mass_asked_g, metrics.mass_final_g). None: off
    fine_pass: FinePassSettings = FinePassSettings()   # #318 (section 24.6): the settings of a fine pass made from this YAML; no effect on this run's passes


# --- mirror block (CONTRACTS.md section 26.1) -------------------------------
# The engine reads these through `topopt/symmetry.py`. The planes are checked here, at load time.


@dataclass(frozen=True)
class MirrorPlane:
    axis: str                 # x | y | z: the axis of the plane normal
    at_mm: float              # the coordinate of the plane on that axis, mm
    align: str = "auto"       # auto | face | centre: where the plane lies in the voxel grid (section 26.1)


# --- A5: probe regions and motion limits (CONTRACTS.md section 31) ---------


@dataclass(frozen=True)
class MeasureRegion:
    """A named probe region: an STL that gives a node set. It changes no label and carries no force."""
    name: str
    path: Path


@dataclass(frozen=True)
class DisplacementLimit:
    """Limit on the mean displacement of a region, in mm. One row for each name in `cases`."""
    name: str
    region: str                    # a regions.measure or a regions.load name
    limit_mm: float
    direction: tuple[float, float, float] | None   # unit vector; None: the norm of the mean vector
    sense: str                     # abs | max | min (max and min need a direction)
    cases: tuple[str, ...]
    price: float | None = None     # MMA price of a violation; None: the default of the row (1000)


@dataclass(frozen=True)
class AlignmentLimit:
    """Limit on the relative shift (mm) and tilt (rad) of two regions. One row for each limit and case."""
    name: str
    region_a: str
    region_b: str
    axis: tuple[float, float, float]   # unit vector; shift and tilt are measured across it
    shift_limit_mm: float | None
    tilt_limit_rad: float | None
    cases: tuple[str, ...]
    price: float | None = None


@dataclass(frozen=True)
class Limits:
    displacement: tuple[DisplacementLimit, ...] = ()
    alignment: tuple[AlignmentLimit, ...] = ()


BUCKLING_KS_Q = 40.0              # the KS sharpness of the buckling row when `buckling.ks_q` is not set (the pipeline value)


@dataclass(frozen=True)
class Buckling:
    """The linear buckling check of the final design (CONTRACTS.md section 33) and the buckling row (section 39). Both off by default.

    `check` is the check after the run. `min_factor` (a number above 0, no default) turns the row of the optimizer on: the lowest
    buckling load factor of each load case is at least `min_factor`. `ks_q` is the KS sharpness of the row (None: `BUCKLING_KS_Q`).
    `modes`, `max_seconds` and `max_gb` serve both.
    """
    check: bool = False
    modes: int = 6                # how many of the lowest load factors for each load case
    density: str = "printed"      # printed (the design at the threshold 0.5) | gray (the optimizer density). Of the check only
    max_seconds: float = 120.0    # time budget of the eigen solve of one load case
    max_gb: float | None = None   # memory budget of the check in GB. None: the RAM of this machine
    min_factor: float | None = None   # the row: lowest load factor, as a multiplier on the load case. None: no row
    ks_q: float | None = None         # the row: KS sharpness. None: BUCKLING_KS_Q


# --- print block (CONTRACTS.md section 18) ---------------------------------
# The print report (topopt/print_report.py) reads these. The optimizer ignores them.

DEFAULT_BED_MM = 220.0
DEFAULT_PRINTER_TYPE = "fdm"
DEFAULT_BUILD_DIRECTION = (0.0, 0.0, 1.0)
# Printer type -> (default self-supporting angle in degrees from the plate, support needed).
# A face is an overhang if its angle to the plate is below the angle. Sources: CONTRACTS.md section 18.
# `sls`: the powder bed supports the part, so no support is needed. The angle is only the reference
# for the overhang area that the report still gives.
PRINTER_TYPES = {
    "fdm": (45.0, True),
    "sla": (45.0, True),
    "lpbf": (45.0, True),
    "sls": (45.0, False),
}
# Limits of the print check (CONTRACTS.md section 32). A limit is a number that the YAML gives. Only two have a
# default; the report labels it "assumed". A limit with no default has no value until the YAML gives it:
# the check then reports the fact and its state is "unknown".
PRINT_LIMIT_KEYS = ("min_wall_mm", "min_wall_area_pct", "max_closed_void_mm3", "max_bodies",
                    "max_overhang_area_mm2", "max_support_volume_mm3", "min_plate_contact_mm2")
PRINT_LIMIT_DEFAULTS = {"max_bodies": 1, "min_wall_area_pct": 0.5}
PRINT_KEYS = ("build_direction", "bed_mm", "printer_type", "self_supporting_angle_deg", "printer_model") + PRINT_LIMIT_KEYS
# The keys that a printer profile fills when the YAML names a `printer_model` and does not give the key (CONTRACTS.md section 18.7).
# The source label of such a key is `profile:<id>`.
PRINT_DERIVABLE_KEYS = ("bed_mm", "printer_type")
_AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
_AXIS_NAME = re.compile(r"^([+-]?)([xyz])$")


def _limit_value_ok(key: str, v) -> bool:
    """The rule of `_print_block` for one limit key, as a yes or no (for a `run.json` that a person edited)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return False
    if key == "max_bodies":
        return isinstance(v, int) and v >= 1
    if key == "min_wall_area_pct":
        return 0.0 <= v <= 100.0
    return v > 0.0 if key == "min_wall_mm" else v >= 0.0


@dataclass(frozen=True)
class PrintSettings:
    build_direction: tuple[float, float, float] = DEFAULT_BUILD_DIRECTION   # unit vector, up, away from the plate
    bed_mm: tuple[float, float] = (DEFAULT_BED_MM, DEFAULT_BED_MM)           # bed size in mm
    printer_type: str = DEFAULT_PRINTER_TYPE
    self_supporting_angle_deg: float | None = None   # None: the default of the printer type
    given: tuple[str, ...] = ()                      # the keys that the YAML gave; the others are assumed (or derived)
    limits: tuple[tuple[str, float], ...] = ()       # the limit keys that the YAML gave, as (key, value) pairs
    profile: "printers.PrinterProfile | None" = None  # the printer of `printer_model`, with its capability facts; None without a model
    derived: tuple[str, ...] = ()                    # the keys that the profile filled (a subset of PRINT_DERIVABLE_KEYS); their source is `profile:<id>`
    derived_from: tuple[tuple[str, str], ...] = ()   # (key, printer id) for a derived key that ANOTHER printer than `profile` filled (printcheck
    #                                                  joins the settings of a run.json and of a YAML that name two printers); never in run.json

    @property
    def printer_model(self) -> str | None:
        """The id of the printer profile, or None."""
        return self.profile.id if self.profile is not None else None

    @property
    def build_height_mm(self) -> float | None:
        """The build height of the printer profile, or None (no profile, or the catalog does not state it): the height is then not limited."""
        return self.profile.build_height_mm if self.profile is not None else None

    def key_source(self, key: str) -> str | None:
        """`profile:<id>` for a key that the profile filled, else None."""
        if self.profile is None or key not in self.derived:
            return None
        return f"profile:{dict(self.derived_from).get(key, self.profile.id)}"

    def limit(self, key: str):
        """The value of a limit key: the YAML value, else the default (`PRINT_LIMIT_DEFAULTS`), else None."""
        return dict(self.limits).get(key, PRINT_LIMIT_DEFAULTS.get(key))

    def limit_source(self, key: str) -> str:
        """`yaml` if the YAML gave the key, `assumed` if a default applies, else `none`."""
        if key in dict(self.limits):
            return "yaml"
        return "assumed" if key in PRINT_LIMIT_DEFAULTS else "none"

    @property
    def angle_deg(self) -> float:
        """The self-supporting angle that the report uses: the explicit value, else the printer default."""
        if self.self_supporting_angle_deg is not None:
            return float(self.self_supporting_angle_deg)
        return PRINTER_TYPES[self.printer_type][0]

    @property
    def support_needed(self) -> bool:
        return PRINTER_TYPES[self.printer_type][1]

    def to_dict(self) -> dict:
        """The form stored in run.json (key `print`)."""
        d = {"build_direction": list(self.build_direction), "bed_mm": list(self.bed_mm),
             "printer_type": self.printer_type,
             "self_supporting_angle_deg": self.self_supporting_angle_deg,
             "given": list(self.given)}
        if self.limits:       # only when the YAML has a limit key: a run without one keeps the old run.json
            d["limits"] = dict(self.limits)
        if self.profile is not None:   # only with a printer_model: a run without one keeps the old run.json
            d["printer_model"] = self.profile.id
            d["catalog_version"] = self.profile.catalog_version or None    # a user printer has no catalog version unless its file gives one
            d["profile"] = self.profile.facts()
            d["profile_sha256"] = self.profile.facts_sha256()   # two runs can be compared: the same digest, the same printer facts
            d["derived"] = list(self.derived)
        return d

    @classmethod
    def from_dict(cls, d) -> "PrintSettings":
        """Read the `print` key of run.json. A missing or bad value gives the default (an assumed value)."""
        if not isinstance(d, dict):
            return cls()
        try:
            bd = tuple(float(v) for v in d["build_direction"])
            n = math.sqrt(sum(v * v for v in bd))
            bd = tuple(v / n for v in bd) if len(bd) == 3 and n > DEGENERATE else DEFAULT_BUILD_DIRECTION
        except (KeyError, TypeError, ValueError):
            bd = DEFAULT_BUILD_DIRECTION
        try:
            bed = tuple(float(v) for v in d["bed_mm"])
            bed = bed if len(bed) == 2 and all(v > 0 for v in bed) else cls().bed_mm
        except (KeyError, TypeError, ValueError):
            bed = cls().bed_mm
        printer = d.get("printer_type")
        printer = printer if isinstance(printer, str) and printer in PRINTER_TYPES else DEFAULT_PRINTER_TYPE
        angle = d.get("self_supporting_angle_deg")
        angle = float(angle) if isinstance(angle, (int, float)) and not isinstance(angle, bool) \
            and 0.0 <= angle <= 90.0 else None
        given = d.get("given")
        given = tuple(k for k in given if k in PRINT_KEYS) if isinstance(given, list) else ()
        raw = d.get("limits")
        raw = raw if isinstance(raw, dict) else {}
        limits = tuple((k, raw[k]) for k in PRINT_LIMIT_KEYS if _limit_value_ok(k, raw.get(k)))
        profile = printers.PrinterProfile.from_facts(d.get("printer_model"), d.get("catalog_version"), d.get("profile"))
        derived = d.get("derived")
        derived = tuple(k for k in derived if k in PRINT_DERIVABLE_KEYS) if profile is not None and isinstance(derived, list) else ()
        return cls(bd, bed, printer, angle, given, limits, profile, derived)


# --- interfaces block (CONTRACTS.md section 23) ---------------------------
# Exact shapes (bore, boss, tube, plane face, STL solid) that the post step unites with the final mesh
# or cuts out of it. The optimizer sees them only as labels (topopt/interfaces.py, topopt/voxelize.py).

INTERFACE_KINDS = ("bore", "boss", "tube", "face", "stl")
INTERFACE_OPS = ("union", "cut")
_RUN_JSON_FIX = "Options: correct the key in run.json, or run the problem again (the run writes this key from the YAML)"
_KIND_OP = {"bore": "cut", "boss": "union", "tube": "union", "face": "cut"}   # `stl` takes its `op` from the YAML
_CYLINDER_KEYS = ("name", "kind", "point", "axis", "radius_mm", "length_mm")
INTERFACE_KEYS = {   # kind -> (required keys, optional keys)
    "bore": (_CYLINDER_KEYS, ("stock_mm", "full_surface")),
    "boss": (_CYLINDER_KEYS, ("stock_mm", "full_surface")),
    "tube": (_CYLINDER_KEYS + ("inner_radius_mm",), ("stock_mm", "full_surface")),
    "face": (("name", "kind", "point", "normal"), ("stock_mm", "radius_mm", "full_surface")),   # radius_mm: a bounded cut (a disc), else the half-space; full_surface: only with radius_mm
    "stl": (("name", "kind", "file", "op"), ()),
}


@dataclass(frozen=True)
class Interface:
    """One entry of the `interfaces` list. `axis` is the cylinder axis, or the normal of a `face` (unit vector)."""
    name: str
    kind: str                                  # bore | boss | tube | face | stl
    op: str                                    # cut | union
    stock_mm: float = 0.0
    point: tuple[float, float, float] | None = None
    axis: tuple[float, float, float] | None = None
    radius_mm: float | None = None             # bore, boss, tube: radius (tube: outer radius); face: optional radius of the cut (a disc)
    inner_radius_mm: float | None = None       # tube only
    length_mm: float | None = None
    path: Path | None = None                   # stl only (absolute)
    full_surface: bool = False                 # this surface must be complete (CONTRACTS.md section 23, "A surface that must be complete")

    def to_dict(self, yaml_dir=None) -> dict:
        """The form stored in run.json (key `interfaces`). An STL path is relative to the YAML folder, or absolute."""
        d: dict = {"name": self.name, "kind": self.kind, "op": self.op}
        if self.kind == "stl":
            q = Path(self.path)
            try:
                d["file"] = q.resolve().relative_to(Path(yaml_dir).resolve()).as_posix() if yaml_dir else str(q)
            except ValueError:
                d["file"] = str(q)
            return d
        d["stock_mm"] = self.stock_mm
        d["point"] = list(self.point)
        d["normal" if self.kind == "face" else "axis"] = list(self.axis)
        for k in ("radius_mm", "inner_radius_mm", "length_mm"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.full_surface:   # the key is there only when it is true: a run.json without it is bit-equal to before
            d["full_surface"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict, yaml_dir=None, where: str = "interface") -> "Interface":
        """Read one entry of the `interfaces` key of run.json. A relative STL path starts at the YAML folder.

        `where` names the entry in an error (`interfaces[2]` when the caller has the index). run.json is a file that the program wrote,
        but a person or an agent can edit it: an entry that is not a mapping, or a `kind` that is not one of `INTERFACE_KINDS`
        (also a list or a mapping), is a `UserError` that names the key. There is no default kind.
        """
        if not isinstance(d, dict):
            raise UserError(f"'{where}' must be a mapping, got {d!r}. {_RUN_JSON_FIX}")
        kind = d.get("kind")
        if not isinstance(kind, str) or kind not in INTERFACE_KINDS:
            raise UserError(f"'{where}.kind' must be one of {', '.join(INTERFACE_KINDS)}, got {kind!r}. {_RUN_JSON_FIX}")
        if kind == "stl":
            q = Path(d["file"])
            if not q.is_absolute() and yaml_dir is not None:
                q = Path(yaml_dir) / q
            return cls(d["name"], kind, d["op"], 0.0, path=q)

        def vec(key, unit=False):
            v = [float(x) for x in d[key]]
            if unit:   # run.json may be edited by hand: make `axis` and `normal` a unit vector
                n = math.sqrt(sum(c * c for c in v))
                if n < DEGENERATE:
                    raise UserError(f"interface '{d['name']}': '{key}' must not be the zero vector, got {d[key]!r}")
                if abs(n - 1.0) > 1e-12:   # a unit vector from the loader stays as it is (bit-equal round trip)
                    v = [c / n + 0.0 for c in v]
            return tuple(v)

        return cls(d["name"], kind, _KIND_OP[kind], float(d.get("stock_mm", 0.0)), vec("point"),
                   vec("normal" if kind == "face" else "axis", unit=True),
                   *[(float(d[k]) if d.get(k) is not None else None) for k in ("radius_mm", "inner_radius_mm", "length_mm")],
                   full_surface=bool(d.get("full_surface", False)))


# --- acceptance block (CONTRACTS.md section 36) -----------------------------
# The ratchet (topopt/ratchet.py) reads these. The optimizer ignores them. A number in a rule comes from the YAML with
# its source text (the D7 pattern: value + source). The engine has no default number for a rule.

ACCEPTANCE_RULES = ("watertight", "bodies", "loose_island", "stress", "stiffness", "mass", "print_gate",
                    "displacement", "alignment", "buckling", "symmetry")
AGAINST_PARENT = ("absolute", "not_worse", "cap_or_parent")   # the parent forms of the stress rule
SOURCE_NOT_RECORDED = "source not recorded"
# rule -> (required keys, optional keys). A number key `K` also takes `K_source`, and a rule with a number key takes `source`.
_RULE_KEYS = {
    "watertight": ((), ()),
    "bodies": ((), ()),
    "loose_island": (("max_mm3",), ("relative_to_parent",)),
    "stress": ((), ("margin", "against_parent", "cases", "tolerance_pct")),
    "stiffness": (("max_drop_pct",), ("cases",)),
    "mass": ((), ("must_fall", "max_rise_pct")),
    "print_gate": ((), ()),
    "displacement": ((), ("against_parent", "cases", "tolerance_pct")),
    "alignment": ((), ("against_parent", "cases", "tolerance_pct")),
    "buckling": ((), ("min_factor", "against_parent", "cases", "tolerance_pct")),     # min_factor is required unless against_parent: not_worse
    "symmetry": (("max_mm",), ()),
}
_RULE_NUMBERS = {   # number key -> the bounds of `_num`
    "max_mm3": dict(lo=0.0), "margin": dict(lo=0.0, hi=1.0, lo_open=True), "max_drop_pct": dict(lo=0.0),
    "max_rise_pct": dict(lo=0.0), "min_factor": dict(lo=0.0, lo_open=True), "max_mm": dict(lo=0.0),
    "tolerance_pct": dict(lo=0.0),
}


# --- bore_checks block (CONTRACTS.md section 40) -------------------------------------
# `topopt verify` reads the block (the Goodman check and the press-fit check at the bores of a part). The optimizer ignores it.
# The four facet knobs have the values of the check of the earlier pipeline of the same author (pipeline/s9_print/final/pressfit.py). They
# are optional YAML fields. A material, interference or cup number has no default: the YAML gives it
# with its source text (decision D7: no presets, no defaults, no factors).
DEFAULT_RADIUS_TOLERANCE_MM = 0.35   # pressfit.py:45 (`TOL`), used at :50: a facet vertex is on the bore when |r - R| is at most this
DEFAULT_END_MARGIN_MM = 0.35         # pressfit.py:45 (`TOL`), used at :50: the facet centroid keeps this distance from the ends of the range
DEFAULT_NORMAL_LIMIT = 0.35          # pressfit.py:50: a wall facet has |n . axis| below this
DEFAULT_SAMPLE_BAND_MM = 1.2         # pressfit.py:127: the sample set has the points with |r - R| below this
BORE_KNOB_KEYS = ("radius_tolerance_mm", "end_margin_mm", "normal_limit", "sample_band_mm")
BORE_KNOB_DEFAULT_SOURCE = {   # the text that the report gives next to a knob that has its default
    "radius_tolerance_mm": "pipeline s9_print/final/pressfit.py:45 (TOL), used at :50",
    "end_margin_mm": "pipeline s9_print/final/pressfit.py:45 (TOL), used at :50",
    "normal_limit": "pipeline s9_print/final/pressfit.py:50",
    "sample_band_mm": "pipeline s9_print/final/pressfit.py:127",
}


@dataclass(frozen=True)
class AcceptanceRule:
    """One entry of `acceptance.rules`: the rule name, the keys the YAML gave, and the source text of each number."""
    rule: str
    options: tuple[tuple[str, object], ...] = ()   # (key, value) in YAML order; the value of `cases` is a tuple
    sources: tuple[tuple[str, str], ...] = ()      # (number key, source text); a number with no source is not listed

    def get(self, key: str, default=None):
        return dict(self.options).get(key, default)

    def source_of(self, key: str) -> str:
        """The source text of a number key, or `SOURCE_NOT_RECORDED`."""
        return dict(self.sources).get(key, SOURCE_NOT_RECORDED)


@dataclass(frozen=True)
class Acceptance:
    """The optional top-level `acceptance` block: which rules judge a candidate, and how."""
    rules: tuple[AcceptanceRule, ...]
    mesh_size_mm: float | None = None   # the mesh size of the judge; None: the size that verify picks for the start part
    percentile: float | None = None     # the percentile by volume of the stress rule; None: the default of verify
    owner_gate: bool = False            # True: the runner stops before it promotes an accepted candidate


@dataclass(frozen=True)
class BoreKnobs:
    """The four facet knobs. `given` lists the knobs that the YAML gave (the others have the default); `sources` the source text of a knob."""
    radius_tolerance_mm: float = DEFAULT_RADIUS_TOLERANCE_MM
    end_margin_mm: float = DEFAULT_END_MARGIN_MM
    normal_limit: float = DEFAULT_NORMAL_LIMIT
    sample_band_mm: float = DEFAULT_SAMPLE_BAND_MM
    given: tuple[str, ...] = ()
    sources: tuple[tuple[str, str], ...] = ()

    def origin(self, key: str) -> str:
        return "yaml" if key in self.given else "default"

    def source_of(self, key: str) -> str | None:
        return dict(self.sources).get(key)


@dataclass(frozen=True)
class BoreSpec:
    """One bore of `bore_checks.bores`. The contact range goes from `point` along the unit `axis` for `length_mm`."""
    name: str
    point: tuple[float, float, float]
    axis: tuple[float, float, float]            # unit vector
    radius_mm: float
    length_mm: float
    geometry_from: str                          # "yaml" or "interface:<name>"
    diametral_interference_mm: float | None = None
    mean_stress_mpa: float | None = None
    sources: tuple[tuple[str, str], ...] = ()   # (number key, source text); a number with no source is not listed

    def source_of(self, key: str) -> str | None:
        return dict(self.sources).get(key)


@dataclass(frozen=True)
class GoodmanMaterial:
    """One entry of `bore_checks.goodman.materials`: the strengths of the YAML (MPa). `se_mpa` is the fully corrected endurance limit."""
    label: str
    su_mpa: float
    sy_mpa: float
    se_mpa: float
    sources: tuple[tuple[str, str], ...] = ()

    def source_of(self, key: str) -> str | None:
        return dict(self.sources).get(key)


@dataclass(frozen=True)
class Cup:
    """One cup variant of `bore_checks.press_fit.cups`: the spring stiffness `k_mpa_per_mm`, or `e_mpa`, `nu` and `wall_mm` (the Lame form)."""
    name: str
    k_mpa_per_mm: float | None = None
    e_mpa: float | None = None
    nu: float | None = None
    wall_mm: float | None = None
    sources: tuple[tuple[str, str], ...] = ()

    @property
    def form(self) -> str:
        return "k" if self.k_mpa_per_mm is not None else "lame"

    def source_of(self, key: str) -> str | None:
        return dict(self.sources).get(key)


@dataclass(frozen=True)
class BoreChecks:
    """The optional top-level `bore_checks` block. `goodman` is True when the YAML has a `goodman` block (B4a), `press_fit` when it has a `press_fit` block (B4b)."""
    bores: tuple[BoreSpec, ...]
    knobs: BoreKnobs = BoreKnobs()
    materials: tuple[GoodmanMaterial, ...] = ()
    cups: tuple[Cup, ...] = ()
    goodman: bool = False
    press_fit: bool = False

    def checks(self) -> tuple[str, ...]:
        return tuple(n for n, on in (("goodman", self.goodman), ("pressfit", self.press_fit)) if on)


@dataclass(frozen=True)
class Config:
    name: str
    yaml_path: Path
    regions: Regions
    load_cases: tuple[LoadCase, ...]
    material: Material
    settings: Settings
    assumptions: tuple[str, ...] = ()  # notes of the author; the solver ignores them
    mirror: tuple[MirrorPlane, ...] = ()   # the `mirror` key: planes of mirror symmetry (section 26.1)
    limits: Limits = field(default_factory=lambda: Limits())   # A5 (CONTRACTS.md section 31)
    buckling: Buckling = Buckling()
    print_settings: PrintSettings = PrintSettings()   # the `print` block; the defaults when it is absent
    interfaces: tuple[Interface, ...] = ()   # the `interfaces` block; empty when it is absent
    springs: tuple[SpringSpec, ...] = ()   # CONTRACTS.md section 35
    acceptance: Acceptance | None = None   # CONTRACTS.md section 36; None when the YAML has no block
    joints: tuple[JointSpec, ...] = ()     # CONTRACTS.md section 38
    bore_checks: BoreChecks | None = None   # the `bore_checks` block (section 40); None when it is absent
    not_printed: tuple[NotPrinted, ...] = ()   # `regions.not_printed` (section 9.1): bodies that `result/part_only.stl` leaves out
    filament: FilamentUse | None = None    # `material.filament` (section 18.8): the filament and the source of each material number; None without the key

    def has_springs(self) -> bool:
        """True when the YAML has a `springs` entry."""
        return bool(self.springs)

    def has_joints(self) -> bool:
        """True when the YAML has a `joints` entry."""
        return bool(self.joints)

    def has_partial_supports(self) -> bool:
        """True when a `fixed` entry fixes fewer than the three axes (a roller or a slider)."""
        return any(a != ALL_AXES for a in self.regions.fixed_axes)

    def uses_extended_loads(self) -> bool:
        """True when the YAML has a moment, a remote force (`at`), a pressure, a per-axis support, a spring or a joint.

        A reader that knows only plain forces and full clamps (a late PR, `topopt verify`)
        tests this one value and adds its own refusal. CONTRACTS.md section 26.2.
        """
        return (self.has_partial_supports() or self.has_springs() or self.has_joints()
                or any(f.moment is not None or f.at is not None for c in self.load_cases for f in c.forces)
                or any(c.pressures for c in self.load_cases))


# --- the key names of each block (CONTRACTS.md section 41) ----------------------------------------------------------------
# The one place for the key names that the loader accepts. The loader (`_check_keys`) and `topopt/levers.py` read these constants,
# and the levers test derives the accepted key paths from `KEY_BLOCKS`. A new key goes into the constant of its block, and it then
# needs a card in `topopt/levers_cards.py`: the test fails for a key that has no card.
LEVERS_RESOURCE = "topopt://levers"
LEVERS_HELP = f"The cards of all keys: the MCP resource {LEVERS_RESOURCE}, or the command 'topopt levers'"

TOP_REQUIRED = ("name", "regions", "load_cases", "settings")
TOP_OPTIONAL = ("material", "assumptions", "mirror", "limits", "buckling", "print", "interfaces", "springs", "acceptance", "joints",
                "bore_checks")
REGIONS_REQUIRED = ("design", "fixed", "load")   # `fixed` is optional when the YAML has `springs` or `joints`: they are supports too
REGIONS_OPTIONAL = ("keep_out", "solid", "measure", "not_printed")
REGIONS_SUPPORTED_REQUIRED = tuple(k for k in REGIONS_REQUIRED if k != "fixed")   # the keys when the YAML has `springs` or `joints`
REGIONS_SUPPORTED_OPTIONAL = ("fixed", *REGIONS_OPTIONAL)
FIXED_ENTRY_KEYS = ("file", "axes")              # the mapping form of an entry of `regions.fixed`
LOAD_CASE_REQUIRED = ("name",)
LOAD_CASE_OPTIONAL = ("forces", "pressures", "weight", "compliance_limit", "stress_limit", "safety_factor", "allowable_mpa",
                      "allowable_source")
FORCE_REQUIRED = ("region",)
FORCE_OPTIONAL = ("force", "moment", "at", "coupling")
PRESSURE_REQUIRED = ("region", "value_mpa")
MATERIAL_OPTIONAL = ("E", "nu", "density", "filament")
# The refusal for a YAML (or a part spec) that has both keys. One text: the loader and the part builder (`regions_builder`) say it.
BOTH_VOLUME_KEYS_TEXT = (
    "'settings' has both volume_fraction and target_mass_g: the YAML takes one of them, and the engine does not choose. "
    "Options: keep settings.target_mass_g when the user asked for a mass in grams, or keep settings.volume_fraction when the "
    "user asked for a share of the material; remove the other key. The words of the user say which one they asked for")
SETTINGS_REQUIRED = ("voxel_size", "iterations")   # and exactly one of `volume_fraction`, `target_mass_g` (I7: the loader checks it)
SETTINGS_OPTIONAL = ("volume_fraction", "target_mass_g", "filter_radius", "penal", "beta_schedule", "move_limit", "solver", "case_combine", "max_active_voxels",
                     "max_grid_voxels", "objective", "stress_q", "stress_power", "stress_scale", "stress_step_acceptance", "stop_rules",
                     "filter_solid_neighbours", "min_member_mm", "robust", "eta_eroded", "mesh_method", "envelope_fill", "gpu_device", "gpu_case",
                     "gpu_tol", "gpu_backward_error", "gpu_max_iter", "perimeter_max_mm2", "objective_scale",
                     "part_only_min_film_mm", "fine_pass")
STOP_RULES_KEYS = ("enabled", "objective_change", "window", "grayness_max", "time_budget_s")
MIRROR_REQUIRED = ("normal", "at_mm")
MIRROR_OPTIONAL = ("align",)
LIMITS_OPTIONAL = ("displacement", "alignment")
DISPLACEMENT_REQUIRED = ("region", "limit_mm")
DISPLACEMENT_OPTIONAL = ("name", "direction", "sense", "cases", "price")
ALIGNMENT_REQUIRED = ("region_a", "region_b", "axis")
ALIGNMENT_OPTIONAL = ("name", "shift_limit_mm", "tilt_limit_rad", "cases", "price")
BUCKLING_OPTIONAL = ("check", "modes", "density", "max_seconds", "max_gb", "min_factor", "ks_q")
# `print`: PRINT_KEYS. `interfaces` entries: INTERFACE_KEYS (by kind). Beam sections of a joint: SECTION_KEYS (by kind).
SPRING_REQUIRED = ("region",)
SPRING_OPTIONAL = ("stiffness", "stiffness_per_area")
JOINT_REQUIRED = ("region", "to", "section")
JOINT_OPTIONAL = ("origin", "orient", "bond_stiffness_per_area", "E", "nu")
ACCEPTANCE_REQUIRED = ("rules",)
ACCEPTANCE_OPTIONAL = ("mesh_size_mm", "percentile", "owner_gate")
# bore_checks (CONTRACTS.md section 40.2). A number key `K` of an entry also takes `K_source` (the D7 pattern: value + source text).
BORE_CHECKS_REQUIRED = ("bores",)
BORE_CHECKS_OPTIONAL = (*BORE_KNOB_KEYS, *(f"{k}_source" for k in BORE_KNOB_KEYS), "goodman", "press_fit")
BORE_GEOMETRY_KEYS = ("point", "axis", "radius_mm", "length_mm")
BORE_NUMBER_KEYS = ("radius_mm", "length_mm", "diametral_interference_mm", "mean_stress_mpa")   # the numbers of a bore that take a source
BORE_REQUIRED = ("name",)
BORE_OPTIONAL = ("interface", *BORE_GEOMETRY_KEYS, "diametral_interference_mm", "mean_stress_mpa", "source",
                 *(f"{k}_source" for k in BORE_NUMBER_KEYS))
GOODMAN_REQUIRED = ("materials",)
GOODMAN_STRENGTH_KEYS = ("su_mpa", "sy_mpa", "se_mpa")
GOODMAN_MATERIAL_OPTIONAL = (*GOODMAN_STRENGTH_KEYS, "source", *(f"{k}_source" for k in GOODMAN_STRENGTH_KEYS))
PRESS_FIT_REQUIRED = ("cups",)
CUP_NUMBER_KEYS = ("k_mpa_per_mm", "e_mpa", "nu", "wall_mm")
CUP_REQUIRED = ("name",)
CUP_OPTIONAL = (*CUP_NUMBER_KEYS, "source", *(f"{k}_source" for k in CUP_NUMBER_KEYS))
# acceptance.rules[]: the keys of all rules together. `_RULE_KEYS` holds the keys of each rule; a rule with a number also takes `source`.
ACCEPTANCE_RULE_KEYS = ("rule", *dict.fromkeys(k for req, opt in _RULE_KEYS.values() for k in (*req, *opt)), "source",
                        *(f"{k}_source" for k in _RULE_NUMBERS))
INTERFACE_ALL_KEYS = tuple(dict.fromkeys(k for req, opt in INTERFACE_KEYS.values() for k in (*req, *opt)))
SECTION_REQUIRED = {kind: ("kind", *keys) for kind, keys in SECTION_KEYS.items()}   # the keys of the section of one kind
SECTION_ALL_KEYS = ("kind", *dict.fromkeys(k for keys in SECTION_KEYS.values() for k in keys))

KEY_BLOCKS = {   # block path -> (required keys, optional keys). The top level is "". `[]` is one entry of a list.
    "": (TOP_REQUIRED, TOP_OPTIONAL),
    "regions": (REGIONS_REQUIRED, REGIONS_OPTIONAL),
    "regions.fixed[]": (FIXED_ENTRY_KEYS, ()),
    "load_cases[]": (LOAD_CASE_REQUIRED, LOAD_CASE_OPTIONAL),
    "load_cases[].forces[]": (FORCE_REQUIRED, FORCE_OPTIONAL),
    "load_cases[].pressures[]": (PRESSURE_REQUIRED, ()),
    "material": ((), MATERIAL_OPTIONAL),
    "settings": (SETTINGS_REQUIRED, SETTINGS_OPTIONAL),
    "settings.stop_rules": ((), STOP_RULES_KEYS),
    "settings.fine_pass": ((), FINE_PASS_KEYS),
    "mirror[]": (MIRROR_REQUIRED, MIRROR_OPTIONAL),
    "limits": ((), LIMITS_OPTIONAL),
    "limits.displacement[]": (DISPLACEMENT_REQUIRED, DISPLACEMENT_OPTIONAL),
    "limits.alignment[]": (ALIGNMENT_REQUIRED, ALIGNMENT_OPTIONAL),
    "buckling": ((), BUCKLING_OPTIONAL),
    "print": ((), PRINT_KEYS),
    "interfaces[]": ((), INTERFACE_ALL_KEYS),   # by kind: INTERFACE_KEYS
    "springs[]": (SPRING_REQUIRED, SPRING_OPTIONAL),
    "joints[]": (JOINT_REQUIRED, JOINT_OPTIONAL),
    "joints[].section": ((), SECTION_ALL_KEYS),   # by kind: SECTION_KEYS
    "acceptance": (ACCEPTANCE_REQUIRED, ACCEPTANCE_OPTIONAL),
    "acceptance.rules[]": ((), ACCEPTANCE_RULE_KEYS),   # by rule: _RULE_KEYS
    "bore_checks": (BORE_CHECKS_REQUIRED, BORE_CHECKS_OPTIONAL),
    "bore_checks.bores[]": (BORE_REQUIRED, BORE_OPTIONAL),
    "bore_checks.goodman": (GOODMAN_REQUIRED, ()),
    "bore_checks.goodman.materials.<label>": ((), GOODMAN_MATERIAL_OPTIONAL),
    "bore_checks.press_fit": (PRESS_FIT_REQUIRED, ()),
    "bore_checks.press_fit.cups[]": (CUP_REQUIRED, CUP_OPTIONAL),
}


def accepted_key_paths() -> tuple[str, ...]:
    """Every key path that the loader accepts, from `KEY_BLOCKS`: `settings.penal`, `load_cases[].forces[].force`, `name`.

    A key that two blocks of one path share (the keys of all interface kinds) is listed once.
    """
    paths = []
    for block, (required, optional) in KEY_BLOCKS.items():
        paths += [f"{block}.{k}" if block else k for k in (*required, *optional)]
    return tuple(dict.fromkeys(paths))


# --- small helpers -------------------------------------------------------


class _Ctx:
    """Holds the file name for error messages."""

    def __init__(self, path: Path, printer_catalog=None, filament_catalog=None):
        self.path = path
        self.printer_catalog = printer_catalog      # the printers to search for `print.printer_model`; None: the work folder's
        self.filament_catalog = filament_catalog    # the filaments to search for `material.filament`; None: the work folder's

    def err(self, msg: str) -> UserError:
        return UserError(f"{msg} (file: {self.path})")


def unknown_key_text(key, where, known) -> str:
    """The text for a key that a block does not know. One wording: the loader (`_check_keys`) and the part builder (`regions_builder`) say it."""
    return (f"unknown key '{key}' in '{where}'. Known keys: {', '.join(known) or 'none'}. "
            f"Options: use a known key, or remove '{key}'. {LEVERS_HELP}")


def _check_keys(ctx, d, where, required, optional, hints=None):
    if not isinstance(d, dict):
        raise ctx.err(f"'{where}' must be a mapping")
    for k in d:
        if k not in required and k not in optional:
            raise ctx.err(unknown_key_text(k, where, [*required, *optional]))
    for k in required:
        if k not in d:
            raise ctx.err(f"missing key '{k}' in '{where}'" + (f": {hints[k]}" if hints and k in hints else ""))


def _num(ctx, v, where, *, lo=None, hi=None, lo_open=False, hi_open=False, integer=False):
    ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    if integer:
        ok = isinstance(v, int) and not isinstance(v, bool)
    try:
        ok = ok and math.isfinite(v)
    except OverflowError:  # an integer too large for a float
        raise ctx.err(f"'{where}' must be a finite number")
    if not ok:
        kind = "an integer" if integer else "a number"
        raise ctx.err(f"'{where}' must be {kind}, got {v!r}")
    if lo is not None and (v < lo or (lo_open and v == lo)):
        raise ctx.err(f"'{where}' must be {'>' if lo_open else '>='} {lo}, got {v!r}")
    if hi is not None and (v > hi or (hi_open and v == hi)):
        raise ctx.err(f"'{where}' must be {'<' if hi_open else '<='} {hi}, got {v!r}")
    return v if integer else float(v)


def _numeric_text(v):
    """A string that float() reads is that number. YAML 1.1 loads 5e-3 as text (it needs 5.0e-3).

    `compliance_limit`, the stress keys (`stress_limit`, `safety_factor`, `allowable_mpa`), the A5 limit keys
    (`limit_mm`, `shift_limit_mm`, `tilt_limit_rad`), the number limits of the `print` block, the `at_mm` of a mirror
    plane, `perimeter_max_mm2` and the two gpu-mg numbers `gpu_tol` and `gpu_backward_error` (through `_num_text`) use this:
    values such as 5e-3 or 1e2 are usual there (small limits in N*mm and large limits in mm3: write 1e5 or
    5e-3). Any other string stays a string, and `_num` reports it.
    """
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return v


def _num_text(ctx, v, where, **kw):
    """`_num` for a key where a text such as "1e2" is read as a number (see `_numeric_text`)."""
    return _num(ctx, _numeric_text(v), where, **kw)


def _str(ctx, v, where):
    if not isinstance(v, str) or not v.strip():
        raise ctx.err(f"'{where}' must be a non-empty string, got {v!r}")
    return v


def _stl(ctx, v, where, base: Path) -> Path:
    name = _str(ctx, v, where)
    p = Path(name)
    if not p.is_absolute():
        p = base / p
    p = Path(p).resolve()
    if not p.is_file():
        raise ctx.err(f"STL file not found: {p} (used in '{where}')")
    return p


def _stl_list(ctx, v, where, base, *, required):
    if not isinstance(v, list):
        raise ctx.err(f"'{where}' must be a list of STL files, got {v!r}")
    if required and not v:
        raise ctx.err(f"'{where}' needs at least one STL file")
    return tuple(_stl(ctx, x, f"{where}[{i}]", base) for i, x in enumerate(v))


def _stop_rules(ctx, v) -> StopRules:
    """`settings.stop_rules`: false (default), true, or a mapping of thresholds (a mapping turns the rules on).

    Keys: enabled (default true in a mapping), objective_change, window, grayness_max, time_budget_s.
    A text such as "1e-4" is read as a number (YAML 1.1 loads 1e-4 as text).
    """
    d = StopRules()
    if isinstance(v, bool):
        return StopRules(enabled=v)
    if not isinstance(v, dict):
        raise ctx.err(f"'settings.stop_rules' must be true, false or a mapping of thresholds, got {v!r}")
    _check_keys(ctx, v, "settings.stop_rules", [], STOP_RULES_KEYS)
    enabled = v.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ctx.err(f"'settings.stop_rules.enabled' must be true or false, got {enabled!r}")
    budget = v.get("time_budget_s", d.time_budget_s)
    return StopRules(
        enabled=enabled,
        objective_change=_num_text(ctx, v.get("objective_change", d.objective_change),
                                   "settings.stop_rules.objective_change", lo=0.0, lo_open=True),
        window=_num(ctx, v.get("window", d.window), "settings.stop_rules.window", lo=1, integer=True),
        grayness_max=_num_text(ctx, v.get("grayness_max", d.grayness_max), "settings.stop_rules.grayness_max",
                               lo=0.0, hi=1.0, lo_open=True),
        time_budget_s=None if budget is None else _num_text(ctx, budget, "settings.stop_rules.time_budget_s",
                                                            lo=0.0, lo_open=True))


def _fine_pass(ctx, v) -> FinePassSettings:
    """`settings.fine_pass`: absent (the default), or a mapping with the key `objective_change`, a finite number above 0.

    A text such as "1e-3" is read as a number. Each refusal names the key path and gives an example.
    """
    if v is None:
        return FinePassSettings()
    if not isinstance(v, dict):
        raise ctx.err(f"'settings.fine_pass' must be a mapping with the key objective_change, got {v!r}. Example: {FINE_PASS_EXAMPLE}")
    for k in v:
        if k not in FINE_PASS_KEYS:
            raise ctx.err(f"{unknown_key_text(k, 'settings.fine_pass', list(FINE_PASS_KEYS))} Example: {FINE_PASS_EXAMPLE}")
    if "objective_change" not in v:
        return FinePassSettings()
    value = _numeric_text(v["objective_change"])
    try:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    except OverflowError:
        ok = False
    if not ok:
        raise ctx.err(f"'settings.fine_pass.objective_change' must be a finite number above 0 (the relative change of the objective at which a fine "
                      f"pass has settled), got {v['objective_change']!r}. Example: {FINE_PASS_EXAMPLE}")
    return FinePassSettings(objective_change=float(value))


def _mirror_block(ctx, raw) -> tuple[MirrorPlane, ...]:
    """Read the optional `mirror` list (CONTRACTS.md section 26.1). Each refusal is one line with the supported form."""
    if raw is None:
        return ()

    def bad(msg):
        return ctx.err(f"{msg}. {MIRROR_HELP}")

    if not isinstance(raw, list):
        raise bad(f"'mirror' must be a list of planes, got {raw!r}")
    if len(raw) > MAX_PLANES:
        raise bad(f"'mirror' has {len(raw)} planes, and at most {MAX_PLANES} are supported (one for each axis)")
    planes = []
    for i, d in enumerate(raw):
        w = f"mirror[{i}]"
        if not isinstance(d, dict):
            raise bad(f"'{w}' must be a mapping with normal and at_mm, got {d!r}")
        for k in d:
            if k not in (*MIRROR_REQUIRED, *MIRROR_OPTIONAL):
                raise bad(f"unknown key '{k}' in '{w}'. Known keys: {', '.join([*MIRROR_REQUIRED, *MIRROR_OPTIONAL])}. "
                          f"Options: use a known key, or remove '{k}'. {LEVERS_HELP}")
        for k in MIRROR_REQUIRED:
            if k not in d:
                raise bad(f"missing key '{k}' in '{w}'")
        try:
            axis = axis_of_normal(d["normal"])
        except ValueError as e:
            raise bad(f"'{w}.normal': {e}")
        at = _numeric_text(d["at_mm"])   # YAML 1.1 loads 1e-3 as text
        if not isinstance(at, (int, float)) or isinstance(at, bool) or not math.isfinite(at):
            raise bad(f"'{w}.at_mm' must be a finite number (mm), got {at!r}")
        align = d.get("align", "auto")
        if align not in ALIGNS:
            raise bad(f"'{w}.align' must be {', '.join(ALIGNS)}, got {align!r}")
        for j, q in enumerate(planes):
            if q.axis == axis:
                raise bad(f"'{w}' has the axis {axis}, like 'mirror[{j}]': each axis can have one plane")
        planes.append(MirrorPlane(axis, float(at), align))
    return tuple(planes)


# --- load and support types (CONTRACTS.md section 26.2) ----------------------


def _vec3(ctx, v, where, *, unit=False):
    if not isinstance(v, list) or len(v) != 3:
        raise ctx.err(f"'{where}' must be a list of 3 numbers, got {v!r}")
    vec = [_num(ctx, x, f"{where}[{i}]") for i, x in enumerate(v)]
    if unit:
        n = math.sqrt(sum(c * c for c in vec))
        if n < DEGENERATE:
            raise ctx.err(f"'{where}' must not be the zero vector")
        vec = [c / n + 0.0 for c in vec]   # + 0.0: no negative zero
    return tuple(vec)


def _axes(ctx, v, where):
    """A list of the axes x, y, z (distinct, at least one) as a sorted tuple of 0, 1, 2."""
    letters = {"x": 0, "y": 1, "z": 2}
    ok = isinstance(v, list) and v and all(isinstance(a, str) and a.lower() in letters for a in v)
    if not ok or len({a.lower() for a in v}) != len(v):
        raise ctx.err(f"'{where}' must be a list of the axes x, y, z (distinct, at least one), "
                      f"for example [y] for a roller; got {v!r}")
    return tuple(sorted(letters[a.lower()] for a in v))


_NO_SUPPORT = (
    "without a fixed region and without a spring, all 6 rigid-body modes are free (translation along x, y, z and "
    "rotation about x, y, z), so the stiffness matrix is singular. Options: add a fixed region to 'regions.fixed' "
    "(a path is a full clamp), or add a spring on a load region in 'springs' "
    "(for example {region: mount, stiffness: [50, 50, 50]}; the springs section of CONTRACTS.md), or add a beam to a "
    "fixed point on a load region in 'joints' (for example {region: mount, to: [0, 0, -30], section: {kind: round, d: 4}}; "
    "the joints section of CONTRACTS.md)")


def _fixed_list(ctx, v, where, base, required=True):
    """`regions.fixed`: STL paths, each one a full clamp, or `{file, axes}` for a per-axis support.

    Return `(paths, axes)`. `axes` is `()` when no entry is a mapping (the form of every older YAML).
    The list may be empty only when the YAML has springs (`required` False).
    """
    if required and v == []:
        raise ctx.err(f"'{where}' needs at least one STL file: {_NO_SUPPORT}")
    if not isinstance(v, list) or not any(isinstance(x, dict) for x in v):
        return _stl_list(ctx, v, where, base, required=required), ()
    paths, axes = [], []
    for i, x in enumerate(v):
        w = f"{where}[{i}]"
        if isinstance(x, dict):
            _check_keys(ctx, x, w, FIXED_ENTRY_KEYS, [])
            paths.append(_stl(ctx, x["file"], f"{w}.file", base))
            axes.append(_axes(ctx, x["axes"], f"{w}.axes"))
        else:
            paths.append(_stl(ctx, x, w, base))
            axes.append(ALL_AXES)
    return tuple(paths), tuple(axes)


_RIGID_TEXT = (
    "a rigid coupling (RBE2) makes the nodes of the region move as one body. That needs constraint "
    "elimination or a bordered system, and this engine has neither. The engine has the distributing coupling "
    "(RBE3-like): the force and the moment are spread over the nodes of the region, and no stiffness is added. "
    "Options: remove 'coupling' (distributing is the default), or model the stiff link as a load region of "
    "solid voxels. A bordered rigid segment (as in the topology pipeline) is a later option")


def _force_entry(ctx, f, fw, load_names):
    """One entry of `forces`: `region` and `force`, with the optional `moment`, `at` and `coupling`."""
    _check_keys(ctx, f, fw, FORCE_REQUIRED, FORCE_OPTIONAL)
    if "force" not in f and "moment" not in f:
        raise ctx.err(f"missing key 'force' in '{fw}' (an entry needs 'force', 'moment', or both)")
    reg = _str(ctx, f["region"], f"{fw}.region")
    if reg not in load_names:
        raise ctx.err(
            f"'{fw}.region' is '{reg}', not a load region "
            f"(known: {', '.join(sorted(load_names))})")
    if "at" in f and "force" not in f:
        raise ctx.err(f"'{fw}.at' is the point where 'force' acts, so it needs 'force'. "
                      "A 'moment' has no point of application")
    coupling = f.get("coupling", "distributing")
    if coupling == "rigid":
        raise ctx.err(f"'{fw}.coupling' is 'rigid': {_RIGID_TEXT}")
    if coupling != "distributing":
        raise ctx.err(f"'{fw}.coupling' must be 'distributing' (the coupling of this engine), got {coupling!r}")
    force = _vec3(ctx, f["force"], f"{fw}.force") if "force" in f else (0.0, 0.0, 0.0)
    moment = _vec3(ctx, f["moment"], f"{fw}.moment") if "moment" in f else None
    at = _vec3(ctx, f["at"], f"{fw}.at") if "at" in f else None
    return LoadForce(reg, force, moment, at)


def _pressure_entry(ctx, p, pw, regions, load_names):
    """One entry of `pressures`: a load region name and a pressure in MPa."""
    _check_keys(ctx, p, pw, PRESSURE_REQUIRED, [])
    reg = _str(ctx, p["region"], f"{pw}.region")
    if reg not in load_names:
        if reg in {q.stem for q in regions.fixed} | {q.name for q in regions.fixed}:
            raise ctx.err(
                f"'{pw}.region' is '{reg}', a fixed region (its nodes are fixed along its axes), so the "
                "force of a pressure on it along those axes would be lost. A pressure acts on a name of "
                "regions.load. Options: put the loaded body in regions.load, with a face toward empty space or "
                "a keep-out region")
        if reg in {regions.design.stem, regions.design.name}:
            raise ctx.err(
                f"'{pw}.region' is '{reg}', the design space. A pressure on the design surface changes "
                "with the design, and the engine has no design-dependent load. A pressure acts on a name "
                "of regions.load. Options: put the loaded body in regions.load")
        raise ctx.err(f"'{pw}.region' is '{reg}', not a load region (known: {', '.join(sorted(load_names))})")
    return PressureLoad(reg, _num(ctx, p["value_mpa"], f"{pw}.value_mpa"))


# --- A5: parse regions.measure and limits ---------------------------------

SENSES = ("abs", "max", "min")


def _measure_regions(ctx, raw, load_names, base: Path) -> tuple[MeasureRegion, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ctx.err("'regions.measure' must be a mapping of name to STL file")
    out = []
    for k, v in raw.items():
        name = _str(ctx, k, "regions.measure name")
        if name in load_names:
            raise ctx.err(f"'regions.measure.{name}' has the name of a load region; "
                          f"a row names a region, so the names must differ")
        out.append(MeasureRegion(name, _stl(ctx, v, f"regions.measure.{name}", base)))
    return tuple(out)


def _unit_vector(ctx, v, where):
    if not isinstance(v, list) or len(v) != 3:
        raise ctx.err(f"'{where}' must be a list of 3 numbers, got {v!r}")
    vec = [_num(ctx, x, f"{where}[{i}]") for i, x in enumerate(v)]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        raise ctx.err(f"'{where}' must not be the zero vector")
    return tuple(x / norm for x in vec)


def _row_cases(ctx, v, where, case_names):
    """`cases`: one case name, or a list of names. A missing key means every case."""
    if v is None:
        return tuple(case_names)
    names = [v] if isinstance(v, str) else v
    if not isinstance(names, list) or not names:
        raise ctx.err(f"'{where}' must be a case name or a list of case names, got {v!r}")
    for i, n in enumerate(names):
        if n not in case_names:
            raise ctx.err(f"'{where}[{i}]' is {n!r}, not a load case (known: {', '.join(case_names)})")
    if len(set(names)) != len(names):
        raise ctx.err(f"'{where}' names a load case twice: {names}")
    return tuple(names)


def _region_name(ctx, v, where, region_names):
    name = _str(ctx, v, where)
    if name not in region_names:
        known = ", ".join(sorted(region_names)) or "none: add regions.measure or use a load region"
        raise ctx.err(f"'{where}' is {name!r}, not a measure or a load region (known: {known})")
    return name


def _price(ctx, row, where):
    return _num(ctx, row["price"], f"{where}.price", lo=0.0, lo_open=True) if "price" in row else None


def _row_name(ctx, row, where, default, seen, sense="abs"):
    """The name of a row. The pair (sense, name) is unique: a `max` and a `min` row may share a region."""
    name = _str(ctx, row["name"], f"{where}.name") if "name" in row else default
    if (sense, name) in seen:
        both = f"the name {name!r}" if sense == "abs" else f"the name {name!r} and the sense {sense!r}"
        raise ctx.err(f"'{where}' has {both}, as an earlier row; give each row its own 'name'")
    seen.add((sense, name))
    return name


def _limits(ctx, raw, case_names, region_names) -> Limits:
    if raw is None:
        return Limits()
    _check_keys(ctx, raw, "limits", [], LIMITS_OPTIONAL)
    disp, align = [], []
    for kind, rows in (("displacement", raw.get("displacement")), ("alignment", raw.get("alignment"))):
        if rows is not None and not isinstance(rows, list):
            raise ctx.err(f"'limits.{kind}' must be a list of rows")
    seen = set()
    for i, row in enumerate(raw.get("displacement") or []):
        w = f"limits.displacement[{i}]"
        _check_keys(ctx, row, w, DISPLACEMENT_REQUIRED, DISPLACEMENT_OPTIONAL)
        region = _region_name(ctx, row["region"], f"{w}.region", region_names)
        limit = _num(ctx, _numeric_text(row["limit_mm"]), f"{w}.limit_mm", lo=0.0, lo_open=True)
        direction = _unit_vector(ctx, row["direction"], f"{w}.direction") if "direction" in row else None
        sense = row.get("sense", "abs")
        if sense not in SENSES:
            raise ctx.err(f"'{w}.sense' must be abs, max or min, got {sense!r}")
        if direction is None and "sense" in row:
            raise ctx.err(f"'{w}.sense' needs a 'direction'. Without a direction the row limits the "
                          f"norm of the mean displacement vector, and a norm has no sign")
        name = _row_name(ctx, row, w, region, seen, sense)
        disp.append(DisplacementLimit(name, region, limit, direction, sense,
                                      _row_cases(ctx, row.get("cases"), f"{w}.cases", case_names),
                                      _price(ctx, row, w)))
    seen = set()
    for i, row in enumerate(raw.get("alignment") or []):
        w = f"limits.alignment[{i}]"
        _check_keys(ctx, row, w, ALIGNMENT_REQUIRED, ALIGNMENT_OPTIONAL)
        a = _region_name(ctx, row["region_a"], f"{w}.region_a", region_names)
        b = _region_name(ctx, row["region_b"], f"{w}.region_b", region_names)
        if a == b:
            raise ctx.err(f"'{w}' compares region {a!r} with itself; name two different regions")
        axis = _unit_vector(ctx, row["axis"], f"{w}.axis")
        lim = {k: _num(ctx, _numeric_text(row[k]), f"{w}.{k}", lo=0.0, lo_open=True)
               for k in ("shift_limit_mm", "tilt_limit_rad") if k in row}
        if not lim:
            raise ctx.err(f"'{w}' needs 'shift_limit_mm' or 'tilt_limit_rad' (or both)")
        name = _row_name(ctx, row, w, f"{a}-{b}", seen)
        align.append(AlignmentLimit(name, a, b, axis, lim.get("shift_limit_mm"),
                                    lim.get("tilt_limit_rad"),
                                    _row_cases(ctx, row.get("cases"), f"{w}.cases", case_names),
                                    _price(ctx, row, w)))
    return Limits(tuple(disp), tuple(align))


def _buckling(ctx, b) -> Buckling:
    """The optional block `buckling` (CONTRACTS.md sections 33 and 39)."""
    _check_keys(ctx, b, "buckling", [], BUCKLING_OPTIONAL)
    d = Buckling()
    check = b.get("check", d.check)
    if not isinstance(check, bool):
        raise ctx.err(f"'buckling.check' must be true or false, got {check!r}")
    density = b.get("density", d.density)
    if density not in ("printed", "gray"):
        raise ctx.err(f"'buckling.density' must be printed or gray, got {density!r}")
    max_gb = b.get("max_gb", d.max_gb)
    if max_gb is not None:
        max_gb = _num(ctx, max_gb, "buckling.max_gb", lo=0.0, lo_open=True)
    min_factor, ks_q = b.get("min_factor", d.min_factor), b.get("ks_q", d.ks_q)
    if min_factor is not None:
        min_factor = _num(ctx, min_factor, "buckling.min_factor", lo=0.0, lo_open=True)
    if ks_q is not None:
        ks_q = _num(ctx, ks_q, "buckling.ks_q", lo=0.0, lo_open=True)
    return Buckling(
        check=check,
        modes=_num(ctx, b.get("modes", d.modes), "buckling.modes", lo=1, integer=True),
        density=density,
        max_seconds=_num(ctx, b.get("max_seconds", d.max_seconds), "buckling.max_seconds",
                         lo=0.0, lo_open=True),
        max_gb=max_gb,
        min_factor=min_factor,
        ks_q=ks_q,
    )


def _print_block(ctx, raw) -> PrintSettings:
    """Read the optional `print` block (CONTRACTS.md section 18). Unknown keys are an error."""
    if raw is None:
        raw = {}
    _check_keys(ctx, raw, "print", [], PRINT_KEYS)
    given = [k for k in PRINT_KEYS if k in raw]
    bd = DEFAULT_BUILD_DIRECTION
    if "build_direction" in raw:
        v = raw["build_direction"]
        if isinstance(v, str):
            m = _AXIS_NAME.match(v.strip().lower())   # one optional sign and one axis letter
            if m is None:
                raise ctx.err(f"'print.build_direction' must be a vector of 3 numbers or an axis "
                              f"name (+x, -x, +y, -y, +z, -z), got {v!r}")
            sign = -1.0 if m.group(1) == "-" else 1.0
            bd = tuple(sign * c + 0.0 for c in _AXIS_VECTORS[m.group(2)])   # + 0.0: no negative zero
        else:
            if not isinstance(v, list) or len(v) != 3:
                raise ctx.err(f"'print.build_direction' must be a vector of 3 numbers or an axis "
                              f"name (+x, -x, +y, -y, +z, -z), got {v!r}")
            vec = [_num(ctx, x, f"print.build_direction[{i}]") for i, x in enumerate(v)]
            n = math.sqrt(sum(c * c for c in vec))
            if n < DEGENERATE:
                raise ctx.err("'print.build_direction' must not be the zero vector")
            bd = tuple(c / n + 0.0 for c in vec)
    profile = _printer_profile(ctx, raw["printer_model"]) if "printer_model" in raw else None
    derived = []
    bed = (DEFAULT_BED_MM, DEFAULT_BED_MM)
    if "bed_mm" in raw:
        v = raw["bed_mm"]
        if isinstance(v, list):
            if len(v) != 2:
                raise ctx.err(f"'print.bed_mm' must be one number or a list of 2 numbers, got {v!r}")
            bed = tuple(_num(ctx, x, f"print.bed_mm[{i}]", lo=0.0, lo_open=True) for i, x in enumerate(v))
        else:
            side = _num(ctx, v, "print.bed_mm", lo=0.0, lo_open=True)
            bed = (side, side)
    elif profile is not None and profile.bed_mm is not None:      # the YAML wins; the profile fills a key that it does not give
        bed = tuple(profile.bed_mm)
        derived.append("bed_mm")
    printer = DEFAULT_PRINTER_TYPE
    if "printer_type" not in raw and profile is not None:
        printer = profile.printer_type
        derived.append("printer_type")
    if "printer_type" in raw:
        v = raw["printer_type"]
        printer = v.strip().lower() if isinstance(v, str) else v
        if not isinstance(printer, str) or printer not in PRINTER_TYPES:   # a list or mapping is a user error, not an unhashable crash
            raise ctx.err(f"'print.printer_type' must be one of {', '.join(PRINTER_TYPES)}, got {v!r}")
    angle = None
    if "self_supporting_angle_deg" in raw:
        angle = _num(ctx, raw["self_supporting_angle_deg"], "print.self_supporting_angle_deg",
                     lo=0.0, hi=90.0)
    limits = []
    for key in PRINT_LIMIT_KEYS:
        if key in raw:
            if key == "max_bodies":
                v = _num(ctx, raw[key], f"print.{key}", lo=1, integer=True)
            elif key == "min_wall_area_pct":
                v = _num(ctx, _numeric_text(raw[key]), f"print.{key}", lo=0.0, hi=100.0)
            else:
                v = _num(ctx, _numeric_text(raw[key]), f"print.{key}", lo=0.0, lo_open=(key == "min_wall_mm"))
            limits.append((key, v))
    return PrintSettings(bd, bed, printer, angle, tuple(given), tuple(limits), profile, tuple(derived))


def _printer_profile(ctx, value) -> "printers.PrinterProfile":
    """The printer of `print.printer_model` (an id, or a name or alias of the catalog). A text that is not one printer is a one-line error
    that names the candidates or the suggestions and the ways on (CONTRACTS.md section 18.7)."""
    if not isinstance(value, str) or not value.strip():
        raise ctx.err(f"'print.printer_model' must be the name or the id of a printer as text, got {value!r}. {printers.WAYS_ON}")
    try:
        # the shipped catalog and the user printers of the work folder, unless the caller gives its own catalog
        res = printers.resolve(value, ctx.printer_catalog or printers.work_catalog())
    except printers.CatalogError as exc:
        raise PrinterNotFound(str(ctx.err(f"'print.printer_model' cannot be checked: {exc}. Remove printer_model and give bed_mm and printer_type")))
    if res.status != "match":
        raise PrinterNotFound(str(ctx.err(printers.refusal_text(res))))
    return res.profile


def parse_print_block(raw, yaml_path="<yaml>", printer_catalog=None) -> PrintSettings:
    """Check a raw `print` mapping (for tools that read the YAML themselves). Raise `UserError`.

    `printer_catalog` (`topopt.printers.Catalog`): the printers that `printer_model` is searched in; the default is the shipped
    catalog with the user printers of the work folder (CONTRACTS.md section 18.7)."""
    return _print_block(_Ctx(Path(yaml_path), printer_catalog), raw)


def _material_block(ctx, m) -> Material:
    """Read the numbers of the optional `material` block (CONTRACTS.md section 2). Unknown keys are an error.

    The key `filament` is read by `_material_and_filament`; this function reads `E`, `nu` and `density` only."""
    _check_keys(ctx, m, "material", [], MATERIAL_OPTIONAL)
    d = Material()
    return Material(
        E=_num(ctx, m.get("E", d.E), "material.E", lo=0.0, lo_open=True),
        nu=_num(ctx, m.get("nu", d.nu), "material.nu", lo=-1.0, hi=0.5, lo_open=True, hi_open=True),
        density=_num(ctx, m.get("density", d.density), "material.density", lo=0.0, lo_open=True),
    )


def _filament_found(ctx, value) -> "tuple[materials_catalog.FilamentProfile, materials_catalog.Catalog]":
    """The filament of `material.filament` (an id, or a name or alias of the catalog) and the catalog that was searched. A text that is not
    one filament is a one-line error that names the candidates or the suggestions and the ways on (CONTRACTS.md section 18.8)."""
    if not isinstance(value, str) or not value.strip():
        raise ctx.err(f"'material.filament' must be the name or the id of a filament as text, got {value!r}. {materials_catalog.WAYS_ON}")
    try:
        # the shipped catalog and the user filaments of the work folder, unless the caller gives its own catalog
        catalog = ctx.filament_catalog or materials_catalog.work_catalog()
        res = materials_catalog.resolve(value, catalog)
    except materials_catalog.CatalogError as exc:
        raise FilamentNotFound(str(ctx.err(f"'material.filament' cannot be checked: {exc}. Remove filament and give E and density")))
    if res.status != "match":
        raise FilamentNotFound(str(ctx.err(materials_catalog.refusal_text(res))))
    return res.profile, catalog


def _filament_profile(ctx, value) -> "materials_catalog.FilamentProfile":
    """The filament of `material.filament` (see `_filament_found`)."""
    return _filament_found(ctx, value)[0]


def _material_and_filament(ctx, m) -> tuple[Material, FilamentUse | None]:
    """The `Material` of the block and, with `material.filament`, the filament and the source of each number.

    A key that the YAML gives wins (source `yaml`). The filament gives `E` (the lower of the XY and Z modulus of its sheet, or of its bending
    modulus when the sheet gives no tensile one), `density`, and `nu` when the profile has one; a number that the profile does not have is
    labelled `assumed`: `nu` stays at the default of `Material`, and a missing density is the median of the shipped rows of the same family
    that state one (the default of `Material` when no such row exists). A sheet that gives a density range gives the middle of it.
    A filament with no tensile and no bending modulus is an error when the YAML gives no `E`: a default E would change the result by a wrong factor.
    """
    material = _material_block(ctx, m)
    if "filament" not in m:
        return material, None
    profile, catalog = _filament_found(ctx, m["filament"])
    own, d = f"filament:{profile.id}", Material()
    keys: list[tuple[str, float, str, str | None]] = []
    if "E" in m:
        keys.append(("E", material.E, SOURCE_YAML, None))
    else:
        found = materials_catalog.modulus(profile)
        if found is None:
            raise ctx.err(materials_catalog.no_modulus_text(profile))
        material = replace(material, E=_num(ctx, found[0], "material.E", lo=0.0, lo_open=True))
        keys.append(("E", material.E, own, found[1]))
    if "nu" in m:
        keys.append(("nu", material.nu, SOURCE_YAML, None))
    elif profile.nu is not None:
        material = replace(material, nu=profile.nu)
        keys.append(("nu", material.nu, own, "the value of the profile"))
    else:
        keys.append(("nu", d.nu, SOURCE_ASSUMED, NU_ASSUMED_REASON))
    if "density" in m:
        keys.append(("density", material.density, SOURCE_YAML, None))
    elif (sheet := materials_catalog.sheet_density(profile)) is not None:
        material = replace(material, density=sheet[0])
        keys.append(("density", material.density, own, sheet[1]))
    elif (family := materials_catalog.family_density(profile, catalog)) is not None:
        material = replace(material, density=family[0])
        keys.append(("density", material.density, SOURCE_ASSUMED, family[1]))
    else:
        keys.append(("density", d.density, SOURCE_ASSUMED, "the catalog has no density for this filament; the default of the problem file"))
    return material, FilamentUse(profile, tuple(keys))


def parse_material_block(raw, yaml_path="<yaml>", filament_catalog=None) -> Material:
    """Check a raw `material` mapping with the rules of the problem YAML (the preferences file uses it). Raise `UserError`.

    With `filament`, the numbers are the ones that the filament gives (`_material_and_filament`). `filament_catalog`
    (`topopt.materials_catalog.Catalog`) is the catalog that `filament` is searched in; the default is the shipped catalog with the user
    filaments of the work folder."""
    return _material_and_filament(_Ctx(Path(yaml_path), filament_catalog=filament_catalog), {} if raw is None else raw)[0]


def parse_filament_name(value, yaml_path="<yaml>", filament_catalog=None) -> "materials_catalog.FilamentProfile":
    """The filament that a name, an alias or an id gives (the resolver of `material.filament`, with no numbers taken from it). Raise `UserError`
    (`FilamentNotFound` for a text that is not one filament). The preferences file uses it."""
    return _filament_profile(_Ctx(Path(yaml_path), filament_catalog=filament_catalog), value)


def parse_filament_block(raw, yaml_path="<yaml>", filament_catalog=None) -> tuple[Material, FilamentUse | None]:
    """Like `parse_material_block`, and also the `FilamentUse` (None without `filament`): for tools that show where each number comes from."""
    return _material_and_filament(_Ctx(Path(yaml_path), filament_catalog=filament_catalog), {} if raw is None else raw)


def _voxel_size(ctx, value, where="settings.voxel_size") -> float:
    """The one rule of a voxel size in mm: a number above 0. The loader and `parse_voxel_size` (the preferences file) both call it."""
    return _num(ctx, value, where, lo=0.0, lo_open=True)


def parse_voxel_size(value, yaml_path="<yaml>", where="settings.voxel_size") -> float:
    """Check a voxel size in mm with the rule of `settings.voxel_size` (a number above 0). `where` names the key in the error."""
    return _voxel_size(_Ctx(Path(yaml_path)), value, where)


def _acceptance_block(ctx, raw, case_names, has_mirror) -> Acceptance | None:
    """Read the optional `acceptance` block (CONTRACTS.md section 36). Unknown keys and unknown rules are errors."""
    if raw is None:
        return None
    _check_keys(ctx, raw, "acceptance", ACCEPTANCE_REQUIRED, ACCEPTANCE_OPTIONAL)
    mesh = None
    if "mesh_size_mm" in raw:
        mesh = _num_text(ctx, raw["mesh_size_mm"], "acceptance.mesh_size_mm", lo=0.0, lo_open=True)
    percentile = None
    if "percentile" in raw:
        percentile = _num_text(ctx, raw["percentile"], "acceptance.percentile", lo=0.0, hi=100.0, lo_open=True)
    gate = raw.get("owner_gate", False)
    if not isinstance(gate, bool):
        raise ctx.err(f"'acceptance.owner_gate' must be true or false, got {gate!r}")
    items = raw["rules"]
    if not isinstance(items, list) or not items:
        raise ctx.err("'acceptance.rules' must be a list with one rule or more, for example [{rule: watertight}]")
    rules, seen = [], set()
    for i, item in enumerate(items):
        w = f"acceptance.rules[{i}]"
        if not isinstance(item, dict) or "rule" not in item:
            raise ctx.err(f"'{w}' must be a mapping with the key 'rule' (known rules: {', '.join(ACCEPTANCE_RULES)})")
        name = item["rule"]
        if name not in ACCEPTANCE_RULES:
            raise ctx.err(f"'{w}.rule' is {name!r}, not a rule (known rules: {', '.join(ACCEPTANCE_RULES)})")
        if name in seen:
            raise ctx.err(f"'{w}' lists the rule '{name}' again: one entry for each rule")
        seen.add(name)
        required, optional = _RULE_KEYS[name]
        numbers = [k for k in required + optional if k in _RULE_NUMBERS]
        extra = (["source"] if numbers else []) + [f"{k}_source" for k in numbers]
        body = {k: v for k, v in item.items() if k != "rule"}
        known = ["rule", *optional, *extra]    # `rule` is read before the check, so the list of known keys names it
        _check_keys(ctx, body, w, list(required), known)
        options, sources = [], {}
        for key, value in body.items():
            if key == "source" or key.endswith("_source") and key[:-7] in _RULE_NUMBERS:
                text = _str(ctx, value, f"{w}.{key}")
                if key != "source":
                    sources[key[:-7]] = text
                continue
            if key in _RULE_NUMBERS:
                value = _num_text(ctx, value, f"{w}.{key}", **_RULE_NUMBERS[key])
            elif key in ("must_fall", "relative_to_parent"):
                if not isinstance(value, bool):
                    raise ctx.err(f"'{w}.{key}' must be true or false, got {value!r}")
            elif key == "against_parent":
                if value not in AGAINST_PARENT:
                    raise ctx.err(f"'{w}.against_parent' must be {', '.join(AGAINST_PARENT)}, got {value!r}")
            elif key == "cases":
                value = _row_cases(ctx, value, f"{w}.cases", case_names)
            options.append((key, value))
        for k in sources:
            if k not in body:
                raise ctx.err(f"'{w}.{k}_source' names the number '{k}', and the rule has no '{k}'. Remove the source, "
                              "or give the number")
        if "source" in body:      # the rule-wide source covers each number that has no source of its own
            for k in numbers:
                if k in body and k not in sources:
                    sources[k] = body["source"]
        rule = AcceptanceRule(name, tuple(options), tuple(sources.items()))
        if name == "mass":
            must_fall = rule.get("must_fall", False)
            if must_fall == ("max_rise_pct" in body):
                raise ctx.err(f"'{w}' needs exactly one of must_fall: true (the mass falls) and max_rise_pct "
                              f"(the mass may rise by this many percent); it has "
                              f"{'both' if must_fall else 'neither'}")
        if name == "buckling":
            form = rule.get("against_parent", "absolute")
            if form == "not_worse" and "min_factor" in body:
                raise ctx.err(f"'{w}.min_factor' is the lowest load factor that is allowed, and against_parent: not_worse reads no limit. "
                              "Options: against_parent: cap_or_parent (the factor, or the parent value); or remove min_factor")
            if form != "not_worse" and "min_factor" not in body:
                raise ctx.err(f"'{w}' needs min_factor (the lowest buckling load factor that is allowed), or against_parent: not_worse "
                              "(the factor does not fall below the parent value)")
        if name in ("stress", "displacement", "alignment", "buckling"):
            form = rule.get("against_parent", "absolute")
            if "tolerance_pct" in body and form == "absolute":
                raise ctx.err(f"'{w}.tolerance_pct' is the move to the worse side of the parent value that is allowed, and "
                              "against_parent is 'absolute' (no parent value is read). Options: add against_parent: "
                              "not_worse or cap_or_parent; or remove tolerance_pct")
            if "margin" in body and form == "not_worse":
                raise ctx.err(f"'{w}.margin' is a fraction of the stress limit, and against_parent: not_worse reads no limit. "
                              "Options: against_parent: cap_or_parent (the limit and the parent); or remove margin")
        if name == "symmetry" and not has_mirror:
            raise ctx.err(f"'{w}' is the symmetry rule, and the YAML has no top-level 'mirror'. The rule reads the "
                          "planes from it. Options: add a 'mirror' block, or leave the rule out")
        rules.append(rule)
    return Acceptance(tuple(rules), mesh, percentile, gate)


def _interfaces_block(ctx, raw, base: Path) -> tuple[Interface, ...]:
    """Read the optional `interfaces` list (CONTRACTS.md section 23). Unknown keys are an error."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ctx.err(f"'interfaces' must be a list of mappings, got {raw!r}")
    out, seen = [], set()
    for i, e in enumerate(raw):
        w = f"interfaces[{i}]"
        if not isinstance(e, dict):
            raise ctx.err(f"'{w}' must be a mapping, got {e!r}")
        kind = e.get("kind")
        if kind not in INTERFACE_KINDS:
            raise ctx.err(f"'{w}.kind' must be one of {', '.join(INTERFACE_KINDS)}, got {kind!r}")
        full = e.get("full_surface", False)
        if "full_surface" in e:
            if not isinstance(full, bool):
                raise ctx.err(f"'{w}.full_surface' must be true or false, got {full!r}")
            if full and kind == "stl":
                raise ctx.err(f"'{w}.full_surface' is true on an STL solid: the key is for a bore, boss, tube, or a face with 'radius_mm'. "
                              "An STL solid has no surface that the tool can measure. Options: remove 'full_surface'; or use a bore, boss or tube")
            if full and kind == "face" and "radius_mm" not in e:
                raise ctx.err(f"'{w}.full_surface' is true on a face with no 'radius_mm': a face without 'radius_mm' cuts a half-space, "
                              "so it has no complete surface to measure. Options: add 'radius_mm' (a bounded face, a disc); or remove 'full_surface'")
            e = {k: v for k, v in e.items() if not (k == "full_surface" and kind == "stl")}   # `false` on an STL solid is a no-op
        req, opt = INTERFACE_KEYS[kind]
        if kind == "stl" and "stock_mm" in e:
            raise ctx.err(f"'{w}.stock_mm': an STL solid has no stock (it is an exact union or cut only). "
                          "Use a bore, boss, tube or face if you need stock")
        for k in e:
            if k not in req and k not in opt:
                raise ctx.err(f"unknown key '{k}' in '{w}' (kind '{kind}' takes: {', '.join(req + opt)}). {LEVERS_HELP}")
        for k in req:
            if k not in e:
                raise ctx.err(f"missing key '{k}' in '{w}' (kind '{kind}')")
        name = _str(ctx, e["name"], f"{w}.name")
        if name in seen:
            raise ctx.err(f"duplicate interface name '{name}'")
        seen.add(name)
        if kind == "stl":
            op = e["op"]
            if op not in INTERFACE_OPS:
                raise ctx.err(f"'{w}.op' must be union or cut, got {op!r}")
            out.append(Interface(name, kind, op, 0.0, path=_stl(ctx, e["file"], f"{w}.file", base)))
            continue
        stock = _num(ctx, e.get("stock_mm", 0.0), f"{w}.stock_mm", lo=0.0)
        point = _vec3(ctx, e["point"], f"{w}.point")
        if kind == "face":
            bound = _num(ctx, e["radius_mm"], f"{w}.radius_mm", lo=0.0, lo_open=True) if "radius_mm" in e else None
            out.append(Interface(name, kind, "cut", stock, point, _vec3(ctx, e["normal"], f"{w}.normal", unit=True), bound,
                                 full_surface=full))
            continue
        axis = _vec3(ctx, e["axis"], f"{w}.axis", unit=True)
        radius = _num(ctx, e["radius_mm"], f"{w}.radius_mm", lo=0.0, lo_open=True)
        length = _num(ctx, e["length_mm"], f"{w}.length_mm", lo=0.0, lo_open=True)
        inner = None
        if kind == "bore" and stock >= radius:
            raise ctx.err(f"'{w}.stock_mm' is {stock}, not less than radius_mm {radius}: the as-printed bore would have no hole")
        if kind == "tube":
            inner = _num(ctx, e["inner_radius_mm"], f"{w}.inner_radius_mm", lo=0.0, lo_open=True)
            if inner >= radius:
                raise ctx.err(f"'{w}.inner_radius_mm' is {inner}, not less than radius_mm {radius}")
            if stock >= inner:
                raise ctx.err(f"'{w}.stock_mm' is {stock}, not less than inner_radius_mm {inner}: the as-printed tube would have no hole")
        out.append(Interface(name, kind, _KIND_OP[kind], stock, point, axis, radius, inner, length, full_surface=full))
    return tuple(out)


def parse_interfaces_block(raw, yaml_path="<yaml>") -> tuple[Interface, ...]:
    """Check a raw `interfaces` list (for tools that read the YAML themselves). Raise `UserError`."""
    return _interfaces_block(_Ctx(Path(yaml_path)), raw, Path(yaml_path).parent)


# --- bore_checks parser ---------------------------------------------------


_BC_SEE = "see CONTRACTS.md section 40.2"


def _bc_keys(ctx, d, where, required, optional):
    """`_check_keys` with the known keys in the text of an unknown key, and an option in every text (the facts and the options of CONTRACTS.md section 40.2)."""
    if not isinstance(d, dict):
        raise ctx.err(f"'{where}' must be a mapping, got {d!r}. Options: write '{where}' as lines of 'key: value'; {_BC_SEE}")
    for k in d:
        if k not in required and k not in optional:
            raise ctx.err(f"unknown key '{k}' in '{where}'. Known keys: {', '.join(list(required) + list(optional)) or 'none'}. "
                          f"Options: use one of the known keys, or remove '{k}'; {_BC_SEE}. {LEVERS_HELP}")
    for k in required:
        if k not in d:
            raise ctx.err(f"missing key '{k}' in '{where}'. Options: add the key '{k}'; {_BC_SEE}")


class _Bare:
    """A context with no file name in its errors. The block `bore_checks` calls the shared checks (`_num`, `_str`, `_vec3`) with it, adds an option to
    the text, and gives the text to the real context, which adds the file. The shared checks, and the texts of the other YAML blocks, do not change."""

    @staticmethod
    def err(msg):
        return UserError(msg)


def _bc_option(ctx, check, *args, option, **kw):
    try:
        return check(_Bare, *args, **kw)
    except UserError as exc:
        raise ctx.err(f"{exc}. Options: {option}; {_BC_SEE}") from None


def _bc_num(ctx, v, where, **kw):
    """`_num_text` for the block. A wrong value gets an option that names the range in words (the numbers come from the call)."""
    rng = []
    if kw.get("lo") is not None:
        rng.append(f"{'above' if kw.get('lo_open') else 'at least'} {kw['lo']:g}")
    if kw.get("hi") is not None:
        rng.append(f"{'below' if kw.get('hi_open') else 'at most'} {kw['hi']:g}")
    return _bc_option(ctx, _num_text, v, where, option=f"give a number{' ' + ' and '.join(rng) if rng else ''} (a text such as 1.0e-3 is read as a number)", **kw)


def _bc_str(ctx, v, where, what):
    """`_str` for the block: `what` is named in the option."""
    return _bc_option(ctx, _str, v, where, option=f"give {what} as a non-empty text")


def _bc_vec3(ctx, v, where, what, *, unit=False):
    """`_vec3` for the block: `what` is named in the option."""
    return _bc_option(ctx, _vec3, v, where, option=f"give {what}", unit=unit)


def _bc_sources(ctx, e, where, have, need):
    """The source texts of the numbers of one entry (the D7 pattern: a number `K` takes `K_source`; `source` covers the numbers with no source of their own).

    `have` are the number keys that the entry gives, `need` the keys among them that are physical or process data: they need a source (E7).
    A `K_source` for a number that the entry does not give is an error (E12). Return `((key, text), ...)` in the order of `have`."""
    entry = _bc_str(ctx, e["source"], f"{where}.source", "the source of the numbers of the entry") if "source" in e else None
    for k in e:
        if isinstance(k, str) and k.endswith("_source") and k[:-7] not in have:
            raise ctx.err(f"'{where}.{k}' names the number '{k[:-7]}', and the entry does not give it. "
                          f"Options: remove the source, or give the number")
    out = []
    for k in have:
        own = _bc_str(ctx, e[f"{k}_source"], f"{where}.{k}_source", f"the source of '{k}'") if f"{k}_source" in e else None
        text = own if own is not None else entry
        if text is None:
            if k in need:
                raise ctx.err(f"'{where}.{k}' has no source. A physical or process number needs its source text "
                              f"(decision D7: topopt has no presets, no defaults and applies no factors). "
                              f"Options: add '{k}_source: <text>' to the entry, or add one 'source: <text>' that covers every number "
                              f"of the entry that has no source of its own, or remove the number")
            continue
        out.append((k, text))
    return tuple(out)


def _bc_knobs(ctx, b) -> BoreKnobs:
    bounds = {"radius_tolerance_mm": dict(lo=0.0, lo_open=True), "end_margin_mm": dict(lo=0.0),
              "normal_limit": dict(lo=0.0, hi=1.0, lo_open=True), "sample_band_mm": dict(lo=0.0, lo_open=True)}
    d = BoreKnobs()
    vals, given = {}, []
    for k in BORE_KNOB_KEYS:
        if k in b:
            vals[k] = _bc_num(ctx, b[k], f"bore_checks.{k}", **bounds[k])
            given.append(k)
        else:
            vals[k] = getattr(d, k)
    sources = []
    for k in BORE_KNOB_KEYS:
        if f"{k}_source" in b:
            if k not in b:
                raise ctx.err(f"'bore_checks.{k}_source' names the knob '{k}', and the block does not give it. "
                              f"Options: remove the source, or give the knob (the default is {getattr(d, k)})")
            sources.append((k, _bc_str(ctx, b[f"{k}_source"], f"bore_checks.{k}_source", f"the source of the knob '{k}'")))
    return BoreKnobs(**vals, given=tuple(given), sources=tuple(sources))


def _bc_bore(ctx, e, w, interfaces, knobs) -> BoreSpec:
    geom = BORE_GEOMETRY_KEYS
    _bc_keys(ctx, e, w, BORE_REQUIRED, BORE_OPTIONAL)
    name = _bc_str(ctx, e["name"], f"{w}.name", "the name of the bore")
    if "interface" in e:
        extra = [k for k in geom if k in e]
        if extra:
            raise ctx.err(f"'{w}' gives 'interface' and also {', '.join(extra)}: the bore has two sources of its geometry. "
                          f"Options: give the four keys point, axis, radius_mm and length_mm, or give interface only")
        iname = _bc_str(ctx, e["interface"], f"{w}.interface", "the name of a bore entry of 'interfaces'")
        hit = [i for i in interfaces if i.name == iname]
        if not hit or hit[0].kind != "bore":
            known = ", ".join(f"{i.name} ({i.kind})" for i in interfaces) or "none: the YAML has no 'interfaces'"
            raise ctx.err(f"'{w}.interface' is {iname!r}, not a bore entry of 'interfaces' (known: {known}). "
                          f"Options: give the name of a bore entry, or give the four keys point, axis, radius_mm and length_mm")
        i = hit[0]
        point, axis, radius, length, geometry_from = i.point, i.axis, i.radius_mm, i.length_mm, f"interface:{iname}"
        have = []
    else:
        missing = [k for k in geom if k not in e]
        if missing:
            raise ctx.err(f"'{w}' needs 'interface' or the four keys point, axis, radius_mm and length_mm; missing: {', '.join(missing)}. "
                          f"Options: give the missing keys, or give interface (the name of a bore entry of 'interfaces')")
        point = _bc_vec3(ctx, e["point"], f"{w}.point", "three numbers [x, y, z] in mm")
        axis = _bc_vec3(ctx, e["axis"], f"{w}.axis", "three numbers [x, y, z] that are not all 0 (the vector is made a unit vector)", unit=True)
        radius = _bc_num(ctx, e["radius_mm"], f"{w}.radius_mm", lo=0.0, lo_open=True)
        length = _bc_num(ctx, e["length_mm"], f"{w}.length_mm", lo=0.0, lo_open=True)
        geometry_from = "yaml"
        have = ["radius_mm", "length_mm"]
    if length <= 2.0 * knobs.end_margin_mm:
        raise ctx.err(f"'{w}' has length_mm {length} and the end margin {knobs.end_margin_mm} mm at each end, so the contact range is empty "
                      f"(length_mm must be above 2 x end_margin_mm). Options: a longer length_mm, or a smaller bore_checks.end_margin_mm")
    interference = mean = None
    if "diametral_interference_mm" in e:
        interference = _bc_num(ctx, e["diametral_interference_mm"], f"{w}.diametral_interference_mm", lo=0.0, lo_open=True)
        have.append("diametral_interference_mm")
    if "mean_stress_mpa" in e:
        mean = _bc_num(ctx, e["mean_stress_mpa"], f"{w}.mean_stress_mpa")
        have.append("mean_stress_mpa")
    sources = _bc_sources(ctx, e, w, have, ("diametral_interference_mm", "mean_stress_mpa"))
    return BoreSpec(name, point, axis, radius, length, geometry_from, interference, mean, sources)


def _bc_materials(ctx, raw, where) -> tuple[GoodmanMaterial, ...]:
    _bc_keys(ctx, raw, where, GOODMAN_REQUIRED, [])
    mats = raw["materials"]
    if not isinstance(mats, dict) or not mats:
        raise ctx.err(f"'{where}.materials' must be a mapping of a material label to its strengths, with one entry or more, got {mats!r}. "
                      f"Options: give one entry for each material, 'label: {{su_mpa: ..., sy_mpa: ..., se_mpa: ..., source: ...}}'; {_BC_SEE}")
    keys = GOODMAN_STRENGTH_KEYS
    out = []
    for label, m in mats.items():
        label = _bc_str(ctx, label, f"{where}.materials label", "the label of the material")
        w = f"{where}.materials.{label}"
        _bc_keys(ctx, m, w, [], GOODMAN_MATERIAL_OPTIONAL)
        missing = [k for k in keys if k not in m]
        if missing:
            raise ctx.err(f"'{w}' needs all three strengths su_mpa, sy_mpa and se_mpa; missing: {', '.join(missing)}. "
                          f"Options: give all three, each with its source (se_mpa is the fully corrected endurance limit; topopt applies no factor)")
        vals = {k: _bc_num(ctx, m[k], f"{w}.{k}", lo=0.0, lo_open=True) for k in keys}
        out.append(GoodmanMaterial(label, **vals, sources=_bc_sources(ctx, m, w, list(keys), keys)))
    return tuple(out)


def _bc_cups(ctx, raw, where, bores) -> tuple[Cup, ...]:
    _bc_keys(ctx, raw, where, PRESS_FIT_REQUIRED, [])
    cl = raw["cups"]
    if not isinstance(cl, list) or not cl:
        raise ctx.err(f"'{where}.cups' must be a list of cup variants, with one entry or more, got {cl!r}. "
                      f"Options: give one entry for each cup variant, '- {{name: ..., k_mpa_per_mm: ..., source: ...}}'; {_BC_SEE}")
    out, seen = [], set()
    for i, c in enumerate(cl):
        w = f"{where}.cups[{i}]"
        _bc_keys(ctx, c, w, CUP_REQUIRED, CUP_OPTIONAL)
        name = _bc_str(ctx, c["name"], f"{w}.name", "the name of the cup variant")
        if name in seen:
            raise ctx.err(f"duplicate cup name '{name}' in '{where}.cups'. Options: one name for each cup variant")
        seen.add(name)
        lame = [k for k in ("e_mpa", "nu", "wall_mm") if k in c]
        if "k_mpa_per_mm" in c and lame:
            raise ctx.err(f"'{w}' gives k_mpa_per_mm and also {', '.join(lame)}: a cup has one form. "
                          f"Options: give k_mpa_per_mm, or give all three of e_mpa, nu and wall_mm")
        if "k_mpa_per_mm" not in c and len(lame) < 3:
            missing = [k for k in ("e_mpa", "nu", "wall_mm") if k not in c]
            raise ctx.err(f"'{w}' needs k_mpa_per_mm, or all three of e_mpa, nu and wall_mm; "
                          f"it gives {', '.join(lame) or 'none of them'}, missing: {', '.join(missing)}. "
                          f"Options: give k_mpa_per_mm, or give e_mpa, nu and wall_mm")
        vals, have = {}, []
        if "k_mpa_per_mm" in c:
            vals["k_mpa_per_mm"] = _bc_num(ctx, c["k_mpa_per_mm"], f"{w}.k_mpa_per_mm", lo=0.0, lo_open=True)
            have.append("k_mpa_per_mm")
        else:
            vals["e_mpa"] = _bc_num(ctx, c["e_mpa"], f"{w}.e_mpa", lo=0.0, lo_open=True)
            vals["nu"] = _bc_num(ctx, c["nu"], f"{w}.nu", lo=0.0, hi=0.5, hi_open=True)
            vals["wall_mm"] = _bc_num(ctx, c["wall_mm"], f"{w}.wall_mm", lo=0.0, lo_open=True)
            have += ["e_mpa", "nu", "wall_mm"]
            for b in bores:
                if vals["wall_mm"] > b.radius_mm:
                    raise ctx.err(f"'{w}.wall_mm' is {vals['wall_mm']} and the bore '{b.name}' has radius_mm {b.radius_mm}: "
                                  f"the cup wall is thicker than the bore radius. Options: a wall_mm of at most the bore radius, or give k_mpa_per_mm")
        out.append(Cup(name, **vals, sources=_bc_sources(ctx, c, w, have, have)))
    return tuple(out)


def _bore_checks(ctx, raw, interfaces) -> BoreChecks:
    """Read the optional `bore_checks` block (CONTRACTS.md section 40). Unknown keys are an error."""
    where = "bore_checks"
    _bc_keys(ctx, raw, where, BORE_CHECKS_REQUIRED, BORE_CHECKS_OPTIONAL)
    if "goodman" not in raw and "press_fit" not in raw:
        raise ctx.err(f"'{where}' has neither 'goodman' nor 'press_fit': the block would compute nothing. "
                      f"Options: add a 'goodman' block (the Goodman check) or a 'press_fit' block, or remove 'bore_checks'")
    knobs = _bc_knobs(ctx, raw)
    bl = raw["bores"]
    if not isinstance(bl, list) or not bl:
        raise ctx.err(f"'{where}.bores' must be a list with one bore or more, got {bl!r}. Options: one entry for each bore")
    bores, seen = [], set()
    for i, e in enumerate(bl):
        b = _bc_bore(ctx, e, f"{where}.bores[{i}]", interfaces, knobs)
        if b.name in seen:
            raise ctx.err(f"duplicate bore name '{b.name}' in '{where}.bores'. Options: one name for each bore")
        seen.add(b.name)
        bores.append(b)
    press = "press_fit" in raw
    for b in bores:
        if press and b.diametral_interference_mm is None:
            raise ctx.err(f"'{where}.bores' entry '{b.name}' has no diametral_interference_mm, and the block has 'press_fit'. "
                          f"Options: give diametral_interference_mm with its source, or remove 'press_fit'")
        if not press and b.diametral_interference_mm is not None:
            raise ctx.err(f"'{where}.bores' entry '{b.name}' gives diametral_interference_mm, and the block has no 'press_fit'. "
                          f"Options: add a 'press_fit' block, or remove the key")
        if press and b.mean_stress_mpa is not None:
            raise ctx.err(f"'{where}.bores' entry '{b.name}' gives mean_stress_mpa, and the block has 'press_fit': the press fit gives the mean stress, "
                          f"so there would be two sources of it. Options: remove mean_stress_mpa, or remove 'press_fit'")
    mats = _bc_materials(ctx, raw["goodman"], f"{where}.goodman") if "goodman" in raw else ()
    cups = _bc_cups(ctx, raw["press_fit"], f"{where}.press_fit", bores) if press else ()
    return BoreChecks(tuple(bores), knobs, mats, cups, goodman="goodman" in raw, press_fit=press)


def parse_bore_checks_block(raw, interfaces=(), yaml_path="<yaml>") -> BoreChecks:
    """Check a raw `bore_checks` mapping (for tools that read the YAML themselves). `interfaces` are the parsed `interfaces`. Raise `UserError`."""
    return _bore_checks(_Ctx(Path(yaml_path)), raw, tuple(interfaces))


_ROTATIONAL = ("rot", "moment", "torsion", "twist", "bending", "angular", "krx", "kry", "krz")


def _spring_entry(ctx, e, ew, load_names):
    """One entry of `springs` (section 35): a load region name and its stiffness, total or per area."""
    rot = (f"is a rotational spring. A spring has force stiffness only: [kx, ky, kz] in N/mm. Options: model the "
           "rotational stiffness with two force springs on two load regions at a lever arm (a pair of springs k at "
           "a distance d gives k d^2 / 2 for each axis of the pair), or wait for a later beam support")
    if isinstance(e, dict):
        for k in e:
            if isinstance(k, str) and any(w in k.lower() for w in _ROTATIONAL):
                raise ctx.err(f"'{ew}.{k}' {rot}")
    _check_keys(ctx, e, ew, SPRING_REQUIRED, SPRING_OPTIONAL)
    reg = _str(ctx, e["region"], f"{ew}.region")
    if reg not in load_names:
        raise ctx.err(f"'{ew}.region' is '{reg}', not a load region (known: {', '.join(sorted(load_names))}). "
                      "A spring acts on the exposed faces of a name of regions.load. "
                      "Options: add the body that carries the spring to regions.load")
    if ("stiffness" in e) == ("stiffness_per_area" in e):
        raise ctx.err(f"'{ew}' needs exactly one of 'stiffness' (N/mm, the total for the region) and "
                      f"'stiffness_per_area' (N/mm3); it has {'both' if 'stiffness' in e else 'neither'}")
    key = "stiffness" if "stiffness" in e else "stiffness_per_area"
    v = e[key]
    if key == "stiffness_per_area" and isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v, v, v]         # one number: the same on all axes
    if isinstance(v, list) and len(v) == 6:
        raise ctx.err(f"'{ew}.{key}' has 6 numbers: it {rot}")
    k = _vec3(ctx, v, f"{ew}.{key}")
    if min(k) < 0.0:
        raise ctx.err(f"'{ew}.{key}' must not be negative, got {list(k)}")
    if max(k) == 0.0:
        raise ctx.err(f"'{ew}.{key}' is all zero: the spring has no stiffness. Remove the entry, or give a stiffness")
    return SpringSpec(reg, k, None) if key == "stiffness" else SpringSpec(reg, None, k)


_SECTION_FORMS = ("{kind: round, d}, {kind: tube, od, id}, {kind: rect, b, h} or {kind: props, A, Iy, Iz, J} "
                  "(mm, mm2, mm4)")


def _section(ctx, v, where) -> SectionSpec:
    """The beam section of a joint: `kind` and its values, all above zero (a tube needs `id` below `od`)."""
    if not isinstance(v, dict) or not isinstance(v.get("kind"), str) or v["kind"] not in SECTION_KEYS:
        raise ctx.err(f"'{where}' must be a mapping with a kind: {_SECTION_FORMS}; got {v!r}")
    kind = v["kind"]
    keys = SECTION_KEYS[kind]
    _check_keys(ctx, v, where, SECTION_REQUIRED[kind], [])
    vals = tuple(_num_text(ctx, v[k], f"{where}.{k}", lo=0.0, lo_open=True) for k in keys)
    if kind == "tube" and vals[1] >= vals[0]:
        raise ctx.err(f"'{where}.id' is {vals[1]:g}, not less than od {vals[0]:g}: the tube would have no wall")
    return SectionSpec(kind, vals)


JOINT_AXIS_OPTIONS = "Options: move 'to' or 'origin', or give another 'orient'"     # the one text of both places that find the problem


def joint_axis_problem(origin, to, orient) -> str | None:
    """The fact when the beam from `origin` to `to` has no direction, or `orient` lies along it. Else None."""
    d = [float(b) - float(a) for a, b in zip(origin, to)]
    length = math.sqrt(sum(c * c for c in d))
    if length < DEGENERATE:
        return (f"the far point {list(to)} is the reference point {[float(c) for c in origin]}: the beam has length "
                f"{length:g} mm and no direction")
    if orient is not None:
        o = [float(c) for c in orient]
        cross = [d[1] * o[2] - d[2] * o[1], d[2] * o[0] - d[0] * o[2], d[0] * o[1] - d[1] * o[0]]
        on = math.sqrt(sum(c * c for c in o))
        if math.sqrt(sum(c * c for c in cross)) < DEGENERATE * length * on:
            return (f"'orient' {[float(c) for c in orient]} is parallel to the beam axis {[c / length for c in d]}: "
                    "it cannot give the section y axis")
    return None


def _joint_entry(ctx, e, ew, load_names) -> JointSpec:
    """One entry of `joints` (section 38): a load region, the far point, the beam section and the optional keys."""
    _check_keys(ctx, e, ew, JOINT_REQUIRED, JOINT_OPTIONAL)
    reg = _str(ctx, e["region"], f"{ew}.region")
    if reg not in load_names:
        raise ctx.err(f"'{ew}.region' is '{reg}', not a load region (known: {', '.join(sorted(load_names))}). "
                      "A joint bonds the exposed faces of a name of regions.load to a beam. "
                      "Options: add the body that carries the joint to regions.load")
    to = _vec3(ctx, e["to"], f"{ew}.to")
    origin = _vec3(ctx, e["origin"], f"{ew}.origin") if "origin" in e else None
    orient = _vec3(ctx, e["orient"], f"{ew}.orient", unit=True) if "orient" in e else None
    section = _section(ctx, e["section"], f"{ew}.section")
    bond = _num_text(ctx, e["bond_stiffness_per_area"], f"{ew}.bond_stiffness_per_area", lo=0.0, lo_open=True) \
        if "bond_stiffness_per_area" in e else None
    E = _num(ctx, e["E"], f"{ew}.E", lo=0.0, lo_open=True) if "E" in e else None
    nu = _num(ctx, e["nu"], f"{ew}.nu", lo=-1.0, hi=0.5, lo_open=True, hi_open=True) if "nu" in e else None
    if origin is not None:      # without 'origin' the reference point comes from the faces: the problem builder checks then
        bad = joint_axis_problem(origin, to, orient)
        if bad:
            raise ctx.err(f"'{ew}': {bad}. {JOINT_AXIS_OPTIONS}")
    return JointSpec(reg, to, section, origin, orient, bond, E, nu)


def _not_printed(ctx, raw, regions: Regions, base: Path) -> tuple[NotPrinted, ...]:
    """`regions.not_printed`: region files that are in the problem but not in the print (CONTRACTS.md section 9.1).

    Each entry is a file of the `fixed`, `load`, `solid` or `keep_out` regions. It is matched by its RESOLVED path (the way `_stl`
    resolves it), not by its file name, so two files with one name in two folders stay two files. An entry that matches no
    region file is an error that names the entry and the region files. The same file twice in the list counts once.
    """
    paths = _stl_list(ctx, raw, "regions.not_printed", base, required=False)
    held = ([("fixed", p.name, p) for p in regions.fixed] + [("load", lr.name, lr.path) for lr in regions.load]
            + [("solid", p.name, p) for p in regions.solid] + [("keep_out", p.name, p) for p in regions.keep_out])

    def shown(q: Path) -> str:
        try:
            return q.relative_to(base.resolve()).as_posix()
        except ValueError:
            return str(q)

    out: list[NotPrinted] = []
    for i, path in enumerate(paths):
        hits = [NotPrinted(role, name, q) for role, name, q in held if q == path]
        if not hits:
            files = "; ".join(f"{role}: " + ", ".join(shown(q) for r2, _n, q in held if r2 == role)
                              for role in NOT_PRINTED_ROLES if any(r2 == role for r2, _n, _q in held))
            what = ("That is the design region: the part itself" if path == regions.design
                    else "It is not a file of any region")
            raise ctx.err(f"'regions.not_printed[{i}]' is {shown(path)}. {what}. Give a file that the fixed, load, solid or "
                          f"keep_out regions name (the match is by resolved path). Region files: {files}")
        out += [h for h in hits if h not in out]
    return tuple(out)


# --- main loader ---------------------------------------------------------


def _refuse_part_spec(ctx: "_Ctx", data) -> None:
    """Say that the file is a part spec for `build_regions` (`regions.design` is a list of shapes), not a problem YAML.

    It runs before the key checks: a part spec has keys of its own (for example `fixed_axes` at the top level), and the
    unknown-key text would hide this one. Anything else that is wrong with the file is left to the checks that follow."""
    regions = data.get("regions") if isinstance(data, dict) else None
    design = regions.get("design") if isinstance(regions, dict) else None
    if isinstance(design, list) and design and all(isinstance(x, dict) for x in design):
        raise ctx.err("'regions.design' is a list of shapes: this looks like a part spec for build_regions, not a problem YAML "
                      "(a problem YAML names an STL file there). build_regions writes the region STL files and the problem YAML "
                      "from the spec: the MCP tool `build_regions`, or `python -m topopt.regions_builder SPEC.yaml --out DIR`. "
                      "Then give the YAML in DIR to topopt")


def load_config(path) -> Config:
    """Read a problem YAML file and return a checked `Config`."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise UserError(f"YAML file not found: {path.resolve()}")
    path = path.resolve()
    ctx = _Ctx(path)
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ctx.err(f"YAML syntax error: {str(e).splitlines()[0] if str(e) else e}")
    base = path.parent

    _refuse_part_spec(ctx, data)
    _check_keys(ctx, data, "top level", TOP_REQUIRED, TOP_OPTIONAL)
    name = _str(ctx, data["name"], "name")

    # assumptions: a list of short strings, for people. The solver ignores them.
    asm = data.get("assumptions")
    asm = [] if asm is None else asm
    if not isinstance(asm, list) or not all(isinstance(x, str) and x.strip() for x in asm):
        raise ctx.err("'assumptions' must be a list of non-empty strings")

    # regions
    r = data["regions"]
    with_springs = bool(data.get("springs")) or bool(data.get("joints"))   # a spring and a joint are supports: `fixed` may be missing or empty
    if with_springs:
        _check_keys(ctx, r, "regions", REGIONS_SUPPORTED_REQUIRED, REGIONS_SUPPORTED_OPTIONAL)
    else:
        _check_keys(ctx, r, "regions", REGIONS_REQUIRED, REGIONS_OPTIONAL, {"fixed": _NO_SUPPORT})
    design = _stl(ctx, r["design"], "regions.design", base)
    fixed, fixed_axes = _fixed_list(ctx, r.get("fixed", []), "regions.fixed", base, required=not with_springs)
    keep_out = _stl_list(ctx, r.get("keep_out", []), "regions.keep_out", base, required=False)
    solid = _stl_list(ctx, r.get("solid", []), "regions.solid", base, required=False)
    load_raw = r["load"]
    if not isinstance(load_raw, dict) or not load_raw:
        raise ctx.err("'regions.load' must be a mapping of name to STL file, with one entry or more")
    load = tuple(
        LoadRegion(_str(ctx, k, "regions.load name"), _stl(ctx, v, f"regions.load.{k}", base))
        for k, v in load_raw.items()
    )
    load_names = {lr.name for lr in load}
    measure = _measure_regions(ctx, r.get("measure"), load_names, base)
    regions = Regions(design, fixed, keep_out, load, solid, fixed_axes=fixed_axes, measure=measure)

    # load cases
    lcs = data["load_cases"]
    if not isinstance(lcs, list) or not lcs:
        raise ctx.err("'load_cases' must be a list with one case or more")
    cases = []
    seen = set()
    for i, c in enumerate(lcs):
        w = f"load_cases[{i}]"
        _check_keys(ctx, c, w, LOAD_CASE_REQUIRED, LOAD_CASE_OPTIONAL)
        if "forces" not in c and "pressures" not in c:
            raise ctx.err(f"missing key 'forces' in '{w}'")
        cname = _str(ctx, c["name"], f"{w}.name")
        if cname in seen:
            raise ctx.err(f"duplicate load case name '{cname}'")
        seen.add(cname)
        weight = _num(ctx, c.get("weight", 1.0), f"{w}.weight", lo=0.0, lo_open=True)
        limit = c.get("compliance_limit")
        if "compliance_limit" in c:
            limit = _num(ctx, _numeric_text(limit), f"{w}.compliance_limit", lo=0.0, lo_open=True)
        stress = {}
        for key in ("stress_limit", "safety_factor", "allowable_mpa"):
            if key in c:
                stress[key] = _num_text(ctx, c[key], f"{w}.{key}", lo=0.0, lo_open=True)
        if "allowable_source" in c:
            stress["allowable_source"] = _str(ctx, c["allowable_source"], f"{w}.allowable_source")
        if "safety_factor" in stress and "stress_limit" in stress:
            raise ctx.err(f"'{w}' has both safety_factor and stress_limit: stress_limit is the final limit and "
                          f"takes no safety_factor. Write allowable_mpa + allowable_source + safety_factor, "
                          f"or a final stress_limit with no safety_factor")
        if "safety_factor" in stress and "allowable_mpa" not in stress:
            raise ctx.err(f"'{w}.safety_factor' needs '{w}.allowable_mpa' (limit = allowable_mpa / safety_factor); "
                          f"or write a final '{w}.stress_limit' with no safety_factor")
        fl = c.get("forces", [])
        pl = c.get("pressures", [])
        if not isinstance(pl, list):
            raise ctx.err(f"'{w}.pressures' must be a list of {{region, value_mpa}} entries")
        if not isinstance(fl, list) or not (fl or pl):
            raise ctx.err(f"'{w}.forces' must be a list with one force or more")
        forces = [_force_entry(ctx, f, f"{w}.forces[{j}]", load_names) for j, f in enumerate(fl)]
        pressures = [_pressure_entry(ctx, p, f"{w}.pressures[{j}]", regions, load_names)
                     for j, p in enumerate(pl)]
        cases.append(LoadCase(cname, weight, tuple(forces), limit, tuple(pressures), **stress))

    # material
    material, filament = _material_and_filament(ctx, data.get("material", {}))

    # settings
    s = data["settings"]
    _check_keys(ctx, s, "settings", SETTINGS_REQUIRED, SETTINGS_OPTIONAL)
    # I7: the allowed material is given by exactly one key. `target_mass_g` needs the voxel grid, so `topopt.problem.build_problem` turns it into
    # `volume_fraction` (CONTRACTS.md section 3). The loader checks the keys and the sign only.
    if "volume_fraction" in s and "target_mass_g" in s:
        raise ctx.err(BOTH_VOLUME_KEYS_TEXT)
    if "volume_fraction" not in s and "target_mass_g" not in s:
        raise ctx.err("missing key 'volume_fraction' or 'target_mass_g' in 'settings': the YAML needs one of them. Options: settings.volume_fraction "
                      "(a number above 0 and up to 1: the mean density of the free design voxels), or settings.target_mass_g (a number above 0: "
                      "the asked mass of the part in grams; it sets the allowed material, and the engine derives the fraction from the voxel grid). Write target_mass_g when "
                      "the user stated a mass in grams")
    ds = Settings(1.0, 0.5, 1, 1.0)
    h = _voxel_size(ctx, s["voxel_size"])
    solver = s.get("solver", ds.solver)
    if solver not in SOLVER_VALUES:
        raise ctx.err(f"'settings.solver' must be auto, cholesky, pardiso, amg or gpu-mg, got {solver!r}")
    gpu_device = s.get("gpu_device", ds.gpu_device)
    if isinstance(gpu_device, bool) or not isinstance(gpu_device, (int, str)) or str(gpu_device).strip() == "":
        raise ctx.err(f"'settings.gpu_device' must be auto, a device number or a part of a device name, "
                      f"got {gpu_device!r}")
    gpu_case = s.get("gpu_case", ds.gpu_case)
    if gpu_case not in GPU_CASES:
        raise ctx.err(f"'settings.gpu_case' must be {', '.join(GPU_CASES)}, got {gpu_case!r}")
    # G2: both are read by gpu-mg only, and checked for any solver (like gpu_case). YAML 1.1 reads 1e-6 as text: _num_text accepts it.
    gpu_tol = _num_text(ctx, s.get("gpu_tol", ds.gpu_tol), "settings.gpu_tol", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    gpu_backward_error = _num_text(ctx, s.get("gpu_backward_error", ds.gpu_backward_error), "settings.gpu_backward_error",
                                   lo=0.0, hi=GPU_BACKWARD_ERROR_LIMIT, hi_open=True)
    # P2b: the cap of the first PCG pass of a gpu-mg solve. An integer: the PCG counts iterations. No upper bound.
    gpu_max_iter = s.get("gpu_max_iter", ds.gpu_max_iter)
    if isinstance(gpu_max_iter, bool) or not isinstance(gpu_max_iter, int) or gpu_max_iter < 1:
        raise ctx.err(f"'settings.gpu_max_iter' must be an integer of 1 or more (the number of PCG iterations of the first pass of a gpu-mg "
                      f"solve; a pass of 0 iterations solves nothing), got {gpu_max_iter!r}")
    case_combine = s.get("case_combine", ds.case_combine)
    if case_combine not in ("sum", "max"):
        raise ctx.err(f"'settings.case_combine' must be sum or max, got {case_combine!r}")
    objective = s.get("objective", ds.objective)
    if objective not in ("compliance", "mass"):
        raise ctx.err(f"'settings.objective' must be compliance or mass, got {objective!r}")
    if objective == "mass" and "target_mass_g" in s:
        raise ctx.err("'settings.target_mass_g' does not go with settings.objective 'mass': that objective minimizes the mass, and there "
                      "volume_fraction is only the start density, so a mass target would set a start density and not the mass of the part. "
                      "Options: remove settings.target_mass_g and write settings.volume_fraction (the start density) with the objective mass; or "
                      "use settings.objective compliance with settings.target_mass_g to set the allowed material from that mass")
    objective_scale = s.get("objective_scale", ds.objective_scale)
    if objective_scale not in OBJECTIVE_SCALES:
        raise ctx.err(f"'settings.objective_scale' must be {' or '.join(OBJECTIVE_SCALES)}, got {objective_scale!r}")
    stress_scale = s.get("stress_scale", ds.stress_scale)
    if stress_scale != "auto":
        text_number = _numeric_text(stress_scale)
        if isinstance(text_number, bool) or not isinstance(text_number, (int, float)):
            raise ctx.err(f"'settings.stress_scale' must be auto or a number > 0, got {stress_scale!r}")
        stress_scale = _num(ctx, text_number, "settings.stress_scale", lo=0.0, lo_open=True)
    step_acceptance = s.get("stress_step_acceptance", ds.stress_step_acceptance)
    if not isinstance(step_acceptance, bool):
        raise ctx.err(f"'settings.stress_step_acceptance' must be true or false, got {step_acceptance!r}")
    rule = s.get("filter_solid_neighbours", ds.filter_solid_neighbours)
    if rule not in ("density_one", "ignore"):
        raise ctx.err(f"'settings.filter_solid_neighbours' must be density_one or ignore, got {rule!r}")
    member = None
    if "min_member_mm" in s:
        member = _num(ctx, s["min_member_mm"], "settings.min_member_mm", lo=0.0, lo_open=True)
    robust = s.get("robust", False)   # YAML 1.1 loads the word off as false and on as true
    if robust is True:
        robust = "eroded"
    elif robust is False:
        robust = "off"
    if robust not in ROBUST_MODES:
        raise ctx.err(f"'settings.robust' must be off, eroded or wall, got {robust!r}")
    perimeter = None
    if "perimeter_max_mm2" in s:   # a number above 0 in mm2; YAML 1.1 reads 1.2e4 as text, which _num_text accepts
        perimeter = _num_text(ctx, s["perimeter_max_mm2"], "settings.perimeter_max_mm2", lo=0.0, lo_open=True)
    part_only_film = None   # P4b-6: a number above 0 in mm, or absent or null (off). No upper bound. YAML 1.1 reads 1e-2 as text: _num_text accepts it
    if s.get("part_only_min_film_mm") is not None:
        try:
            part_only_film = _num_text(_Bare, s["part_only_min_film_mm"], "settings.part_only_min_film_mm", lo=0.0, lo_open=True)
        except UserError as exc:
            raise ctx.err(f"{exc}. It is the wall thickness in mm below which the film at a contact face of result/part_only.stl is cut "
                          "off. Options: write a number above 0 in mm, or remove the key (or write null) to keep the film") from None
    target_mass = None   # I7: grams, a number above 0 (no upper bound: the problem build compares it with the mass of all active voxels). YAML 1.1 reads 3e1 as text: _num_text accepts it
    if "target_mass_g" in s:
        try:
            target_mass = _num_text(_Bare, s["target_mass_g"], "settings.target_mass_g", lo=0.0, lo_open=True)
        except UserError as exc:
            raise ctx.err(f"{exc}. It is the asked mass of the part in grams; it sets the allowed material. Options: write a number above 0 in grams, or remove the key "
                          "and write settings.volume_fraction") from None
    eta_eroded = None
    if "eta_eroded" in s:
        eta_eroded = _num(ctx, s["eta_eroded"], "settings.eta_eroded", lo=ETA_NOMINAL, hi=1.0,
                          lo_open=True, hi_open=True)
    wall_mm = early_print = None
    if robust == "wall":   # section 22b: the wall w is print.min_wall_mm; eta_e defaults to 0.7; R = max(w / L(eta_e), 1.5 voxels)
        early_print = _print_block(ctx, data.get("print"))
        wall_mm = early_print.limit("min_wall_mm")
        if wall_mm is None:
            raise ctx.err("'settings.robust' is wall, and 'print.min_wall_mm' is not set. The wall rule takes the wall w from print.min_wall_mm "
                          "(the wall limit of the print check, in mm). Options: set print.min_wall_mm (for example 1.2); or set settings.robust "
                          "to eroded (no wall rule: settings.filter_radius or settings.min_member_mm sets the size)")
    if "filter_radius" in s:   # an explicit radius wins over the radius that min_member_mm or the wall rule gives
        source, radius = "settings", s["filter_radius"]
    elif wall_mm is not None:
        source, radius = "wall", wall_radius(wall_mm, WALL_ETA_DEFAULT if eta_eroded is None else eta_eroded, h)["radius_mm"]
    elif member is not None:
        source, radius = "min_member_mm", radius_from_min_member(member)
    else:
        source, radius = "default", 1.5 * h
    mesh_method = s.get("mesh_method", ds.mesh_method)
    if mesh_method not in ("sdf", "blur"):
        raise ctx.err(f"'settings.mesh_method' must be sdf or blur, got {mesh_method!r}")
    envelope_fill = s.get("envelope_fill", ds.envelope_fill)
    if envelope_fill is True or envelope_fill is False:   # YAML 1.1 reads an unquoted `on` and `off` as booleans
        envelope_fill = "on" if envelope_fill else "off"
    if envelope_fill not in ENVELOPE_FILL_VALUES:
        raise ctx.err(f"'settings.envelope_fill' must be auto, on or off, got {envelope_fill!r}")
    bs = s.get("beta_schedule", [list(x) for x in ds.beta_schedule])
    if not isinstance(bs, list) or not bs:
        raise ctx.err("'settings.beta_schedule' must be a list of [iteration, beta] pairs")
    sched = []
    for i, p in enumerate(bs):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ctx.err(f"'settings.beta_schedule[{i}]' must be [iteration, beta], got {p!r}")
        it = _num(ctx, p[0], f"settings.beta_schedule[{i}][0]", lo=0, integer=True)
        beta = _num(ctx, p[1], f"settings.beta_schedule[{i}][1]", lo=0.0, lo_open=True)
        if sched and it <= sched[-1][0]:
            raise ctx.err("'settings.beta_schedule' iterations must increase")
        sched.append((it, beta))
    if sched[0][0] != 0:
        raise ctx.err("'settings.beta_schedule' must start at iteration 0")
    settings = Settings(
        voxel_size=h,
        volume_fraction=(_num(ctx, s["volume_fraction"], "settings.volume_fraction", lo=0.0, hi=1.0, lo_open=True)
                         if "volume_fraction" in s else None),   # None with target_mass_g: build_problem derives it
        iterations=_num(ctx, s["iterations"], "settings.iterations", lo=1, integer=True),
        filter_radius=_num(ctx, radius, "settings.filter_radius", lo=0.0, lo_open=True),
        penal=_num(ctx, s.get("penal", ds.penal), "settings.penal", lo=1.0),
        beta_schedule=tuple(sched),
        move_limit=_num(ctx, s.get("move_limit", ds.move_limit), "settings.move_limit",
                        lo=0.0, hi=1.0, lo_open=True),
        solver=solver,
        case_combine=case_combine,
        max_active_voxels=_num(ctx, s.get("max_active_voxels", ds.max_active_voxels),
                               "settings.max_active_voxels", lo=1, integer=True),
        max_grid_voxels=_num(ctx, s.get("max_grid_voxels", ds.max_grid_voxels),
                             "settings.max_grid_voxels", lo=1, integer=True),
        objective=objective,
        stress_q=_num_text(ctx, s.get("stress_q", ds.stress_q), "settings.stress_q", lo=0.0, lo_open=True),
        stress_power=_num_text(ctx, s.get("stress_power", ds.stress_power), "settings.stress_power",
                               lo=0.0, hi=1.0, lo_open=True),
        stress_scale=stress_scale,
        stress_step_acceptance=step_acceptance,
        stop_rules=_stop_rules(ctx, s.get("stop_rules", False)),
        fine_pass=_fine_pass(ctx, s.get("fine_pass")),
        filter_solid_neighbours=rule, min_member_mm=member, filter_radius_source=source,
        robust=robust, eta_eroded=eta_eroded, wall_mm=None if wall_mm is None else float(wall_mm),
        mesh_method=mesh_method,
        envelope_fill=envelope_fill,
        gpu_device=str(gpu_device).strip(),
        gpu_case=gpu_case,
        gpu_tol=gpu_tol,
        gpu_backward_error=gpu_backward_error,
        gpu_max_iter=gpu_max_iter,
        perimeter_max_mm2=perimeter,
        objective_scale=objective_scale,
        part_only_min_film_mm=part_only_film,
        target_mass_g=target_mass,
    )
    # springs (section 35)
    sl = data.get("springs") or []
    if not isinstance(sl, list):
        raise ctx.err("'springs' must be a list of {region, stiffness} entries")
    springs = [_spring_entry(ctx, e, f"springs[{i}]", load_names) for i, e in enumerate(sl)]
    for i, e in enumerate(springs):
        if e.region in {q.region for q in springs[:i]}:
            raise ctx.err(f"'springs[{i}].region' is '{e.region}' again: one entry for each region "
                          "(give the three axes in one stiffness list)")
    # joints (section 38)
    jl = data.get("joints") or []
    if not isinstance(jl, list):
        raise ctx.err("'joints' must be a list of {region, to, section} entries")
    joints = [_joint_entry(ctx, e, f"joints[{i}]", load_names) for i, e in enumerate(jl)]
    for i, e in enumerate(joints):
        if e.region in {q.region for q in joints[:i]}:
            raise ctx.err(f"'joints[{i}].region' is '{e.region}' again: one joint for each region")
    mirror = _mirror_block(ctx, data.get("mirror"))
    limits = _limits(ctx, data.get("limits"), [c.name for c in cases],
                     load_names | {m.name for m in measure})
    print_settings = early_print if early_print is not None else _print_block(ctx, data.get("print"))
    interfaces = _interfaces_block(ctx, data.get("interfaces"), base)
    acceptance = _acceptance_block(ctx, data.get("acceptance"), [c.name for c in cases], bool(mirror))
    bore_checks = _bore_checks(ctx, data["bore_checks"], interfaces) if "bore_checks" in data else None
    not_printed = _not_printed(ctx, r.get("not_printed", []), regions, base)   # P4b-2: after the regions, so that it can match them
    config = Config(name, path, regions, tuple(cases), material, settings, tuple(asm), mirror=mirror, limits=limits,
                    buckling=_buckling(ctx, data.get("buckling", {})), print_settings=print_settings,
                    interfaces=interfaces, springs=tuple(springs), acceptance=acceptance, joints=tuple(joints),
                    bore_checks=bore_checks, not_printed=not_printed, filament=filament)
    from .loads import no_load_refusal   # U21: a case with no load next to a row that cannot work with it (loads imports config, so here)
    refusal = no_load_refusal(config)
    if refusal:
        raise ctx.err(refusal)
    return config
