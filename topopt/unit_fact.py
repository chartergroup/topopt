"""A design space STL that looks like the wrong unit (TC-3, finding #162; CONTRACTS.md section 3, "The unit of the region STLs").

An STL has no unit field. The loader reads the vertex numbers as mm, and the voxel size, the forces and the printer bed are in mm too.
A part drawn in inches loads 25.4 times too small, and a part drawn in metres 1000 times too small. This module gives a fact, never a
refusal: the user or the agent decides.

The rule (one test, no new number): let L be the largest bounding-box size of the design space STL, in the units of the file read as mm.
The fact shows when `L * 25.4` still fits the longer side of the print bed: `L * 25.4 <= max(print.bed_mm)`. The factors 25.4 and 1000 are
unit conversions, not limits. The bed is the one that `topopt check` uses (`print.bed_mm`, else the bed of `print.printer_model`, else the
default of 220 mm per side). The longer side is a choice: a part may be turned on the bed, and the longer side gives the fact in more
cases. A fact that does not apply costs nothing, and the text says that a part that is really this small can ignore it.

Where it shows: `optimize.problem_facts` (so `topopt check`, `topopt check --json` and the start of `run.log`), `cli.check_data` `warnings`
(so `check_problem` and the `check` of `build_regions`), and the three stops of the loader (`voxelize.voxelize`) that a part of the wrong unit
meets before any fact is built: "region ... has no voxel", "grid would have N voxels" and "model has N active voxels" (`unit_hint`).

Reads: the loader keeps the size of the design mesh that it read (`remember_size`), so `topopt check` reads the design STL no extra time. Where no
loader ran before (for example `problem_facts` at the start of a run), `design_size` reads the file once more (`process=False`, bounds only)
and keeps the size for the next call. One file version is kept: its path, modification time and byte size.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

INCH_MM = 25.4       # inch to mm: a unit conversion, not a limit
METRE_MM = 1000.0    # metre to mm: a unit conversion, not a limit

_LAST: tuple[tuple[str, int, int], float] | None = None   # (path, mtime_ns, byte size) of the last design STL, and its largest size


def largest_size(mesh) -> float:
    """The largest bounding-box size of a mesh, in the units of the file."""
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    return float((hi - lo).max())


def _version(path) -> tuple[str, int, int] | None:
    """The version of a file (resolved path, modification time, byte size), or None when it cannot be read."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(Path(path).resolve()), st.st_mtime_ns, st.st_size)


def remember_size(path, mesh) -> float:
    """Keep the largest size of the design mesh that the loader has read, so that `design_size` does not read the file again."""
    global _LAST
    size = largest_size(mesh)
    key = _version(path)
    if key is not None:
        _LAST = (key, size)
    return size


def design_size(config) -> float | None:
    """The largest bounding-box size of the design space STL of a `Config`, or None when the file cannot be read.

    The size of the same file version comes from the last read (the loader's mesh, or an earlier call). Else the file is read here; the
    loader reads it again when it builds the problem and gives the real error there, so a file that cannot be read here gives no fact and
    no error. `process=False` skips the merge of the vertices: the bounds do not need it.
    """
    global _LAST
    import trimesh

    key = _version(config.regions.design)
    if key is not None and _LAST is not None and _LAST[0] == key:
        return _LAST[1]
    try:
        size = largest_size(trimesh.load(config.regions.design, file_type="stl", force="mesh", process=False))
    except Exception:  # noqa: BLE001 - a fact is never an error; the loader names the real problem
        return None
    if key is not None:
        _LAST = (key, size)
    return size


def bed_long_mm(config) -> float:
    """The longer side of the print bed in mm (`print.bed_mm`, the bed of the printer profile, or the default)."""
    return float(max(config.print_settings.bed_mm))


def looks_wrong(size: float | None, bed_long: float) -> bool:
    """True when `size * 25.4` still fits the longer side of the bed. A size that is not a positive number gives False."""
    return size is not None and math.isfinite(size) and size > 0.0 and size * INCH_MM <= bed_long


def _n(v: float) -> str:
    """A size in mm with 4 significant digits, without an exponent."""
    return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}"


def bed_basis(config) -> str:
    """Where the bed size comes from, in words (the key of the YAML, the printer profile, or the default)."""
    ps = config.print_settings
    source = ps.key_source("bed_mm")          # `profile:<id>` when the printer profile filled it
    if source is not None:
        return f"the bed of the printer {source.split(':', 1)[1]}, print.printer_model"
    if "bed_mm" in ps.given:
        return "print.bed_mm of the YAML"
    if ps.printer_model is not None:          # the profile states no bed (`bed_mm: null` in the catalog)
        return f"the default: the profile of the printer {ps.printer_model} states no bed, and print.bed_mm is not set"
    return "the default, print.bed_mm is not set"


def unit_fact(name: str, size: float, bed_long: float, basis: str) -> str:
    """The `fact:` line (one line) for a design space of `size` mm at its largest, in the file `name`."""
    return (f"fact: the largest size of the design space ({name}) is {_n(size)} mm, read as mm. An STL has no unit, so the file can be in inches or in "
            f"metres. The size is {_n(size * INCH_MM)} mm after x {INCH_MM:g} (inches) and {_n(size * METRE_MM)} mm after x {METRE_MM:g} (metres). "
            f"This fact shows because {_n(size * INCH_MM)} mm, the size after x {INCH_MM:g}, still fits the longer side of the print bed "
            f"({_n(bed_long)} mm: {basis}). "
            f"options: scale every STL file of the YAML (regions, measure regions, interface solids) by {INCH_MM:g} or by {METRE_MM:g} (or by the "
            f"factor of another unit, for example 10 for cm), and also the lengths in the YAML that use the unit of the files (for example "
            f"interface points and sizes; voxel_size and the bed are read as mm); or keep the files as they are. A part that is really this small "
            f"can ignore this fact.")


def unit_fact_lines(config) -> list[str]:
    """The fact for a `Config` as a list of 0 or 1 lines. The same text goes to `problem_facts` and to the `warnings` of `check_data`."""
    size, bed = design_size(config), bed_long_mm(config)
    if size is None or not looks_wrong(size, bed):
        return []
    return [unit_fact(config.regions.design.name, size, bed, bed_basis(config))]


def unit_hint(config, size: float | None) -> str:
    """One sentence for the three loader stops ("region ... has no voxel", the grid guard, the active-voxel guard), or "" when the rule does
    not apply.

    `size` is the largest size of the design space mesh that the loader has read already. The sentence ends with a full stop.
    """
    if size is None or not looks_wrong(size, bed_long_mm(config)):
        return ""
    return (f"Check the unit of the STL files: the largest size of the design space ({config.regions.design.name}) is {_n(size)} mm read as mm, "
            f"and it is {_n(size * INCH_MM)} mm after x {INCH_MM:g} (inches) or {_n(size * METRE_MM)} mm after x {METRE_MM:g} (metres).")
