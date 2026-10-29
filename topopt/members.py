"""Minimum member size and the filter (CONTRACTS.md section 22a).

`settings.min_member_mm` is the smallest member (wall or rod) that the author wants. The
filter radius follows from it by ONE rule:

    filter_radius = RADIUS_PER_MEMBER * min_member_mm

This is an ESTIMATE, not a guarantee. A density filter and a Heaviside projection do not
set a minimum size. The rule only makes thinner members less likely. Measure the thickness of the final
part (`metrics.json`, the thickness report of the print check) before you rely on it.

This module has no import from the rest of the package, so `config.py` can use it. It loads
numpy only inside the functions that need it: `topopt.cli` must not load numpy at import.
Every function returns facts (numbers and text). None of them stops a run.
"""

from __future__ import annotations

import math

# One rule, one constant. The reason and the measurement are in the plan document.
#
# The reason: at density 0.5 a hat filter keeps a thin WALL of thickness t only when
# t > 0.5328 * R. A thinner wall falls below the projection threshold and vanishes (continuum
# value; the 0.5 level of the centre value of a plate). So the radius at which a wall of the
# requested thickness just survives is R = t / 0.5328 = 1.877 * t. A ROD survives only when its
# diameter is above 0.936 * R, so a rod needs 1.76 * t: the rule removes thin rods too. The
# calibration (docs/a3a-min-member-calibration.json) measured the result on a voxel grid:
# the 5th percentile of the interior thickness is 8 to 22 % below the request in that case
# (see the plan, "Stated error"; the numbers are there and not here, so that they stay in one
# place). R = t is the rod rule (R = 1.068 * t): easier on memory, but its 5th percentile is
# about 40 % below the request. The pipeline had no formula (a grid search, R = 3.75 mm for a
# 4 mm gate, at its own eta_e 0.55). Its value is context only.
RADIUS_PER_MEMBER = 1.877

# A member needs 2 voxels across to carry load and 3 to be reliable (`member_facts`). The fine pass reads the 3 as the voxels across
# that the part needs (`fine_pass.part_voxel`).
MEMBER_ACROSS = 3.0

RULE_TEXT = f"filter_radius = {RADIUS_PER_MEMBER:g} * min_member_mm"
ESTIMATE_NOTE = ("estimate, not a guarantee: a filter and a projection do not set a minimum "
                 "member size. It sets the interior members: a skin at a free face of the design "
                 "space can be about 1 voxel. Measure the thickness of the final part")

# Continuum values of the hat filter at projection level 0.5, as a share of the radius R: a wall
# (plate) survives above WALL_CRITICAL * R and a rod above ROD_CRITICAL * R. At a free face of
# the design space the filter has one side, and a wall survives from half of that value.
WALL_CRITICAL, ROD_CRITICAL, FACE_WALL_CRITICAL = 0.5328, 0.936, 0.2664

# Memory of the density filter (#333). The run applies it as an FFT convolution (`filter.ConvFilter`) and stores no matrix, so its
# memory follows the padded grid of the FFT (each side n + 2 ceil(R), rounded up to a fast FFT length; `filter.fft_shape`), not the
# radius cubed. The transform of the kernel stays (16 B for each complex value, about 8 B a padded cell); the setup and each apply
# (F @ x, FT @ y) need the field, its transform and its inverse for a moment. The arrays of design size come on top: the weight sums
# and b stay, and an apply makes copies of the design values. Measured on Linux on 2026-10-27 (peak RSS above the process, six
# grids of 0.18 M to 8 M cells, R 3 to 12 voxels): 33 to 50 B a padded cell at the peak of the setup or of an apply, 12 to 23 B
# held between applies. A grid that is almost all design, with a small R, measured up to 61.4 B a padded cell (100 x 400 x 100,
# R 1.5, 96 % design); at the same padded grid, each design voxel added 14 B to the peak (200^3, R 1.5, 30 % and 99 % design).
# The estimate uses 56 B a padded cell plus 24 B a design voxel, for the build and for the run, above every measured peak.
# (The sparse matrix that the run built before #333 needed 28 B held and 60 B at its build for each stencil entry: 6.5 GiB held at
# 140,000 design voxels and R 7.5 voxels, where the FFT filter needs about 0.15 GiB.)
FILTER_BYTES_PER_PADDED_CELL = 56.0
FILTER_BYTES_PER_DESIGN_VOXEL = 24.0


# The default filter radius in voxels (`settings.filter_radius` absent and no `min_member_mm`). The wall rule never goes below it.
DEFAULT_RADIUS_VOXELS = 1.5


def _cumulative(t: float) -> float:
    """Integral of the wall kernel k(s) = 1 - 3 s^2 + 2 s^3 (|s| < 1, units of R; the integral of k over s is 1) from 0 to t. Odd in t."""
    a = min(abs(t), 1.0)
    q = a - a ** 3 + 0.5 * a ** 4
    return q if t >= 0 else -q


def nominal_width_per_radius(eta_e: float) -> float:
    """L(eta_e): the width of the NOMINAL design (filtered value 0.5 or more) of the thinnest wall that survives the eroded design, in filter radii.

    Continuum value for the 3D hat kernel w = R - d and an infinite plate (beta to infinity). The thinnest design wall a whose
    centre reaches `eta_e` is `robust.plate_limit_per_radius`. The filtered value at the distance x from the centre of that wall is
    Q(x + a/2) - Q(x - a/2), with Q the integral of the kernel above. The nominal design ends where it falls to 0.5. L is twice that
    distance: 0.406 at eta_e 0.55, 0.789 at 0.7, 0.976 at 0.8 (CONTRACTS.md section 22b). A wall of this width, or more, at the
    nominal design survives the erosion at `eta_e`, so R = w / L(eta_e) aims the nominal wall at w.
    """
    from .robust import plate_limit_per_radius   # not at the top: robust.py is a sibling with no import of this module either
    u = 0.5 * plate_limit_per_radius(float(eta_e))
    lo, hi = 0.0, 1.0 + u
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _cumulative(mid + u) - _cumulative(mid - u) >= 0.5:
            lo = mid
        else:
            hi = mid
    return 2.0 * 0.5 * (lo + hi)


def wall_radius(wall_mm: float, eta_e: float, h: float) -> dict:
    """The filter radius of the wall rule (`settings.robust: wall`): R = max(w / L(eta_e), 1.5 voxels).

    Return the radius and the numbers it comes from. `floor_wins` is true when 1.5 voxels is above w / L: the grid then
    sets the radius, and the nominal wall can be thicker than w.
    """
    L = nominal_width_per_radius(eta_e)
    from_wall = float(wall_mm) / L
    floor = DEFAULT_RADIUS_VOXELS * float(h)
    return {"radius_mm": max(from_wall, floor), "wall_mm": float(wall_mm), "eta_e": float(eta_e), "nominal_width_per_radius": L,
            "radius_from_wall_mm": from_wall, "voxel_floor_mm": floor, "floor_wins": floor > from_wall}


def radius_for_grid(st, h: float) -> float:
    """The filter radius in mm that a `Settings` gives on the voxel `h` (the fine pass asks for another voxel than the YAML).

    `default`: 1.5 voxels. `wall` (`settings.robust: wall`): `wall_radius` of the wall `st.wall_mm` and the eta_e of the settings, so the
    radius follows the grid as the wall rule says: max(w / L, 1.5 voxels). Else (`settings`, `min_member_mm`): the radius in mm, as set.
    """
    if st.filter_radius_source == "default":
        return DEFAULT_RADIUS_VOXELS * float(h)
    if st.filter_radius_source == "wall" and st.wall_mm is not None:
        from .robust import eta_eroded_of
        return wall_radius(st.wall_mm, eta_eroded_of(st)[0], h)["radius_mm"]
    return float(st.filter_radius)


def radius_from_min_member(min_member_mm: float) -> float:
    """Filter radius in mm for a requested minimum member size in mm (the rule above)."""
    return RADIUS_PER_MEMBER * float(min_member_mm)


def stencil_count(radius_voxels: float) -> int:
    """Number of voxel offsets with distance below `radius_voxels`: the filter entries of one row.

    This is the row length of `build_filter` in the interior of a design (the self weight
    included). Near a boundary or a frozen region the rows are shorter.
    """
    m = int(math.ceil(radius_voxels))
    if m > 120:   # an offset list this big costs memory itself: use the volume of the ball
        return int(round(4.0 / 3.0 * math.pi * radius_voxels ** 3))
    import numpy as np   # not at the top: `topopt.cli` must not load numpy (the thread cap runs first)

    g = np.arange(-m, m + 1, dtype=np.float64)
    d2 = g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2
    return int((d2 < radius_voxels ** 2).sum())


def filter_size(radius_mm: float, h: float, n_design: int, grid_shape) -> dict:
    """Facts about the size of the filter (an FFT convolution, #333): its kernel, the padded grid and the estimated memory.

    `grid_shape` is the shape of the voxel grid. The memory follows the padded grid and the design voxels; it does not depend on the
    radius much: the padded grid grows by 2 ceil(R) in each direction. `held_bytes_estimate` (the run: the kernel transform and one apply) and `build_peak_bytes_estimate` (the
    setup) are the same model. `entries_per_row` is the number of weights of the kernel (the length of an interior row of the
    matrix of `filter.build_filter`)."""
    from .filter import fft_shape   # not at the top: `topopt.cli` must not load numpy (the thread cap runs first)

    R = radius_mm / h
    per_row = stencil_count(R)
    shape = fft_shape(grid_shape, int(math.ceil(R)))
    cells = int(math.prod(shape))
    est = int(cells * FILTER_BYTES_PER_PADDED_CELL + int(n_design) * FILTER_BYTES_PER_DESIGN_VOXEL)
    return {"radius_voxels": round(R, 3), "entries_per_row": per_row, "method": "fft", "design_voxels": int(n_design),
            "fft_shape": list(shape), "fft_cells": cells,
            "build_peak_bytes_estimate": est, "build_peak_gib_estimate": round(est / 2 ** 30, 2),
            "held_bytes_estimate": est, "held_gib_estimate": round(est / 2 ** 30, 2)}


def size_text(nbytes: float) -> str:
    """Bytes as text: MiB below 1 GiB, else GiB (for example '37.2 MiB' and '2.01 GiB')."""
    gib = nbytes / 2 ** 30
    return f"{gib:.3g} GiB" if gib >= 1 else f"{nbytes / 2 ** 20:.3g} MiB"


def min_feature_note(R: float, h: float, requested_mm: float | None = None, wall: dict | None = None) -> str:
    """What the filter radius R (mm) means for the size of a member. One text for `check`, `run.log`,
    `run.json` and `metrics.json`.

    `requested_mm` is the request when R was derived from it (`filter_radius_source` is
    `min_member_mm`). Then the text says what the request gets: a wall of that size survives and a
    rod needs more. Else it gives the sizes that follow from R. The values are continuum values at
    the interior of the design space (A3a plan, "The rule").

    `wall` is the `wall_rule` block of `filter_report` (`settings.robust: wall`). The filter is padded with void there, so a wall at a free
    face is as thick as an inner wall, and the text says so and not that a skin can be half as thin.
    """
    rod = ROD_CRITICAL * R
    if wall is not None:
        nominal = wall["nominal_width_per_radius"] * R
        return (f"nominal walls of about {nominal:.3g} mm or more survive the erosion at eta_e {wall['eta_e']:g} (the wall rule; continuum value, "
                f"w {wall['wall_mm']:g} mm); rods need more (the continuum rod value at eta 0.5 is {rod:.3g} mm, and the eroded design asks for more). "
                f"The grid sets a floor of about one voxel ({h:g} mm). The filter is padded with void: a wall at a face of the design space "
                f"is as thick as an inner wall")
    if requested_mm is not None:
        head = (f"walls of about {requested_mm:g} mm survive; rods need about {rod:.3g} mm "
                f"({rod / requested_mm:.2f} x min_member_mm)")
    else:
        head = f"walls survive from about {WALL_CRITICAL * R:.3g} mm and rods from about {rod:.3g} mm"
    return (f"{head} (continuum values, interior of the design space; thinner ones vanish). The grid sets "
            f"a floor of about one voxel ({h:g} mm). A skin at a free face of the design space can be about "
            f"half as thin ({FACE_WALL_CRITICAL * R:.3g} mm for a wall)")


def filter_report(settings, h: float, n_design: int | None = None, grid_shape=None) -> dict:
    """The facts about the filter radius, the solid rule and the minimum member of a config.

    `settings` is a `Settings` object. The dict is in `derived.filter` of `check` and `run.json`,
    and in `metrics.json`. All the numbers come from one place, so they agree. `size` (`filter_size`) is there when `n_design`
    and `grid_shape` are given.
    """
    R = float(settings.filter_radius)
    req = settings.min_member_mm
    wall = wall_rule_block(settings, h)
    d = {
        "radius_mm": R,
        "radius_source": settings.filter_radius_source,   # default | settings | min_member_mm | wall
        "solid_neighbours": settings.filter_solid_neighbours,
        "min_member_mm": None if req is None else float(req),
        "min_member_radius_mm": None if req is None else radius_from_min_member(req),
        "min_feature_mm": max(float(h), R),
        "min_feature_note": min_feature_note(
            R, float(h), float(req) if (req is not None and settings.filter_radius_source == "min_member_mm") else None, wall),
    }
    if wall is not None:   # only with settings.robust: wall, so that every other record keeps its keys
        d["wall_rule"] = wall
    if req is not None and wall is None:   # with the wall rule the request is only reported, and its note (a free-face skin) does not hold
        d["min_member_rule"] = RULE_TEXT
        d["min_member_note"] = ESTIMATE_NOTE
    if n_design is not None and grid_shape is not None:
        d["size"] = filter_size(R, h, n_design, grid_shape)
    d["facts"] = member_facts(settings, h) + wall_facts(settings, h, wall)
    return d


def wall_rule_block(settings, h: float) -> dict | None:
    """The numbers of the wall rule for `derived.filter` and `metrics.json`: None unless `settings.robust` is `wall`.

    The radius that the rule gives (`radius_from_wall_mm`, `radius_rule_mm`) is also given when `settings.filter_radius` wins, so that
    the record shows what the rule would have set. `padded` is true: the density filter is padded with void in this mode.
    """
    if settings.robust != "wall" or settings.wall_mm is None:
        return None
    from .robust import eta_eroded_of
    eta_e, eta_source = eta_eroded_of(settings)
    r = wall_radius(settings.wall_mm, eta_e, h)
    return {"wall_mm": r["wall_mm"], "wall_source": "print.min_wall_mm", "eta_e": eta_e, "eta_e_source": eta_source,
            "nominal_width_per_radius": r["nominal_width_per_radius"], "radius_from_wall_mm": r["radius_from_wall_mm"],
            "voxel_floor_mm": r["voxel_floor_mm"], "radius_rule_mm": r["radius_mm"], "floor_wins": r["floor_wins"],
            "rule": WALL_RULE_TEXT, "padded": True}


WALL_RULE_TEXT = "filter_radius = max(w / L(eta_e), 1.5 voxels), w = print.min_wall_mm"


def wall_facts(settings, h: float, wall: dict | None) -> list[str]:
    """Facts about the wall rule with options. An empty list without it. None of them stops a run."""
    if wall is None:
        return []
    facts = []
    L, w, R = wall["nominal_width_per_radius"], wall["wall_mm"], float(settings.filter_radius)
    if settings.filter_radius_source == "settings":
        facts.append(
            f"settings.filter_radius {R:g} mm is set, so it is used and the wall rule does not set the radius. The rule would give "
            f"{wall['radius_rule_mm']:.4g} mm (w {w:g} mm, eta_e {wall['eta_e']:g}). With {R:g} mm the nominal wall that survives the erosion is "
            f"about {L * R:.3g} mm (continuum value), against print.min_wall_mm {w:g} mm. The density filter is padded with void in this mode. "
            f"Options: remove settings.filter_radius to use {wall['radius_rule_mm']:.4g} mm; or keep it.")
    elif wall["floor_wins"]:
        floor = wall["voxel_floor_mm"]
        facts.append(
            f"the wall rule gives w / L = {wall['radius_from_wall_mm']:.3g} mm, which is below the default radius of 1.5 voxels ({floor:.3g} mm), "
            f"so the grid sets the radius and the nominal wall that survives the erosion is about {L * floor:.3g} mm (continuum value), "
            f"above print.min_wall_mm {w:g} mm. Options: voxel_size {wall['radius_from_wall_mm'] / DEFAULT_RADIUS_VOXELS:.3g} mm or less lets "
            f"the rule set the radius; or keep the grid and accept the thicker walls.")
    return facts


def wall_min_member_fact(settings, h: float) -> str | None:
    """The fact for `min_member_mm` next to `robust: wall`: it is only reported. The radius comes from the wall rule, or from `settings.filter_radius`."""
    if settings.robust != "wall" or settings.min_member_mm is None:
        return None
    req = float(settings.min_member_mm)
    if settings.filter_radius_source == "settings":
        from .robust import eta_eroded_of
        rule = wall_radius(settings.wall_mm, eta_eroded_of(settings)[0], h)["radius_mm"]
        return (f"settings.min_member_mm {req:g} mm is set, and settings.robust is wall: min_member_mm is only reported. The filter radius is "
                f"settings.filter_radius {float(settings.filter_radius):g} mm. Removing settings.filter_radius gives the radius of the wall rule "
                f"({rule:.4g} mm), not the radius of min_member_mm. Options: remove settings.min_member_mm; remove settings.filter_radius to use "
                f"{rule:.4g} mm; or keep both.")
    return (f"settings.min_member_mm {req:g} mm is set, and settings.robust is wall: the wall rule sets the filter "
            f"radius ({float(settings.filter_radius):.4g} mm), so min_member_mm is only reported. Options: remove settings.min_member_mm; "
            f"or set settings.robust to eroded to let min_member_mm derive the radius.")


def member_facts(settings, h: float) -> list[str]:
    """Facts about `min_member_mm` with options. An empty list when it is not set."""
    req = settings.min_member_mm
    if req is None:
        return []
    req, h = float(req), float(h)
    derived = radius_from_min_member(req)
    facts = []
    if req < 2.0 * h:
        facts.append(
            f"min_member_mm {req:g} mm is {req / h:.2g} voxels at voxel_size {h:g} mm. A member needs "
            f"2 voxels or more to carry load, and 3 or more to be reliable. Options: voxel_size "
            f"{req / MEMBER_ACROSS:.3g} mm or less gives 3 voxels across; min_member_mm {2.0 * h:g} mm or more "
            f"gives 2 voxels at this grid; or keep both and accept that the grid sets the member size.")
    if settings.robust == "wall":   # the wall rule or settings.filter_radius sets the radius; min_member_mm is only reported
        facts.append(wall_min_member_fact(settings, h))
    elif settings.filter_radius_source == "settings":
        used = float(settings.filter_radius)
        if abs(used - derived) > 0.01 * derived:
            facts.append(
                f"settings.filter_radius {used:g} mm is set, so it is used. settings.min_member_mm "
                f"{req:g} mm would give a filter radius of {derived:.3g} mm ({RULE_TEXT}). Options: "
                f"remove settings.filter_radius to use {derived:.3g} mm; remove settings.min_member_mm "
                f"to drop the request; or keep both (min_member_mm is then only reported).")
    elif derived <= h:
        facts.append(
            f"the derived filter radius {derived:.3g} mm is not above one voxel ({h:g} mm): the filter "
            f"has no neighbour to average with, and the member size follows the grid.")
    return facts


def filter_lines(rep: dict) -> list[str]:
    """Text lines for `topopt check` and the run log, from `filter_report`.

    The facts of `rep["facts"]` are not in these lines: the caller prints them as warnings.
    """
    wall = rep.get("wall_rule")
    if wall is not None:   # settings.robust: wall
        how = ("settings.filter_radius wins" if rep["radius_source"] == "settings" else
               f"the wall rule: {wall['rule']}; w {wall['wall_mm']:g} mm / L {wall['nominal_width_per_radius']:.4g} = "
               f"{wall['radius_from_wall_mm']:.4g} mm, 1.5 voxels = {wall['voxel_floor_mm']:.4g} mm")
        head = (f"filter radius {rep['radius_mm']:g} mm ({how}; eta_e {wall['eta_e']:g}); the filter is padded with void: a wall at a face of "
                f"the design space is as thick as an inner wall")
        if rep["min_member_mm"] is not None:
            head += f"; min_member_mm {rep['min_member_mm']:g} mm is only reported"
    elif rep["min_member_mm"] is None:
        src = {"default": "default 1.5 voxels", "settings": "settings.filter_radius"}[rep["radius_source"]]
        head = f"filter radius {rep['radius_mm']:g} mm ({src})"
    else:
        how = ("used" if rep["radius_source"] == "min_member_mm" else
               f"not used: settings.filter_radius {rep['radius_mm']:g} mm wins")
        head = (f"min_member_mm {rep['min_member_mm']:g} mm: derived filter radius "
                f"{rep['min_member_radius_mm']:.3g} mm ({how}; {rep['min_member_rule']}; "
                f"{rep['min_member_note']}); filter radius {rep['radius_mm']:g} mm")
    lines = [f"{head}; solid neighbours {rep['solid_neighbours']}"]
    sz = rep.get("size")
    if sz:
        lines.append(f"filter size: {sz['entries_per_row']} weights per voxel, applied as an FFT on a padded grid of "
                     f"{' x '.join(str(n) for n in sz['fft_shape'])} cells, memory about "
                     f"{size_text(sz['held_bytes_estimate'])} (estimate)")
    return lines


def min_feature_line(rep: dict) -> str:
    """The text of the `min feature` line (`check` and `run.log`) from `filter_report`."""
    return (f"about {rep['min_feature_mm']:g} mm: the larger of one voxel and the filter radius, a rough size "
            f"for a rod. {rep['min_feature_note']}")
