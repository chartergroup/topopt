"""The stress section of `result/metrics.json` (A2, CONTRACTS.md section 17).

Two kinds of numbers, never mixed:

- relaxed numbers of the optimizer (KS value, relaxed peak). They come from the last history line.
  They are "relaxed, not physical": the stress is scaled by rho^0.5 (KB pitfall relaxed-stress-not-physical).
- physical numbers of the final design: one solve with the SIMP stiffness of the last snapshot, the stress
  `D0 * strain` with the full modulus E at the centre of each hex8 voxel (`refine.stress.compute_stress`),
  for the voxels with rho >= 0.5. The peak and p99.9 by volume are over the design voxels and over the solid
  voxels that carry no clamp and no load (`peak_scope`). Left out: the FIXED voxels and the voxels of a load region
  that a load case uses, because loads and clamps are spread over whole voxels there. A solid body that no case
  loads or clamps (for example a load region that no case uses) has real stress and is counted.
  `peak_incl_forced_solid_MPa` adds all of the left-out voxels.

The report also says whether the part carries the load (`load_path_facts`): for each load region, does the
thresholded design (and `final.stl`) connect it to a fixed region? The stress rows can be met by a design that has
no material between the load and the supports (a sub-resolution member: see CONTRACTS.md section 17).

The statistics are estimates. A verdict (pass or fail) for a statistic needs measured accuracy data
(B-051: `bench/stress_accuracy.py`, `bench/stress_accuracy.json`). `ACCURACY` holds what is committed.
Without an entry, the report says "estimate, accuracy not measured" and gives no verdict. At this time
no statistic has a verdict. The report gives the peak and p99.9 as estimates with the error range, the safety factors
with their range from the p99.9 band, and the volume above the limit.

The band of each statistic is the union of two sets of data (F1-8c, Refs #129): the bench (a beam, a plate with a hole and
the hook against P2: 11 rows, `bench/stress_accuracy.json`) and 5 public parts (51 cells, the voxel value against the P2 value
of `topopt verify`: `bench/stress_public_parts.json`). It is the largest error measured, and not a bound for other parts: the
public parts widened the band of the bench, the member gate did not keep the error inside the bench band, and each end of the
band is set by one row or cell. `ACCURACY[stat]["sources"]` keeps the two sets apart and names the row or cell that sets each
end; every text that gives a band or a range names the source of each end (`band_sources_text`). For the stress of the part,
run `topopt verify`.

`stress_section(run_dir)` can be called again later (for example after B2 wrote `verify/verify.json`).
`update_metrics(run_dir)` does that and writes the section into `metrics.json`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from topopt.resource_errors import reraise_if_resource

log = logging.getLogger("topopt.stress_report")
STRESS_MISSING = "the stress block of result/metrics.json"   # what a resource error leaves unwritten (#449)

PERCENTILE = 99.9
MIN_TOP_VOXELS = 5           # the top 0.1 % of the volume needs this many voxels, else p99.9 is the peak
MEMBER_TOP_PERCENT = 1.0     # the voxels at or above the p99 of the case count as "high stress"
MIN_MEMBER_VOXELS_EST = 3.5  # the bench band was measured for members of 4 voxels or more (estimate: see member_thickness)
NOT_MEASURED = "estimate, accuracy not measured"
RELAXED_LABEL = "relaxed, not physical"
PHYSICAL_LABEL = "full E, rho >= 0.5, design voxels"
_EXTRA_LABEL = " and {n} solid voxels that carry no clamp and no load"
GREY_LABEL = "grey design: the SIMP snapshot as the optimizer ran it, " + PHYSICAL_LABEL
THRESHOLDED_LABEL = "thresholded design: rho >= 0.5 solid, the rest void, " + PHYSICAL_LABEL
RELAXED_MET_MEANS = ("the relaxed KS row of the optimizer is at or below the limit (g <= 0.005); with a KS scale c below 1 "
                     "the relaxed peak can still be above the limit (`peak_above_limit` says so). "
                     "Not a statement about the part")
MET_NOTE = ' "Met" is about the relaxed measure, not about the part.'
PEAK_ABOVE_MEANS = ("the relaxed peak is above stress_limit_MPa by more than the tolerance (peak_MPa > stress_limit_MPa * "
                    "(1 + peak_above_limit_tol)); the test of the run log line `stress peak above the limit`, whatever `met` says")
VERIFY_FIELD = "cases.<case>.von_mises.p99_9_by_volume_MPa"
VERIFY_FILE = "verify/verify.json"
VERIFY_OPTION = ("run verify for the stress of the part (`topopt verify`, or the tool `verify_run`): "
                 f"{VERIFY_FILE} has no P2 value for this case")
BENCH_FILE = "bench/stress_accuracy.json"
PUBLIC_PARTS_FILE = "bench/stress_public_parts.json"
PART_STRESS_CHECK = "For the stress of the part, run `topopt verify`"

# The error of the voxel p99.9 on 5 public parts (F1-8, F1-8c). The package does not read `bench/`, so this is a literal copy of
# `bench/stress_public_parts.py` `fact_text(summary)`; `tests/test_stress_public_parts.py` checks that the two are equal.
# "Outside the bench band" is about the bench band alone: the band of the report is the union and contains every cell.
PUBLIC_PARTS_FACT = (
    'on 5 public parts (51 cells) the error of the voxel p99.9 against the P2 p99.9 of `topopt verify` (voxel / P2 - '
    '1) was -58 % (hook, pull_off, run type MS, voxel 2.4 mm) to +57 % (phone_stand, poke, run type MS, voxel 2.45 '
    'mm), in both directions: the voxel value was above P2 in 23 cells, mostly grey mass-objective designs, and up to '
    '2.4 times below P2 on a coarse part with thin members; 15 of the 51 cells were outside the bench band (-54 % to '
    '+13 %, bench/stress_accuracy.json), and the member gate did not keep the error inside the bench band (6 of the '
    '18 cells that met it were above +13 %); source bench/stress_public_parts.json')


def _signed(n) -> str:
    """A whole number of percent with its sign: '+57', '-59', '0'."""
    return "0" if n == 0 else f"{int(n):+d}"


def range_text(under_pct, over_pct) -> str:
    """The band as a range of percent of the reference: '-59 % to +57 %'."""
    return f"{_signed(-under_pct)} % to {_signed(over_pct)} %"


def band_range_text(acc: dict) -> str:
    """The rounded band of an `ACCURACY` entry (the union of the bench and the public cells): '-59 % to +57 %'."""
    return range_text(acc["under_pct"], acc["over_pct"])


def _set_name(acc: dict, which: str) -> str:
    return "the bench" if which == "bench" else f"the {acc['sources']['public']['parts']} public parts"


def band_sources_text(acc: dict, measured: bool = False) -> str:
    """Where each end of the band of an `ACCURACY` entry comes from: the set (the bench or the public parts), the row or the
    cell that sets the end, the measured value, and the data file. With `measured` the text gives the measured ends only
    ('lowest', 'highest'), without the data files, and not the rounded edges of the band."""
    src, used, ends = acc["sources"], [], []
    for side, key, edge in (("lower", "measured_min", -acc["under_pct"]), ("upper", "measured_max", acc["over_pct"])):
        which = src[f"{side}_end"]
        blk, used = src[which], used + [which]
        where = f"{_set_name(acc, which)}: {blk[f'{key}_from']}"
        if measured:
            ends.append(f"{'lowest' if side == 'lower' else 'highest'} {blk[f'{key}_pct']:+.2f} % from {where}")
        else:
            ends.append(f"{side} end {_signed(edge)} % (measured {blk[f'{key}_pct']:+.2f} %) from {where}")
    if measured:
        return "; ".join(ends)
    return "; ".join(ends) + "; data: " + ", ".join(src[w]["data"] for w in ("bench", "public") if w in used)


def bench_band_text(acc: dict) -> str:
    """The band of the bench alone, with the row that sets each end: '-54 % to +13 % (lower end, measured -53.32 %: ...)'."""
    b = acc["sources"]["bench"]
    return (f"{range_text(b['under_pct'], b['over_pct'])} (lower end, measured {b['measured_min_pct']:+.2f} %: "
            f"{b['measured_min_from']}; upper end, measured {b['measured_max_pct']:+.2f} %: {b['measured_max_from']}; "
            f"data: {b['data']})")


# Measured accuracy of the voxel statistics (B-051, F1-8c): a copy of `verdict_rule` in `bench/stress_accuracy.json`
# (made by `bench/stress_accuracy.py`; a test checks that the two are equal). A statistic without an entry has
# no verdict ("estimate, accuracy not measured"). An entry with "verdict": False has data and no verdict.
# The band is the union of the rows of the bench (a beam, a plate with a hole, the hook against B2) and the 51 cells of 5 public
# parts (`bench/stress_public_parts.json`): the error of the voxel value against the reference is between -under_pct % and
# +over_pct %, the largest error measured on each side, rounded up. `measured_min_pct` and `measured_max_pct` are the union;
# `sources` has the bench and the public ends apart, the row or cell that sets each one, and which set sets each end of the
# band. It is not a bound for other parts, and the member gate (an estimated 3.5 voxels or more, `member_check`) did not keep
# the error of the public cells inside the bench band (`PUBLIC_PARTS_FACT`).
# At this time both entries have "verdict": False (no verdict without data).
# The p99.9 band is used for the range estimate of the safety factor (`safety_entry`), labelled as an estimate.
ACCURACY: dict = {'p99_9': {'verdict': False,
           'under_pct': 59,
           'over_pct': 57,
           'measured_min_pct': -58.32,
           'measured_max_pct': 56.91,
           'min_voxels': 4,
           'min_voxels_hole_diameter': 4,
           'data': 'bench/stress_accuracy.json and bench/stress_public_parts.json',
           'cases': ['beam', 'hole', 'hook_vs_B2', 'public_parts'],
           'sources': {'bench': {'data': 'bench/stress_accuracy.json',
                                 'under_pct': 54,
                                 'over_pct': 13,
                                 'measured_min_pct': -53.32,
                                 'measured_min_from': 'the hook, pull_off, voxel 1.2 mm, against P2',
                                 'measured_max_pct': 12.95,
                                 'measured_max_from': 'plate with a hole, 8 voxels across the hole, gross scaling of '
                                                      'the reference'},
                       'public': {'data': 'bench/stress_public_parts.json',
                                  'parts': 5,
                                  'cells': 51,
                                  'under_pct': 59,
                                  'over_pct': 57,
                                  'measured_min_pct': -58.32,
                                  'measured_min_from': 'hook, pull_off, run type MS, voxel 2.4 mm',
                                  'measured_max_pct': 56.91,
                                  'measured_max_from': 'phone_stand, poke, run type MS, voxel 2.45 mm'},
                       'lower_end': 'public',
                       'upper_end': 'public'},
           'valid_when': 'the members at the high-stress voxels have an estimated 3.5 voxels or more through them (the '
                         'report checks this with `member_thickness`), and a hole or fillet has at least 4 voxels '
                         'across its diameter (the report does not check this). It is the condition of the bench and '
                         'not a guarantee: on public parts 6 of the 18 cells that met the member gate had an error '
                         'outside the bench band (-54 % to +13 %, bench/stress_accuracy.json; cells in '
                         'bench/stress_public_parts.json)',
           'data_base': {'rows': 62,
                         'rows_where_p99_9_is_a_true_percentile': 27,
                         'rows_where_p99_9_equals_the_peak': 35,
                         'bench': {'shapes': 3,
                                   'real_parts': 1,
                                   'rows': 11,
                                   'rows_where_p99_9_is_a_true_percentile': 7,
                                   'rows_where_p99_9_equals_the_peak': 4},
                         'public': {'parts': 5,
                                    'cells': 51,
                                    'cells_where_p99_9_is_a_true_percentile': 20,
                                    'cells_where_p99_9_equals_the_peak': 31},
                         'note': 'bench shapes: beam, plate with a hole, and the hook against B2 (the only real part '
                                 'of the bench); public parts: bracket, hook, ladle, phone_stand, shelf_bracket (a '
                                 'cell is a part, a load case and a run type); the hook is in both sets, with other '
                                 'runs (bench 1.2 mm, public 2.4 mm)'},
           'reason': 'no verdict for now (no verdict without data). The band of the error of the '
                     'voxel p99.9 (voxel value / reference - 1) is -59 % to +57 % of the reference, the union of the '
                     'bench and the public parts: lower end -59 % (measured -58.32 %) from the 5 public parts: hook, '
                     'pull_off, run type MS, voxel 2.4 mm; upper end +57 % (measured +56.91 %) from the 5 public '
                     'parts: phone_stand, poke, run type MS, voxel 2.45 mm; data: bench/stress_public_parts.json. Each '
                     'end is set by one row or cell, with no margin. The bench alone gave -54 % to +13 % (lower end, '
                     'measured -53.32 %: the hook, pull_off, voxel 1.2 mm, against P2; upper end, measured +12.95 %: '
                     'plate with a hole, 8 voxels across the hole, gross scaling of the reference; data: '
                     'bench/stress_accuracy.json). The member gate does not predict the error: the hook at 1.2 mm '
                     'passes it (p10 3.84 voxels) and has -53 %, while the beam at 4 voxels has -28 %. The data base '
                     'is 11 rows of the bench (3 shapes, 1 real part) and 51 cells of 5 public parts; 4 of the 11 rows '
                     'and 31 of the 51 cells have fewer than 5000 voxels, so their p99.9 is the peak. The band is the '
                     'largest error measured, not a bound for other parts: on 5 public parts (51 cells) the error of '
                     'the voxel p99.9 against the P2 p99.9 of `topopt verify` (voxel / P2 - 1) was -58 % (hook, '
                     'pull_off, run type MS, voxel 2.4 mm) to +57 % (phone_stand, poke, run type MS, voxel 2.45 mm), '
                     'in both directions: the voxel value was above P2 in 23 cells, mostly grey mass-objective '
                     'designs, and up to 2.4 times below P2 on a coarse part with thin members; 15 of the 51 cells '
                     'were outside the bench band (-54 % to +13 %, bench/stress_accuracy.json), and the member gate '
                     'did not keep the error inside the bench band (6 of the 18 cells that met it were above +13 %); '
                     'source bench/stress_public_parts.json. For the stress of the part, run `topopt verify`'},
 'peak': {'verdict': False,
          'under_pct': 88,
          'over_pct': 0,
          'measured_min_pct': -87.14,
          'measured_max_pct': -4.82,
          'min_voxels': 4,
          'min_voxels_hole_diameter': 4,
          'data': 'bench/stress_accuracy.json and bench/stress_public_parts.json',
          'cases': ['beam', 'hole', 'hook_vs_B2', 'public_parts'],
          'sources': {'bench': {'data': 'bench/stress_accuracy.json',
                                'under_pct': 72,
                                'over_pct': 0,
                                'measured_min_pct': -71.46,
                                'measured_min_from': 'the hook, pull_off, voxel 1.2 mm, against P2',
                                'measured_max_pct': -4.82,
                                'measured_max_from': 'beam, 16 voxels through the thickness'},
                      'public': {'data': 'bench/stress_public_parts.json',
                                 'parts': 5,
                                 'cells': 51,
                                 'under_pct': 88,
                                 'over_pct': 0,
                                 'measured_min_pct': -87.14,
                                 'measured_min_from': 'ladle, full_scoop, run type MF, voxel 3.25 mm, variant S2',
                                 'measured_max_pct': -5.39,
                                 'measured_max_from': 'shelf_bracket, lean, run type MS, voxel 1.6 mm'},
                      'lower_end': 'public',
                      'upper_end': 'bench'},
          'valid_when': 'the members at the high-stress voxels have an estimated 3.5 voxels or more through them (the '
                        'report checks this with `member_thickness`), and a hole or fillet has at least 4 voxels '
                        'across its diameter (the report does not check this). It is the condition of the bench and '
                        'not a guarantee: on public parts 3 of the 18 cells that met the member gate had a voxel peak '
                        'more than 72 % below the P2 peak (72 % is the lower end of the bench band, '
                        'bench/stress_accuracy.json; cells in bench/stress_public_parts.json)',
          'data_base': {'rows': 62,
                        'rows_where_p99_9_is_a_true_percentile': 27,
                        'rows_where_p99_9_equals_the_peak': 35,
                        'bench': {'shapes': 3,
                                  'real_parts': 1,
                                  'rows': 11,
                                  'rows_where_p99_9_is_a_true_percentile': 7,
                                  'rows_where_p99_9_equals_the_peak': 4},
                        'public': {'parts': 5,
                                   'cells': 51,
                                   'cells_where_p99_9_is_a_true_percentile': 20,
                                   'cells_where_p99_9_equals_the_peak': 31},
                        'note': 'bench shapes: beam, plate with a hole, and the hook against B2 (the only real part of '
                                'the bench); public parts: bracket, hook, ladle, phone_stand, shelf_bracket (a cell is '
                                'a part, a load case and a run type); the hook is in both sets, with other runs (bench '
                                '1.2 mm, public 2.4 mm)'},
          'reason': 'the voxel peak is below the reference in every measured case (the centre of a voxel lies half a '
                    'voxel below the surface), the sign is not known for other shapes, and the peak of a corner or '
                    'load point depends on the voxel size. The band of the error (voxel value / reference - 1) is -88 '
                    '% to 0 % of the reference: lower end -88 % (measured -87.14 %) from the 5 public parts: ladle, '
                    'full_scoop, run type MF, voxel 3.25 mm, variant S2; upper end 0 % (measured -4.82 %) from the '
                    'bench: beam, 16 voxels through the thickness; data: bench/stress_accuracy.json, '
                    'bench/stress_public_parts.json. The reference of the public cells is the P2 peak at the '
                    'quadrature points; the P2 peak depends on the mesh of `topopt verify` (`limits` in '
                    'bench/stress_public_parts.json). For the stress of the part, run `topopt verify`'}}


def top_tail_voxels(n, p=PERCENTILE) -> float:
    """Voxels in the top (100 - p) % of the volume of n equal voxels: n * (100 - p) / 100 (not rounded)."""
    return round(n * (100.0 - p) / 100.0, 9)    # rounded: 5000 * 0.1 / 100 is 4.9999999999997 in floating point


def p99_9_equals_peak(n, p=PERCENTILE) -> bool:
    """True when the top 0.1 % of the volume has fewer than `MIN_TOP_VOXELS` voxels.

    The step-wise p99.9 of n equal voxels is the value at index ceil(0.999 n) - 1: for fewer than about 1000
    voxels it is the largest value, and for fewer than 5000 voxels it rests on the top 1 to 4 voxels. The
    statistic is then a peak and not a percentile (REVIEW M4).
    """
    return n > 0 and top_tail_voxels(n, p) < MIN_TOP_VOXELS


def p99_9_rank_from_top(n, p=PERCENTILE) -> int:
    """Rank from the top (1 = the peak) of the voxel that `percentile_by_volume` returns for n equal voxels."""
    cw = np.arange(1, n + 1) / n
    return int(n - min(int(np.searchsorted(cw, p / 100.0, side="left")), n - 1))


def equals_peak_text(n, p99_9=None, peak=None) -> str:
    """The phrase for a p99.9 that rests on the top 1 to 4 voxels. With the two values (MPa) and n >= 1000 it adds
    the rank, because the two values differ there (R1-N1): p99.9 is the peak only for fewer than 1000 voxels."""
    text = (f"p99.9 equals the peak at this voxel count ({n} voxels: the top 0.1 % of the volume is "
            f"{top_tail_voxels(n):.2g} voxels, fewer than {MIN_TOP_VOXELS})")
    k = p99_9_rank_from_top(n) if n > 0 else 1
    if k > 1 and p99_9 is not None and peak is not None:
        text += f" (p99.9 is the voxel at rank {k} from the top: {p99_9:.4g} MPa; peak {peak:.4g} MPa)"
    return text


def percentile_by_volume(values, p=PERCENTILE) -> float:
    """Step-wise percentile by volume. All voxels have the same volume, so the weights are equal.

    The rule of the P2 check of the pipeline (`weighted_percentile`, `p2_core.py`): the first
    sample whose cumulative weight reaches p, no interpolation.
    """
    v = np.sort(np.asarray(values, float).ravel())
    cw = np.arange(1, v.size + 1) / v.size
    return float(v[min(int(np.searchsorted(cw, p / 100.0, side="left")), v.size - 1)])


def peak_scope(stress: dict, problem) -> np.ndarray:
    """Which voxels of `compute_stress` (rho >= 0.5) count for the peak and p99.9: a bool array (n,).

    The design voxels, and the solid voxels that carry no clamp and no load. Left out: the FIXED voxels and the
    voxels of a load region that a load case uses. A load region that no load case uses is a kept-solid body with
    no load: its stress is real, so it counts.
    """
    from topopt.voxelize import FIXED, LOAD

    ijk = np.floor((np.asarray(stress["centers"]) - np.asarray(problem.origin)) / problem.h).astype(int)
    label = problem.labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    rid = problem.load_region_ids[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    cfg = problem.config
    used = ({f.region for c in cfg.load_cases for f in c.forces} | {p.region for c in cfg.load_cases for p in c.pressures}
            | {s.region for s in cfg.springs} | {j.region for j in cfg.joints})   # a spring or joint region holds the part (sections 35, 38)
    used_ids = [i for i, lr in enumerate(cfg.regions.load) if lr.name in used]
    return ~((label == FIXED) | ((label == LOAD) & np.isin(rid, used_ids)))


def physical_statistics(stress: dict, limits: dict | None = None, h: float | None = None,
                        scope: np.ndarray | None = None) -> list[dict]:
    """Physical numbers of each load case from `compute_stress`. Voxels with rho >= 0.5 in `scope`.

    `scope` (n,) bool says which voxels count (`peak_scope`); the default is the design voxels.
    `limits` ({case name: MPa}) and `h` (voxel size, mm) add the voxels above the limit and their volume.
    All numbers are voxel estimates.
    """
    design = stress["design"] if scope is None else scope
    extra = int((design & ~stress["design"]).sum())
    out = []
    for i, name in enumerate(stress["case_names"]):
        vm = stress["vm_cases"][i]
        d = vm[design]
        n = int(d.size)
        entry = {
            "name": name, "label": PHYSICAL_LABEL if not extra else PHYSICAL_LABEL + _EXTRA_LABEL.format(n=extra),
            "peak_MPa": float(d.max()) if d.size else 0.0,
            "p99_9_by_volume_MPa": percentile_by_volume(d) if d.size else 0.0,
            "voxels": n,
            "p99_9_equals_peak": p99_9_equals_peak(n),
            "p99_9_rank_from_top": p99_9_rank_from_top(n) if n else None,
            "top_0_1_percent_voxels": round(top_tail_voxels(n), 3),
            "peak_incl_forced_solid_MPa": float(vm.max()) if vm.size else 0.0,
        }
        if limits and name in limits:
            above = int((d > limits[name]).sum())
            entry["peak_over_limit"] = entry["peak_MPa"] / limits[name]
            entry["voxels_above_limit"] = above
            if h:
                entry["volume_above_limit_mm3"] = float(above * h ** 3)
            entry["above_limit_note"] = ("estimate from voxel stress: the voxel peak was below the reference in "
                                         "every measured case (bench and public parts: see `verdict.peak.status`), so "
                                         "the true volume above the limit can be larger")
        if extra:
            entry["solid_voxels_without_clamp_or_load"] = extra
        if entry["p99_9_equals_peak"]:
            entry["p99_9_note"] = equals_peak_text(n, entry["p99_9_by_volume_MPa"], entry["peak_MPa"])
        out.append(entry)
    return out


def member_thickness(solid, ijk) -> np.ndarray:
    """Estimate of the voxels through the member at the voxels `ijk` (n, 3) of the boolean grid `solid`.

    Twice the largest distance to a void voxel centre in the 5 x 5 x 5 neighbourhood (a scipy distance
    transform). For a flat member of n voxels it gives n for an even n and n + 1 for an odd n (up to 6 or 7,
    then it saturates). It is an estimate for a gate, not a measure of the thickness.
    """
    from scipy.ndimage import distance_transform_edt, maximum_filter

    edt = distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1, 1:-1]
    est = 2.0 * maximum_filter(edt, size=5)
    return est[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def member_check(stress: dict, solid, origin, h, case_index: int, scope: np.ndarray | None = None) -> dict:
    """The estimated member thickness at the high-stress voxels of one case (top 1 % of the voxels in `scope`)."""
    vm, design = stress["vm_cases"][case_index], (stress["design"] if scope is None else scope)
    if not design.any():
        return {"voxels": 0, "p10_voxels": None, "median_voxels": None, "met": False}
    top = design & (vm >= np.percentile(vm[design], 100.0 - MEMBER_TOP_PERCENT))
    ijk = np.floor((stress["centers"][top] - np.asarray(origin)) / h).astype(int)
    t = member_thickness(solid, ijk)
    p10 = float(np.percentile(t, 10))
    return {"voxels": int(top.sum()), "p10_voxels": p10, "median_voxels": float(np.median(t)),
            "min_needed": MIN_MEMBER_VOXELS_EST, "met": bool(p10 >= MIN_MEMBER_VOXELS_EST)}


def verdict(stat: str, value: float, limit: float, member: dict | None = None, n_voxels: int | None = None,
            peak: float | None = None) -> dict:
    """Pass or fail of one statistic against the limit, with the measured error band.

    Only a statistic whose `ACCURACY` entry has `verdict: True` gets one. At this time none has (the
    entry of p99.9 says why; its `error_band_pct` has the band, `sources` with the row or cell that sets each end, and the
    data base). The rule for such an entry: the error of the voxel value is between -under
    and +over percent. The reference value is then between `value / (1 + over)` and `value / (1 - under)`.
    Pass: even the high end is at or below the limit. Fail: even the low end is above the limit. Otherwise
    the limit lies inside the error band: no pass or fail. p99.9 has no verdict when it equals the peak
    (`n_voxels` below 5000).
    """
    acc = ACCURACY.get(stat)
    if not acc:
        return {"status": NOT_MEASURED, "pass": None, "error_band_pct": None}
    if not acc.get("verdict", True):
        band = {"measured_min_pct": acc["measured_min_pct"], "measured_max_pct": acc["measured_max_pct"],
                "data": acc["data"], "sources": acc.get("sources"), "data_base": acc.get("data_base")}
        status = f"no verdict for this statistic: {acc['reason']}"
        if stat == "p99_9" and n_voxels is not None and p99_9_equals_peak(n_voxels):
            status = f"{equals_peak_text(n_voxels, value, peak)}; " + status
        return {"status": status, "pass": None, "error_band_pct": band}
    if stat == "p99_9" and n_voxels is not None and p99_9_equals_peak(n_voxels):
        return {"status": "no verdict: " + equals_peak_text(n_voxels, value, peak), "pass": None, "error_band_pct": None}
    if member is not None and not member["met"]:
        return {"status": ("no verdict: the members at the high-stress voxels are thinner than the band was "
                           f"measured for (estimate {member['p10_voxels']} voxels, needed {MIN_MEMBER_VOXELS_EST:g})"
                           if member["p10_voxels"] is not None else "no verdict: no design voxel with rho >= 0.5"),
                "pass": None, "error_band_pct": None}
    under, over = acc["under_pct"] / 100.0, acc["over_pct"] / 100.0
    lo = value / (1.0 + over)
    hi = value / (1.0 - under) if under < 1.0 else float("inf")
    band = {"under_pct": acc["under_pct"], "over_pct": acc["over_pct"], "min_voxels": acc["min_voxels"],
            "valid_when": acc.get("valid_when"), "data": acc["data"], "sources": acc.get("sources"),
            "reference_low_MPa": lo, "reference_high_MPa": hi}
    if hi <= limit:
        return {"status": "pass with the measured error band", "pass": True, "error_band_pct": band}
    if lo > limit:
        return {"status": "fail with the measured error band", "pass": False, "error_band_pct": band}
    return {"status": "the limit lies inside the error band: no pass or fail", "pass": None,
            "error_band_pct": band}


def _factor(limit, value) -> float | None:
    return float(limit / value) if value > 0 else None


def member_gate_text(member: dict | None) -> str:
    """The member gate in words: the bench band was measured for members of 4 voxels or more (estimate 3.5). On public
    parts the gate did not keep the error inside the bench band, and the text says so. It gives no band: the bench band, the
    source of each of its ends and the public-part numbers are in `verdict.p99_9.status`."""
    if not member or member.get("p10_voxels") is None:
        return "not known (no design voxel with rho >= 0.5)"
    state = "met" if member["met"] else "not met"
    return (f"{state}: the members at the high-stress voxels are an estimated {member['p10_voxels']:.2g} voxels thick "
            f"(p10), the bench band was measured for {MIN_MEMBER_VOXELS_EST:g} or more; on public parts this gate did not "
            f"keep the error inside the bench band ({PUBLIC_PARTS_FILE}; `verdict.p99_9.status` has the band, the source of each "
            f"end and the numbers)")


def safety_entry(stat: str, reference: float, value: float, phys: dict, member: dict | None) -> dict | None:
    """One achieved safety factor `reference / value` with the status of its accuracy (REVIEW M6).

    `reference` is the limit or the allowable (MPa). For p99.9 the entry has `range_estimate`: the factor
    for the reference value between `value / (1 + over)` and `value / (1 - under)` (the band: the union of the bench and the
    public cells, with the source of each end in the `status`), the member gate status, and the data base. For the peak, and
    for a p99.9 that equals the peak, there is no accuracy band: the entry says so and gives the measured fact that the voxel
    peak was below the reference, with the cell or row of its lowest and highest value.
    """
    f = _factor(reference, value)
    if f is None:
        return None
    peak_acc, p_acc = ACCURACY.get("peak"), ACCURACY.get("p99_9")
    below = (f"the voxel peak was below the reference in every measured case: {band_sources_text(peak_acc, measured=True)}; "
             f"so the true factor can be lower than {f:.3g}"
             if peak_acc and peak_acc.get("measured_max_pct", 1) < 0 else None)
    if stat == "p99_9" and not phys.get("p99_9_equals_peak") and p_acc:
        under, over = p_acc["under_pct"] / 100.0, p_acc["over_pct"] / 100.0
        lo = reference * (1.0 - under) / value
        hi = reference * (1.0 + over) / value
        return {"value": f, "range_estimate": [lo, hi],
                "status": (f"estimate: factor range from the error band of p99.9 ({band_range_text(p_acc)} of the reference; "
                           f"{band_sources_text(p_acc)}; data base: {_data_base_text(p_acc)}); "
                           + ("no verdict" if not p_acc.get("verdict", True) else "verdict possible")
                           + ". The band is the largest error measured and not a bound for other parts "
                           f"(`verdict.p99_9.status` has the public-part fact). {PART_STRESS_CHECK}"),
                "member_gate": member_gate_text(member)}
    status = "no accuracy band"
    if stat == "p99_9":
        status = f"{equals_peak_text(phys['voxels'], phys.get('p99_9_by_volume_MPa'), phys.get('peak_MPa'))}; no accuracy band"
    return {"value": f, "range_estimate": None, "status": status, "note": below}


def _data_base_text(acc: dict) -> str:
    db = acc.get("data_base") or {}
    if not db:
        return "data base not recorded"
    bench, pub = db["bench"], db.get("public")
    text = f"bench {bench['shapes']} shapes, {bench['real_parts']} real part" + ("s" if bench["real_parts"] != 1 else "")
    return text + (f"; {pub['parts']} public parts, {pub['cells']} cells" if pub else "")


def safety_text(case: dict, phys: dict, factors: dict | None = None, member: dict | None = None) -> str:
    """The achieved safety factor in plain words. A voxel estimate, not a check: the text says so."""
    lim, peak, p = case["stress_limit_MPa"], phys["peak_MPa"], phys["p99_9_by_volume_MPa"]
    if phys["voxels"] == 0:
        return "No design voxel has rho >= 0.5, so there is no physical stress to compare with the limit."
    same = phys.get("p99_9_equals_peak")
    parts = [f"Voxel estimate, not a check: the peak is {peak:.4g} MPa"
             + ("" if same else f" and p99.9 by volume is {p:.4g} MPa")
             + f" (physical, full E, {phys['voxels']} design voxels with rho >= 0.5)."]
    if same:
        parts.append(equals_peak_text(phys["voxels"], p, peak) + ".")
    ranged = peaked = False
    peak_acc, p_acc = ACCURACY.get("peak"), ACCURACY.get("p99_9")
    below = bool(peak_acc and peak_acc.get("measured_max_pct", 1) < 0)
    for label, ref in (("limit", lim), ("allowable", case.get("allowable_mpa"))):
        if not ref or peak <= 0:
            continue
        parts.append(f"The {label} {ref:.4g} MPa is {ref / peak:.3g} times the peak (no accuracy band"
                     + ("; the true factor can be lower" if below else "") + ").")
        peaked = peaked or below
        if not same and p > 0:
            e = (factors or {}).get("against_limit" if label == "limit" else "against_allowable", {}) or {}
            e = e.get("p99_9") or {}
            rng = e.get("range_estimate")
            ranged = ranged or bool(rng)
            parts.append(f"The {label} is {ref / p:.3g} times p99.9"
                         + (f", an estimated range of {rng[0]:.3g} to {rng[1]:.3g} from the error band of p99.9 below "
                            f"(member gate {'met' if member and member['met'] else 'not met'}; no verdict)." if rng else "."))
    # the bands are said once, with the source of each end
    if peaked:
        assert peak_acc is not None
        parts.append(f"The voxel peak was {-peak_acc['measured_max_pct']:.0f} % to {-peak_acc['measured_min_pct']:.0f} % below the "
                     f"reference in every measured case: {band_sources_text(peak_acc, measured=True)}.")
    if ranged:
        assert p_acc is not None
        parts.append(f"The error band of p99.9 is {band_range_text(p_acc)} of the reference: {band_sources_text(p_acc)}. It is the "
                     f"largest error measured and not a bound for other parts (`verdict.p99_9.status` has the public-part fact). "
                     f"{PART_STRESS_CHECK}.")
    return " ".join(parts)


def read_verify(run_dir, case_name):
    """The P2 p99.9 of a case from `verify/verify.json` (B2), or None when the file or field is missing."""
    path = Path(run_dir) / "verify" / "verify.json"
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text())
        return float(d["cases"][case_name]["von_mises"]["p99_9_by_volume_MPa"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_verified_file(run_dir):
    """The file that `topopt verify` meshed (`run.verified_file` of `verify/verify.json`), or None when the file, the block or the key is missing."""
    path = Path(run_dir) / "verify" / "verify.json"
    try:
        name = json.loads(path.read_text())["run"]["verified_file"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return str(name) if name else None


def read_verify_coverage(run_dir):
    """`(partial, coverage)` of `verify/verify.json` (TC-2, CONTRACTS.md section 44), or None when the file or one of the keys is missing."""
    path = Path(run_dir) / "verify" / "verify.json"
    try:
        d = json.loads(path.read_text())
        partial, cov = d["partial"], d["coverage"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (partial, cov) if isinstance(partial, bool) and isinstance(cov, dict) else None


def independent_note(verified_file):
    """The note of `independent_check`: which file verify meshed, and for `part_only.stl` that the two models are not the same part."""
    base = "its p99.9 is the independent value. The voxel value is an estimate."
    if verified_file is None:
        return f"B2 (topopt verify) meshes the finished part with tets and solves with quadratic elements; {base}"
    text = f"B2 (topopt verify) meshed {verified_file} with tets and solves with quadratic elements; {base}"
    if verified_file == "part_only.stl":
        text += (" The P2 model is part_only.stl: the part without the bodies of regions.not_printed (CONTRACTS.md section 9.1), with their"
                 " boundary conditions at the contact faces. The voxel model is the whole design, with those bodies,"
                 " so p2_over_voxel compares two different models.")
    return text


def independent_check(run_dir, name, limit, voxel_p99_9) -> dict:
    """The key `independent_check` of a case: the P2 p99.9 of `topopt verify` (`verify/verify.json`) next to the voxel p99.9 and the limit.

    `found` is false and the P2 numbers are None where the file, the case or the field is missing (verify has not run). `p2_over_limit` is the
    P2 p99.9 over the stress limit of the case: the stress of the part against the user's limit (an estimate, no verdict).
    When `verify.json` has the keys `partial` and `coverage` (TC-2), the dict also has `verify_partial` and `verify_coverage`: the P2 value
    comes from a verify run whose other sub-checks may have failed or not run, and a reader of this dict must see that.
    """
    ver = read_verify(run_dir, name)
    vfile = read_verified_file(run_dir) if ver is not None else None
    out = {"file": VERIFY_FILE, "field": VERIFY_FIELD.replace("<case>", name),
           "found": ver is not None, "p2_p99_9_by_volume_MPa": ver,
           "voxel_p99_9_by_volume_MPa": voxel_p99_9,
           "p2_over_voxel": ver / voxel_p99_9 if ver is not None and voxel_p99_9 > 0 else None,
           "p2_over_limit": ver / limit if ver is not None and limit > 0 else None,
           "verified_file": vfile, "note": independent_note(vfile)}
    cov = read_verify_coverage(run_dir) if ver is not None else None
    if cov is not None:
        out["verify_partial"], out["verify_coverage"] = cov
    return out


def relaxed_numbers(last_history, case_name):
    """The relaxed numbers of the last pass for one case, or None (old run).

    `met` says whether the row is met at the last pass (g at or below `NOT_MET`). The row is a target that
    the optimizer tries to meet, so the report says whether it did.
    """
    from topopt.formulation import NOT_MET
    from topopt.ks_stress import PEAK_ABOVE_TOL, peak_above_limit

    for e in (last_history or {}).get("constraints") or []:
        if e.get("name") == f"stress:{case_name}" and "relaxed_peak_MPa" in e:
            lim = e.get("stress_limit_MPa")
            return {"label": RELAXED_LABEL, "ks_MPa": e["ks_MPa"], "peak_MPa": e["relaxed_peak_MPa"],
                    "row_value": e["value"], "row_g": e["g"], "met": bool(e["g"] <= NOT_MET),
                    "met_when_g_at_or_below": NOT_MET, "met_means": RELAXED_MET_MEANS,
                    "peak_above_limit": None if lim is None else peak_above_limit(e["relaxed_peak_MPa"], lim),
                    "peak_above_limit_tol": PEAK_ABOVE_TOL, "peak_above_limit_means": PEAK_ABOVE_MEANS,
                    "c": e["c"], "q": e["q"], "iteration": last_history["iteration"]}
    return None


def input_changes(problem, run) -> dict:
    """Compare the forces and the material of a rebuilt problem with `run.json` (REVIEW m7).

    Returns `{"cases": {name: [text]}, "all": [text], "not_compared": [text]}`. `cases` holds the load cases
    whose forces differ, `all` the material differences (they change every case). `not_compared` lists what
    `run.json` does not hold (a run made before `run.json` kept E and nu in the stress block).
    """
    import numpy as np

    cfg = problem.config
    out = {"cases": {}, "all": [], "not_compared": []}
    was_cases = {c["name"]: c for c in run.get("load_cases", [])}
    ext = cfg.uses_extended_loads()   # A4b (section 26.2): run.json has pressure entries and moments that the YAML forces do not
    if ext:
        out["not_compared"].append("forces: a moment, a remote force, a pressure, a per-axis support, a spring or a joint is not compared with run.json")
    for c in ([] if ext else cfg.load_cases):
        was = was_cases.get(c.name)
        now = sorted((f.region, tuple(float(v) for v in f.force)) for f in c.forces)
        if was is None:
            out["cases"][c.name] = ["the load case is not in run.json"]
            continue
        old = sorted((f["region"], tuple(float(v) for v in f["force"])) for f in was["forces"])
        same = len(now) == len(old) and all(a[0] == b[0] and np.allclose(a[1], b[1], rtol=1e-12, atol=0.0)
                                            for a, b in zip(now, old))
        if not same:
            out["cases"][c.name] = [f"forces: run.json has {old}, the YAML now has {now} (region, N)"]
    mat = cfg.material
    saved = (run.get("stress") or {}).get("material")
    if saved:
        for key, now_v in (("E", mat.E), ("nu", mat.nu), ("density", mat.density)):
            if key in saved and not np.isclose(float(saved[key]), float(now_v), rtol=1e-12, atol=0.0):
                out["all"].append(f"material.{key}: run.json has {saved[key]:g}, the YAML now has {now_v:g}")
    else:
        dens = (run.get("derived") or {}).get("density_g_cm3")
        if dens is not None and not np.isclose(float(dens), mat.density, rtol=1e-12, atol=0.0):
            out["all"].append(f"material.density: run.json has {dens:g}, the YAML now has {mat.density:g}")
        out["not_compared"].append("material.E and material.nu: this run.json does not hold them")
    return out


def step_acceptance_summary(history) -> dict | None:
    """Facts of step acceptance from the history (the `step_acceptance` records), or None for a run without them."""
    steps = [r["step_acceptance"] for r in history if r.get("step_acceptance")]
    if not steps:
        return None
    return {"passes": len(steps),
            "tries_rejected": sum(len(s["rejected"]) + (0 if s["accepted"] else 1) for s in steps),
            "passes_with_a_rejected_try": sum(1 for s in steps if s["rejected"] or not s["accepted"]),
            "passes_that_took_a_step_after_all_tries_were_rejected": sum(1 for s in steps if not s["accepted"]),
            "smallest_move_limit": min(s["move"] for s in steps)}


def stage_ends(history) -> list[dict]:
    """The largest g of each beta stage at its last pass: `{pass, beta, row, g, feasible}`, and the last pass."""
    from topopt.formulation import NOT_MET

    out = []
    for i, r in enumerate(history):
        last = i == len(history) - 1
        if (r.get("stage") and r["stage"].get("from_beta") is not None) or last:
            ref = r if last else history[i - 1]
            if not ref.get("constraints"):
                continue
            worst = max(ref["constraints"], key=lambda c: c["g"])
            out.append({"pass": ref["iteration"], "beta": ref["beta"], "row": worst["name"], "g": worst["g"],
                        "feasible": bool(worst["g"] <= NOT_MET)})
    return out


LOAD_PATH_RULE = ("thresholded design: the design voxels with rho >= 0.5 and all fixed and load voxels are solid; "
                  "two solid voxels are joined when they share a face (6-connected). A load region is connected when one "
                  "joined body holds a voxel of the region and a voxel of a fixed region")
FINAL_STL_RULE = ("final.stl: the bodies of the mesh (triangles joined by shared vertices). A load region is connected when "
                  "one body contains the centre of a voxel of the region and the centre of a fixed voxel")
LOAD_PATH_OPTIONS = ("options: a larger settings.filter_radius (thicker members), settings.objective compliance with a "
                     "volume_fraction and the stress limits as rows (the volume row keeps material between the load and "
                     "the supports), a finer voxel_size, or a lower stress_limit (all of these change the problem or "
                     "the cost: the tool does not decide)")
MAX_STL_BODIES = 200         # more bodies than this: the mesh check is not made
MAX_STL_POINTS = 300         # voxel centres of a region that are tested against each body


def _region_points(problem, mask) -> np.ndarray:
    """Centres (n, 3) in mm of the voxels of `mask`, at most `MAX_STL_POINTS` of them (evenly spread)."""
    ijk = np.argwhere(mask)
    if len(ijk) > MAX_STL_POINTS:
        ijk = ijk[np.linspace(0, len(ijk) - 1, MAX_STL_POINTS).astype(int)]
    return np.asarray(problem.origin, float) + problem.h * (ijk + 0.5)


def _holds_a_point(body, pts) -> bool:
    """True when `body` holds at least one of `pts`: the same value as `body.contains(pts).any()`, in chunks.

    It stops at the first chunk with a point inside (`topopt.post.contains_chunks`).
    """
    from topopt.post import contains_chunks

    return any(chunk.any() for chunk in contains_chunks(body, pts))


def _stl_connectivity(problem, final_stl, regions) -> dict:
    """Which load regions does `final.stl` connect to a fixed region. `regions` is `{name: mask}`. Never raises."""
    import trimesh

    from topopt.springs import support_mask

    try:
        mesh = trimesh.load(str(final_stl), process=True, force="mesh")
        bodies = mesh.split(only_watertight=False)
        if not len(bodies):
            return {"error": f"{Path(final_stl).name} has no triangles"}
        if len(bodies) > MAX_STL_BODIES:
            return {"error": f"final.stl has {len(bodies)} bodies: the check is not made above {MAX_STL_BODIES}"}
        fixed_pts = _region_points(problem, support_mask(problem))       # FIXED, and a spring region (CONTRACTS.md section 35)
        has_fixed = [_holds_a_point(b, fixed_pts) for b in bodies]
        out = {"bodies": int(len(bodies)), "bodies_with_a_fixed_region": int(sum(has_fixed)), "regions": {}}
        for name, mask in regions.items():
            pts = _region_points(problem, mask)
            inside = [_holds_a_point(b, pts) for b in bodies]
            out["regions"][name] = {"connected_to_fixed": bool(any(a and b for a, b in zip(inside, has_fixed))),
                                    "touches_a_body": bool(any(inside)), "bodies_with_the_region": int(sum(inside))}
        return out
    except Exception as exc:   # a missing or unreadable mesh never stops the report
        return {"error": f"{type(exc).__name__}: {exc}"}


def load_path_facts(problem, snap, final_stl=None) -> dict:
    """For each load region: does the thresholded part connect it to a fixed region? A fact, not a verdict.

    The stress rows can be met by a design that has no material between the load and the supports: a member
    thinner than the voxel size lets the relaxed stress stay low, and the smoothed mesh drops it. So the report
    says it. Two checks: the thresholded design of the last snapshot (`LOAD_PATH_RULE`), and `final.stl` when the
    file exists (`FINAL_STL_RULE`). Never raises: an error becomes `{"error": ...}`.
    """
    try:
        return _load_path_facts(problem, snap, final_stl)
    except Exception as exc:
        log.warning("load path check failed: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_load_path_facts(run_dir, run, problem, snap, final_stl) -> dict:
    """`load_path_facts` for `metrics.load_paths` of every run (I6). Without the `Problem` of the run, the problem is rebuilt from the YAML of
    the run (`refine.build.problem_from_run`). The import is here and not in `topopt.post`, so the post step reaches the refine package only
    through this module (`tests/test_refine.py`). Raises on a rebuild error; the caller turns it into `{"error": text}`.
    """
    if problem is None:
        from topopt.refine.build import problem_from_run
        problem = problem_from_run(Path(run_dir), run)
    return load_path_facts(problem, snap, final_stl)


def _load_path_facts(problem, snap, final_stl) -> dict:
    from scipy.ndimage import generate_binary_structure, label

    from topopt.springs import support_mask
    from topopt.voxelize import DESIGN, FIXED, LOAD

    labels = problem.labels
    snap3 = np.asarray(snap, np.uint8).reshape(problem.shape)
    solid = ((labels == DESIGN) & (snap3 >= 128)) | (labels == FIXED) | (labels == LOAD)
    comp, n_bodies = label(solid, structure=generate_binary_structure(3, 1))
    fixed_bodies = set(np.unique(comp[support_mask(problem)]).tolist()) - {0}     # FIXED, and a spring region (section 35)
    cfg = problem.config
    masks = {lr.name: problem.load_region_ids == i for i, lr in enumerate(cfg.regions.load)}
    regions = {}
    for name, mask in masks.items():
        held = set(np.unique(comp[mask]).tolist()) - {0}
        regions[name] = {
            "used_by": [c.name for c in cfg.load_cases
                        if any(f.region == name for f in c.forces) or any(p.region == name for p in c.pressures)],
            "voxels": int(mask.sum()),
            "thresholded_design": {"connected_to_fixed": bool(held & fixed_bodies), "bodies_with_the_region": len(held)}}
    out = {"rule": LOAD_PATH_RULE, "thresholded_design_bodies": int(n_bodies),
           "thresholded_design_bodies_with_a_fixed_region": len(fixed_bodies), "regions": regions,
           "final_stl": None, "final_stl_rule": FINAL_STL_RULE}
    if final_stl is not None and Path(final_stl).is_file():
        stl = _stl_connectivity(problem, final_stl, masks)
        out["final_stl"] = {k: v for k, v in stl.items() if k != "regions"}
        for name, r in stl.get("regions", {}).items():
            regions[name]["final_stl"] = r
    return out


def case_load_path(facts: dict | None, case_name: str) -> dict | None:
    """The load path facts of one case: the regions that its forces use, and the ones that are not connected."""
    if not facts or "regions" not in facts:
        return None
    mine = {n: r for n, r in facts["regions"].items() if case_name in r["used_by"]}
    bad = [n for n, r in mine.items()
           if not r["thresholded_design"]["connected_to_fixed"]
           or (r.get("final_stl") is not None and not r["final_stl"]["connected_to_fixed"])]
    return {"regions": {n: {"thresholded_design": r["thresholded_design"]["connected_to_fixed"],
                            "final_stl": None if r.get("final_stl") is None else r["final_stl"]["connected_to_fixed"]}
                        for n, r in mine.items()},
            "all_connected": not bad, "not_connected": bad}


def load_path_sentence(facts: dict, lp: dict, relaxed_met: bool | None) -> str:
    """The text for a case whose load region is not connected to a fixed region (empty when all are)."""
    if not lp or lp["all_connected"]:
        return ""
    parts = []
    for name in lp["not_connected"]:
        r = facts["regions"][name]
        what = []
        if not r["thresholded_design"]["connected_to_fixed"]:
            what.append("the thresholded part has no material between load region " f"'{name}' and the fixed regions")
        stl = r.get("final_stl")
        if stl is not None and not stl["connected_to_fixed"]:
            text = (f"final.stl has no material at load region '{name}'" if not stl["touches_a_body"] else
                    f"final.stl has no body that holds load region '{name}' and a fixed region")
            if r["thresholded_design"]["connected_to_fixed"]:
                text += " (the thresholded design joins them, the smoothed mesh does not)"
            what.append(text)
        parts.append(" and ".join(what))
    body = "; ".join(parts) + "; this is not a part that carries the load. " + LOAD_PATH_OPTIONS + "."
    return ("But " + body) if relaxed_met else (body[0].upper() + body[1:])


def _peak_sentence(lim, phys, thr) -> tuple[str, list[bool]]:
    """The sentence with the voxel estimates of the peak, and a flag for each estimate: above the limit or not."""
    g = phys["peak_MPa"]
    if thr and "peak_MPa" in thr:
        t = thr["peak_MPa"]
        lo, hi = sorted((g, t))
        return (f"voxel estimate of the part stress, peak: {lo:.3g} to {hi:.3g} MPa (thresholded design {t:.3g}, "
                f"grey design {g:.3g}); limit {lim:.3g} MPa", [g > lim, t > lim])
    return f"voxel estimate of the part stress, peak: {g:.3g} MPa (grey design); limit {lim:.3g} MPa", [g > lim]


def case_summary(name: str, entry: dict, load_facts: dict | None = None) -> str:
    """One text for a load case that puts the relaxed result and the voxel estimates of the part stress together.

    The relaxed row is a measure of the optimizer (rho^0.5 scale). "Met" is about that measure, not about the part
    (A2 R1-1), and the text says so for a case with a relaxed row, met or not. The text gives facts and options, and says
    where the estimates are uncertain. It ends with the stress of the part from `topopt verify` (the P2 p99.9 and its ratio
    to the limit, with the source) when `independent_check` has it, else with the option to run verify (F4-29). It goes into
    `metrics.json` as `stress.cases.<case>.summary` and into the run log. `load_facts` (`load_path_facts`) adds a
    sentence when a load region of the case is not connected to a fixed region: the relaxed result can meet the
    rows with no material between the load and the supports.
    """
    lim, rel, phys, thr = entry["stress_limit_MPa"], entry.get("relaxed"), entry.get("physical"), entry.get("physical_thresholded")
    parts = []
    if rel:
        parts.append(f"the optimizer row is {'met' if rel['met'] else 'NOT met'} for the relaxed stress (relaxed peak "
                     f"{rel['peak_MPa']:.3g} MPa, limit {lim:.3g} MPa, g {rel['row_g']:+.4f})")
        if rel.get("peak_above_limit"):
            parts.append("the relaxed peak is above the limit (`relaxed.peak_above_limit`)")
    note = MET_NOTE if rel else ""
    lp_text = load_path_sentence(load_facts, entry.get("load_path"), rel["met"] if rel else None)
    if not phys:
        changed = (entry.get("input_changed") or {}).get("fact", "there are no physical numbers for this case")
        return ("; ".join(parts) + ". " if parts else "") + changed + (" " + lp_text if lp_text else "") + note
    if not phys["voxels"]:
        return ("; ".join(parts) + ". " + (lp_text + " " if lp_text else "")
                + "No design voxel has rho >= 0.5, so there is no voxel estimate of the part stress." + note
                + part_stress_tail(rel, entry, lim, False))
    sentence, above = _peak_sentence(lim, phys, thr)
    if all(above):
        where = ("Both voxel estimates are above the limit" if len(above) > 1 else "The voxel estimate is above the limit")
        where += (f" (the limit is {lim / phys['peak_MPa']:.2g} times the grey peak; {phys.get('voxels_above_limit', 0)} of "
                  f"{phys['voxels']} voxels are above {lim:.3g} MPa in the grey design)")
    elif any(above):
        where = (f"The two voxel estimates are on different sides of the limit ({phys.get('voxels_above_limit', 0)} of "
                 f"{phys['voxels']} voxels are above {lim:.3g} MPa in the grey design)")
    else:
        where = ("Both voxel estimates are at or below the limit" if len(above) > 1 else
                 "The voxel estimate is at or below the limit")
        if phys["peak_MPa"] > 0:      # a case that applies no load (U21) has the peak 0.0: no ratio
            where += f" (the limit is {lim / phys['peak_MPa']:.2g} times the grey peak)"
    text = ((("; ".join(parts) + ". ") if parts else "") + (lp_text + " " if lp_text else "")
            + sentence[0].upper() + sentence[1:] + ". " + where + ".")
    text += note
    peak_acc, member = ACCURACY.get("peak"), entry.get("member_check")
    unsure = "The voxel estimates have no accuracy band for the peak"
    if peak_acc and peak_acc.get("measured_max_pct", 1) < 0:
        unsure += (f": the voxel peak was {-peak_acc['measured_max_pct']:.0f} to {-peak_acc['measured_min_pct']:.0f} % below the "
                   f"reference in the measured cases ({band_sources_text(peak_acc, measured=True)})")
    if member and member.get("p10_voxels") is not None:
        unsure += f"; the members at the high-stress voxels are an estimated {member['p10_voxels']:.2g} voxels thick (p10)"
    text += " " + unsure + ". No pass or fail."
    return text + part_stress_tail(rel, entry, lim, bool(rel and rel["met"] and any(above)))


def part_stress_tail(rel, entry, lim, options: bool) -> str:
    """The end of the case summary (F4-29): the stress of the part, or the way to get it. Empty for a case with no relaxed row.

    With the P2 p99.9 of `topopt verify` in `independent_check` (`found`): the value, its ratio to the limit and its source. Without it:
    one option to run verify. `options` (the relaxed row is met and a voxel estimate is above the limit) adds the other options.
    When that verify result is partial (`independent_check.verify_partial`, TC-2), one more sentence gives the `partial:` line of its coverage.
    """
    if not rel:
        return ""
    ic = entry.get("independent_check") or {}
    p2 = ic.get("p2_p99_9_by_volume_MPa") if ic.get("found") else None
    if p2 is None:
        return (" options: a lower stress_limit, a finer voxel size, or " + VERIFY_OPTION if options
                else " option: " + VERIFY_OPTION)
    text = (f" The stress of the part from the independent check: P2 p99.9 by volume {p2:.3g} MPa, {p2 / lim:.2g} times the limit "
            f"{lim:.3g} MPa (tets of {ic.get('verified_file') or 'the finished mesh'}, quadratic elements; source {ic.get('file', VERIFY_FILE)}, "
            f"{ic.get('field', VERIFY_FIELD)}).")
    if ic.get("verify_partial") and isinstance(ic.get("verify_coverage"), dict):      # TC-2: the P2 value stands; the other checks of that verify run do not all stand
        from .check_coverage import line
        text += f" That verify result is {line(True, ic['verify_coverage'])}."
    return text + (" options: a lower stress_limit, or a finer voxel size" if options else "")


def summary_log_lines(section: dict | None) -> list[str]:
    """Run log lines for the stress section of `metrics.json`: one fact line for each load case."""
    if not section or "cases" not in section:
        return []
    return [f"fact: stress {name}: {case['summary']}" for name, case in section["cases"].items() if case.get("summary")]


def refresh_from_verify(metrics, run_dir) -> bool:
    """Read the P2 value of `verify/verify.json` again into the stress cases of `metrics` (a dict from `result/metrics.json`), in memory (F4-29).

    The post step of a run writes `independent_check` and `summary` before `topopt verify` has run, and nothing writes them again
    (`update_metrics` does, with a solve). A reader of a finished run (`run_result`, the levers of the run) calls this, so that the
    text next to "met" is true now: the P2 p99.9 and its ratio to the limit when verify has run, else the option to run it. Only a case
    whose `independent_check` differs from the file gets a new `independent_check` and a new `summary`. No file is written and the
    function never raises. Returns True when a case changed.
    """
    try:
        section = metrics.get("stress") if isinstance(metrics, dict) else None
        cases = section.get("cases") if isinstance(section, dict) else None
        if not isinstance(cases, dict):
            return False
        changed = False
        for name, entry in cases.items():
            old = entry.get("independent_check") if isinstance(entry, dict) else None
            if not isinstance(old, dict) or not isinstance(entry.get("stress_limit_MPa"), (int, float)):
                continue
            new = independent_check(run_dir, name, float(entry["stress_limit_MPa"]), old.get("voxel_p99_9_by_volume_MPa") or 0.0)
            if new != old:
                entry["independent_check"] = new
                entry["summary"] = case_summary(name, entry, section.get("load_paths"))
                changed = True
        return changed
    except Exception as exc:    # a reader must not stop for a text
        log.warning("stress report: verify numbers not read again: %s", exc)
        return False


def stress_section(run_dir, run=None, history=None, problem=None, stress=None, cache=None, load_paths=None) -> dict:
    """Build the `stress` section for `metrics.json`. An error becomes `{"error": ...}`, except a resource error (a memory error, a failed thread
    start, a native library that cannot be mapped): it raises `PostResourceError` and the post step stops (CONTRACTS.md section 19, #449).

    `run` and `history` are read from the run folder when not given. `problem` is rebuilt from the
    YAML of the run when not given (the report then checks that the forces and the material still equal
    those of `run.json`). `stress` (the result of `compute_stress`) can be passed in. `cache` (a dict) gets
    the result of `compute_stress` under "stress", so that the refine step does not solve again.
    `load_paths` (I6) is the result of `load_path_facts` for this run that the caller already made (`post.finish` writes it as
    `metrics.load_paths` in every run): the section uses it and makes no second check of `final.stl`.
    """
    run_dir = Path(run_dir)
    try:
        return _stress_section(run_dir, run, history, problem, stress, cache, load_paths)
    except Exception as exc:   # the report must not stop the post-process, except for a resource error (#449)
        reraise_if_resource(exc, "the stress report", missing=STRESS_MISSING)
        log.warning("stress report failed: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}", "note": "the stress report was not made"}


def _stress_log(run_dir, what: str):
    """A writer of one line of `run.log` for the stress solve of `what` (gpu-mg case C with joints: the extension of its harmonic columns, which starts cold in
    each stress solve and is the larger part of its cost, P2d-D3h). No file and no key of a JSON file."""
    from topopt.runinfo import append_log_line
    return lambda line: append_log_line(run_dir, f"stress report, {what}: {line}")


def _stress_section(run_dir, run, history, problem, stress, cache, load_paths=None) -> dict:
    from topopt.post import _last_snapshot, _read_history, build_field
    from topopt.refine.build import problem_from_run
    from topopt.refine.stress import compute_stress

    run = json.loads((run_dir / "run.json").read_text()) if run is None else run
    history = _read_history(run_dir) if history is None else history
    spec = run.get("stress")
    if not spec:
        return {"error": "this run has no stress row (run.json has no 'stress' key)"}
    rebuilt = problem is None
    problem = problem_from_run(run_dir, run) if rebuilt else problem
    changes = input_changes(problem, run) if rebuilt else {"cases": {}, "all": [], "not_compared": []}
    skip = {c["name"] for c in spec["cases"] if changes["all"] or c["name"] in changes["cases"]}
    snap = np.frombuffer(_last_snapshot(run_dir).read_bytes(), np.uint8)
    last = history[-1] if history else None
    if load_paths is None:   # `post.finish` passes the facts that it made for `metrics.load_paths`: one check of final.stl for the run
        load_paths = load_path_facts(problem, snap, run_dir / "result" / "final.stl")
    cases, phys, solid, scope = {}, {}, None, None
    thr_phys, thr_error, thr_seconds = {}, None, None
    if len(skip) < len(spec["cases"]):
        if stress is None:
            stress = compute_stress(problem, snap, log=_stress_log(run_dir, "grey design"), solver=run.get("solver"))
        if cache is not None:
            cache["stress"] = stress
        solid = build_field(snap.astype(np.float32).reshape(problem.shape) / 255.0, problem.labels) >= 0.5
        limits = {c["name"]: c["stress_limit_MPa"] for c in spec["cases"]}
        scope = peak_scope(stress, problem)
        phys = {p["name"]: p for p in physical_statistics(stress, limits, float(problem.h), scope)}
        # the thresholded design (R1-3): the same snapshot, rho >= 0.5 solid and the rest void; one more factor and
        # solve for all cases. It never stops the report.
        try:
            t0 = time.monotonic()
            thr_stress = compute_stress(problem, np.where(snap >= 128, 255, 0).astype(np.uint8), log=_stress_log(run_dir, "thresholded design"),
                                        solver=run.get("solver"))
            thr_phys = {p["name"]: p for p in physical_statistics(thr_stress, limits, float(problem.h),
                                                                  peak_scope(thr_stress, problem))}
            thr_seconds = round(time.monotonic() - t0, 2)
        except Exception as exc:
            reraise_if_resource(exc, "the thresholded stress estimate of the stress report", missing=STRESS_MISSING)
            log.warning("thresholded stress estimate failed: %s", exc)
            thr_error = f"{type(exc).__name__}: {exc}"
    for c in spec["cases"]:
        name, lim = c["name"], c["stress_limit_MPa"]
        entry = {"stress_limit_MPa": lim, "limit_from": c["limit_from"],
                 "allowable_mpa": c["allowable_mpa"], "allowable_source": c["allowable_source"],
                 "safety_factor": c["safety_factor"], "relaxed": relaxed_numbers(last, name),
                 "load_path": case_load_path(load_paths, name)}
        if name in skip:
            diffs = list(changes["cases"].get(name, [])) + changes["all"]
            entry.update(physical=None, achieved_safety_factor=None, member_check=None, verdict=None,
                         input_changed={"fact": "the inputs of this run changed after the run, so there are no "
                                                "physical numbers for this case. Restore the YAML or run again.",
                                        "differences": diffs})
            cases[name] = entry
            continue
        p = phys[name]
        member = member_check(stress, solid, problem.origin, problem.h, list(stress["case_names"]).index(name), scope)
        n_vox = p["voxels"]
        v_peak = verdict("peak", p["peak_MPa"], lim, member, n_vox)
        v_p = verdict("p99_9", p["p99_9_by_volume_MPa"], lim, member, n_vox, peak=p["peak_MPa"])
        against_limit = {"peak": safety_entry("peak", lim, p["peak_MPa"], p, member),
                         "p99_9": safety_entry("p99_9", lim, p["p99_9_by_volume_MPa"], p, member)}
        against_allowable = ({"peak": safety_entry("peak", c["allowable_mpa"], p["peak_MPa"], p, member),
                              "p99_9": safety_entry("p99_9", c["allowable_mpa"], p["p99_9_by_volume_MPa"], p, member)}
                             if c["allowable_mpa"] else None)
        factors = {"against_limit": against_limit, "against_allowable": against_allowable}
        t = thr_phys.get(name)
        entry.update({
            "physical": {**{k: v for k, v in p.items() if k != "name"}, "design": GREY_LABEL},
            "physical_thresholded": ({**{k: v for k, v in t.items() if k != "name"}, "label": THRESHOLDED_LABEL}
                                     if t else {"error": thr_error or "not made", "label": THRESHOLDED_LABEL}),
            "achieved_safety_factor": {**factors, "text": safety_text({**c, "stress_limit_MPa": lim}, p, factors, member)},
            "member_check": member,
            "verdict": {"peak": v_peak, "p99_9": v_p},
            "independent_check": independent_check(run_dir, name, lim, p["p99_9_by_volume_MPa"])})
        if c.get("explicit_stress_limit_MPa") is not None and c["allowable_mpa"]:
            entry["limit_note"] = (f"stress_limit {lim:.4g} MPa is the final limit; allowable_mpa {c['allowable_mpa']:.4g} MPa "
                                   f"is used for the achieved safety factor against the allowable only")
        if t:
            lo, hi = sorted((p["peak_MPa"], t["peak_MPa"]))
            entry["voxel_peak_range_MPa"] = {"low": lo, "high": hi,
                                             "note": "the peak of the voxel estimate in the thresholded and in the grey design"}
        cases[name] = entry
    for name, entry in cases.items():
        entry["summary"] = case_summary(name, entry, load_paths)
    stats = [k for k, a in ACCURACY.items() if a.get("verdict", True)]
    return {
        "note": ("The stress row is a target that the optimizer tries to meet. `relaxed.met` says whether it was met at "
                 "the last pass, and `stage_ends` whether it was met at the end of each beta stage. "
                 "Relaxed numbers are the optimizer's measure (rho^0.5 * D0 * strain): relaxed, not physical. "
                 "Physical numbers use the full modulus at the centre of each hex8 voxel for the voxels "
                 "with rho >= 0.5, in the grey design (`physical`) and in the thresholded design (`physical_thresholded`). "
                 "They are estimates. `summary` of each case puts the relaxed result and the voxel estimates together. "
                 + (f"A verdict exists only for: {', '.join(stats)}, with the measured error band." if stats else
                    "No statistic has a verdict at this time: the peak and p99.9 are estimates with the error range "
                    "measured on the bench and on public parts, in both directions for p99.9. That range is the largest "
                    "error measured and not a bound for other parts. `verdict.<statistic>.status` has the range, the "
                    f"source of each end and the numbers ({BENCH_FILE}, {PUBLIC_PARTS_FILE}). {PART_STRESS_CHECK}.")),
        "voxel_size_mm": float(run["h"]), "element": "hex8, one point at the voxel centre",
        "voxel_estimates": {"physical": GREY_LABEL, "physical_thresholded": THRESHOLDED_LABEL,
                            "thresholded_solve_s": thr_seconds,
                            "note": "two estimates of the part stress from the same snapshot. They can differ by tens of "
                                    "percent at the grey voxels of a stress design, and can lie on different sides of the "
                                    "limit. `voxel_peak_range_MPa` of each case gives the two peaks"},
        "parameters": {"q": spec["q"], "relaxation_power": spec["power"], "ks_scale": spec["scale"],
                       "step_acceptance": spec.get("step_acceptance")},
        "step_acceptance": step_acceptance_summary(history),
        "stage_ends": stage_ends(history),
        "load_paths": load_paths,
        "input_check": ({"compared_with_run_json": ("" if any(t.startswith("forces:") for t in changes["not_compared"]) else "load case forces, ")
                         + ("material.density" if any(t.startswith("material.E") for t in changes["not_compared"]) else "material"),
                         "not_compared": changes["not_compared"], "changed": bool(skip)} if rebuilt else None),
        "accuracy_data": f"bench/stress_accuracy.json (B-051); public parts: {PUBLIC_PARTS_FILE} (F1-8)",
        "cases": cases,
    }


def update_metrics(run_dir) -> dict:
    """Make the stress section again and write it into `result/metrics.json`. Return the section. A resource error raises `PostResourceError`
    (`stress_section`), and the file is not changed."""
    run_dir = Path(run_dir)
    path = run_dir / "result" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["stress"] = stress_section(run_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(metrics, indent=1))
    tmp.replace(path)
    return metrics["stress"]
