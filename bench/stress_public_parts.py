"""F1-8: the voxel p99.9 against the P2 p99.9 of `topopt verify` on 5 public parts (Refs #129).

The stress report (`topopt.stress_report`) gives a voxel estimate of the p99.9 stress. `bench/stress_accuracy.py` measured its
error on a beam, a plate with a hole and one real part (the band of `ACCURACY["p99_9"]`). This study measured the same
error on 5 public parts: the hook, the bracket, the ladle, and a shelf bracket and a phone stand that the part builder made
(the specs are in `bench/stress_public_parts_specs/`). The data file is `bench/stress_public_parts.json`: a header, the
pre-registered tests and their result, 51 cells (part, load case, run type) and a summary that is a function of the cells.

The run folders of the measurement are not in the repository. The JSON holds everything that the texts and the tests need,
and loading it needs no run folder (`load`). To make the file again you need the two result files of the measurement:

    python bench/stress_public_parts.py --from DIR [--out bench/stress_public_parts.json]

where DIR holds `summary.json` (one record per run) and `analysis.json` (the cells and the pre-registered tests). The script
checks that each cell equals the mean of the ratios of the runs that `summary.json` lists for it, and writes the file.

`fact_text` makes the sentence that the texts of `topopt.stress_report` and of `bench/stress_accuracy.json` quote. The texts in
`topopt/stress_report.py` are literal copies of it (the package must not read `bench/`); `tests/test_stress_public_parts.py`
checks that each copy equals the output of `fact_text` for the summary in the JSON.

F1-8c (Refs #129): the band of the voxel stress in `bench/stress_accuracy.json` is now the union of the bench and these cells
(`stress_accuracy.derive_rule`). The summary here keeps counting against the BENCH band alone (`sources.bench` of
`verdict_rule.p99_9`): the union contains every cell by construction, so "outside the band" has a meaning only for the bench
band. After a change of the bench band, or of `summarize`, make the summary again from the committed cells (no run folder):

    python bench/stress_public_parts.py --resummarize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "stress_public_parts.json"
SOURCE = "bench/stress_public_parts.json"
ACCURACY_FILE = HERE / "stress_accuracy.json"
MASS_OBJECTIVE_TYPES = ("MS", "MF")     # the run types with the objective `mass` (the grey designs of the study)
VOXEL_COUNT_PEAK = 5000                 # below this many voxels in scope the voxel "p99.9" rests on the top 1 to 4 voxels

HEADER = {
    "format": 1,
    "what": ("The voxel p99.9 stress of the stress report (`topopt.stress_report`) against the P2 p99.9 of `topopt verify`, on 5 "
             "public parts: the hook, the bracket, the ladle, and a shelf bracket and a phone stand made with the part builder. "
             "51 cells (part x load case x run type). A measurement only: no code changed."),
    "issue": "chartergroup/topopt#129 (F1-8, step 1)",
    "main": "74a1e46",
    "host": "one shared Linux x86_64 host (PARDISO, 8 threads, one job at a time); not Apple silicon",
    "date": "2026-10-25",
    "script": "bench/stress_public_parts.py",
    "run_folders": ("not in the repository. The cells below are the data. The 24 runs of the hook, the bracket and the ladle come from "
                    "the F1-3 series (main d5002a4); the other runs were made on main 74a1e46. The shelf bracket and the phone stand "
                    "are built from the specs in `bench/stress_public_parts_specs/` (`python -m topopt.regions_builder SPEC --out DIR`)."),
    "definitions": {
        "voxel_value": ("`metrics.stress.cases.<case>.independent_check.voxel_p99_9_by_volume_MPa`: the p99.9 by volume of the GREY "
                        "design (the SIMP snapshot as the optimizer ran it; full E; hex8 centre; voxels with rho >= 0.5 in the peak scope)"),
        "p2_value": ("`verify/verify.json` `cases.<case>.von_mises.p99_9_by_volume_MPa`: `topopt verify` at its default mesh size "
                     "(tets of final.stl, quadratic elements, all tets count)"),
        "p2_over_voxel": ("P2 / voxel (`independent_check.p2_over_voxel`). Above 1 the voxel value is below P2. A cell with a "
                          "repeat run holds the mean of the two ratios, and the mean of the two values"),
        "voxel_error_pct": "100 * (voxel / P2 - 1) = 100 * (1 / p2_over_voxel - 1): the error of the voxel value against P2, as in the bench",
        "cell": "(part, case, run type, variant). The variants S1, S2 and S3 are other designs of the same part (other pass counts)",
        "run_types": {
            "R": "compliance objective with the volume row, no stress row",
            "CS": "compliance objective with the volume row and stress rows (0.9 x the limits of MS)",
            "MS": "objective mass with stress rows, objective_scale stage",
            "MF": "as MS with objective_scale first",
        },
        "member_p10_voxels": "`member_check.p10_voxels` of the case: the estimated voxels through the member at the 1 % highest stress voxels",
        "thinnest_member_voxels": "`metrics.print.thinnest_member.thinnest_member_mm / voxel size` of the run",
        "near_share": ("`verify.json` `von_mises.near_regions.top_volume_share_near.any`: the share of the P2 volume at or above its "
                       "p99.9 that lies within 2 mm of a fixed or load region"),
        "p2_peak_over_voxel_peak": "P2 peak at the quadrature points over the voxel peak (`peak_qp_over_voxel_peak`)",
        "grayness_last_pass": "`history.jsonl` key `grayness` of the last pass",
    },
    "parts": {
        "hook": {"spec": "examples/hook/hook.yaml (F1-3 runs)", "voxel_mm": 2.4},
        "bracket": {"spec": "examples/interfaces/bracket.yaml (F1-3 runs)", "voxel_mm": 1.0},
        "ladle": {"spec": "examples/ladle/ladle.yaml (F1-3 runs)", "voxel_mm": 3.25},
        "shelf_bracket": {"spec": "bench/stress_public_parts_specs/shelf_bracket_R.yaml and shelf_bracket_MS_spec.yaml", "voxel_mm": 1.6},
        "phone_stand": {"spec": "bench/stress_public_parts_specs/phone_stand_R.yaml and phone_stand_MS_spec.yaml", "voxel_mm": 2.45},
        "note": ("`*_R.yaml` has no stress limit (type R). `*_MS_spec.yaml` has the stress limits (the relaxed peak of the R run at its "
                 "last pass, 4 digits); type CS uses 0.9 x these limits. The part builder does not copy `settings.objective`: "
                 "the YAML that it writes got `objective: mass` by hand for the types MS and MF."),
    },
    "preregistration": ("The rules, the definitions and the four tests were written before the first ratio was computed "
                        "(2026-10-25, 09:25Z; three dated amendments before the runs they cover). The file is not in the repository."),
}


def _sig(x, digits=4):
    """Round to `digits` significant digits (None stays None)."""
    return None if x is None else float(f"{x:.{digits}g}")


def error_pct(ratio) -> float:
    """The error of the voxel value against P2 in percent, from p2_over_voxel."""
    return 100.0 * (1.0 / ratio - 1.0)


def cell_label(c: dict) -> str:
    """The name of a cell in the texts: part, load case, run type, voxel size (and the variant, if it is not the base)."""
    return (f"{c['part']}, {c['case']}, run type {c['run_type']}, voxel {c['voxel_mm']:g} mm"
            + ("" if c["variant"] == "base" else f", variant {c['variant']}"))


def bench_band(path: Path = ACCURACY_FILE) -> dict:
    """The band of the BENCH alone for the voxel p99.9 (`sources.bench` of `verdict_rule.p99_9` in `bench/stress_accuracy.json`)."""
    return json.loads(Path(path).read_text())["verdict_rule"]["p99_9"]["sources"]["bench"]


def make_cell(c: dict) -> dict:
    """One cell of the JSON from one cell of `analysis.json`."""
    ratio = _sig(c["ratio"])
    return {
        "part": c["part"], "case": c["case"], "run_type": c["type"], "variant": c["variant"], "runs": list(c["runs"]),
        "voxel_mm": _sig(c["h"]),
        "voxel_p99_9_MPa": _sig(c["vox_p999"]), "p2_p99_9_MPa": _sig(c["p2_p999"]),
        "p2_over_voxel": ratio, "voxel_error_pct": round(error_pct(ratio), 2),
        "voxels_in_scope": _sig(c["n_vox"]), "p99_9_equals_peak": bool(c["eq_peak"]),
        "member_p10_voxels": _sig(c["v_p10"]), "thinnest_member_voxels": _sig(c["v_thin"]),
        "near_share": _sig(c["near_any"]), "p2_peak_over_voxel_peak": _sig(c["peak_qp_ratio"]),
        "grayness_last_pass": _sig(c["gray"]),
    }


def _stats(values) -> dict:
    a = np.asarray(values, float)
    q25, q75 = np.percentile(a, [25, 75])
    return {"min": round(float(a.min()), 3), "max": round(float(a.max()), 3), "max_over_min": round(float(a.max() / a.min()), 3),
            "median": round(float(np.median(a)), 3), "q25": round(float(q25), 3), "q75": round(float(q75), 3)}


def summarize(cells: list[dict], under_pct: float, over_pct: float, gate_min_voxels: float) -> dict:
    """The summary of the cells. A function of the cells and of three numbers that the code owns.

    `under_pct` and `over_pct` are the band of the BENCH alone for the voxel p99.9 (`bench_band`: `sources.bench` of
    `stress_report.ACCURACY["p99_9"]`, not the union with these cells), `gate_min_voxels` the member gate
    (`stress_report.MIN_MEMBER_VOXELS_EST`). The tests recompute this from the cells in the JSON.
    """
    ratio = np.array([c["p2_over_voxel"] for c in cells], float)
    err = 100.0 * (1.0 / ratio - 1.0)
    below, above = err < -under_pct, err > over_pct         # outside the bench band: the voxel value below / above it
    low, high = int(np.argmin(err)), int(np.argmax(err))
    gate = np.array([c["member_p10_voxels"] >= gate_min_voxels for c in cells], bool)
    above_p2 = ratio < 1.0
    mass = np.array([c["run_type"] in MASS_OBJECTIVE_TYPES for c in cells], bool)
    peak = np.array([c["p2_peak_over_voxel_peak"] for c in cells], float)
    types = sorted({c["run_type"] for c in cells}, key=("R", "CS", "MS", "MF").index)
    return {
        "n_cells": len(cells),
        "n_parts": len({c["part"] for c in cells}),
        "parts": sorted({c["part"] for c in cells}),
        "cells_by_part": {p: sum(c["part"] == p for c in cells) for p in sorted({c["part"] for c in cells})},
        "p2_over_voxel": _stats(ratio),
        "voxel_error_pct": {"min": round(float(err.min()), 1), "max": round(float(err.max()), 1),
                            "min_cell": cell_label(cells[low]), "max_cell": cell_label(cells[high])},
        "cells_voxel_below_p2": int((ratio > 1.0).sum()),
        "cells_voxel_above_p2": int(above_p2.sum()),
        "cells_voxel_above_p2_in_mass_objective_runs": int((above_p2 & mass).sum()),
        "by_run_type": {t: {"cells": sum(c["run_type"] == t for c in cells),
                            "cells_voxel_above_p2": int((above_p2 & np.array([c["run_type"] == t for c in cells])).sum()),
                            "median_p2_over_voxel": round(float(np.median([c["p2_over_voxel"] for c in cells if c["run_type"] == t])), 3)}
                        for t in types},
        "bench_band": {"under_pct": under_pct, "over_pct": over_pct,
                       "source": "bench/stress_accuracy.json verdict_rule.p99_9.sources.bench"},
        "cells_outside_bench_band": {"total": int((below | above).sum()), "voxel_below_the_band": int(below.sum()),
                                     "voxel_above_the_band": int(above.sum())},
        "member_gate": {
            "min_p10_voxels": gate_min_voxels,
            "cells_meeting": int(gate.sum()),
            "p2_over_voxel": {"min": round(float(ratio[gate].min()), 3), "max": round(float(ratio[gate].max()), 3)},
            "cells_outside_bench_band": int(((below | above) & gate).sum()),
            "cells_above_bench_band": int((above & gate).sum()),
            "cells_below_bench_band": int((below & gate).sum()),
        },
        "cells_p99_9_equals_peak": int(sum(c["p99_9_equals_peak"] for c in cells)),
        "voxel_peak": {"cells_below_p2_peak": int((peak > 1.0).sum()),
                       "p2_peak_over_voxel_peak": {"min": round(float(peak.min()), 3), "max": round(float(peak.max()), 3)}},
    }


def fact_text(summary: dict) -> str:
    """The public-part fact for the texts: the error range with the cell of each end, its direction, the bench band, the member gate,
    the source. No advice. "Outside the band" is about the BENCH band: the band of the report contains every cell by construction."""
    s, g, band, err = summary, summary["member_gate"], summary["bench_band"], summary["voxel_error_pct"]
    out = s["cells_outside_bench_band"]["total"]
    return (f"on {s['n_parts']} public parts ({s['n_cells']} cells) the error of the voxel p99.9 against the P2 p99.9 of "
            f"`topopt verify` (voxel / P2 - 1) was {err['min']:+.0f} % ({err['min_cell']}) to {err['max']:+.0f} % ({err['max_cell']}), "
            f"in both directions: the voxel value was above P2 in {s['cells_voxel_above_p2']} cells, mostly grey mass-objective "
            f"designs, and up to {s['p2_over_voxel']['max']:.2g} times below P2 on a coarse part with thin members; {out} of the "
            f"{s['n_cells']} cells were outside the bench band ({-band['under_pct']:+g} % to {band['over_pct']:+g} %, "
            f"bench/stress_accuracy.json), and the member gate did not keep the error inside the bench band "
            f"({g['cells_above_bench_band']} of the {g['cells_meeting']} cells that met it were above +{band['over_pct']:g} %); "
            f"source {SOURCE}")


def tests_block(analysis: dict, summary: dict) -> dict:
    """The four pre-registered tests and their result: the verdicts, the voxel-size pairs and the near shares come from
    `analysis.json`; the ratio statistics of H1 and H2 come from the summary, so that the file agrees with itself."""
    h1, h2, h3, h4 = analysis["H1"], analysis["H2"], analysis["H3"], analysis["H4"]
    ratio = summary["p2_over_voxel"]
    pairs = h3["pairs"]
    return {
        "H1": {"rule": "one band: max / min of p2_over_voxel over the cells <= 1.5", "result": h1["verdict"],
               "max_over_min": ratio["max_over_min"]},
        "H2": {"rule": "the finding is general: the median of p2_over_voxel >= 1.5", "result": h2["verdict"],
               "median": ratio["median"], "q25": ratio["q25"], "q75": ratio["q75"]},
        "H3": {"rule": ("resolution: Spearman rho of p2_over_voxel with member_p10_voxels <= -0.6, and the ratio falls with the "
                        "finer voxel in every voxel-size pair"),
               "result": h3["verdict"], "spearman_rho_p10": round(h3["a_p10"]["rho"], 3),
               "pairs": len(pairs), "pairs_that_fall": int(sum(p["falls"] for p in pairs)),
               "note": "the 4 pairs that fall are the hook; the bracket pair rises; the bracket pair with a mass objective could not be made (verify could not mesh it). The pairs are not cells of this file"},
        "H4": {"rule": ("hot spot at a region: in the 3 cells with the largest ratio, more than half of the P2 volume at or above its "
                        "p99.9 lies within 2 mm of a fixed or load region"),
               "result": h4["verdict"], "top3_near_share": [round(t["near_any"], 3) for t in h4["top3"]],
               "note": ("all three cells are the hook, load case pull_off; 44 to 45 % of the hook volume is within 2 mm of a region, "
                        "and the gap stays at 1.6 to 2.0 times when those points are left out")},
    }


def limits_block(analysis: dict) -> list[str]:
    """Facts about the limits of the measurement. The numbers come from `analysis.json`."""
    ch = [100.0 * (m["p2_fine"] / m["p2_default"] - 1.0) for m in analysis["mesh_pairs"]]
    return [
        "one host; run-to-run noise: the median difference between a run and its repeat is "
        f"{analysis['noise']['median_abs_pct']:.1f} % of the ratio ({analysis['noise']['n']} pairs)",
        "5 parts; the cells of one part are not independent. The hook is the only part with ratios above 1.5 (besides one phone stand cell)",
        f"P2 is not a converged reference: a second, finer verify mesh on {len(analysis['mesh_pairs'])} cases changed the P2 p99.9 "
        f"by {min(ch):+.1f} % to {max(ch):+.1f} %; the P2 peak moves more",
        f"cells with fewer than {VOXEL_COUNT_PEAK} voxels in scope have a voxel p99.9 that rests on the top 1 to 4 voxels (`p99_9_equals_peak`)",
        "no part has a thin-wall shell, a bore or a fillet; no pressure load, moment, spring or joint",
    ]


def build(src: Path) -> dict:
    """The document from `summary.json` and `analysis.json` in the folder `src`."""
    from topopt.stress_report import MIN_MEMBER_VOXELS_EST

    runs = json.loads((src / "summary.json").read_text())
    analysis = json.loads((src / "analysis.json").read_text())
    cells = []
    for c in analysis["cells"]:
        mine = [runs[r]["cases"][c["case"]]["ratios"]["p2_over_voxel"] for r in c["runs"]]     # the integrity check
        assert abs(sum(mine) / len(mine) - c["ratio"]) < 1e-9, (c["part"], c["case"], c["type"], c["variant"])
        cells.append(make_cell(c))
    bench = bench_band()
    summary = summarize(cells, bench["under_pct"], bench["over_pct"], MIN_MEMBER_VOXELS_EST)
    return {**HEADER, "tests": tests_block(analysis, summary), "limits": limits_block(analysis), "summary": summary,
            "cells": cells}


def dumps(doc: dict) -> str:
    """JSON text with one cell on each line (`cells` is the last key)."""
    head = json.dumps({k: v for k, v in doc.items() if k != "cells"}, indent=1)
    rows = ",\n".join("  " + json.dumps(c) for c in doc["cells"])
    return head[:-2] + ',\n "cells": [\n' + rows + "\n ]\n}\n"


def load(path: Path = DATA) -> dict:
    """The data file. It needs no run folder."""
    return json.loads(Path(path).read_text())


def resummarize(path: Path = DATA) -> dict:
    """The document with its `summary` made again from its own cells and the bench band alone. No run folder is read."""
    from topopt.stress_report import MIN_MEMBER_VOXELS_EST

    doc = load(path)
    bench = bench_band()
    doc["summary"] = summarize(doc["cells"], bench["under_pct"], bench["over_pct"], MIN_MEMBER_VOXELS_EST)
    return doc


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])    # __doc__ is None under python -OO
    ap.add_argument("--from", dest="src", metavar="DIR", help="the folder with summary.json and analysis.json")
    ap.add_argument("--resummarize", action="store_true",
                    help="no run folder: make `summary` again from the cells in --out and the bench band, and write it back")
    ap.add_argument("--out", default=str(DATA))
    a = ap.parse_args()
    if a.resummarize:
        doc = resummarize(Path(a.out))
        Path(a.out).write_text(dumps(doc))
        print("wrote", a.out, "-", doc["summary"]["n_cells"], "cells")
        print(fact_text(doc["summary"]))
        return
    if not a.src:
        ap.error("--from DIR or --resummarize is required")
    doc = build(Path(a.src))
    Path(a.out).write_text(dumps(doc))
    print("wrote", a.out, "-", doc["summary"]["n_cells"], "cells")
    print(fact_text(doc["summary"]))


if __name__ == "__main__":
    main()
