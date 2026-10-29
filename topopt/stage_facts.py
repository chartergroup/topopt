"""Facts at the last pass of a beta stage and at the last pass of the run (P3a, CONTRACTS.md section 13, "Facts at the end of a stage").

A fact is a number or a sentence. None of them is a test, a limit or a warning rule: the tool says what it measured, and the agent or the user decides.
The facts are made at the last pass of each beta stage and at the last pass of the run, from the density of that pass. They cost no solve.
`run()` keeps them in the history record of that pass as `stage_end`, writes them to the run log, and `build_verdict` copies the ones of the last pass.

- `volume`: how much of the allowed volume the design uses. The volume row is a ceiling: a design far under it counts as "met" in the verdict,
  and nothing else says how much of the allowed volume the run used. Only a run that has a row named `volume` gets this fact.
- `load_path`: which load regions the thresholded design (density 0.5) does not join to a support (a fixed, spring or joint region). It is the
  rule of `stress_report.load_path_facts`, made at each stage end. At a low beta the design is grey, so the thresholded part is not the
  final part: the fact carries the beta of the stage.
- `material` (F1-3a; a run with `settings.objective: mass` and no row named `volume`): the volume and the mass of the thresholded part (density 0.5),
  the material that the run uses. A mass run has no volume row to say it, and the verdict counts a row under its limit as met.
- `grey` (F1-3a; a run with `settings.objective: mass`, only when the grayness of the pass is above `settings.stop_rules.grayness_max`): the design is grey,
  so the thresholded part depends on the cut. The record has the grayness, the volume and the body count of the thresholded part at the cuts 0.4, 0.5 and
  0.6, and the volume spread between 0.4 and 0.6 as a share of the volume at 0.5. It uses `threshold_stability`; no new number: the grayness limit and
  the three cuts exist already. The body count of `final.stl` is known only in the post step (`grey_design_record`).

- `threshold_stability` (I6, `metrics.threshold_stability`): how the thresholded part changes when the cut moves from 0.5 to 0.4 and to 0.6. It is made
  once, in the post step, from the last snapshot, and (F1-3a) at a stage end of a grey mass run. A fact only: no gate, no warning, no change of the verdict.

The module needs no numpy at import time (the verdict module imports it).
"""

from __future__ import annotations

VOLUME_ROW = "volume"       # the name of the volume row (`volume_constraint`)
THRESHOLD = 0.5             # the print threshold of the thresholded part: the density of `stress_report.load_path_facts` and of the final mesh
STABILITY_LEVELS = (0.4, 0.6)   # the two other cuts of `threshold_stability`, one on each side of THRESHOLD (I6)


def grayness(rho) -> float:
    """`mean(4 rho (1 - rho))` over the design voxels: the `grayness` of a history record and the number that the verdict tests against `grayness_max`."""
    import numpy as np

    rho = np.asarray(rho)
    return float(np.mean(4 * rho * (1 - rho)))


def cut_byte(level: float) -> int:
    """The snapshot byte at which a design voxel is solid for the cut `level`: 0.4 -> 102, 0.5 -> 128 (127.5 rounds up, the rule of `thresholded_fraction`), 0.6 -> 153."""
    import math

    return math.ceil(level * 255 - 1e-9)


def thresholded_fraction(rho) -> float:
    """The share of the design voxels with density `THRESHOLD` or more. `rho` is the density of the design voxels (`Model.density`).

    A snapshot stores round(255 rho) and `load_path_facts` tests the byte for 128 or more: the same rule as `rho >= THRESHOLD`.
    """
    import numpy as np

    rho = np.asarray(rho)
    return float((rho >= THRESHOLD).mean()) if rho.size else 0.0


STABILITY_RULE = ("voxel volume and 6-connected body count of the thresholded design at each level: the design voxels with a snapshot density at or above "
                  "the level, plus the fixed and load voxels (the labeling of `stress_report.load_path_facts`). The volume is a voxel count times the "
                  "voxel volume, not the volume of the smoothed mesh. A snapshot stores round(255 rho), so a level cuts at the byte ceil(255 level)")


def stability_levels() -> list[float]:
    """The levels of `threshold_stability`, lowest first: `STABILITY_LEVELS` and `THRESHOLD`."""
    return sorted({*STABILITY_LEVELS, THRESHOLD})


def thresholded_solid(labels, snap, level):
    """The boolean grid of the thresholded part at the cut `level`: the design voxels with a snapshot byte at or above `cut_byte(level)`, plus the fixed and load voxels.

    `labels` is the label grid of the run and `snap` the density bytes (the same shape or flat). The one rule of `threshold_stability` and `material_fact`.
    """
    import numpy as np

    from .voxelize import DESIGN, FIXED, LOAD

    labels = np.asarray(labels, np.uint8)
    snap = np.asarray(snap, np.uint8).reshape(labels.shape)
    return ((labels == DESIGN) & (snap >= cut_byte(level))) | (labels == FIXED) | (labels == LOAD)


def threshold_stability(labels, snap, h) -> dict:
    """The thresholded part at the levels 0.4, 0.5 and 0.6: `{"rule", "levels": [{"level", "volume_mm3", "bodies"}, ...]}`, lowest level first. Never raises.

    `labels` is the label grid of the run, `snap` the density bytes of the last snapshot (the same shape or flat), `h` the voxel size in mm. The thresholded
    design at a level is the design voxels at or above the level, plus the fixed and load voxels (the forced-solid voxels): the volume is their count times
    h^3 and `bodies` is their number of face-connected (6-connected) components, as in `stress_report.load_path_facts`. A higher level holds no more
    voxels, so the volume does not rise with the level. The numbers are a fact: no level is a test, a limit or a warning, and the verdict does not read them.
    The cost is one labeling of the grid for each level (no solve). On an error the result is `{"error": text}`.
    """
    try:
        from scipy.ndimage import generate_binary_structure, label

        structure = generate_binary_structure(3, 1)
        levels = []
        for level in stability_levels():
            solid = thresholded_solid(labels, snap, level)
            _, bodies = label(solid, structure=structure)
            levels.append({"level": level, "volume_mm3": round(float(solid.sum()) * float(h) ** 3, 1), "bodies": int(bodies)})
        return {"rule": STABILITY_RULE, "levels": levels}
    except Exception as exc:    # a fact never stops the post step
        return {"error": f"{type(exc).__name__}: {exc}"}


def volume_fact(constraints, rho) -> dict | None:
    """How much of the allowed volume the design uses, or None for a run with no row named `volume`.

    `constraints` is the list of entries of the pass (`Evaluation.constraints`). The allowed volume is the limit of the volume row, a mean density
    of the design voxels. The keys: `limit` (the allowed mean density), `projected` (the mean density of the projected design, the value of the row),
    `thresholded` (the share of the design voxels at density 0.5 or more), `used_projected` and `used_thresholded` (each of them divided by the limit).
    """
    row = next((c for c in constraints if c.get("name") == VOLUME_ROW), None)
    if row is None or not row.get("limit"):
        return None
    limit, projected, thresholded = float(row["limit"]), float(row["value"]), thresholded_fraction(rho)
    return {"limit": limit, "projected": projected, "thresholded": thresholded,
            "used_projected": projected / limit, "used_thresholded": thresholded / limit}


def load_path_fact(problem, snap) -> dict:
    """Which load regions the thresholded design does not join to a support. `snap` is the density bytes of the full grid (`Model.density_bytes`).

    The keys: `regions` (`{name: joined}`, for each load region), `not_joined` (the names of the regions that are not joined), `bodies` (the joined
    bodies of the thresholded design, with the fixed, load and spring voxels), `bodies_with_a_support`. Or `{"error": text}`: the check never raises.
    """
    from .stress_report import load_path_facts

    facts = load_path_facts(problem, snap)      # one face-connected labeling (scipy.ndimage.label), no solve
    if "error" in facts:
        return {"error": facts["error"]}
    regions = {name: bool(r["thresholded_design"]["connected_to_fixed"]) for name, r in facts["regions"].items()}
    return {"regions": regions, "not_joined": [n for n, joined in regions.items() if not joined],
            "bodies": int(facts["thresholded_design_bodies"]),
            "bodies_with_a_support": int(facts["thresholded_design_bodies_with_a_fixed_region"])}


def is_mass_run(problem) -> bool:
    """True for a run with `settings.objective: mass`. The facts `material` and `grey` are for these runs only (F1-3a)."""
    try:
        return problem.config.settings.objective == "mass"
    except AttributeError:
        return False


def material_fact(problem, snap) -> dict:
    """The material of the thresholded part: `{"volume_mm3", "mass_g", "density_g_cm3"}`, or `{"error": text}`. Never raises.

    The part at the cut `THRESHOLD` is the design voxels at or above it plus the fixed and load voxels (the rule of `threshold_stability` at 0.5).
    The volume is their count times h^3, the mass is that volume times `material.density`. It is not the mass of the finished mesh (`metrics.mass_final_g`).
    """
    try:
        volume = float(thresholded_solid(problem.labels, snap, THRESHOLD).sum()) * float(problem.h) ** 3
        density = float(problem.config.material.density)
        return {"volume_mm3": round(volume, 1), "mass_g": round(volume / 1000.0 * density, 3), "density_g_cm3": density}
    except Exception as exc:    # a fact never stops the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def grey_fact(problem, snap, rho) -> dict | None:
    """The grey design fact, or None when the grayness of the pass is at or below `grayness_max` (the test of the verdict). Never raises.

    The keys: `grayness`, `grayness_max`, `levels` (the `levels` of `threshold_stability`: the volume and the body count of the thresholded part at the
    cuts 0.4, 0.5 and 0.6) and `volume_spread` (the volume at 0.4 minus the volume at 0.6, divided by the volume at 0.5; None when the part is void at 0.5).
    On an error of the cut check: `grayness`, `grayness_max` and `error`.
    """
    try:
        gray, limit = grayness(rho), float(problem.config.settings.stop_rules.grayness_max)
        if not gray > limit:
            return None
        out = {"grayness": gray, "grayness_max": limit}
        stability = threshold_stability(problem.labels, snap, problem.h)
        if "error" in stability:
            return {**out, "error": stability["error"]}
        by = {lv["level"]: lv["volume_mm3"] for lv in stability["levels"]}
        low, high = STABILITY_LEVELS
        mid = by[THRESHOLD]
        return {**out, "volume_spread": (by[low] - by[high]) / mid if mid else None, "levels": stability["levels"]}
    except Exception as exc:    # a fact never stops the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def stage_end(problem, snap, rho, constraints, beta, next_beta) -> dict:
    """The facts at the last pass of a beta stage (`next_beta` is None at the last pass of the run). The history record of that pass keeps it as `stage_end`."""
    out = {"beta": float(beta), "next_beta": None if next_beta is None else float(next_beta)}
    volume = volume_fact(constraints, rho)
    if volume is not None:
        out["volume"] = volume
    out["load_path"] = load_path_fact(problem, snap)
    if is_mass_run(problem):    # F1-3a: a mass run has no volume row to say how much material it uses, and its design can stay grey
        if volume is None:
            out["material"] = material_fact(problem, snap)
        grey = grey_fact(problem, snap, rho)
        if grey is not None:
            out["grey"] = grey
    return out


def volume_text(volume: dict) -> str:
    """The sentence for a `volume_fact`: both numbers are shares of the allowed volume."""
    return (f"volume: {volume['used_projected']:.2f} of the allowed volume used (projected design), "
            f"{volume['used_thresholded']:.2f} (thresholded part at {THRESHOLD:g})")


def load_path_text(load_path: dict, beta: float | None = None) -> str:
    """The sentence for a `load_path_fact`. `beta` is the beta of the stage: a grey design at a low beta is not the final part."""
    at = "" if beta is None else f" at beta {beta:g}"
    if "error" in load_path:
        return f"load path: the check was not made ({load_path['error']})"
    bad = load_path["not_joined"]
    if not load_path["regions"]:
        return "load path: the problem has no load region"
    if not bad:
        return (f"load path: every load region is joined to a fixed, spring or joint region in the thresholded design (density {THRESHOLD:g}){at}")
    names = ", ".join(f"'{n}'" for n in bad)
    what = f"load region {names} is" if len(bad) == 1 else f"load regions {names} are"
    return (f"load path: {what} not joined to a fixed, spring or joint region in the thresholded design (density {THRESHOLD:g}){at}. "
            f"A low beta gives a grey design: the thresholded part is then not the final part")


def material_text(material: dict) -> str:
    """The sentence for a `material_fact`: the volume and the mass of the thresholded part."""
    if "error" in material:
        return f"material: the check was not made ({material['error']})"
    return (f"material: the thresholded part (density {THRESHOLD:g}) has {material['volume_mm3']:.0f} mm3 and {material['mass_g']:.2f} g "
            f"(material density {material['density_g_cm3']:g} g/cm3)")


def _body_count(n: int) -> str:
    return f"{n} body" if n == 1 else f"{n} bodies"


def grey_text(grey: dict, beta: float | None = None, final: bool = False) -> str:
    """The sentence for a `grey_fact`: the design is grey, so the thresholded part depends on the cut. `beta` is the beta of the stage.

    It gives the grayness value (the number of the "Tests:" sentence of the verdict), the volume at the cuts 0.4, 0.5 and 0.6 with the spread as a share of the
    volume at 0.5, and the body count at the three cuts. `final` is true at the last pass of the run: the sentence then says where the body count of `final.stl`
    is, because the post step makes that number.
    """
    if "grayness" not in grey:
        return f"grey design: the check was not made ({grey.get('error', 'no result')})"
    at = "" if beta is None else f" at beta {beta:g}"
    head = f"grey design: grayness {grey['grayness']:.3g} is above grayness_max {grey['grayness_max']:.3g}{at}, so the thresholded part depends on the cut"
    if "levels" not in grey:
        return head + f" (the check of the cuts was not made: {grey.get('error', 'no result')})"
    low, high = STABILITY_LEVELS
    by = {lv["level"]: lv for lv in grey["levels"]}
    spread = grey.get("volume_spread")
    spread_text = "" if spread is None else f", a spread of {spread:.2f} of the volume at {THRESHOLD:g}"
    text = (f"{head}. Its volume is {by[low]['volume_mm3']:.0f} mm3 at the cut {low:g}, {by[THRESHOLD]['volume_mm3']:.0f} mm3 at {THRESHOLD:g} "
            f"and {by[high]['volume_mm3']:.0f} mm3 at {high:g}{spread_text}. "
            f"Its body count is {by[low]['bodies']} at {low:g}, {by[THRESHOLD]['bodies']} at {THRESHOLD:g} and {by[high]['bodies']} at {high:g}")
    if final:
        text += (". The mesh step keeps only the bodies that touch a frozen region (a fixed, load or solid region): "
                 "the body count of final.stl comes after the post step (metrics.grey_design.final_stl_bodies)")
    return text


GREY_OPTIONS = ("options: more passes with a beta schedule that reaches a higher beta (settings.iterations, settings.beta_schedule), "
                "or the compliance objective with settings.volume_fraction or settings.target_mass_g instead of settings.objective: mass")


def stage_end_lines(end: dict | None, it: int, where: str | None = None) -> list[str]:
    """The run log lines for a `stage_end` at pass `it` (none for None): one line for each fact.

    `where` names the place in the line (default: "last pass N", or "end of the beta B stage, pass N"). A run that a request stopped
    at a pass that ends no stage passes `stop at pass N` (P4b-4).
    """
    if not end:
        return []
    if where is None:
        where = (f"last pass {it}" if end.get("next_beta") is None else
                 f"end of the beta {end['beta']:g} stage, pass {it}")
    lines = []
    if end.get("volume"):
        lines.append(f"fact: {where}: {volume_text(end['volume'])}")
    if end.get("material"):
        lines.append(f"fact: {where}: {material_text(end['material'])}")
    if end.get("grey"):
        final = end.get("next_beta") is None
        lines.append(f"fact: {where}: {grey_text(end['grey'], end['beta'], final)}" + (f". {GREY_OPTIONS}" if final else ""))
    if end.get("load_path"):
        lines.append(f"fact: {where}: {load_path_text(end['load_path'], end['beta'])}")
    return lines


def verdict_facts(row: dict | None) -> dict | None:
    """The facts of the last pass for `verdict.json` (`facts`): the `stage_end` of the last history record, with its pass number. None when the record has none."""
    end = (row or {}).get("stage_end")
    if not end:
        return None
    return {"iteration": int(row["iteration"]), **end}


def verdict_sentence(facts: dict | None) -> str:
    """The "Facts:" part of the verdict text: the volume fact (a run with a volume row), the material fact and the grey design fact (a mass run, F1-3a),
    and the load path fact when a load region is not joined. Empty for None and for a run with none of them."""
    if not facts:
        return ""
    parts = []
    if facts.get("volume"):
        parts.append(volume_text(facts["volume"]))
    if facts.get("material"):
        parts.append(material_text(facts["material"]))
    if facts.get("grey"):
        parts.append(grey_text(facts["grey"], facts["beta"], facts.get("next_beta") is None))
    lp = facts.get("load_path")
    if lp and (lp.get("not_joined") or "error" in lp):
        parts.append(load_path_text(lp, facts["beta"]))
    return " Facts: " + "; ".join(parts) + "." if parts else ""


def grey_design_record(facts: dict | None, stability: dict | None, mesh: dict | None) -> dict | None:
    """The post step record `metrics.grey_design` of a grey mass run (F1-3a), or None for any other run. Never raises.

    `facts` is `metrics.verdict.facts` (the `stage_end` of the last pass), `stability` is `metrics.threshold_stability` and `mesh` is `metrics.mesh`.
    The record exists only when the facts of the last pass have the key `grey`: the trigger is the one of the verdict sentence. It holds the number that the
    verdict cannot have, because `verdict.json` is written before the post step: `final_stl_bodies`, the body count of `final.stl` (`mesh.stl_roundtrip.bodies`),
    next to the body count of the thresholded part at the cut 0.5. `mesh_bodies_removed` and `mesh_bodies_removed_mm3` (only with the `sdf` mesh method) tell how
    many bodies the mesh step took out because they touch no frozen region. `text` is the sentence for the run log. A fact only: no test, no limit.
    """
    try:
        grey = (facts or {}).get("grey")
        if not grey:
            return None
        out = {"grayness": grey["grayness"], "grayness_max": grey["grayness_max"]}
        bodies = {lv["level"]: lv["bodies"] for lv in (stability or {}).get("levels", [])}
        low, high = STABILITY_LEVELS
        stl = ((mesh or {}).get("stl_roundtrip") or {}).get("bodies")
        out["thresholded_bodies"] = bodies.get(THRESHOLD)
        out["final_stl_bodies"] = None if stl is None else int(stl)
        removed = (mesh or {}).get("extra_bodies_removed")
        if removed is not None:
            out["mesh_bodies_removed"] = int(removed)
            out["mesh_bodies_removed_mm3"] = float((mesh or {}).get("extra_bodies_volume_mm3") or 0.0)
        if out["thresholded_bodies"] is None or out["final_stl_bodies"] is None:
            out["text"] = "grey design: the body counts were not made (the cut check or the mesh check has no result)"
            return out
        text = (f"grey design: the thresholded part has {_body_count(out['thresholded_bodies'])} at the cut {THRESHOLD:g} "
                f"({bodies[low]} at {low:g}, {bodies[high]} at {high:g}), and final.stl has {_body_count(out['final_stl_bodies'])}. "
                "The first count is on the voxel grid and the second on the smoothed mesh")
        if removed is not None:
            text += (". The mesh step keeps only the bodies that touch a frozen region; it removed "
                     + (f"{_body_count(out['mesh_bodies_removed'])} ({out['mesh_bodies_removed_mm3']:.0f} mm3)" if removed else "no body"))
        out["text"] = text
        return out
    except Exception as exc:    # a fact never stops the post step
        return {"error": f"{type(exc).__name__}: {exc}"}
