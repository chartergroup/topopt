"""The levers of a run: which cards belong to the facts that a run folder shows (CONTRACTS.md section 41.10).

The cards (`topopt/levers_cards.py`) name, for each symptom, the facts that detect it (`Symptom.facts`, section 41.7). A run always holds
many of those facts (a grayness, a mass, a stiffness), so a list of all symptoms in every reply would be noise. This module lists a fact only
when the ENGINE already flags it in the run folder: a verdict test that is not met, a state other than the good one, a note or a warning
text, a boolean that is false (or true, where the engine sets it true for a fact that holds), a value above a constant that the engine already has. The module adds no threshold, tolerance or number of
its own: each row of `FLAGS` names the place in the engine where the flag comes from (`anchor`). A symptom whose fact has no such flag is
not listed; `NOT_FLAGGED` gives the reason for each (the "needs an engine fact" list).

`run_levers(run_dir)` returns the entries (JSON-safe), cheapest layer first. `compact` is the short form for the MCP tool `run_result`, and
`format_text` is the text of `topopt levers --run <run folder>`. The module reads files and writes nothing. It needs no solver.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .formulation import NOT_MET
from .levers import LAYER_SHORT, LAYER_TEXT, render_card
from .levers_cards import CARDS, FACT_SOURCES

# The cheapest layer first ("try the cheapest layer first"). The cost texts of the layers (`levers.LAYER_TEXT`) give this order: layer 3 takes
# seconds and no pass, layer 4 seconds to minutes, layer 2 a part of a run, layer 1 a whole run. Layer numbers are names, not a scale.
COST_ORDER = (3, 4, 2, 1)

RULES = ("false", "true", "text", "in", "not_in", "above")


@dataclass(frozen=True)
class Flag:
    """One fact path of the cards that the engine flags in a run folder.

    `path` is written as in `Symptom.facts`. `rule` says when the fact is flagged (no rule holds a number of its own):
      false  the value is the boolean false (a test that is not met, a check that did not apply)
      true   the value is the boolean true (a test that the engine sets to true when something holds: a relaxed peak above its limit)
      text   the value is a text, or a list or mapping with something in it (a note, a warning, a refusal, options)
      in      the value is one of `arg` (states that the engine names as an end that needs a look)
      not_in  the value is not one of `arg` (`arg` are the good states that the engine names)
      above   the value is a number above `arg` (a constant that the engine already has)
    `joined` are paths that the flag judges: their values are shown with the hit, and the cards that name them join the entry.
    `shown` are paths whose values are shown with the hit in the same way, but they judge nothing: no card is joined by them and the flag does not
    consume them (the P2 value of `topopt verify` next to a stress row that is met, F4-29).
    `says` is a short text. `anchor` names the place in the engine where the flag comes from.
    """
    path: str
    rule: str
    arg: object = None
    says: str = ""
    joined: tuple[str, ...] = ()
    anchor: str = ""
    shown: tuple[str, ...] = ()


# F4-29: "met" of a stress row is the relaxed measure of the optimizer. The stress of the part is the P2 p99.9 of `topopt verify`: `independent_check` of the case
# holds it (read again from verify/verify.json at each read, `stress_report.refresh_from_verify`). The hit shows `found` (false: verify has not run), the value and its ratio to the limit.
P2_SHOWN = ("metrics.stress.cases.<case>.independent_check.found", "metrics.stress.cases.<case>.independent_check.p2_p99_9_by_volume_MPa",
            "metrics.stress.cases.<case>.independent_check.p2_over_limit")
PART_STRESS = ("The stress of the part is in `independent_check` (the P2 p99.9 of `topopt verify`, from verify/verify.json, and its ratio to the limit); "
               "found false means verify has not run: run verify for the stress of the part")

_V = "metrics.verdict"
_M = "metrics.verdict.measures"
_IT = "metrics.interfaces.items[].as_printed.measured.<measure>"

FLAGS = (
    # --- the verdict (topopt/verdict.py): the engine's own tests, with the thresholds of `settings.stop_rules` -------------------------------------------
    Flag(f"{_V}.state", "not_in", ("converged", "stopped_by_rule"), "the run did not end by converging: the pass limit, a request or an unfinished history",
         anchor="verdict.STATES: 'converged' = the objective tests and every row met at the last pass; 'stopped_by_rule' = the tests of a stop rule ended the run"),
    Flag(f"{_M}.grayness.met", "false", None, "the grayness test of the verdict is not met", (f"{_M}.grayness.value",),
         anchor="verdict.measures grayness: met is false when the grayness is above settings.stop_rules.grayness_max"),
    Flag(f"{_M}.objective_change.met", "false", None, "the objective-change test of the verdict is not met",
         (f"{_M}.objective_change.value", f"{_M}.objective_change.threshold"),
         anchor="verdict.measures objective_change: met is false when the change is above settings.stop_rules.objective_change, or not judged"),
    Flag(f"{_M}.last_beta_stage.met", "false", None, "the beta schedule did not reach its last stage",
         anchor="verdict.measures last_beta_stage: met is false when the run ends before the last stage of the schedule"),
    Flag(f"{_V}.facts.load_path.not_joined", "text", None, "a load region is not joined to a support in the thresholded design",
         anchor="stage_facts.verdict_sentence: the verdict text names a load region that is not joined"),
    Flag("metrics.load_paths.regions.<region>.final_stl.connected_to_fixed", "false", None,
         "final.stl does not join a load region to a fixed region (no body of the mesh holds a point of the region and a point of a support)",
         ("metrics.load_paths.regions.<region>.final_stl.touches_a_body", "metrics.load_paths.regions.<region>.final_stl.bodies_with_the_region"),
         anchor="stress_report._stl_connectivity: connected_to_fixed is false when no body of final.stl holds a point of the load region and a point of a fixed region (I6: every run has metrics.load_paths)"),
    Flag(f"{_V}.notes", "text", None, "the verdict was made from the history with the default thresholds, not from the verdict.json of the run",
         anchor="verdict.verdict_for_post: the verdict has a note when the run folder has no matching verdict.json (the source is then 'history')"),
    # --- the constraint rows and the motion rows ------------------------------------------------------------------------------------------------------
    Flag("metrics.stress.cases.<case>.relaxed.met", "false", None, "the relaxed stress row of the optimizer is not met at the last pass (a measure of the optimizer, not of the part). " + PART_STRESS,
         ("metrics.stress.cases.<case>.relaxed.peak_MPa", "metrics.stress.cases.<case>.stress_limit_MPa"),
         anchor="stress_report.relaxed_numbers: met is g <= formulation.NOT_MET at the last pass; met_means says what it means", shown=P2_SHOWN),
    Flag("metrics.stress.cases.<case>.relaxed.peak_above_limit", "true", None,
         "the relaxed peak of a stress row is above its limit by more than the tolerance of the engine, whether the optimizer calls the row met or not (a measure of the optimizer, not of the part). " + PART_STRESS,
         ("metrics.stress.cases.<case>.relaxed.peak_MPa", "metrics.stress.cases.<case>.stress_limit_MPa"),
         anchor="ks_stress.peak_above_limit: peak > limit * (1 + PEAK_ABOVE_TOL), the test of the run log line 'stress peak above the limit'; stress_report.relaxed_numbers writes it",
         shown=("metrics.stress.cases.<case>.relaxed.met", "metrics.stress.cases.<case>.relaxed.c") + P2_SHOWN),
    Flag("metrics.constraints[].g", "above", NOT_MET, "a constraint row is above the 'not met' value of the engine", (f"{_V}.feasible",),
         anchor="formulation.NOT_MET: the run log calls a row not met above this g"),
    Flag(f"{_V}.feasible", "false", None, "a constraint row is not met at the last pass (listed once with the rows when a row is above NOT_MET)",
         anchor="verdict.measures max_g: met is false when the largest g is above FEASIBLE_TOL, or a row has no result"),
    Flag("metrics.rows[].met", "false", None, "a displacement or alignment row is not met",
         ("metrics.rows[].value_end", "metrics.rows[].limit", "metrics.rows[].g_end"),
         anchor="rows_motion.rows_block: met is g_end <= NOT_MET, the rule of the run log"),
    # --- the mesh and the print report -----------------------------------------------------------------------------------------------------------------
    Flag("metrics.mesh.stl_roundtrip.watertight", "false", None, "the STL file, read again, is not a closed manifold",
         anchor="post._mesh_record: the run log warns when the STL file, read again, is not a closed manifold (stl_roundtrip.ok false)"),
    Flag("metrics.mesh.fallback", "text", None, "the SDF mesh step did not run and the blur path made the mesh", ("metrics.mesh.requested",),
         anchor="post._mesh_method: the run log warns and the record has the fallback (the fact of the grid guard, or the error)"),
    Flag("metrics.mesh.extra_bodies_removed", "above", 0, "the SDF mesh step removed bodies that do not touch a frozen region", ("metrics.mesh.extra_bodies_volume_mm3",),
         anchor="smooth_sdf.make_mesh_sdf: the run log warns when it removed bodies (n_extra above 0); the volume target leaves them out"),
    Flag("metrics.print.configured.bed_fit", "false", None, "the part does not fit the bed in the build direction of the YAML",
         anchor="print_report.bed_fit_turn: bed_fit is false when the bed margin of the part is negative"),
    # --- the filter: the warnings about min_member_mm (the note `min_feature_note` is written in every run, so it is not a flag) -----------------------------
    Flag("metrics.filter.facts", "text", None, "the filter report has a warning about min_member_mm or the filter radius",
         anchor="members.member_facts: a warning for min_member_mm below 2 voxels, for both keys set, or for a derived radius not above 1 voxel; the log prints each as a warning"),
    # --- the buckling check: a status other than done, or a fact (the run log warns with each fact) ---------------------------------------------------
    Flag("metrics.buckling.status", "not_in", ("done",), "the buckling check did not run to the end",
         ("metrics.buckling.facts", "metrics.buckling.options"),
         anchor="buckling_voxel.check_problem: status done, skipped or error; the log prints each fact as a warning"),
    Flag("metrics.buckling.facts", "text", None, "the buckling check gives a fact", ("metrics.buckling.options",),
         anchor="buckling_voxel.run_check: the log prints each fact as a warning"),
    # --- the interfaces ------------------------------------------------------------------------------------------------------------------------------
    Flag("metrics.interfaces.error", "text", None, "the exact interface step stopped with an error", ("metrics.interfaces.applied",),
         anchor="interfaces.write_outputs: the record has the error text when the exact step did not apply (applied is then false)"),
    Flag("metrics.interfaces.applied", "false", None, "the exact interface step did not apply",
         anchor="interfaces.write_outputs: applied is false when the exact step did not apply (listed once with the error when there is one)"),
    Flag(f"{_IT}.within_tolerance", "false", None, "a measured feature of an interface is outside its tolerance",
         (f"{_IT}.deviation_mm", f"{_IT}.coverage", f"{_IT}.measured_mm", f"{_IT}.expected_mm"),
         anchor="interfaces.measure: within_tolerance is the test of a measured feature against its tolerance and its coverage"),
    Flag("metrics.verdict.interface_tests[].met", "false", None, "a full_surface interface is not complete: its full coverage is below the threshold of the engine",
         ("metrics.verdict.interface_tests[].full_coverage",),
         anchor="interfaces.full_surface_tests: met is full_coverage >= FULL_COVERAGE_MIN (check D of CONTRACTS.md section 23)"),
    Flag("metrics.interfaces.warnings", "text", None, "the finished part has an interface warning: a frozen region that keeps too little, a full_surface "
         "that is not complete, or a stock with no effect",
         anchor="interfaces: the warnings of checks C, D and E (run_result key interface_warnings)"),
    # --- the part-only file (result/part_only.stl, written when regions.not_printed names a file; topopt/part_only.py) ------------------------------------
    Flag("metrics.part_only.pinches_left", "above", 0, "the cut with settings.part_only_min_film_mm leaves places where the surface touches itself at a vertex",
         anchor="part_only.make: the record has pinches_left only with the setting; above 0 the record has a note that topopt.tet refuses such a surface"),
    Flag("metrics.part_only.watertight", "false", None, "part_only.stl, read again, is not one closed manifold",
         anchor="part_only.make: the run log warns when the file, read again, is not one closed manifold (stl_roundtrip.ok false)"),
    # --- the verify result of this run folder (verify/verify.json) ------------------------------------------------------------------------------------
    Flag("verify.bore_facts[].status", "not_in", ("ok",), "a bore of the bore check has no sample set (empty) or stopped with an error: it is not checked",
         ("verify.bore_facts[].funnel", "verify.bore_facts[].facts"),
         anchor="verify.bore.build_bore: status is ok when the bore has a sample set, else empty; verify.bore._error_entry: error"),
    Flag("verify.gate.passed", "false", None, "the gate of verify (--gate) failed: a check is not met, or a check that the gate asked for could not run",
         ("verify.gate.failed",),
         anchor="verify.gate.evaluate: passed is true when at least one check is gated and every gated check has met true; failed lists the others"),
    # --- run.json and the history ---------------------------------------------------------------------------------------------------------------------
    Flag("run.lost_force.max_lost_percent", "above", 0, "force is lost on nodes that a load and a fixed region share",
         anchor="loads.lost_force_warning: the check warns when a case loses force (lost_magnitude_n above 0)"),
    Flag("run.derived.load_cases[].load_reaches_free_dofs", "false", None,
         "no free dof gets a load from a load case: its load vector is exactly 0.0 on every free dof (the nodal forces add up to 0.0 at every node, or every force lands on fixed dofs; also false for a case whose YAML numbers are all 0)",
         ("run.derived.load_cases[].applies_load",),
         anchor="loads.reaches_free_dofs (problem.build_problem, run.json derived.load_cases): false for a case that the solver sees with no load; loads.no_load_lines writes the fact at the start of run.log"),
    Flag("run.derived.load_cases[].applies_load", "false", None, "a load case applies no load: every force and moment is exactly 0 and no pressure is other than 0",
         anchor="config.LoadCase.applies_load (run.json derived.load_cases): false for a case with no load; loads.no_load_lines writes the fact at the start of run.log (the flag of an older run.json, which has no load_reaches_free_dofs)"),
    Flag("history.step_acceptance.accepted", "false", None, "all tries of the step acceptance were rejected, and the last step was taken",
         ("history.step_acceptance.tries",),
         anchor="optimize.StepAcceptance.step: accepted is false when STEP_TRIES tries failed; the run log prints a fact"),
    # --- the state of the run itself -------------------------------------------------------------------------------------------------------------------
    Flag("run_status", "in", ("stopped", "error"), "the run ended with a request to stop, or with an error",
         anchor="status.json state: stopped (a request ended the run) or error"),
    # --- the ratchet of this run (the log beside the run folder) -----------------------------------------------------------------------------------------
    Flag("ratchet.decision", "in", ("REJECT",), "the last candidate of the ratchet was rejected",
         anchor="ratchet.Outcome.decision: ACCEPT, REJECT or BASELINE"),
    Flag("ratchet.checks[].state", "in", ("fail",), "a rule of the ratchet failed for the last candidate",
         anchor="ratchet.Check.state: pass, fail, unknown, off or no_input"),
)

# A fact of the cards that is in the run folder or a tool but that the engine does not flag. These are the "needs an engine fact" points: the reason
# says what is missing. The paths are written as in `Symptom.facts`. `decisions()` checks that every fact path of the cards is flagged, joined, or here.
NOT_FLAGGED = {
    "the value is a plain measurement; the engine sets no limit for it and its text says no pass or fail": (
        "metrics.mass_final_g", "metrics.mass_saved_pct", "metrics.stiffness_cases[].N_per_mm", "metrics.print.configured.overhang_area_mm2",
        "metrics.print.configured.support_volume_mm3", "metrics.print.options[].overhang_area_mm2", "metrics.print.thinnest_member.area_below_mm2",
        "metrics.print.thinnest_member.thinnest_member_mm", "metrics.interfaces.stock_volume_mm3",
        "metrics.reactions.cases[].joints.<region>.bond_energy_share", "metrics.reactions.cases[].residual_rel",
        "run.joints.joints[].bond_share_max", "run.mirror.end.rho_deviation", "metrics.mesh.volume_error_pct", "run.derived.volume.target_mass_g",
        "metrics.part_only.shortest_edge_mm",
        "verify.supports.cases.<case>.joints.<region>.bond_energy_share", "verify.supports.joints[].bond_share_max",
        "verify.supports.cases.<case>.joints.<region>.bond_force_N", "verify.supports.cases.<case>.joints.<region>.bond_moment_Nmm",
        "verify.supports.cases.<case>.springs.<region>.force_N", "verify.cases.<case>.equilibrium.support_force_N",
        "verify.cases.<case>.equilibrium.support_moment_Nmm", "verify.mesh.bodies.bodies[].support_facets",
        "verify.supports.joints[].area_over_voxel_area", "verify.supports.springs[].area_over_voxel_area"),
    "the asked value of a requirement; the engine sets no test on it": (
        "run.derived.volume.target_mass_asked_g", "metrics.mass_asked_g"),
    "the fact is written in every run (a note or a text); the flag of the same block is another key": (
        "metrics.filter.min_feature_note", "metrics.stress.cases.<case>.summary"),
    "the engine warns about a radius not above 1 voxel only for a radius derived from min_member_mm (a line of metrics.filter.facts); for settings.filter_radius alone it writes the number and no flag": (
        "metrics.filter.size.radius_voxels",),
    "the threshold stability is a fact only (stage_facts.threshold_stability): the engine sets no limit for how far the volume or the body count may move between the levels 0.4, 0.5 and 0.6": (
        "metrics.threshold_stability.levels[].volume_mm3", "metrics.threshold_stability.levels[].bodies"),
    "the stage facts say 'none of them is a test, a limit or a warning rule' (stage_facts.py)": (
        "metrics.verdict.facts.volume.used_projected", "metrics.verdict.facts.volume.used_thresholded", "history.stage_end.volume",
        "history.stage_end.volume.used_projected"),
    "the engine states the measure with no flag ('last_step_change' has threshold none), or the symptom needs a fact that the last pass does not give": (
        "history.change", "metrics.verdict.converged"),
    "the record of a fine pass (CONTRACTS.md section 24.6): where the run started, its voxel, maximum and tolerance; the engine sets no test on it (how the fine pass ended is the verdict state, which is flagged)": (
        "run.fine_from.run_dir", "run.fine_from.voxel_mm", "run.fine_from.source", "run.fine_from.max_passes", "run.fine_from.tolerance", "run.fine_from.tolerance_from"),
    "the recommendation of a fine pass (CONTRACTS.md section 24.6) is a hint for a NEXT run on a finer voxel, not a test that this run failed; run_result shows it in its own key `fine_pass`, and the run log, the check and the part card write the fact line": (
        "run.part_voxel.recommend",),
    "a list that is written in every run that has the block, or in every resumed run; no flag in it": (
        "metrics.verdict.resume_points", "history.constraints[].ks_offset"),
    "the engine reports the number with no pass or fail (the Goodman text says 1 is not a limit of topopt; no allowable is given)": (
        "verify.goodman.lowest_n", "verify.goodman.rows[].n_goodman", "verify.goodman.rows[].peak_over_yield",
        "verify.cases.<case>.von_mises.p99_9_by_volume_MPa", "verify.cases.<case>.von_mises.peak_qp_MPa"),
    "a record of the bore check that is written for every bore (the flag is its status, which joins the entry)": (
        "verify.bore_facts[].near_miss", "verify.bore_facts[].sample_set"),
    "a record of the gate (value, margin, reason, the replaced limit): the flag of the gate is passed, which joins the list failed; the engine sets no second test on a record": (
        "verify.gate.checks.<kind>.<name>.met", "verify.gate.checks.<kind>.<name>.margin", "verify.gate.not_gated[].reason",
        "verify.gate.checks.stress.<case>.loosened", "verify.gate.checks.stress.<case>.replaced_limit"),
    "the buckling number has no engine threshold (a load factor below 1 is judged by buckling.min_factor, a row of the constraints)": (
        "metrics.buckling.lowest_load_factor", "metrics.buckling.cases[].load_factors"),
    "the buckling record of verify: the lowest load factor has no engine threshold (the limit is buckling.min_factor, read by the gate), and the mode facts and the rigid-body record are facts about the shape of the mode, not a limit (CONTRACTS.md section 33)": (
        "verify.buckling.lowest_load_factor", "verify.buckling.cases[].mode_facts", "verify.buckling.cases[].rigid_body"),
    "a measurement of the interface step: the flag is within_tolerance, which the entry shows with its measured value": (
        "metrics.interfaces.items[].as_printed.measured",),
    "the mesh step records the number and the engine gives no flag (its warning is for the extra bodies, which are flagged)": (
        "metrics.mesh.islands_removed", "metrics.mesh.local_loss", "metrics.mesh.local_loss.deep_lost_mm3", "metrics.mesh.method"),
    "the perimeter of the run: the row is in the constraints, and p_final is written with the limit but no flag": (
        "run.perimeter.max_mm2", "run.perimeter.p_final_mm2"),
    "the cards name these for a run that HAS the file (applied true: the check can run); applied false is an error record with an error text that no card names (the run keeps final.stl)": (
        "metrics.part_only.applied", "metrics.part_only.file"),
    "the engine writes the contact numbers of a removed body in every cut; its note compares them with the voxel size only when the film setting is off, and no flag is in the record": (
        "metrics.part_only.contact.<label>.area_below_voxel_mm2", "metrics.part_only.contact.<label>.thinnest_wall_mm"),
    "a fact of the part-only check (topopt verify --part-only): the name of the file it meshed; no pass or fail": (
        "verify.part_only.file",),
    "the card names the whole mapping `yaml_vs_run`, which is in every check; its flag is the key `matches` or the list `differences` (verify.part_only.yaml_vs_run: 'a difference is a fact, not an error'), and no card names that key": (
        "verify.part_only.yaml_vs_run",),
    "the print check is a reply of the tool; no file of the run folder holds it": (
        "print_check.checks[].state", "print_check.checks[].value.area_below_limit_pct", "print_check.checks[].value.area_mm2",
        "print_check.checks[].value.count", "print_check.summary.failed"),
    "the ratchet record: a number or a text of a candidate; the flags of the ratchet are decision and the state of a check": (
        "ratchet.checks[].delta_pct", "ratchet.facts.cases.<case>.p_by_volume_MPa", "ratchet.facts.cases.<case>.stiffness_N_per_mm",
        "ratchet.facts.loose_mm3", "ratchet.facts.mass_g", "ratchet.facts.cases.<case>.buckling_lowest_factor"),
}

LAYERS_LINE = ("cheapest layer first: " + ", ".join(f"{n} ({LAYER_SHORT[n]})" for n in COST_ORDER))
POINTER = ("Card text: the MCP resource topopt://levers (all cards), the resource template topopt://levers/{query} (the cards of one key or word), or the command "
           "`topopt levers <key or word>`. Each card has a kind: requirement (the user's), method (yours) or both; the rule is in the resource. "
           "The tools give facts and options; you decide.")

LABEL_KEYS = ("name", "rule")      # the keys of a list entry that name it (a case, a row, a feature, a rule of the ratchet)
TOOL_FACTS = {"run_status": ("status", ("state",))}      # a tool fact that the run folder holds: (document, key path)
DOCUMENTS = ("metrics", "run", "history", "status", "verify", "ratchet")


# --- the paths ------------------------------------------------------------------------------------------------------------------------------------

def _split(path: str) -> tuple[str, tuple[str, ...]]:
    """The document and the key path of a fact path (`metrics.rows[].met` -> ("metrics", ("rows[]", "met")); a tool fact -> its document)."""
    if path in TOOL_FACTS:
        return TOOL_FACTS[path]
    source, _, rest = path.partition(".")
    return source, tuple(rest.split("."))


def _wild(seg: str) -> bool:
    return seg.endswith("[]") or (seg.startswith("<") and seg.endswith(">"))


def _match(node, segs, names=(), nodes=()):
    """Every match of the key path `segs` in `node`: `(names, nodes)`, with the concrete name and the node after each segment.

    `[]` takes every entry of a list and `<name>` any key of a mapping.
    """
    if not segs:
        yield names, nodes
        return
    seg, rest = segs[0], segs[1:]
    if not isinstance(node, dict):
        return
    if seg.startswith("<") and seg.endswith(">"):
        for k, v in node.items():
            yield from _match(v, rest, names + (str(k),), nodes + (v,))
        return
    key = seg[:-2] if seg.endswith("[]") else seg
    if key not in node:
        return
    child = node[key]
    if seg.endswith("[]"):
        if isinstance(child, list):
            for i, e in enumerate(child):
                yield from _match(e, rest, names + (f"{key}[{i}]",), nodes + (e,))
        return
    yield from _match(child, rest, names + (key,), nodes + (child,))


def flagged(rule: str, arg, value) -> bool:
    """Whether `value` is flagged by `rule` (the rules of `Flag`). A missing value (None) is never flagged."""
    if value is None:
        return False
    if rule == "false":
        return value is False
    if rule == "true":
        return value is True
    if rule == "text":
        return isinstance(value, (str, list, dict)) and len(value) > 0
    if rule == "not_in":
        return isinstance(value, str) and value not in arg
    if rule == "above":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > arg
    if rule == "in":
        return isinstance(value, str) and value in arg
    raise ValueError(f"unknown rule {rule!r} (rules: {', '.join(RULES)})")


def _labels(segs, names, nodes) -> str:
    """The names of the entries that a path went through (the key of a mapping, the `name` or `rule` of a list entry), joined by ' / '."""
    out = []
    for seg, name, node in zip(segs, names, nodes):
        if not _wild(seg):
            continue
        if seg.endswith("[]"):
            label = next((node[k] for k in LABEL_KEYS if isinstance(node, dict) and isinstance(node.get(k), str)), None)
            out.append(label if label is not None else name[name.rindex("["):])
        else:
            out.append(name)
    return " / ".join(out)


def _joined_values(flag: Flag, segs, base_names, nodes, doc) -> dict:
    """The values of the joined paths for one hit: keys are the paths relative to the shared prefix with the flagged path."""
    out = {}
    for j in flag.joined + flag.shown:
        _, jsegs = _split(j)
        k = 0
        while k < min(len(segs) - 1, len(jsegs) - 1) and segs[k] == jsegs[k]:
            k += 1
        base = nodes[k - 1] if k else doc
        for names, got in _match(base, jsegs[k:]):
            out[".".join(names)] = got[-1]
    return out


def _hits(flag: Flag, docs: dict) -> list[dict]:
    """The hits of one flag: `{at, value, of?, with?}` for each place where the flag holds. A history fact is summarised (see `_history_hit`)."""
    source, segs = _split(flag.path)
    doc = docs.get(source)
    if doc is None:
        return []
    if source == "history":
        hit = _history_hit(flag, segs, doc)
        return [hit] if hit else []
    out = []
    for names, nodes in _match(doc, segs):
        if not flagged(flag.rule, flag.arg, nodes[-1]):
            continue
        hit = {"at": flag.path if flag.path in TOOL_FACTS else source + "." + ".".join(names), "value": nodes[-1]}
        label = _labels(segs, names, nodes)
        if label:
            hit["of"] = label
        joined = _joined_values(flag, segs, names, nodes, doc)
        if joined:
            hit["with"] = joined
        out.append(hit)
    return out


def _history_hit(flag: Flag, segs, rows: list) -> dict | None:
    """One hit for a fact of the history: the passes where the flag holds (`passes`, `first_pass`, `last_pass`), and the value and the joined values of the last one."""
    found = []
    for row in rows:
        for names, nodes in _match(row, segs):
            if flagged(flag.rule, flag.arg, nodes[-1]):
                found.append((row, names, nodes))
                break
    if not found:
        return None
    row, names, nodes = found[-1]
    hit = {"at": flag.path, "value": nodes[-1], "passes": len(found), "first_pass": found[0][0].get("iteration"), "last_pass": row.get("iteration")}
    joined = _joined_values(flag, segs, names, nodes, row)
    if joined:
        hit["with"] = joined
    return hit


# --- the files of the run folder ---------------------------------------------------------------------------------------------------------------------

NOT_THERE, BROKEN = "is not there", "does not parse"     # why a document is missing: no such file, or a file that is not a JSON object / not JSON lines


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    """The JSON object in `path`, and why there is none: NOT_THERE (no such file) or BROKEN (not readable as text, not JSON, not an object)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, NOT_THERE
    except (OSError, ValueError, RecursionError):
        return None, BROKEN
    return (data, None) if isinstance(data, dict) else (None, BROKEN)


def _read_lines(path: Path) -> tuple[list | None, int, str | None]:
    """The rows of a JSON-lines file (`history.jsonl`, the ratchet log), the number of lines that were skipped because they do not parse, and why there are no rows (NOT_THERE or BROKEN)."""
    from .runinfo import read_history
    try:
        rows = read_history(path, strict=False).rows
        lines = sum(1 for line in path.read_bytes().decode("utf-8", errors="replace").splitlines() if line.strip())
    except FileNotFoundError:
        return None, 0, NOT_THERE
    except (OSError, RecursionError):
        return None, 0, BROKEN
    return rows, max(0, lines - len(rows)), None


def ratchet_root(run_dir: Path) -> Path:
    """The default root of `ratchet_run`: beside the run folder, `<run folder>-ratchet`."""
    return run_dir.parent / (run_dir.name + "-ratchet")


def _verify_mesh_note(d: Path, verify: dict) -> str | None:
    """A note when verify.json is for another mesh file than the one in result/ now (the check of `verify_status`: `result_matches_final_stl`)."""
    run = verify.get("run") if isinstance(verify.get("run"), dict) else {}
    name, old = run.get("verified_file"), run.get("final_stl_sha256")
    name = name if isinstance(name, str) and name else "final.stl"          # a file name is text; anything else is the default of verify
    try:
        now = hashlib.sha256((d / "result" / name).read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None
    if old is None or now == old:
        return None
    return (f"verify/verify.json is for an older result/{name} (the sha256 differs): its facts are listed, and they describe the older mesh. "
            "`verify_run` makes a result for the mesh that the folder has now.")


def load_documents(run_dir) -> tuple[dict, list[str]]:
    """The documents of a run folder that the flags read, and the notes about what is missing.

    `metrics` is `result/metrics.json`; a run that stopped has none, and then it is the verdict alone (`verdict.json` holds the same document as `metrics.verdict`).
    `run` is run.json, `history` the rows of history.jsonl, `status` is status.json, `verify` is verify/verify.json (when the folder has it), `ratchet` the last record of the ratchet log beside the folder.
    A document that is not there is not in the dict. A note says so for `result/metrics.json`, `run.json` and `history.jsonl`; `status.json`, `verify/verify.json` and the ratchet log
    are optional, and a missing one gives no note. A file that is there and does not parse is a note for each document ("does not parse"); a line of `history.jsonl` or of the ratchet log
    that does not parse is skipped, and one note for the file gives the number of lines (a fact, not an error).
    """
    d = Path(os.path.abspath(run_dir))
    docs, notes = {}, []
    status, why = _read_json(d / "status.json")
    if status is not None:
        docs["status"] = status
    elif why == BROKEN:
        notes.append("status.json does not parse: `run_status` is not read.")
    metrics, why = _read_json(d / "result" / "metrics.json")
    if metrics is not None:
        from .stress_report import refresh_from_verify
        refresh_from_verify(metrics, d)       # F4-29: the P2 value next to "met" is the one of verify/verify.json now (in memory; no file is written)
        docs["metrics"] = metrics
    else:
        from .verdict import read_verdict
        verdict = read_verdict(d)
        state = (status or {}).get("state")
        notes.append(f"result/metrics.json {why}: the run has no {'result' if why == NOT_THERE else 'readable result'}" + (f" (status.json state: {state})" if state else "")
                     + ". The facts of the verdict, run.json and history.jsonl are read; the other facts of metrics.json are missing.")
        if verdict is not None:
            docs["metrics"] = {"verdict": verdict}
    run, why = _read_json(d / "run.json")
    if run is not None:
        docs["run"] = run
    else:
        notes.append(f"run.json {why}: the facts of run.json are missing.")
    rows, skipped, why = _read_lines(d / "history.jsonl")
    if rows is not None:
        docs["history"] = rows
        if skipped:
            notes.append(f"history.jsonl: {skipped} line(s) do not parse and were skipped; the facts of the other lines are read.")
    else:
        notes.append(f"history.jsonl {why}: the facts of the history are missing.")
    verify, why = _read_json(d / "verify" / "verify.json")
    if verify is not None:
        docs["verify"] = verify
        note = _verify_mesh_note(d, verify)
        if note:
            notes.append(note)
    elif why == BROKEN:
        notes.append("verify/verify.json does not parse: the facts of the verify result are missing.")
    ratchet_path = ratchet_root(d) / "ratchet.jsonl"
    rows, skipped, why = _read_lines(ratchet_path)
    if rows is not None:
        last = next((r for r in reversed(rows) if r.get("decision")), None)
        if last is not None:
            docs["ratchet"] = last
        if skipped:
            notes.append(f"{ratchet_path.parent.name}/ratchet.jsonl: {skipped} line(s) do not parse and were skipped; the facts of the other lines are read.")
    elif why == BROKEN:
        notes.append(f"{ratchet_path.parent.name}/ratchet.jsonl does not parse: the facts of the ratchet are missing.")
    return docs, notes


# --- the cards of a flag ------------------------------------------------------------------------------------------------------------------------------

def _layer_cost_call(layer: int) -> tuple[str, str]:
    """The cost and the call of a layer, from the layer text of the reference (`levers.LAYER_TEXT`, which has 'Call: ...')."""
    head, _, call = LAYER_TEXT[layer].partition(" Call: ")
    return head, call


def _cards_of(paths: set) -> list[dict]:
    """The cards that name one of `paths` in a symptom, with the words (and the `when` text, where the symptom has one) of those symptoms: cheapest layer first, the order of the table inside a layer."""
    found = []
    for index, card in enumerate(CARDS):
        named = [s for s in card.symptoms if paths & set(s.facts)]
        if named:
            cost, call = _layer_cost_call(card.layer)
            when = list(dict.fromkeys(s.when for s in named if s.when))
            found.append((COST_ORDER.index(card.layer), index, {
                "key": card.key, "group": card.group, "kind": card.kind, "layer": card.layer, "words": "; ".join(s.words for s in named),
                **({"when": " | ".join(when)} if when else {}), "cost": card.cost or cost, "call": card.call or call}))
    return [c for _, _, c in sorted(found, key=lambda t: t[:2])]


def run_levers(run_dir) -> dict:
    """The flagged facts of a run folder, each with its cards: cheapest layer first.

    The result: `run_dir`, `read` (the documents that were read), `notes` (what is missing), `order` and `entries`. An entry has `fact` (the path as the
    cards write it), `flag` (what the engine says), `hits` (where it holds: `at`, `value`, `of`, `with`) and `cards` (`key`, `group`, `kind`, `layer`,
    `words`, `when` (where a symptom of the card has one), `cost`, `call`). A run folder with no file gives no entry and the notes.
    """
    docs, notes = load_documents(run_dir)
    entries, consumed = [], set()
    for index, flag in enumerate(FLAGS):
        if flag.path in consumed:
            continue
        hits = _hits(flag, docs)
        if not hits:
            continue
        consumed.update(flag.joined)
        cards = _cards_of({flag.path, *flag.joined})
        entries.append((COST_ORDER.index(cards[0]["layer"]) if cards else len(COST_ORDER), index,
                        {"fact": flag.path, "flag": flag.says, "hits": hits, "cards": cards}))
    entries.sort(key=lambda t: t[:2])
    return {"run_dir": str(Path(os.path.abspath(run_dir))), "read": [k for k in DOCUMENTS if k in docs], "notes": notes,
            "order": LAYERS_LINE, "entries": [e for _, _, e in entries]}


# --- the forms ------------------------------------------------------------------------------------------------------------------------------------------

def compact(result: dict) -> dict:
    """The short form for `run_result`: the entries with the card keys, kinds and layers, no card text. `cards` points to the cards."""
    entries = []
    for e in result["entries"]:
        hits = [{k: v for k, v in h.items() if not (k == "at" and v == e["fact"])} for h in e["hits"]]
        cards = [{"key": c["key"], **({"tool": True} if c["group"] == "tool" else {}), "kind": c["kind"], "layer": c["layer"], "words": c["words"],
                  **({"when": c["when"]} if c.get("when") else {})} for c in e["cards"]]
        entries.append({"fact": e["fact"], "flag": e["flag"], "hits": hits, "cards": cards})
    out = {"entries": entries, "order": result["order"], "cards": POINTER}
    if result["notes"]:
        out["notes"] = result["notes"]
    return out


def _value_text(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def format_text(result: dict) -> str:
    """The text of `topopt levers --run <run folder>`: the entries, then the full card of each distinct card (the text of `topopt levers <key>`)."""
    lines = [f"# Levers for the facts of {result['run_dir']}", "",
             f"Read: {', '.join(result['read']) or 'nothing'}. A fact is listed when the engine flags it in this run (CONTRACTS.md section 41.10); "
             f"{LAYERS_LINE}."]
    lines += ["", *[f"Note: {n}" for n in result["notes"]]] if result["notes"] else []
    if not result["entries"]:
        lines += ["", "No fact that the cards name is flagged in this run."]
    seen: dict[tuple, dict] = {}
    for n, e in enumerate(result["entries"], 1):
        lines += ["", f"{n}. {e['fact']}: {e['flag']}"]
        for h in e["hits"]:
            extra = "".join(f"; {k} = {_value_text(v)}" for k, v in h.get("with", {}).items())
            where = (f" ({h['of']})" if h.get("of") else "") + (f" in {h['passes']} pass(es), from {h['first_pass']} to {h['last_pass']}" if "passes" in h else "")
            lines.append(f"   at {h['at']}{where}: {_value_text(h['value'])}{extra}")
        for c in e["cards"]:
            kind = c["kind"] + (", tool" if c["group"] == "tool" else "")
            lines.append(f"   - {c['key']} [{kind}; layer {c['layer']}, {LAYER_SHORT[c['layer']]}]: {c['words']}")
            if c.get("when"):
                lines.append(f"     Present when: {c['when']}")
            seen.setdefault((c["group"], c["key"]), c)
    if seen:
        by_key = {(c.group, c.key): c for c in CARDS}
        lines += ["", "## Cards", "", "The cards above, cheapest layer first, as `topopt levers <key>` prints them.", ""]
        order = sorted(seen.values(), key=lambda c: COST_ORDER.index(c["layer"]))
        lines.append("\n\n".join(render_card(by_key[(c["group"], c["key"])]) for c in order))
    return "\n".join(lines) + "\n"


# --- the decision for every fact path of the cards ---------------------------------------------------------------------------------------------------

def decisions() -> dict[str, str]:
    """For every fact path of the cards: `flagged`, `joined`, or the reason why the run folder cannot flag it. The levers test asserts that no path is left out.

    A path of a source other than a run document (`check_problem`, a bare tool name) is not read from the run folder: a tool gives it when it is called.
    """
    flagged_paths, joined = {f.path for f in FLAGS}, {j for f in FLAGS for j in f.joined}
    reasons = {p: why for why, paths in NOT_FLAGGED.items() for p in paths}
    out = {}
    for card in CARDS:
        for s in card.symptoms:
            for p in s.facts:
                if p in flagged_paths:
                    out[p] = "flagged"
                elif p in joined:
                    out[p] = "joined"
                elif p in reasons:
                    out[p] = reasons[p]
                elif p.partition(".")[0] == "check_problem":
                    out[p] = "the fact is the reply of check_problem (before the run); a run folder keeps it in other keys"
                elif p.partition(".")[0] not in FACT_SOURCES:
                    out[p] = "a tool gives the fact when it is called; nothing in the run folder holds it"
    return out
