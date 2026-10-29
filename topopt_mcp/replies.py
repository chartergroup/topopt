"""The reply key guides of the MCP tools (CONTRACTS.md section 10): the resource template `topopt://replies/{tool}`.

A tool text that lists the meaning of each reply key grows toward the 4096-character cut of the client. For seven tools, these lines
live here, and the tool text names the address of its guide. The guide of run_topopt holds the detail of the fine pass (CONTRACTS.md
section 24.6): its arguments and the keys that it adds to the reply of run_topopt, run_status, run_result and stop_run. It also holds the keys `brief_stated` and `brief_facts`. This module is the one place of the lines: the server serves them,
and the tests read them. It needs only the standard library.
"""
from __future__ import annotations

PREFIX = "topopt://replies/"
TEMPLATE = PREFIX + "{tool}"
NAME = "reply-keys"
TITLE = "Reply keys of a tool"
DESCRIPTION = ("Read-only. What each key of the reply means, for check_problem, build_regions, apply_marks, render_part, marks_to_hints, run_result and environment, "
               "and the arguments and keys of the fine pass of run_topopt, and its brief keys. Replace {tool} with the tool name. The text of the tool names its own address.")

# tool name -> the lines that explain the keys of its reply, in the words that the tool text had before they moved
REPLY_KEYS: dict[str, tuple[str, ...]] = {
    "check_problem": (
        ("Keys: grid, voxel_counts, load_regions, solid_regions (counts for `regions.solid`, or null), active_voxels, forced_solid_share, "
         "free_dofs, connectivity, solver (setting, resolved = what a run uses first, cpu_solver = the CPU solver, platform, error, choice = the facts of the pick), backends (availability and time estimate per solver), "
         "time_estimate (for the solver that runs; with gpu-mg it has model, low_s, high_s, basis (calibration, table, reference or none: "
         "a CPU device with no constants has null seconds) and gpu_mg with the assumptions: a band, not a limit), memory_estimate, "
         "lost_force (per load case: total force, force on nodes that are also fixed and dropped by the solver, lost percent; "
         "max_lost_percent), warnings, assumptions, derived (forces in N per load case, volume basis, masses in g, voxel size, minimum "
         "feature size, `filter` (radius, solid rule, min_member_mm, `min_feature_note`, the size and memory of the filter and the "
         "facts about it), `robust` (settings.robust, off, eroded or wall, and settings.eta_eroded: mode, thresholds, the facts with options), `time_model` "
         "(only for gpu-mg: the basis of the constants and the inputs of the time estimate), estimates, `estimate_bounds` (up to)), "
         "region_planes (the flat faces of the regions that are not on a voxel face: role, name and planes with the axis, side, position, "
         "offset in mm and voxels, larger or smaller than the STL, and `changes_part`; [] if none), recommendation (the offer to install "
         "pypardiso on Linux x86_64, else null), hint, snapshot_disk_mb, solid_pull (the pull of solid neighbours of the default filter "
         "rule), interface_taken (what each cut takes from each frozen region, one entry for each pair: cut, cut_kind, region, voxels, "
         "region_voxels, share, depth_mm, trim_mm, warn; a face, STL cut or keep-out region deeper than trim_mm, max(stock + voxel size, 2 "
         "mm), is also a warning), buckling_off (check and row both off, the default: memory_estimate_bytes of the check; `topopt check` "
         "prints it as one line)."),
        ("Key mirror: the facts of the YAML key `mirror` (planes with the parity and both label mismatch counts, orphans with their "
         "causes, the load asymmetry of each case), or null."),
        ("Key perimeter: the facts of `settings.perimeter_max_mm2` (max_mm2, eps, fixed_part_mm2, p_min_mm2: the smallest possible "
         "perimeter, bound_over_p_min; areas in mm2, voxel face area), or null. A bound below p_min_mm2 gives {\"ok\": false, \"error\": ...} "
         "with the options; the text shows the smallest possible perimeter rounded up to 0.1 mm2, a value that the check accepts as a "
         "bound."),
        ("Key repeated_load_cases: one entry for each group of load cases that have one load: the first case and every later case that has "
         "the same load (relation \"equal\") or the exact opposite (\"opposite\"; the same regions, every force, moment and pressure with the "
         "other sign): cases (the first is the first case of the group), relations (one for each later case), weights, compliance_limits, "
         "stress_limits_mpa, objective, objective_weight (the weight of the group as one case), remote_force, buckling_row, sign_rows (the "
         "rows of the YAML that follow the sign) and limit_rows (the limits rows that list a case of the group); [] if none. The text with "
         "the reason, what differs, the cost and the options is in warnings, as one line for each group that starts with \"fact: load "
         "case\"."),
        ("Key fine_pass: the voxel that the part needs, for a fine pass (CONTRACTS.md section 24.6): voxel_mm (one value: the largest voxel with "
         "2.8 voxels across the thinnest region, 3 across min_member_mm, rounded down to 0.05 mm), run_voxel_mm, recommend (true when voxel_mm is "
         "below the voxel of this run by 0.05 mm or more), limiting (role, name, thickness_mm, across_at_run, across_needed: the feature that sets "
         "the voxel; the thickness is twice the radius of the largest sphere inside the region), max_passes and tolerance (the defaults "
         "of a fine pass), and with recommend the estimate (voxel_mm, active_voxels, free_dofs, max_passes, time_s_max, time_s_run, "
         "memory_bytes: estimates) and fact (the one line that topopt check prints). A fine pass tests voxel_mm on its own grid and goes down in "
         "steps of 0.05 mm if a region has too few voxels across there (the log says it). run_topopt with fine_pass=true, or with fine_from "
         "(a finished run folder), runs the fine pass."),
    ),
    "run_topopt": (
        ("Fine pass (CONTRACTS.md section 24.6): a NEW run on a finer voxel, from the design of a finished run. The same YAML works. "
         "fine_from: the finished run folder (only read); the fine pass runs in out_dir. fine_pass=true: the run in out_dir, then the fine pass "
         "in `<out_dir>-fine` (`fine_run_dir`) if the part needs a finer voxel or fine_voxel is finer than the run; else the process ends after the first run and the log says so. "
         "fine_voxel (mm): the voxel; default: the voxel that the part needs, tested on the grid of the fine pass (it goes down in steps of "
         "0.05 mm if a region has too few voxels across there). With fine_pass=true a fine_voxel that is finer than the run wins: the fine pass runs "
         "at it also when the part needs no finer voxel (a fact in both logs); one that is not finer than the run is refused before the first run. A voxel equal to the parent's is "
         "refused (continue_from adds passes on the same grid), and so is a voxel of the part that is not finer than the parent's. "
         "fine_passes: the maximum number of passes (default: the pass count of the parent). fine_tol: the objective change at or below which "
         "the fine pass has settled and ends (default: `settings.fine_pass.objective_change` of the YAML, else 0.001). stop_rules=false: no "
         "early stop at all, the fine pass runs to the maximum. fine_voxel, fine_passes and fine_tol go with fine_from or fine_pass. "
         "fine_from and fine_pass do not go together, nor with resume, continue_from or more_iterations."),
        ("Reply keys of a fine pass: `fine` (parent, iteration, parent_voxel_mm, voxel_mm and voxel_from, max_passes and max_from, tolerance and "
         "tolerance_from (option, yaml or default), window, schedule_offset, source, facts, and with the voxel of the part limiting, "
         "limiting_role, limiting_thickness_mm and grid_check (start_mm, steps, region, ok, skipped)); a refusal is {\"error\", \"run_dir\", \"fine\": "
         "{\"facts\", \"options\"}} and writes nothing. With fine_pass=true also `fine_run_dir` and `fine_run_dir_rel` (the second folder). "
         "The grid of the fine pass is tested against settings.max_active_voxels, settings.max_grid_voxels and the memory estimate before the "
         "first run starts; a grid that fails one is refused, with the finest fine_voxel that passes in the options "
         "(check_problem `fine_pass.guards` shows the same)."),
        ("The fine pass in the other tools: run_status of the first folder has `fine_run` (run_dir, run_dir_rel, state: not started, starting, "
         "running, postprocess, done, stopped, and after the process has ended failed, refused or none, with message and log). run_result of "
         "a fine run has `fine_from` (the record of the parent, the voxel, the maximum, the tolerance and where each came from) and of any "
         "run `fine_pass` (the voxel that the part needs). stop_run() with no run_dir stops the fine pass when it runs; when the first run "
         "still works, it stops that run and the fine pass does not start. From the post step of the first run until the fine pass has started, "
         "it writes the request in the fine folder (state stop_requested, run_dir the fine folder): the process ends without starting the fine pass "
         "(fine_run.state stopped). continue_from on a fine run folder goes on with the fine voxel, "
         "the shifted beta schedule and the stop rule `settled` of that run."),
        ("The reply has `brief_stated` (true when a current brief is recorded) and `brief_facts`. A changed method setting does not stop: `brief_facts` names it."),
    ),
    "environment": (
        ("On a wrong YAML, `problem` is {\"ok\": false, \"error\": ...}."),
        ("ram and cores give the host values and the cgroup (container) limits."),
        ("`recommendations` are offers as data: when a faster backend is possible but not installed (pypardiso on Linux x86_64, the Accelerate shim on a Mac), each has a reason, the command and license_note (speedup_factor when it is measured). On a Mac without the Accelerate shim, auto runs amg: the offer says how much slower, and the build command is quoted."),
        ("`gpu_solver` is a fact, not an offer: `used` (settings.solver auto runs the GPU solver gpu-mg here), `device`, `case`, `self_test`, `why` and, when no GPU is used, `to_use_the_gpu` (what would make one usable)."),
        ("Keys: topopt_version, code_folder (the folder with topopt/, and examples/ and tools/ in a plugin or a checkout), checkout (git or not, head, branch, dirty, ahead and behind origin AS OF THE LAST FETCH with last_fetch_utc, update_command; nothing is fetched), python (version, executable), platform, packages (versions; null if not installed), tet, backends (per solver: available, note, resolved_by = the values of settings.solver that end in this solver), solver_setting (what auto, cholesky, pardiso, amg and gpu-mg mean here, or the error text), experimental_solvers (empty), opencl (pyopencl, devices, auto_case), auto_solver (what a run uses first), auto_choice, gpu_solver, ram (total_bytes, available_bytes), cores, threads (thread limits and the OpenBLAS cap of topopt run), problem, recommendations, run_interpreter, note."),
        ("`work_folder`: the folder in use and why; `server`: the pid and start time of this server; `other_servers`: other topopt servers (live, or ended with a run that goes on) and their runs."),
    ),
    "run_result": (
        ("`metrics.verdict` is the verdict of the run, with `facts` (see run_status). Grey mass run: `metrics.grey_design` (body counts)."),
        ("The result has `lost_force` (from run.json; force dropped on nodes shared by a load and a fixed region, with the lost percent) when the run folder has it."),
        ("`levers` names the lever cards of the facts that this run shows. It lists a fact only when the engine flags it (a verdict test that is not met, a state other than converged, a warning text, a value above a constant of the engine), with its cards: key, kind (requirement, method, both), layer and the words of the symptom (and `when`, where the symptom says which value of the fact it means), cheapest layer first. Card text: resource topopt://levers (all cards) or the template topopt://levers/{query} (one key or word)."),
        ("With machining stock (an `interfaces` entry with `stock_mm`), `final_stl` is the as-printed mesh and `nominal_stl` (absolute) and `nominal_stl_rel` name the finished mesh."),
        ("With `interfaces`, `interface_warnings` lists the warnings of the finished part (facts, then options): a frozen region that keeps less than 97 % of its volume, a `full_surface` interface that is not complete (below 0.98), a `stock_mm` with no effect. The list is there only if there is a warning. The run does not stop for them."),
        ("`mesh_fallbacks` lists the fallbacks that the mesh step took in any round of the volume calibration (`metrics.mesh.fallbacks`: the round, the step, the error type and text; one line for the same step and error in several rounds of a pass). The list is there only if there is one. The part can differ from a part with no fallback; the run does not stop for it."),
        ("With `regions.not_printed` in the YAML, `part_only_stl` (absolute) and `part_only_stl_rel` name the mesh without those bodies."),
        ("`report_md` (absolute) and `report_md_rel` name `result/report.md`, the one-page part card, written again by this call."),
        ("`print_check`: pass, fail, unknown counts, `failed`, `unknown_with_limit`."),
        ("`metrics.interfaces.integrity` has the numbers; `metrics.verdict.interface_tests` has the `full_surface` tests."),
        ("With `buckling: {check: true}` in the YAML, `metrics.buckling` has the lowest buckling load factors of each load case (a voxel estimate, not a verdict), or the fact and options why the check did not run."),
        ("`metrics.part_only` has: the removed bodies, the volume before and after, the body count, and `contact` (for each body: the contact area, `thinnest_wall_mm`, the thinnest wall behind its contact faces, and `area_below_voxel_mm2`, the contact area with a wall below the voxel size). A note names a wall thinner than the voxel size. The thinnest wall is the lower end of the film, not a value to use."),
        ("`summary` and `independent_check` of the case give the P2 p99.9 of verify_run and its ratio to the limit when verify has run on the folder (read again at each call), else the option to run it. Partial verify: `verify_partial`."),
        ("A run that ended with no result returns `done: false` with the `log_tail`, `failure`, `options` and `possible_causes` of run_status."),
    ),
    "build_regions": (
        ("Keys: ok (false when builder_ok is false), name, run_dir_rel, yaml_rel, files_rel, voxel_size, voxel_auto, retries, builder_ok, "
         "problems (builder errors as facts; the files stay), report (the builder text), check (the check_problem data of the written "
         "YAML), check_error, builder_warnings, other_files_rel, overwrote_existing_folder, warnings, assumptions (the optional spec list "
         "of short strings), derived (the numbers to state before the run) and check[\"lost_force\"] (force lost on nodes shared by a load "
         "and a fixed region; also in warnings)."),
    ),
    "apply_marks": (
        ("Reply keys: marks (id, type, status applied, partly_applied, no_effect, rolled_back, guard, skipped_by_option or unreadable; "
         "modes; volume and mass change; edit_reach_mm; clipped_by_frozen_mm3; thickness_near_mm before and after; parent check; bodies, "
         "closed and dropped_bodies; stress near the mark (a voxel estimate) from result/stress.npz; reason), totals (volume, mass, "
         "bodies, closed, reload from disk, frozen_exact, restored_mm3, triangles_replaced, surface_quality: closed manifold, fold edges "
         "and shortest edge of final.stl and of edited.stl, as topopt verify reads them), routes (this edit, or marks_to_hints for a new "
         "run), files, problems, out_dir_rel, edited_rel and report_rel."),
    ),
    "render_part": (
        ("The facts: PNG paths (absolute, and out_dir_rel and montage_rel relative to the work folder when inside it) with width, height "
         "and bytes, views, the section, for each mesh the role, bbox in mm, triangles, watertight and winding_consistent, the share of "
         "visible pixels that are back faces, missing files, notes, the font and the time."),
        ("The facts load_arrows give, for each load case and force, the force in N and its size, where it acts, the arrow tail and tip in "
         "mm and the color."),
    ),
    "marks_to_hints": (
        ("Reply keys: summary, marks (id, type, status applied, skipped_by_option, no_effect, facts_only, invalid or unreadable; "
         "option_used; voxels; region_file; reason), proposed_yaml (loads: the normal config path; ok is false when it does not load), "
         "labels_changed (voxels that differ from the original), volume_target (before and after), check (voxels, time estimate, "
         "warnings), grid, problems, plus out_dir_rel, yaml_rel and report_rel."),
    ),
}


def uri(tool: str) -> str:
    """The address of the key guide of `tool`."""
    return PREFIX + tool


def guide(tool: str) -> str:
    """The text of the resource for `tool`. A name with no guide gives a short text that names the tools that have one."""
    lines = REPLY_KEYS.get(tool.strip())
    if lines is None:
        return (f"No reply key guide for {tool!r}. The tools that have one: {', '.join(sorted(REPLY_KEYS))}. "
                f"The address is {PREFIX}<tool>, for example {uri('check_problem')}.")
    return f"Reply keys of {tool.strip()}. What each key of the reply means:\n\n" + "\n\n".join(lines) + "\n"
