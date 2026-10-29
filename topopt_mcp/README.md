# topopt MCP server

This server lets Claude start and watch a topopt run. It uses stdio.

Core tools: `environment`, `find_features`, `build_regions`, `record_brief`, `check_problem`, `render_part`, `run_topopt`, `run_status`, `run_result`, `print_check`.
The server instructions name these first. A tool is core when the start-here flow calls it and an eval or its mock calls it (CONTRACTS.md section 10). The section Tools names all 32 tools.

## Tools

Every tool refuses an argument name that it does not have: `{"ok": false, "error": "<tool> did nothing: it has no argument 'x'. The argument names are: ..."}`, before the tool runs (CONTRACTS.md section 10, "Strict argument names"). The input schema has `additionalProperties: false`. A key that a client adds to the schema of every tool for its own use (`wait_for_previous` of Gemini CLI) is dropped first.

- `run_topopt(yaml_path, out_dir=None, override_lock=False, proceed_without_brief=False, continue_from=None, more_iterations=None, fine_from=None, fine_pass=False, fine_voxel=None, fine_passes=None, fine_tol=None)`: start a run in the background. Default folder: `runs/live`.
  The reply has `brief_stated` and `brief_facts`. A YAML with a case file and no current brief gets `awaiting_user` (the facts, the
  options, the override `proceed_without_brief=true`) and nothing starts. A change of a method setting only after the brief (for example
  `settings.solver`) does not stop the run: `brief_facts` names the key with both values.
  `continue_from` (a run folder) with `more_iterations` (N, a whole number of 1 or more, no upper limit) adds passes to a run in a NEW folder (`out_dir`).
  The new run starts from the last checkpoint of the parent (pass K) and has K + N passes. The parent is only read. A parent that finished runs pass K
  again with its update step; a parent that did not finish (stopped, killed) gives a fork that starts at pass K + 1. `out_dir` as the parent, or inside it,
  is an error. A refusal (changed inputs, no checkpoint) is `{"error", "run_dir", "continue": {"facts", "options", "checkpoint"}}` and writes nothing.
  A run that starts has `continue`: the pass K, the pass count, whether pass K runs again, the step settings and the facts. See CONTRACTS.md section 24.3.
  `fine_from` (a run folder that finished) starts a fine pass in a NEW folder (`out_dir`): the same problem on a finer voxel, from the design of the parent. The parent is only read. The voxel is the one that the part needs (one value, tested on the grid of the fine pass and lowered in steps of 0.05 mm if a region has too few voxels across there; the key `fine_pass` of
  `check_problem` has the measured value), or `fine_voxel` (mm; a voxel equal to the parent's is refused, and the error names `continue_from`). The fine pass ends when its loss has settled (the change of the objective between passes is at most `fine_tol`, or `settings.fine_pass.objective_change` of the YAML, default 0.001),
  or after `fine_passes` passes (the maximum; default: the pass count of the parent); `stop_rules=false` turns every early stop off. `fine_pass=true` makes the run in `out_dir`, then the fine pass in `<out_dir>-fine`; the reply names both folders (`run_dir`, `fine_run_dir`) and `run_status` of the first folder
  has `fine_run` (`state` is the real end state after the process ended: `failed`, `refused` or `none` too). `check_problem.fine_pass.guards` and the start of `fine_pass=true` test the fine grid against the guards: a grid that fails one is refused before the first run, with `fine_voxel` options. `stop_run()` with no `run_dir` stops the fine pass when it runs, and the first run while that works; from the post step of the first run until the fine pass has started it writes the request in the fine folder, and the process ends without starting the fine pass. If the part does not need a finer voxel, the process ends after the first run and the log says so, unless `fine_voxel` is given and finer than the run: it wins, and the fine pass runs at it (a fact in both logs). A fine pass starts only with `fine_from` or `fine_pass`. `fine_voxel`, `fine_passes` and `fine_tol` go with one of them; the two do not go with
  each other, `resume`, `continue_from` or `more_iterations`. A run that starts with `fine_from` has `fine` (parent, iteration, voxel_mm, max_passes, tolerance, facts); a refusal is `{"error", "run_dir", "fine": {"facts", "options"}}` and writes nothing.
  `run_result` of a fine run has `fine_from`, and of any run `fine_pass` (the voxel that the part needs). See CONTRACTS.md section 24.6 and `docs/reference/fine-pass.md`.
- `run_status(run_dir=None)`: state, last iteration, and a one-line summary. State `stopped` (a request ended the run), `stop_request` (a
  request that waits in the folder), `checkpoint` and the resume hint are in the reply.
- `stop_run(run_dir=None)`: ask a running run to stop at the end of its current pass. It writes the file `job.stop` and returns at once: no
  signal, no wait. The run saves the checkpoint of that pass, writes the status `stopped` and ends (exit code 5, no post step, no result).
  `run_topopt` with `resume=true` continues it, and the resumed run equals a run that was not stopped (one thread, direct solver). The reply
  has `state` (`stop_requested`, `already_requested`, `postprocess`, `nothing_to_stop`, `gone`, `none`, `several`), `message`, `options`
  and `note`. A run that a build from before this tool started does not read the request. Without `run_dir` it takes the run that this
  server started, else the one live run in the work folder (a run that the CLI started). The CLI has `topopt stop <run_dir>`.
- `run_result(run_dir=None, preview=True)`: final metrics and the path of `final.stl`
  (`final_stl` absolute, `final_stl_rel` relative to the work folder). With machining stock
  (CONTRACTS.md section 23), `final_stl` is the as-printed mesh and `nominal_stl` and
  `nominal_stl_rel` name the finished mesh; without stock the keys are not there. With `regions.not_printed` in the YAML
  (section 9.1), `part_only_stl` and `part_only_stl_rel` name the mesh without those bodies and `metrics.part_only` has the
  facts (removed bodies, volume before and after, body count). With `preview`, it adds
  `preview`: mesh facts (triangles, bounding box, volume, watertight) and three ASCII silhouettes
  (top, front, side) for clients that cannot open files. There is no image renderer, so the
  silhouettes are coarse.
- `verify_run(run_dir=None, estimate_only=True, ..., part_only=False, gate=False)` and `verify_status(run_dir=None, result=True)`: the check of the finished part with quadratic
  tets (CONTRACTS.md section 16). `part_only=true` meshes `result/part_only.stl` (a run with `regions.not_printed`, section 9.1) instead of `nominal.stl` or `final.stl`. The removed bodies become boundary
  conditions at their contact faces: a removed fixed body is clamped, a removed load body takes its traction, a removed solid or keep-out body gives no condition (a fact in the result). The reply has the
  block `part_only` (file, sha256, removed bodies); the facts of each body are in the result of `verify_status`. Without the option, a run that has the file gets `notes`, one line that names the parameter.
  A run without the file gives `ok: false` with the fact and the options.
  `verify_run` needs the extra `verify` (it holds gmsh, scikit-fem and cholespy). When the final STL cannot be meshed at a size and the refusal
  depends on the size (a prepared surface that is refused, or a gmsh failure), the error names `mesh_size_mm` as the next step, or `levels_mm` when
  the call gave levels.
  A done `verify_status` also has `plain`: one plain line for each load case (the compared value, who gave the allowable, the margin), the verdict line `gate: PASSED` or `gate: FAILED (...)` when `gate=true`, and a line for what the check does not cover (CONTRACTS.md section 16).
  `gate=true` adds the key `gate` of the result: `met` for each check and case against the limits that the YAML has (`stress_limit` or `allowable_mpa`, `buckling.min_factor`, `limits` rows), and the exit code 3 if it fails. `allowable_mpa` replaces the stress limits of the YAML in the gate and the record shows the replaced limit (CONTRACTS.md section 16, "The gate").
- `ratchet_run(run_dir=None, source=None, marks=None, root=None, estimate_only=True, yaml_path=None, max_candidates=None,
  mesh_size_mm=None, stop_before_promote=False, overwrite=False, rebaseline=False, override_lock=False)`: judges candidate edits of
  a finished part (`source` is `marks:FILE` or `stl:FOLDER`) against the rules of the block `acceptance.rules` in the problem YAML,
  and keeps the ones that pass (CONTRACTS.md section 36). The YAML needs that block: the shipped examples do not have it, so add it
  to a copy of a YAML and give that copy in `yaml_path`. With `estimate_only` (the default) it starts nothing and gives the rules,
  what is missing and the estimate. A real start returns at once: call `ratchet_status`.
- `ratchet_status(root=None, last=5)`: the state of a ratchet (counts, the current parent, the last records with the reasons, the job).
- `ratchet_approve(root=None, id=None, decline=False)`: the opt-in owner gate. A ratchet that was started with `stop_before_promote`
  waits for this call before it promotes an accepted candidate. It promotes the candidate, or drops it with `decline`.
- `team_guide(role=None)`: the playbooks of the roles of a topopt team (`lead-design`, `cad`, `fea`, `print`, `finish`): what a role does, which
  tools it can use, what does not exist yet and what to report. Without a role it lists the roles. The same texts are served as MCP
  prompts and as the resources `topopt://roles/<id>` (CONTRACTS.md section 25).
- `list_runs()`: run folders in the work folder (and in its `runs/` folder), newest first.
  Each entry has `name`, `path` (relative), `run_dir`, `state`, `iteration`, `iterations`,
  `problem`, `process_alive`, `modified`, `mtime`, `final_stl_rel`. After a refused run in a folder that holds an earlier run, the entry has
  `earlier_run` (after `state`) with `iterations`, `problem` and `final_stl_rel` of that earlier run, and they are `null` at the top
  (CONTRACTS.md section 24.1). `run_history` and `compare_runs` mark such a folder in the same way.
- `run_history(root=None, depth=2, limit=20, detail=False)`: facts of the newest runs under a folder, read from the files on
  disk (settings, main metrics, the YAML of the run, and missing or partial files as `problems`); `count` is the number found.
  It writes nothing.
- `compare_runs(run_a, run_b=None, check_labels=None, variants=None)`: two runs side by side (`run_a`, `run_b`): the setup diff (`run.json`,
  the YAML and STL files on disk with hashes, the voxel labels of each run), the metrics diff, and facts on whether the files on disk can be
  the ones that ran. Or a table (`run_a` is the base, `variants` is a list of run folders): one row for each run (mass, stiffness for each
  load case, verdict, verify, print facts, the path of the newest montage PNG) and the setup diff of each variant against the base.
  Give `run_b` or `variants`, not both. It writes nothing and makes no image; `render_part` makes the images.
- `marks_to_hints(marks, run_dir=None, out_dir=None, options=None, overwrite=False, detail=False, yaml_path=None)`: reads the
  `marks.json` of the Refine page and writes a new folder (it refuses a run folder, the folder of an input, a symbolic link, and
  a folder with files that is not an earlier hints folder): a proposed YAML (the YAML of the run, with a keep-out STL for each remove mark and a forced-solid
  region for each add mark), the region STLs and `hints.json` (the facts and the options for each mark, and the option the YAML
  uses). The first mark wins an overlap. A smooth mark stays a fact. The reply is short; `hints.json` and `detail=true` have the
  full report. No mesh and no input file is edited. Start the rerun with `run_topopt`, then `compare_runs`. See CONTRACTS.md
  section 21.
- `apply_marks(marks, run_dir=None, out_dir=None, options=None, overwrite=False, allow_stale=False, detail=False,
  yaml_path=None)`: edits the finished part (`result/final.stl`) at the marks of the Refine page and writes a new folder:
  `edited.stl` and `apply_marks.json` (the full report). Each mark is one step on a signed distance field in a crop box:
  remove is a cut that protects the parent (then the parent check), add a closing with a ball, smooth an isotropic fairing
  (a point) or a web fairing (a stroke); `smooth_mode` tube rebuilds a whole strut and is asked for. The order is remove,
  smooth, add. The fixed, load and keep-out regions are not changed
  (an edit near them is clipped, and the clip is a fact). A step that gives a part that is not closed, or that splits it, is
  rolled back. The reply gives facts for each mark and the two routes as options (this edit, or `marks_to_hints` and a new
  run); it gives no verdict, and no FE run is made. Marks that were made for another `final.stl` are refused unless
  `allow_stale` is true. A crop box above the voxel guard is a fact with options (`pitch_mm`). The run folder and every input stay
  as they are. See CONTRACTS.md section 29.
- `cleanup_list(run_dir=None, out_dir=None, options=None, overwrite=False, estimate_only=False, detail=False, yaml_path=None,
  root=None)`: scans the finished part for loose bodies, dead-end spurs, thin bumps, uneven struts and flat webs, and writes a
  NEW folder with `marks.json` (a marks file for `apply_marks`, the Refine page and `ratchet_run`) and `cleanup.json` (the full
  report). Each mark has the facts of the candidate: the detector, the place, the limits used with their source, the measured
  sizes, and the stress near it when `verify/fields.npz` is there. `applicability` says how well the strut detectors fit the part.
  `options` set the limits (`pitch_mm`, `max_candidates`, `kinds`, `order` and the limits of the detectors). `estimate_only`
  gives the memory and time estimate and writes nothing. It edits nothing and gives no verdict: `ratchet_run` with
  `marks:` and the `marks_rel` path judges the list. The reply is short; `detail=true` or `cleanup.json` has the full report.
  See CONTRACTS.md section 29.3.
- `check_problem(yaml_path)`: loads and voxelizes a YAML (no optimizer run; with `auto` it runs the short self-test of gpu-mg in a child process, at most 60 s: seconds the first time on a device and driver, then the cached pass) and returns a dict:
  `grid`, `voxel_counts`, `load_regions`, `active_voxels`, `forced_solid_share`, `free_dofs`,
  `connectivity`, `solver` (`setting`, `resolved`, `platform`, `error`: the solver that runs on
  this platform, `resolved` is `null` if the platform cannot run the named solver), `backends`
  (`cholesky`, `pardiso`, `amg`: availability and time estimate per solver), `time_estimate` (for
  the solver that runs, with `amg_total_s`; `limit_s` is the time budget of the YAML, `settings.stop_rules.time_budget_s`, or `null`; it is a stop rule: the budget warning says if the run stops at it), `memory_estimate` (`bytes`, `ram_bytes`; a rough model
  fitted to Cholesky runs), `recommendation` (the offer to install pypardiso on Linux x86_64, else
  `null`), `hint`, `snapshot_disk_mb`, `assumptions`, `derived`, `lost_force`, `repeated_load_cases` (one entry for each group of load cases that have one load or the exact opposite: `cases`, `relations`,
  `weights`, the limits, `sign_rows`, `limit_rows`; `[]` if none; the text is also in `warnings`), `fine_pass` (the voxel that the part needs: `voxel_mm` (one value), `run_voxel_mm`, `recommend`, `limiting`, `estimate`, and `fact`, the one line, when `recommend` is true; section 24.6), `warnings`. `material` (only when the YAML has `material.filament`: the filament, where `E`, `nu` and `density` come from, and `compat`, the fit of the filament with the printer of `print.printer_model`; a fact, not a stop). `preferences_facts` (only when there is a preferences file and the printer type, the bed, another
  print value or the material of the YAML differ from it, or the YAML has no `print` block and the file has one: one text, a fact and no warning; the YAML
  is not changed) Warnings
  do not block a run. A wrong input gives `{"ok": false, "error": ...}`. The time model is fitted to
  Apple silicon Cholesky; PARDISO uses the same model, and AMG uses a rough factor (see CONTRACTS.md).
- `environment(yaml_path=None)`: facts about this machine, no change to it: `topopt_version`,
  `python`, `platform`, `packages`, `backends` (Accelerate `cholesky`, `pardiso`, `amg`: `available`,
  `note`, `resolved_by`), `solver_setting` (what each `settings.solver` value means here), `ram`,
  `cores`, `threads` and `recommendations`. With `yaml_path`, `problem.time_estimate_s` has the
  time estimate for each backend. `preferences` is the preferences file of the work folder as a fact (`path`, `exists`, `values` or `problems`,
  `kept_not_used`, `note`; see `save_preferences`). A recommendation is an offer as data (`reason`, `speedup_factor`,
  the command, `license_note`). The tool installs and builds nothing: ask the user, then run the command.
- `save_preferences(print=None, material=None, quality=None, outputs=None, filaments=None)`: keeps the print bed and printer, the material
  (also `material.filament`), the filaments that the user has (`filaments.available`), the quality (the
  voxel size) and the outputs of the user in `<work folder>/topopt-preferences.yaml`, so the next sessions need not ask again. It writes this
  file only and has no path argument. A section that is not given stays; a given section is merged key by key (`null` removes a key, `{}`
  removes the section); a key that the code does not know stays in the file. The checks are the ones of the problem YAML (`print`,
  `material`, `settings.voxel_size`; `print.printer_model` is checked with the printer catalog); an unknown argument name, an unknown key or a wrong value is refused with the key, the value and the rule (the input schema has `additionalProperties: false`). `environment` shows the
  file in its key `preferences` (`path`, `exists`, `values` or `problems`, `kept_not_used`, `note`). The server reads the file at each call. A
  preference does not change a run by itself: `run_topopt` and `check_problem` do not fill a YAML from it. `build_regions` takes a `material` (also `material.filament`),
  a `print` block or a `settings.voxel_size` that the spec does not have at all, and says so in `preferences_facts`; `check_problem` adds
  `preferences_facts` when the YAML differs from the file. There is no CLI command: a CLI user edits the file. See CONTRACTS.md section 10.2.
- `find_printer(name)`: finds a 3D printer by name in the catalog that ships with topopt (15 printers, `topopt/data/printers.yaml`) and in the
  user printers of the work folder (`topopt-printers.yaml`). It reads and changes nothing. The reply has the same keys for the three statuses:
  `ok`, `status` (`match`, `ambiguous` or `none`), `query`, `catalog_version`, `matched_by`, `origin` (`catalog` or `user`), `profile` (the
  printer and where each value comes from), `derives` (what `print.printer_model: <id>` sets in a run, and what the profile cannot give),
  `candidates` and `differs` (ambiguous), `suggestions` ("did you mean"; never taken by itself) and `vendors` (none), `user_file` (path, exists,
  entries, problems) and `note`. A value that the source does not give is `null` (unknown). In a YAML, `print.printer_model: <id>` sets
  `bed_mm` and `printer_type` when the YAML does not give them, and the print report compares the part height with the build height of the
  printer. `save_preferences(print={"printer_model": ...})` keeps the printer. See CONTRACTS.md section 18.7.
- `save_printer(profile)`: keeps a printer that is not in the catalog in `<work folder>/topopt-printers.yaml`: the agent looks up the maker page,
  shows the user the values with the page address, and calls this tool after the user agrees. The profile has the fields of a catalog entry; it
  needs `id`, `vendor`, `model`, `printer_type`, `bed_mm` and a source (`source_url` or `source: user`), and may give a source for each field. A value that is
  not given is `null`. An entry with the id of a user printer is replaced; an id or a name of a shipped printer is refused. The file is written
  atomically; it has no path argument, and the input schema has `additionalProperties: false`. Reply: `ok`, `path`, `profile`, `created`,
  `replaced`, `note`, `problems`. There is no CLI command: edit the file.
- `find_material(name)`: finds a 3D printing filament by name in the catalog that ships with topopt (`topopt/data/materials.yaml`) and in the
  user filaments of the work folder (`topopt-filaments.yaml`). It reads and changes nothing. The reply has the same keys for the three
  statuses as `find_printer` (`profile` is the filament and where each value comes from; `derives` says what `material.filament: <id>` sets in
  a run: `E` is the lower of the XY and the Z modulus, and `density`; `nu` stays the assumed 0.35, because no data sheet gives it; `makers`
  instead of `vendors`). A family alone ("PLA") is `ambiguous`. A value that no source states is `null` (unknown). See CONTRACTS.md section 18.8.
- `save_material(profile)`: keeps a filament that is not in the catalog in `<work folder>/topopt-filaments.yaml`: the agent looks up the data
  sheet of the maker, shows the user the values with the page address, and calls this tool after the user agrees. The profile needs `id`,
  `maker`, `product`, `family` and a source (`source_url` or `source: user`); the other fields are optional and `null` when not given. A run
  needs a modulus from the filament or from the YAML (the tensile modulus, or the bending modulus when the sheet gives no tensile one); a sheet that gives a density
  range gives the middle of it, and a density that the sheet does not give at all is the median density of its family, labelled `assumed`. The rules are the ones of `save_printer` (replace by id, no
  shadowing of a shipped id or name, atomic write, `additionalProperties: false`, no CLI command).
- `printer_materials(printer=None)`: the filaments of the catalog and of the user file for one printer (the argument, else `print.printer_model`
  of the preferences), grouped by state: `ok`, `care`, `no`, `unknown`. Each filament that is not `ok` has the reasons; a reason names both
  values that it compares. The filaments of `filaments.available` are marked and listed with their state. A state is a fact, not a stop. It
  reads and changes nothing.
- `build_regions(spec, name=None)`: build the region STLs and the problem YAML from a part spec
  given as data (a dict, or YAML text; same schema as the part-to-topopt skill). It writes
  `runs/<name>/` in the work folder (an existing folder with that name is overwritten and
  `overwrote_existing_folder` says so; region names must be safe file names; `live` works and adds a warning). Items in one region join (union). It returns `ok` (false when `builder_ok` is false), `name`,
  `run_dir_rel`, `yaml_rel`, `files_rel` (the files the new YAML uses), `other_files_rel` (other files in the folder, for example from an earlier build; nothing is deleted), `voxel_size`, `voxel_auto`, `retries`, `builder_ok`,
  `problems` (builder errors as facts; the files stay), `report` (the builder text), `check` (the
  `check_problem` data of the written YAML, `null` with `check_error` when the YAML cannot load)
  and `warnings`. When the spec has no `material`, no `print` block or no `settings.voxel_size`, they come from the preferences file (see
  `save_preferences`), are written into the YAML, and the reply has `preferences_facts` (one text for each block, with the file and the
  values); a block that the spec gives, also a part of one, is not changed. Pass `yaml_rel` to `run_topopt`. A wrong spec gives `{"ok": false, "error": ...}`
  (one line). The code is `topopt/regions_builder.py`; the skill script
  `.claude/skills/part-to-topopt/build_regions.py` is a thin wrapper with the same CLI.

- `record_brief(yaml_path, text, assumptions=None, criteria=None)`: record the brief of a problem YAML (what the user wants, and the
  assumptions with their origin: `user`, `inferred`, `default`). It makes the case: `<yaml stem>.case.json` and
  `<yaml stem>.case-log.jsonl` beside the YAML, with the SHA-256 of the YAML and of each region file. A YAML outside the work folder gets
  no case file. See CONTRACTS.md section 30.
- `case_status(yaml_path)`: the brief, the input hashes, the runs of the case (read from their run folders) and what is stale (the changed
  files are named). It changes nothing.

- `render_part(run_dir=None, stl_paths=None, roles=None, views=None, section=None, size=None, out_dir=None, regions=None, max_triangles=None, inline=True, yaml_path=None, loads=None)`:
  draws a run folder, STL files or a problem YAML (`yaml_path`, before a run) as PNG views and one labeled montage (no GL; CONTRACTS.md
  section 15). A YAML render draws one arrow for each force of each load case and gives them in `load_arrows`; `loads=false` draws none. The answer is
  the facts as JSON and, with `inline` true, the montage as an image. With `inline=false` it is the JSON only; the PNG
  files are in `paths`, and a client that can open files reads them.

- `print_check(path=None, yaml_path=None, print_settings=None, resolution_mm=None, strict=False)`:
  checks an STL file or a run folder for printing (CONTRACTS.md section 32; the same as `topopt printcheck`). Geometry only:
  no solve, no `metrics.json` or YAML needed. Each check (`mesh_closed`, `bodies`, `closed_voids`, `bed_fit`, `overhang`,
  `support_volume`, `plate_contact`, `wall_thickness`) has a value, a limit with its source, a state (`pass`, `fail`,
  `unknown`) and a reason. The limits are keys of the YAML `print` block; pass them in `print_settings`, for example
  `{"min_wall_mm": 1.2}`. A check with no limit is `unknown`. `strict=true` is an opt-in gate: `gate.state` is `fail` only
  when a check that has a limit fails, and `gate.exit_code` is then 4. No run stops on it. An input that cannot be read
  gives `{"ok": false, "error": ...}` (one line). With no `path` the last run is used.

- `export_step(run_dir=None, part_only=False, tolerance_mm=None, merge_planes=True, schema="AP214", out=None, estimate_only=False)`:
  writes a faceted STEP file of a finished run (CONTRACTS.md section 42; the same as `topopt step`; extra `step`: OpenCASCADE). It
  wraps `result/final.stl` (or `part_only.stl`) as a closed solid, after a manifold3d simplification (`tolerance_mm`, default the voxel
  size of the run / 20; 0 = none), and merges coplanar faces into planar faces. The reply is the report: the faces, the file size, the
  sampled distance between the STEP surface and the STL (max and mean, both directions), the volume, the time and memory, and the
  checks of the model and of the file read back (`read_back`, `topology`, `before_write`, `valid_solid`, `closed`, `units`, `schema`; none has a
  threshold; a failed check is a fact, `exit_code` 3, and the file is written). The tool starts `python -m topopt step` in a child process. `estimate_only`
  simplifies and counts the faces and writes nothing; it works without the extra. Without the extra the reply is `{"ok": false, "missing_extra": "step", "install_command": ...}`.
  Another failure is `{"ok": false, "error": ..., "exit_code": N}`, with `signal` when the child was killed and `stdout_first_line` when it wrote to stdout.
  No pass runs and no input file changes; the new files are `result/final.step` (`part_only.step`) and its report `result/final.step.json` (`part_only.step.json`);
  with `out` the report is next to that file and the run folder gets none. `out` replaces an existing `.step` or `.stp` file only. The call is synchronous: a
  large export holds it for minutes (a part of 355,000 faces took 613 s), and `estimate_only` gives the time before the work starts.

- `find_features(stl_path)`: lists the holes, bosses and flat faces of an STL file (CONTRACTS.md section 45). Read only: it reads the file in the
  server process and writes nothing. A relative path starts at the work folder. For each cylindrical hole it gives `point`, `axis`, `radius_mm`,
  `length_mm` (the keys of a `bore` interface), `end_point`, `depth` (`through`, `blind`, `closed` or `unknown`, from the faces across the two ends),
  the angle the wall covers, `segments`, `chord_sag_mm`, the fit residual and the holes on the same axis. Convex cylinders are in `bosses`. A flat face
  has an outward unit `normal`, `area_mm2`, `centroid` and `offset_mm`. `method.numbers` gives each number of the detection with the value used, and
  `facts` has plain sentences (an open mesh, a stepped hole, regions left out). About 4 s for a million triangles. A file that is missing or not an STL
  gives `{"ok": false, "error": ...}`.

- `show_problem(yaml_path)`: shows the problem as a 3D view in the chat of a client that shows MCP Apps (the UI extension of MCP):
  the design space, the fixed, keep-out, solid and load regions, an arrow for each load case, and the time and memory estimate of
  `check_problem` (the same checks, no optimizer run). The tool names the page `ui://topopt/problem-view.html` (`text/html;profile=mcp-app`)
  in `_meta.ui.resourceUri`. The meshes go in the `_meta` of the result (for the page only), the facts in `structuredContent`, and a short
  text in `content` (for the model, and for a client without MCP Apps). The page needs no network. Its button "Start the run" calls `run_topopt` through the host when
  the host offers `serverTools` (the click is the consent; when the run folder holds a run, the first click shows what the start replaces and a second click starts; the view then calls `run_status` every `RUN_STATUS_POLL_MS`, shows the pass and the volume, and
  at the end the state and the end numbers, and a button "Discuss the result"); else, if the host offers `message`, it sends a chat message from the user
  (`ui/message`); else the view shows the line to type. A small line in the view shows which of `message`, `serverTools` and `updateModelContext` the host offered. A wrong input gives `{"ok": false, "error": ...}`.

`run_topopt`, `run_status`, `run_result` and `stop_run` also return `*_rel` paths (relative to the work folder
when the path is inside it; absolute otherwise) next to the absolute ones.
`topopt check <yaml>` prints from the same data (`topopt.cli.check_data`).

The server refuses a second run while one is still running.

## Resources

- `topopt://roles/<id>`: the role playbooks (`lead-design`, `cad`, `fea`, `print`, `finish`). It is the same text as the tool `team_guide`.
- `topopt://materials/reference`: data sheet ranges of printed PETG and PLA, with the address of each sheet and what the numbers leave out.
  It is read only. It is a document: no tool returns a number from it, and there is no default allowable (CONTRACTS.md section 25.7).

## Enable it (once)

1. Install: `pip install -e ".[mcp]"` in your environment, and activate it
   (`source .venv/bin/activate`). This gives the command `topopt-mcp`.
2. Start Claude Code in the work folder (the repo folder, or any folder): `claude`.
   Claude Code finds `.mcp.json` there. Example:

   ```json
   {"mcpServers": {"topopt": {"type": "stdio", "command": "topopt-mcp", "args": []}}}
   ```

   If the command is not on the PATH, use the full path of the Python of that
   environment: `"command": "/path/to/python", "args": ["-m", "topopt_mcp.server"]`.
3. Claude Code asks you to approve the project server `topopt`. Say yes.
4. Type `/mcp`. Make sure `topopt` shows as connected.

## Port

The default port is 8765. To use another port, set the env var `TOPOPT_PORT` for the MCP server
(for example `TOPOPT_PORT=8800 claude`). If another process holds the port (an old dashboard, for example),
the run uses the next free port. `run_topopt` passes `--port` to the CLI and returns `dashboard_url` with the port in use.

The job process serves the dashboard after the run until you stop it. With the env var
`EVAL_TOPOPT_NO_WAIT=1`, `run_topopt` also passes `--no-wait`, and the job process ends when the run is done.
`claude plugin eval` uses it for the e2e eval (see `evals/README.md`).

## Dashboard

The page of `dashboard_url` shows the 3D view, the log (`run.log`), the bottom bar and, at the end, the result with the file of the
part (`<run folder>/result/final.stl`). The source of the optimizer with the active step, and the row "Pull-off stiffness per gram",
show only for a run that has a load case named `pull_off`, or when the address has `?demo=1` (CONTRACTS.md section 7).

## Folders

- Python of the child run: the Python that runs the server (`sys.executable`). `TOPOPT_PYTHON` names another one.
- Work folder: env var `TOPOPT_WORKSPACE`, else `TOPOPT_DEFAULT_WORKSPACE`, else the current directory. The Desktop extension sets
  `TOPOPT_DEFAULT_WORKSPACE` (`${HOME}/Documents/topopt`) and the server takes it as the sign that Claude Desktop started it. With it set,
  the notes name Claude Desktop and the fix in Settings > Extensions. Another client has no need to set it.
  A leading `~` and the variables `${HOME}`, `${DOCUMENTS}`, `${DESKTOP}` and `${DOWNLOADS}` in the value are replaced; a value the server cannot use is
  named in the key `work_folder_note` of the replies. `environment` shows `work_folder` and `server` (CONTRACTS.md section 10.1).
  Relative YAML paths and the default run folder `runs/live` start there.
  If a relative YAML is not in the work folder but is in the source checkout, the checkout copy is used.
  Where to change it for each client (the Desktop field, a full quit, the plugin, another client): [The work folder](../docs/reference/install.md#the-work-folder).

## Files

- Run output: `runs/live/`.
- Process output: `runs/live.stdout`. Process id: `runs/live.pid`.
- Registry of servers: each server writes `servers/server-<pid>.json` in the state folder of the user (Linux `$XDG_STATE_HOME/topopt`, else `~/.local/state/topopt`; macOS `~/Library/Application Support/topopt`; `TOPOPT_STATE_DIR` replaces it) and removes it at a clean exit, except while a run process that it started is still there (a run lives on after its server; the next reader deletes the file when that process has ended). A job counts for a server only if it holds the job lock of a run that this server started. `environment` lists the other servers in `other_servers` (a server that has ended is listed while its run process is there, with state `dead`), and `run_topopt` has `other_server_jobs` when one of them has a running job (CONTRACTS.md section 43).

## Test hook

Set `TOPOPT_MCP_CMD` to start a different command. The words `{yaml}`, `{out}` and `{port}` are replaced.
The command gets no run flags: `override_lock`, `resume`, `allow_changed_inputs`, `stop_rules`, `continue_from` and `more_iterations`
are not added to it. A call with `resume` or `continue_from` still starts the template as a plain command, and the reply of
`continue_from` still has the key `continue`. The tests use `tools/fake_run.py` this way.

## Test

```
python -m pytest tests/test_mcp.py
```

## Old run processes

The real CLI stays alive after a run when it hosts the dashboard.
It is then a small dashboard process with the same pid (about 25 MB, not the 590 MB of the optimizer).
A done run does not block a new run.
`run_topopt` stops the old process (SIGTERM, then SIGKILL after 3 s) and then starts the new run.
It refuses to start only while a run is starting, running or in post-process.
