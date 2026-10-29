# Run a problem: the YAML format, the levers and the result

[Back to the README](../../README.md)

## Quick start

Check the example. This command starts no optimizer run (with `auto` on a machine with a GPU it runs a short self-test of gpu-mg in a child process, at most 60 s: seconds the first time on a device and driver, then the cached pass):

```
topopt check examples/hook/hook.yaml
```

The command prints the grid size, the number of free dofs, and a time estimate.
Then start the run with the dashboard:

```
topopt run examples/hook/hook.yaml --out runs/hook --dashboard
```

The `--dashboard` option opens the browser by itself at `http://127.0.0.1:8765/`.
The hook run takes about 2.5 minutes on an M1 Mac (148 s). On Linux with `pardiso` the best of 62 runs on a shared host took 99 s and the median was 462 s (see [Linux x86_64](install.md#linux-x86_64)).
After the run, the process stays alive to keep the final screen up.
It becomes a small dashboard process of about 25 MB, with the same pid.
It prints the pid and how to stop it. Press Ctrl+C to stop it, or run `kill <pid>`.
Add `--no-wait` to exit when the run is done.
The result files are in `runs/hook/result/`: `final.stl`, `metrics.json`, and `preview.bin`.
A run writes `job.json` and `job.DONE` (or `job.FAILED` with the exit code, or `job.STOPPED` when `topopt stop` or the MCP tool `stop_run` ended it after a pass: exit code 5) in the run folder: wall time, peak memory (the peak of the job itself on Linux) and the result. `job.json` also keeps the facts of the machine for the analysis of run time: threads, CPU model, RAM, load, other topopt processes, versions, CPU time and the time of the set-up, the loop and the post step; `run.json` has a `problem_key` that is the same for runs of the same problem (CONTRACTS.md sections 5 and 24.1).
One run holds the lock of a run folder (`job.lock`). A second `topopt run` on that folder stops at once and says which process
holds it. Add `--override-lock` to run anyway. A lock of a process that has stopped is replaced.

To add passes to a run, start a new run from its checkpoint: `topopt run hook.yaml --out runs/hook-more --continue-from runs/hook --more-iterations 20`.
The new folder gets a copy of the history and the snapshots of the parent, and the run goes on to pass K + 20 (K is the pass of the last checkpoint of the parent).
The parent folder does not change. `--out` must be another folder, outside the parent. With one thread and a direct solver, the result equals a run of K + 20 passes
(CONTRACTS.md, section 24.3). A run that was stopped or killed works too: it forks from its last checkpoint. `--resume` is the way to finish a run in its own folder.

## The final mesh

After the last iteration, the post step makes `result/final.stl` from the density. The default method (`settings.mesh_method: sdf`)
smooths a signed distance field and keeps the volume of the density that you got. Members of 2 voxels or more stay (measured on a test part with rods of 2, 3 and 5
voxels: all stay with `sdf`; with `blur` the 2-voxel and 3-voxel rods are gone). The fixed, load and keep-out regions keep their exact STL
shape. The smooth surface does not lie on the faces of these regions, and the exact union or cut of two almost coplanar surfaces leaves a strip of tiny edges. So before
the exact step the post step moves the vertices that lie just outside a region face (within 0.35 fine voxel) a little way into the region, where the exact step removes them;
a move that would turn a face over is made shorter. Measured on one host (c080f62 against the same code without the move, the hook recording, the ladle and the bracket of
`examples`): the edges under 0.05 mm go from 355 to 96 of about 90,000 on the hook, from 1,623 to 185 on the ladle and from 204 to 169 on the bracket; the volume changes by
at most 0.003 %; the new surface is 0.006 to 0.035 mm from the old one on average (the largest distance is 0.19 to 0.89 mm, mostly at the line where the surface meets a region face).
`mesh.snap` in `metrics.json` has the counts. On the hook the step took about 3.5 s before the move. The move adds one pass of 0.1 to 2 s to each build of the calibration
(hook 1.4 to 2.0 s, ladle 0.7 s, bracket 0.1 s; one host with a load of 18) and one more build for `mesh.snap.volume_change_mm3`. The mass changes by a few percent against `blur`, with either sign (hook -4.9 % at 1.2 mm voxels and -8.8 % at 2 mm voxels, ladle +2.1 %),
because the regions have their STL shape and not the voxel staircase. The volume error in `metrics.json` is the residual of a
global volume calibration; `mesh.local_loss` gives the volume that the smoothing took away in one place.
If a step of the mesh build fails in any round and takes a fallback (the mesh stays unmoved or uncut), the part is still written. `mesh.fallbacks` lists each fallback with its round, step and error, `run.log` has a warning line, and the part card and `run_result` (`mesh_fallbacks`) have a note. The part can then differ from a part with no fallback.
A face of the design space that is slanted to the grid axes can show steps (panels) in the result. With `sdf`, the post step can fill them (`settings.envelope_fill: auto` or `on`; the default is `off`, the mesh of the versions before the fill): it grows the result
to the design space surface before the smoothing, and cuts it to the envelope. The mass target does not change. On a coarse, sparse part the growth can be a large share of the mass
(more than 5 % of the target); `auto` then skips the fill. `on` runs it anyway, `off` (default) does not run it (`--envelope-fill auto|on|off` wins over the key).
`mesh.envelope_fill` in `metrics.json` says if it ran and the share of the growth.
The fill costs some stiffness at the same mass, more on a coarse grid: on `examples/ladle/ladle.yaml` the stiffness fell by 1.7 to 3.6 % at the example settings (3.25 mm)
and by 7 to 21 % at 6.5 mm. Set `envelope_fill: auto` or `on` for a part with slanted faces whose look counts more than a few percent of stiffness.
The old method (Gaussian blur) is still there: set `mesh_method: blur` in the YAML file or add `--mesh-method blur`
to `topopt run`. `result/metrics.json` has the key `mesh` with the method, the volume before and after smoothing,
and how well the regions are held. `python tools/check_final_stl.py runs/hook` measures this on the STL file.
If the grid for `sdf` is too big for the memory, the run uses `blur` and says so in a warning and in `metrics.json`.
Details are in [CONTRACTS.md](../../CONTRACTS.md), section 19.

## Bodies that are not printed (`part_only.stl`)

A plate that the part is bolted to, or a block that takes the load, is a fixed, load or solid region: the optimizer needs it, and
`final.stl` holds it. List such files in `regions.not_printed` and the post step also writes `result/part_only.stl`: `final.stl` with those
bodies cut out by an exact boolean. The contact face of the part is then the exact surface of the removed body.

```yaml
regions:
  not_printed: [wall_plate.stl, hook_tip.stl]
settings:
  part_only_min_film_mm: 0.1     # optional, a number above 0, in mm. Default: off
```

- The smoothing of the mesh leaves a thin skin around a frozen body. A skin that is joined to the part by a face stays as a thin wall
  behind the contact face (a film). In a stress check of `part_only.stl` such a wall carries load as a membrane, and a mesher gets slivers from it.
- `result/metrics.json` (key `part_only`, key `contact`) always gives, for each removed body, the contact area, the thinnest wall at its
  contact faces (`thinnest_wall_mm`) and the contact area with a wall below the voxel size (`area_below_voxel_mm2`). If a wall is thinner than
  the voxel size, a note gives these facts and the option.
- The thinnest wall is the lower end of the film, not a value to use: on the hook it is 0.000033 mm, and a value just above it removes almost
  nothing. The value is the wall thickness that you want to cut off. That is a decision about the print or the look; the tool recommends no number.
- `settings.part_only_min_film_mm` cuts off the film: every contact triangle whose wall is thinner than the value loses the material behind
  it, with a prism as deep as the value. On the centre ray of the triangle the prism stops in the middle of the air gap behind the wall; off the
  centre it cuts any material under the triangle, up to its depth. A contact face with a thicker wall stays the exact face of the removed body.
  The facts then also give, for each body, the thin area (`thin_area_mm2`), the area that the cut removes (`cut_area_mm2`; it can be larger, because
  the cut also takes the triangles at a vertex where the thin cut alone would leave the surface touching itself), the thin area that stays
  (`thin_area_after_mm2`) and the thinnest wall that stays. `film_removed_mm3` is all the material that the cut removes: the film and, at the edge of
  a film, other material. The run log has one line with both areas. The wall is measured at the centre of each contact triangle, so a
  wedge of film inside a large triangle can stay. A bad value (0, negative, not a number) is an error in the YAML. The key without `regions.not_printed`
  does nothing, and `topopt check` says so.
- To apply a value to a run that has finished: write the YAML with the new value as a NEW file in the folder of the old YAML (the region STL paths
  are relative to that folder), then resume the run with it: `topopt run <new yaml> --out <run folder> --resume
  --allow-changed-inputs`. The optimization does not run again; the post step does. On the hook example this takes about 8 s on
  the build host. Copy the run folder first if you want to keep the old result. Then read `thin_area_mm2`, `cut_area_mm2` and
  `thin_area_after_mm2`, and change the value if the film that you want gone is still there.
- Details: [CONTRACTS.md](../../CONTRACTS.md), sections 2 and 9.1.

## The YAML format

A YAML file names the STL files for these regions:

- `design`: the space the part can fill.
- `fixed`: the regions that stay clamped. An entry `{file, axes}` fixes only the named axes (a roller).
- `keep_out`: the regions that stay empty.
- `load`: the named regions that carry a force, a moment or a pressure.
- `joints` (optional): a beam from a load region to a fixed point, for a part that hangs on a rod or a pin that is not in the grid.
  An entry is `{region, to: [x, y, z], section: {kind: round, d: 4}}` (mm). The beam can run in any direction. Then `fixed` may be empty.
  Details: [CONTRACTS.md](../../CONTRACTS.md), section 38.

The file also lists the load cases, the material, and the settings
(voxel size, volume fraction, iteration count).
Paths are relative to the YAML file. All STL files use mm and must be closed.
The full format is in [CONTRACTS.md](../../CONTRACTS.md), section 2.

The allowed material is one of two keys in `settings`: `volume_fraction` (the mean density of the free design voxels) or `target_mass_g` (the mass that the user asks for, in grams).
Write exactly one. With `target_mass_g` the engine derives the fraction from the voxel grid and the density, refuses a mass that the forced-solid voxels already exceed or that all active
voxels cannot reach (the message gives the limit in grams), and `topopt check` shows the asked mass and the derived fraction. The finished mass is a result, not the asked number: compare
`metrics.mass_final_g` with it (`metrics.mass_asked_g`, and the run log line "final mesh ... g against ... g asked", give both). Details: [CONTRACTS.md](../../CONTRACTS.md), section 3, "The mass in grams".
A working example is in [examples/hook/](../../examples/hook).

`interfaces` (optional): exact shapes that the post step unites with the part or cuts out of it: a `bore`, a `boss`, a `tube`, a `face` and an STL
solid. Details: [CONTRACTS.md](../../CONTRACTS.md), section 23, and [examples/interfaces/](../../examples/interfaces).

- A `face` cuts everything on the side of its `normal`, inside the grid box. It also cuts a fixed or load region that reaches past the plane,
  for example a sleeve. Give the face a `radius_mm` to cut only the disc of that radius about the line through `point` along `normal`:
  `{name: end_face, kind: face, point: [60, 0, 0], normal: [1, 0, 0], radius_mm: 35}`. A YAML without `radius_mm` works as before.
- `topopt check` (and the MCP tool `check_problem`, key `interface_taken`) counts what each cut takes from each fixed, load or solid region and
  each union shape: a line `interface cut:  <name> (<kind>) takes N of M voxels (P %) of ...`. A `face`, an `stl` cut or a keep-out region that takes
  a region deeper than a machining trim, `max(stock_mm + voxel_size, 2 mm)`, is also a `warning:` with the options. A bore is a fact only: it exists to remove material.
- After the run, `result/metrics.json` key `interfaces.integrity` gives, for each fixed, load or solid region and each union shape, the share of its
  volume that the finished mesh (`nominal.stl`, or `final.stl` with no stock) keeps. The holes of the `bore` and `tube` interfaces are taken out of the
  region first. A region that keeps less than 97 % is a warning in `run.log`, in `run_result` (`interface_warnings`) and in the verdict text. The run does not stop.
- `full_surface: true` on a `bore`, `boss`, `tube`, or a `face` with `radius_mm` says: this surface must be complete (a bond socket, a bearing seat, a press
  fit). The measure then gives `full_area_mm2` and `full_coverage`, and the verdict has one test for the interface: 0.98 or more is met. Any other
  `face`, and an `stl` solid, with `full_surface: true` is a loader error. `coverage` counts only the wall that has material behind it; it does not count
  a wall that is not there. Use `full_surface` for that.
- When `stock_mm` is above 0 and the as-printed surface is under 5 % of the finished area, the stock has no effect, and the run gives a warning.
- `topopt render problem.yaml` draws the regions and the interface shapes before a run. A union shape is solid green, a cut shape is a red ghost, and a
  face is its cut zone in the grid box. Look at it to see a plane that goes through a sleeve. It also draws one arrow for each force of each load
  case, along the force (`--no-loads` draws none): show it to the user to check the force directions before the run.
- When the buckling check is off (the default), `topopt check` prints one line `buckling: off (check and row) ...` with the memory estimate of the check.
  The check stays off. Details: CONTRACTS.md section 33.

Load cases can give the same work to the optimizer. The model is linear. A case with the exact opposite loads of another case (the same
load regions, each force, moment and pressure with the other sign) gives the same compliance and the same von Mises stress. A case with equal loads gives the
same value for every row. `topopt check`, the run log and the MCP tool `check_problem` (key `repeated_load_cases`) say so as a fact, one for each group
of such cases. The fact gives the effect of the weights and of the limits of the cases, and the options: keep one of the cases, or keep all of them when a row
follows the sign (a buckling row, or a displacement limit with the sense `max` or `min`). The run goes on. The test is exact (no tolerance), and cases that are
proportional with another factor are not found. Details: [CONTRACTS.md](../../CONTRACTS.md), section 13.

A load case in which every force and moment is exactly 0 and no pressure is other than 0 applies no load. It adds nothing to the objective, and its weight and its stress limit do nothing.
`topopt check`, the run log and the MCP tool `check_problem` (key `derived.load_cases[].applies_load`) say so as a fact, with the options: give the case a force, a moment or a pressure, or
remove the case. A moment alone is a load, and so are two equal and opposite forces on two regions. The loader refuses the YAML, with options, before the run in these cases: no case
applies a load and the objective is compliance (a run ended with exit 0 and a part that was only the fixed regions); a case with no load has a `compliance_limit`, or is next to
`case_combine: max` with the compliance objective, or next to `buckling.min_factor` (each of these ended with exit 2 and an internal error); a case with no load is next to
`buckling.check` (the run ended with exit 0 and an error record in place of the factors). Details: [CONTRACTS.md](../../CONTRACTS.md), section 13.

A run with fewer `settings.iterations` than the first pass of the last stage of `settings.beta_schedule` (the default schedule: pass 65) never reaches the last beta, and its
design can stay grey. `topopt check`, the run log and the MCP tool `check_problem` say so as a fact before the run. The fact gives the options (more iterations, or the
schedule scaled to the passes, as YAML you can paste) and the price that was measured on the hook example. The run goes on. Details: [CONTRACTS.md](../../CONTRACTS.md), section 13.

## Levers: what each key and tool does

Each YAML key and each engine tool has a lever card. It says what the lever does, what one try costs (a new run, a continued run, the post step only, or a local edit of the
finished part), the words a user says about a result ("too heavy", "extra film here"), and the fact in the result that detects them. A card also says who decides a change.
A method setting (penal, beta schedule, move limit, solver and tolerances, voxel size, iterations, robust, mesh method, smoothing) is the agent's choice: it changes it without asking,
and reports what, why and the price. A user requirement (loads, supports, keep-outs, interfaces, material, stress limits and safety factors, a stated mass or volume target, print limits,
symmetry) can be tightened by the agent without asking, with a report and its measured price. The agent loosens one only with the user's consent, and brings the facts. A key that can be
either follows its origin in the brief record (`record_brief`). This is guidance: no tool refuses a change.

```
topopt levers                   # all cards
topopt levers settings.penal    # the card of one key
topopt levers film              # the cards that match a word of the user
topopt levers --run <run dir>   # the facts of a run that the engine flags, each with the cards that name them (also the block "levers" of run_result)
```

The MCP resource `topopt://levers` has the same text, and the resource template `topopt://levers/{query}` gives the cards of one key or word (like `topopt levers <query>`). An unknown key in the YAML gives the known keys of its block and the name of this command. The table is
`topopt/levers_cards.py`. A new key needs a card, or `tests/test_levers.py` fails. Details: [CONTRACTS.md](../../CONTRACTS.md), section 41.

## Stress limits and the mass objective

A load case can have a stress limit. The optimizer tries to keep the stress of that case at or below it. The row is a target,
not a guarantee: the run log and `result/metrics.json` say whether it was met (`stress.cases.<case>.relaxed.met` and
`stress.stage_ends`). The tool has no default limit (the filament catalog gives `E` and `density`, not an allowable): you (or the agent that writes the YAML) give the stress.
Data sheet ranges of printed PETG and PLA (with their sources and what they leave out) are in the MCP resource
`topopt://materials/reference`, the file `topopt_mcp/materials/reference.md`. It is a document to read, not a table that the tool uses.

```yaml
load_cases:
  - name: pull_off
    forces: [...]
    allowable_mpa: 40.0          # the allowable of the material, MPa
    safety_factor: 8.0           # limit = allowable / safety_factor = 5 MPa (only together with allowable_mpa)
    allowable_source: "where the 40 MPa comes from"
  - name: hang
    forces: [...]
    stress_limit: 1.5            # a final limit in MPa. It takes no safety_factor
settings:
  objective: mass                # minimize the mass; the default is compliance with a volume fraction
  move_limit: 0.03               # the upper limit of the trial step (see below)
```

- Two ways to write a limit: `allowable_mpa` + `allowable_source` + `safety_factor`, or a final `stress_limit` with no
  `safety_factor`. A `safety_factor` together with `stress_limit` is a loader error. A lone `safety_factor` is a loader error too.
- `objective: compliance` (default) is the run as before. A stress limit adds one row to it.
- `objective: mass` minimizes the mass. There is no volume row, and `volume_fraction` is the start density.
  If no load case has a limit, `topopt check` says so as a fact.
- The stress in the optimizer is a relaxed measure (`rho^0.5 * D0 * strain`, von Mises, KS aggregate). It is not the stress of
  the printed part. A met row is not a proof that the printed part is below the limit. In the committed example both rows are met
  for the relaxed measure, while the voxel estimate of the part stress of `pull_off` is above its limit. The run log and
  `result/metrics.json` (`stress.cases.<case>.summary`) put the two together: read that text, not only "met".
- The row assumes the filter rule of this tool: the density filter counts the solid voxels (fixed and load regions) as density 1.
  The pipeline filters over the design elements only, and the filter radius in mm changes the result. The same limit and radius
  give different stress in the two tools.
- Step acceptance (`settings.stress_step_acceptance`, default true): after each MMA step the optimizer solves at the trial design,
  and takes the step only if no stress row gets worse. Else it halves the move limit and tries again (6 tries at most; after the
  last try it takes the step and says so in the log). `move_limit` is the upper limit of the trial step. The run log and
  `history.jsonl` (key `step_acceptance`) list the rejected tries. `false` turns the rule off, for comparisons. A run with no stress row does
  not use it.
- `result/metrics.json` has a `stress` section for each case: relaxed numbers, physical numbers (peak and p99.9 by volume,
  voxel estimates, the volume above the limit) for two designs (`physical`: the grey SIMP snapshot; `physical_thresholded`: rho >= 0.5
  solid, the rest void), the achieved safety factor with a range from the error band of p99.9, the member check, and a `summary`
  text. The two voxel peaks differ by -34 % to +29 % in the measured stress designs and can be on different sides of the limit. The report gives **no pass or fail verdict**: the error band of the voxel p99.9 is -59 % to +57 % of the reference. It is the union of the bench (a beam, a plate with a hole and one real part: -54 % to +13 %, `bench/stress_accuracy.json`) and 5 public parts (51 cells: the voxel p99.9 against the P2 p99.9 of `topopt verify`, `bench/stress_public_parts.json`). The lower end, -58.32 %, is the hook at 2.4 mm (`pull_off`, run type MS) of the public parts; the upper end, +56.91 %, is the phone stand (`poke`, run type MS, 2.45 mm) of the public parts. On the public parts the error was -58 % to +57 %, in both directions: the voxel value was above P2 in 23 cells, mostly grey mass-objective designs, and up to 2.4 times below P2 on a coarse part with thin members, and the member gate did not keep the error inside the bench band. The band is the largest error measured and not a bound for other parts. The voxel peak was below the reference in every measured case, by 5 % to 87 % (lowest -87.14 %: the ladle `full_scoop`, run type MF, voxel 3.25 mm, public parts; highest -4.82 %: the beam with 16 voxels through the thickness, bench). For the stress of the part, run `topopt verify`. When the top 0.1 % of
  the volume is fewer than 5 voxels, p99.9 rests on the top 1 to 4 voxels (below 1000 voxels it is the peak, from 1000 voxels the report gives its rank and both values), and the report says so. "Met" is the relaxed measure of the optimizer, not the
  stress of the part, and the `summary` says so in every case. The text ends with an option to run `topopt verify` for the stress of the part. If `topopt verify` made `verify/verify.json`, the MCP tool `run_result` and
  `topopt levers --run` read it again and show its p99.9 beside the voxel value and its ratio to the limit (`independent_check`, `summary`); `metrics.json` itself keeps the text from the end of the run.
  `relaxed.peak_above_limit` is true when the relaxed peak is above the limit by more than 0.5 % of it, also when the row is met (a `settings.stress_scale` below 1 does that; `topopt check` has a fact for it).
- Measured results of the example (2.4 mm, 220 passes, Linux x86_64, PARDISO): at 1 and 4 threads both rows were met at
  the end and at the end of each beta stage, `final.stl` is one body from the tip to the plate (37.3 to 37.8 g), the run took 56 to 63 s, and
  `topopt verify` gives a stiffness of 667 to 685 N/mm. The limit of `pull_off` is 0.5 MPa: with 5 MPa the relaxed rows were met as well,
  but `final.stl` was the wall plate alone, and with 1 MPa a thin neck joined the tip to the plate. `metrics.json` `stress.load_paths` says for
  each load region whether the part joins it to a fixed region, and the case summary says when it does not. With
  `stress_step_acceptance: false` the 5 MPa form ended with `pull_off` g between +0.006 and +0.094.
  At 1.2 mm, 160 passes: see [examples/hook/README.md](../../examples/hook/README.md).
- Example: [examples/hook/hook_stress.yaml](../../examples/hook/hook_stress.yaml). Details: CONTRACTS.md section 17.
  The accuracy study is `bench/stress_accuracy.py`; the study on public parts is `bench/stress_public_parts.py`; the band is the union of the two.

## A bound on the surface area

An optional setting adds one constraint row that keeps the surface area `P` of the part at or below a bound:

```yaml
settings:
  perimeter_max_mm2: 12000       # optional, a number above 0, in mm2
```

- `P` is the area of the voxel faces between the part and the void, in mm2. It is a staircase surface: in one measured problem it was
  about 1.24 times the area of `final.stl`. Solid regions (fixed, load, solid, spring) count as part. Keep-out voxels and the outside of the
  grid count as void. `P` uses the nominal density (the shape that the printer makes), also in robust mode.
- Without the key the run is the same as before. The row has price 1000, like a volume row, and has no price key.
- `topopt check` prints a `perimeter:` line: the bound, the fixed part (solid voxels next to void), the smallest possible `P` (an exact
  minimum cut of the design space) and the bound divided by the smallest `P`. The ratio is a fact. A bound near the smallest `P` leaves
  little material to spend. In one problem (the `hook_short` example, 40 passes), bounds at 2.1, 1.75, 1.4, 1.17 and 1.05 times the smallest `P`
  gave a compliance of 1.05, 1.34, 8.3, 121 and 1440 times the free run; the table is in CONTRACTS.md section 26.3.
- A bound below the smallest `P` is refused before the run: no design with density 0 or 1 can meet it (a gray design can be a little below it, only near void). The message gives the bound, the fixed part, the smallest `P`
  (rounded up to 0.1 mm2, so the shown value is accepted as a bound) and the options (raise the bound, reduce the solid regions, make the design space bigger). The filter can keep `P` above the smallest
  value, so the smallest `P` is a floor and not a promise. A bound above it can still end not met, and the log says so.
- `run.json` (folder root; `perimeter`: bound, fixed part, smallest `P`, `P` of the start design, `P` of the last design), `history.jsonl`,
  `result/metrics.json` (`constraints`, `rows`) and the run log give the numbers.
- Details: CONTRACTS.md section 26.3.

## Buckling row (optional)

A YAML with `buckling: {min_factor: 150}` adds one row for each load case. The row holds the lowest linear buckling load factor of the case
at or above 150 (a multiplier on the load case as the YAML gives it). It is off without the key, and a run without it does not change.

```yaml
buckling:
  min_factor: 150      # turns the row on
  ks_q: 40             # optional: KS sharpness of the row (default 40)
  modes: 6             # optional: how many of the lowest factors the row holds (default 6)
```

- The row needs a direct solver (PARDISO on Linux x86_64, Cholesky on Apple silicon). With `amg` or `gpu-mg` the run stops at the start with the options.
  Robust mode (`settings.robust: eroded` or `wall`) stops too.
- The eigen solve is the cost. The row runs at the first two passes, at the start of each beta stage, at the last pass, at every pass once the lowest factor
  is within 1.5 times `min_factor`, and else once in 5 passes. `history.jsonl` says for each pass whether the row ran (`evaluated`).
  Measured on the hook (1.2 mm, 4 threads, PARDISO, shared host): 10 to 12 s for an evaluation of one load case, and 1469 s for the example run against 170 s
  without the row. `topopt check` does not include the row in its time estimate: it says how many evaluations the run has (a range), the measured cost of one on the hook, and the worst case for a case that fails.
- The factor is a voxel estimate of the gray design, and it reads high for thin members. A tet model gave a factor of 0.71 to 0.79 times the voxel factor on the hook.
  The tool does not change `min_factor`: you set it with that in mind. `buckling.check: true` gives the factors of the printed design after the run.
- Example: [examples/hook/hook_buckling.yaml](../../examples/hook/hook_buckling.yaml). Details: CONTRACTS.md section 39 (a working number).
