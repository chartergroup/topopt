# Verify the result, ratchet and cleanup

[Back to the README](../../README.md)

## Verify the result

`topopt verify <run_dir>` checks the final STL of a finished run with quadratic tets (P2). It is independent of the voxel model:
another mesh and another element (quadratic tets). The linear solver does not matter for this: every factor gives the same solution, and it can be the one of the optimizer. It reports the von Mises stress for each load case. The MCP tools `verify_run` and
`verify_status` do the same, as a job in the background.

```
uv pip install -e ".[verify]"
topopt verify runs/hook --estimate-only     # the memory and time estimate, no meshing
topopt verify runs/hook                     # about 10 s on the hook at the default mesh size (1.93 mm, 81k dofs), peak about 0.8 GB
topopt verify runs/hook --levels 1.5,1.0    # two mesh sizes: how much the numbers move
topopt verify runs/hook --allowable-mpa 5   # exit code 3 if the p99.9 stress is above 5 MPa
topopt verify runs/hook --buckling          # also the linear buckling load factors of the tet model (about 9 s more on the hook)
topopt verify runs/part --gate              # a pass or a fail from the limits of the YAML (stress, buckling, motion rows); exit code 3 if one fails
topopt verify runs/part                     # with a bore_checks block in the YAML: also the bore facts and the Goodman rows (CONTRACTS.md section 40)
topopt verify runs/part --part-only         # a run with regions.not_printed: mesh result/part_only.stl, the removed bodies become boundary conditions
```

**License.** The extra `verify` holds gmsh (GPL-2.0-or-later, see "Tet mesher"), scikit-fem (BSD-3-Clause) and cholespy.
cholespy: GPL-2.0-or-later in effect: the wheel bundles CHOLMOD Supernodal, SuiteSparse 5.4.0. The code of cholespy itself is BSD-3-Clause,
and the package data says so. The wheel also has the CHOLMOD Core, Cholesky and Partition modules (LGPL-2.1-or-later), AMD, COLAMD and METIS.
It stays an optional extra: never a main dependency, never vendored, and the core of topopt never imports it (a test checks this).
This follows the rule for gmsh. Install the extra only if the licenses fit your use.
(The file [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) lists the license of every package of `pyproject.toml`, with how each was checked.)

**What it does.** It meshes `result/nominal.stl` when that file exists, and `result/final.stl` if not, with tets (default size 2.5 times the mean edge of the STL, about 1.3 to 2.2 voxels for an `sdf` STL; the report prints it). A fixed region fixes all P2 nodes in it. A load region
becomes a uniform traction over its facets: the resultant is the force of the YAML. A mesh node belongs to a region when it is inside the region STL or
within a band of its surface (default 0.5 voxel; the report shows the counts at 0, 0.5 and 1 band). Each load case is solved alone: the case weights
are not used. Keep-out regions are not used. It also compares the YAML with `run.json` (case names, forces, region files) and reports a difference
as a fact.

**Joints and springs.** A part that a `joints` or `springs` entry holds runs in verify with the same physics as the optimizer (CONTRACTS.md section 16). `regions.fixed` can be empty.
A spring adds its stiffness to the tet matrix on the facets of its region. A joint bonds the facets of its region to a rigid body with the same bond stiffness (`c E / h`, `c` = 30, `h` the voxel size of the run)
and the same beam (`joints.beam_stiffness`). The solve uses one bordered matrix and one factor. The report has the key `supports`: the facts of each joint and spring, and for each load case the
force (N) and the moment (N mm) of each bond on the part, the 6 numbers that a bond check of the adhesive needs. A joint region is chosen like a load region (the boundary facets in the band of its STL),
also when no load case puts a force on it. The buckling check runs with the condensed stiffness `S = K + Kb - C D^-1 C^T`. The beam has no geometric stiffness, so those load factors are for the part
and not for the beams. The press-fit stage of the bore checks is skipped with a joint and runs with springs. A moment, a remote force, a pressure and a per-axis support are still refused, with the options.
On a small test part with two joints and no fixed region, the tet stiffness is 0.995 of the voxel stiffness (`stiffness_over_optimizer`).

**Bodies that are not printed (`--part-only`).** A run with `regions.not_printed` in the YAML also writes `result/part_only.stl`: the part without those
bodies (a plate that the part is bolted to, a load block; CONTRACTS.md section 9.1). `topopt verify <run_dir> --part-only` meshes that file. The region rule and the band stay the same.
A removed fixed body clamps the nodes that the rule selects at the contact face. A removed load body gives the traction on the facets that the rule selects, and the resultant stays equal to the
force of the YAML. A removed solid or keep-out body gives no boundary condition, and the report says so: if nothing else holds a piece of the part, the matrix is singular and the
command stops with exit code 1, the facts and the options. For each removed body the report has the selected nodes and facets, their area, the counts at 0, 0.5 and 1 band, and the
contact area of the cut (`part_only` in `verify.json`, and one `removed body` line each in the summary). A run without the file stops with exit code 1 and the options. Without the
option the result does not change, and one `note:` line names the option when the file is there. Machining stock stays in `part_only.stl` (it is cut from `final.stl`); verify
makes no second cut. A thin film that the cut can leave around a removed body is part of the mesh. On the hook it makes the stiffness of this model far lower than the voxel
stiffness, so read the boundary-condition facts first.

**Result.** `<run_dir>/verify/verify.json`, with `fields.npz` (stress of each tet), `mesh.npz` and `status.json`. With `--levels`, the mesh and field
files are in a folder for each size, and `files` in `verify.json` names them. A new run removes the files of the earlier run from `verify/` first.
For each case: von Mises p50, p95, p99 and p99.9 by volume, the peak at the quadrature points and at the vertices, the same values without bad tets,
the top elements, the force and moment residual, the solve residual, the resultant error and the stiffness.
After the summary the command prints the answer to "will it break?" in plain words: one `plain:` line for each load case (the compared value, who gave the allowable, the margin) and a line for what the check does not cover. `verify_status` has the same lines as `plain`. The lines compare two numbers; they promise nothing.

**What the p99.9 stands for.** The p99.9 is the main number, but it is a volume percentile and it depends on the load patch. On the hook (pull-off), 72 to
82 % of the top 0.1 % of the volume is within 2 mm of the tip region, and the load band (which facets take the traction) moves the p99.9 by about
10 % (1.93 MPa at band 0, 2.06 MPa at the default 0.6 mm, 1.86 MPa at 1.2 mm). The report gives the facts to judge it:
`near_regions` (the p99.9 and the peak without the points within `--near-mm` of a region, and the share of the top volume near each region),
`load_band_sensitivity` (the same load at a band of 0 and of 2 times the band, the factor is reused; `--no-sensitivity` switches it off),
`load_centre` (the centre of the optimizer load against the P2 traction centroid, 3.1 mm on the hook, and the moment ratio), `normal_check`
(whether the load facets face one way) and `peak_qp_over_p99_9`. The peak at the quadrature points moves with the mesh: +20 % from 2.0 to 1.5 mm
and from 3.8 to 4.4 MPa (+16 %) from 1.5 to 0.8 mm on the hook, while the p99.9 moved about 5 %.
The P2 stiffness is lower than the voxel stiffness of `metrics.json`. Most of the gap is load modelling (the traction on the tip facets against a force
spread over nodes: 776 against 1091 N/mm for pull-off); the rest (1091 against 1331 N/mm) is the voxel model against the STL. Both stiffnesses are in the report.

**Buckling and motion rows.** Two more checks give facts, not verdicts. They are off unless the run asks for them, and with both off the
result is the same as without them.

- *Buckling.* `--buckling` (or `buckling: {check: true}` in the YAML of the run; `--no-buckling` turns the YAML switch off) gives the lowest linear
  buckling load factors of each load case on the tet model, with the factor of the stress check (no second factor). `buckling.modes` and
  `buckling.max_seconds` of the YAML are the settings. The key `buckling` of `verify.json` has the load factors, the voxel value of the run (`topopt run
  --buckling` or `topopt buckling <run_dir>` writes it into `result/metrics.json`) and `tet_over_voxel`. On the recorded hook run (made with the `blur` mesh method) the tet value is 0.63 (hang) and 0.61
  (pull-off) times the voxel value; for a new run of the same design (the default `sdf` method) it is 0.79 and 0.71. `blur` makes thin members thinner than the voxel design, so the tet result
  depends on the mesh method of the run (`mesh_method` in `run.json`, `mesh.method` in `metrics.json`; a run from before `sdf` has neither key and is a `blur` run): compare runs of the same method. This is the difference of two models, not an error of one: the hex8 voxel element reads high (+22 % on the same solid),
  and the STL geometry, the region rules and the facet traction also differ. The check adds about 9 s and
  about 0.1 GB to the measured peak on the hook at the default size. The memory estimate has a separate term for it. If the estimate with the term is above the
  limit, the check is skipped with the facts and the options, and the stress check goes on.
  Each mode of the buckling record has facts: a local or a global mode (`local` is small against the part volume), and the region with the largest motion and its position (`mode_facts`; a measure region of a `limits` row is a region name too).
  A part that only joints or springs hold (no fixed region) can have a lowest mode that is a motion of the part on its supports (the supports hold half of the energy or more: `support_energy_share`; `kind` is then null). Then verify solves again
  without the six rigid-body motions of the part (key `rigid_body`) and gives both results: `load_factors` is the measured lowest factor and the one that the gate and the ratchet read; `rigid_body.load_factors` is the factor of the part itself, a fact next to it
  (the part itself buckles at that value; the lowest mode is a swing on the supports). Reading the higher one in place of the measured one would loosen `buckling.min_factor`, a requirement. The projected value depends on the mesh grading by about 3 to 4 %
  (`CONTRACTS.md` section 33). The three thresholds of these facts (90 %, 25 %, 0.5) are first values from two test parts, and module constants. The check does not cover the geometric stiffness of the beams
  of a joint (the beams have none), imperfections of the part, nonlinear effects or loads that follow the deformation. It is a linear estimate, not a verdict.
- *Motion rows.* If the YAML has `limits.displacement` or `limits.alignment` rows (`CONTRACTS.md` section 31), verify computes the same rows from the tet
  displacement of the printed STL (the key `motion`): the tet value, the voxel value of the run, their difference, the limit and `tet_over_limit`. There is no
  pass or fail flag unless you give `--gate` (below). A displacement is the mean over the P2 nodes of the region. On a new run (`sdf`) of the hook design with a displacement limit of 0.021 mm, the voxel row is
  0.020998 mm and the tet value is 0.0257 mm, 1.23 times the limit (0.0343 mm, 1.63 times, on the `blur` `final.stl` of the same design). The value is a mean over the P2 nodes, so it moves by 5 to 6 %
  between the mesh sizes 1.55 and 1.78 mm (one shape): use `--levels` to see the change for your design (`motion_tet_value`).

**The gate (`--gate`).** By default `verify` gives facts and no verdict. `--gate` (MCP: `gate=true`) adds a pass or a fail, from limits that the YAML already has. It adds no YAML key.
The result gets the key `gate`: for each check and each case the value, the limit, where the limit comes from, and `met`; and `passed` for the whole gate.

- *Stress.* The stress by volume at `--percentile` (default 99.9) must be at most the limit of the load case: `stress_limit`, or `allowable_mpa / safety_factor`. `--allowable-mpa` replaces the limit of every case.
  When it replaces a limit of the YAML, the record keeps `replaced_limit`, `replaced_source` and `loosened` (true when your number is above the YAML limit), and the `gate:` line says so. A larger number loosens a requirement of the YAML: that needs the consent of the user.
  A case with no limit is listed as "not gated" and does not fail the gate. The margin (limit divided by value) and the highest value at one point are in the record, as for `--allowable-mpa`.
- *The limits are for the voxel model.* The limits in the YAML are the limits of the optimizer, and the gate reads them as limits of the tet value. The two models differ by a measured amount (hook, one part, one host): the p99.9 stress of `pull_off` was 2.45 to 2.67 times a limit
  that the optimizer row had met, the tet buckling factor was 0.61 to 0.79 of the voxel factor, and the tet value of a displacement row was 1.23 to 1.63 times the voxel value (`CONTRACTS.md` sections 16, 31 and 33). So a design that met its limits on the voxel model can fail the gate.
- *Buckling.* If the YAML has `buckling.min_factor`, the gate turns the buckling check on, and the lowest tet load factor of each case (the measured `load_factors[0]`) must be at least `min_factor`. A load case with no compression (a pure tension state) is listed as "not gated" with the reason: nothing can buckle in it.
  The gate line of a case gives the key that the value comes from (`value_source`), the lowest value of the run (`run_lowest_load_factor`) and, when the part floats on soft supports, the value without the rigid-body motions (`projected_value`, a fact that the gate does not read).
- *Motion rows.* The tet value of each row of `limits.displacement` and `limits.alignment` must be at most the limit of its row.
- *A check that could not run fails the gate.* If the buckling check is skipped (memory) or stopped, or a row cannot work, the check has `met: null` and the gate fails. A gate with no limit at all is refused (exit code 1).
- *Levels.* With `--levels` the gate reads the finest level and gives the change from the level before it, so you see if the number still moves with the mesh.

The test is strict: a value on the limit passes, a value past it fails. The gate does not cover fatigue (see the bore checks below), the peaks at sharp features, printing defects or the print direction.
Exit code 3 means "a limit failed": the `--allowable-mpa` limit, or the gate. Without `--gate` the result and the exit codes are as before. The ratchet rules `buckling`, `alignment` and `displacement` (below) use the same facts.

**Bore checks (Goodman and press fit).** If the YAML has the block `bore_checks`, verify also reads the stress of each load case at the wall of the bores and gives the Goodman factor
`n = 1 / (sa/Se + sm/Su)` (`CONTRACTS.md` section 40). The block names the bores (point, axis, radius, length, or the `interface:` of a `bore` entry of `interfaces`), the strengths `su_mpa`, `sy_mpa` and `se_mpa` with a source text for each
(topopt takes no strength from a catalog and applies no factor), and four optional facet settings that have defaults. `sa` is the von Mises p99.9 and maximum of a load case in the quadrature points next to the bore wall; `sm` is the
hoop stress of the press fit (below), or the `mean_stress_mpa` of the bore, or 0. The keys `bore_checks`, `bore_facts` (the wall facets, the area over the analytic area, the sample points, and the numbers that show why a value moves with a setting), `goodman` and `pressfit`
are in `verify.json` only when the block is there. `--no-bore-checks` turns the block off for one call; `--bore-checks` needs the block. The result is a number for comparison, not a verdict: `sa` is a von Mises value and `sm` a normal
stress (the method of the pipeline check), each load case is solved alone and taken fully reversed, and the facets depend on the mesh and on the settings. The stress is not smoothed, so the p99.9 and the maximum carry the
point scatter of the P2 stress, which changes with the mesh (on a thin test ring at a mesh size of 2 mm, p99.9 was 18 percent above the closed form on one mesh and 3 percent on another; `CONTRACTS.md` section 40.7): the text gives the median next to them, and
`--levels` shows how `n` moves between meshes. A `press_fit` block (the cup variants and the diametral interference of each bore, with sources) adds the press-fit check: the cup is a linear spring on the bore wall (`p = k (d_r - u_r)`), one linear solve for each cup variant, and the hoop stress at the bore of each variant is `sm` of its Goodman rows (key `pressfit`: hoop p99.9 and max, the contact pressure as an area mean and at the vertices, the force of the spring and the reaction of the clamps; with a memory estimate before the stage, which is skipped with a record when it does not fit). The load cases are solved with no cup spring. A stiff cup gives vertex pressures that are not a stress measure; the area mean and the hoop stress are the numbers to read (`CONTRACTS.md` section 40.5). The hoop p99.9 (`sm`) of a bore that ends at a free face moves with the mesh at the bore mouth: on ring meshes it was +0.5 % to +2.7 % above the closed form, and +5 % to +7 % on one ring where the top points were at the bore mouths; the text gives the median next to p99.9 and max. On the stored case data of the acceptance runner (level 2b; the data is not shipped), the formula of topopt reproduces the 24 rows of the old pipeline check exactly.

**Exit codes.** 0 it ran; 1 bad input or usage, a part that the tet model cannot solve (a body with no support, a load region that no body touches, a singular matrix), or no linear solver that can run (the one you name with `--solver`, or none for `auto`); the text gives the facts and the options; 2 it could not run (a missing gmsh or scikit-fem, a gmsh failure, a memory refusal, an empty fixed region, a SuperLU factor that ran out of memory);
3 a limit failed: you gave `--allowable-mpa` and a case is above it at the percentile (`--percentile`, default 99.9), or you gave `--gate` and the gate fails. There is no default limit.

**Memory and time.** The estimate is `7.5 GB x (million dofs) x 1.25` for `cholesky`, `pardiso` and `cholespy`. `superlu` has its own model, because its fill depends on the
dofs (the model uses the square; the hook measured about N^1.65) and on how compact the part is (a cube needs several times the memory of a hook with the same dofs). Its estimate is a
central value with no margin, and the line gives a range: the peak was 0.69 to 1.67 times the central value on the 17 meshes of 5 shapes that were measured (1.67 times on a plate,
0.77 times on the hook at 0.8 mm). It is printed before meshing, for every level, with the time in seconds for each
solver that has a measurement (fit on one Linux host; none for `cholesky` on a Mac). The command refuses to start if the memory estimate (for `superlu`: the central value) is above the
available RAM or above `--max-gb`. If only the upper end of the range is above it, the run starts and a `note:` line says so, with the options. `--force` runs anyway. On the Linux hook the peak was 1.2 GB at 1.5 mm (135k dofs), 2.6 GB at 1.0 mm and 4.8 GB at
0.8 mm. A mesh of half a voxel (about 1.5 M dofs) needs 11 to 14 GB: use `--levels` with sizes that fit your RAM.

**MCP.** `verify_run` with `estimate_only` (the default) returns the estimates. With `estimate_only=false` it starts the job in the background and
returns at once. The checks of the section above follow the YAML of the run (`buckling.check`, `limits`, `bore_checks`); there is no argument for them, except `gate=true` for the gate. `verify_status` shows the state and, when the job is done, the result. `part_only=true` is `--part-only` (above). Only one verify job or optimizer run goes at a time.

**Solver.** `--solver auto` takes `cholesky` on macOS arm64, then `pardiso` on Linux x86_64 (extra `pardiso`), then `cholespy` (extra `verify`), then
`superlu` (scipy, BSD-3-Clause, always installed). The output names the factor that ran, why `auto` passed the others, and its time and memory estimate.
cholespy is slower on large meshes (47 s against 16 s at 1.0 mm). `superlu` needs no extra, but its memory and time grow fast with the size, and more for a
compact part than for a slender one like the hook (see the estimate lines). If no factor can run, or the one you name cannot, the
command stops with exit code 1 and lists the facts and the options. `amg` is not offered: it is an iterative method, so its error depends on its tolerance (it is a
possible later option for a low-memory run). A recording (a copy of a finished run folder in `recordings/`) names its YAML relative to its folder (`../../examples/hook/hook.yaml`), so `topopt verify`, `topopt refine` and `topopt buckling` find it from any current folder. These commands write into the run folder, so copy the recording first and run them on the copy (`runs/` has the same depth as `recordings/`, and it is not tracked): `mkdir -p runs && cp -R recordings/<run> runs/hook_replay`, then `topopt refine runs/hook_replay`. A run folder that you move out of the repo tree, or a run from another machine, needs `--yaml examples/hook/hook.yaml`. To add a recording, see `recordings/README.md`.

On Linux, gmsh needs system libraries: `sudo apt install libglu1-mesa libxft2 libxinerama1 libopengl0`.

## Ratchet (keep a candidate edit only if the rules pass)

The ratchet judges candidate edits of a finished part and keeps the ones that pass the rules of the case. It uses `topopt printcheck` and
`topopt verify` (extra `verify`) and makes no candidate of its own. A candidate is a mark of the Refine page (`marks:FILE`, made by
`apply_marks`) or an STL file in a folder (`stl:FOLDER`). The start part is judged first. A candidate that passes every rule is the new parent
(`acc_01`, `acc_02`, ...); a candidate that fails stays in the log with the reason. The start run is never written.

The rules are yours. Add a block `acceptance` to the problem YAML; a rule that you do not list is off, and a rule that you list and that has no input fails
with the reason. Every number has a source text. No shipped YAML has the block `acceptance`, so `python -m topopt.ratchet check runs/hook` stops with
`no rules: the YAML ... has no 'acceptance' block`. Add the block to a copy of the YAML and pass the copy with `--yaml`. Put the copy in the same folder as the
original (the region paths are relative to the YAML), for example `cp examples/hook/hook.yaml examples/hook/hook_ratchet.yaml` (git ignores a file `examples/*/*_ratchet.yaml`, so the copy is not an
untracked file in `git status`). This block fits the hook:

```yaml
acceptance:
  rules:
    - rule: watertight
    - rule: bodies                      # at most print.max_bodies (the YAML; the check says which limit it used)
    - rule: stress                      # the p99.9 stress by volume of each load case
      against_parent: not_worse         # hook.yaml has no stress_limit, so the candidate is compared with the parent
      tolerance_pct: 2
      tolerance_pct_source: "example value only; choose your own from the noise of a remesh"
    - rule: stiffness
      max_drop_pct: 5
      source: "example value only"
    - rule: mass
      must_fall: true
```

The default form of `stress` is `absolute`: the p99.9 stress is at most `load_cases[].stress_limit` (or `allowable_mpa / safety_factor`). `hook_stress.yaml` has
these limits; `hook.yaml` has none, so with `absolute` every candidate fails with `no_input` and the options. `not_worse` needs no limit. A new mesh of a changed
surface moves the stress a little, even for a harmless edit, so with no `tolerance_pct` a rise of any size fails; each check shows how far the value moved
(`delta_pct`).

The rules `buckling`, `alignment` and `displacement` read the tet buckling factor and the motion rows of `verify`. When a rule `buckling` is listed, the judge runs `verify` with the buckling check.
`buckling` needs `min_factor` (the lowest factor that is allowed) or `against_parent: not_worse`; `alignment` and `displacement` use the limits of the `limits` rows of the YAML. All three take `against_parent`
(`absolute`, `not_worse`, `cap_or_parent`), `tolerance_pct` and `cases`, as `stress` does.

```
python -m topopt.ratchet check  runs/hook --yaml examples/hook/hook_ratchet.yaml   # the rules, what each needs, what is missing; no solve
python -m topopt.ratchet run    runs/hook --yaml examples/hook/hook_ratchet.yaml --source stl:candidates/   # or marks:marks_hook.json
python -m topopt.ratchet status runs/hook-ratchet
```

Results are in `runs/hook-ratchet/`: `ratchet.jsonl` (one line for each candidate: the decision, the failed rules, every number with its limit and its
source), `ratchet.json` (the state) and one folder for each parent. Stop it and start it again with the same root: it goes on, if the start part, the rules, the mesh
size and the inputs (the YAML, the region files, `run.json`) are the same. If one of them changed, the call stops and says which one (the old and new value). Then
use a new root; or `--rebaseline`, which judges the current parent again under the new inputs and goes on with the same log; or `--overwrite`, which moves the old
files into `earlier_NN` and starts again. Nothing is deleted. `--stop-before-promote` stops
before a candidate is promoted (exit code 3); `python -m topopt.ratchet approve ROOT` promotes it. The MCP tools are `ratchet_run` (estimate first),
`ratchet_status` and `ratchet_approve`; only one job goes at a time. A judge call took 4 to 28 s on the hook and the ladle. A mark is judged by `stress` and `stiffness`
like any other candidate (the sewn output of `apply_marks` is a mesh that `verify` reads; if an edge under 0.05 mm stays after the preparation of `verify`, `verify` welds the
input once or, if that is refused, cleans the refused surface, prints one `surface:` line, and the ratchet row has the same line in `facts.notes`); if a step falls back to the plain boolean seam or the weld does
not give a closed surface, and `verify` refuses the surface, the log has the verify error and the candidate is a REJECT.
Contract: CONTRACTS.md section 36.

## Cleanup list (find spurs, bumps and loose bodies)

`python -m topopt.cleanup list` (and the MCP tool `cleanup_list`) scans the finished part for loose bodies, dead-end spurs, thin bumps,
uneven struts and flat webs. It writes a NEW folder with `marks.json` (the marks file that `apply_marks`, the Refine page and the ratchet
read) and `cleanup.json` (the full report: the facts of each candidate, the limits used with their source, the estimate). It edits nothing
and gives no verdict. The ratchet judges the list:

```bash
python -m topopt.cleanup list runs/hook --estimate                  # memory and time first; no file
python -m topopt.cleanup list runs/hook                              # runs/hook-cleanup/marks.json + cleanup.json
python -m topopt.ratchet run runs/hook --yaml examples/hook/hook_ratchet.yaml --source marks:runs/hook-cleanup/marks.json --root runs/hook-ratchet
```

The list takes 2 to 12 s (the full hook at 0.5 mm: 6 s; the ladle: 9 s). The judge takes 20 to 32 s for each candidate on the hook. The
limits are options (`--set spur_min_length_mm=4`, `--pitch`, `--max-candidates`, default 10); each default is the value of the research
pipeline, and `cleanup.json` names its source line. The detectors assume a lattice of struts: on a shell or plate part (the ladle) the
header block `applicability` shows how much of the part the members cover. The result depends on the pitch: the full hook gives 2, 1 and
4 candidates at 0.3, 0.5 and 0.6 mm. On the hook the ratchet at 0.6 mm accepted 2 of 4 candidates (volume 21362.8 to 21278.0 mm3) and
rejected the two bump removals that raised the stress. A part with a loose body gets the body marks only; judge them first, then list the
`acc_01` folder of that root. Contract: CONTRACTS.md section 29.3.
