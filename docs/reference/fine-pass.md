# Fine pass: a finer voxel from a finished run

[Back to the README](../../README.md)

A thin feature needs about 3 voxels across to survive. If the voxel of a run is too big for the thinnest feature of the part, the run loses that feature.
A fine pass runs the same problem again on a finer voxel and starts from the design of a finished run. The same YAML works.

## When the part needs one

`topopt check <file.yaml>` prints one line when the part needs a smaller voxel than the run uses. The MCP tool `check_problem` has the same data in the key `fine_pass`.
The line names the voxel that the part needs, the feature that sets it, and estimates of the time and the memory of a fine pass. The voxel of the part is the largest voxel with:

- 2.8 voxels across the thinnest fixed, load and solid region, and the smallest keep-out region,
- 3 voxels across `settings.min_member_mm`.

The voxel is rounded down to 0.05 mm. It is one value. The thickness of a region is twice the radius of the largest sphere inside its mesh (the same quantity as the region check), measured on the exact mesh in a fraction of a second: a box takes 0.03 s, the regions of the hook, the ladle and a 5-inch drone arm 0.03 to 0.7 s. The measure does not depend on the voxel or on the size of the region. It is never above the true thickness, and below it by 0.02 % for a box and by a few tenths of a percent for a curved part (a coarse sphere of 800 faces can be 3 % low).
A plain check, `check_problem`, `build_regions` and the end of every run state this measured value (`topopt check` prints it). A fine pass that is planned or starts (`--fine-from`, `--fine-pass`, `topopt check --fine-from`, the MCP arguments `fine_from` and `fine_pass`) also tests it on its own grid: it voxelizes the load, solid and limiting regions at that voxel and applies the region check. If a region holds too few voxels across there (the grid decides how many voxel centres fall into a thin slab), the voxel goes down in steps of 0.05 mm until the regions pass, and the log says so. So the fine pass can take a voxel below the one that `topopt check` printed. Example: a thin end of 2 mm that is 2.8 voxels across at the measured 0.7 mm can hold only 2 rows of voxel centres on the grid there, so the check says 0.7 mm and the fine pass takes 0.65 mm. Another example: `examples/hook/hook_short.yaml` with a run voxel of 2.0 mm needs 1.25 mm (the wall plate is 3.5 mm thick).
The line shows only when the voxel of the part is below the voxel of the run by 0.05 mm or more.
The end of the run log and the part card (`result/report.md`) show the same line. A fine pass does not start by itself.

The check also tests the fine grid against `settings.max_active_voxels`, `settings.max_grid_voxels` and the memory of the machine. If the fine grid fails one, the line says which, and gives the options: `--fine-voxel H` with the finest voxel that passes, or a higher setting in the YAML.

## Commands

The first run and the fine pass in one command:

```
topopt run part.yaml --out runs/part --fine-pass
```

This makes the normal run in `runs/part`, then the fine pass in the sibling folder `runs/part-fine`. If the part does not need a finer voxel, the command logs a fact and ends after the first run. With `--fine-voxel H` that is finer than the run, the fine pass runs at H also when the part does not need it (the same as `--fine-from ... --fine-voxel H`): the log of both runs says "the part needs about X mm; --fine-voxel H is finer, so the fine pass runs at H as asked". A `--fine-voxel` that is not finer than the run voxel is refused before the first run. If the fine grid fails a guard (see above), the command stops with the options before the first run, and writes nothing.

The fine pass from a run that finished earlier:

```
topopt run part.yaml --out runs/part-fine --fine-from runs/part
topopt check part.yaml --fine-from runs/part
```

The second command shows the plan and writes nothing. The parent folder is only read. MCP: `run_topopt(yaml_path, out_dir, fine_from=PARENT)` or `run_topopt(yaml_path, out_dir, fine_pass=true)`.

| Option (MCP argument) | Meaning |
|---|---|
| `--fine-voxel H` (`fine_voxel`) | The voxel in mm. Default: the voxel that the part needs, tested on the grid of the fine pass. A voxel equal to the voxel of the parent is refused (use `--continue-from` to add passes on the same grid), and so is a voxel of the part that is not finer than the voxel of the parent. `--fine-voxel` forces another voxel. |
| `--fine-passes N` (`fine_passes`) | The maximum number of passes. Default: the pass count of the parent run. |
| `--fine-tol T` (`fine_tol`) | The tolerance of the stop rule `settled` for one run. It wins over the YAML key below. Default 0.001. |
| `--no-stop-rules` (`stop_rules=false`) | No early stop at all: the fine pass runs to its maximum. |
| `--allow-changed-inputs` (`allow_changed_inputs`) | Go on if the YAML or a region file changed since the parent. The voxel, the pass count and the beta schedule are not changes. |

## The tolerance in the YAML

```yaml
settings:
  fine_pass:
    objective_change: 0.0005
```

The key is the tolerance of the stop rule `settled` for every fine pass made from this YAML (`--fine-from` and `--fine-pass`). It has no effect on the passes of a plain run. `--fine-tol` wins for one run. The record in `run.json` (`fine_from.tolerance_from`) says where the value came from: `option`, `yaml` or `default`. The value must be a finite number above 0.
In a measured 5-inch drone arm, 0.001 stopped at pass 21 (+0.46 % against the 80-pass run from zero), 0.0005 at pass 30, and 0.0001 did not stop within 30 passes.

## How it works

- The design of the parent is mapped to the finer grid by place. The volume is fitted to the target with one uniform shift (for `settings.objective: mass`, which has no volume row, to the final mean density of the parent). The optimizer memory starts new.
- The beta schedule goes on from the pass where the parent ended. A restart at a low beta was worse at the same pass in the measured runs.
- The fine pass ends when its loss has settled: the change of the objective between passes is at or below the tolerance (over the last 5 passes at one beta) and the constraints are met. The rule is called `settled`.
  The run log says which test fired. If the maximum of passes ends the run first, the log says so.
- A killed fine pass continues with `--resume`. `--continue-from` adds passes to a fine pass on its fine grid, with the same YAML. A fine pass can be the parent of the next fine pass.
- An error of the fine grid names `--fine-voxel`, and not `settings.voxel_size`.

## What it costs

In two measured parts the result came within 0.5 % and 2.1 % of the compliance of a run from zero at the fine voxel, with fewer passes at the fine voxel.
The first run is extra time. The time and memory numbers of the tools are estimates. Details and the numbers of the measured parts: `CONTRACTS.md` section 24.6.
