# Gravy ladle (example)

A ladle for a kitchen: a grip at one end and a bowl at the other. The optimizer carves the handle from one big block between them. The bowl and the
grip stay as they are. The picture shows the finished part from the side.

![The finished ladle in side view: a grip block, a carved handle with two openings and a half-sphere bowl.](result.png)

```
topopt check examples/ladle/ladle.yaml
topopt run examples/ladle/ladle.yaml --out runs/ladle
```

## Files

| File | Role | What it is |
|---|---|---|
| `design.stl` | design | One block, 154 x 96 x 80 mm, with room above and below the line from the grip to the bowl. |
| `grip.stl` | fixed | The end that the hand holds, 20 x 24 x 16 mm at x = 0..20. |
| `bowl.stl` | load `bowl` | The whole bowl shell: a half sphere of radius 36 mm, a wall of 9 mm, open up, centred at x = 182. The loads act on it. |
| `gravy_space.stl` | keep_out | The scoop cavity: a sphere of radius 26 mm at the bowl centre. |
| `over_opening.stl` | keep_out | A cylinder of radius 36 mm over the opening of the bowl, so that nothing grows over it. |
| `ladle.yaml` | problem | The problem YAML that uses the files above. |
| `ladle_spec.yaml` | part spec | The same part as shapes, not a problem YAML. `python -m topopt.regions_builder examples/ladle/ladle_spec.yaml --out runs/ladle_built` makes the STL files and a YAML from it (`topopt run` refuses the spec itself). |

## Load cases

All three cases have the same weight (1.0) and act on the bowl.

| Case | Force on `bowl` (N) |
|---|---|
| `full_scoop` | 4.9 down (z) |
| `scrape` | 30 back (-x) and 40 down |
| `side_stir` | 18 sideways (+y) and 5 down |

Settings: 3.25 mm voxels (36,542 active voxels), 80 passes, `volume_fraction: 0.10628` of the free design voxels, the default printed polymer
(E 3500 MPa, nu 0.35, 1.24 g/cm3).

## What it shows

The part is a design space that is much bigger than the result. The handle can go above or below the line from the grip to the bowl. The optimizer
leaves a body with two openings in the side view. It also shows the keep-out regions: nothing grows into the scoop or over the opening of the bowl.

## The numbers

The run: 2026-10-28, main 5f2a523, `settings.solver: auto` that picked `gpu-mg` on an integrated Intel Arc GPU. The numbers come from the files of that run folder.

| | Value | Source |
|---|---|---|
| Mass before (the full design space, all solid) | 1,555.5 g | `mass_design_space_g` of `result/metrics.json` |
| Mass after (`final.stl`) | 236.9 g (84.8 % less) | `mass_final_g` |
| Run time | 49.3 s (the 80 passes took 39.4 s, the rest is the post step) | `run_time_s`; `elapsed_s` in `history.jsonl` |
| End of the run | converged at pass 80 (grayness 0.012) | `verdict.json` |
| The mesh | closed, 35,072 triangles | `metrics.json` |

At 3.25 mm voxels the shape is coarse. The mesh of the post step smooths it. For a finer shape, see [Fine pass](../../docs/reference/fine-pass.md).

The gallery of all examples is [examples/README.md](../README.md). The picture is made as written there.

## Licence

The files here are our own work, under the licence of this repository (`LICENSE` at the root).
