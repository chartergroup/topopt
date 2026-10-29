# Bracket with interfaces (example)

A small bracket with three exact shapes: a boss, a bore and a machined face. The optimizer finds the web between the clamped plate and the free end.
The post step unites and cuts the three exact shapes after the smoothing (CONTRACTS.md section 23).

![The finished bracket: a block with a cut-out web, a round boss with a bore and a flat mounting plate.](result.png)

```
topopt check examples/interfaces/bracket.yaml
topopt run examples/interfaces/bracket.yaml --out runs/interfaces
```

## Files

`make_part.py` writes the three region files (`.venv/bin/python examples/interfaces/make_part.py`). The bore, the boss and the face are not STL files:
they are the `interfaces` list of the YAML.

| File | Role | Box (x / y / z, mm) |
|---|---|---|
| `design.stl` | design | 0..40 / 0..20 / 0..16 |
| `plate.stl` | fixed | 0..4 / 0..20 / 0..16 (its outer 2 mm are machined away) |
| `tip.stl` | load `tip` | 37..40 / 8..12 / 0..4 |
| `bracket.yaml` | problem | With a stock of 0.4 and 0.5 mm: `final.stl` is the as-printed part and `nominal.stl` the finished part. |
| `bracket_nostock.yaml` | problem | The same part with no stock: one mesh, no `nominal.stl`. |

The load case `pull_down` is 15 N down (-z) on `tip`. The interfaces:

| Name | Kind | What it does | Stock |
|---|---|---|---|
| `bushing_seat` | boss | A boss of radius 5.5 mm at (28, 10), 20 mm long, along z. The optimizer keeps it solid. | 0.4 mm larger |
| `pin_bore` | bore | A bore of radius 3 mm through the boss and the part. | 0.4 mm smaller in radius |
| `mounting_face` | face | The material at x >= 2 stays. | 0.5 mm further out (x = 1.5) |

Settings: 1.0 mm voxels (12,352 active voxels), 40 passes, `volume_fraction: 0.3`, the default printed polymer.

## What it shows

The exact shapes are not voxels: the round boss and the bore stay round, and the face is flat. With stock, the printed part (`final.stl`) has material to
machine, and `nominal.stl` is the finished part. The picture is `final.stl`.

## The numbers

The run: 2026-10-28, main 5f2a523, `settings.solver: auto` that picked `gpu-mg` on an integrated Intel Arc GPU. The numbers come from the files of that run folder.

| | Value | Source |
|---|---|---|
| Mass before (the full design space, all solid) | 15.3 g | `mass_design_space_g` of `result/metrics.json` |
| Mass after, as printed (`final.stl`) | 6.7 g (56.3 % less) | `mass_final_g` |
| Mass of the finished part (`nominal.stl`) | 6.1 g | `interfaces.nominal.mass_g` of `metrics.json` |
| Run time | 13.5 s (the 40 passes took 7.6 s, the rest is the post step) | `run_time_s`; `elapsed_s` in `history.jsonl` |
| Stiffness at the tip | 642 N/mm | `result/report.md` |
| End of the run | stopped at the pass limit (40), not settled: grayness 0.129 (limit 0.1) | `verdict.json` |

The run stops at the limit of 40 passes before the density is near 0 or 1. It is a short example, not a design to print. For a settled result, raise `iterations`.

The gallery of all examples is [examples/README.md](../README.md). The picture is made as written there.

## Licence

The files here are our own work, under the licence of this repository (`LICENSE` at the root).
