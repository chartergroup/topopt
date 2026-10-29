# Footbridge (example)

A model footbridge for a printer bed. A thin solid skin on top carries the walking surface, two feet stand on the ground at the ends,
and an opening under the span is kept free. The optimizer builds everything between the skin and the feet, at a tenth of the free
design space. The bridge is our own: `make_footbridge.py` makes the region files from named round dimensions (see "The bridge").
The gallery of all examples is [examples/README.md](../README.md). The picture is made as written there.

```
topopt check examples/footbridge/footbridge.yaml
topopt run examples/footbridge/footbridge.yaml --out runs/footbridge
```

`footbridge_short.yaml` is the same problem on a coarser grid, for a first try (about a minute and a half on our host, see "Time and
memory"):

```
topopt run examples/footbridge/footbridge_short.yaml --out runs/footbridge_short
```

The first line of `topopt check` is a note about a fine pass. In `footbridge.yaml` the member of 2.3 mm is 2.1 voxels across and the
note asks for 3 (0.75 mm voxels, about 138 GiB of memory by the estimate). In `footbridge_short.yaml` the skin of 2.5 mm is 1.2 voxels
across and the note asks for 0.85 mm voxels. Neither YAML uses a fine pass.

`topopt check --json` (and `check_problem` of the MCP server) lists warnings that are expected on this example. The two halves of the
walking surface are small or thin (3 mm, about 1 voxel across), and in `footbridge_short.yaml` the skin too. In both runs below,
`final.stl` covers both halves and the skin in full and joins them to the feet (`mesh.checks` and `load_paths` in
`result/metrics.json`). Most region planes are off the voxel grid, so each region in the voxel model is up to half a voxel larger or
smaller than its STL.

## Time and memory

- Our run of `footbridge.yaml` (2026-10-28, main 5f2a523): 2,315 s (38.6 min) on a shared Linux host with an Intel Core Ultra 9 185H and
  its integrated Arc GPU. `settings.solver` is the default `auto`, which picked `gpu-mg`. Before the run, `topopt check` estimated
  2,096 s (up to 5,744 s) and 12.6 GiB. The peak memory of this run was not measured.
- Without a usable GPU, `auto` takes a CPU solver. On the same host `topopt check` then estimates `pardiso`, about 71,500 s (about
  20 hours; up to about 223,000 s) and 30.1 GiB (up to 46.4 GiB). Without a GPU, start with `footbridge_short.yaml`.
- The grid of `footbridge.yaml`: 1.1 mm voxels, 255 x 40 x 88, 394,348 active voxels, 1,134,816 free degrees of freedom. The count is
  above the default limit of 100,000, so the YAML sets `max_active_voxels: 400000`.
- Our run of `footbridge_short.yaml` (the same day and main, the same host): 91.5 s on `gpu-mg` (`auto`). `topopt check` estimated 149 s
  (up to 407 s) and 4.0 GiB; the peak resident memory of the run process was 0.9 GiB. Without a GPU, `topopt check` estimates `pardiso`,
  about 362 s (up to 1,130 s) and 2.8 GiB (up to 4.3 GiB). The grid: 2.1 mm voxels, 58,728 active voxels, 177,888 free degrees of
  freedom.

## The bridge

`make_footbridge.py` builds all region files. The parameters are at the top of the script; after a change, run it again
(`python examples/footbridge/make_footbridge.py`, or `--out DIR` for another folder). It prints each region with its volume and bounds.

- A span of 280 mm and a width of 36 mm, so that the bridge fits a 350 mm printer bed turned 45 degrees.
- The deck: a solid skin 2.5 mm thick under a walking surface 3 mm thick, with the top of the skin at z = 88. The walking surface is
  split at mid-span into two load regions, so that a load case can load one half.
- Two feet, the full width, 24 mm of the span at each end and 30 mm high from the ground.
- Under the span, a half ellipse 200 mm wide at the ground and 56 mm high at mid-span is kept free: the opening of the bridge.
- The design space is the box from the ground up to 1 mm into the skin (the solid skin wins there). It ends 2.5 mm inside the two side
  faces of the skin, so that the side faces of the deck are the flat faces of the skin.

## Coordinates (mm)

x runs along the span, y across it (the centre line is y = 0), z is up. The ground is z = 0.

| Region | File | Role | Box (x / y / z) |
|---|---|---|---|
| Design space | `design.stl` | design | 0..280 / -15.5..15.5 / 0..86.5 |
| Feet (2) | `feet_a.stl`, `feet_b.stl` | fixed | 0..24 and 256..280 / -18..18 / 0..30 |
| Deck skin | `deck_skin.stl` | solid | 0..280 / -18..18 / 85.5..88 |
| Walking surface (2) | `walk_left.stl`, `walk_right.stl` | load `walk_left`, `walk_right` | 0..140 and 140..280 / -18..18 / 88..91 |
| Clearance | `clearance.stl` | keep_out | 40..240 / -22..22 / -5..56: the half ellipse, closed 5 mm below the ground |

## Load cases

| Case | Load | Weight |
|---|---|---|
| `uniform` | 30 N down on each half of the walking surface (60 N in all) | 1 |
| `half_left` | 60 N down on `walk_left` only | 1 |
| `half_right` | 60 N down on `walk_right` only | 1 |

- `uniform`: people spread along the bridge.
- `half_left`, `half_right`: the same load on one half of the span. They load the span unevenly, which `uniform` alone does not.
- Each force is spread over its load region. All three cases carry 60 N in all and have the weight 1.

## Settings

- `volume_fraction: 0.10` of the free design space. The feet, the skin and the walking surface are solid and come on top: the target is
  146,901 mm3, 182.2 g in this material (`topopt check`, "volume basis").
- `voxel_size: 1.1` and `min_member_mm: 2.3`: the smallest member is about 2.1 voxels across. `footbridge_short.yaml` keeps that ratio
  with 2.1 mm voxels and a member of 4.4 mm, and makes 80 passes instead of 200; all other keys are the same.
- `iterations: 200` and `penal: 4.0` (the default is 3.0). They are method settings: change them as you like.
- `filter_solid_neighbours: ignore`: the density filter does not count the voxels of the feet, the skin and the walking surface as
  neighbours, so the design voxels next to them are not pulled to solid (CONTRACTS.md section 22a). The default is `density_one`.
- The `print` block: a large-bed FDM printer (350 mm bed) and the build direction `-z`, so the deck lies on the bed. The print facts of
  the run (overhang, support volume, the thinnest member) are for this direction.
- Material: the default printed polymer of the tool (E 3500 MPa, nu 0.35, 1.24 g/cm3).

## Result of the runs above

| | `footbridge.yaml` | `footbridge_short.yaml` |
|---|---|---|
| Voxel, passes | 1.1 mm, 200 | 2.1 mm, 80 |
| Run time (`gpu-mg`) | 2,315 s (38.6 min) | 91.5 s |
| The design space and the other regions, all solid | 650.85 g | 674.41 g |
| Mass of `final.stl` | 186.14 g (71.4 % less) | 177.64 g (73.7 % less) |
| `uniform`: stiffness (N/mm) | 6,905 | 5,567 |
| `half_left`: stiffness (N/mm) | 4,020 | 2,757 |
| `half_right`: stiffness (N/mm) | 4,008 | 2,744 |
| Bodies, triangles | 1, 248,500 | 1, 43,772 |
| Thinnest member (1 % of the wall samples are thinner) | 1.15 mm | 2.33 mm |
| Surface with walls under 1.2 mm | 1.2 % | 0.07 % |

- Stiffness is the load per mm of motion at the load (N/mm): a larger value is stiffer. The two grids differ (the skin and the walking
  surface are cut at voxel steps), so the masses before differ, and the two columns do not compare one to one.
- Neither run settled. Both stopped at their pass limit, with the verdict `iteration_limit`. In `footbridge.yaml` the objective still
  changed by 0.000106 (relative) in the last 5 passes, against a threshold of 0.0001; the grayness was 0.061 (threshold 0.1: met) and
  the volume row was met. In `footbridge_short.yaml` the change was 0.0191 (not met) and the grayness 0.084 (met). The numbers above are
  those of the shape at the pass limit. More `iterations` give the optimizer more passes.

## Not modelled

Self weight, fatigue, vibration and the weaker bond between printed layers. This is a model on a printer bed, not a footbridge at full
scale. No stress check ran: the YAML has no stress row, and `topopt verify` did not run on these results.

## Licence

`make_footbridge.py`, the region files that it makes, the two YAML files and this README are our own work, under the licence of this
repository (`LICENSE` at the root).
