# 5-inch drone arm (example)

One arm of a 5-inch quadcopter frame. The optimizer keeps the footprint of a plain 6 mm plate arm, may use 6 mm more depth below and
above its plate, and looks for the stiffest printed shape at the mass of that plate arm. The arm is our own: `make_arm.py` makes the
region files from named round dimensions that are typical for a 5-inch frame (see "The arm").

![The finished drone arm: a tapered plate with two large cut-outs, a clamp end and a round motor end.](result.png)

The picture is the `final.stl` of the run in "Time and memory" below. All examples: [examples/README.md](../README.md).

```
topopt check examples/drone_arm/drone_arm.yaml
topopt run examples/drone_arm/drone_arm.yaml --out runs/drone_arm
```

`topopt check` prints two warnings on this example, and both are expected: the motor shaft hole (a keep-out region) cuts up to 3.75 mm
into the motor pad (the load region), and with `eta_eroded: 0.7` the eroded design keeps walls of about 3 mm, thicker than the 2 mm of
`min_member_mm`.

## Time and memory

- Our run (2026-10-27, main 54e4c8f): 970 s (16 min) on a shared Linux host with an Intel Core Ultra 9 185H and its integrated Arc GPU.
  `settings.solver` is the default `auto`, which picked `gpu-mg`. The peak RAM was 10.6 GiB. Before the run, `topopt check` estimated
  571 s (up to 1,564 s) and 14.3 GiB.
- Without a usable GPU, `auto` takes a CPU solver. On the same host `topopt check` then estimates `pardiso`, about 10,700 s (up to about
  33,500 s) and 17.0 GiB (up to 26.1 GiB).
- On a machine with 16 GB of RAM, the GPU run is near the limit (estimate 14.3 GiB, measured peak 10.6 GiB), and without a GPU the
  estimate (17.0 GiB, about 3 hours) is above it. Run `topopt check` first. Here the filter matrix sets the peak, so a larger `voxel_size`
  needs much less memory.
- The grid: 0.5 mm voxels, 159,060 active voxels, 513,573 free degrees of freedom. The count is above the default limit of 100,000, so
  the YAML sets `max_active_voxels: 200000`.

## The arm

`make_arm.py` builds all region files and `arm_original.stl`. The parameters are at the top of the script; after a change, run it again
(`python examples/drone_arm/make_arm.py`, or `--out DIR` for another folder). It prints each region and the mass of the plain plate arm,
which is `target_mass_g` of `drone_arm.yaml`.

- A 226 mm motor-to-motor diagonal: the motor centre is at x = 113.
- The body: a half-circle root end 10 mm wide that starts at x = 6, a straight taper to 5 mm at the motor centre, and a round motor end
  of radius 16.5 mm. The inner corners where the body meets the motor end are rounded with a radius of 6 mm.
- Two M3 clamp bolts at x = 14 and 24, under a 40 mm square body stack turned 45 degrees (a true X frame).
- Four M3 motor slots on the diagonals, from the 16 x 16 mm pattern out to the 19 x 19 mm pattern, so that both motor sizes fit. Below
  the plate, 1.5 mm of room around each slot for the screw head and the driver.
- A shaft hole of radius 4 mm, a solid motor pad of radius 14 mm (z = 2..6) and the turning motor bell plus 1 mm (radius 15 mm) kept
  free above the pad.

## Coordinates (mm)

The frame centre is at the origin. x goes out along the arm to the motor, z is up. The plate of the plain arm is z = 0..6.

| Region | File | Role | Box (x / y / z) |
|---|---|---|---|
| Design space | `design.stl` | design | 6..129.5 / -16.5..16.5 / -6..12 |
| Frame clamp | `frame_clamp.stl` | fixed | 6..28.3 / -5..5 / 0..6 |
| Plate stack | `plate_stack.stl` | keep_out | -28.3..28.3 / -28.3..28.3 / -6.5..0 and 6..12.5 |
| Root bolt holes (2) | `root_bolt_hole_a.stl`, `root_bolt_hole_b.stl` | keep_out | 3.2 mm at x = 14 and 24, through |
| Motor bolt slots (4) | `motor_bolt_hole_1.stl` .. `motor_bolt_hole_4.stl` | keep_out | M3 slots, with head and driver room below |
| Motor shaft | `motor_shaft.stl` | keep_out | r 4, through |
| Motor bell | `motor_bell.stl` | keep_out | r 15, z 6..12.5 |
| Motor pad | `motor_pad.stl` | load `motor_mount` | r 14 at x = 113, z 2..6 |

- The design space is the outline of the plain arm, with the depth z = -6..12. The footprint does not grow, so the arm does not block
  more of the air from the propeller. The propeller is about 20 mm above the top of the plate.
- The clamp is the area of the arm under the body stack, 6 mm thick. The plate stack keeps material out of the space of the body plates
  above and below it.
- `arm_original.stl` is the plain plate arm: the outline with all holes, z = 0..6. `drone_arm_original.yaml` uses it as the comparison.

## Load cases

| Case | Load on `motor_mount` | Weight |
|---|---|---|
| `thrust` | 20 N up (+z) and a torque of 200 N mm about z | 9 |
| `side_hit` | 60 N in +y at (113, 0, 20) | 1 |
| `front_hit` | 60 N in -x at (113, 0, 20) | 1 |

- `thrust`: about 2 kgf, the peak static thrust of a 2207-class motor with a 5-inch propeller at 6S, and the reaction torque of the motor.
  The weight 9 makes the vertical stiffness count like a 60 N vertical hit (it also covers a hard landing).
- `side_hit`, `front_hit`: a crash on the motor bell, 14 mm above the top of the pad. The offset adds twist and bending.

## Settings

- `target_mass_g: 10.801`: the mass of the plain plate arm in the same material (1,452 mm2 x 6 mm = 8,710 mm3 at 1.24 g/cm3). The
  optimizer aims at that mass, so the two shapes have about the same mass (the result table gives both).
- `min_member_mm: 2.0`.
- Method settings for walls that print: `iterations: 200`, a `beta_schedule` up to 32, `robust: eroded` with `eta_eroded: 0.7`. On the
  arm of an earlier version of this example (the same size and loads) at 0.5 mm, with the post step of an earlier main (no fill), they
  reduced the surface with walls under 1.2 mm from 13.5 % (80 passes and the default settings) to 0.42 %, at about the same stiffness and
  mass, for about 4 times the run time (measured on 2026-10-26). They are method settings: change them as you like.
- `envelope_fill: on`: the post step grows the result to the faces of the design space before the smoothing, so that the side faces of the
  arm are flat in `final.stl` and show no voxel steps (CONTRACTS.md section 19). The default is `off`. On this arm the growth is 1.4 % of
  the volume target, below the 5 % cap of `auto`, so `auto` runs the step too; the YAML sets `on` so that the numbers below do not depend
  on that choice. The price, from a post-only resume of the same run with `envelope_fill: off`: with the fill, the compliance of the three
  load cases is 0.4 to 0.5 % higher (without it: 21.8, 572.5 and 63.8 N mm; that voxel model is 0.4 % heavier, 10.84 g against 10.80 g),
  and the area of walls under 1.2 mm is 1.88 % instead of 1.21 %.
- Material: the default printed polymer of the tool (E 3500 MPa, nu 0.35, 1.24 g/cm3).

## Result of the run above

| | Plain 6 mm arm | Optimized arm | Stiffness ratio |
|---|---|---|---|
| Mass of the STL | 10.80 g | 10.79 g | |
| Mass of the voxel model of the comparison | 10.79 g | 10.80 g | |
| `thrust`: compliance (N mm) | 170.7 | 22.0 | 7.8 x |
| `side_hit`: compliance (N mm) | 1,103.4 | 574.8 | 1.9 x |
| `front_hit`: compliance (N mm) | 172.8 | 64.1 | 2.7 x |

- Compliance is the work of the loads (N mm): a lower value is a stiffer part. The stiffness ratio is the plain value divided by the
  optimized one.
- Both shapes are in the same printed polymer and in the same voxel model, and the two voxel models have the same mass within 0.2 % (the
  optimized one is 0.15 % heavier). The arms of real 5-inch frames are mostly carbon fibre, so the numbers compare two shapes of about the
  same mass in one material. They do not say
  that a printed arm is stiffer than a carbon arm.
- The run: 200 passes, grayness 0.035 at the end (beta 32). One body, a closed mesh, no closed voids. The design space and the clamp,
  filled solid, would be 24.7 g (56.2 % less mass).
- Walls: 1.88 % of the surface is thinner than 1.2 mm; 1 % of the wall samples are thinner than 0.98 mm. The YAML sets no
  `print.min_wall_mm`, so the print check of the run says `unknown` for the walls. With `topopt printcheck runs/drone_arm --set
  min_wall_mm=1.2` the wall check fails its limit of 0.5 % of the area.

## How the numbers were measured

1. `topopt run examples/drone_arm/drone_arm_original.yaml --out runs/drone_arm_original`: the plain plate arm, all voxels solid. Pass 0
   in `history.jsonl` has the compliance of each load case (`compliance_cases`, in the order of the load cases).
2. The optimized arm: save a copy of `drone_arm_original.yaml` in `examples/drone_arm/` (the region files are found next to the YAML), set
   its `regions.design` to the absolute path of the `result/final.stl` of the `drone_arm.yaml` run, and run it in the same way.
   `result/metrics.json` of each run gives the mass of its voxel model (`mass_design_space_g`).
3. Both use the same 0.5 mm voxels, clamp, keep-outs, loads and material. `compliance_final` of the run itself is the value of the density
   field, not of the meshed part, so the table does not use it. `verify_run` (quadratic tets) cannot model the torque and the offset forces
   of these load cases yet, so the numbers have no independent check.

## Not modelled

The motor wire channel, fatigue and vibration modes, the print orientation (the layer direction) and the real energy of a crash.

## Licence

`make_arm.py`, the region files that it makes, the two YAML files and this README are our own work, under the licence of this repository
(`LICENSE` at the root). No outline, curve or point comes from the drawing of another frame.
