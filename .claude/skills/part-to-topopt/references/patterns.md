# Patterns for bowls, handles and interfaces

This file holds three patterns for the spec: bowls, cups and containers; handles and grips; surfaces that a neighbour touches.

## Pattern: bowls, cups and containers

Use this for a ladle, a scoop, a cup holder or a dish. `examples/ladle/ladle_spec.yaml` (in the topopt folder, `${CLAUDE_PLUGIN_ROOT}` in the plugin) is the worked case.

- Put the whole bowl shell in `load` (one `bowl` primitive). Make the wall 3 voxels thick or more: wall 9 mm at 3.25 mm voxels.
- Do not put the bowl in `design`. A big shell in the design space uses the volume budget, and the optimizer then drops the handle.
- Put the cavity in `keep_out` with a name that has no "hole" in it (today the screw-hole check finds holes by name, so a cavity named "hole" would count as one) (for example `gravy_space`). Its radius is the bowl radius minus the wall, minus 1 mm (36 - 9 - 1 = 26 mm).
- Add a second keep-out above the opening (a cylinder with the bowl radius, up from the bowl centre, for example `over_opening`). Then no material grows over the opening.
- Make the design space ONE wide block from the grip to the bowl: 154 x 96 x 80 mm, so z runs from -40 to +40 mm. It is about 2 times wider than the handle needs, with room above and below the grip-to-bowl line. An arch, a fork or a swan neck can form. The block must reach 5 to 10 mm into the bowl wall (x 154, the bowl wall starts at x 146).
- The grip block is 24 x 16 mm wide and 20 mm long, so it has 3 voxels across at 3.25 mm.
- Use a low volume fraction for the block (about 0.1 of the free voxels: the spec has 0.10628). The block is big, and a handle needs only a small part of it.
- Use three load cases: `full_scoop` (gravy weight), `scrape` (pull back and down) and `side_stir` (18 N in y). The side force makes the handle less straight.
- Content weight: volume in litres x density x 9.81. Example: 0.5 kg of gravy gives 4.9 N.
- Measured 2026-10-12: `examples/ladle/ladle_spec.yaml`, 3.25 mm voxels, 36,542 active voxels (34,743 free), 145 s real (the estimate said 160 s), 1,555 g block to 226 g (85 percent saved). The run used `volume_fraction: 0.10628` in the YAML, and the spec has the same value now, so the builder writes the pinned `examples/ladle/ladle.yaml` (same settings and same STL files). The target of that run is 233.8 g (derived: (1,799 forced-solid + 0.10628 x 34,743 free) voxels x 3.25^3 mm3 x 1.24 g/cm3). Before P1 the spec value was 0.06, and the builder converted it to 0.10628. The mesh is one watertight body, 220 x 73 x 54 mm. The handle is a wide, curved bar with two windows.
- The wide block is compact (fill 0.66), so the run is slower than the old narrow brick (97 s at 33,592 voxels). Its model estimate is 160 s.

## Pattern: handles and grips

- Put the hand end in `fixed`: a solid block 20 mm or more long, as thick as the handle. The grip is held by the hand, so it is clamped.
- Make the design space a straight box along the handle, about 20 mm wide and 12 mm thick, from the grip to the tip end.
- Give the abuse case a force of 5 to 10 times the static force, and add a sideways part. A handle then keeps a rib on the loaded side.
- Make sure the fixed block, the neck and the tip end touch or overlap by 1 or 2 mm. Then the helper does not report `separate pieces`.

## Pattern: surfaces that a neighbour touches (interfaces)

Write the `interfaces` block in the spec (a top-level key) when a surface needs an exact size, or must be complete (CONTRACTS.md section 23). The helper copies it into the problem YAML as you wrote it, and the loader checks it. For the `screw_holes` of an `stl` item the helper adds a `tube` for each hole after your entries (see "Your own STL" in `SKILL.md`). Do not add it to the YAML by hand after the build.

- A bond socket, a bearing seat, a press fit or a sealing face gets `full_surface: true`. The measure then gives `full_coverage`, and the verdict has one test for it (0.98 or more is met).
- An end face that trims a part gets `radius_mm`. Without it, a `face` cuts everything on the side of its normal, also a sleeve or a load region that reaches past the plane.
- A socket that needs machining stock is a `tube` interface (ring and hole), so the stock is real. A `bore` with `stock_mm` has an effect only where the mesh has material at the as-printed radius. If the region starts at the finished radius, the stock has no effect, and the run says so.
- Before the run: draw the YAML (`topopt render <yaml>`) and read each `interface cut:` line of `check_problem` (`interface_taken`). After the run: read `interfaces.integrity` (what each frozen region keeps), each `full_surface` test and `interface_warnings` of `run_result`.

```yaml
interfaces:   # a motor bracket: a bearing seat, and a trim of the boss end
  - {name: bearing_seat, kind: bore, point: [0, 0, 30], axis: [0, 0, -1], radius_mm: 11.0, length_mm: 7.0, stock_mm: 0.3, full_surface: true}
  - {name: end_trim, kind: face, point: [0, 0, 30], normal: [0, 0, 1], radius_mm: 14.0}
```
