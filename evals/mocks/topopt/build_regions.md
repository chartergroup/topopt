---
# Agent mock: the answer must echo the spec that Claude sends (a fixed answer for another
# spec makes Claude stop, correctly). The real answer below gives the shape and the style.
type: agent
---
You are the tool `build_regions` of the topopt MCP server. It writes region STL files and a problem YAML from a part spec, and checks them. The work folder is /Users/me/topopt-work.

Reply with ONE JSON object and nothing else: no prose, no code fence.

The real answer at the end is for one spec (a T-shirt hook, voxel 2 mm, 40 passes). Give an answer of the same shape for the spec of THIS call:
- name: the `name` argument, else spec.name. run_dir_rel runs/<name>, yaml_rel runs/<name>/<name>.yaml, files_rel: one <region name>.stl for each region of the spec, and the YAML.
- Copy from the spec: the load cases and their forces, the material (if the spec has none: E 2400, nu 0.35, density 1.24), the settings and the assumptions. In `derived.load_cases`, `magnitude_n` is the length of the force vector from the spec, in N. Do not change a force value.
- voxel_size: the spec value; for "auto", a value from 1.0 to 2.0 mm (voxel_auto true).
- check: grid shape = the size of the bounding box of all regions divided by the voxel size; active voxels about 30 to 50 percent of the grid; free_dofs about 3 x active voxels; time_estimate about 0.1 to 1 s per iteration times the iterations; memory well below 16 GB; lost_force 0 when no load region overlaps a fixed region.
- Keep the numbers consistent: the voxel count of each region is about its volume / voxel_size^3, and active voxels are the design space plus the fixed, load and solid regions, minus the keep-out regions.
- The engine guards (defaults; the spec settings can raise them): grid voxels at most 5000000 (max_grid_voxels), active voxels at most 100000 (max_active_voxels). With "auto" the voxel is chosen so that the active voxels stay near 36000 to 41000, so no guard fires. With a fixed voxel_size that breaks a guard, the build stops: ok false (a build that did not build is not ok), builder_ok false, check null, derived null, and `problems`, `check_error` and the ERROR line of `report` hold the message, word for word in this form: "model has <N> active voxels, the limit is 100000 (max_active_voxels). Try voxel_size <V> or more, or raise settings.max_active_voxels in the YAML if this machine has the memory" (or, for the grid: "grid would have <N> voxels at voxel_size <h>, the limit is 5000000 (max_grid_voxels). Try voxel_size <V> or more, or raise settings.max_grid_voxels in the YAML if this machine has the memory"). <V> is the voxel that makes N fit, rounded up to 0.05 mm.
- Else ok true, builder_ok true, problems [], warnings [], unless the spec has a clear error (for example a load case names a region that does not exist): then ok false and `error` with one line.
- You may shorten the long texts and long lists of the example. Keep its keys.
- If the spec is the same as the spec of the example, give the example as it is.

Real answer of the example:
{{file:fixtures/build_regions.json}}
