---
# Agent mock: the real guard stop at voxel 0.3 mm, with the names of the call.
type: agent
---
You are the tool `build_regions` of the topopt MCP server. The work folder is /Users/me/topopt-work.

Reply with ONE JSON object and nothing else: no prose, no code fence.

The real answer below is the grid guard stop for a T-shirt hook at voxel 0.3 mm. Give it for THIS call:
- name: the `name` argument, else spec.name; run_dir_rel runs/<name>; yaml_rel runs/<name>/<name>.yaml; files_rel: one <region name>.stl for each region of the spec, and the YAML; the paths in `report` and `check_error` with this name; `assumptions` from the spec.
- If the spec voxel_size is below 0.35 mm: keep `problems`, the ERROR line of `report` and `check_error` word for word, with the same numbers (6949800 voxels, the limit 5000000, voxel_size 0.3, "Try voxel_size 0.35 or more"); put the voxel size of the spec in place of 0.3. ok false (a build that did not build is not ok), builder_ok false, check null, derived null.
- If the spec voxel_size is 0.35 mm or more, or the spec raises settings.max_grid_voxels to 7000000 or more: the build works. Give ok true, builder_ok true, problems [], and a short `check` with grid, active_voxels (about 1.9 million at 0.35 mm), memory_estimate (about 9 GiB at 0.35 mm, with the RAM of the machine: 16 GiB) and time_estimate (about 2 hours at 0.35 mm). With the guard raised and the voxel still 0.3 mm, the numbers are those of 0.35 mm times 1.6 (the voxel count grows with the cube of 0.35 / 0.3): about 3.0 million active voxels, about 14 GiB (near the RAM) and about 3 hours.

Real answer:
{{file:fixtures/build_regions.json}}
