---
type: agent
---
You are the tool `run_result` of the topopt MCP server. It gives the final metrics and the path of final.stl of a finished run. The work folder is /Users/me/topopt-work.

Reply with ONE JSON object and nothing else: no prose, no code fence.

Give an answer of the same shape as the real answer below, for the run of this conversation (the run folder of the call, else the run that run_topopt started earlier in this run):
- done true; final_stl /Users/me/topopt-work/<run_dir_rel>/result/final.stl and final_stl_rel; run_dir, run_dir_rel.
- metrics: mass_design_space_g, mass_final_g (about volume_fraction x the design mass, a little more), mass_saved_pct, volume_final_mm3, compliance_initial and compliance_final, stiffness_cases with one entry for each load case of the YAML (its name, N_per_mm from 100 to 5000), run_time_s (the elapsed_s of run_status), iterations, solver "cholesky", watertight true, triangles 15000 to 40000. Use the YAML of this run (load case names, material, settings) from the earlier calls.
- lost_force: the lost_force of the check of this YAML.
- You may leave out the long blocks of the example (filter, constraints, rows, mesh, robust, print, layers) and make `preview` short.
- If the run is the run of the example (the T-shirt hook YAML of the example), give the example as it is.

Real answer of the example:
{{file:fixtures/run_result.json}}
