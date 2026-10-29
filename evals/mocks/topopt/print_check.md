---
# Agent mock: the real print check of the short hook, with the paths and the size facts of
# the run of the conversation (a check of another part makes Claude stop).
type: agent
---
You are the tool `print_check` of the topopt MCP server. It checks a finished part for printing (geometry only). The work folder is /Users/me/topopt-work.

Reply with ONE JSON object and nothing else: no prose, no code fence.

Give the real answer below with the same keys, the same checks and the same states: mesh_closed, bodies, closed_voids and bed_fit pass; overhang, support_volume, plate_contact and wall_thickness are unknown because no limit and no build direction is given (keep their reasons). Change only:
- the paths: `input.path` is the `path` of the call (a run folder gets /result/final.stl added; a relative path starts at the work folder). With no path, use the run that run_topopt started earlier in this run.
- the size facts, when an earlier call of this run gives them: the volume and the triangles of run_result (volume_final_mm3, triangles), and a bounding box that fits the regions of the part. Without such a call, keep the numbers of the example.
- If the call gives limits (print_settings or a print block), give pass or fail for those checks with plausible values.

Real answer:
{{file:fixtures/print_check.json}}
