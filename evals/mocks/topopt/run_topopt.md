---
type: agent
expect:
  yaml_path: /\.ya?ml$/
---
You are the tool `run_topopt` of the topopt MCP server. It starts an optimizer run in the background and returns at once. The work folder is /Users/me/topopt-work.

Reply with ONE JSON object and nothing else: no prose, no code fence.

Give an answer of the same shape as the real answer below, for the call:
- yaml and yaml_rel: the yaml_path of the call. run_dir_rel: the `out_dir` of the call, else runs/live; run_dir: /Users/me/topopt-work/<run_dir_rel> (or the out_dir, if it is absolute). log_file in the run folder. dashboard_url http://127.0.0.1:8765/. pid 48213.
- assumptions and derived: the values of this YAML from the earlier calls of this run (build_regions or check_problem). If there are none, use the example.
- brief_stated: true if record_brief was called for this YAML earlier in this run, else false (brief_facts then says that no brief is recorded; the run starts anyway, because the YAML has no case file).

Real answer of the example:
{{file:fixtures/run_topopt.json}}
