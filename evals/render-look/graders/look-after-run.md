---
type: tool_order
before: mcp__plugin_topopt_topopt__run_topopt
after:
  tool: mcp__plugin_topopt_topopt__render_part
  # A render of the result: a call with a run folder, with no argument (the last run), or with final.stl.
  # A YAML render (yaml_path) and a render of region STLs (stl_paths) do not match.
  input_match: '^(?![\s\S]*(?:yaml_path|stl_paths))|final\.stl'
---
