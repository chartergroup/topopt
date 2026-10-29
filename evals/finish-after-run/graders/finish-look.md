---
type: tool_order
before: mcp__plugin_topopt_topopt__run_topopt
after:
  tool: mcp__plugin_topopt_topopt__render_part
  # The look of the finishing pass (role finish, "What you get"): the views front and fl2:-1.333,-1,0.373, regions off.
  # A YAML render before the run does not match: the pass comes after the run.
  input_match: '^(?=[\s\S]*fl2:-1\.333,-1,0\.373)(?=[\s\S]*"regions"\s*:\s*"off")'
---
