---
type: agent
expect:
  yaml_path: /\.ya?ml$/
  text: string
---
You are the tool `record_brief` of the topopt MCP server. It saves the brief of a problem YAML in a case file beside the YAML. The work folder is /Users/me/topopt-work.

Reply with ONE JSON object and nothing else: no prose, no code fence.

Give an answer of the same shape as the real answer below, for the call: recorded true; case = the YAML path with `.case.json` in place of `.yaml`; yaml_rel = the yaml_path of the call; brief.text and brief.assumptions = the `text` and the `assumptions` of the call; replaced true only if record_brief was called for the same YAML earlier in this run. Keep `facts` and `options` short.

Real answer of the example:
{{file:fixtures/record_brief.json}}
