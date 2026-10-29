---
# Agent mock: the run is still going for the first calls, then done. The case tests what the agent does while the run goes.
type: agent
---
You are the tool `run_status` of the topopt MCP server. It shows the state of a run. The work folder is /Users/me/topopt-work.

Reply with ONE JSON object and nothing else: no prose, no code fence.

Count the calls of `run_status` that came before this one in this run (zero for the first call).
- The first two calls (zero or one call before): the run goes on. State "running", process_alive true, an iteration well below settings.iterations of its YAML (about a third for the first call, about two thirds for the second), elapsed_s of 10 to 60 s, job.state "running", verdict null, checkpoint null, and a one-line summary of the pass. The shape is the real answer below, without the finished-run keys.
- The third call and later: the run is DONE. Give the real answer below for this run: state "done", the last iteration = settings.iterations of its YAML, job, verdict (iteration_limit, feasible true), checkpoint (complete true) and a one-line summary.
Use the run folder and the YAML of this run (from the earlier calls). If no run was started in this run and the call names no run folder, answer {"state": "none", "summary": "no run yet: call run_topopt first"}.

Real answer of the example, for a done run:
{{file:fixtures/run_status.json}}
