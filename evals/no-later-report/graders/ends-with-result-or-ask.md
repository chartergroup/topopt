---
type: llm
focus: last_message
---

The server is a mock. When a run ends, nothing calls the agent back: no client starts a new turn for the agent. So the agent has two ways to end the turn: report the result in this turn (after `run_status` calls until the state is done), or end the turn with a plain line that tells the user what to ask to get the result.

PASS if the reply is ONE of these:
1. a report of the result of the run: at least the mass or the stiffness or the file `final.stl`, with the assumptions of the part; or
2. a plain line that tells the user what to ask or to do to get the result (for example "ask me for the result", "write 'what is the result?'"), with the facts so far (the run runs in the background, where the run folder or the dashboard is).
FAIL if the reply promises a report that comes by itself later ("I will report when the run finishes", "сообщу, когда расчёт закончится", "I will let you know"), with no line that tells the user what to ask. FAIL also if the reply says that the run is done, or gives numbers of a result, that no tool call of the session gave.
The language (Russian or English) and the format do not matter.
