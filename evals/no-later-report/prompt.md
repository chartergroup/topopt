---
description: A part from words, run it. The user asks for the result and says nothing about waiting or about a later report. The agent starts the run, and nothing calls the agent back when the run ends (#440), so the reply must not promise a report that comes later. A fresh prompt. The case has its own mock of run_status (mocks/topopt/run_status.md, an agent mock), which answers running for the first two calls and done from the third, so the agent has to decide between calling run_status again in the same turn and ending the turn with a plain line that tells the user what to ask.
expected_outcome: run_topopt, then either run_status calls in the same turn until the state is done and a report of the result, or an end of the turn with a plain line that tells the user what to ask to get the result (for example "what is the result?"). No promise of a later report ("I will report when the run finishes").
tags: [smoke, mock, flow, run]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Нужен настенный крючок для футболки на вешалке. Крепится к стене двумя шурупами, печать из PETG. Сделай его как можно легче через topopt, запусти расчёт и сообщи, что получилось. Уточнять ничего не хочу: прими разумные допущения, скажи их и запускай.
