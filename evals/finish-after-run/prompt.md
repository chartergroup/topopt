---
description: A part from words, run it. After the run Claude does the finishing pass on its own (the look at final.stl with front and fl2, regions off; the print check again) and reports it before the final answer. The user says nothing about a finish. A fresh prompt on the suite mocks, not a resumed history, because the pass belongs to the run flow (start-here, Step 5) and a history made with an older skill text would hold a run with no pass. The mocked run has no print_check in run_result, so the agent calls print_check. The part needs no edit, so the case needs no mock of cleanup_list, apply_marks or ratchet_run.
expected_outcome: run_topopt, then render_part of the result with the views front and fl2 and regions off, then print_check; the reply reports the finishing pass (what it looked at, what it changed or that nothing needed a change, the price of any change in mass and stiffness) and the print check after it, and does not call the part certified.
tags: [smoke, mock, flow, finish]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Нужен настенный крючок для футболки на вешалке. Крепится к стене двумя шурупами, печать из PETG. Сделай его как можно легче через topopt и дай мне STL с итогами расчёта. Уточнять ничего не хочу: прими разумные допущения, скажи их и запускай.
