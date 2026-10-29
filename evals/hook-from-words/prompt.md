---
description: A part from words. Claude builds the regions, states the assumptions, runs the job and reports the result (mocked server).
expected_outcome: build_regions, then run_topopt, run_status, run_result; the reply gives the mass, the stiffness, the time and the path of final.stl.
tags: [smoke, mock, flow]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Хочу напечатать настенный крючок для футболки на вешалке. Крепится к стене двумя шурупами, печатаю из PETG. Сделай его как можно легче через topopt и дай мне STL. Уточнять ничего не хочу: прими разумные допущения, скажи их и запускай.
