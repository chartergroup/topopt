---
description: End to end on the REAL topopt server (one optimizer job). Run only with --tag e2e --mocks off --allow-tools "mcp__plugin_topopt_topopt__*" -j 1.
expected_outcome: build_regions, run_topopt, run_status until done, run_result; the reply gives the mass, the stiffness, the time and the path of final.stl.
tags: [e2e]
runs: 1
max_turns: 120
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Хочу напечатать настенный крючок для футболки на вешалке. Крепится к стене двумя шурупами, печатаю из PETG. Сделай его легче через topopt. Это пробный запуск: возьми грубую сетку, чтобы расчёт шёл не больше пары минут. Уточнять ничего не хочу: прими разумные допущения, скажи их, запусти расчёт, дождись конца и расскажи результат.
