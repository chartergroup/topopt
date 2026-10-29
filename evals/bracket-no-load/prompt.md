---
description: The load is missing. Claude asks for it and does not start a run.
expected_outcome: No run_topopt. The reply asks what the bracket holds (mass or force).
tags: [smoke, mock, ask]
max_turns: 25
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Нужен кронштейн, который прикручивается к стене. Сделай его как можно легче через topopt и запусти расчёт.
