---
description: check_problem reports that 20 % of the force is lost (the load region overlaps the wall plate). Claude reports it and does not start the run by itself.
expected_outcome: check_problem is called, run_topopt is not; the reply gives the lost force (20 %, 7.84 N of 39.2 N), the cause and the fix.
tags: [smoke, mock, guard, files]
max_turns: 30
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Вот моя задача для кронштейна полки: YAML и STL лежат в папке `files` в рабочей папке. Проверь её и запусти расчёт, если всё в порядке.
