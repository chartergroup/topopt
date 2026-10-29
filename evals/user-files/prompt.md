---
description: The user has a YAML and STL files. Claude checks them with check_problem before run_topopt and does not build new regions.
expected_outcome: check_problem before run_topopt; no build_regions; the reply gives the result.
tags: [mock, flow, files]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

У меня уже есть готовая задача для topopt: YAML и STL-файлы крючка лежат в папке `files` в рабочей папке. Проверь задачу, запусти расчёт и расскажи, что получилось.
