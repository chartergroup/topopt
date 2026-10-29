---
description: The user gives the load in kg. The spec must have about 49 N (5 kg x 9.81).
expected_outcome: build_regions with a load case of about 49 N down; the reply states 5 kg -> about 49 N; no run.
tags: [smoke, mock, loads]
max_turns: 25
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Сделай настенный кронштейн под горшок с цветком. Горшок с землёй и водой весит 5 кг и стоит на полочке кронштейна сверху. Кронштейн крепится к стене двумя шурупами, вынос от стены около 120 мм, печать из PETG. Пока только подготовь задачу для topopt и покажи мне допущения. Запускать расчёт не надо.
