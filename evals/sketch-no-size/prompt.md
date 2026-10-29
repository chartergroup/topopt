---
description: A sketch with no dimension. Claude asks for one real size (or a known object for the scale) and does not start a run.
expected_outcome: No run_topopt. The reply asks for a real dimension, or a known object in the picture, for the scale.
tags: [smoke, mock, ask]
max_turns: 25
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Вот эскиз настенного крючка для куртки: файл `files/sketch.png` в рабочей папке. Крепится к стене двумя шурупами, печатаю из PETG. Сделай его как можно легче через topopt и запусти расчёт.
