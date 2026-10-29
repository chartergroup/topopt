---
description: A part from words, and the look. Claude draws the regions before the first run and the part after it with render_part, with no request for a picture, and says what it sees (mocked server).
expected_outcome: render_part with yaml_path before run_topopt; render_part of the run folder after run_topopt; the reply says in a line or two what the render shows, and gives the mass, the stiffness and final.stl.
tags: [smoke, mock, flow, render]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Нужен настенный крючок для футболки на вешалке. Крепится к стене двумя шурупами, печать из PETG. Сделай его как можно легче через topopt и дай мне STL с итогами расчёта. Уточнять ничего не хочу: прими разумные допущения, скажи их и запускай.
