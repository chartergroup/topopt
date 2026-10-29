---
description: The voxel guard stops build_regions at voxel 0.3 mm, which the user asked for. Claude reads the stop and the memory estimate, then raises the guard (a method setting) with the facts and the risk, or offers the larger voxel (a user requirement) with its price and asks first.
expected_outcome: The first build_regions call stops at the guard. The reply gives the stop and the memory estimate against the RAM, then either raises settings.max_grid_voxels and keeps 0.3 mm (it says the RAM, the estimate, the freeze risk, what, why and the price), or offers a larger voxel with its price and asks before it uses it. It does not use a larger voxel without asking.
tags: [smoke, mock, guard]
max_turns: 30
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Сделай настенный крючок для футболки на вешалке: два шурупа в стену, печать из PETG. Мне нужна очень гладкая поверхность, поэтому размер вокселя поставь 0.3 мм, не крупнее. Запускай расчёт.
