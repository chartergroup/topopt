---
description: environment offers the gpu extra. Claude gives the reason and the measured speed-up and installs nothing without the user's yes.
expected_outcome: environment is called; the reply names gpu-mg and the gpu extra (pyopencl), the reason, the measured figures, and asks before an install.
tags: [smoke, mock, environment]
max_turns: 20
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

Прежде чем что-то считать: насколько быстро topopt будет работать на моём компьютере? Можно ли как-то ускорить расчёты?
