---
description: After the drone-arm example the user asks why the part has steps. A fresh prompt, not a resumed history; the user's message gives the run (a finished run of the drone-arm example, its folder and what the render shows). The case has its own mock of run_result (mocks/topopt/run_result.md, an agent mock with the real numbers of the drone-arm run). The run ran with settings.envelope_fill on, as the YAML of the example sets it, so metrics.mesh.envelope_fill has ran true and the long treads that stay are on member surfaces. The other tools use the suite mocks.
expected_outcome: the reply says that the steps are a known limit of the voxel method in this version, not a fault of the input; that a step is one voxel high and already in the mass and stiffness numbers; that the fill already ran in this run (or does not offer it as a new step), so the treads are on member surfaces; at least two routes with their price, a smooth mark with apply_marks and a finer voxel with fine_from (a whole run), and that a finer voxel can still show steps; and that a better smoothing of member surfaces is planned, with no date, no version and no promise of a release.
tags: [mock, explain, finish]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

I ran the drone-arm example with topopt (the run folder is runs/drone_arm/run, the run is done). On the render of final.stl the arm has long flat terraces along its length, like stairs, and the T-shirt hook that I made before had none. Why does my part have steps? Did I give a bad input? Are the steps in the mass and stiffness numbers that you gave me? What can I do about them?
