---
description: The user has an STL of the part itself (a plate with two holes). Claude reads it with find_features, then builds the regions with an stl item that has screw_holes, and does not start the run.
expected_outcome: find_features before build_regions; the spec has an stl item of files/bracket.stl with screw_holes, a fixed ring at the small hole and a load of about 30 N along +x; the reply gives the size in mm, the ring thickness and the assumptions; no run_topopt.
tags: [mock, flow, files]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, TodoWrite]
---

I have an STL of my part: `files/bracket.stl` in the work folder (a plate, in mm). The small hole takes a screw that holds the plate to a wall. A hook hangs in the big hole and pulls the plate with 30 N along +x, away from the wall. I want the plate lighter, in PETG. Only prepare the problem for topopt and show me your assumptions. Do not run the calculation yet.
