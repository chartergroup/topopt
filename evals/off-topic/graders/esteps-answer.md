---
type: llm
focus: last_message
---

PASS if the reply gives the usual e-steps procedure for Marlin: heat the hotend, mark the filament (for example 120 mm above the extruder), extrude a known length (for example 100 mm with G1 E100 F100 in relative mode M83), measure the length that went in, compute new steps = current steps x requested length / actual length, set it with M92 E<value>, and save with M500 (M503 shows the current value).
FAIL if the reply does not give the formula or the M92 / M500 commands, or sends the user to topology optimization.
