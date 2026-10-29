---
type: llm
focus: last_message
---

The user sent a sketch of a wall hook. The sketch has the proportions of the part and no number, no dimension line and no known object (no coin, no standard screw). The user named the load (a jacket), the mounting (two screws) and the material (PETG), and asked for a run.

PASS if the reply asks the user for a real size for the scale: one dimension of the part (for example the length of the arm, the width of the plate or the height of the lip), or a known object that the user can add to the picture (a coin, a standard screw, a ruler). Other questions (the weight of the jacket, the print bed, the screws) can also be there.
FAIL if the reply says that a run started or finished, or shows a result of a run, or takes a size from the sketch with no question about it (for example "I took 100 mm for the arm" and goes on).
The language and the format do not matter.
