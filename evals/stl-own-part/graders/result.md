---
type: llm
focus: last_message
---

The server is a mock, so the numbers of the build are not known in advance.

PASS if the reply does all of these:
- says that the problem is prepared and the run is not started;
- gives the size of the plate in mm (40 x 20 x 4) or says that it read the file as mm;
- says the thickness of the ring round the screw holes (a number in mm, or "half the hole diameter") or says that the holes keep a ring;
- lists assumptions (the load of 30 N, PETG, the mounting at the small hole).
FAIL if the reply says that a run was started or finished, or does not say anything about the holes of the plate.
