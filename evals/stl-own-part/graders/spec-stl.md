---
type: llm
focus: mock_calls
---

Look only at the INPUT (the `spec` argument) of the build_regions call or calls. Do not use the tool answers: they are fixed and do not show what Claude sent.

PASS if the spec of at least one build_regions call has all of these:
- an `stl` item in the design region with a `path` that ends in `bracket.stl` and a `screw_holes` key that names H1 (the small hole, 4 mm) or says `all`;
- a fixed region at the small hole H1: a `screw_ring` item for H1, or another region that is clearly the ring or the wall round H1;
- a load case with a force of about 30 N along +x (from 27 N to 33 N), on a region at the big hole H2.
FAIL if there is no build_regions call, or the part is drawn by hand with boxes and cylinders instead of the `stl` item, or `screw_holes` names only the big hole, or the force is not about 30 N along +x.
