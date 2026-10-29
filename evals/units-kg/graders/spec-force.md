---
type: llm
focus: mock_calls
---

Look only at the INPUT (the `spec` argument) of the build_regions call or calls. Do not use the tool answers: they are fixed and do not show what Claude sent.

PASS if the spec of at least one build_regions call has a load case whose force has a size from 45 N to 54 N and points down (5 kg x 9.81 = 49 N; 50 N with g = 10 is also correct). More load cases with larger forces (an abuse case) are allowed.
FAIL if there is no build_regions call, or no load case in the spec has a force from 45 N to 54 N (for example 5 N, 0.5 N or 5000 N: a wrong unit).
