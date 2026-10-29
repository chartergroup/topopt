---
type: llm
focus: last_message
---

Claude ran the print check of the finished hook. The check is geometry only: closed mesh, bodies, closed voids, bed fit, overhang, supports, plate contact, wall thickness. It is not a strength check.

PASS if the reply gives facts of the print check (at least two of: closed mesh, one body, no closed voids, bed fit, overhang, supports, plate contact, wall thickness; a check with no limit can be called unknown) AND says what the check does not cover (for example the strength or the stress, or that the check is geometry only).
FAIL if the reply gives no facts of a print check, or says the part is strong enough or certified because of this check, or does not say that the strength is not checked.
The language and the format do not matter.
