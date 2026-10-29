"""P9-17 (D15): the two skills offer the stress check once after the first finished run: what it answers, what it costs, and that a
yes or no needs an allowable. It is an offer, not a rule. The lines of the check are in `tests/test_verify_plain.py`.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PROMISES = ("safe", "will not break", "guaranteed", "guarantee", "certain", "will hold", "won't break", "strong enough",
            "no risk")

PART_SKILL = REPO / ".claude" / "skills" / "part-to-topopt" / "SKILL.md"
START_SKILL = REPO / "skills" / "start-here" / "SKILL.md"
COPIES = {PART_SKILL: REPO / ".agents" / "skills" / "part-to-topopt" / "SKILL.md",
          START_SKILL: REPO / ".agents" / "skills" / "start-here" / "SKILL.md"}


def offer_paragraph(path: Path) -> str:
    """The text of the offer: the section 'After the first finished run' (part-to-topopt) or the paragraph that starts so (start-here)."""
    body = path.read_text()
    start = body.index("After the first finished run")
    end = body.index("\n## " if body[start - 3:start] == "## " else "\n\n", start)
    return body[start:end]


@pytest.mark.parametrize("source", [PART_SKILL, START_SKILL])
def test_the_two_copies_of_a_skill_are_equal(source):
    assert source.read_bytes() == COPIES[source].read_bytes()


@pytest.mark.parametrize("source", [PART_SKILL, START_SKILL])
def test_the_skill_offers_the_stress_check_once_with_its_cost_and_the_allowable(source):
    para = offer_paragraph(source)
    for words in ("offer", "once", '"will it break under these loads?"', "verify_run", "estimate_only", "extra `verify`",
                  "allowable", "allowable_mpa", "verify_status", "`plain`", "one plain line for each load case"):
        assert words in para, (source.name, words)
    assert "The user decides" in para or "the user decides" in para


@pytest.mark.parametrize("source", [PART_SKILL, START_SKILL])
def test_the_offer_is_not_a_rule(source):
    low = offer_paragraph(source).lower()
    for word in ("must", "never", "always", "required"):
        assert not re.search(rf"\b{word}\b", low), (source.name, word)
    for word in PROMISES:
        assert word not in low, (source.name, word)


def test_the_offer_of_the_part_skill_is_its_own_section():
    body = PART_SKILL.read_text()
    assert "\n## After the first finished run: offer the stress check\n" in body
    assert body.index("## After the first finished run") > body.index("## Step 5. Print the summary")
