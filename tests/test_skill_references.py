"""The part skill keeps its patterns and its blind test in `references/` and tells when to read them (#207)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".claude" / "skills" / "part-to-topopt"
COPY = ROOT / ".agents" / "skills" / "part-to-topopt"

# reference file -> headings that it holds
MOVED = {
    "references/patterns.md": [
        "## Pattern: bowls, cups and containers",
        "## Pattern: handles and grips",
        "## Pattern: surfaces that a neighbour touches (interfaces)",
    ],
    "references/blind-test.md": ["## Blind test (2026-10-12)"],
}
# reference file -> words that its trigger in SKILL.md must name
TRIGGERS = {
    "references/patterns.md": ["bowl", "cup", "scoop", "ladle", "dish", "container", "handle", "grip",
                               "socket", "seat", "press fit", "sealing face", "end trim", "bore", "boss"],
    "references/blind-test.md": ["blind test"],
}


@pytest.mark.parametrize("folder", [SOURCE, COPY], ids=["source", "copy"])
def test_reference_files_hold_the_moved_headings(folder):
    skill = (folder / "SKILL.md").read_text(encoding="utf-8")
    for rel, headings in MOVED.items():
        ref = folder / rel
        assert ref.is_file(), rel
        text = ref.read_text(encoding="utf-8")
        lines = text.split("\n")
        assert lines[0].startswith("# ") and not lines[0].startswith("## ")      # one title line
        assert lines[1] == "" and lines[2] and not lines[2].startswith("#")      # then one sentence
        for heading in headings:
            assert heading in text.split("\n"), (rel, heading)
            assert heading not in skill.split("\n"), f"{heading} is back in SKILL.md"


@pytest.mark.parametrize("folder", [SOURCE, COPY], ids=["source", "copy"])
def test_skill_names_each_reference_with_its_trigger(folder):
    skill = (folder / "SKILL.md").read_text(encoding="utf-8")
    for rel, words in TRIGGERS.items():
        pointers = [l for l in skill.split("\n") if f"`{rel}` in the folder of this skill" in l]
        assert len(pointers) == 1, rel
        for word in words:
            assert word in pointers[0].lower(), (rel, word)


def test_moved_text_is_not_left_in_skill():
    skill = (SOURCE / "SKILL.md").read_text(encoding="utf-8")
    for marker in ("full_scoop", "bearing_seat", "Put the hand end in `fixed`", "ladle_blind_spec.yaml` (now in"):
        assert marker not in skill, marker


def test_agents_copy_has_the_same_reference_files():
    for rel in MOVED:
        assert (COPY / rel).read_bytes() == (SOURCE / rel).read_bytes(), rel


def test_start_here_step_5_points_to_the_checklist_without_its_items():
    text = (ROOT / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8")
    line = next(l for l in text.split("\n") if l.startswith("Before you report, answer the final review checklist"))
    assert "`team_guide` with `lead-design`" in line
    assert not re.search(r"load inventory|buckling|fatigue", line)
