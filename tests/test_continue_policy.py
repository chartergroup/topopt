"""E2-P1: the block "a change after a finished run" in the skill `start-here` (Step 6) and its pointer in Step 5 (the step that explains `continue_from`).

The block holds facts and offers from the study E2 (warm start from a finished run, report of 2026-10-23, sections 1, 3, 4 and 5):

- a smaller volume: offer "continue from the last run" (measured at -20 % only);
- one factor on all forces, with no stress, displacement or compliance limit: the design is the same, no run is needed;
- a load direction change of 10 degrees or more, or a requirement change: run fresh;
- after any continuation: report the volume and the compliance of the new run, and that a continuation is not proven equal to a fresh run
  (a compliance is compared with another run only at equal volume: no tool gives a compliance at equal volume).

It has no must, never or always, and no replay-probe gate (the threshold of that gate is not known; it is parked).
The numbers are the numbers of the report. A change of a number here is a change of the study, so the test names each one.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The freedom rule: facts and offers, no habitual caps (the same words as tests/test_levers.py and tests/test_talk_block.py).
HABIT = re.compile(r"\b(must|never|always|conservative)\b", re.I)

START = "**A change after a finished run.**"
END = "The marks are points of emphasis"


def skill_text():
    return (ROOT / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8")


def block():
    text = skill_text()
    assert START in text, "the block is not in the skill"
    return text[text.index(START):text.index(END)]


def lower(text):
    return " ".join(text.split()).lower()


def test_the_block_is_in_step_6_after_who_decides_and_before_the_marks():
    text = skill_text()
    step = text.partition("## Step 6. Refine from feedback")[2].partition("\n## ")[0]
    assert START in step
    assert step.index("Who decides") < step.index(START) < step.index(END)
    assert text.count(START) == 1


def test_the_block_names_the_three_cases_and_the_report_rule():
    low = lower(block())
    # 1. a smaller volume: the offer, the call of Step 5, and the one size that was measured
    for phrase in ("less volume", "offer \"continue from the last run\"", "`run_topopt` with `continue_from`, `more_iterations` and `allow_changed_inputs=true`",
                   "measured at -20 % volume only", "no other size is tested"):
        assert phrase in low, phrase
    # 2. one factor on all forces: the design is the same, no run; the cases that need a run
    for phrase in ("one factor on all the forces", "with no stress, displacement or compliance limit", "the design is the same", "a new run is not needed",
                   "byte-identical `final.stl` on all three parts", "with a stress, displacement or compliance limit, or a different factor for each load case, a run is needed"):
        assert phrase in low, phrase
    # 3. a load direction change of 10 degrees or more, or a requirement change: run fresh, and why
    for phrase in ("turned by 10 degrees or more", "less than 10 degrees is not measured", "a change of a requirement", "run fresh",
                   "keeps the old topology", "the old thin members"):
        assert phrase in low, phrase
    # the cheap route of a key that only the post step reads stays as Step 6 has it
    assert "`print.build_direction`" in low and "keeps its cheaper route" in low and "layer 3" in low
    # 4. the report after any continuation
    for phrase in ("after a continuation, tell the user the volume and the compliance of the new run",
                   "compare a compliance with another run only at equal volume",
                   "a continuation is not proven equal to a fresh run"):
        assert phrase in low, phrase


def test_the_numbers_are_the_numbers_of_the_report():
    low = lower(block())
    numbers = {
        "-20 % volume": "report section 1 item 3 and section 5 item 1: vol, volume_fraction x0.8; no other size tested",
        "44 to 55 % fewer passes": "section 1 item 3: 44-55 % fewer passes (bracket 43.9, ladle 46.9, hook 55.0)",
        "0.94 to 1.00 of the fresh run": "section 1 item 3: P2 compliance at equal volume 0.94-1.00 of the fresh run",
        "10 degrees or more": "section 1 item 9 and section 5 item 3",
        "at 30 degrees and equal volume, the compliance was +7.5 % on the bracket and +17 % on the hook, in its worst load case": "section 1 item 5: bracket +7.5 %, hook +17 % (worst load case), at equal volume",
        "69 % and 80 %": "section 1 item 6: thinnest member of hook and ladle against the fresh run (min member = 2 voxels)",
        "2 voxels": "section 1 item 6",
    }
    for number, source in numbers.items():
        assert number.lower() in low, f"{number} ({source})"
    # no other number in the block: a new number needs a line in the report. "5" is Step 5 and "3" is layer 3 of the lever cards.
    found = set(re.findall(r"\d+(?:\.\d+)?", block()))
    assert found == {"20", "44", "55", "0.94", "1.00", "10", "7.5", "17", "30", "2", "69", "80", "3", "5"}, sorted(found)


def test_the_block_is_a_set_of_facts_and_offers_and_has_no_replay_probe_gate():
    text = block()
    assert not HABIT.search(text), HABIT.search(text).group()
    low = lower(text)
    for word in ("probe", "gate", "replay pass", "threshold", "1.01", "1.04"):
        assert word not in low, word


def test_step_5_points_to_the_block():
    text = skill_text()
    step_5 = text.partition("## Step 5. Watch and report")[2].partition("\n## ")[0]
    assert "To add passes to a run that is done" in step_5
    assert "Step 6 says when a continuation fits" in step_5


def test_the_copy_for_the_other_clients_is_equal():
    source = skill_text()
    copy = (ROOT / ".agents" / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8")
    assert copy == source, "run: python3 scripts/sync_agent_skills.py"
    assert START in copy
