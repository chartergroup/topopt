"""P9-3: the block "how to talk to the user" in the server instructions, the skill `start-here` and the lead role.

The block holds offers, not rules: three ways for a request with no numbers (the load question stays in every way), an early
word on whether topology optimization helps, `environment` before the first run with the device and the time as an estimate,
the work folder, the user's language and plain words, and after a run what was checked and what was not.

P9-4a adds one clause to the first reply: the user can pick another solver or device (`settings.solver`, `settings.gpu_device`). The clause
is an offer. The key names and the values come from the code (`SOLVER_VALUES`, the options of `topopt run` and `topopt check`, the arguments of
`run_topopt`), so a change of the code shows here.
"""
import asyncio
import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("mcp")

from topopt.cli import build_parser  # noqa: E402
from topopt.config import SOLVER_VALUES, Settings  # noqa: E402
from topopt_mcp import server  # noqa: E402
from test_plugin_manifest import MCP_TEXT_LIMIT, OLD_CLIENT_CUT  # noqa: E402

# Words that a product text of this project does not use for the block: the freedom rule (an offer, not a rule), and the
# word "conservative" for a time estimate (the measured runs took longer than the estimate; E1, 2026-10-23).
HABIT = re.compile(r"\b(must|never|always|conservative)\b", re.I)

# The same offer in the three texts. One phrase list: a change of the offer is a change in all three.
OFFER = ["offer three ways", "the 2 or 3 questions that change the part most", "a first design from stated guesses",
         "ask the user to confirm the guessed load before the run", "just run it", "Say early whether and how topology optimization helps"]


def lower(text):
    return " ".join(text.split()).lower()


def instructions_block():
    """The block as the instructions have it: the offer, the assumptions, the first-run line, and the paragraph for after a run."""
    text = server.INSTRUCTIONS
    before_run = text[text.index("Offer three ways"):text.index("Tools give facts")]
    after_run = text[text.index("After a run say"):text.index("Descriptions say when to use")]
    return before_run + after_run


def skill_text():
    return (ROOT / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8")


def skill_block():
    text = skill_text()
    return text[text.index("## Talking to the user"):text.index("## Facts to tell the user")]


def role_text():
    return (ROOT / "topopt_mcp" / "roles" / "lead-design.md").read_text(encoding="utf-8")


def skill_step_1():
    text = skill_text()
    return text[text.index("## Step 1. The machine"):text.index("## Step 1b")]


def guard_starts_before_the_cut():
    """The whole instructions text is at or under the cut of the old clients (#380; it was the guard sentence only until then, P9-4a)."""
    assert len(server.INSTRUCTIONS) <= OLD_CLIENT_CUT <= MCP_TEXT_LIMIT


# --- the server instructions ---------------------------------------------------------------------------------------

def test_the_instructions_offer_three_ways_and_keep_the_load_question_in_every_way():
    text = server.INSTRUCTIONS
    for phrase in ("Offer three ways for a request with no numbers", "all the questions", "the 2 or 3 that change the part most",
                   "a first design from stated guesses", "Every way keeps the load question", "in the third, ask the user to confirm the guessed load before the run",
                   '"just run it" skips the offer, not the load.'):
        assert phrase in text, phrase
    assert text.index("Every way keeps the load question") < text.index("in the third, ask the user to confirm")
    # the rule of 2026-10-23 14:15Z stays as it was: a missing load is asked for before the run, also for "run it"
    assert 'If an answer changes the part (load, mounting, size), ask before the run, also when the user says "run it".' in text
    assert "Small details: choose a sensible value, say so." in text        # its pair: a detail that does not change the part (#380 review F2)
    # #380: "Say early whether and how topology optimization helps" left the instructions; the skill has it (tests/test_plugin_manifest.py, MOVED_RULES)
    assert "Say early whether and how topology optimization helps" in skill_block()


CHOICE_CLAUSE = "the choice of another solver or device"


def test_the_instructions_name_environment_before_the_first_run_with_the_device_and_the_time_as_an_estimate():
    text = server.INSTRUCTIONS
    # "then run_status": true before and after the live estimate of run_status (E1-P2), so the text does not wait for it
    sentence = ("Before the first run call environment; tell the user the device, the time estimate "
                "(check_problem, then run_status) and " + CHOICE_CLAUSE + ".")
    assert sentence in text
    assert text.index("One job at a time") < text.index(sentence) < text.index("Tools give facts")      # after the flow, before the guard paragraph
    assert "Name the work folder in the first reply." in text
    assert "Use the user's language, plain words first, numbers on request." in text


def test_the_instructions_say_that_the_user_can_pick_another_solver_or_device_and_how():
    text = server.INSTRUCTIONS
    # the offer is a clause of the first-reply sentence, before the guard paragraph; since #380 the key names and the values are in the description of check_problem
    assert text.index(CHOICE_CLAUSE) < text.index("Tools give facts")
    guard_starts_before_the_cut()
    description = next(t.description for t in asyncio.run(server.mcp.list_tools()) if t.name == "check_problem")
    detail = description[description.index("Another solver:"):]
    assert "settings.solver" in detail and "settings.gpu_device" in detail
    for value in SOLVER_VALUES:
        assert value in detail, value
    assert "--solver of topopt run and check, not of run_topopt" in detail
    assert "For gpu-mg, settings.gpu_device (auto, a number from topopt check, or a part of the name)." in detail
    assert not HABIT.search(CHOICE_CLAUSE + detail), HABIT.search(CHOICE_CLAUSE + detail).group()


def test_the_ways_to_pick_a_solver_or_device_exist_in_the_code():
    """The texts name `settings.solver`, `--solver` of `topopt run` and `topopt check`, and `settings.gpu_device`, and say that `run_topopt` has no solver argument."""
    commands = build_parser()._subparsers._group_actions[0].choices
    for name in ("run", "check"):
        option = next(a for a in commands[name]._actions if "--solver" in a.option_strings)
        assert set(option.choices) == set(SOLVER_VALUES), name
    assert "solver" not in inspect.signature(server.run_topopt).parameters
    assert {"solver", "gpu_device"} <= set(Settings.__dataclass_fields__)
    assert SOLVER_VALUES == ("auto", "cholesky", "pardiso", "amg", "gpu-mg")      # the lists in the three texts


def test_the_instructions_say_after_a_run_what_was_checked_and_what_was_not():
    text = server.INSTRUCTIONS
    sentence = ("After a run say what was checked and what was not (stress, fatigue, the printed part), the files, "
                "and what to ask next (`levers` of run_result).")
    assert sentence in text
    # "pass limit" is the plain word for one verdict state only
    assert 'For iteration_limit say "the run stopped at its pass limit", not "not converged".' in text


def test_the_instructions_block_is_an_offer_and_not_a_rule():
    block = instructions_block()
    assert block.strip()
    assert not HABIT.search(block), HABIT.search(block).group()


def test_the_block_and_the_guard_paragraph_are_in_the_first_2048_characters_and_the_guard_paragraph_keeps_its_facts():
    text = server.INSTRUCTIONS
    guard = text[text.index("Tools give facts"):text.index("After a run say")]
    for fact in ("settings.max_grid_voxels", "max_active_voxels", "method settings", "names its override",
                 "check_problem.memory_estimate against the RAM", "build_regions stops", "the machine can freeze",
                 "report what, why and the price"):
        assert fact in guard, fact
    # everything before the next line, from the offer to the end of the guard paragraph, ends before the cut of the old clients
    assert len(text) <= OLD_CLIENT_CUT
    assert text.index("Offer three ways") < text.index("Before the first run call environment") < text.index("Tools give facts")
    # the last line names the other tools: the rules of those tools are in their descriptions (#380)
    assert text.splitlines()[-1].startswith("Descriptions say when to use: ")


def test_the_instructions_have_room_under_the_one_bound():
    assert len(server.INSTRUCTIONS) <= MCP_TEXT_LIMIT
    assert len(server.mcp.instructions) == len(server.INSTRUCTIONS)


def test_no_test_file_keeps_the_old_bound_of_3000_for_the_instructions():
    """The old bound (below 3000) is replaced by a constant of tests/test_plugin_manifest.py: MCP_TEXT_LIMIT
    from P9-3, and OLD_CLIENT_CUT (2048) for the instructions since #380."""
    old = re.compile(r"len\([\w.]*instructions\)\s*<\s*3000", re.I)
    found = [f.name for f in sorted((ROOT / "tests").glob("test_*.py")) if f.name != Path(__file__).name and old.search(f.read_text(encoding="utf-8"))]
    assert not found, found
    users = [f.name for f in sorted((ROOT / "tests").glob("test_*.py")) if "OLD_CLIENT_CUT" in f.read_text(encoding="utf-8")]
    for name in ("test_mcp.py", "test_verify_mcp.py", "test_load_types.py", "test_plugin_manifest.py"):
        assert name in users, name


# --- the skill and the lead role ----------------------------------------------------------------------------------------

def test_the_skill_has_the_same_offer_and_names_environment_before_the_first_run():
    block = skill_block()
    low = lower(block)
    for phrase in OFFER:
        assert phrase.lower() in low, phrase
    assert "every way keeps the load question" in low
    assert "`askuserquestion` call with three options can carry the offer" in low        # "can", not "must"; another client gets a list
    assert "before the first run, call `environment`" in low and "an estimate, not a limit" in low
    assert "`check_problem` gives the start estimate" in low and "then `run_status` shows the progress" in low
    assert "after pass 2" not in low
    assert "name the work folder in your first reply" in low and "answer in the user's language" in low
    assert not HABIT.search(block), HABIT.search(block).group()
    assert "conservative" not in skill_text().lower()


def test_the_skill_says_that_the_user_can_pick_another_solver_or_device_and_how():
    guard_starts_before_the_cut()
    low = lower(skill_block())
    assert ("tell the user the device and the time, and that the user can pick another solver or device "
            "(`settings.solver`, `settings.gpu_device`; step 1).") in low
    flat = " ".join(skill_step_1().split())
    pick = flat[flat.index("The device that will run is"):flat.index("If `recommendations` has an offer")]
    for fact in ("`auto_solver`", "`opencl.devices`", "Tell the user that they can pick another solver or device.", "`auto` stays the default until the user asks",
                 "In the YAML, `settings.solver` takes", "`topopt run` and `topopt check` take `--solver` with the same values, for one command.",
                 "`run_topopt` has no solver argument: edit the YAML, then call `check_problem`.",
                 "`settings.gpu_device` is `auto`, a number from the list that `topopt check` prints", "or a part of the device name"):
        assert fact in pick, fact
    for value in SOLVER_VALUES:
        assert f"`{value}`" in pick, value
    assert not HABIT.search(pick), HABIT.search(pick).group()


def test_the_skill_says_after_a_run_what_was_checked_and_what_was_not_in_plain_words():
    text = lower(skill_text())
    assert 'for the state `iteration_limit`, say it in plain words: "the run stopped at its pass limit", not "not converged".' in text
    assert "after a run, say what was checked and what was not (stress, fatigue, the printed part), the files the user has, and what the user can ask next (step 6)." in text
    assert "a printable part is not a certified part." in text
    # the rules that stay: the load question, "run it", the install rule
    assert 'ask for it before the run, also when the user says "run it".' in text
    assert "install or build something only after the user agrees." in text


def test_the_skill_copies_are_equal_and_the_skill_text_follows_the_freedom_rule():
    source = skill_text()
    assert (ROOT / ".agents" / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8") == source
    assert not re.search(r"\b(must|never)\b", source, re.I)          # the freedom rule for the whole file (tests/test_client_configs.py has it too)


def test_the_lead_role_has_the_same_offer_and_the_agent_body_is_its_copy():
    role = role_text()
    low = lower(role)
    for phrase in OFFER:
        assert phrase.lower() in low, phrase
    assert "every way keeps the load question" in low
    assert "`askuserquestion` call with three options can carry the offer" in low
    assert "before the first run, call `environment` and tell the user the device and the time, as an estimate" in low
    assert "name the work folder in your first reply" in low and "answer in the user's language, plain words first." in low
    assert 'for the state `iteration_limit`, say "the run stopped at its pass limit", not "not converged".' in low
    assert "(`check_problem` at the start, then `run_status`)" in low
    assert "what was not checked (stress, fatigue, the printed part)" in low
    body = role[4:].partition("\n---\n")[2]          # the whole body: the new first line and the final review too
    assert not HABIT.search(body), HABIT.search(body).group()
    assert "conservative" not in low
    agent = (ROOT / "agents" / "lead-design-engineer.md").read_text(encoding="utf-8")
    assert role[4:].partition("\n---\n")[2].strip() == agent[4:].partition("\n---\n")[2].strip()


def test_the_lead_role_says_that_the_user_can_pick_another_solver_or_device_in_both_files():
    guard_starts_before_the_cut()
    clause = ("(`check_problem` at the start, then `run_status`), and that the user can pick another solver or device "
              "(`settings.solver`, `settings.gpu_device`). Name the work folder in your first reply.")
    for path in (ROOT / "topopt_mcp" / "roles" / "lead-design.md", ROOT / "agents" / "lead-design-engineer.md"):
        text = " ".join(path.read_text(encoding="utf-8").split())
        assert clause in text, path.name
        assert not HABIT.search(clause), HABIT.search(clause).group()


def test_the_instructions_block_does_not_call_an_estimate_conservative():
    assert "conservative" not in server.INSTRUCTIONS.lower()


def test_the_summary_of_part_to_topopt_is_written_in_the_language_of_the_user():
    """A line that said "simple English" would make a user of another language read English (the block says: the user's language)."""
    source = (ROOT / ".claude" / "skills" / "part-to-topopt" / "SKILL.md").read_text(encoding="utf-8")
    assert "simple English" not in source
    assert "Write the summary in the user's language, in plain words, with short sentences." in source
    # the technical numbers on request (review M3 of P9-3; eval N10)
    assert "- The time estimate in plain words, and any WARNING line. Give the voxel size, the active voxel count and the memory estimate when the user asks." in source
    assert (ROOT / ".agents" / "skills" / "part-to-topopt" / "SKILL.md").read_text(encoding="utf-8") == source
