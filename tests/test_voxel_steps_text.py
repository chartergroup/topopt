"""#452: voxel steps on a finished part are a known limit, and four texts say the same facts. They run no model and no solver.

The paragraph is in step 5 of `start-here` (both copies, equal), in the role `finish` (`topopt_mcp/roles/finish.md`, section "Voxel steps are a
known limit") and in its agent copy `agents/finish-engineer.md`. The tests pin:

- the paragraph is the same text in the four files;
- the facts: a known limit of the voxel method in this version, a step is one voxel high, it is in the numbers of the run, the routes with the
  price of each (the look, `apply_marks`, `fine_from`), a finer voxel does not change the tread in voxels, the mesh key is no route, a better
  smoothing is planned;
- the paragraph has no date, no version and no promise of a release, and no claim on fatigue or strength at a step edge;
- the names in the paragraph exist in the code (the tools, the parameters and the keys of the metrics);
- the eval case `voxel-steps`: files, the llm grader (its FAIL clause names a date, a version, a month, "soon" and a release promise), README row.
  There is no regex grader for a date: a pattern false-hit good replies; the llm grader fails a date.
"""
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topopt_mcp import server  # noqa: E402

START_HERE = ("skills/start-here/SKILL.md", ".agents/skills/start-here/SKILL.md")
ROLES = ("topopt_mcp/roles/finish.md", "agents/finish-engineer.md")
CASE = ROOT / "evals" / "voxel-steps"
FIRST = "Steps on the finished part are a known limit of the voxel method in this version, not a fault of the input."
LAST = "stress at a step edge."
# The sentence about the plan: change it here only (the four texts and the rubric follow it; see the eval README).
PLAN_SENTENCE = "A better smoothing of long steps on member surfaces is planned, with no date."


def paragraph(path: str) -> str:
    """The paragraph of the file: from the first sentence to the last one, on one line."""
    text = (ROOT / path).read_text(encoding="utf-8")
    assert text.count(FIRST) == 1, path
    start = text.index(FIRST)
    end = text.index(LAST, start) + len(LAST)
    return text[start:end]


ALL = START_HERE + ROLES


def test_the_four_files_hold_the_same_paragraph():
    first = paragraph(ALL[0])
    assert "\n" not in first
    for path in ALL[1:]:
        assert paragraph(path) == first, path


def test_the_two_copies_of_start_here_are_equal():
    assert (ROOT / START_HERE[0]).read_text(encoding="utf-8") == (ROOT / START_HERE[1]).read_text(encoding="utf-8")


def test_the_paragraph_is_next_to_the_finishing_pass_in_step_5_and_has_its_own_section_in_the_role():
    for path in START_HERE:
        step = (ROOT / path).read_text(encoding="utf-8").partition("## Step 5. Watch and report")[2].partition("\n## ")[0]
        assert FIRST in step, path
        assert step.index("After each successful run, call the finishing pass") < step.index(FIRST) < step.index("To check the stress of the finished part"), path
        assert "Nothing calls you back when the run ends" in step, path     # #450 paragraph stays
    for path in ROLES:
        text = (ROOT / path).read_text(encoding="utf-8")
        assert "\n## Voxel steps are a known limit\n\n" + FIRST in text, path
        section = text.partition("\n## Voxel steps are a known limit\n")[2].partition("\n## ")[0]
        assert section.strip() == paragraph(path), path                  # nothing else in the section: tests/test_finish_role.py cuts only the paragraph
        assert text.index("## What you get") < text.index("## Voxel steps are a known limit") < text.index("## What does not exist yet"), path


@pytest.mark.parametrize("path", ALL)
def test_the_paragraph_has_the_facts(path):
    text = paragraph(path)
    for fact in (
        "a few degrees off a grid plane",
        "each one-voxel step stretches into a long flat tread",
        "a step every 8.9 mm at 0.5 mm and 3.2 degrees",
        "Long, thin parts show them most",
        "curved or steep faces give short steps, which the mesh step smooths or makes lower",
        "A step is one voxel high and is already in the run's numbers",
        "the stiffness comes from the voxel model",
        "`mass_final_g` is the mass of `final.stl`",
        "For long steps, cheapest first:",
        "on the faces of a closed design space, if `metrics.mesh.envelope_fill` shows that the fill did not run, "
        "`settings.envelope_fill` `auto` or `on` in a post-only resume (minutes; the stiffness at the same mass falls a few percent, more on a coarse grid)",
        "on member surfaces, a `smooth` mark with `apply_marks` (seconds; the mass change is reported; its effect on a long step is not measured, "
        "and the stiffness change is known only with a `stiffness` or `stress` rule of `acceptance`)",
        "a finer voxel with `fine_from` (a whole run; the estimate is in `run_result.fine_pass`)",
        "which makes the steps lower but can still show them",
        "`settings.mesh_method` is no route for long steps",
        PLAN_SENTENCE,
        "Nothing is measured on fatigue or stress at a step edge.",
    ):
        assert fact in text, (path, fact)
    assert text.index("`metrics.mesh.envelope_fill`") < text.index("`settings.envelope_fill`") < text.index("`apply_marks`") < text.index("`fine_from`")      # cheapest first


@pytest.mark.parametrize("path", ALL)
def test_the_paragraph_has_no_date_no_version_no_promise_and_no_strength_claim(path):
    text = paragraph(path)
    bad = re.findall(
        r"(?i)\b(?:next|upcoming|coming|future)\s+(?:release|version|update)|(?<!as )\bsoon\b|\b(?:january|february|march|april|june|july|august|september|october|november|december)\b"
        r"|\bQ[1-4]\b|\bin 20\d\d\b|\bv?\d+\.\d+\.\d+\b|\bversion \d|\bwill (?:be|ship|come|fix)\b|\b(?:must|never|always)\b", text)
    assert not bad, (path, bad)
    assert "Nothing is measured on fatigue or stress at a step edge." in text      # the paragraph claims nothing on fatigue or strength
    assert not re.search(r"(?i)safe|certified|strong enough|will not break", text), path


def test_the_paragraph_names_no_product_and_no_person():
    text = paragraph(ALL[0])
    assert not re.search(r"(?i)claude|anthropic|openai|gpt|gemini|codex|cursor", text)


def test_the_names_in_the_paragraph_exist_in_the_code():
    import inspect
    tools = {t.name: t for t in server.mcp._tool_manager.list_tools()}
    for name in ("render_part", "apply_marks", "run_topopt", "compare_runs", "run_result", "ratchet_run"):
        assert name in tools, name
    assert "fine_from" in inspect.signature(tools["run_topopt"].fn).parameters
    post = (ROOT / "topopt" / "post.py").read_text(encoding="utf-8")
    for key in ('"mass_final_g"', '"compliance_final"', '"volume_error_pct"', '"volume_thresholded_mm3"'):
        assert key in post, key
    assert "mass_final = volume / 1000.0 * dens" in post                # the mass is the volume of the mesh times the density
    assert "settings.mesh_method" in (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    contracts = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    assert re.search(r"^#{2,3} 24\.6\.? ", contracts, re.M) and "fine_pass" in contracts
    assert "`settings.envelope_fill`" in contracts and "A resume of a finished run reads the key again (the post step reads it)" in contracts
    assert "| `envelope_fill` (`sdf`) |" in contracts                     # the fact `metrics.mesh.envelope_fill`
    assert "falls by 1.7 to 3.6 %" in contracts and "it falls by 7 to 21 %" in contracts      # a few percent; more on a coarse grid
    from topopt.config import ENVELOPE_FILL_VALUES
    assert {"auto", "on", "off"} <= set(ENVELOPE_FILL_VALUES)


# --- the eval case -------------------------------------------------------------------------------------------------------------------------

def front(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path
    return yaml.safe_load(text.split("---\n")[1]) or {}


def test_the_case_has_its_files_tags_and_a_prompt_with_the_questions():
    head = front(CASE / "prompt.md")
    assert {"mock", "explain", "finish"} <= set(head["tags"])
    assert sorted(p.name for p in CASE.iterdir()) == ["graders", "mocks", "prompt.md"]
    assert sorted(p.name for p in (CASE / "graders").iterdir()) == ["explains-the-limit.md"]
    assert sorted(p.relative_to(CASE).as_posix() for p in (CASE / "mocks").rglob("*") if p.is_file()) == [
        "mocks/topopt/fixtures/run_result.json", "mocks/topopt/run_result.md"]
    prompt = (CASE / "prompt.md").read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "drone-arm" in prompt and "Why does my part have steps?" in prompt and "bad input" in prompt
    assert not re.search(r"(?i)known limit|planned|voxel", prompt)         # the prompt does not hand the answer over


def test_the_llm_grader_asks_for_the_facts_and_fails_a_date_and_a_blame():
    g = CASE / "graders" / "explains-the-limit.md"
    assert front(g)["type"] == "llm" and front(g)["focus"] == "last_message"
    rubric = g.read_text(encoding="utf-8").split("---\n", 2)[2]
    for phrase in ("PASS if", "FAIL if", "known limit of the voxel method", "not a fault of the user's input", "one voxel high",
                   "already in the numbers", "cheapest first", "apply_marks", "fine_from", "a better smoothing", "NO date",
                   "settings.mesh_method", "fatigue", "offers settings.envelope_fill as a new step for this run, which already ran with it", "metrics.mesh.envelope_fill", "a date, a version, a month, \"soon\" or a promise of a release", "NO promise of a release"):
        assert phrase in rubric, phrase


def test_the_readme_has_the_row_and_the_note():
    text = (ROOT / "evals" / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^\| `voxel-steps` \| mock, explain, finish \|", text, re.M)
    assert "`voxel-steps` has no history. It runs a fresh prompt. Its own `mocks/topopt/run_result.md`" in text and "needs the credentials" in text
    assert "claude plugin eval . --case voxel-steps --tag mock --ablation none" in text and "step 5 of `start-here` only" in text and "no `smoke` tag" in text


def test_the_case_files_hold_no_machine_path_and_no_account_fact():
    for f in sorted(CASE.rglob("*")):
        if f.is_file():
            assert not re.search(r"/app/|/home/|/tmp/|/workspace/|@", f.read_text(encoding="utf-8")), f


def test_the_case_mock_of_run_result_is_a_finished_drone_arm_run_with_the_fill_on():
    mock = CASE / "mocks" / "topopt" / "run_result.md"
    assert front(mock)["type"] == "agent"
    text = mock.read_text(encoding="utf-8")
    assert "{{file:fixtures/run_result.json}}" in text and "`ran` true" in text and "Change nothing in" in text
    d = json.loads((CASE / "mocks" / "topopt" / "fixtures" / "run_result.json").read_text(encoding="utf-8"))
    mesh = d["metrics"]["mesh"]
    assert d["done"] is True and d["run_dir_rel"] == "runs/drone_arm/run"
    assert mesh["method"] == "sdf" and mesh["params"]["envelope_fill"] == "on" and mesh["params"]["sigma_mm"] == 0.5 and mesh["params"]["pitch_mm"] == 0.25
    fill = mesh["envelope_fill"]
    assert fill["ran"] is True and round(fill["grown_share_pct"], 2) == 1.45
    assert {"candidates", "grown", "loose_dropped", "grown_inside_mm3", "grown_overshoot_mm3", "target_mm3", "note"} <= set(fill)     # the keys of CONTRACTS section 19, table
    m = d["metrics"]
    assert m["mass_final_g"] == 10.79 and m["mass_design_space_g"] == 24.65 and m["mass_saved_pct"] == 56.2 and m["iterations"] == 200
    readme = (ROOT / "examples" / "drone_arm" / "README.md").read_text(encoding="utf-8")
    assert "10.79 g" in readme and "24.7 g (56.2 % less mass)" in readme and "`envelope_fill: on`" in readme
    assert "envelope_fill: on" in (ROOT / "examples" / "drone_arm" / "drone_arm.yaml").read_text(encoding="utf-8")


def test_the_prompt_and_the_row_say_that_the_fill_already_ran():
    head = front(CASE / "prompt.md")
    assert "envelope_fill" in head["description"] and "member surfaces" in head["description"]
    assert "the fill already ran" in head["expected_outcome"] and "member surfaces" in head["expected_outcome"]
    assert "is planned" in head["expected_outcome"] and "no date" in head["expected_outcome"]
    row = re.search(r"^\| `voxel-steps` \|.*$", (ROOT / "evals" / "README.md").read_text(encoding="utf-8"), re.M).group(0)
    assert "already ran" in row and "member surfaces" in row and "planned" in row
