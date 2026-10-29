"""LAUNCH-3 (#228): the agent looks at the part before it reports, and a sketch gives proportions, not a size.

These are text guards and eval-case guards. They run no model. The eval cases run with `claude plugin eval` (evals/README.md).
What the tests pin:

- `start-here` draws the problem YAML with `render_part` before the first run, draws the run folder after each run, and says
  in one or two lines what the picture shows. A change gets the same look.
- `part-to-topopt` gets the scale of a sketch from one real dimension or a known object, and asks before the run when there is none.
- The role `lead-design` and the server instructions name the two looks.
- The eval cases `render-look` and `sketch-no-size` exist, with graders that can pass in a correct session and fail in a wrong one.
- The mock `render_part.md` names two real fixtures, and no committed file holds a path of the recording machine.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
HABIT = re.compile(r"\b(must|never|always)\b", re.I)          # the freedom rule: facts, reasons and options

START_HERE = [ROOT / "skills" / "start-here" / "SKILL.md", ROOT / ".agents" / "skills" / "start-here" / "SKILL.md"]
PART_SKILL = [ROOT / ".claude" / "skills" / "part-to-topopt" / "SKILL.md", ROOT / ".agents" / "skills" / "part-to-topopt" / "SKILL.md"]

TOOL = "mcp__plugin_topopt_topopt__"
RENDER, RUN = TOOL + "render_part", TOOL + "run_topopt"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def section(text: str, start: str, end: str) -> str:
    """The text from the heading `start` up to the heading `end`."""
    i = text.index(start)
    return text[i:text.index(end, i)]


def front(path: Path) -> dict:
    """The YAML header of an eval file (a prompt or a grader)."""
    text = read(path)
    assert text.startswith("---\n"), path
    return yaml.safe_load(text.split("---\n")[1]) or {}


def body(path: Path) -> str:
    return read(path).split("---\n", 2)[2]


# --- start-here ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", START_HERE, ids=lambda p: str(p.relative_to(ROOT)))
def test_start_here_draws_the_yaml_before_the_first_run(path):
    step3 = section(read(path), "## Step 3. Check", "## Step 4. Run")
    assert "Look at the problem before the first run." in step3
    assert "`render_part` with `yaml_path`" in step3
    assert "one force arrow for each force" in step3                  # the facts of the YAML render
    for question in ("is each load on the face that the user means", "does its arrow point the right way",
                     "is the fixed region where the part is held", "does a keep-out cover the hole"):
        assert question in step3, question
    assert "in one or two lines" in step3
    assert "Draw the YAML again after each change of the YAML or of a region STL, before the next run." in step3
    assert "a few seconds" in step3                                       # the cost, with the reason to look
    assert "`inline=false`" in step3 and "did not see the picture" in step3   # a client that cannot show images


@pytest.mark.parametrize("path", START_HERE, ids=lambda p: str(p.relative_to(ROOT)))
def test_start_here_looks_at_the_part_after_each_run_before_the_report(path):
    text = read(path)
    step5 = section(text, "## Step 5. Watch and report", "## Step 6.")
    assert "After each run, look at the part before you report." in step5
    assert "`render_part` with the run folder" in step5
    assert "in one or two lines" in step5
    assert "magenta back faces" in step5 and "a part that is cut off" in step5
    assert "`stl_paths` for the `edited.stl` of `apply_marks`" in step5     # a change with no run gets the same look
    # the checklist line points to the lead-design checklist; it does not list its items, so it does not repeat the look
    assert "(`team_guide` with `lead-design`)." in step5
    assert "the look at the part" not in step5
    assert "a look at the render" not in text
    # the old offer ("To look at the part, call render_part") is gone: the look is a step, not an offer
    assert "To look at the part, call `render_part`" not in text
    # a change of the problem after a run: the proposed YAML is drawn before the run (Step 6)
    assert "draw the proposed YAML (Step 3)" in section(text, "## Step 6.", "## Roles")


@pytest.mark.parametrize("path", START_HERE, ids=lambda p: str(p.relative_to(ROOT)))
def test_start_here_keeps_the_look_apart_from_the_renders_output_and_points_a_sketch_to_the_skill(path):
    text = read(path)
    assert "It is not the `renders` output of the preferences (Step 1b)" in text
    assert "Look at the picture before you report" in section(text, "## Talking to the user", "## Facts to tell the user")
    step2 = section(text, "## Step 2. The part", "## Step 3. Check")
    assert "A sketch or a photo with no size gives proportions only" in step2 and "Step 1, item 4" in step2


@pytest.mark.parametrize("path", START_HERE, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_new_start_here_text_follows_the_freedom_rule(path):
    """Facts, reasons and options: no habitual must, never or always in the lines that this change adds."""
    text = read(path)
    anchors = ("Look at the picture before you report", "Look at the problem before the first run.",
               "After each run, look at the part before you report.", "A sketch or a photo with no size gives proportions only")
    for anchor in anchors:
        lines = [l for l in text.splitlines() if anchor in l]
        assert len(lines) == 1, anchor                                  # the line exists (a missing line is no pass)
        assert not HABIT.search(lines[0]), (anchor, HABIT.search(lines[0]).group())


def test_the_start_here_copies_are_equal():
    assert read(START_HERE[0]) == read(START_HERE[1])
    assert START_HERE[0].read_bytes() == START_HERE[1].read_bytes()


# --- part-to-topopt -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", PART_SKILL, ids=lambda p: str(p.relative_to(ROOT)))
def test_a_sketch_gives_proportions_and_the_scale_comes_from_one_real_size(path):
    text = read(path)
    step1 = section(text, "## Step 1. Infer the part", "### Load inventory")
    item4 = next(l for l in step1.splitlines() if l.startswith("4. Dimensions"))
    assert "A sketch, a photo or a drawing with no dimension gives proportions, not a size." in item4
    assert "one real dimension that the user gives" in item4
    assert "a known object in the picture (a standard screw, a coin, a known product)" in item4
    assert "ask for one before the run" in item4 and "the size is one of the answers that change the part" in item4
    assert "origin `inferred` in `record_brief`" in item4 and "show the list to the user" in item4
    assert not HABIT.search(item4[item4.index("A sketch, a photo"):item4.index("The part has to fit the print bed.")])
    # item 9 keeps its point: how the part touches other objects
    item9 = next(l for l in step1.splitlines() if l.startswith("9. Use in the world"))
    assert "how the part touches the objects around it" in item9 and "A sketch or a description often does not show this." in item9
    # the assumptions list gets the source of the scale
    assert "- Scale of a sketch: the dimension or the object that gave the scale, and each size taken from the proportions." in text


@pytest.mark.parametrize("path", PART_SKILL, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_renders_output_and_the_own_look_stay_two_things(path):
    text = read(path)
    assert "`renders` (`render_part`: the images that the user gets; your own look in `start-here` is a separate check)" in text


def test_the_part_to_topopt_copies_are_equal():
    assert PART_SKILL[0].read_bytes() == PART_SKILL[1].read_bytes()


# --- the role and the server instructions -------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [ROOT / "topopt_mcp" / "roles" / "lead-design.md", ROOT / "agents" / "lead-design-engineer.md"],
                         ids=lambda p: p.name)
def test_lead_design_names_the_two_looks(path):
    text = read(path)
    item3 = next(l for l in text.splitlines() if l.startswith("3. The run."))
    assert "`render_part` (images: the regions before the first run, the part after each run; not a stress or fit check)" in item3
    assert "8. Look at the render of the part after each run and say what you see at each interface." in text
    assert not HABIT.search(text)


def test_the_description_of_render_part_names_both_looks_and_the_instructions_point_to_it():
    """#380: the instructions have room for the name of render_part only (2048 characters); the two looks are in its description, in the first 2048."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    text = server.render_part.__doc__
    line = "A look before the first run and after each run finds a wrong region or load fast."
    assert line in text and text.index(line) + len(line) <= 2048
    assert "a problem YAML, before a run" in text and "run_dir (renders result/final.stl" in text
    assert not HABIT.search(text[:text.index(line) + len(line)])
    assert "render_part" in next(l for l in server.INSTRUCTIONS.splitlines() if l.startswith("Core tools:"))   # core since #174


# --- the eval cases: what the graders accept ----------------------------------------------------------------------------

def matches(spec, call) -> bool:
    """The rule of the runner for `before` and `after`: a tool name, or {tool, input_match} (a regex on the JSON text of the input)."""
    tool, args = call
    if isinstance(spec, str):
        return tool == spec
    if tool != spec["tool"]:
        return False
    return "input_match" not in spec or re.search(spec["input_match"], json.dumps(args)) is not None


def tool_order_passes(grader: dict, calls: list) -> bool:
    """Pass when the FIRST call that matches `before` comes before the FIRST call that matches `after`; fail when either never happens."""
    first = lambda spec: next((i for i, c in enumerate(calls) if matches(spec, c)), None)   # noqa: E731
    b, a = first(grader["before"]), first(grader["after"])
    return b is not None and a is not None and b < a


YAML_CALL = (RENDER, {"yaml_path": "runs/tshirt_hook/tshirt_hook.yaml"})
RUN_CALL = (RUN, {"yaml_path": "runs/tshirt_hook/tshirt_hook.yaml", "out_dir": "runs/tshirt_hook/run"})
LOOK_CALLS = {
    "a run folder": (RENDER, {"run_dir": "runs/tshirt_hook/run"}),
    "no argument (the last run)": (RENDER, {}),
    "the final STL": (RENDER, {"stl_paths": ["runs/tshirt_hook/run/result/final.stl"]}),
    "a run folder, facts only": (RENDER, {"run_dir": "runs/tshirt_hook/run", "inline": False}),
}
OTHER = [(TOOL + "build_regions", {"spec": {}}), (TOOL + "check_problem", {"yaml_path": "runs/tshirt_hook/tshirt_hook.yaml"})]


def render_look_graders():
    folder = EVALS / "render-look" / "graders"
    return front(folder / "look-before-run.md"), front(folder / "look-after-run.md"), front(folder / "says-look.md")


def test_render_look_case_has_its_tags_and_a_russian_prompt_with_no_word_about_a_picture():
    head = front(EVALS / "render-look" / "prompt.md")
    assert {"smoke", "mock", "flow", "render"} <= set(head["tags"])
    prompt = body(EVALS / "render-look" / "prompt.md")
    assert re.search(r"[А-Яа-я]{4}", prompt)                                           # every eval prompt is in Russian
    assert not re.search(r"картин|рендер|render|изображен|рисун|эскиз|посмотр|покажи|фото|скриншот|вид сбоку", prompt, re.I)
    for word in ("футболк", "PETG", "шуруп", "запускай"):                              # a hook from words, run it
        assert word in prompt, word


def test_render_look_graders_are_a_yaml_look_before_the_run_a_result_look_after_it_and_a_line_in_the_reply():
    before_run, after_run, says = render_look_graders()
    assert before_run["type"] == "tool_order" and before_run["after"] == RUN
    assert before_run["before"]["tool"] == RENDER and "yaml_path" in before_run["before"]["input_match"]
    assert after_run["type"] == "tool_order" and after_run["before"] == RUN and after_run["after"]["tool"] == RENDER
    assert says["type"] == "llm" and says["focus"] == "last_message"
    rubric = body(EVALS / "render-look" / "graders" / "says-look.md")
    assert "PASS if" in rubric and "FAIL if" in rubric and "what the render of the finished part shows" in rubric


@pytest.mark.parametrize("name", sorted(LOOK_CALLS))
def test_a_correct_session_passes_both_order_graders(name):
    before_run, after_run, _ = render_look_graders()
    session = OTHER + [YAML_CALL, RUN_CALL, (TOOL + "run_status", {}), (TOOL + "run_result", {}), LOOK_CALLS[name]]
    assert tool_order_passes(before_run, session)
    assert tool_order_passes(after_run, session), name


def test_a_wrong_session_fails_the_grader_that_it_breaks():
    before_run, after_run, _ = render_look_graders()
    result = (TOOL + "run_result", {})
    look = LOOK_CALLS["a run folder"]
    regions_only = (RENDER, {"stl_paths": ["runs/tshirt_hook/design.stl", "runs/tshirt_hook/hook_tip.stl"]})
    sessions = {
        # name: (session, passes the grader of the look before the run, passes the grader of the look after it)
        "no look at all": (OTHER + [RUN_CALL, result], False, False),
        "the look only before the run": (OTHER + [YAML_CALL, RUN_CALL, result], True, False),
        "the look only after the run": (OTHER + [RUN_CALL, result, look], False, True),
        "the YAML drawn only after the run": (OTHER + [RUN_CALL, result, YAML_CALL], False, False),
        "region STLs before, no YAML render": (OTHER + [regions_only, RUN_CALL, result, look], False, True),
        "region STLs after the run, not the result": (OTHER + [YAML_CALL, RUN_CALL, result, regions_only], True, False),
        "the YAML drawn twice, never the part": (OTHER + [YAML_CALL, RUN_CALL, result, YAML_CALL], True, False),
        "no run at all": (OTHER + [YAML_CALL], False, False),
    }
    for name, (calls, want_before, want_after) in sessions.items():
        assert tool_order_passes(before_run, calls) is want_before, name
        assert tool_order_passes(after_run, calls) is want_after, name


# --- the sketch case ----------------------------------------------------------------------------------------------------

def test_sketch_no_size_case_stages_one_sketch_and_asks_before_a_run():
    case = EVALS / "sketch-no-size"
    head = front(case / "prompt.md")
    assert {"smoke", "mock", "ask"} <= set(head["tags"]) and "Read" in head["allowed_tools"]
    prompt = body(case / "prompt.md")
    assert re.search(r"[А-Яа-я]{4}", prompt)
    for word in ("files/sketch.png", "куртк", "шуруп", "PETG", "запусти"):             # the file, the load, the mounting, the material, "run it"
        assert word in prompt, word
    assert not re.search(r"\d\s*(мм|см|кг|г\b|Н\b)", prompt)                          # the prompt gives no size and no mass
    assert yaml.safe_load(read(case / "case.yaml"))["context"]["scaffold_script"] == "stage-files.sh"
    stage = read(case / "stage-files.sh")
    assert 'cp "$here"/files/*.png files/' in stage and "--scaffold" in stage
    no_run = front(case / "graders" / "no-run.md")
    assert (no_run["type"], no_run["tool"], no_run["min"], no_run["max"], no_run["arm"]) == ("tool_used", RUN, 0, 0, "both")
    asks = front(case / "graders" / "asks-size.md")
    assert asks["type"] == "llm" and asks["focus"] == "last_message"
    rubric = body(case / "graders" / "asks-size.md")
    assert "PASS if the reply asks the user for a real size for the scale" in rubric and "a known object" in rubric
    assert "FAIL if the reply says that a run started or finished" in rubric and "takes a size from the sketch with no question" in rubric


def test_the_sketch_is_a_png_that_make_sketch_draws_again(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_sketch", EVALS / "tools" / "make_sketch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "sketch.png"
    module.draw(out)
    committed = EVALS / "sketch-no-size" / "files" / "sketch.png"
    assert committed.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(committed) as a, Image.open(out) as b:
        assert a.size == b.size and a.tobytes() == b.tobytes()      # the pixels, not the bytes of the compressor
    assert "no number" in read(EVALS / "tools" / "make_sketch.py") and "dimension line" in read(EVALS / "tools" / "make_sketch.py")


# --- the mock, the fixtures and the recorder ----------------------------------------------------------------------------

MACHINE_PATH = re.compile(r"/app/|/home/|/tmp/|/private/var/|/var/folders/|/workspace/")
NEW_EVAL_FILES = [
    *sorted((EVALS / "render-look").rglob("*.md")), *sorted((EVALS / "sketch-no-size").rglob("*.*")),
    EVALS / "mocks" / "topopt" / "render_part.md", EVALS / "mocks" / "topopt" / "fixtures" / "render_part_yaml.json",
    EVALS / "mocks" / "topopt" / "fixtures" / "render_part_run.json", EVALS / "tools" / "make_sketch.py",
]


def test_the_render_part_mock_names_two_fixtures_that_are_real_json_answers():
    mock = EVALS / "mocks" / "topopt" / "render_part.md"
    assert front(mock)["type"] == "agent"
    text = read(mock)
    fixtures = EVALS / "mocks" / "topopt" / "fixtures"
    assert re.findall(r"\{\{file:(fixtures/[\w.]+)\}\}", text) == ["fixtures/render_part_yaml.json", "fixtures/render_part_run.json"]
    yaml_answer = json.loads(read(fixtures / "render_part_yaml.json"))
    run_answer = json.loads(read(fixtures / "render_part_run.json"))
    for answer in (yaml_answer, run_answer):
        assert answer["ok"] is True and answer["inline"] is False                       # the facts only: a mock answers text only
        assert answer["back_face_share"] == 0.0 and answer["missing_files"] == []
        assert answer["montage_rel"].endswith("_montage.png") and not Path(answer["out_dir_rel"]).is_absolute()
    assert yaml_answer["input"]["kind"] == "yaml" and yaml_answer["load_arrows"]["arrow_count"] == 2
    assert {c["case"] for c in yaml_answer["load_arrows"]["cases"]} == {"hang", "pull_off"}
    assert run_answer["input"]["kind"] == "run" and run_answer["load_arrows"] is None
    assert [m["file"] for m in run_answer["meshes"]][0] == "final.stl" and run_answer["meshes"][0]["role"] == "part"
    # the mock says which answer goes with which call, and how a call with no argument is read
    assert "`yaml_path`" in text and "`run_dir`, or has no argument at all" in text and "no run yet: give run_dir or stl_paths" in text
    # the tool exists in the recorded tool list, so the suite can offer it
    tools = json.loads(read(EVALS / "mocks" / "topopt" / "_tools.json"))["tools"]
    assert "render_part" in {t["name"] for t in tools}


def test_the_recorder_makes_the_two_render_fixtures_after_check_problem_and_after_run_result():
    source = read(EVALS / "tools" / "record_fixtures.py")
    yaml_call = 'rec.call("render_part", yaml_path=yaml_rel, inline=False)'
    run_call = 'rec.call("render_part", run_dir=run_dir, inline=False)'
    assert source.index('rec.call("check_problem"') < source.index(yaml_call) < source.index('rec.call("run_topopt"')
    assert source.index('rec.call("run_result"') < source.index(run_call) < source.index('rec.call("print_check"')
    assert '"render_part_yaml"' in source and '"render_part_run"' in source


def test_no_new_eval_file_holds_a_path_of_the_recording_machine():
    for f in NEW_EVAL_FILES:
        assert f.is_file(), f
        if f.suffix == ".png":
            continue
        for n, line in enumerate(read(f).splitlines(), 1):
            assert not MACHINE_PATH.search(line), f"{f.relative_to(ROOT)}:{n}: {line.strip()[:100]}"


def test_the_readme_lists_the_two_cases_the_mock_and_the_fixtures():
    text = read(EVALS / "README.md")
    assert re.search(r"^\| `render-look` \| smoke, mock, flow, render \|", text, re.M)
    assert re.search(r"^\| `sketch-no-size` \| smoke, mock, ask \|", text, re.M)
    assert "`build_regions`, `record_brief`, `check_problem`, `render_part`, `run_topopt`," in text      # the list of the agent mocks
    assert "`render_part_yaml`" in text and "`render_part_run`" in text and "inline=false" in text
    assert "`render_part_yaml.json` after `check_problem`" in text and "`render_part_run.json` after" in text   # the update steps
    assert "tools/make_sketch.py" in text
    # every case folder that has a prompt is in the table
    cases = sorted(p.parent.name for p in EVALS.glob("*/prompt.md"))
    for name in cases:
        assert f"| `{name}` |" in text, name
