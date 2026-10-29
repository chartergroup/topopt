"""FIN-1 (#361) step 4: the eval case `finish-after-run`. Text and grader guards; they run no model (evals/README.md).

What the tests pin:

- the case exists, has its tags and a Russian prompt with no word about a finish or a picture;
- the graders accept a correct session (run, then the finish look, then the print check) and fail a wrong one;
- the look of the grader is the look that the role `finish` names (views front and fl2, regions off);
- the rubric asks for the pass and the print check after it, and fails a "certified" claim;
- the case needs only tools that the suite mocks have, and its README row exists.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
CASE = EVALS / "finish-after-run"
TOOL = "mcp__plugin_topopt_topopt__"
RENDER, RUN, PRINT = TOOL + "render_part", TOOL + "run_topopt", TOOL + "print_check"
ROLE = ROOT / "topopt_mcp" / "roles" / "finish.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def front(path: Path) -> dict:
    text = read(path)
    assert text.startswith("---\n"), path
    return yaml.safe_load(text.split("---\n")[1]) or {}


def body(path: Path) -> str:
    return read(path).split("---\n", 2)[2]


def matches(spec, call) -> bool:
    tool, args = call
    if isinstance(spec, str):
        return tool == spec
    if tool != spec["tool"]:
        return False
    return "input_match" not in spec or re.search(spec["input_match"], json.dumps(args)) is not None


def order_passes(grader: dict, calls: list) -> bool:
    first = lambda spec: next((i for i, c in enumerate(calls) if matches(spec, c)), None)   # noqa: E731
    b, a = first(grader["before"]), first(grader["after"])
    return b is not None and a is not None and b < a


G = lambda name: front(CASE / "graders" / name)   # noqa: E731
RUN_CALL = (RUN, {"yaml_path": "runs/tshirt_hook/tshirt_hook.yaml", "out_dir": "runs/tshirt_hook/run"})
YAML_LOOK = (RENDER, {"yaml_path": "runs/tshirt_hook/tshirt_hook.yaml"})
FINISH_LOOK = (RENDER, {"run_dir": "runs/tshirt_hook/run", "views": ["front", "fl2:-1.333,-1,0.373"], "regions": "off"})
CHECK = (PRINT, {})


def test_the_case_has_its_tags_and_a_russian_prompt_with_no_word_about_a_finish():
    head = front(CASE / "prompt.md")
    assert {"smoke", "mock", "flow", "finish"} <= set(head["tags"])
    assert "fresh prompt" in head["description"] and "history" in head["description"]     # the choice is told
    assert not (CASE / "history.jsonl").exists() and not (CASE / "case.yaml").exists()
    prompt = body(CASE / "prompt.md")
    assert re.search(r"[А-Яа-я]{4}", prompt)
    assert not re.search(r"глад|финиш|доработ|ступен|картин|рендер|render|изображен|посмотр|покажи|проверь", prompt, re.I)
    for word in ("футболк", "PETG", "шуруп", "запускай"):
        assert word in prompt, word


def test_the_look_of_the_grader_is_the_look_of_the_role():
    role = read(ROLE)
    assert "`front` and `fl2:-1.333,-1,0.373`" in role and "`regions` `off`" in role
    pattern = G("finish-look.md")["after"]["input_match"]
    assert "fl2:-1\\.333,-1,0\\.373" in pattern and "regions" in pattern and "off" in pattern


def test_a_correct_session_passes_the_order_graders():
    look, check = G("finish-look.md"), G("print-check-after-run.md")
    for calls in ([RUN_CALL, FINISH_LOOK, CHECK], [YAML_LOOK, RUN_CALL, CHECK, FINISH_LOOK]):
        assert order_passes(look, calls) and order_passes(check, calls)
    spaced = (RENDER, {"views": ["fl2:-1.333,-1,0.373"], "regions": "off"})
    assert order_passes(look, [RUN_CALL, spaced])


def test_a_wrong_session_fails_the_grader_that_it_breaks():
    look, check = G("finish-look.md"), G("print-check-after-run.md")
    plain = (RENDER, {"run_dir": "runs/tshirt_hook/run"})
    regions_on = (RENDER, {"run_dir": "runs/tshirt_hook/run", "views": ["front", "fl2:-1.333,-1,0.373"], "regions": "on"})
    sessions = {
        # name: (session, passes the look grader, passes the print check grader)
        "no finish at all": ([YAML_LOOK, RUN_CALL], False, False),
        "the look before the run only": ([YAML_LOOK, FINISH_LOOK, RUN_CALL], False, False),
        "a default look after the run, not the finish views": ([RUN_CALL, plain, CHECK], False, True),
        "regions on": ([RUN_CALL, regions_on, CHECK], False, True),
        "the print check before the run": ([CHECK, RUN_CALL, FINISH_LOOK], True, False),
        "the look, no print check": ([RUN_CALL, FINISH_LOOK], True, False),
        "no run": ([FINISH_LOOK, CHECK], False, False),
    }
    for name, (calls, want_look, want_check) in sessions.items():
        assert order_passes(look, calls) is want_look, name
        assert order_passes(check, calls) is want_check, name


def test_the_order_graders_have_the_right_shape():
    for name in ("finish-look.md", "print-check-after-run.md"):
        g = G(name)
        assert g["type"] == "tool_order" and g["before"] == RUN
    assert G("finish-look.md")["after"]["tool"] == RENDER and G("print-check-after-run.md")["after"] == PRINT


def test_the_rubric_asks_for_the_pass_and_the_print_check_and_fails_a_certified_claim():
    g = G("reports-finish.md")
    assert g["type"] == "llm" and g["focus"] == "last_message"
    rubric = body(CASE / "graders" / "reports-finish.md")
    assert "PASS if" in rubric and "FAIL if" in rubric
    for phrase in ("the finishing pass", "nothing needed a change", "price in stiffness and mass", "print check after the pass",
                   "certified", "no word about the surface look or the finishing pass"):
        assert phrase in rubric, phrase


def test_the_case_needs_only_tools_that_the_suite_mocks_have():
    mocks = {p.stem for p in (EVALS / "mocks" / "topopt").glob("*.md")}
    assert {"run_topopt", "render_part", "print_check", "run_result", "build_regions"} <= mocks
    assert not (CASE / "mocks").exists()                    # the suite mocks are enough: the mocked part needs no edit
    for name in ("cleanup_list", "apply_marks", "ratchet_run", "team_guide"):
        assert name not in mocks and name not in body(CASE / "prompt.md")


def test_the_readme_has_the_row_and_the_note():
    text = read(EVALS / "README.md")
    assert re.search(r"^\| `finish-after-run` \| smoke, mock, flow, finish \|", text, re.M)
    assert "`finish-after-run` has no mock of its own" in text


def test_the_case_files_hold_no_machine_path_and_no_account_fact():
    for f in sorted(CASE.rglob("*")):
        if f.is_file():
            assert not re.search(r"/app/|/home/|/tmp/|/workspace/|@", read(f)), f
    assert sorted(p.name for p in CASE.iterdir()) == ["graders", "prompt.md"]
