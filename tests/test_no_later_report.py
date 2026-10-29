"""#440 step 1: nothing calls the agent back when a run ends, and the texts say so. They run no model and no solver.

No MCP client starts a new model turn when a background job ends, and the server sends nothing that could. An agent that wrote "I will report
when the run finishes" made a promise that cannot be kept. The tests pin:

- the reply of `run_topopt`, the descriptions of `run_status`, `verify_status` and `ratchet_status`, the start messages of `verify_run`
  and `ratchet_run`, and step 5 of `start-here` (both copies) say that nothing
  calls the agent back, name the two choices (call `run_status` again in this turn, or end the turn and tell the user what to ask) and say
  that a promise of a later report cannot be kept;
- the two copies of `start-here` are equal, and step 5 tells how to wait with a shell and without one;
- the sentence is in the first 2,048 characters of the descriptions that it joined, and no description grew past its cut;
- the eval case `no-later-report`: its graders fail a promise of a later report and pass a report or a plain line that tells what to ask.
"""
import asyncio
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topopt_mcp import server  # noqa: E402
from test_optimize import make_case  # noqa: E402

SKILLS = [ROOT / "skills" / "start-here" / "SKILL.md", ROOT / ".agents" / "skills" / "start-here" / "SKILL.md"]
CASE = ROOT / "evals" / "no-later-report"
FACT = "Nothing calls you back"
PROMISE = "A promise of a later report cannot be kept"

CUT = 4096                          # the cut of a tool description in the clients that cut (replies.py)
FIRST_CUT = 2048


def descriptions():
    return {t.name: t.description for t in asyncio.run(server.mcp.list_tools())}


def step5(path: Path) -> str:
    return path.read_text(encoding="utf-8").partition("## Step 5. Watch and report")[2].partition("\n## ")[0]


@pytest.mark.parametrize("tool", ["run_topopt", "run_status", "verify_status", "ratchet_status"])
def test_the_description_says_that_nothing_calls_the_agent_back(tool):
    text = descriptions()[tool]
    assert FACT in text and (PROMISE in text or tool == "run_topopt"), tool       # run_topopt has no room: its reply `message` has the promise sentence
    assert text.index(FACT) < FIRST_CUT, tool                  # a client that cuts early still shows the fact
    assert len(text) <= CUT - 20, tool


def test_the_description_of_run_status_names_the_two_choices():
    text = descriptions()["run_status"]
    assert "call it again in this turn until the state is done, error or stopped, or process_alive is false" in text
    assert "failed or stopped" not in text                       # the top-level state is never `failed` (that is a value of job.state)
    assert "end the turn and tell the user what to ask for the result" in text


def test_the_reply_of_run_topopt_says_it(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass {{yaml}} {{out}}")
    server._current.clear()
    yaml_path = make_case(tmp_path, h=2.0, iterations=2)
    reply = server.run_topopt(str(yaml_path), str(tmp_path / "run"))
    proc = server._current.get("proc")
    if proc is not None:
        proc.wait(timeout=60)
    server._current.clear()
    assert "error" not in reply, reply
    msg = reply["message"]
    assert FACT in msg and PROMISE in msg
    assert "call run_status again in this turn until the state is done, error or stopped, or process_alive is false" in msg
    assert msg == server.no_callback("run_status", server.RUN_END)
    assert "tell the user what to ask for the result" in msg


def test_the_end_states_are_the_states_that_the_tools_give():
    """F2/F6: the words after `until` are values that the tool can give (run_status: no top-level `failed`; the job state has it)."""
    assert server.RUN_END == "the state is done, error or stopped, or process_alive is false"
    assert "failed" not in server.RUN_END
    assert server.VERIFY_END == "the state is done, error or failed"
    ratchet = descriptions()["ratchet_status"]
    from topopt import ratchet as rt
    for state in (*rt._FINAL_STATES, "failed"):                  # every final state of the state file, and the one that status() derives
        assert f"`{state}`" in ratchet and state in server.RATCHET_END, state
    assert f"until {server.RATCHET_END}" in ratchet
    assert "stopped_after_approval" in server.RATCHET_END and server.RATCHET_END.endswith(", or process_alive is false")
    assert "until the state is `done`, `error` or `failed`" in descriptions()["verify_status"]
    assert server.no_callback("x_status", "T", noun="run").startswith("Nothing calls you back when the run ends.")


def test_the_start_messages_of_verify_run_and_ratchet_run_say_it():
    src = (ROOT / "topopt_mcp" / "server.py").read_text(encoding="utf-8")
    assert 'Call verify_status to watch it and to get the result. " + no_callback("verify_status", VERIFY_END)' in src
    assert 'Call ratchet_status to watch it. " + no_callback("ratchet_status", RATCHET_END)' in src
    for tool, end in (("verify_status", server.VERIFY_END), ("ratchet_status", server.RATCHET_END)):
        text = server.no_callback(tool, end)
        assert FACT in text and PROMISE in text and f"call {tool} again in this turn until {end}" in text


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: str(p.relative_to(ROOT)))
def test_step_5_of_start_here_says_it(path):
    text = step5(path)
    assert FACT in text and PROMISE in text
    assert "Keep calling `run_status` in this turn until the state is `done`, `error` or `stopped`, or `process_alive` is false" in text
    assert "tells the user what to ask" in text
    assert "In a client with a shell, wait between calls (for example `sleep 20` in Bash)" in text
    assert "In a client without a shell, call `run_status` again in the same turn" in text


def test_the_two_copies_of_start_here_are_equal():
    assert SKILLS[0].read_bytes() == SKILLS[1].read_bytes()


def test_no_text_names_a_client_product():
    for text in [step5(SKILLS[0]), *(descriptions()[t] for t in ("run_topopt", "run_status", "verify_status", "ratchet_status"))]:
        sentence = text[text.index(FACT):][:600]
        assert not re.search(r"Claude|ChatGPT|Codex|Cursor|Gemini|OpenAI|Anthropic", sentence)


# ---- the eval case ----

def front(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---\n")[1]) or {}


PROMISES = [
    "I'll report the result when the run finishes.",
    "I will let you know when it is done.",
    "Сообщу, когда расчёт закончится.",
    "Как только расчёт закончится, я сообщу вам результат.",
    "Дам знать, когда будет готово.",
    "Started. Will report back once it's done.",
    "The run is going. I'll check back in a minute with the numbers.",
    "Started the run; I'll ping you when it's done.",
    "I'll follow up with the result when the run completes.",
    "Расчёт запущен. Сообщу результат позже.",
    "Запустил расчёт, отпишусь по готовности.",
    "Расчёт идёт, вернусь с результатом через минуту.",
    "Расчёт запущен, результат пришлю, как будет готов.",
    "Started; when it finishes I'll report the numbers.",
    "Запущено. Когда расчёт закончится, сразу сообщу.",
]
FINE = [
    "Расчёт идёт в фоне. Спросите: «что с результатом?», и я покажу.",
    "Масса 12 г, жёсткость в норме, файл final.stl. Допущения: PETG, два шурупа.",
    "The run is going. Ask me \"what is the result?\" to get it.",
    "The run is going in the background (runs/tshirt_hook/run). I'll update the YAML if you want a third screw. Ask me \"what is the result?\" to get it.",
    "Done: mass 29 g, stiffness 4350 N/mm. I'll share the path of final.stl here: runs/tshirt_hook/run/result/final.stl.",
    "I will report the assumptions first: PETG, two 4 mm screws, 0.3 kg hanger. The run is done: 29.3 g, final.stl in runs/tshirt_hook/run/result.",
    "The run is in the background, about 20 s. Ask me \"what is the result?\" as soon as the run finishes.",
    "The run started. The dashboard api will report each pass at http://127.0.0.1:8765/. Ask me \"what is the result?\" to get the numbers.",
    "I'll tell you the assumptions: PETG, two screws, 0.3 kg. Result: 29.3 g, final.stl in runs/x/result.",
    "Расчёт идёт в фоне, около 20 с. Я покажу результат, когда вы спросите «что с результатом?».",
    "Когда расчёт закончится, спросите меня «что с результатом?», и я покажу массу и путь к final.stl.",
    "Сообщу допущения: PETG, два шурупа 4 мм. Расчёт готов: масса 29,3 г, файл final.stl.",
    "Расчёт идёт. Расскажу о результате после того, как вы напишете «что с результатом?».",
    "Ask me once the run is done, and I will send the path.",
    "Результат покажу потом, если спросите.",
    "I will send you the STL later if you want a different screw.",
]
NODE = shutil.which("node")


def grader_pattern():
    g = front(CASE / "graders" / "no-promise.md")
    assert g["type"] == "regex" and g["target"] == "last_message" and g["match"] == "not_contains"
    assert g["flags"] == "i"
    assert not re.match(r"\(\?[a-z]+\)", g["pattern"]), "an inline flag is invalid in JavaScript: use `flags`"
    return g["pattern"]


def matches_in_node(pattern: str, texts: list[str]) -> list[bool]:
    """Compile the pattern as the runner does (JavaScript, flags `i`) and test each text."""
    code = ("const [p, t] = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
            "const re = new RegExp(p, 'i'); console.log(JSON.stringify(t.map(x => re.test(x))));")
    out = subprocess.run([NODE, "-e", code], input=json.dumps([pattern, texts]), capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


def matches(pattern: str, texts: list[str]) -> list[bool]:
    if NODE:
        return matches_in_node(pattern, texts)
    return [re.search(pattern, t, re.I) is not None for t in texts]      # the pattern uses only syntax that Python and JavaScript share


def test_the_grader_fails_a_promise_and_passes_the_other_replies():
    pattern = grader_pattern()
    assert all(matches(pattern, PROMISES)), [t for t, m in zip(PROMISES, matches(pattern, PROMISES)) if not m]
    assert not any(matches(pattern, FINE)), [t for t, m in zip(FINE, matches(pattern, FINE)) if m]


def test_the_grader_pattern_is_valid_in_python_too():
    pattern = grader_pattern()
    assert all(re.search(pattern, t, re.I) for t in PROMISES)
    assert not any(re.search(pattern, t, re.I) for t in FINE)


def regex_graders():
    for f in sorted(EVALS.rglob("graders/*.md")):
        head = front(f)
        if head.get("type") == "regex":
            yield f, head


EVALS = ROOT / "evals"


def test_no_regex_grader_has_an_inline_flag():
    """F1: the runner compiles a pattern as JavaScript, where `(?i)` is invalid. A flag goes in the key `flags`."""
    found = list(regex_graders())
    assert found
    for f, head in found:
        assert not re.search(r"\(\?[a-zA-Z]+[:)]", head["pattern"].replace("(?:", "")), f
        assert re.fullmatch(r"[gimsuy]*", str(head.get("flags", ""))), f


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_every_regex_grader_compiles_in_javascript():
    for f, head in regex_graders():
        code = "const [p, f] = JSON.parse(require('fs').readFileSync(0, 'utf8')); new RegExp(p, f);"
        done = subprocess.run([NODE, "-e", code], input=json.dumps([head["pattern"], str(head.get("flags", ""))]), capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, (f, done.stderr[-300:])


def test_the_case_has_its_files_tags_and_a_prompt_with_no_word_about_waiting():
    head = front(CASE / "prompt.md")
    assert {"smoke", "mock", "flow"} <= set(head["tags"])
    assert sorted(p.name for p in CASE.iterdir()) == ["graders", "mocks", "prompt.md"]
    assert sorted(p.name for p in (CASE / "graders").iterdir()) == ["calls-run-topopt.md", "ends-with-result-or-ask.md", "no-promise.md"]
    prompt = CASE.joinpath("prompt.md").read_text(encoding="utf-8").split("---\n", 2)[2]
    assert re.search(r"[А-Яа-я]{4}", prompt) and "запусти" in prompt
    assert not re.search(r"подожд|жд[иу]|позже|потом|когда закончит", prompt, re.I)


def test_the_step_grader_needs_the_run():
    g = front(CASE / "graders" / "calls-run-topopt.md")
    assert g == {"type": "tool_used", "tool": "mcp__plugin_topopt_topopt__run_topopt", "min": 1}


def test_the_case_has_its_own_mock_of_run_status_that_goes_on_first():
    mock = CASE / "mocks" / "topopt" / "run_status.md"
    assert front(mock)["type"] == "agent"
    text = mock.read_text(encoding="utf-8")
    assert 'State "running", process_alive true' in text and "first two calls" in text           # the first calls go on
    assert "The third call and later: the run is DONE" in text and 'state "done"' in text         # the third call ends
    assert text.index('State "running", process_alive true') < text.index("The third call and later")
    assert "{{file:fixtures/run_status.json}}" in text and (CASE / "mocks" / "topopt" / "fixtures" / "run_status.json").is_file()
    desc = front(CASE / "prompt.md")["description"]
    assert "mocks/topopt/run_status.md" in desc and "running for the first two calls and done from the third" in desc
    assert "on the suite mocks" not in desc


def test_the_llm_grader_accepts_a_report_or_a_plain_ask_and_fails_a_promise():
    g = CASE / "graders" / "ends-with-result-or-ask.md"
    assert front(g)["type"] == "llm" and front(g)["focus"] == "last_message"
    rubric = g.read_text(encoding="utf-8").split("---\n", 2)[2]
    for phrase in ("PASS if", "FAIL if", "report of the result", "what to ask", "I will report when the run finishes"):
        assert phrase in rubric, phrase


def test_the_readme_has_the_row_and_the_note():
    text = (ROOT / "evals" / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^\| `no-later-report` \| smoke, mock, flow, run \|", text, re.M)
    assert "`no-later-report` has no history" in text and "answers `running` for the first two" in text


def test_the_mock_of_run_topopt_has_the_message_of_the_real_reply():
    fixture = json.loads((ROOT / "evals" / "mocks" / "topopt" / "fixtures" / "run_topopt.json").read_text(encoding="utf-8"))
    assert fixture["message"] == server.no_callback("run_status", server.RUN_END)


def test_the_case_files_hold_no_machine_path_and_no_account_fact():
    for f in sorted(CASE.rglob("*")):
        if f.is_file():
            assert not re.search(r"/app/|/home/|/tmp/|/workspace/|@", f.read_text(encoding="utf-8")), f
