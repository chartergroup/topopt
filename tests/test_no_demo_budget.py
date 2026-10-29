"""P9-8 (D5): the engine has no time budget of its own. An old "240 s demo budget" is gone.

The only time budget is the one of the user, `settings.stop_rules.time_budget_s`. Without it the time estimate is a fact in plain
words and there is no budget warning. With it, `topopt check`, `check_problem` and the part builder compare the estimate with it,
in the same warning shape as before (and the voxel size where the estimate is the budget).
"""
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_optimize import make_case  # noqa: E402

pytest.importorskip("mcp")
from topopt import cli, regions_builder  # noqa: E402
from topopt.cli import check_data, fit_voxel_size, main  # noqa: E402
from topopt_mcp import server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# 0.4 mm voxels, 300 passes: the model estimate is more than 240 s, so the old fixed budget gave a warning for every user.
BIG = dict(h=0.4, iterations=300)


def case_with(tmp_path, **settings):
    p = make_case(tmp_path, **BIG)
    if settings:
        d = yaml.safe_load(p.read_text())
        d["settings"].update(settings)
        p.write_text(yaml.safe_dump(d))
    return p


def time_budget_warnings(warnings):
    return [w for w in warnings if w.startswith(("the time estimate is", "the gpu-mg time estimate", "the time estimate for a run with"))]


def test_no_budget_in_the_yaml_gives_the_estimate_as_a_fact_and_no_warning(tmp_path):
    d = check_data(str(case_with(tmp_path)))
    te = d["time_estimate"]
    assert te["total_s"] > 240 and te["limit_s"] is None                    # the old code warned here, for every user
    assert time_budget_warnings(d["warnings"]) == []
    text = json.dumps(d)
    assert "demo budget" not in text and "conservative" not in text and "240 s" not in text


def test_topopt_check_without_a_budget_says_the_time_in_plain_words_and_nothing_about_a_budget(tmp_path, capsys):
    p = case_with(tmp_path)
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert re.search(r"time estimate:  [\d.]+ s/iteration, about \d+ s in total", out)
    assert "your time budget" not in out and not OLD_WORDS.search(out) and "conservative" not in out and "240 s" not in out   # the words of the message, not the paths in the output (N3)
    assert main(["check", str(p), "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["time_estimate"]["limit_s"] is None and time_budget_warnings(j["warnings"]) == []


def test_check_problem_without_a_budget_has_no_budget_text(tmp_path):
    r = server.check_problem(str(case_with(tmp_path)))
    assert r["ok"] and r["time_estimate"]["total_s"] > 240 and r["time_estimate"]["limit_s"] is None
    text = json.dumps(r)
    assert "demo budget" not in text and "conservative" not in text and "240 s" not in text
    assert time_budget_warnings(r["warnings"]) == []


def test_a_budget_of_the_user_is_compared_with_the_estimate_and_names_the_budget_and_the_voxel(tmp_path, capsys):
    p = case_with(tmp_path, stop_rules={"time_budget_s": 30})
    d = check_data(str(p))
    te = d["time_estimate"]
    assert te["limit_s"] == 30 and te["total_s"] > 30
    (w,) = time_budget_warnings(d["warnings"])
    fit = fit_voxel_size(0.4, te["total_s"], 30, d["solver"]["resolved"])
    assert w.startswith(f"the time estimate is {te['total_s']:.0f} s, above your time budget of 30 s (settings.stop_rules.time_budget_s)")
    assert f"; voxel_size about {fit} mm would fit" in w and fit > 0.4
    assert "demo" not in w and "conservative" not in w
    assert main(["check", str(p)]) == 0                                      # `topopt check` prints the same warning
    assert f"warning: {w}" in capsys.readouterr().out
    r = server.check_problem(str(p))                                         # and so does the MCP tool
    assert r["time_estimate"]["limit_s"] == 30 and w in r["warnings"]


def test_an_estimate_below_the_budget_of_the_user_gives_no_warning_and_the_budget_is_in_limit_s(tmp_path):
    d = check_data(str(case_with(tmp_path, stop_rules={"time_budget_s": 1e6})))
    assert d["time_estimate"]["limit_s"] == 1e6 and time_budget_warnings(d["warnings"]) == []


def test_the_budget_warning_for_a_run_with_stress_rows_names_the_budget_of_the_user(tmp_path):
    p = case_with(tmp_path, stop_rules={"time_budget_s": 30})
    d = yaml.safe_load(p.read_text())
    d["load_cases"][0]["stress_limit"] = 5.0
    p.write_text(yaml.safe_dump(d))
    (w,) = time_budget_warnings(check_data(str(p))["warnings"])
    assert w.startswith("the time estimate for a run with stress rows is ") and "above your time budget of 30 s (settings.stop_rules.time_budget_s) at the high end" in w
    assert "demo" not in w and "conservative" not in w


# --- MJ1 (P9-8 review): `time_budget_s` is a stop rule, and every budget warning says what it does in this YAML ---------------

ON_WORDS = ("the stop rules are on in this YAML (a stop_rules mapping turns them on), so the run stops at the first pass that ends "
            "after {b} s of run time (rule time_budget; the rule converged is on too) and the part is the design of that pass; "
            "--no-stop-rules (MCP run_topopt(stop_rules=false)) turns the rules off for one run")
OFF_WORDS = ("the stop rules are off in this YAML (enabled: false), so the run does not stop at this budget; "
             "--stop-rules (MCP run_topopt(stop_rules=true)) turns them on for one run")


def budget_warning(p):
    (w,) = time_budget_warnings(check_data(str(p))["warnings"])
    return w


@pytest.mark.parametrize("stress", [False, True], ids=["plain", "stress rows"])
def test_the_budget_warning_says_that_the_run_stops_when_the_stop_rules_are_on(tmp_path, stress):
    p = case_with(tmp_path, stop_rules={"time_budget_s": 30})              # a mapping turns the stop rules on
    if stress:
        d = yaml.safe_load(p.read_text())
        d["load_cases"][0]["stress_limit"] = 5.0
        p.write_text(yaml.safe_dump(d))
    w = budget_warning(p)
    assert ON_WORDS.format(b=30) in w
    assert "does not stop" not in w and "the stop rules are off" not in w
    assert w.index("rule time_budget") < w.index("voxel_size about")        # the fact, then the effect, then the offers


@pytest.mark.parametrize("stress", [False, True], ids=["plain", "stress rows"])
def test_the_budget_warning_says_that_the_run_does_not_stop_when_the_stop_rules_are_off(tmp_path, stress):
    p = case_with(tmp_path, stop_rules={"enabled": False, "time_budget_s": 30})      # the budget is set, the rules are off
    if stress:
        d = yaml.safe_load(p.read_text())
        d["load_cases"][0]["stress_limit"] = 5.0
        p.write_text(yaml.safe_dump(d))
    w = budget_warning(p)
    assert OFF_WORDS in w and "above your time budget of 30 s (settings.stop_rules.time_budget_s)" in w
    assert "the run stops" not in w and "rule time_budget" not in w and "the stop rules are on" not in w


def test_the_words_of_the_budget_warning_are_true_in_a_real_run(tmp_path):
    """One small CPU run for each case (30 passes at 2 mm): a mapping stops the run at the budget, `enabled: false` does not."""
    seen = {}
    for name, rules in (("on", {"time_budget_s": 0.05}), ("off", {"enabled": False, "time_budget_s": 0.05})):
        folder = tmp_path / name
        folder.mkdir()
        p = make_case(folder, h=2.0, iterations=30)
        d = yaml.safe_load(p.read_text())
        d["settings"]["stop_rules"] = rules
        p.write_text(yaml.safe_dump(d))
        w = budget_warning(p)
        out = folder / "run"
        assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
        seen[name] = (w, json.loads((out / "verdict.json").read_text()))
    w, v = seen["on"]
    assert ON_WORDS.format(b=0.05) in w
    assert v["state"] == "stopped_by_rule" and v["stop"]["rule"] == "time_budget" and v["final_iteration"] < 30
    w, v = seen["off"]
    assert OFF_WORDS in w
    assert v["state"] == "iteration_limit" and v["final_iteration"] == 30 and v["stop"] is None


SKILL_FILES = [".claude/skills/part-to-topopt/SKILL.md", ".agents/skills/part-to-topopt/SKILL.md",
               "skills/start-here/SKILL.md", ".agents/skills/start-here/SKILL.md"]


def test_no_skill_says_accept_the_longer_run_and_the_rows_give_the_true_options():
    """The budget does not make a run longer: with the stop rules on it ends the run, with `enabled: false` it does nothing."""
    for name in SKILL_FILES:
        assert "accept the longer run" not in (ROOT / name).read_text(encoding="utf-8").lower(), name
    for name in SKILL_FILES[:2]:                                             # the part skill and its copy
        text = (ROOT / name).read_text(encoding="utf-8")
        (row,) = [ln for ln in text.splitlines() if ln.startswith("| `WARNING: the estimate is above your time budget of")]
        assert "It is a stop rule, not a note" in row and "rule `time_budget`" in row and "the design of that pass" in row, name
        assert "With `enabled: false` the budget is set but the rules are off, so the run does not stop at it" in row, name
        assert "a larger voxel" in row and "another budget (ask the user)" in row and "all passes with the stop rules off" in row, name
        assert "`enabled: false` in the YAML, or `run_topopt(stop_rules=false)` for one run" in row, name
        (bullet,) = [ln for ln in text.splitlines() if ln.startswith("- When the YAML sets `settings.stop_rules.time_budget_s`")]
        assert "`time_budget_s` is a stop rule" in bullet and "stops at the first pass that ends after the budget" in bullet, name
        assert "does not stop the build" in bullet and "the post step" in bullet and "With `enabled: false` the run does not stop" in bullet, name
        assert "and it does not stop." not in bullet, name                   # the old sentence said that the warning is all there is


def test_contracts_section_11_says_that_the_budget_is_a_stop_rule_and_names_its_clock():
    text = " ".join((ROOT / "CONTRACTS.md").read_text(encoding="utf-8").split())
    sec = text[text.index("## 11. Time model and the time budget"):text.index("## 12.")]
    assert "The budget is also a stop rule (section 24.2), not a note" in sec and "`cli.budget_effect`" in sec
    assert "the run stops at the first pass that ends after the budget (rule `time_budget`), and the part is the design of that pass" in sec
    assert "`enabled: false`: the budget is set but the rules are off, so the run does not stop at it" in sec
    assert "`topopt run --no-stop-rules` (MCP `run_topopt(stop_rules=false)`)" in sec
    assert "The clock of the rule is not the clock of the estimate" in sec and "It does not count the post step after the last pass" in sec
    assert "The estimate counts `SETUP_S`, the passes and `POST_S`" in sec


def test_the_root_readme_says_that_the_code_panel_is_for_the_talk_screen_only():
    """P9-8 review MN1: the page shows the optimizer code and the step times only for a `pull_off` case or with `?demo=1`."""
    text = (ROOT / "docs" / "reference" / "dashboard-mcp-render.md").read_text(encoding="utf-8")
    assert "The page shows the 3D shape, the log, and a bottom bar." in text
    assert "The optimizer code with the active step shows only on the demo screen: for a run with a load case named `pull_off`, or when the address has `?demo=1`." in text
    assert "the optimizer code with the active step, the log" not in text
    assert "The step times in the code panel (it shows only for a `pull_off` case or with `?demo=1`) are not recorded" in text
    assert "The step times in the code panel are not recorded" not in text


def test_the_e4_runbook_expects_no_budget_warning_for_the_hook(capsys):
    """P9-8 review N4: the hook example sets no `time_budget_s`, so `topopt check` prints no budget warning for it."""
    assert "time_budget_s" not in (ROOT / "examples" / "hook" / "hook.yaml").read_text(encoding="utf-8")
    assert main(["check", str(ROOT / "examples" / "hook" / "hook.yaml")]) == 0
    out = capsys.readouterr().out
    assert "time estimate:" in out and "your time budget" not in out


def test_the_engine_holds_no_budget_of_its_own():
    assert not hasattr(cli, "LIMIT_S") and not hasattr(regions_builder, "LIMIT_S")
    with pytest.raises(TypeError):
        fit_voxel_size(1.0, 300.0)                                           # the limit is a required argument: no default budget
    assert fit_voxel_size(1.0, 25 + 4 * 215.0, 240.0) is not None            # the function itself works for any limit


# --- no user-facing text says "demo budget" -------------------------------------------------------------------------------

SCANNED = ("topopt", "topopt_mcp", "skills", ".claude/skills", ".agents/skills", "agents", "evals/mocks", "evals/voxel-guard/mocks",
           "evals/lost-force/mocks", "evals/print-after-run/mocks")
SUFFIXES = {".py", ".md", ".json", ".html", ".js", ".css", ".yaml", ".yml", ".sh", ".toml", ".txt"}
OLD_WORDS = re.compile(r"demo( time)?[ -]budget|conservative (time )?estimate|conservative number", re.IGNORECASE)   # "demo time budget": the old comment of `LIMIT_S`


def test_no_text_for_a_user_or_an_agent_says_demo_budget_or_calls_the_estimate_conservative():
    found = []
    for folder in SCANNED:
        for f in sorted((ROOT / folder).rglob("*")):
            if not f.is_file() or f.suffix not in SUFFIXES or "vendor" in f.parts or "__pycache__" in f.parts:
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if OLD_WORDS.search(line):
                    found.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
    assert not found, "\n".join(found)
    # the descriptions and the instructions that the MCP server sends
    from topopt_mcp.server import INSTRUCTIONS, mcp
    import asyncio
    texts = [INSTRUCTIONS] + [t.description or "" for t in asyncio.run(mcp.list_tools())]
    assert not [t[:60] for t in texts if OLD_WORDS.search(t)]
