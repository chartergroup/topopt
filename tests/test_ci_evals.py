"""The manual CI run of the plugin evals: `.github/workflows/evals.yml`, the composite action `.github/actions/plugin-eval` and
`scripts/ci/run_evals.py` (CONTRACTS.md 27.1, evals/README.md "Run the evals in CI").

These tests make no model call and no network access. A fake `claude` on PATH stands in for `claude plugin eval`.
"""
import argparse
import copy
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / "scripts" / "ci"
sys.path.insert(0, str(CI))

import _common  # noqa: E402
import check_workflow  # noqa: E402
import run_evals  # noqa: E402

WORKFLOW = ROOT / ".github" / "workflows" / "evals.yml"
ACTION = ROOT / ".github" / "actions" / "plugin-eval" / "action.yml"
LEGS = set(_common.load_spec()["legs"])
START = "gh workflow run evals.yml --repo chartergroup/topopt"


def _wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _action() -> dict:
    return yaml.safe_load(ACTION.read_text())


def _args(**kw) -> list[str]:
    base = {"model": "claude-sonnet-5-5", "judge-model": "claude-haiku-5-5"}
    base.update(kw)
    out = []
    for k, v in base.items():
        out += [f"--{k}", v]
    return out


def _cmd(**kw) -> list[str]:
    """build_command with the defaults of the script and the given values (keys as in the argparse namespace)."""
    values = {"model": "claude-sonnet-5-5", "judge_model": "claude-haiku-5-5", "threshold": run_evals.DEFAULT_THRESHOLD,
              "tag": run_evals.DEFAULT_TAG, "case": "", "runs": run_evals.DEFAULT_RUNS, "concurrency": run_evals.DEFAULT_CONCURRENCY,
              "max_cost_usd": run_evals.DEFAULT_MAX_COST_USD, "output_dir": run_evals.DEFAULT_OUTPUT_DIR}
    values.update(kw)
    return run_evals.build_command(argparse.Namespace(**values))


# --- the workflow file -----------------------------------------------------------------------------

def test_the_evals_workflow_passes_the_checks_and_starts_by_hand_only():
    doc = _wf()
    assert check_workflow.check_workflow(doc, ROOT, LEGS, "evals.yml") == []
    assert check_workflow.main([str(WORKFLOW)]) == 0
    on = check_workflow._on(doc)
    assert set(on) == {"workflow_dispatch"}          # each case is a paid model call: no push, pull_request or schedule
    assert check_workflow.event_rule("evals.yml") == (("workflow_dispatch",), True)


def test_the_model_the_judge_model_and_the_threshold_are_inputs_with_the_defaults_of_the_script():
    inputs = check_workflow._on(_wf())["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"model", "judge_model", "threshold", "tag", "case", "runs", "max_cost_usd"}
    assert inputs["model"]["default"] == "claude-sonnet-5-5" and inputs["judge_model"]["default"] == "claude-haiku-5-5"
    assert inputs["threshold"]["default"] == run_evals.DEFAULT_THRESHOLD and "0: no score gate" in inputs["threshold"]["description"]
    assert inputs["tag"]["type"] == "choice" and tuple(inputs["tag"]["options"]) == run_evals.TAGS
    assert inputs["tag"]["default"] == run_evals.DEFAULT_TAG
    assert inputs["case"]["default"] == ""
    assert inputs["runs"]["default"] == run_evals.DEFAULT_RUNS
    assert inputs["max_cost_usd"]["default"] == run_evals.DEFAULT_MAX_COST_USD
    assert all(v.get("description") for v in inputs.values())


def test_the_workflow_gives_every_input_and_both_secrets_to_the_action():
    steps = _wf()["jobs"]["evals"]["steps"]
    step = [s for s in steps if s.get("uses") == "./.github/actions/plugin-eval"]
    assert len(step) == 1 and steps[0]["uses"].startswith("actions/checkout@")       # the action reads the scripts of the checkout
    w = step[0]["with"]
    assert w["model"] == "${{ inputs.model }}" and w["judge-model"] == "${{ inputs.judge_model }}"
    assert w["threshold"] == "${{ inputs.threshold }}"
    for name in check_workflow._on(_wf())["workflow_dispatch"]["inputs"]:
        assert f"${{{{ inputs.{name} }}}}" in w.values(), name
    assert w["anthropic-api-key"] == "${{ secrets.ANTHROPIC_API_KEY }}"
    assert w["claude-code-oauth-token"] == "${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}"
    assert set(w) <= set(_action()["inputs"])


def test_the_header_says_manual_only_and_how_to_start_it():
    head = "\n".join(WORKFLOW.read_text().splitlines()[:16])
    assert "MANUAL ONLY" in head and START in head and "threshold 0" in head and "(estimate" in head
    assert "ANTHROPIC_API_KEY" in head and "CLAUDE_CODE_OAUTH_TOKEN" in head


def test_the_concurrency_group_never_cancels_a_run():
    c = _wf()["concurrency"]
    assert c["cancel-in-progress"] is False and "github.sha" in c["group"]


# --- the composite action --------------------------------------------------------------------------

def test_the_action_passes_the_checks():
    assert check_workflow.check_action(_action(), ROOT) == []
    assert check_workflow.main([str(ACTION)]) == 0


def test_the_action_defaults_are_the_defaults_of_the_script():
    inputs = _action()["inputs"]
    assert inputs["model"]["required"] is True and inputs["judge-model"]["required"] is True
    want = {"threshold": run_evals.DEFAULT_THRESHOLD, "tag": run_evals.DEFAULT_TAG, "case": "", "runs": run_evals.DEFAULT_RUNS,
            "concurrency": run_evals.DEFAULT_CONCURRENCY, "max-cost-usd": run_evals.DEFAULT_MAX_COST_USD,
            "output-dir": run_evals.DEFAULT_OUTPUT_DIR}
    assert {k: inputs[k]["default"] for k in want} == want
    assert re.fullmatch(r"\d+\.\d+\.\d+", inputs["claude-version"]["default"])   # CI installs a fixed version


def test_the_run_step_gives_each_input_to_the_script_through_the_environment():
    steps = _action()["runs"]["steps"]
    run = [s for s in steps if "scripts/ci/run_evals.py" in s.get("run", "")]
    assert len(run) == 1
    st = run[0]
    assert len(st["run"].splitlines()) == 1 and "${{" not in st["run"]          # no expression in the shell text (no injection)
    env = st["env"]
    for name in _action()["inputs"]:
        if name in ("anthropic-api-key", "claude-code-oauth-token", "artifact-name", "claude-version"):
            continue
        var = "IN_" + name.upper().replace("-", "_")
        assert env[var] == f"${{{{ inputs.{name} }}}}", name
        assert f'--{name} "${var}"' in st["run"], name
    assert "IN_CLAUDE_VERSION" not in env and "--claude-version" not in st["run"]   # only the install step uses the version
    assert env["ANTHROPIC_API_KEY"] == "${{ inputs.anthropic-api-key }}"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "${{ inputs.claude-code-oauth-token }}"
    for s in steps:                       # `claude plugin eval` passes EVAL_* to the agent under test
        assert not any(k.startswith("EVAL_") for k in (s.get("env") or {})), s["name"]


def test_the_action_installs_the_pinned_claude_code_with_node_22():
    steps = _action()["runs"]["steps"]
    node = [s for s in steps if str(s.get("uses", "")).startswith("actions/setup-node@")]
    assert len(node) == 1 and node[0]["with"]["node-version"] == "22"
    install = [s for s in steps if "@anthropic-ai/claude-code@" in s.get("run", "")]
    assert len(install) == 1 and install[0]["env"]["IN_CLAUDE_VERSION"] == "${{ inputs.claude-version }}"


def test_the_upload_runs_after_a_failed_run_and_uploads_the_output_folder():
    up = [s for s in _action()["runs"]["steps"] if str(s.get("uses", "")).startswith("actions/upload-artifact@")]
    assert len(up) == 1 and up[0]["if"] == "${{ !cancelled() }}"
    assert up[0]["with"]["path"] == "${{ inputs.output-dir }}" and up[0]["with"]["retention-days"] == 14


@pytest.mark.parametrize("edit, text", [
    (lambda d: d["runs"].update({"using": "node20"}), "`runs.using` must be composite"),
    (lambda d: d["runs"]["steps"][1].pop("shell"), "needs a `shell`"),
    (lambda d: d["runs"]["steps"][1]["env"].update({"X": "${{ inputs.nothing }}"}), "`inputs.nothing` is not an input"),
    (lambda d: d["inputs"]["model"].pop("description"), "input model: no `description`"),
    (lambda d: d["runs"]["steps"][2].update({"run": "python3 scripts/ci/nope.py"}), "scripts/ci/nope.py does not exist"),
    (lambda d: d["runs"]["steps"][3].update({"if": "success()"}), "does not run on failure"),
    (lambda d: d.update({"branding2": 1}), "unknown top-level key 'branding2'"),
])
def test_the_action_check_finds_these_mistakes(edit, text):
    doc = copy.deepcopy(_action())
    edit(doc)
    problems = check_workflow.check_action(doc, ROOT)
    assert any(text in p for p in problems), (text, problems)


def test_the_workflow_check_finds_a_missing_local_action_and_an_unknown_input():
    doc = copy.deepcopy(_wf())
    doc["jobs"]["evals"]["steps"][1]["uses"] = "./.github/actions/no-such-action"
    doc["jobs"]["evals"]["steps"][1]["with"]["model"] = "${{ inputs.modell }}"
    problems = check_workflow.check_workflow(doc, ROOT, LEGS, "evals.yml")
    assert any("has no action.yml" in p for p in problems) and any("`inputs.modell` is not an input" in p for p in problems), problems
    doc = copy.deepcopy(_wf())
    doc[True]["push"] = None
    assert any("the event push" in p for p in check_workflow.check_workflow(doc, ROOT, LEGS, "evals.yml"))


# --- run_evals.py: the command ---------------------------------------------------------------------

def _value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


@pytest.mark.parametrize("given, passed", [("0", "0"), ("0.0", "0"), ("0.8", "0.8"), ("1", "1"), ("0.75", "0.75")])
def test_the_threshold_goes_to_the_runner_as_given_and_0_is_kept(given, passed):
    assert _value(_cmd(threshold=given), "--threshold") == passed


def test_the_models_go_to_the_runner_and_only_the_mocked_cases_run():
    cmd = _cmd(model="claude-opus-5-5", judge_model="haiku")
    assert cmd[:4] == ["claude", "plugin", "eval", "."]
    assert _value(cmd, "--model") == "claude-opus-5-5" and _value(cmd, "--judge-model") == "haiku"
    assert _value(cmd, "--mocks") == "record" and _value(cmd, "--ablation") == "none" and _value(cmd, "--tag") == "smoke"
    for flag in ("--scaffold", "--trust-plugin", "--no-publish"):
        assert flag in cmd
    for flag in ("--allow-tools", "--allow-real-servers", "--keep-temp", "--publish-report"):
        assert flag not in cmd
    assert "--case" not in cmd
    assert _value(_cmd(case="units-kg"), "--case") == "units-kg"
    assert _value(_cmd(tag="mock", runs="3"), "--runs") == "3"


@pytest.mark.parametrize("kw, text", [
    ({"threshold": "-0.1"}, "not in the range 0 to 1"),
    ({"threshold": "1.5"}, "not in the range 0 to 1"),
    ({"threshold": "abc"}, "not a number"),
    ({"threshold": ""}, "not a number"),
    ({"tag": "e2e"}, "not a mocked tag"),
    ({"runs": "0"}, "runs '0' is not in the range 1 to 50"),
    ({"runs": "x"}, "not an integer"),
    ({"concurrency": "9"}, "not in the range 1 to 8"),
    ({"max_cost_usd": "0"}, "must be more than 0"),
    ({"max_cost_usd": "lots"}, "not a number"),
    ({"model": ""}, "is not a model name"),
    ({"model": "sonnet; rm -rf /"}, "is not a model name"),
    ({"judge_model": "$(id)"}, "is not a model name"),
    ({"case": "a;b"}, "not a glob of case names"),
])
def test_a_bad_input_stops_the_step_with_its_name(kw, text):
    with pytest.raises(_common.CiError, match=text.replace("(", r"\(").replace(")", r"\)")):
        _cmd(**kw)


def test_dry_run_prints_the_command_and_the_gate_and_needs_no_credential(capsys, monkeypatch):
    for k in run_evals.CREDENTIALS:
        monkeypatch.delenv(k, raising=False)
    assert run_evals.main(_args(threshold="0") + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--threshold 0" in out and "score gate: off" in out


def test_in_actions_the_credential_is_needed_and_an_empty_variable_is_removed():
    gha = {"GITHUB_ACTIONS": "true"}
    with pytest.raises(_common.CiError, match="no credential"):
        run_evals.credential_env({**gha, "PATH": "/bin"})
    with pytest.raises(_common.CiError, match="no credential"):
        run_evals.credential_env({**gha, "ANTHROPIC_API_KEY": "", "CLAUDE_CODE_OAUTH_TOKEN": " "})
    env = run_evals.credential_env({**gha, "ANTHROPIC_API_KEY": "", "CLAUDE_CODE_OAUTH_TOKEN": "tok", "PATH": "/bin"})
    assert "ANTHROPIC_API_KEY" not in env and env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok" and env["PATH"] == "/bin"


def test_outside_actions_claude_may_use_its_own_login():
    env = run_evals.credential_env({"PATH": "/bin", "ANTHROPIC_API_KEY": ""})
    assert env == {"PATH": "/bin"}


# --- run_evals.py with a fake claude ---------------------------------------------------------------

FAKE_CLAUDE = textwrap.dedent('''\
    #!{python}
    import json, os, sys
    from pathlib import Path
    args = sys.argv[1:]
    if args == ["--version"]:
        print(os.environ.get("FAKE_VERSION", "2.1.303") + " (Claude Code)")
        sys.exit(0)
    log = Path(os.environ["FAKE_LOG"])
    log.write_text(json.dumps({{"argv": args, "api_key_set": "ANTHROPIC_API_KEY" in os.environ, "cwd": os.getcwd()}}))
    out = Path(args[args.index("--output-dir") + 1])
    result = os.environ.get("FAKE_RESULT")
    if result:
        out.mkdir(parents=True, exist_ok=True)
        (out / "aggregate-result.json").write_text(result)
    sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
''')


def _result(scores: dict, partial: str | None = None) -> str:
    cases = []
    for name, (score, error) in scores.items():
        cases.append({"name": name, "aggregates": {"score": score},
                      "arms": {"with": [{"score": score, "error": error, "passed": score >= 0.8}]}})
    return json.dumps({"schemaVersion": 1, "claudeVersion": "2.1.303", "costUsd": 1.25, "durationSeconds": 99,
                       "partial": partial is not None, "partialReason": partial,
                       "aggregates": {"overallScore": sum(s for s, _ in scores.values()) / len(scores),
                                      "casesPassed": sum(s >= 0.8 for s, _ in scores.values()), "casesTotal": len(scores)},
                       "cases": cases})


@pytest.fixture
def fake(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude = bin_dir / "claude"
    claude.write_text(FAKE_CLAUDE.format(python=sys.executable))
    claude.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    out = tmp_path / "results"

    class Fake:
        def run(self, threshold="0.8", exit_code=0, result=None, **kw):
            monkeypatch.setenv("FAKE_EXIT", str(exit_code))
            if result is None:
                monkeypatch.delenv("FAKE_RESULT", raising=False)
            else:
                monkeypatch.setenv("FAKE_RESULT", result)
            return run_evals.main(_args(threshold=threshold, **{"output-dir": str(out)}, **kw))

        def log(self):
            return json.loads((tmp_path / "log.json").read_text())

        def summary(self):
            return summary.read_text() if summary.is_file() else ""

    Fake.out = out
    return Fake()


def test_a_pass_writes_the_summary_table_and_returns_0(fake):
    assert fake.run(result=_result({"units-kg": (1.0, None), "off-topic": (0.9, None)})) == 0
    s = fake.summary()
    assert "| units-kg | 1.00 | 1 | pass |" in s and "| off-topic | 0.90 | 1 | pass |" in s
    assert "model `claude-sonnet-5-5`" in s and "judge `claude-haiku-5-5`" in s and "score gate 0.8" in s and "$1.25" in s
    log = fake.log()
    assert _value(log["argv"], "--threshold") == "0.8" and log["api_key_set"] and Path(log["cwd"]) == ROOT


def test_a_case_below_the_threshold_fails_the_step(fake):
    with pytest.raises(_common.CiError, match="exit code 1: a case is below the threshold"):
        fake.run(exit_code=1, result=_result({"units-kg": (0.5, None)}))
    assert "| units-kg | 0.50 | 1 | FAIL |" in fake.summary()


def test_threshold_0_turns_the_score_gate_off(fake):
    assert fake.run(threshold="0", result=_result({"units-kg": (0.0, None)})) == 0
    assert _value(fake.log()["argv"], "--threshold") == "0"
    s = fake.summary()
    assert "score gate off (threshold 0)" in s and "| units-kg | 0.00 | 1 | - |" in s


def test_threshold_0_still_fails_when_the_run_fails(fake):
    with pytest.raises(_common.CiError, match="exit code 1: a case did not load, or a run ended on an API error \\(the score gate is off"):
        fake.run(threshold="0", exit_code=1, result=_result({"units-kg": (0.0, "API error 529")}))
    assert "API error 529" in fake.summary()


def test_a_partial_run_fails_and_names_the_reason(fake):
    with pytest.raises(_common.CiError, match="exit code 2: partial run"):
        fake.run(exit_code=2, result=_result({"units-kg": (1.0, None)}, partial="cost_ceiling"))
    assert "**Partial run:** cost_ceiling." in fake.summary()


def test_an_old_result_file_does_not_stand_for_a_run_that_wrote_none(fake):
    fake.out.mkdir(parents=True)
    (fake.out / "aggregate-result.json").write_text(_result({"old-case": (1.0, None)}))
    with pytest.raises(_common.CiError, match="exit code 1"):
        fake.run(exit_code=1)
    s = fake.summary()
    assert "No aggregate-result.json" in s and "old-case" not in s


@pytest.mark.parametrize("version", ["2.1.269", "2.1.311", "3.0.0"])
def test_any_claude_version_runs_and_is_printed(fake, monkeypatch, capsys, version):
    monkeypatch.setenv("FAKE_VERSION", version)
    assert fake.run(result=_result({"units-kg": (1.0, None)})) == 0
    assert f"claude: {version} (Claude Code)" in capsys.readouterr().out
    assert Path(os.environ["FAKE_LOG"]).exists()


def test_no_claude_on_path_stops_before_the_run(fake, monkeypatch, tmp_path):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(_common.CiError, match="cannot run `claude --version`"):
        fake.run(result=_result({"units-kg": (1.0, None)}))
    assert not Path(os.environ["FAKE_LOG"]).exists()


def test_no_credential_in_actions_stops_before_the_run(fake, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(_common.CiError, match="no credential"):
        fake.run(result=_result({"units-kg": (1.0, None)}))
    assert not Path(os.environ["FAKE_LOG"]).exists()


def test_the_summary_names_a_mock_abort():
    result = json.loads(_result({"voxel-guard": (0.0, None)}))
    result["cases"][0]["arms"]["with"][0]["aborted"] = {"server": "topopt", "tool": "build_regions", "reason": "expect: voxel_mm"}
    assert "mock abort: build_regions: expect: voxel_mm" in run_evals.summary(result, 0.8, "m", "j")


# --- the docs --------------------------------------------------------------------------------------

def test_the_readme_and_contracts_say_how_to_start_the_eval_workflow():
    readme = (ROOT / "evals" / "README.md").read_text()
    assert START in readme and "threshold" in readme and "`0`" in readme and ".github/actions/plugin-eval" in readme
    contracts = (ROOT / "CONTRACTS.md").read_text()
    sec = contracts[contracts.index("### 27.1 Files"):contracts.index("### 27.2 Legs")]
    assert "`.github/workflows/evals.yml`" in sec and "`.github/actions/plugin-eval/action.yml`" in sec
