#!/usr/bin/env python3
"""Run the mocked eval cases of the topopt plugin (`claude plugin eval`), write a table to the job summary, and give the exit code.

    python3 scripts/ci/run_evals.py --model claude-sonnet-5-5 --judge-model claude-haiku-5-5 --threshold 0.8
    python3 scripts/ci/run_evals.py --model sonnet --judge-model haiku --threshold 0 --tag mock --runs 3 --dry-run

The run step of `.github/actions/plugin-eval/action.yml` calls this script. The workflow `.github/workflows/evals.yml` starts the action
by hand only. `evals/README.md` tells what the cases test.

- Only the mocked cases run: the tag is `smoke` or `mock`, and `--mocks record` never starts the real topopt server. The case `hook-e2e`
  (tag `e2e`) needs the real server and one optimizer job, so it is not for CI.
- The models are inputs: `--model` (the agent under test) and `--judge-model` (the `llm` graders and the agent mocks). Pin both, so that
  a model change is not read as a plugin change.
- `--threshold` is the lowest case score that passes, 0 to 1. 0 turns the score gate off: then the job fails only when the run fails
  (a case that did not load, a run that ended on an API error, the cost ceiling, a credential that the API refused).
- The credential is `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in the environment. The script removes an empty variable, so that it
  does not hide the other one. In GitHub Actions, with no credential the step stops before the run. On your own machine `claude` can use
  its login instead (then the run counts against the plan usage of that login).
- The script uses the Claude Code that is on PATH, any version with `claude plugin eval` (2.1.269 or later). It prints the version,
  and the summary names it. The action installs the version of its input `claude-version`.
- The results (`aggregate-result.json`, `report.html`, the mock recordings) go to `--output-dir`. The action uploads that folder.

The exit code is the one of `claude plugin eval`: 0 pass, 1 a case below the threshold or a failed run, 2 a partial run (the cost
ceiling or a refused credential). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, md_table, run_main, write_summary  # noqa: E402

TAGS = ("smoke", "mock")                    # the mocked tags; `e2e` needs the real server
CREDENTIALS = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
DEFAULT_TAG = "smoke"
DEFAULT_THRESHOLD = "0.8"
DEFAULT_RUNS = "1"
DEFAULT_CONCURRENCY = "4"
DEFAULT_MAX_COST_USD = "15"
DEFAULT_OUTPUT_DIR = "evals/results/ci"     # git ignores evals/results/
RESULT_FILE = "aggregate-result.json"
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/\[\]-]*")
CASE_GLOB = re.compile(r"[A-Za-z0-9*?._\[\]-]+")
EXIT_TEXT = {
    0: "pass",
    1: "a case is below the threshold, a case did not load, or a run ended on an API error (see the log above)",
    2: "partial run: the cost ceiling stopped it, or the API refused the credential",
    130: "interrupted",
    143: "terminated (for example by the job timeout)",
}
EXIT_TEXT_NO_GATE = "a case did not load, or a run ended on an API error (the score gate is off; see the log above)"


def parse_threshold(text: str) -> float:
    """The threshold as a number from 0 to 1. 0 means no score gate."""
    try:
        value = float(text)
    except ValueError:
        raise CiError(f"threshold {text!r} is not a number; give a value from 0 to 1 (0: no score gate)")
    if not 0.0 <= value <= 1.0:
        raise CiError(f"threshold {text!r} is not in the range 0 to 1 (0: no score gate)")
    return value


def _int_in(name: str, text: str, low: int, high: int) -> int:
    try:
        value = int(text)
    except ValueError:
        raise CiError(f"{name} {text!r} is not an integer from {low} to {high}")
    if not low <= value <= high:
        raise CiError(f"{name} {text!r} is not in the range {low} to {high}")
    return value


def _model(name: str, text: str) -> str:
    if not MODEL_NAME.fullmatch(text or ""):
        raise CiError(f"{name} {text!r} is not a model name (for example claude-sonnet-5-5 or sonnet)")
    return text


def build_command(args: argparse.Namespace) -> list[str]:
    """The `claude plugin eval` command. It checks each input first and raises CiError for a bad value."""
    if args.tag not in TAGS:
        raise CiError(f"tag {args.tag!r} is not a mocked tag; use one of: {', '.join(TAGS)} (the e2e case needs the real server)")
    threshold = parse_threshold(args.threshold)
    runs = _int_in("runs", args.runs, 1, 50)
    concurrency = _int_in("concurrency", args.concurrency, 1, 8)
    try:
        cost = float(args.max_cost_usd)
    except ValueError:
        raise CiError(f"max-cost-usd {args.max_cost_usd!r} is not a number")
    if not cost > 0:
        raise CiError(f"max-cost-usd {args.max_cost_usd!r} must be more than 0")
    cmd = ["claude", "plugin", "eval", ".", "--tag", args.tag]
    if args.case:
        if not CASE_GLOB.fullmatch(args.case):
            raise CiError(f"case {args.case!r} is not a glob of case names (letters, digits, - _ . * ? [ ])")
        cmd += ["--case", args.case]
    cmd += [
        "--model", _model("model", args.model),
        "--judge-model", _model("judge-model", args.judge_model),
        "--threshold", f"{threshold:g}",
        "--runs", str(runs),
        "-j", str(concurrency),
        "--max-cost-usd", f"{cost:g}",
        "--ablation", "none",           # a case that needs the MCP tools cannot pass without the plugin, so the delta tells nothing
        "--mocks", "record",            # the mocks of evals/mocks/; the real topopt server never starts
        "--scaffold",                   # stage-files.sh of the cases with user files (our own scripts)
        "--trust-plugin",
        "--no-publish",
        "--output-dir", args.output_dir,
    ]
    return cmd


def credential_env(environ: dict[str, str]) -> dict[str, str]:
    """A copy of the environment with no empty credential variable.

    In GitHub Actions (GITHUB_ACTIONS=true) a credential variable must be set, else CiError. Outside Actions `claude` can use its own login.
    """
    env = {k: v for k, v in environ.items() if not (k in CREDENTIALS and not v.strip())}
    if environ.get("GITHUB_ACTIONS") == "true" and not any(env.get(k) for k in CREDENTIALS):
        raise CiError("no credential: set the repository secret ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN (see evals/README.md)")
    return env


def claude_version() -> str:
    """The text of `claude --version`. Any version runs; CiError only when `claude` does not run."""
    try:
        return subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        raise CiError(f"cannot run `claude --version`: {e}")


def _note(case: dict) -> str:
    """The first error or mock abort of the runs of a case, else an empty text."""
    for run in (case.get("arms") or {}).get("with") or []:
        if run.get("error"):
            return str(run["error"])
        if run.get("aborted"):
            a = run["aborted"]
            return f"mock abort: {a.get('tool', '?')}: {a.get('reason', '')}" if isinstance(a, dict) else f"mock abort: {a}"
    return ""


def summary(result: dict, threshold: float, model: str, judge_model: str) -> str:
    """The job summary: one line of facts and one row for each case."""
    agg = result.get("aggregates") or {}
    gate = "off (threshold 0)" if threshold == 0 else f"{threshold:g}"
    head = (f"### Plugin evals: model `{model}`, judge `{judge_model}`, score gate {gate}\n\n"
            f"Suite score {agg.get('overallScore', 0):.2f}; cases at or above the threshold {agg.get('casesPassed', '?')} of "
            f"{agg.get('casesTotal', '?')}; cost ${result.get('costUsd', 0):.2f} (list-price estimate); "
            f"{result.get('durationSeconds', '?')} s; Claude Code {result.get('claudeVersion', '?')}.")
    if result.get("partial"):
        head += f"\n\n**Partial run:** {result.get('partialReason', 'no reason given')}."
    rows = []
    for c in result.get("cases") or []:
        score = (c.get("aggregates") or {}).get("score", 0)
        runs = len((c.get("arms") or {}).get("with") or [])
        verdict = "-" if threshold == 0 else ("pass" if score >= threshold else "FAIL")
        rows.append([c.get("name", "?"), f"{score:.2f}", str(runs), verdict, _note(c)[:200]])
    return head + "\n\n" + md_table(["Case", "Score", "Runs", "Gate", "Note"], rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the model of the agent under test (--model)")
    ap.add_argument("--judge-model", required=True, help="the model of the llm graders and the agent mocks (--judge-model)")
    ap.add_argument("--threshold", default=DEFAULT_THRESHOLD, help="lowest case score that passes, 0 to 1; 0: no score gate")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="smoke or mock")
    ap.add_argument("--case", default="", help="optional glob of case names; empty: all cases of the tag")
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="runs per case, 1 to 50")
    ap.add_argument("--concurrency", default=DEFAULT_CONCURRENCY, help="runs at the same time, 1 to 8")
    ap.add_argument("--max-cost-usd", default=DEFAULT_MAX_COST_USD, help="cost ceiling (list-price estimate); exit code 2 when hit")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="folder for aggregate-result.json and report.html (relative to the repo)")
    ap.add_argument("--dry-run", action="store_true", help="check the inputs and print the command; no model call")
    args = ap.parse_args(argv)

    cmd = build_command(args)
    threshold = parse_threshold(args.threshold)
    print("command: " + " ".join(cmd))
    print("score gate: " + ("off (threshold 0): the job fails only when the run fails" if threshold == 0 else f"a case below {threshold:g} fails the job"))
    if args.dry_run:
        return 0
    env = credential_env(dict(os.environ))
    print("claude: " + claude_version())

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / RESULT_FILE).unlink(missing_ok=True)       # an old result must not stand for this run
    sys.stdout.flush()                                # the lines above come before the progress lines of the runner in the log
    code = subprocess.run(cmd, cwd=ROOT, env=env).returncode

    result_path = out / RESULT_FILE
    if result_path.is_file():
        text = summary(json.loads(result_path.read_text()), threshold, args.model, args.judge_model)
        print(text)
        write_summary(text)
    else:
        write_summary(f"### Plugin evals\n\nNo {RESULT_FILE}: the run stopped before it wrote a result (exit code {code}).")
    reason = EXIT_TEXT_NO_GATE if (code == 1 and threshold == 0) else EXIT_TEXT.get(code, "see the log above")
    if code != 0:
        raise CiError(f"claude plugin eval exit code {code}: {reason}")
    print("OK: " + reason)
    return 0


if __name__ == "__main__":
    run_main(main)
