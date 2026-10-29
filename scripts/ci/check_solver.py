#!/usr/bin/env python3
"""Run `topopt check` on the hook example and compare the CPU solver with the one that the leg expects.

    .venv/bin/python scripts/ci/check_solver.py --leg linux-x86_64

A skipped solver test can hide a missing fast backend. This step does not depend on a test: it
asks `topopt check --json` which CPU solver the rule of `settings.solver: auto` gives on this machine
(`solver.cpu_solver`: cholesky, pardiso or amg), and fails when the answer is not `expect_solver` of the leg (legs.toml).
`auto` can pick the GPU solver gpu-mg on a machine that shows an OpenCL GPU, and that pick can differ from run to run of a CI leg
(the runner image decides), so the step does not compare it. It prints the pick (`solver.resolved` and the line of `solver.choice`)
as information. It saves the text output of `topopt check` and the JSON in the artifacts folder.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, get_leg, md_table, run_main, write_summary  # noqa: E402

DEFAULT_YAML = "examples/hook/hook.yaml"


def solver_problem(check_json: dict, expected: str) -> str | None:
    """Return the text of the problem if `check_json` does not report the expected CPU solver, else None.

    The field is `solver.cpu_solver`: the CPU rule for `settings.solver` (for `auto`, the solver that a run uses when the GPU is not used).
    """
    sv = check_json.get("solver") or {}
    if sv.get("cpu_solver") == expected:
        return None
    detail = f"; error: {sv['error']}" if sv.get("error") else ""
    return (f"topopt check reports the CPU solver {sv.get('cpu_solver')!r} (platform {sv.get('platform')!r}, "
            f"settings.solver {sv.get('setting')!r}); the leg expects {expected!r}{detail}")


def auto_pick_text(check_json: dict) -> str:
    """The solver that a run uses first on this machine, with the reason when `auto` chose it. Information: nothing compares it."""
    sv = check_json.get("solver") or {}
    resolved = sv.get("resolved")
    if resolved is None:
        return "none (settings.solver cannot start here)"
    line = (sv.get("choice") or {}).get("line") or ""
    return resolved + (f": {line[len('solver: '):]}" if line.startswith("solver: ") else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leg", required=True)
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args(argv)

    leg = get_leg(args.leg)
    out = Path(args.artifacts).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, "-m", "topopt", "check", args.yaml]
    text = subprocess.run(base, capture_output=True, text=True, cwd=ROOT)
    (out / "topopt-check.txt").write_text(f"$ topopt check {args.yaml}\n(exit code {text.returncode})\n{text.stdout}{text.stderr}")
    sys.stdout.write(text.stdout + text.stderr)
    js = subprocess.run([*base, "--json"], capture_output=True, text=True, cwd=ROOT)
    (out / "topopt-check.json").write_text(js.stdout)
    try:
        data = json.loads(js.stdout)
    except json.JSONDecodeError:
        raise CiError(f"topopt check --json did not print JSON (exit code {js.returncode}): {js.stdout[:200]!r} {js.stderr[:400]!r}")
    problem = solver_problem(data, leg["expect_solver"])
    sv = data.get("solver", {})
    pick = auto_pick_text(data)
    write_summary(f"### `topopt check` ({args.leg})\n\n" + md_table(
        ["CPU solver expected", "CPU solver reported", "platform", "exit code"],
        [[leg["expect_solver"], sv.get("cpu_solver"), sv.get("platform"), f"{text.returncode} / {js.returncode}"]])
        + f"\nThe solver that a run uses first (information, not compared): {pick}\n")
    print(f"information, not compared: a run uses first {pick}")
    if problem:
        raise CiError(problem)
    if text.returncode != 0:
        raise CiError(f"topopt check exited with code {text.returncode}")
    print(f"OK: CPU solver {sv['cpu_solver']} (expected {leg['expect_solver']})")
    return 0


if __name__ == "__main__":
    run_main(main)
