#!/usr/bin/env python3
"""Choose the test legs for a GitHub event and write the matrix for the workflow.

    python3 scripts/ci/plan_legs.py --event workflow_dispatch --legs linux|all [--workflow intel-mac] [--tier fast|full]

A workflow runs the legs that name it: `workflow` in legs.toml, `tests` if the leg has no such key (.github/workflows/tests.yml).
The manual Intel Mac workflow `intel-mac.yml` calls this script with `--workflow intel-mac`.
Each leg runs once for each Python in its `pythons` list (legs.toml; the versions come from SUPPORTED_PYTHONS in _common.py), so the matrix
has one entry for each triple of a leg, a Python and a shard: `{"label": ..., "os": ..., "python": "3.13", "shard": 2, "shards": 4, "suffix": "-s2of4", "part": ", shard 2/4"}`.
A leg that lists two Pythons gives two entries; a leg with `shards = 4` (legs.toml) gives four entries for each Python, and each one runs a part of the test files
(scripts/ci/shards.py). `suffix` and `part` are empty when `shards` is 1.
`tests.yml` starts by hand only (`workflow_dispatch`) and passes its input `legs`:
- `--legs linux` plans the cheap legs: the legs without `on_pull_request = false` in legs.toml (only `linux-x86_64`). The key name is
  historic: it now means "this leg is in the choice `linux`";
- `--legs all` plans every leg of the workflow.
Without `--legs` the script keeps its old rule, which depends on the event: a leg with `on_pull_request = false` does not run on
`pull_request` and runs on every other event. The matrix goes to the file `$GITHUB_OUTPUT` as `matrix=<json>` (first line), then `first=<json>` (the legs of stage 1), `after=<json>` (the other
legs) and `has_after=true|false`. A leg has `stage = 2` in legs.toml when it must wait for stage 1: the workflow runs the job `test` for `first`,
and the job `test-after` for `after` only when `test` has passed. The matrix is also written to the job summary as a table. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pytest import TIERS, tier_selection  # noqa: E402
from _common import DEFAULT_WORKFLOW, CiError, load_spec, md_table, pythons_of, run_main, shards_of, workflow_of, write_summary  # noqa: E402


MATRIX_KEYS = ("label", "os", "python", "shard", "shards", "suffix", "part")      # the keys of each matrix entry; check_workflow.py reads them
LEGS_CHOICES = ("linux", "all")    # the values of `--legs`; the input `legs` of tests.yml has the same options (a test checks that)


def entry(label: str, runner: str, python: str, shard: int, shards: int) -> dict:
    """One matrix entry. `suffix` ends the artifact name and `part` ends the job name; both are empty when the leg has one shard,
    so a leg without a split keeps the names that it had before."""
    return {"label": label, "os": runner, "python": python, "shard": shard, "shards": shards,
            "suffix": f"-s{shard}of{shards}" if shards > 1 else "", "part": f", shard {shard}/{shards}" if shards > 1 else ""}


def plan(event: str, spec: dict | None = None, workflow: str = DEFAULT_WORKFLOW,
         legs: str | None = None, tier: str | None = None) -> tuple[list[dict], list[tuple[str, str]]]:
    """Return (the matrix entries, the skipped legs with the reason of each). A leg of another workflow is not listed.

    A leg gives one entry for each Python of its `pythons` list, in that order, and for each shard of that Python (`shards`, default 1).

    `tier` ("fast" or "full", or None) is the tier of the run: a leg that runs no pytest in it (`pytest_fast = "none"`) gets one shard.

    `legs` is "linux" (only the legs without `on_pull_request = false`), "all" (every leg of the workflow) or None (the rule of the event:
    a leg with `on_pull_request = false` is left out on `pull_request` only).
    """
    if legs is not None and legs not in LEGS_CHOICES:
        raise CiError(f"legs must be one of {', '.join(LEGS_CHOICES)}, not {legs!r}")
    spec = load_spec() if spec is None else spec
    cheap_only = legs == "linux" or (legs is None and event == "pull_request")
    include, skipped = [], []
    for label, leg in spec["legs"].items():
        if workflow_of(leg) != workflow:
            continue
        if cheap_only and not leg.get("on_pull_request", True):
            skipped.append((label, "legs.toml: on_pull_request = false (not in the choice `linux`; the choice `all` runs it)"))
            continue
        n = shards_of(leg, label)
        if tier is not None and tier_selection(leg, tier) is None:
            n = 1               # the leg runs no pytest in this tier: a split would pay the setup of every shard for nothing
        include += [entry(label, leg["runner"], python, shard, n) for python in pythons_of(leg, label) for shard in range(1, n + 1)]
    return include, skipped


def split_stages(include: list[dict], spec: dict) -> tuple[list[dict], list[dict]]:
    """(the entries of stage 1, the entries of the later stages). A leg without `stage` in legs.toml is in stage 1."""
    stage = lambda e: int(spec["legs"][e["label"]].get("stage", 1))      # noqa: E731
    return [e for e in include if stage(e) == 1], [e for e in include if stage(e) > 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--event", required=True, help="the GitHub event name, for example workflow_dispatch")
    ap.add_argument("--workflow", default=DEFAULT_WORKFLOW, help="the workflow whose legs to plan (default: tests)")
    ap.add_argument("--legs", choices=LEGS_CHOICES,
                    help="linux: only the legs without on_pull_request = false; all: every leg of the workflow (default: the rule of the event)")
    ap.add_argument("--tier", choices=TIERS, help="the tier of the run: a leg that runs no pytest in it gets one shard (default: the shards of legs.toml)")
    args = ap.parse_args(argv)
    include, skipped = plan(args.event, workflow=args.workflow, legs=args.legs, tier=args.tier)
    what = f"the event {args.event!r}" if args.legs is None else f"the event {args.event!r} with legs={args.legs}"
    if not include:
        raise CiError(f"no leg of the workflow {args.workflow!r} runs for {what}")
    matrix = json.dumps({"include": include}, separators=(",", ":"))
    print(f"event {args.event}{'' if args.legs is None else f', legs {args.legs}'}: matrix {matrix}")
    for label, why in skipped:
        print(f"leg {label} does not run: {why}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            first, after = split_stages(include, load_spec())
            f.write(f"matrix={matrix}\n")
            f.write(f"first={json.dumps({'include': first}, separators=(',', ':'))}\n")
            f.write(f"after={json.dumps({'include': after}, separators=(',', ':'))}\n")
            f.write(f"has_after={'true' if after else 'false'}\n")
    rows = [[e["label"], e["python"] + e["part"].replace(",", ""), e["os"], "runs"] for e in include] + [[label, "", "", f"does not run: {why}"] for label, why in skipped]
    chosen = "" if args.legs is None else f" with `legs={args.legs}`"
    write_summary(f"### Test legs of the workflow `{args.workflow}` for the event `{args.event}`{chosen}\n\n" + md_table(["leg", "Python", "runner", "state"], rows))
    return 0


if __name__ == "__main__":
    run_main(main)
