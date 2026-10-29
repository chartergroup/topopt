#!/usr/bin/env python3
"""Check that the shards of a leg together ran the same tests as one run would, and list the slowest tests. Standard library only.

    python3 scripts/ci/report_shards.py --artifacts ci-artifacts [--slowest 30]

The job `report` of tests.yml downloads the artifacts `ci-*` into one folder (one sub-folder for each artifact). Each sub-folder of a pytest job holds
`shard.json` (made by scripts/ci/run_pytest.py) and `junit.xml`. The shards of one leg and Python make a group. A group fails when
- a shard of it is missing (its artifact has no `shard.json`);
- the numbers of shards differ inside the group;
- a test ran in two shards;
- the shards together ran another number of tests than `tier_total`, the size of the tier that each shard counted (a shard that ran a part of the
  tier, or none, makes the sum short);
- the junit XML files of the group have a skipped test that the leg does not allow (the same check as `check_skips.py`, on all files together).
  The check also reads `collect.xml` of one shard: the skips of the collection (a test file that skips at import has no node id, so no shard runs it).
A group with one shard has `tier_total` null: nothing to compare, and the skip check runs as in its own job.
The 30 slowest tests of all groups go to the job summary: they are the input for a later move of tests to the slow tier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import CiError, md_table, run_main, write_summary  # noqa: E402
import check_skips  # noqa: E402


def read_shards(artifacts: Path) -> list[dict]:
    """The `shard.json` of each sub-folder of `artifacts`, each with the folder as `dir`. A sub-folder without the file (a job that failed before pytest ended)
    is listed as `{"dir": ..., "missing": True}`."""
    found = []
    for d in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        f = d / "shard.json"
        if f.is_file():
            found.append({**json.loads(f.read_text(encoding="utf-8")), "dir": d})
        else:
            found.append({"dir": d, "missing": True})
    return found


def group_problems(group: list[dict], name: str) -> list[str]:
    """The problems of the shards of one leg and Python."""
    p = []
    sizes = {s["shards"] for s in group}
    if len(sizes) != 1:
        return [f"{name}: the shards say different numbers of shards: {sorted(sizes)}"]
    n = sizes.pop()
    got = sorted(s["shard"] for s in group)
    if got != list(range(1, n + 1)):
        p.append(f"{name}: shards {sorted(set(range(1, n + 1)) - set(got))} are missing (found {got} of {n})")
    where: dict[str, int] = {}
    for s in group:
        for t in s["ran"]:
            if t in where and where[t] != s["shard"]:
                p.append(f"{name}: the test {t} ran in shard {where[t]} and in shard {s['shard']}")
            where.setdefault(t, s["shard"])
    totals = {s["tier_total"] for s in group}
    if None not in totals:
        if len(totals) != 1:
            p.append(f"{name}: the shards count different tiers: {sorted(totals)}")
        else:
            total = totals.pop()
            ran = len(where)
            if ran != total:
                p.append(f"{name}: the tier has {total} tests and the shards ran {ran} (difference {total - ran})")
    return p


def slowest(groups: dict, count: int) -> list[list]:
    """[test, seconds, leg/Python/shard] of the `count` slowest tests of all junit files."""
    from make_tiers import junit_results
    rows = []
    for (leg, py), group in groups.items():
        for s in group:
            j = s["dir"] / "junit.xml"
            if j.is_file():
                rows += [(t, k, f"{leg} {py} shard {s['shard']}/{s['shards']}") for k, (_, t) in junit_results(j).items()]
    rows.sort(key=lambda r: (-r[0], r[1]))
    return [[k, f"{t:.1f}", where] for t, k, where in rows[:count]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", required=True, help="the folder with one sub-folder for each downloaded artifact")
    ap.add_argument("--slowest", type=int, default=30, help="how many slow tests to list (default 30)")
    args = ap.parse_args(argv)
    art = Path(args.artifacts)
    if not art.is_dir():
        raise CiError(f"no artifacts folder at {art}")
    shards = read_shards(art)
    problems = [f"{s['dir'].name}: no shard.json (the pytest step did not finish)" for s in shards if s.get("missing")]
    groups: dict[tuple[str, str], list[dict]] = {}
    for s in shards:
        if not s.get("missing"):
            groups.setdefault((s["leg"], s["python"]), []).append(s)
    rows = []
    for (leg, py), group in sorted(groups.items()):
        name = f"{leg} Python {py}"
        gp = group_problems(group, name)
        problems += gp
        ran = len({t for s in group for t in s["ran"]})
        total = group[0]["tier_total"]
        rows.append([leg, py, f"{len(group)}/{group[0]['shards']}", ran, "not counted" if total is None else total, "FAIL" if gp else "ok"])
        junits = [str(s["dir"] / "junit.xml") for s in group]
        collects = [s["dir"] / "collect.xml" for s in sorted(group, key=lambda s: s["shard"]) if (s["dir"] / "collect.xml").is_file()]
        junits += [str(collects[0])] if collects else []         # every shard collects the same tier: one copy of its skips
        try:
            check_skips.main(["--leg", leg] + [x for j in junits for x in ("--junit", j)])
        except CiError as e:
            problems.append(f"{name}: {e}")
    md = ["### Shards: the same tests as one run", md_table(["leg", "Python", "shards", "tests ran", "tests in the tier", "state"], rows)]
    top = slowest(groups, args.slowest)
    if top:
        md += [f"### The {len(top)} slowest tests", md_table(["test", "seconds", "where"], top)]
    if problems:
        md += ["### Problems"] + [f"- {x}" for x in problems]
    write_summary("\n\n".join(md))
    for r in rows:
        print(" | ".join(str(c) for c in r))
    if problems:
        raise CiError(f"{len(problems)} problem(s):\n" + "\n".join(problems))
    print(f"OK: {len(groups)} group(s), every shard ran, no test twice, the counts agree")
    return 0


if __name__ == "__main__":
    run_main(main)
