#!/usr/bin/env python3
"""Fail when a test that the leg expects to run was skipped.

    .venv/bin/python scripts/ci/check_skips.py --leg linux-x86_64 --junit artifacts/junit.xml [--junit more/junit.xml ...]

`--junit` can repeat: the job `report` of tests.yml checks the junit files of all shards of a leg together (a shard checks its own file).

Every skipped test in the junit XML must match one `allowed_skip` entry of the leg in legs.toml:
the test file is the same (or matches the `file` value as a glob pattern, for example `tests/test_*gpu_mg*.py`),
and the skip message holds the `reason` text. A skip that matches no entry
is a test that the plan expects on this platform and that did not run (for example the gmsh tests when
the gmsh library does not load). An entry that matches nothing is not an error: it is a fact in the summary and in the log.
An xfail (an expected failure; pytest writes it as `<skipped type="pytest.xfail">`) is not a skip: the check leaves it out.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, get_leg, md_table, run_main, write_summary  # noqa: E402


def classname_to_file(classname: str, name: str, root: Path) -> str:
    """The test file of a junit `testcase`, as a path with slashes (for example `tests/test_fe.py`).

    The junit `classname` is the dotted path of the module, plus the class if there is one. The path can
    start with folders above the repo (when pytest chose a root above it), so the function tries every
    window of the dotted name until it finds a file under `root`. A module-level skip (a skip at import or
    collection time) has an empty `classname`, and its `name` is the dotted module name (pytest 9:
    `<testcase classname="" name="tests.test_gpu_mg">`). A `name` that ends in `.py` is taken as a path.
    """
    if not classname and name.endswith(".py"):
        return name.replace("\\", "/")
    parts = (classname or name).split(".")
    for i in range(len(parts)):
        for j in range(len(parts), i, -1):
            cand = "/".join(parts[i:j]) + ".py"
            if (root / cand).is_file():
                return cand
    return "/".join(parts) + ".py"


def skip_reason(message: str, text: str) -> str:
    """The reason of a skip. A skip at collection time has the message "collection skipped", and its text is the
    repr of a tuple `(path, line, "Skipped: <reason>")`. In that case the reason is the third item."""
    if message and message != "collection skipped":
        return message
    try:
        item = ast.literal_eval(text.strip())
        if isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], str):
            return item[2].removeprefix("Skipped: ")
    except (ValueError, SyntaxError):
        pass
    return message or text.strip()


def is_xfail(sk) -> bool:
    """A `<skipped>` element of an expected failure (pytest writes `type="pytest.xfail"`). It is not a skip."""
    return sk.get("type") == "pytest.xfail"


def count_xfail(junit: Path) -> int:
    """The number of expected failures (xfail) in a junit XML file. The skip check leaves them out."""
    return sum(1 for tc in ET.parse(junit).getroot().iter("testcase")
               if tc.find("skipped") is not None and is_xfail(tc.find("skipped")))


def skipped_tests(junit: Path, root: Path = ROOT) -> list[dict]:
    """The skipped tests of a junit XML file: `file`, `name` and `reason` of each. An xfail is not a skip."""
    out = []
    for tc in ET.parse(junit).getroot().iter("testcase"):
        sk = tc.find("skipped")
        if sk is None or is_xfail(sk):
            continue
        name = tc.get("name", "")
        out.append({"file": classname_to_file(tc.get("classname", ""), name, root), "name": name,
                    "reason": skip_reason(sk.get("message") or "", sk.text or "")})
    return out


def match_skips(skips: list[dict], allowed: list[dict]) -> tuple[list[dict], list[int]]:
    """Return (the skips that no entry allows, the number of skips that each entry allowed).

    The `file` of an entry is a glob pattern (`fnmatch`, case-sensitive): a plain path matches only itself, and
    `tests/test_*gpu_mg*.py` matches every gpu-mg test file, also one that a later PR adds."""
    used = [0] * len(allowed)
    unexpected = []
    for s in skips:
        for i, a in enumerate(allowed):
            if fnmatch.fnmatchcase(s["file"], a["file"]) and a["reason"] in s["reason"]:
                used[i] += 1
                s["why"] = a.get("why", "")
                break
        else:
            unexpected.append(s)
    return unexpected, used


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leg", required=True)
    ap.add_argument("--junit", action="append", help="a junit XML file; repeat it for several (default: artifacts/junit.xml)")
    args = ap.parse_args(argv)

    leg = get_leg(args.leg)
    junits = [Path(j).resolve() for j in (args.junit or ["artifacts/junit.xml"])]
    for junit in junits:
        if not junit.is_file():
            raise CiError(f"no junit XML file at {junit} (the pytest step did not write one)")
    allowed = leg.get("allowed_skip", [])
    skips = [s for junit in junits for s in skipped_tests(junit)]
    unexpected, used = match_skips(skips, allowed)

    rows = [[s["file"], s["name"], s["reason"][:110], "UNEXPECTED" if s in unexpected else "allowed: " + s.get("why", "")]
            for s in skips]
    md = [f"### Skipped tests ({args.leg}): {len(skips)}, unexpected: {len(unexpected)}"]
    if rows:
        md.append(md_table(["file", "test", "skip message", "state"], rows))
    idle = [a for a, n in zip(allowed, used) if n == 0]
    idle_text = ""
    if idle:
        idle_text = ("Allowed skips that did not happen on this run (a fact, not an error): " +
                     ", ".join(f"{a['file']} ({a['reason']})" for a in idle))
        md.append(idle_text)
    xfail = sum(count_xfail(junit) for junit in junits)
    if xfail:
        md.append(f"{xfail} expected failure(s) (xfail) are left out of this check.")
    write_summary("\n\n".join(md))
    for r in rows:
        print(" | ".join(r))
    if idle_text:
        print(idle_text)
    if xfail:
        print(f"{xfail} expected failure(s) (xfail) are left out of this check.")
    if unexpected:
        lines = "\n".join(f"  {s['file']}::{s['name']}: {s['reason']}" for s in unexpected)
        raise CiError(f"{len(unexpected)} skipped test(s) that the plan expects to run on {args.leg}:\n{lines}\n"
                      "If a skip is right for this platform, add an allowed_skip entry to scripts/ci/legs.toml.")
    print(f"OK: {len(skips)} skipped test(s), all allowed for {args.leg}")
    return 0


if __name__ == "__main__":
    run_main(main)
