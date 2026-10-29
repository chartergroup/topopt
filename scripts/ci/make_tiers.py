#!/usr/bin/env python3
"""Write the two lists that sort the tests into tiers (tests/tiers/slow.txt and tests/tiers/platform_files.txt), or check them.

    .venv/bin/python scripts/ci/make_tiers.py write --junit a/junit.xml --junit b/junit.xml [--platform-junit mac/junit.xml --other-junit linux/junit.xml]
    .venv/bin/python scripts/ci/make_tiers.py durations --junit a/junit.xml --junit b/junit.xml
    .venv/bin/python scripts/ci/make_tiers.py check

`tests/conftest.py` reads the lists and adds the marker `slow` to each test in `slow.txt` and the marker `platform_sensitive` to each
test in a file of `platform_files.txt`. The tiers of the CI (CONTRACTS.md 27.3) select on these markers.

- slow.txt: the node ids of the tests that took SLOW_SECONDS or more in any of the given junit XML files (the longest time counts).
- platform_files.txt: the test files that can behave differently on macOS and on Linux. A file is in the list when it names the platform in
  its text (PLATFORM_WORDS), or when one of its tests had another outcome (pass, fail, skip) in the macOS junit XML than in the Linux one.

- durations.json (`durations`): the seconds of each test file, for the split of a pytest run into shards (scripts/ci/shards.py). `all` is the sum over the
  tests of the file, `not_slow` the sum over the tests that are not in slow.txt. A test that is in more than one junit XML file counts with its longest time.

A test file that holds the line TIER_SKIP_LINE is in no list: such a file tests a tool that the public repository does not have, so the
lists are the same in both repositories.

`check` fails when a listed test file does not exist (also a file of durations.json). A test that is not in slow.txt is in the fast tier: a new slow test runs in the fast
tier until someone runs `write` again, and the full tier finds it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import os
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, run_main  # noqa: E402

SLOW_SECONDS = 5.0
PLATFORM_WORDS = re.compile(r"cholesky|accelerate|darwin|macos|sys\.platform|platform\.system|VECLIB|\.dylib", re.I)
TIER_DIR = ROOT / "tests" / "tiers"
SLOW_FILE = TIER_DIR / "slow.txt"
PLATFORM_FILE = TIER_DIR / "platform_files.txt"
DURATIONS_FILE = TIER_DIR / "durations.json"
TIER_SKIP_LINE = "# tier lists: leave this file out"


def skipped(root: Path, test_file: str) -> bool:
    """True when the test file holds TIER_SKIP_LINE (it is in no tier list)."""
    p = root / test_file
    return p.is_file() and TIER_SKIP_LINE in p.read_text(encoding="utf-8", errors="replace")


def junit_results(junit: Path) -> dict[str, tuple[str, float]]:
    """{node id: (outcome, seconds)} of a junit XML file. The outcome is pass, fail or skip. The tests have no classes, so the
    junit classname `tests.test_x` and the name give the node id `tests/test_x.py::name`."""
    out = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        file = (tc.get("classname") or "").replace(".", "/") + ".py"
        if not tc.get("classname"):
            continue
        if tc.find("failure") is not None or tc.find("error") is not None:
            outcome = "fail"
        elif tc.find("skipped") is not None:
            outcome = "skip"
        else:
            outcome = "pass"
        out[f"{file}::{tc.get('name')}"] = (outcome, float(tc.get("time", 0)))
    return out


def slow_ids(junits: list[dict], seconds: float = SLOW_SECONDS) -> list[str]:
    """The node ids with `seconds` or more in any of the results."""
    longest: dict[str, float] = {}
    for res in junits:
        for k, (_, t) in res.items():
            longest[k] = max(longest.get(k, 0.0), t)
    return sorted(k for k, t in longest.items() if t >= seconds)


def file_durations(junits: list[dict], slow: set[str]) -> dict[str, dict[str, float]]:
    """{file: {"all": seconds, "not_slow": seconds}} from junit results. A test's time is its longest time in the results. `slow` holds the node ids of slow.txt."""
    longest: dict[str, float] = {}
    for res in junits:
        for k, (_, t) in res.items():
            longest[k] = max(longest.get(k, 0.0), t)
    out: dict[str, dict[str, float]] = {}
    for k, t in longest.items():
        d = out.setdefault(k.split("::")[0], {"all": 0.0, "not_slow": 0.0})
        d["all"] += t
        if k not in slow:
            d["not_slow"] += t
    return {f: {"all": round(d["all"], 1), "not_slow": round(d["not_slow"], 1)} for f, d in sorted(out.items())}


def write_durations(path: Path, files: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"about": "Seconds of each test file in CI runs (scripts/ci/make_tiers.py durations). Used by scripts/ci/shards.py to balance the shards. "
                    "`all`: every test of the file. `not_slow`: the tests that are not in slow.txt (the fast tier).", "files": files}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def platform_files(root: Path, platform_res: dict, other_res: dict) -> list[str]:
    """The test files that name the platform, plus the files with a test that has another outcome in the two results."""
    files = {p.relative_to(root).as_posix() for p in (root / "tests").glob("test_*.py") if PLATFORM_WORDS.search(p.read_text(encoding="utf-8", errors="replace"))}
    for k, (outcome, _) in platform_res.items():
        if k in other_res and other_res[k][0] != outcome:
            files.add(k.split("::")[0])
    return sorted(f for f in files if (root / f).is_file() and not skipped(root, f))


def read_list(path: Path) -> list[str]:
    """The non-empty lines of a list file that do not start with #."""
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def write_list(path: Path, header: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"# {h}\n" for h in header.splitlines()) + "".join(f"{ln}\n" for ln in lines), encoding="utf-8")


def check(root: Path = ROOT) -> list[str]:
    """Problems: a listed file that does not exist."""
    p = [f"slow.txt: no file for {i}" for i in read_list(root / "tests" / "tiers" / "slow.txt") if not (root / i.split("::")[0]).is_file()]
    p += [f"platform_files.txt: no file {f}" for f in read_list(root / "tests" / "tiers" / "platform_files.txt") if not (root / f).is_file()]
    if (root / "tests" / "tiers" / "durations.json").is_file():
        files = json.loads((root / "tests" / "tiers" / "durations.json").read_text(encoding="utf-8")).get("files", {})
        p += [f"durations.json: no file {f}" for f in files if not (root / f).is_file()]
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--junit", action="append", required=True, help="a junit XML file for the durations (repeat it)")
    w.add_argument("--platform-junit", help="the junit XML of the macOS leg")
    w.add_argument("--other-junit", help="the junit XML of a Linux leg, to find the tests with another outcome")
    d = sub.add_parser("durations")
    d.add_argument("--junit", action="append", required=True, help="a junit XML file (repeat it)")
    sub.add_parser("check")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        problems = check()
        for x in problems:
            print("problem:", x)
        print("OK: the tier lists name files that exist" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    results = [junit_results(Path(j)) for j in args.junit]
    if args.cmd == "durations":
        files = {f: d for f, d in file_durations(results, set(read_list(SLOW_FILE))).items()
                 if (ROOT / f).is_file() and not skipped(ROOT, f)}      # a file of an older run may be gone
        write_durations(DURATIONS_FILE, files)
        print(f"durations.json: {len(files)} files, {sum(d['all'] for d in files.values()):.0f} s in all")
        return 0
    slow = [k for k in slow_ids(results) if not skipped(ROOT, k.split("::")[0])]
    write_list(SLOW_FILE, f"Tests of {SLOW_SECONDS:g} s or more in the CI runs that scripts/ci/make_tiers.py read. They get the marker `slow`.\nMade by scripts/ci/make_tiers.py write; see CONTRACTS.md 27.3.", slow)
    if args.platform_junit and args.other_junit:
        files = platform_files(ROOT, junit_results(Path(args.platform_junit)), junit_results(Path(args.other_junit)))
    else:
        files = platform_files(ROOT, {}, {})
    write_list(PLATFORM_FILE, "Test files that can behave differently on macOS and Linux. Their tests get the marker `platform_sensitive`.\nMade by scripts/ci/make_tiers.py write; see CONTRACTS.md 27.3.", files)
    print(f"slow.txt: {len(slow)} tests; platform_files.txt: {len(files)} files")
    return 0


if __name__ == "__main__":
    run_main(main)
