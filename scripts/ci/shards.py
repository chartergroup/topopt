"""Split the test files of one pytest run into shards (parallel CI jobs) by their measured times. Standard library only.

    python3 scripts/ci/shards.py split --shard 2/4 [--tier fast|full] [--leg linux-x86_64] file1.py file2.py ...    # print the files of one shard
    python3 scripts/ci/shards.py show --shards 4 [--key not_slow] tests/test_a.py ...                                # print the load of each shard

`tests/tiers/durations.json` holds the seconds of each test file (`scripts/ci/make_tiers.py durations` writes it from junit XML files of CI runs):
`all` is the sum over the tests of the file, `not_slow` is the sum over the tests that are not in `tests/tiers/slow.txt`. The fast tier does not run
the slow tests, so it balances on `not_slow`; a tier that runs them balances on `all` (`duration_key`).

The split is a greedy fill (longest file first, into the shard with the smallest load). A file that is not in the list gets the median of the
listed times, and a tie breaks by the file name, so the same input gives the same split on every runner. Every file is in exactly one shard.
`scripts/ci/run_pytest.py --shard i/N` uses it, and `scripts/ci/report_shards.py` checks that the shards together ran every test once.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, get_leg, run_main  # noqa: E402

DURATIONS_FILE = ROOT / "tests" / "tiers" / "durations.json"
KEYS = ("all", "not_slow")


def parse_shard(text: str) -> tuple[int, int]:
    """`i/N` -> (i, N). 1 <= i <= N."""
    try:
        i, n = (int(x) for x in str(text).split("/"))
    except ValueError:
        raise CiError(f"--shard must look like 2/4 (shard number / number of shards), not {text!r}") from None
    if n < 1 or not 1 <= i <= n:
        raise CiError(f"--shard {text}: the shard number must be from 1 to the number of shards, and that number must be 1 or more")
    return i, n


def duration_key(selection: str | None) -> str:
    """The sum of `durations.json` that fits a marker expression: `not_slow` when the expression leaves the slow tests out, else `all`."""
    return "not_slow" if selection and "not slow" in selection else "all"


def load_durations(path: Path = DURATIONS_FILE) -> dict[str, dict[str, float]]:
    """{file: {"all": seconds, "not_slow": seconds}}. No file (or no `files`) gives an empty dict: every file then has the same weight."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict):
        raise CiError(f"{path}: expected an object with a `files` object")
    return files


def weights(files: list[str], durations: dict, key: str) -> dict[str, float]:
    """The seconds of each file for the sum `key`. A file that is not in `durations` gets the median of the listed files (1.0 when the list is empty)."""
    known = [float(d[key]) for d in durations.values() if isinstance(d, dict) and key in d]
    median = statistics.median(known) if known else 1.0
    return {f: float(durations[f][key]) if f in durations and key in durations[f] else median for f in files}


def split_files(files: list[str], durations: dict, n: int, key: str = "all") -> list[list[str]]:
    """Split `files` into `n` lists. Every file is in exactly one list. Each list is sorted by name; a list can be empty when there are fewer files than shards."""
    if n < 1:
        raise CiError(f"the number of shards must be 1 or more, not {n}")
    unique = sorted(set(files))
    w = weights(unique, durations, key)
    loads = [0.0] * n
    out: list[list[str]] = [[] for _ in range(n)]
    for f in sorted(unique, key=lambda f: (-w[f], f)):
        i = min(range(n), key=lambda j: (loads[j], j))
        out[i].append(f)
        loads[i] += w[f]
    return [sorted(x) for x in out]


def shard_loads(parts: list[list[str]], durations: dict, key: str) -> list[float]:
    """The summed seconds of each shard (the same weights as the split)."""
    w = weights([f for p in parts for f in p], durations, key)
    return [sum(w[f] for f in p) for p in parts]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("split", "show"):
        p = sub.add_parser(name)
        p.add_argument("files", nargs="+", help="the test files of the run (paths relative to the repo)")
        p.add_argument("--key", choices=KEYS, help="the sum of durations.json (default: from --tier and --leg, else all)")
        p.add_argument("--durations", default=str(DURATIONS_FILE))
    sp = sub.choices["split"]
    sp.add_argument("--shard", required=True, help="i/N")
    sp.add_argument("--tier", choices=("fast", "full"))
    sp.add_argument("--leg")
    sub.choices["show"].add_argument("--shards", type=int, required=True)
    args = ap.parse_args(argv)
    durations = load_durations(Path(args.durations))
    if args.cmd == "split":
        key = args.key
        if key is None and args.tier and args.leg:
            from run_pytest import tier_selection
            key = duration_key(tier_selection(get_leg(args.leg), args.tier))
        i, n = parse_shard(args.shard)
        print("\n".join(split_files(args.files, durations, n, key or "all")[i - 1]))
        return 0
    key = args.key or "all"
    parts = split_files(args.files, durations, args.shards, key)
    for i, (p, s) in enumerate(zip(parts, shard_loads(parts, durations, key)), 1):
        print(f"shard {i}/{args.shards}: {len(p)} files, {s:.0f} s")
    return 0


if __name__ == "__main__":
    run_main(main)
