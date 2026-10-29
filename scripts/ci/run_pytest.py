#!/usr/bin/env python3
"""Run the full pytest suite. Write the junit XML and the log to the artifacts folder.

    .venv/bin/python scripts/ci/run_pytest.py --leg linux-x86_64 [--tier fast|full] [--shard i/N] [--workers N|auto|0] [--threads N|0] [-- extra pytest arguments]

The suite runs in parallel with pytest-xdist (`-n`). The number of workers and the thread cap of each worker
come from the leg in legs.toml (`pytest_workers`, `pytest_threads`); the options override them.
`--workers 0` runs the suite in one process (the same as a plain `pytest`). `--threads 0` sets no thread variable. The thread cap sets OMP_NUM_THREADS,
MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and NUMBA_NUM_THREADS for the tests, because every worker runs BLAS,
OpenMP and numba threads of its own and they multiply. A variable that is already set in the environment keeps its value.
Without pytest-xdist the suite runs in one process and the log says so.

`--tier` chooses the tests (CONTRACTS.md 27.3). `full` (the default) runs every test, except the ones that the key `pytest_full` of the leg
leaves out (a marker expression for `-m`). `fast` runs the tests that the key `pytest_fast` of the leg selects (a marker expression, for
example `not slow`); the value `none` runs no test and writes a junit XML file with zero tests, so the steps after it still work. The markers
`slow` and `platform_sensitive` come from tests/tiers/ (tests/conftest.py).

`--shard i/N` runs one part of the tier (CONTRACTS.md 27.3.2). With N above 1, the script first collects the whole tier (`pytest --collect-only`, same `-m`),
splits the test FILES into N shards by their measured times (scripts/ci/shards.py, tests/tiers/durations.json), and runs the files of shard i only. The
pytest command gets the file names, so no test is deselected and no test shows as skipped. `<artifacts>/shard.json` holds the leg, the Python, the shard,
the number of tests of the whole tier (`tier_total`) and the node ids that this shard ran (`ran`); `scripts/ci/report_shards.py` reads it. With N = 1 the
whole suite runs as before and `tier_total` is null. The collect step also writes `<artifacts>/collect.xml`: a test file that skips at import (a
module-level skip) has no node id, so no shard runs it, and pytest writes its skip into that file; the job `report` checks those skips.
With N above 1, give no test path after `--`: a path would run its tests in every shard, so the script refuses it (use `-k` to run a part of the tier).

The output goes to the screen and to `<artifacts>/pytest.log`. The exit code is that of pytest.
The counts go to the job summary. `check_skips.py` reads the junit XML next.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, get_leg, loadavg, md_table, run_main, write_summary  # noqa: E402
import shards as shard_split  # noqa: E402
from make_tiers import junit_results  # noqa: E402


def junit_counts(junit: Path) -> dict:
    """The counts of a junit XML file: tests, failures, errors, skipped, seconds."""
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "seconds": 0.0}
    for s in suites:
        for k in ("tests", "failures", "errors", "skipped"):
            total[k] += int(s.get(k, 0))
        total["seconds"] += float(s.get("time", 0))
    total["passed"] = total["tests"] - total["failures"] - total["errors"] - total["skipped"]
    return total


# VECLIB_MAXIMUM_THREADS caps Apple Accelerate, which ignores OMP_NUM_THREADS. The first macOS CI run (2026-10-25) had 85 tests that compare runs
# byte for byte fail at the 13th digit; Accelerate threads in a worker are the suspect (CONTRACTS.md 27.3). On Linux the variable has no effect.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
TIERS = ("fast", "full")
NO_TESTS = "none"       # the value of `pytest_fast` for a leg that runs no pytest in the fast tier


def tier_selection(leg: dict, tier: str):
    """The marker expression for `-m` in this tier of the leg: a string, "" for no `-m` (all tests), or None for no test at all."""
    if tier not in TIERS:
        raise CiError(f"tier must be one of {', '.join(TIERS)}, not {tier!r}")
    value = str(leg.get("pytest_fast", "not slow") if tier == "fast" else leg.get("pytest_full", "")).strip()
    return None if value == NO_TESTS else value


def empty_junit(path: Path) -> None:
    """A junit XML file with zero tests (a leg that runs no pytest in this tier)."""
    path.write_text('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" errors="0" failures="0" skipped="0" tests="0" '
                    'time="0.0"></testsuite></testsuites>\n', encoding="utf-8")


def xdist_available() -> bool:
    """True if pytest-xdist can be imported (it is in the extra `dev`)."""
    return importlib.util.find_spec("xdist") is not None


def workers_arg(text) -> str:
    """The `-n` value for a worker setting: a whole number (0 or more) or `auto`. Other text is an error."""
    s = str(text).strip().lower()
    if s == "auto" or (s.isdigit() and int(s) >= 0):
        return s
    raise CiError(f"workers must be a whole number or 'auto', not {text!r}")


def thread_env(threads, environ) -> tuple[dict, list[str]]:
    """The environment for pytest and the names of the variables that this function set.

    `threads` is a whole number of 1 or more, or None (set nothing). A variable that `environ` already has keeps its value.
    """
    env = dict(environ)
    if threads is None:
        return env, []
    if int(threads) < 1:
        raise CiError(f"threads must be 1 or more, not {threads!r}")
    set_now = [v for v in THREAD_VARS if v not in env]
    for v in set_now:
        env[v] = str(int(threads))
    return env, set_now


def pytest_command(python, root, junit, workers, extra=()) -> list[str]:
    """The pytest command line. `workers` is a `workers_arg` value; "0" gives one process (no xdist)."""
    cmd = [python, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider", "--rootdir", str(root), f"--junitxml={junit}"]
    if workers != "0":
        cmd += ["-n", workers]
    return cmd + list(extra)


def collect_command(python, root, extra=(), junit=None) -> list[str]:
    """The pytest command that lists the node ids of the tier without running them (one process, no xdist). With `junit`, pytest also writes
    the skips of the collection (the files that skip at import) to that junit XML file."""
    cmd = [python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "--rootdir", str(root)]
    return cmd + ([f"--junitxml={junit}"] if junit else []) + list(extra)


def parse_node_ids(output: str) -> list[str]:
    """The node ids in the output of `pytest --collect-only -q`: the lines that start with a path of a .py file and have `::`."""
    return [ln.strip() for ln in output.splitlines() if ".py::" in ln and not ln.startswith((" ", "=", "-", "E "))]


def collect_tier(cmd, env, cwd=None) -> list[str]:
    """Run the collect command and return the node ids. A collection error is a `CiError` (a shard must not run a part of a tier it cannot list).
    Exit code 5 (no test collected) gives an empty list."""
    p = subprocess.run(cmd, cwd=cwd or ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode not in (0, 5):
        raise CiError(f"collecting the tier failed (exit code {p.returncode}):\n" + "\n".join(p.stdout.splitlines()[-40:]))
    return parse_node_ids(p.stdout)


# The pytest options whose next argument is a path or a node id but not a test to run (pytest applies them to the collect and to each shard alike).
PATH_OPTIONS = ("--deselect", "--ignore", "--ignore-glob", "--basetemp", "--confcutdir", "--rootdir", "-c", "--junitxml")


def path_args(pytest_args) -> list[str]:
    """The arguments that pytest takes as a test path: a node id (`::`), a `.py` file, or a file or folder that exists in the repo folder. The value
    of an option of PATH_OPTIONS is not one."""
    args = list(pytest_args)
    return [a for i, a in enumerate(args) if not a.startswith("-") and not (i and args[i - 1] in PATH_OPTIONS)
            and ("::" in a or a.endswith(".py") or (ROOT / a).exists())]


def shard_files(ids: list[str], durations: dict, i: int, n: int, key: str) -> list[str]:
    """The test files of shard `i` of `n` (1-based) for the node ids of the tier."""
    return shard_split.split_files(sorted({x.split("::")[0] for x in ids}), durations, n, key)[i - 1]


def write_shard_json(out: Path, leg: str, shard: tuple[int, int], tier: str, tier_total, junit: Path, files: list[str] | None) -> None:
    """`shard.json` for scripts/ci/report_shards.py: which leg, Python, shard and tier this is, the size of the whole tier, and the node ids that ran."""
    ran = sorted(junit_results(junit)) if junit.is_file() else []
    doc = {"leg": leg, "python": f"{sys.version_info[0]}.{sys.version_info[1]}", "shard": shard[0], "shards": shard[1], "tier": tier,
           "tier_total": tier_total, "files": files, "ran": ran}
    (out / "shard.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leg", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--workers", help="pytest-xdist workers: a number, 'auto', or 0 for one process (default: pytest_workers of the leg)")
    ap.add_argument("--threads", type=int, help="threads of each worker for BLAS, OpenMP and numba; 0 sets none (default: pytest_threads of the leg)")
    ap.add_argument("--shard", default="1/1", help="i/N: run shard i of N (default 1/1: all the tests); see the top of this file")
    ap.add_argument("--tier", default="full", choices=TIERS, help="fast: the tests that the key pytest_fast of the leg selects; full: all (default)")
    ap.add_argument("pytest_args", nargs="*", help="more arguments for pytest (put them after --)")
    args = ap.parse_args(argv)

    leg = get_leg(args.leg)               # the leg name must exist
    shard = shard_split.parse_shard(args.shard)
    if shard[1] > 1 and (paths := path_args(args.pytest_args)):
        raise CiError(f"--shard {args.shard}: give no test path after `--` ({', '.join(paths)}); each shard runs the files that the split gives it, "
                      "so a path would run its tests in every shard. Use -k to run a part of the tier")
    workers = workers_arg(args.workers if args.workers is not None else leg.get("pytest_workers", "auto"))
    threads = args.threads if args.threads is not None else leg.get("pytest_threads")
    threads = threads or None                 # 0 (or no value): set no thread variable
    if workers != "0" and not xdist_available():
        print("note: pytest-xdist is not installed (it is in the extra `dev`); the suite runs in one process", flush=True)
        workers = "0"
    env, caps_set = thread_env(threads, os.environ)
    out = Path(args.artifacts).resolve()      # pytest starts in the repo folder, so a relative path would differ
    out.mkdir(parents=True, exist_ok=True)
    junit, log = out / "junit.xml", out / "pytest.log"
    selection = tier_selection(leg, args.tier)
    if selection is None:
        empty_junit(junit)
        text = f"tier {args.tier}: the leg {args.leg} runs no pytest in this tier (pytest_fast = \"{NO_TESTS}\")."
        print(text, flush=True)
        log.write_text(text + "\n", encoding="utf-8")
        write_summary(f"### pytest ({args.leg}, tier {args.tier})\n\nNo test runs in this tier of this leg.")
        write_shard_json(out, args.leg, shard, args.tier, 0, junit, None)
        return 0
    extra = (["-m", selection] if selection else []) + list(args.pytest_args)
    files, tier_total = None, None
    if shard[1] > 1:
        ids = collect_tier(collect_command(sys.executable, ROOT, extra, out / "collect.xml"), env)
        tier_total = len(set(ids))
        files = shard_files(ids, shard_split.load_durations(), shard[0], shard[1], shard_split.duration_key(selection))
        print(f"shard {shard[0]}/{shard[1]}: {len(files)} of {len({x.split('::')[0] for x in ids})} test files, the tier has {tier_total} tests", flush=True)
        if not files:
            empty_junit(junit)
            log.write_text("this shard has no test file\n", encoding="utf-8")
            write_summary(f"### pytest ({args.leg}, tier {args.tier}, shard {shard[0]}/{shard[1]})\n\nThis shard has no test file.")
            write_shard_json(out, args.leg, shard, args.tier, tier_total, junit, [])
            return 0
    cmd = pytest_command(sys.executable, ROOT, junit, workers, list(extra) + (files or []))
    caps = ", ".join(f"{v}={env[v]}" for v in THREAD_VARS if v in env) or "none set"
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    setup = f"workers {workers}, usable CPUs {cpus}, thread variables: {caps}" + (
        f" (set here: {', '.join(caps_set)})" if caps_set else "")
    print("+ " + " ".join(cmd), flush=True)
    print(setup, flush=True)
    t0, load0 = time.monotonic(), loadavg()
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n{setup}\nload average at start: {load0}\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = p.wait()
        wall = time.monotonic() - t0
        f.write(f"exit code {code}, wall {wall:.1f} s, load average at end: {loadavg()}\n")
    if junit.is_file():
        c = junit_counts(junit)
        write_summary(f"### pytest ({args.leg}, tier {args.tier}{', ' + selection if selection else ''}{f', shard {shard[0]}/{shard[1]}' if shard[1] > 1 else ''})\n\n" + md_table(
            ["passed", "failed", "errors", "skipped", "wall time", "workers", "thread variables", "load average (start)", "exit code"],
            [[c["passed"], c["failures"], c["errors"], c["skipped"], f"{wall:.0f} s", workers, caps, load0, code]]))
    else:
        write_summary(f"### pytest ({args.leg})\n\nNo junit XML file. Exit code {code}.")
    write_shard_json(out, args.leg, shard, args.tier, tier_total, junit, files)
    return code


if __name__ == "__main__":
    run_main(main)
