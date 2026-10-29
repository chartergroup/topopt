"""Shared helpers of the CI scripts (standard library only, Python 3.11 or later).

The scripts in this folder hold the logic of the CI steps. The workflow file only calls them.
`legs.toml` holds the facts that differ between the legs (CONTRACTS.md section 27). The workflow files are
`.github/workflows/tests.yml` and `.github/workflows/intel-mac.yml` (active since 2026-10-24, both started by hand only since then; CONTRACTS.md 27.1).
`.github/workflows/evals.yml` (the plugin evals, by hand only) calls `run_evals.py` through the action `.github/actions/plugin-eval`.
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGS_FILE = Path(__file__).resolve().with_name("legs.toml")
DEFAULT_LOCK = "requirements.lock"
DEFAULT_WORKFLOW = "tests"
# The Python versions that the package supports and that CI tests. This tuple is the one place in the CI scripts that names them:
# check_locks.py, wheel_check.py and plan_legs.py read it, and the key `pythons` of a leg in legs.toml picks from it.
# `requires-python` in pyproject.toml must allow exactly these versions (tests/test_ci_scripts.py checks that). To support a new
# Python, add it here, to `requires-python` and to the `pythons` of the legs that run it, in one change.
SUPPORTED_PYTHONS = ("3.13",)        # 3.13 only since #469 (3.12 and 3.13 before)


class CiError(Exception):
    """A step fails with this text. `main_wrapper` prints it and returns exit code 1."""


def load_spec(path: Path = LEGS_FILE) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def get_leg(label: str, spec: dict | None = None) -> dict:
    """The table of one leg. The label is also in the table, as `label`."""
    spec = load_spec() if spec is None else spec
    legs = spec.get("legs", {})
    if label not in legs:
        raise CiError(f"unknown leg {label!r}; the legs in legs.toml are: {', '.join(sorted(legs))}")
    return {"label": label, **legs[label]}


def lock_of(leg: dict) -> str:
    """The lock file of a leg (`lock` in legs.toml), as a path relative to the repo. The default is requirements.lock."""
    return leg.get("lock", DEFAULT_LOCK)


def pythons_of(leg: dict, label: str = "a leg") -> tuple[str, ...]:
    """The Python versions that a leg runs (`pythons` in legs.toml). The key is required, and each version must be in SUPPORTED_PYTHONS."""
    pythons = leg.get("pythons")
    if not isinstance(pythons, list) or not pythons or not all(isinstance(p, str) for p in pythons):
        raise CiError(f"{label}: `pythons` in legs.toml must be a list of Python versions, for example [\"3.13\"]")
    unknown = sorted({p for p in pythons if p not in SUPPORTED_PYTHONS})
    if unknown:
        raise CiError(f"{label}: `pythons` names {', '.join(unknown)}, which is not in SUPPORTED_PYTHONS ({', '.join(SUPPORTED_PYTHONS)}) of scripts/ci/_common.py")
    twice = sorted({p for p in pythons if pythons.count(p) > 1})
    if twice:      # two equal matrix entries would run the same job twice and upload two artifacts with one name
        raise CiError(f"{label}: `pythons` names {', '.join(twice)} more than once")
    return tuple(pythons)


def shards_of(leg: dict, label: str = "a leg") -> int:
    """The number of parallel jobs that share the pytest run of a leg (`shards` in legs.toml). The default is 1 (no split).
    The value must be a whole number of 1 or more (a bool or a float is refused)."""
    shards = leg.get("shards", 1)
    if isinstance(shards, bool) or not isinstance(shards, int) or shards < 1:
        raise CiError(f"{label}: `shards` in legs.toml must be a whole number of 1 or more, not {shards!r}")
    return shards


def workflow_of(leg: dict) -> str:
    """The workflow of a leg (`workflow` in legs.toml): `tests` (.github/workflows/tests.yml) unless the leg names another."""
    return leg.get("workflow", DEFAULT_WORKFLOW)


def get_tier(name: str, spec: dict | None = None) -> dict:
    spec = load_spec() if spec is None else spec
    tiers = spec.get("tiers", {})
    if name not in tiers:
        raise CiError(f"unknown tolerance tier {name!r}; the tiers in legs.toml are: {', '.join(sorted(tiers))}")
    return tiers[name]


def write_summary(markdown: str) -> None:
    """Append text to the job summary of GitHub Actions. Outside Actions, do nothing."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown.rstrip("\n") + "\n\n")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def loadavg() -> str:
    """The load average as text (1, 5 and 15 minutes), or 'not known'."""
    try:
        return " ".join(f"{x:.2f}" for x in os.getloadavg())
    except OSError:
        return "not known"


def run_main(main) -> None:
    """Call `main()` (it returns an exit code). A `CiError` prints its text and gives exit code 1."""
    try:
        code = main()
    except CiError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        code = 1
    sys.exit(code)
