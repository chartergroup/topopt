#!/usr/bin/env python3
"""Check that each lock file still resolves, with wheels only, for the platform that uses it (E4).

    .venv/bin/python scripts/ci/check_locks.py [--leg linux-x86_64] [--python 3.13] [--no-controls] [--artifacts artifacts]

For each target the script runs `uv pip compile <lock> --python-platform <P> --python-version <V> --no-build`, for each supported Python version V
(SUPPORTED_PYTHONS in _common.py, the one place that names them: 3.13 only since #469). With `--python V` it checks that version only; the CI
step passes no `--python`, so it checks every supported version.
`--no-build` makes uv use wheels only, so a pin without a wheel for the platform and the Python stops the resolution.
A target is ok only when it resolves and the resolved name-to-version map equals the map of the lock. An added, a dropped or a changed pin is
a wrong result, and the detail lists it. A lock line with an environment marker (`name==1 ; sys_platform == 'linux'`) is a wrong result too:
a marker can hide a pin on a platform, and the comparison would not see it. An empty lock is a wrong result.

| target | lock | platform |
| intel  | requirements-macos-x86_64.lock | x86_64-apple-darwin |
| main   | requirements.lock | aarch64-apple-darwin |
| main   | requirements.lock | x86_64-unknown-linux-gnu |

A control is a case that must fail. It shows that a pass of a target means something. A control passes only when uv says
"No solution found" and names a pin of the control with the text `<name>==<version> has no usable wheels`, so neither a network error
nor a version conflict counts as a pass. That is the wording of uv 0.12.1, the version that the two workflow files pin (UV_TEXT_VERSION).
Another wording gives a false fail (the control is not ok), never a false pass. The controls are the main lock for x86_64-apple-darwin,
and the Intel lock with pins that have no macOS x86_64 wheel: cryptography, and numba with llvmlite (they move together: numba 0.67.0
with llvmlite 0.45.1 is a version conflict on every platform, and a conflict is not a missing wheel).

A package that is missing from the lock shows as an added pin when another pin needs it. If no other pin needs it, only the name test of
`tests/test_plugin_launcher.py` and `tests/test_locks.py` (same names in both locks) sees it.
`uv` assumes one macOS version for x86_64-apple-darwin: a pass proves a wheel for that version, not for every Intel Mac OS version.

With `--leg`, the script runs only when the leg has `check_locks = true` in legs.toml. It needs `uv` on PATH and the package index.
Each compile log goes to `<artifacts>/lock-checks/`. The exit code is 1 when a target or a control gives the wrong result.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, SUPPORTED_PYTHONS, CiError, get_leg, md_table, run_main, write_summary  # noqa: E402

# The uv version whose error text this script reads. The workflows pin it (astral-sh/setup-uv, `version`); a test compares the two.
UV_TEXT_VERSION = "0.12.1"
# What uv 0.12.1 says when a pin has no wheel for the platform and building from source is off (`--no-build`).
NO_WHEELS = "has no usable wheels"
MAIN_LOCK = "requirements.lock"
INTEL_LOCK = "requirements-macos-x86_64.lock"
INTEL_PLATFORM = "x86_64-apple-darwin"

# Each lock with the platform that uses it. A target must resolve.
TARGETS = (
    {"name": "Intel lock on x86_64-apple-darwin", "lock": INTEL_LOCK, "platform": INTEL_PLATFORM},
    {"name": "main lock on aarch64-apple-darwin", "lock": MAIN_LOCK, "platform": "aarch64-apple-darwin"},
    {"name": "main lock on x86_64-unknown-linux-gnu", "lock": MAIN_LOCK, "platform": "x86_64-unknown-linux-gnu"},
)

# Cases that must fail. `pins` replace the lines of those packages in the lock text (pins that have no macOS x86_64 wheel; empty: the lock as it is).
CONTROLS = (
    {"name": "main lock on x86_64-apple-darwin", "lock": MAIN_LOCK, "platform": INTEL_PLATFORM, "pins": (),
     "meaning": "the main pins have no wheel there, so the Intel lock is needed"},
    {"name": "Intel lock with numba==0.67.0 and llvmlite==0.49.0", "lock": INTEL_LOCK, "platform": INTEL_PLATFORM,
     "pins": (("numba", "0.67.0"), ("llvmlite", "0.49.0")),
     "meaning": "the check sees a pin with no macOS x86_64 wheel (numba and llvmlite move together, so no version conflict can fail it)"},
    {"name": "Intel lock with cryptography==50.0.2", "lock": INTEL_LOCK, "platform": INTEL_PLATFORM, "pins": (("cryptography", "50.0.2"),),
     "meaning": "the same for a second package"},
)


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(text: str, where: str = "the lock") -> dict[str, str]:
    """Name to version for every `name==version` line of a lock file or of the output of `uv pip compile`.

    A line with an environment marker is an error (`CiError`): the check compares names and versions, and a marker can hide a pin on a platform.
    """
    pins = {}
    for n, line in enumerate(text.splitlines(), 1):
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", line)
        if m:
            if line[m.end():].lstrip().startswith(";"):
                raise CiError(f"{where}, line {n} has an environment marker ({line.strip()!r}). The lock check compares names and versions only, and a marker "
                              "can hide a pin on a platform. Write the pin without a marker.")
            pins[canonical(m.group(1))] = m.group(2)
    return pins


def compare_pins(lock: dict[str, str], resolved: dict[str, str]) -> str:
    """Empty when the two maps are equal, else the added, dropped and changed pins as text."""
    parts = []
    added = sorted(f"{n}=={resolved[n]}" for n in set(resolved) - set(lock))
    dropped = sorted(f"{n}=={lock[n]}" for n in set(lock) - set(resolved))
    changed = sorted(f"{n} {lock[n]} -> {resolved[n]}" for n in set(lock) & set(resolved) if lock[n] != resolved[n])
    if added:
        parts.append("added (resolved, not in the lock): " + ", ".join(added))
    if dropped:
        parts.append("dropped (in the lock, not resolved): " + ", ".join(dropped))
    if changed:
        parts.append("changed (lock -> resolved): " + ", ".join(changed))
    return "; ".join(parts)


def with_pin(lock_text: str, name: str, version: str) -> str:
    """The lock text with the pin of `name` replaced. A lock without a line for `name` is an error."""
    pattern = re.compile(rf"^{re.escape(name)}==[^\s;\\]+", re.M | re.I)
    if not pattern.search(lock_text):
        raise CiError(f"cannot make the control '{name}=={version}': the lock has no line for {name}")
    return pattern.sub(f"{name}=={version}", lock_text, count=1)


def _flat(text: str) -> str:
    """The uv text on one line: the box characters of its error layout are gone and the line breaks are one space (uv wraps long lines)."""
    return " ".join(re.sub(r"[│├└╰─▶×]", " ", text).split())


def pins_without_wheels(text: str, pins) -> list[str]:
    """The `name==version` of the control `pins` that the uv text names with `has no usable wheels`. With no `pins`: any pin that it names so."""
    flat = _flat(text)
    if not pins:
        return re.findall(rf"(\S+==\S+) {NO_WHEELS}", flat)
    return [f"{n}=={v}" for n, v in pins if f"{n}=={v} {NO_WHEELS}" in flat]


def classify(returncode: int, text: str) -> str:
    """`resolved`, `unresolvable` (uv found no solution) or `error` (uv missing, network, anything else)."""
    if returncode == 0:
        return "resolved"
    if "No solution found" in text:
        return "unresolvable"
    return "error"


def run_compile(uv: str, lock: Path, platform: str, out: Path, python: str) -> tuple[int, str]:
    """One `uv pip compile` for the Python version `python`. Return (exit code, output text). The resolved pins go to `out`."""
    cmd = [uv, "pip", "compile", str(lock), "--python-platform", platform, "--python-version", python,
           "--no-build", "--no-header", "--no-annotate", "-o", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout + p.stderr


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", text).strip("_")


def check(runner, root: Path, workdir: Path, controls: bool = True) -> list[dict]:
    """Run the targets and the controls. `runner(lock, platform, out)` returns (exit code, text).

    One row for each case: name, kind (`target` or `control`), state, ok, detail. A target is ok when it resolved and the resolved pins equal the pins
    of the lock (state `differs` otherwise; `invalid lock` for a lock with a marker line or with no pin).
    A control is ok when the resolution failed with "No solution found" and the text says `<pin> has no usable wheels` for a pin of the control.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in TARGETS:
        lock = root / t["lock"]
        if not lock.is_file():
            raise CiError(f"{lock} does not exist")
        try:
            have = parse_pins(lock.read_text(), where=lock.name)
        except CiError as e:
            rows.append({"name": t["name"], "kind": "target", "state": "invalid lock", "ok": False, "detail": str(e)})
            continue
        if not have:
            rows.append({"name": t["name"], "kind": "target", "state": "invalid lock", "ok": False,
                         "detail": f"{lock.name} has no `name==version` line, so there is nothing to check"})
            continue
        out = workdir / f"{_slug(t['name'])}.pins.txt"
        rc, text = runner(lock, t["platform"], out)
        (workdir / f"{_slug(t['name'])}.log").write_text(text)
        state = classify(rc, text)
        if state == "resolved":
            got = parse_pins(out.read_text() if out.is_file() else "", where="the resolved pins")
            diff = compare_pins(have, got)
            detail = f"{len(got)} pins resolved, {len(have)} in the lock" + (f"; the resolved pins differ from the lock. {diff}" if diff else "")
            if diff:
                state = "differs"
        else:
            detail = _first_error(text)
        rows.append({"name": t["name"], "kind": "target", "state": state, "ok": state == "resolved", "detail": detail})
    if controls:
        for c in CONTROLS:
            lock = root / c["lock"]
            if not lock.is_file():
                raise CiError(f"{lock} does not exist")
            src = lock
            if c["pins"]:
                src = workdir / f"{_slug(c['name'])}.in.txt"
                text = lock.read_text()
                try:
                    for pin in c["pins"]:
                        text = with_pin(text, *pin)
                except CiError as e:                       # the lock has no line for the package: the control cannot be made, and that is a wrong result
                    rows.append({"name": c["name"], "kind": "control", "state": "invalid lock", "ok": False, "detail": str(e)})
                    continue
                src.write_text(text)
            out = workdir / f"{_slug(c['name'])}.pins.txt"
            rc, text = runner(src, c["platform"], out)
            (workdir / f"{_slug(c['name'])}.log").write_text(text)
            state = classify(rc, text)
            missing = pins_without_wheels(text, c["pins"])
            ok = state == "unresolvable" and bool(missing)
            if ok:
                detail = _first_error(text)
            elif state == "unresolvable":
                want = " or ".join(f"{n}=={v}" for n, v in c["pins"]) or "a pin"
                detail = (f"uv found no solution, but the text does not say '{want} {NO_WHEELS}' (the wording of uv {UV_TEXT_VERSION}; another cause, "
                          f"such as a version conflict, does not count): {_first_error(text)}")
            elif state == "resolved":
                detail = (f"the control resolved, so this check could not see this drift ({c['meaning']}). "
                          "Either the control is out of date, or the pins changed so that it resolves now")
            else:
                detail = f"uv failed for another reason (network?): {_first_error(text)}"
            rows.append({"name": c["name"], "kind": "control", "state": state, "ok": ok, "detail": detail})
    return rows


def _first_error(text: str) -> str:
    """The cause line of a uv error: the line that has 'Because' (with the next line), else the first line that has text."""
    lines = [ln.strip(" ╰─▶×") for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.startswith("Because"):
            return " ".join(lines[i:i + 2])[:300]
    return (lines[0] if lines else "no output")[:300]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leg", help="a leg name of legs.toml; the script runs only when the leg has check_locks = true")
    ap.add_argument("--python", choices=SUPPORTED_PYTHONS, help="check this Python version only (default: every supported version)")
    ap.add_argument("--no-controls", action="store_true", help="run the targets only")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--uv", default=None, help="path of uv (default: uv on PATH)")
    args = ap.parse_args(argv)

    if args.leg and not get_leg(args.leg).get("check_locks", False):
        print(f"leg {args.leg}: check_locks is not set in legs.toml; nothing to do")
        return 0
    uv = args.uv or shutil.which("uv")
    if not uv:
        raise CiError("uv is not on PATH (the workflow installs it with astral-sh/setup-uv)")
    versions = (args.python,) if args.python else SUPPORTED_PYTHONS
    workdir = Path(args.artifacts).resolve() / "lock-checks"
    rows = []
    for v in versions:      # the logs of each version go to their own folder, so that one version never overwrites the other
        got = check(lambda lock, plat, out, v=v: run_compile(uv, lock, plat, out, v), ROOT, workdir / f"python-{v}", controls=not args.no_controls)
        rows += [{**r, "python": v} for r in got]
    table = md_table(["case", "Python", "kind", "state", "result", "detail"],
                     [[r["name"], r["python"], r["kind"], r["state"], "ok" if r["ok"] else "WRONG", r["detail"]] for r in rows])
    print(table)
    write_summary(f"### Lock files ({args.leg or 'no leg'}): `uv pip compile --no-build`, Python {', '.join(versions)}\n\n{table}")
    bad = [r for r in rows if not r["ok"]]
    if bad:
        raise CiError(f"{len(bad)} case(s) gave the wrong result: "
                      + "; ".join(f"{r['name']} ({r['kind']}, {r['state']}, Python {r['python']})" for r in bad))
    print(f"OK: {sum(r['kind'] == 'target' for r in rows)} target(s) resolve"
          + ("" if args.no_controls else f", {sum(r['kind'] == 'control' for r in rows)} control(s) fail as they should")
          + f" (Python {', '.join(versions)})")
    return 0


if __name__ == "__main__":
    run_main(main)
