#!/usr/bin/env python3
"""Install what one CI leg needs into the Python that runs this script (a venv).

    .venv/bin/python scripts/ci/install_env.py --leg linux-x86_64

Steps:
1. System packages of the leg (`system_packages` in legs.toml). Linux: apt. Skip with --no-system.
2. The Python packages, in the `install_mode` of the leg in legs.toml. The lock is `lock` of the leg (default `requirements.lock`;
   the Intel legs use `requirements-macos-x86_64.lock`). The lock gives the pinned version of every package that it names.
   - `project` (default): `uv pip install -c <lock> -e ".[extras]"`.
   - `lock-first` (the Intel legs): close to the install of the launcher on an x86_64 Mac (`scripts/topopt-env.sh`, step 2 and 3 of `install_all`).
     `uv pip install --only-binary numba --only-binary llvmlite --only-binary cryptography -r <lock>` (the same three packages as the launcher: a pin that
     has no wheel stops the install and does not build from source), then `uv pip install -c <lock> --no-deps --no-build-isolation -e .` (the link of
     the launcher: setuptools of the lock, no index), then the requirements of the extras of the leg, with `-c <lock>`. The launcher on main before PR #50 spells the first flags `--no-build-package <name>`. `uv pip install` of uv 0.12.1 and
     0.13.4 rejects that spelling ("unexpected argument", exit 2; it exists for `uv sync`); `--only-binary <name>` is the accepted form for one package, with the
     same effect. PR #50 changes the launcher to `--only-binary <name>`; then the flag spelling is no difference. Other
     differences from the launcher: the third command (the launcher installs no extra),
     and the fallback of the launcher (`pyproject.toml` pins when the lock install fails), which the leg does not test. A Linux leg cannot use `project`
     with the Intel lock: the platform markers of pyproject.toml ask for numba 0.67.0 there, and the lock holds numba 0.62.1.
     The extras are those of the leg that pyproject.toml declares: an extra that this commit does not declare (for example `verify` before PR #21)
     is dropped, and the log says so.
3. The Accelerate shim (`sh topopt/native/build.sh`), if the leg has `build_shim = true`.
4. A list of the installed packages, in `<artifacts>/pip-freeze.txt`.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, get_leg, lock_of, run_main, write_summary  # noqa: E402


def declared_extras(pyproject_text: str) -> list[str]:
    return list(tomllib.loads(pyproject_text)["project"].get("optional-dependencies", {}))


INSTALL_MODES = ("project", "lock-first")
# The packages that the launcher never builds from source on an x86_64 Mac (`no_build_flags` in scripts/topopt-env.sh; a test compares the names).
NO_BUILD_PACKAGES = ("numba", "llvmlite", "cryptography")


def extra_requirements(pyproject_text: str, extras: list[str]) -> list[str]:
    """The requirement strings of the given extras in pyproject.toml, in order, each once (markers stay in the string)."""
    declared = tomllib.loads(pyproject_text)["project"].get("optional-dependencies", {})
    out: list[str] = []
    for e in extras:
        out += [r for r in declared.get(e, []) if r not in out]
    return out


def split_extras(wanted: list[str], declared: list[str]) -> tuple[list[str], list[str]]:
    """Return (extras to install, extras that this commit does not declare). The order is that of `wanted`."""
    return [e for e in wanted if e in declared], [e for e in wanted if e not in declared]


def install_commands(leg: dict, extras: list[str], python: str, *, system: bool, platform: str,
                     extra_reqs: list[str] | None = None) -> list[list[str]]:
    """The commands of steps 1 to 3 as argument lists, in order. `extra_reqs` (the requirements of the extras) is used in mode `lock-first`."""
    cmds: list[list[str]] = []
    if system and leg.get("system_packages") and platform.startswith("linux"):
        cmds.append(["sudo", "apt-get", "update", "-qq"])
        cmds.append(["sudo", "apt-get", "install", "-y", "-qq", *leg["system_packages"]])
    mode = leg.get("install_mode", "project")
    if mode not in INSTALL_MODES:
        raise CiError(f"install_mode {mode!r} is not one of {', '.join(INSTALL_MODES)}")
    lock, uv_install = lock_of(leg), ["uv", "pip", "install", "--python", python]
    if mode == "lock-first":
        no_build = [arg for name in NO_BUILD_PACKAGES for arg in ("--only-binary", name)]      # the launcher on main before PR #50 says --no-build-package; uv pip rejects that
        cmds.append([*uv_install, *no_build, "-r", lock])
        cmds.append([*uv_install, "-c", lock, "--no-deps", "--no-build-isolation", "-e", "."])   # the link of the launcher (#397)
        if extra_reqs:
            cmds.append([*uv_install, "-c", lock, *extra_reqs])
    else:
        target = ".[" + ",".join(extras) + "]" if extras else "."
        cmds.append([*uv_install, "-c", lock, "-e", target])
    if leg.get("build_shim") and platform == "darwin":
        cmds.append(["sh", "topopt/native/build.sh"])
    return cmds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leg", required=True, help="a leg name of legs.toml")
    ap.add_argument("--no-system", action="store_true", help="do not install system packages")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and run none")
    ap.add_argument("--artifacts", default="artifacts", help="folder for pip-freeze.txt")
    args = ap.parse_args(argv)

    leg = get_leg(args.leg)
    pyproject_text = (ROOT / "pyproject.toml").read_text()
    extras, dropped = split_extras(leg["extras"], declared_extras(pyproject_text))
    print(f"leg {args.leg}: lock {lock_of(leg)} (install_mode {leg.get('install_mode', 'project')}), extras {extras}" + (f"; dropped (not declared in pyproject.toml): {dropped}" if dropped else ""))
    if not (ROOT / lock_of(leg)).is_file():
        raise CiError(f"leg {args.leg} names the lock {lock_of(leg)!r}, and the repo has no such file")
    if not args.dry_run and not shutil.which("uv"):
        raise CiError("uv is not on PATH (the workflow installs it with astral-sh/setup-uv)")
    cmds = install_commands(leg, extras, sys.executable, system=not args.no_system, platform=sys.platform,
                            extra_reqs=extra_requirements(pyproject_text, extras))
    for cmd in cmds:
        print("+ " + shlex.join(cmd), flush=True)
        if not args.dry_run and subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise CiError(f"command failed: {shlex.join(cmd)}")
    if not args.dry_run:
        out = Path(args.artifacts).resolve()
        out.mkdir(parents=True, exist_ok=True)
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True, cwd=ROOT)
        (out / "pip-freeze.txt").write_text(freeze.stdout)
        print(f"{len(freeze.stdout.splitlines())} packages installed; list in {out / 'pip-freeze.txt'}")
    note = f"; dropped (not in pyproject.toml): `{', '.join(dropped)}`" if dropped else ""
    write_summary(f"### Install ({args.leg})\n\nLock: `{lock_of(leg)}` (mode `{leg.get('install_mode', 'project')}`). Extras: `{', '.join(extras)}`{note}.")
    return 0


if __name__ == "__main__":
    run_main(main)
