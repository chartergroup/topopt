"""Attach the built files of a release (the wheel, the sdist and the `.mcpb`, the last also as `topopt.mcpb`) to its GitHub release,
and check that they are there.

The release workflow (`.github/workflows/release.yml`) runs this script after `semantic-release version` (RELEASING.md,
"What the workflow does", point 7). The tool uploads only the `assets` of `pyproject.toml` (`install.md`, `CHANGELOG.md`)
to the release that it makes. The files in `dist/` would need its `publish` command, which the workflow does not run, so no
release up to v0.3.0 had them. This script reads the asset list of the release, uploads the files that it does not list yet
(`gh release upload`), reads the list again and exits with code 1 when one of the four is missing. It never replaces a listed
file: a new build gives other bytes for the wheel and the sdist (their archives hold the build time), so a repeat changes nothing.
The `.mcpb` is the file that a Claude Desktop user downloads. `topopt.mcpb` is a byte-equal copy of `topopt-X.Y.Z.mcpb` with a name
that does not change (#409), so `releases/latest/download/topopt.mcpb` always gives the file of the newest release. When the release
lists `topopt-X.Y.Z.mcpb` already (a run for an older release), the copy is made from the listed file (`gh release download`), not from
a new build, and the run fails when the two assets differ in size. `topopt.mcpb` goes up in a second upload, after the release lists
`topopt-X.Y.Z.mcpb`; a release that lists `topopt.mcpb` without `topopt-X.Y.Z.mcpb` fails the run before any upload.

A file to upload that is not in `dist/` yet is built first: a new run after a failed release run has no build step, and a run for
an older release builds the checkout of its tag (`--source`). The build is the one of the build step: `uv build --wheel --sdist`
and `packaging/mcpb/build.py` of the source folder. The version in the `pyproject.toml` of the source folder must be the
version of the tag.

    python3 -P scripts/ci/release_assets.py --tag vX.Y.Z [--source DIR] [--dist DIR]

`--source` is the checkout of the tag (default: the repo of this script). `--dist` is where the files are and go (default: `dist`
of the repo of this script). `GH_TOKEN` must allow the upload (contents: write). The script names no repo: `gh` works on the repo of
the checkout, and `GH_REPO=owner/name` points it to another one (for example a public copy of the repo).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, run_main  # noqa: E402

TAG = re.compile(r"v(\d+\.\d+\.\d+)")
STABLE_MCPB = "topopt.mcpb"   # the name without the version: releases/latest/download/topopt.mcpb (#409)


def version_of_tag(tag: str) -> str:
    m = TAG.fullmatch(tag)
    if m is None:
        raise CiError(f"the tag {tag!r} is not of the form vX.Y.Z")
    return m.group(1)


def source_version(source: Path) -> str:
    return tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def built_files(dist: Path, version: str) -> list[Path]:
    """The three files of a release in `dist` (the names that the build gives; a test checks them against `dist_glob_patterns`)."""
    return [dist / f"topopt-{version}-py3-none-any.whl", dist / f"topopt-{version}.tar.gz", dist / "mcpb" / f"topopt-{version}.mcpb"]


def build(source: Path, dist: Path, run) -> None:
    """The build of the build step (`build_command` in pyproject.toml) for the folder `source`, into `dist`."""
    run(["uv", "build", "--wheel", "--sdist", "--out-dir", str(dist)], cwd=source)
    run([sys.executable, "-P", "packaging/mcpb/build.py", "--out", str(dist / "mcpb")], cwd=source)


def release_assets(tag: str, run) -> dict[str, int]:
    """The assets of the release: name -> size in bytes."""
    out = run(["gh", "release", "view", tag, "--json", "assets"], capture=True)
    try:
        return {a["name"]: int(a["size"]) for a in json.loads(out)["assets"]}
    except (ValueError, KeyError, TypeError) as e:
        raise CiError(f"gh release view {tag} gave no asset list: {e}") from None


def check_same_mcpb(tag: str, assets: dict[str, int], mcpb: str) -> None:
    """The two names of the extension must be the same file: a size that differs fails the run."""
    if mcpb in assets and STABLE_MCPB in assets and assets[mcpb] != assets[STABLE_MCPB]:
        raise CiError(f"the release {tag} lists {mcpb} with {assets[mcpb]} bytes and {STABLE_MCPB} with {assets[STABLE_MCPB]} bytes: "
                      f"the two must be the same file (delete the wrong one on the release page, then run again)")


def missing(files: list[Path], names: list[str]) -> list[str]:
    """The names of the files that the asset list of the release does not have."""
    return [f.name for f in files if f.name not in names]


def _run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    print("+ " + " ".join(cmd), flush=True)
    try:
        done = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE if capture else None)
    except FileNotFoundError:
        raise CiError(f"{cmd[0]} is not installed (not on PATH)") from None
    if done.returncode != 0:
        raise CiError(f"{' '.join(cmd[:3])} failed with exit code {done.returncode}")
    return done.stdout or ""


def attach(tag: str, source: Path, dist: Path, run=_run) -> list[Path]:
    version = version_of_tag(tag)
    found = source_version(source)
    if found != version:
        raise CiError(f"the pyproject.toml of {source} has the version {found}, not {version} of the tag {tag}")
    files = built_files(dist, version)
    mcpb, stable = files[2], dist / "mcpb" / STABLE_MCPB
    allf = [*files, stable]
    assets = release_assets(tag, run)
    listed = list(assets)
    lacking = missing(allf, listed)
    if stable.name in listed and mcpb.name not in listed:
        raise CiError(f"the release {tag} lists {STABLE_MCPB} but not {mcpb.name}, so the copy has no file to match "
                      f"(delete {STABLE_MCPB} on the release page, then run again)")
    check_same_mcpb(tag, assets, mcpb.name)   # before any upload
    todo = [f for f in files if f.name in lacking]
    if not lacking:
        print(f"The release {tag} lists {', '.join(f.name for f in allf)} already: nothing to upload.")
        return allf
    if not all(f.is_file() for f in todo):
        print(f"Not all files to upload are in {dist}: building them from {source}.", flush=True)
        build(source, dist, run)
        absent = [f.name for f in todo if not f.is_file()]
        if absent:
            raise CiError(f"the build of {source} did not make {', '.join(absent)} in {dist}")
    if todo:
        run(["gh", "release", "upload", tag, *map(str, todo)])
        assets = release_assets(tag, run)
        gone = missing(files, list(assets))
        if gone:
            raise CiError(f"the release {tag} does not list {', '.join(gone)} after the upload")
    if stable.name in lacking:
        # A second upload, after the release lists topopt-X.Y.Z.mcpb: topopt.mcpb never goes up alone, so a later run never
        # uploads a new build under the versioned name next to an older copy.
        if mcpb.name in listed:   # the copy must have the bytes of the listed file, not of a new build
            mcpb.parent.mkdir(parents=True, exist_ok=True)   # nothing was built when only the copy is missing
            run(["gh", "release", "download", tag, "--pattern", mcpb.name, "--dir", str(mcpb.parent), "--clobber"])
            if not mcpb.is_file():
                raise CiError(f"gh release download {tag} did not write {mcpb.name} to {mcpb.parent}")
        shutil.copyfile(mcpb, stable)
        run(["gh", "release", "upload", tag, str(stable)])
        assets = release_assets(tag, run)
    gone = missing(allf, list(assets))
    if gone:
        raise CiError(f"the release {tag} does not list {', '.join(gone)} after the upload")
    check_same_mcpb(tag, assets, mcpb.name)
    print(f"The release {tag} has {', '.join(f.name for f in allf)}.")
    return allf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Attach the wheel, the sdist and the .mcpb (also as topopt.mcpb) of a release to its GitHub release, "
                                            "and check them.")
    p.add_argument("--tag", required=True, help="the release tag vX.Y.Z")
    p.add_argument("--source", type=Path, default=ROOT, help="the checkout of the tag (default: this repo)")
    p.add_argument("--dist", type=Path, default=ROOT / "dist", help="the folder of the built files (default: dist of this repo)")
    a = p.parse_args(argv)
    attach(a.tag, a.source.resolve(), a.dist.resolve())
    return 0


if __name__ == "__main__":
    run_main(main)
