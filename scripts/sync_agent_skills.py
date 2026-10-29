#!/usr/bin/env python3
"""Copy the skills that have a copy in `.agents/skills/` from their source, or check that the copies are equal.

Codex CLI, Gemini CLI, Cursor and VS Code read skills from `.agents/skills/`. Claude Code reads `skills/` (plugin)
and `.claude/skills/`. The sources stay where they are. `.agents/skills/<name>/` is a byte copy of its source.

    python3 scripts/sync_agent_skills.py            copy each source over its copy
    python3 scripts/sync_agent_skills.py --check    print the differences and exit with 1 if there are any

Run the copy after each change of a source skill. `tests/test_client_configs.py` imports this file and runs the
same check, so the test and the command agree. The file uses the standard library only (Python 3.8 or newer).

Which files count: in a git checkout, the files that git tracks and the files that git would track (not ignored).
So a `.DS_Store` or a `__pycache__` that `.gitignore` names does not count. Outside git (for example an sdist),
the script skips `__pycache__`, `*.pyc` and `.DS_Store`. A symlink is a difference (some tools refuse symlinks).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# (source folder, copy folder), both relative to the repo root.
PAIRS = [
    ("skills/start-here", ".agents/skills/start-here"),
    (".claude/skills/part-to-topopt", ".agents/skills/part-to-topopt"),
]

_JUNK_NAMES = {".DS_Store"}


def _is_junk(rel: Path) -> bool:
    return "__pycache__" in rel.parts or rel.suffix == ".pyc" or rel.name in _JUNK_NAMES


def _git_files(root: Path, folder: str) -> "set[str] | None":
    """Paths (relative to `folder`) of the files that git tracks or would track. None if git cannot tell."""
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", folder],
            capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = set()
    for raw in out.split(b"\0"):
        if not raw:
            continue
        path = root / os.fsdecode(raw)
        if os.path.lexists(path) and (path.is_symlink() or not path.is_dir()):          # `--cached` also lists a file that is deleted
            found.add(path.relative_to(root / folder).as_posix())
    return found


def files_of(root: Path, folder: str) -> "dict[str, Path]":
    """The files of one skill folder: relative path -> path. See the module text for which files count."""
    base = root / folder
    names = _git_files(root, folder)
    if names is None:
        names = set()
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames + [d for d in dirnames if (Path(dirpath) / d).is_symlink()]:
                rel = (Path(dirpath) / name).relative_to(base)
                if not _is_junk(rel):
                    names.add(rel.as_posix())
    return {name: base / name for name in sorted(names)}


def _exec_bit(path: Path) -> bool:
    return bool(path.lstat().st_mode & 0o111)


def differences(root: Path, pair: "tuple[str, str]") -> "list[str]":
    """What differs between the copy and its source. An empty list means the copy is equal."""
    source, copy = pair
    src, dst = files_of(root, source), files_of(root, copy)
    found = []
    if not src:
        found.append(f"{source}: the source has no files")
    for rel in sorted(set(src) - set(dst)):
        found.append(f"{copy}/{rel}: missing in the copy")
    for rel in sorted(set(dst) - set(src)):
        found.append(f"{copy}/{rel}: not in the source")
    for rel in sorted(set(src) & set(dst)):
        a, b = src[rel], dst[rel]
        if a.is_symlink() or b.is_symlink():
            found.append(f"{rel}: a symlink ({a if a.is_symlink() else b}); use a regular file")
        elif a.read_bytes() != b.read_bytes():
            found.append(f"{copy}/{rel}: the content differs from {source}/{rel}")
        elif _exec_bit(a) != _exec_bit(b):
            found.append(f"{copy}/{rel}: the executable bit differs from {source}/{rel}")
    return found


def copy_pair(root: Path, pair: "tuple[str, str]") -> "list[str]":
    """Make the copy equal to its source. Returns one line for each file that changed."""
    source, copy = pair
    src, dst = files_of(root, source), files_of(root, copy)
    changed = []
    for rel in sorted(set(dst) - set(src)):                       # a file that the source no longer has
        os.remove(dst[rel])
        changed.append(f"removed {copy}/{rel}")
    for rel, path in src.items():
        target = root / copy / rel
        if path.is_symlink():
            raise SystemExit(f"{source}/{rel} is a symlink. Use a regular file (some tools refuse symlinks).")
        if target.is_symlink():
            action = "updated"
        elif not target.exists():
            action = "added"
        elif target.read_bytes() != path.read_bytes():
            action = "updated"
        elif _exec_bit(path) != _exec_bit(target):
            action = "mode"
        else:
            continue
        if target.is_symlink():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        shutil.copymode(path, target)
        changed.append(f"{action} {copy}/{rel}")
    # Remove folders that the removal left empty.
    for dirpath, dirnames, filenames in os.walk(root / copy, topdown=False):
        if not dirnames and not filenames and Path(dirpath) != root / copy:
            os.rmdir(dirpath)
    return changed


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="change nothing; exit with 1 if a copy differs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="the repo root (default: the folder above scripts/)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    status = 0
    for pair in PAIRS:
        if args.check:
            for line in differences(root, pair):
                print(line)
                status = 1
        else:
            changed = copy_pair(root, pair)
            for line in changed:
                print(line)
            if not changed:
                print(f"equal: {pair[1]}")
    if args.check and status:
        print("Fix: python3 scripts/sync_agent_skills.py   (the source is the one to edit)", file=sys.stderr)
    elif args.check:
        print("all copies equal their sources")
    return status


if __name__ == "__main__":
    sys.exit(main())
