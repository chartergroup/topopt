#!/usr/bin/env python3
"""Short release notes for people: one line per merged pull request (#334).

The release workflow (`.github/workflows/release.yml`, `RELEASING.md`) calls this script. It reads the first-parent line of the
branch between the last release tag and HEAD, and it prints the entry of the new version:

    ## vX.Y.Z (date)

    _This release is published under the MIT License._

    ### Features

    - Sentence ([#N](https://github.com/<repo>/pull/N)).

    ---

    **Detailed Changes**: [vOLD...vX.Y.Z](https://github.com/<repo>/compare/vOLD...vX.Y.Z)

Where the data comes from:
- A merge commit `Merge pull request #N from <org>/<branch>` is one pull request. Its body is the PR title.
  The types of the commits that the PR brought (`<merge>^1..<merge>^2`, no merge commits) choose the section:
  the highest type of the title and of these commits wins (feat, fix, perf, docs, chore, ci, test, then the others).
- A squash commit `type(scope): subject (#N)` is one pull request too.
- A squash merge that kept the merge message (one parent, the subject `Merge pull request #N from ...`, the PR title as the first body
  line; #374 made one) is one pull request: the title gives the type, and a `!` or a `BREAKING CHANGE:` line in the body marks it (#442).
- A release commit `chore(release): X.Y.Z` is skipped.
- A direct commit (no PR) gives a line without a link, in the section of its own type. Other merge commits are skipped.

The cleaning of a title: drop `type(scope)!:`, drop the trailing item codes (`(F4-47)`, `(F1-8 step 2)`, `(K1-FU-C)`, `(U21)`) and the
suffix `(requested by <name>)`, put an upper case letter first (not on a code name such as `list_runs`), end the sentence with one period (after the link). Nothing else is
reworded. A `!` after the type, or a `BREAKING CHANGE:` footer in a commit of the PR, puts `**Breaking:**` before the sentence.

Usage (stdlib only; run it as `python3 -P scripts/ci/release_notes.py`):
    release_notes.py --version 0.3.0 [--prev-tag v0.2.0] [--head HEAD] [--date YYYY-MM-DD] [--repo org/name]
                     [--write-changelog CHANGELOG.md] [--notes-file PATH]
With no `--prev-tag` the last tag `v*` that is an ancestor of HEAD is used. In the release build step `--version` defaults to the
environment variable NEW_VERSION that python-semantic-release sets.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REPO = "chartergroup/topopt"
LICENSE_LINE = "_This release is published under the MIT License._"
INSERT_MARKER = "<!-- version list -->"

# Section order. A type that is not in the table goes after these, in alphabetical order; no type goes last.
SECTIONS = [
    ("feat", "Features"),
    ("fix", "Bug Fixes"),
    ("perf", "Performance Improvements"),
    ("docs", "Documentation"),
    ("chore", "Chores"),
    ("ci", "Continuous Integration"),
    ("test", "Testing"),
]
OTHER_TITLES = {"refactor": "Refactoring", "build": "Build System", "style": "Code Style", "revert": "Reverts"}
NO_TYPE = ""
NO_TYPE_TITLE = "Other Changes"

TYPE_RE = re.compile(r"^(?P<type>[A-Za-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<rest>.+)$", re.S)
MERGE_RE = re.compile(r"^Merge pull request #(?P<n>\d+) from \S+")
SQUASH_RE = re.compile(r"\s*\(#(?P<n>\d+)\)\s*$")
RELEASE_RE = re.compile(r"^chore\(release\):\s*\d+\.\d+\.\d+")
BREAKING_RE = re.compile(r"^BREAKING[ -]CHANGE:", re.M)
SUFFIX_RE = re.compile(r"\s*\((?:requested by [^()]*)\)\s*$", re.I)
# An item code is one of two forms, as the first word in the trailing brackets:
#   A: one capital letter, 1 or 2 digits, an optional lower-case letter (U21, I2, H1, P4b, B1);
#   B: a word that starts with a capital letter, has a dash and has a digit (F4-47, K1-FU-C, LAUNCH-3, P2d-D3, Q1-fix, PM-2).
# Then come optional words after a space, a comma, a plus or a colon ("step 2", "+ I7", ", F4-28", ": F4-11", "parts 3 and 4").
# Words that are no code stay: (IPv6 support), (MP4 export), (H100 tested), (STL v2), (MCPB v0.3), (3D), (inches or metres).
CODE_RE = re.compile(
    r"\s*\((?:[A-Z]\d{1,2}[a-z]?|(?=[A-Za-z0-9-]*\d)[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)(?:[ ,+:][^()]*)?\)\s*$"
)


@dataclass
class Entry:
    """One line of the notes."""

    title: str  # the PR title (or the subject of a direct commit), as written
    number: int | None = None  # None for a direct commit
    commit_types: list[str] = field(default_factory=list)  # types of the commits that the PR brought
    breaking: bool = False


def split_type(subject: str) -> tuple[str, bool, str]:
    """Return (type, bang, rest) of a Conventional Commit subject; the type is "" when there is none."""
    m = TYPE_RE.match(subject.strip())
    if not m:
        return NO_TYPE, False, subject.strip()
    return m.group("type").lower(), bool(m.group("bang")), m.group("rest").strip()


def clean_sentence(rest: str) -> str:
    """Drop the trailing `(requested by ...)` and item codes, any order; upper case first letter; no final period."""
    text = rest.strip()
    while True:
        new = SUFFIX_RE.sub("", text)
        new = CODE_RE.sub("", new)
        if new == text:
            break
        text = new
    text = text.strip().rstrip(".").rstrip()
    first = text.split(" ", 1)[0]
    if re.search(r"[_/.`]|[a-z0-9][A-Z]", first):  # a name such as list_runs or MCP-style Name: keep it as it is
        return text
    return text[:1].upper() + text[1:]


def entry_type(entry: Entry) -> str:
    """The highest type of the title and of the commits of the PR."""
    own, _, _ = split_type(entry.title)
    candidates = [own] + list(entry.commit_types)
    order = [t for t, _ in SECTIONS]
    known = [t for t in candidates if t in order]
    if known:
        return min(known, key=order.index)
    others = sorted(t for t in candidates if t and t not in order)
    return others[0] if others else NO_TYPE


def section_title(t: str) -> str:
    if t == NO_TYPE:
        return NO_TYPE_TITLE
    for key, title in SECTIONS:
        if key == t:
            return title
    return OTHER_TITLES.get(t, t.capitalize())


def section_order(types: set[str]) -> list[str]:
    fixed = [t for t, _ in SECTIONS if t in types]
    others = sorted(t for t in types if t and t not in {k for k, _ in SECTIONS})
    return fixed + others + ([NO_TYPE] if NO_TYPE in types else [])


def render_line(entry: Entry, repo: str) -> str:
    _, bang, rest = split_type(entry.title)
    sentence = clean_sentence(rest)
    lead = "**Breaking:** " if (entry.breaking or bang) else ""
    if entry.number is None:
        return f"- {lead}{sentence}."
    return f"- {lead}{sentence} ([#{entry.number}](https://github.com/{repo}/pull/{entry.number}))."


def render_entry(entries: list[Entry], version: str, date: str, prev_tag: str | None, repo: str = DEFAULT_REPO) -> str:
    """The entry text of one release. `entries` are in the order of the history, newest first."""
    groups: dict[str, list[Entry]] = {}
    for e in entries:
        groups.setdefault(entry_type(e), []).append(e)
    out = [f"## v{version} ({date})", "", LICENSE_LINE, ""]
    for t in section_order(set(groups)):
        out += [f"### {section_title(t)}", ""]
        out += [render_line(e, repo) for e in groups[t]]
        out.append("")
    if prev_tag:
        out += [
            "---",
            "",
            f"**Detailed Changes**: [{prev_tag}...v{version}](https://github.com/{repo}/compare/{prev_tag}...v{version})",
            "",
        ]
    return "\n".join(out)


# ---- the git reader ----

def git(*args: str, cwd: str | Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def last_tag(head: str, cwd: str | Path | None = None) -> str | None:
    try:
        return git("describe", "--tags", "--abbrev=0", "--match", "v*", head, cwd=cwd).strip() or None
    except subprocess.CalledProcessError:
        return None


def own_commits(merge: str, cwd: str | Path | None) -> list[tuple[str, str]]:
    """(subject, full message) of the commits that the PR brought, no merge commits."""
    sep = "\x1e"
    raw = git("log", "--no-merges", f"--format=%s\x1f%B{sep}", f"{merge}^1..{merge}^2", cwd=cwd)
    pairs = []
    for chunk in raw.split(sep):
        if chunk.strip():
            subject, _, body = chunk.partition("\x1f")
            pairs.append((subject.strip(), body))
    return pairs


def read_entries(prev_tag: str | None, head: str = "HEAD", cwd: str | Path | None = None) -> list[Entry]:
    rng = f"{prev_tag}..{head}" if prev_tag else head
    sep = "\x1e"
    raw = git("log", "--first-parent", f"--format=%H\x1f%P\x1f%s\x1f%B{sep}", rng, cwd=cwd)
    entries: list[Entry] = []
    for chunk in raw.split(sep):
        if not chunk.strip():
            continue
        sha, parents, subject, body = chunk.strip("\n").split("\x1f", 3)
        sha = sha.strip()
        merge = len(parents.split()) > 1
        m = MERGE_RE.match(subject)
        if m:
            title = next((ln.strip() for ln in body.splitlines()[1:] if ln.strip()), "")
            if not title:
                continue
            if merge:
                commits = own_commits(sha, cwd)
                types = [split_type(s)[0] for s, _ in commits]
                breaking = any(split_type(s)[1] or BREAKING_RE.search(b) for s, b in commits)
            else:  # a squash merge that kept the merge message: no second parent, the title gives the type (#442)
                types = []
                breaking = split_type(title)[1] or bool(BREAKING_RE.search(body))
            entries.append(Entry(title, int(m.group("n")), types, breaking))
        elif merge or RELEASE_RE.match(subject):
            continue
        else:
            s = SQUASH_RE.search(subject)
            breaking = bool(BREAKING_RE.search(body))
            if s:
                entries.append(Entry(SQUASH_RE.sub("", subject), int(s.group("n")), [], breaking))
            else:
                entries.append(Entry(subject, None, [], breaking))
    return entries


def insert_into_changelog(text: str, entry: str, version: str) -> str:
    """Put `entry` after the insertion line. An entry of the same version is replaced, so a second run changes nothing."""
    if text.count(INSERT_MARKER) != 1:
        raise SystemExit(f"the changelog must hold the line {INSERT_MARKER} exactly once")
    head, tail = text.split(INSERT_MARKER)
    rest = re.sub(rf"\A\s*## v{re.escape(version)} \(.*?(?=^## |\Z)", "", tail, count=1, flags=re.S | re.M)
    return head + INSERT_MARKER + "\n\n" + entry.rstrip("\n") + "\n\n" + rest.lstrip("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", default=os.environ.get("NEW_VERSION"), help="new version X.Y.Z (default: NEW_VERSION)")
    ap.add_argument("--prev-tag", help="tag of the last release (default: the last v* tag before HEAD)")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--write-changelog", metavar="FILE", help="insert the entry after the insertion line of FILE")
    ap.add_argument("--notes-file", metavar="FILE", help="also write the entry to FILE (the text of the GitHub release)")
    a = ap.parse_args(argv)
    if not a.version:
        ap.error("--version (or NEW_VERSION) is required")
    version = a.version.lstrip("v")
    prev = a.prev_tag or last_tag(a.head)
    entry = render_entry(read_entries(prev, a.head), version, a.date, prev, a.repo)
    if a.write_changelog:
        p = Path(a.write_changelog)
        p.write_text(insert_into_changelog(p.read_text(encoding="utf-8"), entry, version), encoding="utf-8")
    if a.notes_file:
        Path(a.notes_file).parent.mkdir(parents=True, exist_ok=True)
        Path(a.notes_file).write_text(entry, encoding="utf-8")
    if not (a.write_changelog or a.notes_file):
        sys.stdout.write(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
