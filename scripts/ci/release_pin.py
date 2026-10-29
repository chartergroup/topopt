"""Write the tag of the current version, and the marketplace name of that tag, into the pinned install forms of the install doc.

python-semantic-release runs this script first in its build step (`build_command` in `pyproject.toml`). At that point
it has written the new version into the three version files and has not committed yet. `docs/reference/install.md` is in its
`assets`, so the release commit carries the install doc with the new tag (see RELEASING.md).

The README leads with the moving ref `#latest` (`claude plugin marketplace add chartergroup/topopt#latest`) and names no tag, so
it needs no move. The script only checks it (`readme_problems`): its add command uses `#latest`, its install command is
`topopt@<name of the marketplace file of main>`, and no `chartergroup/topopt#vX.Y.Z` is in it.

Tag. The script reads the release tag that the install doc names now, from the install commands
`claude plugin marketplace add chartergroup/topopt#vX.Y.Z` and `... --marketplace chartergroup/topopt#vX.Y.Z` (the one-command
forms of #357: the prompt, the chat command and the terminal command), and writes the tag of the version in `pyproject.toml`
in every place of the install doc that names that old tag (the commands, the checkout line, the text about the tag).
Other version numbers (for example of Claude Code) stay, and so does the placeholder `#<new-tag>` of the update route.
A one-command form without a tag (`--marketplace chartergroup/topopt` alone) installs the newest `main`, and one with another ref
(`#main`, `#latest`, a branch) installs what that ref holds then, so the script refuses both. (The two-command form with `#latest`
is right, and the install doc has it: only the one-command option is refused.)

Marketplace name. Claude Code registers a marketplace under the `name` in the `.claude-plugin/marketplace.json` of the
tag that a user installs. So the pinned commands must use the name that THE PINNED TAG holds, not the name that `main`
holds. The script reads the name that the install doc uses now, from `claude plugin install topopt@<name>`, and writes the
name of the new tag into the four command forms that carry a marketplace name (see NAME_FORMS). A sentence that only
talks about the names (it has none of these forms, and no tag) stays as it is.
The name of a release is `name` in `.claude-plugin/marketplace.json`, except for the releases in
RELEASES_BEFORE_THE_RENAME (REL-8, #313): the tags `v0.1.0` and `v0.2.0` were made before the marketplace got the name `topopt`,
and their files name it `topopt-demo`. The script does not read git tags: a CI checkout has none.

It exits with code 1 when the install doc has no such install command, more than one tag in such commands, a one-command form
without a tag, or more than one marketplace name in its commands, or when the README breaks its check.

    python3 scripts/ci/release_pin.py [--check]

`--check` changes nothing and exits with code 1 when the install doc names another tag than `v<version>`, or another
marketplace name than the one of that tag, or when the README breaks its check.
"""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The file that holds the pinned install forms, and the one that leads with `#latest`.
INSTALL_DOC = "docs/reference/install.md"
README = "README.md"
ADD_COMMAND = re.compile(r"claude plugin marketplace add chartergroup/topopt#(v\d+\.\d+\.\d+)\b")
# The one-command forms: `claude plugin install topopt --marketplace ...` and `/plugin install topopt --marketplace ...`.
MARKETPLACE_OPTION = re.compile(r"--marketplace chartergroup/topopt#(v\d+\.\d+\.\d+)\b")
# The same option without a release tag: no "#", or a "#" with another ref (#main, #latest, a branch, v1.2.3-rc1, v1.2.3+x). A name character
# after the slug makes it another repo, so that does not count. A full stop that ends a sentence after the tag is no part of the ref.
UNPINNED_OPTION = re.compile(r"--marketplace chartergroup/topopt(?![\w-])(?!#v\d+\.\d+\.\d+(?![\w+/-]|\.\w))")
INSTALL_COMMAND = re.compile(r"claude plugin install topopt@([\w-]+)(?![\w-])")
# The text before and after a marketplace name in the four command forms of the install doc:
# `topopt@<name>`, `claude plugin marketplace remove <name>`, `extraKnownMarketplaces.<name>.` (a settings key) and the
# error text `Marketplace '<name>' not found`. A name has letters, digits, "_" and "-", so a dot after it ends the name.
NAME_FORMS = (
    ("topopt@", ""),
    ("claude plugin marketplace remove ", ""),
    ("extraKnownMarketplaces.", "."),
    ("Marketplace '", "' not found"),
)
# Marketplace name in the `.claude-plugin/marketplace.json` of a release that was made before the rename to `topopt`.
# A fixed record, because a CI checkout has no tags. Add no new release here: every release after the rename names `topopt`.
RELEASES_BEFORE_THE_RENAME = {"0.1.0": "topopt-demo", "0.2.0": "topopt-demo"}


def version(repo: Path = REPO) -> str:
    return tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def marketplace_name(repo: Path = REPO) -> str:
    """The `name` in `.claude-plugin/marketplace.json` of the working tree."""
    return json.loads((repo / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))["name"]


def marketplace_of_release(release_version: str, repo: Path = REPO) -> str:
    """The marketplace name that the tag `v<release_version>` holds."""
    return RELEASES_BEFORE_THE_RENAME.get(release_version) or marketplace_name(repo)


def _name_pattern(before: str, after: str, name: str) -> re.Pattern[str]:
    tail = re.escape(after) if after else r"(?![\w-])"
    return re.compile(r"(?<![\w-])" + re.escape(before) + name + tail)


def readme_names(readme: str) -> set[str]:
    """Every marketplace name that the command forms of the README carry."""
    found: set[str] = set()
    for before, after in NAME_FORMS:
        found.update(_name_pattern(before, after, r"([\w-]+)").findall(readme))
    return found


def readme_facts(readme: str) -> tuple[str, str]:
    """The tag of the add command and the marketplace name of the install command, as the install doc holds them now."""
    if UNPINNED_OPTION.search(readme):
        raise ValueError(f"{INSTALL_DOC} has a one-command install form without the release tag (no tag, or another ref): "
                         "write --marketplace chartergroup/topopt#vX.Y.Z")
    tags = set(ADD_COMMAND.findall(readme)) | set(MARKETPLACE_OPTION.findall(readme))
    if len(tags) != 1:
        raise ValueError(f"{INSTALL_DOC} must name one tag in its install commands; found {sorted(tags) or 'none'}")
    installs = set(INSTALL_COMMAND.findall(readme))
    if len(installs) != 1:
        raise ValueError(f"{INSTALL_DOC} must name one marketplace in its install command `claude plugin install topopt@<name>`; "
                         f"found {sorted(installs) or 'none'}")
    names = readme_names(readme)
    if names != installs:
        raise ValueError(f"{INSTALL_DOC} must use one marketplace name in all its commands; found {sorted(names)}")
    return tags.pop(), installs.pop()


def pin(readme: str, new_version: str, marketplace: str | None = None) -> str:
    """Return the install doc with the tag of its install command replaced by `v<new_version>` everywhere.

    With `marketplace`, the commands of the install doc that carry a marketplace name get that name too. Without it, the names stay.
    """
    old_tag, old_name = readme_facts(readme)
    text = re.sub(rf"(?<![\w.]){re.escape(old_tag)}(?![\w.])", f"v{new_version}", readme)
    if marketplace is not None and marketplace != old_name:
        for before, after in NAME_FORMS:
            text = _name_pattern(before, after, re.escape(old_name)).sub(lambda _m: before + marketplace + after, text)
    return text


def problems(readme: str, release_version: str, marketplace: str) -> list[str]:
    """What is wrong in the install doc for the release `release_version`, whose tag holds the marketplace name `marketplace`."""
    tag, name = readme_facts(readme)
    found = []
    if tag != f"v{release_version}":
        found.append(f"{INSTALL_DOC} names the tag {tag}; the version in pyproject.toml needs the tag v{release_version}")
    if name != marketplace:
        found.append(f"{INSTALL_DOC} uses the marketplace name {name}; the tag v{release_version} holds the marketplace name {marketplace}")
    return found


README_LATEST_ADD = "claude plugin marketplace add chartergroup/topopt#latest"
README_TAGGED = re.compile(r"chartergroup/topopt#v\d")


def readme_problems(readme: str, main_marketplace: str) -> list[str]:
    """What is wrong in the README: it must lead with `#latest` and `topopt@<name of main>`, and name no release tag."""
    found = []
    if README_LATEST_ADD not in readme:
        found.append(f"{README} needs the command `{README_LATEST_ADD}`")
    if README_TAGGED.search(readme):
        found.append(f"{README} names a release tag (`chartergroup/topopt#vX.Y.Z`); the pinned forms belong in {INSTALL_DOC}")
    names = readme_names(readme)      # every `topopt@<name>`: the terminal form and the session form (`/plugin install topopt@<name>`)
    if names != {main_marketplace}:
        found.append(f"{README} uses the marketplace name(s) topopt@{sorted(names) or 'none'}; it needs topopt@{main_marketplace}")
    return found


def main(argv: list[str], repo: Path = REPO) -> int:
    path = repo / INSTALL_DOC
    old = path.read_text(encoding="utf-8")
    release = version(repo)
    marketplace = marketplace_of_release(release, repo)
    readme_found = readme_problems((repo / README).read_text(encoding="utf-8"), marketplace_name(repo))
    for line in readme_found:
        print(f"release_pin: {line}", file=sys.stderr)
    if readme_found and "--check" not in argv:
        return 1      # the README is wrong: write nothing
    try:
        if "--check" in argv:
            found = problems(old, release, marketplace)
            for line in found:
                print(f"release_pin: {line}", file=sys.stderr)
            return 1 if found or readme_found else 0
        new = pin(old, release, marketplace)
    except ValueError as exc:
        print(f"release_pin: {exc}", file=sys.stderr)
        return 1
    if new != old:
        path.write_text(new, encoding="utf-8")
    print(f"release_pin: {INSTALL_DOC} names v{release} and the marketplace {marketplace}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
