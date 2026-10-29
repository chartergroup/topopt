"""The texts that a user or an agent reads hold no number of an issue or a pull request.

After the public export these numbers point to a repository where they do not exist, or name other items. The files that this test reads are
the user and contributor docs and the two launcher snapshots (the lists of the tools and their descriptions). CHANGELOG.md keeps its numbers.
Code comments, CONTRACTS.md and the tests are not read here.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    "README.md", "CONTRIBUTING.md", "RELEASING.md", "docs/clients.md", "docs/install-test-checklist.md", "topopt_mcp/README.md",
    "recordings/README.md", "evals/README.md",
    "topopt_mcp/launcher/legacy.tools.json", "topopt_mcp/launcher/modern.tools.json",
]
TOOLING = [Path("scripts") / "release" / name for name in ("EXPORT" + ".md", "PRIVACY" + ".md")]    # in the private repository only
FOLDERS = ["docs/reference", "examples", "skills", ".agents/skills", ".claude/skills/part-to-topopt", "agents", "topopt_mcp/roles", "docker"]
SUFFIXES = {".md", ".json", ".yaml", ".yml", ".txt", ".sh", ".toml"}

NUMBER = re.compile(r"(?<![\w/&#-])#[1-9][0-9]{1,3}\b")
WORDS = re.compile(r"\b(?:issue|PR|pull request)s? #?[1-9][0-9]{1,3}\b", re.IGNORECASE)

# (file, text of the hit): a hit that is not a reference. Each entry has its reason.
ALLOWED = {
    # ("path", "#123"): "reason",
}


def _files():
    found = [ROOT / f for f in [*FILES, *TOOLING] if (ROOT / f).is_file()]
    for folder in FOLDERS:
        base = ROOT / folder
        if base.is_dir():
            found += [p for p in sorted(base.rglob("*")) if p.is_file() and p.suffix in SUFFIXES]
    return found


def _hits(text):
    return [m.group() for pattern in (NUMBER, WORDS) for m in pattern.finditer(text)]


def test_the_patterns_find_the_forms_of_a_reference():
    for sample in ("see #370 here", "(#12)", "issue 318", "the PR #27", "pull request 5000", "Issues #99"):
        assert _hits(sample), sample
    for sample in ("colour #ff0000", "step #1", "[x](https://github.com/o/r/pull/123)", "a#123", "&#123;"):
        assert not _hits(sample), sample


def test_the_user_texts_hold_no_issue_or_pull_request_number():
    found = []
    for path in _files():
        rel = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for hit in _hits(line):
                if (rel, hit) not in ALLOWED:
                    found.append(f"{rel}:{number}: {hit}")
    assert not found, "numbers of issues or pull requests in user texts:\n" + "\n".join(found)
