"""Short release notes, one line per pull request (REL-10, #334), and the moving `latest` branch (REL-10, #332).

The line logic is in `scripts/ci/release_notes.py`. These tests render fixture histories, run the git reader on a small repo
made in `tmp_path`, and read the workflow and `pyproject.toml` for the wiring (the release workflow cannot run in a test).
"""
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def notes_module():
    sys.path.insert(0, str(REPO / "scripts" / "ci"))
    try:
        import release_notes
    finally:
        sys.path.pop(0)
    return release_notes


rn = notes_module()
E = rn.Entry


def render(entries, prev="v0.1.0"):
    return rn.render_entry(entries, "0.2.0", "2026-10-26", prev, "chartergroup/topopt")


# ---- the cleaning ----

@pytest.mark.parametrize(
    "title, sentence",
    [
        ("fix(run): a run that stops early gives the earlier run back (F4-47)", "A run that stops early gives the earlier run back"),
        ("fix(stress): texts give the public-part error (F1-8 step 2)", "Texts give the public-part error"),
        ("perf(buckling): warm start of the eigen solve (F4-31 step 5a)", "Warm start of the eigen solve"),
        ("feat(mcp): the first reply says how to pick a solver (P9-4a)", "The first reply says how to pick a solver"),
        ("docs(mcp): move the key list to the card (K1-FU-C)", "Move the key list to the card"),
        ("feat(skills): look at the render (LAUNCH-3)", "Look at the render"),
        ("chore(deps): bump the pin (REL-7)", "Bump the pin"),
        ("fix(check): the unit fact reads one STL (TC-3 review)", "The unit fact reads one STL"),
        ("feat(regions): a ring round each screw hole (I2 parts 3 and 4)", "A ring round each screw hole"),
        ("feat(mcp): two tools (I6 + I7)", "Two tools"),
        ("feat(harness): the first eval (H1 PR 4)", "The first eval"),
        ("fix(sdf): three fixes (F4-34, F4-28, F4-29)", "Three fixes"),
        ("feat(design): a print pass (P2d-D3 + D3h)", "A print pass"),
        ("docs(ux): a note (U21)", "A note"),
        ("feat(cli): a stop request (P4b-4: F4-11)", "A stop request"),
        ("feat(verify): a stress check (B1: B-098, part of B-060)", "A stress check"),
        ("feat(materials): a catalog (PM-2) (requested by Some One)", "A catalog"),
        ("fix(x): a thing (Q1-fix)", "A thing"),
        ("fix(x): a thing (B1)", "A thing"),
        ("fix(x): a thing (IPv6 support)", "A thing (IPv6 support)"),
        ("fix(x): a thing (MP4 export)", "A thing (MP4 export)"),
        ("fix(x): a thing (H100 tested)", "A thing (H100 tested)"),
        ("fix(x): a thing (STL v2)", "A thing (STL v2)"),
        ("fix(x): a thing (MCPB v0.3)", "A thing (MCPB v0.3)"),
        ("fix(x): a thing (3D)", "A thing (3D)"),
        ("fix(x): a thing (inches or metres)", "A thing (inches or metres)"),
        ("fix(x): a thing (Mixed-case word)", "A thing (Mixed-case word)"),
        ("fix(run): keep the result (F4-43) (requested by Someone Else)", "Keep the result"),
        ("ci(tests): fast and full tiers (requested by Some One)", "Fast and full tiers"),
        ("test: no real browser tab from a child run", "No real browser tab from a child run"),
        ("docs(readme): a table of problems.", "A table of problems"),
    ],
)
def test_the_sentence_has_no_type_no_item_code_and_no_name(title, sentence):
    _, _, rest = rn.split_type(title)
    assert rn.clean_sentence(rest) == sentence


def test_a_code_name_at_the_start_keeps_its_case():
    for rest in ("list_runs marks the facts", "`topopt check` warns", "pyproject.toml names the tag", "uvx runs"):
        got = rn.clean_sentence(rest)
        assert got == rest if rest != "uvx runs" else got == "Uvx runs"


def test_words_in_parentheses_that_are_not_codes_stay():
    _, _, rest = rn.split_type("docs(readme): a check for inches (or metres) and the (see below) note")
    assert rn.clean_sentence(rest) == "A check for inches (or metres) and the (see below) note"
    _, _, rest = rn.split_type("feat(check): a fact (inches or metres)")
    assert rn.clean_sentence(rest) == "A fact (inches or metres)"
    for kept in ("A mesh (STL v2)", "A bundle (MCPB v0.3)", "A view (3D)"):
        assert rn.clean_sentence(kept) == kept


def test_the_line_form_with_and_without_a_link():
    assert rn.render_line(E("fix(run): keep it (F4-43)", 243), "o/r") == "- Keep it ([#243](https://github.com/o/r/pull/243))."
    assert rn.render_line(E("fix(run): keep it", None), "o/r") == "- Keep it."


def test_a_breaking_change_is_marked():
    assert rn.render_line(E("feat(api)!: drop the old key", 5), "o/r").startswith("- **Breaking:** Drop the old key (")
    assert rn.render_line(E("feat(api): drop the old key", 5, [], True), "o/r").startswith("- **Breaking:** ")


# ---- the render ----

def test_section_order_header_license_and_compare_link():
    entries = [
        E("test: a", 1),
        E("ci(x): b (REL-1)", 2),
        E("chore(x): c", 3),
        E("docs(x): d", 4),
        E("perf(x): e", 5),
        E("fix(x): f", 6),
        E("feat(x): g", 7),
        E("refactor(x): h", 8),
        E("build(x): i", 9),
        E("style(x): j", 10),
        E("something without a type", 11),
    ]
    text = render(entries)
    heads = re.findall(r"^### (.*)$", text, re.M)
    assert heads == [
        "Features", "Bug Fixes", "Performance Improvements", "Documentation", "Chores",
        "Continuous Integration", "Testing", "Build System", "Refactoring", "Code Style", "Other Changes",
    ]
    assert text.startswith("## v0.2.0 (2026-10-26)\n\n_This release is published under the MIT License._\n\n### Features\n")
    assert text.endswith(
        "---\n\n**Detailed Changes**: [v0.1.0...v0.2.0](https://github.com/chartergroup/topopt/compare/v0.1.0...v0.2.0)\n"
    )
    assert "Co-Authored-By" not in text


def test_a_chore_pr_that_holds_a_feat_commit_is_listed_under_features():
    text = render([E("chore(x): a thing", 3, ["chore", "docs", "feat"]), E("docs(x): other", 4, ["docs"])])
    assert text.index("### Features") < text.index("A thing") < text.index("### Documentation")
    assert "### Chores" not in text


def test_the_first_release_has_no_compare_link():
    text = rn.render_entry([E("feat: a", 1)], "0.1.0", "2026-01-01", None)
    assert "Detailed Changes" not in text and "- A ([#1]" in text


def test_an_empty_release_has_a_header_and_no_section():
    text = render([])
    assert "###" not in text
    assert text.startswith("## v0.2.0 (2026-10-26)\n\n_This release")


# ---- the git reader ----

def run(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout


def commit(cwd, msg, name="f"):
    p = Path(cwd) / f"{name}.txt"
    p.write_text(p.read_text() + "x\n" if p.exists() else "x\n")
    run(cwd, "add", "-A")
    run(cwd, "commit", "-m", msg)


def pr(cwd, n, branch, title, *commits):
    """Make a branch with the commits and merge it with --no-ff in the GitHub message form."""
    run(cwd, "checkout", "-q", "-b", branch, "main")
    for i, c in enumerate(commits):
        commit(cwd, c, f"{branch.replace('/', '_')}{i}")
    run(cwd, "checkout", "-q", "main")
    run(cwd, "merge", "--no-ff", "-m", f"Merge pull request #{n} from org/{branch}", "-m", title, branch)


@pytest.fixture
def history(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is missing")
    cwd = tmp_path / "repo"
    cwd.mkdir()
    run(cwd, "init", "-q", "-b", "main")
    commit(cwd, "chore: start")
    run(cwd, "tag", "v0.1.0")
    pr(cwd, 11, "a/one", "fix(run): the first fix (F4-1) (requested by Some One)", "fix(run): the first fix", "docs: a note")
    pr(cwd, 12, "a/two", "chore(x): a chore PR", "chore(x): a chore", "feat(x): a hidden feature")
    pr(cwd, 13, "a/three", "test: no scope here", "test: no scope here")
    pr(cwd, 14, "a/four", "feat(api): drop the old key", "feat(api)!: drop the old key\n\nBREAKING CHANGE: the key is gone")
    commit(cwd, "docs(readme): a squash commit (#15)")
    commit(cwd, "fix(x): a direct commit")
    commit(cwd, "chore(release): 0.2.0\n\n[skip ci]")
    return cwd


def test_the_git_reader_on_a_small_history(history):
    entries = rn.read_entries("v0.1.0", "HEAD", cwd=history)
    by_title = {e.title: e for e in entries}
    assert [e.number for e in entries] == [None, 15, 14, 13, 12, 11]  # newest first; the release commit is skipped
    assert by_title["chore(x): a chore PR"].commit_types == ["feat", "chore"] or sorted(
        by_title["chore(x): a chore PR"].commit_types
    ) == ["chore", "feat"]
    assert by_title["feat(api): drop the old key"].breaking is True
    assert by_title["docs(readme): a squash commit"].number == 15
    assert by_title["fix(x): a direct commit"].number is None
    text = rn.render_entry(entries, "0.2.0", "2026-10-26", "v0.1.0", "o/r")
    assert text.index("### Features") < text.index("A chore PR")  # the hidden feat commit moves the PR
    assert "- **Breaking:** Drop the old key ([#14](https://github.com/o/r/pull/14))." in text
    assert "- The first fix ([#11](https://github.com/o/r/pull/11))." in text  # code and name are gone
    assert "- A direct commit.\n" in text and "- A squash commit ([#15]" in text
    assert "release" not in text.split("Detailed")[0].lower().replace("this release is published", "")
    assert "Some One" not in text and "BREAKING CHANGE" not in text


def squash_with_merge_message(cwd, n, branch, title, more=""):
    """A squash merge that kept the merge message of GitHub, as 08f8922 of #374 (#442): one parent, the subject
    `Merge pull request #N from ...`, the PR title as the first body line."""
    commit(cwd, f"Merge pull request #{n} from org/{branch}\n\n{title}" + (f"\n\n{more}" if more else ""), branch.replace("/", "_"))


def test_a_squash_merge_that_kept_the_merge_message_is_one_pull_request(history):
    squash_with_merge_message(history, 16, "a/five", "feat(examples): add a 5-inch drone-arm example (REL-6)")
    squash_with_merge_message(history, 17, "a/six", "refactor(core): rename the key",
                              "* refactor(core): rename the key\n\nBREAKING CHANGE: the old key is gone")
    squash_with_merge_message(history, 18, "a/seven", "fix(x)!: drop a flag")
    assert len(run(history, "log", "-1", "--format=%P", "HEAD~2").split()) == 1      # one parent, as 08f8922
    entries = rn.read_entries("v0.1.0", "HEAD", cwd=history)
    assert [e.number for e in entries] == [18, 17, 16, None, 15, 14, 13, 12, 11]
    seven, six, five = entries[:3]
    assert (five.title, five.commit_types, five.breaking) == ("feat(examples): add a 5-inch drone-arm example (REL-6)", [], False)
    assert (six.title, six.breaking, seven.breaking) == ("refactor(core): rename the key", True, True)
    text = rn.render_entry(entries, "0.2.0", "2026-10-26", "v0.1.0", "o/r")
    assert "- Add a 5-inch drone-arm example ([#16](https://github.com/o/r/pull/16))." in text
    assert text.index("### Features") < text.index("drone-arm") < text.index("### Bug Fixes")
    assert "- **Breaking:** Rename the key ([#17](https://github.com/o/r/pull/17))." in text and "### Refactoring" in text
    assert "- **Breaking:** Drop a flag ([#18](https://github.com/o/r/pull/18))." in text
    assert "Merge pull request" not in text


def test_the_last_tag_is_found_and_the_cli_prints_the_entry(history):
    assert rn.last_tag("HEAD", cwd=history) == "v0.1.0"
    out = subprocess.run(
        [sys.executable, "-P", str(REPO / "scripts" / "ci" / "release_notes.py"), "--version", "0.2.0", "--date", "2026-10-26"],
        cwd=history, check=True, capture_output=True, text=True,
    ).stdout
    assert out.startswith("## v0.2.0 (2026-10-26)") and "compare/v0.1.0...v0.2.0" in out


def test_the_changelog_insert_keeps_the_marker_once_and_is_idempotent():
    base = "# Changelog\n\nhead\n\n<!-- version list -->\n\n## v0.1.0 (old)\n\n- old\n"
    first = rn.insert_into_changelog(base, "## v0.2.0 (d)\n\n- new\n", "0.2.0")
    again = rn.insert_into_changelog(first, "## v0.2.0 (d)\n\n- new\n", "0.2.0")
    assert first == again
    assert first.count("<!-- version list -->") == 1
    assert first.index("## v0.2.0") < first.index("## v0.1.0") and "- old" in first
    with pytest.raises(SystemExit):
        rn.insert_into_changelog("no marker", "x", "0.2.0")


# ---- the CHANGELOG of this repo ----

def test_the_changelog_entry_of_v0_2_0_is_short_and_has_one_line_per_merge():
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(r"^## v0\.2\.0 .*?(?=^## )", text, re.S | re.M)
    if not m and re.search(r"^## Earlier versions$", text, re.M):
        pytest.skip("the public CHANGELOG folds the entries of the releases before the first public one")
    assert m, "CHANGELOG.md has no v0.2.0 entry"
    entry = m.group(0)
    assert "Co-Authored-By" not in entry and "/commit/" not in entry
    lines = re.findall(r"^- .*\(\[#(\d+)\]\(https://github\.com/chartergroup/topopt/pull/\d+\)\)\.$", entry, re.M)
    assert len(lines) == len(set(lines)) == entry.count("\n- ")
    assert text.count("<!-- version list -->") == 1
    tags = subprocess.run(["git", "tag", "-l", "v0.1.0", "v0.2.0"], cwd=REPO, capture_output=True, text=True).stdout.split()
    if tags != ["v0.1.0", "v0.2.0"]:
        pytest.skip("CI checkouts have no tags")
    merges = subprocess.run(
        ["git", "log", "--first-parent", "--merges", "--format=%s", "v0.1.0..v0.2.0"], cwd=REPO, capture_output=True, text=True
    ).stdout.splitlines()
    numbers = {re.match(r"Merge pull request #(\d+)", s).group(1) for s in merges}
    assert set(lines) == numbers


# ---- the wiring in the release workflow and in pyproject.toml ----

WORKFLOW = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
CONFIG = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["semantic_release"]


def steps():
    """The steps of the job `release`, as (name, text)."""
    body = WORKFLOW.split("    steps:\n", 1)[1]
    parts = re.split(r"^      - name: ", body, flags=re.M)[1:]
    return [(p.split("\n", 1)[0].strip(), p) for p in parts]


def step(name):
    for n, text in steps():
        if n == name:
            return text
    raise AssertionError(f"no step {name!r}; steps: {[n for n, _ in steps()]}")


def test_no_release_branch_matches_latest():
    for name, cfg in CONFIG["branches"].items():
        assert re.search(cfg["match"], "latest") is None, f"branch group {name!r} matches 'latest'"
        assert re.search(cfg["match"], "main") is not None
    assert "latest" not in CONFIG["branches"]


def test_the_notes_come_from_the_script_in_the_build_step():
    assert "scripts/ci/release_notes.py" in CONFIG["build_command"]
    assert "--write-changelog CHANGELOG.md" in CONFIG["build_command"]
    assert "CHANGELOG.md" in CONFIG["assets"]
    # The tool must not write its own long entry, and the GitHub release gets the short text.
    release = step("Release")
    assert "--no-changelog" in release
    assert "gh release edit" in WORKFLOW and "--notes-file" in WORKFLOW


def test_the_dry_run_prints_the_short_notes():
    s = step("Show the short notes of the next release")
    assert "release_notes.py" in s and "inputs.dry_run" not in s  # runs in both modes


def test_the_latest_step_runs_after_the_release_step_and_only_after_a_release():
    names = [n for n, _ in steps()]
    assert names.index("Move the branch latest") > names.index("Release")
    s = step("Move the branch latest")
    assert "inputs.dry_run == 'false'" in s
    assert "steps.release.outputs.released == 'true'" in s
    assert "refs/heads/latest" in s


def push_is_forced(line: str) -> bool:
    """True when a git push line forces: a flag, or a refspec argument that starts with `+`."""
    if re.search(r"(^|\s)(--force\b|--force-with-lease\b|-f\b|--mirror\b)", line):
        return True
    return re.search(r"[\s\"']\+", line) is not None


@pytest.mark.parametrize(
    "line, forced",
    [
        ('git push origin "${sha}:refs/heads/latest"', False),
        ('git push origin "+${sha}:refs/heads/latest"', True),
        ("git push origin +${sha}:refs/heads/latest", True),
        ("git push origin '+abc:refs/heads/latest'", True),
        ('git push --force origin "${sha}:refs/heads/latest"', True),
        ('git push -f origin "${sha}:refs/heads/latest"', True),
        ('git push --force-with-lease origin "${sha}:refs/heads/latest"', True),
    ],
)
def test_the_force_checker_sees_every_form(line, forced):
    assert push_is_forced(line) is forced


def test_the_latest_push_is_a_plain_fast_forward():
    s = step("Move the branch latest")
    push_lines = [ln for ln in s.splitlines() if "git push" in ln]
    assert push_lines, "no git push in the latest step"
    for ln in push_lines:
        assert not push_is_forced(ln), ln
    assert "^{commit}" in s  # the commit of the tag, not the tag object
    assert "latest" not in step("Release").replace("--no-changelog", "")  # the tool does not touch it


def test_a_rerun_on_a_tagged_head_repairs_the_notes_and_the_release():
    s = step("Release")
    assert "git tag --points-at HEAD --list 'v*' | sort -V | tail -n 1" in s
    assert "released=true" in s and "released=false" in s
    assert "[ ! -f dist/release-notes.md ]" in s
    assert '--head "${tag}^"' in s and "--notes-file dist/release-notes.md" in s
    assert 'gh release view "$tag"' in s and 'gh release create "$tag" --verify-tag' in s
    assert s.index("gh release create") < s.index("gh release edit")


def test_the_dry_run_calls_the_tool_once_and_says_where_latest_stays_or_moves():
    assert WORKFLOW.count("semantic-release version --print") == 1
    assert "steps.next.outputs.version" in step("Show the short notes of the next release")
    s = step("Show where latest would move")
    assert "steps.next.outputs.version" in s and "latest stays where it is" in s and "not created yet" in s


def test_releasing_md_and_the_agent_rules_describe_the_notes_and_latest():
    releasing = (REPO / "RELEASING.md").read_text(encoding="utf-8")
    for needle in ("## The release notes", "## The branch latest", "## If the release run fails after the tag", "Workflows: write", "downloads `install.md` and `CHANGELOG.md`", "release_notes.py", "refs/heads/latest", "no force", "^main$", "--no-changelog"):
        assert needle in releasing, needle
    for name in ("CLAUDE.md", "CONTRIBUTING.md"):
        assert "RELEASING.md" in (REPO / name).read_text(encoding="utf-8") and "item code" in (REPO / name).read_text(encoding="utf-8")
