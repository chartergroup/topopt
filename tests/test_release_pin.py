"""Guard for the release rule (LAUNCH-1, #226) and for the name of the repo and of the marketplace (REL-8, #313).

A new user installs a release that was checked, not the code that is on `main` at that moment.
So the pinned install command of `docs/reference/install.md` names the tag `v<version>`, and the version is the same in the three files
that hold it. Claude Code compares only the version of `plugin.json`, so a release with an old version
does not reach the users who have the plugin (see RELEASING.md).

Claude Code registers a marketplace under the name in the `marketplace.json` of the tag that a user installs. The tags
`v0.1.0` and `v0.2.0` were made before the rename to `topopt`, and their files name the marketplace `topopt-demo`. So the README commands
name the marketplace of the PINNED TAG, and `scripts/ci/release_pin.py` writes the name together with the tag.
Since #385 the README leads with `#latest` and `topopt@topopt`, and the pinned forms are in the install doc (the constant DOC).
These tests read the files only (the tests of the pin script call its functions, or run it with `--check`).
"""
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NEW_SLUG = "chartergroup/topopt"
RENAMED_SLUG = "chartergroup/topopt-demo"  # the repo before the rename of 2026-10


def other_owners(text: str) -> set[str]:
    """The owners that a text names for this repo (`owner/topopt`, `owner/topopt-demo`, a github.com URL, `io.github.<owner>.topopt`), except the organization."""
    found = set(re.findall(r"github\.com/([\w.-]+)/topopt", text)) | set(re.findall(r"io\.github\.([\w-]+)\.topopt", text))
    found |= set(re.findall(r"(?:marketplace add |--marketplace )([\w-]+)/topopt", text)) | set(re.findall(r"(?<![\w.$-])([\w-]+)/topopt-demo", text))
    return found - {NEW_SLUG.split("/")[0]}


def test_other_owners_finds_each_form_of_a_url():
    for line in ("https://github.com/someone/topopt-demo", "https://raw.githubusercontent.com/someone/topopt-demo/main/scripts/x.sh",
                 "someone/topopt-demo/blob/main/README.md", "io.github.someone.topopt", "--marketplace someone/topopt"):
        assert other_owners(line) == {"someone"}, line
    assert not other_owners("https://github.com/chartergroup/topopt-demo/blob/main/README.md") and not other_owners("the folder ../topopt-demo")


NEW_URL = f"https://github.com/{NEW_SLUG}"
FIRST_RELEASE_MARKETPLACE = "topopt-demo"  # the name in the marketplace.json of the tags v0.1.0 and v0.2.0

DOC = "docs/reference/install.md"  # holds the pinned install forms (#385); the README leads with `#latest`
VERSION_FILES = ("pyproject.toml", ".claude-plugin/plugin.json", "packaging/mcpb/manifest.json")


def read(name: str) -> str:
    assert (REPO / name).is_file(), f"{name} is missing"
    return (REPO / name).read_text(encoding="utf-8")


def versions() -> dict[str, str]:
    return {
        "pyproject.toml": tomllib.loads(read("pyproject.toml"))["project"]["version"],
        ".claude-plugin/plugin.json": json.loads(read(".claude-plugin/plugin.json"))["version"],
        "packaging/mcpb/manifest.json": json.loads(read("packaging/mcpb/manifest.json"))["version"],
    }


def version() -> str:
    return versions()["pyproject.toml"]


def pin_module():
    sys.path.insert(0, str(REPO / "scripts" / "ci"))
    try:
        import release_pin
    finally:
        sys.path.pop(0)
    return release_pin


def readme_at_the_first_release() -> str:
    """The README as it was at the tag v0.1.0: the tag v0.1.0 and the marketplace of that tag, whatever the README names now."""
    return pin_module().pin(read(DOC), "0.1.0", FIRST_RELEASE_MARKETPLACE)


def test_the_three_version_files_hold_the_same_version():
    found = versions()
    assert set(found) == set(VERSION_FILES)
    assert len(set(found.values())) == 1, f"the version differs between the files: {found}"
    assert re.fullmatch(r"\d+\.\d+\.\d+", version()), f"the version {version()!r} is not X.Y.Z"


def test_the_readme_install_command_names_the_tag_of_the_version():
    readme = read(DOC)
    command = f"claude plugin marketplace add {NEW_SLUG}#v{version()}"
    assert readme.count(command) == 2, f"the install doc needs the command `{command}` in the prompt and in the two-command form"
    # Every other add command is the `#latest` form, the update route with a placeholder for the new tag, or the checkout route.
    # The boundary matters: `chartergroup/topopt-demo#v0.1.0` must not pass as `chartergroup/topopt`.
    adds = re.findall(r"claude plugin marketplace add (\S+)", readme)
    assert sorted(adds) == sorted([f"{NEW_SLUG}#latest"] + [f"{NEW_SLUG}#v{version()}"] * 2 + [f"{NEW_SLUG}#<new-tag>", f"{NEW_SLUG}#<new-tag>", "./topopt"]), adds
    # The checkout route names the same tag.
    assert f"git clone {NEW_URL}\n" in readme
    assert f"git -C topopt checkout v{version()}\n" in readme


def install_prompt(readme: str) -> str:
    """The prompt of the Quick start that a user pastes into Claude Code (#357): the first `text` block."""
    start = readme.index("```text\n") + len("```text\n")
    return readme[start:readme.index("```", start)]


def test_the_readme_one_command_forms_name_the_tag_of_the_version():
    """#357: the prompt, the chat command and the terminal command each carry the release tag."""
    readme = read(DOC)
    one = f"claude plugin install topopt --marketplace {NEW_SLUG}#v{version()}"
    assert f"`/plugin install topopt --marketplace {NEW_SLUG}#v{version()}`" in readme, "the chat form"
    assert f"`{one}`" in readme, "the terminal form"
    sources = re.findall(r"--marketplace ([\w.-]+/[\w.#<>-]+)", readme)       # a source has a slash; the prose word "is" does not
    assert len(sources) == 4 and set(sources) == {f"{NEW_SLUG}#v{version()}"}, sources       # the prompt has two


def test_the_install_prompt_names_the_repo_the_exact_commands_and_the_reload():
    """The prompt gives Claude the exact source and commands, so that Claude composes and searches nothing (#357)."""
    readme = read(DOC)
    prompt = install_prompt(readme)
    name = pin_module().readme_facts(readme)[1]
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    assert f"claude plugin install topopt --marketplace {NEW_SLUG}#v{version()}" in lines
    # The fallback for a Claude Code without --marketplace: the two commands, with the marketplace name of the pinned tag.
    assert f"claude plugin marketplace add {NEW_SLUG}#v{version()}" in lines and f"claude plugin install topopt@{name}" in lines
    # Without the claude command (for example a session with no CLI on PATH), Claude stops and gives the user the chat form. It comes
    # right after the install commands, before the steps that need claude too.
    chat = lines.index(f"claude plugin install topopt@{name}") + 1
    assert lines[chat].startswith("If there is no claude command, stop"), lines[chat]
    assert lines[chat].endswith(f"in the chat: /plugin install topopt --marketplace {NEW_SLUG}#v{version()}"), lines[chat]
    assert f"GitHub repo {NEW_SLUG}." in prompt and "Use no other source." in prompt and "/reload-plugins" in prompt


def test_the_install_prompt_sets_up_the_environment_before_the_reload():
    """#357: after the install and before /reload-plugins, Claude runs plugin-sync of the plugin folder and waits for "ready",
    so the first start of the server connects (no failed start, no 15-minute skip). An unknown error stops Claude."""
    lines = [line.strip() for line in install_prompt(read(DOC)).splitlines() if line.strip()]
    install = lines.index(f"claude plugin install topopt --marketplace {NEW_SLUG}#v{version()}")
    listing = next(i for i, line in enumerate(lines) if "claude plugin list --json" in line)
    sync = lines.index('sh "FOLDER/scripts/topopt-env.sh" plugin-sync')
    assert install < listing < sync, lines
    assert "readFromFolder" in lines[listing] and "installPath" in lines[listing] and "FOLDER" in lines[listing]
    assert '"topopt: ready"' in lines[sync + 1] and "run it again" in lines[sync + 1]
    assert lines[-1].startswith("If another error comes, stop and show it to me.") and lines[-1].endswith("/reload-plugins.")
    launcher = (REPO / "scripts" / "topopt-env.sh").read_text()
    assert "\n  plugin-sync)\n" in launcher and 'say "ready (' in launcher   # the command and the word that the prompt waits for

def test_the_pin_moves_the_one_command_forms_and_the_prompt():
    release_pin = pin_module()
    readme = read(DOC)
    moved = release_pin.pin(readme, "0.3.0", "topopt")
    assert moved.count(f"--marketplace {NEW_SLUG}#v0.3.0") == readme.count(f"--marketplace {NEW_SLUG}#v{version()}") == 4
    prompt = install_prompt(moved)
    assert f"claude plugin marketplace add {NEW_SLUG}#v0.3.0" in prompt and "claude plugin install topopt@topopt\n" in prompt


@pytest.mark.parametrize("readme, message", [
    ("claude plugin marketplace add chartergroup/topopt#v0.2.0\nclaude plugin install topopt@a\n"
     "claude plugin install topopt --marketplace chartergroup/topopt#v0.1.0\n", "one tag"),
    ("claude plugin marketplace add chartergroup/topopt#v0.2.0\nclaude plugin install topopt@a\n"
     "/plugin install topopt --marketplace chartergroup/topopt\n", "without the release tag"),
    ("claude plugin marketplace add chartergroup/topopt#v0.2.0\nclaude plugin install topopt@a\n"
     "`claude plugin install topopt --marketplace chartergroup/topopt`\n", "without the release tag"),
    ("claude plugin marketplace add chartergroup/topopt#v0.2.0\nclaude plugin install topopt@a\n"
     "claude plugin install topopt --marketplace chartergroup/topopt#main\n", "another ref"),
])
def test_a_one_command_form_with_another_tag_or_none_stops_the_pin(readme, message):
    with pytest.raises(ValueError, match=message):
        pin_module().pin(readme, "0.3.0", "topopt")


def test_the_one_command_option_patterns_keep_the_slug_boundary():
    release_pin = pin_module()
    assert release_pin.MARKETPLACE_OPTION.findall("--marketplace chartergroup/topopt#v1.2.3") == ["v1.2.3"]
    assert release_pin.MARKETPLACE_OPTION.findall("--marketplace chartergroup/topopt-demo#v1.2.3") == []
    assert not release_pin.UNPINNED_OPTION.search("--marketplace chartergroup/topopt#v1.2.3")
    assert not release_pin.UNPINNED_OPTION.search("--marketplace chartergroup/topopt-demo")
    assert release_pin.UNPINNED_OPTION.search("--marketplace chartergroup/topopt and")
    for ref in ("#main", "#latest", "#v1.2", "#v1.2.3-rc1", "#ux/357", "#v1.2.3.4", "#v1.2.3+x", "#v1.2.3/x"):                     # a ref that is not a release tag
        assert release_pin.UNPINNED_OPTION.search(f"--marketplace chartergroup/topopt{ref}"), ref
    assert not release_pin.UNPINNED_OPTION.search("type --marketplace chartergroup/topopt#v1.2.3. Then")   # a full stop ends the sentence


@pytest.mark.parametrize("name", ["README.md", DOC])
def test_the_readme_and_the_install_doc_do_not_name_the_old_repo(name):
    text = read(name)
    assert not other_owners(text), "another owner of the repo is named"
    assert RENAMED_SLUG not in text, "the repo is named topopt now"
    assert re.search(rf"{re.escape(NEW_SLUG)}(?![\w-])", text), "the new slug, with a boundary"
    assert "topopt-demo checkout" not in text and "./topopt-demo" not in text, "the clone folder is named after the repo"


def test_the_readme_leads_with_latest_and_names_no_release_tag():
    """#385: the README has the moving ref `#latest` and the marketplace name of main. The tags are in the install doc."""
    release_pin = pin_module()
    readme = read("README.md")
    assert release_pin.readme_problems(readme, "topopt") == []
    lines = [line.strip() for line in readme.splitlines()]
    assert lines.index(f"claude plugin marketplace add {NEW_SLUG}#latest") + 1 == lines.index("claude plugin install topopt@topopt")
    assert f"/plugin marketplace add {NEW_SLUG}#latest" in readme
    assert not re.search(rf"{NEW_SLUG}#v\d", readme), "a release tag in the README would never move"


@pytest.mark.parametrize("readme, message", [
    ("claude plugin marketplace add chartergroup/topopt#v0.3.0\nclaude plugin install topopt@topopt\n", "needs the command"),
    ("claude plugin marketplace add chartergroup/topopt#latest\nclaude plugin install topopt@topopt\n"
     "claude plugin install topopt --marketplace chartergroup/topopt#v0.3.0\n", "names a release tag"),
    ("claude plugin marketplace add chartergroup/topopt#latest\nclaude plugin install topopt@topopt-demo\n", "needs topopt@topopt"),
    ("claude plugin marketplace add chartergroup/topopt#latest\n", "needs topopt@topopt"),
    # the session form carries the name too (F3 of the review)
    ("claude plugin marketplace add chartergroup/topopt#latest\nclaude plugin install topopt@topopt\n"
     "`/plugin install topopt@topopt-demo`\n", "needs topopt@topopt"),
])
def test_the_readme_check_stops_a_pinned_or_wrong_readme(readme, message):
    found = pin_module().readme_problems(readme, "topopt")
    assert any(message in line for line in found), found


def test_the_release_step_stops_on_a_pinned_readme_and_leaves_it_alone(tmp_path, capsys):
    release_pin = pin_module()
    doc = readme_at_the_first_release()
    pinned = "claude plugin marketplace add chartergroup/topopt#v0.3.0\nclaude plugin install topopt@topopt\n"
    repo = fake_repo(tmp_path / "pinned", "0.3.0", doc, readme=pinned)      # the doc names v0.1.0: a run without the README problem would change it
    assert release_pin.main(["--check"], repo) == 1 and release_pin.main([], repo) == 1
    assert "needs the command" in capsys.readouterr().err
    assert (repo / DOC).read_text(encoding="utf-8") == doc, "the script writes nothing when the README is wrong"
    assert (repo / "README.md").read_text(encoding="utf-8") == pinned, "the script never writes the README"


def test_the_readme_session_form_must_name_the_marketplace_of_main():
    """F3: `/plugin install topopt@<name>` in the README is read too, not only the terminal command."""
    readme = read("README.md")
    assert "/plugin install topopt@topopt" in readme
    wrong = readme.replace("/plugin install topopt@topopt", "/plugin install topopt@topopt-demo")
    assert any("needs topopt@topopt" in line for line in pin_module().readme_problems(wrong, "topopt"))


def lines_in_order(text: str, first: str, second: str) -> bool:
    """`first` and `second` are on two lines of one block, `first` before `second`."""
    lines = [line.strip() for line in text.splitlines()]
    return first in lines and second in lines and lines.index(first) < lines.index(second)


def test_the_update_commands_of_a_latest_install_are_in_the_readme_and_the_install_doc_in_this_order():
    """F4: without `marketplace update`, `plugin update` does not see the new release (checked 2026-10-27, LATEST-CHECK.md)."""
    first, second = "claude plugin marketplace update topopt", "claude plugin update topopt@topopt"
    doc = read(DOC)
    assert lines_in_order(doc, first, second), "the install doc: the update route of #latest"
    readme = read("README.md")
    one_line = re.sub(r"\s+", " ", readme)
    assert f"\n{first}\n{second}\n" in readme, "the README: the two commands in one code block, the marketplace update first"
    assert "update an install from `#latest`" in one_line, "the README says that the line is for #latest"
    assert "docs/reference/install.md#update" in readme, "the README links the pinned update"


def test_a_latest_add_command_in_the_install_doc_does_not_count_as_a_tag():
    """The two-command form with `#latest` is right in the install doc; only the one-command option is refused there."""
    release_pin = pin_module()
    doc = read(DOC)
    assert f"claude plugin marketplace add {NEW_SLUG}#latest" in doc
    assert not re.search(rf"--marketplace {NEW_SLUG}#latest", doc)
    assert release_pin.readme_facts(doc)[0] == f"v{version()}"


def test_the_current_files_do_not_name_the_repo_by_its_old_name():
    """The records of the past (CHANGELOG, plan logs, eval data) may keep the old name. These files are read today."""
    names = ["README.md", "RELEASING.md", "CONTRACTS.md", "pyproject.toml", ".claude-plugin/plugin.json",
             ".claude-plugin/marketplace.json", "packaging/mcpb/manifest.json", "topopt/doctor.py", "topopt/gpu_check.py",
             "scripts/ci/release_pin.py", "scripts/plugin-e2e.sh"]
    names += sorted(str(p.relative_to(REPO)) for p in (REPO / ".github" / "workflows").glob("*.yml"))
    for name in names:
        assert RENAMED_SLUG not in read(name), f"{name} names the repo by its old name {RENAMED_SLUG}"


def test_the_install_doc_has_the_update_route_that_the_releasing_file_links_to():
    readme = read(DOC)
    assert "\n#### Update to a newer release\n" in readme
    assert f"{DOC}#update-to-a-newer-release" in read("RELEASING.md")
    assert "README.md#update-to-a-newer-release" not in read("RELEASING.md"), "the section left the README"
    name = pin_module().readme_facts(readme)[1]  # the marketplace name of the pinned tag, as the README holds it
    for command in (f"claude plugin marketplace remove {name}",
                    f"claude plugin update topopt@{name}",
                    f"extraKnownMarketplaces.{name}.source.ref"):
        assert command in readme, f"the update route needs `{command}`"


def test_the_plugin_manifest_names_the_new_repo():
    plugin = json.loads(read(".claude-plugin/plugin.json"))
    assert plugin["homepage"] == NEW_URL
    assert plugin["repository"] == NEW_URL


def test_the_mcpb_manifest_names_the_new_repo():
    manifest = json.loads(read("packaging/mcpb/manifest.json"))
    assert manifest["repository"]["url"] == f"{NEW_URL}.git"
    assert manifest["homepage"] == NEW_URL
    assert manifest["documentation"] == f"{NEW_URL}#readme"
    assert manifest["support"] == f"{NEW_URL}/issues"


def test_the_manifests_name_the_new_owner():
    """The author or owner in each manifest is the organization of the repo, not the old user name."""
    owner = NEW_SLUG.split("/")[0]
    marketplace = json.loads(read(".claude-plugin/marketplace.json"))
    plugin = json.loads(read(".claude-plugin/plugin.json"))
    manifest = json.loads(read("packaging/mcpb/manifest.json"))
    assert marketplace["owner"]["name"] == owner
    assert plugin["author"]["name"] == owner
    assert manifest["author"] == {"name": owner, "url": f"https://github.com/{owner}"}
    assert f"io.github.{owner}.topopt" in manifest["_meta"]
    for name in (".claude-plugin/marketplace.json", ".claude-plugin/plugin.json", "packaging/mcpb/manifest.json"):
        assert not other_owners(read(name)), f"{name} names another owner of the repo"


def test_the_package_metadata_names_the_organization_and_the_repo():
    project = tomllib.loads(read("pyproject.toml"))["project"]
    assert project["authors"] == [{"name": NEW_SLUG.split("/")[0]}]
    assert project["urls"]["Homepage"] == NEW_URL
    assert project["urls"]["Repository"] == NEW_URL
    assert project["urls"]["Issues"] == f"{NEW_URL}/issues"


def test_the_releasing_file_names_the_three_version_files():
    text = read("RELEASING.md")
    for name in VERSION_FILES:
        assert f"`{name}`" in text, f"RELEASING.md needs to name {name}"


def test_the_changelog_has_an_entry_for_the_version():
    # The first entry was written by hand (`## 0.1.0 ...`); the release workflow writes `## vX.Y.Z (date)`.
    entries = re.findall(r"^## v?(\d+\.\d+\.\d+)\b", read("CHANGELOG.md"), re.M)
    assert version() in entries, f"CHANGELOG.md needs a heading `## v{version()}`"
    assert entries[0] == version(), "the newest entry comes first, and it is the version of the files"


def test_the_changelog_has_the_line_where_the_release_workflow_adds_entries():
    text = read("CHANGELOG.md")
    assert text.count("\n<!-- version list -->\n") == 1, "python-semantic-release adds entries only after this line"
    assert text.index("<!-- version list -->") < text.index("\n## "), "the line must come before the first entry"


def test_the_release_build_step_writes_the_tag_into_the_install_doc_and_commits_it():
    config = tomllib.loads(read("pyproject.toml"))["tool"]["semantic_release"]
    assert config["build_command"].startswith("python3 scripts/ci/release_pin.py && ")
    assert DOC in config["assets"] and "README.md" not in config["assets"], "the script writes the install doc, not the README"
    assert pin_module().INSTALL_DOC == DOC


def test_the_pin_script_moves_every_mention_of_the_release_tag_to_a_new_version():
    release_pin = pin_module()
    readme = read(DOC)
    assert release_pin.pin(readme, version()) == readme, "the README on main names the tag of the version"
    moved = release_pin.pin(readme, "0.42.7")
    assert f"claude plugin marketplace add {NEW_SLUG}#v0.42.7" in moved
    assert "git -C topopt checkout v0.42.7\n" in moved
    assert f"v{version()}" not in moved, "no mention of the old tag is left"
    assert f"{NEW_SLUG}#<new-tag>" in moved, "the placeholder of the update route stays"
    other = sorted(set(re.findall(r"\bv\d+\.\d+\.\d+\b", readme)) - {f"v{version()}"})
    assert other == sorted(set(re.findall(r"\bv\d+\.\d+\.\d+\b", moved)) - {"v0.42.7"}), "other versions stay"


# --- the marketplace name of the pinned tag (REL-8, #313) ----------------------------------------------------------

# The paragraph of the update section that talks about the names. It has no tag and none of the four command forms,
# so the pin script must leave it as it is.
NAMES_PARAGRAPH_START = "The releases up to version 0.2.0 name the marketplace `topopt-demo`. Later releases name it `topopt`."


def names_paragraph(readme: str) -> str:
    start = readme.index(NAMES_PARAGRAPH_START)
    return readme[start:readme.index("\n\n", start)]


def test_the_fixed_record_of_the_first_release_names_the_marketplace_of_its_tag():
    release_pin = pin_module()
    assert release_pin.RELEASES_BEFORE_THE_RENAME == {"0.1.0": FIRST_RELEASE_MARKETPLACE, "0.2.0": FIRST_RELEASE_MARKETPLACE}
    assert release_pin.marketplace_of_release("0.1.0") == release_pin.marketplace_of_release("0.2.0") == FIRST_RELEASE_MARKETPLACE
    assert release_pin.marketplace_of_release("0.3.0") == json.loads(read(".claude-plugin/marketplace.json"))["name"] == "topopt"


def test_the_fixed_record_equals_the_marketplace_file_of_the_tag_when_the_tag_is_here():
    """CI checkouts have no tags (the script reads none); a clone has. Then the record must be true."""
    record = pin_module().RELEASES_BEFORE_THE_RENAME
    shown = {v: subprocess.run(["git", "show", f"v{v}:.claude-plugin/marketplace.json"], cwd=REPO, capture_output=True, text=True)
             for v in record}
    if any(r.returncode != 0 for r in shown.values()):
        pytest.skip("a tag of the record is not in this checkout")
    assert {v: json.loads(r.stdout)["name"] for v, r in shown.items()} == record


def test_the_readme_names_the_marketplace_of_the_pinned_tag_in_every_command():
    release_pin = pin_module()
    readme = read(DOC)
    tag, name = release_pin.readme_facts(readme)
    assert tag == f"v{version()}"
    assert name == release_pin.marketplace_of_release(version())
    assert release_pin.readme_names(readme) == {name}
    assert f"claude plugin install topopt@{name}\n" in readme


def test_the_readme_tells_a_user_of_the_first_release_how_to_move_to_a_later_one():
    paragraph = names_paragraph(read(DOC))
    assert "`claude plugin marketplace list`" in paragraph and "Route A" in paragraph
    assert "Route B cannot make this move" in paragraph and "settings entry" in paragraph
    assert not re.search(r"\bv\d+\.\d+\.\d+\b", paragraph), "a tag in the paragraph would move with the pin"
    assert pin_module().readme_names(paragraph) == set(), "a command form in the paragraph would change with the pin"


def test_a_later_release_moves_the_readme_to_the_marketplace_of_its_tag():
    release_pin = pin_module()
    old = readme_at_the_first_release()
    assert release_pin.readme_facts(old) == ("v0.1.0", "topopt-demo")
    moved = release_pin.pin(old, "0.3.0", "topopt")
    assert release_pin.readme_facts(moved) == ("v0.3.0", "topopt")
    # the tag
    assert f"claude plugin marketplace add {NEW_SLUG}#v0.3.0" in moved and "git -C topopt checkout v0.3.0\n" in moved
    assert f"{NEW_SLUG}#<new-tag>" in moved, "the placeholder of the update route stays"
    assert "v0.1.0" not in moved, "no mention of the old tag is left"
    # the marketplace name, in each command form
    assert "claude plugin install topopt@topopt\n" in moved and "claude plugin update topopt@topopt\n" in moved
    assert "claude plugin marketplace remove topopt\n" in moved
    assert "extraKnownMarketplaces.topopt.source.ref" in moved
    assert "Marketplace 'topopt' not found" in moved
    assert "topopt-demo" not in release_pin.readme_names(moved) and release_pin.readme_names(moved) == {"topopt"}
    assert "topopt@topopt-demo" not in moved and "remove topopt-demo" not in moved
    # the paragraph about the names, and every other line that is no command, stay as they are
    assert names_paragraph(moved) == names_paragraph(old)
    changed = [a for a, b in zip(old.splitlines(), moved.splitlines()) if a != b]
    assert len(old.splitlines()) == len(moved.splitlines()) and changed, "the pin changes lines and adds or removes none"
    assert all(re.search(r"v0\.1\.0|topopt-demo", line) for line in changed), "only lines with the tag or the name change"


def test_a_later_release_with_the_same_marketplace_name_changes_only_the_tag():
    release_pin = pin_module()
    old = pin_module().pin(read(DOC), "0.3.0", "topopt")
    moved = release_pin.pin(old, "0.4.0", "topopt")
    assert release_pin.readme_names(moved) == {"topopt"} and "v0.3.0" not in moved and "claude plugin marketplace add " + NEW_SLUG + "#v0.4.0" in moved
    assert release_pin.pin(old, "0.4.0") == moved, "without a name the names stay, and they are the same"


def test_the_pin_without_a_marketplace_name_keeps_the_names():
    release_pin = pin_module()
    old = readme_at_the_first_release()
    assert release_pin.readme_names(release_pin.pin(old, "0.1.1")) == {"topopt-demo"}


def test_the_add_command_with_the_old_slug_is_not_an_install_command():
    """`chartergroup/topopt-demo#v0.1.0` must not match as `chartergroup/topopt`: after the rename the old slug is a mistake."""
    release_pin = pin_module()
    wrong = "claude plugin marketplace add chartergroup/topopt-demo#v0.1.0\nclaude plugin install topopt@topopt-demo\n"
    with pytest.raises(ValueError, match="one tag"):
        release_pin.pin(wrong, "0.3.0", "topopt")
    assert release_pin.ADD_COMMAND.findall("claude plugin marketplace add chartergroup/topopt#v1.2.3") == ["v1.2.3"]


@pytest.mark.parametrize("readme, message", [
    ("claude plugin marketplace add chartergroup/topopt#v0.1.0\n", "one marketplace in its install command"),
    ("claude plugin marketplace add chartergroup/topopt#v0.1.0\nclaude plugin install topopt@a\nclaude plugin install topopt@b\n",
     "one marketplace in its install command"),
    ("claude plugin marketplace add chartergroup/topopt#v0.1.0\nclaude plugin install topopt@a\nclaude plugin marketplace remove b\n",
     "one marketplace name in all its commands"),
    ("claude plugin install topopt@a\n", "one tag"),
])
def test_a_readme_that_is_not_clear_about_the_tag_or_the_name_stops_the_pin(readme, message):
    with pytest.raises(ValueError, match=message):
        pin_module().pin(readme, "0.3.0", "topopt")


def test_check_passes_on_the_real_readme_now():
    done = subprocess.run([sys.executable, "-P", str(REPO / "scripts" / "ci" / "release_pin.py"), "--check"],
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "" and done.stderr == ""


def test_check_fails_on_a_readme_that_names_the_new_marketplace_with_the_first_tag():
    release_pin = pin_module()
    wrong = release_pin.pin(readme_at_the_first_release(), "0.1.0", "topopt")  # topopt@topopt with the tag v0.1.0
    assert "claude plugin install topopt@topopt\n" in wrong and "#v0.1.0" in wrong
    found = release_pin.problems(wrong, "0.1.0", release_pin.marketplace_of_release("0.1.0"))
    assert len(found) == 1 and "marketplace name topopt;" in found[0] and "holds the marketplace name topopt-demo" in found[0]
    assert release_pin.problems(readme_at_the_first_release(), "0.1.0", release_pin.marketplace_of_release("0.1.0")) == []


def test_check_wants_the_name_of_the_marketplace_file_for_a_later_version():
    release_pin = pin_module()
    old = readme_at_the_first_release()
    assert release_pin.problems(old, "0.3.0", release_pin.marketplace_of_release("0.3.0")) == [
        f"{DOC} names the tag v0.1.0; the version in pyproject.toml needs the tag v0.3.0",
        f"{DOC} uses the marketplace name topopt-demo; the tag v0.3.0 holds the marketplace name topopt"]
    assert release_pin.problems(release_pin.pin(old, "0.3.0", "topopt"), "0.3.0", "topopt") == []


GOOD_README = (f"claude plugin marketplace add {NEW_SLUG}#latest\nclaude plugin install topopt@topopt\n")


def fake_repo(root: Path, release: str, doc: str, marketplace: str = "topopt", readme: str = GOOD_README) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / "docs" / "reference").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{release}"\n', encoding="utf-8")
    (root / ".claude-plugin" / "marketplace.json").write_text(json.dumps({"name": marketplace}), encoding="utf-8")
    (root / DOC).write_text(doc, encoding="utf-8")
    (root / "README.md").write_text(readme, encoding="utf-8")
    return root


def test_the_release_step_writes_the_tag_and_the_name_and_the_check_then_passes(tmp_path, capsys):
    release_pin = pin_module()
    old = readme_at_the_first_release()
    # python-semantic-release has written 0.3.0 into the version files; the marketplace file names `topopt`.
    repo = fake_repo(tmp_path / "release", "0.3.0", old)
    assert release_pin.main(["--check"], repo) == 1
    assert "needs the tag v0.3.0" in capsys.readouterr().err
    assert release_pin.main([], repo) == 0
    assert "names v0.3.0 and the marketplace topopt" in capsys.readouterr().out
    written = (repo / DOC).read_text(encoding="utf-8")
    assert written == release_pin.pin(old, "0.3.0", "topopt") and release_pin.readme_facts(written) == ("v0.3.0", "topopt")
    assert release_pin.main(["--check"], repo) == 0
    assert release_pin.main([], repo) == 0 and (repo / DOC).read_text(encoding="utf-8") == written, "a second run changes nothing"


def test_the_first_release_keeps_its_readme_and_the_check_stops_a_new_name_on_it(tmp_path, capsys):
    release_pin = pin_module()
    old = readme_at_the_first_release()
    repo = fake_repo(tmp_path / "first", "0.1.0", old)  # the marketplace file of main names `topopt`; the record names the tag's name
    assert release_pin.main(["--check"], repo) == 0
    assert release_pin.main([], repo) == 0 and (repo / DOC).read_text(encoding="utf-8") == old
    capsys.readouterr()
    bad = fake_repo(tmp_path / "bad", "0.1.0", release_pin.pin(old, "0.1.0", "topopt"))
    assert release_pin.main(["--check"], bad) == 1
    assert "holds the marketplace name topopt-demo" in capsys.readouterr().err
