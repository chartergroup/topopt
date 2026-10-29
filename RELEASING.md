# Releasing topopt

This file is for maintainers. It tells how to make a release. It is the one description of the release process.

## What a release is

A release is a tag `vX.Y.Z` on a commit of `main`, and a GitHub release for that tag, with the wheel, the sdist and the
`.mcpb` file attached (the `.mcpb` also as `topopt.mcpb`). [CHANGELOG.md](CHANGELOG.md) lists the changes of each release.
The README leads with `#latest` (the branch that this workflow moves to the newest release). The pinned install commands in
[docs/reference/install.md](docs/reference/install.md#install-the-claude-code-plugin) name the newest tag:

```
claude plugin install topopt --marketplace chartergroup/topopt#vX.Y.Z
claude plugin marketplace add chartergroup/topopt#vX.Y.Z
```

The first form is also in the install prompt of the install doc and in the chat command `/plugin install topopt --marketplace ...`.
`scripts/ci/release_pin.py` moves the tag in every form, and it stops on a `--marketplace` form without a release tag
(no tag, or another ref such as `#main`).

A new user gets the code of that tag, not the code that is on `main` at that moment.
Claude Code registers the marketplace under the `name` in `.claude-plugin/marketplace.json` OF THAT TAG. So the other pinned
commands of the install doc (`claude plugin install topopt@<name>`) must use the name that the tag holds, and not the name that `main` holds now.
The tags `v0.1.0` and `v0.2.0` hold the name `topopt-demo`. Every later tag holds `topopt` (the repo and the marketplace were renamed).

The workflow `release` (`.github/workflows/release.yml`) makes the release with
[python-semantic-release](https://python-semantic-release.readthedocs.io/) (v9). Its config is in `pyproject.toml`,
section `[tool.semantic_release]`. Agents cannot push tags or start a workflow: a person with write access to the repo
starts it.

## What the workflow does

1. It reads the Conventional Commits on `main` since the last tag `vX.Y.Z`.
2. It picks the next version: `feat` raises the minor number, `fix` and `perf` raise the patch number. `docs`, `test`,
   `chore`, `ci`, `refactor` and untyped commits make no release. While the major number is 0, a breaking change
   (`feat!:` or a `BREAKING CHANGE:` footer) raises the minor number. 1.0.0 is a decision of a person: set it by hand.
3. It writes the version in the three version files (see "The version").
4. The workflow runs the tool with `--no-changelog`, so the tool writes no entry of its own. The entry of the new version is
   written by the build step (next point), after the line `<!-- version list -->` of `CHANGELOG.md`.
   The heading of the entry is `## vX.Y.Z (date)`. See "The release notes".
5. It runs the build step (`build_command`). The first command, `scripts/ci/release_pin.py`, writes two things into the
   install doc (`docs/reference/install.md`). First, the new tag, in every place that names the tag of the install command (the add command, the checkout
   line, the text about the tag). Second, the marketplace name of `.claude-plugin/marketplace.json`, in the four command
   forms that carry a name: `topopt@<name>`, `claude plugin marketplace remove <name>`, `extraKnownMarketplaces.<name>.`
   and `Marketplace '<name>' not found`. It reads the old name from the install command of the install doc. The paragraph in the
   install doc that tells about the two names has no tag and none of these forms, so the script does not change it.
   The script writes nothing into the README. It only checks that the README uses `chartergroup/topopt#latest` and `topopt@<name of main>`, and names no tag.
   The second command, `scripts/ci/release_notes.py`, writes the entry into `CHANGELOG.md` and into `dist/release-notes.md`.
   Then it builds the wheel, the sdist and the `.mcpb`.
6. It commits these files, `docs/reference/install.md` and `CHANGELOG.md` as `chore(release): X.Y.Z`, makes the tag `vX.Y.Z`, and pushes both to `main`.
7. It makes the GitHub release. The tool uploads each `assets` entry of `pyproject.toml` to the release, so
   every release has the downloads `install.md` and `CHANGELOG.md` (the file name only, without the folder). It does not upload the built files: that would be its `publish`
   command, which the workflow does not run. Then the workflow replaces the notes of the release with
   `dist/release-notes.md` (`gh release edit`): the same text as the `CHANGELOG.md` entry. The next step ("Attach the files to the
   release") uploads the wheel, the sdist, `topopt-X.Y.Z.mcpb` and the same `.mcpb` as `topopt.mcpb` with `scripts/ci/release_assets.py`
   (`gh release upload`, only the files that the release does not list yet), and fails when the asset list of the release does not have all
   four, or when the two `.mcpb` assets differ in size. The `.mcpb` is the file that a Claude Desktop user downloads. The workflow installs the `mcpb` CLI first, so the build checks the manifest (`mcpb validate`), and a
   bad manifest stops the run before the release commit.
8. It moves the branch `latest` to the release commit (see "The branch latest").

## The release notes

The notes are short and for people: one line for each merged pull request, under the type sections. The GitHub release gives each line
a link to its PR. The public `CHANGELOG.md` carries the same text without the PR links. This example is the text of a release note:

```
## v0.2.0 (2026-10-26)

_This release is published under the MIT License._

### Features

- `topopt check` warns when the design-space STL looks like it is in inches or metres.

---

**Detailed Changes**: [v0.1.0...v0.2.0](https://github.com/chartergroup/topopt/compare/v0.1.0...v0.2.0)
```

How `scripts/ci/release_notes.py` makes it:

- **The PRs.** Each merge commit `Merge pull request #N from ...` on the first-parent line of `main` since the last tag is one PR.
  The title is the body of the merge commit (the PR title). A squash commit `type(scope): subject (#N)` is one PR too.
  The release commit `chore(release): X.Y.Z` and other merge commits are skipped. A direct commit with no PR gives a line
  without a link, in the section of its own type.
- **The section.** The highest type of the PR title and of the commits that the PR brought: feat, fix, perf, docs, chore, ci,
  test, then other types (build, refactor, style), then lines with no type ("Other Changes"). So a `chore` PR that holds a `feat` commit is
  listed under Features. Sections: Features, Bug Fixes, Performance Improvements, Documentation, Chores, Continuous Integration, Testing.
- **The sentence.** The title without `type(scope):`, without trailing item codes, without a `(requested by ...)` suffix,
  with an upper case first letter (except a code name such as `list_runs`) and one final period, after the link.
  An item code is the first word in the last brackets of the title, in one of two forms: (A) one capital letter, one or two digits and
  an optional lower-case letter (`U21`, `I2`, `H1`, `P4b`, `B1`); (B) a word that starts with a capital letter and has a dash and a digit
  (`F4-47`, `K1-FU-C`, `LAUNCH-3`, `P2d-D3`, `Q1-fix`, `PM-2`). Words may follow it after a space, comma, plus or colon
  (`(F1-8 step 2)`, `(I2 parts 3 and 4)`, `(P4b-4: F4-11)`, `(B1: B-098, part of B-060)`). Normal words stay: `(IPv6 support)`, `(MP4 export)`,
  `(H100 tested)`, `(STL v2)`, `(3D)`. Limits: a code in nested brackets, a code that is not in the last brackets, and a leading code
  (`K1-A: ...`) stay in the line. The title rule keeps them out.
  Nothing else is reworded: the title must be a good sentence. A `!` after the type, or a `BREAKING CHANGE:` footer in a
  commit of the PR, puts **Breaking:** before the sentence. No commit body, no `Co-Authored-By`, no trailers.
- **The check before a release.** The dry run of the workflow (`dry_run=true`) prints the notes of the next release. Read them.
  A bad title is fixed on the PR before it is merged; after the merge the merge commit holds it. Locally:
  `python3 -P scripts/ci/release_notes.py --version X.Y.Z` prints the entry of the next release.

## If the release run fails after the tag

The tool pushes the release commit and the tag, then makes the GitHub release. A later step (the release text, the files of the release,
`latest`) can still fail.
Start a new run of the workflow ("Run workflow") with `dry_run=false` on the same `main`. Do not use "Re-run jobs": a re-run checks out
the commit of the first run, which has no tag. The tool says "already released", and the workflow finds the tag on `HEAD`
(`git tag --points-at HEAD`). Then it does the rest: it writes `dist/release-notes.md` again if the build step did not run
(`release_notes.py --head <tag>^`), creates the GitHub release if it is missing (`gh release create --verify-tag`), replaces its text, builds the wheel, the sdist and the
`.mcpb` (no build step ran) and attaches the ones that the release does not list yet, and moves `latest`. Each part is safe to repeat: a second
push of the same commit gives "Everything up-to-date". Until the files are attached, the newest release has no `topopt.mcpb`, so the link
`releases/latest/download/topopt.mcpb` gives a 404 (also in the minutes between the release and the attach step of a run that works).

If `main` has moved on with no new release, a re-run does not find the tag on `HEAD`. Then do the three steps by hand, in this order, and
push `latest` only after the run of the second step has passed. The push is the one allowed exception to "nobody pushes to `latest` by hand":

```
gh release edit vX.Y.Z --notes-file <file with the entry of vX.Y.Z>
gh workflow run release.yml --repo chartergroup/topopt -f attach_to_tag=vX.Y.Z
git push origin vX.Y.Z^{commit}:refs/heads/latest
```

## Attach the files to an existing release

The releases up to v0.3.0 were made before the step "Attach the files to the release" existed, so they have only `README.md` and
`CHANGELOG.md`. The releases up to v0.4.0 were made before the name `topopt.mcpb` (see "The Desktop extension file"). To attach the
missing files to such a release (v0.3.0 or later), start the workflow on `main` with `attach_to_tag` = the tag, for example `v0.3.0`
(`dry_run` is not used then):

```
gh workflow run release.yml --repo chartergroup/topopt -f attach_to_tag=v0.3.0
```

This runs only the job `attach`: no version, no commit, no tag, no change of `latest`. It checks out `main` for the script, adds a checkout of
the tag, builds the files of that checkout when the release lacks one of them (its own `packaging/mcpb/build.py` and version, with
`mcpb validate`), uploads the ones that the release does not list yet and checks the asset list. It never replaces a file that the release has: a new build gives other bytes for the
wheel and the sdist (their archives hold the build time). So it is safe to repeat. When the release has `topopt-X.Y.Z.mcpb` and lacks
`topopt.mcpb` (v0.4.0), the copy is made from the listed file (`gh release download`), so the two have the same bytes. `topopt.mcpb`
goes up in a second upload, after the release lists `topopt-X.Y.Z.mcpb`. A release that lists `topopt.mcpb` without `topopt-X.Y.Z.mcpb`,
or both with other sizes, fails the run before any upload; the message names the file to delete on the release page.

## The branch latest

`latest` is a branch (not a tag) that always points to the commit of the newest release. Install pages outside this repo can name one
ref: `claude plugin marketplace add chartergroup/topopt#latest` (with `topopt@topopt` after the first release after the rename;
the marketplace name of `v0.1.0` and `v0.2.0` is `topopt-demo`, so do not use `#latest` with `topopt@topopt` before that release).
Only the release workflow moves it: after the release step it pushes the commit of the new tag to `refs/heads/latest`,
fast-forward only, no force. If `latest` does not exist, the push creates it. If the push is not a fast-forward, the step fails: find the reason,
do not force. Nobody pushes to `latest` by hand. `latest` is not a release branch of python-semantic-release
(`branches.main.match` is `^main$`). The README leads with `#latest`, so a first-time user gets the newest release without a tag. The install doc keeps the pinned tag
(reproducible, and `release_pin.py` moves it) for a user who wants a fixed release. A dry run prints where `latest` is now.

## The version

The version is in three files. It must be the same in all three files:

| File | Key |
|---|---|
| `pyproject.toml` | `project.version` |
| `.claude-plugin/plugin.json` | `version` |
| `packaging/mcpb/manifest.json` | `version` |

Claude Code compares only the `version` in `.claude-plugin/plugin.json`. A release with an unchanged version does not reach
the users who have the plugin: `claude plugin update` answers "already at the latest version" and keeps the old files.
(A test with Claude Code 2.1.293 on 2026-10-25 showed this. A new commit with the same version did not change the installed plugin.)

`tests/test_release_pin.py` fails when the three versions differ, when the install doc command does not name
`v<version>`, when the install doc uses another marketplace name than the tag `v<version>` holds, when the README does not use
`#latest` or names a tag, when `CHANGELOG.md` has no
entry for the version or no line `<!-- version list -->`, or when the build step does not write the tag of the install doc.

## The first release after the rename

The install doc on `main` names the repo `chartergroup/topopt`. Until the first release after the rename, it still names the
tag `v0.2.0` and the marketplace name `topopt-demo` of that tag (`topopt@topopt-demo`). This is true at every moment,
because nobody changes the tag `v0.2.0`. The first release after the rename (0.2.1 or 0.3.0) makes `release_pin.py` move
the install doc to `topopt@topopt`, `claude plugin marketplace remove topopt` and `extraKnownMarketplaces.topopt.`. From then
on `main` and the newest tag hold the same name.

`--check` knows that the versions 0.1.0 and 0.2.0 used `topopt-demo` from the record `RELEASES_BEFORE_THE_RENAME` in the script
(`0.1.0` and `0.2.0`; both were released before the rename). CI checkouts have no tags, so the script reads no tags. Do not add a new release to that record.

A user who installed `v0.1.0` or `v0.2.0` has a marketplace named `topopt-demo`. For them the move to the new release is Route A of the
install doc with `topopt-demo` in its first command; Route B cannot change a marketplace name. Say this in the notes of that
release.

## How to start it

The workflow starts by hand only (the same rule as `tests.yml`). Open the Actions tab, select the
workflow "release", and run it on `main`. Or:

```
gh workflow run release.yml --repo chartergroup/topopt -f dry_run=true
```

`dry_run=true` (the default) only prints the next version and runs no tests. Run it first. Then run it with `dry_run=false`.

`dry_run=false` runs no tests. Run `tests.yml` by hand on the commit before you release, and release only when it is green.
To make the tests part of the release, add a job that calls `tests.yml` (it has a `workflow_call` trigger) and make the job `release` need it.

## The first release: v0.1.0

The version is already 0.1.0 in the three files, the install doc names `v0.1.0`, and `CHANGELOG.md` has an entry for 0.1.0
that was written by hand. There are two ways to make this release:

- **By hand (recommended for v0.1.0).** An owner creates the tag `v0.1.0` on `main` and the GitHub release for it, with
  the entry 0.1.0 of `CHANGELOG.md` as the notes. On GitHub: Releases, "Draft a new release", the tag `v0.1.0` on `main`.
  Or with git: `git tag v0.1.0 <commit> && git push origin v0.1.0`. The workflow then starts from this tag, so its first
  release is 0.1.1 or 0.2.0.
- **With the workflow.** With no tag, the tool reads the whole history and picks 0.1.0. A local run on 2026-10-25 made
  the tag `v0.1.0` with no release commit, and `tests/test_release_pin.py` passed. But the notes of the GitHub release
  then come from all the commit messages since the start of the repo, not from the entry that was written by hand.

Nobody has run the install against GitHub with a tag yet. The check below is the first run of that path. Do it right
after the tag exists.

## After a release: check the install

Make a fresh install from the pinned command of the install doc in an empty config folder:

```
cfg=$(mktemp -d)
CLAUDE_CONFIG_DIR=$cfg claude plugin install topopt --marketplace chartergroup/topopt#vX.Y.Z
CLAUDE_CONFIG_DIR=$cfg claude plugin list
```

Then do the same with the two-command form in a second empty folder (`claude plugin marketplace add chartergroup/topopt#vX.Y.Z`,
then `claude plugin install topopt@<marketplace>`).

`<marketplace>` is the `name` in `.claude-plugin/marketplace.json` of that tag: `topopt-demo` for `v0.1.0` and `v0.2.0`, `topopt` for
every later tag. It must be the name that the install doc uses after the release step (`release_pin.py --check`). Also run
`claude plugin marketplace list` once: it must show that name.
`claude plugin list` must show the new version. If the repository is private, git on the computer needs access to it.
Do a second check when `main` has a new commit after the tag. In the same config folder, run
`claude plugin marketplace update` and `claude plugin update topopt@<marketplace>`.
The installed version and files must not change.

Then check the path that the README leads with. Make a fresh install in a third empty config folder, after the workflow moved `latest`:

```
cfg=$(mktemp -d)
CLAUDE_CONFIG_DIR=$cfg claude plugin marketplace add chartergroup/topopt#latest
CLAUDE_CONFIG_DIR=$cfg claude plugin install topopt@topopt
CLAUDE_CONFIG_DIR=$cfg claude plugin list
git ls-remote origin refs/heads/latest
```

`claude plugin list` must show the new version. The commit that `git ls-remote` prints must be the commit of the new tag
(`git rev-parse vX.Y.Z^{commit}`).

## The Desktop extension file

The step "Attach the files to the release" attaches `topopt-X.Y.Z.mcpb`, with the wheel and the sdist, to every GitHub release. It also attaches
the same file as `topopt.mcpb`, a name without the version, so one link always gives the extension of the newest release:
https://github.com/chartergroup/topopt/releases/latest/download/topopt.mcpb (GitHub sends `releases/latest/download/NAME` to the asset
NAME of the newest release). The script checks that the release lists both names with the same size. The file names are in `scripts/ci/release_assets.py` (`built_files`, and `STABLE_MCPB` for `topopt.mcpb`); a test checks the first three
against `dist_glob_patterns` in `pyproject.toml`, which the workflow does not use for the upload (it does not run `semantic-release publish`).
To stop the upload, remove that step from `.github/workflows/release.yml` and change its test.

## One-time set-up (needs a repo admin)

- **Push to `main`.** If `main` is protected, the default `GITHUB_TOKEN` cannot push the release commit. Add the secret
  `RELEASE_TOKEN` (a fine-grained token or a GitHub App token with contents: write that may bypass the rule). The workflow
  uses it when it exists.
- **Push to `latest`.** The push moves `latest` over every commit since the last release. If one of them changes `.github/workflows/*`,
  GitHub can refuse the update for a token without the workflows right. `GITHUB_TOKEN` cannot get this right. A fine-grained `RELEASE_TOKEN`
  needs "Workflows: write" and "Contents: write". The next release will be such a case (the PR that added this step changes `release.yml`),
  so check the first real run. If GitHub refused the push, add the right to the token and repair the run (see "If the release run fails after the tag").

## Commit messages

`type(scope): subject`. The agent rules are in `CLAUDE.md`, section "Commit messages and releases". Merge commits of
pull requests give no version, but their body (the PR title) is the line of the PR in the release notes. Squash or merge so that the `feat`/`fix` commits stay on `main`.
The PR title is a sentence for a user of the tool: no item code, no name.

The PR title becomes a commit message on `main` (the publish step uses it as the message of the branch commit, and a squash
merge uses it too), so a PR with the title `verify: ...` makes no release even if it adds a feature. Check the title before you merge.

## Local check (changes nothing)

```
uv tool run --from 'python-semantic-release>=9.21,<10' semantic-release --noop version --print
```

The tool only releases from `main`; on another branch it says "isn't in any release groups".
`python3 scripts/ci/release_pin.py --check` tells whether the install doc names the tag of the current version and the
marketplace name of that tag, and whether the README keeps `#latest`.

## How users move to a new release

`claude plugin update` alone does not move an install that is pinned to a tag.
The two routes (remove and add again, or change the ref in the settings) are in the section
[Update to a newer release](docs/reference/install.md#update-to-a-newer-release) of the install doc.
A user of `#latest` needs no route: `claude plugin marketplace update topopt` and `claude plugin update topopt@topopt`.
The route with `claude plugin marketplace remove` also deletes the plugin data folder. For topopt this folder holds the
Python environment (about 500 MB on disk), so the next start sets up Python again (1 to 3 minutes). The route that changes
the ref keeps it. Say this in the notes of a release that users must move to.
For the move from `v0.1.0` or `v0.2.0` to the first release after the rename, use the route with `marketplace remove` (see
"The first release after the rename").
