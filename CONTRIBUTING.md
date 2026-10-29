# Contributing to topopt

This file is for a person or a coding agent that changes this repo. To make a part with topopt, read the [README](README.md).
Anyone can contribute. The maintainers use the same cycle as you do. If this file and the code differ, the code wins. Correct this file in the same change.

## The cycle

1. **Issue first.** Open an issue with the form "Task" (`.github/ISSUE_TEMPLATE/task.yml`). It has four parts:
   - Goal: one sentence. What is true when the work is done?
   - Why: who needs it, and what it changes.
   - Acceptance: a check that anyone can run, with the expected result.
   - Evidence: the facts that exist now (numbers, output, `path:line`). Write "none yet" if there are none.

   A short issue is fine. Say in the issue that you want to do the work, so that two people do not take the same one.
2. **Branch.** Make a branch from a fresh `main` for each issue: `git fetch origin && git switch -c fix/short-name origin/main`. One issue, one branch, one PR.
3. **Change.** Make the smallest change that reaches the goal. A change of behavior ships with a test that fails without the change.
   Change the files that go together (the table "Keep files in step" below).
4. **Test.** Run the targeted tests and the contract set (the section "Tests").
5. **Pull request.** The base is `main`. Use the template (`.github/pull_request_template.md`). Its parts:
   - `Closes #N`, so that the merge closes the issue.
   - What changed, and why.
   - Which tests you ran, with the result.
   - "New cards": the findings that are outside your issue (below).

   If your change builds on another open PR, say "contains #X" in the body. Keep the base `main`: a PR with a branch as base merges into that branch and not into `main`.
6. **New cards.** You will find other defects while you work. Do not fix them in the same PR. Write each one in the "New cards" block of the PR:
   one line for what, one for where (`path:line`) and one for the evidence (a command and its output). A maintainer opens an issue for each card.
   Write "none" if there are no findings.
7. **Review.** Someone who did not build the change reviews it. The reviewer reproduces a problem before they write it. The author answers each comment:
   changed (what), not changed (the evidence), or obsolete.
8. **Merge.** A person merges, not the author. CI starts by hand only: a maintainer starts the workflow `tests` (Actions tab, "Run workflow") before a risky merge.
   You do not need to start it.

### Roles in Claude Code

If you work with Claude Code in a clone, the project has two roles for the cycle. They are in `.claude/agents/`.

| Role | Model | Does |
|---|---|---|
| `dev-implementer` | sonnet | Takes one issue. Works on a branch, adds the tests, commits after each step, and writes a short report. |
| `dev-reviewer` | opus | Reviews a diff that it did not write. Reproduces each finding. Changes no file. |

Start the reviewer in a new session or as a sub-agent. The session that wrote the change is not the one to review it.
`agents/` and `topopt_mcp/roles/` are different: they are text for the agents of people who use topopt. They are not rules for you.

### Work with a coding agent

We develop with coding agents, and you can too. You do not need to set anything up: an agent client reads the agent files of the repo by itself.

- Claude Code reads `CLAUDE.md`, which includes `AGENTS.md`, and it has the roles in `.claude/agents/` (above).
- Other clients (Codex, Cursor, Gemini CLI, VS Code) read `AGENTS.md`.
- `AGENTS.md` sends the agent to this file for the cycle. It also tells the agent to run `python3 scripts/dev/session_mode.py` and to read the `Mode:` line.
  Outside the maintainers, the line is `Mode: contributor`: the agent uses the code, this file and `CONTRACTS.md`, and it does not touch the board.
- The rules of this file bind the agent as they bind you: an issue first, a branch from a fresh `main`, a test with a change of behavior, a new finding in "New cards"
  and not in the PR. The review is the same: someone who did not build the change reviews it, also when an agent built it.
- `agents/`, `skills/` and `topopt_mcp/roles/` are for people who make parts with topopt. They are not rules for the agent that changes this repo.

## What is in the repo

| Path | What it holds |
|---|---|
| `topopt/` | The engine and the command `topopt` (`cli.py`). Sub-folders: `dashboard/` (browser files; the server is `dashboard_server.py`), `refine/`, `verify/`, `tet/` (gmsh mesher), `gpu_mg/` (GPU solver), `native/` (C shim for Apple Accelerate). |
| `topopt_mcp/` | The MCP server `topopt-mcp` (`server.py`). `roles/*.md` are the five role texts that the tool `team_guide` returns. |
| `.claude-plugin/`, `skills/`, `.claude/skills/`, `agents/` | The Claude Code plugin. The skill sources are `skills/start-here/` and `.claude/skills/part-to-topopt/`. `agents/*.md` are the five role sub-agents for users. |
| `.claude/` | `settings.json` (the session-start hook), `agents/` (the two roles above) and, in the maintainers' clone, the skill `board`. The plugin does not use them. |
| `.agents/skills/`, `.codex/`, `.cursor/` | Files for other clients. `.agents/skills/` holds copies of the two skills. |
| `.mcp.json` | The MCP server `topopt` for work in a clone. It starts `.venv/bin/python -m topopt_mcp.server`, so it needs the dev install. The plugin does not use it (`docs/clients.md`). |
| `scripts/` | `topopt-env.sh` (the launcher), `ci/` (the CI steps and `legs.toml`), `dev/session_mode.py` (the session-start hook), `sync_agent_skills.py`. |
| `.github/` | The workflows, the issue form and the PR template. |
| `ci/` | The golden file of the short run. |
| `tests/` | pytest tests, one `test_<area>.py` for each area. |
| `examples/`, `recordings/` | Example problems; one recorded run for the replay mode. |
| `tools/`, `bench/`, `packaging/mcpb/` | Helper scripts, solver benchmarks, and the builder of the Claude Desktop extension. None of them is in the wheel. |
| `docs/` | Data files and measurements, and dated plans and studies when they are present. When a plan and the code differ, the code, `CONTRACTS.md` and the README win. |

`CONTRACTS.md` is the spec: the problem YAML (section 2), the CLI (8), the run folder, the snapshots and the result files (5), the post-process (9),
the MCP tools (10), the CI and the golden file (27).
Change the section of a behavior in the same change as the code. Change the README too when a user command or an install step changes.

## Set up

Python 3.13 (the Python that the launcher makes and the Python of every CI leg; tested on Linux x86_64 and on a Mac with Apple silicon). You need `uv`. The steps, the extras and the Mac steps are in `docs/reference/install.md` ("Install", "Extras"). A full set for Linux x86_64:

```
uv venv --python 3.13
uv pip install -e ".[dev,mcp,pardiso,tet,verify]"
source .venv/bin/activate
```

- On Linux, the gmsh wheel needs system libraries (`docs/reference/print-step-tet.md`, "Tet mesher"). This command shows the state of gmsh, `ok`, `missing` or `load_failed`:
  `.venv/bin/python -c "import topopt.tet as t; p = t.probe(); print(p.state, p.detail)"`.
- `.venv/bin/python -m topopt doctor` shows what is wrong with an install and the command that fixes it. It changes nothing.
- `scripts/topopt-env.sh` runs under the macOS `/bin/sh` and under dash, so it has no bash features. Its stdout is the MCP channel: it writes to stderr
  (`sync`, `python`, `topopt` and `doctor` are the exceptions).

## Tests

```
.venv/bin/python -m pytest tests/test_cli.py -q    # the files you touch, while you work
.venv/bin/python -m pytest -n auto                 # the full suite, in parallel (`docs/reference/tests-and-limits.md`, "Tests")
```

- Before you open a PR, run the targeted tests (the files that cover the code you changed) and the contract set:
  `tests/test_levers.py tests/test_mcp.py tests/test_cli.py tests/test_client_configs.py tests/test_plugin_manifest.py tests/test_launcher_snapshot.py`.
  These six files check the public surface: the YAML keys, the MCP tools and texts, the CLI, the client configs, the plugin files and the stored
  copy of the MCP lists that the launcher serves (a changed tool needs `python -P -m topopt_mcp.snapshot`).
- The full suite runs once for a batch of merges, by a maintainer (locally or with CI). A red `main` gets a fix PR first.
- Each worker of `-n auto` needs about 0.65 GB. On a 16 GB machine, use `-n 4`.
- Markers: `slow` (deselect with `-m "not slow"`), `equivalence`, `opencl_context`. `tests/helpers.py` has `needs_accelerate`, `needs_pardiso`,
  `needs_direct_solver`, `needs_gmsh`, `needs_skfem` and `needs_cholespy`. For a gmsh test, use `needs_gmsh`, not `pytest.importorskip("gmsh")`:
  the import raises `OSError` when a system library is missing.
- CI. The workflow files are in `.github/workflows/`. The steps are scripts in `scripts/ci/`.
  `scripts/ci/legs.toml` holds the facts of each platform, the skips included (`CONTRACTS.md` section 27). To run the steps of a leg (`macos-arm64` on a Mac):

```
.venv/bin/python scripts/ci/run_pytest.py --leg linux-x86_64 --workers 4   # writes artifacts/junit.xml; the default is one worker for each CPU
.venv/bin/python scripts/ci/check_skips.py --leg linux-x86_64 --junit artifacts/junit.xml
.venv/bin/python scripts/ci/license_check.py --leg linux-x86_64
```

- A leg skips only the tests that an `allowed_skip` entry of its table names. A skip with no entry fails the leg, and the gmsh tests have no entry on any leg.
  A new skip needs an entry with a `reason` and a `why`.

## Keep files in step

| You change | Also change |
|---|---|
| A YAML key, a CLI option, an MCP tool, an output file | Its section in `CONTRACTS.md` |
| A role text in `topopt_mcp/roles/<id>.md` | The body of `agents/<id>-engineer.md`: it is a copy (`tests/test_team_guide.py`) |
| A skill source (`skills/start-here/`, `.claude/skills/part-to-topopt/`) | The copy in `.agents/skills/`: run `python3 scripts/sync_agent_skills.py` (`--check` only compares) |
| A pin in `pyproject.toml` | Both lock files, and the row in `THIRD_PARTY_NOTICES.md` (`tests/test_locks.py`, `tests/test_third_party_notices.py`) |
| A data file in a package | `[tool.setuptools.package-data]` in `pyproject.toml`, and `DATA_FILES` in `tests/test_packaging.py` |
| `scripts/ci/legs.toml`, a workflow file, the golden file | `CONTRACTS.md` section 27 (`tests/test_ci_scripts.py` checks that they agree) |
| `scripts/dev/clean-room.sh`, `scripts/dev/clean-room-docker.sh`, `docker/clean-room/Dockerfile` | "Clean-room helper" in `docs/reference/install.md`, section 1 of `docs/install-test-checklist.md`, `tests/test_clean_room.py` |
| A tool, a prompt, a resource or the instructions of the MCP server (its lists or its text) | The snapshot that the launcher serves on a first start: run `python -P -m topopt_mcp.snapshot` (`tests/test_launcher_snapshot.py` fails until you do) |
| The text `INSTRUCTIONS` of the MCP server | Keep all of it within 2,048 characters (`OLD_CLIENT_CUT` in `tests/test_plugin_manifest.py`: Claude Code before v2.1.296 cuts it there). A rule for one tool goes into the description of that tool, into the reply key guide of the tool (`topopt://replies/<tool>`) or into the start-here skill, and `MOVED_RULES` of that file finds it there |

## Platforms

- Supported: macOS on Apple silicon, and Linux x86_64. Keep both working. `docs/reference/install.md`, "Platforms", has the state of each platform.
- A solver that needs Intel MKL (for example `pypardiso`) is allowed only as an optional extra on Linux x86_64. It is never a main dependency, and the repo never includes it.

## Memory and time

- Run one multi-GB job at a time: a run with a small voxel size, `topopt verify`, or the full suite with many workers. A whole-domain grid at 0.5 mm voxel size crashed a 16 GB Mac.
- `topopt check <file.yaml>` starts no optimizer run (with `auto` it runs the short self-test of gpu-mg in a child process, at most 60 s: seconds the first time on a device and driver, then the cached pass). It prints the grid, the free dofs, the time estimate and the memory estimate (a rough model, `estimate_memory_bytes` in `topopt/cli.py`).
  The estimate warns when it is above the RAM, and it stops nothing. New code that allocates in proportion to the grid or the dofs adds its term to that model.
- Two voxel guards stop a run (`topopt/voxelize.py`): `settings.max_grid_voxels` (default 5,000,000) is checked before the grid exists, and `settings.max_active_voxels` (default 100,000) after the labels.
  `build_regions` stops when the grid alone, or the full memory estimate (with the filter), is above the RAM, unless the spec has `settings.allow_large_grid: true`
  (`topopt/regions_builder.py`). `docs/reference/tests-and-limits.md`, "Limits and guards", has the rest.

## Show that a change does not move the results

```
.venv/bin/python scripts/ci/short_run.py run --leg linux-x86_64    # about 20 s; use macos-arm64 on a Mac
.venv/bin/python -m pytest -m equivalence                           # the tests that compare with records of an earlier tool
```

- `short_run.py` runs `examples/hook/hook_short.yaml` with one thread and compares four metrics with `ci/golden/hook_short.json`. Tier `bit` allows no difference (same machine, pins and thread count).
  Tier `cross` allows 1e-8 relative on the compliance (another CPU, system or direct solver). The optimizer is not bit-exact between runs with more than one thread.
- The tests with the marker `equivalence` compare the results with records of an earlier tool for one part (fixtures made from its source
  lines), and check the CI machinery of the golden file.
- A change that moves the result on purpose: make a new golden file on Linux with PARDISO (`python scripts/ci/short_run.py make-golden`) and say why in the PR (`CONTRACTS.md` sections 27.4 and 27.9).

## Writing rules

- Write docs, comments, commit messages and issues in English. Use ASD-STE100 Simplified Technical English: short sentences, one idea in each, active voice, simple words.
- Commit messages have the form `type(scope): subject`, for example `fix(cli): name the file in the error`.
- Public text has no paths of your machine, no e-mail addresses (use the GitHub noreply address for your commits) and no internal team role names.
  Read a branch, a PR body and a comment for such facts before you publish them.
- The PR title is one line of the release notes. Write it as a sentence for a user of the tool: no item code (`F4-47`), no name, no commit detail. The merge commit carries the title (`RELEASING.md`, "The release notes").
- Text that the agents of users read (`skills/`, `.claude/skills/part-to-topopt/`, `agents/`, `topopt_mcp/roles/`, the MCP server instructions and tool descriptions) gives facts, estimates and options. The agent decides.
  It has no habitual "must" or "never". Tests check this for several of these files.
- A message says what is wrong, gives the value, and names the next command or option. Example (`topopt/voxelize.py`): `model has <n> active voxels, the limit is <limit> (max_active_voxels).
  Try voxel_size <h> or more, or raise settings.max_active_voxels in the YAML if this machine has the memory`.

## Licenses

- The repo is MIT (`LICENSE`). `THIRD_PARTY_NOTICES.md` has a row for each declared package, with the pin and the license, and it lists the third-party files in the repo.
  When you change a pin, read the license fields of the new version and update the row.
- A copyleft or proprietary package is allowed only as an optional extra that the user installs. Never vendor it. The core does not import `gmsh`, `skfem` or `cholespy` (a test checks it),
  and loads `pypardiso` only when it is installed. No core dependency is copyleft or proprietary.
  Each such package has an entry with its decision in `scripts/ci/license_policy.toml`, and `license_check.py` fails without it (`CONTRACTS.md` section 27.7).
- Code copied from another project: record its source and license in `THIRD_PARTY_NOTICES.md`, section "Third-party files in this repository".
  If the license of some code is not clear, ask in the issue before you copy it.
