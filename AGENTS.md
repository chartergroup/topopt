# AGENTS.md

topopt is a general 3D topology optimizer. A problem YAML and STL region files go in, and a printable STL comes out.
It has a command `topopt`, a browser dashboard that shows the run, and an MCP server (`topopt`; the command is `topopt-mcp`) that lets an agent start and watch runs.

Find your task below. The first entry is for a person who wants a part. The other two are for a person who wants to change this repo.

## 1. Use topopt (make a part)

- Start with the skill `start-here`. In the Claude Code plugin it is `skills/start-here/SKILL.md`. Other clients read the copy `.agents/skills/start-here/SKILL.md`.
  If your client did not load it as a skill, read the file. It leads the user from a part to a printable STL, and it links the other steps.
- The user describes a part in words: use the skill `part-to-topopt` (`.claude/skills/part-to-topopt/SKILL.md`; the copy is in `.agents/skills/part-to-topopt/`).
  The user has STL files: write the problem YAML (`CONTRACTS.md` section 2).
- The tools are in the MCP server `topopt`. README ("Other agent clients"), `docs/reference/any-mcp-client.md` (a guide for any client) and `docs/clients.md` tell how each client gets it. Without the server, use the commands of `docs/reference/problem-and-run.md`, section "Quick start".
- Run `check_problem` (or `topopt check <file.yaml>`) before a run. It starts no optimizer run (with `auto` it runs the short self-test of gpu-mg in a child process, at most 60 s: seconds the first time on a device and driver, then the cached pass), and it gives the time and the memory estimate. Run one heavy job at a time.
- Run files go in `runs/` of the work folder. A part is solved with the tools and the YAML, not with changes to the package code. If a tool lacks a feature that the part needs, tell the user.
- If the tools do not show, or an install is wrong: `sh scripts/topopt-env.sh doctor` (`docs/reference/install.md`, "Check your machine").

## 2. Contribute (change this repo)

Anyone can contribute. Read `CONTRIBUTING.md`. In short: an issue first (Goal, Why, Acceptance, Evidence), a branch, tests, a PR with `Closes #N`,
a review by someone who did not build the change, and a person merges. A finding that is outside your issue goes into the "New cards" block of your PR. Do not fix it in the same PR.

## 3. Maintain (the owners)

The maintainers are the members of the GitHub organization `chartergroup`. They follow `CONTRIBUTING.md` too. They also have the private knowledge base (KB) `chartergroup/topopt-kb` and the project board.

Find your mode before you plan or change code:

- Claude Code prints the mode at session start. A hook does it (`.claude/settings.json`).
- A client without hooks (Codex, Cursor, Gemini CLI, VS Code), or a session with no Mode line in its context: run `python3 scripts/dev/session_mode.py` from the repo root, and read what it prints.

The script gives one of three results:

1. `Mode: maintainer. Knowledge base: <path> ...` The script found the KB on this machine. It made no network call. Read `<kb>/maintainer/START.md` first.
2. `Mode: maintainer (knowledge base not found).` The script did not find the KB, and `gh` says that you are an active member of the organization. This is the only network call. It runs only when the KB is not found,
   and it stops after 10 seconds. The next line tells you to clone the KB next to this repo and to start a new session after the clone. Until then, work as a contributor and do not touch the board.
3. `Mode: contributor.` The script did not find the KB, and `gh` is missing, not logged in, too slow, or does not show you as a member. Follow `CONTRIBUTING.md`. Do not look for the KB, and do not write to the board.
   The KB and the board are for maintainers only. The last line says what a member of the organization does: clone the KB next to this repo
   (`gh repo clone chartergroup/topopt-kb ../topopt-kb`) or set `TOPOPT_KB`, then start a new session. A client without hooks runs the script again after the clone.

The script looks for the KB in this order. The first folder that has a readable `maintainer/START.md` wins:

1. The environment variable `TOPOPT_KB` names the folder of the KB clone.
2. The KB is cloned next to this repo, in the folder `../topopt-kb`.
3. The file `~/.config/topopt/kb-path` holds the folder in its first line (`$XDG_CONFIG_HOME/topopt/kb-path` if that variable is set).
