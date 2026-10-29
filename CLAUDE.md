<!-- By default, Claude Code reads AGENTS.md only when CLAUDE.md is missing (older versions never read it). The text is in AGENTS.md. The next line loads it. -->
@AGENTS.md

## Claude Code only

- At session start a hook (`.claude/settings.json`) runs `scripts/dev/session_mode.py`. It prints `Mode: maintainer`, `Mode: maintainer (knowledge base not found)` or `Mode: contributor`. Work in that mode (AGENTS.md, entry 3).
  If the line is not in your context, run the script yourself.
- The roles for development are in `.claude/agents/`. `dev-implementer` (sonnet) takes one issue and makes the change with tests. `dev-reviewer` (opus) reviews a diff that it did not write.
  `dev-clerk` (haiku) does the board mechanics, in maintainer mode only. Maintainers have this agent and the skill `board`; a public clone does not. `CONTRIBUTING.md` tells when to use them.
- `agents/`, `skills/` and `.claude/skills/part-to-topopt/` are for people who use topopt. They are not rules for you. The skill `board` is for maintainers only; a public clone does not have it.

## Commit messages and releases

Releases are made by python-semantic-release from the commit messages on `main` (`RELEASING.md`).
Write every commit message, and every PR title, as `type(scope): subject`.

- `feat` -> minor release. `fix` and `perf` -> patch release.
- `docs`, `test`, `chore`, `ci`, `refactor`, `build`, `style` -> no release. A message with no type (`verify: ...`) also makes no release.
- A breaking change: `type(scope)!: subject`, or a footer line `BREAKING CHANGE: what breaks`. While the version is 0.x it raises the minor number.
- The PR title becomes a commit message on `main`: the publish step uses it as the message of the branch commit, and a squash merge uses it too. So the title decides the release: use `feat` or `fix` only for a change that a user of the tool sees. A merge note (`chore(merge): ...`) and a review fix of an open PR are `chore` or `test`.
- The subject is short, imperative, lower case, no final dot. Details go in the body.
- The PR title is also one line of the release notes (`RELEASING.md`, "The release notes"). Write the subject as a sentence for a user of the tool: no item code (`F4-47`), no name of a person.
