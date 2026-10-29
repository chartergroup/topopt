# .agents

The folders in `.agents/skills/` are copies. Codex CLI, Gemini CLI, Cursor and VS Code read skills from `.agents/skills/`.
Claude Code does not read this folder. It reads `skills/` (plugin) and `.claude/skills/`.

- `.agents/skills/start-here/` is a copy of `skills/start-here/` (the source).
- `.agents/skills/part-to-topopt/` is a copy of `.claude/skills/part-to-topopt/` (the source).

Edit the source. Then make the copies with one command from the repo root:

    python3 scripts/sync_agent_skills.py            # copy each source over its copy
    python3 scripts/sync_agent_skills.py --check    # change nothing; exit 1 if a copy differs

Run it after each change of a source skill, and after a merge that changes one.
`tests/test_client_configs.py` runs the same check, so a copy that is not current fails the test.
The check counts the files that git tracks. A `.DS_Store` or a `__pycache__` does not count.
