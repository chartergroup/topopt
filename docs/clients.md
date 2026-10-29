# Other agent clients

The topopt MCP server is the same for every client. Claude Code (the plugin) and Claude Desktop (the extension) are
described in the README. This page has the tips for Cursor, Codex CLI, VS Code agent mode and Gemini CLI.

**Start with the guide for any MCP client: [reference/any-mcp-client.md](reference/any-mcp-client.md).** It has the setup, a JSON and a TOML config template,
the typical problems and a few lines per client. This page keeps the shipped configs, the snippets and the quirks of each client.

- The steps for a person who tests the install in a client: [install-test-checklist.md](install-test-checklist.md).
- The contract: `CONTRACTS.md` section 34.

## Status

| Client | Config | Where it lives | What was tested |
|---|---|---|---|
| Cursor | `.cursor/mcp.json`, and a user entry in `~/.cursor/mcp.json` | committed in the repo; the user entry below | A test starts the server with the exact launch of this file (a generic MCP client). Mac test with Cursor 3.27.22 (2026-10-27): the project server stayed Disabled, the user entry worked. |
| Codex CLI | `.codex/config.toml` | committed in the repo | The same test. Not tested in Codex. |
| VS Code agent mode | `.vscode/mcp.json` | snippet below | **Not tested by a human.** A test starts the server with the launch of the snippet. |
| Gemini CLI | `.gemini/settings.json` | snippet below | **Not tested by a human.** A test starts the server with the launch of the snippet. |

Of these four clients, only Cursor has started the topopt server and called a tool: one Mac test (2026-10-27, see "Cursor").
The tests start the server the way the config says and talk to it with the `mcp` SDK. They show that the config is valid and that the launch works. They do not show how a client
behaves. The facts about each client come from its docs, forum posts and source code, read on 2026-10-20. Each fact
comes from one of these sources. The official docs of each client were read again on 2026-10-25, as an analysis of publicly
available information. What changed is in the sections of the clients and in the table "Logs, restart and approval" below. A client can change
its config format. Check the docs of your version.

## What every client needs

- No tool to install first. topopt uses your uv if you have one (uv 0.6.8 or newer); else the setup downloads a pinned uv release and checks its checksum. The launcher gets Python 3.13 itself.
  If that download does not work (offline, or a platform with no pinned uv), install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`).
  If your uv is older than 0.6.8, update it instead with the tool that installed it (`brew upgrade uv`, `pip install --upgrade uv`, or `uv self update`):
  a new uv from the install script comes after the old one on the PATH and does not help.
- A checkout of this repo. The configs start `scripts/topopt-env.sh mcp` in it. This is the same launch as the Claude Code
  plugin. The environment is in `<repo>/.plugin-env/` (the variable `TOPOPT_ENV_DIR` changes the folder).
  A checkout can hold up to three environments, each about 500 MB (Linux, 2026-10-27): `.venv` (the dev install in [Install](reference/install.md#install), used by
  `.mcp.json`), `.plugin-env/` (these client configs) and the plugin data folder of Claude Code. The uv cache adds
  the downloaded files (about 490 MB). uv links those files into the environment, so one environment, the cache and the Python of uv take about 625 MB together (`du -shc`, Linux, 2026-10-27).
- A trust step, once for each project. Every client ignores a project config file until you trust the project (VS Code
  needs only Workspace Trust). Claude Code also asks you to approve the project server of `.mcp.json`. In an untrusted
  project the file is ignored and the topopt tools do not show.
- The tools: 32, and the longest name has 17 characters (`printer_materials`). The tools are `run_topopt`, `run_status`,
  `run_result`, `stop_run`, `render_part`, `list_runs`, `run_history`, `compare_runs`, `marks_to_hints`, `apply_marks`, `check_problem`,
  `verify_run`, `verify_status`, `ratchet_run`, `ratchet_status`, `ratchet_approve`, `cleanup_list`, `build_regions`, `record_brief`,
  `case_status`, `environment`, `save_preferences`, `find_printer`, `save_printer`, `find_material`, `save_material`, `printer_materials`, `print_check`, `export_step`, `team_guide`, `find_features` and `show_problem`. The tool limits that the clients document are above
  these numbers (see "Limits").

## Cursor

File: `.cursor/mcp.json` (committed). Cursor also reads `~/.cursor/mcp.json` for every project. The project file wins for the same server name.

<!-- config: .cursor/mcp.json -->
```json
{
  "mcpServers": {
    "topopt": {
      "type": "stdio",
      "command": "sh",
      "args": ["${workspaceFolder}/scripts/topopt-env.sh", "mcp"],
      "env": { "TOPOPT_WORKSPACE": "${workspaceFolder}" }
    }
  }
}
```

**If the project server stays Disabled, use a user entry.** A Mac test ran Cursor 3.27.22 on 2026-10-27, with release v0.4.0
and the checkout in a git worktree. The project server `topopt` of `.cursor/mcp.json` stayed Disabled in Customize. It went back
to Disabled each time it was turned on, while MCP Logs showed a clean start. The agent had no topopt tools, and it ran its checks
in the terminal with the Python of the setup instead. A user entry worked: Customize showed the 31 tools, and a new Agent chat
called `environment` and got the real result.
Add the entry `topopt-user` under `mcpServers` in `~/.cursor/mcp.json`, with the full path of your checkout (a new file takes the
whole block). The entry has a name of its own, because the project file wins for the same name. Then turn the project server
`topopt` off in Customize, else the tools can show twice, and restart Cursor.

<!-- config: ~/.cursor/mcp.json -->
```json
{
  "mcpServers": {
    "topopt-user": {
      "type": "stdio",
      "command": "sh",
      "args": ["/full/path/to/topopt/scripts/topopt-env.sh", "mcp"],
      "env": { "TOPOPT_WORKSPACE": "/full/path/to/topopt" }
    }
  }
}
```

A user entry serves every folder that you open in Cursor. `TOPOPT_WORKSPACE` is the work folder: the runs go there (here
`<checkout>/runs/`), and relative YAML paths start there. To work in another folder, put that folder in `TOPOPT_WORKSPACE`.
Why the project server stays Disabled is not known. Cursor staff say on the forum that Cursor keeps the on/off state and the
approval of a project server for each workspace path, outside `mcp.json`. They also say that a server of `~/.cursor/mcp.json`
needs no approval for each project (forum.cursor.com, topics 159989 and 167777, read on 2026-10-27).
Not checked: whether the project server stays on in a plain clone.
Not checked in Cursor yet: the setup status during a first start, `check_problem` on the hook, whether the skills show once or
twice, and `team_guide`.

Tips:

- Settings: Customize > MCPs (a toggle for each server). Logs: Output panel (Cmd+Shift+U), "MCP Logs". After you edit `mcp.json`, restart Cursor.
- Trust: the Cursor CLI refuses a project server that you did not approve: `agent mcp enable topopt`. The IDE step is not documented.
- Working folder: Cursor does not start the server in the workspace folder. Use `${workspaceFolder}` in `args` and set `TOPOPT_WORKSPACE`.
- First launch: no start timeout key was found. A cold first start took 63 to 77 s (2026-10-20): run `sync` first, or read the status text of the first call.
- Skills: Cursor reads `.agents/skills/` and `.claude/skills/`, so it can list `part-to-topopt` twice (the copies are equal). Roles: see "Roles without sub-agents".

## Codex CLI

File: `.codex/config.toml` (committed). Codex also reads `~/.codex/config.toml`.

<!-- config: .codex/config.toml -->
```toml
[mcp_servers.topopt]
command = "sh"
args = ["scripts/topopt-env.sh", "mcp"]
startup_timeout_sec = 240
tool_timeout_sec = 300

[mcp_servers.topopt.env]
TOPOPT_MCP_WAIT = "180"
```

- Trust: Codex ignores `.codex/config.toml` in a project that you did not trust.
- Working folder: Codex has no workspace variable. The server starts in the folder where Codex runs, so start Codex in the repo root (the file uses a relative path). In a config of your own, use full paths (guide, part 5).
- Timeouts: the Codex docs say 10 s for the start and 60 s for a tool call; the Codex code (tag `rust-v0.163.0`) says 30 s and 300 s. The file sets both. `TOPOPT_MCP_WAIT` (180) stays shorter than `startup_timeout_sec` (a test checks it).
- Skills: Codex reads `.agents/skills/`, not `.claude/skills/`. Check: `codex mcp list`, or `/mcp` in the session.

## VS Code agent mode (GitHub Copilot)

**Not tested by a human.** File: `.vscode/mcp.json` (not committed; the top key is `servers`).

<!-- config: .vscode/mcp.json -->
```json
{
  "servers": {
    "topopt": {
      "type": "stdio",
      "command": "sh",
      "args": ["${workspaceFolder}/scripts/topopt-env.sh", "mcp"],
      "env": { "TOPOPT_WORKSPACE": "${workspaceFolder}" }
    }
  }
}
```

- Trust: Workspace Trust only.
- Two servers: the committed `.mcp.json` is also a VS Code config (it starts `.venv/bin/python -m topopt_mcp.server`, so it needs the dev install in [Install](reference/install.md#install)). With the snippet in `.vscode/mcp.json`, VS Code has two servers named `topopt`. Without the dev install the `.mcp.json` one fails. With it, the tools show twice (64). Claude Code shows one `topopt` only, because the plugin entry wins there; VS Code has no such rule. Options: use only the committed `.mcp.json` (after the dev install), or only the snippet (or give the snippet another name).
- Limit: a chat request can have 128 tools at most. topopt has 32 (64 when both servers run).
- Control: `MCP: List Servers` gives Start, Stop, Restart and Show Output. `MCP: Reset Trust` resets the MCP trust decisions.

## Gemini CLI

**Not tested by a human.** File: `.gemini/settings.json` (not committed). The README of Gemini CLI says that Antigravity CLI replaced
Gemini CLI for unpaid-tier and Google One users on 2026-06-18. The config of Antigravity CLI was not researched.

<!-- config: .gemini/settings.json -->
```json
{
  "mcpServers": {
    "topopt": { "command": "sh", "args": ["scripts/topopt-env.sh", "mcp"] }
  }
}
```

- Trust: Gemini ignores `.gemini/settings.json` in a folder that you did not trust.
- Working folder: Gemini has no workspace variable. The relative path works only when Gemini starts in the repo root.
- First launch: the key `timeout` (ms) is 600,000 by default. Check: `/mcp`.

## Roles without sub-agents

The five roles (`lead-design`, `cad`, `fea`, `print`, `finish`) are sub-agents in the Claude Code plugin. A client that has no such
sub-agents gets the same texts from the tool `team_guide`:

- `team_guide` with no role lists the five roles.
- `team_guide` with `role` set to `lead-design`, `cad`, `fea`, `print` or `finish` returns the playbook of that role.

One agent can read the roles that it needs and do the work itself. The skill `start-here` says this.
The same texts are MCP prompts and MCP resources (`topopt://roles/<id>`). A client that lists them can use them. The tool
is the way that every client has.

Option for Cursor and VS Code: both read custom agents in the Claude format from `.claude/agents/*.md`. You can copy
`agents/*.md` to `.claude/agents/` in your own work folder. This repo does not commit that copy: Claude Code with the plugin
would show each role twice. Codex needs a TOML file for each agent (`.codex/agents/`, with the key `developer_instructions`),
which is another format. A generator for it is not part of this work.

## Skills

The skill `start-here` has its source in `skills/start-here/`. The skill `part-to-topopt` has its source in
`.claude/skills/part-to-topopt/`. Codex, Gemini, Cursor and VS Code read `.agents/skills/`, so the folder `.agents/skills/` has a
copy of each skill (the `SKILL.md` and the files that it names). The copies are not symlinks: only Codex documents symlinks,
and some tools refuse them. A test (`tests/test_client_configs.py`) checks that each copy equals its source, byte for byte.
Edit the source, then make the copies with one command from the repo root:

    python3 scripts/sync_agent_skills.py            # copy each source over its copy
    python3 scripts/sync_agent_skills.py --check    # change nothing; exit 1 if a copy differs

The test and the command use the same check. Both count the files that git tracks, so a `.DS_Store` does not count.
The skill `part-to-topopt` names its example files "in the folder of this skill" (the Agent Skills rule), so the text is
right in both copies.

Cursor and VS Code read both `.claude/skills/` and `.agents/skills/`, so they can list `part-to-topopt` twice. The two are equal.

## Limits

Facts from the docs, the forum and the code of each client, read on 2026-10-20.

| Client | Start timeout | Tool call timeout | Tool count | Name length |
|---|---|---|---|---|
| Cursor | not documented; about 60 s (staff post) | not documented | 40 tools (forum posts of 2025); not in the docs | 60 characters for server name plus tool name (staff posts, 2026); a longer name only gives a warning |
| Codex CLI | `startup_timeout_sec`; docs 10 s, code 30 s | `tool_timeout_sec`; docs 60 s, code 300 s | no limit found | 128 bytes after cleaning |
| VS Code | no constant found | not found | 128 tools for each chat request | no limit found |
| Gemini CLI | `timeout`, 600,000 ms | the same `timeout` | no limit found | 63 characters; a longer name is cut in the middle |

topopt: 32 tools, longest name 17 characters (`printer_materials`). The longest name with the server name is 23 characters in Cursor
(`topopt` plus `printer_materials`; 28 with the user entry `topopt-user`) and 28 in Gemini CLI (`mcp_topopt_printer_materials`). Cursor forum posts name a limit of 40 tools
(not in the docs): 32 is under it. No client limit applies today.
A tool call of `check_problem` takes about 3 s on a warm start. `run_topopt` returns at once. `export_step` holds the call until the
file is written: 12 to 39 s on the public hooks, and in two runs on Linux 102 s (530k triangles) and 613 s (925k triangles).
The longest is above the tool call timeout of Codex CLI (300 s in the code) and Gemini CLI (600 s). `estimate_only` gives the time
first; `topopt step` in a shell has no client timeout.
If a limit applies later (more tools, a longer name), the options are to shorten the name, to turn tools off in the client
(Codex `disabled_tools`, Gemini `excludeTools`), or to split the server.

## Logs, restart and approval

Read on 2026-10-25 in the official docs of each client (an analysis of publicly available information). The Claude Code and
Claude Desktop rows are here too, because a tester meets the same questions there.

| Client | Where the server log is | How to restart or reconnect | Approval of a project server |
|---|---|---|---|
| Claude Code (plugin) | `/mcp`, the detail view of the server (the `Issue:` row; the launcher does not fail a first start that is not ready: a tool call then returns the setup status as a tool result); `claude mcp list` | `/mcp`, then Reconnect. `/mcp reconnect all` from v2.1.284. A plugin server that failed less than 15 minutes ago is skipped at the next start. | None for a plugin server. It starts when the plugin is enabled. |
| Claude Desktop (extension) | Settings, the Extensions page, section Local MCP servers; `~/Library/Logs/Claude/` (from reading) | Turn the extension off and on, or quit the app (Cmd+Q) and open it again | You install the file and confirm in the dialog |
| Cursor | Output panel, "MCP Logs" | Restart Cursor after you edit `mcp.json`; the toggle in Customize > MCPs | CLI: `agent mcp enable`. IDE: not documented. |
| Codex CLI | The terminal; `log_dir` and `codex-tui.log` | Start Codex again; `/mcp` shows the servers | Trust the project |
| VS Code | `MCP: List Servers`, then Show Output | `MCP: List Servers`, then Restart Server | Workspace Trust; `MCP: Reset Trust` |
| Gemini CLI | The CLI output and the stderr of the server; `/mcp` | Start Gemini again | Trust the folder |

## More

Typical problems and the generic steps: [reference/any-mcp-client.md](reference/any-mcp-client.md#6-typical-problems).
