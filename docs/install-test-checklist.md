# Install test checklist: six clients

This page is for the install tester. The tester is a person who follows the install steps of the README as a new user does, in one
client, and reports what happened. You need no knowledge of the code.

The facts about each client on this page come from an analysis of publicly available information: the official docs of each
client, read on 2026-10-25 (section 9 has the URLs). A client can change its docs or its menus. If a menu name differs on your
computer, write down the name that you see. That is a result too.

The tests of the repo start the server
the way each config says, with a generic MCP client. They do not show how a client behaves. Your run does.

## 1. Before you start (all clients)

Write down these items first.

| Item | Value |
|---|---|
| Date and time (UTC) | |
| Computer, CPU and OS version | |
| Free disk space (you need about 1.5 GB) | |
| Client name and version | |
| `uv --version` (if uv is there) | |
| Size of the uv cache before you start (`du -sh "$(uv cache dir)"`) and of the uv Python folder (`du -sh "$(uv python dir)"`) | |
| Clean state: `clean-room.sh` (`hide`, `shell`, Docker, a new macOS account) or none | |
| Repo: the tag or the commit (`git describe --tags` or `git rev-parse --short HEAD`) | |
| Speed of your link (a rough guess is enough: fast, slow) | |

Rules for every client:

1. A cold first start means: no uv cache and no environment. If uv was used on this computer before, say so. Your times are then
   for a warm cache.
2. Start a clock at each step and write down the time. The launcher prints "ready after N s" when the setup ends.
3. Never stop a step because it is slow. The first start downloads Python 3.13 and about 200 MB of packages. It takes
   1 to 3 minutes in the runs of 2026-10-20. On a slower link (about 0.7 MiB per second) it took 4 minutes 43 seconds
   (measured on 2026-10-25 on Linux).
4. If something fails, copy the whole message. Then run the doctor and copy its output:

   ```
   sh scripts/topopt-env.sh doctor
   ```

   Run it in the repo folder. In the Claude Code plugin, `/topopt:start-here` runs the doctor for you and prints the path of the
   setup log. The setup log is `setup.log` in the environment folder. The doctor names that folder.
5. The command `sh scripts/topopt-env.sh sync` makes the environment in the terminal and waits until it is ready. `status` says if it is
   ready. `doctor` changes nothing. Do not run the case `mcp` by hand: it starts the server and waits for a client.
6. Start from a clean state, so that you see what a new user sees. `sh scripts/dev/clean-room.sh status` says what is there. `hide` (dry run) and `hide --apply` move it away
   into a dated backup folder, and `restore --apply` puts it back after the test. **Restore before the next hide**: `hide --apply` refuses while a backup is not fully restored, and `restore --apply` exits 1 while it skips anything
   (a new cache or a new uv from your test is in the way: move it aside, then run `restore --apply` again; a config that the client wrote again: `restore --apply --keep-changed-configs`). For a CLI client, `sh scripts/dev/clean-room.sh shell` gives a new-user shell, and
   `sh scripts/dev/clean-room-docker.sh` gives a clean Linux container. For a GUI client on a Mac, use `hide` and `restore`, or a new macOS account. Write down which one you used.
   Details: [Clean-room helper](reference/install.md#clean-room-helper-for-the-install-tests). If you do not use it, answer the uv rows above with what was there before.

Expected times (Linux x86_64, uv 0.12.1, measured on 2026-10-25): `sync` from nothing 283 s on a slower link (63 to 77 s in the
E2 runs of 2026-10-20); `sync` with a warm uv cache and no environment 3 s; `status` with a ready environment under 0.1 s; `doctor`
3 s. Nobody has measured these on a Mac.

## 2. Claude Code plugin

Source: README, section "Quick start: Claude Code plugin" (the `#latest` path), and `docs/reference/install.md`, section "Install the Claude Code plugin" (the pinned forms and the prompt). Claude Code waits 30 s for an MCP server (setting `MCP_TIMEOUT`). The prompt
form (install doc, "The install prompt") sets up the environment before the first start, so `topopt` connects at once. After the other forms the setup
runs during the first start: `topopt` still connects at once with all its tools, and a tool call during the setup returns the setup status.

1. You need no uv: topopt uses your uv if you have one (uv 0.6.8 or newer); else the setup downloads a pinned uv release and checks its checksum. Write down whether `uv` was on your PATH (`command -v uv`). Time: seconds.
   Write down `claude --version`. The one-command forms of the install doc need Claude Code 2.1.275 or later in the chat, and 2.1.292 or later
   in a terminal.
2. Install with the first form of the install doc section "The install prompt", the prompt. Make a work folder, go in, start `claude`, and paste the prompt.
   Write down the permission mode in the footer under the input box (for example `auto mode on`). Expect: Claude runs
   `claude plugin install topopt --marketplace chartergroup/topopt#<tag>`; in a mode that asks, approve it. The output ends with
   `Successfully installed plugin: topopt@<marketplace> (scope: user)`. Then Claude runs `claude plugin list --json` and
   `sh "<plugin folder>/scripts/topopt-env.sh" plugin-sync`. Its last line is `topopt: ready (pins=lock): ...`, after 1 to 3 minutes.
   Then Claude tells you to type `/reload-plugins`. Type it.
   Write down: the exact questions that you approved (if any), the exact text of any command that Claude Code blocked, whether
   Claude ran exactly the commands of the prompt, the folder that it took, the time limit that it gave `plugin-sync` (or whether it
   ran it in the background), any other question, what `/reload-plugins` printed, the time from the paste to the end of the install,
   and the time of `plugin-sync`.
   The other forms, one in each test round (write down which): the chat command `/plugin install topopt --marketplace ...` (write
   down each screen: the marketplace check, what the plugin adds, the scope choice, and whether the plugin loads at once), the
   terminal command, or the two commands for an older Claude Code (`claude plugin marketplace add ...`, then
   `claude plugin install topopt@<marketplace>`; expect `Successfully added marketplace: <marketplace>` first).
   `<marketplace>` is the name in the `marketplace.json` of the tag (`topopt-demo` for `v0.1.0` and `v0.2.0`, `topopt` for a later tag).
   The one-command forms take that name from the tag. Time: a few seconds to 1 minute (write it down).
   If the repo is private, git on your computer needs access to it. Claude Code runs git with prompts turned off.
   It uses SSH if your key works without a passphrase, else HTTPS with a credential that is stored
   already (`gh auth login` and `gh auth setup-git` store one). Write down which one worked and the exact error if none did.
   **2a. The `#latest` path of the README** (a second round, in a clean config folder, instead of the prompt): run
   `claude plugin marketplace add chartergroup/topopt#latest` and `claude plugin install topopt@topopt`, or the session form
   `/plugin marketplace add chartergroup/topopt#latest` and `/plugin install topopt@topopt`. Expect: `Successfully added marketplace: topopt`,
   then `Successfully installed plugin: topopt@topopt (scope: user)`; `claude plugin list` shows the version of the newest release.
   Write down: the screens, the time, whether `/reload-plugins` was needed, and the version. Then follow step 3 to 7.
3. In the same session (after `/reload-plugins`), or in a new `claude` in the work folder, type `/topopt:start-here`.
   Expect after the prompt form: `topopt` is connected in `/mcp` at once, the skill reports the setup as ready, and you skip step 4.
   Expect after the other forms: `topopt` is connected in `/mcp` at once, with all its tools (the launcher answers from a stored copy of
   the lists); a call of a tool returns `topopt is still setting up its Python environment.` with the step and the time; the skill reports
   a setup that is running. Time to "ready": 1 to 3 minutes. In the same session the server starts at `/reload-plugins`, so there "the start" below is that command.
   Write down the time from the start of `claude` to the line "ready".
4. When the setup is ready, call a tool of `topopt` again in the same session. Expect: it works, with no `/mcp` and no reconnect.
   Write down if the list of tools changed, and if the client needed a reconnect (that is a finding). To see if the setup is
   ready, ask the skill (it runs `status` with the two paths of the plugin), or run in a terminal the `status` command that the skill prints:
   `TOPOPT_ENV_DIR="$HOME/.claude/plugins/data/topopt-<marketplace>" sh <plugin folder>/scripts/topopt-env.sh status`
   (`$CLAUDE_CONFIG_DIR` in place of `$HOME/.claude` when it is set). The plugin folder is the folder that Claude Code starts the server
   from. The topopt line of `claude mcp list` shows it: the path in front of `/scripts/topopt-env.sh`. `claude mcp list` starts the server
   to check it; whether a failed start there counts for the 15-minute rule below is not checked, so run it when the setup is ready. For a
   marketplace that you added from a local folder, the plugin folder is that folder, not the copy in `~/.claude/plugins/cache/` (checked
   with Claude Code 2.1.293). A plain `sh scripts/topopt-env.sh status` in a checkout looks at the `.plugin-env` of the checkout, not at
   the environment of the plugin, and says "not ready". `status` from another copy of the plugin says "not ready" too, because the stamp
   of the environment names the folder that made it. Do not run the `sync` command that such a `status` prints: it writes the folder of
   that copy into the stamp, and then the server says "not ready". The paths that the skill uses are not checked yet in a logged-in run:
   if the skill and the terminal command do not agree, write down both.
   Also write down what you see if you quit `claude` and start it again during the setup (an older launcher failed, and Claude
   Code skipped a plugin server that failed less than 15 minutes ago and printed "Skipping connection (recent failure cached; ...)"; the
   new launcher does not fail, so no mark should be written).
5. Ask: "List the topopt tools." Copy the reply. Expect the tools named in section 8. Time: seconds.
6. Ask: "Call environment, then check_problem on examples/hook/hook.yaml." Expect: the platform and the solver backends, then
   `ok` true and a time estimate. Time: about 3 s.
7. Optional: open the dashboard link of the skill (usually `http://127.0.0.1:8765/`).
8. Optional, the update of a `#latest` install: run `claude plugin marketplace update topopt`, then `claude plugin update topopt@topopt`, and write down what they answer (checked on Linux with a local mirror, 2026-10-27: "Restart to apply changes"). Also write down what `claude plugin update topopt@<marketplace>` alone answers on a pinned install. (Claude Code compares only the `version`
   of the plugin, so a release with the same version does not reach you.)

## 3. Claude Desktop extension and plugin upload

Source: `docs/reference/install.md`, section "Claude Desktop" (the README keeps a pointer). The target is a Mac with Apple silicon. You must
be signed in to claude.ai in the app. An organization can turn extensions off. Then Settings shows no Extensions page. Write that down.
If you used `clean-room.sh hide`, open Settings, Extensions after `restore`: an extension from an older `.mcpb` can have another id and sit next to the new one. Keep one.

1. Install uv (`brew install uv`): the build of step 2 needs it. Write down where `which uv` points and what `uv --version` says.
   The extension itself needs no uv (it uses one of 0.6.8 or newer, else it downloads its own).
2. In a checkout of the repo, build the file: `uv run --no-project --python 3.13 packaging/mcpb/build.py`.
   Expect: a line with the path `dist/mcpb/topopt-<version>.mcpb`, about 1.8 MB. Time: under 1 s, or about 10 s when uv fetches
   Python 3.13. The script says that the `mcpb` CLI is not on PATH. That is expected.
3. Open the `.mcpb` file with Claude Desktop, or drag it into Settings, Extensions. In newer builds the page is Settings, Desktop app,
   then Manage next to Extensions. Expect: a preview, then Install, then the optional field "Work folder". Write down the menu names that you see.
4. The first start: at once, start a new chat and ask Claude to call `environment`. Expect during the first 1 to 3 minutes a tool
   result with the setup status (the step, the time and when to call again; in the first seconds: "the setup is starting"), and after
   it the answer of the server in the same chat.
   Write down the time until the answer, and any error that the Extensions page shows.
5. Start a new chat. Ask: "List the topopt tools." Copy the reply. Then ask Claude to call `team_guide` with no role.
   Expect: the five roles `lead-design`, `cad`, `fea`, `print` and `finish`.
6. Ask Claude to call `environment`. Expect: a `work_folder` and a `server.started_utc`.
7. Open the log: Settings, the Extensions page, the section "Local MCP servers" shows the status and the logs. The install doc also names
   `~/Library/Logs/Claude/`. Write down which one exists.
8. The plugin upload (the skills): make the zip as step 6 of that section says, then Customize, Plugins, Add, Upload plugin.
   Expect: the plugin `topopt` with the skills `start-here` and `part-to-topopt`. Write down the size of the zip and whether the
   upload needed the variant with a top folder. (The Claude docs say that both forms work.)
9. Optional, the fast solver: the install doc gives the steps for the Xcode Command Line Tools and the build of the library.

## 4. Cursor

Source: `docs/clients.md`, section "Cursor", and the file `.cursor/mcp.json` (committed).

1. Clone the repo and go in. You need no uv: the setup downloads one if you have none or yours is older than 0.6.8 (write down which case you had). Check that the file `.cursor/mcp.json` is there.
2. Choose: run `sh scripts/topopt-env.sh sync` first (advice of the docs), or do not (a cold first start). Write down which.
   The Cursor docs give no start timeout. Write down how long Cursor waited and what it showed.
3. Open the repo folder in Cursor. Open Customize in the sidebar and look for `topopt` under MCPs.
   Expect: `topopt` is listed. Write down if Cursor asks you to approve or enable it. The docs do not say what the IDE shows. In
   the CLI, `agent mcp list` lists the servers and `agent mcp enable topopt` approves one.
   If `topopt` shows Disabled, or goes back to Disabled after you turn it on (the Mac test with Cursor 3.27.22, 2026-10-27), write
   down when it goes back. Then add the user entry of `docs/clients.md`, section "Cursor", turn the project server off, restart
   Cursor, and do the next steps with `topopt-user`. Write down the path in the entry. The entry stays for every folder that
   you open later: before a test of another checkout, change the path or remove the entry.
4. Expect: the server (`topopt`, or `topopt-user` after the user entry) is on and shows its tools. Copy the tool list. If the server failed, open the Output panel (Cmd+Shift+U),
   select "MCP Logs", and copy the last 20 lines. After you change `mcp.json`, restart Cursor.
5. Open a new Agent chat. Prompt: "Use the topopt tools. Call `environment`. Then call `check_problem` on `examples/hook/hook.yaml` and tell me the time
   estimate and the warnings." Expect: `ok` true and a time estimate. Cursor asks for approval of a tool call by default. (From Cursor 3.6 the default mode Auto-review runs allowlisted
   tools at once. Write down which mode you have.)
6. Skills: ask Cursor to list its skills. Expect `start-here` and `part-to-topopt`. Write down if `part-to-topopt` shows once or twice.
7. Write down: did the first call work at once, or did the server need a restart after the first start? If the agent runs
   topopt in the terminal and not through the tools, write it down: then the tools are not attached (step 3).

## 5. Codex CLI

Source: `docs/clients.md`, section "Codex CLI", and the file `.codex/config.toml` (committed).

1. Clone the repo and go in. You need no uv: the setup downloads one if you have none or yours is older than 0.6.8 (write down which case you had). Start Codex in the repo root: `codex`. The relative path in the config needs this.
2. Trust the project when Codex asks. Write down the question. Without trust, Codex ignores `.codex/config.toml`.
3. List the servers: `codex mcp list` in a shell, or `/mcp` in the Codex session. Expect: `topopt` is listed. Write down the start time if Codex
   shows it. The config gives the start 240 s, and the launcher waits 180 s.
4. Copy the tool list. Codex can give the tools a prefix (for example `mcp__topopt__`).
5. After a cold start, write down if the first turn had the tools. Codex waits only 1 s for an optional server when it builds the first
   tool list. A later turn or a restart can be needed.
6. Prompt: the same as in section 4, step 5. Expect: `ok` true and a time estimate.
7. Skills: Codex reads `.agents/skills/`. Expect `start-here` and `part-to-topopt`, each once.
8. Optional: ask Codex to run `sh scripts/topopt-env.sh sync` itself when the environment is not ready. Write down if Codex asks to run
   it outside the sandbox, or the command fails on the network.

## 6. VS Code

Source: `docs/clients.md`, section "VS Code agent mode (GitHub Copilot)". The snippet is not committed. You make the file.

1. Clone the repo and go in. You need no uv: the setup downloads one if you have none or yours is older than 0.6.8 (write down which case you had). Open the folder in VS Code and trust the workspace (Workspace Trust).
2. Make the file `.vscode/mcp.json` from the snippet in `docs/clients.md`. The repo also has a root `.mcp.json` for the dev install.
   VS Code reads it too, so you may see two servers named `topopt`. Write down what you see.
3. Run `MCP: List Servers` from the Command Palette. Select `topopt`. Expect the actions Start Server, Stop Server, Restart Server and
   Show Output. Start the server. Expect: it runs. Write down if VS Code asks for a separate MCP trust (the docs say it does not in a
   trusted workspace).
4. If the server fails, choose Show Output and copy the last 20 lines.
5. Open chat in agent mode. Ask for the topopt tools and copy the reply. A chat request can have 128 tools at most.
6. Prompt: the same as in section 4, step 5.
7. Write down: the VS Code version, and if the Agent Customizations editor (preview) shows `topopt`.

## 7. Gemini CLI

Source: `docs/clients.md`, section "Gemini CLI". The snippet is not committed. You make the file. The Gemini CLI README says that
Antigravity CLI replaced Gemini CLI for unpaid-tier and Google One users on 2026-06-18. Write down which one you have.

1. Clone the repo and go in. You need no uv: the setup downloads one if you have none or yours is older than 0.6.8 (write down which case you had).
2. Make the file `.gemini/settings.json` from the snippet in `docs/clients.md`. Start `gemini` in the repo root. The relative path in the
   snippet needs this.
3. Trust the folder if Gemini asks. Write down the question. The docs say that trusted folders are off by default.
4. Type `/mcp`. Expect: `topopt` with the status connected and the list of tools. Copy it. The default timeout is 600,000 ms, so a
   cold first start can end.
5. If the status is `DISCONNECTED`, copy the CLI output and the stderr of the server. Then run the doctor.
6. Prompt: the same as in section 4, step 5.
7. Skills: Gemini reads `.gemini/skills/` and `.agents/skills/` in a trusted folder. Expect `start-here` and `part-to-topopt`.

## 8. Send back

Send one filled copy of the table of section 1 for each client that you tested, and these items.

| Item | What to send |
|---|---|
| Client and version, OS | From section 1. |
| The times | Install of uv; add and install (or build); first start to "ready"; the first tool call that works after it; warm start. |
| The first reply of the tool list | Copy it as text. It must contain at least these tools: `run_topopt`, `run_status`, `run_result`, `list_runs`, `check_problem`, `build_regions`, `environment`, `render_part`, `verify_run`, `verify_status` and `team_guide`. The full list is in `docs/clients.md`, section "What every client needs". |
| The log lines on a failure | The doctor output, the last 20 lines of `setup.log`, and the last 20 lines of the log of the client (Cursor: MCP Logs; VS Code: Show Output; Claude Desktop: the Extensions page and `~/Library/Logs/Claude/`; Codex CLI and Gemini CLI: the terminal output). |
| Screenshots | The page that lists the server (Cursor: Customize; VS Code: MCP: List Servers; Claude Desktop: Settings, Extensions; Claude Code: `/mcp`) and any error dialog. |
| The open facts | The trust step (what it asks and how often), the start timeout that you saw, if the first start needed `sync`, how many times `part-to-topopt` shows, and the prefix of the tool names. |
| Any step that failed | The step number, the message, and what you did next. |

## 9. Sources

Based on an analysis of publicly available information. Read on 2026-10-25 (UTC). A page that shows its own date or version has it
in the last column.

| Client | URL | Page date or version |
|---|---|---|
| Claude Code | https://code.claude.com/docs/en/plugins/install.md | The page names Claude Code v2.1.292 as a minimum for one command. |
| Claude Code | https://code.claude.com/docs/en/plugins/host-marketplace.md | None shown. |
| Claude Code | https://code.claude.com/docs/en/plugins-reference | None shown. |
| Claude Code | https://code.claude.com/docs/en/mcp.md | The page names versions up to v2.1.304. |
| Claude Code | https://code.claude.com/docs/en/errors.md | None shown. |
| Claude Code | https://code.claude.com/docs/en/env-vars.md | None shown. |
| Claude Desktop | https://claude.com/docs/connectors/custom/desktop-extensions.md | None shown. |
| Claude Desktop | https://claude.com/docs/connectors/building/mcpb.md | None shown. |
| Claude Desktop | https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop | None shown. |
| Claude Desktop | https://claude.com/docs/plugins/overview.md and https://claude.com/docs/plugins/platform-support.md | None shown. |
| Claude Desktop | https://code.claude.com/docs/en/desktop-linux.md | "beta". |
| MCPB format | https://github.com/modelcontextprotocol/mcpb (MANIFEST.md) | "Current version: 0.3. Last updated: 2025-12-02". The CLI `@anthropic-ai/mcpb` is at 2.1.2. |
| Cursor | https://cursor.com/docs/mcp and https://cursor.com/help/customization/mcp.md | None shown. |
| Cursor | https://cursor.com/docs/cli/mcp.md | None shown. |
| Codex CLI | https://developers.openai.com/codex/mcp.md (redirects to learn.chatgpt.com) | None shown. |
| Codex CLI | https://learn.chatgpt.com/docs/config-file/config-basic.md | None shown. |
| VS Code | https://code.visualstudio.com/docs/agents/reference/mcp-configuration | Last modified 2026-10-22 (HTTP header). |
| VS Code | https://code.visualstudio.com/docs/agents/run/manage-mcp-servers | Last modified 2026-10-22 (HTTP header). |
| Gemini CLI | https://geminicli.com/docs/tools/mcp-server/ | Last modified 2026-10-16 (HTTP header). |
| Gemini CLI | https://geminicli.com/docs/cli/trusted-folders/ | Last modified 2026-10-16 (HTTP header). |
| Gemini CLI | https://github.com/google-gemini/gemini-cli (README) | The README notes the change to Antigravity CLI of 2026-06-18. |
