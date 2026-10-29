# Run topopt in any MCP client

[Back to the README](../../README.md)

This page is for a client that has no tested recipe, and for a client that has one but whose recipe you want to understand.
The topopt server is a normal MCP server over stdio. Any client that starts a local stdio server can run it.
The steps are in this order: what to have, the setup, a manual setup, the server on its own, the config, the typical problems, tips per client.

Facts on this page come from a run on Linux x86_64 (uv 0.12.1, 2026-10-27), or from the code (the file and line are named), or from a test
(named). A fact that was not run is marked "Not run here". The Mac facts were not run on a Mac for this page.

## 1. What to have

- A supported platform: an Apple silicon Mac (M1 or newer) with macOS, or a Linux x86_64 computer. An Intel Mac is best effort. Windows is not
  supported ([Platforms](install.md#platforms)).
- `git`, to clone the repo.
- At least 16 GB of memory for the hook example ([Requirements](install.md#requirements)).
- The network for the first setup. It downloads about 205 MB (Python 3.13, and 55 packages). Run on 2026-10-27: 23 s on a fast link, with
  an empty uv cache and no Python. The three folders (the environment about 500 MB, the uv cache about 490 MB, the Python of uv about 110 MB) take about 625 MB on the disk together, because uv links the files of its cache into the environment, so the folders share them (`du -shc`, Linux).
- uv is optional. topopt uses your uv if you have one (uv 0.6.8 or newer); else the setup downloads a pinned uv release and checks its checksum.
  The minimum is `MIN_UV_VERSION` and the pinned release is `PINNED_UV_VERSION` in `scripts/topopt-env.sh`.
  The launcher gets Python 3.13 itself.
- On a Mac, for the Accelerate solver only: the Xcode command line tools (`xcode-select --install`). Without them, only the CPU solver of `settings.solver: auto` changes: it is `amg` in place of `cholesky`. The GPU solver gpu-mg does not need them (`topopt/config.py`, `topopt/solver_choice.py`).
  The build is `sh topopt/native/build.sh`, and it calls `xcrun` (in `topopt/native/build.sh`). On Linux it stops with `xcrun: not found`. Not run here on a Mac.

## 2. The automatic setup

Clone the repo, go into it, and run the launcher:

```
git clone https://github.com/chartergroup/topopt.git
cd topopt
sh scripts/topopt-env.sh sync
```

`sync` makes the environment in `<repo>/.plugin-env/` and prints the path of its Python on stdout. The steps go to stderr: step 1 makes the Python 3.13
environment, step 2 installs the packages of `requirements.lock`, step 3 links the topopt code. It ends with `topopt: ready after 23 s (pins=lock): <path>`
(a cold start, 2026-10-27). A second `sync` on a ready environment ends at once.

Check the result:

- `sh scripts/topopt-env.sh status` prints `topopt: ready (pins=lock): <path>` and exits 0. When the environment is not ready it prints the reason
  and the command to run, and exits 1.
- `sh scripts/topopt-env.sh doctor` prints the facts of the setup (uv, the environment folder, the Python, the stamp, the imports) and, at the end,
  the next command. It exits 0 when everything is ready and every import works, and 1 when a command is needed. It installs nothing and uses no network.
  It takes about 2.5 s, and about 3.5 s with no Python cache files yet (Linux x86_64, no GPU device, 2026-10-27), and it prints progress lines while it runs.

The launcher runs the same way for every client. Other commands of the launcher: `mcp` (the server), `python ARGS...` and `topopt ARGS...` (run the environment Python and the
`topopt` command). The list is in the `case "$cmd" in` at the end of `scripts/topopt-env.sh`.

## 3. A manual setup

Use this when you want to install the packages yourself. The steps are those of the launcher (step 1 to 3 of `install_all` in `scripts/topopt-env.sh`).
Run them in the repo folder:

```
uv venv --python 3.13 /full/path/to/venv
uv pip install --python /full/path/to/venv/bin/python -r requirements.lock
uv pip install --python /full/path/to/venv/bin/python --no-deps --no-build-isolation -e .
```

- Python 3.13 is the Python of the launcher and of the tests, and the only Python that topopt supports (`requires-python = ">=3.13,<3.14"`).
- `requirements.lock` pins the `mcp` package (the line `mcp==2.3.0`), so the server needs no extra. The third command links the code with the
  setuptools of the lock (`--no-build-isolation`), so it asks no index.
  An Intel Mac uses `requirements-macos-x86_64.lock` ([Platforms](install.md#platforms)).
- Optional, Mac only: `sh topopt/native/build.sh` builds the Accelerate solver.
- Run on 2026-10-27 with Python 3.12, the Python of the command then (an empty folder, a warm uv cache): the three commands ended with exit 0, and `<venv>/bin/` had the entry points `topopt` and `topopt-mcp`.

The server command that this gives is `/full/path/to/venv/bin/topopt-mcp`. A config for it has `command` set to that path and no `args`.
It does not start the launcher, so it has no setup status and no repair of a cut setup: the packages must be there before the client starts.

**The launcher can use an environment in another folder.** Set `TOPOPT_ENV_DIR` to a full path. The launcher then makes and uses the environment there
(`<folder>/venv`) in place of `<repo>/.plugin-env/`. Run on 2026-10-27: `TOPOPT_ENV_DIR=/full/path/to/env sh scripts/topopt-env.sh sync` made
the environment there, `status` and `doctor` named it (`env dir ... (chosen by: TOPOPT_ENV_DIR)`), and `<repo>/.plugin-env/` was not touched.
The same variable must be in the `env` of the client config, or the server looks in the default folder. The order of the rules (`ENV_DIR=` near the top of
`scripts/topopt-env.sh`): `TOPOPT_ENV_DIR`, two internal variables (`EVAL_TOPOPT_ENV_DIR` is for the eval harness, `CLAUDE_PLUGIN_DATA` for the Claude Code plugin), the default.

## 4. Start the server on its own

Stdio is the only transport. `main()` in `topopt_mcp/server.py` calls `mcp.run("stdio")` and there is no other branch. So the
client must start the server as a child process. A URL (http or sse) does not work.

- With the launcher: `sh /full/path/to/topopt/scripts/topopt-env.sh mcp`.
- With a manual setup: `/full/path/to/venv/bin/topopt-mcp`.

In a terminal the server prints one line on stderr and then waits for a client on stdin:

```
[topopt_mcp] python: <path of the Python>; work folder: <folder> (from TOPOPT_WORKSPACE)
```

The words in brackets at the end say where the folder came from (`TOPOPT_WORKSPACE`, or `current folder`). Stdout stays empty: it carries the protocol. Run on 2026-10-27 with a ready environment, in a terminal and with stdin from `/dev/null`: both printed this line and no more.

If the environment is not ready, the launcher in a terminal waits up to 24 s for the setup (`TOPOPT_MCP_WAIT` changes it) and prints
`topopt: first start: the setup runs in the background ...` with the path of `setup.log`. A client does not wait: the launcher answers it at once
with the lists of tools from a stored copy, and a tool call returns the setup status until the setup is ready (`CONTRACTS.md` sections 34.1 and 25.5).
`sh scripts/topopt-env.sh doctor` says in which state the setup is.

To see the tools without an agent, use any MCP client library or inspector. Run on 2026-10-27: the `mcp` Python SDK started both commands above
(with `TOPOPT_WORKSPACE` in the env) and `list_tools` gave 31 tools (the server has 32 tools since `show_problem`). A third-party inspector was not run here.

## 5. The config

Every client has these fields under some name:

| Field | Value | Note |
|---|---|---|
| Transport | `stdio` | The only transport. Some clients need the key `type`, some have no such key. |
| Command | `sh` | |
| Args | `["/full/path/to/topopt/scripts/topopt-env.sh", "mcp"]` | A full path. Clients differ in their working folder and in variables like `${workspaceFolder}`, so a full path is the one form that works in all. |
| Env | `TOPOPT_WORKSPACE` = `/full/path/to/topopt` | A full path of the work folder (the runs go to `<folder>/runs/`). A value that is empty, or that has a `${...}` the server does not know, is not used, and the reply says so. Add `TOPOPT_ENV_DIR` here if you use it (part 3). |
| Server name | `topopt-local` | Short and unique. The repo has project files that name a server `topopt` (`.mcp.json`, `.cursor/mcp.json`, `.codex/config.toml`). Cursor takes the project entry when a project file and the user file use the same name. So give your own entry another name. |
| Start timeout | Codex: `startup_timeout_sec`. Gemini CLI: `timeout` (ms). Claude Code: the environment variable `MCP_TIMEOUT` (ms). Cursor, VS Code: no setting found. | The launcher answers at once, so the start is short. |
| Tool call timeout | Codex: `tool_timeout_sec`. Gemini CLI: the same `timeout`. Claude Code, Claude Desktop, Cursor, VS Code: not checked. | `export_step` holds a call until the file is written: 12 to 39 s on the public hooks, and 102 s and 613 s in two runs on Linux (530k and 925k triangles) ([Limits](../clients.md#limits)). Set 900 s or more where the client has a setting. |

The JSON template (the top key is `mcpServers`, as in Cursor, Gemini CLI and Claude Code files; VS Code uses `servers`; Gemini CLI has no key `type`, so drop that line there):

<!-- template: any-client.json -->
```json
{
  "mcpServers": {
    "topopt-local": {
      "type": "stdio",
      "command": "sh",
      "args": ["/full/path/to/topopt/scripts/topopt-env.sh", "mcp"],
      "env": { "TOPOPT_WORKSPACE": "/full/path/to/topopt" }
    }
  }
}
```

The TOML template (Codex CLI style). It follows the committed `.codex/config.toml`, with two differences: full paths and its own name, and a tool
timeout of 900 s in place of 300 s. 900 s is above the longest `export_step` that was measured (613 s). With the committed 300 s, a longer `export_step`
times out in Codex. `TOPOPT_MCP_WAIT = "180"` makes the launcher wait for the first setup, and `startup_timeout_sec = 240` is longer than that wait.
The repo keeps the wait until a live Codex run shows that the status server (the launcher answer during a setup) works there
(`CONTRACTS.md` sections 34.1 and 25.5).

<!-- template: any-client.toml -->
```toml
[mcp_servers.topopt-local]
command = "sh"
args = ["/full/path/to/topopt/scripts/topopt-env.sh", "mcp"]
startup_timeout_sec = 240
tool_timeout_sec = 900

[mcp_servers.topopt-local.env]
TOPOPT_WORKSPACE = "/full/path/to/topopt"
TOPOPT_MCP_WAIT = "180"
```

Replace `/full/path/to/topopt` with the folder of your clone. The work folder can be another folder: put that one in `TOPOPT_WORKSPACE`.
A test (`tests/test_client_configs.py`, `test_the_templates_of_the_guide_start_the_server_as_written`) replaces the placeholder with the repo root,
starts the server with a generic MCP client, and lists the tools.
After you edit the file, restart the client or the server. Most clients read the file only at start.

## 6. Typical problems

- **The tools do not show.** The client ignores a project config file until you trust the project (once for each project). A server can also
  stay Disabled. A Mac test with Cursor 3.27.22 (2026-10-27): the project server stayed Disabled while its log showed a clean start, and an entry in the user file
  with its own name worked. So put the entry in the user-level file, under a name that no project file uses, turn the project server off, and restart the client.
  Then run `sh scripts/topopt-env.sh doctor` (part 2) to see that the setup is fine.
- **The first call says `topopt is still setting up its Python environment.`** The client can mark that call as failed (its result has `isError` true): that is the setup status, not a fault ([the Code tab](install.md#the-code-tab-the-plugin), step 6; Mac tests of the chat extension and the Code tab, 2026-10-27). The setup runs in the background. Run
  `sh scripts/topopt-env.sh status`. It says "not ready: the setup is running" until the end, and `ready` after it. Wait, then call the tool again. To avoid the wait, run `sync` once before the first start.
- **A setup that was cut** (the client closed, the network failed, a kill). Run `sync` again. A marker `<env>/venv/.topopt-step2` makes it install the packages again (`--reinstall`). Run on 2026-10-27, an offline first start: `sync` ended with
  `the packages did not download: a network problem ... Run it again to resume.` (exit 1), and after that `UV_OFFLINE=1 sh scripts/topopt-env.sh sync` with a warm uv cache ended with exit 0 in 2 s.
- **An environment of an older launcher** (Python 3.12, made before the launcher made Python 3.13). The first start after the update makes it again, once: the stamp holds the Python, so `status` says
  `not ready: the environment has Python 3.12, and this launcher makes Python 3.13`, and `sync` prints `step 1 of 3: make the Python 3.13 environment again (the environment has Python 3.12; the packages download again, this needs the network)`. The wheels of numpy, scipy, numba, llvmlite and pyopencl
  are per Python version, so the uv cache of the old environment has none of them: the download is Python 3.13 (33 MiB) and the packages again, about 180 MB (run on 2026-10-27: 21 s on a fast link). Offline it stops with the network message
  (or the uv error of the Python download) and `setup failed. Run it again to resume`; the next start online ends the setup. A later start makes nothing again. topopt supports Python 3.13 only, so an environment of Python 3.12 cannot install this version.
- **An old uv.** If `TOPOPT_UV` names a uv older than 0.6.8, the launcher stops and tries no other uv (`find_uv` in `scripts/topopt-env.sh`). Run on 2026-10-27 with `TOPOPT_UV` set to a uv 0.6.7: `status` and `sync` exit 1 with
  `TOPOPT_UV is set to '<path>', but it is uv 0.6.7, and topopt needs uv 0.6.8 or newer.` An old uv on the PATH is skipped, and the launcher downloads its pinned uv; if that fails, update the old uv with the tool that installed it
  (`brew upgrade uv`, `pip install --upgrade uv`, `uv self update`). A new uv in `~/.local/bin` does not help while the old one comes first on the PATH.
  The message and the fix are in [First-run problems](install.md#first-run-problems) (a test keeps them).
- **Offline first start.** The first setup needs the network (about 205 MB; the same for the first start after an update from Python 3.12). Offline, with an empty uv cache, it stops with the network message above. Make the setup once on a machine with a network,
  or copy a filled uv cache and use `UV_OFFLINE=1`. If the machine has no uv, the launcher cannot download its pinned uv: on a machine with a network, install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
- **Where the logs are.** The server writes to stderr. Where a client keeps that is client-specific (the table in [clients.md](../clients.md#logs-restart-and-approval)). The setup log is `<env>/setup.log`
  (`.plugin-env/setup.log` by default), and `doctor` prints its last lines.

## 7. Tips per client

Each client has a few lines here. The details, with their dates, are in [docs/clients.md](../clients.md).

- **Claude Code.** Use the plugin: `/plugin` and the install steps of [Install the Claude Code plugin](install.md#install-the-claude-code-plugin). The plugin starts the same launcher. A plugin server that failed less than 15 minutes ago is skipped at the next start. `/mcp` shows the server, and `/mcp` then Reconnect restarts it.
- **Claude Desktop.** The chat uses the extension (a `.mcpb` file); the Code tab uses the plugin ([Claude Desktop](install.md#claude-desktop)). The first call can be marked as failed while the setup runs; that is the setup status (part 6).
- **Cursor.** Settings: Customize > MCPs. Files: `.cursor/mcp.json` (project) and `~/.cursor/mcp.json` (user). Quirk: the project server stayed Disabled in a Mac test; use a user entry with its own name ([Cursor](../clients.md#cursor)). Logs: Output panel, "MCP Logs".
- **Codex CLI.** File: `.codex/config.toml` in the project or `~/.codex/config.toml`. Quirks: it has no workspace variable, so use full paths; it ignores the project file until you trust the project; set both timeouts ([Codex CLI](../clients.md#codex-cli)). Check: `codex mcp list`.
- **VS Code (agent mode).** File: `.vscode/mcp.json`, top key `servers`. Quirk: VS Code also reads the root `.mcp.json` of this repo, so a `topopt` entry in both files gives two servers ([VS Code](../clients.md#vs-code-agent-mode-github-copilot)). Control: `MCP: List Servers`.
- **Gemini CLI.** File: `.gemini/settings.json`, top key `mcpServers`, no key `type`. Quirk: the folder must be trusted, and the key `timeout` (ms) covers the start and the tool call ([Gemini CLI](../clients.md#gemini-cli)). Check: `/mcp`.

Of the four other clients, only Cursor had a person test (one Mac test, 2026-10-27). Claude Code and Claude Desktop passed a first-use test on a Mac on 2026-10-27 ([Claude Desktop](install.md#claude-desktop)). The Codex, VS Code and Gemini facts come from their public docs and code, read on 2026-10-20 and 2026-10-25.
Not run here: any client other than the generic `mcp` SDK client.
