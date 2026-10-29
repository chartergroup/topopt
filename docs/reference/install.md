# Install, platforms and your machine

[Back to the README](../../README.md)

## Requirements

- A supported platform: an Apple silicon Mac (M1 or newer) with macOS, or a Linux x86_64 computer. An Intel Mac is best effort.
  See [Platforms](#platforms).
- No tool to install first. topopt uses your uv if you have one (uv 0.6.8 or newer); else the setup downloads a pinned uv release and checks its checksum.
- macOS only: the Xcode command line tools. They give you `clang`. Install them with `xcode-select --install`.
- At least 16 GB of memory for the hook example.

## Platforms

| Platform | State | `settings.solver: auto` gives | Lock file |
|---|---|---|---|
| macOS, Apple silicon (M1 or newer), native arm64 Python | Supported. This is the target. | `cholesky` (Apple Accelerate shim; run `sh topopt/native/build.sh`), else `amg` | `requirements.lock` |
| Linux x86_64 | Supported. | `pardiso` with the extra `pardiso`, else `amg` | `requirements.lock` |
| Intel Mac, or an x86_64 Python under Rosetta on Apple silicon | Best effort. Tested on Linux only. | `amg`. `cholesky` is a user error. | `requirements-macos-x86_64.lock` |
| Linux arm64 | Not tested. `amg` should work. | `amg` | `requirements.lock` |
| Windows | Not supported. | | |

Python 3.13 (`requires-python = ">=3.13,<3.14"`; pip refuses an install on Python 3.12).
The launcher and the desktop extension make Python 3.13 by themselves, the Python of every CI leg. The tests run on Python 3.13 on Linux x86_64 and on a Mac with
Apple silicon (`tests.yml`: the legs `macos-arm64`, `linux-x86_64` and `linux-x86_64-intel-pins` passed on 2026-10-27). Python 3.14 and later is not supported yet.

**Intel Mac (best effort).** The pins numba 0.67.0, llvmlite 0.49.0 and cryptography 50.0.2 of `requirements.lock` have no macOS x86_64 wheel.
So an x86_64 Python on a Mac uses `requirements-macos-x86_64.lock`: the same pins, with numpy, numba, llvmlite and cryptography moved back to
the last versions that have such a wheel. The launcher (`scripts/topopt-env.sh`) picks the lock by the machine type of the Python.
`TOPOPT_ALLOW_SOURCE_BUILD=1` lets it build a missing wheel from source (it needs LLVM and Rust; without it the install stops at once with the package name).

- Solver: `auto` gives `amg`. An explicit `cholesky` is a user error: the Accelerate shim is for Apple silicon, and nobody tested an x86_64 build of it.
  `pardiso` is for Linux x86_64 only (the MKL packages are an optional extra there, never a main dependency). `topopt verify` takes `cholespy` (extra `verify`),
  then `superlu`.
- Time: the estimate of `topopt check` is fitted to one Apple M1 run (and, for `amg`, to one Linux machine). It is not calibrated for an Intel Mac.
  Nobody measured `amg` on an Intel CPU.
- Rosetta: on Apple silicon a native arm64 Python is the fast path. Open the Terminal without Rosetta. The script of the official uv installer picks arm64 on
  an Apple silicon Mac, also in a Terminal that opens with Rosetta (read in the script, not run on a Mac). An x86_64 uv needs another source than that script: the release tarball of uv. The commands below are from a written
  procedure that nobody has run on a Mac yet. `VER` is the value of `PINNED_UV_VERSION` in `scripts/topopt-env.sh`:
  ```
  arch -x86_64 /usr/bin/true && echo "Rosetta works"          # else: softwareupdate --install-rosetta --agree-to-license
  VER=<PINNED_UV_VERSION>
  curl -fsSLO https://github.com/astral-sh/uv/releases/download/$VER/uv-x86_64-apple-darwin.tar.gz
  curl -fsSLO https://github.com/astral-sh/uv/releases/download/$VER/uv-x86_64-apple-darwin.tar.gz.sha256
  shasum -a 256 -c uv-x86_64-apple-darwin.tar.gz.sha256        # expect: OK
  mkdir -p "$HOME/uv-x86_64" && tar -xzf uv-x86_64-apple-darwin.tar.gz --strip-components=1 -C "$HOME/uv-x86_64"
  file -L "$HOME/uv-x86_64/uv"                                 # expect: Mach-O 64-bit executable x86_64
  export PATH="$HOME/uv-x86_64:$PATH"                          # this Terminal tab only
  arch -x86_64 /usr/bin/python3 tools/platform_probe.py        # shows what the machine and the Python are
  ```
- Run `python3 tools/platform_probe.py` to see what your machine is.
- CI: a Linux leg with the pins of the Intel lock (Python 3.13) runs when you start `tests.yml` by hand with `legs=all`. `.github/workflows/intel-mac.yml` runs the suite
  on a hosted Intel Mac runner (`macos-15-intel`), also started by hand. A push or a pull request starts neither. The Intel-pins leg passed on 2026-10-27; `intel-mac.yml` has not run yet: see [Tests](tests-and-limits.md#tests).

## Install the Claude Code plugin

This section is for the Claude Code plugin. The README has the short path. Here are the pins, the checkout, the updates and the
first-start facts. For a checkout install, see [Install](#install) below.

There is nothing to install first. topopt uses your uv if you have one (uv 0.6.8 or newer); else the setup downloads a pinned uv release and checks its checksum. (If that download does not work, install [uv](https://docs.astral.sh/uv/) with
`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`. If your uv is older than 0.6.8, update it instead with the tool
that installed it: see [the first-run problems](#first-run-problems).)

### The two commands and the session form

In a terminal:

```
claude plugin marketplace add chartergroup/topopt#latest
claude plugin install topopt@topopt
```

In a Claude Code session: `/plugin marketplace add chartergroup/topopt#latest`, then `/plugin install topopt@topopt`
(and `/reload-plugins` if Claude Code asks for it). `#latest` is a branch that the release workflow moves to the newest release.
In the Code tab of Claude Desktop, the session form opens the Plugins page of the app instead: see
[The Code tab: the plugin](#the-code-tab-the-plugin).

### The install prompt (Python before the first start)

Use this form when you want the Python environment ready before the first start. Then the tools work from the first start: no setup status.
Each form below installs the release `v0.5.0`.

- **Ask Claude.** Start `claude` in your work folder (see [The work folder](#the-work-folder)) and paste this prompt.
  Claude runs the commands. Depending on the permission mode of Claude Code, it asks you to approve each one, or it decides by
  itself. The prompt also sets up the Python environment of the plugin before the plugin starts, so `topopt` connects at its
  first start.

  ```text
  Install the Claude Code plugin topopt from the GitHub repo chartergroup/topopt. Run exactly this command:
  claude plugin install topopt --marketplace chartergroup/topopt#v0.5.0
  If it says that the option --marketplace is unknown, run these two commands instead:
  claude plugin marketplace add chartergroup/topopt#v0.5.0
  claude plugin install topopt@topopt
  If there is no claude command, stop and tell me to type this in the chat: /plugin install topopt --marketplace chartergroup/topopt#v0.5.0
  Then set up the Python environment of the plugin before it starts (about 200 MB, 1 to 3 minutes).
  Run claude plugin list --json. In the entry of topopt, take readFromFolder if it is there, else installPath. Use it as FOLDER:
  sh "FOLDER/scripts/topopt-env.sh" plugin-sync
  Give it the longest time limit that you can, and wait until it prints "topopt: ready". If a time limit stops it, run it again.
  If another error comes, stop and show it to me. Use no other source. When it is done, tell me to type /reload-plugins.
  ```

- **Type it in the chat** (Claude Code 2.1.275 or later): `/plugin install topopt --marketplace chartergroup/topopt#v0.5.0`
- **Run it in a terminal** (Claude Code 2.1.292 or later): `claude plugin install topopt --marketplace chartergroup/topopt#v0.5.0`
- **An older Claude Code.** Run these two commands in a terminal:

  ```
  claude plugin marketplace add chartergroup/topopt#v0.5.0
  claude plugin install topopt@topopt
  ```

After an install from a terminal, you can set up the environment the same way before the first start: `claude plugin list --json`
shows the folder of the plugin (`readFromFolder`, else `installPath`), and `sh <folder>/scripts/topopt-env.sh plugin-sync` ends with
`topopt: ready`. After an install from the prompt or from a terminal, type `/reload-plugins` in the open session, or start `claude` again.
The tag in each command names the release. For another release, a checkout or the newest `main`,
see [Pin the release, use a checkout or take the newest main](#pin-the-release-use-a-checkout-or-take-the-newest-main).

### What to expect

- **Setup: 1 to 3 minutes.** After the install prompt above (or `plugin-sync`), the environment is ready before the first start,
  and `topopt` connects at once. Otherwise the first start does the setup, as follows.
  The plugin makes its own Python environment. It downloads Python 3.13
  and the packages (about 200 MB to download; about 500 MB on disk). On a fast link on Linux, this took 23 s (2026-10-27).
  An environment that an older launcher made (Python 3.12) is made again once, at the first start (or `plugin-sync`) after the update: it needs the network,
  and it downloads Python 3.13 and the packages again, about 180 MB (21 s on a fast link, 2026-10-27). `status` and `doctor` say why the environment is not ready.
  Offline, the setup stops and the next start online ends it. A later start makes nothing again.
  Claude Code waits only 30 s for an MCP server (see [How long each client waits](#how-long-each-client-waits-on-the-first-start)).
  So the launcher does not wait for the setup. `topopt` connects at once and lists all its tools from a stored copy of its lists.
  While the setup runs, a call of a tool answers with the setup status: the step, the time, and what to do next.
  The setup goes on in the background. When it is done, the tools work in the same session: no `/mcp` and no reconnect
  (checked with Claude Code 2.1.293 on Linux).
  The same in the Code tab of Claude Desktop on a Mac (2026-10-27, release v0.4.0). Not checked yet: Codex CLI, Cursor, VS Code and Gemini CLI.
  The Claude Desktop extension starts the same launcher (seen in the app on a Mac on 2026-10-27, Claude Desktop 3.6556.2).
  On a slow link the download took 4 minutes 43 seconds (Linux, 2026-10-25), so wait for the line "ready" in the doctor or in `status`.
  `/topopt:start-here` gives the commands to check the setup. Later starts take 1 to 2 s.
  The environment is in the plugin data folder (`~/.claude/plugins/data/`), not in the plugin folder,
  so a plugin update does not download it again.
- **Mac: Xcode Command Line Tools.** The fast solver (Apple Accelerate) needs a small C library.
  Building it needs `clang` from the Xcode Command Line Tools (`xcode-select --install`).
  Claude offers to build it and does it only after you agree.
- **No shim: the AMG solver.** Without a working library, `settings.solver: auto` runs the AMG solver
  and prints a warning with the build command. AMG is slower. `topopt check` (the tool `check_problem`)
  gives the time estimate for the solver that runs. Nobody has measured a full AMG run on a Mac.
- **Linux still works.** It needs no shim. `auto` uses PARDISO if `pypardiso` is installed, and AMG if not.
  See [Linux x86_64](#linux-x86_64) for the measured times.
- **If the tools do not show.** Type `/topopt:start-here`. Its first step runs the doctor of the plugin environment. The doctor
  says why (no `uv` and no download of it, a setup that is still running, a setup that was cut) and gives the command to run. See [Check your machine](#check-your-machine).
  The table [First-run problems](#first-run-problems) gives the cause and the fix for each message.

The repo-root `.mcp.json` is for work in a checkout. The plugin does not use it.

### What the setup downloads, and what topopt sends

The setup runs at the first start, or before it with `plugin-sync` (the install prompt above) or `sync`. The launcher
`scripts/topopt-env.sh` does it. It downloads only these, and only over HTTPS. The sizes are from Linux x86_64 (2026-10-27); a Mac
gets other files.

| What | When | From | Check | Download | On disk, and where |
|---|---|---|---|---|---|
| uv 0.13.5 | Only if topopt finds no uv that it can use. It looks in this order: `TOPOPT_UV`, a `uv` on PATH (0.6.8 or newer), a copy that the launcher downloaded before. | The uv releases on GitHub (`https://github.com/astral-sh/uv/releases/download/`), with `curl`, else `wget` | The SHA256 of the archive for each platform is a constant in the launcher. A wrong sum deletes the file before anything is unpacked. | 18 MB | 43 MB in `<env dir>/uv/0.13.5/` |
| Python 3.13 | Only if uv has no Python 3.13 of its own | The Python builds of uv (python-build-standalone). uv 0.13.5 gets them from `releases.astral.sh`. | uv | 35 MB | 110 MB in the Python folder of uv (`uv python dir`, by default `~/.local/share/uv/python`) |
| The packages | The first setup, and again when the lock file changes | PyPI, or the index that your uv is set to use. topopt sets no index. | The exact versions of `requirements.lock` (55 packages, setuptools among them; an x86_64 Python on a Mac uses `requirements-macos-x86_64.lock`). If the lock cannot install on the platform (an error that is not a network error), the pins of `pyproject.toml` and the setuptools pin of the lock. | 150 to 170 MB | 497 MB in `<env dir>/venv`, and the cache of uv (`uv cache dir`, by default `~/.cache/uv`) |
| setuptools 84.2.1 | With the packages: both lock files pin it | PyPI, or the index that your uv is set to use | The exact version of the lock | 0.9 MB | 6 MB in `<env dir>/venv`, and the cache of uv |

- The setup does not download topopt itself. Step 3 links the code of the plugin folder into the environment. For that, uv builds
  topopt with its build backend, setuptools (the last row). Step 2 installs it from the lock, and step 3 builds with that copy
  (`--no-build-isolation`), so step 3 asks no index.
- **After a plugin update.** A new plugin version has a new folder, so the next start runs steps 2 and 3 again. Step 2 finds every
  package of the lock in the environment, and step 3 links the code with the setuptools of the environment, so neither step uses the
  network. The one exception is an update that changes the Python of the launcher (the update to Python 3.13): the environment is made again,
  and the setup downloads Python 3.13 and the packages again (about 180 MB) and needs the network. Measured on 2026-10-27 with `sync` and the network cut (a preload library that refuses connections and name lookups; for the
  uv that the launcher downloads, a static file, also a proxy address that answers nothing), 11 minutes after the last answer of the
  index: ready after 3 s (uv 0.12.1), 4 s (uv 0.6.8) and 3 s (uv 0.13.5), with no network call.
  An update that changes the lock file downloads the new pins in step 2. The first update to a version whose lock pins setuptools is
  such an update. Offline, that start stops with the network message, and the tools work again after a setup with the network. With
  `UV_OFFLINE=1`, uv takes the packages from its cache, and step 2 works offline if the cache has them (measured: the setuptools of the
  earlier setup was there, ready after 4 s).
- `<env dir>` is the plugin data folder that Claude Code gives the plugin, below `~/.claude/plugins/data/`. It stays when the plugin
  updates. For a checkout it is `.plugin-env` in the repo. `TOPOPT_ENV_DIR` sets another folder.
- Measured on Linux x86_64 on 2026-10-27, with no uv and an empty cache of uv: the setup (Python 3.13, with the download of uv) took 26 s, and the network counters of the
  container grew by 223 MB (all its processes, so this is an upper bound). The uv log gives 143 MiB (150 MB) for the 12 largest packages.
  The cache of uv holds the same package files as the environment. On this disk uv linked them, so the cache added only 18 MB (`du -shc`, 2026-10-27).
  On another disk it is a second copy.
- Claude Code does the install itself: it clones this repo from GitHub. That is not a download by topopt.
- **Extras, only on request.** Some tools offer an install command for an optional extra: the PARDISO solver of Linux (`pardiso`),
  the tet mesher (`tet`), the check with a tet model (`verify`) or the STEP export (`step`). topopt only shows the command. It installs
  nothing itself. The command installs from PyPI. The build of gmsh without a GUI (for the `tet` extra on a computer with no display)
  comes from `gmsh.info`. The library of the GPU solver (pyopencl) is one of the 55 packages. On a Mac, `clang` builds the fast solver
  on your computer, with no download.
- **A repair.** After a cut setup, the next setup runs `uv pip install --reinstall`. That checks the index again, so it needs the network,
  but it downloads no file that the cache of uv has.

**What topopt sends: no data, unless you allow diagnostic data.**

- This version has no option for diagnostic data, so it sends nothing. The checks below are the evidence.
- The package code (`topopt/`, `topopt_mcp/`) has no HTTP client and opens no connection to another computer. Its only sockets are
  the dashboard server and a check for a free port, both on `127.0.0.1` (`topopt/dashboard_server.py`, `_free_port` in
  `topopt_mcp/server.py`).
- The dashboard server listens on `127.0.0.1` only, and its pages load nothing from the internet. The main page loads three.js from
  `topopt/dashboard/vendor/`. Each run also writes the refine page, `result/refine.html`; after a run with a dashboard (every run
  that the MCP server starts), the browser opens it at `<dashboard>/refine`. That page holds its own copy of three.js (from
  `topopt/refine/vendor/`), so it shows the part offline too, also when you open the file itself. The dashboard also puts these copies
  into a refine page that an older topopt made, when it serves that page.
- The doctor runs `git` in a checkout to read local facts (for example the commit). It does not fetch.
- We checked this on Linux x86_64 on 2026-10-27, after the setup above. For every process of the launcher, the MCP server, the runs
  and the dashboard server, a library loaded with `LD_PRELOAD` refused each connection to an address that is not on this computer and
  each name lookup, and logged them. A `topopt check` and a `topopt run` of `examples/hook/hook_short.yaml`, and an MCP session
  (`environment`, `check_problem`, `run_topopt`, `run_status`, `run_result` and the main page of the dashboard) worked. No process tried
  to connect to another computer. Such a library cannot watch uv (a static program); in these runs the launcher ran uv only as
  `uv --version`. It does not watch the browser either. So a headless Chromium opened the refine page of such a run, as a file and
  from the dashboard, with the same library and with every host name but `127.0.0.1` unknown to it. The page showed the part, and the
  page itself was its only request. The library refused one connection, of Chromium itself and not of the page: a name lookup at a
  public DNS server over HTTPS.

### Pin the release, use a checkout or take the newest main

The part `#v0.5.0` pins the install to the release `v0.5.0`. The branch `main` can hold work that is not released yet.
To try the newest `main`, leave out `#v0.5.0`. That path has no checks. On that path the marketplace is named `topopt`,
so in the two-command form write that name in the install command. The one-command forms need no name.

To use a checkout, run these three commands in place of the add command. Then run the install command.

```
git clone https://github.com/chartergroup/topopt
git -C topopt checkout v0.5.0
claude plugin marketplace add ./topopt
```

In a Claude Code session, the same commands are `/plugin marketplace add ...` and
`/plugin install topopt@topopt`.
If the repo is private, git on your computer needs access to it. Claude Code runs git with its prompts turned off. It clones over
SSH when your SSH key works for github.com without a passphrase prompt (the host must be in `known_hosts`), else over HTTPS
with a credential that git already has (for example from `gh auth login` or the macOS Keychain). `CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1`
skips the SSH check (Claude Code docs, read on 2026-10-25). Nobody has run this install from GitHub yet.

### Update

**The `#latest` route.** If you added the marketplace with `#latest`, run these two commands, then start a new session
(or type `/reload-plugins`):

```
claude plugin marketplace update topopt
claude plugin update topopt@topopt
```

Claude Code says "Restart to apply changes". The Python environment is reused: the next start links the new code in about 2 s
and downloads nothing. Only a changed lock file downloads packages. The link step needs no network: see
[After a plugin update](#what-the-setup-downloads-and-what-topopt-sends).
Checked on Linux with a local mirror and Claude Code 2.1.293, 2026-10-27.

#### Update to a newer release

A release is a tag of this repo, for example `v0.5.0`. `CHANGELOG.md` lists the changes of each release.
The install command names the tag, so your install stays on that release.
A new commit on `main` does not change it, and a new release does not reach you by itself.
`claude plugin update topopt@topopt` alone does not move a pinned install to a new tag:
it answers "already at the latest version".
If you run the add command with the new tag first, Claude Code refuses it. It says that the source does not match
the entry in your settings. Use one of these two routes. Write the new tag in place of `<new-tag>`.

The releases up to version 0.2.0 name the marketplace `topopt-demo`. Later releases name it `topopt`.
If you installed version 0.1.0 or 0.2.0, `claude plugin marketplace list` shows `topopt-demo`.
To move to a later release, use Route A and remove that marketplace: write `topopt-demo` in its first command.
Route B cannot make this move, because it changes only the ref of the settings entry that holds the old name.

**Route A: remove, add, install.** This is the simple route.

```
claude plugin marketplace remove topopt
claude plugin marketplace add chartergroup/topopt#<new-tag>
claude plugin install topopt@topopt
```

`marketplace remove` also uninstalls the plugin and deletes the plugin data folder.
That folder holds the Python environment, so the next start sets up Python again (1 to 3 minutes, see "What to expect").

**Route B: change the ref.** This route keeps the Python environment.

1. In `~/.claude/settings.json`, change `extraKnownMarketplaces.topopt.source.ref` to the new tag.
2. Run these two commands:

   ```
   claude plugin marketplace add chartergroup/topopt#<new-tag>
   claude plugin update topopt@topopt
   ```

Do not run `claude plugin marketplace update` between step 1 and step 2. It stops with the error "Marketplace 'topopt' not found".

## Install

Run all commands in this file in the repo folder.
The `topopt` commands need the virtual environment.
Run `source .venv/bin/activate` in each new terminal before the first `topopt` command.
Or write `.venv/bin/topopt` in place of `topopt`.

Run these commands in the repo folder. Python 3.13 is the Python of the launcher and of the tests, and the only Python that topopt supports:

```
uv venv --python 3.13
uv pip install -e ".[dev,mcp]"
sh topopt/native/build.sh
source .venv/bin/activate
```

The third command (`build.sh`) builds the solver library `topopt/native/libaccel_chol.dylib`.
The solver loads this file from the folder `topopt/native/`.
To get the tested package versions, use the lock file in place of the second command:
`uv pip install -r requirements.lock` and then `uv pip install --no-deps -e .`.
The `pyproject.toml` install pins only the direct packages, so a sub-package can have a different version.
If the file is missing (or it does not load), `settings.solver: auto` runs the AMG solver and
`topopt run` prints one warning with the build command. AMG is slower (see the estimate of `topopt check`).
With `settings.solver: cholesky`, the run stops with this message:
`.../topopt/native/libaccel_chol.dylib not found. Run topopt/native/build.sh`.
Run `sh topopt/native/build.sh` again to fix it.

### Linux x86_64

Skip the third command (`build.sh`). The Accelerate solver is for macOS only.
With `settings.solver: auto` (the default), a Linux run uses the PARDISO solver
if the optional package `pypardiso` is installed. Without it, the run uses the AMG solver.
Both solve the same equations. On the hook example, the compliance of an AMG run and of a PARDISO run
differs by less than 1e-7 (relative) in all 81 iterations. PARDISO is much faster.
Install it with the `pardiso` extra:

```
uv venv --python 3.13
uv pip install -e ".[dev,mcp,pardiso]"
source .venv/bin/activate
```

`pypardiso` pulls Intel packages. They are about 456 MB in a fresh venv, and their licenses are proprietary:

- `mkl`, `tbb` and `tcmlib`: Intel Simplified Software License.
- `intel-openmp` and `intel-cmplr-lib-ur`: Intel EULA for Developer Tools. It has more limits than the first one (for example, no use in a hosted service).
- `umf`: Apache-2.0.

The `mkl` version is not pinned: `pypardiso` only needs `mkl != 2024.0`.
So the extra is optional: it is not in the main dependencies, and the repo does not include MKL.
To add it later, run `uv pip install -e ".[pardiso]"`.

`topopt check` shows the solver that the run will use. On Linux without `pypardiso`, it also shows
the install command in a line that starts with `recommendation:` (add `--json` for one JSON object, with the field `recommendation`).
`topopt run` prints one `warning:` line with the same hint.

Hook example (`examples/hook/hook.yaml`, 80 iterations), measured on one Linux x86_64 machine with 16 cores / 22 threads:

| Solver | Run time |
|---|---|
| `amg` (no `pypardiso`) | 406 s |
| `pardiso` | 99 s |
| `cholesky` on an M1 Mac, for comparison | 148 s |

The times are for one machine and they are single runs. They are not a promise. The 99 s of `pardiso` is the best of 62 runs of the default hook on a shared Linux host (experiment E1, a run-time study, 2026-10-23): minimum 98.8 s, median 462 s, 90th percentile 743 s, maximum 1,469 s. Do not read 99 s as the usual time.
On Linux, `topopt` sets `OPENBLAS_NUM_THREADS` to 4 (or the number of usable CPUs, if smaller) when you set none of
`OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS` and `OMP_NUM_THREADS`.
Many OpenBLAS threads slow the run down (PARDISO: 149 s without the limit, on the same machine).
Your own value stays.
The setting `solver: cholesky` needs Apple Accelerate. On Linux it gives an error that lists the other solvers.

### How long each client waits on the first start

The first start takes 1 to 3 minutes on a fast link. Each client waits for the server for its own time. The table gives the
time, the setting that changes it, and the source. The launcher is the script `scripts/topopt-env.sh mcp`: every client in the table
starts the server with it (Claude Desktop does so). A client start does not wait for the setup: the launcher stays connected. It answers the list requests of the client from a stored copy of the lists of the real server (`topopt_mcp/launcher/`),
so the client sees all the tools from its first answer. A call of a tool gives the setup status (the step, the time and one next step).
When the setup is ready, the launcher replaces itself with the Python server, and the tools work in the same session. The setup goes
on in the background.

| Client | Waits for the server | Setting | Source |
|---|---|---|---|
| Claude Code (the plugin) | 30 s. A server that has not connected by then shows failed, and a plugin server that failed less than 15 minutes ago is skipped at the next start. The launcher connects before that. | `MCP_TIMEOUT` (ms) | Claude Code docs (see below). Measured with Claude Code 2.1.293 on 2026-10-25: connect timeout after 30000 ms. |
| The launcher | 0 s for a client (stdin is not a terminal): it connects at once and answers from the stored lists until the setup is done. 24 s in a terminal (a person who runs `mcp` by hand). | `TOPOPT_MCP_WAIT` (s) | `scripts/topopt-env.sh` |
| Codex CLI | 240 s in `.codex/config.toml` (the launcher waits 180 s there) | `startup_timeout_sec` | the file in this repo; `docs/clients.md` |
| Gemini CLI | 600 s (600,000 ms) | `timeout` (ms) | `docs/clients.md` |
| Cursor, VS Code | no start time limit found in the docs or the code that we read | none found | `docs/clients.md` |
| Claude Desktop (the extension) | not measured. The extension starts the launcher, which connects at once (measured on Linux with the command of the manifest: the tools after 0.03 s). | none found | `CONTRACTS.md` section 25.5 |

The launcher connects at once, so a client that waits for the server (30 s, 240 s or more) does not time out on the first start.
The exception is the Codex file of this repo: it sets `TOPOPT_MCP_WAIT=180` on purpose until a live Codex run, so Codex waits up to 180 s there.
`TOPOPT_MCP_WAIT` (seconds), when set, wins in both cases: the launcher then waits that long for the setup before it answers. Set it
only to have the real Python server from the first answer: set the wait of the client and `TOPOPT_MCP_WAIT` both longer than the setup. For Claude Code, start it as `MCP_TIMEOUT=450000 TOPOPT_MCP_WAIT=420 claude`, or put both in the `env` block of its
`settings.json`. These values are for the first start only:
- They come from the slowest setup that we measured: 283 s (Linux, 2026-10-25, a cold start on a link of about 0.7 MiB/s).
  420 s gives that setup about 50 % more time. `MCP_TIMEOUT` (450,000 ms = 450 s) is longer than `TOPOPT_MCP_WAIT` (420 s),
  so Claude Code does not stop the launcher while the launcher still waits for the setup.
- `MCP_TIMEOUT` applies to every MCP server of Claude Code: with it, a server that does not start makes Claude Code wait up to
  450 s. Remove both after the first start.

Measured on 2026-10-25 (Linux, Claude Code 2.1.293, a cold start with an empty `uv` cache): with both set longer than the setup,
`topopt` connected in the first session after 126 s, with its 27 tools. Run `sh scripts/topopt-env.sh sync` once in a terminal to
avoid the wait. For the Claude Code plugin, run `plugin-sync` of the plugin folder instead (the [install prompt](#the-install-prompt-python-before-the-first-start) does it):
`sh <plugin folder>/scripts/topopt-env.sh plugin-sync`. It makes the environment in the plugin data folder, the folder that Claude Code
gives the server (`CLAUDE_PLUGIN_DATA`). `claude plugin list --json` shows the plugin folder: `readFromFolder` for a marketplace in a
local folder, else `installPath`. Measured on 2026-10-27 (Linux, Claude Code 2.1.293, a new home with no uv, no Python and no uv
cache): `plugin-sync` took 83 s, and the first session connected `topopt` in 0.7 s, with no failure mark in `mcp-needs-auth-cache.json`.
With an older launcher (it waited 24 s and then answered with an error), the same install without `plugin-sync` failed at the
first start, Claude Code wrote the mark, and the next session did not start `topopt`. The current launcher does not fail a first start.

Sources for Claude Code, read on 2026-10-25: https://code.claude.com/docs/en/env-vars (`MCP_TIMEOUT`, default 30000 ms) and
https://code.claude.com/docs/en/mcp (the 15 minutes).

## Claude Desktop

Claude Desktop for macOS runs topopt in two ways:

- In the **Code tab**, Claude Code runs on your Mac. Add the plugin from the Plugins page of the app:
  [The Code tab: the plugin](#the-code-tab-the-plugin).
- In the **chat**, topopt runs through an extension (a `.mcpb` file), plus a plugin upload for the skills:
  [The chat: the extension](#the-chat-the-extension).

Both passed a first-use test on a Mac on 2026-10-27 with release v0.4.0 (MacBook Air, Apple M4, 16 GB; a clean start with no uv,
no uv cache and no earlier topopt). topopt connected with 31 tools. A tool call during the setup gave the setup status.
After the setup, the same session gave the real result, with no restart: topopt 0.4.0 and the GPU solver on the M4 (its self-test
passed). The chat test ran in Claude Desktop 3.6556.2. The Code tab test did not record the version of the app.

### The Code tab: the plugin

In the Code tab, `/plugin marketplace add chartergroup/topopt#latest` does not run as a command: it opens Settings > Plugins.
Add the plugin there. The labels below are those of the English app (the Mac test of 2026-10-27).

1. Open Settings > Plugins. Or press the + button next to the prompt box and choose Add plugins.
2. Press + Add, choose Add marketplace, then Add from a repository.
3. Type `chartergroup/topopt#latest` in the field URL and press Sync. `#latest` is the newest release.
   If the repo is private, Sync stops with an access error and asks for a personal access token: a GitHub token that can read
   the repo (the page links to GitHub to make a fine-grained token). With the token, the sync works. A public repo needs no
   token.
4. Right after the sync, Discover lists **topopt: topology optimization for 3D printing** (by chartergroup). Press Add on it at once.
   If you close the view first, topopt may not come back. The marketplace stays under + Add > Manage marketplaces, but the plugin
   is not in Discover, in Yours or in Search (a Claude Code issue, anthropics/claude-code#93400, open on 2026-10-27).
   Two routes may help. Neither is checked yet:
   - Remove the marketplace under Manage marketplaces, and add it again (steps 2 to 4).
   - If you have the `claude` command in a terminal: `claude plugin marketplace update topopt`, then
     `claude plugin install topopt@topopt` (the route of that issue).
5. The app shows the dialog **This plugin includes local MCP servers**: the plugin starts a local server (topopt) that can run code
   on your Mac without asking. Press Continue.
   (The other button is Disable plugin. What it does was not checked.)
6. Start a Code tab session in your work folder and ask: Check my machine with topopt.
   The first start sets up Python for the plugin: 1 to 3 minutes, about 200 MB (73 s in the Mac test, with the download of uv and
   Python). During the setup the call answers with the setup status, and the app marks the call as failed: that is the setup
   status, not a fault. In the test, Claude waited and did not call again until the setup was ready. After the setup, the same
   session gives the real result: the topopt version, the solver (in the test: the GPU solver on the M4) and the optional parts that
   are missing (in the test: the Accelerate library, which Claude offered to build, and gmsh).
   [What to expect](#what-to-expect) has the rest of the first start.

Choose a project folder when you start a Code tab session. In the Mac test, a session with no project folder ran in a folder that
the app made for the session, and Claude warned that the runs are deleted with the session
(see [The work folder](#the-work-folder)).
The server log of the plugin is in `~/Library/Logs/Claude/`, in the file `mcp-server-topopt: topology optimization for 3D printing.log`.
Not checked in the app: the update to a newer release.

### The chat: the extension

Claude Desktop for macOS can run topopt in the chat through an extension (a `.mcpb` file).
The extension has the MCP server: the tools, and the role playbooks as the tool `team_guide`, as prompts and as resources.
A plugin upload adds the skills `start-here` and `part-to-topopt`.
The Mac test of 2026-10-27 installed the extension file of the release (`topopt-0.4.0.mcpb`), not a build of step 2. The test had
no plugin upload (step 6).
Claude Desktop has a beta for Debian-based Linux (Ubuntu 22.04 or later, Debian 12 or later; the Claude Code docs, read on
2026-10-25). The docs of the extension format name macOS and Windows only, and nobody has tested the extension on Linux. On Linux,
use the [Claude Code plugin](#install-the-claude-code-plugin).
You must be signed in to claude.ai in the app. An organization can turn extensions off; then Settings has no Extensions page.

Each release from v0.4.0 on has the extension as a download: `topopt-X.Y.Z.mcpb`, and the same file as `topopt.mcpb`. This link gives the `topopt.mcpb`
of the newest release:
https://github.com/chartergroup/topopt/releases/latest/download/topopt.mcpb
With the downloaded file, skip steps 1 and 2. If the repository is private, the link works only when you are signed in to GitHub
with access to the repository.

1. Install [uv](https://docs.astral.sh/uv/) (`brew install uv`): the build of step 2 runs with it. The extension itself needs no uv
   and no Python: it starts the launcher of the plugin (`/bin/sh scripts/topopt-env.sh mcp`), which uses a uv on the PATH of
   Claude Desktop if it is uv 0.6.8 or newer, else downloads its pinned uv release and checks its checksum.
2. Build the extension. Run this in a checkout:

   ```
   uv run --no-project --python 3.13 packaging/mcpb/build.py
   ```

   Use `uv run` and not `python3`: the build script needs Python 3.11 or newer, and the `python3` of a clean Mac is
   usually Python 3.9 (the Xcode Command Line Tools Python). The command lets `uv` fetch Python 3.13 for the script (once). With an older Python the script prints
   this command and stops.
   It writes `dist/mcpb/topopt-<version>.mcpb` (about 1.8 MB). The file is not in the repo (it is git-ignored).
   In a git checkout only the files that git tracks go in, and the script names the files that it left out.
   If the `mcpb` CLI (`npm install -g @anthropic-ai/mcpb`) is on your PATH, the script also checks the manifest with it.
3. Install the extension: open the `.mcpb` file with Claude Desktop, or go to Settings > Extensions and drag the file in.
   (In newer builds the page is Settings > Desktop app, then Manage next to Extensions. Another way is Advanced settings > Install Extension.)
   Claude Desktop shows an optional Work folder. Left empty, it is `~/Documents/topopt` (the dialog shows no default); the folder button
   chooses another one. Runs, STL files and results go in its subfolder `runs/`. After you change it, quit Claude completely (Cmd+Q)
   and open it again (see [The work folder](#the-work-folder)).
4. First start: topopt connects at once with all its tools and sets up its Python environment in the background: Python 3.13
   and about 200 MB of packages (and the pinned uv when the Mac has no usable uv). Expect 1 to 3 minutes (Python 3.12: the setup was ready after
   60 s in the Mac test, and after 59 s on a fast link on Linux). Until then a tool call answers with the setup status; after it the tools work in the same chat.
   If the setup stops (no network, the Mac went to sleep or was shut down), turn the extension off and on: the next start continues from
   the `uv` cache. The environment is in the extension folder (`<extension folder>/.plugin-env`), so an update of the extension removes it,
   and the next start sets it up again from the `uv` cache (on Linux: ready after 3 s).
5. Optional, for the fast solver on a Mac: without the Apple Accelerate library, every run uses the slower AMG solver.
   Claude in the Desktop chat has no shell, so it cannot build the library. You can build it in Terminal. Install the Xcode
   Command Line Tools (`xcode-select --install`), ask Claude to call `environment` (it quotes the command with the
   path of the extension folder), and run that command (`sh '<extension folder>/topopt/native/build.sh'`).
   A Claude Desktop update of the extension replaces the folder and removes the library: run the command again then.
6. Plugin upload (the skills): make the plugin zip from the committed files, then upload it in Claude Desktop under
   Customize > Plugins (the limits are 200 MB and 5,000 files; this zip is about 13 MB and has about 750 files):

   ```
   mkdir -p dist && git archive --format=zip -o dist/topopt-plugin.zip HEAD
   ```

   Claude Desktop takes both forms: the plugin files at the top of the zip, or inside one top folder (Claude docs, read on
   2026-10-25). The command above makes the first form.
   In Claude Desktop chat the plugin gives the skills only. The server comes from the extension of step 3.
   Skills run in the cloud sandbox of the chat, so a skill step that runs a shell command (the launcher check in
   `start-here`) does not reach your Mac. The tool `team_guide` does not need a skill.
7. Test: ask Claude for the tools of topopt, then ask it to call `team_guide` with no role.
   Then ask it to optimize the hook example (`examples/hook/hook.yaml`).

If the extension does not start (these facts come from reading, not from a test on a Mac):

- The logs of Claude Desktop are usually in `~/Library/Logs/Claude/`. Look for the file of the `topopt` server (the names start with `mcp`).
  The Extensions page also shows the status and the logs of each local server (section Local MCP servers).
- No `uv` is no error: the launcher downloads its pinned uv. A Dock app may not read the `PATH` of your shell, so the launcher
  also looks in `~/.local/bin`, `~/.cargo/bin`, `/opt/homebrew/bin` and `/usr/local/bin`. When the download does not work, a tool
  call says why and what to do (the problem table below has the messages).
- A tool call answers with the setup status: the setup is still running. Wait 1 to 3 minutes in all, then ask again. The launcher
  connects at once, so a time limit of Claude Desktop should not stop the first start (not checked in the app; see
  [How long each client waits](#how-long-each-client-waits-on-the-first-start)).

The roles: `team_guide(role)` gives the playbook of one role (`lead-design`, `cad`, `fea`, `print` or `finish`).
The same texts are MCP prompts. Claude Code names a prompt after its server: `/mcp__topopt__<role>` in a checkout with
the repo-root `.mcp.json`, and `/mcp__plugin_topopt_topopt__<role>` for the server of the plugin (the tools of the plugin
server have that prefix; not tested for prompts). In Claude Code, the plugin also has the roles as sub-agents.

## The work folder

The work folder is where topopt keeps your runs (`runs/<name>/`), the files that Claude writes for you, and the start of every relative
path in a tool call. Ask Claude to call `environment`: the key `work_folder` shows the folder in use and where the setting came from.
If the folder is wrong or cannot be written, each reply of the server has a `work_folder_note` that says what to do.
How to change the folder depends on how you started topopt.

### Claude Desktop extension

- The setting is the field **Work folder** in Settings > Extensions > topopt. The folder button chooses a folder. Left empty, the folder is
  `~/Documents/topopt` (the dialog shows no default).
- After a change, quit Claude completely (Cmd+Q on a Mac) and open it again. A new chat or a new Code tab session is not enough: it keeps
  the old server process. `environment` shows `server.started_utc`, the time that the server process started. If it is older than your
  change, this is the old server.
- There is no environment variable to set here. The extension gives the setting to the server.
- The server writes the folder to its log at each start. The logs of Claude Desktop are usually in `~/Library/Logs/Claude/`.

### Claude Code plugin

- The plugin has no setting for the work folder. The work folder is the folder where Claude Code started: where you typed `claude`.
- To change it, quit Claude Code and start `claude` in the other folder.
- A Code tab session in Claude Desktop that you start with no project folder runs in a scratch workspace, and the app removes that folder
  when the session ends (seen on one Mac with Claude Desktop 3.6556.2, 2026-10-23). Choose a project folder when you start the session.
  Not checked: if the server of the plugin starts in that folder. If it does, the runs are lost with it, and the server says so in
  `work_folder_note`.
- Not checked: if a `TOPOPT_WORKSPACE` that you export in the shell before `claude` reaches the server of the plugin. The plugin gives
  the server no environment of its own.

### The command line

- `topopt run <file.yaml> --out <dir>` writes the run to `--out`. A relative path starts at the current folder. The command line needs no
  work folder setting. (If `TOPOPT_WORKSPACE` is set, the checkpoint of the run uses it to decide which region files lie outside the work
  folder.)

### Another MCP client

- Cursor and VS Code: the config in this repo (`.cursor/mcp.json`) and the snippet for VS Code (`docs/clients.md`) set `TOPOPT_WORKSPACE`
  to `${workspaceFolder}`. Open the folder that you want, or change the `env` of the topopt server in `.cursor/mcp.json` or `.vscode/mcp.json`.
- Codex CLI and Gemini CLI: the server starts in the folder where the client runs, so start the client in the folder that you want. To fix
  a folder, set `TOPOPT_WORKSPACE` to a full path in the `env` of the topopt server (Codex: `[mcp_servers.topopt.env]` in
  `.codex/config.toml`; Gemini CLI: `env` in `.gemini/settings.json`). [the client facts](../clients.md) has the facts for each client.
- Any other client ([the guide](any-mcp-client.md) has a JSON and a TOML template): put `TOPOPT_WORKSPACE`, with a full path, in the `env` of the topopt server in the config of that client. Then restart
  the server (or the client). A value that is empty, or that has a `${...}` the server does not know, is not used, and the reply says so.

What is not checked: the steps above come from the report of one Mac (Claude Desktop 3.6556.2, 2026-10-23), from the code, and from tests on
Linux. Nobody has run these on a Mac after the change that removed the default from the field:

- Whether Claude Desktop replaces `${HOME}` in the extension setting `TOPOPT_DEFAULT_WORKSPACE`. If it does not, the server does it and says so.
- Whether macOS asks before the server uses `~/Documents`, and what the reply says after you choose "Don't Allow".
- Whether Claude Desktop keeps the old text `${DOCUMENTS}/topopt` in the field after an update of the extension.

## After the install: update, extras, check

### Update a clone

A clone does not update itself. `topopt doctor` shows how far the clone is behind `origin/main`, but it does not fetch.
So fetch first, then run the doctor. Run these two commands in the repo folder:

```
git fetch origin
.venv/bin/python -m topopt doctor
```

The doctor line `origin/main:` gives the number of commits that the clone is ahead and behind, as of the last fetch, and the time of that fetch.
When the clone is behind, the section "Next commands" gives the update command and then the command that refreshes the packages.
In short, for a clone on the branch `main`, they are (the doctor prints the long form, with `git -C <folder>` and the Python of the environment):

```
git pull --ff-only
uv pip install -e ".[dev,mcp]"
```

- `git pull --ff-only` fetches by itself. It stops with an error, and changes nothing, when your branch has commits of its own and `origin/main` has new ones.
- Add the extras that you use to the second command (see "Extras"). If you installed from the lock file, repeat the two lock commands of "Install" in place of it.
- On another branch, the doctor prints a command that switches to `main` first. To keep your branch, it prints the command that merges `origin/main` into it instead.
- On a Mac, the file `topopt/native/libaccel_chol.dylib` is not in git, so the update keeps it. If the update changed `topopt/native/accel_chol.c`, run `sh topopt/native/build.sh` again. This step has not been run on a Mac for this text.
- A clone that is older than the command has no `topopt doctor`. It answers `invalid choice: 'doctor'`. Run the two commands above (`git pull --ff-only`, then the install line) and run the doctor after that.
- The environment of the launcher (the folder `.plugin-env` of the repo for Cursor and Codex, the plugin data folder for the plugin) follows the lock file by itself: a changed lock changes the stamp, and the next start installs again.
  `sh scripts/topopt-env.sh sync` does the same in a terminal.

### Extras

An extra is an optional group of packages in `pyproject.toml`. None of them is needed to run topopt. The main dependencies hold pyopencl (the OpenCL binding of the GPU solver `gpu-mg`) on an Apple silicon Mac and on Linux x86_64. Quote the brackets in a command: zsh reads `[...]` as a pattern and can stop with `no matches found`.
Join extras with commas in one pair of brackets, for example `".[dev,mcp,gpu]"`.

| Extra | What it gives | Platforms | Install |
|---|---|---|---|
| `dev` | `pytest` and `pytest-xdist` for the tests (see "Tests"). | all | `uv pip install -e ".[dev]"` |
| `mcp` | `mcp==2.3.0`, for the MCP server `topopt-mcp`. On macOS x86_64 it also pins `cryptography==48.0.1`. | all | `uv pip install -e ".[mcp]"` |
| `pardiso` | `pypardiso==0.4.7` (BSD-3-Clause; it pulls in Intel packages under Intel license terms, see "Linux x86_64"): the PARDISO solver (Intel MKL). `auto` uses it when it is installed. | Linux x86_64 only. On another platform the marker leaves the extra empty. | `uv pip install -e ".[pardiso]"` |
| `tet` | `gmsh==4.15.2` (GPL-2.0-or-later): the tet mesher (see "Tet mesher"). | macOS, Linux x86_64 | `uv pip install -e ".[tet]"` |
| `gpu` | `pyopencl==2026.1.4` (MIT): the OpenCL binding of the solver `gpu-mg`. It is a main dependency on an Apple silicon Mac and on Linux x86_64, so you need the extra only on another platform (an Intel Mac) or to add it by hand. | all (no marker) | `uv pip install -e ".[gpu]"` |
| `gpu-test` | `pyopencl==2026.1.4` and, on Linux x86_64, `pocl-binary-distribution==3.0`. It gives a CPU OpenCL device for the tests of `gpu-mg`. A test extra. | pyopencl: macOS, Linux x86_64. pocl: Linux x86_64 only. | `uv pip install -e ".[gpu-test]"` |
| `verify` | `gmsh==4.15.2` (GPL-2.0-or-later), `scikit-fem==12.0.2` and `cholespy==2.2.0` (GPL-2.0-or-later in effect: the wheel bundles CHOLMOD Supernodal): `topopt verify` (see "Verify the result"). | macOS, Linux x86_64 | `uv pip install -e ".[verify]"` |
| `step` | `cadquery-ocp-novtk==7.9.3.1.1` (OpenCASCADE, about 250 MB): `topopt step`, a faceted STEP file (see "STEP file"). The wheel bundles LGPL and GPL-option libraries (THIRD_PARTY_NOTICES.md). | macOS (arm64, x86_64), Linux x86_64 (glibc 2.31 or newer) | `uv pip install -e ".[step]"` |

- The platforms come from the markers in `pyproject.toml`. For `gpu`, `gpu-test` and `verify`, `uv pip compile --no-build` found wheels for Python 3.12 on macOS arm64, macOS x86_64 and Linux x86_64 (2026-10-21).
  The same check for every extra found wheels for Python 3.13 on these three platforms, with the same pins as for 3.12 (2026-10-26).
  Linux arm64 is not in this list: there the core packages have no wheel for the pins (for example `pyamg==5.3.0`), so the same command found no solution for any extra.
- `requirements.lock` and `requirements-macos-x86_64.lock` hold the core, `mcp` and `dev` packages. They hold no package of the other extras.
  The one exception is pyopencl: `requirements.lock` pins it, with the three packages that only it needs (pytools, platformdirs, siphash24). `requirements-macos-x86_64.lock` does not, because the marker of the main dependency leaves out an Intel Mac. CI resolves the pins of `requirements.lock` with wheels only for Apple silicon and Linux x86_64 (`scripts/ci/check_locks.py`).
- [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) has the license of every package, with how it was checked. The core packages are permissive;
  `pardiso`, `step`, `tet` and `verify` bring packages with other terms (the table names them). Install an extra only if its license fits your use.
- gmsh on Linux needs system libraries. See "Tet mesher".

### Check your machine

Two commands show what is wrong with an install and which command fixes it. They only read: they install, build, fetch and delete nothing.
Nobody has run these commands, or `topopt gpu-check`, on a Mac yet (2026-10-21). The Mac lines below come from the code and from tests with a faked platform.
The tools that `gpu-check` calls (`tools/gpu_mg_parity.py`, `tools/bench_gpu_mg.py`) did run on a Mac, an M4 (see the table in "The GPU solver gpu-mg").
Run them in the repo folder:

```
sh scripts/topopt-env.sh doctor
.venv/bin/python -m topopt doctor
```

- `sh scripts/topopt-env.sh doctor` looks at the environment of the launcher: the folder `.plugin-env` in the repo, or the folder that `TOPOPT_ENV_DIR` or the plugin data folder names.
  The Claude Code plugin, Cursor and Codex use it. The command needs only `sh`, so it works when that environment is broken or not there. Without `uv` it says what the setup will download (and gives the install line when no download is possible).
  A plugin user types `/topopt:start-here` instead: its first step runs this command with the plugin data folder.
  It prints one row for each fact: uv, the folder, the stamp, a cut setup, a running setup, and an import test of the packages.
  If the environment Python runs, it then prints the report of `topopt doctor`. It ends with the line `Result:` and the command to run next.
  Exit code 0 means ready. Exit code 1 means an action is needed.
  The command takes about 2.5 s, and about 3.5 s with no Python cache files yet (Linux x86_64, no GPU device, 2026-10-27). It prints a line before the Python report and then the lines of that report as they come, and `topopt doctor` writes one progress line to stderr for each package it tests, so a long wait is no sign of a hang. On a machine with a GPU, it also runs the short self-test of gpu-mg (up to 60 s) until one test has passed on the device and driver.
  If you use only the `.venv` of the checkout, this command says that the launcher has no environment yet. That is a fact about the launcher, not a fault of your `.venv`.
- `topopt doctor` looks at the Python that runs it. Use `.venv/bin/python -m topopt doctor`, or `topopt doctor` after `source .venv/bin/activate`.
  The report has these sections:
  - Checkout: the folder, the revision, the branch, uncommitted changes, and the commits ahead and behind `origin/main` as of the last fetch.
  - Machine: the platform, the Python, the RAM and cores, the solver that `settings.solver: auto` gives, and the OpenCL devices.
  - Packages: each import runs in a child Python, so a cut install shows as a line.
  - Environments: the Pythons that it knows, and which client config starts which one.
  - Problems: printed only when there is one.
  - Offers: facts and options, each with its command. Run a command only after you agree.
  - Next commands: the commands to run, in order.

  An optional package that is not installed, a clone that is behind, and an offer are facts, not problems.
  Exit code 1 means that a core package does not import, that an optional package is installed and does not import, or that the machine facts could not be read.
  `--json` prints the same data as one JSON object. A problem YAML as an argument adds the time estimate.
  The report holds no secret except folder names, so you can send it when you ask for help.
- The MCP tool `environment` gives Claude the machine facts, the `checkout` facts and the same offers. The import tests are in `doctor` only.

On an Apple silicon Mac, the doctor line `solver for settings.solver auto:` says if the Accelerate library works: `cholesky` means yes, and `amg` means no (build it with `sh topopt/native/build.sh`).
With an x86_64 Python on a Mac, `auto` gives `amg` whatever the library.
The line `OpenCL:` says if pyopencl loads and which devices it sees. When pyopencl is installed, the doctor loads the OpenCL drivers to list the devices, as `topopt check` does.

To check the GPU solver `gpu-mg` on your machine, run `topopt gpu-check`. pyopencl comes with topopt on an Apple silicon Mac and on Linux x86_64; on another platform add the `gpu` extra:

```
uv pip install -e ".[dev,mcp,gpu]"
.venv/bin/python -m topopt gpu-check
```

The command needs a checkout, because it runs files of `tools/` and `tests/`. It runs four steps:

1. The facts of the machine and of the OpenCL devices.
2. The parity of one `gpu-mg` solve against a direct solve (`tools/gpu_mg_parity.py`).
3. The gpu tests on the real device (`TOPOPT_TEST_REAL_GPU=1`).
4. A short bench of the hook at 1.2 mm and 1.0 mm, 20 iterations each (`tools/bench_gpu_mg.py`). It takes minutes.

The report is the file `runs/gpu-check/gpu-check.md`. The path starts in the folder where you run the command, and `--out DIR` changes it.
The command prints the same text at the end. Next to it are `gpu-check.json`, a log for each step that ran (`parity.log`, `tests.log`, `bench.log`), and the folder `bench/` when the bench ran.
**`gpu-check.md` is the one file to share.** It holds folder names and no secret.

- `--skip-bench` skips step 4 and `--skip-tests` skips step 3.
- `--device D` names the device: its number in the list that `topopt check` prints, or a part of its name. The default is the first device that is not a CPU.
- `--step-timeout-s S` stops a step that runs longer than S seconds. There is no default limit.
- A step that runs and fails stops the flow, and the report says what to run next. A step that cannot run because a tool is missing is "not run", and the report gives the command that fixes it.
- Exit code 0 means that no step failed and none is "not run". A skipped step does not count.

When step 4 passes, `gpu-check` also saves the constants of your device for the time estimate of `topopt check --solver gpu-mg`: the seconds for each free dof and PCG iteration, and the host part for each free dof.
They go in the file `gpu-constants.json` in the folder `$TOPOPT_STATE_DIR`, or `~/.topopt` when the variable is not set. This is not the report file `gpu-check.json`.
`topopt check` uses the file for the same device and says so. A new `gpu-check` on the same device replaces the old record.

### Clean-room helper for the install tests

An install test must show what a new user sees: no uv, no uv-managed Python, no topopt caches and no client config for topopt. A warm uv cache
changes the first-start time a lot (283 s from nothing, 3 s with a warm cache). `scripts/dev/clean-room.sh` puts a computer into that state, and puts it back.
It is POSIX `sh` (the macOS `/bin/sh` and dash), and it needs no python.

```
sh scripts/dev/clean-room.sh status                 # what a new user would not have; changes nothing; exit 1 when there is something
sh scripts/dev/clean-room.sh hide                   # dry run: says what it would do
sh scripts/dev/clean-room.sh hide --apply           # moves it into $HOME/.topopt-clean-room/<UTC date>/ and writes a manifest
sh scripts/dev/clean-room.sh restore                # dry run
sh scripts/dev/clean-room.sh restore --apply        # puts everything back from the newest backup that is not restored (or --from <backup folder>); exit 1 if it skipped anything
sh scripts/dev/clean-room.sh shell                  # a new-user shell in a temporary home; changes nothing else
```

| Item | Kind | Source of the path |
|---|---|---|
| `~/.local/bin/uv`, `uvx`, `uvw`; `~/.cargo/bin/uv`, `uvx`, `uvw`; any other `uv` on PATH inside your home | moved | the uv installer and cargo |
| The folder of the uv-managed Pythons (`$UV_PYTHON_INSTALL_DIR`, else `$XDG_DATA_HOME/uv/python`, else `~/.local/share/uv/python`), and the links in `~/.local/bin` that point into it | moved | uv |
| The uv cache (`$UV_CACHE_DIR`, else `$XDG_CACHE_HOME/uv`, else `~/.cache/uv`) | moved | uv |
| The registry of live servers (`$TOPOPT_STATE_DIR`, where a leading `~/` is your home, else `~/Library/Application Support/topopt` on macOS, else `$XDG_STATE_HOME/topopt` or `~/.local/state/topopt`) | moved | `topopt/registry.py` |
| `~/.topopt`, the per-user state of gpu-mg (the calibration file `gpu-constants.json` of `topopt gpu-check`), when `TOPOPT_STATE_DIR` is not set; with the variable set, this state is in the folder of the row above | moved | `topopt/gpu_mg/check.py` (`state_dir`) |
| The kernel caches of pyopencl, which the GPU solver uses: `pyopencl` and `pytools` in `$XDG_CACHE_HOME`, else in `~/Library/Caches` on macOS, else in `~/.cache` (warm kernels make the first GPU self-test faster than a new user sees it) | moved | pyopencl (`cache.py`) and pytools (`persistent_dict.py`) |
| `<repo>/.plugin-env` of this checkout (it holds the environment, the downloaded uv and `setup.log`), and the folders named by `TOPOPT_ENV_DIR`, `EVAL_TOPOPT_ENV_DIR` and `CLAUDE_PLUGIN_DATA` | moved | `scripts/topopt-env.sh` |
| `~/.claude/plugins/data/topopt-*` (the plugin data of Claude Code; `$CLAUDE_CONFIG_DIR` replaces `~/.claude`) | moved | `scripts/topopt-env.sh` |
| The Claude Desktop extension folders with `topopt` in the name, in `~/Library/Application Support/Claude/Claude Extensions` and `Claude Extensions Settings` | moved | from reading; not checked on a Mac |
| `uv` from Homebrew | brew-unlink (`brew unlink uv`; restore runs `brew link uv`; never `brew uninstall`) | Homebrew |
| The topopt entry in `~/.cursor/mcp.json`, `~/.gemini/settings.json`, `$CODEX_HOME/config.toml` (`~/.codex`), the VS Code user `mcp.json`, and `claude_desktop_config.json` on macOS | edited-config | `docs/clients.md` |
| A `uv` or a Python outside your home (`/usr/bin/python3`, `/opt/...`, Homebrew folders), the uv tools, the plugin install (`installed_plugins.json`), symbolic-link configs | reported-only | never changed |

- Nothing is deleted. A move is `mv` inside one file system; a backup folder on another file system is refused.
- The client configs also hold your other servers, so a config file is never moved away. `hide` copies the whole file into the backup folder, then writes the file again without the
  topopt entry (an `awk` edit; the script reports the file instead when the edit does not fit). `restore` gives the original back byte for byte, if the file is still the one that `hide` wrote.
  If the client changed it since, `restore` leaves it, says where the original is, and exits 1. Copy the original back by hand (the message gives the `cp` line), or run `restore --apply --keep-changed-configs` to keep the current file (`restore` then marks the backup as restored).
- `restore` never overwrites a path that exists now (a new uv, a new cache from the test). It reports the path, goes on with the rest, lists what is new since `hide`, and exits 1
  with the last line "restore INCOMPLETE". Move the new item away, then run `restore --apply` again. A backup that is fully restored gets a file `restored`.
  If the client wrote the topopt entry into a config again, `restore` still puts the original back when the file minus that entry is the file that `hide` wrote.
- **Restore before the next hide.** `hide --apply` refuses while a backup is not fully restored (in the backup folder in use, in the default one, and in the folders of earlier `hide --backup-dir` runs; the default folder keeps a list `roots` for this), and names it with the restore command. Otherwise the leftovers of the first test would go into a second backup, and the original state would stay hidden.
- The manifest is written ahead: a row goes in before each move or edit. After an interrupt (Ctrl-C) or a crash, `restore --apply` finds every item, and it reports a file in the backup folder that has no row. If `hide` did not reach its end, or skipped an item, the list of new items says so: some of them can be your own items, so do not delete them.
- A config with a topopt entry in a form that the edit does not handle (an inline table or a dotted key in TOML, or an edit that would leave a comma in a bad place in JSON with comments) is reported-only with the manual step, and `hide` does not change it.
  `~/.claude.json` is reported-only when it has a topopt server (`claude mcp remove topopt`).
- The project config files of the checkout (`.cursor/mcp.json`, `.codex/config.toml`, `.mcp.json`) are part of the repo and are not touched.
- `brew unlink uv` takes the Homebrew uv off PATH for all your projects until `restore --apply`; `hide` says so.
- The name of a Claude Desktop extension folder holds the author id of its `.mcpb`. An extension from an older `.mcpb` can have another id than the one that you install in the
  test, so after `restore` two topopt extensions can be side by side. Open Settings, Extensions after `restore`, and keep one.
- The Claude Code plugin install is reported only. For the plugin section of the checklist run `claude plugin uninstall topopt@<marketplace>`, then `claude plugin marketplace remove <marketplace>` (this deletes the plugin data too). `restore` does not install it again.
- Quit the clients and any `claude` or `codex` session before `--apply`.
- `shell` stops (and makes nothing) when leaving out the folders that hold a uv would leave no `sh`, `cat`, `awk` and so on on PATH, for example with a distro uv in `/usr/bin`. Give `--path`, or run `hide --apply` first.
- `shell` makes a temporary folder (`$TMPDIR`, or `--dir`) with HOME and the XDG folders in it, a PATH without uv and without any folder in your home (`--path` sets it by hand), and `npm install -g` into that home.
  Only a few variables go in. `--env NAME` passes one more, for an API key. `--auth-file .codex/auth.json` copies that one file from your home on request. It also sets `TOPOPT_STATE_DIR`, because topopt reads the home of the account database for the registry, not `HOME`.
  The folder stays when you leave; remove it yourself. Use `-- command` to run one command in it.
- Docker (Linux x86_64, CLI clients): `sh scripts/dev/clean-room-docker.sh --env OPENAI_API_KEY` builds `docker/clean-room/Dockerfile` (Ubuntu 24.04, git, curl, Node.js LTS; no uv, no extra Python) and opens a shell
  with a fresh clone of the committed state. No folder of yours is mounted, and no key is in the image. When `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` or `ALL_PROXY`
  (or the lower-case forms) are set, the helper passes them by name to the build and to the container (a host with no direct route needs them); it prints the names, not the values.
  A name that is set in one case only goes in both cases, because apt reads only the lower-case names.

The Mac GUI clients (Claude Desktop, Cursor, VS Code, the Code tab of the desktop app) have two ways. (1) `hide` and `restore` in your own account. (2) A new macOS account
(System Settings, Users & Groups, Add User; the most faithful): it has no uv, no cache and no config, and you delete it after the test. A new account still sees `/opt/homebrew/bin`
(the launcher adds it to PATH when the folder exists). If Homebrew has uv, run `brew unlink uv` first and `brew link uv` after.

### First-run problems

This table lists problems that a first-time user can meet. Each row has the message that the tool prints, the cause and the fix.
Each message is a part of a real output. The test `tests/test_first_run_table.py` finds the message in the code, so a changed message breaks the test.
Look for your message in the first column. The skill `start-here` sends Claude to this table when a tool fails.
If your message is not here, run the doctor (see "Check your machine") and `topopt check <file.yaml>`.

`topopt check` finds the same errors as `topopt run`. It starts no run and it writes no file. The tool `check_problem` gives the same text to Claude.
A `topopt run` that stops on a message about a region, a piece, a closed mesh or a size limit also removes the older result in its `--out` folder.
So run `topopt check` first, or give `--out` a new folder.

| Message or symptom | Cause | Fix |
|---|---|---|
| A call of a topopt tool returns `topopt is still setting up its Python environment.` The server log has `the setup is not ready yet (the first start downloads about 200 MB: 1 to 3 minutes).` The doctor says `a setup is running (pid 436168).` | The first start makes the Python environment and downloads about 200 MB. Claude Code waits only 30 s for the server, so the launcher connects with stored lists and answers a tool call with the setup status. The setup goes on in the background. | Wait the time that the text names (the whole setup takes 1 to 3 minutes). Do not call the tool again at once. Type `/topopt:start-here`, or run `sh scripts/topopt-env.sh status`. It says `ready` when the setup is done. Then the tools work in the same session. |
| The doctor says `an earlier setup was cut or failed.` | A setup stopped before its end: the network failed, the client closed, a process was killed or the power went off. The packages can be incomplete. | Run the command that the doctor prints after `Next command:`. It is the same command with `sync`: it installs the packages again and needs the network (`UV_OFFLINE=1` in front of it works when the uv cache has every file). Then the tools work in the same session. |
| `The uv on PATH is too old: <path> is uv 0.6.7, and topopt needs uv 0.6.8 or newer.` and `The automatic download of uv did not work:` with the reason. A tool call says that `the automatic download of a newer uv failed`. | The first uv on the PATH is older than uv 0.6.8, so the launcher skips it (its setup needs a newer uv), and it could not download its pinned uv either: no network, no `curl` or `wget`, no `sha256sum` or `shasum`, a platform with no pinned uv build, or a download whose SHA256 is not the pinned value. With a working download the setup uses the downloaded uv and says nothing more than one line. | Update that uv with the tool that installed it: `brew upgrade uv`, `pip install --upgrade uv`, or `uv self update` for a uv from the install script. A new uv in `~/.local/bin` does not help while the old one comes first on the PATH. Or remove the cause of the failed download and start again. Then run the `sync` command that the tool call or the doctor gives; the tools work in the same session. |
| `uv is not installed.` and `The automatic download of uv did not work:` with the reason. The doctor row `uv` says `not found:` and `the setup cannot download it here`. | There is no uv on the PATH of the program that starts topopt, and the launcher could not get its pinned uv: no network, no `curl` or `wget`, a platform with no pinned uv build, or a download whose SHA256 is not the pinned value. | Remove the cause that the message gives and run it again, or install uv with the command in the message (or `brew install uv`). The launcher also looks in `~/.local/bin`, `~/.cargo/bin`, `/opt/homebrew/bin` and `/usr/local/bin`. Then run the `sync` command that the status text gives; the tools work in the same session. |
| `the active voxels make 2 separate pieces (voxels that share a face are one piece). 1 of them has no fixed voxel, 1005 voxels in all.` `topopt check` prints it as a warning. `topopt run` stops with it as an error. | A region does not touch the rest of the part. Here the load region lies away from the design space. A piece with no support can move as a rigid body, so the stiffness matrix is singular. A voxel that touches another one only at an edge or a corner is a separate piece. | Move or enlarge the region so that it shares a face with the design space. Or add a fixed region on the piece. The message gives the box and the first voxels of the piece. `topopt render <file.yaml>` draws the regions. |
| `region design has no voxel at voxel_size 1.2. Use a smaller voxel_size or make the region larger` (the name after `region` is the region that failed, for example `load 'tip'`). | No voxel centre is inside the region at this voxel size. Either the region is smaller than a voxel, or the STL files are not in mm. The tool reads the numbers of an STL file as mm: a part drawn in inches is 25.4 times too small, and a part drawn in metres is 1000 times too small. If the voxel size also fits the small part (for example 0.05 mm), there is no message: `topopt check` shows a part of a few mm and a `full block:` mass near 0.0 g. | Run `topopt render <file.yaml>`. It prints the size of each STL file in mm (`bbox_mm`). Export the STL files in mm, or scale them (inches by 25.4, metres by 1000). If the size is right, set a smaller `settings.voxel_size` or make the region larger. |
| `STL is not watertight (not closed)` The message ends with the path of the file. | The mesh has a hole, a missing face or a loose edge, so the tool cannot tell the inside from the outside. | Close the mesh in your CAD or mesh program (fill the holes, export one solid body). Export the STL again. `topopt render <file.yaml>` prints `watertight` for each file. |
| `model has 564298 active voxels, the limit is 100000 (max_active_voxels).` or `grid would have 23388750 voxels at voxel_size 0.2, the limit is 5000000 (max_grid_voxels).` | The voxel size is too small for the part. The memory and the time grow fast with the number of voxels (the hook needs about 3 GB at 1.0 mm). The guard stops the run before it uses the memory. | Use the larger `settings.voxel_size` that the message gives (`Try voxel_size 0.92 or more`). To keep the fine grid, raise `settings.max_active_voxels` or `settings.max_grid_voxels`, but only when the memory estimate of `topopt check` is below your RAM. |
| `load region 'tip' is not joined to a fixed, spring or joint region in the thresholded design (density 0.5) at beta 8.` The run prints it. `topopt printcheck` on the run folder (the tool `print_check`) then says `2 bodies; the limit is 1 (assumed).` | The volume fraction is too low for the loads. The material cannot make one path from the load region to the fixed region, so the result breaks into pieces. The message at the last pass counts. At a low beta, the same message can show a grey design that is not the final part. | Raise `settings.volume_fraction` and run again. Example: the hook at `voxel_size: 2.0` gave 2 bodies at 0.02 and 1 body at 0.04. Look at the check `bodies` of `topopt printcheck`. For a loose body at a normal fraction, see [Cleanup list](verify.md#cleanup-list-find-spurs-bumps-and-loose-bodies). |

### The GPU solver gpu-mg

`gpu-mg` solves the linear system on an OpenCL device, for example the GPU of a Mac, with a matrix-free multigrid. It is the main solver on a machine with a GPU. The CPU solvers (`cholesky`, `pardiso`, `amg`) are the fallback.

`settings.solver: auto` (the default) runs `gpu-mg` when all of these hold:

- pyopencl loads, and OpenCL shows a GPU (`settings.gpu_device` can name another device; a CPU device such as pocl does not count for `auto`), with the features of the engine.
- A short self-test passes: one small cantilever is solved with `gpu-mg` on the device and with a direct solver on the host, and the displacements must agree to a relative difference of 1e-6. A pass is cached for each device and driver version (and for each topopt version and each state of the gpu-mg code) in the file `gpu-selftest.json` in the topopt state folder (`~/.topopt`, or the folder of `TOPOPT_STATE_DIR`). The first time on a device and driver it takes a few seconds, because the driver builds the kernels (9.8 s on an Intel Arc GPU, 2.6 s on a CPU OpenCL device on a busy host). After a pass, the next processes read the file and run no test; `topopt check` says "cached result of <UTC time>" in place of the seconds. A failed test, an error and a timeout are never cached, so the next process tests again. A file that cannot be read or written is no error: the test runs, and `topopt check` says why the cache was not used. The device list and the test run in a child process, and the parent process waits 60 s for it. If the child does not finish, it is stopped, and `auto` runs the CPU solver; the line `solver choice:` names the limit, and `opencl.problem` of `environment` says that the device list did not finish. To skip the wait, set `settings.solver` to a solver name, or set `TOPOPT_AUTO_GPU=0` in the environment (then `check` and `environment` also ask no device list).
- The YAML has no feature that `gpu-mg` cannot run (today the buckling row).

If one does not hold, `auto` runs the CPU solver of the machine, and `topopt check` (the line `solver choice:`) and `run.log` (the line `solver: auto -> ...`) say why.
If `gpu-mg` fails during a run (an error of the device, no device memory, or a solve that is not accepted), the run goes on with the CPU solver from its last checkpoint.
The line in `run.log` says the pass, the kind of failure and the CPU solver. `run.json` has `solver_parts` (which solver ran which passes) and `solver_setting`, and the verdict has one sentence about it.
If the CPU solver does not fit in the RAM of the machine, the run stops with the facts and keeps its checkpoint.

To ask for `gpu-mg`, or for another solver, there are two ways:

- In the YAML file:

  ```yaml
  settings:
    solver: gpu-mg
  ```

- On the command line: `topopt run <file.yaml> --out <dir> --solver gpu-mg`, and `topopt check <file.yaml> --solver gpu-mg` for the estimate.
  The MCP tool `run_topopt` has no solver argument: write `solver: gpu-mg` under `settings` in the YAML.

An explicit `gpu-mg` runs on any OpenCL device, also a CPU device, and it never falls back: a failure stops the run with the facts and the options, as before.
pyopencl is a main dependency on an Apple silicon Mac and on Linux x86_64. On another platform (an Intel Mac) add the `gpu` extra. A fine grid also needs a higher `max_active_voxels` under `settings` (the hook at 0.8 mm or finer does).
The doctor and the MCP tool `environment` give the facts: the key `gpu_solver` says if `auto` runs `gpu-mg` here, with which device, and when it does not, why and what would make a GPU usable.

`topopt check --solver gpu-mg` gives its own time estimate. The model is linear in the free dofs. It is a fit of 49 finished runs on one Intel Arc iGPU, so the number is a mid value with a band (low to high), not a limit: the band is the spread of the PCG iteration counts that were seen. `check` assumes that no other `gpu-mg` run is on the device. How close it is: on the 18 runs of the fit with a recorded `penal` (the case that `check` has), the mid number was 0.46 to 1.29 times the wall time (median 0.75; 8 runs within 25 %, 12 within 50 %). The other 31 runs of the 49 have no recorded `penal` and are not a test of `check`. A GPU with no constants of its own (for example a Mac GPU) gets the numbers of the Intel Arc iGPU, named as a reference, with the line "no constants for this device; run `topopt gpu-check` to measure them". A CPU OpenCL device with no constants gets no seconds at all: the Intel numbers are 2.5 to 5 times too low for it. The first `time estimate:` line of `check` and `derived.time_model` (in `run.json` and in `check_problem`) say whose constants the seconds are. After `topopt gpu-check` the estimate uses the constants of your machine.

Three optional numbers under `settings` decide when a `gpu-mg` solve is done: `gpu_tol` (default `1e-6`, the relative residual of each solve in the
optimizer loop), `gpu_backward_error` (default `1e-15`; `0` turns the rule off) and `gpu_max_iter` (default `1000`, an integer: the PCG iterations of the first pass of a solve in
case B; a restart step keeps its own cap of 300, and case C has no first pass). On a design with a near-void start (a low `volume_fraction` with a high `penal`),
the FP64 residual of any solver stops above `1e-8`; a solve whose backward error is at most `gpu_backward_error` is accepted there, and `run.log` says so once.
On a design with a high stiffness contrast the first pass can need several hundred iterations (a run with 1.35M free dofs: 575 PCG iterations for each solve on average, first pass and restart steps together); a first pass that runs out goes on in restart steps.
With an explicit `gpu-mg`, a solve that is not accepted stops the run with its facts and options, and `gpu_mg.jsonl` has its residual history. With `auto`, the run goes on with the CPU solver. Details: CONTRACTS.md, section 20.

`gpu-mg` takes a YAML with springs, with joints and with per-axis supports (a `fixed` entry with `axes`: a roller or a slider), alone or together, also a part that
joints alone hold (no `fixed` region).
A joint adds a term of rank 6 for each joint to the operator on every level of the multigrid. On a device without 64-bit float (case C), the
joint term is computed in 32-bit float, and the inner PCG removes the rigid modes of each body and 6 harmonic columns for each bonded patch (the rigid motions of the patch,
extended elastically into the part for each design: a 64-bit loop on the host with the 32-bit V-cycle of the device as preconditioner), and solves them in 64-bit on the host (a method setting: no switch; `run.log` says that it is on, with the time of the extension).
`check` and `run.log` print the beam over the bond stiffness on its softest mode, with the cases that were measured
on pocl (a cube that only a joint holds solved at 9.9e-12 and above with the rigid modes removed, and at 3.3e-7 and above without; the hook, with a fixed region and a joint, solved at 9.8e-9;
a part with two tube joints and no fixed region, 600,771 free dofs, stopped with the rigid modes alone and solved with the harmonic columns). The columns take memory (144 bytes for each free dof on the device and 624 on the host for 18 columns).
The memory estimate of `check` does not count them, and it is far below the measured peak also in case B, which has no column. For that two-joint part the estimate is 0.69 GB at 2.2 mm and 1.21 GB at 1.0 mm;
the peak of `topopt run` on pocl, one host, was 2.5 GB and 3.5 GB in case C (the run at 1.0 mm was cut at 1,500 s, so this is the peak up to the cut) and 2.6 GB and 3.7 GB in case B.
The columns are 0.06 GB and 0.46 GB of the case C peak. The cause of the rest of the gap is not known.
The figures are facts, not a limit. If a run stops, its error gives the ratio and the options: another solver, case B on a device with 64-bit float, or a stiffer beam
(a larger section, a shorter beam). The joint and spring runs were measured on pocl (a CPU OpenCL device) and on one Intel Arc iGPU, one host each; Apple silicon is not measured.
Details: CONTRACTS.md, sections 20, 35 and 38.

Measured on **one machine** (a MacBook Air M4, 16 GB), on the hook example, 20 iterations, one run for each size. The numbers are the median seconds for one optimizer iteration:

| Voxel size | Free dofs | Default solver (Accelerate Cholesky) | `gpu-mg` |
|---|---|---|---|
| 1.2 mm | 115,836 | 0.765 s | 0.287 s |
| 1.0 mm | 205,485 | 2.235 s | 0.439 s |
| 0.8 mm | 386,142 | 9.241 s | 0.658 s |

The final compliance agreed to 1e-10 or better in each row. These figures were measured at the target 1e-8, before `gpu_tol` (default `1e-6`) and `gpu_backward_error` existed.
At 0.6 mm (905,010 free dofs), `gpu-mg` ran at 1.919 s for one iteration. The default solver did not run there: its memory estimate (22.5 GiB) is above the 16 GiB of that machine, and the bench script skips such a run. `topopt run` only warns when the estimate is above the RAM.
Another Mac or another GPU can give other numbers. `topopt gpu-check` measures the machine it runs on. The details are in [CONTRACTS.md](../../CONTRACTS.md), section 20.
