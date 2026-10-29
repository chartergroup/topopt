# Changelog

This file lists the changes of each release of topopt.
The newest release is first. There is one entry for each release tag (`vX.Y.Z`).
The release workflow writes each entry: one line for each merged pull request, taken from the PR title
(`scripts/ci/release_notes.py`, see [RELEASING.md](RELEASING.md)).
The install command in [README.md](README.md) uses the branch `latest`, which points to the newest release.
[RELEASING.md](RELEASING.md) tells how a maintainer makes a release.

<!-- version list -->

## v0.5.0 (2026-10-29)

_This release is published under the MIT License._

### Features

- Show the problem as a 3D view in the chat.
- Add a footbridge example.
- Compare a base run with its variants in one table.
- Name the core tools first in the server instructions.
- Refuse an unknown argument name on every tool.
- Make Python 3.13, the Python that CI tests.
- Add the finishing role and call it after each successful run.
- Add a 5-inch drone-arm example.
- Apply the density filter as an FFT convolution, so fine grids are not limited by its memory.
- Add the wall rule, settings.robust: wall.

### Bug Fixes

- Record every mesh fallback of every calibration round.
- One label for load cases with arrows at the same place, past the arrow's tail.
- Stop the run when the part-only cut or the stress report runs out of memory or threads.
- Explain voxel steps on a finished part as a known limit, with the routes and their price.
- Say that the process has stopped for a CLI run that ended natively in the post step.
- Tell the agent that nothing calls it back when a run ends.
- End the run with an error when the mesh step runs out of memory or threads.
- Draw one box for each connected fixed region.
- Name a rising load case and per-axis supports in a failed solve.
- Hold the distance queries of the sdf step to a pairs budget.
- Limit the sdf field to the envelope before marching cubes.
- Show progress while doctor runs its Python step.
- More calibration rounds when the mesh misses the mass target by 0.1 %.
- A plugin update links the code offline, with the setuptools of the lock.
- Serve three.js from the package, so the refine page works offline.
- The server instructions fit the 2048 cut of older Claude Code; run_result states a pass-limit stop.
- Default the envelope fill to off so a run keeps its stiffness at the same mass.
- The setup status names a wait from the download progress.
- Measure the thickness of a region of several bodies the same on Linux and macOS.

### Documentation

- The platform of eight measurement files is Linux x86_64, not the host kernel string.
- Show the lead line as a heading and drop the link to the MP4 clip.
- Show a footbridge run in the hero clip and a fresh hook run in the pictures.
- Add a guide of prompts that change the hook example with words.
- Start the README with what topopt is, the install for each client and the examples.
- Add an examples gallery that the README links to.
- Start-here names the oldest uv and offers the update of an old uv.
- One setup guide for any MCP client, then short tips per client.
- Cursor user entry: name its condition, the work folder, the README row and the contract.
- The Code tab of Claude Desktop adds the plugin through Settings > Plugins.
- Cursor: use a user entry in ~/.cursor/mcp.json when the project server stays Disabled.
- Say what the install downloads and that a run sends nothing.
- The Desktop extension starts the launcher; a uv on PATH must be 0.6.8 or newer.

### Chores

- Clean-room helper for the install tests.

### Continuous Integration

- Read a squash merge that kept the merge message.
- Attach the Desktop extension also as topopt.mcpb, a name without the version.

### Testing

- Add the eval case finish-after-run for the finishing pass.

### Build System

- **Breaking:** Require Python 3.13 to install topopt.

### Refactoring

- Move the patterns and the blind-test note of the part skill to files read on demand.
- Move the reply key lines of run_result, environment and run_topopt into the key guides.

## Earlier versions

Versions 0.1.0 to 0.4.0 were released before this repository was public. Their entries are not listed here.
