# The dashboard, the MCP server and the render

[Back to the README](../../README.md)

## The dashboard

`topopt run ... --dashboard` starts the dashboard in the same process.
You can also show any run folder:

```
topopt dashboard runs/hook
```

The page shows the 3D shape, the log, and a bottom bar.
The optimizer code with the active step shows only on the demo screen: for a run with a load case named `pull_off`, or when the address has `?demo=1`.
The bar shows the iteration, the time, the volume, and the compliance.
From pass 2 it also shows "about N s left" under the time. `run_status` (`time_left`) and `run.log` (one `fact: time:` line) give the same number, with the bound "up to M s"; it holds if no stop rule or request to stop ends the run earlier. In experiment E1 (run-time study; 585 runs of 10,000 free dofs and more on one shared Linux host; no Mac run) the remaining time after pass 2 was within a factor 1.17 (median) and 1.64 (9 of 10 runs) of this number. `topopt check` keeps the start estimate. Details: CONTRACTS.md, section 24.5.
At the end the page shows the final mesh and the result numbers.
The default port is 8765. Use `--port` to change it. Use `--no-open` to keep the browser closed.
If the port is busy, `topopt dashboard` stops with `error: dashboard port 8765 is busy` (exit code 1) and names the `--port` option.
The dashboard works with no internet.

## Refine

At the end of a run, `topopt run` makes `result/refine.html`: the final part with a stress overlay and a tool to place add, remove and smooth marks.
The dashboard result panel shows a "Refine" button when the page exists.
With `--dashboard`, the browser also opens the page after the run.
To make the page for an old run: `topopt refine <run_dir>`.
The page holds its own copy of three.js and needs no network. See [topopt/refine/README.md](../../topopt/refine/README.md).

## MCP server

The MCP server lets Claude start and watch a run.
The steps to enable it are in [topopt_mcp/README.md](../../topopt_mcp/README.md).
The client config starts the command `topopt-mcp` (installed with `pip install -e ".[mcp]"`).
It must be on the PATH of the client, so activate the environment before `claude`.
Example `.mcp.json`: `{"mcpServers": {"topopt": {"type": "stdio", "command": "topopt-mcp", "args": []}}}`.
The server writes into the folder where the client starts it (`runs/live`). To use another folder, start the client there, or set
`TOPOPT_WORKSPACE` in the `env` of this config. [The work folder](install.md#the-work-folder) has the steps for each client.

Claude Code before v2.1.296 cuts the server instructions and each tool description at 2,048 characters.
Newer versions allow 4,096 characters (16,384 for a tool that loads through tool search).
The whole server instructions text is at most 2,048 characters (`tests/test_plugin_manifest.py` fails above that), so the older client reads all of it.
The key text of each tool description is in its first 2,048 characters, and the details follow.
To see the full texts on an older client, set `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH=4096` before you start `claude`
(it needs v2.1.280 or later; the number replaces both limits).
The limits are from the Claude Code MCP docs (code.claude.com/docs/en/mcp, "Tool search for MCP server authors").
The cut at 2,048 characters was measured with Claude Code 2.1.293, on one host.

The tool `environment` gives facts about the machine: the topopt and Python versions, the solver
backends found (Accelerate, pardiso, AMG) and what `settings.solver: auto` means here, RAM, cores,
the thread limits and, if you pass a problem YAML, the time estimate for each backend.
When a faster backend is possible but not installed (`pypardiso` on Linux, the Accelerate library
on a Mac), the answer holds an offer: the reason, the speed-up, the command and the license note.
The key `gpu_solver` says if `settings.solver: auto` runs the GPU solver `gpu-mg` here, and what would make a GPU usable (see [The GPU solver gpu-mg](install.md#the-gpu-solver-gpu-mg)). It is a fact, not an offer.
The key `checkout` gives the git facts of the code folder: the revision, and how far it is behind `origin/main` as of the last fetch.
`topopt doctor` prints the same facts and the offers in one report.
The tool installs and builds nothing. Claude shows you the offer and runs the command only
after you agree. `topopt --version` prints the version.

## Preferences (saved once)

The first time, Claude asks you the model of your printer (only the name, for example "Bambu P1S"; topopt finds the print bed and the printer type), the filaments that you have (only the names), the material, the quality (the voxel size) and the outputs that you want
after a run. It keeps your answers in one file in the work folder: `topopt-preferences.yaml`. Later sessions read the file and do not ask again.
The server reads the file at each call, so a change is live at once (no restart; in Claude Desktop the setting of the extension still
needs a full quit).

```yaml
format: 1
print: {printer_model: bambu-lab-p1s}               # the names of the print block of the problem YAML (no print limits); the model gives the bed and the type
material: {E: 2100, nu: 0.35, density: 1.27, name: PETG}   # or filament: <id> (see "Your filament"); the filament gives E and density
filaments: {available: [bambu-lab-pla-basic]}       # the filaments that you have; printer_materials marks them
quality: {voxel_size: 1.5}                          # mm; a smaller voxel is finer and slower
outputs: [renders, step]                            # renders (render_part), step (export_step), verify (verify_run)
```

- Tell Claude a new answer ("my bed is now 300 mm"): it calls the tool `save_preferences`. Or edit the file by hand. There is no command
  for it in the CLI: edit the file. Every section is optional. A key that topopt does not know stays in the file.
  The tool rewrites the file, so a comment that you add by hand is not kept. A save replaces the file: if the name is a link, the link
  is replaced by a plain file, and the file that it pointed to is not written.
- A preference does not change a run by itself. `topopt run`, `topopt check`, `run_topopt` and `check_problem` do not read the file: the
  problem YAML is the full record of a run. `build_regions` takes a material, a print block or a voxel size from the file only when the
  spec of your request has none, writes it into the YAML and says so. `check_problem` says when a YAML differs from your preferences.
- `environment` shows the file in the key `preferences`. A file with a mistake is a fact there (`problems`), not a crash.

## Render (look at a part)

`topopt render` draws STL files or a run folder as PNG images. It needs no display and no OpenGL.

```
topopt render runs/live                                  # final.stl and the regions, in runs/live/renders/
topopt render part.stl plate.stl --role part --role fixed --section "x<=30"
topopt render part.stl plate.stl --role part,fixed       # a comma list is the same as repeated flags
topopt render problem.yaml                               # before a run: the regions, the interface shapes and the force arrows, in renders/ beside the YAML
topopt render problem.yaml --no-loads                    # the same, with no force arrows
```

It writes one PNG for each view (iso, front, right, top) and one labeled montage, and prints the facts as JSON:
sizes, bbox, triangles, watertight, the share of back faces, missing files, the font and the time.
Colors: part grey-blue, back faces magenta (a flipped or open surface), cut faces flat orange. The design space and
the regions (fixed, load, keep-out) are drawn as ghosts, so they do not hide the part (`--regions solid` or `off` changes this).
`--section x=30` keeps x >= 30 and `--section "x<=30"` keeps x <= 30. With a section the default views are `cut`
(straight at the cut face), `iso-cut` and two side views. All tiles have one scale, with a scale bar. The MCP tool `render_part` gives the montage to Claude as an image
(`inline=false` gives the JSON only). See CONTRACTS.md section 15. The 4-view render of the hook takes 2 to 4 s.
Render uses Pillow (license MIT-CMU), pinned like the other dependencies.

## Open the dashboard before the run

Use this order when you want the page open before the run starts, for example on a second screen. It works for any problem YAML.
The commands show the hook as the example. Give your own YAML in place of `examples/hook/hook.yaml`.

1. Make sure no other program uses port 8765.
2. Start the dashboard first, before the run: `topopt dashboard runs/live`.
   The folder `runs/live` does not need to exist. The page says "Waiting for a run..." until a run starts.
3. Open `http://127.0.0.1:8765/` in the browser if it did not open.
4. Start a run into the same folder. Use the MCP tool `run_topopt` (without `out_dir` it writes to `runs/live` in the work folder), or this command:
   `topopt run examples/hook/hook.yaml --out runs/live --dashboard`.
5. The dashboard resets and shows the new run. It needs no page reload.

If port 8765 is busy, `topopt run --dashboard` uses the next free port. It prints `Dashboard port 8765 is busy. Port 8766 is used.`
and the new URL, and the run goes on. The MCP tool `run_topopt` gives that URL in `dashboard_url`. A dashboard that is already open
on the same run folder (step 2) follows the new run, so its page on port 8765 shows the run too.
The MCP server reads the port from the environment variable `TOPOPT_PORT` (default 8765).
Use the same port for the dashboard and for the MCP server.

## Replay a recorded run

A replay shows a finished run folder as if it was live (the folder format is in CONTRACTS.md, section 5). Any finished run folder works. A clone can hold recorded runs in `recordings/`. Start a replay in a terminal:

```
topopt dashboard <run folder> --replay
```

Give the folder of the run in place of `<run folder>`.
The terminal prints `Replay ready: press Enter to start`.
The page waits. Press Enter to start the clock. Press Enter again to start over.
Use `--speed 2` to run the clock faster.
The page shows a small REPLAY label. Use `--no-badge` to hide it.
A replay writes nothing to the run folder. The step times in the code panel (it shows only for a `pull_off` case or with `?demo=1`) are not recorded, so a replay shows equal shares of the time between two iterations.
See [recordings/README.md](../../recordings/README.md).
