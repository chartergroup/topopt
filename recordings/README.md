# Recordings

This folder holds recorded runs for the replay of the dashboard. A clone may hold none.
A recording is a copy of a finished run folder (CONTRACTS.md section 5).
The replay shows it on a clock, as if it was live. It is a recording, not a new run.

The `yaml_path` of its `run.json` is relative to the run folder (`../../examples/hook/hook.yaml` for a recording of the hook example), so `topopt refine`,
`topopt buckling` and `topopt verify` find the YAML of the example from any current folder, and no user name is in the file.
Keep that tree (`recordings/<run>` next to `examples/`), or give the YAML with `--yaml FILE`.

**Do not run `refine`, `buckling` or `verify` on the recording itself.** They write into the run folder (`result/refine.html`,
`result/stress.npz`, the key `buckling` of `result/metrics.json`, `verify/`), and the recording is committed. Copy it first to
`runs/<name>`. That folder is not tracked, it has the same depth as `recordings/<run>`, so the relative `yaml_path` still works:

```
mkdir -p runs
cp -R recordings/<run> runs/hook_replay
topopt refine runs/hook_replay
topopt buckling runs/hook_replay
topopt verify runs/hook_replay
```

Run them in the repo folder (`runs/` may not exist yet: `mkdir -p runs`). The replay (below) only reads the recording, so it needs no copy.

## Add a recording

A new recording is a copy of a finished run folder. `topopt run` writes the absolute `yaml_path` of the machine that made the run
(CONTRACTS.md section 5), so make the copy portable by hand:

1. Copy the run folder to `recordings/<name>` (the YAML and its STL files go to `examples/<case>/`, the same repo tree).
2. In `recordings/<name>/run.json`, set `yaml_path` to the path of the YAML relative to the run folder, for example
   `../../examples/hook/hook.yaml`. A relative path starts at the run folder, not at the current folder.
3. Remove every other absolute path from the copy: look at `run.json`, `result/metrics.json`, `run.log`, `history.jsonl` and, if the
   run has them, `job.json` and the `result/*.html` pages (`refine.html` holds the path of the source STL). Replace a path with a
   relative one or take the key out.
4. Run `pytest tests/test_run_yaml_path.py`: the test `test_no_tracked_file_of_recordings_or_examples_holds_a_home_path` reads every
   tracked text file under `recordings/` and `examples/` and fails if one holds the start of a home folder (macOS, Linux or Windows;
   the three texts are `HOME_PATHS` in the test). This README must not hold them either. The test reads the tracked files, so run
   `git add` for the new folder first. Check that `topopt refine` finds the YAML on a copy of the folder (see above).

There is no script for this step, and `topopt run` does not write a relative path.

## Replay a recording

Run it in the repo folder. Open the dashboard page in the browser first. Then type this in a terminal:

```
.venv/bin/python -m topopt dashboard recordings/<run> --replay
```

The terminal prints `Replay ready: press Enter to start`. The page shows "Waiting for a run...".
Press Enter to start the clock. Press Enter again to start over from 0.

Options:

- `--speed 2.0` makes the clock faster.
- `--no-badge` hides the small REPLAY label.
- `--port 8765` is the default port.
