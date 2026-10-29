# Evals of the topopt plugin

These evals test how Claude uses the plugin: the skills, the MCP tools and the
order of the steps. They do not test the numbers of the engine. The engine
numbers stay in `tests/`, `ci/` and the golden files.

The runner is `claude plugin eval` (Claude Code 2.1.269 or later). Reference:
<https://code.claude.com/docs/en/plugin-evals>.

## Levels

| Level | What it tests | Where |
|---|---|---|
| 1. Engine | Solver, filters, post-process, numbers | `tests/`, `ci/` (not here) |
| 2. Agent behaviour | The tools Claude calls, the order, what it tells the user | The cases below, on MCP mocks |
| 3. End to end | One real job through the real MCP server | `hook-e2e` (tag `e2e`) |

## Cases

The user prompts are in Russian (the case `stl-own-part` has an English prompt). Each case has a grader on the result (the
reply) and a grader on the steps (the tool calls).

| Case | Tags | The user says | Claude must |
|---|---|---|---|
| `hook-from-words` | smoke, mock, flow | A T-shirt hook in words, run it | `build_regions` before `run_topopt`; report the mass, the stiffness and `final.stl` |
| `bracket-no-load` | smoke, mock, ask | A wall bracket, no load | Not call `run_topopt`; ask about the load |
| `units-kg` | smoke, mock, loads | A 5 kg plant pot, prepare only | Put about 49 N in the `build_regions` spec; say 5 kg -> 49 N; not run |
| `user-files` | mock, flow, files | Own YAML and STLs (`files/`) | `check_problem` before `run_topopt`; no `build_regions` |
| `stl-own-part` | mock, flow, files | Own STL of a plate (`files/bracket.stl`), prepare only | `find_features` before `build_regions`; an `stl` item with `screw_holes` in the spec; no `run_topopt` |
| `voxel-guard` | smoke, mock, guard | Voxel 0.3 mm, not coarser | Read the guard stop and the memory estimate; raise the guard (say the RAM, the estimate, the freeze risk, what, why, the price) or offer a larger voxel with its price and ask first |
| `lost-force` | smoke, mock, guard, files | Own YAML, load region on the wall plate | Report the lost force (20 %); not run |
| `faster-backend` | smoke, mock, environment | How fast? Faster? | Call `environment`; give the reason and the speed-up of gpu-mg; install nothing |
| `print-after-run` | smoke, mock, print | Can I print it? (after a run) | Call `print_check`; give its facts; say that it does not check the strength |
| `off-topic` | smoke, mock, negative | E-steps calibration | No topopt tool; a correct e-steps answer |
| `render-look` | smoke, mock, flow, render | A T-shirt hook in words, run it (no word about a picture) | `render_part` with `yaml_path` before `run_topopt`; `render_part` of the result after it; say in a line or two what the render shows |
| `sketch-no-size` | smoke, mock, ask | A sketch of a wall hook (`files/sketch.png`) with no dimension, run it | Not call `run_topopt`; ask for one real size or a known object for the scale |
| `finish-after-run` | smoke, mock, flow, finish | A T-shirt hook in words, run it (no word about a finish) | After `run_topopt`: `render_part` of the result with the views `front` and `fl2:-1.333,-1,0.373` and `regions` `off`, and `print_check`; report the finishing pass (what it saw, what it changed or that nothing needed a change, the price of a change) and the print check after it; not call the part certified |
| `no-later-report` | smoke, mock, flow, run | A T-shirt hook in words, run it and tell the result (no word about waiting) | After `run_topopt`: no promise of a later report (nothing calls the agent back when the run ends); report the result in the same turn, or end with a plain line that tells the user what to ask |
| `voxel-steps` | mock, explain, finish | After the drone-arm example: why does my part have steps? | Explain the steps as a known limit of the voxel method in this version, not the fault of the input; one voxel high and already in the numbers of the run; the fill (`settings.envelope_fill`) already ran in this example, so the treads are on member surfaces; a smooth mark and `fine_from` with the price of each; a better smoothing of member surfaces is planned, with no date |
| `hook-e2e` | e2e | A T-shirt hook, a fast trial run | Real job: `run_topopt` then `run_result`; report the result |

## Mocks

`mocks/topopt/` holds the suite mocks of the server `topopt`:

- `_tools.json`: the real `tools/list` answer. The mocked tools get the real
  descriptions and input schemas from it.
- `fixtures/<tool>.json`: real answers of the server for one short T-shirt
  hook job (voxel 2 mm, 40 passes): `build_regions`, `record_brief`,
  `check_problem`, `render_part_yaml` (the render of the problem YAML before
  the run), `run_topopt`, `run_status` (state `done`), `run_result`,
  `render_part_run` (the render of the run folder after the run),
  `print_check` and `environment`. The two render fixtures are recorded with
  `inline=false`: the facts only, because a mock answers text only. They hold
  placeholder paths (`/Users/me/...`), not the paths of the machine that
  recorded them (the `font` path is the one of the recording host: a Linux host
  gives a DejaVu path). The `checkout` block (the git state of the recording
  checkout) is removed from `environment.json`.
- `<tool>.md`: one mock for each tool. Some mocks have a light `expect:` guard
  (for example `yaml_path` must end in `.yaml`). A call that breaks a guard
  stops the run with score 0.
- A tool with no mock file is not available to Claude in a mocked run.

Two kinds of mock:

- `environment` is `type: fixed`: it returns the real answer as it is. Its
  answer does not depend on the input.
- `build_regions`, `record_brief`, `check_problem`, `render_part`, `run_topopt`,
  `run_status`, `run_result` and `print_check` are `type: agent`. The judge
  model answers as the server: in the shape of the real answer, with the
  names, loads and paths of the call. Reason: Claude writes its own spec. A
  fixed answer for another spec (other loads, another material) is a server
  error for Claude, and Claude stops the job. That is correct behaviour, but
  the case then tests nothing. The agent mocks cost a few judge calls for each
  run.

Some cases have their own `mocks/topopt/` folder. A file there replaces the
suite mock of the same tool for that case only:

- `voxel-guard`: `build_regions` (agent) gives the real grid guard stop at
  voxel 0.3 mm word for word, with the names of the call. A second call with a
  larger voxel or a raised guard builds, with a memory estimate to read
  against the RAM. The rule of the case: a guard is a method setting (the agent
  decides with the facts); the voxel size that the user gave is a requirement (a
  larger voxel needs the user's answer first).
- `lost-force`: `check_problem` (fixed) is the real check of
  `files/shelf_bracket.yaml`: 20 % of the force is lost.
- `print-after-run`: `print_check` (agent) gives the real print check with the
  facts of the part in `history.jsonl`.
- `stl-own-part`: `find_features` (fixed) is the real answer for `files/bracket.stl`
  (two holes, H1 d 4 mm and H2 d 6 mm). The suite has no mock of this tool.
- `no-later-report`: `run_status` (agent) answers `running` for the first two calls and `done` from the third.
- `voxel-steps`: `run_result` (agent) answers with the finished drone-arm run, whose `metrics.mesh.envelope_fill` has `ran` true.

`finish-after-run` has no mock of its own and no history. It runs a fresh prompt on the suite mocks. The mocked part needs no
edit, so the tools `cleanup_list`, `apply_marks` and `ratchet_run` (no suite mock) are not needed. `run_result` of the suite has no
`print_check`, so the agent calls `print_check`. The case does not pin `team_guide` or a sub-agent: the role leaves the choice
to the agent.

`no-later-report` has no history. It runs a fresh prompt. Its own `mocks/topopt/run_status.md` (agent) answers `running` for the first two
calls and `done` from the third, so the agent can keep calling `run_status` in the same turn or end the turn. The mock of `run_topopt` has the
`message` of the real reply (nothing calls the agent back when the run ends). The regex grader `no-promise` (`flags: i`, a JavaScript pattern)
fails a promise of a later report in English or Russian; `calls-run-topopt` needs the run; the `llm` grader passes a report of the result or
a plain line that tells the user what to ask.

`voxel-steps` has no history. It runs a fresh prompt. Its own `mocks/topopt/run_result.md` (agent) answers with the finished run of the drone-arm example (`fixtures/run_result.json`: the numbers of `examples/drone_arm/README.md`, and `metrics.mesh.envelope_fill` with `ran` true, as the YAML of the example sets `envelope_fill: on`); the other tools use the suite mocks. The message of the user gives the finished run of the drone-arm example and what the render shows, so the case tests the explanation, not the run flow. The suite has no `team_guide` mock and the case allows no sub-agent, so the role `finish` does not reach the agent: the case tests step 5 of `start-here` only. The `llm` grader `explains-the-limit` needs the facts of the paragraph (known limit and not the input, one voxel high and in the numbers, the routes with price including `settings.envelope_fill` for the faces of the design space, planned with no date); the same grader fails a date, a version, a month, "soon" or a promise of a release for the better smoothing (there is no regex grader for it: a pattern false-hit good replies). The case has no `smoke` tag, so the smoke command does not run it: run it by name: `claude plugin eval . --case voxel-steps --tag mock --ablation none --runs 1 --no-publish --trust-plugin --scaffold`, or with the full run (`--tag mock`). A real run of the case needs the credentials of the runner (Claude account and judge model); the texts and the graders are tested without a model (`tests/test_voxel_steps_text.py`).

`print-after-run` continues an earlier conversation: `history.jsonl` is a
mocked `hook-from-words` run (with no `print_check` tool), made small with
`tools/make_history.py`. That script keeps only the user and assistant
messages and removes the local paths and the account facts.

CAUTION: each run of `print-after-run` writes its session beside
`history.jsonl` (`<session id>.jsonl` and a folder with the same name). These
files hold account facts (the e-mail address and the organization). Git
ignores them. Delete them after a run.

## User files (`--scaffold`)

`user-files` and `lost-force` start with the user's YAML and STL files in
`files/` of the workspace. `sketch-no-size` starts with one picture,
`files/sketch.png` (Claude reads it with `Read`). Their `stage-files.sh` copies the
`files/` folder of the case into the workspace. The runner starts it only with
`--scaffold`. Without `--scaffold` these three cases find no files and fail.
The sketch is drawn by `tools/make_sketch.py` (Pillow): a wall hook in side view
with no number and no known object, so the case tests that Claude asks for the
scale.
`context.add_dirs` is not used: it gives Claude read access to a folder of the
case, but the run does not tell Claude the path of that folder.

## Run the evals

Run the commands in the repo root. Each run is a real model call on your
account. Use `--ablation none` for these cases: a case that needs the MCP tools
cannot pass without the plugin, so the with/without delta tells you nothing.

Smoke (the short cases, 1 run each, about 10 minutes, about 3 to 4 USD):

```bash
claude plugin eval . --tag smoke --ablation none --runs 1 --no-publish --trust-plugin --scaffold --max-cost-usd 10
```

Full (all mocked cases, 3 runs each):

```bash
claude plugin eval . --tag mock --ablation none --no-publish --trust-plugin --scaffold --max-cost-usd 30
```

One case while you change it:

```bash
claude plugin eval . --case units-kg --ablation none --runs 1 --no-publish --trust-plugin --scaffold
```

End to end (the real server, one job). Run it only when no other topopt job
runs on this machine, and always with `-j 1`:

```bash
EVAL_TOPOPT_ENV_DIR="$PWD/.plugin-env" EVAL_TOPOPT_NO_WAIT=1 claude plugin eval . --tag e2e --mocks off --allow-tools "mcp__plugin_topopt_topopt__*" --ablation none -j 1 --runs 1 --no-publish --trust-plugin
```

`EVAL_TOPOPT_ENV_DIR` gives the server the venv of this checkout. The run
gives the plugin an empty temporary `CLAUDE_PLUGIN_DATA` and passes only some
variables of the shell (`EVAL_*` is one of them). Without the variable, the
server starts the first-start setup (about 200 MB) and does not connect in
30 s. If `.plugin-env` is not ready, run `sh scripts/topopt-env.sh sync`
first. Do not set the variable to the data folder of the installed plugin.

`EVAL_TOPOPT_NO_WAIT=1` makes the job process end when the run is done. Without
it, the job process of the eval serves the dashboard after the run (first as
`topopt run`, then as `topopt dashboard`), and you must stop it.

Exit code 0: all cases are at or above `--threshold`. Exit code 1: a case is
below it, a case did not load, or a run ended on an API error. Exit code 2: the
cost ceiling stopped the run, or the API refused the credential. The results go
to `evals/results/<time>/` (`report.html`, `aggregate-result.json`). Git
ignores `evals/results/`.

## Run the evals in CI

The workflow `.github/workflows/evals.yml` runs the mocked cases on a GitHub
runner (`ubuntu-24.04`). It starts by hand only, like `tests.yml`: a push or a
pull request never starts it. GitHub starts a manual run only when the workflow
file is on the default branch. Start it from the Actions tab (workflow
`evals`, Run workflow) or with:

```bash
gh workflow run evals.yml --repo chartergroup/topopt -f model=claude-sonnet-5-5 -f judge_model=claude-haiku-5-5 -f threshold=0.8
```

The inputs of the workflow:

| Input | Default | What it does |
|---|---|---|
| `model` | `claude-sonnet-5-5` | The model of the agent under test (`--model`). |
| `judge_model` | `claude-haiku-5-5` | The model of the `llm` graders and the agent mocks (`--judge-model`). |
| `threshold` | `0.8` | The lowest case score that passes, 0 to 1. `0` turns the score gate off. |
| `tag` | `smoke` | `smoke` or `mock`. The e2e case needs the real server and does not run in CI. |
| `case` | empty | A glob of case names (`--case`). Empty: all the cases of the tag. |
| `runs` | `1` | Runs per case. 1 is cheap; 3 gives less noise. |
| `max_cost_usd` | `15` | The cost ceiling (a list-price estimate). The run stops there with exit code 2. |

Pin both models, so that a model change is not read as a plugin change.

With `threshold` `0` no case score fails the job. The job still fails when the
run fails: a case did not load, a run ended on an API error, the cost ceiling
stopped the run, or the API refused the credential.

The steps are in the composite action `.github/actions/plugin-eval`. It
installs Node.js 22 and the pinned Claude Code (2.1.303), calls
`scripts/ci/run_evals.py`, and uploads the output folder (`report.html`,
`aggregate-result.json`, the mock recordings) as the artifact `evals`
(14 days). `run_evals.py` checks the inputs, runs `claude plugin eval` with
`--mocks record --ablation none --scaffold --trust-plugin --no-publish`, and
writes a table of the case scores to the job summary. Check the command with
no model call:

```bash
python3 scripts/ci/run_evals.py --model claude-sonnet-5-5 --judge-model claude-haiku-5-5 --threshold 0 --dry-run
```

The same script runs on your own machine without `--dry-run`. There `claude`
can use its own login (the run counts against the plan usage of that login),
and the results go to `evals/results/ci/`. The script uses the Claude Code on
`PATH`, any version with `claude plugin eval` (2.1.269 or later); it prints
the version, and the summary names it. Only the action pins a version (its
input `claude-version`).

The credential is a repository secret: `ANTHROPIC_API_KEY` (an API key; the
API bill pays) or `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`; the
plan usage of that account pays). With neither secret the run step stops
before a model call. The cases grant no Bash, so the runner needs no sandbox
packages (`bubblewrap`, `socat`).

Cost (estimate): smoke with 1 run per case, about 3 to 4 USD of tokens
(measured on a Mac, not in CI); mock with 3 runs, about 10 USD. The runner time
is one Linux job of about 5 to 15 minutes.

## Update the mocks

Do this after a change of a tool, its schema or its answer. Do not set
`TOPOPT_ENV_DIR` to the data folder of the installed plugin: that syncs the
venv of the installed plugin to this checkout.

The Apple Accelerate shim must be built in this checkout. `sh scripts/topopt-env.sh sync`
installs topopt in editable mode (`-e`): the venv uses the code of this
checkout, so a change of the code needs no new install. But `sync` does not
build the shim. Without the shim, `environment` says `auto_solver: amg`, and
the hook job runs with AMG. Build it once:

```bash
sh topopt/native/build.sh
```

Do not install topopt again without `-e` (for example `uv pip install .`): a
copy of the package in the venv has no shim and does not follow the code of
the checkout.

1. The tool list (no job):

   ```bash
   sh scripts/topopt-env.sh python evals/tools/dump_tools.py "$PWD/scripts/topopt-env.sh" > evals/mocks/topopt/_tools.json
   ```

2. The hook flow (ONE optimizer job of about 20 s; make sure that no other
   topopt job runs). The same script records the two `render_part` fixtures
   (`render_part_yaml.json` after `check_problem`, `render_part_run.json` after
   `run_result`). The work folder must be outside the repo:

   ```bash
   sh scripts/topopt-env.sh python evals/tools/record_fixtures.py "$PWD/scripts/topopt-env.sh" /tmp/topopt-eval-work evals/mocks/topopt/fixtures
   ```

3. The case fixtures and the user files (no job):

   ```bash
   sh scripts/topopt-env.sh python evals/tools/record_cases.py "$PWD/scripts/topopt-env.sh" /tmp/topopt-eval-cases evals
   ```

4. Copy `mocks/topopt/fixtures/print_check.json` to
   `print-after-run/mocks/topopt/fixtures/`. A mock can insert only a file
   below its own folder (`{{file:../...}}` is refused).
5. The history of `print-after-run` (after a change of a skill or a tool text):
   - Move `mocks/topopt/print_check.md` out of the suite for this run, so
     that the earlier run has no `print_check`. Put it back after the run.
   - Run `hook-from-words` once with `--keep-temp`, with the smoke flags.
   - Make the history from the session of that run:

     ```bash
     python3 evals/tools/make_history.py <kept dir>/config/projects/<cwd>/<session>.jsonl "$PWD" > evals/print-after-run/history.jsonl
     ```

   - The kept dir is sealed. Open it with `chmod 700 <kept dir> <kept dir>/sealed`.
   - Write the `run_result` and `run_status` answers of that run to
     `print-after-run/mocks/topopt/fixtures/`. They are in
     `results/<time>/mock-recordings/topopt/` (key `output`). Then the facts
     of the history and of the mocks are the same.
   - Put the facts of the new part in `print-after-run/mocks/topopt/print_check.md`:
     the path, the volume, the triangles, the voxel and the size. The agent
     mock gives those facts. With old facts it says that the file is missing.
   - Delete the kept dir: it holds account facts.
6. Read the rubrics of the `llm` graders. Some of them name the numbers of
   the fixed answers (`voxel-guard`, `lost-force`, `faster-backend`,
   `print-after-run`). Change them if the numbers changed.
7. Run the smoke command.
