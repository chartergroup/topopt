# Tests, acceptance and limits

[Back to the README](../../README.md)

## Tests

```
.venv/bin/python -m pytest -n auto
```

`-n auto` is the fast path: it runs the tests in parallel with pytest-xdist (extra `dev`), one worker for each usable CPU. A plain
`.venv/bin/python -m pytest` still works and runs the tests one after the other. A run of a few files is quick in serial.

On a Mac with Apple silicon, build the Accelerate shim first (`sh topopt/native/build.sh`, once for each checkout). The tests of the
solver `cholesky` (marker `needs_accelerate` in `tests/helpers.py`) run on every such Mac, and without the shim they fail; they do not
skip. The CI leg of that platform builds the shim in its own step (`build_shim` in `scripts/ci/legs.toml`).

Each worker needs memory: 11 workers added about 7 GiB on a Linux host (about 0.65 GB for each worker). So `-n auto` on 10 cores adds about
6.5 GB. On a 16 GB laptop, `-n 4` is a safe start. The threads of all workers multiply, so on a machine with many cores set a small thread cap
for the run (for example `OMP_NUM_THREADS=1`; on 4 CPUs a full run took 619 s with 1 thread, 646 s with 2 and 757 s with no cap).

The CI workflow is `.github/workflows/tests.yml`.
CI runs by hand only. A macOS runner minute counts more than a Linux runner minute. A push or a pull request starts no run. Start it from the Actions tab (workflow
`tests`, Run workflow), or with:

```
gh workflow run tests.yml --repo chartergroup/topopt -f legs=linux
```

The input `tier` (default `fast`) chooses how many tests run: `fast` runs the tests under 5 s (on macOS only the platform tests, on the Intel-pins leg no pytest), `full` runs every test. The Linux leg runs first; macOS and the Intel pins start only when it has passed, and two runs never go at the same time. The input `legs` has two choices. `linux` (the default) runs the suite on Linux x86_64 (leg `linux-x86_64`) and the wheel job.
`all` also runs the macOS arm64 leg and a Linux leg with the pins of `requirements-macos-x86_64.lock`.
Every leg runs the tests on Python 3.13 only (`pythons` in `scripts/ci/legs.toml`, since 2026-10-27); the launcher makes 3.13, the Python of every leg.
The leg `linux-x86_64` checks that both lock files still resolve with wheels only, for every supported Python, 3.13 only (`scripts/ci/check_locks.py`); the wheel job installs the wheel in a clean venv of each Python. The second file
`.github/workflows/intel-mac.yml` runs the suite on a hosted Intel Mac runner, by hand only (`workflow_dispatch`).
Each leg also runs a short
hook run at 1 thread and compares it with `ci/golden/hook_short.json`, and checks the licences of the installed packages.
A wheel job builds the wheel and runs the packaging test outside the source tree. The contract is CONTRACTS.md section 27.
The workflow `.github/workflows/evals.yml` runs the plugin evals on the MCP mocks, by hand only. Its inputs give the models and the score threshold
(`0`: no score gate); each case is a paid model call. See `evals/README.md` ("Run the evals in CI").

## Limits and guards

- A voxel guard stops a run that is too big. The default limits are 100,000 active voxels
  (`max_active_voxels`) and 5,000,000 grid voxels (`max_grid_voxels`).
  The error message gives a larger `voxel_size` that fits.
  You can change the limits in the `settings:` section of the YAML file.
- The time estimate comes from `topopt check`. The solver cost depends on the free dofs.
  The hook uses a voxel size of 1.2 mm. It has about 116,000 free dofs and needs about 1.5 s for each iteration.
  A smaller voxel size is much slower and needs more memory. At 1.0 mm the hook needs about 3 GB.
  `topopt check` shows an "up to" value next to the time estimate and the memory estimate. It was measured in experiment E1: finished runs on one shared Linux host with 10,000 free dofs and more (for `pardiso`, 90 percent of the runs took at most 3.12 times the time estimate and had a peak of at most 1.54 times the memory estimate; the output gives the number of runs). It is a fact, not a limit: no warning or refusal uses it. A Mac or `cholesky` has none (E1 has no run there), and a problem with fewer than 10,000 free dofs has none (there E1 found the time estimate high: the median `pardiso` run took 0.32 times it). The estimate itself is not an upper bound: with 50,000 free dofs and more the `pardiso` runs took a median of 2.19 times it.
- Supported platforms: macOS on Apple silicon (Accelerate Cholesky solver) and Linux x86_64
  (PARDISO with the `pardiso` extra, AMG without it). Windows is not supported. An Intel Mac is best effort (see Platforms).
  The time estimate of `topopt check` has a rough factor for each solver. AMG is slower than PARDISO. The factor depends on the problem (the hook: about 4 times).
- Python 3.13 only. The launcher makes Python 3.13. CI runs the tests on 3.13, on Linux x86_64 and on a Mac with Apple silicon (`tests.yml`, 2026-10-27: the legs `macos-arm64`, `linux-x86_64` and `linux-x86_64-intel-pins` succeeded).
- Status: an early version (0.x). Releases are tags of this repo, and `CHANGELOG.md` lists the changes of each release. The tool is general: it reads any problem YAML with closed STL region files, and the hook is one example. The test suite runs on Linux x86_64 (the build host); the CI workflow runs by hand only and also has a macOS leg (see "Tests"). The Mac numbers in this file come from an M1 (the hook run time) and an M4 (`gpu-mg`). A step that nobody has run on a Mac says so where it is described.
