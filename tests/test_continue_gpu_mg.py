"""A continuation with the solver gpu-mg, on the CPU OpenCL device pocl (P4b-5, CONTRACTS.md section 24.3).

gpu-mg is not byte-exact against itself after a restart: the solve starts from the saved solution, and the Krylov iteration stops at `gpu_tol`.
The measured difference of a continuation from the run of the full length (pocl, 2 mm cantilever, K 10 + 4 passes, `gpu_tol` 1e-6): compliance
3.0e-9, volume fraction 9.3e-11, no voxel of the last snapshot and no byte of `final.stl` differ. Two straight runs are equal (0.0). The limits
are 30 to 100 times these numbers. The timing file `gpu_mg.jsonl` keeps the lines of the parent up to the kept pass, and the new run numbers its
lines after them.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import test_checkpoint as tc
from test_continue import DROP, parent_state, read_json
from test_gpu_mg import case_yaml, dev  # noqa: F401  (the helper of the YAML and the fixture of the device)
from topopt.cli import main

pytestmark = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1

K, N = 10, 4                    # the beta stages of the case start at 10 and 20: the replayed pass is a stage start
TOTAL = K + N
COMPLIANCE_RTOL = 1e-7          # measured 3.0e-9
VOLUME_RTOL = 1e-8              # measured 9.3e-11


def run(yaml_path, out, *flags):
    assert main(["run", str(yaml_path), "--out", str(out), "--quiet", *flags]) == 0


def timing(folder):
    return [json.loads(line) for line in (Path(folder) / "gpu_mg.jsonl").read_text().splitlines()]


def close(base, new, kept_snapshot_levels=1):
    hb, hn = tc.history(base), tc.history(new)
    assert [r["iteration"] for r in hb] == [r["iteration"] for r in hn]
    for a, b in zip(hb, hn):
        assert abs(a["compliance"] - b["compliance"]) <= COMPLIANCE_RTOL * abs(a["compliance"]), a["iteration"]
        assert abs(a["volume_fraction"] - b["volume_fraction"]) <= VOLUME_RTOL * abs(a["volume_fraction"]), a["iteration"]
    last = f"it_{TOTAL:04d}.bin"
    sa, sb = (np.frombuffer((d / "snapshots" / last).read_bytes(), np.uint8).astype(int) for d in (Path(base), Path(new)))
    assert int(np.abs(sa - sb).max()) <= kept_snapshot_levels


@pytest.fixture
def case(tmp_path, dev):
    """The gpu-mg case of TOTAL passes and its run (in this process)."""
    p = case_yaml(tmp_path / "c", solver="gpu-mg", case="C", h=2.0, iterations=TOTAL, gpu_tol=1e-6)
    base = tmp_path / "base"
    run(p, base)
    return p, base


def test_a_gpu_mg_continuation_is_close_to_the_full_run_and_keeps_the_timing_lines_of_the_parent(case, tmp_path):
    p, base = case
    tc.edit(p, lambda d: d["settings"].update(iterations=K))
    parent = tmp_path / "parent"
    run(p, parent)
    tc.edit(p, lambda d: d["settings"].update(iterations=TOTAL))
    before = parent_state(parent)
    out = tmp_path / "new"
    run(p, out, "--continue-from", str(parent), "--more-iterations", str(N))
    close(base, out)
    assert read_json(out / "run.json")["continued_from"]["replay"] is True and read_json(out / "run.json")["solver"] == "gpu-mg"
    lines, old, full = timing(out), timing(parent), timing(base)
    assert [r["line"] for r in lines] == list(range(len(lines))) and [r["iteration"] for r in lines] == [r["iteration"] for r in full]
    kept = [r for r in old if r["iteration"] < K]
    first = lambda d: (d / "gpu_mg.jsonl").read_bytes().splitlines(keepends=True)[:len(kept)]   # noqa: E731
    assert len(kept) == K and first(out) == first(parent)                      # the lines of the passes before K: byte for byte
    assert sum(r["iteration"] == K for r in lines) == sum(r["iteration"] == K for r in old)      # the replayed pass has its lines once
    assert f"gpu_mg.jsonl keeps {len(kept)} line" in tc.log(out) and f"up to iteration {K - 1}" in tc.log(out)
    assert parent_state(parent) == before                                     # the parent keeps its timing file


def test_a_gpu_mg_fork_from_a_killed_run_keeps_the_timing_lines_up_to_the_checkpoint(case, tmp_path):
    p, base = case
    parent = tmp_path / "killed"
    tc.killed(p, parent, "after_checkpoint:6")
    before = parent_state(parent)
    out = tmp_path / "new"
    run(p, out, "--continue-from", str(parent), "--more-iterations", str(TOTAL - 6))
    close(base, out)
    assert read_json(out / "run.json")["continued_from"]["replay"] is False
    lines, full = timing(out), timing(base)
    assert [r["line"] for r in lines] == list(range(len(lines))) and [r["iteration"] for r in lines] == [r["iteration"] for r in full]
    assert "gpu_mg.jsonl keeps" in tc.log(out) and "up to iteration 6" in tc.log(out)
    assert parent_state(parent) == before
