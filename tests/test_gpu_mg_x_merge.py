"""Where the solver gpu-mg (#25) meets other PRs (integration merge, step 18).

- A4b (#38): until P2c gpu-mg clamped all 3 dofs of a fixed node, so a YAML with a per-axis support was refused at the start
  (`check`, `run` and the solver itself; the text was `cli.GPU_MG_AXES`). P2c gives each dof its own flag on every level, and the
  refusal is gone: the tests below pin that `check` and `run` do not stop for a per-axis support.
- D1b (#39): the solver of gpu-mg has no `factor`. A resume then does not use the saved step acceptance solve
  (D1b plan, G1d note). The next pass solves the design again.
No test here needs pyopencl or a device: the devices are faked, as in `test_gpu_mg_device.py`.
"""
import json

import pytest

from test_checkpoint import needs_direct, killed, log, run_args
from test_load_types import case
from test_stress_report import stress_yaml
from topopt import checkpoint as C
from topopt import cli, optimize
from topopt.cli import main
from topopt.gpu_mg import device
from topopt.gpu_mg.device import DeviceInfo


@pytest.fixture
def fake_cpu_device(monkeypatch):
    dev = DeviceInfo(index=0, platform="Test Platform", name="Fast CPU", kind="CPU", fp64=True, version="OpenCL 1.2",
                     max_work_group_size=256, max_mem_alloc=2**30, global_mem=2**32, local_mem=32768,
                     constant_mem=65536, compute_units=8)
    monkeypatch.setattr(device, "pyopencl_problem", lambda: None)
    monkeypatch.setattr(device, "enumerate_devices", lambda: [dev])


def roller_yaml(tmp_path, partial=True):
    """The cantilever of test_load_types with solver gpu-mg. The plate is a full clamp. `partial`: a second support, a roller
    at the far end of the part (axis y only), makes the supports per-axis; the six rigid modes stay blocked by the plate."""
    def mutate(d):
        d["settings"]["solver"] = "gpu-mg"
        if partial:
            d["regions"]["fixed"] = ["plate.stl", {"file": "roller.stl", "axes": ["y"]}]
    return case(tmp_path, mutate=mutate, regions={"roller": ((20, 0, 0), (22, 10, 10))} if partial else None)


def test_check_does_not_stop_gpu_mg_for_a_per_axis_support_and_the_old_refusal_text_is_gone(tmp_path, fake_cpu_device):
    d = cli.check_data(roller_yaml(tmp_path))
    assert d["solver"]["resolved"] == "gpu-mg"
    assert not [w for w in d["warnings"] if "cannot start here" in w or "clamps all 3 dofs" in w]
    assert not hasattr(cli, "GPU_MG_AXES")
    assert d["extended"]["supports"]["axes"] == {"plate.stl": "xyz", "roller.stl": "y"}                       # check lists the axes of each support


def test_check_has_no_such_warning_for_a_full_clamp(tmp_path, fake_cpu_device):
    d = cli.check_data(roller_yaml(tmp_path, partial=False))
    assert not [w for w in d["warnings"] if "clamps all 3 dofs" in w]


def test_run_goes_past_the_start_with_a_per_axis_support(tmp_path, fake_cpu_device, capsys):
    """The old stop (before the solve, no `history.jsonl`) is gone. The fake device cannot make a context, so the run stops later, at the device:
    the run log has the memory line, which comes after the place where the refusal was, and the error is not the old text."""
    out = tmp_path / "out"
    main(["run", str(roller_yaml(tmp_path)), "--out", str(out), "--quiet"])
    assert "clamps all 3 dofs" not in capsys.readouterr().err
    assert "memory estimate: about" in (out / "run.log").read_text()


class NoFactor:
    """A solver as GpuSolver is: it solves, and it has no `factor` method. `inner` is a real solver."""

    def __init__(self, inner):
        self.inner = inner

    def solve(self, B, warm=True):
        return self.inner.solve(B, warm=warm)

    def free(self):
        self.inner.free()


@needs_direct
def test_a_resume_does_not_call_factor_on_a_solver_without_one(tmp_path, monkeypatch):
    """D1b x G1d: the restore of the saved step acceptance solve makes the factor again. A solver with no `factor`
    (gpu-mg) gets a fact instead, and the next pass solves the design again."""
    yaml_path = stress_yaml(tmp_path, iterations=24)
    out = tmp_path / "r"
    killed(yaml_path, out, "after_checkpoint:12")                          # pass 13 has the same beta: the checkpoint has carry_U
    assert "carry_U" in C.read_checkpoint(out).arrays
    make = optimize.Model._make_solver

    def make_solver(self, edof):
        solver, name = make(self, edof)
        return NoFactor(solver), name

    def solve(self, K):                                                     # Model.solve calls `factor`: do it on the inner solver
        self.solver.inner.factor(K)
        return self.solver.inner.solve(self.f_red)

    monkeypatch.setattr(optimize.Model, "_make_solver", make_solver)
    monkeypatch.setattr(optimize.Model, "solve", solve)
    assert main(run_args(yaml_path, out, "--resume")) == 0
    assert "resume: the step acceptance solve of the saved design is not used: the solver has no factor" in log(out)
    assert [json.loads(line)["iteration"] for line in (out / "history.jsonl").read_text().splitlines()] == list(range(25))
