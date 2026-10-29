"""Tests of the GPU first rule during a run (CONTRACTS.md section 20): `GpuFailure` and the fallback of `settings.solver: auto`.

They need pyopencl and one OpenCL device. On this project's Linux host that is pocl on the CPU, so a test lets pocl count as a GPU by a monkeypatch of
`device.counts_as_gpu`. No test needs a real GPU. A failure is injected into a run at a chosen pass by a monkeypatch of the solver: the code that
turns an OpenCL error or a solve that is not accepted into a `GpuFailure` is the real code.
"""
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("pyopencl")
pytestmark = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1

from test_gpu_mg import SMALL, case_yaml  # noqa: E402
from topopt.config import UserError, load_config  # noqa: E402
from topopt.gpu_mg import device  # noqa: E402
from topopt.gpu_mg.device import GpuFailure  # noqa: E402
from topopt.gpu_mg.model import GpuModel, GpuParams, SolveFailure  # noqa: E402
from topopt.problem import build_problem  # noqa: E402


@pytest.fixture(scope="module")
def dev():
    """The device that `auto` takes. The test is skipped when there is none (a UserError)."""
    try:
        return device.select("auto", "auto")[0]
    except UserError as e:
        pytest.skip(f"no usable OpenCL device: {e}")


def small_model(tmp_path, params=SMALL):
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=3)))
    return GpuModel(prob, None, params)


# --- GpuFailure and its kinds -----------------------------------------------------------------------------------


def test_gpu_failure_is_a_user_error_with_a_kind():
    e = GpuFailure("text", "setup")
    assert isinstance(e, UserError) and e.kind == "setup" and e.solver is None and str(e) == "text"
    assert issubclass(SolveFailure, GpuFailure)
    with pytest.raises(ValueError, match="kind must be one of device_error, device_memory, not_accepted, setup"):
        GpuFailure("text", "other")


def test_is_memory_error_by_class_and_by_status_code():
    codes = SimpleNamespace(MEM_OBJECT_ALLOCATION_FAILURE=-4, OUT_OF_RESOURCES=-5, OUT_OF_HOST_MEMORY=-6, INVALID_VALUE=-30)
    cl = SimpleNamespace(MemoryError=type("MemoryError", (Exception,), {}), status_code=codes)
    assert device.is_memory_error(cl, cl.MemoryError("x"))
    for code, expected in ((-4, True), (-5, True), (-6, True), (-30, False)):
        assert device.is_memory_error(cl, SimpleNamespace(code=code)) is expected
    assert not device.is_memory_error(cl, ValueError("no status record"))


@pytest.mark.parametrize("where", ["upload_numerics", "pcg"])
@pytest.mark.parametrize("error,kind", [("RuntimeError", "device_error"), ("LogicError", "device_error"), ("MemoryError", "device_memory")])
def test_a_device_error_is_a_gpu_failure_of_its_kind(tmp_path, monkeypatch, dev, where, error, kind):
    import pyopencl as cl
    from topopt.gpu_mg import engine
    m = small_model(tmp_path)
    cls = getattr(cl, error)

    def broken(*a, **k):
        raise cls("clEnqueueNDRangeKernel failed: OUT_OF_RESOURCES")

    monkeypatch.setattr(engine.GpuMG, where, broken)
    with pytest.raises(GpuFailure) as e:
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    assert e.value.kind == kind and e.value.solver is m.solver and m.solver._freed         # the buffers were released
    assert "reported an error" in str(e.value) and "buffers were released" in str(e.value)


@pytest.mark.usefixtures("dev")
def test_a_solve_that_is_not_accepted_is_a_gpu_failure_of_the_kind_not_accepted(tmp_path):
    m = small_model(tmp_path, GpuParams(min_free=300, max_iter_first=1, max_iter=1, max_refine=1))
    with pytest.raises(GpuFailure, match=r"did not reach the relative residual.*case C.*Options.*'amg'") as e:
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    assert isinstance(e.value, SolveFailure) and e.value.kind == "not_accepted" and e.value.its["failed"]
    assert e.value.solver is m.solver and not m.solver._freed          # the fallback releases it
    m.solver.free()
    m.solver.free()                                                    # a second call does nothing


def test_an_engine_that_cannot_be_made_is_a_gpu_failure_of_the_kind_setup(tmp_path, monkeypatch, dev):
    import pyopencl as cl
    from topopt.gpu_mg import engine
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=3)))

    def broken(*a, **k):
        raise cl.RuntimeError("clBuildProgram failed: BUILD_PROGRAM_FAILURE")

    monkeypatch.setattr(engine, "GpuMG", broken)
    with pytest.raises(GpuFailure, match=r"could not build or allocate the program.*BUILD_PROGRAM_FAILURE") as e:
        GpuModel(prob, None, SMALL)
    assert e.value.kind == "setup" and e.value.solver is None


def test_a_buffer_that_is_larger_than_the_device_allows_is_a_gpu_failure_of_the_kind_setup(tmp_path, monkeypatch, dev):
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=3)))
    small = dataclasses.replace(dev, max_mem_alloc=1024)
    monkeypatch.setattr(device, "select", lambda *a, **k: (small, "C"))
    with pytest.raises(GpuFailure, match=r"the dense inverse of the coarsest level needs") as e:
        GpuModel(prob, None, SMALL)
    assert e.value.kind == "setup"


def test_a_device_that_is_gone_is_a_gpu_failure_of_the_kind_setup(tmp_path, monkeypatch, dev):
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=3)))

    def gone(*a, **k):
        raise UserError("solver 'gpu-mg' found no OpenCL device on this machine.")

    monkeypatch.setattr(device, "select", gone)
    with pytest.raises(GpuFailure, match="found no OpenCL device") as e:
        GpuModel(prob, None, SMALL)
    assert e.value.kind == "setup"


def test_an_error_that_is_not_a_device_error_is_not_a_gpu_failure(tmp_path, monkeypatch, dev):
    from topopt.gpu_mg import engine
    m = small_model(tmp_path)
    monkeypatch.setattr(engine.GpuMG, "pcg", lambda *a, **k: (_ for _ in ()).throw(ValueError("a bug")))
    with pytest.raises(ValueError, match="a bug") as e:
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    assert not isinstance(e.value, GpuFailure)
    monkeypatch.undo()
    m.solver.free()


# --- the record of the parts ----------------------------------------------------------------------------------------


def test_the_parts_of_a_run_are_recorded():
    from topopt import solver_fallback as sf
    parts = []
    sf.part_start(parts, "gpu-mg", "Dev", 0)
    assert parts == [{"solver": "gpu-mg", "device": "Dev", "first_pass": 0, "last_pass": None, "end": None}]
    assert sf.verdict_notes(parts) == []                                           # one part that did not end: nothing to say
    parts[-1]["last_pass"] = 2
    sf.part_start(parts, "gpu-mg", "Dev", 3)                                       # a resume with the same solver goes on in the same part
    assert len(parts) == 1
    sf.part_end(parts, "gpu-mg", "Dev", "device_error", "OpenCL  error\n  text")
    assert parts[0]["end"]["kind"] == "device_error" and parts[0]["end"]["text"] == "OpenCL error text" and parts[0]["end"]["at_utc"].endswith("Z")
    assert parts[0]["last_pass"] == 2
    sf.part_start(parts, "amg", None, 3)
    parts[-1]["last_pass"] = 6
    assert [p["solver"] for p in parts] == ["gpu-mg", "amg"]
    assert sf.verdict_notes(parts) == ["solver: gpu-mg on Dev ran passes 0 to 2 and stopped (device_error); amg ran passes 3 to 6. "
                                       "The cause is in run.json (`solver_parts`) and in run.log."]


def test_a_part_that_stopped_in_its_setup_has_no_pass():
    from topopt import solver_fallback as sf
    parts = []
    sf.part_end(parts, "gpu-mg", "Dev", "setup", "could not build")
    assert parts == [{"solver": "gpu-mg", "device": "Dev", "first_pass": None, "last_pass": None,
                      "end": {"kind": "setup", "text": "could not build", "at_utc": parts[0]["end"]["at_utc"]}}]
    sf.part_start(parts, "amg", None, 0)
    assert len(parts) == 2 and parts[1]["first_pass"] == 0
    assert "gpu-mg on Dev ran no pass and stopped (setup)" in sf.verdict_notes(parts)[0]


def test_a_continuation_with_the_same_solver_goes_on_in_the_last_part_and_another_solver_starts_a_new_one():
    """A continuation equals the run of full length (tests/test_continue.py), and `run.json` is part of that: the same solver gives the same one part."""
    from topopt import solver_fallback as sf
    parts = [{"solver": "gpu-mg", "device": "Dev", "first_pass": 0, "last_pass": 4, "end": None}]
    sf.part_start(parts, "gpu-mg", "Dev", 4)
    assert [(p["first_pass"], p["last_pass"]) for p in parts] == [(0, 4)] and sf.verdict_notes(parts) == []
    sf.part_start(parts, "gpu-mg", "Other device", 4)                              # another device is another part
    sf.part_start(parts, "amg", None, 4)
    assert [(p["solver"], p["device"], p["first_pass"]) for p in parts] == [("gpu-mg", "Dev", 0), ("gpu-mg", "Other device", 4), ("amg", None, 4)]


def test_the_reason_has_no_options():
    from topopt import solver_fallback as sf
    e = GpuFailure("solver 'gpu-mg': the device failed.  Options: set settings.solver to 'amg'", "device_error")
    assert sf.reason_text(e) == "solver 'gpu-mg': the device failed"
    assert len(sf.reason_text(GpuFailure("x" * 2000, "setup"))) == sf.TEXT_LIMIT          # no sentence end inside the limit: cut at the limit
    long_text = "The solve did not reach the target. " + "Reached a value. " * 60 + "A last sentence that is cut away. " * 20
    cut = sf.reason_text(GpuFailure(long_text + " Options: another solver.", "not_accepted"))
    assert len(cut) <= sf.TEXT_LIMIT and cut.endswith("Reached a value") and "Options" not in cut      # ends at a full sentence, without the final dot
    assert sf.cut_text(long_text).endswith("Reached a value.") and sf.cut_text("a  b\n c") == "a b c"


# --- runs -------------------------------------------------------------------------------------------------------------------

ITERATIONS = 6


def auto_yaml(tmp_path, solver="auto", iterations=ITERATIONS, name="case"):
    d = tmp_path / name
    return case_yaml(d, solver=solver, case="auto", h=2.0, iterations=iterations)


@pytest.fixture
def pocl_is_a_gpu(monkeypatch, dev):
    """Let the pocl CPU device count as a GPU for `settings.solver: auto`. The self-test runs on it for real (once for each process).

    The test session sets `TOPOPT_AUTO_GPU=0` (tests/conftest.py); the fixture deletes it, also for the subprocesses of the test."""
    from helpers import in_process_probe
    in_process_probe(monkeypatch)       # the product asks a child process; `counts_as_gpu` is faked in this process
    monkeypatch.delenv("TOPOPT_AUTO_GPU", raising=False)
    monkeypatch.setattr(device, "counts_as_gpu", lambda d: True)
    device.forget_devices()
    pick, why = device.auto_gpu("auto", "auto")
    assert pick is not None, why
    assert device.self_test(*pick)["passed"]     # the real self-test, here: a test that breaks the solver later does not break the self-test of the run
    yield pick
    device.forget_devices()


def inject(kind, after):
    """Make the gpu-mg solver of the next run fail at its design number `after` + 1, that is at pass `after` (one design for each pass).

    The monkeypatch breaks the engine of that one solver (an instance, not the class): the self-test and the solvers of other tests are not changed. The error
    is raised by the engine, so the code that turns it into a `GpuFailure` runs for real. Return the patch to undo it with `monkeypatch.undo`.
    `stagnated` (#378) caps the inner PCG at 0 iterations: a refinement step of case C then adds nothing, and the stagnation test of the loop stops the solve.
    """
    import pyopencl as cl
    from topopt.gpu_mg.model import GpuSolver
    real = GpuSolver.set_design

    def set_design(self, Ee):
        self._p94d_designs = getattr(self, "_p94d_designs", 0) + 1
        if self._p94d_designs == after + 1:
            if kind == "not_accepted":              # a real solve that does not reach its target: the caps leave one iteration
                self.params = dataclasses.replace(self.params, max_iter_first=1, max_iter=1, max_refine=1)
            elif kind == "stagnated":
                self.params = dataclasses.replace(self.params, max_iter=0)
            else:
                cls = cl.MemoryError if kind == "device_memory" else cl.RuntimeError

                def broken(*a, **k):
                    raise cls("clEnqueueNDRangeKernel failed: OUT_OF_RESOURCES")

                self.engine.upload_numerics = broken
        return real(self, Ee)

    return set_design


def run_cli(args):
    from topopt.cli import main
    return main(["run", *map(str, args), "--quiet"])


def read(out):
    out = out if hasattr(out, "joinpath") else __import__("pathlib").Path(out)
    rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    return {"run": json.loads((out / "run.json").read_text()), "rows": rows, "log": (out / "run.log").read_text(),
            "verdict": json.loads((out / "verdict.json").read_text()) if (out / "verdict.json").exists() else None}


def compliances(rows):
    return [r["compliance"] for r in rows]


@pytest.mark.parametrize("kind", ["device_error", "device_memory", "not_accepted"])
def test_a_failure_at_pass_3_goes_on_with_the_cpu_solver_and_the_run_ends_ok(tmp_path, monkeypatch, pocl_is_a_gpu, kind):
    from topopt.gpu_mg.model import GpuSolver
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    pure = tmp_path / "pure"
    assert run_cli([auto_yaml(tmp_path, solver="gpu-mg", name="p"), "--out", pure]) == 0           # the reference: gpu-mg all the way, same inputs
    monkeypatch.setattr(GpuSolver, "set_design", inject(kind, 3))
    out = tmp_path / "auto"
    assert run_cli([auto_yaml(tmp_path, name="a"), "--out", out]) == 0
    got, ref = read(out), read(pure)
    assert [r["iteration"] for r in got["rows"]] == list(range(ITERATIONS + 1))                    # the history is continuous
    assert compliances(got["rows"])[:3] == compliances(ref["rows"])[:3]                            # the passes before the switch are the passes of gpu-mg
    run = got["run"]
    assert run["solver"] == cpu and run["solver_setting"] == "auto"
    first, second = run["solver_parts"]
    assert (first["solver"], first["first_pass"], first["last_pass"]) == ("gpu-mg", 0, 2) and first["device"]
    assert first["end"]["kind"] == kind and first["end"]["at_utc"].endswith("Z") and "Options" not in first["end"]["text"]
    assert second == {"solver": cpu, "device": None, "first_pass": 3, "last_pass": ITERATIONS, "end": None}
    assert (f"solver: gpu-mg stopped at pass 3 ({kind}: ") in got["log"]
    assert f"the run goes on with {cpu} from the checkpoint of pass 2" in got["log"]
    assert "resume: continue from checkpoint.npz, iteration 2 of 6" in got["log"] and "another pick of auto" in got["log"]
    assert "The override is given" not in got["log"]                                               # the other solver is a note, not a change
    verdict = got["verdict"]
    assert verdict["state"] in ("converged", "iteration_limit")
    assert verdict["notes"] == [f"solver: gpu-mg on {first['device']} ran passes 0 to 2 and stopped ({kind}); {cpu} ran passes 3 to {ITERATIONS}. "
                                "The cause is in run.json (`solver_parts`) and in run.log."]
    assert [p.get("reason") for p in verdict["resume_points"]] == ["solver_fallback"] and verdict["resume_points"][0]["iteration"] == 2
    from topopt import checkpoint
    meta = checkpoint.read_checkpoint(out).meta
    assert meta["solver_parts"] == run["solver_parts"] and meta["inputs"]["solver"] == cpu         # the parts are in the checkpoint
    assert (out / "result" / "final.stl").exists()                                                 # the post step ran


@pytest.mark.parametrize("kind", ["stagnated", "not_accepted"])
def test_a_case_c_solve_that_stops_at_pass_3_under_auto_goes_on_with_the_cpu_solver(tmp_path, monkeypatch, pocl_is_a_gpu, kind):
    """#378: a device without 64-bit float (Apple silicon) has no case B, so `auto` runs case C there, and the route of a case C solve that is not accepted
    is the fallback. `gpu_case: C` is set here, as such a device takes it. Both ends of the refinement loop: the stagnation test stops it (`stagnated`), or
    it uses all its steps (`not_accepted`; the end of the #378 runs)."""
    from topopt.gpu_mg.model import GpuSolver
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    monkeypatch.setattr(GpuSolver, "set_design", inject(kind, 3))
    out = tmp_path / "auto"
    assert run_cli([case_yaml(tmp_path / "c", solver="auto", case="C", h=2.0, iterations=ITERATIONS), "--out", out]) == 0
    got = read(out)
    assert [r["iteration"] for r in got["rows"]] == list(range(ITERATIONS + 1))
    first, second = got["run"]["solver_parts"]
    assert (first["solver"], first["first_pass"], first["last_pass"], first["end"]["kind"]) == ("gpu-mg", 0, 2, "not_accepted")
    text = first["end"]["text"]
    stag = "the loop stopped early: the residual fell by less than 10 % over two steps"
    assert "case C" in text and (stag in text) is (kind == "stagnated") and "Options" not in text
    assert (second["solver"], second["first_pass"], second["last_pass"], second["end"]) == (cpu, 3, ITERATIONS, None)
    assert "solver: gpu-mg stopped at pass 3 (not_accepted: " in got["log"] and f"the run goes on with {cpu} from the checkpoint of pass 2" in got["log"]
    records = [json.loads(line) for line in (out / "gpu_mg.jsonl").read_text().splitlines()]
    failed = [r for r in records if r.get("failed")]
    assert len(failed) == 1 and failed[0]["iteration"] == 3 and failed[0]["case"] == "C"
    it = failed[0]["iterations"][-1]
    assert it["stagnated"] is (kind == "stagnated") and it["nonfinite"] is False
    assert set(it["inner"]) == ({0} if kind == "stagnated" else {1})          # stagnated: the steps ran no PCG iteration, the test of the loop stopped them
    assert [p.get("reason") for p in got["verdict"]["resume_points"]] == ["solver_fallback"]


def test_a_failure_in_the_setup_starts_again_at_pass_0_with_the_cpu_solver(tmp_path, monkeypatch, pocl_is_a_gpu):
    import pyopencl as cl
    from topopt.gpu_mg import engine
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    pure = tmp_path / "cpu"
    assert run_cli([auto_yaml(tmp_path, solver=cpu, name="c"), "--out", pure]) == 0
    monkeypatch.setattr(engine, "GpuMG", lambda *a, **k: (_ for _ in ()).throw(cl.RuntimeError("clBuildProgram failed: BUILD_PROGRAM_FAILURE")))
    out = tmp_path / "auto"
    assert run_cli([auto_yaml(tmp_path, name="a"), "--out", out]) == 0
    got, ref = read(out), read(pure)
    assert [r["iteration"] for r in got["rows"]] == list(range(ITERATIONS + 1))
    assert compliances(got["rows"]) == compliances(ref["rows"])                                    # the CPU solver ran every pass
    first, second = got["run"]["solver_parts"]
    assert first["solver"] == "gpu-mg" and first["first_pass"] is None and first["end"]["kind"] == "setup"
    assert (second["solver"], second["first_pass"], second["last_pass"]) == (cpu, 0, ITERATIONS)
    assert "solver: gpu-mg stopped in the setup (setup: " in got["log"] and f"the run goes on with {cpu} from pass 0" in got["log"]
    assert "ran no pass and stopped (setup)" in got["verdict"]["notes"][0]


@pytest.mark.parametrize("case", ["B", "C"])
def test_the_forced_failure_example_switches_to_the_cpu_solver_and_ends_ok(tmp_path, pocl_is_a_gpu, case):
    """examples/hook/hook_gpu_fallback.yaml with only the paths, the length of the run and gpu_case changed: gpu-mg does not accept the first solve (the tolerance
    1e-30 and the backward error 1e-30 are below any FP64 floor), and the run goes on with the CPU solver. This is the check for a real GPU, here on pocl."""
    import re
    from pathlib import Path
    from topopt.solver_choice import resolve_solver
    dev = pocl_is_a_gpu[0]
    if case == "B" and not dev.fp64:
        pytest.skip("case B needs a device with 64-bit float")
    folder = Path(__file__).resolve().parent.parent / "examples" / "hook"
    text = (folder / "hook_gpu_fallback.yaml").read_text()
    text = re.sub(r"(?m)^(\s+(?:design: |- |tip: ))([a-z_]+\.stl)$", lambda m: m.group(1) + str(folder / m.group(2)), text)
    text = text.replace("iterations: 12", "iterations: 3").replace("beta_schedule: [[0, 1], [6, 2]]", "beta_schedule: [[0, 1]]")
    path = tmp_path / "forced.yaml"
    path.write_text(text + f"  gpu_case: {case}\n")
    out = tmp_path / "out"
    assert run_cli([path, "--out", out]) == 0
    got = read(out)
    first, second = got["run"]["solver_parts"]
    assert (first["solver"], first["end"]["kind"], first["end"]["text"].startswith("solver 'gpu-mg': the solve did not reach the relative residual 1e-30")) == ("gpu-mg", "not_accepted", True)
    assert f"case {case}" in first["end"]["text"] and first["end"]["text"].endswith(".") is False and len(first["end"]["text"]) <= 600
    assert (second["solver"], second["first_pass"], second["last_pass"], second["end"]) == (resolve_solver("auto"), 0, 3, None)
    assert got["run"]["solver"] == resolve_solver("auto") and got["run"]["solver_setting"] == "auto"
    assert [r["iteration"] for r in got["rows"]] == [0, 1, 2, 3]
    assert "solver: gpu-mg stopped at pass 0 (not_accepted: " in got["log"] and "the run goes on with" in got["log"]
    assert "ran no pass and stopped (not_accepted)" in got["verdict"]["notes"][0]


def test_a_failure_in_pass_0_starts_again_at_pass_0_and_keeps_the_record_of_the_failed_solve(tmp_path, monkeypatch, pocl_is_a_gpu):
    from topopt.gpu_mg.model import GpuSolver
    from topopt.solver_choice import resolve_solver
    monkeypatch.setattr(GpuSolver, "set_design", inject("not_accepted", 0))
    out = tmp_path / "auto"
    assert run_cli([auto_yaml(tmp_path), "--out", out]) == 0
    got = read(out)
    assert [r["iteration"] for r in got["rows"]] == list(range(ITERATIONS + 1))
    first, second = got["run"]["solver_parts"]
    assert (first["solver"], first["first_pass"], first["last_pass"], first["end"]["kind"]) == ("gpu-mg", 0, None, "not_accepted")
    assert (second["solver"], second["first_pass"]) == (resolve_solver("auto"), 0)
    assert "solver: gpu-mg stopped at pass 0 (not_accepted: " in got["log"] and "from pass 0" in got["log"]
    records = [json.loads(line) for line in (out / "gpu_mg.jsonl").read_text().splitlines()]       # the failed solve stays in the timing file
    assert records and records[-1]["failed"] is True


def test_an_explicit_gpu_mg_never_falls_back(tmp_path, monkeypatch, pocl_is_a_gpu, capsys):
    from topopt.gpu_mg.model import GpuSolver
    monkeypatch.setattr(GpuSolver, "set_design", inject("device_error", 3))
    out = tmp_path / "gpu"
    assert run_cli([auto_yaml(tmp_path, solver="gpu-mg"), "--out", out]) == 1
    assert "reported an error" in capsys.readouterr().err
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "error" and "optimizer iteration 3" in status["message"]
    log = (out / "run.log").read_text()
    assert "the run goes on with" not in log
    assert json.loads((out / "run.json").read_text())["solver_parts"][0]["solver"] == "gpu-mg"


def test_auto_on_a_cpu_runs_as_before_with_one_part_and_no_note(tmp_path, monkeypatch, dev):
    from topopt.solver_choice import resolve_solver
    out = tmp_path / "auto"
    assert run_cli([auto_yaml(tmp_path), "--out", out]) == 0                  # pocl is not a GPU: `auto` takes the CPU rule
    got = read(out)
    cpu = resolve_solver("auto")
    assert got["run"]["solver"] == cpu and got["run"]["solver_setting"] == "auto"
    assert got["run"]["solver_parts"] == [{"solver": cpu, "device": None, "first_pass": 0, "last_pass": ITERATIONS, "end": None}]
    assert got["verdict"]["notes"] == [] and "stopped at pass" not in got["log"]


def test_the_memory_check_stops_the_switch_and_keeps_the_checkpoint(tmp_path, monkeypatch, pocl_is_a_gpu, capsys):
    from topopt import cli
    from topopt.gpu_mg.model import GpuSolver
    monkeypatch.setattr(GpuSolver, "set_design", inject("device_error", 3))
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: 1_000_000)           # the machine has 1 MB: the CPU solver cannot fit
    out = tmp_path / "auto"
    assert run_cli([auto_yaml(tmp_path), "--out", out]) == 1
    err = capsys.readouterr().err
    assert "solver gpu-mg stopped at pass 3 (device_error: " in err and "the run cannot go on with" in err
    assert "the memory estimate of the CPU solver" in err and "(the estimate for gpu-mg was" in err and "this machine has 0.0 GiB RAM" in err
    assert "The checkpoint of pass 2 stays in" in err and "allow_changed_inputs" in err and "voxel_size" in err
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "error"
    from topopt import checkpoint
    assert checkpoint.describe(out)["iteration"] == 2                             # the checkpoint is kept
    parts = json.loads((out / "run.json").read_text())["solver_parts"]
    assert len(parts) == 1 and parts[0]["end"]["kind"] == "device_error" and parts[0]["last_pass"] == 2
    log = (out / "run.log").read_text()
    assert "from the checkpoint of pass 2" not in log and "solver: solver gpu-mg stopped at pass 3" in log        # the run stopped: no line says that it goes on


def test_a_resume_after_a_fallback_with_auto_is_a_note_and_needs_no_override(tmp_path, monkeypatch, pocl_is_a_gpu):
    from topopt.gpu_mg.model import GpuSolver
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    with monkeypatch.context() as m:
        m.setattr(GpuSolver, "set_design", inject("device_error", 3))
        out = tmp_path / "auto"
        yaml_path = auto_yaml(tmp_path)
        assert run_cli([yaml_path, "--out", out]) == 0
    assert read(out)["run"]["solver"] == cpu
    # The next start picks gpu-mg again (pocl counts as a GPU here): the checkpoint says `cpu`, the pick says gpu-mg, and `settings.solver` is auto on both sides.
    assert run_cli([yaml_path, "--out", out, "--resume"]) == 0
    got = read(out)
    assert f"resume: the solver that runs: {cpu} -> gpu-mg (settings.solver is auto in both runs" in got["log"]
    assert "The override is given" not in got["log"].split("resume: continue from")[-1]
    parts = got["run"]["solver_parts"]
    assert [p["solver"] for p in parts] == ["gpu-mg", cpu]       # a finished run: no pass is left, so no part starts (review F2: there is no pass 7)
    assert parts[1]["last_pass"] == ITERATIONS and parts[1]["end"] is None
    assert got["run"]["solver"] == cpu            # the solver of the last part that ran a pass; the post step reads it


def test_a_fallback_does_not_use_the_step_acceptance_solve_that_gpu_mg_made(tmp_path, monkeypatch, pocl_is_a_gpu):
    """Review F3: with a stress row the checkpoint holds the step acceptance solve (`carry_U`) of the saved design. The CPU solver that goes on after a failure
    of gpu-mg does not use the solve of the other solver: the restore says so (`checkpoint.py`, the rule "another solver wrote the checkpoint")."""
    from test_stress_report import stress_yaml
    from topopt.gpu_mg.model import GpuSolver
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    yaml_path = stress_yaml(tmp_path, iterations=24)         # beta 1 to pass 9, 2 from pass 10: the checkpoint of pass 12 has the beta of pass 13
    out = tmp_path / "out"
    monkeypatch.setattr(GpuSolver, "set_design", inject("device_error", 27))          # the 27th design of the run is the one of pass 13 (step acceptance tries more than one at the first passes)
    assert run_cli([yaml_path, "--out", out]) == 0
    got = read(out)
    assert [r["iteration"] for r in got["rows"]] == list(range(25))
    parts = got["run"]["solver_parts"]
    assert [p["solver"] for p in parts] == ["gpu-mg", cpu] and parts[0]["end"]["kind"] == "device_error"
    assert (parts[0]["last_pass"], parts[1]["first_pass"]) == (12, 13)
    fact = (f"resume: the step acceptance solve of the saved design is not used: solver gpu-mg made it and this part runs solver {cpu}. "
            "The next pass solves the design again")
    assert fact in got["log"]
    assert got["log"].index("solver: gpu-mg stopped at pass") < got["log"].index(fact)       # the fact belongs to the restart after the failure


def test_a_continuation_that_fails_before_its_first_pass_goes_on_with_its_plan(tmp_path, monkeypatch, pocl_is_a_gpu):
    from topopt.gpu_mg.model import GpuSolver
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    yaml_path = auto_yaml(tmp_path, iterations=4)
    parent = tmp_path / "parent"
    assert run_cli([yaml_path, "--out", parent]) == 0                                   # gpu-mg ran all passes
    assert [p["solver"] for p in read(parent)["run"]["solver_parts"]] == ["gpu-mg"]
    monkeypatch.setattr(GpuSolver, "set_design", inject("device_error", 0))             # the first design of the new run fails: no checkpoint of its own yet
    new = tmp_path / "new"
    assert run_cli([yaml_path, "--out", new, "--continue-from", parent, "--more-iterations", "2"]) == 0
    got = read(new)
    assert [r["iteration"] for r in got["rows"]] == list(range(7))                      # 0..4 from the parent, 5 and 6 are new... replay: pass 4 runs again
    parts = got["run"]["solver_parts"]
    assert [p["solver"] for p in parts] == ["gpu-mg", cpu]                              # the part of the parent goes on in the new run and ends at the failure, then the CPU part
    assert (parts[0]["first_pass"], parts[0]["last_pass"], parts[0]["end"]["kind"]) == (0, 4, "device_error")
    assert (parts[1]["first_pass"], parts[1]["last_pass"], parts[1]["end"]) == (4, 6, None)
    assert read(parent)["run"]["solver_parts"][0]["end"] is None                        # the record of the parent is not changed
    assert f"solver: gpu-mg stopped at pass 4 (device_error: " in got["log"] and f"the run goes on with {cpu} from the checkpoint of pass 4 of the parent run" in got["log"]
    assert "continue: " in got["log"]


# --- a fine pass (#318) with a failure of gpu-mg ---------------------------------------------------------------------------------


def fine_parent(tmp_path):
    """A finished parent run at 2 mm (6 passes) for a fine pass: `(yaml, run folder)`."""
    from test_optimize import make_case
    yaml_path = make_case(tmp_path / "case", h=2.0, iterations=6) if (tmp_path / "case").mkdir() is None else None
    parent = tmp_path / "parent"
    assert run_cli([yaml_path, "--out", parent]) == 0
    return yaml_path, parent


FINE_START_LINE = "fine pass: start design from pass 6"


@pytest.mark.parametrize("after", [0, 3])
def test_a_fine_pass_with_auto_goes_on_with_the_cpu_solver_after_a_failure_of_gpu_mg(tmp_path, monkeypatch, pocl_is_a_gpu, after):
    """Merge of #318 and #176: `cli._run_folder` starts a fine pass through `run_with_fallback` with a factory for the start design.
    A failure at pass 0 starts again at pass 0 with a new `FineStart`: its log lines appear once for each start, and the record stays. A failure at a later pass
    restarts from the checkpoint, which holds the design and the `fine_from` record: the start design is not made again."""
    from topopt.gpu_mg.model import GpuSolver
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    yaml_path, parent = fine_parent(tmp_path)
    monkeypatch.setattr(GpuSolver, "set_design", inject("device_error", after))
    out = tmp_path / "fine"
    assert run_cli([yaml_path, "--out", out, "--fine-from", parent, "--fine-voxel", "1.0"]) == 0
    got = read(out)
    rec = got["run"]["fine_from"]
    assert rec["run_dir"] == str(parent.resolve()) and rec["voxel_mm"] == 1.0 and rec["iteration"] == 6 and rec["map"]["points"] > 0
    parts = got["run"]["solver_parts"]
    assert [p["solver"] for p in parts] == ["gpu-mg", cpu] and parts[0]["end"]["kind"] == "device_error" and parts[1]["end"] is None
    assert got["run"]["solver"] == cpu and [r["iteration"] for r in got["rows"]] == list(range(len(got["rows"])))
    log = got["log"]
    assert "fine pass: the solver that runs: gpu-mg -> " not in log           # the plan of the fine pass compares the pick of `auto` (gpu-mg), not the CPU rule
    stop_at = log.index("solver: gpu-mg stopped at pass")
    before, after_stop = log[:stop_at], log[stop_at:]
    assert before.count(FINE_START_LINE) == 1                                            # the first start: its facts once
    if after == 0:
        assert f"the run goes on with {cpu} from pass 0" in log
        assert after_stop.count(FINE_START_LINE) == 1                                    # the new start has a new FineStart: its facts once, not twice
        assert parts[0]["last_pass"] is None and parts[1]["first_pass"] == 0
    else:
        assert f"the run goes on with {cpu} from the checkpoint of pass {after - 1}" in log
        assert FINE_START_LINE not in after_stop                                         # the restart reads the design and the record from the checkpoint
        assert parts[0]["last_pass"] == after - 1 and parts[1]["first_pass"] == after
        from topopt.checkpoint import read_checkpoint
        assert read_checkpoint(out).meta["fine_from"] == rec                             # the record in the checkpoint is the record of run.json


# --- the stress solve of the post step -------------------------------------------------------------------------------------


def stress_problem(tmp_path, solver, name):
    return build_problem(load_config(case_yaml(tmp_path / name, solver=solver, case="auto", h=2.0, iterations=2)))


def snapshot_of(problem):
    return np.random.default_rng(5).integers(80, 256, int(np.prod(problem.shape))).astype(np.uint8)


def test_the_stress_solve_of_an_auto_run_goes_on_with_the_cpu_solver_when_gpu_mg_fails(tmp_path, monkeypatch, pocl_is_a_gpu):
    from topopt.gpu_mg.model import GpuSolver
    from topopt.refine.stress import compute_stress
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    ref_prob = stress_problem(tmp_path, cpu, "ref")
    snap = snapshot_of(ref_prob)
    ref = compute_stress(ref_prob, snap)
    prob = stress_problem(tmp_path, "auto", "auto")
    lines = []
    clean = compute_stress(prob, snap, log=lines.append)                       # no failure: gpu-mg solves it (pocl counts as a GPU here)
    assert not [line for line in lines if "stopped" in line]
    assert np.max(np.abs(clean["vm_cases"] - ref["vm_cases"])) < 1e-6 * ref["vm_cases"].max()
    monkeypatch.setattr(GpuSolver, "set_design", inject("device_error", 0))
    lines.clear()
    got = compute_stress(prob, snap, log=lines.append)
    assert [line for line in lines if "stopped" in line] == [
        line for line in lines if line.startswith("solver: gpu-mg stopped in this solve (device_error: ")]
    assert lines[-1].endswith(f"); this solve goes on with {cpu}")
    assert np.array_equal(got["vm_cases"], ref["vm_cases"])                    # the CPU solver made it: the numbers of the reference


def test_the_stress_solve_of_an_explicit_gpu_mg_has_no_fallback(tmp_path, monkeypatch, pocl_is_a_gpu):
    from topopt.gpu_mg.model import GpuSolver
    from topopt.refine.stress import compute_stress
    prob = stress_problem(tmp_path, "gpu-mg", "g")
    monkeypatch.setattr(GpuSolver, "set_design", inject("device_error", 0))
    with pytest.raises(GpuFailure, match="reported an error"):
        compute_stress(prob, snapshot_of(prob))


def test_a_run_that_ended_on_a_cpu_solver_does_not_try_gpu_mg_in_its_stress_solve(tmp_path, monkeypatch, pocl_is_a_gpu):
    from topopt.gpu_mg import model
    from topopt.refine.stress import compute_stress
    from topopt.solver_choice import resolve_solver
    cpu = resolve_solver("auto")
    prob = stress_problem(tmp_path, "auto", "auto")
    monkeypatch.setattr(model, "solve_design", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gpu-mg was tried")))
    compute_stress(prob, snapshot_of(prob), solver=cpu)                         # run.json says the run ended on the CPU solver
    with pytest.raises(AssertionError, match="gpu-mg was tried"):
        compute_stress(prob, snapshot_of(prob), solver="gpu-mg")                # run.json says gpu-mg: the rule of a run holds


def test_the_stress_solve_checks_the_memory_of_the_cpu_solver_before_the_switch(tmp_path, monkeypatch, pocl_is_a_gpu):
    from topopt import cli
    from topopt.gpu_mg.model import GpuSolver
    from topopt.refine.stress import compute_stress
    prob = stress_problem(tmp_path, "auto", "auto")
    monkeypatch.setattr(GpuSolver, "set_design", inject("device_error", 0))
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: 1_000_000)
    with pytest.raises(UserError, match=r"solver gpu-mg stopped in the stress solve \(device_error: .*the solve cannot go on with .*: the memory estimate"):
        compute_stress(prob, snapshot_of(prob))


def test_the_callers_of_the_post_step_give_the_solver_of_the_run():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "topopt"
    for name, calls in (("stress_report.py", 2), ("refine/build.py", 1)):
        text = (root / name).read_text()
        assert text.count("= compute_stress(") == calls and text.count('solver=run.get("solver")') == calls, name
