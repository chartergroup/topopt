"""P2b item 1: `settings.gpu_max_iter` (the cap of the first PCG pass of a gpu-mg solve) and the last restart step.

Part 1 needs no OpenCL: the setting in the loader, the checkpoint record and its resume fact, and the loosened last step with a
stand-in PCG whose residual falls by exactly the tolerance it is given. Part 2 runs the real engine on the OpenCL device (pocl on the
host of the tests): the field reaches `GpuParams`, the first pass takes it and a restart step takes the other cap, and one hard design
(the cantilever of `tests/test_gpu_mg.py` at 0.5 mm, 52,920 free dofs) needs fewer PCG iterations with the cap 1000 than with 300.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_optimize import make_case
from topopt import checkpoint as C
from topopt.cli import main
from topopt.config import GPU_MAX_ITER_DEFAULT, UserError, load_config
from topopt.problem import build_problem

opencl = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1


def yaml_with(tmp_path, **settings):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = make_case(tmp_path, h=2.0, iterations=3)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(settings)
    p.write_text(yaml.safe_dump(d))
    return p


# --- part 1: no OpenCL ------------------------------------------------------------------------------------------------


def test_the_setting_has_the_default_1000_and_takes_an_integer_for_any_solver(tmp_path):
    assert GPU_MAX_ITER_DEFAULT == 1000
    assert load_config(yaml_with(tmp_path / "a")).settings.gpu_max_iter == 1000
    for solver in ("auto", "pardiso", "amg", "gpu-mg"):
        assert load_config(yaml_with(tmp_path / solver, solver=solver, gpu_max_iter=250)).settings.gpu_max_iter == 250
    assert load_config(yaml_with(tmp_path / "one", gpu_max_iter=1)).settings.gpu_max_iter == 1
    assert load_config(yaml_with(tmp_path / "big", gpu_max_iter=10 ** 6)).settings.gpu_max_iter == 10 ** 6      # no upper bound


@pytest.mark.parametrize("bad", [0, -5, 1.5, 1000.0, True, "1000", None, [300]])
def test_a_value_that_is_not_an_integer_of_1_or_more_is_an_error_with_the_reason(tmp_path, bad):
    with pytest.raises(UserError) as e:
        load_config(yaml_with(tmp_path, gpu_max_iter=bad))
    text = str(e.value)
    assert "'settings.gpu_max_iter' must be an integer of 1 or more" in text
    assert "PCG iterations of the first pass of a gpu-mg solve" in text and f"got {bad!r}" in text and "(file: " in text


def test_the_effective_settings_leave_the_default_out_and_show_another_value(tmp_path):
    """So a checkpoint of a topopt without the key still resumes (with any solver); another value is a change."""
    assert "gpu_max_iter" not in C.effective_settings(load_config(yaml_with(tmp_path / "a")))["settings"]
    assert "gpu_max_iter" not in C.effective_settings(load_config(yaml_with(tmp_path / "b", gpu_max_iter=1000)))["settings"]
    assert C.effective_settings(load_config(yaml_with(tmp_path / "c", gpu_max_iter=500)))["settings"]["gpu_max_iter"] == 500


def test_a_changed_first_pass_cap_in_the_yaml_is_a_changed_input_that_a_resume_needs_the_override_for(tmp_path):
    """CONTRACTS 20 (review R1, N5): the text of a failed solve offers a larger `settings.gpu_max_iter`; a resume after that edit stops until
    `--allow-changed-inputs` is given. The key is not one of the soft keys (the notes of a resume)."""
    old = C.input_hashes(load_config(yaml_with(tmp_path / "a", solver="gpu-mg")))
    new = C.input_hashes(load_config(yaml_with(tmp_path / "b", solver="gpu-mg", gpu_max_iter=2000)))
    changes, notes = C.diff_inputs(old, new)
    assert [c for c in changes if c.startswith("settings.gpu_max_iter")] == ["settings.gpu_max_iter: <not set> -> 2000"] and not notes


def test_the_resume_fact_for_the_first_pass_cap(tmp_path):
    cfg = load_config(yaml_with(tmp_path, solver="gpu-mg"))
    fact = lambda meta, solver="gpu-mg", changes=(): C._gpu_max_iter_fact(meta, cfg, solver, list(changes))      # noqa: E731
    same = {"gpu_mg": {"tol": 1e-6, "backward_error": 1e-15, "max_iter": 1000}}
    assert fact(same) is None                                                       # the same cap: nothing to say
    g2 = fact({"gpu_mg": {"tol": 1e-6, "backward_error": 1e-15}})                   # a checkpoint of G2: the cap was 300 then
    assert g2 is not None and "capped the first PCG pass of a gpu-mg solve at 300 iterations" in g2
    assert "settings.gpu_max_iter 1000" in g2 and "written before settings.gpu_max_iter existed" in g2 and "case B only" in g2
    moved = fact({"gpu_mg": {"tol": 1e-6, "backward_error": 1e-15, "max_iter": 600}})
    assert moved is not None and "at 600 iterations" in moved and "settings.gpu_max_iter 1000" in moved
    assert "the YAML is the same: a default of topopt changed" in moved
    assert fact({"gpu_mg": {"max_iter": 600}}, changes=["settings.gpu_max_iter: 600 -> <not set>"]) is None     # the change line says it
    assert fact({"gpu_mg": {"max_iter": 600}}, solver="pardiso") is None            # the fact is for the solver gpu-mg
    assert fact({}) is None                                                         # a checkpoint from before G2: `_gpu_mg_fact` speaks


class ContractingPcg:
    """The device PCG as a stand-in on K = I: the residual of a step falls by exactly the factor `rtol` that the step is given."""

    def __init__(self):
        self.calls = []

    def pcg(self, rhs, rtol, maxit=300):
        self.calls.append((rtol, maxit))
        return (1.0 - rtol) * rhs, {"executed": 1}


class Identity:
    """The part of `HostMG` that the loop uses, for K = I on 4 nodes."""

    def __init__(self):
        self.levels = [SimpleNamespace(fix=np.zeros((4, 3), np.uint8))]

    def apply_fine(self, x, y):
        y[...] = x

    def apply_fine_abs(self, x, y):
        y[...] = np.abs(x)


def test_the_last_step_does_not_end_far_below_the_target_and_an_early_step_is_unchanged():
    pytest.importorskip("pyopencl")                     # `engine` imports pyopencl at the top (S-cl)
    from topopt.gpu_mg import engine
    B = np.ones((1, 4, 3))
    tol = 1e-5
    g_old, g_new = ContractingPcg(), ContractingPcg()
    _, old = engine.solve_refined(g_old, Identity(), B, tol, 1e-2)
    _, new = engine.solve_refined(g_new, Identity(), B, tol, 1e-2, last_step_margin=engine.LAST_STEP_MARGIN)
    # the loop of d19c7fb: every step at 1e-2; the residual goes 1, 1e-2, 1e-4, 1e-6 and ends 10 times below the target
    assert [t for t, _ in g_old.calls] == [1e-2, 1e-2, 1e-2] and old["rel_hist"][-1][0] == pytest.approx(1e-6)
    assert old["inner_tols"] == [1e-2, 1e-2, 1e-2]
    # the margin: the first two steps start at 1 and 1e-2 (above 50 times the target): unchanged. The last starts at 1e-4 and needs the
    # factor 0.5 * 1e-5 / 1e-4 = 0.05, so it ends at half the target.
    assert [t for t, _ in g_new.calls] == [1e-2, 1e-2, pytest.approx(0.05)] and new["inner_tols"] == [t for t, _ in g_new.calls]
    assert new["converged"] and new["steps"] == 3 and new["rel_hist"][-1][0] == pytest.approx(engine.LAST_STEP_MARGIN * tol)
    assert engine.LAST_STEP_MARGIN == 0.5


def test_the_loose_step_is_never_stricter_than_the_inner_tolerance_and_follows_the_largest_residual():
    pytest.importorskip("pyopencl")                     # `engine` imports pyopencl at the top (S-cl)
    from topopt.gpu_mg import engine
    g = ContractingPcg()
    B = np.ones((2, 4, 3))
    B[1] *= 1e-3                                       # a second column with another scale: the steps normalise each column
    _, info = engine.solve_refined(g, Identity(), B, 1e-9, 1e-1, last_step_margin=0.5)
    assert all(t >= 1e-1 for t in info["inner_tols"])                                # max(inner_rtol, ...): never stricter than 1e-1
    assert info["converged"] and max(info["rel_hist"][-1]) < 1e-9


# --- part 2: the real engine on the OpenCL device ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def dev():
    """The device that `auto` takes. The test is skipped when there is none (a UserError), or without pyopencl."""
    pytest.importorskip("pyopencl")
    from topopt.gpu_mg import device
    try:
        return device.select("auto", "auto")[0]
    except UserError as e:
        pytest.skip(f"no usable OpenCL device: {e}")


def gpu_model(tmp_path, params, *, h=1.0, case="B"):
    """The `GpuModel` of the test cantilever (`test_gpu_mg.case_yaml`) with explicit parameters."""
    import test_gpu_mg
    from topopt.gpu_mg.model import GpuModel
    return GpuModel(build_problem(load_config(test_gpu_mg.case_yaml(tmp_path, case=case, h=h, iterations=2))), None, params)


def binary_design(n_design, share=0.3):
    """The binary design of the G1d review (M2): 0 or 1 at random, `share` solid (void is 1e-6 of solid)."""
    rng = np.random.default_rng(7)
    rng.uniform(0.3, 0.9, n_design)
    return (rng.uniform(size=n_design) < share).astype(float)


@opencl
def test_the_setting_reaches_gpu_params_and_the_stress_solve_and_the_checkpoint_record(tmp_path, dev, monkeypatch):
    from topopt.gpu_mg import model as gm
    from topopt.gpu_mg.model import GpuModel, GpuParams
    from topopt.optimize import reduced_system
    assert GpuParams().max_iter_first == GPU_MAX_ITER_DEFAULT == 1000 and GpuParams().max_iter == 300      # the restart-step cap is the old one
    prob = build_problem(load_config(yaml_with(tmp_path, solver="gpu-mg", gpu_max_iter=777)))
    m = GpuModel(prob)
    try:
        assert (m._params.max_iter_first, m._params.max_iter) == (777, 300)
        assert m.checkpoint_meta()["gpu_mg"]["max_iter"] == 777
        assert m.solver.facts()["params"]["max_iter_first"] == 777
        seen = []
        init = gm.GpuSolver.__init__

        def spy(self, *a, **k):
            seen.append(a[7])
            init(self, *a, **k)
        monkeypatch.setattr(gm.GpuSolver, "__init__", spy)
        edof, aidx, free = reduced_system(prob)
        gm.solve_design(prob, edof, aidx, free, m.stiffness(np.full(m.n_design, 0.5)), m.f_red)
        assert (seen[0].max_iter_first, seen[0].max_iter) == (777, 300)                  # the one-off stress solve takes the setting too
    finally:
        m.solver.free()


@opencl
def test_the_first_pass_takes_the_setting_and_a_restart_step_takes_the_other_cap(tmp_path, dev, monkeypatch):
    import test_gpu_mg
    from topopt.gpu_mg import engine
    from topopt.gpu_mg.model import GpuParams
    test_gpu_mg.need_case("B", dev)
    calls = []
    real = engine.GpuMG.pcg

    def spy(self, B, rtol, maxit=300):
        calls.append((rtol, maxit))
        return real(self, B, rtol, maxit)
    monkeypatch.setattr(engine.GpuMG, "pcg", spy)
    m = gpu_model(tmp_path, GpuParams(max_iter_first=40, tol=1e-8))      # 7,260 free dofs, the binary design needs about 300 PCG iterations
    try:
        m.solve(m.stiffness(binary_design(m.n_design)))
        its = m.solver.records[-1]["iterations"][0]
    finally:
        m.solver.free()
    first, steps = calls[0], calls[1:]
    assert first[1] == 40 and its["executed"] == 40                              # the first pass: the setting
    assert steps and all(maxit == 300 for _, maxit in steps)                     # the restart steps: GpuParams.max_iter
    assert its["refine_steps"] == len(steps) and its["accepted_by"] == "tol" and its["true_rel"] < 1e-8


@opencl
@pytest.mark.parametrize("cap", [1, 7])
def test_case_c_has_no_first_pass_so_the_setting_does_not_apply(tmp_path, dev, monkeypatch, cap):
    from topopt.gpu_mg import engine
    from topopt.gpu_mg.model import GpuParams
    calls = []
    real = engine.GpuMG.pcg

    def spy(self, B, rtol, maxit=300):
        calls.append(maxit)
        return real(self, B, rtol, maxit)
    monkeypatch.setattr(engine.GpuMG, "pcg", spy)
    m = gpu_model(tmp_path, GpuParams(max_iter_first=cap, min_free=300), case="C", h=2.0)
    try:
        m.solve(m.stiffness(np.full(m.n_design, 0.6)))
    finally:
        m.solver.free()
    assert calls and set(calls) == {300}                                         # every step is capped by GpuParams.max_iter


@opencl
@pytest.mark.parametrize("case", ["B", "C"])
def test_the_run_log_says_which_cap_caps_what(tmp_path, dev, case):
    import test_gpu_mg
    test_gpu_mg.need_case(case, dev)
    p = test_gpu_mg.case_yaml(tmp_path, case=case, h=2.0, iterations=2)
    run = tmp_path / "run"
    assert main(["run", str(p), "--out", str(run), "--quiet"]) == 0
    line = [x for x in (run / "run.log").read_text().splitlines() if x.startswith("gpu-mg: device")][0]
    if case == "B":
        assert "PCG cap 1000 in the first pass (settings.gpu_max_iter) and 300 in a restart step" in line
    else:
        assert "PCG cap 300 in a refinement step (settings.gpu_max_iter does not apply to case C)" in line
    ck = C.read_checkpoint(run)
    assert ck.meta["gpu_mg"]["max_iter"] == 1000


@opencl
def test_a_failed_case_b_solve_names_the_cap_as_an_option(tmp_path, dev):
    import test_gpu_mg
    from topopt.gpu_mg.model import GpuParams
    test_gpu_mg.need_case("B", dev)
    m = gpu_model(tmp_path, GpuParams(min_free=300, max_iter_first=3, max_iter=1, max_refine=1), h=2.0)
    try:
        with pytest.raises(UserError) as e:
            m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    finally:
        m.solver.free()
    assert "a larger settings.gpu_max_iter (the cap of the first pass, now 3); a stiffer start design" in str(e.value)
    assert "the first pass is capped at 3, settings.gpu_max_iter" in str(e.value)


def hard_design(m, prob):
    """Filtered random blobs: 35 % of the design voxels solid (E 1), the rest void (E 1e-6). The `blobs` design of the P2 cap experiment."""
    from scipy.ndimage import gaussian_filter
    flat = m.active_index[m.design_pos]
    noise = gaussian_filter(np.random.default_rng(7).standard_normal(prob.shape), 1.5)
    thr = np.quantile(noise.reshape(-1)[flat], 0.65)
    return np.where(noise.reshape(-1)[flat] > thr, 1.0, 0.0)


def assert_cap_totals(totals):
    """`totals[cap] = (total PCG iterations, iterations of the first pass, restart steps or None)`. Cap 1000 needs fewer iterations in all than cap 300,
    and its first pass stays under the cap. The first pass stops on its recurrence residual, and on a real device (the Intel Arc, case B: 744 iterations)
    the true residual can end a little above the target, so one short restart step (8 iterations) follows. That is the right result (review R1, F3)."""
    assert totals[1000][0] < totals[300][0], totals
    assert totals[300][1] == 300 and totals[300][2] >= 1, totals                  # cap 300: restart steps were needed
    assert totals[1000][1] < 1000 and (totals[1000][2] or 0) <= 1, totals         # cap 1000: the first pass converged, at most one short restart step


def test_the_cap_comparison_takes_the_numbers_of_the_pocl_cpu_and_of_the_arc():
    """The two measured results of the hard design: pocl (748 in the first pass, no restart) and the Intel Arc (744, then one step of 8)."""
    assert_cap_totals({300: (1200, 300, 3), 1000: (748, 748, None)})
    assert_cap_totals({300: (1134, 300, 3), 1000: (752, 744, 1)})
    with pytest.raises(AssertionError):                                           # a first pass that needs two restart steps is not "converged"
        assert_cap_totals({300: (1134, 300, 3), 1000: (900, 600, 2)})
    with pytest.raises(AssertionError):
        assert_cap_totals({300: (800, 300, 3), 1000: (900, 744, 1)})              # more iterations in all with cap 1000


@opencl
def test_a_hard_design_needs_fewer_iterations_with_the_first_pass_cap_1000_than_with_300(tmp_path, dev):
    """The P2 cap experiment in short: case B, 52,920 free dofs, 4 levels, gpu_tol 1e-6,
    the blobs design. Cap 300: the first pass runs out and the restart steps hit their own cap again. Cap 1000: the first pass converges."""
    import test_gpu_mg
    from topopt.gpu_mg.model import GpuModel, GpuParams
    test_gpu_mg.need_case("B", dev)
    p = test_gpu_mg.case_yaml(tmp_path, case="B", h=0.5, iterations=1, gpu_tol=1e-6)
    d = yaml.safe_load(p.read_text())
    d["settings"]["penal"] = 6.0
    p.write_text(yaml.safe_dump(d))
    prob = build_problem(load_config(p))
    totals = {}
    for cap in (300, 1000):
        m = GpuModel(prob, None, GpuParams(min_free=300, tol=1e-6, max_iter_first=cap))
        try:
            m.solve(m.stiffness(hard_design(m, prob)))
            its = m.solver.records[-1]["iterations"][0]
        finally:
            m.solver.free()
        assert its["accepted_by"] == "tol" and its["true_rel"] < 1e-6
        totals[cap] = (its["executed"] + sum(its.get("inner", [])), its["executed"], its.get("refine_steps"))
    assert_cap_totals(totals)
    # measured on pocl (P2 research, before the last-step change): cap 300 gave 1,200 iterations (300 + 3 x 300), cap 2000 gave 748


@opencl
def test_the_last_step_ends_nearer_to_the_target_and_runs_fewer_iterations(tmp_path, dev):
    """The binary design of the review (7,260 free dofs): the first pass is capped at 300 and not converged, then the restart loop starts
    from its result. The same restart with and without the margin: the same start, fewer PCG iterations, the target still reached."""
    import test_gpu_mg
    from topopt.gpu_mg import engine
    from topopt.gpu_mg.model import GpuParams
    test_gpu_mg.need_case("B", dev)
    tol = 1e-8
    m = gpu_model(tmp_path, GpuParams(tol=tol, max_iter_first=300))
    try:
        s = m.solver
        s.set_design(m.stiffness(binary_design(m.n_design)))
        Bn = np.zeros((s.nr, s.nn0 * 3))
        for i in range(s.nr):
            Bn[i][s.fmap] = m.f_red[:, i]
        Bn = Bn.reshape(s.nr, s.nn0, 3)
        X, first = s.engine.pcg(Bn, tol, maxit=300)
        assert not first["converged"]                                            # the first pass ran out
        _, old = engine.solve_refined(s.engine, s.host, Bn, tol, 1e-2, maxit=300, u0=X)
        _, new = engine.solve_refined(s.engine, s.host, Bn, tol, 1e-2, maxit=300, u0=X, last_step_margin=engine.LAST_STEP_MARGIN)
    finally:
        m.solver.free()
    assert old["converged"] and new["converged"] and max(new["rel_hist"][-1]) < tol
    assert new["inner_tols"][-1] > 1e-2 >= min(new["inner_tols"][:-1], default=1e-2)    # the last step was loosened, an earlier one was not
    assert new["inner_iters_total"] < old["inner_iters_total"], (old["inner_iters"], new["inner_iters"])
    assert max(new["rel_hist"][-1]) > max(old["rel_hist"][-1])                    # and it ended nearer to the target than the old last step
