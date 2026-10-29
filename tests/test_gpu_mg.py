"""Tests of the solver gpu-mg on an OpenCL device (CONTRACTS.md section 20).

They need pyopencl and one OpenCL device. On this project's Linux host that is pocl on the CPU
(`pip install -e .[gpu-test]`), so the kernels are checked on a CPU, not on a GPU. `conftest.py` stops the
session (Linux) or skips these tests (macOS) if a GPU is visible and `TOPOPT_TEST_REAL_GPU` is not "1".
Without pyopencl, the module skips. The module selects no device at import: the fixture `dev` does, and
it skips the test when there is no usable device.

Tolerances come from measured data.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse.linalg as spla
import yaml

pytest.importorskip("pyopencl")
pytestmark = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1

from test_optimize import make_case  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.config import UserError, load_config  # noqa: E402
from topopt.fe import Assembler  # noqa: E402
from topopt.gpu_mg import device, kernels  # noqa: E402
from topopt.gpu_mg.model import GpuModel, GpuParams  # noqa: E402
from topopt.optimize import E_MIN, Model, make_model, make_solver, reduced_system  # noqa: E402
from topopt.problem import build_problem  # noqa: E402
from topopt.snapshot import RunWriter  # noqa: E402
from topopt import solver_choice  # noqa: E402
from topopt.solver_choice import resolve_solver  # noqa: E402
from topopt.solver import LinearSolver  # noqa: E402

# Tolerances. Measured on pocl: the largest relative difference of
# the solution is 1.9e-10 (case C) and 5e-12 (case B); of an adjoint solve 1.1e-9; of the gradient 4.5e-10;
# of the compliance of 10 optimizer iterations 2.0e-10. The limits are 10 to 50 times these numbers.
SOLVE_RTOL = 1e-8
GRAD_RTOL = 1e-8
FD_BAR = 1e-7        # best central difference, absolute error over the largest gradient (measured 1.8e-8)
RUN_RTOL = 1e-8
SMALL = GpuParams(min_free=300)      # a small grid still gets a multigrid hierarchy of 3 levels

CASES = ["C", "B"]


@pytest.fixture(scope="session")
def dev():
    """The device that `auto` takes. The test is skipped when there is none (a UserError)."""
    try:
        return device.select("auto", "auto")[0]
    except UserError as e:
        pytest.skip(f"no usable OpenCL device: {e}")


def need_case(case, dev):
    """Skip a test of case B on a device without 64-bit floats."""
    if case == "B" and not dev.fp64:
        pytest.skip(f"case B needs 64-bit floats; {dev.name} has none")


def need_direct_reference():
    """Skip when the solver that `settings.solver: auto` gives is `amg`, an iterative solver.

    The stress test below compares gpu-mg with that solver at 1e-8. A direct solver (Cholesky, PARDISO) is exact to
    rounding, so the bar measures gpu-mg. AMG stops at its own tolerance: with it the difference is about 6e-7 of the
    peak stress, which measures AMG. `auto` gives AMG on Linux without pypardiso and on a Mac without the Accelerate shim.
    """
    if solver_choice.resolve_solver("auto") == "amg":
        pytest.skip("the reference is amg, an iterative solver: the difference to gpu-mg would measure the amg tolerance")


def case_yaml(tmp_path, solver="gpu-mg", case="auto", h=1.0, iterations=30, three_cases=False, gpu_tol=1e-8):
    """The test case with `solver: gpu-mg`. `gpu_tol` is 1e-8 here, the tolerance that the parity bars of this file are made for
    (the default of the setting is 1e-6 since G2)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = make_case(tmp_path, h=h, iterations=iterations)
    d = yaml.safe_load(p.read_text())
    d["settings"].update({"solver": solver, "gpu_case": case, "gpu_tol": gpu_tol})
    if three_cases:
        d["load_cases"].append({"name": "up", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, 2.0]}]})
    p.write_text(yaml.safe_dump(d))
    return p


class DirectModel(Model):
    """The reference: the default model with SuperLU on the assembled matrix (the same on every platform)."""

    def _make_solver(self, edof):
        return None, "superlu"

    def solve(self, K):
        return spla.splu(K.tocsc()).solve(self.f_red)


def direct_solution(prob, rho_of):
    """The reference model and the LU factor of the stiffness matrix of the density `rho_of(ref)`."""
    ref = DirectModel(prob)
    return ref, spla.splu(ref.stiffness(rho_of(ref)).tocsc())


@pytest.fixture(scope="module", params=CASES)
def model(request, tmp_path_factory, dev):
    need_case(request.param, dev)
    tmp = tmp_path_factory.mktemp(f"gpu_{request.param}")
    prob = build_problem(load_config(case_yaml(tmp, case=request.param, h=1.0, three_cases=True)))
    m = GpuModel(prob, None, SMALL)
    yield m
    m.solver.free()


@pytest.fixture(scope="module")
def design(model):
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.9, model.n_design)
    rho_tilde, rho = model.density(x, 2.0)
    return x, rho


# --- the kernel source ---------------------------------------------------------------------------


def test_case_c_source_has_no_64_bit_type_and_no_pragma():
    for nr in (1, 2):
        src = kernels.source("C", nr)
        assert "double" not in src
        assert "fp64" not in src.lower() and "pragma OPENCL EXTENSION" not in src
        assert "work_group_reduce" not in src                  # an OpenCL 2.0 function
    assert "double" in kernels.source("B", 2) and "cl_khr_fp64" in kernels.source("B", 2)


def test_case_c_program_builds_with_opencl_1_2(dev):
    import pyopencl as cl
    ctx = cl.Context([dev.obj])
    options = kernels.build_options("C")
    assert "-cl-std=CL1.2" in options and "CL2" not in options
    cl.Program(ctx, kernels.source("C", 2)).build(options=options)


def test_case_b_program_builds_with_opencl_1_2(dev):
    need_case("B", dev)
    import pyopencl as cl
    ctx = cl.Context([dev.obj])
    cl.Program(ctx, kernels.source("B", 2)).build(options=kernels.build_options("B"))


# --- the solve against a direct solver --------------------------------------------------------------


def test_solve_agrees_with_a_direct_solver(model, design):
    x, rho = design
    ref, lu = direct_solution(model.problem, lambda m: rho)
    U = model.solve(model.stiffness(rho))
    Ur = lu.solve(ref.f_red)
    assert U.shape == Ur.shape == (model.n_free, 3)
    for j in range(3):
        assert np.linalg.norm(U[:, j] - Ur[:, j]) / np.linalg.norm(Ur[:, j]) < SOLVE_RTOL
    c = np.einsum("ij,ij->j", model.f_red, U)
    cr = np.einsum("ij,ij->j", ref.f_red, Ur)
    assert np.max(np.abs(c - cr) / np.abs(cr)) < SOLVE_RTOL
    rec = model.solver.records[-1]
    assert rec["case"] == model.solver.case and rec["solves"] == 2 and rec["total_s"] > 0   # 3 load cases: 2 blocks of 2 columns


def test_adjoint_solve_with_other_columns_and_a_zero_column(model, design):
    """`solve(B, warm=False)`: one, three or four columns, and a column of zeros (used for adjoint loads)."""
    x, rho = design
    ref, lu = direct_solution(model.problem, lambda m: rho)
    model.solve(model.stiffness(rho))
    rng = np.random.default_rng(2)
    for k in (1, 3, 4):
        B = rng.standard_normal((model.n_free, k))
        if k == 4:
            B[:, 2] = 0.0
        X = model.solver.solve(B, warm=False)
        Xr = lu.solve(B)
        assert X.shape == Xr.shape
        assert np.linalg.norm(X - Xr) / np.linalg.norm(Xr) < SOLVE_RTOL
        assert np.all(X[:, 2] == 0.0) if k == 4 else True
    b = rng.standard_normal(model.n_free)
    assert model.solver.solve(b).shape == (model.n_free,)
    assert np.linalg.norm(model.solver.solve(b) - lu.solve(b)) / np.linalg.norm(lu.solve(b)) < SOLVE_RTOL


def test_point_solve_adjoint_uses_the_gpu_solver(model, design):
    x, rho = design
    ref, lu = direct_solution(model.problem, lambda m: rho)
    rho_tilde, rho2 = model.density(x, 2.0)
    U = model.solve(model.stiffness(rho2))
    point = model.design_point(x, 2.0, rho_tilde, rho2, U)
    rhs = np.random.default_rng(3).standard_normal(model.n_free)
    lam = point.solve_adjoint(rhs)
    assert np.linalg.norm(lam - lu.solve(rhs)) / np.linalg.norm(lu.solve(rhs)) < SOLVE_RTOL


# --- a design with a high contrast: both cases converge (review M2) -------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_a_binary_design_with_30_percent_solid_converges_in_both_cases(tmp_path, case, dev):
    """Rho is 0 or 1 (void is 1e-6 of solid), 7,260 free dofs, default parameters (tolerance 1e-8).

    Measured on pocl: the plain 64-bit PCG stops at 1.6e-6 after
    300 iterations on this design. Case B restarts in the refinement loop and ends with a true residual of
    1.1e-10 (2 steps). Case C: 6.95e-9 (4 steps), error of u 1.4e-7 against SuperLU. The bars are the tolerance
    for the true residual (the stop test) and 1e-6 for the error of u (7 times the measured 1.4e-7).
    """
    need_case(case, dev)
    prob = build_problem(load_config(case_yaml(tmp_path, case=case, h=1.0)))
    ref = DirectModel(prob)
    rng = np.random.default_rng(7)
    rng.uniform(0.3, 0.9, ref.n_design)               # the same stream as the hard-design check of the review
    rho = (rng.uniform(size=ref.n_design) < 0.3).astype(float)
    K = ref.stiffness(rho)
    Ur = spla.splu(K.tocsc()).solve(ref.f_red)
    m = GpuModel(prob, None, GpuParams(max_iter_first=300))      # the cap of the first pass of G1d (the default is 1000 since P2b); 300 per restart step
    try:
        U = m.solve(m.stiffness(rho))
        rec = m.solver.records[-1]
    finally:
        m.solver.free()
    res = K @ U - ref.f_red
    for j in range(U.shape[1]):
        assert np.linalg.norm(res[:, j]) / np.linalg.norm(ref.f_red[:, j]) < 1e-8
        assert np.linalg.norm(U[:, j] - Ur[:, j]) / np.linalg.norm(Ur[:, j]) < 1e-6
    its = rec["iterations"][0]
    if case == "B":
        assert its["executed"] == 300 and its["refine_steps"] >= 1      # the first pass ran out, the restart finished
    else:
        assert 1 < its["refine_steps"] <= 12


def test_case_b_gives_the_same_result_as_before_when_the_pcg_converges_in_the_first_pass(tmp_path, dev):
    """No restart step on an easy design: `refine_steps` is not in the record."""
    need_case("B", dev)
    prob = build_problem(load_config(case_yaml(tmp_path, case="B", h=2.0)))
    m = GpuModel(prob, None, SMALL)
    try:
        m.solve(m.stiffness(np.full(m.n_design, 0.6)))
        its = m.solver.records[-1]["iterations"][0]
    finally:
        m.solver.free()
    assert "refine_steps" not in its and its["executed"] < 100


# --- the gradient ---------------------------------------------------------------------------------


def test_gradient_passes_a_finite_difference_check(model, design):
    """Central differences with a step sweep, judged by the absolute error (KB: fd-check-noise-floor)."""
    x, _ = design
    beta = 2.0
    tight = GpuParams(min_free=300, tol=1e-10)
    old = model.solver.params
    model.solver.params = tight
    try:
        c0, dc, _v, _dv = model.evaluate(x, beta)
        rng = np.random.default_rng(5)
        order = np.argsort(-np.abs(dc))
        picks = list(order[:3]) + list(rng.choice(order[len(order) // 2:], 3, replace=False))
        scale = np.abs(dc).max()
        for e in picks:
            best = np.inf
            for step in (3e-3, 1e-3, 3e-4):
                xp, xm = x.copy(), x.copy()
                xp[e] += step
                xm[e] -= step
                fd = (model.evaluate(xp, beta)[0] - model.evaluate(xm, beta)[0]) / (2 * step)
                best = min(best, abs(fd - dc[e]))
            assert best < FD_BAR * scale, (e, best, scale)
    finally:
        model.solver.params = old


def test_gradient_equals_the_gradient_of_a_direct_solver(model, design):
    x, _ = design
    c, dc, v, dv = model.evaluate(x, 2.0)
    ref = DirectModel(model.problem)
    cr, dcr, vr, dvr = ref.evaluate(x, 2.0)
    assert abs(c - cr) / abs(cr) < SOLVE_RTOL
    assert np.max(np.abs(dc - dcr)) < GRAD_RTOL * np.abs(dcr).max()
    assert np.allclose(dv, dvr, rtol=0, atol=1e-14) and v == pytest.approx(vr)


# --- no matrix ----------------------------------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_the_run_does_not_assemble_or_factor_a_matrix(tmp_path, monkeypatch):
    """With gpu-mg, `Assembler`, `LinearSolver` and a sparse LU are never used."""
    import scipy.sparse.linalg as sla
    import topopt.optimize as opt

    def forbidden(*a, **k):
        raise AssertionError("a global stiffness matrix was assembled or factored")

    monkeypatch.setattr(Assembler, "__init__", forbidden)
    monkeypatch.setattr(Assembler, "K", forbidden)
    monkeypatch.setattr(LinearSolver, "__init__", forbidden)
    monkeypatch.setattr(LinearSolver, "factor", forbidden)
    monkeypatch.setattr(sla, "splu", forbidden)
    monkeypatch.setattr(opt, "make_solver", forbidden)
    prob = build_problem(load_config(case_yaml(tmp_path, h=2.0, iterations=3)))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    opt.run(prob, out)
    hist = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert len(hist) == 4
    m = make_model(prob)
    assert m.assembler is None and type(m.solver).__name__ == "GpuSolver" and m.solver_name == "gpu-mg"
    m.solver.free()


def test_other_solver_values_make_the_default_model(tmp_path):
    prob = build_problem(load_config(case_yaml(tmp_path, solver="amg", h=2.0, iterations=2)))
    m = make_model(prob)
    assert type(m) is Model and m.assembler is not None and m.solver_name == "amg"


@pytest.mark.usefixtures("dev")
def test_refine_style_solver_for_gpu_mg_is_a_fact_stop(tmp_path):
    prob = build_problem(load_config(case_yaml(tmp_path, h=2.0, iterations=2)))
    _e, _a, free = reduced_system(prob)
    with pytest.raises(UserError, match=r"matrix-free.*Options.*'auto'.*'amg'"):
        make_solver("gpu-mg", free, prob.shape, prob.h)


# --- work-group sizes come from queries -----------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_work_group_sizes_come_from_the_kernel_queries(tmp_path, monkeypatch):
    from topopt.gpu_mg import engine
    prob = build_problem(load_config(case_yaml(tmp_path, h=2.0, iterations=2)))
    m = GpuModel(prob, None, SMALL)
    g = m.solver.engine
    assert g.wg & (g.wg - 1) == 0 and g.wgv & (g.wgv - 1) == 0
    assert g.wg <= min(g.limits[n] for n in ("cheb", "cheb_dot", "spmv_dot_o", "resid_v"))
    assert g.wgv <= min(g.limits[n] for n in ("gemv", "p_update", "upd_xr"))
    x = np.full(m.n_design, 0.5)
    _, rho = m.density(x, 1.0)
    U0 = m.solve(m.stiffness(rho))
    m.solver.free()
    # A device with a small limit for one kernel: the sizes follow it, and the answer stays the same.
    real = engine.GpuMG.kernel_limit
    monkeypatch.setattr(engine.GpuMG, "kernel_limit",
                        lambda self, name: 8 if name == "cheb_dot" else (16 if name == "upd_xr" else real(self, name)))
    m2 = GpuModel(prob, None, SMALL)
    g2 = m2.solver.engine
    assert (g2.wg, g2.wgv) == (8, 16)
    U1 = m2.solve(m2.stiffness(rho))
    assert np.linalg.norm(U1 - U0) / np.linalg.norm(U0) < SOLVE_RTOL
    m2.solver.free()


# --- a device without 64-bit floats ----------------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_device_without_fp64_gives_case_c_and_stops_a_forced_case_b(tmp_path, monkeypatch):
    real = device.enumerate_devices()
    for d in real:
        d.fp64 = False
    monkeypatch.setattr(device, "enumerate_devices", lambda: real)
    prob = build_problem(load_config(case_yaml(tmp_path, case="auto", h=2.0, iterations=2)))
    m = GpuModel(prob, None, SMALL)
    assert m.solver.case == "C"
    U = m.solve(m.stiffness(np.full(m.n_design, 0.6)))
    assert np.all(np.isfinite(U))
    m.solver.free()
    forced_b = build_problem(load_config(case_yaml(tmp_path / "b", case="B", h=2.0, iterations=2)))
    with pytest.raises(UserError, match=r"no 64-bit float.*Options"):
        GpuModel(forced_b, None, SMALL)


def test_case_b_is_the_automatic_case_only_with_fp64(tmp_path, dev):
    prob = build_problem(load_config(case_yaml(tmp_path, case="auto", h=2.0, iterations=2)))
    m = GpuModel(prob, None, SMALL)
    assert m.solver.case == ("B" if dev.fp64 else "C")
    m.solver.free()


# --- errors are facts, there is no fallback -----------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_no_convergence_stops_with_facts_and_options(tmp_path):
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=2)))
    m = GpuModel(prob, None, GpuParams(min_free=300, max_iter_first=1, max_iter=1, max_refine=1))
    with pytest.raises(UserError, match=r"did not reach the relative residual.*case C.*Options.*'amg'"):
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    m.solver.free()


def test_a_grid_that_is_too_thin_is_a_fact(tmp_path):
    """Every coarse node is Dirichlet: no free dof on the coarsest level."""
    from topopt.gpu_mg.hierarchy import HierarchyError, HostMG
    prob = build_problem(load_config(case_yaml(tmp_path, h=2.0, iterations=2)))
    edof, aidx, _free = reduced_system(prob)
    fixed_all = np.arange(3 * int(np.prod(np.array(prob.shape) + 1)))
    with pytest.raises(HierarchyError, match="no free dof"):
        HostMG(prob.shape, edof, aidx, fixed_all, prob.h, 0.35, min_free=10)


# --- a whole run ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_run_agrees_with_the_default_solver(tmp_path, case, capsys, dev):
    need_case(case, dev)
    ref_yaml = case_yaml(tmp_path / "ref", solver="auto", h=2.0, iterations=10)
    gpu_yaml = case_yaml(tmp_path / "gpu", solver="gpu-mg", case=case, h=2.0, iterations=10)
    assert main(["run", str(ref_yaml), "--out", str(tmp_path / "r_ref"), "--quiet"]) == 0
    assert main(["run", str(gpu_yaml), "--out", str(tmp_path / "r_gpu"), "--quiet"]) == 0
    h_ref = [json.loads(line) for line in (tmp_path / "r_ref" / "history.jsonl").read_text().splitlines()]
    h_gpu = [json.loads(line) for line in (tmp_path / "r_gpu" / "history.jsonl").read_text().splitlines()]
    assert len(h_ref) == len(h_gpu) == 11
    c_ref = np.array([h["compliance"] for h in h_ref])
    c_gpu = np.array([h["compliance"] for h in h_gpu])
    direct = resolve_solver("auto") in ("cholesky", "pardiso")      # AMG is an iteration: a looser limit
    assert np.max(np.abs(c_gpu - c_ref) / c_ref) < (RUN_RTOL if direct else 1e-3)
    run_json = json.loads((tmp_path / "r_gpu" / "run.json").read_text())
    assert run_json["solver"] == "gpu-mg"
    log = (tmp_path / "r_gpu" / "run.log").read_text()
    assert f"case {case}" in log and "gpu-mg: device" in log
    # the refine page (stress step) uses the matrix-free solver too: no global matrix, the page is made
    assert "refine failed" not in log and "refine: " in log
    assert (tmp_path / "r_gpu" / "result" / "refine.html").is_file()
    timing = [json.loads(line) for line in (tmp_path / "r_gpu" / "gpu_mg.jsonl").read_text().splitlines()]
    assert len(timing) == 11
    assert {"setup_s", "upload_s", "solve_s", "map_s", "iterations", "case"} <= set(timing[0])
    status = json.loads((tmp_path / "r_gpu" / "status.json").read_text())
    assert status["state"] == "done"


@pytest.mark.usefixtures("dev")
def test_the_flag_solver_selects_gpu_mg_for_a_yaml_with_another_value(tmp_path):
    p = case_yaml(tmp_path, solver="auto", h=2.0, iterations=2)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet", "--solver", "gpu-mg"]) == 0
    assert json.loads((tmp_path / "r" / "run.json").read_text())["solver"] == "gpu-mg"


@pytest.mark.usefixtures("dev")
def test_a_run_with_a_bad_device_stops_before_the_solve(tmp_path, capsys):
    p = case_yaml(tmp_path, h=2.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    d["settings"]["gpu_device"] = "no such device"
    p.write_text(yaml.safe_dump(d))
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 1
    err = capsys.readouterr().err
    assert "settings.gpu_device is 'no such device'" in err and "Options" in err
    assert not (tmp_path / "r" / "history.jsonl").exists()


def test_environment_report_lists_the_real_device(dev, monkeypatch):
    from topopt.environment import environment_data
    monkeypatch.delenv("TOPOPT_AUTO_GPU", raising=False)      # the session sets 0: then the facts ask no device list (a child lists the pocl device here)
    data = environment_data()
    names = [d["name"] for d in data["opencl"]["devices"]]
    assert dev.name in names
    entry = next(d for d in data["opencl"]["devices"] if d["name"] == dev.name)
    assert entry["fp64"] is dev.fp64 and entry["kind"] == dev.kind
    assert data["solver_setting"]["gpu-mg"]["resolved"] == "gpu-mg" and data["experimental_solvers"] == {}


@pytest.mark.usefixtures("dev")
def test_free_twice_is_safe_and_a_solve_after_free_is_a_clear_error(tmp_path):
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=2)))
    m = GpuModel(prob, None, SMALL)
    ref = DirectModel(prob)
    rho = np.full(ref.n_design, 0.5)
    U = m.solve(m.stiffness(rho))
    assert np.all(np.isfinite(U))
    m.solver.free()
    m.solver.free()                      # the second call does nothing
    with pytest.raises(RuntimeError, match="free\\(\\) was called"):
        m.solver.solve(np.ones(m.n_free))


# --- the stress step of the post run uses the matrix-free solver (review M1) ------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_the_stress_step_with_gpu_mg_agrees_with_the_default_solver(tmp_path, case, dev, monkeypatch):
    """`compute_stress` for settings.solver gpu-mg: no global matrix, the same stress as the default solver.

    Measured on pocl, random snapshot, 1.0 to 2.0 mm: the largest difference of the von Mises stress over the
    peak is 1.9e-10 (B) and 2.5e-11 (C); of the compliance 6.9e-14 (B) and 9.8e-12 (C). The bars are 1e-8.
    """
    need_case(case, dev)
    need_direct_reference()
    import topopt.refine.stress as stress_mod
    ref_prob = build_problem(load_config(case_yaml(tmp_path / "ref", solver="auto", h=1.5)))
    gpu_prob = build_problem(load_config(case_yaml(tmp_path / "gpu", solver="gpu-mg", case=case, h=1.5)))
    snap = np.random.default_rng(11).integers(0, 256, int(np.prod(ref_prob.shape))).astype(np.uint8)
    ref = stress_mod.compute_stress(ref_prob, snap)

    def forbidden(*a, **k):
        raise AssertionError("a global stiffness matrix was assembled or factored")

    monkeypatch.setattr(stress_mod, "Assembler", forbidden)
    monkeypatch.setattr(stress_mod, "make_solver", forbidden)
    got = stress_mod.compute_stress(gpu_prob, snap)
    assert np.array_equal(got["centers"], ref["centers"]) and got["case_names"] == ref["case_names"]
    assert np.max(np.abs(got["vm_cases"] - ref["vm_cases"])) < 1e-8 * ref["vm_cases"].max()
    assert np.max(np.abs(got["compliance_cases"] - ref["compliance_cases"]) / ref["compliance_cases"]) < 1e-8


def test_the_stress_reference_must_be_a_direct_solver(monkeypatch):
    """Review M4: with `amg` as the `auto` solver the stress test is skipped, with the reason; a direct solver runs it."""
    monkeypatch.setattr(solver_choice, "resolve_solver", lambda setting: "amg")
    with pytest.raises(pytest.skip.Exception, match="the reference is amg, an iterative solver"):
        need_direct_reference()
    for direct in ("cholesky", "pardiso"):
        monkeypatch.setattr(solver_choice, "resolve_solver", lambda setting, d=direct: d)
        need_direct_reference()                      # no skip


def test_a_run_with_gpu_mg_makes_the_refine_page_and_the_metrics(tmp_path, dev):
    """The post run and the refine page work with gpu-mg (the page needs the stress step)."""
    p = case_yaml(tmp_path, h=2.0, iterations=4)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 0
    assert (tmp_path / "r" / "result" / "refine.html").is_file() and (tmp_path / "r" / "result" / "stress.npz").is_file()
    log = (tmp_path / "r" / "run.log").read_text()
    assert "refine failed" not in log and "matrix-free" not in log


# --- a device error is a fact with options (review m5) --------------------------------------------------------------


def _error_classes():
    import pyopencl as cl
    return [cl.RuntimeError, cl.MemoryError, cl.LogicError]


@pytest.mark.parametrize("where", ["upload_numerics", "pcg"])
@pytest.mark.parametrize("error", ["RuntimeError", "MemoryError", "LogicError"])
def test_a_device_error_is_a_user_error_and_releases_the_engine(tmp_path, monkeypatch, dev, where, error):
    import pyopencl as cl
    from topopt.gpu_mg import engine
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=2)))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    out.step(7, "solve")
    m = GpuModel(prob, out, SMALL)
    cls = getattr(cl, error)

    def broken(*a, **k):
        raise cls("clEnqueueNDRangeKernel failed: OUT_OF_RESOURCES")

    monkeypatch.setattr(engine.GpuMG, where, broken)
    with pytest.raises(UserError) as e:
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    text = str(e.value)
    assert dev.name in text and "case C" in text and "optimizer iteration 7" in text      # the facts
    assert "OUT_OF_RESOURCES" in text and error in text
    assert "settings.gpu_case to 'C'" in text and "another settings.gpu_device" in text and "'amg'" in text   # the options
    assert "buffers were released" in text and m.solver._freed
    with pytest.raises(RuntimeError, match="free\\(\\) was called"):
        m.solver.solve(np.ones(m.n_free))
    m.solver.free()                                                   # a second free does nothing


@pytest.mark.usefixtures("dev")
def test_an_error_that_is_not_an_opencl_error_is_not_changed(tmp_path, monkeypatch):
    from topopt.gpu_mg import engine
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=2)))
    m = GpuModel(prob, None, SMALL)
    monkeypatch.setattr(engine.GpuMG, "pcg", lambda *a, **k: (_ for _ in ()).throw(ValueError("a bug")))
    with pytest.raises(ValueError, match="a bug"):
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    monkeypatch.undo()
    m.solver.free()


@pytest.mark.usefixtures("dev")
def test_a_device_error_ends_a_run_with_exit_1_and_a_status(tmp_path, monkeypatch, capsys):
    import pyopencl as cl
    from topopt.gpu_mg import engine
    p = case_yaml(tmp_path, case="C", h=2.0, iterations=3)

    def broken(*a, **k):
        raise cl.RuntimeError("out of resources")

    monkeypatch.setattr(engine.GpuMG, "pcg", broken)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 1
    assert "reported an error" in capsys.readouterr().err
    status = json.loads((tmp_path / "r" / "status.json").read_text())
    assert status["state"] == "error" and "optimizer iteration 0" in status["message"]


# --- the timing file (review m8) -------------------------------------------------------------------------------------


def test_records_have_the_iteration_a_kind_and_the_adjoint_solves(tmp_path, dev):
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=2)))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    m = GpuModel(prob, out, SMALL)
    x = np.full(m.n_design, 0.5)
    rho_tilde, rho = m.density(x, 1.0)
    out.step(3, "solve")
    U = m.solve(m.stiffness(rho))                       # the solve of the loop
    out.step(3, "sensitivity")
    point = m.design_point(x, 1.0, rho_tilde, rho, U)
    point.solve_adjoint(np.random.default_rng(1).standard_normal(m.n_free))       # an adjoint solve
    out.step(3, "update")
    m.solve(m.stiffness(np.full(m.n_design, 0.4)))      # a trial design (the step acceptance of A2)
    out.step(4, "sensitivity")
    m.design_point(x, 1.0, rho_tilde, rho, U).solve_adjoint(np.ones(m.n_free))
    m.solver.free()
    lines = [json.loads(line) for line in (tmp_path / "run" / "gpu_mg.jsonl").read_text().splitlines()]
    assert [(r["line"], r["kind"], r["iteration"], r["solve"]) for r in lines] == [
        (0, "main", 3, 0), (1, "adjoint", 3, 0), (2, "trial", 3, 1), (3, "adjoint", 4, 1)]
    assert lines == m.solver.records
    for r in lines:
        assert {"setup_s", "upload_s", "solve_s", "map_s", "total_s", "iterations", "solves", "case"} <= set(r)
    adj = lines[1]
    assert adj["solve_s"] > 0 and adj["setup_s"] == 0 and adj["upload_s"] == 0 and adj["solves"] == 1   # the time is recorded
    assert adj["total_s"] == pytest.approx(adj["solve_s"] + adj["map_s"], abs=1e-4)


def test_the_loop_writes_one_main_record_for_each_iteration(tmp_path, dev):
    p = case_yaml(tmp_path, case="C", h=2.0, iterations=4)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 0
    lines = [json.loads(line) for line in (tmp_path / "r" / "gpu_mg.jsonl").read_text().splitlines()]
    assert [(r["kind"], r["iteration"]) for r in lines] == [("main", i) for i in range(5)]    # no adjoint: compliance only


@pytest.mark.parametrize("case", CASES)
def test_a_load_case_with_no_load_never_reaches_the_pcg_and_costs_no_inner_iterations(tmp_path, dev, case):
    """U21 review M1. A zero column would never meet the PCG stop test (`residual < rtol^2 * |b|^2` is `0 < 0`) and would run to the cap.
    It does not get there: `GpuSolver.solve` takes only the live columns and a short block repeats a live column (case B and case C), and
    `solve_refined` gives a column with a zero residual a copy of a live one (case C). So a case that applies no load (`LoadCase.applies_load`
    false) next to a normal case gives the same inner counts as the normal case alone, and its compliance is 0."""
    need_case(case, dev)

    def run(name, zero):
        p = case_yaml(tmp_path / name, case=case, h=2.0, iterations=3)
        d = yaml.safe_load(p.read_text())
        d["load_cases"] = d["load_cases"][:1]                                              # the case `down`
        if zero:
            d["load_cases"].insert(0, {"name": "zero", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, 0]}]})
        p.write_text(yaml.safe_dump(d))
        assert main(["run", str(p), "--out", str(tmp_path / name / "r"), "--quiet"]) == 0
        recs = [json.loads(line) for line in (tmp_path / name / "r" / "gpu_mg.jsonl").read_text().splitlines()]
        hist = [json.loads(line) for line in (tmp_path / name / "r" / "history.jsonl").read_text().splitlines()]
        return recs, hist
    alone, hist_alone = run("alone", zero=False)
    both, hist_both = run("both", zero=True)
    def counts(recs):       # the iterations of the solve of each pass (case C: `inner`, `refine_steps`; case B: `executed`; `pcg` has a copy)
        return [{k: r["iterations"][0][k] for k in ("inner", "refine_steps", "executed") if k in r["iterations"][0]} for r in recs]
    assert counts(alone) and counts(both) == counts(alone)                                                 # the same inner counts as the normal case
    assert all(r["accepted_by"] == "tol" and "failed" not in r for r in both)                              # no solve was stopped by the cap
    assert hist_both[0]["compliance_cases"][0] == 0.0                                                      # `zero` has compliance 0 ...
    assert hist_both[0]["compliance"] == pytest.approx(hist_alone[0]["compliance"], rel=RUN_RTOL)          # ... and its weight adds nothing


# --- the timing file on a resume (G1e, D1b) ---------------------------------------------------------------------------
# A new run starts a new `gpu_mg.jsonl`. A resume cuts it to the checkpoint iteration (like `history.jsonl`) and appends.


def timing_lines(folder):
    return [json.loads(line) for line in (folder / "gpu_mg.jsonl").read_text().splitlines()]


def timing_part(prob, out, iterations, resume_from=None, complete=False):
    """A model that makes one main record and one adjoint record for each iteration of `iterations` (no optimizer loop).
    `resume_from` is the checkpoint iteration of a resume (a stand-in for the `ResumePlan`, which has `ck.iteration` and
    `ck.meta["complete"]`)."""
    resume = None if resume_from is None else SimpleNamespace(ck=SimpleNamespace(iteration=resume_from, meta={"complete": complete}))
    m = GpuModel(prob, out, SMALL, resume=resume)
    for it in iterations:
        out.step(it, "solve")
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
        m.solver.solve(np.ones(m.n_free))                 # an adjoint solve: a record of kind "adjoint"
    m.solver.free()
    return m


def timing_case(tmp_path):
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=2)))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    return prob, out


def test_a_resume_keeps_the_records_up_to_the_checkpoint_and_numbers_the_new_ones_after_them(tmp_path, dev):
    prob, out = timing_case(tmp_path)
    timing_part(prob, out, range(0, 5))                                # the first part: iterations 0 to 4, 10 lines
    before = (tmp_path / "run" / "gpu_mg.jsonl").read_bytes().splitlines(keepends=True)
    m = timing_part(prob, out, range(3, 6), resume_from=2)             # the checkpoint is at iteration 2
    lines = timing_lines(tmp_path / "run")
    assert (tmp_path / "run" / "gpu_mg.jsonl").read_bytes().startswith(b"".join(before[:6]))   # iterations 0 to 2: byte for byte
    assert [(r["line"], r["kind"], r["iteration"], r["solve"]) for r in lines] == [
        (n, kind, it, it) for n, (it, kind) in enumerate((i, k) for i in range(6) for k in ("main", "adjoint"))]
    assert lines == m.solver.records                                   # the list in memory is the file
    log = (tmp_path / "run" / "run.log").read_text()
    assert "resume: gpu_mg.jsonl keeps 6 lines up to iteration 2, drops 4 lines for iterations 3 to 4; the lines of this run follow" in log


def test_a_new_run_replaces_the_timing_file(tmp_path, dev):
    prob, out = timing_case(tmp_path)
    timing_part(prob, out, range(0, 4))
    assert len(timing_lines(tmp_path / "run")) == 8
    timing_part(prob, out, range(0, 2))                                # no resume: a new file
    lines = timing_lines(tmp_path / "run")
    assert [r["line"] for r in lines] == [0, 1, 2, 3] and [r["iteration"] for r in lines] == [0, 0, 1, 1]
    assert "resume:" not in (tmp_path / "run" / "run.log").read_text()


def test_a_resume_drops_a_partial_last_line(tmp_path, dev):
    prob, out = timing_case(tmp_path)
    timing_part(prob, out, range(0, 3))
    with open(tmp_path / "run" / "gpu_mg.jsonl", "ab") as f:
        f.write(b'{"line": 6, "kind": "main", "itera')              # a kill in the middle of a write
    timing_part(prob, out, range(3, 4), resume_from=2)
    lines = timing_lines(tmp_path / "run")                             # every line parses, and the numbers have no gap
    assert [r["line"] for r in lines] == list(range(8)) and [r["iteration"] for r in lines] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert "drops 1 line(s) that do not parse (a partial last line, or a damaged line)" in (tmp_path / "run" / "run.log").read_text()


def test_a_resume_without_a_timing_file_starts_one_and_says_so(tmp_path, dev):
    prob, out = timing_case(tmp_path)
    timing_part(prob, out, range(0, 2), resume_from=1)
    assert [r["line"] for r in timing_lines(tmp_path / "run")] == [0, 1, 2, 3]
    assert "resume: gpu_mg.jsonl is missing: a new file starts" in (tmp_path / "run" / "run.log").read_text()


def test_the_resume_fact_for_an_empty_file_a_missing_newline_and_a_file_that_ends_early(tmp_path, dev):
    """Review n1 and n2: a real fact in each case, the same facts as `apply_plan` gives for `history.jsonl`."""
    prob, out = timing_case(tmp_path)
    path, log = tmp_path / "run" / "gpu_mg.jsonl", tmp_path / "run" / "run.log"
    path.write_bytes(b"")                                              # an empty file: no "up to iteration None"
    timing_part(prob, out, range(2, 3), resume_from=1)
    text = log.read_text()
    assert "gpu_mg.jsonl keeps no line, the file ends with no line, before the checkpoint iteration 1" in text and "None" not in text
    assert [r["line"] for r in timing_lines(tmp_path / "run")] == [0, 1]
    path.write_bytes(b"")
    timing_part(prob, out, range(0, 3), resume_from=2)                 # a first part with 3 passes (6 lines) ...
    path.write_bytes(path.read_bytes().rstrip(b"\n"))                  # ... and the kill came before the last newline
    timing_part(prob, out, range(3, 4), resume_from=2)
    text = log.read_text()
    assert "adds the newline that the last kept line lacked (the kill came before it)" in text
    assert [r["line"] for r in timing_lines(tmp_path / "run")] == list(range(8))
    lines = path.read_bytes().splitlines()                             # the file ends at iteration 1, the checkpoint is at 3
    path.write_bytes(b"\n".join(lines[:4]) + b"\n")
    timing_part(prob, out, range(4, 5), resume_from=3)
    assert "the file ends at iteration 1, before the checkpoint iteration 3: the lines in between are not in the file" in log.read_text()


def test_a_resume_removes_the_part_file_that_a_kill_inside_the_cut_leaves(tmp_path, dev):
    prob, out = timing_case(tmp_path)
    timing_part(prob, out, range(0, 3))
    (tmp_path / "run" / "gpu_mg.jsonl.tmp").write_text("part of a cut")
    before = (tmp_path / "run" / "gpu_mg.jsonl").read_bytes()
    timing_part(prob, out, range(3, 4), resume_from=2)                 # the checkpoint is at the last line: the cut rewrites nothing
    assert (tmp_path / "run" / "gpu_mg.jsonl").read_bytes().startswith(before)
    assert not (tmp_path / "run" / "gpu_mg.jsonl.tmp").exists()


@pytest.mark.parametrize("point, dropped", [("after_checkpoint:3", False), ("before_checkpoint:3", True)])
def test_a_killed_run_resumes_with_the_timing_records_of_both_parts(tmp_path, dev, point, dropped):
    """GOAL G1e 1: SIGKILL of a gpu-mg run (pocl CPU device), `--resume`, then the file holds the lines of both parts.
    `before_checkpoint:3`: the lines of pass 3 are on disk but its checkpoint is not, so the resume runs pass 3 again
    and the file keeps one set of lines for it. A fresh run in the same folder replaces the file."""
    from test_checkpoint import finish, killed, log, resume
    yaml_path = case_yaml(tmp_path, case="C", h=2.0, iterations=6)
    out = tmp_path / "r"
    killed(yaml_path, out, point)
    first = (out / "gpu_mg.jsonl").read_bytes()
    assert [r["iteration"] for r in timing_lines(out)] == [0, 1, 2, 3]       # the killed run wrote passes 0 to 3
    resume(yaml_path, out)
    data = (out / "gpu_mg.jsonl").read_bytes()
    kept = 3 if dropped else 4
    assert data.startswith(b"".join(first.splitlines(keepends=True)[:kept]))        # the first part is not cut
    lines = timing_lines(out)
    assert [(r["line"], r["kind"], r["iteration"], r["solve"]) for r in lines] == [(i, "main", i, i) for i in range(7)]
    assert f"resume: gpu_mg.jsonl keeps {kept} lines up to iteration {kept - 1}" in log(out)
    assert ("drops 1 line for iteration 3;" in log(out)) is dropped
    finish(yaml_path, out)                                                    # a new run in the same folder
    fresh = timing_lines(out)
    assert [r["line"] for r in fresh] == list(range(7)) and "resume:" not in log(out)
    assert (out / "gpu_mg.jsonl").read_bytes() != data                        # replaced, not extended: still 7 lines


# --- the OpenCL drivers are asked once (review m1) ------------------------------------------------------------------------


def test_a_run_asks_the_opencl_drivers_once(tmp_path, monkeypatch, dev):
    import pyopencl as cl
    calls = []
    real = cl.get_platforms
    monkeypatch.setattr(cl, "get_platforms", lambda: calls.append(1) or real())
    device.forget_devices()
    p = case_yaml(tmp_path, case="C", h=2.0, iterations=2)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 0
    assert len(calls) == 1


# --- the self-test of `settings.solver: auto` (GPU first) -------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_self_test_passes_on_the_device(dev, case):
    """The self-test runs on pocl: a small cantilever, gpu-mg against SuperLU, at the constants of `device`."""
    need_case(case, dev)
    device.forget_devices()
    r = device.self_test(dev, case)
    assert r["passed"] is True and r["error"] is None and r["case"] == case and r["device"] == dev.name
    assert r["rel_diff"] <= device.SELFTEST_MAX_DIFF and r["free_dofs"] == 3888 and r["seconds"] > 0
    assert r["tol"] == device.SELFTEST_TOL and r["max_diff"] == device.SELFTEST_MAX_DIFF
    assert device.self_test(dev, case) == r             # cached for the process


def test_self_test_fails_below_the_limit_and_names_the_numbers(dev):
    from topopt.gpu_mg import selftest
    r = selftest.run(dev, "C", device.SELFTEST_TOL, 1e-30)      # a limit that no solve reaches
    assert r["passed"] is False and r["error"] is None and 1e-30 < r["rel_diff"] < 1e-6


def test_self_test_reports_a_failure_of_the_solve_as_a_result(dev, monkeypatch):
    from topopt.gpu_mg import model, selftest

    def refuse(self, B, warm=True):
        raise model.SolveFailure("solver 'gpu-mg': the solve did not reach the relative residual", {})
    monkeypatch.setattr(model.GpuSolver, "solve", refuse)
    r = selftest.run(dev, "C", device.SELFTEST_TOL, device.SELFTEST_MAX_DIFF)
    assert r["passed"] is False and r["rel_diff"] is None and r["error"].startswith("SolveFailure: solver 'gpu-mg'") and r["seconds"] > 0


def test_self_test_reports_a_device_that_cannot_build_as_a_result(dev, monkeypatch):
    from topopt.gpu_mg import engine, selftest

    def no_build(*a, **k):
        raise engine.cl.Error("build failed")
    monkeypatch.setattr(engine, "GpuMG", no_build)
    r = selftest.run(dev, "C", device.SELFTEST_TOL, device.SELFTEST_MAX_DIFF)
    assert r["passed"] is False and r["rel_diff"] is None and "could not build or allocate the program" in r["error"]
