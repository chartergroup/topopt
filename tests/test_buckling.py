"""Tests of the linear buckling check (CONTRACTS.md section 33): topopt.buckling and topopt.buckling_voxel.

The Euler column is the reference: a fixed-free column with a uniform load on the free end has the critical
load `Pcr = pi^2 E I / (2 L)^2`. The load factor of the voxel model is `Pcr_FEM / P`. With `P = Pcr` it is
the ratio to Euler. The tolerance comes from a refinement series: the factor
converges from above (the full integration hex8 voxels are too stiff in bending).
"""
import dataclasses
import json
import time

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
import yaml

from topopt import buckling as bk
from topopt import solver_choice
from topopt import buckling_voxel as bv
from topopt.cli import main
from topopt.config import UserError, load_config
from topopt.fe import Assembler, edof_table, hex8_ke
from topopt.problem import build_problem
from topopt.solver import LinearSolver

E, NU = 110000.0, 0.34       # the column of the pipeline (validate_euler.py): 5 mm weak side, PLA-free metal
E_MIN, PENAL = 1e-6, 3.0
# The direct solver of this machine: Apple Cholesky on macOS arm64, PARDISO on Linux x86_64 with the extra.
DIRECT = ("cholesky" if solver_choice.is_macos_arm64() and solver_choice.accelerate_problem() is None
          else "pardiso" if solver_choice.is_linux_x86_64() and solver_choice.pypardiso_available() else None)
needs_direct = pytest.mark.skipif(
    DIRECT is None, reason="needs a direct solver (cholesky on macOS arm64, or pardiso on Linux x86_64)")


class Lu:
    """A factor of K from SciPy SuperLU: the same `.solve` as the direct backends, on every platform."""

    def __init__(self, K):
        self.lu = spla.splu(K.tocsc())

    def solve(self, b):
        return self.lu.solve(np.asarray(b, dtype=float))


class Column:
    """A fixed-free column of `a x b` voxels across (weak side `a`, 5 mm) and `20 a` voxels long.

    The base nodes are fixed. The load is a uniform traction on the end face with the resultant
    `P_factor * Pcr`. `box` puts the column in a larger design box. `halo` is the density outside the column.
    """

    def __init__(self, a, b, P_factor=1.0, box=None, halo=None, tension=False):
        self.a, self.b = a, b
        h = 5.0 / a
        nz = 20 * a
        nx, ny = box or (a, b)
        shape = (nx, ny, nz)
        i0, j0 = (nx - a) // 2, (ny - b) // 2
        rho = np.full(shape, 0.0 if halo is None else halo)
        rho[i0:i0 + a, j0:j0 + b, :] = 1.0
        edof, active_index = edof_table(shape, np.ones(shape, bool))
        ny1, nz1 = ny + 1, nz + 1

        def node(i, j, k):
            return (i * ny1 + j) * nz1 + k

        ii, jj = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
        base = node(ii, jj, 0).ravel()
        fixed = (3 * base[:, None] + np.arange(3)).ravel()
        free = np.setdiff1d(np.unique(edof), fixed)
        n_dof_all = 3 * (nx + 1) * (ny + 1) * (nz + 1)
        rho_e = rho.reshape(-1)[active_index]
        self.model = bv.VoxelModel(
            edof=edof, free=free, Ee=E * (E_MIN + rho_e ** PENAL * (1 - E_MIN)), weight=rho_e ** PENAL,
            rho=rho_e, h=h, E=E, nu=NU, n_dof_all=n_dof_all)
        self.h, self.length = h, nz * h
        self.Pcr = np.pi ** 2 * E * (b * h) * (a * h) ** 3 / 12.0 / (2 * self.length) ** 2
        P = P_factor * self.Pcr * (-1.0 if tension else 1.0)
        f = np.zeros(n_dof_all)
        for ia in range(a):
            for ib in range(b):
                for da in (0, 1):
                    for db in (0, 1):
                        f[3 * node(i0 + ia + da, j0 + ib + db, nz) + 2] -= P / (a * b) / 4.0
        self.f_red = f[free][:, None]
        self.K = self.model.asm.K(self.model.Ee)
        self.fac = Lu(self.K)

    def u_all(self):
        u = np.zeros(self.model.n_dof_all)
        u[self.model.free] = self.fac.solve(self.f_red[:, 0])
        return u

    def run(self, **kw):
        return bv.run_cases(self.model, self.fac, self.f_red, ["column"], **kw)[0]


# ---------------------------------------------------------------- Kg: symmetry and an independent assembly

def reference_kg(col):
    """Kg by plain loops over the 8 Gauss points and a SciPy coo assembly (no Assembler, no six blocks)."""
    m = col.model
    sig = bv.element_stress(col.u_all(), m.edof, m.E, m.nu, m.h)
    S = np.zeros((m.edof.shape[0], 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig[:, 5]
    g, h = 1 / np.sqrt(3), m.h
    sgn = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)   # the node order of fe.LOCAL_NODES
    ke = np.zeros((m.edof.shape[0], 8, 8))
    for gx in (-g, g):
        for gy in (-g, g):
            for gz in (-g, g):
                dn = np.stack([0.125 * sgn[:, 0] * (1 + sgn[:, 1] * gy) * (1 + sgn[:, 2] * gz),
                               0.125 * sgn[:, 1] * (1 + sgn[:, 0] * gx) * (1 + sgn[:, 2] * gz),
                               0.125 * sgn[:, 2] * (1 + sgn[:, 0] * gx) * (1 + sgn[:, 1] * gy)], axis=1) * 2 / h
                ke += np.einsum("ai,nij,bj->nab", dn, S, dn) * (h / 2) ** 3
    ke *= m.weight[:, None, None]
    full = np.einsum("nab,rc->narbc", ke, np.eye(3)).reshape(-1, 24, 24)
    n_all = m.n_dof_all
    rows = np.repeat(m.edof[:, :, None], 24, axis=2).ravel()
    cols = np.repeat(m.edof[:, None, :], 24, axis=1).ravel()
    Kg = sp.coo_matrix((full.ravel(), (rows, cols)), shape=(n_all, n_all)).tocsr()
    return Kg[m.free][:, m.free]


def test_kg_is_symmetric_and_equals_the_independent_assembly():
    col = Column(3, 4)
    Kg = bv.hex8_kg(col.model.asm, col.model.edof, col.u_all(), col.model.h, E, NU, col.model.weight)
    assert bk.sparse_symmetry_error(Kg) < 1e-12
    ref = reference_kg(col)
    assert abs(Kg - ref).max() / abs(ref).max() < 1e-12
    assert np.array_equal(col.model.asm.ke, hex8_ke(1.0, NU, col.h))   # the Assembler is as it was


def test_element_stress_of_the_column_is_the_axial_stress():
    col = Column(3, 4)
    sig = bv.element_stress(col.u_all(), col.model.edof, E, NU, col.h)
    area = (3 * col.h) * (4 * col.h)
    assert sig[:, 2].mean() == pytest.approx(-col.Pcr / area, rel=1e-3)
    k = np.arange(sig.shape[0]) % 60                               # the layer of each voxel (20 a = 60 layers)
    mid = (k > 20) & (k < 40)                                      # away from the clamp: a uniform state
    assert abs(sig[mid, 3:]).max() < 1e-3 * abs(sig[mid, 2]).mean()


def test_kg_blocks_scale_with_the_voxel_size():
    assert np.allclose(bv.hex8_kg_blocks(2.5), 2.5 * bv.hex8_kg_blocks(1.0))
    sym = bv.hex8_kg_blocks(1.0)
    assert np.allclose(sym, sym.transpose(0, 2, 1))


def test_kg_is_off_where_the_weight_is_zero():
    col = Column(3, 4, box=(7, 7), halo=0.0)
    Kg = bv.hex8_kg(col.model.asm, col.model.edof, col.u_all(), col.model.h, E, NU, col.model.weight)
    none = bv.hex8_kg(col.model.asm, col.model.edof, col.u_all(), col.model.h, E, NU, np.zeros_like(col.model.weight))
    assert abs(none).max() == 0.0 and abs(Kg).max() > 0.0


# ---------------------------------------------------------------- the Euler column

def test_euler_column_refinement_series():
    """The factor converges to Euler from above: +4.1 percent at 4 voxels across, +2.0 percent at 6 (measured)."""
    errs = []
    for a, b, limit in ((4, 6, 0.05), (6, 9, 0.025)):
        rec = Column(a, b).run(modes=3)
        lam = rec["load_factors"]
        assert rec["status"] == "ok" and len(lam) == 3
        err = lam[0] - 1.0
        errs.append(err)
        assert 0.0 < err < limit, (a, b, lam[0])
        assert rec["residual_max"] < 1e-6
        assert not rec["near_repeated"]            # a rectangular section has no double mode
    assert errs[1] < 0.6 * errs[0]                  # the error falls with the refinement (about 1/n^2)


def test_euler_load_factor_scales_with_one_over_the_load():
    base = Column(3, 4, P_factor=1.0).run(modes=2)["load_factors"][0]
    for p in (0.5, 2.0, 10.0):
        lam = Column(3, 4, P_factor=p).run(modes=2)["load_factors"][0]
        assert lam * p == pytest.approx(base, rel=1e-6)


def test_near_repeated_flag_on_a_square_column():
    rec = Column(4, 4).run(modes=3)
    lam = rec["load_factors"]
    assert lam[1] == pytest.approx(lam[0], rel=1e-6)       # the two bending axes
    assert rec["near_repeated"] and rec["near_repeated_modes"][0] == [1, 2]
    assert rec["min_rel_gap"] < 1e-6
    assert not Column(4, 6).run(modes=3)["near_repeated"]


@needs_direct
def test_euler_through_the_real_solver_backend():
    """The factor of the platform (`LinearSolver`: cholesky or pardiso) gives the same factors as SuperLU."""
    col = Column(4, 6)
    solver = LinearSolver(DIRECT, block3=True)
    solver.factor(col.K)
    try:
        rec = bv.run_cases(col.model, solver, col.f_red, ["column"], modes=3)[0]
    finally:
        solver.free()
    ref = col.run(modes=3)
    assert rec["load_factors"] == pytest.approx(ref["load_factors"], rel=1e-6)


# ---------------------------------------------------------------- void voxels add no geometric stiffness

@pytest.mark.parametrize("halo", [0.0, 0.01])
def test_column_in_a_mostly_empty_design_box(halo):
    """A column in a box of 8 x 8 voxels where the other 48 columns of voxels are void (or 1 percent)."""
    lone = Column(3, 4).run(modes=3)
    boxed = Column(3, 4, box=(8, 8), halo=halo).run(modes=3)
    assert boxed["load_factors"][0] == pytest.approx(lone["load_factors"][0], rel=1e-3)
    assert boxed["lowest_mode_energy_share_rho_lt_half"] < 1e-3      # the lowest mode is the column mode


def test_a_stress_weight_of_one_in_void_gives_a_false_mode():
    """The control of the test above: if void voxels carried Kg, the lowest mode would live in the void."""
    col = Column(3, 4, box=(5, 6), halo=0.0)
    lone = Column(3, 4).run(modes=3)["load_factors"][0]
    col.model.weight = np.ones_like(col.model.weight)
    rec = col.run(modes=1)
    assert rec["load_factors"][0] < 0.01 * lone              # measured: 0.0008 against 1.07
    assert rec["lowest_mode_energy_share_rho_lt_half"] > 0.5


# ---------------------------------------------------------------- facts: no compression, time budget, convergence

def test_tension_column_does_not_converge_in_a_few_restarts_and_the_result_is_a_fact():
    """The unit model of a column in tension: ARPACK gets 10 restarts and gives a fact, not an error."""
    t0 = time.perf_counter()
    rec = Column(3, 4, tension=True).run(modes=3, maxiter=10, max_seconds=5.0)
    assert time.perf_counter() - t0 < 5.0
    assert rec["status"] == bk.NOT_CONVERGED and rec["load_factors"] == []
    assert "did not converge" in rec["facts"][0]
    assert any("topopt buckling <run_dir> --modes" in o for o in rec["options"])


def single_voxel(sign):
    """One voxel, the base nodes fixed, an axial load of 10 N spread on the top nodes (12 free dofs: the dense path)."""
    shape = (1, 1, 1)
    edof, active_index = edof_table(shape, np.ones(shape, bool))
    fixed = np.array([3 * n + c for n in (0, 2, 4, 6) for c in range(3)])   # the nodes of the base (k = 0)
    free = np.setdiff1d(np.unique(edof), fixed)
    n_all = 3 * 8
    m = bv.VoxelModel(edof=edof, free=free, Ee=np.full(1, E), weight=np.ones(1), rho=np.ones(1), h=1.0, E=E,
                      nu=NU, n_dof_all=n_all)
    f = np.zeros(n_all)
    f[[3 * n + 2 for n in (1, 3, 5, 7)]] = sign * 10.0 / 4                 # the nodes of the top (k = 1)
    return m, f[free][:, None]


def test_tension_with_the_dense_path_has_no_positive_factor():
    """A real model (not a made-up matrix) with no compression: status no_positive_factor, a fact with options."""
    m, f = single_voxel(+1.0)
    rec = bv.run_cases(m, Lu(m.asm.K(m.Ee)), f, ["pull"], modes=3)[0]
    assert rec["status"] == bk.NO_POSITIVE and rec["load_factors"] == [] and rec["facts"] and rec["options"]
    m, f = single_voxel(-1.0)
    assert bv.run_cases(m, Lu(m.asm.K(m.Ee)), f, ["push"], modes=3)[0]["status"] == bk.OK


def test_time_budget_stops_the_eigen_solve_with_a_fact():
    col = Column(3, 4)
    res = bk.buckling_modes(bv.hex8_kg(col.model.asm, col.model.edof, col.u_all(), col.h, E, NU, col.model.weight),
                            col.K, col.fac, k=3, max_seconds=0.0)
    assert res["status"] == bk.TIME_BUDGET and res["lam"].size == 0
    rec = bk.case_summary(res, "c", 3)
    assert rec["status"] == bk.TIME_BUDGET and "time budget" in rec["facts"][0] and rec["options"]


def _pencil(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    K = sp.csc_matrix(A @ A.T + n * np.eye(n))
    G = rng.standard_normal((n, n))
    Kg = sp.csc_matrix(-(G + G.T) / 2.0)
    return Kg, K


def test_eigen_wrapper_matches_a_dense_solution():
    import scipy.linalg as sla
    Kg, K = _pencil(80)
    res = bk.buckling_modes(Kg, K, Lu(K), k=4)
    mu = np.sort(sla.eigh(-Kg.toarray(), K.toarray(), eigvals_only=True))[::-1][:4]
    assert res["status"] == bk.OK and res["method"] == "arpack"
    assert np.sort(res["mu"])[::-1] == pytest.approx(mu, rel=1e-6)
    assert res["lam"] == pytest.approx(1.0 / mu, rel=1e-6)
    assert np.all(np.diff(res["lam"]) >= 0) and res["residual"].max() < 1e-6


def test_eigen_wrapper_uses_the_dense_solver_for_a_tiny_problem():
    Kg, K = _pencil(10)
    res = bk.buckling_modes(Kg, K, Lu(K), k=3)
    assert res["method"] == "dense" and res["lam"].size == 3


def test_a_factor_of_zero_or_below_is_inf():
    n = 10                                                  # the dense solver gives all pairs
    K = sp.identity(n, format="csc")
    Kg = sp.diags(np.r_[np.full(n - 1, 1.0), -2.0]).tocsc()       # -Kg has one positive and n-1 negative values
    res = bk.buckling_modes(Kg, K, Lu(K), k=3)
    assert res["lam"][0] == pytest.approx(0.5) and np.isinf(res["lam"][1:]).all()
    rec = bk.case_summary(res, "c", 3)
    assert rec["load_factors"] == [pytest.approx(0.5)] and "mu <= 0" in rec["facts"][0]
    none = bk.case_summary(bk.buckling_modes(sp.identity(n, format="csc"), K, Lu(K), k=2), "t", 2)   # tension: mu < 0
    assert none["status"] == bk.NO_POSITIVE and none["load_factors"] == [] and none["options"]


def test_near_repeated_function():
    assert bk.near_repeated([1.0, 1.001, 3.0]) == (True, pytest.approx(0.001), [[1, 2]])
    flag, gap, pairs = bk.near_repeated([1.0, 1.5, 3.0, np.inf])
    assert (flag, pairs) == (False, []) and gap == pytest.approx(0.5)
    assert bk.near_repeated([2.0]) == (False, None, [])


def test_write_metrics_key_goes_last_and_replaces(tmp_path):
    (tmp_path / "result").mkdir()
    (tmp_path / "result" / "metrics.json").write_text(json.dumps({"a": 1, "buckling": {"old": True}, "z": 2}))
    path = bk.write_metrics_key(tmp_path, {"status": "done"})
    m = json.loads(path.read_text())
    assert list(m) == ["a", "z", "buckling"] and m["buckling"] == {"status": "done"}
    (tmp_path / "result" / "metrics.json").unlink()
    assert bk.write_metrics_key(tmp_path, {"x": 1}).name == "buckling.json"


# ---------------------------------------------------------------- the check of a run: config, skips, CLI

def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def make_case(tmp_path, h=2.0, iterations=2, extra=None, solver=None):
    """A small cantilever (the case of test_optimize) with an optional `buckling` block."""
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", (18, 4, 0), (20, 6, 3))
    d = {"name": "cantilever",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]},
                        {"name": "side", "forces": [{"region": "tip", "force": [1.0, 3.0, 0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.5, "iterations": iterations,
                      "beta_schedule": [[0, 1], [1, 2]]}}
    if solver:
        d["settings"]["solver"] = solver
    if extra is not None:
        d["buckling"] = extra
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def uniform_snapshot(problem, value=255):
    return np.full(int(np.prod(problem.shape)), value, dtype=np.uint8)


def test_config_default_is_off_and_the_block_is_read(tmp_path):
    cfg = load_config(make_case(tmp_path))
    assert cfg.buckling == dataclasses.replace(cfg.buckling) and cfg.buckling.check is False
    assert (cfg.buckling.modes, cfg.buckling.density, cfg.buckling.max_gb) == (6, "printed", None)
    on = load_config(make_case(tmp_path, extra={"check": True, "modes": 3, "density": "gray",
                                                "max_seconds": 30, "max_gb": 4}))
    b = on.buckling
    assert (b.check, b.modes, b.density, b.max_seconds, b.max_gb) == (True, 3, "gray", 30.0, 4.0)


@pytest.mark.parametrize("block,text", [
    ({"chek": True}, "unknown key 'chek'"),
    ({"check": "yes"}, "'buckling.check' must be true or false"),
    ({"density": "blurred"}, "'buckling.density' must be printed or gray"),
    ({"modes": 0}, "'buckling.modes'"),
    ({"modes": 2.5}, "'buckling.modes' must be an integer"),
    ({"max_seconds": 0}, "'buckling.max_seconds'"),
    ({"max_gb": -1}, "'buckling.max_gb'"),
])
def test_config_rejects_a_bad_block(tmp_path, block, text):
    with pytest.raises(UserError, match=text):
        load_config(make_case(tmp_path, extra=block))


def test_amg_gives_a_fact_with_options_and_no_result(tmp_path):
    problem = build_problem(load_config(make_case(tmp_path, solver="amg")))
    res = bv.check_problem(problem, uniform_snapshot(problem))
    assert res["status"] == "skipped" and res["reason"] == "solver_not_direct" and res["solver"] == "amg"
    assert res["cases"] == [] and res["lowest_load_factor"] is None
    assert "no such factor" in res["facts"][0] and " GB" in res["facts"][0] and "GiB" not in res["facts"][0]
    assert any("cholesky" in o for o in res["options"]) and any("pardiso" in o for o in res["options"])
    assert res["memory_estimate_bytes"] > 0 and res["free_dofs"] > 0


def test_memory_budget_and_voxel_guard_give_facts(tmp_path):
    problem = build_problem(load_config(make_case(tmp_path)))
    snap = uniform_snapshot(problem)
    mem = bv.check_problem(problem, snap, max_gb=1e-6)
    guard = bv.check_problem(
        dataclasses.replace(problem, config=dataclasses.replace(
            problem.config, settings=dataclasses.replace(problem.config.settings, max_active_voxels=10))), snap)
    if mem["status"] == "skipped" and mem["reason"] == "solver_not_direct":
        pytest.skip("the solver of this machine is not direct: the memory budget is not reached")
    assert mem["status"] == "skipped" and mem["reason"] == "memory" and "buckling.max_gb" in mem["facts"][0]
    assert "GB" in mem["facts"][0] and "GiB" not in mem["facts"][0]       # one unit: the unit of max_gb
    assert guard["status"] == "skipped" and guard["reason"] == "voxel_guard"
    assert "settings.max_active_voxels" in guard["options"][0]


def test_memory_estimate_is_near_the_measured_hook_peak():
    est = bv.estimate_bytes(112_200, 115_836)
    assert 1.9e9 < est < 2.4e9          # the hook measured 1.92 GB
    assert bv.estimate_bytes(112_200, 115_836, modes=20) > est


@needs_direct
def test_check_problem_on_the_platform_solver(tmp_path):
    problem = build_problem(load_config(make_case(tmp_path)))
    res = bv.check_problem(problem, uniform_snapshot(problem), modes=3)
    assert res["status"] == "done" and res["solver"] in bv.DIRECT_SOLVERS and res["density"] == "printed"
    assert [c["name"] for c in res["cases"]] == ["down", "side"]
    for c in res["cases"]:
        assert c["status"] == "ok" and len(c["load_factors"]) == 3 and c["load_factors"][0] > 0
    assert res["lowest_load_factor"]["value"] == min(c["load_factors"][0] for c in res["cases"])
    assert res["peak_rss_mb"] > 0 and res["seconds"] > 0 and res["factor_seconds"] >= 0
    gray = bv.check_problem(problem, uniform_snapshot(problem, 200), modes=2, density="gray")
    assert gray["density"] == "gray" and gray["threshold"] is None and gray["gray_voxels"] > 0


def run_cli(tmp_path, *flags, extra=None, solver=None):
    out = tmp_path / "run"
    code = main(["run", str(make_case(tmp_path, extra=extra, solver=solver)), "--out", str(out), "--quiet", *flags])
    return code, out, json.loads((out / "result" / "metrics.json").read_text())


def test_off_by_default_nothing_is_imported_or_added(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the check ran")
    monkeypatch.setattr(bv, "run_check", boom)
    code, out, metrics = run_cli(tmp_path)
    assert code == 0 and "buckling" not in metrics
    assert "buckling" not in (out / "run.log").read_text()
    second = tmp_path / "second"
    second.mkdir()
    code, out, metrics = run_cli(second, "--no-buckling", extra={"check": True})
    assert code == 0 and "buckling" not in metrics        # the flag turns a YAML check off


def test_a_check_that_fails_never_stops_the_run(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("something broke")
    monkeypatch.setattr(bv, "run_check", boom)
    code, out, metrics = run_cli(tmp_path, "--buckling")
    assert code == 0 and "buckling" not in metrics
    assert "warning: buckling check failed: RuntimeError: something broke" in (out / "run.log").read_text()
    assert json.loads((out / "status.json").read_text())["state"] == "done"


def test_cli_amg_run_records_the_fact_and_goes_on(tmp_path):
    code, out, metrics = run_cli(tmp_path, "--buckling", solver="amg")
    b = metrics["buckling"]
    assert code == 0 and list(metrics)[-1] == "buckling"
    assert b["status"] == "skipped" and b["reason"] == "solver_not_direct" and b["cases"] == []
    log = (out / "run.log").read_text()
    assert "buckling check skipped" in log and "warning: buckling: The check needs a direct factor" in log
    assert "buckling: option: Set settings.solver: " in log or "buckling: option: cholesky: " in log
    assert json.loads((out / "status.json").read_text())["state"] == "done"


@needs_direct
def test_cli_check_adds_the_key_last_and_leaves_the_run_alone(tmp_path):
    code, out, on = run_cli(tmp_path, "--buckling", extra={"modes": 2})
    b = on["buckling"]
    assert code == 0 and list(on)[-1] == "buckling" and b["status"] == "done" and b["modes"] == 2
    log = (out / "run.log").read_text()
    assert "buckling check (voxel estimate, printed design," in log          # the label of a cheap number (n2)
    assert "strain energy of the lowest mode in voxels below 0.5:" in log     # where the mode lives, in each case line
    assert b["loose_solid_voxels"] == 0 and all("lowest_mode_energy_box_mm" in c for c in b["cases"])
    assert all(c["status"] == "ok" and len(c["load_factors"]) == 2 for c in b["cases"])
    other = tmp_path / "other"
    other.mkdir()
    code, _, off = run_cli(other)
    assert code == 0
    for key in ("iterations", "solver", "watertight", "triangles", "mass_final_g"):
        assert on[key] == off[key]
    assert on["compliance_final"] == pytest.approx(off["compliance_final"], rel=1e-8)
    assert set(on) - set(off) == {"buckling"} and set(off) - set(on) == set()


# ---------------------------------------------------------------- where the lowest mode lives (false modes are facts)

class Grid:
    """A voxel model from a density grid, the base layer (k = 0) fixed, a load on the top faces of some columns.

    `cells` lists the (i, j) columns that carry the load `P` (-z, spread over the top nodes of layer `kz`).
    The model knows the loose voxels and the voxel centres, like `build_model`.
    """

    def __init__(self, rho, h, cells, P, kz=None, density="printed"):
        shape = rho.shape
        nx, ny, nz = shape
        self.rho_grid = rho
        edof, active_index = edof_table(shape, np.ones(shape, bool))

        def node(i, j, k):
            return (i * (ny + 1) + j) * (nz + 1) + k

        ii, jj = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
        fixed = (3 * node(ii, jj, 0).ravel()[:, None] + np.arange(3)).ravel()
        free = np.setdiff1d(np.unique(edof), fixed)
        n_all = 3 * (nx + 1) * (ny + 1) * (nz + 1)
        rho_e = rho.reshape(-1)[active_index].astype(float)
        anchor = np.zeros(shape, bool)
        anchor[:, :, 0] = True
        loose = bv.loose_mask(rho >= 0.5, anchor).reshape(-1)[active_index]
        ijk = np.stack(np.unravel_index(active_index, shape), axis=1)
        self.model = bv.VoxelModel(
            edof=edof, free=free, Ee=E * (E_MIN + rho_e ** PENAL * (1 - E_MIN)), weight=rho_e ** PENAL, rho=rho_e,
            h=h, E=E, nu=NU, n_dof_all=n_all, loose=loose, centers=h * (ijk + 0.5), density=density)
        kz = nz if kz is None else kz
        f = np.zeros(n_all)
        for (i, j) in cells:
            for di in (0, 1):
                for dj in (0, 1):
                    f[3 * node(i + di, j + dj, kz) + 2] -= P / len(cells) / 4
        self.f_red = f[free][:, None]
        self.K = self.model.asm.K(self.model.Ee)
        self.fac = Lu(self.K)

    def run(self, **kw):
        return bv.run_cases(self.model, self.fac, self.f_red, ["case"], K=self.K, **kw)[0]


def euler_load(a, b, h, length):
    return np.pi ** 2 * E * (b * h) * (a * h) ** 3 / 12.0 / (2 * length) ** 2


def column_grid(void_layer=False):
    a, b, h = 4, 6, 5.0 / 4
    nz = 20 * a
    rho = np.ones((a, b, nz))
    if void_layer:
        rho[:, :, nz // 2] = 0.0         # a void layer: the top half is a loose piece
    return rho, h, [(i, j) for i in range(a) for j in range(b)], euler_load(a, b, h, nz * h)


def test_loose_mask_uses_face_connections():
    solid = np.zeros((6, 3, 3), bool)
    solid[0:2, 1, 1] = True                  # anchored: a bar of two voxels along x from the anchor
    solid[3, 1, 1] = True                    # one voxel with a gap to the bar: loose
    solid[2, 1, 2] = True                    # touches the bar (voxel (1,1,1)) only on an edge: loose
    anchor = np.zeros_like(solid)
    anchor[0, 1, 1] = True
    loose = bv.loose_mask(solid, anchor)
    assert loose.sum() == 2 and loose[3, 1, 1] and loose[2, 1, 2] and not loose[0:2, 1, 1].any()


def test_an_intact_column_has_no_false_mode_fact_and_says_where_the_mode_is():
    rho, h, cells, P = column_grid()
    g = Grid(rho, h, cells, P)
    rec = g.run(modes=2)
    assert rec["status"] == "ok" and rec["facts"] == [] and not g.model.loose.any()
    assert rec["lowest_mode_energy_share_rho_lt_half"] == 0.0 and rec["lowest_mode_motion_share_loose"] == 0.0
    assert rec["shares_of_mode"] == "lowest"
    (x0, y0, z0), (x1, y1, z1) = rec["lowest_mode_energy_box_mm"]
    cx, cy, cz = rec["lowest_mode_energy_center_mm"]
    assert 0 < x0 <= cx <= x1 < 5.0 and 0 < y0 <= cy <= y1 < 7.5 and 0 < z0 <= cz <= z1 < 100.0
    assert z0 < 50.0                          # a cantilever column bends most near the clamp


def test_a_loose_piece_gives_the_false_mode_facts():
    """Printed design with a void layer: the top half is held only by E_min. Review case: lambda 1e-4, share 0.9999."""
    rho, h, cells, P = column_grid(void_layer=True)
    g = Grid(rho, h, cells, P)
    rec = g.run(modes=2)
    assert g.model.loose.sum() == 4 * 6 * 39
    assert rec["load_factors"][0] < 1e-2                      # the factor is in the record: no verdict
    assert rec["lowest_mode_energy_share_rho_lt_half"] > 0.9 and rec["lowest_mode_motion_share_loose"] > 0.5
    text = " ".join(rec["facts"])
    assert "percent of the strain energy of the lowest mode is in voxels with a density below 0.5" in text
    assert "936 solid voxels" in text and "not face-connected to a fixed region" in text
    assert "lie in the box" in text and "no verdict" not in text.lower()
    assert any("topopt buckling <run_dir> --density gray" in o for o in rec["options"])
    assert any("topopt refine" in o for o in rec["options"])


def test_an_edge_only_strut_gives_the_false_mode_facts():
    """A staircase strut whose voxels touch on edges only: the nodes join them, no face carries stress."""
    h, n = 1.0, 24
    rho = np.zeros((n + 2, 2, n + 2))
    rho[:, :, 0] = 1.0                                       # base plate
    rho[n - 1:n + 2, :, n + 1] = 1.0                         # a small plate on top of the strut
    for k in range(1, n + 1):
        rho[k, :, k] = 1.0
    cells = [(i, j) for i in range(n - 1, n + 2) for j in range(2)]
    g = Grid(rho, h, cells, 1.0)
    rec = g.run(modes=2)
    assert g.model.loose.sum() >= n                          # the strut and the top plate are loose
    assert rec["lowest_mode_energy_share_rho_lt_half"] > 0.9 and rec["lowest_mode_motion_share_loose"] > 0.9
    assert any("not face-connected" in f for f in rec["facts"]) and rec["options"]


@pytest.mark.parametrize("g_member", [0.2, 0.45])
def test_a_thin_gray_member_gives_the_false_mode_fact_in_the_gray_density_only(g_member):
    """Twin pillars: A is solid, B is thin and gray (0.2 or 0.45). The gray model has a low mode in B (review: 0.63
    and 0.55 against 0.9855 for the printed part). The printed model of the same design has no such mode."""
    h, nz = 1.25, 80
    tw = np.zeros((10, 4, nz + 2))
    tw[0:4, 0:4, :nz] = 1.0
    tw[8:10, 1:3, :nz] = g_member
    tw[:, :, nz:] = 1.0                                      # a stiff cap of two layers
    P = euler_load(4, 4, h, nz * h)
    cells = [(i, j) for i in range(10) for j in range(4)]
    gray = Grid(tw, h, cells, P, density="gray").run(modes=2)
    assert gray["lowest_mode_energy_share_rho_lt_half"] > 0.9 and gray["load_factors"][0] < 0.7
    assert any("thin gray member" in f for f in gray["facts"])
    assert any("--density printed" in o for o in gray["options"])    # the option names the other density
    printed = np.where(tw < 0.5, 0.0, tw)
    rec = Grid(printed, h, cells, P).run(modes=2)
    assert rec["facts"] == [] and rec["load_factors"][0] > 0.95 and rec["lowest_mode_energy_share_rho_lt_half"] < 0.01


def test_a_partial_result_marks_the_shares_as_those_of_the_lowest_converged_mode(monkeypatch):
    real = bk.buckling_modes

    def not_converged(*a, **k):
        res = real(*a, **k)
        res["status"] = bk.NOT_CONVERGED
        return res

    monkeypatch.setattr(bk, "buckling_modes", not_converged)
    rec = Column(3, 4).run(modes=3)
    assert rec["status"] == bk.NOT_CONVERGED and rec["load_factors"] == [] and rec["partial_load_factors"]
    assert rec["shares_of_mode"].startswith("lowest converged") and "lowest_mode_energy_share_rho_lt_half" in rec


def test_a_large_residual_is_a_fact_even_when_the_factors_are_returned():
    base = dict(status=bk.OK, lam=np.array([2.0, 3.0]), k=2, converged=2, solves=10, seconds=0.1, maxiter=100,
                max_seconds=None)
    bad = bk.case_summary({**base, "residual": np.array([3e-3, 1e-9])}, "c", 2)
    assert bad["status"] == bk.OK and bad["load_factors"] == [2.0, 3.0]
    assert "relative residual of the pairs is up to 3.0e-03" in bad["facts"][0] and bad["options"]
    good = bk.case_summary({**base, "residual": np.array([1.4e-8, 1e-9])}, "c", 2)    # the hook level
    assert good["facts"] == []


def test_ltb_cantilever_gives_the_recorded_factor_with_bending_and_shear():
    """Lateral-torsional buckling of a deep cantilever (t 2 mm, d 12 mm, L 120 mm, E 2000, nu 0.35, 2 voxels across
    the thickness): the stress has bending and shear (txz), so a wrong shear slot of the stress or of Kg changes
    the factor. The Euler column has almost no shear. The review measured 1.13877 (the P2 limit is 1.048: the
    voxels read high, section 4.1 of the plan). The load is the beam formula 4.013 sqrt(E Iz G J) / L^2."""
    E2, nu2, T, D, L = 2000.0, 0.35, 2.0, 12.0, 120.0
    G = E2 / (2 * (1 + nu2))
    Iz, J = D * T ** 3 / 12, D * T ** 3 / 3 * (1 - 0.63 * T / D + 0.052 * (T / D) ** 5)
    pcr = 4.013 * np.sqrt(E2 * Iz * G * J) / L ** 2
    nt = 2
    h = T / nt
    shape = (int(round(L / h)), nt, int(round(D / h)))
    nx, ny, nz = shape
    edof, active_index = edof_table(shape, np.ones(shape, bool))

    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    jj, kk = np.meshgrid(np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    fixed = (3 * node(0, jj, kk).ravel()[:, None] + np.arange(3)).ravel()       # the clamped face x = 0
    free = np.setdiff1d(np.unique(edof), fixed)
    n_all = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    m = bv.VoxelModel(edof=edof, free=free, Ee=np.full(edof.shape[0], E2), weight=np.ones(edof.shape[0]),
                      rho=np.ones(edof.shape[0]), h=h, E=E2, nu=nu2, n_dof_all=n_all)
    f = np.zeros(n_all)
    for j in range(ny):
        for k in range(nz):
            for dj in (0, 1):
                for dk in (0, 1):
                    f[3 * node(nx, j + dj, k + dk) + 2] -= pcr / (ny * nz) / 4        # the tip load, -z
    K = m.asm.K(m.Ee)
    fac = Lu(K)
    rec = bv.run_cases(m, fac, f[free][:, None], ["tip"], modes=2, K=K)[0]
    assert rec["status"] == "ok" and rec["load_factors"][0] == pytest.approx(1.13877, rel=2e-4)
    assert 1.05 < rec["load_factors"][0] < 1.2
    sig = bv.element_stress(np.bincount(free, weights=fac.solve(f[free]), minlength=n_all), edof, E2, nu2, h)
    assert np.abs(sig[:, 5]).max() > 0.1 * np.abs(sig[:, 0]).max()               # the shear txz is there
    # the mode is a sideways (y) and twisting motion, not a bend in the load plane
    phi = np.zeros(n_all)
    phi[free] = bk.buckling_modes(bv.hex8_kg(m.asm, edof, np.bincount(free, weights=fac.solve(f[free]),
                                                                      minlength=n_all), h, E2, nu2, m.weight),
                                  K, fac, k=1)["phi"][:, 0]
    assert np.abs(phi[1::3]).max() > np.abs(phi[2::3]).max()


def test_run_cases_with_the_matrix_of_the_caller_does_not_assemble_k_again():
    col = Column(3, 4)
    calls = []
    asm = col.model.asm
    original = asm.K
    asm.K = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    try:
        with_k = bv.run_cases(col.model, col.fac, col.f_red, ["c"], modes=2, K=col.K)[0]
        n_with = len(calls)
        calls.clear()
        without = bv.run_cases(col.model, col.fac, col.f_red, ["c"], modes=2)[0]
    finally:
        asm.K = original
    assert n_with == 6 and len(calls) == 7                  # six calls for Kg, plus one for K when it is not given
    assert with_k["load_factors"] == pytest.approx(without["load_factors"], rel=1e-12)


# ---------------------------------------------------------------- the run: status order, Ctrl+C, a kill

def test_the_run_is_not_done_before_the_check_ends(tmp_path, monkeypatch):
    """A poll of status.json and metrics.json during a run with the check on never sees `done` without the key."""
    import threading
    real = bv.run_check

    def slow(*a, **k):
        time.sleep(0.4)             # a window for the poll: the check of this small model is very short
        return real(*a, **k)

    monkeypatch.setattr(bv, "run_check", slow)
    out = tmp_path / "run"
    seen, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            try:
                st = json.loads((out / "status.json").read_text())
                metrics = json.loads((out / "result" / "metrics.json").read_text())
            except (OSError, ValueError):
                time.sleep(0.002)
                continue
            seen.append((st["state"], st.get("message"), "buckling" in metrics))
            time.sleep(0.002)

    t = threading.Thread(target=poll)
    t.start()
    try:
        code = main(["run", str(make_case(tmp_path, solver="amg")), "--out", str(out), "--quiet", "--buckling"])
    finally:
        stop.set()
        t.join()
    assert code == 0
    assert ("postprocess", "buckling check", False) in seen        # the poll saw the check running
    assert not [s for s in seen if s[0] == "done" and not s[2]]    # never `done` without the key
    assert json.loads((out / "status.json").read_text())["state"] == "done"
    assert "buckling" in json.loads((out / "result" / "metrics.json").read_text())


def test_ctrl_c_during_the_check_leaves_a_finished_run_and_an_error_record(tmp_path, monkeypatch):
    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(bv, "run_check", interrupted)
    code, out, metrics = run_cli(tmp_path, "--buckling")
    b = metrics["buckling"]
    assert code == 0 and list(metrics)[-1] == "buckling"
    assert b["status"] == "error" and b["reason"] == "interrupted" and b["cases"] == []
    assert "Ctrl+C" in b["message"] and any("topopt buckling <run_dir>" in o for o in b["options"])
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "done"                                  # the run is a finished run
    assert "warning: buckling check interrupted (Ctrl+C)" in (out / "run.log").read_text()
    assert (out / "result" / "final.stl").is_file()


# ---------------------------------------------------------------- topopt buckling <run_dir>

def test_buckling_command_repeats_the_check_and_changes_nothing_else(tmp_path, capsys):
    code, out, before = run_cli(tmp_path, solver="amg")             # a finished run, the check off
    assert "buckling" not in before
    status_before = (out / "status.json").read_text()
    assert main(["buckling", str(out), "--modes", "2", "--density", "gray"]) == 3     # amg: a fact, exit 3
    after = json.loads((out / "result" / "metrics.json").read_text())
    b = after["buckling"]
    assert list(after)[-1] == "buckling" and {k: v for k, v in after.items() if k != "buckling"} == before
    assert b["status"] == "skipped" and b["reason"] == "solver_not_direct" and b["density"] == "gray"
    assert (out / "status.json").read_text() == status_before     # a done run stays as it is
    log = (out / "run.log").read_text()
    assert "topopt buckling: repeat the check (modes 2, density gray)" in log and "buckling check skipped" in log
    assert "buckling check skipped" in capsys.readouterr().out


@needs_direct
def test_buckling_command_with_a_direct_solver_writes_the_factors(tmp_path):
    code, out, before = run_cli(tmp_path, "--buckling", extra={"modes": 3})
    assert before["buckling"]["modes"] == 3 and before["buckling"]["density"] == "printed"
    assert main(["buckling", str(out), "--modes", "2", "--density", "gray", "--max-seconds", "30"]) == 0
    b = json.loads((out / "result" / "metrics.json").read_text())["buckling"]
    assert b["status"] == "done" and b["density"] == "gray" and b["modes"] == 2 and b["max_seconds"] == 30.0
    assert all(len(c["load_factors"]) == 2 for c in b["cases"])


def test_buckling_command_repairs_the_status_of_a_killed_check(tmp_path):
    code, out, _ = run_cli(tmp_path, solver="amg")
    st = json.loads((out / "status.json").read_text())
    (out / "status.json").write_text(json.dumps({**st, "state": "postprocess", "step": "postprocess",
                                                 "message": "buckling check"}))    # what a kill leaves
    assert main(["buckling", str(out)]) == 3
    fixed = json.loads((out / "status.json").read_text())
    assert fixed["state"] == "done" and fixed["message"] == "" and fixed["iteration"] == st["iteration"]
    assert "it is done now" in (out / "run.log").read_text()


@pytest.mark.parametrize("state,message", [("running", ""), ("starting", ""), ("postprocess", "")])
def test_buckling_command_refuses_a_run_that_is_not_finished(tmp_path, capsys, state, message):
    code, out, before = run_cli(tmp_path, solver="amg")
    st = json.loads((out / "status.json").read_text())
    (out / "status.json").write_text(json.dumps({**st, "state": state, "message": message}))
    assert main(["buckling", str(out)]) == 1
    assert "is not finished" in capsys.readouterr().err
    assert json.loads((out / "result" / "metrics.json").read_text()) == before


def test_buckling_command_input_errors(tmp_path, capsys):
    assert main(["buckling", str(tmp_path)]) == 1
    assert "not a run folder" in capsys.readouterr().err
    code, out, _ = run_cli(tmp_path, solver="amg")
    (out / "result" / "metrics.json").unlink()
    assert main(["buckling", str(out)]) == 1
    assert "no result/metrics.json" in capsys.readouterr().err
    second = tmp_path / "second"
    second.mkdir()
    code, out2, _ = run_cli(second, solver="amg")
    for flags in (["--modes", "0"], ["--modes", "2.5"], ["--max-seconds", "-1"], ["--max-gb", "x"]):
        assert main(["buckling", str(out2), *flags]) == 1
        assert "must be" in capsys.readouterr().err


# ---------------------------------------------------------------- topopt check: a skip that the check can predict

def test_check_warns_before_the_run_when_the_check_will_be_skipped(tmp_path, capsys):
    from topopt.cli import check_data
    d = check_data(make_case(tmp_path, solver="amg", extra={"check": True}))
    assert d["buckling"]["will_run"] is False and d["buckling"]["reason"] == "solver_not_direct"
    w = [x for x in d["warnings"] if x.startswith("buckling check:")]
    assert len(w) == 1 and "it will be skipped" in w[0] and "cholesky" in w[0] and "GB" in w[0]
    assert d["buckling"]["memory_estimate_bytes"] > 0
    assert main(["check", str(tmp_path / "c.yaml")]) == 0
    assert "warning: buckling check: it is on in the YAML, but it will be skipped" in capsys.readouterr().out
    assert main(["check", str(tmp_path / "c.yaml"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["buckling"]["will_run"] is False


def test_check_gives_one_fact_line_and_no_warning_when_the_check_is_off(tmp_path, capsys):
    """Off (the default): no `buckling` key, no warning, no `buckling check:` line. One fact line says what the check is and what it costs (H1 PR 2)."""
    from topopt.cli import check_data
    d = check_data(make_case(tmp_path, solver="amg"))
    assert "buckling" not in d and not [x for x in d["warnings"] if "buckling" in x]
    assert d["buckling_off"]["would_run"] is False and d["buckling_off"]["reason"] == "solver_not_direct"
    assert main(["check", str(tmp_path / "c.yaml")]) == 0
    out = capsys.readouterr().out
    assert "buckling check:" not in out and "warning: buckling" not in out
    [line] = [l for l in out.splitlines() if "buckling" in l]
    assert line.startswith("buckling: off (check and row). A part with members in compression can buckle: buckling.check true gives the load factors")
    assert "it needs a direct solver" in line


@needs_direct
def test_check_gives_the_memory_estimate_when_the_check_will_run(tmp_path, capsys):
    from topopt.cli import check_data
    d = check_data(make_case(tmp_path, extra={"check": True}))
    assert d["buckling"]["will_run"] is True and not [x for x in d["warnings"] if x.startswith("buckling check:")]
    assert main(["check", str(tmp_path / "c.yaml")]) == 0
    assert "buckling check: on in the YAML (printed design, 6 modes): memory estimate about" in capsys.readouterr().out


def test_preflight_names_the_memory_and_voxel_guards(tmp_path):
    cfg = load_config(make_case(tmp_path, extra={"check": True, "max_gb": 0.001}))
    mem = bv.preflight(cfg, "pardiso", 1000, 500, 5000, ram_bytes=None)
    assert mem["will_run"] is False and mem["reason"] == "memory" and "buckling.max_gb = 0.001" in mem["facts"][0]
    cfg = load_config(make_case(tmp_path, extra={"check": True}))
    guard = bv.preflight(cfg, "pardiso", 1000, 10 ** 9, 5000, ram_bytes=None)
    assert guard["reason"] == "voxel_guard"
    none = bv.preflight(cfg, None, 1000, 500, 5000)
    assert none["reason"] == "solver_unavailable" and not none["will_run"]
    assert bv.preflight(cfg, "pardiso", 1000, 500, 5000, ram_bytes=64e9)["will_run"] is True


# ---------------------------------------------------------------- tension through the CLI

def test_cli_tension_column_gives_factors_far_above_the_compression_case(tmp_path):
    """A column in tension, through `topopt run --buckling`. The case has local compression at the clamp, so the
    status is `done` with positive factors that are in the thousands (Euler compression is 1). It is not an error and
    not `no_positive_factor`. The review measured 4107 for 4 across."""
    h, across = 5.0 / 3, 3
    length = 20 * across * h
    box(tmp_path / "plate.stl", (0, 0, -h), (5, 7.5, 0))
    box(tmp_path / "column.stl", (0, 0, 0), (5, 7.5, length))
    box(tmp_path / "top.stl", (0, 0, length), (5, 7.5, length + h))
    pcr = np.pi ** 2 * E * (7.5 * 5 ** 3 / 12.0) / (2.0 * (length + h / 2.0)) ** 2
    spec = {"name": "pull", "regions": {"design": "column.stl", "fixed": ["plate.stl"], "load": {"top": "top.stl"}},
            "load_cases": [{"name": "pull", "forces": [{"region": "top", "force": [0.0, 0.0, float(pcr)]}]}],
            "material": {"E": E, "nu": NU},
            "settings": {"voxel_size": h, "volume_fraction": 0.95, "iterations": 1},
            "buckling": {"check": True, "modes": 3}}
    (tmp_path / "pull.yaml").write_text(yaml.safe_dump(spec))
    out = tmp_path / "run"
    assert main(["run", str(tmp_path / "pull.yaml"), "--out", str(out), "--quiet"]) == 0
    b = json.loads((out / "result" / "metrics.json").read_text())["buckling"]
    if b["status"] == "skipped":
        pytest.skip("the solver of this machine is not direct")
    case = b["cases"][0]
    assert b["status"] == "done" and case["status"] == "ok" and case["load_factors"][0] > 1000
    assert b["lowest_load_factor"]["value"] == case["load_factors"][0]
