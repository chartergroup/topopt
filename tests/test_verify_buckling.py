"""The linear buckling check of `topopt verify` (B3, CONTRACTS.md sections 16 and 33): the P2 geometric stiffness, the
Euler column, and the records of the check.

The library tests need scikit-fem (the `verify` extra). The tests of the gmsh column need gmsh (`needs_gmsh`).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_gmsh, needs_skfem  # noqa: E402
from verify_mesh import box_mesh  # noqa: E402

pytestmark = needs_skfem

E, NU = 3500.0, 0.3


# --------------------------------------------------------------------------- #
# The geometric stiffness against an independent assembly (scikit-fem)
# --------------------------------------------------------------------------- #

def jittered_box(seed=1, jitter=0.1):
    p, t = box_mesh((5.0, 3.0, 2.0), (5, 3, 2))
    return p + jitter * np.random.default_rng(seed).uniform(-1.0, 1.0, p.shape), t


def skfem_kg(model, sig, order):
    """`Kg` by the own assembly of scikit-fem: a scalar P2 basis, the stress as a field at the quadrature points, then x I3."""
    from skfem import Basis, BilinearForm, ElementTetP2, asm

    basis = Basis(model.mesh, ElementTetP2(), intorder=order)
    assert basis.quadrature[0].shape[1] == sig.shape[1]

    @BilinearForm
    def kg_form(u, v, w):
        s = w["sig"]                                           # (6, nel, nq)
        S = [[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]]
        out = 0
        for i in range(3):
            for j in range(3):
                out = out + u.grad[i] * S[i][j] * v.grad[j]
        return out

    Ks = asm(kg_form, basis, sig=np.transpose(sig, (2, 0, 1)))
    return sp.kron(Ks, sp.identity(3), format="csr")


def test_kg_equals_the_scikit_fem_assembly_on_a_jittered_mesh():
    """The class of the K test: agreement to round-off (1e-12 of the largest entry)."""
    from topopt.verify.p2 import STRESS_ORDER, P2Model
    p, t = jittered_box()
    m = P2Model(p, t)
    u = np.random.default_rng(3).standard_normal(m.N) * 1e-3
    sig, _lam, _w = m.stress_qp(u, E, NU, STRESS_ORDER)
    Kg = m.geometric_stiffness(sig)
    ref = skfem_kg(m, sig, STRESS_ORDER)
    assert Kg.shape == ref.shape == (m.N, m.N)
    assert abs(Kg - ref).max() / abs(ref).max() < 1e-12


def test_kg_is_exactly_symmetric_and_has_the_sign_of_the_stress():
    """A compressive stress gives a negative semi-definite `Kg`; a tensile stress gives a positive one."""
    from topopt.verify.p2 import STRESS_ORDER, P2Model, quadrature
    p, t = jittered_box(seed=2)
    m = P2Model(p, t)
    nq = quadrature(STRESS_ORDER)[0].shape[0]
    sig = np.zeros((m.nel, nq, 6))
    sig[..., 2] = -2.0                                         # sigma_zz = -2 MPa everywhere
    Kg = m.geometric_stiffness(sig)
    assert abs(Kg - Kg.T).max() == 0.0
    ev = np.linalg.eigvalsh(Kg.toarray())
    assert ev.max() < 1e-12 * abs(ev).max() and ev.min() < 0.0
    assert np.linalg.eigvalsh(m.geometric_stiffness(-sig).toarray()).min() > -1e-12 * abs(ev).max()


def test_kg_of_a_uniform_stress_is_the_stress_times_the_laplace_matrix_of_the_direction():
    """For `sigma = s e_z e_z^T`, `Kg = s * int d_z N_a d_z N_b dV (x) I3`: the form of the z part of the stiffness."""
    from topopt.verify.p2 import STRESS_ORDER, P2Model, quadrature
    p, t = jittered_box(seed=4)
    m = P2Model(p, t)
    nq = quadrature(STRESS_ORDER)[0].shape[0]
    sig = np.zeros((m.nel, nq, 6))
    sig[..., 2] = 1.0
    Kg = m.geometric_stiffness(sig)
    # a rigid translation has no strain, and a field linear in z has du/dz = c: u^T Kg u = s c^2 V (per direction)
    X = m.node_xyz
    for comp in range(3):
        u = np.zeros((m.nn, 3))
        u[:, comp] = 0.7 * X[2]
        uv = u.ravel()
        assert uv @ (Kg @ uv) == pytest.approx(0.7 ** 2 * m.volume(), rel=1e-12)
    rigid = np.tile([1.0, -2.0, 0.5], m.nn)
    assert abs(rigid @ (Kg @ rigid)) < 1e-12


@pytest.mark.parametrize("chunk", [1, 7, 50])
def test_kg_with_small_chunks_equals_the_one_chunk_result(chunk):
    from topopt.verify.p2 import STRESS_ORDER, P2Model
    p, t = jittered_box(seed=5)
    one, many = P2Model(p, t, chunk=10 ** 6), P2Model(p, t, chunk=chunk)
    u = np.random.default_rng(6).standard_normal(one.N)
    sig = one.stress_qp(u, E, NU, STRESS_ORDER)[0]
    assert abs(one.geometric_stiffness(sig) - many.geometric_stiffness(sig)).max() < 1e-12 * abs(one.geometric_stiffness(sig)).max()


def test_kg_refuses_a_stress_array_of_another_shape():
    from topopt.verify.p2 import P2Model
    p, t = jittered_box()
    m = P2Model(p, t)
    u = np.random.default_rng(3).standard_normal(m.N)
    sig4, _l, _w = m.stress_qp(u, E, NU, 4)             # the rule of order 4 has 11 points: not the 14 of order 5
    with pytest.raises(ValueError, match="shape"):
        m.geometric_stiffness(sig4)
    assert m.geometric_stiffness(sig4, order=4).shape == (m.N, m.N)


def test_strain_energy_of_the_tets_sums_to_half_u_k_u():
    from topopt.verify.p2 import P2Model
    p, t = jittered_box(seed=7)
    m = P2Model(p, t)
    u = np.random.default_rng(8).standard_normal(m.N)
    e = m.strain_energy(u, E, NU)
    assert e.shape == (m.nel,) and (e > 0).all()
    assert e.sum() == pytest.approx(0.5 * u @ (m.stiffness(E, NU) @ u), rel=1e-12)


# --------------------------------------------------------------------------- #
# The Euler column (fixed-free, P_cr = pi^2 E I / (2 L)^2)
# --------------------------------------------------------------------------- #

EULER_E, EULER_NU, EULER_L = 110000.0, 0.34, 100.0


def euler_factors(model, fixed_nodes, top_facets, pcr, k=4):
    """The lowest load factors of the column under the load `pcr` at the free end (a factor of 1 is the Euler load)."""
    from topopt import buckling as bk
    from topopt.verify import bc
    from topopt.verify.factor import make_factor
    from topopt.verify.p2 import STRESS_ORDER
    K = model.stiffness(EULER_E, EULER_NU)
    fixed = bc.fixed_dofs(fixed_nodes)
    free = np.setdiff1d(np.arange(model.N), fixed)
    f, _info = bc.traction_load(model, top_facets, [0.0, 0.0, -pcr])
    Kff = K[free][:, free].tocsr()
    fac = make_factor(Kff, "superlu")
    try:
        u = np.zeros(model.N)
        u[free] = fac.solve(f[free])
        sig = model.stress_qp(u, EULER_E, EULER_NU, STRESS_ORDER)[0]
        Kgff = model.geometric_stiffness(sig)[free][:, free].tocsr()
        res = bk.buckling_modes(Kgff, Kff, fac, k=k, tol=1e-10)
    finally:
        fac.free()
    return res


@pytest.mark.parametrize("across", [2, 3, 4])
def test_euler_column_on_a_tensor_mesh_is_within_0p3_percent(across):
    """The pipeline measured 1.00272, 1.00177 and 1.00142 at 2, 3 and 4 cells across (6 tets per cell, P2)."""
    from topopt.verify import regions
    from topopt.verify.p2 import P2Model
    B = 5.0
    p, t = box_mesh((B, B, EULER_L), (across, across, 20 * across))
    m = P2Model(p, t)
    pcr = np.pi ** 2 * EULER_E * (B ** 4 / 12.0) / (2.0 * EULER_L) ** 2
    bf = regions.boundary_facets(m.t.T)
    top = bf[np.all(m.p[2][bf] > EULER_L - 1e-9, axis=1)]
    res = euler_factors(m, np.flatnonzero(m.node_xyz[2] < 1e-9), top, pcr)
    assert res["status"] == "ok" and res["converged"] == 4
    assert res["lam"][0] == pytest.approx(1.0, abs=3e-3)
    assert res["lam"][0] > 1.0                                  # the tensor mesh reads high, as the pipeline record
    assert res["residual"].max() < 1e-7


def column_stl(path, section, length=EULER_L):
    import trimesh
    box = trimesh.creation.box(extents=(section[0], section[1], length))
    box.apply_translation(-box.bounds[0])
    box.export(path)
    return path


def gmsh_column(tmp_path, section, size):
    """Mesh a column STL with the mesher of `topopt verify`, load it with `-P_cr` on the top, return `(factors, model, I)`."""
    from topopt.tet import mesh_stl
    from topopt.verify import regions
    from topopt.verify.p2 import P2Model
    from topopt.verify.run import DEFAULT_OPTIMIZE
    stl = column_stl(tmp_path / f"col_{section[0]:g}x{section[1]:g}.stl", section)
    mesh = mesh_stl(stl, size=size, threads=1, optimize=DEFAULT_OPTIMIZE, log=None)
    m = P2Model.from_tetmesh(mesh)
    inertia = min(section[0] * section[1] ** 3, section[1] * section[0] ** 3) / 12.0          # the weak axis
    pcr = np.pi ** 2 * EULER_E * inertia / (2.0 * EULER_L) ** 2
    bf = regions.boundary_facets(m.t.T)
    top = bf[np.all(m.p.T[bf][:, :, 2] > EULER_L - 1e-6, axis=1)]
    return euler_factors(m, np.flatnonzero(m.node_xyz[2] < 1e-6), top, pcr), m


@needs_gmsh
@pytest.mark.parametrize("section, size", [((5.0, 5.0), 2.5), ((5.0, 5.0), 1.5), ((6.0, 4.0), 2.0)])
def test_euler_column_through_the_gmsh_path_is_within_1p5_percent_at_two_or_more_elements_across(tmp_path, section, size):
    """The free mesh is not monotone and can read below 1: the error was -1.37 to +0.14 % on 8 meshes (B3 research)."""
    res, _m = gmsh_column(tmp_path, section, size)
    assert min(section) / size >= 2.0
    assert res["status"] == "ok" and res["converged"] == 4
    assert abs(res["lam"][0] - 1.0) < 1.5e-2


@needs_gmsh
def test_a_square_column_sets_the_near_repeated_flag_and_a_rectangular_one_does_not(tmp_path):
    """The double mode of the square section splits by 2e-5 to 5e-3 on a free mesh: the 1 percent rule of the wrapper sees it."""
    from topopt import buckling as bk
    sq, _ = gmsh_column(tmp_path, (5.0, 5.0), 2.0)
    rec = bk.case_summary(sq, "square", 4)
    assert rec["near_repeated"] is True and rec["near_repeated_modes"][0] == [1, 2]
    assert rec["min_rel_gap"] < bk.NEAR_REPEATED_REL
    rect, _ = gmsh_column(tmp_path, (6.0, 4.0), 2.0)
    rec = bk.case_summary(rect, "rect", 4)
    assert rec["near_repeated"] is False and rec["min_rel_gap"] > 0.1


# --------------------------------------------------------------------------- #
# The texts of a case record: no voxel words for the tet model, the voxel texts unchanged
# --------------------------------------------------------------------------- #

def fake_result(status="ok", lam=(2.0, 3.0), k=2, residual=1e-9, max_seconds=120.0):
    return {"status": status, "lam": np.array(lam, float), "mu": 1.0 / np.array(lam, float), "k": k, "converged": len(lam),
            "solves": 7, "seconds": 1.5, "maxiter": 100, "max_seconds": max_seconds, "residual": np.full(len(lam), residual)}


CASES_OF_TEXT = [("no_positive", dict(lam=(np.inf, np.inf))), ("not_converged", dict(status="not_converged", lam=(5.0,))),
                 ("time_budget", dict(status="time_budget", lam=())), ("residual", dict(residual=3e-3))]
VOXEL_WORDS = ("voxel", "refine", "e_min", "density", "topopt buckling")


@pytest.mark.parametrize("label, kw", CASES_OF_TEXT)
def test_the_texts_of_a_tet_case_name_no_voxel(label, kw):
    from topopt import buckling as bk
    from topopt.verify import buckling_p2
    rec = bk.case_summary(fake_result(**kw), "c", 2, rerun=buckling_p2.rerun_text(), texts=buckling_p2.tet_texts())
    assert rec["facts"] and rec["options"]
    for t in rec["facts"] + rec["options"]:
        assert not any(w in t.lower() for w in VOXEL_WORDS), t
    assert "{" not in " ".join(rec["options"])


def test_the_time_budget_option_names_the_faster_factors_and_the_yaml_key():
    from topopt import buckling as bk
    from topopt.verify import buckling_p2
    rec = bk.case_summary(fake_result(status="time_budget", lam=()), "c", 2, rerun="topopt verify <run_dir>", texts=buckling_p2.tet_texts())
    text = " ".join(rec["options"])
    assert "buckling.max_seconds" in text and "cholesky" in text and "pardiso" in text and "cholespy" in text
    assert "--mesh-size" in text and "topopt verify <run_dir>" in text


def test_the_voxel_texts_of_a_case_summary_are_unchanged():
    """The texts of the voxel check (A6a) are the default of `case_summary`: B3 did not change one character."""
    from topopt import buckling as bk
    r = "topopt buckling <run_dir>"
    none = bk.case_summary(fake_result(lam=(np.inf, np.inf)), "c", 2)
    assert none["options"] == ["Look at the stress of this case in the refine page (topopt refine)."]
    nc = bk.case_summary(fake_result(status="not_converged", lam=(5.0,)), "c", 2)
    assert nc["options"] == [f"Ask for fewer modes (`{r} --modes N`), or look at the stress of this case (topopt refine)."]
    tb = bk.case_summary(fake_result(status="time_budget", lam=()), "c", 2)
    assert tb["options"] == [f"Run `{r} --max-seconds S` with a larger S, or use a larger voxel_size."]
    rs = bk.case_summary(fake_result(residual=3e-3), "c", 2)
    assert rs["facts"] == ["The eigen solve did not reach its tolerance: the relative residual of the pairs is up to 3.0e-03 (above 1e-06; "
                           "a normal case has 1e-8 to 1e-10), so the factors of this case are less exact than usual. K can be close to "
                           "singular: a part that is held only by the stiffness of void voxels (E_min) does this."]
    assert rs["options"] == ["Look for a loose piece or a thin neck in the part (topopt refine, result/final.stl), "
                             f"and compare with the other density (`{r} --density gray` or `printed`)."]


def test_energy_location_gives_the_centre_and_the_box_of_the_energy():
    from topopt import buckling as bk
    centers = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]])
    loc = bk.energy_location([1.0, 7.0, 1.0, 1.0], centers, share=0.6)
    assert loc["lowest_mode_energy_box_mm"] == [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    assert loc["lowest_mode_energy_center_mm"] == [12.0, 0.0, 0.0]
    assert bk.energy_location([0.0, 0.0], centers[:2]) == {}


# --------------------------------------------------------------------------- #
# The record of the check (a small column, no gmsh)
# --------------------------------------------------------------------------- #

class Column:
    """A 6 x 6 x 60 column of P2 tets, clamped at z = 0, with a compressive end load. Holds the pieces `run_check` needs."""

    def __init__(self, solver="superlu"):
        from topopt.verify import bc, regions
        from topopt.verify.factor import make_factor
        from topopt.verify.p2 import STRESS_ORDER, P2Model, quadrature
        p, t = box_mesh((6.0, 6.0, 60.0), (3, 3, 30))
        self.model = m = P2Model(p, t)
        self.E, self.nu = E, NU
        self.K = m.stiffness(E, NU)
        fixed = bc.fixed_dofs(np.flatnonzero(m.node_xyz[2] < 1e-9))
        self.free = np.setdiff1d(np.arange(m.N), fixed)
        bf = regions.boundary_facets(m.t.T)
        top = bf[np.all(m.p[2][bf] > 60.0 - 1e-9, axis=1)]
        f, _ = bc.traction_load(m, top, [0.0, 0.0, -10.0])
        self.Kff = self.K[self.free][:, self.free].tocsr()
        self.fac = make_factor(self.Kff, solver)
        u = np.zeros(m.N)
        u[self.free] = self.fac.solve(f[self.free])
        self.push = m.stress_qp(u, E, NU, STRESS_ORDER)[0]
        nq = quadrature(STRESS_ORDER)[0].shape[0]
        self.tension = np.zeros((m.nel, nq, 6))
        self.tension[..., :3] = 1.0                              # sigma = +1 MPa in all three directions: Kg is positive definite
        self.solver = solver

    def check(self, tmp_path, names=("push",), sigs=None, modes=4, max_seconds=120.0, metrics=None, cases=None):
        from types import SimpleNamespace

        from topopt.config import Buckling
        from topopt.verify import buckling_p2
        (tmp_path / "result").mkdir(exist_ok=True)
        if metrics is not None:
            (tmp_path / "result" / "metrics.json").write_text(__import__("json").dumps(metrics))
        cfg = SimpleNamespace(buckling=Buckling(check=True, modes=modes, max_seconds=max_seconds),
                              load_cases=[SimpleNamespace(name=n) for n in names])
        info = SimpleNamespace(run_dir=tmp_path)
        sigs = dict(sigs) if sigs is not None else {n: self.push.copy() for n in names}
        lines = []
        rec = buckling_p2.run_check(self.model, cfg, info, fac=self.fac, Kff=self.Kff, free=self.free, sigs=sigs,
                                    cases=cases or {}, solver=self.solver, E=self.E, nu=self.nu,
                                    mem={"term_bytes": 1}, say=lines.append)
        return rec, sigs, lines


@pytest.fixture(scope="module")
def column():
    c = Column()
    yield c
    c.fac.free()


def voxel_metrics(**lowest):
    return {"buckling": {"status": "done", "density": "printed", "solver": "pardiso",
                         "cases": [{"name": n, "status": "ok", "load_factors": [v, 2 * v],
                                    "lowest_mode_energy_center_mm": [1, 2, 3], "lowest_mode_energy_box_mm": [[0, 0, 0], [1, 1, 1]]}
                                   for n, v in lowest.items()]}}


def test_a_case_record_has_the_factors_the_voxel_value_and_the_ratio(column, tmp_path):
    rec, sigs, lines = column.check(tmp_path, ("push",), metrics=voxel_metrics(push=40.0),
                                    cases={"push": {"stiffness_over_optimizer": 0.8}})
    assert rec["status"] == "done" and rec["solver"] == "superlu" and rec["modes"] == 4
    assert rec["kind"].startswith("linear buckling of the P2 tet model") and rec["factor"].startswith("the plain factor")
    assert rec["eigen_solver"].startswith("arpack") and rec["tol"] == 1e-8 and rec["maxiter"] == 100 and rec["max_seconds"] == 120.0
    assert set(rec["not_used_keys"]) == {"buckling.density", "buckling.max_gb"} and rec["loose_piece"]["applicable"] is False
    assert rec["free_dofs"] == len(column.free) and set(rec["threads"]) == {"MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}
    (c,) = rec["cases"]
    # the Euler load of this column is E I pi^2 / (4 L^2) = 259.6 N; the end load is 10 N
    assert c["status"] == "ok" and 20.0 < c["load_factors"][0] < 30.0 and len(c["load_factors"]) == 4
    assert c["load_factors"] == sorted(c["load_factors"]) and c["modes_converged"] == 4 and c["residual_max"] < 1e-7
    assert c["solves"] > 10 and c["kg_seconds"] >= 0 and c["seconds"] > 0 and c["seconds_per_solve"] > 0
    assert c["voxel"]["lowest"] == 40.0 and c["voxel"]["density"] == "printed" and c["voxel"]["solver"] == "pardiso"
    assert c["voxel"]["mode_energy_center_mm"] == [1, 2, 3] and c["voxel"]["source"].startswith("result/metrics.json")
    assert c["tet_over_voxel"] == pytest.approx(c["load_factors"][0] / 40.0, rel=1e-12)
    assert c["stiffness_over_optimizer"] == 0.8
    assert len(c["lowest_mode_energy_center_mm"]) == 3 and len(c["lowest_mode_energy_box_mm"]) == 2
    # the lowest mode of a clamped column bends about a horizontal axis: the energy sits at the clamp, low in z
    assert 0.0 <= c["lowest_mode_energy_center_mm"][2] < 30.0
    assert rec["lowest_load_factor"] == {"case": "push", "value": c["load_factors"][0]}
    assert rec["facts"] == [] and rec["options"] == []
    assert sigs == {} and any("lowest load factor" in x and "tet over voxel" in x for x in lines)   # the stress array is freed


def test_without_a_voxel_value_the_record_says_why_and_names_the_command(column, tmp_path):
    for metrics, why in ((None, "cannot be read"), ({"solver": "pardiso"}, "no key buckling"),
                         ({"buckling": {"status": "skipped", "reason": "memory"}}, "'skipped' (memory)")):
        sub = tmp_path / why.split()[0].strip("'")
        sub.mkdir()
        rec, _, _ = column.check(sub, metrics=metrics)
        (c,) = rec["cases"]
        assert c["voxel"] is None and c["tet_over_voxel"] is None and c["load_factors"]
        assert any(why in f for f in rec["facts"]), rec["facts"]
        assert any("topopt buckling <run_dir>" in o for o in rec["options"])


def test_a_case_without_compression_is_a_fact_and_the_next_case_still_runs(column, tmp_path):
    rec, _, lines = column.check(tmp_path, ("tension", "push"), sigs={"tension": column.tension, "push": column.push})
    a, b = rec["cases"]
    assert a["name"] == "tension" and a["status"] == "no_positive_factor" and a["load_factors"] == []
    assert a["facts"] and a["options"] and "compression" in a["facts"][0]
    assert b["name"] == "push" and b["status"] == "ok" and b["load_factors"]
    assert rec["lowest_load_factor"]["case"] == "push" and rec["status"] == "done"
    assert any(f.startswith("tension: ") for f in rec["facts"]) and any(o.startswith("tension: ") for o in rec["options"])


def test_a_case_that_does_not_converge_is_a_fact_with_options(column, tmp_path, monkeypatch):
    from topopt import buckling as bk
    real = bk.buckling_modes
    monkeypatch.setattr(bk, "buckling_modes", lambda *a, **k: real(*a, **{**k, "maxiter": 1, "ncv": 8}))
    rec, _, _ = column.check(tmp_path, modes=6)
    (c,) = rec["cases"]
    assert c["status"] == "not_converged" and c["load_factors"] == [] and c["near_repeated"] is False
    assert any("did not converge" in f for f in c["facts"]) and any("buckling.modes" in o for o in c["options"])
    assert rec["status"] == "done" and rec["lowest_load_factor"] is None


def test_a_case_that_uses_the_time_budget_names_the_options(column, tmp_path):
    rec, _, _ = column.check(tmp_path, max_seconds=1e-9)
    (c,) = rec["cases"]
    assert c["status"] == "time_budget" and c["load_factors"] == [] and rec["max_seconds"] == 1e-9
    assert any("time budget" in f for f in c["facts"])
    text = " ".join(c["options"])
    assert "buckling.max_seconds" in text and "cholesky" in text and "pardiso" in text and "cholespy" in text


def test_a_failure_in_one_case_is_a_record_and_the_other_case_goes_on(column, tmp_path, monkeypatch):
    from topopt.verify.p2 import P2Model
    real = P2Model.geometric_stiffness
    calls = []

    def flaky(self, sig, order=5):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("no room for Kg")
        return real(self, sig, order)
    monkeypatch.setattr(P2Model, "geometric_stiffness", flaky)
    rec, _, _ = column.check(tmp_path, ("a", "b"))
    a, b = rec["cases"]
    assert a["status"] == "error" and "MemoryError: no room for Kg" in a["message"] and a["facts"] and a["options"]
    assert b["status"] == "ok" and b["load_factors"] and rec["status"] == "done"
    assert rec["lowest_load_factor"]["case"] == "b"


def test_voxel_values_read_the_key_buckling_of_metrics_json(tmp_path):
    from types import SimpleNamespace

    from topopt.verify.buckling_p2 import voxel_values
    info = SimpleNamespace(run_dir=tmp_path)
    assert voxel_values(info)[1] == "result/metrics.json cannot be read"
    (tmp_path / "result").mkdir()
    (tmp_path / "result" / "metrics.json").write_text(__import__("json").dumps(
        {"buckling": {"status": "done", "density": "gray", "solver": "cholesky", "cases": [
            {"name": "a", "status": "ok", "load_factors": [3.5, 4.0]},
            {"name": "b", "status": "no_positive_factor", "load_factors": []}]}}))
    vals, why = voxel_values(info)
    assert why is None and vals["a"]["lowest"] == 3.5 and vals["a"]["density"] == "gray" and vals["b"]["lowest"] is None
    assert vals["b"]["status"] == "no_positive_factor"


def test_the_memory_decision_skips_a_check_that_does_not_fit_and_force_lifts_it():
    from types import SimpleNamespace

    from topopt.config import Buckling
    from topopt.verify import buckling_p2
    cfg = SimpleNamespace(buckling=Buckling(check=True))
    lim = {"limit_bytes": 1_400_000_000, "limit_source": "--max-gb"}
    rec, mem = buckling_p2.memory_decision(cfg, "pardiso", 400_000_000, 1_300_000_000, lim, False, 1.5)
    assert rec["status"] == "skipped" and rec["reason"] == "memory" and rec["cases"] == [] and rec["lowest_load_factor"] is None
    assert "1.70 GB" in rec["facts"][0] and "1.40 GB" in rec["facts"][0] and "stress check of verify goes on" in rec["facts"][0]
    assert any("--max-gb" in o and "--force" in o for o in rec["options"])
    assert mem["estimate_with_check_bytes"] == 1_700_000_000 and rec["memory"] == mem
    assert buckling_p2.memory_decision(cfg, "pardiso", 400_000_000, 1_300_000_000, lim, True, 1.5)[0] is None
    assert buckling_p2.memory_decision(cfg, "pardiso", 50_000_000, 1_300_000_000, lim, False, 1.5)[0] is None
    assert buckling_p2.memory_decision(cfg, "pardiso", 1, 1, {"limit_bytes": None, "limit_source": None}, False, 1.5)[0] is None


def test_the_memory_term_follows_the_counts_and_is_larger_than_the_measured_growth_of_the_hook():
    from topopt.verify import memory
    # the hook, P2 at 1.496 mm: 139,014 dofs, 77,967 free, nnz(K) 10,058,742, nnz(Kff) 5,525,235, 25,612 tets; measured growth 72 to 234 MB
    exact = memory.buckling_bytes(139_014, 6, free_dofs=77_967, nnz_k=10_058_742, nnz_kff=5_525_235, tets=25_612)
    assert 240e6 < exact < 450e6
    before = memory.buckling_bytes(139_014, 6)               # before meshing: all dofs count as free
    assert exact < before < 500e6
    assert memory.buckling_bytes(139_014, 20) > before         # more modes: more Lanczos vectors (ncv = 2 k + 1 above 24)
    assert memory.buckling_bytes(278_028, 6) > 1.9 * before - 2 * memory.BUCKLING_TEMP_BYTES
    # the pieces of the formula, by hand
    got = memory.buckling_bytes(1000, 6, free_dofs=800, nnz_k=75_000, nnz_kff=60_000, tets=200)
    assert got == int(12 * (75_000 / 3 + 2 * 60_000 / 3) + 8 * 800 * (24 + 8) + 672 * 200 + 200e6)


def test_the_levels_block_gets_the_change_of_the_lowest_load_factor_and_of_the_rows():
    from topopt.verify.run import levels_block

    def level(size, hang, pull, row):
        vm = {"p99_9_by_volume_MPa": 1.0, "peak_qp_MPa": 2.0, "peak_vertex_MPa": 3.0}
        out = dict(size_mm=size, mesh=dict(dofs=1000, tets=200), cases={"hang": dict(von_mises=vm), "pull_off": dict(von_mises=vm)})
        if hang is not None:
            out["buckling"] = dict(status="done", cases=[dict(name="hang", load_factors=[hang, 2 * hang]),
                                                         dict(name="pull_off", load_factors=[pull] if pull else [])])
        if row is not None:
            out["motion"] = dict(status="done", rows=[dict(name="displacement:tip:pull_off", tet_value=row)])
        return out
    lv = levels_block([level(2.0, 100.0, 10.0, 0.04), level(1.0, 90.0, None, 0.03)])
    assert lv["rows"][0]["buckling_lowest_load_factor"] == {"hang": 100.0, "pull_off": 10.0}
    assert lv["rows"][1]["buckling_lowest_load_factor"] == {"hang": 90.0, "pull_off": None}
    assert lv["relative_change"][0]["buckling_lowest_load_factor"] == {"hang": pytest.approx(-0.1), "pull_off": None}
    assert lv["relative_change_first_to_last"]["motion_tet_value"] == {"displacement:tip:pull_off": pytest.approx(-0.25)}
    # a skipped check or no check: the block has none of the new keys
    plain = levels_block([level(2.0, None, None, None), level(1.0, None, None, None)])
    assert not any("buckling" in k or "motion" in k for k in plain["rows"][0]) and "motion_tet_value" not in plain["relative_change"][0]


def test_the_memory_term_counts_the_stress_of_every_load_case():
    from types import SimpleNamespace

    from topopt.config import Buckling
    from topopt.verify import buckling_p2, memory
    one = memory.buckling_bytes(139_014, 6, free_dofs=77_967, nnz_k=10_058_742, nnz_kff=5_525_235, tets=25_612)
    ten = memory.buckling_bytes(139_014, 6, free_dofs=77_967, nnz_k=10_058_742, nnz_kff=5_525_235, tets=25_612, cases=10)
    assert ten - one == 9 * memory.BUCKLING_STRESS_BYTES_PER_TET * 25_612          # the stress of every case is kept until the check starts
    assert memory.buckling_bytes(139_014, 6, cases=0) == memory.buckling_bytes(139_014, 6)         # at least one case
    # before meshing: the estimate of a level follows the number of cases
    from topopt.verify.run import level_estimate
    geom = {"volume_mm3": 21_363.0, "area_mm2": 12_000.0}
    lim = {"limit_bytes": None}
    est2 = level_estimate(geom, 1.5, lim, n_cases=2, buckling_modes=6)
    est6 = level_estimate(geom, 1.5, lim, n_cases=6, buckling_modes=6)
    tets = memory.BUCKLING_TETS_PER_DOF * est2["estimated_dofs"]
    assert est6["buckling_bytes"] - est2["buckling_bytes"] == pytest.approx(4 * memory.BUCKLING_STRESS_BYTES_PER_TET * tets, rel=1e-3)
    # after meshing: the term of the check uses the real counts and the cases of the YAML
    model = SimpleNamespace(N=1000, nel=200)
    K = sp.random(1000, 1000, density=0.05, format="csr", random_state=1)
    cfg2 = SimpleNamespace(buckling=Buckling(check=True), load_cases=[object()] * 2)
    cfg5 = SimpleNamespace(buckling=Buckling(check=True), load_cases=[object()] * 5)
    free = np.arange(800)
    t2, t5 = buckling_p2.term_bytes(cfg2, model, K, K, free), buckling_p2.term_bytes(cfg5, model, K, K, free)
    assert t5 - t2 == 3 * memory.BUCKLING_STRESS_BYTES_PER_TET * 200


def test_the_error_options_name_memory_and_mesh_size_not_a_direct_factor():
    from types import SimpleNamespace

    from topopt.config import Buckling
    from topopt.verify import buckling_p2
    rec = buckling_p2.error_record(SimpleNamespace(buckling=Buckling(check=True)), "pardiso", "RuntimeError: x")
    assert len(rec["options"]) == 1 and "direct factor" not in rec["options"][0]
    assert "--max-gb" in rec["options"][0] and "--force" in rec["options"][0] and "--mesh-size" in rec["options"][0]


def test_the_thread_record_has_the_accelerate_variable():
    from topopt.verify import buckling_p2
    assert "VECLIB_MAXIMUM_THREADS" in buckling_p2.THREAD_VARS and set(buckling_p2.threads_fact()) == set(buckling_p2.THREAD_VARS)
