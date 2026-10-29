"""The P2 library of `topopt.verify` (CONTRACTS.md section 16): the golden numbers of the pipeline, a patch test,
a slender cantilever against beam theory, and the factors on one system.

All tests here need scikit-fem (the `verify` extra). They need no gmsh.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_cholespy, needs_pardiso, needs_skfem  # noqa: E402
from verify_mesh import box_mesh  # noqa: E402

# Every test of this file compares the demo with a record of the old pipeline: pytest -m equivalence.
pytestmark = [needs_skfem, pytest.mark.equivalence]

# The fixture is an .npz file stored as base64 text, because the PR tool of this repo publishes text files only.
GOLDEN = Path(__file__).parent / "fixtures" / "verify_golden.npz.b64"
E, NU = 3500.0, 0.3


@pytest.fixture(scope="module")
def golden():
    assert GOLDEN.is_file(), f"golden fixture missing: {GOLDEN} (make it with tools/make_verify_golden.py)"
    with np.load(io.BytesIO(base64.b64decode(GOLDEN.read_text()))) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# Golden numbers of the pipeline code (p2_core, revision d43bcf3)
# --------------------------------------------------------------------------- #

def test_golden_fixture_names_its_source(golden):
    assert str(golden["source_rev"]) == "d43bcf3"
    assert str(golden["source_file"]) == "pipeline/lib/p2_core.py"
    assert len(str(golden["source_sha256"])) == 64


def test_golden_stiffness_matrix(golden):
    from topopt.verify.p2 import P2Model
    m = P2Model(golden["p"], golden["t"])
    assert m.N == int(golden["N"])
    K = m.stiffness(float(golden["E"]), float(golden["nu"])).tocsr()
    K.sort_indices()
    ref = sp.csr_matrix((golden["K_data"], golden["K_indices"], golden["K_indptr"]), shape=K.shape)
    assert K.nnz == ref.nnz
    np.testing.assert_allclose(K.data, ref.data, rtol=1e-9, atol=1e-9 * np.abs(ref.data).max())
    np.testing.assert_array_equal(K.indices, ref.indices)


def test_golden_numbering_and_geometry(golden):
    from topopt.verify.p2 import P2Model
    m = P2Model(golden["p"], golden["t"])
    np.testing.assert_array_equal(m.enodes, golden["enodes"])
    np.testing.assert_allclose(m.node_xyz, golden["node_xyz"], rtol=1e-12)
    np.testing.assert_allclose(m.vol, golden["vol"], rtol=1e-12)


def test_golden_stress_at_quadrature_points_and_vertices(golden):
    from topopt.verify.p2 import P2Model
    from topopt.verify.stats import von_mises
    m = P2Model(golden["p"], golden["t"])
    sig, lam, w = m.stress_qp(golden["u"], float(golden["E"]), float(golden["nu"]))
    scale = np.abs(golden["stress_qp"]).max()
    np.testing.assert_allclose(sig, golden["stress_qp"], rtol=1e-9, atol=1e-9 * scale)
    np.testing.assert_allclose(lam, golden["lam"], rtol=1e-12)
    np.testing.assert_allclose(w, golden["w"], rtol=1e-12)
    sv = m.stress_at_vertices(golden["u"], float(golden["E"]), float(golden["nu"]))
    np.testing.assert_allclose(sv, golden["stress_at_vertices"], rtol=1e-9,
                               atol=1e-9 * np.abs(golden["stress_at_vertices"]).max())
    np.testing.assert_allclose(von_mises(sig), golden["vm_qp"], rtol=1e-9, atol=1e-9 * golden["vm_qp"].max())


def test_golden_weighted_percentile(golden):
    from topopt.verify.p2 import P2Model
    from topopt.verify.stats import von_mises, weighted_percentile
    got = weighted_percentile(golden["wp_values"], golden["wp_weights"], golden["wp_pct"])
    np.testing.assert_allclose(got, golden["wp_result"], rtol=1e-9)
    m = P2Model(golden["p"], golden["t"])
    sig, _lam, w = m.stress_qp(golden["u"], float(golden["E"]), float(golden["nu"]))
    wv = w[None, :] * m.vol[:, None]
    got = weighted_percentile(von_mises(sig), wv, golden["wp_pct"])
    np.testing.assert_allclose(got, golden["wp_result_vm"], rtol=1e-9)


def test_weighted_percentile_is_a_step_function():
    from topopt.verify.stats import weighted_percentile
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert weighted_percentile(v, w, 50.0)[0] == 2.0           # first sample at or above 50 %
    assert weighted_percentile(v, w, 100.0)[0] == 4.0
    assert weighted_percentile(v, w, 0.0)[0] == 1.0
    assert weighted_percentile(v, [1, 1, 1, 97], 99.9)[0] == 4.0
    assert weighted_percentile(v, [97, 1, 1, 1], 50.0)[0] == 1.0   # the weight moves the percentile


# --------------------------------------------------------------------------- #
# Patch test
# --------------------------------------------------------------------------- #

def test_patch_test_linear_displacement_gives_exact_constant_stress():
    """An imposed linear displacement field gives the exact constant stress at every point, to 1e-8."""
    from topopt.verify.p2 import P2Model, lame
    from topopt.verify.stats import von_mises
    p, t = box_mesh((4.0, 3.0, 2.0), (4, 3, 2))
    p = p + 0.08 * np.random.default_rng(3).uniform(-1, 1, p.shape)            # no tet is special
    m = P2Model(p, t)
    G = np.array([[1.0, 0.3, -0.2], [0.1, -0.5, 0.4], [0.25, 0.05, 0.8]]) * 1e-3   # du_i / dx_j
    u0 = np.array([0.01, -0.02, 0.03])
    X = m.node_xyz
    u = (G @ X + u0[:, None]).T.reshape(-1)
    eps = 0.5 * (G + G.T)
    lam_L, mu = lame(E, NU)
    sig_exact = 2 * mu * eps + lam_L * np.trace(eps) * np.eye(3)
    voigt = np.array([sig_exact[0, 0], sig_exact[1, 1], sig_exact[2, 2], sig_exact[0, 1], sig_exact[1, 2], sig_exact[0, 2]])
    sig, _lam, _w = m.stress_qp(u, E, NU)
    sv = m.stress_at_vertices(u, E, NU)
    scale = np.abs(voigt).max()
    assert np.abs(sig - voigt).max() < 1e-8 * scale
    assert np.abs(sv - voigt).max() < 1e-8 * scale
    vm = von_mises(sig)
    assert np.ptp(vm) < 1e-8 * vm.max()
    # constant stress is in equilibrium: the nodal forces K u are zero at nodes inside the box
    K = m.stiffness(E, NU)
    f = K @ u
    nodes = X.T
    # the jitter moved the outer nodes, so test the nodes that are away from the outer layer
    interior = np.repeat(np.all((nodes > 0.45) & (nodes < np.array([4.0, 3.0, 2.0]) - 0.45), axis=1), 3)
    assert interior.sum() > 30
    assert np.abs(f[interior]).max() < 1e-8 * np.abs(f).max()


def test_quadratic_displacement_gives_exact_linear_stress(golden):
    """P2 elements hold a quadratic displacement field exactly: the stress is linear and exact at all points."""
    from topopt.verify.p2 import P2Model, lame
    p, t = golden["p"], golden["t"]
    m = P2Model(p, t)
    X = m.node_xyz
    rng = np.random.default_rng(5)
    C = rng.normal(size=(3, 10)) * 1e-3
    mono = np.stack([np.ones(X.shape[1]), X[0], X[1], X[2], X[0] * X[1], X[1] * X[2], X[0] * X[2],
                     X[0] ** 2, X[1] ** 2, X[2] ** 2])
    u = (C @ mono).T.reshape(-1)

    def stress_at(P):
        x, y, z = P
        dmono = np.array([[0, 1, 0, 0, y, 0, z, 2 * x, 0, 0],
                          [0, 0, 1, 0, x, z, 0, 0, 2 * y, 0],
                          [0, 0, 0, 1, 0, y, x, 0, 0, 2 * z]], float)
        grad = C @ dmono.T                              # du_i / dx_j
        eps = 0.5 * (grad + grad.T)
        lam_L, mu = lame(E, NU)
        s = 2 * mu * eps + lam_L * np.trace(eps) * np.eye(3)
        return np.array([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2]])

    sig, lam, _w = m.stress_qp(u, E, NU)
    pts = m.qp_points(lam)                       # (nel, nq, 3)
    exact = np.array([[stress_at(P) for P in row] for row in pts[::6]])
    assert np.abs(sig[::6] - exact).max() < 1e-8 * np.abs(exact).max()
    sv = m.stress_at_vertices(u, E, NU)
    vp = m.p.T[m.t.T]                            # (nel, 4, 3)
    exact_v = np.array([[stress_at(P) for P in row] for row in vp[::6]])
    assert np.abs(sv[::6] - exact_v).max() < 1e-8 * np.abs(exact_v).max()


def test_stiffness_is_symmetric_and_has_the_rigid_body_modes_in_its_null_space():
    from topopt.verify.p2 import P2Model
    p, t = box_mesh((3.0, 2.0, 2.0), (3, 2, 2))
    m = P2Model(p, t)
    K = m.stiffness(E, NU)
    assert abs(K - K.T).max() < 1e-9 * abs(K).max()
    X = m.node_xyz
    trans = np.zeros(m.N)
    trans[0::3] = 1.0
    rot = np.zeros(m.N)
    rot[0::3], rot[1::3] = -X[1], X[0]                      # rotation about z
    for v in (trans, rot):
        assert np.abs(K @ v).max() < 1e-9 * abs(K).max() * np.abs(v).max()
    assert m.volume() == pytest.approx(12.0)
    assert m.N == 3 * (m.nv + m.ne)


# --------------------------------------------------------------------------- #
# Slender cantilever against beam theory
# --------------------------------------------------------------------------- #

L_BEAM, H_BEAM, TIP_FORCE = 96.0, 6.0, 10.0       # L / h = 16


def cantilever(solver_name=None):
    """Solve the cantilever with the library: clamp at x = 0, a uniform traction on the end face (total -10 N in z)."""
    from topopt.verify import bc, regions
    from topopt.verify.factor import RefinedFactor, make_factor
    from topopt.verify.p2 import P2Model
    p, t = box_mesh((L_BEAM, H_BEAM, H_BEAM), (32, 2, 2))
    m = P2Model(p, t)
    K = m.stiffness(E, NU)
    fixed = bc.fixed_dofs(np.flatnonzero(m.node_xyz[0] < 1e-9))
    bf = regions.boundary_facets(m.t.T)
    end = bf[np.all(m.p[0][bf] > L_BEAM - 1e-9, axis=1)]
    f, info = bc.traction_load(m, end, [0.0, 0.0, -TIP_FORCE])
    free = np.setdiff1d(np.arange(m.N), fixed)
    Kff = K[free][:, free]
    u = np.zeros(m.N)
    if solver_name is None:
        from scipy.sparse.linalg import spsolve
        u[free] = spsolve(Kff.tocsc(), f[free])
    else:
        rf = RefinedFactor(Kff, make_factor(Kff, solver_name), steps=3)
        u[free] = rf.solve(f[free])
        rf.free()
    return m, u, info


def test_cantilever_tip_deflection_and_bending_stress_match_beam_theory():
    """The reference is Euler-Bernoulli beam theory (no shear).

    Timoshenko shear adds +0.30 % to the deflection at L/h = 16, and the Dirichlet clamp (all nodes of the end plane)
    makes the root stiffer. The -0.39 % match of this mesh is partly these two cancelling (review of B2: the
    h-refinement 16, 32, 48, 64 and 96 cells in x gives -1.2, -0.39, -0.25, -0.20, -0.16 %). The limit of 2 % is
    the acceptance criterion; it is not a claim of 0.39 % accuracy.
    """
    m, u, info = cantilever()
    I = H_BEAM ** 4 / 12.0
    # tip deflection: the mean of uz over the nodes of the end face, against F L^3 / (3 E I)
    tip = np.flatnonzero(m.node_xyz[0] > L_BEAM - 1e-9)
    deflection = u.reshape(-1, 3)[tip, 2].mean()
    beam = -TIP_FORCE * L_BEAM ** 3 / (3 * E * I)
    assert deflection == pytest.approx(beam, rel=0.02)
    # bending stress at the section x = L/2, on the top face z = h: sigma = M (h/2) / I
    sv = m.stress_at_vertices(u, E, NU)
    xs = m.p.T[m.t.T]
    x_sec = L_BEAM / 2
    on_top = (np.abs(xs[..., 0] - x_sec) < 1e-6) & (np.abs(xs[..., 2] - H_BEAM) < 1e-6)
    assert on_top.sum() >= 12
    theory = TIP_FORCE * (L_BEAM - x_sec) * (H_BEAM / 2) / I
    assert sv[..., 0][on_top].mean() == pytest.approx(theory, rel=0.05)
    # the bottom face is in compression by the same amount
    on_bottom = (np.abs(xs[..., 0] - x_sec) < 1e-6) & (np.abs(xs[..., 2]) < 1e-6)
    assert sv[..., 0][on_bottom].mean() == pytest.approx(-theory, rel=0.05)
    assert info["resultant_N"] == pytest.approx([0.0, 0.0, -TIP_FORCE], abs=1e-12)


def test_cantilever_equilibrium_residual_is_tiny():
    from topopt.verify import bc, regions
    m, u, _ = cantilever()
    K = m.stiffness(E, NU)
    fixed = bc.fixed_dofs(np.flatnonzero(m.node_xyz[0] < 1e-9))
    bf = regions.boundary_facets(m.t.T)
    end = bf[np.all(m.p[0][bf] > L_BEAM - 1e-9, axis=1)]
    f, _ = bc.traction_load(m, end, [0.0, 0.0, -TIP_FORCE])
    eq = bc.equilibrium(m, K, u, f, fixed)
    assert eq["force_relative_residual"] < 1e-9
    assert eq["moment_relative_residual"] < 1e-9
    # the support carries the load: the reaction is +10 N in z
    assert eq["reaction_force_N"][2] == pytest.approx(TIP_FORCE, rel=1e-9)


# --------------------------------------------------------------------------- #
# The factors give one solution (RefinedFactor)
# --------------------------------------------------------------------------- #

def _available_factors():
    from topopt.verify.factor import factor_availability
    return [n for n, v in factor_availability().items() if v["available"]]


def test_a_factor_is_always_available_because_superlu_needs_only_scipy():
    assert "superlu" in _available_factors()


def test_refined_solutions_are_equal_on_every_available_backend():
    names = _available_factors()
    assert "superlu" in names
    m, u_ref, _ = cantilever()                      # the reference: SciPy SuperLU on this small system
    sols = {}
    for name in names:
        _m, u, _ = cantilever(name)
        sols[name] = u
        assert np.abs(u - u_ref).max() < 1e-8 * np.abs(u_ref).max(), name
    for a in names:
        for b in names:
            assert np.abs(sols[a] - sols[b]).max() < 1e-8 * np.abs(sols[a]).max(), (a, b)


@needs_cholespy
def test_cholespy_factor_solves_one_vector_and_many_columns():
    from topopt.verify.factor import CholespyFactor
    R = sp.random(200, 200, density=0.03, random_state=2)
    A = (R @ R.T + sp.identity(200) * 200).tocsr()
    f = CholespyFactor(A)
    b = np.random.default_rng(1).normal(size=(200, 3))
    X = f.solve(b)
    assert X.shape == (200, 3)
    assert np.abs(A @ X - b).max() < 1e-9
    x = f.solve(b[:, 0])
    assert x.shape == (200,)
    np.testing.assert_allclose(x, X[:, 0], rtol=1e-10, atol=1e-12)
    f.free()


def test_superlu_factor_solves_one_vector_and_many_columns():
    from topopt.verify.factor import SuperluFactor
    R = sp.random(200, 200, density=0.03, random_state=2)
    A = (R @ R.T + sp.identity(200) * 200).tocsr()
    f = SuperluFactor(A)
    b = np.random.default_rng(1).normal(size=(200, 3))
    X = f.solve(b)
    assert X.shape == (200, 3)
    assert np.abs(A @ X - b).max() < 1e-9
    x = f.solve(b[:, 0])
    assert x.shape == (200,)
    np.testing.assert_allclose(x, X[:, 0], rtol=1e-10, atol=1e-12)
    f.free()


def test_superlu_factor_takes_the_full_matrix_as_csr_or_csc():
    """The factor scales and converts the matrix itself. It reads every entry, so it needs the full symmetric matrix."""
    from topopt.verify.factor import make_factor
    R = sp.random(120, 120, density=0.05, random_state=5)
    A = (R @ R.T + sp.identity(120) * 120)
    b = np.random.default_rng(3).normal(size=120)
    for M in (A.tocsr(), A.tocsc()):
        x = make_factor(M, "superlu").solve(b)
        assert np.abs(A @ x - b).max() < 1e-9


def test_superlu_factor_of_one_triangle_gives_a_wrong_solution_and_no_error():
    """A fact about the trap (the same as the lower triangle of cholespy, reversed): SuperLU reads every entry. A matrix with
    one triangle only is a different matrix; its test solve is made with that matrix, so it does not see the mistake."""
    from topopt.verify.factor import make_factor
    R = sp.random(120, 120, density=0.05, random_state=5)
    A = (R @ R.T + sp.identity(120) * 120).tocsr()
    b = np.random.default_rng(3).normal(size=120)
    x_full = make_factor(A, "superlu").solve(b)
    x_upper = make_factor(sp.triu(A).tocsr(), "superlu").solve(b)                   # no error
    assert np.abs(A @ x_full - b).max() < 1e-9
    assert np.linalg.norm(x_upper - x_full) / np.linalg.norm(x_full) > 1e-3        # measured 1e-2: the matrix was not the full one


@needs_cholespy
def test_cholespy_factor_of_an_upper_triangle_gives_a_wrong_solution_and_no_error():
    """A fact about the trap (E3b, review n3): cholespy reads the lower triangle only, so an upper-only matrix is a diagonal matrix for it.
    The test solve is made with the matrix that CHOLMOD factored, so it does not see the mistake (as for SuperLU, see above). Measured:
    a test residual of 1.4e-16 and a solution error of 9e-3. `run.py` always gives the full matrix; the residual that the report
    gives (`RefinedFactor` against the full matrix) is the one that shows a wrong triangle."""
    from topopt.verify.factor import CholespyFactor, RefinedFactor
    R = sp.random(120, 120, density=0.05, random_state=5)
    A = (R @ R.T + sp.identity(120) * 120).tocsr()
    b = np.random.default_rng(3).normal(size=120)
    x_full = CholespyFactor(A).solve(b)
    x_upper = CholespyFactor(sp.triu(A).tocsr()).solve(b)                          # no error
    assert np.abs(A @ x_full - b).max() < 1e-9
    assert np.linalg.norm(x_upper - x_full) / np.linalg.norm(x_full) > 1e-3        # measured 9e-3: the matrix was not the full one
    rf = RefinedFactor(A, CholespyFactor(sp.triu(A).tocsr()), steps=0)             # the residual against the full matrix shows it
    assert rf.residual_norms(b, rf.solve(b))[0] > 1e-3


def test_superlu_factor_keeps_double_precision_for_a_float32_matrix():
    from topopt.verify.factor import make_factor
    R = sp.random(120, 120, density=0.05, random_state=5)
    A = (R @ R.T + sp.identity(120) * 120).tocsr().astype(np.float32)
    b = np.random.default_rng(3).normal(size=120)
    x = make_factor(A, "superlu").solve(b)
    assert x.dtype == np.float64
    assert np.linalg.norm(A.astype(np.float64) @ x - b) / np.linalg.norm(b) < 1e-12      # single precision would give 1e-6


def test_superlu_runs_out_of_memory_as_a_verify_error_with_the_facts_and_the_options(monkeypatch):
    """SuperLU raises `MemoryError` when its fill is too large (seen at 431k dofs with the default order)."""
    import scipy.sparse.linalg as spla

    from topopt.verify import names
    from topopt.verify.errors import VerifyError
    from topopt.verify.factor import SuperluFactor

    def boom(*a, **k):
        raise MemoryError("splu")
    monkeypatch.setattr(spla, "splu", boom)
    A = sp.identity(50, format="csr") * 3.0
    with pytest.raises(VerifyError) as e:
        SuperluFactor(A)
    text = str(e.value)
    assert "ran out of memory" in text and "50 free dofs" in text and "--mesh-size" in text and "--solver" in text
    assert "pypardiso" in text and "cholespy" in text
    assert "32-bit indices" in text and "not measured" in text                         # more RAM may not help near 2^31 entries
    with names.use_style("mcp"), pytest.raises(VerifyError) as e2:
        SuperluFactor(A)
    assert "mesh_size_mm" in str(e2.value) and "--" not in str(e2.value)


def test_superlu_stops_on_a_singular_matrix_with_the_singular_system_error():
    """The Laplacian of a path (constants in the kernel): SuperLU itself raises "Factor is exactly singular" (a zero pivot)."""
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor
    n = 60
    L = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="lil")
    L[0, 0] = 1
    L[n - 1, n - 1] = 1                                  # the graph Laplacian of a path: singular (constants)
    with pytest.raises(SingularSystem) as e:
        make_factor(L.tocsr(), "superlu")
    assert "superlu" in str(e.value) and "Options" in str(e.value)


def _spring_to_ground_path_laplacian(n, eps):
    """A valid, very ill-conditioned matrix: the Laplacian of a path plus eps times the identity (a floating part on a soft spring)."""
    L = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="lil")
    L[0, 0] = 1
    L[n - 1, n - 1] = 1
    return (L + eps * sp.identity(n)).tocsr()


def test_superlu_factors_a_valid_ill_conditioned_matrix_and_still_stops_on_a_singular_one(monkeypatch):
    """The threshold of the test solve is 1e-3. A matrix with a soft spring (eps 1e-12) is valid and gives a test residual
    of about 1e-5, which the old threshold (1e-6) called singular; a singular matrix gives a residual above 1."""
    from topopt.verify import factor
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import RefinedFactor, make_factor
    assert factor.FACTOR_TEST_RESIDUAL == 1e-3
    A = _spring_to_ground_path_laplacian(400, 1e-12)
    with monkeypatch.context() as mp:                                         # the old threshold: this matrix was refused
        mp.setattr(factor, "FACTOR_TEST_RESIDUAL", 1e-6)
        with pytest.raises(SingularSystem) as old:
            make_factor(A, "superlu")
    assert "relative residual of" in str(old.value) and "threshold 1e-06" in str(old.value)      # the text has both numbers
    b = np.random.default_rng(1).normal(size=400)
    x = RefinedFactor(A, make_factor(A, "superlu"), steps=3).solve(b)             # the new one: it factors, the solve is good
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-3
    with pytest.raises(SingularSystem) as new:                                  # a floating part with no spring (zero pivot)
        make_factor(_spring_to_ground_path_laplacian(400, 0.0), "superlu")
    assert "superlu" in str(new.value)


@needs_cholespy
def test_cholespy_stops_on_a_singular_matrix_with_a_known_null_space():
    """The Laplacian of a path has the constants in its kernel, so it is singular whatever the roundoff. CHOLMOD may raise "not positive
    definite" or may factor it with a tiny pivot; in both cases the factor ends in `SingularSystem` (exit code 1, the facts and the
    options). With cholespy 2.2.0 and 400 nodes (measured on numpy 2.3.5 and 2.5.3) CHOLMOD is silent and the test solve stops it
    (a relative residual of 2.7); the test accepts both routes, so it does not depend on the roundoff."""
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor
    with pytest.raises(SingularSystem) as e:
        make_factor(_spring_to_ground_path_laplacian(400, 0.0), "cholespy")
    assert "cholespy" in str(e.value) and "Options" in str(e.value)
    assert e.value.facts["solver"] == "cholespy" and e.value.facts["reason"] == "singular_matrix"
    assert "not positive definite" in e.value.facts["detail"] or "test solve with a relative residual of" in e.value.facts["detail"]


@needs_cholespy
@pytest.mark.parametrize("wrong", [0.0, np.nan], ids=["zeros", "nan"])
def test_cholespy_test_solve_stops_a_factor_that_cholmod_does_not(monkeypatch, wrong):
    """A CHOLMOD that finds no problem and gives a wrong solution (numpy 2.3.5, the floating body of the tests): the test solve
    after the factor raises the error. A zero solution has a relative residual of 1; a NaN solution is refused too (`not r < limit`)."""
    import cholespy
    from topopt.verify import factor
    from topopt.verify.errors import SingularSystem

    class NoErrorWrongSolution:
        def __init__(self, n, rows, cols, data, kind):
            pass

        def solve(self, b, x):
            x[:] = wrong

    monkeypatch.setattr(cholespy, "CholeskySolverD", NoErrorWrongSolution)
    A = _spring_to_ground_path_laplacian(50, 1.0)                       # a valid matrix: only the fake solve is wrong
    with pytest.raises(SingularSystem) as e:
        factor.make_factor(A, "cholespy")
    text = str(e.value)
    assert "cholespy" in text and "Options" in text
    assert "cholespy gave a test solve with a relative residual of" in e.value.facts["detail"]
    assert f"threshold {factor.FACTOR_TEST_RESIDUAL:.0e}" in e.value.facts["detail"] and "--solver" in e.value.facts["detail"]


def _cholespy_reference_solve(A, b):
    """The solve of `CholespyFactor` as it was before E3b, written out: scale by the diagonal, factor the lower triangle, solve."""
    from cholespy import CholeskySolverD, MatrixType
    A = A.tocsr()
    n = A.shape[0]
    sc = 1.0 / np.sqrt(A.diagonal())
    C = sp.tril(sp.diags(sc) @ A @ sp.diags(sc), format="coo")
    f = CholeskySolverD(n, C.row.astype(np.int32), C.col.astype(np.int32), C.data.astype(np.float64), MatrixType.COO)
    B = np.ascontiguousarray(b.reshape(n, -1) * sc[:, None])
    X = np.zeros_like(B)
    f.solve(B, X)
    return X * sc[:, None]


@needs_cholespy
def test_cholespy_test_solve_does_not_change_the_solution_of_a_valid_matrix():
    """The same bits as the solve of the factor before the test solve was added; the lower triangle alone gives the same factor."""
    from topopt.verify.factor import CholespyFactor
    R = sp.random(200, 200, density=0.03, random_state=2)
    A = (R @ R.T + sp.identity(200) * 200).tocsr()
    b = np.random.default_rng(1).normal(size=(200, 3))
    ref = _cholespy_reference_solve(A, b)
    X = CholespyFactor(A).solve(b)
    assert np.array_equal(X, ref)
    assert np.array_equal(CholespyFactor(sp.tril(A).tocsr()).solve(b), ref)      # a lower-only matrix is not refused
    assert np.array_equal(CholespyFactor(A).solve(b[:, 0]), _cholespy_reference_solve(A, b[:, 0])[:, 0])      # the same shape: one column
    assert np.abs(A @ X - b).max() < 1e-9


@needs_cholespy
def test_cholespy_and_superlu_share_one_threshold_and_cholespy_factors_a_valid_ill_conditioned_matrix(monkeypatch):
    """A floating part on a soft spring (eps 1e-12) is valid; its test residual is about 1e-5 for both factors, below the shared 1e-3.
    There is one constant: with 1e-6 both factors refuse the matrix, and the text gives the value."""
    from topopt.verify import factor
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import RefinedFactor, make_factor
    n = 400
    A = _spring_to_ground_path_laplacian(n, 1e-12)
    b = np.random.default_rng(1).normal(size=n)
    x = RefinedFactor(A, make_factor(A, "cholespy"), steps=3).solve(b)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-3
    assert not hasattr(factor, "SUPERLU_TEST_RESIDUAL")                            # no second constant with the same value
    with monkeypatch.context() as mp:
        mp.setattr(factor, "FACTOR_TEST_RESIDUAL", 1e-6)
        for name in ("cholespy", "superlu"):
            with pytest.raises(SingularSystem) as e:
                make_factor(A, name)
            assert "relative residual of" in e.value.facts["detail"] and "threshold 1e-06" in e.value.facts["detail"]


@needs_pardiso
def test_linear_solver_factor_pardiso_matches_cholespy():
    from topopt.verify.factor import make_factor
    R = sp.random(300, 300, density=0.02, random_state=4)
    A = (R @ R.T + sp.identity(300) * 300).tocsr()
    b = np.random.default_rng(2).normal(size=(300, 2))
    x1 = make_factor(A, "pardiso").solve(b)
    assert np.abs(A @ x1 - b).max() < 1e-9


def test_the_residual_of_a_zero_load_column_is_not_defined_and_does_not_divide_by_zero():
    """E3b: `residual_norms` and `backward_error` give NaN for a column with b = 0 (no division, no RuntimeWarning); the other column is exact."""
    import warnings
    from topopt.verify.factor import RefinedFactor, make_factor
    R = sp.random(60, 60, density=0.1, random_state=3)
    A = (R @ R.T + sp.identity(60) * 60).tocsr()
    rf = RefinedFactor(A, make_factor(A, "superlu"), steps=3)
    b = np.column_stack([np.random.default_rng(5).normal(size=60), np.zeros(60)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x = rf.solve(b)
        r, e = rf.residual_norms(b, x), rf.backward_error(b, x)
    assert r[0] < 1e-13 and e[0] < 1e-13
    assert np.isnan(r[1]) and np.isnan(e[1]) and not np.any(x[:, 1])


def test_refinement_lowers_the_residual_of_a_poor_factor():
    """RefinedFactor works over any object with solve/free: a factor of a nearby matrix gets refined to the exact solution."""
    from topopt.verify.factor import RefinedFactor
    rng = np.random.default_rng(0)
    R = sp.random(150, 150, density=0.05, random_state=7)
    A = (R @ R.T + sp.identity(150) * 150).tocsr()

    class Rough:                                         # solves with a 1e-3 perturbed matrix
        def __init__(self):
            from scipy.sparse.linalg import splu
            self._lu = splu((A * 1.001).tocsc())

        def solve(self, b):
            return self._lu.solve(b)

        def free(self):
            pass

    b = rng.normal(size=150)
    plain = Rough().solve(b)
    refined = RefinedFactor(A, Rough(), steps=4).solve(b)
    r_plain = np.linalg.norm(A @ plain - b) / np.linalg.norm(b)
    r_ref = np.linalg.norm(A @ refined - b) / np.linalg.norm(b)
    assert r_plain > 1e-4
    assert r_ref < 1e-12


# --------------------------------------------------------------------------- #
# An independent reference (scikit-fem's own assembly), a jittered mesh, small chunks (review m6)
# --------------------------------------------------------------------------- #

def jittered_box(seed=1, jitter=0.1):
    """A 5 x 3 x 2 box of 1 mm cells with the nodes moved by up to `jitter`: tets of different size and shape."""
    p, t = box_mesh((5.0, 3.0, 2.0), (5, 3, 2))
    return p + jitter * np.random.default_rng(seed).uniform(-1.0, 1.0, p.shape), t


def test_stiffness_equals_the_scikit_fem_assembly_on_a_jittered_mesh():
    """The same P2 elasticity assembled by scikit-fem (its own basis, intorder 4): agreement to round-off."""
    from skfem import Basis, ElementTetP2, ElementVector, MeshTet, asm
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    from topopt.verify.p2 import P2Model
    p, t = jittered_box()
    m = P2Model(p, t)
    K = m.stiffness(E, NU).tocsr()
    ref = asm(linear_elasticity(*lame_parameters(E, NU)),
              Basis(MeshTet(np.asarray(p, float), np.asarray(t, np.int64)), ElementVector(ElementTetP2()), intorder=4)).tocsr()
    assert abs(K - ref).max() / abs(ref).max() < 1e-12


@pytest.mark.parametrize("chunk", [1, 7, 50])
def test_chunked_assembly_and_stress_equal_the_one_chunk_result(chunk):
    from topopt.verify.p2 import P2Model
    p, t = jittered_box(seed=2)
    one, many = P2Model(p, t, chunk=10 ** 6), P2Model(p, t, chunk=chunk)
    assert len(list(many.chunks())) > 3 and len(list(one.chunks())) == 1
    assert abs(one.stiffness(E, NU) - many.stiffness(E, NU)).max() / abs(one.stiffness(E, NU)).max() < 1e-12
    u = np.random.default_rng(3).normal(size=one.N)
    np.testing.assert_allclose(many.stress_qp(u, E, NU)[0], one.stress_qp(u, E, NU)[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(many.stress_at_vertices(u, E, NU), one.stress_at_vertices(u, E, NU), rtol=1e-12, atol=1e-12)


def test_golden_numbers_with_small_chunks(golden):
    """The golden mesh has one chunk by default. With chunk=7 the pipeline numbers are still the same."""
    from topopt.verify.p2 import P2Model
    m = P2Model(golden["p"], golden["t"], chunk=7)
    assert len(list(m.chunks())) > 5
    K = m.stiffness(float(golden["E"]), float(golden["nu"])).tocsr()
    K.sort_indices()
    ref = sp.csr_matrix((golden["K_data"], golden["K_indices"], golden["K_indptr"]), shape=K.shape)
    np.testing.assert_allclose(K.data, ref.data, rtol=1e-9, atol=1e-9 * np.abs(ref.data).max())
    sig = m.stress_qp(golden["u"], float(golden["E"]), float(golden["nu"]))[0]
    np.testing.assert_allclose(sig, golden["stress_qp"], rtol=1e-9, atol=1e-9 * np.abs(golden["stress_qp"]).max())
