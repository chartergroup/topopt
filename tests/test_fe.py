"""Tests for topopt.fe and topopt.solver."""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from helpers import needs_accelerate, needs_pardiso
from topopt.fe import Assembler, edof_table, hex8_ke, rigid_body_modes, LOCAL_NODES
from topopt.solver import LinearSolver

# The solver backends. Each one runs only where the platform has it.
METHODS = [pytest.param("cholesky", marks=needs_accelerate),
           pytest.param("pardiso", marks=needs_pardiso),
           "amg"]


def make_solver(method, free, shape, h=1.0):
    """A LinearSolver of this backend. AMG gets the nullspace and a tight tolerance."""
    if method == "amg":
        return LinearSolver("amg", nullspace=rigid_body_modes(free, shape, h), rtol=1e-12)
    return LinearSolver(method)


def test_ke_symmetric_and_rigid_modes():
    ke = hex8_ke(1000.0, 0.3, 2.0)
    assert np.allclose(ke, ke.T)
    # 6 zero modes
    w = np.linalg.eigvalsh(ke)
    assert np.sum(np.abs(w) < 1e-8 * w.max()) == 6
    assert w.min() > -1e-8 * w.max()
    # rigid motion gives zero force
    xyz = 2.0 * LOCAL_NODES
    u = np.zeros(24)
    u[0::3] = 0.1 + 0.2 * xyz[:, 1] - 0.3 * xyz[:, 2]
    u[1::3] = -0.4 - 0.2 * xyz[:, 0] + 0.1 * xyz[:, 2]
    u[2::3] = 0.7 + 0.3 * xyz[:, 0] - 0.1 * xyz[:, 1]
    assert np.abs(ke @ u).max() < 1e-8 * np.abs(ke).max()


def test_ke_uniform_strain_energy():
    # u_x = eps * x on a cube: energy = 0.5 * (lam + 2 mu) eps^2 * h^3
    E, nu, h, eps = 1000.0, 0.3, 2.0, 1e-3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    ke = hex8_ke(E, nu, h)
    u = np.zeros(24)
    u[0::3] = eps * h * LOCAL_NODES[:, 0]
    assert u @ ke @ u * 0.5 == pytest.approx(0.5 * (lam + 2 * mu) * eps**2 * h**3, rel=1e-10)


def _beam(nx, ny, nz, h=1.0, E=1000.0, nu=0.3, mask=None):
    mask = np.ones((nx, ny, nz), bool) if mask is None else mask
    edof, idx = edof_table((nx, ny, nz), mask)
    nn = (nx + 1) * (ny + 1) * (nz + 1)
    used = np.zeros(3 * nn, bool)
    used[edof.ravel()] = True
    ii, jj, kk = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    node = (ii * (ny + 1) + jj) * (nz + 1) + kk
    fixed_nodes = node[0].ravel()
    fixed = np.zeros(3 * nn, bool)
    for c in range(3):
        fixed[3 * fixed_nodes + c] = True
    free = np.flatnonzero(used & ~fixed)
    return edof, idx, free, node, hex8_ke(E, nu, h)


def test_assembler_matches_scipy_coo():
    rng = np.random.default_rng(1)
    mask = rng.random((5, 4, 3)) < 0.7
    mask[0] = True
    edof, idx, free, node, ke = _beam(5, 4, 3, mask=mask)
    Ee = rng.random(edof.shape[0]) + 0.1
    asm = Assembler(edof, ke, free)
    K = asm.K(Ee)
    nd = 3 * (6 * 5 * 4)
    rows = np.repeat(edof, 24, axis=1).ravel()
    cols = np.tile(edof, (1, 24)).ravel()
    vals = (Ee[:, None] * ke.ravel()[None, :]).ravel()
    Kf = sp.coo_matrix((vals, (rows, cols)), shape=(nd, nd)).tocsc()[free][:, free]
    assert abs(K - Kf).max() < 1e-9 * abs(Kf).max()
    assert K.has_sorted_indices
    # second call reuses the pattern
    K2 = asm.K(2 * Ee)
    assert K2.indices is K.indices or np.shares_memory(K2.indices, K.indices)
    assert abs(K2 - 2 * Kf).max() < 1e-9 * abs(Kf).max()


@pytest.mark.parametrize("method", METHODS)
def test_cantilever_tip_deflection(method):
    # Slender beam L=40, 4x4 cross section, tip load. Beam theory + shear.
    nx, ny, nz, h, E, nu = 40, 4, 4, 1.0, 1000.0, 0.3
    edof, idx, free, node, ke = _beam(nx, ny, nz, h, E, nu)
    asm = Assembler(edof, ke, free)
    K = asm.K(np.ones(edof.shape[0]))
    P = 1.0
    f = np.zeros(3 * node.size)
    tip = node[nx].ravel()
    f[3 * tip + 2] = -P / tip.size
    fr = f[free]
    ls = make_solver(method, free, (nx, ny, nz), h)
    ls.factor(K)
    u = ls.solve(fr)
    w = np.mean(u[np.searchsorted(free, 3 * tip + 2)])
    L, b = nx * h, ny * h
    I = b**4 / 12
    beam = P * L**3 / (3 * E * I)
    shear = P * L / (5.0 / 6.0 * E / (2 * (1 + nu)) * b * b)
    assert abs(w) == pytest.approx(beam + shear, rel=0.06)
    # residual check
    assert np.linalg.norm(K @ u - fr) < 1e-8 * np.linalg.norm(fr)


@needs_accelerate
def test_cholesky_equals_amg():
    nx, ny, nz = 16, 6, 6
    edof, idx, free, node, ke = _beam(nx, ny, nz)
    rng = np.random.default_rng(2)
    Ee = np.where(rng.random(edof.shape[0]) < 0.4, 1.0, 1e-3)
    Ee[:ny * nz] = 1.0
    K = Assembler(edof, ke, free).K(Ee)
    B = rng.standard_normal((K.shape[0], 2))
    ns = rigid_body_modes(free, (nx, ny, nz))
    a = LinearSolver("cholesky")
    a.factor(K)
    xa = a.solve(B)
    b = LinearSolver("amg", nullspace=ns, rtol=1e-10)
    b.factor(K)
    xb = b.solve(B)
    assert np.linalg.norm(xa - xb) < 1e-5 * np.linalg.norm(xa)
    # a second factor reuses the cholesky pattern cache
    a.factor(Assembler(edof, ke, free).K(Ee * 2))
    assert np.allclose(a.solve(B), xa / 2, rtol=1e-6, atol=1e-9 * abs(xa).max())
    c = LinearSolver("auto", nullspace=ns)
    c.factor(K)
    assert c.method == "cholesky"


@needs_pardiso
def test_pardiso_equals_amg():
    nx, ny, nz = 16, 6, 6
    edof, idx, free, node, ke = _beam(nx, ny, nz)
    rng = np.random.default_rng(2)
    Ee = np.where(rng.random(edof.shape[0]) < 0.4, 1.0, 1e-3)
    Ee[:ny * nz] = 1.0
    asm = Assembler(edof, ke, free)
    K = asm.K(Ee)
    B = rng.standard_normal((K.shape[0], 2))
    ns = rigid_body_modes(free, (nx, ny, nz))
    a = LinearSolver("pardiso")
    a.factor(K)
    xa = a.solve(B)
    assert xa.shape == B.shape
    b = LinearSolver("amg", nullspace=ns, rtol=1e-10)
    b.factor(K)
    xb = b.solve(B)
    assert np.linalg.norm(xa - xb) < 1e-5 * np.linalg.norm(xa)
    assert np.linalg.norm(K @ xa - B) < 1e-9 * np.linalg.norm(B)
    # a second factor with the same pattern takes the numeric-only path: phase 22, no new analysis
    phases = []
    real_set_phase = a._ps.set_phase
    a._ps.set_phase = lambda ph: (phases.append(ph), real_set_phase(ph))[1]
    a.factor(asm.K(Ee * 2))
    assert phases == [22]
    a._ps.set_phase = real_set_phase
    assert np.allclose(a.solve(B), xa / 2, rtol=1e-6, atol=1e-9 * abs(xa).max())
    assert a.solve(B[:, 0]).shape == (K.shape[0],)
    # "auto" picks PARDISO here: Linux x86_64 with pypardiso
    c = LinearSolver("auto", nullspace=ns)
    c.factor(K)
    assert c.method == "pardiso"
    # free() releases PARDISO, and the solver can factor again
    a.free()
    a.factor(K)
    assert np.allclose(a.solve(B), xa, rtol=1e-8, atol=1e-12 * abs(xa).max())
    for s in (a, b, c):
        s.free()


@pytest.mark.parametrize("method", [pytest.param("cholesky", marks=needs_accelerate),
                                    pytest.param("pardiso", marks=needs_pardiso)])
def test_solver_pattern_change_and_refactor(method):
    ls = LinearSolver(method)
    for shape in [(6, 3, 3), (8, 4, 3), (6, 3, 3)]:
        edof, idx, free, node, ke = _beam(*shape)
        asm = Assembler(edof, ke, free)
        rng = np.random.default_rng(3)
        for _ in range(2):  # second call takes the refactor path
            K = asm.K(rng.random(edof.shape[0]) + 0.5)
            b = rng.standard_normal(K.shape[0])
            ls.factor(K)
            x = ls.solve(b)
            assert np.linalg.norm(K @ x - b) < 1e-8 * np.linalg.norm(b)
