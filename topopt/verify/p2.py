"""Quadratic tetrahedra (P2): stiffness matrix and stress recovery. CONTRACTS.md section 16.

Ported from `pipeline/lib/p2_core.py` of an earlier pipeline of the same author (revision d43bcf3), without the code of that project
and the bordered system. The numbers are the same: the golden fixtures of
`tests/fixtures/verify_golden.npz.b64` compare them at rtol 1e-9. The geometric stiffness (`geometric_stiffness`,
for the linear buckling check, B3) is the formula of `p2_core.geometric_stiffness`; `tests/test_verify_buckling.py`
compares it with an independent scikit-fem assembly.

`P2Model` takes a linear tet mesh (4 nodes per tet) and lifts it to 10 nodes per tet itself.
The edge nodes sit at the middle of each edge (straight-sided elements). Layout of the pipeline:
`p` is (3, nv) and `t` is (4, nel). `P2Model.from_tetmesh` takes the B1 `TetMesh` with (n, 3) and (m, 4).

The numbering is the one of scikit-fem `ElementVector(ElementTetP2())`: the vertices first, then the
edges in the order of `mesh.edges`. The dof of node i and component c is `3 * i + c`.

Shape functions in barycentric coordinates l_i: vertex `N_i = l_i (2 l_i - 1)`, edge `N_ij = 4 l_i l_j`.
Local node order: 4 vertices, then the edges (0,1) (1,2) (0,2) (0,3) (1,3) (2,3).

Quadrature: order 2 (4 points) is exact for the stiffness. Order 5 (14 points, positive weights) is
used for the stress.

scikit-fem (the `verify` extra) gives the edge numbering and the quadrature rules. It is imported only
inside `P2Model.__init__` and `quadrature`, so `import topopt.verify.p2` needs no extra.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .errors import VerifyUnavailable

LOCAL_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
STRESS_ORDER = 5       # 14 points, positive weights
STIFF_ORDER = 2        # 4 points, exact for the P2 stiffness


def require_skfem():
    """Return the module `skfem`, or raise `VerifyUnavailable` with the install step."""
    try:
        import skfem
    except ImportError as exc:
        from .factor import verify_install_command
        raise VerifyUnavailable(
            "topopt verify needs the package scikit-fem (BSD-3-Clause) from the 'verify' extra. "
            f"Install the extra with: {verify_install_command()} ({exc})") from exc
    return skfem


def lame(E: float, nu: float):
    """Lame constants (lambda, mu) from Young's modulus and Poisson's ratio."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


# --------------------------------------------------------------------------- #
# Reference-element data
# --------------------------------------------------------------------------- #

def quadrature(order: int):
    """Barycentric points lam (nq,4) and weights w (nq,) that sum to 1."""
    require_skfem()
    from skfem.quadrature import get_quadrature_tet
    X, W = get_quadrature_tet(order)
    lam = np.column_stack([1.0 - X.sum(axis=0), X[0], X[1], X[2]])
    return lam, 6.0 * np.asarray(W)


def p2_grad_coeffs(lam: np.ndarray) -> np.ndarray:
    """C (nq,10,4) with grad N_a = sum_b C[q,a,b] grad l_b."""
    nq = lam.shape[0]
    C = np.zeros((nq, 10, 4))
    for i in range(4):
        C[:, i, i] = 4.0 * lam[:, i] - 1.0
    for k, (i, j) in enumerate(LOCAL_EDGES):
        C[:, 4 + k, i] = 4.0 * lam[:, j]
        C[:, 4 + k, j] = 4.0 * lam[:, i]
    return C


def tet_geometry(p: np.ndarray, t: np.ndarray, chunk: int = 200_000):
    """gl (nel,4,3): gradient of the barycentric coordinates; vol (nel,): volumes."""
    n = t.shape[1]
    gl = np.empty((n, 4, 3))
    vol = np.empty(n)
    for s in range(0, n, chunk):
        X = p[:, t[:, s:s + chunk]].transpose(2, 1, 0)           # (m,4,3)
        A = np.ones((X.shape[0], 4, 4))
        A[:, :, 1:] = X
        inv = np.linalg.inv(A)                                    # (m,4,4)
        gl[s:s + chunk] = inv[:, 1:, :].transpose(0, 2, 1)        # [e,a,d] = d l_a / d x_d
        vol[s:s + chunk] = np.abs(np.linalg.det(X[:, 1:, :] - X[:, :1, :])) / 6.0
    return gl, vol


# --------------------------------------------------------------------------- #
# Block sparsity pattern (node x node, 3x3 blocks)
# --------------------------------------------------------------------------- #

class BlockPattern:
    """Pattern of the node-block matrix and the map from element-pair entries to blocks."""

    def __init__(self, enodes: np.ndarray, nn: int):
        nel, nl = enodes.shape
        a_idx, b_idx = np.divmod(np.arange(nl * nl), nl)
        rows = enodes[:, a_idx].astype(np.int64)
        cols = enodes[:, b_idx].astype(np.int64)
        key = rows * nn + cols
        del rows, cols
        uniq, inv = np.unique(key.ravel(), return_inverse=True)
        del key
        self.perm = inv.reshape(nel, nl * nl).astype(np.int32)
        del inv
        br = uniq // nn
        self.indices = (uniq % nn).astype(np.int32)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(br, minlength=nn))]).astype(np.int64)
        self.nnzb = int(uniq.size)
        self.nn = nn
        self.nl = nl

    def accumulate9(self, data9: np.ndarray, blocks: np.ndarray, start: int):
        """data9 (nnzb,9) += element blocks (m, nl*nl, 9) of the elements start:start+m."""
        perm = self.perm[start:start + blocks.shape[0]].ravel()
        for c in range(9):
            data9[:, c] += np.bincount(perm, weights=blocks[:, :, c].ravel(), minlength=self.nnzb)

    def to_csr(self, data9: np.ndarray) -> sp.csr_matrix:
        M = sp.bsr_matrix((data9.reshape(-1, 3, 3), self.indices, self.indptr),
                          shape=(3 * self.nn, 3 * self.nn))
        return M.tocsr()

    def accumulate1(self, data: np.ndarray, blocks: np.ndarray, start: int):
        """data (nnzb,) += scalar element blocks (m, nl*nl) of the elements start:start+m."""
        perm = self.perm[start:start + blocks.shape[0]].ravel()
        data += np.bincount(perm, weights=blocks.ravel(), minlength=self.nnzb)

    def to_csr_kron3(self, data: np.ndarray) -> sp.csr_matrix:
        """The scalar node matrix (nn, nn) with the entries `data` (nnzb,), as the matrix `S (x) I3` (3 nn, 3 nn)."""
        S = sp.csr_matrix((data, self.indices, self.indptr), shape=(self.nn, self.nn))
        return sp.kron(S, sp.identity(3, format="csr"), format="csr")


# --------------------------------------------------------------------------- #
# Model: mesh + P2 numbering + element geometry
# --------------------------------------------------------------------------- #

class P2Model:
    """A linear tet mesh with the P2 numbering, and the P2 stiffness and stress for it.

    Attributes: `p` (3, nv), `t` (4, nel), `edges` (2, ne), `t2e` (6, nel), `nv`, `nel`, `ne`, `nn`
    (nodes: vertices and edges), `N` (dofs, `3 * nn`), `node_xyz` (3, nn), `enodes` (nel, 10),
    `gl` (nel, 4, 3), `vol` (nel,).
    """

    def __init__(self, p: np.ndarray, t: np.ndarray, chunk: int = 12000):
        """`p` (3, nv) float, `t` (4, nel) int. scikit-fem gives the edge numbering."""
        require_skfem()
        from skfem import MeshTet
        self.mesh = MeshTet(np.asarray(p, float), np.asarray(t, np.int64))
        mesh = self.mesh
        self.p, self.t = mesh.p, mesh.t
        self.nv, self.nel = mesh.p.shape[1], mesh.t.shape[1]
        self.edges, self.t2e = mesh.edges, mesh.t2e
        self.ne = self.edges.shape[1]
        self.nn = self.nv + self.ne
        self.N = 3 * self.nn
        self.node_xyz = np.hstack([self.p, 0.5 * (self.p[:, self.edges[0]] + self.p[:, self.edges[1]])])
        self.enodes = np.vstack([self.t, self.nv + self.t2e]).T.astype(np.int64)     # (nel,10)
        self.gl, self.vol = tet_geometry(self.p, self.t)
        self.chunk = chunk
        self._pattern = None
        self._edge_keys = None

    @classmethod
    def from_tetmesh(cls, mesh, chunk: int = 12000) -> "P2Model":
        """Build from a `topopt.tet.TetMesh` (nodes (n, 3), tets (m, 4))."""
        return cls(np.ascontiguousarray(np.asarray(mesh.nodes, float).T),
                   np.ascontiguousarray(np.asarray(mesh.tets, np.int64).T), chunk=chunk)

    @property
    def pattern(self) -> BlockPattern:
        if self._pattern is None:
            self._pattern = BlockPattern(self.enodes, self.nn)
        return self._pattern

    def centroids(self) -> np.ndarray:
        """Centroid of each tet, (nel, 3)."""
        return self.p[:, self.t].mean(axis=1).T

    def volume(self) -> float:
        return float(self.vol.sum())

    def chunks(self):
        for s in range(0, self.nel, self.chunk):
            yield slice(s, min(s + self.chunk, self.nel))

    def edge_node_ids(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Node index (vertices first, then edges) of the edge with the end vertices `a` and `b`.

        `a` and `b` are arrays of vertex indices. Raise `KeyError` if an edge is not in the mesh.
        """
        if self._edge_keys is None:
            e_lo = np.minimum(self.edges[0], self.edges[1]).astype(np.int64)
            e_hi = np.maximum(self.edges[0], self.edges[1]).astype(np.int64)
            keys = e_lo * self.nv + e_hi
            order = np.argsort(keys, kind="stable")
            self._edge_keys = (keys[order], order)
        sorted_keys, order = self._edge_keys
        want = np.minimum(a, b).astype(np.int64) * self.nv + np.maximum(a, b).astype(np.int64)
        pos = np.minimum(np.searchsorted(sorted_keys, want), len(sorted_keys) - 1)
        if not np.all(sorted_keys[pos] == want):
            raise KeyError("an edge of the facet list is not an edge of the mesh")
        return self.nv + order[pos]

    # ---------------- stiffness ---------------- #
    def stiffness(self, E: float, nu: float, verbose: bool = False) -> sp.csr_matrix:
        """The stiffness matrix (N, N) in CSR form, for an isotropic material."""
        lam_L, mu = lame(E, nu)
        lam, w = quadrature(STIFF_ORDER)
        C = p2_grad_coeffs(lam)                                         # (nq,10,4)
        pat = self.pattern
        data = np.zeros((pat.nnzb, 9))
        I3 = np.eye(3)
        t0 = time.time()
        for sl in self.chunks():
            gl = self.gl[sl]                                            # (m,4,3)
            G = np.einsum("qab,ebd->eqad", C, gl)                       # (m,nq,10,3)
            wv = w[None, :] * self.vol[sl][:, None]                     # (m,nq)
            outer = np.einsum("eq,eqai,eqbk->eabik", wv, G, G, optimize=True)     # sum_q w g_ai g_bk
            dots = np.einsum("eabii->eab", outer)
            blk = lam_L * outer + mu * outer.transpose(0, 1, 2, 4, 3) + mu * dots[..., None, None] * I3
            pat.accumulate9(data, blk.reshape(blk.shape[0], 100, 9), sl.start)
        K = pat.to_csr(data)
        if verbose:
            print(f"    P2 stiffness assembled: {K.shape[0]:,} dofs, nnz {K.nnz:,}, {time.time() - t0:.1f} s", flush=True)
        return K

    # ---------------- geometric stiffness ---------------- #
    def geometric_stiffness(self, sig: np.ndarray, order: int = STRESS_ORDER) -> sp.csr_matrix:
        """The geometric stiffness `Kg` (N, N) in CSR form, for the Cauchy stress `sig` (nel, nq, 6).

        `sig` is the stress at the points of the quadrature rule `order` (`stress_qp(u, E, nu, order)`; Voigt order xx, yy,
        zz, xy, yz, xz). `Kg = sum_q w_q vol grad(N_a)^T sigma grad(N_b)`, one scalar 10 x 10 block for each tet, placed
        on the three directions (`Kg = Ks (x) I3`). The sign is the one of the buckling problem `K phi = lambda (-Kg) phi`:
        a compressive stress gives a negative `Kg`. The element blocks are symmetrised, so `Kg` is exactly symmetric.
        """
        lam, w = quadrature(order)
        if sig.shape != (self.nel, lam.shape[0], 6):
            raise ValueError(f"sig has the shape {sig.shape}, but the stress at the {lam.shape[0]} points of order {order} "
                             f"of {self.nel} tets is {(self.nel, lam.shape[0], 6)}")
        C = p2_grad_coeffs(lam)                                         # (nq,10,4)
        pat = self.pattern
        data = np.zeros(pat.nnzb)
        for sl in self.chunks():
            G = np.einsum("qab,ebd->eqad", C, self.gl[sl])              # (m,nq,10,3)
            s = sig[sl]
            S = np.empty(s.shape[:2] + (3, 3))
            S[..., 0, 0], S[..., 1, 1], S[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
            S[..., 0, 1] = S[..., 1, 0] = s[..., 3]
            S[..., 1, 2] = S[..., 2, 1] = s[..., 4]
            S[..., 0, 2] = S[..., 2, 0] = s[..., 5]
            wv = w[None, :] * self.vol[sl][:, None]                     # (m,nq)
            GS = np.einsum("eqai,eqij->eqaj", G, S)                     # grad(N_a)^T sigma
            blk = np.einsum("eq,eqaj,eqbj->eab", wv, GS, G, optimize=True)
            blk = 0.5 * (blk + blk.transpose(0, 2, 1))
            pat.accumulate1(data, blk.reshape(blk.shape[0], 100), sl.start)
        return pat.to_csr_kron3(data)

    # ---------------- stress ---------------- #
    def _stress_at(self, u: np.ndarray, E: float, nu: float, lam: np.ndarray) -> np.ndarray:
        """Cauchy stress at the barycentric points `lam` (nq, 4) of every tet: (nel, nq, 6)."""
        lam_L, mu = lame(E, nu)
        C = p2_grad_coeffs(lam)
        U = np.asarray(u, float).reshape(-1, 3)
        sig = np.empty((self.nel, lam.shape[0], 6))
        for sl in self.chunks():
            ue = U[self.enodes[sl]]                                     # (m,10,3)
            G = np.einsum("qab,ebd->eqad", C, self.gl[sl])              # (m,nq,10,3)
            H = np.einsum("eai,eqaj->eqij", ue, G)                      # du_i/dx_j
            eps = 0.5 * (H + H.transpose(0, 1, 3, 2))
            tr = eps[..., 0, 0] + eps[..., 1, 1] + eps[..., 2, 2]
            s = 2.0 * mu * eps
            for k in range(3):
                s[..., k, k] += lam_L * tr
            sig[sl, :, 0] = s[..., 0, 0]; sig[sl, :, 1] = s[..., 1, 1]; sig[sl, :, 2] = s[..., 2, 2]
            sig[sl, :, 3] = s[..., 0, 1]; sig[sl, :, 4] = s[..., 1, 2]; sig[sl, :, 5] = s[..., 0, 2]
        return sig

    def stress_qp(self, u: np.ndarray, E: float, nu: float, order: int = STRESS_ORDER):
        """Cauchy stress at the quadrature points: `sig` (nel, nq, 6), Voigt order xx, yy, zz, xy, yz, xz.

        Also returns the barycentric points `lam` (nq, 4) and the weights `w` (nq,) (sum 1).
        """
        lam, w = quadrature(order)
        return self._stress_at(u, E, nu, lam), lam, w

    def strain_energy(self, u: np.ndarray, E: float, nu: float) -> np.ndarray:
        """The strain energy `1/2 u_e^T K_e u_e` of each tet for the displacement `u`: (nel,). The sum is `1/2 u^T K u`."""
        lam_L, mu = lame(E, nu)
        lam, w = quadrature(STIFF_ORDER)
        C = p2_grad_coeffs(lam)
        U = np.asarray(u, float).reshape(-1, 3)
        out = np.empty(self.nel)
        for sl in self.chunks():
            G = np.einsum("qab,ebd->eqad", C, self.gl[sl])              # (m,nq,10,3)
            H = np.einsum("eai,eqaj->eqij", U[self.enodes[sl]], G)      # du_i/dx_j
            eps = 0.5 * (H + H.transpose(0, 1, 3, 2))
            tr = eps[..., 0, 0] + eps[..., 1, 1] + eps[..., 2, 2]
            density = mu * np.einsum("eqij,eqij->eq", eps, eps) + 0.5 * lam_L * tr ** 2     # 1/2 sigma : eps
            out[sl] = (density * (w[None, :] * self.vol[sl][:, None])).sum(axis=1)
        return out

    def stress_at_vertices(self, u: np.ndarray, E: float, nu: float) -> np.ndarray:
        """Stress at the 4 vertices of each tet, (nel, 4, 6). The stress of a P2 tet is linear,
        so the largest von Mises value in the tet is at a vertex (von Mises is convex)."""
        return self._stress_at(u, E, nu, np.eye(4))

    def qp_points(self, lam: np.ndarray, elems: np.ndarray | slice | None = None) -> np.ndarray:
        """Physical coordinates of barycentric points: (m, nq, 3)."""
        X = self.p[:, self.t if elems is None else self.t[:, elems]].transpose(2, 1, 0)     # (nel,4,3)
        return np.einsum("qb,ebd->eqd", lam, X)
