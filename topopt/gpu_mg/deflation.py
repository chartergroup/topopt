"""Deflation of the rigid modes and of the bonded patches in the inner PCG of gpu-mg case C with joints (P2d "D3" and "D3h"). This module has no OpenCL.

The cause. A part that joints hold has a soft rigid motion: in it, `Kb x` and `C D^-1 C^T x` are each of the size of the bond stiffness, and their
difference (the operator) is of the size of the beam. A 32-bit device (case C) loses that difference when the beam is soft against the bond, so the
inner PCG of the refinement loop breaks down or stagnates. A part that is only pinned (one node, one edge) has soft rotations too. The 32-bit error
of the joint term is mostly a net force and a net moment on each bonded patch (the nodes that carry the coupling `C` of one joint), and the part
answers it in the soft motions where a sleeve moves against the rest (research P2d-2).

The method. These motions are removed from the inner PCG, and they are solved exactly on the host in FP64. `Z` has two kinds of column:

* The rigid columns (D3): 6 rigid modes for each body (3 translations, 3 rotations about the centroid of the body, scaled by its radius of gyration).
  A body is a connected component of the active elements of level 0 (an element joins its 8 nodes). A mode is zero on every Dirichlet dof (per dof: the
  `fix` flags of level 0), and the modes of one body are orthonormalized with a QR. A mode that the Dirichlet dofs make dependent (a body that is
  clamped on all its dofs has none) is dropped by the numerical rank of that body. These columns depend on the topology only: they are made once.
* The harmonic columns (D3h): for each bonded patch, its rigid motions (the same rank rule, cut at the Dirichlet dofs: 6 for a free patch) extended
  elastically into the part. The extension is the minimum of the element energy `u^T K u` with the motion prescribed on the patch, the other patches
  held at zero and the Dirichlet dofs at zero, and with no joint term: `K_ff x = -K_fs u_s`. A column is zero on every other patch, except at a node
  that two patches share (two joint regions that touch): `__init__` writes the motion of each patch on all its nodes, so at a shared node the columns of
  both patches are not zero, each holds the motion of its own patch there, and the node is no unknown of the extension (it is on a patch). `extend` solves
  it for each design with an FP64 PCG on the host; the preconditioner is the 32-bit V-cycle of the device, so a caller gives it as a function. The columns depend on the design. The extension
  of the previous design is the start of the next one (warm start). A patch with no free node (every bonded dof of its joint is Dirichlet, so the
  coupling `C` of that joint is cut) has no column: `per_patch` holds 0 for it, `Zt` has no row for it, and the joint adds nothing to `Z`.

For each design, `set_design` makes `Z` (the rigid columns, then an orthonormal basis of the harmonic columns that is orthogonal to them), `AZ = A Z`
(one `HostMG.apply_fine` for each column: the FP64 operator with the joint terms) and `E = Z^T A Z`. The harmonic columns of all patches of a body that no
Dirichlet dof holds add up to its rigid modes (the sum of the extensions of the same rigid motion is that rigid motion), so the raw columns are nearly
dependent on the rigid columns, and `E` of the raw columns has a large condition number (3.6e8 to 3.4e10 on the two-joint part (2 tube joints, no fixed region, 600,771 free dofs at 1.0 mm) at full size in the research; 2.8e6
at 2.0 mm here, tolerance 1e-4). The orthonormal basis keeps `E` a compression of `A`: its condition number is not above that of `A` on the span (9.6e4 at 2.0 mm,
tolerance 1e-4; 5.4e5 at 1e-3 and 6.3e4 at 1e-5). Both solved the two-joint part in 4 steps for each block at 1e-4. The basis is a Euclidean QR, not an A-orthonormal one.
* The device projects the output `z` of the V-cycle in each iteration: `z <- z - Z E^-1 (AZ)^T z`, so that `z` is A-orthogonal to `Z`
  (`engine.py`, kernels `defl_dot`, `defl_coef`, `defl_apply`).
* `engine.GpuMG.pcg` makes the coarse correction around the device loop in FP64: `a = E^-1 Z^T b`, the device solves `b - A Z a`, and the
  result is `x + Z a`.

The projection is A-orthogonal and exact for any `Z` that has full column rank, so a poor `Z` costs iterations and never gives a wrong answer:
the acceptance of a solve is still the FP64 true residual of the host. This is why an extension that stops above its tolerance is a fact and not an
error. The program of the device has `dk` columns (a compile-time number); a design with fewer independent columns pads `Z`, `AZ` and `E^-1` with zeros.
The measurements are in CONTRACTS.md section 20.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components

# The relative residual `|K_ff x + K_fs u_s| / |K_fs u_s|` at which the extension of a patch motion into the part is accepted (P2d-D3h). The one number of
# the harmonic columns. It is used in one place (`Deflation.extend`); a caller can give another tolerance. Any tolerance gives a valid deflation (a column
# that is not exactly harmonic is still a column of `Z`), so the tolerance changes the speed and never the answer. It is a method value, not a user requirement:
# a module constant, not a setting.
# Measured on pocl, one Linux x86_64 host (4 threads, shared with other jobs, so seconds are not exact; one run for each tolerance, and the step counts near the
# inner tolerance move with the thread count), `gpu_tol` 1e-6, revision 133e5a1, case C with the
# 18 columns on the two-joint part at 2.0 mm (97,257 free dofs, the first design, a uniform density, `GpuSolver` by a script):
#   tolerance   extension (cold)                   cond(E)   refinement steps in the four blocks             solve
#   1e-3        1,352 iterations, 7 chunks, 21 s   5.4e5     4, 4, 5, 4 (9 of 17 steps at the cap of 300)    45 s
#   1e-4        3,380 iterations, 14 chunks, 83 s  9.6e4     4, 4, 4, 4 (at most 46 inner iterations)        19 s
#   1e-5        7,687 iterations, 29 chunks, 132 s 6.3e4     4, 4, 4, 4 (at most 40 inner iterations)         8 s
# All three give the compliance of case B within 1.3e-8 (relative). So at 2.0 mm a looser tolerance costs less in the extension and more in the solve, and
# 1e-3 reaches the cap of a step in 9 steps (the research prototype solved this two-joint part at 1e-3 in 4 steps for each block: the difference is not explained).
# At 1.0 mm (600,771 free dofs) 1e-4 gave 5 steps in each block (672 to 953 inner iterations; 3 steps at the cap), after 2,385 extension iterations (187 s).
# The research (P2d-2, prototype on this host and on the Intel Arc) found at 1.0 mm 6 to 7 steps and 1,800 to 2,100 inner
# iterations for each block at 1e-3, 5 steps at 1e-4 and no more benefit at 1e-5; these were not repeated here. A warm start from the previous design took
# 0 iterations in the two-joint part runs (the design space is small there) and 438 against 871 in the unit test (host, Jacobi, density changed by up to 5 percent, 1e-6).
EXTENSION_TOL = 1e-4


class DeflationError(ValueError):
    """The host part of the deflation cannot continue (the matrix `E` is not positive definite or not finite, or the extension gave a value that is not
    finite). The text holds the fact."""


def body_labels(L):
    """`(number of bodies, labels)` of the compact nodes of level `L`: the connected components of the active elements (an element joins its 8 nodes)."""
    en = L.enodes
    rows, cols = np.repeat(en[:, 0], 7), en[:, 1:].reshape(-1)
    graph = sps.coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(L.nn, L.nn))
    nb, lab = connected_components(graph, directed=False)
    return int(nb), lab.astype(np.int64)


def node_positions(L):
    """(nn, 3) positions of the compact nodes of level `L` in units of its element size, from the box node numbers."""
    g = L.nidx.astype(np.int64)
    ny1, nz1 = L.nodedims[1], L.nodedims[2]
    return np.stack([g // (ny1 * nz1), (g // nz1) % ny1, g % nz1], axis=1).astype(np.float64) * L.h


def body_modes(pos, fixed):
    """The orthonormal rigid modes of one body (or of one bonded patch) as columns `(3 n, r)` with `r <= 6`.

    `pos` (n, 3): the nodes of the body, `fixed` (n, 3) bool: the Dirichlet dofs (per dof). The 6 modes are zero on the Dirichlet dofs. The QR of
    the 6 columns is followed by an SVD of its `R` (6 x 6) that drops the directions with no energy in the free dofs: the number of columns is
    the numerical rank of the cut modes (the rank rule of `numpy.linalg.matrix_rank`: it comes from the machine epsilon, it is no setting).
    The columns that remain span the cut modes, so they are zero on the Dirichlet dofs.
    """
    n = pos.shape[0]
    r = pos - pos.mean(axis=0)
    rad = np.sqrt((r ** 2).sum(axis=1).mean())
    M = np.zeros((n, 3, 6))
    for a in range(3):
        M[:, a, a] = 1.0 / np.sqrt(n)
        e = np.zeros(3)
        e[a] = 1.0
        M[:, :, 3 + a] = np.cross(e[None, :], r) / (rad * np.sqrt(n)) if rad > 0 else 0.0
    M[fixed] = 0.0                      # per dof: a node with one fixed axis keeps its other axes
    M = M.reshape(3 * n, 6)
    Q, R = np.linalg.qr(M)
    U, s, _ = np.linalg.svd(R)
    rank = int((s > s[0] * max(M.shape) * np.finfo(np.float64).eps).sum()) if s[0] > 0 else 0
    B = Q @ U[:, :rank]
    B[fixed.reshape(-1)] = 0.0          # exact zeros on the Dirichlet dofs (the rounding of Q is below 1e-16)
    return B


def patch_nodes(host):
    """For each joint (in the order of the border dofs), the compact nodes of level 0 that carry its coupling `C`: the bonded patch (sorted, int64).
    A node whose three dofs are all fixed has no entry in `C` (the Dirichlet entries are cut), so it is not in the patch. A joint whose bonded dofs are all
    fixed gives an empty array (the caller makes no column for it). The list is empty without joints."""
    b = host.border
    if b is None or not b.k:
        return []
    lb = b.levels[0]
    out = []
    for m in range(b.k // 6):
        blk = np.abs(lb.C[:, 6 * m:6 * m + 6]).reshape(lb.n_nodes, 3, 6).sum(axis=(1, 2))
        out.append(lb.nodes[blk > 0].astype(np.int64))
    return out


def orthonormal_basis(cols, against=None):
    """An orthonormal basis `(n, r)` of the span of the rows of `cols` (m, n), made orthogonal to the orthonormal rows of `against` first (twice: once is
    not enough for nearly dependent rows). The number of columns is the numerical rank (the rule of `body_modes`: the machine epsilon and the shape)."""
    A = np.array(cols, dtype=np.float64, copy=True)
    if against is not None and against.shape[0]:
        for _ in range(2):
            A -= (A @ against.T) @ against
    if A.shape[0] == 0:
        return np.zeros((A.shape[1], 0))
    Q, R = np.linalg.qr(A.T)
    U, s, _ = np.linalg.svd(R)
    rank = int((s > s[0] * max(A.shape) * np.finfo(np.float64).eps).sum()) if s[0] > 0 else 0
    B = Q @ U[:, :rank]
    if against is not None and against.shape[0] and rank:
        # a nearly dependent row of `A` is small after the projection, so its direction holds the rounding of the projection: project the unit vectors once more
        B -= against.T @ (against @ B)
        B, _ = np.linalg.qr(B)
    return B


class Deflation:
    """The rigid columns and the patch columns of a hierarchy, and the numbers of a design. See the module text.

    Attributes after the build: `nbodies`, `n_rigid`, `n_harm` (the columns of the patches that can move), `dk` (= both, the columns of the program),
    `per_body` (rigid columns of each body), `per_patch` (harmonic columns of each patch), `Zt` (dk, 3 nn) float64, one column in each row (the layout of
    the device buffer; the harmonic rows are zero until `set_design`), `t_topology` (seconds). After `extend` and `set_design`: `H` (n_harm, 3 nn) the
    harmonic columns as the extension made them, `k` (the columns of `Z` that are in use: the others are zero rows), `AZt` (dk, 3 nn), `E`, `Einv`
    (k x k), `Zt32`, `AZt32`, `Einv32` (float32 copies for the device, `Einv32` is dk x dk), `t_design` (seconds), `ext` (the facts of the last `extend`).
    """

    def __init__(self, host):
        t0 = time.perf_counter()
        L = host.levels[0]
        self.n3 = 3 * L.nn
        nb, lab = body_labels(L)
        self.nbodies = nb
        pos, fixed = node_positions(L), L.fix.astype(bool)
        order = np.argsort(lab, kind="stable")
        start = np.searchsorted(lab[order], np.arange(nb + 1))
        bases, self.per_body, self.nodes_per_body = [], [], []
        for b in range(nb):
            nodes = order[start[b]:start[b + 1]]
            B = body_modes(pos[nodes], fixed[nodes])
            bases.append((nodes, B))
            self.per_body.append(int(B.shape[1]))
            self.nodes_per_body.append(int(nodes.size))
        self.n_rigid = int(sum(self.per_body))
        # the bonded patches: the motions that the extension prescribes (zero on the Dirichlet dofs, as the rigid modes are)
        self.patches = patch_nodes(host)
        rows, self.per_patch = [], []
        on_patch = np.zeros(self.n3, bool)
        for nodes in self.patches:
            if nodes.size == 0:                  # every bonded dof of this joint is Dirichlet (its C block is cut): nothing of it moves, so it has no column
                self.per_patch.append(0)
                continue
            B = body_modes(pos[nodes], fixed[nodes])
            dofs = (3 * nodes[:, None] + np.arange(3)[None, :]).reshape(-1)
            R = np.zeros((B.shape[1], self.n3))
            R[:, dofs] = B.T
            rows.append(R)
            self.per_patch.append(int(B.shape[1]))
            on_patch[dofs] = True
        self.U = np.concatenate(rows) if rows else np.zeros((0, self.n3))      # (n_harm, 3 nn): the prescribed values, zero off the patches
        self.n_harm = int(self.U.shape[0])
        self.free = ~on_patch & ~fixed.reshape(-1)           # the unknowns of the extension: neither on a patch nor Dirichlet
        self.dk = self.n_rigid + self.n_harm
        self.Zt = np.zeros((self.dk, self.n3))
        col = 0
        for nodes, B in bases:
            dofs = (3 * nodes[:, None] + np.arange(3)[None, :]).reshape(-1)
            self.Zt[col:col + B.shape[1], dofs] = B.T
            col += B.shape[1]
        self.Zt32 = self.Zt.astype(np.float32)
        self.H = None                    # the harmonic columns of the last extension (the start of the next one)
        self.k = self.n_rigid            # the columns in use
        self.AZt = self.AZt32 = self.E = self.Einv = self.Einv32 = self._chol = None
        self.ext = None                  # the facts of the last `extend`
        self._fresh = False              # `extend` has made the harmonic columns that `set_design` has not used yet
        self.ext_total = {"designs": 0, "warm": 0, "iterations": 0, "seconds": 0.0, "above_tol": 0, "rel_max": 0.0}
        self.t_topology = time.perf_counter() - t0
        self.t_design = 0.0

    @property
    def ready(self) -> bool:
        """True after `set_design`: the numbers of a design are made."""
        return self._chol is not None

    @property
    def z_changes(self) -> bool:
        """True when `Z` depends on the design (there are harmonic columns): the device copy of `Z` is made again for each design."""
        return self.n_harm > 0

    # ------------------------------------------------------------------ the extension of the patch motions
    def _K(self, host, X):
        """`K X` of the rows `X` (m, 3 nn) in FP64: the element stiffness of the design, with zeros on the Dirichlet dofs (no joint term)."""
        out = np.empty_like(X)
        y = np.zeros((host.levels[0].nn, 3))
        for c in range(X.shape[0]):
            host.apply_elements(np.ascontiguousarray(X[c]).reshape(-1, 3), y)
            out[c] = y.reshape(-1)
        return out

    def _chunk(self, host, precond, r, x, act, thr, maxit):
        """One chunk of the flexible PCG for `K_ff x = rhs` from the residual `r` (m, 3 nn): at most `maxit` iterations, stops when every column of `act`
        has |r| below `thr`. `x` and `r` are updated in place. Return `(iterations, breakdown)`; a breakdown is a direction with `p^T K p` not positive."""
        free = self.free
        zold = p = rzold = None
        it = 0
        with np.errstate(all="ignore"):
            for it in range(1, maxit + 1):
                z = np.array(precond(r), dtype=np.float64)
                if not np.isfinite(z).all():
                    raise DeflationError("the preconditioner (the V-cycle of the device) gave a value that is not finite in the extension of the patch motions")
                z[:, ~free] = 0.0
                rz = np.einsum("ij,ij->i", r, z)
                if p is None:
                    p = z.copy()
                else:
                    beta = np.where(act, (rz - np.einsum("ij,ij->i", r, zold)) / rzold, 0.0)       # Polak-Ribiere: the preconditioner is a 32-bit V-cycle
                    p = z + beta[:, None] * p
                zold, rzold = z, rz
                Ap = self._K(host, p)
                Ap[:, ~free] = 0.0
                pAp = np.einsum("ij,ij->i", p, Ap)
                if np.any(act & ~((pAp > 0) & (rz > 0))):
                    return it, True
                alpha = np.where(act, rz / pAp, 0.0)
                x += alpha[:, None] * p
                r -= alpha[:, None] * Ap
                act &= np.linalg.norm(r, axis=1) > thr
                if not act.any():
                    break
        return it, False

    def extend(self, host, precond, block, maxit, stagnation_fall, tol=None):
        """Make the harmonic columns for the design that `host.setup_host` made last, and return the facts (a dict). Needs `set_design` after it.

        `precond(R)`: the preconditioner, a function of a residual block `R` (`block`, 3 nn) float64 that returns a block of the same shape (the V-cycle of
        the device; it does not have to be zero on the unknowns' complement: the caller of `precond` zeroes it). `block`: the rows of one call (the columns
        that one launch of the engine handles; a short last block repeats its last column). `maxit`: the iterations of one chunk (the existing
        `GpuParams.max_iter`). `stagnation_fall`: the existing `engine.STAGNATION_FALL`.

        The columns are solved in blocks. A block runs in chunks of at most `maxit` iterations with a restart (the true residual) between the chunks, as the
        refinement loop does. A block is done when every column has a relative residual (against the norm of the right-hand side) of at most `tol`
        (`EXTENSION_TOL` when `tol` is None), or when the largest residual of the columns that are not done fell by less than `stagnation_fall` over two
        chunks. A column that stops above `tol` is still a valid column of `Z`; the facts say so (`converged`, `stopped`, `rel_max`).
        The start is the extension of the previous design (`warm`) if there is one, else zero.

        There is no hard bound on the iterations of a block. The stops are the tolerance, the stagnation rule and a breakdown; there is no cap
        (no habitual cap while no run comes near this worst case). The worst case, as a fact: a residual that falls by exactly 10 percent over each two chunks passes the rule, and goes from 1 to
        `tol` 1e-4 in `ln(1e-4) / ln(0.9)` = 87 pairs, about 175 chunks, that is 52,500 iterations of one block (chunks of 300). At the measured 78.6 ms for an
        iteration (pocl, 4 threads, 1.0 mm, the two-joint part: 2,385 iterations in 187.4 s, one run) this is about 69 minutes for one block, and the extension has
        one block for each `nr` columns (6 blocks for the 12 columns of that two-joint part). This case is not measured: no run showed a fall that slow (every block of
        the two-joint part runs stopped by the tolerance, after 263 to 574 iterations for a block at 1.0 mm on pocl, one run, and 306 to 506 on the Arc).

        Facts: `columns`, `warm`, `tol`, `iterations` (PCG iterations over all blocks and chunks; each costs one V-cycle and one product with K),
        `iterations_per_block`, `chunks`, `rel_max` (the largest final relative residual), `converged` (all columns at `tol`), `stopped` (a list of the
        reason of each block: "tol", "stagnation" or "breakdown"), `seconds`.
        """
        t0 = time.perf_counter()
        tol = EXTENSION_TOL if tol is None else float(tol)
        nh = self.n_harm
        warm = bool(self.H is not None and nh and np.all(np.isfinite(self.H)))
        H = np.empty((nh, self.n3))
        its_blocks, stopped, chunks, rel_max = [], [], 0, 0.0
        free = self.free
        for st in range(0, nh, block):
            cols = list(range(st, min(st + block, nh)))
            cols += [cols[-1]] * (block - len(cols))            # a short block repeats its last column
            Ub = self.U[cols]
            rhs = -self._K(host, Ub)
            rhs[:, ~free] = 0.0
            bn = np.linalg.norm(rhs, axis=1)
            x = np.zeros_like(rhs)
            if warm:
                x = self.H[cols] - Ub                           # the free part of the previous extension
                x[:, ~free] = 0.0
            rel_hist, n_its, why = [], 0, "tol"
            while True:
                r = rhs - self._K(host, x)
                r[:, ~free] = 0.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.where(bn > 0, np.linalg.norm(r, axis=1) / bn, 0.0)
                if not np.all(np.isfinite(rel)):
                    raise DeflationError(f"the extension of the patch motions into the part gave a residual that is not finite (block of columns {cols[0]} "
                                         f"to {cols[-1]} of {nh})")
                rel_hist.append(rel)
                pending = rel > tol
                if not pending.any():
                    why = "tol"
                    break
                if why == "breakdown":                          # the chunk before broke down: this residual is the last fact
                    break
                k = len(rel_hist) - 1
                if k >= 2 and np.max(rel[pending]) > (1.0 - stagnation_fall) * np.max(rel_hist[k - 2][pending]):
                    why = "stagnation"
                    break
                n, broke = self._chunk(host, precond, r, x, pending.copy(), tol * bn, maxit)
                n_its += n
                chunks += 1
                if broke:
                    why = "breakdown"                           # measure the residual once more, then stop
            real = min(block, nh - st)                          # the rows that are columns (the others repeat the last one)
            H[st:st + real] = x[:real] + Ub[:real]
            rel_max = max(rel_max, float(rel_hist[-1][:real].max()))
            its_blocks.append(n_its)
            stopped.append(why)
        self.H = H
        self._fresh = True
        self.ext = {"columns": nh, "warm": warm, "tol": tol, "iterations": int(sum(its_blocks)), "iterations_per_block": its_blocks, "chunks": chunks,
                    "rel_max": rel_max, "converged": bool(rel_max <= tol), "stopped": stopped, "seconds": time.perf_counter() - t0}
        tot = self.ext_total
        tot["designs"] += 1
        tot["warm"] += int(warm)
        tot["iterations"] += self.ext["iterations"]
        tot["seconds"] += self.ext["seconds"]
        tot["above_tol"] += int(not self.ext["converged"])
        tot["rel_max"] = max(tot["rel_max"], rel_max)
        return self.ext

    # ------------------------------------------------------------------ the numbers of a design
    def set_design(self, host) -> float:
        """Make `Z`, `AZ`, `E` and `E^-1` for the design that `host.setup_host` made last (after `extend`, when there are harmonic columns). Return the
        seconds. Raise `DeflationError` when `E` is not symmetric positive definite and finite (the text says why)."""
        t0 = time.perf_counter()
        if self.n_harm and not self._fresh:
            raise RuntimeError("the deflation has harmonic columns that are not made for this design: call extend after setup_host")
        self._fresh = False
        self._chol = None
        L = host.levels[0]
        if self.n_harm:
            Q = orthonormal_basis(self.H, self.Zt[:self.n_rigid])
            self.k = self.n_rigid + Q.shape[1]
            self.Zt[self.n_rigid:] = 0.0
            self.Zt[self.n_rigid:self.k] = Q.T
        k = self.k
        if self.AZt is None:
            self.AZt = np.zeros_like(self.Zt)
        self.AZt[k:] = 0.0
        y = np.zeros((L.nn, 3))
        for j in range(k):
            host.apply_fine(self.Zt[j].reshape(-1, 3), y)
            self.AZt[j] = y.reshape(-1)
        Zk, AZk = self.Zt[:k], self.AZt[:k]
        E = Zk @ AZk.T
        E = 0.5 * (E + E.T)
        if not np.all(np.isfinite(E)):
            raise DeflationError(f"the matrix E = Z^T A Z of the {k} columns is not finite")
        try:
            chol = sla.cho_factor(E, lower=True, check_finite=False)
        except np.linalg.LinAlgError as e:
            raise DeflationError(f"the matrix E = Z^T A Z of the {k} columns is not positive definite in FP64 (the host operator of the "
                                 f"design does not resolve the softest motion): {e}") from e
        Einv = sla.cho_solve(chol, np.eye(k), check_finite=False)
        self.E, self.Einv = E, 0.5 * (Einv + Einv.T)
        self._chol = chol
        if self.AZt32 is None:
            self.AZt32 = np.zeros(self.Zt.shape, np.float32)
        self.Zt32[...] = self.Zt
        self.AZt32[...] = self.AZt
        self.Einv32 = np.zeros((self.dk, self.dk), np.float32)
        self.Einv32[:k, :k] = self.Einv
        self.t_design = time.perf_counter() - t0
        return self.t_design

    def coarse(self, Bf):
        """The FP64 coarse correction of the right-hand sides `Bf` (nr, 3 nn): `(a, B2)` with `a = E^-1 Z^T b` as (nr, k) and `B2 = b - A Z a`."""
        if not self.ready:
            raise RuntimeError("the deflation has no design: call set_design after setup_host")
        k = self.k
        a = sla.cho_solve(self._chol, self.Zt[:k] @ Bf.T, check_finite=False).T
        return a, Bf - a @ self.AZt[:k]

    def correction(self, a):
        """`Z a` for the coefficients `a` (nr, k), as (nr, 3 nn)."""
        return a @ self.Zt[:self.k]

    def release(self):
        """Drop the host arrays: `Z` and `AZ` (float64 and float32), the prescribed values, the harmonic columns of the last extension, `E` and its factors.
        `GpuSolver.free` calls it (the engine holds this object too, so dropping the solver's reference alone would keep the arrays). The object cannot make
        a design after this. The facts (`summary`, `host_bytes`, `ext`, `ext_total`) stay: they come from the shapes and the numbers of the run."""
        self.Zt = self.Zt32 = self.AZt = self.AZt32 = self.U = self.H = None
        self.E = self.Einv = self.Einv32 = self._chol = None
        self.free = None
        self._fresh = False

    def host_bytes(self) -> int:
        """Bytes of the host arrays of the deflation once a design is made (from the shapes): `Z` and `AZ` in float64 and float32 (12 bytes for each entry
        of each), the prescribed values and the harmonic columns of the last extension (8 bytes for each entry), and `E^-1`. The work space of the QR in
        `set_design` (about the harmonic columns twice again) comes on top, for the time of that call."""
        return int(self.dk * self.n3 * 24 + 2 * self.n_harm * self.n3 * 8 + self.dk * self.dk * 4)

    def summary(self) -> dict:
        """Facts for the run log and the tests."""
        return {"bodies": self.nbodies, "columns": self.dk, "columns_per_body": list(self.per_body), "rigid_columns": self.n_rigid,
                "harmonic_columns": self.n_harm, "patches": len(self.patches), "columns_per_patch": list(self.per_patch), "active_columns": self.k,
                "topology_s": self.t_topology, "design_s": self.t_design, "host_bytes": self.host_bytes()}
