"""Host side of the gpu-mg solver: the multigrid hierarchy on the CPU (numba, FP64).

Ported from the G1/G1b research code. This module has no OpenCL. It builds the level
tables once for a problem, and for each design it makes the numbers that the device
needs: the element stiffness of each level, the inverse 3x3 diagonal blocks, the largest
eigenvalue for the Chebyshev smoother, and the dense inverse of the coarsest level.

Layout. Each level keeps only the nodes of its active elements (compact numbering, in
ascending order of the box node number).

  enodes (nel, 8)  int32  element -> compact node ids, local order = topopt.fe.LOCAL_NODES
  nelem  (nn, 8)   int32  node -> element id for each local slot (-1: none), gather form
  Ee     (nel,)           stiffness factor of the element (E of SIMP)
  kel    (8, 8, 3, 3)     unit hex8 matrix of the level (scales with the element size)
  fix    (nn, 3)   uint8  1 = Dirichlet dof (eliminated)

A vector is (nn, 3) and keeps zeros on the Dirichlet dofs. The coarse levels have 2x
coarsening in each axis. A coarse element is active if one child is active. The stiffness
of a coarse element is the mean of its 8 children (an inactive child counts 0). A dof of a
coarse node is Dirichlet if the same dof of any fine node within one step of it is Dirichlet
(the "any" rule, for each dof). The flags are for each dof on every level (P2c): a per-axis
support (a roller or a slider) fixes only its axes, and a full clamp fixes all 3 dofs of a node.
The rule is per dof because the transfer acts on each dof alone: a fine dof is zero when the
same dof of every parent is zero, so the coarse levels keep every support of the fine level
and the six rigid-body modes stay blocked on every level (this says nothing about other coarse mechanisms). A rule for each node would fix all 3 dofs of a coarse node next to a roller.

Springs and joints (P2b, `border.py`). With a `GridBorder` the hierarchy holds a `Border`: the terms `Kb - C D^-1 C^T` of the springs and of the
bonds of joints on every level (Galerkin products of the transfer). They are in `apply_fine` and `apply_fine_abs`, in the power iteration,
in the diagonal 3 x 3 blocks of the smoother (`diag_blocks`: the arguments `bidx` and `eblk`) and in the dense coarsest matrix. The element
stiffness of the coarse levels stays rediscretized.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from numba import njit, prange

from ..fe import LOCAL_NODES, hex8_ke
from .border import Border, GridBorder

LN = LOCAL_NODES.astype(np.int64)

# The coarsening stops when a level has at most this many free dofs. Recorded default.
MIN_FREE_DOFS = 3000
MAX_LEVELS = 8


class HierarchyError(ValueError):
    """The grid cannot make a usable hierarchy. The text holds the fact."""


def kel_from_ke(ke, dtype=np.float64):
    """(24, 24) element matrix -> (8, 8, 3, 3) blocks [a, b, row, col]."""
    return np.ascontiguousarray(ke.reshape(8, 3, 8, 3).transpose(0, 2, 1, 3), dtype=dtype)


# --------------------------------------------------------------------------- numba kernels (FP64)

@njit(parallel=True, fastmath=True, cache=True)
def matvec(x, y, nelem, enodes, Ee, kel, fix):
    """y = A x. Dirichlet dofs of y are zero. The gather form has no write race."""
    nn = x.shape[0]
    for n in prange(nn):
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for a in range(8):
            e = nelem[n, a]
            if e < 0:
                continue
            t0 = 0.0
            t1 = 0.0
            t2 = 0.0
            for b in range(8):
                m = enodes[e, b]
                u0 = x[m, 0]
                u1 = x[m, 1]
                u2 = x[m, 2]
                t0 += kel[a, b, 0, 0] * u0 + kel[a, b, 0, 1] * u1 + kel[a, b, 0, 2] * u2
                t1 += kel[a, b, 1, 0] * u0 + kel[a, b, 1, 1] * u1 + kel[a, b, 1, 2] * u2
                t2 += kel[a, b, 2, 0] * u0 + kel[a, b, 2, 1] * u1 + kel[a, b, 2, 2] * u2
            E = Ee[e]
            s0 += E * t0
            s1 += E * t1
            s2 += E * t2
        y[n, 0] = 0.0 if fix[n, 0] else s0
        y[n, 1] = 0.0 if fix[n, 1] else s1
        y[n, 2] = 0.0 if fix[n, 2] else s2


@njit(parallel=True, fastmath=True, cache=True)
def diag_blocks(nelem, Ee, kel, fix, dinv, bidx, eblk):
    """dinv[n] = inverse of the 3x3 diagonal block of node n (6 numbers, symmetric).

    A Dirichlet dof has an identity row and column in the block. `bidx[n]` >= 0 is the place of node n in `eblk` (nb, 3, 3), the
    block of the springs and joints (`Border.diag_terms`) that is added to the element blocks before the inversion (P2b).
    """
    nn = nelem.shape[0]
    for n in prange(nn):
        d = np.zeros((3, 3), dtype=np.float64)
        for a in range(8):
            e = nelem[n, a]
            if e < 0:
                continue
            for r in range(3):
                for c in range(3):
                    d[r, c] += Ee[e] * kel[a, a, r, c]
        if bidx[n] >= 0:
            for r in range(3):
                for c in range(3):
                    d[r, c] += eblk[bidx[n], r, c]
        for r in range(3):
            if fix[n, r]:
                for c in range(3):
                    d[r, c] = 0.0
                    d[c, r] = 0.0
                d[r, r] = 1.0
        det = (d[0, 0] * (d[1, 1] * d[2, 2] - d[1, 2] * d[2, 1])
               - d[0, 1] * (d[1, 0] * d[2, 2] - d[1, 2] * d[2, 0])
               + d[0, 2] * (d[1, 0] * d[2, 1] - d[1, 1] * d[2, 0]))
        dinv[n, 0] = (d[1, 1] * d[2, 2] - d[1, 2] * d[2, 1]) / det
        dinv[n, 1] = (d[0, 2] * d[2, 1] - d[0, 1] * d[2, 2]) / det
        dinv[n, 2] = (d[0, 1] * d[1, 2] - d[0, 2] * d[1, 1]) / det
        dinv[n, 3] = (d[0, 0] * d[2, 2] - d[0, 2] * d[2, 0]) / det
        dinv[n, 4] = (d[0, 2] * d[1, 0] - d[0, 0] * d[1, 2]) / det
        dinv[n, 5] = (d[0, 0] * d[1, 1] - d[0, 1] * d[1, 0]) / det


@njit(parallel=True, fastmath=True, cache=True)
def apply_dinv(r, dinv, d):
    """d = Dinv r (block Jacobi)."""
    for n in prange(r.shape[0]):
        r0 = r[n, 0]
        r1 = r[n, 1]
        r2 = r[n, 2]
        d[n, 0] = dinv[n, 0] * r0 + dinv[n, 1] * r1 + dinv[n, 2] * r2
        d[n, 1] = dinv[n, 1] * r0 + dinv[n, 3] * r1 + dinv[n, 4] * r2
        d[n, 2] = dinv[n, 2] * r0 + dinv[n, 4] * r1 + dinv[n, 5] * r2


@njit(parallel=True, fastmath=True, cache=True)
def dot(a, b):
    s = 0.0
    for n in prange(a.shape[0]):
        s += a[n, 0] * b[n, 0] + a[n, 1] * b[n, 1] + a[n, 2] * b[n, 2]
    return s


# --------------------------------------------------------------------------- level tables

class Level:
    """The tables of one level. See the module text."""


def finish_topology(L):
    L.nelem = np.full((L.nn, 8), -1, np.int32)
    for a in range(8):
        L.nelem[L.enodes[:, a], a] = np.arange(L.nel, dtype=np.int32)
    nx, ny, nz = L.shape
    L.nodedims = (nx + 1, ny + 1, nz + 1)


def build_fine(shape, edof, active_index, fixed_dofs, h):
    """The fine level from the tables of the model (grid dof numbers)."""
    nx, ny, nz = shape
    L = Level()
    L.shape = (nx, ny, nz)
    L.h = h
    L.eidx = np.asarray(active_index, np.int64)
    nodes = edof[:, 0::3] // 3
    nidx = np.unique(nodes)
    box_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    L.lookup = np.full(box_nodes, -1, np.int32)
    L.lookup[nidx] = np.arange(nidx.size, dtype=np.int32)
    L.nidx = nidx.astype(np.int32)
    L.enodes = np.ascontiguousarray(L.lookup[nodes])
    L.nn, L.nel = nidx.size, nodes.shape[0]
    # One flag for each dof (P2c): a per-axis support fixes only its axes. A full clamp fixes all 3 dofs of a node, as before. A fixed dof on a node
    # that is not a node of the active elements has no place in the level and is skipped (as the node-level rule did).
    fd = np.asarray(fixed_dofs, dtype=np.int64)
    place = L.lookup[fd // 3]
    on = place >= 0
    L.fix = np.zeros((L.nn, 3), np.uint8)
    L.fix[place[on], fd[on] % 3] = 1
    finish_topology(L)
    return L


def coarsen(F):
    """The next coarser level (2x in each axis, the "any" Dirichlet rule for each dof)."""
    nx, ny, nz = F.shape
    cx, cy, cz = (nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2
    C = Level()
    C.shape = (cx, cy, cz)
    C.h = 2 * F.h
    e = F.eidx
    i = e // (ny * nz)
    j = (e // nz) % ny
    k = e % nz
    par = ((i // 2) * cy + j // 2) * cz + k // 2
    C.eidx, inv = np.unique(par, return_inverse=True)
    C.inv = inv.astype(np.int32)         # fine element position -> coarse element position
    C.nel = C.eidx.size
    I = C.eidx // (cy * cz)
    J = (C.eidx // cz) % cy
    K = C.eidx % cz
    gn = (((I[:, None] + LN[None, :, 0]) * (cy + 1) + (J[:, None] + LN[None, :, 1])) * (cz + 1)
          + (K[:, None] + LN[None, :, 2]))
    nidx = np.unique(gn)
    C.nidx = nidx.astype(np.int32)
    C.nn = nidx.size
    C.lookup = np.full((cx + 1) * (cy + 1) * (cz + 1), -1, np.int32)
    C.lookup[nidx] = np.arange(C.nn, dtype=np.int32)
    C.enodes = np.ascontiguousarray(C.lookup[gn])
    nzc1, nyc1 = cz + 1, cy + 1
    k = nidx % nzc1
    j = (nidx // nzc1) % nyc1
    i = nidx // (nzc1 * nyc1)
    C.fix = np.zeros((C.nn, 3), np.uint8)
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            for dc in (-1, 0, 1):
                fi, fj, fk = 2 * i + da, 2 * j + db, 2 * k + dc
                ok = (fi >= 0) & (fj >= 0) & (fk >= 0) & (fi <= nx) & (fj <= ny) & (fk <= nz)
                fn = np.where(ok, (np.clip(fi, 0, nx) * (ny + 1) + np.clip(fj, 0, ny)) * (nz + 1)
                              + np.clip(fk, 0, nz), 0)
                fc = np.where(ok, F.lookup[fn], -1)
                has = fc >= 0
                C.fix[has] |= F.fix[fc[has]]
    finish_topology(C)
    return C


def child_average(C, Ee):
    """Stiffness of the coarse elements: the mean of the 8 children."""
    return np.bincount(C.inv, weights=Ee, minlength=C.nel) / 8.0


def n_free(L) -> int:
    return int(3 * L.nn - int(L.fix.sum()))


# --------------------------------------------------------------------------- hierarchy

class HostMG:
    """The hierarchy for one problem. `setup_host(Ee)` makes the numbers for one design."""

    def __init__(self, shape, edof, active_index, fixed_dofs, h, nu,
                 min_free=MIN_FREE_DOFS, max_levels=MAX_LEVELS, border: GridBorder | None = None):
        ke = hex8_ke(1.0, nu, h)
        F = build_fine(shape, edof, active_index, fixed_dofs, h)
        levels = [F]
        # At least two levels: the coarsest level is solved with a dense inverse.
        while not ((n_free(F) <= min_free and len(levels) >= 2) or len(levels) >= max_levels):
            F = coarsen(F)
            levels.append(F)
        for l, L in enumerate(levels):
            L.kel = kel_from_ke(ke * 2.0 ** l)
        coarsest = levels[-1]
        if n_free(coarsest) == 0:
            raise HierarchyError(
                f"the coarsest level of the multigrid hierarchy has no free dof (grid {shape}, "
                f"{len(levels)} levels): the grid is too small or too thin for gpu-mg")
        self.levels = levels
        self.ke_unit = ke
        self.border = None if border is None else Border(border, levels)       # springs and joints on every level (P2b)
        for l, L in enumerate(levels):
            L.bidx = np.full(L.nn, -1, np.int32)
            L.eblk = np.zeros((0, 3, 3))
            if self.border is not None:
                nodes, blk = self.border.diag_terms(l)
                L.bidx[nodes] = np.arange(nodes.size, dtype=np.int32)
                L.eblk = np.ascontiguousarray(blk)
        self.Ainv_full = None
        self.cK = None
        self.cfree = None

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def setup_host(self, Ee, lmax_iters=12, reuse_power_vector=False, seed=0):
        """Make the numbers for the element stiffnesses `Ee` of the fine level.

        Return a dict of timers (s). The results are in `levels[l].Ee0`, `.dinv`, `.lmax` and in
        `Ainv_full` (the dense inverse of the coarsest level, zero on Dirichlet rows and columns).
        `reuse_power_vector`: start the power iteration from the vector of the last design.
        """
        T = {}
        lv = self.levels
        t0 = time.perf_counter()
        lv[0].Ee0 = np.ascontiguousarray(Ee, dtype=np.float64)
        for l in range(1, len(lv)):
            lv[l].Ee0 = child_average(lv[l], lv[l - 1].Ee0)
        for l, L in enumerate(lv):
            if L.nn and not hasattr(L, "r"):
                L.r = np.zeros((L.nn, 3))
                L.d = np.zeros((L.nn, 3))
        T["avg"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for L in lv:
            if not hasattr(L, "dinv"):
                L.dinv = np.zeros((L.nn, 6))
            diag_blocks(L.nelem, L.Ee0, L.kel, L.fix, L.dinv, L.bidx, L.eblk)
        T["dinv"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        for l, L in enumerate(lv[:-1]):
            if getattr(L, "vpow", None) is None or not reuse_power_vector:
                v = rng.standard_normal((L.nn, 3))
                v[L.fix.astype(bool)] = 0
            else:
                v = L.vpow
            lam = 1.0
            for _ in range(lmax_iters):
                matvec(v, L.r, L.nelem, L.enodes, L.Ee0, L.kel, L.fix)
                if self.border is not None:
                    self.border.add_to(l, v, L.r)
                apply_dinv(L.r, L.dinv, L.d)
                nrm = np.sqrt(dot(L.d, L.d))
                lam = nrm / np.sqrt(dot(v, v))
                v = L.d / nrm
            L.vpow = v
            L.lmax = 1.1 * lam
        T["eig"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._coarse_inverse(lv[-1])
        T["coarse"] = time.perf_counter() - t0
        T["total"] = T["avg"] + T["dinv"] + T["eig"] + T["coarse"]
        return T

    def _coarse_inverse(self, C):
        ke = C.kel.transpose(0, 2, 1, 3).reshape(24, 24)     # back to [3a + r, 3b + c]
        dofs = (3 * C.enodes[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)
        free = (C.fix.reshape(-1) == 0)
        fid = -np.ones(3 * C.nn, np.int64)
        fid[free] = np.arange(free.sum())
        rows = np.repeat(dofs, 24, axis=1).reshape(-1)
        cols = np.tile(dofs, (1, 24)).reshape(-1)
        vals = (C.Ee0[:, None] * ke.reshape(1, -1)).reshape(-1)
        r, c = fid[rows], fid[cols]
        m = (r >= 0) & (c >= 0)
        Kc = sp.coo_matrix((vals[m], (r[m], c[m])), shape=(int(free.sum()),) * 2).tocsc()
        Kd = Kc.toarray()
        if self.border is not None:
            self.border.add_dense(self.n_levels - 1, Kd, fid)       # Kb - C D^-1 C^T of the coarsest level (P2b)
        Ai = sla.inv(Kd, check_finite=False)
        Ai = 0.5 * (Ai + Ai.T)
        nd = 3 * C.nn
        idx = np.flatnonzero(free)
        full = np.zeros((nd, nd))
        full[np.ix_(idx, idx)] = Ai
        self.Ainv_full = full
        self.cK, self.cfree = Kc, free

    def apply_fine(self, x, y):
        """y = A x on the fine level in FP64 (Dirichlet dofs of y are zero)."""
        L = self.levels[0]
        matvec(x, y, L.nelem, L.enodes, L.Ee0, L.kel, L.fix)
        if self.border is not None:
            self.border.add_to(0, x, y)

    def apply_elements(self, x, y):
        """y = K x on the fine level in FP64: the element stiffness only, without the support terms of the springs and of the joints (P2d, "D3h").
        The harmonic columns of the deflation minimize this energy (`deflation.Deflation.extend`). Dirichlet dofs of y are zero."""
        L = self.levels[0]
        matvec(x, y, L.nelem, L.enodes, L.Ee0, L.kel, L.fix)

    def apply_fine_abs(self, x, y):
        """y = |K| |x| on the fine level in FP64 (G2). `|K|` is the sum of the absolute element matrices: `E |ke|` for each element.

        The backward error `eta = |r| / (| |K| |u| | + |b|)` of a solve uses it (CONTRACTS.md section 20). No matrix is built: it is the
        gather loop of `apply_fine` with `|kel|` and `|x|`. It is an upper bound of the entrywise `|K| |x|` (the entries of
        different elements at one node pair may cancel in K), so `eta` is at most a small factor too low, never high.
        """
        L = self.levels[0]
        if getattr(L, "kel_abs", None) is None:
            L.kel_abs = np.ascontiguousarray(np.abs(L.kel))
        matvec(np.ascontiguousarray(np.abs(x)), y, L.nelem, L.enodes, L.Ee0, L.kel_abs, L.fix)
        if self.border is not None:     # P2b: `|Kb| |x|` and `|C| |D^-1| |C^T| |x|` (an upper bound of the entrywise terms, like the element part)
            self.border.add_abs_to(0, x, y)

    def coarsest_dense_bytes(self, itemsize: int) -> int:
        """Bytes of the dense inverse of the coarsest level."""
        nd = 3 * self.levels[-1].nn
        return nd * nd * itemsize
