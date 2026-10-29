"""Springs and joints in the gpu-mg multigrid (P2b): the support terms on every level, on the host. This module has no OpenCL.

A run with springs (CONTRACTS.md section 35) or joints (section 38) solves  A u = f  with

    A = K + Kb - C D^-1 C^T

`K` is the stiffness of the elements (gpu-mg applies it on the grid and does not assemble it). `Kb` is the sparse bond and spring matrix:
it does not depend on the density. `C` (3 dofs of a bonded node by `k = 6 m` border dofs) and `D` (`k` by `k`, block diagonal) are the
coupling and the self term of the `m` rigid bodies of the joints, so `- C D^-1 C^T` is the Schur complement of the border: the displacement `u` of
this system is the grid block of the bordered system `[[K + Kb, C], [C^T, D]]` of the other solvers. A run with springs only has `k = 0`.

The multigrid needs the terms on every level (measured in the research of P2): a border in the coarsest matrix only is worse than none.

* `Kb_l = P^T Kb_{l-1} P` and `C_l = P^T C_{l-1}` (Galerkin, once, on the host; `P` is the trilinear transfer of the device with the Dirichlet
  mask of the coarse level). The cancellation in `A x` for a rigid motion of the bonded body is exact only for Galerkin products: rigid motions are
  linear fields and the transfer keeps them. The stiffness `K_l` stays rediscretized, as without the terms.
* The diagonal 3 x 3 blocks of the smoother of every level get `Kb_l[n, n] - C_l[n] D^-1 C_l[n]^T` (`diag_terms`).
* The dense inverse of the coarsest level is made from `K_c + Kb_c - C_c D^-1 C_c^T` (`add_dense`).
* The FP64 residual (`HostMG.apply_fine`) and the absolute operator of the backward-error rule (`HostMG.apply_fine_abs`) use `add_to` and
  `add_abs_to`: `|Kb| |x|` and `|C| |D^-1| |C^T| |x|` (an upper bound of the entrywise value, as the element part is).
* The sum `C^T x` of these two (`ct_dot`) is a pairwise sum (F1-7). In a rigid motion of a body that only joints hold, `Kb x` and `C D^-1 C^T x`
  are each far larger than their difference, so the rounding of `C^T x` is multiplied by that ratio in the residual. A BLAS product `C^T x` sums
  the bonded dofs one after the other, and its rounding grows with their number; a pairwise sum keeps it near the unit roundoff. The sum is
  numpy's own, so it is the same on macOS and Linux (CONTRACTS.md section 20 has the measured numbers).

Only the nodes that carry a term are stored for each level (`LevelBorder.nodes`), and the host and the device work on those.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from ..springs import sum_duplicates
from .transfer import prolongation_matrix


@dataclass
class GridBorder:
    """The support terms of a problem on the dofs of the grid (3 for each node of the whole box), as `springs.py` and `joints.py` make them.

    `rows`, `cols`, `vals`: `Kb` of the springs and of the bonds of the joints, duplicates added. `c_rows`, `c_cols`, `c_vals`: `C` as triplets
    (a grid dof, a border dof `0 <= j < k`, a value). `D`: (k, k). The entries on fixed dofs are cut by `Border`, as the CPU path cuts them.
    `facts["beam_over_bond_min"]`: the smallest generalized eigenvalue of (beam stiffness `B`, bond stiffness `D_bond`) over the joints (None
    without joints). The operator `Kb - C D^-1 C^T` is a difference of two terms of the size of the bond in a rigid motion of the body, and the
    result is of the size of the beam. A 32-bit device (case C) rounds at 6e-8 of the bond term, so it can lose the soft part of a joint. This is a
    fact about 32-bit float, not a limit of the solver: where a run stops depends on the part (see `check_line` for the measured cases). Since P2d
    case C removes the rigid modes of each body from its inner PCG and solves them in 64-bit float on the host (`deflation.py`); the cancellation
    above is in those rigid motions. Since P2d-D3h it also removes, for each bonded patch (the nodes that carry the coupling `C` of a joint), 6 harmonic
    columns: the rigid motions of the patch extended elastically into the part. The error of the joint term is mostly a net force and moment on each
    patch, and a part answers it in the soft motions where a sleeve moves against the rest, which the rigid modes of the whole body do not cover.
    """

    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    c_rows: np.ndarray
    c_cols: np.ndarray
    c_vals: np.ndarray
    D: np.ndarray
    facts: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.D.shape[0])


FP32_ROUNDING = 2.0 ** -24      # 6.0e-8: the relative rounding of one 32-bit float operation (a fact about the float type, not a limit of a solve)


def beam_over_bond(joint_set) -> tuple | None:
    """`(region, ratio)` of the joint with the softest beam, or None without joints.

    The ratio is the smallest generalized eigenvalue of (beam stiffness `B`, bond stiffness `D_bond`): the beam over the bond on the softest mode of
    the joint. A joint run on a 32-bit device (case C) computes the difference of two terms of the size of the bond in 32-bit float (see `GridBorder`).
    """
    out = None
    for j in [] if joint_set is None else joint_set.joints:
        try:
            lam = float(scipy.linalg.eigh(j.B, j.D_bond, eigvals_only=True).min())
        except np.linalg.LinAlgError:
            continue
        if out is None or lam < out[1]:
            out = (j.name, lam)
    return out


def check_line(joint_set) -> str | None:
    """The fact of `topopt check` for a joint run on gpu-mg (a warning with the options), or None without joints.

    It gives measured facts and no rule: it does not predict a stop. Source of the numbers: the P2b review (2026-10-22, pocl, case C without the
    deflation, `gpu_tol` 1e-6; scripts `scratch/p2b-review/q5_case_c_map.py` and `hook/hook_c.py`) and P2d (2026-10-23, pocl, case C with the
    deflation, `gpu_tol` 1e-6, the cube of `tools/joint_check.py`, n 8, one load; `tests/test_gpu_mg_deflation.py`, CONTRACTS.md section 20), and
    P2d-D3h (2026-10-23, pocl, one Linux x86_64 host, revision 133e5a1, `gpu_tol` 1e-6; the two-joint part at 1.0 mm;
    and one Intel Arc iGPU, a frozen checkout of d577246, one run). Every count in the text is one run. The memory figures are the
    formulas of `deflation.Deflation.host_bytes` and `GpuMG.nbytes_parts` and the peak resident memory of `topopt run` (`ru_maxrss`) of the two-joint part at 2.2 mm and at
    1.0 mm (the run at 1.0 mm was cut at 1,500 s), in case C and in case B, against the estimate of `check` before F4-36 (610 MB + 1,000 bytes for each free dof) and the model of F4-36
    (2026-10-25, pocl, revision 74a1e46: `check.py`, `estimate_memory`; the hook and four other part families, 47 runs).
    """
    softest = beam_over_bond(joint_set)
    if softest is None:
        return None
    region, lam = softest
    return (f"joints on gpu-mg: the softest beam is that of '{region}', {lam:.1e} of the bond stiffness on its softest mode. A device without 64-bit float "
            f"(case C) computes the joint term in 32-bit float, which rounds at {FP32_ROUNDING:.0e} of a term of the size of the bond stiffness. "
            "Case C removes from its inner PCG the rigid modes of each body and 6 harmonic columns for each bonded patch. A harmonic column is a rigid motion of the patch, "
            "extended elastically into the part for each design. The extension is a Krylov loop in 64-bit float on the host; its preconditioner is the 32-bit V-cycle of the device. "
            "The coarse problem of the columns is solved in 64-bit float on the host (P2d, P2d-D3h; run.log says so). "
            "Measured on pocl, one host, case C: a cube that only a joint holds solved at 9.9e-12 and above with the rigid modes removed (2 to 3 refinement steps) "
            "and stopped at 9.9e-14 and below; without them (P2b review) it solved at 3.3e-7 and above and stopped at 1.1e-7 and below; "
            "the hook example (a fixed region and a joint) solved at 9.8e-9 without them. "
            "A part with 2 tube joints and no fixed region (1.0 mm, 600,771 free dofs, beam over bond 4.9e-10) stopped with the rigid modes alone "
            "(relative residual 5.3e-6 against 1e-6) and solved with the harmonic columns too. One run on pocl (4 threads) took 5 refinement steps in each of 4 blocks "
            "and 187 s for the extension of the first design; the next design needed 0 extension iterations, because its design space is small "
            "(a large design change is not measured on a real part). One run on an Intel Arc iGPU took 5 steps in each block and 131 s for the extension. "
            "The adjoint solves are the hard ones at 1.0 mm: two of the 9 adjoint solves of each design took 6 to 7 steps and about 1,800 inner iterations "
            "(118 s each on pocl, 32 to 37 s on the Arc), against 81 to 83 PCG iterations in case B (4.4 to 4.7 s on pocl). "
            "The columns take memory: K columns (6 for each body, 6 for each bonded patch; H of them harmonic) take 8 K bytes for each free dof on the device and "
            "24 K + 16 H bytes on the host (144 and 624 bytes for 18 columns, one body with 2 joints). The memory estimate of check counts them in the loop phase of case C (1,000 bytes for each free dof; "
            "a block with 2 joints measured 0.95 and 0.97 kB for each free dof, case C against case B of the same part). "
            "Before F4-36 the estimate was 0.69 GB at 2.2 mm and 1.21 GB at 1.0 mm. The peak memory of the topopt run of that two-joint part on pocl was 2.5 GB at 2.2 mm (75,489 free dofs) "
            "and 3.5 GB at 1.0 mm in case C (the run at 1.0 mm was cut at 1,500 s: this is the peak up to the cut), and 2.6 GB and 3.7 GB in case B, which has no column. "
            "The columns are 0.06 GB and 0.46 GB of the case C peak. The rest of the gap is not the columns and not the joints: the hook, a part with no joint, peaks at 2.4 to 2.5 times the old estimate "
            "at 0.6 to 0.8 mm on main, in the mesh step of the post-process, which grows with the free dofs to the power 2/3 and with the layout of the region shapes (F4-36). "
            "The estimate is now at least 2.6 GB at 2.2 mm and 10.3 GB at 1.0 mm for this two-joint part (4.6 GB for a layout like the hook example); what in the shapes of the two-joint part set its peak is not measured here. "
            "A solve is accepted only after a 64-bit residual check, so a stop gives no wrong answer. "
            "Options if a run stops: settings.gpu_case B on a device with 64-bit float, another solver (pardiso, amg), or a stiffer beam (a larger section, a shorter beam)")


def grid_border(problem) -> GridBorder | None:
    """The support terms of `problem` (springs and joints), or None when it has neither."""
    spr, jt = problem.springs, problem.joints
    if spr is None and jt is None:
        return None
    parts = ([spr.triplets()] if spr is not None else []) + ([jt.triplets()] if jt is not None else [])
    n_dof = (jt if jt is not None else spr).n_dof
    rows, cols, vals = sum_duplicates(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                                      np.concatenate([p[2] for p in parts]), n_dof)
    joints = [] if jt is None else jt.joints
    k = 6 * len(joints)
    softest = beam_over_bond(jt)
    soft = None if softest is None else softest[1]
    D = np.zeros((k, k))
    c_rows, c_cols, c_vals = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)], [np.zeros(0)]
    for m, j in enumerate(joints):
        coo = j.C.tocoo()
        c_rows.append(np.asarray(j.crow, dtype=np.int64)[coo.row])
        c_cols.append(6 * m + coo.col.astype(np.int64))
        c_vals.append(coo.data)
        D[6 * m:6 * m + 6, 6 * m:6 * m + 6] = j.D
    return GridBorder(rows, cols, vals, np.concatenate(c_rows), np.concatenate(c_cols), np.concatenate(c_vals), D,
                      {"springs": 0 if spr is None else len(spr.regions), "joints": len(joints), "beam_over_bond_min": soft})


CT_TILE = 4096      # columns of `CT` that `ct_dot` multiplies at once: a tile of the cache, not a limit (any tile gives a pairwise sum: the same accuracy class)


def ct_dot(CT, xs):
    """`CT @ xs` for the contiguous rows of `CT` (k, n) and the vector `xs` (n,): each row is summed with numpy's pairwise sum (F1-7).

    A BLAS product `C.T @ xs` of the same numbers adds the n terms one after the other, and its rounding grows with n. The pairwise sum of
    numpy (the sum along the contiguous axis of an array) has a rounding of order log n; it is numpy code, so the result is the same on macOS and
    Linux. The products are made for a tile of `CT_TILE` columns at a time (a temporary of k x tile numbers and not k x n); the sums of the tiles
    are added pairwise too. No long double: it is float64 on Apple silicon.
    """
    n = xs.size
    if n <= CT_TILE:
        return np.add.reduce(CT * xs, axis=1)
    parts = np.empty((CT.shape[0], -(-n // CT_TILE)))
    for q, a in enumerate(range(0, n, CT_TILE)):
        np.add.reduce(CT[:, a:a + CT_TILE] * xs[a:a + CT_TILE], axis=1, out=parts[:, q])
    return np.add.reduce(parts, axis=1)


def _add_at(y, dofs, t):
    """`y.reshape(-1)[dofs] += t`, and a clear error when `y` is not C-contiguous (a reshape would copy and the sum would be lost)."""
    if not y.flags.c_contiguous:
        raise ValueError("the output vector of the support terms must be C-contiguous")
    y.reshape(-1)[dofs] += t


class LevelBorder:
    """The terms of one level. `nodes` (nb,) are the node numbers of the level that carry a term (sorted), `dofs` (3 nb,) their dofs.

    `Kb` is (3 nb, 3 nb) CSR on those dofs and `C` is (3 nb, k) dense: the rows of the other dofs of the level are zero.
    """

    __slots__ = ("nodes", "dofs", "Kb", "C", "nn", "Kb_abs", "CT", "CT_abs")

    def __init__(self, nodes, Kb, C, nn):
        self.nodes = np.asarray(nodes, dtype=np.int32)
        self.dofs = (3 * self.nodes.astype(np.int64)[:, None] + np.arange(3)[None, :]).reshape(-1)
        self.Kb, self.C, self.nn = Kb, C, int(nn)
        self.CT = np.ascontiguousarray(C.T)                      # (k, 3 nb), each row contiguous: `ct_dot` sums a row pairwise (F1-7)
        self.Kb_abs, self.CT_abs = abs(Kb), np.abs(self.CT)      # for the absolute operator of the backward-error rule

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.size)


def _compress(Kb, C, nn) -> LevelBorder:
    """`LevelBorder` from the matrices on all the dofs of a level: `Kb` (3 nn, 3 nn) and `C` (3 nn, k), both sparse."""
    Kb = sp.csr_matrix(Kb)
    Kb.eliminate_zeros()
    C = sp.csr_matrix(C)
    C.eliminate_zeros()
    kc = Kb.tocoo()
    nodes = np.unique(np.concatenate([kc.row // 3, kc.col // 3, C.tocoo().row // 3]))
    dofs = (3 * nodes.astype(np.int64)[:, None] + np.arange(3)[None, :]).reshape(-1)
    return LevelBorder(nodes, Kb[dofs][:, dofs].tocsr(), C[dofs].toarray(), nn)


class Border:
    """The terms on every level of a `HostMG` hierarchy. See the module text."""

    def __init__(self, grid: GridBorder, levels):
        self.k = grid.k
        self.facts = dict(grid.facts)
        self.Dinv = np.zeros((self.k, self.k))
        for m in range(self.k // 6):                    # D is block diagonal: one 6 x 6 block for each joint
            blk = grid.D[6 * m:6 * m + 6, 6 * m:6 * m + 6]
            inv = np.linalg.inv(blk)
            self.Dinv[6 * m:6 * m + 6, 6 * m:6 * m + 6] = 0.5 * (inv + inv.T)
        self.Dinv_abs = np.abs(self.Dinv)
        self.levels: list[LevelBorder] = [self._fine(grid, levels[0])]
        for l in range(1, len(levels)):
            self.levels.append(self._coarsen(self.levels[-1], levels[l - 1], levels[l]))

    # -------------------------------------------------------------- construction
    def _fine(self, g: GridBorder, L0) -> LevelBorder:
        """Level 0: the grid dofs to the compact dofs of the level, with the entries on fixed dofs cut."""
        nd = 3 * L0.nn
        fixed = L0.fix.reshape(-1).astype(bool)

        def compact(grid_dofs, what):
            comp = L0.lookup[grid_dofs // 3].astype(np.int64)
            if np.any(comp < 0):
                raise RuntimeError(f"gpu-mg: a node of the {what} is not a node of the active elements (internal consistency error)")
            return 3 * comp + grid_dofs % 3

        r, c = compact(g.rows, "spring or bond matrix"), compact(g.cols, "spring or bond matrix")
        keep = ~(fixed[r] | fixed[c])
        Kb = sp.coo_matrix((g.vals[keep], (r[keep], c[keep])), shape=(nd, nd)).tocsr()
        cr = compact(g.c_rows, "bond coupling")
        keep = ~fixed[cr]
        C = sp.coo_matrix((g.c_vals[keep], (cr[keep], g.c_cols[keep])), shape=(nd, self.k)).tocsr()
        return _compress(Kb, C, L0.nn)

    def _coarsen(self, fine: LevelBorder, F, Cl) -> LevelBorder:
        """The next coarser level by the Galerkin product with the transfer of the device (the Dirichlet dofs of the coarse level are zero columns)."""
        Ps = prolongation_matrix(F, Cl)
        P3 = sp.kron(Ps[fine.nodes], sp.identity(3), format="csr") @ sp.diags(1.0 - Cl.fix.reshape(-1).astype(np.float64))
        Kbc = P3.T @ fine.Kb @ P3
        Cc = P3.T @ sp.csr_matrix(fine.C)
        return _compress(Kbc, Cc, Cl.nn)

    # -------------------------------------------------------------- the host operator
    def add_to(self, l: int, x, y):
        """`y += (Kb_l - C_l D^-1 C_l^T) x` for the vectors `x`, `y` of level `l` (nn, 3) in FP64."""
        lb = self.levels[l]
        xs = x.reshape(-1)[lb.dofs]
        t = lb.Kb @ xs
        if self.k:
            t = t - lb.C @ (self.Dinv @ ct_dot(lb.CT, xs))
        _add_at(y, lb.dofs, t)

    def add_abs_to(self, l: int, x, y):
        """`y += |Kb_l| |x| + |C_l| |D^-1| |C_l^T| |x|`: an upper bound of the entrywise absolute operator (the norm of the backward error)."""
        lb = self.levels[l]
        xs = np.abs(x.reshape(-1)[lb.dofs])
        t = lb.Kb_abs @ xs
        if self.k:
            t = t + lb.CT_abs.T @ (self.Dinv_abs @ ct_dot(lb.CT_abs, xs))
        _add_at(y, lb.dofs, t)

    def diag_terms(self, l: int):
        """`(nodes, blocks)`: the nodes of level `l` that carry a term, and the (nb, 3, 3) change of their diagonal 3 x 3 blocks, `Kb[n, n] - C[n] D^-1 C[n]^T`."""
        lb = self.levels[l]
        nb = lb.n_nodes
        blk = np.zeros((nb, 3, 3))
        bsr = lb.Kb.tobsr(blocksize=(3, 3))
        bsr.sum_duplicates()
        rows = np.repeat(np.arange(nb), np.diff(bsr.indptr))
        on = bsr.indices == rows
        blk[rows[on]] = bsr.data[on]
        if self.k:
            cn = lb.C.reshape(nb, 3, self.k)
            blk -= np.einsum("bik,kl,bjl->bij", cn, self.Dinv, cn)
        return lb.nodes, blk

    def add_dense(self, l: int, M, free_index):
        """`M[free, free] += Kb_l - C_l D^-1 C_l^T` for the dense matrix `M` of the free dofs of level `l`. `free_index` (3 nn,): the place of each dof in `M`, -1 for a Dirichlet dof."""
        lb = self.levels[l]
        T = lb.Kb.toarray()
        if self.k:
            T = T - lb.C @ self.Dinv @ lb.C.T
        place = free_index[lb.dofs]
        ok = place >= 0
        M[np.ix_(place[ok], place[ok])] += T[np.ix_(ok, ok)]

    # -------------------------------------------------------------- the device tables
    def device_tables(self, l: int):
        """`(nodes, kptr, kcol, kval, cn)` for the kernels: the bonded nodes of level `l` (nb,), the block rows of `Kb` (nb + 1, nnz node numbers, (nnz, 3, 3) blocks),
        and `C` as (nb, 3, k) blocks."""
        lb = self.levels[l]
        nb = lb.n_nodes
        bsr = lb.Kb.tobsr(blocksize=(3, 3))
        bsr.sum_duplicates()
        kptr = bsr.indptr.astype(np.int32)
        kcol = lb.nodes[bsr.indices].astype(np.int32)
        return lb.nodes, kptr, kcol, np.ascontiguousarray(bsr.data), np.ascontiguousarray(lb.C.reshape(nb, 3, self.k))

    def summary(self) -> dict:
        """Facts for the run log and the tests: the border dofs, the nodes and the blocks of `Kb` of each level."""
        return {"border_dofs": self.k, **self.facts, "nodes": [lb.n_nodes for lb in self.levels],
                "kb_blocks": [int(lb.Kb.tobsr(blocksize=(3, 3)).data.shape[0]) for lb in self.levels]}
