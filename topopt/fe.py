"""Hex8 voxel finite elements: element matrix, dof table, fast assembler.

Grid numbering follows CONTRACTS.md section 3. The Assembler works on the
reduced system: it keeps only the free dofs. The index of a dof in the reduced
system is its position in the sorted array `free_dofs`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit, prange

# Local node offsets (di, dj, dk). Order is fixed by CONTRACTS.md section 4.
LOCAL_NODES = np.array(
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=np.int64)


def hex8_ke(E: float, nu: float, h: float) -> np.ndarray:
    """Stiffness matrix (24, 24) of one cube of edge h. 2x2x2 Gauss."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    sgn = 2.0 * LOCAL_NODES - 1.0  # natural coordinates of the nodes, -1 or +1
    g = 1.0 / np.sqrt(3.0)
    ke = np.zeros((24, 24))
    detj = (h / 2.0) ** 3
    for gx in (-g, g):
        for gy in (-g, g):
            for gz in (-g, g):
                # d N / d xi, then chain rule with dxi/dx = 2/h
                dn = np.empty((8, 3))
                for a in range(8):
                    sx, sy, sz = sgn[a]
                    dn[a, 0] = 0.125 * sx * (1 + sy * gy) * (1 + sz * gz)
                    dn[a, 1] = 0.125 * sy * (1 + sx * gx) * (1 + sz * gz)
                    dn[a, 2] = 0.125 * sz * (1 + sx * gx) * (1 + sy * gy)
                dn *= 2.0 / h
                B = np.zeros((6, 24))
                for a in range(8):
                    dx, dy, dz = dn[a]
                    c = 3 * a
                    B[0, c] = dx
                    B[1, c + 1] = dy
                    B[2, c + 2] = dz
                    B[3, c], B[3, c + 1] = dy, dx
                    B[4, c + 1], B[4, c + 2] = dz, dy
                    B[5, c], B[5, c + 2] = dz, dx
                ke += B.T @ D @ B * detj
    return 0.5 * (ke + ke.T)


def isotropic_D(E: float, nu: float) -> np.ndarray:
    """6x6 matrix D0 (engineering shear strain), the same as in `hex8_ke`."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def strain_matrix_centre(h: float) -> np.ndarray:
    """B matrix (6, 24) of a hex8 cube of edge h at the element centre.

    The local dof order is the order of `fe.LOCAL_NODES`, node-major, then x, y, z.
    """
    sgn = 2.0 * LOCAL_NODES - 1.0  # -1 or +1
    dn = 0.125 * sgn * (2.0 / h)   # d N / d x at the centre, shape (8, 3)
    B = np.zeros((6, 24))
    for a in range(8):
        dx, dy, dz = dn[a]
        c = 3 * a
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c], B[3, c + 1] = dy, dx
        B[4, c + 1], B[4, c + 2] = dz, dy
        B[5, c], B[5, c + 2] = dz, dx
    return B


def von_mises(sig: np.ndarray) -> np.ndarray:
    """Von Mises stress of stress rows (n, 6): sx, sy, sz, txy, tyz, txz."""
    sx, sy, sz, txy, tyz, txz = (sig[:, i] for i in range(6))
    q = (sx * sx + sy * sy + sz * sz - sx * sy - sy * sz - sz * sx
         + 3.0 * (txy * txy + tyz * tyz + txz * txz))
    return np.sqrt(np.maximum(q, 0.0))


def edof_table(problem_shape, active_mask):
    """Dof table of the active voxels, in grid numbering.

    Returns (edof (n_el, 24) int64, active_index (n_el,) int64).
    """
    nx, ny, nz = (int(s) for s in problem_shape)
    mask = np.asarray(active_mask).reshape(-1).astype(bool)
    if mask.size != nx * ny * nz:
        raise ValueError("active_mask does not match the grid shape")
    idx = np.flatnonzero(mask).astype(np.int64)
    i = idx // (ny * nz)
    j = (idx // nz) % ny
    k = idx % nz
    n0 = (i * (ny + 1) + j) * (nz + 1) + k
    off = (LOCAL_NODES[:, 0] * (ny + 1) + LOCAL_NODES[:, 1]) * (nz + 1) + LOCAL_NODES[:, 2]
    nodes = n0[:, None] + off[None, :]
    edof = (3 * nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)
    return edof, idx


def triplets_aligned(free_dofs) -> bool:
    """True if the free dofs come in complete x, y, z triplets (3x3 blocks possible)."""
    fd = np.asarray(free_dofs)
    return fd.size % 3 == 0 and bool(np.all(fd[0::3] % 3 == 0) and np.all(fd[1::3] == fd[0::3] + 1)
                                     and np.all(fd[2::3] == fd[0::3] + 2))


def rigid_body_modes(free_dofs, shape, h=1.0):
    """Six rigid-body modes on the free dofs, shape (n_free, 6).

    Uses grid dof numbers (CONTRACTS.md section 3). Used as AMG near-nullspace.
    """
    nx, ny, nz = (int(s) for s in shape)
    fd = np.asarray(free_dofs, dtype=np.int64)
    node = fd // 3
    comp = fd % 3
    i = node // ((ny + 1) * (nz + 1))
    j = (node // (nz + 1)) % (ny + 1)
    k = node % (nz + 1)
    x = np.stack([i, j, k], axis=1).astype(np.float64) * h
    x -= x.mean(axis=0)
    B = np.zeros((fd.size, 6))
    r = np.arange(fd.size)
    for c in range(3):  # translations
        B[r[comp == c], c] = 1.0
    # rotations: a x (r), with a = e_x, e_y, e_z
    m = comp == 0
    B[r[m], 4] = x[m, 2]
    B[r[m], 5] = -x[m, 1]
    m = comp == 1
    B[r[m], 3] = -x[m, 2]
    B[r[m], 5] = x[m, 0]
    m = comp == 2
    B[r[m], 3] = x[m, 1]
    B[r[m], 4] = -x[m, 0]
    return B


# --------------------------------------------------------------------------
# numba kernels (setup)
# --------------------------------------------------------------------------

@njit(cache=True)
def _mark_slots(nodes_c, slotpos, present):
    n_el = nodes_c.shape[0]
    for e in range(n_el):
        for b in range(8):
            cb = nodes_c[e, b]
            for a in range(8):
                present[cb, slotpos[a, b]] = 1


@njit(cache=True)
def _build_pattern(gnode, lookup, present, offs, rd, ndof_red):
    """Build reduced CSC pattern and the map full entry -> reduced entry."""
    nn = gnode.shape[0]
    cnt = np.zeros(nn, np.int64)
    rank = np.zeros((nn, 27), np.uint8)
    for c in range(nn):
        r = 0
        for s in range(27):
            if present[c, s]:
                rank[c, s] = r
                r += 1
        cnt[c] = r
    fstart = np.zeros(nn + 1, np.int64)
    for c in range(nn):
        fstart[c + 1] = fstart[c] + 9 * cnt[c]
    nfull = fstart[nn]
    fmap = np.full(nfull, -1, np.int32)
    # first pass: count kept entries
    indptr = np.zeros(ndof_red + 1, np.int64)
    nkeep = 0
    for c in range(nn):
        for cc in range(3):
            col = rd[3 * c + cc]
            p = fstart[c] + cc * 3 * cnt[c]
            if col < 0:
                continue
            q = 0
            for s in range(27):
                if not present[c, s]:
                    continue
                na = lookup[gnode[c] + offs[s]]
                for r in range(3):
                    row = rd[3 * na + r]
                    if row >= 0:
                        fmap[p + 3 * q + r] = nkeep
                        nkeep += 1
                q += 1
            indptr[col + 1] = nkeep
    indices = np.empty(nkeep, np.int32)
    for c in range(nn):
        for cc in range(3):
            col = rd[3 * c + cc]
            if col < 0:
                continue
            p = fstart[c] + cc * 3 * cnt[c]
            q = 0
            for s in range(27):
                if not present[c, s]:
                    continue
                na = lookup[gnode[c] + offs[s]]
                for r in range(3):
                    m = fmap[p + 3 * q + r]
                    if m >= 0:
                        indices[m] = rd[3 * na + r]
                q += 1
    return indptr, indices, fmap, fstart, cnt, rank


@njit(cache=True)
def _make_qtab(nodes_c, slotpos, rank):
    n_el = nodes_c.shape[0]
    q = np.empty((n_el, 8, 8), np.uint8)
    for e in range(n_el):
        for b in range(8):
            cb = nodes_c[e, b]
            for a in range(8):
                q[e, a, b] = rank[cb, slotpos[a, b]]
    return q


@njit(parallel=True, cache=True)
def _fill_color(order, Ee, ke, nodes_c, qtab, fstart, cnt, fmap, data):
    for t in prange(order.shape[0]):
        e = order[t]
        ee = Ee[e]
        for b in range(8):
            cb = nodes_c[e, b]
            base = fstart[cb]
            w = 3 * cnt[cb]
            for a in range(8):
                q = qtab[e, a, b]
                for c in range(3):
                    for r in range(3):
                        m = fmap[base + c * w + 3 * q + r]
                        if m >= 0:
                            data[m] += ee * ke[3 * a + r, 3 * b + c]


class Assembler:
    """Fast assembler of the reduced stiffness matrix.

    edof: (n_el, 24) grid dof numbers. ke: (24, 24) unit element matrix.
    free_dofs: sorted grid dof numbers of the free dofs.
    K(Ee) returns sum_e Ee[e] * ke as csc_matrix (n_free, n_free), full
    symmetric. The index arrays are shared between calls (same objects).
    """

    def __init__(self, edof, ke, free_dofs):
        edof = np.asarray(edof)
        free_dofs = np.asarray(free_dofs, dtype=np.int64)
        if np.any(np.diff(free_dofs) <= 0):
            raise ValueError("free_dofs must be sorted and unique")
        self.n_el = edof.shape[0]
        self.n = free_dofs.size
        self.ke = np.ascontiguousarray(ke, dtype=np.float64)
        nodes = (edof[:, 0::3] // 3).astype(np.int64)  # (n_el, 8)
        s_i = int(nodes[0, 1] - nodes[0, 0])
        s_j = int(nodes[0, 3] - nodes[0, 0])
        # compact node numbers, ascending in grid node number
        maxn = int(nodes.max()) + 2 + s_i + s_j  # room for neighbour lookups
        lookup = np.full(maxn + 1 + 2 * (s_i + s_j + 1), -1, np.int32)
        self._pad = s_i + s_j + 1  # lookup is shifted so negative offsets are safe
        used = np.zeros(lookup.size, bool)
        used[nodes.ravel() + self._pad] = True
        gnode = np.flatnonzero(used) - self._pad
        nn = gnode.size
        lookup[gnode + self._pad] = np.arange(nn, dtype=np.int32)
        nodes_c = lookup[nodes + self._pad].astype(np.int32)
        # 27 neighbour slots, sorted by grid node offset
        codes = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
        offv = np.array([a * s_i + b * s_j + c for a, b, c in codes])
        order = np.argsort(offv, kind="stable")
        offs = offv[order].astype(np.int64)
        code_to_slot = {codes[o]: s for s, o in enumerate(order)}
        slotpos = np.zeros((8, 8), np.int64)  # slot of node a in column of node b
        for a in range(8):
            for b in range(8):
                d = tuple(int(v) for v in LOCAL_NODES[a] - LOCAL_NODES[b])
                slotpos[a, b] = code_to_slot[d]
        present = np.zeros((nn, 27), np.uint8)
        _mark_slots(nodes_c, slotpos, present)
        # reduced dof map: compact dof -> reduced index or -1
        rd = np.full(3 * nn, -1, np.int64)
        fnode = free_dofs // 3
        cn = lookup[fnode + self._pad]
        if np.any(cn < 0):
            raise ValueError("a free dof belongs to a node with no element")
        rd[3 * cn.astype(np.int64) + free_dofs % 3] = np.arange(free_dofs.size)
        # lookup shifted: wrap into a function of (gnode + off) without pad
        lk = lookup  # index = gnode + off + pad
        offs_p = offs + self._pad
        gnode_p = gnode  # kernel adds offs_p to gnode, lookup index = gnode + offs + pad
        # kernel uses lookup[gnode + offs]; so give it a pre-shifted offset array
        indptr, indices, fmap, fstart, cnt, rank = _build_pattern(
            gnode_p, lk, present, offs_p, rd, free_dofs.size)
        self._indptr = indptr.astype(np.int32)
        self._indices = indices
        self._nnz = indices.size
        self._fmap, self._fstart, self._cnt = fmap, fstart, cnt
        self._nodes_c = nodes_c
        self._qtab = _make_qtab(nodes_c, slotpos, rank)
        # colour groups: elements of one colour share no node
        n0 = nodes[:, 0]
        ci = n0 // s_i
        cj = (n0 % s_i) // s_j
        ck = n0 % s_j
        color = ((ci & 1) * 4 + (cj & 1) * 2 + (ck & 1)).astype(np.int8)
        self._groups = [np.flatnonzero(color == c).astype(np.int64) for c in range(8)]
        self._groups = [g for g in self._groups if g.size]

    def K(self, Ee: np.ndarray) -> sp.csc_matrix:
        Ee = np.ascontiguousarray(Ee, dtype=np.float64)
        if Ee.shape != (self.n_el,):
            raise ValueError("Ee must have shape (n_el,)")
        data = np.zeros(self._nnz)
        for g in self._groups:
            _fill_color(g, Ee, self.ke, self._nodes_c, self._qtab,
                        self._fstart, self._cnt, self._fmap, data)
        K = sp.csc_matrix((data, self._indices, self._indptr),
                          shape=(self.n, self.n), copy=False)
        return K
