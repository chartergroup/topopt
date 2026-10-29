"""The transfer between two multigrid levels of gpu-mg, as tables (for the device) and as a sparse matrix (for the host).

This module has no OpenCL. `engine.py` (the device) and `border.py` (the springs and joints on every level, P2b) both read it.
The transfer is trilinear: a fine node takes the mean of its coarse parents (weight `0.5 ** (number of odd coordinates)` for each parent),
and the restriction is the transpose.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_transfer_tables(F, C):
    """Tables for the transfer between the fine level F and the coarse level C.

    Prolongation: `ptab` (8, F.nn) with the coarse node of each parent slot (-1: none) and the
    weight `pw` (F.nn). Restriction: `rtab` (27, C.nn) with the fine node of each neighbour slot
    (-1: none). The transfer is trilinear and the restriction is its transpose.
    """
    nxf1, nyf1, nzf1 = F.nodedims
    nxc1, nyc1, nzc1 = C.nodedims
    g = F.nidx.astype(np.int64)
    k = g % nzf1
    j = (g // nzf1) % nyf1
    i = g // (nzf1 * nyf1)
    nodd = (i & 1) + (j & 1) + (k & 1)
    ptab = -np.ones((8, F.nn), np.int32)
    fill = np.zeros(F.nn, np.int64)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                valid = ((di == 0) | ((i & 1) == 1)) & ((dj == 0) | ((j & 1) == 1)) & ((dk == 0) | ((k & 1) == 1))
                ci = np.clip(i // 2 + di, 0, nxc1 - 1)
                cj = np.clip(j // 2 + dj, 0, nyc1 - 1)
                ck = np.clip(k // 2 + dk, 0, nzc1 - 1)
                cid = C.lookup[(ci * nyc1 + cj) * nzc1 + ck]
                rows = np.flatnonzero(valid & (cid >= 0))
                ptab[fill[rows], rows] = cid[rows]
                fill[rows] += 1
    pw = (0.5 ** nodd).astype(np.float64)
    gc = C.nidx.astype(np.int64)
    K = gc % nzc1
    J = (gc // nzc1) % nyc1
    I = gc // (nzc1 * nyc1)
    rtab = -np.ones((27, C.nn), np.int32)
    s = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                fi, fj, fk = 2 * I + di, 2 * J + dj, 2 * K + dk
                ok = (fi >= 0) & (fi < nxf1) & (fj >= 0) & (fj < nyf1) & (fk >= 0) & (fk < nzf1)
                fn = np.where(ok, (np.clip(fi, 0, nxf1 - 1) * nyf1 + np.clip(fj, 0, nyf1 - 1)) * nzf1
                              + np.clip(fk, 0, nzf1 - 1), 0)
                rtab[s] = np.where(ok, F.lookup[fn], -1)
                s += 1
    return ptab, pw, rtab


def prolongation_matrix(F, C):
    """The scalar prolongation `P` (F.nn, C.nn) as a CSR matrix: `P[f, c] = pw[f]` for each parent `c` of the fine node `f`.

    It is the matrix of the `prolong_add` kernel without the Dirichlet masks (`P kron I3` acts on the 3 dofs of each node). The restriction
    of the device is its transpose.
    """
    ptab, pw, _ = build_transfer_tables(F, C)
    valid = ptab >= 0
    rows = np.broadcast_to(np.arange(F.nn), ptab.shape)[valid]
    return sp.csr_matrix((np.broadcast_to(pw, ptab.shape)[valid], (rows, ptab[valid])), shape=(F.nn, C.nn))
