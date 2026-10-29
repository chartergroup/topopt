"""Small structured tet meshes for the tests of `topopt.verify` (no gmsh needed)."""
import itertools

import numpy as np

# Kuhn split of a cube: 6 tets around the diagonal 0-7. Cubes next to each other share consistent faces.
CUBE_TETS = ((0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7))


def box_mesh(lengths, cells):
    """A box [0, Lx] x [0, Ly] x [0, Lz] of cells[0] x cells[1] x cells[2] cubes, 6 tets each.

    Return `p` (3, nv) and `t` (4, nel) (the layout of `P2Model`).
    """
    nx, ny, nz = cells
    idx = {c: i for i, c in enumerate(itertools.product(range(nx + 1), range(ny + 1), range(nz + 1)))}
    p = np.array(list(idx), float) * (np.array(lengths, float) / np.array(cells, float))
    t = []
    for cx, cy, cz in itertools.product(range(nx), range(ny), range(nz)):
        corner = [(cx + (v & 1), cy + ((v >> 1) & 1), cz + ((v >> 2) & 1)) for v in range(8)]
        for tet in CUBE_TETS:
            t.append([idx[corner[v]] for v in tet])
    return p.T.copy(), np.array(t, np.int64).T.copy()


def annulus_mesh(a, b, length, n_r, n_t, n_z, sweep=2 * np.pi):
    """A thick ring (inner radius `a`, outer radius `b`, along z from 0 to `length`) of n_r x n_t x n_z cells, 6 tets each.

    Every cell is split by the Kuhn rule in index space, so the neighbours share their faces. With `sweep` below 2 pi the ring is an open sector
    (n_t + 1 columns; no wrap). The inner wall has the vertices exactly at radius `a`: its facets are chords (a polygon). Return `p` (3, nv) and
    `t` (4, nel). Half of the tets have a negative signed volume; `P2Model` takes the absolute value.
    """
    full = sweep >= 2 * np.pi
    cols = n_t if full else n_t + 1
    r = np.linspace(a, b, n_r + 1)
    th = sweep * np.arange(cols) / n_t
    z = np.linspace(0.0, length, n_z + 1)

    def idx(i, j, k):
        return (i * cols + (j % cols if full else j)) * (n_z + 1) + k

    p = np.array([[r[i] * np.cos(th[j]), r[i] * np.sin(th[j]), z[k]] for i in range(n_r + 1) for j in range(cols) for k in range(n_z + 1)])
    t = []
    for i, j, k in itertools.product(range(n_r), range(n_t), range(n_z)):
        for perm in itertools.permutations(range(3)):
            c = [0, 0, 0]
            path = [idx(i, j, k)]
            for ax in perm:
                c[ax] = 1
                path.append(idx(i + c[0], j + c[1], k + c[2]))
            t.append(path)
    return p.T.copy(), np.array(t, np.int64).T.copy()
