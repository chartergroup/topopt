"""Density filter, Heaviside projection and SIMP rules.

The filter averages the density of the neighbours inside the filter radius
(linear hat weights) on the voxel grid. The optimizer applies it as an FFT
convolution (`make_filter`, #333): no matrix is stored, so its memory grows with
the grid and not with the radius cubed. `build_filter` builds the same filter as a
sparse matrix: it is the reference that the tests compare with. Solid voxels
(FIXED, LOAD and `regions.solid`) follow the rule `solid_neighbours`:
`density_one` (the default) counts them as neighbours with density 1. `ignore`
drops them from the sum and from the weights, so each row averages the design
voxels only (the rule of the pipeline). See CONTRACTS.md section 22a.

With `pad_void` (only `settings.robust: wall`, CONTRACTS.md section 22b) every neighbour that is
not a design or a solid voxel counts as VOID in the weights: outside the design space, outside the
grid, and in a keep-out region. The weights are then those of an unbounded design space, so a wall at
a face is as thick as a wall inside. Solid voxels keep their rule (`density_one`: density 1;
`ignore`: out of the sum and out of the weights), never void.

`heaviside_project`, `heaviside_deriv`, `simp_E` and `simp_dEdrho` are copied
from an earlier project of the same author (topopt_engine.py).
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from .voxelize import DESIGN, FIXED, LOAD


SOLID_RULES = ("density_one", "ignore")


def build_filter(labels: np.ndarray, h: float, radius: float, solid_neighbours: str = "density_one",
                 pad_void: bool = False):
    """Return `(F, FT, b)` for the design voxels, with F as a sparse matrix (the reference of `make_filter`).

    rho_tilde = F @ x + b, where x is the density of the design voxels.
    FT is the transpose of F as a csr matrix (for the chain rule).
    The matrix has `stencil_count(R)` entries in an interior row, so its memory grows with (radius / voxel)^3 for each design voxel
    (about 6.5 GiB at 140,000 design voxels and R 7.5 voxels): the optimizer uses `make_filter`, which gives the same numbers.
    `solid_neighbours` is the rule for FIXED and LOAD voxels (solid regions have the label LOAD):
    "density_one" gives them density 1 through `b`. "ignore" leaves them out of the weights, so
    every row of F sums to 1 and b is 0.
    `pad_void` (default off) counts every neighbour that is not a design or a solid voxel as void: the weights
    sum over the whole kernel, so a row near a face or a keep-out region sums to less than 1. The entries of F
    are the same voxels as with it off; only the divisor of the weights changes. Solid voxels keep their rule.
    """
    if solid_neighbours not in SOLID_RULES:
        raise ValueError(f"solid_neighbours must be one of {SOLID_RULES}, got {solid_neighbours!r}")
    nx, ny, nz = labels.shape
    flat = labels.reshape(-1)
    design = np.flatnonzero(flat == DESIGN)
    n = design.size
    solid = (flat == FIXED) | (flat == LOAD)
    dropped = solid if solid_neighbours == "ignore" else np.zeros_like(solid)   # pad_void: solid voxels that leave the weights
    if solid_neighbours == "ignore":
        solid = np.zeros_like(solid)   # no solid neighbour: the rows renormalise over design voxels
    design_index = np.full(flat.size, -1, dtype=np.int64)
    design_index[design] = np.arange(n)
    i = design // (ny * nz)
    j = (design // nz) % ny
    k = design % nz
    R = radius / h  # radius in voxel units
    m = int(np.ceil(R))
    rows, cols, vals = [], [], []
    wsum = np.zeros(n)  # sum of the weights of all neighbours
    wsolid = np.zeros(n)  # sum of the weights of the solid neighbours
    for di in range(-m, m + 1):
        for dj in range(-m, m + 1):
            for dk in range(-m, m + 1):
                d = np.sqrt(di * di + dj * dj + dk * dk)
                if d >= R:
                    continue
                w = R - d
                ni, nj, nk = i + di, j + dj, k + dk
                ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny) & (nk >= 0) & (nk < nz)
                nb = np.where(ok, (np.clip(ni, 0, nx - 1) * ny + np.clip(nj, 0, ny - 1)) * nz
                              + np.clip(nk, 0, nz - 1), 0)
                col = np.where(ok, design_index[nb], -1)
                is_design = col >= 0
                is_solid = ok & solid[nb]
                if pad_void:   # every neighbour counts (void included), but a solid voxel of the rule `ignore`
                    wsum += w * ~(ok & dropped[nb])
                else:
                    wsum += w * (is_design | is_solid)
                wsolid += w * is_solid
                rows.append(np.flatnonzero(is_design))
                cols.append(col[is_design])
                vals.append(np.full(int(is_design.sum()), w))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals) / wsum[rows]
    F = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return F, F.T.tocsr(), wsolid / wsum


def hat_kernel(R: float) -> tuple[int, np.ndarray]:
    """The weights of the filter on the offsets of a (2m+1)^3 box, m = ceil(R): w = R - d inside the radius (d < R), else 0.

    The same weights as the loop of `build_filter` (the voxel itself has the weight R)."""
    m = int(np.ceil(R))
    g = np.arange(-m, m + 1, dtype=np.float64)
    d = np.sqrt(g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2)
    return m, np.where(d < R, R - d, 0.0)


class ConvFilter:
    """The filter of `build_filter` without a matrix (#333): the same weights, applied as an FFT convolution on the grid.

    With K the hat kernel and wsum the row sums of the weights: F @ x = conv(X) / wsum at the design voxels, X = x on the design
    voxels and 0 elsewhere. K is symmetric, so FT @ y = conv(Y / wsum) at the design voxels, Y = y on the design voxels.
    wsum = conv(design | solid) (`pad_void`: sum(K) - conv(dropped)) and b = conv(solid) / wsum, as in `build_filter`. A convolution
    of a 0/1 field is 0 or at least the smallest weight of K, so those two are set to 0 below half of that weight: the FFT
    rounding (about 1e-13) leaves no false weight. (Not when R is within rounding of a lattice distance, for example
    2.1 / 0.3 = 7.000000000000001: then the smallest weight is about 1e-15, and values of about 1e-16 can differ from the matrix.)
    The results equal those of the matrix to rounding (about 1e-15).
    Memory: the transform of K on the padded grid stays (16 B for each complex value, about 8 B a padded cell); one apply
    needs about 40 B a padded cell for a moment. The arrays of design size come on top (`members.filter_size`). The padded
    grid is the grid plus 2m in each direction, rounded up to a fast FFT length (`fft_shape`).
    """

    def __init__(self, labels: np.ndarray, h: float, radius: float, solid_neighbours: str = "density_one",
                 pad_void: bool = False, workers: int = 1):
        if solid_neighbours not in SOLID_RULES:
            raise ValueError(f"solid_neighbours must be one of {SOLID_RULES}, got {solid_neighbours!r}")
        self.shape = tuple(int(s) for s in labels.shape)
        self.m, kernel = hat_kernel(radius / h)
        self.fshape = fft_shape(self.shape, self.m)
        self.workers = workers
        self.K = sfft.rfftn(kernel, s=self.fshape, workers=workers)
        floor = 0.5 * float(kernel[kernel > 0].min())          # a convolution of a 0/1 field below this is 0
        self.mask = labels == DESIGN
        self.n = int(self.mask.sum())
        solid = (labels == FIXED) | (labels == LOAD)
        dropped = solid if solid_neighbours == "ignore" else np.zeros_like(solid)   # pad_void: solid voxels that leave the weights
        if solid_neighbours == "ignore":
            solid = np.zeros_like(solid)
        if pad_void:   # every neighbour counts (void and out of the grid too), but a solid voxel of the rule `ignore`
            wsum = float(kernel.sum()) - self._snap(self.conv(dropped.astype(np.float64)), floor)[self.mask]
        else:
            wsum = self._snap(self.conv((self.mask | solid).astype(np.float64)), floor)[self.mask]
        self.wsum = wsum
        self.b = self._snap(self.conv(solid.astype(np.float64)), floor)[self.mask] / wsum

    @staticmethod
    def _snap(a: np.ndarray, floor: float) -> np.ndarray:
        a[a < floor] = 0.0
        return a

    def conv(self, a: np.ndarray) -> np.ndarray:
        """The convolution of the grid field `a` with K, on the grid (zero outside it)."""
        spec = sfft.rfftn(a, s=self.fshape, workers=self.workers)
        spec *= self.K
        out = sfft.irfftn(spec, s=self.fshape, workers=self.workers)
        m, (nx, ny, nz) = self.m, self.shape
        return out[m:m + nx, m:m + ny, m:m + nz]

    def forward(self, x: np.ndarray) -> np.ndarray:
        X = np.zeros(self.shape)
        X[self.mask] = x
        return self.conv(X)[self.mask] / self.wsum

    def transpose(self, y: np.ndarray) -> np.ndarray:
        Y = np.zeros(self.shape)
        Y[self.mask] = y / self.wsum
        return self.conv(Y)[self.mask]


class _FilterOperator(LinearOperator):
    """F (or FT) of a `ConvFilter` behind the `@` of a matrix: a vector (n,) or a block (n, k) of columns."""

    def __init__(self, conv: ConvFilter, transposed: bool):
        self.conv, self.transposed = conv, transposed
        super().__init__(dtype=np.float64, shape=(conv.n, conv.n))

    def _matvec(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return self.conv.transpose(x) if self.transposed else self.conv.forward(x)

    def _rmatvec(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return self.conv.forward(x) if self.transposed else self.conv.transpose(x)

    def _transpose(self):
        return _FilterOperator(self.conv, not self.transposed)

    _adjoint = _transpose   # real weights


def fft_shape(shape, m: int) -> tuple[int, ...]:
    """The padded grid of the FFT filter: each side n + 2m (a linear convolution, no wrap), rounded up to a fast FFT length."""
    return tuple(int(sfft.next_fast_len(int(n) + 2 * int(m), real=True)) for n in shape)


def make_filter(labels: np.ndarray, h: float, radius: float, solid_neighbours: str = "density_one", pad_void: bool = False):
    """Return `(F, FT, b)` like `build_filter`, with F and FT as operators of a `ConvFilter` (no stored matrix).

    `F @ x` and `FT @ y` take a vector or a block of columns, as the matrix does. With no design voxel it returns the (empty) matrix.
    """
    if not np.any(labels == DESIGN):
        return build_filter(labels, h, radius, solid_neighbours, pad_void=pad_void)
    conv = ConvFilter(labels, h, radius, solid_neighbours, pad_void=pad_void)
    return _FilterOperator(conv, False), _FilterOperator(conv, True), conv.b


def filtered_density(F, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """rho_tilde = F @ x + b for a design x in [0, 1], cut at 0 from below.

    The exact value is 0 or more. The FFT filter can give about -1e-16 where the matrix gives 0, and a negative density makes
    rho ** penal undefined for a penal that is not a whole number, so the rounding is cut off. With the matrix it changes nothing."""
    return np.maximum(F @ x + b, 0.0)


def solid_pull(labels: np.ndarray, h: float, radius: float, solid_neighbours: str = "density_one",
               threshold: float = 0.1, pad_void: bool = False) -> dict:
    """How far the solid neighbours pull the design voxels (the `b` of `build_filter`), without the matrix.

    With `density_one`, a design voxel with density 0 next to a fixed, load or solid voxel gets a
    filtered density `b` above 0 (KB pitfall filter-solid-bias). `b` is the weight share of the
    solid neighbours, the same number as `build_filter` gives (also with `pad_void`, where the divisor is the whole kernel). It is computed here with an FFT
    convolution of the hat kernel, so it does not build the matrix (check at any radius is cheap).
    Return the number of design voxels with `b > threshold`, the largest `b`, and the rule.
    """
    from scipy.signal import fftconvolve

    if solid_neighbours not in SOLID_RULES:
        raise ValueError(f"solid_neighbours must be one of {SOLID_RULES}, got {solid_neighbours!r}")
    design = labels == DESIGN
    n = int(design.sum())
    solid = (labels == FIXED) | (labels == LOAD)
    if solid_neighbours == "ignore" or not solid.any() or n == 0:
        return {"rule": solid_neighbours, "design_voxels": n, "threshold": threshold, "voxels_above": 0,
                "share_above": 0.0, "b_max": 0.0, "solid_voxels": int(solid.sum())}
    R = radius / h
    m = int(np.ceil(R))
    g = np.arange(-m, m + 1, dtype=np.float64)
    d = np.sqrt(g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2)
    kernel = np.where(d < R, R - d, 0.0)
    if pad_void:   # every neighbour counts, void included: the divisor is the sum of the whole kernel (rule density_one here)
        wsum = np.full(design.shape, float(kernel.sum()))
    else:
        wsum = fftconvolve((design | solid).astype(np.float64), kernel, mode="same")
    wsolid = fftconvolve(solid.astype(np.float64), kernel, mode="same")
    b = np.where(design, wsolid / np.maximum(wsum, 1e-300), 0.0)
    above = int((b > threshold).sum())
    return {"rule": solid_neighbours, "design_voxels": n, "threshold": threshold, "voxels_above": above,
            "share_above": round(above / n, 4), "b_max": round(float(b.max()), 4), "solid_voxels": int(solid.sum())}


def solid_pull_line(sp: dict | None) -> str | None:
    """One text line for `check` and `run.log` from `solid_pull`. None when there is nothing to say."""
    if not sp or not sp["solid_voxels"]:
        return None
    if sp["rule"] == "ignore":
        return "solid pull: none (rule ignore: solid voxels are not neighbours of a design voxel)"
    return (f"solid pull: {sp['voxels_above']:,} of {sp['design_voxels']:,} design voxels "
            f"({100 * sp['share_above']:.1f} percent) have a filtered density above {sp['threshold']:g} "
            f"from solid neighbours alone (rule density_one); the largest is {sp['b_max']:g}. "
            f"settings.filter_solid_neighbours: ignore removes this pull")


# --- copied from an earlier project ------------------------------------------


def heaviside_project(rho_tilde: np.ndarray, beta: float, eta: float) -> np.ndarray:
    a = np.tanh(beta * eta)
    b = np.tanh(beta * (1.0 - eta))
    return (a + np.tanh(beta * (rho_tilde - eta))) / (a + b)


def heaviside_deriv(rho_tilde: np.ndarray, beta: float, eta: float) -> np.ndarray:
    a = np.tanh(beta * eta)
    b = np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (rho_tilde - eta)) ** 2) / (a + b)


def simp_E(rho: np.ndarray, p: float, E0: float, Emin: float) -> np.ndarray:
    return Emin + rho ** p * (E0 - Emin)


def simp_dEdrho(rho: np.ndarray, p: float, E0: float, Emin: float) -> np.ndarray:
    return p * np.maximum(rho, 1e-12) ** (p - 1) * (E0 - Emin)
