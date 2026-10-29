"""The FFT filter (`filter.make_filter`, #333) against the reference matrix of `filter.build_filter`.

The optimizer applies the filter as a convolution and stores no matrix. These tests check that it gives the numbers of the matrix
(to rounding) for both solid rules, with and without `pad_void`, for vectors and blocks of columns, and that its exact zeros are
where the matrix has them.
"""
import numpy as np
import pytest

from topopt.filter import ConvFilter, build_filter, fft_shape, filtered_density, hat_kernel, make_filter
from topopt.voxelize import DESIGN, EMPTY, FIXED, KEEP_OUT, LOAD

TOL = 1e-12

# A label mix with all the kinds that the filter tells apart: design, solid (fixed, load), and the rest (empty, keep-out).
KINDS = [DESIGN, DESIGN, DESIGN, DESIGN, FIXED, LOAD, EMPTY, KEEP_OUT]


def random_labels(rng, shape):
    return rng.choice(KINDS, size=shape).astype(np.int8)


CASES = [((7, 9, 11), 1.0, 1.1), ((12, 5, 8), 1.0, 2.0), ((10, 10, 10), 0.5, 1.3), ((16, 9, 6), 2.0, 7.3),
         ((6, 6, 6), 1.0, 0.6), ((20, 14, 9), 1.0, 4.5)]   # (shape, h, radius): a radius below one voxel, a whole number of voxels


@pytest.mark.parametrize("rule", ["density_one", "ignore"])
@pytest.mark.parametrize("pad_void", [False, True])
@pytest.mark.parametrize("shape, h, radius", CASES)
def test_the_fft_filter_gives_the_numbers_of_the_matrix(rule, pad_void, shape, h, radius):
    rng = np.random.default_rng(hash((shape, rule, pad_void)) % 2 ** 32)
    labels = random_labels(rng, shape)
    F, FT, b = build_filter(labels, h, radius, rule, pad_void=pad_void)
    G, GT, c = make_filter(labels, h, radius, rule, pad_void=pad_void)
    n = F.shape[0]
    assert G.shape == F.shape == (n, n)
    x, y = rng.random(n), rng.standard_normal(n)
    X, Y = rng.random((n, 3)), rng.standard_normal((n, 4))
    assert np.max(np.abs(G @ x - F @ x)) < TOL
    assert np.max(np.abs(GT @ y - FT @ y)) < TOL
    assert np.max(np.abs(G @ X - F @ X)) < TOL and (G @ X).shape == (n, 3)          # a block of columns, as `gradient_wrt_x` passes
    assert np.max(np.abs(GT @ Y - FT @ Y)) < TOL and (GT @ Y).shape == (n, 4)
    assert np.max(np.abs(G.T @ y - FT @ y)) < TOL and np.max(np.abs(GT.T @ x - F @ x)) < TOL
    assert np.max(np.abs(c - b)) < TOL


@pytest.mark.parametrize("rule", ["density_one", "ignore"])
def test_b_is_exactly_zero_where_the_matrix_gives_zero(rule):
    rng = np.random.default_rng(7)
    labels = np.full((24, 12, 10), DESIGN, dtype=np.int8)
    labels[:3] = FIXED                                     # solid at one end: the far end has no solid neighbour
    labels[rng.random(labels.shape) < 0.1] = EMPTY
    _, _, b = build_filter(labels, 1.0, 2.7, rule)
    _, _, c = make_filter(labels, 1.0, 2.7, rule)
    assert np.array_equal(c == 0, b == 0) and (b == 0).sum() > 0
    if rule == "ignore":
        assert not c.any()


def test_a_design_that_is_zero_far_from_a_solid_filters_to_zero_and_never_below():
    labels = np.full((30, 8, 8), DESIGN, dtype=np.int8)
    labels[:2] = LOAD
    G, _, c = make_filter(labels, 1.0, 3.2)
    x = np.zeros(G.shape[0])
    x[:200] = 1.0
    rho_tilde = filtered_density(G, c, x)
    assert rho_tilde.min() >= 0.0 and np.array_equal(rho_tilde, np.maximum(G @ x + c, 0.0))
    assert np.all(rho_tilde[-64:] < 1e-12)                 # the last layer has no density and no solid inside the radius


def test_no_design_voxel_gives_the_empty_matrix():
    labels = np.full((4, 4, 4), FIXED, dtype=np.int8)
    F, FT, b = make_filter(labels, 1.0, 2.0)
    assert F.shape == (0, 0) and FT.shape == (0, 0) and b.size == 0


def test_the_kernel_and_the_padded_grid():
    m, k = hat_kernel(2.0)
    assert m == 2 and k.shape == (5, 5, 5) and k[2, 2, 2] == 2.0 and k[0, 2, 2] == 0.0      # d == R gets no weight
    assert int((k > 0).sum()) == 27                                                       # the offsets with d < 2: 1 + 6 + 12 + 8
    for n, s in zip((316, 120, 38), fft_shape((316, 120, 38), 8)):
        assert s >= n + 16                                                                # a linear convolution: no wrap
    cf = ConvFilter(np.full((9, 7, 5), DESIGN, dtype=np.int8), 1.0, 1.5)
    assert cf.fshape == fft_shape((9, 7, 5), 2) and cf.K.shape == (cf.fshape[0], cf.fshape[1], cf.fshape[2] // 2 + 1)


def test_the_optimizer_uses_the_fft_filter():
    """`Model` builds the operators of `make_filter`, not the matrix."""
    import inspect

    from topopt import optimize
    src = inspect.getsource(optimize.Model.__init__)
    assert "make_filter(" in src and "build_filter(" not in src


@pytest.mark.parametrize("shape, radius", [((60, 40, 30), 4.0), ((90, 30, 12), 7.5), ((80, 60, 40), 1.5)])
def test_the_arrays_of_the_filter_stay_under_the_memory_model(shape, radius):
    """The numpy arrays of the setup and of one apply each way stay under the model of `members.filter_size`: 56 B for each cell of the
    padded grid plus 24 B for each design voxel (tracemalloc sees the numpy buffers; the peak RSS measured with the FFT scratch is in the
    comment of the constants). The grids are almost all design; the last one has a small radius, so its padded grid is close to the grid."""
    import tracemalloc

    from topopt import members
    labels = np.full(shape, DESIGN, dtype=np.int8)
    labels[:3] = FIXED
    x = np.random.default_rng(0).random(int((labels == DESIGN).sum()))
    tracemalloc.start()
    try:
        G, GT, c = make_filter(labels, 1.0, radius)
        y = filtered_density(G, c, x)
        g = GT @ y
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    size = members.filter_size(radius, 1.0, x.size, shape)
    cells = size["fft_cells"]
    assert g.shape == x.shape and 8 * cells < peak < size["held_bytes_estimate"], (peak / cells, size["held_bytes_estimate"] / cells)
