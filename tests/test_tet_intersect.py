"""Tests for `topopt/tet/intersect.py` (P4, F2-6): the places where a triangle surface crosses itself. No gmsh is needed."""
import numpy as np
import pytest
import trimesh

from topopt.tet import intersect
from topopt.tet.intersect import describe_places, find_crossings


def _sphere(pushed: float | None = None):
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    V = np.array(m.vertices)
    if pushed is not None:
        V[0] *= pushed                     # one vertex through the other side of the sphere
    return V, np.array(m.faces)


def _cube(lo, size=1.0):
    m = trimesh.creation.box(extents=(size, size, size))
    m.apply_translation(np.asarray(lo, float) + size / 2)
    return np.array(m.vertices), np.array(m.faces)


def _soup(*parts):
    """Several meshes in one surface, with no vertex shared between them."""
    V = np.vstack([v for v, _ in parts])
    off = np.cumsum([0] + [len(v) for v, _ in parts[:-1]])
    return V, np.vstack([f + o for (_, f), o in zip(parts, off)])


def test_a_clean_sphere_has_no_crossing():
    V, F = _sphere()
    c = find_crossings(V, F)
    assert c.total == 0 and not c and c.pairs.shape == (0, 2) and c.points.shape == (0, 3)


def test_a_vertex_pushed_through_the_other_side_gives_crossings_where_it_went():
    V, F = _sphere(pushed=-1.3)
    c = find_crossings(V, F)
    assert c.total > 0 and len(c.pairs) == c.total and c.points.shape == (c.total, 3)
    assert np.all(np.linalg.norm(c.points, axis=1) < 13.1)          # the points are on the faces of the sphere and the spike
    for i, j in c.pairs:
        assert not set(F[i]) & set(F[j])                            # faces that share a vertex are not tested
    # the sphere without the push is clean, so the crossings are the push
    assert find_crossings(*_sphere()).total == 0


def test_the_counts_that_the_tet_tests_use_as_fixed_numbers():
    """`tests/test_tet.py` has these numbers as constants (PUSHED_CROSSINGS, CUBES_CROSSINGS), so that its tests do not need the
    finder to know what to expect from the mesher."""
    assert find_crossings(*_sphere(pushed=-1.3)).total == 10
    assert find_crossings(*_soup(_cube((0, 0, 0)), _cube((0.5, 0.5, 0.5)))).total == 18


def test_two_cubes_that_overlap_cross_inside_the_overlap():
    a, b = _cube((0, 0, 0)), _cube((0.5, 0.5, 0.5))
    V, F = _soup(a, b)
    c = find_crossings(V, F)
    assert c.total > 0
    assert np.all(c.points >= 0.5 - 1e-9) and np.all(c.points <= 1.0 + 1e-9)    # the cut lines lie in the overlap box


def test_a_touch_is_not_a_crossing():
    """A cube that stands on the face of another one (coplanar contact, the vertices apart) and one that
    touches it with a corner give no crossing: an edge that ends on a face is a touch."""
    V, F = _soup(_cube((0, 0, 0)), _cube((1, 0, 0)))               # face to face
    assert find_crossings(V, F).total == 0
    V, F = _soup(_cube((0, 0, 0)), _cube((1, 1, 1)))               # corner to corner
    assert find_crossings(V, F).total == 0
    V, F = _soup(_cube((0, 0, 0)), _cube((3, 0, 0)))               # apart
    assert find_crossings(V, F).total == 0


def test_the_list_is_made_as_long_as_the_count(monkeypatch):
    V, F = _sphere(pushed=-1.3)
    want = find_crossings(V, F)
    monkeypatch.setattr(intersect, "FIRST_CAPACITY", 2)            # the first list is too short: one more pass fills all
    got = find_crossings(V, F)
    assert want.total > 2 and got.total == want.total
    assert np.array_equal(got.pairs, want.pairs) and np.allclose(got.points, want.points)


def _far_triangle(corner, size=1.0):
    c = np.asarray(corner, float)
    return np.array([c, c + [size, 0, 0], c + [0, size, 0]]), np.array([[0, 1, 2]])


def test_a_large_box_around_small_faces_does_not_make_more_grid_cells_than_faces(monkeypatch):
    """Two small triangles 1000 mm apart make the bounding box 1000 mm wide; the median edge stays at 1.5 mm. With 3 x the median
    edge for the cell, the grid had 1.1e7 cells for 1,284 faces (the arithmetic for 600 mm and 0.3 mm gives 2.4 GB; P4 review, L3).
    The cell is now at least (box volume / faces) ** (1/3), and the crossings are the same."""
    sphere = _sphere(pushed=-1.3)
    V, F = _soup(sphere, _far_triangle((-500, -500, -500)), _far_triangle((500, 500, 500)))
    seen = []
    real = intersect._crossings

    def spy(V_, F_, cell, origin, nx, ny, nz, tol, capacity):
        seen.append((cell, nx * ny * nz))
        return real(V_, F_, cell, origin, nx, ny, nz, tol, capacity)
    monkeypatch.setattr(intersect, "_crossings", spy)
    c = find_crossings(V, F)
    median = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    want = max(3.0 * median, (np.prod(V.max(axis=0) - V.min(axis=0)) / len(F)) ** (1 / 3))
    cell, cells = seen[-1]
    assert cell == pytest.approx(want) and cell > 3.0 * median * 5
    assert cells <= 4 * len(F)
    assert c.total == 10 and c.total == find_crossings(*sphere).total


def test_a_surface_in_a_box_that_fits_keeps_the_cell_of_three_median_edges(monkeypatch):
    V, F = _sphere(pushed=-1.3)
    seen = []
    real = intersect._crossings
    monkeypatch.setattr(intersect, "_crossings", lambda V_, F_, cell, *a: (seen.append(cell), real(V_, F_, cell, *a))[1])
    find_crossings(V, F)
    median = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    assert seen[-1] == pytest.approx(3.0 * median)


def test_a_surface_with_one_face_has_no_crossing():
    V, F = _cube((0, 0, 0))
    assert find_crossings(V, F[:1]).total == 0


def test_describe_places_gives_a_short_list_and_the_rest_as_a_count():
    pts = np.arange(36, dtype=float).reshape(12, 3)
    text = describe_places(pts)
    assert text.startswith("(0.00, 1.00, 2.00), (3.00, 4.00, 5.00)") and text.endswith("and 2 more")
    assert "more" not in describe_places(pts[:3])
