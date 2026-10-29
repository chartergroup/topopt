"""#429: the trimesh distance queries of the `sdf` mesh step hold about `smooth_sdf.PAIRS` point-face pairs at a time.

`trimesh.proximity.closest_point` (and `signed_distance`, which calls it) makes its arrays for all the pairs of a point and a nearby face of
one call at once. On a generated lattice of 6.2 million fine voxels (a mesh of a million vertices), one query of the snap took 3.5 GB more
and one query of the cut to the envelope 2.9 GB; a snap of 37,000 points to a region of 2,000 faces took 3.0 GB. The step now counts the
pairs of each point first and asks runs of points with up to PAIRS pairs (`_query`). The answer of each point is the same as in one call.
"""
import numpy as np
import pytest
import trimesh
import trimesh.proximity

from topopt import smooth_sdf as ss


def box(lo, hi):
    return trimesh.creation.box(bounds=[lo, hi])


@pytest.fixture
def calls(monkeypatch):
    """(points, point-face pairs) of each outer call of `trimesh.proximity.closest_point` and `signed_distance` (the pairs as trimesh finds them)."""
    seen, depth = [], [0]

    def counted(real):
        def inner(mesh, points):
            if depth[0] == 0:
                seen.append((len(points), sum(len(c) for c in trimesh.proximity.nearby_faces(mesh, points))))
            depth[0] += 1
            try:
                return real(mesh, points)
            finally:
                depth[0] -= 1
        return inner

    monkeypatch.setattr(trimesh.proximity, "closest_point", counted(trimesh.proximity.closest_point))
    monkeypatch.setattr(trimesh.proximity, "signed_distance", counted(trimesh.proximity.signed_distance))
    return seen


def in_one_call(monkeypatch):
    """Make `_query` give all the points in one call, as before #429 (the reference answer)."""
    monkeypatch.setattr(ss, "PAIRS", 10 ** 15, raising=False)


def within_budget(seen):
    return all(n == 1 or pairs <= ss.PAIRS for n, pairs in seen)


def test_the_budget_is_the_one_of_the_triangle_distance():
    assert (ss.PAIRS, ss.NEAREST) == (262144, 32)
    assert ss._TriangleDistance.__call__.__defaults__[:2] == (ss.NEAREST, ss.PAIRS)


def test_the_pairs_of_a_point_are_the_faces_that_trimesh_looks_at():
    sphere = trimesh.creation.icosphere(subdivisions=3)
    pts = np.random.default_rng(3).uniform(-1.2, 1.2, (500, 3))
    want = [len(c) for c in trimesh.proximity.nearby_faces(sphere, pts)]
    np.testing.assert_array_equal(ss._query_pairs(sphere, pts), want)


def test_query_takes_runs_of_points_up_to_the_budget_and_joins_the_answers_in_order(monkeypatch):
    p = ss.PAIRS
    pairs = np.array([p // 2, p // 2, p // 3, 2 * p, 1, 1, 1])     # the 4th point alone is above the budget: a call of its own
    monkeypatch.setattr(ss, "_query_pairs", lambda mesh, points: pairs[:len(points)])
    sizes = []

    def three(mesh, points):
        sizes.append(len(points))
        return points, points[:, 1], np.arange(len(points))

    pts = np.random.default_rng(1).normal(size=(7, 3))
    close, dist, idx = ss._query(three, box([0, 0, 0], [1, 1, 1]), pts)
    assert sizes == [2, 1, 1, 3]
    np.testing.assert_array_equal(close, pts)
    np.testing.assert_array_equal(dist, pts[:, 1])
    np.testing.assert_array_equal(idx, [0, 1, 0, 0, 0, 1, 2])
    sizes.clear()
    ss._query(three, box([0, 0, 0], [1, 1, 1]), pts[:2])           # within the budget: the call itself
    ss._query(three, box([0, 0, 0], [1, 1, 1]), pts[:0])
    assert sizes == [2, 0]


def test_the_snap_holds_the_budget_and_moves_each_vertex_as_in_one_call(calls, monkeypatch):
    """24,581 vertices near the faces of the region [-1, 1]^3 (band 0.1 mm), about 10 faces each: on main the snap asked all of them in
    one call. A budget of 16,384 pairs keeps the test small."""
    monkeypatch.setattr(ss, "PAIRS", 16384, raising=False)
    rng = np.random.default_rng(7)
    n = 24581
    side = rng.integers(0, 3, n)
    verts = rng.uniform(-0.9, 0.9, (n, 3))
    verts[np.arange(n), side] = np.where(rng.random(n) < 0.5, -1.0, 1.0) * (1.0 + rng.uniform(-0.15, 0.15, n))
    region = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    new, facts = ss.snap_into_regions(verts, np.zeros((0, 3), np.int64), [region], 0.1)
    assert within_budget(calls) and len(calls) > 10
    assert facts["moved"] > 1000 and facts["of"] == n
    in_one_call(monkeypatch)
    calls.clear()
    ref, ref_facts = ss.snap_into_regions(verts, np.zeros((0, 3), np.int64), [region], 0.1)
    assert max(c[0] for c in calls) > 10000                         # the reference asks all the points at once
    np.testing.assert_array_equal(new, ref)
    assert facts == ref_facts


def test_a_point_with_many_faces_near_it_takes_a_small_call(calls, monkeypatch):
    """Points inside a round region of 1,280 faces, far from its surface: each has hundreds of faces in its box (the snap of k2_bracket had
    such points). A small budget makes the split visible; the answers are those of one call."""
    monkeypatch.setattr(ss, "PAIRS", 4096, raising=False)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    pts = np.random.default_rng(5).uniform(-0.5, 0.5, (300, 3))
    close, dist, tri = ss._query(trimesh.proximity.closest_point, sphere, pts)
    sd = ss._query(trimesh.proximity.signed_distance, sphere, pts)
    assert within_budget(calls) and len(calls) > 20
    in_one_call(monkeypatch)
    ref_close, ref_dist, ref_tri = trimesh.proximity.closest_point(sphere, pts)
    np.testing.assert_array_equal(close, ref_close)
    np.testing.assert_array_equal(dist, ref_dist)
    np.testing.assert_array_equal(tri, ref_tri)
    np.testing.assert_array_equal(sd, trimesh.proximity.signed_distance(sphere, pts))


def test_the_cut_to_the_envelope_holds_the_budget_and_gives_the_same_cut(calls, monkeypatch):
    """A box mesh of 24,578 vertices that goes 2 mm past the face x = 8 of the design space: on main the cut asked all of them in one call.
    A budget of 16,384 pairs keeps the test small."""
    monkeypatch.setattr(ss, "PAIRS", 16384, raising=False)
    slab = box([0, 2, 2], [10, 8, 8])
    for _ in range(6):
        slab = slab.subdivide()
    v, f = np.asarray(slab.vertices, np.float64), np.asarray(slab.faces, np.int64)
    env = ss.make_envelope(box([0, 0, 0], [8, 10, 10]), [], [])
    out_v, out_f, facts = ss.cut_to_envelope(v, f, env)
    assert within_budget(calls) and max(c[0] for c in calls) < len(v)
    assert facts["cut"] is True and facts["vertices_beyond"] > 1000 and facts["largest_excess_mm"] == pytest.approx(2.0)
    in_one_call(monkeypatch)
    calls.clear()
    ref_v, ref_f, ref_facts = ss.cut_to_envelope(v, f, env)
    assert max(c[0] for c in calls) == len(v)
    np.testing.assert_array_equal(out_v, ref_v)
    np.testing.assert_array_equal(out_f, ref_f)
    assert facts == ref_facts
