"""The clean retry of `prepare_surface` (F1-5, CONTRACTS.md section 14).

The defect: `topopt verify` refused a part with "prepared surface has an edge of 0.02 mm (limit 0.05 mm); the weld retry was tried
and did not help". The part had a bore (an exact cut, float64) next to a keep-out socket (a float32 STL of another number of
segments). The boolean of two tessellations of one wall leaves hundreds of edges from 1e-5 to 5e-2 mm. The collapse of the surface
preparation removes them, then the tangential smoothing pulls two vertices together again. The weld of the input cannot help: it
would join long chains of the short edges (a vertex would move by more than the weld radius, or it would make fold edges).

The retry: when the weld of the input did not help, the prepared surface that was refused is cleaned (`clean_short_edges`): the
collapse of the edges under the limit runs again, and the edges that stay are welded. It adds no number: the limit is `min_edge_mm`.
It runs last, so a surface that is prepared today (at the first try, or after the weld of the input) is never touched.

The tests build the cases from a sphere (the smoothing is changed in the test to pull two vertices together, which is what it does
on the real part), and from a block with a bore and a socket (public geometry, the chain of the `post` step, no binary fixture).
The tests need numpy, scipy, numba and manifold3d only, except the one that meshes (`needs_gmsh`).
"""

import json
import math

import manifold3d as m3
import numpy as np
import pytest
import trimesh

from helpers import needs_gmsh, on_the_platform_of_the_measurement
from test_tet_weld import box_with_hole, hair_ring, hair_tube_sphere, short_edges, with_a_hair_tube, write
from topopt import interfaces as I
from topopt import smooth_sdf
from topopt.config import Interface
from topopt.tet import surface as ts

LIMIT = ts.MIN_EDGE_REJECT_MM


# --- the mechanism: the smoothing makes an edge under the limit --------------------------------------------------------------

def sphere(subdivisions=3, radius=5.0):
    sph = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.array(sph.vertices), np.array(sph.faces)


def pull_two_vertices_together(mp, distance=0.03):
    """Change `smooth_surface` of the module: after the real smoothing, move one vertex next to a neighbour (`distance` mm). It is
    the effect that the measured part shows (the smoothing of a surface with edges of a few hundredths of a millimetre moves two
    vertices to less than the limit). The edge is the first edge of the table whose end vertices have six neighbours: the ring around
    the vertex is convex, so no face turns over. Returns the list of the (vertex, neighbour) pairs that the test made."""
    real, made = ts.smooth_surface, []

    def smooth(V, F, *a, **k):
        V = np.array(real(V, F, *a, **k))
        eu, ev, _cnt = ts.edge_table(F, len(V))
        valence = np.bincount(np.concatenate([eu, ev]), minlength=len(V))
        for u, v in zip(eu, ev):
            if valence[u] == 6 and valence[v] == 6:
                d = V[u] - V[v]
                V[u] = V[v] + d * (distance / np.linalg.norm(d))
                made.append((int(u), int(v)))
                break
        return V
    mp.setattr(ts, "smooth_surface", smooth)
    return made


def test_the_clean_retry_prepares_a_surface_that_the_smoothing_left_with_an_edge_under_the_limit(tmp_path, monkeypatch):
    """The input sphere has no edge under the limit, so the weld of the input has nothing to do. The first try is refused for the edge
    that the smoothing made, and the clean retry collapses it: the surface is prepared, and the last step says what it did."""
    V, F = sphere()
    path = write(tmp_path / "sphere.stl", V, F)
    assert short_edges(V, F) == 0
    made = pull_two_vertices_together(monkeypatch)
    prep = ts.prepare_surface(path, 1.0, verbose=False)
    assert made, "the smoothing of the test did not run"
    r = prep.prepared
    assert r["closed_manifold"] and r["edge_min_mm"] >= LIMIT and r["bodies"] == 1 and r["fold_edges"] == 0
    clean = prep.steps[-1]
    assert clean["step"] == "clean_short_edges" and clean["limit_mm"] == LIMIT                # the limit of verify, not a copy of it
    assert (clean["short_edges_before"], clean["short_edges_after_collapse"], clean["edges_welded"]) == (1, 0, 0)
    assert clean["faces_before"] - clean["faces_after"] == 2 and clean["vertices_before"] - clean["vertices_after"] == 1
    assert clean["refused_first"].startswith("prepared surface has an edge of 0.03") and "(limit 0.05 mm)" in clean["refused_first"]
    assert "no edge under 0.05 mm" in clean["weld_refused"]                                       # the weld of the input had nothing to do
    assert abs(clean["volume_change_mm3"]) < 1.0 and clean["deviation_from_refused_mm"]["max_mm"] < 1.0
    assert [s["step"] for s in prep.steps[:3]] == ["split_long_edges", "flip_edges", "collapse_decimate"]   # the steps of the first try stay
    assert "weld_short_edges" not in str(prep.steps) and prep.original["faces"] == len(F)
    assert prep.prepared["deviation_from_input_mm"] == ts.surface_deviation(*ts.load_stl_surface(path), prep.V, prep.F)


def test_without_the_clean_retry_the_refusal_is_the_first_one_and_it_says_why(tmp_path, monkeypatch):
    """The text of the refusal when the clean retry cannot help: the first refusal, the reason of the weld, the reason of the clean."""
    V, F = sphere()
    path = write(tmp_path / "sphere.stl", V, F)
    pull_two_vertices_together(monkeypatch)
    monkeypatch.setattr(ts, "clean_short_edges", lambda *a, **k: (_ for _ in ()).throw(ts._CleanFailed("no clean today")))
    with pytest.raises(ts.SurfaceRejected) as e:
        ts.prepare_surface(path, 1.0, verbose=False)
    text = str(e.value)
    assert text.startswith("prepared surface has an edge of 0.03") and "(limit 0.05 mm); the weld retry was tried and did not help: " in text
    assert text.endswith("; the clean retry of the prepared surface did not help: no clean today")
    assert "no edge under 0.05 mm" in text


def test_a_surface_that_is_prepared_today_never_reaches_the_clean_retry(tmp_path, monkeypatch):
    """The clean retry is the last one. An input that is prepared at the first try, and an input that the weld rescues, give the same
    surface as before, and `clean_short_edges` is not called."""
    def never(*_a, **_k):
        raise AssertionError("the clean retry ran for a surface that is prepared without it")
    monkeypatch.setattr(ts, "clean_short_edges", never, raising=False)           # the same test runs on the code before F1-5
    V, F = sphere(subdivisions=2)
    first = ts.prepare_surface(write(tmp_path / "plain.stl", V, F), 2.0, verbose=False)
    assert all(s["step"] not in ("weld_short_edges", "clean_short_edges") for s in first.steps)
    Vt, Ft = hair_tube_sphere()
    rescued = ts.prepare_surface(write(tmp_path / "tube.stl", Vt, Ft), 1.0, verbose=False)
    assert rescued.steps[0]["step"] == "weld_short_edges" and "clean_short_edges" not in str(rescued.steps)
    assert rescued.prepared["closed_manifold"] and rescued.prepared["edge_min_mm"] >= LIMIT


# --- clean_short_edges alone ---------------------------------------------------------------------------------------------------

def split_edge(V, F, distance):
    """(V, F) of the sphere with the first edge split by a vertex `distance` mm from the first end: one edge under the limit."""
    u, v = int(F[0, 0]), int(F[0, 1])
    m = V[u] + (V[v] - V[u]) * (distance / np.linalg.norm(V[v] - V[u]))
    k = len(V)
    faces = []
    for f in F.tolist():
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            if (f[a], f[b]) in ((u, v), (v, u)):
                faces += [[f[a], k, f[c]], [k, f[b], f[c]]]
                break
        else:
            faces.append(f)
    return np.vstack([V, m]), np.array(faces)


def test_the_collapse_removes_an_edge_that_it_can_and_the_weld_has_nothing_to_do():
    V, F = split_edge(*sphere(subdivisions=2), 0.03)
    assert short_edges(V, F) == 1
    Vc, Fc, facts, rep = ts.clean_short_edges(V, F, LIMIT, 0)
    assert short_edges(Vc, Fc) == 0 and rep["closed_manifold"] and rep["edge_min_mm"] >= LIMIT and rep == ts.surface_report(Vc, Fc)
    assert (facts["short_edges_before"], facts["short_edges_after_collapse"], facts["edges_welded"], facts["vertices_merged"],
            facts["passes"]) == (1, 0, 0, 0, 0)
    assert (facts["vertices_before"], facts["vertices_after"], facts["faces_before"], facts["faces_after"]) == (len(V), len(V) - 1, len(F), len(F) - 2)
    assert facts["volume_change_mm3"] == pytest.approx(facts["volume_after_mm3"] - facts["volume_before_mm3"], abs=1e-9)
    assert facts["deviation_from_refused_mm"]["max_mm"] <= LIMIT                              # a vertex went onto its neighbour


def test_the_edges_that_the_collapse_cannot_remove_are_welded():
    """The tube between two holes of 2e-5 mm: no collapse keeps the link condition, so the edges stay and the weld joins them."""
    V, F = ts._compact(*hair_tube_sphere())
    assert short_edges(V, F) == 6
    Vc, Fc, facts, rep = ts.clean_short_edges(V, F, LIMIT, 3)
    assert short_edges(Vc, Fc) == 0 and rep["closed_manifold"] and rep["bodies"] == 1
    assert facts["short_edges_before"] == 6 and facts["short_edges_after_collapse"] == 6
    assert (facts["edges_welded"], facts["vertices_merged"], facts["passes"]) == (6, 4, 1) and facts["max_vertex_move_mm"] <= LIMIT


@pytest.mark.parametrize("make,max_folds,reason", [
    (lambda: sphere(subdivisions=2), 0, r"^the prepared surface has no edge under 0\.05 mm: there is nothing to clean$"),
    (lambda: hair_ring(), 0, r"^\d+ of \d+ edges under 0\.05 mm stay after the collapse \(shortest [\d.]+ mm\), and the weld of them is "
                             r"refused: the weld leaves no triangle$"),
    (lambda: hair_tube_sphere(), 0, r"^3 of 6 edges under 0\.05 mm stay after the collapse \(shortest [\d.]+ mm\), and the weld of them is "
                                    r"refused: the welded surface is not a closed manifold .*first pinch vertex at "),
    (lambda: hair_tube_sphere(width=(2e-5, 0.3)), 10, r"^\d+ of \d+ edges under 0\.05 mm stay after the collapse \(shortest [\d.]+ mm\), "
                                                  r"and the weld of them is refused: the welded surface is not a closed manifold"),
])
def test_the_clean_retry_says_why_it_cannot_help(make, max_folds, reason):
    V, F = make()
    with pytest.raises(ts._CleanFailed, match=reason):
        ts.clean_short_edges(V, F, LIMIT, max_folds)


def test_a_through_hole_is_coarsened_and_not_filled(tmp_path):
    """A hole of 0.3 mm radius (64 segments, edges of 0.029 mm) is a chain of short edges, but it is a real feature. The weld of
    the input would fill it (see `test_tet_weld.py`). The collapse of the clean retry gives the ring fewer, longer edges and the weld
    has nothing to do: the hole stays (Euler characteristic 0), the volume changes by under 1 % of the hole, the wall moves by under
    0.01 mm."""
    V, F, _path = box_with_hole(tmp_path, 0.3)
    assert short_edges(V, F) == 128
    Vc, Fc, facts, rep = ts.clean_short_edges(V, F, LIMIT, 0)
    assert short_edges(Vc, Fc) == 0 and rep["closed_manifold"] and rep["bodies"] == 1 and rep["vertices"] - rep["edges"] + rep["faces"] == 0   # Euler 0: the hole is there
    assert facts["edges_welded"] == 0 and facts["short_edges_before"] == 128
    assert abs(facts["volume_change_mm3"]) < 0.01 * math.pi * 0.3 ** 2 * 4.0 and facts["deviation_from_refused_mm"]["max_mm"] < 0.01


def test_a_hole_and_a_hair_tube_in_one_file_keep_the_refusal_with_the_reason_of_the_clean_retry(tmp_path):
    """The first try is refused (the tube); the weld of the input is refused (it would fill the hole); the clean retry is refused too
    (the weld of the edges that stay changes the number of bodies). The text has the first refusal and the two reasons."""
    V, F, _path = box_with_hole(tmp_path, 0.3)
    path = with_a_hair_tube(tmp_path, V, F, "hole_and_tube")
    with pytest.raises(ts.SurfaceRejected) as e:
        ts.prepare_surface(path, 1.0, verbose=False)
    text = str(e.value)
    assert "the weld moves a vertex by 0.3 mm, more than the weld radius 0.05 mm; the clean retry of the prepared surface did not help: " in text
    assert "the weld of them is refused: the weld changes the number of bodies (" in text


# --- what the weld of the clean step may change (F1-5 review: MN1, MN2, MN3, N6) ---------------------------------------------

def spike_on_a_flat_top(height=0.2, side=0.04):
    """A cube of 10 mm whose top face has a tiny triangle (two edges of `side` mm, the third 1.4 times longer), with a spike on it:
    the apex is `height` mm above the triangle and has three faces. The edges of the triangle (two are under the limit) cannot
    collapse: the link condition fails (their end vertices have three common neighbours). The weld joins the three corners of the
    triangle; the faces of the apex become faces of no area and go, so the apex is removed. It is the apex of the spikes that the
    plain boolean of `bowl_smooth` leaves (review MN3), made by hand. Returns (V, F); the vertices 8 to 10 are the triangle, 11 the apex."""
    from scipy.spatial import Delaunay
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    V0, F0 = np.array(box.vertices), np.array(box.faces)
    on_top = np.all(V0[F0][:, :, 2] == 5.0, axis=1)
    corners = [i for i in range(len(V0)) if V0[i, 2] == 5.0]
    ia = len(V0)
    V = np.vstack([V0, [(0.0, 0.0, 5.0), (side, 0.0, 5.0), (0.0, side, 5.0)]])
    ids = np.array(corners + [ia, ia + 1, ia + 2])
    top = []
    for t in Delaunay(V[ids][:, :2]).simplices:
        f = ids[t]
        if np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])[2] < 0:
            f = f[[0, 2, 1]]
        top.append(f)
    top = np.array(top)
    tiny = np.flatnonzero(np.all(np.isin(top, [ia, ia + 1, ia + 2]), axis=1))[0]
    a, b, c = top[tiny]
    V = np.vstack([V, V[[a, b, c]].mean(axis=0) + [0.0, 0.0, height]])
    apex = len(V) - 1
    return V, np.vstack([F0[~on_top], np.delete(top, tiny, axis=0), [[a, b, apex], [b, c, apex], [c, a, apex]]])


def test_a_spike_on_short_base_edges_is_removed_and_the_radius_does_not_bound_how_far(tmp_path):
    """Review MN3. The weld counts the move of the vertices that it welds (here 0.03 mm, under the radius 0.05 mm). The apex is on no
    short edge, so it is a group of one that does not move, and all its faces go: it is removed, 0.2 mm from the cleaned surface. The
    facts give that distance (the radius is not a bound on the change of the surface)."""
    V, F = spike_on_a_flat_top()
    assert short_edges(V, F) == 2 and ts.surface_report(V, F)["closed_manifold"] and ts.surface_report(V, F)["fold_edges"] == 0
    Vc, Fc, facts, rep = ts.clean_short_edges(V, F, LIMIT, 0)
    assert rep["closed_manifold"] and short_edges(Vc, Fc) == 0 and len(Vc) == len(V) - 3
    assert (facts["short_edges_before"], facts["short_edges_after_collapse"], facts["edges_welded"], facts["vertices_dropped"]) == (2, 2, 2, 1)
    assert 0 < facts["max_vertex_move_mm"] < LIMIT                                    # the radius holds for the vertices that are welded
    assert facts["removed_vertices"] == 4                                              # the three corners (welded to their mean) and the apex
    assert facts["removed_vertex_distance_max_mm"] == pytest.approx(0.2, abs=1e-9) and facts["removed_vertex_distance_max_mm"] > 3 * LIMIT


def test_the_distance_of_the_removed_vertices_is_exact_and_not_a_sample(tmp_path):
    """`deviation_from_refused_mm` uses a sample of 8,000 vertices. With a second body of 10,000 vertices, and the four removed vertices
    given the ids that the sample leaves out, the sample sees nothing (0.0 mm) and the fact of the removed vertices still has 0.2 mm."""
    V, F = spike_on_a_flat_top()
    big = trimesh.creation.icosphere(subdivisions=5, radius=20.0)
    V2 = np.vstack([V, np.array(big.vertices) + [50.0, 0.0, 0.0]])
    F2 = np.vstack([F, np.array(big.faces) + len(V)])
    sampled = set(np.random.RandomState(0).choice(len(V2), 8000, replace=False).tolist())          # what `surface_deviation` takes
    perm = np.arange(len(V2))
    for old, new in zip((8, 9, 10, 11), [i for i in range(len(V2)) if i not in sampled][:4]):
        perm[old], perm[new] = new, old
    Vp = np.empty_like(V2)
    Vp[perm] = V2
    Vc, Fc, facts, _rep = ts.clean_short_edges(Vp, perm[F2], LIMIT, 0)
    assert facts["deviation_from_refused_mm"]["samples"] == 8000 and facts["deviation_from_refused_mm"]["max_mm"] == 0.0
    assert facts["removed_vertices"] == 4 and facts["removed_vertex_distance_max_mm"] == pytest.approx(0.2, abs=1e-9)


def test_the_weld_gives_the_rows_of_the_faces_that_stay_and_none_of_them_turns_over():
    """MN2. `kept_faces` maps each face of the welded surface to the face it comes from; the normals of the two have a positive dot
    product for every face (a sign test, no limit). A wrong map shows turned faces, so the test can fail."""
    V, F = spike_on_a_flat_top()
    rows = []
    Vw, Fw, _facts = ts.weld_short_edges(V, F, LIMIT, kept_faces=rows)
    assert len(rows[0]) == len(Fw) == len(F) - 6 and len(set(rows[0].tolist())) == len(Fw)
    assert ts._turned_faces(V, F[rows[0]], Vw, Fw) == 0
    assert ts._turned_faces(V, F[rows[0][::-1]], Vw, Fw) > 0
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert ts._turned_faces(a, np.array([[0, 1, 2], [0, 1, 2]]), a, np.array([[0, 1, 2], [0, 2, 1]])) == 1
    assert ts._turned_faces(a, np.array([[0, 1, 2]]), a, np.array([[0, 1, 3]])) == 0                  # a face of no area does not count


def test_the_clean_step_refuses_a_weld_that_turns_a_face_over(monkeypatch):
    """MN2. No natural part turns a face over (the review looked for one). The weld is changed in the test to give the welded surface
    inside out (a closed oriented manifold with no fold edge, so the fold repair has nothing to do): the clean step refuses, and the
    reason gives the count. (A turned face that makes a fold edge is removed by the fold repair of F4-27 before this test.)"""
    V, F = spike_on_a_flat_top()
    real = ts.weld_short_edges

    def turning(V_, F_, limit, report=None, kept_faces=None, check_folds=True):
        Vw, Fw, facts = real(V_, F_, limit, report=report, kept_faces=kept_faces, check_folds=check_folds)
        return Vw, Fw[:, [0, 2, 1]], facts
    with monkeypatch.context() as mp:
        mp.setattr(ts, "weld_short_edges", turning)
        with pytest.raises(ts._CleanFailed, match=rf"^2 of 2 edges under 0\.05 mm stay after the collapse \(shortest 0\.0400 mm\), and the weld "
                                                  rf"of them turns {len(F) - 6} faces over \(the normal of a face that stays has a negative dot product"):
            ts.clean_short_edges(V, F, LIMIT, 0)
    Vc, Fc, _facts, _rep = ts.clean_short_edges(V, F, LIMIT, 0)                          # the same call without the change is accepted
    assert short_edges(Vc, Fc) == 0


def test_the_clean_step_keeps_no_more_fold_edges_than_the_refused_surface_has(tmp_path, monkeypatch):
    """MN1. The input of this part has more fold edges (7) than the refused surface (0). The clean step is given the lower number,
    not the fold edges of the input: it could add 7 and pass before."""
    V, F = sphere()
    path = write(tmp_path / "sphere.stl", V, F)
    pull_two_vertices_together(monkeypatch)
    real_prepare, real_clean, given = ts._prepare_arrays, ts.clean_short_edges, []

    def more_folds_in_the_input(*a, **k):
        try:
            return real_prepare(*a, **k)
        except ts._ShortEdgeRejected as refused:
            refused.original = dict(refused.original, fold_edges=7)
            raise

    def spy(V_, F_, limit, max_folds):
        given.append(max_folds)
        return real_clean(V_, F_, limit, max_folds)
    monkeypatch.setattr(ts, "_prepare_arrays", more_folds_in_the_input)
    monkeypatch.setattr(ts, "clean_short_edges", spy)
    prep = ts.prepare_surface(path, 1.0, verbose=False)
    assert prep.steps[-1]["step"] == "clean_short_edges" and given == [0]


def test_the_clean_step_refuses_a_cleaned_surface_without_volume(monkeypatch):
    """N6. The check of the volume of the clean step. The report of the collapsed surface is changed in the test (the first call of
    `surface_report`): a real collapse of a sphere does not reach this state."""
    V, F = split_edge(*sphere(subdivisions=2), 0.03)
    real, calls = ts.surface_report, []

    def report(V_, F_):
        calls.append(1)
        r = real(V_, F_)
        return dict(r, volume_mm3=0.0) if len(calls) == 1 else r
    monkeypatch.setattr(ts, "surface_report", report)
    with pytest.raises(ts._CleanFailed, match="^the volume of the cleaned surface is not positive$"):
        ts.clean_short_edges(V, F, LIMIT, 0)


# --- F4-27: the fold edges that the collapse and the weld of the clean step make ---------------------------------------------
# The first try of the preparation runs the fold repair after its collapse. Before F4-27 the clean step did not: it refused a
# surface for the fold edges that its own collapse (or weld) had just made. The surfaces of the phone stands of the F-series
# (STL of the SDF mesher, about 5 % of the edges under the limit at the seam of a frozen region) were refused for this reason.

def tube_with_rings():
    """The tube of `hair_tube_sphere` with two rings (12 edges under the limit, 3 fold edges in the tube), as the clean step gets it."""
    return ts._compact(*hair_tube_sphere(subdivisions=1, rings=2))


def test_the_clean_step_repairs_the_fold_edges_that_its_collapse_makes():
    """The collapse of the 12 edges makes 3 fold edges (two faces on top of each other with opposite normals). The refused surface is
    allowed none. The step runs the fold repair of the first try on the collapsed surface and keeps it when no fold edge is left."""
    V, F = tube_with_rings()
    assert short_edges(V, F) == 12
    Vc, Fc, facts, rep = ts.clean_short_edges(V, F, LIMIT, 0)
    assert rep["closed_manifold"] and rep["fold_edges"] == 0 and rep["edge_min_mm"] >= LIMIT and rep["bodies"] == 1 and rep == ts.surface_report(Vc, Fc)
    assert facts["fold_repairs"] == [dict(stage="collapse", fold_edges_made=3, fold_edges_left=0, vertices_removed=2, faces_turned=0)]
    assert (facts["short_edges_before"], facts["short_edges_after_collapse"], facts["edges_welded"]) == (12, 9, 9)    # the weld joins the 9 that stay
    assert facts["max_vertex_move_mm"] <= LIMIT and abs(facts["volume_change_mm3"]) < 1e-3


def test_a_clean_step_without_fold_edges_has_no_fold_repair():
    V, F = split_edge(*sphere(subdivisions=2), 0.03)
    assert ts.clean_short_edges(V, F, LIMIT, 0)[2]["fold_repairs"] == []


def test_the_clean_step_says_how_many_fold_edges_the_repair_leaves(monkeypatch):
    """The reason of a refusal that stays gives the fold edges that the collapse made and the fold edges that the repair leaves."""
    V, F = tube_with_rings()
    monkeypatch.setattr(ts, "_repair_folds", lambda V_, F_, max_fixes=400, verbose=False, rows=None: (F_, 3, 3, np.arange(len(F_))), raising=False)
    with pytest.raises(ts._CleanFailed, match=r"^the collapsed surface has more fold edges \(3\) than the refused surface has \(0\), "
                                              r"and the fold repair leaves 3$"):
        ts.clean_short_edges(V, F, LIMIT, 0)


def test_the_fold_repair_judges_a_collapse_on_the_faces_around_it_and_gets_the_answer_of_the_whole_surface():
    """`_collapse_is_good_local` and the judge on the whole surface (`_collapse_is_good_global`, the one of f86ac04) give the same
    answer for every directed edge of a surface with fold edges. The test checks that both answers occur."""
    V, F = ts._compact(*hair_tube_sphere(subdivisions=1, rings=2))
    ptr, idx = ts._csr_incidence(F, len(V))
    eu, ev, _cnt = ts.edge_table(F, len(V))
    base = ts.fold_edge_count(V, F)
    assert base == 3 and ts.surface_is_closed_manifold(V, F)
    answers = []
    for u, v in zip(eu.tolist(), ev.tolist()):
        for keep, rem in ((u, v), (v, u)):
            F2, _kept = ts._collapse_faces(F, keep, rem)
            answers.append((ts._collapse_is_good_local(V, F, ptr, idx, keep, rem), F2 is not None and ts._collapse_is_good_global(V, F2, base)))
    local, whole = zip(*answers)
    assert local == whole and 0 < sum(whole) < len(whole)


def test_the_fold_repair_tests_the_whole_surface_once_and_not_once_for_each_candidate(monkeypatch):
    """The cost of f6a (959 s): every candidate collapse was judged with a pass over the whole surface. Now one pass judges the start
    (is the surface a closed manifold?) and the faces around the edge judge each candidate. The result is the same as before."""
    V, F = ts._compact(*hair_tube_sphere(subdivisions=2, rings=2))
    calls, real = [], ts.surface_is_closed_manifold
    monkeypatch.setattr(ts, "surface_is_closed_manifold", lambda *a, **k: calls.append(1) or real(*a, **k))
    F2, before, after = ts.repair_folds(V, F)
    assert (before, after, len(F) - len(F2)) == (3, 0, 4) and len(calls) == 1


def test_the_fold_repair_of_a_surface_that_is_not_a_closed_manifold_is_judged_on_the_whole_surface():
    """The local judge needs a closed oriented manifold. An open surface gets the judge of the whole surface, as before F4-27."""
    V, F = ts._compact(*hair_tube_sphere(subdivisions=1, rings=2))
    Fo = F[1:]                                                      # one face less: an open surface
    assert not ts.surface_is_closed_manifold(V, Fo)
    F2, before, after = ts.repair_folds(V, Fo)
    assert (F2 == Fo).all() and after == before > 0                 # no collapse of an open surface is a closed manifold


# --- the part: a bore against a keep-out socket --------------------------------------------------------------------------------

BLOCK = ((0.0, 0.0, 0.0), (24.0, 20.0, 14.0))
BORE_RADIUS, SOCKET_SEGMENTS, SOCKET_EXTRA_RADIUS, SOCKET_TOP = 3.2917, 256, 1e-5, 6.393


def bore_in_a_socket(tmp_path, centre):
    """The STL bytes (as `final.stl`: float32) of a block from the `post` chain of a part with a bore and a keep-out socket.
    The socket is an STL of 256 segments and the radius of the bore plus 1e-5 mm, so its wall is the bore wall within 1e-5 mm. The
    bore is the exact cut of the interface step (a polygon of `I.segments(r)` segments). The chain is the one of `smooth_sdf.build`
    (the block, the exact cut of the keep-out STL, the float32 clean-up) and then `interfaces.apply_step`. Public geometry only."""
    cx, cy = centre
    socket = m3.Manifold.cylinder(SOCKET_TOP + 1.0, BORE_RADIUS + SOCKET_EXTRA_RADIUS, BORE_RADIUS + SOCKET_EXTRA_RADIUS,
                                  SOCKET_SEGMENTS).translate((cx, cy, -1.0)).to_mesh()
    socket_path = tmp_path / "socket.stl"
    trimesh.Trimesh(np.asarray(socket.vert_properties)[:, :3], np.asarray(socket.tri_verts), process=False).export(str(socket_path))
    block = trimesh.creation.box(bounds=list(BLOCK))
    body = smooth_sdf._manifold_of(np.asarray(block.vertices, float).astype(np.float32).astype(np.float64), np.asarray(block.faces, np.int64))
    v, f = smooth_sdf._from_manifold(body - smooth_sdf._load_manifold(socket_path))
    v, f = smooth_sdf._orient(v, f)
    v, f, _info = smooth_sdf._cleanup_float32(v.astype(np.float32).astype(np.float64), f, 1e-5)
    box_ = (tuple(np.asarray(BLOCK[0]) - 3.0), tuple(np.asarray(BLOCK[1]) + 3.0))
    bore = Interface("b", "bore", "cut", 0.0, (cx, cy, 20.0), (0, 0, -1), BORE_RADIUS, None, 30.0)
    ctx = I.Context([bore], tuple(np.asarray(b, float) for b in box_), I.Shapes([bore], box_), False, None, [socket_path])
    v2, f2, res = I.apply_step(ctx, v, f)
    vv, ff = res.variants["as_printed"]
    return trimesh.Trimesh(vv, ff, process=False).export(file_type="stl")


# The first centre is a case that the surface preparation of f3af713 refuses (h 1.0 mm): "prepared surface has an edge of 0.0227 mm
# (limit 0.05 mm); the weld retry was tried and did not help: the welded surface has more fold edges (2) than the input (0)". Near
# it (a centre 0.0013 mm away, 0.09 mm away) the same part is prepared at the first try: the refusal depends on where the float32
# vertices fall, so the other rows are cases that must stay prepared.
@pytest.mark.parametrize("centre,h", [((12.9987, 10.9105), 1.0), ((12.9987, 10.9105), 1.5), ((13.0, 11.0), 1.0)])
def test_a_bore_against_a_socket_of_another_tessellation_is_prepared(tmp_path, centre, h):
    data = bore_in_a_socket(tmp_path, centre)
    path = tmp_path / "part.stl"
    path.write_bytes(data)
    V, F = ts.load_stl_surface(path)
    rep0 = ts.surface_report(V, F)
    assert rep0["closed_manifold"] and rep0["bodies"] == 1
    # The sliver edges of the input. The count is the boolean output of manifold3d 3.5.4 (the pinned version) on Linux x86_64, where it was
    # measured: 436 edges under the limit at the first centre. On macOS (Apple silicon) the output, and so the route of this natural case,
    # can differ (review N7). The test asserts the result below, not the route; `test_the_clean_retry_prepares_a_surface_that_the_smoothing_
    # left_with_an_edge_under_the_limit` pins the route.
    assert short_edges(V, F) > 400 and rep0["edge_min_mm"] < 1e-4
    prep = ts.prepare_surface(str(path), h, verbose=False)
    r = prep.prepared
    assert r["closed_manifold"] and r["bodies"] == 1 and r["edge_min_mm"] >= LIMIT and r["fold_edges"] <= rep0["fold_edges"]
    assert r["volume_mm3"] == pytest.approx(rep0["volume_mm3"], rel=0.01)
    assert r["deviation_from_input_mm"]["max_mm"] < 0.1 * h
    assert [s["step"] for s in prep.steps].count("clean_short_edges") <= 1


def test_the_weld_can_leave_the_check_of_the_fold_edges_to_the_caller(tmp_path):
    """`check_folds=False` (the clean step uses it, then runs the fold repair): the weld does not refuse for the fold edges that it makes. The
    natural case is the bore against a socket of `test_a_bore_against_a_socket_of_another_tessellation_is_prepared`, where the weld
    of the input makes fold edges on the platform of the measurement (two, manifold3d 3.5.4, Linux x86_64)."""
    path = tmp_path / "part.stl"
    path.write_bytes(bore_in_a_socket(tmp_path, (12.9987, 10.9105)))
    V, F = ts.load_stl_surface(path)
    folds0 = ts.surface_report(V, F)["fold_edges"]
    Vw, Fw, _facts = ts.weld_short_edges(V, F, LIMIT, check_folds=False)
    r = ts.surface_report(Vw, Fw)
    assert r["closed_manifold"] and r["bodies"] == 1
    if r["fold_edges"] > folds0:
        with pytest.raises(ts._WeldFailed, match=rf"^the welded surface has more fold edges \({r['fold_edges']}\) than the input \({folds0}\)$"):
            ts.weld_short_edges(V, F, LIMIT)
    else:
        assert ts.weld_short_edges(V, F, LIMIT)[2]["edges_welded"] > 0


# --- the seam of the SDF mesher against the frozen regions (F4-27) -----------------------------------------------------------------
# `smooth_sdf.make_mesh_sdf` unites the smooth surface with the exact STL of a `fixed` or `load` region and cuts a `keep_out` region. The
# smooth surface lies a few micrometres off the exact face, so the exact boolean leaves a strip of steps and slivers along the seam: on
# the phone stands of the F-series about 5 % of the edges are under 0.05 mm, 98 % of them within 0.02 mm of a region face. Here a small
# part makes the same seam: a plate of one voxel (`fixed`), a shelf (`load`), a slot (`keep_out`) and two struts. The field is made by
# hand (no optimizer), the mesh is the real mesher. 2.8 % of its edges are under the limit.

SEAM_H = 0.7                                                              # the voxel of the field, mm
SEAM_STRUTS = [((9.0, 4.3, 0.7), (2.0, 4.5, 8.6), 1.07), ((4.2, 2.7, 0.7), (6.4, 4.3, 8.6), 1.49)]   # (from, to, radius), mm


def seam_inputs(folder, tilt_deg=0.0):
    """Write the region STLs of the part above into `folder`. Return `(rho, labels, regions)`: the field and the keywords of `make_mesh_sdf`.

    `tilt_deg` turns the shelf (the `load` seat) about the y axis through its centre: a tilted face, as on a sloped seat. The faces of a
    box are axis-aligned, so their corners are float32-exact; a tilted face is not, and it is where the noise of the exact boolean is
    worst (F4-27b). The part is then 1.4 mm higher and the struts end on the lower face of the shelf.
    """
    from topopt import post
    top = 10.0 if not tilt_deg else 11.4
    boxes = {"plate": ((0, 0, 0), (14, 6, 0.7)), "seat": ((0, 0, 8.6), (10, 6, 10)), "slot": ((2, 1.5, 0.7), (5, 4.5, 6)),
             "design": ((0, 0, 0), (14, 6, top))}
    turn = trimesh.transformations.rotation_matrix(np.radians(tilt_deg), (0, 1, 0), (5.0, 3.0, 9.3))
    for name, (lo, hi) in boxes.items():
        lo, hi = np.array(lo, float), np.array(hi, float)
        solid = trimesh.creation.box(extents=hi - lo)
        solid.apply_translation((lo + hi) / 2)
        if name == "seat" and tilt_deg:
            solid.apply_transform(turn)
        solid.export(str(folder / f"{name}.stl"))
    shape = (21, 10, 16 if not tilt_deg else 17)
    centres = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) * SEAM_H for n in shape], indexing="ij"), -1)
    labels = np.zeros(shape, np.uint8)
    for name, tag in (("design", post.DESIGN), ("plate", post.FIXED), ("seat", post.LOAD), ("slot", post.KEEP_OUT)):
        lo, hi = boxes[name]
        at = centres
        if name == "seat" and tilt_deg:   # the frame of the box: undo the turn
            back = np.linalg.inv(turn)
            at = centres @ back[:3, :3].T + back[:3, 3]
        labels[((at >= lo) & (at < hi)).all(-1)] = tag
    struts = SEAM_STRUTS
    if tilt_deg:
        under = lambda x: 9.3 - (x - 5.0) * np.tan(np.radians(tilt_deg)) - 0.7 / np.cos(np.radians(tilt_deg))
        struts = [(a, (b[0], b[1], float(under(b[0]))), r) for a, b, r in SEAM_STRUTS]
    dist = np.full(shape, np.inf)
    for a, b, radius in struts:
        a, b = np.array(a), np.array(b)
        t = np.clip(((centres - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
        dist = np.minimum(dist, np.linalg.norm(centres - (a + t[..., None] * (b - a)), axis=-1) - radius)
    rho = post.build_field(np.clip(0.5 - dist / SEAM_H, 0.0, 1.0).astype(np.float32), labels)
    regions = dict(fixed_stls=[folder / "plate.stl"], load_stls=[folder / "seat.stl"], keep_out_stls=[folder / "slot.stl"])
    return rho, labels, regions


def no_snap(verts, faces, surfaces, band_mm):
    """`snap_into_regions` that moves nothing: the seam as it was before F4-27b."""
    return np.asarray(verts, np.float64), {"moved": 0}


def seam_mesh(folder, snap=True, tilt_deg=0.0):
    """`make_mesh_sdf` of the seam part. Return `(verts, faces, info)`. `snap=False` leaves the vertices where the marching cubes put them."""
    from topopt import post
    rho, labels, regions = seam_inputs(folder, tilt_deg)
    with pytest.MonkeyPatch.context() as mp:
        if not snap:   # raising=False: a checkout from before F4-27b has no such function, and its seam is the one to make
            mp.setattr(smooth_sdf, "snap_into_regions", no_snap, raising=False)
        return smooth_sdf.make_mesh_sdf(rho, labels, np.zeros(3), SEAM_H, up=post.UPSAMPLE, pad=post.PAD, **regions)


def seam_part(folder, snap=True, tilt_deg=0.0):
    """The path of the STL (float32, as `final.stl`) of the part above: 14 x 6 x 10 mm, plate 0.7 mm thick, shelf 1.4 mm thick.

    `snap=False` makes the part with the seam as it was before F4-27b (see `seam_mesh`). The tests of the route through the clean retry
    (the weld and the fold repair after it) need that seam.
    """
    v, f, _info = seam_mesh(folder, snap, tilt_deg)
    path = folder / "seam_part.stl"
    path.write_bytes(trimesh.Trimesh(v, f, process=False).export(file_type="stl"))
    return path


@pytest.fixture(scope="module")
def seam_stl(tmp_path_factory):
    return seam_part(tmp_path_factory.mktemp("seam"), snap=False)


def test_a_part_with_the_seam_of_the_sdf_mesher_is_prepared_at_a_size_where_the_fold_repair_is_needed(seam_stl):
    """At 0.3 mm the first try and the weld of the input are refused; the clean retry welds the one edge that stays, and that weld makes
    a fold edge (two faces on top of each other along the seam). Before F4-27 the clean retry refused the part for this fold edge
    ("the welded surface has more fold edges (1) than the surface before the weld ...", `verify` could not mesh the part). It now runs the fold
    repair of the first try on the welded surface. The route is the one measured with manifold3d 3.5.4 on Linux x86_64; on another platform the
    boolean output can differ, and the test asserts the result only (as `test_a_bore_against_a_socket_...`)."""
    V, F = ts.load_stl_surface(seam_stl)
    rep0 = ts.surface_report(V, F)
    assert rep0["closed_manifold"] and rep0["bodies"] == 1 and short_edges(V, F) > 0.02 * (3 * len(F) // 2)     # a seam: over 2 % of the edges
    prep = ts.prepare_surface(str(seam_stl), 0.3, verbose=False)
    r = prep.prepared
    assert r["closed_manifold"] and r["bodies"] == 1 and r["edge_min_mm"] >= LIMIT and r["fold_edges"] <= rep0["fold_edges"]
    assert r["volume_mm3"] == pytest.approx(rep0["volume_mm3"], rel=1e-3) and r["deviation_from_input_mm"]["max_mm"] < 0.3
    if on_the_platform_of_the_measurement():
        clean = prep.steps[-1]
        assert clean["step"] == "clean_short_edges" and "the welded surface has more fold edges" in clean["weld_refused"]
        assert [(x["stage"], x["fold_edges_made"], x["fold_edges_left"]) for x in clean["fold_repairs"]] == [("weld", 1, 0)]
        assert [x["faces_turned"] for x in clean["fold_repairs"]] == [1]       # the weld turned a face and the repair turned it back
        assert clean["limit_mm"] == LIMIT and abs(clean["volume_change_mm3"]) < 0.01 and clean["max_vertex_move_mm"] <= LIMIT


@needs_gmsh
def test_gmsh_meshes_the_part_with_the_seam_and_the_facts_have_the_fold_repair(seam_stl):
    from topopt.tet import mesh_stl
    mesh = mesh_stl(seam_stl, size=0.3)
    assert mesh.facts["volume_ratio_stl"] == pytest.approx(1.0, rel=0.01) and mesh.facts["quality_min"] > 0.01
    if on_the_platform_of_the_measurement():
        assert [x["stage"] for x in mesh.facts["surface_clean_retry"]["fold_repairs"]] == ["weld"]


# --- the snap of the vertices at the region faces (F4-27b) -------------------------------------------------------------------
# `smooth_sdf.snap_into_regions` moves the vertices that lie just outside a region face into the region, so that the exact boolean removes
# the sheet and leaves no strip of micro-steps. The part above has axis-aligned faces only. A tilted face is not float32-exact, so the noise of
# the exact boolean is worst there. Here the shelf is tilted by 8 degrees: without the snap 94 edges lie under 1e-3 mm, the shortest 5e-7 mm.
# The part without the snap is made with `no_snap` (the identity).

TILT = 8.0   # degrees


def seam_run(tmp_path_factory, name, snap, tilt_deg):
    folder = tmp_path_factory.mktemp(name)
    v, f, info = seam_mesh(folder, snap, tilt_deg)
    path = folder / "seam_part.stl"
    path.write_bytes(trimesh.Trimesh(v, f, process=False).export(file_type="stl"))
    return path, info


@pytest.fixture(scope="module")
def tilted_snap(tmp_path_factory):
    return seam_run(tmp_path_factory, "tilted_snap", True, TILT)


@pytest.fixture(scope="module")
def tilted_plain(tmp_path_factory):
    return seam_run(tmp_path_factory, "tilted_plain", False, TILT)


@pytest.fixture(scope="module")
def flat_snap(tmp_path_factory):
    return seam_run(tmp_path_factory, "flat_snap", True, 0.0)


def edge_lengths(path):
    V, F = ts.load_stl_surface(str(path))
    eu, ev, _cnt = ts.edge_table(F, len(V))
    return np.linalg.norm(V[eu] - V[ev], axis=1)


def test_a_tilted_seat_leaves_a_strip_of_slivers_without_the_snap_and_almost_none_with_it(tilted_snap, tilted_plain):
    """Measured with manifold3d 3.5.4 on Linux x86_64: without the snap 733 edges under 0.05 mm, 94 of them under 1e-3 mm (the shortest
    5e-7 mm); with it 109 and 0. The edges under 0.05 mm that stay lie on the exact face (the retriangulation along the union line)."""
    plain, snapped = edge_lengths(tilted_plain[0]), edge_lengths(tilted_snap[0])
    assert (plain < 1e-3).sum() >= 50, "the part does not show the seam of the exact boolean: the test would prove nothing"
    assert (snapped < 1e-3).sum() <= 0.05 * (plain < 1e-3).sum()
    assert (snapped < 0.05).sum() < (plain < 0.05).sum() / 3
    assert tilted_snap[1]["snap"]["moved"] > 0


@pytest.mark.parametrize("which", ["flat_snap", "tilted_snap"])
def test_the_part_with_the_snap_is_prepared_at_the_first_try_without_fold_edges_or_crossings(which, request):
    """Without the snap the part with the flat shelf has a fold edge and the surface preparation needs the clean retry at 0.3 mm (see the
    tests above). With it the first try prepares the surface. The route is the one measured on Linux x86_64; on another platform the boolean
    output can differ, and the test asserts the result only."""
    from topopt.tet import intersect
    path, info = request.getfixturevalue(which)
    V, F = ts.load_stl_surface(str(path))
    rep = ts.surface_report(V, F)
    assert rep["closed_manifold"] and rep["bodies"] == 1 and rep["fold_edges"] == 0 and intersect.find_crossings(V, F).total == 0
    prep = ts.prepare_surface(str(path), 0.3, verbose=False)
    r = prep.prepared
    assert r["closed_manifold"] and r["bodies"] == 1 and r["edge_min_mm"] >= LIMIT and r["fold_edges"] == 0
    if on_the_platform_of_the_measurement():
        assert [x["step"] for x in prep.steps][-1] == "repair_folds"
    assert info["snap"]["moved"] > 0


def test_the_region_faces_stay_exact_after_the_snap_moved_the_vertices(tilted_snap):
    """The plate and the tilted seat are held (nothing of them is missing), the slot is empty: exact booleans on the STL as written."""
    path, info = tilted_snap
    assert info["snap"]["moved"] > 0
    V, F = ts.load_stl_surface(str(path))
    folder = path.parent
    res = smooth_sdf.region_checks(V, F, {"fixed:plate": folder / "plate.stl", "load:seat": folder / "seat.stl"}, {"slot": folder / "slot.stl"})
    for name, fact in res["fixed"].items():
        assert fact["uncovered_mm3"] < 1e-3 and fact["max_outside_mm"] < 1e-3 and fact["region_mm3"] > 10, (name, fact)
    assert res["keep_out"]["slot"]["intersect_mm3"] < 1e-3


def test_the_snap_keeps_the_volume_within_the_calibration_and_the_shape_close_to_the_part_without_it(tilted_snap, tilted_plain):
    """The move takes material at the region faces, and the global level shift of the calibration pays it back (CONTRACTS.md section 19,
    step 7). The part with the snap has the volume that the calibration meets, and the volume of the part without it within 0.1 %."""
    (path1, info1), (path0, info0) = tilted_snap, tilted_plain
    assert abs(info1["calibration"][-1]["error"]) < 5e-4
    v1, v0 = ts.surface_report(*ts.load_stl_surface(str(path1)))["volume_mm3"], ts.surface_report(*ts.load_stl_surface(str(path0)))["volume_mm3"]
    assert v1 == pytest.approx(v0, rel=1e-3)
    assert info1["snap"]["moved"] > 0
    assert abs(info1["snap"]["volume_change_mm3"]) < 0.02 * v1     # at one level shift the move takes a little material at the faces
    assert info1["snap"]["volume_change_mm3"] < 0                  # the sheet outside the faces goes; nothing is added


def test_the_facts_of_the_snap_are_the_counts_and_the_band_and_the_largest_move(tilted_snap):
    from topopt import post
    snap = tilted_snap[1]["snap"]
    band = smooth_sdf.SdfParams().pin_tol * SEAM_H / post.UPSAMPLE          # the band that pins vertices in the Taubin step: no new number
    assert snap["band_mm"] == pytest.approx(band) and snap["depth_mm"] == pytest.approx(band)
    assert 0 < snap["moved"] <= snap["in_band"] <= snap["of"]
    assert 0 < snap["largest_move_mm"] <= 2 * band * (1 + 1e-9)             # a vertex in the band goes to one band deep: at most 2 bands
    assert snap["turned_faces"] >= 0 and snap["halving_rounds"] >= 0 and snap["target_outside"] == 0   # no region of this part is thinner than a band
    assert (snap["turned_faces"] > 0) == (snap["halving_rounds"] > 0)
    assert json.loads(json.dumps(snap)) == snap                             # the fact goes to metrics.json
    assert smooth_sdf.SdfParams().pin_tol == 0.35                           # the band is the one of the Taubin pins


# --- who turns a face over: the weld or the fold repair after it (F4-27 review, MJ1; MN2) -----------------------------------------
# On the f7c part (attempt 2) the refusal said "the weld of them turns 1 face over". The weld moves a vertex by 0.05 mm at most. The vertex
# that turned the face had moved 0.19 mm: it was the collapse of the fold repair after the weld. The reason now names the stage.

def test_the_turned_faces_are_counted_for_the_stage_that_turns_them():
    """`_turned_faces_by_stage` gives (turned by the first stage, turned by the second) for the faces that are turned in the end. Four
    triangles, one for each case: turned by the first stage and kept; turned by the second only; turned by the first and back by the second
    (not turned in the end); never turned."""
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    turned = tri[[0, 2, 1]]                                           # the same corners in the other order: the normal points the other way
    F = np.arange(12).reshape(4, 3)
    V0 = np.vstack([tri] * 4)
    V1 = np.vstack([turned, tri, turned, tri])
    V2 = np.vstack([turned, turned, tri, tri])
    assert ts._turned_faces(V0, F, V2, F) == 2 and ts._turned_faces_by_stage(V0, F, V1, F, V2, F) == (1, 1)
    assert ts._turned_faces_by_stage(V0, F, V0, F, V0, F) == (0, 0)
    assert ts._turn_reason([(0, "the weld of them"), (2, "the fold repair after the weld")]) == (
        "the fold repair after the weld turns 2 faces over (the normal of a face that stays has a negative dot product with its normal before the weld)")
    assert ts._turn_reason([(1, "the weld of them"), (1, "the fold repair after the weld")]).startswith(
        "the weld of them turns 1 face over and the fold repair after the weld turns 1 face over (")


def prepare_with_the_weld_stage_repair_changed(seam_stl, change):
    """`prepare_surface` of the seam part at 0.3 mm, where the clean step runs the fold repair after its weld (the route measured on Linux
    x86_64). `change(V, F, report, stays)` returns the (V, F, report, stays) that the step gives back, for the stage `weld` only."""
    real = ts._repair_folds_step

    def step(V, F, stage, repairs):
        out = real(V, F, stage, repairs)
        return change(*out, repairs) if stage == "weld" else out
    mp = pytest.MonkeyPatch()
    mp.setattr(ts, "_repair_folds_step", step)
    try:
        return ts.prepare_surface(str(seam_stl), 0.3, verbose=False)
    finally:
        mp.undo()


def test_a_face_that_the_fold_repair_after_the_weld_turns_over_is_refused_with_the_repair_as_the_source(seam_stl):
    """MJ1. The repair of the weld stage is changed in the test so that one vertex of the largest face moves to the other side of the
    opposite edge (it turns the face, and the faces around the vertex). The report is the one of the real repair (the move is the thing
    under test, and a turned face in a flat part would also make a fold edge). The weld did not turn these faces, so the reason must not
    name the weld. Before the change it said "the weld of them turns N faces over"."""
    if not on_the_platform_of_the_measurement():
        pytest.skip("the route of the seam part (a fold repair after the weld) was measured on Linux x86_64")

    def turn_the_largest_face(V, F, rep, stays, repairs):
        i = int(np.argmax(np.linalg.norm(ts._face_normals(V, F), axis=1)))
        a, b, c = F[i]
        V = V.copy()
        V[a] = V[b] + V[c] - V[a]                                     # the point reflection of a through the middle of the edge b-c
        return V, F, rep, stays
    with pytest.raises(ts.SurfaceRejected, match=r"the clean retry of the prepared surface did not help: \d+ of \d+ edges under 0\.05 mm stay "
                                                 r"after the collapse \(shortest [\d.]+ mm\), and the fold repair after the weld turns \d+ faces? over "
                                                 r"\(the normal of a face that stays has a negative dot product with its normal before the weld\)$") as e:
        prepare_with_the_weld_stage_repair_changed(seam_stl, turn_the_largest_face)
    assert "the weld of them turns" not in str(e.value)


def test_a_fold_repair_after_the_weld_that_leaves_a_fold_edge_is_refused_with_the_count(seam_stl):
    """MN2 (a). The check after the repair of the weld stage (M9 of the review survived without a test). The repair is changed to leave the
    one fold edge that the weld made: the reason gives the fold edges before and after the repair."""
    if not on_the_platform_of_the_measurement():
        pytest.skip("the route of the seam part (a fold repair after the weld) was measured on Linux x86_64")

    def leave_the_fold(V, F, rep, stays, repairs):
        return V, F, dict(rep, fold_edges=1), stays
    with pytest.raises(ts.SurfaceRejected, match=r"and the weld of them is refused: the welded surface has more fold edges \(1\) than the "
                                                 r"surface before the weld \(0\), and the fold repair leaves 1$"):
        prepare_with_the_weld_stage_repair_changed(seam_stl, leave_the_fold)


def test_a_fold_repair_that_leaves_an_edge_under_the_limit_is_refused(seam_stl):
    """MN2 (b). The check of the shortest edge after a fold repair (M4 of the review survived without a test). The report of the repair
    is changed to give an edge of 0.01 mm (a repair collapse moves a vertex onto a neighbour, and the new edges of the faces around it
    can be short)."""
    if not on_the_platform_of_the_measurement():
        pytest.skip("the route of the seam part (a fold repair after the weld) was measured on Linux x86_64")

    def short_edge(V, F, rep, stays, repairs):
        return V, F, dict(rep, edge_min_mm=0.01), stays
    with pytest.raises(ts.SurfaceRejected, match=r"the clean retry of the prepared surface did not help: the fold repair leaves an edge of "
                                                 r"0\.0100 mm, under 0\.05 mm$"):
        prepare_with_the_weld_stage_repair_changed(seam_stl, short_edge)


@needs_gmsh
def test_the_mesh_facts_name_the_clean_retry_only_when_it_ran(tmp_path, monkeypatch):
    from topopt.tet import mesh_stl
    V, F = sphere()
    path = write(tmp_path / "sphere.stl", V, F)
    plain = mesh_stl(path, size=1.0)
    assert "surface_clean_retry" not in plain.facts and "surface_weld_retry" not in plain.facts
    pull_two_vertices_together(monkeypatch)
    mesh = mesh_stl(path, size=1.0)
    clean = mesh.facts["surface_clean_retry"]
    assert clean["step"] == "clean_short_edges" and clean["short_edges_before"] == 1 and "surface_weld_retry" not in mesh.facts
    assert mesh.facts["volume_ratio_stl"] == pytest.approx(1.0, rel=0.05) and mesh.facts["quality_min"] > 0.05


# --- the facts reach the text, the compact facts of verify.json and the ratchet row -----------------------------------------


def a_clean_step():
    V, F = split_edge(*sphere(subdivisions=2), 0.03)
    return ts.clean_short_edges(V, F, LIMIT, 0)[2]


def test_the_clean_line_says_what_the_cleaning_did():
    from topopt.verify.report import clean_line
    line = clean_line(a_clean_step())
    assert line.startswith("surface: the prepared surface was refused for an edge under 0.05 mm and the weld of the input did not help, ")
    assert "so its 1 edge under the limit was cleaned: 1 collapsed, 2 faces removed, 1 vertex removed (the farthest is 0." in line     # N1: singular
    assert "welded" not in line and "merged" not in line                  # N1: no edge was welded, so there is no weld part
    assert "mm from the cleaned surface)" in line and "volume change " in line and "\n" not in line


def test_the_clean_line_with_a_weld_gives_the_exact_distance_of_the_removed_vertices():
    """The numbers of the plain-boolean `bowl_smooth` at 1.2 mm (review MN3): the weld moved a vertex by 0.0419 mm, and a vertex that it
    removed was 0.1777 mm from the cleaned surface. The radius does not bound the second number, so the line gives it."""
    from topopt.verify.report import clean_line
    c = dict(a_clean_step(), short_edges_before=9, short_edges_after_collapse=3, edges_welded=3, vertices_merged=3, max_vertex_move_mm=0.0419,
             faces_before=2134, faces_after=2114, removed_vertices=12, removed_vertex_distance_max_mm=0.17768, volume_change_mm3=-0.00234)
    assert clean_line(c).endswith("so its 9 edges under the limit were cleaned: 6 collapsed, 3 welded (3 vertices merged, largest move 0.0419 mm), "
                                  "20 faces removed, 12 vertices removed (the farthest is 0.1777 mm from the cleaned surface), "
                                  "volume change -0.0023 mm3")
    old = {k: v for k, v in c.items() if k not in ("removed_vertices", "removed_vertex_distance_max_mm")}       # a verify.json of an earlier build
    assert "removed (the farthest" not in clean_line(old) and "vertices merged" in clean_line(old)


def test_the_clean_line_says_what_the_fold_repair_of_the_cleaning_did():
    """F4-27. The collapse (or the weld) of the cleaning made fold edges and the fold repair removed them: the line says how many. A step
    without the fact (`verify.json` of an earlier build) and a step with no repair have no such part."""
    from topopt.verify.report import clean_line
    c = dict(a_clean_step(), fold_repairs=[dict(stage="collapse", fold_edges_made=3, fold_edges_left=0, vertices_removed=2, faces_turned=0),
                                           dict(stage="weld", fold_edges_made=2, fold_edges_left=1, vertices_removed=1, faces_turned=0)])
    assert "2 faces removed, 4 fold edges repaired, 1 vertex removed (the farthest" in clean_line(c)
    assert "made by the cleaning" not in clean_line(c)               # review MN3: the first number is the fold edges before the repair, whoever made them
    one = dict(c, fold_repairs=[dict(stage="weld", fold_edges_made=2, fold_edges_left=1, vertices_removed=1, faces_turned=0)])
    assert "2 faces removed, 1 fold edge repaired, " in clean_line(one)
    turned = dict(c, fold_repairs=[dict(c["fold_repairs"][0], faces_turned=3), dict(c["fold_repairs"][1], faces_turned=1)])
    assert "4 fold edges repaired (the repair turned 4 faces over), " in clean_line(turned)
    assert "turned" not in clean_line(c) and "fold" not in clean_line(a_clean_step())
    assert "fold" not in clean_line({k: v for k, v in c.items() if k != "fold_repairs"})
    old = dict(c, fold_repairs=[{k: v for k, v in r.items() if k != "faces_turned"} for r in c["fold_repairs"]])        # verify.json of this build before MN3
    assert "4 fold edges repaired, " in clean_line(old)


def test_the_summary_of_a_verify_run_has_the_clean_line_and_only_then():
    from topopt.verify.report import clean_line, summary_lines
    clean = a_clean_step()
    res = {"run": {"name": "x"}, "size_mm": 1.0, "backend": {"solver": "none"}, "band_mm": 0.5, "cases": {}, "regions": {"load": []},
           "timing_s": {"verify_total_s": 1.0}, "memory": {"peak_rss_bytes": 1, "estimate_after_meshing_bytes": 1},
           "mesh": {"tets": 1, "dofs": 3, "quality_min": 0.3, "below_0p05": 0, "p2_volume_over_stl": 1.0}}
    assert clean_line(clean) not in summary_lines(res)
    res["mesh"]["surface_clean_retry"] = clean
    lines = summary_lines(res)
    assert lines.count(clean_line(clean)) == 1 and lines[2] == clean_line(clean)             # after the two lines of the mesh


def test_verify_json_keeps_the_clean_fact_in_the_compact_mesh_facts():
    import json
    from topopt.verify.run import _compact_mesh_facts
    clean = a_clean_step()
    assert json.loads(json.dumps(clean)) == clean                                  # verify.json can hold the step
    assert _compact_mesh_facts({"tets": 5, "surface_clean_retry": clean, "other": 1}) == {"tets": 5, "surface_clean_retry": clean}


def test_the_ratchet_row_has_a_note_for_the_clean_retry_and_no_other_fact_changes():
    from topopt import ratchet as rt
    from topopt.verify.report import clean_line
    case = {"stiffness_N_per_mm": 10.0, "compliance_Nmm": 1.0, "max_displacement_mm": 0.1,
            "von_mises": {"percentile": 99.9, "percentile_by_volume_MPa": 5.0, "peak_qp_MPa": 8.0}}
    report = {"size_mm": 1.0, "cases": {"a": case}, "mesh": {"tets": 10}}
    plain = rt.verify_facts(report)
    clean = a_clean_step()
    got = rt.verify_facts(dict(report, mesh={"tets": 10, "surface_clean_retry": clean}))
    assert got["notes"] == [clean_line(clean)] and got["sources"]["notes"] == "verify.json:mesh.surface_clean_retry"
    assert {k: v for k, v in got.items() if k not in ("notes", "sources")} == {k: v for k, v in plain.items() if k != "sources"}


# --- gmsh fails on a surface that a retry made (F1-5 review, MJ1) ----------------------------------------------------------
# The error must say that the surface is not the one of the first try, and the retries of the mesher must stop when the refinement
# does not change the surface. gmsh and the surface preparation are faked here (the fake gives the step of the retry); the real
# chain is in `test_apply_marks_seam.py` (the plain-boolean `bowl_smooth`).

REFUSED_FIRST = "prepared surface has an edge of 0.0033 mm (limit 0.05 mm)"
CLEAN_STEP = dict(a_clean_step(), refused_first=REFUSED_FIRST,
                  weld_refused="the weld moves a vertex by 0.0818555 mm, more than the weld radius 0.05 mm")
WELD_STEP = dict(step="weld_short_edges", limit_mm=0.05, short_edges_in_input=6, edges_welded=6, refused_first=REFUSED_FIRST)


def clean_sentence(where=""):
    return (f"The surface that gmsh got{where} is the prepared surface that was refused ({REFUSED_FIRST}), with its 1 edge under "
            "0.05 mm cleaned (clean retry)")


def fake_preparation(monkeypatch, surfaces, steps):
    """`prepare_surface` that returns the given (V, F) one after the other (the last again and again), with the steps of a retry.
    Returns the list of the `size_field` of each call."""
    fields = []

    def fake(path, h, **kw):
        fields.append(kw.get("size_field"))
        V, F = surfaces[min(len(fields) - 1, len(surfaces) - 1)]
        return ts.PreparedSurface(V, F, {}, {"deviation_from_input_mm": {"mean_mm": 0.0, "p99_mm": 0.0, "max_mm": 0.0, "samples": 0}},
                                  [dict(s) for s in steps])
    monkeypatch.setattr(ts, "prepare_surface", fake)
    return fields


def gmsh_fails(monkeypatch, points=((0.0, 0.0, 10.0),), message="algorithm 10: HXT 3D mesh failed"):
    from test_tet import mesher
    calls = []

    def fake(V, F, size, **kw):
        calls.append(len(F))
        raise mesher.MeshingFailed(message, list(points))
    monkeypatch.setattr(mesher, "mesh_surface", fake)
    return calls


def a_sphere_stl(tmp_path):
    from test_tet import _sphere_stl
    m = _sphere_stl(tmp_path / "sphere.stl", subdiv=2)
    return tmp_path / "sphere.stl", (np.array(m.vertices), np.array(m.faces))


def moved(surface, dx):
    return surface[0] + np.array([dx, 0.0, 0.0]), surface[1]


def test_the_error_after_every_attempt_names_the_clean_retry_and_the_first_refusal(monkeypatch, tmp_path):
    """Before the fix the text was the errors of gmsh and the crossing counts: the reader could not see that the surface was the
    refused surface with its short edges cleaned."""
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    fake_preparation(monkeypatch, [moved(surface, 0.0), moved(surface, 0.5)], [CLEAN_STEP])       # another surface in each attempt
    gmsh_fails(monkeypatch)
    with pytest.raises(MeshingFailed, match=r"every surface attempt \(2\)") as e:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=2)
    text = str(e.value)
    assert text.endswith(". " + clean_sentence(" in attempt 2"))
    assert "Crossings found in the prepared surface in each attempt: 0, 0. " in text
    assert e.value.crossings["surface_retry"] == CLEAN_STEP and e.value.crossings["prepared"]["attempts"] == [0, 0]
    assert "stopped" not in e.value.crossings and json.loads(json.dumps(e.value.crossings)) == e.value.crossings


def test_the_error_without_a_retry_has_no_sentence_and_no_retry_fact(monkeypatch, tmp_path):
    """A surface from the first try: the text and the facts are the ones of P4."""
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    fake_preparation(monkeypatch, [moved(surface, 0.0), moved(surface, 0.5)], [])
    gmsh_fails(monkeypatch)
    with pytest.raises(MeshingFailed, match=r"every surface attempt \(2\)") as e:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=2)
    assert "surface that gmsh got" not in str(e.value) and "retry" not in str(e.value) and "surface_retry" not in e.value.crossings
    gmsh_fails(monkeypatch, points=(), message="algorithm 10: nope")                     # the plain re-raise: the error of gmsh as it is
    with pytest.raises(MeshingFailed) as e:
        mesh_stl(stl, isolate=False, size=3.0)
    assert str(e.value) == "algorithm 10: nope" and e.value.crossings is None


# --- a tet of zero volume is a failed attempt (F4-27) -------------------------------------------------------------------------------
# f6a at 0.973 mm: gmsh meshed the cleaned surface (49,072 tets) and one tet had four nodes in one plane. `mesh_stl` gave the mesh
# back, and `topopt verify` stopped with "internal error: Singular matrix" (the gradient matrix of the tet has no inverse).

GOOD_TET = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
FLAT_TET = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]


def gmsh_gives(monkeypatch, tets_of_each_call):
    """`mesh_surface` that returns, for the call k, one tet for each list of 4 points of `tets_of_each_call[k]` (the last list again and
    again). Returns the list of the numbers of faces of the calls."""
    from test_tet import mesher
    calls = []

    def fake(V, F, size, **kw):
        calls.append(len(F))
        tets = tets_of_each_call[min(len(calls) - 1, len(tets_of_each_call) - 1)]
        p = np.array([q for tet in tets for q in tet], float).T
        t = np.arange(4 * len(tets)).reshape(len(tets), 4).T
        return p, t, dict(volume_ratio=1.0, algorithm=10, seconds=0.0)
    monkeypatch.setattr(mesher, "mesh_surface", fake)
    return calls


def test_a_mesh_with_a_tet_of_zero_volume_is_a_failed_attempt_and_the_next_attempt_refines_there(monkeypatch, tmp_path):
    from test_tet import _ok_probe
    from topopt.tet import mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    fields = fake_preparation(monkeypatch, [moved(surface, 0.0), moved(surface, 0.5)], [])
    calls = gmsh_gives(monkeypatch, [[GOOD_TET, FLAT_TET], [GOOD_TET]])
    mesh = mesh_stl(stl, isolate=False, size=3.0, max_attempts=3)
    assert len(calls) == 2 and mesh.facts["attempts"] == 2 and len(mesh.tets) == 1
    failed = mesh.facts["failed_attempts"][0]
    assert failed["error"] == "gmsh made 1 tet(s) of zero volume, the first at (0.5000, 0.5000, 0.0000) mm"
    assert failed["points"] == [[0.5, 0.5, 0.0]] and fields[0] is None and fields[1] is not None       # attempt 2 refines at the tet


def test_every_attempt_with_a_tet_of_zero_volume_ends_with_the_count_and_the_place(monkeypatch, tmp_path):
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    fake_preparation(monkeypatch, [moved(surface, 0.0), moved(surface, 0.5)], [])
    gmsh_gives(monkeypatch, [[GOOD_TET, FLAT_TET, FLAT_TET]])
    with pytest.raises(MeshingFailed, match=r"every surface attempt \(2\): gmsh made 2 tet\(s\) of zero volume, "
                                            r"the first at \(0\.5000, 0\.5000, 0\.0000\) mm"):
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=2)


def test_a_tet_of_no_volume_or_that_the_stress_check_cannot_invert_is_found():
    """MN1. `zero_volume_centres` lists a tet when the 3 x 3 determinant of the edge vectors or the 4 x 4 determinant of [1 x y z] (the
    matrix that `verify.p2.tet_geometry` inverts) is exactly 0. The two are different numbers in floating point:
    - `tilted` has a 3 x 3 determinant that is not 0 (9.4e-15), and `numpy.linalg.inv` of its 4 x 4 matrix raises "Singular matrix": the
      first version of the check (3 x 3 only) let it pass and `verify` stopped with an internal error;
    - `no_volume` has a 3 x 3 determinant of exactly 0, and `inv` inverts its 4 x 4 matrix (the gradients are rounding noise). With the
      4 x 4 test alone the f6a tet at (71.9904, 14.9940, 0.2686) was not listed, the refinement of the next attempt changed, and f6a was refused."""
    from topopt.tet.mesher import signed_volumes, zero_volume_centres
    tilted = np.array([[79.12, 77.29, 50.09], [81.22, 82.6, 52.18], [78.9, 76.12, 48.78], [79.78, 79.45, 51.62]])
    no_volume = np.array([[44.21, 90.19, 2.36], [42.91, 89.25, 3.14], [39.04, 86.54, 5.38], [37.9, 86.0, 5.8]])

    def inverts(X):
        A = np.ones((1, 4, 4))
        A[0, :, 1:] = X
        try:
            np.linalg.inv(A)
            return True
        except np.linalg.LinAlgError:
            return False
    vol = lambda X: signed_volumes(X, np.arange(4)[None])[0]
    if inverts(tilted) or vol(tilted) == 0.0 or not inverts(no_volume) or vol(no_volume) != 0.0:
        pytest.skip("this LAPACK gives another rounding for the examples")
    good = np.array(GOOD_TET) + 100.0
    p = np.vstack([good, tilted, np.array(FLAT_TET), no_volume]).T          # (3, 16)
    t = np.arange(16).reshape(4, 4).T                                       # (4, 4): a good tet, tilted, a tet in the plane z = 0, no_volume
    centres = zero_volume_centres(p, t)
    assert centres.shape == (3, 3) and np.allclose(centres, [tilted.mean(axis=0), np.array(FLAT_TET).mean(axis=0), no_volume.mean(axis=0)])
    assert zero_volume_centres(p[:, :4], t[:, :1]).shape == (0, 3) and zero_volume_centres(p, t[:, :0]).shape == (0, 3)


def two_attempts_with_a_flat_tet_in_the_first(monkeypatch, tmp_path, refuse_second=None):
    """The sphere, a preparation that gives another surface for each attempt (or refuses the second), and a mesher that makes a flat
    tet in attempt 1 and a good tet after it. Returns (stl, the list of the log lines)."""
    from test_tet import _ok_probe
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    fake_preparation(monkeypatch, [moved(surface, 0.0), moved(surface, 0.5)], [])
    if refuse_second:
        good = ts.prepare_surface

        def refusing(path, h, **kw):
            if kw.get("size_field") is not None:
                raise ts.SurfaceRejected(refuse_second)
            return good(path, h, **kw)
        monkeypatch.setattr(ts, "prepare_surface", refusing)
    gmsh_gives(monkeypatch, [[GOOD_TET, FLAT_TET], [GOOD_TET]])
    return stl


def test_the_log_line_of_a_flat_tet_does_not_say_intersection_points(monkeypatch, tmp_path):
    """MN4. The line said "gmsh failed (1 intersection point(s) reported)" for a tet of zero volume."""
    from topopt.tet import mesh_stl
    stl = two_attempts_with_a_flat_tet_in_the_first(monkeypatch, tmp_path)
    lines = []
    mesh_stl(stl, isolate=False, size=3.0, max_attempts=3, log=lines.append)
    failed = [x for x in lines if "zero volume" in x]
    assert failed == ["gmsh made 1 tet(s) of zero volume, the first at (0.5000, 0.5000, 0.0000) mm"] and not any("intersection point" in x for x in lines)


def test_a_refusal_after_an_attempt_with_a_flat_tet_says_so(monkeypatch, tmp_path):
    """MN4. The text of a late refusal named only "gmsh failed on the prepared surface"; README promises what gmsh did."""
    from topopt.tet import mesh_stl
    stl = two_attempts_with_a_flat_tet_in_the_first(monkeypatch, tmp_path, refuse_second="prepared surface has an edge of 0.0076 mm (limit 0.05 mm)")
    with pytest.raises(ts.SurfaceRejected) as e:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=3)
    assert str(e.value) == ("gmsh made 1 tet(s) of zero volume on the prepared surface of attempt 1, and the surface preparation for attempt 2 "
                            "(local refinement at 1 point(s)) was refused: prepared surface has an edge of 0.0076 mm (limit 0.05 mm)")


class FoundCrossings:
    """What `find_crossings` gives (the part of it that `mesh_stl` reads)."""
    def __init__(self, points):
        self.points = np.array(points, float).reshape(-1, 3)
        self.total = len(self.points)


def test_the_centre_of_a_flat_tet_is_not_a_crossing_of_the_input(monkeypatch, tmp_path):
    """MN4. The places of a failed attempt were all tested against the crossings of the input STL, and a flat tet within one mesh size of
    such a crossing ended the retries as "the STL itself crosses itself". The seam of the F-series STLs has crossings of the input. A flat
    tet is not a crossing, so the retries go on (attempt 2 gives a mesh). The same place named by gmsh for a crossing still stops them."""
    from test_tet import mesher
    from topopt.tet import MeshingFailed, mesh_stl
    stl = two_attempts_with_a_flat_tet_in_the_first(monkeypatch, tmp_path)
    monkeypatch.setattr(mesher, "_search", lambda V, F, surface, attempt, failed, say: FoundCrossings([[0.5, 0.5, 0.0]] if surface == "input" else []))
    mesh = mesh_stl(stl, isolate=False, size=3.0, max_attempts=3)
    assert mesh.facts["attempts"] == 2
    gmsh_fails(monkeypatch, points=((0.5, 0.5, 0.0),))
    with pytest.raises(MeshingFailed) as e:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=3)
    assert e.value.crossings["stopped"] == "input_crosses_itself"


def test_a_late_refusal_keeps_the_crossings_of_the_attempt_before(monkeypatch, tmp_path):
    """MN5. `topopt verify` names the regions near the crossings with `MeshingFailed.crossings` (`_crossing_bodies`). The late refusal had
    none, so the user did not see where the 11 (f7c) or 33 (f0_root) crossings of attempt 1 were."""
    from test_tet import _ok_probe, mesher
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    fake_preparation(monkeypatch, [moved(surface, 0.0)], [CLEAN_STEP])
    monkeypatch.setattr(ts, "prepare_surface", lambda path, h, **kw: (_ for _ in ()).throw(ts.SurfaceRejected("refused")) if kw.get("size_field") is not None
                        else ts.PreparedSurface(surface[0], surface[1], {}, {}, [dict(CLEAN_STEP)]))
    monkeypatch.setattr(mesher, "_search", lambda V, F, surface, attempt, failed, say:
                        FoundCrossings([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] if surface == "prepared" else []))
    gmsh_fails(monkeypatch)
    with pytest.raises(ts.SurfaceRejected) as e:
        mesh_stl_of(stl)
    assert str(e.value).startswith("gmsh failed on the prepared surface of attempt 1 (it crosses itself at 2 place(s)), and the surface preparation ")
    c = e.value.crossings
    assert c["prepared"] == {"count": 2, "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "attempts": [2]} and c["surface_retry"] == CLEAN_STEP
    assert c["input"] == {"count": 0, "points": []}
    assert json.loads(json.dumps(c)) == c


def mesh_stl_of(stl):
    from topopt.tet import mesh_stl
    return mesh_stl(stl, isolate=False, size=3.0)


def test_a_refusal_of_the_preparation_in_a_later_attempt_says_what_gmsh_did_in_the_attempt_before(monkeypatch, tmp_path):
    """F4-27 (f7c): attempt 1 had a prepared surface and gmsh failed on it; the preparation with local refinement for attempt 2 was
    refused for a short edge. The error was only the refusal, as if the first preparation had failed."""
    from test_tet import _ok_probe
    from topopt.tet import mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    calls = []

    def fake(path, h, **kw):
        calls.append(kw.get("size_field"))
        if len(calls) > 1:
            raise ts.SurfaceRejected("prepared surface has an edge of 0.0076 mm (limit 0.05 mm)")
        return ts.PreparedSurface(surface[0], surface[1], {}, {"deviation_from_input_mm": {"mean_mm": 0.0, "p99_mm": 0.0, "max_mm": 0.0, "samples": 0}}, [])
    monkeypatch.setattr(ts, "prepare_surface", fake)
    gmsh_fails(monkeypatch)
    with pytest.raises(ts.SurfaceRejected) as e:
        mesh_stl(stl, isolate=False, size=3.0)
    assert str(e.value) == ("gmsh failed on the prepared surface of attempt 1, and the surface preparation for attempt 2 (local refinement at 1 point(s)) "
                            "was refused: prepared surface has an edge of 0.0076 mm (limit 0.05 mm)")
    assert e.value.crossings == {"prepared": {"count": 0, "points": [], "attempts": [0]},          # MN5: the facts of the attempt before are there
                                 "input": {"count": 0, "points": []}}
    first = ts.SurfaceRejected("prepared surface has an edge of 0.0076 mm (limit 0.05 mm)")           # a refusal in attempt 1 is not changed
    calls.clear()
    monkeypatch.setattr(ts, "prepare_surface", lambda *a, **k: (_ for _ in ()).throw(first))
    with pytest.raises(ts.SurfaceRejected) as e:
        mesh_stl(stl, isolate=False, size=3.0)
    assert e.value is first


def test_the_plain_error_of_gmsh_gets_the_sentence_when_a_retry_made_the_surface(monkeypatch, tmp_path):
    """gmsh names no place and the search finds no crossing: the old code re-raised the error of gmsh with no fact. A retry made the
    surface, so the error says so, and the fact is there."""
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    calls = gmsh_fails(monkeypatch, points=(), message="algorithm 10: nope")
    fake_preparation(monkeypatch, [surface], [CLEAN_STEP])
    with pytest.raises(MeshingFailed) as e:
        mesh_stl(stl, isolate=False, size=3.0)
    assert len(calls) == 1 and str(e.value) == "algorithm 10: nope. " + clean_sentence()
    assert e.value.crossings["surface_retry"] == CLEAN_STEP


def test_the_weld_retry_is_named_too(monkeypatch, tmp_path):
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    gmsh_fails(monkeypatch, points=(), message="algorithm 10: nope")
    fake_preparation(monkeypatch, [surface], [WELD_STEP, dict(step="split_long_edges")])
    with pytest.raises(MeshingFailed) as e:
        mesh_stl(stl, isolate=False, size=3.0)
    assert str(e.value) == ("algorithm 10: nope. The surface that gmsh got is the input surface with its 6 edges under 0.05 mm welded "
                            f"and prepared again (weld retry; the first preparation was refused: {REFUSED_FIRST})")
    assert e.value.crossings["surface_retry"] == WELD_STEP


def test_the_stop_for_an_stl_that_crosses_itself_names_the_retry(monkeypatch, tmp_path):
    from test_tet import _ok_probe, _pushed
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl = tmp_path / "pushed.stl"
    V, F = _pushed(-1.3)
    trimesh.Trimesh(V, F, process=False).export(str(stl))
    gmsh_fails(monkeypatch)
    fake_preparation(monkeypatch, [(V, F)], [CLEAN_STEP])
    with pytest.raises(MeshingFailed) as e:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=10)
    text = str(e.value)
    assert e.value.crossings["stopped"] == "input_crosses_itself" and "The STL itself crosses itself at" in text
    assert text.index(clean_sentence()) < text.index("Options: repair the STL")
    assert e.value.crossings["surface_retry"] == CLEAN_STEP


def test_the_stop_for_crossings_that_do_not_become_fewer_names_the_retry_and_the_same_surface(monkeypatch, tmp_path):
    from test_tet import _ok_probe, _pushed
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, _surface = a_sphere_stl(tmp_path)
    calls = gmsh_fails(monkeypatch)
    fake_preparation(monkeypatch, [_pushed(-1.3)], [CLEAN_STEP])                         # the same surface in each attempt
    with pytest.raises(MeshingFailed) as e:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=10)
    text = str(e.value)
    assert len(calls) == 2 and e.value.crossings["stopped"] == "no_progress"
    assert "did not make them fewer" in text and clean_sentence(" in attempt 2") in text
    assert "the surface of attempt 2 is the same as the surface of attempt 1, so the next attempt would repeat this one" in text
    assert e.value.crossings["same_surface_as_attempt"] == 1 and e.value.crossings["surface_retry"] == CLEAN_STEP


def test_a_crash_of_gmsh_names_the_retry(monkeypatch, tmp_path):
    from test_tet import _ok_probe, mesher
    from topopt.tet import MeshingCrashed, mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    fake_preparation(monkeypatch, [surface], [CLEAN_STEP])

    def crash(V, F, size, **kw):
        raise MeshingCrashed("died", dict(signal="SIGSEGV", returncode=-11, size=size, optimize=kw["optimize"]))
    monkeypatch.setattr(mesher, "mesh_surface_isolated", crash)
    with pytest.raises(MeshingCrashed) as e:
        mesh_stl(stl, size=3.0, fallback_optimize=())
    assert str(e.value).endswith(" " + clean_sentence() + ".") and "SIGSEGV" in str(e.value)
    assert e.value.crossings["surface_retry"] == CLEAN_STEP and len(e.value.crashes) == 1


def test_retries_stop_when_the_prepared_surface_does_not_change(monkeypatch, tmp_path):
    """The case of the review (`bowl_smooth` at 1.2 mm): gmsh names a place that no search of crossings sees (0 crossings), the
    refinement around it gives the same surface, and the old code made 10 attempts that repeated one another. The retries now end at
    the first attempt that got the surface of the attempt before. No number: the surfaces are equal."""
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    calls = gmsh_fails(monkeypatch)
    fields = fake_preparation(monkeypatch, [moved(surface, 0.0), moved(surface, 0.5), moved(surface, 0.5), moved(surface, 1.0)], [CLEAN_STEP])
    with pytest.raises(MeshingFailed) as e:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=10)
    text = str(e.value)
    assert len(calls) == 3 and len(fields) == 3                                         # attempt 3 got the surface of attempt 2
    assert e.value.crossings["stopped"] == "no_progress" and e.value.crossings["same_surface_as_attempt"] == 2
    assert e.value.crossings["prepared"]["count"] == 0 and e.value.crossings["prepared"]["attempts"] == [0, 0, 0]
    assert "The refinement points did not change the prepared surface: the surface of attempt 3 is the same as the surface of attempt 2, " \
           "so the next attempt would repeat this one." in text
    assert "The search finds no place where the prepared surface crosses itself" in text and "gmsh named (mm): (0.00, 0.00, 10.00)" in text
    assert clean_sentence(" in attempt 3") in text and "Option: try a mesh size nearer to the mean edge of the STL" in text
    assert "every surface attempt" not in text


def test_a_surface_that_changes_in_each_attempt_is_not_stopped_early(monkeypatch, tmp_path):
    """The guard of the rule above: other surfaces in each attempt (the refinement works), so all the attempts are made."""
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, surface = a_sphere_stl(tmp_path)
    calls = gmsh_fails(monkeypatch)
    fake_preparation(monkeypatch, [moved(surface, 0.5 * i) for i in range(5)], [])
    with pytest.raises(MeshingFailed, match=r"every surface attempt \(5\)"):
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=5)
    assert len(calls) == 5


def test_the_same_vertices_with_other_faces_are_another_surface(monkeypatch, tmp_path):
    """The equality that stops the retries is on `V` and on `F`: a surface with the same vertices and the faces turned over is not the
    same surface, so the second attempt is not stopped (it fails for the same reason as the first, and the limit ends it)."""
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl
    _ok_probe(monkeypatch)
    stl, (V, F) = a_sphere_stl(tmp_path)
    calls = gmsh_fails(monkeypatch)
    fake_preparation(monkeypatch, [(V, F), (V, F[:, [0, 2, 1]])], [])
    with pytest.raises(MeshingFailed, match=r"every surface attempt \(2\)"):
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=2)
    assert len(calls) == 2
