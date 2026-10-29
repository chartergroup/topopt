"""The weld retry of `prepare_surface` (C3c, CONTRACTS.md section 14).

The defect: `topopt verify` refused an `apply_marks` output with "prepared surface has an edge of 0.0000 mm (limit 0.05 mm)".
Marching cubes had put 2 to 5 vertices within a few micrometres of one node (a flat face of the part on the lattice). The edge
collapse of the surface preparation removes most of these groups, but not a hair-thin tube (three vertices around a handle of
1e-5 mm: no collapse keeps the link condition).

The retry: when the prepared surface still has an edge under `MIN_EDGE_REJECT_MM`, the input surface is welded with that same limit
(`weld_short_edges`) and prepared again. It runs only after that refusal. These tests build the case from a sphere and a tube of
the cross-section 2e-5 mm between two of its vertices (a miniature of the defect), so no binary fixture is in the repository. The
real case (the posts part at a sub-voxel shift) is in `tests/test_cleanup_ratchet.py`. The tests need numpy, scipy and numba only,
except the one that meshes (`needs_gmsh`).
"""
import manifold3d as m3
import numpy as np
import pytest
import trimesh

from helpers import needs_gmsh
from topopt.tet import surface as ts
from topopt.verify.report import weld_line

LIMIT = ts.MIN_EDGE_REJECT_MM


def _fan(F, v):
    """The faces around the vertex v as ordered pairs (a, b): the face (v, a, b) is followed by the face (v, b, c)."""
    ring = {}
    for f in F:
        if v in f:
            k = list(f).index(v)
            ring[int(f[(k + 1) % 3])] = int(f[(k + 2) % 3])
    a, out = next(iter(ring)), []
    for _ in range(len(ring)):
        out.append((a, ring[a]))
        a = ring[a]
    return out


def hair_tube_sphere(width=(2e-5, 2e-5), subdivisions=2, radius=5.0, rings=0):
    """(V, F) of a closed manifold: an icosphere with a straight tube of triangular cross-section between two opposite vertices.
    Each end is a vertex that became a small triangular hole (side about `width` mm, one value for each end); the tube joins the two
    holes. With the default both holes are near-zero: their three edges each are near-zero edges on the seam between the sphere and the
    tube, and each of them fails the link condition. `rings` is the number of extra rings of three vertices along the tube (each
    ring has three near-zero edges too, and the weld leaves no face on it)."""
    sph = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    V = [np.asarray(p, float) for p in sph.vertices]
    F = [tuple(int(x) for x in f) for f in sph.faces]
    v1 = 0
    v2 = int(np.argmin(sph.vertices @ sph.vertices[v1]))                 # the vertex farthest from v1
    holes = []
    for v, w in zip((v1, v2), width):
        fan = _fan(F, v)
        n = V[v] / np.linalg.norm(V[v])
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        new = []
        for s in range(3):
            V.append(V[v] + w * (np.cos(2 * np.pi * s / 3) * e1 + np.sin(2 * np.pi * s / 3) * e2))
            new.append(len(V) - 1)
        F = [f for f in F if v not in f]
        sector = [i * 3 // len(fan) for i in range(len(fan))]             # the fan of v goes to the three new vertices
        edges = []
        for i, (a, b) in enumerate(fan):
            F.append((new[sector[i]], a, b))
            if sector[i] != sector[i - 1]:                                # a face between two sectors closes the cut
                A, B = new[sector[i - 1]], new[sector[i]]
                F.append((A, a, B))
                edges.append((B, A))                                      # the directed edge that the hole has in these faces
        holes.append(edges)

    def cycle(edges):
        nxt = dict(edges)
        c = [edges[0][0]]
        while nxt[c[-1]] != c[0]:
            c.append(nxt[c[-1]])
        return c

    c1, c2 = cycle(holes[0]), cycle(holes[1])
    m = {c1[j]: c2[(-j) % 3] for j in range(3)}                           # the second hole is met the other way round
    layers = [{p: p for p in c1}]
    for t in range(1, rings + 1):                                         # rings along the tube, the same cross-section
        layer = {}
        for p in c1:
            V.append(V[v1] + (V[v2] - V[v1]) * t / (rings + 1) + (V[p] - V[v1]))
            layer[p] = len(V) - 1
        layers.append(layer)
    layers.append(m)
    for A, B in zip(layers, layers[1:]):
        for p, q in holes[0]:
            F.append((A[q], A[p], B[p]))
            F.append((A[q], B[p], B[q]))
    return np.array(V), np.array(F, np.int64)


def box_with_hole(tmp_path, radius, segments=64):
    """(V, F, path): a box 10 x 10 x 4 mm with a through-hole of the given radius, as `prepare_surface` loads it. The ring of the hole
    has edges of 2 pi r / segments mm: under the verify limit for a hole up to about 0.5 mm."""
    box = m3.Manifold.cube([10.0, 10.0, 4.0]).translate([-5.0, -5.0, -2.0])
    part = (box - m3.Manifold.cylinder(6.0, radius, radius, segments).translate([0.0, 0.0, -3.0])).to_mesh()
    path = write(tmp_path / f"hole_{radius:g}.stl", np.asarray(part.vert_properties[:, :3], float), np.asarray(part.tri_verts, np.int64))
    V, F = ts.load_stl_surface(path)
    return V, F, path


def hair_ring(n=24, radius=3.0, width=2e-5, shift=(0.0, 0.0, 0.0)):
    """(V, F) of a closed manifold that is a ring of the triangular cross-section `width` mm: a body of no size (volume 1e-8 mm3)."""
    V, F = [], []
    for i in range(n):
        th = 2 * np.pi * i / n
        r, z = np.array([np.cos(th), np.sin(th), 0.0]), np.array([0.0, 0.0, 1.0])
        for k in range(3):
            phi = 2 * np.pi * k / 3
            V.append(radius * r + width * (np.cos(phi) * r + np.sin(phi) * z) + np.asarray(shift))
    for i in range(n):
        j = (i + 1) % n
        for k in range(3):
            k2 = (k + 1) % 3
            a, b, c, d = 3 * i + k, 3 * i + k2, 3 * j + k2, 3 * j + k
            F += [(a, b, c), (a, c, d)]
    V, F = np.array(V), np.array(F, np.int64)
    if np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) < 0:
        F = F[:, ::-1].copy()
    return V, F


def write(path, V, F):
    trimesh.Trimesh(V, F, process=False).export(str(path))
    return str(path)


def short_edges(V, F):
    eu, ev, _cnt = ts.edge_table(F, len(V))
    return int((np.linalg.norm(V[eu] - V[ev], axis=1) < LIMIT).sum())


def retry_off(mp):
    """Switch the retries off: the weld says that it cannot help, and so does the clean retry of F1-5 (it welds the edges that stay
    with `weld_short_edges` too, review MN5), so `prepare_surface` gives the first refusal."""
    def off(*_a, **_k):
        raise ts._WeldFailed("the weld is switched off in this test")

    def clean_off(*_a, **_k):
        raise ts._CleanFailed("the clean retry is switched off in this test")
    mp.setattr(ts, "weld_short_edges", off)
    mp.setattr(ts, "clean_short_edges", clean_off, raising=False)           # the test runs on the code before F1-5 too


def test_the_hair_tube_is_a_closed_manifold_with_six_edges_under_the_limit(tmp_path):
    V, F = hair_tube_sphere()
    r = ts.surface_report(V, F)
    assert r["closed_manifold"] and r["bodies"] == 1 and r["volume_mm3"] > 0, r
    assert short_edges(V, F) == 6 and r["edge_min_mm"] < 1e-4 and r["fold_edges"] == 3        # a tube of no width has fold edges


def test_the_refusal_before_and_the_prepared_surface_after_the_weld_retry(tmp_path, monkeypatch):
    """The near-zero edges on the seam of the tube: no collapse can remove them, so the first try is refused (the old behaviour).
    With the retry the input is welded and prepared; the step is the first of `steps` and says what the weld did."""
    V, F = hair_tube_sphere()
    path = write(tmp_path / "tube.stl", V, F)
    with monkeypatch.context() as mp:
        retry_off(mp)
        with pytest.raises(ts.SurfaceRejected, match=r"prepared surface has an edge of 0\.0000 mm \(limit 0\.05 mm\)"):
            ts.prepare_surface(path, 1.0, verbose=False)
    prep = ts.prepare_surface(path, 1.0, verbose=False)
    r = prep.prepared
    assert r["closed_manifold"] and r["edge_min_mm"] >= LIMIT and r["bodies"] == 1 and r["fold_edges"] == 0
    weld = prep.steps[0]
    assert weld["step"] == "weld_short_edges" and weld["limit_mm"] == LIMIT                # the limit of verify, not a copy of it
    assert (weld["short_edges_in_input"], weld["edges_welded"], weld["vertices_merged"], weld["passes"]) == (6, 6, 4, 1)
    assert (weld["bodies_before"], weld["bodies_after"]) == (1, 1)
    assert abs(weld["volume_change_mm3"]) < 1e-3 and weld["volume_after_mm3"] == pytest.approx(weld["volume_before_mm3"], rel=1e-6)
    assert weld["refused_first"].startswith("prepared surface has an edge of 0.0000 mm")
    assert [s["step"] for s in prep.steps[1:]][:2] == ["split_long_edges", "flip_edges"]   # the steps of the old path follow
    # the prepared surface is the sphere: the tube had no volume
    sph = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    assert ts.surface_report(prep.V, prep.F)["volume_mm3"] == pytest.approx(sph.volume, rel=0.1)
    assert prep.prepared["deviation_from_input_mm"]["max_mm"] < 0.5 and prep.original["faces"] == len(F)


def test_an_input_that_is_prepared_at_the_first_try_is_not_touched(tmp_path, monkeypatch):
    """The retry runs only after the refusal. A sphere with a near-zero edge (an edge split 1e-5 mm from its end) is prepared by the
    collapse alone: the weld is not called, and the faces are those of the old path (`_prepare_arrays` is that path)."""
    sph = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    V, F = np.array(sph.vertices), np.array(sph.faces)
    u, v = int(F[0, 0]), int(F[0, 1])                                       # split the edge u-v: a vertex m next to u
    m = V[u] + (V[v] - V[u]) * (1e-5 / np.linalg.norm(V[v] - V[u]))
    V = np.vstack([V, m])
    k = len(V) - 1
    faces = []
    for f in F.tolist():
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):                   # the faces on the edge, in both directions
            if (f[a], f[b]) in ((u, v), (v, u)):
                faces += [[f[a], k, f[c]], [k, f[b], f[c]]]
                break
        else:
            faces.append(f)
    F2 = np.array(faces)
    assert ts.surface_report(V, F2)["closed_manifold"] and short_edges(V, F2) == 1
    for name, (VV, FF) in (("split", (V, F2)), ("plain", (np.array(sph.vertices), np.array(sph.faces)))):
        path = write(tmp_path / f"{name}.stl", VV, FF)
        with monkeypatch.context() as mp:
            def never(*_a, **_k):
                raise AssertionError("the weld retry ran for an input that is prepared at the first try")
            mp.setattr(ts, "weld_short_edges", never)
            got = ts.prepare_surface(path, 2.0, verbose=False)
        V0, F0 = ts.load_stl_surface(path)
        old = ts._prepare_arrays(V0, F0, 2.0, 0.0, method="collapse", verbose=False, eps_factor=0.1, size_field=None,
                                 min_edge_mm=LIMIT)
        assert np.array_equal(got.V, old.V) and np.array_equal(got.F, old.F)
        assert [s["step"] for s in got.steps] == [s["step"] for s in old.steps] and "weld_short_edges" not in str(got.steps)


def test_a_failed_retry_keeps_the_first_refusal_and_says_why(tmp_path):
    # the weld of a ring of no size leaves no triangle
    V, F = hair_ring()
    path = write(tmp_path / "ring.stl", V, F)
    with pytest.raises(ts.SurfaceRejected) as e:
        ts.prepare_surface(path, 1.0, verbose=False)
    assert isinstance(e.value, ts.PreparedSurfaceRejected)                      # a refusal of the surface prepared for the size
    text = str(e.value)
    assert text.startswith("prepared surface has an edge of 0.0000 mm (limit 0.05 mm); the weld retry was tried")
    assert text.endswith("the weld leaves no triangle")
    # a second body of no size: the weld would remove it, so the body count changes, and the weld is refused
    Vt, Ft = hair_tube_sphere()
    Vr, Fr = hair_ring(shift=(20.0, 0.0, 0.0))
    path = write(tmp_path / "two.stl", np.vstack([Vt, Vr]), np.vstack([Ft, Fr + len(Vt)]))
    assert ts.surface_report(np.vstack([Vt, Vr]), np.vstack([Ft, Fr + len(Vt)]))["bodies"] == 2
    with pytest.raises(ts.SurfaceRejected) as e:
        ts.prepare_surface(path, 1.0, verbose=False)
    assert isinstance(e.value, ts.PreparedSurfaceRejected)
    assert "the weld retry was tried and did not help: the weld changes the number of bodies (2 to 1)" in str(e.value)
    assert str(e.value).startswith("prepared surface has an edge of 0.0000 mm")


def test_the_weld_says_when_there_is_nothing_to_weld(tmp_path):
    """A refusal can come from an edge that the preparation made itself: the input has no edge under the limit, the weld has nothing
    to do, and the first refusal stays with that fact."""
    sph = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    with pytest.raises(ts._WeldFailed, match="no edge under 0.05 mm"):
        ts.weld_short_edges(np.array(sph.vertices), np.array(sph.faces), LIMIT)


def test_a_weld_that_pinches_the_surface_is_refused(tmp_path):
    """One end of the tube is a near-zero hole and the other is 0.3 mm wide. The weld of the near-zero end would join the tube to the
    sphere at one vertex that two sheets share (a pinch), so the weld is refused: the first refusal stays, with the reason."""
    V, F = hair_tube_sphere(width=(2e-5, 0.3))
    assert ts.surface_report(V, F)["closed_manifold"] and short_edges(V, F) == 3
    with pytest.raises(ts._WeldFailed, match=r"not a closed manifold \(.*non-manifold vertices 1\); first pinch vertex at \(-?[\d.]+, -?[\d.]+, -?[\d.]+\)$"):
        ts.weld_short_edges(V, F, LIMIT)
    path = write(tmp_path / "pinch.stl", V, F)
    with pytest.raises(ts.SurfaceRejected, match=r"edge of .*; the weld retry was tried and did not help: the welded surface is not a closed") as e:
        ts.prepare_surface(path, 1.0, verbose=False)
    assert isinstance(e.value, ts.PreparedSurfaceRejected)


@needs_gmsh
def test_the_mesh_facts_name_the_weld_retry_only_when_it_ran(tmp_path):
    from topopt.tet import mesh_stl
    V, F = hair_tube_sphere()
    mesh = mesh_stl(write(tmp_path / "tube.stl", V, F), size=2.0)
    weld = mesh.facts["surface_weld_retry"]
    assert weld["step"] == "weld_short_edges" and weld["vertices_merged"] == 4 and (weld["bodies_before"], weld["bodies_after"]) == (1, 1)
    assert mesh.facts["volume_ratio_stl"] == pytest.approx(1.0, rel=0.1)
    sph = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    plain = mesh_stl(write(tmp_path / "plain.stl", np.array(sph.vertices), np.array(sph.faces)), size=2.0)
    assert "surface_weld_retry" not in plain.facts


# --- C3c fix round 1 (review M1, m1, m2, m3, n4) ---------------------------------------------------------------------------------

def with_a_hair_tube(tmp_path, V, F, name):
    """An STL with the part and, far from it, the hair-tube sphere: the first try of `prepare_surface` is refused for the tube, so the
    retry runs on the whole input (the part is welded too)."""
    Vt, Ft = hair_tube_sphere()
    Vt = Vt + np.array([30.0, 0.0, 0.0])
    return write(tmp_path / f"{name}.stl", np.vstack([V, Vt]), np.vstack([F, Ft + len(V)]))


def test_a_through_hole_is_not_filled_by_the_weld(tmp_path):
    """M1 of the review. The ring of a hole of 0.3 mm radius (64 segments) has edges of 0.029 mm: a chain of short edges. Its weld
    would join the ring to one vertex and fill the hole (+1.13 mm3). The vertices of the ring would move by 0.3 mm, six times the
    radius of the weld, so the weld is refused."""
    V, F, _path = box_with_hole(tmp_path, 0.3)
    assert short_edges(V, F) == 128 and ts.surface_report(V, F)["closed_manifold"]
    with pytest.raises(ts._WeldFailed, match=r"^the weld moves a vertex by 0\.3 mm, more than the weld radius 0\.05 mm$"):
        ts.weld_short_edges(V, F, LIMIT)
    # through `prepare_surface`: the tube in the same file makes the first try refuse, the weld is refused, the first refusal stays
    path = with_a_hair_tube(tmp_path, V, F, "hole_and_tube")
    with pytest.raises(ts.SurfaceRejected) as e:
        ts.prepare_surface(path, 1.0, verbose=False)
    assert isinstance(e.value, ts.PreparedSurfaceRejected)
    # the first refusal and the reason of the weld stay at the start; the last retry (F1-5) adds its own reason: it does not fill the hole either
    assert str(e.value).startswith("prepared surface has an edge of 0.0000 mm (limit 0.05 mm); the weld retry was tried and did not help: "
                                   "the weld moves a vertex by 0.3 mm, more than the weld radius 0.05 mm; the clean retry of the "
                                   "prepared surface did not help: ")
    assert "the weld of them is refused: the weld changes the number of bodies" in str(e.value)


@pytest.mark.parametrize("radius,welded", [(0.04, True), (0.06, False)])
def test_the_weld_radius_is_the_limit_of_the_move(tmp_path, radius, welded):
    """A hole of 0.04 mm radius is a hair handle of the size that the weld exists to remove: its ring moves by 0.04 mm, under the
    radius 0.05 mm, and it is welded (the genus changes, and the facts say so). A hole of 0.06 mm radius is refused."""
    V, F, _path = box_with_hole(tmp_path, radius)
    if not welded:
        with pytest.raises(ts._WeldFailed, match=r"the weld moves a vertex by 0\.06 mm, more than the weld radius 0\.05 mm"):
            ts.weld_short_edges(V, F, LIMIT)
        return
    Vw, Fw, facts = ts.weld_short_edges(V, F, LIMIT)
    assert facts["max_vertex_move_mm"] == pytest.approx(0.04, abs=1e-4) and facts["max_vertex_move_mm"] <= LIMIT
    assert (facts["genus_before"], facts["genus_after"]) == ([1], [0]) and (facts["euler_before"], facts["euler_after"]) == ([0], [2])
    assert ts.surface_report(Vw, Fw)["closed_manifold"] and short_edges(Vw, Fw) == 0


def test_a_feature_that_goes_with_all_its_faces_counts_in_the_move(tmp_path):
    """A pin that is a cone: a base ring of 0.04 mm radius on the plate and a top ring of 0.3 mm radius, 2 mm up, rings of fine edges.
    The weld joins each ring to one vertex and every face of the pin is left with two equal vertices: the pin goes, with its top ring.
    The base ring moves by 0.04 mm only, under the radius, but the top ring moves by 0.3 mm. The vertices that no face is left on
    count in the move, so the weld is refused."""
    box = m3.Manifold.cube([10.0, 10.0, 4.0]).translate([-5.0, -5.0, -2.0])
    pin = m3.Manifold.cylinder(2.0, 0.04, 0.3, 64).translate([0.0, 0.0, 2.0])
    part = (box + pin).to_mesh()
    path = write(tmp_path / "cone_pin.stl", np.asarray(part.vert_properties[:, :3], float), np.asarray(part.tri_verts, np.int64))
    V, F = ts.load_stl_surface(path)
    assert ts.surface_report(V, F)["closed_manifold"] and short_edges(V, F) == 128
    with pytest.raises(ts._WeldFailed, match=r"^the weld moves a vertex by 0\.3 mm, more than the weld radius 0\.05 mm$"):
        ts.weld_short_edges(V, F, LIMIT)


def test_the_hair_tube_still_welds_and_its_genus_change_is_a_fact(tmp_path):
    """The tube is a handle (genus 1): the weld removes it. A change of the topology is not a refusal; the facts say it, for each body."""
    V, F = hair_tube_sphere()
    path = write(tmp_path / "tube.stl", V, F)
    weld = ts.prepare_surface(path, 1.0, verbose=False).steps[0]
    assert (weld["euler_before"], weld["euler_after"]) == ([0], [2]) and (weld["genus_before"], weld["genus_after"]) == ([1], [0])
    assert 0 < weld["max_vertex_move_mm"] < 1e-4 and weld["max_vertex_move_mm"] <= weld["limit_mm"]
    assert weld["volume_change_mm3"] < 0 and weld["volume_change_mm3"] == pytest.approx(weld["volume_after_mm3"] - weld["volume_before_mm3"], abs=1e-9)
    line = weld_line(weld)
    assert "4 vertices merged and 0 dropped" in line and "largest move 0.0000 mm (the weld radius is 0.05 mm)" in line
    assert line.endswith("genus change: body 1 from genus 1 to 0 (Euler characteristic 0 to 2)")
    assert weld_line(dict(weld, genus_after=weld["genus_before"], euler_after=weld["euler_before"])).endswith("genus unchanged")


def test_the_weld_counts_merged_and_dropped_vertices_apart(tmp_path):
    """A ring of three vertices in the middle of the tube has no face left after the weld: its vertices are dropped, not merged into
    another vertex. The two counts and the vertex count after add up to the vertices before."""
    V, F = ts._compact(*hair_tube_sphere(rings=1))                           # as `load_stl_surface` gives it: every vertex is used
    assert short_edges(V, F) == 9 and ts.surface_report(V, F)["closed_manifold"]
    Vw, Fw, w = ts.weld_short_edges(V, F, LIMIT)
    assert (w["vertices_merged"], w["vertices_dropped"]) == (4, 3)
    assert w["vertices_before"] == w["vertices_after"] + w["vertices_merged"] + w["vertices_dropped"] and w["vertices_after"] == len(Vw)
    assert w["short_edges_in_input"] == 9 and (w["genus_before"], w["genus_after"]) == ([1], [0])


def test_a_refusal_for_another_reason_does_not_start_the_retry(tmp_path, monkeypatch):
    """The retry answers one refusal only. An input that is not closed, and a prepared surface that is not closed, keep their refusal."""
    sph = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    V, F = np.array(sph.vertices), np.array(sph.faces)

    def never(*_a, **_k):
        raise AssertionError("the weld retry ran after a refusal that is not the short edge")
    monkeypatch.setattr(ts, "weld_short_edges", never)
    with pytest.raises(ts.SurfaceRejected, match="input surface is not a closed oriented manifold: open edges 3") as e:
        ts.prepare_surface(write(tmp_path / "open.stl", V, F[1:]), 2.0, verbose=False)
    assert not isinstance(e.value, ts.PreparedSurfaceRejected)                  # a refusal of the input STL: no other size helps
    assert "weld" not in str(e.value)
    real, calls = ts.surface_report, []

    def report(V_, F_):                                                       # the report of the prepared surface says: not closed
        calls.append(1)
        r = real(V_, F_)
        return dict(r, closed_manifold=False) if len(calls) == 2 else r
    monkeypatch.setattr(ts, "surface_report", report)
    with pytest.raises(ts.SurfaceRejected, match="^prepared surface is not a closed manifold") as e:
        ts.prepare_surface(write(tmp_path / "sphere.stl", V, F), 2.0, verbose=False)
    assert isinstance(e.value, ts.PreparedSurfaceRejected)
    assert len(calls) == 2 and "weld" not in str(e.value)


def test_the_second_preparation_that_refuses_keeps_the_first_refusal_in_the_text(tmp_path, monkeypatch):
    """The weld succeeds and the preparation of the welded surface refuses: the user reads the first refusal and the second one."""
    V, F = hair_tube_sphere()
    path = write(tmp_path / "tube.stl", V, F)
    real, calls = ts._prepare_arrays, []

    def twice(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise ts.SurfaceRejected("the second preparation refused")
        return real(*a, **k)
    monkeypatch.setattr(ts, "_prepare_arrays", twice)
    monkeypatch.setattr(ts, "clean_short_edges", lambda *a, **k: (_ for _ in ()).throw(ts._CleanFailed("no clean today")))   # F1-5: the last retry
    with pytest.raises(ts.SurfaceRejected) as e:
        ts.prepare_surface(path, 1.0, verbose=False)
    assert isinstance(e.value, ts.PreparedSurfaceRejected)
    assert len(calls) == 2 and str(e.value) == ("prepared surface has an edge of 0.0000 mm (limit 0.05 mm); the weld retry was tried "
                                                "and did not help: the second preparation refused; the clean retry of the "
                                                "prepared surface did not help: no clean today")


def test_the_deviation_of_a_retried_surface_is_measured_against_the_input(tmp_path):
    V, F = hair_tube_sphere(width=(2e-5, 2e-5), rings=1)
    path = write(tmp_path / "tube.stl", V, F)
    prep = ts.prepare_surface(path, 1.0, verbose=False)
    V0, F0 = ts.load_stl_surface(path)
    assert prep.prepared["deviation_from_input_mm"] == ts.surface_deviation(V0, F0, prep.V, prep.F)
    assert prep.original["faces"] == len(F0) and prep.original["vertices"] == len(V0)


def test_the_checks_of_the_weld_for_fold_edges_and_for_the_volume(monkeypatch):
    """The weld refuses a welded surface with more fold edges than the input, and one that has no volume. The report of the welded
    surface is changed in the test: the weld of a real input does not reach these states on the fixtures here."""
    V, F = hair_tube_sphere()
    real = ts.surface_report

    def patched(**change):
        calls = []

        def report(V_, F_):
            calls.append(1)
            r = real(V_, F_)
            return dict(r, **change) if len(calls) == 2 else r              # the first call is the input, the second the weld
        return report
    with monkeypatch.context() as mp:
        mp.setattr(ts, "surface_report", patched(fold_edges=4))
        with pytest.raises(ts._WeldFailed, match=r"^the welded surface has more fold edges \(4\) than the input \(3\)$"):
            ts.weld_short_edges(V, F, LIMIT)
    with monkeypatch.context() as mp:
        mp.setattr(ts, "surface_report", patched(volume_mm3=0.0))
        with pytest.raises(ts._WeldFailed, match="^the volume of the welded surface is not positive$"):
            ts.weld_short_edges(V, F, LIMIT)


def test_a_refusal_says_where_the_surface_is_not_a_closed_manifold():
    sph = trimesh.creation.icosphere(subdivisions=1, radius=5.0)
    V, F = np.array(sph.vertices), np.array(sph.faces)
    opened = F[1:]                                                          # one face gone: three open edges
    rep = ts.surface_report(V, opened)
    text = ts._where_not_manifold(V, opened, rep)
    assert text.startswith("; first open edge at (") and ts.surface_report(V, F)["closed_manifold"]
    mids = [0.5 * (V[F[0, i]] + V[F[0, (i + 1) % 3]]) for i in range(3)]      # the three open edges are the edges of the missing face
    assert any(f"({', '.join(f'{x:.4f}' for x in m)})" in text for m in mids)
    extra = np.vstack([F, [F[0, 0], F[0, 1], len(V) - 1]])                  # a third face on an edge
    assert "; first non-manifold edge at (" in ts._where_not_manifold(V, extra, ts.surface_report(V, extra))
    assert ts._where_not_manifold(V, F, ts.surface_report(V, F)) == ""
