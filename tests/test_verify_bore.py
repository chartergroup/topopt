"""The bore module of `topopt verify` (B4, plan section 4 and 10.3): wall facets, the funnel, the sample set and its statistics.

The tests use a structured ring (`annulus_mesh`, no gmsh): the inner wall is a polygon with a known area, so every count is known by hand.
"""
import math

import numpy as np
import pytest

from helpers import needs_skfem
from verify_mesh import annulus_mesh

pytestmark = needs_skfem

A, B, L = 21.9875, 25.25, 30.0
N_R, N_T, N_Z = 2, 72, 8
E, NU = 197000.0, 0.28


def spec(radius=A, length=L, point=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), name="ring"):
    from topopt.config import BoreSpec
    return BoreSpec(name, point, axis, radius, length, "yaml")


def knobs(**kw):
    from topopt.config import BoreKnobs
    return BoreKnobs(**kw)


@pytest.fixture(scope="module")
def ring():
    from topopt.verify.p2 import P2Model
    p, t = annulus_mesh(A, B, L, N_R, N_T, N_Z)
    return P2Model(p, t)


def test_the_wall_facets_of_a_full_ring(ring):
    from topopt.verify import bore
    g = bore.find_facets(ring.mesh, spec(), knobs())
    n_wall = N_T * N_Z * 2
    assert g["facets"] == n_wall == len(g["ids"]) == 1152
    # the polygon area: n_t chords of length 2 a sin(pi/n_t), each over the full length
    assert g["area_mm2"] == pytest.approx(L * N_T * 2 * A * math.sin(math.pi / N_T), rel=1e-12)
    n_boundary = 2 * n_wall + 2 * (N_R * N_T * 2)                       # inner wall, outer wall, two end faces
    assert g["funnel"] == dict(boundary=n_boundary, radius=n_wall, axial=n_wall, normal=n_wall, side=n_wall)
    assert g["analytic_area_mm2"] == pytest.approx(2 * math.pi * A * (L - 0.7)) and g["analytic_area_full_mm2"] == pytest.approx(2 * math.pi * A * L)
    assert g["area_over_analytic"] == pytest.approx(g["area_mm2"] / g["analytic_area_mm2"])
    assert g["vertex_radius_mm"] == pytest.approx([A, A], rel=1e-12) and g["largest_vertex_deviation_mm"] < 1e-12
    assert g["vertex_s_range_mm"] == pytest.approx([0.0, L]) and g["mean_facet_edge_mm"] > 0
    assert g["near_miss"] == dict(facets=0, area_mm2=0.0, deviation_mm=None)
    assert g["closes_cylinder"] is True and g["largest_angular_gap_deg"] < g["largest_facet_width_deg"] <= 2 * 360 / N_T
    # the facets are on the wall: the vertices are at radius a and no facet is on the outer wall or an end face
    xyz = ring.mesh.p[:, ring.mesh.facets[:, g["ids"]]]
    assert np.allclose(np.hypot(xyz[0], xyz[1]), A, rtol=0, atol=1e-12)


def test_end_margin_drops_the_facets_near_the_ends(ring):
    from topopt.verify import bore
    dz = L / N_Z
    # the centroids of the two triangles of a quad are at z0 + dz/3 and z0 + 2 dz/3
    zc = np.array([k * dz + f * dz for k in range(N_Z) for f in (1 / 3, 2 / 3)])
    for margin in (0.0, 1.5, 2.0):
        want = int(((zc >= margin) & (zc <= L - margin)).sum()) * N_T
        got = bore.find_facets(ring.mesh, spec(), knobs(end_margin_mm=margin))["facets"]
        assert got == want, (margin, got, want)
    assert want < N_T * N_Z * 2


def test_radius_tolerance_and_the_radius(ring):
    from topopt.verify import bore
    g = bore.find_facets(ring.mesh, spec(radius=A + 0.5), knobs(radius_tolerance_mm=0.5 - 1e-9))
    assert g["facets"] == 0 and g["funnel"]["radius"] == 0 and g["funnel"]["side"] == 0
    assert bore.find_facets(ring.mesh, spec(radius=A + 0.5), knobs(radius_tolerance_mm=0.5 + 1e-9))["facets"] == 1152      # the test is "<="


def test_normal_limit_zero_gives_no_facet(ring):
    """`bore.py` is called directly: the YAML needs a normal limit above 0, but the rule `|n . axis| < limit` is checked here (REVIEW n2)."""
    from topopt.verify import bore
    g = bore.find_facets(ring.mesh, spec(), knobs(normal_limit=0.0))
    assert g["facets"] == 0 and g["funnel"]["axial"] == 1152 and g["funnel"]["normal"] == 0


def test_a_wrong_point_gives_no_facet_and_the_funnel_says_which_test_removed_them(ring):
    from topopt.verify import bore
    b = bore.build_bore(ring, spec(point=(50.0, 0.0, 0.0)), knobs())
    assert not b.ok and b.entry["status"] == "empty" and b.entry["facets"] == 0
    assert b.entry["funnel"]["radius"] == 0 and b.entry["funnel"]["boundary"] > 0
    assert "0 within the radius tolerance" in b.entry["facts"][0]
    assert any("No boundary facet of the mesh passes the four tests of the bore 'ring'" in f for f in b.entry["facts"])
    assert any("--mesh-size" in o or "mesh_size" in o or "mesh size" in o for o in b.entry["options"])


def test_the_outer_wall_of_the_same_radius_kind_is_not_taken(ring):
    """A boss or a shaft: the facets at radius b have the outward normal away from the axis, so test 4 removes them (REVIEW m5)."""
    from topopt.verify import bore
    g = bore.find_facets(ring.mesh, spec(radius=B), knobs())
    f = g["funnel"]
    assert (f["radius"], f["axial"], f["normal"], f["side"]) == (1152, 1152, 1152, 0) and g["facets"] == 0


def test_a_half_ring_does_not_close_a_cylinder_and_the_full_ring_does(ring):
    from topopt.verify import bore
    from topopt.verify.p2 import P2Model
    p, t = annulus_mesh(A, B, L, N_R, 36, N_Z, sweep=np.pi)
    half = P2Model(p, t)
    g = bore.find_facets(half.mesh, spec(), knobs())
    assert g["facets"] == 36 * N_Z * 2 and g["closes_cylinder"] is False and g["largest_angular_gap_deg"] > 180.0
    b = bore.build_bore(half, spec(), knobs())
    assert any("do not close a cylinder" in f for f in b.entry["facts"]) and b.ok
    full = bore.build_bore(ring, spec(), knobs())
    assert not any("do not close a cylinder" in f for f in full.entry["facts"])


def test_one_column_missing_gives_the_note(ring):
    from topopt.verify import bore
    from topopt.verify.p2 import P2Model
    p, t = annulus_mesh(A, B, L, N_R, N_T, N_Z)
    cen = p[:, t].mean(axis=1)
    ang = np.arctan2(cen[1], cen[0]) % (2 * np.pi)
    keep = ~((ang > 0.0) & (ang < 2 * np.pi / N_T))             # the cells of the first angular column
    g = bore.find_facets(P2Model(p, t[:, keep]).mesh, spec(), knobs())
    assert g["facets"] == (N_T - 1) * N_Z * 2 and g["closes_cylinder"] is False
    assert 2 * 360 / N_T * 0.7 < g["largest_angular_gap_deg"] < 2 * 360 / N_T * 1.2


def test_near_miss_facets_are_counted_and_named(ring):
    """One row of wall vertices moved 0.6 mm inward: more than the tolerance (0.35) and less than the band (1.2)."""
    from topopt.verify import bore
    from topopt.verify.p2 import P2Model
    p, t = annulus_mesh(A, B, L, N_R, N_T, N_Z)
    row = (np.abs(np.hypot(p[0], p[1]) - A) < 1e-9) & (np.abs(p[2] - L / N_Z) < 1e-9)
    p[:2, row] *= (A - 0.6) / A
    m = P2Model(p, t)
    b = bore.build_bore(m, spec(), knobs())
    nm = b.entry["near_miss"]
    assert nm["facets"] == 4 * N_T                     # the facets of the two quad rows that touch the moved vertices
    assert nm["deviation_mm"] == pytest.approx([0.6, 0.6], abs=1e-12) and nm["area_mm2"] > 0
    assert b.entry["facets"] == 1152 - 4 * N_T
    assert any("fail the radius tolerance (deviation 0.600 to 0.600 mm)" in f for f in b.entry["facts"])
    assert any("larger radius_tolerance_mm" in o for o in b.entry["options"])
    # a tolerance above the deviation takes them into the set; no near miss is left
    g = bore.find_facets(m.mesh, spec(), knobs(radius_tolerance_mm=0.7))
    assert g["facets"] == 1152 and g["near_miss"]["facets"] == 0


def test_the_sample_set_is_one_tet_layer_and_its_size_is_the_geometry(ring):
    from topopt.verify import bore
    from topopt.verify.p2 import STRESS_ORDER, quadrature
    b = bore.build_bore(ring, spec(), knobs())
    s = b.sample
    assert s.tets == len(b.ids) == 1152                                 # one tet owns each wall facet (REVIEW n1)
    own = np.unique(ring.mesh.f2t[0, b.ids])
    lam, w = quadrature(STRESS_ORDER)
    X = ring.qp_points(lam, own)
    r = np.hypot(X[..., 0], X[..., 1])
    near = np.abs(r - A) < 1.2
    assert s.n_qp == int(near.sum()) > 0
    assert s.volume_mm3 == pytest.approx(float((w[None, :] * ring.vol[own][:, None])[near].sum()), rel=1e-12)
    assert b.entry["sample_set"] == dict(tets=1152, points=s.n_qp, volume_mm3=s.volume_mm3, band_mm=1.2)
    # a band that holds no point (the test is "<", so a band of 0 holds none): the bore is dropped with a fact
    none = bore.build_bore(ring, spec(), knobs(sample_band_mm=0.0))
    assert not none.ok and none.entry["status"] == "empty" and none.entry["sample_set"] is None
    assert any("No quadrature point" in f for f in none.entry["facts"]) and none.entry["facets"] == 1152


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def test_statistics_of_an_axial_stretch_are_exact(ring):
    """u_z = eps z is exact in P2: the stress is uniform. hoop = lambda eps, von Mises = 2 mu eps, so all statistics are equal."""
    from topopt.verify import bore
    eps = 1e-3
    u = np.zeros(ring.N)
    u[2::3] = eps * ring.node_xyz[2]
    sig, _, _ = ring.stress_qp(u, E, NU)
    lam_, mu = lame(E, NU)
    st = bore.build_bore(ring, spec(), knobs()).sample.stats(sig)
    assert st["n_qp"] > 0
    for k in ("p99_9", "max", "median", "min"):
        assert st[f"hoop_{k}_MPa"] == pytest.approx(lam_ * eps, rel=1e-10)
        assert st[f"vm_{k}_MPa"] == pytest.approx(2 * mu * eps, rel=1e-10)


def test_statistics_of_a_radial_stretch_are_exact(ring):
    """u = eps (x, y, 0): hoop = 2 (lambda + mu) eps and von Mises = 2 mu eps (sigma_zz = 2 lambda eps)."""
    from topopt.verify import bore
    eps = 1e-3
    u = np.zeros(ring.N)
    u[0::3] = eps * ring.node_xyz[0]
    u[1::3] = eps * ring.node_xyz[1]
    sig, _, _ = ring.stress_qp(u, E, NU)
    lam_, mu = lame(E, NU)
    st = bore.build_bore(ring, spec(), knobs()).sample.stats(sig)
    assert st["hoop_p99_9_MPa"] == pytest.approx(2 * (lam_ + mu) * eps, rel=1e-10) and st["hoop_min_MPa"] == pytest.approx(st["hoop_max_MPa"], rel=1e-10)
    assert st["vm_max_MPa"] == pytest.approx(2 * mu * eps, rel=1e-10)


TILT_POINT = np.array([390.0, 0.0, 617.0])
TILT_AXIS = np.array([0.30070579950427306, 0.0, -0.9537169507482269])


@pytest.fixture(scope="module")
def tilted_ring():
    """The ring of `ring`, turned so that its axis is TILT_AXIS and moved to TILT_POINT: the hoop vector has all three components."""
    from topopt.verify.p2 import P2Model
    p, t = annulus_mesh(A, B, L, N_R, N_T, N_Z)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, TILT_AXIS)
    c = z @ TILT_AXIS
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = np.eye(3) + K + K @ K * ((1 - c) / (v @ v))                  # Rodrigues: z -> TILT_AXIS
    return P2Model(np.ascontiguousarray(((rot @ p).T + TILT_POINT).T), t)


def test_the_hoop_stress_on_a_tilted_axis_is_the_tensor_product(tilted_ring):
    """Review m-3 of B4a: every other hoop test has a z-axis ring (e_t has no z part, so two of the three shear terms are 0).

    A random uniform stress on a ring with a tilted axis: `u = G x` is exact in P2, so the stress is the constant tensor `lambda tr(e) I + 2 mu e`
    with `e = sym(G)`. The reference is the full product `e_t . S . e_t` with `e_t = axis x e_r` at each point of the sample set, written with the
    tensor and not with the Voigt vector of the code (a wrong order of `xy`, `yz`, `xz` gives an error of tens of MPa here).
    """
    from topopt.verify import bore
    from topopt.verify.p2 import STRESS_ORDER, quadrature
    from topopt.verify.stats import von_mises, weighted_percentile
    m = tilted_ring
    G = np.random.default_rng(3).normal(size=(3, 3)) * 1e-3
    u = np.zeros(m.N)
    for i in range(3):
        u[i::3] = G[i] @ m.node_xyz
    sig, _, _ = m.stress_qp(u, E, NU)
    lam_, mu = lame(E, NU)
    eps = 0.5 * (G + G.T)
    S = lam_ * np.trace(eps) * np.eye(3) + 2.0 * mu * eps
    b = bore.build_bore(m, spec(point=tuple(TILT_POINT), axis=tuple(TILT_AXIS)), knobs())
    assert b.ok and b.entry["facets"] == 1152 and b.entry["closes_cylinder"] is True
    s = b.sample
    X = m.qp_points(quadrature(STRESS_ORDER)[0], s.elem)[np.arange(len(s.elem)), s.qp]          # the points of the sample set
    d = X - TILT_POINT
    er = d - np.outer(d @ TILT_AXIS, TILT_AXIS)
    er /= np.linalg.norm(er, axis=1)[:, None]
    et = np.cross(TILT_AXIS, er)
    hoop = np.einsum("ni,ij,nj->n", et, S, et)
    assert hoop.max() - hoop.min() > 50.0 and np.abs(et[:, 2]).max() > 0.2          # the case has an angle dependence and a z part of e_t
    st = s.stats(sig)
    w = s.weight
    ref = dict(zip(("p99_9", "median"), weighted_percentile(hoop, w, [99.9, 50.0])))
    assert st["hoop_max_MPa"] == pytest.approx(hoop.max(), abs=1e-9) and st["hoop_min_MPa"] == pytest.approx(hoop.min(), abs=1e-9)
    assert st["hoop_p99_9_MPa"] == pytest.approx(ref["p99_9"], abs=1e-9) and st["hoop_median_MPa"] == pytest.approx(ref["median"], abs=1e-9)
    vm = float(von_mises(np.array([[S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[1, 2], S[0, 2]]]))[0])
    for k in ("p99_9", "max", "median", "min"):
        assert st[f"vm_{k}_MPa"] == pytest.approx(vm, rel=1e-10)                           # a uniform stress: one von Mises value


def test_p99_9_is_the_step_rule_of_the_weights(ring):
    """A stress that is 1 at one point of the set with the largest value and 0 elsewhere: p99.9 is that value only when the point carries 0.1 % of the volume."""
    from topopt.verify import bore
    from topopt.verify.stats import weighted_percentile
    s = bore.build_bore(ring, spec(), knobs()).sample
    sig = np.zeros((ring.nel, 14, 6))
    sig[s.elem[0], s.qp[0], 0] = 7.0                     # one point with sxx = 7: hoop = et_x^2 7, von Mises = 7
    st = s.stats(sig)
    assert st["vm_max_MPa"] == 7.0 and st["vm_min_MPa"] == 0.0
    vm = np.zeros(s.n_qp)
    vm[0] = 7.0
    assert st["vm_p99_9_MPa"] == float(weighted_percentile(vm, s.weight, [99.9])[0]) == 0.0      # the point carries far less than 0.1 % of the volume
    assert st["vm_median_MPa"] == 0.0


def test_the_entry_is_plain_json(ring):
    import json
    from topopt.verify import bore
    b = bore.build_bore(ring, spec(), knobs())
    text = json.dumps(b.entry)
    assert "ids" not in b.entry and json.loads(text)["status"] == "ok"
    assert json.loads(text)["geometry_from"] == "yaml" and json.loads(text)["funnel"]["side"] == 1152


def test_a_bore_whose_build_raises_is_dropped_with_a_fact_and_the_others_go_on(ring, monkeypatch):
    """Review m-5 of B4a: a failure in the build of one bore is a fact in its entry (the bore, the error type, the text, an option); the next bore is built."""
    import json
    from topopt.config import BoreChecks
    from topopt.verify import bore
    real = bore.build_bore

    def build_bore(model, spec_, knobs_):
        if spec_.name == "bad":
            raise RuntimeError("a made up\nfailure")
        return real(model, spec_, knobs_)

    monkeypatch.setattr(bore, "build_bore", build_bore)
    good, bad, after = spec(name="good"), spec(name="bad"), spec(name="after", length=L / 2)
    out = bore.build(ring, BoreChecks((good, bad, after), knobs()))
    assert [b.spec.name for b in out] == ["good", "bad", "after"] and [b.ok for b in out] == [True, False, True]
    e = out[1].entry
    assert e["status"] == "error" and e["stage"] == "build" and e["message"] == "RuntimeError: a made up failure" and out[1].drop_reason == "error"
    assert "'bad'" in e["facts"][0] and "RuntimeError: a made up failure" in e["facts"][0] and "the other bores" in e["facts"][0] and e["options"]
    assert "facets" not in e and json.loads(json.dumps(e)) == e                  # an entry of a failed bore has none of the funnel keys
    assert out[0].entry["facets"] == 1152 and out[2].entry["status"] == "ok"


def test_fail_drops_a_bore_after_the_build(ring):
    from topopt.verify import bore
    b = bore.build_bore(ring, spec(), knobs())
    assert b.ok and b.entry["status"] == "ok"
    b.fail(ValueError("late"), "stats")
    assert not b.ok and b.entry["status"] == "error" and b.entry["stage"] == "stats" and "statistics of the stress" in b.entry["facts"][0]
    assert b.entry["message"] == "ValueError: late" and b.drop_reason == "error"


def test_a_quadrature_point_on_the_axis_is_not_in_the_sample_set():
    """Review n-6 of B4a: a point on the axis has no radial direction, so its hoop vector would be 0/0. It happens only when the radius is below the band."""
    from topopt.verify import bore
    from topopt.verify.p2 import STRESS_ORDER, quadrature
    nq = len(quadrature(STRESS_ORDER)[1])

    class Mesh:
        f2t = np.array([[0], [-1]])

    class Model:
        mesh = Mesh()
        vol = np.array([2.0])

        def qp_points(self, lam, el):
            X = np.tile([1.0, 0.0, 5.0], (len(el), nq, 1))
            X[:, 0] = [0.0, 0.0, 5.0]                                  # the first point is on the axis
            X[:, 1:, 2] += np.arange(1, nq)                            # the others are at radius 1 along the axis
            return X

    s = bore.SampleSet(Model(), np.array([0]), spec(radius=1.0, length=20.0), 1.2)       # radius 1.0 is below the band 1.2: the axis is in the band
    assert s.n_qp == nq - 1 and 0 not in s.qp
    st = s.stats(np.ones((1, nq, 6)))
    assert all(np.isfinite(v) for v in st.values())


def test_the_percentile_constant():
    from topopt.verify import bore
    assert bore.BORE_PERCENTILE == 99.9
