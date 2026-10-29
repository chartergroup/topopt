"""Regions, loads and supports of `topopt verify` (CONTRACTS.md section 16, "Region rule" and "Traction rule").

A box mesh of 1 mm cubes and region STLs made with trimesh. No gmsh needed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_skfem  # noqa: E402
from verify_mesh import box_mesh  # noqa: E402

pytestmark = needs_skfem

CELLS = (4, 3, 2)


@pytest.fixture(scope="module")
def model():
    from topopt.verify.p2 import P2Model
    p, t = box_mesh(CELLS, CELLS)                      # 4 x 3 x 2 mm, 1 mm cubes
    return P2Model(p, t)


@pytest.fixture(scope="module")
def bfacets(model):
    from topopt.verify.regions import boundary_facets
    return boundary_facets(model.t.T)


def region_stl(tmp_path, name, lo, hi):
    import trimesh
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    path = tmp_path / f"{name}.stl"
    box.export(str(path))
    return path


def test_boundary_facets_are_the_outer_surface(model, bfacets):
    from topopt.verify.regions import facet_areas
    assert len(bfacets) == 2 * 2 * (4 * 3 + 3 * 2 + 4 * 2)           # 2 triangles per outer square
    assert len({tuple(f) for f in bfacets}) == len(bfacets)
    assert facet_areas(model.p.T, bfacets).sum() == pytest.approx(2 * (12 + 6 + 8))
    on_surface = np.any([(model.p[a][bfacets] < 1e-9).all(axis=1) | (model.p[a][bfacets] > CELLS[a] - 1e-9).all(axis=1)
                         for a in range(3)], axis=0)
    assert on_surface.all()


def test_fixed_region_takes_the_nodes_inside_or_within_the_band(model, bfacets, tmp_path):
    from topopt.verify.regions import select_region
    path = region_stl(tmp_path, "wall", (-1, -1, -1), (0.5, 4, 3))
    X = model.node_xyz
    for band in (0.1, 0.5, 1.0):
        sel = select_region(model, bfacets, path, "wall", "fixed", band)
        np.testing.assert_array_equal(sel.node_mask, X[0] <= 0.5 + band + 1e-9)
        assert sel.facts["nodes"] == int(sel.node_mask.sum()) > 0
        assert sel.facts["band_mm"] == band
    # the band matters: a node at x = 1 (0.5 outside the region wall) is in for band 0.5 and out for band 0.4
    on_x1 = np.isclose(X[0], 1.0)
    assert select_region(model, bfacets, path, "wall", "fixed", 0.5).node_mask[on_x1].all()
    assert not select_region(model, bfacets, path, "wall", "fixed", 0.4).node_mask[on_x1].any()


def test_report_has_counts_at_0_half_and_one_band(model, bfacets, tmp_path):
    from topopt.verify.regions import select_region
    path = region_stl(tmp_path, "wall", (-1, -1, -1), (0.5, 4, 3))
    sel = select_region(model, bfacets, path, "wall", "fixed", 1.0)
    counts = sel.facts["counts_by_band"]
    assert set(counts) == {"0", "0.5x", "1x"}
    n = [counts[k]["nodes"] for k in ("0", "0.5x", "1x")]
    assert n[0] < n[1] <= n[2] and n[2] == sel.facts["nodes"]
    X = model.node_xyz
    assert n[0] == int((X[0] <= 0.5 + 1e-9).sum())
    assert n[1] == int((X[0] <= 1.0 + 1e-9).sum())
    for key in ("region", "role", "band_mm", "nodes", "vertices", "facets", "area_mm2", "centroid_mm"):
        assert key in sel.facts


def test_load_region_takes_the_boundary_facets_with_all_3_vertices_in_the_band(model, bfacets, tmp_path):
    from topopt.verify.regions import select_region
    path = region_stl(tmp_path, "tip", (3.5, -1, -1), (6, 4, 3))
    sel = select_region(model, bfacets, path, "tip", "load", 0.1)
    assert len(sel.facets) == 2 * 3 * 2                               # the 6 squares of the end face x = 4
    assert sel.facts["area_mm2"] == pytest.approx(6.0)
    assert sel.facts["centroid_mm"] == pytest.approx([4.0, 1.5, 1.0])
    assert np.all(model.p[0][sel.facets] > 3.9)
    wide = select_region(model, bfacets, path, "tip", "load", 0.6)    # the band reaches x = 2.9: the last cell column
    assert len(wide.facets) > len(sel.facets)
    assert np.all(model.p[0][wide.facets] > 2.9)


def test_traction_resultant_equals_the_force(model, bfacets, tmp_path):
    from topopt.verify import bc
    from topopt.verify.regions import select_region
    path = region_stl(tmp_path, "tip", (3.5, -1, -1), (6, 4, 3))
    sel = select_region(model, bfacets, path, "tip", "load", 0.1)
    for F in ([0.0, 0.0, -1.96], [20.89, 5.21, -20.89], [-3.3, 1e-3, 7.1]):
        f, info = bc.traction_load(model, sel.facets, F)
        res = f.reshape(-1, 3).sum(axis=0)
        assert bc.resultant_error(res, F) <= 1e-10
        assert bc.resultant_error(info["resultant_N"], F) <= 1e-10
        assert info["area_mm2"] == pytest.approx(6.0)
        assert info["traction_MPa"] == pytest.approx(np.array(F) / 6.0)
    # uniform per area: the force goes to the edge midpoints only, and each facet gets its area share
    nz = np.flatnonzero(np.abs(f.reshape(-1, 3)).sum(axis=1))
    assert nz.min() >= model.nv                                        # no vertex node carries load
    assert len(nz) == len(np.unique(np.concatenate([
        model.edge_node_ids(sel.facets[:, 0], sel.facets[:, 1]), model.edge_node_ids(sel.facets[:, 1], sel.facets[:, 2]),
        model.edge_node_ids(sel.facets[:, 0], sel.facets[:, 2])])))


def test_traction_moment_equals_the_moment_of_a_uniform_traction(model, bfacets, tmp_path):
    """Uniform traction on the end face x = 4: the moment about the face centre is zero."""
    from topopt.verify import bc
    from topopt.verify.regions import select_region
    path = region_stl(tmp_path, "tip", (3.5, -1, -1), (6, 4, 3))
    sel = select_region(model, bfacets, path, "tip", "load", 0.1)
    f, _ = bc.traction_load(model, sel.facets, [0.0, 0.0, -5.0])
    X = model.node_xyz.T
    M = np.cross(X - np.array([4.0, 1.5, 1.0]), f.reshape(-1, 3)).sum(axis=0)
    assert np.abs(M).max() < 1e-12


def test_fixed_dofs_are_all_three_dofs_of_the_nodes():
    from topopt.verify.bc import fixed_dofs
    d = fixed_dofs(np.array([5, 2, 5]))
    np.testing.assert_array_equal(d, [6, 7, 8, 15, 16, 17])


def test_empty_fixed_region_names_the_region_and_the_nearest_distance(model, bfacets, tmp_path):
    from topopt.verify.errors import EmptyRegion, VerifyError
    from topopt.verify.regions import select_region
    path = region_stl(tmp_path, "far_wall", (10, -1, -1), (11, 4, 3))
    with pytest.raises(EmptyRegion) as e:
        select_region(model, bfacets, path, "far_wall", "fixed", 0.6)
    assert isinstance(e.value, VerifyError)
    assert "far_wall" in str(e.value) and "6" in str(e.value) and "0.6" in str(e.value)
    assert e.value.facts["nearest_distance_mm"] == pytest.approx(6.0, abs=1e-6)
    assert e.value.facts["region"] == "far_wall" and e.value.facts["role"] == "fixed"


def test_empty_load_region_names_the_region_and_the_nearest_distance(model, bfacets, tmp_path):
    from topopt.verify.errors import EmptyRegion
    from topopt.verify.regions import select_region
    path = region_stl(tmp_path, "far_tip", (7, -1, -1), (9, 4, 3))
    with pytest.raises(EmptyRegion) as e:
        select_region(model, bfacets, path, "far_tip", "load", 0.6)
    assert "far_tip" in str(e.value) and "no boundary facet" in str(e.value)
    assert e.value.facts["nearest_distance_mm"] == pytest.approx(3.0, abs=1e-6)


def test_load_region_inside_the_part_has_nodes_but_no_facets(model, bfacets, tmp_path):
    from topopt.verify.errors import EmptyRegion
    from topopt.verify.regions import select_region
    path = region_stl(tmp_path, "inside", (1.2, 1.2, 0.6), (2.8, 1.8, 1.4))
    with pytest.raises(EmptyRegion):
        select_region(model, bfacets, path, "inside", "load", 0.1)
    assert select_region(model, bfacets, path, "inside", "fixed", 0.1).facts["nodes"] > 0


def test_equilibrium_balances_load_and_reaction(model, bfacets, tmp_path):
    from topopt.verify import bc
    from topopt.verify.factor import RefinedFactor, make_factor
    from topopt.verify.regions import select_region
    from topopt.verify.factor import factor_availability
    avail = [n for n, v in factor_availability().items() if v["available"]]
    if not avail:
        pytest.skip("no factor can run here")
    wall = select_region(model, bfacets, region_stl(tmp_path, "wall", (-1, -1, -1), (0.2, 4, 3)), "wall", "fixed", 0.1)
    tip = select_region(model, bfacets, region_stl(tmp_path, "tip", (3.5, -1, -1), (6, 4, 3)), "tip", "load", 0.1)
    fixed = bc.fixed_dofs(np.flatnonzero(wall.node_mask))
    free = np.setdiff1d(np.arange(model.N), fixed)
    f, _ = bc.traction_load(model, tip.facets, [1.0, -2.0, 3.0])
    K = model.stiffness(3500.0, 0.3)
    Kff = K[free][:, free]
    rf = RefinedFactor(Kff, make_factor(Kff, avail[0]), steps=3)
    u = np.zeros(model.N)
    u[free] = rf.solve(f[free])
    eq = bc.equilibrium(model, K, u, f, fixed)
    assert eq["force_relative_residual"] < 1e-10
    assert eq["moment_relative_residual"] < 1e-10
    assert eq["reaction_force_N"] == pytest.approx([-1.0, 2.0, -3.0], abs=1e-9)


def test_equilibrium_cannot_find_a_wrong_load(model, bfacets, tmp_path):
    """Twice the load gives the same small residual: the check tests K and the solve, not the load (see bc.py)."""
    from topopt.verify import bc
    from topopt.verify.factor import RefinedFactor, factor_availability, make_factor
    from topopt.verify.regions import select_region
    avail = [n for n, v in factor_availability().items() if v["available"]]
    if not avail:
        pytest.skip("no factor can run here")
    wall = select_region(model, bfacets, region_stl(tmp_path, "wall", (-1, -1, -1), (0.2, 4, 3)), "wall", "fixed", 0.1)
    tip = select_region(model, bfacets, region_stl(tmp_path, "tip", (3.5, -1, -1), (6, 4, 3)), "tip", "load", 0.1)
    fixed = bc.fixed_dofs(np.flatnonzero(wall.node_mask))
    free = np.setdiff1d(np.arange(model.N), fixed)
    K = model.stiffness(3500.0, 0.3)
    Kff = K[free][:, free]
    rf = RefinedFactor(Kff, make_factor(Kff, avail[0]), steps=3)
    res = []
    for scale in (1.0, 2.0):
        f, info = bc.traction_load(model, tip.facets, [1.0, -2.0, 3.0])
        f = f * scale
        u = np.zeros(model.N)
        u[free] = rf.solve(f[free])
        res.append(bc.equilibrium(model, K, u, f, fixed)["force_relative_residual"])
        # the check that does see the load: the resultant of the nodal forces against the YAML force
        assert bc.resultant_error(f.reshape(-1, 3).sum(axis=0), np.array([1.0, -2.0, 3.0])) == pytest.approx(scale - 1.0, abs=1e-12)
    assert max(res) < 1e-10                      # both are "fine"


# --------------------------------------------------------------------------- #
# The traction against scikit-fem's FacetBasis on a jittered mesh with facets of different area (review m6)
# --------------------------------------------------------------------------- #

def _jittered_model():
    from topopt.verify.p2 import P2Model
    p, t = box_mesh((5.0, 3.0, 2.0), (5, 3, 2))
    p = p + 0.12 * np.random.default_rng(5).uniform(-1.0, 1.0, p.shape)
    return P2Model(p, t)


def _skfem_traction(m, facets, tr):
    from skfem import ElementTetP2, ElementVector, FacetBasis, LinearForm, asm
    fs = np.sort(m.mesh.facets, axis=0).T
    index = {tuple(r): i for i, r in enumerate(fs)}
    fidx = np.array([index[tuple(r)] for r in np.sort(facets, axis=1)])
    fb = FacetBasis(m.mesh, ElementVector(ElementTetP2()), facets=fidx, intorder=4)

    @LinearForm
    def load(v, w):
        return tr[0] * v[0] + tr[1] * v[1] + tr[2] * v[2]
    return asm(load, fb)


@pytest.mark.parametrize("which", ["end face", "two sides", "all"])
def test_traction_equals_the_scikit_fem_facet_integral_with_unequal_facets(which):
    from topopt.verify import bc, regions
    m = _jittered_model()
    bf = regions.boundary_facets(m.t.T)
    cx = m.p.T[bf].mean(axis=1)
    sel = {"end face": cx[:, 0] > 4.5, "two sides": (cx[:, 0] > 4.5) | (cx[:, 2] > 1.5), "all": np.ones(len(bf), bool)}[which]
    facets = bf[sel]
    a = regions.facet_areas(m.p.T, facets)
    assert a.max() / a.min() > 1.5                                  # the facets are not alike
    F = np.array([1.3, -2.1, 0.7])
    f, info = bc.traction_load(m, facets, F)
    ref = _skfem_traction(m, facets, F / info["area_mm2"])
    assert np.abs(f - ref).max() < 1e-12 * max(1.0, np.abs(ref).max())
    # the resultant and the moment about the origin are exact: P2 reproduces the linear field x
    c = m.p.T[facets].mean(axis=1)
    M_exact = np.cross((a[:, None] * c).sum(axis=0), F / info["area_mm2"])
    M_nodal = np.cross(m.node_xyz.T, f.reshape(-1, 3)).sum(axis=0)
    assert np.abs(M_nodal - M_exact).max() < 1e-12 * np.abs(M_exact).max()
    assert bc.resultant_error(f.reshape(-1, 3).sum(axis=0), F) < 1e-14
    # the test can see the typical bug: an equal split F / (3 k) per facet instead of F a / (3 A)
    wrong = np.zeros(m.N)
    ids = np.stack([m.edge_node_ids(facets[:, 0], facets[:, 1]), m.edge_node_ids(facets[:, 1], facets[:, 2]),
                    m.edge_node_ids(facets[:, 0], facets[:, 2])], axis=1)
    for k in range(3):
        np.add.at(wrong, 3 * ids.ravel() + k, np.repeat(np.full(len(facets), F[k] / (3 * len(facets))), 3))
    assert np.abs(wrong - ref).max() > 1e-3


# --------------------------------------------------------------------------- #
# Normals, near-region masks, the load centre (review M2, m7)
# --------------------------------------------------------------------------- #

def test_outward_normals_and_normal_coherence(model):
    from topopt.verify.regions import boundary_facets, facet_areas, normal_coherence, outward_normals
    bf, owner = boundary_facets(model.t.T, return_elems=True)
    assert len(owner) == len(bf)
    n = outward_normals(model.p.T, model.t.T, bf, owner)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    end = np.all(model.p[0][bf] > CELLS[0] - 1e-9, axis=1)
    np.testing.assert_allclose(n[end], np.tile([1.0, 0.0, 0.0], (end.sum(), 1)), atol=1e-12)
    start = np.all(model.p[0][bf] < 1e-9, axis=1)
    np.testing.assert_allclose(n[start], np.tile([-1.0, 0.0, 0.0], (start.sum(), 1)), atol=1e-12)
    a = facet_areas(model.p.T, bf)
    assert normal_coherence(n[end], a[end])["coherence"] == pytest.approx(1.0)
    assert normal_coherence(n[end], a[end])["mean_normal"] == pytest.approx([1.0, 0.0, 0.0])
    both = end | start                                                # two opposite faces of the same area
    assert normal_coherence(n[both], a[both])["coherence"] < 1e-12
    assert normal_coherence(n, a)["coherence"] < 1e-12                # a closed surface cancels
    assert normal_coherence(n[:0], a[:0])["coherence"] is None


def test_load_region_gets_a_normal_check_and_a_band_that_reaches_the_far_face_shows_it(tmp_path):
    """A thin plate 0.8 mm thick: a band of 1 mm takes facets of both faces, and the coherence falls (review m7)."""
    from topopt.verify.p2 import P2Model
    from topopt.verify.regions import boundary_facets, select_region
    p, t = box_mesh((4.0, 3.0, 0.8), (4, 3, 1))
    m = P2Model(p, t)
    bf, owner = boundary_facets(m.t.T, return_elems=True)
    # a region that covers the top face z = 0.8 only
    top = region_stl(tmp_path, "top", (-1, -1, 0.7), (5, 4, 2))
    tight = select_region(m, bf, top, "top", "load", 0.01, belems=owner)
    wide = select_region(m, bf, top, "top", "load", 1.0, belems=owner)
    assert tight.facts["normal_check"]["coherence"] == pytest.approx(1.0)
    assert wide.facts["normal_check"]["coherence"] < 0.2
    assert wide.facts["facets"] > tight.facts["facets"]


def test_near_region_mask_follows_the_distance_to_the_region(model, bfacets, tmp_path):
    from topopt.verify.p2 import STRESS_ORDER, quadrature
    from topopt.verify.regions import near_region_mask, select_region
    path = region_stl(tmp_path, "tip", (3.0, -1, -1), (6, 4, 3))       # the region is x >= 3 (inside the part: no band reach needed)
    sel = select_region(model, bfacets, path, "tip", "fixed", 0.1)
    qp = model.qp_points(quadrature(STRESS_ORDER)[0])
    for d in (0.4, 1.0):
        mask = near_region_mask(model, sel, qp, d)
        assert mask.shape == qp.shape[:2]
        x = qp[..., 0]
        dist = np.maximum(3.0 - x, 0.0)                              # the exact distance to the box (the other sides are far)
        assert mask[dist <= d - 0.2].all()
        # a tet with a node inside the region counts as inside, so a mask can reach a little farther than d
        far_tets = (model.node_xyz[0][model.enodes].max(axis=1) < 3.0 - d - 0.2)
        assert not mask[far_tets].any()


def test_near_region_stats_gives_the_p99_9_without_the_patch_and_the_shares():
    from topopt.verify.stats import near_region_stats
    n = 1000
    vm = np.tile(np.linspace(1.0, 10.0, n)[:, None], (1, 2))          # 2000 points, volume weight 1 each
    wv = np.ones_like(vm)
    near_a = np.zeros(vm.shape, bool)
    near_a[900:, :] = True                                             # the highest tets are near region a
    near_b = np.zeros(vm.shape, bool)
    near_b[:100, :] = True                                             # the lowest are near b
    p999 = float(np.sort(vm.ravel())[int(np.ceil(0.999 * vm.size)) - 1])
    rec = near_region_stats(vm, wv, p999, {"a": near_a, "b": near_b}, 2.0)
    assert rec["distance_mm"] == 2.0 and rec["regions"] == ["a", "b"]
    assert rec["volume_fraction_excluded"] == pytest.approx(0.2)
    assert rec["top_volume_share_near"]["a"] == pytest.approx(1.0)     # the top 0.1 % is all in a
    assert rec["top_volume_share_near"]["b"] == pytest.approx(0.0)
    assert rec["top_volume_share_near"]["none"] == pytest.approx(0.0)
    assert rec["p99_9_by_volume_MPa_excluding_near_regions"] < p999     # the top points are gone
    assert rec["peak_qp_MPa_excluding_near_regions"] == pytest.approx(vm[899, 0])
    everything = near_region_stats(vm, wv, p999, {"all": np.ones(vm.shape, bool)}, 2.0)
    assert everything["p99_9_by_volume_MPa_excluding_near_regions"] is None


def test_load_centre_gives_the_distance_and_the_moment_ratio():
    from topopt.verify.bc import load_centre
    out = load_centre([10.0, 0.0, 1.0], [10.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0])
    assert out["distance_mm"] == pytest.approx(1.0)
    assert out["moment_ratio"] == pytest.approx(1.0)                   # the vertical offset does not change M = r x F for F along z
    out = load_centre([12.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0])
    assert out["distance_mm"] == pytest.approx(2.0) and out["moment_ratio"] == pytest.approx(1.2)
    none = load_centre([1.0, 0.0, 0.0], None, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0])
    assert none["optimizer_center_mm"] is None and none["moment_ratio"] is None
