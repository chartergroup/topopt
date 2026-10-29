"""The A5 rows (displacement and alignment limits) in `topopt verify` (B3, CONTRACTS.md sections 16 and 31).

The tet model here is a cantilever of P2 tets without gmsh (a structured mesh), with the region STLs of the voxel tests
(`test_optimize.make_case`). The tests with gmsh are in `test_verify_buckling_run.py`.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_skfem  # noqa: E402
from test_optimize import box, make_case  # noqa: E402
from verify_mesh import box_mesh  # noqa: E402

pytestmark = needs_skfem

E, NU = 3500.0, 0.3
BAND = 0.6

LIMITS = {
    "displacement": [
        {"name": "tipz", "region": "tip", "limit_mm": 0.01, "direction": [0, 0, -1], "cases": ["down"]},
        {"name": "tipn", "region": "tip", "limit_mm": 0.02},                                           # norm of the mean vector, both cases
        {"name": "midmax", "region": "mid", "limit_mm": 0.01, "direction": [0, 0, -1], "sense": "max", "cases": ["down"]},
        {"name": "midmin", "region": "mid", "limit_mm": 0.01, "direction": [0, 0, -1], "sense": "min", "cases": ["down"]},
    ],
    "alignment": [
        {"region_a": "a", "region_b": "b", "axis": [1, 0, 0], "shift_limit_mm": 0.002, "tilt_limit_rad": 5e-4, "cases": ["down"]},
    ],
}


def write_case(tmp_path, limits=LIMITS, extra_boxes=None):
    """The cantilever of `make_case` with the probe regions mid (a slab), a and b (slabs left and right of it)."""
    p = make_case(tmp_path, h=2.0, iterations=2)
    box(tmp_path / "mid.stl", (8, 0, 0), (12, 10, 10))
    box(tmp_path / "a.stl", (6, 0, 0), (8, 10, 10))
    box(tmp_path / "b.stl", (14, 0, 0), (16, 10, 10))
    measure = {"mid": "mid.stl", "a": "a.stl", "b": "b.stl"}
    for name, (lo, hi) in (extra_boxes or {}).items():
        box(tmp_path / f"{name}.stl", lo, hi)
        measure[name] = f"{name}.stl"
    d = yaml.safe_load(p.read_text())
    d["regions"]["measure"] = measure
    d["limits"] = limits
    d["material"] = {"E": E, "nu": NU}
    p.write_text(yaml.safe_dump(d))
    return p


class Cantilever:
    """A 20 x 10 x 10 cantilever of P2 tets, clamped at x = 0, with the loads of the cases `down` and `side` on the tip."""

    def __init__(self, tmp_path, limits=LIMITS, extra_boxes=None, solver="superlu"):
        from topopt.config import load_config
        from topopt.verify import bc, regions
        from topopt.verify.factor import make_factor
        from topopt.verify.p2 import P2Model
        self.cfg = load_config(write_case(tmp_path, limits, extra_boxes))
        p, t = box_mesh((20.0, 10.0, 10.0), (10, 5, 5))
        self.model = m = P2Model(p, t)
        self.X = m.node_xyz.T
        self.bfacets, belems = regions.boundary_facets(m.t.T, return_elems=True)
        rm_ = regions.load_region_mesh(self.cfg.regions.fixed[0])
        fixed_sel = regions.select_region(m, self.bfacets, self.cfg.regions.fixed[0], "plate", "fixed", BAND, region_mesh=rm_)
        tip = self.cfg.regions.load[0]
        self.load_sel = {tip.name: regions.select_region(m, self.bfacets, tip.path, tip.name, "load", BAND, belems=belems)}
        fixed = bc.fixed_dofs(np.flatnonzero(fixed_sel.node_mask))
        self.free = np.setdiff1d(np.arange(m.N), fixed)
        F = np.column_stack([bc.traction_load(m, self.load_sel["tip"].facets, fo.force)[0]
                             for case in self.cfg.load_cases for fo in case.forces[:1]])
        K = m.stiffness(E, NU)
        Kff = K[self.free][:, self.free].tocsr()
        fac = make_factor(Kff, solver)
        self.U = fac.solve(F[self.free])                          # (n_free, n_cases)
        fac.free()
        self.run_dir = tmp_path

    def u_all(self, j):
        u = np.zeros(self.model.N)
        u[self.free] = self.U[:, j]
        return u.reshape(-1, 3)

    def record(self, U=None, metrics=None):
        from topopt.verify.motion import motion_record
        if metrics is not None:
            (self.run_dir / "result").mkdir(exist_ok=True)
            (self.run_dir / "result" / "metrics.json").write_text(json.dumps(metrics))
        return motion_record(self.cfg, SimpleNamespace(run_dir=self.run_dir), self.model, self.free, self.load_sel,
                             self.bfacets, BAND, self.U if U is None else U)

    def field(self, f):
        """The free part of the displacement field `f(x) -> (3,)` at every node, for both cases: (n_free, 2)."""
        u = np.array([f(x) for x in self.X]).ravel()
        return np.column_stack([u[self.free]] * 2)


@pytest.fixture(scope="module")
def cantilever(tmp_path_factory):
    return Cantilever(tmp_path_factory.mktemp("cant"))


# --------------------------------------------------------------------------- #
# The role measure of select_region
# --------------------------------------------------------------------------- #

def test_select_region_measure_takes_the_nodes_inside_the_region_and_needs_no_facet(cantilever):
    from topopt.verify import regions
    m, cfg = cantilever.model, cantilever.cfg
    mid = next(mr for mr in cfg.regions.measure if mr.name == "mid")
    sel = regions.select_region(m, cantilever.bfacets, mid.path, "mid", "measure", BAND)
    nodes = np.flatnonzero(sel.node_mask)
    x = cantilever.X[nodes, 0]
    assert sel.role == "measure" and len(nodes) > 0 and x.min() >= 8.0 - BAND - 1e-9 and x.max() <= 12.0 + BAND + 1e-9
    assert sel.facts["nodes"] == len(nodes) and not sel.empty
    # the slab crosses the whole section: it holds interior nodes that no boundary facet has
    on_boundary = np.zeros(m.nn, bool)
    on_boundary[m.edge_node_ids(*np.sort(cantilever.bfacets[:, [0, 1]], axis=1).T)] = True
    assert (~on_boundary[nodes]).any()


def test_a_measure_region_outside_the_part_is_an_empty_region_with_the_distance(tmp_path):
    from topopt.verify import regions
    from topopt.verify.errors import EmptyRegion
    c = Cantilever(tmp_path, extra_boxes={"far": ((50, 0, 0), (54, 10, 10))})
    far = next(mr for mr in c.cfg.regions.measure if mr.name == "far")
    with pytest.raises(EmptyRegion, match=r"measure region 'far'.*matches no mesh node.*nearest mesh point is 30"):
        regions.select_region(c.model, c.bfacets, far.path, "far", "measure", BAND)


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #

def test_a_displacement_row_is_the_mean_over_the_p2_nodes_of_the_region(cantilever):
    rec = cantilever.record()
    assert rec["status"] == "done" and rec["kind"].startswith("displacement and alignment rows (A5)")
    by = {r["name"]: r for r in rec["rows"]}
    assert list(by) == ["displacement:tipz:down", "displacement:tipn:down", "displacement:tipn:side", "displacement_max:midmax:down",
                        "displacement_min:midmin:down", "shift:a-b:down", "tilt:a-b:down"]
    nodes = np.flatnonzero(cantilever.load_sel["tip"].node_mask)
    down, side = cantilever.u_all(0), cantilever.u_all(1)
    row = by["displacement:tipz:down"]
    assert row["tet_value"] == pytest.approx(abs(down[nodes].mean(axis=0) @ [0, 0, -1]), rel=1e-12)
    assert row["tet_value"] > 0 and row["unit"] == "mm" and row["kind"] == "displacement" and row["case"] == "down"
    assert row["limit"] == 0.01 and row["tet_over_limit"] == pytest.approx(row["tet_value"] / 0.01, rel=1e-14)
    assert by["displacement:tipn:down"]["tet_value"] == pytest.approx(np.linalg.norm(down[nodes].mean(axis=0)), rel=1e-12)
    assert by["displacement:tipn:side"]["tet_value"] == pytest.approx(np.linalg.norm(side[nodes].mean(axis=0)), rel=1e-12)
    r = rec["regions"]["tip"]
    assert r["nodes"] == len(nodes) and r["vertices"] == int(np.sum(nodes < cantilever.model.nv)) and r["fixed_nodes"] == 0
    assert r["role"] == "load region" and len(r["centre_mm"]) == 3
    assert "equal-weight mean over the P2 nodes" in rec["rule"] and rec["band_mm"] == BAND


def test_the_signs_of_max_and_min_rows_follow_a5(cantilever):
    by = {r["name"]: r for r in cantilever.record()["rows"]}
    y = by["displacement_max:midmax:down"]["tet_value"]            # y = the mean along -z of the slab
    assert by["displacement_min:midmin:down"]["tet_value"] == pytest.approx(-y, rel=1e-12)
    assert y > 0                                                    # a cantilever under -z moves down
    assert by["displacement_max:midmax:down"]["unit"] == "mm"


def test_a_rigid_motion_of_the_body_gives_no_shift_and_no_tilt(cantilever):
    t, theta, c = np.array([0.3, -0.2, 0.1]), np.array([1e-3, -2e-3, 3e-3]), np.array([10.0, 5.0, 5.0])
    U = cantilever.field(lambda x: t + np.cross(theta, x - c))
    rows = {r["name"]: r for r in cantilever.record(U)["rows"]}
    assert rows["shift:a-b:down"]["tet_value"] < 1e-12 and rows["tilt:a-b:down"]["tet_value"] < 1e-12


def test_a_relative_translation_of_the_two_regions_is_the_shift(cantilever):
    delta = 0.01
    U = cantilever.field(lambda x: np.array([0.0, delta, 0.0]) if x[0] < 10.0 else np.zeros(3))
    rows = {r["name"]: r for r in cantilever.record(U)["rows"]}
    assert rows["shift:a-b:down"]["tet_value"] == pytest.approx(delta, rel=1e-12)
    assert rows["tilt:a-b:down"]["tet_value"] < 1e-12
    assert rows["shift:a-b:down"]["tet_over_limit"] == pytest.approx(delta / 0.002, rel=1e-12)


def test_a_relative_rotation_of_the_two_regions_is_the_tilt(cantilever):
    phi = 4e-4                                                    # rotation about y of the left half only, about its own centre of region a
    ca = np.array([7.0, 5.0, 5.0])
    U = cantilever.field(lambda x: np.cross([0.0, phi, 0.0], x - ca) if x[0] < 10.0 else np.zeros(3))
    rows = {r["name"]: r for r in cantilever.record(U)["rows"]}
    assert rows["tilt:a-b:down"]["tet_value"] == pytest.approx(phi, rel=1e-9)


def test_the_voxel_values_of_the_run_give_the_difference_and_the_ratio(tmp_path):
    c = Cantilever(tmp_path)
    voxel = [{"name": "displacement:tipz:down", "kind": "displacement", "unit": "mm", "limit": 0.01, "value_start": 1.0,
              "value_end": 0.004, "g_end": -0.6, "met": True}]
    rec = c.record(metrics={"rows": voxel})
    by = {r["name"]: r for r in rec["rows"]}
    row = by["displacement:tipz:down"]
    assert row["voxel_value"] == 0.004 and row["difference"] == pytest.approx(row["tet_value"] - 0.004, rel=1e-14)
    assert row["tet_over_voxel"] == pytest.approx(row["tet_value"] / 0.004, rel=1e-14)
    assert "met" not in row and "met" not in rec                  # the limit is a fact; no verdict
    other = by["displacement:tipn:down"]
    assert other["voxel_value"] is None and other["difference"] is None and other["tet_over_voxel"] is None
    assert any("no voxel value" in f for f in rec["facts"]) and any("optimizer again" in o for o in rec["options"])
    assert "gray" in rec["voxel_value_source"] and "not the value of the printed STL" in rec["voxel_value_source"]


def test_a_region_without_a_node_makes_a_skipped_record_with_the_fact(tmp_path):
    limits = {"displacement": [{"name": "far", "region": "far", "limit_mm": 0.01}]}
    c = Cantilever(tmp_path, limits, extra_boxes={"far": ((50, 0, 0), (54, 10, 10))})
    rec = c.record()
    assert rec["status"] == "skipped" and rec["reason"] == "rows_cannot_work" and rec["rows"] == []
    assert "measure region 'far'" in rec["facts"][0] and "Verify goes on without the rows" in rec["options"][0]


def test_a_rigid_fit_of_rank_below_6_makes_a_skipped_record_with_the_rank_message(tmp_path):
    limits = {"alignment": [{"region_a": "dot", "region_b": "b", "axis": [1, 0, 0], "shift_limit_mm": 0.002}]}
    c = Cantilever(tmp_path, limits, extra_boxes={"dot": ((9.9, 4.9, 4.9), (10.1, 5.1, 5.1))})     # holds the one node (10, 5, 5)
    rec = c.record()
    assert rec["status"] == "skipped" and rec["reason"] == "rows_cannot_work"
    assert "rank 3 of 6" in rec["facts"][0] and "one node" in rec["facts"][0]
    assert rec["regions"]["dot"]["nodes"] == 1


def test_no_rows_in_the_yaml_gives_no_record(tmp_path):
    from topopt.config import load_config
    from topopt.verify import motion
    p = make_case(tmp_path, h=2.0, iterations=2)
    cfg = load_config(p)
    assert not motion.wanted(cfg)
    assert motion.motion_record(cfg, SimpleNamespace(run_dir=tmp_path), None, None, {}, None, BAND, None) is None


def test_the_voxel_rows_are_read_from_metrics_json(tmp_path):
    from topopt.verify.motion import voxel_rows
    info = SimpleNamespace(run_dir=tmp_path)
    assert voxel_rows(info) == {}
    (tmp_path / "result").mkdir()
    (tmp_path / "result" / "metrics.json").write_text(json.dumps({"rows": [{"name": "a", "value_end": 1.5}, "junk", {"nokey": 1}]}))
    assert voxel_rows(info) == {"a": {"name": "a", "value_end": 1.5}}
    (tmp_path / "result" / "metrics.json").write_text("{")
    assert voxel_rows(info) == {}


def test_bind_nodes_builds_the_mean_operator_from_a_dof_map_and_a_lookup():
    """The operator of a displacement row is the mean over the listed nodes: a tiny dof map and a hand-made lookup."""
    import scipy.sparse as sp
    from topopt import rows_motion as rm
    from topopt.config import DisplacementLimit
    cfg = SimpleNamespace(load_cases=[SimpleNamespace(name="c")],
                          limits=SimpleNamespace(displacement=(DisplacementLimit("d", "r", 0.1, (0.0, 0.0, 1.0), "abs", ("c",), None),),
                                                 alignment=()))
    pos = np.full(12, rm.FIXED_DOF)
    pos[3:] = np.arange(9)
    ms = rm.MotionSet(cfg)
    facts = ms.bind_nodes(pos, 9, lambda name: rm.RegionNodes(name, np.array([1, 2, 3]), np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]), 3, 0))
    assert facts == [] and len(ms.rows) == 1 and ms.rows[0].name == "displacement:d:c"
    G = ms.rows[0].G
    assert isinstance(G, sp.csr_matrix) and G.shape == (1, 9)
    np.testing.assert_allclose(G.toarray(), [[0, 0, 1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3]], atol=1e-15)


def test_bind_and_bind_nodes_give_the_same_operators_for_the_same_regions(tmp_path):
    """`bind` (the voxel grid) is `bind_nodes` with the voxel dof map and `region_nodes` as the lookup: the optimizer path."""
    from topopt import rows_motion as rm
    from topopt.config import load_config
    from topopt.optimize import Model
    from topopt.problem import build_problem
    cfg = load_config(write_case(tmp_path))
    problem = build_problem(cfg)
    model = Model(problem)
    voxel = rm.MotionSet(cfg)
    facts_voxel = voxel.bind(model.problem, model.free_dofs)
    pos = rm.dof_map(model.problem.shape, model.free_dofs, model.problem.fixed_dofs)
    nodes = rm.MotionSet(cfg)
    facts_nodes = nodes.bind_nodes(pos, len(model.free_dofs), lambda name: rm.region_nodes(cfg, model.problem, pos, name))
    assert facts_voxel == facts_nodes
    assert [r.name for r in voxel.rows] == [r.name for r in nodes.rows] and len(voxel.rows) == 7           # 4 displacement rows + 2 cases, shift, tilt
    for a, b in zip(voxel.rows, nodes.rows):
        assert a.G.shape == b.G.shape and a.G.shape[1] == len(model.free_dofs) and a.G.nnz > 0
        assert abs(a.G - b.G).max() == 0.0                                  # the same operator, entry by entry
        assert (a.case, a.limit, a.sense, a.eps) == (b.case, b.limit, b.sense, b.eps)


def test_the_error_record_of_the_rows_has_the_fact_and_the_options():
    from topopt.verify.motion import error_record
    from topopt.verify.report import motion_lines
    rec = error_record("ValueError: bad measure STL")
    assert rec["status"] == "error" and rec["reason"] == "exception" and rec["rows"] == [] and rec["message"] == "ValueError: bad measure STL"
    assert rec["facts"] == ["The rows stopped: ValueError: bad measure STL"] and "Verify goes on without the rows" in rec["options"][0]
    assert "met" not in rec
    assert motion_lines(rec)[0] == "motion rows: error (exception)" and len(motion_lines(rec)) == 3


def test_a_row_on_a_fixed_region_says_the_value_is_zero_on_the_tets_not_that_it_steers_the_design(tmp_path):
    limits = {"displacement": [{"name": "wall", "region": "wall", "limit_mm": 0.01}]}
    c = Cantilever(tmp_path, limits, extra_boxes={"wall": ((-2, 0, 0), (-0.5, 10, 10))})          # the nodes at x = 0 are in the clamped plate
    rec = c.record()
    assert rec["status"] == "done" and all(r["tet_value"] == 0.0 for r in rec["rows"])
    text = " ".join(rec["facts"])
    assert "every node of 'wall' is fixed" in text and "the value is always 0 on the tet model" in text
    assert "steer the design" not in text and "move the region off the fixed region, or drop the row" in text

