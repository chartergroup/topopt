"""Tests for the displacement and alignment limits (A5, CONTRACTS.md section 31).

Tolerances (each one is stated at the test that uses it; the values are measured):
- FD gradients, direct solver (and AMG with rtol 1e-13): best of two central steps, relative to the
  largest gradient of the row (the A1 error scale), bar FD_DIRECT.
- AMG at the run tolerance (rtol 1e-5): the analytic gradient against the analytic gradient of a direct
  solver at the same x (all voxels, A1 error scale), bar AMG_GRAD_RUN. A central difference cannot see the
  error of the adjoint solve: its own error at step 1e-2 is about 1e-4.
- Rigid-motion identities: RIGID_TOL (round-off of a few thousand sums).
- Cantilever known answer: CANTILEVER_TOL against Timoshenko theory at 8 elements across the depth.
- A row that is active ends with g at or below FEAS_TOL (the 0.5 % of the run log).
"""

import json

import numpy as np
import pytest
import trimesh
import yaml

from test_optimize import box, make_case
from topopt import rows_motion as rm
from topopt.cli import check_data, main
from topopt.config import UserError, load_config
from topopt.formulation import NOT_MET, Formulation, weighted_compliance
from topopt.mma import MmaState
from topopt.optimize import Model, formulation_from_config, minimize, run
from topopt.problem import build_problem
from topopt.snapshot import RunWriter

FD_DIRECT = 1e-6        # measured worst 7e-7 (shift row, step 1e-3), 4e-7 at step 1e-4
AMG_GRAD_RUN = 5e-4     # measured worst 3.4e-4 (shift row, case down): the error of the adjoint solve (rtol 1e-5)
RIGID_TOL = 1e-12
CANTILEVER_TOL = 0.05   # measured 2.1 % at 8 elements across the depth
FEAS_TOL = NOT_MET


# --- cases ---------------------------------------------------------------


def limits_case(tmp_path, limits, h=2.0, iterations=5, solver=None, measure=True):
    """The cantilever of test_optimize with probe regions mid, a and b, and the given `limits`."""
    p = make_case(tmp_path, h=h, iterations=iterations)
    box(tmp_path / "mid.stl", (8, 0, 0), (12, 10, 10))
    box(tmp_path / "a.stl", (6, 0, 0), (8, 10, 10))
    box(tmp_path / "b.stl", (14, 0, 0), (16, 10, 10))
    d = yaml.safe_load(p.read_text())
    if measure:
        d["regions"]["measure"] = {"mid": "mid.stl", "a": "a.stl", "b": "b.stl"}
    if limits is not None:
        d["limits"] = limits
    if solver:
        d["settings"]["solver"] = solver
    p.write_text(yaml.safe_dump(d))
    return p


ALL_ROWS = {
    "displacement": [
        {"region": "mid", "limit_mm": 0.02, "direction": [0, 0, -1]},                       # |y|, two cases
        {"region": "tip", "limit_mm": 0.03},                                                # norm of the mean vector
        {"name": "up", "region": "mid", "limit_mm": 0.02, "direction": [0, 0, -1], "sense": "max", "cases": ["down"]},
        {"name": "dn", "region": "mid", "limit_mm": 0.02, "direction": [0, 0, -1], "sense": "min", "cases": ["down"]},
    ],
    "alignment": [
        {"region_a": "a", "region_b": "b", "axis": [1, 0, 0], "shift_limit_mm": 0.002, "tilt_limit_rad": 5e-4},
    ],
}


def sub(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir()
    return folder


def make_model(tmp_path, limits=None, rtol=None, **kw):
    m = Model(build_problem(load_config(limits_case(tmp_path, limits, **kw))))
    if rtol is not None and m.solver_name == "amg":
        m.solver.rtol = rtol
    return m


def evaluate(m, form, x, beta=2.0):
    m.solver._x_prev = None             # AMG: cold start (KB pitfall amg-warm-start-stale-solution)
    rho_tilde, rho = m.density(x, beta)
    point = m.design_point(x, beta, rho_tilde, rho, m.solve(m.stiffness(rho)))
    state = MmaState(m.n_design)
    state.f0_ref = 1.0
    return form.evaluate(point, state)


def fd_error(m, form, steps, samples=8, seed=3):
    """Worst relative error of the rows against central differences, the best of `steps`.

    Error scale of A1: |dg| of the voxel, but at least 1e-3 times the largest |dg| of the row.
    Returns {row name: error}.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 0.7, m.n_design)
    ev = evaluate(m, form, x)
    names = [c.name for c in form.constraints]
    voxels = rng.choice(m.n_design, size=samples, replace=False)
    best = {n: np.inf for n in names}
    for eps in steps:
        worst = {n: 0.0 for n in names}
        for e in voxels:
            xp, xm = x.copy(), x.copy()
            xp[e] += eps
            xm[e] -= eps
            fd = (evaluate(m, form, xp).g - evaluate(m, form, xm).g) / (2 * eps)
            for r, n in enumerate(names):
                scale = max(abs(ev.dg[r, e]), 1e-3 * np.abs(ev.dg[r]).max())
                worst[n] = max(worst[n], abs(fd[r] - ev.dg[r, e]) / scale)
        for n in names:
            best[n] = min(best[n], worst[n])
    return best


# --- the YAML keys ----------------------------------------------------------


def test_limits_load_with_defaults(tmp_path):
    cfg = load_config(limits_case(tmp_path, ALL_ROWS))
    assert sorted(m.name for m in cfg.regions.measure) == ["a", "b", "mid"]
    d0, d1, d2, d3 = cfg.limits.displacement
    assert d0.name == "mid" and d0.sense == "abs" and d0.cases == ("down", "side")   # all cases by default
    assert d0.direction == (0.0, 0.0, -1.0) and d1.direction is None and d1.price is None
    assert (d2.sense, d2.cases, d3.sense) == ("max", ("down",), "min")
    al, = cfg.limits.alignment
    assert al.name == "a-b" and al.axis == (1.0, 0.0, 0.0) and al.cases == ("down", "side")
    assert (al.shift_limit_mm, al.tilt_limit_rad) == (0.002, 5e-4)


def test_no_limits_is_the_old_config(tmp_path):
    cfg = load_config(make_case(tmp_path))
    assert cfg.limits.displacement == () and cfg.limits.alignment == () and cfg.regions.measure == ()
    assert formulation_from_config(cfg, 10).constraints[0].name == "volume"
    assert len(formulation_from_config(cfg, 10).constraints) == 1


def test_direction_is_normalised_and_text_numbers_are_read(tmp_path):
    cfg = load_config(limits_case(tmp_path, {
        "displacement": [{"region": "mid", "limit_mm": "5e-3", "direction": [0, 3, 4]}],
        "alignment": [{"region_a": "a", "region_b": "b", "axis": [2, 0, 0], "tilt_limit_rad": "1e-3"}]}))
    assert cfg.limits.displacement[0].limit_mm == 5e-3
    assert cfg.limits.displacement[0].direction == pytest.approx((0, 0.6, 0.8))
    assert cfg.limits.alignment[0].axis == (1.0, 0.0, 0.0) and cfg.limits.alignment[0].shift_limit_mm is None


@pytest.mark.parametrize("limits, text", [
    ({"displacement": [{"region": "nope", "limit_mm": 1}]}, "'limits.displacement[0].region' is 'nope', not a measure or a load region"),
    ({"displacement": [{"region": "mid"}]}, "missing key 'limit_mm' in 'limits.displacement[0]'"),
    ({"displacement": [{"region": "mid", "limit_mm": 0}]}, "'limits.displacement[0].limit_mm' must be > 0"),
    ({"displacement": [{"region": "mid", "limit_mm": 1, "sense": "max"}]}, "'limits.displacement[0].sense' needs a 'direction'"),
    ({"displacement": [{"region": "mid", "limit_mm": 1, "direction": [0, 0, 1], "sense": "up"}]}, "must be abs, max or min, got 'up'"),
    ({"displacement": [{"region": "mid", "limit_mm": 1, "direction": [0, 0, 0]}]}, "must not be the zero vector"),
    ({"displacement": [{"region": "mid", "limit_mm": 1, "direction": [0, 1]}]}, "must be a list of 3 numbers"),
    ({"displacement": [{"region": "mid", "limit_mm": 1, "cases": ["x"]}]}, "'limits.displacement[0].cases[0]' is 'x', not a load case (known: down, side)"),
    ({"displacement": [{"region": "mid", "limit_mm": 1, "cases": ["down", "down"]}]}, "names a load case twice"),
    ({"displacement": [{"region": "mid", "limit_mm": 1, "price": -1}]}, "'limits.displacement[0].price' must be > 0"),
    ({"displacement": [{"region": "mid", "limit_mm": 1}, {"region": "mid", "limit_mm": 2}]}, "give each row its own 'name'"),
    ({"displacement": [{"region": "mid", "limit_mm": 1, "bogus": 1}]}, "unknown key 'bogus' in 'limits.displacement[0]'"),
    ({"alignment": [{"region_a": "a", "region_b": "a", "axis": [1, 0, 0], "tilt_limit_rad": 1}]}, "compares region 'a' with itself"),
    ({"alignment": [{"region_a": "a", "region_b": "b", "axis": [1, 0, 0]}]}, "needs 'shift_limit_mm' or 'tilt_limit_rad'"),
    ({"alignment": [{"region_a": "a", "region_b": "b", "tilt_limit_rad": 1}]}, "missing key 'axis' in 'limits.alignment[0]'"),
    ({"alignment": [{"region_a": "a", "region_b": "b", "axis": [1, 0, 0], "tilt_limit_rad": -1}]}, "'limits.alignment[0].tilt_limit_rad' must be > 0"),
    ({"stress": []}, "unknown key 'stress' in 'limits'"),
    ({"displacement": {"region": "mid"}}, "'limits.displacement' must be a list of rows"),
])
def test_limit_errors_are_one_line_with_the_key(tmp_path, limits, text):
    p = limits_case(tmp_path, limits)
    with pytest.raises(UserError) as e:
        load_config(p)
    msg = str(e.value)
    assert text in msg and "\n" not in msg and f"(file: {p}" in msg


def test_a_max_and_a_min_row_may_share_a_region_without_a_name(tmp_path):
    """The rule is on the pair (sense, name): the full row names differ (displacement_max / displacement_min)."""
    row = lambda sense: {"region": "mid", "limit_mm": 0.02, "direction": [0, 0, -1], "sense": sense, "cases": ["down"]}
    cfg = load_config(limits_case(tmp_path, {"displacement": [row("max"), row("min")]}))
    assert [d.name for d in cfg.limits.displacement] == ["mid", "mid"]
    assert [r.name for r in rm.row_specs(cfg)] == ["displacement_max:mid:down", "displacement_min:mid:down"]
    # the same sense and name twice is still an error, and the text names the sense
    with pytest.raises(UserError) as e:
        load_config(limits_case(sub(tmp_path, "twice"), {"displacement": [row("max"), row("max")]}))
    assert "'limits.displacement[1]' has the name 'mid' and the sense 'max', as an earlier row" in str(e.value)
    # an abs row and a max row on one region, with the default name, are two kinds
    abs_row = {"region": "mid", "limit_mm": 0.02, "direction": [0, 0, -1]}
    cfg = load_config(limits_case(sub(tmp_path, "mixed"), {"displacement": [abs_row, row("max")]}))
    assert [r.name for r in rm.row_specs(cfg)][:3] == ["displacement:mid:down", "displacement:mid:side",
                                                       "displacement_max:mid:down"]


def test_measure_region_errors(tmp_path):
    p = limits_case(tmp_path, None)
    d = yaml.safe_load(p.read_text())
    d["regions"]["measure"] = {"tip": "mid.stl"}   # the name of a load region
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="'regions.measure.tip' has the name of a load region"):
        load_config(p)
    d["regions"]["measure"] = {"mid": "missing.stl"}
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="STL file not found"):
        load_config(p)
    d["regions"]["measure"] = ["mid.stl"]
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="'regions.measure' must be a mapping of name to STL file"):
        load_config(p)


# --- node sets and the facts about them --------------------------------------


def test_node_sets_use_the_8_corner_rule(tmp_path):
    from topopt.loads import node_ids
    cfg = load_config(limits_case(tmp_path, ALL_ROWS))
    prob = build_problem(cfg)
    m = Model(prob)
    pos = rm.dof_map(prob.shape, m.free_dofs, prob.fixed_dofs)
    tip = rm.region_nodes(cfg, prob, pos, "tip")
    np.testing.assert_array_equal(tip.nodes, node_ids(prob.load_region_ids == 0))
    mid = rm.region_nodes(cfg, prob, pos, "mid")
    # mid box x 8..12, h = 2: voxel centres at 9 and 11 -> 2 voxels in x, all 5 x 5 in (y, z) -> 3 x 6 x 6 nodes
    assert mid.nodes.size == 3 * 6 * 6 and mid.n_void == 0 and mid.n_fixed == 0
    # positions in mm: the nodes of the two voxel columns at x = 9 and 11 lie at x = 8, 10 and 12
    assert mid.coords[:, 0].min() == pytest.approx(8.0)
    assert set(np.unique(mid.coords[:, 0])) == {8.0, 10.0, 12.0}
    m.solver.free()


def test_a_region_on_the_fixed_plate_is_a_fact_and_all_fixed_gives_value_zero(tmp_path):
    """The plate (x -2..0) is fixed. A probe inside it: every node fixed, value 0, gradient 0, no crash."""
    p = limits_case(tmp_path, None)
    box(tmp_path / "inplate.stl", (-2, 2, 2), (0, 8, 8))
    d = yaml.safe_load(p.read_text())
    d["regions"]["measure"] = {"inplate": "inplate.stl"}
    d["limits"] = {"displacement": [{"region": "inplate", "limit_mm": 0.5, "direction": [0, 0, 1]}]}
    p.write_text(yaml.safe_dump(d))
    m = Model(build_problem(load_config(p)))
    lines = []
    form = formulation_from_config(m.problem.config, m.n_design, m, lines.append)
    assert any("every node of 'inplate' is fixed" in t and "Options" in t for t in lines)
    assert any("nodes are fixed" in t for t in lines)
    solves = []
    real = m.solver.solve
    m.solver.solve = lambda B, warm=True: (solves.append(B.ndim), real(B, warm=warm))[1]
    ev = evaluate(m, form, np.full(m.n_design, 0.5))
    row = ev.constraints[1]
    assert row["value"] == pytest.approx(0.01 * 0.5) and row["g"] == pytest.approx(0.01 - 1)   # only the guard
    assert np.all(ev.dg[1] == 0)
    assert solves == [2]   # the forward solve only (2 columns): the row did no adjoint solve
    m.solver.free()


def keepout_case(tmp_path, keep_out, probe, rows=None):
    """Cantilever 0..10 (h = 2) with a keep-out box outside the design box, and a probe region `probe`.

    The keep-out box extends the grid and carries no element. A probe that overlaps it has nodes without a
    dof. The second region `other` is a slab of the design box. `rows` default: one alignment row.
    """
    box(tmp_path / "design.stl", (0, 0, 0), (10, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", (8, 4, 0), (10, 6, 2))
    box(tmp_path / "ko.stl", *keep_out)
    box(tmp_path / "probe.stl", *probe)
    box(tmp_path / "other.stl", (2, 0, 0), (4, 10, 10))
    d = {"name": "keepout",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"], "keep_out": ["ko.stl"],
                     "load": {"tip": "tip.stl"}, "measure": {"probe": "probe.stl", "other": "other.stl"}},
         "load_cases": [{"name": "p", "forces": [{"region": "tip", "force": [0, 0, -1.0]}]}],
         "limits": rows or {"alignment": [{"region_a": "probe", "region_b": "other", "axis": [1, 0, 0],
                                           "tilt_limit_rad": 1e-3}]},
         "settings": {"voxel_size": 2.0, "volume_fraction": 0.4, "iterations": 3}}
    p = tmp_path / "keepout.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_nodes_without_a_dof_are_left_out_with_a_fact(tmp_path):
    """The probe spans the design box and a keep-out column: the nodes that touch no element have no displacement."""
    p = keepout_case(tmp_path, ((10, 10, 0), (12, 12, 10)), ((8, 8, 4), (12, 12, 6)),
                     {"displacement": [{"region": "probe", "limit_mm": 0.5}]})
    m = Model(build_problem(load_config(p)))
    lines = []
    form = formulation_from_config(m.problem.config, m.n_design, m, lines.append)
    assert lines == ["fact: region 'probe': 10 of 18 nodes touch no solid or design voxel and have no "
                     "displacement; they are left out. Options: to measure them, put the region on solid or on a "
                     "load part; otherwise no change"]
    # 8 nodes remain: the mean is over these
    assert form.constraints[1].name == "displacement:probe:p"
    m.solver.free()


@pytest.mark.parametrize("probe, rank, nodes", [
    (((10, 10, 10), (12, 12, 12)), 3, "one node"),                    # touches the design box at a corner
    (((10, 10, 4), (12, 12, 6)), 5, "two nodes"),                     # along an edge: 2 nodes
    (((10, 10, 2), (12, 12, 8)), 5, "4 nodes on one line"),           # along an edge: 4 nodes
])
def test_a_fit_of_rank_below_6_stops_before_the_solve_with_the_fact_and_the_options(tmp_path, capsys, probe, rank, nodes):
    """A probe whose nodes with a dof are on one line (or one node) cannot give a rotation: UserError at
    `check` and at `run`, before any solve, with the rank, the node count and the options."""
    p = keepout_case(tmp_path, probe, probe)
    want = f"alignment row 'probe-other' cannot work: the rigid fit of region 'probe' has rank {rank} of 6 ({nodes})"
    with pytest.raises(UserError) as e:
        check_data(p)
    assert want in str(e.value) and "Options: a larger region, a region that is not on one line, or a displacement row in its place" in str(e.value)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 1
    err = capsys.readouterr().err
    assert want in err and f"(file: {p}" in err and len(err.strip().splitlines()) == 1
    assert json.loads((out / "status.json").read_text())["state"] == "error"
    assert not list((out / "snapshots").glob("*.bin"))       # no iteration ran


def test_the_same_probe_works_for_a_displacement_row(tmp_path):
    """A mean needs no rotation: a probe of one node is fine for a displacement row."""
    p = keepout_case(tmp_path, ((10, 10, 10), (12, 12, 12)), ((10, 10, 10), (12, 12, 12)),
                     {"displacement": [{"region": "probe", "limit_mm": 0.5}]})
    assert any("7 of 8 nodes touch no solid or design voxel" in w for w in check_data(p)["warnings"])   # a fact
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0


@pytest.mark.parametrize("row", [
    {"displacement": [{"region": "probe", "limit_mm": 0.5}]},
    {"alignment": [{"region_a": "probe", "region_b": "other", "axis": [1, 0, 0], "tilt_limit_rad": 1e-3}]},
])
def test_a_probe_with_no_node_in_the_model_is_a_user_error(tmp_path, row):
    p = keepout_case(tmp_path, ((14, 0, 0), (16, 2, 2)), ((14, 0, 0), (16, 2, 2)), row)
    with pytest.raises(UserError, match=r"cannot work: region 'probe' has no node that carries a dof \(\d+ nodes, all of "
                                        r"them outside the model\). Options: move the region onto the part"):
        check_data(p)


# --- rigid fit and rows: no finite elements -------------------------------------------


def rigid_field(coords, translation, rotation, origin):
    """u = T + Theta x (x - o) at the points."""
    return np.asarray(translation) + np.cross(rotation, coords - np.asarray(origin))


@pytest.mark.equivalence
def test_rigid_fit_equals_the_pinv_of_the_pipeline_rows():
    """The sparse fit of this module equals the dense `pinv([I, -skew(r)])` of the pipeline (`rigid_motion_rows`)."""
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 10, (40, 3))
    nodes = np.arange(40)
    reg = rm.RegionNodes("r", nodes, coords, 40, 0)
    pos = np.arange(3 * 40, dtype=np.int64)
    fit = rm.rigid_fit(reg, pos, 3 * 40)
    c = coords.mean(axis=0)
    A = np.vstack([np.hstack([np.eye(3), -rm.skew(r)]) for r in coords - c])
    P = np.linalg.pinv(A)
    np.testing.assert_allclose(fit.translation.toarray(), P[:3], atol=1e-14)
    np.testing.assert_allclose(fit.rotation.toarray(), P[3:], atol=1e-14)
    assert fit.rank == np.linalg.matrix_rank(A) == 6


@pytest.mark.parametrize("coords, rank", [
    ([[1, 2, 3]], 3),                                                    # one node
    ([[0, 0, 0], [1, 2, 3]], 5),                                         # two nodes
    ([[0, 0, 0], [1, 1, 1], [2, 2, 2]], 5),                              # three on a line
    ([[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]], 5),                   # four on a line
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 6),                              # three, not on a line
    ([[x, y, 0] for x in range(4) for y in range(4)], 6),                # a plane
    ([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], 6),    # the 8 nodes of one voxel
])
def test_fit_rank_is_the_rank_of_the_fit_matrix(coords, rank):
    coords = np.array(coords, dtype=float)
    centre, M, got = rm.fit_matrix(coords)
    A = np.vstack([np.hstack([np.eye(3), -rm.skew(r)]) for r in coords - coords.mean(axis=0)])
    assert got == rank == np.linalg.matrix_rank(A)


def two_region_setup(tmp_path):
    p = limits_case(tmp_path, {"alignment": [{"region_a": "a", "region_b": "b", "axis": [1, 0.2, 0.1],
                                              "shift_limit_mm": 1.0, "tilt_limit_rad": 1e-2,
                                              "cases": ["down"]}]})
    cfg = load_config(p)
    prob = build_problem(cfg)
    m = Model(prob)
    ms = rm.MotionSet(cfg)
    ms.bind_model(m)
    nodes = {n: rm.region_nodes(cfg, prob, ms_pos(m), n) for n in ("a", "b")}
    return m, ms, nodes


def ms_pos(m):
    return rm.dof_map(m.problem.shape, m.free_dofs, m.problem.fixed_dofs)


def coords_of_all_nodes(m):
    nx, ny, nz = m.problem.shape
    ijk = np.stack(np.unravel_index(np.arange((nx + 1) * (ny + 1) * (nz + 1)), (nx + 1, ny + 1, nz + 1)), axis=1)
    return np.asarray(m.problem.origin) + m.problem.h * ijk


def test_whole_body_rigid_motion_gives_zero_shift_and_tilt(tmp_path):
    m, ms, _ = two_region_setup(tmp_path)
    X = coords_of_all_nodes(m)
    rng = np.random.default_rng(4)
    for _ in range(3):
        u_full = rigid_field(X, rng.normal(size=3), 0.01 * rng.normal(size=3), rng.uniform(0, 20, 3)).ravel()
        u = u_full[m.free_dofs]
        for row in ms.rows:
            assert np.abs(row.y(u)).max() < RIGID_TOL * max(1.0, np.abs(u).max()), row.name
    m.solver.free()


def test_prescribed_rigid_motion_of_one_region_is_recovered_exactly(tmp_path):
    """Region a moves as a rigid body, region b stays. The rows give what the geometry says.

    Expected values come from the definition, not from the formula of the module:
    tilt = the part of (theta_a - theta_b) across the axis;
    shift = the part across the axis of u_a(p) - u_b(p) at the midpoint p of the two node means.
    """
    m, ms, regs = two_region_setup(tmp_path)
    X = coords_of_all_nodes(m)
    A, B = regs["a"], regs["b"]
    ca, cb = A.coords.mean(axis=0), B.coords.mean(axis=0)
    rng = np.random.default_rng(7)
    Ta, Tb = rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.1
    Oa, Ob = rng.uniform(0, 20, 3), rng.uniform(0, 20, 3)
    Wa, Wb = 0.003 * rng.normal(size=3), 0.002 * rng.normal(size=3)
    u_full = np.zeros((X.shape[0], 3))
    u_full[A.nodes] = rigid_field(A.coords, Ta, Wa, Oa)
    u_full[B.nodes] = rigid_field(B.coords, Tb, Wb, Ob)
    u = u_full.ravel()[m.free_dofs]
    e1, e2 = rm.transverse_axes([1, 0.2, 0.1])
    mid = 0.5 * (ca + cb)
    rel = rigid_field(mid[None], Ta, Wa, Oa)[0] - rigid_field(mid[None], Tb, Wb, Ob)[0]
    shift = np.array([e1 @ rel, e2 @ rel])
    tilt = np.array([e1 @ (Wa - Wb), e2 @ (Wa - Wb)])
    got = {r.name.split(":")[0]: r.y(u) for r in ms.rows}
    np.testing.assert_allclose(got["shift"], shift, rtol=0, atol=RIGID_TOL)
    np.testing.assert_allclose(got["tilt"], tilt, rtol=0, atol=RIGID_TOL)
    # a common rigid motion on top changes nothing
    common = rigid_field(X, rng.normal(size=3), 0.01 * rng.normal(size=3), rng.uniform(0, 20, 3)).ravel()
    for r in ms.rows:
        np.testing.assert_allclose(r.y(u + common[m.free_dofs]), r.y(u), rtol=0, atol=1e-11)
    m.solver.free()


def test_swapping_the_two_regions_changes_only_the_sign(tmp_path):
    m, ms, regs = two_region_setup(tmp_path)
    cfg = m.problem.config
    swapped = yaml.safe_load(cfg.yaml_path.read_text())
    swapped["limits"]["alignment"][0].update(region_a="b", region_b="a")
    cfg.yaml_path.write_text(yaml.safe_dump(swapped))
    ms2 = rm.MotionSet(load_config(cfg.yaml_path))
    ms2.bind_model(m)
    u = np.random.default_rng(1).normal(size=m.n_free)
    for r1, r2 in zip(ms.rows, ms2.rows):
        np.testing.assert_allclose(r2.y(u), -r1.y(u), rtol=1e-12, atol=1e-15)
    m.solver.free()


# --- gradients (FD) ----------------------------------------------------------------


def all_rows_formulation(m):
    form = formulation_from_config(m.problem.config, m.n_design, m)
    return Formulation(weighted_compliance(), form.constraints)


def test_fd_all_rows_direct(tmp_path):
    """Every row kind: adjoint gradient against central differences. A direct solver, or AMG at rtol 1e-13."""
    m = make_model(tmp_path, ALL_ROWS, rtol=1e-13)
    form = all_rows_formulation(m)
    err = fd_error(m, form, steps=(1e-3, 1e-4))
    worst = {n: e for n, e in err.items() if n != "volume"}
    assert len(worst) == 10      # 2 + 2 displacement, 1 + 1 signed, 2 shift, 2 tilt
    assert max(worst.values()) < FD_DIRECT, worst
    m.solver.free()


def test_amg_gradient_at_the_run_tolerance_matches_the_direct_gradient(tmp_path):
    """AMG with the run tolerance (rtol 1e-5, cold start) against a direct solver, at the same x.

    The analytic gradient of every row, all voxels, on the A1 error scale (|dg| of the voxel, at least 1e-3
    times the largest |dg| of the row). The error is that of the adjoint solve at rtol 1e-5: with 1e-8 (12 CG
    iterations instead of 8) the worst is 5.2e-5 (plan section 8, follow-up). Measured worst 3.4e-4 (shift row,
    case down), 1.2e-4 for the displacement rows.
    """
    md = make_model(sub(tmp_path, "direct"), ALL_ROWS, rtol=1e-13)   # no direct solver here: AMG at 1e-13 is the reference
    ma = make_model(sub(tmp_path, "amg"), ALL_ROWS, solver="amg")
    assert ma.solver_name == "amg" and ma.solver.rtol == pytest.approx(1e-5)
    x = np.random.default_rng(3).uniform(0.3, 0.7, md.n_design)
    ref = evaluate(md, all_rows_formulation(md), x)
    got = evaluate(ma, all_rows_formulation(ma), x)
    names = [c["name"] for c in ref.constraints]
    assert len(names) == 11 and [c["name"] for c in got.constraints] == names
    worst = {}
    for r, n in enumerate(names):
        if n == "volume":
            continue
        scale = np.maximum(np.abs(ref.dg[r]), 1e-3 * np.abs(ref.dg[r]).max())
        worst[n] = float((np.abs(got.dg[r] - ref.dg[r]) / scale).max())
    assert len(worst) == 10 and max(worst.values()) < AMG_GRAD_RUN, worst
    md.solver.free()
    ma.solver.free()


def test_fd_where_the_guard_matters(tmp_path):
    """A limit so large that eps (1 % of it) is about the size of y: the smooth norm is not the norm here."""
    row = lambda lim: {"displacement": [{"region": "mid", "limit_mm": lim, "direction": [0, 0, -1],
                                         "cases": ["down"]}]}
    m0 = make_model(sub(tmp_path, "a"), row(1.0), rtol=1e-13)
    x = np.random.default_rng(3).uniform(0.3, 0.7, m0.n_design)
    ev = evaluate(m0, formulation_from_config(m0.problem.config, m0.n_design, m0), x)
    y = rm.physical_value("displacement:mid:down", ev.constraints[1]["value"], 1.0)
    m0.solver.free()
    m = make_model(sub(tmp_path, "b"), row(100 * y), rtol=1e-13)    # eps = y
    err = fd_error(m, formulation_from_config(m.problem.config, m.n_design, m), steps=(1e-3, 1e-4))
    assert err["displacement:mid:down"] < FD_DIRECT
    m.solver.free()


def test_the_guard_changes_the_value_by_less_than_eps_squared_over_2y():
    limit = 2.0
    eps = rm.EPS_FRACTION * limit
    for y in (0.0, 0.1 * limit, 0.5 * limit, limit):
        row = rm.MotionRow("displacement:x:c", 0, None, limit, "abs", eps)
        v, w = row.value_and_slope(np.array([y]))
        assert abs(v - abs(y)) <= (eps if y == 0 else eps ** 2 / (2 * abs(y))) * (1 + 1e-12)
        assert w[0] == pytest.approx(y / v)
    assert rm.physical_value("displacement:x:c", row.value_and_slope(np.array([0.3]))[0], limit) == pytest.approx(0.3)
    assert rm.physical_value("displacement_max:x:c", -0.4, limit) == -0.4


# --- known answers -------------------------------------------------------------------


def cantilever(tmp_path, n, rows, iterations=1):
    """Uniform square beam L = 40, b = 4, E = 3500, nu = 0.35, n elements across the depth.

    A one-voxel layer at the tip carries the load (1 N down), a one-voxel plate is fixed.
    """
    h = 4.0 / n
    box(tmp_path / "design.stl", (0, 0, 0), (40, 4, 4))
    box(tmp_path / "plate.stl", (-h, 0, 0), (0, 4, 4))
    box(tmp_path / "tip.stl", (40 - h, 0, 0), (40, 4, 4))
    d = {"name": "beam",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "p", "forces": [{"region": "tip", "force": [0, 0, -1.0]}]}],
         "limits": rows,
         "settings": {"voxel_size": h, "volume_fraction": 0.5, "iterations": iterations},
         "material": {"E": 3500.0, "nu": 0.35}}
    p = tmp_path / "beam.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def timoshenko_tip(L, b, E, nu, h, P=1.0):
    """Tip deflection at the load centroid a = L - h/2 (the one-voxel tip layer spreads the load over two node planes)."""
    a = L - h / 2
    I, A, G = b ** 4 / 12, b * b, E / (2 * (1 + nu))
    kappa = 10 * (1 + nu) / (12 + 11 * nu)    # Cowper
    return P * a ** 3 / (3 * E * I) + P * a / (kappa * G * A)


def test_uniform_cantilever_tip_displacement(tmp_path):
    """The row on the tip region, along the load, with uniform density 1: Timoshenko theory.

    Measured FE / theory: 0.9791 at n = 8 and 0.9876 at n = 16 (the 3D grid is stiffer than the beam: the
    converged value is about 0.991). The bar CANTILEVER_TOL is 5 %. The row also equals c / F exactly,
    because the load is spread in equal shares over the nodes of the region.
    """
    p = cantilever(tmp_path, 8, {"displacement": [{"region": "tip", "limit_mm": 1.0, "direction": [0, 0, -1]}]})
    m = Model(build_problem(load_config(p)))
    n = m.n_design
    one = np.ones(n)
    U = m.solve(m.stiffness(one))
    point = m.design_point(one, 1.0, one, one, U)
    ms = rm.MotionSet(m.problem.config)
    ms.bind_model(m)
    row = ms.rows[0]
    y = row.y(U[:, 0])[0]
    ref = timoshenko_tip(40.0, 4.0, 3500.0, 0.35, 0.5)
    assert abs(y / ref - 1) < CANTILEVER_TOL
    assert y / ref == pytest.approx(0.979, abs=2e-3)           # the measured value: a change here is a model change
    assert y == pytest.approx(point.compliance / 1.0, rel=1e-10)   # FE to FE: c = F * mean(u_z)
    assert rm.physical_value("displacement:tip:p", row.evaluate(point).value, 1.0) == pytest.approx(y, rel=1e-12)
    m.solver.free()


def test_uniform_cantilever_alignment_of_two_slabs(tmp_path):
    """Shift and tilt of two cross-sections of the uniform beam against the closed form.

    Slabs at x = 10 and x = 30 (one voxel thick). Closed form (Euler-Bernoulli bending plus the shear
    deflection with the Cowper factor), load P at the centroid a of the tip layer. Measured FE / closed
    form at n = 8: tilt 0.9905, shift 0.9822. The bars are 1 % and 2 % (the discretisation error is the
    difference to 1; the rows sit at the centres of the slabs, not at x = 10 and 30 exactly).
    """
    n, L, b, E, nu, P = 8, 40.0, 4.0, 3500.0, 0.35, 1.0
    h = b / n
    rows = {"alignment": [{"region_a": "s1", "region_b": "s2", "axis": [1, 0, 0],
                           "shift_limit_mm": 1.0, "tilt_limit_rad": 1.0}]}
    p = cantilever(tmp_path, n, rows)
    box(tmp_path / "s1.stl", (10 - h, 0, 0), (10, 4, 4))
    box(tmp_path / "s2.stl", (30 - h, 0, 0), (30, 4, 4))
    d = yaml.safe_load(p.read_text())
    d["regions"]["measure"] = {"s1": "s1.stl", "s2": "s2.stl"}
    p.write_text(yaml.safe_dump(d))
    m = Model(build_problem(load_config(p)))
    one = np.ones(m.n_design)
    U = m.solve(m.stiffness(one))
    ms = rm.MotionSet(m.problem.config)
    ms.bind_model(m)
    got = {r.name.split(":")[0]: np.linalg.norm(r.y(U[:, 0])) for r in ms.rows}
    # slab centres: voxels x in (10 - h, 10): centre 10 - h/2 (node mean), same for s2
    xa, xb, a = 10 - h / 2, 30 - h / 2, L - h / 2
    I, A, G = b ** 4 / 12, b * b, E / (2 * (1 + nu))
    kappa = 10 * (1 + nu) / (12 + 11 * nu)

    def phi(x):     # bending rotation (a load P at x = a, clamped at x = 0), positive: with the deflection
        return P * (a * x - x * x / 2) / (E * I)

    def w(x):       # deflection: bending and shear
        return P * (a * x * x / 2 - x ** 3 / 6) / (E * I) + P * x / (kappa * G * A)

    tilt_ref = phi(xb) - phi(xa)
    shift_ref = abs((w(xb) - w(xa)) - 0.5 * (phi(xa) + phi(xb)) * (xb - xa))
    assert abs(got["tilt"] / tilt_ref - 1) < 0.01
    assert abs(got["shift"] / shift_ref - 1) < 0.02
    m.solver.free()


def case_values(tmp_path, rows, iterations=40, h=1.0, **kw):
    m = make_model(tmp_path, rows, h=h, iterations=iterations, **kw)
    res = minimize(m, formulation_from_config(m.problem.config, m.n_design, m))
    m.solver.free()
    return res


def physical(res, name):
    c = next(c for c in res.constraints if c["name"] == name)
    return rm.physical_value(name, c["value"], c["limit"]), c


def test_active_displacement_limit_is_met_and_costs_compliance(tmp_path):
    """A limit 15 % below the value of the free design: met (g <= FEAS_TOL), compliance not lower.

    Measured (h = 1, 40 iterations): the free run ends at 2.33e-3 mm at the mid-span probe; the limit
    at 0.85 of it is met within 1e-3 and the weighted compliance is 1.2 times that of the free run.
    """
    name = "displacement:mid:down"
    row = lambda lim: {"displacement": [{"region": "mid", "limit_mm": lim, "direction": [0, 0, -1],
                                         "cases": ["down"]}]}
    (tmp_path / "a").mkdir()
    free = case_values(tmp_path / "a", row(1.0))
    free_value, _ = physical(free, name)
    (tmp_path / "b").mkdir()
    res = case_values(tmp_path / "b", row(0.85 * free_value))
    value, c = physical(res, name)
    assert c["limit"] == pytest.approx(0.85 * free_value)
    assert c["g"] <= FEAS_TOL and value <= c["limit"] * (1 + FEAS_TOL)
    assert next(k for k in res.constraints if k["name"] == "volume")["g"] <= FEAS_TOL
    assert res.history[-1]["compliance"] >= free.history[-1]["compliance"] * (1 - 1e-3)
    assert res.history[-1]["compliance"] > 1.05 * free.history[-1]["compliance"]
    assert not res.unmet
    assert res.history[0]["constraints"][1]["g"] > 1.0   # the uniform start is far above the limit


def test_impossible_limit_is_a_fact_and_the_price_key_keeps_the_volume(tmp_path):
    """A limit that the design cannot meet is no error. With price 1000 (the default) the row and the volume
    share the miss; with `price: 100` (as the compliance limits) the volume stays at its target and only the
    limit row is unmet, with the options in the run log. Measured: see the plan."""
    row = lambda lim, price: {"displacement": [dict({"region": "tip", "limit_mm": lim, "direction": [0, 0, -1],
                                                     "cases": ["down"]}, **({"price": price} if price else {}))]}
    (tmp_path / "a").mkdir()
    free_value, _ = physical(case_values(tmp_path / "a", row(1.0, None)), "displacement:tip:down")
    results = {}
    for price in (None, 100.0):
        sub = tmp_path / f"p{price}"
        sub.mkdir()
        p = limits_case(sub, row(0.3 * free_value, price), h=1.0, iterations=40)
        prob = build_problem(load_config(p))
        out = RunWriter(sub / "run", quiet=True)
        out.clean()
        run(prob, out)
        results[price] = (json.loads((sub / "run" / "history.jsonl").read_text().splitlines()[-1]),
                          (sub / "run" / "run.log").read_text())
    last, log = results[100.0]
    by = {c["name"]: c for c in last["constraints"]}
    assert by["volume"]["g"] <= FEAS_TOL and by["displacement:tip:down"]["g"] > FEAS_TOL
    assert "constraint not met: displacement:tip:down" in log and "constraint not met: volume" not in log
    # the options of the row (the cause sentence before them is A1 text on this base and A5 text with #22: the gate test of the patch checks it)
    assert "options: raise settings.volume_fraction, relax the limit, or move the region onto the load path" in log
    last, log = results[None]
    by = {c["name"]: c for c in last["constraints"]}
    assert by["volume"]["g"] > FEAS_TOL      # the default price: the volume gives way too
    assert "constraint not met: displacement:tip:down" in log


def test_active_limits_of_both_row_kinds_run_together(tmp_path):
    """Both kinds in one run: it ends, the rows are in the history, and the run log has no error."""
    rows = {"displacement": [{"region": "mid", "limit_mm": 0.003, "direction": [0, 0, -1], "cases": ["down"]}],
            "alignment": [{"region_a": "a", "region_b": "b", "axis": [1, 0, 0], "tilt_limit_rad": 2e-4,
                           "cases": ["down"]}]}
    res = case_values(tmp_path, rows, iterations=30)
    names = [c["name"] for c in res.constraints]
    assert names == ["volume", "displacement:mid:down", "tilt:a-b:down"]
    assert all(c["g"] <= 0.05 for c in res.constraints)


# --- the report ------------------------------------------------------------------------


def hist(values, limit=2.0, name="displacement:mid:down", g=None):
    out = []
    for i, v in enumerate(values):
        gi = v / limit - 1 if g is None else g[i]
        out.append({"iteration": i, "constraints": [
            {"name": "volume", "value": 0.3, "limit": 0.3, "g": 0.0, "violation": 0.0},
            {"name": name, "value": v, "limit": limit, "g": gi, "violation": max(gi, 0)}]})
    return out


def test_rows_block_values_limit_and_met():
    eps = rm.EPS_FRACTION * 2.0
    start, end = 3.0, 1.5
    h = hist([np.hypot(start, eps), 2.5, np.hypot(end, eps)])
    block = rm.rows_block(h)
    assert len(block) == 1
    r = block[0]
    assert (r["name"], r["kind"], r["unit"], r["limit"]) == ("displacement:mid:down", "displacement", "mm", 2.0)
    assert r["value_start"] == pytest.approx(start, rel=1e-12) and r["value_end"] == pytest.approx(end, rel=1e-12)
    assert r["met"] is True and r["g_end"] < 0
    miss = rm.rows_block(hist([3.0, 2.5, 2.2]))[0]
    assert miss["met"] is False and miss["g_end"] == pytest.approx(0.1)
    edge = rm.rows_block(hist([3.0, 2.5, 2.0 * (1 + 0.9 * NOT_MET)]))[0]
    assert edge["met"] is True
    assert rm.rows_block([]) == [] and rm.rows_block(hist([1.0])[:0]) == []


def test_rows_block_signed_rows_and_other_rows():
    h = hist([-0.4, 0.5], name="displacement_min:x:c", limit=1.0)
    h[0]["constraints"].append({"name": "compliance:c", "value": 2.0, "limit": 5.0, "g": -0.6, "violation": 0.0})
    h[1]["constraints"].append({"name": "compliance:c", "value": 2.0, "limit": 5.0, "g": -0.6, "violation": 0.0})
    block = rm.rows_block(h)
    assert [r["name"] for r in block] == ["displacement_min:x:c"]       # volume and compliance rows are not A5 rows
    assert (block[0]["value_start"], block[0]["value_end"]) == (-0.4, 0.5)   # no guard: the value as it is
    assert rm.rows_block([{"iteration": 0}]) == []                       # an old run: no constraints


def test_metrics_json_of_a_run_has_the_rows_block(tmp_path):
    p = limits_case(tmp_path, ALL_ROWS, h=2.0, iterations=4)
    out_dir = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out_dir), "--quiet"]) == 0
    metrics = json.loads((out_dir / "result" / "metrics.json").read_text())
    rows = metrics["rows"]
    assert [r["name"] for r in rows] == [
        "displacement:mid:down", "displacement:mid:side", "displacement:tip:down", "displacement:tip:side",
        "displacement_max:up:down", "displacement_min:dn:down", "shift:a-b:down", "shift:a-b:side",
        "tilt:a-b:down", "tilt:a-b:side"]
    hist0 = [json.loads(line) for line in (out_dir / "history.jsonl").read_text().splitlines()]
    for r in rows:
        first = next(c for c in hist0[0]["constraints"] if c["name"] == r["name"])
        last = next(c for c in hist0[-1]["constraints"] if c["name"] == r["name"])
        assert r["limit"] == last["limit"] and r["g_end"] == last["g"]
        assert r["met"] == (last["g"] <= NOT_MET)
        assert r["value_start"] == pytest.approx(rm.physical_value(r["name"], first["value"], r["limit"]))
    run_json = json.loads((out_dir / "run.json").read_text())
    assert [c["name"] for c in run_json["constraints"]][:2] == ["volume", "displacement:mid:down"]   # as before: name and limit
    assert all(set(c) == {"name", "limit"} for c in run_json["constraints"])


def test_a_run_without_limits_has_an_empty_rows_block(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=3)
    out_dir = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out_dir), "--quiet"]) == 0
    assert json.loads((out_dir / "result" / "metrics.json").read_text())["rows"] == []


# --- a row that cannot work is a stop before the solve ------------------------------------


def test_rank_message_names_the_row_the_region_the_rank_and_the_options():
    reg = rm.RegionNodes("probe", np.array([3, 4]), np.array([[0.0, 0, 0], [1.0, 0, 0]]), 8, 0)
    fit = rm.rigid_fit(reg, np.arange(3 * 5, dtype=np.int64), 15)
    assert fit.rank == 5 and fit.rotation is None
    text = rm.rank_message("alignment row 'x'", "probe", fit)
    assert "alignment row 'x' cannot work" in text and "rank 5 of 6" in text and "two nodes" in text
    assert "Options: a larger region, a region that is not on one line, or a displacement row in its place" in text
    assert "\n" not in text


def test_check_gives_the_facts_of_the_regions(tmp_path):
    ok = check_data(limits_case(sub(tmp_path, "a"), ALL_ROWS))["warnings"]
    assert not [w for w in ok if w.startswith("limits:")]
    p = limits_case(sub(tmp_path, "b"), None)
    box(tmp_path / "b" / "inplate.stl", (-2, 2, 2), (0, 8, 8))
    d = yaml.safe_load(p.read_text())
    d["regions"]["measure"] = {"inplate": "inplate.stl"}
    d["limits"] = {"displacement": [{"region": "inplate", "limit_mm": 0.5}]}
    p.write_text(yaml.safe_dump(d))
    warnings = check_data(p)["warnings"]
    assert any(w.startswith("limits: ") and "every node of 'inplate' is fixed" in w for w in warnings)


def test_plain_text_check_prints_the_limits_fact(tmp_path, capsys):
    """`topopt check` without --json prints the `limits:` facts too (CONTRACTS.md section 31)."""
    p = keepout_case(tmp_path, ((10, 10, 0), (12, 12, 10)), ((8, 8, 4), (12, 12, 6)),
                     {"displacement": [{"region": "probe", "limit_mm": 0.5}]})
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    lines = [t for t in out.splitlines() if t.startswith("warning: limits: ")]
    assert len(lines) == 1 and "region 'probe': 10 of 18 nodes touch no solid or design voxel" in lines[0]
    assert "Options: to measure them, put the region on solid or on a load part" in lines[0]
    assert main(["check", str(p), "--json"]) == 0
    assert any(w.startswith("limits: ") for w in json.loads(capsys.readouterr().out)["warnings"])


def test_a_region_with_no_voxel_in_the_grid_is_a_user_error(tmp_path):
    p = limits_case(tmp_path, None)
    box(tmp_path / "far.stl", (100, 100, 100), (102, 102, 102))
    d = yaml.safe_load(p.read_text())
    d["regions"]["measure"] = {"far": "far.stl"}
    d["limits"] = {"displacement": [{"region": "far", "limit_mm": 1.0}]}
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="measure region 'far' has no voxel inside the grid at voxel_size 2.0"):
        check_data(p)


# --- the default run does not change ------------------------------------------------------


def test_the_default_formulation_has_no_motion_rows(tmp_path):
    m = Model(build_problem(load_config(make_case(tmp_path, h=2.0))))
    form = formulation_from_config(m.problem.config, m.n_design, m)
    assert [c.name for c in form.constraints] == ["volume"]
    m.solver.free()


def test_rows_are_added_after_the_compliance_limits(tmp_path):
    p = limits_case(tmp_path, {"displacement": [{"region": "mid", "limit_mm": 1.0, "cases": ["side"]}]})
    d = yaml.safe_load(p.read_text())
    d["load_cases"][1]["compliance_limit"] = 5.0
    p.write_text(yaml.safe_dump(d))
    cfg = load_config(p)
    form = formulation_from_config(cfg, 10)
    assert [c.name for c in form.constraints] == ["volume", "compliance:side", "displacement:mid:side"]
    row = form.constraints[2]
    assert (row.form, row.price, row.limit) == ("linear", rm.MOTION_PRICE, 1.0) and row.hint == rm.HINT


def test_a_config_without_rows_does_not_bind_even_with_a_model(tmp_path, monkeypatch):
    """The default path allocates nothing for the rows: `motion_set` returns before `bind_model`."""
    m = Model(build_problem(load_config(make_case(tmp_path, h=2.0))))
    monkeypatch.setattr(rm, "dof_map", lambda *a, **k: pytest.fail("dof_map called for a config with no row"))
    ms = rm.motion_set(m.problem.config, m, lambda t: pytest.fail(t))
    assert ms.specs == [] and ms.rows == [] and ms._bound_to is None and ms.constraints() == []
    m.solver.free()


def test_the_lazy_bind_sends_its_facts_to_the_log(tmp_path):
    """No model at `formulation_from_config`: the set binds at the first evaluation, and `log` gets the facts then."""
    p = keepout_case(tmp_path, ((10, 10, 0), (12, 12, 10)), ((8, 8, 4), (12, 12, 6)),
                     {"displacement": [{"region": "probe", "limit_mm": 0.5}]})
    m = Model(build_problem(load_config(p)))
    lines = []
    form = formulation_from_config(m.problem.config, m.n_design, None, lines.append)
    assert lines == []
    ev = evaluate(m, form, np.full(m.n_design, 0.5))
    assert len(ev.constraints) == 2
    assert len(lines) == 1 and lines[0].startswith("fact: region 'probe': 10 of 18 nodes")
    evaluate(m, form, np.full(m.n_design, 0.4))
    assert len(lines) == 1          # bound once
    m.solver.free()


def test_rows_without_a_model_bind_at_the_first_evaluation(tmp_path):
    """`formulation_from_config(config, n_design)` (no model) is what `run.json` uses: it needs no model."""
    m = make_model(tmp_path, ALL_ROWS)
    form = formulation_from_config(m.problem.config, m.n_design)
    assert len(form.constraints) == 1 + 10       # the volume and the ten rows of ALL_ROWS
    ev = evaluate(m, form, np.full(m.n_design, 0.5))
    assert len(ev.constraints) == len(form.constraints)
    m.solver.free()


def test_physical_values_equal_the_history_values_without_the_guard(tmp_path):
    """`MotionSet.physical_values(U)` (the end report of a robust run) is the history value without the guard."""
    m = make_model(tmp_path, ALL_ROWS, iterations=3)
    form = formulation_from_config(m.problem.config, m.n_design, m)
    seen = []
    res = minimize(m, form, on_iteration=lambda rec, point: seen.append(point))
    got = form.motion.physical_values(seen[-1].U)
    assert list(got) == [c["name"] for c in res.constraints[1:]]
    for c in res.constraints[1:]:
        assert got[c["name"]] == pytest.approx(rm.physical_value(c["name"], c["value"], c["limit"]), rel=1e-12)
    assert got["displacement_min:dn:down"] == -got["displacement_max:up:down"]
    row = form.motion.rows[0]
    assert row.measure(seen[-1]) == pytest.approx(res.constraints[1]["value"], rel=1e-14)
    m.solver.free()
