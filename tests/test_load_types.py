"""A4b: moment, remote force, pressure and per-axis supports (CONTRACTS.md section 26.2).

The cantilever of `make_case`: design x 0..20, y 0..10, z 0..10; plate x -2..0; tip box x 18..20, y 4..6, z 0..3.
At voxel size 1 the grid origin is (-2, 0, 0), and node (i, j, k) is at (i - 2, j, k).
"""

import json

import numpy as np
import pytest
import trimesh
import yaml

from topopt.config import UserError, load_config
from topopt.loads import (AUDIT, check_rigid_modes, exposed_faces, extended_facts, extended_lines,
                          lost_force, lost_force_warning, moment_spread, node_positions)
from topopt.optimize import Model
from topopt.problem import build_problem
from topopt.voxelize import DESIGN, EMPTY, FIXED, KEEP_OUT, LOAD

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def case(tmp_path, h=1.0, mutate=None, regions=None):
    """The cantilever. `regions` adds STL files: {name: (lo, hi)}. `mutate(d)` edits the YAML dict."""
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", (18, 4, 0), (20, 6, 3))
    for name, (lo, hi) in (regions or {}).items():
        box(tmp_path / f"{name}.stl", lo, hi)
    d = {
        "name": "cantilever",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
        "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
        "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 4, "solver": "amg"},
    }
    if mutate:
        mutate(d)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def prob_of(tmp_path, **kw):
    return build_problem(load_config(case(tmp_path, **kw)))


def sub(tmp_path, name):
    """A new empty folder, to build a second case in the same test."""
    (tmp_path / name).mkdir()
    return tmp_path / name


def entries(prob, c=0):
    return prob.extended["entries"][c]


# --- the YAML ---------------------------------------------------------------------------


def test_plain_yaml_is_not_extended(tmp_path):
    cfg = load_config(case(tmp_path))
    assert not cfg.uses_extended_loads() and not cfg.has_partial_supports()
    assert cfg.regions.fixed_axes == () and cfg.load_cases[0].pressures == ()
    f = cfg.load_cases[0].forces[0]
    assert f.moment is None and f.at is None
    assert build_problem(cfg).extended is None


@pytest.mark.parametrize("path", ["examples/hook/hook.yaml", "examples/ladle/ladle.yaml"])
def test_the_example_yamls_stay_valid(path):
    cfg = load_config(REPO / path)
    assert not cfg.uses_extended_loads()


def test_the_new_keys_parse(tmp_path):
    def mutate(d):
        d["regions"]["fixed"] = ["plate.stl", {"file": "roller.stl", "axes": ["Y", "z"]}]
        d["load_cases"] = [
            {"name": "a", "forces": [{"region": "tip", "moment": [0, 5.0, 0]}]},
            {"name": "b", "forces": [{"region": "tip", "force": [0, 0, -1], "at": [30, 5, 1],
                                      "coupling": "distributing"}]},
            {"name": "c", "pressures": [{"region": "tip", "value_mpa": 0.25}]},
        ]
    cfg = load_config(case(tmp_path, mutate=mutate, regions={"roller": ((-2, 0, 0), (0, 10, 10))}))
    assert cfg.regions.fixed_axes == ((0, 1, 2), (1, 2))
    assert len(cfg.regions.fixed) == 2 and all(p.name.endswith(".stl") for p in cfg.regions.fixed)
    a, b, c = cfg.load_cases
    assert a.forces[0].force == (0.0, 0.0, 0.0) and a.forces[0].moment == (0.0, 5.0, 0.0)
    assert b.forces[0].at == (30.0, 5.0, 1.0) and b.forces[0].moment is None
    assert c.forces == () and c.pressures[0].value_mpa == 0.25
    assert cfg.uses_extended_loads() and cfg.has_partial_supports()


@pytest.mark.parametrize("what,mutate", [
    ("moment", lambda d: d["load_cases"][0]["forces"][0].update(moment=[0, 1, 0])),
    ("at", lambda d: d["load_cases"][0]["forces"][0].update(at=[1, 2, 3])),
    ("pressure", lambda d: d["load_cases"][0].update(pressures=[{"region": "tip", "value_mpa": 1.0}])),
    ("axes", lambda d: d["regions"].update(fixed=[{"file": "plate.stl", "axes": ["y"]}])),
])
def test_each_new_key_makes_the_config_extended(tmp_path, what, mutate):
    assert load_config(case(tmp_path, mutate=mutate)).uses_extended_loads(), what


def test_all_three_axes_is_a_full_clamp(tmp_path):
    def mutate(d):
        d["regions"]["fixed"] = [{"file": "plate.stl", "axes": ["x", "y", "z"]}]
    cfg = load_config(case(tmp_path, mutate=mutate))
    assert cfg.regions.fixed_axes == ((0, 1, 2),) and not cfg.has_partial_supports()
    assert not cfg.uses_extended_loads()
    assert np.array_equal(build_problem(cfg).fixed_dofs, prob_of(sub(tmp_path, "plain")).fixed_dofs)


def _fm(d, **kw):
    d["load_cases"][0]["forces"][0].update(kw)


@pytest.mark.parametrize("mutate,text", [
    (lambda d: _fm(d, at=[1, 2, 3], force=None) or d["load_cases"][0]["forces"][0].pop("force"), "needs 'force'"),
    (lambda d: _fm(d, coupling="rigid"), "RBE3"),
    (lambda d: _fm(d, coupling="soft"), "'distributing'"),
    (lambda d: _fm(d, moment=[1, 2]), "3 numbers"),
    (lambda d: _fm(d, moment=[1, 2, "x"]), "moment[2]"),
    (lambda d: d["load_cases"][0]["forces"][0].pop("force"), "missing key 'force'"),
    (lambda d: d["load_cases"][0].pop("forces"), "missing key 'forces'"),
    (lambda d: d["load_cases"][0].update(forces=[]), "one force or more"),
    (lambda d: d["load_cases"][0].update(pressures=[{"region": "plate", "value_mpa": 1.0}]), "a fixed region"),
    (lambda d: d["load_cases"][0].update(pressures=[{"region": "design", "value_mpa": 1.0}]), "the design space"),
    (lambda d: d["load_cases"][0].update(pressures=[{"region": "zzz", "value_mpa": 1.0}]), "not a load region"),
    (lambda d: d["load_cases"][0].update(pressures=[{"region": "tip"}]), "missing key 'value_mpa'"),
    (lambda d: d["load_cases"][0].update(pressures=[{"region": "tip", "value_mpa": "x"}]), "value_mpa"),
    (lambda d: d["load_cases"][0].update(pressures=5), "must be a list"),
    (lambda d: d["regions"].update(fixed=[{"file": "plate.stl", "axes": ["w"]}]), "list of the axes"),
    (lambda d: d["regions"].update(fixed=[{"file": "plate.stl", "axes": ["x", "x"]}]), "distinct"),
    (lambda d: d["regions"].update(fixed=[{"file": "plate.stl", "axes": []}]), "list of the axes"),
    (lambda d: d["regions"].update(fixed=[{"file": "plate.stl"}]), "missing key 'axes'"),
])
def test_yaml_errors_of_the_new_keys(tmp_path, mutate, text):
    with pytest.raises(UserError) as e:
        load_config(case(tmp_path, mutate=mutate))
    assert text in str(e.value) and "(file:" in str(e.value)


def test_the_rigid_coupling_refusal_names_the_alternative(tmp_path):
    with pytest.raises(UserError) as e:
        load_config(case(tmp_path, mutate=lambda d: _fm(d, coupling="rigid")))
    msg = str(e.value)
    assert "RBE2" in msg and "RBE3" in msg and "distributing" in msg and "Options" in msg and "\n" not in msg


# --- moment and remote force --------------------------------------------------------------


def test_moment_spread_gives_force_free_moment_for_a_random_cloud():
    rng = np.random.default_rng(3)
    r = rng.normal(size=(40, 3)) * [5.0, 2.0, 1.0]
    r -= r.mean(axis=0)
    m = np.array([3.0, -7.0, 11.0])
    f = moment_spread(r, m)
    assert np.abs(f.sum(axis=0)).max() < AUDIT * np.abs(f).sum()
    assert np.abs(np.cross(r, f).sum(axis=0) - m).max() < AUDIT * np.abs(m).max()


def test_collinear_nodes_cannot_carry_the_moment_about_their_line():
    r = np.zeros((5, 3))
    r[:, 0] = np.arange(5) - 2.0       # nodes on the x axis
    with pytest.raises(UserError) as e:
        moment_spread(r, np.array([4.0, 0.0, 0.0]), "load region 'bar'")
    msg = str(e.value)
    assert "load region 'bar'" in msg and "cannot carry the moment" in msg and "Options" in msg
    f = moment_spread(r, np.array([0.0, 4.0, 0.0]))     # a moment about y is carried
    assert np.abs(np.cross(r, f).sum(axis=0) - [0, 4.0, 0]).max() < 1e-12


def test_a_moment_on_a_region_audits_to_1e_9(tmp_path):
    m = np.array([10.0, -20.0, 30.0])
    prob = prob_of(tmp_path, mutate=lambda d: d["load_cases"].append(
        {"name": "mom", "forces": [{"region": "tip", "moment": m.tolist()}]}))
    e, = entries(prob, 1)
    assert e.kind == "moment" and np.array_equal(e.force_n, [0, 0, 0])
    f3 = prob.forces[1].reshape(-1, 3)
    assert np.abs(f3.sum(axis=0)).max() < AUDIT * np.abs(f3).sum()
    pos = node_positions(np.arange(f3.shape[0]), prob.shape, prob.origin, prob.h)
    centroid = pos[e.nodes].mean(axis=0)
    assert np.allclose(e.about, centroid, atol=1e-12)
    got = np.cross(pos - centroid, f3).sum(axis=0)
    assert np.abs(got - m).max() < AUDIT * np.abs(m).max()
    assert e.info["audit_force_rel"] < AUDIT and e.info["audit_moment_rel"] < AUDIT
    assert set(np.flatnonzero(np.abs(f3).sum(axis=1))) <= set(e.nodes.tolist())    # only the region nodes
    assert np.array_equal(prob.forces[0], prob_of(sub(tmp_path, "plain")).forces[0])    # the other case is unchanged


def test_a_remote_force_is_the_force_plus_its_moment_about_the_centroid(tmp_path):
    F, at = np.array([1.0, -2.0, 3.0]), np.array([30.0, 7.0, -4.0])
    prob = prob_of(tmp_path, mutate=lambda d: d["load_cases"].append(
        {"name": "far", "forces": [{"region": "tip", "force": F.tolist(), "at": at.tolist()}]}))
    e, = entries(prob, 1)
    assert e.kind == "remote_force"
    assert np.allclose(e.moment_nmm, np.cross(at - e.about, F), atol=1e-12)
    f3 = prob.forces[1].reshape(-1, 3)
    pos = node_positions(np.arange(f3.shape[0]), prob.shape, prob.origin, prob.h)
    assert np.allclose(f3.sum(axis=0), F, atol=1e-12)
    # The line of action passes through `at`: the nodal forces have no moment about that point.
    assert np.abs(np.cross(pos - at, f3).sum(axis=0)).max() < 1e-9 * np.abs(f3).sum() * 30


def test_a_remote_force_at_the_centroid_equals_a_plain_force(tmp_path):
    def one(d, **kw):
        d["load_cases"] = [{"name": "a", "forces": [{"region": "tip", "force": [1.0, 2.0, 3.0], **kw}]}]
    prob = prob_of(tmp_path, mutate=lambda d: one(d, at=[19.0, 5.0, 1.5]))     # the node centroid of the tip
    plain = prob_of(sub(tmp_path, "plain"), mutate=lambda d: one(d))
    assert np.allclose(prob.forces, plain.forces, atol=1e-14)


def test_a_moment_adds_no_constraint_and_no_stiffness(tmp_path):
    base = prob_of(tmp_path)
    mom = prob_of(sub(tmp_path, "m"), mutate=lambda d: d["load_cases"].append(
        {"name": "mom", "forces": [{"region": "tip", "moment": [0, 9.0, 0]}]}))
    assert np.array_equal(base.fixed_dofs, mom.fixed_dofs) and np.array_equal(base.labels, mom.labels)
    assert np.array_equal(base.forces[0], mom.forces[0])


def test_end_moment_on_a_beam_matches_beam_theory(tmp_path):
    """A bar 40 x 8 x 8 mm, fixed at x = 0, moment M about y on the end slab: w(x) = -M x^2 / (2 E I)."""
    h, M, L, b = 2.0, 1000.0, 40.0, 8.0
    box(tmp_path / "design.stl", (0, 0, 0), (L, b, b))
    box(tmp_path / "plate.stl", (-2 * h, 0, 0), (0, b, b))
    box(tmp_path / "tip.stl", (L - 2 * h, 0, 0), (L, b, b))
    d = {"name": "beam", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "m", "forces": [{"region": "tip", "moment": [0, M, 0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 2, "solver": "amg"}}
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(d))
    prob = build_problem(load_config(tmp_path / "b.yaml"))
    model = Model(prob)
    model.solver.rtol = 1e-12
    U = model.solve(model.stiffness(np.ones(model.n_design)))[:, 0]
    model.solver.free()
    u = np.zeros(model.n_dof_all)
    u[model.free_dofs] = U
    nx, ny, nz = prob.shape
    i = int(round((L - h - prob.origin[0]) / h))      # the node plane at the centroid of the end slab
    w = u[3 * ((i * (ny + 1) + ny // 2) * (nz + 1) + nz // 2) + 2]
    I = b * b ** 3 / 12.0
    exact = -M * (L - h) ** 2 / (2 * model.E * I)
    assert w < 0 and 0.9 < w / exact < 1.02          # the first order hex elements are a little stiff in bending
    c = float(model.f_red[:, 0] @ U)                  # compliance = M theta
    assert 0.9 < c / (M * M * (L - h) / (model.E * I)) < 1.02


# --- pressure ------------------------------------------------------------------------------


def test_pressure_on_the_tip_by_hand(tmp_path):
    p = 0.5
    prob = prob_of(tmp_path, mutate=lambda d: d["load_cases"].append(
        {"name": "p", "pressures": [{"region": "tip", "value_mpa": p}]}))
    e, = entries(prob, 1)
    # Tip x 18..20, y 4..6, z 0..3: its +x face (x = 20) and its -z face (z = 0) are on the grid border.
    assert e.kind == "pressure" and e.info["faces"] == 6 + 4 and e.info["area_mm2"] == 10.0
    assert np.allclose(e.force_n, [-p * 6.0, 0.0, p * 4.0])
    nodes = {tuple(pos): f for pos, f in zip(e.pos, e.f)}     # position (mm) -> nodal force
    assert len(nodes) == 12 + 9 - 3                           # the nodes of the two faces, 3 shared on their edge
    assert np.allclose(nodes[(18.0, 4.0, 0.0)], [0.0, 0.0, p / 4])          # one -z face
    assert np.allclose(nodes[(18.0, 5.0, 0.0)], [0.0, 0.0, 2 * p / 4])      # two -z faces
    assert np.allclose(nodes[(20.0, 5.0, 0.0)], [-2 * p / 4, 0.0, 2 * p / 4])   # two +x faces and two -z faces
    assert np.allclose(nodes[(20.0, 5.0, 1.0)], [-4 * p / 4, 0.0, 0.0])    # four +x faces: they push in (-x)
    assert np.allclose(prob.forces[1].reshape(-1, 3).sum(axis=0), e.force_n)


def test_pressure_resultant_is_p_times_the_projected_area_of_a_dome(tmp_path):
    """A solid ball whose upper half stands out of the design box: the resultant is p x the footprint area."""
    h, p = 1.0, 0.8
    ball = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    ball.apply_translation((10, 10, 10))
    ball.export(tmp_path / "ball.stl")
    box(tmp_path / "design.stl", (0, 0, 0), (20, 20, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 20, 10))
    d = {"name": "dome", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"ball": "ball.stl"}},
         "load_cases": [{"name": "p", "pressures": [{"region": "ball", "value_mpa": p}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 2, "solver": "amg"}}
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(d))
    prob = build_problem(load_config(tmp_path / "d.yaml"))
    e, = entries(prob)
    # The footprint: the columns of the ball that reach above the design box (z index 10 and up).
    top = prob.labels[..., int(round((10 - prob.origin[2]) / h)):]
    columns = int((top == LOAD).any(axis=2).sum())
    assert columns > 20
    assert np.allclose(e.force_n, [0.0, 0.0, -p * columns * h * h], atol=1e-12)
    assert np.allclose(prob.forces[0].reshape(-1, 3).sum(axis=0), e.force_n, atol=1e-12)
    # Faces toward the design box get no pressure: a node deep in the ball is not loaded.
    assert e.info["audit_force_rel"] < AUDIT


def test_pressure_faces_toward_design_fixed_and_other_load_get_none():
    lab = np.full((5, 3, 3), EMPTY, dtype=np.uint8)
    lab[0, :, :] = FIXED
    lab[1, :, :] = DESIGN
    lab[2, 1, 1] = LOAD                          # voxel A: design on its -x side
    lab[2, 1, 2] = LOAD                          # voxel B above A (the same region): outside the grid on its +z side
    lab[3, 1, 1] = KEEP_OUT                      # on the +x side of A
    got = {(a, s): int(m.sum()) for a, s, m in exposed_faces(lab, lab == LOAD)}
    assert got[(0, -1)] == 0                     # toward design: none
    assert got[(0, +1)] == 2                     # A toward keep-out, B toward empty
    assert got[(1, -1)] == 2 and got[(1, +1)] == 2     # toward empty
    assert got[(2, -1)] == 1 and got[(2, +1)] == 1     # A toward empty (z index 0), B toward the outside of the grid
    lab[2, 1, 0] = FIXED                         # a fixed voxel under A
    got = {(a, s): int(m.sum()) for a, s, m in exposed_faces(lab, lab == LOAD)}
    assert got[(2, -1)] == 0                     # toward fixed: none


def test_pressure_on_a_region_with_no_free_face_is_refused(tmp_path):
    def mutate(d):
        d["regions"]["load"]["core"] = "core.stl"
        d["load_cases"].append({"name": "p", "pressures": [{"region": "core", "value_mpa": 1.0}]})
    with pytest.raises(UserError) as e:
        prob_of(tmp_path, mutate=mutate, regions={"core": ((8, 3, 3), (12, 7, 7))})
    msg = str(e.value)
    assert "'core'" in msg and "no face" in msg and "Options" in msg


# --- per-axis supports ----------------------------------------------------------------------


def _nid(shape, i, j, k):
    return (i * (shape[1] + 1) + j) * (shape[2] + 1) + k


def test_per_axis_support_fixes_only_its_axes_by_hand(tmp_path):
    prob = prob_of(tmp_path, mutate=lambda d: d["regions"].update(fixed=[
        {"file": "plate.stl", "axes": ["y"]}, {"file": "pin.stl", "axes": ["x", "z"]}]),
        regions={"pin": ((0, 0, 0), (1, 10, 10))})
    s = prob.shape
    # plate voxels i 0..1 (nodes i 0..2); pin voxel i = 2 (nodes i 2..3). Node (2, ., .) is in both.
    expect = set()
    for i in range(3):
        for j in range(11):
            for k in range(11):
                expect.add(3 * _nid(s, i, j, k) + 1)
    for i in range(2, 4):
        for j in range(11):
            for k in range(11):
                expect.update([3 * _nid(s, i, j, k), 3 * _nid(s, i, j, k) + 2])
    assert prob.fixed_dofs.dtype == np.int64 and prob.fixed_dofs.tolist() == sorted(expect)


def test_per_axis_support_follows_the_label_priority(tmp_path):
    def mutate(d):
        d["regions"]["fixed"] = [{"file": "plate.stl", "axes": ["y", "z"]}, {"file": "pin.stl", "axes": ["x"]}]
        d["regions"]["keep_out"] = ["cut.stl"]
    # A cut through the plate (j 6..9): its voxels are KEEP_OUT, so they are not part of the support.
    prob = prob_of(tmp_path, mutate=mutate, regions={"cut": ((-2, 6, 0), (0, 10, 10)), "pin": ((0, 0, 0), (1, 10, 10))})
    s = prob.shape
    dofs = prob.fixed_dofs
    node, comp = dofs // 3, dofs % 3
    i, j = node // ((s[1] + 1) * (s[2] + 1)), (node // (s[2] + 1)) % (s[1] + 1)
    assert j[comp > 0].max() == 6 and set(i[comp > 0]) == {0, 1, 2}          # y, z: the plate voxels j 0..5, nodes j 0..6
    assert j[comp == 0].max() == 10 and set(i[comp == 0]) == {2, 3}          # x: the pin, not cut


def test_a_roller_alone_leaves_three_rigid_modes_free(tmp_path):
    with pytest.raises(UserError) as e:
        prob_of(tmp_path, mutate=lambda d: d["regions"].update(fixed=[{"file": "plate.stl", "axes": ["x"]}]))
    msg = str(e.value)
    assert "3 of the 6 rigid-body modes free" in msg
    assert "translation along y" in msg and "translation along z" in msg and "rotation about x" in msg
    assert "translation along x" not in msg and "Options" in msg and "\n" not in msg


def test_a_single_axis_support_can_be_enough_with_others(tmp_path):
    """Three rollers on three planes block all six modes."""
    def mutate(d):
        d["regions"]["fixed"] = [{"file": "plate.stl", "axes": ["x"]},
                                 {"file": "ry.stl", "axes": ["y"]}, {"file": "rz.stl", "axes": ["z"]}]
    prob = prob_of(tmp_path, mutate=mutate,
                   regions={"ry": ((0, -2, 0), (20, 0, 10)), "rz": ((0, 0, -2), (20, 10, 0))})
    facts = extended_facts(prob)["supports"]
    assert facts["free_modes"] == 0 and facts["singular_value_ratio"] > 1e-4
    assert facts["axes"] == {"plate.stl": "x", "ry.stl": "y", "rz.stl": "z"}
    assert np.all(np.diff(prob.fixed_dofs) > 0)


def test_rigid_mode_test_on_plain_cases():
    shape = (4, 4, 4)
    full = np.array([3 * ((i * 5 + j) * 5 + k) + a for i in (0,) for j in range(5) for k in range(5) for a in range(3)])
    assert check_rigid_modes(full, shape, 1.0)["free_modes"] == 0
    line = np.array([3 * ((0 * 5 + 0) * 5 + k) + a for k in range(5) for a in range(3)])    # nodes on one line
    with pytest.raises(UserError) as e:
        check_rigid_modes(line, shape, 1.0)
    assert "rotation about z" in str(e.value)
    one = np.array([0, 1, 2])                                              # one node
    with pytest.raises(UserError):
        check_rigid_modes(one, shape, 1.0)


def test_an_under_constrained_yaml_ends_in_a_user_error_for_run_and_check(tmp_path, capsys):
    from topopt.cli import main
    p = case(tmp_path, mutate=lambda d: d["regions"].update(fixed=[{"file": "plate.stl", "axes": ["y", "z"]}]))
    assert main(["check", str(p)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "rigid-body modes free" in err and len(err.strip().splitlines()) == 1
    assert main(["run", str(p), "--out", str(tmp_path / "r")]) == 1
    status = json.loads((tmp_path / "r" / "status.json").read_text())
    assert status["state"] == "error" and "rigid-body" in status["message"]


# --- lost force and moment ------------------------------------------------------------------


def touch_case(d):
    """A load box that touches the plate: x 0..2, y 4..6, z 0..3 shares its x = 0 nodes with the plate."""
    d["regions"]["load"] = {"tip": "touch.stl"}
    d["load_cases"] = [{"name": "a", "forces": [{"region": "tip", "force": [1.0, 2.0, 3.0]}]}]


def test_lost_force_counts_only_the_fixed_axes(tmp_path):
    def mutate(d):
        touch_case(d)
        d["regions"]["fixed"] = [{"file": "plate.stl", "axes": ["y"]}, {"file": "pin.stl", "axes": ["x", "z"]}]
    # The box has 3 x 3 x 4 = 36 nodes. The 12 nodes at x = 0 are on the plate, which fixes only y.
    # The pin (x 0..1, y 0..2) does not touch the box (y 4..6). It makes the supports complete.
    prob = prob_of(tmp_path, mutate=mutate, regions={"touch": ((0, 4, 0), (2, 6, 3)), "pin": ((0, 0, 0), (1, 2, 10))})
    lf = lost_force(prob.config, prob.load_region_ids, prob.fixed_dofs, prob.forces)
    k = lf["cases"][0]
    assert np.allclose(k["lost_n"], [0.0, 2.0 * 12 / 36, 0.0], atol=1e-12)
    assert k["regions"][0]["fixed_nodes"] == 12 and np.allclose(k["regions"][0]["lost_n"], [0.0, 2.0 * 12 / 36, 0.0])
    assert lost_force_warning(lf) is not None


def test_full_clamp_lost_force_is_unchanged_by_the_per_axis_code(tmp_path):
    prob = prob_of(tmp_path, mutate=touch_case, regions={"touch": ((0, 4, 0), (2, 6, 3))})
    lf = lost_force(prob.config, prob.load_region_ids, prob.fixed_dofs, prob.forces)
    k = lf["cases"][0]
    assert np.allclose(k["lost_n"], np.array([1.0, 2.0, 3.0]) * 12 / 36) and "lost_moment_nmm" not in k


def test_a_lost_moment_is_a_fact(tmp_path):
    from topopt.cli import lost_force_lines
    M = np.array([0.0, 40.0, 0.0])

    def mutate(d):
        d["regions"]["load"] = {"tip": "touch.stl"}
        d["load_cases"] = [{"name": "m", "forces": [{"region": "tip", "moment": M.tolist()}]}]
    prob = prob_of(tmp_path, mutate=mutate, regions={"touch": ((0, 4, 0), (2, 6, 3))})
    lf = lost_force(prob.config, prob.load_region_ids, prob.fixed_dofs, prob.forces, prob.extended["entries"])
    k = lf["cases"][0]
    f3 = prob.forces[0].reshape(-1, 3)
    e, = prob.extended["entries"][0]
    fixed_nodes = np.unique(prob.fixed_dofs // 3)
    pos = node_positions(np.arange(f3.shape[0]), prob.shape, prob.origin, prob.h)
    about_origin = np.cross(pos[fixed_nodes], f3[fixed_nodes]).sum(axis=0)                 # brute force
    about_centroid = np.cross(pos[fixed_nodes] - e.about, f3[fixed_nodes]).sum(axis=0)
    assert np.allclose(k["lost_moment_nmm"], about_origin, atol=1e-12) and k["lost_moment_magnitude_nmm"] > 0
    assert k["lost_moment_about_mm"] == [0.0, 0.0, 0.0]            # the case value is about one point: the origin
    assert np.allclose(k["regions"][0]["lost_moment_nmm"], about_centroid, atol=1e-12)
    assert k["lost_moment_percent"] > 0.0
    assert k["regions"][0]["moment_about_mm"] == [float(v) for v in e.about]
    assert "lost moment" in lost_force_warning(lf)
    assert any(line.startswith("lost moment:") for line in lost_force_lines(lf))
    # No overlap: no lost moment, no warning.
    ok = prob_of(sub(tmp_path, "ok"), mutate=lambda d: d["load_cases"].append(
        {"name": "m", "forces": [{"region": "tip", "moment": M.tolist()}]}))
    lf2 = lost_force(ok.config, ok.load_region_ids, ok.fixed_dofs, ok.forces, ok.extended["entries"])
    assert lf2["cases"][1]["lost_moment_magnitude_nmm"] == 0.0 and lost_force_warning(lf2) is None
    # A case of plain forces has the keys of a YAML without the new types, also next to a case with a moment.
    assert "lost_moment_nmm" not in lf2["cases"][0] and "kind" not in str(lf2["cases"][0]["regions"])


@pytest.mark.parametrize("moment", [[0.0, 40.0, 0.0], [0.0, 0.0, 30.0], [200.0, 0.0, 0.0]])
def test_a_pure_moment_on_the_plate_gives_finite_numbers_and_the_moment_text(tmp_path, moment):
    """A case with no resultant force: no percent of the resultant (it would be of round-off, up to 1e18)."""
    from topopt.cli import lost_force_lines

    def mutate(d):
        d["regions"]["load"] = {"tip": "touch.stl"}
        d["load_cases"] = [{"name": "twist", "forces": [{"region": "tip", "moment": moment}]}]
    prob = prob_of(tmp_path, mutate=mutate, regions={"touch": ((0, 4, 0), (2, 6, 3))})
    lf = lost_force(prob.config, prob.load_region_ids, prob.fixed_dofs, prob.forces, prob.extended["entries"])
    k, = lf["cases"]
    scale = float(np.linalg.norm(prob.extended["entries"][0][0].f, axis=1).sum())
    # The lost force is a real net force of the dropped part of the spread, or round-off (0) when it cancels.
    assert k["lost_magnitude_n"] == 0.0 or k["lost_magnitude_n"] > 1e-9 * scale
    assert k["lost_percent_of_sum"] == pytest.approx(100.0 * k["lost_magnitude_n"] / scale)
    assert lf["max_lost_percent"] < 1000.0 and np.isfinite(lf["max_lost_percent"])
    assert k["lost_moment_magnitude_nmm"] > 0 and 0.0 < k["lost_moment_percent"] < 1000.0
    text = lost_force_warning(lf)
    assert "no resultant force" not in text or k["lost_magnitude_n"] > 0     # said only with a lost force
    assert "cancel" not in text and "lost moment" in text
    lines = lost_force_lines(lf)
    assert any(ln.startswith("lost moment:") and "about the origin" in ln for ln in lines)
    if k["lost_magnitude_n"] > 0:
        assert k["lost_percent"] is None
        assert any("no resultant force, a moment" in ln for ln in lines)


def test_the_run_log_has_the_warning_prefix_for_a_lost_force_and_a_lost_moment(tmp_path):
    """m3: a reader of the `warning:` lines of run.log also sees a lost moment, in the same way as a lost force."""
    from topopt.cli import main

    def mutate(d):
        d["regions"]["load"] = {"tip": "touch.stl"}
        d["load_cases"] = [{"name": "twist", "forces": [{"region": "tip", "moment": [0.0, 40.0, 0.0]}]}]
    p = case(tmp_path, mutate=mutate, regions={"touch": ((0, 4, 0), (2, 6, 3))})
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text().splitlines()
    assert any(ln.startswith("warning: lost force: twist:") and "no resultant force" in ln for ln in log)
    assert any(ln.startswith("warning: lost moment: twist:") and "about the origin" in ln for ln in log)
    assert not any(ln.startswith(("lost force:", "lost moment:")) for ln in log)     # no line without the prefix


# --- run.json, the facts and the stiffness --------------------------------------------------


def test_run_json_has_kind_and_the_resultant(tmp_path):
    from topopt.snapshot import RunWriter
    def mutate(d):
        d["load_cases"] = [
            {"name": "plain", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]},
            {"name": "mom", "forces": [{"region": "tip", "moment": [0, 5.0, 0]}]},
            {"name": "far", "forces": [{"region": "tip", "force": [0, 0, -1.0], "at": [30, 5, 1]}]},
            {"name": "both", "forces": [{"region": "tip", "force": [1, 0, 0], "moment": [0, 0, 2.0]}]},
            {"name": "press", "pressures": [{"region": "tip", "value_mpa": 0.5}]},
        ]
    prob = prob_of(tmp_path, mutate=mutate)
    (tmp_path / "out").mkdir()
    RunWriter(tmp_path / "out", 0.0, quiet=True).write_run_json(prob, 100, 100, "amg")
    run = json.loads((tmp_path / "out" / "run.json").read_text())
    plain, mom, far, both, press = (c["forces"][0] for c in run["load_cases"])
    assert set(plain) == {"region", "force", "center", "kind"} and plain["kind"] == "force"
    assert mom["kind"] == "moment" and mom["force"] == [0, 0, 0] and mom["moment"] == [0, 5.0, 0]
    assert far["kind"] == "remote_force" and far["at"] == [30, 5, 1] and far["force"] == [0, 0, -1.0]
    assert far["moment_about_mm"] == pytest.approx(mom["moment_about_mm"])
    assert both["kind"] == "force_moment" and both["force"] == [1, 0, 0] and "at" not in both
    assert press["kind"] == "pressure" and press["pressure_mpa"] == 0.5 and press["area_mm2"] == 10.0
    assert press["force"] == pytest.approx([-3.0, 0.0, 2.0])
    assert run["derived"]["load_cases"][4]["pressures"][0]["faces"] == 10
    assert run["lost_force"]["cases"][1]["lost_moment_magnitude_nmm"] == 0.0
    assert "supports" not in run


def test_run_json_lists_the_supports(tmp_path):
    from topopt.snapshot import RunWriter
    def mutate(d):
        d["regions"]["fixed"] = [{"file": "plate.stl", "axes": ["x"]}, {"file": "ry.stl", "axes": ["y"]},
                                 {"file": "rz.stl", "axes": ["z"]}]
    prob = prob_of(tmp_path, mutate=mutate,
                   regions={"ry": ((0, -2, 0), (20, 0, 10)), "rz": ((0, 0, -2), (20, 10, 0))})
    (tmp_path / "out").mkdir()
    RunWriter(tmp_path / "out", 0.0, quiet=True).write_run_json(prob, 100, 100, "amg")
    run = json.loads((tmp_path / "out" / "run.json").read_text())
    assert run["supports"]["axes"] == {"plate.stl": "x", "ry.stl": "y", "rz.stl": "z"}
    assert run["supports"]["free_modes"] == 0


def test_the_stiffness_is_null_with_a_reason_for_a_moment_or_a_pressure(tmp_path):
    from test_post import make_hole_run
    from topopt import post
    run, _ = make_hole_run(tmp_path)
    run["load_cases"] = [
        {"name": "plain", "weight": 1.0, "forces": [{"region": "t", "force": [0, 0, -3.0], "center": [0, 0, 0],
                                                      "kind": "force"}]},
        {"name": "mom", "weight": 1.0, "forces": [{"region": "t", "force": [0, 0, 0], "center": [0, 0, 0],
                                                    "kind": "moment", "moment": [0, 4.0, 0]}]},
        {"name": "mixed", "weight": 1.0, "forces": [
            {"region": "t", "force": [0, 0, -3.0], "center": [0, 0, 0], "kind": "force"},
            {"region": "t", "force": [0, 0, 1.0], "center": [0, 0, 0], "kind": "pressure"}]},
        {"name": "old", "weight": 1.0, "forces": [{"region": "t", "force": [0, 0, -3.0], "center": [0, 0, 0]}]},
    ]
    (tmp_path / "run.json").write_text(json.dumps(run))
    hist = [{"iteration": i, "elapsed_s": 1.0 + i, "compliance": 10.0 / (i + 1), "compliance_cases": [5.0] * 4}
            for i in range(2)]
    (tmp_path / "history.jsonl").write_text("\n".join(json.dumps(r) for r in hist) + "\n")
    st = {c["name"]: c for c in post.finish(tmp_path)["stiffness_cases"]}
    assert st["plain"] == {"name": "plain", "N_per_mm": 9.0 / 5.0}
    assert st["old"] == {"name": "old", "N_per_mm": 9.0 / 5.0}           # no `kind`: a plain force, as before
    for name, word in (("mom", "moment"), ("mixed", "pressure")):
        assert st[name]["N_per_mm"] is None and word in st[name]["reason"]


def test_check_data_and_the_text_lines_give_the_facts(tmp_path):
    from topopt.cli import check_data
    p = case(tmp_path, mutate=lambda d: d["load_cases"].extend([
        {"name": "mom", "forces": [{"region": "tip", "moment": [0, 6.0, 0]}]},
        {"name": "p", "pressures": [{"region": "tip", "value_mpa": 0.5}]}]))
    d = check_data(p)
    lines = extended_lines(d["extended"])
    assert any(l.startswith("load:") and "moment [0, 6, 0] N mm" in l and "audit" in l for l in lines)
    assert any(l.startswith("pressure:") and "10 faces" in l and "p x area" in l for l in lines)
    ex = d["extended"]["loads"][1]["entries"][0]
    assert ex["audit_moment_rel"] < AUDIT and ex["kind"] == "moment"
    assert d["derived"]["load_cases"][2]["pressures"][0]["force_n"] == pytest.approx([-3.0, 0.0, 2.0])
    json.dumps(d, default=str)                      # the facts are JSON
    plain = check_data(case(sub(tmp_path, "plain")))
    assert "extended" not in plain and "kind" not in plain["derived"]["load_cases"][0]["forces"][0]


# --- the solver paths -----------------------------------------------------------------------


def mixed_case(d):
    d["load_cases"] = [
        {"name": "mixed", "weight": 2.0, "forces": [
            {"region": "tip", "force": [1.0, 0.5, -2.0], "at": [24.0, 5.0, 3.0]},
            {"region": "tip", "moment": [3.0, -4.0, 5.0]}],
         "pressures": [{"region": "tip", "value_mpa": 0.3}]},
        {"name": "side", "forces": [{"region": "tip", "force": [0, 1.0, 0]}]}]
    d["regions"]["fixed"] = ["plate.stl", {"file": "side.stl", "axes": ["y"]}]


def test_gradients_match_finite_differences_with_the_new_loads(tmp_path):
    prob = prob_of(tmp_path, h=2.0, mutate=mixed_case, regions={"side": ((0, -2, 0), (4, 0, 10))})
    m = Model(prob)
    eps, tol = 1e-6, 1e-5
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13        # the check needs a tight solve (KB pitfall fd-check-noise-floor)
        eps, tol = 1e-3, 1e-4        # AMG-CG is not exact: a bigger step and a looser limit
    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 0.7, m.n_design)
    c0, dc, _, _ = m.evaluate(x, 2.0)
    assert np.isfinite(c0) and dc.shape == (m.n_design,)
    floor = 1e-2 * np.abs(dc).max()      # a small slope is judged on the absolute scale of the others
    for e in rng.choice(m.n_design, size=min(12, m.n_design), replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        fd = (m.evaluate(xp, 2.0)[0] - m.evaluate(xm, 2.0)[0]) / (2 * eps)
        assert abs(fd - dc[e]) < tol * max(abs(dc[e]), floor)
    m.solver.free()


def _compliance(prob, solver):
    from dataclasses import replace
    cfg = prob.config
    cfg = replace(cfg, settings=replace(cfg.settings, solver=solver))
    m = Model(replace(prob, config=cfg))
    if solver == "amg":
        m.solver.rtol = 1e-12
    c, _, _, _ = m.evaluate(np.full(m.n_design, 0.5), 1.0)
    m.solver.free()
    return float(np.sum(c))


def test_the_direct_solver_of_the_platform_agrees_with_amg_on_the_new_loads(tmp_path):
    """pardiso (Linux x86_64 with the extra) or cholesky (macOS with Accelerate); none: the test skips."""
    from topopt.solver_choice import resolve_solver
    try:
        direct = resolve_solver("auto")
    except UserError as e:
        pytest.skip(str(e))
    if direct == "amg":
        pytest.skip("this host has no direct solver (no pypardiso, no Accelerate shim)")
    prob = prob_of(tmp_path, h=2.0, mutate=mixed_case, regions={"side": ((0, -2, 0), (4, 0, 10))})
    assert _compliance(prob, "amg") == pytest.approx(_compliance(prob, direct), rel=1e-6)


def _bar(tmp_path, tag, h, half, slab, load, L=40.0, b=8.0):
    """A bar x 0..L, section b x b, clamped at x = 0 on a plate, loaded on its end slab. Full: y -b/2..b/2.
    Half: y 0..b/2 with half the force and a roller slab (axes y) at the plane y = 0:
    'inside' is y 0..h (solid voxels of the part), 'outside' is y -h..0. Return the compliance at rho = 1."""
    d = sub(tmp_path, tag)
    y0 = 0.0 if half else -b / 2
    box(d / "design.stl", (0, y0, 0), (L, b / 2, b))
    box(d / "plate.stl", (-2 * h, y0, 0), (0, b / 2, b))
    box(d / "tip.stl", (L - 2 * h, y0, 0), (L, b / 2, b))
    fixed = ["plate.stl"]
    if half:
        lo, hi = ((0, 0, 0), (L, h, b)) if slab == "inside" else ((0, -h, 0), (L, 0, b))
        box(d / "roll.stl", lo, hi)
        fixed.append({"file": "roll.stl", "axes": ["y"]})
    force = np.array(load, float) * (0.5 if half else 1.0)
    y = {"name": tag, "regions": {"design": "design.stl", "fixed": fixed, "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "c", "forces": [{"region": "tip", "force": force.tolist()}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 2, "solver": "amg"}}
    (d / "bar.yaml").write_text(yaml.safe_dump(y))
    prob = build_problem(load_config(d / "bar.yaml"))
    if half:
        assert prob.extended["supports"]["free_modes"] == 0
    m = Model(prob)
    m.solver.rtol = 1e-12
    U = m.solve(m.stiffness(np.ones(m.n_design)))
    c = float(np.einsum("ij,ij->j", m.f_red, U)[0])
    m.solver.free()
    return c


@pytest.mark.parametrize("load", [[0.0, 0.0, -10.0], [10.0, 0.0, 0.0]], ids=["bend_z", "pull_x"])
def test_a_roller_plane_gives_the_symmetry_result(tmp_path, load):
    """The half bar with a roller slab at the plane y = 0 against the full bar: 2 c_half / c_full is 1 for an exact plane.

    A per-axis support region is a solid body (label FIXED, in the final mesh). Its nodes are fixed along the named axes
    only, so it carries load along the other axes and adds its stiffness. Measured at h = 1 mm (pardiso, rho = 1):
    one voxel inside the part 0.969 (bend) and 0.968 (pull); one voxel outside the part 0.788 and 0.783, and the
    outside slab does not converge with h (0.878 at h = 0.5 mm): it is extra material. The slab goes inside the part.
    """
    h = 1.0
    full = _bar(tmp_path, "full", h, False, None, load)
    inside = 2.0 * _bar(tmp_path, "inside", h, True, "inside", load) / full
    outside = 2.0 * _bar(tmp_path, "outside", h, True, "outside", load) / full
    assert abs(inside - 1.0) < 0.05              # the limit: 5 percent at h = 1 mm
    assert outside < 0.9 < inside                # the outside slab makes the half model 12 to 22 percent too stiff


@pytest.mark.parametrize("h, low", [(2.0, 0.92), (1.0, 0.97)])
def test_a_torque_gives_the_rotation_of_beam_theory(tmp_path, h, low):
    """Bar 80 x 8 x 8, torque T about x on the end slab: the rotation of the loaded nodes against T x_c / (G J),
    J = 0.1406 a^4. Measured (amg, rho = 1): 0.939 at h = 2, 0.981 at h = 1, 0.994 at h = 0.5 mm. It converges to 1."""
    L, b, T = 80.0, 8.0, 1000.0
    box(tmp_path / "design.stl", (0, 0, 0), (L, b, b))
    box(tmp_path / "plate.stl", (-2 * h, 0, 0), (0, b, b))
    box(tmp_path / "tip.stl", (L - 2 * h, 0, 0), (L, b, b))
    y = {"name": "torsion", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "t", "forces": [{"region": "tip", "moment": [T, 0.0, 0.0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 2, "solver": "amg"}}
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(y))
    prob = build_problem(load_config(tmp_path / "t.yaml"))
    m = Model(prob)
    m.solver.rtol = 1e-12
    U = m.solve(m.stiffness(np.ones(m.n_design)))
    c = float(np.einsum("ij,ij->j", m.f_red, U)[0])      # c = T x (the rotation of the loaded nodes)
    m.solver.free()
    e, = prob.extended["entries"][0]
    G = prob.config.material.E / (2 * (1 + prob.config.material.nu))
    ref = T * e.about[0] / (G * 0.140577 * b ** 4)
    assert low < c / T / ref < 1.0


# --- the part builder, the text and the dashboard -------------------------------------------


def builder_spec(**over):
    spec = {
        "name": "pressed",
        "regions": {
            "design": [{"box": {"min": [0, 0, 0], "max": [40, 20, 20]}}],
            "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 20]}}],
                      "rail": [{"box": {"min": [0, -4, 0], "max": [40, 0, 20]}}]},
            "load": {"pad": [{"box": {"min": [30, 2, 20], "max": [40, 18, 24]}}]},
        },
        "fixed_axes": {"rail": ["y"]},
        "load_cases": [{"name": "press", "pressures": [{"region": "pad", "value_mpa": 0.05}]}],
        "settings": {"volume_fraction": 0.3, "voxel_size": 2.0},
    }
    spec.update(over)
    return spec


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    return tmp_path / "ws"


def test_the_builder_writes_a_per_axis_entry_and_accepts_a_pressure_only_load(work):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    r = server.build_regions(builder_spec())
    assert r["ok"], r
    doc = yaml.safe_load((work / r["yaml_rel"]).read_text())
    assert doc["regions"]["fixed"] == ["wall.stl", {"file": "rail.stl", "axes": ["y"]}]
    assert (work / "runs/pressed/rail.stl").is_file()          # the entry is a mapping: its file is still listed
    assert "runs/pressed/rail.stl" in r["files_rel"]
    cfg = load_config(work / r["yaml_rel"])
    assert [p.name for p in cfg.regions.fixed] == ["wall.stl", "rail.stl"]
    assert cfg.regions.fixed_axes == ((0, 1, 2), (1,))
    assert cfg.load_cases[0].pressures[0].value_mpa == 0.05


@pytest.mark.parametrize("change, words", [
    ({"fixed_axes": {"nope": ["y"]}}, "regions.fixed has no such name"),
    ({"fixed_axes": {"rail": ["q"]}}, "list of the axes x, y, z"),
    ({"fixed_axes": ["rail"]}, "must be a mapping"),
    ({"load_cases": [{"name": "p", "pressures": [{"region": "wall", "value_mpa": 0.1}]}]}, "a pressure on it would be lost"),
    ({"load_cases": [{"name": "p", "pressures": [{"region": "design", "value_mpa": 0.1}]}]}, "changes with the design"),
    ({"load_cases": [{"name": "p", "forces": [{"region": "pad", "force": [0, 0, -1]}]}], "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [40, 20, 20]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 20]}}]},
        "load": {"pad": [{"box": {"min": [30, 2, 20], "max": [40, 18, 24]}}],
                 "idle": [{"box": {"min": [0, 2, 20], "max": [6, 18, 24]}}]}}},
     "has no force or pressure"),
])
def test_the_builder_refuses_with_a_fact(work, change, words):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    r = server.build_regions(builder_spec(**change))
    assert r["ok"] is False and words in r["error"], r


def test_the_texts_name_the_new_keys_and_stay_short():
    skill = (REPO / ".claude/skills/part-to-topopt/SKILL.md").read_text()
    for word in ("moment: [Mx, My, Mz]", "at: [x, y, z]", "pressures", "value_mpa", "fixed_axes",
                 "rigid-body modes", "RBE3-like", "n/a"):
        assert word in skill, word
    pytest.importorskip("mcp")
    from topopt_mcp import server
    for fn in (server.build_regions, server.run_topopt):
        assert len(fn.__doc__) < 4096
    assert all(w in server.build_regions.__doc__ for w in ("moment", "pressures", "fixed_axes"))
    assert "extended" in server.run_topopt.__doc__
    from test_plugin_manifest import OLD_CLIENT_CUT   # the bound of the instructions since #380: the cut of the older client
    assert len(server.INSTRUCTIONS) <= OLD_CLIENT_CUT


def test_the_dashboard_handles_null_stiffness_and_a_zero_force():
    js = (REPO / "topopt/dashboard/js/app.js").read_text()
    assert "c.N_per_mm == null" in js and "po.N_per_mm != null" in js    # n/a, no toFixed on null
    assert "if (!f) return;" in js and ">= 1e-12" in js                 # no arrow for a case of moments only
    assert "f.at ? new THREE.Vector3(...f.at)" in js                    # a remote force: the arrow is at `at`


# --- topopt verify (#21) refuses the new types before any mesh ---------------------------------


def _fake_run(tmp_path, mutate):
    """A run folder that `read_run` accepts (run.json, result/final.stl); its YAML has the new type `mutate` adds."""
    yaml_path = case(sub(tmp_path, "y"), mutate=mutate, regions={"side": ((0, -2, 0), (4, 0, 10))})
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    (run / "result" / "final.stl").write_bytes(b"solid x\nendsolid x\n")
    (run / "run.json").write_text(json.dumps({"yaml_path": str(yaml_path), "h": 1.0}))
    return run


@pytest.mark.parametrize("what,mutate", [
    ("a pressure", lambda d: d["load_cases"][0].update(pressures=[{"region": "tip", "value_mpa": 0.1}])),
    ("a moment", lambda d: d["load_cases"][0]["forces"][0].update(moment=[0.0, 5.0, 0.0])),
    ("a remote force", lambda d: d["load_cases"][0]["forces"][0].update(at=[30.0, 5.0, 1.0])),
    ("a per-axis support", lambda d: d["regions"].update(fixed=["plate.stl", {"file": "side.stl", "axes": ["y"]}])),
])
def test_topopt_verify_refuses_the_new_types_with_the_options(tmp_path, capsys, what, mutate):
    from topopt.cli import main
    from topopt.verify.run import preflight
    run = _fake_run(tmp_path, mutate)
    with pytest.raises(UserError) as e:
        preflight(run)
    text = str(e.value)
    assert what in text and "does not model" in text and "Options:" in text and "plain forces and full clamps" in text
    capsys.readouterr()
    assert main(["verify", str(run)]) == 1                       # the CLI: exit code 1, one error line, no traceback
    err = capsys.readouterr().err
    assert err.startswith("error:") and "does not model" in err and "Traceback" not in err
    assert not (run / "verify").exists()                         # nothing was meshed or written
    from topopt_mcp import server                                # the MCP tool reaches the same `preflight`
    for estimate_only in (True, False):
        res = server.verify_run(run_dir=str(run), estimate_only=estimate_only)
        assert res["ok"] is False and res["exit_code"] == 1 and "does not model" in res["error"], res


def test_topopt_verify_accepts_a_plain_yaml_past_the_guard(tmp_path):
    from topopt.verify.run import _check_loads_and_supports
    _check_loads_and_supports(load_config(case(tmp_path)))       # no error: plain forces and full clamps
